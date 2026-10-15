# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused RMSNorm and weighted TOPK backward kernels.

The exact path preserves two independent reductions: the RMSNorm projection
and scale-and-sum's 1024-column, four-warp score gradient. Expert rows of the
same token share grad_y in L2 or in shared memory.
"""

import math
import operator
from functools import partial
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr, Float32, Int32

from .._environment import is_blackwell_gpu
from .._quack import copy_utils
from .._quack.reduce import row_reduce
from ._fused_rmsnorm_combine_base import _FusedRMSNormCombineBase
from ._rmsnorm_common import _fake_div
from ._rmsnorm_fwd_kernel import RMSNormStats


class FusedRMSNormCombineBwdExact(_FusedRMSNormCombineBase):
    """Bitwise-exact backward matching the two standalone kernels.

    The scale gradient follows Triton's 1024-column chunking and four-warp
    reduction. The input gradient independently follows the standalone
    RMSNorm row-reduction order.

    Clustered or token-dependent standalone reductions use a two-pass schedule:
    retain the per-warp projection partials, then reload 1024-column chunks for
    both gradients without materializing full FP32 intermediates.

    Adjacent blocks process experts of the same token to reuse grad_y in L2.
    Large training batches can additionally share grad_y within a CTA: four
    expert rows through 16K columns, or eight through 32K, reuse one FP32
    shared-memory row. Both reduction trees remain unchanged.
    """

    _MAX_REGISTER_DIM = 8192
    # Four elements preserve the score-gradient partition and keep both
    # BF16 and FP32 copies within a 16-byte vector.
    _EXACT_BWD_VECSIZE = 4

    @staticmethod
    def _exact_bwd_vecsize(N: int) -> int:
        """Return the fixed-width exact-backward vector size for `N`."""
        return math.gcd(N, FusedRMSNormCombineBwdExact._EXACT_BWD_VECSIZE)

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        grad_y_dtype: Type[cutlass.Numeric],
        N: int,
        TOPK: int,
        *,
        projection_threads_per_row: int = 0,
        projection_cluster_n: int = 0,
        use_shared_grad_y: bool = False,
        recompute_rstd: bool = False,
        use_kahan: bool = False,
        has_expert_ids: bool = False,
    ):
        super().__init__(dtype, N, TOPK, stage=1, has_expert_ids=has_expert_ids)
        assert use_shared_grad_y or not recompute_rstd
        self.recompute_rstd = recompute_rstd
        self.use_kahan = use_kahan
        self.grad_y_dtype = grad_y_dtype
        self.projection_threads_per_row = projection_threads_per_row
        self.projection_cluster_n = projection_cluster_n
        self.use_shared_grad_y = use_shared_grad_y
        self.rows_per_cta = (4 if N <= 16384 else 8) if use_shared_grad_y else 1
        self.cluster_n = 1
        self._is_blackwell = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 10
        )
        if not self._is_blackwell and N > 4096:
            self._num_threads_val = 256
        else:
            self._num_threads_val = 128

    def _threads_per_row(self):
        return self._num_threads_val

    def _num_threads(self):
        return self._num_threads_val

    def can_implement(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None = None,
    ) -> bool:
        return (
            self._supports_expert_ids(mExpertIds, mX.shape[0])
            and mX.element_type == self.dtype
            and mGradY.element_type == self.grad_y_dtype
            and mX.element_type.width <= 32
            and mGradY.element_type.width <= 32
            and mGradX.element_type.width <= 32
            and mX.shape[1] == self.N
            and mGradY.shape[1] == self.N
            and mGradX.shape[1] == self.N
            and self._is_16b_aligned(mGradY, dim=1)
            and self._is_16b_aligned(mX, dim=1)
            and self._is_16b_aligned(mGradX, dim=1)
            and mRstd.element_type == Float32
        )

    @cute.jit
    def __call__(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        if const_expr(self.use_shared_grad_y):
            self._launch_shared_grad_y(
                mGradY,
                mX,
                mW,
                mRstd,
                mGradX,
                mGradW,
                mExpertIds,
                eps,
                stream,
            )
            return
        if const_expr(self.projection_threads_per_row > 0):
            self._launch_chunked(
                mGradY, mX, mW, mRstd, mGradX, mGradW, mExpertIds, stream
            )
            return
        assert mX.element_type == self.dtype
        vecsize = self._exact_bwd_vecsize(self.N)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        B = mGradY.shape[0]
        self.kernel(
            mGradY,
            mX,
            mW,
            mRstd,
            mGradX,
            mGradW,
            mExpertIds,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            vecsize,
        ).launch(
            grid=[B * self.TOPK, 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        vecsize: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        bidx, kidx = row // self.TOPK, row % self.TOPK
        TOPK = const_expr(self.TOPK)
        dscale_block_d = const_expr(1024)
        dscale_elems_per_thread = const_expr(dscale_block_d // threads_per_row)
        dscale_chunks = const_expr(cute.ceil_div(self.N, dscale_block_d))

        tv_layout = tiled_copy.layout_tv_tiled
        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        grad_w_num_warps = const_expr(4)
        smem = cutlass.utils.SmemAllocator()

        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )
        sGradWByWarp = smem.allocate_tensor(
            Float32,
            cute.make_layout((dscale_chunks, grad_w_num_warps)),
            byte_alignment=4,
        )
        if const_expr(threads_per_row == 256):
            sGradYHigh = smem.allocate_tensor(
                Float32,
                cute.make_layout((dscale_chunks, 128, dscale_elems_per_thread)),
                byte_alignment=4,
            )
            sXNormHigh = smem.allocate_tensor(
                Float32,
                cute.make_layout((dscale_chunks, 128, dscale_elems_per_thread)),
                byte_alignment=4,
            )
        else:
            sGradYHigh = None
            sXNormHigh = None

        DIM = mX.shape[1]
        is_even_N = const_expr(DIM == tiler_mn[1])

        thr_copy = tiled_copy.get_slice(tidx)
        self._initialize_cluster(tidx, mbar_ptr, num_warps)
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        topk_tiler_mn = (TOPK, tiler_mn[1])
        gX = cute.local_tile(mX, topk_tiler_mn, (bidx, 0))
        gRstd = cute.local_tile(mRstd, (TOPK,), (bidx,))
        gW = cute.local_tile(mW, (TOPK,), (bidx,))
        gGradW = cute.local_tile(mGradW, (TOPK,), (bidx,))
        gGradX = cute.local_tile(mGradX, topk_tiler_mn, (bidx, 0))
        gX_k = cute.local_tile(gX, tiler_mn, (kidx, 0))
        gGradX_k = cute.local_tile(gGradX, tiler_mn, (kidx, 0))

        idX = cute.make_identity_tensor(mX.shape)
        cX = cute.local_tile(idX, topk_tiler_mn, (bidx, 0))
        cX_0 = cute.local_tile(cX, tiler_mn, (0, 0))
        tXpX = (
            copy_utils.predicate_k(thr_copy.partition_S(cX_0), limit=DIM)
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        tXgX_k = thr_copy.partition_S(gX_k)
        tXrX = cute.make_fragment_like(tXgX_k)
        gGradY = cute.local_tile(mGradY, tiler_mn, (bidx, 0))
        tXgGradY = thr_copy.partition_S(gGradY)
        tXrGradY = cute.make_fragment_like(tXgGradY)
        tXrWdy = tXrGradY
        if const_expr(mGradY.element_type != Float32):
            tXrWdy = cute.make_fragment_like(tXrGradY, Float32)
        if not is_even_N:
            tXrGradY.fill(0.0)
            tXrX.fill(0.0)
        copy(tXgGradY, tXrGradY)
        copy(tXgX_k, tXrX)
        valid = self._route_valid(mExpertIds, row)
        self._mask_fragment(tXrX, valid)
        self._mask_fragment(tXrGradY, valid)
        rstd_k = self._route_scalar(gRstd[kidx], valid)
        w_k = self._route_scalar(gW[kidx], valid)
        grad_y = tXrGradY.load().to(Float32)
        x = tXrX.load().to(Float32)
        x_norm = x * rstd_k

        frag_elems = const_expr(cute.size(tXrX))
        for chunk_idx in cutlass.range_constexpr(dscale_chunks):
            chunk_frag = const_expr(chunk_idx * dscale_elems_per_thread)
            grad_w_thread = Float32(0.0)

            if const_expr(threads_per_row == 256):
                assert sGradYHigh is not None
                assert sXNormHigh is not None
                for offset in cutlass.range_constexpr(dscale_elems_per_thread):
                    frag_idx = const_expr(chunk_frag + offset)
                    high_grad_y = Float32(0.0)
                    high_x_norm = Float32(0.0)
                    if const_expr(frag_idx < frag_elems):
                        high_grad_y = grad_y[frag_idx]
                        high_x_norm = x_norm[frag_idx]
                    if tidx >= 128:
                        high_tid = tidx - 128
                        sGradYHigh[chunk_idx, high_tid, offset] = high_grad_y
                        sXNormHigh[chunk_idx, high_tid, offset] = high_x_norm
                cute.arch.barrier()

            first_idx = const_expr(chunk_frag + 1)
            if const_expr(first_idx < frag_elems):
                grad_w_thread = self._mul_rn_f32(grad_y[first_idx], x_norm[first_idx])

            second_idx = const_expr(chunk_frag)
            if const_expr(second_idx < frag_elems):
                grad_w_thread = self._fma_rn_f32(
                    grad_y[second_idx], x_norm[second_idx], grad_w_thread
                )

            for offset in cutlass.range_constexpr(2, dscale_elems_per_thread):
                frag_idx = const_expr(chunk_frag + offset)
                if const_expr(frag_idx < frag_elems):
                    grad_w_thread = self._fma_rn_f32(
                        grad_y[frag_idx], x_norm[frag_idx], grad_w_thread
                    )

            if const_expr(threads_per_row == 256):
                assert sGradYHigh is not None
                assert sXNormHigh is not None
                if tidx < 128:
                    for offset in cutlass.range_constexpr(dscale_elems_per_thread):
                        grad_w_thread = self._fma_rn_f32(
                            sGradYHigh[chunk_idx, tidx, offset],
                            sXNormHigh[chunk_idx, tidx, offset],
                            grad_w_thread,
                        )

            grad_w_warp = cute.arch.warp_reduction(grad_w_thread, operator.add)
            if lane_idx == 0 and warp_idx < grad_w_num_warps:
                sGradWByWarp[chunk_idx, warp_idx] = grad_w_warp

        for i in cutlass.range_constexpr(cute.size(tXrWdy)):
            tXrWdy[i] = self._mul_rn_f32(grad_y[i], w_k)
        wdy = tXrWdy.load()

        mean_xhat_wdy_thread = Float32(0.0)
        for i in cutlass.range_constexpr(cute.size(tXrX)):
            mean_xhat_wdy_thread = self._fma_rn_f32(
                x_norm[i], wdy[i], mean_xhat_wdy_thread
            )
        mean_xhat_wdy = (
            row_reduce(
                mean_xhat_wdy_thread,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
            )
            / DIM
        )

        grad_w_k = Float32(0.0)
        for chunk_idx in cutlass.range_constexpr(dscale_chunks):
            grad_w_chunk = Float32(0.0)
            if warp_idx == 0 and lane_idx < grad_w_num_warps:
                grad_w_chunk = sGradWByWarp[chunk_idx, lane_idx]
            grad_w_chunk = cute.arch.warp_reduction(
                grad_w_chunk,
                operator.add,
                threads_in_group=grad_w_num_warps,
            )
            if tidx == 0:
                grad_w_k += grad_w_chunk
        if tidx == 0:
            gGradW[kidx] = grad_w_k

        grad_x = (wdy - x_norm * mean_xhat_wdy) * rstd_k
        tXgGradX_k = thr_copy.partition_D(gGradX_k)
        tXrGradX = cute.make_fragment_like(tXgGradX_k)
        tXrGradX.store(grad_x.to(tXrGradX.element_type))
        copy(tXrGradX, tXgGradX_k)

    @cute.jit
    def _launch_shared_grad_y(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert self._is_blackwell
        assert self.TOPK == 8 and 7168 <= self.N <= 32768
        assert mX.element_type == cutlass.BFloat16
        assert mGradY.element_type in (Float32, cutlass.BFloat16)
        # (0, 0) selects the register kernel's single-CTA projection tree.
        projection_threads = self.projection_threads_per_row or self._threads_per_row()
        cluster_n = self.projection_cluster_n or 1
        vecsize = self._exact_bwd_vecsize(self.N)
        assert projection_threads in (32, 64, 128)
        assert 0 < (projection_threads // 32) * cluster_n <= 32
        num_threads = 128 * self.rows_per_cta
        shard_n = cute.ceil_div(self.N, vecsize * projection_threads * cluster_n) * (
            vecsize * projection_threads
        )
        alignment = vecsize * num_threads
        padded_n = (
            cute.ceil_div(max(self.N, shard_n * cluster_n), alignment) * alignment
        )
        stage_copy = copy_utils.tiled_copy_2d(
            mGradY.element_type, num_threads, num_threads, vecsize
        )
        projection_copy = copy_utils.tiled_copy_2d(
            self.dtype, projection_threads, projection_threads, vecsize
        )
        chunk_copy = copy_utils.tiled_copy_2d(self.dtype, 128, 128, vecsize)
        self.shared_grad_y_kernel(
            mGradY,
            mX,
            mW,
            mRstd,
            mGradX,
            mGradW,
            mExpertIds,
            eps,
            stage_copy,
            projection_copy,
            chunk_copy,
            projection_threads,
            cluster_n,
            shard_n,
            padded_n,
            vecsize,
        ).launch(
            grid=[mGradY.shape[0] * (self.TOPK // self.rows_per_cta), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def shared_grad_y_kernel(  # noqa: C901
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        eps: Float32,
        stage_copy: cute.TiledCopy,
        projection_copy: cute.TiledCopy,
        chunk_copy: cute.TiledCopy,
        projection_threads: cutlass.Constexpr[int],
        cluster_n: cutlass.Constexpr[int],
        shard_n: cutlass.Constexpr[int],
        padded_n: cutlass.Constexpr[int],
        vecsize: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        row_group = tidx // 128
        local_tid = tidx % 128
        row = block * self.rows_per_cta + row_group
        bidx = block // (self.TOPK // self.rows_per_cta)
        lane = cute.arch.lane_idx()
        warp = local_tid // 32
        smem = cutlass.utils.SmemAllocator()
        sy_layout = cute.make_ordered_layout((1, padded_n), order=(1, 0))
        sY = smem.allocate_tensor(mGradY.element_type, sy_layout, byte_alignment=16)
        projection_warps = const_expr(projection_threads // 32)
        chunks = const_expr(cute.ceil_div(self.N, 1024))
        sProjection = smem.allocate_tensor(
            Float32,
            cute.make_layout((projection_warps * cluster_n, self.rows_per_cta)),
            byte_alignment=4,
        )
        sGradW = smem.allocate_tensor(
            Float32,
            cute.make_layout((4, chunks, self.rows_per_cta)),
            byte_alignment=4,
        )
        gY = cute.local_tile(mGradY, (1, padded_n), (bidx, 0))
        idY = cute.make_identity_tensor(mGradY.shape)
        cy = cute.local_tile(idY, (1, padded_n), (bidx, 0))
        load_thread = stage_copy.get_slice(tidx)
        load_pred = copy_utils.predicate_k(load_thread.partition_S(cy), limit=self.N)
        copy_utils.copy(
            load_thread.partition_S(gY),
            load_thread.partition_D(sY),
            pred=load_pred,
            is_async=True,
        )
        cute.arch.cp_async_commit_group()
        if const_expr(self.recompute_rstd):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.rows_per_cta, (4, 1)), stride=(4, (1, 4))),
                byte_alignment=8,
            )
            rstd = self._recompute_rstd(
                mX,
                row,
                local_tid,
                chunk_copy,
                reduction_buffer,
                eps,
                vecsize,
                mExpertIds,
            )
            if local_tid == 0:
                mRstd[row] = rstd
        else:
            rstd = mRstd[row]
        valid = self._route_valid(mExpertIds, row)
        rstd = self._route_scalar(rstd, valid)
        weight = self._route_scalar(mW[row], valid)
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        idX = cute.make_identity_tensor(mX.shape)
        if local_tid < projection_threads:
            pt = projection_copy.get_slice(local_tid)
            for shard in cutlass.range(cluster_n, unroll_full=True):
                gx = cute.local_tile(mX, (1, shard_n), (row, shard))
                sy = cute.local_tile(sY, (1, shard_n), (0, shard))
                cx = cute.local_tile(idX, (1, shard_n), (row, shard))
                rx = cute.make_fragment_like(pt.partition_S(gx))
                ry = cute.make_fragment_like(pt.partition_S(sy))
                pred = copy_utils.predicate_k(pt.partition_S(cx), limit=self.N)
                rx.fill(0.0)
                copy_utils.copy(pt.partition_S(gx), rx, pred=pred)
                cute.autovec_copy(pt.partition_S(sy), ry)
                self._mask_fragment(rx, valid)
                self._mask_fragment(ry, valid)
                dot = Float32(0.0)
                for i in cutlass.range_constexpr(cute.size(rx)):
                    xhat = self._mul_rn_f32(rx[i].to(Float32), rstd)
                    wdy = self._mul_rn_f32(ry[i], weight)
                    dot = self._fma_rn_f32(xhat, wdy, dot)
                dot = cute.arch.warp_reduction(dot, operator.add)
                if lane == 0:
                    sProjection[warp + projection_warps * shard, row_group] = dot
        cute.arch.barrier()
        projection = Float32(0.0)
        if lane < projection_warps * cluster_n:
            projection += sProjection[lane, row_group]
        projection = cute.arch.warp_reduction(projection, operator.add) / self.N
        ct = chunk_copy.get_slice(local_tid)
        for chunk in cutlass.range_constexpr(chunks):
            gx = cute.local_tile(mX, (1, 1024), (row, chunk))
            sy = cute.local_tile(sY, (1, 1024), (0, chunk))
            gdx = cute.local_tile(mGradX, (1, 1024), (row, chunk))
            cx = cute.local_tile(idX, (1, 1024), (row, chunk))
            coords = ct.partition_S(cx)
            pred = copy_utils.predicate_k(coords, limit=self.N)
            rx = cute.make_fragment_like(ct.partition_S(gx))
            ry = cute.make_fragment_like(ct.partition_S(sy))
            rdx = cute.make_fragment_like(ct.partition_D(gdx))
            rx.fill(0.0)
            copy_utils.copy(ct.partition_S(gx), rx, pred=pred)
            cute.autovec_copy(ct.partition_S(sy), ry)
            self._mask_fragment(rx, valid)
            self._mask_fragment(ry, valid)
            xhat = rx.load().to(Float32) * rstd
            dy = ry.load()
            dw = self._mul_rn_f32(dy[1], xhat[1])
            dw = self._fma_rn_f32(dy[0], xhat[0], dw)
            for i in cutlass.range_constexpr(2, cute.size(rx)):
                dw = self._fma_rn_f32(dy[i], xhat[i], dw)
            dw = cute.arch.warp_reduction(dw, operator.add)
            if lane == 0:
                sGradW[warp, chunk, row_group] = dw
            rwdy = cute.make_fragment_like(ry, Float32)
            for i in cutlass.range_constexpr(cute.size(ry)):
                rwdy[i] = self._mul_rn_f32(dy[i], weight)
            dx = (rwdy.load() - xhat * projection) * rstd
            rdx.store(dx.to(rdx.element_type))
            copy_utils.copy(rdx, ct.partition_D(gdx), pred=pred)
        cute.arch.barrier()
        if local_tid < 32:
            dw = Float32(0.0)
            for chunk in cutlass.range_constexpr(chunks):
                partial = Float32(0.0)
                if lane < 4:
                    partial = sGradW[lane, chunk, row_group]
                partial = cute.arch.warp_reduction(
                    partial, operator.add, threads_in_group=4
                )
                dw += partial
            if local_tid == 0:
                mGradW[row] = dw

    @cute.jit
    def _recompute_rstd(
        self,
        mX: cute.Tensor,
        row: Int32,
        local_tid: Int32,
        chunk_copy: cute.TiledCopy,
        reduction_buffer: cute.Tensor,
        eps: Float32,
        vecsize: cutlass.Constexpr[int],
        mExpertIds: cute.Tensor | None,
    ) -> Float32:
        """Reproduce `RMSNormStats` bit for bit: 128 threads per row, one
        vector-major accumulation chain per thread over the columns
        `RMSNormStats` tiles, then the same row reduction."""
        assert vecsize == RMSNormStats._VECTOR_SIZE
        ct = chunk_copy.get_slice(local_tid)
        valid = self._route_valid(mExpertIds, row)
        idX = cute.make_identity_tensor(mX.shape)
        stats_cols = const_expr(cute.ceil_div(self.N, 128 * vecsize) * 128 * vecsize)
        total = Float32(0.0)
        compensation = Float32(0.0)
        for chunk in cutlass.range_constexpr(cute.ceil_div(self.N, 1024)):
            gx = cute.local_tile(mX, (1, 1024), (row, chunk))
            cx = cute.local_tile(idX, (1, 1024), (row, chunk))
            rx = cute.make_fragment_like(ct.partition_S(gx))
            rx.fill(0.0)
            copy_utils.copy(
                ct.partition_S(gx),
                rx,
                pred=copy_utils.predicate_k(ct.partition_S(cx), limit=self.N),
            )
            self._mask_fragment(rx, valid)
            for i in cutlass.range_constexpr(cute.size(rx)):
                if const_expr(chunk * 1024 + i // vecsize * 128 * vecsize < stats_cols):
                    value = rx[i].to(Float32)
                    if const_expr(self.use_kahan):
                        square = value * value
                        corrected = square + compensation
                        updated = total + corrected
                        compensation = corrected - (updated - total)
                        total = updated
                    else:
                        total = self._fma_rn_f32(value, value, total)
        sum_sq = row_reduce(
            total, cute.ReductionOp.ADD, 128, reduction_buffer, None, init_val=0.0
        )
        return self._route_scalar(
            cute.math.rsqrt(sum_sq / self.N + eps, fastmath=False), valid
        )

    @cute.jit
    def _launch_chunked(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        assert self._is_blackwell
        assert mX.element_type == cutlass.BFloat16
        assert mGradY.element_type == Float32
        projection_threads = self.projection_threads_per_row
        cluster_n = self.projection_cluster_n
        threads_per_row = self._threads_per_row()
        vecsize = self._exact_bwd_vecsize(self.N)
        assert projection_threads in (32, 64, 128)
        assert 0 < cluster_n * (projection_threads // 32) <= 32
        shard_n = cute.ceil_div(self.N, vecsize * projection_threads * cluster_n) * (
            vecsize * projection_threads
        )
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, threads_per_row, vecsize
        )
        projection_copy = copy_utils.tiled_copy_2d(
            self.dtype, projection_threads, projection_threads, vecsize
        )
        self.chunked_kernel(
            mGradY,
            mX,
            mW,
            mRstd,
            mGradX,
            mGradW,
            mExpertIds,
            tiled_copy,
            projection_copy,
            shard_n,
            cluster_n,
            threads_per_row,
            vecsize,
        ).launch(
            grid=[mGradY.shape[0] * self.TOPK, 1, 1],
            block=[threads_per_row, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def chunked_kernel(  # noqa: C901
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        tiled_copy: cute.TiledCopy,
        projection_copy: cute.TiledCopy,
        shard_n: cutlass.Constexpr[int],
        cluster_n: cutlass.Constexpr[int],
        threads_per_row: cutlass.Constexpr[int],
        vecsize: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        bidx = row // self.TOPK
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        thr_copy = tiled_copy.get_slice(tidx)
        projection_threads = const_expr(self.projection_threads_per_row)
        projection_warps = const_expr(projection_threads // 32)
        smem = cutlass.utils.SmemAllocator()
        sProjection = smem.allocate_tensor(
            Float32, cute.make_layout((projection_warps, cluster_n)), byte_alignment=4
        )
        chunks = const_expr(cute.ceil_div(self.N, 1024))
        grad_w_num_warps = const_expr(threads_per_row // 32)
        sGradW = smem.allocate_tensor(
            Float32, cute.make_layout((grad_w_num_warps, chunks)), byte_alignment=4
        )
        valid = self._route_valid(mExpertIds, row)
        rstd = self._route_scalar(mRstd[row], valid)
        weight = self._route_scalar(mW[row], valid)
        idX = cute.make_identity_tensor(mX.shape)
        if tidx < projection_threads:
            projection_thr_copy = projection_copy.get_slice(tidx)
            for shard in cutlass.range(cluster_n):
                gX = cute.local_tile(mX, (1, shard_n), (row, shard))
                gY = cute.local_tile(mGradY, (1, shard_n), (bidx, shard))
                cX = cute.local_tile(idX, (1, shard_n), (row, shard))
                rX = cute.make_fragment_like(projection_thr_copy.partition_S(gX))
                rY = cute.make_fragment_like(projection_thr_copy.partition_S(gY))
                pred = copy_utils.predicate_k(
                    projection_thr_copy.partition_S(cX), limit=self.N
                )
                rX.fill(0.0)
                rY.fill(0.0)
                copy_utils.copy(projection_thr_copy.partition_S(gX), rX, pred=pred)
                copy_utils.copy(projection_thr_copy.partition_S(gY), rY, pred=pred)
                self._mask_fragment(rX, valid)
                self._mask_fragment(rY, valid)
                dot = Float32(0.0)
                for i in cutlass.range_constexpr(cute.size(rX)):
                    x_hat = self._mul_rn_f32(rX[i].to(Float32), rstd)
                    wdy = self._mul_rn_f32(rY[i], weight)
                    dot = self._fma_rn_f32(x_hat, wdy, dot)
                dot = cute.arch.warp_reduction(dot, operator.add)
                if lane == 0:
                    sProjection[warp, shard] = dot
        cute.arch.barrier()
        projection = Float32(0.0)
        if lane < projection_warps * cluster_n:
            projection += sProjection[lane]
        projection = cute.arch.warp_reduction(projection, operator.add) / self.N

        for chunk in cutlass.range(chunks):
            gX = cute.local_tile(mX, (1, 1024), (row, chunk))
            gY = cute.local_tile(mGradY, (1, 1024), (bidx, chunk))
            gDX = cute.local_tile(mGradX, (1, 1024), (row, chunk))
            cX = cute.local_tile(idX, (1, 1024), (row, chunk))
            coords = thr_copy.partition_S(cX)
            pred = copy_utils.predicate_k(coords, limit=self.N)
            rX = cute.make_fragment_like(thr_copy.partition_S(gX))
            rY = cute.make_fragment_like(thr_copy.partition_S(gY))
            rDX = cute.make_fragment_like(thr_copy.partition_D(gDX))
            rX.fill(0.0)
            rY.fill(0.0)
            copy_utils.copy(thr_copy.partition_S(gX), rX, pred=pred)
            copy_utils.copy(thr_copy.partition_S(gY), rY, pred=pred)
            self._mask_fragment(rX, valid)
            self._mask_fragment(rY, valid)
            x_hat = rX.load().to(Float32) * rstd
            dy = rY.load()
            dw = self._mul_rn_f32(dy[1], x_hat[1])
            dw = self._fma_rn_f32(dy[0], x_hat[0], dw)
            for i in cutlass.range_constexpr(2, cute.size(rX)):
                dw = self._fma_rn_f32(dy[i], x_hat[i], dw)
            dw = cute.arch.warp_reduction(dw, operator.add)
            if lane == 0:
                sGradW[warp, chunk] = dw
            rWdy = cute.make_fragment_like(rY)
            for i in cutlass.range_constexpr(cute.size(rY)):
                rWdy[i] = self._mul_rn_f32(dy[i], weight)
            dx = (rWdy.load() - x_hat * projection) * rstd
            rDX.store(dx.to(rDX.element_type))
            copy_utils.copy(rDX, thr_copy.partition_D(gDX), pred=pred)
        cute.arch.barrier()
        dw = Float32(0.0)
        for chunk in cutlass.range(chunks):
            partial = Float32(0.0)
            if lane < grad_w_num_warps:
                partial = sGradW[lane, chunk]
            partial = cute.arch.warp_reduction(
                partial, operator.add, threads_in_group=grad_w_num_warps
            )
            dw += partial
        if tidx == 0:
            mGradW[row] = dw


class FusedRMSNormCombineBwd(_FusedRMSNormCombineBase):
    """Backward kernel for fused RMSNorm + weighted TOPK reduction.

    Given grad_y [B, DIM], computes:
      grad_x [B*TOPK, DIM] — gradient w.r.t. input
      grad_w [B*TOPK]      — gradient w.r.t. weights

    Math:
      x_norm[k] = x[k] * rstd[k]
      grad_w[k] = sum_d(grad_y[d] * x_norm[k,d])
      mean[k] = grad_w[k] * w[k] / DIM
      grad_x[k,d] = (grad_y[d] * w[k] - x_norm[k,d] * mean[k]) * rstd[k]
    """

    # Keep the staged FP32 grad_y row within 48 KiB, plus kernel scratch.
    _MAX_DIM = 12288

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        grad_y_dtype: Type[cutlass.Numeric],
        N: int,
        TOPK: int,
        has_expert_ids: bool = False,
    ):
        super().__init__(dtype, N, TOPK, stage=1, has_expert_ids=has_expert_ids)
        self.grad_y_dtype = grad_y_dtype
        self.cluster_n = 1
        self._num_threads_val = 256 if is_blackwell_gpu() else 128

    def _threads_per_row(self):
        return self._num_threads_val

    def _num_threads(self):
        return self._num_threads_val

    def can_implement(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None = None,
    ) -> bool:
        return (
            self._supports_expert_ids(mExpertIds, mX.shape[0])
            and mX.element_type == self.dtype
            and mGradY.element_type == self.grad_y_dtype
            and mX.shape[1] == self.N
            and mGradY.shape[1] == self.N
            and mGradX.shape[1] == self.N
            and self._is_16b_aligned(mGradY, dim=1)
            and self._is_16b_aligned(mX, dim=1)
            and self._is_16b_aligned(mGradX, dim=1)
            and mRstd.element_type == Float32
        )

    @cute.jit
    def __call__(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        vecsize = const_expr(
            _fake_div(
                self.N,
                (mX.element_type, mGradY.element_type, mGradX.element_type),
            )
        )
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        B = mGradY.shape[0]
        self.kernel(
            mGradY,
            mX,
            mW,
            mRstd,
            mGradX,
            mGradW,
            mExpertIds,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            vecsize,
        ).launch(
            grid=[B, 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        vecsize: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        TOPK = const_expr(self.TOPK)
        grad_w_num_warps = const_expr(cute.size(tiled_copy) // cute.arch.WARP_SIZE)
        smem = cutlass.utils.SmemAllocator()

        sGradY = smem.allocate_tensor(
            mGradY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        sGradWByWarp = smem.allocate_tensor(
            Float32,
            cute.make_layout((grad_w_num_warps,)),
            byte_alignment=4,
        )
        sGradW = smem.allocate_tensor(
            Float32,
            cute.make_layout((1,)),
            byte_alignment=4,
        )

        DIM = mX.shape[1]
        is_even_N = const_expr(DIM == tiler_mn[1])

        thr_copy = tiled_copy.get_slice(tidx)
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        # Carve this CTA's TOPK expert rows once, then slice by local k block.
        topk_tiler_mn = (TOPK, tiler_mn[1])
        gX = cute.local_tile(mX, topk_tiler_mn, (bidx, 0))
        gRstd = cute.local_tile(mRstd, (TOPK,), (bidx,))
        gW = cute.local_tile(mW, (TOPK,), (bidx,))
        gGradW = cute.local_tile(mGradW, (TOPK,), (bidx,))
        gGradX = cute.local_tile(mGradX, topk_tiler_mn, (bidx, 0))

        # Predication (shared for all rows — same DIM)
        idX = cute.make_identity_tensor(mX.shape)
        cX = cute.local_tile(idX, topk_tiler_mn, (bidx, 0))
        cX_0 = cute.local_tile(cX, tiler_mn, (0, 0))
        tXpX = (
            copy_utils.predicate_k(thr_copy.partition_S(cX_0), limit=DIM)
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        gX_0 = cute.local_tile(gX, tiler_mn, (0, 0))
        tXrX = cute.make_fragment_like(thr_copy.partition_S(gX_0))
        gGradY = cute.local_tile(mGradY, tiler_mn, (bidx, 0))
        tXgGradY = thr_copy.partition_S(gGradY)
        tXsGradY = thr_copy.partition_D(sGradY)
        # Predicated `CopyG2SOp` zero-fills out-of-bounds vector atoms.
        copy(tXgGradY, tXsGradY, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        tXrGradY = cute.make_fragment_like(tXsGradY)
        tXrWdy = tXrGradY
        if const_expr(mGradY.element_type != Float32):
            tXrWdy = cute.make_fragment_like(tXrGradY, Float32)

        for kidx in cutlass.range(TOPK, unroll=1):
            gX_k = cute.local_tile(gX, tiler_mn, (kidx, 0))
            gGradX_k = cute.local_tile(gGradX, tiler_mn, (kidx, 0))
            tXgX_k = thr_copy.partition_S(gX_k)
            if not is_even_N:
                tXrX.fill(0.0)
            copy(tXgX_k, tXrX)
            valid = self._route_valid(mExpertIds, bidx * TOPK + kidx)
            self._mask_fragment(tXrX, valid)
            rstd_k = self._route_scalar(gRstd[kidx], valid)
            w_k = self._route_scalar(gW[kidx], valid)
            x_norm = tXrX.load().to(Float32) * rstd_k
            cute.autovec_copy(tXsGradY, tXrGradY)
            self._mask_fragment(tXrGradY, valid)
            grad_y = tXrGradY.load().to(Float32)

            grad_w_thread = (grad_y * x_norm).reduce(
                cute.ReductionOp.ADD,
                init_val=Float32(0.0),
                reduction_profile=0,
            )
            grad_w_warp = cute.arch.warp_reduction(grad_w_thread, operator.add)
            if lane_idx == 0:
                sGradWByWarp[warp_idx] = grad_w_warp

            cute.arch.barrier()

            grad_w_k = Float32(0.0)
            if warp_idx == 0 and lane_idx < grad_w_num_warps:
                grad_w_k = sGradWByWarp[lane_idx]
            grad_w_k = cute.arch.warp_reduction(
                grad_w_k,
                operator.add,
                threads_in_group=grad_w_num_warps,
            )
            if tidx == 0:
                gGradW[kidx] = grad_w_k
                sGradW[0] = grad_w_k
            cute.arch.barrier()

            for i in cutlass.range(cute.size(tXrWdy), unroll_full=True):
                tXrWdy[i] = self._mul_rn_f32(grad_y[i], w_k)
            wdy = tXrWdy.load()
            mean_xhat_wdy = self._mul_rn_f32(sGradW[0], w_k) / DIM
            tXgGradX_k = thr_copy.partition_D(gGradX_k)
            grad_x = (wdy - x_norm * mean_xhat_wdy) * rstd_k
            tXrX.store(grad_x.to(tXrX.element_type))
            copy(tXrX, tXgGradX_k)
