# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused RMSNorm and weighted TOPK forward kernels.

Normalize [B * TOPK, DIM] expert rows and combine them into [B, DIM].
The exact path preserves the standalone RMSNorm and scale-and-sum reductions.
"""

import math
import operator
from functools import partial
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr, Float32, Int32, Int64, Uint32
from cutlass.base_dsl.runtime.cuda import get_device_attribute
from cutlass.cute.nvgpu import cpasync

from .._environment import num_sms_per_device
from .._quack import copy_utils
from .._quack.reduce import row_reduce, warp_reduce
from .._quack.rmsnorm_config import RmsNormFwdConfig
from .._quant_conversion import _cvt_bf16x2_to_f32x2
from ._fused_rmsnorm_combine_base import _FusedRMSNormCombineBase
from ._rmsnorm_common import _make_row_tma_atom
from ._rmsnorm_fwd_kernel import FusedRMSNormFwd


class FusedRMSNormCombineFwd(_FusedRMSNormCombineBase):
    """Forward kernel for fused RMSNorm + weighted TOPK reduction.

    Exact mode matches standalone RMSNorm's reduction layout. Clustered
    launches partition DIM across ``cluster_n`` blocks per batch element.
    Each batch element processes TOPK rows, computing per-row RMS
    normalization and weighted TOPK reduction into [DIM].

    Args:
        dtype: Element type of the input tensor (e.g. BFloat16).
        norm_dtype: Element type used for the normalized input.
        N: Feature dimension (DIM). Compile-time constant.
        TOPK: Number of expert rows per batch element. Compile-time constant.
        use_kahan: Use Kahan-compensated sum-of-squares instead of naive FP32.
        require_bitwise: Preserve the standalone reduction order.
        tile_d: Match the tiled standalone scale-and-sum accumulation order.
    """

    # Wide rows use at most eight standalone shards of at most 16K columns.
    _MAX_DIM = 128 * 1024
    _DIRECT_GMEM_LOAD_MAX_DIM = 2304
    _FOUR_STAGE_ROW_ELEMENTS = 64
    _MAX_PIPELINED_ROW_ELEMENTS = 80
    _REGISTER_ALLOCATION_GRANULARITY = 8
    _REGISTER_OVERHEAD_ESTIMATE = 32
    _REDUCTION_SCRATCH_BYTES_ESTIMATE = 1024
    # Larger clusters require opting into non-portable CUDA cluster sizes.
    _MAX_PORTABLE_CLUSTER_SIZE = 8

    @staticmethod
    def _fwd_threads_per_row(N: int) -> int:
        for limit, threads in [(3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        norm_dtype: Type[cutlass.Numeric],
        N: int,
        TOPK: int,
        use_kahan: bool = False,
        require_bitwise: bool = True,
        tile_d: bool = False,
        copy_input: bool = False,
        has_weight: bool = False,
        has_expert_ids: bool = False,
    ):
        super().__init__(dtype, N, TOPK, stage=2, has_expert_ids=has_expert_ids)
        # The input-scale gamma participates in the sum-of-squares, so it must
        # be applied before both reduction variants; only the plain FP32 path
        # carries it (its consumers are inference-only).
        assert not (has_weight and use_kahan), (
            "input-scale weight is not supported with Kahan summation"
        )
        assert not (has_weight and copy_input), (
            "input-scale weight is forward-only; the training input copy "
            "never coexists with it"
        )
        self.norm_dtype = norm_dtype
        self.use_kahan = use_kahan
        self.require_bitwise = require_bitwise
        self.tile_d = tile_d
        self.copy_input = copy_input
        self.has_weight = has_weight
        self.cluster_n = 1
        arch_major = FusedRMSNormFwd._current_arch_major() or 0
        self._is_blackwell = 10 <= arch_major < 12
        self._supports_clusters = arch_major != 0 and bool(
            get_device_attribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH,
                torch.cuda.current_device(),
            )
        )
        self._standalone_config = RmsNormFwdConfig.from_analytical_heuristic(
            N, dtype.width, arch_major=arch_major
        )

    def _threads_per_row(self):
        if self.require_bitwise:
            # This kernel processes one row per CTA and needs a complete warp.
            return max(32, self._standalone_config.threads_per_row)
        return self._fwd_threads_per_row(self.N)

    def _num_threads(self):
        return self._threads_per_row()

    def _load_stages(self, shard_n):
        threads_per_row = self._threads_per_row()
        elements_per_thread = cute.ceil_div(shard_n, threads_per_row)
        if (
            shard_n <= self._DIRECT_GMEM_LOAD_MAX_DIM
            or threads_per_row < 64
            or elements_per_thread > self._MAX_PIPELINED_ROW_ELEMENTS
        ):
            return 1
        # Larger register fragments leave less shared-memory headroom for rows.
        return 4 if elements_per_thread <= self._FOUR_STAGE_ROW_ELEMENTS else 3

    def _min_blocks_per_mp(self):
        shard_n = cute.ceil_div(self.N, self.cluster_n)
        if shard_n <= self._DIRECT_GMEM_LOAD_MAX_DIM or not torch.cuda.is_available():
            return 1
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        num_threads = self._num_threads()
        elements_per_thread = cute.ceil_div(shard_n, self._threads_per_row())
        # Budget two registers per row element plus kernel state: at D=9216
        # (72 elements per thread) a 128-register cap spills, while ~170
        # registers do not. The input-scale gamma keeps one 16-bit fragment
        # resident per thread.
        weight_registers = elements_per_thread // 2 if self.has_weight else 0
        target_registers = (
            cute.ceil_div(
                2 * elements_per_thread
                + weight_registers
                + self._REGISTER_OVERHEAD_ESTIMATE,
                self._REGISTER_ALLOCATION_GRANULARITY,
            )
            * self._REGISTER_ALLOCATION_GRANULARITY
        )
        # PyTorch exposes this native field but omits it from its type stubs.
        register_limited_blocks = (
            properties.regs_per_multiprocessor  # ty: ignore[unresolved-attribute]
            // (num_threads * target_registers)
        )
        load_stages = self._load_stages(shard_n)
        shared_bytes_per_block = (
            load_stages * shard_n * self.dtype.width // 8
            + self._REDUCTION_SCRATCH_BYTES_ESTIMATE
        )
        shared_limited_blocks = (
            properties.shared_memory_per_multiprocessor // shared_bytes_per_block
        )
        return max(1, min(register_limited_blocks, shared_limited_blocks))

    def _set_cluster_n(self):
        if not self._supports_clusters:
            self.cluster_n = 1
            return
        if self.require_bitwise:
            cluster_n = self._standalone_config.cluster_n
            # Clamping this path would change the reference reduction tree.
            if cluster_n > self._MAX_PORTABLE_CLUSTER_SIZE:
                raise ValueError(
                    f"bitwise RMSNorm/combine requires {cluster_n} blocks per cluster; "
                    f"the portable limit is {self._MAX_PORTABLE_CLUSTER_SIZE}"
                )
        else:
            elements_per_thread = cute.ceil_div(self.N, self._threads_per_row())
            cluster_n = 1
            # The deeper 32-element clustering is measured for the wide D=12288
            # training shape; D=4096 inference measured ~240 us/iteration slower
            # with it, so narrower rows keep the original 64-element threshold.
            cluster_elements = 32 if self.N >= 12288 else 64
            while cute.ceil_div(elements_per_thread, cluster_n) > cluster_elements:
                cluster_n *= 2
        self.cluster_n = min(cluster_n, self._MAX_PORTABLE_CLUSTER_SIZE)

    def can_implement(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mXCopy: cute.Tensor,
        mG: cute.Tensor | None = None,
        mExpertIds: cute.Tensor | None = None,
    ) -> bool:
        if (mG is None) == self.has_weight:
            return False
        if mG is not None and not (
            mG.element_type == self.dtype
            and mG.shape[1] == self.N
            and self._is_16b_aligned(mG, dim=1)
        ):
            return False
        return (
            self._supports_expert_ids(mExpertIds, mX.shape[0])
            and mX.element_type == self.dtype
            and mXCopy.element_type == self.dtype
            and mX.shape[1] == self.N
            and mXCopy.shape == mX.shape
            and mY.shape[1] == self.N
            and self._is_16b_aligned(mX, dim=1)
            and self._is_16b_aligned(mXCopy, dim=1)
            and self._is_16b_aligned(mY, dim=1)
            and self._vector_size(mY.element_type.width) % 2 == 0
            and mRstd.element_type == Float32
        )

    def _vector_size(self, output_dtype_width: int) -> int:
        largest_dtype_width = max(
            self.dtype.width,
            self.norm_dtype.width,
            output_dtype_width,
        )
        return math.gcd(self.N, 128 // largest_dtype_width)

    @staticmethod
    def _x_copy_target(
        mXCopy: cute.Tensor,
        mXCopyOffset: cute.Tensor | None,
        mXCopyCondition: cute.Tensor | None,
    ):
        """Return the input-copy destination and whether to store it.

        With an offset, ``mXCopy`` aliases byte zero of an activation buffer: it
        is rebased by the device byte offset, and the store is skipped when the
        device condition is non-zero (the row is recomputed instead)."""
        if mXCopyOffset is None:
            return mXCopy, cutlass.Boolean(True)
        ptr = cute.make_ptr(
            mXCopy.element_type,
            mXCopy.iterator.toint() + Int64(mXCopyOffset[0]),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        save = Int32(mXCopyCondition[0]) == Int32(0)
        return cute.make_tensor(ptr, mXCopy.layout), save

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        mXCopy: cute.Tensor,
        mXCopyOffset: cute.Tensor | None,
        mXCopyCondition: cute.Tensor | None,
        mG: cute.Tensor | None,
        eps: Float32,
        gain_center: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert (mG is None) != self.has_weight
        self._set_cluster_n()
        vecsize = self._vector_size(mY.element_type.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        load_stages = self._load_stages(tiler_mn[1])
        num_threads = tiled_copy.size
        B = mY.shape[0]
        self.kernel(
            mX,
            mW,
            mY,
            mRstd,
            mExpertIds,
            mXCopy,
            mXCopyOffset,
            mXCopyCondition,
            mG,
            eps,
            gain_center,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            load_stages,
        ).launch(
            grid=[B, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
            min_blocks_per_mp=self._min_blocks_per_mp(),
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        mXCopy: cute.Tensor,
        mXCopyOffset: cute.Tensor | None,
        mXCopyCondition: cute.Tensor | None,
        mG: cute.Tensor | None,
        eps: Float32,
        gain_center: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        load_stages: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        TOPK = const_expr(self.TOPK)
        tv_layout = tiled_copy.layout_tv_tiled
        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout((*tiler_mn, load_stages), order=(1, 0, 2)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )
        use_pairwise_init = const_expr(
            mY.element_type.width == 16 and self.norm_dtype.width == 32 and TOPK > 1
        )
        sW = smem.allocate_tensor(Float32, cute.make_layout(TOPK), byte_alignment=4)

        DIM = mX.shape[1]
        is_even_N = const_expr(DIM == tiler_mn[1] * self.cluster_n)
        thr_copy = tiled_copy.get_slice(tidx)
        self._initialize_cluster(tidx, mbar_ptr, num_warps)
        tXsX = thr_copy.partition_D(sX)

        topk_tiler_mn = (TOPK, tiler_mn[1])
        gX = cute.local_tile(mX, topk_tiler_mn, (bidx, cluster_y))
        mXCopy, save_copy = self._x_copy_target(mXCopy, mXCopyOffset, mXCopyCondition)
        gXCopy = cute.local_tile(mXCopy, topk_tiler_mn, (bidx, cluster_y))
        gW = cute.local_tile(mW, (TOPK,), (bidx,))
        gRstd = cute.local_tile(mRstd, (TOPK,), (bidx,))
        idX = cute.make_identity_tensor(mX.shape)
        cX = cute.local_tile(idX, topk_tiler_mn, (bidx, cluster_y))
        gX_0 = cute.local_tile(gX, tiler_mn, (0, 0))
        cX_0 = cute.local_tile(cX, tiler_mn, (0, 0))
        tXcX_full = thr_copy.partition_S(cX_0)
        tXcX_0 = tXcX_full[(0, None), None, None]
        tXpX = copy_utils.predicate_k(tXcX_full, limit=DIM) if not is_even_N else None
        copy = partial(copy_utils.copy, pred=tXpX)
        use_direct_gmem_load = const_expr(tiler_mn[1] <= self._DIRECT_GMEM_LOAD_MAX_DIM)

        tXgX_0 = thr_copy.partition_S(gX_0)
        tXrX = cute.make_fragment_like(tXgX_0)
        tXrAcc = cute.make_fragment_like(tXgX_0, Float32)
        assert cute.size(tXrAcc) % 2 == 0
        tXrAcc.fill(0.0)

        if const_expr(self.has_weight):
            # Input-scale gamma: one [DIM]-shard fragment, loaded once and
            # shared by every TOPK row. The gain center is folded here with
            # torch's `weight + gain_center` semantics (fp32 add, one RN round
            # back to the input dtype), so the per-row multiplies below see the
            # same scaled weight the standalone composition materializes.
            gG = cute.local_tile(mG, tiler_mn, (0, cluster_y))
            tXgG = thr_copy.partition_S(gG)
            tXrG = cute.make_fragment_like(tXgG)
            if not is_even_N:
                tXrG.fill(0.0)
            copy(tXgG, tXrG)
            tXrG.store((tXrG.load().to(Float32) + gain_center).to(mX.element_type))

        if tidx < TOPK:
            valid = self._route_valid(mExpertIds, bidx * TOPK + tidx)
            sW[tidx] = self._route_scalar(gW[tidx], valid)
        cute.arch.barrier()

        for reduction_i in cutlass.range_constexpr(TOPK):
            # scale_and_sum initializes with expert 1, then accumulates expert 0.
            k = const_expr(
                1 - reduction_i if TOPK > 1 and reduction_i < 2 else reduction_i
            )
            gX_k = cute.local_tile(gX, tiler_mn, (k, 0))
            tXgX_k = thr_copy.partition_S(gX_k)
            tXsX_cur = self._load_input_row(
                gX,
                tXgX_k,
                tXsX,
                tXrX,
                tXpX,
                thr_copy,
                tiler_mn,
                reduction_i,
                load_stages,
            )

            if const_expr(self.copy_input):
                if save_copy:
                    gXCopy_k = cute.local_tile(gXCopy, tiler_mn, (k, 0))
                    tXgXCopy_k = thr_copy.partition_D(gXCopy_k)
                    copy(tXrX, tXgXCopy_k)

            valid = self._route_valid(mExpertIds, bidx * TOPK + k)
            self._mask_fragment(tXrX, valid)
            reduction_stage = const_expr(reduction_i % self.stage)
            local_sum_sq = self._sum_squares(
                tXrX, tXrG if const_expr(self.has_weight) else None
            )
            local_sum_sq = self._route_scalar(local_sum_sq, valid)
            sum_sq = row_reduce(
                local_sum_sq,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, reduction_stage],
                mbar_ptr + reduction_stage
                if const_expr(self.cluster_n > 1)
                else mbar_ptr,
                phase=const_expr((reduction_i // self.stage) % 2),
                init_val=Float32(0.0),
                hook_fn=cute.arch.cluster_wait
                if const_expr(self.cluster_n > 1)
                else None,
            )

            rstd = self._route_scalar(
                cute.math.rsqrt(sum_sq / DIM + eps, fastmath=False), valid
            )
            if tidx == 0 and cluster_y == 0:
                gRstd[k] = rstd

            if const_expr(not use_direct_gmem_load):
                cute.autovec_copy(tXsX_cur, tXrX)
            x = self._input_values(tXrX, tXrG if const_expr(self.has_weight) else None)
            x_norm = (x * rstd).to(self.norm_dtype).to(Float32)
            if const_expr(self.has_expert_ids):
                masked_norm = cute.make_fragment_like(tXrX, Float32)
                masked_norm.store(x_norm)
                self._mask_fragment(masked_norm, valid)
                x_norm = masked_norm.load()
            self._accumulate_topk(
                x_norm, tXrAcc, sW, tXcX_full, k, reduction_i, use_pairwise_init
            )

        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        tXgY = thr_copy.partition_D(gY)
        row = tXcX_0[0][0]
        if row < mX.shape[0]:
            copy_utils.cvt_copy(tiled_copy, tXrAcc, tXgY, pred=tXpX, retile=True)

    @cute.jit
    def _load_input_row(
        self,
        gX: cute.Tensor,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        tXrX: cute.Tensor,
        tXpX: cute.Tensor | None,
        thr_copy: cute.ThrCopy,
        tiler_mn: cute.Shape,
        reduction_i: cutlass.Constexpr[int],
        load_stages: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        """Load one expert row, maintaining the shared-memory prefetch ring."""
        copy = partial(copy_utils.copy, pred=tXpX)
        tXsX_cur = tXsX[None, None, None, reduction_i % load_stages]
        if const_expr(tiler_mn[1] <= self._DIRECT_GMEM_LOAD_MAX_DIM):
            if const_expr(tXpX is not None):
                tXrX.fill(0.0)
            copy(tXgX, tXrX)
        else:
            if const_expr(load_stages > 1):
                if const_expr(reduction_i == 0):
                    for stage in cutlass.range_constexpr(
                        min(load_stages - 1, self.TOPK)
                    ):
                        k = const_expr(
                            1 - stage if self.TOPK > 1 and stage < 2 else stage
                        )
                        gX_k = cute.local_tile(gX, tiler_mn, (k, 0))
                        copy(
                            thr_copy.partition_S(gX_k),
                            tXsX[None, None, None, stage],
                            is_async=True,
                        )
                        cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(
                    min(load_stages - 2, self.TOPK - reduction_i - 1)
                )
                next_i = const_expr(reduction_i + load_stages - 1)
                if const_expr(next_i < self.TOPK):
                    gX_next = cute.local_tile(gX, tiler_mn, (next_i, 0))
                    copy(
                        thr_copy.partition_S(gX_next),
                        tXsX[None, None, None, next_i % load_stages],
                        is_async=True,
                    )
                    cute.arch.cp_async_commit_group()
            else:
                copy(tXgX, tXsX_cur, is_async=True)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX_cur, tXrX)
        return tXsX_cur

    @cute.jit
    def _input_values(self, rX: cute.Tensor, rG: cute.Tensor | None) -> cute.TensorSSA:
        x = rX.load().to(Float32)
        if const_expr(self.has_weight):
            assert rG is not None
            # The standalone input-scale multiply rounds back to the input dtype.
            x = (x * rG.load().to(Float32)).to(self.dtype).to(Float32)
        return x

    @cute.jit
    def _sum_squares(self, rX: cute.Tensor, rG: cute.Tensor | None):
        if const_expr(self.use_kahan):
            return FusedRMSNormFwd._sum_squares_kahan(rX)
        x = self._input_values(rX, rG)
        # Preserve the standalone pre-Blackwell reduction order.
        return self._fma_dot_f32(x, x) if const_expr(self._is_blackwell) else x * x

    @cute.jit
    def _accumulate_topk(
        self,
        x_norm: cute.TensorSSA,
        tXrAcc: cute.Tensor,
        sW: cute.Tensor,
        tXcX_full: cute.Tensor,
        k: cutlass.Constexpr[int],
        reduction_i: cutlass.Constexpr[int],
        use_pairwise_init: cutlass.Constexpr[bool],
    ):
        """Preserve scale_and_sum's packed rounding and first-two-expert order."""
        if const_expr(use_pairwise_init and reduction_i == 0):
            tXrAcc.store(x_norm)
        elif const_expr(use_pairwise_init and reduction_i == 1):
            w0 = sW[0]
            w1 = sW[1]
            if const_expr(self._is_blackwell):
                # The even vector size makes fragment index parity match column
                # parity and guarantees every element participates in a pair.
                for i in cutlass.range_constexpr(0, cute.size(tXrAcc), 2):
                    x0 = (x_norm[i], x_norm[i + 1])
                    x1 = (tXrAcc[i], tXrAcc[i + 1])
                    x1w1 = cute.arch.mul_packed_f32x2(x1, (w1, w1), rnd="rn")
                    if const_expr(self.tile_d or self.N <= 1024):
                        acc = cute.arch.fma_packed_f32x2(x0, (w0, w0), x1w1, rnd="rn")
                    else:
                        x0w0 = cute.arch.mul_packed_f32x2(x0, (w0, w0), rnd="rn")
                        acc = cute.arch.fma_packed_f32x2(
                            (x1[0], x0[1]),
                            (w1, w0),
                            (x0w0[0], x1w1[1]),
                            rnd="rn",
                        )
                    tXrAcc[i] = acc[0]
                    tXrAcc[i + 1] = acc[1]
            else:
                for i in cutlass.range_constexpr(cute.size(tXrAcc)):
                    x0w0 = self._mul_rn_f32(x_norm[i], w0)
                    x1w1 = self._mul_rn_f32(tXrAcc[i], w1)
                    if (
                        const_expr(self.tile_d or self.N <= 1024)
                        or tXcX_full[i][1] % 2 != 0
                    ):
                        tXrAcc[i] = self._fma_rn_f32(x_norm[i], w0, x1w1)
                    else:
                        tXrAcc[i] = self._fma_rn_f32(tXrAcc[i], w1, x0w0)
        elif const_expr(use_pairwise_init):
            w_k = sW[k]
            if const_expr(self._is_blackwell):
                for i in cutlass.range_constexpr(0, cute.size(tXrAcc), 2):
                    acc = cute.arch.fma_packed_f32x2(
                        (x_norm[i], x_norm[i + 1]),
                        (w_k, w_k),
                        (tXrAcc[i], tXrAcc[i + 1]),
                        rnd="rn",
                    )
                    tXrAcc[i] = acc[0]
                    tXrAcc[i + 1] = acc[1]
            else:
                for i in cutlass.range_constexpr(cute.size(tXrAcc)):
                    tXrAcc[i] = self._fma_rn_f32(x_norm[i], w_k, tXrAcc[i])
        else:
            tXrAcc.store(tXrAcc.load() + x_norm * sW[k])


class FusedRMSNormCombineFwdParallelTOPK(FusedRMSNormCombineFwd):
    """Normalize and combine BF16 expert rows in one CTA without predication."""

    _PARALLEL_INPUT_CACHE_BYTES = 64 * 1024

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        norm_dtype: Type[cutlass.Numeric],
        N: int,
        TOPK: int,
        use_kahan: bool = False,
        require_bitwise: bool = True,
        tile_d: bool = False,
        copy_input: bool = False,
        has_weight: bool = False,
        has_expert_ids: bool = False,
    ):
        assert (
            dtype is cutlass.BFloat16
            and norm_dtype is Float32
            and self.supports_shape(N, TOPK)
            and not (use_kahan or copy_input or has_weight)
        ), (
            "parallel TOPK requires BF16, FP32 norm, aligned row tiles, and no Kahan/copy/gamma"
        )
        super().__init__(
            dtype,
            norm_dtype,
            N,
            TOPK,
            use_kahan=use_kahan,
            require_bitwise=require_bitwise,
            tile_d=tile_d,
            copy_input=copy_input,
            has_weight=has_weight,
            has_expert_ids=has_expert_ids,
        )
        assert self._is_blackwell, "parallel TOPK requires SM10/SM11"

    @staticmethod
    def supports_shape(N: int, TOPK: int) -> bool:
        threads_per_row = FusedRMSNormCombineFwd._fwd_threads_per_row(N)
        threads = TOPK * threads_per_row
        # Both the input and four-element output copies are unpredicated.
        # Wider standalone BF16 rows use clustered reductions for bitwise parity.
        return (
            TOPK >= 2
            and TOPK & (TOPK - 1) == 0
            and threads <= 1024
            and 0 < N <= 16384
            and N % (4 * threads) == 0
        )

    def _num_threads(self):
        return self.TOPK * self._threads_per_row()

    @classmethod
    def prefers_prefill(cls, N: int, TOPK: int) -> bool:
        if not cls.supports_shape(N, TOPK):
            return False
        threads_per_row = FusedRMSNormCombineFwd._fwd_threads_per_row(N)
        row_elements = N // threads_per_row
        cached_rows = (
            TOPK * N * cutlass.BFloat16.width // 8 <= cls._PARALLEL_INPUT_CACHE_BYTES
        )
        # Full-CTA output partitioning avoids the wide serial row's spill pressure.
        return (cached_rows and row_elements <= cls._FOUR_STAGE_ROW_ELEMENTS) or (
            TOPK * threads_per_row == 1024
            and row_elements > cls._MAX_PIPELINED_ROW_ELEMENTS
        )

    def _min_blocks_per_mp(self):
        return 1

    def _set_cluster_n(self):
        self.cluster_n = 1

    def _sum_sq_chunk(self, words: cute.Tensor, begin: int, end: int, acc: Float32):
        # Same element order as `_fma_dot_f32` over the BF16 fragment.
        for j in range(begin, end):
            lo, hi = _cvt_bf16x2_to_f32x2(words[j])
            acc = self._fma_rn_f32(lo, lo, acc)
            acc = self._fma_rn_f32(hi, hi, acc)
        return acc

    @cute.jit
    def _combine_rows(
        self,
        rows: cute.Tensor,
        rstd: cute.Tensor,
        weights,
        output_copy: cute.TiledCopy,
        tYgY: cute.Tensor,
        next_words: cute.Tensor | None = None,
        *,
        mExpertIds: cute.Tensor | None = None,
        token: Int32 = 0,
    ) -> Float32:
        """Combine the TOPK rows into `tYgY`. With `next_words`, also reduce
        their sum of squares, a chunk after each slot, and return it. Rows 0
        and 1 share the first slot (`supports_shape` requires TOPK >= 2)."""
        tidx, _, _ = cute.arch.thread_idx()
        output_thread = output_copy.get_slice(tidx)
        accum = cute.make_fragment_like(tYgY, Float32)
        # Match scale_and_sum's tiled or alternating-lane first-pair FMA order.
        first_pair_tiled = const_expr(self.tile_d or self.N <= 1024)
        sum_sq = Float32(0.0)
        num_slots = const_expr(self.TOPK - 1)
        for slot in cutlass.range_constexpr(num_slots):
            row_words = []
            for row in cutlass.range_constexpr(0 if slot == 0 else slot + 1, slot + 2):
                row_tile = cute.local_tile(rows, (1, self.N), (row, 0))
                row_input = cute.make_fragment_like(tYgY, self.dtype)
                if const_expr(rows.memspace == cute.AddressSpace.gmem):
                    copy_utils.copy(output_thread.partition_S(row_tile), row_input)
                else:
                    cute.autovec_copy(output_thread.partition_S(row_tile), row_input)
                valid = self._route_valid(mExpertIds, token * self.TOPK + row)
                self._mask_fragment(row_input, valid)
                row_words.append(cute.recast_tensor(row_input, Uint32))
            k = const_expr(slot + 1)
            valid0 = self._route_valid(mExpertIds, token * self.TOPK)
            valid1 = self._route_valid(mExpertIds, token * self.TOPK + k)
            r0 = self._route_scalar(rstd[0], valid0)
            r1 = self._route_scalar(rstd[k], valid1)
            w0 = self._route_scalar(weights[0], valid0)
            w1 = self._route_scalar(weights[k], valid1)
            for j in cutlass.range_constexpr(cute.size(row_words[0])):
                i = const_expr(2 * j)
                if const_expr(slot == 0):
                    x0e, x0o = _cvt_bf16x2_to_f32x2(row_words[0][j])
                    x1e, x1o = _cvt_bf16x2_to_f32x2(row_words[1][j])
                    if const_expr(first_pair_tiled):
                        x1n = cute.arch.mul_packed_f32x2((x1e, x1o), (r1, r1), rnd="rn")
                        x0n = cute.arch.mul_packed_f32x2((x0e, x0o), (r0, r0), rnd="rn")
                        result = cute.arch.mul_packed_f32x2(x1n, (w1, w1), rnd="rn")
                        result = cute.arch.fma_packed_f32x2(
                            x0n, (w0, w0), result, rnd="rn"
                        )
                    else:
                        # Even: x1n * w1 + rn(x0n * w0); odd: x0n * w0 + rn(x1n * w1).
                        lead = cute.arch.mul_packed_f32x2(
                            (x0e, x1o), (r0, r1), rnd="rn"
                        )
                        cross = cute.arch.mul_packed_f32x2(
                            (x1e, x0o), (r1, r0), rnd="rn"
                        )
                        lead = cute.arch.mul_packed_f32x2(lead, (w0, w1), rnd="rn")
                        result = cute.arch.fma_packed_f32x2(
                            cross, (w1, w0), lead, rnd="rn"
                        )
                else:
                    values = cute.arch.mul_packed_f32x2(
                        _cvt_bf16x2_to_f32x2(row_words[0][j]), (r1, r1), rnd="rn"
                    )
                    result = cute.arch.fma_packed_f32x2(
                        values, (w1, w1), (accum[i], accum[i + 1]), rnd="rn"
                    )
                accum[i], accum[i + 1] = result
            if const_expr(next_words is not None):
                num_words = const_expr(cute.size(next_words))
                sum_sq = self._sum_sq_chunk(
                    next_words,
                    slot * num_words // num_slots,
                    (slot + 1) * num_words // num_slots,
                    sum_sq,
                )
        # The streamed output is only known to be 8-byte aligned.
        output = cute.make_fragment_like(tYgY)
        output.store(accum.load().to(self.dtype))
        cute.autovec_copy(output, tYgY)
        return sum_sq

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        mXCopy: cute.Tensor,
        mXCopyOffset: cute.Tensor | None,
        mXCopyCondition: cute.Tensor | None,
        mG: cute.Tensor | None,
        eps: Float32,
        gain_center: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        load_stages: cutlass.Constexpr[int],
    ):
        assert mY.element_type is cutlass.BFloat16
        assert tiler_mn[0] == self.TOPK and tiler_mn[1] == self.N
        assert mXCopyOffset is None and mXCopyCondition is None
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = tidx // threads_per_row
        smem = cutlass.utils.SmemAllocator()
        reread_rows = const_expr(
            self.TOPK * self.N * self.dtype.width // 8
            > self._PARALLEL_INPUT_CACHE_BYTES
        )
        sRstd = smem.allocate_tensor(
            Float32, cute.make_layout((self.TOPK,)), byte_alignment=16
        )
        if const_expr(not reread_rows):
            sInput = smem.allocate_tensor(
                self.dtype,
                cute.make_ordered_layout((self.TOPK, self.N), order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer, mbar = self._allocate_reduction_buffer_and_mbar(
            smem, tiled_copy.layout_tv_tiled
        )
        thr_copy = tiled_copy.get_slice(tidx)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        tXgX = thr_copy.partition_S(gX)
        tXrX = cute.make_fragment_like(tXgX)
        copy_utils.copy(tXgX, tXrX)
        valid = self._route_valid(mExpertIds, bidx * self.TOPK + row)
        self._mask_fragment(tXrX, valid)
        if const_expr(not reread_rows):
            cute.autovec_copy(tXrX, thr_copy.partition_D(sInput))
        x = tXrX.load().to(Float32)
        sum_sq = row_reduce(
            self._fma_dot_f32(x, x),
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar,
            init_val=0.0,
        )
        rstd = self._route_scalar(
            cute.math.rsqrt(sum_sq / self.N + eps, fastmath=False), valid
        )
        if tidx % threads_per_row == 0:
            mRstd[bidx * self.TOPK + row] = rstd
            sRstd[row] = rstd
        cute.arch.barrier()

        output_copy = copy_utils.tiled_copy_2d(
            self.dtype, self._num_threads(), self._num_threads(), 4
        )
        output_thread = output_copy.get_slice(tidx)
        output_tile = (1, self.N)
        gY = cute.local_tile(mY, output_tile, (bidx, 0))
        tYgY = output_thread.partition_D(gY)
        self._combine_rows(
            gX if const_expr(reread_rows) else sInput,
            sRstd,
            cute.local_tile(mW, (self.TOPK,), (bidx,)),
            output_copy,
            tYgY,
            mExpertIds=mExpertIds,
            token=bidx,
        )


class FusedRMSNormCombineFwdParallelTOPKStream(FusedRMSNormCombineFwdParallelTOPK):
    """Persistent parallel TOPK: each CTA streams tokens through a TMA ring.

    Threads read the same fragment elements in the same order as
    `FusedRMSNormCombineFwdParallelTOPK`, so outputs and rstd match it bitwise.
    The input must be 16-byte aligned for TMA.
    """

    _STREAM_SHARED_BYTES = 200 * 1024
    _STREAM_MAX_STAGES = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_stages = self.stream_stages(self.N, self.TOPK)
        assert self.num_stages >= 2, "streamed TOPK rows must double-buffer"
        self.num_sms = num_sms_per_device()

    @classmethod
    def stream_stages(cls, N: int, TOPK: int) -> int:
        token_bytes = TOPK * N * cutlass.BFloat16.width // 8
        return min(cls._STREAM_MAX_STAGES, cls._STREAM_SHARED_BYTES // token_bytes)

    @staticmethod
    def prefers_stream(B: int, N: int, TOPK: int) -> bool:
        """Stream once every persistent CTA owns at least `stages + 2` tokens.

        Below that the per-token kernel still fits in one resident wave and ties
        or wins (GB300, 8 x 4096: stream is 19% slower at 256 tokens and even
        from 384 through 768, 7-11% faster from 1024 through 16384)."""
        stages = FusedRMSNormCombineFwdParallelTOPKStream.stream_stages(N, TOPK)
        return stages >= 2 and B >= (stages + 2) * num_sms_per_device()

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        mXCopy: cute.Tensor,
        mXCopyOffset: cute.Tensor | None,
        mXCopyCondition: cute.Tensor | None,
        mG: cute.Tensor | None,
        eps: Float32,
        gain_center: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype and mG is None
        assert mXCopyOffset is None and mXCopyCondition is None
        self._set_cluster_n()
        vecsize = self._vector_size(mY.element_type.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        tma_atom, mX_tma = _make_row_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX, tiler_mn
        )
        self.kernel(
            tma_atom,
            mX_tma,
            mW,
            mY,
            mRstd,
            mExpertIds,
            eps,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            self.num_stages,
        ).launch(
            grid=[cutlass.min(mY.shape[0], self.num_sms), 1, 1],
            block=[tiled_copy.size, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _load_token(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        full: cute.Pointer,
        token: Int32,
        stage: Int32,
    ):
        token_bytes = const_expr(self.TOPK * self.N * self.dtype.width // 8)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full + stage, token_bytes)
        # cutlass-dsl 4.6 elects the issuing lane inside a bulk-TMA copy, so the
        # whole warp must call it: a single-lane caller deadlocks.
        cute.copy(
            tma_atom, tXgX[None, token], tXsX[None, stage], tma_bar_ptr=full + stage
        )

    def _token_words(
        self,
        thr_copy: cute.TiledCopy,
        sX: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        token: Int32,
    ) -> cute.Tensor:
        """Load this thread's row fragment and clear inactive routes before packing."""
        tXsX = thr_copy.partition_S(sX)
        tXrX = cute.make_fragment_like(tXsX)
        cute.autovec_copy(tXsX, tXrX)
        row = cute.arch.thread_idx()[0] // self._threads_per_row()
        self._mask_fragment(
            tXrX, self._route_valid(mExpertIds, token * self.TOPK + row)
        )
        return cute.recast_tensor(tXrX, Uint32)

    @cute.jit
    def _publish_scales(
        self,
        sPart: cute.Tensor,
        sScale: cute.Tensor,
        weight: Float32,
        mRstd: cute.Tensor,
        eps: Float32,
        token: Int32,
        store_rstd,
        warps_per_row: cutlass.Constexpr[int],
        mExpertIds: cute.Tensor | None,
    ):
        """Lane r < TOPK turns row r's warp partials into rstd and stores it with
        the row weight in this warp's private scale slot."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        if lane < self.TOPK:
            base = lane * warps_per_row
            # block_reduce's butterfly over lanes holding the warp partials
            # (zeros elsewhere) associates as below.
            if const_expr(warps_per_row == 1):
                sum_sq = sPart[base]
            elif const_expr(warps_per_row == 2):
                sum_sq = sPart[base] + sPart[base + 1]
            else:
                assert warps_per_row == 4
                sum_sq = (sPart[base] + sPart[base + 2]) + (
                    sPart[base + 1] + sPart[base + 3]
                )
            valid = self._route_valid(mExpertIds, token * self.TOPK + lane)
            rstd = self._route_scalar(
                cute.math.rsqrt(sum_sq / self.N + eps, fastmath=False), valid
            )
            sScale[lane, 0, warp] = rstd
            sScale[lane, 1, warp] = self._route_scalar(weight, valid)
            if warp == 0:
                if store_rstd:
                    mRstd[token * self.TOPK + lane] = rstd
        cute.arch.sync_warp()

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        tma_atom: cute.CopyAtom,
        mX_tma: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        load_stages: cutlass.Constexpr[int],
    ):
        """Software-pipelined: iteration i combines token i while reducing
        token i + 1's sums of squares, with one CTA barrier per token."""
        assert mY.element_type is cutlass.BFloat16
        assert tiler_mn[0] == self.TOPK and tiler_mn[1] == self.N
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        warps_per_row = const_expr(threads_per_row // cute.arch.WARP_SIZE)
        num_warps = const_expr(self.TOPK * warps_per_row)
        smem = cutlass.utils.SmemAllocator()
        sInput = smem.allocate_tensor(
            self.dtype,
            cute.make_ordered_layout((self.TOPK, self.N, load_stages), order=(1, 0, 2)),
            byte_alignment=128,
        )
        # Warp partials by token parity: token i + 2 overwrites a slot only after
        # the barrier that follows every warp's read of token i's.
        sPart = smem.allocate_tensor(
            Float32, cute.make_layout((num_warps, 2)), byte_alignment=16
        )
        # Per-warp (rstd, weight) slot for the token being combined, so publishing
        # token i + 1's needs no second CTA barrier.
        sScale = smem.allocate_tensor(
            Float32, cute.make_layout((self.TOPK, 2, num_warps)), byte_alignment=16
        )
        full = smem.allocate_array(Int64, num_elems=load_stages)
        if tidx == 0:
            for stage in cutlass.range_constexpr(load_stages):
                cute.arch.mbarrier_init(full + stage, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        tXsX_tma, tXgX_tma = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sInput, 0, 2),
            cute.group_modes(cute.local_tile(mX_tma, tiler_mn, (None, 0)), 0, 2),
        )

        num_tokens = (mY.shape[0] - bidx + grid_x - 1) // grid_x
        if warp == 0:
            for stage in cutlass.range_constexpr(load_stages):
                if stage < num_tokens:
                    self._load_token(
                        tma_atom,
                        tXgX_tma,
                        tXsX_tma,
                        full,
                        bidx + stage * grid_x,
                        stage,
                    )
        thr_copy = tiled_copy.get_slice(tidx)
        num_threads = const_expr(self._num_threads())
        output_vec = const_expr(8 if self.N % (8 * num_threads) == 0 else 4)
        output_copy = copy_utils.tiled_copy_2d(
            self.dtype, num_threads, num_threads, output_vec
        )
        output_thread = output_copy.get_slice(tidx)
        output_tile = (1, self.N)
        weight_lane = lane % self.TOPK

        weight = Float32(mW[bidx * self.TOPK + weight_lane])
        cute.arch.mbarrier_wait(full, 0)
        words = self._token_words(thr_copy, sInput[None, None, 0], mExpertIds, bidx)
        sum_sq = self._sum_sq_chunk(words, 0, cute.size(words), Float32(0.0))
        sum_sq = warp_reduce(sum_sq, operator.add, threads_in_group=cute.arch.WARP_SIZE)
        if lane == 0:
            sPart[warp, 0] = sum_sq
        cute.arch.barrier()
        self._publish_scales(
            sPart[None, 0],
            sScale,
            weight,
            mRstd,
            eps,
            bidx,
            True,
            warps_per_row,
            mExpertIds,
        )

        for i in cutlass.range(num_tokens, unroll=1):
            token = bidx + i * grid_x
            has_next = i + 1 < num_tokens
            next_token = bidx + cutlass.min(i + 1, num_tokens - 1) * grid_x
            weight = Float32(mW[next_token * self.TOPK + weight_lane])
            if has_next:
                cute.arch.mbarrier_wait(
                    full + (i + 1) % load_stages, ((i + 1) // load_stages) % 2
                )
            scales = cute.make_rmem_tensor((self.TOPK, 2), Float32)
            cute.autovec_copy(sScale[None, None, warp], scales)
            gY = cute.local_tile(mY, output_tile, (token, 0))
            # The last iteration reduces a stale stage with no TMA in flight and
            # discards the result.
            sum_sq = self._combine_rows(
                sInput[None, None, i % load_stages],
                scales[None, 0],
                scales[None, 1],
                output_copy,
                output_thread.partition_D(gY),
                self._token_words(
                    thr_copy,
                    sInput[None, None, (i + 1) % load_stages],
                    mExpertIds,
                    next_token,
                ),
                mExpertIds=mExpertIds,
                token=token,
            )
            sum_sq = warp_reduce(
                sum_sq, operator.add, threads_in_group=cute.arch.WARP_SIZE
            )
            if lane == 0:
                sPart[warp, (i + 1) % 2] = sum_sq
            cute.arch.barrier()
            # Every thread has combined token i, so its stage is free.
            if warp == 0:
                if i + load_stages < num_tokens:
                    self._load_token(
                        tma_atom,
                        tXgX_tma,
                        tXsX_tma,
                        full,
                        token + load_stages * grid_x,
                        i % load_stages,
                    )
            self._publish_scales(
                sPart[None, (i + 1) % 2],
                sScale,
                weight,
                mRstd,
                eps,
                next_token,
                has_next,
                warps_per_row,
                mExpertIds,
            )


class FusedRMSNormCombineFwdParallelTOPKPrefetch(FusedRMSNormCombineFwdParallelTOPK):
    """Persistent parallel TOPK for tokens too wide to double-buffer in SMEM.

    Each CTA walks its tokens, rereading rows from gmem as
    `FusedRMSNormCombineFwdParallelTOPK` does, and TMA-prefetches the next
    token into L2 once the current token's rstd is published, so the HBM read
    overlaps the combine pass. Outputs and rstd match the per-token kernel
    bitwise. The input must be 16-byte aligned for TMA.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            FusedRMSNormCombineFwdParallelTOPKStream.stream_stages(self.N, self.TOPK)
            < 2
        ), "rows that fit a TMA ring should stream"
        self.num_sms = num_sms_per_device()

    @staticmethod
    def prefers_prefetch(B: int, N: int, TOPK: int) -> bool:
        """Prefetch once every persistent CTA owns at least six tokens.

        GB300, 8 x 12288: 5-20% faster from 1024 through 16384 tokens, even at
        768 and 8% slower at 256. A second token in flight thrashes L2 (29%
        slower than per-token), and prefetching before the rstd reduction
        leaves 16384 tokens 13% rather than 20% faster."""
        stages = FusedRMSNormCombineFwdParallelTOPKStream.stream_stages(N, TOPK)
        return stages < 2 and B >= 6 * num_sms_per_device()

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        mXCopy: cute.Tensor,
        mXCopyOffset: cute.Tensor | None,
        mXCopyCondition: cute.Tensor | None,
        mG: cute.Tensor | None,
        eps: Float32,
        gain_center: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype and mG is None
        assert mXCopyOffset is None and mXCopyCondition is None
        self._set_cluster_n()
        vecsize = self._vector_size(mY.element_type.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        tma_atom, mX_tma = _make_row_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX, tiler_mn
        )
        self.kernel(
            tma_atom,
            mX_tma,
            mX,
            mW,
            mY,
            mRstd,
            mExpertIds,
            eps,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cutlass.min(mY.shape[0], self.num_sms), 1, 1],
            block=[tiled_copy.size, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom: cute.CopyAtom,
        mX_tma: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mExpertIds: cute.Tensor | None,
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        assert mY.element_type is cutlass.BFloat16
        assert tiler_mn[0] == self.TOPK and tiler_mn[1] == self.N
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        row = tidx // threads_per_row
        smem = cutlass.utils.SmemAllocator()
        # Double-buffered by token parity: with one warp per row, row_reduce has
        # no CTA barrier, so a warp can publish token i + 1's rstd while others
        # still read token i's in pass 2.
        sRstd = smem.allocate_tensor(
            Float32, cute.make_layout((self.TOPK, 2)), byte_alignment=16
        )
        # Prefetches never touch SMEM; tma_partition only needs the tile layout.
        sAnchor = smem.allocate_tensor(
            self.dtype, cute.make_layout(128 // (self.dtype.width // 8)), 128
        )
        sTile = cute.make_tensor(
            sAnchor.iterator,
            cute.make_ordered_layout((self.TOPK, self.N), order=(1, 0)),
        )
        reduction_buffer, mbar = self._allocate_reduction_buffer_and_mbar(
            smem, tiled_copy.layout_tv_tiled
        )
        _, tXgX_tma = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sTile, 0, 2),
            cute.group_modes(cute.local_tile(mX_tma, tiler_mn, (None, 0)), 0, 2),
        )

        num_tokens = (mY.shape[0] - bidx + grid_x - 1) // grid_x
        # cutlass-dsl 4.6 elects the issuing lane inside bulk-TMA ops, so the
        # whole warp issues each prefetch.
        if warp == 0:
            cute.prefetch(tma_atom, tXgX_tma[None, bidx])
        thr_copy = tiled_copy.get_slice(tidx)
        output_copy = copy_utils.tiled_copy_2d(
            self.dtype, self._num_threads(), self._num_threads(), 4
        )
        output_thread = output_copy.get_slice(tidx)
        output_tile = (1, self.N)
        for i in cutlass.range(num_tokens, unroll=1):
            token = bidx + i * grid_x
            gX = cute.local_tile(mX, tiler_mn, (token, 0))
            tXgX = thr_copy.partition_S(gX)
            tXrX = cute.make_fragment_like(tXgX)
            copy_utils.copy(tXgX, tXrX)
            valid = self._route_valid(mExpertIds, token * self.TOPK + row)
            self._mask_fragment(tXrX, valid)
            x = tXrX.load().to(Float32)
            sum_sq = row_reduce(
                self._fma_dot_f32(x, x),
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar,
                init_val=0.0,
            )
            rstd = self._route_scalar(
                cute.math.rsqrt(sum_sq / self.N + eps, fastmath=False), valid
            )
            if tidx % threads_per_row == 0:
                mRstd[token * self.TOPK + row] = rstd
                sRstd[row, i % 2] = rstd
            cute.arch.barrier()
            if warp == 0:
                if i + 1 < num_tokens:
                    cute.prefetch(tma_atom, tXgX_tma[None, token + grid_x])

            gY = cute.local_tile(mY, output_tile, (token, 0))
            tYgY = output_thread.partition_D(gY)
            self._combine_rows(
                gX,
                sRstd[None, i % 2],
                cute.local_tile(mW, (self.TOPK,), (token,)),
                output_copy,
                tYgY,
                mExpertIds=mExpertIds,
                token=token,
            )
