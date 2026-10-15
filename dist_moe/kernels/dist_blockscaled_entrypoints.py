# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device entrypoints for staged blockscaled dispatch and combine."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from ._dsl_compat import thread_exit
from .activation_buffer import (
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_COL_Q_OFFSET,
    ACTIVATION_COL_SCALE_OFFSET,
    ACTIVATION_SOURCE_X_OFFSET,
    ACTIVATION_SOURCE_Y_OFFSET,
)
from .activation_buffer_kernel import (
    _activation_buffer_col_scale_storage_byte_extent,
    _activation_buffer_row_global_scale_tensor,
    _activation_buffer_row_scale_storage_byte_extent,
    _activation_buffer_tensor_with_byte_extent,
)
from .blockscaled_grouped_gemm import NVFP4
from .combine_swiglu_quant import (
    _combine_swiglu_bwd_quant_producer_body,
    _combine_swiglu_fwd_quant_producer_body,
)
from .grouped_gemm_kernel import (
    _activation_buffer_rows,
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
)
from .params import (
    CombineSwigluQuantParams,
    GroupedGemmEpilogPipeline,
    GroupedGemmMmaPipeline,
    GroupedGemmPipelineSync,
    GroupedGemmProblem,
    GroupedGemmTmaPipeline,
    params_from_kernel,
)
from .tile_scheduler import stage_expert_metadata


@cute.kernel
def dispatch_kernel(  # noqa: C901
    self,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tma_atom_sfa: cute.CopyAtom,
    tma_atom_sfb: cute.CopyAtom,
    tma_atom_c: cute.CopyAtom,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    mC: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    sfb_tma_smem_layout_staged: cute.Layout,
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: cute.Tile,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    split_sizes: cute.Tensor,
    counter_ptr: cute.Pointer,
    tensormaps: cute.Tensor,
    mDispatchQuantGatherPtrs: cute.Tensor,
    mDispatchQuantWorkCounter: cute.Tensor,
    mDispatchQuantDoneCounter: cute.Tensor,
    mDispatchQuantRowQWords: cute.Tensor,
    mDispatchQuantRowScale: cute.Tensor,
    mDispatchQuantRowGlobalScaleInv: cute.Tensor,
    mDispatchQuantColQWords: cute.Tensor,
    mDispatchQuantColScale: cute.Tensor,
    G: cutlass.Constexpr[int],
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    local_rank: cutlass.Int32,
    dispatch_quant_total_tiles: cutlass.Int32,
    use_device_tensormaps: cutlass.Constexpr[bool],
    activation_buffer_base_ptr: cutlass.Int64,
    activation_buffer_size_bytes: cutlass.Int64,
    activation_offsets: cute.Tensor,
    use_activation_buffer: cutlass.Constexpr[bool],
    conditional_execution: cute.Tensor,
    use_conditional_execution: cutlass.Constexpr[bool],
    b_global_scale_inv_ptr: cutlass.Int64,
    use_global_scale_inv: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(use_conditional_execution):
        if conditional_execution[0] == cutlass.Int32(0):
            thread_exit()
    if cutlass.const_expr(use_activation_buffer):
        activation_rows = _activation_buffer_rows(split_sizes, G)
        activation_value_count = cutlass.Int64(activation_rows) * cutlass.Int64(K)
        activation_q_byte_extent = (
            activation_value_count
            * cutlass.Int64(cutlass.const_expr(self.format.a_dtype.width))
            // cutlass.Int64(8)
        )
        row_scale_byte_extent = _activation_buffer_row_scale_storage_byte_extent(
            activation_rows,
            K,
            sf_vec_size=self.sf_vec_size,
            sf_dtype_width=self.format.sf_dtype.width,
        )
        col_scale_byte_extent = _activation_buffer_col_scale_storage_byte_extent(
            activation_rows,
            K,
            sf_vec_size=self.sf_vec_size,
            sf_dtype_width=self.format.sf_dtype.width,
        )
        mDispatchQuantRowQWords = _activation_buffer_tensor_with_byte_extent(
            mDispatchQuantRowQWords,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_A_Q_OFFSET,
            cutlass.Uint32,
            activation_q_byte_extent,
        )
        mDispatchQuantRowScale = _activation_buffer_tensor_with_byte_extent(
            mDispatchQuantRowScale,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_A_SCALE_OFFSET,
            cutlass.Uint8,
            row_scale_byte_extent,
        )
        if cutlass.const_expr(use_global_scale_inv):
            mDispatchQuantRowGlobalScaleInv = (
                _activation_buffer_row_global_scale_tensor(
                    mDispatchQuantRowGlobalScaleInv,
                    activation_buffer_base_ptr,
                    activation_buffer_size_bytes,
                    activation_offsets,
                    activation_rows,
                    K,
                    sf_vec_size=self.sf_vec_size,
                    sf_dtype_width=self.format.sf_dtype.width,
                )
            )
        mDispatchQuantColQWords = _activation_buffer_tensor_with_byte_extent(
            mDispatchQuantColQWords,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_COL_Q_OFFSET,
            cutlass.Uint32,
            activation_q_byte_extent,
        )
        mDispatchQuantColScale = _activation_buffer_tensor_with_byte_extent(
            mDispatchQuantColScale,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_COL_SCALE_OFFSET,
            cutlass.Uint8,
            col_scale_byte_extent,
        )
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    cluster_cta_rank = cutlass.Int32(0)
    pred_cta0 = True
    if cutlass.const_expr(self.NUM_CTAS == 2):
        cluster_cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pred_cta0 = cluster_cta_rank == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    ab_full_mbar = storage.smem_full_mbar.data_ptr()
    ab_empty_mbar = storage.smem_empty_mbar.data_ptr()
    tmem_full_mbar = storage.tmem_full_mbar.data_ptr()
    tmem_empty_mbar = storage.tmem_empty_mbar.data_ptr()
    tile_consumer_mbar = storage.tile_consumer_mbar.data_ptr()
    tile_producer_mbar = storage.tile_producer_mbar.data_ptr()
    tile_cta_bar_mbar = (
        storage.tile_cta_bar_mbar.data_ptr() if self.NUM_TILE_CTA_BARS > 0 else None
    )
    tmem_dealloc_mbar = (
        storage.tmem_dealloc_mbar.data_ptr() if self.NUM_CTAS == 2 else None
    )
    cross_seam_mbar = (
        storage.cross_seam_mbar.data_ptr() if self.OVERLAPPING_ACCUM else None
    )
    tile_id_smem_ptr = storage.tile_id_smem.data_ptr()
    dispatch_quant_tile_smem_ptr = storage.dispatch_quant_tile_smem.data_ptr()
    dispatch_quant_gather_ptr_smem_ptr = (
        storage.dispatch_quant_gather_ptr_smem.data_ptr()
    )
    a_global_scale_inv_smem_ptr = (
        storage.a_global_scale_inv.data_ptr() if use_global_scale_inv else None
    )
    tmem_holding_buf = storage.tmem_holding_buf

    if warp_idx == self.EPILOG_WARP_IDS[0]:
        with cute.arch.elect_one():
            for i in range(self.NUM_SMEM_BUFFERS):
                cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                cute.arch.mbarrier_init(ab_full_mbar + i, 1)
            for i in range(self.NUM_TMEM_BUFFERS):
                cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                cute.arch.mbarrier_init(tmem_empty_mbar + i, self.NUM_CTAS)
            for i in range(self.NUM_TILE_BUFFERS):
                cute.arch.mbarrier_init(tile_consumer_mbar + i, 1)
                cute.arch.mbarrier_init(tile_producer_mbar + i, self.NUM_CTAS)
            if cutlass.const_expr(self.NUM_TILE_CTA_BARS > 0):
                for i in range(self.NUM_TILE_CTA_BARS):
                    cute.arch.mbarrier_init(tile_cta_bar_mbar + i, 1)
                cute.arch.mbarrier_init(tmem_dealloc_mbar, 32)
            if cutlass.const_expr(self.OVERLAPPING_ACCUM):
                cute.arch.mbarrier_init(cross_seam_mbar, self.NUM_CTAS)

    cute.arch.mbarrier_init_fence()
    if cutlass.const_expr(self.NUM_CTAS == 2):
        cute.arch.fence_proxy(kind="async.shared", space="cluster")
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    if warp_idx == self.EPILOG_WARP_IDS[0]:
        cute.arch.alloc_tmem(
            self.num_tmem_alloc_cols,
            tmem_holding_buf,
            is_two_cta=(self.NUM_CTAS == 2),
        )

    split_sizes = stage_expert_metadata(
        split_sizes,
        cute.recast_ptr(storage.tensormap_buffer.data_ptr(), dtype=cutlass.Int32),
        G,
        synchronize=False,
    )
    cute.arch.barrier(
        barrier_id=_BAR_FULL_CTA_SYNC, number_of_threads=self.THREADS_PER_CTA
    )

    sA = storage.sA.get_tensor(
        a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
    )
    sB = storage.sB.get_tensor(
        b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
    )
    sC = storage.sC.get_tensor(
        epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
    )
    sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
    sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
    sSFB_tma = storage.sSFB.get_tensor(sfb_tma_smem_layout_staged)

    gA = cute.local_tile(
        mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
    )
    gB = cute.local_tile(
        mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
    )
    gC = cute.local_tile(
        mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
    )
    gSFA = cute.local_tile(
        mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
    )
    gSFB = cute.local_tile(
        mSFB,
        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
        (None, None, None),
    )

    bid = cute.arch.block_idx()
    mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cluster_cta_rank)
    block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
        cluster_cta_rank
    )

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCgSFA = thr_mma.partition_A(gSFA)
    tCgSFB = thr_mma_sfb.partition_B(gSFB)

    a_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
    )
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    b_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)

    sfb_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
    )
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        block_in_cluster_coord_sfb_vmnk[1],
        sfb_cta_layout,
        cute.group_modes(sSFB_tma, 0, 3),
        cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C((self.BLOCK_SIZE_M, self.BLOCK_SIZE_N))
    tCtAcc_fake = tiled_mma.make_fragment_C(
        cute.append(acc_shape, self.NUM_TMEM_BUFFERS)
    )
    if cutlass.const_expr(self.OVERLAPPING_ACCUM):
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(
                    tCtAcc_fake.stride[0],
                    tCtAcc_fake.stride[1],
                    tCtAcc_fake.stride[2],
                    (self.BLOCK_SIZE_N - self._num_sf_tmem_cols)
                    * tCtAcc_fake.stride[0][1],
                ),
            ),
        )

    a_full_mcast_mask = None
    b_full_mcast_mask = None
    sfa_full_mcast_mask = None
    sfb_full_mcast_mask = None
    ab_empty_mcast_mask = None
    acc_full_mcast_mask = None
    if cutlass.const_expr(self.NUM_CTAS == 2):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        sfa_full_mcast_mask = a_full_mcast_mask
        sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_sfb_vmnk,
            block_in_cluster_coord_sfb_vmnk,
            mcast_mode=1,
        )
        block_in_cluster_coord_vmnk_peer = (
            block_in_cluster_coord_vmnk[0] ^ 1,
            *block_in_cluster_coord_vmnk[1:],
        )
        a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
        )
        b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
        )
        ab_empty_mcast_mask = (
            a_full_mcast_mask
            | b_full_mcast_mask
            | a_full_mcast_mask_peer
            | b_full_mcast_mask_peer
        )
        acc_full_mcast_mask = cute.make_layout_image_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
        )

    num_tma_load_bytes, _, _ = self._num_tma_load_bytes(
        sA,
        sB,
        sSFA,
        sSFB,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_tma_smem_layout_staged,
        tiled_mma,
    )

    tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)
        cpasync.prefetch_descriptor(tma_atom_c)

    problem = GroupedGemmProblem(
        split_sizes=split_sizes,
        groups=G,
        mnk=(M, N, K),
        local_rank=local_rank,
    )
    sync = GroupedGemmPipelineSync(
        ab_full_mbar=ab_full_mbar,
        ab_empty_mbar=ab_empty_mbar,
        tmem_full_mbar=tmem_full_mbar,
        tmem_empty_mbar=tmem_empty_mbar,
        tile_consumer_mbar=tile_consumer_mbar,
        tile_producer_mbar=tile_producer_mbar,
        tile_cta_bar_mbar=tile_cta_bar_mbar,
        tile_id_smem_ptr=tile_id_smem_ptr,
        cross_seam_mbar=cross_seam_mbar,
        counter_ptr=counter_ptr,
        tma_mcast_masks=(
            a_full_mcast_mask,
            b_full_mcast_mask,
            sfa_full_mcast_mask,
            sfb_full_mcast_mask,
        ),
        ab_empty_mcast_mask=ab_empty_mcast_mask,
        acc_full_mcast_mask=acc_full_mcast_mask,
        cluster_cta_rank=cluster_cta_rank,
        pred_cta0=pred_cta0,
    )

    if warp_idx == self.TMA_AB_WARP_ID:
        pipeline = GroupedGemmTmaPipeline(
            tma_atom_a=tma_atom_a,
            tma_atom_b=tma_atom_b,
            tma_atom_sfa=tma_atom_sfa,
            tma_atom_sfb=tma_atom_sfb,
            gA=tAgA,
            gB=tBgB,
            gSFA=tAgSFA,
            gSFB=tBgSFB,
            sA=tAsA,
            sB=tBsB,
            sSFA=tAsSFA,
            sSFB=tBsSFB,
            num_tma_load_bytes=num_tma_load_bytes,
        )
        self._tma_producer_body(
            pipeline=pipeline,
            problem=problem,
            sync=sync,
            tensormap_manager=tensormap_manager,
            tensormaps=tensormaps,
            tile_producer_hook_tensor=mDispatchQuantDoneCounter,
            use_device_tensormaps=use_device_tensormaps,
        )
    elif warp_idx == self.MMA_WARP_ID:
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf,
        )
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tCtSFA, tCtSFB, tCtSFB_field = self._scale_tmem_tensors(
            tmem_ptr,
            tCtAcc_base,
            tiled_mma,
            self.mma_tiler,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
        )
        pipeline = GroupedGemmMmaPipeline(
            tiled_mma=tiled_mma,
            tCrA=tCrA,
            tCrB=tCrB,
            tCtAcc_base=tCtAcc_base,
            tCtSFA=tCtSFA,
            tCtSFA_copy=tCtSFA,
            tCtSFB=tCtSFB_field,
            tCtSFB_copy=tCtSFB,
            sSFA=sSFA,
            sSFB=sSFB,
        )
        self._mma_consumer_body(pipeline, problem, sync)
    elif warp_idx in self.EPILOG_WARP_IDS:
        epilogue_tidx = tidx - cutlass.Int32(
            self.EPILOG_WARP_IDS[0] * self.THREADS_PER_WARP
        )
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf,
        )
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        (
            tiled_copy_t2r,
            _tTR_tAcc_base_00,
            tTR_rAcc,
        ) = self._epilog_tmem_copy_and_partition(
            tidx=epilogue_tidx,
            tAcc=tCtAcc_base,
            gC_mnl=tCgC,
            epi_tile=epi_tile,
            use_2cta_instrs=self.NUM_CTAS == 2,
            mma_m_idx=0,
            mma_n_idx=0,
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, sC.element_type)
        (
            tiled_copy_r2s,
            tRS_rC,
            tRS_sC,
        ) = self._epilog_smem_copy_and_partition(
            tiled_copy_t2r=tiled_copy_t2r,
            tTR_rC=tTR_rC,
            tidx=epilogue_tidx,
            sC=sC,
        )
        pipeline = GroupedGemmEpilogPipeline(
            c_atom=tma_atom_c,
            tCtAcc_base=tCtAcc_base,
            c_tensor=tCgC,
            sC=sC,
            epi_tile=epi_tile,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            c_layout=self.c_layout,
            c_dtype=self.c_dtype,
            a_dtype_width=self.a_dtype.width,
        )
        epilog_copy = (tTR_rAcc, tiled_copy_r2s, tRS_rC, tRS_sC)
        self._epilog_consumer_body(
            tidx=epilogue_tidx,
            pipeline=pipeline,
            epilog_copy=epilog_copy,
            problem=problem,
            sync=sync,
            tensormap_manager=tensormap_manager,
            tensormaps=tensormaps,
            use_device_tensormaps=use_device_tensormaps,
            a_global_scale_inv_smem_ptr=a_global_scale_inv_smem_ptr,
            a_global_scale_inv_ptr=cutlass.Int64(
                mDispatchQuantRowGlobalScaleInv.iterator.toint()
            ),
            b_global_scale_inv_ptr=b_global_scale_inv_ptr,
            use_a_global_scale_inv=use_global_scale_inv,
            use_b_global_scale_inv=use_global_scale_inv,
            postprocess_output=mDispatchQuantColQWords,
            postprocess_scale=mDispatchQuantColScale,
            postprocess_store=self.interleaved_fc13,
        )
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=(self.NUM_CTAS == 2))
        cute.arch.barrier(
            barrier_id=_BAR_EPILOG_SYNC,
            number_of_threads=32 * len(self.EPILOG_WARP_IDS),
        )
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            if cutlass.const_expr(self.NUM_CTAS == 2):
                cute.arch.mbarrier_arrive(
                    tmem_dealloc_mbar,
                    peer_cta_rank_in_cluster=cluster_cta_rank ^ 1,
                )
                cute.arch.mbarrier_wait(tmem_dealloc_mbar, 0)
            cute.arch.dealloc_tmem(
                tmem_ptr,
                self.num_tmem_alloc_cols,
                is_two_cta=(self.NUM_CTAS == 2),
            )
    else:
        if warp_idx >= self.DISPATCH_QUANT_FIRST_WARP:
            self._dispatch_quant_producer_body(
                mWorkCounter=mDispatchQuantWorkCounter,
                mDoneCounter=mDispatchQuantDoneCounter,
                tile_smem_ptr=dispatch_quant_tile_smem_ptr,
                dispatch_ptr_smem_ptr=dispatch_quant_gather_ptr_smem_ptr,
                mGatherPtrs=mDispatchQuantGatherPtrs,
                mRowQWords=mDispatchQuantRowQWords,
                mRowScale=mDispatchQuantRowScale,
                mRowGlobalScaleInv=mDispatchQuantRowGlobalScaleInv,
                mColQWords=mDispatchQuantColQWords,
                mColScale=mDispatchQuantColScale,
                split_sizes=split_sizes,
                G=G,
                M=M,
                K=K,
                total_tiles=dispatch_quant_total_tiles,
                local_rank=local_rank,
                use_global_scale_inv=use_global_scale_inv,
            )

    if cutlass.const_expr(self.NUM_CTAS == 2):
        # Keep every active thread alive until peer DSMEM accesses complete.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()


@cute.kernel
def combine_kernel(  # noqa: C901
    self,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tma_atom_sfa: cute.CopyAtom,
    tma_atom_sfb: cute.CopyAtom,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    mC: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    sfb_tma_smem_layout_staged: cute.Layout,
    c_smem_layout_staged: cute.Layout,
    epi_tile: cute.Tile,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    split_sizes: cute.Tensor,
    counter_ptr: cute.Pointer,
    mScatter: cute.Tensor,
    mCombineSwigluX: cute.Tensor,
    mCombineSwigluY: cute.Tensor,
    mCombineSwigluWorkCounter: cute.Tensor,
    mCombineSwigluDoneCounter: cute.Tensor,
    mCombineSwigluRowQWords: cute.Tensor,
    mCombineSwigluRowScale: cute.Tensor,
    mCombineSwigluRowGlobalScaleInv: cute.Tensor,
    mCombineSwigluColQWords: cute.Tensor,
    mCombineSwigluColScale: cute.Tensor,
    tensormaps: cute.Tensor,
    elem_size_bytes_c: cutlass.Constexpr[int],
    G: cutlass.Constexpr[int],
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    local_rank: cutlass.Int32,
    use_device_tensormaps: cutlass.Constexpr[bool],
    activation_buffer_base_ptr: cutlass.Int64,
    activation_buffer_size_bytes: cutlass.Int64,
    activation_offsets: cute.Tensor,
    use_activation_buffer: cutlass.Constexpr[bool],
    conditional_execution: cute.Tensor,
    use_conditional_execution: cutlass.Constexpr[bool],
    b_global_scale_inv_ptr: cutlass.Int64,
    use_global_scale_inv: cutlass.Constexpr[bool],
    nvfp4_recip_lut_ptr: cutlass.Int64,
):
    if cutlass.const_expr(use_conditional_execution):
        if conditional_execution[0] == cutlass.Int32(0):
            thread_exit()
    if cutlass.const_expr(use_activation_buffer):
        activation_rows = _activation_buffer_rows(split_sizes, G)
        source_element_size = cutlass.Int64(
            cutlass.const_expr(mCombineSwigluX.element_type.width // 8)
        )
        if cutlass.const_expr(self.mode == self.COMBINE_SWIGLU_FWD_MODE):
            has_activation_rows = cutlass.Int64(activation_rows > cutlass.Int32(0))
            source_x_byte_extent = (
                has_activation_rows
                * (cutlass.Int64(2) * cutlass.Int64(activation_rows) - cutlass.Int64(1))
                * cutlass.Int64(K)
                * source_element_size
            )
            source_y_byte_extent = source_x_byte_extent
        else:
            source_x_byte_extent = (
                cutlass.Int64(activation_rows)
                * cutlass.Int64(K // cutlass.Int32(2))
                * source_element_size
            )
            source_y_byte_extent = (
                cutlass.Int64(activation_rows) * cutlass.Int64(K) * source_element_size
            )
        activation_value_count = cutlass.Int64(activation_rows) * cutlass.Int64(K)
        activation_q_byte_extent = (
            activation_value_count
            * cutlass.Int64(cutlass.const_expr(self.format.a_dtype.width))
            // cutlass.Int64(8)
        )
        row_scale_byte_extent = _activation_buffer_row_scale_storage_byte_extent(
            activation_rows,
            K,
            sf_vec_size=self.sf_vec_size,
            sf_dtype_width=self.format.sf_dtype.width,
        )
        col_scale_byte_extent = _activation_buffer_col_scale_storage_byte_extent(
            activation_rows,
            K,
            sf_vec_size=self.sf_vec_size,
            sf_dtype_width=self.format.sf_dtype.width,
        )
        mCombineSwigluX = _activation_buffer_tensor_with_byte_extent(
            mCombineSwigluX,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_SOURCE_X_OFFSET,
            mCombineSwigluX.element_type,
            source_x_byte_extent,
        )
        mCombineSwigluY = _activation_buffer_tensor_with_byte_extent(
            mCombineSwigluY,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_SOURCE_Y_OFFSET,
            mCombineSwigluY.element_type,
            source_y_byte_extent,
        )
        mCombineSwigluRowQWords = _activation_buffer_tensor_with_byte_extent(
            mCombineSwigluRowQWords,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_A_Q_OFFSET,
            cutlass.Uint32,
            activation_q_byte_extent,
        )
        mCombineSwigluRowScale = _activation_buffer_tensor_with_byte_extent(
            mCombineSwigluRowScale,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_A_SCALE_OFFSET,
            cutlass.Uint8,
            row_scale_byte_extent,
        )
        if cutlass.const_expr(use_global_scale_inv):
            mCombineSwigluRowGlobalScaleInv = (
                _activation_buffer_row_global_scale_tensor(
                    mCombineSwigluRowGlobalScaleInv,
                    activation_buffer_base_ptr,
                    activation_buffer_size_bytes,
                    activation_offsets,
                    activation_rows,
                    K,
                    sf_vec_size=self.sf_vec_size,
                    sf_dtype_width=self.format.sf_dtype.width,
                )
            )
        mCombineSwigluColQWords = _activation_buffer_tensor_with_byte_extent(
            mCombineSwigluColQWords,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_COL_Q_OFFSET,
            cutlass.Uint32,
            activation_q_byte_extent,
        )
        mCombineSwigluColScale = _activation_buffer_tensor_with_byte_extent(
            mCombineSwigluColScale,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            ACTIVATION_COL_SCALE_OFFSET,
            cutlass.Uint8,
            col_scale_byte_extent,
        )
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    cluster_cta_rank = cutlass.Int32(0)
    pred_cta0 = True
    if cutlass.const_expr(self.NUM_CTAS == 2):
        cluster_cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pred_cta0 = cluster_cta_rank == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    ab_full_mbar = storage.smem_full_mbar.data_ptr()
    ab_empty_mbar = storage.smem_empty_mbar.data_ptr()
    tmem_full_mbar = storage.tmem_full_mbar.data_ptr()
    tmem_empty_mbar = storage.tmem_empty_mbar.data_ptr()
    tile_consumer_mbar = storage.tile_consumer_mbar.data_ptr()
    tile_producer_mbar = storage.tile_producer_mbar.data_ptr()
    tile_cta_bar_mbar = (
        storage.tile_cta_bar_mbar.data_ptr() if self.NUM_TILE_CTA_BARS > 0 else None
    )
    tmem_dealloc_mbar = (
        storage.tmem_dealloc_mbar.data_ptr() if self.NUM_CTAS == 2 else None
    )
    cross_seam_mbar = (
        storage.cross_seam_mbar.data_ptr() if self.OVERLAPPING_ACCUM else None
    )
    tile_id_smem_ptr = storage.tile_id_smem.data_ptr()
    scatter_ptr_smem_ptr = storage.scatter_ptr_smem.data_ptr()
    a_global_scale_inv_smem_ptr = (
        storage.a_global_scale_inv.data_ptr() if use_global_scale_inv else None
    )
    combine_swiglu_tile_smem_ptr = None
    if cutlass.const_expr(
        self.mode == self.COMBINE_SWIGLU_FWD_MODE
        or self.mode == self.COMBINE_SWIGLU_BWD_MODE
    ):
        combine_swiglu_tile_smem_ptr = storage.dispatch_quant_tile_smem.data_ptr()
    tmem_holding_buf = storage.tmem_holding_buf

    if warp_idx == self.EPILOG_WARP_IDS[0]:
        with cute.arch.elect_one():
            for i in range(self.NUM_SMEM_BUFFERS):
                cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                cute.arch.mbarrier_init(ab_full_mbar + i, 1)
            for i in range(self.NUM_TMEM_BUFFERS):
                cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                cute.arch.mbarrier_init(tmem_empty_mbar + i, self.NUM_CTAS)
            for i in range(self.NUM_TILE_BUFFERS):
                cute.arch.mbarrier_init(tile_consumer_mbar + i, 1)
                cute.arch.mbarrier_init(tile_producer_mbar + i, self.NUM_CTAS)
            if cutlass.const_expr(self.NUM_TILE_CTA_BARS > 0):
                for i in range(self.NUM_TILE_CTA_BARS):
                    cute.arch.mbarrier_init(tile_cta_bar_mbar + i, 1)
                cute.arch.mbarrier_init(tmem_dealloc_mbar, 32)
            if cutlass.const_expr(self.OVERLAPPING_ACCUM):
                cute.arch.mbarrier_init(cross_seam_mbar, self.NUM_CTAS)

    cute.arch.mbarrier_init_fence()
    if cutlass.const_expr(self.NUM_CTAS == 2):
        cute.arch.fence_proxy(kind="async.shared", space="cluster")
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
    else:
        cute.arch.barrier(
            barrier_id=_BAR_FULL_CTA_SYNC,
            number_of_threads=self.THREADS_PER_CTA,
        )

    if warp_idx == self.EPILOG_WARP_IDS[0]:
        cute.arch.alloc_tmem(
            self.num_tmem_alloc_cols,
            tmem_holding_buf,
            is_two_cta=(self.NUM_CTAS == 2),
        )

    split_sizes = stage_expert_metadata(
        split_sizes,
        cute.recast_ptr(storage.tensormap_buffer.data_ptr(), dtype=cutlass.Int32),
        G,
        synchronize=False,
    )
    cute.arch.barrier(
        barrier_id=_BAR_FULL_CTA_SYNC,
        number_of_threads=self.THREADS_PER_CTA,
    )

    sA = storage.sA.get_tensor(
        a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
    )
    sB = storage.sB.get_tensor(
        b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
    )
    sC = storage.sC.get_tensor(c_smem_layout_staged)
    sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
    sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
    sSFB_tma = storage.sSFB.get_tensor(sfb_tma_smem_layout_staged)

    gA = cute.local_tile(
        mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
    )
    gB = cute.local_tile(
        mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
    )
    gC = cute.local_tile(
        mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
    )
    gSFA = cute.local_tile(
        mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
    )
    gSFB = cute.local_tile(
        mSFB,
        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
        (None, None, None),
    )

    bid = cute.arch.block_idx()
    mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cluster_cta_rank)
    block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
        cluster_cta_rank
    )

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCgSFA = thr_mma.partition_A(gSFA)
    tCgSFB = thr_mma_sfb.partition_B(gSFB)

    a_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
    )
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    b_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)

    sfb_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
    )
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        block_in_cluster_coord_sfb_vmnk[1],
        sfb_cta_layout,
        cute.group_modes(sSFB_tma, 0, 3),
        cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C((self.BLOCK_SIZE_M, self.BLOCK_SIZE_N))
    tCtAcc_fake = tiled_mma.make_fragment_C(
        cute.append(acc_shape, self.NUM_TMEM_BUFFERS)
    )
    if cutlass.const_expr(self.OVERLAPPING_ACCUM):
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(
                    tCtAcc_fake.stride[0],
                    tCtAcc_fake.stride[1],
                    tCtAcc_fake.stride[2],
                    (self.BLOCK_SIZE_N - self._num_sf_tmem_cols)
                    * tCtAcc_fake.stride[0][1],
                ),
            ),
        )

    a_full_mcast_mask = None
    b_full_mcast_mask = None
    sfa_full_mcast_mask = None
    sfb_full_mcast_mask = None
    ab_empty_mcast_mask = None
    acc_full_mcast_mask = None
    if cutlass.const_expr(self.NUM_CTAS == 2):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        sfa_full_mcast_mask = a_full_mcast_mask
        sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_sfb_vmnk,
            block_in_cluster_coord_sfb_vmnk,
            mcast_mode=1,
        )
        block_in_cluster_coord_vmnk_peer = (
            block_in_cluster_coord_vmnk[0] ^ 1,
            *block_in_cluster_coord_vmnk[1:],
        )
        a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
        )
        b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
        )
        ab_empty_mcast_mask = (
            a_full_mcast_mask
            | b_full_mcast_mask
            | a_full_mcast_mask_peer
            | b_full_mcast_mask_peer
        )
        acc_full_mcast_mask = cute.make_layout_image_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
        )

    num_tma_load_bytes, _, _ = self._num_tma_load_bytes(
        sA,
        sB,
        sSFA,
        sSFB,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_tma_smem_layout_staged,
        tiled_mma,
    )

    tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)

    if cutlass.const_expr(
        self.mode == self.COMBINE_SWIGLU_FWD_MODE
        or self.mode == self.COMBINE_SWIGLU_BWD_MODE
    ):
        tile_producer_hook_tensor = mCombineSwigluDoneCounter
    else:
        tile_producer_hook_tensor = split_sizes

    problem = GroupedGemmProblem(
        split_sizes=split_sizes,
        groups=G,
        mnk=(M, N, K),
        local_rank=local_rank,
    )
    sync = GroupedGemmPipelineSync(
        ab_full_mbar=ab_full_mbar,
        ab_empty_mbar=ab_empty_mbar,
        tmem_full_mbar=tmem_full_mbar,
        tmem_empty_mbar=tmem_empty_mbar,
        tile_consumer_mbar=tile_consumer_mbar,
        tile_producer_mbar=tile_producer_mbar,
        tile_cta_bar_mbar=tile_cta_bar_mbar,
        tile_id_smem_ptr=tile_id_smem_ptr,
        cross_seam_mbar=cross_seam_mbar,
        counter_ptr=counter_ptr,
        tma_mcast_masks=(
            a_full_mcast_mask,
            b_full_mcast_mask,
            sfa_full_mcast_mask,
            sfb_full_mcast_mask,
        ),
        ab_empty_mcast_mask=ab_empty_mcast_mask,
        acc_full_mcast_mask=acc_full_mcast_mask,
        cluster_cta_rank=cluster_cta_rank,
        pred_cta0=pred_cta0,
    )

    if warp_idx == self.TMA_AB_WARP_ID:
        pipeline = GroupedGemmTmaPipeline(
            tma_atom_a=tma_atom_a,
            tma_atom_b=tma_atom_b,
            tma_atom_sfa=tma_atom_sfa,
            tma_atom_sfb=tma_atom_sfb,
            gA=tAgA,
            gB=tBgB,
            gSFA=tAgSFA,
            gSFB=tBgSFB,
            sA=tAsA,
            sB=tBsB,
            sSFA=tAsSFA,
            sSFB=tBsSFB,
            num_tma_load_bytes=num_tma_load_bytes,
        )
        self._tma_producer_body(
            pipeline=pipeline,
            problem=problem,
            sync=sync,
            tensormap_manager=tensormap_manager,
            tensormaps=tensormaps,
            tile_producer_hook_tensor=tile_producer_hook_tensor,
            use_device_tensormaps=use_device_tensormaps,
        )
    elif warp_idx == self.MMA_WARP_ID:
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf,
        )
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tCtSFA, tCtSFB, tCtSFB_field = self._scale_tmem_tensors(
            tmem_ptr,
            tCtAcc_base,
            tiled_mma,
            self.mma_tiler,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
        )
        pipeline = GroupedGemmMmaPipeline(
            tiled_mma=tiled_mma,
            tCrA=tCrA,
            tCrB=tCrB,
            tCtAcc_base=tCtAcc_base,
            tCtSFA=tCtSFA,
            tCtSFA_copy=tCtSFA,
            tCtSFB=tCtSFB_field,
            tCtSFB_copy=tCtSFB,
            sSFA=sSFA,
            sSFB=sSFB,
        )
        self._mma_consumer_body(pipeline, problem, sync)
    elif warp_idx in self.EPILOG_WARP_IDS:
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf,
        )
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        (
            tiled_copy_t2r,
            _tTR_tAcc_base_00,
            tTR_rAcc,
        ) = self._epilog_tmem_copy_and_partition(
            tidx=tidx,
            tAcc=tCtAcc_base,
            gC_mnl=tCgC,
            epi_tile=epi_tile,
            use_2cta_instrs=self.NUM_CTAS == 2,
            mma_m_idx=0,
            mma_n_idx=0,
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, sC.element_type)
        (
            tiled_copy_r2s,
            tRS_rC,
            tRS_sC,
        ) = self._epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, tidx, sC)
        self._combine_epilog_consumer_body(
            tidx=tidx,
            tCtAcc_base=tCtAcc_base,
            tCgC=tCgC,
            sC=sC,
            tTR_rAcc=tTR_rAcc,
            tiled_copy_r2s=tiled_copy_r2s,
            tRS_rC=tRS_rC,
            tRS_sC=tRS_sC,
            epi_tile=epi_tile,
            tmem_full_mbar=tmem_full_mbar,
            tmem_empty_mbar=tmem_empty_mbar,
            tile_consumer_mbar=tile_consumer_mbar,
            tile_producer_mbar=tile_producer_mbar,
            tile_id_smem_ptr=tile_id_smem_ptr,
            cross_seam_mbar=cross_seam_mbar,
            mScatter=mScatter,
            scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
            elem_size_bytes_c=elem_size_bytes_c,
            split_sizes=split_sizes,
            cluster_cta_rank=cluster_cta_rank,
            G=G,
            M=M,
            N=N,
            K=K,
            local_rank=local_rank,
            a_global_scale_inv_smem_ptr=a_global_scale_inv_smem_ptr,
            a_global_scale_inv_ptr=cutlass.Int64(
                mCombineSwigluRowGlobalScaleInv.iterator.toint()
            ),
            b_global_scale_inv_ptr=b_global_scale_inv_ptr,
            use_global_scale_inv=use_global_scale_inv,
        )
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=(self.NUM_CTAS == 2))
        cute.arch.barrier(
            barrier_id=_BAR_EPILOG_SYNC,
            number_of_threads=self.EPILOG_WG_THREADS,
        )
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            if cutlass.const_expr(self.NUM_CTAS == 2):
                cute.arch.mbarrier_arrive(
                    tmem_dealloc_mbar,
                    peer_cta_rank_in_cluster=cluster_cta_rank ^ 1,
                )
                cute.arch.mbarrier_wait(tmem_dealloc_mbar, 0)
            cute.arch.dealloc_tmem(
                tmem_ptr,
                self.num_tmem_alloc_cols,
                is_two_cta=(self.NUM_CTAS == 2),
            )
    else:
        if cutlass.const_expr(self.mode == self.COMBINE_SWIGLU_FWD_MODE):
            if warp_idx >= self.COMBINE_SWIGLU_QUANT_FIRST_WARP:
                if cutlass.const_expr(self.format is NVFP4):
                    quant_warp = warp_idx - self.COMBINE_SWIGLU_QUANT_FIRST_WARP
                    if quant_warp < cutlass.Int32(
                        self.nvfp4_swiglu_quant_groups
                        * self.DISPATCH_QUANT_WARPS_PER_GROUP
                    ):
                        recip_lut_ptr = cute.make_ptr(
                            cutlass.Float32,
                            nvfp4_recip_lut_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        mNvfp4RecipLut = cute.make_tensor(
                            recip_lut_ptr,
                            cute.make_ordered_layout((256,), order=(0,)),
                        )
                        self._combine_swiglu_nvfp4_row_quant_producer_body(
                            mWorkCounter=mCombineSwigluWorkCounter,
                            mDoneCounter=mCombineSwigluDoneCounter,
                            tile_smem_ptr=combine_swiglu_tile_smem_ptr,
                            mFwdX=mCombineSwigluX,
                            mFwdY=mCombineSwigluY,
                            mRowQWords=mCombineSwigluRowQWords,
                            mRowScale=mCombineSwigluRowScale,
                            mRowGlobalScaleInv=mCombineSwigluRowGlobalScaleInv,
                            mNvfp4RecipLut=mNvfp4RecipLut,
                            split_sizes=split_sizes,
                            G=G,
                            M=M,
                            local_rank=local_rank,
                            precomputed_swiglu=self.combine_precomputed_swiglu,
                        )
                else:
                    _combine_swiglu_fwd_quant_producer_body(
                        params_from_kernel(CombineSwigluQuantParams, self),
                        mWorkCounter=mCombineSwigluWorkCounter,
                        mDoneCounter=mCombineSwigluDoneCounter,
                        tile_smem_ptr=combine_swiglu_tile_smem_ptr,
                        mFwdX=mCombineSwigluX,
                        mFwdY=mCombineSwigluY,
                        mRowQWords=mCombineSwigluRowQWords,
                        mRowScale=mCombineSwigluRowScale,
                        mColQWords=mCombineSwigluColQWords,
                        mColScale=mCombineSwigluColScale,
                        split_sizes=split_sizes,
                        G=G,
                        M=M,
                        K=K,
                        total_tiles=cutlass.Int32(0),
                        local_rank=local_rank,
                    )
        elif cutlass.const_expr(self.mode == self.COMBINE_SWIGLU_BWD_MODE):
            if warp_idx >= self.COMBINE_SWIGLU_QUANT_FIRST_WARP:
                _combine_swiglu_bwd_quant_producer_body(
                    params_from_kernel(CombineSwigluQuantParams, self),
                    mWorkCounter=mCombineSwigluWorkCounter,
                    mDoneCounter=mCombineSwigluDoneCounter,
                    tile_smem_ptr=combine_swiglu_tile_smem_ptr,
                    mDz=mCombineSwigluX,
                    mH1=mCombineSwigluY,
                    mRowQWords=mCombineSwigluRowQWords,
                    mRowScale=mCombineSwigluRowScale,
                    mDxyColQWords=mCombineSwigluColQWords,
                    mDxyColScale=mCombineSwigluColScale,
                    split_sizes=split_sizes,
                    G=G,
                    M=M,
                    K=K,
                    local_rank=local_rank,
                )

    if cutlass.const_expr(self.NUM_CTAS == 2):
        # Keep every active thread alive until peer DSMEM accesses complete.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
