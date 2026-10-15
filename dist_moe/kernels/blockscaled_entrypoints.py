# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel entrypoints for blockscaled grouped GEMM."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from . import sm103_blockscaled_helpers as sm103
from .grouped_gemm_kernel import (
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
)
from .params import (
    GroupedGemmEpilogPipeline,
    GroupedGemmMmaPipeline,
    GroupedGemmPipelineSync,
    GroupedGemmProblem,
    GroupedGemmTmaPipeline,
)
from .tile_scheduler import stage_expert_metadata


@cute.kernel
def kernel(  # noqa: C901
    self,
    # ---- TMA atoms / global tensors ---------------------------------
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
    # ---- Layouts -----------------------------------------------------
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
    # ---- Group + scheduler inputs -----------------------------------
    split_sizes: cute.Tensor,
    counter_ptr: cute.Pointer,
    tensormaps: cute.Tensor,
    # ---- Problem dimensions -----------------------------------------
    G: cutlass.Constexpr[int],
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    local_rank: cutlass.Int32,
    use_device_tensormaps: cutlass.Constexpr[bool],
    a_global_scale_inv_ptr: cutlass.Int64,
    b_global_scale_inv_ptr: cutlass.Int64,
    use_a_global_scale_inv: cutlass.Constexpr[bool],
    use_b_global_scale_inv: cutlass.Constexpr[bool],
    # ---- Constexpr flags --------------------------------------------
):
    # ----- Warp / cluster identity (TLX prologue) -------------------
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    cluster_cta_rank = cutlass.Int32(0)
    pred_cta0 = True
    if cutlass.const_expr(self.NUM_CTAS == 2):
        cluster_cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pred_cta0 = cluster_cta_rank == 0

    # ----- SMEM / mbarrier allocation -------------------------------
    smem = utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    ab_full_mbar = storage.smem_full_mbar.data_ptr()
    ab_empty_mbar = storage.smem_empty_mbar.data_ptr()
    sf_full_mbar = (
        storage.sf_smem_full_mbar.data_ptr() if self.USE_SM103_ULTRA else None
    )
    sf_empty_mbar = (
        storage.sf_smem_empty_mbar.data_ptr() if self.USE_SM103_ULTRA else None
    )
    tmem_full_mbar = storage.tmem_full_mbar.data_ptr()
    tmem_empty_mbar = storage.tmem_empty_mbar.data_ptr()
    tile_consumer_mbar = storage.tile_id_consumer_mbar.data_ptr()
    tile_producer_mbar = storage.tile_id_producer_mbar.data_ptr()
    tile_cta_bar_mbar = (
        storage.tile_cta_bar_mbar.data_ptr() if self.NUM_TILE_CTA_BARS > 0 else None
    )
    tmem_dealloc_mbar = (
        storage.tmem_dealloc_mbar.data_ptr() if self.NUM_CTAS == 2 else None
    )
    cross_seam_mbar = (
        storage.cross_seam_mbar.data_ptr() if self.OVERLAPPING_ACCUM else None
    )
    a_global_scale_inv_smem_ptr = (
        storage.a_global_scale_inv.data_ptr() if use_a_global_scale_inv else None
    )
    tile_id_smem_ptr = storage.tile_id_smem.data_ptr()
    tmem_holding_buf = storage.tmem_holding_buf

    # mbarrier arrive_count mirrors the base kernel: the unified TMA
    # warp arms expect_tx with the combined A+B+SF byte count, so no
    # extra arrives are needed for the SF half.
    if warp_idx == self.EPILOG_WARP_IDS[0]:
        with cute.arch.elect_one():
            for i in range(self.NUM_SMEM_BUFFERS):
                cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                cute.arch.mbarrier_init(ab_full_mbar + i, 1)
            if cutlass.const_expr(self.USE_SM103_ULTRA):
                for i in range(
                    self.SM103_SF_RING_TILES * sm103.sf_segments(self.sf_vec_size)
                ):
                    cute.arch.mbarrier_init(sf_empty_mbar + i, 1)
                    cute.arch.mbarrier_init(sf_full_mbar + i, 1)
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

    # ----- TMEM allocation (epilogue warps own it) -------------------
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

    # Prologue partitioning: A/B identical to the base kernel; SFA
    # rides the A partition (M axis), SFB rides the cta_group=ONE
    # ``tiled_mma_sfb`` so its M partition matches the SFB layout.
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

    # SFA/SFB use the same slicing pattern as A/B so we can reuse
    # ``partition_A/B`` from tiled_mma / tiled_mma_sfb.
    if cutlass.const_expr(self.USE_SM103_ULTRA):
        ab_mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            self.mma_tiler[2] // 2,
        )
        mma_sf_tiler = sm103.mma_sf_tiler(self.cta_tile_shape_mnk, self.sf_vec_size)
        gA = cute.local_tile(
            mA,
            cute.slice_(ab_mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gB = cute.local_tile(
            mB,
            cute.slice_(ab_mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gSFA = cute.local_tile(
            mSFA,
            cute.slice_(mma_sf_tiler, (None, 0, None)),
            (None, None, None),
        )
        gSFB = cute.local_tile(
            mSFB,
            cute.slice_(mma_sf_tiler, (0, None, None)),
            (None, None, None),
        )
    else:
        gA = cute.local_tile(
            mA,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gB = cute.local_tile(
            mB,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gSFA = cute.local_tile(
            mSFA,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gSFB = cute.local_tile(
            mSFB,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
    gC = cute.local_tile(
        mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
    )

    bid = cute.arch.block_idx()
    mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cluster_cta_rank)
    block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
        cluster_cta_rank
    )

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
    tCgC = thr_mma.partition_C(gC)
    if cutlass.const_expr(self.USE_SM103_ULTRA):
        tCgA_tmp = thr_mma.partition_A(gA)
        cta_tCgA = cute.make_tensor(
            tCgA_tmp.iterator,
            sm103.append_coalesce_layout(tCgA_tmp.layout),
        )
        tCgA = cute.make_tensor(
            cta_tCgA.iterator,
            cute.tiled_divide(
                cta_tCgA.layout,
                (cute.size(tiled_mma.tv_layout_A[1][0]), 128),
            ),
        )
        tCgB_tmp = thr_mma.partition_B(gB)
        cta_tCgB = cute.make_tensor(
            tCgB_tmp.iterator,
            sm103.append_coalesce_layout(tCgB_tmp.layout),
        )
        tCgB = cute.make_tensor(
            cta_tCgB.iterator,
            cute.tiled_divide(
                cta_tCgB.layout,
                (cute.size(tiled_mma.tv_layout_B[1][0]), 128),
            ),
        )
        tCgSFA = cute.make_tensor(
            gSFA.iterator,
            cute.tiled_divide(
                gSFA.layout,
                (mma_sf_tiler[0], mma_sf_tiler[2]),
            ),
        )
        tCgSFB = cute.make_tensor(
            gSFB.iterator,
            cute.tiled_divide(
                gSFB.layout,
                (mma_sf_tiler[1], mma_sf_tiler[2]),
            ),
        )
    else:
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgSFA = thr_mma.partition_A(gSFA)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)

    # TMA partitioning: A/B identical to base. SFA reuses the A
    # cta_layout; SFB needs the SFB cluster layout for the
    # cta_group=ONE mma_sfb tiler.
    a_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
    )
    if cutlass.const_expr(self.USE_SM103_ULTRA):
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 1),
        )
    else:
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
    if cutlass.const_expr(self.USE_SM103_ULTRA):
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 1),
        )
    else:
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

    sfa_cta_layout = a_cta_layout
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        block_in_cluster_coord_vmnk[2],
        sfa_cta_layout,
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    # ``filter_zeros`` collapses zero-strided modes that come from SF
    # tensors sharing A's tile layout but with a different atom (the
    # cutlass blockscaled reference does the same).
    tAsSFA = cute.filter_zeros(tAsSFA)
    if cutlass.const_expr(not self.USE_SM103_ULTRA):
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
    if cutlass.const_expr(not self.USE_SM103_ULTRA):
        tBgSFB = cute.filter_zeros(tBgSFB)

    # MMA fragments.
    if cutlass.const_expr(self.USE_SM103_ULTRA):
        tCrA = sA
        tCrB = sB
    else:
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C((self.BLOCK_SIZE_M, self.BLOCK_SIZE_N))
    tCtAcc_fake = tiled_mma.make_fragment_C(
        cute.append(acc_shape, self.NUM_TMEM_BUFFERS)
    )
    # OVERLAPPING_ACCUM packs both acc stages into the seam shared
    # with the SF block by shrinking the stage stride from BLOCK_N
    # to ``BLOCK_N - num_sf_tmem_cols``. Mirrors cutlass dense ref
    # ``dense_blockscaled_gemm_persistent.py:960-977``.
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

    # Multicast masks: A/B as in base kernel. SFA reuses A's mask
    # (same M cohort); SFB needs its own image on the SFB cluster
    # layout.
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

    (
        num_tma_load_bytes,
        num_tma_load_bytes_ab,
        num_tma_load_bytes_sf,
    ) = self._num_tma_load_bytes(
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

    # Prefetch the 5 host-built TMA descriptors into L1 before the
    # producer loops (per flash_fwd.py:1192-1196).
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
        sf_full_mbar=sf_full_mbar,
        sf_empty_mbar=sf_empty_mbar,
    )

    # =================================================================
    # Warp-group dispatch.
    # =================================================================
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
            num_tma_load_bytes_ab=num_tma_load_bytes_ab,
            num_tma_load_bytes_sf=num_tma_load_bytes_sf,
        )
        self._tma_producer_body(
            pipeline=pipeline,
            problem=problem,
            sync=sync,
            tensormap_manager=tensormap_manager,
            tensormaps=tensormaps,
            tile_producer_hook_tensor=split_sizes,
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
        tCtSFA_field = tCtSFA
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            sfa_field_layout, sfb_field_layout = sm103.make_sf_tmem_field_layouts(
                tiled_mma,
                self.mma_tiler,
                self.cta_tile_shape_mnk,
                self.sf_vec_size,
            )
            tCtSFA_field = cute.make_tensor(
                tCtSFA.iterator,
                sfa_field_layout,
            )
            tCtSFB_field = cute.make_tensor(
                tCtSFB.iterator,
                sfb_field_layout,
            )

        pipeline = GroupedGemmMmaPipeline(
            tiled_mma=tiled_mma,
            tCrA=tCrA,
            tCrB=tCrB,
            tCtAcc_base=tCtAcc_base,
            tCtSFA=tCtSFA_field,
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
            tTR_tAcc_base_00,
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
        ) = self._epilog_smem_copy_and_partition(
            tiled_copy_t2r=tiled_copy_t2r,
            tTR_rC=tTR_rC,
            tidx=tidx,
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
            tidx=tidx,
            pipeline=pipeline,
            epilog_copy=epilog_copy,
            problem=problem,
            sync=sync,
            tensormap_manager=tensormap_manager,
            tensormaps=tensormaps,
            use_device_tensormaps=use_device_tensormaps,
            a_global_scale_inv_smem_ptr=a_global_scale_inv_smem_ptr,
            a_global_scale_inv_ptr=a_global_scale_inv_ptr,
            b_global_scale_inv_ptr=b_global_scale_inv_ptr,
            use_a_global_scale_inv=use_a_global_scale_inv,
            use_b_global_scale_inv=use_b_global_scale_inv,
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

    if cutlass.const_expr(self.NUM_CTAS == 2):
        # Keep every active thread alive until peer DSMEM accesses complete.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
