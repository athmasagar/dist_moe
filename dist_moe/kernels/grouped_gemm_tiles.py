# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side per-tile MMA-consumer bodies of the native grouped GEMM and
the per-atom accumulator hand-off they share with its epilogues.

Free ``@cute.jit`` functions hoisted out of ``GroupedGemmKernel``;
compile-time attributes are passed explicitly as ``cutlass.Constexpr``
parameters (the ``blockscaled_gemm_tiles.py`` convention).
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05

from .tile_scheduler import _get_bufidx_phase


def split_atom_accum(num_tmem_buffers: int, num_mmas: int) -> bool:
    """Whether the accumulator is handed to the epilogue one M atom at a time.

    With a single accumulator buffer, the next tile's MMAs wait for the whole
    epilogue drain. When that buffer holds two M atoms, each atom gets its own
    tmem_full / tmem_empty barrier pair, so one atom drains while the other
    keeps accumulating.
    """
    return num_tmem_buffers == 1 and num_mmas == 2


@cute.jit
def native_mma_tile(
    tiled_mma: cute.TiledMma,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
    ab_full_mbar: cute.Pointer,
    ab_empty_mbar: cute.Pointer,
    tmem_full_mbar: cute.Pointer,
    tmem_empty_mbar: cute.Pointer,
    ab_empty_mcast_mask,
    acc_full_mcast_mask,
    pred_cta0: cutlass.Boolean,
    num_k_tiles: cutlass.Int32,
    tmem_buf: cutlass.Int32,
    tmem_phase: cutlass.Int32,
    accum_cnt_smem: cutlass.Int32,
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
    cta_group: cutlass.Constexpr,
) -> cutlass.Int32:
    """One tile into accumulator buffer ``tmem_buf``, handed to the epilogue
    whole on ``tmem_full_mbar + tmem_buf``. Returns the advanced SMEM stage
    counter."""
    # In 2-CTA mode, ONLY the leader CTA waits on tmem_empty
    # and ab_full. Peer CTA's MMA body must NOT wait on its
    # local bars — only leader's bars receive arrives.
    # (Matches the cute/gemm reference.)
    if (NUM_CTAS == 1) or pred_cta0:
        cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_buf, tmem_phase ^ 1)

    for kk in cutlass.range(num_k_tiles, unroll=1):
        smem_buf, smem_phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
        # Wait for TMA loads to complete (leader only in 2-CTA).
        if (NUM_CTAS == 1) or pred_cta0:
            cute.arch.mbarrier_wait(ab_full_mbar + smem_buf, smem_phase)

        # In 2-CTA mode (cta_group=TWO), only the LEADER CTA
        # issues `cute.gemm` and `tcgen05.commit`. The MMA
        # atom broadcasts to the peer CTA via the cluster
        # mcast_mask. Peer CTA just observes — its bars get
        # arrives via the multicast commit. (Matches the
        # cute/gemm reference.)
        if pred_cta0:
            tiled_mma.set(tcgen05.Field.ACCUMULATE, kk > 0)
            tCtAcc_slot = tCtAcc_base[(None, None, None, tmem_buf)]
            num_kblocks = cute.size(tCrA, mode=[2])
            for kblock_idx in cutlass.range_constexpr(num_kblocks):
                kblock_coord = (None, None, kblock_idx, smem_buf)
                cute.gemm(
                    tiled_mma,
                    tCtAcc_slot,
                    tCrA[kblock_coord],
                    tCrB[kblock_coord],
                    tCtAcc_slot,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            # Signal smem_empty: tcgen05.commit broadcasts the
            # arrive to all CTAs in the cluster via mcast_mask
            # (one arrive per CTA's bar; matches arrive_count).
            with cute.arch.elect_one():
                tcgen05.commit(
                    ab_empty_mbar + smem_buf,
                    ab_empty_mcast_mask,
                    cta_group,
                )

        accum_cnt_smem += cutlass.Int32(1)

    # ---- Signal tmem_full to epilogue -----------------------
    # Normal path (num_k_tiles > 0): leader's tcgen05.commit
    # signals tmem_full. mcast_mask broadcasts to both CTAs'
    # bars (1 arrive each). Peer issues nothing.
    #
    # Empty-contraction path (num_k_tiles == 0, e.g. wgrad
    # with k_g=0): tcgen05.commit cta_group=2 with no prior
    # tcgen05.mma was empirically traced to NOT arrive on
    # both CTAs' bars (mbarrier desync after >=2 prior
    # non-empty groups; cuda-gdb confirmed). Workaround:
    # leader issues a LOCAL arrive + REMOTE arrive on peer
    # (same arrive topology as the mcast version, leader-
    # sourced).
    #
    # CORRECTNESS INVARIANT (load-bearing): tcgen05.mma from
    # any prior non-empty group must have RETIRED before the
    # next epilog reads tmem on this same buf. The
    # `mbarrier_arrive` pair below does NOT fence — the
    # fence comes from the chain:
    #   (1) prior epilog's tcgen05.ld implicitly waited for
    #       prior tcgen05.mma to retire,
    #   (2) prior epilog signaled tmem_empty,
    #   (3) this MMA's tmem_empty wait at start of tile
    #       gates entry to this empty handler.
    # Removing the tmem_empty wait or the ld -> tmem_empty
    # ordering breaks the empty-group path. Epilog detects
    # num_k_tiles==0 and writes zeros.
    if num_k_tiles == 0:
        if cutlass.const_expr(NUM_CTAS == 2):
            if pred_cta0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_full_mbar + tmem_buf)
                    cute.arch.mbarrier_arrive(
                        tmem_full_mbar + tmem_buf,
                        peer_cta_rank_in_cluster=1,
                    )
        else:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_full_mbar + tmem_buf)
    elif cutlass.const_expr(NUM_CTAS == 2):
        if pred_cta0:
            with cute.arch.elect_one():
                tcgen05.commit(
                    tmem_full_mbar + tmem_buf,
                    acc_full_mcast_mask,
                    cta_group,
                )
    else:
        with cute.arch.elect_one():
            tcgen05.commit(
                tmem_full_mbar + tmem_buf,
                None,
                cta_group,
            )
    return accum_cnt_smem


@cute.jit
def _issue_acc_atom(
    tiled_mma: cute.TiledMma,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_slot: cute.Tensor,
    smem_buf: cutlass.Int32,
    accumulate: cutlass.Boolean,
    mma_m_idx: cutlass.Constexpr[int],
):
    """Issue one SMEM stage's k-blocks for M atom ``mma_m_idx`` only."""
    tCtAcc_atom = cute.local_tile(
        tCtAcc_slot, (tCtAcc_slot.shape[0], 1, 1), (0, mma_m_idx, 0)
    )
    # Fresh handle: the ACCUMULATE set() chain must not escape the
    # caller's dynamic loop / branch regions, which do not thread it.
    atom_mma = cute.make_tiled_mma(cute.make_mma_atom(tiled_mma.op))
    atom_mma.set(tcgen05.Field.ACCUMULATE, accumulate)
    for kblock_idx in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        tCrA_kblock = tCrA[(None, None, kblock_idx, smem_buf)]
        tCrA_atom = cute.local_tile(
            tCrA_kblock, (tCrA_kblock.shape[0], 1), (0, mma_m_idx)
        )
        cute.gemm(
            atom_mma,
            tCtAcc_atom,
            tCrA_atom,
            tCrB[(None, None, kblock_idx, smem_buf)],
            tCtAcc_atom,
        )
        atom_mma.set(tcgen05.Field.ACCUMULATE, True)


@cute.jit
def _commit(mbar: cute.Pointer, mcast_mask, cta_group: cutlass.Constexpr):
    """Arrive on ``mbar`` once this thread's issued MMAs retire; the masks
    are None in 1-CTA mode."""
    with cute.arch.elect_one():
        tcgen05.commit(mbar, mcast_mask, cta_group)


@cute.jit
def split_atom_mma_tile(  # noqa: C901
    tiled_mma: cute.TiledMma,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_slot: cute.Tensor,
    ab_full_mbar: cute.Pointer,
    ab_empty_mbar: cute.Pointer,
    tmem_full_mbar: cute.Pointer,
    tmem_empty_mbar: cute.Pointer,
    ab_empty_mcast_mask,
    acc_full_mcast_mask,
    pred_cta0: cutlass.Boolean,
    num_k_tiles: cutlass.Int32,
    tmem_phase: cutlass.Int32,
    accum_cnt_smem: cutlass.Int32,
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
    cta_group: cutlass.Constexpr,
) -> cutlass.Int32:
    """One tile on a single two-atom accumulator buffer, each M atom handed
    off on its own ``tmem_full_mbar + atom`` / ``tmem_empty_mbar + atom``.

    The first ``lead`` k-tiles issue atom 0 only, holding their SMEM stages,
    as soon as the epilogue frees atom 0. Atom 1 catches up on the held
    stages once the epilogue frees it. The last ``tail`` k-tiles mirror this,
    so atom 0 reaches the epilogue while atom 1 still accumulates. Each window
    holds at most NUM_SMEM_BUFFERS stages, so the TMA producer cannot
    deadlock. Returns the advanced SMEM stage counter.
    """
    lead = cutlass.min(num_k_tiles, cutlass.Int32(NUM_SMEM_BUFFERS))
    tail = cutlass.min(num_k_tiles - lead, cutlass.Int32(NUM_SMEM_BUFFERS))
    body_end = num_k_tiles - tail
    if pred_cta0:
        cute.arch.mbarrier_wait(tmem_empty_mbar, tmem_phase ^ 1)
        if num_k_tiles == 0:
            cute.arch.mbarrier_wait(tmem_empty_mbar + 1, tmem_phase ^ 1)
            # Leader-sourced arrives, as in `native_mma_tile`'s
            # empty-contraction path.
            with cute.arch.elect_one():
                for atom in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_arrive(tmem_full_mbar + atom)
                    if cutlass.const_expr(NUM_CTAS == 2):
                        cute.arch.mbarrier_arrive(
                            tmem_full_mbar + atom,
                            peer_cta_rank_in_cluster=1,
                        )
        else:
            for kk in cutlass.range(lead, unroll=1):
                smem_buf, smem_phase = _get_bufidx_phase(
                    accum_cnt_smem + kk, NUM_SMEM_BUFFERS
                )
                cute.arch.mbarrier_wait(ab_full_mbar + smem_buf, smem_phase)
                _issue_acc_atom(tiled_mma, tCrA, tCrB, tCtAcc_slot, smem_buf, kk > 0, 0)
            if tail == 0:
                _commit(tmem_full_mbar, acc_full_mcast_mask, cta_group)
            cute.arch.mbarrier_wait(tmem_empty_mbar + 1, tmem_phase ^ 1)
            for kk in cutlass.range(lead, unroll=1):
                smem_buf, _ = _get_bufidx_phase(accum_cnt_smem + kk, NUM_SMEM_BUFFERS)
                _issue_acc_atom(tiled_mma, tCrA, tCrB, tCtAcc_slot, smem_buf, kk > 0, 1)
                _commit(ab_empty_mbar + smem_buf, ab_empty_mcast_mask, cta_group)
            for kk in cutlass.range(lead, body_end, unroll=1):
                smem_buf, smem_phase = _get_bufidx_phase(
                    accum_cnt_smem + kk, NUM_SMEM_BUFFERS
                )
                cute.arch.mbarrier_wait(ab_full_mbar + smem_buf, smem_phase)
                _issue_acc_atom(tiled_mma, tCrA, tCrB, tCtAcc_slot, smem_buf, True, 0)
                _issue_acc_atom(tiled_mma, tCrA, tCrB, tCtAcc_slot, smem_buf, True, 1)
                _commit(ab_empty_mbar + smem_buf, ab_empty_mcast_mask, cta_group)
            for kk in cutlass.range(body_end, num_k_tiles, unroll=1):
                smem_buf, smem_phase = _get_bufidx_phase(
                    accum_cnt_smem + kk, NUM_SMEM_BUFFERS
                )
                cute.arch.mbarrier_wait(ab_full_mbar + smem_buf, smem_phase)
                _issue_acc_atom(tiled_mma, tCrA, tCrB, tCtAcc_slot, smem_buf, True, 0)
            if tail > 0:
                _commit(tmem_full_mbar, acc_full_mcast_mask, cta_group)
            for kk in cutlass.range(body_end, num_k_tiles, unroll=1):
                smem_buf, _ = _get_bufidx_phase(accum_cnt_smem + kk, NUM_SMEM_BUFFERS)
                _issue_acc_atom(tiled_mma, tCrA, tCrB, tCtAcc_slot, smem_buf, True, 1)
                _commit(ab_empty_mbar + smem_buf, ab_empty_mcast_mask, cta_group)
            _commit(tmem_full_mbar + 1, acc_full_mcast_mask, cta_group)
    return accum_cnt_smem + num_k_tiles


@cute.jit
def release_acc(
    tmem_empty_slot: cute.Pointer,
    leader_warp: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    barrier_id: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
):
    """Free an accumulator slot (a whole buffer, or one M atom under
    split-atom hand-off) to the MMA warp once every epilogue warp's TMEM
    loads of it have retired.

    The ``tcgen05.wait::ld`` fence is per-warp; the named barrier is what
    makes the leader's arrive cover warps with loads still in flight. SMEM ->
    GMEM stores are deliberately not waited on: they gate C SMEM reuse, not
    TMEM reuse. In 2-CTA both CTAs arrive on the leader's barrier
    (arrive_count=2), so its next MMA cannot clobber cluster-shared TMEM the
    peer still reads.
    """
    cute.arch.fence_view_async_tmem_load()
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == leader_warp:
        with cute.arch.elect_one():
            if cutlass.const_expr(NUM_CTAS == 2):
                cute.arch.mbarrier_arrive(tmem_empty_slot, peer_cta_rank_in_cluster=0)
            else:
                cute.arch.mbarrier_arrive(tmem_empty_slot)
