# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Per-group tensormap preparation folded into the chunked Mega kernel."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .activation_buffer import MISSING_ACTIVATION_OFFSET
from .blockscaled_gemm_tiles import (
    _prepare_blockscaled_problem_tensormaps,
)
from .dispatch_quant import _wait_counter_at_least
from .params import BlockscaledTensormapProblem

FOLDED_TENSORMAP_SMEM_INT64 = 5 * 16
# counter[0] is the dynamic tile counter; counter[1] counts published problems.
_READY_COUNTER_OFFSET = 1


def staged_metadata_int64(
    groups: int, work_info_fields: int, num_tile_buffers: int
) -> int:
    """Int64 words of the per-group sizes and tile work info staged in SMEM."""
    return (groups + work_info_fields * num_tile_buffers + 1) // 2


def folded_tensormap_smem_offset(
    groups: int, work_info_fields: int, num_tile_buffers: int
) -> int:
    """Int64 offset of the descriptor scratch past the staged expert metadata."""
    metadata_int64 = staged_metadata_int64(groups, work_info_fields, num_tile_buffers)
    return (metadata_int64 + 15) // 16 * 16


@cute.jit
def build_folded_tensormaps(
    params_pair,
    split_sizes: cute.Tensor,
    tensormaps: cute.Tensor,
    counter_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    gemm_mnk_pair,
    problem_type_pair: cutlass.Constexpr,
    tensormap_base_pair: cutlass.Constexpr,
    G: cutlass.Constexpr[int],
    builder_warp: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
    sf_dtype: cutlass.Constexpr,
    sf_vec_size: cutlass.Constexpr[int],
    use_sm103_ultra: cutlass.Constexpr[bool],
) -> None:
    """Build the (group, gemm) problems grid-strided over CTAs; one warp per CTA."""
    tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)
    bid, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    ready_ptr = cute.recast_ptr(
        counter_ptr + _READY_COUNTER_OFFSET, dtype=cutlass.Uint32
    )
    missing_offsets = (
        cutlass.Int64(MISSING_ACTIVATION_OFFSET),
        cutlass.Int64(MISSING_ACTIVATION_OFFSET),
        cutlass.Int64(MISSING_ACTIVATION_OFFSET),
        cutlass.Int64(MISSING_ACTIVATION_OFFSET),
        cutlass.Int64(MISSING_ACTIVATION_OFFSET),
    )
    slot = cutlass.Int32(bid)
    while slot < cutlass.Int32(2 * G):
        g = slot % cutlass.Int32(G)
        gemm = slot // cutlass.Int32(G)
        for w in cutlass.range_constexpr(2):
            if gemm == cutlass.Int32(w):
                prob_m, prob_n, prob_k = gemm_mnk_pair[w]
                problem = BlockscaledTensormapProblem(
                    split_sizes=split_sizes,
                    groups=G,
                    mnk=(prob_m, prob_n, prob_k),
                    problem_type=problem_type_pair[w],
                    tensormap_base=tensormap_base_pair[w],
                    prepare_g=g,
                    warp_idx=cutlass.Int32(0),
                )
                _prepare_blockscaled_problem_tensormaps(
                    params=params_pair[w],
                    problem=problem,
                    tensormaps=tensormaps,
                    tensormap_manager=tensormap_manager,
                    tensormap_smem_ptr_base=smem_ptr,
                    activation_buffer_operand_offsets=missing_offsets,
                    activation_buffer_size_bytes=cutlass.Int64(0),
                    use_activation_buffer=False,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    SWAP_AB=SWAP_AB,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    use_sm103_ultra=use_sm103_ultra,
                    tensormap_warp_id=builder_warp,
                )
        cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_warp()
        if cute.arch.lane_idx() == cutlass.Int32(0):
            cute.arch.atomic_add(
                ready_ptr, cutlass.Uint32(1), sem="release", scope="gpu"
            )
        slot += cutlass.Int32(grid)


@cute.jit
def wait_folded_tensormaps(counter_ptr: cute.Pointer, G: cutlass.Constexpr[int]):
    _wait_counter_at_least(
        cute.recast_ptr(counter_ptr + _READY_COUNTER_OFFSET, dtype=cutlass.Uint32),
        cutlass.Uint32(2 * G),
    )
