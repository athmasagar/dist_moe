# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native (BF16/FP16) SwiGLU-backward producer for the fused DGRAD COMBINE.

Warps 8-11 of a ``COMBINE_SWIGLU_BWD`` dist kernel compute
``grad_h1 = swiglu_bwd(grad_h2, h1)`` in ``ROWS``-row items and release each
activation M tile to the GEMM's TMA producer through ``a_buff_counter`` once
all ``BLOCK_SIZE_M // ROWS`` items of that tile have landed. With ``emit_h2``
each lane also overwrites the grad_h2 words it consumed with
``h2 = swiglu_fwd(h1)``, bitwise equal to the Triton ``swiglu_fwd``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from ._quant_conversion import (
    _cvt_f16x2_to_f32x2,
    _cvt_f32x2_to_f16x2_rn,
)
from .blockscaled_quantization_common import (
    _apply_block_scaled_quant_dxy_fwd_producer_f32x2,
    _apply_block_scaled_quant_dxy_producer_f32x2,
    _mul_f32x2,
    _swiglu_bwd_dxy_from_sigmoid_f32x2,
)
from .combine_swiglu_quant import (
    _combine_swiglu_quant_fetch_warp_tile_batch,
    _combine_swiglu_quant_signal_warp_tile_done,
)
from .grouped_gemm_kernel import _remap_m_tile_idx
from .params import ceil_div
from .swiglu_fragment import _sigmoid_rn_f32_batch

# One 16-byte vector of b16 per lane; a warp pass covers 256 columns.
NATIVE_SWIGLU_BWD_COLS_PER_LANE: int = 8
NATIVE_SWIGLU_BWD_COLS_PER_PASS: int = 32 * NATIVE_SWIGLU_BWD_COLS_PER_LANE
NATIVE_SWIGLU_BWD_PASSES_PER_BATCH: int = 4
NATIVE_SWIGLU_BWD_ROWS: int = 1


def native_swiglu_bwd_col_multiple() -> int:
    """Intermediate-dim multiple the producer's unmasked column loop needs."""
    return NATIVE_SWIGLU_BWD_COLS_PER_PASS


@cute.jit
def _b16x8_gmem(addr: cutlass.Int64) -> cute.Tensor:
    return cute.make_tensor(
        cute.make_ptr(cutlass.Uint32, addr, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout(4),
    )


@cute.jit
def _native_swiglu_bwd_words(
    dz_w: cute.Tensor,
    x_w: cute.Tensor,
    y_w: cute.Tensor,
    dx_r: cute.Tensor,
    dy_r: cute.Tensor,
    h2_r: cute.Tensor,
    dtype: cutlass.Constexpr,
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    emit_h2: cutlass.Constexpr[bool],
) -> None:
    """SwiGLU-bwd (and, with ``emit_h2``, -fwd) of one lane's four packed
    b16x2 words."""
    if cutlass.const_expr(fast_math or clamped):
        for w in cutlass.range_constexpr(4):
            dz2 = _cvt_f16x2_to_f32x2(dz_w[w], dtype)
            x2 = _cvt_f16x2_to_f32x2(x_w[w], dtype)
            y2 = _cvt_f16x2_to_f32x2(y_w[w], dtype)
            if cutlass.const_expr(emit_h2):
                dx2, dy2, h22 = _apply_block_scaled_quant_dxy_fwd_producer_f32x2(
                    dz2, x2, y2, fast_math, clamped, alpha, limit
                )
                h2_r[w] = _cvt_f32x2_to_f16x2_rn(h22[0], h22[1], dtype)
            else:
                dx2, dy2 = _apply_block_scaled_quant_dxy_producer_f32x2(
                    dz2, x2, y2, fast_math, clamped, alpha, limit
                )
            dx_r[w] = _cvt_f32x2_to_f16x2_rn(dx2[0], dx2[1], dtype)
            dy_r[w] = _cvt_f32x2_to_f16x2_rn(dy2[0], dy2[1], dtype)
    else:
        x_f = cute.make_rmem_tensor(8, Float32)
        sig = cute.make_rmem_tensor(8, Float32)
        for w in cutlass.range_constexpr(4):
            x0, x1 = _cvt_f16x2_to_f32x2(x_w[w], dtype)
            x_f[2 * w] = x0
            x_f[2 * w + 1] = x1
        _sigmoid_rn_f32_batch(x_f, sig, 8)
        for w in cutlass.range_constexpr(4):
            y2 = _cvt_f16x2_to_f32x2(y_w[w], dtype)
            dx2, dy2, x_sig2 = _swiglu_bwd_dxy_from_sigmoid_f32x2(
                _cvt_f16x2_to_f32x2(dz_w[w], dtype),
                (x_f[2 * w], x_f[2 * w + 1]),
                y2,
                (sig[2 * w], sig[2 * w + 1]),
            )
            dx_r[w] = _cvt_f32x2_to_f16x2_rn(dx2[0], dx2[1], dtype)
            dy_r[w] = _cvt_f32x2_to_f16x2_rn(dy2[0], dy2[1], dtype)
            if cutlass.const_expr(emit_h2):
                h22 = _mul_f32x2(x_sig2, y2)
                h2_r[w] = _cvt_f32x2_to_f16x2_rn(h22[0], h22[1], dtype)


@cute.jit
def _native_swiglu_bwd_passes(
    dz_row: cutlass.Int64,
    h1_row: cutlass.Int64,
    dxy_row: cutlass.Int64,
    half_bytes: cutlass.Int64,
    col_off: cutlass.Int64,
    NUM_PASSES: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    emit_h2: cutlass.Constexpr[bool],
) -> None:
    """``NUM_PASSES`` consecutive passes starting ``col_off`` bytes into the
    row: every load is issued before the first compute."""
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=128
    )
    pass_bytes: cutlass.Constexpr[int] = NATIVE_SWIGLU_BWD_COLS_PER_PASS * (
        dtype.width // 8
    )
    dz_v = []
    x_v = []
    y_v = []
    for u in cutlass.range_constexpr(NUM_PASSES):
        off = col_off + cutlass.Int64(u * pass_bytes)
        dz_r = cute.make_rmem_tensor(4, cutlass.Uint32)
        x_r = cute.make_rmem_tensor(4, cutlass.Uint32)
        y_r = cute.make_rmem_tensor(4, cutlass.Uint32)
        cute.copy(copy_atom, _b16x8_gmem(dz_row + off), dz_r)
        cute.copy(copy_atom, _b16x8_gmem(h1_row + off), x_r)
        cute.copy(copy_atom, _b16x8_gmem(h1_row + half_bytes + off), y_r)
        dz_v.append(dz_r)
        x_v.append(x_r)
        y_v.append(y_r)
    for u in cutlass.range_constexpr(NUM_PASSES):
        off = col_off + cutlass.Int64(u * pass_bytes)
        dx_r = cute.make_rmem_tensor(4, cutlass.Uint32)
        dy_r = cute.make_rmem_tensor(4, cutlass.Uint32)
        h2_r = cute.make_rmem_tensor(4, cutlass.Uint32)
        _native_swiglu_bwd_words(
            dz_v[u],
            x_v[u],
            y_v[u],
            dx_r,
            dy_r,
            h2_r,
            dtype,
            fast_math,
            clamped,
            alpha,
            limit,
            emit_h2,
        )
        cute.copy(copy_atom, dx_r, _b16x8_gmem(dxy_row + off))
        cute.copy(copy_atom, dy_r, _b16x8_gmem(dxy_row + half_bytes + off))
        if cutlass.const_expr(emit_h2):
            cute.copy(copy_atom, h2_r, _b16x8_gmem(dz_row + off))


@cute.jit
def _native_swiglu_bwd_row(
    dz_row: cutlass.Int64,
    h1_row: cutlass.Int64,
    dxy_row: cutlass.Int64,
    half_bytes: cutlass.Int64,
    batches: cutlass.Int32,
    tail_passes: cutlass.Int32,
    dtype: cutlass.Constexpr,
    PASSES_PER_BATCH: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    emit_h2: cutlass.Constexpr[bool],
) -> None:
    """One lane's share of a row; `*_row` already include the lane column."""
    pass_bytes: cutlass.Constexpr[int] = NATIVE_SWIGLU_BWD_COLS_PER_PASS * (
        dtype.width // 8
    )
    batch_bytes: cutlass.Constexpr[int] = pass_bytes * PASSES_PER_BATCH
    for b in cutlass.range(batches, unroll=1):
        _native_swiglu_bwd_passes(
            dz_row,
            h1_row,
            dxy_row,
            half_bytes,
            cutlass.Int64(b) * cutlass.Int64(batch_bytes),
            PASSES_PER_BATCH,
            dtype,
            fast_math,
            clamped,
            alpha,
            limit,
            emit_h2,
        )
    tail_off = cutlass.Int64(batches) * cutlass.Int64(batch_bytes)
    for t in cutlass.range(tail_passes, unroll=1):
        _native_swiglu_bwd_passes(
            dz_row,
            h1_row,
            dxy_row,
            half_bytes,
            tail_off + cutlass.Int64(t) * cutlass.Int64(pass_bytes),
            1,
            dtype,
            fast_math,
            clamped,
            alpha,
            limit,
            emit_h2,
        )


@cute.jit
def native_swiglu_bwd_producer_body(
    mWorkCounter: cute.Tensor,
    mDoneCounter: cute.Tensor,
    split_sizes: cute.Tensor,
    dz_base_ptr: cutlass.Int64,
    dz_s0: cutlass.Int32,
    h1_base_ptr: cutlass.Int64,
    h1_s0: cutlass.Int32,
    dxy_base_ptr: cutlass.Int64,
    dxy_s0: cutlass.Int32,
    K: cutlass.Int32,
    local_rank: cutlass.Int32,
    producer_tidx: cutlass.Int32,
    dtype: cutlass.Constexpr,
    G: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    ROWS: cutlass.Constexpr[int],
    PASSES_PER_BATCH: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    emit_h2: cutlass.Constexpr[bool] = False,
) -> None:
    """Per-warp item loop. Items walk (group, rotated M tile, row chunk) in
    the GEMM's tile order; every M tile, including a group's partial tail,
    receives exactly ``BLOCK_SIZE_M // ROWS`` done signals."""
    if cutlass.const_expr(BLOCK_SIZE_M % ROWS != 0):
        raise ValueError("ROWS must divide BLOCK_SIZE_M")
    ITEMS_PER_TILE: cutlass.Constexpr[int] = BLOCK_SIZE_M // ROWS
    elem_bytes: cutlass.Constexpr[int] = dtype.width // 8
    lane = producer_tidx % cutlass.Int32(32)
    intermediate = K // cutlass.Int32(2)
    half_bytes = cutlass.Int64(intermediate) * cutlass.Int64(elem_bytes)
    passes = intermediate // cutlass.Int32(NATIVE_SWIGLU_BWD_COLS_PER_PASS)
    batches = passes // cutlass.Int32(PASSES_PER_BATCH)
    tail_passes = passes - batches * cutlass.Int32(PASSES_PER_BATCH)
    lane_bytes = cutlass.Int64(lane * NATIVE_SWIGLU_BWD_COLS_PER_LANE * elem_bytes)

    item = _combine_swiglu_quant_fetch_warp_tile_batch(mWorkCounter, lane, 1)
    item_start = cutlass.Int32(0)
    act_tile_start = cutlass.Int32(0)
    start_m = cutlass.Int32(0)
    for g in cutlass.range(G, unroll=1):
        m_size = cutlass.Int32(split_sizes[g])
        group_act_tiles = ceil_div(m_size, cutlass.Int32(BLOCK_SIZE_M))
        item_end = item_start + group_act_tiles * cutlass.Int32(ITEMS_PER_TILE)
        while (item >= item_start) and (item < item_end):
            local = item - item_start
            act_tile = local // cutlass.Int32(ITEMS_PER_TILE)
            chunk = local - act_tile * cutlass.Int32(ITEMS_PER_TILE)
            act_tile = _remap_m_tile_idx(
                act_tile, group_act_tiles, local_rank, WORLD_SIZE
            )
            row0 = act_tile * cutlass.Int32(BLOCK_SIZE_M) + chunk * cutlass.Int32(ROWS)
            for r in cutlass.range_constexpr(ROWS):
                if row0 + cutlass.Int32(r) < m_size:
                    grow = cutlass.Int64(start_m + row0 + cutlass.Int32(r))
                    _native_swiglu_bwd_row(
                        dz_base_ptr
                        + grow * cutlass.Int64(dz_s0) * cutlass.Int64(elem_bytes)
                        + lane_bytes,
                        h1_base_ptr
                        + grow * cutlass.Int64(h1_s0) * cutlass.Int64(elem_bytes)
                        + lane_bytes,
                        dxy_base_ptr
                        + grow * cutlass.Int64(dxy_s0) * cutlass.Int64(elem_bytes)
                        + lane_bytes,
                        half_bytes,
                        batches,
                        tail_passes,
                        dtype,
                        PASSES_PER_BATCH,
                        fast_math,
                        clamped,
                        alpha,
                        limit,
                        emit_h2,
                    )
            _combine_swiglu_quant_signal_warp_tile_done(
                mDoneCounter, act_tile_start + act_tile, lane
            )
            item = _combine_swiglu_quant_fetch_warp_tile_batch(mWorkCounter, lane, 1)
        start_m += m_size
        act_tile_start += group_act_tiles
        item_start = item_end
