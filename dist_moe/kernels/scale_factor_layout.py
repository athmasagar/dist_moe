# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL kernels for block-scaled scale-factor layouts."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from ._cuda_context import ensure_cuda_driver_context
from ._jit_cache import jit_cache
from ._quack_compile_utils import make_fake_stream, make_fake_tensor as fake_tensor

_SUPPORTED_SCALE_DTYPES = {
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e8m0fnu,
}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _can_use_u64_loads(scale: torch.Tensor, cols: int) -> bool:
    return (
        cols % 8 == 0
        and scale.stride(-1) == 1
        and scale.data_ptr() % 8 == 0
        and all(stride % 8 == 0 for stride in scale.stride()[:-1])
    )


class _NaturalToCublasBlocked:
    def __init__(
        self,
        rows_per_group: int,
        cols: int,
        has_groups: bool,
        has_offsets: bool,
        use_u64: bool,
    ):
        self.rows_per_group = rows_per_group
        self.cols = cols
        self.has_groups = has_groups
        self.has_offsets = has_offsets
        self.use_u64 = use_u64
        self.n_col_blocks = _ceil_div(cols, 4)
        self.cols_qwords = cols // 8
        self.row_blocks_per_group = _ceil_div(rows_per_group, 128)
        self.col_atom_tile = 8
        self.num_threads = 256

    @cute.jit
    def __call__(
        self,
        mScaleBytes: cute.Tensor,
        mScaleQwords: cute.Tensor,
        mOffsets: cute.Tensor,
        mOutBytes: cute.Tensor,
        mOutWords: cute.Tensor,
        mOutQwords: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mScaleBytes.element_type == cutlass.Uint8
        assert mScaleQwords.element_type == cutlass.Uint64
        assert mOffsets.element_type == cutlass.Int32
        assert mOutBytes.element_type == cutlass.Uint8
        assert mOutWords.element_type == cutlass.Uint32
        assert mOutQwords.element_type == cutlass.Uint64
        if cutlass.const_expr(self.has_offsets):
            # Jagged SFA launch: one CTA per group, 128-row block, col tile.
            self.kernel(
                mScaleBytes,
                mScaleQwords,
                mOffsets,
                mOutBytes,
                mOutWords,
                mOutQwords,
            ).launch(
                grid=[
                    mOffsets.shape[0],
                    self.row_blocks_per_group,
                    cute.ceil_div(self.n_col_blocks, self.col_atom_tile),
                ],
                block=[self.num_threads, 1, 1],
                stream=stream,
            )
        else:
            # Dense SFA/SFB launch: flatten groups into the CTA x dimension.
            groups = cutlass.Int64(1)
            if cutlass.const_expr(self.has_groups):
                groups = mScaleBytes.shape[0]
            self.kernel(
                mScaleBytes,
                mScaleQwords,
                mOffsets,
                mOutBytes,
                mOutWords,
                mOutQwords,
            ).launch(
                grid=[
                    groups * self.row_blocks_per_group,
                    cute.ceil_div(self.n_col_blocks, self.col_atom_tile),
                    1,
                ],
                block=[self.num_threads, 1, 1],
                stream=stream,
            )

    @cute.kernel
    def kernel(  # noqa: C901 -- constexpr branches specialize to one hot path.
        self,
        mScaleBytes: cute.Tensor,
        mScaleQwords: cute.Tensor,
        mOffsets: cute.Tensor,
        mOutBytes: cute.Tensor,
        mOutWords: cute.Tensor,
        mOutQwords: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        # Initialize logical group/row/col coordinates for the dense case.
        # The offsets branch below rewrites them for jagged grouped SFA.
        group = cutlass.Int64(0)
        row_block = bidx
        col_tile_id = bidy
        group_start = cutlass.Int64(0)
        group_rows = cutlass.Int64(self.rows_per_group)
        output_group_start = cutlass.Int64(0)

        if cutlass.const_expr(self.has_offsets):
            # Jagged SFA grid directly exposes group, row block, and col tile.
            group = cutlass.Int64(bidx)
            row_block = bidy
            col_tile_id = bidz

            # Offsets are cumulative natural-row ends for each M group.
            if group > 0:
                group_start = cutlass.Int64(mOffsets[group - 1])
            group_end = cutlass.Int64(mOffsets[group])
            group_rows = group_end - group_start

            # Compute this group's output row base after per-group 128-row pads.
            scan = cutlass.Int32(0)
            prev_end = cutlass.Int64(0)
            # TODO(shikaili): Pass precomputed output starts for large-G grouped
            # MoE to avoid this O(G) prefix scan in every CTA.
            while scan < group:
                end = cutlass.Int64(mOffsets[scan])
                scan_rows = end - prev_end
                output_group_start += ((scan_rows + 127) // 128) * 128
                prev_end = end
                scan += 1
        else:
            # Dense SFA has one group; dense SFB maps flattened CTAs to groups.
            if cutlass.const_expr(self.has_groups):
                group = bidx // self.row_blocks_per_group
                row_block = bidx - group * self.row_blocks_per_group
            output_group_start = cutlass.Int64(group) * self.row_blocks_per_group * 128

        # Select the vectorized or scalar-byte movement path at compile time.
        if cutlass.const_expr(self.use_u64):
            self._kernel_u64(
                mScaleQwords,
                mOutWords,
                mOutQwords,
                tidx,
                group,
                row_block,
                col_tile_id,
                group_start,
                group_rows,
                output_group_start,
            )
        else:
            self._kernel_bytes(
                mScaleBytes,
                mOutBytes,
                tidx,
                group,
                row_block,
                col_tile_id,
                group_start,
                group_rows,
                output_group_start,
            )

    @cute.jit
    def _kernel_u64(
        self,
        mScaleQwords: cute.Tensor,
        mOutWords: cute.Tensor,
        mOutQwords: cute.Tensor,
        tidx,
        group,
        row_block,
        col_tile_id,
        group_start,
        group_rows,
        output_group_start,
    ):
        if cutlass.const_expr(self.has_offsets):
            col_pair_tile: cutlass.Constexpr[int] = self.col_atom_tile // 2
            for iter_idx in cutlass.range_constexpr(
                (128 * col_pair_tile) // self.num_threads
            ):
                # Each thread moves one 8-byte pair of adjacent 4-column atoms.
                offset = iter_idx * self.num_threads + tidx
                row_in_128 = offset // col_pair_tile
                col_pair = col_tile_id * col_pair_tile + offset % col_pair_tile
                row_in_group = row_block * 128 + row_in_128

                # Out-of-bounds rows/cols become explicit zero-filled padding.
                packed = cutlass.Uint64(0)
                if (row_in_group < group_rows) and (col_pair < self.cols_qwords):
                    packed = mScaleQwords[group_start + row_in_group, col_pair]

                # Split the uint64 into two uint32 rows in cuBLAS atom order.
                col_block = col_pair * 2
                row_mod_32 = row_in_128 % 32
                row_quad = row_in_128 // 32
                atom = row_block * self.n_col_blocks + col_block
                dst_byte = (
                    output_group_start * self.n_col_blocks * 4
                    + atom * 512
                    + row_mod_32 * 16
                    + row_quad * 4
                )
                if (row_block * 128 < group_rows) and (col_pair < self.cols_qwords):
                    # hi is one 128-word atom row after lo in blocked u32 order.
                    word_offset = dst_byte // 4
                    mOutWords[word_offset] = cutlass.Uint32(
                        packed & cutlass.Uint64(0xFFFFFFFF)
                    )
                    mOutWords[word_offset + 128] = cutlass.Uint32(packed >> 32)
        else:
            # Dense u64 path stages natural rows, then repacks for coalesced stores.
            smem = cutlass.utils.SmemAllocator()
            sTile = smem.allocate_tensor(
                cutlass.Uint32,
                cute.make_layout((128, self.col_atom_tile)),
                byte_alignment=128,
            )

            col_pair_tile: cutlass.Constexpr[int] = self.col_atom_tile // 2
            for iter_idx in cutlass.range_constexpr(
                (64 * self.col_atom_tile) // self.num_threads
            ):
                # Load one uint64 and stage its two 4-column atoms as uint32s.
                offset = iter_idx * self.num_threads + tidx
                row_in_128 = offset // col_pair_tile
                col_pair_in_tile = offset - row_in_128 * col_pair_tile
                col_pair = col_tile_id * col_pair_tile + col_pair_in_tile
                packed = cutlass.Uint64(0)
                if col_pair < self.cols_qwords:
                    if cutlass.const_expr(self.has_groups):
                        packed = mScaleQwords[
                            group, row_block * 128 + row_in_128, col_pair
                        ]
                    else:
                        packed = mScaleQwords[row_block * 128 + row_in_128, col_pair]
                    sTile[row_in_128, col_pair_in_tile * 2] = cutlass.Uint32(
                        packed & cutlass.Uint64(0xFFFFFFFF)
                    )
                    sTile[row_in_128, col_pair_in_tile * 2 + 1] = cutlass.Uint32(
                        packed >> 32
                    )

            # All staged rows must be visible before threads repack output order.
            cute.arch.barrier()

            for iter_idx in cutlass.range_constexpr(
                (64 * self.col_atom_tile) // self.num_threads
            ):
                # Repack two 32-bit quadrant rows into one contiguous uint64 store.
                offset = iter_idx * self.num_threads + tidx
                col_in_tile = offset // 64
                qword_in_atom = offset - col_in_tile * 64
                col_block = col_tile_id * self.col_atom_tile + col_in_tile
                row_mod_32 = qword_in_atom // 2
                row_quad_lo = (qword_in_atom - row_mod_32 * 2) * 2
                row_lo = row_quad_lo * 32 + row_mod_32
                row_hi = row_lo + 32
                atom = row_block * self.n_col_blocks + col_block
                dst_byte = (
                    output_group_start * self.n_col_blocks * 4
                    + atom * 512
                    + qword_in_atom * 8
                )
                if col_block < self.n_col_blocks:
                    # sTile is indexed by natural row; qword_in_atom encodes SF order.
                    lo = cutlass.Uint64(sTile[row_lo, col_in_tile])
                    hi = cutlass.Uint64(sTile[row_hi, col_in_tile])
                    packed = lo | (hi << 32)
                    mOutQwords[dst_byte // 8] = packed

    @cute.jit
    def _kernel_bytes(
        self,
        mScaleBytes: cute.Tensor,
        mOutBytes: cute.Tensor,
        tidx,
        group,
        row_block,
        col_tile_id,
        group_start,
        group_rows,
        output_group_start,
    ):
        tile_cols: cutlass.Constexpr[int] = self.col_atom_tile * 4
        for iter_idx in cutlass.range_constexpr((128 * tile_cols) // self.num_threads):
            # Scalar fallback maps one thread lane to one output byte.
            offset = iter_idx * self.num_threads + tidx
            row_in_128 = offset // tile_cols
            col_in_tile = offset - row_in_128 * tile_cols
            col_block = col_tile_id * self.col_atom_tile + col_in_tile // 4
            col_in_4 = col_in_tile - (col_in_tile // 4) * 4
            row_in_group = row_block * 128 + row_in_128
            col = col_block * 4 + col_in_4

            value = cutlass.Uint8(0)
            # Load the natural byte when in bounds; otherwise keep padding zero.
            if (row_in_group < group_rows) and (col < self.cols):
                if cutlass.const_expr(self.has_groups and not self.has_offsets):
                    value = mScaleBytes[group, row_in_group, col]
                else:
                    value = mScaleBytes[group_start + row_in_group, col]

            if (row_block * 128 < group_rows) and (col_block < self.n_col_blocks):
                # Convert natural [row, col] to the cuBLAS 128x4 atom byte offset.
                atom = row_block * self.n_col_blocks + col_block
                dst_byte = (
                    output_group_start * self.n_col_blocks * 4
                    + atom * 512
                    + (row_in_128 % 32) * 16
                    + (row_in_128 // 32) * 4
                    + col_in_4
                )
                out_cols = mOutBytes.shape[1]
                # mOutBytes is 2D for dtype preservation; dst_byte is flat layout.
                out_row = dst_byte // out_cols
                out_col = dst_byte - out_row * out_cols
                mOutBytes[out_row, out_col] = value


@jit_cache
def _compile_natural_to_cublas_blocked(
    rows_per_group: int,
    cols: int,
    has_groups: bool,
    has_offsets: bool,
    use_u64: bool,
):
    group = cute.sym_int()
    offsets_group = cute.sym_int()
    rows = cute.sym_int()
    out_rows = cute.sym_int()
    out_cols = cute.sym_int()
    out_word_elems = cute.sym_int()
    out_qword_elems = cute.sym_int()
    div = 8 if use_u64 else 1
    if has_offsets:
        scale_cute = fake_tensor(cutlass.Uint8, (rows, cols), div)
    elif has_groups:
        scale_cute = fake_tensor(cutlass.Uint8, (group, rows_per_group, cols), div)
    else:
        scale_cute = fake_tensor(cutlass.Uint8, (rows_per_group, cols), div)
    if use_u64:
        if has_offsets:
            scale_qwords_cute = fake_tensor(cutlass.Uint64, (rows, cols // 8), 1)
        elif has_groups:
            scale_qwords_cute = fake_tensor(
                cutlass.Uint64,
                (group, rows_per_group, cols // 8),
                1,
            )
        else:
            scale_qwords_cute = fake_tensor(
                cutlass.Uint64,
                (rows_per_group, cols // 8),
                1,
            )
    else:
        scale_qwords_cute = fake_tensor(cutlass.Uint64, (1,), 1)
    offsets_cute = fake_tensor(cutlass.Int32, (offsets_group,), 1)
    out_cute = fake_tensor(cutlass.Uint8, (out_rows, out_cols), div)
    out_words_cute = fake_tensor(cutlass.Uint32, (out_word_elems,), 1)
    out_qwords_cute = fake_tensor(cutlass.Uint64, (out_qword_elems,), 1)
    return cute.compile(
        _NaturalToCublasBlocked(
            rows_per_group,
            cols,
            has_groups,
            has_offsets,
            use_u64,
        ),
        scale_cute,
        scale_qwords_cute,
        offsets_cute,
        out_cute,
        out_words_cute,
        out_qwords_cute,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def natural_to_cublas_blocked(
    scale: torch.Tensor,
    *,
    is_a: bool,
) -> torch.Tensor:
    if scale.dtype not in _SUPPORTED_SCALE_DTYPES:
        raise TypeError(f"Unsupported scale dtype for CuTe SF layout: {scale.dtype}")
    if scale.ndim == 2:
        rows_per_group, cols = scale.shape
        groups = 1
        has_groups = False
    elif scale.ndim == 3 and not is_a:
        groups, rows_per_group, cols = scale.shape
        has_groups = True
    else:
        raise ValueError(
            "CUBLAS_BLOCKED SF layout expects a 2D SFA scale or a 3D SFB "
            f"scale with is_a=False; got shape={tuple(scale.shape)}, is_a={is_a}"
        )

    n_col_blocks = _ceil_div(cols, 4)
    padded_rows_per_group = _ceil_div(rows_per_group, 128) * 128
    scale_bytes = scale.view(torch.uint8)
    use_u64 = rows_per_group % 128 == 0 and _can_use_u64_loads(scale_bytes, cols)
    # The tiled byte and u64 paths both write complete blocked atoms,
    # including zero-filled row/column padding.
    out = torch.empty(
        (groups * padded_rows_per_group, n_col_blocks * 4),
        dtype=scale.dtype,
        device=scale.device,
    )
    if scale.numel() == 0:
        return out

    out_bytes = out.view(torch.uint8)
    out_words = out_bytes.view(torch.uint32).view(-1)
    dummy_offsets = torch.empty((1,), dtype=torch.int32, device=scale.device)
    if use_u64:
        scale_qwords = scale_bytes.view(torch.uint64)
        out_qwords = out_bytes.view(torch.uint64).view(-1)
    else:
        scale_qwords = torch.empty((1,), dtype=torch.uint64, device=scale.device)
        out_qwords = torch.empty((1,), dtype=torch.uint64, device=scale.device)

    with ensure_cuda_driver_context():
        _compile_natural_to_cublas_blocked(
            rows_per_group,
            cols,
            has_groups,
            False,
            use_u64,
        )(
            scale_bytes,
            scale_qwords,
            dummy_offsets,
            out_bytes,
            out_words,
            out_qwords,
        )
    return out
