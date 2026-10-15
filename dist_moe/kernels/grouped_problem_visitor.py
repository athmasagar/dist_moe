# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decode linear grouped-GEMM tiles into expert-local work records."""

from typing import List

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import extract_mlir_values, Int32, new_from_mlir_values

from .config import _WGRAD, uses_paged_blockscaled_scale_rows
from .params import (
    GroupedMmaWorkTileInfo,
    GroupedWorkTileInfo,
)


@cute.jit
def blockscaled_scale_row_start(
    scale_start_m: Int32,
    row_offset: Int32,
    SWAP_MN: cutlass.Constexpr[bool],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
) -> Int32:
    if cutlass.const_expr(not SWAP_MN):
        return scale_start_m + row_offset
    if cutlass.const_expr(uses_paged_blockscaled_scale_rows(BLOCK_SIZE_N)):
        scale_page_rows: cutlass.Constexpr[int] = ((BLOCK_SIZE_N + 127) // 128) * 128
        return (
            scale_start_m
            + row_offset // Int32(BLOCK_SIZE_N) * Int32(scale_page_rows)
            + row_offset % Int32(BLOCK_SIZE_N)
        )
    return scale_start_m + row_offset


@cute.jit
def advance_blockscaled_scale_start(
    scale_start_m: Int32,
    group_rows: Int32,
    SWAP_MN: cutlass.Constexpr[bool],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
) -> Int32:
    if cutlass.const_expr(SWAP_MN and uses_paged_blockscaled_scale_rows(BLOCK_SIZE_N)):
        scale_page_rows: cutlass.Constexpr[int] = ((BLOCK_SIZE_N + 127) // 128) * 128
        return scale_start_m + (
            (group_rows + Int32(BLOCK_SIZE_N - 1)) // Int32(BLOCK_SIZE_N)
        ) * Int32(scale_page_rows)
    return scale_start_m + group_rows


@cute.jit
def _group_sizes(
    split_sizes: cute.Tensor,
    group_idx: Int32,
    M: Int32,
    N: Int32,
    K: Int32,
    problem_type: cutlass.Constexpr[int],
    SWAP_MN: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(problem_type == _WGRAD):
        m_size, n_size, k_size = M, N, Int32(split_sizes[group_idx])
    else:
        m_size, n_size, k_size = Int32(split_sizes[group_idx]), N, K
    if cutlass.const_expr(SWAP_MN):
        return n_size, m_size, k_size
    return m_size, n_size, k_size


@cute.jit
def _tile_grid(
    m_size: Int32,
    n_size: Int32,
    k_size: Int32,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    BLOCK_SIZE_K: cutlass.Constexpr[int],
    FORCE_N_MAJOR: cutlass.Constexpr[bool],
    NUM_N_CLUSTERS: cutlass.Constexpr[int],
    SWAP_MN: cutlass.Constexpr[bool],
    SKIP_EMPTY_K: cutlass.Constexpr[bool] = False,
):
    num_m_tiles = (m_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_tiles = (n_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_k_tiles = (k_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    logical_num_m_tiles = num_m_tiles
    logical_num_n_tiles = num_n_tiles
    logical_m_size = m_size
    logical_n_size = n_size
    if cutlass.const_expr(SWAP_MN):
        logical_num_m_tiles = num_n_tiles
        logical_num_n_tiles = num_m_tiles
        logical_m_size = n_size
        logical_n_size = m_size

    num_tiles = num_m_tiles * num_n_tiles
    if cutlass.const_expr(NUM_N_CLUSTERS > 1):
        if cutlass.const_expr(FORCE_N_MAJOR) or not (logical_m_size < logical_n_size):
            n_tiles_per_cluster = (
                logical_num_n_tiles + NUM_N_CLUSTERS - 1
            ) // NUM_N_CLUSTERS
            num_tiles = (
                logical_num_m_tiles * n_tiles_per_cluster * Int32(NUM_N_CLUSTERS)
            )
    if cutlass.const_expr(SKIP_EMPTY_K):
        # Leaves the group's output untouched, which accumulation into it allows.
        if k_size == Int32(0):
            num_tiles = Int32(0)
    return num_m_tiles, num_n_tiles, num_k_tiles, num_tiles


@cute.jit
def _interleave_m_tile_idx(i: Int32, n: Int32, seg: Int32, K: cutlass.Constexpr[int]):
    """Bijection on ``[0, n)`` that visits the ``ceil(n / seg)`` source
    segments of ``seg`` M tiles round-robin, ``K`` tiles at a time.

    Read as a matrix of segment columns and ``K``-row blocks, the walk goes
    block by block and column by column within a block. Consecutive tiles
    then come from different source ranks, so no single sender's arrival
    gates a run of tiles. Identity when ``seg <= K``.
    """
    full_cols = n // seg
    rem = n % seg
    q = rem // K
    p1 = full_cols * (q + 1) * K + rem
    blk = Int32(0)
    pre = Int32(0)
    if i < p1:
        blk = cutlass.min(i // (K * (full_cols + 1)), q)
        pre = blk * K * (full_cols + 1)
    else:
        blk = q + 1 + (i - p1) // (full_cols * K)
        pre = full_cols * blk * K + rem
    rows = cutlass.min(Int32(K), seg - blk * K)
    off = i - pre
    return (off // rows) * seg + blk * K + off % rows


@cute.jit
def _tile_coords(
    tile_idx: Int32,
    num_m_tiles: Int32,
    num_n_tiles: Int32,
    m_size: Int32,
    n_size: Int32,
    FORCE_N_MAJOR: cutlass.Constexpr[bool],
    NUM_N_CLUSTERS: cutlass.Constexpr[int],
    local_rank: Int32,
    WORLD_SIZE: cutlass.Constexpr[int],
    SWAP_MN: cutlass.Constexpr[bool],
    M_INTERLEAVE: cutlass.Constexpr[int] = 0,
    DIAG_N_BANDS: cutlass.Constexpr[bool] = False,
):
    logical_tile_m_idx = Int32(0)
    logical_tile_n_idx = Int32(0)
    logical_num_m_tiles = num_m_tiles
    logical_num_n_tiles = num_n_tiles
    logical_m_size = m_size
    logical_n_size = n_size
    if cutlass.const_expr(SWAP_MN):
        logical_num_m_tiles = num_n_tiles
        logical_num_n_tiles = num_m_tiles
        logical_m_size = n_size
        logical_n_size = m_size

    if cutlass.const_expr(FORCE_N_MAJOR) or not (logical_m_size < logical_n_size):
        n_tiles_per_cluster = (
            logical_num_n_tiles + NUM_N_CLUSTERS - 1
        ) // NUM_N_CLUSTERS
        cluster_tiles = logical_num_m_tiles * n_tiles_per_cluster
        cluster_idx = tile_idx // cluster_tiles
        within_cluster = tile_idx % cluster_tiles
        logical_tile_m_idx = within_cluster // n_tiles_per_cluster
        if cutlass.const_expr(DIAG_N_BANDS and NUM_N_CLUSTERS > 1):
            # Pass p visits M tile m in band (p + m) % NUM_N_CLUSTERS.
            cluster_idx = (cluster_idx + logical_tile_m_idx) % NUM_N_CLUSTERS
        logical_tile_n_idx = (
            cluster_idx * n_tiles_per_cluster + within_cluster % n_tiles_per_cluster
        )
    else:
        logical_tile_m_idx = tile_idx % logical_num_m_tiles
        logical_tile_n_idx = tile_idx // logical_num_m_tiles

    if cutlass.const_expr(WORLD_SIZE > 1):
        tiles_per_rank = (logical_num_m_tiles + WORLD_SIZE - 1) // WORLD_SIZE
        if cutlass.const_expr(M_INTERLEAVE > 0):
            logical_tile_m_idx = _interleave_m_tile_idx(
                logical_tile_m_idx, logical_num_m_tiles, tiles_per_rank, M_INTERLEAVE
            )
        m_offset = (local_rank * tiles_per_rank) % logical_num_m_tiles
        logical_tile_m_idx = (logical_tile_m_idx + m_offset) % logical_num_m_tiles
    if cutlass.const_expr(SWAP_MN):
        return logical_tile_n_idx, logical_tile_m_idx
    return logical_tile_m_idx, logical_tile_n_idx


class GroupedProblemVisitor:
    """Stateful linear-tile decoder with a cached current expert."""

    def __init__(
        self,
        split_sizes: cute.Tensor,
        M: Int32,
        N: Int32,
        K: Int32,
        group_idx: Int32,
        tile_start: Int32,
        tile_end: Int32,
        split_prefix: Int32,
        m_tile_prefix: Int32,
        n_tile_prefix: Int32,
        scale_split_prefix: Int32,
        m_size: Int32,
        n_size: Int32,
        k_size: Int32,
        num_m_tiles: Int32,
        num_n_tiles: Int32,
        num_k_tiles: Int32,
        G: cutlass.Constexpr[int],
        problem_type: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        local_rank: Int32,
        WORLD_SIZE: cutlass.Constexpr[int],
        SWAP_MN: cutlass.Constexpr[bool],
        M_SUBTILES: cutlass.Constexpr[int],
        M_INTERLEAVE: cutlass.Constexpr[int],
        DIAG_N_BANDS: cutlass.Constexpr[bool] = False,
        SKIP_EMPTY_K: cutlass.Constexpr[bool] = False,
    ):
        self.split_sizes = split_sizes
        self.M = M
        self.N = N
        self.K = K
        self.group_idx = group_idx
        self.tile_start = tile_start
        self.tile_end = tile_end
        self.split_prefix = split_prefix
        self.m_tile_prefix = m_tile_prefix
        self.n_tile_prefix = n_tile_prefix
        self.scale_split_prefix = scale_split_prefix
        self.m_size = m_size
        self.n_size = n_size
        self.k_size = k_size
        self.num_m_tiles = num_m_tiles
        self.num_n_tiles = num_n_tiles
        self.num_k_tiles = num_k_tiles
        self.G = G
        self.problem_type = problem_type
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_N = BLOCK_SIZE_N
        self.BLOCK_SIZE_K = BLOCK_SIZE_K
        self.FORCE_N_MAJOR = FORCE_N_MAJOR
        self.NUM_N_CLUSTERS = NUM_N_CLUSTERS
        self.local_rank = local_rank
        self.WORLD_SIZE = WORLD_SIZE
        self.SWAP_MN = SWAP_MN
        self.M_SUBTILES = M_SUBTILES
        self.M_INTERLEAVE = M_INTERLEAVE
        self.DIAG_N_BANDS = DIAG_N_BANDS
        self.SKIP_EMPTY_K = SKIP_EMPTY_K

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = list(extract_mlir_values(self.split_sizes))
        for value in (
            self.M,
            self.N,
            self.K,
            self.group_idx,
            self.tile_start,
            self.tile_end,
            self.split_prefix,
            self.m_tile_prefix,
            self.n_tile_prefix,
            self.scale_split_prefix,
            self.m_size,
            self.n_size,
            self.k_size,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.local_rank,
        ):
            values.extend(extract_mlir_values(value))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "GroupedProblemVisitor":
        tensor_value_count = len(extract_mlir_values(self.split_sizes))
        split_sizes = new_from_mlir_values(
            self.split_sizes, values[:tensor_value_count]
        )
        old_values = (
            self.M,
            self.N,
            self.K,
            self.group_idx,
            self.tile_start,
            self.tile_end,
            self.split_prefix,
            self.m_tile_prefix,
            self.n_tile_prefix,
            self.scale_split_prefix,
            self.m_size,
            self.n_size,
            self.k_size,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.local_rank,
        )
        runtime_values = values[tensor_value_count:]
        assert len(runtime_values) == len(old_values)
        new_values = [
            new_from_mlir_values(old_value, [value])
            for old_value, value in zip(old_values, runtime_values)
        ]
        return GroupedProblemVisitor(
            split_sizes=split_sizes,
            M=new_values[0],
            N=new_values[1],
            K=new_values[2],
            group_idx=new_values[3],
            tile_start=new_values[4],
            tile_end=new_values[5],
            split_prefix=new_values[6],
            m_tile_prefix=new_values[7],
            n_tile_prefix=new_values[8],
            scale_split_prefix=new_values[9],
            m_size=new_values[10],
            n_size=new_values[11],
            k_size=new_values[12],
            num_m_tiles=new_values[13],
            num_n_tiles=new_values[14],
            num_k_tiles=new_values[15],
            G=self.G,
            problem_type=self.problem_type,
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            BLOCK_SIZE_K=self.BLOCK_SIZE_K,
            FORCE_N_MAJOR=self.FORCE_N_MAJOR,
            NUM_N_CLUSTERS=self.NUM_N_CLUSTERS,
            local_rank=new_values[16],
            WORLD_SIZE=self.WORLD_SIZE,
            SWAP_MN=self.SWAP_MN,
            M_SUBTILES=self.M_SUBTILES,
            M_INTERLEAVE=self.M_INTERLEAVE,
            DIAG_N_BANDS=self.DIAG_N_BANDS,
            SKIP_EMPTY_K=self.SKIP_EMPTY_K,
        )

    @staticmethod
    @cute.jit
    def create(
        split_sizes: cute.Tensor,
        M: Int32,
        N: Int32,
        K: Int32,
        G: cutlass.Constexpr[int],
        problem_type: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        local_rank: Int32,
        WORLD_SIZE: cutlass.Constexpr[int],
        SWAP_MN: cutlass.Constexpr[bool] = False,
        M_SUBTILES: cutlass.Constexpr[int] = 1,
        M_INTERLEAVE: cutlass.Constexpr[int] = 0,
        DIAG_N_BANDS: cutlass.Constexpr[bool] = False,
        SKIP_EMPTY_K: cutlass.Constexpr[bool] = False,
    ) -> "GroupedProblemVisitor":
        """``SKIP_EMPTY_K`` schedules no tiles for a group with an empty
        contraction, whose output then stays untouched."""
        group_idx = Int32(0)
        m_size, n_size, k_size = _group_sizes(
            split_sizes, group_idx, M, N, K, problem_type, SWAP_MN
        )
        num_m_tiles, num_n_tiles, num_k_tiles, num_tiles = _tile_grid(
            m_size,
            n_size,
            k_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            FORCE_N_MAJOR,
            NUM_N_CLUSTERS,
            SWAP_MN,
            SKIP_EMPTY_K,
        )
        num_tiles *= Int32(M_SUBTILES)
        return GroupedProblemVisitor(
            split_sizes,
            M,
            N,
            K,
            group_idx,
            Int32(0),
            num_tiles,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            m_size,
            n_size,
            k_size,
            num_m_tiles,
            num_n_tiles,
            num_k_tiles,
            G,
            problem_type,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            FORCE_N_MAJOR,
            NUM_N_CLUSTERS,
            local_rank,
            WORLD_SIZE,
            SWAP_MN,
            M_SUBTILES,
            M_INTERLEAVE,
            DIAG_N_BANDS,
            SKIP_EMPTY_K,
        )

    @cute.jit
    def _advance_group_to_contain(self, tile_idx: Int32) -> None:
        while tile_idx >= self.tile_end and self.group_idx < self.G:
            group_rows = Int32(self.split_sizes[self.group_idx])
            self.split_prefix += group_rows
            self.scale_split_prefix = advance_blockscaled_scale_start(
                self.scale_split_prefix,
                group_rows,
                self.SWAP_MN,
                self.BLOCK_SIZE_N,
            )
            self.m_tile_prefix += self.num_m_tiles
            self.n_tile_prefix += self.num_n_tiles
            self.tile_start = self.tile_end
            self.group_idx += Int32(1)
            if self.group_idx < self.G:
                self.m_size, self.n_size, self.k_size = _group_sizes(
                    self.split_sizes,
                    self.group_idx,
                    self.M,
                    self.N,
                    self.K,
                    self.problem_type,
                    self.SWAP_MN,
                )
                (
                    self.num_m_tiles,
                    self.num_n_tiles,
                    self.num_k_tiles,
                    num_tiles,
                ) = _tile_grid(
                    self.m_size,
                    self.n_size,
                    self.k_size,
                    self.BLOCK_SIZE_M,
                    self.BLOCK_SIZE_N,
                    self.BLOCK_SIZE_K,
                    self.FORCE_N_MAJOR,
                    self.NUM_N_CLUSTERS,
                    self.SWAP_MN,
                    self.SKIP_EMPTY_K,
                )
                num_tiles *= Int32(self.M_SUBTILES)
                self.tile_end += num_tiles

    @cute.jit
    def _work_tile_coords(self, parent_tile_idx: Int32):
        return _tile_coords(
            parent_tile_idx,
            self.num_m_tiles,
            self.num_n_tiles,
            self.m_size,
            self.n_size,
            self.FORCE_N_MAJOR,
            self.NUM_N_CLUSTERS,
            self.local_rank,
            self.WORLD_SIZE,
            self.SWAP_MN,
            self.M_INTERLEAVE,
            self.DIAG_N_BANDS,
        )

    @cute.jit
    def get_work(self, tile_idx: Int32) -> GroupedWorkTileInfo:
        self._advance_group_to_contain(tile_idx)
        work = GroupedWorkTileInfo(Int32(-1), *([Int32(0)] * 11))
        if self.group_idx < self.G and tile_idx < self.tile_end:
            local_tile_idx = tile_idx - self.tile_start
            parent_tile_idx = local_tile_idx // Int32(self.M_SUBTILES)
            subtile_idx = local_tile_idx % Int32(self.M_SUBTILES)
            tile_m_idx, tile_n_idx = self._work_tile_coords(parent_tile_idx)
            work = GroupedWorkTileInfo(
                self.group_idx,
                tile_m_idx,
                tile_n_idx,
                subtile_idx,
                self.num_k_tiles,
                self.m_size,
                self.n_size,
                self.k_size,
                self.split_prefix,
                self.m_tile_prefix,
                self.n_tile_prefix,
                self.scale_split_prefix,
            )
        return work

    @cute.jit
    def get_mma_work(self, tile_idx: Int32) -> GroupedMmaWorkTileInfo:
        self._advance_group_to_contain(tile_idx)
        work = GroupedMmaWorkTileInfo(Int32(-1), Int32(0), Int32(0))
        if self.group_idx < self.G and tile_idx < self.tile_end:
            parent_tile_idx = (tile_idx - self.tile_start) // Int32(self.M_SUBTILES)
            _, tile_n_idx = self._work_tile_coords(parent_tile_idx)
            work = GroupedMmaWorkTileInfo(self.group_idx, self.num_k_tiles, tile_n_idx)
        return work
