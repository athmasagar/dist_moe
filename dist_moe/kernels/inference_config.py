# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared same-format inference policy for launch and physical storage planning."""

from __future__ import annotations

import math

import torch

from .blockscaled_config import (
    blockscaled_mega_decode_config,
    blockscaled_staged_decode_config,
    BlockScaledFormatSpec,
    MXFP8_E4M3,
    NVFP4,
)
from .config import (
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    estimate_blockscaled_tmem_cols,
    narrow_blockscaled_inference_token_tile,
    SM100_TMEM_COLS,
    uses_paged_blockscaled_scale_rows,
)

_DECODE_MAX_TOKENS_PER_RANK = 128


def _blockscaled_inference_config(  # noqa: C901 -- retain the source policy as one ordered decision
    *,
    format: BlockScaledFormatSpec,
    num_tokens: int,
    topk: int,
    num_local_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_sms: int | None,
    routing_capacity: int | None,
    mega: bool,
    interleaved_fc13: bool,
    config: dict[str, int | bool] | None = None,
    input_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, int | bool] | None:
    """Resolve the source inference policy without framework-owned state.

    Args:
        format: Same-format MXFP8 E4M3 or NVFP4 activation/weight specialization.
        num_tokens: Rank-local runtime token count, not receive capacity.
        topk: Routes per input token.
        num_local_experts: Number of experts owned by this rank.
        hidden_dim: Logical input/output feature dimension.
        intermediate_dim: Logical SwiGLU output feature dimension.
        num_sms: Resolved launch SM count; None leaves device-less planning logical.
        routing_capacity: Static receive capacity, including routing padding.
        mega: Whether the two forward GEMMs use the chunked Mega launcher.
        interleaved_fc13: Whether prepared W13 rows use the inference layout.
        config: Explicit config, which bypasses automatic inference selection.
        input_dtype: Native input dtype used by the compact-row admission gate.

    Returns:
        Source-selected config, or None when ordinary source defaults apply.

    Raises:
        ValueError: If an unsupported format is requested.
    """
    if format not in (MXFP8_E4M3, NVFP4):
        raise ValueError("inference policy supports same-format MXFP8 E4M3 or NVFP4")
    if config is not None:
        return dict(config)
    num_tokens_per_expert = (
        num_tokens * topk + num_local_experts - 1
    ) // num_local_experts
    if num_tokens_per_expert <= 0:
        return None
    staged = not mega
    if staged and (
        num_tokens > _DECODE_MAX_TOKENS_PER_RANK
        or (not interleaved_fc13 and num_tokens_per_expert > 32)
    ):
        return None
    config_fn = (
        blockscaled_staged_decode_config if staged else blockscaled_mega_decode_config
    )
    for capacity in (None, routing_capacity):
        try:
            resolved = config_fn(
                format,
                num_tokens_per_expert=num_tokens_per_expert,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_local_experts=num_local_experts,
                num_sms=num_sms,
                routing_capacity=capacity,
            )
        except ValueError:
            return None
        if (
            mega
            and interleaved_fc13
            and input_dtype == torch.bfloat16
            and resolved["BLOCK_SIZE_N"] <= num_tokens_per_expert <= 256
            and resolved["BLOCK_SIZE_N"] in (32, 64, 128, 256)
            and num_sms is not None
            and (
                resolved["BLOCK_SIZE_N"] == num_tokens_per_expert
                or num_local_experts
                * min(hidden_dim, 2 * intermediate_dim)
                * int(resolved["NUM_CTAS"])
                >= num_sms * int(resolved["BLOCK_SIZE_M"])
            )
        ):
            compact_n = min(256, max(64, ((3 * num_tokens_per_expert + 63) // 64) * 32))
            if compact_n == 96 and format is MXFP8_E4M3 and resolved["NUM_MMAS"] == 1:
                compact_n = 128
            if resolved["NUM_MMAS"] == 1 or (
                resolved["NUM_CTAS"] == 2
                and resolved["BLOCK_SIZE_M"] == 256 * resolved["NUM_MMAS"]
                and not resolved.get("USE_SM103_ULTRA", False)
                and estimate_blockscaled_tmem_cols(
                    block_m=int(resolved["BLOCK_SIZE_M"]),
                    block_n=compact_n,
                    num_mmas=int(resolved["NUM_MMAS"]),
                    num_tmem_buffers=2,
                    overlapping_accum=False,
                )
                <= SM100_TMEM_COLS
            ):
                resolved = {
                    **resolved,
                    "COMPACT_DECODE_ROWS": True,
                    "BLOCK_SIZE_N": compact_n,
                    "MMA_ATOM_N": compact_n,
                    "PIPELINE_CHUNK_ROWS": math.lcm(128, compact_n),
                    "BLOCK_SIZE_K": 256 if format is NVFP4 else 128,
                    "OVERLAPPING_ACCUM": False,
                    "STAGE_ALL_FC13": True,
                    "FC2_TILES_PER_SCHEDULE_ITEM": 1,
                    "PIPELINE_LEAD_CHUNKS": 1,
                    "SWIGLU_VALUES_PER_THREAD": 8,
                }
        if (
            mega
            and resolved["NUM_MMAS"] == 1
            and not resolved.get("USE_SM103_ULTRA", False)
        ):
            independent_cols = estimate_blockscaled_tmem_cols(
                block_m=int(resolved["BLOCK_SIZE_M"]),
                block_n=int(resolved["BLOCK_SIZE_N"]),
                num_mmas=int(resolved["NUM_MMAS"]),
                num_tmem_buffers=2,
            )
            if independent_cols > SM100_TMEM_COLS:
                sf_cols = independent_cols - 2 * int(resolved["BLOCK_SIZE_N"])
                overlap_cols = estimate_blockscaled_tmem_cols(
                    block_m=int(resolved["BLOCK_SIZE_M"]),
                    block_n=int(resolved["BLOCK_SIZE_N"]),
                    num_mmas=int(resolved["NUM_MMAS"]),
                    num_tmem_buffers=2,
                    overlapping_accum=True,
                )
                if overlap_cols <= SM100_TMEM_COLS and sf_cols < int(
                    resolved["BLOCK_SIZE_N"]
                ):
                    resolved = {**resolved, "OVERLAPPING_ACCUM": True}
        if (
            mega
            and interleaved_fc13
            and 32 <= resolved["BLOCK_SIZE_N"] <= 256
            and resolved["BLOCK_SIZE_N"] % 32 == 0
            and resolved["NUM_CTAS"] == 2
            and resolved["BLOCK_SIZE_M"] // resolved["NUM_MMAS"] == 256
            and resolved.get("MMA_ATOM_N", resolved["BLOCK_SIZE_N"])
            == resolved["BLOCK_SIZE_N"]
            and not resolved.get("USE_SM103_ULTRA", False)
        ):
            resolved["NATIVE_MMA_TAILS"] = True
            resolved["MMA_ATOM_N"] = resolved["BLOCK_SIZE_N"]
        if num_tokens_per_expert > int(resolved["BLOCK_SIZE_N"]) and (
            staged
            or num_tokens_per_expert > 256
            or not resolved.get("NATIVE_MMA_TAILS", False)
        ):
            return None
        routing_multiple = _inference_routing_multiple(resolved, mega=mega)
        if (
            routing_capacity is not None
            and routing_multiple is not None
            and routing_capacity % routing_multiple
        ):
            continue
        if mega:
            resolved["_AUTO_NUM_TOKENS_PER_EXPERT"] = num_tokens_per_expert
        narrowed = narrow_blockscaled_inference_token_tile(
            resolved,
            weight_bits=format.b_dtype.width,
            num_tokens_per_expert=num_tokens_per_expert,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_local_experts=num_local_experts,
            num_sms=num_sms,
        )
        if narrowed is not resolved:
            multiple = _inference_routing_multiple(narrowed, mega=mega)
            if (
                routing_capacity is None
                or multiple is None
                or routing_capacity % multiple == 0
            ):
                return narrowed
        return resolved
    return None


def _inference_routing_multiple(config: dict[str, int | bool], *, mega: bool) -> int:
    """Return the source routing alignment for one resolved inference config."""
    if config.get("COMPACT_DECODE_ROWS", False):
        return 32
    if mega or config.get("SWAP_AB", False):
        token_tile = int(
            config.get("BLOCK_SIZE_N", DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF)
        )
        if (
            uses_paged_blockscaled_scale_rows(token_tile)
            or token_tile > DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        ):
            return token_tile
    return DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
