# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalized native and prepared block-scaled expert weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _interleave_w13_rows(rows_ENK: torch.Tensor) -> torch.Tensor:
    """Place quantized gate/up rows in adjacent groups of eight.

    Args:
        rows_ENK: Canonical contiguous W13 data or natural scales with shape
            ``[E, 2F, K]``. The first F rows are gate, the second F are up.

    Returns:
        Contiguous storage of the same shape and dtype in fused inference order.

    Raises:
        ValueError: If the rows cannot form complete gate/up groups of eight.
    """
    experts, doubled_intermediate, width = rows_ENK.shape
    if doubled_intermediate % 16:
        raise ValueError("fused inference W13 requires F divisible by eight")
    # Byte views permit row rearrangement for float8 dtypes without arithmetic.
    storage_ENK = rows_ENK.view(torch.uint8)
    intermediate = doubled_intermediate // 2
    # B = F / 8 complete row groups; each group pairs eight gate/up rows.
    gate_EB8K = storage_ENK[:, :intermediate].view(experts, -1, 8, width)
    up_EB8K = storage_ENK[:, intermediate:].view(experts, -1, 8, width)
    interleaved_ENK = torch.stack((gate_EB8K, up_EB8K), dim=2).reshape_as(storage_ENK)
    return interleaved_ENK.view(rows_ENK.dtype)


@dataclass(frozen=True)
class BlockscaledWeightSpec:
    """Normalized native and optional prequantized expert weight operands.

    Args:
        w_native: Native grouped expert weight.
        w_q_fprop: Quantized FPROP operand.
        w_scale_fprop: FPROP scale-factor layout.
        w_q_dgrad: Optional quantized DGRAD operand.
        w_scale_dgrad: Optional DGRAD scale-factor layout.
        w_global_scale: Optional per-expert NVFP4 global scale.
        w_global_scale_inv: Optional precomputed reciprocal global scale.
    """

    w_native: torch.Tensor
    w_q_fprop: torch.Tensor | None = None
    w_scale_fprop: torch.Tensor | None = None
    w_q_dgrad: torch.Tensor | None = None
    w_scale_dgrad: torch.Tensor | None = None
    w_global_scale: torch.Tensor | None = None
    w_global_scale_inv: torch.Tensor | None = None

    @property
    def is_prequantized(self) -> bool:
        """Return whether FPROP quantized operands are present."""
        return self.w_q_fprop is not None
