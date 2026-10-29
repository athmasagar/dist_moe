# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP32 SiTU-GLU arithmetic for the shared SwiGLU-family launchers.

See docs/bf16_execution.md for the execution and storage contracts. The shared
launchers retain tiling, pointer chasing, and gradient-store ordering.
"""

import triton
import triton.language as tl

from .fp_math import sigmoid


@triton.jit
def _situ_glu_sigmoid(
    x,
    tanh_gate,
    SITU_GLU_GATE_BETA: tl.constexpr,
    FAST_MATH: tl.constexpr,
):
    """Evaluate sigmoid from the FP32 gate tanh when the gate bound is four."""
    if SITU_GLU_GATE_BETA == 4.0:
        return 0.5 + tanh_gate / (1.0 + tanh_gate * tanh_gate)
    return sigmoid(x, FAST_MATH=FAST_MATH)


@triton.jit
def _situ_glu_fwd(
    x,
    y,
    SITU_GLU_GATE_BETA: tl.constexpr,
    SITU_GLU_UP_BETA: tl.constexpr,
    FAST_MATH: tl.constexpr,
):
    """Apply bounded SiTU-GLU in FP32 without changing tanh precision."""
    tanh_gate = tl.extra.cuda.libdevice.tanh(x / SITU_GLU_GATE_BETA)
    gate = (
        SITU_GLU_GATE_BETA
        * tanh_gate
        * _situ_glu_sigmoid(
            x,
            tanh_gate,
            SITU_GLU_GATE_BETA=SITU_GLU_GATE_BETA,
            FAST_MATH=FAST_MATH,
        )
    )
    up = SITU_GLU_UP_BETA * tl.extra.cuda.libdevice.tanh(y / SITU_GLU_UP_BETA)
    return gate * up


@triton.jit
def _situ_glu_bwd_gate(
    x,
    y,
    dz,
    SITU_GLU_GATE_BETA: tl.constexpr,
    SITU_GLU_UP_BETA: tl.constexpr,
    FAST_MATH: tl.constexpr,
):
    """Compute the gate gradient and retain intermediates for the up gradient."""
    tanh_gate = tl.extra.cuda.libdevice.tanh(x / SITU_GLU_GATE_BETA)
    bounded_gate = SITU_GLU_GATE_BETA * tanh_gate
    sigmoid_x = _situ_glu_sigmoid(
        x,
        tanh_gate,
        SITU_GLU_GATE_BETA=SITU_GLU_GATE_BETA,
        FAST_MATH=FAST_MATH,
    )
    gate = bounded_gate * sigmoid_x
    tanh_up = tl.extra.cuda.libdevice.tanh(y / SITU_GLU_UP_BETA)
    bounded_up = SITU_GLU_UP_BETA * tanh_up
    d_bounded_gate = 1.0 - tanh_gate * tanh_gate
    d_gate = d_bounded_gate * sigmoid_x + bounded_gate * sigmoid_x * (1.0 - sigmoid_x)
    dx = dz * bounded_up * d_gate
    return dx, gate, tanh_up


@triton.jit
def _situ_glu_bwd_up(dz, gate, tanh_up):
    """Compute the up gradient after the shared launcher stores the gate gradient."""
    d_bounded_up = 1.0 - tanh_up * tanh_up
    dy = dz * gate * d_bounded_up
    return dy
