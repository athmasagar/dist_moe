# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared numerical contract for fused RMSNorm/combine kernels."""

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, const_expr, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

from .._quack.reduction_base import ReductionBase


class _FusedRMSNormCombineBase(ReductionBase):
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        N: int,
        TOPK: int,
        stage: int,
        has_expert_ids: bool = False,
    ):
        super().__init__(dtype, N, stage=stage)
        self.TOPK = TOPK
        self.has_expert_ids = has_expert_ids

    def _supports_expert_ids(self, expert_ids: cute.Tensor | None, rows) -> bool:
        """Check that the optional route operand matches the compiled variant."""
        return (expert_ids is None and not self.has_expert_ids) or (
            self.has_expert_ids
            and expert_ids is not None
            and expert_ids.element_type in (Int32, Int64)
            and expert_ids.shape == (rows,)
        )

    @cute.jit
    def _route_valid(self, expert_ids: cute.Tensor | None, row: Int32) -> Boolean:
        """Return whether a flattened expert route participates in arithmetic."""
        valid = Boolean(True)
        if const_expr(self.has_expert_ids):
            assert expert_ids is not None
            valid = Boolean(expert_ids[row] >= 0)
        return valid

    @cute.jit
    def _mask_fragment(self, fragment: cute.Tensor, valid: Boolean) -> None:
        """Clear invalid route operands before NaNs can enter a reduction."""
        if const_expr(self.has_expert_ids):
            if not valid:
                fragment.fill(0.0)

    @cute.jit
    def _route_scalar(self, value: Float32, valid: Boolean) -> Float32:
        """Select an exact zero for inactive routes, including nonfinite inputs."""
        if const_expr(self.has_expert_ids):
            if not valid:
                value = Float32(0.0)
        return value

    @staticmethod
    def _is_16b_aligned(tensor: cute.Tensor, dim: int) -> bool:
        return (tensor.shape[dim] * tensor.element_type.width) % 128 == 0

    @staticmethod
    @dsl_user_op
    def _mul_rn_f32(a, b, *, loc=None, ip=None) -> Float32:
        """Round the product separately from a subsequent FMA."""
        return cute.math.mul(
            Float32(a),
            Float32(b),
            rounding=cute.math.RoundingMode.NEAREST_EVEN,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    @dsl_user_op
    def _fma_rn_f32(a, b, c, *, loc=None, ip=None) -> Float32:
        # Keep FP32 FFMA: folding BF16 inputs into FHFMA slows the forward dot.
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [
                    Float32(a).ir_value(loc=loc, ip=ip),
                    Float32(b).ir_value(loc=loc, ip=ip),
                    Float32(c).ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @cute.jit
    def _fma_dot_f32(self, x: cute.TensorSSA, y: cute.TensorSSA) -> Float32:
        acc = Float32(0.0)
        for i in cutlass.range(cute.size(x.shape), unroll_full=True):
            acc = self._fma_rn_f32(x[i], y[i], acc)
        return acc

    @cute.jit
    def _scale_rn(self, x: cute.TensorSSA, scale: Float32) -> cute.TensorSSA:
        result = cute.make_rmem_tensor(x.shape, Float32)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            result[i] = self._mul_rn_f32(x[i], scale)
        return result.load()
