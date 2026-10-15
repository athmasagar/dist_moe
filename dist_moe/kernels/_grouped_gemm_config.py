# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-neutral grouped-GEMM contracts."""

import enum

BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT: int = 128
BLOCKSCALED_DISPATCH_DIM_ALIGNMENT: int = 128
DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF: int = 128

NATIVE_SWIZZLE_GROUP_SIZE = 16
BLOCKSCALED_SWIZZLE_GROUP_SIZE = 8
NATIVE_SWIGLU_EPILOGUE_TILE_N_MULTIPLE = 2 * NATIVE_SWIZZLE_GROUP_SIZE
# Natural-FC13 fused SwiGLU forward: half the 64-column epilogue subtile.
NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE = 32
WGRAD_DISPATCH_TILE_M = 64


class ProblemType(enum.IntEnum):
    """Grouped-GEMM problem selector; kernels receive ``.value`` as a ``tl.constexpr``."""

    FPROP = 1
    DGRAD = 2
    WGRAD = 3


def uses_paged_blockscaled_scale_rows(row_multiple: int) -> bool:
    """Return whether grouped scales use independently padded row pages."""
    return row_multiple in (32, 64) or (row_multiple >= 96 and row_multiple % 128 != 0)
