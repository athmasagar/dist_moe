# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL fused RMSNorm + weighted TOPK reduction.

Public API:
    fused_rmsnorm_combine(x, weights, eps) -> Tensor
        x:       [B, TOPK, DIM] bf16 contiguous
        weights: [B, TOPK]      f32  contiguous
        returns: [B, DIM]       bf16 (with autograd support)

Kernel properties:
  - Optional Kahan-compensated sum-of-squares for near-FP64 accuracy in FP32.
  - Efficient vectorized global→shared→register transfers via CuTe cp.async.
"""

import math

import cutlass.cute as cute
import torch
from cutlass import BFloat16, Boolean, Float32, Int64
from torch import Tensor

from .._cuda_context import ensure_cuda_driver_context
from .._environment import is_blackwell_gpu
from .._jit_cache import jit_cache
from .._quack.cute_dsl_utils import torch2cute_dtype_map
from .._quack_compile_utils import make_fake_stream, make_fake_tensor as fake_tensor
from .._tensor_views import _same_view, _views_overlap
from ..triton.broadcast_n_reduction import (
    broadcast_and_scale,
    scale_and_sum,
    SCALE_AND_SUM_TILE_D_MAX_TOKENS,
)
from ..triton.conditional_copy import conditional_copy_activations
from ._fused_rmsnorm_combine_bwd_kernel import (
    FusedRMSNormCombineBwd,
    FusedRMSNormCombineBwdExact,
)
from ._fused_rmsnorm_combine_fwd_kernel import (
    FusedRMSNormCombineFwd,
    FusedRMSNormCombineFwdParallelTOPK,
    FusedRMSNormCombineFwdParallelTOPKPrefetch,
    FusedRMSNormCombineFwdParallelTOPKStream,
)
from ._rmsnorm_bwd_kernel import FusedRMSNormBwd
from ._rmsnorm_common import (
    _aligned,
    _fake_div,
    _normalize_for_alignment,
)
from ._rmsnorm_fwd_kernel import RMSNormStats
from .rmsnorm import cute_rmsnorm_bwd, cute_rmsnorm_fwd

# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------


def _fwd_kernel_cls(
    dtype,
    norm_dtype,
    output_dtype,
    N,
    TOPK,
    use_kahan,
    tile_d,
    copy_input,
    has_weight,
    decode_batch,
    require_bitwise,
    stream_tokens=False,
) -> type[FusedRMSNormCombineFwd]:
    """Select the pinned forward schedule for the shape, dtype, and copy contract."""
    if (
        dtype == BFloat16
        and norm_dtype == Float32
        and output_dtype == BFloat16
        and FusedRMSNormCombineFwdParallelTOPK.supports_shape(N, TOPK)
        and (
            tile_d
            or decode_batch
            or (
                require_bitwise
                and FusedRMSNormCombineFwdParallelTOPK.prefers_prefill(N, TOPK)
            )
        )
        and not (use_kahan or copy_input or has_weight)
        and is_blackwell_gpu()
        and torch.cuda.get_device_capability()[0] < 12
    ):
        if stream_tokens:
            if FusedRMSNormCombineFwdParallelTOPKStream.stream_stages(N, TOPK) >= 2:
                return FusedRMSNormCombineFwdParallelTOPKStream
            return FusedRMSNormCombineFwdParallelTOPKPrefetch
        return FusedRMSNormCombineFwdParallelTOPK
    return FusedRMSNormCombineFwd


@jit_cache
def _compile_fwd(
    dtype,
    norm_dtype,
    output_dtype,
    weight_dtype,
    N,
    TOPK,
    use_kahan,
    require_bitwise,
    tile_d,
    copy_input,
    has_weight,
    decode_batch,
    copy_condition_dtype=None,
    stream_tokens=False,
    expert_id_dtype=None,
    has_expert_ids=False,
):
    """Cache a forward variant, including route-ID dtype and device-gated input copying."""
    batch_sym = cute.sym_int()
    batch_topk_sym = cute.sym_int()
    div = math.gcd(N, *(128 // dt.width for dt in [dtype, norm_dtype, output_dtype]))
    x_cute = fake_tensor(dtype, (batch_topk_sym, N), div)
    w_cute = fake_tensor(weight_dtype, (batch_topk_sym,))
    y_cute = fake_tensor(output_dtype, (batch_sym, N), div)
    rstd_cute = fake_tensor(Float32, (batch_topk_sym,))
    expert_ids_cute = fake_tensor(expert_id_dtype, (batch_topk_sym,))
    x_copy_cute = fake_tensor(dtype, (batch_topk_sym, N), div)
    x_copy_offset_cute = None
    x_copy_condition_cute = None
    if copy_condition_dtype is not None:
        x_copy_offset_cute = fake_tensor(Int64, (1,))
        x_copy_condition_cute = fake_tensor(copy_condition_dtype, (1,))
    g_cute = fake_tensor(dtype, (1, N), div) if has_weight else None
    kernel_cls = _fwd_kernel_cls(
        dtype,
        norm_dtype,
        output_dtype,
        N,
        TOPK,
        use_kahan,
        tile_d,
        copy_input,
        has_weight,
        decode_batch,
        require_bitwise,
        stream_tokens,
    )
    kernel = kernel_cls(
        dtype,
        norm_dtype,
        N,
        TOPK,
        use_kahan=use_kahan,
        require_bitwise=require_bitwise,
        tile_d=tile_d,
        copy_input=copy_input,
        has_weight=has_weight,
        has_expert_ids=has_expert_ids,
    )
    if not kernel.can_implement(
        x_cute,
        w_cute,
        y_cute,
        rstd_cute,
        x_copy_cute,
        g_cute,
        expert_ids_cute,
    ):
        raise ValueError(
            "FusedRMSNormCombineFwd requires 16-byte aligned "
            f"row shapes; got N={N}, input_bits={dtype.width}, "
            f"output_bits={output_dtype.width}"
        )
    return cute.compile(
        kernel,
        x_cute,
        w_cute,
        y_cute,
        rstd_cute,
        expert_ids_cute,
        x_copy_cute,
        x_copy_offset_cute,
        x_copy_condition_cute,
        g_cute,
        Float32(0),  # eps
        Float32(0),  # gain_center
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _bwd_divisibility(
    dtype,
    grad_y_dtype,
    N,
    use_exact_kernel,
) -> int:
    """Return the element divisibility required by backward vector copies.

    The exact kernel pins its reduction grouping; the optimized kernel shares
    `_fake_div` with its emitted vector width.
    """
    return (
        FusedRMSNormCombineBwdExact._exact_bwd_vecsize(N)
        if use_exact_kernel
        # `grad_x` is required to match `x`, so repeat `dtype` to mirror the
        # kernel's `(x, grad_y, grad_x)` vector-width inputs exactly.
        else _fake_div(N, (dtype, grad_y_dtype, dtype))
    )


def _prepare_bwd_grad_y(
    grad_y: Tensor,
    norm_output_dtype: torch.dtype,
    *,
    require_bitwise: bool,
    x: Tensor | None = None,
) -> Tensor:
    """Prepare the backward gradient, optionally preserving the low-HBM BF16 path.

    With ``x``, BF16 gradients bound for the shared-grad_y kernel stay BF16; it
    upcasts them exactly."""
    if grad_y.dtype == torch.bfloat16 and (
        not require_bitwise
        or (x is not None and _bwd_shares_grad_y(x, require_bitwise))
    ):
        return grad_y
    return grad_y.to(
        dtype=norm_output_dtype,
        memory_format=torch.contiguous_format,
    ).contiguous()


@jit_cache
def _compile_bwd(
    dtype,
    grad_y_dtype,
    N,
    TOPK,
    use_exact_kernel,
    projection_threads_per_row=0,
    projection_cluster_n=0,
    use_shared_grad_y=False,
    recompute_rstd=False,
    use_kahan=False,
    expert_id_dtype=None,
    has_expert_ids=False,
):
    """Cache the exact or optimized backward schedule and its optional route operand."""
    batch_sym = cute.sym_int()
    batch_topk_sym = cute.sym_int()
    div = _bwd_divisibility(
        dtype,
        grad_y_dtype,
        N,
        use_exact_kernel,
    )
    grad_y_cute = fake_tensor(grad_y_dtype, (batch_sym, N), div)
    x_cute = fake_tensor(dtype, (batch_topk_sym, N), div)
    w_cute = fake_tensor(Float32, (batch_topk_sym,))
    rstd_cute = fake_tensor(Float32, (batch_topk_sym,))
    grad_x_cute = fake_tensor(dtype, (batch_topk_sym, N), div)
    grad_w_cute = fake_tensor(Float32, (batch_topk_sym,))
    expert_ids_cute = fake_tensor(expert_id_dtype, (batch_topk_sym,))
    if use_exact_kernel:
        kernel = FusedRMSNormCombineBwdExact(
            dtype,
            grad_y_dtype,
            N,
            TOPK,
            projection_threads_per_row=projection_threads_per_row,
            projection_cluster_n=projection_cluster_n,
            use_shared_grad_y=use_shared_grad_y,
            recompute_rstd=recompute_rstd,
            use_kahan=use_kahan,
            has_expert_ids=has_expert_ids,
        )
    else:
        kernel = FusedRMSNormCombineBwd(
            dtype, grad_y_dtype, N, TOPK, has_expert_ids=has_expert_ids
        )
    if not kernel.can_implement(
        grad_y_cute,
        x_cute,
        w_cute,
        rstd_cute,
        grad_x_cute,
        grad_w_cute,
        expert_ids_cute,
    ):
        raise ValueError(
            "FusedRMSNormCombineBwd cannot implement the requested dtype widths "
            "or 16-byte row alignment; "
            f"got N={N}, x_bits={dtype.width}, "
            f"grad_y_bits={grad_y_dtype.width}, grad_x_bits={dtype.width}"
        )
    return cute.compile(
        kernel,
        grad_y_cute,
        x_cute,
        w_cute,
        rstd_cute,
        grad_x_cute,
        grad_w_cute,
        expert_ids_cute,
        Float32(0),  # eps
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


def _is_supported_fwd_dim(dim: int) -> bool:
    """Return whether a feature width fits the fused forward kernel contract."""
    return 256 <= dim <= FusedRMSNormCombineFwd._MAX_DIM and dim % 128 == 0


def supports_accelerated_fwd_contract(
    *,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    topk: int,
    dim: int,
) -> bool:
    """Return whether dtypes, top-k, and width support the fused MoE forward.

    The predicate covers BF16 inputs, FP32 scores and normalized values,
    BF16/FP32 output, top-k two or eight, and 128-aligned widths in
    [256, 131072]. It does not inspect tensor devices, strides, or values."""
    return (
        input_dtype == torch.bfloat16
        and weight_dtype == torch.float32
        and norm_output_dtype == torch.float32
        and output_dtype in (torch.bfloat16, torch.float32)
        and topk in (2, 8)
        and _is_supported_fwd_dim(dim)
    )


def _supports_accelerated_fwd(
    x: Tensor,
    weights: Tensor,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    use_kahan: bool,
) -> bool:
    """Check the tensor layout and scalar policy needed by the accelerated forward."""
    _, topk, dim = x.shape
    return (
        x.is_cuda
        and x.is_contiguous()
        and weights.is_contiguous()
        and supports_accelerated_fwd_contract(
            input_dtype=x.dtype,
            weight_dtype=weights.dtype,
            norm_output_dtype=norm_output_dtype,
            output_dtype=output_dtype,
            topk=topk,
            dim=dim,
        )
        and not use_kahan
    )


def _fused_rmsnorm_combine_bwd_kind(
    *,
    capability: int,
    dim: int,
    require_bitwise: bool,
    input_dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Resolve the pinned backward family without changing its reduction geometry."""
    if (
        10 <= capability < 12
        and require_bitwise
        and input_dtype == torch.bfloat16
        and _is_supported_fwd_dim(dim)
    ):
        return "fused_exact"
    max_fused_dim = (
        FusedRMSNormCombineBwdExact._MAX_REGISTER_DIM
        if require_bitwise or capability < 10
        else FusedRMSNormCombineBwd._MAX_DIM
    )
    if (require_bitwise and capability == 9 and dim == 256) or dim > max_fused_dim:
        return "standalone"
    return "fused_exact" if require_bitwise or capability < 10 else "fused_optimized"


def _exact_bwd_projection_geometry(
    dim: int,
    num_rows: int,
    *,
    use_legacy_bwd_reduction: bool = False,
) -> tuple[int, int]:
    """Match the standalone backward projection tree for the width and row-count bucket.

    A zero pair selects the register schedule; a nonzero pair specifies the
    projection threads per row and the number of column shards."""
    register_max_dim = FusedRMSNormCombineBwdExact._MAX_REGISTER_DIM
    if use_legacy_bwd_reduction and dim <= register_max_dim:
        return 0, 0
    # Token count can change the standalone reduction tree. Cache the
    # resolved geometry in _compile_bwd, not every possible token count.
    config = FusedRMSNormBwd._blackwell_launch_config(
        dim,
        dtype_in=16,
        dtype_out=32,
        T_hint=1 << max(0, int(num_rows) - 1).bit_length(),
    )
    threads, cluster = config["threads_per_row"], config["cluster_n"]
    if dim <= register_max_dim and (threads, cluster) == (128, 1):
        return 0, 0
    return threads, cluster


def _rmsnorm_weighted_topk_reduction_unfused_fwd(
    x: Tensor,
    weights: Tensor,
    eps: float,
    *,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    weight: Tensor | None = None,
    gain_center: float = 0.0,
    use_kahan: bool = False,
    expert_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compose standalone normalization and reduction, clearing inactive routes at each boundary."""
    B, TOPK, DIM = x.shape
    x_masked = (
        x if expert_ids is None else torch.where((expert_ids >= 0).unsqueeze(-1), x, 0)
    )
    x_2d = x_masked.reshape(B * TOPK, DIM)
    x_norm, rstd = cute_rmsnorm_fwd(
        x=x_2d,
        w=weight,
        eps=eps,
        output_dtype=norm_output_dtype,
        gain_center=gain_center,
        input_scale=weight is not None,
        use_kahan_summation=use_kahan,
    )
    x_norm_3d = x_norm.reshape(B, TOPK, DIM)
    if expert_ids is not None:
        valid = expert_ids >= 0
        x_norm_3d = torch.where(valid.unsqueeze(-1), x_norm_3d, 0)
        rstd = torch.where(valid.reshape(-1), rstd, 0)
    y, _ = scale_and_sum(
        x=x_norm_3d,
        scale=weights,
        expert_ids=expert_ids,
        return_copy=False,
        output_dtype=output_dtype,
    )
    return y, rstd.reshape(B, TOPK), x_norm_3d


def _validate_fwd_out(
    out: Tensor,
    x: Tensor,
    weights: Tensor,
    output_dtype: torch.dtype,
    x_copy: Tensor | None,
    weight: Tensor | None,
    expert_ids: Tensor | None,
    x_copy_buffer: Tensor | None,
) -> None:
    """Reject output views that overwrite a live input or saved-state buffer."""
    if (
        out.shape != (x.shape[0], x.shape[2])
        or out.dtype != output_dtype
        or out.device != x.device
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError(
            "out must be contiguous, 16-byte aligned, and match the output's "
            "shape, dtype, and device"
        )
    if any(
        tensor is not None and _views_overlap(out, tensor)
        for tensor in (x, weights, x_copy, weight, expert_ids, x_copy_buffer)
    ):
        raise ValueError("out must not overlap an input or x_copy")


def _copy_condition_dtype(condition: Tensor):
    """Map the device copy predicate to the CuTe dtype used by its compiled operand."""
    if condition.dtype == torch.bool:
        return Boolean
    return torch2cute_dtype_map[condition.dtype]


def _validate_x_copy_buffer(
    x: Tensor,
    x_copy: Tensor | None,
    buffer: Tensor | None,
    offset: Tensor | None,
    condition: Tensor | None,
) -> bool:
    """Validate the device-gated activation-buffer copy; True when requested."""
    if buffer is None and offset is None and condition is None:
        return False
    if buffer is None or offset is None or condition is None:
        raise ValueError(
            "x_copy_buffer, x_copy_offset, and x_copy_condition go together"
        )
    if x_copy is not None:
        raise ValueError("x_copy and x_copy_buffer are exclusive")
    if (
        buffer.dtype != torch.uint8
        or buffer.device != x.device
        or not buffer.is_contiguous()
        or buffer.data_ptr() % 16
        or buffer.numel() < x.numel() * x.element_size()
    ):
        raise ValueError(
            "x_copy_buffer must be a contiguous, 16-byte aligned uint8 tensor "
            "on x's device holding at least one copy of x"
        )
    if offset.shape != (1,) or offset.dtype != torch.int64 or offset.device != x.device:
        raise ValueError("x_copy_offset must be a [1] int64 tensor on x's device")
    if (
        condition.shape != (1,)
        or condition.dtype not in (torch.bool, torch.int32)
        or condition.device != x.device
    ):
        raise ValueError(
            "x_copy_condition must be a [1] bool or int32 tensor on x's device"
        )
    return True


def _copy_input_unfused(
    x: Tensor,
    x_copy: Tensor | None,
    buffer: Tensor | None,
    offset: Tensor | None,
    condition: Tensor | None,
) -> None:
    """Copy the unmodified input into direct or device-offset-backed saved-state storage."""
    if x_copy is not None:
        x_copy.copy_(x)
    if buffer is not None:
        conditional_copy_activations(
            condition=condition,
            lhs=None,
            lhs_offset=None,
            rhs=x,
            rhs_offset=offset,
            activation_buffer=buffer,
            copy_to_buffer=True,
        )


def _x_copy_view(x_2d: Tensor, x_copy: Tensor | None, buffer: Tensor | None) -> Tensor:
    """The kernel's copy destination; a buffer view aliases byte zero and the
    kernel rebases it by the device offset."""
    if buffer is not None:
        num_bytes = x_2d.numel() * x_2d.element_size()
        return buffer[:num_bytes].view(x_2d.dtype).view(x_2d.shape)
    return x_2d if x_copy is None else x_copy.reshape(x_2d.shape)


def _fused_rmsnorm_combine_fwd(
    x: Tensor,
    weights: Tensor,
    eps: float = 1e-5,
    use_kahan: bool = False,
    *,
    norm_output_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    require_bitwise: bool = True,
    x_copy: Tensor | None = None,
    x_copy_buffer: Tensor | None = None,
    x_copy_offset: Tensor | None = None,
    x_copy_condition: Tensor | None = None,
    weight: Tensor | None = None,
    gain_center: float = 0.0,
    out: Tensor | None = None,
    expert_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Launch the fused forward kernel.

    Args:
        x: [B, TOPK, DIM] bf16 contiguous tensor.
        weights: [B, TOPK] f32 contiguous.
        eps: Epsilon for floating-point stability.
        use_kahan: Use Kahan-compensated sum-of-squares.
        require_bitwise: Preserve the standalone operator's exact reduction
            order.
        x_copy: Optional preallocated tensor receiving an exact copy of `x`.
        x_copy_buffer: Optional uint8 activation buffer receiving the copy of
            `x` at the device byte offset ``x_copy_offset`` ([1] int64), unless
            the device ``x_copy_condition`` ([1] bool or int32) is non-zero.
            The offset must be 16-byte aligned and leave room for one copy of
            `x`; it is on device, so it is not checked. Exclusive with
            ``x_copy``. The base kernel stores the copy from the rows it holds;
            otherwise a separate gated copy runs after the launch.
        weight: Optional contiguous [DIM] input-scale gamma matching x's dtype.
            The kernel normalizes ``(weight + gain_center) * x`` in the single
            fused launch — no separate pre-scaling kernel — with the same
            per-element rounding as ``cute_rmsnorm_fwd(input_scale=True)``.
            Forward-only: no backward kernel consumes it.
        gain_center: Additive center folded into ``weight`` in-kernel.
        out: Optional contiguous, 16-byte-aligned [B, DIM] output buffer;
            must not overlap any input or ``x_copy``.
        expert_ids: Optional contiguous int32/int64 ``[B, TOPK]`` route IDs.
            Negative routes are excluded from normalization and reduction.

    Returns:
        (y, rstd) where y is [B, DIM] bf16 and rstd is [B, TOPK] f32.
    """
    if norm_output_dtype is None:
        norm_output_dtype = x.dtype
    if output_dtype is None:
        output_dtype = x.dtype
    _validate_expert_ids(x, expert_ids)
    if out is not None:
        _validate_fwd_out(
            out, x, weights, output_dtype, x_copy, weight, expert_ids, x_copy_buffer
        )
    if x_copy is not None and (
        x_copy.shape != x.shape
        or x_copy.dtype != x.dtype
        or x_copy.device != x.device
        or not x_copy.is_contiguous()
    ):
        raise ValueError(
            "x_copy must be a contiguous tensor matching x's shape, dtype, and device"
        )
    buffer_copy = _validate_x_copy_buffer(
        x, x_copy, x_copy_buffer, x_copy_offset, x_copy_condition
    )
    if weight is not None:
        _validate_input_scale_weight(x, weight)
        if use_kahan:
            raise NotImplementedError(
                "input-scale weight is not supported with Kahan summation"
            )
        if x_copy is not None or buffer_copy:
            raise NotImplementedError(
                "input-scale weight is forward-only; the training input copy "
                "is not supported with it"
            )
    if not use_kahan and not _supports_accelerated_fwd(
        x=x,
        weights=weights,
        norm_output_dtype=norm_output_dtype,
        output_dtype=output_dtype,
        use_kahan=use_kahan,
    ):
        y, rstd, _ = _rmsnorm_weighted_topk_reduction_unfused_fwd(
            x=x,
            weights=weights,
            eps=eps,
            norm_output_dtype=norm_output_dtype,
            output_dtype=output_dtype,
            weight=weight,
            gain_center=gain_center,
            use_kahan=use_kahan,
            expert_ids=expert_ids,
        )
        _copy_input_unfused(x, x_copy, x_copy_buffer, x_copy_offset, x_copy_condition)
        if out is not None:
            out.copy_(y)
            y = out
        return y, rstd

    B, TOPK, DIM = x.shape
    x_2d = x.reshape(B * TOPK, DIM)
    w_flat = weights.reshape(B * TOPK)
    g_2d = None if weight is None else weight.reshape(1, DIM)
    expert_ids_flat = None if expert_ids is None else expert_ids.reshape(B * TOPK)
    expert_id_dtype = (
        None if expert_ids is None else torch2cute_dtype_map[expert_ids.dtype]
    )

    y = (
        out
        if out is not None
        else torch.empty(B, DIM, dtype=output_dtype, device=x.device)
    )
    rstd = torch.empty(B * TOPK, dtype=torch.float32, device=x.device)

    dtype = torch2cute_dtype_map[x.dtype]
    norm_dtype = torch2cute_dtype_map[norm_output_dtype]
    out_dtype = torch2cute_dtype_map[output_dtype]
    weight_dtype = torch2cute_dtype_map[weights.dtype]
    tile_d = B <= SCALE_AND_SUM_TILE_D_MAX_TOKENS
    decode_batch = B <= 256
    # TMA needs a 16-byte aligned global address.
    stream_tokens = x_2d.data_ptr() % 16 == 0 and (
        FusedRMSNormCombineFwdParallelTOPKStream.prefers_stream(B, DIM, TOPK)
        or FusedRMSNormCombineFwdParallelTOPKPrefetch.prefers_prefetch(B, DIM, TOPK)
    )
    # CTAs store the copy while others still read x, so an overlapping buffer
    # takes the separate copy.
    fuse_buffer_copy = (
        buffer_copy
        and not _views_overlap(x, x_copy_buffer)
        and _fwd_kernel_cls(
            dtype,
            norm_dtype,
            out_dtype,
            DIM,
            TOPK,
            use_kahan,
            tile_d,
            False,
            weight is not None,
            decode_batch,
            require_bitwise,
            stream_tokens,
        )
        is FusedRMSNormCombineFwd
    )
    x_copy_2d = _x_copy_view(x_2d, x_copy, x_copy_buffer if fuse_buffer_copy else None)
    with ensure_cuda_driver_context():
        _compile_fwd(
            dtype,
            norm_dtype,
            out_dtype,
            weight_dtype,
            DIM,
            TOPK,
            use_kahan,
            require_bitwise,
            tile_d,
            x_copy is not None or fuse_buffer_copy,
            weight is not None,
            decode_batch,
            _copy_condition_dtype(x_copy_condition) if fuse_buffer_copy else None,
            stream_tokens,
            expert_id_dtype=expert_id_dtype,
            has_expert_ids=expert_ids is not None,
        )(
            x_2d,
            w_flat,
            y,
            rstd,
            expert_ids_flat,
            x_copy_2d,
            x_copy_offset if fuse_buffer_copy else None,
            x_copy_condition if fuse_buffer_copy else None,
            g_2d,
            eps,
            gain_center,
        )
    if buffer_copy and not fuse_buffer_copy:
        _copy_input_unfused(x, None, x_copy_buffer, x_copy_offset, x_copy_condition)

    return y, rstd.reshape(B, TOPK)


def _bwd_shares_grad_y(x: Tensor, require_bitwise: bool) -> bool:
    """Whether the backward shares one grad_y row across adjacent expert rows
    in a CTA. Smaller batches and narrower rows keep the register schedule;
    the choice is cached per schedule, not per token count."""
    if not x.is_cuda:
        return False
    B, TOPK, DIM = x.shape
    capability = torch.cuda.get_device_capability(x.device)[0]
    kind = _fused_rmsnorm_combine_bwd_kind(
        capability=capability,
        dim=DIM,
        require_bitwise=require_bitwise,
        input_dtype=x.dtype,
    )
    return (
        10 <= capability < 12
        and kind == "fused_exact"
        and x.dtype == torch.bfloat16
        and TOPK == 8
        and 7168 <= DIM <= 32768
        and B >= 512
    )


def _bwd_recomputes_rstd(dim: int, use_kahan: bool) -> bool:
    """Whether the shared-grad_y kernel can reproduce `RMSNormStats` itself:
    one CTA and 128 threads per row."""
    config = RMSNormStats(dim, use_kahan).config
    return (
        _is_supported_fwd_dim(dim)
        and config.threads_per_row == 128
        and config.cluster_n == 1
    )


def _fused_rmsnorm_combine_bwd(
    grad_y: Tensor,
    x: Tensor,
    weights: Tensor,
    rstd: Tensor | None,
    *,
    eps: float = 1e-5,
    require_bitwise: bool = True,
    use_kahan: bool = False,
    grad_x: Tensor | None = None,
    use_legacy_bwd_reduction: bool = False,
    expert_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Launch the fused backward kernel.

    The fused paths normalize read-only inputs to the alignment required by their
    vectorized copies. A caller-provided `grad_x` must be contiguous. If its base
    address is insufficiently aligned, the kernel writes an aligned temporary and
    copies the result into the requested buffer.

    Args:
        grad_y: [B, DIM] bf16/fp32 tensor; arbitrary strides are supported.
        x: [B, TOPK, DIM] bf16 tensor with unit inner stride. Compatible row-strided
            views bind directly; other layouts incur a full-size aligned clone.
        weights: [B, TOPK] f32 contiguous.
        rstd: Optional [B, TOPK] f32 contiguous tensor from forward. When it is
            absent, the fused path recomputes it and the fallback path obtains
            it while recomputing the normalized input.
        use_kahan: Must match the forward's. Every path that recomputes the
            sum-of-squares needs it, or the gradients belong to a different
            normalization than the one the forward applied.
        grad_x: Optional preallocated [B, TOPK, DIM] bf16 output.
        use_legacy_bwd_reduction: Preserve the original fused reduction through
            width 8192.
        expert_ids: Optional contiguous int32/int64 ``[B, TOPK]`` route IDs.
            Negative routes receive zero input and score gradients.

    Returns:
        (grad_x, grad_weights) where grad_x is [B, TOPK, DIM] bf16
        and grad_weights is [B, TOPK] f32.
    """

    B, TOPK, DIM = x.shape
    _validate_expert_ids(x, expert_ids)
    expert_ids_flat = None if expert_ids is None else expert_ids.reshape(B * TOPK)
    expert_id_dtype = (
        None if expert_ids is None else torch2cute_dtype_map[expert_ids.dtype]
    )
    if grad_x is not None and (
        grad_x.shape != x.shape
        or grad_x.dtype != x.dtype
        or grad_x.device != x.device
        or not grad_x.is_contiguous()
    ):
        raise ValueError(
            "grad_x must be a contiguous tensor matching x's shape, dtype, and device"
        )

    capability = torch.cuda.get_device_capability(x.device)[0]
    bwd_kind = _fused_rmsnorm_combine_bwd_kind(
        capability=capability,
        dim=DIM,
        require_bitwise=require_bitwise,
        input_dtype=x.dtype,
    )
    if bwd_kind == "standalone":
        # Without an explicit output dtype, `broadcast_and_scale` derives
        # `grad_x_norm`'s dtype from `dy`. Preserve the historical FP32 input so
        # standalone RMSNorm backward continues to receive FP32 `grad_x_norm`.
        fallback_grad_y = grad_y.float().contiguous()
        fallback_x = (
            x
            if expert_ids is None
            else torch.where((expert_ids >= 0).unsqueeze(-1), x, 0)
        )
        x_norm, unfused_rstd = cute_rmsnorm_fwd(
            fallback_x.reshape(B * TOPK, DIM),
            None,
            eps=eps,
            output_dtype=torch.float32,
            use_kahan_summation=use_kahan,
        )
        grad_x_norm, grad_weights = broadcast_and_scale(
            dy=fallback_grad_y,
            scale=weights,
            x=x_norm.reshape(B, TOPK, DIM),
            expert_ids=expert_ids,
        )
        grad_x_out, _ = cute_rmsnorm_bwd(
            dy=grad_x_norm.reshape(B * TOPK, DIM),
            x=fallback_x.reshape(B * TOPK, DIM),
            w=None,
            rstd=unfused_rstd,
            dx=None if grad_x is None else grad_x.reshape(B * TOPK, DIM),
        )
        if expert_ids is not None:
            grad_x_out.masked_fill_((expert_ids_flat < 0).unsqueeze(-1), 0)
            grad_weights.masked_fill_(expert_ids < 0, 0)
        return grad_x_out.reshape(B, TOPK, DIM), grad_weights

    use_exact_kernel = bwd_kind == "fused_exact"
    use_blackwell_exact = (
        10 <= capability < 12 and use_exact_kernel and x.dtype == torch.bfloat16
    )
    x_2d = x.reshape(B * TOPK, DIM)
    use_shared_grad_y = _bwd_shares_grad_y(x, require_bitwise)
    recompute_rstd = (
        rstd is None and use_shared_grad_y and _bwd_recomputes_rstd(DIM, use_kahan)
    )
    if recompute_rstd:
        # Written by the backward kernel.
        rstd = torch.empty(B, TOPK, dtype=torch.float32, device=x.device)
    elif rstd is None:
        if use_blackwell_exact and _is_supported_fwd_dim(DIM):
            # Match the FP32 normalization layout while writing only statistics.
            # This avoids the temporary combine output and its weighted reduction.
            x_2d = _normalize_for_alignment(x_2d, RMSNormStats._VECTOR_SIZE)
            rstd = torch.empty(B * TOPK, dtype=torch.float32, device=x.device)
            with ensure_cuda_driver_context():
                RMSNormStats.compile(DIM, use_kahan, expert_id_dtype)(
                    x_2d, rstd, expert_ids_flat, eps
                )
            rstd = rstd.reshape(B, TOPK)
        else:
            recomputed_output, rstd = _fused_rmsnorm_combine_fwd(
                x,
                weights,
                eps=eps,
                norm_output_dtype=torch.float32,
                output_dtype=x.dtype,
                require_bitwise=require_bitwise,
                use_kahan=use_kahan,
                expert_ids=expert_ids,
            )
            del recomputed_output

    projection_threads_per_row, projection_cluster_n = 0, 0
    if use_blackwell_exact:
        projection_threads_per_row, projection_cluster_n = (
            _exact_bwd_projection_geometry(
                DIM,
                B * TOPK,
                use_legacy_bwd_reduction=use_legacy_bwd_reduction,
            )
        )
    # The shared-grad_y kernel stages BF16 grad_y and upcasts it exactly.
    keep_bf16_grad_y = use_shared_grad_y and grad_y.dtype == torch.bfloat16
    if (use_shared_grad_y or projection_threads_per_row) and not keep_bf16_grad_y:
        grad_y = grad_y.float()

    dtype = torch2cute_dtype_map[x.dtype]
    grad_y_dtype = torch2cute_dtype_map[grad_y.dtype]
    div = _bwd_divisibility(
        dtype,
        grad_y_dtype,
        DIM,
        use_exact_kernel,
    )

    grad_y = _normalize_for_alignment(grad_y, div)
    x_2d = _normalize_for_alignment(x_2d, div)
    requested_grad_x = None if grad_x is None else grad_x.reshape(B * TOPK, DIM)
    grad_x_out = (
        requested_grad_x
        if requested_grad_x is not None and _aligned(requested_grad_x, div)
        else torch.empty(B * TOPK, DIM, dtype=x.dtype, device=x.device)
    )
    if (
        grad_x_out is requested_grad_x
        and _views_overlap(x_2d, grad_x_out)
        and not _same_view(x_2d, grad_x_out)
    ):
        grad_x_out = torch.empty(B * TOPK, DIM, dtype=x.dtype, device=x.device)
    if grad_x_out is requested_grad_x and _views_overlap(grad_y, grad_x_out):
        grad_y = grad_y.clone(memory_format=torch.contiguous_format)
    w_flat = weights.reshape(B * TOPK)
    rstd_flat = rstd.reshape(B * TOPK)

    grad_w = torch.empty(B * TOPK, dtype=torch.float32, device=x.device)

    with ensure_cuda_driver_context():
        # The optimized backward is GB10x-only. On older GPUs, preserve the
        # fused kernel and reduction topology used before this optimization.
        _compile_bwd(
            dtype,
            grad_y_dtype,
            DIM,
            TOPK,
            use_exact_kernel,
            projection_threads_per_row,
            projection_cluster_n,
            use_shared_grad_y,
            recompute_rstd,
            recompute_rstd and use_kahan,
            expert_id_dtype=expert_id_dtype,
            has_expert_ids=expert_ids is not None,
        )(
            grad_y,
            x_2d,
            w_flat,
            rstd_flat,
            grad_x_out,
            grad_w,
            expert_ids_flat,
            Float32(eps),
        )

    if requested_grad_x is not None and grad_x_out is not requested_grad_x:
        requested_grad_x.copy_(grad_x_out)
        grad_x_out = requested_grad_x
    return grad_x_out.reshape(B, TOPK, DIM), grad_w.reshape(B, TOPK)


# ---------------------------------------------------------------------------
# Input-scale (gamma) validation
# ---------------------------------------------------------------------------


def _validate_expert_ids(x: Tensor, expert_ids: Tensor | None) -> None:
    """Validate the optional per-route mask without inspecting device values."""
    if expert_ids is not None and (
        expert_ids.shape != x.shape[:2]
        or expert_ids.dtype not in (torch.int32, torch.int64)
        or expert_ids.device != x.device
        or not expert_ids.is_contiguous()
    ):
        raise ValueError(
            "expert_ids must be contiguous int32/int64 with shape [B, TOPK]"
        )


def _validate_input_scale_weight(x: Tensor, weight: Tensor) -> None:
    """Validate the forward-only input-scale weight without inspecting device values."""
    if (
        weight.shape != (x.shape[-1],)
        or weight.dtype != x.dtype
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "fused_rmsnorm_combine weight must be a contiguous [DIM] tensor "
            f"matching x's dtype and device; got weight shape "
            f"{tuple(weight.shape)} dtype {weight.dtype} for x shape "
            f"{tuple(x.shape)} dtype {x.dtype}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class _FusedRMSNormCombine(torch.autograd.Function):
    """Fused RMSNorm followed by weighted top-k reduction.

    Scalarless by default; an optional ``weight`` selects the input-scaled
    form ``rmsnorm((weight + gain_center) * x)`` in the same single kernel.
    The weighted form is inference/forward-only: no backward supports it.
    Supported forward calls use the accelerated CuTe kernel. Unsupported
    calls raise instead of silently falling back.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weights: Tensor,
        eps: float = 1e-5,
        norm_output_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
        require_bitwise: bool = True,
        use_kahan: bool = False,
        weight: Tensor | None = None,
        gain_center: float = 0.0,
    ) -> Tensor:
        if norm_output_dtype is None:
            norm_output_dtype = x.dtype
        if output_dtype is None:
            output_dtype = x.dtype

        if not _supports_accelerated_fwd(
            x=x,
            weights=weights,
            norm_output_dtype=norm_output_dtype,
            output_dtype=output_dtype,
            use_kahan=False,
        ):
            raise ValueError(
                "fused_rmsnorm_combine only supports "
                "128-aligned MoE dimensions with bf16 inputs, fp32 weights, "
                "topk=2/8, fp32 RMSNorm output, and bf16/fp32 reduction output"
            )
        y, rstd = _fused_rmsnorm_combine_fwd(
            x=x,
            weights=weights,
            eps=eps,
            norm_output_dtype=norm_output_dtype,
            output_dtype=output_dtype,
            require_bitwise=require_bitwise,
            use_kahan=use_kahan,
            weight=weight,
            gain_center=gain_center,
        )

        ctx.save_for_backward(x, weights, rstd)
        ctx.norm_output_dtype = norm_output_dtype
        ctx.eps = eps
        ctx.require_bitwise = require_bitwise
        ctx.use_kahan = use_kahan
        ctx.had_weight = weight is not None
        return y

    @staticmethod
    def backward(
        ctx,
        grad_y: Tensor,
    ) -> tuple[Tensor | None, ...]:
        # The public wrapper rejects grad-requiring inputs with a weight; this
        # backstops direct .apply() misuse — the scalarless backward below
        # would silently produce gradients for the wrong forward.
        assert not ctx.had_weight, (
            "fused_rmsnorm_combine input-scale weight is inference/forward-only"
        )
        x, weights, rstd = ctx.saved_tensors
        # Preserve the public autograd path's historical FP32 reduction topology;
        # only DistMoE opts into the low-HBM BF16 preparation helper.
        grad_y = grad_y.to(
            dtype=ctx.norm_output_dtype,
            memory_format=torch.contiguous_format,
        ).contiguous()
        grad_x, grad_weights = _fused_rmsnorm_combine_bwd(
            grad_y,
            x,
            weights,
            rstd,
            eps=ctx.eps,
            require_bitwise=ctx.require_bitwise,
            use_kahan=ctx.use_kahan,
        )
        return (grad_x, grad_weights) + (None,) * 7


def fused_rmsnorm_combine(
    x: Tensor,
    weights: Tensor,
    eps: float = 1e-5,
    *,
    norm_output_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    require_bitwise: bool = True,
    use_kahan: bool = False,
    weight: Tensor | None = None,
    gain_center: float = 0.0,
) -> Tensor:
    """Fused RMSNorm + weighted top-k reduction for MoE.

    Supports 128-aligned widths from 256 through 131072. On Blackwell, bitwise
    backward follows the standalone reduction geometry for the width, token
    count without materializing FP32 expert intermediates.

    Args:
        x: [B, TOPK, DIM] contiguous tensor.
        weights: [B, TOPK] contiguous routing weights.
        eps: RMSNorm epsilon.
        norm_output_dtype: dtype materialized by the standalone RMSNorm kernel.
        output_dtype: dtype materialized by the standalone top-k reduction.
        require_bitwise: Preserve bitwise equivalence with the standalone operators.
            Set to ``False`` to enable the HBM-bound fused backward and permit
            DIM clustering when batch parallelism is limited.
        use_kahan: Use Kahan-compensated FP32 sum-of-squares instead of naive
            FP32.
        weight: Optional contiguous [DIM] gamma matching ``x``'s dtype. Selects
            the input-scaled norm ``rmsnorm((weight + gain_center) * x)`` — the
            ``rmsnorm_with_input_scale`` form — applied inside the same single
            fused kernel. Inference/forward-only: the call raises when any
            input requires grad.
        gain_center: Additive center for ``weight`` (e.g. 1.0 for
            zero-centered gamma), folded in-kernel. Only meaningful with
            ``weight``.

    Returns:
        [B, DIM] output matching ``cute_rmsnorm_fwd`` (with
        ``input_scale=True`` when ``weight`` is given) followed by
        ``scale_and_sum`` bit-for-bit when ``require_bitwise=True``.

    Raises:
        ValueError: If the input is outside the fused contract.
        NotImplementedError: If an input-scale weight requires gradients.
    """
    if (
        not require_bitwise
        and x.shape[-1] > FusedRMSNormCombineBwd._MAX_DIM
        and torch.is_grad_enabled()
        and (x.requires_grad or weights.requires_grad)
    ):
        raise ValueError(
            "non-bitwise fused RMSNorm dimensions above "
            f"{FusedRMSNormCombineBwd._MAX_DIM} are inference-only"
        )
    if (
        weight is not None
        and torch.is_grad_enabled()
        and (x.requires_grad or weights.requires_grad or weight.requires_grad)
    ):
        raise NotImplementedError(
            "the input-scale weight of fused_rmsnorm_combine is "
            "inference/forward-only; no backward supports it"
        )
    return _FusedRMSNormCombine.apply(
        x,
        weights,
        eps,
        norm_output_dtype,
        output_dtype,
        require_bitwise,
        use_kahan,
        weight,
        gain_center,
    )
