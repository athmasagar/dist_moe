# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch geometry and resource policy for distributed blockscaled GEMM."""

from __future__ import annotations

import logging

import torch

from ..formats import CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
from .activation_buffer import _ceil_div
from .blockscaled_grouped_gemm import (
    _host_tensormaps_available,
    _ParsedGemmProblem,
    _torch_dtype,
    auto_blockscaled_config,
    blockscaled_staged_decode_config,
    blockscaled_training_config,
    BlockScaledFormatSpec,
    inference_fprop_static_scheduler,
    MXFP4,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
)
from .config import (
    _blockscaled_ab_stage_storage_bytes,
    BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES,
    BLOCKSCALED_DISPATCH_DIM_ALIGNMENT,
    blockscaled_epilogue_subtile_divisor,
    BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT,
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    derive_blockscaled_dist_combine_config,
    derive_blockscaled_epilogue_tile_n,
    derive_blockscaled_num_tmem_buffers,
    EPILOGUE_SUBTILE_AUTO,
    SM100_TMEM_COLS,
    SMEM_LAUNCH_SAFETY_MARGIN_BYTES,
    uses_paged_blockscaled_scale_rows,
)
from .dist_blockscaled_grouped_gemm_kernel import (
    _FP4_FORMAT_NAMES,
    DistBlockScaledGroupedGemmKernel,
)

logger: logging.Logger = logging.getLogger(__name__)
_DEFAULT_M_MULTIPLE_OF = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF


def _blockscaled_split_size_alignment(m_multiple_of: int) -> int:
    return (
        m_multiple_of
        if uses_paged_blockscaled_scale_rows(m_multiple_of)
        else _DEFAULT_M_MULTIPLE_OF
    )


def _use_parallel_dense_epilogue(config: dict, *, a_width: int) -> bool:
    """Select parallel transfers for overlapping dense BF16 epilogues."""
    block_m = int(config["BLOCK_SIZE_M"]) // int(config["NUM_MMAS"])
    if (
        not config.get("OVERLAPPING_ACCUM", False)
        or not config.get("SWAP_AB", False)
        or config["NUM_CTAS"] != 2
        or block_m != 256
    ):
        return False
    epilogue_n = (
        derive_blockscaled_epilogue_tile_n(
            block_m=block_m,
            block_n=int(config["BLOCK_SIZE_N"]),
            num_ctas=2,
            c_stage_dtype=torch.bfloat16,
        )
        if config["EPILOGUE_SUBTILE"] == EPILOGUE_SUBTILE_AUTO
        else int(config["BLOCK_SIZE_N"])
        // blockscaled_epilogue_subtile_divisor(
            epilogue_subtile=int(config["EPILOGUE_SUBTILE"]),
            c_width=16,
            a_width=a_width,
            full_width=bool(config.get("EPILOGUE_TILE_FULL_WIDTH", False)),
        )
    )
    return epilogue_n >= 16 and epilogue_n % 16 == 0


def _validate_swap_ab_row_multiple(config: dict, m_multiple_of: int) -> None:
    if not config.get("SWAP_AB", False):
        return
    block_size_n = int(config["BLOCK_SIZE_N"])
    if not uses_paged_blockscaled_scale_rows(block_size_n):
        if block_size_n % 128 != 0:
            raise ValueError(
                "SWAP_AB block-scaled scale storage supports BLOCK_SIZE_N "
                f"32, 64, or a multiple of 128; got BLOCK_SIZE_N={block_size_n}"
            )
        return
    if m_multiple_of != block_size_n:
        raise ValueError(
            "SWAP_AB paged scale storage requires m_multiple_of to equal "
            f"BLOCK_SIZE_N; got m_multiple_of={m_multiple_of}, "
            f"BLOCK_SIZE_N={block_size_n}"
        )


def _make_dist_combine_config(
    *,
    config: dict,
    format: BlockScaledFormatSpec,
    auto_selected: bool,
    c_stage_dtype: torch.dtype,
    num_c_stages: int,
) -> dict:
    cfg = dict(config)
    cfg["NUM_C_STAGES"] = num_c_stages
    if auto_selected and format.name not in _FP4_FORMAT_NAMES:
        cfg = derive_blockscaled_dist_combine_config(
            cfg,
            sf_vec_size=format.sf_vec_size,
            a_dtype=_torch_dtype(format.a_dtype),
            b_dtype=_torch_dtype(format.b_dtype),
            c_stage_dtype=c_stage_dtype,
            num_c_stages=num_c_stages,
        )
    return cfg


def _staged_use_device_tensormaps(
    cfg: dict,
    *,
    activation_buffer: torch.Tensor | None,
    problem_type: int,
    m_multiple_of: int,
    n: int,
    groups: int,
) -> bool:
    return (
        (cfg.get("SWAP_AB", False) and cfg.get("BLOCK_SIZE_N") in (32, 64))
        or activation_buffer is not None
        or not _host_tensormaps_available(
            cfg,
            problem_type=problem_type,
            m_multiple_of=m_multiple_of,
            N=n,
            G=groups,
        )
    )


def _staged_fp4_mma_atom_n(
    *,
    config: dict,
    format: BlockScaledFormatSpec,
    n: int,
    k: int,
    mixed_operand_widths: bool,
) -> int:
    block_n = int(config["BLOCK_SIZE_N"])
    if mixed_operand_widths or (
        format is MXFP4 and (n, k) in ((8192, 4096), (4096, 4096))
    ):
        return block_n
    preferred_atom_n = min(128, block_n)
    return preferred_atom_n if block_n % preferred_atom_n == 0 else block_n


def _can_use_fused_dispatch_quant(
    *,
    rows: int,
    dim: int,
    dtype: torch.dtype,
    format: BlockScaledFormatSpec,
    blockscaled_dispatch: bool = False,
    m_multiple_of: int = BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT,
) -> bool:
    supported_formats = (MXFP8_E4M3, MXFP8_E5M2, MXFP4)
    if blockscaled_dispatch:
        supported_formats += (NVFP4,)
    return (
        format in supported_formats
        and dtype in (torch.bfloat16, torch.float16)
        and m_multiple_of > 0
        and rows % m_multiple_of == 0
        and dim % BLOCKSCALED_DISPATCH_DIM_ALIGNMENT == 0
    )


def _can_use_fused_combine_swiglu_quant(
    *,
    rows: int,
    dim: int,
    dtype: torch.dtype,
    format: BlockScaledFormatSpec,
    allow_nvfp4: bool = False,
) -> bool:
    return _can_use_fused_dispatch_quant(
        rows=rows,
        dim=dim,
        dtype=dtype,
        format=format,
        blockscaled_dispatch=allow_nvfp4,
    )


# Measured crossover for the fused chunked-mega path (larger contraction shapes, EP4,
# 100-iter CUDA-graph means): high-throughput producers win -17.8% at 8,192
# routed rows and -36.8% at 16,384 vs the legacy topology, and are neutral
# at 2,048-4,096 rows, so the mega floor admits everything down to 16
# rows/SM while decode-size batches stay legacy.
_NVFP4_HIGH_THROUGHPUT_MEGA_MIN_ROWS_PER_SM: int = 16


# Staged crossover depends on the hidden dim (dispatch K / combine N): with
# 16,384 routed rows at EP4, high-throughput producers measured -10.8% on
# contraction (D=12288) and -4.4% on larger contraction (D=14336) but +18.7% on square
# (D=4096); 8,192 rows were +18% or worse on every shape.
_NVFP4_HIGH_THROUGHPUT_STAGED_LARGE_HIDDEN_MIN_DIM: int = 12288


_NVFP4_HIGH_THROUGHPUT_STAGED_LARGE_HIDDEN_MIN_ROWS_PER_SM: int = 64


def _use_nvfp4_high_throughput_producers(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    mega: bool = False,
) -> bool:
    # Small routed batches cannot amortize the extra producer state and work
    # groups; the floors are the measured crossovers documented at the
    # constants above. The mega floor still keeps decode-size batches on the
    # legacy topology.
    if mega:
        rows_per_sm_floor = _NVFP4_HIGH_THROUGHPUT_MEGA_MIN_ROWS_PER_SM
    elif hidden_dim >= _NVFP4_HIGH_THROUGHPUT_STAGED_LARGE_HIDDEN_MIN_DIM:
        rows_per_sm_floor = _NVFP4_HIGH_THROUGHPUT_STAGED_LARGE_HIDDEN_MIN_ROWS_PER_SM
    else:
        rows_per_sm_floor = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    return format is NVFP4 and rows >= num_sms * rows_per_sm_floor


def _staged_epilogue_shape(
    config: dict,
    *,
    format: BlockScaledFormatSpec,
    c_stage_dtype: torch.dtype,
) -> tuple[int, int]:
    block_m = int(config["BLOCK_SIZE_M"])
    block_n = int(config["BLOCK_SIZE_N"])
    num_ctas = int(config["NUM_CTAS"])
    epi_m = min(block_m // num_ctas, 128)
    epilogue_subtile = int(config["EPILOGUE_SUBTILE"])
    if epilogue_subtile == 0:
        epi_n = derive_blockscaled_epilogue_tile_n(
            block_m=block_m,
            block_n=block_n,
            num_ctas=num_ctas,
            c_stage_dtype=c_stage_dtype,
        )
    else:
        bytes_scale = max(1, c_stage_dtype.itemsize * 8 // format.a_dtype.width)
        epi_n = block_n // (epilogue_subtile * bytes_scale)
    return epi_m, epi_n


def _staged_c_smem_bytes(
    config: dict,
    *,
    format: BlockScaledFormatSpec,
    mode: int,
    c_stage_dtype: torch.dtype,
) -> tuple[int, int, int]:
    epi_m, epi_n = _staged_epilogue_shape(
        config,
        format=format,
        c_stage_dtype=c_stage_dtype,
    )
    num_c_stages = int(config["NUM_C_STAGES"])
    if mode == DistBlockScaledGroupedGemmKernel.DISPATCH_MODE:
        elements = epi_m * epi_n
        stage_stride = elements
    else:
        skew = DistBlockScaledGroupedGemmKernel._c_smem_skew(
            int(config["NUM_TMEM_BUFFERS"])
        )
        if bool(config.get("SWAP_AB", False)):
            major_stride = epi_m + skew
            elements = (epi_n - 1) * major_stride + epi_m
            stage_stride = epi_n * major_stride
        else:
            major_stride = epi_n + skew
            elements = (epi_m - 1) * major_stride + epi_n
            stage_stride = epi_m * major_stride
    elements += (num_c_stages - 1) * stage_stride
    return elements * c_stage_dtype.itemsize, epi_m, epi_n


def _staged_operand_smem_bytes(
    config: dict,
    *,
    format: BlockScaledFormatSpec,
) -> tuple[int, int, int, int]:
    block_m = int(config["BLOCK_SIZE_M"])
    block_n = int(config["BLOCK_SIZE_N"])
    block_k = int(config["BLOCK_SIZE_K"])
    num_ctas = int(config["NUM_CTAS"])
    num_smem = int(config["NUM_SMEM_BUFFERS"])
    cta_m = block_m // num_ctas
    sf_k_blocks = block_k // format.sf_vec_size
    needs_unpack_tma = format.a_dtype.width != format.b_dtype.width
    a_smem_width = (
        8 if needs_unpack_tma and format.a_dtype.width < 8 else format.a_dtype.width
    )
    b_smem_width = (
        8 if needs_unpack_tma and format.b_dtype.width < 8 else format.b_dtype.width
    )
    return (
        cta_m * block_k * a_smem_width // 8 * num_smem,
        block_n // num_ctas * block_k * b_smem_width // 8 * num_smem,
        cta_m * sf_k_blocks * num_smem,
        _ceil_div(block_n, 128) * 128 * sf_k_blocks * num_smem,
    )


def _staged_producer_storage_counts(
    *,
    format: BlockScaledFormatSpec,
    mode: int,
    blockscaled_dispatch: bool,
    nvfp4_high_throughput: bool,
    combine_swiglu_k: int | None,
) -> tuple[int, int]:
    _, dispatch_groups = DistBlockScaledGroupedGemmKernel._dispatch_quant_topology(
        format=format,
        blockscaled_dispatch=blockscaled_dispatch,
        nvfp4_high_throughput=nvfp4_high_throughput,
    )
    producer_modes = (
        DistBlockScaledGroupedGemmKernel.DISPATCH_MODE,
        DistBlockScaledGroupedGemmKernel.COMBINE_SWIGLU_FWD_MODE,
        DistBlockScaledGroupedGemmKernel.COMBINE_SWIGLU_BWD_MODE,
    )
    n_dispatch_tiles = dispatch_groups if mode in producer_modes else 0
    if (
        format is NVFP4
        and mode == DistBlockScaledGroupedGemmKernel.COMBINE_SWIGLU_FWD_MODE
    ):
        _, quant_groups = DistBlockScaledGroupedGemmKernel._nvfp4_swiglu_quant_topology(
            nvfp4_high_throughput=nvfp4_high_throughput,
            combine_swiglu_k=combine_swiglu_k,
        )
        n_dispatch_tiles = max(
            n_dispatch_tiles,
            quant_groups * DistBlockScaledGroupedGemmKernel.NVFP4_ROW_REDUCTION_SLOTS,
        )
    n_gather_ptrs = (
        dispatch_groups * format.sf_vec_size
        if (
            mode == DistBlockScaledGroupedGemmKernel.DISPATCH_MODE
            and blockscaled_dispatch
        )
        else 0
    )
    return n_dispatch_tiles, n_gather_ptrs


def _append_aligned_smem_regions(offset: int, regions: tuple[int, ...]) -> int:
    for region_bytes in regions:
        offset = _ceil_div(offset, 1024) * 1024 + region_bytes
    return offset


def _finish_staged_smem_allocation(
    config: dict,
    *,
    offset: int,
    groups: int,
) -> int:
    tensormap_bytes = max(5 * 128, 8 * ((groups + 1) // 2))
    offset = _ceil_div(offset, 128) * 128 + tensormap_bytes
    num_ctas = int(config["NUM_CTAS"])
    barrier_count = (
        2 * int(config["NUM_SMEM_BUFFERS"])
        + 2 * int(config["NUM_TMEM_BUFFERS"])
        + 2 * int(config["NUM_TILE_BUFFERS"])
        + (3 if num_ctas == 2 else 0)
        + int(bool(config.get("OVERLAPPING_ACCUM", False)))
    )
    return _ceil_div(offset + 8 * barrier_count + 4, 1024) * 1024


def _estimate_staged_dist_smem_bytes(
    config: dict,
    *,
    format: BlockScaledFormatSpec,
    groups: int,
    mode: int,
    blockscaled_dispatch: bool,
    nvfp4_high_throughput: bool,
    use_global_scale_inv: bool,
    combine_swiglu_k: int | None,
    c_stage_dtype: torch.dtype,
) -> int:
    block_m = int(config["BLOCK_SIZE_M"])
    block_n = int(config["BLOCK_SIZE_N"])
    num_ctas = int(config["NUM_CTAS"])
    num_tile = int(config["NUM_TILE_BUFFERS"])
    c_smem_bytes, epi_m, epi_n = _staged_c_smem_bytes(
        config,
        format=format,
        mode=mode,
        c_stage_dtype=c_stage_dtype,
    )
    cta_m = block_m // num_ctas
    offset = _append_aligned_smem_regions(
        0,
        (c_smem_bytes, *_staged_operand_smem_bytes(config, format=format)),
    )
    n_dispatch_tiles, n_gather_ptrs = _staged_producer_storage_counts(
        format=format,
        mode=mode,
        blockscaled_dispatch=blockscaled_dispatch,
        nvfp4_high_throughput=nvfp4_high_throughput,
        combine_swiglu_k=combine_swiglu_k,
    )
    if use_global_scale_inv:
        offset = _ceil_div(offset, 128) * 128
        offset += 4 * (block_n if config.get("SWAP_AB", False) else cta_m)
    offset += 4 * num_tile
    offset += 4 * n_dispatch_tiles
    offset += 8 * n_gather_ptrs
    offset += 8 * max(epi_m, epi_n)
    return _finish_staged_smem_allocation(config, offset=offset, groups=groups)


def _verify_staged_smem_estimate(
    kernel: DistBlockScaledGroupedGemmKernel,
    config: dict,
    *,
    format: BlockScaledFormatSpec,
    groups: int,
    mode: int,
    blockscaled_dispatch: bool,
    nvfp4_high_throughput: bool,
    use_global_scale_inv: bool,
    combine_swiglu_k: int | None,
    c_stage_dtype: torch.dtype,
) -> None:
    if kernel._smem_estimate_verified or not hasattr(kernel, "shared_storage"):
        return
    estimated = _estimate_staged_dist_smem_bytes(
        config,
        format=format,
        groups=groups,
        mode=mode,
        blockscaled_dispatch=blockscaled_dispatch,
        nvfp4_high_throughput=nvfp4_high_throughput,
        use_global_scale_inv=use_global_scale_inv,
        combine_swiglu_k=combine_swiglu_k,
        c_stage_dtype=c_stage_dtype,
    )
    actual = kernel.shared_storage.size_in_bytes()
    if estimated < actual:
        raise AssertionError(
            "staged block-scaled SMEM estimate undercounts kernel storage: "
            f"estimated={estimated}, actual={actual}, mode={mode}, config={config}"
        )
    if estimated > actual:
        logger.warning(
            "Staged block-scaled SMEM estimate is conservative: "
            f"estimated={estimated}, actual={actual}, mode={mode}, config={config}"
        )
    kernel._smem_estimate_verified = True


def _initialize_staged_pipeline_depths(
    config: dict,
    *,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    num_c_stages: int | None = None,
) -> dict:
    cfg = dict(config)
    if num_c_stages is None:
        num_c_stages = int(
            blockscaled_staged_decode_config(
                format,
                weight_dtype=weight_format,
            )["NUM_C_STAGES"]
        )
    cfg["NUM_C_STAGES"] = num_c_stages
    block_n = int(cfg["BLOCK_SIZE_N"])
    num_mmas = int(cfg["NUM_MMAS"])
    max_tmem = min(
        int(cfg["NUM_TMEM_BUFFERS"]),
        max(1, SM100_TMEM_COLS // (block_n * num_mmas)),
    )
    cfg["NUM_TMEM_BUFFERS"] = derive_blockscaled_num_tmem_buffers(
        block_m=int(cfg["BLOCK_SIZE_M"]),
        block_n=block_n,
        num_mmas=num_mmas,
        overlapping_accum=bool(cfg.get("OVERLAPPING_ACCUM", False)),
        max_tmem_buffers=max_tmem,
    )
    cfg["NUM_TILE_BUFFERS"] = int(cfg["NUM_TMEM_BUFFERS"]) + 1
    return cfg


def _uses_derived_staged_pipeline(
    config: dict,
    *,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
) -> bool:
    base_config = blockscaled_staged_decode_config(
        format,
        weight_dtype=weight_format,
    )
    geometry_keys = (
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "NUM_MMAS",
        "EPILOGUE_SUBTILE",
    )
    return any(int(config[key]) != int(base_config[key]) for key in geometry_keys)


def _staged_smem_search_limit(
    config: dict,
    *,
    format: BlockScaledFormatSpec,
    budget: int,
) -> int:
    cta_m = int(config["BLOCK_SIZE_M"]) // int(config["NUM_CTAS"])
    cta_n = int(config["BLOCK_SIZE_N"]) // int(config["NUM_CTAS"])
    block_k = int(config["BLOCK_SIZE_K"])
    a_dtype = _torch_dtype(format.a_dtype)
    b_dtype = _torch_dtype(format.b_dtype)
    per_stage_floor = _blockscaled_ab_stage_storage_bytes(
        block_mn=cta_m,
        block_k=block_k,
        dtype=a_dtype,
        other_dtype=b_dtype,
    ) + _blockscaled_ab_stage_storage_bytes(
        block_mn=cta_n,
        block_k=block_k,
        dtype=b_dtype,
        other_dtype=a_dtype,
    )
    return max(2, budget // max(1, per_stage_floor))


def derive_staged_blockscaled_pipeline_config(
    config: dict,
    *,
    format: BlockScaledFormatSpec,
    activation_format: BlockScaledFormatSpec | None = None,
    weight_format: BlockScaledFormatSpec | None = None,
    num_c_stages: int | None = None,
    groups: int,
    mode: int,
    blockscaled_dispatch: bool,
    nvfp4_high_throughput: bool,
    use_global_scale_inv: bool,
    combine_swiglu_k: int | None,
    c_stage_dtype: torch.dtype,
) -> dict:
    cfg = _initialize_staged_pipeline_depths(
        config,
        format=format if activation_format is None else activation_format,
        weight_format=weight_format,
        num_c_stages=num_c_stages,
    )
    budget = BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES - SMEM_LAUNCH_SAFETY_MARGIN_BYTES
    max_smem = min(
        max(8, int(cfg["NUM_SMEM_BUFFERS"])),
        _staged_smem_search_limit(cfg, format=format, budget=budget),
    )
    for num_smem in range(max_smem, 1, -1):
        cfg["NUM_SMEM_BUFFERS"] = num_smem
        if (
            _estimate_staged_dist_smem_bytes(
                cfg,
                format=format,
                groups=groups,
                mode=mode,
                blockscaled_dispatch=blockscaled_dispatch,
                nvfp4_high_throughput=nvfp4_high_throughput,
                use_global_scale_inv=use_global_scale_inv,
                combine_swiglu_k=combine_swiglu_k,
                c_stage_dtype=c_stage_dtype,
            )
            <= budget
        ):
            return cfg
    raise ValueError(
        "no staged block-scaled pipeline fits "
        f"BLOCK_M={cfg['BLOCK_SIZE_M']}, BLOCK_N={cfg['BLOCK_SIZE_N']}"
    )


def _use_t6_mxfp8_prefill_fc13_unroll(
    *,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    rows: int,
    num_sms: int,
    n: int,
    k: int,
    interleaved_fc13: bool,
) -> bool:
    # Two-way K-loop unrolling is retained only for measured contraction MXFP8 FC13
    # prefill; it adds register pressure without a win on other contractions.
    return (
        (weight_format is None or format is weight_format)
        and format is MXFP8_E4M3
        and interleaved_fc13
        and n == 2 * 3072
        and k == 12288
        and rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    )


def _dispatch_scale_copy_atom_aligned(
    *,
    format: BlockScaledFormatSpec,
    hidden_dim: int,
) -> bool:
    # The vector scale copy moves whole 4-column scale atoms and its tail
    # guard only predicates the first column of each atom, so the hidden dim
    # must cover a multiple of CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM scale
    # columns; partial atoms must stay on the scalar path.
    return hidden_dim % (CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM * format.sf_vec_size) == 0


def _select_blockscaled_config(
    *,
    problem: _ParsedGemmProblem,
    num_sms: int,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None,
    m_multiple_of: int,
    problem_type: int,
    activation_buffer: torch.Tensor | None,
    derive_pipeline_depth: bool,
) -> dict:
    """Pin the training config when persisting bundles to an activation buffer;
    otherwise defer to the auto selector (with the inference static scheduler)."""
    if activation_buffer is not None and not derive_pipeline_depth:
        return blockscaled_training_config(format)
    return auto_blockscaled_config(
        GM=problem.GM,
        G=problem.G,
        N=problem.N,
        K=problem.K,
        num_sms=num_sms,
        format=format,
        weight_format=weight_format,
        m_multiple_of=m_multiple_of,
        problem_type=problem_type,
        static_scheduler=inference_fprop_static_scheduler(
            inference_mode=derive_pipeline_depth,
            format=format,
            weight_format=weight_format,
            N=problem.N,
            K=problem.K,
            problem_type=problem_type,
        ),
    )
