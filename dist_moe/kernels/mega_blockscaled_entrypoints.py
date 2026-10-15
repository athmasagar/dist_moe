# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch and device entrypoints for Mega blockscaled grouped GEMM."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from ._dsl_compat import thread_exit
from .activation_buffer import (
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_COL_Q_OFFSET,
    ACTIVATION_COL_SCALE_OFFSET,
    ACTIVATION_SOURCE_X_OFFSET,
    ACTIVATION_SOURCE_Y_OFFSET,
    MEGA_FIRST_GEMM_OFFSET_BASE,
    MEGA_SECOND_GEMM_OFFSET_BASE,
    MISSING_ACTIVATION_OFFSET,
)
from .activation_buffer_kernel import (
    _activation_buffer_col_scale_storage_byte_extent,
    _activation_buffer_row_scale_storage_byte_extent,
    _activation_buffer_tensor_with_byte_extent,
)
from .blockscaled_gemm_tiles import (
    _prepare_blockscaled_problem_tensormaps,
)
from .blockscaled_grouped_gemm_kernel import (
    _activation_buffer_operand_offsets,
    _activation_buffer_pointer_strides,
)
from .combine_swiglu_quant import (
    _combine_swiglu_bwd_quant_producer_body,
)
from .gemm_warp_roles import (
    _mega_mma_consumer_body,
    _mega_problem_specs,
    _mega_tma_producer_body,
)
from .grouped_gemm_kernel import (
    _activation_buffer_rows,
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
    _swap_if,
    _transpose_c_if_swap,
)
from .params import (
    BlockscaledGemmKernelParams as _MegaGemmKernelParams,
    BlockscaledPointerStrideArgs,
    BlockscaledTensormapParams as _MegaTensormapParams,
    BlockscaledTensormapProblem,
    CombineSwigluQuantParams,
    DispatchQuantSync as _MegaDispatchQuantSync,
    GroupedGemmPipelineSync as _MegaPipelineSync,
    GroupedGemmProblem as _MegaProblem,
    MegaPipelineParams,
    params_from_kernel,
    swap_ab_pointer_strides,
)
from .tile_scheduler import stage_expert_metadata
from .weight_borrow import WeightBorrowArgs


@cute.kernel
def prepare_dispatch_bprop_kernel(  # noqa: C901
    self,
    dgrad: _MegaTensormapParams,
    wgrad: _MegaTensormapParams,
    split_sizes: cute.Tensor,
    counter_ptr: cute.Pointer,
    tensormaps: cute.Tensor,
    G: cutlass.Constexpr[int],
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    extra_counter_ptr: cute.Pointer,
    extra_counter_count: cutlass.Int32,
    activation_buffer_base_ptr: cutlass.Int64,
    activation_buffer_size_bytes: cutlass.Int64,
    activation_offsets: cute.Tensor,
    use_activation_buffer: cutlass.Constexpr[bool],
    weight_borrow_prepare=None,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    prepare_g, prepare_w, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    storage = smem.allocate(self.prepare_shared_storage)
    tensormap_buffer_ptr = storage.tensormap_buffer.data_ptr()
    dgrad_tensormap_smem_ptr = tensormap_buffer_ptr
    wgrad_tensormap_smem_ptr = (
        tensormap_buffer_ptr + 16 * self.MEGA_TENSORMAP_DESCRIPTOR_COUNT // 2
    )

    tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)

    if (
        prepare_g == cutlass.Int32(0)
        and prepare_w == cutlass.Int32(0)
        and tidx == cutlass.Int32(0)
    ):
        cute.arch.store(counter_ptr, cutlass.Int32(0))

    extra_counters = cute.make_tensor(
        extra_counter_ptr,
        cute.make_ordered_layout((extra_counter_count,), order=(0,)),
    )
    block_dim_x, _, _ = cute.arch.block_dim()
    counter_zero_thread_idx = (
        prepare_w * cutlass.Int32(G) + prepare_g
    ) * block_dim_x + tidx
    counter_zero_threads = cutlass.Int32(G) * cutlass.Int32(2) * block_dim_x
    zero_iters = (
        extra_counter_count + counter_zero_threads - cutlass.Int32(1)
    ) // counter_zero_threads
    for i in cutlass.range(zero_iters, unroll=1):
        counter_idx = i * counter_zero_threads + counter_zero_thread_idx
        if counter_idx < extra_counter_count:
            extra_counters[counter_idx] = cutlass.Int32(0)

    tensormap_smem_ptr_pair = (
        dgrad_tensormap_smem_ptr,
        wgrad_tensormap_smem_ptr,
    )
    gemm_mnk_pair, problem_type_pair = _mega_problem_specs(
        M,
        N,
        K,
        MEGA_FORWARD_MODE=self.MEGA_FORWARD_MODE,
        mode=self.mode,
        MEGA_DGRAD_PROBLEM_TYPE=self.MEGA_DGRAD_PROBLEM_TYPE,
        MEGA_WGRAD_PROBLEM_TYPE=self.MEGA_WGRAD_PROBLEM_TYPE,
    )
    tensormap_base_pair = (
        self.MEGA_DGRAD_TENSORMAP_BASE,
        self.MEGA_WGRAD_TENSORMAP_BASE,
    )
    activation_offset_base_pair = (
        MEGA_FIRST_GEMM_OFFSET_BASE,
        MEGA_SECOND_GEMM_OFFSET_BASE,
    )
    for w in cutlass.range_constexpr(2):
        if prepare_w == cutlass.Int32(w):
            params = (dgrad, wgrad)[w]
            activation_buffer_operand_offsets = (
                cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                cutlass.Int64(MISSING_ACTIVATION_OFFSET),
            )
            if cutlass.const_expr(use_activation_buffer):
                activation_buffer_operand_offsets = _activation_buffer_operand_offsets(
                    activation_offsets,
                    activation_offset_base_pair[w],
                    self.SWAP_AB,
                )
                pointer_strides = _activation_buffer_pointer_strides(
                    (params.base_ptrs, params.strides, params.sf_strides),
                    activation_buffer_base_ptr,
                    activation_offsets,
                    activation_offset_base_pair[w],
                    self.SWAP_AB,
                )
                base_ptrs, tensor_strides, sf_strides = pointer_strides
                params = _MegaTensormapParams(
                    params.tma_atom_a,
                    params.tma_atom_b,
                    params.tma_atom_sfa,
                    params.tma_atom_sfb,
                    params.tma_atom_c,
                    base_ptrs=base_ptrs,
                    strides=tensor_strides,
                    sf_strides=sf_strides,
                    elem_sizes=params.elem_sizes,
                    dtypes=params.dtypes,
                )
            prob_m, prob_n, prob_k = gemm_mnk_pair[w]
            problem = BlockscaledTensormapProblem(
                split_sizes=split_sizes,
                groups=G,
                mnk=(prob_m, prob_n, prob_k),
                problem_type=problem_type_pair[w],
                tensormap_base=tensormap_base_pair[w],
                prepare_g=prepare_g,
                warp_idx=warp_idx,
            )
            borrow_a_base = cutlass.Int64(0)
            borrow_a_row_bytes = cutlass.Int64(0)
            borrow_sfa_base = cutlass.Int64(0)
            borrow_sfa_row_bytes = cutlass.Int64(0)
            if cutlass.const_expr(self.WEIGHT_BORROW_SLOTS > 0):
                borrow_a_base = weight_borrow_prepare[4 * w + 0]
                borrow_a_row_bytes = weight_borrow_prepare[4 * w + 1]
                borrow_sfa_base = weight_borrow_prepare[4 * w + 2]
                borrow_sfa_row_bytes = weight_borrow_prepare[4 * w + 3]
            _prepare_blockscaled_problem_tensormaps(
                params=params,
                problem=problem,
                tensormaps=tensormaps,
                tensormap_manager=tensormap_manager,
                tensormap_smem_ptr_base=tensormap_smem_ptr_pair[w],
                activation_buffer_operand_offsets=(activation_buffer_operand_offsets),
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                use_activation_buffer=use_activation_buffer,
                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                SWAP_AB=self.SWAP_AB,
                sf_dtype=self.sf_dtype,
                sf_vec_size=self.sf_vec_size,
                ring_b_rows=self.ACTIVATION_RING_B_ROWS,
                use_sm103_ultra=self.USE_SM103_ULTRA,
                borrow_slots=self.WEIGHT_BORROW_SLOTS,
                borrow_a_base=borrow_a_base,
                borrow_a_row_bytes=borrow_a_row_bytes,
                borrow_sfa_base=borrow_sfa_base,
                borrow_sfa_row_bytes=borrow_sfa_row_bytes,
            )


@cute.jit
def __call__(  # noqa: C901
    self,
    tensor_a: cute.Tensor,
    tensor_b: cute.Tensor,
    tensor_c: cute.Tensor,
    tensor_sfa: cute.Tensor,
    tensor_sfb: cute.Tensor,
    wgrad_tensor_a: cute.Tensor,
    wgrad_tensor_b: cute.Tensor,
    wgrad_tensor_c: cute.Tensor,
    wgrad_tensor_sfa: cute.Tensor,
    wgrad_tensor_sfb: cute.Tensor,
    split_sizes: cute.Tensor,
    counter: cute.Tensor,
    tensormaps: cute.Tensor,
    gather_ptrs: cute.Tensor,
    dispatch_quant_work_counter: cute.Tensor,
    dispatch_quant_done_counter: cute.Tensor,
    dispatch_quant_row_q_words: cute.Tensor,
    dispatch_quant_row_scale: cute.Tensor,
    dispatch_quant_row_global_scale_inv: cute.Tensor,
    dispatch_quant_col_q_words: cute.Tensor,
    dispatch_quant_col_scale: cute.Tensor,
    combine_scatter_ptrs: cute.Tensor,
    combine_dz: cute.Tensor,
    combine_h1: cute.Tensor,
    activation_gather_ptrs: cute.Tensor,
    activation_quant_work_counter: cute.Tensor,
    activation_quant_done_counter: cute.Tensor,
    activation_quant_row_q_words: cute.Tensor,
    activation_quant_row_scale: cute.Tensor,
    activation_quant_row_global_scale_inv: cute.Tensor,
    activation_quant_col_q_words: cute.Tensor,
    activation_quant_col_scale: cute.Tensor,
    dgrad_pointer_strides: BlockscaledPointerStrideArgs,
    wgrad_pointer_strides: BlockscaledPointerStrideArgs,
    dgrad_elem_sizes: cutlass.Constexpr[tuple[int, int, int]],
    wgrad_elem_sizes: cutlass.Constexpr[tuple[int, int, int]],
    wgrad_c_dtype: cutlass.Constexpr[type[cutlass.Numeric]],
    wgrad_output_accum: cutlass.Constexpr[bool],
    G: cutlass.Constexpr[int],
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    local_rank: cutlass.Int32,
    num_clusters: int,
    dispatch_quant_total_tiles: cutlass.Int32,
    dispatch_quant_counter_zero_count: cutlass.Int32,
    row_done_counter_offset: cutlass.Int32,
    col_done_counter_offset: cutlass.Int32,
    activation_quant_total_tiles: cutlass.Int32,
    activation_row_done_counter_offset: cutlass.Int32,
    activation_col_done_counter_offset: cutlass.Int32,
    combine: cutlass.Constexpr[bool],
    activation_dispatch: cutlass.Constexpr[bool],
    use_device_tensormaps: cutlass.Constexpr[bool],
    activation_buffer_base_ptr: cutlass.Int64,
    activation_buffer_size_bytes: cutlass.Int64,
    activation_offsets: cute.Tensor,
    use_activation_buffer: cutlass.Constexpr[bool],
    conditional_execution: cute.Tensor,
    use_conditional_execution: cutlass.Constexpr[bool],
    dgrad_b_global_scale_inv_ptr: cutlass.Int64,
    wgrad_b_global_scale_inv_ptr: cutlass.Int64,
    use_global_scale_inv: cutlass.Constexpr[bool],
    nvfp4_recip_lut_ptr: cutlass.Int64,
    stream: cuda.CUstream,
    weight_borrow_window_ptrs: cute.Tensor | None = None,
    weight_borrow_src_rank: cute.Tensor | None = None,
    weight_borrow_src_slot: cute.Tensor | None = None,
    weight_borrow_scalars=None,
    weight_borrow_prepare=None,
):
    # Tensors cross the launch boundary as top-level args; re-bundle them
    # with the scalar half for the kernel-side plumbing.
    weight_borrow_args = None
    if cutlass.const_expr(weight_borrow_window_ptrs is not None):
        s = weight_borrow_scalars
        weight_borrow_args = WeightBorrowArgs(
            window_ptrs=weight_borrow_window_ptrs,
            src_rank=weight_borrow_src_rank,
            src_slot=weight_borrow_src_slot,
            slot_buf_base=s[0],
            w13_win_off=s[1],
            w13_slot_off=s[2],
            w13_row_bytes=s[3],
            s13_win_off=s[4],
            s13_slot_off=s[5],
            s13_row_bytes=s[6],
            w2_win_off=s[7],
            w2_slot_off=s[8],
            w2_row_bytes=s[9],
            s2_win_off=s[10],
            s2_slot_off=s[11],
            s2_row_bytes=s[12],
        )
    eff_tensor_a, eff_tensor_b = _swap_if(self.SWAP_AB, tensor_a, tensor_b)
    eff_tensor_sfa, eff_tensor_sfb = _swap_if(self.SWAP_AB, tensor_sfa, tensor_sfb)
    dgrad_eff_pointer_strides = swap_ab_pointer_strides(
        dgrad_pointer_strides,
        self.SWAP_AB,
    )
    dgrad_elem_size_a, dgrad_elem_size_b, dgrad_elem_size_c = dgrad_elem_sizes
    eff_elem_a, eff_elem_b = _swap_if(
        self.SWAP_AB,
        dgrad_elem_size_a,
        dgrad_elem_size_b,
    )
    dgrad_eff_elem_sizes = (eff_elem_a, eff_elem_b, dgrad_elem_size_c)
    wgrad_eff_tensor_a, wgrad_eff_tensor_b = _swap_if(
        self.SWAP_AB,
        wgrad_tensor_a,
        wgrad_tensor_b,
    )
    wgrad_eff_tensor_sfa, wgrad_eff_tensor_sfb = _swap_if(
        self.SWAP_AB,
        wgrad_tensor_sfa,
        wgrad_tensor_sfb,
    )
    wgrad_eff_pointer_strides = swap_ab_pointer_strides(
        wgrad_pointer_strides,
        self.SWAP_AB,
    )
    wgrad_elem_size_a, wgrad_elem_size_b, wgrad_elem_size_c = wgrad_elem_sizes
    wgrad_eff_elem_a, wgrad_eff_elem_b = _swap_if(
        self.SWAP_AB,
        wgrad_elem_size_a,
        wgrad_elem_size_b,
    )
    wgrad_eff_elem_sizes = (
        wgrad_eff_elem_a,
        wgrad_eff_elem_b,
        wgrad_elem_size_c,
    )

    tensor_c_eff = _transpose_c_if_swap(tensor_c, self.SWAP_AB)
    wgrad_tensor_c_eff = _transpose_c_if_swap(wgrad_tensor_c, self.SWAP_AB)

    dgrad_layout = self._setup_gemm_layout_bundle(
        tensor_a=eff_tensor_a,
        tensor_b=eff_tensor_b,
        tensor_c=tensor_c,
        tensor_c_eff=tensor_c_eff,
    )
    wgrad_layout = self._setup_gemm_layout_bundle(
        tensor_a=wgrad_eff_tensor_a,
        tensor_b=wgrad_eff_tensor_b,
        tensor_c=wgrad_tensor_c,
        tensor_c_eff=wgrad_tensor_c_eff,
    )

    dgrad_tma_atoms, dgrad_tma_tensors = self._make_first_gemm_tma_atoms(
        layout=dgrad_layout,
        tensor_a=eff_tensor_a,
        tensor_b=eff_tensor_b,
        tensor_c_eff=tensor_c_eff,
        tensor_sfa=eff_tensor_sfa,
        tensor_sfb=eff_tensor_sfb,
        output_accum=False,
    )
    wgrad_tma_atoms, wgrad_tma_tensors = self._make_gemm_tma_atoms(
        layout=wgrad_layout,
        tensor_a=wgrad_eff_tensor_a,
        tensor_b=wgrad_eff_tensor_b,
        tensor_c_eff=wgrad_tensor_c_eff,
        tensor_sfa=wgrad_eff_tensor_sfa,
        tensor_sfb=wgrad_eff_tensor_sfb,
        output_accum=wgrad_output_accum,
    )
    dgrad_kernel_params = self._make_gemm_kernel_params(
        dgrad_layout,
        dgrad_tma_atoms,
        dgrad_tma_tensors,
    )
    wgrad_kernel_params = self._make_gemm_kernel_params(
        wgrad_layout,
        wgrad_tma_atoms,
        wgrad_tma_tensors,
    )
    dgrad_tensormap_params = self._make_tensormap_params(
        dgrad_tma_atoms,
        dgrad_eff_pointer_strides,
        dgrad_eff_elem_sizes,
        (dgrad_layout.a_dtype, dgrad_layout.b_dtype, dgrad_layout.c_dtype),
    )
    wgrad_tensormap_params = self._make_tensormap_params(
        wgrad_tma_atoms,
        wgrad_eff_pointer_strides,
        wgrad_eff_elem_sizes,
        (wgrad_layout.a_dtype, wgrad_layout.b_dtype, wgrad_c_dtype),
    )

    self.shared_storage = self._make_shared_storage(
        dgrad_layout,
        wgrad_layout,
        combine,
        G,
        use_a_global_scale_inv=use_global_scale_inv,
    )
    self.prepare_shared_storage = self._make_prepare_shared_storage()
    grid = (num_clusters * self.NUM_CTAS, 1, 1)

    fold_tensormap_prep = use_device_tensormaps and self.FOLD_TENSORMAP_PREP
    fold_kwargs = {}
    if cutlass.const_expr(fold_tensormap_prep):
        fold_kwargs = {
            "fold_dgrad": dgrad_tensormap_params,
            "fold_wgrad": wgrad_tensormap_params,
        }
    if cutlass.const_expr(use_device_tensormaps and not fold_tensormap_prep):
        self.prepare_dispatch_bprop_kernel(
            dgrad=dgrad_tensormap_params,
            wgrad=wgrad_tensormap_params,
            split_sizes=split_sizes,
            counter_ptr=counter.iterator,
            tensormaps=tensormaps,
            G=G,
            M=M,
            N=N,
            K=K,
            extra_counter_ptr=dispatch_quant_work_counter.iterator,
            extra_counter_count=dispatch_quant_counter_zero_count,
            activation_buffer_base_ptr=activation_buffer_base_ptr,
            activation_buffer_size_bytes=activation_buffer_size_bytes,
            activation_offsets=activation_offsets,
            use_activation_buffer=use_activation_buffer,
            weight_borrow_prepare=weight_borrow_prepare,
        ).launch(
            grid=(G, 2, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    self.dispatch_bprop_kernel(
        dgrad=dgrad_kernel_params,
        wgrad=wgrad_kernel_params,
        split_sizes=split_sizes,
        counter_ptr=counter.iterator,
        tensormaps=tensormaps,
        mDispatchQuantGatherPtrs=gather_ptrs,
        mDispatchQuantWorkCounter=dispatch_quant_work_counter,
        mDispatchQuantDoneCounter=dispatch_quant_done_counter,
        mDispatchQuantRowQWords=dispatch_quant_row_q_words,
        mDispatchQuantRowScale=dispatch_quant_row_scale,
        mDispatchQuantRowGlobalScaleInv=(dispatch_quant_row_global_scale_inv),
        mDispatchQuantColQWords=dispatch_quant_col_q_words,
        mDispatchQuantColScale=dispatch_quant_col_scale,
        mCombineScatterPtrs=combine_scatter_ptrs,
        mCombineDz=combine_dz,
        mCombineH1=combine_h1,
        row_done_counter_offset=row_done_counter_offset,
        col_done_counter_offset=col_done_counter_offset,
        mActivationGatherPtrs=activation_gather_ptrs,
        mActivationQuantWorkCounter=activation_quant_work_counter,
        mActivationQuantDoneCounter=activation_quant_done_counter,
        mActivationQuantRowQWords=activation_quant_row_q_words,
        mActivationQuantRowScale=activation_quant_row_scale,
        mActivationQuantRowGlobalScaleInv=(activation_quant_row_global_scale_inv),
        mActivationQuantColQWords=activation_quant_col_q_words,
        mActivationQuantColScale=activation_quant_col_scale,
        activation_quant_total_tiles=activation_quant_total_tiles,
        activation_row_done_counter_offset=activation_row_done_counter_offset,
        activation_col_done_counter_offset=activation_col_done_counter_offset,
        G=G,
        M=M,
        N=N,
        K=K,
        local_rank=local_rank,
        dispatch_quant_total_tiles=dispatch_quant_total_tiles,
        combine=combine,
        activation_dispatch=activation_dispatch,
        use_device_tensormaps=use_device_tensormaps,
        activation_buffer_base_ptr=activation_buffer_base_ptr,
        activation_buffer_size_bytes=activation_buffer_size_bytes,
        activation_offsets=activation_offsets,
        use_activation_buffer=use_activation_buffer,
        conditional_execution=conditional_execution,
        use_conditional_execution=use_conditional_execution,
        dgrad_b_global_scale_inv_ptr=dgrad_b_global_scale_inv_ptr,
        wgrad_b_global_scale_inv_ptr=wgrad_b_global_scale_inv_ptr,
        use_global_scale_inv=use_global_scale_inv,
        nvfp4_recip_lut_ptr=nvfp4_recip_lut_ptr,
        weight_borrow_args=weight_borrow_args,
        **fold_kwargs,
    ).launch(
        grid=grid,
        block=(self.THREADS_PER_CTA, 1, 1),
        cluster=(*self.cluster_shape_mn, 1),
        stream=stream,
    )


@cute.kernel
def dispatch_bprop_kernel(  # noqa: C901
    self,
    dgrad: _MegaGemmKernelParams,
    wgrad: _MegaGemmKernelParams,
    split_sizes: cute.Tensor,
    counter_ptr: cute.Pointer,
    tensormaps: cute.Tensor,
    mDispatchQuantGatherPtrs: cute.Tensor,
    mDispatchQuantWorkCounter: cute.Tensor,
    mDispatchQuantDoneCounter: cute.Tensor,
    mDispatchQuantRowQWords: cute.Tensor,
    mDispatchQuantRowScale: cute.Tensor,
    mDispatchQuantRowGlobalScaleInv: cute.Tensor,
    mDispatchQuantColQWords: cute.Tensor,
    mDispatchQuantColScale: cute.Tensor,
    mCombineScatterPtrs: cute.Tensor,
    mCombineDz: cute.Tensor,
    mCombineH1: cute.Tensor,
    mActivationGatherPtrs: cute.Tensor,
    mActivationQuantWorkCounter: cute.Tensor,
    mActivationQuantDoneCounter: cute.Tensor,
    mActivationQuantRowQWords: cute.Tensor,
    mActivationQuantRowScale: cute.Tensor,
    mActivationQuantRowGlobalScaleInv: cute.Tensor,
    mActivationQuantColQWords: cute.Tensor,
    mActivationQuantColScale: cute.Tensor,
    row_done_counter_offset: cutlass.Int32,
    col_done_counter_offset: cutlass.Int32,
    activation_quant_total_tiles: cutlass.Int32,
    activation_row_done_counter_offset: cutlass.Int32,
    activation_col_done_counter_offset: cutlass.Int32,
    G: cutlass.Constexpr[int],
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    local_rank: cutlass.Int32,
    dispatch_quant_total_tiles: cutlass.Int32,
    combine: cutlass.Constexpr[bool],
    activation_dispatch: cutlass.Constexpr[bool],
    use_device_tensormaps: cutlass.Constexpr[bool],
    activation_buffer_base_ptr: cutlass.Int64,
    activation_buffer_size_bytes: cutlass.Int64,
    activation_offsets: cute.Tensor,
    use_activation_buffer: cutlass.Constexpr[bool],
    conditional_execution: cute.Tensor,
    use_conditional_execution: cutlass.Constexpr[bool],
    dgrad_b_global_scale_inv_ptr: cutlass.Int64,
    wgrad_b_global_scale_inv_ptr: cutlass.Int64,
    use_global_scale_inv: cutlass.Constexpr[bool],
    nvfp4_recip_lut_ptr: cutlass.Int64,
    weight_borrow_args=None,
):
    if cutlass.const_expr(use_conditional_execution):
        if conditional_execution[0] == cutlass.Int32(0):
            thread_exit()
    if cutlass.const_expr(use_activation_buffer):
        activation_rows = _activation_buffer_rows(split_sizes, G)
        dispatch_value_count = cutlass.Int64(activation_rows) * cutlass.Int64(K)
        dispatch_q_byte_extent = (
            dispatch_value_count
            * cutlass.Int64(cutlass.const_expr(self.format.a_dtype.width))
            // cutlass.Int64(8)
        )
        dispatch_row_scale_byte_extent = (
            _activation_buffer_row_scale_storage_byte_extent(
                activation_rows,
                K,
                sf_vec_size=self.sf_vec_size,
                sf_dtype_width=self.format.sf_dtype.width,
            )
        )
        dispatch_col_scale_byte_extent = (
            _activation_buffer_col_scale_storage_byte_extent(
                activation_rows,
                K,
                sf_vec_size=self.sf_vec_size,
                sf_dtype_width=self.format.sf_dtype.width,
            )
        )
        activation_value_count = cutlass.Int64(activation_rows) * cutlass.Int64(N)
        activation_q_byte_extent = (
            activation_value_count
            * cutlass.Int64(cutlass.const_expr(self.format.a_dtype.width))
            // cutlass.Int64(8)
        )
        activation_row_scale_byte_extent = (
            _activation_buffer_row_scale_storage_byte_extent(
                activation_rows,
                N,
                sf_vec_size=self.sf_vec_size,
                sf_dtype_width=self.format.sf_dtype.width,
            )
        )
        activation_col_scale_byte_extent = (
            _activation_buffer_col_scale_storage_byte_extent(
                activation_rows,
                N,
                sf_vec_size=self.sf_vec_size,
                sf_dtype_width=self.format.sf_dtype.width,
            )
        )
        mDispatchQuantRowQWords = _activation_buffer_tensor_with_byte_extent(
            mDispatchQuantRowQWords,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_A_Q_OFFSET,
            cutlass.Uint32,
            dispatch_q_byte_extent,
        )
        mDispatchQuantRowScale = _activation_buffer_tensor_with_byte_extent(
            mDispatchQuantRowScale,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_A_SCALE_OFFSET,
            cutlass.Uint8,
            dispatch_row_scale_byte_extent,
        )
        mDispatchQuantColQWords = _activation_buffer_tensor_with_byte_extent(
            mDispatchQuantColQWords,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_COL_Q_OFFSET,
            cutlass.Uint32,
            dispatch_q_byte_extent,
        )
        mDispatchQuantColScale = _activation_buffer_tensor_with_byte_extent(
            mDispatchQuantColScale,
            activation_buffer_base_ptr,
            activation_buffer_size_bytes,
            activation_offsets,
            MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_COL_SCALE_OFFSET,
            cutlass.Uint8,
            dispatch_col_scale_byte_extent,
        )
        if cutlass.const_expr(combine):
            source_element_size = cutlass.Int64(
                cutlass.const_expr(mCombineDz.element_type.width // 8)
            )
            if cutlass.const_expr(self.mode == self.MEGA_FORWARD_MODE):
                has_activation_rows = cutlass.Int64(activation_rows > cutlass.Int32(0))
                source_byte_extent = (
                    has_activation_rows
                    * (
                        cutlass.Int64(2) * cutlass.Int64(activation_rows)
                        - cutlass.Int64(1)
                    )
                    * cutlass.Int64(N)
                    * source_element_size
                )
                source_x_byte_extent = source_byte_extent
                source_y_byte_extent = source_byte_extent
            else:
                source_x_byte_extent = (
                    cutlass.Int64(activation_rows)
                    * cutlass.Int64(K // cutlass.Int32(2))
                    * source_element_size
                )
                source_y_byte_extent = (
                    cutlass.Int64(activation_rows)
                    * cutlass.Int64(K)
                    * source_element_size
                )
            mCombineDz = _activation_buffer_tensor_with_byte_extent(
                mCombineDz,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_SOURCE_X_OFFSET,
                mCombineDz.element_type,
                source_x_byte_extent,
            )
            mCombineH1 = _activation_buffer_tensor_with_byte_extent(
                mCombineH1,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_SOURCE_Y_OFFSET,
                mCombineH1.element_type,
                source_y_byte_extent,
            )
        if cutlass.const_expr(activation_dispatch):
            mActivationQuantRowQWords = _activation_buffer_tensor_with_byte_extent(
                mActivationQuantRowQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_A_Q_OFFSET,
                cutlass.Uint32,
                activation_q_byte_extent,
            )
            mActivationQuantRowScale = _activation_buffer_tensor_with_byte_extent(
                mActivationQuantRowScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_A_SCALE_OFFSET,
                cutlass.Uint8,
                activation_row_scale_byte_extent,
            )
            mActivationQuantColQWords = _activation_buffer_tensor_with_byte_extent(
                mActivationQuantColQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_COL_Q_OFFSET,
                cutlass.Uint32,
                activation_q_byte_extent,
            )
            mActivationQuantColScale = _activation_buffer_tensor_with_byte_extent(
                mActivationQuantColScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_COL_SCALE_OFFSET,
                cutlass.Uint8,
                activation_col_scale_byte_extent,
            )
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    cluster_cta_rank = cutlass.Int32(0)
    pred_cta0 = True
    if cutlass.const_expr(self.NUM_CTAS == 2):
        cluster_cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pred_cta0 = cluster_cta_rank == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    ab_full_mbar = storage.smem_full_mbar.data_ptr()
    ab_empty_mbar = storage.smem_empty_mbar.data_ptr()
    tmem_full_mbar = storage.tmem_full_mbar.data_ptr()
    tmem_empty_mbar = storage.tmem_empty_mbar.data_ptr()
    tile_consumer_mbar = storage.tile_consumer_mbar.data_ptr()
    tile_producer_mbar = storage.tile_producer_mbar.data_ptr()
    tile_cta_bar_mbar = (
        storage.tile_cta_bar_mbar.data_ptr() if self.NUM_TILE_CTA_BARS > 0 else None
    )
    tmem_dealloc_mbar = (
        storage.tmem_dealloc_mbar.data_ptr() if self.NUM_CTAS == 2 else None
    )
    cross_seam_mbar = (
        storage.cross_seam_mbar.data_ptr() if self.OVERLAPPING_ACCUM else None
    )
    tile_id_smem_ptr = storage.tile_id_smem.data_ptr()
    dispatch_tile_id_smem_ptr = storage.dispatch_tile_id_smem.data_ptr()
    dispatch_ptr_smem_ptr = storage.dispatch_ptr_smem.data_ptr()
    scatter_ptr_smem_ptr = storage.dispatch_ptr_smem.data_ptr()
    if cutlass.const_expr(combine):
        scatter_ptr_smem_ptr = storage.scatter_ptr_smem.data_ptr()
    tmem_holding_buf = storage.tmem_holding_buf

    if warp_idx == self.EPILOG_WARP_IDS[0]:
        with cute.arch.elect_one():
            for i in range(self.NUM_SMEM_BUFFERS):
                cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                cute.arch.mbarrier_init(ab_full_mbar + i, 1)
            for i in range(self.NUM_TMEM_BUFFERS):
                cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                cute.arch.mbarrier_init(tmem_empty_mbar + i, self.NUM_CTAS)
            for i in range(self.NUM_TILE_BUFFERS):
                cute.arch.mbarrier_init(tile_consumer_mbar + i, 1)
                cute.arch.mbarrier_init(tile_producer_mbar + i, self.NUM_CTAS)
            if cutlass.const_expr(self.NUM_TILE_CTA_BARS > 0):
                for i in range(self.NUM_TILE_CTA_BARS):
                    cute.arch.mbarrier_init(tile_cta_bar_mbar + i, 1)
                cute.arch.mbarrier_init(tmem_dealloc_mbar, 32)
            if cutlass.const_expr(self.OVERLAPPING_ACCUM):
                cute.arch.mbarrier_init(cross_seam_mbar, self.NUM_CTAS)

    cute.arch.mbarrier_init_fence()
    if cutlass.const_expr(self.NUM_CTAS == 2):
        cute.arch.fence_proxy(kind="async.shared", space="cluster")
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    if warp_idx == self.EPILOG_WARP_IDS[0]:
        cute.arch.alloc_tmem(
            self.num_tmem_alloc_cols,
            tmem_holding_buf,
            is_two_cta=(self.NUM_CTAS == 2),
        )

    metadata_smem_ptr = cute.recast_ptr(
        storage.tensormap_buffer.data_ptr(), dtype=cutlass.Int32
    )
    split_sizes = stage_expert_metadata(
        split_sizes,
        metadata_smem_ptr,
        G,
        synchronize=False,
    )
    work_info_smem_ptr = metadata_smem_ptr + cutlass.Int32(G)
    cute.arch.barrier(
        barrier_id=_BAR_FULL_CTA_SYNC,
        number_of_threads=self.THREADS_PER_CTA,
    )
    bid = cute.arch.block_idx()
    dgrad_state = self._make_gemm_prologue(
        dgrad,
        storage,
        cluster_cta_rank,
        bid,
    )
    wgrad_state = self._make_gemm_prologue(
        wgrad,
        storage,
        cluster_cta_rank,
        bid,
    )

    a_full_mcast_mask = None
    b_full_mcast_mask = None
    sfa_full_mcast_mask = None
    sfb_full_mcast_mask = None
    ab_empty_mcast_mask = None
    acc_full_mcast_mask = None
    if cutlass.const_expr(self.NUM_CTAS == 2):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            dgrad.cluster_layout_vmnk,
            dgrad_state.block_in_cluster_coord_vmnk,
            mcast_mode=2,
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            dgrad.cluster_layout_vmnk,
            dgrad_state.block_in_cluster_coord_vmnk,
            mcast_mode=1,
        )
        sfa_full_mcast_mask = a_full_mcast_mask
        sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
            dgrad.cluster_layout_sfb_vmnk,
            dgrad_state.block_in_cluster_coord_sfb_vmnk,
            mcast_mode=1,
        )
        block_in_cluster_coord_vmnk_peer = (
            dgrad_state.block_in_cluster_coord_vmnk[0] ^ 1,
            *dgrad_state.block_in_cluster_coord_vmnk[1:],
        )
        a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
            dgrad.cluster_layout_vmnk,
            block_in_cluster_coord_vmnk_peer,
            mcast_mode=2,
        )
        b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
            dgrad.cluster_layout_vmnk,
            block_in_cluster_coord_vmnk_peer,
            mcast_mode=1,
        )
        ab_empty_mcast_mask = (
            a_full_mcast_mask
            | b_full_mcast_mask
            | a_full_mcast_mask_peer
            | b_full_mcast_mask_peer
        )
        acc_full_mcast_mask = cute.make_layout_image_mask(
            dgrad.cluster_layout_vmnk,
            dgrad_state.block_in_cluster_coord_vmnk,
            mode=0,
        )

    dgrad_num_tma_load_bytes, _, _ = self._num_tma_load_bytes(
        sA=dgrad_state.sA,
        sB=dgrad_state.sB,
        sSFA=dgrad_state.sSFA,
        sSFB=dgrad_state.sSFB,
        a_smem_layout_staged=dgrad.a_smem_layout_staged,
        b_smem_layout_staged=dgrad.b_smem_layout_staged,
        sfa_smem_layout_staged=dgrad.sfa_smem_layout_staged,
        sfb_tma_smem_layout_staged=dgrad.sfb_tma_smem_layout_staged,
        tiled_mma=dgrad.tiled_mma,
    )
    wgrad_num_tma_load_bytes, _, _ = self._num_tma_load_bytes(
        sA=wgrad_state.sA,
        sB=wgrad_state.sB,
        sSFA=wgrad_state.sSFA,
        sSFB=wgrad_state.sSFB,
        a_smem_layout_staged=wgrad.a_smem_layout_staged,
        b_smem_layout_staged=wgrad.b_smem_layout_staged,
        sfa_smem_layout_staged=wgrad.sfa_smem_layout_staged,
        sfb_tma_smem_layout_staged=wgrad.sfb_tma_smem_layout_staged,
        tiled_mma=wgrad.tiled_mma,
    )
    dgrad_tma = self._make_tma_pipeline(dgrad, dgrad_state, dgrad_num_tma_load_bytes)
    wgrad_tma = self._make_tma_pipeline(wgrad, wgrad_state, wgrad_num_tma_load_bytes)
    problem = _MegaProblem(
        split_sizes=split_sizes,
        groups=G,
        mnk=(M, N, K),
        local_rank=local_rank,
    )
    sync = _MegaPipelineSync(
        ab_full_mbar=ab_full_mbar,
        ab_empty_mbar=ab_empty_mbar,
        tmem_full_mbar=tmem_full_mbar,
        tmem_empty_mbar=tmem_empty_mbar,
        tile_consumer_mbar=tile_consumer_mbar,
        tile_producer_mbar=tile_producer_mbar,
        tile_cta_bar_mbar=tile_cta_bar_mbar,
        tile_id_smem_ptr=tile_id_smem_ptr,
        cross_seam_mbar=cross_seam_mbar,
        counter_ptr=counter_ptr,
        tma_mcast_masks=(
            a_full_mcast_mask,
            b_full_mcast_mask,
            sfa_full_mcast_mask,
            sfb_full_mcast_mask,
        ),
        ab_empty_mcast_mask=ab_empty_mcast_mask,
        acc_full_mcast_mask=acc_full_mcast_mask,
        cluster_cta_rank=cluster_cta_rank,
        pred_cta0=pred_cta0,
    )
    dispatch_quant_sync = _MegaDispatchQuantSync(
        done_counter=mDispatchQuantDoneCounter,
        done_counter_offsets=(
            row_done_counter_offset,
            col_done_counter_offset,
        ),
    )
    activation_quant_sync = _MegaDispatchQuantSync(
        done_counter=mActivationQuantDoneCounter,
        done_counter_offsets=(
            activation_row_done_counter_offset,
            activation_col_done_counter_offset,
        ),
    )

    tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)
    if warp_idx == 0:
        for w in cutlass.range_constexpr(2):
            params = (dgrad, wgrad)[w]
            cpasync.prefetch_descriptor(params.tma_atom_a)
            cpasync.prefetch_descriptor(params.tma_atom_b)
            cpasync.prefetch_descriptor(params.tma_atom_sfa)
            cpasync.prefetch_descriptor(params.tma_atom_sfb)
            cpasync.prefetch_descriptor(params.tma_atom_c)

    if warp_idx == self.TMA_AB_WARP_ID:
        _mega_tma_producer_body(
            params_from_kernel(MegaPipelineParams, self),
            dgrad=dgrad_tma,
            wgrad=wgrad_tma,
            problem=problem,
            sync=sync,
            dispatch_quant=dispatch_quant_sync,
            activation_quant=activation_quant_sync,
            tensormap_manager=tensormap_manager,
            tensormaps=tensormaps,
            combine=combine,
            activation_dispatch=activation_dispatch,
            work_info_smem_ptr=work_info_smem_ptr,
            use_device_tensormaps=use_device_tensormaps,
            format=self.format,
        )
    elif warp_idx == self.MMA_WARP_ID:
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf,
        )
        dgrad_mma = self._make_mma_pipeline(dgrad, dgrad_state, tmem_ptr)
        wgrad_mma = self._make_mma_pipeline(wgrad, wgrad_state, tmem_ptr)
        _mega_mma_consumer_body(
            params_from_kernel(MegaPipelineParams, self),
            dgrad=dgrad_mma,
            wgrad=wgrad_mma,
            problem=problem,
            sync=sync,
            work_info_smem_ptr=work_info_smem_ptr,
            cta_group=self.cta_group,
            sf_dtype=self.sf_dtype,
        )
    elif warp_idx in self.EPILOG_WARP_IDS:
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf,
        )
        if cutlass.const_expr(combine and self.mode != self.MEGA_FORWARD_MODE):
            assert dgrad_state.combine_sC is not None
            dgrad_sC = dgrad_state.combine_sC
        else:
            dgrad_sC = dgrad_state.sC
        dgrad_epilog = self._make_epilog_pipeline(
            params=dgrad,
            state=dgrad_state,
            sC=dgrad_sC,
            tidx=tidx,
            tmem_ptr=tmem_ptr,
        )
        if cutlass.const_expr(combine and self.mode == self.MEGA_FORWARD_MODE):
            assert wgrad_state.combine_sC is not None
            wgrad_sC = wgrad_state.combine_sC
        else:
            wgrad_sC = wgrad_state.sC
        wgrad_epilog = self._make_epilog_pipeline(
            params=wgrad,
            state=wgrad_state,
            sC=wgrad_sC,
            tidx=tidx,
            tmem_ptr=tmem_ptr,
        )
        self._mega_epilog_consumer_body(
            tidx=tidx,
            dgrad=dgrad_epilog,
            wgrad=wgrad_epilog,
            problem=problem,
            sync=sync,
            activation_quant=activation_quant_sync,
            tensormap_manager=tensormap_manager,
            tensormaps=tensormaps,
            mScatter=mCombineScatterPtrs,
            scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
            combine=combine,
            work_info_smem_ptr=work_info_smem_ptr,
            use_device_tensormaps=use_device_tensormaps,
        )
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=(self.NUM_CTAS == 2))
        cute.arch.barrier(
            barrier_id=_BAR_EPILOG_SYNC,
            number_of_threads=32 * len(self.EPILOG_WARP_IDS),
        )
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            if cutlass.const_expr(self.NUM_CTAS == 2):
                cute.arch.mbarrier_arrive(
                    tmem_dealloc_mbar,
                    peer_cta_rank_in_cluster=cluster_cta_rank ^ 1,
                )
                cute.arch.mbarrier_wait(tmem_dealloc_mbar, 0)
            cute.arch.dealloc_tmem(
                tmem_ptr,
                self.num_tmem_alloc_cols,
                is_two_cta=(self.NUM_CTAS == 2),
            )
    else:
        if cutlass.const_expr(self.mode == self.MEGA_FORWARD_MODE):
            if (warp_idx >= self.DISPATCH_QUANT_FIRST_WARP) & (
                warp_idx < self.FORWARD_INPUT_QUANT_FIRST_WARP
            ):
                self._mega_forward_swiglu_quant_producer_body(
                    mWorkCounter=mActivationQuantWorkCounter,
                    mDoneCounter=mActivationQuantDoneCounter,
                    h1_done_counter_offset=(activation_col_done_counter_offset),
                    mFwdX=mCombineDz,
                    mFwdY=mCombineH1,
                    mScatter=mCombineScatterPtrs,
                    mRowQWords=mActivationQuantRowQWords,
                    mRowScale=mActivationQuantRowScale,
                    mColQWords=mActivationQuantColQWords,
                    mColScale=mActivationQuantColScale,
                    split_sizes=split_sizes,
                    G=G,
                    M=M,
                    K=N,
                    local_rank=local_rank,
                )
            elif warp_idx >= self.FORWARD_INPUT_QUANT_FIRST_WARP:
                self._mega_dispatch_quant_producer_body(
                    mWorkCounter=mDispatchQuantWorkCounter,
                    mDoneCounter=mDispatchQuantDoneCounter,
                    tile_smem_ptr=dispatch_tile_id_smem_ptr,
                    dispatch_ptr_smem_ptr=dispatch_ptr_smem_ptr,
                    mGatherPtrs=mDispatchQuantGatherPtrs,
                    mRowQWords=mDispatchQuantRowQWords,
                    mRowScale=mDispatchQuantRowScale,
                    mColQWords=mDispatchQuantColQWords,
                    mColScale=mDispatchQuantColScale,
                    split_sizes=split_sizes,
                    G=G,
                    M=M,
                    K=K,
                    total_tiles=dispatch_quant_total_tiles,
                    local_rank=local_rank,
                    row_done_counter_offset=row_done_counter_offset,
                    col_done_counter_offset=col_done_counter_offset,
                )
        elif cutlass.const_expr(combine):
            if cutlass.const_expr(activation_dispatch):
                if (warp_idx >= self.COMBINE_SWIGLU_QUANT_FIRST_WARP) & (
                    warp_idx < self.COMBINE_ACTIVATION_DISPATCH_FIRST_WARP
                ):
                    _combine_swiglu_bwd_quant_producer_body(
                        params_from_kernel(CombineSwigluQuantParams, self),
                        mWorkCounter=mDispatchQuantWorkCounter,
                        mDoneCounter=mDispatchQuantDoneCounter,
                        mDz=mCombineDz,
                        mH1=mCombineH1,
                        mScatter=mCombineScatterPtrs,
                        mRowQWords=mDispatchQuantRowQWords,
                        mRowScale=mDispatchQuantRowScale,
                        mDxyColQWords=mDispatchQuantColQWords,
                        mDxyColScale=mDispatchQuantColScale,
                        split_sizes=split_sizes,
                        G=G,
                        M=M,
                        K=K,
                        local_rank=local_rank,
                        col_done_counter_offset=col_done_counter_offset,
                        signal_col_tile_done=True,
                    )
                if warp_idx >= self.COMBINE_ACTIVATION_DISPATCH_FIRST_WARP:
                    self._mega_dispatch_quant_producer_body(
                        mWorkCounter=mActivationQuantWorkCounter,
                        mDoneCounter=mActivationQuantDoneCounter,
                        tile_smem_ptr=dispatch_tile_id_smem_ptr,
                        dispatch_ptr_smem_ptr=dispatch_ptr_smem_ptr,
                        mGatherPtrs=mActivationGatherPtrs,
                        mRowQWords=mActivationQuantRowQWords,
                        mRowScale=mActivationQuantRowScale,
                        mColQWords=mActivationQuantColQWords,
                        mColScale=mActivationQuantColScale,
                        split_sizes=split_sizes,
                        G=G,
                        M=M,
                        K=N,
                        total_tiles=activation_quant_total_tiles,
                        local_rank=local_rank,
                        row_done_counter_offset=(activation_row_done_counter_offset),
                        col_done_counter_offset=(activation_col_done_counter_offset),
                        row_quant=False,
                    )
            elif warp_idx >= self.COMBINE_SWIGLU_QUANT_FIRST_WARP:
                _combine_swiglu_bwd_quant_producer_body(
                    params_from_kernel(CombineSwigluQuantParams, self),
                    mWorkCounter=mDispatchQuantWorkCounter,
                    mDoneCounter=mDispatchQuantDoneCounter,
                    mDz=mCombineDz,
                    mH1=mCombineH1,
                    mScatter=mCombineScatterPtrs,
                    mRowQWords=mDispatchQuantRowQWords,
                    mRowScale=mDispatchQuantRowScale,
                    mDxyColQWords=mDispatchQuantColQWords,
                    mDxyColScale=mDispatchQuantColScale,
                    split_sizes=split_sizes,
                    G=G,
                    M=M,
                    K=K,
                    local_rank=local_rank,
                    col_done_counter_offset=col_done_counter_offset,
                    signal_col_tile_done=True,
                )
        elif warp_idx >= self.DISPATCH_QUANT_FIRST_WARP:
            self._mega_dispatch_quant_producer_body(
                mWorkCounter=mDispatchQuantWorkCounter,
                mDoneCounter=mDispatchQuantDoneCounter,
                tile_smem_ptr=dispatch_tile_id_smem_ptr,
                dispatch_ptr_smem_ptr=dispatch_ptr_smem_ptr,
                mGatherPtrs=mDispatchQuantGatherPtrs,
                mRowQWords=mDispatchQuantRowQWords,
                mRowScale=mDispatchQuantRowScale,
                mColQWords=mDispatchQuantColQWords,
                mColScale=mDispatchQuantColScale,
                split_sizes=split_sizes,
                G=G,
                M=M,
                K=K,
                total_tiles=dispatch_quant_total_tiles,
                local_rank=local_rank,
                row_done_counter_offset=row_done_counter_offset,
                col_done_counter_offset=col_done_counter_offset,
            )

    if cutlass.const_expr(self.NUM_CTAS == 2):
        # Keep every active thread alive until peer DSMEM accesses complete.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
