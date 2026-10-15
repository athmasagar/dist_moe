# Sources and visual contracts

The page uses the public Dist-MoE checkout supplied for this task. The reference
blog informs its visual language; all figures were drawn for this guide.

| Figure | Source | What it depicts |
| --- | --- | --- |
| Overview | [BF16 execution](https://github.com/meta-pytorch/dist_moe/blob/main/docs/bf16_execution.md#forward) | Source publication, peer gather into HBM, TMA/SMEM, FP32 TMEM, BF16 h1, FP32 SwiGLU, BF16 h2, W2, peer scatter, and owner reduction. |
| 01 · Assignments and return slots | [Running example](https://github.com/meta-pytorch/dist_moe/blob/main/docs/bf16_execution.md#running-example) | Documented route assignments and expert order. Scores/output components are illustrative; sums precede BF16 rounding. |
| 02 · MXFP8 quantization | [Quantization implementation](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/kernels/triton_quantization/blockscaled_quantize.py) and [MXFP8 design](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_kernel_design.md) | Finite example: round-up E8M0 scale for amax/448 and nearest-even E4M3 values. No non-finite or optional half-range specialization is exposed in the example. |
| 02 · NVFP4 structure | [NVFP4 design](https://github.com/meta-pytorch/dist_moe/blob/main/docs/nvfp4_kernel_design.md) | Two E2M1 values/byte, E4M3 scales per 16 values, FP32 inverse per source row, global weight factors, and aligned packed source stride. Not a numerical quantizer emulator. |
| 02 · Intermediate formats | [BF16 execution](https://github.com/meta-pytorch/dist_moe/blob/main/docs/bf16_execution.md), [MXFP8 execution](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_execution.md), [NVFP4 execution](https://github.com/meta-pytorch/dist_moe/blob/main/docs/nvfp4_execution.md) | Ordinary non-interleaved paths with plain SwiGLU and score reduction. Optional postprocessing and native backward specializations are outside the table. |
| 03 · Allocation and VMM | [VMM guide](https://github.com/meta-pytorch/dist_moe/blob/main/docs/vmm.md) | Allocation proportions and explicit device/host/device backing in the same memory figure. Physical granularity/padding is excluded. |
| 03 · Memory calculations | [Planner classes](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/_activation_buffer.py) and [API planning](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/api.py) | Ordinary BF16/MXFP8 training formulas, conservative per-expert capacity padding, aligned qdata/scales, saved state, and recompute scratch peaks. Slot controls use the public mutually exclusive exact-byte and fractional-capacity policies, with minimum/balanced/useful-maximum bounds. |
| 04A · Layer packing and dynamic recompute | [Memory planner](https://github.com/meta-pytorch/dist_moe/blob/main/docs/memory_planner.md) and [BF16 device planner](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/_activation_buffer_planner_kernel.py) | Four layers with T=2, D=F=256, K=2, EP=2, two physical slots, and receive counts [4,4], [6,2], [3,5], [5,3]. Full state or source input appends to slot 1; temporary views use a shared scratch suffix. Worst-rank look-ahead includes future input reserve. Saved and scratch extents, forward offsets, and backward cursor rewind follow the BF16 planner. A separate decision view compares both ranks’ hypothetical full-state totals (including the future input reserve) with slot capacity, then shows the resulting backward path. Its bars describe a candidate budget, not physical allocations. |
| 04B · Retention and offsets | [Device planner](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/_activation_buffer_planner_kernel.py) | Exact forward-view offsets for T=2, D=F=256, K=2, E=4, EP=2, three slots, one layer/slot. Save uses balanced full-save capacity; recompute uses minimum input capacity. |
| 05 · GPU tiles | [BF16 kernel design](https://github.com/meta-pytorch/dist_moe/blob/main/docs/bf16_kernel_design.md) | Complete width-D row gathering, followed by TMA selection of A[B<sub>M</sub>,B<sub>K</sub>], B[B<sub>K</sub>,B<sub>N</sub>], and accumulation of C[B<sub>M</sub>,B<sub>N</sub>], memory residency, gather completion, TMA/MMA/epilogue handoffs, and predicated tails. Geometry is a logical shape sketch, not a lane-layout diagram. |
| 06 · Producer/consumer overlap | [MXFP8 design](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_kernel_design.md#mega-forward) and [chunked kernel](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/kernels/chunked_mega_blockscaled_grouped_gemm_kernel.py) | Chunk-local operand readiness, launch ordering, and co-live row scratch. Snapshots are conceptual dependencies, not actual scheduling or durations. No activation ring is used. |
| 07 · Lifetimes and later slot reuse | [Pipeline slots](https://github.com/meta-pytorch/dist_moe/blob/main/docs/pipeline_activation_slots.md) | Exact documented first nine local actions, continued into a complete teaching schedule. Liveness is computed before coloring, with one interval per stage–microbatch pair or whole microbatch. Counts are measured after each action; the full trace peaks at 6 pairs and 4 microbatches (the documented prefix peaks at 5 and 4). The first view has no physical assignments. The second uses a simple schedule timeline with one row per physical slot, colored owner intervals, and the selected slot’s address. Coloring uses half-open intervals ending after terminal backward. Capacity calculations are expandable. The continuation omits distributed communication and is not a prescribed full PP schedule. |
| API/code | [README](https://github.com/meta-pytorch/dist_moe), [MXFP8 guide](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_execution.md) | BF16 quickstart copied from the public README; other tabs are labeled integration/configuration fragments. |

## Important distinctions

- Caller routing is external. Symmetric communication and rank-local activation
  storage have different visibility and ownership.
- BF16 full-save retains x_gathered/h1/h3; h2 stays temporary and is recreated
  by SwiGLU in backward. Recompute retains only source x.
- Ordinary MXFP8 full-save retains column x/h2 bundles, dense h1/h3; forward row
  operands remain temporary. Staged row operands can alias; Mega co-feeds them
  and needs disjoint subslots.
- Natural scale count differs from reserved scale byte extent. Row scratch uses
  the source allocator’s 4× natural scale extent; column bundles use their own
  orientation. qdata plus natural scale bytes are not a complete allocation.
- Several MoE layers append to the same owner’s slot; there is no allocation
  per layer. Each forward records its offsets and slot ID. Matching backwards
  unwind that slot in LIFO order. The future-input reserve is a budget check
  and has no allocated tensor or host backing.
- Dynamic recompute is determined by cumulative saved bytes on the worst EP
  rank, not solely by the locally inspected row count. Device predicates gate
  replay while the host launch topology remains fixed.
- Offset zero can be valid. A mode’s unused BF16 fields also use zero; predicates
  determine consumption. The visual does not treat every zero as a null pointer.
- Lifetime bars show retained ownership and temporary values. The purple
  diamond marks the last reader of the depicted forward value; the reserved
  saved-state slot may remain owned until backward releases it. Exact CUDA instruction times are not inferred from geometry.
- Symmetric dispatch and combine buffers are both peer-addressable device HBM.
  The overview teaches hardware residency; Figure 01 teaches token IDs, expert
  order, and top-k return-slot indices.
- Whole-microbatch and stage–microbatch plans differ in both slot count and depth.
  The example uses two layers/stage and an illustrative 16 MiB per layer.

## Validation

The source planner classes were executed without loading the CUDA runtime to
produce 288 independent expected cases over 48 shape/scratch combinations,
including D > F, multiple slots, EP-size clamping, VMM on/off, fractional
capacity factors, byte alignment, and overprovisioned slot budgets. The actual
API slot-selection branch supplies the expected slot counts. The browser
model matched every checked byte count and capacity bound. Additional checks
execute the actual BF16 planner functions with a CPU arithmetic adapter over
498 capacity/rank cases (3,984 forward/backward calls). The four-layer model
passes 22,410 comparisons of decisions, saved totals, forward/backward tensor
offsets, and cursor restoration. This checks the source arithmetic without
executing CUDA. Additional checks cover FP8 finite-value encoding, worked offset bounds/alignment, and slot
non-aliasing. Browser verification and visual review cover all phases/modes;
this is validation of the educational page, not GPU correctness or performance.

## OSS capability comparison

The motivation table is shared by the article and presentation slide 3. Its
source-evidence fold links to the reviewed public snapshots: Dist-MoE
`60ce816`, MoK `f3a8bd2`, Comet/Flux `19831ca`, and Mega MoE/DeepGEMM `1fc9c66`.
It preserves the unconfirmed CUDA graph entries for MoK and Mega MoE, scopes
Comet's graph limitation to the reviewed dynamic-routing path, distinguishes
MoK macrobatch replay from Dist-MoE's layer-budget policy, and distinguishes
Comet training primitives from Mega MoE's forward-only fused API. The additional
shared-expert row describes the current public interfaces. It is a capability
comparison, with no measured throughput ranking.

## Branding and credits

The header uses the [official PyTorch logo assets](https://github.com/pytorch-fdn/artwork/blob/main/projects/PyTorch/info.md): the full-color/white horizontal variant for dark mode, full-color horizontal variant for light mode, and full-color icon for compact layouts. Transparent canvas margins are reduced for the header while preserving the artwork and surrounding clear space. Assets are bundled locally in `assets/`.

The blog byline names Shikai Li and Sanket Purandare. The visible copyright notice follows the Meta Platforms, Inc. and affiliates wording in the supplied repository license.

## Presentation view

The browser deck uses the article's canonical HTML and the same `Moe`, `Views`,
`Examples`, and `Explainer` objects. It moves live figures between the article
and slide canvas. Camera windows are fixed per slide and do not change between builds or
control modes. They do not change the data or memory arithmetic. The layout
measures the shared source across supported builds/modes without animation,
restores the selected state, and reserves the largest title, summary, setup,
legend, and readout bands. All visible build counters use Step numbering. Complete diagrams remain accessible
through Full diagram, and detailed source explanations populate speaker notes.

The setup figure illustrates one token selecting two of four expert MLPs. It
separates token data from the router’s expert IDs and combine scores. The
canonical Key ideas section supplies static slide takeaways; stepped slides
derive their changing summaries from the shared figure captions. The complete
layer-packing figure follows all nine calls in one slide, showing saved state,
recompute decisions, planner offsets, and backward releases together. The
28 KiB default saves L2 and retains only L3's replay input. Article and slides
use the same source-derived planner records.
Pipeline liveness checkpoints select 0F0, 2B0, 0F4, 0B7 to show acquisition,
individual-stage release, the whole-trace peak, and final completion. The next
slide selects 2F0, 2B0, 0F3 to show a physical slot’s acquisition, release, and
next owner. Both views use the same schedule and source-derived lifetime rules.
