# Sources and visual contracts

The page uses the public Dist-MoE checkout supplied for this task. The reference
blog informs its visual language; all figures were drawn for this guide.

| Figure | Source | What it depicts |
| --- | --- | --- |
| Overview | [MXFP8 execution](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_execution.md) | Eight op-level steps: publication/routing/planning, fused Mega forward, owner reduction, backward planning and optional replay, gradient publication, fused W2 backward, fused W13/SwiGLU backward, and input-gradient reduction. Peer dispatch/combine HBM and logical intermediates connect the ops. |
| 01 · Routing metadata and symmetric payloads | [BF16 forward](https://github.com/meta-pytorch/dist_moe/blob/main/docs/bf16_execution.md#forward), [routing orchestration](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/_routing_metadata.py), [routing kernels](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/kernels/triton/dist_dispatch_routing_kernel.py), and [communication buffers](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/_buffers.py) | One-time peer-base rendezvous; int32 count in a 128-byte header and int16 IDs; local expert-major forward/backward pointer construction; BF16 source staging, peer gather/scatter, and owner-side reduction. IDs cross ranks; scores remain local. |
| 02 · MXFP8 quantization | [Quantization implementation](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/kernels/triton_quantization/blockscaled_quantize.py) and [MXFP8 design](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_kernel_design.md) | Finite example: round-up E8M0 scale for amax/448 and nearest-even E4M3 values. No non-finite or optional half-range specialization is exposed in the example. |
| 02 · NVFP4 structure | [NVFP4 design](https://github.com/meta-pytorch/dist_moe/blob/main/docs/nvfp4_kernel_design.md) | Two E2M1 values/byte, E4M3 scales per 16 values, FP32 inverse per source row, global weight factors, and aligned packed source stride. Not a numerical quantizer emulator. |
| 02 · Intermediate formats | [BF16 execution](https://github.com/meta-pytorch/dist_moe/blob/main/docs/bf16_execution.md), [MXFP8 execution](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_execution.md), [NVFP4 execution](https://github.com/meta-pytorch/dist_moe/blob/main/docs/nvfp4_execution.md) | Ordinary non-interleaved paths with plain SwiGLU and score reduction. Optional postprocessing and native backward specializations are outside the table. |
| 03 · Allocation and VMM | [VMM guide](https://github.com/meta-pytorch/dist_moe/blob/main/docs/vmm.md) | Allocation proportions and explicit device/host/device backing in the same memory figure. Physical granularity/padding is excluded. |
| 03 · Memory calculations | [Planner classes](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/_activation_buffer.py) and [API planning](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/api.py) | Ordinary BF16/MXFP8 training formulas, conservative per-expert capacity padding, aligned qdata/scales, saved state, and recompute scratch peaks. Slot controls use the public mutually exclusive exact-byte and fractional-capacity policies, with minimum/balanced/useful-maximum bounds. |
| 04A · Layer packing and dynamic recompute | [Memory planner](https://github.com/meta-pytorch/dist_moe/blob/main/docs/memory_planner.md) and [BF16 device planner](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/_activation_buffer_planner_kernel.py) | Four layers with T=2, D=F=256, K=2, EP=2, two physical slots, and receive counts [4,4], [6,2], [3,5], [5,3]. Full state or source input appends to slot 1; temporary views use a shared scratch suffix. Worst-rank look-ahead includes future input reserve. Saved and scratch extents, forward offsets, and backward cursor rewind follow the BF16 planner. A separate decision view compares both ranks’ hypothetical full-state totals (including the future input reserve) with slot capacity, then shows the resulting backward path. Its bars describe a candidate budget, not physical allocations. |
| 04B · Retention and offsets | [Device planner](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/_activation_buffer_planner_kernel.py) | Exact forward-view offsets for T=2, D=F=256, K=2, E=4, EP=2, three slots, one layer/slot. Save uses balanced full-save capacity; recompute uses minimum input capacity. |
| 05 · GPU tiles | [BF16 kernel design](https://github.com/meta-pytorch/dist_moe/blob/main/docs/bf16_kernel_design.md) | Complete width-D row gathering, followed by TMA selection of A[B<sub>M</sub>,B<sub>K</sub>], B[B<sub>K</sub>,B<sub>N</sub>], and accumulation of C[B<sub>M</sub>,B<sub>N</sub>], memory residency, gather completion, TMA/MMA/epilogue handoffs, and predicated tails. Geometry is a logical shape sketch, not a lane-layout diagram. |
| 06 · Resident warp roles and chunk readiness | [MXFP8 design](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_kernel_design.md#mega-forward), [chunked launch](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/kernels/chunked_mega_blockscaled_grouped_gemm.py), [CTA topology](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/kernels/chunked_mega_blockscaled_grouped_gemm_kernel.py), and [warp work loops](https://github.com/meta-pytorch/dist_moe/blob/main/dist_moe/kernels/gemm_warp_roles.py) | Fixed roles coexist in each resident CTA and process multiple W13/W2 work records. The narrow non-interleaved MXFP8 topology has 16 warps/512 threads; the wide non-interleaved topology has 20/640. Readiness snapshots depict dependencies, with no measured warp-issue timing or occupancy. |
| 07 · Lifetimes and later slot reuse | [Pipeline slots](https://github.com/meta-pytorch/dist_moe/blob/main/docs/pipeline_activation_slots.md) | Exact documented first nine local actions, continued into a complete teaching schedule. Liveness is computed before coloring, with one interval per stage–microbatch pair or whole microbatch. Counts are measured after each action; the full trace peaks at 6 pairs and 4 microbatches (the documented prefix peaks at 5 and 4). Live counts come from unassigned lifetimes; the first view uses illustrative interval coloring to compare the two ownership policies. The second applies that coloring to physical-slot rows and shows the selected slot’s address. Coloring uses half-open intervals ending after terminal backward. Capacity calculations are expandable. The continuation omits distributed communication and is not a prescribed full PP schedule. |
| API/code | [README](https://github.com/meta-pytorch/dist_moe), [MXFP8 guide](https://github.com/meta-pytorch/dist_moe/blob/main/docs/mxfp8_execution.md) | Complete BF16 and MXFP8 teaching programs adapted from the public README and execution guide, with one example selector. The Figure 03 option generates only planning code. The displayed explain() stdout is source-validated. |

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

The motivation table is shared by the article and motivation slide. Its
source-evidence fold links to the reviewed public snapshots: Dist-MoE
`60ce816`, MoK `f3a8bd2`, Comet/Flux `19831ca`, and Mega MoE/DeepGEMM `1fc9c66`.
It records CUDA graph compatibility for MoK and Mega MoE, scopes
Comet's graph limitation to the reviewed dynamic-routing path, distinguishes
MoK macrobatch replay from Dist-MoE's layer-budget policy, and distinguishes
Comet training primitives from Mega MoE's forward-only fused API. The additional
shared-expert row describes the current public interfaces. It is a capability
comparison, with no measured throughput ranking.

## Branding and credits

The header uses the [official PyTorch logo assets](https://github.com/pytorch-fdn/artwork/blob/main/projects/PyTorch/info.md): the full-color/white horizontal variant for dark mode, full-color horizontal variant for light mode, and full-color icon for compact layouts. Transparent canvas margins are reduced for the header while preserving the artwork and surrounding clear space. Assets are bundled locally in `assets/`.

The blog byline names Shikai Li and Sanket Purandare. The visible copyright notice follows the Meta Platforms, Inc. and affiliates wording in the supplied repository license.

Figure 04B uses bundled KaTeX 0.16.45 with Computer Modern-style math fonts,
italic variables, true indices, and mathematical operator spacing. Its equations
use the same native HTML typesetting as the Section 2 vector equations, with
MathML accessibility. Code names are shown separately from the notation; the
lifetime intervals are native HTML in the same row as each forward equation
and final reader. The diamond marks the last read, while the green interval
continues through slot release. Operation columns express dependency order,
not measured duration. Readers follow the documented backward order: route-score
gradient reads h3, W2 weight gradient reads h2, SwiGLU backward reads h1, and
W13 weight gradient reads x_gathered. Expand the input-gradient chain and SiLU
derivative for the remaining equations.
The renderer's MIT license and the fonts' original notices and SIL Open Font
License 1.1 are bundled in `assets/katex/`.

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
presentation metadata supplies topic titles; shared introductory prose and figure
captions supply takeaways and step descriptions. The complete
layer-packing figure follows all nine calls in one slide, showing saved state,
recompute decisions, planner offsets, and backward releases together. The
28 KiB default saves L2 and retains only L3's replay input. Article and slides
use the same source-derived planner records.
Pipeline liveness steps select 0F0, 2B0, 0F4, 0B7 to show acquisition,
individual-stage release, the whole-trace peak, and final completion. The next
slide selects 2F0, 2B0, 0F3 to show a physical slot’s acquisition, release, and
next owner. Both views use the same schedule and source-derived lifetime rules.

Article headings pair topic names with blue interpretation subtitles. Slide 4
borrows the supported-modes table from the API section for its second step and
restores it afterward. The last slide contains only resource links.

## Standalone measurements

The authors supplied `report_isolated_v6/report.html`, 63 result summaries,
153 phase summaries, and cached measurement/routing/allocator records dated
15 October 2026. The public snapshot contains numerical fields and neutral
case descriptors. Article plots, tables, prose, and slides share
`results-data.js`; JSON and CSV downloads are generated from it. The standalone
slide reports use the same case lookup and useful-work/peak calculations, grouped
by workload: EP8/16 inference, EP32/64 training, and EP16/64 width/imbalance sweeps.

- T = 2,048 per rank; D = 7,168; E = 384; top-k = 6; F = 3,072 by default,
  with width controls 1,536/3,072/6,144. One routed layer, no shared expert,
  TP/PP/DP = 1. The published workload dimensions follow the
  [MoK benchmark defaults](https://github.com/cursor/mixture-of-kittens/blob/f3a8bd2432457c5a2a2ea13915f0dcd0224581e4/benchmarks/bench_mok.py).
  All measurements in this snapshot are Dist-MoE.
- Five 30-iteration CUDA-graph windows follow ten warmups. Each window uses
  the maximum rank, followed by the median across windows. Forward, backward,
  and combined medians are independently reduced; they need not add exactly.
- Useful forward GEMM work is 6TKDF per GPU on average. Backward is 2× forward;
  complete training is 3× forward. Useful throughput excludes padding/replay
  FLOPs and includes communication time. It is not a hardware utilization
  counter or a model-quality result.
- The supplied NVIDIA GB300/Blackwell Ultra theoretical references are
  2,500 TFLOP/s BF16, 5,000 TFLOP/s FP8, and explicitly 15,000 TFLOP/s dense
  FP4 per GPU. The listed sparse values are not the normalization denominator.
  HBM is 8,000 GB/s. NVLink is 1,800 GB/s bidirectional, or 900 GB/s per direction.
  These are theoretical datasheet peaks, not benchmarked ceilings.
- NVLink TX/RX come from a separate hardware-counter collection of at least
  one second, converted to decimal GB/s. HBM is modeled useful bytes over
  iteration time; training uses the report's coarse 3× forward byte model. It cannot
  establish actual DRAM utilization or communication/compute overlap.
- EP32 and EP64 training use different sites. Some allocations shared a
  fabric and overlapped in lifetime; actual timing-window overlap is unknown.
  The results describe their observed allocation conditions and do not
  establish uncontended peak performance or fixed-global-batch strong scaling.
- Training uses one slot with activation/scratch factors 8, no host backing,
  and no capacity-triggered recompute. Arena reservations and PyTorch peak
  allocation counters are distinct views and must not be summed.

The metrics primer separates a counted numerator from measured execution
time. Padding/replay arithmetic is outside nominal useful GEMM FLOPs and was
not separately instrumented in this report. Quantization, nonlinearity,
communication, and synchronization are necessary timed work, not part of that
GEMM numerator. Unoccupied reserved memory does not imply executed padding work.

The effective-HBM worked example uses the cached EP64 balanced MXFP8 Mega
record: 765,313,024 forward-model bytes and 2,295,939,072 modeled training bytes,
divided by 2.6411285400390625 ms, give 869.3022839267198 GB/s. The byte model
includes input rows, routed activations, local weights with scales, and routing
scores. Its backward 2×/combined 3× multipliers are coarse assumptions, not
measured traffic passes. The separate rank-0 NVML GPM TX value is
239,035.92260694358 MiB/s, converted with 2^20/10^9 to 250.64733158349847 GB/s.
Rank mean and maximum hardware-counter rates can differ from this rank-0 value.

The two scratch views use the forward and backward planner offsets in
`_activation_buffer_planner_kernel.py`. With ordinary BF16 training and eligible
state omitted, forward scratch is aligned x_gathered + h1 + h2; h3 uses symmetric
combine HBM. Backward’s recompute peak adds gradients and reuse. Both views
use the same device received-row bound. The public API’s three excerpts are
derived from the same full BF16/MXFP8 example programs as the article, ordered
as configure, inspect the logical plan, and execute. The plan output is
shared with the article. The planning-only option skips execution.

## Matched MoK iteration-time comparison

The supplied report dated October 16, 2026 provides the measured iteration times
and saved-state evidence. The public [numeric snapshot](data/mok-comparison.json),
[execution CSV](data/mok-comparison.csv), [pipeline pairs](data/mok-comparison-pairs.csv),
and [definitions](data/mok-comparison-methodology.md) preserve 110 executions,
44 matched workloads, and 66 individual Dist pipeline comparisons.
Both providers use GB300, CUDA 13.0, matched shapes/routing, CUDA graphs,
and one BF16 shared expert. The later standalone suite has no shared expert.

The comparison uses iteration time throughout. The plotted ratio is MoK time
when divided by Dist-MoE time for the same workload. Eager controls and timing
window classifications are omitted; measured values are preserved in the
original workspace report. Packing, imbalance, and width use paired zero-based iteration-time columns.
Both precisions share the same time scale within each EP; speedup labels
compare the two implementations within each workload.

Packing executes ten forwards before reverse-order backward. Layer-imbalance
profiles A/B average 2.31/2.39 and peak at 7.5/7.2 on the tracked GPU.
The native MoK budget rounds requested factor 2.5 to an effective 2.667 per layer.
The two/three hot layers are partly saved under its native source policy.
Dist capacity-induced recompute flags are zero across tested ranks and layers.
The saved-state diagram explicitly uses normalized layer demand, with one unit
representing a balanced layer. Each MoK bin caps at the native effective factor
32768/12288, showing retained demand, unused capacity, and excess demand
recomputed under its source policy. Dist-MoE packs segments proportional to the
profile factors into an illustrative pool of 10 × 2.5 = 25 units. Demand totals
are 23.1/23.9, with 1.9/1.1 units spare. Fixed input state, precision formats,
and padding make physical byte allocations different from this normalized
illustration. Capacity factors, replay patterns, and zero Dist capacity flags
are checked across EP32/64 and BF16/MXFP8.

Every chart and table uses the same provider colors, explicit EP/precision labels,
iteration-time units, and mode labels. The source data retains every measured
Dist pipeline. The observed speedup is not attributed solely to avoided
recompute; total allocated HBM remains distinct from saved-state flexibility.

Paired metrics use the existing report fields, checked against all 110 original
execution rows: usefulTFPerGPU, modeledHBMGBps, meanTXGBps, and meanRXGBps.
Useful forward work is 6*T*(K+1)*D*F per GPU; training triples it and packing
multiplies it by the layer count. This is average useful work per GPU divided
by the slowest-GPU iteration time, excluding padding and replay. HBM uses the
report's logical-byte model, with a coarse 3× forward model for training;
it is not physical DRAM throughput. TX/RX are NVML GPM hardware byte rates,
collected separately for at least one second and averaged across ranks in
decimal GB/s. Their ordering need not match iteration-time ordering. These
rates do not isolate transfer overlap or prove proximity to hardware peaks.

## Logical plan display

The Section 8 text consoles use the existing browser planner and the exact
diagnostic formatting in MemoryPlan.explain(). Only text styling is applied
to the returned string; its fields, words, units, and line breaks are preserved.
The programs and previews use one teaching shape (T=16, D=F=256, top-2,
four local experts, EP2), with activation factor 1 and device scratch factor 1.
BF16 resolves 64 KiB per slot plus 96 KiB scratch; MXFP8 resolves 792 KiB
per slot plus 1,757,184 B scratch because its per-expert row bound is padded.
The Figure 03 preview instead uses that figure's current configuration.

Nine logical previews, including fractional budgets, exact bytes, host scratch,
and a GiB rounding tie, are checked against the repository's actual planner
classes, plan_memory() logic, and MemoryPlan.explain() method without importing
CUDA. The raw text preserves the API's rounding and byte counts. Physical
VMM section fields remain unresolved because the shown call omits device.

The standalone slide columns encode useful PFLOP/s for balanced cases and
normalized whole-layer cost per busiest-GPU received row for imbalance. The latter
is `(t / max_received_rows) / (balanced_t / balanced_max_received_rows)` for the
same EP, format, and pipeline. It characterizes scaling within this implementation;
the earlier matched MoK experiments provide comparative evidence. Width reports
show every F control for useful compute, dense theoretical percentage, and modeled
HBM, including the flat/non-monotonic EP64 training values.

## Lightning presentation

The Lightning edition selects references to the main presentation’s scene objects
in the requested order and uses the user-supplied conference SVG as its complete
title slide. `assets/lightning-title.svg` is bundled byte-for-byte from the supplied
file. It is self-contained and contains no external resource references. All
imported diagrams, models, measurements, and controls remain shared.

The three Lightning memory slides use the same `Moe.memory`, saved/scratch tensor
parts, and `Moe.packing` snapshots as the detailed figures. Brief renderers change
labels and composition only. The working-set comparison uses the same received-row
bound for saved-state, forward scratch, and backward scratch. Returning to the
article restores the detailed renderers.

## Mega backward and shared warp roles

Figure 06 and the new backward scene use the public asynchronous MXFP8 path:

- `dist_moe/_blockscaled.py` calls `dist_blockscaled_grouped_gemm_dgrad_wgrad_dispatch`
  for W2 before `dist_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine`
  for W13. These are separate launch and CTA lifetimes.
- `dist_moe/kernels/mega_blockscaled_grouped_gemm_kernel.py` and its base class
  assign epilogue warps 0–3, MMA warp 4, TMA warp 5, and producer warps 6–15.
- `dist_moe/kernels/gemm_warp_roles.py` selects DGRAD or WGRAD pipelines using
  each work record’s `problem_idx` on the same MMA consumer.
- `MegaProblemVisitor` in `dist_moe/kernels/tile_scheduler.py` lays out each
  expert’s DGRAD tile range followed by its WGRAD range. The schematic does not
  imply strict alternation of individual tiles or measured execution durations.
- The chunked forward kernel sends W13 and W2 work through the same TMA/MMA
  and epilogue roles. Its illustrated narrow layout differs from backward’s split.

The operand labels describe row-gradient/DGRAD-weight views and column-gradient/
column-activation WGRAD views, rather than treating their storage orientation as
an ordinary row-batch multiplication equation. All MMAs accumulate in FP32;
DGRAD stores are BF16 and WGRAD destinations may be BF16 or FP32.

The aligned forward/backward views share scheduling, operand, and hardware
renderers. Forward records illustrate configured chunk lead and readiness;
backward records show the selected weight kernel’s per-expert DGRAD/WGRAD
ranges. Schematic strips do not imply equal tile counts or measured durations.

## Op-level MXFP8 Mega overview

The overview follows `docs/mxfp8_execution.md` and `dist_moe/_blockscaled.py`:
publication/routing/forward planning, one fused forward compute call,
`scale_and_sum`, backward planning with optional forward replay,
`broadcast_and_scale`, W2 `dgrad_wgrad_dispatch`, W13
`dgrad_wgrad_swiglu_bwd_combine`, and `reduce_from_topk`. Setup and
conditional replay each group related operations into one of eight steps.
The three compute labels omit the common `dist_blockscaled_grouped_gemm_`
prefix. Gathered x and h2 are logical row/column MXFP8 bundles, h1 is BF16,
and h3 is scattered into peer combine HBM. The boxes do not imply separate
dense BF16 x_gathered/h2 allocations.

## MoK’s reported baseline results

The four unchanged PNGs in `assets/mok-reported/` come from the user-supplied
“MoK Results” folder and are attributed to
[Cursor’s Mixture-of-Kittens blog](https://cursor.com/blog/mixture-of-kittens)
and [the MoK paper](https://arxiv.org/pdf/2609.36070), §4.1 and Figure 7.
The blog’s single-MoE-layer benchmarks use GB300 NVL72, EP64, and 2,048 input
tokens/GPU, and compare NCCL/PyTorch, DeepEP/PyTorch, DeepEP/TransformerEngine,
and HybridEP/Megatron in BF16/MXFP8 forward/backward.

MoK leads the four forward shapes in both precisions and all four MXFP8 backward
shapes. The BF16 backward DeepSeek-V4-Pro plot has DeepEP/PyTorch slightly ahead.
These reported comparisons motivate MoK as a strong training baseline; they are
not an exhaustive claim over every OSS implementation or a Dist-MoE measurement.

The decks reuse the original plot pixels in wide SVG frames; the axes, model
labels, headings, and shared legend are typeset for presentation readability.
No throughput values are estimated or digitized. Each panel links to its full
source image, which the article continues to show unchanged.

The paper reports effective expert-FFN throughput from the median of 100
slowest-rank timings after 500 warmups. Its Table 1 gives exact MXFP8 timings
for MoK and the fastest baseline, but not all series in the four plots. Those
timings are not used to reconstruct missing chart values.
