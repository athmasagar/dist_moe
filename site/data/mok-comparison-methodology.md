# Matched CUDA-graph MoK and Dist-MoE comparison

Report date: October 16, 2026. NVIDIA GB300 GPUs, CUDA 13.0.
The numerical snapshot contains 110 accepted backend executions, 44 matched
workloads, 66 individual Dist pipeline comparisons, and 3,960 accepted rank
traces. It covers expert-layer performance, not end-to-end model training.

## Workloads

T = 2,048 input tokens per rank, D = 7,168, 384 routed experts, top-6,
and one BF16 shared expert. F = 3,072 except the width sweep, which uses
1,536 / 3,072 / 6,144. Inference uses EP8/16; training uses EP32/64.
Only EP16 inference and EP64 training were measured in the width sweep.

Both providers receive matched shapes, routing, and weights in the same runtime.
Dist uses its concurrent shared-expert stream, native MXFP8 fast math, and
automatic BF16 fused SwiGLU backward. MoK uses its native fused shared/routed
path and native quantization and communication settings. Weight preparation
and quantization happen before timing.

Routing imbalance is maximum received routes divided by mean received routes.
The mean is T*K = 12,288 routes per rank. Single-layer cases target 1 / 4 / 8.
The width control at F = 3,072 is a separate measurement from the balanced
single-layer control; compare providers within the same case.

## Timing and correctness

Ten eager warmup iterations precede capture and replay verification.
Forward and complete backward are captured with one shared graph pool.
Schedule construction and both routed/shared expert work are inside capture.
Graph outputs and gradients are compared with independent eager references
after inputs and output gradients change at stable addresses. Routing remains
fixed within a captured case; this experiment does not test changing routing
values within the same graph.

Primary graph timing uses five windows of 30 iterations, followed by five
matched eager windows of 30 iterations. CUDA events measure GPU elapsed time;
each phase's median uses per-window maximum-rank latency. Combined training
times forward plus backward. Phase medians are summarized separately and need
not add exactly to the combined median.

Graph/eager pairs use the same pipeline, weights, routing, runtime, and
allocation; graph pools remain resident during eager controls. Their ratio
measures the full graph execution treatment. It does not isolate launch,
autograd, allocation, or stream scheduling overhead. Timing order is fixed,
not randomized. Profiling, memory diagnostics, and NVLink counters are
collected separately; no profiler runs in the timing windows.

## Ratios and pipeline selection

MoK/Dist speedup = MoK graph milliseconds / Dist graph milliseconds.
For the same useful workload, 1.40x means 40% more useful throughput, or
about 28.6% less latency; it does not mean 40% less latency.

The primary chart selects the best valid Dist graph median for each workload.
Every Staged/Mega result remains in the CSV and JSON. All 44 selected medians
and all 66 individual Dist pipeline medians are lower than MoK in this suite.
Observed ranges do not overlap in 43 workloads; one overlaps.

Whiskers are the conservative ratios formed from the providers' observed
five-window extrema. They describe observed variation, not confidence
intervals or a statistical significance test. The eager control follows
the selected graph pipeline, rather than selecting a different eager winner.

## Ten-layer activation packing

Ten independent inputs and weight sets run all forwards before reverse-order
backward. Their saved state is simultaneously live. There are no model
connections between these layers, attention, or optimizer steps.

Rank 0's target load factors by layer are:

- A: 7.5, 2.5, 4.5, 0.6, 1.8, 0.9, 2.2, 0.7, 1.6, 0.8; mean 2.31.
- B: 0.7, 5.8, 1.2, 0.6, 3.6, 1.0, 7.2, 0.8, 2.3, 0.7; mean 2.39.

Other ranks divide the remaining routes equally. The load bars are routing
targets, not tensor-byte extents. The exact native rounded schedule and replay
rows are retained separately in each MoK packing case's capacity record.

Dist uses one shared activation slot with capacity factor 2.5, device scratch
factor 8, and zero host scratch. All recorded capacity-induced recompute flags
are zero across the tested layers and ranks. The slot bar uses the actual
saved bytes on rank 0 after all ten forwards and the actual reserved slot
capacity. It does not estimate per-layer byte sizes from the load factors.

MoK's requested factor-2.5 per-layer saved-state budget is 30,720 rows.
Native 4,096-row minibatches round it to 32,768 retained rows per layer,
an effective factor of 2.667. Hot layers still exceed this limit. Reported
native replay rows are derived from the source policy and observed schedules,
not measured hardware replay counters. All eight MoK packing cases have
policy-derived replay; no Dist packing case has a capacity-induced replay flag.

These factors are different allocation policies, not equal byte budgets.
Packing supports borrowing spare capacity across layers. The experiment does
not isolate what portion of the latency advantage is caused by avoided replay.

## Phases, bandwidth, and memory

Useful GEMM FLOPs per GPU are 6*T*(K+1)*D*F for forward and
18*T*(K+1)*D*F for training. Packing multiplies work by ten. Padding and
replay are excluded. The shared expert is BF16 even in MXFP8 routed cases.

TX/RX values are means of separately measured per-rank NVLink counters.
Effective HBM is analytical useful bytes divided by latency, using a coarse
forward/backward/combined byte model. It is not measured physical DRAM traffic.
Allocated and reserved peaks are device totals with graph pools resident;
activation-slot capacity describes a narrower allocation owner.

Capacity flexibility does not guarantee lower total allocated HBM.
In EP64 MXFP8 packing, the selected Dist pipeline peaks at 49.73 GiB versus
48.37 GiB for MoK. Balanced BF16 EP64 also has a slightly faster MoK
forward-only phase, while complete forward-plus-backward still favors Dist.
Shared-fabric activity and fixed measurement order limit causal attribution.

## Published data

- `mok-comparison.csv`: all 110 executions and their measured numeric metrics.
- `mok-comparison-pairs.csv`: all 66 Dist pipeline comparisons against MoK.
- `mok-comparison.json`: the same cases, selected pairs, observed ranges,
  routing profiles, and packing capacity evidence.
