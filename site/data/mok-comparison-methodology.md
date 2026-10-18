# MoK comparison: iteration-time definitions

Report date: October 16, 2026. NVIDIA GB300 GPUs, CUDA 13.0.
The public data includes 110 executions, 44 matched workloads, and
66 individual Dist-MoE pipeline comparisons.

## What one iteration includes

| Workload | Timed work |
| --- | --- |
| Inference, EP8/16 | One layer forward |
| Training, EP32/64 | One layer forward plus backward |
| Packing, EP32/64 | Ten layer forwards followed by reverse-order backward |

T = 2,048 input tokens per GPU, D = 7,168, 384 routed experts, top-6,
and one BF16 shared expert in both implementations. F = 3,072 unless
the width sweep sets it to 1,536 / 3,072 / 6,144. Both use CUDA graphs.
Weights and prepared quantized weights are initialized before timing.

Iteration time is the median of five 30-iteration GPU timing windows,
reduced across GPUs using the slowest GPU. Forward, backward, and combined
times are reduced separately and need not add exactly.

The plotted speedup is MoK iteration time divided by Dist-MoE iteration time
for the same workload and precision. A 1.40x speedup gives 40% more useful
throughput and about 28.6% less iteration time.
The faster measured Dist-MoE pipeline is selected per case; all pipelines
remain available in the source data.

## Layer imbalance and saved state

The ten-layer profiles set imbalance factors on the tracked GPU:

- A: 7.5, 2.5, 4.5, 0.6, 1.8, 0.9, 2.2, 0.7, 1.6, 0.8; average 2.31.
- B: 0.7, 5.8, 1.2, 0.6, 3.6, 1.0, 7.2, 0.8, 2.3, 0.7; average 2.39.

1x means balanced load on this GPU. Other GPUs divide the remaining work.
All ten independent layer forwards are issued before reverse-order backward.

MoK's requested activation factor 2.5 is rounded to approximately 2.67 by
its native per-layer allocation policy. Two layers in profile A and three
in profile B exceed this per-layer budget. Their state is partly retained;
the remainder is recomputed in backward. This pattern follows the native
source policy and recorded schedules.

Dist-MoE shares one activation slot across all layers at activation factor 2.5
and device scratch factor 8, with zero host scratch. Its recorded capacity
flags are zero in every packing case. The diagrams show the eligible state
normally retained by each implementation, rather than all mathematical
intermediates or byte-proportional tensor extents.

These factors describe different allocation policies and do not imply equal
byte budgets. The experiment does not isolate the proportion of the speedup
caused solely by avoided recompute. Total allocated memory includes weights,
scratch, and graph state: EP64 MXFP8 packing peaks at 49.73 GiB for Dist-MoE
and 48.37 GiB for MoK.

## Scope and data

These are independent expert-layer workloads, not connected transformer or
end-to-end training measurements. Native quantization policies are retained;
Dist uses MXFP8 fast math and automatic BF16 fused SwiGLU backward.
Profiling, memory diagnostics, and NVLink collection run separately from timing.

- `mok-comparison.csv`: all iteration-time measurements, using `iterationMS`.
- `mok-comparison-pairs.csv`: all 66 individual Dist-MoE pipeline comparisons.
- `mok-comparison.json`: the same measurements and saved-state evidence.
