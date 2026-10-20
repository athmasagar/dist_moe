# Dist-MoE illustrated guide

A static educational article, browser slide deck, and stepped PDF view. They share
one set of explanations, diagrams, controls, examples, and measurements.
Article headings name each topic, with blue subtitles carrying its takeaway.

## Preview

From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open [the article](http://127.0.0.1:8000/),
[the deck](http://127.0.0.1:8000/?view=slides), or
[the print view](http://127.0.0.1:8000/?view=print).
There is no build step. Scripts, official PyTorch artwork, KaTeX, fonts, and
licenses are bundled. Serve the complete `site/` directory. See
[DEPLOYMENT.md](DEPLOYMENT.md) for GitHub Pages and the branch preview.

## Reading and presentation flow

The introduction establishes the MoE operation and physical GPU path, then motivates
saved-state management with the capability table and matched MoK results.
Eight chapters cover routing, precision, memory planning, layer packing,
kernel handoffs, pipeline liveness, hardware performance, and the public API.

| Figure | Question it answers |
| --- | --- |
| Setup | What does routing and weighted expert combine compute? |
| Overview | Where does a row travel through peer HBM, local memory, and tensor cores? |
| 01 | How do IDs and pointer tables select gather and return addresses? |
| 02 | How do qdata, scales, and quantization axes determine tensor formats? |
| 03 | Which simultaneously live tensors determine slot and scratch bytes? |
| 04A | How do layers pack state, reserve replay inputs, and rewind offsets? |
| 04B | Which backward readers require each value to survive? |
| 05 | How do differently shaped tiles move through HBM, SMEM, and TMEM? |
| 06 | How do resident CTA roles coordinate ready/consumed handoffs? |
| 07 | How do owner lifetime and layer depth determine reusable slot capacity? |
| Metrics, 08–10 | How are rates constructed, and how do hardware, imbalance, and width affect them? |
| API | How do configuration, plan inspection, and execution connect? |

The default deck has 27 slides and 74 printed steps. The Staged kernel view adds
two steps. The planning-only API example omits execution. Presentation titles name the topic
independently of article headings; summaries use shared prose, live captions, or
the measured cases. Previous slide anchors redirect to their current scenes.

| Control | Action |
| --- | --- |
| Left / Right | Move through steps, then slides |
| Shift + Left / Right | Change slides directly |
| Home / End | First / last slide |
| C / N | Controls / notes |
| V | Focused / full diagram, where the figure offers both |
| T / F | Theme / full screen |
| Export PDF | One 16:9 page per selected step |

PDF export keeps figure settings and fixed diagram framing. In the browser print
dialog, enable background graphics and disable headers/footers. Figures advance
by user input; dots illustrate the active handoff. Reduced-motion settings suppress
fades and dot motion. Authors and the Meta copyright are shared by both views.

Slide 4 steps through OSS capabilities and the shared supported-modes table.
The final slide contains resource links, with its summary and footer omitted.

Slides 7 and 8 have separate EP16 inference and EP64 training steps, with both
precisions visible and chart/table pairs arranged side by side. Standalone results
show both measured EPs for each mode; imbalance and width compare EP16 inference
with EP64 training, including every measured precision/kernel variant. The report
renderer uses the same cases, rate calculations, and hardware references as the article.
Balanced result columns show useful PFLOP/s per GPU. Routing imbalance compares
whole-layer cost per busiest-GPU received row with the balanced reference. Width
reports pair useful PFLOP/s with dense-peak percentage and show modeled HBM rates
for every measured width; training HBM is not assumed to increase monotonically.

## Lightning

Open the [Lightning deck](http://127.0.0.1:8000/?view=slides&deck=lightning), or choose
Lightning in the presentation selector. The conference title uses the supplied
SVG unchanged. Its remaining slides reference the main scene objects in this order:
2, 4, 5, 6, 7, 8, 26, 3, 18, 21, 22, 23, 24, 25, 27.

Lightning has 16 slides and 36 default steps. Every imported slide retains the
main deck's controls, content, and step sequence. Export PDF keeps `deck=lightning`
and includes every selected step. The original full deck stays at 27 slides and
74 default steps. The article header links to both versions.

## Source layout

| Source | Responsibility |
| --- | --- |
| `index.html` | Article prose, equations, controls, credits, and shared slide summaries |
| `models.js` | Ordinary BF16/MXFP8 training arithmetic, offsets, interval coloring, and logical plan text |
| `diagrams.js`, `article-layout.js` | Shared SVG drawings and responsive panel placement |
| `examples.js`, `app.js` | Complete teaching programs, controls, syntax styling, and live readouts |
| `presentation.js`, `pdf-export.js` | Topic titles, scene pacing, shared-content mounting, and print snapshots |
| `styles.css`, the feature CSS files | Article, results, comparison, presentation, and print styles |

The API has one example selector. BF16 and MXFP8 use one teaching shape to generate
complete programs and their logical plan output. The Figure 03 option generates
planning code from its controls. The console preserves the exact `explain()`
string; physical VMM section details require `plan_memory(..., device=...)` on a GPU.

## Measurements

- `results-data.js`: 63-case Dist-MoE-only suite with zero shared experts.
- `comparison-data.js`: 110 executions, 44 matched MoK workloads, and 66 Dist
  pipeline pairs, with one BF16 shared expert in both implementations.
- `data/`: matching JSON/CSV downloads and comparison measurement definitions.

Keep timing populations and useful-work counts within their own suites. Rates
and numerical captions are generated from the snapshots. Hardware references
are dense theoretical NVIDIA peaks. NVLink uses hardware counters; HBM uses a
logical-byte model. Normalized saved-state illustrations are labeled separately
from byte-proportional allocation views.

Regenerate standalone downloads after editing its shared snapshot:

```sh
python3 site/data/export_results.py
```

Read [SOURCES.md](SOURCES.md) before changing arithmetic or a source claim, and
[VALIDATION.md](VALIDATION.md) for the audit and required checks. Preserve measured
values, source qualifications, licenses, the exact API output, and every offered
control when changing the presentation.
