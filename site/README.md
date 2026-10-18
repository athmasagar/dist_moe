# Dist-MoE illustrated guide

An educational page for Dist-MoE’s speed-of-light distributed expert kernels,
with visual direction from [ezyang’s DeepSeek-V3 studies](https://deepseek-v3.ezyang.com/studies/03-roofline.html).

## Preview and deploy

From the workspace root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
```

Open the [article](http://127.0.0.1:8765/) or the [browser slide deck](http://127.0.0.1:8765/?view=slides). The page also works when opening `index.html`
directly; clipboard permissions depend on the browser.

Deploy these files and the artwork folder together to a static host:

- `index.html`
- `styles.css`
- `models.js`
- `diagrams.js`
- `examples.js`
- `app.js`
- `article-layout.js`
- `presentation.js`
- `presentation.css`
- `pdf-export.js`
- `pdf-export.css`
- `results.js`, `results.css`, `results-data.js`
- `data/` (public measurement JSON/CSV downloads)
- `assets/` (official PyTorch logos, bundled KaTeX, fonts, and licenses)

There is no build step or external browser dependency. KaTeX and its fonts are
bundled locally for mathematical typesetting. Fonts, SVG diagrams,
syntax highlighting, numerical examples, and animation run locally. The page
uses modern browser APIs including native dialogs. Publishing is a separate step.

## Browser presentation

The **Present** link switches the article into a 23-slide presentation. The
deck uses a 16:9 canvas with stable diagram framing and the same live controls.
A generic MoE setup precedes the capability comparison (slide 3) and GPU
placement. The comparison table and its visible qualifications come directly
from the article; source links are included in speaker notes. Static slides use
the article’s Key ideas; stepped slides summarize the current phase from the shared figure
caption. Slide 9 now follows the whole routing path in seven steps: symmetric
peer mappings, publication, ID exchange, pointer construction, gather, scatter,
and weighted reduction. The article includes the same figure plus exact
header, payload, and pointer definitions.

Slide 13 compares forward and backward scratch at the same receive-row bound
in two steps, before slide 14 applies three capacity policies. Those presets
use activation factor 1.5 with device/total scratch 4, activation factor 0.5
with device scratch 4 and VMM total scratch 16, and a fixed 32 GiB activation
cap with device/total scratch 16.

Slide 15 pairs forward equations and backward dependencies with value lifetimes.
Slide 16 uses Figure 04A through all nine calls: L0–L2 keep eligible state,
L3 retains only its replay input, and backwards release saved offsets.
Slide 17 keeps all six tile handoffs together. Slide 18 pairs the complete
producer/consumer graph with the SM/CTA warp layout; Mega is the default,
with Staged selectable. The same resident TMA/MMA and epilogue roles serve
both W13 and W2 tasks. There is no separate memory-results slide. Slide 26 combines configuration,
context creation/execution, and plan.explain() in three steps; the API excerpts
and full programs share the article’s code source.

Figure 07 first compares stage–microbatch and whole-microbatch lifetimes and
peak counts. Its second tab shows the simple schedule timeline, with one row
per physical Dist-MoE slot and colored owner intervals showing release and
reuse. Slot capacity calculations are expandable details.
Slide 19 uses four selected steps (start, stage completion, peak overlap, finish);
slide 20 uses three (acquire, release, reassign). Every slide uses the same
Step counter, numbered within its visible sequence. The controls drawer holds
parameters and modes; step navigation stays in the presentation controls.

- **Left / Right**: move through diagram steps, then through slides.
- **Shift + Left / Right**: change slides directly.
- **Home / End**: first or last slide.
- **C**: figure controls. **N**: speaker notes. **T**: theme.
- **F**: full screen, with the slide filling the screen.
- **Presenter**: open notes in a separate window for a second display.
- **Article**: return to the article with the current settings preserved.
- **V** or **Full diagram / Focused view**: toggle the complete diagram and its
  fixed focused frame. Full figures do not offer a redundant focus toggle.
- **Zoom**: enlarge the slide for closer inspection.
- **Export PDF** (or **Cmd / Ctrl + P** in the deck): open a printable deck with
  one 16:9 page for every selected presentation step. Choose **Save as PDF**,
  enable **Background graphics**, and disable browser **Headers and footers**.

### PDF with every step

PDF export follows the same curated steps as the presentation, including the
short pipeline sequences. Each page carries its slide, step, and PDF page number,
the current step's summary, and its fixed diagram framing. The default Mega
deck produces 71 pages; selecting Staged produces 73. Static slides appear
once. Shape, format, packing, tile, and other figure settings carry into the print
tab. The capacity-policy slide intentionally walks all three shared presets.
The print tab keeps SVG graphics and text; it does not rasterize the slides.
Use its **Save as PDF** button again if the first print dialog is canceled.

### One content source

`index.html` contains the article's text, equations, figure controls, author
credit, and copyright. The deck moves the actual live figures into its slide
canvas and derives slide copy and notes from the same content. It does not
maintain a second article or duplicate the SVG renderers.

`models.js`, `diagrams.js`, `examples.js`, and `app.js` supply the same
calculations, graphics, code examples, and interaction logic to both views.
Changes to these shared sources appear after refreshing either view. Figure
settings also carry between views when using Present and Article.

`results-data.js` is the canonical 63-case measurement snapshot.
`results.js` generates the same result diagrams, tables, numerical prose, and
step summaries for the article and deck. A metrics primer on slide 21 precedes
four complete result slides: inference, training, routing skew, and expert width.
Each has one step with all compared plots and raw-number rows visible together.
The article and deck use the same data and renderers; article panels reflow
within their column, while the deck keeps its full diagrams. Peak
references are theoretical NVIDIA datasheet values, with dense compute
denominators and one-direction NVLink bandwidth. HBM is an analytical traffic
estimate. The earlier layer-packing example explains its planner decisions separately
from these single-layer performance measurements.

After editing the measurement snapshot, regenerate the downloadable numerical
exports with `python3 site/data/export_results.py`. The JSON includes timing
windows and per-rank routing/saved-state arrays; CSV contains the raw summary
numbers. Inference and training retain their independently selected EP settings
when entering the deck or exporting steps. Full timing ranges remain in the
article and speaker notes; the slide table uses medians beside plotted ranges.

`presentation.js` defines the slide sequence, source selectors, camera windows,
and navigation. `presentation.css` defines the slide layout. If you move
diagram elements, review their presentation close-ups as part of the change.

## Teaching sequence

The introduction shows the complete hardware path of a remote row. Eight
chapters explain routing metadata, expert values/quantization, memory/VMM,
multi-layer packing, dynamic recompute, lifetimes/offsets, tile and warp-group handoffs, pipeline slot reuse, and the API.
Each figure has a distinct teaching role; the full precision reference lives in
the quantization primer before memory planning.

Figures include:

- a nine-step forward path through peer HBM, NVLink, local HBM, SMEM,
  FP32 TMEM, SwiGLU, and symmetric combine;
- a fixed two-GPU routing map that follows one token through seven steps,
  shows dispatch capacity versus active forward rows, and returns each result
  to its original combine k slot using source-derived byte addresses;
- finite MXFP8 quantization, qdata bytes, E8M0 scales, and outlier underflow;
- NVFP4 packing/global-scale structure and intermediate-precision tables;
- a combined allocation view with device/host/device virtual-memory backing;
- tensor-by-tensor costs, model dimensions, and scratch-factor controls;
- a minimum-to-maximum-useful slot slider, fractional capacity-factor input,
  and exact-byte policy matching the public API;
- a four-layer BF16 stack with rank-specific received rows, saved-state packing,
  remaining-input reserves, dynamic recompute, and LIFO cursor rewinds;
- BF16/MXFP8 lifetimes with an explicit last-reader legend and worked offsets;
- distinct A/B/C shapes, labeled moving dots, K slices, and predicated tails;
- Staged/Mega producer–consumer readiness in the same chapter as GPU tiles;
- a 32-action liveness comparison with separate resource rows and peak counts,
  followed by physical-slot ownership, address resolution, and reuse;
- syntax-highlighted, numbered, copyable code and a generated memory config.

The default allocation example uses a slot capacity factor of 0.35, device
scratch factor 1.5, and total scratch factor 2. For BF16 this gives 25.85 GiB
of saved slots, 6.94 GiB of device scratch, and 2.31 GiB of host overflow.
The figure scales these regions from their calculated bytes.

The capacity-policy slide walks three shared presets: more saved state,
less saved state with VMM overflow scratch, and an exact activation cap.
The second uses activation factor 0.5, device scratch factor 4, and total
scratch factor 16. The diagram shows the hot HBM tier and pinned host backing
within one GPU virtual address range. The other two presets disable VMM.

The setup slide includes the article's typeset expert equations beside the
routing explanation. Pipeline diagrams use one fixed color per reusable slot
under each policy; stage IDs remain text labels. An outline marks acquisition
without changing the slot color.

The page credits Shikai Li and Sanket Purandare as the blog authors and includes
the Meta copyright notice. The header uses official PyTorch artwork: a horizontal logo
on desktop and its icon on smaller screens.

Dark is the default. Light mode persists after an explicit toggle. Use Previous,
Next step, phase tabs, or a slider to choose each state. Views cross-fade without
changing-text highlights. Colored dots illustrate only the selected handoff,
with stationary payload labels beside each active edge, a payload key below
the controls, and no automatic phase advancement. Reduced-motion
settings suppress fades and dot movement. Checkpoint redraws preserve the selected
focused/full framing for both the live diagram and its fade snapshot. Each
slide reserves the maximum text/readout space across its builds and supported
modes so a shorter summary cannot enlarge the diagram. Camera windows stay
fixed within a slide; zoom and full/focused changes are user-controlled.
Expansion preserves the live figure
and its controls.

## Accuracy and maintenance

Read [SOURCES.md](SOURCES.md) before changing an explanation or formula.
The numerical planner model covers ordinary BF16 and MXFP8 training, including
Staged/Mega row-scratch differences, per-expert row padding, and D > F gradient
scratch. It is not the full Python API or an inference planner.

The planner arithmetic was checked against the actual public `ModelConfig` and
`BlockscaledStorageConfig` classes and API slot-selection code across 288 cases in 48 shape/scratch
combinations. Worked offsets,
FP8 format encoding, and schedule coloring received additional checks. The
four-layer model also matches 498 capacity/rank fixtures from the actual BF16
forward and backward planner arithmetic, including saved offsets and rewinds. Browser
QA exercises every diagram phase and mode, text bounds, expansion, syntax
highlighting, dark/light modes, and desktop/phone framing. No GPU kernel behavior
is changed or benchmarked by this website.

Do not reinterpret animation pace as measured latency. NVFP4’s figure shows
its packing and global-scale structure; it does not emulate its full numerical
quantizer. CUDA VMM granularity and allocator padding require the actual package
planner with a device argument.

## Article layout

Figures 01, 06, 09, and 10 fit the article column without horizontal scrolling.
Their panels stack on narrow screens; mobile tables retain every raw value
as labeled fields. The slide view keeps the same full SVG panels and data.
Figure 11 and its repeated memory-results section have been removed.

Figure 01 uses W₁₃, h₁, h₂, W₂, and h₃ throughout, matching the setup and
Section 2 equations. Figure 04B aligns each forward equation, allocated lifetime, and final-reader
equation in one row using that section’s HTML math typesetting. Figure 08 stacks compute and transport
panels when the article column narrows; its tables use labeled rows on phones.
Figure 06 uses the article’s standard label sizes. These changes also appear
in the deck and its step-by-step PDF export.

## Matched MoK comparison

The capability comparison is followed by four static result slides: saved-state
policies, ten-layer training, single-layer imbalance, and expert width.
The article and deck share `comparison-data.js`, `comparison.js`, and the
same live HTML figures. There are no result EP or precision selectors.

The saved-state view shows both layer-imbalance profiles and distinguishes
MoK's partly saved hot layers from the eligible state saved by Dist-MoE across
all ten layers. Status cells indicate ownership and retention, not byte sizes.
Both BF16 and MXFP8 have the shown capacity-decision pattern.

All performance charts use BF16/MXFP8 panels and one consistent EP color key.
Packing, imbalance, and width use grouped horizontal speedup bars. Bars start
at 1× parity and share the same scale; conditions are separate groups rather
than a connected trend. Imbalance charts contain all four EPs together. Every raw table names its
EP and execution mode in the caption, and includes explicit precision, condition,
Dist-MoE iteration time, MoK iteration time, and speedup columns.
The slides emphasize EP16 inference and EP64 training; the article exposes
all measured EPs without dropdowns. Four comparison slides follow the
capability table; each has one step showing the complete comparison.

`data/mok-comparison.json` and the JavaScript snapshot contain the same
110 executions, 44 matched workloads, and 66 individual Dist pipeline pairs.
The public time field is `iterationMS`. Graph/eager controls and timing-window
comparisons are omitted from the presentation and exported comparison data.
Both implementations use CUDA graphs, stated once as measurement context.
The original supplied report is preserved in the workspace.

Iteration time covers forward for inference, forward plus backward for training,
and ten layer forwards/backwards for packing. Dist pipeline selection, native
recompute policy, and total allocated memory are described in the optional
measurement definitions. The later standalone suite has zero shared experts;
the MoK suite has one BF16 shared expert in both implementations.

Article headings state the mechanism or measured takeaway throughout the
introduction and all eight chapters. Slide titles use the same source, with
data-dependent performance titles regenerated from the selected measurements.
