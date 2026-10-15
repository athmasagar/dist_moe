# Verification of the illustrated guide

## Source and numerical checks

- Executed the actual public `ModelConfig` and `BlockscaledStorageConfig`
  classes plus the API's slot-selection branch without loading CUDA.
- Compared 288 source-generated cases across 48 shape/scratch combinations:
  BF16/MXFP8, Staged/Mega, D > F, multiple slots, fractional scratch bounds,
  EP-size clamping, and VMM on/off. Slot cases include factor 0, 0.35, and 1,
  unaligned exact bytes, the useful maximum, and an overprovisioned request.
- Passed 4,141 checks covering source byte counts and capacity bounds,
  factor/slider conversion, finite FP8 encoding, worked offset extents and
  alignment, and schedule slot non-aliasing.
- Confirmed the concrete outlier example: the selected tiny value is exact
  at the default scale and underflows to zero with the outlier's scale.
- Parsed embedded and generated Python examples, checked that generated code
  uses one slot policy and actual control values, and verified linked source
  file paths against the supplied public checkout.

- Executed the actual BF16 forward/backward planner AST through a CPU arithmetic
  adapter for all 249 selectable four-layer capacities on both ranks: 498
  fixtures and 3,984 planner calls. Passed 22,410 comparisons of decisions,
  cumulative saved bytes, exact forward/backward offsets, and LIFO rewinds.

## Browser and visual checks

- Exercised every sequence phase, retention/address mode, quantization format,
  outlier state, precision direction, producer/consumer chunk focus, and
  pipeline-slot policy at desktop (1440 px) and phone (390 px) widths.
- Checked all offered A/B tile shapes, full/tail modes, first/last K slices,
  and memory calculations for both small and D > F configurations.
- Passed 44 page interaction checks and 45 focused control checks per viewport,
  with no reported page overflow, SVG text clipping, or text collisions.
- Checked one-click advancement, stable phases without autoplay, cross-fade
  cleanup, rapid repeated clicks, stable Next-step control positions, plain
  labeled dots, and the absence of
  changing-text highlights.
- Checked slot slider bounds and minimum preset, fractional factors, exact
  alignment, mode conversion, invalid-input feedback, useful-maximum presets,
  overprovisioned typed requests, and changing global allocation views.
- Passed 334 edge-label checks per viewport across all moving-edge phases,
  chunk focuses, pipelines, and offered tile shapes. Stationary callouts match
  their active paths; no text was clipped or covered and moving dots did not
  intersect their labels.
- Checked that labels remain fixed as dots advance and disappear when their
  associated edge becomes inactive.
- Passed 23 notation checks per viewport. Tile dimensions use genuine HTML
  and SVG subscripts in prose, controls, captions, payload keys, and edge
  labels. Checked smaller glyph size, lowered baseline, and the absence of
  raw underscore notation in rendered mathematical text.
- Checked live figure expansion/restoration, code expansion, syntax colors,
  dark/light mode switching, and consistent dispatch/combine visibility.
- Verified the official PyTorch logo variants, visible author byline and Meta
  copyright notice in dark/light themes at desktop and phone widths. All
  artwork loaded and no page overflow was reported.
- Reviewed rendered figures, allocation controls, lifetimes, pointer maps,
  quantization, and the code panel through multiple visual passes in WebKit.

## Updated shared presentation

- Verified all 25 slides and their selected phase checkpoints at 1600, 1280,
  and 390 px widths: 181 checks per viewport, with no reported text clipping,
  text collisions, or page overflow.
- Checked changing step summaries, name-only title byline, V keyboard toggling,
  wider focus frames, selected packing/recompute snapshots, and the preserved
  four/three-checkpoint pipeline pacing.
- Verified the same live packing figure moves into the deck and returns to the
  article with its capacity setting and original viewBox preserved.
- Checked all nine packing states on both ranks at minimum, mixed, larger, and
  full-save capacities. Reviewed the larger packing and recompute slide panes.
- Updated the shared allocation example to emphasize saved slots and device
  scratch: factor 0.35 for slots, factor 1.5 for device scratch, and factor 2
  for total scratch. Byte proportions are calculated from those inputs.

- Verified the separate packing/decision article views and slides 13/14 at
  desktop and phone widths: 99 checks per viewport, with no reported clipping
  or text collisions. Checks cover all nine states, both ranks, three budgets,
  source-derived rank totals, save/replay paths, stable topic titles, and live
  figure/settings restoration.

- Separated pipeline liveness from slot assignment. The liveness model has no
  physical slot IDs; the coloring consumes copies of those same intervals.
  Passed 59 comparisons with independent action replay and the documented
  nine-action counts/assignment prefix; the complete trace peaks at 6 pairs
  and 4 microbatches.
- Verified Figure 07’s lifetime comparison and restored schedule-coloring
  timeline as separate views. The liveness labels are enlarged; the reuse
  timeline has physical-slot rows and the selected address, with capacity
  calculations in expandable details. Its repeated live-count chart is removed.

- Checked every action in the restored coloring timeline under both policies,
  the three presentation reuse checkpoints, full/focused view switching, and
  article restoration at desktop and phone widths. Passed 26 interaction
  checks per viewport with no reported clipping or text collisions.

## Complete deck framing audit

- Checked all 25 slides at 1600 × 1000, 1280 × 800, and 390 × 844.
- Exercised 635 slide/control states per viewport, including all visible build
  sequences in both directions; full/focused modes; BF16/MXFP8/NVFP4 forward
  and backward formats; quantization/outlier selections; packing budgets and
  ranks; retention modes and selected offsets; Staged/Mega chunks; all offered
  tile shapes, tails, and K slices; slot policies; memory policies; and code tabs.
- Passed 6,305 assertions per viewport. Sampled 2,010 transition frames per
  viewport (6,030 total), checking live/snapshot viewBoxes and screen geometry.
  No automatic camera changes, scale/position jumps, clipping, or text
  collisions were reported in the final audit.
- Confirmed slide 13 stays at one frame through Step 5 / 5. Slide 8 shows the
  complete quantization figure; its readout space stays fixed across formats.
  Removed phase-dependent camera changes and redundant focus controls.
- Reviewed rendered images for all 25 slides, plus the reported problem slides.
- Increased edge-callout line spacing for subscripts and separated the fusion
  chunk heading from those callouts. Re-ran 334 edge-label checks per article
  viewport: no clipped/covered text or dot paths crossing labels.
- Confirmed presentation sizing is removed from canonical article nodes on
  exit. Source settings and diagrams remain shared between both views.

## PDF export with every presentation step

- Compared all 65 default print pages with live deck navigation: 650 matching
  fields covering scene order, selected steps, phase numbers, titles, summaries,
  setup, scope, visible SVG text, and viewBoxes.
- Passed the complete print-layout audit, including page dimensions, text
  bounds, stable framing, page counters, unique IDs, and per-page SVG definitions.
- Verified the 61-page Mega sequence, NVFP4/MXFP8 settings, light theme, packing
  budget/rank, whole-microbatch policy, fractional slot factors, exact slot bytes,
  and generated code. No text clipping was reported in those cases.
- Checked Export PDF, Cmd+P, popup-block recovery, print-dialog dispatch/reopening,
  malformed-settings feedback, and preservation of the live slide, camera,
  controls, and URL. Checked toolbar access at 390 and 760 px.
- Generated the complete 65-page sample from the same print frames, reopened
  it with a strict PDF reader, and rendered all pages with Poppler. Reviewed the
  rendered pages; graphics and text remain vector content. A font-dependent
  diamond was replaced with a vector icon. The final render reports no errors.
- Included the source policy declaration and weight preparation in the MXFP8
  presentation code excerpts. Kept the canonical two-column quantization axis
  layout and PyTorch wordmark in smaller presentation viewports.

## Coherent examples and restored context

- Consolidated the deck to 21 slides. The full packing, tile hierarchy, and
  producer/consumer graph stay visible through their steps. Separate recompute,
  pointer-map, tensor-core crop, and chunk-table slides were removed.
- The capacity slide has three shared-control builds: factor 1.5 / scratch 4,
  factor 0.5 / scratch 16, and exact 32 GiB / scratch 16. For the default BF16
  shape these give 107.5, 36.5, and 32 GiB per slot, respectively.
- Checked the 28 KiB example against the actual-source planner fixtures on both
  ranks: 56 comparisons of decisions, saved cursors, totals, and offsets passed.
  L2's 27,136-byte test fits; L3's 35,840-byte test does not.
- Checked the expert-route backward equations against independent finite
  differences for input, W13, W2, and score gradients. Maximum error was below
  1.3e-10 in the double-precision numerical example.
- Passed the complete 57-step sequence at 1600, 1280, and 390 px. The full
  layout audit checks text bounds and every displayed SVG label. Checked 43
  navigation, mode, full-frame, and article-restoration assertions per viewport.
- Verified that the lifetime slide does not inherit the address view's crop,
  manual packing budgets and Staged/Mega selection survive navigation, and
  old slide links resolve to their consolidated examples.
- The PDF sequence at that revision had 57 default Mega pages, or 59 with Staged selected.
  The capacity slide walks all three examples. The print-layout audit passed
  2,893 checks, including source order, page dimensions, label bounds, unique
  SVG references, and constant framing.

These export checks used macOS WebKit. Chrome startup was blocked by process
permissions in the execution environment, and automated native print-dialog
verification did not complete. The sample PDF was assembled from WebKit PDF
captures of the shared print frames; browser Save as PDF remains a manual check.

These checks verify the educational page and its stated model. They do not
execute or benchmark GPU kernels, emulate every precision specialization, or
resolve physical CUDA mapping granularity on a GPU.

## Source-reviewed motivation comparison

- The article and slide 3 use the same live comparison table. All six rows,
  four implementation columns, and two qualifications remain visible.
  Speaker notes and the article evidence fold contain 14 pinned source links.
- The 22-slide deck passes the full 58-step layout sequence at 1600, 1280,
  and 390 px. Focused interaction checks cover source identity, table bounds,
  footnote spacing, theme changes, navigation, and restoration to the article.
- The default Mega print sequence has 58 pages; Staged has 60. The comparison
  appears once in either sequence. The default print-layout audit passes
  2,911 checks. The rendered comparison PDF page was reviewed with all columns,
  qualifiers, footer counts, and source-review date intact.

## KaTeX equation typography

- Figure 04B uses locally bundled KaTeX and fonts. Tests cover all four
  BF16/MXFP8 and save/recompute combinations at desktop and narrow sizes,
  including equation bounds, tensor-label spacing, address-view cleanup,
  article restoration, and MathML output.
- The full 58-step deck passes layout checks at 1600, 1280, and 390 px.
  The printable deck passes 3,138 layout and glyph-bound checks. The exported
  equation page was rendered and inspected; its mathematical text remains
  selectable in the PDF.

## Combined setup, VMM preset, and stable slot colors

- Removed the separate expert-math slide. Slide 2 uses the original article
  equation node; navigation restores it to the precision section. The deck
  now has 21 slides and 57 default Mega builds (59 with Staged).
- The full sequence passes layout checks at 1600, 1280, and 390 px.
  Focused interactions check all three capacity presets, VMM backing and
  generated configuration, shared-source restoration, unique slot colors,
  color agreement across both lifetime policies and both diagrams, and
  focused/full framing through slot transitions.
- The 57-page printable deck passes 3,238 layout, glyph, and reference checks.
  Rendered PDF pages were inspected for the combined setup, host VMM example,
  maximum liveness, and physical-slot reuse. Page counters and slide titles
  were verified after assembly. Text and equations remain selectable.

## Peak liveness and slot depth

- Both liveness panels show their reserved budget. The article and the
  peak-liveness slide step explain why six 2-layer slots reserve 192 MiB,
  while four 4-layer slots reserve 256 MiB: a 64 MiB (25%) saving at the
  illustrative 16 MiB per layer.
- All 57 steps pass layout checks at 1600, 1280, and 390 px. The printable
  deck passes 3,246 checks, and the revised peak-budget PDF page was rendered
  and inspected. The comparison uses the same computed slot counts, depths,
  and per-layer budgets as the ownership diagram.

## Standalone result figures and shared data

- Added four article figures and five shared result slides: inference and
  training hardware references, routing skew, expert width, and arena/policy
  behavior. The deck has 26 slides and 70 default Mega PDF builds (72 with
  Staged). New diagrams retain their full framing across steps and modes.
- Checked 2,988 source/export comparisons across 63 cases and 153 phase
  summaries. Published measurements match the supplied summary and cached
  timing windows. Script, JSON, and CSV snapshots agree. Numerical fields are
  exported without native environment paths or infrastructure identifiers.
- Dense theoretical references are 2.5/5/15 PFLOP/s per GPU for BF16/FP8/FP4.
  NVLink uses 900 GB/s per direction. HBM is explicitly modeled; derived
  busiest-rank rates are distinguished from hardware counters. The packing
  view is explicitly a projection, not an empirical multi-layer result.
- The full deck passes 3,040/3,037/3,037 layout checks at 1600/1280/390 px.
  Focused result checks pass 2,098 comparisons per viewport, covering every
  supported mode and step, numerical invariants, table coverage, remembered
  EP settings, reset, source identity, and article restoration.
- The printable deck passes 3,677 checks. The 70-page PDF was assembled,
  verified for titles/counters and page dimensions, and rendered for visual
  inspection of compute, bandwidth, skew, width, policy, and packing views.
  Graphics and mathematical text remain vector/selectable. No GPU benchmark
  was rerun by the website work.

## Metrics primer

- Added one shared metrics slide before the result plots. It separates
  nominal useful GEMM FLOPs, CUDA-event latency, NVML GPM TX/RX rates, and
  modeled HBM byte rate. Necessary non-GEMM work is distinguished from
  additional padding/replay arithmetic; neither is silently removed from time.
- Verified the worked HBM byte sum, 3× training multiplier, and MiB/s-to-GB/s
  NVLink conversion against the cached record. The additional executed FLOPs
  are explicitly not separately measured in this report.
- The 27-slide, 71-step deck passes 3,176/3,173/3,173 checks at 1600/1280/390 px.
  KaTeX glyph bounds are checked inside their definition and slide; invisible
  strut overflow is not used as a proxy for visible clipping. The printable
  deck passes 3,815 checks, and the metrics PDF page was rendered and inspected.
