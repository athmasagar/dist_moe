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

These export checks used macOS WebKit. Chrome startup was blocked by process
permissions in the execution environment, and automated native print-dialog
verification did not complete. The sample PDF was assembled from WebKit PDF
captures of the shared print frames; browser Save as PDF remains a manual check.

These checks verify the educational page and its stated model. They do not
execute or benchmark GPU kernels, emulate every precision specialization, or
resolve physical CUDA mapping granularity on a GPU.
