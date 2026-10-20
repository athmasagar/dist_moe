# Illustrated-guide validation

## Scope

The guide is a static explanation of the public execution model. Its browser
planner covers ordinary BF16 and MXFP8 training, including Staged/Mega scratch
lifetimes, expert row padding, slot policies, and D > F gradient storage.
GPU kernel execution and hardware measurements are supplied by the repository
and benchmark reports; the website does not run those kernels.

## Checks after an edit

1. **Source accuracy.** Follow the linked execution and planner guides in
   [SOURCES.md](SOURCES.md). Distinguish routing demand, allocation capacity,
   saved ownership, last reads, and hardware measurements.
2. **Static integrity.** Parse edited JavaScript and HTML; check project-relative
   assets, CSS font references, internal anchors, unique IDs, and matching
   measurement snapshots/downloads. The Pages assembler checks local assets.
3. **Arithmetic.** Compare selected memory configurations with the actual public
   planner classes and `plan_memory()` logic. Cover BF16/MXFP8, both pipelines,
   fractional factors, exact bytes, clamping, host scratch, and D > F. Check
   packing offsets, remaining-input reserves, and slot interval non-overlap.
4. **API output.** Parse both complete programs and generated planning code.
   Inputs must be defined before use. Console text and Copy output must match
   `MemoryPlan.explain()` exactly, including byte counts and rounding.
5. **Interactions.** Exercise each offered figure step and mode, rapid clicks,
   expansion/restoration, theme, and example switching. The planning-only example
   has no execution step; full-program expansion must not hide other sections.
6. **Presentation.** Check all scenes in both directions. Summaries follow the
   selected step; focus windows and SVG placement stay fixed within a scene.
   Inspect actual text glyphs as well as container bounds. Verify that slide 4
   restores the supported-modes table to the article and that the final slide
   contains resource links with no table, summary, or footer.
7. **PDF.** Check the selected scene/step sequence, fixed 16:9 pages, counter and
   author/copyright metadata, local SVG definitions, and rendered changed pages.
   Every selected step is exported once.

## Data conventions

| Quantity | Construction |
| --- | --- |
| Iteration time | Median of five 30-iteration windows, using the maximum rank in each window |
| Useful compute | Counted GEMM work divided by whole-operation iteration time |
| NVLink TX/RX | Separate NVML GPM counter run, converted to decimal GB/s |
| Effective HBM rate | Logical byte model divided by iteration time; a coarse training model |
| Hardware percentages | Useful throughput divided by dense theoretical datasheet peaks |
| Comparison slot illustration | Explicitly normalized layer demand; physical allocations also contain inputs, formats, and padding |

Preserve all 63 standalone cases and all 110 comparison executions, 44 matched
workloads, and 66 pipeline pairs. The suites use different shared-expert counts.
Do not infer physical HBM utilization, isolated communication overlap, or a
measured padding/replay FLOP count from the available byte models.

## Audit record

The arithmetic has been compared with source-generated planner and four-layer
packing fixtures, including forward/backward offsets and LIFO rewinds. The
logical API previews have been checked against the public planner and its
exact diagnostic text. UI and presentation checks cover desktop, laptop,
tablet, and phone sizes, with rendered inspection of changed figures and PDF pages.
The default Mega/BF16 deck contains 29 scenes and 83 steps; selected Staged or
planning-only modes change the step count. These counts are checked from the
actual scene sequence rather than from retired slide descriptions.

The Lightning edition references 17 main scenes in the requested order, preceded
by the supplied 16:9 SVG. Three compact memory scenes follow backward slide 12,
giving 21 slides and 57 default steps. Validate tensor strip proportions against
the model, soft versus hard capacity labels, and every saved/recomputed/released packing snapshot. Check
that returning to the article restores its detailed diagrams. Validate the
shared scene identities and compare imported step captions/camera windows with
the full deck. PDF export must retain the edition parameter and render the
original title artwork without added headers or footers.

Check Figure 06’s Forward/Backward selector across article widths and verify
that forward restores Staged/Mega controls. The backward scene follows main
slide 19 and Lightning slide 11. Validate all nine steps, row/column operands,
shared MMA/TMA/epilogue roles, separate W2/W13 launch lifetimes, and stable framing.
Verify role IDs and launch order against the kernel classes and runtime source.

Compare forward and backward framing as well as each view’s step-to-step
framing. Both directions use the same responsive layout and renderer helpers
for scheduled work, operands, and hardware. Validate the selected stream,
operation, and forward title when switching Staged/Mega.

Validate the overview’s eight-step forward/backward sequence against the public
MXFP8 execution guide, including conditional replay before gradient publication.
Check its article, phone, and landscape deck projections; labels and framing
must fit throughout. Its diagram and captions contain no tile-level details.

For the MoK baseline slide, check all four original images, source attribution,
main/Lightning order, collapsed article state, expansion, and restore behavior.
Also validate the extended supported-modes title in its second build.
