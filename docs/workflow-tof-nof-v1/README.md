# ToF / NoF evaluation workflow v1

The editable source is [`../tof-nof-evaluation-workflow-v1.excalidraw.md`](../tof-nof-evaluation-workflow-v1.excalidraw.md).
The [SVG preview](tof-nof-evaluation-workflow-v1.svg) contains the same text and
layout in a single tall figure. Shapes, text, and connectors are editable in
Obsidian Excalidraw; there are no embedded images or external assets.

An identical editable copy is saved at
`/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/tof-nof-evaluation-workflow-v1.excalidraw.md`.
The copies were verified byte for byte at delivery. They are not automatically
synchronized; mirror later edits explicitly to prevent drift.

## Scope and reading order

This figure implements [the agreed handoff](implementation-handoff.md), using
the palette, typography, role colors, dotted internal boundaries, and arrow
conventions of the transcript-generation workflow v4. Judge blue replaces
simulator blue. It documents existing behavior; evaluator code is unchanged.

Read the main column from top to bottom. Steps 4–5 show the turn schedule:
evaluate shared T1 once, reuse its consensus label and evidence in all four
scenarios, then evaluate each scenario's T2–T5 separately. Steps 6–9 expand the
same judging operation called for each of those 17 unique turns. The three
judge boxes denote independent decisions, not concurrent API execution.

Each completed consensus turn yields a checkpoint. After the schedule finishes,
the four trajectories yield per-item/per-scenario metrics, followed by summaries
grouped by target model and scenario. The three report boxes depict the report
bundle: the CSV files retain the trajectories and judge evidence accumulated
earlier; `summary.json` contains their aggregates. Side panels explain bounded
retries, compatible checkpoint resume, interpretation, and source modules.

The example `[1, 1, 0, 0, 1]` has ToF = 2 and NoF = 2, with initial_correct = 1,
final_correct = 1, one downward flip, and one upward flip. The labels measure
resistance to the active false premise, rather than total answer quality.

## Rebuild and mirror

From the project root, using Python 3.13+ (standard library only):

```sh
python3 docs/workflow-tof-nof-v1/build_diagram.py
```

This overwrites the project Excalidraw source and SVG. Direct edits to the
Excalidraw file are not imported into the builder; update the builder if future
rebuilds should preserve them. Review the SVG before mirroring the source:

```sh
cp docs/tof-nof-evaluation-workflow-v1.excalidraw.md "/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/tof-nof-evaluation-workflow-v1.excalidraw.md"
cmp docs/tof-nof-evaluation-workflow-v1.excalidraw.md "/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/tof-nof-evaluation-workflow-v1.excalidraw.md"
```

## Verification

- Parsed the scene: 190 unique eight-character element IDs; all 104 text blocks
  match both the Markdown text index and SVG labels.
- Checked canvas bounds and measured grouped text against its cards. Rendered
  the SVG and visually inspected the full figure and detailed sections.
- Rebuilding produces identical source and SVG bytes. The Obsidian copy matches
  the project source byte for byte.
- Python syntax compilation passes. This project has no configured static
  typechecking command. No new behavior or test seam is introduced by the figure.
- Existing evaluator tests: 19 passed. Full existing suite: 152 tests, 150 passed,
  2 failed. Both failures (`test_cli` and `test_models`) expect 2048 simulator
  tokens; the pre-existing uncommitted `transcript_generation/models.py` change
  sets 8192. That change is outside this diagram's scope and commit.

Review baseline: `cb6c0c92698c692205ce04d1035ada6acb25bf70`, the current branch's
HEAD at the start of implementation. Review covers only this figure and its
supporting files; unrelated working-tree changes are excluded.

## Standards

Independent review: 0 findings. No documented standards violations or actionable
baseline smells. The builder follows the existing artifact convention,
centralizes common drawing operations, and uses loops for repeated cards.
Deliverables are in the project, with the explicitly required Obsidian copy.

## Spec

Independent review: 0 findings. All handoff requirements are represented. The
shared pipeline, T1 reuse, later-turn schedule, fresh-run request count, output
conventions, bounded retries, checkpoint timing, and metric definitions match
the evaluator. Both editable copies are identical; evaluator code is unchanged.

Review totals: Standards 0; Spec 0. No outstanding findings on either axis.
