# Transcript generation workflow v4

The editable source is `transcript-generation-workflow-v4.excalidraw.md`.
The SVG and single-page A3 landscape PDF contain the same figure. Both
exports remain vector graphics; use the editable source for layout changes.

A matching editable copy is saved in the Obsidian vault:
`/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/transcript-generation-workflow-v4.excalidraw.md`.
The project and vault copies are not automatically synchronized. All v3
artifacts, including the existing Obsidian source, are unchanged.

## Scope and interpretation

The figure implements `implementation-handoff.md`, with the prior design
requirements in `../workflow-v3/implementation-handoff.md`. It documents
the existing generator without changing generation behavior or settings.

Each history panel has one shared semantic-draft and validation path.
FORMAL and HUMANLIKE branch only at surface realization and merge at a
shared styled-pushback validation gate. The gate applies separately to each
candidate; the two style texts and their resulting transcripts stay separate.
The numbered gates describe conceptual checks, not separate API calls.
FORMAL reuses the draft's passed semantic and surface-style checks.

Without history, all four neutral drafts are validated sequentially before
styling. One shared draft set supplies both styles. Four fixed pushbacks per
style are prepared before the target transcript loop and reused across target
models. The simulator may receive earlier user pushbacks, but never earlier
target responses. It is not called during the target loop.

With history, the shared column describes a process run separately for each
scenario and target model. The next draft uses the current visible dialogue,
including the latest target response. Each validated pushback enters the
visible transcript before the next target response is generated. The feedback
loop continues through T5, then stops. These adaptive pushbacks are not pooled
across target models or styles.

The setup band shows shared inputs, not exact batch scheduler order:
`batch.py` can prepare fixed pushbacks before obtaining the first target
response. Hidden controls come from the dataset and scenario, not from r₁.
The initial response is generated once per target model and reused across
scenarios. The target sees the evolving visible dialogue in all scenarios;
hidden controls stay internal.

Implementation references: `transcript_generation/models.py`, `batch.py`,
`generator.py`, `prompts.py`, and `validation.py`.

## Rebuild

From the project root:

```sh
python3 docs/workflow-v4/build_diagram.py
rsvg-convert --format pdf --page-width 420mm --page-height 297mm \
  --width 384mm --left 18mm --top 10mm \
  --output docs/workflow-v4/transcript-generation-workflow-v4.pdf \
  docs/workflow-v4/transcript-generation-workflow-v4.svg
```

The builder uses only the Python standard library. PDF export requires
`rsvg-convert`. Rebuilding overwrites the project source and SVG; copy the
editable source to Obsidian explicitly after reviewing it. Direct Excalidraw
edits are not imported back into the builder. The v4 builder is independent
of the v3 builder, so future v4 changes do not require editing v3.

## Verification

- Parsed all 145 editable elements and checked unique eight-character IDs.
  All 86 text blocks match the text-index format used by the installed
  Obsidian Excalidraw plugin and match the SVG labels. One sans-serif font
  is used throughout.
- Rebuilt the source and SVG and verified identical content. Confirmed
  the Obsidian v4 copy matches the project source byte for byte.
- Rendered and visually inspected the SVG and PDF. The PDF has one
  searchable A3 landscape page; text and connectors fit their intended areas.
- Compared v3 file checksums before and after implementation, including
  the existing Obsidian source: unchanged.
- Python syntax compilation passed. The project has no configured static
  typechecking command. No new test seam was needed for this diagram.
- Existing batch tests: 9 passed. Full existing suite: 144 tests,
  142 passed and 2 failed. `test_cli` and `test_models` expect a 2048-token
  simulator default; the pre-existing uncommitted `models.py` change sets
  it to 8192. That change is outside this diagram's scope and commit.

## Standards

Independent review: 0 findings. No documented standards violations or
actionable baseline smells. The v4 builder follows the v3 artifact convention,
shares repeated panel logic, and keeps generated outputs reproducible. Its
independent builder preserves v3 unchanged.

## Spec

Independent review: 0 findings. The diagram satisfies the shared paths,
style-only branches, merged validation gates, context and timing distinctions,
regeneration pattern, and deliverables. SVG and PDF renders are legible;
source and SVG labels match, and the Obsidian editable copy matches byte for
byte. No unrequested behavioral changes are included.

Review totals: Standards 0; Spec 0. No outstanding findings on either axis.
