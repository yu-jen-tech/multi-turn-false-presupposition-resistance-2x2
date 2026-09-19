# Transcript generation workflow v3

The editable source is `transcript-generation-workflow-v3.excalidraw.md`.
The SVG and single-page landscape PDF contain the same figure. The PDF is
laid out on A3 to keep the detailed labels readable; use the vector SVG or
Excalidraw source when adapting it to another page size.

A matching editable copy is saved beside v2 in the Obsidian vault:
`/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/transcript-generation-workflow-v3.excalidraw.md`.
The v2 file is unchanged. These copies are not automatically synchronized.

## Scope and interpretation

The figure implements `implementation-handoff.md`. It documents the existing
generator; it does not change prompts, generation behavior, or model settings.
The top band shows shared inputs, not the exact batch scheduler order:
`batch.py` can prepare fixed pushbacks before obtaining the initial target
response. Hidden controls come from the dataset and scenario, not from that
response.

The left cells repeat the draft path for readability. They refer to **one
shared set of four validated neutral semantic drafts**, not independently
generated sets per style. All four drafts are prepared first; each style then
gets four validated fixed pushbacks before the target transcript loop. Formal
reuses the passed semantic and formal surface checks from the draft. The
numbered gates describe conceptual checks, not separate API calls.

The right cells generate a new pushback for each turn using the latest target
response. Feedback stops after the T5 response. The target receives the
evolving visible dialogue in all four scenarios; hidden control data never
becomes a target message.

Implementation references: `transcript_generation/models.py`, `batch.py`,
`generator.py`, `prompts.py`, and `validation.py`.

## Rebuild

From the project root:

```sh
python3 docs/workflow-v3/build_diagram.py
rsvg-convert --format pdf --page-width 420mm --page-height 297mm \
  --width 396mm --left 12mm --top 18mm \
  --output docs/workflow-v3/transcript-generation-workflow-v3.pdf \
  docs/workflow-v3/transcript-generation-workflow-v3.svg
```

The builder uses only the Python standard library. PDF export requires
`rsvg-convert`. Rebuilding overwrites the project source and SVG; copy the
editable source to Obsidian explicitly after reviewing it. Direct Excalidraw
edits are not imported back into the builder.

## Verification

- Parsed the JSON scene and checked unique eight-character IDs, the text
  index against the installed Obsidian plugin's parser, and the consistent
  sans-serif font.
- Rendered and visually inspected the full PDF; checked that text remains
  searchable and that the output has one landscape page.
- Python syntax compilation passed. This project has no configured static
  typechecking command; the artifact builder has no external dependencies.
- Existing batch tests: 9 passed.
- Full existing suite: 144 tests, 142 passed, 2 failed. The failures in
  `test_cli` and `test_models` expect a 2048-token simulator default; the
  pre-existing uncommitted `models.py` change sets it to 8192. That change
  is outside this diagram's scope and is excluded from its commit.
- Independent code-review skill reviews: Standards 0 findings; Spec 0 findings.
