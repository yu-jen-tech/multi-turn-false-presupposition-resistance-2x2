# Transcript-generation workflow v5 — implementation handoff

## Next-session focus

Implement v5 of the transcript-generation workflow diagram using the agreed
design below. The current session was planning only; no v5 files were created
or edited. The next agent should wait for the user's explicit implementation
instruction before changing files.

## Source of truth

Use the current v4 diagram and its implementation notes as the semantic
baseline:

- Project source: `/Users/Tony/codex-projects/bachelor-thesis/multi-turn-false-presupposition-resistance-2x2/docs/workflow-v4/transcript-generation-workflow-v4.excalidraw.md`
- Project builder: `/Users/Tony/codex-projects/bachelor-thesis/multi-turn-false-presupposition-resistance-2x2/docs/workflow-v4/build_diagram.py`
- v4 README: `/Users/Tony/codex-projects/bachelor-thesis/multi-turn-false-presupposition-resistance-2x2/docs/workflow-v4/README.md`
- Obsidian v4 copy: `/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/transcript-generation-workflow-v4.excalidraw.md`

Preserve the v4 experiment semantics unless the user gives a later change.
Do not modify v4 or earlier versions.

## Agreed v5 design

Create two variants of the same workflow:

1. **A4 portrait version** — the two history panels must still be displayed
   in parallel, side by side. They must be redesigned as narrow, vertically
   flowing columns; do not simply scale the wide v4 diagram down.
2. **A4 landscape version** — keep the two history panels side by side with
   more horizontal room.

Both variants should use one shared setup band, one shared turn-strategy
strip, aligned workflow stages, and one shared legend. The two history panels
should be aligned row by row so readers can compare them directly.

## Main changes from v4

### Integrated validation and regeneration

Remove the large standalone `VALIDATION & REGENERATION` panel at the bottom.
Integrate regeneration into the main paths with simple feedback arrows:

- Gate A failure loops back to semantic draft generation.
- Gate B failure loops back to the relevant style-realization step.
- Passing candidates continue through the normal forward arrow.

Keep the meaning of the two gates clear, but avoid a separate reusable
workflow block.

### More compact rectangles

Make rectangles fit their actual text instead of using unnecessarily wide
fixed widths. Prefer short cards containing only:

- a concise stage label;
- one data-flow sentence;
- at most one short constraint.

Use line wrapping and compact vertical spacing where needed. Aim for normal
labels of approximately 8 pt or larger in the final thesis PDF.

### Explanatory notes

Move secondary explanations into compact yellow `Explanatory note` elements.
Use approximately these shared notes:

- Hidden controls come from the dataset and scenario.
- Without history: fixed pushbacks are prepared before the target loop.
- With history: each pushback uses the current visible dialogue.
- The target sees visible messages, never hidden controls.

Do not repeat the same explanation inside every card.

### Stage labels

Remove the repeated v4 numbers `2`–`7`. Use short labels such as:

`Generate draft → Validate draft → Realize style → Validate pushback → Target loop`

The semantic order and behavior must remain unchanged; only the presentation
should become simpler.

## Semantics that must remain visible

Retain the shared setup:

`Original question q → initial target response r₁ → hidden control data`

Retain the compact `T2 → T3 → T4 → T5` strategy strip.

The diagram must still show:

- the same false premise and turn strategies across scenarios;
- FORMAL versus HUMANLIKE as a surface-realization difference;
- WITHOUT versus WITH SIMULATOR HISTORY as a simulator-context and timing
  difference;
- the target's visible dialogue versus hidden internal controls;
- semantic draft validation before style realization;
- style-specific validation after realization;
- validation failure causing regeneration;
- fixed pushbacks prepared before the target loop without history;
- adaptive per-turn pushbacks with history.

FORMAL should reuse the valid neutral semantic draft as polished, grammatical,
complete wording. HUMANLIKE should rewrite it with mild, recoverable surface
noise while preserving meaning and turn strategy.

## Deliverables

Create a new project directory:

`/Users/Tony/codex-projects/bachelor-thesis/multi-turn-false-presupposition-resistance-2x2/docs/workflow-v5/`

Expected project artifacts:

- editable portrait Excalidraw Markdown source;
- editable landscape Excalidraw Markdown source;
- matching portrait and landscape SVG files;
- matching one-page A4 portrait and A4 landscape PDFs;
- a README explaining the two variants, build commands, and verification.

Copy the editable sources and useful exports to:

`/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/`

Use the v4/v3 artifact convention for stable Excalidraw IDs and text-index
entries. Do not modify `thesis/thesis.tex` automatically; the thesis source
currently uses A4 portrait by default, and the user will choose which figure
variant to include later.

## Verification checklist

- Both variants are exactly one page in their intended A4 orientation.
- The portrait variant still displays both history panels side by side.
- The landscape variant remains readable and spacious.
- Rectangle widths closely match their text content.
- No large standalone validation/regeneration panel remains.
- Gate A and Gate B each have clear failure feedback arrows.
- The two panels have aligned, comparable stages.
- Explanatory notes reduce repeated prose without hiding essential semantics.
- Final-PDF text is readable, targeting approximately 8 pt or larger.
- SVG and PDF text/connectors fit within the page bounds.
- Both Excalidraw Markdown files are compatible with the Obsidian Excalidraw
  plugin.
- v4, v3, and the existing Obsidian sources remain unchanged.

## Suggested skills

- `visualize:visualize` — inspect the portrait and landscape layouts and help
  with visual readability and page-fit QA.
- `pdf:pdf` — render and verify the generated one-page PDF exports if needed.
- `code-review` — review the v5 artifact changes against this handoff after
  implementation.
