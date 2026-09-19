# Transcript-generation workflow v4 — implementation handoff

## Next-session focus

Implement a v4 version of the transcript-generation workflow diagram. Do not
modify v3. The next agent should use this document together with the existing
v3 artifacts and implementation handoff as the source of truth.

## Discussion outcome

The v3 diagram repeats much of the FORMAL and HUMANLIKE workflow. The agreed
v4 simplification is to keep one shared main workflow column inside each
history condition and branch only where the surface style is realized.

The user confirmed all decisions below and requested that implementation wait
until the next session explicitly receives the instruction `implement`.

## Agreed v4 structure

Use two main panels:

1. WITHOUT SIMULATOR HISTORY
2. WITH SIMULATOR HISTORY

Each panel contains one shared workflow path for semantic draft generation,
format/length/lexical checks, and semantic validation. At the style-realization
step, branch into:

- **FORMAL** — reuse the validated neutral semantic draft as a polished,
  grammatical, complete pushback.
- **HUMANLIKE** — rewrite the validated neutral semantic draft with mild,
  recoverable surface noise while preserving meaning and turn strategy.

The two branches merge again at styled-pushback validation. The validation
gate is shared visually but should make its style-specific criteria clear:
FORMAL checks polished grammatical wording; HUMANLIKE checks the requested
mild surface noise while protecting meaning-bearing anchors. Failed candidates
are discarded and regenerated.

## Meaning of the two history panels

WITHOUT SIMULATOR HISTORY:

- Prepare T2–T5 sequentially before the target transcript loop.
- The simulator may receive the original question and earlier generated user
  pushbacks, but it does not receive earlier target responses.
- Store four fixed pushbacks per style and reuse them across target models.
- During the target loop, the target sees the evolving visible dialogue; the
  simulator is not called again.

WITH SIMULATOR HISTORY:

- Generate the next pushback per turn.
- The simulator reads the original question and current visible dialogue,
  including the latest target response.
- Validate the styled pushback, put it into the visible transcript, and then
  generate the next target response.
- Continue through T5. These pushbacks are adaptive and are not shared across
  target models in the same way as the no-history pushbacks.

## Shared content to retain

Keep the shared setup before the history split:

`Original question q → target response r₁ → hidden control data`

Explain that hidden control data contains false-premise alternatives,
correction alternatives, and scenario metadata. Keep `r₁` as generated once
per target model and reused across scenarios where appropriate.

Show that all scenarios preserve the same false premise and turn strategies.
The diagram should make clear that:

- FORMAL versus HUMANLIKE changes surface realization.
- WITH versus WITHOUT HISTORY changes simulator context and generation timing.
- The target sees the visible dialogue, while hidden control data remains
  internal.
- Validation failure causes regeneration.

Use a compact T2 → T3 → T4 → T5 strip and a reusable validation/regeneration
pattern. Do not print full prompts, validator JSON, model IDs, retry counts, or
trace metadata.

## Deliverables

Create a new v4 editable Excalidraw source, SVG, PDF, README, and matching
Obsidian copy, following the v3 artifact convention. Keep v3 unchanged.

Relevant existing references:

- Project v3 source and assets:
  `/Users/Tony/codex-projects/bachelor-thesis/multi-turn-false-presupposition-resistance-2x2/docs/workflow-v3/`
- Obsidian v3 source:
  `/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/transcript-generation-workflow-v3.excalidraw.md`
- Original v3 design handoff:
  `/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/transcript-generation-workflow-v3-implementation-handoff.md`
- Implementation references:
  `multi-turn-false-presupposition-resistance-2x2/transcript_generation/models.py`
  `batch.py`, `generator.py`, `prompts.py`, and `validation.py`

## Acceptance checklist

- The two history panels each have one shared main workflow column.
- FORMAL and HUMANLIKE branch only at style realization.
- The branches merge at styled-pushback validation.
- The style-specific validation meaning is visible.
- The timing/context difference between the two history panels is visible.
- Simulator-visible and target-visible information are clearly distinguished.
- v3 is unchanged.
- The editable source is compatible with Obsidian Excalidraw.
- SVG and PDF are generated and visually checked.

## Suggested skills

- `visualize:visualize` — optional, for visual inspection and layout QA.
- `code-review` — review the generated artifact changes against this handoff.
- `documents:documents` is not needed; the deliverable is an Excalidraw
  diagram and its SVG/PDF exports, not a Word document.
