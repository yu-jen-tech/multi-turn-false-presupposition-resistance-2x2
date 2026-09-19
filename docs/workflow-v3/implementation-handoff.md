# Transcript-generation workflow v3 — implementation handoff

## Next-session objective

Implement a new Excalidraw workflow diagram, `transcript-generation-workflow-v3.excalidraw.md`, beside the current v2 source:

`/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/transcript-generation-workflow-v2.excalidraw.md`

Do not modify v2. The v3 should make the transcript-generation process easy to understand for a thesis reader who knows basic machine learning but does not know this codebase.

## Design decisions already settled

- Use one landscape overview, readable both in Obsidian and as a thesis-PDF figure.
- Put shared setup at the top, then use a 2×2 scenario matrix:
  - rows: formal / humanlike
  - columns: without simulator history / with simulator history
- Use reader-friendly scenario labels, with exact implementation names in smaller text if useful.
- Keep the experimental design visually dominant; do not branch the main diagram by target model.
- Use one readable sans-serif font. Use monospace only for notation such as `q`, `r₁`, and `T2`.
- Use colors for process/model roles, not for the experimental factors:
  - shared/input
  - simulator
  - validator
  - target model
  - explanatory note
- Add a small connector legend:
  - solid arrow = data flow
  - loop arrow = regeneration or feedback
  - dotted boundary = hidden/internal information
- Use numbered conceptual steps, not a separate number for every retry or turn.
- Every important operation box should use this compact structure:

  `operation`

  `Reads → Produces`

  `short purpose`

- Use “neutral semantic draft (baseline)” in the main labels rather than relying on “baseline” alone.
- Show a compact `T2 confusion → T3 reassertion → T4 anecdote → T5 skepticism` sequence. Expand T2 as the representative loop; do not repeat four full pipelines.
- Show one reusable validation/regeneration pattern rather than duplicating every failure loop.
- Include a short footer explaining what is shared and what differs.
- Do not include full prompts, validator JSON fields, model IDs, retry counts, or trace metadata.
- Save the new file in this folder. Do not implement it until the user explicitly says `implement`.

## Required workflow semantics

### Shared setup

Show one shared path before the split:

`Original question q → target response r₁ → hidden control data`

The hidden-data note should mention false-premise alternatives, correction alternatives, and scenario metadata. Do not print the actual dataset alternatives.

`r₁` is generated once per target model and reused across all four scenarios.

### Without simulator history

Show that the simulator does not receive earlier target-assistant messages. Be precise: the simulator may receive the original question and earlier generated user pushbacks, but not earlier target responses.

The four turns are prepared sequentially before the target transcript loop:

1. Generate neutral semantic drafts for T2–T5.
2. Run format/length/lexical checks and semantic validation.
3. Formal condition: reuse the validated neutral draft as the formal pushback.
4. Humanlike condition: rewrite the validated neutral draft with mild surface noise.
5. Validate the styled pushback for semantic preservation and requested surface style.
6. Store four fixed pushbacks per style; these can be reused across target models.
7. Run the target transcript loop with the fixed pushbacks. The target sees the evolving visible dialogue; the simulator is not called during this loop.

Use a compact sequence strip for `T2 → T3 → T4 → T5` rather than four repeated pipelines.

### With simulator history

Show a per-turn generation loop:

1. The simulator reads the original question plus the current visible dialogue, including the latest target response.
2. Generate a neutral semantic draft for the next strategy.
3. Run format/length/lexical checks and semantic validation.
4. Formal condition: reuse the validated neutral draft.
5. Humanlike condition: rewrite it with mild surface noise.
6. Validate the styled pushback.
7. Put the validated pushback into the visible transcript.
8. Generate the next target response.
9. That target response becomes visible context for the next simulator call.

Repeat through T5, then stop. With-history pushbacks are adaptive and are not shared across target models in the same way as the no-history pushbacks.

## Style and validation meaning

Make the style difference explicit:

- Formal: polished, grammatical, complete wording; reuse the validated semantic draft.
- Humanlike: same meaning and turn strategy, but mild recoverable surface noise; rewrite the draft.

Make validation roles explicit:

- Semantic validation checks preservation of the false premise, turn strategy, pressure goal, and absence of correction leakage/topic change.
- Surface-style validation checks formal or humanlike realization while protecting meaning-bearing anchors.
- Deterministic format/length/lexical checks can be shown as a small label inside the validation gate.
- Failed candidates are discarded and regenerated.

Target responses should have only a compact note such as “usable completion check; invalid/truncated response retries.” Do not give target-response validation the same visual weight as pushback validation.

## Suggested visual hierarchy

1. Title: `Transcript generation workflow`
2. Shared setup band across the full canvas.
3. Two history columns, each containing formal and humanlike rows.
4. Within each row, show the relevant generation/styling/validation path and target-visible loop.
5. Use clear labels for simulator-visible versus target-visible information.
6. Finish with the shared/different summary footer:
   - all scenarios preserve the same false premise and turn strategies;
   - formal versus humanlike changes surface realization;
   - with versus without history changes simulator context and generation timing.

## Success criterion

A reader should be able to answer these questions without reading the implementation:

1. What is shared across all four scenarios?
2. What changes between formal and humanlike transcripts?
3. What changes between with-history and without-history transcripts?
4. What does the simulator see?
5. What does the target model see?
6. When is each pushback generated?
7. What happens when validation fails?

## Relevant implementation references

- Scenario definitions and exact names: `multi-turn-false-presupposition-resistance-2x2/transcript_generation/models.py`
- Generation order and reuse behavior: `multi-turn-false-presupposition-resistance-2x2/transcript_generation/batch.py` and `generator.py`
- Prompt context and style instructions: `multi-turn-false-presupposition-resistance-2x2/transcript_generation/prompts.py`
- Validation data model and checks: `multi-turn-false-presupposition-resistance-2x2/transcript_generation/validation.py`
- Current visual reference: `/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/transcript-generation-workflow-v2.excalidraw.md`

## Status

This session only clarified and documented the design. No v3 diagram has been created or edited.

## Suggested skills

- `visualize:visualize` — optional, for visual inspection or layout QA after creating v3.
- `grilling` — only if the implementation reveals a genuinely new design ambiguity; the current design decisions are already settled.

