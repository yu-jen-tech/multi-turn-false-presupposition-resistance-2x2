# Handoff: dialogue UI v2 with independent judge results

## Status and next task

The design is confirmed. Implement a new v2 dialogue UI that lets the user inspect independent judge results for each turn. No v2 page has been implemented yet.

Keep the v1 page unchanged. Use [dialogue-ui-v1.html](../v1/dialogue-ui-v1.html) as the UI base and [the v1 handoff](../v1/HANDOFF.md) for inherited selector, transcript, and conversation behavior.

Suggested page path: `dialogue-ui/v2/dialogue-ui-v2.html`. Keep v2-specific supporting files under `dialogue-ui/v2/`.

## Agreed design

### Selectors and state

- Keep v1's target, item, style, and history controls. Add a judge selector.
- Start with these judge directory IDs from `results-v3-independent/judges/`:
  - `gpt-oss-120b`
  - `qwen3.8`
  - `gpt-6-sol`
- Put the supported judge IDs in one small local config/index file so the list is easy to update if a judge changes. Load the selected judge's configuration from its `manifest.json`.
- Preserve v1's other defaults: target `qwen3.5:27b`, item `2019_b-03118`, style `formal`, history `without_simulator_history`. Default judge: `gpt-oss-120b`.
- Store the judge and all existing selections in URL query parameters and restore them on reload.

### Judge configuration in the left panel

Show the selected judge's provider, model ID, temperature, reasoning effort, and maximum tokens in a compact always-visible card. Put the endpoint, allowed parameters, seed, and parse retries under a “More settings” disclosure. Reflect values from the selected judge's manifest; represent null or absent values as unspecified rather than inventing defaults.

### Per-turn results in the right panel

- Keep the heading “Resistance trajectory” and its five-turn structure, but replace the hand-written v1 trajectory interpretations with data from the selected judge.
- For each turn T1–T5, show the `judge_label` (`1` or `0`), `decision_type`, and `reason`. Keep the reason expanded and visible by default.
- Preserve the exact decision type values from the data: `rejects_or_corrects`, `accepts_or_builds_on`, `mixed`, `qualified_hypothetical`, and `unclear_or_nonresponsive`.
- If no judgment exists for the selected combination, show a clear “No judge result available” state. Do not substitute another judge, target, item, style, or history condition. Keep the selected dialogue visible when its transcript exists.

## Data sources and join keys

Use the user-requested result set at `multi-turn-false-presupposition-resistance-2x2/results-v3-independent/` (not the separate `results-v3-independent-v2-16k/` tree).

- Per-turn labels, decision types, reasons, and turn dialogue fields are in `judges/<judge-id>/scores/<target-slug>/<scenario>/tof_nof_judgments.csv`.
- Judge settings are in `judges/<judge-id>/manifest.json`.
- `<scenario>` is `<style>_<history>`, such as `formal_without_simulator_history`.
- Join a judgment row to the selected dialogue turn by `id`, `target_model`, `scenario`, and `turn`; use the enclosing `<judge-id>` to select the judge result set. CSV turn values are `1` through `5`.
- `tof_nof.csv` is trajectory-level and does not include the per-turn reason. Use `tof_nof_judgments.csv` for the requested display.
- The existing v1 page reads the question and transcript from the experiment data/transcript directories; keep those sources for the center conversation and use the judgments CSV for the right panel.

Current manifest examples: `gpt-oss-120b` has model `oski/inferenz-gpt-oss-120b`, temperature `0.0`, reasoning effort `high`; `qwen3.8` has model `openai/qwen3.8`, temperature `0.0`, reasoning effort `xhigh`; `gpt-6-sol` has model `gpt-6-sol`, provider `openai`, unspecified temperature, and reasoning effort `high`. All three manifests specify `max_tokens` 16384. Read the manifests at runtime so the UI follows the data if settings change.

Coverage is not the same across judges. In the currently inspected data, the v1 default dialogue has five rows for `gpt-oss-120b`; `qwen3.8` and `gpt-6-sol` have no rows for that target combination. The `gpt-6-sol` results currently cover `openai/qwen3.8`. Keep the missing-result state functional because coverage can change.

## Suggested skills

- `prototype` — implement and refine the standalone dialogue UI.
