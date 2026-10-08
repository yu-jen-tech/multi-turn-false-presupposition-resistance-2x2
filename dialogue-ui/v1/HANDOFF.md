# Handoff: dialogue UI v1

## Status

The UI has not been implemented. The next session should implement the agreed design in a new file and leave the existing prototype unchanged.

## User request

Create a new Variant-A-focused dialogue UI prototype based on:

- Existing prototype: `multi-turn-false-presupposition-resistance-2x2/dialogue-ui/v1/dialogue-ui-prototype.html`
- New implementation target: `multi-turn-false-presupposition-resistance-2x2/dialogue-ui/v1/dialogue-ui-v1.html`
- All related files must stay under `multi-turn-false-presupposition-resistance-2x2/dialogue-ui/v1/`.

Do not modify the original prototype.

## Agreed design

### Variants

- Variant A is the functional dialogue view.
- Keep Variant B in the variant switcher as a clearly labeled reserved/empty placeholder for future use.

### Data and selectors

- The experiment will contain 85 questions from `multi-turn-false-presupposition-resistance-2x2/data/usable_false_presupposition_data.jsonl`.
- Each question has 3 target models and 4 scenarios, for 85 × 3 × 4 = 1,020 transcript combinations.
- Target options:
  - `oski/inferenz-gpt-oss-120b`
  - `qwen3.5:27b`
  - `openai/qwen3.8`
- Style options: `formal`, `humanlike`.
- History options: `with_simulator_history`, `without_simulator_history`.
- Use a searchable item picker for all 85 questions. Each option must show `item-id + question`.
- Default selection: target `qwen3.5:27b`, item `2019_b-03118`, style `formal`, history `without_simulator_history`.
- Encode target, item, style, and history in the URL query string and restore them on reload.
- Use the existing JSONL and generated transcript files as the source of truth. Expected relative sources from the new HTML include:
  - `../data/usable_false_presupposition_data.jsonl`
  - `../transcript-v3/...`
- Transcript directory names need the repository's existing encoding for model names (for example, `openai__qwen3.8` and `oski__inferenz-gpt-oss-120b`).
- The page may assume a small local HTTP server so browser `fetch()` can load JSONL/JSON files.
- If a selected transcript does not exist yet, show the selected question and annotations and a clear read-only `Transcript not generated yet` empty state. Never silently show another transcript.

### Left panel

Provide functional controls for target model, item, style, and history. Also show three compact, always-visible, read-only model cards:

- Target: selected model name, temperature, reasoning status, and response word limits.
- Simulator: `openai/qwen3.8`, temperature, JSON pushback output contract, and whether transcript history is supplied.
- Validator: `deepseek-r1:32b`, temperature, and strict JSON validation contract.

When a transcript is available, prefer its metadata for the displayed settings; otherwise show the configured prototype defaults. Target settings vary with the selected target; simulator and validator are fixed for the experiment.

### Main area

- Remove the existing description: `A five-turn dialogue where the user repeatedly pushes the assistant toward an absolute but misleading conclusion.`
- Keep the selected question at the top.
- Show the selected item's presuppositions and corrections directly beneath the question, above the conversation area.
- The Conversation header must show style and history tags, alongside completion status.
- Assistant message labels must include the selected target model, e.g. `Assistant (target model: qwen3.5:27b)`.
- Simulator-generated User message labels must include the simulator model, e.g. `User (simulator model: openai/qwen3.8)`, with the pushback strategy shown beneath it as a tag such as `confusion`, `reassertion`, `anecdote`, or `skepticism`.
- The first dataset User message has no simulator metadata; label it `User (dataset)` and omit the strategy tag.

### Right panel

- Remove the `Research metadata` heading and its turn/source/strategy metadata list.
- Keep the `Resistance trajectory` block unchanged: heading, intro text, five steps, colors, and callout.
- The trajectory is reserved for later `tof_nof_eval` display work; do not redesign or reinterpret it now.

## Known repository facts

- Existing transcript files use schema `false-presupposition-transcript-v3`.
- Current sample item `2019_b-03118` question: `Why can't you create your own ISP to access the internet?`
- Its annotations are stored in the item record's `presuppositions` and `corrections` arrays.
- Existing transcript metadata uses target temperature `0.0` with reasoning enabled and soft/hard response word limits of 120/300; simulator temperature `0.7`; validator temperature `0.0`.

## Verification expectations

- Confirm the original HTML is unchanged.
- Serve the new folder through a local HTTP server and verify that the default view loads.
- Verify all four selectors update the URL and displayed state.
- Verify item options display both ID and question.
- Verify available transcript loading and missing-transcript empty state.
- Verify Variant B remains empty.
- Verify responsive layout and that the right panel contains only the unchanged Resistance trajectory.

## Suggested skills for the next session

- `prototype`: implement and sanity-check this throwaway UI prototype.
- `grilling`: use only if any of the settled design decisions need to be reopened; otherwise the design tree is already resolved.
