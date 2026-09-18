# False-presupposition resistance experiment

This project generates five-turn dialogues for four 2×2 false-presupposition
scenarios and evaluates the target responses with ToF/NoF labels.

The `humanlike` simulator condition uses noisy conversational language: each
follow-up should contain multiple small, recoverable surface imperfections,
usually two or three, such as fillers, informal contractions, lowercase starts,
minor typos, grammar omissions, informal punctuation, or short fragments. This
is an operational style manipulation, not a human-annotated naturalness score.
Formal pushbacks reuse the validated canonical baseline directly. Semantic and
surface-style validators keep the false premise and its protected anchors intact.
The target judge silently normalizes harmless surface noise before applying the
existing false-presupposition criterion.

## Default models

| Role | Default |
| --- | --- |
| Simulator | `openai/qwen3.8` |
| Targets | `oski/inferenz-gpt-oss-120b`, `qwen3.5:27b`, `openai/qwen3.8` |
| Semantic validator | `deepseek-r1:32b` |
| Judge 1 | `oski/inferenz-mistral-small-4-119b` via KI4BUW |
| Judge 2 | `deepseek-r1:32b` via KI4BUW |
| Judge 3 | `gpt-5.6-sol` via the official OpenAI Responses API |

The default OpenAI-compatible endpoint is `https://llm.ki4buw.de/v1`.
Set `KI4BUW_API_KEY` for generation and the first two judges, and set
`OPENAI_API_KEY` for the third judge. `KI4BUW_BASE_URL` and `OPENAI_BASE_URL`
are optional provider-specific overrides. The evaluator also accepts the
provider-specific CLI flags `--ki4buw-api-key`, `--ki4buw-base-url`,
`--openai-api-key`, and `--openai-base-url`. Ambiguous shared `JUDGE_*`
credentials are intentionally not used by the mixed-provider panel.

Generated transcripts use the model-first layout below. Model names are made
safe for directory names by replacing `/` and `\\` with `__`.

```text
transcript-v3/<sanitized-target-model>/<scenario>/<item-id>.json
```

## Commands

Generate transcripts:

```bash
uv run python -m transcript_generation
```

Evaluate complete four-scenario transcript panels with all three judges:

```bash
uv run python -m tof_nof_evaluation
```

The evaluator checks exact shared turn-1 equality before model calls, scores that
turn once per judge, and reuses the complete decisions across all four scenarios.
Later turns receive the dialogue prefix only. Each judge returns a decision type
and reason; the binary label is derived in code. The primary trajectory is the
unweighted majority of the three labels, and a valid `2:1` decision is final.

For a fixed three-item pilot, select the source IDs before inspecting results:

```bash
uv run python -m tof_nof_evaluation \
  --item-id ITEM_1 --item-id ITEM_2 --item-id ITEM_3
```

The evaluator treats the selected items and target models as a required Cartesian
product, so the command above must find all 9 complete item × target panels before
the first judge call. Without filters, it requires every item in
`data/usable_false_presupposition_data.jsonl` for all three configured targets.
`--limit N` limits source items, not individual panels, so it still requires all
selected targets for each of those `N` items.

After the pilot, inspect every `2:1` decision, every `mixed` and
`unclear_or_nonresponsive` decision, and any unreasonable reasons or persistent
parse failures. If the shared rubric needs a change, revise it once, bump/freeze
the prompt and schema versions, and rerun the same three items. Use that frozen
version for the full evaluation; the pilot is a qualitative check, not human
validation.

Consensus artifacts are written without overwriting older single-judge output:

```text
results-v3-consensus/
├── checkpoints/<target-model>/<item-id>.json
└── consensus/<target-model>/<scenario>/
    ├── tof_nof.csv
    └── tof_nof_judgments.csv
```

`tof_nof.csv` contains the consensus labels, ToF, NoF, initial/final correctness,
and directed flip counts. `tof_nof_judgments.csv` retains every judge's provider,
model, decision type, derived label, reason, raw output, and provider metadata.

Run the paired 2×2 analysis after every selected item/target has all four scenarios:

```bash
uv run python factorial_analysis.py
```

The analysis keeps target models separate, uses consensus ToF/NoF for the paired
bootstrap, and reports pairwise binary agreement, Cohen's kappa, unanimous versus
`2:1` decisions, and disagreement examples. These are judgments by three LLMs
under a fixed rubric; there is no human-gold validation. Treat style and history
as generation conditions rather than perfectly isolated causal interventions.
