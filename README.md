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
| Judge 1 | `gpt-5.4-mini` via the official OpenAI Responses API |
| Judge 2 | `openai/qwen3.8` via KI4BUW |
| Judge 3 | `oski/inferenz-mistral-small-4-119b` via KI4BUW |

The default OpenAI-compatible endpoint is `https://llm.ki4buw.de/v1`.
Set `KI4BUW_API_KEY` for generation and judges 2 and 3, and set
`OPENAI_API_KEY` for judge 1. `KI4BUW_BASE_URL` and `OPENAI_BASE_URL`
are optional provider-specific overrides. The evaluator also accepts the
provider-specific CLI flags `--ki4buw-api-key`, `--ki4buw-base-url`,
`--openai-api-key`, and `--openai-base-url`. Ambiguous shared `JUDGE_*`
credentials are intentionally not used by the mixed-provider panel.

All judges use reasoning with an initial budget of **8192 tokens**:

| Judge | Reasoning request | Temperature |
| --- | --- | --- |
| `gpt-5.4-mini` | `reasoning={"effort": "xhigh"}` | Omitted |
| `openai/qwen3.8` | `reasoning_effort="xhigh"` | `0.0` |
| `oski/inferenz-mistral-small-4-119b` | `reasoning_effort="high"` | `0.0` |

Both KI4BUW requests include `allowed_openai_params=["reasoning_effort"]`
through the SDK's extra request body. OpenAI temperature is omitted because
[GPT-5.4 guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4)
only permits it with reasoning disabled;
[GPT-5.4 Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
supports `xhigh`. Unsupported configurations fail; API retries retain the same
reasoning settings. Inspect finish reasons and reasoning-token usage in
`judge_metadata` and the raw response before increasing the token budget.

Mistral's leading `[THINK]...[/THINK]` blocks are removed before strict JSON
parsing. Only the cleaned visible output is stored in `judge_output` or included
in a parse-retry prompt; an unfinished thinking block provides no visible output.
Malformed responses with unexpected thinking-block placement are also omitted
from retry prompts to prevent replaying hidden reasoning.
The complete provider payload, including Mistral's inline thinking and Qwen's
separate `reasoning_content`, stays in `judge_raw_response`. Decisions come only
from the visible JSON's `decision_type`. Checkpoint identity includes the ordered
models/providers, reasoning effort, temperature policy, gateway allowlist, and
other judge settings; incompatible checkpoints cannot resume.

### Gateway timeouts during evaluation

KI4BUW judges use streaming Chat Completions. The evaluator accumulates the
complete response before parsing it; visible content, separate reasoning fields,
finish reasons, and usage are retained in the assembled `judge_raw_response`.
An interrupted stream or one without a finish reason is retried, never scored
as a completed answer. Each attempt closes its stream before retrying.

HTTP 524 means the gateway connected to its backend but did not receive a timely
response. Healthy model status does not rule out a slow request or server queue.
Streaming helps when the gateway forwards chunks promptly; it cannot fix a
backend that sends no data before the proxy deadline. Raising
`--request-timeout-seconds` does not raise Cloudflare's timeout.

API retries honor `Retry-After` (seconds or HTTP date) and the gateway's JSON
`retry_after`. For 524, the first retry waits at least 120 seconds; subsequent
retry backoff doubles, with the exponential factor capped at 64. The default is
one retry per API call; `--api-retries N` sets the retry limit. Logs name the
judge and delay. Final failures retain the original exception and add the judge,
turn, and elapsed time. Invalid-decision diagnostics include the finish reason,
reasoning-token count when supplied, and visible-output length without printing
reasoning text.

Rerun the same evaluation command with `--resume` after a failure. Completed
consensus turns remain reusable: this transport change does not change judge
settings, rubric, or checkpoint identity. A turn is saved only after all three
judges succeed, so an interrupted turn still reruns all three judges. Persistent
524s despite streaming require KI4BUW to investigate backend latency or proxy
configuration; provide the error's timestamp and Cloudflare ray ID.

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
the prompt version (change the checkpoint schema version only for structural changes), and rerun the same three items. Use that frozen
version for the full evaluation; the pilot is a qualitative check, not human
validation.

Consensus artifacts are written without overwriting older single-judge output:

```text
results-v3-consensus-rubric-v2/
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
uv run python factorial_analysis.py --results-dir results-v3-consensus-rubric-v2
```

The analysis keeps target models separate, uses consensus ToF/NoF for the paired
bootstrap, and reports pairwise binary agreement, Cohen's kappa, unanimous versus
`2:1` decisions, and disagreement examples. These are judgments by three LLMs
under a fixed rubric; there is no human-gold validation. Treat style and history
as generation conditions rather than perfectly isolated causal interventions.
