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
The judge is instructed to ignore harmless stylistic noise when evaluating the
current assistant response against the false-presupposition resistance rubric.

## Default models

| Role | Default |
| --- | --- |
| Simulator | `openai/qwen3.8` |
| Targets | `oski/inferenz-gpt-oss-120b`, `qwen3.5:27b`, `openai/qwen3.8` |
| Semantic and surface-style validator | `deepseek-r1:32b` |
| Judge alias `qwen3.8` | `openai/qwen3.8` via KI4BUW (`openai-compatible`) |
| Judge alias `gpt-6-sol` | `gpt-6-sol` via OpenAI Responses (`openai`) |
| Judge alias `gpt-oss-120b` | `oski/inferenz-gpt-oss-120b` via KI4BUW (`openai-compatible`) |

The judges run independently and retain disagreements. There is no majority or
averaged label in new outputs. Only the three judges above are supported. Retired GPT-5.4-mini and Mistral
records remain untouched. Importing old consensus checkpoints is no longer supported.

Set `KI4BUW_API_KEY` for generation or pending Qwen/GPT-OSS judgments; set
`OPENAI_API_KEY` for pending GPT-6-sol judgments. During evaluation, only providers
with missing work need credentials. Completed evaluation resumes, exports, and
analysis need no credentials and construct no clients. The generation CLI requires
a KI4BUW API key even with `--resume`. `KI4BUW_BASE_URL` defaults to
`https://llm.ki4buw.de/v1`; `OPENAI_BASE_URL` is optional. Provider-specific CLI
flags override these settings. Ambiguous shared `JUDGE_*` credentials are unused.

All three judges default to **16,384 tokens**. Qwen and GPT-6-sol use reasoning
effort `xhigh`; GPT-OSS uses `high`.
GPT uses no temperature; Qwen and GPT-OSS use temperature `0.0` and
`allowed_openai_params=["reasoning_effort"]`.

Offline export and analysis accept stored positive token budgets and nonnegative
parse-retry limits, but the saved reasoning effort, temperature, seed, and allowed
parameters must match the current validation policy. Older Qwen or GPT-6-sol checkpoints using
`high` are rejected, including during offline export and analysis. Resume requires
an exact configuration match; when settings change, start in a separate output
root.

The visible decision determines the binary label. Full raw responses and metadata
remain available for auditing. Checkpoint identity includes the full provider,
model, endpoint, JudgeConfig, prompt version and generated-message hash, plus
source-file hashes. A peer judge's presence or order is not part of identity.

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

Rerun the same command with `--resume` after a failure. Every successful judgment
is saved atomically, so completed work is reused. The failure identifies its judge
and turn; other judges can be run separately. Persistent 524s despite streaming
require gateway/backend investigation.

Generated transcripts use the model-first layout below. Model names are made
safe for directory names by replacing `/` and `\` with `__`.

```text
transcript-v3/<sanitized-target-model>/<scenario>/<item-id>.json
```

## Commands

Generate transcripts:

```bash
uv run python -m transcript_generation
```

Evaluate all three independent judges (default), or select one judge, target, and item:

```bash
uv run python -m tof_nof_evaluation --judge qwen3.8 \
  --target-model 'qwen3.5:27b' --item-id 2019_b-04245
# Resume only this judge. Omit --judge to select all three.
uv run python -m tof_nof_evaluation --judge qwen3.8 \
  --target-model 'qwen3.5:27b' --item-id 2019_b-04245 --resume
```

`--judge`, `--target-model`, and `--item-id` can be repeated. Unknown or duplicate
judge aliases fail before writes/calls. A panel is one selected item × target ×
all four five-turn scenarios. Shared turn 1 must be identical and is judged once:
**17 unique judgments per judge/panel**, with additional calls possible for retries.
All selected sources and checkpoints are preflighted before any paid call.
Without target/item filters, all dataset items and the three
configured targets are required. `--limit` selects item IDs in lexical order; it
does not necessarily select the fixed five pilot IDs.

An existing selected checkpoint requires `--resume` or a new output root. Each
judge namespace accepts one configuration. Changed source, prompt, endpoint, or
budget cannot silently replace or reuse evaluations.

```text
results-v3-independent/
  judges/<judge-alias>/
    manifest.json
    checkpoints/<target-component>/<item-id>.json
    scores/<target-component>/<scenario>/tof_nof.csv
    scores/<target-component>/<scenario>/tof_nof_judgments.csv
    summary.json
```

Every score row identifies its judge and contains its five labels, ToF, NoF,
initial/final correctness, and upward/downward flips. Judgment rows retain source
text, decisions, reasons, metadata, and raw responses. Summaries declare
`label_source: single_judge`. ToF counts leading 1s before the first 0; NoF counts
adjacent changes. Initial failures remain included; a 0 means failure to meet the
resistance rubric, not necessarily endorsement of the false premise.

Exports rebuild cumulatively from all validated checkpoints in the affected
namespace. Later batches retain earlier rows. Only complete five-turn trajectories
enter score tables; summaries report partial coverage and complete panel counts.

Export and analyze independent judgments offline:

```bash
uv run python -m tof_nof_evaluation export --output-dir results-v3-independent
uv run python factorial_analysis.py \
  --results-dir results-v3-independent --output-dir analysis-v3-independent
# Observed means and sample standard deviations, without bootstrapping:
uv run python factorial_analysis.py \
  --results-dir results-v3-independent --output-dir analysis-v3-independent \
  --no-bootstrap
```

Previously imported independent checkpoints remain supported when their schema,
sources, prompts, and judge configuration pass current validation. Source
checkpoint provenance is preserved on resume. Run from this experiment repository
so stored relative source paths resolve correctly.

Analysis defaults to available judges and targets, accepts repeatable `--judge`
and `--target-model`, and requires all four complete scenarios per analyzed item.
It writes separate paired item-bootstrap effects under `analysis-v3-independent/
judges/<alias>/`. Select exactly two judges with repeatable `--judge` for a
comparison. It uses common complete panels, deduplicates shared
turn 1, validates matching evidence, and reports binary agreement, kappa, decision
cross-tabs, both reasons, trajectory differences, and effects recomputed on the
intersection. Judges are alternative measurements, never pooled observations.
Empty intersections and undefined kappa are explicit. Analysis supports independent
judge results only; legacy consensus analysis and import have been removed.

Condition summaries report the observed mean and sample standard deviation for
ToF, NoF, initial/final correctness, and downward/upward flips. Correctness means
retain the `initial_correct_rate` and `final_correct_rate` names; deviations use
`std_<outcome>`. Factor effects report the mean per-item paired contrast in
`estimate` and its sample standard deviation in `std`, including effects on the
common panels in judge comparisons. Standard deviations use `ddof=1` (divide by
`n - 1`) and are null in JSON or empty in CSV when there is only one item. They
describe variation across observed items, rather than uncertainty in the mean.

These descriptive statistics do not use bootstrap samples. The default still
adds 10,000 paired bootstrap replicates for 95% confidence intervals. Use
`--no-bootstrap` (or `--bootstrap-replicates 0`) to skip resampling entirely;
confidence levels, interval bounds, bootstrap seed, and bootstrap method are then
null in JSON and empty in CSV where applicable.

All evaluation/export writers use an OS advisory lock per judge namespace.
Analysis takes reader locks and refuses an active writer. Different judges can run
in separate processes; overlapping access to the same judge fails clearly. Do not
read CSV files directly while a writer is active; use the locking analysis command.
After an interrupted export, run `export` to rebuild every derived file.

Agreement is not accuracy. Human-reviewed held-out validation and real-judge
boundary calibration remain future work. Five pilot items cannot establish
generalizable style/history effects.
