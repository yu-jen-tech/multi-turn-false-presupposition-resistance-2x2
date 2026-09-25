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
| Judge alias `qwen3.8` | `openai/qwen3.8` via KI4BUW (`openai-compatible`) |
| Judge alias `gpt-6-sol` | `gpt-6-sol` via OpenAI Responses (`openai`) |
| Judge alias `gpt-oss-120b` | `oski/inferenz-gpt-oss-120b` via KI4BUW (`openai-compatible`) |

The judges run independently and retain disagreements. There is no majority or
averaged label in new outputs. Historical Mistral records remain unchanged. The `gpt-5.4-mini` alias remains
available for historical results; default runs use `gpt-6-sol`.

Set `KI4BUW_API_KEY` for generation or pending Qwen/GPT-OSS judgments; set
`OPENAI_API_KEY` for pending GPT judgments. Only providers with missing work need
credentials. Completed resumes, legacy imports, exports, and analysis need no
credentials and construct no clients. `KI4BUW_BASE_URL` defaults to
`https://llm.ki4buw.de/v1`; `OPENAI_BASE_URL` is optional. Provider-specific CLI
flags override these settings. Ambiguous shared `JUDGE_*` credentials are unused.

All three judges now default to **16,384 tokens** and reasoning effort `high`.
GPT uses no temperature; Qwen and GPT-OSS use temperature `0.0` and
`allowed_openai_params=["reasoning_effort"]`. Historical `xhigh` pilot records
remain readable for offline import, export, and analysis. Their saved configuration
differs from the new live defaults, so start new `high` runs in a separate output
root. The rubric, streaming, backoff, JSON-fence parsing, and parse retries are
unchanged.

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
safe for directory names by replacing `/` and `\\` with `__`.

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
all four five-turn scenarios. Shared turn 1 must be identical and costs one call:
**17 calls per judge/panel**. All selected sources and checkpoints are preflighted
before any paid call. Without target/item filters, all dataset items and the three
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

Import and analyze saved v2 judgments offline:

```bash
uv run python -m tof_nof_evaluation import-legacy \
  --input-dir results-v3-consensus-v2-16k --output-dir results-v3-independent-v2-16k
uv run python -m tof_nof_evaluation export --output-dir results-v3-independent-v2-16k
uv run python factorial_analysis.py \
  --results-dir results-v3-independent-v2-16k --output-dir analysis-v3-independent-v2-16k
```

Imports preserve original settings, labels, reasons, raw responses, and source
checkpoint hashes. Identical imports are idempotent; conflicts fail without
replacement. The old results stay untouched. Run from this experiment repository
so stored relative source paths resolve correctly.

Analysis defaults to available judges and targets, accepts repeatable `--judge`
and `--target-model`, and requires all four complete scenarios per analyzed item.
It writes separate paired item-bootstrap effects under `analysis-v3-independent/
judges/<alias>/`. Select exactly two judges with repeatable `--judge` for a
comparison. It uses common complete panels, deduplicates shared
turn 1, validates matching evidence, and reports binary agreement, kappa, decision
cross-tabs, both reasons, trajectory differences, and effects recomputed on the
intersection. Judges are alternative measurements, never pooled observations.
Empty intersections and undefined kappa are explicit. To reproduce historical
consensus analysis, use `--input-format legacy-consensus` with its old results root
and a separate output directory.

All evaluation/import/export writers use an OS advisory lock per judge namespace.
Analysis takes reader locks and refuses an active writer. Different judges can run
in separate processes; overlapping access to the same judge fails clearly. Do not
read CSV files directly while a writer is active; use the locking analysis command.
After an interrupted export, run `export` to rebuild every derived file.

See [the current workflow](docs/independent-judge-workflow.md) and
[the pilot comparison](docs/independent-judge-pilot-comparison.md). The v1 workflow,
audit, and rubric proposal documents are historical. Agreement is not accuracy;
human-reviewed held-out validation and real-judge boundary calibration remain
future work. Five pilot items cannot establish generalizable style/history effects.
