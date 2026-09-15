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
| Judge | `oski/inferenz-mistral-small-4-119b` via `openai-compatible` |

The default OpenAI-compatible endpoint is `https://llm.ki4buw.de/v1`.
Set `KI4BUW_API_KEY` for generation and for the default judge transport.
`KI4BUW_BASE_URL` overrides the built-in endpoint; an explicit `--base-url`
or `--judge-base-url` takes precedence. The judge also accepts the existing
`JUDGE_API_KEY` and `JUDGE_BASE_URL` overrides. The optional official OpenAI
judge run uses `JUDGE_API_KEY` or `OPENAI_API_KEY`.

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

Evaluate the stored transcripts with the default judge:

```bash
uv run python -m tof_nof_evaluation
```

Optionally evaluate the same stored transcripts with the official OpenAI
provider and GPT-5.5. This writes results under a separate judge-model
directory:

```bash
uv run python -m tof_nof_evaluation \
  --judge-provider openai \
  --judge-model gpt-5.5
```
