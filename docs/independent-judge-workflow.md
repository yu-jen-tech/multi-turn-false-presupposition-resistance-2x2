# Independent judge workflow

Current as of 2026-09-23. Replaces the fixed three-judge execution workflow.
Historical v1 diagrams and audit documents are preserved as historical records.

1. Generate and validate complete four-scenario panels for the selected items and
   targets. The evaluator preflights source evidence, shared turn 1, all selected
   checkpoints, and existing cumulative data before any model request.
2. Select `qwen3.8`, `gpt-5.4-mini`, `gpt-oss-120b`, or all three with repeatable
   `--judge`. Only a judge with pending turns constructs its provider client or
   requires credentials.
3. Acquire the selected namespace locks. Each namespace has one full evaluation
   configuration. Independent processes may write different judges; the same
   namespace rejects overlapping access. OS locks release on process termination;
   the persistent `.lock` file is not a stale-lock marker and must not be deleted.
4. Evaluate judges sequentially. Save shared turn 1 once, then scenario-specific
   prefixes for turns 2–5. Save atomically after every successful judgment. On
   failure, export completed trajectories, report the judge/turn, and exit nonzero.
5. Resume with identical settings and `--resume`. Completed judgments need no new
   calls. A peer judge's presence/order is irrelevant. Existing checkpoints cannot
   be overwritten by omitting `--resume`; choose another root for new settings.
6. Rebuild exports from all compatible checkpoints, including earlier batches.
   Partial trajectories stay only in checkpoints; the summary reports their
   counts. Use offline `export` after an interrupted derived-file write.
7. Analyze each judge × target separately with a paired item bootstrap. Incomplete
   panels fail with their IDs and coverage. Select exactly two judges for a
   comparison. It uses the common item panels for each target, validates
   source/prompt hashes and text, deduplicates turn 1, and recalculates both
   judges' effects on the shared sample.

## Saved 16k pilot

From the experiment repository:

```bash
uv run python -m tof_nof_evaluation import-legacy \
  --input-dir results-v3-consensus-v2-16k --output-dir results-v3-independent-v2-16k
uv run python -m tof_nof_evaluation export --output-dir results-v3-independent-v2-16k
uv run python factorial_analysis.py \
  --results-dir results-v3-independent-v2-16k --output-dir analysis-v3-independent-v2-16k
```

These commands are offline. Import validates legacy majority consistency only to
check source integrity; it never uses majority labels in new results. It locates
judges by provider/model, preserves the original settings, and records source
checkpoint path/SHA-256. No semantic judgment is recomputed or corrected.

The saved pilot used `xhigh`, so it remains available for offline import, export,
and analysis but cannot be resumed with the new `high` live policy. Use a new
output root for new judgments. Select the five pilot IDs explicitly:
`2019_b-03118`, `2019_b-03762`, `2019_b-04245`, `2019_b-05522`, `2019_b-06898`.
Select only available pilot targets (`oski/inferenz-gpt-oss-120b` and
`qwen3.5:27b`); omitting target selection requires all three configured targets.
The default token budget is 16384 for new runs.

## Interpretation

The prompt version remains `tof-nof-multi-judge-v2` because the rubric bytes did
not change. The new checkpoint and result schemas are
`tof-nof-independent-judge-checkpoint-v1` and
`tof-nof-independent-judge-results-v1`. Every output uses `single_judge` labels.

ToF counts leading 1s; NoF counts adjacent binary changes. Keep initially incorrect
trajectories and report initial/final correctness plus directed flips. Zero is a
rubric failure, not necessarily endorsement. Agreement measures consistency,
not correctness; never retry valid decisions until judges agree. Human-reviewed
held-out validation and real-judge calibration remain separate methodology work.
