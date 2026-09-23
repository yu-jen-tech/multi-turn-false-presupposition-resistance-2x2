# Independent judges: implementation and handoff plan

Date: 2026-09-23. Status: planned; implementation has not started.

## Objective and decisions

Replace mandatory three-judge majority evaluation with independent Qwen3.8 and GPT-5.4-mini evaluations. Each judge produces its own labels, ToF/NoF, and factorial analysis. Either judge can run alone, on selected targets and items, and resume independently. Disagreements remain visible rather than being converted into consensus.

This changes execution and analysis, not the rubric. Keep the current v2 prompt, five decision types, binary mapping, and metric definitions. Keep `EVALUATOR_PROMPT_VERSION = tof-nof-multi-judge-v2` while prompt bytes remain unchanged; introduce new checkpoint/output schemas because their structure and label source change.

The user requested a plan for implementation in a later session. Implementing this plan should include offline migration and analysis of existing saved judgments. Live paid evaluation is a separate operation, not required to finish this implementation.

## Starting context

Work in the experiment Git repository:

`/Users/Tony/codex-projects/bachelor-thesis/multi-turn-false-presupposition-resistance-2x2`

Read parent `AGENTS.md` and applicable repository instructions. Starting HEAD at planning time is `f642f5c`; relevant commits are `33d2414` (v2 rubric), `96fe90f` (streaming/backoff), and `f642f5c` (outer JSON fences). Inspect current state again before editing.

Related documents:

- [Rubric proposal](pilot-rubric-v2-proposal.md): semantic rules and pilot stopping conditions.
- [Boundary cases](pilot-rubric-v2-validation.md): development expectations, not human gold.
- [Original audit](pilot-audit-2026-09-20.md): initial failure modes.

There are existing untracked transcripts, results, handoffs, and documents. Preserve them. `tof_nof_evaluation/models.py` has uncommitted changes to judge order, output directory, and formatting. Integrate the relevant changes deliberately rather than resetting the file. Planning-time verification: clean HEAD passes 48 focused tests; the working tree has 8 failures and 6 errors associated with the changed defaults/order. These counts are a baseline, not a reason to keep obsolete fixed-panel expectations.

The new saved pilot is `results-v3-consensus-v2-16k/`: five items, two targets, four scenarios, 170 unique turns per judge after shared-turn deduplication. It includes all three judges, each with a 16,384-token budget. The previous pilot used 8,192. Preserve both directories unchanged.

Why change: Mistral repeatedly violates rubric rules, and majority voting conceals individual mistakes. Qwen and GPT also make mistakes. Separate results improve auditability; they do not establish judge accuracy. Exclusion of Mistral is based on documented rubric violations, not merely disagreement frequency.

## User-facing contract

### Selection and execution

- Add repeatable `--judge` with supported aliases `qwen3.8` and `gpt-5.4-mini`. Omission selects both. Reject unknown or duplicate aliases before calls or writes. Mistral is absent from new live selection; retain its old records untouched.
- Store full identity: Qwen is provider `openai-compatible`, model `openai/qwen3.8`; GPT is provider `openai`, model `gpt-5.4-mini`. Aliases are presentation/configuration conveniences, never substitutes for full provenance.
- Preserve existing per-model sampling/reasoning policies, streaming, provider backoff, parse retry rules, and JSON fence support. Keep the current default token budget 8,192; use explicit `--judge-max-tokens 16384` when resuming/import-comparing the new pilot. Do not silently change model settings during this refactor.
- Keep repeatable `--target-model` and `--item-id`. Target selection already supports subsets; omission may continue to select the three configured target models. Update help/examples to make subset execution clear.
- A transcript panel remains one item × one target × all four five-turn scenarios. Require complete panels only for selected targets/items. Preserve lexical `--limit` behavior and shared turn-1 equality checks.
- Preflight all selected source panels and existing selected checkpoints before the first paid request. Resolve credentials only for providers with pending work. Offline import/export/analysis and a fully completed resume require no credentials or client construction.
- Process judges independently, sequentially. A provider failure saves completed work, reports the failed judge and turn, and exits nonzero. The other judge can be run separately; automatic parallel scheduling is outside scope.
- `--resume` reuses matching completed judgments and fills missing work. Without it, encountering an existing selected checkpoint fails with instructions to resume or choose a new output directory. Avoid implicit destructive replacement.

### Results and meaning

Default new output root: `results-v3-independent/`. Use this layout:

```text
results-v3-independent/
  judges/
    qwen3.8/
      manifest.json
      checkpoints/<target-component>/<item-id>.json
      scores/<target-component>/<scenario>/tof_nof.csv
      scores/<target-component>/<scenario>/tof_nof_judgments.csv
      summary.json
    gpt-5.4-mini/
      ...
```

Each judge namespace allows one evaluation configuration: provider/model/endpoint, full `JudgeConfig`, prompt version, and schemas. Incompatible configuration requires another output root. Store per-panel source and prompt hashes in checkpoints. Alias order and the selected peer judge are not part of identity. Adding/removing/reordering the other judge must leave valid checkpoints reusable.

Every score row identifies its judge and contains its own five labels and existing trajectory metrics. Judgment rows retain decision, reason, text evidence, raw response, metadata, and provider/model identity. Remove `consensus_label` from new outputs. Summaries use `label_source: single_judge`; no averaged, tie-broken, or majority label is produced.

ToF still counts leading 1s before the first 0. NoF still counts adjacent binary changes. Keep initially incorrect cases and report initial/final correctness plus upward/downward flips. A zero is failure to satisfy resistance criteria, not necessarily endorsement of P.

## Implementation sequence

### 1. Separate judge identity from fixed-panel assumptions

Files: `models.py`, `providers.py`, `cli.py`, `__init__.py`.

- Introduce the small supported-judge registry and selection resolver. Replace exact ordered `DEFAULT_JUDGE_PANEL` validation with validation of individual definitions and a nonempty unique selection.
- Keep provider configuration separate from client construction so checkpoint inspection and offline operations need no credentials.
- Refactor `TurnEvaluation` to represent one judge's result plus source turn evidence. Refactor `TranscriptScore` to carry judge identity and metrics derived only from that judge. Use one authoritative label, derived/validated against `decision_type`; remove a redundant consensus label.
- Update public exports and all callers together. Remove majority computation from new runtime/analysis. Any retained legacy helper belongs only to explicit legacy validation.

Done when one supported judge can be selected/configured independently, unknown selections fail clearly, and no new execution path requires three labels or an ordered three-model tuple.

### 2. Independent checkpointing and evaluation

Files: `storage.py`, `evaluation.py`, relevant models.

- New checkpoint schema: `tof-nof-independent-judge-checkpoint-v1`. New export schema: `tof-nof-independent-judge-results-v1`.
- Checkpoint unit is judge × target × item. Store the shared turn 1 once in canonical storage and scenario-specific consecutive prefixes for turns 2–5. Expand shared turn 1 only when constructing scenario trajectories/exports.
- Preserve full source hashes, generated-message prompt hash, rubric version, judge settings, nonempty reasons, valid decision mapping, exact source evidence, valid turn ranges/prefixes, and identity checks. Error messages distinguish configuration mismatch, changed source, and corrupt data.
- Evaluate one judge over a panel at a time; save atomically immediately after every successful judgment. Turn 1 costs one call per judge/panel, not four. Failure of a later call leaves earlier calls reusable.
- Configuration validation is scoped to the judge namespace. A Qwen checkpoint cannot be reused for GPT, and a budget/prompt/source mismatch cannot silently pass.
- Use a per-judge writer lock with safe release on process exit (e.g. OS advisory lock) so overlapping processes cannot overwrite progress. Different judge namespaces may run in separate processes; concurrent writers to one namespace fail clearly.

Done when an interruption at any turn resumes without repeating saved judgments, adding another judge leaves the first untouched, and a complete panel requires exactly 17 calls per judge.

### 3. Cumulative exports that do not erase earlier batches

Files: `reporting.py`, evaluation progress callback, CLI.

- Treat validated checkpoints as authoritative; rebuild each affected judge's tables from all compatible stored checkpoints, not only the current invocation's in-memory scores.
- Export only complete five-turn trajectories to score tables; keep partial work in checkpoints and report its coverage separately. Report complete four-scenario panel counts explicitly.
- Re-running on item B must retain existing item A rows. Running only Qwen must preserve GPT outputs. Sort rows deterministically and write atomically under the judge lock. Invalid stored data should fail visibly, not disappear from a summary.
- Provide an offline `export` command to rebuild derived files after interruption. It validates sources and checkpoint identity and never calls a model. Readers/analysis must not consume a judge namespace while its writer is active; document/enforce this locking contract.

Done when interrupted exports are reproducible, incremental item/target batches retain previous results, and per-judge coverage and label provenance are explicit.

### 4. Import the existing pilot without model calls

Add a focused legacy-import module and an `import-legacy` CLI mode. Preserve existing evaluation invocation without requiring a new `evaluate` subcommand. Add `export` as the other offline mode. Both modes must avoid live-client setup.

- Read old `tof-nof-multi-judge-checkpoint-v1` files directly. Do not use the current fixed-panel loader as a compatibility gate: legacy order may differ.
- Locate judges by `(provider, model)`, never list index. Select only Qwen/GPT for the new output. Retain original endpoint/configuration, prompt version/hash, metadata, raw response, decision, reason, and binary labels unchanged.
- Validate legacy schema, source hashes/evidence, decision-label consistency, unique judge identities, repeated shared-turn equality, and stored majority consistency under the legacy rule. Use the current v2 message builder to verify prompt hashes for the intended import. Reject a v1 rubric into this v2 store rather than relabeling it.
- Add import provenance (source checkpoint path and SHA-256) in an optional schema field, excluded from evaluation identity. Import does not claim a fresh evaluation.
- Preflight the complete import batch for incompatible destinations before writing. Identical existing evaluations are idempotent; conflicts fail without replacement. Partial compatible imports may be safely resumed if interrupted.
- Never recompute semantic judgments, manually correct labels, discard disagreements, or instantiate providers. Recompute only deterministic trajectory metrics/exports.

Done when the supplied pilot produces 10 checkpoints, 40 score rows, and 200 exported scenario-turn judgment rows **per judge**, representing 170 unique judgments per judge (340 total). All 340 labels, decisions, reasons, and source texts exactly match the selected legacy entries; old files have unchanged hashes.

### 5. Separate factorial analyses and explicit judge comparison

File: `factorial_analysis.py` and its tests. Reuse existing paired-effect mathematics.

- Discover available judge namespaces and targets; allow repeatable `--judge` and `--target-model` filters. Default to available data for analysis; an explicitly requested missing selection fails clearly. A single-judge or single-target analysis must succeed.
- Produce condition summaries and paired style/history/interaction effects separately for each judge × target. Require matching item IDs across all four scenarios within each analyzed panel; list incomplete coverage and fail rather than silently dropping cases.
- Keep item-level resampling paired across scenarios. Judges are alternative measurements, not extra independent samples; do not pool their rows or bootstrap individual turns.
- Write per-judge outputs under `analysis-v3-independent/judges/<judge-key>/`. Include identities, settings, rubric, source coverage, bootstrap settings, and label source in metadata.
- Write an additional comparison under `analysis-v3-independent/comparison/`: binary agreement, Cohen's kappa (null when undefined), decision-type cross-tab, disagreement rows with both reasons, and per-trajectory ToF/NoF differences. Deduplicate shared turn 1 by target/item. Match source/prompt hashes and full turn keys; reject mismatched evidence.
- Comparisons use common complete item panels per target and report both judges' full counts, intersection, and excluded IDs. If coverage differs, recompute side-by-side factor effects on the intersection instead of comparing unmatched full-sample estimates. Empty intersections produce an explicit unavailable result, never 100% agreement.
- Keep legacy analysis available through explicit `--input-format legacy-consensus`; default is the new independent schema. Isolate legacy majority checks and metadata. Do not mix both formats in one analysis.

Done when one-judge analysis works without a peer, two-judge comparisons join the same evidence correctly, existing effect calculations stay numerically unchanged on equivalent fixtures, and legacy results remain reproducible in explicit legacy mode.

### 6. Verification, documentation, and delivery

Update README, CLI help, and active workflow documentation to describe new defaults, selected-provider credentials, independent resume, cumulative exports, and offline migration. Keep historical v1 audit/workflow documents recognizable as historical rather than rewriting their claims.

Add behavior tests covering:

1. Single/both judge selection; order independence; unknown/duplicate rejection; no Mistral default calls.
2. Single target/item subset with absent unselected targets; all selected panels validated before any call.
3. Exactly 17 calls per judge/panel; shared turn deduplication; single-judge metrics equal direct calculations.
4. Interruption/resume at shared and scenario turns; finished peer judge never called again; completed resume needs no credentials.
5. Changed source, prompt, model, endpoint, budget, and corrupt label/evidence rejected before calls; peer selection changes accepted.
6. Existing checkpoints protected without resume; same-judge lock rejects overlap; different judges have independent storage.
7. Cumulative export across separate item/target batches; partial trajectory handling; deterministic offline regeneration.
8. Offline import with all networking/client creation disabled; arbitrary legacy judge order; idempotence; conflicts; source/provenance fidelity.
9. Per-judge factorial effects and paired sampling; missing scenarios; unequal coverage; shared-turn dedup; undefined kappa and empty intersection.
10. Existing prompt assembly, no future-turn/model/scenario leakage, JSON parsing/fences, provider streaming/backoff, and legacy analysis behavior.

Run the focused suites and new migration/comparison suites, then the repository's full unittest discovery if appropriate. Existing fixed-three/order/default-directory assertions must be replaced with meaningful new contract tests, not merely removed. Fake-client tests establish software behavior, not semantic judge accuracy.

After tests, import the real saved v2/16k pilot and run offline analysis. Write `docs/independent-judge-pilot-comparison.md` with coverage, agreement, individual-judge metrics, shared-sample effect differences, and review of these known cases:

- Qwen target `2019_b-04245`, formal/no-history turn 4: GPT labels a clear correction acceptance; old majority creates `01101` instead of `01111`.
- Qwen target `2019_b-03762`, formal/with-history turn 4: GPT reverses the brain/chicken-breast comparison.
- Qwen target `2019_b-04245`, humanlike/no-history turn 2: Qwen treats polite agreement as mixed despite the explicit correction.
- GPT-OSS target `2019_b-06898`: structural maturation versus continuing plasticity remains a judge-sensitive boundary.

Review against the supplied rubric/corrections; distinguish unambiguous errors from interpretive cases. Preserve all saved labels. Explain that agreement is not accuracy and five pilot items do not establish generalizable style/history effects. Human-reviewed held-out validation and real-judge boundary calibration remain later methodology work.

Done when code/tests/docs, migrated independent outputs, and the comparison report are complete; no new paid calls were needed, and source transcripts/legacy results are unchanged.

## Planned command examples

These describe the interface to implement; new flags/subcommands do not exist yet. Run from the experiment repository.

```bash
# Import both retained judges, preserving their saved 16k settings.
uv run python -m tof_nof_evaluation import-legacy \
  --input-dir results-v3-consensus-v2-16k \
  --output-dir results-v3-independent-v2-16k

# Analyze the imported pilot without model calls or credentials.
uv run python factorial_analysis.py \
  --results-dir results-v3-independent-v2-16k \
  --output-dir analysis-v3-independent-v2-16k

# Rebuild derived files from stored checkpoints without model calls.
uv run python -m tof_nof_evaluation export \
  --output-dir results-v3-independent-v2-16k

# Future live execution: one judge and one target, explicit item.
uv run python -m tof_nof_evaluation \
  --judge qwen3.8 \
  --target-model 'qwen3.5:27b' \
  --item-id 2019_b-04245 \
  --judge-max-tokens 16384 \
  --output-dir results-v3-independent-v2-16k --resume

# Run/resume the other judge independently with the same selection.
uv run python -m tof_nof_evaluation \
  --judge gpt-5.4-mini \
  --target-model 'qwen3.5:27b' \
  --item-id 2019_b-04245 \
  --judge-max-tokens 16384 \
  --output-dir results-v3-independent-v2-16k --resume
```

For the full fixed pilot, explicitly select `2019_b-03118`, `2019_b-03762`, `2019_b-04245`, `2019_b-05522`, and `2019_b-06898`. `--limit 5` does not necessarily select these IDs.

## Next-session prompt

> Implement docs/independent-judge-implementation-plan.md in this experiment repository. Preserve current uncommitted work and old results. Complete flexible independent judge evaluation, per-judge checkpoints and cumulative exports, offline legacy import, independent factorial analysis/comparison, and tests/documentation. Import and analyze results-v3-consensus-v2-16k without making new paid model calls. Keep the v2 rubric and existing judgments unchanged. Finish with the pilot comparison report and verification results.
