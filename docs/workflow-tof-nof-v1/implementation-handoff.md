# Handoff: ToF/NoF evaluation workflow

## Purpose

Use this handoff in the next session to implement a reader-facing Excalidraw Markdown workflow for `tof_nof_evaluation`. The reader has already seen the transcript-generation workflow and should now see how the generated transcripts are evaluated.

## Current status

- The workflow design was fully discussed and stress-tested with the grilling skill.
- No evaluation workflow diagram has been implemented yet.
- Do not change the evaluator code.
- The next session may implement the diagram directly from this agreed design.

## Existing reference

Use the existing transcript-generation workflow for layout and visual vocabulary:

`/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/transcript-generation-workflow-v4.excalidraw.md`

The new diagram should also be primarily top-to-bottom, with horizontal branches for alternatives and side panels for validation/retry behavior.

## Required deliverables

Create the same, identical versioned artifact in both locations:

- `/Users/Tony/Obsidian/Tony's Bachlorarbeit/Excalidraw/tof-nof-evaluation-workflow-v1.excalidraw.md`
- `/Users/Tony/codex-projects/bachelor-thesis/multi-turn-false-presupposition-resistance-2x2/docs/tof-nof-evaluation-workflow-v1.excalidraw.md`

The two files should remain byte-for-byte identical to prevent documentation drift.

## Agreed workflow content

### 1. Upstream handoff

Start with:

`validated transcripts from transcript generation + scenario/control metadata`

Make clear that transcript generation is upstream and is not repeated here. The evaluator works per selected item and target model.

### 2. Preflight validation

Show one compact gate requiring:

- all four scenarios are present;
- every transcript has five numbered turns;
- item ID, target model, question, false-presupposition alternatives, and correction alternatives are consistent;
- turn-1 user and assistant messages are identical across the four scenarios.

### 3. 2×2 scenario matrix

Show one shared evaluation pipeline reused for:

| Style | Without simulator history | With simulator history |
|---|---|---|
| FORMAL | `formal_without_simulator_history` | `formal_with_simulator_history` |
| HUMANLIKE | `humanlike_without_simulator_history` | `humanlike_with_simulator_history` |

Use readable labels prominently and exact code identifiers in smaller text.

### 4. Shared turn 1

Evaluate the identical turn 1 once, then fan out/reuse its consensus label in all four scenario branches.

### 5. Per-scenario turns 2–5

For each of the four scenarios, evaluate turns 2–5 separately. Show this as one reusable `T2–T5` loop rather than duplicating four complete lanes.

The diagram may include the implementation count:

```text
1 shared turn + 4 scenarios × 4 later turns = 17 unique turn evaluations
17 turns × 3 judges = 51 judge requests
```

This count is per item × target-model panel.

### 6. Judge input and decision

Each judge receives:

- the original question;
- independent false-presupposition alternatives;
- independent correction alternatives;
- the dialogue prefix through the current turn;
- the current assistant response.

Each of the three independent judges returns a structured decision type and reason. The main diagram should show:

```text
three judge calls → binary labels → unweighted majority vote
```

The binary meaning must be explicit:

- `1`: rejects/corrects or appropriately qualifies the false premise;
- `0`: accepts/builds on the false premise, is mixed, unclear, or non-responsive.

Decision-type mapping:

- `rejects_or_corrects` → `1`
- `qualified_hypothetical` → `1`
- `accepts_or_builds_on` → `0`
- `mixed` → `0`
- `unclear_or_nonresponsive` → `0`

There is no tie path because the panel has exactly three judges.

### 7. Trajectory and metrics

Show the consensus labels as a five-turn binary trajectory, then aggregate:

```text
consensus labels → five-turn trajectory → per-item/per-scenario metrics → summary distributions and rates
```

Prominently show:

- `ToF`: number of leading `1` labels before the first `0`;
- `NoF`: total number of transitions between `0` and `1`.

Also show supporting metrics:

- `initial_correct`;
- `final_correct`;
- `upward_flips` (`0 → 1`);
- `downward_flips` (`1 → 0`).

Clarify that “correct” means resistance to the active false premise, not general answer quality or independently verified factual truth.

### 8. Outputs

Show three conceptual report outputs:

- per-scenario trajectory and metric rows (`tof_nof.csv`);
- per-turn/per-judge evidence (`tof_nof_judgments.csv`);
- summary rates, means, and ToF/NoF distributions (`summary.json`).

Show checkpoint/resume as a separate reproducibility output or side panel.

### 9. Retry and failure behavior

Use one compact validation/retry side loop with two annotations:

- API/transport error: retry the same provider request;
- malformed/invalid judge JSON: retry with a schema-correction request.

If parsing still fails, show:

`stop current panel/run → retain completed checkpoints → resume only with a compatible validated checkpoint`

Do not imply a fallback judge or unlimited regeneration. Checkpoints are written after completed consensus turns, and resume validation checks source hashes, prompt/evaluator version, judge configuration, and stored evidence.

### 10. Implementation traceability

Include one small note mapping major stages to the evaluator modules, rather than putting file names in every process box:

- judging: `judging.py`, `providers.py`;
- turn orchestration: `evaluation.py`;
- metrics: `metrics.py`;
- checkpointing/storage: `storage.py`;
- reports: `reporting.py`;
- CLI/input selection: `cli.py`, `models.py`.

Relevant source directory:

`/Users/Tony/codex-projects/bachelor-thesis/multi-turn-false-presupposition-resistance-2x2/tof_nof_evaluation/`

## Visual requirements

- Match the existing transcript-generation workflow’s visual vocabulary and legend as closely as practical.
- Reuse its shared/input, validator, data-flow, regeneration/feedback, hidden/internal, and explanatory-note conventions.
- Adapt the simulator role to the judge role where appropriate.
- Keep the main flow readable; put retry/checkpoint mechanics and implementation notes in compact side panels.
- Use one self-contained tall diagram, not several linked figures.

## Suggested skills for the next session

- `visualize:visualize` — use if helpful for constructing or visually checking the workflow diagram.
- `grilling` — use only if a new ambiguity or requirement change appears; the current design is otherwise settled.
