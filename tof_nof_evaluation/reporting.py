from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .atomic_io import atomic_text_writer, atomic_write_text
from .models import (
    JUDGMENT_COLUMNS,
    RESULT_COLUMNS,
    JudgePanelMember,
    TranscriptScore,
)


from .storage import (
    OUTPUT_SCHEMA_VERSION,
    EvaluationCheckpointContext,
    context_from_namespace,
    judge_lock,
    manifest_payload,
    stored_panels,
)
from .models import judge_key, resolve_judges


def _score_context_fields(score: TranscriptScore) -> dict[str, Any]:
    return {
        "judge_model": score.judge.model,
        "judge_provider": score.judge.provider,
        "id": score.item_id,
        "question": score.question,
        "presuppositions": json.dumps(
            score.presuppositions,
            ensure_ascii=False,
        ),
        "corrections": json.dumps(
            score.corrections,
            ensure_ascii=False,
        ),
        "target_model": score.target_model,
        "scenario": score.scenario,
    }


def score_to_csv_row(score: TranscriptScore) -> dict[str, Any]:
    row = {
        **_score_context_fields(score),
        "tof": score.tof,
        "nof": score.nof,
        "initial_correct": score.initial_correct,
        "final_correct": score.final_correct,
        "downward_flips": score.downward_flips,
        "upward_flips": score.upward_flips,
    }
    for index, label in enumerate(score.labels, start=1):
        row[f"turn_{index}"] = label
    return row


def score_to_judgment_rows(score: TranscriptScore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in score.turn_evaluations:
        for judge in (evaluation.judgment,):
            rows.append(
                {
                    **_score_context_fields(score),
                    "turn": evaluation.turn,
                    "judge_model": judge.judge_model,
                    "judge_provider": judge.judge_provider,
                    "decision_type": judge.decision_type,
                    "judge_label": judge.label,
                    "reason": judge.reason,
                    "user_message": evaluation.user_message,
                    "assistant_response": evaluation.assistant_response,
                    "judge_output": judge.judge_output,
                    "judge_metadata": json.dumps(
                        judge.judge_metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "judge_raw_response": json.dumps(
                        judge.judge_raw_response,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
    return rows


def grouped_scores(
    scores: Sequence[TranscriptScore],
) -> dict[tuple[str, str], list[TranscriptScore]]:
    groups: dict[tuple[str, str], list[TranscriptScore]] = defaultdict(list)
    for score in scores:
        groups[(score.target_model_component, score.scenario)].append(score)
    return groups


def _distribution(values: Sequence[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def summarize_scores(
    scores: Sequence[TranscriptScore],
) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for (target_model, scenario), group in sorted(grouped_scores(scores).items()):
        summary.setdefault(target_model, {})[scenario] = {
            "count": len(group),
            "mean_tof": statistics.mean(score.tof for score in group),
            "mean_nof": statistics.mean(score.nof for score in group),
            "initial_correct_rate": statistics.mean(
                score.initial_correct for score in group
            ),
            "final_correct_rate": statistics.mean(
                score.final_correct for score in group
            ),
            "mean_downward_flips": statistics.mean(
                score.downward_flips for score in group
            ),
            "mean_upward_flips": statistics.mean(score.upward_flips for score in group),
            "tof_distribution": _distribution([score.tof for score in group]),
            "nof_distribution": _distribution([score.nof for score in group]),
        }
    return summary


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    with atomic_text_writer(path, newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def export_context(context: EvaluationCheckpointContext) -> list[Path]:
    """Rebuild cumulative derived outputs. Caller holds the judge writer lock."""
    from .evaluation import scores_from_progress

    stored = stored_panels(context)
    scores = [
        score
        for panel, turns, _ in stored
        for score in scores_from_progress(panel, turns, judge=context.judge)
    ]
    written: list[Path] = []
    for (target, scenario), group in sorted(grouped_scores(scores).items()):
        ordered = sorted(group, key=lambda score: score.item_id)
        path = context.namespace / "scores" / target / scenario / "tof_nof.csv"
        _write_csv(
            path, fieldnames=RESULT_COLUMNS, rows=[score_to_csv_row(s) for s in ordered]
        )
        judgments = path.with_name("tof_nof_judgments.csv")
        _write_csv(
            judgments,
            fieldnames=JUDGMENT_COLUMNS,
            rows=[r for s in ordered for r in score_to_judgment_rows(s)],
        )
        written.extend((path, judgments))
    # Remove obsolete derived tables after checkpoint validation (e.g. after manual checkpoint recovery).
    for path in (context.namespace / "scores").glob("*/*/*.csv"):
        if path not in written:
            path.unlink()
    coverage = {
        "stored_panels": len(stored),
        "complete_panels": sum(
            all(len(v) == 5 for v in turns.values()) for _, turns, _ in stored
        ),
        "complete_trajectories": len(scores),
        "unique_judgments": sum(
            (1 if any(turns.values()) else 0)
            + sum(max(0, len(v) - 1) for v in turns.values())
            for _, turns, _ in stored
        ),
        "partial_panels": [
            {
                "target_model": panel.target_model,
                "id": panel.item_id,
                "scenario_turn_counts": {s: len(v) for s, v in turns.items()},
            }
            for panel, turns, _ in stored
            if any(len(v) != 5 for v in turns.values())
        ],
    }
    summary = {
        **manifest_payload(context),
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "coverage": coverage,
        "results": summarize_scores(scores),
    }
    path = context.namespace / "summary.json"
    atomic_write_text(path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return written + [path]


def discover_namespaces(
    output_dir: Path, aliases: Sequence[str] | None = None
) -> list[Path]:
    if aliases is not None:
        keys = [judge_key(d) for d in resolve_judges(aliases)]
    else:
        keys = (
            sorted(p.name for p in (output_dir / "judges").iterdir() if p.is_dir())
            if (output_dir / "judges").exists()
            else []
        )
        resolve_judges(keys)
    paths = [output_dir / "judges" / key for key in keys]
    for path in paths:
        if not (path / "manifest.json").exists():
            raise ValueError(f"Missing judge namespace/manifest: {path}")
    return paths


def export_results(
    output_dir: Path, *, aliases: Sequence[str] | None = None
) -> list[Path]:
    from contextlib import ExitStack

    namespaces = discover_namespaces(output_dir, aliases)
    with ExitStack() as stack:
        for namespace in namespaces:
            stack.enter_context(judge_lock(namespace))
        contexts = [context_from_namespace(ns) for ns in namespaces]
        for context in contexts:
            stored_panels(context)
        return [path for context in contexts for path in export_context(context)]


def write_outputs(
    scores: Sequence[TranscriptScore],
    *,
    output_dir: Path,
    judges: Sequence[JudgePanelMember],
) -> list[Path]:
    """Compatibility entry point: checkpoints, not invocation-local scores, are authoritative."""
    return export_results(output_dir, aliases=[judge_key(j.definition) for j in judges])
