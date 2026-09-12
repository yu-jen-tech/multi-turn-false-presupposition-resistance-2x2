from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from transcript_generation.storage import model_path_component

from .atomic_io import atomic_text_writer, atomic_write_text
from .models import REASON_COLUMNS, RESULT_COLUMNS, TranscriptScore


def score_to_csv_row(score: TranscriptScore) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": score.item_id,
        "question": score.question,
        "presupposition": score.presupposition,
        "correction": score.correction,
        "target_model": score.target_model,
        "scenario": score.scenario,
        "tof": score.tof,
        "nof": score.nof,
    }
    for index, label in enumerate(score.labels, start=1):
        row[f"turn_{index}"] = label
    return row


def score_to_reason_rows(score: TranscriptScore) -> list[dict[str, Any]]:
    return [
        {
            "id": score.item_id,
            "question": score.question,
            "presupposition": score.presupposition,
            "correction": score.correction,
            "target_model": score.target_model,
            "scenario": score.scenario,
            "turn": evaluation.turn,
            "label": evaluation.label,
            "reason": evaluation.reason,
            "user_message": evaluation.user_message,
            "assistant_response": evaluation.assistant_response,
            "judge_output": evaluation.judge_output,
        }
        for evaluation in score.turn_evaluations
    ]


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


def write_outputs(
    scores: Sequence[TranscriptScore],
    *,
    output_dir: Path,
    judge_model: str,
) -> list[Path]:
    judge_output_dir = output_dir / model_path_component(judge_model)
    written_paths: list[Path] = []
    for (target_model, scenario), group in sorted(grouped_scores(scores).items()):
        output_path = (
            judge_output_dir / target_model / scenario / "tof_nof.csv"
        )
        rows = [
            score_to_csv_row(score)
            for score in sorted(group, key=lambda current: current.item_id)
        ]
        _write_csv(output_path, fieldnames=RESULT_COLUMNS, rows=rows)
        written_paths.append(output_path)

        reasons_path = output_path.with_name("tof_nof_reasons.csv")
        reason_rows = [
            row
            for score in sorted(group, key=lambda current: current.item_id)
            for row in score_to_reason_rows(score)
        ]
        _write_csv(
            reasons_path,
            fieldnames=REASON_COLUMNS,
            rows=reason_rows,
        )
        written_paths.append(reasons_path)

    summary_path = judge_output_dir / "summary.json"
    atomic_write_text(
        summary_path,
        json.dumps(summarize_scores(scores), ensure_ascii=False, indent=2) + "\n",
    )
    written_paths.append(summary_path)
    return written_paths
