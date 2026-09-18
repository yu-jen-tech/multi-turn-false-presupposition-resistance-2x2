from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .atomic_io import atomic_text_writer, atomic_write_text
from .models import (
    EVALUATOR_PROMPT_VERSION,
    JUDGMENT_COLUMNS,
    RESULT_COLUMNS,
    JudgePanelMember,
    TranscriptScore,
)


CONSENSUS_OUTPUT_DIRNAME = "consensus"
OUTPUT_SCHEMA_VERSION = "tof-nof-majority-consensus-v1"


def _score_context_fields(score: TranscriptScore) -> dict[str, Any]:
    return {
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
        for judge in evaluation.judge_evaluations:
            rows.append(
                {
                    **_score_context_fields(score),
                    "turn": evaluation.turn,
                    "consensus_label": evaluation.label,
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
            "mean_upward_flips": statistics.mean(
                score.upward_flips for score in group
            ),
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


def _judge_summary(judges: Sequence[JudgePanelMember]) -> list[dict[str, Any]]:
    return [
        {
            "provider": judge.provider,
            "endpoint": judge.endpoint,
            **asdict(judge.config),
        }
        for judge in judges
    ]


def write_outputs(
    scores: Sequence[TranscriptScore],
    *,
    output_dir: Path,
    judges: Sequence[JudgePanelMember],
) -> list[Path]:
    consensus_dir = output_dir / CONSENSUS_OUTPUT_DIRNAME
    written_paths: list[Path] = []
    for (target_model, scenario), group in sorted(grouped_scores(scores).items()):
        output_path = consensus_dir / target_model / scenario / "tof_nof.csv"
        ordered = sorted(group, key=lambda current: current.item_id)
        _write_csv(
            output_path,
            fieldnames=RESULT_COLUMNS,
            rows=[score_to_csv_row(score) for score in ordered],
        )
        written_paths.append(output_path)

        judgments_path = output_path.with_name("tof_nof_judgments.csv")
        _write_csv(
            judgments_path,
            fieldnames=JUDGMENT_COLUMNS,
            rows=[
                row
                for score in ordered
                for row in score_to_judgment_rows(score)
            ],
        )
        written_paths.append(judgments_path)

    summary_path = consensus_dir / "summary.json"
    atomic_write_text(
        summary_path,
        json.dumps(
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
                "label_source": "unweighted_binary_majority",
                "judges": _judge_summary(judges),
                "results": summarize_scores(scores),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    written_paths.append(summary_path)
    return written_paths
