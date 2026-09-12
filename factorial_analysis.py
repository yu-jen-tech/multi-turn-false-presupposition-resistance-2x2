#!/usr/bin/env python3
"""Analyze the completed 2x2 false-presupposition experiment.

The analysis keeps every judge and target model separate. It computes condition
summaries, paired within-item bootstrap intervals for the two main effects and
their interaction, and agreement between judges on identical stored turns.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tof_nof_evaluation import (
    REASON_COLUMNS,
    RESULT_COLUMNS,
    calculate_nof,
    calculate_tof,
)
from transcript_generation import SCENARIOS
from transcript_generation.storage import model_path_component


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis"
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95

SCENARIO_ORDER = tuple(scenario.name for scenario in SCENARIOS)
SCENARIO_FACTORS = {
    scenario.name: {
        "style": scenario.style,
        "simulator_history": scenario.simulator_history,
    }
    for scenario in SCENARIOS
}

EFFECT_DEFINITIONS = {
    "humanlike_minus_formal": (
        "Average humanlike score minus average formal score across both "
        "simulator-history levels."
    ),
    "with_minus_without_history": (
        "Average score with simulator history minus average score without "
        "simulator history across both styles."
    ),
    "style_by_history_interaction": (
        "(humanlike with history - humanlike without history) - "
        "(formal with history - formal without history)."
    ),
}


class FactorialAnalysisError(ValueError):
    """Raised when result artifacts cannot support a valid comparison."""


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _display_model_name(path_component: str) -> str:
    return path_component.replace("__", "/")


def discover_judge_models(results_dir: Path) -> tuple[str, ...]:
    if not results_dir.is_dir():
        raise FactorialAnalysisError(
            f"Results directory does not exist: {results_dir}"
        )
    judges = sorted(
        path.name
        for path in results_dir.iterdir()
        if path.is_dir() and any(path.glob("*/*/tof_nof.csv"))
    )
    if not judges:
        raise FactorialAnalysisError(
            f"No complete-score CSV files found under {results_dir}"
        )
    return tuple(judges)


def _required_columns(
    path: Path,
    fieldnames: Sequence[str] | None,
    required: Sequence[str],
) -> None:
    missing = sorted(set(required) - set(fieldnames or ()))
    if missing:
        raise FactorialAnalysisError(
            f"{path} is missing columns: {', '.join(missing)}"
        )


def _parse_binary(value: str, *, path: Path, line_number: int) -> int:
    try:
        label = int(value)
    except ValueError as error:
        raise FactorialAnalysisError(
            f"{path}:{line_number} has non-integer label {value!r}"
        ) from error
    if label not in (0, 1):
        raise FactorialAnalysisError(
            f"{path}:{line_number} has label {label}; expected 0 or 1"
        )
    return label


def _read_score_file(
    path: Path,
    *,
    expected_target_component: str,
    expected_scenario: str,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FactorialAnalysisError(f"Missing score file: {path}")

    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        _required_columns(path, reader.fieldnames, RESULT_COLUMNS)
        for line_number, row in enumerate(reader, start=2):
            item_id = row["id"].strip()
            if not item_id:
                raise FactorialAnalysisError(f"{path}:{line_number} has empty id")
            if item_id in rows:
                raise FactorialAnalysisError(
                    f"{path} contains duplicate id {item_id!r}"
                )
            if row["scenario"] != expected_scenario:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} says scenario {row['scenario']!r}; "
                    f"expected {expected_scenario!r}"
                )
            if model_path_component(row["target_model"]) != expected_target_component:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has target {row['target_model']!r}; "
                    f"expected path component {expected_target_component!r}"
                )

            labels = tuple(
                _parse_binary(
                    row[f"turn_{turn}"], path=path, line_number=line_number
                )
                for turn in range(1, 6)
            )
            try:
                tof = int(row["tof"])
                nof = int(row["nof"])
            except ValueError as error:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has non-integer ToF or NoF"
                ) from error
            if tof != calculate_tof(labels) or nof != calculate_nof(labels):
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has metrics inconsistent with labels"
                )

            rows[item_id] = {
                "id": item_id,
                "question": row["question"],
                "presupposition": row["presupposition"],
                "correction": row["correction"],
                "target_model": row["target_model"],
                "scenario": row["scenario"],
                "labels": labels,
                "tof": tof,
                "nof": nof,
            }
    if not rows:
        raise FactorialAnalysisError(f"{path} contains no score rows")
    return rows


def load_score_panels(
    results_dir: Path,
) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    panels: dict[
        str, dict[str, dict[str, dict[str, dict[str, Any]]]]
    ] = {}
    target_sets: list[set[str]] = []

    for judge_component in discover_judge_models(results_dir):
        judge_dir = results_dir / judge_component
        target_components = {
            path.name
            for path in judge_dir.iterdir()
            if path.is_dir() and any(path.glob("*/tof_nof.csv"))
        }
        if not target_components:
            raise FactorialAnalysisError(
                f"No target-model score directories found under {judge_dir}"
            )
        target_sets.append(target_components)
        judge_panels: dict[
            str, dict[str, dict[str, dict[str, Any]]]
        ] = {}

        for target_component in sorted(target_components):
            scenario_rows = {
                scenario: _read_score_file(
                    judge_dir / target_component / scenario / "tof_nof.csv",
                    expected_target_component=target_component,
                    expected_scenario=scenario,
                )
                for scenario in SCENARIO_ORDER
            }
            item_sets = [set(rows) for rows in scenario_rows.values()]
            if any(items != item_sets[0] for items in item_sets[1:]):
                raise FactorialAnalysisError(
                    f"{judge_dir / target_component} does not contain the same "
                    "item IDs in all four scenarios"
                )

            first_scenario = scenario_rows[SCENARIO_ORDER[0]]
            for item_id in sorted(first_scenario):
                reference = first_scenario[item_id]
                for scenario in SCENARIO_ORDER[1:]:
                    current = scenario_rows[scenario][item_id]
                    for field in ("question", "presupposition", "correction"):
                        if current[field] != reference[field]:
                            raise FactorialAnalysisError(
                                f"Item {item_id!r} differs on {field} across "
                                f"scenarios for {judge_component}/{target_component}"
                            )
            judge_panels[target_component] = scenario_rows
        panels[judge_component] = judge_panels

    if any(targets != target_sets[0] for targets in target_sets[1:]):
        raise FactorialAnalysisError(
            "Judge result directories do not contain the same target models"
        )
    return panels


def _condition_summary(
    *,
    judge_component: str,
    target_component: str,
    scenario: str,
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = [rows[item_id] for item_id in sorted(rows)]
    labels = np.asarray([row["labels"] for row in ordered], dtype=float)
    tofs = np.asarray([row["tof"] for row in ordered], dtype=float)
    nofs = np.asarray([row["nof"] for row in ordered], dtype=float)
    late_recovery = np.logical_and(
        np.any(labels[:, :4] == 0, axis=1), labels[:, 4] == 1
    )
    return {
        "judge_model": _display_model_name(judge_component),
        "target_model": _display_model_name(target_component),
        "scenario": scenario,
        **SCENARIO_FACTORS[scenario],
        "n": len(ordered),
        "mean_tof": _rounded(np.mean(tofs)),
        "mean_nof": _rounded(np.mean(nofs)),
        **{
            f"turn_{turn}_resistance_rate": _rounded(
                np.mean(labels[:, turn - 1])
            )
            for turn in range(1, 6)
        },
        "final_turn_resistance_rate": _rounded(np.mean(labels[:, 4])),
        "late_recovery_rate": _rounded(np.mean(late_recovery)),
    }


def _effect_vectors(
    scenario_values: Mapping[str, np.ndarray[Any, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    formal_without = scenario_values["formal_without_simulator_history"]
    formal_with = scenario_values["formal_with_simulator_history"]
    humanlike_without = scenario_values["humanlike_without_simulator_history"]
    humanlike_with = scenario_values["humanlike_with_simulator_history"]
    return {
        "humanlike_minus_formal": (
            humanlike_without
            + humanlike_with
            - formal_without
            - formal_with
        )
        / 2,
        "with_minus_without_history": (
            formal_with
            + humanlike_with
            - formal_without
            - humanlike_without
        )
        / 2,
        "style_by_history_interaction": (
            humanlike_with - humanlike_without
        )
        - (formal_with - formal_without),
    }


def _bootstrap_interval(
    values: np.ndarray[Any, Any],
    *,
    indices: np.ndarray[Any, Any],
) -> tuple[float, float]:
    bootstrap_means = np.mean(values[indices], axis=1)
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    lower, upper = np.quantile(bootstrap_means, (alpha, 1 - alpha))
    return _rounded(lower), _rounded(upper)


def _factor_effects(
    panels: Mapping[
        str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]
    ],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    index_cache: dict[int, np.ndarray[Any, Any]] = {}
    effects: list[dict[str, Any]] = []

    for judge_component, judge_panels in sorted(panels.items()):
        for target_component, scenarios in sorted(judge_panels.items()):
            item_ids = sorted(scenarios[SCENARIO_ORDER[0]])
            n = len(item_ids)
            if n not in index_cache:
                index_cache[n] = np.random.default_rng(bootstrap_seed).integers(
                    0, n, size=(bootstrap_replicates, n)
                )
            indices = index_cache[n]
            for outcome in ("tof", "nof"):
                scenario_values = {
                    scenario: np.asarray(
                        [scenarios[scenario][item_id][outcome] for item_id in item_ids],
                        dtype=float,
                    )
                    for scenario in SCENARIO_ORDER
                }
                for effect_name, values in _effect_vectors(
                    scenario_values
                ).items():
                    lower, upper = _bootstrap_interval(values, indices=indices)
                    effects.append(
                        {
                            "judge_model": _display_model_name(judge_component),
                            "target_model": _display_model_name(target_component),
                            "outcome": outcome,
                            "effect": effect_name,
                            "definition": EFFECT_DEFINITIONS[effect_name],
                            "n": n,
                            "estimate": _rounded(np.mean(values)),
                            "confidence_level": CONFIDENCE_LEVEL,
                            "ci_lower": lower,
                            "ci_upper": upper,
                        }
                    )
    return effects


def _read_reason_file(
    path: Path,
    *,
    expected_target_component: str,
    expected_scenario: str,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path.is_file():
        raise FactorialAnalysisError(f"Missing judge-reason file: {path}")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        _required_columns(path, reader.fieldnames, REASON_COLUMNS)
        for line_number, row in enumerate(reader, start=2):
            if row["scenario"] != expected_scenario:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has unexpected scenario"
                )
            if model_path_component(row["target_model"]) != expected_target_component:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has unexpected target model"
                )
            try:
                turn = int(row["turn"])
            except ValueError as error:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has non-integer turn"
                ) from error
            if turn not in range(1, 6):
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has turn {turn}; expected 1-5"
                )
            item_id = row["id"].strip()
            key = (expected_scenario, item_id, turn)
            if not item_id or key in records:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has empty or duplicate turn key"
                )
            records[key] = {
                "label": _parse_binary(
                    row["label"], path=path, line_number=line_number
                ),
                "evidence": tuple(
                    row[field]
                    for field in (
                        "question",
                        "presupposition",
                        "correction",
                        "user_message",
                        "assistant_response",
                    )
                ),
            }
    if not records:
        raise FactorialAnalysisError(f"{path} contains no judge decisions")
    return records


def _cohens_kappa(first: Sequence[int], second: Sequence[int]) -> float | None:
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    first_positive = sum(first) / len(first)
    second_positive = sum(second) / len(second)
    expected = (
        first_positive * second_positive
        + (1 - first_positive) * (1 - second_positive)
    )
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def _judge_agreement(
    panels: Mapping[
        str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]
    ],
    *,
    results_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    judge_components = sorted(panels)
    target_components = sorted(next(iter(panels.values())))
    pairwise: list[dict[str, Any]] = []
    unanimity: list[dict[str, Any]] = []
    turn_1_consistency: list[dict[str, Any]] = []

    for target_component in target_components:
        by_judge: dict[
            str, dict[tuple[str, str, int], dict[str, Any]]
        ] = {}
        for judge_component in judge_components:
            records: dict[tuple[str, str, int], dict[str, Any]] = {}
            for scenario in SCENARIO_ORDER:
                records.update(
                    _read_reason_file(
                        results_dir
                        / judge_component
                        / target_component
                        / scenario
                        / "tof_nof_reasons.csv",
                        expected_target_component=target_component,
                        expected_scenario=scenario,
                    )
                )
            by_judge[judge_component] = records

            item_ids = sorted(
                {
                    item_id
                    for scenario, item_id, turn in records
                    if turn == 1
                }
            )
            disagreeing_items = 0
            for item_id in item_ids:
                turn_1_records = [
                    records[(scenario, item_id, 1)]
                    for scenario in SCENARIO_ORDER
                ]
                if any(
                    current["evidence"] != turn_1_records[0]["evidence"]
                    for current in turn_1_records[1:]
                ):
                    raise FactorialAnalysisError(
                        f"Turn 1 evidence differs across scenarios for "
                        f"{judge_component}/{target_component}/{item_id}"
                    )
                if len({record["label"] for record in turn_1_records}) > 1:
                    disagreeing_items += 1
            turn_1_consistency.append(
                {
                    "judge_model": _display_model_name(judge_component),
                    "target_model": _display_model_name(target_component),
                    "n_items": len(item_ids),
                    "consistent_items": len(item_ids) - disagreeing_items,
                    "disagreeing_items": disagreeing_items,
                    "consistency_rate": _rounded(
                        (len(item_ids) - disagreeing_items) / len(item_ids)
                    ),
                }
            )

        key_sets = [set(records) for records in by_judge.values()]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise FactorialAnalysisError(
                f"Judges do not cover identical turns for {target_component}"
            )
        keys = sorted(key_sets[0])
        reference_judge = judge_components[0]
        for key in keys:
            reference = by_judge[reference_judge][key]["evidence"]
            for judge_component in judge_components[1:]:
                if by_judge[judge_component][key]["evidence"] != reference:
                    raise FactorialAnalysisError(
                        f"Judges evaluated different evidence for {target_component} "
                        f"at {key}"
                    )

        for first_judge, second_judge in itertools.combinations(
            judge_components, 2
        ):
            first = [by_judge[first_judge][key]["label"] for key in keys]
            second = [by_judge[second_judge][key]["label"] for key in keys]
            kappa = _cohens_kappa(first, second)
            pairwise.append(
                {
                    "target_model": _display_model_name(target_component),
                    "judge_a": _display_model_name(first_judge),
                    "judge_b": _display_model_name(second_judge),
                    "n_turn_decisions": len(keys),
                    "agreement_rate": _rounded(
                        sum(a == b for a, b in zip(first, second))
                        / len(keys)
                    ),
                    "cohens_kappa": None if kappa is None else _rounded(kappa),
                }
            )

        unanimous_count = sum(
            len(
                {
                    by_judge[judge_component][key]["label"]
                    for judge_component in judge_components
                }
            )
            == 1
            for key in keys
        )
        unanimity.append(
            {
                "target_model": _display_model_name(target_component),
                "judges": [
                    _display_model_name(component)
                    for component in judge_components
                ],
                "n_turn_decisions": len(keys),
                "unanimous_count": unanimous_count,
                "unanimous_rate": _rounded(unanimous_count / len(keys)),
            }
        )
    return pairwise, unanimity, turn_1_consistency


def analyze_results(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise FactorialAnalysisError(
            "bootstrap_replicates must be a positive integer"
        )
    panels = load_score_panels(results_dir)
    condition_summaries = [
        _condition_summary(
            judge_component=judge_component,
            target_component=target_component,
            scenario=scenario,
            rows=scenarios[scenario],
        )
        for judge_component, judge_panels in sorted(panels.items())
        for target_component, scenarios in sorted(judge_panels.items())
        for scenario in SCENARIO_ORDER
    ]
    pairwise, unanimity, turn_1_consistency = _judge_agreement(
        panels, results_dir=results_dir
    )
    try:
        displayed_results_dir = str(
            results_dir.resolve().relative_to(Path(__file__).resolve().parent)
        )
    except ValueError:
        displayed_results_dir = str(results_dir.resolve())
    return {
        "metadata": {
            "results_dir": displayed_results_dir,
            "scenario_order": list(SCENARIO_ORDER),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_method": (
                "Paired percentile bootstrap over item IDs. The same resampled "
                "item-index matrix is used for panels with the same item count."
            ),
            "tof_direction": "Higher means longer uninterrupted resistance.",
            "nof_direction": "Higher means more changes between resistance labels.",
            "late_recovery_definition": (
                "At least one label 0 in turns 1-4 and label 1 at turn 5."
            ),
            "effect_definitions": EFFECT_DEFINITIONS,
        },
        "condition_summaries": condition_summaries,
        "factor_effects": _factor_effects(
            panels,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "judge_agreement": pairwise,
        "judge_unanimity": unanimity,
        "turn_1_scenario_consistency": turn_1_consistency,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise FactorialAnalysisError(f"Cannot write empty table: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_outputs(
    analysis: Mapping[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "factorial_analysis.json"
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs = [json_path]
    for key, filename in (
        ("condition_summaries", "condition_summaries.csv"),
        ("factor_effects", "factor_effects.csv"),
        ("judge_agreement", "judge_agreement.csv"),
        ("judge_unanimity", "judge_unanimity.csv"),
        (
            "turn_1_scenario_consistency",
            "turn_1_scenario_consistency.csv",
        ),
    ):
        path = output_dir / filename
        _write_csv(path, analysis[key])
        outputs.append(path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute condition summaries, paired 2x2 bootstrap effects, and "
            "cross-judge agreement from completed ToF/NoF results."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    analysis = analyze_results(
        args.results_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    paths = write_analysis_outputs(analysis, args.output_dir)
    for path in paths:
        print(f"Wrote {path}")
    return paths


if __name__ == "__main__":
    main()
