#!/usr/bin/env python3
"""Analyze majority-consensus results from the paired 2x2 experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tof_nof_evaluation import (
    JUDGMENT_COLUMNS,
    RESULT_COLUMNS,
    calculate_nof,
    calculate_tof,
    calculate_trajectory_metrics,
    decision_label,
)
from transcript_generation import SCENARIOS
from transcript_generation.storage import model_path_component


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results-v3-consensus"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis-v3-consensus"
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95
CONSENSUS_DIRNAME = "consensus"

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


def _consensus_dir(results_dir: Path) -> Path:
    return results_dir / CONSENSUS_DIRNAME


def discover_target_models(results_dir: Path) -> tuple[str, ...]:
    consensus_dir = _consensus_dir(results_dir)
    if not consensus_dir.is_dir():
        raise FactorialAnalysisError(
            f"Consensus results directory does not exist: {consensus_dir}"
        )
    targets = sorted(
        path.name
        for path in consensus_dir.iterdir()
        if path.is_dir() and any(path.glob("*/tof_nof.csv"))
    )
    if not targets:
        raise FactorialAnalysisError(
            f"No consensus score CSV files found under {consensus_dir}"
        )
    return tuple(targets)


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


def _parse_integer(
    value: str,
    *,
    path: Path,
    line_number: int,
    description: str,
) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise FactorialAnalysisError(
            f"{path}:{line_number} has non-integer {description} {value!r}"
        ) from error


def _parse_binary(value: str, *, path: Path, line_number: int) -> int:
    label = _parse_integer(
        value,
        path=path,
        line_number=line_number,
        description="label",
    )
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
                    f"{path}:{line_number} has unexpected target model"
                )
            labels = tuple(
                _parse_binary(
                    row[f"turn_{turn}"],
                    path=path,
                    line_number=line_number,
                )
                for turn in range(1, 6)
            )
            stored = {
                field: _parse_integer(
                    row[field],
                    path=path,
                    line_number=line_number,
                    description=field,
                )
                for field in (
                    "tof",
                    "nof",
                    "initial_correct",
                    "final_correct",
                    "downward_flips",
                    "upward_flips",
                )
            }
            metrics = calculate_trajectory_metrics(labels)
            expected = {
                "tof": calculate_tof(labels),
                "nof": calculate_nof(labels),
                "initial_correct": metrics.initial_correct,
                "final_correct": metrics.final_correct,
                "downward_flips": metrics.downward_flips,
                "upward_flips": metrics.upward_flips,
            }
            if stored != expected:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has metrics inconsistent with labels"
                )
            rows[item_id] = {
                "id": item_id,
                "question": row["question"],
                "presuppositions": row["presuppositions"],
                "corrections": row["corrections"],
                "target_model": row["target_model"],
                "scenario": row["scenario"],
                "labels": labels,
                **stored,
            }
    if not rows:
        raise FactorialAnalysisError(f"{path} contains no score rows")
    return rows


ScorePanels = dict[str, dict[str, dict[str, dict[str, Any]]]]


def load_score_panels(results_dir: Path) -> ScorePanels:
    panels: ScorePanels = {}
    consensus_dir = _consensus_dir(results_dir)
    for target_component in discover_target_models(results_dir):
        scenario_rows = {
            scenario: _read_score_file(
                consensus_dir / target_component / scenario / "tof_nof.csv",
                expected_target_component=target_component,
                expected_scenario=scenario,
            )
            for scenario in SCENARIO_ORDER
        }
        item_sets = [set(rows) for rows in scenario_rows.values()]
        if any(items != item_sets[0] for items in item_sets[1:]):
            raise FactorialAnalysisError(
                f"{consensus_dir / target_component} does not contain the same "
                "item IDs in all four scenarios"
            )
        first_scenario = scenario_rows[SCENARIO_ORDER[0]]
        for item_id in sorted(first_scenario):
            reference = first_scenario[item_id]
            for scenario in SCENARIO_ORDER[1:]:
                current = scenario_rows[scenario][item_id]
                for field in ("question", "presuppositions", "corrections"):
                    if current[field] != reference[field]:
                        raise FactorialAnalysisError(
                            f"Item {item_id!r} differs on {field} across scenarios "
                            f"for {target_component}"
                        )
        panels[target_component] = scenario_rows
    return panels


def _condition_summary(
    *,
    target_component: str,
    scenario: str,
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = [rows[item_id] for item_id in sorted(rows)]
    return {
        "target_model": _display_model_name(target_component),
        "scenario": scenario,
        **SCENARIO_FACTORS[scenario],
        "n": len(ordered),
        "mean_tof": _rounded(np.mean([row["tof"] for row in ordered])),
        "mean_nof": _rounded(np.mean([row["nof"] for row in ordered])),
        "initial_correct_rate": _rounded(
            np.mean([row["initial_correct"] for row in ordered])
        ),
        "final_correct_rate": _rounded(
            np.mean([row["final_correct"] for row in ordered])
        ),
        "mean_downward_flips": _rounded(
            np.mean([row["downward_flips"] for row in ordered])
        ),
        "mean_upward_flips": _rounded(
            np.mean([row["upward_flips"] for row in ordered])
        ),
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
    panels: ScorePanels,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    index_cache: dict[int, np.ndarray[Any, Any]] = {}
    effects: list[dict[str, Any]] = []
    for target_component, scenarios in sorted(panels.items()):
        item_ids = sorted(scenarios[SCENARIO_ORDER[0]])
        n = len(item_ids)
        if n not in index_cache:
            index_cache[n] = np.random.default_rng(bootstrap_seed).integers(
                0,
                n,
                size=(bootstrap_replicates, n),
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
            for effect_name, values in _effect_vectors(scenario_values).items():
                lower, upper = _bootstrap_interval(values, indices=indices)
                effects.append(
                    {
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


JudgeIdentity = tuple[str, str]


def _read_judgment_file(
    path: Path,
    *,
    expected_target_component: str,
    expected_scenario: str,
) -> dict[tuple[str, str, int, JudgeIdentity], dict[str, Any]]:
    if not path.is_file():
        raise FactorialAnalysisError(f"Missing judge-detail file: {path}")
    records: dict[tuple[str, str, int, JudgeIdentity], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        _required_columns(path, reader.fieldnames, JUDGMENT_COLUMNS)
        for line_number, row in enumerate(reader, start=2):
            if row["scenario"] != expected_scenario:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has unexpected scenario"
                )
            if model_path_component(row["target_model"]) != expected_target_component:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has unexpected target model"
                )
            item_id = row["id"].strip()
            turn = _parse_integer(
                row["turn"],
                path=path,
                line_number=line_number,
                description="turn",
            )
            if not item_id or turn not in range(1, 6):
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has invalid item ID or turn"
                )
            model = row["judge_model"].strip()
            provider = row["judge_provider"].strip()
            if not model or not provider:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has empty judge identity"
                )
            identity = (provider, model)
            key = (expected_scenario, item_id, turn, identity)
            if key in records:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has duplicate judge turn"
                )
            label = _parse_binary(
                row["judge_label"],
                path=path,
                line_number=line_number,
            )
            try:
                expected_label = decision_label(row["decision_type"])
            except ValueError as error:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has unknown decision type"
                ) from error
            if label != expected_label:
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has decision type/label mismatch"
                )
            if not row["reason"].strip():
                raise FactorialAnalysisError(
                    f"{path}:{line_number} has an empty judge reason"
                )
            records[key] = {
                "identity": identity,
                "model": model,
                "provider": provider,
                "label": label,
                "decision_type": row["decision_type"],
                "reason": row["reason"],
                "consensus_label": _parse_binary(
                    row["consensus_label"],
                    path=path,
                    line_number=line_number,
                ),
                "evidence": tuple(
                    row[field]
                    for field in (
                        "question",
                        "presuppositions",
                        "corrections",
                        "user_message",
                        "assistant_response",
                    )
                ),
                "judge_output": row["judge_output"],
                "judge_metadata": row["judge_metadata"],
                "judge_raw_response": row["judge_raw_response"],
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


def _agreement_analysis(
    panels: ScorePanels,
    *,
    results_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairwise: list[dict[str, Any]] = []
    unanimity: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    consensus_dir = _consensus_dir(results_dir)

    for target_component, scenarios in sorted(panels.items()):
        raw_records: dict[
            tuple[str, str, int, JudgeIdentity], dict[str, Any]
        ] = {}
        for scenario in SCENARIO_ORDER:
            raw_records.update(
                _read_judgment_file(
                    consensus_dir
                    / target_component
                    / scenario
                    / "tof_nof_judgments.csv",
                    expected_target_component=target_component,
                    expected_scenario=scenario,
                )
            )

        grouped: dict[
            tuple[str, str, int], dict[JudgeIdentity, dict[str, Any]]
        ] = {}
        for (scenario, item_id, turn, identity), record in raw_records.items():
            grouped.setdefault((scenario, item_id, turn), {})[identity] = record

        judge_order: list[JudgeIdentity] | None = None
        for key, by_judge in sorted(grouped.items()):
            if len(by_judge) != 3:
                raise FactorialAnalysisError(
                    f"Expected exactly three judges for {target_component} at {key}"
                )
            identities = list(by_judge)
            if judge_order is None:
                judge_order = identities
            elif set(identities) != set(judge_order):
                raise FactorialAnalysisError(
                    f"Judge panel changes within target {target_component}"
                )
            records = [by_judge[identity] for identity in judge_order]
            if any(record["evidence"] != records[0]["evidence"] for record in records[1:]):
                raise FactorialAnalysisError(
                    f"Judges evaluated different evidence for {target_component} at {key}"
                )
            consensus_labels = {record["consensus_label"] for record in records}
            if len(consensus_labels) != 1:
                raise FactorialAnalysisError(
                    f"Consensus label differs across judge rows at {key}"
                )
            expected_consensus = (
                1 if sum(record["label"] for record in records) >= 2 else 0
            )
            if expected_consensus != records[0]["consensus_label"]:
                raise FactorialAnalysisError(
                    f"Consensus label is not the judge majority at {key}"
                )
            score_label = scenarios[key[0]][key[1]]["labels"][key[2] - 1]
            if expected_consensus != score_label:
                raise FactorialAnalysisError(
                    f"Consensus score and judge details disagree at {key}"
                )
        assert judge_order is not None

        unique: dict[
            tuple[str, str, int], dict[JudgeIdentity, dict[str, Any]]
        ] = {}
        item_ids = sorted(scenarios[SCENARIO_ORDER[0]])
        for item_id in item_ids:
            shared = [grouped[(scenario, item_id, 1)] for scenario in SCENARIO_ORDER]
            reference = shared[0]
            for current in shared[1:]:
                for identity in judge_order:
                    comparable_fields = (
                        "label",
                        "decision_type",
                        "reason",
                        "consensus_label",
                        "evidence",
                        "judge_output",
                        "judge_metadata",
                        "judge_raw_response",
                    )
                    if any(
                        current[identity][field] != reference[identity][field]
                        for field in comparable_fields
                    ):
                        raise FactorialAnalysisError(
                            f"Shared turn 1 decision differs across scenarios for "
                            f"{target_component}/{item_id}"
                        )
            unique[("shared_turn_1", item_id, 1)] = reference
            for scenario in SCENARIO_ORDER:
                for turn in range(2, 6):
                    unique[(scenario, item_id, turn)] = grouped[
                        (scenario, item_id, turn)
                    ]

        keys = sorted(unique)
        for first_identity, second_identity in itertools.combinations(judge_order, 2):
            first = [unique[key][first_identity]["label"] for key in keys]
            second = [unique[key][second_identity]["label"] for key in keys]
            agreement_count = sum(a == b for a, b in zip(first, second))
            kappa = _cohens_kappa(first, second)
            pairwise.append(
                {
                    "target_model": _display_model_name(target_component),
                    "judge_a_provider": first_identity[0],
                    "judge_a": first_identity[1],
                    "judge_b_provider": second_identity[0],
                    "judge_b": second_identity[1],
                    "n_turn_decisions": len(keys),
                    "agreement_count": agreement_count,
                    "agreement_rate": _rounded(agreement_count / len(keys)),
                    "cohens_kappa": None if kappa is None else _rounded(kappa),
                }
            )

        unanimous_count = 0
        for scope, item_id, turn in keys:
            by_judge = unique[(scope, item_id, turn)]
            labels = [by_judge[identity]["label"] for identity in judge_order]
            if len(set(labels)) == 1:
                unanimous_count += 1
                continue
            reference = by_judge[judge_order[0]]
            disagreements.append(
                {
                    "target_model": _display_model_name(target_component),
                    "id": item_id,
                    "scenario": scope,
                    "turn": turn,
                    "consensus_label": reference["consensus_label"],
                    "user_message": reference["evidence"][3],
                    "assistant_response": reference["evidence"][4],
                    "judge_decisions": json.dumps(
                        [
                            {
                                "provider": identity[0],
                                "model": identity[1],
                                "decision_type": by_judge[identity]["decision_type"],
                                "label": by_judge[identity]["label"],
                                "reason": by_judge[identity]["reason"],
                            }
                            for identity in judge_order
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
        two_to_one_count = len(keys) - unanimous_count
        unanimity.append(
            {
                "target_model": _display_model_name(target_component),
                "judges": json.dumps(
                    [
                        {"provider": provider, "model": model}
                        for provider, model in judge_order
                    ],
                    ensure_ascii=False,
                ),
                "n_turn_decisions": len(keys),
                "unanimous_count": unanimous_count,
                "unanimous_rate": _rounded(unanimous_count / len(keys)),
                "two_to_one_count": two_to_one_count,
                "two_to_one_rate": _rounded(two_to_one_count / len(keys)),
            }
        )
    return pairwise, unanimity, disagreements


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
            target_component=target_component,
            scenario=scenario,
            rows=scenarios[scenario],
        )
        for target_component, scenarios in sorted(panels.items())
        for scenario in SCENARIO_ORDER
    ]
    pairwise, unanimity, disagreements = _agreement_analysis(
        panels,
        results_dir=results_dir,
    )
    try:
        displayed_results_dir = str(results_dir.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        displayed_results_dir = str(results_dir.resolve())
    return {
        "metadata": {
            "results_dir": displayed_results_dir,
            "label_source": "Majority judgment of three LLM judges under a fixed rubric.",
            "scenario_order": list(SCENARIO_ORDER),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_method": (
                "Paired percentile bootstrap over item IDs. The same resampled "
                "item-index matrix is used for panels with the same item count."
            ),
            "tof_interpretation": (
                "Number of leading correct labels before the first failure."
            ),
            "nof_interpretation": (
                "Number of adjacent stance changes; lower is not automatically better."
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
        "disagreement_examples": disagreements,
    }


OUTPUT_TABLES = (
    ("condition_summaries", "condition_summaries.csv"),
    ("factor_effects", "factor_effects.csv"),
    ("judge_agreement", "judge_agreement.csv"),
    ("judge_unanimity", "judge_unanimity.csv"),
    ("disagreement_examples", "disagreement_examples.csv"),
)
DISAGREEMENT_COLUMNS = (
    "target_model",
    "id",
    "scenario",
    "turn",
    "consensus_label",
    "user_message",
    "assistant_response",
    "judge_decisions",
)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fieldnames: Sequence[str] | None = None,
) -> None:
    if rows:
        fieldnames = list(rows[0])
    elif empty_fieldnames is not None:
        fieldnames = list(empty_fieldnames)
    else:
        raise FactorialAnalysisError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_outputs(
    analysis: Mapping[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "factorial_analysis.json"
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs = [json_path]
    for key, filename in OUTPUT_TABLES:
        path = output_dir / filename
        _write_csv(
            path,
            analysis[key],
            empty_fieldnames=(
                DISAGREEMENT_COLUMNS if key == "disagreement_examples" else None
            ),
        )
        outputs.append(path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute consensus condition summaries, paired 2x2 bootstrap effects, "
            "and three-judge agreement."
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
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
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
