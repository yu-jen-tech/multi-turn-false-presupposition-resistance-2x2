#!/usr/bin/env python3
"""Separate paired 2x2 analyses of independent judges and evidence-matched comparison."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

from legacy_factorial_analysis import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    EFFECT_DEFINITIONS,
    FactorialAnalysisError,
    _condition_summary,
    _factor_effects,
    _cohens_kappa,
)
from tof_nof_evaluation.atomic_io import atomic_write_text
from tof_nof_evaluation.evaluation import scores_from_progress
from tof_nof_evaluation.models import SCENARIO_ORDER, DECISION_LABELS
from tof_nof_evaluation.reporting import (
    discover_namespaces,
    score_to_csv_row,
    _write_csv,
)
from tof_nof_evaluation.storage import (
    context_from_namespace,
    judge_lock,
    manifest_payload,
    stored_panels,
)
from transcript_generation.storage import model_path_component

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results-v3-independent"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis-v3-independent"


def _load_judge(namespace, requested_targets):
    context = context_from_namespace(namespace)
    stored = stored_panels(context)
    available = {panel.target_model for panel, _, _ in stored}
    if requested_targets is None:
        targets = available
    else:
        targets = set()
        for requested in requested_targets:
            matches = {
                t for t in available if requested in (t, model_path_component(t))
            }
            if not matches:
                raise FactorialAnalysisError(
                    f"Missing requested target {requested!r} for {namespace.name}"
                )
            targets.update(matches)
    if not targets:
        raise FactorialAnalysisError(f"No stored target panels for {namespace.name}")
    panels: dict[str, Any] = {}
    evidence: dict[tuple[str, str], Any] = {}
    incomplete = []
    for panel, turns, payload in stored:
        if panel.target_model not in targets:
            continue
        if any(len(v) != 5 for v in turns.values()):
            incomplete.append(
                {
                    "target_model": panel.target_model,
                    "id": panel.item_id,
                    "scenario_turn_counts": {s: len(v) for s, v in turns.items()},
                }
            )
            continue
        target = model_path_component(panel.target_model)
        scenarios = panels.setdefault(target, {s: {} for s in SCENARIO_ORDER})
        for score in scores_from_progress(panel, turns, judge=context.judge):
            scenarios[score.scenario][score.item_id] = score_to_csv_row(score)
        evidence[(target, panel.item_id)] = (turns, payload)
    if incomplete:
        raise FactorialAnalysisError(
            "Incomplete coverage; all four five-turn scenarios required: "
            + json.dumps(incomplete)
        )
    return context, panels, evidence


def _comparison(loaded, *, bootstrap_replicates, bootstrap_seed):
    if len(loaded) != 2:
        return {
            "status": "unavailable",
            "reason": "Two judges are required",
            "targets": [],
            "disagreements": [],
            "trajectory_differences": [],
            "intersection_factor_effects": [],
        }
    first, second = sorted(loaded)
    _, a_panels, a_evidence = loaded[first]
    _, b_panels, b_evidence = loaded[second]
    result: dict[str, Any] = {
        "status": "unavailable",
        "judges": [first, second],
        "targets": [],
        "disagreements": [],
        "trajectory_differences": [],
        "intersection_factor_effects": [],
        "metadata": {
            "label_source": "single_judge",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "judges": {
                key: manifest_payload(value[0]) for key, value in loaded.items()
            },
        },
    }
    for target in sorted(set(a_panels) | set(b_panels)):
        a_ids = set(a_panels.get(target, {}).get(SCENARIO_ORDER[0], {}))
        b_ids = set(b_panels.get(target, {}).get(SCENARIO_ORDER[0], {}))
        common = sorted(a_ids & b_ids)
        coverage = {
            first: {"full_count": len(a_ids), "excluded_ids": sorted(a_ids - b_ids)},
            second: {"full_count": len(b_ids), "excluded_ids": sorted(b_ids - a_ids)},
            "intersection_count": len(common),
            "intersection_ids": common,
        }
        summary = {
            "target_model": target,
            "coverage": coverage,
            "unique_turns": 0,
            "binary_agreement": None,
            "cohens_kappa": None,
            "decision_agreement": None,
            "decision_type_cross_tab": {
                a: {b: 0 for b in DECISION_LABELS} for a in DECISION_LABELS
            },
            "status": "unavailable",
        }
        left, right = [], []
        matching_decisions = 0
        for item in common:
            a_turns, a_payload = a_evidence[(target, item)]
            b_turns, b_payload = b_evidence[(target, item)]
            if (
                a_payload["panel"] != b_payload["panel"]
                or a_payload["evaluation"]["prompt_sha256"]
                != b_payload["evaluation"]["prompt_sha256"]
            ):
                raise FactorialAnalysisError(
                    f"Mismatched source/prompt hashes: {target}/{item}"
                )
            for scenario in SCENARIO_ORDER:
                a_score, b_score = (
                    a_panels[target][scenario][item],
                    b_panels[target][scenario][item],
                )
                result["trajectory_differences"].append(
                    {
                        "target_model": target,
                        "id": item,
                        "scenario": scenario,
                        "first_judge": first,
                        "second_judge": second,
                        "first_labels": "".join(
                            str(t.label) for t in a_turns[scenario]
                        ),
                        "second_labels": "".join(
                            str(t.label) for t in b_turns[scenario]
                        ),
                        "first_tof": a_score["tof"],
                        "second_tof": b_score["tof"],
                        "first_nof": a_score["nof"],
                        "second_nof": b_score["nof"],
                        "tof_difference_second_minus_first": b_score["tof"]
                        - a_score["tof"],
                        "nof_difference_second_minus_first": b_score["nof"]
                        - a_score["nof"],
                    }
                )
                for a, b in zip(a_turns[scenario], b_turns[scenario]):
                    if (a.turn, a.user_message, a.assistant_response) != (
                        b.turn,
                        b.user_message,
                        b.assistant_response,
                    ):
                        raise FactorialAnalysisError("Mismatched turn evidence")
                    if a.turn == 1 and scenario != SCENARIO_ORDER[0]:
                        continue
                    aj, bj = a.judgment, b.judgment
                    left.append(a.label)
                    right.append(b.label)
                    matching_decisions += aj.decision_type == bj.decision_type
                    summary["decision_type_cross_tab"][aj.decision_type][
                        bj.decision_type
                    ] += 1
                    if (a.label, aj.decision_type) != (b.label, bj.decision_type):
                        result["disagreements"].append(
                            {
                                "target_model": target,
                                "id": item,
                                "scenario": "shared_turn_1"
                                if a.turn == 1
                                else scenario,
                                "turn": a.turn,
                                "first_judge": first,
                                "second_judge": second,
                                "first_label": a.label,
                                "second_label": b.label,
                                "first_decision": aj.decision_type,
                                "second_decision": bj.decision_type,
                                "first_reason": aj.reason,
                                "second_reason": bj.reason,
                                "user_message": a.user_message,
                                "assistant_response": a.assistant_response,
                                "prompt_sha256": a_payload["evaluation"][
                                    "prompt_sha256"
                                ],
                            }
                        )
        if common:
            result["status"] = "available"
            summary.update(
                status="available",
                unique_turns=len(left),
                binary_agreement=sum(a == b for a, b in zip(left, right)) / len(left),
                cohens_kappa=_cohens_kappa(left, right),
                decision_agreement=matching_decisions / len(left),
            )
            for key, panels in ((first, a_panels), (second, b_panels)):
                matched = {
                    target: {
                        s: {i: panels[target][s][i] for i in common}
                        for s in SCENARIO_ORDER
                    }
                }
                result["intersection_factor_effects"].extend(
                    {"judge": key, **effect}
                    for effect in _factor_effects(
                        matched,
                        bootstrap_replicates=bootstrap_replicates,
                        bootstrap_seed=bootstrap_seed,
                    )
                )
        result["targets"].append(summary)
    if result["status"] == "unavailable":
        result["reason"] = "No common complete item panels"
    return result


def analyze_results(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    judges: Sequence[str] | None = None,
    target_models: Sequence[str] | None = None,
    input_format: str = "independent",
) -> dict[str, Any]:
    if input_format == "legacy-consensus":
        from legacy_factorial_analysis import analyze_results as legacy_analyze
        from transcript_generation.models import DEFAULT_TARGET_MODELS

        if judges is not None:
            raise FactorialAnalysisError(
                "--judge cannot filter legacy consensus measurements"
            )
        return legacy_analyze(
            results_dir,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            expected_target_models=target_models or DEFAULT_TARGET_MODELS,
        )
    if input_format != "independent":
        raise FactorialAnalysisError("Unknown input format")
    if bootstrap_replicates <= 0:
        raise FactorialAnalysisError("bootstrap_replicates must be positive")
    namespaces = discover_namespaces(results_dir, judges)
    with ExitStack() as stack:
        for namespace in namespaces:
            stack.enter_context(judge_lock(namespace, reader=True))
        loaded = {ns.name: _load_judge(ns, target_models) for ns in namespaces}
        analyses = {}
        for key, (context, panels, evidence) in loaded.items():
            metadata = {
                **manifest_payload(context),
                "results_dir": str(results_dir),
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
                "confidence_level": CONFIDENCE_LEVEL,
                "bootstrap_method": "Paired percentile bootstrap over item IDs; same item sampled across four scenarios.",
                "effect_definitions": EFFECT_DEFINITIONS,
                "source_coverage": [
                    {
                        "target_model": target,
                        "id": item,
                        "sources": payload["panel"]["sources"],
                        "prompt_sha256": payload["evaluation"]["prompt_sha256"],
                    }
                    for (target, item), (_, payload) in sorted(evidence.items())
                ],
            }
            analyses[key] = {
                "metadata": metadata,
                "condition_summaries": [
                    _condition_summary(
                        target_component=t, scenario=s, rows=panels[t][s]
                    )
                    for t in sorted(panels)
                    for s in SCENARIO_ORDER
                ],
                "factor_effects": _factor_effects(
                    panels,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
        return {
            "judges": analyses,
            "comparison": _comparison(
                loaded,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            ),
        }


def write_analysis_outputs(
    analysis: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[Path]:
    if "judges" not in analysis:
        from legacy_factorial_analysis import write_analysis_outputs as legacy_write

        return legacy_write(analysis, output_dir)
    outputs = []
    for key, result in analysis["judges"].items():
        directory = output_dir / "judges" / key
        path = directory / "factorial_analysis.json"
        atomic_write_text(path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        outputs.append(path)
        for table in ("condition_summaries", "factor_effects"):
            path = directory / f"{table}.csv"
            rows = result[table]
            _write_csv(path, fieldnames=list(rows[0]), rows=rows)
            outputs.append(path)
    directory = output_dir / "comparison"
    comparison = analysis["comparison"]
    path = directory / "comparison.json"
    atomic_write_text(path, json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    outputs.append(path)
    for table, empty_columns in (
        (
            "disagreements",
            ["target_model", "id", "scenario", "turn", "first_reason", "second_reason"],
        ),
        (
            "trajectory_differences",
            [
                "target_model",
                "id",
                "scenario",
                "tof_difference_second_minus_first",
                "nof_difference_second_minus_first",
            ],
        ),
        (
            "intersection_factor_effects",
            ["judge", "target_model", "outcome", "effect", "n", "estimate"],
        ),
    ):
        rows = comparison[table]
        path = directory / f"{table}.csv"
        _write_csv(path, fieldnames=list(rows[0]) if rows else empty_columns, rows=rows)
        outputs.append(path)
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--judge", action="append")
    parser.add_argument("--target-model", action="append")
    parser.add_argument(
        "--input-format",
        choices=("independent", "legacy-consensus"),
        default="independent",
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    analysis = analyze_results(
        args.results_dir,
        judges=args.judge,
        target_models=args.target_model,
        input_format=args.input_format,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    paths = write_analysis_outputs(analysis, args.output_dir)
    for path in paths:
        print(f"Wrote {path}")
    return paths


if __name__ == "__main__":
    main()
