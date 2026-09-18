from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from factorial_analysis import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    SCENARIO_ORDER,
    FactorialAnalysisError,
    analyze_results,
    parse_args,
    write_analysis_outputs,
)
from tof_nof_evaluation import (
    DEFAULT_JUDGE_PANEL,
    JUDGMENT_COLUMNS,
    RESULT_COLUMNS,
    calculate_nof,
    calculate_tof,
    calculate_trajectory_metrics,
)
from transcript_generation.storage import model_path_component


TARGET_MODEL = "target/model"
JUDGES = tuple((judge.provider, judge.model) for judge in DEFAULT_JUDGE_PANEL)


def labels_for_tof(tof: int) -> tuple[int, ...]:
    return tuple([1] * tof + [0] * (5 - tof))


def decision_type(label: int) -> str:
    return "rejects_or_corrects" if label == 1 else "accepts_or_builds_on"


def write_consensus_panel(
    results_dir: Path,
    *,
    scenario_tofs: dict[str, list[int]],
    judge_overrides: dict[tuple[str, str, int, int], int] | None = None,
    target_model: str = TARGET_MODEL,
) -> None:
    judge_overrides = judge_overrides or {}
    for scenario in SCENARIO_ORDER:
        directory = (
            results_dir
            / "consensus"
            / model_path_component(target_model)
            / scenario
        )
        directory.mkdir(parents=True, exist_ok=True)
        score_rows: list[dict[str, object]] = []
        judgment_rows: list[dict[str, object]] = []
        for item_index, tof in enumerate(scenario_tofs[scenario], start=1):
            item_id = f"item-{item_index}"
            labels = labels_for_tof(tof)
            metrics = calculate_trajectory_metrics(labels)
            common = {
                "id": item_id,
                "question": f"Question {item_index}?",
                "presuppositions": f'["False premise {item_index}."]',
                "corrections": f'["Correction {item_index}."]',
                "target_model": target_model,
                "scenario": scenario,
            }
            score_rows.append(
                {
                    **common,
                    **{
                        f"turn_{turn}": labels[turn - 1]
                        for turn in range(1, 6)
                    },
                    "tof": calculate_tof(labels),
                    "nof": calculate_nof(labels),
                    "initial_correct": metrics.initial_correct,
                    "final_correct": metrics.final_correct,
                    "downward_flips": metrics.downward_flips,
                    "upward_flips": metrics.upward_flips,
                }
            )
            for turn, consensus_label in enumerate(labels, start=1):
                user_message = (
                    f"Question {item_index}?"
                    if turn == 1
                    else f"User evidence {scenario} item {item_index} turn {turn}."
                )
                assistant_response = (
                    f"Shared first answer {item_index}."
                    if turn == 1
                    else f"Assistant evidence {scenario} item {item_index} turn {turn}."
                )
                for judge_index, (provider, model) in enumerate(JUDGES):
                    label = judge_overrides.get(
                        (scenario, item_id, turn, judge_index),
                        consensus_label,
                    )
                    kind = decision_type(label)
                    judgment_rows.append(
                        {
                            **common,
                            "turn": turn,
                            "consensus_label": consensus_label,
                            "judge_model": model,
                            "judge_provider": provider,
                            "decision_type": kind,
                            "judge_label": label,
                            "reason": f"Judge {judge_index} chose {kind}.",
                            "user_message": user_message,
                            "assistant_response": assistant_response,
                            "judge_output": json.dumps(
                                {
                                    "decision_type": kind,
                                    "reason": "Reason.",
                                }
                            ),
                            "judge_metadata": json.dumps({"model": model}),
                            "judge_raw_response": json.dumps(
                                {"id": f"raw-{judge_index}"}
                            ),
                        }
                    )

        with (directory / "tof_nof.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=RESULT_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(score_rows)
        with (directory / "tof_nof_judgments.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=JUDGMENT_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(judgment_rows)


def base_scenario_tofs() -> dict[str, list[int]]:
    return {
        "formal_without_simulator_history": [1, 2],
        "formal_with_simulator_history": [2, 3],
        "humanlike_without_simulator_history": [3, 4],
        "humanlike_with_simulator_history": [5, 5],
    }


class FactorialAnalysisTests(unittest.TestCase):
    def test_cli_defaults_use_consensus_locations(self) -> None:
        args = parse_args([])
        self.assertEqual(DEFAULT_RESULTS_DIR, args.results_dir)
        self.assertEqual(DEFAULT_OUTPUT_DIR, args.output_dir)
        self.assertTrue(str(args.results_dir).endswith("results-v3-consensus"))
        self.assertTrue(str(args.output_dir).endswith("analysis-v3-consensus"))

    def test_computes_consensus_paired_effects_and_descriptive_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            write_consensus_panel(
                results_dir,
                scenario_tofs=base_scenario_tofs(),
            )
            analysis = analyze_results(
                results_dir,
                bootstrap_replicates=200,
                bootstrap_seed=7,
                expected_target_models=(TARGET_MODEL,),
            )

        effects = {
            (row["outcome"], row["effect"]): row
            for row in analysis["factor_effects"]
        }
        self.assertAlmostEqual(
            2.25,
            effects[("tof", "humanlike_minus_formal")]["estimate"],
        )
        self.assertAlmostEqual(
            1.25,
            effects[("tof", "with_minus_without_history")]["estimate"],
        )
        self.assertAlmostEqual(
            0.5,
            effects[("tof", "style_by_history_interaction")]["estimate"],
        )
        self.assertEqual(4, len(analysis["condition_summaries"]))
        final_summary = next(
            row
            for row in analysis["condition_summaries"]
            if row["scenario"] == "humanlike_with_simulator_history"
        )
        self.assertEqual(2, final_summary["n"])
        self.assertEqual(5.0, final_summary["mean_tof"])
        self.assertEqual(1.0, final_summary["initial_correct_rate"])
        self.assertEqual(1.0, final_summary["final_correct_rate"])
        self.assertEqual(0.0, final_summary["mean_downward_flips"])
        self.assertEqual(0.0, final_summary["mean_upward_flips"])
        self.assertNotIn("judge_model", final_summary)

    def test_reports_three_pairwise_agreements_and_deduplicates_shared_turn_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            scenario = "formal_with_simulator_history"
            write_consensus_panel(
                results_dir,
                scenario_tofs=base_scenario_tofs(),
                judge_overrides={(scenario, "item-1", 2, 2): 0},
            )
            analysis = analyze_results(
                results_dir,
                bootstrap_replicates=20,
                bootstrap_seed=3,
                expected_target_models=(TARGET_MODEL,),
            )

        agreement = analysis["judge_agreement"]
        self.assertEqual(3, len(agreement))
        self.assertTrue(all(row["n_turn_decisions"] == 34 for row in agreement))
        by_pair = {(row["judge_a"], row["judge_b"]): row for row in agreement}
        first_model = DEFAULT_JUDGE_PANEL[0].model
        second_model = DEFAULT_JUDGE_PANEL[1].model
        third_model = DEFAULT_JUDGE_PANEL[2].model
        self.assertEqual(
            34,
            by_pair[(first_model, second_model)]["agreement_count"],
        )
        self.assertEqual(
            33,
            by_pair[(first_model, third_model)]["agreement_count"],
        )
        self.assertAlmostEqual(
            33 / 34,
            by_pair[(first_model, third_model)]["agreement_rate"],
            places=6,
        )
        self.assertIsNotNone(
            by_pair[(first_model, third_model)]["cohens_kappa"]
        )

        unanimity = analysis["judge_unanimity"][0]
        self.assertEqual(34, unanimity["n_turn_decisions"])
        self.assertEqual(33, unanimity["unanimous_count"])
        self.assertEqual(1, unanimity["two_to_one_count"])
        self.assertAlmostEqual(33 / 34, unanimity["unanimous_rate"], places=6)
        self.assertAlmostEqual(1 / 34, unanimity["two_to_one_rate"], places=6)

        disagreements = analysis["disagreement_examples"]
        self.assertEqual(1, len(disagreements))
        self.assertEqual("item-1", disagreements[0]["id"])
        self.assertEqual(2, disagreements[0]["turn"])
        decisions = json.loads(disagreements[0]["judge_decisions"])
        self.assertEqual(3, len(decisions))
        self.assertEqual([1, 1, 0], [decision["label"] for decision in decisions])

    def test_rejects_an_incomplete_four_scenario_item_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            incomplete = base_scenario_tofs()
            incomplete[SCENARIO_ORDER[-1]] = [5]
            write_consensus_panel(results_dir, scenario_tofs=incomplete)

            with self.assertRaisesRegex(
                FactorialAnalysisError,
                "same item IDs in all four scenarios",
            ):
                analyze_results(
                    results_dir,
                    bootstrap_replicates=20,
                    bootstrap_seed=3,
                    expected_target_models=(TARGET_MODEL,),
                )

    def test_rejects_a_missing_target_or_cross_target_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            write_consensus_panel(
                results_dir,
                scenario_tofs=base_scenario_tofs(),
            )
            with self.assertRaisesRegex(
                FactorialAnalysisError,
                "missing expected target",
            ):
                analyze_results(
                    results_dir,
                    bootstrap_replicates=20,
                    expected_target_models=(TARGET_MODEL, "missing/model"),
                )

            one_item = {
                scenario: values[:1]
                for scenario, values in base_scenario_tofs().items()
            }
            write_consensus_panel(
                results_dir,
                scenario_tofs=one_item,
                target_model="second/model",
            )
            with self.assertRaisesRegex(
                FactorialAnalysisError,
                "same item IDs across target models",
            ):
                analyze_results(
                    results_dir,
                    bootstrap_replicates=20,
                    expected_target_models=(TARGET_MODEL, "second/model"),
                )

    def test_rejects_stored_descriptive_metrics_inconsistent_with_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            write_consensus_panel(
                results_dir,
                scenario_tofs=base_scenario_tofs(),
            )
            path = next(results_dir.glob("consensus/*/*/tof_nof.csv"))
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            rows[0]["upward_flips"] = "99"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=RESULT_COLUMNS,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(
                FactorialAnalysisError,
                "metrics inconsistent with labels",
            ):
                analyze_results(
                    results_dir,
                    bootstrap_replicates=20,
                    bootstrap_seed=3,
                    expected_target_models=(TARGET_MODEL,),
                )

    def test_writes_machine_readable_output_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            write_consensus_panel(
                results_dir,
                scenario_tofs=base_scenario_tofs(),
            )
            analysis = analyze_results(
                results_dir,
                bootstrap_replicates=20,
                bootstrap_seed=3,
                expected_target_models=(TARGET_MODEL,),
            )
            paths = write_analysis_outputs(analysis, root / "analysis")

            self.assertEqual(6, len(paths))
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(any(path.name == "disagreement_examples.csv" for path in paths))
            self.assertIn('"bootstrap_replicates": 20', paths[0].read_text())


if __name__ == "__main__":
    unittest.main()
