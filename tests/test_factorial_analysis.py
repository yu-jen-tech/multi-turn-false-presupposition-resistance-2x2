from __future__ import annotations

import csv
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
    REASON_COLUMNS,
    RESULT_COLUMNS,
    calculate_nof,
)
from transcript_generation.storage import model_path_component


TARGET_MODEL = "target/model"
JUDGE_A = "judge/a"
JUDGE_B = "judge/b"


def labels_for_tof(tof: int) -> tuple[int, ...]:
    return tuple([1] * tof + [0] * (5 - tof))


def write_panel(
    results_dir: Path,
    *,
    judge_model: str,
    scenario_tofs: dict[str, list[int]],
) -> None:
    for scenario in SCENARIO_ORDER:
        directory = (
            results_dir
            / model_path_component(judge_model)
            / model_path_component(TARGET_MODEL)
            / scenario
        )
        directory.mkdir(parents=True, exist_ok=True)
        score_rows: list[dict[str, object]] = []
        reason_rows: list[dict[str, object]] = []
        for item_index, tof in enumerate(scenario_tofs[scenario], start=1):
            item_id = f"item-{item_index}"
            labels = labels_for_tof(tof)
            common = {
                "id": item_id,
                "question": f"Question {item_index}?",
                "presuppositions": f'["False premise {item_index}."]',
                "corrections": f'["Correction {item_index}."]',
                "target_model": TARGET_MODEL,
                "scenario": scenario,
            }
            score_rows.append(
                {
                    **common,
                    **{
                        f"turn_{turn}": labels[turn - 1]
                        for turn in range(1, 6)
                    },
                    "tof": tof,
                    "nof": calculate_nof(labels),
                }
            )
            for turn, label in enumerate(labels, start=1):
                reason_rows.append(
                    {
                        **common,
                        "turn": turn,
                        "label": label,
                        "reason": f"Reason for label {label}.",
                        "user_message": (
                            f"Question {item_index}?"
                            if turn == 1
                            else f"User evidence {scenario} item {item_index} "
                            f"turn {turn}."
                        ),
                        "assistant_response": (
                            f"Shared first answer {item_index}."
                            if turn == 1
                            else f"Assistant evidence {scenario} item {item_index} "
                            f"turn {turn}."
                        ),
                        "judge_output": (
                            f'{{"label": {label}, "reason": "Reason."}}'
                        ),
                    }
                )

        with (directory / "tof_nof.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=RESULT_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(score_rows)
        with (directory / "tof_nof_reasons.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=REASON_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(reason_rows)


def base_scenario_tofs() -> dict[str, list[int]]:
    return {
        "formal_without_simulator_history": [1, 2],
        "formal_with_simulator_history": [2, 3],
        "humanlike_without_simulator_history": [3, 4],
        "humanlike_with_simulator_history": [5, 5],
    }


class FactorialAnalysisTests(unittest.TestCase):
    def test_cli_defaults_use_list_native_v3_locations(self) -> None:
        args = parse_args([])

        self.assertEqual(DEFAULT_RESULTS_DIR, args.results_dir)
        self.assertEqual(DEFAULT_OUTPUT_DIR, args.output_dir)
        self.assertTrue(str(args.results_dir).endswith("results-v3"))
        self.assertTrue(str(args.output_dir).endswith("analysis-v3"))

    def test_computes_paired_factor_effects_and_condition_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            for judge in (JUDGE_A, JUDGE_B):
                write_panel(
                    results_dir,
                    judge_model=judge,
                    scenario_tofs=base_scenario_tofs(),
                )

            analysis = analyze_results(
                results_dir,
                bootstrap_replicates=200,
                bootstrap_seed=7,
            )

            effects = {
                (row["judge_model"], row["outcome"], row["effect"]): row
                for row in analysis["factor_effects"]
            }
            self.assertAlmostEqual(
                2.25,
                effects[(JUDGE_A, "tof", "humanlike_minus_formal")][
                    "estimate"
                ],
            )
            self.assertAlmostEqual(
                1.25,
                effects[(JUDGE_A, "tof", "with_minus_without_history")][
                    "estimate"
                ],
            )
            self.assertAlmostEqual(
                0.5,
                effects[(JUDGE_A, "tof", "style_by_history_interaction")][
                    "estimate"
                ],
            )
            summaries = analysis["condition_summaries"]
            self.assertEqual(8, len(summaries))
            final_summary = next(
                row
                for row in summaries
                if row["judge_model"] == JUDGE_A
                and row["scenario"] == "humanlike_with_simulator_history"
            )
            self.assertEqual(2, final_summary["n"])
            self.assertEqual(5.0, final_summary["mean_tof"])
            self.assertEqual(1.0, final_summary["final_turn_resistance_rate"])

    def test_compares_judges_only_on_identical_turn_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            write_panel(
                results_dir,
                judge_model=JUDGE_A,
                scenario_tofs=base_scenario_tofs(),
            )
            changed = base_scenario_tofs()
            changed["formal_without_simulator_history"] = [0, 2]
            write_panel(
                results_dir,
                judge_model=JUDGE_B,
                scenario_tofs=changed,
            )

            analysis = analyze_results(
                results_dir,
                bootstrap_replicates=20,
                bootstrap_seed=3,
            )

            agreement = analysis["judge_agreement"]
            self.assertEqual(1, len(agreement))
            self.assertEqual(40, agreement[0]["n_turn_decisions"])
            self.assertAlmostEqual(39 / 40, agreement[0]["agreement_rate"])
            self.assertIsNotNone(agreement[0]["cohens_kappa"])
            self.assertAlmostEqual(
                39 / 40,
                analysis["judge_unanimity"][0]["unanimous_rate"],
            )
            consistency = {
                row["judge_model"]: row
                for row in analysis["turn_1_scenario_consistency"]
            }
            self.assertEqual(0, consistency[JUDGE_A]["disagreeing_items"])
            self.assertEqual(1, consistency[JUDGE_B]["disagreeing_items"])

    def test_rejects_an_incomplete_four_scenario_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            write_panel(
                results_dir,
                judge_model=JUDGE_A,
                scenario_tofs=base_scenario_tofs(),
            )
            missing = (
                results_dir
                / model_path_component(JUDGE_A)
                / model_path_component(TARGET_MODEL)
                / SCENARIO_ORDER[-1]
                / "tof_nof.csv"
            )
            missing.unlink()

            with self.assertRaisesRegex(FactorialAnalysisError, "Missing score"):
                analyze_results(
                    results_dir,
                    bootstrap_replicates=20,
                    bootstrap_seed=3,
                )

    def test_writes_machine_readable_output_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            for judge in (JUDGE_A, JUDGE_B):
                write_panel(
                    results_dir,
                    judge_model=judge,
                    scenario_tofs=base_scenario_tofs(),
                )
            analysis = analyze_results(
                results_dir,
                bootstrap_replicates=20,
                bootstrap_seed=3,
            )

            paths = write_analysis_outputs(analysis, root / "analysis")

            self.assertEqual(6, len(paths))
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertIn('"bootstrap_replicates": 20', paths[0].read_text())


if __name__ == "__main__":
    unittest.main()
