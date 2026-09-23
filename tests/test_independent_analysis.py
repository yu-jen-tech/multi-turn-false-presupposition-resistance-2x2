from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from tests.test_independent_judges import member
from tests.test_tof_nof_evaluator import FakeJudgeClient, write_panel_transcripts
from tof_nof_evaluation.evaluation import evaluate_transcript_files
from factorial_analysis import analyze_results


class IndependentAnalysisTests(unittest.TestCase):
    def test_separate_metrics_paired_effects_and_shared_turn_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_panel_transcripts(root / "sources")
            evaluate_transcript_files(
                transcript_dir=root / "sources",
                output_dir=root / "results",
                judges=[
                    member(FakeJudgeClient([1] * 17)),
                    member(FakeJudgeClient([0] * 17), "gpt-5.4-mini"),
                ],
            )
            result = analyze_results(root / "results", bootstrap_replicates=20)
            self.assertEqual({"qwen3.8", "gpt-5.4-mini"}, set(result["judges"]))
            self.assertEqual(
                5, result["judges"]["qwen3.8"]["condition_summaries"][0]["mean_tof"]
            )
            self.assertEqual(
                0,
                result["judges"]["gpt-5.4-mini"]["condition_summaries"][0]["mean_tof"],
            )
            comparison = result["comparison"]["targets"][0]
            self.assertEqual(17, comparison["unique_turns"])
            self.assertEqual(0, comparison["binary_agreement"])
            self.assertEqual(17, len(result["comparison"]["disagreements"]))
            single = analyze_results(
                root / "results", judges=["qwen3.8"], bootstrap_replicates=20
            )
            self.assertEqual("unavailable", single["comparison"]["status"])

    def test_unequal_coverage_uses_intersection_and_empty_intersection_is_unavailable(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for item in ("item-1", "item-2", "item-3"):
                write_panel_transcripts(root / "sources", item_id=item)

            def run(alias, ids, labels):
                evaluate_transcript_files(
                    transcript_dir=root / "sources",
                    output_dir=root / "results",
                    judges=[member(FakeJudgeClient(labels), alias)],
                    item_ids=set(ids),
                )

            run("qwen3.8", ["item-1"], [1] * 17)
            run("gpt-5.4-mini", ["item-2"], [1] * 17)
            result = analyze_results(root / "results", bootstrap_replicates=20)
            self.assertEqual("unavailable", result["comparison"]["status"])
            self.assertIsNone(result["comparison"]["targets"][0]["binary_agreement"])
            run("qwen3.8", ["item-2", "item-3"], [1] * 34)
            result = analyze_results(root / "results", bootstrap_replicates=20)
            summary = result["comparison"]["targets"][0]
            self.assertEqual(1, summary["coverage"]["intersection_count"])
            self.assertEqual(
                ["item-1", "item-3"], summary["coverage"]["qwen3.8"]["excluded_ids"]
            )
            self.assertEqual(17, summary["unique_turns"])
            self.assertIsNone(summary["cohens_kappa"])
            self.assertEqual(
                {1},
                {e["n"] for e in result["comparison"]["intersection_factor_effects"]},
            )
            self.assertEqual(
                {3}, {e["n"] for e in result["judges"]["qwen3.8"]["factor_effects"]}
            )

    def test_partial_panel_and_explicit_missing_selections_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_panel_transcripts(root / "sources")
            with self.assertRaises(KeyboardInterrupt):
                evaluate_transcript_files(
                    transcript_dir=root / "sources",
                    output_dir=root / "results",
                    judges=[member(FakeJudgeClient([1] * 17, interrupt_after_calls=6))],
                )
            with self.assertRaisesRegex(ValueError, "Incomplete coverage.*item-1"):
                analyze_results(root / "results")
            with self.assertRaisesRegex(ValueError, "Missing.*gpt-5.4-mini"):
                analyze_results(root / "results", judges=["gpt-5.4-mini"])
            with self.assertRaisesRegex(ValueError, "Missing requested target"):
                analyze_results(root / "results", target_models=["missing"])

    def test_known_paired_effects_and_explicit_legacy_mode_match(self):
        from tests.test_factorial_analysis import write_consensus_panel, TARGET_MODEL

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_panel_transcripts(root / "sources")
            # formal without = 5, formal with = 1, humanlike without = 2, humanlike with = 3.
            calls = [1] + [1] * 4 + [0] * 4 + [1, 0, 0, 0] + [1, 1, 0, 0]
            evaluate_transcript_files(
                transcript_dir=root / "sources",
                output_dir=root / "results",
                judges=[member(FakeJudgeClient(calls))],
            )
            result = analyze_results(root / "results", bootstrap_replicates=20)
            from tof_nof_evaluation.models import SCENARIO_ORDER

            write_consensus_panel(
                root / "legacy",
                scenario_tofs=dict(zip(SCENARIO_ORDER, [[5], [1], [2], [3]])),
            )
            legacy = analyze_results(
                root / "legacy",
                input_format="legacy-consensus",
                target_models=[TARGET_MODEL],
                bootstrap_replicates=20,
            )
            effects = result["judges"]["qwen3.8"]["factor_effects"]
            self.assertEqual(legacy["factor_effects"], effects)
            self.assertEqual(
                [-0.5, -1.5, 5.0],
                [e["estimate"] for e in effects if e["outcome"] == "tof"],
            )
