from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from factorial_analysis import (
    FactorialAnalysisError,
    analyze_results,
    parse_args,
    write_analysis_outputs,
)
from tests.test_independent_judges import member
from tests.test_tof_nof_evaluator import FakeJudgeClient, write_panel_transcripts
from tof_nof_evaluation.evaluation import evaluate_transcript_files


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
                    member(FakeJudgeClient([0] * 17), "gpt-6-sol"),
                ],
            )
            result = analyze_results(root / "results", bootstrap_replicates=20)
            self.assertEqual({"qwen3.8", "gpt-6-sol"}, set(result["judges"]))
            self.assertEqual(
                5, result["judges"]["qwen3.8"]["condition_summaries"][0]["mean_tof"]
            )
            self.assertIsNone(
                result["judges"]["qwen3.8"]["condition_summaries"][0]["std_tof"]
            )
            self.assertIsNone(result["judges"]["qwen3.8"]["factor_effects"][0]["std"])
            self.assertEqual(
                0,
                result["judges"]["gpt-6-sol"]["condition_summaries"][0]["mean_tof"],
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
            run("gpt-6-sol", ["item-2"], [1] * 17)
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
            with self.assertRaisesRegex(ValueError, "Missing.*gpt-6-sol"):
                analyze_results(root / "results", judges=["gpt-6-sol"])
            with self.assertRaisesRegex(ValueError, "Missing requested target"):
                analyze_results(root / "results", target_models=["missing"])

    def test_known_paired_effects(self):
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
            effects = result["judges"]["qwen3.8"]["factor_effects"]
            self.assertEqual(
                [-0.5, -1.5, 5.0],
                [e["estimate"] for e in effects if e["outcome"] == "tof"],
            )

    def test_observed_means_and_sample_stds_without_bootstrap(self):
        import csv
        import json
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for item in ("item-1", "item-2"):
                write_panel_transcripts(root / "sources", item_id=item)
            # Item 1 ToF: [5, 1, 2, 3]; item 2 ToF: [1, 5, 4, 1].
            first = [1] + [1] * 4 + [0] * 4 + [1, 0, 0, 0] + [1, 1, 0, 0]
            second = [1] + [0] * 4 + [1] * 4 + [1, 1, 1, 0] + [0, 1, 0, 1]
            evaluate_transcript_files(
                transcript_dir=root / "sources",
                output_dir=root / "results",
                judges=[
                    member(FakeJudgeClient(first + second)),
                    member(FakeJudgeClient(first + second), "gpt-6-sol"),
                ],
            )
            with patch(
                "numpy.random.default_rng",
                side_effect=AssertionError("Unexpected bootstrap"),
            ):
                result = analyze_results(root / "results", bootstrap_replicates=0)
            judge = result["judges"]["qwen3.8"]
            summary = judge["condition_summaries"][0]
            expected = {
                "n": 2,
                "mean_tof": 3.0,
                "std_tof": 2.828427,
                "mean_nof": 0.5,
                "std_nof": 0.707107,
                "initial_correct_rate": 1.0,
                "std_initial_correct": 0.0,
                "final_correct_rate": 0.5,
                "std_final_correct": 0.707107,
                "mean_downward_flips": 0.5,
                "std_downward_flips": 0.707107,
                "mean_upward_flips": 0.0,
                "std_upward_flips": 0.0,
            }
            self.assertEqual(expected, {key: summary[key] for key in expected})
            tof_effects = [e for e in judge["factor_effects"] if e["outcome"] == "tof"]
            self.assertEqual([-0.5, -0.5, -1.0], [e["estimate"] for e in tof_effects])
            self.assertEqual([0.0, 1.414214, 8.485281], [e["std"] for e in tof_effects])
            nof_effects = [e for e in judge["factor_effects"] if e["outcome"] == "nof"]
            self.assertEqual([1.25, 0.75, 1.5], [e["estimate"] for e in nof_effects])
            self.assertEqual(
                [1.06066, 0.353553, 3.535534], [e["std"] for e in nof_effects]
            )
            for effects in (
                judge["factor_effects"],
                result["comparison"]["intersection_factor_effects"],
            ):
                for effect in effects:
                    self.assertIsNone(effect["confidence_level"])
                    self.assertIsNone(effect["ci_lower"])
                    self.assertIsNone(effect["ci_upper"])
            for metadata in (judge["metadata"], result["comparison"]["metadata"]):
                self.assertEqual(1, metadata["standard_deviation_ddof"])
                self.assertEqual(0, metadata["bootstrap_replicates"])
                for field in ("bootstrap_seed", "confidence_level", "bootstrap_method"):
                    self.assertIsNone(metadata[field])
            matched = [
                {key: value for key, value in effect.items() if key != "judge"}
                for effect in result["comparison"]["intersection_factor_effects"]
                if effect["judge"] == "qwen3.8"
            ]
            self.assertEqual(judge["factor_effects"], matched)
            bootstrapped = analyze_results(root / "results", bootstrap_replicates=20)
            self.assertEqual(
                judge["condition_summaries"],
                bootstrapped["judges"]["qwen3.8"]["condition_summaries"],
            )
            for observed, sampled in zip(
                judge["factor_effects"],
                bootstrapped["judges"]["qwen3.8"]["factor_effects"],
            ):
                self.assertEqual(observed["estimate"], sampled["estimate"])
                self.assertEqual(observed["std"], sampled["std"])
                self.assertIsNotNone(sampled["ci_lower"])
                self.assertIsNotNone(sampled["ci_upper"])
            write_analysis_outputs(result, root / "analysis")
            directory = root / "analysis/judges/qwen3.8"
            self.assertEqual(
                judge, json.loads((directory / "factorial_analysis.json").read_text())
            )
            with (directory / "condition_summaries.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual("2.828427", row["std_tof"])
            with (directory / "factor_effects.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual("8.485281", rows[2]["std"])
            self.assertEqual("", rows[2]["ci_lower"])

    def test_no_bootstrap_cli_and_negative_replicates(self):
        import contextlib
        import io

        self.assertEqual(10_000, parse_args([]).bootstrap_replicates)
        self.assertEqual(0, parse_args(["--no-bootstrap"]).bootstrap_replicates)
        self.assertEqual(
            0, parse_args(["--bootstrap-replicates", "0"]).bootstrap_replicates
        )
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args(["--no-bootstrap", "--bootstrap-replicates", "20"])
        with self.assertRaisesRegex(FactorialAnalysisError, "nonnegative"):
            analyze_results(Path("unused"), bootstrap_replicates=-1)

    def test_removed_consensus_mode_is_rejected(self):
        import contextlib
        import io

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            parse_args(["--input-format", "legacy-consensus"])
        self.assertEqual(2, error.exception.code)
        with self.assertRaisesRegex(FactorialAnalysisError, "Unknown input format"):
            analyze_results(Path("unused"), input_format="legacy-consensus")
        self.assertEqual("independent", parse_args([]).input_format)

    def test_writes_independent_output_bundle(self):
        import csv
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_panel_transcripts(root / "sources")
            evaluate_transcript_files(
                transcript_dir=root / "sources",
                output_dir=root / "results",
                judges=[member(FakeJudgeClient([1] * 17))],
            )
            result = analyze_results(root / "results", bootstrap_replicates=20)
            paths = write_analysis_outputs(result, root / "analysis")
            self.assertEqual(7, len(paths))
            self.assertTrue(all(path.is_file() for path in paths))
            directory = root / "analysis/judges/qwen3.8"
            self.assertEqual(
                result["judges"]["qwen3.8"],
                json.loads((directory / "factorial_analysis.json").read_text()),
            )
            with (directory / "factor_effects.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(6, len(rows))
            self.assertEqual({0.0}, {float(row["estimate"]) for row in rows})
            comparison = root / "analysis/comparison/comparison.json"
            self.assertEqual(result["comparison"], json.loads(comparison.read_text()))

    def test_comparison_rejects_different_source_hashes_even_when_text_matches(self):
        import hashlib
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_panel_transcripts(root / "sources")
            evaluate_transcript_files(
                transcript_dir=root / "sources",
                output_dir=root / "results",
                judges=[
                    member(FakeJudgeClient([1] * 17)),
                    member(FakeJudgeClient([1] * 17), "gpt-6-sol"),
                ],
            )
            path = next(
                (root / "results/judges/gpt-6-sol").glob("checkpoints/*/*.json")
            )
            payload = json.loads(path.read_text())
            for source in payload["panel"]["sources"]:
                copied = root / "other-sources" / source["scenario"] / "item-1.json"
                copied.parent.mkdir(parents=True)
                copied.write_bytes(Path(source["path"]).read_bytes() + b"\n")
                source["path"] = str(copied)
                source["sha256"] = hashlib.sha256(copied.read_bytes()).hexdigest()
            path.write_text(json.dumps(payload))
            # Each namespace independently validates against its own source files.
            analyze_results(
                root / "results", judges=["gpt-6-sol"], bootstrap_replicates=20
            )
            with self.assertRaisesRegex(ValueError, "Mismatched source/prompt hashes"):
                analyze_results(root / "results", bootstrap_replicates=20)
