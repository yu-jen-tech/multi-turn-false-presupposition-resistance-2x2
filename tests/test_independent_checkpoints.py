"""Existing independent checkpoints remain usable after retiring legacy import."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factorial_analysis import analyze_results
from tests.test_independent_judges import member
from tests.test_tof_nof_evaluator import FakeJudgeClient, write_panel_transcripts
from tof_nof_evaluation.cli import main
from tof_nof_evaluation.evaluation import evaluate_transcript_files


class IndependentCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = self.root / "sources"
        self.results = self.root / "results"
        write_panel_transcripts(self.sources)

    def evaluate(self, client=None, **kwargs):
        return evaluate_transcript_files(
            transcript_dir=self.sources,
            output_dir=self.results,
            judges=[member(client)],
            **kwargs,
        )

    def add_provenance(self):
        path = next(self.results.glob("judges/qwen3.8/checkpoints/*/*.json"))
        payload = json.loads(path.read_text())
        provenance = {
            "source_checkpoint": "archived/checkpoints/target__model/item-1.json",
            "sha256": "a" * 64,
        }
        payload["import_provenance"] = provenance
        path.write_text(json.dumps(payload))
        return path, provenance

    def test_completed_checkpoint_with_provenance_resumes_exports_and_analyzes_offline(
        self,
    ):
        scores = self.evaluate(FakeJudgeClient([1] * 17))
        expected = analyze_results(self.results, bootstrap_replicates=20)
        exports = {path: path.read_bytes() for path in self.results.rglob("*.csv")}
        self.assertTrue(exports)
        checkpoint, _ = self.add_provenance()
        saved = checkpoint.read_bytes()

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "tof_nof_evaluation.cli.create_judge_client",
                side_effect=AssertionError("offline operation created a client"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            resumed = self.evaluate(
                resume=True,
                client_factory=lambda _: self.fail("completed resume created a client"),
            )
            main(["export", "--output-dir", str(self.results), "--judge", "qwen3.8"])
            actual = analyze_results(self.results, bootstrap_replicates=20)

        self.assertEqual(scores, resumed)
        self.assertEqual(expected, actual)
        self.assertEqual(saved, checkpoint.read_bytes())
        self.assertEqual(exports, {path: path.read_bytes() for path in exports})

    def test_partial_checkpoint_retains_provenance_when_resumed(self):
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate(FakeJudgeClient([1] * 17, interrupt_after_calls=6))
        checkpoint, provenance = self.add_provenance()
        previous = json.loads(checkpoint.read_text())
        client = FakeJudgeClient([1] * 11)
        scores = self.evaluate(client, resume=True)
        current = json.loads(checkpoint.read_text())

        self.assertEqual(11, len(client.calls))
        self.assertEqual(4, len(scores))
        self.assertEqual(provenance, current["import_provenance"])
        self.assertEqual(previous["shared_turn_1"], current["shared_turn_1"])
        for scenario, prefix in previous["scenario_turns"].items():
            self.assertEqual(prefix, current["scenario_turns"][scenario][: len(prefix)])
        result = analyze_results(self.results, bootstrap_replicates=20)
        self.assertEqual(
            [5.0] * 4,
            [
                row["mean_tof"]
                for row in result["judges"]["qwen3.8"]["condition_summaries"]
            ],
        )

    def test_removed_import_command_fails_before_writing(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                main(
                    [
                        "import-legacy",
                        "--input-dir",
                        str(self.root / "archive"),
                        "--output-dir",
                        str(self.results),
                    ]
                )
        self.assertEqual(2, error.exception.code)
        self.assertIn("unrecognized arguments", stderr.getvalue())
        self.assertFalse(self.results.exists())
