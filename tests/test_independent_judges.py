from __future__ import annotations

import unittest

from tof_nof_evaluation.models import resolve_judges


class SelectionTests(unittest.TestCase):
    def test_selection_is_independent_and_rejects_duplicates(self):
        self.assertEqual(
            ["openai/qwen3.8", "gpt-5.4-mini"], [j.model for j in resolve_judges(None)]
        )
        self.assertEqual(
            ["gpt-5.4-mini"], [j.model for j in resolve_judges(["gpt-5.4-mini"])]
        )
        self.assertEqual(
            list(reversed(resolve_judges(None))),
            list(resolve_judges(["gpt-5.4-mini", "qwen3.8"])),
        )
        for aliases in (["mistral"], ["qwen3.8", "qwen3.8"], []):
            with self.assertRaises(ValueError):
                resolve_judges(aliases)


import csv
import json
import tempfile
from pathlib import Path
from dataclasses import replace
from tests.test_tof_nof_evaluator import FakeJudgeClient, write_panel_transcripts
from tof_nof_evaluation import evaluate_transcript_files
from tof_nof_evaluation.reporting import export_results
from tof_nof_evaluation.storage import judge_lock


def member(client=None, alias="qwen3.8"):
    from tof_nof_evaluation.models import JudgePanelMember, default_judge_config

    definition = resolve_judges([alias])[0]
    return JudgePanelMember(
        definition.provider,
        default_judge_config(definition),
        "https://example.test/v1",
        client,
    )


class IndependentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sources = self.root / "transcripts"
        self.output = self.root / "results"
        write_panel_transcripts(self.sources)

    def run_judges(self, judges, **kwargs):
        return evaluate_transcript_files(
            transcript_dir=self.sources, output_dir=self.output, judges=judges, **kwargs
        )

    def test_one_judge_17_calls_and_complete_resume_without_client(self):
        client = FakeJudgeClient([1, 1, 0, 1, 0] + [1, 0, 1, 0] * 3)
        scores = self.run_judges([member(client)])
        self.assertEqual(17, len(client.calls))
        self.assertEqual(
            (2, 3, 2, 1),
            (
                scores[0].tof,
                scores[0].nof,
                scores[0].downward_flips,
                scores[0].upward_flips,
            ),
        )
        self.assertEqual(scores, self.run_judges([member()], resume=True))
        namespace = self.output / "judges/qwen3.8"
        checkpoint = next(namespace.glob("checkpoints/*/*.json"))
        raw = json.loads(checkpoint.read_text())
        self.assertEqual(1, raw["shared_turn_1"]["turn"])
        self.assertEqual(
            [2, 3, 4, 5],
            [t["turn"] for t in next(iter(raw["scenario_turns"].values()))],
        )
        self.assertEqual(
            1,
            json.loads((namespace / "summary.json").read_text())["coverage"][
                "complete_panels"
            ],
        )

    def test_interruption_and_added_peer_leave_completed_work_reusable(self):
        for stop in (0, 1, 6, 16):
            with self.subTest(stop=stop):
                self.output = self.root / f"results-{stop}"
                client = FakeJudgeClient([1] * 17, interrupt_after_calls=stop)
                with self.assertRaises(KeyboardInterrupt):
                    self.run_judges([member(client)])
                resumed = FakeJudgeClient([1] * 17)
                self.run_judges([member(resumed)], resume=True)
                self.assertEqual(17 - stop, len(resumed.calls))
                peer = FakeJudgeClient([0] * 17)
                scores = self.run_judges(
                    [member(peer, "gpt-5.4-mini"), member()], resume=True
                )
                self.assertEqual(8, len(scores))
                self.assertEqual(17, len(peer.calls))

    def test_cumulative_batches_and_offline_regeneration(self):
        self.run_judges([member(FakeJudgeClient([1] * 17))])
        write_panel_transcripts(self.sources, item_id="item-2")
        self.run_judges([member(FakeJudgeClient([0] * 17))], item_ids={"item-2"})
        path = next((self.output / "judges/qwen3.8/scores").glob("*/*/tof_nof.csv"))
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(["item-1", "item-2"], [r["id"] for r in rows])
        self.assertNotIn("consensus_label", rows[0])
        before = path.read_bytes()
        export_results(self.output)
        self.assertEqual(before, path.read_bytes())

    def test_existing_checkpoint_protected_and_all_selected_preflighted(self):
        self.run_judges([member(FakeJudgeClient([1] * 17))])
        client = FakeJudgeClient([1] * 34)
        with self.assertRaisesRegex(ValueError, "--resume"):
            self.run_judges([member(client)])
        write_panel_transcripts(self.sources, item_id="item-2", omit_scenario=3)
        with self.assertRaisesRegex(ValueError, "four scenarios"):
            self.run_judges([member(client)], resume=True)
        self.assertEqual([], client.calls)

    def test_locks_exclude_same_judge_but_allow_peer(self):
        with judge_lock(self.output / "judges/qwen3.8"):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.run_judges([member(FakeJudgeClient([1] * 17))])
            self.run_judges([member(FakeJudgeClient([1] * 17), "gpt-5.4-mini")])

    def test_corrupt_checkpoint_and_configuration_changes_fail_before_client_setup(
        self,
    ):
        self.run_judges([member(FakeJudgeClient([1] * 17))])
        path = next((self.output / "judges/qwen3.8").glob("checkpoints/*/*.json"))
        original = path.read_text()
        mutations = [
            lambda p: p["evaluation"].update(prompt_sha256="changed"),
            lambda p: p["evaluation"]["judge"].update(endpoint="changed"),
            lambda p: p["evaluation"]["judge"].update(model="changed"),
            lambda p: p["evaluation"]["judge"].update(max_tokens=16384),
            lambda p: p["shared_turn_1"]["judgment"].update(label=0),
            lambda p: p["shared_turn_1"]["judgment"].update(reason=""),
            lambda p: p["shared_turn_1"].update(assistant_response="changed"),
            lambda p: next(iter(p["scenario_turns"].values()))[0].update(turn=3),
        ]
        for mutate in mutations:
            payload = json.loads(original)
            mutate(payload)
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                self.run_judges(
                    [member()],
                    resume=True,
                    client_factory=lambda _: self.fail("client setup before preflight"),
                )
            path.write_text(original)
        for changed in (
            replace(member(), endpoint="changed"),
            replace(member(), config=replace(member().config, max_tokens=16384)),
        ):
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                self.run_judges([changed], resume=True)
        source = next(self.sources.glob("*/*/*.json"))
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source transcript panel has changed"):
            self.run_judges([member()], resume=True)

    def test_partial_trajectories_export_only_when_complete_and_lock_blocks_readers(
        self,
    ):
        client = FakeJudgeClient([1] * 17, interrupt_after_calls=6)
        with self.assertRaises(KeyboardInterrupt):
            self.run_judges([member(client)])
        namespace = self.output / "judges/qwen3.8"
        summary = json.loads((namespace / "summary.json").read_text())
        self.assertEqual(
            (0, 1, 6),
            tuple(
                summary["coverage"][k]
                for k in (
                    "complete_panels",
                    "complete_trajectories",
                    "unique_judgments",
                )
            ),
        )
        with judge_lock(namespace):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                export_results(self.output)
            from factorial_analysis import analyze_results

            with self.assertRaisesRegex(RuntimeError, "busy"):
                analyze_results(self.output)

    def test_later_selected_checkpoint_is_preflighted_before_pending_calls(self):
        write_panel_transcripts(self.sources, item_id="item-2")
        self.run_judges([member(FakeJudgeClient([1] * 17))], item_ids={"item-2"})
        path = next((self.output / "judges/qwen3.8").glob("checkpoints/*/*.json"))
        payload = json.loads(path.read_text())
        payload["shared_turn_1"]["judgment"]["label"] = 0
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            self.run_judges(
                [member()],
                resume=True,
                client_factory=lambda _: self.fail("preflight did not finish"),
            )

    def test_invalid_alias_cli_has_no_output_and_completed_resume_no_credentials(self):
        from tof_nof_evaluation.cli import main

        for aliases in (["mistral"], ["qwen3.8", "qwen3.8"]):
            with self.assertRaises(SystemExit):
                main(
                    ["--output-dir", str(self.output)]
                    + [v for a in aliases for v in ("--judge", a)]
                )
        self.assertFalse(self.output.exists())
        from tof_nof_evaluation.providers import configure_judges
        from tof_nof_evaluation.models import SUPPORTED_JUDGES

        configured = configure_judges(environ={}, aliases=["qwen3.8"])
        self.assertIsNone(configured[0].client)
        self.assertEqual(SUPPORTED_JUDGES["qwen3.8"], configured[0].definition)

    def test_raw_reasoning_survives_checkpoint_and_export(self):
        visible = '{"decision_type":"mixed","reason":"Both positions."}'
        raw = {
            "choices": [
                {
                    "message": {
                        "content": visible,
                        "reasoning_content": "Saved original reasoning",
                    }
                }
            ]
        }
        self.run_judges(
            [
                member(
                    FakeJudgeClient(
                        [0] * 17, responses=[visible] * 17, raw_responses=[raw] * 17
                    )
                )
            ]
        )
        path = next(
            (self.output / "judges/qwen3.8").glob("scores/*/*/tof_nof_judgments.csv")
        )
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(5, len(rows))
        self.assertEqual(raw, json.loads(rows[0]["judge_raw_response"]))
        self.assertEqual(visible, rows[0]["judge_output"])
        self.assertNotIn("consensus_label", rows[0])

    def test_cli_completed_resume_needs_no_credentials_or_sdk(self):
        from tof_nof_evaluation.cli import main
        from transcript_generation.models import DEFAULT_TARGET_MODELS
        from transcript_generation.storage import write_transcript
        from tests.test_tof_nof_evaluator import transcript_for
        from unittest.mock import patch

        sources = self.root / "cli-sources"
        target = DEFAULT_TARGET_MODELS[0]
        for index in range(4):
            transcript = transcript_for(scenario_index=index).model_copy(
                update={"target_model": target}
            )
            write_transcript(transcript, sources, overwrite=False)
        dataset = self.root / "dataset.jsonl"
        dataset.write_text(json.dumps(transcript.item.model_dump()) + "\n")
        evaluate_transcript_files(
            transcript_dir=sources,
            output_dir=self.output,
            judges=[member(FakeJudgeClient([1] * 17))],
            required_target_models={target},
            required_item_ids={"item-1"},
        )
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("tof_nof_evaluation.cli.load_dotenv_if_present"),
            patch(
                "openai.OpenAI",
                side_effect=AssertionError("SDK must not be constructed"),
            ),
        ):
            main(
                [
                    "--judge",
                    "qwen3.8",
                    "--target-model",
                    target,
                    "--item-id",
                    "item-1",
                    "--dataset-jsonl",
                    str(dataset),
                    "--transcript-dir",
                    str(sources),
                    "--output-dir",
                    str(self.output),
                    "--ki4buw-base-url",
                    "https://example.test/v1",
                    "--resume",
                ]
            )

    def test_selected_items_limit_and_shared_mismatch_preflight(self):
        from tof_nof_evaluation import load_transcript_panels

        write_panel_transcripts(self.sources, item_id="item-2")
        write_panel_transcripts(self.sources, item_id="item-10")
        panels = load_transcript_panels(
            self.sources, target_models={"target/model"}, limit=2
        )
        self.assertEqual(["item-1", "item-10"], [p.item_id for p in panels])
        path = next(self.sources.glob("*/*/item-2.json"))
        payload = json.loads(path.read_text())
        payload["turns"][0]["assistant"] = "Changed shared first turn"
        path.write_text(json.dumps(payload))
        # Unselected broken panel does not obstruct the selected item.
        self.run_judges(
            [member(FakeJudgeClient([1] * 17))],
            item_ids={"item-1"},
            required_target_models={"target/model"},
            required_item_ids={"item-1"},
        )
        with self.assertRaisesRegex(ValueError, "shared turn 1 assistant"):
            self.run_judges(
                [member()], resume=True, client_factory=lambda _: self.fail("preflight")
            )

    def test_corrupt_shared_turn_storage_cannot_be_encoded_as_four_copies(self):
        self.run_judges([member(FakeJudgeClient([1] * 17))])
        path = next((self.output / "judges/qwen3.8").glob("checkpoints/*/*.json"))
        payload = json.loads(path.read_text())
        shared = payload["shared_turn_1"]
        payload["shared_turn_1"] = None
        for turns in payload["scenario_turns"].values():
            turns.insert(0, shared)
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "shared turn"):
            export_results(self.output)
