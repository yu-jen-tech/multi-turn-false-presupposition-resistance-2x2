from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_independent_judges import member
from tests.test_tof_nof_evaluator import FakeJudgeClient, write_panel_transcripts
from tof_nof_evaluation.evaluation import evaluate_transcript_files
from tof_nof_evaluation.legacy_import import import_legacy
from tof_nof_evaluation.reporting import export_results


def legacy_fixture(root):
    sources, independent, legacy = (
        root / name for name in ("sources", "independent", "legacy")
    )
    write_panel_transcripts(sources)
    evaluate_transcript_files(
        transcript_dir=sources,
        output_dir=independent,
        judges=[
            member(FakeJudgeClient([1] * 17)),
            member(FakeJudgeClient([0] * 17), "gpt-5.4-mini"),
        ],
    )
    paths = list(independent.glob("judges/*/checkpoints/*/*.json"))
    qwen = json.loads(next(p for p in paths if "qwen3.8" in p.parts).read_text())
    gpt = json.loads(next(p for p in paths if "gpt-5.4-mini" in p.parts).read_text())
    mistral_config = {
        **qwen["evaluation"]["judge"],
        "model": "oski/inferenz-mistral-small-4-119b",
        "reasoning_effort": "high",
    }
    scenarios = {}
    for scenario in qwen["scenario_turns"]:
        scenarios[scenario] = []
        for q, g in zip(
            [qwen["shared_turn_1"]] + qwen["scenario_turns"][scenario],
            [gpt["shared_turn_1"]] + gpt["scenario_turns"][scenario],
        ):
            m = {**q["judgment"], "judge_model": mistral_config["model"]}
            scenarios[scenario].append(
                {
                    **{k: v for k, v in q.items() if k != "judgment"},
                    "label": 1,
                    "judge_evaluations": [m, g["judgment"], q["judgment"]],
                }
            )
    payload = {
        "schema_version": "tof-nof-multi-judge-checkpoint-v1",
        "panel": qwen["panel"],
        "evaluation": {
            **{k: v for k, v in qwen["evaluation"].items() if k != "judge"},
            "judges": [
                mistral_config,
                gpt["evaluation"]["judge"],
                qwen["evaluation"]["judge"],
            ],
        },
        "turn_evaluations": scenarios,
    }
    path = legacy / "checkpoints/target__model/item-1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    return legacy, path, payload


class ImportTests(unittest.TestCase):
    def test_historical_xhigh_pilot_remains_importable_and_exportable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy, path, payload = legacy_fixture(root)
            for judge in payload["evaluation"]["judges"]:
                if judge["model"] in ("openai/qwen3.8", "gpt-5.4-mini"):
                    judge["reasoning_effort"] = "xhigh"
            path.write_text(json.dumps(payload))

            output = root / "imported"
            import_legacy(legacy, output)
            export_results(output)
            self.assertEqual(
                {"qwen3.8", "gpt-5.4-mini"},
                {namespace.name for namespace in (output / "judges").iterdir()},
            )
            for alias in ("qwen3.8", "gpt-5.4-mini"):
                manifest = json.loads((output / "judges" / alias / "manifest.json").read_text())
                self.assertEqual("xhigh", manifest["judge"]["reasoning_effort"])

    def test_offline_import_arbitrary_order_is_idempotent_and_preserves_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy, source, payload = legacy_fixture(root)
            original = source.read_bytes()
            with patch("openai.OpenAI", side_effect=AssertionError("offline")):
                import_legacy(legacy, root / "imported")
                before = {
                    p: p.read_bytes() for p in (root / "imported").rglob("*.json")
                }
                import_legacy(legacy, root / "imported")
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertEqual(original, source.read_bytes())
            q = json.loads(
                next(
                    (root / "imported/judges/qwen3.8").glob("checkpoints/*/*.json")
                ).read_text()
            )
            self.assertEqual(
                payload["turn_evaluations"][next(iter(payload["turn_evaluations"]))][0][
                    "judge_evaluations"
                ][2],
                q["shared_turn_1"]["judgment"],
            )
            self.assertEqual(str(source), q["import_provenance"]["source_checkpoint"])

    def test_invalid_legacy_batch_fails_without_destination_writes(self):
        mutations = [
            lambda p: p["evaluation"].update(
                evaluator_prompt_version="tof-nof-multi-judge-v1"
            ),
            lambda p: p["evaluation"].update(prompt_sha256="changed"),
            lambda p: p["evaluation"]["judges"].append(p["evaluation"]["judges"][0]),
            lambda p: next(iter(p["turn_evaluations"].values()))[0].update(label=0),
            lambda p: next(iter(p["turn_evaluations"].values()))[0][
                "judge_evaluations"
            ][0].update(label=0),
            lambda p: next(iter(p["turn_evaluations"].values()))[0].update(
                assistant_response="changed"
            ),
            lambda p: next(iter(p["turn_evaluations"].values()))[0][
                "judge_evaluations"
            ][0].update(reason="different shared turn"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy, path, payload = legacy_fixture(root)
            for mutate in mutations:
                changed = copy.deepcopy(payload)
                mutate(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    import_legacy(legacy, root / "invalid")
                self.assertFalse((root / "invalid").exists())

    def test_conflicting_destination_does_not_modify_other_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy, path, payload = legacy_fixture(root)
            import_legacy(legacy, root / "imported", aliases=["gpt-5.4-mini"])
            target = next((root / "imported").glob("judges/*/checkpoints/*/*.json"))
            changed = json.loads(target.read_text())
            changed["scenario_turns"][next(iter(changed["scenario_turns"]))][0][
                "judgment"
            ]["reason"] = "Different saved evaluation"
            target.write_text(json.dumps(changed))
            before = target.read_bytes()
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                import_legacy(legacy, root / "imported")
            self.assertEqual(before, target.read_bytes())
            self.assertFalse((root / "imported/judges/qwen3.8/manifest.json").exists())
