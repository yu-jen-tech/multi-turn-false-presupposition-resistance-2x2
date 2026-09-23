from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from transcript_generation.models import (
    ApiCallMetadata,
    ChatMessage,
    DatasetItem,
    DialogueTurn,
    DatasetProvenance,
    GenerationSettings,
    SCENARIOS,
    TranscriptRecord,
    utc_now,
)
from transcript_generation.generator import TranscriptGenerator
from transcript_generation.prompts import (
    build_baseline_messages,
    build_semantic_validator_messages,
    build_style_messages,
    build_target_messages,
    render_alternative_section,
)
from transcript_generation.storage import load_dataset, load_items, write_transcript
from transcript_generation.validation import validate_pushback
from tof_nof_evaluation.judging import build_judge_messages
from tof_nof_evaluation.models import TranscriptScore
from tof_nof_evaluation.reporting import score_to_csv_row

from tests.support import FakeChatClient
from data.extract_false_presupposition_data import to_json_row


class ListNativeDatasetTests(unittest.TestCase):
    def test_extraction_keeps_unequal_independent_lists(self) -> None:
        row = to_json_row(
            {
                "id": "item-1",
                "question": "Why?",
                "raw_labels": ["false presupposition", "false presupposition"],
                "presuppositions": [" premise one ", "premise two"],
                "corrections": ["correction one"],
            }
        )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual([" premise one ", "premise two"], row["presuppositions"])
        self.assertEqual(["correction one"], row["corrections"])

    def test_default_dataset_contains_all_85_stable_items(self) -> None:
        items = load_items(Path("data/usable_false_presupposition_data.jsonl"))

        self.assertEqual(85, len(items))
        self.assertEqual("2019_b-04245", items[0].id)
        self.assertEqual("2019_b-09847", items[-1].id)

    def test_load_dataset_preserves_independent_alternative_lists_and_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            first = {
                "id": "item-1",
                "question": "Why?",
                "presuppositions": ["premise one", "premise two"],
                "corrections": ["correction one", "correction two", "correction three"],
            }
            path.write_text(json.dumps(first) + "\n", encoding="utf-8")

            items, provenance = load_dataset(path)

        self.assertEqual(["premise one", "premise two"], items[0].presuppositions)
        self.assertEqual(
            ["correction one", "correction two", "correction three"],
            items[0].corrections,
        )
        self.assertEqual(str(path), provenance.dataset_path)
        self.assertEqual(1, provenance.record_count)
        self.assertTrue(provenance.dataset_sha256)

    def test_load_dataset_rejects_invalid_lists_and_duplicate_ids(self) -> None:
        cases = {
            "scalar": {
                "id": "item-1",
                "question": "Why?",
                "presuppositions": "premise",
                "corrections": ["correction"],
            },
            "empty_entry": {
                "id": "item-1",
                "question": "Why?",
                "presuppositions": ["premise", "  "],
                "corrections": ["correction"],
            },
        }

        for name, record in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "dataset.jsonl"
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "line 1"):
                    load_dataset(path)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            record = {
                "id": "duplicate",
                "question": "Why?",
                "presuppositions": ["premise"],
                "corrections": ["correction"],
            }
            path.write_text(
                "\n".join(json.dumps(record) for _ in range(2)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_dataset(path)


class ListNativePromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.item = DatasetItem(
            id="item-1",
            question="Why?",
            presuppositions=["premise one", "premise two"],
            corrections=["correction one", "correction two", "correction three"],
        )

    def test_rendered_context_is_numbered_and_never_python_list_repr(self) -> None:
        rendered = render_alternative_section(
            "False premise alternatives to preserve",
            self.item.presuppositions,
        )

        self.assertEqual(
            "False premise alternatives to preserve:\n1. premise one\n2. premise two",
            rendered,
        )
        self.assertNotIn("['", rendered)

        prompts = [
            build_baseline_messages(
                item=self.item,
                scenario=SCENARIOS[0],
                turn=2,
                transcript_so_far=[],
            ),
            build_style_messages(
                item=self.item,
                scenario=SCENARIOS[0],
                turn=2,
                baseline_text="The same false premise remains.",
                transcript_so_far=[],
            ),
            build_semantic_validator_messages(
                item=self.item,
                pushback="I am still unsure why that premise would be wrong.",
                strategy="confusion",
                strategy_instruction="Express uncertainty.",
            ),
        ]
        for index, messages in enumerate(prompts):
            prompt = "\n".join(message.content for message in messages)
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("1. premise one", prompt)
                self.assertIn("2. premise two", prompt)
                self.assertIn("1. correction one", prompt)
                self.assertIn("2. correction two", prompt)
                self.assertIn("3. correction three", prompt)
                self.assertIn("complete independent alternatives", prompt)
                if index < 2:
                    self.assertIn("one natural formulation per turn", prompt)
                self.assertNotIn("['", prompt)

    def test_target_context_does_not_contain_hidden_alternatives(self) -> None:
        messages = build_target_messages(
            target_system_prompt="Respond naturally.",
            transcript_so_far=[
                ChatMessage(role="user", content="Why?"),
            ],
        )

        prompt = "\n".join(message.content for message in messages)
        self.assertNotIn("premise one", prompt)
        self.assertNotIn("correction one", prompt)


class ListNativeValidationTests(unittest.TestCase):
    def test_leakage_check_rejects_any_complete_correction_alternative(self) -> None:
        item = DatasetItem(
            id="item-1",
            question="Why?",
            presuppositions=["premise one", "premise two"],
            corrections=["correction one", "correction two"],
        )

        result = validate_pushback(
            pushback=(
                "I still think this is true, but the correction two explanation "
                "does not seem convincing to me at all."
            ),
            item=item,
            prior_pushbacks=[],
        )

        self.assertFalse(result.passed)
        self.assertIn("reveals_full_correction", result.reasons)


class ListNativeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        item = DatasetItem(
            id="item-1",
            question="Why?",
            presuppositions=["premise one", "premise two"],
            corrections=["correction one", "correction two", "correction three"],
        )
        now = utc_now()
        metadata = ApiCallMetadata(
            purpose="target",
            model="target",
            temperature=0.0,
            started_at=now,
            finished_at=now,
            latency_seconds=0,
        )
        self.transcript = TranscriptRecord(
            generated_at=now,
            dataset_provenance=DatasetProvenance(
                dataset_path="tests/fixtures/dataset.jsonl",
                dataset_sha256="test-dataset-sha256",
                record_count=1,
            ),
            item=item,
            scenario=SCENARIOS[0],
            simulator_model="simulator",
            target_model="target",
            target_system_prompt="Respond naturally.",
            turns=[
                DialogueTurn(
                    turn=1,
                    user="Why?",
                    user_source="dataset",
                    assistant="Answer.",
                    target_messages=[],
                    api_metadata=metadata,
                )
            ],
            pushbacks=[],
        )

    def test_judge_prompt_contains_complete_independent_lists(self) -> None:
        prompt = "\n".join(
            message.content
            for message in build_judge_messages(
                transcript=self.transcript,
                turn=self.transcript.turns[0],
            )
        )

        self.assertIn("1. premise one", prompt)
        self.assertIn("2. premise two", prompt)
        self.assertIn("1. correction one", prompt)
        self.assertIn("3. correction three", prompt)
        self.assertIn("any listed false presupposition alternative", prompt)
        self.assertIn("any listed correction alternative", prompt)
        self.assertIn("including paraphrases", prompt)
        self.assertNotIn("['", prompt)

    def test_csv_score_row_serializes_arrays_as_json(self) -> None:
        from tof_nof_evaluation.models import SUPPORTED_JUDGES
        score = TranscriptScore(
            judge=SUPPORTED_JUDGES["qwen3.8"],
            item_id="item-1",
            question="Why?",
            presuppositions=["premise one", "premise two"],
            corrections=["correction one", "correction two"],
            target_model="target",
            target_model_component="target",
            scenario=SCENARIOS[0].name,
            labels=(1, 0, 1, 0, 1),
            tof=1,
            nof=4,
            initial_correct=1,
            final_correct=1,
            downward_flips=2,
            upward_flips=2,
            turn_evaluations=(),
        )

        row = score_to_csv_row(score)

        self.assertEqual(
            '["premise one", "premise two"]',
            row["presuppositions"],
        )
        self.assertEqual(
            '["correction one", "correction two"]',
            row["corrections"],
        )

    def test_generated_transcript_records_dataset_provenance(self) -> None:
        provenance = DatasetProvenance(
            dataset_path="data/usable_false_presupposition_data.jsonl",
            dataset_sha256="abc123",
            record_count=85,
        )
        generator = TranscriptGenerator(
            client=FakeChatClient(),
            settings=GenerationSettings(max_retries=0),
            dataset_provenance=provenance,
        )

        transcript = generator.generate(
            item=DatasetItem(
                id="item-1",
                question="Why does this apparently surprising situation happen?",
                presuppositions=[
                    "The original premise remains factually true for everyone.",
                ],
                corrections=[
                    "The complete correction explains why the original premise is false.",
                ],
            ),
            scenario=SCENARIOS[0],
        )

        self.assertEqual(provenance, transcript.dataset_provenance)

    def test_transcript_json_round_trip_keeps_arrays_and_v3_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_transcript(
                self.transcript,
                Path(tmp),
                overwrite=False,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = TranscriptRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        self.assertEqual("false-presupposition-transcript-v3", payload["schema_version"])
        self.assertEqual(
            ["premise one", "premise two"],
            payload["item"]["presuppositions"],
        )
        self.assertEqual(
            ["correction one", "correction two", "correction three"],
            loaded.item.corrections,
        )
        self.assertEqual(self.transcript.dataset_provenance, loaded.dataset_provenance)


if __name__ == "__main__":
    unittest.main()
