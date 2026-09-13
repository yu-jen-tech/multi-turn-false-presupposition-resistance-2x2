from __future__ import annotations

import json
import tempfile
from pathlib import Path

from transcript_generation import (
    DatasetItem,
    GenerationSettings,
    SCENARIOS,
    TranscriptGenerator,
    load_transcript,
)
from transcript_generation.storage import (
    resumable_transcript_exists,
    reusable_baselines,
    reusable_first_response,
    reusable_pushbacks,
    select_items,
    transcript_path,
    write_transcript,
)

from tests.support import (
    TEST_DATASET_PROVENANCE,
    FakeChatClient,
    TranscriptGenerationTestCase,
)


class TranscriptStorageTests(TranscriptGenerationTestCase):
    def generator(self, target_model: str = "oski/inferenz-gpt-oss-120b") -> TranscriptGenerator:
        return TranscriptGenerator(
            client=FakeChatClient(),
            settings=GenerationSettings(
                target_model=target_model,
                max_retries=0,
            ),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        )

    def test_write_transcript_uses_scenario_and_item_folders(self) -> None:
        transcript = self.generator().generate(
            item=self.item,
            scenario=SCENARIOS[2],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = write_transcript(transcript, Path(tmp), overwrite=False)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            Path(tmp)
            / "humanlike_without_simulator_history"
            / "oski__inferenz-gpt-oss-120b"
            / "2019_b-05111.json",
            path,
        )
        self.assertEqual("false-presupposition-transcript-v3", payload["schema_version"])
        self.assertEqual("2019_b-05111", payload["item"]["id"])
        self.assertIn("baseline", payload["pushbacks"][0])
        self.assertNotIn(
            "preserves_baseline_semantics",
            payload["pushbacks"][0]["baseline"]["validation"]["semantic"],
        )
        self.assertEqual(
            "simulator_baseline",
            payload["pushbacks"][0]["baseline"]["api_metadata"]["purpose"],
        )
        self.assertEqual(
            "simulator_style",
            payload["pushbacks"][0]["api_metadata"]["purpose"],
        )

    def test_transcript_path_uses_single_target_model_folder(self) -> None:
        path = transcript_path(
            Path("transcript"),
            scenario=SCENARIOS[0],
            target_model="oski/inferenz-gpt-oss-120b",
            item_id="abc",
        )

        self.assertEqual(
            Path("transcript")
            / "formal_without_simulator_history"
            / "oski__inferenz-gpt-oss-120b"
            / "abc.json",
            path,
        )

    def test_reusable_first_response_loads_resume_state(self) -> None:
        transcript = self.generator().generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_transcript(transcript, output_dir, overwrite=False)
            response = reusable_first_response(
                output_dir,
                scenarios=SCENARIOS[:2],
                target_model="oski/inferenz-gpt-oss-120b",
                item_id=self.item.id,
            )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual("I disagree. target response 1", response.content)
        self.assertEqual(
            transcript.turns[0].api_metadata.call_id,
            response.metadata.call_id,
        )

    def test_reusable_first_response_rejects_inconsistent_resume_state(self) -> None:
        generator = self.generator()
        first = generator.generate(item=self.item, scenario=SCENARIOS[0])
        second = generator.generate(item=self.item, scenario=SCENARIOS[1])

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_transcript(first, output_dir, overwrite=False)
            write_transcript(second, output_dir, overwrite=False)

            with self.assertRaisesRegex(RuntimeError, "different first answers"):
                reusable_first_response(
                    output_dir,
                    scenarios=SCENARIOS[:2],
                    target_model="oski/inferenz-gpt-oss-120b",
                    item_id=self.item.id,
                )

    def test_reusable_pushbacks_loads_consistent_resume_state(self) -> None:
        first = self.generator("model-a").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        second = self.generator("model-b").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first_path = write_transcript(first, output_dir, overwrite=False)
            write_transcript(second, output_dir, overwrite=False)
            pushbacks = reusable_pushbacks(
                output_dir,
                scenario=SCENARIOS[0],
                item_id=self.item.id,
            )

        self.assertIsNotNone(pushbacks)
        assert pushbacks is not None
        self.assertEqual(
            [pushback.text for pushback in first.pushbacks],
            [pushback.text for pushback in pushbacks],
        )
        self.assertTrue(
            all(
                pushback.pushback_source
                == "reused_from_without_simulator_history"
                for pushback in pushbacks
            )
        )
        self.assertEqual(str(first_path), pushbacks[0].source_transcript_path)

    def test_reusable_baselines_loads_one_shared_baseline_sequence(self) -> None:
        first = self.generator("model-a").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        second = self.generator("model-b").generate(
            item=self.item,
            scenario=SCENARIOS[2],
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_transcript(first, output_dir, overwrite=False)
            write_transcript(second, output_dir, overwrite=False)
            baselines = reusable_baselines(
                output_dir,
                scenarios=SCENARIOS,
                item_id=self.item.id,
            )

        self.assertIsNotNone(baselines)
        assert baselines is not None
        self.assertEqual(
            [pushback.baseline.text for pushback in first.pushbacks],
            [baseline.text for baseline in baselines],
        )

    def test_reusable_baselines_rejects_conflicting_validation_configuration(self) -> None:
        first = self.generator("model-a").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        second = self.generator("model-b").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        assert second.pushbacks[0].baseline is not None
        assert second.pushbacks[0].baseline.semantic_validation_api_metadata is not None
        second.pushbacks[0].baseline.semantic_validation_api_metadata.model = (
            "different-validator"
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_transcript(first, output_dir, overwrite=False)
            write_transcript(second, output_dir, overwrite=False)

            with self.assertRaisesRegex(RuntimeError, "conflicting baseline"):
                reusable_baselines(
                    output_dir,
                    scenarios=SCENARIOS,
                    item_id=self.item.id,
                )

    def test_resume_rejects_pre_semantic_validation_transcripts(self) -> None:
        transcript = self.generator("model-a").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            valid_path = write_transcript(transcript, output_dir, overwrite=False)
            self.assertTrue(resumable_transcript_exists(valid_path))

            self.strip_semantic_validation(transcript)
            stale_path = write_transcript(transcript, output_dir, overwrite=True)

            self.assertIsNone(
                reusable_pushbacks(
                    output_dir,
                    scenario=SCENARIOS[0],
                    item_id=self.item.id,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "semantically validated"):
                resumable_transcript_exists(stale_path)

    def test_resume_rejects_styled_records_without_baseline_alignment(self) -> None:
        transcript = self.generator("model-a").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        assert transcript.pushbacks[0].validation.semantic is not None
        transcript.pushbacks[0].validation.semantic.preserves_baseline_semantics = None

        with tempfile.TemporaryDirectory() as tmp:
            path = write_transcript(transcript, Path(tmp), overwrite=False)

            with self.assertRaisesRegex(RuntimeError, "semantically validated"):
                resumable_transcript_exists(path)

    def test_resume_rejects_styled_records_without_style_validation(self) -> None:
        transcript = self.generator("model-a").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        assert transcript.pushbacks[0].validation.semantic is not None
        transcript.pushbacks[0].validation.semantic.matches_requested_style = None

        with tempfile.TemporaryDirectory() as tmp:
            path = write_transcript(transcript, Path(tmp), overwrite=False)

            with self.assertRaisesRegex(RuntimeError, "semantically validated"):
                resumable_transcript_exists(path)

    def test_reusable_pushbacks_rejects_inconsistent_resume_state(self) -> None:
        first = self.generator("model-a").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        second = self.generator("model-b").generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        second.pushbacks[0].text = (
            "I'm confused; the original false premise still stands, so why are you "
            "saying otherwise?"
        )
        second.turns[1].user = second.pushbacks[0].text

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_transcript(first, output_dir, overwrite=False)
            write_transcript(second, output_dir, overwrite=False)

            with self.assertRaisesRegex(RuntimeError, "different pushbacks"):
                reusable_pushbacks(
                    output_dir,
                    scenario=SCENARIOS[0],
                    item_id=self.item.id,
                )

    def test_select_items_respects_question_count(self) -> None:
        items = [
            self.item,
            DatasetItem(
                id="second",
                question="Second?",
                presuppositions=["False second premise."],
                corrections=["Second correction."],
            ),
            DatasetItem(
                id="third",
                question="Third?",
                presuppositions=["False third premise."],
                corrections=["Third correction."],
            ),
        ]

        selected = select_items(items, item_index=1, num_questions=2)

        self.assertEqual(["second", "third"], [item.id for item in selected])

    def test_scalar_transcript_is_not_accepted_or_resumable(self) -> None:
        transcript = self.generator().generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )
        payload = transcript.model_dump(mode="json")
        payload["schema_version"] = "false-presupposition-transcript-v2"
        payload["item"]["presuppositions"] = "False premise."
        payload["item"]["corrections"] = "Gold correction."

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scalar-transcript.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_transcript(path)
            with self.assertRaisesRegex(RuntimeError, "current schema"):
                resumable_transcript_exists(path)
