from __future__ import annotations

import json
import tempfile
from pathlib import Path

from transcript_generation import (
    BatchGenerationPlan,
    GenerationSettings,
    SCENARIOS,
    generate_transcripts,
)

from tests.support import (
    TEST_DATASET_PROVENANCE,
    FakeChatClient,
    TranscriptGenerationTestCase,
)


class BatchGenerationTests(TranscriptGenerationTestCase):
    def test_shares_pushbacks_across_target_models(self) -> None:
        client = FakeChatClient()

        with tempfile.TemporaryDirectory() as tmp:
            written = generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0]],
                    settings=[
                        GenerationSettings(target_model="model-a", max_retries=0),
                        GenerationSettings(target_model="model-b", max_retries=0),
                    ],
                    output_dir=Path(tmp) / "transcript",
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                ),
                client=client,
                progress=lambda _: None,
            )
            transcripts = [
                json.loads(path.read_text(encoding="utf-8")) for path in written
            ]

        self.assertEqual(2, len(written))
        self.assertEqual(8, client.simulator_count)
        self.assertEqual(8, client.validator_count)
        self.assertEqual(4, client.simulator_baseline_count)
        self.assertEqual(4, client.simulator_style_count)
        self.assertEqual(10, client.target_count)
        self.assertEqual(
            [pushback["text"] for pushback in transcripts[0]["pushbacks"]],
            [pushback["text"] for pushback in transcripts[1]["pushbacks"]],
        )
        self.assertEqual(
            ["model-a", "model-b"],
            [transcript["target_model"] for transcript in transcripts],
        )

    def test_shares_baselines_across_styles_and_target_models(self) -> None:
        client = FakeChatClient()

        with tempfile.TemporaryDirectory() as tmp:
            written = generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0], SCENARIOS[2]],
                    settings=[
                        GenerationSettings(target_model="model-a", max_retries=0),
                        GenerationSettings(target_model="model-b", max_retries=0),
                    ],
                    output_dir=Path(tmp) / "transcript",
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                ),
                client=client,
                progress=lambda _: None,
            )
            transcripts = [
                json.loads(path.read_text(encoding="utf-8")) for path in written
            ]

        self.assertEqual(4, client.simulator_baseline_count)
        self.assertEqual(8, client.simulator_style_count)
        self.assertEqual(12, client.validator_count)
        baselines = {
            pushback["baseline"]["text"]
            for transcript in transcripts
            for pushback in transcript["pushbacks"]
        }
        self.assertEqual(4, len(baselines))
        for turn in range(4):
            self.assertEqual(
                1,
                len({
                    transcript["pushbacks"][turn]["baseline"]["text"]
                    for transcript in transcripts
                }),
            )
        for scenario_name in (
            "formal_without_simulator_history",
            "humanlike_without_simulator_history",
        ):
            scenario_transcripts = [
                transcript
                for transcript in transcripts
                if transcript["scenario"]["name"] == scenario_name
            ]
            self.assertEqual(
                1,
                len({
                    tuple(pushback["text"] for pushback in transcript["pushbacks"])
                    for transcript in scenario_transcripts
                }),
            )

    def test_with_history_does_not_share_pushbacks_between_branches(self) -> None:
        client = FakeChatClient()

        with tempfile.TemporaryDirectory() as tmp:
            generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[1], SCENARIOS[3]],
                    settings=[GenerationSettings(target_model="model-a", max_retries=0)],
                    output_dir=Path(tmp) / "transcript",
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                ),
                client=client,
                progress=lambda _: None,
            )

        self.assertEqual(8, client.simulator_baseline_count)
        self.assertEqual(8, client.simulator_style_count)
        self.assertEqual(16, client.validator_count)
        style_prompts = [
            call["messages"][-1].content
            for call in client.calls
            if call["purpose"] == "simulator_style"
        ]
        self.assertIn("target response 1", style_prompts[0])
        self.assertIn("target response 1", style_prompts[4])
        self.assertIn("target response 6", style_prompts[5])

    def test_resumes_complete_transcripts_without_api_calls(self) -> None:
        settings = [GenerationSettings(target_model="model-a", max_retries=0)]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "transcript"
            generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0]],
                    settings=settings,
                    output_dir=output_dir,
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                ),
                client=FakeChatClient(),
                progress=lambda _: None,
            )
            resumed_client = FakeChatClient()
            progress_messages: list[str] = []

            written = generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0]],
                    settings=settings,
                    output_dir=output_dir,
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                    resume=True,
                ),
                client=resumed_client,
                progress=progress_messages.append,
            )

        self.assertEqual(1, len(written))
        self.assertEqual([], resumed_client.calls)
        self.assertEqual([f"Skipping existing {written[0]}"], progress_messages)

    def test_reuses_persisted_pushbacks_for_a_new_target_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "transcript"
            generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0]],
                    settings=[
                        GenerationSettings(target_model="model-a", max_retries=0)
                    ],
                    output_dir=output_dir,
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                ),
                client=FakeChatClient(),
                progress=lambda _: None,
            )
            resumed_client = FakeChatClient()

            written = generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0]],
                    settings=[
                        GenerationSettings(target_model="model-a", max_retries=0),
                        GenerationSettings(target_model="model-b", max_retries=0),
                    ],
                    output_dir=output_dir,
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                    resume=True,
                ),
                client=resumed_client,
                progress=lambda _: None,
            )
            new_transcript = json.loads(written[1].read_text(encoding="utf-8"))

        self.assertEqual(0, resumed_client.simulator_count)
        self.assertEqual(0, resumed_client.validator_count)
        self.assertEqual(5, resumed_client.target_count)
        self.assertTrue(
            all(
                pushback["pushback_source"]
                == "reused_from_without_simulator_history"
                for pushback in new_transcript["pushbacks"]
            )
        )

    def test_reuses_first_response_for_a_new_scenario(self) -> None:
        settings = [GenerationSettings(target_model="model-a", max_retries=0)]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "transcript"
            original_client = FakeChatClient(call_id_prefix="original")
            original_path = generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0]],
                    settings=settings,
                    output_dir=output_dir,
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                ),
                client=original_client,
                progress=lambda _: None,
            )[0]
            original = json.loads(original_path.read_text(encoding="utf-8"))
            resumed_client = FakeChatClient()

            written = generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=SCENARIOS[:2],
                    settings=settings,
                    output_dir=output_dir,
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                    resume=True,
                ),
                client=resumed_client,
                progress=lambda _: None,
            )
            added = json.loads(written[1].read_text(encoding="utf-8"))

        self.assertEqual(4, resumed_client.target_count)
        self.assertEqual(
            original["turns"][0]["assistant"],
            added["turns"][0]["assistant"],
        )
        self.assertEqual(
            original["turns"][0]["api_metadata"]["call_id"],
            added["turns"][0]["api_metadata"]["call_id"],
        )

    def test_overwrite_regenerates_existing_transcripts(self) -> None:
        settings = [GenerationSettings(target_model="model-a", max_retries=0)]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "transcript"
            original_path = generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0]],
                    settings=settings,
                    output_dir=output_dir,
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                ),
                client=FakeChatClient(),
                progress=lambda _: None,
            )[0]
            original = json.loads(original_path.read_text(encoding="utf-8"))
            overwrite_client = FakeChatClient(call_id_prefix="overwrite")

            overwritten_path = generate_transcripts(
                plan=BatchGenerationPlan(
                    items=[self.item],
                    scenarios=[SCENARIOS[0]],
                    settings=settings,
                    output_dir=output_dir,
                    dataset_provenance=TEST_DATASET_PROVENANCE,
                    resume=True,
                    overwrite=True,
                ),
                client=overwrite_client,
                progress=lambda _: None,
            )[0]
            overwritten = json.loads(overwritten_path.read_text(encoding="utf-8"))

        self.assertEqual(8, overwrite_client.simulator_count)
        self.assertEqual(8, overwrite_client.validator_count)
        self.assertEqual(5, overwrite_client.target_count)
        self.assertNotEqual(
            original["turns"][0]["api_metadata"]["call_id"],
            overwritten["turns"][0]["api_metadata"]["call_id"],
        )
