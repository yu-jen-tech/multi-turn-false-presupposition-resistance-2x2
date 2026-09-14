from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import BaseModel, ValidationError

from transcript_generation.models import (
    BatchGenerationPlan,
    CallPurpose,
    DatasetItem,
    DatasetProvenance,
    DEFAULT_SIMULATOR_MODEL,
    DEFAULT_TARGET_MODELS,
    DEFAULT_VALIDATOR_MODEL,
    TARGET_RESPONSE_HARD_WORD_LIMIT,
    TARGET_RESPONSE_SOFT_WORD_LIMIT,
    TARGET_RESPONSE_WORD_LIMIT,
    GenerationSettings,
    SCENARIOS,
    TRANSCRIPT_SCHEMA_VERSION,
)


class GenerationModelTests(unittest.TestCase):
    def test_new_schema_and_simulator_call_purposes_are_declared(self) -> None:
        self.assertEqual(
            "false-presupposition-transcript-v3",
            TRANSCRIPT_SCHEMA_VERSION,
        )
        self.assertEqual(120, TARGET_RESPONSE_WORD_LIMIT)
        self.assertEqual(120, TARGET_RESPONSE_SOFT_WORD_LIMIT)
        self.assertEqual(200, TARGET_RESPONSE_HARD_WORD_LIMIT)
        purposes: tuple[CallPurpose, ...] = (
            "simulator_baseline",
            "simulator_style",
            "target",
            "validator",
        )
        self.assertEqual(
            ["simulator_baseline", "simulator_style", "target", "validator"],
            list(purposes),
        )

    def test_batch_plan_rejects_duplicate_dataset_ids(self) -> None:
        item = DatasetItem(
            id="duplicate",
            question="Question?",
            presuppositions=["False premise."],
            corrections=["Correction."],
        )

        with self.assertRaisesRegex(ValidationError, "unique dataset IDs"):
            BatchGenerationPlan(
                items=[item, item],
                scenarios=[SCENARIOS[0]],
                settings=[GenerationSettings(target_model="model-a")],
                output_dir=Path("output"),
                dataset_provenance=DatasetProvenance(
                    dataset_path="dataset.jsonl",
                    dataset_sha256="sha256",
                    record_count=1,
                ),
            )

    def test_batch_generation_plan_is_a_frozen_pydantic_model(self) -> None:
        item = DatasetItem(
            id="item-1",
            question="Question?",
            presuppositions=["False premise."],
            corrections=["Correction."],
        )
        settings = GenerationSettings(target_model="model-a")
        plan = BatchGenerationPlan(
            items=[item],
            scenarios=[SCENARIOS[0]],
            settings=[settings],
            output_dir=Path("output"),
            dataset_provenance=DatasetProvenance(
                dataset_path="dataset.jsonl",
                dataset_sha256="sha256",
                record_count=1,
            ),
        )

        self.assertIsInstance(plan, BaseModel)
        self.assertEqual([item], plan.items)
        self.assertEqual([SCENARIOS[0]], plan.scenarios)
        self.assertEqual([settings], plan.settings)
        self.assertEqual(Path("output"), plan.output_dir)
        self.assertFalse(plan.resume)
        self.assertFalse(plan.overwrite)
        with self.assertRaises(ValidationError):
            plan.output_dir = Path("different-output")

    def test_batch_generation_plan_preserves_sequence_and_path_inputs(self) -> None:
        item = DatasetItem(
            id="item-1",
            question="Question?",
            presuppositions=["False premise."],
            corrections=["Correction."],
        )
        settings = GenerationSettings(target_model="model-a")

        plan = BatchGenerationPlan(
            items=(item,),
            scenarios=(SCENARIOS[0],),
            settings=(settings,),
            output_dir="output",
            dataset_provenance=DatasetProvenance(
                dataset_path="dataset.jsonl",
                dataset_sha256="sha256",
                record_count=1,
            ),
            resume=True,
            overwrite=True,
        )

        self.assertEqual((item,), plan.items)
        self.assertEqual((SCENARIOS[0],), plan.scenarios)
        self.assertEqual((settings,), plan.settings)
        self.assertEqual(Path("output"), plan.output_dir)
        self.assertTrue(plan.resume)
        self.assertTrue(plan.overwrite)

    def test_batch_generation_plan_rejects_unknown_constructor_fields(self) -> None:
        with self.assertRaises(ValidationError):
            BatchGenerationPlan(
                items=[],
                scenarios=[],
                settings=[],
                output_dir=Path("output"),
                dataset_provenance=DatasetProvenance(
                    dataset_path="dataset.jsonl",
                    dataset_sha256="sha256",
                    record_count=0,
                ),
                resmue=True,
            )

    def test_scenario_declares_when_pushbacks_are_shared(self) -> None:
        self.assertTrue(SCENARIOS[0].shares_pushbacks_across_target_models)
        self.assertFalse(SCENARIOS[1].shares_pushbacks_across_target_models)
        self.assertTrue(SCENARIOS[2].shares_pushbacks_across_target_models)
        self.assertFalse(SCENARIOS[3].shares_pushbacks_across_target_models)
        self.assertNotIn(
            "shares_pushbacks_across_target_models",
            SCENARIOS[0].model_dump(),
        )

    def test_default_generation_settings_are_multi_model(self) -> None:
        settings = GenerationSettings()

        self.assertEqual(
            (
                "oski/inferenz-gpt-oss-120b",
                "qwen3.5:27b",
                "openai/qwen3.8",
            ),
            DEFAULT_TARGET_MODELS,
        )
        self.assertEqual("openai/qwen3.8", DEFAULT_SIMULATOR_MODEL)
        self.assertEqual("deepseek-r1:32b", DEFAULT_VALIDATOR_MODEL)
        self.assertEqual(DEFAULT_SIMULATOR_MODEL, settings.simulator_model)
        self.assertEqual(DEFAULT_VALIDATOR_MODEL, settings.validator_model)
        self.assertEqual(512, settings.validator_max_tokens)
        self.assertEqual(8192, settings.target_max_tokens)
        self.assertEqual(50, settings.max_retries)
        self.assertIn("Respond naturally", settings.target_system_prompt)
        self.assertIn("at most 120 words", settings.target_system_prompt)
        self.assertIn("Never exceed 200 words", settings.target_system_prompt)
        self.assertNotIn("presupposition", settings.target_system_prompt)


if __name__ == "__main__":
    unittest.main()
