from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import BaseModel, ValidationError

from transcript_generation.models import (
    BatchGenerationPlan,
    CallPurpose,
    DatasetItem,
    DEFAULT_TARGET_MODELS,
    TARGET_RESPONSE_WORD_LIMIT,
    GenerationSettings,
    SCENARIOS,
    TRANSCRIPT_SCHEMA_VERSION,
)


class GenerationModelTests(unittest.TestCase):
    def test_new_schema_and_simulator_call_purposes_are_declared(self) -> None:
        self.assertEqual(
            "false-presupposition-transcript-v2",
            TRANSCRIPT_SCHEMA_VERSION,
        )
        self.assertEqual(150, TARGET_RESPONSE_WORD_LIMIT)
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

    def test_batch_generation_plan_is_a_frozen_pydantic_model(self) -> None:
        item = DatasetItem(
            id="item-1",
            question="Question?",
            presuppositions="False premise.",
            corrections="Correction.",
        )
        settings = GenerationSettings(target_model="model-a")
        plan = BatchGenerationPlan(
            items=[item],
            scenarios=[SCENARIOS[0]],
            settings=[settings],
            output_dir=Path("output"),
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
            presuppositions="False premise.",
            corrections="Correction.",
        )
        settings = GenerationSettings(target_model="model-a")

        plan = BatchGenerationPlan(
            items=(item,),
            scenarios=(SCENARIOS[0],),
            settings=(settings,),
            output_dir="output",
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
            ("oski/inferenz-gpt-oss-120b", "qwen3.5:27b"),
            DEFAULT_TARGET_MODELS,
        )
        self.assertEqual(512, settings.target_max_tokens)
        self.assertIn("Respond naturally", settings.target_system_prompt)
        self.assertIn("under 150 words", settings.target_system_prompt)
        self.assertNotIn("presupposition", settings.target_system_prompt)


if __name__ == "__main__":
    unittest.main()
