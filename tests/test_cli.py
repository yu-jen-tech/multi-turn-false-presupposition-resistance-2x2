from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from transcript_generation.cli import main, parse_args
from transcript_generation.models import (
    DEFAULT_DATASET_JSONL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SIMULATOR_MODEL,
    DEFAULT_TARGET_MAX_TOKENS,
    DEFAULT_TARGET_MODELS,
    DatasetProvenance,
)


class TranscriptGenerationCliTests(unittest.TestCase):
    def test_main_uses_ki4buw_environment_for_compatible_client(self) -> None:
        with (
            patch("transcript_generation.cli.load_dotenv_if_present"),
            patch.dict(
                "os.environ",
                {
                    "KI4BUW_API_KEY": "compatible-secret",
                    "KI4BUW_BASE_URL": "https://compatible.example/v1",
                },
                clear=True,
            ),
            patch(
                "transcript_generation.cli.load_dataset",
                return_value=(
                    [],
                    DatasetProvenance(
                        dataset_path="dataset.jsonl",
                        dataset_sha256="sha256",
                        record_count=0,
                    ),
                ),
            ),
            patch("transcript_generation.cli.select_items", return_value=[]),
            patch("transcript_generation.cli.OpenAICompatibleChatClient") as client,
            patch("transcript_generation.cli.generate_transcripts", return_value=[]),
        ):
            main([])

        client.assert_called_once_with(
            api_key="compatible-secret",
            base_url="https://compatible.example/v1",
            request_timeout_seconds=120,
            api_retries=1,
        )

    def test_package_module_exposes_existing_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "transcript_generation", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn(
            "Generate 5-turn false-presupposition transcripts",
            result.stdout,
        )
        for option in (
            "--dataset-jsonl",
            "--filtered-csv",
            "--output-dir",
            "--item-index",
            "--num-questions",
            "--scenario",
            "--simulator-model",
            "--validator-model",
            "--target-model",
            "--simulator-temperature",
            "--validator-temperature",
            "--target-temperature",
            "--simulator-max-tokens",
            "--validator-max-tokens",
            "--target-max-tokens",
            "--request-timeout-seconds",
            "--api-retries",
            "--seed",
            "--max-retries",
            "--api-key",
            "--base-url",
            "--resume",
            "--overwrite",
        ):
            with self.subTest(option=option):
                self.assertIn(option, result.stdout)

    def test_parse_args_preserves_all_existing_defaults(self) -> None:
        args = parse_args([])

        expected_defaults = {
            "dataset_jsonl": DEFAULT_DATASET_JSONL,
            "output_dir": DEFAULT_OUTPUT_DIR,
            "item_index": 0,
            "num_questions": None,
            "scenario": None,
            "simulator_model": DEFAULT_SIMULATOR_MODEL,
            "validator_model": None,
            "target_models": list(DEFAULT_TARGET_MODELS),
            "simulator_temperature": 0.7,
            "validator_temperature": 0.0,
            "target_temperature": 0.0,
            "simulator_max_tokens": 180,
            "validator_max_tokens": 180,
            "target_max_tokens": DEFAULT_TARGET_MAX_TOKENS,
            "request_timeout_seconds": 120,
            "api_retries": 1,
            "seed": None,
            "max_retries": 2,
            "api_key": None,
            "base_url": None,
            "overwrite": False,
            "resume": False,
        }
        self.assertEqual(expected_defaults, vars(args))

    def test_parse_args_accepts_existing_repeatable_model_and_validator_flags(self) -> None:
        args = parse_args([])

        self.assertEqual(list(DEFAULT_TARGET_MODELS), args.target_models)
        self.assertIsNone(args.scenario)
        self.assertIsNone(args.validator_model)
        self.assertEqual(0.0, args.validator_temperature)
        self.assertEqual(180, args.validator_max_tokens)

        custom = parse_args(
            ["--target-model", "model-a", "--target-model", "model-b"]
        )
        self.assertEqual(["model-a", "model-b"], custom.target_models)

        validator = parse_args(
            [
                "--validator-model",
                "validator-model",
                "--validator-temperature",
                "0.2",
                "--validator-max-tokens",
                "64",
            ]
        )
        self.assertEqual("validator-model", validator.validator_model)
        self.assertEqual(0.2, validator.validator_temperature)
        self.assertEqual(64, validator.validator_max_tokens)

    def test_parse_args_rejects_unrequested_num_items_alias(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--num-items", "2"])

    def test_parse_args_rejects_removed_fixed_script_replay_scenario(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--scenario", "fixed_script_replay"])


if __name__ == "__main__":
    unittest.main()
