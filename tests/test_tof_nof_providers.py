from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tof_nof_evaluation import (
    JudgeClientConfig,
    OpenAICompatibleChatClient,
    OpenAIResponsesChatClient,
    create_judge_client,
    main,
    resolve_judge_client_config,
)
from transcript_generation import ChatMessage


class OpenAIResponsesChatClientTests(unittest.TestCase):
    def test_omits_temperature_for_the_official_responses_api(self) -> None:
        response = Mock()
        response.id = "resp-1"
        response.output_text = '{"label": 1, "reason": "It corrects the claim."}'
        response.model_dump.return_value = {"id": "resp-1"}
        sdk_client = Mock()

        def create_response(**request: object) -> Mock:
            if "temperature" in request:
                raise RuntimeError(
                    "Unsupported parameter: 'temperature' is not supported "
                    "with this model."
                )
            return response

        sdk_client.responses.create.side_effect = create_response

        with patch("openai.OpenAI", return_value=sdk_client):
            client = OpenAIResponsesChatClient(
                api_key="openai-secret",
                request_timeout_seconds=30,
                api_retries=0,
            )
            result = client.complete(
                purpose="validator",
                model="gpt-test",
                messages=[
                    ChatMessage(role="system", content="Judge the response."),
                    ChatMessage(role="user", content="Response to judge."),
                ],
                temperature=0.0,
                max_tokens=256,
                seed=None,
            )

        sdk_client.responses.create.assert_called_once_with(
            model="gpt-test",
            input=[
                {"role": "system", "content": "Judge the response."},
                {"role": "user", "content": "Response to judge."},
            ],
            max_output_tokens=256,
            store=False,
            timeout=30,
        )
        self.assertEqual(response.output_text, result.content)
        self.assertEqual("resp-1", result.metadata.raw_response_id)
        self.assertEqual({"id": "resp-1"}, result.raw_response)

    def test_rejects_seed_instead_of_silently_ignoring_it(self) -> None:
        with patch("openai.OpenAI"):
            client = OpenAIResponsesChatClient(
                api_key="openai-secret",
                api_retries=0,
            )

        with self.assertRaisesRegex(ValueError, "does not support seed"):
            client.complete(
                purpose="validator",
                model="gpt-test",
                messages=[ChatMessage(role="user", content="Judge this.")],
                temperature=0.0,
                max_tokens=256,
                seed=42,
            )


class OpenAICompatibleChatClientTests(unittest.TestCase):
    def test_preserves_temperature_for_compatible_providers(self) -> None:
        response = Mock()
        response.id = "chatcmpl-1"
        response.choices = [
            Mock(message=Mock(content='{"label": 1, "reason": "Correction."}'))
        ]
        response.model_dump.return_value = {"id": "chatcmpl-1"}
        sdk_client = Mock()
        sdk_client.chat.completions.create.return_value = response

        with patch("openai.OpenAI", return_value=sdk_client):
            client = OpenAICompatibleChatClient(
                api_key="compatible-secret",
                request_timeout_seconds=30,
                api_retries=0,
            )
            client.complete(
                purpose="validator",
                model="compatible-model",
                messages=[ChatMessage(role="user", content="Judge this.")],
                temperature=0.25,
                max_tokens=256,
                seed=None,
            )

        sdk_client.chat.completions.create.assert_called_once_with(
            model="compatible-model",
            messages=[{"role": "user", "content": "Judge this."}],
            temperature=0.25,
            max_tokens=256,
            timeout=30,
        )


class JudgeProviderSelectionTests(unittest.TestCase):
    def test_client_factory_rejects_an_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported judge provider"):
            create_judge_client(
                provider="unknown",
                config=JudgeClientConfig(
                    api_key="secret",
                    base_url=None,
                    request_timeout_seconds=120,
                    api_retries=1,
                ),
            )

    def test_compatible_provider_uses_ki4buw_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_dir = root / "transcripts"
            transcript_dir.mkdir()

            with (
                patch("tof_nof_evaluation.cli.load_dotenv_if_present"),
                patch.dict(
                    "os.environ",
                    {
                        "KI4BUW_API_KEY": "compatible-secret",
                        "KI4BUW_BASE_URL": "https://compatible.example/v1",
                    },
                    clear=True,
                ),
                patch(
                    "tof_nof_evaluation.cli.create_judge_client"
                ) as client_factory,
            ):
                main(
                    [
                        "--transcript-dir",
                        str(transcript_dir),
                        "--output-dir",
                        str(root / "results"),
                    ]
                )

        client_factory.assert_called_once_with(
            provider="openai-compatible",
            config=JudgeClientConfig(
                api_key="compatible-secret",
                base_url="https://compatible.example/v1",
                request_timeout_seconds=120,
                api_retries=1,
            ),
        )

    def test_cli_selects_the_official_openai_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_dir = root / "transcripts"
            transcript_dir.mkdir()

            with (
                patch("tof_nof_evaluation.cli.load_dotenv_if_present"),
                patch.dict(
                    "os.environ",
                    {
                        "OPENAI_API_KEY": "openai-secret",
                    },
                    clear=True,
                ),
                patch(
                    "tof_nof_evaluation.cli.create_judge_client"
                ) as client_factory,
            ):
                main(
                    [
                        "--transcript-dir",
                        str(transcript_dir),
                        "--output-dir",
                        str(root / "results"),
                        "--judge-provider",
                        "openai",
                        "--judge-model",
                        "gpt-test",
                    ]
                )

        client_factory.assert_called_once_with(
            provider="openai",
            config=JudgeClientConfig(
                api_key="openai-secret",
                base_url=None,
                request_timeout_seconds=120,
                api_retries=1,
            ),
        )

    def test_provider_credentials_are_not_shared_between_transports(self) -> None:
        with self.assertRaisesRegex(ValueError, "KI4BUW_API_KEY"):
            resolve_judge_client_config(
                provider="openai-compatible",
                api_key=None,
                base_url=None,
                environ={"OPENAI_API_KEY": "openai-secret"},
                request_timeout_seconds=120,
                api_retries=1,
            )

        config = resolve_judge_client_config(
            provider="openai",
            api_key=None,
            base_url=None,
            environ={
                "KI4BUW_API_KEY": "compatible-secret",
                "KI4BUW_BASE_URL": "https://compatible.example/v1",
                "OPENAI_API_KEY": "openai-secret",
            },
            request_timeout_seconds=120,
            api_retries=1,
        )
        self.assertEqual("openai-secret", config.api_key)
        self.assertIsNone(config.base_url)

    def test_legacy_judge_environment_overrides_are_preserved(self) -> None:
        config = resolve_judge_client_config(
            provider="openai-compatible",
            api_key=None,
            base_url=None,
            environ={
                "JUDGE_API_KEY": "legacy-secret",
                "JUDGE_BASE_URL": "https://legacy.example/v1",
            },
            request_timeout_seconds=120,
            api_retries=1,
        )
        self.assertEqual("legacy-secret", config.api_key)
        self.assertEqual("https://legacy.example/v1", config.base_url)


if __name__ == "__main__":
    unittest.main()
