from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from tof_nof_evaluation import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_PANEL,
    DEFAULT_OUTPUT_DIR,
    JUDGE_DECISION_JSON_SCHEMA,
    JudgeClientConfig,
    OpenAICompatibleChatClient,
    OpenAIResponsesChatClient,
    create_default_judge_panel,
    parse_args,
    resolve_judge_client_config,
)
from transcript_generation import ChatMessage


class OpenAIResponsesChatClientTests(unittest.TestCase):
    def test_sends_zero_temperature_and_strict_decision_schema(self) -> None:
        response = Mock()
        response.id = "resp-1"
        response.output_text = (
            '{"decision_type":"rejects_or_corrects","reason":"Correction."}'
        )
        response.model_dump.return_value = {"id": "resp-1"}
        sdk_client = Mock()
        sdk_client.responses.create.return_value = response

        with patch("openai.OpenAI", return_value=sdk_client):
            client = OpenAIResponsesChatClient(
                api_key="openai-secret",
                request_timeout_seconds=30,
                api_retries=0,
            )
            result = client.complete(
                purpose="validator",
                model="gpt-5.6-sol",
                messages=[
                    ChatMessage(role="system", content="Judge the response."),
                    ChatMessage(role="user", content="Response to judge."),
                ],
                temperature=0.0,
                max_tokens=8192,
                seed=None,
            )

        sdk_client.responses.create.assert_called_once_with(
            model="gpt-5.6-sol",
            input=[
                {"role": "system", "content": "Judge the response."},
                {"role": "user", "content": "Response to judge."},
            ],
            temperature=0.0,
            max_output_tokens=8192,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "false_presupposition_judge_decision",
                    "strict": True,
                    "schema": JUDGE_DECISION_JSON_SCHEMA,
                }
            },
            store=False,
            timeout=30,
        )
        self.assertEqual(response.output_text, result.content)
        self.assertEqual("resp-1", result.metadata.raw_response_id)
        self.assertEqual({"id": "resp-1"}, result.raw_response)

    def test_rejects_nonzero_temperature_and_seed(self) -> None:
        with patch("openai.OpenAI"):
            client = OpenAIResponsesChatClient(
                api_key="openai-secret",
                api_retries=0,
            )

        for temperature, seed, message in (
            (0.1, None, "temperature"),
            (0.0, 42, "seed"),
        ):
            with self.subTest(temperature=temperature, seed=seed):
                with self.assertRaisesRegex(ValueError, message):
                    client.complete(
                        purpose="validator",
                        model="gpt-5.6-sol",
                        messages=[ChatMessage(role="user", content="Judge this.")],
                        temperature=temperature,
                        max_tokens=8192,
                        seed=seed,
                    )


class OpenAICompatibleChatClientTests(unittest.TestCase):
    def test_preserves_zero_temperature_for_compatible_providers(self) -> None:
        response = Mock()
        response.id = "chatcmpl-1"
        response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(
                    content=(
                        '{"decision_type":"rejects_or_corrects",'
                        '"reason":"Correction."}'
                    )
                ),
            )
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
                temperature=0.0,
                max_tokens=8192,
                seed=None,
            )

        sdk_client.chat.completions.create.assert_called_once_with(
            model="compatible-model",
            messages=[{"role": "user", "content": "Judge this."}],
            temperature=0.0,
            max_tokens=8192,
            timeout=30,
        )


class JudgePanelProviderTests(unittest.TestCase):
    def test_default_panel_has_exact_ordered_provider_model_pairs(self) -> None:
        self.assertEqual(
            [
                ("openai-compatible", "oski/inferenz-mistral-small-4-119b"),
                ("openai-compatible", "deepseek-r1:32b"),
                ("openai", "gpt-5.6-sol"),
            ],
            [(judge.provider, judge.model) for judge in DEFAULT_JUDGE_PANEL],
        )
        self.assertEqual(8192, DEFAULT_JUDGE_MAX_TOKENS)

    def test_panel_requires_both_provider_credentials_before_creating_clients(self) -> None:
        with patch("tof_nof_evaluation.providers.create_judge_client") as factory:
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                create_default_judge_panel(
                    environ={"KI4BUW_API_KEY": "compatible-secret"},
                    ki4buw_api_key=None,
                    ki4buw_base_url=None,
                    openai_api_key=None,
                    openai_base_url=None,
                    request_timeout_seconds=30,
                    api_retries=1,
                    max_tokens=8192,
                    parse_retries=1,
                )

        factory.assert_not_called()

    def test_panel_creates_one_client_per_provider_and_shares_ki4buw_client(self) -> None:
        compatible_client = Mock(name="compatible-client")
        openai_client = Mock(name="openai-client")
        with patch(
            "tof_nof_evaluation.providers.create_judge_client",
            side_effect=[compatible_client, openai_client],
        ) as factory:
            panel = create_default_judge_panel(
                environ={
                    "KI4BUW_API_KEY": "compatible-secret",
                    "OPENAI_API_KEY": "openai-secret",
                },
                ki4buw_api_key=None,
                ki4buw_base_url=None,
                openai_api_key=None,
                openai_base_url=None,
                request_timeout_seconds=30,
                api_retries=1,
                max_tokens=8192,
                parse_retries=2,
            )

        self.assertEqual(2, factory.call_count)
        self.assertIs(panel[0].client, compatible_client)
        self.assertIs(panel[1].client, compatible_client)
        self.assertIs(panel[2].client, openai_client)
        self.assertTrue(all(member.config.temperature == 0.0 for member in panel))
        self.assertTrue(all(member.config.seed is None for member in panel))
        self.assertTrue(all(member.config.parse_retries == 2 for member in panel))

    def test_credentials_and_endpoints_do_not_cross_provider_routes(self) -> None:
        with self.assertRaisesRegex(ValueError, "KI4BUW_API_KEY"):
            resolve_judge_client_config(
                provider="openai-compatible",
                api_key=None,
                base_url=None,
                environ={
                    "OPENAI_API_KEY": "openai-secret",
                    "JUDGE_API_KEY": "ambiguous-secret",
                },
                request_timeout_seconds=30,
                api_retries=1,
            )

        openai_config = resolve_judge_client_config(
            provider="openai",
            api_key=None,
            base_url=None,
            environ={
                "KI4BUW_API_KEY": "compatible-secret",
                "OPENAI_API_KEY": "openai-secret",
                "JUDGE_BASE_URL": "https://ambiguous.example/v1",
            },
            request_timeout_seconds=30,
            api_retries=1,
        )
        self.assertEqual("openai-secret", openai_config.api_key)
        self.assertIsNone(openai_config.base_url)

    def test_cli_defaults_to_new_consensus_location_and_scoped_credentials(self) -> None:
        args = parse_args([])

        self.assertEqual(DEFAULT_OUTPUT_DIR, args.output_dir)
        self.assertTrue(str(args.output_dir).endswith("results-v3-consensus"))
        self.assertEqual(8192, args.judge_max_tokens)
        self.assertIsNone(args.ki4buw_api_key)
        self.assertIsNone(args.openai_api_key)
        self.assertIsNone(args.item_id)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--judge-model", "other-model"])


if __name__ == "__main__":
    unittest.main()
