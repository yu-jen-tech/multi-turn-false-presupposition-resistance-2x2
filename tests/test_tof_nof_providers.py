from __future__ import annotations

import contextlib
import io
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from tof_nof_evaluation import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_PANEL,
    DEFAULT_OUTPUT_DIR,
    JUDGE_DECISION_JSON_SCHEMA,
    JudgeClientConfig,
    JudgeConfig,
    OpenAICompatibleChatClient,
    OpenAIResponsesChatClient,
    create_default_judge_panel,
    create_judge_client,
    default_judge_config,
    parse_args,
    resolve_judge_client_config,
)
from transcript_generation import ChatMessage


class OpenAIResponsesChatClientTests(unittest.TestCase):
    def test_sends_high_without_temperature_and_strict_decision_schema(self) -> None:
        response = Mock()
        response.id = "resp-1"
        response.output_text = (
            '{"decision_type":"rejects_or_corrects","reason":"Correction."}'
        )
        raw = {
            "id": "resp-1", "status": "completed",
            "usage": {"output_tokens_details": {"reasoning_tokens": 456}},
            "output": [{"type": "reasoning", "summary": []}],
        }
        response.model_dump.return_value = raw
        sdk_client = Mock()
        sdk_client.responses.create.return_value = response

        with patch("openai.OpenAI", return_value=sdk_client):
            client = OpenAIResponsesChatClient(
                api_key="openai-secret",
                request_timeout_seconds=30,
                api_retries=0,
            )
            result = client.complete(
                config=JudgeConfig(
                    model="gpt-6-sol", temperature=None, reasoning_effort="high"
                ),
                messages=[
                    ChatMessage(role="system", content="Judge the response."),
                    ChatMessage(role="user", content="Response to judge."),
                ],
            )

        sdk_client.responses.create.assert_called_once_with(
            model="gpt-6-sol",
            input=[
                {"role": "system", "content": "Judge the response."},
                {"role": "user", "content": "Response to judge."},
            ],
            reasoning={"effort": "high"},
            max_output_tokens=16384,
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
        self.assertEqual("resp-1", result.metadata["raw_response_id"])
        self.assertEqual(raw, result.raw_response)
        self.assertIsNone(result.metadata["temperature"])
        self.assertEqual("completed", result.metadata["finish_reason"])
        self.assertEqual(456, result.metadata["reasoning_tokens"])
        self.assertEqual(16384, result.metadata["requested_max_tokens"])

    def test_rejects_unsupported_or_nonreasoning_configs_before_requests(self) -> None:
        with patch("openai.OpenAI") as sdk:
            for definition in DEFAULT_JUDGE_PANEL:
                config = default_judge_config(definition)
                client = create_judge_client(
                    provider=definition.provider,
                    config=JudgeClientConfig("secret", None, 30, 0),
                )
                for changed in (
                    replace(config, model="unsupported-model"),
                    replace(config, reasoning_effort=None),
                    replace(config, reasoning_effort="xhigh"),
                    replace(config, temperature=0.0 if config.temperature is None else None),
                    replace(config, seed=42),
                    replace(config, allowed_openai_params=() if config.allowed_openai_params else ("reasoning_effort",)),
                ):
                    with self.subTest(provider=definition.provider, config=changed):
                        with self.assertRaises(ValueError):
                            client.complete(
                                config=changed,
                                messages=[ChatMessage(role="user", content="Judge this.")],
                            )
            sdk.return_value.responses.create.assert_not_called()
            sdk.return_value.chat.completions.create.assert_not_called()
            sdk.return_value.chat.completions.stream.assert_not_called()

    def test_provider_rejection_never_retries_without_reasoning(self) -> None:
        for definition in DEFAULT_JUDGE_PANEL:
            with self.subTest(model=definition.model):
                sdk = Mock()
                create = (
                    sdk.responses.create if definition.provider == "openai"
                    else sdk.chat.completions.stream
                )
                create.side_effect = RuntimeError("unsupported reasoning")
                with patch("openai.OpenAI", return_value=sdk), patch("tof_nof_evaluation.providers.time.sleep"):
                    client = create_judge_client(
                        provider=definition.provider,
                        config=JudgeClientConfig("secret", None, 30, 1),
                    )
                    with self.assertRaisesRegex(RuntimeError, "unsupported reasoning"):
                        client.complete(
                            config=default_judge_config(definition),
                            messages=[ChatMessage(role="user", content="Judge this.")],
                        )
                self.assertEqual(2, create.call_count)
                self.assertEqual(create.call_args_list[0], create.call_args_list[1])
                sent = create.call_args.kwargs
                if definition.provider == "openai":
                    self.assertEqual({"effort": "high"}, sent["reasoning"])
                    self.assertNotIn("temperature", sent)
                else:
                    self.assertEqual("high", sent["extra_body"]["reasoning_effort"])
                    self.assertEqual(["reasoning_effort"], sent["extra_body"]["allowed_openai_params"])


class OpenAICompatibleChatClientTests(unittest.TestCase):
    def test_judges_send_high_effort_and_gateway_allowlist(self) -> None:
        visible = '{"decision_type":"mixed","reason":"Accepts then corrects."}'
        for model, effort in (
            ("openai/qwen3.8", "high"),
            ("oski/inferenz-gpt-oss-120b", "high"),
            ("oski/inferenz-mistral-small-4-119b", "high"),
        ):
            with self.subTest(model=model):
                raw = {
                    "id": "chatcmpl-reasoning",
                    "choices": [{"finish_reason": "stop", "message": {
                        "content": visible,
                        "reasoning_content": "Separate reasoning; not a decision.",
                    }}],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 123}},
                }
                response = Mock()
                response.model_dump.return_value = raw
                response.choices = [Mock(message=Mock(content=visible), finish_reason="stop")]
                sdk_client = Mock()
                sdk_client.chat.completions.stream.return_value = contextlib.nullcontext(
                    Mock(get_final_completion=Mock(return_value=response))
                )
                with patch("openai.OpenAI", return_value=sdk_client):
                    client = create_judge_client(
                        provider="openai-compatible",
                        config=JudgeClientConfig("secret", "https://gateway.example/v1", 30, 0),
                    )
                    result = client.complete(
                        config=JudgeConfig(
                            model=model, reasoning_effort=effort,
                            allowed_openai_params=("reasoning_effort",),
                        ),
                        messages=[ChatMessage(role="user", content="Judge this.")],
                    )
                sdk_client.chat.completions.stream.assert_called_once_with(
                    model=model,
                    messages=[{"role": "user", "content": "Judge this."}],
                    temperature=0.0,
                    max_tokens=16384,
                    extra_body={
                        "reasoning_effort": effort,
                        "allowed_openai_params": ["reasoning_effort"],
                    },
                    stream_options={"include_usage": True},
                    timeout=30,
                )
                self.assertEqual(visible, result.content)
                self.assertEqual(raw, result.raw_response)
                self.assertEqual("stop", result.metadata["finish_reason"])
                self.assertEqual(123, result.metadata["reasoning_tokens"])

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
    def test_compatible_judge_client_defaults_to_ki4buw_endpoint(self) -> None:
        with patch("openai.OpenAI") as sdk:
            create_judge_client(
                provider="openai-compatible",
                config=JudgeClientConfig("secret", None, 30, 0),
            )
        self.assertEqual("https://llm.ki4buw.de/v1", sdk.call_args.kwargs["base_url"])

    def test_default_selection_has_three_supported_provider_model_pairs(self) -> None:
        self.assertEqual(
            [
                ("openai-compatible", "openai/qwen3.8"),
                ("openai", "gpt-6-sol"),
                ("openai-compatible", "oski/inferenz-gpt-oss-120b"),
            ],
            [(judge.provider, judge.model) for judge in DEFAULT_JUDGE_PANEL],
        )
        self.assertEqual(16384, DEFAULT_JUDGE_MAX_TOKENS)

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

    def test_panel_creates_one_client_per_selected_provider(self) -> None:
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
        self.assertIs(panel[1].client, openai_client)
        self.assertEqual([0.0, None, 0.0], [member.config.temperature for member in panel])
        self.assertEqual(["high", "high", "high"], [member.config.reasoning_effort for member in panel])
        self.assertEqual(
            [("reasoning_effort",), (), ("reasoning_effort",)],
            [member.config.allowed_openai_params for member in panel],
        )
        self.assertTrue(all(member.config.seed is None for member in panel))
        self.assertTrue(all(member.config.parse_retries == 2 for member in panel))
        self.assertIs(panel[2].client, compatible_client)

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

    def test_cli_defaults_to_independent_location_and_scoped_credentials(self) -> None:
        args = parse_args([])

        self.assertEqual(DEFAULT_OUTPUT_DIR, args.output_dir)
        self.assertTrue(str(args.output_dir).endswith("results-v3-independent"))
        self.assertEqual(16384, args.judge_max_tokens)
        self.assertIsNone(args.ki4buw_api_key)
        self.assertIsNone(args.openai_api_key)
        self.assertIsNone(args.item_id)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--judge-model", "other-model"])


if __name__ == "__main__":
    unittest.main()
