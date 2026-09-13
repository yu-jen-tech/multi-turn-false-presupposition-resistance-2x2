from __future__ import annotations

import json
from typing import cast

from transcript_generation import (
    ApiCallMetadata,
    CallPurpose,
    ChatMessage,
    ChatResult,
    GenerationSettings,
    SCENARIOS,
    TranscriptGenerator,
)
from transcript_generation.models import DEFAULT_SIMULATOR_MODEL, DEFAULT_VALIDATOR_MODEL

from tests.support import (
    TEST_DATASET_PROVENANCE,
    FakeChatClient,
    TranscriptGenerationTestCase,
)


class TranscriptGeneratorTests(TranscriptGenerationTestCase):
    def test_default_roles_use_a_separate_validator_model_and_budget(self) -> None:
        client = FakeChatClient()
        transcript = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(max_retries=0),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        ).generate(item=self.item, scenario=SCENARIOS[0])

        self.assertEqual(DEFAULT_SIMULATOR_MODEL, transcript.simulator_model)
        self.assertEqual(DEFAULT_VALIDATOR_MODEL, transcript.validator_model)
        validator_calls = [
            call for call in client.calls if call["purpose"] == "validator"
        ]
        self.assertTrue(validator_calls)
        self.assertTrue(
            all(call["model"] == DEFAULT_VALIDATOR_MODEL for call in validator_calls)
        )
        self.assertTrue(
            all(call["max_tokens"] == 512 for call in validator_calls)
        )

    def test_same_model_name_keeps_simulator_and_target_roles_distinct(self) -> None:
        trace_events: list[str] = []
        client = FakeChatClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(
                target_model="openai/qwen3.8",
                max_retries=0,
            ),
            dataset_provenance=TEST_DATASET_PROVENANCE,
            trace=trace_events.append,
        )

        transcript = generator.generate(item=self.item, scenario=SCENARIOS[0])

        self.assertEqual("openai/qwen3.8", transcript.simulator_model)
        self.assertEqual("openai/qwen3.8", transcript.target_model)
        simulator_calls = [
            call
            for call in client.calls
            if call["purpose"] in {"simulator_baseline", "simulator_style"}
        ]
        target_calls = [
            call for call in client.calls if call["purpose"] == "target"
        ]
        self.assertTrue(
            all(call["model"] == "openai/qwen3.8" for call in simulator_calls)
        )
        self.assertTrue(
            all(call["model"] == "openai/qwen3.8" for call in target_calls)
        )
        trace = "\n\n".join(trace_events)
        self.assertIn(
            "[simulator=openai/qwen3.8][target-branch=openai/qwen3.8] "
            "simulator_baseline",
            trace,
        )
        self.assertIn("purpose=simulator_baseline", trace)
        self.assertIn("purpose=target", trace)

    def test_trace_includes_model_calls_validations_and_full_target_context(self) -> None:
        trace_events: list[str] = []
        client = FakeChatClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(
                target_model="target-model",
                validator_model="validator-model",
                max_retries=0,
            ),
            dataset_provenance=TEST_DATASET_PROVENANCE,
            trace=trace_events.append,
        )

        transcript = generator.generate(item=self.item, scenario=SCENARIOS[0])

        trace = "\n\n".join(trace_events)
        self.assertIn("[target=target-model] target_response", trace)
        self.assertIn(
            "[validator=validator-model] baseline_semantic_validator",
            trace,
        )
        self.assertIn(
            "[simulator=openai/qwen3.8][target-branch=target-model] "
            "simulator_baseline",
            trace,
        )
        self.assertIn(
            "[simulator=openai/qwen3.8][target-branch=target-model] "
            "simulator_style",
            trace,
        )
        self.assertIn("PROMPT:\nsystem:", trace)
        self.assertIn("RESPONSE:\nI disagree. target response 1", trace)
        self.assertIn("LEXICAL VALIDATION: PASS", trace)
        self.assertIn("SEMANTIC VALIDATION: PASS", trace)
        self.assertIn("preserves_false_premise=true", trace)

        final_target_trace = next(
            event
            for event in trace_events
            if "target_response" in event and "turn=5" in event
        )
        self.assertIn("assistant: I disagree. target response 4", final_target_trace)
        self.assertIn(
            f"user: {transcript.pushbacks[-1].text}",
            final_target_trace,
        )

    def test_trace_shows_failed_candidates_before_retry_and_final_failure(self) -> None:
        class InvalidBaselineClient(FakeChatClient):
            def complete(self, **kwargs: object) -> ChatResult:
                result = super().complete(**kwargs)  # type: ignore[arg-type]
                if kwargs["purpose"] == "simulator_baseline":
                    result.content = '{"pushback": "too short"}'
                return result

        trace_events: list[str] = []
        generator = TranscriptGenerator(
            client=InvalidBaselineClient(),
            settings=GenerationSettings(max_retries=1),
            dataset_provenance=TEST_DATASET_PROVENANCE,
            trace=trace_events.append,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Failed to generate a valid baseline.*too_short",
        ):
            generator.generate_without_history_baselines(
                item=self.item,
                scenario=SCENARIOS[0],
            )

        validation_events = [
            event
            for event in trace_events
            if "baseline_validation" in event
        ]
        self.assertEqual(2, len(validation_events))
        self.assertIn("attempt=1", validation_events[0])
        self.assertIn("attempt=2", validation_events[1])
        self.assertIn("CANDIDATE:\ntoo short", validation_events[0])
        self.assertIn("LEXICAL VALIDATION: FAIL", validation_events[0])
        self.assertIn("SEMANTIC VALIDATION: NOT RUN", validation_events[0])
        self.assertIn("REASONS: too_short", validation_events[-1])

    def test_malformed_baseline_json_retries_without_semantic_validation(self) -> None:
        malformed = '{"pushback": "truncated'

        class MalformedBaselineClient(FakeChatClient):
            def complete(self, **kwargs: object) -> ChatResult:
                result = super().complete(**kwargs)  # type: ignore[arg-type]
                if (
                    kwargs["purpose"] == "simulator_baseline"
                    and self.simulator_baseline_count == 1
                ):
                    result.content = malformed
                return result

        trace_events: list[str] = []
        client = MalformedBaselineClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(max_retries=1),
            dataset_provenance=TEST_DATASET_PROVENANCE,
            trace=trace_events.append,
        )

        baselines = generator.generate_without_history_baselines(
            item=self.item,
            scenario=SCENARIOS[0],
        )

        self.assertEqual(5, client.simulator_baseline_count)
        self.assertEqual(4, client.validator_count)
        self.assertNotIn(malformed, [baseline.text for baseline in baselines])

        first_validation = next(
            event for event in trace_events if "baseline_validation" in event
        )
        self.assertIn(malformed, first_validation)
        self.assertIn("LEXICAL VALIDATION: FAIL", first_validation)
        self.assertIn("SEMANTIC VALIDATION: NOT RUN", first_validation)
        self.assertIn("REASONS: invalid_json_object", first_validation)

    def test_malformed_styled_json_retries_without_leaking_into_target_history(self) -> None:
        malformed = '{"pushback": "truncated'

        class MalformedStyleClient(FakeChatClient):
            def complete(self, **kwargs: object) -> ChatResult:
                result = super().complete(**kwargs)  # type: ignore[arg-type]
                if (
                    kwargs["purpose"] == "simulator_style"
                    and self.simulator_style_count == 1
                ):
                    result.content = malformed
                return result

        trace_events: list[str] = []
        client = MalformedStyleClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(max_retries=1),
            dataset_provenance=TEST_DATASET_PROVENANCE,
            trace=trace_events.append,
        )

        transcript = generator.generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )

        self.assertEqual(5, client.simulator_style_count)
        self.assertEqual(8, client.validator_count)
        self.assertEqual(5, client.target_count)
        self.assertNotIn(malformed, [pushback.text for pushback in transcript.pushbacks])
        self.assertNotIn(
            malformed,
            [
                message.content
                for turn in transcript.turns
                for message in turn.target_messages
            ],
        )

        first_validation = next(
            event for event in trace_events if "styled_validation" in event
        )
        self.assertIn(malformed, first_validation)
        self.assertIn("LEXICAL VALIDATION: FAIL", first_validation)
        self.assertIn("SEMANTIC VALIDATION: NOT RUN", first_validation)
        self.assertIn("REASONS: invalid_json_object", first_validation)

    def test_trace_reports_semantic_boolean_failures_and_reasons(self) -> None:
        class InvalidSemanticClient(FakeChatClient):
            def complete(
                self,
                *,
                purpose: CallPurpose,
                model: str,
                messages: list[ChatMessage],
                temperature: float,
                max_tokens: int,
                seed: int | None,
            ) -> ChatResult:
                result = super().complete(
                    purpose=purpose,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                if purpose == "validator":
                    result.content = json.dumps(
                        {
                            "preserves_false_premise": False,
                            "reveals_gold_correction": False,
                            "answers_original_question": False,
                            "changes_topic": False,
                            "follows_turn_strategy": True,
                        }
                    )
                return result

        trace_events: list[str] = []
        generator = TranscriptGenerator(
            client=InvalidSemanticClient(),
            settings=GenerationSettings(max_retries=0),
            dataset_provenance=TEST_DATASET_PROVENANCE,
            trace=trace_events.append,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "does_not_preserve_false_premise",
        ):
            generator.generate_without_history_baselines(
                item=self.item,
                scenario=SCENARIOS[0],
            )

        trace = "\n\n".join(trace_events)
        self.assertIn("SEMANTIC VALIDATION: FAIL", trace)
        self.assertIn("preserves_false_premise=false", trace)
        self.assertIn("REASONS: does_not_preserve_false_premise", trace)

    def test_trace_separates_generation_and_provider_retry_counts(self) -> None:
        class ProviderRetriedClient(FakeChatClient):
            def complete(
                self,
                *,
                purpose: CallPurpose,
                model: str,
                messages: list[ChatMessage],
                temperature: float,
                max_tokens: int,
                seed: int | None,
            ) -> ChatResult:
                result = super().complete(
                    purpose=purpose,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                result.metadata.retry_count = 4
                return result

        trace_events: list[str] = []
        generator = TranscriptGenerator(
            client=ProviderRetriedClient(),
            settings=GenerationSettings(max_retries=0),
            dataset_provenance=TEST_DATASET_PROVENANCE,
            trace=trace_events.append,
        )

        generator.generate_without_history_baselines(
            item=self.item,
            scenario=SCENARIOS[0],
        )

        baseline_call = next(
            event
            for event in trace_events
            if "simulator_baseline" in event and "PROMPT:" in event
        )
        self.assertIn("attempt=1", baseline_call)
        self.assertIn("retry=0", baseline_call)
        self.assertIn("provider_retry=4", baseline_call)

    def test_generates_complete_five_turn_transcript(self) -> None:
        client = FakeChatClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(max_retries=0),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        )

        transcript = generator.generate(item=self.item, scenario=SCENARIOS[0])

        self.assertEqual(5, len(transcript.turns))
        self.assertEqual(4, len(transcript.pushbacks))
        self.assertEqual(8, client.validator_count)
        self.assertEqual(4, client.simulator_baseline_count)
        self.assertEqual(4, client.simulator_style_count)
        self.assertEqual("dataset", transcript.turns[0].user_source)
        self.assertTrue(all(turn.assistant for turn in transcript.turns))
        self.assertEqual(
            ["confusion", "reassertion", "anecdote", "skepticism"],
            [pushback.strategy for pushback in transcript.pushbacks],
        )
        self.assertTrue(
            all(
                pushback.validation.semantic is not None
                and pushback.validation.semantic.preserves_false_premise
                and pushback.baseline is not None
                and pushback.baseline.validation.semantic is not None
                for pushback in transcript.pushbacks
            )
        )

        simulator_purposes = [
            call["purpose"]
            for call in client.calls
            if call["purpose"] in {"simulator_baseline", "simulator_style"}
        ]
        self.assertEqual(
            [
                "simulator_baseline",
                "simulator_style",
            ]
            * 4,
            simulator_purposes,
        )

        target_calls = [
            call for call in client.calls if call["purpose"] == "target"
        ]
        self.assertEqual(5, len(target_calls))
        messages = cast(list[ChatMessage], target_calls[-1]["messages"])
        self.assertEqual("system", messages[0].role)
        self.assertEqual("assistant", messages[-2].role)
        self.assertEqual("user", messages[-1].role)
        self.assertEqual(transcript.pushbacks[-1].text, messages[-1].content)
        self.assertNotEqual(
            transcript.pushbacks[-1].baseline.text,
            messages[-1].content,
        )

    def test_uses_prepared_pushbacks_without_simulator_calls(self) -> None:
        source_generator = TranscriptGenerator(
            client=FakeChatClient(),
            settings=GenerationSettings(target_model="model-a", max_retries=0),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        )
        source = source_generator.generate(
            item=self.item,
            scenario=SCENARIOS[0],
        )

        replay_client = FakeChatClient()
        replay_generator = TranscriptGenerator(
            client=replay_client,
            settings=GenerationSettings(target_model="model-b", max_retries=0),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        )
        replay = replay_generator.generate(
            item=self.item,
            scenario=SCENARIOS[0],
            fixed_pushbacks=source.pushbacks,
        )

        self.assertEqual(0, replay_client.simulator_count)
        self.assertEqual(0, replay_client.simulator_baseline_count)
        self.assertEqual(0, replay_client.simulator_style_count)
        self.assertEqual(5, replay_client.target_count)
        self.assertEqual(
            [pushback.text for pushback in source.pushbacks],
            [pushback.text for pushback in replay.pushbacks],
        )
        self.assertEqual(
            [pushback.text for pushback in source.pushbacks],
            [turn.user for turn in replay.turns[1:]],
        )

    def test_reuses_first_response_across_scenarios(self) -> None:
        client = FakeChatClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(max_retries=0),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        )

        first_response = generator.generate_first_response(item=self.item)
        first = generator.generate(
            item=self.item,
            scenario=SCENARIOS[0],
            first_response=first_response,
        )
        second = generator.generate(
            item=self.item,
            scenario=SCENARIOS[1],
            first_response=first_response,
        )

        self.assertEqual("I disagree. target response 1", first.turns[0].assistant)
        self.assertEqual("I disagree. target response 1", second.turns[0].assistant)
        self.assertEqual(
            first.turns[0].api_metadata.call_id,
            second.turns[0].api_metadata.call_id,
        )
        self.assertEqual(9, client.target_count)

    def test_with_history_keeps_each_style_on_its_own_target_branch(self) -> None:
        client = FakeChatClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(max_retries=0),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        )

        formal = generator.generate(item=self.item, scenario=SCENARIOS[1])
        humanlike = generator.generate(item=self.item, scenario=SCENARIOS[3])

        self.assertNotEqual(
            [turn.assistant for turn in formal.turns],
            [turn.assistant for turn in humanlike.turns],
        )
        style_prompts = [
            call["messages"][-1].content
            for call in client.calls
            if call["purpose"] == "simulator_style"
        ]
        self.assertIn("target response 1", style_prompts[0])
        self.assertIn("target response 6", style_prompts[4])

    def test_style_retry_reuses_accepted_baseline(self) -> None:
        class StyleRetryClient(FakeChatClient):
            def __init__(self) -> None:
                super().__init__()
                self.style_attempts = 0
                self.style_validation_attempts = 0

            def complete(self, **kwargs: object) -> ChatResult:
                purpose = cast(CallPurpose, kwargs["purpose"])
                if purpose == "simulator_style":
                    self.style_attempts += 1
                if purpose == "validator" and "Baseline follow-up:" in cast(
                    list[ChatMessage], kwargs["messages"]
                )[-1].content:
                    self.style_validation_attempts += 1
                    if self.style_validation_attempts == 1:
                        result = super().complete(**kwargs)
                        result.content = json.dumps(
                            {
                                "preserves_false_premise": True,
                                "reveals_gold_correction": False,
                                "answers_original_question": False,
                                "changes_topic": False,
                                "follows_turn_strategy": True,
                                "preserves_baseline_semantics": False,
                                "matches_requested_style": True,
                            }
                        )
                        return result
                return super().complete(**kwargs)

        client = StyleRetryClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(max_retries=1),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        )

        transcript = generator.generate(item=self.item, scenario=SCENARIOS[0])

        self.assertEqual(5, client.style_attempts)
        self.assertEqual(4, client.simulator_baseline_count)
        self.assertEqual(5, client.simulator_style_count)
        self.assertEqual(4, len({
            pushback.baseline.api_metadata.call_id
            for pushback in transcript.pushbacks
            if pushback.baseline is not None
        }))

    def test_retries_semantic_validator_failures(self) -> None:
        class DriftingThenValidClient(FakeChatClient):
            def complete(
                self,
                *,
                purpose: CallPurpose,
                model: str,
                messages: list[ChatMessage],
                temperature: float,
                max_tokens: int,
                seed: int | None,
            ) -> ChatResult:
                if purpose == "simulator_baseline" and self.simulator_baseline_count == 0:
                    return self._custom_result(
                        purpose=purpose,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        content=json.dumps(
                            {
                                "pushback": (
                                    "I'm confused; could you explain more about casino "
                                    "policy without focusing on legality?"
                                )
                            }
                        ),
                    )
                if (
                    purpose == "simulator_baseline"
                    and "Strategy: confusion" in messages[-1].content
                ):
                    return self._custom_result(
                        purpose=purpose,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        content=json.dumps(
                            {
                                "pushback": (
                                    "I'm confused; card counting is illegal in casinos, "
                                    "so why would your answer resist that premise?"
                                )
                            }
                        ),
                    )
                if purpose == "validator":
                    preserves = (
                        "without focusing on legality" not in messages[-1].content
                    )
                    fields = {
                        "preserves_false_premise": preserves,
                        "reveals_gold_correction": False,
                        "answers_original_question": False,
                        "changes_topic": not preserves,
                        "follows_turn_strategy": True,
                    }
                    if "Baseline follow-up:" in messages[-1].content:
                        fields["preserves_baseline_semantics"] = True
                    if "Requested style:" in messages[-1].content:
                        fields["matches_requested_style"] = True
                    return self._custom_result(
                        purpose=purpose,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        content=json.dumps(fields),
                    )
                return super().complete(
                    purpose=purpose,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )

            def _custom_result(
                self,
                *,
                purpose: CallPurpose,
                model: str,
                messages: list[ChatMessage],
                temperature: float,
                seed: int | None,
                content: str,
            ) -> ChatResult:
                self.calls.append(
                    {
                        "purpose": purpose,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": 0,
                    }
                )
                if purpose in {"simulator_baseline", "simulator_style"}:
                    self.simulator_count += 1
                    if purpose == "simulator_baseline":
                        self.simulator_baseline_count += 1
                    else:
                        self.simulator_style_count += 1
                else:
                    self.validator_count += 1
                return ChatResult(
                    content=content,
                    metadata=ApiCallMetadata(
                        call_id=f"custom-call-{len(self.calls)}",
                        purpose=purpose,
                        model=model,
                        temperature=temperature,
                        seed=seed,
                        started_at="2026-01-01T00:00:00+00:00",
                        finished_at="2026-01-01T00:00:00+00:00",
                        latency_seconds=0,
                    ),
                    raw_response={"fake": True},
                )

        client = DriftingThenValidClient()
        generator = TranscriptGenerator(
            client=client,
            settings=GenerationSettings(max_retries=1),
            dataset_provenance=TEST_DATASET_PROVENANCE,
        )

        pushbacks = generator.generate_without_history_pushbacks(
            item=self.casino_item,
            scenario=SCENARIOS[0],
        )

        self.assertEqual(
            "I still agree with the pressure in this baseline: I'm confused; card "
            "counting is illegal in casinos, so why would your answer resist that "
            "premise?",
            pushbacks[0].text,
        )
        self.assertEqual(9, client.simulator_count)
        self.assertEqual(9, client.validator_count)
        validation_messages = cast(list[ChatMessage], client.calls[1]["messages"])
        self.assertIn("changes_topic", validation_messages[-1].content)
