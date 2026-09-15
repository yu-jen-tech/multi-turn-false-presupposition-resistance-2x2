from __future__ import annotations

import json

from .client import ChatClient, extract_finish_reason, extract_reasoning_tokens
from .models import (
    BaselineRecord,
    ChatMessage,
    ChatResult,
    DatasetItem,
    DatasetProvenance,
    DialogueTurn,
    GenerationSettings,
    MAX_TARGET_MAX_TOKENS,
    PushbackRecord,
    Scenario,
    TARGET_REASONING_ENABLED,
    TARGET_RESPONSE_HARD_WORD_LIMIT,
    TARGET_RESPONSE_SOFT_WORD_LIMIT,
    TranscriptRecord,
    ValidationResult,
    utc_now,
)
from .prompts import (
    TURN_STRATEGIES,
    build_baseline_messages,
    build_baseline_semantic_validator_messages,
    build_styled_semantic_validator_messages,
    build_style_messages,
    build_surface_style_validator_messages,
    build_target_messages,
)
from .trace import TraceCallback, emit_model_call, emit_validation
from .validation import (
    PushbackParseError,
    parse_baseline_aware_semantic_validation,
    parse_pushback,
    parse_semantic_validation,
    parse_styled_semantic_validation,
    parse_surface_style_validation,
    semantic_validation_reasons,
    surface_style_validation_reasons,
    target_response_rejection_reasons,
    validate_pushback,
)


class TranscriptGenerator:
    def __init__(
        self,
        *,
        client: ChatClient,
        settings: GenerationSettings,
        dataset_provenance: DatasetProvenance,
        trace: TraceCallback | None = None,
    ):
        self.client = client
        self.settings = settings
        self.dataset_provenance = dataset_provenance
        self.trace = trace

    def _validator_model(self) -> str:
        return self.settings.validator_model

    def _target_complete(
        self,
        *,
        item: DatasetItem,
        scenario: str,
        turn: int,
        messages: list[ChatMessage],
        trace_scope: str,
    ) -> ChatResult:
        requested_max_tokens = self.settings.target_max_tokens
        effective_max_tokens = min(
            max(requested_max_tokens, 1),
            MAX_TARGET_MAX_TOKENS,
        )
        last_result: ChatResult | None = None
        last_reasons: list[str] = ["not_generated"]

        for retry_count in range(self.settings.max_retries + 1):
            result = self.client.complete(
                purpose="target",
                model=self.settings.target_model,
                messages=messages,
                temperature=self.settings.target_temperature,
                max_tokens=effective_max_tokens,
                seed=self.settings.seed,
            )
            provider_retry_count = result.metadata.retry_count
            result.metadata.retry_count = retry_count
            result.content = result.content.strip()
            finish_reason = self._target_finish_reason(result)
            self._annotate_target_metadata(
                result,
                finish_reason=finish_reason,
                requested_max_tokens=requested_max_tokens,
                effective_max_tokens=effective_max_tokens,
            )
            emit_model_call(
                self.trace,
                label=self._target_label(trace_scope),
                operation="target_response",
                purpose="target",
                model=self.settings.target_model,
                item=item,
                scenario=scenario,
                turn=turn,
                attempt=retry_count + 1,
                messages=messages,
                content=result.content,
                metadata=result.metadata,
                provider_retry_count=provider_retry_count,
                generation_retry_count=retry_count,
            )

            reasons = target_response_rejection_reasons(
                content=result.content,
                finish_reason=finish_reason,
            )
            last_result = result
            last_reasons = reasons
            if not reasons:
                return result

            if finish_reason == "length":
                effective_max_tokens = min(
                    effective_max_tokens * 2,
                    MAX_TARGET_MAX_TOKENS,
                )

        assert last_result is not None
        raise RuntimeError(
            f"Failed to generate a valid target response for {item.id} "
            f"{scenario} turn {turn}: {last_reasons}"
        )

    @staticmethod
    def _target_finish_reason(result: ChatResult) -> str | None:
        if result.metadata.finish_reason is not None:
            return result.metadata.finish_reason
        return extract_finish_reason(result.raw_response)

    @classmethod
    def _annotate_target_metadata(
        cls,
        result: ChatResult,
        *,
        finish_reason: str | None,
        requested_max_tokens: int,
        effective_max_tokens: int,
    ) -> None:
        result.metadata.finish_reason = finish_reason
        result.metadata.requested_max_tokens = requested_max_tokens
        result.metadata.effective_max_tokens = effective_max_tokens
        result.metadata.visible_word_count = len(result.content.split())
        result.metadata.visible_word_target = TARGET_RESPONSE_SOFT_WORD_LIMIT
        result.metadata.visible_word_limit = TARGET_RESPONSE_HARD_WORD_LIMIT
        result.metadata.reasoning_enabled = TARGET_REASONING_ENABLED
        if result.metadata.reasoning_tokens is None:
            result.metadata.reasoning_tokens = extract_reasoning_tokens(
                result.raw_response
            )

    def _validator_complete(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
        turn: int,
        attempt: int,
        messages: list[ChatMessage],
        operation: str,
    ) -> ChatResult:
        result = self.client.complete(
            purpose="validator",
            model=self._validator_model(),
            messages=messages,
            temperature=self.settings.validator_temperature,
            max_tokens=self.settings.validator_max_tokens,
            seed=self.settings.seed,
        )
        provider_retry_count = result.metadata.retry_count
        result.metadata.retry_count = attempt - 1
        emit_model_call(
            self.trace,
            label=f"[validator={self._validator_model()}]",
            operation=operation,
            purpose="validator",
            model=self._validator_model(),
            item=item,
            scenario=scenario.name,
            turn=turn,
            attempt=attempt,
            messages=messages,
            content=result.content,
            metadata=result.metadata,
            provider_retry_count=provider_retry_count,
            generation_retry_count=attempt - 1,
        )
        return result

    def _first_turn_target_messages(self, *, item: DatasetItem) -> list[ChatMessage]:
        return build_target_messages(
            target_system_prompt=self.settings.target_system_prompt,
            transcript_so_far=[ChatMessage(role="user", content=item.question)],
        )

    def generate_first_response(self, *, item: DatasetItem) -> ChatResult:
        return self._target_complete(
            item=item,
            scenario="shared",
            turn=1,
            messages=self._first_turn_target_messages(item=item),
            trace_scope="shared",
        )

    def generate_without_history_baselines(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
    ) -> list[BaselineRecord]:
        if not scenario.shares_pushbacks_across_target_models:
            raise ValueError(
                "Shared baseline pre-generation only applies to "
                "without_simulator_history scenarios"
            )

        baselines: list[BaselineRecord] = []
        transcript_messages = [ChatMessage(role="user", content=item.question)]
        for turn in sorted(TURN_STRATEGIES):
            baseline = self._generate_valid_baseline(
                item=item,
                scenario=scenario,
                turn=turn,
                transcript_so_far=transcript_messages,
                prior_baselines=[record.text for record in baselines],
                trace_scope="shared",
            )
            baselines.append(baseline)
            transcript_messages.append(ChatMessage(role="user", content=baseline.text))
        return baselines

    def generate_without_history_pushbacks(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
        fixed_baselines: list[BaselineRecord] | None = None,
    ) -> list[PushbackRecord]:
        if not scenario.shares_pushbacks_across_target_models:
            raise ValueError(
                "Shared pushback pre-generation only applies to "
                "without_simulator_history scenarios"
            )

        expected_turns = sorted(TURN_STRATEGIES)
        baselines = (
            self.generate_without_history_baselines(
                item=item,
                scenario=scenario,
            )
            if fixed_baselines is None
            else fixed_baselines
        )
        self._validate_baseline_turns(baselines, expected_turns)

        pushbacks: list[PushbackRecord] = []
        transcript_messages = [ChatMessage(role="user", content=item.question)]
        for index, turn in enumerate(expected_turns):
            pushback = self._generate_valid_styled_pushback(
                item=item,
                scenario=scenario,
                turn=turn,
                baseline=baselines[index],
                transcript_so_far=transcript_messages,
                prior_pushbacks=[record.text for record in pushbacks],
                trace_scope="shared",
            )
            pushbacks.append(pushback)
            transcript_messages.append(ChatMessage(role="user", content=pushback.text))
        return pushbacks

    def generate(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
        first_response: ChatResult | None = None,
        fixed_pushbacks: list[PushbackRecord] | None = None,
    ) -> TranscriptRecord:
        expected_pushback_turns = sorted(TURN_STRATEGIES)
        if fixed_pushbacks is not None:
            fixed_turns = [pushback.turn for pushback in fixed_pushbacks]
            if fixed_turns != expected_pushback_turns:
                raise ValueError(
                    "fixed_pushbacks must contain turns "
                    f"{expected_pushback_turns}; got {fixed_turns}"
                )

        transcript_messages = [ChatMessage(role="user", content=item.question)]
        target_messages = self._first_turn_target_messages(item=item)
        if first_response is None or self._target_response_is_invalid(first_response):
            first_response = self._target_complete(
                item=item,
                scenario=scenario.name,
                turn=1,
                messages=target_messages,
                trace_scope="target",
            )
        else:
            first_response.content = first_response.content.strip()
            self._annotate_target_metadata(
                first_response,
                finish_reason=self._target_finish_reason(first_response),
                requested_max_tokens=(
                    first_response.metadata.requested_max_tokens
                    or self.settings.target_max_tokens
                ),
                effective_max_tokens=(
                    first_response.metadata.effective_max_tokens
                    or self.settings.target_max_tokens
                ),
            )

        turns = [
            DialogueTurn(
                turn=1,
                user=item.question,
                user_source="dataset",
                assistant=first_response.content,
                target_messages=target_messages,
                api_metadata=first_response.metadata,
                raw_response=first_response.raw_response,
            )
        ]
        pushbacks: list[PushbackRecord] = []
        transcript_messages.append(
            ChatMessage(role="assistant", content=first_response.content)
        )

        for turn in expected_pushback_turns:
            if fixed_pushbacks is None:
                baseline = self._generate_valid_baseline(
                    item=item,
                    scenario=scenario,
                    turn=turn,
                    transcript_so_far=transcript_messages,
                    prior_baselines=[
                        record.baseline.text
                        for record in pushbacks
                        if record.baseline is not None
                    ],
                    trace_scope="target",
                )
                pushback = self._generate_valid_styled_pushback(
                    item=item,
                    scenario=scenario,
                    turn=turn,
                    baseline=baseline,
                    transcript_so_far=transcript_messages,
                    prior_pushbacks=[record.text for record in pushbacks],
                    trace_scope="target",
                )
            else:
                pushback = fixed_pushbacks[
                    turn - expected_pushback_turns[0]
                ]
            pushbacks.append(pushback)
            transcript_messages.append(ChatMessage(role="user", content=pushback.text))

            target_messages = build_target_messages(
                target_system_prompt=self.settings.target_system_prompt,
                transcript_so_far=transcript_messages,
            )
            target_response = self._target_complete(
                item=item,
                scenario=scenario.name,
                turn=turn,
                messages=target_messages,
                trace_scope="target",
            )
            turns.append(
                DialogueTurn(
                    turn=turn,
                    user=pushback.text,
                    user_source="simulator",
                    strategy=pushback.strategy,
                    assistant=target_response.content,
                    target_messages=target_messages,
                    api_metadata=target_response.metadata,
                    raw_response=target_response.raw_response,
                )
            )
            transcript_messages.append(
                ChatMessage(role="assistant", content=target_response.content)
            )

        return TranscriptRecord(
            generated_at=utc_now(),
            dataset_provenance=self.dataset_provenance,
            item=item,
            scenario=scenario,
            simulator_model=self.settings.simulator_model,
            target_model=self.settings.target_model,
            validator_model=self._validator_model(),
            target_system_prompt=self.settings.target_system_prompt,
            target_response_soft_word_limit=TARGET_RESPONSE_SOFT_WORD_LIMIT,
            target_response_hard_word_limit=TARGET_RESPONSE_HARD_WORD_LIMIT,
            target_reasoning_enabled=TARGET_REASONING_ENABLED,
            turns=turns,
            pushbacks=pushbacks,
        )

    @staticmethod
    def _target_response_is_invalid(result: ChatResult) -> bool:
        finish_reason = TranscriptGenerator._target_finish_reason(result)
        return bool(
            target_response_rejection_reasons(
                content=result.content,
                finish_reason=finish_reason,
            )
        )

    @staticmethod
    def _validate_baseline_turns(
        baselines: list[BaselineRecord],
        expected_turns: list[int],
    ) -> None:
        baseline_turns = [baseline.turn for baseline in baselines]
        if baseline_turns != expected_turns:
            raise ValueError(
                "fixed_baselines must contain turns "
                f"{expected_turns}; got {baseline_turns}"
            )
        for baseline, turn in zip(baselines, expected_turns):
            strategy, strategy_instruction = TURN_STRATEGIES[turn]
            if (
                baseline.strategy != strategy
                or baseline.strategy_instruction != strategy_instruction
            ):
                raise ValueError(
                    "fixed_baselines contain a strategy mismatch for turn "
                    f"{turn}"
                )

    @staticmethod
    def _parse_and_validate_pushback(
        *,
        raw: str,
        item: DatasetItem,
        prior_pushbacks: list[str],
    ) -> tuple[str, str, ValidationResult]:
        candidate_for_trace = raw
        try:
            pushback_text = parse_pushback(raw)
        except PushbackParseError:
            return (
                "",
                candidate_for_trace,
                ValidationResult(
                    passed=False,
                    reasons=["invalid_json_object"],
                ),
            )

        return (
            pushback_text,
            pushback_text,
            validate_pushback(
                pushback=pushback_text,
                item=item,
                prior_pushbacks=prior_pushbacks,
            ),
        )

    def _generate_valid_baseline(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
        turn: int,
        transcript_so_far: list[ChatMessage],
        prior_baselines: list[str],
        trace_scope: str,
    ) -> BaselineRecord:
        strategy, strategy_instruction = TURN_STRATEGIES[turn]
        last_result: ChatResult | None = None
        last_validation = ValidationResult(
            passed=False,
            reasons=["not_generated"],
        )

        for retry_count in range(self.settings.max_retries + 1):
            messages = build_baseline_messages(
                item=item,
                scenario=scenario,
                turn=turn,
                transcript_so_far=transcript_so_far,
            )
            result = self.client.complete(
                purpose="simulator_baseline",
                model=self.settings.simulator_model,
                messages=messages,
                temperature=self.settings.simulator_temperature,
                max_tokens=self.settings.simulator_max_tokens,
                seed=self.settings.seed,
            )
            provider_retry_count = result.metadata.retry_count
            result.metadata.retry_count = retry_count
            emit_model_call(
                self.trace,
                label=self._simulator_label(trace_scope),
                operation="simulator_baseline",
                purpose="simulator_baseline",
                model=self.settings.simulator_model,
                item=item,
                scenario=scenario.name,
                turn=turn,
                attempt=retry_count + 1,
                messages=messages,
                content=result.content,
                metadata=result.metadata,
                provider_retry_count=provider_retry_count,
                generation_retry_count=retry_count,
            )
            (
                baseline_text,
                candidate_for_trace,
                lexical_validation,
            ) = self._parse_and_validate_pushback(
                raw=result.content,
                item=item,
                prior_pushbacks=prior_baselines,
            )
            validation = lexical_validation
            semantic_result: ChatResult | None = None
            semantic_was_run = False
            if lexical_validation.passed:
                semantic_messages = build_baseline_semantic_validator_messages(
                    item=item,
                    pushback=baseline_text,
                    strategy=strategy,
                    strategy_instruction=strategy_instruction,
                )
                semantic_was_run = True
                semantic_result = self._validator_complete(
                    item=item,
                    scenario=scenario,
                    turn=turn,
                    attempt=retry_count + 1,
                    messages=semantic_messages,
                    operation="baseline_semantic_validator",
                )
                try:
                    semantic = parse_semantic_validation(semantic_result.content)
                except (ValueError, json.JSONDecodeError) as error:
                    validation = ValidationResult(
                        passed=False,
                        reasons=[f"semantic_validator_parse_error:{error}"],
                    )
                else:
                    semantic_reasons = semantic_validation_reasons(semantic)
                    validation = ValidationResult(
                        passed=not semantic_reasons,
                        reasons=semantic_reasons,
                        semantic=semantic,
                    )

            emit_validation(
                self.trace,
                label=self._simulator_label(trace_scope),
                operation="baseline_validation",
                item=item,
                scenario=scenario.name,
                turn=turn,
                attempt=retry_count + 1,
                candidate=candidate_for_trace,
                lexical=lexical_validation,
                validation=validation,
                semantic_was_run=semantic_was_run,
            )

            last_result = result
            last_validation = validation
            if validation.passed:
                return BaselineRecord(
                    turn=turn,
                    strategy=strategy,
                    strategy_instruction=strategy_instruction,
                    text=baseline_text,
                    simulator_messages=messages,
                    validation=validation,
                    api_metadata=result.metadata,
                    raw_response=result.raw_response,
                    semantic_validation_api_metadata=(
                        semantic_result.metadata
                        if semantic_result is not None
                        else None
                    ),
                    semantic_validation_raw_response=(
                        semantic_result.raw_response
                        if semantic_result is not None
                        else {}
                    ),
                )

        assert last_result is not None
        raise RuntimeError(
            f"Failed to generate a valid baseline for {item.id} "
            f"{scenario.name} turn {turn}: {last_validation.reasons}"
        )

    def _generate_valid_styled_pushback(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
        turn: int,
        baseline: BaselineRecord,
        transcript_so_far: list[ChatMessage],
        prior_pushbacks: list[str],
        trace_scope: str,
    ) -> PushbackRecord:
        strategy, strategy_instruction = TURN_STRATEGIES[turn]
        last_result: ChatResult | None = None
        last_validation = ValidationResult(
            passed=False,
            reasons=["not_generated"],
        )

        for retry_count in range(self.settings.max_retries + 1):
            messages = build_style_messages(
                item=item,
                scenario=scenario,
                turn=turn,
                baseline_text=baseline.text,
                transcript_so_far=transcript_so_far,
            )
            result = self.client.complete(
                purpose="simulator_style",
                model=self.settings.simulator_model,
                messages=messages,
                temperature=self.settings.simulator_temperature,
                max_tokens=self.settings.simulator_max_tokens,
                seed=self.settings.seed,
            )
            provider_retry_count = result.metadata.retry_count
            result.metadata.retry_count = retry_count
            emit_model_call(
                self.trace,
                label=self._simulator_label(trace_scope),
                operation="simulator_style",
                purpose="simulator_style",
                model=self.settings.simulator_model,
                item=item,
                scenario=scenario.name,
                turn=turn,
                attempt=retry_count + 1,
                messages=messages,
                content=result.content,
                metadata=result.metadata,
                provider_retry_count=provider_retry_count,
                generation_retry_count=retry_count,
            )
            (
                pushback_text,
                candidate_for_trace,
                lexical_validation,
            ) = self._parse_and_validate_pushback(
                raw=result.content,
                item=item,
                prior_pushbacks=prior_pushbacks,
            )
            validation = lexical_validation
            semantic_result: ChatResult | None = None
            semantic_was_run = False
            surface_style_result: ChatResult | None = None
            surface_style_was_run = False
            if lexical_validation.passed:
                semantic_messages = build_styled_semantic_validator_messages(
                    item=item,
                    pushback=pushback_text,
                    strategy=strategy,
                    strategy_instruction=strategy_instruction,
                    baseline_text=baseline.text,
                    style=scenario.style,
                )
                semantic_was_run = True
                semantic_result = self._validator_complete(
                    item=item,
                    scenario=scenario,
                    turn=turn,
                    attempt=retry_count + 1,
                    messages=semantic_messages,
                    operation="styled_semantic_validator",
                )
                try:
                    semantic = parse_styled_semantic_validation(
                        semantic_result.content
                    )
                except (ValueError, json.JSONDecodeError) as error:
                    validation = ValidationResult(
                        passed=False,
                        reasons=[f"semantic_validator_parse_error:{error}"],
                    )
                else:
                    semantic_reasons = semantic_validation_reasons(semantic)
                    validation = ValidationResult(
                        passed=not semantic_reasons,
                        reasons=semantic_reasons,
                        semantic=semantic,
                    )
                    if not semantic_reasons:
                        surface_style_messages = build_surface_style_validator_messages(
                            item=item,
                            pushback=pushback_text,
                            baseline_text=baseline.text,
                            strategy=strategy,
                            strategy_instruction=strategy_instruction,
                            style=scenario.style,
                        )
                        surface_style_was_run = True
                        surface_style_result = self._validator_complete(
                            item=item,
                            scenario=scenario,
                            turn=turn,
                            attempt=retry_count + 1,
                            messages=surface_style_messages,
                            operation="surface_style_validator",
                        )
                        try:
                            surface_style = parse_surface_style_validation(
                                surface_style_result.content
                            )
                        except (ValueError, json.JSONDecodeError) as error:
                            validation = ValidationResult(
                                passed=False,
                                reasons=[
                                    f"surface_style_validator_parse_error:{error}"
                                ],
                                semantic=semantic,
                            )
                        else:
                            surface_style_reasons = surface_style_validation_reasons(
                                surface_style
                            )
                            validation = ValidationResult(
                                passed=not surface_style_reasons,
                                reasons=surface_style_reasons,
                                semantic=semantic,
                                surface_style=surface_style,
                            )

            emit_validation(
                self.trace,
                label=self._simulator_label(trace_scope),
                operation="styled_validation",
                item=item,
                scenario=scenario.name,
                turn=turn,
                attempt=retry_count + 1,
                candidate=candidate_for_trace,
                lexical=lexical_validation,
                validation=validation,
                semantic_was_run=semantic_was_run,
                surface_style_was_run=surface_style_was_run,
            )

            last_result = result
            last_validation = validation
            if validation.passed:
                return PushbackRecord(
                    turn=turn,
                    strategy=strategy,
                    strategy_instruction=strategy_instruction,
                    text=pushback_text,
                    simulator_messages=messages,
                    validation=validation,
                    api_metadata=result.metadata,
                    raw_response=result.raw_response,
                    semantic_validation_api_metadata=(
                        semantic_result.metadata
                        if semantic_result is not None
                        else None
                    ),
                    semantic_validation_raw_response=(
                        semantic_result.raw_response
                        if semantic_result is not None
                        else {}
                    ),
                    surface_style_validation_api_metadata=(
                        surface_style_result.metadata
                        if surface_style_result is not None
                        else None
                    ),
                    surface_style_validation_raw_response=(
                        surface_style_result.raw_response
                        if surface_style_result is not None
                        else {}
                    ),
                    baseline=baseline,
                )

        assert last_result is not None
        raise RuntimeError(
            f"Failed to generate a valid styled pushback for {item.id} "
            f"{scenario.name} turn {turn}: {last_validation.reasons}"
        )

    def _target_label(self, trace_scope: str) -> str:
        if trace_scope == "shared":
            return f"[shared][target={self.settings.target_model}]"
        return f"[target={self.settings.target_model}]"

    def _simulator_label(self, trace_scope: str) -> str:
        if trace_scope == "shared":
            return f"[shared][simulator={self.settings.simulator_model}]"
        return (
            f"[simulator={self.settings.simulator_model}]"
            f"[target-branch={self.settings.target_model}]"
        )
