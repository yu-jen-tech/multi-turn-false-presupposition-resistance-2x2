from __future__ import annotations

import json

from .client import ChatClient
from .models import (
    BaselineRecord,
    ChatMessage,
    ChatResult,
    DatasetItem,
    DatasetProvenance,
    DialogueTurn,
    GenerationSettings,
    PushbackRecord,
    Scenario,
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
    build_target_messages,
)
from .validation import (
    parse_baseline_aware_semantic_validation,
    parse_pushback,
    parse_semantic_validation,
    parse_styled_semantic_validation,
    semantic_validation_reasons,
    validate_pushback,
)


class TranscriptGenerator:
    def __init__(
        self,
        *,
        client: ChatClient,
        settings: GenerationSettings,
        dataset_provenance: DatasetProvenance,
    ):
        self.client = client
        self.settings = settings
        self.dataset_provenance = dataset_provenance

    def _validator_model(self) -> str:
        return self.settings.validator_model or self.settings.simulator_model

    def _target_complete(self, *, messages: list[ChatMessage]) -> ChatResult:
        return self.client.complete(
            purpose="target",
            model=self.settings.target_model,
            messages=messages,
            temperature=self.settings.target_temperature,
            max_tokens=self.settings.target_max_tokens,
            seed=self.settings.seed,
        )

    def _validator_complete(self, *, messages: list[ChatMessage]) -> ChatResult:
        return self.client.complete(
            purpose="validator",
            model=self._validator_model(),
            messages=messages,
            temperature=self.settings.validator_temperature,
            max_tokens=self.settings.validator_max_tokens,
            seed=self.settings.seed,
        )

    def _first_turn_target_messages(self, *, item: DatasetItem) -> list[ChatMessage]:
        return build_target_messages(
            target_system_prompt=self.settings.target_system_prompt,
            transcript_so_far=[ChatMessage(role="user", content=item.question)],
        )

    def generate_first_response(self, *, item: DatasetItem) -> ChatResult:
        return self._target_complete(
            messages=self._first_turn_target_messages(item=item)
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
        if first_response is None:
            first_response = self._target_complete(messages=target_messages)

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
                )
                pushback = self._generate_valid_styled_pushback(
                    item=item,
                    scenario=scenario,
                    turn=turn,
                    baseline=baseline,
                    transcript_so_far=transcript_messages,
                    prior_pushbacks=[record.text for record in pushbacks],
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
            target_response = self._target_complete(messages=target_messages)
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
            turns=turns,
            pushbacks=pushbacks,
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

    def _generate_valid_baseline(
        self,
        *,
        item: DatasetItem,
        scenario: Scenario,
        turn: int,
        transcript_so_far: list[ChatMessage],
        prior_baselines: list[str],
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
            result.metadata.retry_count = retry_count
            baseline_text = parse_pushback(result.content)
            validation = validate_pushback(
                pushback=baseline_text,
                item=item,
                prior_pushbacks=prior_baselines,
            )
            semantic_result: ChatResult | None = None
            if validation.passed:
                semantic_messages = build_baseline_semantic_validator_messages(
                    item=item,
                    pushback=baseline_text,
                    strategy=strategy,
                    strategy_instruction=strategy_instruction,
                )
                semantic_result = self._validator_complete(messages=semantic_messages)
                semantic_result.metadata.retry_count = retry_count
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
            result.metadata.retry_count = retry_count
            pushback_text = parse_pushback(result.content)
            validation = validate_pushback(
                pushback=pushback_text,
                item=item,
                prior_pushbacks=prior_pushbacks,
            )
            semantic_result: ChatResult | None = None
            if validation.passed:
                semantic_messages = build_styled_semantic_validator_messages(
                    item=item,
                    pushback=pushback_text,
                    strategy=strategy,
                    strategy_instruction=strategy_instruction,
                    baseline_text=baseline.text,
                    style=scenario.style,
                )
                semantic_result = self._validator_complete(messages=semantic_messages)
                semantic_result.metadata.retry_count = retry_count
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
                    baseline=baseline,
                )

        assert last_result is not None
        raise RuntimeError(
            f"Failed to generate a valid styled pushback for {item.id} "
            f"{scenario.name} turn {turn}: {last_validation.reasons}"
        )
