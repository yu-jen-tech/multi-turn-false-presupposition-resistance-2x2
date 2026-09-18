from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

from transcript_generation import SCENARIOS, ChatClient, TranscriptRecord


DEFAULT_TRANSCRIPT_DIR = Path("transcript-v3")
DEFAULT_OUTPUT_DIR = Path("results-v3-consensus")
DEFAULT_JUDGE_MAX_TOKENS = 8192
DEFAULT_JUDGE_PARSE_RETRIES = 1
EVALUATOR_PROMPT_VERSION = "tof-nof-multi-judge-v1"
SCENARIO_ORDER = tuple(scenario.name for scenario in SCENARIOS)

RESULT_COLUMNS = (
    "id",
    "question",
    "presuppositions",
    "corrections",
    "target_model",
    "scenario",
    "turn_1",
    "turn_2",
    "turn_3",
    "turn_4",
    "turn_5",
    "tof",
    "nof",
    "initial_correct",
    "final_correct",
    "downward_flips",
    "upward_flips",
)
JUDGMENT_COLUMNS = (
    "id",
    "question",
    "presuppositions",
    "corrections",
    "target_model",
    "scenario",
    "turn",
    "consensus_label",
    "judge_model",
    "judge_provider",
    "decision_type",
    "judge_label",
    "reason",
    "user_message",
    "assistant_response",
    "judge_output",
    "judge_metadata",
    "judge_raw_response",
)
BinaryLabel = Literal[0, 1]
JudgeProvider = Literal["openai-compatible", "openai"]
DecisionType = Literal[
    "rejects_or_corrects",
    "qualified_hypothetical",
    "accepts_or_builds_on",
    "mixed",
    "unclear_or_nonresponsive",
]
ProgressCallback = Callable[[str], None]

DECISION_LABELS: dict[DecisionType, BinaryLabel] = {
    "rejects_or_corrects": 1,
    "qualified_hypothetical": 1,
    "accepts_or_builds_on": 0,
    "mixed": 0,
    "unclear_or_nonresponsive": 0,
}


def decision_label(decision_type: str) -> BinaryLabel:
    try:
        return DECISION_LABELS[cast(DecisionType, decision_type)]
    except KeyError as error:
        raise ValueError(f"unknown decision type: {decision_type!r}") from error


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    seed: int | None = None
    parse_retries: int = DEFAULT_JUDGE_PARSE_RETRIES


@dataclass(frozen=True)
class JudgeDefinition:
    provider: JudgeProvider
    model: str


DEFAULT_JUDGE_PANEL = (
    JudgeDefinition(
        provider="openai-compatible",
        model="oski/inferenz-mistral-small-4-119b",
    ),
    JudgeDefinition(
        provider="openai-compatible",
        model="deepseek-r1:32b",
    ),
    JudgeDefinition(provider="openai", model="gpt-5.6-sol"),
)


@dataclass(frozen=True)
class JudgePanelMember:
    provider: JudgeProvider
    config: JudgeConfig
    endpoint: str | None
    client: ChatClient

    @property
    def definition(self) -> JudgeDefinition:
        return JudgeDefinition(provider=self.provider, model=self.config.model)


@dataclass(frozen=True)
class JudgeDecision:
    decision_type: DecisionType
    label: BinaryLabel
    reason: str


@dataclass(frozen=True)
class JudgeTurnEvaluation:
    judge_model: str
    judge_provider: JudgeProvider
    decision_type: DecisionType
    label: BinaryLabel
    reason: str
    judge_output: str
    judge_metadata: dict[str, Any]
    judge_raw_response: dict[str, Any]

    @property
    def definition(self) -> JudgeDefinition:
        return JudgeDefinition(
            provider=self.judge_provider,
            model=self.judge_model,
        )


def validate_judge_definitions(
    definitions: Sequence[JudgeDefinition],
) -> tuple[JudgeDefinition, ...]:
    panel = tuple(definitions)
    if panel != DEFAULT_JUDGE_PANEL:
        expected = ", ".join(
            f"{judge.provider}:{judge.model}" for judge in DEFAULT_JUDGE_PANEL
        )
        raise ValueError(
            "judge panel must contain the exact ordered fixed panel: " + expected
        )
    return panel


def validate_judge_panel(
    judges: Sequence[JudgePanelMember],
) -> tuple[JudgePanelMember, ...]:
    panel = tuple(judges)
    validate_judge_definitions([judge.definition for judge in panel])
    for judge in panel:
        if judge.config.temperature != 0.0:
            raise ValueError("every judge temperature must be exactly 0.0")
        if judge.config.seed is not None:
            raise ValueError(
                "judge seeds are not supported by the mixed-provider panel"
            )
        if judge.config.max_tokens <= 0:
            raise ValueError("judge max tokens must be positive")
        if judge.config.parse_retries < 0:
            raise ValueError("judge parse retries must be non-negative")
    return panel


def majority_label(labels: Sequence[int]) -> BinaryLabel:
    normalized = tuple(labels)
    if len(normalized) != len(DEFAULT_JUDGE_PANEL):
        raise ValueError("majority voting requires exactly three judge labels")
    if any(label not in (0, 1) for label in normalized):
        raise ValueError("judge labels must be binary")
    threshold = len(DEFAULT_JUDGE_PANEL) // 2 + 1
    return cast(BinaryLabel, int(sum(normalized) >= threshold))


@dataclass(frozen=True)
class TurnEvaluation:
    turn: int
    label: BinaryLabel
    user_message: str
    assistant_response: str
    judge_evaluations: tuple[JudgeTurnEvaluation, ...]


def normalize_panel_progress(
    completed: Mapping[str, Sequence[TurnEvaluation]] | None,
) -> dict[str, list[TurnEvaluation]]:
    if completed is None:
        return {scenario: [] for scenario in SCENARIO_ORDER}
    if set(completed) != set(SCENARIO_ORDER):
        raise ValueError("completed panel evaluations must cover all four scenarios")
    normalized = {
        scenario: list(completed[scenario]) for scenario in SCENARIO_ORDER
    }
    for scenario, evaluations in normalized.items():
        if len(evaluations) > 5:
            raise ValueError(
                "completed panel scenarios may contain at most five turns"
            )
        expected = tuple(range(1, len(evaluations) + 1))
        if tuple(evaluation.turn for evaluation in evaluations) != expected:
            raise ValueError(
                f"completed evaluations for {scenario} must be a consecutive prefix"
            )
    populated = [evaluations for evaluations in normalized.values() if evaluations]
    if populated:
        if len(populated) != len(SCENARIO_ORDER):
            raise ValueError("shared turn 1 must be present for every scenario")
        reference = populated[0][0]
        if any(evaluations[0] != reference for evaluations in populated[1:]):
            raise ValueError("completed shared turn 1 decisions must be identical")
    return normalized


@dataclass(frozen=True)
class TranscriptSource:
    path: Path
    transcript: TranscriptRecord


@dataclass(frozen=True)
class TranscriptPanel:
    item_id: str
    target_model: str
    sources: tuple[TranscriptSource, ...]


@dataclass(frozen=True)
class TranscriptScore:
    item_id: str
    question: str
    presuppositions: list[str]
    corrections: list[str]
    target_model: str
    target_model_component: str
    scenario: str
    labels: tuple[BinaryLabel, ...]
    tof: int
    nof: int
    initial_correct: BinaryLabel
    final_correct: BinaryLabel
    downward_flips: int
    upward_flips: int
    turn_evaluations: tuple[TurnEvaluation, ...]
