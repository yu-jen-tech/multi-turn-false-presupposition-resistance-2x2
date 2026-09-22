from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, cast

from transcript_generation import SCENARIOS, ChatMessage, TranscriptRecord


DEFAULT_TRANSCRIPT_DIR = Path("transcript-v3")
DEFAULT_OUTPUT_DIR = Path("results-v3-consensus-rubric-v2")
DEFAULT_JUDGE_MAX_TOKENS = 8192
DEFAULT_JUDGE_PARSE_RETRIES = 1
EVALUATOR_PROMPT_VERSION = "tof-nof-multi-judge-v2"
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
    temperature: float | None = 0.0
    reasoning_effort: Literal["high", "xhigh"] | None = None
    allowed_openai_params: tuple[str, ...] = ()
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    seed: int | None = None
    parse_retries: int = DEFAULT_JUDGE_PARSE_RETRIES


@dataclass(frozen=True)
class JudgeResult:
    content: str
    metadata: dict[str, Any]
    raw_response: dict[str, Any]


class JudgeClient(Protocol):
    def complete(
        self, *, config: JudgeConfig, messages: list[ChatMessage]
    ) -> JudgeResult:
        ...


@dataclass(frozen=True)
class JudgeDefinition:
    provider: JudgeProvider
    model: str


DEFAULT_JUDGE_PANEL = (
    JudgeDefinition(provider="openai", model="gpt-5.4-mini"),
    JudgeDefinition(provider="openai-compatible", model="openai/qwen3.8"),
    JudgeDefinition(
        provider="openai-compatible",
        model="oski/inferenz-mistral-small-4-119b",
    ),
)


def default_judge_config(definition: JudgeDefinition) -> JudgeConfig:
    """Return the fixed reasoning and sampling policy for one panel member."""
    if definition not in DEFAULT_JUDGE_PANEL:
        raise ValueError(f"Unsupported judge definition: {definition!r}")
    if definition.provider == "openai":
        return JudgeConfig(
            model=definition.model, temperature=None, reasoning_effort="xhigh"
        )
    return JudgeConfig(
        model=definition.model,
        reasoning_effort=("xhigh" if definition.model == "openai/qwen3.8" else "high"),
        allowed_openai_params=("reasoning_effort",),
    )


def validate_judge_config(provider: JudgeProvider, config: JudgeConfig) -> None:
    expected = default_judge_config(JudgeDefinition(provider, config.model))
    if config != replace(
        expected, max_tokens=config.max_tokens, parse_retries=config.parse_retries
    ):
        raise ValueError(
            "judge configuration must match the fixed reasoning, temperature, "
            "seed, and allowed_openai_params policy"
        )
    if config.max_tokens <= 0:
        raise ValueError("judge max tokens must be positive")
    if config.parse_retries < 0:
        raise ValueError("judge parse retries must be non-negative")


@dataclass(frozen=True)
class JudgePanelMember:
    provider: JudgeProvider
    config: JudgeConfig
    endpoint: str | None
    client: JudgeClient

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
        validate_judge_config(judge.provider, judge.config)
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
