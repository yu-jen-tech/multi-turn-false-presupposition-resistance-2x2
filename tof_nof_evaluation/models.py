from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

from transcript_generation import ChatClient, TranscriptRecord


DEFAULT_TRANSCRIPT_DIR = Path("transcript-v3")
DEFAULT_OUTPUT_DIR = Path("results-v3-consensus")
DEFAULT_JUDGE_MAX_TOKENS = 8192
DEFAULT_JUDGE_PARSE_RETRIES = 1
EVALUATOR_PROMPT_VERSION = "tof-nof-multi-judge-v1"

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


@dataclass(frozen=True)
class TurnEvaluation:
    turn: int
    label: BinaryLabel
    user_message: str
    assistant_response: str
    judge_evaluations: tuple[JudgeTurnEvaluation, ...]


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
