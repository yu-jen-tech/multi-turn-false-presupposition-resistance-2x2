from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


DEFAULT_TRANSCRIPT_DIR = Path("transcript-v3")
DEFAULT_OUTPUT_DIR = Path("results-v3")
DEFAULT_JUDGE_MODEL = "qwen3.5:27b"
DEFAULT_JUDGE_MAX_TOKENS = 256
DEFAULT_JUDGE_PARSE_RETRIES = 1

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
)
REASON_COLUMNS = (
    "id",
    "question",
    "presuppositions",
    "corrections",
    "target_model",
    "scenario",
    "turn",
    "label",
    "reason",
    "user_message",
    "assistant_response",
    "judge_output",
)

BinaryLabel = Literal[0, 1]
JudgeProvider = Literal["openai-compatible", "openai"]
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class JudgeConfig:
    model: str = DEFAULT_JUDGE_MODEL
    temperature: float = 0.0
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    seed: int | None = None
    parse_retries: int = DEFAULT_JUDGE_PARSE_RETRIES


@dataclass(frozen=True)
class JudgeDecision:
    label: BinaryLabel
    reason: str


@dataclass(frozen=True)
class TurnEvaluation:
    turn: int
    label: BinaryLabel
    reason: str
    user_message: str
    assistant_response: str
    judge_output: str
    judge_metadata: dict[str, Any]
    judge_raw_response: dict[str, Any]


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
    turn_evaluations: tuple[TurnEvaluation, ...]
