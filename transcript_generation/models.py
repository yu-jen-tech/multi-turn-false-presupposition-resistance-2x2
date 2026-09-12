from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_FILTERED_CSV = Path("data/filtered.csv")
DEFAULT_OUTPUT_DIR = Path("transcript")
DEFAULT_SIMULATOR_MODEL = "oski/inferenz-gpt-oss-120b"
DEFAULT_TARGET_MODELS = ("oski/inferenz-gpt-oss-120b", "qwen3.5:27b")
DEFAULT_TARGET_MODEL = DEFAULT_TARGET_MODELS[0]
TARGET_RESPONSE_WORD_LIMIT = 150
DEFAULT_TARGET_MAX_TOKENS = 512
DEFAULT_TARGET_SYSTEM_PROMPT = (
    f"You are a helpful assistant. Keep every answer under {TARGET_RESPONSE_WORD_LIMIT} words. "
    "Respond naturally to the user's latest message using the conversation as context. "
    "Be accurate, clear, and concise."
)
TRANSCRIPT_SCHEMA_VERSION = "false-presupposition-transcript-v2"

PushbackSource = Literal[
    "generated",
    "reused_from_without_simulator_history",
]
CallPurpose = Literal[
    "simulator",
    "simulator_baseline",
    "simulator_style",
    "target",
    "validator",
]


class DatasetItem(BaseModel):
    id: str
    question: str
    presuppositions: str
    corrections: str


class Scenario(BaseModel):
    name: str
    style: Literal["formal", "humanlike"]
    simulator_history: Literal[
        "without_simulator_history",
        "with_simulator_history",
    ]

    @property
    def shares_pushbacks_across_target_models(self) -> bool:
        return self.simulator_history == "without_simulator_history"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ApiCallMetadata(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    purpose: CallPurpose
    model: str
    temperature: float
    seed: int | None = None
    endpoint: str | None = None
    started_at: str
    finished_at: str
    latency_seconds: float
    retry_count: int = 0
    raw_response_id: str | None = None


class ChatResult(BaseModel):
    content: str
    metadata: ApiCallMetadata
    raw_response: dict[str, Any] = Field(default_factory=dict)


class SemanticValidationResult(BaseModel):
    preserves_false_premise: bool
    reveals_gold_correction: bool
    answers_original_question: bool
    changes_topic: bool
    follows_turn_strategy: bool
    preserves_baseline_semantics: bool | None = None
    matches_requested_style: bool | None = None


class ValidationResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    semantic: SemanticValidationResult | None = None


class BaselineRecord(BaseModel):
    turn: int
    strategy: str
    strategy_instruction: str
    text: str
    simulator_messages: list[ChatMessage]
    validation: ValidationResult
    api_metadata: ApiCallMetadata
    raw_response: dict[str, Any] = Field(default_factory=dict)
    semantic_validation_api_metadata: ApiCallMetadata | None = None
    semantic_validation_raw_response: dict[str, Any] = Field(
        default_factory=dict)


class PushbackRecord(BaseModel):
    turn: int
    strategy: str
    strategy_instruction: str
    text: str
    simulator_messages: list[ChatMessage]
    validation: ValidationResult
    api_metadata: ApiCallMetadata
    raw_response: dict[str, Any] = Field(default_factory=dict)
    semantic_validation_api_metadata: ApiCallMetadata | None = None
    semantic_validation_raw_response: dict[str, Any] = Field(
        default_factory=dict
    )
    baseline: BaselineRecord | None = None
    pushback_source: PushbackSource = "generated"
    source_scenario: str | None = None
    source_target_model: str | None = None
    source_transcript_path: str | None = None


class DialogueTurn(BaseModel):
    turn: int
    user: str
    user_source: Literal["dataset", "simulator"]
    strategy: str | None = None
    assistant: str
    target_messages: list[ChatMessage]
    api_metadata: ApiCallMetadata
    raw_response: dict[str, Any] = Field(default_factory=dict)


class TranscriptRecord(BaseModel):
    schema_version: str = TRANSCRIPT_SCHEMA_VERSION
    generated_at: str
    item: DatasetItem
    scenario: Scenario
    simulator_model: str
    validator_model: str | None = None
    target_model: str
    target_system_prompt: str
    turns: list[DialogueTurn]
    pushbacks: list[PushbackRecord]


class GenerationSettings(BaseModel):
    simulator_model: str = DEFAULT_SIMULATOR_MODEL
    validator_model: str | None = None
    target_model: str = DEFAULT_TARGET_MODEL
    simulator_temperature: float = 0.7
    validator_temperature: float = 0.0
    target_temperature: float = 0.0
    seed: int | None = None
    simulator_max_tokens: int = 180
    validator_max_tokens: int = 180
    target_max_tokens: int = DEFAULT_TARGET_MAX_TOKENS
    max_retries: int = 2
    target_system_prompt: str = DEFAULT_TARGET_SYSTEM_PROMPT


class BatchGenerationPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    items: Sequence[DatasetItem]
    scenarios: Sequence[Scenario]
    settings: Sequence[GenerationSettings]
    output_dir: Path
    resume: bool = False
    overwrite: bool = False


SCENARIOS = (
    Scenario(
        name="formal_without_simulator_history",
        style="formal",
        simulator_history="without_simulator_history",
    ),
    Scenario(
        name="formal_with_simulator_history",
        style="formal",
        simulator_history="with_simulator_history",
    ),
    Scenario(
        name="humanlike_without_simulator_history",
        style="humanlike",
        simulator_history="without_simulator_history",
    ),
    Scenario(
        name="humanlike_with_simulator_history",
        style="humanlike",
        simulator_history="with_simulator_history",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
