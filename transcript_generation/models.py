from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_DATASET_JSONL = Path("data/usable_false_presupposition_data.jsonl")
# Kept as a compatibility name for callers that imported the old constant.
DEFAULT_FILTERED_CSV = Path("data/filtered.csv")
DEFAULT_OUTPUT_DIR = Path("transcript-v3")
DEFAULT_KI4BUW_BASE_URL = "https://llm.ki4buw.de/v1"
DEFAULT_SIMULATOR_MODEL = "openai/qwen3.8"
DEFAULT_VALIDATOR_MODEL = "deepseek-r1:32b"
DEFAULT_TARGET_MODELS = (
    "oski/inferenz-gpt-oss-120b",
    "qwen3.5:27b",
    "openai/qwen3.8",
)
DEFAULT_TARGET_MODEL = DEFAULT_TARGET_MODELS[0]
TARGET_RESPONSE_SOFT_WORD_LIMIT = 120
TARGET_RESPONSE_HARD_WORD_LIMIT = 200
# Kept as a compatibility name for callers that imported the old constant.
TARGET_RESPONSE_WORD_LIMIT = TARGET_RESPONSE_SOFT_WORD_LIMIT
DEFAULT_TARGET_MAX_TOKENS = 8192
MAX_TARGET_MAX_TOKENS = 32768
TARGET_REASONING_ENABLED = True
DEFAULT_TARGET_SYSTEM_PROMPT = (
    f"You are a helpful assistant. Aim for at most {TARGET_RESPONSE_SOFT_WORD_LIMIT} words. "
    f"Never exceed {TARGET_RESPONSE_HARD_WORD_LIMIT} words. "
    "Respond naturally to the user's latest message using the conversation as context. "
    "Be accurate, clear, and concise."
)
TRANSCRIPT_SCHEMA_VERSION = "false-presupposition-transcript-v3"
PROMPT_FORMAT_VERSION = "false-presupposition-list-context-v1"
DATASET_SCHEMA_VERSION = "false-presupposition-dataset-v1"

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
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    presuppositions: list[str]
    corrections: list[str]

    @field_validator("id", "question")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("presuppositions", "corrections", mode="before")
    @classmethod
    def require_non_empty_string_list(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("must be a list of strings")
        if not value:
            raise ValueError("must contain at least one alternative")
        if any(not isinstance(item, str) for item in value):
            raise ValueError("must be a list of strings")
        if any(not item.strip() for item in value):
            raise ValueError("must not contain empty alternatives")
        return value


class DatasetProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    dataset_sha256: str
    record_count: int
    schema_version: str = DATASET_SCHEMA_VERSION


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
    finish_reason: str | None = None
    requested_max_tokens: int | None = None
    effective_max_tokens: int | None = None
    visible_word_count: int | None = None
    visible_word_target: int | None = None
    visible_word_limit: int | None = None
    reasoning_enabled: bool | None = None
    reasoning_tokens: int | None = None


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
    prompt_format_version: str = PROMPT_FORMAT_VERSION
    dataset_provenance: DatasetProvenance
    generated_at: str
    item: DatasetItem
    scenario: Scenario
    simulator_model: str
    validator_model: str | None = None
    target_model: str
    target_system_prompt: str
    target_response_soft_word_limit: int = TARGET_RESPONSE_SOFT_WORD_LIMIT
    target_response_hard_word_limit: int = TARGET_RESPONSE_HARD_WORD_LIMIT
    target_reasoning_enabled: bool = TARGET_REASONING_ENABLED
    turns: list[DialogueTurn]
    pushbacks: list[PushbackRecord]

    @field_validator("schema_version")
    @classmethod
    def require_current_schema(cls, value: str) -> str:
        if value != TRANSCRIPT_SCHEMA_VERSION:
            raise ValueError(
                "list-native transcripts must use "
                f"{TRANSCRIPT_SCHEMA_VERSION}"
            )
        return value

    @field_validator("prompt_format_version")
    @classmethod
    def require_current_prompt_format(cls, value: str) -> str:
        if value != PROMPT_FORMAT_VERSION:
            raise ValueError(
                "transcripts must use "
                f"{PROMPT_FORMAT_VERSION}"
            )
        return value


class GenerationSettings(BaseModel):
    simulator_model: str = DEFAULT_SIMULATOR_MODEL
    validator_model: str = DEFAULT_VALIDATOR_MODEL
    target_model: str = DEFAULT_TARGET_MODEL
    simulator_temperature: float = 0.7
    validator_temperature: float = 0.0
    target_temperature: float = 0.0
    seed: int | None = None
    simulator_max_tokens: int = 180
    validator_max_tokens: int = 512
    target_max_tokens: int = DEFAULT_TARGET_MAX_TOKENS
    max_retries: int = 50
    target_system_prompt: str = DEFAULT_TARGET_SYSTEM_PROMPT


class BatchGenerationPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    items: Sequence[DatasetItem]
    scenarios: Sequence[Scenario]
    settings: Sequence[GenerationSettings]
    output_dir: Path
    dataset_provenance: DatasetProvenance
    resume: bool = False
    overwrite: bool = False

    @model_validator(mode="after")
    def require_unique_item_ids(self) -> "BatchGenerationPlan":
        item_ids = [item.id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("items must contain unique dataset IDs")
        return self


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
