"""Generate and persist false-presupposition evaluation transcripts."""

from .batch import generate_transcripts
from .client import ChatClient, OpenAICompatibleChatClient
from .generator import TranscriptGenerator
from .models import (
    ApiCallMetadata,
    BatchGenerationPlan,
    CallPurpose,
    ChatMessage,
    ChatResult,
    DatasetItem,
    DatasetProvenance,
    DialogueTurn,
    GenerationSettings,
    PushbackRecord,
    PushbackSource,
    SCENARIOS,
    Scenario,
    SemanticValidationResult,
    TranscriptRecord,
    ValidationResult,
)
from .storage import load_transcript


__all__ = [
    "ApiCallMetadata",
    "BatchGenerationPlan",
    "CallPurpose",
    "ChatClient",
    "ChatMessage",
    "ChatResult",
    "DatasetItem",
    "DatasetProvenance",
    "DialogueTurn",
    "GenerationSettings",
    "OpenAICompatibleChatClient",
    "PushbackRecord",
    "PushbackSource",
    "SCENARIOS",
    "Scenario",
    "SemanticValidationResult",
    "TranscriptGenerator",
    "TranscriptRecord",
    "ValidationResult",
    "generate_transcripts",
    "load_transcript",
]
