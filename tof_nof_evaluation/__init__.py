"""Reusable ToF/NoF transcript evaluation."""

from transcript_generation import OpenAICompatibleChatClient

from .cli import main, parse_args, print_progress
from .evaluation import evaluate_transcript, evaluate_transcript_files
from .judging import (
    build_judge_messages,
    judge_turn,
    parse_judge_decision,
    parse_judge_label,
)
from .metrics import calculate_nof, calculate_tof
from .models import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PARSE_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSCRIPT_DIR,
    REASON_COLUMNS,
    RESULT_COLUMNS,
    BinaryLabel,
    JudgeConfig,
    JudgeDecision,
    JudgeProvider,
    ProgressCallback,
    TranscriptScore,
    TurnEvaluation,
)
from .providers import (
    DEFAULT_JUDGE_PROVIDER,
    JUDGE_PROVIDERS,
    JudgeClientConfig,
    OpenAIResponsesChatClient,
    create_judge_client,
    resolve_judge_client_config,
)
from .reporting import (
    grouped_scores,
    score_to_csv_row,
    score_to_reason_rows,
    summarize_scores,
    write_outputs,
)

__all__ = [
    "DEFAULT_JUDGE_MAX_TOKENS",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_PARSE_RETRIES",
    "DEFAULT_JUDGE_PROVIDER",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TRANSCRIPT_DIR",
    "JUDGE_PROVIDERS",
    "REASON_COLUMNS",
    "RESULT_COLUMNS",
    "BinaryLabel",
    "JudgeConfig",
    "JudgeClientConfig",
    "JudgeDecision",
    "JudgeProvider",
    "OpenAICompatibleChatClient",
    "OpenAIResponsesChatClient",
    "ProgressCallback",
    "TranscriptScore",
    "TurnEvaluation",
    "build_judge_messages",
    "calculate_nof",
    "calculate_tof",
    "create_judge_client",
    "evaluate_transcript",
    "evaluate_transcript_files",
    "grouped_scores",
    "judge_turn",
    "main",
    "parse_args",
    "parse_judge_decision",
    "parse_judge_label",
    "print_progress",
    "resolve_judge_client_config",
    "score_to_csv_row",
    "score_to_reason_rows",
    "summarize_scores",
    "write_outputs",
]
