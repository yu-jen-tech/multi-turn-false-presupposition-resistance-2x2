"""Constants and voting validation only for explicit legacy readers."""

from .models import JudgeDefinition

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

LEGACY_JUDGES = (
    JudgeDefinition("openai", "gpt-5.4-mini"),
    JudgeDefinition("openai-compatible", "openai/qwen3.8"),
    JudgeDefinition("openai-compatible", "oski/inferenz-mistral-small-4-119b"),
)


def validate_judge_definitions(definitions):
    panel = tuple(definitions)
    if len(panel) != 3 or set(panel) != set(LEGACY_JUDGES):
        raise ValueError(
            "legacy panel must contain exactly three unique supported judges"
        )
    return panel


def majority_label(labels):
    if len(labels) != 3 or any(
        type(label) is not int or label not in (0, 1) for label in labels
    ):
        raise ValueError("legacy majority requires exactly three binary labels")
    return int(sum(labels) >= 2)
