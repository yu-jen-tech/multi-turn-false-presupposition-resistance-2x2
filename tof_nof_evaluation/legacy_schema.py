"""Constants and voting validation only for explicit legacy readers."""

from dataclasses import replace

from .models import JudgeConfig, JudgeDefinition, JudgeProvider

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


def validate_legacy_config(provider: JudgeProvider, config: JudgeConfig) -> None:
    """Validate archived panel settings without enabling retired runtime judges."""
    definition = JudgeDefinition(provider, config.model)
    if definition not in LEGACY_JUDGES:
        raise ValueError(f"Unsupported legacy judge definition: {definition!r}")
    expected = JudgeConfig(
        model=config.model,
        temperature=None if provider == "openai" else 0.0,
        reasoning_effort="high",
        allowed_openai_params=() if provider == "openai" else ("reasoning_effort",),
        max_tokens=config.max_tokens,
        parse_retries=config.parse_retries,
    )
    allowed = [expected]
    if definition in LEGACY_JUDGES[:2]:
        allowed.append(replace(expected, reasoning_effort="xhigh"))
    if config not in allowed or config.max_tokens <= 0 or config.parse_retries < 0:
        raise ValueError("Invalid legacy judge configuration")
