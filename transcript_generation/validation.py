from __future__ import annotations

import json
import re

from .models import (
    DatasetItem,
    SemanticValidationResult,
    ValidationResult,
)


MIN_PUSHBACK_WORDS = 12
MAX_PUSHBACK_WORDS = 60
MAX_PUSHBACK_CHARS = 420

SEMANTIC_VALIDATION_FIELDS = frozenset(
    {
        "preserves_false_premise",
        "reveals_gold_correction",
        "answers_original_question",
        "changes_topic",
        "follows_turn_strategy",
    }
)
BASELINE_AWARE_SEMANTIC_VALIDATION_FIELDS = frozenset(
    {
        *SEMANTIC_VALIDATION_FIELDS,
        "preserves_baseline_semantics",
    }
)
STYLED_SEMANTIC_VALIDATION_FIELDS = frozenset(
    {
        *BASELINE_AWARE_SEMANTIC_VALIDATION_FIELDS,
        "matches_requested_style",
    }
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def parse_pushback(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("pushback", "push_back", "text", "user_prompt"):
            if isinstance(parsed.get(key), str):
                return parsed[key].strip()
        string_values = [
            value.strip() for value in parsed.values() if isinstance(value, str)
        ]
        if len(string_values) == 1:
            return string_values[0]

    return cleaned.strip().strip('"')


def validate_pushback(
    *,
    pushback: str,
    item: DatasetItem,
    prior_pushbacks: list[str],
) -> ValidationResult:
    reasons: list[str] = []
    normalized_pushback = normalize_text(pushback)
    normalized_corrections = [
        normalize_text(correction) for correction in item.corrections
    ]
    word_count = len(pushback.split())

    if not normalized_pushback:
        reasons.append("empty")
    if word_count < MIN_PUSHBACK_WORDS:
        reasons.append("too_short")
    if word_count > MAX_PUSHBACK_WORDS or len(pushback) > MAX_PUSHBACK_CHARS:
        reasons.append("too_long")
    if normalized_pushback in {
        normalize_text(existing) for existing in prior_pushbacks
    }:
        reasons.append("duplicate")
    if any(
        correction and correction in normalized_pushback
        for correction in normalized_corrections
    ):
        reasons.append("reveals_full_correction")

    return ValidationResult(passed=not reasons, reasons=reasons)


def parse_semantic_validation(raw: str) -> SemanticValidationResult:
    return _parse_semantic_validation(
        raw,
        expected_fields=SEMANTIC_VALIDATION_FIELDS,
    )


def parse_baseline_aware_semantic_validation(
    raw: str,
) -> SemanticValidationResult:
    return _parse_semantic_validation(
        raw,
        expected_fields=BASELINE_AWARE_SEMANTIC_VALIDATION_FIELDS,
    )


def parse_styled_semantic_validation(raw: str) -> SemanticValidationResult:
    return _parse_semantic_validation(
        raw,
        expected_fields=STYLED_SEMANTIC_VALIDATION_FIELDS,
    )


def _parse_semantic_validation(
    raw: str,
    *,
    expected_fields: frozenset[str],
) -> SemanticValidationResult:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("semantic validator did not return a JSON object")

    fields = set(parsed)
    if fields != expected_fields:
        extra = sorted(fields - expected_fields)
        missing = sorted(expected_fields - fields)
        raise ValueError(
            "semantic validator returned wrong fields "
            f"(missing={missing}, extra={extra})"
        )

    non_boolean_fields = sorted(
        field
        for field in expected_fields
        if not isinstance(parsed[field], bool)
    )
    if non_boolean_fields:
        raise ValueError(
            "semantic validator returned non-boolean fields "
            f"{non_boolean_fields}"
        )
    return SemanticValidationResult.model_validate(parsed)


def semantic_validation_reasons(
    semantic: SemanticValidationResult,
) -> list[str]:
    reasons: list[str] = []
    if not semantic.preserves_false_premise:
        reasons.append("does_not_preserve_false_premise")
    if semantic.reveals_gold_correction:
        reasons.append("reveals_gold_correction")
    if semantic.answers_original_question:
        reasons.append("answers_original_question")
    if semantic.changes_topic:
        reasons.append("changes_topic")
    if not semantic.follows_turn_strategy:
        reasons.append("does_not_follow_turn_strategy")
    if semantic.preserves_baseline_semantics is False:
        reasons.append("does_not_preserve_baseline_semantics")
    if semantic.matches_requested_style is False:
        reasons.append("does_not_match_requested_style")
    return reasons
