from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from .models import (
    ApiCallMetadata,
    ChatMessage,
    DatasetItem,
    SemanticValidationResult,
    SurfaceStyleValidationResult,
    ValidationResult,
)


TraceCallback = Callable[[str], None]

_DELIMITER = "─" * 88


def emit_model_call(
    callback: TraceCallback | None,
    *,
    label: str,
    operation: str,
    purpose: str,
    model: str,
    item: DatasetItem,
    scenario: str,
    turn: int | None,
    attempt: int,
    messages: Sequence[ChatMessage],
    content: str,
    metadata: ApiCallMetadata,
    provider_retry_count: int | None = None,
    generation_retry_count: int | None = None,
) -> None:
    if callback is None:
        return

    context = _context(
        label=label,
        operation=operation,
        purpose=purpose,
        model=model,
        item=item,
        scenario=scenario,
        turn=turn,
        attempt=attempt,
        metadata=metadata,
        provider_retry_count=provider_retry_count,
        generation_retry_count=generation_retry_count,
    )
    prompt = "\n".join(
        f"{message.role}: {message.content}" for message in messages
    )
    callback(
        "\n".join(
            [
                f"{_DELIMITER} {context} {_DELIMITER}",
                "PROMPT:",
                prompt,
                "",
                "RESPONSE:",
                content,
                _DELIMITER,
            ]
        )
    )


def emit_validation(
    callback: TraceCallback | None,
    *,
    label: str,
    operation: str,
    item: DatasetItem,
    scenario: str,
    turn: int,
    attempt: int,
    candidate: str,
    lexical: ValidationResult,
    validation: ValidationResult,
    semantic_was_run: bool,
    surface_style_was_run: bool = False,
) -> None:
    if callback is None:
        return

    semantic = validation.semantic
    lines = [
        f"{_DELIMITER} {label} {operation} | item={item.id} "
        f"| scenario={scenario} | turn={turn} | attempt={attempt} {_DELIMITER}",
        "CANDIDATE:",
        candidate,
        f"LEXICAL VALIDATION: {_status(lexical.passed)}",
        f"LEXICAL REASONS: {_reasons(lexical.reasons)}",
        (
            f"SEMANTIC VALIDATION: {_status(validation.passed)}"
            if semantic_was_run
            else "SEMANTIC VALIDATION: NOT RUN"
        ),
    ]
    if semantic is not None:
        lines.append(f"SEMANTIC FIELDS: {_semantic_fields(semantic)}")
    surface_style = validation.surface_style
    if surface_style_was_run:
        surface_status = (
            surface_style is not None
            and surface_style.matches_requested_style
            and surface_style.preserves_protected_anchors
        )
        lines.append(f"SURFACE STYLE VALIDATION: {_status(surface_status)}")
        if surface_style is not None:
            lines.append(
                f"SURFACE STYLE FIELDS: {_surface_style_fields(surface_style)}"
            )
    else:
        lines.append("SURFACE STYLE VALIDATION: NOT RUN")
    lines.extend(
        [
            f"REASONS: {_reasons(validation.reasons)}",
            _DELIMITER,
        ]
    )
    callback("\n".join(lines))


def emit_reuse(
    callback: TraceCallback | None,
    *,
    label: str,
    operation: str,
    item_id: str,
    source_path: Path | str,
    scenario: str | None = None,
    target_model: str | None = None,
) -> None:
    if callback is None:
        return

    fields = [f"item={item_id}"]
    if scenario is not None:
        fields.append(f"scenario={scenario}")
    if target_model is not None:
        fields.append(f"target={target_model}")
    fields.append(f"source={source_path}")
    callback(f"{label} REUSE {operation} | " + " | ".join(fields))


def _context(
    *,
    label: str,
    operation: str,
    purpose: str,
    model: str,
    item: DatasetItem,
    scenario: str,
    turn: int | None,
    attempt: int,
    metadata: ApiCallMetadata,
    provider_retry_count: int | None,
    generation_retry_count: int | None,
) -> str:
    fields = [
        f"purpose={purpose}",
        f"item={item.id}",
        f"scenario={scenario}",
    ]
    if turn is not None:
        fields.append(f"turn={turn}")
    fields.extend(
        [
            f"attempt={attempt}",
            f"retry={generation_retry_count if generation_retry_count is not None else metadata.retry_count}",
            f"model={model}",
            f"latency={metadata.latency_seconds:.3f}s",
        ]
    )
    if metadata.finish_reason is not None:
        fields.append(f"finish_reason={metadata.finish_reason}")
    if metadata.requested_max_tokens is not None:
        fields.append(f"requested_max_tokens={metadata.requested_max_tokens}")
    if metadata.effective_max_tokens is not None:
        fields.append(f"max_tokens={metadata.effective_max_tokens}")
    if metadata.visible_word_count is not None:
        fields.append(f"visible_words={metadata.visible_word_count}")
    if metadata.reasoning_tokens is not None:
        fields.append(f"reasoning_tokens={metadata.reasoning_tokens}")
    if metadata.reasoning_enabled is not None:
        fields.append(f"reasoning_enabled={metadata.reasoning_enabled}")
    if provider_retry_count is not None:
        fields.insert(-2, f"provider_retry={provider_retry_count}")
    return f"{label} {operation} | " + " | ".join(fields)


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _reasons(reasons: Sequence[str]) -> str:
    return ", ".join(reasons) if reasons else "none"


def _semantic_fields(semantic: SemanticValidationResult) -> str:
    return ", ".join(
        f"{name}={str(value).lower()}"
        for name, value in semantic.model_dump(mode="json").items()
        if value is not None
    )


def _surface_style_fields(surface_style: SurfaceStyleValidationResult) -> str:
    return ", ".join(
        f"{name}={str(value).lower()}"
        for name, value in surface_style.model_dump(mode="json").items()
    )
