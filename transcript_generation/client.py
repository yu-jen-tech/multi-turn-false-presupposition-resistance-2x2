from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Protocol

from .models import (
    ApiCallMetadata,
    CallPurpose,
    ChatMessage,
    ChatResult,
    DEFAULT_KI4BUW_BASE_URL,
    utc_now,
)


def extract_finish_reason(raw_response: dict[str, Any]) -> str | None:
    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices:
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    return None


def _finish_reason(response: Any, raw_response: dict[str, Any]) -> str | None:
    reason = extract_finish_reason(raw_response)
    if reason is not None:
        return reason

    response_choices = getattr(response, "choices", None)
    if response_choices:
        reason = getattr(response_choices[0], "finish_reason", None)
        if isinstance(reason, str):
            return reason
    return None


def extract_reasoning_tokens(raw_response: dict[str, Any]) -> int | None:
    usage = raw_response.get("usage")
    if not isinstance(usage, dict):
        return None

    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
        if isinstance(value, int):
            return value

    value = usage.get("reasoning_tokens")
    return value if isinstance(value, int) else None


def resolve_ki4buw_base_url(
    base_url: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    return (
        base_url
        or environment.get("KI4BUW_BASE_URL")
        or DEFAULT_KI4BUW_BASE_URL
    )


class ChatClient(Protocol):
    def complete(
        self,
        *,
        purpose: CallPurpose,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> ChatResult:
        ...


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        request_timeout_seconds: float = 120,
        api_retries: int = 1,
    ):
        from openai import OpenAI

        resolved_base_url = resolve_ki4buw_base_url(base_url)
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": resolved_base_url,
        }
        kwargs["timeout"] = request_timeout_seconds
        self._client = OpenAI(**kwargs)
        self._base_url = resolved_base_url
        self._request_timeout_seconds = request_timeout_seconds
        self._api_retries = api_retries

    def complete(
        self,
        *,
        purpose: CallPurpose,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> ChatResult:
        started = time.monotonic()
        started_at = utc_now()
        request: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            request["seed"] = seed

        last_error: Exception | None = None
        for retry_count in range(self._api_retries + 1):
            try:
                # Collect streamed chunks within the retry boundary so an
                # interrupted stream is retried rather than saved as complete.
                with self._client.chat.completions.stream(
                    **request,
                    stream_options={"include_usage": True},
                    timeout=self._request_timeout_seconds,
                ) as stream:
                    # The SDK's final parser raises on finish_reason="length".
                    # Preserve it for the generator's validation/retry policy,
                    # after consuming all chunks (including final usage).
                    stream.until_done()
                    response = stream.current_completion_snapshot
                    if not response.choices or response.choices[0].finish_reason is None:
                        raise RuntimeError("Generation stream ended without a finish reason")
            except Exception as error:
                last_error = error
                if retry_count >= self._api_retries:
                    raise
                time.sleep(120 + retry_count)
                continue

            finished_at = utc_now()
            raw = response.model_dump(mode="json")
            content = response.choices[0].message.content or ""
            metadata = ApiCallMetadata(
                purpose=purpose,
                model=model,
                temperature=temperature,
                seed=seed,
                endpoint=self._base_url,
                started_at=started_at,
                finished_at=finished_at,
                latency_seconds=round(time.monotonic() - started, 3),
                retry_count=retry_count,
                raw_response_id=getattr(response, "id", None),
                finish_reason=_finish_reason(response, raw),
                requested_max_tokens=max_tokens,
                effective_max_tokens=max_tokens,
                reasoning_tokens=extract_reasoning_tokens(raw),
            )
            return ChatResult(
                content=content.strip(),
                metadata=metadata,
                raw_response=raw,
            )

        assert last_error is not None
        raise last_error
