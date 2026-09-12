from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from transcript_generation import (
    ApiCallMetadata,
    CallPurpose,
    ChatClient,
    ChatMessage,
    ChatResult,
    OpenAICompatibleChatClient,
)
from transcript_generation.models import utc_now

from .models import JudgeProvider


class OpenAIResponsesChatClient:
    """Adapt the official OpenAI Responses API to the judge chat interface."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        request_timeout_seconds: float = 120,
        api_retries: int = 1,
    ) -> None:
        from openai import OpenAI

        client_options: dict[str, Any] = {
            "api_key": api_key,
            "timeout": request_timeout_seconds,
            "max_retries": 0,
        }
        if base_url:
            client_options["base_url"] = base_url
        self._client = OpenAI(**client_options)
        self._base_url = base_url
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
        if seed is not None:
            raise ValueError("The OpenAI Responses API does not support seed.")

        request: dict[str, Any] = {
            "model": model,
            "input": [message.model_dump(mode="json") for message in messages],
            "max_output_tokens": max_tokens,
            "store": False,
        }
        started = time.monotonic()
        started_at = utc_now()

        last_error: Exception | None = None
        for retry_count in range(self._api_retries + 1):
            try:
                response = self._client.responses.create(
                    **request,
                    timeout=self._request_timeout_seconds,
                )
            except Exception as error:
                last_error = error
                if retry_count >= self._api_retries:
                    raise
                time.sleep(1 + retry_count)
                continue

            metadata = ApiCallMetadata(
                purpose=purpose,
                model=model,
                temperature=temperature,
                started_at=started_at,
                finished_at=utc_now(),
                latency_seconds=round(time.monotonic() - started, 3),
                retry_count=retry_count,
                raw_response_id=getattr(response, "id", None),
                endpoint=self._base_url,
            )
            return ChatResult(
                content=(response.output_text or "").strip(),
                metadata=metadata,
                raw_response=response.model_dump(mode="json"),
            )

        assert last_error is not None
        raise last_error


@dataclass(frozen=True)
class JudgeClientConfig:
    api_key: str
    base_url: str | None
    request_timeout_seconds: float
    api_retries: int


@dataclass(frozen=True)
class JudgeProviderProfile:
    client_factory: Callable[..., ChatClient]
    api_key_environment: tuple[str, ...]
    base_url_environment: tuple[str, ...]
    missing_api_key_message: str


DEFAULT_JUDGE_PROVIDER: JudgeProvider = "openai-compatible"
PROVIDER_PROFILES: dict[str, JudgeProviderProfile] = {
    "openai-compatible": JudgeProviderProfile(
        client_factory=OpenAICompatibleChatClient,
        api_key_environment=("JUDGE_API_KEY", "KI4BUW_API_KEY"),
        base_url_environment=("JUDGE_BASE_URL", "KI4BUW_BASE_URL"),
        missing_api_key_message=(
            "Missing judge API key. Set JUDGE_API_KEY or KI4BUW_API_KEY; "
            "or pass --judge-api-key."
        ),
    ),
    "openai": JudgeProviderProfile(
        client_factory=OpenAIResponsesChatClient,
        api_key_environment=("JUDGE_API_KEY", "OPENAI_API_KEY"),
        base_url_environment=("JUDGE_BASE_URL", "OPENAI_BASE_URL"),
        missing_api_key_message=(
            "Missing judge API key. Set JUDGE_API_KEY or OPENAI_API_KEY; "
            "or pass --judge-api-key."
        ),
    ),
}
JUDGE_PROVIDERS = tuple(PROVIDER_PROFILES)


def _provider_profile(provider: str) -> JudgeProviderProfile:
    try:
        return PROVIDER_PROFILES[provider]
    except KeyError as error:
        raise ValueError(f"Unsupported judge provider: {provider!r}") from error


def _first_value(
    explicit: str | None,
    environ: Mapping[str, str],
    names: tuple[str, ...],
) -> str | None:
    if explicit:
        return explicit
    return next((environ[name] for name in names if environ.get(name)), None)


def resolve_judge_client_config(
    *,
    provider: str,
    api_key: str | None,
    base_url: str | None,
    environ: Mapping[str, str],
    request_timeout_seconds: float,
    api_retries: int,
) -> JudgeClientConfig:
    """Resolve one provider's connection settings and credential precedence."""
    profile = _provider_profile(provider)
    resolved_api_key = _first_value(
        api_key,
        environ,
        profile.api_key_environment,
    )
    if not resolved_api_key:
        raise ValueError(profile.missing_api_key_message)
    return JudgeClientConfig(
        api_key=resolved_api_key,
        base_url=_first_value(
            base_url,
            environ,
            profile.base_url_environment,
        ),
        request_timeout_seconds=request_timeout_seconds,
        api_retries=api_retries,
    )


def create_judge_client(
    *,
    provider: str,
    config: JudgeClientConfig,
) -> ChatClient:
    """Create the transport adapter selected for judge model calls."""
    profile = _provider_profile(provider)
    return profile.client_factory(
        api_key=config.api_key,
        base_url=config.base_url,
        request_timeout_seconds=config.request_timeout_seconds,
        api_retries=config.api_retries,
    )
