from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from transcript_generation import ChatMessage
from transcript_generation.client import extract_finish_reason, extract_reasoning_tokens
from transcript_generation.models import DEFAULT_KI4BUW_BASE_URL, utc_now

from .models import (
    DECISION_LABELS,
    JudgeConfig,
    JudgeClient,
    JudgeResult,
    validate_judge_config,
    JudgePanelMember,
    JudgeProvider,
    default_judge_config,
    resolve_judges,
)


JUDGE_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision_type": {
            "type": "string",
            "enum": list(DECISION_LABELS),
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["decision_type", "reason"],
    "additionalProperties": False,
}


def _retry_delay(error: Exception, retry_count: int) -> float:
    """Respect gateway backoff, including Cloudflare's JSON error body."""
    minimum = 120.0 if getattr(error, "status_code", None) == 524 else 1.0
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    body = getattr(error, "body", None)
    candidates = [headers.get("retry-after")]
    if isinstance(body, dict):
        candidates.append(body.get("retry_after"))
    delays = [minimum * 2 ** min(retry_count, 6)]
    for value in candidates:
        if value is None or isinstance(value, bool):
            continue
        try:
            delay = float(value)
        except (TypeError, ValueError):
            if not isinstance(value, str):
                continue
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(delay) and delay >= 0:
            delays.append(delay)
    return max(delays)


class _ReasoningJudgeClient(ABC):
    """Keep reasoning requests and raw responses inside the evaluator transport."""

    provider: JudgeProvider
    default_base_url: str | None = None

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
        resolved_base_url = base_url or self.default_base_url
        if resolved_base_url:
            client_options["base_url"] = resolved_base_url
        self._client = OpenAI(**client_options)
        self._base_url = resolved_base_url
        self._request_timeout_seconds = request_timeout_seconds
        self._api_retries = api_retries

    def complete(
        self,
        *,
        config: JudgeConfig,
        messages: list[ChatMessage],
    ) -> JudgeResult:
        validate_judge_config(self.provider, config)

        started = time.monotonic()
        started_at = utc_now()

        last_error: Exception | None = None
        for retry_count in range(self._api_retries + 1):
            try:
                response = self._create_response(config, messages)
            except Exception as error:
                last_error = error
                if retry_count >= self._api_retries:
                    raise
                delay = _retry_delay(error, retry_count)
                logging.getLogger(__name__).warning(
                    "Judge %s:%s request failed (%s, status=%s); "
                    "retrying in %.1fs (%d/%d).",
                    self.provider,
                    config.model,
                    type(error).__name__,
                    getattr(error, "status_code", "unavailable"),
                    delay,
                    retry_count + 1,
                    self._api_retries,
                )
                time.sleep(delay)
                continue

            raw = response.model_dump(mode="json")
            content, finish_reason, reasoning_tokens = self._response_fields(
                response, raw
            )
            metadata = {
                "call_id": str(uuid4()),
                "purpose": "validator",
                "model": config.model,
                "temperature": config.temperature,
                "seed": config.seed,
                "reasoning_effort": config.reasoning_effort,
                "reasoning_enabled": True,
                "allowed_openai_params": list(config.allowed_openai_params),
                "started_at": started_at,
                "finished_at": utc_now(),
                "latency_seconds": round(time.monotonic() - started, 3),
                "retry_count": retry_count,
                "raw_response_id": raw.get("id"),
                "endpoint": self._base_url,
                "finish_reason": finish_reason,
                "requested_max_tokens": config.max_tokens,
                "effective_max_tokens": config.max_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
            return JudgeResult(
                content=content.strip(),
                metadata=metadata,
                raw_response=raw,
            )

        assert last_error is not None
        raise last_error

    @abstractmethod
    def _create_response(
        self, config: JudgeConfig, messages: list[ChatMessage]
    ) -> Any: ...

    @abstractmethod
    def _response_fields(
        self, response: Any, raw: dict[str, Any]
    ) -> tuple[str, str | None, int | None]: ...


class OpenAIResponsesChatClient(_ReasoningJudgeClient):
    provider: JudgeProvider = "openai"

    def _create_response(self, config: JudgeConfig, messages: list[ChatMessage]) -> Any:
        request: dict[str, Any] = {
            "model": config.model,
            "input": [message.model_dump(mode="json") for message in messages],
            "reasoning": {"effort": config.reasoning_effort},
            "max_output_tokens": config.max_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "false_presupposition_judge_decision",
                    "strict": True,
                    "schema": JUDGE_DECISION_JSON_SCHEMA,
                }
            },
            "store": False,
        }
        return self._client.responses.create(
            **request, timeout=self._request_timeout_seconds
        )

    def _response_fields(
        self, response: Any, raw: dict[str, Any]
    ) -> tuple[str, str | None, int | None]:
        finish_reason = (raw.get("incomplete_details") or {}).get("reason") or raw.get(
            "status"
        )
        details = (raw.get("usage") or {}).get("output_tokens_details") or {}
        return (
            response.output_text or "",
            finish_reason,
            details.get("reasoning_tokens"),
        )


class OpenAICompatibleJudgeClient(_ReasoningJudgeClient):
    provider: JudgeProvider = "openai-compatible"
    default_base_url = DEFAULT_KI4BUW_BASE_URL

    def _create_response(self, config: JudgeConfig, messages: list[ChatMessage]) -> Any:
        request: dict[str, Any] = {
            "model": config.model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "extra_body": {
                "reasoning_effort": config.reasoning_effort,
                "allowed_openai_params": list(config.allowed_openai_params),
            },
        }
        # Consume within the retry boundary: failures can occur after HTTP 200,
        # while reading chunks. The SDK accumulates content and extra reasoning
        # fields separately and closes the connection on either outcome.
        with self._client.chat.completions.stream(
            **request,
            stream_options={"include_usage": True},
            timeout=self._request_timeout_seconds,
        ) as stream:
            response = stream.get_final_completion()
            if not response.choices or response.choices[0].finish_reason is None:
                raise RuntimeError("Judge stream ended without a finish reason")
            return response

    def _response_fields(
        self, response: Any, raw: dict[str, Any]
    ) -> tuple[str, str | None, int | None]:
        return (
            response.choices[0].message.content or "",
            extract_finish_reason(raw),
            extract_reasoning_tokens(raw),
        )


@dataclass(frozen=True)
class JudgeClientConfig:
    api_key: str
    base_url: str | None
    request_timeout_seconds: float
    api_retries: int


@dataclass(frozen=True)
class JudgeProviderProfile:
    client_factory: Callable[..., JudgeClient]
    api_key_environment: tuple[str, ...]
    base_url_environment: tuple[str, ...]
    default_base_url: str | None
    missing_api_key_message: str


PROVIDER_PROFILES: dict[str, JudgeProviderProfile] = {
    "openai-compatible": JudgeProviderProfile(
        client_factory=OpenAICompatibleJudgeClient,
        api_key_environment=("KI4BUW_API_KEY",),
        base_url_environment=("KI4BUW_BASE_URL",),
        default_base_url=DEFAULT_KI4BUW_BASE_URL,
        missing_api_key_message=(
            "Missing KI4BUW judge API key. Set KI4BUW_API_KEY or pass --ki4buw-api-key."
        ),
    ),
    "openai": JudgeProviderProfile(
        client_factory=OpenAIResponsesChatClient,
        api_key_environment=("OPENAI_API_KEY",),
        base_url_environment=("OPENAI_BASE_URL",),
        default_base_url=None,
        missing_api_key_message=(
            "Missing OpenAI judge API key. Set OPENAI_API_KEY or pass --openai-api-key."
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
    resolved_base_url = (
        _first_value(
            base_url,
            environ,
            profile.base_url_environment,
        )
        or profile.default_base_url
    )
    return JudgeClientConfig(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        request_timeout_seconds=request_timeout_seconds,
        api_retries=api_retries,
    )


def create_judge_client(
    *,
    provider: str,
    config: JudgeClientConfig,
) -> JudgeClient:
    """Create the transport adapter selected for judge model calls."""
    profile = _provider_profile(provider)
    return profile.client_factory(
        api_key=config.api_key,
        base_url=config.base_url,
        request_timeout_seconds=config.request_timeout_seconds,
        api_retries=config.api_retries,
    )


def configure_judges(
    *,
    environ: Mapping[str, str],
    aliases: Sequence[str] | None = None,
    ki4buw_base_url: str | None = None,
    openai_base_url: str | None = None,
    max_tokens: int = 8192,
    parse_retries: int = 1,
) -> tuple[JudgePanelMember, ...]:
    """Resolve evaluation identity only; no credentials or clients are needed."""
    definitions = resolve_judges(aliases)
    explicit = {"openai-compatible": ki4buw_base_url, "openai": openai_base_url}
    members = []
    for definition in definitions:
        profile = _provider_profile(definition.provider)
        endpoint = (
            _first_value(
                explicit[definition.provider], environ, profile.base_url_environment
            )
            or profile.default_base_url
        )
        config = replace(
            default_judge_config(definition),
            max_tokens=max_tokens,
            parse_retries=parse_retries,
        )
        validate_judge_config(definition.provider, config)
        members.append(JudgePanelMember(definition.provider, config, endpoint))
    return tuple(members)


def create_default_judge_panel(
    *,
    environ: Mapping[str, str],
    ki4buw_api_key: str | None,
    ki4buw_base_url: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
    request_timeout_seconds: float,
    api_retries: int,
    max_tokens: int,
    parse_retries: int,
    aliases: Sequence[str] | None = None,
) -> tuple[JudgePanelMember, ...]:
    """Convenience for callers that explicitly need clients immediately.

    CLI uses configure_judges and constructs clients only for pending work.
    """
    if request_timeout_seconds <= 0 or api_retries < 0:
        raise ValueError(
            "request timeout must be positive and API retries non-negative"
        )
    judges = configure_judges(
        environ=environ,
        aliases=aliases,
        ki4buw_base_url=ki4buw_base_url,
        openai_base_url=openai_base_url,
        max_tokens=max_tokens,
        parse_retries=parse_retries,
    )
    keys = {"openai-compatible": ki4buw_api_key, "openai": openai_api_key}
    configs = {
        j.provider: resolve_judge_client_config(
            provider=j.provider,
            api_key=keys[j.provider],
            base_url=j.endpoint,
            environ=environ,
            request_timeout_seconds=request_timeout_seconds,
            api_retries=api_retries,
        )
        for j in judges
    }
    clients = {
        provider: create_judge_client(provider=provider, config=config)
        for provider, config in configs.items()
    }
    return tuple(replace(j, client=clients[j.provider]) for j in judges)
