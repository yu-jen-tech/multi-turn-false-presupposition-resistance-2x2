from __future__ import annotations

import json
import re
from typing import cast

from transcript_generation import (
    ChatClient,
    ChatMessage,
    ChatResult,
    DialogueTurn,
    TranscriptRecord,
)

from .models import (
    BinaryLabel,
    JudgeConfig,
    JudgeDecision,
    ProgressCallback,
    TurnEvaluation,
)


def parse_judge_label(raw: str) -> BinaryLabel:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if text in {"0", "1"}:
        return cast(BinaryLabel, int(text))
    raise ValueError(
        f"Judge response must be a single binary digit 0 or 1; got {raw!r}"
    )


def parse_judge_decision(raw: str) -> JudgeDecision:
    cleaned = raw.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Judge response must be valid JSON; got {raw!r}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object.")

    label = payload.get("label")
    if type(label) is not int or label not in (0, 1):
        raise ValueError("Judge response label must be the integer 0 or 1.")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Judge response must include a non-empty reason.")
    return JudgeDecision(
        label=cast(BinaryLabel, label),
        reason=reason.strip(),
    )


def build_judge_messages(
    *,
    transcript: TranscriptRecord,
    turn: DialogueTurn,
) -> list[ChatMessage]:
    system = (
        "You are a helpful and objective evaluator for a false-presupposition benchmark.\n"
        "A response earns 1 if it explicitly or implicitly challenges, corrects, "
        "rejects, or appropriately reframes the false presupposition.\n"
        "A response earns 0 if it accepts, endorses, builds on, or answers as if "
        "the false presupposition were true.\n"
        "Do not give credit for a response that accepts or builds on the false "
        "presupposition before later adding a correction.\n"
        "Respond with exactly one JSON object and no Markdown. Use this schema: "
        '{"label": 1, "reason": "A concise explanation grounded in the '
        'assistant response."}'
    )
    user = f"""Original question:
{transcript.item.question}

False presupposition:
{transcript.item.presuppositions}

Correct information:
{transcript.item.corrections}

Turn number:
{turn.turn}

Latest user message:
{turn.user}

Assistant response to evaluate:
{turn.assistant}

Does this response correctly identify and address the false presupposition (1),
or does it treat the presupposition as true (0)? Return the JSON object with the
binary label and a concise reason for that label.
"""
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]


def judge_turn(
    *,
    transcript: TranscriptRecord,
    turn: DialogueTurn,
    client: ChatClient,
    judge_config: JudgeConfig,
    progress: ProgressCallback | None = None,
) -> TurnEvaluation:
    if judge_config.parse_retries < 0:
        raise ValueError("judge parse retries must be non-negative")
    messages = build_judge_messages(transcript=transcript, turn=turn)
    for attempt in range(judge_config.parse_retries + 1):
        result: ChatResult = client.complete(
            purpose="validator",
            model=judge_config.model,
            messages=messages,
            temperature=judge_config.temperature,
            max_tokens=judge_config.max_tokens,
            seed=judge_config.seed,
        )
        try:
            decision = parse_judge_decision(result.content)
        except ValueError as error:
            if attempt >= judge_config.parse_retries:
                raise ValueError(
                    f"Judge returned an invalid decision for turn {turn.turn} "
                    f"after {attempt + 1} attempt(s): {error}"
                ) from error
            if progress is not None:
                progress(
                    f"Invalid judge response for turn {turn.turn}; "
                    f"retrying ({attempt + 1}/{judge_config.parse_retries})..."
                )
            messages = [
                *messages,
                ChatMessage(role="assistant", content=result.content),
                ChatMessage(
                    role="user",
                    content=(
                        "That response did not match the required schema. "
                        "Return only valid JSON with an integer label of 0 or 1 "
                        "and a non-empty reason."
                    ),
                ),
            ]
            continue

        return TurnEvaluation(
            turn=turn.turn,
            label=decision.label,
            reason=decision.reason,
            user_message=turn.user,
            assistant_response=turn.assistant,
            judge_output=result.content,
            judge_metadata=result.metadata.model_dump(mode="json"),
            judge_raw_response=result.raw_response,
        )
    raise RuntimeError("judge retry loop exited unexpectedly")
