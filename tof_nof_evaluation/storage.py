from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from dialogue_extractor.dialogue_extractor import ordered_dialogue_turns
from transcript_generation import TranscriptRecord
from transcript_generation.storage import model_path_component

from .atomic_io import atomic_write_text
from .judging import build_judge_messages
from .models import JudgeConfig, TurnEvaluation


CHECKPOINT_SCHEMA_VERSION = "tof-nof-evaluation-checkpoint-v2"


@dataclass(frozen=True)
class EvaluationCheckpointContext:
    output_dir: Path
    judge_config: JudgeConfig
    judge_provider: str
    judge_endpoint: str | None


def evaluation_checkpoint_path(
    context: EvaluationCheckpointContext,
    *,
    transcript: TranscriptRecord,
) -> Path:
    return (
        context.output_dir
        / model_path_component(context.judge_config.model)
        / model_path_component(transcript.target_model)
        / transcript.scenario.name
        / "evaluations"
        / f"{transcript.item.id}.json"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _prompt_sha256(
    transcript: TranscriptRecord,
    *,
    transcript_path: Path,
) -> str:
    prompt_payload = [
        [message.model_dump(mode="json") for message in build_judge_messages(
            transcript=transcript,
            turn=turn,
        )]
        for turn in ordered_dialogue_turns(transcript, path=transcript_path)
    ]
    serialized = json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(serialized)


def _judge_payload(
    context: EvaluationCheckpointContext,
    *,
    transcript: TranscriptRecord,
    transcript_path: Path,
) -> dict[str, Any]:
    return {
        "provider": context.judge_provider,
        "endpoint": context.judge_endpoint,
        **asdict(context.judge_config),
        "prompt_sha256": _prompt_sha256(
            transcript,
            transcript_path=transcript_path,
        ),
    }


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    description: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{description} fields do not match the current checkpoint schema"
        )


def _turn_evaluations_from_payload(
    payload: object,
) -> tuple[TurnEvaluation, ...]:
    if not isinstance(payload, list):
        raise ValueError("checkpoint turn evaluations must be a list")
    evaluation_fields = {field.name for field in fields(TurnEvaluation)}
    turn_evaluations: list[TurnEvaluation] = []
    for evaluation_payload in payload:
        if not isinstance(evaluation_payload, dict):
            raise ValueError("checkpoint turn evaluation must be a JSON object")
        _require_exact_fields(
            evaluation_payload,
            evaluation_fields,
            description="checkpoint turn evaluation",
        )
        turn_evaluations.append(TurnEvaluation(**evaluation_payload))
    return tuple(turn_evaluations)


def _validate_turn_evaluations(
    turn_evaluations: Sequence[TurnEvaluation],
    *,
    transcript: TranscriptRecord,
) -> None:
    if len(turn_evaluations) > 5:
        raise ValueError(
            "checkpoint must contain at most 5 turn evaluations"
        )
    expected_turn_numbers = tuple(range(1, len(turn_evaluations) + 1))
    if tuple(evaluation.turn for evaluation in turn_evaluations) != (
        expected_turn_numbers
    ):
        raise ValueError(
            "checkpoint turn evaluations must be a consecutive prefix of turns 1-5"
        )
    turns_by_number = {turn.turn: turn for turn in transcript.turns}
    for evaluation in turn_evaluations:
        if type(evaluation.label) is not int or evaluation.label not in (0, 1):
            raise ValueError("checkpoint labels must be binary integers")
        if not evaluation.reason.strip():
            raise ValueError("checkpoint reasons must be non-empty")
        source_turn = turns_by_number[evaluation.turn]
        if (
            evaluation.user_message != source_turn.user
            or evaluation.assistant_response != source_turn.assistant
        ):
            raise ValueError(
                "checkpoint turn evaluations do not match the source transcript"
            )


def write_evaluation_checkpoint(
    turn_evaluations: Sequence[TurnEvaluation],
    *,
    context: EvaluationCheckpointContext,
    transcript: TranscriptRecord,
    transcript_path: Path,
) -> Path:
    _validate_turn_evaluations(turn_evaluations, transcript=transcript)
    path = evaluation_checkpoint_path(context, transcript=transcript)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_transcript": {
            "path": str(transcript_path),
            "sha256": _source_sha256(transcript_path),
        },
        "judge": _judge_payload(
            context,
            transcript=transcript,
            transcript_path=transcript_path,
        ),
        "turn_evaluations": [
            asdict(evaluation) for evaluation in turn_evaluations
        ],
    }
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    return path


def load_evaluation_checkpoint(
    *,
    context: EvaluationCheckpointContext,
    transcript: TranscriptRecord,
    transcript_path: Path,
) -> tuple[Path, tuple[TurnEvaluation, ...]] | None:
    path = evaluation_checkpoint_path(context, transcript=transcript)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must be a JSON object")
        _require_exact_fields(
            payload,
            {
                "schema_version",
                "source_transcript",
                "judge",
                "turn_evaluations",
            },
            description="checkpoint",
        )
        if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema version is stale")
        source = payload["source_transcript"]
        if not isinstance(source, dict):
            raise ValueError("checkpoint source transcript must be an object")
        _require_exact_fields(
            source,
            {"path", "sha256"},
            description="checkpoint source transcript",
        )
        if source["sha256"] != _source_sha256(transcript_path):
            raise ValueError("source transcript content has changed")

        judge = payload["judge"]
        expected_judge = _judge_payload(
            context,
            transcript=transcript,
            transcript_path=transcript_path,
        )
        if not isinstance(judge, dict):
            raise ValueError("checkpoint judge configuration must be an object")
        _require_exact_fields(
            judge,
            set(expected_judge),
            description="checkpoint judge configuration",
        )
        if judge["prompt_sha256"] != expected_judge["prompt_sha256"]:
            raise ValueError("judge prompt has changed")
        if judge != expected_judge:
            raise ValueError("judge configuration has changed")

        turn_evaluations = _turn_evaluations_from_payload(
            payload["turn_evaluations"]
        )
        _validate_turn_evaluations(
            turn_evaluations,
            transcript=transcript,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Cannot resume from stale or invalid evaluation checkpoint {path}: "
            f"{error}. Rerun without --resume to replace it."
        ) from error
    return path, turn_evaluations
