from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from dialogue_extractor.dialogue_extractor import ordered_dialogue_turns
from transcript_generation import SCENARIOS
from transcript_generation.storage import model_path_component

from .atomic_io import atomic_write_text
from .judging import build_judge_messages
from .models import (
    EVALUATOR_PROMPT_VERSION,
    JudgePanelMember,
    JudgeTurnEvaluation,
    TranscriptPanel,
    TranscriptSource,
    TurnEvaluation,
    decision_label,
    majority_label,
    validate_judge_panel,
)


CHECKPOINT_SCHEMA_VERSION = "tof-nof-multi-judge-checkpoint-v1"
SCENARIO_ORDER = tuple(scenario.name for scenario in SCENARIOS)


@dataclass(frozen=True)
class EvaluationCheckpointContext:
    output_dir: Path
    judges: tuple[JudgePanelMember, ...]


def evaluation_checkpoint_path(
    context: EvaluationCheckpointContext,
    *,
    panel: TranscriptPanel,
) -> Path:
    return (
        context.output_dir
        / "checkpoints"
        / model_path_component(panel.target_model)
        / f"{panel.item_id}.json"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_by_scenario(panel: TranscriptPanel) -> dict[str, TranscriptSource]:
    by_scenario = {
        source.transcript.scenario.name: source for source in panel.sources
    }
    if set(by_scenario) != set(SCENARIO_ORDER):
        raise ValueError("checkpoint panel must contain all four scenarios")
    return by_scenario


def _source_payload(panel: TranscriptPanel) -> list[dict[str, str]]:
    by_scenario = _source_by_scenario(panel)
    return [
        {
            "scenario": scenario,
            "path": str(by_scenario[scenario].path),
            "sha256": _sha256(by_scenario[scenario].path.read_bytes()),
        }
        for scenario in SCENARIO_ORDER
    ]


def _prompt_sha256(panel: TranscriptPanel) -> str:
    by_scenario = _source_by_scenario(panel)
    prompt_payload: list[dict[str, Any]] = []
    canonical = by_scenario[SCENARIO_ORDER[0]]
    canonical_turns = ordered_dialogue_turns(
        canonical.transcript,
        path=canonical.path,
    )
    prompt_payload.append(
        {
            "scope": "shared_turn_1",
            "messages": [
                message.model_dump(mode="json")
                for message in build_judge_messages(
                    transcript=canonical.transcript,
                    turn=canonical_turns[0],
                )
            ],
        }
    )
    for scenario in SCENARIO_ORDER:
        source = by_scenario[scenario]
        turns = ordered_dialogue_turns(source.transcript, path=source.path)
        for turn in turns[1:]:
            prompt_payload.append(
                {
                    "scope": f"{scenario}:turn_{turn.turn}",
                    "messages": [
                        message.model_dump(mode="json")
                        for message in build_judge_messages(
                            transcript=source.transcript,
                            turn=turn,
                        )
                    ],
                }
            )
    serialized = json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(serialized)


def _judge_panel_payload(
    context: EvaluationCheckpointContext,
) -> list[dict[str, Any]]:
    judges = validate_judge_panel(context.judges)
    return [
        {
            "provider": judge.provider,
            "endpoint": judge.endpoint,
            **asdict(judge.config),
        }
        for judge in judges
    ]


def _identity_payload(
    context: EvaluationCheckpointContext,
    *,
    panel: TranscriptPanel,
) -> dict[str, Any]:
    return {
        "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
        "prompt_sha256": _prompt_sha256(panel),
        "judges": _judge_panel_payload(context),
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


def _turn_from_payload(payload: object) -> TurnEvaluation:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint turn evaluation must be a JSON object")
    turn_fields = {field.name for field in fields(TurnEvaluation)}
    _require_exact_fields(
        payload,
        turn_fields,
        description="checkpoint turn evaluation",
    )
    raw_judges = payload["judge_evaluations"]
    if not isinstance(raw_judges, list):
        raise ValueError("checkpoint judge evaluations must be a list")
    judge_fields = {field.name for field in fields(JudgeTurnEvaluation)}
    judge_evaluations: list[JudgeTurnEvaluation] = []
    for raw_judge in raw_judges:
        if not isinstance(raw_judge, dict):
            raise ValueError("checkpoint judge evaluation must be an object")
        _require_exact_fields(
            raw_judge,
            judge_fields,
            description="checkpoint judge evaluation",
        )
        judge_evaluations.append(JudgeTurnEvaluation(**raw_judge))
    return TurnEvaluation(
        turn=payload["turn"],
        label=payload["label"],
        user_message=payload["user_message"],
        assistant_response=payload["assistant_response"],
        judge_evaluations=tuple(judge_evaluations),
    )


def _turns_from_payload(
    payload: object,
) -> dict[str, tuple[TurnEvaluation, ...]]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint scenario evaluations must be an object")
    if set(payload) != set(SCENARIO_ORDER):
        raise ValueError("checkpoint must contain exactly all four scenarios")
    result: dict[str, tuple[TurnEvaluation, ...]] = {}
    for scenario in SCENARIO_ORDER:
        raw_turns = payload[scenario]
        if not isinstance(raw_turns, list):
            raise ValueError("checkpoint scenario turns must be a list")
        result[scenario] = tuple(_turn_from_payload(turn) for turn in raw_turns)
    return result


def _validate_turn_evaluations(
    turns_by_scenario: Mapping[str, Sequence[TurnEvaluation]],
    *,
    context: EvaluationCheckpointContext,
    panel: TranscriptPanel,
) -> None:
    if set(turns_by_scenario) != set(SCENARIO_ORDER):
        raise ValueError("checkpoint must cover all four scenarios")
    source_by_scenario = _source_by_scenario(panel)
    judges = validate_judge_panel(context.judges)
    expected_judges = [judge.definition for judge in judges]
    populated = 0
    shared_turns: list[TurnEvaluation] = []
    for scenario in SCENARIO_ORDER:
        evaluations = list(turns_by_scenario[scenario])
        if len(evaluations) > 5:
            raise ValueError("checkpoint scenarios may contain at most five turns")
        if evaluations:
            populated += 1
            shared_turns.append(evaluations[0])
        expected_turns = tuple(range(1, len(evaluations) + 1))
        if tuple(evaluation.turn for evaluation in evaluations) != expected_turns:
            raise ValueError(
                "checkpoint turn evaluations must be a consecutive prefix"
            )
        source = source_by_scenario[scenario]
        transcript_turns = ordered_dialogue_turns(
            source.transcript,
            path=source.path,
        )
        for evaluation in evaluations:
            source_turn = transcript_turns[evaluation.turn - 1]
            if (
                evaluation.user_message != source_turn.user
                or evaluation.assistant_response != source_turn.assistant
            ):
                raise ValueError(
                    "checkpoint turn evidence does not match the source transcript"
                )
            if len(evaluation.judge_evaluations) != len(expected_judges):
                raise ValueError("checkpoint turns must contain exactly three judges")
            actual_judges = [
                judge.definition for judge in evaluation.judge_evaluations
            ]
            if actual_judges != expected_judges:
                raise ValueError("checkpoint judge identities or order changed")
            for judge in evaluation.judge_evaluations:
                if judge.label != decision_label(judge.decision_type):
                    raise ValueError(
                        "checkpoint judge label is inconsistent with decision type"
                    )
                if not judge.reason.strip():
                    raise ValueError("checkpoint judge reasons must be non-empty")
            expected_consensus = majority_label(
                [
                    judge.label
                    for judge in evaluation.judge_evaluations
                ]
            )
            if evaluation.label != expected_consensus:
                raise ValueError("checkpoint consensus label is not the majority")
    if populated not in (0, len(SCENARIO_ORDER)):
        raise ValueError("shared turn 1 must be stored for every scenario")
    if shared_turns and any(turn != shared_turns[0] for turn in shared_turns[1:]):
        raise ValueError("checkpoint shared turn 1 decisions differ across scenarios")


def write_evaluation_checkpoint(
    turns_by_scenario: Mapping[str, Sequence[TurnEvaluation]],
    *,
    context: EvaluationCheckpointContext,
    panel: TranscriptPanel,
) -> Path:
    _validate_turn_evaluations(
        turns_by_scenario,
        context=context,
        panel=panel,
    )
    path = evaluation_checkpoint_path(context, panel=panel)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "panel": {
            "item_id": panel.item_id,
            "target_model": panel.target_model,
            "sources": _source_payload(panel),
        },
        "evaluation": _identity_payload(context, panel=panel),
        "turn_evaluations": {
            scenario: [asdict(turn) for turn in turns_by_scenario[scenario]]
            for scenario in SCENARIO_ORDER
        },
    }
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    return path


def load_evaluation_checkpoint(
    *,
    context: EvaluationCheckpointContext,
    panel: TranscriptPanel,
) -> tuple[Path, dict[str, tuple[TurnEvaluation, ...]]] | None:
    path = evaluation_checkpoint_path(context, panel=panel)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must be a JSON object")
        _require_exact_fields(
            payload,
            {"schema_version", "panel", "evaluation", "turn_evaluations"},
            description="checkpoint",
        )
        if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema version is stale")
        expected_panel = {
            "item_id": panel.item_id,
            "target_model": panel.target_model,
            "sources": _source_payload(panel),
        }
        if payload["panel"] != expected_panel:
            raise ValueError("source transcript panel has changed")
        expected_evaluation = _identity_payload(context, panel=panel)
        actual_evaluation = payload["evaluation"]
        if not isinstance(actual_evaluation, dict):
            raise ValueError("checkpoint evaluation identity must be an object")
        _require_exact_fields(
            actual_evaluation,
            set(expected_evaluation),
            description="checkpoint evaluation identity",
        )
        if actual_evaluation["prompt_sha256"] != expected_evaluation["prompt_sha256"]:
            raise ValueError("judge prompt has changed")
        if actual_evaluation != expected_evaluation:
            raise ValueError("judge panel configuration has changed")
        turns = _turns_from_payload(payload["turn_evaluations"])
        _validate_turn_evaluations(
            turns,
            context=context,
            panel=panel,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Cannot resume from stale or invalid evaluation checkpoint {path}: "
            f"{error}. Rerun without --resume to replace it."
        ) from error
    return path, turns
