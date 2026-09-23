from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from dialogue_extractor.dialogue_extractor import ordered_dialogue_turns
from transcript_generation import load_transcript
from transcript_generation.storage import model_path_component
from .atomic_io import atomic_write_text
from .judging import build_judge_messages
from .models import (
    EVALUATOR_PROMPT_VERSION,
    JudgeConfig,
    JudgePanelMember,
    JudgeTurnEvaluation,
    SCENARIO_ORDER,
    TranscriptPanel,
    TranscriptSource,
    TurnEvaluation,
    decision_label,
    judge_key,
    normalize_panel_progress,
    validate_judge_panel,
)

CHECKPOINT_SCHEMA_VERSION = "tof-nof-independent-judge-checkpoint-v1"
OUTPUT_SCHEMA_VERSION = "tof-nof-independent-judge-results-v1"


@dataclass(frozen=True)
class EvaluationCheckpointContext:
    output_dir: Path
    judge: JudgePanelMember

    @property
    def namespace(self) -> Path:
        return self.output_dir / "judges" / judge_key(self.judge.definition)


@contextmanager
def judge_lock(namespace: Path, *, reader: bool = False) -> Iterator[None]:
    """All readers and writers cooperate through a persistent OS lock file.

    Never unlink it: replacing the inode could permit simultaneous writers.
    OS releases the advisory lock even on abrupt process termination.
    """
    namespace.mkdir(parents=True, exist_ok=True)
    with (namespace / ".lock").open("a+") as handle:
        try:
            fcntl.flock(
                handle, (fcntl.LOCK_SH if reader else fcntl.LOCK_EX) | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            raise RuntimeError(
                f"Judge namespace is busy (active writer/reader): {namespace}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def evaluation_checkpoint_path(
    context: EvaluationCheckpointContext, *, panel: TranscriptPanel
) -> Path:
    return (
        context.namespace
        / "checkpoints"
        / model_path_component(panel.target_model)
        / f"{panel.item_id}.json"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_by_scenario(panel: TranscriptPanel) -> dict[str, TranscriptSource]:
    by_scenario = {source.transcript.scenario.name: source for source in panel.sources}
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


def judge_payload(judge: JudgePanelMember) -> dict[str, Any]:
    validate_judge_panel((judge,))
    return {
        "provider": judge.provider,
        "endpoint": judge.endpoint,
        **asdict(judge.config),
        "allowed_openai_params": list(judge.config.allowed_openai_params),
    }


def judge_from_payload(payload: dict[str, Any]) -> JudgePanelMember:
    config = {
        key: value
        for key, value in payload.items()
        if key not in ("provider", "endpoint")
    }
    config["allowed_openai_params"] = tuple(config["allowed_openai_params"])
    judge = JudgePanelMember(
        payload["provider"], JudgeConfig(**config), payload["endpoint"]
    )
    if payload != judge_payload(judge):
        raise ValueError("corrupt judge configuration")
    return judge


def manifest_payload(context: EvaluationCheckpointContext) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
        "label_source": "single_judge",
        "judge": judge_payload(context.judge),
    }


def validate_manifest(
    context: EvaluationCheckpointContext, *, create: bool = False
) -> None:
    path = context.namespace / "manifest.json"
    expected = manifest_payload(context)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(
                f"Judge configuration mismatch in {path}; choose another output root"
            )
    elif any((context.namespace / "checkpoints").glob("*/*.json")):
        raise ValueError(
            f"Missing manifest for stored checkpoints in {context.namespace}"
        )
    elif create:
        atomic_write_text(
            path, json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
        )


def context_from_namespace(namespace: Path) -> EvaluationCheckpointContext:
    payload = json.loads((namespace / "manifest.json").read_text())
    context = EvaluationCheckpointContext(
        namespace.parent.parent, judge_from_payload(payload["judge"])
    )
    if context.namespace != namespace:
        raise ValueError("judge namespace identity mismatch")
    validate_manifest(context)
    return context


def _identity_payload(
    context: EvaluationCheckpointContext, *, panel: TranscriptPanel
) -> dict[str, Any]:
    return {
        "evaluator_prompt_version": EVALUATOR_PROMPT_VERSION,
        "prompt_sha256": _prompt_sha256(panel),
        "judge": judge_payload(context.judge),
    }


def _turn_from_payload(payload: Any) -> TurnEvaluation:
    if not isinstance(payload, dict) or set(payload) != {
        f.name for f in fields(TurnEvaluation)
    }:
        raise ValueError("corrupt checkpoint turn fields")
    raw = payload["judgment"]
    if not isinstance(raw, dict) or set(raw) != {
        f.name for f in fields(JudgeTurnEvaluation)
    }:
        raise ValueError("corrupt checkpoint judgment fields")
    return TurnEvaluation(**{**payload, "judgment": JudgeTurnEvaluation(**raw)})


def validate_turn_evaluations(
    turns_by_scenario: Mapping[str, Sequence[TurnEvaluation]],
    *,
    context: EvaluationCheckpointContext,
    panel: TranscriptPanel,
) -> dict[str, list[TurnEvaluation]]:
    normalized = normalize_panel_progress(turns_by_scenario)
    sources = _source_by_scenario(panel)
    for scenario, evaluations in normalized.items():
        source = sources[scenario]
        turns = ordered_dialogue_turns(source.transcript, path=source.path)
        for evaluation in evaluations:
            if type(evaluation.turn) is not int:
                raise ValueError("corrupt turn number")
            turn = turns[evaluation.turn - 1]
            if (
                evaluation.user_message != turn.user
                or evaluation.assistant_response != turn.assistant
            ):
                raise ValueError(
                    "corrupt checkpoint evidence does not match source transcript"
                )
            judge = evaluation.judgment
            if judge.definition != context.judge.definition:
                raise ValueError("checkpoint judge identity mismatch")
            if type(judge.label) is not int or judge.label != decision_label(
                judge.decision_type
            ):
                raise ValueError("corrupt judge label inconsistent with decision type")
            if not isinstance(judge.reason, str) or not judge.reason.strip():
                raise ValueError("checkpoint reasons must be non-empty")
            if (
                not isinstance(judge.judge_output, str)
                or not isinstance(judge.judge_metadata, dict)
                or not isinstance(judge.judge_raw_response, dict)
            ):
                raise ValueError("corrupt judgment output/metadata/raw response")
    return normalized


def checkpoint_payload(
    turns_by_scenario: Mapping[str, Sequence[TurnEvaluation]],
    *,
    context: EvaluationCheckpointContext,
    panel: TranscriptPanel,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    normalized = validate_turn_evaluations(
        turns_by_scenario, context=context, panel=panel
    )
    canonical = normalized[SCENARIO_ORDER[0]]
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "panel": {
            "item_id": panel.item_id,
            "target_model": panel.target_model,
            "sources": _source_payload(panel),
        },
        "evaluation": _identity_payload(context, panel=panel),
        "shared_turn_1": asdict(canonical[0]) if canonical else None,
        "scenario_turns": {
            s: [asdict(t) for t in normalized[s][1:]] for s in SCENARIO_ORDER
        },
    }
    if provenance is not None:
        payload["import_provenance"] = provenance
    return payload


def write_evaluation_checkpoint(
    turns_by_scenario: Mapping[str, Sequence[TurnEvaluation]],
    *,
    context: EvaluationCheckpointContext,
    panel: TranscriptPanel,
    provenance: dict[str, str] | None = None,
) -> Path:
    payload = checkpoint_payload(
        turns_by_scenario, context=context, panel=panel, provenance=provenance
    )
    path = evaluation_checkpoint_path(context, panel=panel)
    # Preserve import provenance on later live resume.
    if provenance is None and path.exists():
        previous = json.loads(path.read_text())
        if "import_provenance" in previous:
            payload["import_provenance"] = previous["import_provenance"]
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def decode_checkpoint(
    payload: Any, *, context: EvaluationCheckpointContext, panel: TranscriptPanel
) -> dict[str, tuple[TurnEvaluation, ...]]:
    expected_fields = {
        "schema_version",
        "panel",
        "evaluation",
        "shared_turn_1",
        "scenario_turns",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) - {"import_provenance"} != expected_fields
    ):
        raise ValueError("corrupt checkpoint fields")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version mismatch")
    if payload["panel"] != {
        "item_id": panel.item_id,
        "target_model": panel.target_model,
        "sources": _source_payload(panel),
    }:
        raise ValueError("source transcript panel has changed")
    expected = _identity_payload(context, panel=panel)
    if payload["evaluation"].get("prompt_sha256") != expected["prompt_sha256"]:
        raise ValueError("judge prompt has changed")
    if payload["evaluation"] != expected:
        raise ValueError("judge configuration mismatch")
    if "import_provenance" in payload:
        provenance = payload["import_provenance"]
        if (
            not isinstance(provenance, dict)
            or set(provenance) != {"source_checkpoint", "sha256"}
            or not all(isinstance(v, str) and v for v in provenance.values())
        ):
            raise ValueError("corrupt import provenance")
    shared = (
        ()
        if payload["shared_turn_1"] is None
        else (_turn_from_payload(payload["shared_turn_1"]),)
    )
    raw = payload["scenario_turns"]
    if (
        not isinstance(raw, dict)
        or set(raw) != set(SCENARIO_ORDER)
        or any(not isinstance(v, list) for v in raw.values())
    ):
        raise ValueError("corrupt scenario prefixes")
    if not shared and any(raw.values()):
        raise ValueError(
            "corrupt checkpoint: scenario turns require canonical shared turn 1"
        )
    turns = {
        s: shared + tuple(_turn_from_payload(t) for t in raw[s]) for s in SCENARIO_ORDER
    }
    validate_turn_evaluations(turns, context=context, panel=panel)
    return turns


def load_evaluation_checkpoint(
    *, context: EvaluationCheckpointContext, panel: TranscriptPanel
):
    path = evaluation_checkpoint_path(context, panel=panel)
    if not path.exists():
        return None
    try:
        turns = decode_checkpoint(
            json.loads(path.read_text()), context=context, panel=panel
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(
            f"Invalid checkpoint {path}: {error}. Choose another output root."
        ) from error
    return path, turns


def panel_from_payload(payload: dict[str, Any]) -> TranscriptPanel:
    from .evaluation import _sources_by_scenario

    raw = payload["panel"]
    panel = TranscriptPanel(
        raw["item_id"],
        raw["target_model"],
        tuple(
            TranscriptSource(
                Path(source["path"]), load_transcript(Path(source["path"]))
            )
            for source in raw["sources"]
        ),
    )
    _sources_by_scenario(panel)
    return panel


def stored_panels(context: EvaluationCheckpointContext):
    """Caller holds the namespace lock; validate every checkpoint, including partial work."""
    validate_manifest(context)
    result = []
    for path in sorted((context.namespace / "checkpoints").glob("*/*.json")):
        payload = json.loads(path.read_text())
        panel = panel_from_payload(payload)
        if path != evaluation_checkpoint_path(context, panel=panel):
            raise ValueError(f"checkpoint path identity mismatch: {path}")
        turns = decode_checkpoint(payload, context=context, panel=panel)
        result.append((panel, turns, payload))
    return result
