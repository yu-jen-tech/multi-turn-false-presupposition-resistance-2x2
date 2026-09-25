"""Offline migration of validated v2 legacy judgments without relabeling."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

from .atomic_io import atomic_write_text
from .legacy_schema import majority_label, validate_judge_definitions, validate_legacy_config
from .models import (
    EVALUATOR_PROMPT_VERSION,
    JudgeDefinition,
    JudgeTurnEvaluation,
    SCENARIO_ORDER,
    TurnEvaluation,
    resolve_judges,
)
from .reporting import export_context
from .storage import (
    EvaluationCheckpointContext,
    _prompt_sha256,
    _sha256,
    _source_payload,
    checkpoint_payload,
    evaluation_checkpoint_path,
    judge_from_payload,
    judge_lock,
    load_evaluation_checkpoint,
    panel_from_payload,
    stored_panels,
    validate_manifest,
    validate_turn_evaluations,
)


def _read_legacy(path: Path, selected: Sequence[JudgeDefinition]):
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if (
        set(payload) != {"schema_version", "panel", "evaluation", "turn_evaluations"}
        or payload["schema_version"] != "tof-nof-multi-judge-checkpoint-v1"
    ):
        raise ValueError(f"Invalid legacy schema: {path}")
    panel = panel_from_payload(payload)
    if payload["panel"]["sources"] != _source_payload(panel):
        raise ValueError(f"Legacy source transcript changed: {path}")
    evaluation = payload["evaluation"]
    if set(evaluation) != {"evaluator_prompt_version", "prompt_sha256", "judges"}:
        raise ValueError("Invalid legacy evaluation fields")
    if evaluation["evaluator_prompt_version"] != EVALUATOR_PROMPT_VERSION:
        raise ValueError("Legacy rubric is not v2; cannot import into this store")
    if evaluation["prompt_sha256"] != _prompt_sha256(panel):
        raise ValueError("Legacy judge prompt hash changed")
    configs = evaluation["judges"]
    identities = [JudgeDefinition(c["provider"], c["model"]) for c in configs]
    validate_judge_definitions(identities)
    raw_scenarios = payload["turn_evaluations"]
    if set(raw_scenarios) != set(SCENARIO_ORDER):
        raise ValueError("Legacy checkpoint must contain all scenarios")
    extracted: dict[JudgeDefinition, dict[str, list[TurnEvaluation]]] = {
        identity: {s: [] for s in SCENARIO_ORDER} for identity in identities
    }
    for scenario, turns in raw_scenarios.items():
        if not isinstance(turns, list):
            raise ValueError("Invalid legacy scenario turns")
        for turn in turns:
            if set(turn) != {
                "turn",
                "label",
                "user_message",
                "assistant_response",
                "judge_evaluations",
            }:
                raise ValueError("Invalid legacy turn fields")
            records = [JudgeTurnEvaluation(**j) for j in turn["judge_evaluations"]]
            validate_judge_definitions([r.definition for r in records])
            if type(turn["label"]) is not int or turn["label"] != majority_label(
                [r.label for r in records]
            ):
                raise ValueError("Legacy majority is inconsistent")
            for record in records:
                extracted[record.definition][scenario].append(
                    TurnEvaluation(
                        turn["turn"],
                        turn["user_message"],
                        turn["assistant_response"],
                        record,
                    )
                )
    # Validate every original judge, including excluded Mistral, without imposing
    # an ordering or constructing a provider. Repeated shared turn must match exactly.
    from .models import JudgeConfig, JudgePanelMember

    for identity, config in zip(identities, configs):
        settings = {
            k: v for k, v in config.items() if k not in ("provider", "endpoint")
        }
        settings["allowed_openai_params"] = tuple(settings["allowed_openai_params"])
        judge = JudgePanelMember(
            identity.provider, JudgeConfig(**settings), config["endpoint"]
        )
        validate_legacy_config(judge.provider, judge.config)
        context = EvaluationCheckpointContext(Path("."), judge)
        validate_turn_evaluations(extracted[identity], context=context, panel=panel)
    provenance = {"source_checkpoint": str(path), "sha256": _sha256(raw_bytes)}
    return [
        (
            panel,
            judge_from_payload(configs[identities.index(identity)]),
            extracted[identity],
            provenance,
        )
        for identity in selected
    ]


def import_legacy(
    input_dir: Path, output_dir: Path, *, aliases: Sequence[str] | None = None
) -> list[Path]:
    # Qwen is the only legacy judge still supported for independent results.
    selected = resolve_judges(("qwen3.8",) if aliases is None else aliases)
    if any(identity.model != "openai/qwen3.8" for identity in selected):
        raise ValueError("Only qwen3.8 can be imported from the legacy panel")
    paths = sorted((input_dir / "checkpoints").glob("*/*.json"))
    if not paths:
        raise ValueError(f"No legacy checkpoints found under {input_dir}")
    batch = [entry for path in paths for entry in _read_legacy(path, selected)]
    contexts: dict[Path, EvaluationCheckpointContext] = {}
    staged: dict[Path, tuple[EvaluationCheckpointContext, Any, dict[str, Any]]] = {}
    # Preflight all source identities and configurations before writing any destination data.
    for panel, judge, turns, provenance in batch:
        context = EvaluationCheckpointContext(output_dir, judge)
        previous = contexts.setdefault(context.namespace, context)
        if previous.judge != judge:
            raise ValueError("Legacy batch contains incompatible judge configurations")
        path = evaluation_checkpoint_path(context, panel=panel)
        payload = checkpoint_payload(
            turns, context=context, panel=panel, provenance=provenance
        )
        if path in staged:
            raise ValueError(f"Duplicate legacy panel: {path}")
        staged[path] = context, panel, payload
    with ExitStack() as stack:
        for context in contexts.values():
            stack.enter_context(judge_lock(context.namespace))
            validate_manifest(context)
            stored_panels(context)
        for path, (context, panel, payload) in staged.items():
            if path.exists():
                load_evaluation_checkpoint(context=context, panel=panel)
                existing = json.loads(path.read_text())
                if {k: v for k, v in existing.items() if k != "import_provenance"} != {
                    k: v for k, v in payload.items() if k != "import_provenance"
                }:
                    raise ValueError(
                        f"Conflicting destination checkpoint; no replacement: {path}"
                    )
        for context in contexts.values():
            validate_manifest(context, create=True)
        for path, (_, _, payload) in staged.items():
            if not path.exists():
                atomic_write_text(
                    path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
                )
        return list(staged) + [
            path for context in contexts.values() for path in export_context(context)
        ]
