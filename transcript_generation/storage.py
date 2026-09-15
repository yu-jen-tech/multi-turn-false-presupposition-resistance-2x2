from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from .models import (
    BaselineRecord,
    ChatResult,
    DatasetItem,
    DialogueTurn,
    PushbackRecord,
    PushbackSource,
    Scenario,
    TRANSCRIPT_SCHEMA_VERSION,
    DatasetProvenance,
    TranscriptRecord,
)
from .prompts import TURN_STRATEGIES
from .validation import semantic_validation_reasons


def model_path_component(model_name: str) -> str:
    component = model_name.strip().replace("/", "__").replace("\\", "__")
    return component or "unknown-model"


def transcript_path(
    output_dir: Path,
    *,
    scenario: Scenario,
    target_model: str,
    item_id: str,
) -> Path:
    return (
        output_dir
        / model_path_component(target_model)
        / scenario.name
        / f"{item_id}.json"
    )


def load_transcript(path: Path) -> TranscriptRecord:
    data = path.read_text(encoding="utf-8")
    return TranscriptRecord.model_validate_json(data)


def copy_pushback_record(
    record: PushbackRecord,
    **updates: Any,
) -> PushbackRecord:
    return record.model_copy(update=updates)


def pushback_has_semantic_validation(pushback: PushbackRecord) -> bool:
    semantic = pushback.validation.semantic
    surface_style = pushback.validation.surface_style
    return (
        pushback.validation.passed
        and semantic is not None
        and semantic.preserves_baseline_semantics is True
        and surface_style is not None
        and surface_style.matches_requested_style
        and surface_style.preserves_protected_anchors
        and not semantic_validation_reasons(semantic)
    )


def baseline_has_semantic_validation(baseline: BaselineRecord) -> bool:
    semantic = baseline.validation.semantic
    return (
        baseline.validation.passed
        and semantic is not None
        and not semantic_validation_reasons(semantic)
    )


def complete_pushbacks_from_transcript(
    transcript: TranscriptRecord,
    *,
    path: Path,
    require_semantic_validation: bool = True,
) -> list[PushbackRecord] | None:
    if transcript.schema_version != TRANSCRIPT_SCHEMA_VERSION:
        return None
    expected_turns = sorted(TURN_STRATEGIES)
    if len(transcript.pushbacks) != len(expected_turns):
        return None
    if [pushback.turn for pushback in transcript.pushbacks] != expected_turns:
        return None
    if len(transcript.turns) < len(expected_turns) + 1:
        return None

    for pushback, turn in zip(transcript.pushbacks, transcript.turns[1:]):
        if (
            turn.turn != pushback.turn
            or turn.user != pushback.text
            or turn.strategy != pushback.strategy
        ):
            raise RuntimeError(
                "Cannot reuse pushbacks because transcript turns and pushback records "
                f"do not match in {path}"
            )
        if pushback.baseline is None or not baseline_has_semantic_validation(
            pushback.baseline
        ):
            return None
        if require_semantic_validation and not pushback_has_semantic_validation(
            pushback
        ):
            return None
    return list(transcript.pushbacks)


def resumable_transcript_exists(
    path: Path,
    *,
    dataset_provenance: DatasetProvenance | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        transcript = load_transcript(path)
    except ValidationError as error:
        raise RuntimeError(
            "Cannot resume from an existing transcript that is not the current schema "
            f"and list-native format: {path}"
        ) from error
    if dataset_provenance is not None:
        if transcript.dataset_provenance != dataset_provenance:
            raise RuntimeError(
                "Cannot resume from a transcript generated with a different dataset "
                f"provenance: {path}"
            )
    if complete_pushbacks_from_transcript(transcript, path=path) is None:
        raise RuntimeError(
            "Cannot resume from an existing transcript that is not the current schema and "
            "does not contain complete, semantically validated baseline/style "
            f"pushbacks: {path}. "
            "Rerun with --overwrite or remove the stale transcript."
        )
    return True


def pushback_signature(
    pushbacks: Sequence[PushbackRecord],
) -> list[tuple[int, str, str, str]]:
    return [
        (
            pushback.turn,
            pushback.strategy,
            pushback.strategy_instruction,
            pushback.text,
        )
        for pushback in pushbacks
    ]


def baseline_signature(
    baselines: Sequence[BaselineRecord],
) -> list[tuple[int, str, str, str]]:
    return [
        (
            baseline.turn,
            baseline.strategy,
            baseline.strategy_instruction,
            baseline.text,
        )
        for baseline in baselines
    ]


def _api_configuration_signature(metadata: Any) -> tuple[Any, ...]:
    return (
        metadata.purpose,
        metadata.model,
        metadata.temperature,
        metadata.seed,
        metadata.endpoint,
    )


def _baseline_validation_signature(
    baselines: Sequence[BaselineRecord],
) -> list[tuple[Any, ...]]:
    signatures: list[tuple[Any, ...]] = []
    for baseline in baselines:
        semantic = baseline.validation.semantic
        validator_metadata = baseline.semantic_validation_api_metadata
        signatures.append(
            (
                baseline.validation.passed,
                tuple(baseline.validation.reasons),
                semantic.model_dump(mode="json") if semantic is not None else None,
                _api_configuration_signature(baseline.api_metadata),
                (
                    _api_configuration_signature(validator_metadata)
                    if validator_metadata is not None
                    else None
                ),
            )
        )
    return signatures


def pushbacks_with_source(
    pushbacks: Sequence[PushbackRecord],
    *,
    source: PushbackSource,
    source_scenario: str,
    source_target_model: str,
    source_transcript_path: Path,
) -> list[PushbackRecord]:
    return [
        copy_pushback_record(
            pushback,
            pushback_source=source,
            source_scenario=source_scenario,
            source_target_model=source_target_model,
            source_transcript_path=str(source_transcript_path),
        )
        for pushback in pushbacks
    ]


def reusable_pushbacks(
    output_dir: Path,
    *,
    scenario: Scenario,
    item_id: str,
) -> list[PushbackRecord] | None:
    if not scenario.shares_pushbacks_across_target_models:
        return None

    if not output_dir.exists():
        return None

    existing: list[tuple[Path, TranscriptRecord, list[PushbackRecord]]] = []
    for path in sorted(output_dir.glob(f"*/{scenario.name}/{item_id}.json")):
        transcript = load_transcript(path)
        if transcript.item.id != item_id or transcript.scenario.name != scenario.name:
            raise RuntimeError(
                "Cannot reuse pushbacks because transcript metadata does not match "
                f"its path: {path}"
            )
        pushbacks = complete_pushbacks_from_transcript(transcript, path=path)
        if pushbacks is None:
            continue
        existing.append((path, transcript, pushbacks))

    if not existing:
        return None

    first_path, first_transcript, first_pushbacks = existing[0]
    first_signature = pushback_signature(first_pushbacks)
    for path, _, pushbacks in existing[1:]:
        if pushback_signature(pushbacks) != first_signature:
            raise RuntimeError(
                "Cannot resume generation because existing target-model transcripts "
                f"for {item_id} in {scenario.name} have different pushbacks: "
                f"{first_path} and {path}"
            )

    return pushbacks_with_source(
        first_pushbacks,
        source="reused_from_without_simulator_history",
        source_scenario=first_transcript.scenario.name,
        source_target_model=first_transcript.target_model,
        source_transcript_path=first_path,
    )


def reusable_baselines(
    output_dir: Path,
    *,
    scenarios: Sequence[Scenario],
    item_id: str,
) -> list[BaselineRecord] | None:
    reusable = reusable_baselines_with_source(
        output_dir,
        scenarios=scenarios,
        item_id=item_id,
    )
    return reusable[0] if reusable is not None else None


def reusable_baselines_with_source(
    output_dir: Path,
    *,
    scenarios: Sequence[Scenario],
    item_id: str,
) -> tuple[list[BaselineRecord], Path] | None:
    historyless_scenarios = [
        scenario
        for scenario in scenarios
        if scenario.shares_pushbacks_across_target_models
    ]
    existing: list[tuple[Path, list[BaselineRecord]]] = []
    for scenario in historyless_scenarios:
        if not output_dir.exists():
            continue
        for path in sorted(output_dir.glob(f"*/{scenario.name}/{item_id}.json")):
            transcript = load_transcript(path)
            if (
                transcript.item.id != item_id
                or transcript.scenario.name != scenario.name
            ):
                raise RuntimeError(
                    "Cannot reuse baselines because transcript metadata does not "
                    f"match its path: {path}"
                )
            pushbacks = complete_pushbacks_from_transcript(transcript, path=path)
            if pushbacks is None:
                continue
            baselines = [
                pushback.baseline
                for pushback in pushbacks
                if pushback.baseline is not None
            ]
            if len(baselines) != len(pushbacks):
                continue
            existing.append((path, baselines))

    if not existing:
        return None

    first_path, first_baselines = existing[0]
    first_signature = baseline_signature(first_baselines)
    first_validation_signature = _baseline_validation_signature(first_baselines)
    for path, baselines in existing[1:]:
        if baseline_signature(baselines) != first_signature:
            raise RuntimeError(
                "Cannot resume generation because existing transcripts have "
                f"different baselines for {item_id}: {first_path} and {path}"
            )
        if _baseline_validation_signature(baselines) != first_validation_signature:
            raise RuntimeError(
                "Cannot resume generation because existing transcripts have "
                f"conflicting baseline validation/configuration for {item_id}: "
                f"{first_path} and {path}"
            )
    return first_baselines, first_path


def reusable_first_response(
    output_dir: Path,
    *,
    scenarios: Sequence[Scenario],
    target_model: str,
    item_id: str,
) -> ChatResult | None:
    reusable = reusable_first_response_with_source(
        output_dir,
        scenarios=scenarios,
        target_model=target_model,
        item_id=item_id,
    )
    return reusable[0] if reusable is not None else None


def reusable_first_response_with_source(
    output_dir: Path,
    *,
    scenarios: Sequence[Scenario],
    target_model: str,
    item_id: str,
) -> tuple[ChatResult, Path] | None:
    existing: list[tuple[Path, DialogueTurn]] = []
    for scenario in scenarios:
        path = transcript_path(
            output_dir,
            scenario=scenario,
            target_model=target_model,
            item_id=item_id,
        )
        if not path.exists():
            continue
        transcript = load_transcript(path)
        if transcript.schema_version != TRANSCRIPT_SCHEMA_VERSION:
            continue
        if not transcript.turns:
            continue
        existing.append((path, transcript.turns[0]))

    if not existing:
        return None

    first_path, first_turn = existing[0]
    for path, turn in existing[1:]:
        if turn.assistant != first_turn.assistant:
            raise RuntimeError(
                "Cannot resume generation because existing scenario transcripts "
                f"for {item_id} with {target_model} have different first answers: "
                f"{first_path} and {path}"
            )

    return (
        ChatResult(
            content=first_turn.assistant,
            metadata=first_turn.api_metadata,
            raw_response=first_turn.raw_response,
        ),
        first_path,
    )


def select_items(
    items: Sequence[DatasetItem],
    *,
    item_index: int,
    num_questions: int | None,
) -> list[DatasetItem]:
    if not items:
        raise SystemExit("No dataset items found.")
    if item_index < 0 or item_index >= len(items):
        raise SystemExit(f"--item-index must be between 0 and {len(items) - 1}")
    if num_questions is not None and num_questions <= 0:
        raise SystemExit("--num-questions must be greater than 0")

    end_index = (
        len(items)
        if num_questions is None
        else item_index + num_questions
    )
    if end_index > len(items):
        raise SystemExit(
            f"--item-index + --num-questions must not exceed {len(items)} items"
        )
    return list(items[item_index:end_index])


def load_dataset(
    dataset_jsonl_path: Path,
) -> tuple[list[DatasetItem], DatasetProvenance]:
    raw_data = dataset_jsonl_path.read_bytes()
    try:
        text = raw_data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Cannot decode dataset as UTF-8: {dataset_jsonl_path}"
        ) from error

    items: list[DatasetItem] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {dataset_jsonl_path} on line {line_number}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected a JSON object in {dataset_jsonl_path} on line {line_number}"
            )
        try:
            item = DatasetItem.model_validate(payload)
        except ValidationError as error:
            raise ValueError(
                f"Invalid dataset record in {dataset_jsonl_path} on line "
                f"{line_number}: {error}"
            ) from error
        if item.id in seen_ids:
            raise ValueError(
                f"Dataset {dataset_jsonl_path} contains duplicate id {item.id!r} "
                f"on line {line_number}"
            )
        seen_ids.add(item.id)
        items.append(item)

    if not items:
        raise ValueError(f"Dataset contains no records: {dataset_jsonl_path}")

    provenance = DatasetProvenance(
        dataset_path=str(dataset_jsonl_path),
        dataset_sha256=hashlib.sha256(raw_data).hexdigest(),
        record_count=len(items),
    )
    return items, provenance


def load_items(dataset_jsonl_path: Path) -> list[DatasetItem]:
    return load_dataset(dataset_jsonl_path)[0]


def write_transcript(
    transcript: TranscriptRecord,
    output_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    path = transcript_path(
        output_dir,
        scenario=transcript.scenario,
        target_model=transcript.target_model,
        item_id=transcript.item.id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    payload = transcript.model_dump(mode="json")
    for pushback in payload["pushbacks"]:
        if pushback.get("baseline") is None:
            pushback.pop("baseline", None)
        for record in (pushback, pushback.get("baseline")):
            if record is None:
                continue
            validation = record.get("validation", {})
            if validation.get("surface_style") is None:
                validation.pop("surface_style", None)
            semantic = record.get("validation", {}).get("semantic")
            if (
                semantic is not None
                and semantic.get("preserves_baseline_semantics") is None
            ):
                semantic.pop("preserves_baseline_semantics", None)
            if (
                semantic is not None
                and semantic.get("matches_requested_style") is None
            ):
                semantic.pop("matches_requested_style", None)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return path
