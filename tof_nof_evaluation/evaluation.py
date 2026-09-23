from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from dialogue_extractor.dialogue_extractor import (
    ordered_dialogue_turns,
    transcript_files,
)
from transcript_generation import TranscriptRecord, load_transcript
from transcript_generation.storage import model_path_component

from .judging import judge_turn
from .metrics import calculate_tof, calculate_trajectory_metrics
from .models import (
    JudgePanelMember,
    JudgeClient,
    ProgressCallback,
    SCENARIO_ORDER,
    TranscriptPanel,
    TranscriptScore,
    TranscriptSource,
    TurnEvaluation,
    normalize_panel_progress,
    judge_key,
    validate_judge_panel,
)
from .storage import (
    EvaluationCheckpointContext,
    evaluation_checkpoint_path,
    load_evaluation_checkpoint,
    write_evaluation_checkpoint,
    judge_lock,
    stored_panels,
    validate_manifest,
    validate_turn_evaluations,
)


def _sources_by_scenario(
    panel: TranscriptPanel,
) -> dict[str, TranscriptSource]:
    by_scenario: dict[str, TranscriptSource] = {}
    for source in panel.sources:
        scenario = source.transcript.scenario.name
        if scenario in by_scenario:
            raise ValueError(
                f"Panel {panel.item_id}/{panel.target_model} contains duplicate "
                f"scenario {scenario!r}"
            )
        by_scenario[scenario] = source
    if set(by_scenario) != set(SCENARIO_ORDER):
        missing = sorted(set(SCENARIO_ORDER) - set(by_scenario))
        extra = sorted(set(by_scenario) - set(SCENARIO_ORDER))
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ValueError(
            f"Panel {panel.item_id}/{panel.target_model} must contain exactly all "
            f"four scenarios ({'; '.join(details)})"
        )

    reference: TranscriptRecord | None = None
    first_user: str | None = None
    first_assistant: str | None = None
    for scenario in SCENARIO_ORDER:
        source = by_scenario[scenario]
        transcript = source.transcript
        if transcript.item.id != panel.item_id:
            raise ValueError("panel item ID does not match a source transcript")
        if transcript.target_model != panel.target_model:
            raise ValueError("panel target model does not match a source transcript")
        turns = ordered_dialogue_turns(transcript, path=source.path)
        if reference is None:
            reference = transcript
            first_user = turns[0].user
            first_assistant = turns[0].assistant
            continue
        if (
            transcript.item.question != reference.item.question
            or transcript.item.presuppositions != reference.item.presuppositions
            or transcript.item.corrections != reference.item.corrections
        ):
            raise ValueError(
                f"Panel {panel.item_id}/{panel.target_model} factual context differs "
                "across scenarios"
            )
        if turns[0].user != first_user:
            raise ValueError(
                f"Panel {panel.item_id}/{panel.target_model} shared turn 1 user text "
                "must be exactly identical across all four scenarios"
            )
        if turns[0].assistant != first_assistant:
            raise ValueError(
                f"Panel {panel.item_id}/{panel.target_model} shared turn 1 assistant "
                "response must be exactly identical across all four scenarios"
            )
    return by_scenario


def _judge_single_turn(
    *, transcript, turn_index, source_path, judge, progress
) -> TurnEvaluation:
    turn = ordered_dialogue_turns(transcript, path=source_path)[turn_index - 1]
    try:
        result = judge_turn(
            transcript=transcript, turn=turn, judge=judge, progress=progress
        )
    except Exception as error:
        raise RuntimeError(
            f"Judge {judge_key(judge.definition)} failed at {transcript.item.id}/"
            f"{transcript.target_model}/{transcript.scenario.name}/turn {turn_index}: {error}"
        ) from error
    return TurnEvaluation(turn.turn, turn.user, turn.assistant, result)


def evaluate_transcript_panel(
    panel: TranscriptPanel,
    *,
    judge: JudgePanelMember,
    progress: ProgressCallback | None = None,
    completed_turn_evaluations: Mapping[str, Sequence[TurnEvaluation]] | None = None,
    turn_recorded: Callable[[Mapping[str, Sequence[TurnEvaluation]]], None]
    | None = None,
) -> list[TranscriptScore]:
    """Evaluate one complete item x target panel with one independent judge."""
    validate_judge_panel((judge,))
    sources = _sources_by_scenario(panel)
    completed = normalize_panel_progress(completed_turn_evaluations)
    validate_turn_evaluations(
        completed, context=EvaluationCheckpointContext(Path("."), judge), panel=panel
    )

    if not any(completed.values()):
        canonical = sources[SCENARIO_ORDER[0]]
        if progress is not None:
            progress("Judging shared turn 1/17...")
        shared_turn = _judge_single_turn(
            transcript=canonical.transcript,
            turn_index=1,
            source_path=canonical.path,
            judge=judge,
            progress=progress,
        )
        for scenario in SCENARIO_ORDER:
            completed[scenario].append(shared_turn)
        if turn_recorded is not None:
            turn_recorded(completed)

    unique_turn_number = 1 + sum(
        max(0, len(evaluations) - 1) for evaluations in completed.values()
    )
    for scenario in SCENARIO_ORDER:
        source = sources[scenario]
        for turn_index in range(len(completed[scenario]) + 1, 6):
            unique_turn_number += 1
            if progress is not None:
                progress(
                    f"Judging unique turn {unique_turn_number}/17 "
                    f"({scenario}, turn {turn_index})..."
                )
            evaluation = _judge_single_turn(
                transcript=source.transcript,
                turn_index=turn_index,
                source_path=source.path,
                judge=judge,
                progress=progress,
            )
            completed[scenario].append(evaluation)
            if turn_recorded is not None:
                turn_recorded(completed)

    return scores_from_progress(panel, completed, judge=judge)


def scores_from_progress(
    panel, completed, *, judge: JudgePanelMember
) -> list[TranscriptScore]:
    sources = _sources_by_scenario(panel)
    scores: list[TranscriptScore] = []
    for scenario in SCENARIO_ORDER:
        evaluations = tuple(completed[scenario])
        if len(evaluations) != 5:
            continue
        labels = tuple(e.label for e in evaluations)
        metrics = calculate_trajectory_metrics(labels)
        transcript = sources[scenario].transcript
        scores.append(
            TranscriptScore(
                judge=judge.definition,
                item_id=transcript.item.id,
                question=transcript.item.question,
                presuppositions=list(transcript.item.presuppositions),
                corrections=list(transcript.item.corrections),
                target_model=transcript.target_model,
                target_model_component=model_path_component(transcript.target_model),
                scenario=scenario,
                labels=labels,
                tof=calculate_tof(labels),
                nof=metrics.nof,
                initial_correct=metrics.initial_correct,
                final_correct=metrics.final_correct,
                downward_flips=metrics.downward_flips,
                upward_flips=metrics.upward_flips,
                turn_evaluations=evaluations,
            )
        )
    return scores


def load_transcript_panels(
    transcript_dir: Path,
    *,
    target_models: set[str] | None = None,
    item_ids: set[str] | None = None,
    required_target_models: set[str] | None = None,
    required_item_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[TranscriptPanel]:
    grouped: dict[tuple[str, str], list[TranscriptSource]] = defaultdict(list)
    for path in transcript_files(transcript_dir):
        transcript = load_transcript(path)
        target_component = model_path_component(transcript.target_model)
        if (
            target_models is not None
            and transcript.target_model not in target_models
            and target_component not in target_models
        ):
            continue
        if item_ids is not None and transcript.item.id not in item_ids:
            continue
        grouped[(transcript.target_model, transcript.item.id)].append(
            TranscriptSource(path=path, transcript=transcript)
        )

    if (required_target_models is None) != (required_item_ids is None):
        raise ValueError(
            "required target models and item IDs must be provided together"
        )
    if required_target_models is not None and required_item_ids is not None:
        selected_item_ids = sorted(required_item_ids)
        if limit is not None:
            selected_item_ids = selected_item_ids[:limit]
        keys = sorted(
            (
                (target_model, item_id)
                for target_model in required_target_models
                for item_id in selected_item_ids
            ),
            key=lambda key: (key[0], key[1]),
        )
    else:
        selected_item_ids = sorted({item_id for _, item_id in grouped})
        if limit is not None:
            selected_item_ids = selected_item_ids[:limit]
        selected_items = set(selected_item_ids)
        keys = sorted(
            (key for key in grouped if key[1] in selected_items),
            key=lambda key: (key[0], key[1]),
        )
    missing = [key for key in keys if key not in grouped]
    if missing:
        details = ", ".join(
            f"{target_model} x {item_id}" for target_model, item_id in missing
        )
        raise ValueError(f"missing required complete transcript panels: {details}")
    panels = [
        TranscriptPanel(
            item_id=item_id,
            target_model=target_model,
            sources=tuple(grouped[(target_model, item_id)]),
        )
        for target_model, item_id in keys
    ]
    # Preflight every selected panel before the first paid judge request.
    for panel in panels:
        _sources_by_scenario(panel)
    return panels


def evaluate_transcript_files(
    *,
    transcript_dir: Path,
    judges: Sequence[JudgePanelMember],
    output_dir: Path,
    resume: bool = False,
    target_models: set[str] | None = None,
    item_ids: set[str] | None = None,
    required_target_models: set[str] | None = None,
    required_item_ids: set[str] | None = None,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
    scores_recorded: Callable[[Sequence[TranscriptScore]], None] | None = None,
    client_factory: Callable[[JudgePanelMember], JudgeClient] | None = None,
) -> list[TranscriptScore]:
    from .reporting import export_context

    judges = validate_judge_panel(judges)
    panels = load_transcript_panels(
        transcript_dir,
        target_models=target_models,
        item_ids=item_ids,
        required_target_models=required_target_models,
        required_item_ids=required_item_ids,
        limit=limit,
    )
    contexts = [EvaluationCheckpointContext(output_dir, judge) for judge in judges]
    scores: list[TranscriptScore] = []
    # Lock all selected namespaces before preflight. Nonblocking acquisition prevents deadlocks.
    with ExitStack() as stack:
        for context in contexts:
            stack.enter_context(judge_lock(context.namespace))
        completed = {}
        for context in contexts:
            validate_manifest(context)
            stored_panels(
                context
            )  # Cumulative exports must not conceal corrupt older batches.
            for panel in panels:
                path = evaluation_checkpoint_path(context, panel=panel)
                if path.exists() and not resume:
                    raise ValueError(
                        f"Checkpoint already exists: {path}. Use --resume or another output directory."
                    )
                saved = load_evaluation_checkpoint(context=context, panel=panel)
                completed[(context.namespace, panel.target_model, panel.item_id)] = (
                    saved[1] if saved else None
                )
        # No credentials/client setup until every selected source and checkpoint has passed.
        for context in contexts:
            judge = context.judge
            pending = any(
                (turns := completed[(context.namespace, p.target_model, p.item_id)])
                is None
                or any(len(v) != 5 for v in turns.values())
                for p in panels
            )
            if pending and judge.client is None:
                if client_factory is None:
                    raise ValueError(
                        f"No client for pending judge {judge_key(judge.definition)}"
                    )
                judge = replace(judge, client=client_factory(judge))
            validate_manifest(context, create=True)
            try:
                for panel in panels:
                    saved = completed[
                        (context.namespace, panel.target_model, panel.item_id)
                    ]
                    if progress:
                        progress(
                            f"Evaluating {judge_key(judge.definition)}: {panel.target_model} x {panel.item_id}"
                        )

                    def record_turns(current):
                        write_evaluation_checkpoint(
                            current, context=context, panel=panel
                        )

                    panel_scores = evaluate_transcript_panel(
                        panel,
                        judge=judge,
                        progress=progress,
                        completed_turn_evaluations=saved,
                        turn_recorded=record_turns,
                    )
                    scores.extend(panel_scores)
                    export_context(context)
                    if scores_recorded:
                        scores_recorded(scores)
            except BaseException:
                export_context(context)
                raise
    return scores
