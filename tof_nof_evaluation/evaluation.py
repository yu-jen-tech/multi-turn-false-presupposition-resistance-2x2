from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from dialogue_extractor.dialogue_extractor import (
    ordered_dialogue_turns,
    transcript_files,
)
from transcript_generation import SCENARIOS, TranscriptRecord, load_transcript
from transcript_generation.storage import model_path_component

from .judging import judge_turn
from .metrics import calculate_tof, calculate_trajectory_metrics
from .models import (
    JudgePanelMember,
    ProgressCallback,
    TranscriptPanel,
    TranscriptScore,
    TranscriptSource,
    TurnEvaluation,
)
from .storage import (
    EvaluationCheckpointContext,
    evaluation_checkpoint_path,
    load_evaluation_checkpoint,
    write_evaluation_checkpoint,
)


SCENARIO_ORDER = tuple(scenario.name for scenario in SCENARIOS)


def _validate_judges(judges: Sequence[JudgePanelMember]) -> None:
    if len(judges) != 3:
        raise ValueError("judge panel must contain exactly three judges")
    identities = [(judge.provider, judge.config.model) for judge in judges]
    if len(set(identities)) != 3:
        raise ValueError("judge panel must contain three unique provider/model pairs")
    for judge in judges:
        if not judge.config.model.strip():
            raise ValueError("judge model names must be non-empty")
        if judge.config.temperature != 0.0:
            raise ValueError("every judge temperature must be exactly 0.0")
        if judge.config.seed is not None:
            raise ValueError("judge seeds are not supported by the mixed-provider panel")
        if judge.config.max_tokens <= 0:
            raise ValueError("judge max tokens must be positive")
        if judge.config.parse_retries < 0:
            raise ValueError("judge parse retries must be non-negative")


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


def _judge_consensus_turn(
    *,
    transcript: TranscriptRecord,
    turn_index: int,
    source_path: Path,
    judges: Sequence[JudgePanelMember],
    progress: ProgressCallback | None,
) -> TurnEvaluation:
    turn = ordered_dialogue_turns(transcript, path=source_path)[turn_index - 1]
    judge_evaluations = tuple(
        judge_turn(
            transcript=transcript,
            turn=turn,
            judge=judge,
            progress=progress,
        )
        for judge in judges
    )
    label = 1 if sum(evaluation.label for evaluation in judge_evaluations) >= 2 else 0
    return TurnEvaluation(
        turn=turn.turn,
        label=label,
        user_message=turn.user,
        assistant_response=turn.assistant,
        judge_evaluations=judge_evaluations,
    )


def _completed_turns(
    completed: Mapping[str, Sequence[TurnEvaluation]] | None,
) -> dict[str, list[TurnEvaluation]]:
    if completed is None:
        return {scenario: [] for scenario in SCENARIO_ORDER}
    if set(completed) != set(SCENARIO_ORDER):
        raise ValueError("completed panel evaluations must cover all four scenarios")
    normalized = {
        scenario: list(completed[scenario]) for scenario in SCENARIO_ORDER
    }
    for scenario, evaluations in normalized.items():
        expected = tuple(range(1, len(evaluations) + 1))
        if tuple(evaluation.turn for evaluation in evaluations) != expected:
            raise ValueError(
                f"completed evaluations for {scenario} must be a consecutive prefix"
            )
    populated = [evaluations for evaluations in normalized.values() if evaluations]
    if populated:
        if len(populated) != len(SCENARIO_ORDER):
            raise ValueError("shared turn 1 must be present for every scenario")
        reference = populated[0][0]
        if any(evaluations[0] != reference for evaluations in populated[1:]):
            raise ValueError("completed shared turn 1 decisions must be identical")
    return normalized


def evaluate_transcript_panel(
    panel: TranscriptPanel,
    *,
    judges: Sequence[JudgePanelMember],
    progress: ProgressCallback | None = None,
    completed_turn_evaluations: Mapping[
        str, Sequence[TurnEvaluation]
    ] | None = None,
    turn_recorded: Callable[
        [Mapping[str, Sequence[TurnEvaluation]]], None
    ] | None = None,
) -> list[TranscriptScore]:
    """Evaluate one complete item x target panel through three judges."""
    _validate_judges(judges)
    sources = _sources_by_scenario(panel)
    completed = _completed_turns(completed_turn_evaluations)

    if not any(completed.values()):
        canonical = sources[SCENARIO_ORDER[0]]
        if progress is not None:
            progress("Judging shared turn 1/17...")
        shared_turn = _judge_consensus_turn(
            transcript=canonical.transcript,
            turn_index=1,
            source_path=canonical.path,
            judges=judges,
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
            evaluation = _judge_consensus_turn(
                transcript=source.transcript,
                turn_index=turn_index,
                source_path=source.path,
                judges=judges,
                progress=progress,
            )
            completed[scenario].append(evaluation)
            if turn_recorded is not None:
                turn_recorded(completed)

    scores: list[TranscriptScore] = []
    for scenario in SCENARIO_ORDER:
        source = sources[scenario]
        evaluations = tuple(completed[scenario])
        labels = tuple(evaluation.label for evaluation in evaluations)
        metrics = calculate_trajectory_metrics(labels)
        transcript = source.transcript
        scores.append(
            TranscriptScore(
                item_id=transcript.item.id,
                question=transcript.item.question,
                presuppositions=list(transcript.item.presuppositions),
                corrections=list(transcript.item.corrections),
                target_model=transcript.target_model,
                target_model_component=model_path_component(
                    transcript.target_model
                ),
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

    keys = sorted(grouped, key=lambda key: (key[0], key[1]))
    if limit is not None:
        keys = keys[:limit]
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
    limit: int | None = None,
    progress: ProgressCallback | None = None,
    scores_recorded: Callable[[Sequence[TranscriptScore]], None] | None = None,
) -> list[TranscriptScore]:
    _validate_judges(judges)
    panels = load_transcript_panels(
        transcript_dir,
        target_models=target_models,
        item_ids=item_ids,
        limit=limit,
    )
    if progress is not None:
        progress(f"Discovered {len(panels)} complete transcript panel(s).")

    context = EvaluationCheckpointContext(
        output_dir=output_dir,
        judges=tuple(judges),
    )
    scores: list[TranscriptScore] = []
    for panel_index, panel in enumerate(panels, start=1):
        checkpoint_path = evaluation_checkpoint_path(context, panel=panel)
        completed: Mapping[str, Sequence[TurnEvaluation]] | None = None
        if resume:
            checkpoint = load_evaluation_checkpoint(context=context, panel=panel)
            if checkpoint is not None:
                checkpoint_path, completed = checkpoint
        if progress is not None:
            progress(
                f"Evaluating panel {panel_index}/{len(panels)}: "
                f"{panel.item_id} x {panel.target_model}"
            )

        def record_turns(
            current: Mapping[str, Sequence[TurnEvaluation]],
        ) -> None:
            write_evaluation_checkpoint(
                current,
                context=context,
                panel=panel,
            )

        panel_scores = evaluate_transcript_panel(
            panel,
            judges=judges,
            progress=progress,
            completed_turn_evaluations=completed,
            turn_recorded=record_turns,
        )
        scores.extend(panel_scores)
        if scores_recorded is not None:
            scores_recorded(scores)
        if progress is not None:
            progress(f"Completed panel; checkpoint {checkpoint_path}")
    return scores
