from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from dialogue_extractor.dialogue_extractor import (
    ordered_dialogue_turns,
    transcript_files,
)
from transcript_generation import ChatClient, TranscriptRecord, load_transcript
from transcript_generation.storage import model_path_component

from .judging import judge_turn
from .metrics import calculate_nof, calculate_tof
from .models import JudgeConfig, ProgressCallback, TranscriptScore, TurnEvaluation
from .storage import (
    EvaluationCheckpointContext,
    evaluation_checkpoint_path,
    load_evaluation_checkpoint,
    write_evaluation_checkpoint,
)


def evaluate_transcript(
    transcript: TranscriptRecord,
    *,
    path: Path,
    client: ChatClient,
    judge_config: JudgeConfig,
    progress: ProgressCallback | None = None,
    completed_turn_evaluations: Sequence[TurnEvaluation] = (),
    turn_recorded: Callable[[Sequence[TurnEvaluation]], None] | None = None,
) -> TranscriptScore:
    turns = ordered_dialogue_turns(transcript, path=path)
    turn_evaluations = list(completed_turn_evaluations)
    expected_completed_turns = tuple(range(1, len(turn_evaluations) + 1))
    if tuple(evaluation.turn for evaluation in turn_evaluations) != (
        expected_completed_turns
    ):
        raise ValueError(
            "completed turn evaluations must be a consecutive prefix"
        )
    for turn in turns[len(turn_evaluations):]:
        turn_index = turn.turn
        if progress is not None:
            progress(f"Judging turn {turn_index}/{len(turns)}...")
        evaluation = judge_turn(
            transcript=transcript,
            turn=turn,
            client=client,
            judge_config=judge_config,
            progress=progress,
        )
        turn_evaluations.append(evaluation)
        if turn_recorded is not None:
            turn_recorded(turn_evaluations)
        if progress is not None:
            progress(
                f"Turn {turn_index}/{len(turns)} judged: "
                f"label={evaluation.label}"
            )
    labels = tuple(evaluation.label for evaluation in turn_evaluations)
    return TranscriptScore(
        item_id=transcript.item.id,
        question=transcript.item.question,
        presupposition=transcript.item.presuppositions,
        correction=transcript.item.corrections,
        target_model=transcript.target_model,
        target_model_component=model_path_component(transcript.target_model),
        scenario=transcript.scenario.name,
        labels=labels,
        tof=calculate_tof(labels),
        nof=calculate_nof(labels),
        turn_evaluations=tuple(turn_evaluations),
    )


def evaluate_transcript_files(
    *,
    transcript_dir: Path,
    client: ChatClient,
    judge_config: JudgeConfig,
    output_dir: Path,
    judge_provider: str = "openai-compatible",
    judge_endpoint: str | None = None,
    resume: bool = False,
    scenarios: set[str] | None = None,
    target_models: set[str] | None = None,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
    scores_recorded: Callable[[Sequence[TranscriptScore]], None] | None = None,
) -> list[TranscriptScore]:
    scores: list[TranscriptScore] = []
    checkpoint_context = EvaluationCheckpointContext(
        output_dir=output_dir,
        judge_config=judge_config,
        judge_provider=judge_provider,
        judge_endpoint=judge_endpoint,
    )
    paths = transcript_files(transcript_dir)
    if progress is not None:
        progress(f"Discovered {len(paths)} transcript file(s).")
    selected_count = 0
    for path in paths:
        transcript = load_transcript(path)
        target_model_component = model_path_component(transcript.target_model)
        if scenarios is not None and transcript.scenario.name not in scenarios:
            continue
        if (
            target_models is not None
            and transcript.target_model not in target_models
            and target_model_component not in target_models
        ):
            continue

        if limit is not None and selected_count >= limit:
            break
        selected_count += 1

        completed_turn_evaluations: Sequence[TurnEvaluation] = ()
        checkpoint_path = evaluation_checkpoint_path(
            checkpoint_context,
            transcript=transcript,
        )
        if resume:
            checkpoint = load_evaluation_checkpoint(
                context=checkpoint_context,
                transcript=transcript,
                transcript_path=path,
            )
            if checkpoint is not None:
                checkpoint_path, completed_turn_evaluations = checkpoint

        if progress is not None:
            if len(completed_turn_evaluations) == 5:
                progress(f"Skipping existing evaluation {checkpoint_path}")
            elif completed_turn_evaluations:
                progress(
                    f"Resuming transcript {selected_count} at turn "
                    f"{len(completed_turn_evaluations) + 1}: {path}"
                )
            else:
                progress(f"Evaluating transcript {selected_count}: {path}")

        def record_turns(
            turn_evaluations: Sequence[TurnEvaluation],
        ) -> None:
            write_evaluation_checkpoint(
                turn_evaluations,
                context=checkpoint_context,
                transcript=transcript,
                transcript_path=path,
            )

        score = evaluate_transcript(
            transcript,
            path=path,
            client=client,
            judge_config=judge_config,
            progress=progress,
            completed_turn_evaluations=completed_turn_evaluations,
            turn_recorded=record_turns,
        )
        scores.append(score)
        if scores_recorded is not None:
            scores_recorded(scores)
        if progress is not None:
            progress(
                f"Completed transcript {selected_count}: "
                f"ToF={score.tof}, NoF={score.nof}; recorded {checkpoint_path}"
            )
    if progress is not None:
        progress(f"Finished evaluating {len(scores)} transcript(s).")
    return scores
