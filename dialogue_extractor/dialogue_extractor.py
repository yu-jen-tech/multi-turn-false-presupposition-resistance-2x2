#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from transcript_generation import (
    DialogueTurn,
    TranscriptRecord,
    load_transcript,
)
from transcript_generation.storage import model_path_component


DEFAULT_TRANSCRIPT_DIR = Path("transcript")
DEFAULT_OUTPUT_DIR = Path("dialogue")
REQUIRED_TURN_COUNT = 5


def dialogue_columns(turn_count: int) -> tuple[str, ...]:
    columns = ["question", "response_1"]
    for turn_number in range(2, turn_count + 1):
        columns.extend((f"prompt_{turn_number}", f"response_{turn_number}"))
    return tuple(columns)


DIALOGUE_COLUMNS = dialogue_columns(REQUIRED_TURN_COUNT)


def transcript_files(transcript_dir: Path) -> list[Path]:
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory does not exist: {transcript_dir}")
    return sorted(path for path in transcript_dir.glob("*/*/*.json") if path.is_file())


def ordered_dialogue_turns(transcript: TranscriptRecord, *, path: Path) -> list[DialogueTurn]:
    if len(transcript.turns) != REQUIRED_TURN_COUNT:
        raise ValueError(f"{path} must contain exactly 5 turns")

    turns_by_number: dict[int, DialogueTurn] = {}
    for turn in transcript.turns:
        if turn.turn in turns_by_number:
            raise ValueError(f"{path} contains duplicate turn {turn.turn}")
        turns_by_number[turn.turn] = turn

    expected_turns = set(range(1, REQUIRED_TURN_COUNT + 1))
    if set(turns_by_number) != expected_turns:
        raise ValueError(f"{path} must contain exactly 5 turns numbered 1-5")

    return [turns_by_number[number] for number in range(1, REQUIRED_TURN_COUNT + 1)]


def clean_dialogue_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def dialogue_row(transcript: TranscriptRecord, *, path: Path) -> dict[str, str]:
    turns = ordered_dialogue_turns(transcript, path=path)
    row = {
        "question": clean_dialogue_text(transcript.item.question),
        "response_1": clean_dialogue_text(turns[0].assistant),
    }
    for turn_number in range(2, REQUIRED_TURN_COUNT + 1):
        turn = turns[turn_number - 1]
        row[f"prompt_{turn_number}"] = clean_dialogue_text(turn.user)
        row[f"response_{turn_number}"] = clean_dialogue_text(turn.assistant)
    return row


def extract_dialogues(*, transcript_dir: Path, output_dir: Path) -> list[Path]:
    groups: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for path in transcript_files(transcript_dir):
        transcript = load_transcript(path)
        target_model = model_path_component(transcript.target_model)
        scenario = transcript.scenario.name
        groups[(target_model, scenario)].append(
            (transcript.item.id, dialogue_row(transcript, path=path))
        )

    written_paths: list[Path] = []
    for (target_model, scenario), rows_with_ids in sorted(groups.items()):
        rows = [row for _, row in sorted(rows_with_ids, key=lambda item: item[0])]
        output_path = output_dir / target_model / scenario / "dialogue.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=DIALOGUE_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        written_paths.append(output_path)

    return written_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pure 5-turn dialogues from transcript JSON files into CSVs."
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPT_DIR,
        help="Root transcript directory with <scenario>/<target-model>/*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for <target-model>/<scenario>/dialogue.csv files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    written_paths = extract_dialogues(
        transcript_dir=args.transcript_dir,
        output_dir=args.output_dir,
    )
    for path in written_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
