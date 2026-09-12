#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FALSE_PRESUPPOSITION = "false presupposition"
DATA_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract canonical false-presupposition data from CREPE JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "crepe_dataset.jsonl",
        help="Path to the CREPE JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "false_presupposition_data.jsonl",
        help="Path for the complete extracted JSONL file.",
    )

    return parser.parse_args()


def nonempty_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None

    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        cleaned.append(item.strip())
    return cleaned


def is_double_false_presupposition(item: dict[str, Any]) -> bool:
    labels = item.get("raw_labels")
    return (
        isinstance(labels, list)
        and len(labels) == 2
        and all(label == FALSE_PRESUPPOSITION for label in labels)
    )


def to_json_row(item: dict[str, Any]) -> dict[str, Any] | None:
    if not is_double_false_presupposition(item):
        return None

    item_id = item.get("id")
    question = item.get("question")
    presuppositions = nonempty_string_list(item.get("presuppositions"))
    corrections = nonempty_string_list(item.get("corrections"))

    if not all(isinstance(value, str) and value.strip() for value in (item_id, question)):
        return None
    if presuppositions is None or corrections is None:
        return None
    if len(presuppositions) != len(corrections):
        return None

    return {
        "id": item_id.strip(),
        "question": question.strip(),
        "presuppositions": presuppositions,
        "corrections": corrections,
    }


def load_candidates(input_path: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}") from error

            if not isinstance(item, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")

            row = to_json_row(item)
            if row is not None:
                candidates.append(row)

    return candidates


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.input)
    write_jsonl(candidates, args.output)

    print(f"Wrote {len(candidates)} rows to {args.output}")


if __name__ == "__main__":
    main()
