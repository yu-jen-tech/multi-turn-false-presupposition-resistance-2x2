#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_LIMIT = 100
DEFAULT_SEED = 66
DATA_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample false-presupposition data from canonical JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "false_presupposition_data.jsonl",
        help="Path to the complete extracted JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "false_presupposition_sample.jsonl",
        help="Path for the sampled JSONL file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of rows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used before selecting rows.",
    )

    return parser.parse_args()


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}"
                )
            rows.append(row)

    return rows


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("--limit must be positive")
    if len(rows) < limit:
        raise ValueError(
            f"Only found {len(rows)} canonical rows, but {limit} were requested"
        )

    shuffled_rows = rows[:]
    random.Random(seed).shuffle(shuffled_rows)
    return shuffled_rows[:limit]


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    sampled_rows = sample_rows(rows, limit=args.limit, seed=args.seed)
    write_jsonl(sampled_rows, args.output)

    print(f"Wrote {len(sampled_rows)} rows to {args.output}")
    print(f"Sampled from {len(rows)} canonical rows with seed {args.seed}")


if __name__ == "__main__":
    main()
