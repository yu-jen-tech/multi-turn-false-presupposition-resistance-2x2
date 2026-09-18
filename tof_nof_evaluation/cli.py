from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from transcript_generation.cli import load_dotenv_if_present
from transcript_generation.models import DEFAULT_REQUEST_TIMEOUT_SECONDS

from .evaluation import evaluate_transcript_files
from .models import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_PARSE_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSCRIPT_DIR,
)
from .providers import create_default_judge_panel
from .reporting import write_outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate complete 5-turn transcript panels with three judges and "
            "majority-consensus ToF/NoF."
        )
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPT_DIR,
        help="Root transcript directory with all four scenarios per item and target.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
    )
    parser.add_argument(
        "--judge-parse-retries",
        type=int,
        default=DEFAULT_JUDGE_PARSE_RETRIES,
        help="Retries only malformed or invalid structured judge output.",
    )
    parser.add_argument("--ki4buw-api-key", default=None)
    parser.add_argument("--ki4buw-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument(
        "--target-model",
        action="append",
        default=None,
        help="Target model name or path component. May be repeated.",
    )
    parser.add_argument(
        "--item-id",
        action="append",
        default=None,
        help=(
            "Fixed source item ID to evaluate. May be repeated; all four "
            "scenarios are always loaded."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of complete item x target panels after filters.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching multi-judge panel checkpoint.",
    )
    return parser.parse_args(argv)


def print_progress(message: str) -> None:
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv_if_present()
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive when provided.")
    try:
        judges = create_default_judge_panel(
            environ=os.environ,
            ki4buw_api_key=args.ki4buw_api_key,
            ki4buw_base_url=args.ki4buw_base_url,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            request_timeout_seconds=args.request_timeout_seconds,
            api_retries=args.api_retries,
            max_tokens=args.judge_max_tokens,
            parse_retries=args.judge_parse_retries,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None

    panel_description = ", ".join(
        f"{judge.provider}:{judge.config.model}" for judge in judges
    )
    print_progress(f"Starting majority-consensus evaluation with {panel_description}.")
    scores = evaluate_transcript_files(
        transcript_dir=args.transcript_dir,
        judges=judges,
        output_dir=args.output_dir,
        resume=args.resume,
        target_models=(
            set(args.target_model) if args.target_model is not None else None
        ),
        item_ids=set(args.item_id) if args.item_id is not None else None,
        limit=args.limit,
        progress=print_progress,
        scores_recorded=lambda current_scores: write_outputs(
            current_scores,
            output_dir=args.output_dir,
            judges=judges,
        ),
    )
    print_progress("Writing consensus evaluation outputs...")
    written = write_outputs(
        scores,
        output_dir=args.output_dir,
        judges=judges,
    )
    print_progress(f"Evaluated {len(scores) // 4} complete panel(s).")
    for path in written:
        print_progress(f"Wrote {path}")
