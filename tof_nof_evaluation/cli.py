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
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PARSE_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSCRIPT_DIR,
    JudgeConfig,
)
from .providers import (
    DEFAULT_JUDGE_PROVIDER,
    JUDGE_PROVIDERS,
    create_judge_client,
    resolve_judge_client_config,
)
from .reporting import write_outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 5-turn false-presupposition transcripts with ToF and NoF."
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPT_DIR,
        help="Root transcript directory with <target-model>/<scenario>/<item-id>.json files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--judge-provider",
        choices=JUDGE_PROVIDERS,
        default=DEFAULT_JUDGE_PROVIDER,
        help=(
            "Judge transport: the current OpenAI-compatible Chat Completions "
            "provider, or the official OpenAI Responses API."
        ),
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument(
        "--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS
    )
    parser.add_argument(
        "--judge-parse-retries",
        type=int,
        default=DEFAULT_JUDGE_PARSE_RETRIES,
        help="Retries when a judge response is not valid label/reason JSON.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--judge-api-key",
        "--api-key",
        dest="judge_api_key",
        default=None,
        help=(
            "API key for the judge provider. The --api-key spelling is kept "
            "as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--judge-base-url",
        "--base-url",
        dest="judge_base_url",
        default=None,
        help=(
            "OpenAI-compatible base URL for the judge provider. The "
            "--base-url spelling is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario name to evaluate. May be passed multiple times.",
    )
    parser.add_argument(
        "--target-model",
        action="append",
        default=None,
        help=(
            "Target model name or path component to evaluate. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of transcripts to evaluate after filters.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse completed per-transcript evaluation checkpoints whose "
            "source transcript and judge configuration still match."
        ),
    )
    return parser.parse_args(argv)


def print_progress(message: str) -> None:
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv_if_present()
    args = parse_args(argv)
    try:
        client_config = resolve_judge_client_config(
            provider=args.judge_provider,
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
            environ=os.environ,
            request_timeout_seconds=args.request_timeout_seconds,
            api_retries=args.api_retries,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive when provided.")
    if args.judge_parse_retries < 0:
        raise SystemExit("--judge-parse-retries must be non-negative.")

    client = create_judge_client(
        provider=args.judge_provider,
        config=client_config,
    )
    judge_config = JudgeConfig(
        model=args.judge_model,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        seed=args.seed,
        parse_retries=args.judge_parse_retries,
    )
    print_progress(
        f"Starting ToF/NoF evaluation with judge model {judge_config.model}."
    )
    scores = evaluate_transcript_files(
        transcript_dir=args.transcript_dir,
        client=client,
        judge_config=judge_config,
        output_dir=args.output_dir,
        judge_provider=args.judge_provider,
        judge_endpoint=client_config.base_url,
        resume=args.resume,
        scenarios=set(args.scenario) if args.scenario is not None else None,
        target_models=(
            set(args.target_model) if args.target_model is not None else None
        ),
        limit=args.limit,
        progress=print_progress,
        scores_recorded=lambda current_scores: write_outputs(
            current_scores,
            output_dir=args.output_dir,
            judge_model=judge_config.model,
        ),
    )
    print_progress("Writing evaluation outputs...")
    written = write_outputs(
        scores,
        output_dir=args.output_dir,
        judge_model=judge_config.model,
    )
    print_progress(f"Evaluated {len(scores)} transcript(s).")
    for path in written:
        print_progress(f"Wrote {path}")
