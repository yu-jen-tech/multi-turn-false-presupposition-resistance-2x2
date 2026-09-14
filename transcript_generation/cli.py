from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .batch import generate_transcripts
from .client import OpenAICompatibleChatClient, resolve_ki4buw_base_url
from .models import (
    DEFAULT_DATASET_JSONL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SIMULATOR_MODEL,
    DEFAULT_TARGET_MAX_TOKENS,
    DEFAULT_TARGET_MODELS,
    DEFAULT_VALIDATOR_MODEL,
    BatchGenerationPlan,
    GenerationSettings,
    SCENARIOS,
)
from .storage import load_dataset, select_items


def write_trace(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_dotenv_if_present(dotenv_path: Path = Path(".env")) -> None:
    if not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 5-turn false-presupposition transcripts for the "
            "2x2 pushback scenarios."
        )
    )
    parser.add_argument(
        "--dataset-jsonl",
        "--filtered-csv",
        dest="dataset_jsonl",
        type=Path,
        default=DEFAULT_DATASET_JSONL,
        help=(
            "JSONL dataset with complete presupposition and correction lists. "
            "The --filtered-csv spelling is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root transcript directory with <target-model>/<scenario>/<item-id>.json files.",
    )
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of dataset questions to generate, starting at --item-index.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help=(
            "Scenario to generate. May be passed multiple times. "
            "Defaults to the original four 2x2 scenarios."
        ),
    )
    parser.add_argument("--simulator-model", default=DEFAULT_SIMULATOR_MODEL)
    parser.add_argument(
        "--validator-model",
        default=DEFAULT_VALIDATOR_MODEL,
        help=(
            "Model used for semantic pushback validation. "
            f"Defaults to {DEFAULT_VALIDATOR_MODEL}."
        ),
    )
    parser.add_argument(
        "--target-model",
        dest="target_models",
        action="append",
        default=None,
        help=(
            "Target model to generate. May be passed multiple times. "
            f"Defaults to: {', '.join(DEFAULT_TARGET_MODELS)}."
        ),
    )
    parser.add_argument("--simulator-temperature", type=float, default=0.7)
    parser.add_argument("--validator-temperature", type=float, default=0.0)
    parser.add_argument("--target-temperature", type=float, default=0.0)
    parser.add_argument("--simulator-max-tokens", type=int, default=180)
    parser.add_argument("--validator-max-tokens", type=int, default=512)
    parser.add_argument(
        "--target-max-tokens",
        type=int,
        default=DEFAULT_TARGET_MAX_TOKENS,
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=120)
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=50)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed generation traces to stderr.",
    )
    args = parser.parse_args(argv)
    if args.target_models is None:
        args.target_models = list(DEFAULT_TARGET_MODELS)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv_if_present()
    args = parse_args(argv)

    api_key = (
        args.api_key
        or os.environ.get("KI4BUW_API_KEY")
    )
    base_url = resolve_ki4buw_base_url(args.base_url, environ=os.environ)
    if not api_key:
        raise SystemExit(
            "Missing API key. Set KI4BUW_API_KEY or pass --api-key."
        )

    items, dataset_provenance = load_dataset(args.dataset_jsonl)
    selected_items = select_items(
        items,
        item_index=args.item_index,
        num_questions=args.num_questions,
    )
    selected_scenarios = [
        scenario
        for scenario in SCENARIOS
        if args.scenario is None or scenario.name in args.scenario
    ]
    settings = [
        GenerationSettings(
            simulator_model=args.simulator_model,
            validator_model=args.validator_model,
            target_model=target_model,
            simulator_temperature=args.simulator_temperature,
            validator_temperature=args.validator_temperature,
            target_temperature=args.target_temperature,
            seed=args.seed,
            simulator_max_tokens=args.simulator_max_tokens,
            validator_max_tokens=args.validator_max_tokens,
            target_max_tokens=args.target_max_tokens,
            max_retries=args.max_retries,
        )
        for target_model in args.target_models
    ]
    client = OpenAICompatibleChatClient(
        api_key=api_key,
        base_url=base_url,
        request_timeout_seconds=args.request_timeout_seconds,
        api_retries=args.api_retries,
    )
    trace = write_trace if args.verbose else None
    written = generate_transcripts(
        plan=BatchGenerationPlan(
            items=selected_items,
            scenarios=selected_scenarios,
            settings=settings,
            output_dir=args.output_dir,
            dataset_provenance=dataset_provenance,
            resume=args.resume,
            overwrite=args.overwrite,
        ),
        client=client,
        progress=lambda message: print(message, flush=True),
        trace=trace,
    )

    print(
        f"Prepared {len(written)} transcript files for "
        f"{len(selected_items)} item(s) and "
        f"{len(args.target_models)} requested target model(s):"
    )
    for path in written:
        print(path)
