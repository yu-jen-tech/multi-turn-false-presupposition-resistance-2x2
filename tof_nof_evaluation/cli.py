from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from transcript_generation.cli import load_dotenv_if_present
from transcript_generation.models import (
    DEFAULT_DATASET_JSONL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TARGET_MODELS,
)
from transcript_generation.storage import load_items, model_path_component

from .evaluation import evaluate_transcript_files
from .models import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_PARSE_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSCRIPT_DIR,
)
from .providers import (
    configure_judges,
    create_judge_client,
    resolve_judge_client_config,
)
from .reporting import export_results
from .models import resolve_judges


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected complete panels independently with supported judges. "
            "Use import-legacy or export for offline operations."
        )
    )
    parser.add_argument(
        "--judge",
        action="append",
        help="qwen3.8, gpt-5.4-mini, or gpt-oss-120b; repeat to select judges (default: all).",
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=DEFAULT_DATASET_JSONL,
        help="Dataset manifest used to require every selected item x target panel.",
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
        help=(
            "Maximum number of source items after filters; every selected "
            "target is still required for each item."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matching independent judge checkpoints.",
    )
    return parser.parse_args(argv)


def print_progress(message: str) -> None:
    print(message, flush=True)


def _selected_target_models(requested: Sequence[str] | None) -> set[str]:
    if requested is None:
        return set(DEFAULT_TARGET_MODELS)
    selected: set[str] = set()
    for value in requested:
        matches = [
            model
            for model in DEFAULT_TARGET_MODELS
            if value in (model, model_path_component(model))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Unknown target model {value!r}; expected one of "
                + ", ".join(DEFAULT_TARGET_MODELS)
            )
        selected.add(matches[0])
    return selected


def _selected_item_ids(
    requested: Sequence[str] | None,
    *,
    dataset_jsonl: Path,
) -> set[str]:
    items = load_items(dataset_jsonl)
    available = {item.id for item in items}
    if requested is None:
        return available
    selected = set(requested)
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(
            "Unknown item ID(s) for the selected dataset: " + ", ".join(unknown)
        )
    return selected


def main(argv: Sequence[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in ("import-legacy", "export"):
        mode = values.pop(0)
        parser = argparse.ArgumentParser(
            description=f"Offline {mode}; never constructs model clients."
        )
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        parser.add_argument("--judge", action="append")
        if mode == "import-legacy":
            parser.add_argument("--input-dir", type=Path, required=True)
        args = parser.parse_args(values)
        try:
            if mode == "export":
                written = export_results(args.output_dir, aliases=args.judge)
            else:
                from .legacy_import import import_legacy

                written = import_legacy(
                    args.input_dir, args.output_dir, aliases=args.judge
                )
        except (ValueError, RuntimeError, OSError) as error:
            raise SystemExit(str(error)) from None
        for path in written:
            print_progress(f"Wrote {path}")
        return
    args = parse_args(values)
    try:
        resolve_judges(
            args.judge
        )  # Reject invalid selection before writes or credential handling.
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive when provided")
        if args.request_timeout_seconds <= 0 or args.api_retries < 0:
            raise ValueError(
                "request timeout must be positive and API retries non-negative"
            )
        load_dotenv_if_present()
        target_models = _selected_target_models(args.target_model)
        item_ids = _selected_item_ids(args.item_id, dataset_jsonl=args.dataset_jsonl)
        judges = configure_judges(
            environ=os.environ,
            aliases=args.judge,
            ki4buw_base_url=args.ki4buw_base_url,
            openai_base_url=args.openai_base_url,
            max_tokens=args.judge_max_tokens,
            parse_retries=args.judge_parse_retries,
        )

        def client_factory(judge):
            config = resolve_judge_client_config(
                provider=judge.provider,
                api_key=args.openai_api_key
                if judge.provider == "openai"
                else args.ki4buw_api_key,
                base_url=judge.endpoint,
                environ=os.environ,
                request_timeout_seconds=args.request_timeout_seconds,
                api_retries=args.api_retries,
            )
            return create_judge_client(provider=judge.provider, config=config)

        scores = evaluate_transcript_files(
            transcript_dir=args.transcript_dir,
            judges=judges,
            output_dir=args.output_dir,
            resume=args.resume,
            target_models=target_models,
            item_ids=item_ids,
            required_target_models=target_models,
            required_item_ids=item_ids,
            limit=args.limit,
            progress=print_progress,
            client_factory=client_factory,
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
    print_progress(
        f"Evaluated {len(scores)} independent judge trajectories; cumulative outputs saved."
    )
