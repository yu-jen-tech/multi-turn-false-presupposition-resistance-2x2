#!/usr/bin/env python3
"""Load CSV, JSON, and JSONL files into pandas DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd


DataFormat = Literal["csv", "json", "jsonl"]

# Change this path when using the loader interactively. Call
# ``load_dataframe()`` afterwards to load the selected file.
INPUT_PATH = Path(__file__).resolve(
).parents[1] / "data" / "crepe_dataset.jsonl"

_FORMAT_ALIASES: dict[str, DataFormat] = {
    "csv": "csv",
    "json": "json",
    "jsonl": "jsonl",
    "ndjson": "jsonl",
}
_SUPPORTED_FORMATS = ", ".join(sorted(_FORMAT_ALIASES))


def _normalise_format(file_format: str) -> DataFormat:
    normalised = file_format.lower().lstrip(".")

    try:
        return _FORMAT_ALIASES[normalised]
    except KeyError as error:
        raise ValueError(
            f"Unsupported file format {file_format!r}. "
            f"Expected one of: {_SUPPORTED_FORMATS}."
        ) from error


def _detect_format(input_path: Path) -> DataFormat:
    """Detect a supported format from the filename, including compression."""
    for suffix in reversed(input_path.suffixes):
        format_name = _FORMAT_ALIASES.get(suffix.lower().lstrip("."))
        if format_name is not None:
            return format_name

    raise ValueError(
        f"Could not determine the file format from {input_path.name!r}. "
        f"Use file_format=... or one of: {_SUPPORTED_FORMATS}."
    )


def load_dataframe(
    input_path: str | Path | None = None,
    *,
    file_format: str | None = None,
    **read_options: Any,
) -> pd.DataFrame:
    """Read a CSV, JSON, or JSONL file into a pandas DataFrame.

    The format is detected from the filename unless ``file_format`` is given.
    Additional keyword arguments are passed to the corresponding pandas
    reader, such as ``sep`` for CSV or ``orient`` for JSON.

    JSONL files are expected to contain one JSON record per line. Blank lines
    are accepted by pandas and ignored by the JSONL reader.
    """
    path = INPUT_PATH if input_path is None else Path(input_path)
    data_format = (
        _normalise_format(file_format)
        if file_format is not None
        else _detect_format(path)
    )

    if data_format == "csv":
        return pd.read_csv(path, **read_options)

    if data_format == "json":
        return pd.read_json(path, **read_options)

    if "lines" in read_options and read_options["lines"] is not True:
        raise ValueError("JSONL input must be read with lines=True.")

    read_options["lines"] = True
    return pd.read_json(path, **read_options)


# Display the DataFrame when this module is run interactively.
crepe_dataset_df = load_dataframe(
    "data/crepe_dataset.jsonl", file_format="jsonl")
