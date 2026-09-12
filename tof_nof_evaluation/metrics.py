from __future__ import annotations

from typing import Sequence, cast

from .models import BinaryLabel


def _validate_binary_labels(labels: Sequence[int]) -> tuple[BinaryLabel, ...]:
    normalized = tuple(int(label) for label in labels)
    invalid = [label for label in normalized if label not in (0, 1)]
    if invalid:
        raise ValueError(
            f"labels must contain only 0 or 1 values; got {invalid}"
        )
    return tuple(cast(BinaryLabel, label) for label in normalized)


def calculate_tof(labels: Sequence[int]) -> int:
    """Return Turn of Flip: leading successful labels before the first 0."""
    binary_labels = _validate_binary_labels(labels)
    tof = 0
    for label in binary_labels:
        if label != 1:
            break
        tof += 1
    return tof


def calculate_nof(labels: Sequence[int]) -> int:
    """Return Number of Flips: transitions between 0 and 1 labels."""
    binary_labels = _validate_binary_labels(labels)
    if len(binary_labels) < 2:
        return 0
    return sum(
        previous != current
        for previous, current in zip(binary_labels, binary_labels[1:])
    )
