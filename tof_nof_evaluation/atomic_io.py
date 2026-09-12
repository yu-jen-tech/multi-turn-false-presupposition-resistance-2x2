from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


@contextmanager
def atomic_text_writer(
    path: Path,
    *,
    newline: str | None = None,
) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline=newline,
        ) as file:
            yield file
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    with atomic_text_writer(path) as file:
        file.write(text)
