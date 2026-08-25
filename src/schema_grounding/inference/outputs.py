from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


class ResumableJsonl:
    """Append to a partial JSONL and atomically publish it on completion."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.partial_path = path.with_suffix(path.suffix + ".partial")

    @property
    def complete(self) -> bool:
        return self.path.is_file()

    def progress(self, key: str = "id") -> tuple[int, str | None]:
        if not self.partial_path.exists():
            return 0, None
        count = 0
        last_key: str | None = None
        with self.partial_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Corrupt partial output {self.partial_path}:{line_number}; remove it to restart this stage"
                    ) from error
                value = row.get(key)
                if not isinstance(value, str):
                    raise ValueError(f"Partial row {line_number} has no string {key!r}")
                count += 1
                last_key = value
        return count, last_key

    def open_append(self) -> TextIO:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.partial_path.open("a", encoding="utf-8", newline="\n")

    def publish(self) -> None:
        if not self.partial_path.is_file():
            raise FileNotFoundError(f"No partial output to publish: {self.partial_path}")
        self.partial_path.replace(self.path)


def write_json_line(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
