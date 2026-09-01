from __future__ import annotations

import os
import re
from pathlib import Path

LATEST_CHECKPOINT_POINTER = "latest_checkpoint"
TRAINING_IN_PROGRESS = "training-in-progress"
_CHECKPOINT_NAME_RE = re.compile(r"checkpoint-(?P<step>\d+)")


class CheckpointNotReadyError(RuntimeError):
    pass


def write_latest_checkpoint_pointer(output_dir: str | os.PathLike[str], value: str) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / LATEST_CHECKPOINT_POINTER
    temporary = directory / f".{LATEST_CHECKPOINT_POINTER}.tmp"
    temporary.write_text(f"{value}\n", encoding="utf-8")
    os.replace(temporary, pointer)


def resolve_latest_checkpoint_pointer(output_dir: str | os.PathLike[str]) -> Path | None:
    directory = Path(output_dir)
    pointer = directory / LATEST_CHECKPOINT_POINTER
    if not pointer.is_file():
        return None

    checkpoint_name = pointer.read_text(encoding="utf-8").strip()
    if checkpoint_name == TRAINING_IN_PROGRESS:
        raise CheckpointNotReadyError(
            f"{pointer} shows that the latest fresh run stopped or is still running before its first checkpoint."
        )
    if _CHECKPOINT_NAME_RE.fullmatch(checkpoint_name) is None:
        raise ValueError(f"Invalid checkpoint pointer in {pointer}: {checkpoint_name!r}")
    checkpoint = directory / checkpoint_name
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint selected by {pointer} does not exist: {checkpoint}")
    return checkpoint


def resolve_automatic_resume_checkpoint(output_dir: str | os.PathLike[str]) -> str | None:
    directory = Path(output_dir)
    pointed_checkpoint = resolve_latest_checkpoint_pointer(directory)
    if pointed_checkpoint is not None:
        return str(pointed_checkpoint)

    legacy_checkpoints = []
    if directory.is_dir():
        for child in directory.iterdir():
            match = _CHECKPOINT_NAME_RE.fullmatch(child.name)
            if child.is_dir() and match is not None:
                legacy_checkpoints.append((int(match.group("step")), child))
    if not legacy_checkpoints:
        return None
    return str(max(legacy_checkpoints, key=lambda item: item[0])[1])
