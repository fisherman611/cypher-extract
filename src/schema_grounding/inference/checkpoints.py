from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECKPOINT_ROOT = Path("results")
DEFAULT_MODEL_FAMILY = "qwen3"
SUPPORTED_MODEL_FAMILIES = ("qwen3", "llama3", "qwen2.5_coder")
DEFAULT_METHODS = (
    "teacher_lora",
    "sft",
    "fkl",
    "rkl",
    "sfkl",
    "srkl",
    "csd",
    "hpd",
    "amid",
    "fdd_sfkl",
    "fdd_srkl",
    "distillm_adaptive_sfkl",
    "distillm_adaptive_srkl",
)
_CHECKPOINT_RE = re.compile(r"(?:^|/)checkpoint-(?P<step>\d+)$")


@dataclass(frozen=True)
class LastCheckpoint:
    checkpoint_root: str
    model_family: str
    method: str
    step: int
    subfolder: str

    @property
    def path(self) -> Path:
        return Path(self.checkpoint_root, *self.subfolder.split("/"))

    @property
    def uri(self) -> str:
        return str(self.path)


def select_last_checkpoint(paths: Iterable[str], method_prefix: str) -> tuple[int, str]:
    candidates: list[tuple[int, str]] = []
    normalized_prefix = method_prefix.strip("/")
    for path in paths:
        normalized_path = path.replace("\\", "/").rstrip("/")
        match = _CHECKPOINT_RE.search(normalized_path)
        if match and normalized_path.startswith(f"{normalized_prefix}/"):
            candidates.append((int(match.group("step")), normalized_path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-N directory found under {normalized_prefix}")
    return max(candidates, key=lambda item: item[0])


def resolve_last_checkpoint(
    method: str,
    *,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> LastCheckpoint:
    if method not in DEFAULT_METHODS:
        raise ValueError(f"Unsupported method {method!r}; expected one of {', '.join(DEFAULT_METHODS)}")
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported model family {model_family!r}; expected one of {', '.join(SUPPORTED_MODEL_FAMILIES)}"
        )

    root = Path(checkpoint_root).expanduser().resolve()
    method_prefix = f"{model_family}/{method}"
    method_directory = root / model_family / method
    if not method_directory.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {method_directory}")

    paths = (path.relative_to(root).as_posix() for path in method_directory.iterdir() if path.is_dir())
    step, subfolder = select_last_checkpoint(paths, method_prefix)
    return LastCheckpoint(str(root), model_family, method, step, subfolder)


def resolve_checkpoint_directory(checkpoint: LastCheckpoint) -> Path:
    """Validate and return a local adapter or full-model checkpoint directory."""

    directory = checkpoint.path
    if not directory.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {directory}")

    adapter_config = directory / "adapter_config.json"
    if adapter_config.is_file():
        if not (directory / "adapter_model.safetensors").is_file() and not (
            directory / "adapter_model.bin"
        ).is_file():
            raise FileNotFoundError(f"Missing adapter weights in {directory}")
    else:
        full_model_files = list(directory.glob("model*.safetensors")) + list(
            directory.glob("pytorch_model*.bin")
        )
        if not (directory / "config.json").is_file() or not full_model_files:
            raise FileNotFoundError(f"Missing adapter or full-model weights in {directory}")
    return directory
