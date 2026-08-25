from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

DEFAULT_REPO_ID = "distillation-sql/nothing-extract"
DEFAULT_MODEL_FAMILY = "qwen3"
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
_INFERENCE_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)


@dataclass(frozen=True)
class LastCheckpoint:
    repo_id: str
    revision: str
    method: str
    step: int
    subfolder: str

    @property
    def uri(self) -> str:
        return f"hf://{self.repo_id}/{self.subfolder}"


def _path(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("path")
    else:
        value = getattr(item, "path", None)
    if not isinstance(value, str):
        raise TypeError(f"Hugging Face tree item has no path: {item!r}")
    return value


def select_last_checkpoint(paths: Iterable[str], method_prefix: str) -> tuple[int, str]:
    candidates: list[tuple[int, str]] = []
    normalized_prefix = method_prefix.strip("/")
    for path in paths:
        match = _CHECKPOINT_RE.search(path.rstrip("/"))
        if match and path.rstrip("/").startswith(f"{normalized_prefix}/"):
            candidates.append((int(match.group("step")), path.rstrip("/")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-N directory found under {normalized_prefix}")
    return max(candidates, key=lambda item: item[0])


def resolve_last_checkpoint(
    method: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    model_family: str = DEFAULT_MODEL_FAMILY,
    revision: str = "main",
    api: HfApi | None = None,
    token: str | None = None,
) -> LastCheckpoint:
    if method not in DEFAULT_METHODS:
        raise ValueError(f"Unsupported method {method!r}; expected one of {', '.join(DEFAULT_METHODS)}")
    token = token or os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN")
    method_prefix = f"{model_family}/{method}"
    tree = (api or HfApi()).list_repo_tree(
        repo_id=repo_id,
        path_in_repo=method_prefix,
        recursive=False,
        revision=revision,
        token=token,
    )
    step, subfolder = select_last_checkpoint((_path(item) for item in tree), method_prefix)
    return LastCheckpoint(repo_id, revision, method, step, subfolder)


def download_inference_checkpoint(
    checkpoint: LastCheckpoint,
    *,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    """Download an adapter without DeepSpeed/optimizer checkpoint state."""

    token = token or os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN")
    allow_patterns = [f"{checkpoint.subfolder}/{filename}" for filename in _INFERENCE_FILES]
    snapshot = snapshot_download(
        repo_id=checkpoint.repo_id,
        revision=checkpoint.revision,
        allow_patterns=allow_patterns,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
    )
    directory = Path(snapshot, checkpoint.subfolder)
    required = (directory / "adapter_config.json", directory / "adapter_model.safetensors")
    if not required[0].is_file():
        raise FileNotFoundError(f"Missing adapter config in {checkpoint.uri}")
    if not required[1].is_file() and not (directory / "adapter_model.bin").is_file():
        raise FileNotFoundError(f"Missing adapter weights in {checkpoint.uri}")
    return directory
