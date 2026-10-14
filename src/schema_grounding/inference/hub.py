from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from schema_grounding.inference.checkpoints import select_last_checkpoint

# Training state is large and not needed to run a checkpoint for inference.
HF_CHECKPOINT_IGNORE_PATTERNS = (
    "*/global_step*/*",
    "*/optimizer.pt",
    "*/scheduler.pt",
    "*/rng_state*.pth",
    "*/training_args.bin",
    "*/zero_to_fp32.py",
    "*/zero_to_torch.py",
    "*/latest",
    "*/distillm_state_rank*.pt",
)


def latest_checkpoint_steps(
    repo_files: Iterable[str],
    *,
    subfolder: str,
    model_family: str,
    methods: Sequence[str],
    repo_id: str = "the repository",
) -> dict[str, int]:
    """Return the newest ``checkpoint-N`` step of every method in a repository file listing."""

    files = [path for path in repo_files]
    steps: dict[str, int] = {}
    for method in methods:
        prefix = f"{subfolder}/{model_family}/{method}"
        pattern = re.compile(rf"^{re.escape(prefix)}/(checkpoint-\d+)/")
        checkpoint_directories = {
            f"{prefix}/{match.group(1)}" for path in files if (match := pattern.match(path)) is not None
        }
        if not checkpoint_directories:
            raise FileNotFoundError(f"No checkpoint-N directory for {method!r} under {prefix}/ in {repo_id}")
        steps[method], _ = select_last_checkpoint(checkpoint_directories, prefix)
    return steps


def _list_repo_files(repo_id: str, revision: str | None, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi(token=token).list_repo_files(repo_id, repo_type="model", revision=revision)


def _download(
    repo_id: str,
    *,
    local_dir: Path,
    allow_patterns: list[str],
    revision: str | None,
    token: str | None,
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=list(HF_CHECKPOINT_IGNORE_PATTERNS),
        token=token,
    )


def download_hf_checkpoints(
    repo_id: str,
    *,
    subfolder: str,
    model_family: str,
    methods: Sequence[str],
    local_dir: Path,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download the inference files of each method's newest checkpoint and return the checkpoint root.

    The repository mirrors ``results/``: ``<subfolder>/<model_family>/<method>/checkpoint-N/``.
    Files are written under ``local_dir`` with the same layout, so the returned
    ``local_dir/<subfolder>`` works as a local ``--checkpoint-root``. Already downloaded
    files are not fetched again.
    """

    steps = latest_checkpoint_steps(
        _list_repo_files(repo_id, revision, token),
        subfolder=subfolder,
        model_family=model_family,
        methods=methods,
        repo_id=repo_id,
    )
    allow_patterns = [
        f"{subfolder}/{model_family}/{method}/checkpoint-{step}/*" for method, step in steps.items()
    ]
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    _download(repo_id, local_dir=local_dir, allow_patterns=allow_patterns, revision=revision, token=token)
    return local_dir / subfolder
