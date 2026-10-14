from pathlib import Path

import pytest
from huggingface_hub.utils import filter_repo_objects

from schema_grounding.inference import hub
from schema_grounding.inference.checkpoints import LastCheckpoint, resolve_checkpoint_directory

CHECKPOINT_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "global_step{step}/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
    "global_step{step}/mp_rank_00_model_states.pt",
    "latest",
    "resume_manifest.json",
    "rng_state_0.pth",
    "scheduler.pt",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    "zero_to_fp32.py",
    "zero_to_torch.py",
)


def checkpoint(prefix: str, step: int) -> list[str]:
    return [f"{prefix}/checkpoint-{step}/{name.format(step=step)}" for name in CHECKPOINT_FILES]


REPO_FILES = [
    ".gitattributes",
    *checkpoint("lora/qwen3/fkl", 519),
    *checkpoint("lora/qwen3/fkl", 2595),
    *checkpoint("lora/qwen3/fkl", 1038),
    *checkpoint("lora/qwen3/teacher_lora", 2595),
    *checkpoint("lora_normalized/qwen3/fkl", 3000),
    "lora/qwen3/fkl/eval/predictions.jsonl",
    "inference/lora/qwen3/seed42/fkl/cypherbench/metrics.json",
]


def test_latest_checkpoint_steps_picks_the_highest_step_per_method() -> None:
    steps = hub.latest_checkpoint_steps(
        REPO_FILES, subfolder="lora", model_family="qwen3", methods=["fkl", "teacher_lora"]
    )
    assert steps == {"fkl": 2595, "teacher_lora": 2595}
    normalized = hub.latest_checkpoint_steps(
        REPO_FILES, subfolder="lora_normalized", model_family="qwen3", methods=["fkl"]
    )
    assert normalized == {"fkl": 3000}


def test_latest_checkpoint_steps_names_the_missing_method() -> None:
    with pytest.raises(FileNotFoundError, match="'srkl' under lora/qwen3/srkl/ in owner/repo"):
        hub.latest_checkpoint_steps(
            REPO_FILES, subfolder="lora", model_family="qwen3", methods=["fkl", "srkl"], repo_id="owner/repo"
        )
    with pytest.raises(FileNotFoundError, match="llama3"):
        hub.latest_checkpoint_steps(REPO_FILES, subfolder="lora", model_family="llama3", methods=["fkl"])


def test_download_requests_only_the_newest_checkpoints_without_training_state(monkeypatch, tmp_path: Path) -> None:
    calls: dict = {}
    monkeypatch.setattr(hub, "_list_repo_files", lambda repo_id, revision, token: REPO_FILES)

    def fake_download(repo_id, *, local_dir, allow_patterns, revision, token):
        calls.update(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns, revision=revision, token=token)

    monkeypatch.setattr(hub, "_download", fake_download)
    root = hub.download_hf_checkpoints(
        "owner/repo",
        subfolder="lora",
        model_family="qwen3",
        methods=["fkl", "teacher_lora"],
        local_dir=tmp_path / "owner__repo",
        revision="abc123",
        token="secret",
    )

    assert root == tmp_path / "owner__repo" / "lora"
    assert (tmp_path / "owner__repo").is_dir()
    assert calls["revision"] == "abc123" and calls["token"] == "secret"
    assert calls["allow_patterns"] == [
        "lora/qwen3/fkl/checkpoint-2595/*",
        "lora/qwen3/teacher_lora/checkpoint-2595/*",
    ]

    # Apply the same filters Hugging Face uses to confirm which files would be fetched.
    selected = set(
        filter_repo_objects(
            REPO_FILES,
            allow_patterns=calls["allow_patterns"],
            ignore_patterns=list(hub.HF_CHECKPOINT_IGNORE_PATTERNS),
        )
    )
    fkl = "lora/qwen3/fkl/checkpoint-2595/"
    assert {path[len(fkl) :] for path in selected if path.startswith(fkl)} == {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "resume_manifest.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
    }
    assert not any("checkpoint-519" in path or "checkpoint-1038" in path for path in selected)
    assert not any("global_step" in path or path.endswith(("optimizer.pt", "scheduler.pt")) for path in selected)
    assert not any(path.startswith(("inference/", "lora_normalized/")) for path in selected)


def test_download_returns_a_root_that_the_local_checkpoint_resolver_accepts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hub, "_list_repo_files", lambda repo_id, revision, token: REPO_FILES)

    def fake_download(repo_id, *, local_dir, allow_patterns, revision, token):
        directory = local_dir / "lora" / "qwen3" / "fkl" / "checkpoint-2595"
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(hub, "_download", fake_download)
    root = hub.download_hf_checkpoints(
        "owner/repo", subfolder="lora", model_family="qwen3", methods=["fkl"], local_dir=tmp_path / "cache"
    )
    resolved = LastCheckpoint(str(root), "qwen3", "fkl", 2595, "qwen3/fkl/checkpoint-2595")
    assert resolve_checkpoint_directory(resolved) == root / "qwen3" / "fkl" / "checkpoint-2595"
