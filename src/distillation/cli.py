from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from transformers.trainer_utils import IntervalStrategy

from .arguments import DistillationArguments
from .utils import print_rank, resolve_hf_path, seed_everything

_HF_PATH_KEYS = {
    "model_name_or_path": None,
    "adapter_name_or_path": None,
    "ref_model": None,
    "ref_model_adapters": None,
    "dataset_dir": "auto",
}

_REQUIRED_PLACEHOLDERS = {
    "ref_model_adapters": "REPLACE_WITH_TEACHER_ADAPTER",
    "dataset": "REPLACE_WITH_TRAIN_DATASET",
    "eval_dataset": "REPLACE_WITH_EVAL_DATASET",
}


def _validate_deepspeed_platform(deepspeed_config: str | None) -> None:
    """Fail early when a DeepSpeed config is selected outside Linux/WSL2."""

    if deepspeed_config is not None and sys.platform != "linux":
        raise RuntimeError(
            "DeepSpeed training is supported only on Linux/WSL2. "
            "Run this config from a Linux CUDA environment or override deepspeed=null for a non-DeepSpeed SFT run."
        )


def _requires_teacher(config: dict[str, Any]) -> bool:
    """Return whether the merged config needs a reference model."""

    method = str(config.get("distill_method", "fkl"))
    method = DistillationArguments._ALIASES.get(method, method)
    try:
        kd_ratio = float(config.get("kd_ratio", 0.7))
    except (TypeError, ValueError) as exc:
        raise ValueError("kd_ratio must be numeric.") from exc
    return method != "sft" and kd_ratio > 0.0


def _load_config(argv: Sequence[str]) -> dict[str, Any]:
    if not argv:
        raise ValueError("Usage: kd-train <config.yaml> [key=value ...]")
    config_path = Path(argv[0])
    if config_path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("The first argument must be a YAML or JSON configuration file.")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    file_config = OmegaConf.load(config_path)
    override_config = OmegaConf.from_cli(list(argv[1:]))
    config = OmegaConf.to_container(OmegaConf.merge(file_config, override_config), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping.")
    return config


def _resolve_config_paths(config: dict[str, Any]) -> None:
    for key, repo_type in _HF_PATH_KEYS.items():
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            config[key] = [resolve_hf_path(item, repo_type=repo_type) for item in value]
        elif isinstance(value, str):
            config[key] = resolve_hf_path(value, repo_type=repo_type)


def _validate_required_placeholders(config: dict[str, Any]) -> None:
    required_placeholders = dict(_REQUIRED_PLACEHOLDERS)
    if not _requires_teacher(config):
        required_placeholders.pop("ref_model_adapters")
    unresolved = [key for key, placeholder in required_placeholders.items() if config.get(key) == placeholder]
    if unresolved:
        fields = ", ".join(unresolved)
        raise ValueError(
            f"Unresolved YAML placeholders: {fields}. "
            "Replace them in the YAML or pass key=value overrides to the training command."
        )


def parse_arguments(argv: Sequence[str]) -> tuple[Any, Any, Any, Any, Any, DistillationArguments]:
    config = _load_config(argv)
    _validate_required_placeholders(config)
    distillation_args, llamafactory_config = DistillationArguments.split_config(config)
    if not distillation_args.uses_kd:
        # Do not even pass stale teacher fields from a copied KD YAML to
        # LlamaFactory.  Besides being unnecessary, this avoids resolving an
        # hf:// adapter path for a run that cannot use it.
        llamafactory_config.pop("ref_model", None)
        llamafactory_config.pop("ref_model_adapters", None)
    _resolve_config_paths(llamafactory_config)

    # This remains strict: once custom KD fields are removed, LlamaFactory
    # rejects every unknown or deprecated key in the YAML.
    from llamafactory.hparams import get_train_args

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(llamafactory_config)
    if finetuning_args.stage != "sft":
        raise ValueError("KD baselines require LlamaFactory stage=sft.")
    _validate_deepspeed_platform(training_args.deepspeed)
    if distillation_args.uses_kd and finetuning_args.ref_model is None:
        raise ValueError("A teacher must be provided through LlamaFactory ref_model.")
    if distillation_args.uses_kd and training_args.deepspeed is None:
        raise ValueError("KD baselines require a DeepSpeed configuration.")
    if distillation_args.uses_kd and (not training_args.bf16 or training_args.fp16):
        raise ValueError("KD baselines require bf16=true and fp16=false for both student and teacher.")
    if distillation_args.is_adaptive and (
        not training_args.do_eval or training_args.eval_strategy is IntervalStrategy.NO
    ):
        raise ValueError("Adaptive DistiLLM requires do_eval=true and a non-'no' eval_strategy.")
    distillation_args.validate_runtime(
        cutoff_len=data_args.cutoff_len,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
    )
    return model_args, data_args, training_args, finetuning_args, generating_args, distillation_args


def main(argv: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(sys.argv[1:] if argv is None else argv)
    model_args, data_args, training_args, finetuning_args, generating_args, distillation_args = parsed
    seed_everything(training_args.seed, rank_offset=True, deterministic=training_args.full_determinism)
    print_rank(
        f"Starting {distillation_args.distill_method} with LlamaFactory "
        f"and DeepSpeed={training_args.deepspeed is not None}."
    )

    from .workflow import run_kd

    run_kd(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        distillation_args,
    )


if __name__ == "__main__":
    main()
