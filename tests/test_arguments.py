from pathlib import Path

import pytest
import yaml

from distillation.arguments import DistillationArguments
from distillation.cli import _validate_deepspeed_platform, _validate_required_placeholders

CONFIG_PATHS = sorted(
    path
    for family in ("llama3", "qwen3")
    for path in (Path("configs") / family).glob("*.yaml")
)
ALL_TRAIN_CONFIG_PATHS = sorted(
    path
    for directory in ("distillation", "llama3", "qwen3")
    for path in (Path("configs") / directory).glob("*.yaml")
)
REDUNDANT_RUNTIME_DEFAULTS = {
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "fp16",
    "gradient_checkpointing",
    "logging_strategy",
}
BASELINE_CONFIG_NAMES = {
    "csd.yaml",
    "amid.yaml",
    "da_kd.yaml",
    "distillm_adaptive_sfkl.yaml",
    "distillm_adaptive_srkl.yaml",
    "fdd_sfkl.yaml",
    "fdd_srkl.yaml",
    "fkl.yaml",
    "hpd.yaml",
    "rkl.yaml",
    "sfkl.yaml",
    "srkl.yaml",
    "sft.yaml",
}


@pytest.mark.parametrize("family", ["llama3", "qwen3"])
def test_each_model_family_has_all_baseline_configs(family: str) -> None:
    config_paths = list((Path("configs") / family).glob("*.yaml"))
    actual = {path.name for path in config_paths}
    assert actual == BASELINE_CONFIG_NAMES


@pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIG_PATHS)
def test_train_configs_omit_redundant_runtime_defaults(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert REDUNDANT_RUNTIME_DEFAULTS.isdisjoint(config)


@pytest.mark.parametrize(
    ("teacher_path", "kd_path"),
    [
        ("configs/distillation/teacher_lora_qwen3.yaml", "configs/qwen3/fkl.yaml"),
        ("configs/distillation/teacher_lora_llama3.yaml", "configs/llama3/fkl.yaml"),
    ],
)
def test_teacher_lora_output_is_wired_into_family_kd_configs(teacher_path: str, kd_path: str) -> None:
    teacher = yaml.safe_load(Path(teacher_path).read_text(encoding="utf-8"))
    kd = yaml.safe_load(Path(kd_path).read_text(encoding="utf-8"))

    assert teacher["distill_method"] == "sft"
    assert teacher["kd_ratio"] == 0.0
    assert teacher["finetuning_type"] == "lora"
    assert "ref_model" not in teacher
    assert "ref_model_adapters" not in teacher
    assert teacher["model_name_or_path"] == kd["ref_model"]
    assert teacher["output_dir"] == kd["ref_model_adapters"]
    assert teacher["dataset"] == kd["dataset"]
    assert teacher["eval_dataset"] == kd["eval_dataset"]
    assert teacher["dataset_dir"] == kd["dataset_dir"]


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/distillation/student_sft.yaml"),
        Path("configs/llama3/sft.yaml"),
        Path("configs/qwen3/sft.yaml"),
    ],
)
def test_student_sft_configs_use_full_finetuning(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["finetuning_type"] == "full"
    assert not any(key.startswith("lora_") for key in config)


@pytest.mark.parametrize("config_path", sorted(Path("configs/qwen3").glob("*.yaml")))
def test_qwen_configs_follow_template_defaults(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model_name_or_path"] == "Qwen/Qwen3-0.6B"
    if config["distill_method"] != "sft":
        assert config["ref_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert config["template"] == "qwen3_nothink"
    assert config["cutoff_len"] == 892
    assert config["per_device_train_batch_size"] == 2
    assert config["per_device_eval_batch_size"] == 16
    assert config["gradient_accumulation_steps"] == 8
    if config["distill_method"].startswith("fdd_"):
        assert config["student_layer_mapping"] == [13, 27]
        assert config["teacher_layer_mapping"] == [17, 35]


def test_adaptive_method_requires_student_generation() -> None:
    with pytest.raises(ValueError, match="requires student_gen=true"):
        DistillationArguments(distill_method="adaptive_srkl")


def test_static_method_rejects_student_generation() -> None:
    with pytest.raises(ValueError, match="only valid for adaptive"):
        DistillationArguments(distill_method="rkl", student_gen=True)


def test_fdd_requires_two_matching_layer_lists() -> None:
    with pytest.raises(ValueError, match="at least two"):
        DistillationArguments(
            distill_method="fdd_srkl",
            fdd_weight=0.2,
            student_layer_mapping=[7],
            teacher_layer_mapping=[15],
        )


def test_fdd_sfkl_uses_sfkl_token_divergence() -> None:
    args = DistillationArguments(
        distill_method="fdd_sfkl",
        fdd_weight=0.2,
        student_layer_mapping=[7, 15],
        teacher_layer_mapping=[15, 31],
    )
    assert args.uses_fdd is True
    assert args.base_method == "sfkl"


def test_method_alias_is_canonicalized() -> None:
    args = DistillationArguments(distill_method="adaptive-srkl", student_gen=True)
    assert args.distill_method == "adaptive_srkl"
    assert args.base_method == "srkl"


def test_amid_method_is_supported() -> None:
    args = DistillationArguments(distill_method="amid", amid_div_name="ab", amid_div_order="rq")
    assert args.base_method == "amid"


def test_hpd_method_is_supported() -> None:
    args = DistillationArguments(distill_method="hpd", hpd_sample_in_fp32=False)
    assert args.base_method == "hpd"
    assert args.hpd_sample_in_fp32 is False


def test_explicit_sft_mode_is_teacher_free() -> None:
    args = DistillationArguments(distill_method="sft", kd_ratio=0.0)
    assert args.is_sft is True
    assert args.uses_kd is False
    assert args.uses_da_kd is False
    assert args.base_method == "sft"


def test_zero_kd_ratio_is_sft_for_existing_methods() -> None:
    args = DistillationArguments(distill_method="da_kd", kd_ratio=0.0)
    assert args.is_sft is True
    assert args.uses_kd is False
    assert args.uses_da_kd is False


def test_zero_kd_ratio_ignores_stale_adaptive_options() -> None:
    args = DistillationArguments(
        distill_method="adaptive_srkl",
        kd_ratio=0.0,
        student_gen=True,
        fdd_weight=0.2,
        student_layer_mapping=[7, 15],
        teacher_layer_mapping=[15, 31],
    )
    assert args.is_sft is True
    assert args.is_adaptive is False
    assert args.uses_fdd is False


def test_explicit_sft_rejects_nonzero_kd_ratio() -> None:
    with pytest.raises(ValueError, match="distill_method=sft requires kd_ratio=0"):
        DistillationArguments(distill_method="sft", kd_ratio=0.5)


def test_da_kd_uses_bdl_and_dynamic_data_updates() -> None:
    args = DistillationArguments(distill_method="da-kd", da_kd_schedule="linear")
    assert args.distill_method == "da_kd"
    assert args.base_method == "bdl"
    assert args.uses_da_kd is True


def test_da_kd_rejects_zero_bdl_and_invalid_audit_size() -> None:
    with pytest.raises(ValueError, match="identically zero"):
        DistillationArguments(distill_method="da_kd", bdl_lambda=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        DistillationArguments(distill_method="da_kd", da_kd_audit_samples=-1)


@pytest.mark.parametrize("family", ["qwen3", "llama3"])
def test_da_kd_configs_keep_five_epochs_and_enable_cypher_audit(family: str) -> None:
    config = yaml.safe_load((Path("configs") / family / "da_kd.yaml").read_text(encoding="utf-8"))
    assert config["num_train_epochs"] == 5
    assert config["da_kd_audit_samples"] == 10


@pytest.mark.parametrize("field, value", [("amid_div_name", "js"), ("amid_div_order", "pp")])
def test_amid_arguments_validate_choices(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="amid"):
        DistillationArguments(**{field: value})


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_baseline_config_student_generation_matrix(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    distillation_args, _ = DistillationArguments.split_config(config)
    if distillation_args.uses_kd:
        expected_adapter = (
            "results/qwen3/teacher_lora" if config_path.parent.name == "qwen3" else "results/llama3/teacher_lora"
        )
        assert config["ref_model_adapters"] == expected_adapter
    else:
        assert "ref_model_adapters" not in config
    assert config["dataset"] == "cypher_prepared_train"
    assert config["eval_dataset"] == "cypher_prepared_eval"
    assert config["dataset_dir"] == "data/llamafactory"
    expected_template = "qwen3_nothink" if config_path.parent.name == "qwen3" else "llama3"
    assert config["template"] == expected_template
    assert config["packing"] is False
    assert config["ignore_pad_token_for_loss"] is True
    assert config["train_on_prompt"] is False
    assert "mask_history" not in config
    assert config["predict_with_generate"] is True
    assert config["dataloader_num_workers"] == 1
    assert config["dataloader_drop_last"] is False
    assert config["optim"] == "adamw_torch"
    assert config["lr_scheduler_type"] == "cosine_with_min_lr"
    expected_min_lr_rate = 0.01 if distillation_args.uses_da_kd else 0.001
    assert config["lr_scheduler_kwargs"] == {"min_lr_rate": expected_min_lr_rate}
    if distillation_args.is_adaptive:
        assert config["student_gen"] is True
        expected_context_length = 797 if config_path.parent.name == "qwen3" else 810
        assert config["rollout_context_length"] == expected_context_length
        assert config["repetition_penalty"] == 1.0
    else:
        adaptive_only = {
            "student_gen",
            "gen_top_p",
            "init_threshold",
            "loss_eps",
            "capacity",
            "replay_ratio",
        }
        assert adaptive_only.isdisjoint(config)
        assert "rollout_context_length" not in config


def test_unresolved_required_placeholders_are_rejected() -> None:
    with pytest.raises(ValueError, match="ref_model_adapters, dataset, eval_dataset"):
        _validate_required_placeholders(
            {
                "ref_model_adapters": "REPLACE_WITH_TEACHER_ADAPTER",
                "dataset": "REPLACE_WITH_TRAIN_DATASET",
                "eval_dataset": "REPLACE_WITH_EVAL_DATASET",
            }
        )


def test_overridden_required_placeholders_are_accepted() -> None:
    _validate_required_placeholders(
        {
            "ref_model_adapters": "hf://owner/repo/adapter",
            "dataset": "my_train",
            "eval_dataset": "my_eval",
        }
    )


def test_sft_does_not_require_teacher_placeholder() -> None:
    _validate_required_placeholders(
        {
            "distill_method": "sft",
            "kd_ratio": 0.0,
            "dataset": "my_train",
            "eval_dataset": "my_val",
        }
    )


def test_deepspeed_platform_validation_allows_linux(monkeypatch) -> None:
    monkeypatch.setattr("distillation.cli.sys.platform", "linux")
    _validate_deepspeed_platform("configs/deepspeed/ds_config_bf16.json")


def test_deepspeed_platform_validation_rejects_windows(monkeypatch) -> None:
    monkeypatch.setattr("distillation.cli.sys.platform", "win32")
    with pytest.raises(RuntimeError, match="Linux/WSL2"):
        _validate_deepspeed_platform("configs/deepspeed/ds_config_bf16.json")
    _validate_deepspeed_platform(None)


def test_runtime_capacity_validation() -> None:
    args = DistillationArguments(
        distill_method="adaptive_sfkl",
        student_gen=True,
        capacity=2,
        rollout_context_length=50,
    )
    with pytest.raises(ValueError, match="one per-device"):
        args.validate_runtime(cutoff_len=100, per_device_train_batch_size=4)


def test_adaptive_method_validates_rollout_context_limit() -> None:
    args = DistillationArguments(distill_method="adaptive_sfkl", student_gen=True, rollout_context_length=128)
    with pytest.raises(ValueError, match="rollout_context_length"):
        args.validate_runtime(cutoff_len=128, per_device_train_batch_size=1)
