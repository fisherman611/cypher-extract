from pathlib import Path


def test_qwen3_sft_trial_runs_only_requested_training_configs() -> None:
    script = Path("scripts/run_qwen3_sft_trial.sh").read_text(encoding="utf-8")

    assert "--settings lora_normalized" in script
    assert "--student-methods none" in script
    assert '"configs/qwen3_full_finetune/sft.yaml"' in script
    assert '"configs/qwen3_full_finetune_normalized_loss/sft.yaml"' in script
    assert "teacher_full_qwen3" not in script


def test_qwen3_sft_trial_skips_completed_students() -> None:
    script = Path("scripts/run_qwen3_sft_trial.sh").read_text(encoding="utf-8")

    assert '[[ -s "${output_dir}/train_results.json" ]]' in script
    assert '[[ -s "${output_dir}/trainer_state.json" ]]' in script
    assert 'has_final_model_weights "${output_dir}"' in script
    assert 'if training_is_complete "${PROJECT_ROOT}/${output_dir}"; then' in script


def test_qwen3_sft_trial_rejects_shared_output_override() -> None:
    script = Path("scripts/run_qwen3_sft_trial.sh").read_text(encoding="utf-8")

    assert 'if [[ "${override}" == output_dir=* ]]; then' in script
