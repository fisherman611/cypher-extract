#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_OVERRIDES=("$@")

for override in "${TRAIN_OVERRIDES[@]}"; do
  if [[ "${override}" == output_dir=* ]]; then
    echo "Do not pass output_dir to this multi-run script; each job has its own output directory." >&2
    exit 2
  fi
done

cd "${PROJECT_ROOT}"

has_final_model_weights() {
  local output_dir="$1"
  local filename
  local weight_files=(
    adapter_model.safetensors
    adapter_model.bin
    model.safetensors
    model.safetensors.index.json
    pytorch_model.bin
    pytorch_model.bin.index.json
  )

  for filename in "${weight_files[@]}"; do
    if [[ -s "${output_dir}/${filename}" ]]; then
      return 0
    fi
  done
  return 1
}

training_is_complete() {
  local output_dir="$1"

  [[ -s "${output_dir}/train_results.json" ]] || return 1
  [[ -s "${output_dir}/trainer_state.json" ]] || return 1
  has_final_model_weights "${output_dir}"
}

run_student() {
  local label="$1"
  local config_path="$2"
  local output_dir="$3"

  if training_is_complete "${PROJECT_ROOT}/${output_dir}"; then
    echo "Training already completed; skipping ${label}"
    echo "Output: ${PROJECT_ROOT}/${output_dir}"
    return 0
  fi

  echo
  echo "============================================================"
  echo "Training ${label}"
  echo "Config: ${config_path}"
  echo "Output: ${output_dir}"
  echo "============================================================"
  bash scripts/train.sh "${config_path}" "${TRAIN_OVERRIDES[@]}"
}

echo "Running Qwen3 normalized LoRA teacher through run_teacher_student.sh"
bash scripts/run_teacher_student.sh \
  --families qwen3 \
  --settings lora_normalized \
  --student-methods none \
  --phase train \
  -- "${TRAIN_OVERRIDES[@]}"

run_student \
  "Qwen3 SFT full fine-tune" \
  "configs/qwen3_full_finetune/sft.yaml" \
  "results/full_finetune/qwen3/sft"

run_student \
  "Qwen3 SFT full fine-tune with normalized loss" \
  "configs/qwen3_full_finetune_normalized_loss/sft.yaml" \
  "results/full_finetune_normalized/qwen3/sft"

echo
echo "Qwen3 SFT trial completed successfully."
