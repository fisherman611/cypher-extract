#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/qwen2.5_coder"
TEACHER_CONFIG="${PROJECT_ROOT}/configs/distillation/teacher_lora_qwen2.5_coder.yaml"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
if [[ "${RESULTS_ROOT}" != /* ]]; then
  RESULTS_ROOT="${PROJECT_ROOT}/${RESULTS_ROOT}"
fi
FAMILY_RESULTS="${RESULTS_ROOT}/qwen2.5_coder"
TEACHER_ADAPTER="${FAMILY_RESULTS}/teacher_lora"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${QWEN2_5_CODER_LOG_DIR:-${FAMILY_RESULTS}/run_all_logs/${RUN_ID}}"
CONFIG_NAMES=(
  sft.yaml
  fkl.yaml
  rkl.yaml
  sfkl.yaml
  srkl.yaml
  csd.yaml
  hpd.yaml
  amid.yaml
  fdd_sfkl.yaml
  fdd_srkl.yaml
  distillm_adaptive_sfkl.yaml
  distillm_adaptive_srkl.yaml
)
for override in "$@"; do
  if [[ "${override}" == resume_from_checkpoint=* ]]; then
    echo "Do not resume through train_all_qwen2_5_coder.sh: one checkpoint cannot be applied to every method." >&2
    echo "Resume one run with scripts/train.sh and its matching config/output_dir." >&2
    exit 2
  fi
done
if [[ ! -f "${TEACHER_CONFIG}" ]]; then
  echo "Missing teacher config: ${TEACHER_CONFIG}" >&2
  exit 2
fi
for config_name in "${CONFIG_NAMES[@]}"; do
  if [[ ! -f "${CONFIG_DIR}/${config_name}" ]]; then
    echo "Missing config: ${CONFIG_DIR}/${config_name}" >&2
    exit 2
  fi
done
fresh_outputs=("${TEACHER_ADAPTER}")
for config_name in "${CONFIG_NAMES[@]}"; do
  fresh_outputs+=("${FAMILY_RESULTS}/${config_name%.yaml}")
done
for output_dir in "${fresh_outputs[@]}"; do
  for checkpoint in "${output_dir}"/checkpoint-*; do
    if [[ -d "${checkpoint}" ]]; then
      echo "Refusing fresh train-all run: old checkpoint found at ${checkpoint}" >&2
      echo "Choose a new RESULTS_ROOT. Resume a single run through scripts/train.sh instead." >&2
      exit 2
    fi
  done
done
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
echo
echo "============================================================"
echo "Running configs/distillation/teacher_lora_qwen2.5_coder.yaml"
echo "Output: ${TEACHER_ADAPTER}"
echo "Log: ${LOG_DIR}/teacher_lora_qwen2.5_coder.log"
echo "============================================================"
# Every full run starts by training the teacher. A failed teacher cannot be
# skipped because all KD methods depend on the adapter it produces.
if bash scripts/train.sh configs/distillation/teacher_lora_qwen2.5_coder.yaml "$@" \
  "output_dir=${TEACHER_ADAPTER}" 2>&1 | tee "${LOG_DIR}/teacher_lora_qwen2.5_coder.log"; then
  echo "Completed: teacher_lora_qwen2.5_coder"
else
  status=${PIPESTATUS[0]}
  echo "Failed: teacher_lora_qwen2.5_coder (exit ${status}); dependent Qwen2.5-Coder runs were not started." >&2
  exit "${status}"
fi
if [[ ! -f "${TEACHER_ADAPTER}/adapter_config.json" ]]; then
  echo "Teacher run completed but did not create ${TEACHER_ADAPTER}/adapter_config.json" >&2
  exit 1
fi
failed=()
for config_name in "${CONFIG_NAMES[@]}"; do
  method="${config_name%.yaml}"
  config_path="configs/qwen2.5_coder/${config_name}"
  log_path="${LOG_DIR}/${method}.log"
  output_dir="${FAMILY_RESULTS}/${method}"
  method_overrides=("output_dir=${output_dir}")
  if [[ "${method}" != "sft" ]]; then
    method_overrides+=("ref_model_adapters=${TEACHER_ADAPTER}")
  fi
  echo
  echo "============================================================"
  echo "Running ${config_path}"
  echo "Output: ${output_dir}"
  echo "Log: ${log_path}"
  echo "============================================================"
  if bash scripts/train.sh "${config_path}" "$@" "${method_overrides[@]}" 2>&1 | tee "${log_path}"; then
    echo "Completed: ${method}"
  else
    status=${PIPESTATUS[0]}
    failed+=("${method}:${status}")
    echo "Failed: ${method} (exit ${status})" >&2
    if [[ "${CONTINUE_ON_ERROR:-0}" != "1" ]]; then
      echo "Set CONTINUE_ON_ERROR=1 to continue after a failed run." >&2
      exit "${status}"
    fi
  fi
done
if (( ${#failed[@]} > 0 )); then
  echo "Failed runs: ${failed[*]}" >&2
  exit 1
fi
echo
echo "All Qwen2.5-Coder configs completed successfully."
echo "Logs: ${LOG_DIR}"
