#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/qwen"
TEACHER_CONFIG="${PROJECT_ROOT}/configs/distillation/teacher_sft.yaml"
TEACHER_ADAPTER="${PROJECT_ROOT}/results/qwen3/teacher_lora"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${QWEN_LOG_DIR:-${PROJECT_ROOT}/results/qwen3/run_all_logs/${RUN_ID}}"

CONFIG_NAMES=(
  sft.yaml
  fkl.yaml
  rkl.yaml
  sfkl.yaml
  srkl.yaml
  csd.yaml
  hpd.yaml
  amid.yaml
  da_kd.yaml
  fdd_sfkl.yaml
  fdd_srkl.yaml
  distillm_adaptive_sfkl.yaml
  distillm_adaptive_srkl.yaml
)

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

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo
echo "============================================================"
echo "Running configs/distillation/teacher_sft.yaml"
echo "Log: ${LOG_DIR}/teacher_sft.log"
echo "============================================================"

# Every full run starts by training the teacher. A failed teacher cannot be
# skipped because all KD methods depend on the adapter it produces.
if bash scripts/train.sh configs/distillation/teacher_sft.yaml "$@" 2>&1 | tee "${LOG_DIR}/teacher_sft.log"; then
  echo "Completed: teacher_sft"
else
  status=${PIPESTATUS[0]}
  echo "Failed: teacher_sft (exit ${status}); dependent Qwen runs were not started." >&2
  exit "${status}"
fi

if [[ ! -f "${TEACHER_ADAPTER}/adapter_config.json" ]]; then
  echo "Teacher run completed but did not create ${TEACHER_ADAPTER}/adapter_config.json" >&2
  exit 1
fi

failed=()
for config_name in "${CONFIG_NAMES[@]}"; do
  method="${config_name%.yaml}"
  config_path="configs/qwen/${config_name}"
  log_path="${LOG_DIR}/${method}.log"

  echo
  echo "============================================================"
  echo "Running ${config_path}"
  echo "Log: ${log_path}"
  echo "============================================================"

  if bash scripts/train.sh "${config_path}" "$@" 2>&1 | tee "${log_path}"; then
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
echo "All Qwen configs completed successfully."
echo "Logs: ${LOG_DIR}"
