#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/infer_all.sh <model-family> <setting> [inference options ...]" >&2
  echo "Model families: qwen3, llama3, qwen2.5_coder" >&2
  echo "Settings: lora, lora_normalized, full_finetune, full_finetune_normalized" >&2
  exit 2
fi

MODEL_FAMILY="$1"
SETTING="$2"
shift 2

case "${MODEL_FAMILY}" in
  qwen3 | llama3 | qwen2.5_coder) ;;
  *)
    echo "Unsupported model family: ${MODEL_FAMILY}" >&2
    exit 2
    ;;
esac

# Full-finetune students distill from the lora / lora_normalized teacher, which is
# inferred under its own setting; teacher_full can still be requested via --methods.
FULL_METHODS="sft,fkl,rkl,sfkl,srkl,csd,hpd,amid,fdd_sfkl,fdd_srkl,distillm_adaptive_sfkl,distillm_adaptive_srkl"
case "${SETTING}" in
  lora)
    DEFAULT_CHECKPOINT_ROOT="results/lora"
    DEFAULT_OUTPUT_ROOT="results/inference/lora"
    METHODS="all"
    ;;
  lora_normalized)
    DEFAULT_CHECKPOINT_ROOT="results/lora_normalized"
    DEFAULT_OUTPUT_ROOT="results/inference/lora_normalized"
    METHODS="all"
    ;;
  full_finetune)
    DEFAULT_CHECKPOINT_ROOT="results/full_finetune"
    DEFAULT_OUTPUT_ROOT="results/inference/full_finetune"
    METHODS="${FULL_METHODS}"
    ;;
  full_finetune_normalized)
    DEFAULT_CHECKPOINT_ROOT="results/full_finetune_normalized"
    DEFAULT_OUTPUT_ROOT="results/inference/full_finetune_normalized"
    METHODS="${FULL_METHODS}"
    ;;
  *)
    echo "Unsupported inference setting: ${SETTING}" >&2
    exit 2
    ;;
esac

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DEFAULT_CHECKPOINT_ROOT}}"

# Optional selector settings (leave unset for the default greedy selector on the test split):
#   SELECTOR_THRESHOLD=0.7   label a unit YES when P(YES) >= 0.7
#   INFERENCE_SPLIT=dev      run the dev split, e.g. to tune the threshold
# Each combination gets its own default output root because a run directory
# cannot be resumed under a different selector configuration.
EXTRA_ARGS=()
OUTPUT_SUFFIX=""
if [[ -n "${INFERENCE_SPLIT:-}" ]]; then
  EXTRA_ARGS+=(--split "${INFERENCE_SPLIT}")
  if [[ "${INFERENCE_SPLIT}" != "test" ]]; then
    OUTPUT_SUFFIX+="_${INFERENCE_SPLIT}"
  fi
fi
if [[ -n "${SELECTOR_THRESHOLD:-}" ]]; then
  EXTRA_ARGS+=(--selector-threshold "${SELECTOR_THRESHOLD}")
  OUTPUT_SUFFIX+="_thr${SELECTOR_THRESHOLD}"
fi
OUTPUT_ROOT="${INFERENCE_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}${OUTPUT_SUFFIX}}"

cd "${PROJECT_ROOT}"
python scripts/infer_two_stage.py \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --methods "${METHODS}" \
  --model-family "${MODEL_FAMILY}" \
  --output-dir "${OUTPUT_ROOT}/${MODEL_FAMILY}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
