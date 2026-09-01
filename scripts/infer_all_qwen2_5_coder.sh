#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
python scripts/infer_two_stage.py \
  --checkpoint-root results \
  --methods all \
  --model-family qwen2.5_coder \
  --output-dir results/inference/qwen2.5_coder \
  "$@"
