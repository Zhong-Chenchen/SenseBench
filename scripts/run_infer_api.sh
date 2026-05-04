#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 1
fi

MODEL="${MODEL:-api:qwen3.5-plus}"
mkdir -p "${ROOT_DIR}/logs" "${SENSEBENCH_OUTPUT_DIR:-${ROOT_DIR}/outputs/inference}"

EXTRA_ARGS=()
if [[ -n "${OPENAI_API_BASE:-}" ]]; then
  EXTRA_ARGS+=(--api_base "${OPENAI_API_BASE}")
fi

python "${ROOT_DIR}/infer.py" \
  --model "${MODEL}" \
  --api_key "${OPENAI_API_KEY}" \
  "${EXTRA_ARGS[@]}" \
  --data_root "${SENSEBENCH_DATA_ROOT:-${ROOT_DIR}/data}" \
  --output_dir "${SENSEBENCH_OUTPUT_DIR:-${ROOT_DIR}/outputs/inference}" \
  2>&1 | tee "${ROOT_DIR}/logs/infer_api_$(date +%Y%m%d_%H%M%S).log"

