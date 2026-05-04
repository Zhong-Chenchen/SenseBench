#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${ROOT_DIR}/logs" "${SENSEBENCH_OUTPUT_DIR:-${ROOT_DIR}/outputs/inference}"

python "${ROOT_DIR}/infer.py" \
  --model "${MODEL}" \
  --backend pt \
  --attn "${ATTN_IMPL:-eager}" \
  --data_root "${SENSEBENCH_DATA_ROOT:-${ROOT_DIR}/data}" \
  --output_dir "${SENSEBENCH_OUTPUT_DIR:-${ROOT_DIR}/outputs/inference}" \
  2>&1 | tee "${ROOT_DIR}/logs/infer_pt_$(date +%Y%m%d_%H%M%S).log"

