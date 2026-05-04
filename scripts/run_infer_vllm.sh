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

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${#CUDA_DEVICE_ARRAY[@]}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "${ROOT_DIR}/logs" "${SENSEBENCH_OUTPUT_DIR:-${ROOT_DIR}/outputs/inference}"

export SENSEBENCH_BATCH_SIZE="${BATCH_SIZE}"

python "${ROOT_DIR}/infer.py" \
  --model "${MODEL}" \
  --backend vllm \
  --vllm_tensor_parallel_size "${TP_SIZE}" \
  --vllm_gpu_memory_utilization "${GPU_MEM_UTIL}" \
  --vllm_max_num_seqs "${MAX_NUM_SEQS}" \
  --vllm_enforce_eager \
  --truncation_strategy right \
  --enable_thinking false \
  --data_root "${SENSEBENCH_DATA_ROOT:-${ROOT_DIR}/data}" \
  --output_dir "${SENSEBENCH_OUTPUT_DIR:-${ROOT_DIR}/outputs/inference}" \
  2>&1 | tee "${ROOT_DIR}/logs/infer_vllm_$(date +%Y%m%d_%H%M%S).log"
