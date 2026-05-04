#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python "${ROOT_DIR}/eval.py" \
  --data-root "${SENSEBENCH_DATA_ROOT:-${ROOT_DIR}/data}" \
  --output-root "${SENSEBENCH_OUTPUT_DIR:-${ROOT_DIR}/outputs/inference}" \
  "$@"

