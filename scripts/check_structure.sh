#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONDONTWRITEBYTECODE=1

test -f src/infer.py
test -d configs
test -f src/eval.py
test -d ms-swift/swift
test -d other_models
test -d build_dataset
test -f data/questions.jsonl
test -d data/images

python - <<'PY'
from pathlib import Path

for filename in ["src/infer.py", "src/eval.py", "src/table_generate.py", "src/api_infer.py", "configs/__init__.py", "configs/infer_config.py", "configs/infer_utils.py", "configs/jsonl_utils.py", "configs/table_rules.json"]:
    source = Path(filename).read_text(encoding="utf-8")
    compile(source, filename, "exec")

models = sorted(p.parent.name for p in Path("other_models").glob("*/rs_infer.py"))
print("RS adapters:", ", ".join(models))
assert "GeoChat" in models
assert "LHRS-Bot" in models
assert "EarthDial" in models
PY

echo "Sensebench structure check passed."
