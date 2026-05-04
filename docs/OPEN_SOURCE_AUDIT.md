# Open-Source Cleanup Audit

## Core Chain

`infer.py` loads one inference engine through `configs.infer_utils.load_inference_engine()`, reads the unified `data/questions.jsonl` file by default, resolves image paths under `data`, and writes model outputs to `outputs/inference/<model>/predictions.jsonl`. The older `data/bench/{single,Multi}/problems` task enumeration remains available with `--output-format legacy`.

`configs/infer_utils.py` keeps three inference paths:

- OpenAI-compatible API models through `APIEngineAdapter`.
- Remote-sensing adapters discovered from `RS_models/*/rs_infer.py`.
- Standard local VLMs through the vendored `ms-swift` package in `./ms-swift`.

`eval.py` reads `predictions.jsonl`, writes `evaluated.jsonl`, and optionally uses an OpenAI-compatible LLM judge for description metrics. It still accepts the older per-task JSON output layout.

## Kept

- `infer.py`, `configs/infer_utils.py`, `configs/infer_config.py`, `eval.py`, `table_generate.py`, and `configs/table_rules.json`: benchmark inference, evaluation, and table generation.
- `data/`: materialized benchmark data and `add/` build inputs, preserving the original JSON/image path contract.
- `build_dataset/`: distortion operators, question generation scripts, and merge/mark utilities.
- `RS_models/EarthDial`, `RS_models/GeoChat`, `RS_models/LHRS-Bot`: adapters currently discoverable by `infer_utils`.
- `RS_models/TEOChat`: retained as optional model source code, but not enabled in the benchmark adapter registry because it has no `rs_infer.py`.
- `ms-swift/`: local inference dependency for standard VLMs; docs/examples/tests/dev metadata were trimmed, but the `swift` package and install metadata remain.
- `scripts/`: small, parameterized entrypoints for PT, vLLM, API inference, evaluation, and structure checks.

## Not Kept

- `cache_dir/`: model/download cache and OSS URL map cache.
- `output/`, `log/`, `log1/`: generated inference/evaluation artifacts.
- `__pycache__/`, `.DS_Store`, `.vscode/`, `.claude/`, `.env`: local machine/editor/runtime files.
- `delete/`: scratch plotting/test artifacts.
- Most of `codex_code/`: paper collection, upload, batch helper, and one-off experimental scripts not needed for the benchmark core.
- Duplicate top-level scripts such as `infer copy.py` and `eval_1.py`.
- Destructive or notebook-only build helpers from `build_dataset`.

## Risks

- `ms-swift` is required for non-`rs:` local models; do not remove it unless replacing the import path with a documented package dependency.
- `GeoChat` and `LHRS-Bot` need their internal source packages, not just `rs_infer.py`.
- RS checkpoints are not included. Use `--rs_model_path NAME=PATH` or the documented environment variables.
- `TEOChat` remains optional source code only until a compatible `RS_models/TEOChat/rs_infer.py` adapter is added.
