# Sensebench

Sensebench is the cleaned open-source layout of the RS image-quality benchmark code from `RSIQA-main`. It keeps the real benchmark inference, evaluation, dataset-building, RS model adapters, and local `ms-swift` integration while removing generated caches, logs, outputs, and one-off experiment residue.

## What Is Included

- `infer.py`: benchmark inference entrypoint.
- `eval.py`: benchmark evaluation and summary export.
- `configs/`: core inference configs, JSONL helpers, and table-generation rules.
- `RS_models/`: RS adapter layer and retained model source code.
- `ms-swift/`: vendored local `ms-swift` dependency used by standard VLM inference.
- `build_dataset/`: distortion and question-building utilities.
- `configs/`: table-generation rules and other small static configs.
- `data/`: benchmark data layout expected by inference/evaluation.
- `scripts/`: stable command-line entrypoints.

## Install

```bash
cd /home/anxiao/zhongchen/Sensebench
pip install -r requirements.txt
pip install -e ./ms-swift
```

If you already have a working environment, keep using it. The scripts accept `CONDA_ENV=<env>` but do not hardcode a local environment name.

## Inference

Standard local model through `ms-swift` PT backend:

```bash
MODEL=Qwen/Qwen3-VL-8B-Instruct CUDA_VISIBLE_DEVICES=0 bash scripts/run_infer_pt.sh
```

Standard local model through vLLM:

```bash
MODEL=Qwen/Qwen3-VL-8B-Instruct CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_infer_vllm.sh
```

OpenAI-compatible API model:

```bash
OPENAI_API_KEY=sk-... MODEL=api:gpt-4o bash scripts/run_infer_api.sh
```

RS adapters:

```bash
MODEL=rs:GeoChat SENSEBENCH_GEOCHAT_MODEL_PATH=/path/to/geochat-7B bash scripts/run_infer_pt.sh
MODEL=rs:LHRS-Bot SENSEBENCH_LHRS_BOT_MODEL_PATH=/path/to/LHRS-Bot bash scripts/run_infer_pt.sh
MODEL=rs:EarthDial SENSEBENCH_EARTHDIAL_MODEL_PATH=/path/to/EarthDial_4B_RGB bash scripts/run_infer_pt.sh
```

You can also pass checkpoint overrides directly:

```bash
python infer.py --model rs:GeoChat --rs_model_path GeoChat=/path/to/geochat-7B
```

## Evaluation

```bash
bash scripts/run_eval.sh
```

Inference now defaults to the unified JSONL flow:

- input questions: `data/questions.jsonl`
- raw model outputs: `outputs/inference/<model>/predictions.jsonl`
- evaluated outputs: `outputs/inference/<model>/evaluated.jsonl`
- summary matrices: `outputs/inference/results_*.csv|xlsx`

To generate paper-style tables after evaluation:

```bash
python table_generate.py --output-root outputs/inference --out-dir outputs/inference/tables
```

Default output root is `./outputs/inference`. Override it with:

```bash
SENSEBENCH_OUTPUT_DIR=/path/to/results bash scripts/run_eval.sh
```

## Data

Default data root is `./data`, and `infer.py` automatically reads `./data/questions.jsonl` when present. Override it with `SENSEBENCH_DATA_ROOT=/path/to/data`, `SENSEBENCH_QUESTIONS_JSONL=/path/to/questions.jsonl`, or `python infer.py --data_root /path/to/data --input-jsonl /path/to/questions.jsonl`.

See `docs/DATA.md` for the expected layout.

## Sanity Check

```bash
bash scripts/check_structure.sh
```

This checks the core files, dataset directories, RS adapter discovery, and Python syntax without loading model weights.

## Cleanup Notes

The detailed retention/removal rationale is in `docs/OPEN_SOURCE_AUDIT.md`.
