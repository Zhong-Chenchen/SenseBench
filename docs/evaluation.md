# 📊 Evaluation Workflow

This page explains the full SenseBench evaluation pipeline:

1. `infer` runs model predictions.
2. `eval` scores predictions against the ground truth.
3. `table_generate` aggregates the final numbers into paper-style tables.

For `Description`, the evaluation step must use `llm judge`.

---

## 🌱 Environment

Create a dedicated environment for SenseBench and install the project dependencies.

Example:

```bash
conda create -n sensebench python=3.10 -y
conda activate sensebench
pip install -r requirements.txt
```

If you want to use a local `llm judge` server for `Description`, install and start that server in a separate environment before running evaluation.

### LLM-as-Judge Environment

For `Description`, SenseBench follows the same pattern as CHOICE: run the evaluator in the main environment, and run the judge model in a separate serving environment.

Create a second environment for the judge:

```bash
conda create -n sensebench-judge python=3.10 -y
conda activate sensebench-judge
pip install lmdeploy openai
```

Start an OpenAI-compatible local server with LMDeploy:

```bash
export LMDEPLOY_USE_MODELSCOPE=True

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='0,1' \
lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct --server-port 23333 --tp 2
```

The evaluator uses the OpenAI-compatible endpoint at `http://0.0.0.0:23333/v1` by default, so you only need to keep the server running while `eval.py` is executing.

---

## 🚀 Run Inference

Run all commands from the repository root.

Use `infer.py` to generate predictions from the benchmark data.

Example:

```bash
python infer.py \
  --model AIDC-AI/Ovis2.5-9B \
  --data_root data \
  --output_dir outputs/inference \
  --output-format both
```

Key points:

- `--input-jsonl` lets you point to a custom unified JSONL file.
- If `--input-jsonl` is omitted, the script looks for `<data_root>/questions.jsonl`.
- `--output-format jsonl` writes `predictions.jsonl`.
- `--output-format legacy` writes the legacy task folders.
- `--output-format both` writes both layouts.

Model outputs are saved under:

```text
<output_dir>/<model_name>/
```

---

## 📝 Run Evaluation

`eval.py` reads the predictions and computes task-level scores.

Basic evaluation:

```bash
python eval.py \
  --output-root outputs/inference \
  --data-root data
```

This will:

- read each model folder under `--output-root`
- score the prediction files
- write `evaluated.jsonl` back into each model directory

### Template-based scoring

For the standard perception tasks, the evaluator can score answers directly from the predicted option strings.

### `Description` with `llm judge`

`Description` is not a simple template-matching task. To evaluate it properly, enable the LLM-based scorer:

```bash
python eval.py \
  --output-root outputs/inference \
  --data-root data \
  --description-llm \
  --llm-base-url http://0.0.0.0:23333/v1 \
  --llm-api-key sk-123456
```

This produces description scores for:

- `completeness`
- `correctness`
- `faithfulness`

If you also want the multiple-choice tasks to use the LLM judge, add `--use-llm`.

Optional flag:

```bash
--write-intra-splits
```

Use this when you want per-split json files for the multi-image perception statistics.

---

## 📋 Generate Tables

After evaluation, use `table_generate.py` to build the summary tables.

Example:

```bash
python table_generate.py \
  --output-root outputs/inference \
  --out-dir outputs/inference/tables
```

Supported table names:

- `single_perception`
- `multi_perception`
- `description_single`
- `description_multi`

By default, the script writes xlsx tables into the output folder.

---

## 📁 Output Files

The main files you will see are:

- `predictions.jsonl`: raw inference outputs
- `evaluated.jsonl`: scored outputs after `eval.py`
- `*.xlsx`: final summary tables from `table_generate.py`

For `Description`, the important step is to enable `--description-llm`, otherwise the description metrics will not be computed the way the benchmark expects.
