# SenseBench Data Download

## 1. Install the download tool

Make sure you have `huggingface_hub` installed:

```bash
pip install -U huggingface_hub
```

If you have not logged in to Hugging Face yet, run:

```bash
hf auth login
```

---

## 2. Download the dataset

Run the following command from anywhere:

```bash
hf download Zhongchenchen/SenseBench \
  --repo-type dataset \
  --local-dir data_huggingface
```

This will download the full dataset repository into `data_huggingface`.

---

## 3. Restore PNG folders

If you want the original PNG-style layout, restore the parquet shards into a new folder under `data_huggingface/`:

```bash
python scripts/restore_hf_dataset.py \
  --input-dir data_huggingface \
  --output-dir data_huggingface/original_png
```

This will recreate the image folders under `data_huggingface/original_png/` using the paths stored in `questions.jsonl`.

---

## 4. Use the restored data

If you want to run inference or evaluation on the restored PNG layout, point the data root to the restored folder:

```bash
export SENSEBENCH_DATA_ROOT=data_huggingface/original_png
```

or pass it directly to the scripts:

```bash
python src/infer.py --data_root data_huggingface/original_png
python src/eval.py --data-root data_huggingface/original_png
```

---

## 5. Notes

- The Hugging Face dataset is the public distribution used for release and sharing.
- If you want to run the benchmark with the original local layout, keep using the repository `data/` directory or the restored `data_huggingface/original_png/` directory.
- For reproduction and evaluation, always make sure the image files and the metadata files come from the same dataset version.
