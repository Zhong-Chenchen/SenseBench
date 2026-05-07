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

The file `data_huggingface/questions.jsonl` is the restoration manifest used to map parquet rows back to image paths. To avoid confusion with the evaluation `questions.jsonl`, rename it first:

```bash
mv data_huggingface/questions.jsonl data_huggingface/restore_manifest.jsonl
```

Then restore the parquet shards into a new folder under `data_huggingface/`:

```bash
python scripts/restore_hf_dataset.py \
  --input-dir data_huggingface \
  --output-dir data_huggingface/original_png \
  --manifest-file data_huggingface/restore_manifest.jsonl \
  --eval-questions-file data/questions.jsonl
```

This will recreate the image folders under `data_huggingface/original_png/` using the paths stored in `restore_manifest.jsonl`, and copy the real evaluation `questions.jsonl` into the restored folder.

If you are working with the released subset instead of the full dataset, use `store.jsonl` as the restoration manifest and keep the compact `questions.jsonl` for evaluation:

```bash
python scripts/restore_hf_dataset.py \
  --input-dir data_huggingface_subset \
  --output-dir data_huggingface_subset/original_png \
  --manifest-file data_huggingface_subset/store.jsonl \
  --eval-questions-file data_huggingface_subset/questions.jsonl
```

In that case, `questions_full.jsonl` is only a local convenience copy of the complete question set and does not need to be used during restoration.

---

## 4. Use the restored data

If you want to run inference or evaluation on the restored PNG layout, point the data root to the restored folder:

```bash
export SENSEBENCH_DATA_ROOT=data_huggingface/original_png
```

---

## 5. Notes

- The Hugging Face dataset is the public distribution used for release and sharing.
- If you want to run the benchmark with the original local layout, keep using the repository `data/` directory or the restored `data_huggingface/original_png/` directory.
- For reproduction and evaluation, always make sure the image files and the metadata files come from the same dataset version.
