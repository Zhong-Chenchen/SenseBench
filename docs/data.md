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

## 3. Use the downloaded data

If you want to use the downloaded Hugging Face data with this repository, point the data root to the downloaded folder:

```bash
export SENSEBENCH_DATA_ROOT=data_huggingface
```

or pass it directly to the scripts:

```bash
python src/infer.py --data_root data_huggingface
python src/eval.py --data-root data_huggingface
```

---

## 4. Notes

- The Hugging Face dataset is the public distribution used for release and sharing.
- If you want to run the benchmark with the original local layout, keep using the repository `data/` directory.
- For reproduction and evaluation, always make sure the image files and the metadata files come from the same dataset version.
