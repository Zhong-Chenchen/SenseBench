"""Convert SenseBench data to Parquet format and upload to Hugging Face."""

import json
import argparse
from pathlib import Path

from datasets import Dataset, Features, Value, Image
from huggingface_hub import HfApi


FEATURES = Features({
    "id": Value("string"),
    "image_1": Image(),
    "image_2": Image(),
    "question": Value("string"),
    "answer": Value("string"),
    "image_count": Value("string"),
    "modality": Value("string"),
    "task": Value("string"),
    "domain": Value("string"),
    "distortion_family": Value("string"),
    "distortion_type": Value("string"),
    "distortion_complexity": Value("string"),
    "comparison": Value("string"),
})


def generate_samples(data_dir: Path):
    """Yield one sample at a time without loading images into memory."""
    questions_file = data_dir / "questions.jsonl"

    if not questions_file.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_file}")

    with open(questions_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            paths = record["images"]
            meta = record["meta"]

            image_1_path = data_dir / paths[0]
            image_2_path = data_dir / paths[1] if len(paths) > 1 else None

            if not image_1_path.exists():
                raise FileNotFoundError(f"Missing image_1 for id={record['id']}: {image_1_path}")
            if image_2_path is not None and not image_2_path.exists():
                raise FileNotFoundError(f"Missing image_2 for id={record['id']}: {image_2_path}")

            yield {
                "id": record["id"],
                "image_1": str(image_1_path),
                "image_2": str(image_2_path) if image_2_path else None,
                "question": record["question"],
                "answer": record["answer"],
                "image_count": meta["image_count"],
                "modality": meta["modality"],
                "task": meta["task"],
                "domain": meta["domain"],
                "distortion_family": meta["distortion_family"],
                "distortion_type": meta["distortion_type"],
                "distortion_complexity": meta["distortion_complexity"],
                "comparison": meta["comparison"],
            }

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1} samples...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--repo-id", default="Zhongchenchen/SenseBench")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Building dataset...")
    ds = Dataset.from_generator(
        generate_samples,
        gen_kwargs={"data_dir": data_dir},
        features=FEATURES,
        split="test",
    )
    print(f"Dataset: {ds.num_rows} rows, {ds.num_columns} columns")
    print(f"Features: {ds.features}")

    print(f"Pushing to {args.repo_id}...")
    ds.push_to_hub(
        args.repo_id,
        private=args.private,
        split="test",
        embed_external_files=True,
    )
    print("Done!")


if __name__ == "__main__":
    main()
