"""Convert SenseBench data to Parquet format and upload to HuggingFace."""

import json
import argparse
from pathlib import Path
from datasets import Dataset, Features, Value, Image


def generate_samples(data_dir: Path):
    """Yield one sample at a time to avoid loading all images into memory."""
    from PIL import Image as PILImage

    questions_file = data_dir / "questions.jsonl"
    with open(questions_file) as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            paths = record["images"]

            img1 = PILImage.open(data_dir / paths[0]).convert("RGB")
            img2 = PILImage.open(data_dir / paths[1]).convert("RGB") if len(paths) > 1 else None

            yield {
                "id": record["id"],
                "image_1": img1,
                "image_2": img2,
                "question": record["question"],
                "answer": record["answer"],
                "image_count": record["meta"]["image_count"],
                "modality": record["meta"]["modality"],
                "task": record["meta"]["task"],
                "domain": record["meta"]["domain"],
                "distortion_family": record["meta"]["distortion_family"],
                "distortion_type": record["meta"]["distortion_type"],
                "distortion_complexity": record["meta"]["distortion_complexity"],
                "comparison": record["meta"]["comparison"],
            }

            # Close to free memory
            img1.close()
            if img2:
                img2.close()

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1} samples...")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--repo-id", default="Zhongchenchen/SenseBench")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--save-local", default=None, help="Save parquet to local dir first, skip upload")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.save_local:
        print(f"Building dataset to local dir: {args.save_local}")
        ds = Dataset.from_generator(
            generate_samples,
            gen_kwargs={"data_dir": data_dir},
            features=FEATURES,
            split="test",
        )
        print(f"Dataset: {ds.num_rows} rows")
        ds.save_to_disk(args.save_local)
        print(f"Saved to {args.save_local}")
    else:
        print("Building and uploading dataset...")
        ds = Dataset.from_generator(
            generate_samples,
            gen_kwargs={"data_dir": data_dir},
            features=FEATURES,
            split="test",
        )
        print(f"Dataset: {ds.num_rows} rows")
        print(f"Pushing to {args.repo_id}...")
        ds.push_to_hub(args.repo_id, private=args.private)
        print("Done!")


if __name__ == "__main__":
    main()
