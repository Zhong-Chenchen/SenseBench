"""Convert SenseBench data to Parquet format and upload to Hugging Face."""

import json
import argparse
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi


IMAGE_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("image_1", IMAGE_STRUCT),
    pa.field("image_2", IMAGE_STRUCT, nullable=True),
    pa.field("question", pa.string()),
    pa.field("answer", pa.string()),
    pa.field("image_count", pa.string()),
    pa.field("modality", pa.string()),
    pa.field("task", pa.string()),
    pa.field("domain", pa.string()),
    pa.field("distortion_family", pa.string()),
    pa.field("distortion_type", pa.string()),
    pa.field("distortion_complexity", pa.string()),
    pa.field("comparison", pa.string(), nullable=True),
])


def build_table(data_dir: Path):
    """Build a PyArrow table from questions.jsonl, embedding images as bytes."""
    ids, image_1_list, image_2_list = [], [], []
    questions, answers = [], []
    image_counts, modalities, tasks, domains = [], [], [], []
    distortion_families, distortion_types = [], []
    distortion_complexities, comparisons = [], []

    questions_file = data_dir / "questions.jsonl"
    with open(questions_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            paths = record["images"]
            meta = record["meta"]

            img1_path = data_dir / paths[0]
            img2_path = data_dir / paths[1] if len(paths) > 1 else None

            if not img1_path.exists():
                raise FileNotFoundError(f"Missing image: {img1_path}")
            if img2_path and not img2_path.exists():
                raise FileNotFoundError(f"Missing image: {img2_path}")

            ids.append(record["id"])
            image_1_list.append({"bytes": img1_path.read_bytes(), "path": img1_path.name})
            image_2_list.append({"bytes": img2_path.read_bytes(), "path": img2_path.name} if img2_path else None)
            questions.append(record["question"])
            answers.append(record["answer"])
            image_counts.append(meta["image_count"])
            modalities.append(meta["modality"])
            tasks.append(meta["task"])
            domains.append(meta["domain"])
            distortion_families.append(meta["distortion_family"])
            distortion_types.append(meta["distortion_type"])
            distortion_complexities.append(meta["distortion_complexity"])
            comparisons.append(meta["comparison"])

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1} samples...")

    print(f"Total samples: {len(ids)}")

    table = pa.table({
        "id": ids,
        "image_1": image_1_list,
        "image_2": image_2_list,
        "question": questions,
        "answer": answers,
        "image_count": image_counts,
        "modality": modalities,
        "task": tasks,
        "domain": domains,
        "distortion_family": distortion_families,
        "distortion_type": distortion_types,
        "distortion_complexity": distortion_complexities,
        "comparison": comparisons,
    }, schema=SCHEMA)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--repo-id", default="Zhongchenchen/SenseBench")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Building table...")
    table = build_table(data_dir)

    # Save to temp directory as parquet shards
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        num_shards = 10
        rows_per_shard = (len(table) + num_shards - 1) // num_shards
        parquet_paths = []
        for idx in range(num_shards):
            start = idx * rows_per_shard
            end = min(start + rows_per_shard, len(table))
            if start >= len(table):
                break
            shard = table.slice(start, end - start)
            shard_path = tmp_dir / f"data-{idx:05d}-of-{num_shards:05d}.parquet"
            pq.write_table(shard, shard_path, row_group_size=100)
            parquet_paths.append(shard_path)
            print(f"  Wrote shard {idx}: {shard_path.name} ({end - start} rows)")

        # Copy README if exists
        readme_src = data_dir.parent / "README.md"
        if readme_src.exists():
            shutil.copy(readme_src, tmp_dir / "README.md")

        print(f"Uploading to {args.repo_id}...")
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(tmp_dir),
            path_in_repo=".",
        )
        print("Done!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
