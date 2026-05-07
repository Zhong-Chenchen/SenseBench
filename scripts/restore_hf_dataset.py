"""Restore SenseBench Hugging Face parquet shards back into PNG folders."""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def _load_question_map(path: Path) -> dict[str, dict[str, str]]:
    qmap: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            sample_id = str(obj.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"Missing id in manifest jsonl at line {lineno}")
            qmap[sample_id] = {
                "image_1": str(obj.get("image_1", "")).strip(),
                "image_2": str(obj.get("image_2", "")).strip(),
            }
    return qmap


def _iter_parquet_rows(parquet_root: Path):
    parquet_files = sorted(parquet_root.glob("data/*.parquet"))
    if not parquet_files:
        parquet_files = sorted(parquet_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found under: {parquet_root}")

    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        for row in table.to_pylist():
            yield parquet_file, row


def _resolve_manifest_path(input_dir: Path, manifest_file: Path | None) -> Path:
    if manifest_file is not None:
        manifest_file = manifest_file if manifest_file.is_absolute() else (input_dir / manifest_file)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Missing manifest jsonl: {manifest_file}")
        return manifest_file

    for candidate in (
        input_dir / "restore_manifest.jsonl",
        input_dir / "manifest.jsonl",
        input_dir / "questions_restore.jsonl",
        input_dir / "questions.jsonl",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No manifest jsonl found under: {input_dir}")


def restore_dataset(*, input_dir: Path, output_dir: Path, manifest_file: Path | None = None, eval_questions_file: Path | None = None, overwrite: bool = False) -> None:
    manifest_path = _resolve_manifest_path(input_dir, manifest_file)

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == input_dir:
        output_dir = input_dir / "original_png"
        print(f"[WARN] output-dir was the same as input-dir; using {output_dir} instead.")

    qmap = _load_question_map(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for parquet_file, row in _iter_parquet_rows(input_dir):
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        if sample_id not in qmap:
            raise KeyError(f"Sample id not found in manifest jsonl: {sample_id} (from {parquet_file.name})")

        image_meta = qmap[sample_id]
        for key in ("image_1", "image_2"):
            rel_path = image_meta.get(key, "")
            if not rel_path:
                continue
            image_struct = row.get(key)
            if not image_struct:
                continue
            payload = image_struct.get("bytes")
            if payload is None:
                continue
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                continue
            out_path.write_bytes(payload)
        written += 1
        if written % 1000 == 0:
            print(f"Restored {written} samples...")

    if eval_questions_file is not None:
        eval_questions_file = eval_questions_file if eval_questions_file.is_absolute() else (input_dir / eval_questions_file)
        if not eval_questions_file.is_file():
            raise FileNotFoundError(f"Missing eval questions jsonl: {eval_questions_file}")
        restored_questions = output_dir / "questions.jsonl"
        if eval_questions_file.resolve() != restored_questions.resolve():
            shutil.copy2(eval_questions_file, restored_questions)
    readme_src = input_dir / "README.md"
    if readme_src.is_file():
        restored_readme = output_dir / "README.md"
        if readme_src.resolve() != restored_readme.resolve():
            shutil.copy2(readme_src, restored_readme)

    print(f"Done. Restored {written} samples into {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Restore SenseBench HF parquet shards to PNG folders.")
    parser.add_argument("--input-dir", type=Path, default=Path("data_huggingface"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_huggingface/original_png"))
    parser.add_argument("--manifest-file", type=Path, default=None, help="Manifest jsonl used to map parquet rows back to image paths.")
    parser.add_argument("--eval-questions-file", type=Path, default=None, help="Optional evaluation questions.jsonl to copy into the restored output folder.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    restore_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_file=args.manifest_file,
        eval_questions_file=args.eval_questions_file,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
