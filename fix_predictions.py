"""Fix predictions.jsonl field names to match eval.py expectations.

Transforms:
  prediction -> response
  answer -> label
  Adds query (question text) from questions.jsonl
  Adds top1/ability/bench/pair_type for evaluate_model_jsonl
"""

import json
from pathlib import Path

QUESTIONS = Path("data/questions.jsonl")
OUTPUT_ROOT = Path("outputs/inference")


def derive_ability(distortion_family, distortion_type, domain):
    if distortion_family == "sar" or (domain == "remote_sensing" and distortion_type in {"Speckle", "Sidelobe"}):
        return "sar"
    if distortion_family == "multi_distortion" or distortion_type == "multi_distortion":
        return "multi"
    return "optical"


def needs_fix(r):
    """Check if a record uses the old format."""
    return "prediction" in r and "response" not in r


def fix_one(pred_path, questions):
    rows = []
    fixed = 0
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            if not needs_fix(r):
                rows.append(r)
                continue

            rid = r["id"]
            task = r.get("task", "")
            family = r.get("distortion_family", "")
            dtype = r.get("distortion_type", "")
            image_count = r.get("image_count", "single")
            domain = r.get("domain", "")
            comparison = r.get("comparison")

            rows.append({
                "id": rid,
                "response": r["prediction"],
                "label": r["answer"],
                "query": questions.get(rid, ""),
                "top1": task,
                "task": dtype,
                "ability": derive_ability(family, dtype, domain),
                "bench": "multi" if image_count in ("multi", "multiple") else "single",
                "pair_type": comparison if comparison else "all",
            })
            fixed += 1

    with open(pred_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return fixed


def main():
    questions = {}
    with open(QUESTIONS) as f:
        for line in f:
            q = json.loads(line)
            questions[q["id"]] = q["question"]

    for model_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        pred_path = model_dir / "predictions.jsonl"
        if not pred_path.exists():
            continue
        n = fix_one(pred_path, questions)
        if n > 0:
            print(f"Fixed {n} entries: {pred_path}")
        else:
            print(f"Already correct: {pred_path}")


if __name__ == "__main__":
    main()
