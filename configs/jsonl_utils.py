import json
from pathlib import Path
from typing import Iterable


PERCEPTION_TOP1 = {"whether", "what", "how"}


def read_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield lineno, obj


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_pair_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "none":
        return "all"
    if text in {"intra", "intra-image", "intra_image", "intraimage"}:
        return "intra-image"
    if text in {
        "inter",
        "temporal",
        "inter-temporal",
        "inter_temporal",
        "intertemporal",
        "intra-temporal",
        "intra_temporal",
        "intratemporal",
        "intra-temple",
        "intra_temple",
        "intratemple",
    }:
        return "inter-temporal"
    return text


def normalize_question_record(obj: dict) -> dict[str, str]:
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    image_count = str(meta.get("image_count") or obj.get("bench") or "single").strip().lower()
    bench = "multi" if image_count in {"multi", "multiple", "pair", "paired"} else "single"

    top1 = str(meta.get("task") or obj.get("top1") or obj.get("task_type") or "").strip()
    family = str(meta.get("distortion_family") or obj.get("ability") or "").strip()
    task = str(meta.get("distortion_type") or obj.get("task") or "").strip()
    domain = str(meta.get("domain") or "").strip().lower()
    complexity = str(meta.get("distortion_complexity") or "").strip().lower()

    if family.lower() == "sar" or domain == "remote_sensing" and task in {"Speckle", "Sidelobe"}:
        ability = "sar"
    elif family == "multi_distortion" or task == "multi_distortion" or complexity == "multi":
        ability = "multi"
    else:
        ability = "optical"

    pair_type = normalize_pair_type(meta.get("comparison") or obj.get("pair_type") or obj.get("data_type"))
    return {
        "bench": bench,
        "top1": top1,
        "ability": ability,
        "task": task,
        "pair_type": pair_type,
    }


def get_record_images(obj: dict) -> list[str]:
    images = obj.get("images")
    if isinstance(images, list):
        return [str(x) for x in images if x]
    if isinstance(images, str) and images:
        return [images]
    if obj.get("image_path"):
        return [str(obj["image_path"])]

    out = []
    idx = 1
    while obj.get(f"image_{idx}"):
        out.append(str(obj[f"image_{idx}"]))
        idx += 1
    return out
