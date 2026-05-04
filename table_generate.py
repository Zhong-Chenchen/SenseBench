import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from configs.jsonl_utils import normalize_pair_type as normalize_jsonl_pair_type
from configs.jsonl_utils import normalize_question_record, read_jsonl


PERCEPTION_TOP1 = {"whether", "what", "how"}
DESCRIPTION_METRICS = ("completeness", "correctness", "faithfulness")
DESCRIPTION_METRIC_TITLES = {
    "completeness": "Completeness",
    "correctness": "Correctness",
    "faithfulness": "Faithfulness",
}
KNOWN_ABILITIES = {"blur", "cloud", "compression", "correction", "missing", "noise", "multi", "sar"}
PAIR_TYPES = {"intra-image", "inter-temporal"}


def _safe_to_excel(df: pd.DataFrame, path: Path, *, index: bool = False) -> bool:
    try:
        df.to_excel(path, index=index)
        return True
    except ModuleNotFoundError as exc:
        print(f"[WARN] Failed to write Excel file {path}: {exc}. Install with `pip install openpyxl`.")
        return False
    except Exception as exc:
        print(f"[WARN] Failed to write Excel file {path}: {exc}")
        return False


def _normalize_model_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _load_model_order_file(path: Path | None) -> list[str]:
    if not path or not Path(path).is_file():
        return []
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _is_model_dir(path: Path) -> bool:
    """Heuristic: a model output dir contains a single/ or multi/ subdir."""

    if not path.is_dir():
        return False
    if (path / "evaluated.jsonl").is_file() or (path / "predictions.jsonl").is_file():
        return True
    if (path / "single").is_dir() or (path / "multi").is_dir():
        return True
    # Legacy layout: top1 directories directly under model dir.
    for top1 in ("whether", "what", "how", "description"):
        if (path / top1).is_dir():
            return True
    return False


def _load_rules(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Rules file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise SystemExit(f"Failed to parse rules json: {path} ({exc})")
    if not isinstance(data, dict):
        raise SystemExit(f"Rules json must be an object: {path}")
    return data


def _resolve_model_row_order(*, desired: list[str], available: list[str]) -> list[str]:
    if not available:
        return []
    if not desired:
        return list(available)

    available_set = set(available)
    norm_to_available: dict[str, list[str]] = {}
    for a in available:
        norm_to_available.setdefault(_normalize_model_name(a), []).append(a)

    ordered: list[str] = []
    used: set[str] = set()
    for d in desired:
        if d in available_set and d not in used:
            ordered.append(d)
            used.add(d)
            continue
        candidates = norm_to_available.get(_normalize_model_name(d), [])
        for c in candidates:
            if c not in used:
                ordered.append(c)
                used.add(c)
                break

    for a in available:
        if a not in used:
            ordered.append(a)
    return ordered


@dataclass(frozen=True)
class TaskRecord:
    model: str
    bench: str  # single|multi
    pair_type: str  # all|intra-image|inter-temporal
    top1: str   # whether|what|how|description
    ability: str
    task: str
    accuracy: float


def _iter_task_files(model_dir: Path):
    """Yield (bench, top1, ability, task, path) for task json files.

    Supports:
      - <model>/<single|multi>/<top1>/<ability>/<task>.json
      - <model>/<top1>/<ability>/<task>.json  (legacy => bench='single')

    Ignores split files like <task>__intra-image.json.
    """

    # current layout
    for bench in ("single", "multi"):
        root = model_dir / bench
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if "__" in path.stem:
                continue
            rel = path.relative_to(root)
            parts = rel.parts
            if len(parts) != 3:
                continue
            top1, ability, filename = parts
            task = Path(filename).stem
            if top1 not in {"whether", "what", "how", "description"}:
                continue
            yield bench, top1, ability, task, path

    # legacy layout
    for path in model_dir.rglob("*.json"):
        if "__" in path.stem:
            continue
        try:
            rel = path.relative_to(model_dir)
        except Exception:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        top1, ability, filename = parts
        if top1 not in {"whether", "what", "how", "description"}:
            continue
        task = Path(filename).stem
        yield "single", top1, ability, task, path


def _load_task_accuracy(path: Path) -> float | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, list) or not data:
        return None

    total = 0
    correct = 0
    for obj in data:
        if not isinstance(obj, dict):
            continue
        if "result" not in obj:
            continue
        val = obj.get("result")
        if isinstance(val, bool):
            total += 1
            if val:
                correct += 1
    if total == 0:
        return None
    return correct / total


def _normalize_pair_type(value: object) -> str:
    dt = str(value or "").strip().lower()
    if not dt:
        return "unknown"
    if dt in {"intra-image", "intra_image", "intraimage"}:
        return "intra-image"
    # user sometimes says intra-temporal, bench uses inter-temporal.
    if dt in {
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
    return dt


def _load_task_accuracy_split(path: Path, *, pair_type: str) -> float | None:
    """Compute accuracy for a task json restricted to a specific pair_type."""

    target = _normalize_pair_type(pair_type)
    if target not in PAIR_TYPES:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None

    total = 0
    correct = 0
    for obj in data:
        if not isinstance(obj, dict):
            continue
        if "result" not in obj:
            continue
        if _normalize_pair_type(obj.get("data_type")) != target:
            continue
        val = obj.get("result")
        if isinstance(val, bool):
            total += 1
            if val:
                correct += 1
    if total == 0:
        return None
    return correct / total


def _collect_records(output_root: Path, *, models: list[str] | None = None) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    model_filter = set(models) if models else None

    for item in sorted(output_root.iterdir()):
        if not _is_model_dir(item):
            continue
        if model_filter is not None and item.name not in model_filter:
            continue

        jsonl_records = _collect_records_from_jsonl_model(item)
        if jsonl_records:
            records.extend(jsonl_records)
            continue

        for bench, top1, ability, task, path in _iter_task_files(item):
            acc = _load_task_accuracy(path)
            if acc is None:
                continue
            records.append(
                TaskRecord(
                    model=item.name,
                    bench=bench,
                    pair_type="all",
                    top1=top1,
                    ability=ability,
                    task=task,
                    accuracy=float(acc),
                )
            )

    return records


def _collect_records_from_jsonl_model(model_dir: Path) -> list[TaskRecord]:
    path = model_dir / "evaluated.jsonl"
    if not path.is_file():
        return []

    stats: dict[tuple[str, str, str, str, str], dict[str, int]] = {}

    def update(*, bench: str, pair_type: str, top1: str, ability: str, task: str, result: bool):
        key = (bench, pair_type, top1, ability, task)
        stat = stats.setdefault(key, {"correct": 0, "total": 0})
        stat["total"] += 1
        if result:
            stat["correct"] += 1

    try:
        iterator = read_jsonl(path)
        for _, obj in iterator:
            if not isinstance(obj, dict) or not isinstance(obj.get("result"), bool):
                continue
            norm = normalize_question_record(obj)
            bench = str(obj.get("bench") or norm["bench"])
            top1 = str(obj.get("top1") or norm["top1"])
            ability = str(obj.get("ability") or norm["ability"])
            task = str(obj.get("task") or norm["task"])
            pair_type = normalize_jsonl_pair_type(obj.get("pair_type") or obj.get("data_type") or norm["pair_type"])
            result = bool(obj["result"])
            update(bench=bench, pair_type="all", top1=top1, ability=ability, task=task, result=result)
            if bench == "multi" and top1 in PERCEPTION_TOP1 and pair_type in {"intra-image", "inter-temporal"}:
                update(bench=bench, pair_type=pair_type, top1=top1, ability=ability, task=task, result=result)
    except Exception:
        return []

    out: list[TaskRecord] = []
    for (bench, pair_type, top1, ability, task), stat in sorted(stats.items()):
        total = stat["total"]
        if not total:
            continue
        out.append(
            TaskRecord(
                model=model_dir.name,
                bench=bench,
                pair_type=pair_type,
                top1=top1,
                ability=ability,
                task=task,
                accuracy=stat["correct"] / total,
            )
        )
    return out


def _as_score_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        out: list[int] = []
        for x in value:
            if isinstance(x, bool):
                continue
            if isinstance(x, int):
                out.append(x)
        return out
    return []


def _collect_description_counts(
    output_root: Path,
    *,
    bench: str,
    models: list[str] | None = None,
) -> dict[str, dict[str, list[int]]]:
    """Return per-model metric counts: metric -> [c0,c1,c2]."""

    out: dict[str, dict[str, list[int]]] = {}
    model_filter = set(models) if models else None

    for item in sorted(output_root.iterdir()):
        if not _is_model_dir(item):
            continue
        if model_filter is not None and item.name not in model_filter:
            continue

        counts: dict[str, list[int]] = {m: [0, 0, 0] for m in DESCRIPTION_METRICS}

        jsonl_path = item / "evaluated.jsonl"
        if jsonl_path.is_file():
            try:
                for _, obj in read_jsonl(jsonl_path):
                    if not isinstance(obj, dict):
                        continue
                    norm = normalize_question_record(obj)
                    obj_bench = str(obj.get("bench") or norm["bench"])
                    obj_top1 = str(obj.get("top1") or norm["top1"])
                    if obj_bench != bench or obj_top1 != "description":
                        continue
                    ds = obj.get("description_scores")
                    if not isinstance(ds, dict):
                        continue
                    for metric in DESCRIPTION_METRICS:
                        vals = _as_score_list(ds.get(metric))
                        for v in vals:
                            if v in (0, 1, 2):
                                counts[metric][v] += 1
            except Exception:
                pass
            out[item.name] = counts
            continue

        for b, top1, ability, task, path in _iter_task_files(item):
            if b != bench or top1 != "description":
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, list) or not data:
                continue
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                ds = obj.get("description_scores")
                if not isinstance(ds, dict):
                    continue
                for metric in DESCRIPTION_METRICS:
                    vals = _as_score_list(ds.get(metric))
                    for v in vals:
                        if v in (0, 1, 2):
                            counts[metric][v] += 1

        out[item.name] = counts

    return out


def _pct_str(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.2f}%"


def _build_description_table(
    *,
    output_root: Path,
    bench: str,
    model_order: list[str],
    rules: dict,
    models: list[str] | None,
) -> pd.DataFrame:
    section = rules.get(f"description_{bench}", {})
    if not isinstance(section, dict):
        raise SystemExit(f"Rules error: description_{bench} must be an object")
    exclude_models = section.get("exclude_models", [])
    if not isinstance(exclude_models, list):
        raise SystemExit(f"Rules error: description_{bench}.exclude_models must be a list")
    exclude_set = set(str(x) for x in exclude_models)

    counts_by_model = _collect_description_counts(output_root, bench=bench, models=models)

    def build_metric_cells(metric: str, counts: list[int]):
        c0, c1, c2 = (counts + [0, 0, 0])[:3]
        total = c0 + c1 + c2
        if total <= 0:
            return None, None, None, None
        p0 = 100.0 * c0 / total
        p1 = 100.0 * c1 / total
        p2 = 100.0 * c2 / total
        score = (1.0 * c1 + 2.0 * c2) / total
        return _pct_str(p0), _pct_str(p1), _pct_str(p2), round(score, 2)

    # MultiIndex columns for a paper-like look.
    # Note: writing MultiIndex columns requires index=True in some pandas versions.
    col_tuples: list[tuple[str, str]] = []
    for metric in DESCRIPTION_METRICS:
        title = DESCRIPTION_METRIC_TITLES.get(metric, metric)
        col_tuples.extend(
            [
                (title, "P0"),
                (title, "P1"),
                (title, "P2"),
                (title, "score↑"),
            ]
        )
    col_tuples.append(("Sum.↑", ""))
    columns = pd.MultiIndex.from_tuples(col_tuples)

    index_models: list[str] = []
    rows: list[dict[tuple[str, str], object]] = []
    for model in model_order:
        if model in exclude_set:
            continue
        model_counts = counts_by_model.get(model)
        index_models.append(model)
        row: dict[tuple[str, str], object] = {}

        scores_for_sum: list[float] = []
        for metric in DESCRIPTION_METRICS:
            title = DESCRIPTION_METRIC_TITLES.get(metric, metric)
            p0, p1, p2, score = build_metric_cells(metric, (model_counts or {}).get(metric, [0, 0, 0]))
            row[(title, "P0")] = p0
            row[(title, "P1")] = p1
            row[(title, "P2")] = p2
            row[(title, "score↑")] = score
            if isinstance(score, (int, float)):
                scores_for_sum.append(float(score))

        row[("Sum.↑", "")] = round(sum(scores_for_sum), 2) if len(scores_for_sum) == len(DESCRIPTION_METRICS) else None
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.index = index_models
    df.index.name = "Model (variant)"
    return df


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _row_mean(row: dict[str, object], keys: list[str]) -> float | None:
    values: list[float] = []
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)):
            values.append(float(v))
    return _mean(values)


def _macro_mean(records: list[TaskRecord]) -> float | None:
    return _mean([r.accuracy for r in records])


def _is_in_bucket(r: TaskRecord, items: set[str], *, general_items: set[str] | None = None) -> bool:
    """Interpret a simple user list as a bucket definition.

    - If an item equals an ability name, it includes all tasks under that ability.
    - Otherwise an item is treated as a task name.
    - Special handling for cloud:
        If 'cloud' is in rs_centric items, it means include cloud tasks EXCEPT those
        explicitly listed in general (e.g., cloud_haze).
    """

    if not items:
        return False

    if r.ability in items:
        # cloud exception: only apply when we are using rs_centric items
        if r.ability == "cloud" and general_items is not None:
            if r.task in general_items:
                return False
        return True

    if r.task in items:
        return True

    return False


def _build_single_perception_table(*, records: list[TaskRecord], model_order: list[str], rules: dict) -> pd.DataFrame:
    sp = rules.get("single_perception", {})
    if not isinstance(sp, dict):
        raise SystemExit("Rules error: single_perception must be an object")

    general_items = sp.get("general", [])
    rs_items = sp.get("rs_centric", [])
    single_items = sp.get("single_tasks", [])
    multi_items = sp.get("multi_tasks", [])
    if not isinstance(general_items, list) or not isinstance(rs_items, list) or not isinstance(single_items, list) or not isinstance(multi_items, list):
        raise SystemExit("Rules error: single_perception.general/rs_centric/single_tasks/multi_tasks must be lists")
    general_set = set(str(x) for x in general_items)
    rs_set = set(str(x) for x in rs_items)
    single_set = set(str(x) for x in single_items)
    multi_set = set(str(x) for x in multi_items)

    rows: list[dict[str, object]] = []
    by_model: dict[str, list[TaskRecord]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)

    # SAR is never included in single/multi aggregate columns.
    sar_tasks = {"Speckle", "Sidelobe"}

    def fmt(x: float | None):
        return round(float(x), 4) if x is not None else None

    sum_keys = ["whether", "what", "how", "general", "RS-centric", "single", "multi"]

    for model in model_order:
        model_recs = by_model.get(model, [])
        single_perc = [r for r in model_recs if r.bench == "single" and r.pair_type == "all" and r.top1 in PERCEPTION_TOP1]

        row: dict[str, object] = {"model": model}
        row["whether"] = fmt(_macro_mean([r for r in model_recs if r.bench == "single" and r.pair_type == "all" and r.top1 == "whether"]))
        row["what"] = fmt(_macro_mean([r for r in model_recs if r.bench == "single" and r.pair_type == "all" and r.top1 == "what"]))
        row["how"] = fmt(_macro_mean([r for r in model_recs if r.bench == "single" and r.pair_type == "all" and r.top1 == "how"]))

        row["general"] = fmt(_macro_mean([r for r in single_perc if _is_in_bucket(r, general_set)]))
        row["RS-centric"] = fmt(_macro_mean([r for r in single_perc if _is_in_bucket(r, rs_set, general_items=general_set)]))

        # Column "single": mean over the explicit single-distortion task list (task-level).
        row["single"] = fmt(_macro_mean([r for r in single_perc if r.task in single_set and not (r.ability == "sar" or r.task in sar_tasks)]))

        # Column "multi": mean over the explicit multi-distortion task list (task-level).
        # This corresponds to the single-bench multi-distortion tasks (e.g., multi_distortion).
        row["multi"] = fmt(_macro_mean([r for r in single_perc if r.task in multi_set and not (r.ability == "sar" or r.task in sar_tasks)]))

        # Column "sum": mean over whether/what/how columns (ignoring missing).
        row["sum"] = fmt(_row_mean(row, ["whether", "what", "how"]))

        rows.append(row)

    return pd.DataFrame(rows)


def _build_multi_perception_table(
    *,
    records: list[TaskRecord],
    model_order: list[str],
    rules: dict,
    output_root: Path,
) -> pd.DataFrame:
    mp = rules.get("multi_perception", {})
    if not isinstance(mp, dict):
        raise SystemExit("Rules error: multi_perception must be an object")

    general_items = mp.get("general", [])
    rs_items = mp.get("rs_centric", [])
    exclude_models = mp.get("exclude_models", [])
    if not isinstance(general_items, list) or not isinstance(rs_items, list):
        raise SystemExit("Rules error: multi_perception.general and rs_centric must be lists")
    if not isinstance(exclude_models, list):
        raise SystemExit("Rules error: multi_perception.exclude_models must be a list")
    general_set = set(str(x) for x in general_items)
    rs_set = set(str(x) for x in rs_items)
    exclude_set = set(str(x) for x in exclude_models)

    by_model: dict[str, list[TaskRecord]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)

    def fmt(x: float | None):
        return round(float(x), 4) if x is not None else None

    sum_keys = ["whether", "what", "how", "general", "RS-centric", "intra-image", "intra-temporal"]

    rows: list[dict[str, object]] = []
    for model in model_order:
        if model in exclude_set:
            continue
        model_recs = by_model.get(model, [])

        multi_perc_recs = [r for r in model_recs if r.bench == "multi" and r.top1 in PERCEPTION_TOP1]
        multi_perc_all = [r for r in multi_perc_recs if r.pair_type == "all"]

        row: dict[str, object] = {"model": model}
        row["whether"] = fmt(_macro_mean([r for r in model_recs if r.bench == "multi" and r.pair_type == "all" and r.top1 == "whether"]))
        row["what"] = fmt(_macro_mean([r for r in model_recs if r.bench == "multi" and r.pair_type == "all" and r.top1 == "what"]))
        row["how"] = fmt(_macro_mean([r for r in model_recs if r.bench == "multi" and r.pair_type == "all" and r.top1 == "how"]))

        row["general"] = fmt(_macro_mean([r for r in multi_perc_all if _is_in_bucket(r, general_set)]))
        row["RS-centric"] = fmt(_macro_mean([r for r in multi_perc_all if _is_in_bucket(r, rs_set)]))

        # intra-image / inter-temporal: compute macro mean over multi-bench perception tasks
        model_dir = output_root / model
        intra_scores: list[float] = [float(r.accuracy) for r in multi_perc_recs if r.pair_type == "intra-image"]
        inter_scores: list[float] = [float(r.accuracy) for r in multi_perc_recs if r.pair_type == "inter-temporal"]

        if not intra_scores and not inter_scores and _is_model_dir(model_dir) and (model_dir / "multi").is_dir():
            for bench, top1, ability, task, path in _iter_task_files(model_dir):
                if bench != "multi" or top1 not in PERCEPTION_TOP1:
                    continue
                # Skip split files (already filtered by _iter_task_files).
                a_intra = _load_task_accuracy_split(path, pair_type="intra-image")
                if a_intra is not None:
                    intra_scores.append(float(a_intra))
                a_inter = _load_task_accuracy_split(path, pair_type="inter-temporal")
                if a_inter is not None:
                    inter_scores.append(float(a_inter))

        row["intra-image"] = fmt(_mean(intra_scores))
        # Column name requested as intra-temporal; data value comes from inter-temporal in json.
        row["intra-temporal"] = fmt(_mean(inter_scores))

        # Column "sum": mean over whether/what/how columns (ignoring missing).
        row["sum"] = fmt(_row_mean(row, ["whether", "what", "how"]))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate summary tables from evaluated json outputs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("SENSEBENCH_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs" / "inference")),
        help="Directory containing model outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default="outputs/inference/tables",
        help="Directory to write tables (default: --output-root).",
    )
    parser.add_argument(
        "--model-order-file",
        type=Path,
        default=None,
        help="Override model order file. If omitted, uses rules file 'model_order_file'.",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "table_rules.json",
        help="Rules json file defining column composition.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of models.",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=["single_perception", "multi_perception", "description_single", "description_multi"],
        help="Tables to generate. Supported: single_perception, multi_perception, description_single, description_multi",
    )
    args = parser.parse_args()

    output_root: Path = args.output_root
    if not output_root.exists():
        raise SystemExit(f"output directory not found: {output_root}")

    out_dir: Path = args.out_dir if args.out_dir is not None else output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = _load_rules(args.rules)
    rules_model_order_file = rules.get("model_order_file")
    default_order_path = Path(rules_model_order_file) if isinstance(rules_model_order_file, str) and rules_model_order_file else Path("model_order.txt")
    # Resolve relative paths against the rules file location.
    if not default_order_path.is_absolute():
        default_order_path = args.rules.parent / default_order_path

    available_models = sorted([p.name for p in output_root.iterdir() if _is_model_dir(p)])
    order_path = args.model_order_file or default_order_path
    if order_path is not None and not Path(order_path).is_absolute():
        order_path = args.rules.parent / Path(order_path)
    desired = _load_model_order_file(order_path)
    if not desired:
        print(f"[WARN] Model order file not found or empty: {order_path}. Falling back to directory order.")
    model_order = _resolve_model_row_order(desired=desired, available=available_models)
    if args.models:
        keep = set(args.models)
        model_order = [m for m in model_order if m in keep]

    # Diagnostics: help user keep model_order.txt in sync.
    if desired:
        desired_norm = {_normalize_model_name(x) for x in desired}
        available_norm = {_normalize_model_name(x) for x in available_models}
        missing = [x for x in desired if _normalize_model_name(x) not in available_norm]
        if missing:
            print("[WARN] These names in model_order_file did not match any output directory:")
            for x in missing:
                print(f"  - {x}")
        not_listed = [x for x in available_models if _normalize_model_name(x) not in desired_norm]
        if not_listed:
            print("[WARN] These output directories were not listed in model_order_file (they will be appended at the end):")
            for x in not_listed:
                print(f"  - {x}")

    tables = [t.strip() for t in (args.tables or []) if t.strip()]
    need_acc_records = any(t in {"single_perception", "multi_perception"} for t in tables)
    records: list[TaskRecord] = []
    if need_acc_records:
        records = _collect_records(output_root, models=args.models)
        if not records:
            raise SystemExit("No task records found. Ensure evaluated json/jsonl files contain boolean `result` fields (run eval.py first).")
    for tname in tables:
        if tname == "single_perception":
            df = _build_single_perception_table(records=records, model_order=model_order, rules=rules)
            xlsx_path = out_dir / "single_perception.xlsx"
        elif tname == "multi_perception":
            df = _build_multi_perception_table(records=records, model_order=model_order, rules=rules, output_root=output_root)
            xlsx_path = out_dir / "multi_perception.xlsx"
        elif tname == "description_single":
            df = _build_description_table(
                output_root=output_root,
                bench="single",
                model_order=model_order,
                rules=rules,
                models=args.models,
            )
            xlsx_path = out_dir / "description_single.xlsx"
        elif tname == "description_multi":
            df = _build_description_table(
                output_root=output_root,
                bench="multi",
                model_order=model_order,
                rules=rules,
                models=args.models,
            )
            xlsx_path = out_dir / "description_multi.xlsx"
        else:
            raise SystemExit(
                f"Unknown table: {tname} (supported: single_perception, multi_perception, description_single, description_multi)"
            )

        wrote = _safe_to_excel(df, xlsx_path, index=tname in {"description_single", "description_multi"})
        if not wrote:
            raise SystemExit("Failed to write .xlsx output (install openpyxl or check permissions).")
        print(f"Saved table to {xlsx_path}")



if __name__ == "__main__":
    main()
