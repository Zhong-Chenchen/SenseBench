import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
from tqdm import tqdm

from configs.infer_config import RSBENCH_TASKS, RSBENCH_TASKS_MULTI
from configs.jsonl_utils import normalize_pair_type as normalize_jsonl_pair_type
from configs.jsonl_utils import normalize_question_record, read_jsonl, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent


class AccEvaluator:
    OPTION_SET = {"A", "B", "C", "D", "E"}

    def __init__(
        self,
        *,
        use_llm: bool = False,
        llm_api_key: str = "sk-123456",
        llm_base_url: str = "http://0.0.0.0:23333/v1",
        llm_timeout: float = 10.0,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 8,
    ) -> None:
        self.use_llm = use_llm
        self._llm_model = None
        self._llm_client = None
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._llm_timeout = llm_timeout
        self._llm_temperature = llm_temperature
        self._llm_max_tokens = max(1, int(llm_max_tokens))

    def extract_full_option(self, query, option_letter):
        option_letter = (option_letter or "").strip().upper()
        if option_letter not in self.OPTION_SET:
            return option_letter
        option_marker = f"\n{option_letter}."
        start_index = query.find(option_marker)
        if start_index == -1:
            return option_letter
        start_index += len(option_marker)
        remaining = query[start_index:]
        next_markers = [f"\n{letter}." for letter in self.OPTION_SET if letter != option_letter]
        end_positions = [remaining.find(marker) for marker in next_markers if remaining.find(marker) != -1]
        end_index = min(end_positions) if end_positions else len(remaining)
        option_content = remaining[:end_index].strip()
        return f"{option_letter}.{option_content}" if option_content else option_letter

    def _extract_option_set(self, response):
        if not response:
            return set()
        text = response.strip()
        tokens = [tok for tok in re.split(r"[^A-Z]", text.upper()) if tok in self.OPTION_SET]
        if tokens:
            return set(tokens)
        compact = text.upper().replace(" ", "")
        if compact and all(ch in self.OPTION_SET for ch in compact):
            return set(compact)
        return set()

    def general_evaluater(self, response, labels):
        labels = labels if isinstance(labels, list) else [labels]
        label_set = set()
        label_text = {}
        for item in labels:
            if not item:
                continue
            if "." in item:
                letter, text = item.split(".", 1)
                letter = letter.strip().upper()
                label_set.add(letter)
                label_text[letter] = text.strip().lower()
            else:
                letter = item.strip().upper()
                label_set.add(letter)
                label_text.setdefault(letter, "")
        if not label_set:
            return False

        pred_set = self._extract_option_set(response)
        if not pred_set:
            response_lower = (response or "").lower()
            for letter, text in label_text.items():
                if text and text in response_lower:
                    pred_set.add(letter)
        if not pred_set:
            return False
        return pred_set == label_set

    def llm_evaluator(self, query: str, response: str, label_text: str) -> bool:
        if not self.use_llm:
            return False
        if OpenAI is None:
            raise ValueError("No client found, please refer to the evaluation.md")
        if self._llm_client is None:
            try:
                self._llm_client = OpenAI(api_key=self._llm_api_key, base_url=self._llm_base_url)
                model_name = self._llm_client.models.list().data[0].id
                self._llm_model = model_name
            except Exception:
                raise ValueError("No client found, please refer to the evaluation.md")
        prompt_template = (
            "Question: {question}\n"
            "Ground Truth Answer: {ground_truth}\n"
            "Predicted Answer: {predicted}\n"
            'Does the predicted answer match the ground truth? Reply with exactly one word: "True" or "False".'
        )
        question_body = "\n".join((query or "").splitlines()[1:]) if query else ""
        filled_prompt = prompt_template.format(
            question=question_body,
            ground_truth=label_text or "",
            predicted=response or "",
        )
        try:
            completion = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": filled_prompt,
                            }
                        ],
                    }
                ],
                timeout=self._llm_timeout,
                temperature=self._llm_temperature,
                max_tokens=self._llm_max_tokens,
                stop=["\n"],
            )
            reply = (completion.choices[0].message.content or "").strip().lower()
        except Exception as exc:
            print(f"[WARN] LLM judge failed ({exc}); falling back to heuristic.")
            return False

        match = re.search(r"\b(true|false)\b", reply)
        if match:
            return match.group(1) == "true"

        # Lenient fallback: some judge models ignore formatting and output explanations.
        # Try to infer from common phrases.
        # Note: order matters (e.g., 'does not match' should map to False).
        negative_markers = [
            "does not match",
            "doesn't match",
            "do not match",
            "not match",
            "mismatch",
            "incorrect",
            "not correct",
            "fails to match",
        ]
        positive_markers = [
            "match",
            "matches",
            "correct",
            "is correct",
            "aligned",
        ]
        for phrase in negative_markers:
            if phrase in reply:
                return False
        for phrase in positive_markers:
            if phrase in reply:
                return True

        print(f"[WARN] LLM judge returned unexpected content (unable to parse): {reply[:300]!r}")
        return False


class DescriptionScorer:
    METRIC_BUILDERS = {
    # 1. Completeness: Focus on coverage of degradation factors
    "completeness": lambda response, label: (
        f"""
Evaluate whether the description "{response}" fully covers the low-level degradation factors mentioned in the reference description "{label}".

Focus specifically on degradation factors such as blur, noise, compression artifacts, exposure issues, color distortion, etc.
Rate the completeness as follows:
- Score 2: The description includes all or almost all degradation factors present in the reference.
- Score 1: The description misses some key degradation factors but captures others.
- Score 0: The description misses most or all of the key degradation factors mentioned in the reference.

Output ONLY a single integer digit (0, 1, or 2). Do NOT output any additional text, reasoning, or markdown.
""".strip()
    ),

    # 2. Correctness: Focus on factual alignment and lack of contradictions
    "correctness": lambda response, label: (
        f"""
Evaluate whether the description "{response}" is factually aligned with the reference description "{label}" and free from contradictions regarding visual quality.

The correctness metric strictly penalizes contradictory descriptions (e.g., saying "sharp" when the reference says "blur", or "clean" when the reference says "noisy").
Rate the correctness as follows:
- Score 2: The description is factually aligned with the reference and contains no contradictions.
- Score 1: The description contains minor contradictions or ambiguities regarding the visual attributes.
- Score 0: The description contains strong or direct contradictions to the reference (e.g., opposite attributes).

Output ONLY a single integer digit (0, 1, or 2). Do NOT output any additional text, reasoning, or markdown.
""".strip()
    ),

    # 3. Faithfulness: Focus on visual grounding and penalizing hallucinations/high-level semantics
#     "faithfulness": lambda response, label: (
#         f"""
# Evaluate whether the description "{response}" is faithful to the low-level visual quality, avoiding hallucinations and irrelevant high-level semantics.

# Reference description: "{label}"

# The faithfulness metric verifies that the generated claims are visually grounded.
# - It strictly penalizes "hallucinations" (mentioning artifacts or flaws not present in the reference).
# - It penalizes irrelevant high-level semantics (describing the image content/scene rather than its quality/degradation).

# Rate the faithfulness as follows:
# - Score 2: The description is strictly grounded in low-level visual quality, with no hallucinations or irrelevant high-level content descriptions.
# - Score 1: The description contains minor hallucinations or mixes in some irrelevant high-level semantics.
# - Score 0: The description is dominated by hallucinations or focuses entirely on high-level content (e.g., describing objects) instead of visual quality.

# Output ONLY a single integer digit (0, 1, or 2). Do NOT output any additional text, reasoning, or markdown.
# """.strip()
#     ),
# }
        "faithfulness": lambda response, label: (
        f"""
    Evaluate whether the description "{response}" is faithful to the low-level visual quality in the reference, avoiding hallucinations and irrelevant high-level semantics.

    Reference description: "{label}"

    The faithfulness metric verifies that the generated claims are visually grounded in the reference description.
    - It strictly penalizes "hallucinations" (mentioning artifacts, flaws, severity levels, or affected visual details not supported by the reference).
    - It penalizes irrelevant high-level semantics (describing the image content/scene rather than its quality/degradation).
    - If a description adds unsupported concrete visual-quality details, it should not receive Score 2, even if the added details are plausible.
    - If the primary distortion type is inconsistent with the reference, the score should be 0.

    Rate the faithfulness as follows:
    - Score 2: The description is strictly grounded in low-level visual quality, with no unsupported concrete claims, no severity mismatch, and no irrelevant high-level content descriptions.
    - Score 1: The description is generally grounded but contains minor hallucinations, slight severity mismatch, vague unsupported quality claims, or mixes in some irrelevant high-level semantics.
    - Score 0: The description is dominated by hallucinations, contradicts the reference, identifies the wrong primary distortion, or focuses entirely on high-level content (e.g., describing objects) instead of visual quality.

    Output ONLY a single integer digit (0, 1, or 2). Do NOT output any additional text, reasoning, or markdown.
    """.strip()
    ),}

    def __init__(
        self,
        *,
        repeat: int = 5,
        llm_api_key: str = "sk-123456",
        llm_base_url: str = "http://0.0.0.0:23333/v1",
        llm_timeout: float = 10.0,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 8,
    ) -> None:
        if OpenAI is None:
            raise ValueError("OpenAI package is required for description scoring.")
        self.repeat = max(1, int(repeat))
        self._api_key = llm_api_key
        self._base_url = llm_base_url
        self._timeout = llm_timeout
        self._temperature = llm_temperature
        self._max_tokens = max(1, int(llm_max_tokens))
        self._client = None
        self._model = None
        self._stats = {}

    def _ensure_client(self):
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            self._model = self._client.models.list().data[0].id

    def _call_llm(self, prompt: str) -> str:
        self._ensure_client()
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a scoring assistant. Always reply with exactly one digit (0, 1, or 2). Never output any other text.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            timeout=self._timeout,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return (completion.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_score(reply: str) -> int | None:
        # 第一优先级：寻找最后一个 Score/Rating X 或 [[X]] 模式
        # 用最后一个而非第一个，避免 LLM 复述评分标准时误匹配
        matches = re.findall(r"(?:score|rating)\s*[:\-]?\s*([012])", reply, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1])

        # 第二优先级：寻找独立的 0, 1, 2 (取最后一个)
        matches = re.findall(r"\b([012])\b", reply)
        if matches:
            return int(matches[-1])

        # 第三优先级：文本中出现的最后一个 0, 1, 2
        matches = re.findall(r"([012])", reply)
        if matches:
            return int(matches[-1])
        return None

    def _accumulate(self, model: str, bench: str, top1: str, ability: str, task: str, metric: str, score: int):
        key = (model, bench, top1, ability, task, metric)
        if key not in self._stats:
            self._stats[key] = {"counts": Counter(), "total": 0, "sum": 0.0}
        stat = self._stats[key]
        stat["counts"][score] += 1
        stat["total"] += 1
        stat["sum"] += score

    def score_sample(
        self,
        *,
        model: str,
        bench: str,
        top1: str,
        ability: str,
        task: str,
        response: str,
        label: str,
    ) -> dict[str, list[int]]:
        response = (response or "").strip()
        label = (label or "").strip()
        if not response or not label:
            return {}

        metrics_scores: dict[str, list[int]] = {}
        for metric, builder in self.METRIC_BUILDERS.items():
            prompt = builder(response, label)
            metric_scores: list[int] = []
            for _ in range(self.repeat):
                try:
                    reply = self._call_llm(prompt)
                except Exception as exc:
                    print(f"[WARN] Description scoring failed for {metric}: {exc}")
                    continue
                score = self._parse_score(reply)
                if score is None:
                    print(f"[WARN] Unable to parse score for {metric}: {reply!r}")
                    continue
                metric_scores.append(score)
                self._accumulate(model, bench, top1, ability, task, metric, score)
            if metric_scores:
                metrics_scores[metric] = metric_scores
        return metrics_scores

    def collect_model_records(self, model: str) -> list[dict[str, object]]:
        records = []
        keys_to_remove = []
        for key, stat in self._stats.items():
            key_model, bench, top1, ability, task, metric = key
            if key_model != model:
                continue
            total = stat["total"]
            counts = stat["counts"]
            if not total:
                keys_to_remove.append(key)
                continue
            p0 = counts.get(0, 0) / total
            p1 = counts.get(1, 0) / total
            p2 = counts.get(2, 0) / total
            avg_score = stat["sum"] / total
            records.append(
                {
                    "model": key_model,
                    "bench": bench,
                    "top1": top1,
                    "ability": ability,
                    "task": task,
                    "metric": metric,
                    "total_runs": total,
                    "score0_count": counts.get(0, 0),
                    "score1_count": counts.get(1, 0),
                    "score2_count": counts.get(2, 0),
                    "P0": round(p0, 4),
                    "P1": round(p1, 4),
                    "P2": round(p2, 4),
                    "score": round(avg_score, 4),
                    "score_sum": stat["sum"],
                }
            )
            keys_to_remove.append(key)
        for key in keys_to_remove:
            self._stats.pop(key, None)
        return records


def load_results_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_results_jsonl(file_path: Path) -> list[dict]:
    return [obj for _, obj in read_jsonl(file_path)]


def _safe_to_excel(df: pd.DataFrame, path: Path) -> bool:
    """Write an .xlsx if Excel dependencies exist; otherwise skip."""

    try:
        df.to_excel(path)
        return True
    except ModuleNotFoundError as exc:
        # Most common: openpyxl missing.
        print(f"[WARN] Failed to write Excel file {path}: {exc}. Install with `pip install openpyxl`.")
        return False
    except Exception as exc:
        print(f"[WARN] Failed to write Excel file {path}: {exc}")
        return False


def _format_label_for_llm(label_field) -> str:
    if isinstance(label_field, list):
        parts = [str(item).strip() for item in label_field if str(item).strip()]
        return "\n".join(parts)
    return str(label_field or "").strip()


def _task_configs_for_model_dir(model_dir: Path):
    """Return iterable of (bench_type, tasks_dict, base_dir).

    Supports both historical layout:
      <model>/<top1>/<ability>/<task>.json
    and current layout:
      <model>/<single|multi>/<top1>/<ability>/<task>.json
    """

    has_single = (model_dir / "single").is_dir()
    has_multi = (model_dir / "multi").is_dir()
    if has_single or has_multi:
        items = []
        if has_single:
            items.append(("single", RSBENCH_TASKS, model_dir / "single"))
        if has_multi:
            items.append(("multi", RSBENCH_TASKS_MULTI, model_dir / "multi"))
        return items
    return [("single", RSBENCH_TASKS, model_dir)]


def _load_bench_problem_map(*, data_root: Path, bench_type: str, top1: str, task: str) -> dict[str, dict]:
    """Load bench problems and return id->problem mapping."""

    root = data_root / "bench" / ("Multi" if bench_type == "multi" else "single")
    problem_path = Path(root) / "problems" / top1 / f"{task}.json"
    if not problem_path.is_file():
        return {}
    try:
        with open(problem_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return {}
    out = {}
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("id"):
                out[str(it["id"])]=it
    return out


def _normalize_pair_type(data_type: object) -> str:
    """Normalize multi-pair data_type values from bench into stable buckets."""
    out = normalize_jsonl_pair_type(data_type)
    return "unknown" if out == "all" else out


def _evaluate_one_sample(*, evaluator: AccEvaluator, obj: dict, top1_level: str) -> bool:
    label_field = obj.get("label", "")

    if isinstance(label_field, list):
        raw_labels = label_field
    else:
        raw_labels = [label_field]

    label_letters: list[str] = []
    for raw_label in raw_labels:
        if not raw_label:
            continue
        text = str(raw_label).strip()
        if not text:
            continue
        upper_text = text.upper()
        tokens = [tok for tok in re.findall(r"[A-E]", upper_text) if tok in evaluator.OPTION_SET]
        if tokens:
            label_letters.extend(tokens)
            continue
        if "." in upper_text:
            letter = upper_text.split(".", 1)[0].strip()
            if letter in evaluator.OPTION_SET:
                label_letters.append(letter)
                continue
        if upper_text in evaluator.OPTION_SET:
            label_letters.append(upper_text)

    seen_letters: set[str] = set()
    label_letters = [letter for letter in label_letters if not (letter in seen_letters or seen_letters.add(letter))]
    label_full = [evaluator.extract_full_option(obj.get("query", ""), letter) for letter in label_letters]
    label_full = [item for item in label_full if item]

    if evaluator.use_llm and top1_level == "description":
        label_for_llm = _format_label_for_llm(obj.get("label", ""))
        result = evaluator.llm_evaluator(obj.get("query", ""), obj.get("response", ""), label_for_llm)
        if not result:
            result = evaluator.general_evaluater(obj.get("response", ""), label_full)
    else:
        result = evaluator.general_evaluater(obj.get("response", ""), label_full)
    return bool(result)


def evaluate_model(
    model_dir: Path,
    evaluator: AccEvaluator,
    description_scorer=None,
    *,
    data_root: Path,
    write_intra_splits: bool = True,
):
    records: list[dict[str, object]] = []

    for bench_type, tasks_dict, base_dir in _task_configs_for_model_dir(model_dir):
        for top1_level, ability_map in tasks_dict.items():
            for ability_name, task_list in ability_map.items():
                for task_name in task_list:
                    file_path = base_dir / top1_level / ability_name / f"{task_name}.json"
                    if not file_path.is_file():
                        continue

                    try:
                        data = load_results_file(file_path)
                    except (json.JSONDecodeError, OSError) as exc:
                        print(f"[WARN] Failed to load {file_path}: {exc}. Skipping this task.")
                        continue

                    # Enrich multi whether/how/what with intra split tags from bench.
                    bench_map = {}
                    if bench_type == "multi" and top1_level in {"whether", "how", "what"}:
                        bench_map = _load_bench_problem_map(data_root=data_root, bench_type=bench_type, top1=top1_level, task=task_name)

                    correct = 0
                    total = len(data) if isinstance(data, list) else 0
                    if not isinstance(data, list):
                        print(f"[WARN] Unexpected file content (not a list): {file_path}")
                        continue

                    for obj in tqdm(
                        data,
                        desc=f"{model_dir.name} | {bench_type}/{top1_level}/{ability_name}/{task_name}",
                        leave=False,
                    ):
                        if not isinstance(obj, dict):
                            continue

                        if bench_map:
                            sample_id = obj.get("id")
                            sample_id_str = str(sample_id) if sample_id is not None else ""
                            if sample_id_str and sample_id_str in bench_map:
                                dt = bench_map[sample_id_str].get("data_type")
                            else:
                                dt = None
                            if dt:
                                obj["data_type"] = _normalize_pair_type(dt)

                        result = _evaluate_one_sample(evaluator=evaluator, obj=obj, top1_level=top1_level)
                        obj["result"] = result
                        if result:
                            correct += 1

                        if top1_level == "description" and description_scorer is not None:
                            desc_scores = description_scorer.score_sample(
                                model=model_dir.name,
                                bench=bench_type,
                                top1=top1_level,
                                ability=ability_name,
                                task=task_name,
                                response=obj.get("response", ""),
                                label=_format_label_for_llm(obj.get("label", "")),
                            )
                            if desc_scores:
                                obj["description_scores"] = desc_scores

                    accuracy_value = (correct / total) if total else 0.0
                    accuracy = round(accuracy_value, 4)
                    save_results_file(file_path, data)

                    records.append(
                        {
                            "model": model_dir.name,
                            "bench": bench_type,
                            "pair_type": "all",
                            "top1": top1_level,
                            "ability": ability_name,
                            "task": task_name,
                            "accuracy": accuracy,
                            "accuracy_raw": accuracy_value,
                            "correct": correct,
                            "total": total,
                        }
                    )

                    # Multi intra split statistics + write split json for comparison.
                    if bench_type == "multi" and top1_level in {"whether", "how", "what"}:
                        # Partition by obj['data_type'] (bench provides intra-image / inter-temporal).
                        partitions: dict[str, list[dict]] = {"intra-image": [], "inter-temporal": [], "unknown": []}
                        for obj in data:
                            if not isinstance(obj, dict):
                                continue
                            dt = _normalize_pair_type(obj.get("data_type"))
                            partitions.setdefault(dt, [])
                            partitions[dt].append(obj)

                        for dt, subset in partitions.items():
                            if not subset:
                                continue
                            sub_correct = sum(1 for o in subset if isinstance(o, dict) and o.get("result") is True)
                            sub_total = len(subset)
                            sub_acc_val = (sub_correct / sub_total) if sub_total else 0.0
                            records.append(
                                {
                                    "model": model_dir.name,
                                    "bench": bench_type,
                                    "pair_type": dt,
                                    "top1": top1_level,
                                    "ability": ability_name,
                                    "task": task_name,
                                    "accuracy": round(sub_acc_val, 4),
                                    "accuracy_raw": sub_acc_val,
                                    "correct": sub_correct,
                                    "total": sub_total,
                                }
                            )

                            if write_intra_splits:
                                split_path = file_path.with_name(f"{task_name}__{dt}.json")
                                save_results_file(split_path, subset)

    description_records = description_scorer.collect_model_records(model_dir.name) if description_scorer else []
    return records, description_records


def evaluate_model_jsonl(
    model_dir: Path,
    evaluator: AccEvaluator,
    description_scorer=None,
):
    input_path = model_dir / "predictions.jsonl"
    if not input_path.is_file():
        input_path = model_dir / "evaluated.jsonl"
    if not input_path.is_file():
        return [], []

    try:
        data = load_results_jsonl(input_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to load {input_path}: {exc}. Skipping this model.")
        return [], []

    group_stats: dict[tuple[str, str, str, str, str], dict[str, int]] = {}

    def update_group(*, bench: str, pair_type: str, top1: str, ability: str, task: str, result: bool):
        key = (bench, pair_type, top1, ability, task)
        stat = group_stats.setdefault(key, {"correct": 0, "total": 0})
        stat["total"] += 1
        if result:
            stat["correct"] += 1

    for obj in tqdm(data, desc=f"{model_dir.name} | jsonl", leave=False):
        if not isinstance(obj, dict):
            continue

        norm = normalize_question_record(obj)
        bench = str(obj.get("bench") or norm["bench"])
        top1_level = str(obj.get("top1") or norm["top1"])
        ability_name = str(obj.get("ability") or norm["ability"])
        task_name = str(obj.get("task") or norm["task"])
        pair_type = normalize_jsonl_pair_type(obj.get("pair_type") or obj.get("data_type") or norm["pair_type"])

        obj["bench"] = bench
        obj["top1"] = top1_level
        obj["ability"] = ability_name
        obj["task"] = task_name
        obj["pair_type"] = pair_type

        result = _evaluate_one_sample(evaluator=evaluator, obj=obj, top1_level=top1_level)
        obj["result"] = result

        update_group(bench=bench, pair_type="all", top1=top1_level, ability=ability_name, task=task_name, result=result)
        if bench == "multi" and top1_level in {"whether", "how", "what"} and pair_type in {"intra-image", "inter-temporal"}:
            update_group(bench=bench, pair_type=pair_type, top1=top1_level, ability=ability_name, task=task_name, result=result)

        if top1_level == "description" and description_scorer is not None:
            desc_scores = description_scorer.score_sample(
                model=model_dir.name,
                bench=bench,
                top1=top1_level,
                ability=ability_name,
                task=task_name,
                response=obj.get("response", ""),
                label=_format_label_for_llm(obj.get("label", "")),
            )
            if desc_scores:
                obj["description_scores"] = desc_scores

    write_jsonl(model_dir / "evaluated.jsonl", data)

    records: list[dict[str, object]] = []
    for (bench, pair_type, top1, ability, task), stat in sorted(group_stats.items()):
        total = stat["total"]
        correct = stat["correct"]
        accuracy_value = (correct / total) if total else 0.0
        records.append(
            {
                "model": model_dir.name,
                "bench": bench,
                "pair_type": pair_type,
                "top1": top1,
                "ability": ability,
                "task": task,
                "accuracy": round(accuracy_value, 4),
                "accuracy_raw": accuracy_value,
                "correct": correct,
                "total": total,
            }
        )

    description_records = description_scorer.collect_model_records(model_dir.name) if description_scorer else []
    return records, description_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate RSBench model outputs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("SENSEBENCH_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "inference")),
        help="Directory that contains model inference outputs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("SENSEBENCH_DATA_ROOT", PROJECT_ROOT / "data")),
        help="Dataset root used to read bench metadata for multi-image split statistics.",
    )
    parser.add_argument(
        "--write-intra-splits",
        action="store_true",
        help="For multi whether/how/what, write per-task split json files: <task>__intra-image.json and <task>__intra-temple.json.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model directory names to evaluate (e.g. Falcon LHRS-Bot). If omitted, evaluates all subfolders under --output-root.",
    )
    parser.add_argument("--use-llm", action="store_true", help="Use an LLM judge for multiple-choice tasks (description fallback).")
    parser.add_argument(
        "--description-llm",
        action="store_true",
        help="Use an LLM judge to compute description metrics (completeness, correctness, faithfulness).",
    )
    parser.add_argument("--description-repeat", type=int, default=1, help="Number of repeated LLM evaluations per description metric.")
    parser.add_argument("--llm-api-key", type=str, default="sk-123456", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--llm-base-url", type=str, default="http://0.0.0.0:23333/v1", help="Base URL for the OpenAI-compatible endpoint.")
    parser.add_argument("--llm-timeout", type=float, default=10.0, help="Timeout (seconds) for LLM API calls.")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Sampling temperature for LLM API calls.")
    parser.add_argument("--llm-max-tokens", type=int, default=2, help="Max tokens for LLM judge/scorer replies.")
    args = parser.parse_args()

    output_root = args.output_root
    data_root = args.data_root
    if not output_root.exists():
        raise SystemExit(f"output directory not found: {output_root}")

    evaluator = AccEvaluator(
        use_llm=args.use_llm,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_timeout=args.llm_timeout,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
    )
    description_scorer = None
    if args.description_llm:
        description_scorer = DescriptionScorer(
            repeat=args.description_repeat,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            llm_timeout=args.llm_timeout,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
        )
    model_filter = set(args.models) if args.models else None
    for item in sorted(output_root.iterdir()):
        if item.is_dir():
            if model_filter is not None and item.name not in model_filter:
                continue
            if (item / "predictions.jsonl").is_file() or (item / "evaluated.jsonl").is_file():
                evaluate_model_jsonl(
                    item,
                    evaluator,
                    description_scorer,
                )
            else:
                evaluate_model(
                    item,
                    evaluator,
                    description_scorer,
                    data_root=data_root,
                    write_intra_splits=args.write_intra_splits,
                )

    print("Evaluation complete. Wrote evaluated.jsonl for the processed models.")


if __name__ == "__main__":
    main()
