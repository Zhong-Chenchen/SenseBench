import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import time
import argparse
import torch
import gc
from configs.infer_config import (
    MODEL_LIST,
    RSBENCH_TASKS,
    RSBENCH_TASKS_MULTI,
    sys_prompt_2choice,
    sys_prompt_3choice,
    sys_prompt_4choice,
    sys_prompt_multi,
)
from configs.infer_utils import load_inference_engine, RSBenchEvaluator, JsonlRSBenchInferencer

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", nargs="+", help="Model IDs (e.g. ZhipuAI/glm-4v-9b or rs:GeoChat)")
    parser.add_argument("--backend", default="pt", choices=["pt", "vllm"], help="Inference backend")
    parser.add_argument(
        "--attn",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention impl: 'flash_attention_2' (fastest, requires flash-attn), 'sdpa' (torch>=2.1, PyTorch native efficient attn), or 'eager' (slow, fallback)",
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=256)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_auto_parallel", action="store_true")
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--enable_thinking", default="false")
    parser.add_argument("--truncation_strategy", default="right")
    parser.add_argument("--api_key", default=None, help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--api_base", default=None, help="API base URL (or set OPENAI_API_BASE env var)")
    parser.add_argument("--api_workers", type=int, default=16, help="Concurrent workers for API inference")
    parser.add_argument(
        "--data_root",
        default=os.environ.get("SENSEBENCH_DATA_ROOT", str(PROJECT_ROOT / "data")),
        help="Dataset root containing questions.jsonl/images, or the legacy bench/single and bench/Multi layout.",
    )
    parser.add_argument(
        "--input-jsonl",
        default=os.environ.get("SENSEBENCH_QUESTIONS_JSONL"),
        help="Unified JSONL question file. Defaults to <data_root>/questions.jsonl when present.",
    )
    parser.add_argument(
        "--output-format",
        default=os.environ.get("SENSEBENCH_OUTPUT_FORMAT", "jsonl"),
        choices=["jsonl", "legacy", "both"],
        help="Write unified predictions.jsonl, legacy per-task json files, or both.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing inference outputs.")
    parser.add_argument(
        "--output_dir",
        default=os.environ.get("SENSEBENCH_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "inference")),
        help="Root directory for inference outputs.",
    )
    parser.add_argument(
        "--rs_model_path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override an RS adapter checkpoint path, e.g. GeoChat=/path/to/geochat-7B. Can be repeated.",
    )

    args = parser.parse_args()
    if not args.model:
        args.model = MODEL_LIST
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.input_jsonl:
        args.input_jsonl = str(Path(args.input_jsonl).expanduser().resolve())
    else:
        candidate = Path(args.data_root) / "questions.jsonl"
        args.input_jsonl = str(candidate) if candidate.is_file() else None
    return args


def get_sys_prompt(top1, ability):
    if top1 == "whether":
        return sys_prompt_2choice
    if top1 == "what":
        return sys_prompt_multi if ability == "multi" else sys_prompt_4choice
    if top1 == "how":
        return sys_prompt_3choice
    return None


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    use_jsonl = args.output_format in {"jsonl", "both"}
    use_legacy = args.output_format in {"legacy", "both"}
    if use_jsonl and not args.input_jsonl:
        raise SystemExit("JSONL inference requested, but no --input-jsonl was provided and <data_root>/questions.jsonl was not found.")

    bench_configs = [
        ("single", data_root / "bench" / "single", RSBENCH_TASKS),
        ("multi", data_root / "bench" / "Multi", RSBENCH_TASKS_MULTI),
    ]

    failed = []
    for model_id in args.model:
        print(f"\n=== Running: {model_id} ===")
        try:
            engine, model_dir_name, is_rs = load_inference_engine(model_id, args)
        except Exception as e:
            print(f"[ERROR] Load failed: {e}")
            failed.append((model_id, "load", str(e)))
            continue

        try:
            if use_jsonl:
                out_file = Path(args.output_dir) / model_dir_name / "predictions.jsonl"
                print(f">> JSONL inference: {model_dir_name} -> {out_file}")
                inferencer = JsonlRSBenchInferencer(
                    jsonl_path=args.input_jsonl,
                    image_root_path=str(data_root),
                    out_file=out_file,
                    sys_prompt_fn=get_sys_prompt,
                    engine=engine,
                    overwrite=args.overwrite,
                    batch_size=int(os.environ.get("SENSEBENCH_BATCH_SIZE", "8")),
                )
                inferencer.vlm_inference()

            if use_legacy:
                for bench_type, root, tasks in bench_configs:
                    if is_rs and bench_type == "multi" and not getattr(engine, "supports_multi_image", True):
                        print(f"[SKIP] {model_dir_name} does not support multi-image bench")
                        continue

                    for top1, top2_dict in tasks.items():
                        for ability, task_list in top2_dict.items():
                            for task_name in task_list:
                                print(f">> Task: {model_dir_name} [{bench_type}/{top1}/{ability}/{task_name}]")
                                out_path = os.path.join(args.output_dir, model_dir_name, bench_type, top1, ability)
                                os.makedirs(out_path, exist_ok=True)

                                try:
                                    evaluator = RSBenchEvaluator(
                                        root_path=str(root),
                                        image_root_path=str(data_root),
                                        out_path=out_path,
                                        sys_prompt=get_sys_prompt(top1, ability),
                                        engine=engine,
                                        top1_level_name=top1,
                                        ability_name=ability,
                                        task_name=task_name,
                                    )
                                    evaluator.vlm_inference()
                                except Exception as e:
                                    print(f"[ERROR] Task failed: {e}")
                                    failed.append((model_id, task_name, str(e)))
        finally:
            if "engine" in locals():
                del engine
            gc.collect()
            torch.cuda.empty_cache()
            if not is_rs:
                print("[INFO] Waiting 20s for GPU cleanup...")
                time.sleep(20)

    if failed:
        print("\nFailures:", failed)
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()
