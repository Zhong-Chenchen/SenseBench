import os
import os.path as osp
import json
import torch
import sys
import importlib.util
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from configs.jsonl_utils import get_record_images, normalize_question_record, read_jsonl, write_jsonl

# Repository paths.
config_dir = Path(__file__).resolve().parent
root_dir = config_dir.parent
DEFAULT_OSS_URL_MAP = config_dir / "cache_dir" / "oss_url_map.json"

# --- 1. Environment & Dependency Handling ---
# Lazy import placeholders
PtEngine = None
RequestConfig = None
InferRequest = None
get_template = None
seed_everything = None
VllmEngine = None

def _import_swift_modules():
    global PtEngine, RequestConfig, InferRequest, get_template, seed_everything, VllmEngine
    if PtEngine is not None:
        return

    sys.path.insert(0, str(root_dir / "ms-swift"))
    try:
        from swift.llm import PtEngine as _Pt, RequestConfig as _RC, InferRequest as _IR, get_template as _GT
        from swift.utils import seed_everything as _SE
        PtEngine, RequestConfig, InferRequest, get_template, seed_everything = _Pt, _RC, _IR, _GT, _SE
    except ImportError as e:
        raise ImportError(f"Failed to import ms-swift modules. Ensure ms-swift is available if using standard models. Error: {e}")

    try:
        from swift.llm import VllmEngine as _VE
        VllmEngine = _VE
    except ImportError:
        VllmEngine = None

# Minimal local implementations for RS-only inference to avoid swift dependency
@dataclass
class LocalRequestConfig:
    max_tokens: int = 128
    temperature: float = 0.0

@dataclass
class LocalInferRequest:
    messages: List[dict]
    images: Optional[List[str]] = None

def local_seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. API Engine Adapter (OpenAI-compatible: GPT-4o, Claude, Qwen-VL, etc.) ---

class APIEngineAdapter:
    """Wraps any OpenAI-compatible vision API."""

    def __init__(self, model: str, api_key: str, api_base: Optional[str] = None, workers: int = 16):
        import openai, json
        self._model = model
        self._workers = workers
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base or "https://api.openai.com/v1",
            max_retries=6,
        )
        self.supports_multi_image = True
        # Optional OSS URL map. If absent, images are sent as base64 data URLs.
        url_map_path = Path(os.environ.get("SENSEBENCH_OSS_URL_MAP", str(DEFAULT_OSS_URL_MAP)))
        self._url_map: dict = json.loads(url_map_path.read_text()) if url_map_path.exists() else {}
        if self._url_map:
            print(f"[INFO] OSS URL map loaded: {len(self._url_map)} entries, base64 encoding disabled.")

    def _image_url(self, abs_path: str) -> str:
        # Try OSS URL first (fast), fallback to base64 (slow)
        try:
            rel = str(Path(abs_path).relative_to(root_dir / "data"))
        except ValueError:
            rel = None
        if rel and rel in self._url_map:
            return self._url_map[rel]
        import base64, mimetypes
        _mime_fallback = {
            ".tif": "image/tiff", ".tiff": "image/tiff",
            ".jp2": "image/jp2",  ".j2k":  "image/jp2",
            ".bmp": "image/bmp",  ".webp": "image/webp",
        }
        ext = Path(abs_path).suffix.lower()
        mime = mimetypes.guess_type(abs_path)[0] or _mime_fallback.get(ext, "image/jpeg")
        with open(abs_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"

    class _Choice:
        def __init__(self, content):
            self.message = type("Msg", (), {"content": content})()

    class _Resp:
        def __init__(self, content):
            self.choices = [APIEngineAdapter._Choice(content)]

    def infer(self, requests, request_config):
        resps = []
        for req in requests:
            content = []
            for img_path in (req.images or []):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_url(img_path)},
                })
            content.append({"type": "text", "text": req.messages[-1]["content"]})
            messages = list(req.messages[:-1])
            messages.append({"role": "user", "content": content})
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=getattr(request_config, "max_tokens", 128),
                temperature=getattr(request_config, "temperature", 0.0),
            )
            resps.append(self._Resp(resp.choices[0].message.content))
        return resps


# --- 3. RS Model Discovery & Adapter ---

def _list_rs_model_dirnames() -> List[str]:
    rs_root = root_dir / "other_models"
    if not rs_root.exists():
        return []
    names = []
    for p in rs_root.iterdir():
        if p.is_dir() and (p / "rs_infer.py").exists() and not p.name.startswith("__"):
            names.append(p.name)
    return sorted(names)

def _resolve_rs_model_dirname(model_id: str) -> Optional[str]:
    model_id = str(model_id).strip()
    if not model_id:
        return None
    
    # 提取核心名称，允许 rs: 前缀。如果包含 / 且不是自定义RS路径，通常不是RS模型
    wanted = model_id[3:].strip() if model_id.lower().startswith("rs:") else model_id
    if not wanted or "/" in wanted:
        return None

    candidates = _list_rs_model_dirnames()
    
    # 1. 精确匹配 (Case-insensitive)
    for dirname in candidates:
        if dirname.lower() == wanted.lower():
            return dirname
            
    # 2. 模糊匹配: wanted 是 dirname 的子串 (e.g. lhrs -> LHRS-Bot)
    # 取最接近的（最长匹配或首个匹配），这里简单取匹配到的第一个
    for dirname in candidates:
        if wanted.lower() in dirname.lower():
            print(f"[INFO] Auto-resolved RS model '{model_id}' to '{dirname}'")
            return dirname
            
    return None

def _parse_rs_model_path_overrides(args) -> dict:
    overrides = {}
    for item in getattr(args, "rs_model_path", []) or []:
        if "=" not in item:
            raise ValueError(f"--rs_model_path must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if name and path:
            overrides[name.lower()] = path
    return overrides

def _rs_model_env_var(dirname: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in dirname.upper())
    return f"SENSEBENCH_RS_MODEL_PATH_{safe}"

def _sanitize_tag(text: str) -> str:
    tag = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag.strip("_")

def _resolve_rs_checkpoint_path(rs_dirname: str, args) -> Optional[str]:
    overrides = _parse_rs_model_path_overrides(args)
    if rs_dirname.lower() in overrides:
        return overrides[rs_dirname.lower()]
    env_value = os.environ.get(_rs_model_env_var(rs_dirname))
    return env_value or None

class RSEngineAdapter:
    """Adapts arbitrary other_models/*/rs_infer.py to the swift Engine interface."""
    def __init__(self, *, runtime, infer_mod):
        self._runtime = runtime
        self._infer_mod = infer_mod
        self.supports_multi_image = bool(getattr(infer_mod, "SUPPORTS_MULTI_IMAGE", False))

    class _Choice:
         def __init__(self, content): self.message = type('Msg', (), {'content': content})()
    
    class _Resp:
         def __init__(self, content): self.choices = [RSEngineAdapter._Choice(content)]

    def infer(self, requests, request_config):
        if hasattr(self._infer_mod, "generate_batch"):
            texts = self._infer_mod.generate_batch(
                runtime=self._runtime,
                images_list=[getattr(req, "images", None) or [] for req in requests],
                prompts=[req.messages[-1]["content"] for req in requests],
                temperature=getattr(request_config, "temperature", 0.0),
                max_new_tokens=getattr(request_config, "max_tokens", 128),
                supports_multi_image=self.supports_multi_image,
            )
            return [self._Resp(text) for text in texts]

        resps = []
        for req in requests:
            text = self._infer_mod.generate(
                runtime=self._runtime,
                images=getattr(req, "images", None) or [],
                prompt=req.messages[-1]["content"],
                temperature=getattr(request_config, "temperature", 0.0),
                max_new_tokens=getattr(request_config, "max_tokens", 128),
                supports_multi_image=self.supports_multi_image,
            )
            resps.append(self._Resp(text))
        return resps

# --- 3. Engine Loader ---

def load_inference_engine(model_id, args):
    """
    Loads proper engine (API, RS-custom, Pt, or Vllm).
    Returns: (engine, model_dir_name, is_rs_model)
    """
    # API model: e.g. api:gpt-4o, api:qwen-vl-max, api:claude-3-5-sonnet-20241022
    if str(model_id).startswith("api:"):
        api_model = model_id[4:].strip()
        api_key = getattr(args, "api_key", None) or os.environ.get("OPENAI_API_KEY", "")
        api_base = getattr(args, "api_base", None) or os.environ.get("OPENAI_API_BASE", None)
        workers = getattr(args, "api_workers", 16)
        engine = APIEngineAdapter(model=api_model, api_key=api_key, api_base=api_base, workers=workers)
        model_dir_name = "api_" + api_model.replace("/", "_").replace(":", "_")
        return engine, model_dir_name, False

    rs_dirname = _resolve_rs_model_dirname(str(model_id))
    is_rs_model = rs_dirname is not None

    backend = args.backend
    # [Fix] deepseek-vl 存在 config 兼容性问题，且 vLLM 支持尚不稳定，强制降级为 pt 后端
    if "deepseek" in str(model_id).lower() and "chat" in str(model_id).lower():
         if backend == "vllm":
            print(f"[WARN] Forcing backend='pt' for {model_id} due to vLLM compatibility issues.")
            backend = "pt"
         # [Fix] 强制使用 eager attention，因为 deepseek-vl 在 transformers 中暂不支持 FA2
         if hasattr(args, 'attn') and args.attn == 'flash_attention_2':
            print(f"[WARN] Forcing attn='eager' for {model_id} (FA2 not supported in transformers).")
            args.attn = 'eager'

    if is_rs_model:
        # Load RS custom model
        infer_path = root_dir / "other_models" / rs_dirname / "rs_infer.py"
        safe_mod_name = "".join(ch if ch.isalnum() else "_" for ch in rs_dirname)
        spec = importlib.util.spec_from_file_location(f"sensebench_rs_{safe_mod_name}", infer_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load RS adapter: {infer_path}")
        infer_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(infer_mod)
        if not hasattr(infer_mod, "load_runtime"):
             raise AttributeError(f"RS model '{rs_dirname}' missing load_runtime()")
        
        model_path = _resolve_rs_checkpoint_path(rs_dirname, args)
        if model_path:
            runtime = infer_mod.load_runtime(model_path=model_path)
        else:
            runtime = infer_mod.load_runtime()
        engine = RSEngineAdapter(runtime=runtime, infer_mod=infer_mod)
        backend_tag = _sanitize_tag(runtime.get("backend", "rs"))
        model_tag = runtime.get("model_name") or runtime.get("model_path") or model_id
        model_tag = _sanitize_tag(Path(str(model_tag)).name)
        model_dir_name = "_".join([rs_dirname, backend_tag, model_tag]) if model_tag else "_".join([rs_dirname, backend_tag])
    else:
        # Load standard swift model
        _import_swift_modules()
        if PtEngine is None:
            raise ImportError("ms-swift not found. Install it to use standard models.")
        
        model_dir_name = model_id.replace("/", "_")
        
        if backend == 'pt':
            engine = PtEngine(model_id, torch.bfloat16, max_batch_size=1, device_map="auto", attn_impl=args.attn)
        elif backend == 'vllm':
            if VllmEngine is None:
                raise ImportError(
                    "VllmEngine is not available (vllm not installed in this env). "
                    "Use --backend pt or install vllm in this environment."
                )
            # Simplified VLLM loading logic for conciseness
            vllm_len = args.vllm_max_model_len or (args.max_length or 16384)
            engine = VllmEngine(
                model_id, torch.bfloat16,
                max_model_len=vllm_len,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_num_seqs=args.vllm_max_num_seqs,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                pipeline_parallel_size=args.vllm_pipeline_parallel_size,
                enforce_eager=args.vllm_enforce_eager
            )
        else:
            raise ValueError(f"Unknown backend: {args.backend}")

        # Configure template (standard models only)
        if hasattr(engine, 'max_model_len'):
            t_len = min(args.max_length or engine.max_model_len, engine.max_model_len)
        else:
            t_len = args.max_length

        # enable_thinking = (args.enable_thinking == 'true') if args.enable_thinking != 'auto' else None
        # Force disable thinking to ensure standard behavior
        enable_thinking = False
        engine.default_template = get_template(
            engine.model_meta.template, engine.processor,
            max_length=t_len, truncation_strategy=args.truncation_strategy,
            enable_thinking=enable_thinking
        )

    return engine, model_dir_name, is_rs_model

# --- 4. Evaluator ---

class RSBenchEvaluator:
    def __init__(self, *, root_path, image_root_path, out_path, sys_prompt, engine, top1_level_name, ability_name, task_name):
        self.root_path = root_path
        self.image_root_path = image_root_path
        self.out_path = out_path
        self.sys_prompt = sys_prompt
        self.engine = engine
        self.top1_level_name = top1_level_name
        self.task_name = task_name
        
        # Decide which Config/Request class to use
        if isinstance(engine, (RSEngineAdapter, APIEngineAdapter)):
            self.ConfCls = LocalRequestConfig
            self.ReqCls = LocalInferRequest
            self.seed_fn = local_seed_everything
        else:
            # Assume Swift Engine. Swift must be imported.
            if RequestConfig is None:
                _import_swift_modules()
            self.ConfCls = RequestConfig
            self.ReqCls = InferRequest
            self.seed_fn = seed_everything

        self.request_config = self.ConfCls(max_tokens=128, temperature=0.0)

    def vlm_inference(self):
        self.seed_fn(42)
        task_file = osp.join(self.root_path, "problems", self.top1_level_name, f"{self.task_name}.json")
        out_file = osp.join(self.out_path, f"{self.task_name}.json")
        
        if osp.exists(out_file):
            print("Already exists, skipping")
            return

        with open(task_file, "r") as f:
            datas = json.load(f)

        def _infer_one(data):
            images = self._collect_images(data)
            if len(images) > 1 and getattr(self.engine, "supports_multi_image", True) is False:
                return None
            q = data["question"]
            content = f"{self.sys_prompt}{q}" if self.sys_prompt else q
            req = self.ReqCls(messages=[{"role": "user", "content": content}], images=images or None)
            resp = self.engine.infer([req], request_config=self.request_config)
            res = {
                "id": data["id"],
                "query": content,
                "response": resp[0].choices[0].message.content,
                "label": data["answer"],
            }
            if "image_path" in data:
                res["image_path"] = data["image_path"]
            else:
                for k, v in data.items():
                    if k.startswith("image_"):
                        res[k] = v
            return res

        res_list = []
        skipped_cnt = 0

        if isinstance(self.engine, APIEngineAdapter):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = [None] * len(datas)
            with ThreadPoolExecutor(max_workers=self.engine._workers) as executor:
                future_to_idx = {executor.submit(_infer_one, d): i for i, d in enumerate(datas)}
                for future in tqdm(as_completed(future_to_idx), total=len(datas), desc="Infer"):
                    i = future_to_idx[future]
                    result = future.result()
                    if result is None:
                        skipped_cnt += 1
                    else:
                        results[i] = result
            res_list = [r for r in results if r is not None]
        else:
            for data in tqdm(datas, desc="Infer"):
                result = _infer_one(data)
                if result is None:
                    skipped_cnt += 1
                else:
                    res_list.append(result)

        if skipped_cnt:
            print(f"[SKIP] {skipped_cnt}/{len(datas)} samples (multi-image support disabled)")

        with open(out_file, "w") as f:
            json.dump(res_list, f, indent=4, ensure_ascii=False)

    def _collect_images(self, data):
        images = []
        # Single image path
        if "image_path" in data and data["image_path"]:
            p = self._resolve_path(data["image_path"])
            if p: images.append(p)
        # Multi image (image_1, image_2...)
        else:
            idx = 1
            while f"image_{idx}" in data:
                p = self._resolve_path(data[f"image_{idx}"])
                if p: images.append(p)
                idx += 1
        return images

    def _resolve_path(self, p):
        if not p: return None
        if p.startswith(("http", "/")) and osp.exists(p): return p
        cand = osp.join(self.image_root_path, p.lstrip("/"))
        return cand if osp.exists(cand) else p


class JsonlRSBenchInferencer:
    def __init__(
        self,
        *,
        jsonl_path,
        image_root_path,
        out_file,
        sys_prompt_fn,
        engine,
        overwrite: bool = False,
        batch_size: int = 8,
    ):
        self.jsonl_path = str(jsonl_path)
        self.image_root_path = str(image_root_path)
        self.out_file = str(out_file)
        self.sys_prompt_fn = sys_prompt_fn
        self.engine = engine
        self.overwrite = bool(overwrite)
        self.batch_size = max(1, int(batch_size))

        if isinstance(engine, (RSEngineAdapter, APIEngineAdapter)):
            self.ConfCls = LocalRequestConfig
            self.ReqCls = LocalInferRequest
            self.seed_fn = local_seed_everything
        else:
            if RequestConfig is None:
                _import_swift_modules()
            self.ConfCls = RequestConfig
            self.ReqCls = InferRequest
            self.seed_fn = seed_everything

        self.request_config = self.ConfCls(max_tokens=128, temperature=0.0)

    def vlm_inference(self):
        self.seed_fn(42)
        if osp.exists(self.out_file) and not self.overwrite:
            print(f"Already exists, skipping {self.out_file}")
            return
        rows = [obj for _, obj in read_jsonl(self.jsonl_path)]
        skipped_cnt = 0

        def _build_request(data):
            norm = normalize_question_record(data)
            images = self._collect_images(data)
            if len(images) > 1 and getattr(self.engine, "supports_multi_image", True) is False:
                return None

            question = data.get("question", "")
            sys_prompt = self.sys_prompt_fn(norm["top1"], norm["ability"])
            content = f"{sys_prompt}{question}" if sys_prompt else question
            req = self.ReqCls(messages=[{"role": "user", "content": content}], images=images or None)
            return req, content, question, norm, data

        def _format_result(data, content, question, norm, response):
            return {
                "id": data.get("id"),
                "bench": norm["bench"],
                "top1": norm["top1"],
                "ability": norm["ability"],
                "task": norm["task"],
                "pair_type": norm["pair_type"],
                "query": content,
                "question": question,
                "response": response,
                "label": data.get("answer", data.get("label", "")),
                "images": get_record_images(data),
                "meta": data.get("meta", {}),
            }

        def _infer_batch(batch_data):
            prepared = []
            for data in batch_data:
                item = _build_request(data)
                if item is None:
                    continue
                prepared.append(item)

            if not prepared:
                return [], len(batch_data)

            batch_reqs = [item[0] for item in prepared]
            try:
                resp_list = self.engine.infer(batch_reqs, request_config=self.request_config)
                if len(resp_list) != len(prepared):
                    raise RuntimeError("engine returned mismatched batch size")
            except Exception:
                resp_list = []
                for req, _, _, _, _ in prepared:
                    try:
                        resp = self.engine.infer([req], request_config=self.request_config)
                        resp_list.append(resp[0])
                    except Exception:
                        resp_list.append(None)

            results = []
            for (req, content, question, norm, data), resp in zip(prepared, resp_list):
                if resp is None:
                    results.append(None)
                    continue
                response = resp.choices[0].message.content
                results.append(_format_result(data, content, question, norm, response))
            return results, len(batch_data) - len(prepared)

        def _infer_one(data):
            prepared = _build_request(data)
            if prepared is None:
                return None
            req, content, question, norm, data = prepared
            resp = self.engine.infer([req], request_config=self.request_config)
            response = resp[0].choices[0].message.content
            return _format_result(data, content, question, norm, response)

        if isinstance(self.engine, APIEngineAdapter):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = [None] * len(rows)
            with ThreadPoolExecutor(max_workers=self.engine._workers) as executor:
                future_to_idx = {executor.submit(_infer_one, d): i for i, d in enumerate(rows)}
                for future in tqdm(as_completed(future_to_idx), total=len(rows), desc="Infer"):
                    i = future_to_idx[future]
                    result = future.result()
                    if result is None:
                        skipped_cnt += 1
                    else:
                        results[i] = result
            res_list = [r for r in results if r is not None]
        else:
            res_list = []
            for i in tqdm(range(0, len(rows), self.batch_size), desc="Infer"):
                batch = rows[i:i + self.batch_size]
                batch_results, batch_skipped = _infer_batch(batch)
                skipped_cnt += batch_skipped
                for result in batch_results:
                    if result is not None:
                        res_list.append(result)

        if skipped_cnt:
            print(f"[SKIP] {skipped_cnt}/{len(rows)} samples (multi-image support disabled)")

        write_jsonl(self.out_file, res_list)

    def _collect_images(self, data):
        images = []
        for raw in get_record_images(data):
            p = self._resolve_path(raw)
            if p:
                images.append(p)
        return images

    def _resolve_path(self, p):
        if not p:
            return None
        if str(p).startswith(("http://", "https://")):
            return p
        if osp.isabs(p) and osp.exists(p):
            return p
        cand = osp.join(self.image_root_path, str(p).lstrip("/"))
        return cand if osp.exists(cand) else p
