"""Async API inference for SenseBench with retry, timeout, and incremental saving."""

import asyncio
import json
import argparse
import os
import base64
import io
from pathlib import Path

import aiohttp
import requests
from PIL import Image as PILImage
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from configs.infer_config import sys_prompt_2choice, sys_prompt_3choice, sys_prompt_4choice, sys_prompt_multi

PROMPT_MAP = {
    "whether": sys_prompt_2choice,
    "what": sys_prompt_4choice,
    "how": sys_prompt_3choice,
    "description": None,
}

MODELS = {
    "qwen": {"model": "qwen3.5-plus", "base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key_env": "QWEN_API_KEY", "protocol": "openai"},
    "gpt": {"model": "gpt-5.4", "base": "https://api.openai-proxy.org/v1", "key_env": "GPT_API_KEY", "protocol": "openai"},
    "gemini": {"model": "gemini-3.1-pro-preview", "base": "https://api.openai-proxy.org/google", "key_env": "GEMINI_API_KEY", "protocol": "gemini"},
}

UNSUPPORTED_EXTS = {".tif", ".tiff", ".jp2", ".j2k", ".bmp"}
MAX_SIZE = (512, 512)
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3


async def download_and_resize(session, url):
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        data = await resp.read()
    img = PILImage.open(io.BytesIO(data)).convert("RGB")
    if img.width > MAX_SIZE[0] or img.height > MAX_SIZE[1]:
        img.thumbnail(MAX_SIZE, PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load_image_bytes(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
    if img.width > MAX_SIZE[0] or img.height > MAX_SIZE[1]:
        img.thumbnail(MAX_SIZE, PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


async def call_openai(session, cfg, messages, max_tokens):
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg["model"],
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    }
    url = f"{cfg['base']}/chat/completions"
    async with session.post(url, json=body, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"API {resp.status}: {data}")
        return data["choices"][0]["message"]["content"]


async def infer_one_openai(session, cfg, entry, sem, pbar):
    async with sem:
        meta = entry["meta"]
        task = meta["task"]
        sys_prompt = PROMPT_MAP.get(task)
        max_tokens = 8 if task in ("whether", "what", "how") else 192

        for attempt in range(MAX_RETRIES):
            try:
                user_parts = []
                for part in entry["body"]["messages"][0]["content"]:
                    if part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        png_bytes = await download_and_resize(session, url)
                        b64 = base64.b64encode(png_bytes).decode()
                        user_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                    else:
                        user_parts.append(part)

                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": user_parts})
                answer = await call_openai(session, cfg, messages, max_tokens)

                pbar.update(1)
                return _build_result(meta, entry, answer)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                pbar.update(1)
                return _build_result(meta, entry, f"[ERROR] {e}")


async def infer_one_gemini(client, model, entry, sem, pbar):
    async with sem:
        meta = entry["meta"]
        task = meta["task"]
        sys_prompt = PROMPT_MAP.get(task)
        max_tokens = 8 if task in ("whether", "what", "how") else 192

        for attempt in range(MAX_RETRIES):
            try:
                def _call():
                    import google.genai as genai
                    content_parts = []
                    for part in entry["body"]["messages"][0]["content"]:
                        if part["type"] == "image_url":
                            url = part["image_url"]["url"]
                            img_bytes, mime = _load_image_bytes(url)
                            content_parts.append(genai.types.Part.from_bytes(data=img_bytes, mime_type=mime))
                        elif part["type"] == "text":
                            content_parts.append(part["text"])

                    config = genai.types.GenerateContentConfig(
                        maxOutputTokens=max_tokens,
                        temperature=0.0,
                        system_instruction=sys_prompt,
                        thinkingConfig=genai.types.ThinkingConfig(thinkingBudget=0),
                    )
                    resp = client.models.generate_content(
                        model=model,
                        contents=content_parts,
                        config=config,
                    )
                    return resp.text

                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(None, _call)

                pbar.update(1)
                return _build_result(meta, entry, answer)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                pbar.update(1)
                return _build_result(meta, entry, f"[ERROR] {e}")


def _derive_ability(family, distortion_type, domain):
    if family == "sar" or (domain == "remote_sensing" and distortion_type in {"Speckle", "Sidelobe"}):
        return "sar"
    if family == "multi_distortion" or distortion_type == "multi_distortion":
        return "multi"
    return "optical"


def _build_result(meta, entry, answer):
    query = ""
    for part in entry["body"]["messages"][0]["content"]:
        if part["type"] == "text":
            query = part["text"]
            break

    task = meta["task"]
    family = meta.get("distortion_family", "")
    dtype = meta.get("distortion_type", "")
    image_count = meta.get("image_count", "single")
    domain = meta.get("domain", "")
    comparison = meta.get("comparison")

    return {
        "id": meta["record_id"],
        "bench": "multi" if image_count in ("multi", "multiple") else "single",
        "top1": task,
        "ability": _derive_ability(family, dtype, domain),
        "task": dtype,
        "pair_type": comparison if comparison else "all",
        "query": query,
        "question": query,
        "response": answer,
        "label": meta["answer"],
        "meta": {
            "image_count": image_count,
            "modality": meta.get("modality", ""),
            "task": task,
            "domain": domain,
            "distortion_family": family,
            "distortion_type": dtype,
            "comparison": comparison,
        },
    }


async def main_async(args):
    cfg = MODELS[args.model]
    api_key = args.api_key or os.environ.get(cfg["key_env"])
    if not api_key:
        raise SystemExit(f"Set {cfg['key_env']} in .env or pass --api-key")
    cfg = {**cfg, "api_key": api_key}

    output = Path(args.output) if args.output else Path(f"outputs/inference/{args.model}/predictions.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(args.input) as f:
        for line in f:
            entries.append(json.loads(line))
    if args.limit:
        entries = entries[:args.limit]

    done_ids = set()
    if output.exists():
        with open(output) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already done")

    todo = [e for e in entries if e["meta"]["record_id"] not in done_ids]
    print(f"Model: {cfg['model']} @ {cfg['base']} ({cfg['protocol']})")
    print(f"Todo: {len(todo)} / {len(entries)}, concurrency: {args.workers}")

    sem = asyncio.Semaphore(args.workers)
    pbar = tqdm(total=len(todo), desc=args.model)

    connector = aiohttp.TCPConnector(limit=args.workers + 4, limit_per_host=args.workers + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        if cfg["protocol"] == "gemini":
            import google.genai as genai
            client = genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(baseUrl=cfg["base"]))
            tasks = [asyncio.create_task(infer_one_gemini(client, cfg["model"], entry, sem, pbar)) for entry in todo]
        else:
            tasks = [asyncio.create_task(infer_one_openai(session, cfg, entry, sem, pbar)) for entry in todo]

        with open(output, "a") as out:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                except Exception as e:
                    result = {"response": f"[FATAL] {e}"}
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

    pbar.close()
    print(f"Done. Results: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--input", default="data/api_input.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
