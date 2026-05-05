"""Async API inference for SenseBench with retry, timeout, and incremental saving."""

import asyncio
import json
import argparse
import os
import base64
import io
from pathlib import Path

import aiohttp
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
    "qwen": {"model": "qwen3.5-plus", "base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key_env": "QWEN_API_KEY", "thinking": False},
    "gpt": {"model": "gpt-5.4", "base": "https://api.openai-proxy.org/v1", "key_env": "GPT_API_KEY", "thinking": None},
    "gemini": {"model": "gemini-3.1-pro-preview", "base": "https://api.openai-proxy.org/google", "key_env": "GEMINI_API_KEY", "thinking": False, "gemini": True},
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


async def call_openai(session, cfg, messages, max_tokens, thinking):
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
    if thinking is not None:
        body["enable_thinking"] = thinking

    url = f"{cfg['base']}/chat/completions"
    async with session.post(url, json=body, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"API {resp.status}: {data}")
        return data["choices"][0]["message"]["content"]


async def call_gemini(session, cfg, contents, max_tokens, thinking):
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    gen_config = {"maxOutputTokens": max_tokens, "temperature": 0.0}
    if not thinking:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}

    body = {
        "contents": [{"parts": contents}],
        "generationConfig": gen_config,
    }
    if cfg.get("sys_prompt"):
        body["systemInstruction"] = {"parts": [{"text": cfg["sys_prompt"]}]}

    url = f"{cfg['base']}/models/{cfg['model']}:generateContent"
    async with session.post(url, json=body, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"Gemini {resp.status}: {data}")
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def infer_one(session, cfg, entry, sem, pbar):
    async with sem:
        meta = entry["meta"]
        task = meta["task"]
        sys_prompt = PROMPT_MAP.get(task)
        max_tokens = 8 if task in ("whether", "what", "how") else 192

        for attempt in range(MAX_RETRIES):
            try:
                # Build message content
                user_parts = []
                for part in entry["body"]["messages"][0]["content"]:
                    if part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        png_bytes = await download_and_resize(session, url)
                        b64 = base64.b64encode(png_bytes).decode()
                        user_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                    else:
                        user_parts.append(part)

                if cfg.get("gemini"):
                    # Gemini format
                    gemini_parts = []
                    for part in entry["body"]["messages"][0]["content"]:
                        if part["type"] == "image_url":
                            url = part["image_url"]["url"]
                            png_bytes = await download_and_resize(session, url)
                            gemini_parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png_bytes).decode()}})
                        else:
                            gemini_parts.append({"text": part["text"]})
                    cfg_copy = {**cfg, "sys_prompt": sys_prompt}
                    answer = await call_gemini(session, cfg_copy, gemini_parts, max_tokens, cfg["thinking"])
                else:
                    messages = []
                    if sys_prompt:
                        messages.append({"role": "system", "content": sys_prompt})
                    messages.append({"role": "user", "content": user_parts})
                    answer = await call_openai(session, cfg, messages, max_tokens, cfg.get("thinking"))

                pbar.update(1)
                return _build_result(meta, answer)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                pbar.update(1)
                return _build_result(meta, f"[ERROR] {e}")


def _build_result(meta, answer):
    return {
        "id": meta["record_id"],
        "prediction": answer,
        "answer": meta["answer"],
        **{k: meta[k] for k in ["task", "distortion_family", "distortion_type", "image_count", "modality", "domain", "comparison"]},
    }


async def main_async(args):
    cfg = MODELS[args.model]
    api_key = args.api_key or os.environ.get(cfg["key_env"])
    if not api_key:
        raise SystemExit(f"Set {cfg['key_env']} in .env or pass --api-key")
    cfg = {**cfg, "api_key": api_key}

    output = Path(args.output) if args.output else Path(f"outputs/inference/{args.model}/predictions.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load input
    entries = []
    with open(args.input) as f:
        for line in f:
            entries.append(json.loads(line))
    if args.limit:
        entries = entries[:args.limit]

    # Skip already done
    done_ids = set()
    if output.exists():
        with open(output) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already done")

    todo = [e for e in entries if e["meta"]["record_id"] not in done_ids]
    print(f"Model: {cfg['model']} @ {cfg['base']}")
    print(f"Todo: {len(todo)} / {len(entries)}, concurrency: {args.workers}")

    sem = asyncio.Semaphore(args.workers)
    pbar = tqdm(total=len(todo), desc=args.model)

    connector = aiohttp.TCPConnector(limit=args.workers + 4, limit_per_host=args.workers + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(infer_one(session, cfg, entry, sem, pbar)) for entry in todo]

        with open(output, "a") as out:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                except Exception as e:
                    result = {"prediction": f"[FATAL] {e}"}
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
