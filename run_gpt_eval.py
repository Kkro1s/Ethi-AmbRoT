#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python run_gpt_eval.py [--dataset ...] [--output ...]

默认结果文件：output/gpt_predictions.jsonl

Env: OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1), OPENAI_MODEL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

from common_eval_utils import (
    DEFAULT_EVAL_OUTPUT_DIR,
    build_prompt,
    configure_shared_eval_args,
    extract_json_object,
    append_jsonl,
    load_dataset,
    load_done_ids,
    load_env_candidates,
    parse_model_record,
)

DEFAULT_OUTPUT = DEFAULT_EVAL_OUTPUT_DIR / "gpt_predictions.jsonl"


def _openai_config() -> tuple[str, str, str, float]:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("Missing OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    model = (os.environ.get("OPENAI_MODEL") or "").strip()
    if not model:
        print("Missing OPENAI_MODEL.", file=sys.stderr)
        sys.exit(1)
    timeout_raw = os.environ.get("OPENAI_TIMEOUT")
    try:
        timeout_sec = float(timeout_raw) if timeout_raw else 180.0
    except ValueError:
        timeout_sec = 180.0
    return api_key, base_url, model, timeout_sec


def main() -> int:
    try:
        from openai import OpenAI
    except ImportError:
        print("Install with: pip install openai", file=sys.stderr)
        return 1

    load_env_candidates(HERE)

    ap = argparse.ArgumentParser(description="Run OpenAI-compatible GPT on Chambi benchmark compact")
    configure_shared_eval_args(ap, DEFAULT_OUTPUT)
    args = ap.parse_args()

    api_key, base_url, model, timeout_sec = _openai_config()
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    try:
        items = load_dataset(args.dataset)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1

    done_ids = load_done_ids(args.output)
    model_name = model
    new_count = 0
    total = len(items)

    for item in items:
        if args.limit is not None and new_count >= args.limit:
            break
        sid = item.get("source_chambi_id")
        input_text = item.get("input_text")
        if sid is None or not isinstance(input_text, str):
            print(f"Skip malformed row: {item!r}", file=sys.stderr)
            continue
        if sid in done_ids:
            continue

        user_content = build_prompt(input_text)
        raw = ""
        parsed = None
        ok = False
        err: str | None = None
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = extract_json_object(raw)
            ok = True
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        rec = parse_model_record(sid, input_text, model_name, raw, parsed, ok, err)
        append_jsonl(args.output, rec)
        if ok:
            done_ids.add(sid)
        new_count += 1
        print(f"[{new_count}] source_chambi_id={sid} success={ok} (dataset size {total})", flush=True)
        time.sleep(max(0.0, args.sleep))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
