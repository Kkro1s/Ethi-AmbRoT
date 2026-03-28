#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从仓库根目录运行::

    python scripts/run_qwen_eval.py [--dataset ...] [--output ...] [--limit N] [--sleep SEC]

默认结果：仓库根下 ``output/qwen_predictions.jsonl``（可用 ``--output`` 覆盖）。

Env: QWEN_API_KEY (or QWEN_MAX_API_KEY / DASHSCOPE_API_KEY), QWEN_BASE_URL (optional,
default https://dashscope.aliyuncs.com/compatible-mode/v1), QWEN_MODEL.
Optional: QWEN_TIMEOUT or QWEN_MAX_TIMEOUT (seconds, default 180).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ethi_ambrot.common_eval_utils import (
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

DEFAULT_OUTPUT = DEFAULT_EVAL_OUTPUT_DIR / "qwen_predictions.jsonl"


def _qwen_config() -> tuple[str, str, str, float]:
    api_key = (
        os.environ.get("QWEN_API_KEY")
        or os.environ.get("QWEN_MAX_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        print(
            "Missing API key. Set one of: QWEN_API_KEY, QWEN_MAX_API_KEY, DASHSCOPE_API_KEY",
            file=sys.stderr,
        )
        sys.exit(1)
    base_url = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("QWEN_MAX_API_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip()
    model = (
        os.environ.get("QWEN_MODEL") or os.environ.get("QWEN_MAX_MODEL_NAME") or "qwen-max"
    ).strip()
    timeout_raw = os.environ.get("QWEN_MAX_TIMEOUT") or os.environ.get("QWEN_TIMEOUT")
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

    load_env_candidates(REPO_ROOT)

    ap = argparse.ArgumentParser(description="Run Qwen (DashScope-compatible) on Chambi benchmark compact")
    configure_shared_eval_args(ap, DEFAULT_OUTPUT)
    args = ap.parse_args()

    api_key, base_url, model, timeout_sec = _qwen_config()
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
            print(f"Skip malformed row (missing id or input_text): {item!r}", file=sys.stderr)
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
