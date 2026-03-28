#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从仓库根目录运行::

    python scripts/llm/run_qwen_eval.py [--phase 1|2] [--phase1-jsonl ...] [--dataset ...] [--output ...]

phase 2（Phase 2-main）必须提供 ``--phase1-jsonl``，仅跑 phase1 中通过双解读过滤的样本。

默认结果：``output/test1/qwen.jsonl`` / ``output/test2/qwen.jsonl``。

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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ethi_ambrot.common_eval_utils import (
    append_jsonl,
    build_phase2_main_record,
    build_user_content_for_phase,
    configure_shared_eval_args,
    dataset_by_chambi_id,
    default_eval_jsonl_path,
    load_dataset,
    load_done_ids,
    load_env_candidates,
    parse_model_record,
    parse_response_for_phase,
    phase2_cli_error,
)
from ethi_ambrot.eval_prompt import build_prompt_phase2_main
from ethi_ambrot.phase2_main import iter_phase2_main_candidates

_PROVIDER = "qwen"


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
    configure_shared_eval_args(ap)
    args = ap.parse_args()
    err_msg = phase2_cli_error(args.phase, args.phase1_jsonl)
    if err_msg:
        print(err_msg, file=sys.stderr)
        return 1

    if args.output is None:
        args.output = default_eval_jsonl_path(_PROVIDER, args.phase)

    api_key, base_url, model, timeout_sec = _qwen_config()
    client: Any = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    try:
        items = load_dataset(args.dataset)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1

    dataset_idx = dataset_by_chambi_id(items)
    done_ids = load_done_ids(args.output, eval_phase=args.phase)
    model_name = model
    new_count = 0

    if args.phase == 1:
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

            user_content, prep_err = build_user_content_for_phase(1, item, input_text)
            if prep_err:
                rec = parse_model_record(
                    sid, input_text, model_name, "", None, False, prep_err, eval_phase=1
                )
                append_jsonl(args.output, rec)
                new_count += 1
                print(f"[{new_count}] source_chambi_id={sid} phase=1 success=False ({prep_err})", flush=True)
                time.sleep(max(0.0, args.sleep))
                continue

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
                parsed, perr = parse_response_for_phase(1, raw)
                if perr:
                    err = perr
                    ok = False
                else:
                    ok = True
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            rec = parse_model_record(sid, input_text, model_name, raw, parsed, ok, err, eval_phase=1)
            append_jsonl(args.output, rec)
            if ok:
                done_ids.add(sid)
            new_count += 1
            print(f"[{new_count}] source_chambi_id={sid} phase=1 success={ok} (dataset {total})", flush=True)
            time.sleep(max(0.0, args.sleep))
    else:
        candidates = iter_phase2_main_candidates(args.phase1_jsonl)
        total = len(candidates)
        for sid, p1_text, ra, rb in candidates:
            if args.limit is not None and new_count >= args.limit:
                break
            if sid in done_ids:
                continue
            row = dataset_idx.get(sid)
            if row is None:
                print(f"Skip sid={sid} (not in dataset)", file=sys.stderr)
                continue
            input_text = row.get("input_text")
            if not isinstance(input_text, str):
                print(f"Skip sid={sid} (dataset input_text invalid)", file=sys.stderr)
                continue
            if isinstance(p1_text, str) and p1_text.strip() and p1_text.strip() != input_text.strip():
                print(
                    f"warning sid={sid}: phase1 input_text differs from dataset; using dataset",
                    file=sys.stderr,
                )
            user_content = build_prompt_phase2_main(input_text, ra, rb)
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
                parsed, perr = parse_response_for_phase(2, raw)
                if perr:
                    err = perr
                    ok = False
                else:
                    ok = True
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            rec = build_phase2_main_record(
                sid,
                input_text,
                model_name,
                raw,
                parsed,
                ok,
                err,
                reading_a=ra,
                reading_b=rb,
            )
            append_jsonl(args.output, rec)
            if ok:
                done_ids.add(sid)
            new_count += 1
            print(f"[{new_count}] source_chambi_id={sid} phase=2 success={ok} (candidates {total})", flush=True)
            time.sleep(max(0.0, args.sleep))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
