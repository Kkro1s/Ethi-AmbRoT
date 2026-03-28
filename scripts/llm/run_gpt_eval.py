#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    python scripts/llm/run_gpt_eval.py [--phase 1|2] [--phase1-jsonl ...] [--dataset ...]

phase 2 须提供 ``--phase1-jsonl``。默认：``output/test1/gpt.jsonl`` / ``output/test2/gpt.jsonl``

Env: OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1), OPENAI_MODEL.
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

_PROVIDER = "gpt"


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

    load_env_candidates(REPO_ROOT)

    ap = argparse.ArgumentParser(description="Run OpenAI-compatible GPT on Chambi benchmark compact")
    configure_shared_eval_args(ap)
    args = ap.parse_args()
    err_msg = phase2_cli_error(args.phase, args.phase1_jsonl)
    if err_msg:
        print(err_msg, file=sys.stderr)
        return 1

    if args.output is None:
        args.output = default_eval_jsonl_path(_PROVIDER, args.phase)

    api_key, base_url, model, timeout_sec = _openai_config()
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
                print(f"Skip malformed row: {item!r}", file=sys.stderr)
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
                sid, input_text, model_name, raw, parsed, ok, err, reading_a=ra, reading_b=rb
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
