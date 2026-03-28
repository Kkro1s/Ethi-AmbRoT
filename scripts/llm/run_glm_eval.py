#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    python scripts/llm/run_glm_eval.py [--phase 1|2] [--dataset ...] [--output ...]

默认：``output/test1/glm.jsonl`` / ``output/test2/glm.jsonl``

使用 **OpenAI 兼容** 客户端连接智谱（与官方示例同一 REST 前缀），免费模型示例见：
https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash

环境变量：

- ``GLM_API_KEY``（必填）：控制台 API Key
- ``GLM_BASE_URL``（可选）：默认 ``https://open.bigmodel.cn/api/paas/v4``
- ``GLM_MODEL``（可选）：未设置时默认 ``glm-4.7-flash``（与上文档一致；可按控制台更换）
- ``GLM_THINKING``：设为 ``1`` / ``true`` / ``enabled`` 时在请求体中加入
  ``thinking: {\"type\": \"enabled\"}``（长推理时可能更易超时；JSON 任务可先不设）
- ``GLM_MAX_TOKENS``：可选，整数，传给 ``max_tokens``（文档示例可用 65536；评测 JSON 一般较小）
- ``GLM_TIMEOUT``：秒，默认 180
- ``GLM_RATE_LIMIT_RETRIES``：遇 429 时最多**连续请求**次数，默认 ``10``（含首次）
- ``GLM_RATE_LIMIT_BASE_WAIT``：首次 429 后等待秒数，默认 ``30``，之后指数退避（×2）；若响应带 ``Retry-After`` 会与之取较大值
- ``GLM_COOLDOWN_STARTUP_SEC``：在**第一次**调用 API 前先 sleep（秒）；刚遭遇 1302 时可设 ``60``～``180`` 再跑
- ``GLM_BACKOFF_JITTER``：设为 ``1`` 时在等待时间上增加至多 20% 随机抖动，避免固定节拍撞上窗口
- ``GLM_MIN_REQUEST_INTERVAL``：每条请求后的**最短间隔**（秒），与命令行 ``--sleep`` **取较大值**，
  用于缓解官方 **1302 用户速率限制**（见控制台说明）。示例：``.env`` 里写 ``GLM_MIN_REQUEST_INTERVAL=5``。

单进程评测本身**并发为 1**；请勿同时开多个 ``run_glm_eval`` 或与 IDE/其它工具**共用同一 Key 狂刷**，否则仍易 429。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ethi_ambrot.common_eval_utils import (
    append_jsonl,
    build_user_content_for_phase,
    configure_shared_eval_args,
    default_eval_jsonl_path,
    load_dataset,
    load_done_ids,
    load_env_candidates,
    parse_model_record,
    parse_response_for_phase,
)

_PROVIDER = "glm"

# 与智谱开放文档中 GLM-4.7-Flash 的 model 字段一致；可通过 GLM_MODEL 覆盖
_DEFAULT_GLM_MODEL = "glm-4.7-flash"


def _glm_config() -> tuple[str, str, str, float]:
    api_key = (os.environ.get("GLM_API_KEY") or "").strip()
    if not api_key:
        print("Missing GLM_API_KEY.", file=sys.stderr)
        sys.exit(1)
    base_url = (
        os.environ.get("GLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
    ).strip()
    base_url = base_url.rstrip("/")
    # OpenAI SDK 会自行拼接 /chat/completions；若 .env 误写完整路径则去掉尾部
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")].rstrip("/")
    model = (os.environ.get("GLM_MODEL") or _DEFAULT_GLM_MODEL).strip()
    timeout_raw = os.environ.get("GLM_TIMEOUT")
    try:
        timeout_sec = float(timeout_raw) if timeout_raw else 180.0
    except ValueError:
        timeout_sec = 180.0
    return api_key, base_url, model, timeout_sec


def _thinking_extra_body() -> dict[str, Any] | None:
    flag = (os.environ.get("GLM_THINKING") or "").strip().lower()
    if flag in ("1", "true", "yes", "enabled"):
        return {"thinking": {"type": "enabled"}}
    return None


def _max_tokens_param() -> dict[str, int]:
    raw = (os.environ.get("GLM_MAX_TOKENS") or "").strip()
    if not raw:
        return {}
    try:
        return {"max_tokens": int(raw)}
    except ValueError:
        return {}


def _message_text(msg: Any) -> str:
    """ Prefer content; some thinking-mode replies expose reasoning in an extra field. """
    text = (getattr(msg, "content", None) or "").strip()
    if text:
        return text
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is not None:
        return str(reasoning).strip()
    return ""


def _rate_limit_retry_config() -> tuple[int, float]:
    try:
        retries = int((os.environ.get("GLM_RATE_LIMIT_RETRIES") or "10").strip())
    except ValueError:
        retries = 10
    retries = max(1, retries)
    try:
        base = float((os.environ.get("GLM_RATE_LIMIT_BASE_WAIT") or "30").strip())
    except ValueError:
        base = 30.0
    base = max(1.0, base)
    return retries, base


def _apply_backoff_jitter(wait: float) -> float:
    if (os.environ.get("GLM_BACKOFF_JITTER") or "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return wait
    return wait * (1.0 + random.random() * 0.20)


def _retry_after_seconds(err: Exception) -> float | None:
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    raw = None
    try:
        raw = resp.headers.get("retry-after")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _startup_cooldown() -> None:
    raw = (os.environ.get("GLM_COOLDOWN_STARTUP_SEC") or "").strip()
    if not raw:
        return
    try:
        sec = float(raw)
    except ValueError:
        print(
            f"Ignore invalid GLM_COOLDOWN_STARTUP_SEC={raw!r}",
            file=sys.stderr,
        )
        return
    if sec <= 0:
        return
    print(
        f"GLM_COOLDOWN_STARTUP_SEC={sec:.0f}s: wait before first API call…",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(sec)


def _chat_create_with_glm_retry(
    client: Any,
    create_kw: dict[str, Any],
) -> Any:
    """Retry on Zhipu/OpenAI-compatible 429 RateLimitError with exponential backoff."""
    from openai import RateLimitError

    retries, base_wait = _rate_limit_retry_config()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return client.chat.completions.create(**create_kw)
        except RateLimitError as e:
            last_err = e
            if attempt >= retries - 1:
                raise
            wait = base_wait * (2**attempt)
            ra = _retry_after_seconds(e)
            if ra is not None:
                wait = max(wait, ra)
            wait = _apply_backoff_jitter(wait)
            print(
                f"429 限流 (1302): 第 {attempt + 1}/{retries} 次请求失败，"
                f"等待 {wait:.1f}s 后重试…",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def main() -> int:
    try:
        from openai import OpenAI
    except ImportError:
        print("Install with: pip install openai", file=sys.stderr)
        return 1

    load_env_candidates(REPO_ROOT)

    ap = argparse.ArgumentParser(description="Run GLM (Zhipu OpenAI-compatible) on Chambi benchmark compact")
    configure_shared_eval_args(ap)
    args = ap.parse_args()
    if args.output is None:
        args.output = default_eval_jsonl_path(_PROVIDER, args.phase)

    raw_interval = (os.environ.get("GLM_MIN_REQUEST_INTERVAL") or "").strip()
    if raw_interval:
        try:
            floor_s = float(raw_interval)
            if floor_s >= 0:
                args.sleep = max(args.sleep, floor_s)
        except ValueError:
            print(
                f"Ignore invalid GLM_MIN_REQUEST_INTERVAL={raw_interval!r}",
                file=sys.stderr,
            )

    api_key, base_url, model, timeout_sec = _glm_config()
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    _startup_cooldown()

    try:
        items = load_dataset(args.dataset)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1

    done_ids = load_done_ids(args.output, eval_phase=args.phase)
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

        user_content, prep_err = build_user_content_for_phase(args.phase, item, input_text)
        if prep_err:
            rec = parse_model_record(
                sid,
                input_text,
                model_name,
                "",
                None,
                False,
                prep_err,
                eval_phase=args.phase,
            )
            append_jsonl(args.output, rec)
            new_count += 1
            print(
                f"[{new_count}] source_chambi_id={sid} phase={args.phase} success=False ({prep_err})",
                flush=True,
            )
            time.sleep(max(0.0, args.sleep))
            continue

        raw = ""
        parsed = None
        ok = False
        err: str | None = None
        try:
            create_kw: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": user_content}],
                "temperature": 0.2,
            }
            create_kw.update(_max_tokens_param())
            extra = _thinking_extra_body()
            if extra is not None:
                create_kw["extra_body"] = extra
            resp = _chat_create_with_glm_retry(client, create_kw)
            raw = _message_text(resp.choices[0].message)
            parsed, perr = parse_response_for_phase(args.phase, raw)
            if perr:
                err = perr
                ok = False
            else:
                ok = True
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        rec = parse_model_record(
            sid, input_text, model_name, raw, parsed, ok, err, eval_phase=args.phase
        )
        append_jsonl(args.output, rec)
        if ok:
            done_ids.add(sid)
        new_count += 1
        print(
            f"[{new_count}] source_chambi_id={sid} phase={args.phase} success={ok} (dataset size {total})",
            flush=True,
        )
        time.sleep(max(0.0, args.sleep))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
