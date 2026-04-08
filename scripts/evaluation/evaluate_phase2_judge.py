#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2-main：LLM-as-judge 离线测评（主分数来自 judge，非语义相似度自动指标）。

环境变量（必填）::
    JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL

可选：JUDGE_TIMEOUT（秒，默认 180）

用法（仓库根）::

    python scripts/evaluation/evaluate_phase2_judge.py \\
        --predictions output/test2/glm.jsonl \\
        --detail-output output/phase2_judge_eval/glm_detail.jsonl \\
        --summary-output output/phase2_judge_eval/glm_summary.json

默认 ``--resume``：跳过 detail 中已成功 judge 的样本；``--no-resume`` 对队列内样本一律再次调用 judge（会重复追加；summary 按 sid 取最后一行）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ethi_ambrot.common_eval_utils import append_jsonl, load_dataset, load_env_candidates
from ethi_ambrot.judge_prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_message_both_readings,
    parse_judge_response_dual,
)
from ethi_ambrot.phase2_main import (
    is_valid_phase2_eval_row,
    normalize_dimension,
)


_DEFAULT_GOLD = REPO_ROOT / "data" / "ethi_ambrot_benchmark_compact.json"
_DEFAULT_OUT_DIR = REPO_ROOT / "output" / "phase2_judge_eval"


def _coerce_sid(x: Any) -> int:
    if isinstance(x, bool):
        raise TypeError("source_ethi_ambrot_id cannot be bool")
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str) and x.strip():
        return int(x.strip(), 10)
    return int(x)


def load_predictions_last(path: Path) -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict) or o.get("source_ethi_ambrot_id") is None:
                continue
            try:
                cid = _coerce_sid(o["source_ethi_ambrot_id"])
            except (TypeError, ValueError):
                continue
            by_id[cid] = o
    return by_id


def dataset_by_id(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in items:
        sid = row.get("source_ethi_ambrot_id")
        if sid is None:
            continue
        try:
            k = _coerce_sid(sid)
        except (TypeError, ValueError):
            continue
        out[k] = row
    return out


def readings_by_id(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    readings = row.get("readings")
    if not isinstance(readings, list):
        return None, None
    a, b = None, None
    for rd in readings:
        if not isinstance(rd, dict):
            continue
        rid = rd.get("reading_id")
        if rid == "A":
            a = rd
        elif rid == "B":
            b = rd
    return a, b


def load_completed_judge_sids(detail_path: Path) -> set[int]:
    """只看每个 sid 最后一行：成功且 judge 双分完整则视为已完成。"""
    last_by_sid = reload_all_details(detail_path)
    done: set[int] = set()
    for o in last_by_sid:
        if o.get("success") is not True:
            continue
        je = o.get("judge_eval")
        if not isinstance(je, dict):
            continue
        ja, jb = je.get("reading_a"), je.get("reading_b")
        if not isinstance(ja, dict) or not isinstance(jb, dict):
            continue
        if ja.get("score_rot") not in (1, 2, 3) or jb.get("score_rot") not in (1, 2, 3):
            continue
        try:
            done.add(_coerce_sid(o["source_ethi_ambrot_id"]))
        except (TypeError, ValueError):
            continue
    return done


def _norm_dim_compare(s: str | None) -> str | None:
    """比较用：优先四分类枚举，否则 NFKC strip lower。"""
    d = normalize_dimension(str(s) if s is not None else "")
    if d is not None:
        return d
    t = unicodedata.normalize("NFKC", (str(s) if s is not None else "").strip()).lower()
    return t if t else None


def primary_dimension_match(gold_pri: Any, pred_pri: Any) -> bool:
    g = _norm_dim_compare(str(gold_pri) if gold_pri is not None else "")
    p = _norm_dim_compare(str(pred_pri) if pred_pri is not None else "")
    if g is None and p is None:
        return True
    if g is None or p is None:
        return False
    return g == p


def value_in_predicted_set(gold_pri: Any, pred_pri: Any, pred_sec: Any) -> bool:
    g = _norm_dim_compare(str(gold_pri) if gold_pri is not None else "")
    if g is None:
        return False
    for x in (pred_pri, pred_sec):
        if _norm_dim_compare(str(x) if x is not None else "") == g:
            return True
    return False


def _pred_side_copy(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "norm_activation": str(d.get("norm_activation") or ""),
        "ethical_obligation": str(d.get("ethical_obligation") or ""),
        "prescriptive_advice": str(d.get("prescriptive_advice") or ""),
        "primary_dimension": d.get("primary_dimension"),
        "secondary_dimension": d.get("secondary_dimension"),
        "value_reason": str(d.get("value_reason") or ""),
    }


def _gold_pack(reading: dict[str, Any]) -> dict[str, Any]:
    return {
        "paraphrase": reading.get("paraphrase", "") if isinstance(reading.get("paraphrase"), str) else "",
        "gold_rot": reading.get("gold_rot") if isinstance(reading.get("gold_rot"), dict) else {},
        "gold_value_alignment": reading.get("gold_value_alignment")
        if isinstance(reading.get("gold_value_alignment"), dict)
        else {},
    }


def judge_config() -> tuple[str, str, str, float]:
    api_key = (os.environ.get("JUDGE_API_KEY") or "").strip()
    if not api_key:
        print("Missing JUDGE_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)
    base_url = (os.environ.get("JUDGE_BASE_URL") or "").strip().rstrip("/")
    model = (os.environ.get("JUDGE_MODEL") or "").strip()
    if not base_url:
        print("Missing JUDGE_BASE_URL in environment.", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("Missing JUDGE_MODEL in environment.", file=sys.stderr)
        sys.exit(1)
    # 常见误配：把模型名写进 BASE_URL、把 API 根 URL 写进 MODEL
    if "://" in model or model.startswith("http"):
        print(
            "error: JUDGE_MODEL looks like a URL. Model id should be e.g. glm-4-air; "
            "put the OpenAI-compatible base URL in JUDGE_BASE_URL (must start with https://).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not base_url.startswith(("http://", "https://")):
        print(
            "error: JUDGE_BASE_URL must include the protocol, e.g. https://open.bigmodel.cn/api/paas/v4 "
            f"(got {base_url!r}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")].rstrip("/")
    raw_t = os.environ.get("JUDGE_TIMEOUT")
    try:
        timeout = float(raw_t) if raw_t else 180.0
    except ValueError:
        timeout = 180.0
    return api_key, base_url, model, timeout


def _reading_text(reading: dict[str, Any] | None) -> str:
    if not isinstance(reading, dict):
        return ""
    p = reading.get("paraphrase")
    return p.strip() if isinstance(p, str) else ""


def _normalize_reading_text(s: str | None) -> str:
    t = unicodedata.normalize("NFKC", str(s or "")).strip().lower()
    t = " ".join(t.split())
    return t


def _reading_similarity(a: str | None, b: str | None) -> float:
    na = _normalize_reading_text(a)
    nb = _normalize_reading_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return 1.0 if na == nb else 0.0


def align_phase2_sides(
    *,
    pred_reading_a_text: str,
    pred_reading_b_text: str,
    pred_a: dict[str, Any],
    pred_b: dict[str, Any],
    gold_a_obj: dict[str, Any],
    gold_b_obj: dict[str, Any],
) -> dict[str, Any]:
    gold_a_text = _reading_text(gold_a_obj)
    gold_b_text = _reading_text(gold_b_obj)

    score_keep = _reading_similarity(pred_reading_a_text, gold_a_text) + _reading_similarity(pred_reading_b_text, gold_b_text)
    score_swap = _reading_similarity(pred_reading_a_text, gold_b_text) + _reading_similarity(pred_reading_b_text, gold_a_text)

    if score_keep >= score_swap:
        return {
            "assignment": "predA→goldA, predB→goldB",
            "assignment_score": score_keep,
            "reading_a_input": pred_reading_a_text,
            "reading_b_input": pred_reading_b_text,
            "gold_pack_a": _gold_pack(gold_a_obj),
            "gold_pack_b": _gold_pack(gold_b_obj),
            "pred_a": pred_a,
            "pred_b": pred_b,
        }

    return {
        "assignment": "predA→goldB, predB→goldA",
        "assignment_score": score_swap,
        "reading_a_input": pred_reading_a_text,
        "reading_b_input": pred_reading_b_text,
        "gold_pack_a": _gold_pack(gold_b_obj),
        "gold_pack_b": _gold_pack(gold_a_obj),
        "pred_a": pred_a,
        "pred_b": pred_b,
    }


def build_detail_record(
    *,
    sid: int,
    input_text: str,
    model_name: str,
    judge_model: str,
    prediction_file: str,
    ra_in: str,
    rb_in: str,
    gold_pack_a: dict[str, Any],
    gold_pack_b: dict[str, Any],
    pred_a: dict[str, Any],
    pred_b: dict[str, Any],
    judge_eval: dict[str, Any] | None,
    item_score: float | None,
    success: bool,
    error: str | None,
    assignment: str | None,
    assignment_score: float | None,
) -> dict[str, Any]:
    return {
        "source_ethi_ambrot_id": sid,
        "input_text": input_text,
        "model_name": model_name,
        "judge_model": judge_model,
        "prediction_file": prediction_file,
        "reading_a_input": ra_in,
        "reading_b_input": rb_in,
        "assignment": assignment,
        "assignment_score": assignment_score,
        "gold": {
            "reading_a": {
                "paraphrase": gold_pack_a.get("paraphrase", ""),
                "gold_rot": gold_pack_a.get("gold_rot", {}),
                "gold_value_alignment": gold_pack_a.get("gold_value_alignment", {}),
            },
            "reading_b": {
                "paraphrase": gold_pack_b.get("paraphrase", ""),
                "gold_rot": gold_pack_b.get("gold_rot", {}),
                "gold_value_alignment": gold_pack_b.get("gold_value_alignment", {}),
            },
        },
        "prediction": {"reading_a": pred_a, "reading_b": pred_b},
        "judge_eval": judge_eval,
        "item_score": item_score,
        "success": success,
        "error": error,
    }


def summarize_from_details(
    detail_rows: list[dict[str, Any]],
    *,
    gold_path: Path,
    predictions_path: Path,
    detail_output: Path,
    summary_output: Path,
    judge_model: str,
    judge_base_url: str,
    num_gold_items: int,
    num_prediction_items: int,
    num_matched_items: int,
    num_phase2_items: int,
    phase2_eligible_sids: set[int],
) -> dict[str, Any]:
    predictions_path_str = str(predictions_path.resolve())
    scoped_rows = [
        r for r in detail_rows
        if str(r.get("prediction_file") or "") == predictions_path_str
        and str(r.get("judge_model") or "") == judge_model
    ]

    judged = [r for r in scoped_rows if r.get("success") is True and isinstance(r.get("judge_eval"), dict)]
    je_list = []
    for r in judged:
        try:
            sid = _coerce_sid(r["source_ethi_ambrot_id"])
        except (TypeError, ValueError, KeyError):
            continue
        if sid in phase2_eligible_sids:
            je_list.append(r)
    num_judged_items = len(je_list)

    parse_success_rate = (
        num_phase2_items / max(num_matched_items, 1) if num_matched_items else 0.0
    )
    judge_success_rate = num_judged_items / max(num_phase2_items, 1) if num_phase2_items else 0.0

    def dist_for(side: str) -> dict[str, int]:
        d = {"1": 0, "2": 0, "3": 0}
        for r in je_list:
            je = r["judge_eval"]
            if not isinstance(je, dict):
                continue
            block = je.get(side)
            if not isinstance(block, dict):
                continue
            sr = block.get("score_rot")
            if sr in (1, 2, 3):
                d[str(int(sr))] += 1
        return d

    dist_a = dist_for("reading_a")
    dist_b = dist_for("reading_b")
    dist_overall = {"1": 0, "2": 0, "3": 0}
    for r in je_list:
        je = r["judge_eval"]
        if not isinstance(je, dict):
            continue
        for side in ("reading_a", "reading_b"):
            block = je.get(side)
            if not isinstance(block, dict):
                continue
            sr = block.get("score_rot")
            if sr in (1, 2, 3):
                dist_overall[str(int(sr))] += 1

    scores_a: list[float] = []
    scores_b: list[float] = []
    item_scores: list[float] = []
    for r in je_list:
        je = r["judge_eval"]
        if not isinstance(je, dict):
            continue
        sa = je.get("reading_a", {})
        sb = je.get("reading_b", {})
        if isinstance(sa, dict) and sa.get("score_rot") in (1, 2, 3):
            scores_a.append(float(sa["score_rot"]))
        if isinstance(sb, dict) and sb.get("score_rot") in (1, 2, 3):
            scores_b.append(float(sb["score_rot"]))
        if r.get("item_score") is not None:
            item_scores.append(float(r["item_score"]))

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    avg_a = avg(scores_a)
    avg_b = avg(scores_b)
    avg_overall_reading = (
        (sum(scores_a) + sum(scores_b)) / (len(scores_a) + len(scores_b))
        if (scores_a or scores_b)
        else 0.0
    )
    avg_item_score = avg(item_scores)

    # supplementary: judge boolean submetrics + primary/value-set automatic checks
    norm_match_a = norm_match_b = 0
    obligation_match_a = obligation_match_b = 0
    advice_match_a = advice_match_b = 0
    value_match_a = value_match_b = 0
    p_corr_a = p_corr_b = 0
    v_hit_a = v_hit_b = 0
    for r in je_list:
        je = r.get("judge_eval")
        if isinstance(je, dict):
            ja = je.get("reading_a")
            jb = je.get("reading_b")
            if isinstance(ja, dict):
                norm_match_a += 1 if ja.get("norm_match") is True else 0
                obligation_match_a += 1 if ja.get("obligation_match") is True else 0
                advice_match_a += 1 if ja.get("advice_match") is True else 0
                value_match_a += 1 if ja.get("value_match") is True else 0
            if isinstance(jb, dict):
                norm_match_b += 1 if jb.get("norm_match") is True else 0
                obligation_match_b += 1 if jb.get("obligation_match") is True else 0
                advice_match_b += 1 if jb.get("advice_match") is True else 0
                value_match_b += 1 if jb.get("value_match") is True else 0

        gold = r.get("gold")
        pred = r.get("prediction")
        if not isinstance(gold, dict) or not isinstance(pred, dict):
            continue
        ga, gb = gold.get("reading_a"), gold.get("reading_b")
        pa, pb = pred.get("reading_a"), pred.get("reading_b")
        if isinstance(ga, dict) and isinstance(pa, dict):
            gva = ga.get("gold_value_alignment") if isinstance(ga.get("gold_value_alignment"), dict) else {}
            gp = gva.get("primary_dimension")
            pp = pa.get("primary_dimension")
            if primary_dimension_match(gp, pp):
                p_corr_a += 1
            if value_in_predicted_set(gp, pa.get("primary_dimension"), pa.get("secondary_dimension")):
                v_hit_a += 1
        if isinstance(gb, dict) and isinstance(pb, dict):
            gvb = gb.get("gold_value_alignment") if isinstance(gb.get("gold_value_alignment"), dict) else {}
            gp = gvb.get("primary_dimension")
            pp = pb.get("primary_dimension")
            if primary_dimension_match(gp, pp):
                p_corr_b += 1
            if value_in_predicted_set(gp, pb.get("primary_dimension"), pb.get("secondary_dimension")):
                v_hit_b += 1

    n_sup = len(je_list)
    norm_match_accuracy_a = norm_match_a / max(n_sup, 1)
    norm_match_accuracy_b = norm_match_b / max(n_sup, 1)
    norm_match_accuracy_overall = (norm_match_a + norm_match_b) / max(2 * n_sup, 1)
    obligation_match_accuracy_a = obligation_match_a / max(n_sup, 1)
    obligation_match_accuracy_b = obligation_match_b / max(n_sup, 1)
    obligation_match_accuracy_overall = (obligation_match_a + obligation_match_b) / max(2 * n_sup, 1)
    advice_match_accuracy_a = advice_match_a / max(n_sup, 1)
    advice_match_accuracy_b = advice_match_b / max(n_sup, 1)
    advice_match_accuracy_overall = (advice_match_a + advice_match_b) / max(2 * n_sup, 1)
    value_match_accuracy_a = value_match_a / max(n_sup, 1)
    value_match_accuracy_b = value_match_b / max(n_sup, 1)
    value_match_accuracy_overall = (value_match_a + value_match_b) / max(2 * n_sup, 1)
    primary_dimension_accuracy_a = p_corr_a / max(n_sup, 1)
    primary_dimension_accuracy_b = p_corr_b / max(n_sup, 1)
    primary_dimension_accuracy_overall = (p_corr_a + p_corr_b) / max(2 * n_sup, 1)
    value_in_predicted_set_accuracy_a = v_hit_a / max(n_sup, 1)
    value_in_predicted_set_accuracy_b = v_hit_b / max(n_sup, 1)
    value_in_predicted_set_accuracy_overall = (v_hit_a + v_hit_b) / max(2 * n_sup, 1)

    ts = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "meta": {
            "evaluator": "phase2_judge_eval",
            "primary_metrics_are_judge_based": True,
            "supplementary_note": "metrics_supplementary 为字符串层自动指标，非 Phase 2 主结论。",
            "generated_at_utc": ts,
            "gold_path": str(gold_path.resolve()),
            "predictions_path": str(predictions_path.resolve()),
            "judge_model": judge_model,
            "judge_base_url": judge_base_url,
            "detail_output": str(detail_output.resolve()),
            "summary_output": str(summary_output.resolve()),
        },
        "metrics": {
            "num_gold_items": num_gold_items,
            "num_prediction_items": num_prediction_items,
            "num_matched_items": num_matched_items,
            "num_phase2_items": num_phase2_items,
            "num_judged_items": num_judged_items,
            "parse_success_rate": parse_success_rate,
            "judge_success_rate": judge_success_rate,
            "avg_score_rot_a": avg_a,
            "avg_score_rot_b": avg_b,
            "avg_score_rot_overall": avg_overall_reading,
            "avg_item_score": avg_item_score,
            "score_rot_distribution_a": dist_a,
            "score_rot_distribution_b": dist_b,
            "score_rot_distribution_overall": dist_overall,
        },
        "metrics_supplementary": {
            "description": "非主分数：judge 布尔子项一致率 + 规范化后的 primary 匹配与 gold primary 是否落在 pred 维集合。",
            "norm_match_accuracy_a": norm_match_accuracy_a,
            "norm_match_accuracy_b": norm_match_accuracy_b,
            "norm_match_accuracy_overall": norm_match_accuracy_overall,
            "obligation_match_accuracy_a": obligation_match_accuracy_a,
            "obligation_match_accuracy_b": obligation_match_accuracy_b,
            "obligation_match_accuracy_overall": obligation_match_accuracy_overall,
            "advice_match_accuracy_a": advice_match_accuracy_a,
            "advice_match_accuracy_b": advice_match_accuracy_b,
            "advice_match_accuracy_overall": advice_match_accuracy_overall,
            "value_match_accuracy_a": value_match_accuracy_a,
            "value_match_accuracy_b": value_match_accuracy_b,
            "value_match_accuracy_overall": value_match_accuracy_overall,
            "primary_dimension_accuracy_a": primary_dimension_accuracy_a,
            "primary_dimension_accuracy_b": primary_dimension_accuracy_b,
            "primary_dimension_accuracy_overall": primary_dimension_accuracy_overall,
            "value_in_predicted_set_accuracy_a": value_in_predicted_set_accuracy_a,
            "value_in_predicted_set_accuracy_b": value_in_predicted_set_accuracy_b,
            "value_in_predicted_set_accuracy_overall": value_in_predicted_set_accuracy_overall,
        },
    }


def reload_all_details(path: Path) -> list[dict[str, Any]]:
    """按文件顺序读入，同一 source_ethi_ambrot_id 只保留最后一次（与续跑追加一致）。"""
    by_sid: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict) or o.get("source_ethi_ambrot_id") is None:
                continue
            try:
                k = _coerce_sid(o["source_ethi_ambrot_id"])
            except (TypeError, ValueError):
                continue
            by_sid[k] = o
    return [by_sid[k] for k in sorted(by_sid.keys())]


def main() -> int:
    try:
        from openai import OpenAI
    except ImportError:
        print("Install with: pip install openai", file=sys.stderr)
        return 1

    load_env_candidates(REPO_ROOT)

    ap = argparse.ArgumentParser(description="Phase 2 judge evaluation (LLM-as-judge)")
    ap.add_argument("--gold", type=Path, default=_DEFAULT_GOLD, help="Benchmark JSON")
    ap.add_argument("--predictions", "-p", type=Path, required=True, help="Phase 2 JSONL")
    ap.add_argument("--detail-output", type=Path, default=None, help="Per-item judge JSONL")
    ap.add_argument("--summary-output", type=Path, default=None, help="Summary JSON")
    ap.add_argument("--limit", type=int, default=None, help="Max new judge calls (after resume skip)")
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default true: skip ids already successfully judged in detail-output; --no-resume to disable",
    )
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds after each judge API call")
    args = ap.parse_args()

    out_dir = _DEFAULT_OUT_DIR
    detail_out = args.detail_output or (out_dir / "phase2_judge_detail.jsonl")
    summary_out = args.summary_output or (out_dir / "phase2_judge_summary.json")

    if not args.predictions.is_file():
        print(f"Not a file: {args.predictions}", file=sys.stderr)
        return 1
    if not args.gold.is_file():
        print(f"Not a file: {args.gold}", file=sys.stderr)
        return 1

    api_key, base_url, judge_model, timeout_sec = judge_config()
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    try:
        items = load_dataset(args.gold)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1

    ds = dataset_by_id(items)
    pred_map = load_predictions_last(args.predictions)
    prediction_file_str = str(args.predictions.resolve())

    done_sids: set[int] = set()
    if args.resume:
        done_sids = load_completed_judge_sids(detail_out)

    num_gold_items = len(items)
    num_prediction_items = len(pred_map)
    num_matched = sum(1 for sid in ds if sid in pred_map)

    queue: list[tuple[int, dict[str, Any]]] = []
    for sid in sorted(ds.keys()):
        pr = pred_map.get(sid)
        if pr is None:
            continue
        if not is_valid_phase2_eval_row(pr):
            continue
        queue.append((sid, pr))

    num_phase2_items = len(queue)
    phase2_eligible_sids = {sid for sid, _ in queue}
    new_calls = 0

    for sid, pr in queue:
        if args.limit is not None and new_calls >= args.limit:
            break
        if args.resume and sid in done_sids:
            continue

        gold_row = ds[sid]
        ra_obj, rb_obj = readings_by_id(gold_row)
        if ra_obj is None or rb_obj is None:
            rec = build_detail_record(
                sid=sid,
                input_text=str(pr.get("input_text") or gold_row.get("input_text") or ""),
                model_name=str(pr.get("model_name") or ""),
                judge_model=judge_model,
                prediction_file=prediction_file_str,
                ra_in=str(pr.get("reading_a") or ""),
                rb_in=str(pr.get("reading_b") or ""),
                gold_pack_a={},
                gold_pack_b={},
                pred_a={},
                pred_b={},
                judge_eval=None,
                item_score=None,
                success=False,
                error="missing_gold_readings_AB",
                assignment=None,
                assignment_score=None,
            )
            append_jsonl(detail_out, rec)
            continue

        input_text = gold_row.get("input_text")
        if not isinstance(input_text, str):
            input_text = str(pr.get("input_text") or "")

        parsed = pr.get("parsed_response")
        assert isinstance(parsed, dict)
        pa_raw = parsed["reading_a"]
        pb_raw = parsed["reading_b"]
        pred_a = _pred_side_copy(pa_raw)
        pred_b = _pred_side_copy(pb_raw)

        ra_in = str(pr.get("reading_a") or "")
        rb_in = str(pr.get("reading_b") or "")

        aligned = align_phase2_sides(
            pred_reading_a_text=ra_in,
            pred_reading_b_text=rb_in,
            pred_a=pred_a,
            pred_b=pred_b,
            gold_a_obj=ra_obj,
            gold_b_obj=rb_obj,
        )
        pack_a = aligned["gold_pack_a"]
        pack_b = aligned["gold_pack_b"]

        user_msg = build_judge_user_message_both_readings(
            input_text=input_text,
            reading_a_text=aligned["reading_a_input"],
            reading_b_text=aligned["reading_b_input"],
            gold_a=pack_a,
            gold_b=pack_b,
            pred_a=aligned["pred_a"],
            pred_b=aligned["pred_b"],
        )

        raw_judge = ""
        judge_eval: dict[str, Any] | None = None
        err: str | None = None
        ok = False
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            raw_judge = (resp.choices[0].message.content or "").strip()
            judge_eval = parse_judge_response_dual(raw_judge)
            if judge_eval is None:
                err = "judge_parse_failed"
            else:
                ok = True
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        item_score: float | None = None
        if ok and judge_eval:
            sa = judge_eval["reading_a"]["score_rot"]
            sb = judge_eval["reading_b"]["score_rot"]
            item_score = (float(sa) + float(sb)) / 2.0

        rec = build_detail_record(
            sid=sid,
            input_text=input_text,
            model_name=str(pr.get("model_name") or ""),
            judge_model=judge_model,
            prediction_file=prediction_file_str,
            ra_in=aligned["reading_a_input"],
            rb_in=aligned["reading_b_input"],
            gold_pack_a=pack_a,
            gold_pack_b=pack_b,
            pred_a=aligned["pred_a"],
            pred_b=aligned["pred_b"],
            judge_eval=judge_eval,
            item_score=item_score,
            success=ok,
            error=err,
            assignment=aligned["assignment"],
            assignment_score=float(aligned["assignment_score"]),
        )
        append_jsonl(detail_out, rec)
        new_calls += 1
        status = "ok" if ok else (err or "fail")
        print(f"[{new_calls}] source_ethi_ambrot_id={sid} judge={status}", flush=True)
        time.sleep(max(0.0, args.sleep))

    # Summary from full detail file
    all_detail = reload_all_details(detail_out)
    summary_doc = summarize_from_details(
        all_detail,
        gold_path=args.gold,
        predictions_path=args.predictions,
        detail_output=detail_out,
        summary_output=summary_out,
        judge_model=judge_model,
        judge_base_url=base_url,
        num_gold_items=num_gold_items,
        num_prediction_items=num_prediction_items,
        num_matched_items=num_matched,
        num_phase2_items=num_phase2_items,
        phase2_eligible_sids=phase2_eligible_sids,
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        json.dumps(summary_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
