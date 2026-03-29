#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 解读恢复评估：对比 gold benchmark 与 phase1 JSONL 预测。

用法（仓库根）::

    python scripts/evaluation/evaluate_phase1.py \\
        --predictions output/test1/glm.jsonl

默认将结构化结果写入 ``output/phase1_eval/``（summary JSON + details JSONL）。
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_GOLD = REPO_ROOT / "data" / "Chambi_benchmark_compact.json"
_DEFAULT_PHASE1_EVAL_DIR = REPO_ROOT / "output" / "phase1_eval"

# 与 parse_test1_response 中「无」处理对齐，并含常见占位
_PLACEHOLDER_RAW: frozenset[str] = frozenset(
    {
        "",
        "无",
        "无。",
        "none",
        "null",
        "n/a",
        "n/a.",
        "n.a.",
        "na",
        "不适用",
        "—",
        "-",
        "－",
    }
)

_EXTRA_PUNCT = "。，、；：？！「」『』【】《》〈〉…—·～・""''（）＂＇"


def coerce_chambi_id(x: Any) -> int:
    if isinstance(x, bool):
        raise TypeError("source_chambi_id cannot be bool")
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str) and x.strip():
        return int(x.strip(), 10)
    return int(x)


def load_gold_dataset(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("gold JSON must be a list of items")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"gold item {i} must be an object")
        if item.get("source_chambi_id") is None:
            raise ValueError(f"gold item {i} missing source_chambi_id")
        row = dict(item)
        row["source_chambi_id"] = coerce_chambi_id(row["source_chambi_id"])
        out.append(row)
    return out


def load_predictions(path: Path) -> dict[int, dict[str, Any]]:
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
            if not isinstance(o, dict) or o.get("source_chambi_id") is None:
                continue
            try:
                cid = coerce_chambi_id(o["source_chambi_id"])
            except (TypeError, ValueError):
                continue
            by_id[cid] = o
    return by_id


def normalize_text(s: str | None) -> str:
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).strip()
    if not t:
        return ""
    # 小写化（中文不受影响）
    t = t.lower()
    remove = set(string.punctuation) | set(_EXTRA_PUNCT)
    t = "".join(ch for ch in t if ch not in remove)
    t = re.sub(r"\s+", "", t)
    return t


def normalize_for_distinctness(s: str | None) -> str:
    """
    判断两条解读是否“实质为同一句”时用：保留全部标点，避免因去标点把不同解读合并。
    NFKC + 首尾 strip + 空白压成单空格 + 小写。
    """
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    return t.lower()


def text_similarity(a: str | None, b: str | None) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _is_placeholder_token(t: str) -> bool:
    u = t.strip().lower()
    if u in _PLACEHOLDER_RAW:
        return True
    u2 = u.rstrip("。. ")
    return u2 in _PLACEHOLDER_RAW or u2 in {"none", "null"}


def is_valid_reading(s: str | None) -> bool:
    if s is None:
        return False
    if not isinstance(s, str):
        s = str(s)
    raw = s.strip()
    if not raw:
        return False
    if _is_placeholder_token(raw):
        return False
    if not normalize_text(raw):
        return False
    return True


def extract_gold_paraphrases(item: dict[str, Any]) -> list[str]:
    readings = item.get("readings")
    if not isinstance(readings, list):
        return []
    by_id: dict[str, str] = {}
    ordered: list[str] = []
    for rd in readings:
        if not isinstance(rd, dict):
            continue
        p = rd.get("paraphrase")
        if not isinstance(p, str) or not p.strip():
            continue
        rid = rd.get("reading_id")
        if rid in ("A", "B"):
            by_id[rid] = p.strip()
        else:
            ordered.append(p.strip())
    if "A" in by_id and "B" in by_id:
        return [by_id["A"], by_id["B"]]
    if len(readings) >= 2:
        r0, r1 = readings[0], readings[1]
        if isinstance(r0, dict) and isinstance(r1, dict):
            p0, p1 = r0.get("paraphrase"), r1.get("paraphrase")
            if isinstance(p0, str) and isinstance(p1, str) and p0.strip() and p1.strip():
                return [p0.strip(), p1.strip()]
    return ordered


def has_two_valid_readings(pred_row: dict[str, Any] | None) -> bool:
    if pred_row is None:
        return False
    if pred_row.get("success") is not True:
        return False
    pr = pred_row.get("parsed_response")
    if not isinstance(pr, dict):
        return False
    a, b = pr.get("reading_a"), pr.get("reading_b")
    if not is_valid_reading(a):
        return False
    if not is_valid_reading(b):
        return False
    if normalize_for_distinctness(a) == normalize_for_distinctness(b):
        return False
    return True


def effective_predicted_reading_count(pred_row: dict[str, Any] | None) -> int:
    if pred_row is None:
        return 0
    pr = pred_row.get("parsed_response")
    if not isinstance(pr, dict):
        return 0
    a, b = pr.get("reading_a"), pr.get("reading_b")
    sa = a if isinstance(a, str) else (str(a) if a is not None else "")
    sb = b if isinstance(b, str) else (str(b) if b is not None else "")
    if not is_valid_reading(sa):
        return 0
    if not is_valid_reading(sb):
        return 1
    if normalize_for_distinctness(sa) == normalize_for_distinctness(sb):
        return 1
    return 2


def evaluate_item(
    gold_item: dict[str, Any],
    pred_row: dict[str, Any] | None,
    threshold: float,
) -> dict[str, Any]:
    cid = gold_item["source_chambi_id"]
    input_text = gold_item.get("input_text", "")
    gold_readings = extract_gold_paraphrases(gold_item)

    errors: list[str] = []
    pred_reading_a: str | None = None
    pred_reading_b: str | None = None

    if pred_row is None:
        errors.append("missing_prediction")
    else:
        pr = pred_row.get("parsed_response")
        if isinstance(pr, dict):
            a, b = pr.get("reading_a"), pr.get("reading_b")
            pred_reading_a = a if isinstance(a, str) else ("" if a is None else str(a))
            pred_reading_b = b if isinstance(b, str) else ("" if b is None else str(b))
        else:
            errors.append("missing_parsed_response")

    if not gold_readings:
        errors.append("empty_gold_readings")

    ra = pred_reading_a if pred_reading_a is not None else ""
    rb = pred_reading_b if pred_reading_b is not None else ""

    covered: list[bool] = []
    best_sims: list[float] = []
    for g in gold_readings:
        sim_a = text_similarity(g, ra)
        sim_b = text_similarity(g, rb)
        best = max(sim_a, sim_b)
        best_sims.append(best)
        covered.append(best >= threshold)

    gold_a_best: float | None = best_sims[0] if len(best_sims) > 0 else None
    gold_b_best: float | None = best_sims[1] if len(best_sims) > 1 else None

    if len(gold_readings) == 0:
        item_recall = 0.0
    else:
        item_recall = sum(1 for c in covered if c) / len(gold_readings)

    double_correct = (
        len(gold_readings) == 2 and len(covered) == 2 and covered[0] and covered[1]
    )

    return {
        "source_chambi_id": cid,
        "input_text": input_text,
        "gold_readings": gold_readings,
        "pred_reading_a": pred_reading_a,
        "pred_reading_b": pred_reading_b,
        "has_two_valid_readings": has_two_valid_readings(pred_row),
        "gold_a_best_similarity": gold_a_best,
        "gold_b_best_similarity": gold_b_best,
        "covered_gold_readings": covered,
        "reading_coverage_recall": item_recall,
        "double_correct": double_correct,
        "errors": errors,
    }


def compute_summary(
    detail_rows: list[dict[str, Any]],
    num_prediction_items: int,
    pred_effective_counts: list[int],
) -> dict[str, Any]:
    num_gold_items = len(detail_rows)
    num_matched_items = sum(
        1 for d in detail_rows if "missing_prediction" not in d.get("errors", [])
    )
    num_missing_predictions = num_gold_items - num_matched_items

    two_ok = sum(
        1
        for d in detail_rows
        if "missing_prediction" not in d.get("errors", []) and d["has_two_valid_readings"]
    )
    two_reading_recovery_rate = (
        two_ok / num_matched_items if num_matched_items > 0 else 0.0
    )

    total_gold_readings = sum(len(d["gold_readings"]) for d in detail_rows)
    covered_total = sum(
        sum(1 for c in d["covered_gold_readings"] if c) for d in detail_rows
    )
    reading_coverage_recall = (
        covered_total / total_gold_readings if total_gold_readings > 0 else 0.0
    )

    dual_gold = [d for d in detail_rows if len(d["gold_readings"]) == 2]
    double_ok = sum(1 for d in dual_gold if d["double_correct"])
    double_correct_rate = (
        double_ok / len(dual_gold) if dual_gold else 0.0
    )

    if pred_effective_counts:
        average_predicted_reading_count = sum(pred_effective_counts) / len(
            pred_effective_counts
        )
    else:
        average_predicted_reading_count = 0.0

    return {
        "num_gold_items": num_gold_items,
        "num_prediction_items": num_prediction_items,
        "num_matched_items": num_matched_items,
        "num_missing_predictions": num_missing_predictions,
        "two_reading_recovery_rate": two_reading_recovery_rate,
        "reading_coverage_recall": reading_coverage_recall,
        "double_correct_rate": double_correct_rate,
        "average_predicted_reading_count": average_predicted_reading_count,
    }


def build_phase1_summary_document(
    metrics: dict[str, Any],
    *,
    gold_path: Path,
    predictions_path: Path,
    threshold: float,
    summary_output: Path,
    detail_output: Path,
) -> dict[str, Any]:
    """顶层结构化 summary：元信息 + 指标（metrics 内不再重复 threshold）。"""
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "meta": {
            "evaluator": "phase1_reading_recovery",
            "generated_at_utc": ts,
            "gold_path": str(gold_path.resolve()),
            "predictions_path": str(predictions_path.resolve()),
            "threshold": threshold,
            "summary_output": str(summary_output.resolve()),
            "detail_output": str(detail_output.resolve()),
        },
        "metrics": metrics,
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def enrich_detail_record(row: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    """在逐条结果上标注 schema，与 summary 的 meta.threshold 一致。"""
    out = dict(row)
    out["eval_schema_version"] = "1.0"
    out["similarity_threshold"] = threshold
    return out


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Phase 1 reading recovery vs gold")
    ap.add_argument(
        "--gold",
        type=Path,
        default=_DEFAULT_GOLD,
        help="Gold benchmark JSON (default: data/Chambi_benchmark_compact.json under repo root)",
    )
    ap.add_argument(
        "--predictions",
        "-p",
        type=Path,
        required=True,
        help="Phase 1 predictions JSONL",
    )
    ap.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=f"Summary JSON 路径（默认: {_DEFAULT_PHASE1_EVAL_DIR}/phase1_eval_summary.json）",
    )
    ap.add_argument(
        "--detail-output",
        type=Path,
        default=None,
        help=f"Details JSONL 路径（默认: {_DEFAULT_PHASE1_EVAL_DIR}/phase1_eval_details.jsonl）",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for coverage (default: 0.7)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"结果目录：未分别指定 summary/detail 时写入此目录（默认: {_DEFAULT_PHASE1_EVAL_DIR}）",
    )
    args = ap.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else _DEFAULT_PHASE1_EVAL_DIR
    summary_out = args.summary_output if args.summary_output is not None else (out_dir / "phase1_eval_summary.json")
    detail_out = args.detail_output if args.detail_output is not None else (out_dir / "phase1_eval_details.jsonl")

    if args.threshold < 0.0 or args.threshold > 1.0:
        print("threshold must be in [0, 1]", file=sys.stderr)
        return 1

    if not args.gold.is_file():
        print(f"Not a file: {args.gold}", file=sys.stderr)
        return 1
    if not args.predictions.is_file():
        print(f"Not a file: {args.predictions}", file=sys.stderr)
        return 1

    try:
        gold_items = load_gold_dataset(args.gold)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as e:
        print(f"Gold load error: {e}", file=sys.stderr)
        return 1

    pred_map = load_predictions(args.predictions)
    num_prediction_items = len(pred_map)
    pred_effective_counts = [effective_predicted_reading_count(p) for p in pred_map.values()]

    detail_rows: list[dict[str, Any]] = []
    for g in gold_items:
        cid = g["source_chambi_id"]
        pr = pred_map.get(cid)
        detail_rows.append(evaluate_item(g, pr, args.threshold))

    metrics = compute_summary(
        detail_rows,
        num_prediction_items,
        pred_effective_counts,
    )
    summary_doc = build_phase1_summary_document(
        metrics,
        gold_path=args.gold,
        predictions_path=args.predictions,
        threshold=args.threshold,
        summary_output=summary_out,
        detail_output=detail_out,
    )

    save_json(summary_out, summary_doc)
    detail_for_file = [enrich_detail_record(r, threshold=args.threshold) for r in detail_rows]
    append_jsonl(detail_out, detail_for_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
