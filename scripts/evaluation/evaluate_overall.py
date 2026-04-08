#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overall End-to-End Benchmark Evaluator: combines Phase 1 and Phase 2 results.

用法（仓库根）::

    python scripts/evaluation/evaluate_overall.py \\
        --phase1-details output/phase1_eval/phase1_eval_details.jsonl \\
        --phase2-details output/phase2_judge_eval/phase2_judge_detail.jsonl \\
        --output-dir output/overall_eval

计分逻辑：
- Phase 1 未严格恢复双解读（double_correct != true）→ 该样本得 0 分
- Phase 2 未成功 judge（success != true 或 item_score 不在 [1,2,3]）→ 该样本得 0 分
- 两阶段都成功 → 该样本得分 = (phase2_item_score - 1) / 2（将 1-3 映射到 0-1）

Overall score = (1/N) * Σ s_i
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ethi_ambrot.common_eval_utils import load_dataset

_DEFAULT_GOLD = REPO_ROOT / "data" / "ethi_ambrot_benchmark_compact.json"
_DEFAULT_OUT_DIR = REPO_ROOT / "output" / "overall_eval"


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


def load_phase_details(path: Path) -> dict[int, dict[str, Any]]:
    """Load phase details JSONL, keep only last occurrence per source_ethi_ambrot_id."""
    by_sid: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict):
                continue
            # Support both source_ethi_ambrot_id and legacy source_chambi_id
            sid = o.get("source_ethi_ambrot_id") or o.get("source_chambi_id")
            if sid is None:
                continue
            try:
                k = _coerce_sid(sid)
            except (TypeError, ValueError):
                continue
            by_sid[k] = o
    return by_sid


def compute_item_score(
    phase1_row: dict[str, Any] | None,
    phase2_row: dict[str, Any] | None,
) -> tuple[float, str]:
    """
    Returns (normalized_score, status_code).

    Status codes:
    - "e2e_success": both phases passed
    - "phase1_fail": phase1 did not strictly recover dual readings
    - "phase2_fail": phase2 did not get valid judge score
    - "missing": no phase1 or phase2 detail found
    """
    if phase1_row is None:
        return 0.0, "missing"

    if not phase1_row.get("double_correct"):
        return 0.0, "phase1_fail"

    if phase2_row is None:
        return 0.0, "phase2_fail"

    if phase2_row.get("success") is not True:
        return 0.0, "phase2_fail"

    item_score = phase2_row.get("item_score")
    if item_score is None:
        return 0.0, "phase2_fail"

    try:
        score_val = float(item_score)
    except (TypeError, ValueError):
        return 0.0, "phase2_fail"

    if not (1.0 <= score_val <= 3.0):
        return 0.0, "phase2_fail"

    normalized = (score_val - 1.0) / 2.0
    return normalized, "e2e_success"


def build_detail_record(
    sid: int,
    phase1_row: dict[str, Any] | None,
    phase2_row: dict[str, Any] | None,
    item_score_normalized: float,
    status: str,
) -> dict[str, Any]:
    return {
        "source_ethi_ambrot_id": sid,
        "phase1_double_correct": phase1_row.get("double_correct") if phase1_row else None,
        "phase2_success": phase2_row.get("success") if phase2_row else None,
        "phase2_item_score_raw": phase2_row.get("item_score") if phase2_row else None,
        "item_score_normalized": item_score_normalized,
        "status": status,
        "eval_schema_version": "1.0",
    }


def build_overall_summary(
    *,
    num_gold_items: int,
    num_e2e_success: int,
    num_phase1_fail: int,
    num_phase2_fail: int,
    num_missing: int,
    overall_e2e_score: float,
    phase1_summary_path: Path | None,
    phase1_details_path: Path,
    phase2_summary_path: Path | None,
    phase2_details_path: Path,
    gold_path: Path,
    summary_output: Path,
    detail_output: Path,
    phase1_summary: dict[str, Any] | None,
    phase2_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    num_phase1_pass = num_e2e_success + num_phase2_fail
    phase1_gate_rate = num_phase1_pass / max(num_gold_items, 1)
    phase2_conditional_success_rate = (
        num_e2e_success / max(num_phase1_pass, 1) if num_phase1_pass > 0 else 0.0
    )

    ts = datetime.now(timezone.utc).isoformat()

    meta: dict[str, Any] = {
        "evaluator": "overall_end_to_end",
        "generated_at_utc": ts,
        "phase1_details_path": str(phase1_details_path.resolve()),
        "phase2_details_path": str(phase2_details_path.resolve()),
        "gold_path": str(gold_path.resolve()),
        "summary_output": str(summary_output.resolve()),
        "detail_output": str(detail_output.resolve()),
    }

    if phase1_summary_path:
        meta["phase1_summary_path"] = str(phase1_summary_path.resolve())
    if phase2_summary_path:
        meta["phase2_summary_path"] = str(phase2_summary_path.resolve())

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "meta": meta,
        "metrics": {
            "num_gold_items": num_gold_items,
            "num_e2e_success": num_e2e_success,
            "num_phase1_fail": num_phase1_fail,
            "num_phase2_fail": num_phase2_fail,
            "num_missing": num_missing,
            "phase1_gate_rate": phase1_gate_rate,
            "phase2_conditional_success_rate": phase2_conditional_success_rate,
            "overall_e2e_score": overall_e2e_score,
            "overall_e2e_score_percent": overall_e2e_score * 100.0,
        },
    }

    if phase1_summary and isinstance(phase1_summary.get("metrics"), dict):
        result["phase1_metrics_reference"] = phase1_summary["metrics"]

    if phase2_summary and isinstance(phase2_summary.get("metrics"), dict):
        result["phase2_metrics_reference"] = phase2_summary["metrics"]

    return result


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute overall end-to-end benchmark score from Phase 1 and Phase 2 evaluations"
    )
    ap.add_argument(
        "--phase1-details",
        type=Path,
        required=True,
        help="Phase 1 details JSONL (e.g., output/phase1_eval/phase1_eval_details.jsonl)",
    )
    ap.add_argument(
        "--phase2-details",
        type=Path,
        required=True,
        help="Phase 2 details JSONL (e.g., output/phase2_judge_eval/phase2_judge_detail.jsonl)",
    )
    ap.add_argument(
        "--phase1-summary",
        type=Path,
        default=None,
        help="Optional Phase 1 summary JSON for reference metrics",
    )
    ap.add_argument(
        "--phase2-summary",
        type=Path,
        default=None,
        help="Optional Phase 2 summary JSON for reference metrics",
    )
    ap.add_argument(
        "--gold",
        type=Path,
        default=_DEFAULT_GOLD,
        help="Gold benchmark JSON (default: data/ethi_ambrot_benchmark_compact.json)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Override summary JSON path",
    )
    ap.add_argument(
        "--detail-output",
        type=Path,
        default=None,
        help="Override details JSONL path",
    )
    args = ap.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else _DEFAULT_OUT_DIR
    summary_out = args.summary_output if args.summary_output is not None else (out_dir / "overall_summary.json")
    detail_out = args.detail_output if args.detail_output is not None else (out_dir / "overall_details.jsonl")

    # Load gold
    if not args.gold.is_file():
        print(f"Gold file not found: {args.gold}", file=sys.stderr)
        return 1

    try:
        gold_items = load_dataset(args.gold)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Gold load error: {e}", file=sys.stderr)
        return 1

    num_gold_items = len(gold_items)

    # Load phase details
    if not args.phase1_details.is_file():
        print(f"Phase 1 details not found: {args.phase1_details}", file=sys.stderr)
        return 1

    if not args.phase2_details.is_file():
        print(f"Phase 2 details not found: {args.phase2_details}", file=sys.stderr)
        return 1

    phase1_details = load_phase_details(args.phase1_details)
    phase2_details = load_phase_details(args.phase2_details)

    # Load optional summaries
    phase1_summary: dict[str, Any] | None = None
    phase2_summary: dict[str, Any] | None = None

    if args.phase1_summary and args.phase1_summary.is_file():
        try:
            phase1_summary = json.loads(args.phase1_summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: could not load phase1 summary: {e}", file=sys.stderr)

    if args.phase2_summary and args.phase2_summary.is_file():
        try:
            phase2_summary = json.loads(args.phase2_summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: could not load phase2 summary: {e}", file=sys.stderr)

    # Compute per-item scores
    detail_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    status_counts = {"e2e_success": 0, "phase1_fail": 0, "phase2_fail": 0, "missing": 0}

    for item in gold_items:
        sid = item["source_ethi_ambrot_id"]
        phase1_row = phase1_details.get(sid)
        phase2_row = phase2_details.get(sid)

        item_score_normalized, status = compute_item_score(phase1_row, phase2_row)
        scores.append(item_score_normalized)
        status_counts[status] += 1

        detail_rows.append(build_detail_record(sid, phase1_row, phase2_row, item_score_normalized, status))

    overall_e2e_score = sum(scores) / max(len(scores), 1)

    # Build summary
    summary_doc = build_overall_summary(
        num_gold_items=num_gold_items,
        num_e2e_success=status_counts["e2e_success"],
        num_phase1_fail=status_counts["phase1_fail"],
        num_phase2_fail=status_counts["phase2_fail"],
        num_missing=status_counts["missing"],
        overall_e2e_score=overall_e2e_score,
        phase1_summary_path=args.phase1_summary,
        phase1_details_path=args.phase1_details,
        phase2_summary_path=args.phase2_summary,
        phase2_details_path=args.phase2_details,
        gold_path=args.gold,
        summary_output=summary_out,
        detail_output=detail_out,
        phase1_summary=phase1_summary,
        phase2_summary=phase2_summary,
    )

    # Write outputs
    save_json(summary_out, summary_doc)
    append_jsonl(detail_out, detail_rows)

    print(f"Overall E2E Score: {overall_e2e_score:.4f} ({overall_e2e_score * 100:.2f}%)")
    print(f"Summary: {summary_out}")
    print(f"Details: {detail_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
