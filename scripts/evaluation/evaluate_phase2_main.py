#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线评估 Phase 2-main 预测（与数据集 gold 对齐）。

用法（仓库根）::

    python scripts/evaluation/evaluate_phase2_main.py \\
        --predictions output/test2/glm.jsonl \\
        --dataset data/Chambi_benchmark_compact.json \\
        --output-report output/eval_reports/glm_phase2_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ethi_ambrot.common_eval_utils import load_dataset
from ethi_ambrot.phase2_main import normalize_dimension

ROT_KEYS = ("norm_activation", "ethical_obligation", "prescriptive_advice")


def _norm_text(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "").strip())


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(_norm_text(a).replace("，", " ").split()), set(_norm_text(b).replace("，", " ").split())
    sa, sb = {x for x in sa if x}, {x for x in sb if x}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _readings_by_id(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict):
                rows.append(o)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Phase 2-main predictions vs gold")
    ap.add_argument("--predictions", "-p", type=Path, required=True, help="Phase2-main JSONL")
    ap.add_argument("--dataset", "-d", type=Path, required=True, help="Benchmark JSON with gold_rot")
    ap.add_argument(
        "--output-report",
        "-o",
        type=Path,
        default=None,
        help="Write summary JSON (default: print to stdout only)",
    )
    args = ap.parse_args()

    if not args.predictions.is_file():
        print(f"Not a file: {args.predictions}", file=sys.stderr)
        return 1
    pred_rows = _load_jsonl(args.predictions)
    try:
        items = load_dataset(args.dataset)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1

    ds = {it.get("source_chambi_id"): it for it in items if it.get("source_chambi_id") is not None}

    n_pred = 0
    n_ok_parse = 0
    n_skipped_no_gold = 0
    n_skipped_bad_shape = 0

    rot_a_match = [0, 0]
    rot_b_match = [0, 0]
    jaccard_a: list[float] = []
    jaccard_b: list[float] = []

    val_pri_a = [0, 0]
    val_pri_b = [0, 0]
    gold_pri_in_pred_dims_a = [0, 0]
    gold_pri_in_pred_dims_b = [0, 0]

    diff_same_norm = 0
    diff_same_obl = 0
    diff_same_adv = 0
    diff_same_pri = 0
    diff_all_four_same = 0
    n_diff_sampled = 0

    def _triplet_exact_and_jac(pred: dict[str, Any], gold_rot: dict[str, Any]) -> tuple[bool, float]:
        ok_all = True
        jacs: list[float] = []
        for k in ROT_KEYS:
            gv = str(gold_rot.get(k) or "")
            pv = str(pred.get(k) or "")
            if _norm_text(gv) != _norm_text(pv):
                ok_all = False
            jacs.append(_token_jaccard(gv, pv))
        return ok_all, sum(jacs) / 3.0

    for rec in pred_rows:
        if rec.get("eval_phase") != 2:
            continue
        n_pred += 1
        if rec.get("success") is not True:
            continue
        pr = rec.get("parsed_response")
        if not isinstance(pr, dict):
            continue
        pa, pb = pr.get("reading_a"), pr.get("reading_b")
        if not isinstance(pa, dict) or not isinstance(pb, dict):
            n_skipped_bad_shape += 1
            continue
        n_ok_parse += 1

        sid = rec.get("source_chambi_id")
        row = ds.get(sid)
        if row is None:
            n_skipped_no_gold += 1
            continue
        g_a, g_b = _readings_by_id(row)
        if g_a is None or g_b is None:
            n_skipped_no_gold += 1
            continue

        gold_rota = g_a.get("gold_rot") if isinstance(g_a.get("gold_rot"), dict) else {}
        gold_rotb = g_b.get("gold_rot") if isinstance(g_b.get("gold_rot"), dict) else {}
        gold_va = g_a.get("gold_value_alignment") if isinstance(g_a.get("gold_value_alignment"), dict) else {}
        gold_vb = g_b.get("gold_value_alignment") if isinstance(g_b.get("gold_value_alignment"), dict) else {}

        for pred_block, gold_rot, acc_match, jacc_agg in (
            (pa, gold_rota, rot_a_match, jaccard_a),
            (pb, gold_rotb, rot_b_match, jaccard_b),
        ):
            tri_ok, jmean = _triplet_exact_and_jac(pred_block, gold_rot)
            if tri_ok:
                acc_match[0] += 1
            acc_match[1] += 1
            jacc_agg.append(jmean)

        gpri_a = gold_va.get("primary_dimension") if gold_va else None
        gpri_b = gold_vb.get("primary_dimension") if gold_vb else None
        pp_a = normalize_dimension(str(pa.get("primary_dimension", "")))
        pp_b = normalize_dimension(str(pb.get("primary_dimension", "")))
        ps_a = normalize_dimension(str(pa.get("secondary_dimension", "")))
        ps_b = normalize_dimension(str(pb.get("secondary_dimension", "")))

        for gpri, pp, ps, acc_pri, acc_in in (
            (gpri_a, pp_a, ps_a, val_pri_a, gold_pri_in_pred_dims_a),
            (gpri_b, pp_b, ps_b, val_pri_b, gold_pri_in_pred_dims_b),
        ):
            if gpri is None:
                continue
            gpn = normalize_dimension(str(gpri))
            if gpn is None:
                continue
            acc_pri[1] += 1
            pred_dims = {x for x in (pp, ps) if x}
            if pp == gpn:
                acc_pri[0] += 1
            acc_in[1] += 1
            if gpn in pred_dims:
                acc_in[0] += 1

        n_diff_sampled += 1
        sn = _norm_text(str(pa.get("norm_activation", ""))) == _norm_text(
            str(pb.get("norm_activation", ""))
        )
        so = _norm_text(str(pa.get("ethical_obligation", ""))) == _norm_text(
            str(pb.get("ethical_obligation", ""))
        )
        sa = _norm_text(str(pa.get("prescriptive_advice", ""))) == _norm_text(
            str(pb.get("prescriptive_advice", ""))
        )
        sp = pp_a is not None and pp_b is not None and pp_a == pp_b
        if sn:
            diff_same_norm += 1
        if so:
            diff_same_obl += 1
        if sa:
            diff_same_adv += 1
        if sp:
            diff_same_pri += 1
        if sn and so and sa and sp:
            diff_all_four_same += 1

    def _rate(num: int, den: int) -> float | None:
        return None if den == 0 else num / den

    summary = {
        "n_prediction_rows_phase2_two_readings": n_pred,
        "n_evaluated_parse_ok": n_ok_parse,
        "n_skipped_no_gold_or_reading": n_skipped_no_gold,
        "n_skipped_bad_parsed_shape": n_skipped_bad_shape,
        "rot": {
            "reading_a": {
                "exact_match_all_three_fields": _rate(rot_a_match[0], rot_a_match[1]),
                "n_compared": rot_a_match[1],
                "mean_token_jaccard_triplet_avg": sum(jaccard_a) / len(jaccard_a) if jaccard_a else None,
            },
            "reading_b": {
                "exact_match_all_three_fields": _rate(rot_b_match[0], rot_b_match[1]),
                "n_compared": rot_b_match[1],
                "mean_token_jaccard_triplet_avg": sum(jaccard_b) / len(jaccard_b) if jaccard_b else None,
            },
        },
        "value_primary": {
            "reading_a_primary_match_gold": _rate(val_pri_a[0], val_pri_a[1]),
            "reading_b_primary_match_gold": _rate(val_pri_b[0], val_pri_b[1]),
            "reading_a_gold_primary_in_pred_dimensions": _rate(
                gold_pri_in_pred_dims_a[0], gold_pri_in_pred_dims_a[1]
            ),
            "reading_b_gold_primary_in_pred_dimensions": _rate(
                gold_pri_in_pred_dims_b[0], gold_pri_in_pred_dims_b[1]
            ),
        },
        "differential": {
            "n_pairs": n_diff_sampled,
            "frac_same_norm_activation": _rate(diff_same_norm, n_diff_sampled),
            "frac_same_ethical_obligation": _rate(diff_same_obl, n_diff_sampled),
            "frac_same_prescriptive_advice": _rate(diff_same_adv, n_diff_sampled),
            "frac_same_primary_dimension": _rate(diff_same_pri, n_diff_sampled),
            "frac_all_four_identical_ab": _rate(diff_all_four_same, n_diff_sampled),
        },
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output_report}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
