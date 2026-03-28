#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将评测 JSONL（如 output/test1/glm.jsonl）导出为单个 JSON 文件，便于查看与比对。

用法（仓库根目录）::

    python scripts/dataset/export_predictions_json.py --input output/test1/glm.jsonl
    python scripts/dataset/export_predictions_json.py -i output/test1/glm.jsonl -o output/test1/glm.json --mode benchmark

--mode full        每行一条完整记录（含 raw_response、error 等），按 source_chambi_id 排序
--mode benchmark   仅 success 且含 parsed_response 的条，每条为 { **parsed_response, \"source_chambi_id\": ... , \"eval_phase\"? }，按 id 排序
                   （phase1 字段；phase2-main：嵌套 reading_a/b；旧 phase2 可能为 phase2_placeholder；历史 ambiguity_judgment / ambiguity_type）

可同时 --validate：自动识别 test1 / test2 占位 / new（ambiguity_judgment）/ legacy（ambiguity_type）并做粗粒度校验。

依赖：标准库即可。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ALLOWED_COARSE = frozenset(
    {"lexical", "semantic", "syntactic", "pragmatic", "overlapping", "combinational"}
)
_ALLOWED_COARSE_PRIMARY = _ALLOWED_COARSE | {"none"}
_ALLOWED_DIMS = frozenset({"Family", "Mianzi", "Harmony", "Public Morality"})
_ALLOWED_DIMS_PRIMARY = _ALLOWED_DIMS | {None}  # JSON null
_ALLOWED_STATUS = frozenset({"open", "resolved", "partially_resolved"})
_ALLOWED_JUDGMENT_STATUS = frozenset(
    {"open", "partially_resolved", "resolved", "none"}
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: skip invalid JSON ({e})", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def sort_key(row: dict[str, Any]) -> tuple[int, Any]:
    sid = row.get("source_chambi_id")
    try:
        return (0, int(sid))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return (1, str(sid))


def to_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("success") is not True:
            continue
        pr = r.get("parsed_response")
        if not isinstance(pr, dict):
            continue
        sid = r.get("source_chambi_id")
        item = {**pr, "source_chambi_id": sid}
        ep = r.get("eval_phase")
        if ep is not None:
            item["eval_phase"] = ep
        out.append(item)
    out.sort(key=sort_key)
    return out


def validate_test1_readings_item(obj: dict[str, Any], idx: int) -> list[str]:
    """Current test1: original_sentence, reading_a (required), reading_b optional."""
    errs: list[str] = []
    if not isinstance(obj.get("original_sentence"), str):
        errs.append(f"[{idx}] test1: original_sentence must be string")
    ra = obj.get("reading_a")
    if not isinstance(ra, str) or not ra.strip():
        errs.append(f"[{idx}] test1: reading_a must be non-empty string")
    if "reading_b" in obj and not isinstance(obj["reading_b"], str):
        errs.append(f"[{idx}] test1: reading_b must be string")
    return errs


def validate_test1_legacy_ambiguity_item(obj: dict[str, Any], idx: int) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj.get("has_ambiguity"), bool):
        errs.append(f"[{idx}] test1(legacy): has_ambiguity must be boolean")
    for k in ("ambiguity_explanation", "reading_a", "reading_b"):
        if k in obj and not isinstance(obj[k], str):
            errs.append(f"[{idx}] test1(legacy): {k} must be string")
    return errs


def validate_test2_placeholder_item(obj: dict[str, Any], idx: int) -> list[str]:
    errs: list[str] = []
    if obj.get("phase2_placeholder") is not True:
        errs.append(f"[{idx}] test2: expected phase2_placeholder true")
    ft = obj.get("free_text")
    if not isinstance(ft, str) or not ft.strip():
        errs.append(f"[{idx}] test2: free_text must be non-empty string")
    return errs


def validate_benchmark_item(obj: dict[str, Any], idx: int) -> list[str]:
    errs: list[str] = []
    if obj.get("input_text") is None:
        errs.append(f"[{idx}] missing input_text")
    amb = obj.get("ambiguity_type")
    if not isinstance(amb, dict):
        errs.append(f"[{idx}] ambiguity_type not object")
    else:
        c = amb.get("coarse")
        if c not in _ALLOWED_COARSE:
            errs.append(f"[{idx}] ambiguity_type.coarse={c!r} not in allowed set")
        s = amb.get("ambiguity_status")
        if s not in _ALLOWED_STATUS:
            errs.append(f"[{idx}] ambiguity_type.ambiguity_status={s!r} unexpected")
    vd = obj.get("value_dimension")
    if not isinstance(vd, dict):
        errs.append(f"[{idx}] value_dimension not object")
    else:
        if vd.get("primary") not in _ALLOWED_DIMS:
            errs.append(f"[{idx}] value_dimension.primary={vd.get('primary')!r}")
        sec = vd.get("secondary")
        if sec is not None and not isinstance(sec, list):
            errs.append(f"[{idx}] value_dimension.secondary must be list or omit")
        elif isinstance(sec, list):
            for x in sec:
                if x not in _ALLOWED_DIMS:
                    errs.append(f"[{idx}] value_dimension.secondary has bad dim {x!r}")
    readings = obj.get("readings")
    if not isinstance(readings, list) or len(readings) != 2:
        errs.append(
            f"[{idx}] readings must be list of length 2, got {repr(readings)[:100]}"
        )
    else:
        for j, rd in enumerate(readings):
            if not isinstance(rd, dict):
                errs.append(f"[{idx}] readings[{j}] not object")
                continue
            if rd.get("reading_id") not in ("A", "B"):
                errs.append(f"[{idx}] readings[{j}].reading_id={rd.get('reading_id')!r}")
            for k in ("paraphrase", "gold_rot", "gold_value_alignment"):
                if k not in rd:
                    errs.append(f"[{idx}] readings[{j}] missing {k}")
            gva = rd.get("gold_value_alignment")
            if isinstance(gva, dict):
                pd = gva.get("primary_dimension")
                if pd not in _ALLOWED_DIMS:
                    errs.append(f"[{idx}] readings[{j}].gold_value_alignment.primary_dimension={pd!r}")
                sd = gva.get("secondary_dimension")
                if sd is not None and sd not in _ALLOWED_DIMS:
                    errs.append(f"[{idx}] readings[{j}].secondary_dimension={sd!r}")
    return errs


def validate_multi_ambiguity_item(obj: dict[str, Any], idx: int) -> list[str]:
    """Legacy JSON benchmark schema (ambiguity_judgment / ambiguity_types / …)."""
    errs: list[str] = []
    if obj.get("input_text") is None:
        errs.append(f"[{idx}] missing input_text")

    aj = obj.get("ambiguity_judgment")
    if not isinstance(aj, dict):
        errs.append(f"[{idx}] ambiguity_judgment not object")
    else:
        if not isinstance(aj.get("is_ambiguous"), bool):
            errs.append(f"[{idx}] ambiguity_judgment.is_ambiguous must be boolean")
        st = aj.get("ambiguity_status")
        if st not in _ALLOWED_JUDGMENT_STATUS:
            errs.append(f"[{idx}] ambiguity_judgment.ambiguity_status={st!r}")
        if not isinstance(aj.get("explanation"), str):
            errs.append(f"[{idx}] ambiguity_judgment.explanation must be string")

    at = obj.get("ambiguity_types")
    if not isinstance(at, dict):
        errs.append(f"[{idx}] ambiguity_types not object")
    else:
        p = at.get("primary")
        if p not in _ALLOWED_COARSE_PRIMARY:
            errs.append(f"[{idx}] ambiguity_types.primary={p!r}")
        sec = at.get("secondary")
        if not isinstance(sec, list):
            errs.append(f"[{idx}] ambiguity_types.secondary must be array")
        elif isinstance(sec, list):
            for x in sec:
                if x not in _ALLOWED_COARSE:
                    errs.append(f"[{idx}] ambiguity_types.secondary bad label {x!r}")
        if not isinstance(at.get("reason"), str):
            errs.append(f"[{idx}] ambiguity_types.reason must be string")

    ev = obj.get("evidence")
    if not isinstance(ev, dict):
        errs.append(f"[{idx}] evidence not object")
    else:
        if not isinstance(ev.get("ambiguous_span"), str):
            errs.append(f"[{idx}] evidence.ambiguous_span must be string")
        if not isinstance(ev.get("explanation"), str):
            errs.append(f"[{idx}] evidence.explanation must be string")

    cr = obj.get("candidate_readings")
    if not isinstance(cr, list):
        errs.append(f"[{idx}] candidate_readings must be array")
    elif len(cr) > 2:
        errs.append(f"[{idx}] candidate_readings length must be 0–2, got {len(cr)}")
    else:
        for j, x in enumerate(cr):
            if not isinstance(x, str):
                errs.append(f"[{idx}] candidate_readings[{j}] must be string")

    vd = obj.get("value_dimension")
    if not isinstance(vd, dict):
        errs.append(f"[{idx}] value_dimension not object")
    else:
        pr = vd.get("primary")
        if pr not in _ALLOWED_DIMS_PRIMARY:
            errs.append(f"[{idx}] value_dimension.primary={pr!r}")
        sec2 = vd.get("secondary")
        if not isinstance(sec2, list):
            errs.append(f"[{idx}] value_dimension.secondary must be array")
        elif isinstance(sec2, list):
            for x in sec2:
                if x not in _ALLOWED_DIMS:
                    errs.append(f"[{idx}] value_dimension.secondary bad dim {x!r}")
        if not isinstance(vd.get("reason"), str):
            errs.append(f"[{idx}] value_dimension.reason must be string")

    return errs


def validate_phase2_main_item(obj: dict[str, Any], idx: int) -> list[str]:
    errs: list[str] = []
    req = (
        "norm_activation",
        "ethical_obligation",
        "prescriptive_advice",
        "primary_dimension",
        "secondary_dimension",
        "value_reason",
    )
    for side in ("reading_a", "reading_b"):
        block = obj.get(side)
        if not isinstance(block, dict):
            errs.append(f"[{idx}] phase2_main.{side} not object")
            continue
        for k in req:
            if k not in block or not isinstance(block[k], str):
                errs.append(f"[{idx}] phase2_main.{side}.{k} must be string")
    return errs


def validate_item_auto(obj: dict[str, Any], idx: int) -> list[str]:
    if isinstance(obj.get("reading_a"), dict) and "norm_activation" in obj["reading_a"]:
        if isinstance(obj.get("reading_b"), dict) and "norm_activation" in obj["reading_b"]:
            return validate_phase2_main_item(obj, idx)
    if obj.get("phase2_placeholder") is True:
        return validate_test2_placeholder_item(obj, idx)
    if "has_ambiguity" in obj:
        return validate_test1_legacy_ambiguity_item(obj, idx)
    if "original_sentence" in obj:
        return validate_test1_readings_item(obj, idx)
    if isinstance(obj.get("ambiguity_judgment"), dict):
        return validate_multi_ambiguity_item(obj, idx)
    if isinstance(obj.get("ambiguity_type"), dict):
        return validate_benchmark_item(obj, idx)
    return [
        f"[{idx}] unknown schema: expected phase2_main, test1, phase2_placeholder, ambiguity_judgment, or ambiguity_type"
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Export eval JSONL to a single JSON file")
    ap.add_argument("--input", "-i", type=Path, required=True, help="Input .jsonl path")
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output .json path (default: same stem as input, .json)",
    )
    ap.add_argument(
        "--mode",
        choices=("full", "benchmark"),
        default="benchmark",
        help="full=所有字段; benchmark=仅成功预测的 schema 行（对齐紧凑 benchmark）",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Print schema validation messages for benchmark-shaped rows",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Not a file: {args.input}", file=sys.stderr)
        return 1
    out_path = args.output
    if out_path is None:
        out_path = args.input.with_suffix(".json")

    raw_rows = load_jsonl(args.input)
    if args.mode == "full":
        raw_rows.sort(key=sort_key)
        payload = raw_rows
    else:
        payload = to_benchmark_rows(raw_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(payload)} records → {out_path}")

    if args.validate and args.mode == "benchmark":
        all_errs: list[str] = []
        for i, item in enumerate(payload):
            all_errs.extend(validate_item_auto(item, i))
        if all_errs:
            print("Validation issues:", file=sys.stderr)
            for e in all_errs:
                print(f"  {e}", file=sys.stderr)
            return 2
        print("Validation: no issues detected (coarse-level schema check).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
