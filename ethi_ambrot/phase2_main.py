"""
Phase 2-main：双解读样本过滤、模型输出解析、价值维度枚举。

Phase1 JSONL → ``iter_phase2_main_candidates`` → runner 仅用 input_text + reading_a/b 调用模型；
gold 标签不包含在模型提示中，仅用于离线评测。
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from ethi_ambrot.common_eval_utils import _extract_labeled_section, _strip_markdown_json_fence

PHASE2_VALUE_DIMENSIONS = frozenset({"Family", "Mianzi", "Harmony", "Public Morality"})

PHASE2_BLOCK_LABELS = ["社会规范", "伦理义务", "建议", "主要价值维度", "次级价值维度", "理由"]

_LABEL_TO_KEY = {
    "社会规范": "norm_activation",
    "伦理义务": "ethical_obligation",
    "建议": "prescriptive_advice",
    "主要价值维度": "primary_dimension",
    "次级价值维度": "secondary_dimension",
    "理由": "value_reason",
}

DIMENSION_ALIASES_CN: dict[str, str] = {
    "家庭": "Family",
    "面子": "Mianzi",
    "和谐": "Harmony",
    "公德": "Public Morality",
    "公共道德": "Public Morality",
}


def normalize_for_comparison(s: str) -> str:
    """用于判断两条 reading 是否实质相同（非语义模型）。"""
    t = unicodedata.normalize("NFKC", (s or "").strip().lower())
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[，。、；：！？""''（）(),.\u002d—_]", "", t)
    return t


def is_reading_b_placeholder(s: str) -> bool:
    x = (s or "").strip().lower()
    return x in (
        "",
        "无",
        "无。",
        "none",
        "n/a",
        "na",
        "不适用",
        "—",
        "-",
        "nil",
        "null",
    )


_PHASE2_PRED_SIDE_KEYS = frozenset(
    {
        "norm_activation",
        "ethical_obligation",
        "prescriptive_advice",
        "primary_dimension",
        "secondary_dimension",
        "value_reason",
    }
)


def is_valid_phase2_eval_row(record: dict[str, Any]) -> bool:
    """
    Phase 2 JSONL 行是否满足 judge 测评门槛：双解读、解析成功、嵌套 parsed_response 含两侧六字段。
    """
    ep = record.get("eval_phase")
    if ep is not None and ep != 2:
        return False
    if record.get("success") is not True:
        return False
    ra = record.get("reading_a")
    rb = record.get("reading_b")
    if not isinstance(ra, str) or not ra.strip():
        return False
    if not isinstance(rb, str) or is_reading_b_placeholder(rb):
        return False
    if normalize_for_comparison(ra) == normalize_for_comparison(rb):
        return False
    pr = record.get("parsed_response")
    if not isinstance(pr, dict):
        return False
    pa, pb = pr.get("reading_a"), pr.get("reading_b")
    if not isinstance(pa, dict) or not isinstance(pb, dict):
        return False
    if not _PHASE2_PRED_SIDE_KEYS.issubset(pa.keys()):
        return False
    if not _PHASE2_PRED_SIDE_KEYS.issubset(pb.keys()):
        return False
    return True


def is_valid_two_reading_item(record: dict[str, Any]) -> bool:
    """phase1 记录是否为 Phase 2-main 可用的双解读样本。"""
    if record.get("success") is not True:
        return False
    pr = record.get("parsed_response")
    if not isinstance(pr, dict):
        return False
    ra = pr.get("reading_a")
    rb = pr.get("reading_b")
    if not isinstance(ra, str) or not ra.strip():
        return False
    if not isinstance(rb, str) or is_reading_b_placeholder(rb):
        return False
    if normalize_for_comparison(ra) == normalize_for_comparison(rb):
        return False
    return True


def _last_success_phase1_by_id(path: Path) -> dict[Any, dict[str, Any]]:
    """同 id 保留 JSONL 中**最后一条**成功 phase1 记录（与追加续跑一致）。"""
    out: dict[Any, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("success") is not True:
                continue
            pr = rec.get("parsed_response")
            if not isinstance(pr, dict):
                continue
            sid = rec.get("source_ethi_ambrot_id")
            if sid is None:
                continue
            ep = rec.get("eval_phase")
            if ep is not None and ep != 1:
                continue
            out[sid] = rec
    return out


def _sort_sid(sid: Any) -> tuple[int, Any]:
    try:
        return (0, int(sid))
    except (TypeError, ValueError):
        return (1, str(sid))


def iter_phase2_main_candidates(phase1_jsonl_path: Path) -> list[tuple[Any, str, str, str]]:
    """
    从 phase1 JSONL 得到 (source_ethi_ambrot_id, input_text, reading_a, reading_b) 列表；
    input_text 来自 phase1 行内字段（runner 中会以 dataset 覆盖为权威）。
    """
    by_id = _last_success_phase1_by_id(phase1_jsonl_path)
    cand: list[tuple[Any, str, str, str]] = []
    for sid in sorted(by_id.keys(), key=_sort_sid):
        rec = by_id[sid]
        if not is_valid_two_reading_item(rec):
            continue
        pr = rec["parsed_response"]
        assert isinstance(pr, dict)
        ra = pr["reading_a"].strip()
        rb = pr["reading_b"].strip()
        it = rec.get("input_text")
        input_text = it if isinstance(it, str) else ""
        cand.append((sid, input_text, ra, rb))
    return cand


def _parse_phase2_block(block: str) -> dict[str, str] | None:
    if not (block or "").strip():
        return None
    out: dict[str, str] = {}
    for i, label in enumerate(PHASE2_BLOCK_LABELS):
        next_l = PHASE2_BLOCK_LABELS[i + 1 :]
        val = _extract_labeled_section(block, label, next_l)
        out[_LABEL_TO_KEY[label]] = val
    for k in (
        "norm_activation",
        "ethical_obligation",
        "prescriptive_advice",
        "primary_dimension",
        "value_reason",
    ):
        if not (out.get(k) or "").strip():
            return None
    return out


def parse_phase2_main_response(raw: str) -> dict[str, Any] | None:
    """
    解析 Phase 2-main 半结构化输出 → ``{ reading_a: {...}, reading_b: {...} }``。
    失败返回 None。
    """
    if not raw or not str(raw).strip():
        return None
    t = _strip_markdown_json_fence(str(raw).strip())
    # 模型常在「解读」与 A/B 之间加空格（如「解读 A」）
    ma = re.search(r"【解读\s*A\s*】", t)
    mb = re.search(r"【解读\s*B\s*】", t)
    if not ma or not mb or mb.start() < ma.end():
        return None
    block_a = t[ma.end() : mb.start()].strip()
    block_b = t[mb.end() :].strip()
    pa = _parse_phase2_block(block_a)
    pb = _parse_phase2_block(block_b)
    if pa is None or pb is None:
        return None

    # 归一化 value dimensions
    pa["primary_dimension"] = normalize_dimension(pa.get("primary_dimension", ""))
    pa["secondary_dimension"] = normalize_dimension(pa.get("secondary_dimension", ""))

    pb["primary_dimension"] = normalize_dimension(pb.get("primary_dimension", ""))
    pb["secondary_dimension"] = normalize_dimension(pb.get("secondary_dimension", ""))

    return {"reading_a": pa, "reading_b": pb}


def normalize_dimension(s: str) -> str | None:
    """将模型输出的维度字符串归一化到四选一；无法识别则返回 None。"""
    x = (s or "").strip()
    xl = x.lower()

    # 统一把“无/None/空值/不明显”类表达归一化为 None
    if not x or xl in {"none", "null", "n/a", "na"} or x in {"无", "不明显", "不适用", "暂无"}:
        return None

    # 已经是标准英文标签
    if x in PHASE2_VALUE_DIMENSIONS:
        return x

    # 大小写不一致时归一化
    for d in PHASE2_VALUE_DIMENSIONS:
        if xl == d.lower():
            return d

    # 中文别名映射
    if x in DIMENSION_ALIASES_CN:
        return DIMENSION_ALIASES_CN[x]

    return None
