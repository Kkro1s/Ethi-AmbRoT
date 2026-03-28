#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Chambi.json 调用 Qwen，生成与规定完全一致的「精简版 benchmark」JSON。

**输出文件**（默认）：仅含**已成功生成**的条目，按 `source_chambi_id` 升序排列；每条在 schema 之外多一个
`source_chambi_id`（对应 Chambi 的 `id`），便于续跑与对齐；提交论文前可自行删除该键。

若需要与 `Chambi.json` **等长**、缺项填 `null`，请加 `--pad-nulls`。

## 运行示例

  export QWEN_API_KEY=sk-...
  export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  export QWEN_MODEL=qwen-max

  python scripts/dataset/generate_rot_dataset.py --input data/Chambi.json --output data/Chambi_benchmark.json
  python scripts/dataset/generate_rot_dataset.py --input data/Chambi.json --output out.json --start 0 --end 100 --max-items 50

环境变量可兼用 QWEN_MAX_* / DASHSCOPE_API_KEY。
自动依次尝试加载 .env：仓库根目录、当前工作目录、仓库根/ethi_ambrot_app、上级/ethi_ambrot_app（不覆盖已 export 的变量）。

依赖: pip install openai
Python 3.10+
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ALLOWED_COARSE = frozenset(
    {"lexical", "semantic", "syntactic", "pragmatic", "overlapping", "combinational"}
)
ALLOWED_VALUE_DIMS = frozenset({"Family", "Mianzi", "Harmony", "Public Morality"})
ALLOWED_AMB_STATUS = frozenset({"open", "resolved", "partially_resolved"})

# 紧凑输出时附加，便于续跑（不属于论文 schema 时可删除）
META_CHAMBI_ID = "source_chambi_id"


def strip_chambi_meta(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != META_CHAMBI_ID}


def system_prompt_benchmark() -> str:
    return """You are an expert annotator building a Chinese ambiguity–value alignment benchmark.

You will receive one Chambi-style record: ambiguous `premise`, `hypothesis`, two `disambiguations` (each expands one reading of the premise), and `category` (rough source tag in Chinese—use only as a soft hint).

Produce ONE JSON object. Use ONLY these keys at every level—no additional keys anywhere:

1) input_text — must equal the original ambiguous `premise` string exactly.

2) ambiguity_type — object with:
   - coarse: one of lexical | semantic | syntactic | pragmatic | overlapping | combinational (English, lowercase).
   - ambiguity_status: usually "open" when both readings are plausible; otherwise resolved / partially_resolved.
   - explanation: short Chinese explanation of why the premise is ambiguous.

3) value_dimension — object for the sample as a whole:
   - primary: one of Family | Mianzi | Harmony | Public Morality (English, exact spelling).
   - secondary: JSON array of zero or more of those same labels (use [] if none).
   - reason: Chinese justification for this labeling.

4) readings — array of EXACTLY two objects, in order:
   - First object: reading_id "A", aligned with disambiguations[0].premise (paraphrase should match or tightly paraphrase that expanded premise).
   - Second object: reading_id "B", aligned with disambiguations[1].premise.
   Each reading has:
   - paraphrase (Chinese, disambiguated wording),
   - gold_rot: norm_activation, ethical_obligation, prescriptive_advice (Chinese; concrete, non-empty),
   - gold_value_alignment: primary_dimension (one of the four), secondary_dimension (same set or null), value_reason (Chinese).

Rules:
- Do not collapse to a single preferred reading unless one is clearly untenable; default ambiguity_status to open.
- gold_value_alignment may differ slightly between readings A and B when the ethical emphasis shifts.
- secondary_dimension must be either null or one of: Family, Mianzi, Harmony, Public Morality (same spelling as primary_dimension).
- Output valid JSON only. No markdown. No commentary outside JSON."""


def _disambiguation_pair(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dis = item.get("disambiguations")
    if not isinstance(dis, list) or len(dis) < 2:
        return {}, {}
    da = dis[0] if isinstance(dis[0], dict) else {}
    db = dis[1] if isinstance(dis[1], dict) else {}
    return da, db


def build_user_prompt(item: dict[str, Any]) -> str:
    premise = (item.get("premise") or "").strip()
    hypothesis = (item.get("hypothesis") or "").strip()
    category = (item.get("category") or "").strip()
    dis_a, dis_b = _disambiguation_pair(item)

    context = {
        "premise": premise,
        "hypothesis": hypothesis,
        "category_hint": category,
        "disambiguation_0_for_reading_A": {
            "premise": dis_a.get("premise", ""),
            "nli_label_vs_hypothesis": dis_a.get("label", ""),
        },
        "disambiguation_1_for_reading_B": {
            "premise": dis_b.get("premise", ""),
            "nli_label_vs_hypothesis": dis_b.get("label", ""),
        },
        "mapping_hints_for_coarse_field": (
            "Chambi category is NOT the same as coarse types; infer coarse from linguistic mechanism. "
            "Rough hints: 词汇→often lexical or combinational; 语法→syntactic; 语义→semantic; "
            "指代/不完整→semantic or pragmatic; 停顿/重音→overlapping or pragmatic as appropriate."
        ),
    }

    template = {
        "input_text": premise,
        "ambiguity_type": {
            "coarse": "semantic",
            "ambiguity_status": "open",
            "explanation": "（中文：为何有歧义）",
        },
        "value_dimension": {
            "primary": "Public Morality",
            "secondary": ["Harmony"],
            "reason": "（中文：为何整体归入该价值维度）",
        },
        "readings": [
            {
                "reading_id": "A",
                "paraphrase": "（与 disambiguation_0 一致或等价改写）",
                "gold_rot": {
                    "norm_activation": "",
                    "ethical_obligation": "",
                    "prescriptive_advice": "",
                },
                "gold_value_alignment": {
                    "primary_dimension": "Public Morality",
                    "secondary_dimension": "Harmony",
                    "value_reason": "",
                },
            },
            {
                "reading_id": "B",
                "paraphrase": "",
                "gold_rot": {
                    "norm_activation": "",
                    "ethical_obligation": "",
                    "prescriptive_advice": "",
                },
                "gold_value_alignment": {
                    "primary_dimension": "Public Morality",
                    "secondary_dimension": None,
                    "value_reason": "",
                },
            },
        ],
    }

    return (
        "Fill the following template with real content. "
        "Keep keys and nesting identical. "
        "`input_text` must be exactly the same string as `premise` in context.\n\n"
        "TEMPLATE (shape reference):\n"
        + json.dumps(template, ensure_ascii=False, indent=2)
        + "\n\nCONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not parse JSON object", text, 0)


def _normalize_secondary_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if s in ALLOWED_VALUE_DIMS:
            return [s]
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip() in ALLOWED_VALUE_DIMS:
                out.append(x.strip())
        return out
    return []


def _clean_readings_align_dims(obj: dict[str, Any]) -> None:
    readings = obj.get("readings")
    if not isinstance(readings, list):
        return
    for r in readings:
        if not isinstance(r, dict):
            continue
        gva = r.get("gold_value_alignment")
        if not isinstance(gva, dict):
            continue
        pdim = gva.get("primary_dimension")
        if isinstance(pdim, str) and pdim.strip() not in ALLOWED_VALUE_DIMS:
            pass
        sd = gva.get("secondary_dimension")
        if isinstance(sd, str) and sd.strip() in ALLOWED_VALUE_DIMS:
            gva["secondary_dimension"] = sd.strip()
        elif sd is None:
            gva["secondary_dimension"] = None
        else:
            gva["secondary_dimension"] = None


def postprocess_benchmark(
    obj: dict[str, Any],
    chambi_premise: str,
) -> dict[str, Any]:
    """Pin input_text; normalize enums where possible."""
    out = copy.deepcopy(obj)
    out["input_text"] = chambi_premise.strip()

    amb = out.get("ambiguity_type")
    if isinstance(amb, dict) and isinstance(amb.get("coarse"), str):
        c = amb["coarse"].strip().lower()
        if c in ALLOWED_COARSE:
            amb["coarse"] = c
    if isinstance(amb, dict) and isinstance(amb.get("ambiguity_status"), str):
        s = amb["ambiguity_status"].strip().lower()
        if s in ALLOWED_AMB_STATUS:
            amb["ambiguity_status"] = s

    if isinstance(out.get("value_dimension"), dict):
        vd = out["value_dimension"]
        if isinstance(vd.get("primary"), str):
            for cand in ALLOWED_VALUE_DIMS:
                if vd["primary"].strip() == cand or vd["primary"].strip().lower() == cand.lower():
                    vd["primary"] = cand
                    break
        vd["secondary"] = _normalize_secondary_list(vd.get("secondary"))

    _clean_readings_align_dims(out)

    readings = out.get("readings")
    if isinstance(readings, list):
        for idx, rid in enumerate(("A", "B")):
            if idx >= len(readings):
                break
            br = readings[idx]
            if isinstance(br, dict):
                br["reading_id"] = rid
                gva = br.get("gold_value_alignment")
                if isinstance(gva, dict) and isinstance(gva.get("primary_dimension"), str):
                    pd = gva["primary_dimension"].strip()
                    for cand in ALLOWED_VALUE_DIMS:
                        if pd == cand or pd.lower() == cand.lower():
                            gva["primary_dimension"] = cand
                            break

    return out


def _txt(x: Any) -> str:
    if x is None:
        return ""
    return x.strip() if isinstance(x, str) else str(x).strip()


def _secondary_dim_value(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    s = _txt(v)
    for cand in ALLOWED_VALUE_DIMS:
        if s == cand or s.lower() == cand.lower():
            return cand
    return None


def _force_primary_dimension(s: Any) -> str:
    t = _txt(s)
    for cand in ALLOWED_VALUE_DIMS:
        if t == cand or t.lower() == cand.lower():
            return cand
    return "Public Morality"


def canonical_compact_item(b: dict[str, Any]) -> dict[str, Any]:
    """
    裁剪为规定 schema：仅含 input_text、ambiguity_type、value_dimension、readings；
    readings 内仅含 reading_id、paraphrase、gold_rot、gold_value_alignment。
    """
    amb_src = b.get("ambiguity_type") if isinstance(b.get("ambiguity_type"), dict) else {}
    vd_src = b.get("value_dimension") if isinstance(b.get("value_dimension"), dict) else {}
    readings_src = b.get("readings") if isinstance(b.get("readings"), list) else []

    ambiguity_type = {
        "coarse": _txt(amb_src.get("coarse")),
        "ambiguity_status": _txt(amb_src.get("ambiguity_status")),
        "explanation": _txt(amb_src.get("explanation")),
    }
    value_dimension = {
        "primary": _force_primary_dimension(vd_src.get("primary")),
        "secondary": _normalize_secondary_list(vd_src.get("secondary")),
        "reason": _txt(vd_src.get("reason")),
    }

    readings: list[dict[str, Any]] = []
    for idx, rid in enumerate(("A", "B")):
        r = readings_src[idx] if idx < len(readings_src) and isinstance(readings_src[idx], dict) else {}
        grot = r.get("gold_rot") if isinstance(r.get("gold_rot"), dict) else {}
        gva = r.get("gold_value_alignment") if isinstance(r.get("gold_value_alignment"), dict) else {}
        readings.append(
            {
                "reading_id": rid,
                "paraphrase": _txt(r.get("paraphrase")),
                "gold_rot": {
                    "norm_activation": _txt(grot.get("norm_activation")),
                    "ethical_obligation": _txt(grot.get("ethical_obligation")),
                    "prescriptive_advice": _txt(grot.get("prescriptive_advice")),
                },
                "gold_value_alignment": {
                    "primary_dimension": _force_primary_dimension(gva.get("primary_dimension")),
                    "secondary_dimension": _secondary_dim_value(gva.get("secondary_dimension")),
                    "value_reason": _txt(gva.get("value_reason")),
                },
            }
        )

    return {
        "input_text": _txt(b.get("input_text")),
        "ambiguity_type": ambiguity_type,
        "value_dimension": value_dimension,
        "readings": readings,
    }


def is_valid_compact(c: Any) -> bool:
    if not isinstance(c, dict):
        return False
    need_top = {"input_text", "ambiguity_type", "value_dimension", "readings"}
    if set(c.keys()) != need_top:
        return False
    if not c.get("input_text"):
        return False
    amb = c.get("ambiguity_type")
    if not isinstance(amb, dict):
        return False
    if set(amb.keys()) != {"coarse", "ambiguity_status", "explanation"}:
        return False
    vd = c.get("value_dimension")
    if not isinstance(vd, dict):
        return False
    if set(vd.keys()) != {"primary", "secondary", "reason"}:
        return False
    if not isinstance(vd.get("secondary"), list):
        return False
    if vd.get("primary") not in ALLOWED_VALUE_DIMS:
        return False
    for s in vd["secondary"]:
        if s not in ALLOWED_VALUE_DIMS:
            return False
    readings = c.get("readings")
    if not isinstance(readings, list) or len(readings) != 2:
        return False
    for r in readings:
        if not isinstance(r, dict):
            return False
        if set(r.keys()) != {"reading_id", "paraphrase", "gold_rot", "gold_value_alignment"}:
            return False
        gr = r.get("gold_rot")
        if not isinstance(gr, dict) or set(gr.keys()) != {
            "norm_activation",
            "ethical_obligation",
            "prescriptive_advice",
        }:
            return False
        gva = r.get("gold_value_alignment")
        if not isinstance(gva, dict) or set(gva.keys()) != {
            "primary_dimension",
            "secondary_dimension",
            "value_reason",
        }:
            return False
        if gva.get("secondary_dimension") is not None:
            if gva["secondary_dimension"] not in ALLOWED_VALUE_DIMS:
                return False
        if gva.get("primary_dimension") not in ALLOWED_VALUE_DIMS:
            return False
        if not r.get("paraphrase"):
            return False
        if not all(_txt(gr.get(k)) for k in ("norm_activation", "ethical_obligation", "prescriptive_advice")):
            return False
    return True


def extract_compact_from_saved_row(
    row: Any,
    chambi_premise: str,
) -> dict[str, Any] | None:
    """从已落盘行提取纯 schema（支持旧版含 Chambi+benchmark 或已是纯 schema）。"""
    if row is None:
        return None
    if not isinstance(row, dict):
        return None
    row = strip_chambi_meta(row) if META_CHAMBI_ID in row else row
    inner: dict[str, Any] | None = None
    if isinstance(row.get("benchmark"), dict):
        inner = row["benchmark"]
        base_premise = (row.get("premise") or chambi_premise or "").strip()
    elif "input_text" in row and "readings" in row and "premise" not in row:
        inner = row
        base_premise = chambi_premise or _txt(row.get("input_text"))
    else:
        return None
    processed = postprocess_benchmark(inner, base_premise)
    return canonical_compact_item(processed)


def _load_env_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


def load_env_for_qwen(script_dir: Path) -> list[Path]:
    """按顺序加载 .env（不覆盖已有环境变量）。返回尝试过的路径列表。"""
    candidates = [
        script_dir / ".env",
        Path.cwd() / ".env",
        script_dir / "ethi_ambrot_app" / ".env",
        script_dir.parent / "ethi_ambrot_app" / ".env",
    ]
    seen: set[Path] = set()
    tried: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        tried.append(rp)
        _load_env_file(rp)
    return tried


def _api_config(env_tried: list[Path]) -> tuple[str, str, str]:
    api_key = (
        os.environ.get("QWEN_API_KEY")
        or os.environ.get("QWEN_MAX_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
    )
    if not api_key:
        lines = [
            "未检测到 API Key。请先配置环境变量之一：",
            "  QWEN_API_KEY、QWEN_MAX_API_KEY 或 DASHSCOPE_API_KEY",
            "",
            "或在以下路径创建 .env（写入 QWEN_API_KEY=sk-... 等，一行一个）：",
        ]
        for p in env_tried:
            exists = "（存在）" if p.is_file() else "（不存在）"
            lines.append(f"  {p}{exists}")
        lines.extend(
            [
                "",
                "也可在终端临时导出：",
                '  export QWEN_API_KEY="sk-..."',
                '  export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"',
                '  export QWEN_MODEL="qwen-max"',
            ]
        )
        raise RuntimeError("\n".join(lines))
    base_url = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("QWEN_MAX_API_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = (
        os.environ.get("QWEN_MODEL")
        or os.environ.get("QWEN_MAX_MODEL_NAME")
        or "qwen-max"
    )
    return api_key, base_url, model


def call_qwen_for_item(
    client: Any,
    model: str,
    user_content: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    from openai import APIError, RateLimitError  # type: ignore

    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt_benchmark()},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content or ""
            return extract_json_object(content)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            last_err = str(e)
        except RateLimitError:
            last_err = "RateLimitError"
            time.sleep(2**attempt)
            continue
        except APIError as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
            continue
        time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(last_err or "Qwen call failed after retries")


def load_prev_into_by_id(prev: list[Any], full_input: list[Any]) -> dict[Any, dict[str, Any]]:
    """从已存在输出恢复 by_id：支持「与输入等长且含 null」或「紧凑列表」。"""
    by_id: dict[Any, dict[str, Any]] = {}
    if not isinstance(prev, list) or not prev:
        return by_id

    premise_to_id: dict[str, Any] = {}
    for ch in full_input:
        if isinstance(ch, dict) and ch.get("id") is not None:
            p = (ch.get("premise") or "").strip()
            if p not in premise_to_id:
                premise_to_id[p] = ch["id"]

    aligned = len(prev) == len(full_input)
    if aligned:
        for i, cell in enumerate(prev):
            if cell is None or not isinstance(cell, dict):
                continue
            ch = full_input[i] if i < len(full_input) else None
            if not isinstance(ch, dict):
                continue
            rid = cell.get(META_CHAMBI_ID)
            if rid is None:
                rid = ch.get("id")
            prem_hint = (ch.get("premise") or "").strip()
            comp = extract_compact_from_saved_row(cell, prem_hint)
            if comp is not None and is_valid_compact(comp):
                by_id[rid] = comp
        return by_id

    # 紧凑列表：带 source_chambi_id，或凭 input_text 匹配 premise
    for cell in prev:
        if not isinstance(cell, dict):
            continue
        rid = cell.get(META_CHAMBI_ID)
        core = strip_chambi_meta(cell) if META_CHAMBI_ID in cell else cell
        if rid is None and core.get("input_text"):
            rid = premise_to_id.get(_txt(core["input_text"]))
        if rid is None:
            continue
        prem_final = ""
        for ch in full_input:
            if isinstance(ch, dict) and ch.get("id") == rid:
                prem_final = (ch.get("premise") or "").strip()
                break
        if not prem_final:
            prem_final = _txt(core.get("input_text"))
        comp = extract_compact_from_saved_row(core, prem_final)
        if comp is not None and is_valid_compact(comp):
            by_id[rid] = comp
    return by_id


def _write_output(
    output_path: Path,
    by_id: dict[Any, dict[str, Any]],
    full_input: list[Any],
    pad_nulls: bool,
) -> None:
    if pad_nulls:
        out_list: list[Any] = []
        for item in full_input:
            if not isinstance(item, dict):
                out_list.append(None)
                continue
            rid = item.get("id")
            c = by_id.get(rid) if rid is not None else None
            if c is not None and is_valid_compact(c):
                out_list.append(c)
            else:
                out_list.append(None)
    else:
        out_list = []
        for rid in sorted(by_id.keys(), key=lambda x: (str(type(x).__name__), x)):
            c = by_id[rid]
            if not is_valid_compact(c):
                continue
            out_list.append({**c, META_CHAMBI_ID: rid})
    output_path.write_text(
        json.dumps(out_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("请安装: pip install openai", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    env_tried = load_env_for_qwen(repo_root)

    ap = argparse.ArgumentParser(
        description="Chambi → Qwen → 输出与规定一致的纯精简 schema 数组（与输入等长，缺项为 null）"
    )
    ap.add_argument("--input", type=Path, required=True, help="Chambi.json 等 JSON 数组")
    ap.add_argument("--output", type=Path, required=True, help="输出 JSON 数组")
    ap.add_argument("--start", type=int, default=None, help="起始下标（含）")
    ap.add_argument("--end", type=int, default=None, help="结束下标（不含）")
    ap.add_argument("--max-items", type=int, default=None, help="区间内最多处理条数")
    ap.add_argument("--sleep", type=float, default=0.4, help="每条 API 后休眠（秒）")
    ap.add_argument(
        "--pad-nulls",
        action="store_true",
        help="输出与 Chambi 等长的数组，未完成填 null（默认：紧凑输出，无 null）",
    )
    args = ap.parse_args()

    full_input = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(full_input, list):
        print("输入须为 JSON 数组", file=sys.stderr)
        return 1

    try:
        api_key, base_url, model = _api_config(env_tried)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    timeout_raw = os.environ.get("QWEN_MAX_TIMEOUT") or os.environ.get("QWEN_TIMEOUT")
    try:
        timeout_sec = float(timeout_raw) if timeout_raw else 180.0
    except ValueError:
        timeout_sec = 180.0
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    start = 0 if args.start is None else args.start
    end = len(full_input) if args.end is None else args.end
    work_slice = full_input[start:end]
    if args.max_items is not None and args.max_items > 0:
        work_slice = work_slice[: args.max_items]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    by_id: dict[Any, dict[str, Any]] = {}
    if args.output.exists():
        try:
            prev = json.loads(args.output.read_text(encoding="utf-8"))
            if isinstance(prev, list) and prev:
                if len(prev) != len(full_input):
                    print(
                        f"注意: 已有输出条数 {len(prev)} 与输入 {len(full_input)}；"
                        f"{'按索引对齐恢复' if len(prev) == len(full_input) else '按紧凑列表或 input_text 匹配恢复'}。",
                        file=sys.stderr,
                    )
                by_id = load_prev_into_by_id(prev, full_input)
        except json.JSONDecodeError:
            pass

    done_ids = {rid for rid, row in by_id.items() if is_valid_compact(row)}

    total = len(work_slice)
    to_skip = sum(
        1 for x in work_slice if isinstance(x, dict) and x.get("id") in done_ids
    )
    print(
        f"模型={model} | 本 run 条数={total} | 将跳过(已有合法紧凑 schema)={to_skip}"
    )

    for i, item in enumerate(work_slice):
        if not isinstance(item, dict):
            continue
        rid = item.get("id")
        if rid is None:
            print(f"[{i+1}/{total}] skip: 缺少 id")
            continue
        if rid in done_ids:
            print(f"[{i+1}/{total}] id={rid} skip")
            continue

        premise = (item.get("premise") or "").strip()
        user_msg = build_user_prompt(item)
        try:
            raw_bench = call_qwen_for_item(client, model, user_msg, max_retries=3)
            bench = postprocess_benchmark(raw_bench, premise)
            compact = canonical_compact_item(bench)
            if not is_valid_compact(compact):
                raise ValueError(
                    "模型返回经裁剪后未通过 schema 校验（缺字段、readings 非 2 条或价值维度不合法等）"
                )
            by_id[rid] = compact
            done_ids.add(rid)
            print(f"[{i+1}/{total}] id={rid} OK")
        except Exception as ex:
            by_id.pop(rid, None)
            print(f"[{i+1}/{total}] id={rid} ERR: {ex}")

        _write_output(args.output, by_id, full_input, args.pad_nulls)
        if args.sleep > 0:
            time.sleep(args.sleep)

    _write_output(args.output, by_id, full_input, args.pad_nulls)
    print(f"已写入: {args.output} | 有效条数={sum(1 for _rid, c in by_id.items() if is_valid_compact(c))} | pad_nulls={args.pad_nulls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
