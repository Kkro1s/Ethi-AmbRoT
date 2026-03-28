"""
Shared helpers for multi-model Chambi benchmark evaluation (JSONL, resume-safe).

Paths ``DEFAULT_DATASET`` / ``DEFAULT_EVAL_OUTPUT_DIR`` are relative to the **repository root**
(one level above this package), not the user’s home directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from ethi_ambrot.eval_prompt import build_prompt_phase2_main, build_prompt_test1, build_prompt_test2


def extract_reading_paraphrases(item: dict[str, Any]) -> tuple[str, str] | None:
    """Return (reading_a, reading_b) paraphrase strings for test2; None if unavailable."""
    readings = item.get("readings")
    if not isinstance(readings, list) or len(readings) < 2:
        return None
    by_id: dict[str, str] = {}
    for rd in readings:
        if not isinstance(rd, dict):
            continue
        p = rd.get("paraphrase")
        if not isinstance(p, str) or not p.strip():
            continue
        rid = rd.get("reading_id")
        if rid in ("A", "B"):
            by_id[rid] = p.strip()
    if "A" in by_id and "B" in by_id:
        return by_id["A"], by_id["B"]
    r0, r1 = readings[0], readings[1]
    if not isinstance(r0, dict) or not isinstance(r1, dict):
        return None
    p0, p1 = r0.get("paraphrase"), r1.get("paraphrase")
    if isinstance(p0, str) and isinstance(p1, str) and p0.strip() and p1.strip():
        return p0.strip(), p1.strip()
    return None


def _extract_labeled_section(full: str, label: str, next_labels: list[str]) -> str:
    """Text after 'label：' until the earliest following section header in next_labels."""
    m = re.search(re.escape(label) + r"\s*[:：]\s*", full)
    if not m:
        return ""
    start = m.end()
    end = len(full)
    tail = full[start:]
    for nl in next_labels:
        nm = re.search(re.escape(nl) + r"\s*[:：]", tail)
        if nm:
            end = min(end, start + nm.start())
    return full[start:end].strip()


def parse_test1_response(raw: str) -> dict[str, Any] | None:
    """
    Parse test1 free-text: 原句子, 解读A (required), 解读B (optional).
    Returns None if 解读A is missing or empty. 解读B may be absent or 「无」.
    """
    if not raw or not str(raw).strip():
        return None
    t = str(raw).strip()
    orig = _extract_labeled_section(t, "原句子", ["解读A", "解读B"])
    read_a = _extract_labeled_section(t, "解读A", ["解读B"])
    read_b = _extract_labeled_section(t, "解读B", [])
    if not read_a:
        return None
    rb = read_b.strip()
    if rb in ("无", "无。", "N/A", "n/a", "—", "-"):
        rb = ""
    return {
        "original_sentence": orig,
        "reading_a": read_a,
        "reading_b": rb,
    }


def parse_test1_response_legacy_ambiguity(raw: str) -> dict[str, Any] | None:
    """Older test1 layout: 是否有歧义 / 歧义解释 / 解读A / 解读B."""
    if not raw or not str(raw).strip():
        return None
    t = str(raw).strip()
    m0 = re.search(r"是否有歧义\s*[:：]\s*(是|否)", t)
    if not m0:
        return None
    has_ambiguity = m0.group(1) == "是"
    amb_exp = _extract_labeled_section(t, "歧义解释", ["解读A", "解读B"])
    read_a = _extract_labeled_section(t, "解读A", ["解读B"])
    read_b = _extract_labeled_section(t, "解读B", [])
    return {
        "has_ambiguity": has_ambiguity,
        "ambiguity_explanation": amb_exp,
        "reading_a": read_a,
        "reading_b": read_b,
    }


def parse_test1_response_auto(raw: str) -> dict[str, Any] | None:
    """
    Prefer legacy layout when ``是否有歧义`` is present (old prompts); otherwise
    use 原句子 / 解读A / 解读B layout.
    """
    t = str(raw or "").strip()
    if re.search(r"是否有歧义\s*[:：]", t):
        return parse_test1_response_legacy_ambiguity(raw)
    return parse_test1_response(raw)


def build_user_content_for_phase(
    phase: int,
    item: dict[str, Any],
    input_text: str,
) -> tuple[str | None, str | None]:
    """
    Build the user message for the given eval phase.
    Returns (content, error); error is set when phase 2 lacks two paraphrases.
    """
    if phase == 1:
        return build_prompt_test1(input_text), None
    pair = extract_reading_paraphrases(item)
    if pair is None:
        return None, "missing_readings_paraphrases"
    ra, rb = pair
    return build_prompt_phase2_main(input_text, ra, rb), None


def parse_response_for_phase(phase: int, raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Parse model output after a successful API return (caller still handles HTTP errors).
    Returns (parsed_dict, None) on success, or (None, error_code) on failure.
    """
    if phase == 1:
        p = parse_test1_response_auto(raw)
        if p is None:
            return None, "test1_parse_failed"
        return p, None
    from ethi_ambrot.phase2_main import parse_phase2_main_response

    p2 = parse_phase2_main_response(raw)
    if p2 is None:
        return None, "phase2_parse_failed"
    return p2, None

# --- 仓库根目录（ethi_ambrot 的上一级）---
_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PKG_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_DATASET = DEFAULT_DATA_DIR / "Chambi_benchmark_compact.json"

# 默认评测结果目录（各 run_*_eval.py 在此下放独立 JSONL；可用 --output 覆盖整路径）
DEFAULT_EVAL_OUTPUT_DIR = REPO_ROOT / "output"


def default_eval_jsonl_path(provider_slug: str, phase: int) -> Path:
    """
    Default JSONL path per provider and phase — phase1/2 分目录::

        output/test1/qwen.jsonl
        output/test2/qwen.jsonl
    """
    if phase not in (1, 2):
        raise ValueError(f"phase must be 1 or 2, got {phase}")
    return DEFAULT_EVAL_OUTPUT_DIR / f"test{phase}" / f"{provider_slug}.jsonl"


_FENCE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _strip_markdown_json_fence(text: str) -> str:
    """Remove a single wrapping ``` / ```json fence, or extract the first fenced block if embedded in other text."""
    raw = text.strip()
    m_full = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw, re.IGNORECASE)
    if m_full:
        return m_full.group(1).strip()
    m_in = _FENCE_BLOCK.search(text)
    if m_in:
        return m_in.group(1).strip()
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """
    1) Strip markdown code fences when present.
    2) json.loads on the whole string.
    3) On failure, take substring from first '{' to last '}' and json.loads.
    4) Require a JSON object (dict). Otherwise raise ValueError.
    """
    t = _strip_markdown_json_fence(text)
    candidates: list[str] = []
    if t:
        candidates.append(t)
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        brace_slice = t[start : end + 1]
        if brace_slice not in candidates:
            candidates.append(brace_slice)

    last_err: Exception | None = None
    for chunk in candidates:
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(obj, dict):
            return obj
        last_err = ValueError(f"Expected JSON object, got {type(obj).__name__}")

    if last_err is not None:
        raise ValueError(f"Could not parse JSON object: {last_err}") from last_err
    raise ValueError("Could not parse JSON object: empty or no brace segment")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset item {i} is not an object")
        out.append(row)
    return out


def phase2_cli_error(phase: int, phase1_jsonl: Path | None) -> str | None:
    """``--phase 2`` 须提供已存在的 ``--phase1-jsonl``；否则返回英文错误信息。"""
    if phase != 2:
        return None
    if phase1_jsonl is None:
        return "error: --phase 2 requires --phase1-jsonl"
    if not phase1_jsonl.is_file():
        return f"not a file: {phase1_jsonl}"
    return None


def dataset_by_chambi_id(items: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """``source_chambi_id`` → 整行 benchmark 对象（权威 ``input_text`` / gold）。"""
    out: dict[Any, dict[str, Any]] = {}
    for row in items:
        sid = row.get("source_chambi_id")
        if sid is not None:
            out[sid] = row
    return out


def load_done_ids(jsonl_path: Path, eval_phase: int | None = None) -> set[Any]:
    """
    Only ids with success is True and a non-None parsed_response are skipped (safe resume).
    When ``eval_phase`` is set, only records with matching ``eval_phase`` count; legacy lines
    without ``eval_phase`` are treated as phase 1 when ``eval_phase == 1``, and ignored for phase 2.
    """
    done: set[Any] = set()
    if not jsonl_path.is_file():
        return done
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("success") is not True:
                continue
            if record.get("parsed_response") is None:
                continue
            if eval_phase is not None:
                rp = record.get("eval_phase")
                if rp is None:
                    if eval_phase != 1:
                        continue
                elif rp != eval_phase:
                    continue
            sid = record.get("source_chambi_id")
            if sid is not None:
                done.add(sid)
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def strip_chambi_meta(item: dict[str, Any]) -> dict[str, Any]:
    """Fields used for eval only (no gold fields passed to the model)."""
    if "source_chambi_id" not in item or "input_text" not in item:
        raise KeyError("item must contain source_chambi_id and input_text")
    return {
        "source_chambi_id": item["source_chambi_id"],
        "input_text": item["input_text"],
    }


def parse_model_record(
    source_chambi_id: Any,
    input_text: str,
    model_name: str,
    raw_response: str,
    parsed_response: dict[str, Any] | None,
    success: bool,
    error: str | None,
    *,
    eval_phase: int = 1,
) -> dict[str, Any]:
    return {
        "source_chambi_id": source_chambi_id,
        "input_text": input_text,
        "model_name": model_name,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "success": success,
        "error": error,
        "eval_phase": eval_phase,
    }


def build_phase2_main_record(
    source_chambi_id: Any,
    input_text: str,
    model_name: str,
    raw_response: str,
    parsed_response: dict[str, Any] | None,
    success: bool,
    error: str | None,
    *,
    reading_a: str,
    reading_b: str,
) -> dict[str, Any]:
    """Phase 2-main JSONL 行：含 input_mode 与写入模型时的两条解读。"""
    return {
        "source_chambi_id": source_chambi_id,
        "input_text": input_text,
        "model_name": model_name,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "success": success,
        "error": error,
        "eval_phase": 2,
        "input_mode": "two_readings",
        "reading_a": reading_a,
        "reading_b": reading_b,
    }


def configure_shared_eval_args(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path | None = None,
) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to Chambi_benchmark_compact.json (JSON array)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Append-only JSONL (default: output/test<phase>/<provider>.jsonl unless set elsewhere)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=1,
        help="1=phase1；2=Phase 2-main（须配合 --phase1-jsonl）",
    )
    parser.add_argument(
        "--phase1-jsonl",
        type=Path,
        default=None,
        help="phase1 输出 JSONL；--phase 2 时必填，用于双解读过滤与 reading 来源",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max new items to process (after resume skip)")
    parser.add_argument("--sleep", type=float, default=0.4, help="Seconds to sleep after each API call")


def load_env_file(path: Path) -> None:
    """Load KEY=VAL lines from .env without overriding existing os.environ (same rules as generate_rot_dataset)."""
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


def load_env_candidates(repo_root: Path | None = None) -> list[Path]:
    """Load ``.env`` from repo root, cwd, and legacy ``ethi_ambrot_app`` paths."""
    root = REPO_ROOT if repo_root is None else repo_root
    candidates = [
        root / ".env",
        Path.cwd() / ".env",
        root / "ethi_ambrot_app" / ".env",
        root.parent / "ethi_ambrot_app" / ".env",
    ]
    seen: set[Path] = set()
    tried: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        tried.append(rp)
        load_env_file(rp)
    return tried

