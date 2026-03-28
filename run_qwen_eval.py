#!/usr/bin/env python3
"""从仓库根目录转发到 ``scripts/run_qwen_eval.py``，兼容旧命令 ``python run_qwen_eval.py``。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raise SystemExit(
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_qwen_eval.py"), *sys.argv[1:]],
        check=False,
    ).returncode
)
