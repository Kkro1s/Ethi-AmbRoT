#!/usr/bin/env python3
"""从仓库根目录转发到 ``scripts/run_doubao_eval.py``。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raise SystemExit(
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_doubao_eval.py"), *sys.argv[1:]],
        check=False,
    ).returncode
)
