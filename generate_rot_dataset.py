#!/usr/bin/env python3
"""从仓库根目录转发到 ``scripts/generate_rot_dataset.py``。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raise SystemExit(
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_rot_dataset.py"), *sys.argv[1:]],
        check=False,
    ).returncode
)
