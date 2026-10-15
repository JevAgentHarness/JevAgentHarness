"""The experiment configs match the pre-registration (scripts/prereg_check.py): registered arm names, sizes, seed
ranges, calibration mirrors. Keeps the configs locked to docs/PREREGISTRATION.md from the freeze on."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_configs_match_the_preregistration():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "prereg_check.py"), "--quiet"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 failure(s)" in r.stdout
