"""The effective-registration manifest (scripts/prereg_effective.py -> docs/PREREG_EFFECTIVE.md) renders, names every
test of the 11-test Holm family and every registered arm, and the committed copy is what the configs / fits / registration
produce today (header commit / date line aside), so the derived view cannot drift from its sources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prereg_effective.py"
COMMITTED = ROOT / "docs" / "PREREG_EFFECTIVE.md"
HOLM_FAMILY = ["C1F", "C1L", "C2F", "C2L", "C3", "C4", "C5", "C6F", "C6L", "C7", "C8"]
GEN_PREFIX = "- Generated:"      # the one header line (commit, date) that may differ between two renders


def _strip_generated(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(GEN_PREFIX))


def test_manifest_renders_and_matches_the_committed_copy(tmp_path):
    out = tmp_path / "PREREG_EFFECTIVE.md"
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr                                   # (a)
    text = out.read_text()
    for label in HOLM_FAMILY:                                                        # (b) every test of the family
        assert f"**{label}**" in text, label
    sys.path.insert(0, str(ROOT / "scripts"))
    import prereg_check as PC

    for cfg, arms in PC.REGISTERED_SIZES.items():                                    # (b) every registered arm
        for arm in arms:
            assert f"`{arm}`" in text, f"{cfg}: {arm}"
    assert "PREREGISTRATION.md wins" in text and "## 5. Deviations so far" in text
    assert COMMITTED.exists(), "run python scripts/prereg_effective.py and commit docs/PREREG_EFFECTIVE.md"
    assert _strip_generated(COMMITTED.read_text()) == _strip_generated(text), (      # (c) no drift
        "docs/PREREG_EFFECTIVE.md differs from what the configs / fits / registration render now: "
        "run python scripts/prereg_effective.py and commit the result")
