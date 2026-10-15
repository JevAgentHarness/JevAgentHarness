"""scripts/hard_ni.py: paired NI outcomes for Table 1's post-Stage-1 rows, read from generated keys (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hard_ni as H  # noqa: E402


def _write(d: Path, body: str) -> None:
    (d / "x.tex").write_text(body, encoding="utf-8")


def test_rows_read_from_keys_and_missing_rows_skipped(tmp_path):
    _write(tmp_path, "\\defres{e4b/jevagent/vs_deep/wins}{8}\n\\defres{e4b/jevagent/vs_deep/losses}{3}\n"
                     "\\defres{e4b/n_eval}{100}\n\\defres{e6a/cascade/wins}{4}\n")
    res = H.compute(H.read_keys(tmp_path))
    assert set(res) == {"e4b"}                      # e6a lacks its losses and n keys
    r = res["e4b"]
    assert (r["n"], r["wins"], r["losses"]) == (100, 8, 3) and abs(r["delta"] - 5.0) < 1e-9
    assert r["lo"] < r["delta"] < r["hi"] and r["ni"] in {"yes", "no"}


def test_no_discordant_loss_is_non_inferior_and_heavy_loss_is_not(tmp_path):
    _write(tmp_path, "\\defres{e7b/jevagent_best/wins}{0}\n\\defres{e7b/jevagent_best/losses}{0}\n"
                     "\\defres{e7b/jevagent_best_rl/wins}{0}\n\\defres{e7b/jevagent_best_rl/losses}{131}\n"
                     "\\defres{e7b/jevagent_best_rl/n}{200}\n")
    keys = H.read_keys(tmp_path)
    keys["e7b/jevagent_best/n"] = "1{,}000"          # thousands separator as the generated keys print it
    res = H.compute(keys)
    assert res["e7b_jev"]["ni"] == "yes" and res["e7b_rl"]["ni"] == "no"


def test_e7f_arm_follows_the_paper_order_and_tex_has_signed_values(tmp_path):
    _write(tmp_path, "\\defres{e7f/rl_tier/wins}{9}\n\\defres{e7f/rl_tier/losses}{10}\n\\defres{e7f/rl_tier/n}{80}\n")
    res = H.compute(H.read_keys(tmp_path))
    tex = H.to_tex(res)
    assert "\\defres{hardni/e7f/n}{80}" in tex and "\\defres{hardni/e7f/delta}{$-$1.2}" in tex
    assert "\\defres{hardni/e7f/ni}{no}" in tex and "\\defres{hardni/margin}{3}" in tex
