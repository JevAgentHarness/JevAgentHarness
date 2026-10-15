#!/usr/bin/env python3
"""SUPERSEDED for the paper by scripts/ni_power_exact.py (exact summation for the A6.17 rule); kept as the Monte-Carlo
companion of scripts/ni_size_sim.py (round-6 evidence, results/reviews/2026-10-01c/).

Power of the A6.14 NI decision rule (Tango one-sided score test; exact loss-count test when wins = 0 or fewer than 5
discordant tasks) at the registered sizes, for the cells of the paper's power table (App. G, Table tab:power), next
to the pre-A6.14 bootstrap rule. Reuses scripts/ni_size_sim.py (same data-generating
process: paired binary outcomes with discordance d and true difference Delta; reference success p_ref).

    python scripts/ni_power_table.py --reps 20000 --out results/reviews/2026-10-01c/ni_power_table.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ni_size_sim import LEVELS, Tables, cells, run_point  # noqa: E402

# (label, n, m, p_ref, d, delta, losses_only) -- the rows of the paper's table (A6.1), plus the C7 rows
ROWS = [
    ("C1F/C2F/C6F (Flash, p_ref 0.98)", 500, 0.03, 0.98, 0.02, -0.01, False),
    ("", 500, 0.03, 0.98, 0.05, 0.0, False),
    ("", 500, 0.03, 0.98, 0.10, 0.0, False),
    ("", 500, 0.03, 0.98, 0.10, -0.01, False),
    ("ceiling reference, losses only (p_ref 1)", 500, 0.03, 1.0, 0.02, -0.02, True),
    ("", 500, 0.03, 1.0, 0.01, -0.01, True),
    ("C1L/C2L/C6L (Flash-Lite, p_ref 0.80)", 500, 0.03, 0.80, 0.20, 0.0, False),
    ("", 500, 0.03, 0.80, 0.30, 0.0, False),
    ("", 500, 0.03, 0.80, 0.20, 0.03, False),
    ("", 500, 0.03, 0.80, 0.30, 0.05, False),
    ("C7 (tau^2 pass^1 majority, n 149, m 5 pt, p_ref 0.80)", 149, 0.05, 0.80, 0.20, 0.0, False),
    ("", 149, 0.05, 0.80, 0.30, 0.03, False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--out", default="results/reviews/2026-10-01c/ni_power_table.md")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tabs: dict[tuple[int, float], Tables] = {}
    lines = ["# Power of the NI decision rules at the registered sizes (A6.14 rule R5 vs the pre-A6.14 bootstrap R1)", "",
             f"reps {a.reps} per cell, B {a.boot} (bootstrap only), seed {a.seed}; data-generating process of "
             "`scripts/ni_size_sim.py` (paired binary outcomes; d = P(loss) + P(win); Delta = P(win) - P(loss)).", "",
             "| primary | n | m | p_ref | d | true Δ | R5 α=0.05 | R5 Holm floor 0.05/11 | R1 α=0.05 | R1 floor |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    raw = []
    for label, n, m, p_ref, d, delta, lo in ROWS:
        c = cells(p_ref, delta, d, lo)
        if c is None:
            lines.append(f"| {label} | {n} | {m} | {p_ref} | {d} | {delta:+.2f} | infeasible | | | |")
            continue
        tab = tabs.setdefault((n, m), Tables(n, m))
        res = run_point(rng, tab, c["win"], c["loss"], a.reps, a.boot)
        r = res["rates"]
        row = {"label": label, "n": n, "m": m, "p_ref": p_ref, "d": d, "delta": delta, "losses_only": lo,
               "R5": {lv: r[lv]["R5"]["rate"] for lv in r}, "R1": {lv: r[lv]["R1"]["rate"] for lv in r}}
        raw.append(row)
        k05 = [lv for lv in r if abs(LEVELS[lv] - 0.05) < 1e-9][0]
        kfl = [lv for lv in r if abs(LEVELS[lv] - 0.05 / 11) < 1e-9][0]
        lines.append(f"| {label} | {n} | {m:.2f} | {p_ref} | {d:.2f} | {delta:+.2f} | {r[k05]['R5']['rate']:.2f} | "
                     f"{r[kfl]['R5']['rate']:.2f} | {r[k05]['R1']['rate']:.2f} | {r[kfl]['R1']['rate']:.2f} |")
        print(lines[-1], flush=True)
    lines += ["", f"runtime {time.time() - t0:.0f} s"]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps({"reps": a.reps, "boot": a.boot, "seed": a.seed, "rows": raw}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
