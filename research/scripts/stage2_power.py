#!/usr/bin/env python
"""Exact power of the registered NI rule (A6.21 (a), ``bb_ordered``) for the Stage-2 confirmation
(docs/PREREG_A7_STAGE2.md §5): the NI clause of S2-1, S2-2, S2-4 (n = 500 fresh JevGym tasks, m = 3 pt) at the four
Holm levels of the four-test Stage-2 family (0.05/4 ... 0.05/1); the secondary rows, Holm within the four-row secondary
table of A7.9 (floor 0.05/4): the Flash-Lite row (S2-1L) and the L3
secondary rows (tau^2 telecom majority pass, m = 5 pt; ALFWorld valid_seen, m = 3 pt). Exact summation over the
paired trinomial counts with the certified rejection regions of ``scripts/ni_exact_size.CertifiedRule`` (the method of
``scripts/ni_power_exact.py``); no Monte Carlo.

The cost / latency clauses (one-sided paired t-tests) and S2-3 (paired t-tests on per-task decision accuracy) are
powered by the normal approximation in the A7 text; this script covers the NI clauses only.

    python scripts/stage2_power.py --design a7 [--out results/reviews/2026-10-02]   # the A7 table (stage2_power.md)
    python scripts/stage2_power.py [--telecom-n N]       # as amended by A7.9-A7.16 (stage2_power_amended.md):
        # the six-row secondary table (S2-1T, S2-1L, S2-L3a, S2-L3b, S2-1X, S2-4S; Holm floor 0.05/6), S2-L3a on
        # single-attempt success at the planning n of A7.15 (640, 800) -- and at the frozen n with --telecom-n,
        # recomputed before evaluation (A7.15) --, S2-1X at n = 500 / 300 (A7.12 (c)). Planning statements only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ni_exact_size import CertifiedRule  # noqa: E402
from ni_power_exact import feasibility, scenario  # noqa: E402

LEVELS = {"0.05/4": 0.05 / 4, "0.05/3": 0.05 / 3, "0.05/2": 0.05 / 2, "0.05/1": 0.05}
LEVELS5 = {"0.05/5": 0.05 / 5, **LEVELS}          # the five-row secondary table of A7.12 (g)
LEVELS6 = {"0.05/6": 0.05 / 6, **LEVELS5}         # the six-row secondary table (A7.18 (f))
# (group, n, m, p_ref or None, d, delta, losses_only)
ROWS = [
    ("S2-1 / S2-4 / S2-2 NI clause (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.02, 0.0, False),
    ("S2-1 / S2-4 / S2-2 NI clause (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.02, -0.01, False),
    ("S2-1 / S2-4 / S2-2 NI clause (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.04, 0.0, False),
    ("S2-1 / S2-4 / S2-2 NI clause (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.04, -0.01, False),
    ("S2-1 / S2-4 / S2-2 NI clause (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.04, -0.02, False),
    ("S2-1 / S2-4 / S2-2 NI clause (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.06, -0.02, False),
    ("ceiling ref. (losses only, 1.00)", 500, 0.03, 1.0, 0.01, -0.01, True),
    ("ceiling ref. (losses only, 1.00)", 500, 0.03, 1.0, 0.02, -0.02, True),
    ("S2-1L secondary (Flash-Lite, ref. 0.80)", 500, 0.03, 0.80, 0.20, 0.0, False),
    ("S2-1L secondary (Flash-Lite, ref. 0.80)", 500, 0.03, 0.80, 0.20, 0.03, False),
    ("S2-L3a tau^2 telecom (majority pass, n = 114, ref. 0.60)", 114, 0.05, 0.60, 0.15, 0.0, False),
    ("S2-L3a tau^2 telecom (majority pass, n = 114, ref. 0.60)", 114, 0.05, 0.60, 0.25, 0.0, False),
    ("S2-L3a tau^2 telecom (majority pass, n = 114, ref. 0.60)", 114, 0.05, 0.60, 0.25, 0.03, False),
    ("S2-L3b ALFWorld valid_seen (n = 140, ref. 0.80)", 140, 0.03, 0.80, 0.10, 0.0, False),
    ("S2-L3b ALFWorld valid_seen (n = 140, ref. 0.80)", 140, 0.03, 0.80, 0.20, 0.0, False),
    ("S2-L3b ALFWorld valid_seen (n = 140, ref. 0.80)", 140, 0.03, 0.80, 0.20, 0.05, False),
]

# as amended (A7.9-A7.16): (group, n, m, p_ref, d, delta, losses_only, table) -- planning scenarios, not results
FAMILY_ROWS = [r + ("family",) for r in ROWS[:8]]
SECONDARY_ROWS = [
    ("S2-1L (Flash-Lite, ref. 0.80)", 500, 0.03, 0.80, 0.20, 0.0, False),
    ("S2-1L (Flash-Lite, ref. 0.80)", 500, 0.03, 0.80, 0.20, 0.03, False),
    ("S2-L3a telecom (single attempt, n = 640, ref. 0.60)", 640, 0.05, 0.60, 0.15, 0.0, False),
    ("S2-L3a telecom (single attempt, n = 640, ref. 0.60)", 640, 0.05, 0.60, 0.25, 0.0, False),
    ("S2-L3a telecom (single attempt, n = 800, ref. 0.60)", 800, 0.05, 0.60, 0.15, 0.0, False),
    ("S2-L3a telecom (single attempt, n = 800, ref. 0.60)", 800, 0.05, 0.60, 0.25, 0.0, False),
    ("S2-L3b ALFWorld valid_seen (n = 140, ref. 0.80)", 140, 0.03, 0.80, 0.20, 0.0, False),
    ("S2-1X (DeepSeek-V4-Pro, n = 500, ref. 0.90)", 500, 0.03, 0.90, 0.06, 0.0, False),
    ("S2-1X (DeepSeek-V4-Pro, n = 500, ref. 0.90)", 500, 0.03, 0.90, 0.10, 0.0, False),
    ("S2-1X (DeepSeek-V4-Pro, n = 300, ref. 0.90)", 300, 0.03, 0.90, 0.06, 0.0, False),
    ("S2-1X (DeepSeek-V4-Pro, n = 300, ref. 0.90)", 300, 0.03, 0.90, 0.10, 0.0, False),
    ("S2-4S (StartLux-4B, n = 500, ref. 0.98)", 500, 0.03, 0.98, 0.02, 0.0, False),
    ("S2-4S (StartLux-4B, n = 500, ref. 0.98)", 500, 0.03, 0.98, 0.04, 0.0, False),
    ("S2-4S (StartLux-4B, n = 500, ref. 0.98)", 500, 0.03, 0.98, 0.04, -0.01, False),
]


def amended_rows(telecom_n: int | None) -> list[tuple]:
    rows = FAMILY_ROWS + [r + ("secondary",) for r in SECONDARY_ROWS]
    if telecom_n:
        rows += [(f"S2-L3a telecom (single attempt, FROZEN n = {telecom_n}, ref. 0.60)", telecom_n, 0.05, 0.60, d, 0.0,
                  False, "secondary") for d in (0.15, 0.25)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/reviews/2026-10-02")
    ap.add_argument("--eps-grid", type=float, default=0.005)
    ap.add_argument("--design", choices=("a7", "amended"), default="amended",
                    help="a7: the registered A7 table (stage2_power.*); amended: A7.9-A7.16 (stage2_power_amended.*)")
    ap.add_argument("--telecom-n", type=int, default=None, help="add S2-L3a rows at the frozen telecom n (A7.15)")
    a = ap.parse_args()
    if a.design == "a7":
        design = [r + ("family" if i < 8 else "secondary4",) for i, r in enumerate(ROWS)]
    else:
        design = amended_rows(a.telecom_n)
    levels_of = {"family": LEVELS, "secondary4": LEVELS, "secondary": LEVELS6}
    t0 = time.time()
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    rho_max: dict[tuple[int, float], float] = {}
    for _, n, m, _, d, _, _, tab in design:
        rho_max[(n, m, tab)] = max(rho_max.get((n, m, tab), 0.0), d)
    rules: dict[tuple[int, float, str], CertifiedRule] = {}
    rows = []
    for group, n, m, p_ref, d, delta, lo, tab in design:
        if (n, m, tab) not in rules:
            rules[(n, m, tab)] = CertifiedRule(n, m, levels_of[tab], eps_grid=a.eps_grid,
                                               rho_max=min(1.0, rho_max[(n, m, tab)] + 0.05), log=log)
        p_win, p_loss = scenario(d, delta, lo)
        pr = rules[(n, m, tab)].probs(p_win, p_loss)
        row = {"group": group, "table": tab, "n": n, "m": m, "p_ref": p_ref, "d": d, "delta": delta,
               "losses_only": lo, "p_win": p_win, "p_loss": p_loss, **feasibility(p_win, p_loss, p_ref),
               "power": pr["reject"], "mass_beyond_Dmax": pr["mass_beyond_Dmax"]}
        rows.append(row)
        log(f"[{time.time() - t0:6.1f}s] {group} d={d} delta={delta:+.2f}: " +
            ", ".join(f"{lv} {row['power'][lv]:.3f}" for lv in levels_of[tab]))
    res = {"design": a.design, "rows": rows, "levels": LEVELS, "levels_secondary": LEVELS6 if a.design == "amended"
           else LEVELS, "rule": "bb_ordered (A6.21 (a))", "date": time.strftime("%Y-%m-%d"),
           "runtime_s": time.time() - t0, "command": "python " + " ".join(sys.argv),
           "checks": {f"{n}/{m}/{t}": rules[(n, m, t)].info for (n, m, t) in rules}}
    a.out.mkdir(parents=True, exist_ok=True)
    stem = "stage2_power" if a.design == "a7" else "stage2_power_amended"
    (a.out / f"{stem}.json").write_text(json.dumps(res, indent=1))
    if a.design == "amended":
        _amended_md(res, a.out / f"{stem}.md")
        print(f"done in {res['runtime_s']:.0f}s -> {a.out / (stem + '.md')}", file=sys.stderr)
        return
    L = ["# Stage-2 power of the NI clauses (A7 §5; exact, A6.21 rule `bb_ordered`)", "",
         f"`scripts/stage2_power.py` ({res['date']}): exact summation over the paired trinomial counts with the certified "
         "rejection regions of `scripts/ni_exact_size.CertifiedRule`. Levels: the four Holm steps of the four-test "
         "Stage-2 family (floor 0.05/4). The S2-1L and L3 rows are secondary: Holm within the four-row secondary "
         "table of A7.9 (S2-1T, S2-1L, S2-L3a, S2-L3b; floor 0.05/4). A scenario is "
         "feasible at reference success p_ref iff P(win) ≤ 1 − p_ref and P(loss) ≤ p_ref.", "",
         "| scenario | n | m | d | true Δ | feasible | 0.05/4 | 0.05/3 | 0.05/2 | 0.05 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pw = r["power"]
        feas = "–" if r["feasible"] is None else ("yes" if r["feasible"] else "no")
        L.append(f"| {r['group']} | {r['n']} | {r['m'] * 100:g} pt | {r['d'] * 100:g} % | {r['delta'] * 100:+g} pt | {feas} | "
                 f"{pw['0.05/4']:.2f} | {pw['0.05/3']:.2f} | {pw['0.05/2']:.2f} | {pw['0.05/1']:.2f} |")
    L += ["", f"Runtime {res['runtime_s']:.0f} s.", ""]
    (a.out / "stage2_power.md").write_text("\n".join(L))
    print(f"done in {res['runtime_s']:.0f}s -> {a.out / 'stage2_power.md'}", file=sys.stderr)



def _amended_md(res: dict, path: Path) -> None:
    L = ["# Stage-2 power of the NI clauses as amended (A7.9-A7.16; exact, A6.21 rule `bb_ordered`)", "",
         f"`scripts/stage2_power.py` ({res['date']}): exact summation over the paired trinomial counts with the certified "
         "rejection regions of `scripts/ni_exact_size.CertifiedRule`. Planning statements, not results: the reference "
         "success and discordance d are assumptions. Family rows: the four Holm steps of S2-1..S2-4 (floor 0.05/4). "
         "Secondary rows: the six-row table S2-1T, S2-1L, S2-L3a, S2-L3b, S2-1X, S2-4S (A7.9 (c), A7.12 (g), A7.18 (f); "
         "floor 0.05/6). "
         "S2-L3a is single-attempt task success on A7.15's sample (planning n 640 / 800; the frozen n is added with "
         "`--telecom-n` before evaluation). A scenario is feasible at reference success p_ref iff P(win) ≤ 1 − p_ref and "
         "P(loss) ≤ p_ref.", "",
         "| scenario | table | n | m | d | true Δ | feasible | 0.05/6 | 0.05/5 | 0.05/4 | 0.05/3 | 0.05/2 | 0.05 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        pw = r["power"]
        feas = "–" if r["feasible"] is None else ("yes" if r["feasible"] else "no")
        six = f"{pw['0.05/6']:.2f}" if "0.05/6" in pw else "–"
        five = f"{pw['0.05/5']:.2f}" if "0.05/5" in pw else "–"
        L.append(f"| {r['group']} | {r['table']} | {r['n']} | {r['m'] * 100:g} pt | {r['d'] * 100:g} % | "
                 f"{r['delta'] * 100:+g} pt | {feas} | {six} | {five} | {pw['0.05/4']:.2f} | {pw['0.05/3']:.2f} | "
                 f"{pw['0.05/2']:.2f} | {pw['0.05/1']:.2f} |")
    L += ["", f"Runtime {res['runtime_s']:.0f} s.", ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
