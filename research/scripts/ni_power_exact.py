#!/usr/bin/env python
"""Exact power of the NI decision rules at the registered sizes, by exact summation over the paired trinomial counts
(no Monte Carlo): power = sum_{W + L <= n} Multinomial(n; p_win, p_loss, 1 - p_win - p_loss)(W, L) * 1[rule rejects],
p_win = (d + Delta) / 2, p_loss = (d - Delta) / 2 for discordance d and true difference Delta (losses-only rows:
p_win = 0, p_loss = d).

Round 8 (default; results/reviews/2026-10-02/ni_power_exact_r8.{json,md}):
  * rules: the A6.21 rule ``bb_ordered`` (metrics.ni_certified_p; exact rejection region from
    scripts/ni_exact_size.CertifiedRule) next to the superseded A6.17 (``registered_u``) and A6.14 (``registered``);
  * feasibility: a scenario is feasible against a reference success p_ref iff P(win) = P(arm succeeds, reference
    fails) <= 1 - p_ref and P(loss) <= p_ref. Rows are grouped by the pilot reference rates (Flash 0.98, ceiling 1.00,
    Flash-Lite 0.80, tau^2 0.80, C8 0.93); the three round-7 Flash rows that are infeasible at 0.98 are kept only as
    *generic* scenarios, with the largest reference success at which each is feasible;
  * conjunction power of the two-family primaries (C1, C2, C6: p = max(p_Jev, p_Intern) <= level iff both pass), at
    alpha = 0.05 and at the Holm floor 0.05/11, assuming both families face the same scenario and their tests are
    independent: pi_conj = pi^2. The Frechet bound min(pi_Jev, pi_Intern) = pi is the upper limit under any dependence;
    the shared reference arm makes the two tests positively dependent, which moves pi_conj from pi^2 towards pi.

    python scripts/ni_power_exact.py [--out results/reviews/2026-10-02] [--legacy]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ni_exact_size import LEVELS, CertifiedRule, ExactSize  # noqa: E402

# round 7 (A6.14 / A6.17; the paper's table before A6.21): (label, n, m, p_ref, d, delta, losses_only)
ROWS_LEGACY = [
    ("C1F/C2F/C6F (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.02, -0.01, False),
    ("", 500, 0.03, 0.98, 0.05, 0.0, False),
    ("", 500, 0.03, 0.98, 0.10, 0.0, False),
    ("", 500, 0.03, 0.98, 0.10, -0.01, False),
    ("ceiling ref. (losses only)", 500, 0.03, 1.0, 0.02, -0.02, True),
    ("", 500, 0.03, 1.0, 0.01, -0.01, True),
    ("C1L/C2L/C6L (Lite, ref. 0.80)", 500, 0.03, 0.80, 0.20, 0.0, False),
    ("", 500, 0.03, 0.80, 0.30, 0.0, False),
    ("", 500, 0.03, 0.80, 0.20, 0.03, False),
    ("", 500, 0.03, 0.80, 0.30, 0.05, False),
    ("C7 (tau^2, majority pass)", 149, 0.05, 0.80, 0.20, 0.0, False),
    ("", 149, 0.05, 0.80, 0.30, 0.03, False),
]
# round 8: (group, n, m, p_ref or None for generic, d, delta, losses_only)
ROWS = [
    ("C1F/C2F/C6F (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.02, 0.0, False),
    ("C1F/C2F/C6F (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.02, -0.01, False),
    ("C1F/C2F/C6F (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.04, 0.0, False),
    ("C1F/C2F/C6F (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.04, -0.01, False),
    ("C1F/C2F/C6F (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.05, -0.01, False),
    ("C1F/C2F/C6F (Flash, ref. 0.98)", 500, 0.03, 0.98, 0.06, -0.02, False),
    ("ceiling ref. (losses only, 1.00)", 500, 0.03, 1.0, 0.01, -0.01, True),
    ("ceiling ref. (losses only, 1.00)", 500, 0.03, 1.0, 0.02, -0.02, True),
    ("C1L/C2L/C6L (Lite, ref. 0.80)", 500, 0.03, 0.80, 0.20, 0.0, False),
    ("C1L/C2L/C6L (Lite, ref. 0.80)", 500, 0.03, 0.80, 0.30, 0.0, False),
    ("C1L/C2L/C6L (Lite, ref. 0.80)", 500, 0.03, 0.80, 0.20, 0.03, False),
    ("C1L/C2L/C6L (Lite, ref. 0.80)", 500, 0.03, 0.80, 0.30, 0.05, False),
    ("C7 (tau^2 majority pass, ref. 0.80)", 149, 0.05, 0.80, 0.20, 0.0, False),
    ("C7 (tau^2 majority pass, ref. 0.80)", 149, 0.05, 0.80, 0.30, 0.0, False),
    ("C7 (tau^2 majority pass, ref. 0.80)", 149, 0.05, 0.80, 0.30, 0.03, False),
    ("C8 (E1 typed question, ref. 0.93)", 1114, 0.03, 0.93, 0.06, 0.0, False),
    ("C8 (E1 typed question, ref. 0.93)", 1114, 0.03, 0.93, 0.10, 0.0, False),
    ("C8 (E1 typed question, ref. 0.93)", 1114, 0.03, 0.93, 0.10, 0.03, False),
    ("generic (no reference constraint), n = 500", 500, 0.03, None, 0.05, 0.0, False),
    ("generic (no reference constraint), n = 500", 500, 0.03, None, 0.10, 0.0, False),
    ("generic (no reference constraint), n = 500", 500, 0.03, None, 0.10, -0.01, False),
]
TWO_FAMILY = ("C1F/C2F/C6F", "C1L/C2L/C6L", "ceiling")    # the ceiling rows are Flash-pair scenarios at ref. 1.00
RULES = ["bb_ordered", "registered_u", "registered"]
POWER_LEVELS = ["0.05/1", "0.05/11"]


def scenario(d: float, delta: float, losses_only: bool) -> tuple[float, float]:
    return (0.0, d) if losses_only else ((d + delta) / 2, (d - delta) / 2)


def feasibility(p_win: float, p_loss: float, p_ref: float | None) -> dict[str, float | bool | None]:
    """Feasible against reference success p_ref iff P(win) <= 1 - p_ref and P(loss) <= p_ref (the 2x2 cells
    both = p_ref - P(loss) and neither = 1 - p_ref - P(win) are then non-negative); the interval of reference
    successes at which the scenario is feasible is [P(loss), 1 - P(win)]."""
    lo, hi = p_loss, 1.0 - p_win
    ok = None if p_ref is None else bool(lo - 1e-12 <= p_ref <= hi + 1e-12)
    return {"feasible": ok, "p_ref_min": lo, "p_ref_max": hi}


def legacy(out: Path, prune: float) -> None:
    """The round-7 table (A6.17 and A6.14 rules, results/reviews/2026-10-01d/ni_power_exact.{json,md})."""
    t0 = time.time()
    tabs: dict[tuple[int, float], ExactSize] = {}
    rows = []
    for label, n, m, p_ref, d, delta, lo in ROWS_LEGACY:
        if (n, m) not in tabs:
            ex = ExactSize(n, m)
            ex.bb_pvalues(prune)
            tabs[(n, m)] = ex
        ex = tabs[(n, m)]
        p_win, p_loss = scenario(d, delta, lo)
        lp = ex.logpmf_alt(p_win, p_loss)
        mask = lp > np.log(prune)
        pmf = np.exp(lp[mask])
        row = {"label": label, "n": n, "m": m, "p_ref": p_ref, "d": d, "delta": delta, "losses_only": lo,
               "p_win": p_win, "p_loss": p_loss, "mass": float(pmf.sum()), "power": {}}
        for rule in ("registered_u", "registered"):
            row["power"][rule] = {lv: float(pmf @ ex.rejects(rule, LEVELS[lv], mask)) for lv in POWER_LEVELS}
        rows.append(row)
    res = {"rows": rows, "rules": ["registered_u", "registered"], "levels": POWER_LEVELS, "prune": prune,
           "date": time.strftime("%Y-%m-%d"), "runtime_s": time.time() - t0, "command": "python " + " ".join(sys.argv)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "ni_power_exact.json").write_text(json.dumps(res, indent=1))
    L = ["# Exact power of the NI decision rules at the registered sizes (A6.17 rule, with the A6.14 rule next to it)", "",
         f"`scripts/ni_power_exact.py` ({res['date']}): exact summation over the paired trinomial counts of the rejection "
         "sets of `scripts/ni_exact_size.py` (no Monte Carlo; replaces the 20,000-replicate table of "
         "`results/reviews/2026-10-01c/ni_power_table.md`). Rows: the paper's Table tab:power (App. G). A6.17 = Berger-Boos "
         "exact unconditional p of Tango's score statistic on its branch (gamma = 1e-4) with the exact loss-count fallback at "
         "W = 0 or W + L < 5; A6.14 = the asymptotic p_T with the same fallback.", "",
         "| primary | n | m | d | true Δ | A6.17 α=0.05 | A6.17 floor 0.05/11 | A6.14 α=0.05 | A6.14 floor |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pu, pr = r["power"]["registered_u"], r["power"]["registered"]
        L.append(f"| {r['label']} | {r['n']} | {r['m'] * 100:g} | {r['d'] * 100:g}% | {r['delta'] * 100:+g} | "
                 f"{pu['0.05/1']:.2f} | {pu['0.05/11']:.2f} | {pr['0.05/1']:.2f} | {pr['0.05/11']:.2f} |")
    L += ["", f"Runtime {res['runtime_s']:.0f} s. Mass kept per row ≥ {min(r['mass'] for r in rows):.6f}.", ""]
    (out / "ni_power_exact.md").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--prune", type=float, default=1e-15)
    ap.add_argument("--eps-grid", type=float, default=0.005)
    ap.add_argument("--legacy", action="store_true", help="the round-7 table only")
    a = ap.parse_args()
    if a.legacy:
        legacy(a.out or ROOT / "results/reviews/2026-10-01d", a.prune)
        return
    out = a.out or ROOT / "results/reviews/2026-10-02"
    t0 = time.time()
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    levels = {lv: LEVELS[lv] for lv in POWER_LEVELS}
    old: dict[tuple[int, float], ExactSize] = {}
    new: dict[tuple[int, float], CertifiedRule] = {}
    rho_max = {}
    for _, n, m, _, d, _, _ in ROWS:
        rho_max[(n, m)] = max(rho_max.get((n, m), 0.0), d)
    rows = []
    for group, n, m, p_ref, d, delta, lo in ROWS:
        key = (n, m)
        if key not in old:
            ex = ExactSize(n, m)
            ex.bb_pvalues(a.prune)
            old[key] = ex
            log(f"[{time.time() - t0:6.1f}s] n={n} m={m}: A6.21 exact rejection regions")
            new[key] = CertifiedRule(n, m, levels, eps_grid=a.eps_grid, rho_max=min(1.0, rho_max[key] + 0.05), log=log)
        ex, cr = old[key], new[key]
        p_win, p_loss = scenario(d, delta, lo)
        lp = ex.logpmf_alt(p_win, p_loss)
        mask = lp > np.log(a.prune)
        pmf = np.exp(lp[mask])
        row = {"group": group, "n": n, "m": m, "p_ref": p_ref, "d": d, "delta": delta, "losses_only": lo,
               "p_win": p_win, "p_loss": p_loss, **feasibility(p_win, p_loss, p_ref), "mass": float(pmf.sum()),
               "power": {}}
        for rule in ("registered_u", "registered"):
            row["power"][rule] = {lv: float(pmf @ ex.rejects(rule, LEVELS[lv], mask)) for lv in POWER_LEVELS}
        pr = cr.probs(p_win, p_loss)
        row["power"]["bb_ordered"] = {lv: pr["reject"][lv] for lv in POWER_LEVELS}
        row["mass_beyond_Dmax"] = pr["mass_beyond_Dmax"]
        if group.split(" ")[0] in TWO_FAMILY:
            pi = row["power"]["bb_ordered"]
            row["conjunction"] = {lv: {"independent": pi[lv] ** 2, "upper": pi[lv]} for lv in POWER_LEVELS}
        rows.append(row)
        log(f"[{time.time() - t0:6.1f}s] {group} d={d} delta={delta:+.2f}: " +
            ", ".join(f"{r} {row['power'][r]['0.05/1']:.3f}/{row['power'][r]['0.05/11']:.3f}" for r in RULES))
    checks = {f"{n}/{m}": {**cr.info, "bracket_check": cr.check_brackets(per_level=4)} for (n, m), cr in new.items()}
    res = {"rows": rows, "rules": RULES, "levels": POWER_LEVELS, "prune": a.prune, "eps_grid": a.eps_grid,
           "checks": checks, "date": time.strftime("%Y-%m-%d"), "runtime_s": time.time() - t0,
           "command": "python " + " ".join(sys.argv)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "ni_power_exact_r8.json").write_text(json.dumps(res, indent=1))
    write_md(out / "ni_power_exact_r8.md", res)
    print(f"done in {res['runtime_s']:.0f}s -> {out / 'ni_power_exact_r8.md'}", file=sys.stderr)


def _feas(r: dict) -> str:
    if r["feasible"] is None:
        return f"generic (feasible for ref. ≤ {r['p_ref_max']:.3f})"
    return "yes" if r["feasible"] else f"**no** (needs ref. ≤ {r['p_ref_max']:.3f})"


def write_md(path: Path, res: dict) -> None:
    rows = res["rows"]
    L = ["# Exact power of the NI decision rules at the registered sizes (A6.21 rule; A6.17 / A6.14 for comparison)", "",
         f"`scripts/ni_power_exact.py` ({res['date']}), exact summation over the paired trinomial counts; raw numbers in "
         "`ni_power_exact_r8.json`. P(win) = (d + Δ)/2, P(loss) = (d − Δ)/2 (losses only: P(win) = 0, P(loss) = d). "
         "A6.21 = `bb_ordered` (`metrics.ni_certified_p`, its exact rejection region from "
         "`scripts/ni_exact_size.CertifiedRule`); A6.17 = grid-maximised p_U with the exact loss-count switch; "
         "A6.14 = asymptotic score p with the same switch. **Feasible**: a scenario is possible against reference "
         "success p_ref iff P(win) ≤ 1 − p_ref and P(loss) ≤ p_ref.", "",
         "## 1. Power per primary scenario", "",
         "| scenario group | n | m (pt) | d | true Δ (pt) | P(win) | feasible at the group's ref. | A6.21 α = 0.05 | A6.21 0.05/11 | "
         "A6.17 0.05 | A6.17 0.05/11 | A6.14 0.05 | A6.14 0.05/11 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    last = None
    for r in rows:
        g = r["group"] if r["group"] != last else ""
        last = r["group"]
        pb, pu, pr = r["power"]["bb_ordered"], r["power"]["registered_u"], r["power"]["registered"]
        L.append(f"| {g} | {r['n']} | {r['m'] * 100:g} | {r['d'] * 100:g}% | {r['delta'] * 100:+g} | {r['p_win']:.3f} | "
                 f"{_feas(r)} | {pb['0.05/1']:.3f} | {pb['0.05/11']:.3f} | {pu['0.05/1']:.3f} | {pu['0.05/11']:.3f} | "
                 f"{pr['0.05/1']:.3f} | {pr['0.05/11']:.3f} |")
    L += ["", "## 2. Two-family primaries (C1, C2, C6): conjunction power of the A6.21 rule", "",
          "The conjunction passes iff both families' p ≤ the level. Same scenario for both families; *independent*: "
          "π² (the two tests independent); *upper*: min(π_Jev, π_Intern) = π, the Fréchet bound under any dependence. "
          "Both families are compared with the same reference runs, which induces positive dependence; the "
          "conjunction power then lies between the two columns.", "",
          "| scenario group | d | true Δ (pt) | single family α = 0.05 | conjunction (indep.) 0.05 | single 0.05/11 | conjunction (indep.) 0.05/11 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "conjunction" not in r:
            continue
        c, pb = r["conjunction"], r["power"]["bb_ordered"]
        L.append(f"| {r['group']} | {r['d'] * 100:g}% | {r['delta'] * 100:+g} | {pb['0.05/1']:.3f} | "
                 f"{c['0.05/1']['independent']:.3f} | {pb['0.05/11']:.3f} | {c['0.05/11']['independent']:.3f} |")
    L += ["", "## 3. Reading", ""]
    L += reading(rows)
    L += ["", "## 4. Checks", ""]
    for key, c in res["checks"].items():
        bc = c["bracket_check"]
        L.append(f"- n/m = {key}: {c['n_nodes']:,} nuisance nodes, {c['n_gray']} gray cells decided by `ni_certified_p` "
                 f"(all converged: {c['gray_all_converged']}); bracket re-check on {bc['checked']} cells: "
                 f"{len(bc['violations'])} violation(s).")
    L.append(f"- Largest mass beyond the evaluated discordant counts: {max(r['mass_beyond_Dmax'] for r in rows):.1e}; "
             f"smallest kept mass (round-7 pruning): {min(r['mass'] for r in rows):.6f}.")
    L += ["", f"Runtime {res['runtime_s']:.0f} s. Command: `{res['command']}`.", ""]
    path.write_text("\n".join(L))


def reading(rows: list[dict]) -> list[str]:
    def get(group_prefix: str, d: float, delta: float) -> dict | None:
        return next((r for r in rows if r["group"].startswith(group_prefix) and abs(r["d"] - d) < 1e-9
                     and abs(r["delta"] - delta) < 1e-9), None)
    out = []
    f1, f0 = get("C1F", 0.02, -0.01), get("C1F", 0.04, 0.0)
    if f1 and f0:
        out.append(f"1. **Flash pairs (ref. 0.98).** Feasible scenarios need P(win) ≤ 0.02, so d + Δ ≤ 4 pt. At d = 2 %, Δ = −1 pt "
                   f"the A6.21 rule has power {f1['power']['bb_ordered']['0.05/1']:.2f} at α = 0.05 and "
                   f"{f1['power']['bb_ordered']['0.05/11']:.2f} at the Holm floor; at d = 4 %, Δ = 0 (every reference failure "
                   f"becomes a win) {f0['power']['bb_ordered']['0.05/1']:.2f} and {f0['power']['bb_ordered']['0.05/11']:.2f}. "
                   "The round-7 rows d = 5 % and 10 % at Δ ≥ −1 pt are infeasible at this reference rate and are shown only "
                   "as generic scenarios.")
    c2 = get("ceiling", 0.02, -0.02)
    if c2:
        out.append(f"2. **Ceiling reference.** A persistent 2-pt loss (losses only) is certified with power "
                   f"{c2['power']['bb_ordered']['0.05/1']:.2f} at α = 0.05 and {c2['power']['bb_ordered']['0.05/11']:.2f} at the floor.")
    l0, l3 = get("C1L", 0.20, 0.0), get("C1L", 0.20, 0.03)
    if l0 and l3:
        out.append(f"3. **Flash-Lite pairs (ref. 0.80).** At Δ = 0 with d = 20 % power is {l0['power']['bb_ordered']['0.05/1']:.2f} / "
                   f"{l0['power']['bb_ordered']['0.05/11']:.2f} (α = 0.05 / floor); only a 3-pt gain gives "
                   f"{l3['power']['bb_ordered']['0.05/1']:.2f} / {l3['power']['bb_ordered']['0.05/11']:.2f}.")
    t0 = get("C7", 0.20, 0.0)
    if t0:
        out.append(f"4. **C7 (n = 149, m = 5 pt).** At Δ = 0, d = 20 %: {t0['power']['bb_ordered']['0.05/1']:.2f} / "
                   f"{t0['power']['bb_ordered']['0.05/11']:.2f}.")
    c8 = get("C8", 0.10, 0.0)
    if c8:
        out.append(f"5. **C8 (n = 1,114).** At Δ = 0, d = 10 %: {c8['power']['bb_ordered']['0.05/1']:.2f} / "
                   f"{c8['power']['bb_ordered']['0.05/11']:.2f}.")
    conj = [r for r in rows if "conjunction" in r]
    if conj:
        best = max(conj, key=lambda r: r["conjunction"]["0.05/11"]["independent"])
        out.append(f"6. **Conjunction.** Under independence the best two-family scenario reaches "
                   f"{best['conjunction']['0.05/1']['independent']:.2f} at α = 0.05 and {best['conjunction']['0.05/11']['independent']:.2f} "
                   f"at the floor ({best['group']}, d = {best['d'] * 100:g} %, Δ = {best['delta'] * 100:+g} pt); the Flash-Lite "
                   f"conjunctions at Δ = 0 are ≤ {max(r['conjunction']['0.05/1']['independent'] for r in conj if r['group'].startswith('C1L') and r['delta'] == 0):.2f}. "
                   "No primary is adequately powered in every plausible scenario; 'adequate' is withdrawn.")
    diff = max(abs(r["power"]["bb_ordered"][lv] - r["power"]["registered_u"][lv]) for r in rows for lv in POWER_LEVELS)
    out.append(f"7. **Cost of the certified rule.** The largest change in power against A6.17 over all rows and both levels is "
               f"{diff:.3f}.")
    return out


if __name__ == "__main__":
    main()
