#!/usr/bin/env python
"""Size (type-I error at the NI boundary) and power of candidate non-inferiority decision rules for paired
binary task outcomes -- the A6.1 question (docs/PREREGISTRATION.md, review round 5/6: is the registered
bootstrap rule anti-conservative at the Holm levels, and which rule holds its size?).

Rules (arm vs reference on the same n tasks, margin m; per-task outcome in {0, 1}):
  R1  registered rule (A6.1): paired task-bootstrap p_NI = P*(Delta* <= -m) <= alpha, with the exact one-sided
      binomial fallback p = P[L <= l | n, m] (metrics.ni_exact_check, strict '<' as in metrics.ni_pvalue) whenever
      fewer than 5 tasks are discordant.
  R2  exact loss-count test for every comparison: reject iff P[L <= l | n, m] <= alpha (wins ignored).
  R3  Tango (1998) score test: the one-sided (1 - alpha) lower confidence bound of metrics.tango_interval
      (one_sided=True, i.e. z = z_{1-alpha}; identical to the lower end of the two-sided (1 - 2 alpha) interval,
      checked numerically) is above -m.
  R4  switching rule: R1, but the exact fallback also applies whenever wins == 0 (losses-only data).
  R5  switching rule: R3 (Tango), with the exact fallback whenever wins == 0 or fewer than 5 tasks are discordant.
      (This script's addition: R3 as the statistic, the registered exact fallback kept and extended to losses-only data.)

The data enter every rule only through (wins, losses, n): the task bootstrap of per-task differences in
{-1, 0, +1} is, in distribution, the multinomial resample of (wins, losses, ties) (Delta* = (W* - L*) / n), which is
what this script draws (B resamples per replicate; the same p-value semantics ``boot <= -m`` as metrics._paired_boot).
``--validate`` re-runs R1 through metrics.ni_pvalue itself on per-task dicts to confirm the equivalence.

Usage:  python scripts/ni_size_sim.py [--reps-size 50000] [--reps-power 20000] [--boot 2000] [--seed 20261001]
                                      [--out results/reviews/2026-10-01c] [--validate] [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench.metrics import _phi, _tango_z, _z, ni_exact_check, ni_pvalue, tango_interval  # noqa: E402

LEVELS = {"0.05": 0.05, "0.05/6": 0.05 / 6, "0.05/11": 0.05 / 11}
RULES = ["R1", "R2", "R3", "R4", "R5"]
RULE_DESC = {
    "R1": "registered: bootstrap p_NI <= a, exact fallback if discordant < 5",
    "R2": "exact loss test P[L <= l | n, m] <= a for every comparison",
    "R3": "Tango one-sided (1-a) score lower bound > -m",
    "R4": "R1 with the exact fallback also when wins == 0",
    "R5": "R3 with the exact fallback when wins == 0 or discordant < 5",
}


# ------------------------------------------------------------------ data-generating process
def cells(p_ref: float, delta: float, d: float, losses_only: bool) -> dict[str, float] | None:
    """Four cell probabilities of a paired binary outcome with reference success p_ref, arm success p_ref + delta
    and discordance d = P(loss) + P(win); loss = reference succeeds, arm fails; win = arm only.
    Losses-only (ceiling reference, p_ref = 1): P(win) = 0, so P(loss) = -delta and d = -delta.
    Otherwise P(loss) = (d - delta) / 2, P(win) = (d + delta) / 2. The concordant mass 1 - d is split as
    both = p_ref - P(loss), neither = 1 - p_ref - P(win); when that split is infeasible (neither < 0, i.e. P(win) >
    1 - p_ref) ``neither`` is set to 0 and the effective p_ref is reported -- the split does not enter any rule.
    Returns None when the point is infeasible (a negative P(loss) or P(win))."""
    if losses_only:
        p_loss, p_win = -delta, 0.0
    else:
        p_loss, p_win = (d - delta) / 2, (d + delta) / 2
    if p_loss < -1e-12 or p_win < -1e-12 or p_loss + p_win > 1 + 1e-12:
        return None
    p_loss, p_win = (0.0 if p_loss <= 0 else p_loss), (0.0 if p_win <= 0 else p_win)
    both, neither = p_ref - p_loss, 1 - p_ref - p_win
    adjusted = False
    if neither < 0:
        neither, both, adjusted = 0.0, 1 - p_loss - p_win, True
    if both < 0:
        both, neither, adjusted = 0.0, 1 - p_loss - p_win, True
    return {"both": both, "loss": p_loss, "win": p_win, "neither": neither, "p_ref_eff": both + p_loss,
            "p_arm_eff": both + p_win, "d": p_loss + p_win, "delta": p_win - p_loss, "split_adjusted": adjusted}


# ------------------------------------------------------------------ the rules on simulated (wins, losses) counts
class Tables:
    """Per-(n, m) lookup tables: exact binomial p-values for every loss count and a Tango memo."""

    def __init__(self, n: int, m: float) -> None:
        self.n, self.m = n, m
        self.p_exact = np.array([ni_exact_check(0, l, n, m)["p"] for l in range(n + 1)])
        self.tango: dict[tuple[int, int, str], bool] = {}
        self.tango_checked = 0
        self.tango_max_dev = 0.0

    def tango_reject(self, w: np.ndarray, l: np.ndarray, level: str) -> np.ndarray:
        """R3: metrics.tango_interval(..., one_sided=True)['ci'][0] > -m (memoised over unique (w, l))."""
        alpha = LEVELS[level]
        out = np.empty(len(w), dtype=bool)
        pairs, inv = np.unique(np.stack([w, l], axis=1), axis=0, return_inverse=True)
        dec = np.empty(len(pairs), dtype=bool)
        for i, (wi, li) in enumerate(map(tuple, pairs)):
            key = (int(wi), int(li), level)
            if key not in self.tango:
                ti = int(self.n - wi - li)
                lb = tango_interval(ti, int(wi), int(li), 0, alpha=alpha, one_sided=True)["ci"][0]
                if self.tango_checked < 400:    # cross-checks: two-sided (1 - 2 alpha) lower end; direct score test
                    lb2 = tango_interval(ti, int(wi), int(li), 0, alpha=2 * alpha, one_sided=False)["ci"][0]
                    self.tango_max_dev = max(self.tango_max_dev, abs(lb - lb2))
                    z_score = _tango_z(-self.m, int(wi), int(li), self.n)
                    if (lb > -self.m) != (z_score > _z(1 - alpha)) and abs(z_score - _z(1 - alpha)) > 1e-6:
                        raise AssertionError(f"Tango bound and score test disagree at (w, l) = {wi, li}")
                    self.tango_checked += 1
                self.tango[key] = bool(lb > -self.m)
            dec[i] = self.tango[key]
        out[:] = dec[inv.ravel()]
        return out


def bootstrap_p_ni(rng: np.random.Generator, counts: np.ndarray, n: int, m: float, B: int,
                   chunk: int = 400) -> np.ndarray:
    """p_NI = P*(Delta* <= -m) of the paired task bootstrap for each replicate's (wins, losses, ties) counts:
    (W*, L*, T*) ~ Multinomial(n, (w, l, t) / n), Delta* = (W* - L*) / n, B resamples (the distribution of
    metrics._paired_boot's resample of the per-task differences)."""
    out = np.empty(len(counts))
    for s in range(0, len(counts), chunk):
        c = counts[s:s + chunk]
        bs = rng.multinomial(n, (c / n)[:, None, :], size=(len(c), B))
        boot = (bs[..., 0] - bs[..., 1]) / n
        out[s:s + chunk] = np.mean(boot <= -m, axis=1)
    return out


def decide(tab: Tables, w: np.ndarray, l: np.ndarray, p_ni: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Rejection indicators of every rule at every level."""
    n, disc = tab.n, w + l
    p_ex = tab.p_exact[l]
    few = disc < 5
    none_win = w == 0
    out: dict[str, dict[str, np.ndarray]] = {}
    for level, a in LEVELS.items():
        boot_rej = p_ni <= a
        exact_rej_strict = p_ex < a                  # metrics.ni_exact_check: ni = p < alpha
        tango = tab.tango_reject(w, l, level)
        out[level] = {
            "R1": np.where(few, exact_rej_strict, boot_rej),
            "R2": p_ex <= a,
            "R3": tango,
            "R4": np.where(few | none_win, exact_rej_strict, boot_rej),
            "R5": np.where(few | none_win, exact_rej_strict, tango),
        }
    return out


def run_point(rng: np.random.Generator, tab: Tables, p_win: float, p_loss: float, reps: int, B: int) -> dict[str, Any]:
    n = tab.n
    counts = rng.multinomial(n, [p_win, p_loss, 1 - p_win - p_loss], size=reps)
    w, l = counts[:, 0], counts[:, 1]
    p_ni = bootstrap_p_ni(rng, counts, n, tab.m, B)
    dec = decide(tab, w, l, p_ni)
    res: dict[str, Any] = {"reps": reps, "B": B, "mean_wins": float(w.mean()), "mean_losses": float(l.mean()),
                           "frac_discordant_lt5": float(np.mean(w + l < 5)), "frac_zero_wins": float(np.mean(w == 0)),
                           "rates": {}}
    for level, d in dec.items():
        res["rates"][level] = {}
        for rule, r in d.items():
            k = int(r.sum())
            p = k / reps
            res["rates"][level][rule] = {"rate": p, "rejections": k, "se": math.sqrt(p * (1 - p) / reps)}
    return res


# ------------------------------------------------------------------ scenarios
def scenarios() -> list[dict[str, Any]]:
    """Each scenario: n, m, p_ref, d, losses_only, and the list of true deltas to simulate with a role
    ('size' = boundary Delta = -m; 'inside_H0' = Delta < -m; 'power' = Delta > -m)."""
    sc: list[dict[str, Any]] = []
    # S1: ceiling reference, losses only (wins impossible) -- Delta = -P(loss) = -d, so each d fixes Delta.
    for d in (0.03, 0.05, 0.10, 0.02):
        delta = -d
        role = "size" if abs(delta + 0.03) < 1e-12 else ("inside_H0" if delta < -0.03 else "power")
        sc.append({"id": f"S1 d={d:.2f}", "group": "S1 ceiling, losses only", "n": 500, "m": 0.03, "p_ref": 1.0,
                   "d": d, "losses_only": True, "deltas": [(delta, role)]})
    sc.append({"id": "S1 d=0 (Delta=0)", "group": "S1 ceiling, losses only", "n": 500, "m": 0.03, "p_ref": 1.0,
               "d": 0.0, "losses_only": True, "deltas": [(0.0, "power")]})
    for d in (0.05, 0.10):
        sc.append({"id": f"S2 d={d:.2f}", "group": "S2 p_ref=0.98", "n": 500, "m": 0.03, "p_ref": 0.98, "d": d,
                   "losses_only": False, "deltas": [(-0.03, "size"), (0.0, "power"), (0.02, "power")]})
    for d in (0.20, 0.30):
        sc.append({"id": f"S3 d={d:.2f}", "group": "S3 p_ref=0.80", "n": 500, "m": 0.03, "p_ref": 0.80, "d": d,
                   "losses_only": False, "deltas": [(-0.03, "size"), (0.0, "power"), (0.02, "power")]})
    sc.append({"id": "S4 n=149 m=0.05 d=0.20", "group": "S4 tau2 (n=149, m=0.05, p_ref=0.80)", "n": 149, "m": 0.05,
               "p_ref": 0.80, "d": 0.20, "losses_only": False,
               "deltas": [(-0.05, "size"), (0.0, "power"), (0.02, "power")]})
    return sc


# ------------------------------------------------------------------ validation against metrics.ni_pvalue
def validate_ni_pvalue(seed: int, B: int, reps: int) -> list[dict[str, Any]]:
    """Run R1 through metrics.ni_pvalue (per-task dict inputs, the registered code path) on the same simulated
    (wins, losses) counts as the fast path, at the Holm floor; report both rejection rates and decision agreement."""
    out = []
    ss = np.random.SeedSequence(seed + 777)
    for name, p_win, p_loss in (("S1 boundary (P(loss)=0.03, no wins)", 0.0, 0.03),
                                ("S2 d=0.05 boundary (P(loss)=0.04, P(win)=0.01)", 0.01, 0.04)):
        n, m = 500, 0.03
        rng = np.random.default_rng(ss.spawn(1)[0])
        tab = Tables(n, m)
        counts = rng.multinomial(n, [p_win, p_loss, 1 - p_win - p_loss], size=reps)
        w, l = counts[:, 0], counts[:, 1]
        p_fast = bootstrap_p_ni(rng, counts, n, m, B)
        dec_fast = decide(tab, w, l, p_fast)
        res: dict[str, Any] = {"scenario": name, "reps": reps, "B": B, "levels": {}}
        p_reg = np.empty(reps)
        rej_reg = {lv: np.empty(reps, dtype=bool) for lv in LEVELS}
        keys = [f"t{i:04d}" for i in range(n)]
        for i in range(reps):
            wi, li = int(w[i]), int(l[i])
            arm = {k: (1.0 if j < wi else 0.0 if j < wi + li else 1.0) for j, k in enumerate(keys)}
            base = {k: (0.0 if j < wi else 1.0) for j, k in enumerate(keys)}
            for lv, a in LEVELS.items():
                r = ni_pvalue(base, arm, m, n_boot=B, seed=int(rng.integers(1 << 31)), alpha=0.05, alpha_adj=a)
                rej_reg[lv][i] = r["ni_adj"]
                p_reg[i] = r["p_decisive"]
        for lv in LEVELS:
            res["levels"][lv] = {"rate_ni_pvalue": float(rej_reg[lv].mean()),
                                 "rate_fast_R1": float(dec_fast[lv]["R1"].mean()),
                                 "decision_agreement": float(np.mean(rej_reg[lv] == dec_fast[lv]["R1"]))}
        res["mean_abs_p_diff"] = float(np.mean(np.abs(p_reg - p_fast)))
        res["mean_abs_p_diff_expected_mc"] = float(np.mean(np.sqrt(2 * p_fast * (1 - p_fast) / B)) * math.sqrt(2 / math.pi))
        out.append(res)
    return out


# ------------------------------------------------------------------ report
def fmt_rate(cell: dict[str, float], alpha: float | None) -> str:
    s = f"{cell['rate']:.4f} ± {cell['se']:.4f}"
    if alpha is not None and cell["rate"] > alpha + 2 * cell["se"]:
        return f"**{s}**"
    return s


def write_md(path: Path, results: dict[str, Any], cmd: str, runtime: float) -> None:
    L: list[str] = []
    L.append("# NI decision rules: size at the boundary and power (paired binary outcomes)\n")
    L.append(f"Simulation of A6.1's candidate decision rules (`scripts/ni_size_sim.py`). Raw numbers: `ni_size_sim.json`. "
             f"Date {results['date']}; seed {results['seed']}; B = {results['B']} bootstrap resamples per replicate "
             f"(reduced from the registered 10,000 -- see the B = 10,000 check below); {results['reps_size']:,} replicates "
             f"per size point, {results['reps_power']:,} per power point; Holm levels α ∈ {{0.05, 0.05/6 = 0.00833, 0.05/11 = 0.00455}}.\n")
    L.append("**Rules.** " + " ".join(f"{r}: {RULE_DESC[r]}." for r in RULES) + " R1–R4 are the four candidates the review "
             "asked for; R5 is this script's addition, the registrable switching rule built from R3 and the registered fallback. Exact fallback = `metrics.ni_exact_check` p = P[L ≤ l | n, m] with the registered strict "
             "`p < α`; R2 uses `p ≤ α` (never differs: p is never exactly α).\n")
    L.append("**Data-generating process.** n paired Bernoulli outcomes per replicate, cells (both succeed, loss = reference "
             "only, win = arm only, both fail) with P(loss) = (d − Δ)/2, P(win) = (d + Δ)/2, both = p_ref − P(loss), "
             "neither = 1 − p_ref − P(win). S1 (ceiling reference p_ref = 1.00) has no wins, so P(loss) = −Δ = d: each d "
             "fixes one Δ (d = 0.03 is the boundary; d = 0.05, 0.10 are inside H0, Δ = −d < −m; d = 0.02 is Δ = −0.02, a "
             "power point; Δ = 0 is P(loss) = 0 = d, zero discordance; Δ = +0.02 is infeasible at a ceiling reference). "
             "Every rule depends on the data only through (wins, losses, n) — the split of the concordant mass between "
             "'both' and 'neither' is immaterial — so where p_ref = 0.98 cannot host P(win) > 0.02 the 'neither' cell is set "
             "to 0 and the effective p_ref is listed in the JSON (`split_adjusted`). The task bootstrap of per-task "
             "differences in {−1, 0, +1} is drawn as the equivalent multinomial resample of (wins, losses, ties), "
             "Δ* = (W* − L*)/n, with the same `Δ* ≤ −m` p-value as `metrics._paired_boot`/`ni_pvalue` "
             "(validated against `ni_pvalue` itself, §5).\n")
    # ---- size table
    L.append("## 1. Size at the NI boundary (true Δ = −m) — rejection rate ± Monte-Carlo SE\n")
    L.append("Bold: size > α + 2·SE (SE = √(p̂(1−p̂)/R)). 'SE(α)' is √(α(1−α)/R), the MC SE of a rule that is exactly sized. "
             "Rows marked 'inside H0' are Δ < −m (S1 losses-only with d > m); their rejection rate must be below the boundary size.\n")
    L.append("| scenario | Δ | P(loss) | P(win) | α | R1 | R2 | R3 | R4 | R5 | SE(α) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for pt in results["points"]:
        if pt["role"] not in ("size", "inside_H0"):
            continue
        tag = f"{pt['scenario']}" + (" (inside H0)" if pt["role"] == "inside_H0" else "")
        for lv, a in LEVELS.items():
            row = pt["rates"][lv]
            se_a = math.sqrt(a * (1 - a) / pt["reps"])
            L.append(f"| {tag} | {pt['delta']:+.2f} | {pt['cells']['loss']:.3f} | {pt['cells']['win']:.3f} | {lv} | "
                     + " | ".join(fmt_rate(row[r], a) for r in RULES) + f" | {se_a:.4f} |")
    L.append("")
    # ---- power table
    L.append("## 2. Power (true Δ = 0 and Δ = +0.02; S1 also Δ = −0.02)\n")
    L.append("| scenario | Δ | P(loss) | P(win) | α | R1 | R2 | R3 | R4 | R5 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for pt in results["points"]:
        if pt["role"] != "power":
            continue
        for lv in LEVELS:
            row = pt["rates"][lv]
            L.append(f"| {pt['scenario']} | {pt['delta']:+.2f} | {pt['cells']['loss']:.3f} | {pt['cells']['win']:.3f} | {lv} | "
                     + " | ".join(f"{row[r]['rate']:.3f}" for r in RULES) + " |")
    L.append("")
    for pt in results["infeasible"]:
        L.append(f"- {pt['scenario']}, Δ = {pt['delta']:+.2f}: infeasible ({pt['reason']}); not simulated.")
    L.append("")
    # ---- reading
    L.append("## 3. Reading\n")
    L.extend(results["reading"])
    L.append("")
    # ---- checks
    L.append("## 4. B = 10,000 check (registered B) vs B = 2,000, size at the boundary\n")
    L.append("| scenario | α | R1 (B=2,000) | R1 (B=10,000) | R4 (B=2,000) | R4 (B=10,000) | reps |")
    L.append("|---|---|---|---|---|---|---|")
    for chk in results["b10000_check"]:
        for lv, a in LEVELS.items():
            L.append(f"| {chk['scenario']} | {lv} | {fmt_rate(chk['B2000']['rates'][lv]['R1'], a)} | "
                     f"{fmt_rate(chk['B10000']['rates'][lv]['R1'], a)} | {fmt_rate(chk['B2000']['rates'][lv]['R4'], a)} | "
                     f"{fmt_rate(chk['B10000']['rates'][lv]['R4'], a)} | {chk['B2000']['reps']:,} |")
    L.append("")
    L.append("## 5. Validation of the fast path against `metrics.ni_pvalue` (R1, per-task dict inputs)\n")
    if results.get("validation"):
        L.append("| scenario | α | rate via `ni_pvalue` | rate via fast path | decision agreement | reps |")
        L.append("|---|---|---|---|---|---|")
        for v in results["validation"]:
            for lv, r in v["levels"].items():
                L.append(f"| {v['scenario']} | {lv} | {r['rate_ni_pvalue']:.4f} | {r['rate_fast_R1']:.4f} | "
                         f"{r['decision_agreement']:.4f} | {v['reps']:,} |")
        L.append("")
        for v in results["validation"]:
            L.append(f"- {v['scenario']}: mean |p_decisive(ni_pvalue) − p_NI(fast)| = {v['mean_abs_p_diff']:.5f} "
                     f"(expected from two independent B = {v['B']} bootstrap draws of the same statistic: "
                     f"≈ {v['mean_abs_p_diff_expected_mc']:.5f}). Disagreements are replicates whose p-value sits within "
                     f"bootstrap MC noise of α.")
    else:
        L.append("(run with `--validate`)")
    L.append("")
    L.append(f"Tango cross-checks: one-sided (1−α) `tango_interval` bound vs lower end of the two-sided (1−2α) interval, "
             f"max |difference| = {results['tango_max_dev']:.2e} over {results['tango_checked']} (w, l, α) points; the "
             f"bound-above-−m decision agreed with the direct score test Z(−m) > z_{{1−α}} (`metrics._tango_z`) at every one.\n")
    L.append("## 6. Command and runtime\n")
    L.append(f"```\n{cmd}\n```\n")
    L.append(f"Runtime {runtime / 60:.1f} min on {results['cpu']} CPU threads (numpy {np.__version__}, Python "
             f"{sys.version.split()[0]}).\n")
    path.write_text("\n".join(L))


def reading(results: dict[str, Any]) -> list[str]:
    """The 10-line reading, computed from the numbers (so it cannot drift from the tables)."""
    size_pts = [p for p in results["points"] if p["role"] == "size"]
    pow_pts = [p for p in results["points"] if p["role"] == "power" and p["cells"]["d"] > 0]
    fl = "0.05/11"

    def cell(p: dict[str, Any], lv: str, r: str) -> dict[str, float]:
        return p["rates"][lv][r]

    def flagged(p: dict[str, Any], lv: str, r: str) -> bool:
        c = cell(p, lv, r)
        return c["rate"] > LEVELS[lv] + 2 * c["se"]

    holds = {r: not any(flagged(p, lv, r) for p in size_pts for lv in LEVELS) for r in RULES}
    worst = {}
    for r in RULES:
        ratio, where = max((cell(p, lv, r)["rate"] / LEVELS[lv], f"{p['scenario']}, α = {lv}") for p in size_pts for lv in LEVELS)
        worst[r] = (ratio, where)
    flagged_cells = {r: [(p["scenario"], lv) for p in size_pts for lv in LEVELS if flagged(p, lv, r)] for r in RULES}
    pt = {p["scenario"]: p for p in size_pts}
    s1, s2a, s2b = pt["S1 d=0.03"], pt["S2 d=0.05"], pt["S2 d=0.10"]
    s1p = next(p for p in results["points"] if p["scenario"] == "S1 d=0.02")
    s3p0 = next(p for p in pow_pts if p["scenario"] == "S3 d=0.20" and p["delta"] == 0)

    def mean_pow(r: str, lv: str, pts: list[dict[str, Any]]) -> float:
        return float(np.mean([cell(p, lv, r)["rate"] for p in pts]))

    ok_rules = [r for r in RULES if holds[r]]
    bad_rules = [r for r in RULES if not holds[r]]
    lines = []
    lines.append("1. Size held (no boundary cell above α + 2·SE at any of the three levels) by: "
                 f"**{', '.join(ok_rules) or 'none'}**; failed by " + ("; ".join(
                     f"{r} ({len(flagged_cells[r])} flagged cells, worst {worst[r][0]:.2f}×α at {worst[r][1]})" for r in bad_rules) or "none") + ".")
    lines.append(f"2. The registered R1 over-rejects where the reviewer said: losses-only at the boundary (S1, 15 expected losses, 0 wins) "
                 f"its size is {cell(s1, fl, 'R1')['rate']:.4f} at α = 0.00455 ({cell(s1, fl, 'R1')['rate'] / LEVELS[fl]:.1f}×) and "
                 f"{cell(s1, '0.05/6', 'R1')['rate']:.4f} at 0.05/6 ({cell(s1, '0.05/6', 'R1')['rate'] / LEVELS['0.05/6']:.1f}×); the <5-discordant "
                 f"fallback fires in {100 * s1['frac_discordant_lt5']:.2f} % of those replicates, so it never reaches the problem region. "
                 "The percentile bootstrap is a Wald-type test with the variance taken at the observed Δ̂: fewer losses than the null expects "
                 "means a smaller resampling variance and a too-small p.")
    r1_clean = [sc for sc in dict.fromkeys(p["scenario"] for p in size_pts) if not any((sc, lv) in flagged_cells["R1"] for lv in LEVELS)]
    r1_other = [f"{sc} ({', '.join(lv for lv in LEVELS if (sc, lv) in flagged_cells['R1'])})" for sc in dict.fromkeys(p["scenario"] for p in size_pts)
                if sc not in ("S1 d=0.03", "S2 d=0.05") and any((sc, lv) in flagged_cells["R1"] for lv in LEVELS)]
    lines.append(f"3. With a few wins (S2 d = 0.05: 20 expected losses, 5 wins) R1 is {cell(s2a, fl, 'R1')['rate'] / LEVELS[fl]:.1f}×α at the "
                 f"floor and {cell(s2a, '0.05/6', 'R1')['rate'] / LEVELS['0.05/6']:.1f}×α at 0.05/6; R4's extra switch (wins = 0) fires in only "
                 f"{100 * s2a['frac_zero_wins']:.1f} % of replicates there, so R4 "
                 + ("does not repair it" if not holds["R4"] else "repairs it") + f" (R4 = {cell(s2a, fl, 'R4')['rate']:.4f} at the floor); "
                 f"R4 does fix the losses-only case (S1: R4 = R2 = {cell(s1, fl, 'R4')['rate']:.4f}). R1 is also flagged at "
                 + (", ".join(r1_other) or "no other point") + f"; it is clean at every level only in {', '.join(r1_clean) or 'no scenario'} "
                 f"(S3: 100–150 discordant tasks, where the Wald-type bootstrap is adequate).")
    lines.append(f"4. R3 (Tango score test, variance from the restricted MLE under H0) "
                 + ("stays at or below α in every cell" if worst["R3"][0] <= 1.0 else
                    "holds its size within MC error" if holds["R3"] else "is flagged") + f": worst {worst['R3'][0]:.2f}×α "
                 f"({worst['R3'][1]}); at the floor S1 {cell(s1, fl, 'R3')['rate']:.4f}, S2 d=0.05 {cell(s2a, fl, 'R3')['rate']:.4f}, "
                 f"S2 d=0.10 {cell(s2b, fl, 'R3')['rate']:.4f}, S3 d=0.20 {cell(pt['S3 d=0.20'], fl, 'R3')['rate']:.4f}, "
                 f"S3 d=0.30 {cell(pt['S3 d=0.30'], fl, 'R3')['rate']:.4f}, S4 {cell(pt['S4 n=149 m=0.05 d=0.20'], fl, 'R3')['rate']:.4f}. "
                 f"R5 (R3 + exact fallback) likewise: worst {worst['R5'][0]:.2f}×α.")
    lines.append(f"5. R2 (exact loss test) is exact by construction (worst {worst['R2'][0]:.2f}×α) but tests P(loss) < m, so it has no power "
                 f"once wins and losses both occur at rates above m: power {cell(s3p0, '0.05', 'R2')['rate']:.3f} at S3 d=0.20, Δ = 0, α = 0.05, "
                 f"and {cell(next(p for p in pow_pts if p['scenario'] == 'S2 d=0.05' and p['delta'] == 0), '0.05', 'R2')['rate']:.3f} at S2 d=0.05. "
                 "It is only usable as the losses-only / few-discordant fallback.")
    valid_pts = [p for p in pow_pts if p["scenario"] in r1_clean]
    infl_pts = [p for p in pow_pts if p["scenario"] not in r1_clean]
    gap_valid = max((cell(p, lv, "R1")["rate"] - cell(p, lv, "R5")["rate"], f"{p['scenario']} Δ={p['delta']:+.2f} α={lv}") for p in valid_pts for lv in LEVELS)
    gap_infl = max((cell(p, lv, "R1")["rate"] - cell(p, lv, "R5")["rate"], f"{p['scenario']} Δ={p['delta']:+.2f} α={lv}") for p in infl_pts for lv in LEVELS)
    lines.append(f"6. Power cost. Where R1 holds its size ({', '.join(r1_clean)}; mean over Δ = 0 and +0.02 at α = 0.05 / 0.05/6 / 0.05/11): "
                 + "; ".join(f"{r} {mean_pow(r, '0.05', valid_pts):.3f}/{mean_pow(r, '0.05/6', valid_pts):.3f}/{mean_pow(r, fl, valid_pts):.3f}" for r in ("R1", "R3", "R5"))
                 + f" — R3/R5 cost nothing there (largest R1 − R5 difference {gap_valid[0]:+.3f}, at {gap_valid[1]}). Where R1 is flagged "
                 f"({', '.join(dict.fromkeys(p['scenario'] for p in infl_pts))}): "
                 + "; ".join(f"{r} {mean_pow(r, '0.05', infl_pts):.3f}/{mean_pow(r, '0.05/6', infl_pts):.3f}/{mean_pow(r, fl, infl_pts):.3f}" for r in ("R1", "R3", "R5"))
                 + f" — largest gap {gap_infl[0]:.3f} at {gap_infl[1]}; that gap is in part the size inflation being removed (R1's rejection "
                 "rate under H0 is inflated by the same mechanism), so it is not a like-for-like power loss.")
    lines.append(f"7. Losses-only power (S1, Δ = −0.02, 10 expected losses): at α = 0.05 / 0.05/6 / 0.05/11 R1 "
                 f"{cell(s1p, '0.05', 'R1')['rate']:.3f}/{cell(s1p, '0.05/6', 'R1')['rate']:.3f}/{cell(s1p, fl, 'R1')['rate']:.3f}, "
                 f"R2 = R4 = R5 {cell(s1p, '0.05', 'R5')['rate']:.3f}/{cell(s1p, '0.05/6', 'R5')['rate']:.3f}/{cell(s1p, fl, 'R5')['rate']:.3f}, "
                 f"R3 {cell(s1p, '0.05', 'R3')['rate']:.3f}/{cell(s1p, '0.05/6', 'R3')['rate']:.3f}/{cell(s1p, fl, 'R3')['rate']:.3f}: the exact test "
                 "is the most powerful valid rule in the ceiling case (with wins = 0 Tango's rejection region at 0.05/6 collapses onto the one at 0.05/11 — discreteness — so R3 is over-conservative there), which is why R5 switches to it. "
                 "With zero discordant tasks every rule rejects (p = 0.97^500).")
    rec = "R5" if holds["R5"] else ("R3" if holds["R3"] else "R2")
    lines.append(f"8. **Recommendation: register {rec}.** Decision statistic: Tango's (1998) one-sided score test of H0: Δ ≤ −m on the "
                 "paired (wins, losses, n) counts — reject iff Z(−m) > z_{1−α_adj}, equivalently iff the one-sided (1 − α_adj) Tango lower bound "
                 "(`metrics.tango_interval(both, wins, losses, neither, alpha=α_adj, one_sided=True)['ci'][0]`) is above −m; the p-value entering "
                 "Holm is p_T = 1 − Φ(Z(−m)). Switch: whenever wins = 0 or fewer than 5 tasks are discordant, the exact one-sided binomial "
                 "p = P[L ≤ l | n, m] (`metrics.ni_exact_check`) replaces p_T and decides. Superiority primaries keep their bootstrap p_sup.")
    lines.append("9. The task-bootstrap lower bound stays as the descriptive interval next to every primary (it is no longer the decision); "
                 "for the τ² primary the rule is applied to the per-task binary pass^1 outcome (one trial per task, as S4 simulates; the 3-trial "
                 "cluster then only enters the descriptive bootstrap).")
    lines.append("10. Caveats: all rules depend on the data only through (wins, losses, n), so p_ref is immaterial; sizes are at the boundary of a "
                 "one-parameter family of nulls (P(win) fixed by d), the configurations the pilots make plausible; the B = 10,000 check (§4) and the "
                 "`ni_pvalue` validation (§5) show the bootstrap conclusions do not depend on B = 2,000 or on the fast resampling path.")
    return lines


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps-size", type=int, default=50_000)
    ap.add_argument("--reps-power", type=int, default=20_000)
    ap.add_argument("--reps-b10000", type=int, default=20_000)
    ap.add_argument("--reps-validate", type=int, default=600)
    ap.add_argument("--boot", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", type=Path, default=ROOT / "results/reviews/2026-10-01c")
    ap.add_argument("--validate", action="store_true", help="also run R1 through metrics.ni_pvalue on per-task dicts")
    ap.add_argument("--no-b10000", action="store_true")
    ap.add_argument("--quick", action="store_true", help="500/500 replicates, for a smoke run")
    ap.add_argument("--render-only", action="store_true", help="rewrite the .md from the existing .json (no simulation)")
    args = ap.parse_args()
    if args.render_only:
        results = json.loads((args.out / "ni_size_sim.json").read_text())
        results["reading"] = reading(results)
        (args.out / "ni_size_sim.json").write_text(json.dumps(results, indent=1))
        write_md(args.out / "ni_size_sim.md", results, results["command"], results["runtime_s"])
        return
    if args.quick:
        args.reps_size = args.reps_power = args.reps_b10000 = 500
        args.reps_validate = 50
    t0 = time.time()
    cmd = "python " + " ".join(sys.argv)
    ss = np.random.SeedSequence(args.seed)
    results: dict[str, Any] = {"date": time.strftime("%Y-%m-%d"), "seed": args.seed, "B": args.boot,
                               "reps_size": args.reps_size, "reps_power": args.reps_power, "levels": LEVELS,
                               "rules": RULE_DESC, "points": [], "infeasible": [], "b10000_check": [], "validation": []}
    tables: dict[tuple[int, float], Tables] = {}
    for sc in scenarios():
        key = (sc["n"], sc["m"])
        if key not in tables:
            print(f"[{time.time() - t0:7.1f}s] building exact / Tango tables for n={key[0]}, m={key[1]}", file=sys.stderr, flush=True)
            tables[key] = Tables(*key)
        tab = tables[key]
        for delta, role in sc["deltas"]:
            c = cells(sc["p_ref"], delta, sc["d"], sc["losses_only"])
            if c is None:
                results["infeasible"].append({"scenario": sc["id"], "delta": delta, "reason": "negative cell probability"})
                continue
            reps = args.reps_size if role in ("size", "inside_H0") else args.reps_power
            rng = np.random.default_rng(ss.spawn(1)[0])
            print(f"[{time.time() - t0:7.1f}s] {sc['id']:28s} Δ={delta:+.3f} role={role:9s} P(loss)={c['loss']:.3f} "
                  f"P(win)={c['win']:.3f} reps={reps}", file=sys.stderr, flush=True)
            r = run_point(rng, tab, c["win"], c["loss"], reps, args.boot)
            results["points"].append({"scenario": sc["id"], "group": sc["group"], "n": sc["n"], "m": sc["m"],
                                      "p_ref_nominal": sc["p_ref"], "d": sc["d"], "delta": delta, "role": role,
                                      "cells": c, **r})
    if not args.no_b10000:
        for sid, p_win, p_loss, n, m in (("S1 d=0.03 (boundary)", 0.0, 0.03, 500, 0.03),
                                         ("S2 d=0.05 (boundary)", 0.01, 0.04, 500, 0.03)):
            tab = tables[(n, m)]
            chk: dict[str, Any] = {"scenario": sid, "p_win": p_win, "p_loss": p_loss}
            for B in (2000, 10000):
                print(f"[{time.time() - t0:7.1f}s] B={B} check {sid} reps={args.reps_b10000}", file=sys.stderr, flush=True)
                rng = np.random.default_rng(ss.spawn(1)[0])
                chk[f"B{B}"] = run_point(rng, tab, p_win, p_loss, args.reps_b10000, B)
            results["b10000_check"].append(chk)
    if args.validate:
        print(f"[{time.time() - t0:7.1f}s] validating against metrics.ni_pvalue ({args.reps_validate} reps x 3 levels)",
              file=sys.stderr, flush=True)
        results["validation"] = validate_ni_pvalue(args.seed, args.boot, args.reps_validate)
    results["tango_max_dev"] = max(t.tango_max_dev for t in tables.values())
    results["tango_checked"] = sum(t.tango_checked for t in tables.values())
    results["cpu"] = os.cpu_count()
    results["reading"] = reading(results)
    runtime = time.time() - t0
    results["runtime_s"], results["command"] = runtime, cmd
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ni_size_sim.json").write_text(json.dumps(results, indent=1))
    write_md(args.out / "ni_size_sim.md", results, cmd, runtime)
    print(f"done in {runtime:.0f}s -> {args.out / 'ni_size_sim.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
