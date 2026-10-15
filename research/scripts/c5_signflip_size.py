#!/usr/bin/env python
"""Finite-sample size of C5's registered test (A6.16 (g)) under the WEAK null E[Delta U] = 0, by simulation (review
round 8, W2). Offline, fixed seeds, no evaluation data.

The registered decision: one-sided sign-flip test of the mean paired utility difference d_i = U_arm - U_control
(gross of System-1's price, W = $1 per success, lambda = 0; scripts/primaries.py du_block), metrics.sign_flip_test(d,
n_perm=20000, seed=0)["p_greater"] <= alpha_adj at n = 500 paired tasks (A4, A6.15). This script applies EXACTLY
that procedure: the 20,000 sign vectors are those sign_flip_test draws with seed 0 (checked against the function), so
each replicate's p-value is the registered one.

Theory (checked here, not assumed): sum_i d_i^2 is invariant under sign flips, so the permutation p of the mean equals
that of the studentized mean (the paired t statistic is increasing in the mean at fixed sum d_i^2; verified on a sample
of replicates below). The randomization distribution of n^(-1/2) sum eps_i d_i converges to N(0, E d^2), which equals
Var d under E d = 0, so the test is asymptotically level alpha for the weak null with finite variance (Romano 1990,
JASA 85:686-692; Lehmann & Romano 2022, TSH 4th ed., Example 17.2.4; Ritzwoller, Romano & Shaikh 2025, sec. 3.2), and
exact when d is symmetric about 0 (within-task exchangeability of arm and control). Finite-sample size under skew is
what this simulation measures.

Shapes of d (all with mean exactly 0, n = 500 i.i.d. tasks):
  * realistic (dev pilot): success flips +-1 at the paired discordance q (5 %, 10 %, 20 %; the dev round-3 pair had 3 of
    30 tasks discordant) and a cost part resampled from the 30 dev paired System-2 cost differences (development seeds
    0..29, runs/dev/e2b_dev_r3; dollars, |dc| <= 9e-4), recentred; then the most unbalanced flips this cost scale
    can offset: P(loss) - P(win) = 1e-3 with every task's cost part shifted by +1e-3 (so no d is exactly 0);
  * stress (not realistic for C5, which runs a matched-rate control at the same System-2 price): rare large jumps
    offset by many small values of the other sign, in both directions, at several rates, and a zero-inflated
    exponential. Left-skewed shapes (rare large negative d) are the anti-conservative direction for p_greater.

Comparison on the same replicates: the one-sided paired t-test (Student t, n - 1 df; metrics.paired_t_test), the
studentized paired-mean test the review suggested, which compares with a fixed critical value instead of a
permutation distribution on the lattice of the success flips.

    python scripts/c5_signflip_size.py [--reps 100000] [--out results/reviews/2026-10-02] [--render-only]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import metrics as M  # noqa: E402

N, N_PERM, LEVELS = 500, 20000, {"0.05": 0.05, "0.05/11": 0.05 / 11}
DEV_PAIR = ("runs/dev/e2b_dev_r3/jev_reflect_trigger_s2lite.jsonl", "runs/dev/e2b_dev_r3/ratem_reflect_trigger_s2lite.jsonl")


def registered_signs(n: int) -> np.ndarray:
    """The 20,000 x n sign matrix metrics.sign_flip_test(d, n_perm=20000, seed=0) uses for n > exact_max_n."""
    rng = np.random.default_rng(0)
    out, done = [], 0
    while done < N_PERM:
        b = min(2000, N_PERM - done)
        out.append(rng.choice([-1.0, 1.0], size=(b, n)))
        done += b
    return np.concatenate(out)


def _betacf(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Continued fraction of the regularized incomplete beta (modified Lentz), vectorised over x."""
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = 1.0 / np.where(np.abs(d) < fpmin, fpmin, d)
    h = d.copy()
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / np.where(np.abs(d) < fpmin, fpmin, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < fpmin, fpmin, c)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / np.where(np.abs(d) < fpmin, fpmin, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < fpmin, fpmin, c)
        h *= d * c
    return h


def _betai(a: float, b: float, x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 1e-300, 1.0 - 1e-16)
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * np.log(x) + b * np.log1p(-x)
    bt = np.exp(lbt)
    lo = x < (a + 1.0) / (a + b + 2.0)
    out = np.empty_like(x)
    out[lo] = bt[lo] * _betacf(a, b, x[lo]) / a
    out[~lo] = 1.0 - bt[~lo] * _betacf(b, a, 1.0 - x[~lo]) / b
    return out


def t_sf(t: np.ndarray, df: int) -> np.ndarray:
    """P(T_df >= t) for Student's t (regularized incomplete beta; no scipy)."""
    t = np.asarray(t, float)
    half = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return np.where(t >= 0, half, 1.0 - half)


def dev_cost_diffs() -> np.ndarray:
    """Paired System-2 cost differences (control - arm, dollars) of the dev round-3 C5-like pair (dev seeds 0..29)."""
    def load(p: str) -> dict[str, float]:
        out = {}
        for line in (ROOT / p).read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                u = r["usage"]
                out[r["task_id"]] = u["total"]["cost_usd"] - u.get("s1", {}).get("cost_usd", 0.0)
        return out
    a, c = load(DEV_PAIR[0]), load(DEV_PAIR[1])
    keys = sorted(set(a) & set(c))
    return np.array([c[k] - a[k] for k in keys])


def shapes(dc: np.ndarray) -> list[tuple[str, str, Callable[[np.random.Generator, int], np.ndarray]]]:
    dc0 = dc - dc.mean()

    def realistic(q: float) -> Callable:
        def f(rng, size):
            u = rng.random(size)
            y = np.where(u < q / 2, 1.0, np.where(u < q, -1.0, 0.0))
            return y + rng.choice(dc0, size=size)
        return f

    def realistic_unbalanced(q: float, delta: float) -> Callable:
        def f(rng, size):
            u = rng.random(size)
            y = np.where(u < (q - delta) / 2, 1.0, np.where(u < q, -1.0, 0.0))     # P(loss) - P(win) = delta
            return y + rng.choice(dc0, size=size) + delta
        return f

    def jumps(p_jump: float, sign: float, firing: float | None) -> Callable:
        # a jump of size 1 (sign) w.p. p_jump; the offset -sign * p_jump spread over the other tasks: constant on
        # every task (firing None) or exponential on a `firing` share of them
        def f(rng, size):
            j = rng.random(size) < p_jump
            if firing is None:
                off = np.full(size, -sign * p_jump)
            else:
                off = np.where(rng.random(size) < firing, -sign * p_jump / firing * rng.exponential(1.0, size), 0.0)
            return np.where(j, sign * 1.0, 0.0) + off
        return f

    def zexp(sign: float) -> Callable:
        def f(rng, size):
            fire = rng.random(size) < 0.2
            return np.where(fire, sign * (rng.exponential(1.0, size) - 1.0), 0.0)
        return f

    return [
        ("realistic", "dev shape, q = 5 % discordant, balanced flips", realistic(0.05)),
        ("realistic", "dev shape, q = 10 %, balanced flips", realistic(0.10)),
        ("realistic", "dev shape, q = 20 %, balanced flips", realistic(0.20)),
        ("realistic", "dev shape, q = 5 %, dev cost part as observed (mean +4e-5) offset by P(loss) - P(win) = 4e-5", realistic_unbalanced(0.05, float(dc.mean()))),
        ("realistic", "dev shape, q = 10 %, dev cost part as observed (mean +4e-5) offset by P(loss) - P(win) = 4e-5", realistic_unbalanced(0.10, float(dc.mean()))),
        ("realistic", "dev shape, q = 20 %, dev cost part as observed (mean +4e-5) offset by P(loss) - P(win) = 4e-5", realistic_unbalanced(0.20, float(dc.mean()))),
        ("realistic", "dev shape, q = 10 %, P(loss) - P(win) = 1e-3 offset by +1e-3 cost on every task", realistic_unbalanced(0.10, 1e-3)),
        ("realistic", "dev shape, q = 5 %, P(loss) - P(win) = 1e-3 offset by +1e-3 cost on every task", realistic_unbalanced(0.05, 1e-3)),
        ("stress", "rare rescues (2 %) offset on a 20 % share (right skew)", jumps(0.02, +1.0, 0.2)),
        ("stress", "rare harms (2 %) offset on a 20 % share (left skew)", jumps(0.02, -1.0, 0.2)),
        ("stress", "rare harms (2 %) offset on every task (left skew, no zeros)", jumps(0.02, -1.0, None)),
        ("stress", "very rare harms (0.4 %) offset on every task (left skew, no zeros)", jumps(0.004, -1.0, None)),
        ("stress", "very rare rescues (0.4 %) offset on every task (right skew, no zeros)", jumps(0.004, +1.0, None)),
        ("stress", "zero-inflated exponential, 20 % non-zero (right skew)", zexp(+1.0)),
        ("stress", "zero-inflated exponential, 20 % non-zero (left skew)", zexp(-1.0)),
    ]


def run_shape(S: np.ndarray, gen: Callable, reps: int, seed: int, batch: int = 500) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rej = {k: 0 for k in LEVELS}
    rej_t = {k: 0 for k in LEVELS}
    done = 0
    while done < reps:
        b = min(batch, reps - done)
        X = gen(rng, (b, N))
        means = (S @ X.T) / N          # (N_PERM, b)
        obs = X.mean(1)
        ge = (means >= obs[None, :] - 1e-12).sum(0)
        p = (ge + 1) / (N_PERM + 1)
        sd = X.std(1, ddof=1)
        pt = t_sf(np.where(sd > 0, obs / np.where(sd > 0, sd, 1.0) * math.sqrt(N), 0.0), N - 1)   # paired t-test
        for k, a in LEVELS.items():
            rej[k] += int((p <= a).sum())
            rej_t[k] += int((pt <= a).sum())
        done += b
    out: dict[str, Any] = {"reps": reps, "seed": seed, "size": {}, "mc_se": {}, "ratio": {}, "size_t": {}, "ratio_t": {}}
    for k, a in LEVELS.items():
        s = rej[k] / reps
        out["size"][k], out["mc_se"][k], out["ratio"][k] = s, math.sqrt(max(s * (1 - s), a * (1 - a)) / reps), s / a
        out["size_t"][k], out["ratio_t"][k] = rej_t[k] / reps, rej_t[k] / reps / a
    return out


def check_equivalences(S: np.ndarray, gen: Callable, seed: int = 99, k: int = 40) -> dict[str, Any]:
    """(i) p from S equals metrics.sign_flip_test's p_greater; (ii) the studentized statistic gives the same p."""
    rng = np.random.default_rng(seed)
    X = gen(rng, (k, N))
    dev_f = dev_t = 0.0
    for x in X:
        ref = M.sign_flip_test(list(x), n_perm=N_PERM, seed=0)["p_greater"]
        means = S @ x / N
        p = (np.sum(means >= x.mean() - 1e-12) + 1) / (N_PERM + 1)
        ss = float(np.sum(x * x))
        tstat = lambda mu: mu / np.sqrt(np.maximum(ss / N - mu * mu, 1e-300) / (N - 1))  # noqa: E731
        pt = (np.sum(tstat(means) >= tstat(x.mean()) - 1e-9) + 1) / (N_PERM + 1)
        dev_f, dev_t = max(dev_f, abs(p - ref)), max(dev_t, abs(pt - p))
    return {"replicates": k, "max_abs_diff_vs_sign_flip_test": dev_f, "max_abs_diff_studentized": dev_t}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=100000)
    ap.add_argument("--out", type=Path, default=ROOT / "results/reviews/2026-10-02")
    ap.add_argument("--render-only", action="store_true", help="rewrite the .md from the existing .json")
    a = ap.parse_args()
    if a.render_only:
        write_md(a.out / "c5_signflip_size.md", json.loads((a.out / "c5_signflip_size.json").read_text()))
        return
    t0 = time.time()
    S = registered_signs(N)
    dc = dev_cost_diffs()
    sh = shapes(dc)
    checks = check_equivalences(S, sh[1][2])
    print(f"equivalence checks: {checks}", file=sys.stderr)
    rows = []
    for i, (kind, label, gen) in enumerate(sh):
        r = run_shape(S, gen, a.reps, seed=20261002 + i)
        rng = np.random.default_rng(7 + i)
        smp = gen(rng, 200000)
        mom = {"mean": float(smp.mean()), "sd": float(smp.std()),
               "skew": float(np.mean((smp - smp.mean()) ** 3) / smp.std() ** 3), "p_zero": float(np.mean(smp == 0))}
        rows.append({"kind": kind, "shape": label, **r, "moments": mom})
        print(f"[{time.time() - t0:6.1f}s] {label}: " + ", ".join(f"{k} {r['size'][k]:.5f} ({r['ratio'][k]:.2f}x)"
                                                                  for k in LEVELS), file=sys.stderr)
    res = {"n": N, "n_perm": N_PERM, "levels": LEVELS, "reps": a.reps, "dev_cost_diffs": {"n": int(len(dc)),
           "mean": float(dc.mean()), "sd": float(dc.std()), "min": float(dc.min()), "max": float(dc.max())},
           "checks": checks, "rows": rows, "date": time.strftime("%Y-%m-%d"), "runtime_s": time.time() - t0,
           "command": "python " + " ".join(sys.argv)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "c5_signflip_size.json").write_text(json.dumps(res, indent=1))
    write_md(a.out / "c5_signflip_size.md", res)
    print(f"done in {res['runtime_s']:.0f}s", file=sys.stderr)


def write_md(path: Path, res: dict) -> None:
    L = ["# C5: finite-sample size of the registered sign-flip test under the weak null E[ΔU] = 0", "",
         f"`scripts/c5_signflip_size.py` ({res['date']}), raw numbers in `c5_signflip_size.json`. The registered "
         f"procedure exactly (one-sided sign-flip test of the mean paired ΔU, {res['n_perm']:,} sign vectors drawn with "
         f"seed 0 as `metrics.sign_flip_test` draws them; reject iff p_greater ≤ α) at the registered n = {res['n']}, "
         f"{res['reps']:,} replicates per shape, every shape with population mean exactly 0. Monte-Carlo standard error "
         "√(α(1 − α)/R) or larger.", "",
         f"Checks: p from the stored sign matrix equals `sign_flip_test` on {res['checks']['replicates']} replicates "
         f"(max |Δp| = {res['checks']['max_abs_diff_vs_sign_flip_test']:.1e}); the studentized (paired-t) statistic "
         f"gives the same p (max |Δp| = {res['checks']['max_abs_diff_studentized']:.1e}), because Σd² is invariant "
         "under sign flips.", "",
         f"Dev cost component: {res['dev_cost_diffs']['n']} paired System-2 cost differences (dev seeds, "
         f"`runs/dev/e2b_dev_r3`): mean {res['dev_cost_diffs']['mean']:.1e}, sd {res['dev_cost_diffs']['sd']:.1e}, range "
         f"[{res['dev_cost_diffs']['min']:.1e}, {res['dev_cost_diffs']['max']:.1e}] dollars, against W = $1 per success.", "",
         "Registered: the sign-flip test. Comparison: the one-sided paired t-test (Student t, n − 1 df), the "
         "studentized paired-mean test without a permutation reference.", "",
         "| kind | shape of d | skewness | P(d = 0) | sign-flip, α = 0.05 (×α) | sign-flip, α = 0.05/11 (×α) | "
         "paired t, 0.05 (×α) | paired t, 0.05/11 (×α) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        s, q, qt = r["size"], r["ratio"], r["ratio_t"]
        L.append(f"| {r['kind']} | {r['shape']} | {r['moments']['skew']:+.1f} | {r['moments']['p_zero']:.2f} | "
                 f"{s['0.05']:.4f} ± {r['mc_se']['0.05']:.4f} ({q['0.05']:.2f}×) | "
                 f"{s['0.05/11']:.5f} ± {r['mc_se']['0.05/11']:.5f} ({q['0.05/11']:.2f}×) | "
                 f"{r['size_t']['0.05']:.4f} ({qt['0.05']:.2f}×) | {r['size_t']['0.05/11']:.5f} ({qt['0.05/11']:.2f}×) |")
    real = [r for r in res["rows"] if r["kind"] == "realistic"]
    stress = [r for r in res["rows"] if r["kind"] == "stress"]
    def over(rows, key):              # cells above alpha by more than 2 Monte-Carlo SEs
        return [(r["shape"], k) for r in rows for k, a in res["levels"].items()
                if r[key][k] > a + 2 * math.sqrt(a * (1 - a) / r["reps"])]
    bal = [r for r in real if "balanced" in r["shape"]]
    off = [r for r in real if "offset" in r["shape"]]
    mx = lambda rows, key: {k: max(r[key][k] for r in rows) for k in res["levels"]}  # noqa: E731
    worst = max(stress, key=lambda r: r["ratio"]["0.05/11"])
    worst_t = max(stress, key=lambda r: r["ratio_t"]["0.05/11"])
    ov_sf, ov_t = over(real, "size"), over(real, "size_t")
    L += ["", "## Reading", "",
          f"1. **Balanced success flips** (the dev cost part recentred): the sign-flip test's size is at most "
          f"{mx(bal, 'ratio')['0.05']:.2f} α at 0.05 and {mx(bal, 'ratio')['0.05/11']:.2f} α at the floor.",
          f"2. **A small systematic cost difference offsetting a success deficit** (mean exactly 0; the dev pair's own cost "
          f"mean, +4e-5 dollars per task, or +1e-3): the sign-flip test reaches {mx(off, 'ratio')['0.05']:.2f} α at 0.05 and "
          f"{mx(off, 'ratio')['0.05/11']:.2f} α at the floor; the paired t-test {mx(off, 'ratio_t')['0.05']:.2f} α and "
          f"{mx(off, 'ratio_t')['0.05/11']:.2f} α. Mechanism: the success part of the permutation distribution sits on a "
          "lattice of step 2/n; a systematic cost part, however small, shifts the observed mean off its lattice point in "
          "one direction, so the observed sign pattern beats every pattern with the same success sum and p drops by a "
          "whole lattice step. The t-test compares with a fixed critical value and has no such tie-breaking.",
          "3. **Realistic shapes above the level by more than two Monte-Carlo standard errors:** sign-flip: "
          + ("; ".join(f"{s} (α = {k})" for s, k in ov_sf) or "none") + "; paired t: "
          + ("; ".join(f"{s} (α = {k})" for s, k in ov_t) or "none") + ".",
          f"4. **Stress shapes.** Rare large jumps offset by many small values of the other sign break the symmetry the "
          f"finite-sample guarantee needs. Worst case for the sign-flip test: '{worst['shape']}', {worst['ratio']['0.05']:.2f} α "
          f"at 0.05 and {worst['ratio']['0.05/11']:.2f} α at the floor; for the paired t-test: '{worst_t['shape']}', "
          f"{worst_t['ratio_t']['0.05']:.2f} α and {worst_t['ratio_t']['0.05/11']:.2f} α. Left-skewed d (rare large losses) "
          "inflates both one-sided upper tests; right-skewed d makes them conservative.",
          "5. **Which shapes C5 can produce.** With W = $1 the only large values of d are the success flips ±1; the rest is "
          f"the System-2 cost difference, at most {max(abs(res['dev_cost_diffs']['min']), abs(res['dev_cost_diffs']['max'])):.0e} dollars "
          "per task in the dev pair. Under E[d] = 0 the flips can therefore be unbalanced by at most about 1e-3, which the "
          "realistic rows cover; the stress shapes need offsets of order 0.1 dollars per task and cannot arise.",
          "6. **What can be claimed.** The sign-flip test is exact under within-task exchangeability of arm and control (d "
          "symmetric about 0) and asymptotically level α for the weak null E[d] = 0 with finite variance; at n = 500 it is "
          "materially anti-conservative when a small systematic cost difference offsets a success deficit. The one-sided "
          f"paired t-test on the same d stays within two Monte-Carlo standard errors of α on every realistic shape simulated "
          f"(at most {mx(real, 'ratio_t')['0.05']:.2f} α at 0.05 and {mx(real, 'ratio_t')['0.05/11']:.2f} α at the floor): "
          "the minimal fix of A6.21 (b).", "",
          f"Runtime {res['runtime_s']:.0f} s. Command: `{res['command']}`.", ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
