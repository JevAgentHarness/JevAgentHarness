#!/usr/bin/env python
"""Exact size (type-I error under the NI null) of the non-inferiority decision rules by exact summation over the
paired trinomial counts -- no Monte Carlo. Round 8 (default output results/reviews/2026-10-02/ni_exact_size_r8.*):
the A6.21 rule ``bb_ordered`` (metrics.ni_certified_p) next to the superseded A6.14 / A6.17 rules; ``--legacy``
reproduces the round-7 artefact (results/reviews/2026-10-01d/ni_exact_size.*).

bb_ordered (A6.21) rejects at (W, L) iff p(W, L) = min(1, gamma + U) <= alpha, U the certified branch-and-bound
supremum. Its rejection region is computed EXACTLY, cell by cell, without running the branch-and-bound on every cell:
  * a fixed nuisance node grid (uniform in the Fisher arclength of the boundary path, plus every Clopper-Pearson end
    point) gives, per discordant count D and level, two critical ranks of the ordering statistic: crit_L (some node
    inside the nuisance set has tail probability > alpha - gamma above it, so the deployed p > alpha: never rejects)
    and crit_C (the Hoelder bound of metrics.ni_holder_logbound over every node interval covering the set is <=
    min(beta - abs_tol, beta / (1 + rel_tol)), beta = alpha - gamma, so the deployed branch-and-bound value is <= beta:
    always rejects);
  * every cell with crit_C < rank <= crit_L ("gray") is decided by metrics.ni_certified_p itself.
So the size of the deployed rule is computed exactly at each listed parameter value (boundary points w and interior
null points); the validity proof (A6.21) is what covers the parameter values in between.

Round-7 docstring follows (A6.14 / A6.17 rules, still computed as comparison columns):

Exact size (type-I error at the NI boundary) of the registered non-inferiority decision rule (A6.14) by exact
summation over the paired trinomial counts -- no Monte Carlo (review round 6/7: "evaluate its rejection region over
a dense nuisance grid, especially near zero win probability, at all eleven Holm levels; exact summation over the
paired trinomial counts avoids Monte-Carlo ambiguity").

The data enter every rule only through (wins W, losses L, n). Under the boundary null Delta = -m with win probability
w the per-task outcome is trinomial: P(win) = w, P(loss) = w + m, P(tie) = 1 - 2w - m, so

    size(w) = sum_{W + L <= n} Multinomial(n; w, w + m, 1 - 2w - m)(W, L) * 1[rule rejects at (W, L)].

Rules (all decided on the counts; the decision is memoised per (n, m, W, L) and compared with every level):
  registered  A6.14: Tango's (1998) one-sided score test p_T = 1 - Phi(Z(-m)) (metrics._tango_z, metrics._norm_sf),
              reject iff p_T <= alpha, EXCEPT that whenever W == 0 or W + L < 5 the exact one-sided binomial test
              p = P[L <= l | n, m] (metrics.ni_exact_check(...)["p"]) decides with the registered strict p < alpha
              (metrics.ni_pvalue: rule = "exact", ni_adj = exact["ni"]).
  exact_only  the exact binomial test for every comparison (reviewer option 2).
  tango_only  the Tango score test for every comparison.
  switch_w0   Tango with the exact fallback at W == 0 only;  switch_lt5: at W + L < 5 only (the switch comparison).
  registered_u  A6.17: as `registered`, but the Tango branch rejects iff p_U <= alpha, p_U = max_w P_w[Z >= z_obs and
              not switched] the exact unconditional p-value of the same statistic on the branch's own region of the
              sample space (W >= 1 and W + L >= 5; metrics.ni_unconditional_p; nuisance grid
              metrics.ni_unconditional_w_grid, step 0.0025). Its rejection region at level alpha is {not switched,
              Z >= z*(alpha)} with z*(alpha) = max over the nuisance grid of the exact (1 - alpha) quantile of Z on
              that region, so the branch's size is <= alpha at every grid w by construction; the whole two-branch
              rule's size (branch + exact fallback, disjoint regions) is what the tables report. The W_GRID below is
              mostly OFF the maximisation grid (0.0005, 0.002, 0.003, 0.0075, 0.015, ...), so the tables also check
              the off-grid residual. (p_U also includes the observed win rate in its maximisation, which can only
              raise p_U, so the sizes here bound the rule's.)
  uncond_only the A6.17 Tango branch alone (its exact fallback region never rejects).

The multinomial pmf is evaluated in log space (math.lgamma log-factorials; the full-triangle sum is checked against
1), pairs with pmf < --prune (1e-15) are dropped (the dropped mass is reported) and the Tango p-value is evaluated
once per (W, L) pair on the union of the kept pairs over the w grid. The exact p-value table P[L <= l | n, m] is
metrics._binom_cdf (the function ni_exact_check calls) and is checked against ni_exact_check itself; the Tango
decision p_T <= alpha is cross-checked against metrics.tango_interval(one_sided=True)["ci"][0] > -m on the pairs
nearest each level.

Usage:  python scripts/ni_exact_size.py [--out results/reviews/2026-10-02] [--prune 1e-15] [--render-only] [--legacy]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.metrics import (_binom_cdf, _norm_sf, _tango_z, _tango_z_vec, _z, ni_exact_check,  # noqa: E402
                                      ni_unconditional_p, ni_unconditional_w_grid, tango_interval)

# ------------------------------------------------------------------ settings
LEVELS: dict[str, float] = {f"0.05/{k}": 0.05 / k for k in range(11, 0, -1)}      # the eleven Holm levels, floor first
LEVELS["0.10"] = 0.10                                                                # reference only
HOLM_LEVELS = [lv for lv in LEVELS if lv != "0.10"]
KEY_LEVELS = ["0.05/11", "0.05/6", "0.05/1"]
W_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15,
          0.20, 0.25, 0.30, 0.35]
SETTINGS = [(500, 0.03, "JevGym primaries (C1/C2/C6, A5.4 sizes)"),
            (149, 0.05, "tau^2 majority-pass endpoint (C7)"),
            (300, 0.03, "C8 static items, lower bracket"),
            (1000, 0.03, "C8 static items, upper bracket"),
            (1114, 0.03, "C8 primary: BFCL tool_select items after the parent-item exclusion (A6.18 (d))"),
            (560, 0.05, "Stage 2, S2-L3a: telecom single-attempt success at the frozen n (PREREG_A7_STAGE2 A7.15, A7.6)")]
SWITCH_SETTINGS = [(500, 0.03), (149, 0.05)]
RULES = ["registered", "exact_only", "tango_only", "registered_u"]
VARIANTS = ["tango_only", "switch_w0", "switch_lt5", "registered", "exact_only", "uncond_only", "registered_u"]
VARIANT_DESC = {"tango_only": "Tango only", "switch_w0": "exact at W = 0 only", "switch_lt5": "exact at W + L < 5 only",
                "registered": "both (A6.14)", "exact_only": "exact only", "uncond_only": "p_U branch alone (A6.17, no exact part)",
                "registered_u": "A6.17 (p_U + switch)"}
FLAG_REL = 1.02


# ------------------------------------------------------------------ exact enumeration for one (n, m)
class ExactSize:
    """All (W, L) pairs of the triangle W + L <= n for one (n, m): the per-pair p-values of both component tests
    (memoised: they do not depend on w) and the exact size of every rule at every level for a given w."""

    def __init__(self, n: int, m: float, exact_below: int = 5) -> None:
        self.n, self.m, self.exact_below = n, m, exact_below
        self.lf = np.array([math.lgamma(k + 1) for k in range(n + 1)])          # log k!
        W, L = np.indices((n + 1, n + 1))
        self.W, self.L = W, L
        self.T = n - W - L
        self.tri = self.T >= 0
        self.few = (W + L) < exact_below
        self.w0 = W == 0
        # exact one-sided binomial p = P[L <= l | n, m]: metrics._binom_cdf, the function ni_exact_check calls
        self.p_exact_l = np.array([_binom_cdf(l, n, m) for l in range(n + 1)])
        self.p_exact = self.p_exact_l[L]
        self.p_tango = np.full((n + 1, n + 1), np.nan)
        self.n_tango_evals = 0
        # A6.17: Tango's Z on the whole triangle (vectorised closed form) and, after bb_pvalues(), the Berger-Boos p
        self.z = np.where(self.tri, _tango_z_vec(-m, W, L, n), -np.inf)
        self.p_bb: np.ndarray | None = None
        self.bb: dict[str, Any] = {}
        # critical values of the exact test: largest l with p < alpha (-1 if none) and the test's own size at P(loss) = m
        self.exact_crit = {}
        for lv, a in LEVELS.items():
            ok = np.nonzero(self.p_exact_l < a)[0]
            lstar = int(ok.max()) if len(ok) else -1
            self.exact_crit[lv] = {"l_star": lstar, "size_at_m": float(self.p_exact_l[lstar]) if lstar >= 0 else 0.0}

    def check_exact_table(self, upto: int = 80) -> dict[str, Any]:
        """The exact p table must coincide with metrics.ni_exact_check(...)["p"] (bit-identical, same function)."""
        dev = 0.0
        for l in range(min(upto, self.n) + 1):
            for wins in (0, 1, 3):
                dev = max(dev, abs(ni_exact_check(wins, l, self.n, self.m, 0.05)["p"] - self.p_exact_l[l]))
        return {"l_checked_upto": min(upto, self.n), "max_abs_dev": dev}

    def logpmf(self, w: float) -> np.ndarray:
        """log Multinomial(n; w, w + m, 1 - 2w - m)(W, L) on the triangle, -inf outside it."""
        n, m = self.n, self.m
        pl, pt = w + m, 1.0 - 2.0 * w - m
        if pt < -1e-12:
            raise ValueError(f"infeasible: 2w + m > 1 at w = {w}")
        W, L, T = self.W, self.L, np.clip(self.T, 0, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            lw = np.where(W == 0, 0.0, W * np.log(w)) if w == 0 else W * np.log(w)
            lt = np.where(T == 0, 0.0, T * np.log(pt)) if pt <= 0 else T * np.log(pt)
            lp = self.lf[n] - self.lf[W] - self.lf[L] - self.lf[T] + lw + L * np.log(pl) + lt
        return np.where(self.tri, lp, -np.inf)

    def logpmf_alt(self, p_win: float, p_loss: float) -> np.ndarray:
        """log Multinomial(n; p_win, p_loss, 1 - p_win - p_loss)(W, L) on the triangle for an arbitrary alternative
        (power computations, scripts/ni_power_exact.py); -inf outside it."""
        n = self.n
        pt = 1.0 - p_win - p_loss
        if pt < -1e-12 or p_win < 0 or p_loss < 0:
            raise ValueError(f"infeasible alternative p_win = {p_win}, p_loss = {p_loss}")
        W, L, T = self.W, self.L, np.clip(self.T, 0, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            lw = np.where(W == 0, 0.0, W * np.log(p_win)) if p_win == 0 else W * np.log(p_win)
            ll = np.where(L == 0, 0.0, L * np.log(p_loss)) if p_loss == 0 else L * np.log(p_loss)
            lt = np.where(T == 0, 0.0, T * np.log(pt)) if pt <= 0 else T * np.log(pt)
            lp = self.lf[n] - self.lf[W] - self.lf[L] - self.lf[T] + lw + ll + lt
        return np.where(self.tri, lp, -np.inf)

    def rejects(self, rule: str, alpha: float, mask: np.ndarray) -> np.ndarray:
        """Rejection indicator of ``rule`` at level ``alpha`` on the masked pairs (registered = A6.14, registered_u =
        A6.17 after bb_pvalues(), exact_only, tango_only)."""
        self.ensure_tango(mask)
        pT, pE, few, w0 = self.p_tango[mask], self.p_exact[mask], self.few[mask], self.w0[mask]
        rt, re = pT <= alpha, pE < alpha
        if rule == "registered":
            return np.where(w0 | few, re, rt)
        if rule == "registered_u":
            return np.where(w0 | few, re, self.p_bb[mask] <= alpha)
        if rule == "exact_only":
            return re
        if rule == "tango_only":
            return rt
        raise ValueError(rule)

    def ensure_tango(self, mask: np.ndarray) -> None:
        """Evaluate p_T = 1 - Phi(Z(-m)) (metrics._tango_z / _norm_sf) once per newly needed (W, L) pair."""
        need = mask & np.isnan(self.p_tango)
        for wi, li in np.argwhere(need):
            self.p_tango[wi, li] = _norm_sf(_tango_z(-self.m, int(wi), int(li), self.n))
            self.n_tango_evals += 1

    def cp_intervals(self, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact two-sided (1 - gamma) Clopper-Pearson bounds of the discordance rate rho for every D = W + L = 0..n
        (vectorised bisection on the binomial cdf in log space; the same interval metrics.clopper_pearson returns)."""
        n = self.n
        D = np.arange(n + 1)
        i = np.arange(n + 1)
        base = self.lf[n] - self.lf[i] - self.lf[n - i]
        half = gamma / 2.0

        def cdf(q: np.ndarray) -> np.ndarray:                           # P[X <= D | n, q[D]] for every D at once
            qq = np.clip(q, 1e-300, 1 - 1e-16)[:, None]
            lp = base[None, :] + i[None, :] * np.log(qq) + (n - i)[None, :] * np.log1p(-qq)
            return np.cumsum(np.exp(lp), axis=1)[D, D]

        lo, hi = np.zeros(n + 1), D / n                                 # lower: P[X >= D | q] = half
        for _ in range(60):
            mid = (lo + hi) / 2
            c = cdf(mid)
            up = np.where(D > 0, 1.0 - np.concatenate([[0.0], c[:-1]]), 1.0)   # P[X >= D | mid] = 1 - P[X <= D - 1]
            go = up < half
            lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
        lower = np.where(D == 0, 0.0, (lo + hi) / 2)
        lo, hi = D / n, np.ones(n + 1)                                  # upper: P[X <= D | q] = half
        for _ in range(60):
            mid = (lo + hi) / 2
            go = cdf(mid) > half
            lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
        upper = np.where(D == n, 1.0, (lo + hi) / 2)
        return lower, upper

    def bb_pvalues(self, prune: float, gamma: float = 1e-4, step: float = 0.0025) -> dict[str, Any]:
        """The A6.17 p-value of every (W, L) pair on the Tango branch's region (W >= 1, W + L >= exact_below):
        p_U = max over the grid w inside the Berger-Boos (1 - gamma) set C(D) of P_w[Z >= z(W, L), not switched]
        + gamma (metrics.ni_unconditional_p evaluates the same maximum plus the set's end points and the point
        estimate, which can only raise it, so the sizes computed here bound the deployed rule's from above).
        +inf on the switch region."""
        n, m = self.n, self.m
        grid = ni_unconditional_w_grid(m, step)
        branch = self.tri & ~(self.w0 | self.few)
        rlo, rhi = self.cp_intervals(gamma)
        w_cap = (1.0 - m) / 2.0
        wlo, whi = np.clip((rlo - m) / 2.0, 0.0, w_cap), np.clip((rhi - m) / 2.0, 0.0, w_cap)
        Dmat = self.W + self.L
        zb = self.z[branch]
        Db = Dmat[branch]
        pmax = np.zeros(zb.shape)
        for w in grid:
            lp = self.logpmf(float(w))
            mask = lp > math.log(prune)
            pmf = np.where(branch[mask], np.exp(lp[mask]), 0.0)
            z = self.z[mask]
            uz, inv = np.unique(z, return_inverse=True)                       # ascending unique z
            mass = np.bincount(inv, weights=pmf, minlength=len(uz))
            tail = np.cumsum(mass[::-1])[::-1]                                  # tail[j] = P_w[Z >= uz[j], not switched]
            idx = np.searchsorted(uz, zb - 1e-12, side="left")
            t_at = np.where(idx < len(uz), tail[np.minimum(idx, len(uz) - 1)], 0.0)
            inset = (wlo[Db] - 1e-12 <= w) & (w <= whi[Db] + 1e-12)
            pmax = np.where(inset, np.maximum(pmax, t_at), pmax)
        self.p_bb = np.full((n + 1, n + 1), np.inf)
        self.p_bb[branch] = np.minimum(1.0, pmax + gamma)
        self.bb = {"gamma": gamma, "step": step, "n_grid": int(len(grid)), "branch_pairs": int(branch.sum()),
                   "w_set_check": {"D": [int(d) for d in (0, 5, 50, 100)],
                                   "w_lo": [float(wlo[d]) for d in (0, 5, 50, 100)], "w_hi": [float(whi[d]) for d in (0, 5, 50, 100)]}}
        return self.bb

    def sizes(self, w: float, prune: float) -> dict[str, Any]:
        lp = self.logpmf(w)
        full = float(np.exp(lp[self.tri]).sum())
        mask = lp > math.log(prune)
        self.ensure_tango(mask)
        pmf = np.exp(lp[mask])
        pT, pE, few, w0 = self.p_tango[mask], self.p_exact[mask], self.few[mask], self.w0[mask]
        pbb = self.p_bb[mask] if self.p_bb is not None else None
        out: dict[str, Any] = {"w": w, "p_loss": w + self.m, "p_tie": 1 - 2 * w - self.m, "E_wins": self.n * w,
                               "E_losses": self.n * (w + self.m), "pairs_kept": int(mask.sum()),
                               "pruned_mass": 1.0 - float(pmf.sum()), "full_triangle_sum": full,
                               "P_W0": float(pmf @ w0), "P_lt5": float(pmf @ few), "P_switch": float(pmf @ (w0 | few)),
                               "size": {}}
        for lv, a in LEVELS.items():
            rt, re = pT <= a, pE < a                                   # Tango: p_T <= alpha; exact: strict p < alpha
            out["size"][lv] = {
                "tango_only": float(pmf @ rt),
                "exact_only": float(pmf @ re),
                "switch_w0": float(pmf @ np.where(w0, re, rt)),
                "switch_lt5": float(pmf @ np.where(few, re, rt)),
                "registered": float(pmf @ np.where(w0 | few, re, rt)),
            }
            if pbb is not None:
                ru = pbb <= a
                out["size"][lv]["uncond_only"] = float(pmf @ (ru & ~(w0 | few)))          # the p_U branch alone
                out["size"][lv]["registered_u"] = float(pmf @ np.where(w0 | few, re, ru))
        return out

    def check_tango_interval(self, per_level: int = 25) -> dict[str, Any]:
        """p_T <= alpha must agree with tango_interval(..., one_sided=True)['ci'][0] > -m on the evaluated pairs
        nearest each level (both sides of it), except within 1e-6 of the boundary z."""
        have = ~np.isnan(self.p_tango)
        Ws, Ls, ps = self.W[have], self.L[have], self.p_tango[have]
        checked, disagreements, max_dev = 0, [], 0.0
        for lv, a in LEVELS.items():
            order = np.argsort(np.abs(np.log(np.maximum(ps, 1e-300)) - math.log(a)))[:per_level]
            zc = _z(1 - a)
            for i in order:
                wi, li = int(Ws[i]), int(Ls[i])
                lb = tango_interval(self.n - wi - li, wi, li, 0, alpha=a, one_sided=True)["ci"][0]
                z = _tango_z(-self.m, wi, li, self.n)
                checked += 1
                if (lb > -self.m) != (ps[i] <= a) and abs(z - zc) > 1e-6:
                    disagreements.append({"level": lv, "W": wi, "L": li, "p_T": float(ps[i]), "lcb": lb})
                max_dev = max(max_dev, abs(_norm_sf(z) - ps[i]))
        return {"checked": checked, "disagreements": disagreements}


# ------------------------------------------------------------------ A6.21: the exact rejection region of bb_ordered
_CR: dict[str, Any] = {}          # fork-shared state of the CertifiedRule being built (read by the pool workers)


def _cr_nodes(lo: int, hi: int) -> tuple[int, dict[str, np.ndarray]]:
    """Pool worker: critical ranks of nodes[lo:hi]. For node w: the window cells' trinomial pmf sorted by rank of the
    ordering statistic, cumulated (F_lo(k) = P_w[rank <= k] over the window, F_hi = F_lo (1 + 1e-9) + outside mass
    (1 + 1e-6) + 1e-15 >= the true tail, exactly as metrics.ni_tail_sup); kL = max k with F_lo(k) <= beta,
    k* = max k with F_hi(k) <= the Hoelder-inverted thresholds of the four adjacent half/whole intervals."""
    S = _CR
    n, m, K, rank, nodes = S["n"], S["m"], S["K"], S["rank"], S["nodes"]
    lf = M._lgamma_table(n)
    nl = len(S["beta"])
    out = {k: np.empty((nl, hi - lo), np.int64) for k in ("kL", "kAc", "kAb", "kBc", "kBa")}
    for jj, j in enumerate(range(lo, hi)):
        w = float(nodes[j])
        Wa, Wb = M._binom_window(n, w, w, lf, 1e-11)
        La, Lb = M._binom_window(n, w + m, w + m, lf, 1e-11)
        Wv, Lv = np.arange(Wa, Wb + 1)[:, None], np.arange(La, Lb + 1)[None, :]
        T = n - Wv - Lv
        ok = T >= 0
        Wc, Lc, Tc = np.broadcast_to(Wv, ok.shape)[ok], np.broadcast_to(Lv, ok.shape)[ok], T[ok]
        lw = math.log(w) if w > 0 else -1e300
        lt = math.log(1.0 - 2.0 * w - m) if 1.0 - 2.0 * w - m > 0 else -1e300
        pm = np.exp(lf[n] - lf[Wc] - lf[Lc] - lf[Tc] + Wc * lw + Lc * math.log(w + m) + Tc * lt)
        rk = rank[Wc, Lc]
        o = np.argsort(rk, kind="stable")
        rs, cs = rk[o], np.cumsum(pm[o])
        pw = np.exp(M._binom_logpmf_rows(n, [w], lf))[0]
        pl = np.exp(M._binom_logpmf_rows(n, [w + m], lf))[0]
        off = (pw[:Wa].sum() + pw[Wb + 1:].sum() + pl[:La].sum() + pl[Lb + 1:].sum()) * (1 + 1e-6) + 1e-15

        def kmax_lo(c: np.ndarray) -> np.ndarray:
            c = np.asarray(c, float)
            idx = np.searchsorted(cs, c, side="right")             # first cell whose cumulated mass exceeds c
            res = np.where(idx >= len(cs), K - 1, rs[np.minimum(idx, len(cs) - 1)] - 1)
            return np.where(c < 0, -1, res)

        def kmax_hi(logc: np.ndarray) -> np.ndarray:
            return kmax_lo((np.exp(logc) - off) / (1.0 + 1e-9))

        out["kL"][:, jj] = kmax_lo(S["beta"])
        for key, thr in (("kAc", "fA_c"), ("kAb", "fA_b"), ("kBc", "fB_c"), ("kBa", "fB_a")):
            out[key][:, jj] = kmax_hi(S[thr][:, j])
    return lo, out


def _cr_pvalue(cell: tuple[int, int]) -> tuple[tuple[int, int], dict[str, Any]]:
    """Pool worker: the deployed A6.21 p-value of one cell (metrics.ni_certified_p)."""
    r = M.ni_certified_p(cell[0], cell[1], _CR["n"], _CR["m"], exact_below=_CR["exact_below"])
    return cell, {"p": r["p"], "converged": r["converged"], "n_evals": r["n_evals"]}


class CertifiedRule:
    """Exact rejection regions of the deployed A6.21 rule (``bb_ordered``) at every level of ``levels`` for one (n, m)
    (see the module docstring). ``rho_max``: the largest discordance at which sizes / powers are evaluated; cells with
    D beyond Dmax (P(D > Dmax) < 1e-15 at rho_max) are left unrejected and their mass is reported."""

    def __init__(self, n: int, m: float, levels: dict[str, float], gamma: float = M.NI_BB_GAMMA,
                 exact_below: int = 5, eps_grid: float = 0.005, f_ref: float = 4e-3, rho_max: float = 1.0,
                 workers: int | None = None, log=print) -> None:
        t0 = time.time()
        self.n, self.m, self.gamma, self.exact_below = n, m, gamma, exact_below
        self.levels = dict(levels)
        self.lv = list(self.levels)
        self.workers = workers or os.cpu_count() or 1
        wmax = (1.0 - m) / 2.0
        st = M.ni_order_table(n, m, exact_below)
        W, L = np.indices((n + 1, n + 1))
        self.tri = W + L <= n
        self.W, self.L = W, L
        vals = np.unique(st[self.tri])
        rank = np.full((n + 1, n + 1), -1, np.int64)
        rank[self.tri] = np.searchsorted(vals, st[self.tri])
        kk_of = np.searchsorted(vals, vals + M.NI_BB_TIE, side="right") - 1     # tail = {s~ <= s~(x) + tie}
        self.rank, self.K = rank, len(vals)
        self.kk = np.where(self.tri, kk_of[np.maximum(rank, 0)], -1)
        self.D = np.where(self.tri, W + L, 0)
        # nominal nuisance sets per D (vectorised Clopper-Pearson; the deployed set is this widened by 1e-9, so nodes
        # in [a, b] lie inside it and node intervals covering [a - 2e-9, b + 2e-9] cover it)
        ex = ExactSize.__new__(ExactSize)
        ex.n, ex.lf = n, np.array([math.lgamma(k + 1) for k in range(n + 1)])
        lo, hi = ex.cp_intervals(gamma)
        self.a, self.b = np.clip((lo - m) / 2.0, 0.0, wmax), np.clip((hi - m) / 2.0, 0.0, wmax)
        if rho_max < 1.0:
            sd = math.sqrt(n * rho_max * (1.0 - rho_max))
            self.Dmax = min(n, int(n * rho_max + 9.0 * sd + 10))
        else:
            self.Dmax = n
        Dn = self.Dmax + 1
        w_top = min(wmax, float(self.b[:Dn].max()) + 2e-9)
        grid = self._arclength_nodes(w_top, eps_grid, f_ref)
        ends = np.concatenate([self.a[:Dn], self.b[:Dn], np.clip(self.a[:Dn] - 2e-9, 0.0, wmax),
                               np.clip(self.b[:Dn] + 2e-9, 0.0, wmax)])
        self.nodes = np.unique(np.concatenate([grid, ends]))
        self.info: dict[str, Any] = {"n_grid": int(len(grid)), "n_nodes": int(len(self.nodes)), "K": self.K,
                                     "Dmax": self.Dmax, "w_top": w_top, "eps_grid": eps_grid, "f_ref": f_ref}
        log(f"  n={n} m={m}: {self.K:,} distinct order values, {len(self.nodes):,} nuisance nodes "
            f"({len(grid):,} arclength grid), D <= {self.Dmax} [{time.time() - t0:.1f}s]")
        self._node_pass()
        self.info["t_nodes"] = time.time() - t0
        log(f"  node pass done [{time.time() - t0:.1f}s]")
        self._crit()
        self._gray()
        self.info["t_total"] = time.time() - t0
        log(f"  {self.info['n_gray']} gray cells decided by metrics.ni_certified_p (max evals "
            f"{self.info['gray_max_evals']}, all converged: {self.info['gray_all_converged']}) [{time.time() - t0:.1f}s]")

    def _arclength_nodes(self, w_top: float, eps_grid: float, f_ref: float) -> np.ndarray:
        """Nodes uniform in phi(w) = int_0^w sqrt(I(t)) dt (I the per-trial Fisher information of the boundary path),
        step 2 eps / sqrt(2 n log(1/f_ref)): the Hoelder bound of a half interval then inflates a tail of size f_ref
        by about exp(eps). Only the number of gray cells depends on this choice, never the computed region."""
        m = self.m
        s = np.linspace(0.0, 1.0, 400001)
        w = w_top * (1.0 - np.cos(np.pi * s)) / 2.0
        dw = w_top * np.pi / 2.0 * np.sin(np.pi * s)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.sqrt(1.0 / w + 1.0 / (w + m) + 4.0 / (1.0 - 2.0 * w - m)) * dw
        g = np.where(np.isfinite(g), g, np.nan)
        idx = np.arange(len(g))
        good = ~np.isnan(g)
        g = np.interp(idx, idx[good], g[good])
        phi = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(s))])
        dphi = 2.0 * eps_grid / math.sqrt(2.0 * self.n * math.log(1.0 / f_ref))
        nodes = np.interp(np.arange(0.0, phi[-1], dphi), phi, w)
        return np.unique(np.concatenate([nodes, [0.0, w_top]]))

    def _finv(self, logbeta: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """log of the largest F with min_q F^(1-1/q) M_q(a, x)^(n/q) (1 + 1e-9) <= beta (the inverse of
        metrics.ni_holder_logbound over the same exponent grid), shape (levels, len(a)); -inf if none."""
        qs = M._NI_Q_GRID
        bad, logm = M._ni_holder_logm(np.atleast_1d(a), np.atleast_1d(x), self.m, qs)
        out = np.empty((len(logbeta), len(np.atleast_1d(a))))
        for i, lb in enumerate(np.asarray(logbeta, float) - 1e-9):
            with np.errstate(invalid="ignore", over="ignore"):
                val = (lb - (self.n / qs) * logm) * (qs / (qs - 1.0))
            val = np.where(np.isnan(val), -np.inf, val)
            out[i] = np.where(bad, -np.inf, val.max(-1))
        return out

    def _node_pass(self) -> None:
        nodes = self.nodes
        J, nl = len(nodes), len(self.lv)
        alphas = np.array([self.levels[k] for k in self.lv])
        beta = alphas - self.gamma
        beta_c = np.minimum(beta - M.NI_BB_ABS_TOL, beta / (1.0 + M.NI_BB_REL_TOL))
        logbeta_c = np.log(np.maximum(beta_c, 1e-300))
        mid = 0.5 * (nodes[:-1] + nodes[1:])
        thr = {k: np.full((nl, J), -np.inf) for k in ("fA_c", "fA_b", "fB_c", "fB_a")}
        thr["fA_c"][:, :-1] = self._finv(logbeta_c, nodes[:-1], mid)
        thr["fA_b"][:, :-1] = self._finv(logbeta_c, nodes[:-1], nodes[1:])
        thr["fB_c"][:, 1:] = self._finv(logbeta_c, nodes[1:], mid)
        thr["fB_a"][:, 1:] = self._finv(logbeta_c, nodes[1:], nodes[:-1])
        _CR.clear()
        _CR.update({"n": self.n, "m": self.m, "K": self.K, "rank": self.rank, "nodes": nodes, "beta": beta,
                    "exact_below": self.exact_below, **thr})
        chunks = [(i, min(J, i + 256)) for i in range(0, J, 256)]
        res = {k: np.empty((nl, J), np.int64) for k in ("kL", "kAc", "kAb", "kBc", "kBa")}
        if self.workers > 1:
            with mp.get_context("fork").Pool(self.workers) as pool:
                parts = pool.starmap(_cr_nodes, chunks)
        else:
            parts = [_cr_nodes(lo, hi) for lo, hi in chunks]
        for lo, part in parts:
            for k, v in part.items():
                res[k][:, lo:lo + v.shape[1]] = v
        self.kL = res["kL"]
        # interval [j, j+1]: split at the midpoint (both halves), or one end point covering the whole interval
        self.kI = np.maximum(np.minimum(res["kAc"][:, :-1], res["kBc"][:, 1:]),
                             np.maximum(res["kAb"][:, :-1], res["kBa"][:, 1:]))

    def _crit(self) -> None:
        nodes, nl = self.nodes, len(self.lv)
        wmax = (1.0 - self.m) / 2.0
        self.critL = np.full((nl, self.n + 1), -1, np.int64)       # D > Dmax: never rejected (mass reported)
        self.critC = np.full((nl, self.n + 1), -1, np.int64)
        for D in range(self.Dmax + 1):
            a, b = self.a[D], self.b[D]
            j0, j1 = np.searchsorted(nodes, a, side="left"), np.searchsorted(nodes, b, side="right") - 1
            self.critL[:, D] = self.kL[:, j0:j1 + 1].min(1)
            lo, hi = max(0.0, a - 2e-9), min(wmax, b + 2e-9)
            i0 = max(0, int(np.searchsorted(nodes, lo, side="right")) - 1)
            i1 = min(len(nodes) - 2, max(i0, int(np.searchsorted(nodes, hi, side="left")) - 1))
            self.critC[:, D] = self.kI[:, i0:i1 + 1].min(1)
        if np.any(self.critC > self.critL):
            raise AssertionError("conservative critical rank above the lower one: bracket implementation error")

    def _gray(self) -> None:
        nl = len(self.lv)
        cells: set[tuple[int, int]] = set()
        self.gray_masks = []
        for i in range(nl):
            g = self.tri & (self.kk > self.critC[i][self.D]) & (self.kk <= self.critL[i][self.D]) & (self.D <= self.Dmax)
            self.gray_masks.append(g)
            cells |= {(int(wi), int(li)) for wi, li in zip(*np.nonzero(g))}
        cells_l = sorted(cells)
        if self.workers > 1 and len(cells_l) > 8:
            with mp.get_context("fork").Pool(self.workers) as pool:
                res = dict(pool.map(_cr_pvalue, cells_l, chunksize=4))
        else:
            res = dict(_cr_pvalue(c) for c in cells_l)
        self.gray_p = res
        self.info.update({"n_gray": len(cells_l), "gray_per_level": [int(g.sum()) for g in self.gray_masks],
                          "gray_max_evals": max((r["n_evals"] for r in res.values()), default=0),
                          "gray_all_converged": all(r["converged"] for r in res.values())})
        self.reject = np.zeros((nl, self.n + 1, self.n + 1), bool)
        P = np.full((self.n + 1, self.n + 1), np.inf)
        for (wi, li), r in res.items():
            P[wi, li] = r["p"]
        for i, lv in enumerate(self.lv):
            self.reject[i] = (self.tri & (self.kk <= self.critC[i][self.D]) & (self.D <= self.Dmax)) | \
                             (self.gray_masks[i] & (P <= self.levels[lv]))

    def check_brackets(self, per_level: int = 6, seed: int = 0) -> dict[str, Any]:
        """Verify the two bracket implications on cells just inside crit_C and just outside crit_L (random D per level):
        the deployed p must be <= alpha resp. > alpha. Returns the counts and any violation."""
        rng = np.random.default_rng(seed)
        todo: list[tuple[str, int, int, bool]] = []
        for i, lv in enumerate(self.lv):
            for D in rng.choice(np.arange(5, self.Dmax + 1), size=min(per_level, self.Dmax - 4), replace=False):
                Wl = np.arange(D + 1)
                kkD = self.kk[Wl, D - Wl]
                for crit, want in ((self.critC[i][D], True), (self.critL[i][D], False)):
                    sel = np.nonzero(kkD <= crit)[0] if want else np.nonzero(kkD > crit)[0]
                    if len(sel):
                        j = sel[np.argmax(kkD[sel])] if want else sel[np.argmin(kkD[sel])]
                        todo.append((lv, int(Wl[j]), int(D - Wl[j]), want))
        cells = sorted({(wi, li) for _, wi, li, _ in todo})
        _CR.update({"n": self.n, "m": self.m, "exact_below": self.exact_below})
        if self.workers > 1 and len(cells) > 8:
            with mp.get_context("fork").Pool(self.workers) as pool:
                ps = dict(pool.map(_cr_pvalue, cells, chunksize=2))
        else:
            ps = dict(_cr_pvalue(c) for c in cells)
        bad = [{"level": lv, "W": wi, "L": li, "p": ps[(wi, li)]["p"], "expected_reject": want}
               for lv, wi, li, want in todo if (ps[(wi, li)]["p"] <= self.levels[lv]) != want]
        return {"checked": len(todo), "cells": len(cells), "violations": bad}

    def logpmf(self, p_win: float, p_loss: float) -> np.ndarray:
        n = self.n
        p_tie = 1.0 - p_win - p_loss
        if min(p_win, p_loss) < 0 or p_tie < -1e-12:
            raise ValueError(f"infeasible (p_win, p_loss) = ({p_win}, {p_loss})")
        lf = M._lgamma_table(n)
        W, L = self.W, self.L
        T = np.clip(n - W - L, 0, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            def term(k, p):
                return np.where(k == 0, 0.0, k * math.log(p)) if p > 0 else np.where(k == 0, 0.0, -np.inf)
            lp = lf[n] - lf[W] - lf[L] - lf[T] + term(W, p_win) + term(L, p_loss) + term(T, max(p_tie, 0.0))
        return np.where(self.tri, lp, -np.inf)

    def probs(self, p_win: float, p_loss: float) -> dict[str, Any]:
        """Exact rejection probability of the deployed rule at every level under the trinomial (p_win, p_loss)."""
        pm = np.exp(self.logpmf(p_win, p_loss))
        beyond = float(pm[self.tri & (self.D > self.Dmax)].sum())
        return {"reject": {lv: float(pm[self.reject[i]].sum()) for i, lv in enumerate(self.lv)},
                "mass_beyond_Dmax": beyond, "total_mass": float(pm.sum())}


# ------------------------------------------------------------------ aggregation
def run(prune: float, exact_below: int, log=print) -> dict[str, Any]:
    results: dict[str, Any] = {"levels": LEVELS, "holm_levels": HOLM_LEVELS, "w_grid": W_GRID, "prune": prune,
                               "exact_below": exact_below, "flag_rel": FLAG_REL, "settings": []}
    t0 = time.time()
    for n, m, label in SETTINGS:
        log(f"[{time.time() - t0:6.1f}s] n={n} m={m}: building tables")
        ex = ExactSize(n, m, exact_below)
        chk_exact = ex.check_exact_table()
        log(f"[{time.time() - t0:6.1f}s] n={n} m={m}: A6.17 Berger-Boos p-values on the Tango branch's region")
        bb = ex.bb_pvalues(prune)
        grid = []
        for w in W_GRID:
            grid.append(ex.sizes(w, prune))
        log(f"[{time.time() - t0:6.1f}s] n={n} m={m}: {ex.n_tango_evals} Tango evaluations, cross-checking the interval")
        chk_tango = ex.check_tango_interval()
        st: dict[str, Any] = {"n": n, "m": m, "label": label, "grid": grid, "exact_crit": ex.exact_crit, "bb": bb,
                              "checks": {"exact_table_vs_ni_exact_check": chk_exact, "tango_vs_interval": chk_tango,
                                         "tango_pairs_evaluated": ex.n_tango_evals,
                                         "max_abs_dev_full_triangle_sum": max(abs(1 - g["full_triangle_sum"]) for g in grid),
                                         "max_pruned_mass": max(g["pruned_mass"] for g in grid),
                                         "max_pairs_kept": max(g["pairs_kept"] for g in grid)},
                              "max": {}, "flags": []}
        for lv, a in LEVELS.items():
            st["max"][lv] = {}
            for rule in VARIANTS:
                i = int(np.argmax([g["size"][lv][rule] for g in grid]))
                s = grid[i]["size"][lv][rule]
                st["max"][lv][rule] = {"size": s, "w": grid[i]["w"], "ratio": s / a}
            for g in grid:
                s = g["size"][lv]["registered"]
                if s > FLAG_REL * a:
                    st["flags"].append({"level": lv, "alpha": a, "w": g["w"], "size": s, "ratio": s / a})
                su = g["size"][lv]["registered_u"]
                if su > a * (1 + 1e-9):
                    st.setdefault("flags_u", []).append({"level": lv, "alpha": a, "w": g["w"], "size": su, "ratio": su / a})
        st.setdefault("flags_u", [])
        results["settings"].append(st)
    results["runtime_s"] = time.time() - t0
    return results


# ------------------------------------------------------------------ report
def _lvl(lv: str) -> str:
    return lv if lv == "0.10" else (lv.replace("/1", "") if lv == "0.05/1" else lv)


def _st(results: dict[str, Any], n: int, m: float) -> dict[str, Any]:
    return next(s for s in results["settings"] if s["n"] == n and s["m"] == m)


def _cell(s: float, a: float, bold_if_flag: bool = True) -> str:
    txt = f"{s:.5f} ({s / a:.3f}×)"
    return f"**{txt}**" if bold_if_flag and s > FLAG_REL * a else txt


def reading(results: dict[str, Any]) -> list[str]:
    """Six lines computed from the numbers (so they cannot drift from the tables)."""
    fl, top, ref = "0.05/11", "0.05/1", "0.10"
    sts = results["settings"]
    cells = [(st, g, lv, LEVELS[lv]) for st in sts for g in st["grid"] for lv in HOLM_LEVELS]

    def ratio(c, rule="registered"):
        return c[1]["size"][c[2]][rule] / c[3]

    def where(c):
        return f"n = {c[0]['n']}, m = {c[0]['m']}, w = {c[1]['w']}, α = {_lvl(c[2])}"

    worst = max(cells, key=ratio)
    best = min(cells, key=ratio)
    flags_holm = [(st, f) for st in sts for f in st["flags"] if f["level"] != ref]
    flags_ref = [(st, f) for st in sts for f in st["flags"] if f["level"] == ref]
    per_n = {st["n"]: max(ratio(c) for c in cells if c[0] is st) for st in sts}
    per_n_txt = ", ".join(f"{per_n[n]:.3f} α at n = {n}" for n in sorted(per_n, key=lambda n: -per_n[n]))
    flagged_switch = max((f_st["grid"][W_GRID.index(f["w"])]["P_switch"] for f_st, f in flags_holm + flags_ref), default=0.0)
    # where the maximum over w sits
    argmax_w = [st["max"][lv]["registered"]["w"] for st in sts for lv in HOLM_LEVELS]
    n_big = sum(w >= 0.15 for w in argmax_w)
    tango_worst = max(cells, key=lambda c: ratio(c, "tango_only"))
    exact_worst = max(cells, key=lambda c: ratio(c, "exact_only"))
    # the switch region
    small_max = {st["n"]: max(ratio(c) for c in cells if c[0] is st and c[1]["w"] <= 0.01) for st in sts}
    g0 = {st["n"]: st["grid"][0] for st in sts}
    lstar = {st["n"]: st["exact_crit"][fl]["l_star"] for st in sts}

    def r0(n, lv, rule):
        return g0[n]["size"][lv][rule] / LEVELS[lv]

    s500, s149, s300, s1000 = (_st(results, 500, 0.03), _st(results, 149, 0.05), _st(results, 300, 0.03), _st(results, 1000, 0.03))
    # effect of the W + L < 5 part of the switch beyond the W = 0 part (switch_w0 - registered: positive = it lowers the size)
    lt5_effect, lt5_min = [], 0.0
    for st in sts:
        d = max((g["size"][lv]["switch_w0"] - g["size"][lv]["registered"], g["w"], lv) for g in st["grid"] for lv in LEVELS)
        lt5_min = min(lt5_min, min(g["size"][lv]["switch_w0"] - g["size"][lv]["registered"] for g in st["grid"] for lv in LEVELS))
        lt5_effect.append((st["n"], d, max(g["P_lt5"] for g in st["grid"])))
    lt5_txt = "; ".join(f"n = {n}: P(W + L < 5) ≤ {p:.4f}, largest change {d[0]:.5f} (w = {d[1]}, α = {_lvl(d[2])})" for n, d, p in lt5_effect)
    lt5_dir = "it never raises the size" if lt5_min >= -1e-12 else f"it raises the size by at most {-lt5_min:.5f} somewhere"
    # Tango alone on losses-only data: where it over-rejects at 0.05 and at 0.10
    over05 = [n for n in (149, 300, 500, 1000) if r0(n, top, "tango_only") > 1]
    same05 = [n for n in (149, 300, 500, 1000) if abs(g0[n]["size"][top]["tango_only"] - g0[n]["size"][top]["registered"]) < 1e-12]
    over10 = [n for n in (149, 300, 500, 1000) if r0(n, ref, "tango_only") > 1]
    lines = []
    lines.append(
        f"1. **Size.** Over the {len(sts)} × {len(W_GRID)} × {len(HOLM_LEVELS)} = {len(cells)} (n, m, w, Holm-level) cells the registered rule's exact size "
        f"exceeds 1.02 α in **{len(flags_holm)}** cell(s)"
        + (": " + "; ".join(f"{f['size']:.5f} = {f['ratio']:.3f} α (+{f['size'] - f['alpha']:.5f} absolute) at n = {st['n']}, w = {f['w']}, α = {_lvl(f['level'])}" for st, f in flags_holm) if flags_holm else "")
        + ("; at the 0.10 reference level also " + ", ".join(f"{f['ratio']:.3f} α at n = {st['n']}, w = {f['w']}" for st, f in flags_ref) if flags_ref else "")
        + f". The largest size relative to its level is {ratio(worst):.3f} α at {where(worst)}; per n the maximum over all Holm levels and w is {per_n_txt}. "
        f"So the rule holds its size to within {100 * (max(per_n.values()) - 1):.1f} % relative at every Holm level and nuisance value, and to within "
        f"{100 * (max(v for n, v in per_n.items() if n != 149) - 1):.1f} % at n ≥ 300; every exceedance is a cell where Tango decides "
        f"(P(switch) ≤ {flagged_switch:.1e} there) and is real (no Monte-Carlo error, dropped mass ≤ {max(st['checks']['max_pruned_mass'] for st in sts):.0e}).")
    lines.append(
        f"2. **Most anti-conservative.** The maximum over w sits at w ≥ 0.15 (≥ 33 % discordant tasks) in {n_big} of {len(argmax_w)} (setting, level) pairs: "
        f"there the score test's exact size oscillates about α from both sides (discreteness of the counts), with an amplitude that shrinks with n "
        f"({', '.join(f'{per_n[n]:.3f}' for n in (149, 300, 500, 1000))} α at n = 149 / 300 / 500 / 1000) and is largest at n = 149, the smallest sample. "
        f"Tango alone is at most {ratio(tango_worst, 'tango_only'):.3f} α ({where(tango_worst)}; losses-only data, see line 4); the exact-only rule at most "
        f"{ratio(exact_worst, 'exact_only'):.3f} α ({where(exact_worst)}): it is exactly sized at w = 0 up to discreteness and tests the stronger condition "
        f"P(loss) < m, so for every w > 0 its size falls below α and toward 0.")
    lines.append(
        f"3. **Most conservative.** The smallest size relative to its level is {ratio(best):.3f} α at {where(best)}. Near w = 0 the decision is the exact "
        f"binomial test, whose discreteness leaves a gap below α (critical loss count l* at the floor: {lstar[149]} at n = 149, {lstar[300]} at n = 300, "
        f"{lstar[500]} at n = 500, {lstar[1000]} at n = 1000), and for 0 < w ≲ 0.01 the mixture stays well below the level (the exact test because P(loss) = w + m > m; "
        f"line 4); at the floor the registered size at w = 0 is {g0[500]['size'][fl]['registered']:.5f} ({r0(500, fl, 'registered'):.3f} α, n = 500), "
        f"{g0[149]['size'][fl]['registered']:.5f} ({r0(149, fl, 'registered'):.3f} α, n = 149), {g0[300]['size'][fl]['registered']:.5f} ({r0(300, fl, 'registered'):.3f} α, n = 300), "
        f"{g0[1000]['size'][fl]['registered']:.5f} ({r0(1000, fl, 'registered'):.3f} α, n = 1000) — the price of exactness on losses-only data, not a defect of the switch.")
    lines.append(
        f"4. **What the switch does near w = 0.** On losses-only data (w = 0) Tango alone (size at w = 0, n = 149 / 300 / 500 / 1000: "
        f"{r0(149, top, 'tango_only'):.3f} / {r0(300, top, 'tango_only'):.3f} / {r0(500, top, 'tango_only'):.3f} / {r0(1000, top, 'tango_only'):.3f} α at α = 0.05; "
        f"{r0(149, ref, 'tango_only'):.3f} / {r0(300, ref, 'tango_only'):.3f} / {r0(500, ref, 'tango_only'):.3f} / {r0(1000, ref, 'tango_only'):.3f} α at 0.10; "
        f"{r0(149, fl, 'tango_only'):.3f} / {r0(300, fl, 'tango_only'):.3f} / {r0(500, fl, 'tango_only'):.3f} / {r0(1000, fl, 'tango_only'):.3f} α at the floor) "
        f"over-rejects at α = 0.05 for n = {', '.join(map(str, over05))} and at 0.10 for n = {', '.join(map(str, over10))}, and is over-conservative at the floor "
        f"for n = {', '.join(str(n) for n in (149, 300, 500, 1000) if r0(n, fl, 'tango_only') < 0.5)}"
        + (f" (at n = {', '.join(map(str, same05))} its W = 0 rejection region at 0.05 coincides with the exact test's)" if same05 else "") +
        f". The W = 0 fallback replaces these by the exact test's size "
        f"({r0(149, top, 'registered'):.3f} / {r0(300, top, 'registered'):.3f} / {r0(500, top, 'registered'):.3f} / {r0(1000, top, 'registered'):.3f} α at 0.05; "
        f"{r0(149, ref, 'registered'):.3f} / {r0(300, ref, 'registered'):.3f} / {r0(500, ref, 'registered'):.3f} / {r0(1000, ref, 'registered'):.3f} α at 0.10; "
        f"{r0(149, fl, 'registered'):.3f} / {r0(300, fl, 'registered'):.3f} / {r0(500, fl, 'registered'):.3f} / {r0(1000, fl, 'registered'):.3f} α at the floor): "
        f"it removes every over-rejection on losses-only data and, where Tango is over-conservative at the floor, restores size (hence power). The W + L < 5 part of "
        f"the switch changes the size only where P(W + L < 5) is non-negligible ({lt5_txt}) and {lt5_dir}. Over the switch region w ≤ 0.01 the registered "
        f"size stays at or below {small_max[500]:.3f} / {small_max[149]:.3f} / {small_max[300]:.3f} / {small_max[1000]:.3f} α (n = 500 / 149 / 300 / 1000) at every Holm level, "
        f"so the data-dependent switch never pushes the mixture above the level: the only cells above α are Tango-decided cells at w ≥ 0.15.")
    lines.append(
        f"5. **C8 bracket.** At m = 0.03 the registered rule's maximum size over w is "
        f"{s300['max'][fl]['registered']['size']:.5f} ({s300['max'][fl]['registered']['ratio']:.3f} α) at n = 300 and "
        f"{s1000['max'][fl]['registered']['size']:.5f} ({s1000['max'][fl]['registered']['ratio']:.3f} α) at n = 1000 at the floor, "
        f"{s300['max'][top]['registered']['size']:.5f} ({s300['max'][top]['registered']['ratio']:.3f} α) and "
        f"{s1000['max'][top]['registered']['size']:.5f} ({s1000['max'][top]['registered']['ratio']:.3f} α) at α = 0.05 "
        f"(n = 500: {s500['max'][fl]['registered']['size']:.5f} = {s500['max'][fl]['registered']['ratio']:.3f} α and "
        f"{s500['max'][top]['registered']['size']:.5f} = {s500['max'][top]['registered']['ratio']:.3f} α); over all eleven Holm levels the maximum is "
        f"{per_n[300]:.3f} α at n = 300 and {per_n[1000]:.3f} α at n = 1000, so any retained C8 count in [300, 1000] is covered to within about 1 % relative; "
        f"the exact computation is re-run at the final count by adding it to `SETTINGS` (≈ 10 s); the registered C8 "
        f"primary size n = 1114 (A6.18 (d)) is its own row: registered max {_st(results, 1114, 0.03)['max'][fl]['registered']['ratio']:.3f} α at "
        f"the floor and {_st(results, 1114, 0.03)['max'][top]['registered']['ratio']:.3f} α at 0.05, A6.17 rule "
        f"{_st(results, 1114, 0.03)['max'][fl]['registered_u']['ratio']:.3f} α / {_st(results, 1114, 0.03)['max'][top]['registered_u']['ratio']:.3f} α.")
    lines.append(
        "6. **Must anything change?** Not the rule: the registered A6.14 procedure (Tango p_T ≤ α_adj with the exact binomial fallback at W = 0 or "
        f"W + L < 5, strict p < α_adj) controls the type-I error at Δ = −m to within {100 * (max(per_n.values()) - 1):.1f} % relative over the whole grid and all "
        f"eleven Holm levels, the excess being the score component's finite-sample discreteness at n = 149 (≤ {max(f['size'] - f['alpha'] for st, f in flags_holm) if flags_holm else 0:.5f} absolute) "
        "and ≤ 1.1 % at n ≥ 300; the switch itself never adds size. The wording must change: replace 'holds its size within Monte-Carlo error' by this "
        "exact-summation statement and keep the reviewer's caveat that the score component is an approximate (asymptotic) procedure, not a finite-sample "
        "exact NI test — a strictly ≤ α rule at n = 149 would need an exact unconditional (nuisance-maximised) test, which the reviewer explicitly did not ask for.")
    # 7. the A6.17 rule
    if all("registered_u" in g["size"][fl] for st in sts for g in st["grid"]):
        ru_max = max(cells, key=lambda c: ratio(c, "registered_u"))
        ru_per_n = {st["n"]: max(ratio(c, "registered_u") for c in cells if c[0] is st) for st in sts}
        u_min = min(cells, key=lambda c: ratio(c, "registered_u"))
        pow_loss = []
        for st in sts:
            # power cost of p_U over p_T at the floor and at 0.05: size ratio registered_u / registered at the largest-w cells
            for lv in (fl, top):
                g = st["grid"][-1]
                pow_loss.append((st["n"], _lvl(lv), g["size"][lv]["registered_u"] / max(g["size"][lv]["registered"], 1e-300)))
        pl_txt = "; ".join(f"n = {n}, α = {lv}: {r:.3f}" for n, lv, r in pow_loss)
        flags_u = [(st, f) for st in sts for f in st.get("flags_u", [])]
        lines.append(
            "7. **A6.17 (Berger-Boos exact unconditional p of the Tango statistic on the branch's region, `metrics.ni_unconditional_p`).** Replacing p_T "
            "by p_U = max over w in the (1 − γ) Clopper-Pearson set for w (from the discordant count; γ = 1e-4) of P_w[Z ≥ z_obs, W ≥ 1, W + L ≥ 5] + γ, "
            "w on the grid {0, 0.0005, 0.001, 0.0025, 0.005, …, (1 − m)/2}, gives a valid p-value (Berger & Boos 1994) whatever w is. "
            "On the w values of this table, mostly off that grid, the combined A6.17 rule (*registered_u*: p_U with the W = 0 / W + L < 5 "
            f"exact fallback) has maximum size {ru_max[1]['size'][ru_max[2]]['registered_u']:.5f} = {ratio(ru_max, 'registered_u'):.3f} α at {where(ru_max)} "
            f"(per n: {', '.join(f'{ru_per_n[n]:.3f} α at n = {n}' for n in sorted(ru_per_n, key=lambda n: -ru_per_n[n]))}); "
            + (f"**{len(flags_u)} off-grid cell(s) exceed α** (largest {max(f['ratio'] for _, f in flags_u):.4f} α)" if flags_u else
               "no cell exceeds α, at any level, including the 0.10 reference")
            + f"; the most conservative cell is {ratio(u_min, 'registered_u'):.3f} α at {where(u_min)} (the exact fallback's discreteness, as in line 3). "
            f"Size of the A6.17 rule relative to the A6.14 rule at the largest-discordance cell (w = {sts[0]['grid'][-1]['w']}; the power cost of exactness "
            f"under the null, ≈ the ratio of the two rejection probabilities near the boundary): {pl_txt}. The switch (exact fallback) never raises the size "
            "of the p_U branch either (§3 columns).")
    return lines

def write_md(path: Path, results: dict[str, Any], cmd: str) -> None:
    L: list[str] = []
    L.append("# Exact size of the registered NI decision rules (A6.14, A6.17) over a dense nuisance grid\n")
    L.append(f"`scripts/ni_exact_size.py`, exact summation over the paired trinomial counts (no Monte Carlo); raw numbers in "
             f"`ni_exact_size.json`. Date {results['date']}. Boundary null Δ = −m: per task P(win) = w, P(loss) = w + m, "
             f"P(tie) = 1 − 2w − m; size(w) = Σ_(W+L≤n) Multinomial(n; w, w+m, 1−2w−m)(W, L) · 1[rule rejects at (W, L)]. "
             f"w grid: {', '.join(str(w) for w in W_GRID)} ({len(W_GRID)} values). Levels: the eleven Holm levels of an 11-test family "
             f"α_k = 0.05/(12 − k), k = 1..11 (0.05/11 = {LEVELS['0.05/11']:.5f} … 0.05), plus 0.10 for reference. "
             f"Pairs with pmf < {results['prune']:g} are dropped (largest dropped mass over all cells: "
             f"{max(st['checks']['max_pruned_mass'] for st in results['settings']):.2e}; largest |1 − Σ pmf| over the full triangle: "
             f"{max(st['checks']['max_abs_dev_full_triangle_sum'] for st in results['settings']):.1e}).\n")
    L.append("**Rules.** *registered* (A6.14): Tango's one-sided score test p_T = 1 − Φ(Z(−m)) (`metrics._tango_z`, `_norm_sf`), reject iff "
             f"p_T ≤ α, except that whenever W = 0 or W + L < {results['exact_below']} the exact one-sided binomial test p = P[L ≤ l | n, m] "
             "(`metrics.ni_exact_check(...)['p']`, strict p < α as registered) decides. *exact_only*: the exact test for every comparison. "
             "*tango_only*: Tango for every comparison. The switch comparison (§3) adds *switch_w0* (exact at W = 0 only) and *switch_lt5* "
             f"(exact at W + L < {results['exact_below']} only). *registered_u* (A6.17): the Tango branch decides by its Berger-Boos exact "
             "unconditional p-value p_U = max over w in the (1 − γ) confidence set of P_w[Z ≥ z_obs, not switched] + γ (`metrics.ni_unconditional_p`; "
             "nuisance grid `metrics.ni_unconditional_w_grid`, step 0.0025; γ = 1e-4) with the same exact fallback; *uncond_only*: that branch alone. "
             "Bold: registered (A6.14) size > "
             f"{FLAG_REL:.2f} α; in §1b bold marks an A6.17 size above α.\n")
    # ---- (1) max size per (n, m, level)
    L.append("## 1. Maximum size over the w grid per (n, m, level), with the w at which it occurs\n")
    L.append("| n | m | α | registered: max size (×α) | at w | exact_only: max (×α) | at w | tango_only: max (×α) | at w |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for st in results["settings"]:
        for lv, a in LEVELS.items():
            mx = st["max"][lv]
            L.append(f"| {st['n']} | {st['m']} | {_lvl(lv)} = {a:.5f} | {_cell(mx['registered']['size'], a)} | {mx['registered']['w']} | "
                     f"{_cell(mx['exact_only']['size'], a, False)} | {mx['exact_only']['w']} | "
                     f"{_cell(mx['tango_only']['size'], a, False)} | {mx['tango_only']['w']} |")
    L.append("")
    flags = [(st, f) for st in results["settings"] for f in st["flags"]]
    if flags:
        L.append(f"**Flagged cells (registered size > {FLAG_REL:.2f} α):**\n")
        for st, f in flags:
            L.append(f"- n = {st['n']}, m = {st['m']}, w = {f['w']}, α = {_lvl(f['level'])}: size {f['size']:.5f} = {f['ratio']:.3f} α")
    else:
        L.append(f"No cell of the registered rule exceeds {FLAG_REL:.2f} α at any (n, m, w, level), including the 0.10 reference level.")
    L.append("")
    # ---- (1b) the A6.17 rule
    if all("bb" in st for st in results["settings"]):
        bb0 = results["settings"][0]["bb"]
        L.append("## 1b. A6.17 rule (Berger-Boos exact unconditional p_U of the Tango statistic on its branch + the same exact fallback): maximum size over the w grid\n")
        L.append(f"p_U = max over the grid w (step {bb0['step']}) inside the (1 − γ) Clopper-Pearson set for w (γ = {bb0['gamma']:g}, from the discordant "
                 "count D = W + L) of P_w[Z ≥ z_obs, W ≥ 1, W + L ≥ 5] + γ; the w columns are this table's grid, mostly off the maximisation grid, "
                 "and the maximisation here omits the set's end points and the point estimate that `metrics.ni_unconditional_p` adds, so these sizes "
                 "bound the deployed rule's from above.\n")
        L.append("| n | m | α | registered_u: max size (×α) | at w | uncond_only (branch alone): max (×α) | at w | A6.14 registered: max (×α) | at w |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for st in results["settings"]:
            for lv, a in LEVELS.items():
                mx = st["max"][lv]
                su = mx["registered_u"]["size"]
                cell_u = f"{su:.5f} ({su / a:.3f}×)"
                if su > a * (1 + 1e-9):
                    cell_u = f"**{cell_u}**"
                L.append(f"| {st['n']} | {st['m']} | {_lvl(lv)} = {a:.5f} | {cell_u} | {mx['registered_u']['w']} | "
                         f"{_cell(mx['uncond_only']['size'], a, False)} | {mx['uncond_only']['w']} | "
                         f"{_cell(mx['registered']['size'], a, False)} | {mx['registered']['w']} |")
        L.append("")
        flags_u = [(st, f) for st in results["settings"] for f in st.get("flags_u", [])]
        if flags_u:
            L.append("**A6.17 cells above α (off-grid residual):**\n")
            for st, f in flags_u:
                L.append(f"- n = {st['n']}, m = {st['m']}, w = {f['w']}, α = {_lvl(f['level'])}: size {f['size']:.6f} = {f['ratio']:.4f} α")
        else:
            L.append("No cell of the A6.17 rule exceeds α at any (n, m, w, level) of this table, including the 0.10 reference level.")
        L.append("")
    # ---- (2) full grid for the registered rule at the three key levels
    L.append("## 2. Registered rule: exact size over the full w grid at α = 0.05/11, 0.05/6, 0.05\n")
    L.append("P(switch) = P(W = 0 or W + L < 5) under the boundary null, the probability that the exact fallback decides.\n")
    for st in results["settings"]:
        L.append(f"### n = {st['n']}, m = {st['m']} — {st['label']}\n")
        L.append("| w | P(loss) | E[W] | E[L] | P(switch) | " + " | ".join(f"α = {_lvl(lv)} ({LEVELS[lv]:.5f})" for lv in KEY_LEVELS) + " |")
        L.append("|---|---|---|---|---|" + "---|" * len(KEY_LEVELS))
        for g in st["grid"]:
            L.append(f"| {g['w']} | {g['p_loss']:.4f} | {g['E_wins']:.2f} | {g['E_losses']:.1f} | {g['P_switch']:.4f} | "
                     + " | ".join(_cell(g["size"][lv]["registered"], LEVELS[lv]) for lv in KEY_LEVELS) + " |")
        L.append("")
    # ---- (3) the switch comparison
    L.append("## 3. Switch comparison (what the exact fallback does near w = 0)\n")
    L.append("Exact size of the Tango test with no fallback, with the fallback at W = 0 only, at W + L < 5 only, at both (the registered rule), "
             "and of the exact test alone.\n")
    for n, m in SWITCH_SETTINGS:
        st = _st(results, n, m)
        for lv in ("0.05/11", "0.05/1"):
            a = LEVELS[lv]
            L.append(f"### n = {n}, m = {m}, α = {_lvl(lv)} = {a:.5f}\n")
            L.append("| w | P(W = 0) | P(W + L < 5) | " + " | ".join(VARIANT_DESC[v] for v in VARIANTS) + " |")
            L.append("|---|---|---|" + "---|" * len(VARIANTS))
            for g in st["grid"]:
                L.append(f"| {g['w']} | {g['P_W0']:.4f} | {g['P_lt5']:.4f} | "
                         + " | ".join(_cell(g["size"][lv][v], a, v == "registered") for v in VARIANTS) + " |")
            L.append("")
    # ---- appendix: registered size / alpha at all twelve levels
    L.append("## 3b. Registered rule at all twelve levels: size / α over the w grid\n")
    L.append("Ratio of the exact size to the level (bold > 1.02); the eleven Holm levels of an 11-test family and the 0.10 reference.\n")
    for st in results["settings"]:
        L.append(f"### n = {st['n']}, m = {st['m']}\n")
        L.append("| w | " + " | ".join(_lvl(lv) for lv in LEVELS) + " |")
        L.append("|---|" + "---|" * len(LEVELS))
        for g in st["grid"]:
            cells = []
            for lv, a in LEVELS.items():
                r = g["size"][lv]["registered"] / a
                cells.append(f"**{r:.3f}**" if r > FLAG_REL else f"{r:.3f}")
            L.append(f"| {g['w']} | " + " | ".join(cells) + " |")
        L.append("")
    # ---- (4) reading
    L.append("## 4. Reading\n")
    L.extend(results["reading"])
    L.append("")
    # ---- checks
    L.append("## 5. Checks\n")
    for st in results["settings"]:
        c = st["checks"]
        L.append(f"- n = {st['n']}, m = {st['m']}: exact p table (`metrics._binom_cdf`) vs `ni_exact_check(...)['p']` for l ≤ "
                 f"{c['exact_table_vs_ni_exact_check']['l_checked_upto']} and wins ∈ {{0, 1, 3}}: max |Δp| = "
                 f"{c['exact_table_vs_ni_exact_check']['max_abs_dev']:.1e}; Tango p_T evaluated on {c['tango_pairs_evaluated']:,} (W, L) pairs "
                 f"(at most {c['max_pairs_kept']:,} kept per w); p_T ≤ α vs `tango_interval(one_sided=True)['ci'][0] > −m` on the "
                 f"{c['tango_vs_interval']['checked']} pairs nearest the twelve levels: {len(c['tango_vs_interval']['disagreements'])} disagreements; "
                 f"full-triangle Σ pmf deviates from 1 by at most {c['max_abs_dev_full_triangle_sum']:.1e}; dropped mass ≤ {c['max_pruned_mass']:.1e}. "
                 f"Exact-test critical loss counts l* (p < α) at the floor / 0.05: {st['exact_crit']['0.05/11']['l_star']} / "
                 f"{st['exact_crit']['0.05/1']['l_star']} with sizes {st['exact_crit']['0.05/11']['size_at_m']:.5f} / "
                 f"{st['exact_crit']['0.05/1']['size_at_m']:.5f} at P(loss) = m (= the registered size at w = 0).")
    L.append("")
    L.append("## 6. Command and runtime\n")
    L.append(f"```\n{cmd}\n```\n")
    L.append(f"Runtime {results['runtime_s'] / 60:.1f} min ({results['runtime_s']:.0f} s) on {results['cpu']} CPU threads "
             f"(numpy {np.__version__}, Python {sys.version.split()[0]}); no scipy (log-gamma multinomial pmf).\n")
    path.write_text("\n".join(L))


# ------------------------------------------------------------------ round 8: the A6.21 rule (bb_ordered)
W_DENSE = sorted(set(W_GRID) | {round(k * 0.0025, 6) for k in range(0, 161)} | {0.0001, 0.00025, 0.00075, 0.0015})
INTERIOR_E = [0.01, 0.03, 0.10]                       # P(loss) - P(win) - m > 0: strictly inside the null
INTERIOR_W = [0.0, 0.005, 0.02, 0.05, 0.10, 0.20]
RHO_MAX = 0.86                                         # >= the largest discordance evaluated (W_DENSE max 0.4: 2w + m <= 0.85)


def run_r8(prune: float, exact_below: int, eps_grid: float = 0.005, log=print) -> dict[str, Any]:
    """Round-7 tables (A6.14 / A6.17 columns) plus the exact size of the deployed A6.21 rule at the W_GRID points, a
    dense boundary grid (W_DENSE) and interior null points, at the eleven Holm levels and 0.10."""
    results = run(prune, exact_below, log=log)
    t0 = time.time()
    results.update({"w_dense": W_DENSE, "interior_e": INTERIOR_E, "interior_w": INTERIOR_W, "rho_max": RHO_MAX,
                    "eps_grid": eps_grid, "gamma": M.NI_BB_GAMMA, "rel_tol": M.NI_BB_REL_TOL,
                    "abs_tol": M.NI_BB_ABS_TOL, "tie": M.NI_BB_TIE})
    assert max(2 * w + st["m"] for st in results["settings"] for w in W_DENSE) <= RHO_MAX
    assert max(2 * w + st["m"] + e for st in results["settings"] for w in INTERIOR_W for e in INTERIOR_E) <= RHO_MAX
    for st in results["settings"]:
        n, m = st["n"], st["m"]
        log(f"[{time.time() - t0:6.1f}s] n={n} m={m}: A6.21 exact rejection regions")
        cr = CertifiedRule(n, m, LEVELS, exact_below=exact_below, eps_grid=eps_grid, rho_max=RHO_MAX, log=log)
        chk = cr.check_brackets(per_level=3)
        log(f"[{time.time() - t0:6.1f}s]   bracket check: {chk['checked']} cells, {len(chk['violations'])} violations")
        beyond = 0.0
        for g in st["grid"]:
            r = cr.probs(g["w"], g["w"] + m)
            beyond = max(beyond, r["mass_beyond_Dmax"])
            for lv in LEVELS:
                g["size"][lv]["bb_ordered"] = r["reject"][lv]
        dense = []
        for w in W_DENSE:
            r = cr.probs(w, w + m)
            beyond = max(beyond, r["mass_beyond_Dmax"])
            dense.append({"w": w, "size": r["reject"]})
        interior = []
        for e in INTERIOR_E:
            for w in INTERIOR_W:
                pl = w + m + e
                if w + pl > 1:
                    continue
                r = cr.probs(w, pl)
                beyond = max(beyond, r["mass_beyond_Dmax"])
                interior.append({"p_win": w, "p_loss": pl, "excess": e, "w_star": (w + pl - m) / 2, "size": r["reject"]})
        bb: dict[str, Any] = {"info": cr.info, "bracket_check": chk, "mass_beyond_Dmax_max": beyond, "dense_max": {},
                              "interior_max": {}, "dense": dense, "interior": interior}
        for lv, a in LEVELS.items():
            i = int(np.argmax([d["size"][lv] for d in dense]))
            bb["dense_max"][lv] = {"size": dense[i]["size"][lv], "w": dense[i]["w"], "ratio": dense[i]["size"][lv] / a}
            j = int(np.argmax([d["size"][lv] for d in interior]))
            bb["interior_max"][lv] = {"size": interior[j]["size"][lv], "p_win": interior[j]["p_win"],
                                      "p_loss": interior[j]["p_loss"], "ratio": interior[j]["size"][lv] / a}
            k = int(np.argmax([g["size"][lv]["bb_ordered"] for g in st["grid"]]))
            st["max"][lv]["bb_ordered"] = {"size": st["grid"][k]["size"][lv]["bb_ordered"], "w": st["grid"][k]["w"],
                                           "ratio": st["grid"][k]["size"][lv]["bb_ordered"] / a}
        st["bb"] = bb
        log(f"[{time.time() - t0:6.1f}s] n={n}: max size / alpha over the dense grid "
            f"{max(v['ratio'] for v in bb['dense_max'].values()):.4f}, interior {max(v['ratio'] for v in bb['interior_max'].values()):.4f}")
    results["runtime_s"] += time.time() - t0
    return results


def reading_r8(results: dict[str, Any]) -> list[str]:
    """Reading lines computed from the numbers."""
    sts = results["settings"]
    fl, top = "0.05/11", "0.05/1"
    worst = max(((st, lv, st["bb"]["dense_max"][lv]) for st in sts for lv in HOLM_LEVELS), key=lambda t: t[2]["ratio"])
    worst_ref = max(st["bb"]["dense_max"]["0.10"]["ratio"] for st in sts)
    worst_in = max(((st, lv, st["bb"]["interior_max"][lv]) for st in sts for lv in LEVELS), key=lambda t: t[2]["ratio"])
    n_cells = len(sts) * len(results["w_dense"]) * len(HOLM_LEVELS)
    per_n = {st["n"]: max(st["bb"]["dense_max"][lv]["ratio"] for lv in HOLM_LEVELS) for st in sts}
    n_over = sum(d["size"][lv] > LEVELS[lv] * (1 + 1e-9) for st in sts for d in st["bb"]["dense"] + st["bb"]["interior"]
                 for lv in LEVELS)
    old_u = {st["n"]: max(st["max"][lv]["registered_u"]["ratio"] for lv in HOLM_LEVELS) for st in sts}
    old_r = {st["n"]: max(st["max"][lv]["registered"]["ratio"] for lv in HOLM_LEVELS) for st in sts}

    def rel(st, lv, rule):              # bb_ordered size relative to `rule` at the largest-w cell of W_GRID
        g = st["grid"][-1]["size"][lv]
        return g["bb_ordered"] / max(g[rule], 1e-300)
    pw = "; ".join(f"n = {st['n']}: {rel(st, fl, 'registered_u'):.3f} / {rel(st, top, 'registered_u'):.3f}" for st in sts)
    viol = sum(len(st["bb"]["bracket_check"]["violations"]) for st in sts)
    nchk = sum(st["bb"]["bracket_check"]["checked"] for st in sts)
    gray = sum(st["bb"]["info"]["n_gray"] for st in sts)
    conv = all(st["bb"]["info"]["gray_all_converged"] for st in sts)
    small = {st["n"]: min(st["bb"]["dense_max"][lv]["ratio"] for lv in HOLM_LEVELS) for st in sts}
    lines = [
        f"1. **Size of the A6.21 rule.** Over the {len(sts)} × {len(results['w_dense'])} × {len(HOLM_LEVELS)} = {n_cells:,} "
        f"(n, m, w, Holm level) boundary cells the deployed rule's exact size is at most {worst[2]['ratio']:.4f} α "
        f"(n = {worst[0]['n']}, w = {worst[2]['w']}, α = {_lvl(worst[1])}); per n the maximum over the Holm levels is "
        + ", ".join(f"{per_n[n]:.4f} α at n = {n}" for n in sorted(per_n, key=lambda k: -per_n[k]))
        + f". At the 0.10 reference level the maximum is {worst_ref:.4f} α. "
        + (f"**{n_over} cell(s) exceed their level.**" if n_over else "No cell exceeds its level, including 0.10."),
        f"2. **Interior of the null.** At the {len(sts[0]['bb']['interior'])} interior points per setting "
        f"(P(loss) − P(win) = m + e, e ∈ {{{', '.join(str(e) for e in results['interior_e'])}}}) the largest size is "
        f"{worst_in[2]['ratio']:.4f} α (n = {worst_in[0]['n']}, P(win) = {worst_in[2]['p_win']}, P(loss) = "
        f"{worst_in[2]['p_loss']:.3f}, α = {_lvl(worst_in[1])}), well below the boundary maxima. The proof covers these "
        "points through its least-favourable step; the numbers are a check.",
        "3. **Against the superseded rules** (maximum over the 20-point grid and the Holm levels): A6.14 (asymptotic "
        "score p with the switch) " + ", ".join(f"{old_r[n]:.3f} α" for n in sorted(old_r)) + "; A6.17 (grid-maximised p_U "
        "with the switch) " + ", ".join(f"{old_u[n]:.3f} α" for n in sorted(old_u)) + "; A6.21 "
        + ", ".join(f"{max(st['max'][lv]['bb_ordered']['ratio'] for lv in HOLM_LEVELS):.3f} α" for st in sorted(sts, key=lambda s: s['n']))
        + f" (n = {', '.join(str(n) for n in sorted(old_r))}). The A6.21 rule's rejection probability at the "
        f"largest-discordance cell (w = {sts[0]['grid'][-1]['w']}) relative to A6.17's, at the floor / at 0.05: {pw}.",
        f"4. **Conservativeness.** The smallest of the per-level maxima is {min(small.values()):.3f} α; the rule is most "
        "conservative near w = 0, where the loss-count ordering's discreteness and γ = 10⁻⁴ (2.2 % of the floor) leave a gap.",
        f"5. **Exactness of the computation.** Every rejection region is the deployed rule's: {gray} gray cells were decided by "
        f"`metrics.ni_certified_p` itself (all converged: {conv}); {nchk} cells just inside / outside the two "
        f"brackets were re-decided by `ni_certified_p` with {viol} disagreement(s); the mass beyond the evaluated "
        f"discordant counts is ≤ {max(st['bb']['mass_beyond_Dmax_max'] for st in sts):.1e}.",
    ]
    return lines


def write_md_r8(path: Path, results: dict[str, Any], cmd: str) -> None:
    L: list[str] = []
    L.append("# Exact size of the A6.21 NI rule (bb_ordered), with the superseded A6.14 / A6.17 rules\n")
    L.append(f"`scripts/ni_exact_size.py` (review round 8), {results['date']}; raw numbers in `ni_exact_size_r8.json`. "
             "Null: per task P(win) = p_W, P(loss) = p_L, P(tie) = 1 − p_W − p_L with p_L − p_W ≥ m. Boundary points "
             "p_L = p_W + m are indexed by w = p_W. Size = Σ_(W+L≤n) Multinomial(n; p_W, p_L, p_T)(W, L) · 1[rule rejects "
             "at (W, L)], exact summation (no Monte Carlo). Levels: the eleven Holm levels α_k = 0.05/(12 − k) and 0.10.\n")
    L.append("**A6.21 rule (`bb_ordered`, `metrics.ni_certified_p`).** Ordering statistic s~ = monotone hull of the log "
             "nominal p of the A6.14 switching rule over the whole outcome space; p = min{1, γ + U}, U a certified upper "
             "bound (branch-and-bound with Hölder bounds) on sup_w P_w[s~ ≤ s~_obs + 10⁻⁹] over the Berger–Boos set "
             f"(Clopper–Pearson 1 − γ for the discordance, γ = {results['gamma']:g}); stopping rule U − L ≤ "
             f"max({results['abs_tol']:g}, {results['rel_tol']:g} L). Reject iff p ≤ α. Its rejection region is computed "
             "exactly: grid brackets decide all cells but the gray ones, which `ni_certified_p` decides (module docstring). "
             "**Superseded:** *registered* (A6.14: Tango p_T with the exact loss-count switch at W = 0 or W + L < 5) and "
             "*registered_u* (A6.17: grid-maximised p_U on the score branch with the same switch).\n")
    L.append("## 1. Maximum exact size per (n, m, level)\n")
    L.append(f"A6.21: over the dense boundary grid ({len(results['w_dense'])} values of w in [0, 0.4]) and over the interior "
             "null points (§3). A6.14 / A6.17: over the 20-point grid of round 7.\n")
    L.append("| n | m | α | A6.21 boundary max (×α) | at w | A6.21 interior max (×α) | A6.17 max (×α) | A6.14 max (×α) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for st in results["settings"]:
        for lv, a in LEVELS.items():
            d, i = st["bb"]["dense_max"][lv], st["bb"]["interior_max"][lv]
            mx = st["max"][lv]
            cell = f"{d['size']:.5f} ({d['ratio']:.4f}×)"
            if d["ratio"] > 1 + 1e-9:
                cell = f"**{cell}**"
            L.append(f"| {st['n']} | {st['m']} | {_lvl(lv)} = {a:.5f} | {cell} | {d['w']} | {i['size']:.5f} ({i['ratio']:.4f}×) | "
                     f"{mx['registered_u']['size']:.5f} ({mx['registered_u']['ratio']:.3f}×) | "
                     f"{mx['registered']['size']:.5f} ({mx['registered']['ratio']:.3f}×) |")
    L.append("")
    L.append("## 2. Exact size on the round-7 grid at α = 0.05/11, 0.05/6, 0.05 (A6.21 / A6.17 / A6.14)\n")
    for st in results["settings"]:
        L.append(f"### n = {st['n']}, m = {st['m']} — {st['label']}\n")
        L.append("| w | " + " | ".join(f"α = {_lvl(lv)}: A6.21 / A6.17 / A6.14" for lv in KEY_LEVELS) + " |")
        L.append("|---|" + "---|" * len(KEY_LEVELS))
        for g in st["grid"]:
            L.append(f"| {g['w']} | " + " | ".join(
                f"{g['size'][lv]['bb_ordered']:.5f} / {g['size'][lv]['registered_u']:.5f} / {g['size'][lv]['registered']:.5f}"
                for lv in KEY_LEVELS) + " |")
        L.append("")
    L.append("## 3. Interior null points (A6.21): exact size at α = 0.05/11 and 0.05\n")
    L.append("w* = (p_W + p_L − m)/2 is the boundary point with the same discordance, which dominates the interior point in the "
             "proof.\n")
    L.append("| n | m | P(win) | P(loss) | e | w* | size at 0.05/11 (×α) | size at 0.05 (×α) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for st in results["settings"]:
        for p in st["bb"]["interior"]:
            L.append(f"| {st['n']} | {st['m']} | {p['p_win']} | {p['p_loss']:.3f} | {p['excess']} | {p['w_star']:.4f} | "
                     f"{p['size']['0.05/11']:.6f} ({p['size']['0.05/11'] / LEVELS['0.05/11']:.3f}×) | "
                     f"{p['size']['0.05/1']:.5f} ({p['size']['0.05/1'] / LEVELS['0.05/1']:.3f}×) |")
    L.append("")
    L.append("## 4. Reading\n")
    L.extend(results["reading"])
    L.append("")
    L.append("## 5. Checks\n")
    for st in results["settings"]:
        bi, ck = st["bb"]["info"], st["bb"]["bracket_check"]
        L.append(f"- n = {st['n']}, m = {st['m']}: {bi['K']:,} distinct values of the ordering statistic; {bi['n_nodes']:,} "
                 f"nuisance nodes ({bi['n_grid']:,} on the arclength grid, step parameter {bi['eps_grid']}); "
                 f"{bi['n_gray']} gray cells (per level {', '.join(map(str, bi['gray_per_level']))}) decided by "
                 f"`ni_certified_p` (max {bi['gray_max_evals']:,} evaluations, all converged: {bi['gray_all_converged']}); "
                 f"bracket re-check on {ck['checked']} cells ({ck['cells']} distinct): {len(ck['violations'])} violation(s); "
                 f"mass beyond D = {bi['Dmax']}: ≤ {st['bb']['mass_beyond_Dmax_max']:.1e}; round-7 checks as in "
                 f"`results/reviews/2026-10-01d/ni_exact_size.md` (Tango interval agreement: "
                 f"{len(st['checks']['tango_vs_interval']['disagreements'])} disagreements).")
    L.append("")
    L.append("## 6. Command and runtime\n")
    L.append(f"```\n{cmd}\n```\n")
    L.append(f"Runtime {results['runtime_s'] / 60:.1f} min ({results['runtime_s']:.0f} s) on {results['cpu']} CPU threads "
             f"(numpy {np.__version__}, Python {sys.version.split()[0]}); no scipy.\n")
    path.write_text("\n".join(L))


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default results/reviews/2026-10-02; --legacy: results/reviews/2026-10-01d)")
    ap.add_argument("--prune", type=float, default=1e-15, help="drop (W, L) pairs whose pmf is below this")
    ap.add_argument("--exact-below", type=int, default=5, help="exact fallback when W + L < this (metrics.ni_pvalue exact_below)")
    ap.add_argument("--eps-grid", type=float, default=0.005, help="A6.21 node-grid step parameter (only the gray-cell count depends on it)")
    ap.add_argument("--render-only", action="store_true", help="rewrite the .md from the existing .json")
    ap.add_argument("--legacy", action="store_true", help="the round-7 tables only (ni_exact_size.{json,md})")
    args = ap.parse_args()
    out = args.out or ROOT / ("results/reviews/2026-10-01d" if args.legacy else "results/reviews/2026-10-02")
    stem = "ni_exact_size" if args.legacy else "ni_exact_size_r8"
    out_json, out_md = out / f"{stem}.json", out / f"{stem}.md"
    read, write = (reading, write_md) if args.legacy else (reading_r8, write_md_r8)
    if args.render_only:
        results = json.loads(out_json.read_text())
        results["reading"] = read(results)
        out_json.write_text(json.dumps(results, indent=1))
        write(out_md, results, results["command"])
        return
    cmd = "python " + " ".join(sys.argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    results = run(args.prune, args.exact_below, log=log) if args.legacy else \
        run_r8(args.prune, args.exact_below, args.eps_grid, log=log)
    results.update({"date": time.strftime("%Y-%m-%d"), "command": cmd, "cpu": os.cpu_count(),
                    "numpy": np.__version__, "python": sys.version.split()[0]})
    results["reading"] = read(results)
    out.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=1))
    write(out_md, results, cmd)
    print(f"done in {results['runtime_s']:.0f}s -> {out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
