"""Numerical checks of the delegation theory (paper Sec. 3.4 and App. A) on synthetic decision populations.

    python scripts/check_propositions.py [--seed 0] [--n-pop 60] [--mc 40000] [--json OUT.json]

Offline and fast (numpy only, a few seconds). Every check draws random populations -- confidence distributions,
calibration curves g(c), LLM accuracy curves q_L(c), costs, stakes, gate rates -- and compares the closed forms of
the paper with (a) the exact expected utility of the discretised population and (b) a Monte-Carlo simulation of
individual decisions with common random numbers. Exit code 1 if any check fails.

Checks (keys of the report):

  prop1        Prop. 1 (A1-A3): tau* = q_L - dC/V (clipped) minimises expected loss over every threshold, over
               random deterministic and randomised rules that see only c, and in the Monte-Carlo simulation no
               threshold beats tau* by more than 4.5 standard errors.
  general      Eq. (tau-general): the pointwise rule "accept iff A(c) = g(c) - q_L(c) + dC/V >= 0" is optimal among
               rules that see only c, for random non-monotone g and q_L; the loss gap of any threshold rule equals
               V * sum over disagreements of |A(c)| (exact identity); the best threshold attains the pointwise loss
               iff the accept set is an upper set of the support (in particular whenever A is non-decreasing).
  statics      Comparative statics. (i) Monotone: lowering q_L pointwise, raising g pointwise, raising dC or lowering
               V never raises the optimal threshold (smallest accepted c and lower edge of the accepted tail).
               (ii) Correlated errors with the marginal q_L held fixed move the threshold in NEITHER fixed direction:
               with g(c) = c and A(c) non-decreasing, tau_general < tau_prop1 iff q_L(tau_prop1) < mean q_L; for
               linear q_L(c) = alpha + s c, tau_general - tau_prop1 = s (tau_prop1 - E[c]) / (1 - s). Both signs
               occur in the random population (the claim "q_L rising with c lowers the threshold" is false in
               general). (iii) Uniform over-confidence g(c) = c - xi under A2 raises the threshold by xi.
  gain         Eq. (gain): the expected gain of the optimal cascade over LLM-only is
               E[(dC - (q_L(c) - g(c)) V)^+] - kappa (exact and Monte Carlo).
  guard        GUARD with a damage term: a rejection costs dC, turns a wrong proposal right w.p. eta and a right one
               wrong w.p. delta; reject iff p < tau_G = (eta - dC/V) / (eta + delta). delta = 0 gives
               1 - dC/(eta V); the guard rejects anything iff eta > dC/V; using the delta = 0 formula when delta > 0
               has positive regret; the guard is the trigger with need = wrong, B = eta V, H = delta V, C = dC.
  trigger      TRIGGER: fire iff p >= (C + H)/(B + H) (exact and Monte Carlo).
  prop2        Prop. 2: simulated gates with given (n, pi, TPR, FPR, B, H, C1, C0, kappa) against a random gate at
               the matched rate, where firing costs C1 at a needed and C0 at an unneeded opportunity: mean
               Delta U = n [(B + H - (C1 - C0)) pi (1 - pi) J - kappa], exactly (closed form against the definition)
               and in Monte Carlo; firing costs cancel only when C1 = C0, and the constant-cost formula
               n [(B + H) pi (1 - pi) J - kappa] is detectably wrong when C1 != C0; the realised-rate correction
               -n dnu [pi (B - C1) - (1 - pi) (H + C0)]; at the matched rate the gate's accuracy exceeds the random
               gate's by 2 pi (1 - pi) J (exact and Monte Carlo); the guard version with damage (B = eta W,
               H = delta W, J = 1 - FAR - FRR) and a retry cost that depends on whether the proposal was wrong.
  auroc        max_tau J <= 2 AUROC - 1 for concave (binormal) ROC curves; AUROC = (1 + J)/2 for a binary score
               (``metrics.auroc``, ties 1/2); a non-concave ROC on which the bound fails (the condition is needed).
  code         The repository's estimators against the closed forms: ``threshold_theory.scalar_threshold_projection``
               (deployed C1 estimator) and ``plug_in_thresholds`` on exact synthetic logs, ``loss_matrix`` against
               the Prop. 1 loss, ``gate_analysis``'s matched-rate VoI against Eq. (voi) (an exact finite-sample
               identity), ``meta_gold.predicted_gain`` and ``metrics.voi_identity`` against Eq. (voi) and the
               rate correction.

Synthetic populations only: no number produced here is an experimental result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:          # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

TOL = 1e-10          # exact (discretised-population) comparisons
Z_MC = 4.5           # Monte-Carlo comparisons: standard errors (maxima over ~10^3 correlated comparisons)


# ====================================================================== populations
def beta_support(rng: np.random.Generator, K: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """A discretised confidence distribution on (0, 1): K equally spaced support points with Beta(a, b) weights."""
    a, b = rng.uniform(0.6, 6.0), rng.uniform(0.6, 4.0)
    c = (np.arange(K) + 0.5) / K
    w = c ** (a - 1) * (1 - c) ** (b - 1)
    return c, w / w.sum()


def wiggle(rng: np.random.Generator, c: np.ndarray, amp: float) -> np.ndarray:
    """A smooth random non-monotone perturbation of amplitude <= amp."""
    out = np.zeros_like(c)
    for _ in range(3):
        out += rng.uniform(-1, 1) * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * c + rng.uniform(0, 2 * np.pi))
    return amp * out / 3.0


def costs(rng: np.random.Generator) -> tuple[float, float]:
    """(dC, V): call cost and stakes with dC / V spread over [0, 0.4]."""
    V = float(rng.choice([0.1, 1.0, 10.0]) * rng.uniform(0.5, 2.0))
    return float(V * rng.uniform(0.0, 0.4)), V


# ====================================================================== exact losses (discretised population)
def loss_rule(accept: np.ndarray, w: np.ndarray, g: np.ndarray, q: np.ndarray, dC: float, V: float) -> float:
    """Expected loss per decision of a (possibly randomised) rule accept[k] = P(accept | c_k) (kappa omitted: paid in
    both branches)."""
    acc = accept.astype(float)
    return float(np.sum(w * (acc * (1 - g) * V + (1 - acc) * ((1 - q) * V + dC))))


def loss_threshold(tau: float, c: np.ndarray, w: np.ndarray, g: np.ndarray, q: np.ndarray, dC: float,
                   V: float) -> float:
    return loss_rule(c >= tau, w, g, q, dC, V)


def tau_grid(c: np.ndarray, extra: tuple[float, ...] = ()) -> np.ndarray:
    """Every distinct threshold rule on the support: each support point, below it and above it."""
    return np.unique(np.concatenate([c, [-1.0, 2.0], np.asarray(extra, float)]))


def smallest_accepted(c: np.ndarray, acc: np.ndarray) -> float:
    """The crossing: the smallest accepted c (inf when nothing is accepted)."""
    return float(c[acc].min()) if acc.any() else math.inf


def tail_edge(c: np.ndarray, acc: np.ndarray) -> float:
    """The lower edge of the accepted upper tail: the smallest c from which every higher support point is accepted
    (inf when the top point is rejected). ``c`` sorted ascending."""
    if not acc.any() or not acc[-1]:
        return math.inf
    j = len(acc) - 1
    while j > 0 and acc[j - 1]:
        j -= 1
    return float(c[j])


def is_upper_set(acc: np.ndarray) -> bool:
    """acc (on ascending support) is of the form [False ... False True ... True]."""
    return bool(not acc.any() or acc[int(np.argmax(acc)):].all())


# ====================================================================== Monte Carlo
def mc_draw(rng: np.random.Generator, w: np.ndarray, g: np.ndarray, q: np.ndarray, n: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n decisions: support index, System-1 correct, LLM correct (independent given c)."""
    k = rng.choice(len(w), size=n, p=w)
    return k, rng.random(n) < g[k], rng.random(n) < q[k]


def mc_losses(taus: np.ndarray, c: np.ndarray, k: np.ndarray, s1: np.ndarray, s2: np.ndarray, dC: float, V: float
              ) -> np.ndarray:
    """(n_tau x n) per-decision realised losses with common random numbers."""
    acc = c[k][None, :] >= taus[:, None]
    return np.where(acc, V * (~s1)[None, :], V * (~s2)[None, :] + dC)


def beats_z(L: np.ndarray, ref: int) -> float:
    """max over rows i of (mean(L_ref) - mean(L_i)) / sqrt(SE_ref^2 + SE_i^2): how far, in standard errors, any rule
    beats row ``ref`` in the simulation. The unpaired standard error is conservative under common random numbers, and
    unlike the paired one it does not degenerate when two rules differ on a handful of decisions whose realised
    loss differences happen to be identical."""
    n = L.shape[1]
    se = L.std(axis=1, ddof=1) / math.sqrt(n)
    d = L[ref].mean() - L.mean(axis=1)
    den = np.sqrt(se[ref] ** 2 + se ** 2)
    z = np.where(den > 0, d / np.where(den > 0, den, 1.0), np.where(d > 1e-12, np.inf, 0.0))
    return float(z.max())


def mc_vs_exact_z(L: np.ndarray, exact: np.ndarray) -> float:
    """max over rows of |mean(L_i) - exact_i| / SE_i: the simulation reproduces the closed-form expected loss."""
    n = L.shape[1]
    vmax = float(np.abs(L).max()) + 1e-12
    # floor: a rare branch of size <= vmax and mean |exact| has SE >= sqrt(vmax |exact| / n) even when it never occurs
    se = np.maximum(L.std(axis=1, ddof=1) / math.sqrt(n), np.sqrt(vmax * np.abs(exact) / n) + vmax / n)
    return float(np.max(np.abs(L.mean(axis=1) - exact) / se))


def abs_z(x: np.ndarray, target: float) -> float:
    """|mean(x) - target| / SE(mean), the SE floored as in :func:`mc_vs_exact_z`: a sample in which a rare branch
    never occurs has (near) zero spread but cannot resolve that branch's contribution."""
    n = len(x)
    gap = abs(float(x.mean()) - target)
    vmax = max(float(np.abs(x).max()), abs(target)) + 1e-12
    se = max(float(x.std(ddof=1)) / math.sqrt(n), math.sqrt(vmax * gap / n) + vmax / n)
    return gap / se


# ====================================================================== Prop. 1
def check_prop1(rng: np.random.Generator, n_pop: int, n_mc: int) -> dict[str, Any]:
    worst_gap, worst_rand, worst_z, worst_fit = 0.0, 0.0, 0.0, 0.0
    for _ in range(n_pop):
        c, w = beta_support(rng)
        q = float(rng.uniform(0.5, 1.0))
        dC, V = costs(rng)
        g, qv = c.copy(), np.full_like(c, q)                       # A1, A2
        tau = min(max(q - dC / V, 0.0), 1.0)
        L_star = loss_threshold(tau, c, w, g, qv, dC, V)
        taus = tau_grid(c, (tau,))
        L_all = np.array([loss_threshold(t, c, w, g, qv, dC, V) for t in taus])
        worst_gap = max(worst_gap, L_star - float(L_all.min()))
        # random rules that see only c: deterministic subsets and randomised acceptance probabilities
        for _ in range(20):
            sub = rng.random(len(c)) < rng.uniform(0.1, 0.9)
            rnd = rng.random(len(c))
            worst_rand = max(worst_rand, L_star - loss_rule(sub, w, g, qv, dC, V),
                             L_star - loss_rule(rnd, w, g, qv, dC, V))
        # Monte Carlo with common random numbers
        k, s1, s2 = mc_draw(rng, w, g, qv, n_mc)
        grid = np.unique(np.concatenate([np.linspace(0, 1, 41), [tau]]))
        L = mc_losses(grid, c, k, s1, s2, dC, V)
        worst_z = max(worst_z, beats_z(L, int(np.where(grid == tau)[0][0])))
        exact = np.array([loss_threshold(t, c, w, g, qv, dC, V) for t in grid])
        worst_fit = max(worst_fit, mc_vs_exact_z(L, exact))
    ok = worst_gap <= TOL and worst_rand <= TOL and worst_z <= Z_MC and worst_fit <= Z_MC
    return {"ok": ok, "n_pop": n_pop, "exact_gap_vs_best_threshold": worst_gap,
            "exact_gap_vs_random_rules": worst_rand, "mc_max_z_any_threshold_beats_tau_star": worst_z,
            "mc_vs_exact_loss_max_abs_z": worst_fit}


# ====================================================================== Eq. tau-general
def random_curves(rng: np.random.Generator, c: np.ndarray, monotone: bool) -> tuple[np.ndarray, np.ndarray]:
    """(g, q_L) on the support: monotone=True gives a non-decreasing A(c) = g - q_L (+ const)."""
    if monotone:
        s = rng.uniform(0.0, 0.9)
        g = np.clip(c + rng.uniform(-0.1, 0.05), 0, 1)
        q = np.clip(rng.uniform(0.6, 1.0) + s * (c - c.mean()) * rng.uniform(0, 1), 0, 1)
        # enforce a non-decreasing A(c) exactly (clipping can break it at the ends)
        a = np.maximum.accumulate(g - q)
        g = q + a
        keep = (g >= 0) & (g <= 1)
        if not keep.all():
            g = np.clip(g, 0, 1)
            q = np.clip(g - a, 0, 1)
        return g, q
    g = np.clip(c + rng.uniform(-0.15, 0.05) + wiggle(rng, c, 0.3), 0, 1)
    q = np.clip(rng.uniform(0.5, 1.0) + rng.uniform(-0.3, 0.3) * (c - 0.5) + wiggle(rng, c, 0.3), 0, 1)
    return g, q


def check_general(rng: np.random.Generator, n_pop: int, n_mc: int) -> dict[str, Any]:
    worst_opt, worst_id, worst_z, iff_fail, n_upper, n_not_upper, mono_fail = 0.0, 0.0, 0.0, 0, 0, 0, 0
    for i in range(n_pop):
        c, w = beta_support(rng)
        mono = i % 3 == 0
        g, q = random_curves(rng, c, mono)
        dC, V = costs(rng)
        if i % 3 == 2:                        # an advantage oscillating around 0: accept sets with islands
            dC *= 0.25
            q = np.clip(rng.uniform(0.5, 0.8) + wiggle(rng, c, 0.1), 0, 1)
            osc = rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * rng.uniform(1.5, 4.0) * c + rng.uniform(0, 2 * np.pi))
            g = np.clip(q + osc - dC / V + rng.uniform(-0.02, 0.02), 0, 1)
        a = g - q + dC / V
        pw = a >= 0
        L_pw = loss_rule(pw, w, g, q, dC, V)
        taus = tau_grid(c)
        L_thr = np.array([loss_threshold(t, c, w, g, q, dC, V) for t in taus])
        for t, Lt in zip(taus, L_thr):           # exact identity: gap = V sum_{disagree} w |a|
            dis = (c >= t) != pw
            worst_id = max(worst_id, abs((Lt - L_pw) - V * float(np.sum(w[dis] * np.abs(a[dis])))))
        for _ in range(20):
            worst_opt = max(worst_opt, L_pw - loss_rule(rng.random(len(c)) < 0.5, w, g, q, dC, V),
                            L_pw - loss_rule(rng.random(len(c)), w, g, q, dC, V))
        worst_opt = max(worst_opt, L_pw - float(L_thr.min()))
        # a threshold attains the pointwise loss iff the accept set is an upper set (no exact zeros here)
        upper = is_upper_set(pw)
        n_upper += upper
        n_not_upper += not upper
        if upper != (float(L_thr.min()) - L_pw <= 1e-12):
            iff_fail += 1
        if mono and not upper:
            mono_fail += 1
        k, s1, s2 = mc_draw(rng, w, g, q, n_mc)
        Lpw_mc = np.where(pw[k], V * ~s1, V * ~s2 + dC)
        grid = np.linspace(0, 1, 41)
        L = np.vstack([Lpw_mc[None, :], mc_losses(grid, c, k, s1, s2, dC, V)])
        worst_z = max(worst_z, beats_z(L, 0))
    ok = worst_opt <= TOL and worst_id <= 1e-9 and iff_fail == 0 and mono_fail == 0 and worst_z <= Z_MC
    return {"ok": ok, "n_pop": n_pop, "exact_gap_pointwise_vs_any_rule": worst_opt,
            "max_identity_error": worst_id, "threshold_attains_pointwise_iff_upper_set_failures": iff_fail,
            "monotone_a_but_not_threshold": mono_fail, "n_upper_set": n_upper, "n_not_upper_set": n_not_upper,
            "mc_max_z_any_threshold_beats_pointwise": worst_z}


# ====================================================================== comparative statics
def crossing_continuous(alpha: float, s: float, k: float) -> float:
    """g(c) = c, q_L(c) = alpha + s c, 0 <= s < 1: A(c) = (1 - s) c - alpha + k >= 0 iff c >= (alpha - k)/(1 - s)."""
    return (alpha - k) / (1 - s)


def check_statics(rng: np.random.Generator, n_pop: int) -> dict[str, Any]:
    mono_viol = 0
    for _ in range(n_pop):
        c, w = beta_support(rng)
        g, q = random_curves(rng, c, monotone=False)
        dC, V = costs(rng)
        base = g - q + dC / V >= 0
        u = np.abs(wiggle(rng, c, 0.2)) + rng.uniform(0, 0.05)
        variants = {
            "q_L lower": g - np.clip(q - u, 0, 1) + dC / V >= 0,
            "g higher": np.clip(g + u, 0, 1) - q + dC / V >= 0,
            "dC higher": g - q + (dC * 1.5 + 0.01 * V) / V >= 0,
            "V lower": g - q + dC / (V * 0.5) >= 0,
        }
        for acc in variants.values():
            if not (acc >= base).all():           # nested accept sets
                mono_viol += 1
            if smallest_accepted(c, acc) > smallest_accepted(c, base) + 1e-12:
                mono_viol += 1
            if tail_edge(c, acc) > tail_edge(c, base) + 1e-12:
                mono_viol += 1
    # (ii) correlated errors with the same marginal q_L
    up = down = sign_fail = lin_fail = exact_fail = 0
    for _ in range(4 * n_pop):
        c, w = beta_support(rng, 800)
        Ec = float(np.sum(w * c))
        s = float(rng.uniform(0.05, 0.6))
        dC, V = costs(rng)
        k = dC / V
        alpha = float(rng.uniform(0.2, 1.0 - s))
        q = alpha + s * c
        qbar = float(np.sum(w * q))
        tau1 = qbar - k
        tau_g = crossing_continuous(alpha, s, k)
        if not (0.02 < tau1 < 0.98 and 0.02 < tau_g < 0.98):
            continue
        if abs(tau_g - tau1 - s * (tau1 - Ec) / (1 - s)) > 1e-9:
            lin_fail += 1
        if np.sign(tau_g - tau1) != np.sign((alpha + s * tau1) - qbar):
            sign_fail += 1
        # the crossing is the exact optimum of the discretised population (up to the support spacing)
        L = np.array([loss_threshold(t, c, w, c, q, dC, V) for t in tau_grid(c)])
        best = tau_grid(c)[int(np.argmin(L))]
        if abs(best - tau_g) > 1.0 / len(c) + 1e-9:
            exact_fail += 1
        up += tau_g > tau1
        down += tau_g < tau1
    # (iii) over-confidence shift
    shift_fail = 0
    for _ in range(n_pop):
        c, w = beta_support(rng)
        qv, delta = float(rng.uniform(0.5, 0.9)), float(rng.uniform(0.0, 0.1))
        dC, V = costs(rng)
        acc = (c - delta) - qv + dC / V >= 0
        target = qv - dC / V + delta
        if acc.any() and not acc.all() and abs(smallest_accepted(c, acc) - float(c[c >= target - 1e-12].min())) > 1e-12:
            shift_fail += 1
    ok = mono_viol == 0 and sign_fail == 0 and lin_fail == 0 and exact_fail == 0 and shift_fail == 0 \
        and up > 0 and down > 0
    return {"ok": ok, "monotone_statics_violations": mono_viol,
            "correlated_errors": {"threshold_above_prop1": up, "threshold_below_prop1": down,
                                  "sign_rule_failures": sign_fail, "linear_formula_failures": lin_fail,
                                  "crossing_not_optimal": exact_fail},
            "overconfidence_shift_failures": shift_fail,
            "note": "both directions occur: q_L(c) rising with c does not by itself lower the optimal threshold"}


# ====================================================================== Eq. gain
def check_gain(rng: np.random.Generator, n_pop: int, n_mc: int) -> dict[str, Any]:
    worst_exact, worst_z = 0.0, 0.0
    for _ in range(n_pop):
        c, w = beta_support(rng)
        g, q = random_curves(rng, c, monotone=False)
        dC, V = costs(rng)
        kappa = float(rng.uniform(0, 0.2) * dC)
        pw = g - q + dC / V >= 0
        L_llm = loss_rule(np.zeros(len(c), bool), w, g, q, dC, V)
        L_cas = loss_rule(pw, w, g, q, dC, V) + kappa
        closed = float(np.sum(w * np.maximum(dC - (q - g) * V, 0.0))) - kappa
        worst_exact = max(worst_exact, abs((L_llm - L_cas) - closed))
        k, s1, s2 = mc_draw(rng, w, g, q, n_mc)
        d = (V * ~s2 + dC) - (np.where(pw[k], V * ~s1, V * ~s2 + dC) + kappa)
        worst_z = max(worst_z, abs_z(d, closed))
    return {"ok": worst_exact <= 1e-9 and worst_z <= Z_MC, "n_pop": n_pop, "max_exact_error": worst_exact,
            "mc_max_abs_z": worst_z}


# ====================================================================== guard with damage, trigger
def guard_loss(tau: float, p: np.ndarray, w: np.ndarray, dC: float, V: float, eta: float, delta: float) -> float:
    rej = p < tau
    acc_loss = (1 - p) * V
    rej_loss = dC + (1 - p) * (1 - eta) * V + p * delta * V
    return float(np.sum(w * np.where(rej, rej_loss, acc_loss)))


def tau_guard(dC: float, V: float, eta: float, delta: float) -> float:
    """tau_G = (eta - dC/V) / (eta + delta): reject iff p < tau_G."""
    return (eta - dC / V) / (eta + delta) if eta + delta > 0 else -math.inf


def tau_trigger(B: float, H: float, C: float) -> float:
    """tau_T = (C + H) / (B + H): fire iff p >= tau_T."""
    return (C + H) / (B + H) if B + H > 0 else math.inf


def check_guard(rng: np.random.Generator, n_pop: int, n_mc: int) -> dict[str, Any]:
    worst_gap, worst_z, useless_fail, special_fail, map_fail, regret_d0 = 0.0, 0.0, 0, 0, 0, 0.0
    for i in range(n_pop):
        p, w = beta_support(rng)
        dC, V = costs(rng)
        eta = float(rng.uniform(0.0, 1.0))
        delta = 0.0 if i % 4 == 0 else float(rng.uniform(0.0, 0.5))
        tg = tau_guard(dC, V, eta, delta)
        taus = tau_grid(p, (tg,))
        L = np.array([guard_loss(t, p, w, dC, V, eta, delta) for t in taus])
        L_star = guard_loss(tg, p, w, dC, V, eta, delta)
        worst_gap = max(worst_gap, L_star - float(L.min()))
        # the guard rejects anything (beats never rejecting) iff eta > dC / V
        # (on the support, away from the boundary eta = dC/V where it depends on the grid)
        if abs(eta - dC / V) > 2.0 / len(p) and bool((p < tg).any()) != (eta > dC / V):
            useless_fail += 1
        if delta == 0.0 and eta > 0 and abs(tg - (1 - dC / (eta * V))) > 1e-12:
            special_fail += 1
        # the guard is the trigger whose need is a wrong proposal
        if eta + delta > 0 and abs((1 - tau_trigger(eta * V, delta * V, dC)) - tg) > 1e-12:
            map_fail += 1
        # ignoring the damage term (the delta = 0 formula) is not optimal when delta > 0
        if delta > 0 and eta > 0:
            regret_d0 = max(regret_d0, guard_loss(1 - dC / (eta * V), p, w, dC, V, eta, delta) - L_star)
        # Monte Carlo: proposals right w.p. p; a rejection fixes / damages with eta / delta
        k = rng.choice(len(w), size=n_mc, p=w)
        right = rng.random(n_mc) < p[k]
        u_fix, u_dmg = rng.random(n_mc), rng.random(n_mc)
        grid = np.unique(np.concatenate([np.linspace(0, 1, 41), [min(max(tg, -1.0), 2.0)]]))
        rows = []
        for t in grid:
            rej = p[k] < t
            after = np.where(right, u_dmg >= delta, u_fix < eta)
            final_right = np.where(rej, after, right)
            rows.append(V * ~final_right + dC * rej)
        Lmc = np.array(rows)
        worst_z = max(worst_z, beats_z(Lmc, int(np.argmin(np.abs(grid - min(max(tg, -1.0), 2.0))))))
        worst_z = max(worst_z, mc_vs_exact_z(Lmc, np.array([guard_loss(t, p, w, dC, V, eta, delta) for t in grid])))
    ok = worst_gap <= TOL and worst_z <= Z_MC and useless_fail == 0 and special_fail == 0 and map_fail == 0 \
        and regret_d0 > 0
    return {"ok": ok, "n_pop": n_pop, "exact_gap_vs_best_threshold": worst_gap, "mc_max_z": worst_z,
            "useless_iff_eta_le_dC_over_V_failures": useless_fail, "delta0_special_case_failures": special_fail,
            "guard_as_trigger_failures": map_fail, "max_regret_of_ignoring_damage": regret_d0}


def check_trigger(rng: np.random.Generator, n_pop: int, n_mc: int) -> dict[str, Any]:
    worst_gap, worst_z = 0.0, 0.0
    for _ in range(n_pop):
        p, w = beta_support(rng)
        B, H = float(rng.uniform(0.05, 1.0)), float(rng.uniform(0.0, 0.5))
        C = float(rng.uniform(0.0, 0.3) * B)
        tt = tau_trigger(B, H, C)

        def value(t: float) -> float:
            fire = p >= t
            return float(np.sum(w * fire * (p * B - (1 - p) * H - C)))

        taus = tau_grid(p, (tt,))
        worst_gap = max(worst_gap, max(value(t) for t in taus) - value(tt))
        k = rng.choice(len(w), size=n_mc, p=w)
        need = rng.random(n_mc) < p[k]
        grid = np.unique(np.concatenate([np.linspace(0, 1, 41), [tt]]))
        U = np.array([(p[k] >= t) * (B * need - H * ~need - C) for t in grid])
        worst_z = max(worst_z, beats_z(-U, int(np.where(grid == tt)[0][0])),
                      mc_vs_exact_z(U, np.array([value(t) for t in grid])))
    return {"ok": worst_gap <= TOL and worst_z <= Z_MC, "n_pop": n_pop, "exact_gap": worst_gap, "mc_max_z": worst_z}


# ====================================================================== Prop. 2
def simulate_gate(rng: np.random.Generator, E: int, n: int, pi: float, tpr: float, fpr: float, B: float, H: float,
                  C: float, kappa: float, dnu: float = 0.0, C0: float | None = None) -> np.ndarray:
    """Per-episode U_gate - U_random over E episodes of n opportunities (binomial sufficient statistics: the
    number of needed opportunities, and the firings among needed / unneeded ones for each gate). A firing costs ``C``
    at a needed opportunity and ``C0`` at an unneeded one (default ``C0 = C``: a need-independent cost)."""
    C1, C0 = C, (C if C0 is None else C0)
    nu = pi * tpr + (1 - pi) * fpr
    nr = min(max(nu + dnu, 0.0), 1.0)
    n1 = rng.binomial(n, pi, size=E)
    n0 = n - n1
    f1, f0 = rng.binomial(n1, tpr), rng.binomial(n0, fpr)
    r1, r0 = rng.binomial(n1, nr), rng.binomial(n0, nr)
    u_gate = f1 * (B - C1) - f0 * (H + C0) - n * kappa
    u_rand = r1 * (B - C1) - r0 * (H + C0)
    return (u_gate - u_rand).astype(float)


def simulate_accuracy_gap(rng: np.random.Generator, E: int, n: int, pi: float, tpr: float, fpr: float
                          ) -> np.ndarray:
    """Per-episode (correct decisions of the gate - those of the matched-rate random gate) / n: a decision is
    correct if it fires at a needed or skips an unneeded opportunity."""
    nu = pi * tpr + (1 - pi) * fpr
    n1 = rng.binomial(n, pi, size=E)
    n0 = n - n1
    f1, f0 = rng.binomial(n1, tpr), rng.binomial(n0, fpr)
    r1, r0 = rng.binomial(n1, nu), rng.binomial(n0, nu)
    return ((f1 + n0 - f0) - (r1 + n0 - r0)).astype(float) / n


def voi_closed(n: float, pi: float, J: float, B: float, H: float, kappa: float, dC: float = 0.0) -> float:
    """Eq. (voi): n [(B + H - dC) pi (1 - pi) J - kappa], where dC = C1 - C0 is the extra firing cost at a needed
    opportunity (0 for a need-independent firing cost, the case the paper's Prop. 2 states)."""
    return n * ((B + H - dC) * pi * (1 - pi) * J - kappa)


def voi_exact(n: float, pi: float, tpr: float, fpr: float, B: float, H: float, C1: float, C0: float, kappa: float,
              dnu: float = 0.0) -> float:
    """E[U_gate - U_random] from the definition, without the algebra of the proof: per opportunity the gate fires
    w.p. TPR / FPR given need, the random gate w.p. nu + dnu whatever the need, and a firing yields B - C1 at a
    needed and -(H + C0) at an unneeded opportunity; the gate's query costs kappa."""
    nr = pi * tpr + (1 - pi) * fpr + dnu
    gate = pi * tpr * (B - C1) - (1 - pi) * fpr * (H + C0) - kappa
    rand = pi * nr * (B - C1) - (1 - pi) * nr * (H + C0)
    return n * (gate - rand)


def accuracy_gap(pi: float, tpr: float, fpr: float) -> float:
    """The gate's accuracy minus the matched-rate random gate's, from the definition (closed form: 2 pi (1 - pi) J)."""
    nu = pi * tpr + (1 - pi) * fpr
    return (pi * tpr + (1 - pi) * (1 - fpr)) - (pi * nu + (1 - pi) * (1 - nu))


# a fixed population with a large need-dependent cost (C1 = 0.4, C0 = 0): the constant-cost formula is off by
# n (C1 - C0) pi (1 - pi) J = 1.6, which the simulation must detect (power of the prop2 check)
POWER_POP = {"n": 20, "pi": 0.5, "tpr": 0.9, "fpr": 0.1, "B": 0.5, "H": 0.2, "C1": 0.4, "C0": 0.0, "kappa": 0.01}


def check_prop2(rng: np.random.Generator, n_pop: int, n_mc: int) -> dict[str, Any]:
    worst_z, worst_z_corr, worst_z_guard, worst_z_acc = 0.0, 0.0, 0.0, 0.0
    worst_exact, worst_exact_corr, worst_cancel, worst_acc = 0.0, 0.0, 0.0, 0.0
    bias_fail, n_need_dep = 0, 0
    for i in range(n_pop):
        n = int(rng.integers(1, 30))
        pi, tpr, fpr = float(rng.uniform(0.02, 0.98)), float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
        B, H = float(rng.uniform(0, 1)), float(rng.uniform(0, 0.5))
        C1, kappa = float(rng.uniform(0, 0.2)), float(rng.uniform(0, 0.02))
        # every other population has a firing cost that depends on need; the others a constant one
        C0 = C1 if i % 2 == 0 else float(rng.uniform(0, 0.2))
        n_need_dep += C0 != C1
        J = tpr - fpr
        cl = voi_closed(n, pi, J, B, H, kappa, C1 - C0)
        worst_exact = max(worst_exact, abs(voi_exact(n, pi, tpr, fpr, B, H, C1, C0, kappa) - cl))
        # a constant cost cancels at the matched rate: the value does not depend on it
        worst_cancel = max(worst_cancel, abs(voi_exact(n, pi, tpr, fpr, B, H, C1, C1, kappa)
                                             - voi_closed(n, pi, J, B, H, kappa)))
        # a need-dependent one does not: the constant-cost formula is off by exactly n (C1 - C0) pi (1 - pi) J
        bias = voi_exact(n, pi, tpr, fpr, B, H, C1, C0, kappa) - voi_closed(n, pi, J, B, H, kappa)
        if abs(bias + n * (C1 - C0) * pi * (1 - pi) * J) > 1e-12:
            bias_fail += 1
        d = simulate_gate(rng, n_mc, n, pi, tpr, fpr, B, H, C1, kappa, C0=C0)
        worst_z = max(worst_z, abs_z(d, cl))
        # realised-rate correction when the random arm fires at nu + dnu
        nu = pi * tpr + (1 - pi) * fpr
        dnu = float(rng.uniform(-0.1, 0.1))
        dnu = min(max(nu + dnu, 0.0), 1.0) - nu
        cl2 = cl - n * dnu * (pi * (B - C1) - (1 - pi) * (H + C0))
        worst_exact_corr = max(worst_exact_corr, abs(voi_exact(n, pi, tpr, fpr, B, H, C1, C0, kappa, dnu) - cl2))
        d2 = simulate_gate(rng, n_mc, n, pi, tpr, fpr, B, H, C1, kappa, dnu, C0=C0)
        worst_z_corr = max(worst_z_corr, abs_z(d2, cl2))
        # accuracy at the matched rate: the gate beats the random gate by 2 pi (1 - pi) J
        worst_acc = max(worst_acc, abs(accuracy_gap(pi, tpr, fpr) - 2 * pi * (1 - pi) * J))
        worst_z_acc = max(worst_z_acc, abs_z(simulate_accuracy_gap(rng, n_mc, n, pi, tpr, fpr), 2 * pi * (1 - pi) * J))
        # guard version: need = wrong proposal (prob pi); reject = fire; a rejection fixes a wrong proposal w.p. eta
        # and damages a right one w.p. delta, at cost Cw after a wrong and Cr after a right proposal; utility W per
        # right decision
        Wv, eta, delta = float(rng.uniform(0.5, 2)), float(rng.uniform(0, 1)), float(rng.uniform(0, 0.4))
        far, frr = float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
        Cw = float(rng.uniform(0, 0.2))
        Cr = Cw if i % 2 == 0 else float(rng.uniform(0, 0.2))
        E = n_mc
        wrong = rng.random((E, n)) < pi
        rej = np.where(wrong, rng.random((E, n)) < 1 - far, rng.random((E, n)) < frr)
        nu = pi * (1 - far) + (1 - pi) * frr
        rej_r = rng.random((E, n)) < nu
        fix, dmg = rng.random((E, n)) < eta, rng.random((E, n)) < delta

        def util(r: np.ndarray) -> np.ndarray:
            right_after = np.where(r, np.where(wrong, fix, ~dmg), ~wrong)
            return (Wv * right_after - np.where(wrong, Cw, Cr) * r).sum(1)

        dg = util(rej) - n * kappa - util(rej_r)
        clg = voi_closed(n, pi, 1 - far - frr, eta * Wv, delta * Wv, kappa, Cw - Cr)
        worst_z_guard = max(worst_z_guard, abs_z(dg, clg))
    # power: on a population with a large need-dependent cost the simulation rejects the constant-cost formula
    P = POWER_POP
    dp = simulate_gate(rng, max(n_mc, 4000), P["n"], P["pi"], P["tpr"], P["fpr"], P["B"], P["H"], P["C1"],
                       P["kappa"], C0=P["C0"])
    J = P["tpr"] - P["fpr"]
    z_right = abs_z(dp, voi_closed(P["n"], P["pi"], J, P["B"], P["H"], P["kappa"], P["C1"] - P["C0"]))
    z_const = abs_z(dp, voi_closed(P["n"], P["pi"], J, P["B"], P["H"], P["kappa"]))
    ok = (worst_z <= Z_MC and worst_z_corr <= Z_MC and worst_z_guard <= Z_MC and worst_z_acc <= Z_MC
          and worst_exact <= TOL and worst_exact_corr <= TOL and worst_cancel <= TOL and worst_acc <= TOL
          and bias_fail == 0 and n_need_dep > 0 and z_right <= Z_MC and z_const > 2 * Z_MC)
    return {"ok": ok, "n_pop": n_pop, "n_need_dependent_cost": n_need_dep,
            "exact_max_error": worst_exact, "exact_max_error_rate_correction": worst_exact_corr,
            "constant_cost_cancels_max_error": worst_cancel, "need_dependent_bias_failures": bias_fail,
            "accuracy_gap_exact_max_error": worst_acc,
            "mc_max_abs_z": worst_z, "mc_max_abs_z_rate_correction": worst_z_corr,
            "mc_max_abs_z_accuracy_gap": worst_z_acc, "mc_max_abs_z_guard_with_damage": worst_z_guard,
            "power_pop": {"z_need_dependent_formula": z_right, "z_constant_cost_formula": z_const}}


# ====================================================================== AUROC and J
def check_auroc(rng: np.random.Generator) -> dict[str, Any]:
    from statistics import NormalDist

    from jevharness.bench.metrics import auroc

    N = NormalDist()
    bound_fail = 0
    for d in np.linspace(0.0, 4.0, 41):             # binormal equal-variance: concave ROC
        auc = N.cdf(d / math.sqrt(2))
        jmax = 2 * N.cdf(d / 2) - 1                  # Youden max at the midpoint threshold
        ts = np.linspace(-6, 10, 4001)
        jgrid = max(1 - N.cdf(t - d) - (1 - N.cdf(t)) for t in ts)
        if jgrid > 2 * auc - 1 + 1e-9 or abs(jgrid - jmax) > 1e-3:
            bound_fail += 1
    # binary score: AUROC (ties 1/2) = (1 + J) / 2 exactly
    bin_err = 0.0
    for _ in range(50):
        y = (rng.random(400) < rng.uniform(0.1, 0.9)).astype(float)
        tpr, fpr = rng.uniform(0, 1), rng.uniform(0, 1)
        s = np.where(y == 1, rng.random(400) < tpr, rng.random(400) < fpr).astype(float)
        if y.min() == y.max():
            continue
        J = s[y == 1].mean() - s[y == 0].mean()
        bin_err = max(bin_err, abs(auroc(s, y) - (1 + J) / 2))
    # a non-concave ROC: (0,0) -> (0.5,0) -> (0.5,1) -> (1,1): AUROC 0.5, max J 0.5 > 2 AUROC - 1 = 0
    s = np.array([3] * 50 + [2] * 100 + [1] * 50, float)        # 100 negatives at levels 3 / 1, 100 positives at 2
    y = np.array([0] * 50 + [1] * 100 + [0] * 50, float)
    auc_nc = auroc(s, y)
    jmax_nc = max(float((s[y == 1] >= t).mean() - (s[y == 0] >= t).mean()) for t in (1, 2, 3, 4))
    ok = bound_fail == 0 and bin_err <= 1e-12 and jmax_nc > 2 * auc_nc - 1 + 0.4
    return {"ok": ok, "binormal_bound_failures": bound_fail, "binary_score_max_error": bin_err,
            "nonconcave_counterexample": {"auroc": auc_nc, "max_J": jmax_nc}}


# ====================================================================== the repository's estimators
def exact_records(c_levels: np.ndarray, n_per: int, g: np.ndarray, q: np.ndarray, other: np.ndarray | None = None
                  ) -> list[dict[str, Any]]:
    """Synthetic paired decision records with exact empirical accuracies: at level c_j, round(n g_j) System-1-correct
    and round(n q_j) LLM-correct decisions; ``other[j]`` of them answered OTHER (not executable)."""
    recs = []
    for j, cj in enumerate(c_levels):
        n1, n2 = int(round(n_per * g[j])), int(round(n_per * q[j]))
        no = int(other[j]) if other is not None else 0
        for i in range(n_per):
            recs.append({"arm": "syn", "task_id": f"t{j}-{i}", "step": 0, "c": float(cj), "s1_ok": i < n1,
                         "s2_ok": i < n2, "acceptable": i < n_per - no, "s1_cost": 0.0, "s2_cost": 0.0,
                         "s1_lat_ms": 0.0, "s2_lat_ms": 0.0, "final_ok": i < n2, "episode_success": True})
    return recs


def check_code(rng: np.random.Generator, n_pop: int) -> dict[str, Any]:
    from jevharness.bench import meta_gold as MG
    from jevharness.bench import threshold_theory as TT
    from jevharness.bench.metrics import voi_identity

    proj_fail, prop1_fail, loss_fail, gate_fail, pred_fail, ident_fail = 0, 0, 0, 0, 0, 0
    details: list[str] = []
    for i in range(n_pop):
        L = int(rng.integers(4, 12))
        c_levels = np.round(np.sort(rng.choice(np.arange(10, 101), size=L, replace=False)) / 100.0, 2)
        n_per = 40
        g = np.round(np.clip(c_levels + rng.uniform(-0.15, 0.1, L), 0, 1) * n_per) / n_per
        q = np.round(np.clip(rng.uniform(0.6, 1.0) + rng.uniform(-0.2, 0.2, L), 0, 1) * n_per) / n_per
        V = float(rng.choice([0.1, 1.0, 10.0]))
        dC = float(V * rng.uniform(0.0, 0.2))
        recs = exact_records(c_levels, n_per, g, q)
        # the deployed estimator: one bin per tie group (n_per >= min_count), advantage per bin, accepted tail
        res = TT.scalar_threshold_projection(recs, V, dC, n_bins=L, min_count=20)
        acc = g - q + dC / V >= -1e-12
        want_tail, want_cross = tail_edge(c_levels, acc), smallest_accepted(c_levels, acc)
        if len(res["bins"]) != L or res["tau_hat"] != want_tail or res["crossing"] != want_cross:
            proj_fail += 1
            details.append(f"projection pop {i}: got {res['tau_hat']}/{res['crossing']}, want {want_tail}/{want_cross}")
        gains = [float(np.sum((g - q + dC / V)[j:])) for j in range(L)]
        jg = int(np.argmax(gains))
        if is_upper_set(acc) and acc.any() and res["tau_gain"] != res["tau_hat"]:
            proj_fail += 1
            details.append(f"projection pop {i}: tail {res['tau_hat']} != gain-max {res['tau_gain']} on an upper set")
        if gains[jg] > 1e-12 and res["tau_gain"] != float(c_levels[jg]):
            proj_fail += 1
        # Prop. 1's marginal rule
        pi_ = TT.plug_in_thresholds(recs, V, dC)
        if abs(pi_["prop1"] - min(max(float(q.mean()) - dC / V, 0.0), 1.0)) > 1e-12:
            prop1_fail += 1
        # loss_matrix = the Prop. 1 loss V (1 - correct) + dC 1[defer], per decision
        a = TT._arrays(recs)
        taus = [0.0, 0.5, float(c_levels[L // 2]), 1.0, math.inf]
        M = TT.loss_matrix(a, taus, V, dC).mean(1)
        for t, m in zip(taus, M):
            want = float(np.mean(np.where(c_levels >= t, (1 - g) * V, (1 - q) * V + dC)))
            if abs(m - want) > 1e-12:
                loss_fail += 1
    # gate_analysis: VoI over the matched-rate random gate = V (e + h) pi (1 - pi) J, an exact empirical identity
    for i in range(n_pop):
        n = 300
        y = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(int)
        score = np.clip(0.5 * y + rng.normal(0.25, 0.25, n), 0, 1)
        recs = [{"score": float(s), "y": int(t), "task_id": f"t{j % 30}"} for j, (s, t) in enumerate(zip(score, y))]
        V, e, h, cf = float(rng.uniform(0.1, 10)), float(rng.uniform(0, 1)), float(rng.uniform(0, 0.5)), \
            float(rng.uniform(0, 0.1))
        out = TT.gate_analysis(recs, V, c_f=cf, e=e, h=h, n_boot=20, seed=i)
        pi = float(y.mean())
        for row in out["frontier"]:
            if any(math.isnan(row[k]) for k in ("tpr", "fpr")):
                continue
            want = V * (e + h) * pi * (1 - pi) * row["J"]
            if abs(row["voi"] - want) > 1e-9 * max(1.0, V):
                gate_fail += 1
        # the gate's plug-in threshold target is tau_T = (C + H) / (B + H) with B = V e, H = V h
        if (e + h) > 0 and abs(out["target_g"] - tau_trigger(V * e, V * h, cf)) > 1e-12:
            gate_fail += 1
    # meta_gold.predicted_gain and metrics.voi_identity = Eq. (voi) with B = W e, H = W h, n = n_bar
    for _ in range(n_pop):
        W, n_bar = float(rng.choice([0.1, 1.0, 10.0])), float(rng.uniform(1, 20))
        pi, J, kappa = float(rng.uniform(0.05, 0.95)), float(rng.uniform(-0.3, 1)), float(rng.uniform(0, 0.01))
        e, h, e_raw, h_raw = (float(x) for x in rng.uniform(0, 1, 4))
        dr = float(rng.uniform(-0.05, 0.05))
        cell = {"gate": "reflection", "stats": {"pi": pi, "J": J}, "efficacy": {"e": e, "h": h, "e_raw": e_raw,
                                                                               "h_raw": h_raw},
                "n_bar": n_bar, "kappa": kappa, "estimable": True, "delta_u": {}}
        pg = MG.predicted_gain(cell, W)
        if abs(pg["pred_net"] - voi_closed(n_bar, pi, J, W * e, W * h, kappa)) > 1e-12 or \
                abs(pg["raw"]["pred_net"] - voi_closed(n_bar, pi, J, W * e_raw, W * h_raw, kappa)) > 1e-12:
            pred_fail += 1
        guard = MG.predicted_gain({**cell, "gate": "stop_guard"}, W)     # guards: raw (vs executing) is primary
        if abs(guard["pred_net"] - guard["raw"]["pred_net"]) > 1e-15:
            pred_fail += 1
        cf = float(rng.uniform(0, 0.1))
        vi = voi_identity(n_bar, W, e, h, pi, J, kappa, dr, cf)
        if abs(vi["delta_u"] - voi_closed(n_bar, pi, J, W * e, W * h, kappa)) > 1e-12 or \
                abs(vi["rate_correction"] + n_bar * dr * (pi * W * e - (1 - pi) * W * h - cf)) > 1e-12:
            ident_fail += 1
    ok = not any((proj_fail, prop1_fail, loss_fail, gate_fail, pred_fail, ident_fail))
    return {"ok": ok, "scalar_threshold_projection_failures": proj_fail, "plug_in_prop1_failures": prop1_fail,
            "loss_matrix_failures": loss_fail, "gate_analysis_voi_failures": gate_fail,
            "predicted_gain_failures": pred_fail, "voi_identity_failures": ident_fail, "details": details[:5]}


# ====================================================================== driver
CHECKS: dict[str, Callable[..., dict[str, Any]]] = {
    "prop1": lambda rng, a: check_prop1(rng, a.n_pop, a.mc),
    "general": lambda rng, a: check_general(rng, a.n_pop, a.mc),
    "statics": lambda rng, a: check_statics(rng, a.n_pop),
    "gain": lambda rng, a: check_gain(rng, a.n_pop, a.mc),
    "guard": lambda rng, a: check_guard(rng, a.n_pop, a.mc),
    "trigger": lambda rng, a: check_trigger(rng, a.n_pop, a.mc),
    "prop2": lambda rng, a: check_prop2(rng, a.n_pop, a.mc),
    "auroc": lambda rng, a: check_auroc(rng),
    "code": lambda rng, a: check_code(rng, a.n_pop),
}


def run(seed: int = 0, n_pop: int = 60, mc: int = 40000, only: list[str] | None = None) -> dict[str, Any]:
    args = argparse.Namespace(n_pop=n_pop, mc=mc)
    out: dict[str, Any] = {"seed": seed, "n_pop": n_pop, "mc": mc, "simulated": True}
    for i, (name, fn) in enumerate(CHECKS.items()):
        if only and name not in only:
            continue
        out[name] = fn(np.random.default_rng([seed, i]), args)
    out["ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict) and "ok" in v)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pop", type=int, default=60, help="random populations per check")
    ap.add_argument("--mc", type=int, default=40000, help="Monte-Carlo decisions (episodes for Prop. 2) per population")
    ap.add_argument("--only", nargs="+", choices=sorted(CHECKS), default=None)
    ap.add_argument("--json", default=None, help="also write the report to this file")
    a = ap.parse_args(argv)
    res = run(a.seed, a.n_pop, a.mc, a.only)
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{'PASS' if v.get('ok') else 'FAIL'}  {k:8s} " + json.dumps({x: y for x, y in v.items() if x != 'ok'},
                                                                                default=float))
    print("ALL PASS" if res["ok"] else "SOME CHECK FAILED")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
