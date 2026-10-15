"""Threshold theory from decision logs: tests of Prop. 1 (paper §3) and of its general form.

    python -m jevharness.bench.threshold_theory RUN_DIR --arm jev_shadow --dp tool_select \
        [--V 0.1 1 10] [--task-value 1.0] [--lam-per-hour 0] [--rule general|qL|prop1] \
        [--train-frac 0.5] [--seed 0] [--n-boot 1000] [--taus h1 | T ...] [--out FILE.json] [--format md|json|latex]
    python -m jevharness.bench.threshold_theory RUN_DIR --calibration-arm --arm jev_shadow_cal --dp tool_select \
        [--task-value 1.0] [--lam-per-hour 0] [--w-grid 0.1 10] [--lam-grid 0.5 5] [--out FILE.json]
        A6 / P0-6: fit tau-hat* for the C1 arms from a CALIBRATION shadow arm (seeds 5000-5149) and emit the
        scalar-threshold projection of Eq. 3 (:func:`scalar_threshold_projection`) at the primary operating point
        W = $1, lambda = 0, with the W / lambda sensitivity rows; the C1 arm's ``tau`` is ``tau_hat`` of the primary row.
        A6.5's sample-size rule (A6.22 (b) 5) decides on the fit's ``n_paired``: below 30 paired decisions ``tau_hat``
        is the marginal ``prop1`` value, below 10 it is 0.9; ``tau_rule`` records the choice next to the threshold.

Prop. 1 (A1 calibration, A2 q_L independent of c, A3 additive loss): accept System-1's answer iff
c >= tau* = q_L - dC / V. Without A1-A2 the pointwise rule is: accept iff g(c) - q_L(c) >= -dC / V (Eq. 3).
From the logs of one decision point this module estimates every quantity and checks the prediction:

1. reliability of the System-1 confidence c: equal-width bins, ECE / adaptive ECE / Brier / AUROC / AURC
   (the scalars come from :mod:`jevharness.bench.metrics`);
2. q_L(c), the LLM's accuracy on the *same* decision as a function of c: bins plus a monotone (isotonic,
   direction chosen by fit) curve, and a test of A2 (logistic-regression slope with a Wald CI, and a
   permutation test of corr(c, 1[LLM correct]));
3. dC, the cost of deferring = mean cost of the LLM call(s) that make the decision (+ lambda x their latency),
   and kappa, System-1's price per decision (paid in both branches, so it does not move the threshold). For
   ``stop`` dC = 0 (``ZERO_DEFER_COST_DPS``): the LLM takes the stop decision inside the step's tool call, which
   the harness makes whether or not System-1 decides stop (a "continue" still needs the LLM's next call, a
   "stop" a forced ``finish`` call), so deferring it buys no LLM call; the logged step-call cost would credit a
   stop delegation with a saving that does not exist. Pass ``--delta-c`` to use another value;
4. V, the stakes of one decision: user-given (``--V``; several values allowed, e.g. the P0-7 grid 0.1 1 10) or
   estimated counterfactually as ``W x (S_right - S_wrong)``: the success rate of episodes in which every
   *executed* decision at the point was right minus that of episodes with at least one wrong executed
   decision, times the per-episode value ``W`` (``--task-value``). This is an APPROXIMATION: harder episodes
   both contain more errors and fail more often (V overestimated), recovery after an error is credited
   (as it should be), and several wrong decisions in one episode are counted once;
5. the plug-in threshold, three ways (all reported; ``--rule`` picks the primary one):
   ``prop1``   q_L - dC/V with the marginal q_L (Eq. 2, A1-A3);
   ``qL``      advantage a(c) = c - q_L(c) + dC/V (A1 kept, A2 relaxed);
   ``general`` advantage a(c) = g(c) - q_L(c) + dC/V (A1 and A2 relaxed; g = isotonic calibration map);
   accepting beats deferring in expectation iff a(c) >= 0 (Eq. 3). When a is non-decreasing, the optimal
   rule is the threshold at the smallest such c (``<rule>_crossing``). The reported threshold is the one that
   maximises the plug-in expected gain sum_{c_i >= tau} a(c_i) over the observed confidences; it equals
   the crossing in the monotone case and is not thrown off by isolated noisy regions of the fitted curves.
   ``<rule>_interval`` says whether {a >= 0} is a threshold region, and ``<rule>_disagreement`` gives the
   fraction of decisions on which the threshold and the pointwise rule differ;
6. the simulated cascade frontier: for each tau, the accuracy / delegation / cost / latency the point would get
   if System-1's answer were accepted when c >= tau (and it is not OTHER) and the LLM's *logged* decision used
   otherwise -- exact for shadow arms, which log both answers at every state;
7. regret, on held-out tasks (seeded split by task id): per-decision expected loss V (1 - acc) + dC 1[defer]
   of tau-hat (estimated on the training tasks) and of the default tau = 0.8, minus that of the best swept tau
   on the test tasks; CIs are paired task-level bootstrap CIs of the loss difference to that tau. The deployed C1
   estimator (:func:`scalar_threshold_projection`, fitted on the training tasks) is an extra row
   (``tau_hat_projection``). The registered H1 grid is ``--taus h1`` (:data:`H1_TAUS`); the default is a fine grid;
8. value of information of the confidence gate: its expected utility minus that of a random gate that
   accepts at the same rate (analytically, no sampling noise), with a task-level bootstrap CI;
9. H1 clause 2 (PREREGISTRATION A6.22 (c)): the over-confidence gap of System-1's paired Choice decisions on the
   training split (mean top-label confidence minus accuracy, one-sided 95 % task-bootstrap lower bound) and the
   recalibrated-default row -- the held-out regret of tau = 0.8 against the best swept tau with its 95 % CI on the raw
   confidences and after temperature scaling (``decision.calibration``, fitted per question kind on the training
   split, applied to the test split's logged distributions) -- per V (``by_V[].h1_clause2``).

Input records: every ``DecisionLog`` of the chosen arm(s) at the decision point. Shadow records carry both
answers; cascade records have the LLM's answer only when they deferred (c < tau of the arm), so pooling
cascade arms biases q_L(c) towards low-confidence states and the frontier is only valid for tau at or below
the arm's own threshold -- use a shadow arm for the headline test. Guard records at tool selection verify a
proposal (a different question) and are skipped. Units: V, dC and kappa are USD per decision (the ledger's
unit); ``--lam-per-hour`` prices latency in USD per hour of critical-path time. Simulated runs are flagged:
pipeline tests and labelled simulation studies only, never headline numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..harness.trajectory import load_trajectories
from ..harness.tools import OTHER
from .metrics import (acceptable_of, aurc, auroc, brier, cf_efficacy, ece, ece_adaptive, has_gold, is_correct,
                      p_yes_of)

NAN = float("nan")
INF = float("inf")
RULES = ("general", "qL", "prop1")
# decision points of the LLM call(s) whose cost is the price of deferring a decision (log fallback when a
# DecisionLog carries no s2_cost_usd: tool arguments and stop are decided by the step's tool call)
LLM_DPS = {"tool_select": ("tool_select",), "tool_params": ("tool_params", "tool_select"),
           "stop": ("tool_select", "tool_params", "stop"), "action": ("action",),
           "reflection": ("reflection",), "routing": ("routing",)}
# decision points whose deferral costs no extra LLM call (module docstring, item 3): dC = 0 per decision
ZERO_DEFER_COST_DPS = frozenset({"stop"})


# ====================================================================== records
def _s1_prediction(r: Mapping[str, Any]) -> tuple[Any, float | None]:
    """System-1's argmax answer and the confidence that refers to it. For yes/no (Noul) records this is
    (p_yes >= 1/2, max(p_yes, 1 - p_yes)) from the logged distribution -- guard / trigger records log the
    *thresholded* value, whose confidence would not refer to it."""
    dist = r.get("s1_dist")
    if not r.get("options") and isinstance(dist, Mapping) and dist.get("yes") is not None:
        p = float(dist["yes"])
        return p >= 0.5, max(p, 1.0 - p)
    if r.get("s1_value") is None:
        return None, None
    conf = r.get("s1_conf")
    return r.get("s1_value"), (float(conf) if conf is not None else None)


def _call_index(t: Mapping[str, Any]) -> dict[tuple[Any, str, str, bool], list[float]]:
    """(step, component, decision point, shadow) -> [cost, latency_ms] summed over the episode's calls."""
    idx: dict[tuple[Any, str, str, bool], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for c in t.get("calls") or []:
        k = (c.get("step"), str(c.get("component")), str(c.get("decision_point")),
             bool((c.get("meta") or {}).get("shadow")))
        idx[k][0] += float(c.get("cost_usd") or 0.0)
        idx[k][1] += float(c.get("latency_ms") or 0.0)
    return idx


def decision_pairs(trajs: Iterable[Mapping[str, Any]], dp: str, arm: str | None = None,
                   gold: str = "acceptable") -> list[dict[str, Any]]:
    """One record per logged decision at ``dp``: System-1 confidence ``c`` and correctness ``s1_ok``, the LLM's
    correctness ``s2_ok`` on the same state (shadow, and deferred cascade decisions; None otherwise),
    ``acceptable`` (System-1's answer can be executed: not OTHER), the executed decision's correctness
    ``final_ok``, the per-decision System-1 / LLM cost and latency, and the episode outcome.

    Golds (docs/META_GOLD_SPEC.md §9.6): tool_select is judged against the acceptable tool set AT(s) when logged
    (``gold="canonical"``: the canonical next tool); stop against ``finish_ok`` at finish steps and
    ``world_complete`` elsewhere (the harness logs ``s2_correct`` accordingly). Doomed / unobserved opportunities are
    skipped as unlabelled. ``traj.probes`` is never read: probes are not decisions of the arm."""
    out: list[dict[str, Any]] = []
    for t in trajs:
        idx: dict[Any, list[float]] | None = None
        success = bool((t.get("result") or {}).get("success"))
        for st in t.get("steps") or []:
            for r in st.get("decisions") or []:
                if r.get("dp") != dp:
                    continue
                mode = str(r.get("mode"))
                if mode == "guard" and dp != "stop":   # verifies a proposal: not the same question
                    continue
                labelled = has_gold(r, gold)
                use_at = dp == "tool_select" and gold != "canonical" and acceptable_of(r) is not None
                pred, c = _s1_prediction(r)
                s1_ok = None if pred is None or not labelled else bool(is_correct(r, pred, gold))
                s2_ok = None
                if labelled and (mode.startswith("shadow") or mode in ("cascade", "off")):
                    if use_at and r.get("s2_value") is not None:
                        s2_ok = bool(is_correct(r, r["s2_value"], gold))
                    elif r.get("s2_correct") is not None:
                        s2_ok = bool(r["s2_correct"])
                    elif r.get("s2_value") is not None:
                        s2_ok = bool(is_correct(r, r["s2_value"], gold))
                final_ok = None
                if r.get("final_value") is not None and labelled:
                    if use_at and not mode.startswith("shadow") and mode not in ("guard", "trigger"):
                        final_ok = bool(is_correct(r, r["final_value"], gold))
                    elif r.get("s2_correct") is not None and r.get("final_value") == r.get("s2_value"):
                        final_ok = s2_ok if use_at else bool(r["s2_correct"])
                    elif not mode.startswith("shadow") and mode not in ("guard", "trigger"):
                        final_ok = bool(is_correct(r, r["final_value"], gold))
                elif mode.startswith("shadow") or mode == "off":
                    final_ok = s2_ok
                shadow = bool(r.get("shadow")) or mode.startswith("shadow")
                s1c, s1l = r.get("s1_cost_usd"), r.get("s1_latency_ms")
                s2c, s2l = r.get("s2_cost_usd"), r.get("s2_latency_ms")
                if dp in ZERO_DEFER_COST_DPS and s2_ok is not None:
                    s2c, s2l = 0.0, 0.0
                if (s1c is None and pred is not None) or (s2c is None and s2_ok is not None):
                    idx = idx if idx is not None else _call_index(t)
                    step = st.get("step")
                    if s1c is None and pred is not None:
                        s1c, s1l = idx.get((step, "s1", dp, shadow), [NAN, NAN])
                    if s2c is None and s2_ok is not None:
                        vals = [idx[(step, "s2", d, False)] for d in LLM_DPS.get(dp, (dp,))
                                if (step, "s2", d, False) in idx]
                        s2c, s2l = (sum(v[0] for v in vals), sum(v[1] for v in vals)) if vals else (NAN, NAN)
                out.append({"arm": arm or t.get("config"), "task_id": str(t.get("task_id")), "step": st.get("step"),
                            "mode": mode, "shadow": shadow, "c": c, "s1_ok": s1_ok, "s2_ok": s2_ok,
                            "acceptable": pred is not None and str(pred) != OTHER, "final_ok": final_ok,
                            "s1_cost": _f(s1c), "s1_lat_ms": _f(s1l), "s2_cost": _f(s2c), "s2_lat_ms": _f(s2l),
                            "episode_success": success, "s1_value": r.get("s1_value"),
                            "tool": st.get("tool") if dp != "tool_select" else None,
                            "question": r.get("question"),
                            # H1 clause 2 (A6.22 (c)): the logged distribution and gold label for the temperature fit
                            "kind": "noul" if (not r.get("options") and isinstance(r.get("s1_dist"), Mapping)
                                               and r["s1_dist"].get("yes") is not None) else "choice",
                            "s1_dist": r.get("s1_dist"), "gold": r.get("gold")})
    return out


def _f(x: Any) -> float:
    return NAN if x is None else float(x)


def paired(recs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Records with System-1 confidence and both correctness labels (the input of items 2 and 5-8)."""
    return [r for r in recs if r.get("c") is not None and r.get("s1_ok") is not None and r.get("s2_ok") is not None]


# ====================================================================== monotone regression
def isotonic_fit(x: Sequence[float], y: Sequence[float], increasing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares monotone regression (pool-adjacent-violators) of y on x. Returns (sorted unique x, fitted
    value at each); predict with ``np.interp(c, *fit)`` (flat beyond the ends, linear in between, so still
    monotone)."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    ux, inv = np.unique(xa, return_inverse=True)
    w = np.bincount(inv).astype(float)
    m = np.bincount(inv, weights=ya) / w
    sign = 1.0 if increasing else -1.0
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for mi, wi in zip(sign * m, w):
        vals.append(float(mi))
        wts.append(float(wi))
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            ww = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / ww
            n = cnt[-2] + cnt[-1]
            del vals[-1], wts[-1], cnt[-1]
            vals[-1], wts[-1], cnt[-1] = v, ww, n
    return ux, sign * np.repeat(np.asarray(vals), cnt)


def monotone_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Isotonic fit in whichever direction fits better (weighted SSE)."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    best: dict[str, Any] | None = None
    for inc in (True, False):
        ux, fy = isotonic_fit(xa, ya, inc)
        sse = float(np.sum((np.interp(xa, ux, fy) - ya) ** 2))
        if best is None or sse < best["sse"] - 1e-12:
            best = {"direction": "increasing" if inc else "decreasing", "x": ux, "y": fy, "sse": sse}
    assert best is not None
    return best


# ====================================================================== 1. reliability
def _bin_index(c: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin of each confidence, with the edge convention of ``metrics.ece`` ((lo, hi], first bin
    closed)."""
    return np.clip(np.ceil(c * n_bins).astype(int) - 1, 0, n_bins - 1)


def reliability(conf: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> dict[str, Any]:
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    if len(c) == 0:
        return {"n": 0}
    b = _bin_index(c, n_bins)
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": int((b == i).sum()),
             "conf": float(c[b == i].mean()), "acc": float(y[b == i].mean())}
            for i in range(n_bins) if (b == i).any()]
    return {"n": int(len(c)), "acc": float(y.mean()), "mean_conf": float(c.mean()), "ece": ece(c, y, n_bins),
            "ace": ece_adaptive(c, y, n_bins), "brier": brier(c, y), "auroc": auroc(c, y), "aurc": aurc(c, y),
            "bins": bins}


# ====================================================================== 2. q_L(c) and the A2 test
def logistic_slope(x: Sequence[float], y: Sequence[float], alpha: float = 0.05) -> dict[str, float]:
    """Logistic regression y ~ 1 + x (IRLS, x centred): slope, Wald standard error and (1 - alpha) CI. nan when
    y has no variation."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) < 3 or ya.min() == ya.max() or xa.min() == xa.max():
        return {"slope": NAN, "se": NAN, "ci_lo": NAN, "ci_hi": NAN, "z": NAN}
    X = np.column_stack([np.ones_like(xa), xa - xa.mean()])
    beta = np.zeros(2)
    ridge = 1e-8 * np.eye(2)
    H = np.eye(2)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = np.clip(p * (1 - p), 1e-12, None)
        H = X.T @ (X * W[:, None]) + ridge
        step = np.linalg.solve(H, X.T @ (ya - p) - ridge @ beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    se = float(math.sqrt(max(np.linalg.inv(H)[1, 1], 0.0)))
    zq = NormalDist().inv_cdf(1 - alpha / 2)
    return {"slope": float(beta[1]), "se": se, "ci_lo": float(beta[1] - zq * se), "ci_hi": float(beta[1] + zq * se),
            "z": float(beta[1] / se) if se > 0 else NAN}


def permutation_test(x: Sequence[float], y: Sequence[float], n_perm: int = 2000, seed: int = 0) -> dict[str, float]:
    """Two-sided permutation test of H0: y independent of x, statistic = Pearson correlation."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) < 3 or xa.std() == 0 or ya.std() == 0:
        return {"r": NAN, "p": NAN, "n_perm": 0}
    xs = (xa - xa.mean()) / xa.std()
    r = float(np.mean(xs * (ya - ya.mean()) / ya.std()))
    rng = np.random.default_rng(seed)
    count, done = 0, 0
    while done < n_perm:
        k = min(500, n_perm - done)
        perm = np.array([rng.permutation(ya) for _ in range(k)])
        rp = ((perm - ya.mean()) / ya.std()) @ xs / len(xs)
        count += int(np.sum(np.abs(rp) >= abs(r) - 1e-12))
        done += k
    return {"r": r, "p": (1 + count) / (1 + n_perm), "n_perm": n_perm}


def q_l_curve(recs: Sequence[Mapping[str, Any]], n_bins: int = 10, n_perm: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> dict[str, Any]:
    """q_L(c) on paired records: bins (with System-1 accuracy g in the same bin), a monotone fit, the
    error overlap of the two systems, and the A2 tests."""
    ps = paired(recs)
    if not ps:
        return {"n": 0}
    c = np.array([r["c"] for r in ps], float)
    s2 = np.array([float(r["s2_ok"]) for r in ps])
    s1 = np.array([float(r["s1_ok"]) for r in ps])
    b = _bin_index(c, n_bins)
    fit = monotone_fit(c, s2)
    bins = []
    for i in range(n_bins):
        m = b == i
        if m.any():
            bins.append({"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": int(m.sum()), "conf": float(c[m].mean()),
                         "q_L": float(s2[m].mean()), "g": float(s1[m].mean()),
                         "q_L_fit": float(np.interp(c[m].mean(), fit["x"], fit["y"]))})
    slope = logistic_slope(c, s2, alpha)
    perm = permutation_test(c, s2, n_perm, seed)
    s1_wrong = s1 == 0
    return {"n": len(ps), "q_L": float(s2.mean()), "s1_acc": float(s1.mean()), "bins": bins,
            "fit": {"direction": fit["direction"], "x": fit["x"].tolist(), "y": fit["y"].tolist()},
            "a2": {"logit_slope": slope, "perm": perm,
                   "rejected": bool((slope["ci_lo"] > 0 or slope["ci_hi"] < 0) or (perm["p"] < alpha)),
                   "alpha": alpha},
            "overlap": {"p_llm_wrong_given_s1_wrong": float(1 - s2[s1_wrong].mean()) if s1_wrong.any() else NAN,
                        "p_llm_wrong_given_s1_right": float(1 - s2[~s1_wrong].mean()) if (~s1_wrong).any() else NAN,
                        "phi": _phi(s1, s2)}}


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


# ====================================================================== 3. costs
def cost_terms(recs: Sequence[Mapping[str, Any]], lam_per_hour: float = 0.0,
               delta_c: float | None = None, couple_by_step: bool = False) -> dict[str, Any]:
    """dC = mean LLM cost of the decision (+ lambda x latency) and kappa = mean System-1 cost per decision.

    ``couple_by_step`` (tool_params, PREREGISTRATION A6.11 (c)): several argument decisions of one call share ONE
    avoidable System-2 call, so the avoided cost is credited once per (task, step) and split equally over that
    step's argument decisions instead of being credited to every slot in full (an approximation stated as such:
    accepting one argument avoids nothing while another slot still needs the call)."""
    lam = lam_per_hour / 3600.0 / 1000.0          # USD per ms
    share: dict[tuple[Any, Any, Any], int] = defaultdict(int)
    if couple_by_step:
        for r in recs:
            if r.get("s2_ok") is not None:
                share[(r.get("arm"), r.get("task_id"), r.get("step"))] += 1

    def split(r: Mapping[str, Any], v: Any) -> float:
        k = share.get((r.get("arm"), r.get("task_id"), r.get("step")), 1) if couple_by_step else 1
        return NAN if v is None else float(v) / max(1, k)

    s2c = np.array([split(r, r["s2_cost"]) for r in recs if r.get("s2_ok") is not None], float)
    s2l = np.array([split(r, r["s2_lat_ms"]) for r in recs if r.get("s2_ok") is not None], float)
    s1c = np.array([r["s1_cost"] for r in recs if r.get("c") is not None], float)
    s1l = np.array([r["s1_lat_ms"] for r in recs if r.get("c") is not None], float)

    def mean(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        return float(a.mean()) if len(a) else NAN

    d_usd, d_ms, k_usd, k_ms = mean(s2c), mean(s2l), mean(s1c), mean(s1l)
    est = (0.0 if math.isnan(d_usd) else d_usd) + lam * (0.0 if math.isnan(d_ms) else d_ms)
    return {"delta_c": float(delta_c) if delta_c is not None else est, "delta_c_source":
            "user" if delta_c is not None else ("ledger, split over the step's argument decisions" if couple_by_step
                                                else "ledger"), "llm_usd": d_usd, "llm_latency_ms": d_ms,
            "slots_per_call": (float(np.mean(list(share.values()))) if share else NAN) if couple_by_step else 1.0,
            "kappa": (0.0 if math.isnan(k_usd) else k_usd) + lam * (0.0 if math.isnan(k_ms) else k_ms),
            "s1_usd": k_usd, "s1_latency_ms": k_ms, "lam_per_hour": lam_per_hour,
            "cost_ratio_llm_over_s1": d_usd / k_usd if k_usd and not math.isnan(k_usd) else NAN}


# ====================================================================== 4. stakes V
def counterfactual_v(recs: Sequence[Mapping[str, Any]], task_value: float = 1.0, n_boot: int = 1000,
                     seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """V-hat = W x (S_right - S_wrong) over episodes (see the module docstring: an approximation)."""
    by_ep: dict[tuple[Any, str], list[bool]] = defaultdict(list)
    succ: dict[tuple[Any, str], bool] = {}
    for r in recs:
        if r.get("final_ok") is None:
            continue
        k = (r.get("arm"), r["task_id"])
        by_ep[k].append(bool(r["final_ok"]))
        succ[k] = bool(r["episode_success"])
    right = np.array([float(succ[k]) for k, v in by_ep.items() if all(v)])
    wrong = np.array([float(succ[k]) for k, v in by_ep.items() if not all(v)])
    out: dict[str, Any] = {"task_value": task_value, "n_right": len(right), "n_wrong": len(wrong),
                           "success_right": float(right.mean()) if len(right) else NAN,
                           "success_wrong": float(wrong.mean()) if len(wrong) else NAN}
    if not len(right) or not len(wrong):
        return {**out, "V": NAN, "ci": [NAN, NAN]}
    v = task_value * (right.mean() - wrong.mean())
    rng = np.random.default_rng(seed)
    br = right[rng.integers(0, len(right), (n_boot, len(right)))].mean(1)
    bw = wrong[rng.integers(0, len(wrong), (n_boot, len(wrong)))].mean(1)
    boot = task_value * (br - bw)
    return {**out, "V": float(v), "ci": [float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))]}


# ====================================================================== 5. plug-in thresholds
def plug_in_thresholds(recs: Sequence[Mapping[str, Any]], V: float, delta_c: float) -> dict[str, Any]:
    """tau-hat under each rule (see the module docstring), on paired records."""
    ps = [r for r in paired(recs) if r["acceptable"]]
    if not ps or not V > 0:
        return {"prop1": NAN, "qL": NAN, "general": NAN, "n": len(ps)}
    c = np.array([r["c"] for r in ps], float)
    s1 = np.array([float(r["s1_ok"]) for r in ps])
    s2 = np.array([float(r["s2_ok"]) for r in paired(recs)])
    cs = np.array([r["c"] for r in paired(recs)], float)
    k = delta_c / V
    q = monotone_fit(cs, s2)
    gx, gy = isotonic_fit(c, s1, True)
    grid, inv, cnt = np.unique(c, return_inverse=True, return_counts=True)
    q_at = np.interp(grid, q["x"], q["y"])
    out: dict[str, Any] = {"n": len(ps), "k": k, "q_L": float(s2.mean()),
                           "prop1": float(min(max(s2.mean() - k, 0.0), 1.0))}
    for rule, g_at in (("qL", grid), ("general", np.interp(grid, gx, gy))):
        adv = g_at - q_at + k                          # > 0: accepting beats deferring at this c
        ok = adv >= -1e-12
        # plug-in expected gain of the threshold rule "accept iff c >= grid_j" over deferring everything
        gain = np.cumsum((adv * cnt)[::-1])[::-1]
        j = int(np.argmax(gain))
        tau = float(grid[j]) if gain[j] > 1e-12 else INF
        out[rule] = tau
        out[f"{rule}_crossing"] = float(grid[np.argmax(ok)]) if ok.any() else INF
        out[f"{rule}_interval"] = bool(not ok.any() or ok[np.argmax(ok):].all())
        out[f"{rule}_disagreement"] = float(np.mean(ok[inv] != (c >= tau)))
    return out


# ====================================================================== 5b. A6 / P0-6: scalar-threshold projection of Eq. 3
CAL_SEEDS = (5000, 5149)                 # the calibration shadow arms' seeds (jev_shadow_cal, jev_shadow_cal_s2lite)


def equal_mass_bins(c: np.ndarray, n_bins: int = 10, min_count: int = 20) -> list[tuple[float, float]]:
    """Equal-mass bins of the confidences: edges at the quantiles of ``c`` snapped to observed values (tie groups are
    never split; A6.16); a bin with fewer than ``min_count``
    decisions is merged with its upper neighbour (the last one with the one below it) until every bin has at least
    ``min_count`` (or one bin remains). Returns [(lo, hi)] with lo inclusive, hi exclusive (the last hi = +inf), the
    first lo = -inf."""
    c = np.asarray(c, float)
    if len(c) == 0:
        return [(-INF, INF)]
    n_bins = max(1, min(n_bins, len(c)))
    # A6.16 (review round 7): identical deployed confidence values are never split across bins -- every edge is
    # snapped to a distinct observed value, so a whole tie group (e.g. the reported-one set) stays in one bin and
    # the deployed `>=` comparator reproduces the bin's acceptance counts; fewer than n_bins bins may result
    distinct = np.unique(c)
    edges = []
    for q in np.linspace(0, 1, n_bins + 1)[1:-1]:
        v = float(np.quantile(c, q))
        k = int(np.searchsorted(distinct, v, side="left"))
        if k < len(distinct):
            edges.append(float(distinct[k]))
    edges = sorted(set(e for e in edges if e > float(distinct[0])))
    bins = [(-INF, edges[0])] + [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)] + [(edges[-1], INF)] \
        if edges else [(-INF, INF)]

    def count(b: tuple[float, float]) -> int:
        return int(np.sum((c >= b[0]) & (c < b[1])))

    i = 0
    while len(bins) > 1 and i < len(bins):
        if count(bins[i]) < min_count:
            j = i + 1 if i + 1 < len(bins) else i - 1       # merge with the upper neighbour, the last with the lower
            lo, hi = min(bins[i][0], bins[j][0]), max(bins[i][1], bins[j][1])
            k = min(i, j)
            bins = bins[:k] + [(lo, hi)] + bins[max(i, j) + 1:]
            i = k
        else:
            i += 1
    return bins


def scalar_threshold_projection(recs: Sequence[Mapping[str, Any]], V: float, delta_c: float, n_bins: int = 10,
                                min_count: int = 20) -> dict[str, Any]:
    """The deployed C1 rule (A6 / P0-6): the pointwise rule of Eq. 3 -- accept System-1's answer at confidence c iff
    a(c) = g(c) - q_L(c) + dC / V >= 0 (g = System-1 accuracy at c, q_L = the LLM's accuracy at c) -- estimated in
    ``n_bins`` equal-mass bins of c with at least ``min_count`` decisions each (smaller bins merged;
    :func:`equal_mass_bins`), projected onto ONE scalar threshold:

        tau_hat = the smallest c at which Eq. 3 accepts and keeps accepting: the lower edge of the lowest bin from
        which every bin above also has a(c) >= 0 (the accepting tail); +inf (defer everything) when no bin accepts.

    When a(c) is monotone (the Prop. 1 regime) this is exactly the crossing of Eq. 3; when it is not, bins below the
    tail that accept in isolation are listed as ``islands`` and the pointwise rule is the secondary row
    (``pointwise``: per-bin accept flags, and ``disagreement`` = the share of decisions on which the scalar rule and
    the pointwise rule differ). ``crossing`` (the first accepting bin's lower edge) and ``tau_gain`` (the bin edge
    maximising the plug-in gain of the scalar rule, as in :func:`plug_in_thresholds`) are reported next to it. A bin
    whose q_L (or g) is undefined falls back to the global q_L (resp. g). The confidence edge reported is the lowest
    OBSERVED confidence of the tail bin (the arm accepts iff c >= tau_hat)."""
    ps = paired(recs)
    out: dict[str, Any] = {"n": len(ps), "V": V, "delta_c": delta_c, "n_bins": n_bins, "min_count": min_count}
    if not ps or not V > 0:
        return {**out, "tau_hat": NAN, "crossing": NAN, "tau_gain": NAN, "bins": [], "monotone": None,
                "note": "no paired records or V <= 0"}
    k = delta_c / V
    c = np.array([r["c"] for r in ps], float)
    s2 = np.array([float(r["s2_ok"]) for r in ps])
    s1 = np.array([float(r["s1_ok"]) for r in ps])
    acc = np.array([bool(r["acceptable"]) for r in ps])
    q_glob, g_glob = float(s2.mean()), float(s1[acc].mean()) if acc.any() else NAN
    bins = []
    for lo, hi in equal_mass_bins(c, n_bins, min_count):
        m = (c >= lo) & (c < hi)
        ma = m & acc
        q = float(s2[m].mean()) if m.any() else q_glob
        g = float(s1[ma].mean()) if ma.any() else g_glob
        a = g - q + k
        bins.append({"lo": lo, "hi": hi, "c_min": float(c[m].min()) if m.any() else NAN,
                     "c_mean": float(c[m].mean()) if m.any() else NAN, "n": int(m.sum()), "n_acceptable": int(ma.sum()),
                     "q_L": q, "g": g, "advantage": a, "accept": bool(a >= -1e-12),
                     "fallback": {"q_L": not m.any(), "g": not ma.any()}})
    flags = [b["accept"] for b in bins]
    out["bins"] = bins
    out["k"] = k
    # the accepting tail
    tail = None
    for j in range(len(bins)):
        if all(flags[j:]):
            tail = j
            break
    out["tail_bin"] = tail
    out["tau_hat"] = float(bins[tail]["c_min"]) if tail is not None else INF
    first = next((j for j, f in enumerate(flags) if f), None)
    out["crossing"] = float(bins[first]["c_min"]) if first is not None else INF
    out["islands"] = [j for j, f in enumerate(flags) if f and (tail is None or j < tail)]
    out["monotone"] = bool(tail is not None and first == tail) if first is not None else True
    # the gain-maximising scalar threshold over the bins (ties -> the smallest c)
    gains = [sum(b["advantage"] * b["n"] for b in bins[j:]) for j in range(len(bins))]
    jg = int(np.argmax(gains))
    out["tau_gain"] = float(bins[jg]["c_min"]) if gains[jg] > 1e-12 else INF
    # pointwise (secondary) vs the scalar rule
    point = np.zeros(len(c), bool)
    scalar = c >= out["tau_hat"]
    for b, f in zip(bins, flags):
        point |= ((c >= b["lo"]) & (c < b["hi"])) & f
    out["pointwise"] = {"accept_rate": float(point.mean()), "bins_accepting": [j for j, f in enumerate(flags) if f]}
    out["scalar_accept_rate"] = float(scalar.mean())
    out["disagreement"] = float(np.mean(point != scalar))
    return out


def task_seeds(recs: Sequence[Mapping[str, Any]]) -> list[int]:
    """JevGym seeds of the records' tasks (``jevgym-<family>-<seed>-d<n>-<style>[-strict]`` task ids: the third
    field; other ids: their last numeric field, if any; a ``#<trial>`` suffix is ignored)."""
    seeds = set()
    for r in recs:
        parts = str(r.get("task_id", "")).split("#")[0].split("-")
        if parts and parts[0] == "jevgym" and len(parts) >= 3 and parts[2].isdigit():
            seeds.add(int(parts[2]))
        else:
            nums = [x for x in parts if x.isdigit()]
            if nums:
                seeds.add(int(nums[-1]))
    return sorted(seeds)


JEVGYM_FATAL_TOOLS = frozenset({"cancel_order", "refund", "ship_order", "finish"})  # strict JevGym: irreversible / final
_TOOL_IN_Q = re.compile(r"of tool `([^`]+)`")


def decision_tool(r: Mapping[str, Any], dp: str) -> str | None:
    """The tool a record's wrong System-1 answer would act through: the answer itself (tool_select), else the tool
    whose argument the question asks about (parsed from the question text, falling back to the executed tool)."""
    if dp == "tool_select":
        v = r.get("s1_value")
        return str(v) if v is not None else None
    m = _TOOL_IN_Q.search(str(r.get("question") or ""))
    return m.group(1) if m else (str(r["tool"]) if r.get("tool") else None)


def fatal_share(recs: Sequence[Mapping[str, Any]], dp: str = "tool_select",
                fatal_tools: Iterable[str] = JEVGYM_FATAL_TOOLS) -> dict[str, Any]:
    """P(fatal | System-1 wrong) on the paired records: the share of System-1's wrong, executable answers that act
    through an irreversible or final tool (``fatal_tools``; strict JevGym: cancel / refund / ship / finish). The
    A6.11 (c) stakes estimate is V = W x this share. Records whose tool cannot be determined count as fatal
    (conservative)."""
    fatal = set(fatal_tools)
    wrong = [r for r in paired(recs) if r.get("s1_ok") is False and r.get("acceptable")]
    if not wrong:
        return {"n_wrong": 0, "n_fatal": 0, "p_fatal": NAN, "unknown_tool": 0}
    tools = [decision_tool(r, dp) for r in wrong]
    n_fatal = sum(1 for t in tools if t is None or t in fatal)
    return {"n_wrong": len(wrong), "n_fatal": n_fatal, "p_fatal": n_fatal / len(wrong),
            "unknown_tool": sum(1 for t in tools if t is None),
            "by_tool": dict(sorted(((t or "?", tools.count(t)) for t in set(tools)), key=lambda kv: -kv[1]))}


def fit_calibration_arm(recs: Sequence[Mapping[str, Any]], task_value: float = 1.0, lam_per_hour: float = 0.0,
                        w_grid: Sequence[float] = (0.1, 10.0), lam_grid: Sequence[float] = (0.5, 5.0),
                        n_bins: int = 10, min_count: int = 20, n_boot: int = 1000, seed: int = 0,
                        delta_c: float | None = None, cal_seeds: tuple[int, int] | None = CAL_SEEDS,
                        v_fallback: str | float | None = "W", dp: str = "tool_select",
                        fatal_tools: Iterable[str] = JEVGYM_FATAL_TOOLS) -> dict[str, Any]:
    """The A6 / P0-6 calibration-arm fit of tau-hat* for the C1 arms, on the decision records
    (:func:`decision_pairs`) of a calibration shadow arm (``jev_shadow_cal`` / ``jev_shadow_cal_s2lite``, seeds
    5000-5149; ``cal_seeds`` checks the task seeds and notes any outside the range). The H1 fit at the primary
    operating point W = ``task_value`` ($1), lambda = ``lam_per_hour`` (0): V = W (S_right - S_wrong)
    (:func:`counterfactual_v`), dC from the ledger (:func:`cost_terms`), then the scalar projection
    (:func:`scalar_threshold_projection`) -- ``primary.tau_hat`` is the C1 arm's ``tau``. Sensitivity rows: every
    W in ``w_grid`` at lambda = 0 and every lambda in ``lam_grid`` at W = task_value. The existing plug-in rules
    (:func:`plug_in_thresholds`) are reported next to the projection for reference.

    ``v_fallback`` (PREREGISTRATION A6.11): the counterfactual V needs episodes in which System-2 itself decided
    wrongly; a calibration log without one (Flash on JevGym: 150/150 episodes right) leaves V undefined. Then V is
    set to the conservative bound W (``"W"``: every accepted wrong decision loses the task, which on strict JevGym
    tasks is what an accepted wrong state-changing tool does), a number is taken as V at W = ``task_value`` and
    scaled with W on the sensitivity rows, ``"fatal"`` takes V = W x P(fatal | System-1 wrong) from
    :func:`fatal_share` (review round 6; W when System-1 made no wrong decision either), and ``None`` keeps tau-hat
    undefined. ``row["V_source"]`` says which."""
    out: dict[str, Any] = {"n_records": len(recs), "n_paired": len(paired(recs)), "task_value": task_value,
                           "lam_per_hour": lam_per_hour, "n_bins": n_bins, "min_count": min_count, "notes": [],
                           "v_fallback": v_fallback, "fatal_share": fatal_share(recs, dp, fatal_tools)}
    seeds = task_seeds(recs)
    out["seeds"] = {"n_tasks": len({str(r.get("task_id")) for r in recs}), "min": seeds[0] if seeds else None,
                    "max": seeds[-1] if seeds else None}
    if cal_seeds and seeds and (seeds[0] < cal_seeds[0] or seeds[-1] > cal_seeds[1]):
        bad = [x for x in seeds if x < cal_seeds[0] or x > cal_seeds[1]]
        out["notes"].append(f"{len(bad)} task seed(s) outside the calibration range {cal_seeds[0]}-{cal_seeds[1]} "
                            f"(e.g. {bad[:5]}): not a registered calibration arm")

    def row(W: float, lam: float) -> dict[str, Any]:
        v = counterfactual_v(recs, W, n_boot, seed)
        cost = cost_terms(recs, lam, delta_c, couple_by_step=(dp == "tool_params"))
        V = v["V"]
        r: dict[str, Any] = {"W": W, "lam_per_hour": lam, "V": V, "V_ci": v.get("ci"), "delta_c": cost["delta_c"],
                             "delta_c_source": cost["delta_c_source"], "slots_per_call": cost.get("slots_per_call"),
                             "kappa": cost["kappa"], "V_source": "counterfactual",
                             "n_wrong_episodes": v.get("n_wrong")}
        if not V > 0:
            if v_fallback is None or v_fallback == "none":
                r.update({"tau_hat": NAN, "note": "V is not positive: tau-hat undefined"})
                return r
            why = ("no episode with a wrong System-2 decision in the calibration log" if not v.get("n_wrong")
                   else "counterfactual V not positive")
            fs = out["fatal_share"]
            if v_fallback == "fatal" and fs["n_wrong"] > 0:
                V = float(W) * float(fs["p_fatal"])
                label = f"fatal share {fs['n_fatal']}/{fs['n_wrong']}"
            elif v_fallback == "fatal":
                V, label = float(W), "W (System-1 made no wrong decision either)"
            elif v_fallback == "W":
                V, label = float(W), "W"
            else:
                V, label = float(v_fallback) * float(W) / float(task_value), str(v_fallback)
            if not V > 0:
                r.update({"tau_hat": NAN, "V": V, "V_source": f"fallback {label}: not positive",
                          "note": f"V = {V:g} by the A6.11 fallback ({label}): not positive, tau-hat undefined"})
                return r
            r.update({"V": V, "V_ci": None, "V_source": f"fallback {label} ({why}; A6.11)",
                      "note": f"V = {V:g} by the A6.11 fallback ({label}): {why}"})
        proj = scalar_threshold_projection(recs, V, cost["delta_c"], n_bins, min_count)
        r.update({"projection": proj, "tau_hat": proj["tau_hat"], "crossing": proj["crossing"],
                  "tau_gain": proj["tau_gain"], "monotone": proj["monotone"], "disagreement": proj["disagreement"],
                  "plug_in": plug_in_thresholds(recs, V, cost["delta_c"])})
        return r

    out["primary"] = row(task_value, lam_per_hour)
    out["sensitivity"] = [row(W, 0.0) for W in w_grid] + [row(task_value, lam) for lam in lam_grid]
    rule = sample_size_rule(out["n_paired"])
    for r in [out["primary"], *out["sensitivity"]]:
        apply_sample_size_rule(r, rule)
    out["tau_hat"] = out["primary"]["tau_hat"]
    out["tau_rule"] = {"rule": rule, "n_paired": out["n_paired"], "registered": "A6.5; A6.22 (b) 5",
                       "text": SAMPLE_SIZE_RULE_TEXT[rule]}
    return out


# PREREGISTRATION A6.5's sample-size rule (never withdrawn; A6.22 (b) 5 applies it to every decision point the fit
# serves, the tau^2 refit included): the fit's ``n_paired`` (paired decisions of the decision point) decides
MIN_PAIRED_PROJECTION = 30        # >= 30: the deployed scalar-threshold projection (A6.11 (c).7, A6.19)
MIN_PAIRED_PROP1 = 10             # 10-29: the marginal ``prop1`` value the fit reports (``primary.plug_in.prop1``)
FIXED_TAU = 0.9                   # < 10: tau = 0.9
SAMPLE_SIZE_RULE_TEXT = {
    "projection": f"n_paired >= {MIN_PAIRED_PROJECTION}: the scalar-threshold projection (A6.5, A6.11 (c).7, A6.19)",
    "prop1": f"{MIN_PAIRED_PROP1} <= n_paired < {MIN_PAIRED_PROJECTION}: the marginal prop1 value q_L - dC/V "
             "(plug_in.prop1; A6.5)",
    "fixed": f"n_paired < {MIN_PAIRED_PROP1}: tau = {FIXED_TAU} (A6.5)"}


def sample_size_rule(n_paired: int) -> str:
    """A6.5 / A6.22 (b) 5: ``projection`` (>= 30 paired decisions), ``prop1`` (10-29) or ``fixed`` (< 10, tau 0.9)."""
    return ("projection" if n_paired >= MIN_PAIRED_PROJECTION else "prop1" if n_paired >= MIN_PAIRED_PROP1
            else "fixed")


def apply_sample_size_rule(r: dict[str, Any], rule: str) -> None:
    """Write the A6.5 choice into one fit row next to its threshold: ``tau_rule``, and for ``prop1`` / ``fixed`` the
    replaced projection under ``tau_hat_projection`` (the projection itself stays under ``projection``)."""
    r["tau_rule"] = rule
    if rule == "projection":
        return
    r["tau_hat_projection"] = r.get("tau_hat", NAN)
    r["tau_hat"] = float((r.get("plug_in") or {}).get("prop1", NAN)) if rule == "prop1" else FIXED_TAU


def calibration_arm_run(run_dir: str | Path, arms: Sequence[str], dp: str = "tool_select", gold: str = "acceptable",
                        **kw: Any) -> dict[str, Any]:
    """Load the calibration arm(s) of ``run_dir`` and run :func:`fit_calibration_arm` on their records at ``dp``."""
    run_dir = Path(run_dir)
    files = {p.stem: p for p in sorted(run_dir.glob("*.jsonl"))}
    missing = [a for a in arms if a not in files]
    if missing:
        raise FileNotFoundError(f"arm(s) not found in {run_dir}: {missing} (have: {sorted(files)})")
    recs = [r for a in arms for r in decision_pairs(load_trajectories(files[a]), dp, arm=a, gold=gold)]
    out = fit_calibration_arm(recs, dp=dp, **kw)
    out["meta"] = {"run_dir": str(run_dir), "arms": list(arms), "dp": dp, "gold": gold}
    return out


def calibration_markdown(res: Mapping[str, Any]) -> str:
    rows = [res["primary"]] + list(res.get("sensitivity") or [])
    out = [f"## tau-hat* from the calibration arm ({', '.join((res.get('meta') or {}).get('arms', []))}, "
           f"dp = {(res.get('meta') or {}).get('dp')})", "",
           f"paired records: {res['n_paired']}; tasks: {res['seeds']['n_tasks']} (seeds {res['seeds']['min']}-"
           f"{res['seeds']['max']}); bins: {res['n_bins']} equal-mass, >= {res['min_count']} decisions", "",
           "| row | W | λ $/h | V | dC | tau-hat (deployed, A6.5 rule) | crossing | gain-max | monotone | disagreement | prop1 | general |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        pi = r.get("plug_in") or {}
        out.append(f"| {'primary' if i == 0 else 'sensitivity'} | {r['W']:g} | {r['lam_per_hour']:g} | "
                   f"{_fmt(r.get('V'))} | {_fmt(r.get('delta_c'), 5)} | {_fmt(r.get('tau_hat'))} | "
                   f"{_fmt(r.get('crossing'))} | {_fmt(r.get('tau_gain'))} | {r.get('monotone')} | "
                   f"{_fmt(r.get('disagreement'))} | {_fmt(pi.get('prop1'))} | {_fmt(pi.get('general'))} |")
    for i, r in enumerate(rows):
        if r.get("note"):
            out.append(f"\nnote ({'primary' if i == 0 else 'sensitivity'} W={r['W']:g}, λ={r['lam_per_hour']:g}): {r['note']}")
    tr = res.get("tau_rule") or {}
    if tr:
        out.append(f"\ndeployed tau (A6.5 sample-size rule, A6.22 (b) 5): {_fmt(res.get('tau_hat'))} -- {tr.get('text')}")
    for n in res.get("notes") or []:
        out.append(f"\nnote: {n}")
    return "\n".join(out)


# ====================================================================== 6. frontier
def _arrays(recs: Sequence[Mapping[str, Any]], lam_per_hour: float = 0.0) -> dict[str, np.ndarray]:
    ps = paired(recs)
    lam = lam_per_hour / 3600.0 / 1000.0

    def arr(key: str) -> np.ndarray:
        a = np.array([r[key] for r in ps], float)
        return np.nan_to_num(a, nan=0.0)

    return {"c": np.array([r["c"] for r in ps], float), "s1": np.array([float(r["s1_ok"]) for r in ps]),
            "s2": np.array([float(r["s2_ok"]) for r in ps]), "ok": np.array([bool(r["acceptable"]) for r in ps]),
            "s1_cost": arr("s1_cost"), "s2_cost": arr("s2_cost"), "s1_lat": arr("s1_lat_ms"),
            "s2_lat": arr("s2_lat_ms"), "dc": arr("s2_cost") + lam * arr("s2_lat_ms"),
            "task": np.array([f"{r.get('arm')}\x00{r['task_id']}" for r in ps])}


def _accept(a: Mapping[str, np.ndarray], taus: Sequence[float]) -> np.ndarray:
    return (a["c"][None, :] >= np.asarray(taus, float)[:, None]) & a["ok"][None, :]


def loss_matrix(a: Mapping[str, np.ndarray], taus: Sequence[float], V: float,
                delta_c: float | None = None) -> np.ndarray:
    """(n_tau x n) per-decision expected loss V (1 - correct) + dC_i 1[defer] (dC_i: the decision's logged
    LLM cost, or a fixed ``delta_c``)."""
    acc = _accept(a, taus)
    correct = np.where(acc, a["s1"][None, :], a["s2"][None, :])
    dc = a["dc"] if delta_c is None else np.full(len(a["c"]), float(delta_c))
    return V * (1.0 - correct) + (~acc) * dc[None, :]


def cascade_frontier(recs: Sequence[Mapping[str, Any]], taus: Sequence[float], lam_per_hour: float = 0.0,
                     V: float | None = None) -> dict[str, Any]:
    a = _arrays(recs, lam_per_hour)
    n = len(a["c"])
    if n == 0:
        return {"n": 0, "rows": []}
    acc = _accept(a, taus)
    correct = np.where(acc, a["s1"][None, :], a["s2"][None, :])
    rows = []
    for i, t in enumerate(taus):
        row = {"tau": float(t), "acc": float(correct[i].mean()), "delegation": float(acc[i].mean()),
               "llm_calls": float((~acc[i]).mean()),
               "cost_usd": float((a["s1_cost"] + (~acc[i]) * a["s2_cost"]).mean()),
               "latency_ms": float((a["s1_lat"] + (~acc[i]) * a["s2_lat"]).mean())}
        if V is not None:
            row["loss"] = float((V * (1 - correct[i]) + (~acc[i]) * a["dc"]).mean())
        rows.append(row)
    refs = {"llm_only": {"acc": float(a["s2"].mean()), "cost_usd": float(a["s2_cost"].mean()),
                         "latency_ms": float(a["s2_lat"].mean())},
            "s1_only": {"acc": float(np.where(a["ok"], a["s1"], a["s2"]).mean()),
                        "cost_usd": float((a["s1_cost"] + (~a["ok"]) * a["s2_cost"]).mean()),
                        "latency_ms": float((a["s1_lat"] + (~a["ok"]) * a["s2_lat"]).mean())}}
    return {"n": n, "rows": rows, "references": refs}


# ====================================================================== 7. regret on held-out tasks
def split_tasks(recs: Sequence[Mapping[str, Any]], train_frac: float = 0.5,
                seed: int = 0) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Seeded split by task id (all decisions of a task on one side)."""
    tasks = sorted({str(r["task_id"]) for r in recs})
    random.Random(seed).shuffle(tasks)
    k = int(round(train_frac * len(tasks)))
    train = set(tasks[:k])
    return [r for r in recs if str(r["task_id"]) in train], [r for r in recs if str(r["task_id"]) not in train]


def _task_boot(a: Mapping[str, np.ndarray], n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_boot x n_tasks) resampling counts and the task index of every record."""
    uniq, tidx = np.unique(a["task"], return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    W = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws]).astype(float)
    return W, tidx


def regret(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], V: float, V_plugin: float,
           taus: Sequence[float], default_tau: float = 0.8, rule: str = "general", lam_per_hour: float = 0.0,
           delta_c: float | None = None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Regret of tau-hat (from ``train``, with stakes ``V_plugin``) and of ``default_tau`` against the best swept
    tau on ``test`` under the loss with stakes ``V``; bootstrap over test tasks."""
    cost_tr = cost_terms(train, lam_per_hour, delta_c)
    th = plug_in_thresholds(train, V_plugin, cost_tr["delta_c"])
    a_tr, a_te = _arrays(train, lam_per_hour), _arrays(test, lam_per_hour)
    if len(a_te["c"]) == 0 or len(a_tr["c"]) == 0 or not V > 0:
        return {"n_train": len(a_tr["c"]), "n_test": len(a_te["c"]), "thresholds": th}
    grid = list(taus)
    L_tr = loss_matrix(a_tr, grid, V, delta_c).mean(1)
    tau_sweep = float(grid[int(np.argmin(L_tr))])
    cand = {"tau_hat": th[rule], "tau_default": float(default_tau), "tau_sweep_train": tau_sweep,
            **{f"tau_hat_{r}": th[r] for r in RULES if r != rule}}
    # the deployed C1 estimator (A6.11 (c).7) fitted on the same training tasks, as an extra row: H1's tau-hat is the
    # isotonic plug-in above; this row shows the held-out regret of the estimator the C1 arms actually deploy
    proj = scalar_threshold_projection(train, V_plugin, cost_tr["delta_c"]).get("tau_hat", NAN)
    if not math.isnan(proj):
        cand["tau_hat_projection"] = float(proj)
    taus_all = grid + [cand[k] for k in cand]
    M = loss_matrix(a_te, taus_all, V, delta_c)                    # (n_tau x n)
    L = M.mean(1)
    ng = len(grid)
    best = int(np.argmin(L[:ng]))
    W, tidx = _task_boot(a_te, n_boot, seed)
    S = np.stack([np.bincount(tidx, weights=M[i], minlength=W.shape[1]) for i in range(len(taus_all))])
    N = np.bincount(tidx, minlength=W.shape[1]).astype(float)
    Lb = (W @ S.T) / (W @ N)[:, None]                             # (n_boot x n_tau)
    ref = Lb[:, best]                  # paired against the (fixed) test-optimal tau: re-minimising over the
    q = (alpha / 2, 1 - alpha / 2)     # grid in every replicate would bias the regret CI upwards
    out: dict[str, Any] = {"n_train": len(a_tr["c"]), "n_test": len(a_te["c"]), "V": V, "V_plugin": V_plugin,
                           "delta_c_train": cost_tr["delta_c"], "rule": rule, "thresholds": th,
                           "tau_best_test": float(grid[best]), "loss_best_test": float(L[best]), "candidates": {}}
    for j, (name, tau) in enumerate(cand.items()):
        i = ng + j
        rb = Lb[:, i] - ref
        out["candidates"][name] = {"tau": tau, "loss": float(L[i]), "regret": float(L[i] - L[best]),
                                   "regret_over_V": float((L[i] - L[best]) / V),
                                   "regret_ci": [float(np.quantile(rb, q[0])), float(np.quantile(rb, q[1]))]}
    i_hat, i_def = ng, ng + 1
    d = Lb[:, i_def] - Lb[:, i_hat]
    out["default_minus_hat"] = {"diff": float(L[i_def] - L[i_hat]),
                                "ci": [float(np.quantile(d, q[0])), float(np.quantile(d, q[1]))],
                                "p_hat_not_better": float(np.mean(d <= 0))}
    return out


# ====================================================================== 7b. H1 clause 2: over-confidence, recalibrated default
def overconfidence_gap(recs: Sequence[Mapping[str, Any]], n_boot: int = 1000, seed: int = 0,
                       alpha: float = 0.05) -> dict[str, Any]:
    """PREREGISTRATION A6.22 (c), H1 clause 2: the over-confidence gap of System-1's paired Choice decisions (on the
    30 % training split): mean top-label confidence minus accuracy, with a task bootstrap (tasks resampled, the gap
    recomputed over their decisions). ``lb`` is the one-sided (1 - alpha) lower bound (the alpha quantile; ``ci`` the
    two-sided interval, descriptive); the cell is *over-confident* iff ``lb`` > 0."""
    ps = [r for r in paired(recs) if r.get("kind", "choice") == "choice"]
    if not ps:
        return {"n": 0, "n_tasks": 0, "gap": NAN, "lb": NAN, "ci": [NAN, NAN], "over_confident": None}
    d = np.array([float(r["c"]) - float(bool(r["s1_ok"])) for r in ps])
    a = {"task": np.array([f"{r.get('arm')}\x00{r['task_id']}" for r in ps])}
    W, tidx = _task_boot(a, n_boot, seed)
    S = np.bincount(tidx, weights=d, minlength=W.shape[1])
    N = np.bincount(tidx, minlength=W.shape[1]).astype(float)
    boot = (W @ S) / (W @ N)
    lb = float(np.quantile(boot, alpha))
    return {"n": len(ps), "n_tasks": int(W.shape[1]), "mean_conf": float(np.mean([r["c"] for r in ps])),
            "acc": float(np.mean([float(bool(r["s1_ok"])) for r in ps])), "gap": float(d.mean()), "lb": lb,
            "lb_level": 1 - alpha, "ci": [float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))],
            "over_confident": bool(lb > 0)}


def recalibrate(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]],
                min_per_kind: int = 50, temperatures: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]],
                                                                                            dict[str, Any]]:
    """A6.22 (c), H1 clause 2: temperatures fitted on the training split's paired decisions with
    :func:`jevharness.decision.calibration.fit_from_records` (per question kind -- ``choice`` / ``noul`` -- with the
    global T for a kind below ``min_per_kind``), applied to the test split's LOGGED distributions (``s1_dist``): the
    recalibrated confidence is the top-label probability of the rescaled distribution (``max(p, 1 - p)`` for yes/no).
    The argmax, hence System-1's answer and its correctness, is unchanged. ``temperatures`` (a ``fit_from_records``
    result, e.g. ``{"global": 1.0, "per_kind": {}}``) skips the fit. A test record without a logged distribution keeps
    its raw confidence (counted in ``n_kept_raw``)."""
    from ..decision.calibration import fit_from_records, scale_distribution

    if temperatures is None:
        fit = [{"s1_dist": r["s1_dist"], "gold": r["gold"]} for r in paired(train)
               if isinstance(r.get("s1_dist"), Mapping) and r.get("gold") is not None]
        temperatures = fit_from_records(fit, min_per_kind)
    cal = {"global": float(temperatures.get("global", 1.0)), "per_kind": dict(temperatures.get("per_kind") or {}),
           "n": temperatures.get("n")}
    out: list[dict[str, Any]] = []
    kept = 0
    for r in test:
        dist = r.get("s1_dist")
        if r.get("c") is None or not isinstance(dist, Mapping) or not dist:
            kept += int(r.get("c") is not None)
            out.append(dict(r))
            continue
        kind = "noul" if set(dist) == {"yes", "no"} else "choice"
        sd = scale_distribution({str(k): float(v) for k, v in dist.items()}, float(cal["per_kind"].get(kind, cal["global"])))
        out.append({**r, "c_raw": r["c"], "c": float(max(sd.values()))})
    cal.update({"n_test_rescaled": sum(1 for r in out if "c_raw" in r), "n_kept_raw": kept})
    return out, cal


def default_regret(test: Sequence[Mapping[str, Any]], V: float, taus: Sequence[float], default_tau: float = 0.8,
                   lam_per_hour: float = 0.0, delta_c: float | None = None, n_boot: int = 1000, seed: int = 0,
                   alpha: float = 0.05) -> dict[str, Any]:
    """Held-out regret of the fixed ``default_tau`` against the best swept tau of ``taus`` on ``test`` with the paired
    task-bootstrap 95 % CI -- the ``tau_default`` row of :func:`regret`, computed the same way (same loss, same
    bootstrap draws), so that it can be re-evaluated on recalibrated confidences (A6.22 (c), H1 clause 2)."""
    a = _arrays(test, lam_per_hour)
    if len(a["c"]) == 0 or not V > 0:
        return {"n_test": len(a["c"])}
    grid = list(taus)
    M = loss_matrix(a, grid + [float(default_tau)], V, delta_c)
    L = M.mean(1)
    ng = len(grid)
    best = int(np.argmin(L[:ng]))
    W, tidx = _task_boot(a, n_boot, seed)
    S = np.stack([np.bincount(tidx, weights=M[i], minlength=W.shape[1]) for i in (best, ng)])
    N = np.bincount(tidx, minlength=W.shape[1]).astype(float)
    Lb = (W @ S.T) / (W @ N)[:, None]
    rb = Lb[:, 1] - Lb[:, 0]
    q = (alpha / 2, 1 - alpha / 2)
    return {"n_test": len(a["c"]), "tau_default": float(default_tau), "tau_best_test": float(grid[best]),
            "loss_default": float(L[ng]), "loss_best_test": float(L[best]), "regret": float(L[ng] - L[best]),
            "regret_ci": [float(np.quantile(rb, q[0])), float(np.quantile(rb, q[1]))]}


def h1_clause2(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], V: float,
               taus: Sequence[float], default_tau: float = 0.8, lam_per_hour: float = 0.0,
               delta_c: float | None = None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
               overconfidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """PREREGISTRATION A6.22 (c), H1 clause 2, for one (backend, decision point, System-2) cell at stakes ``V``: the
    over-confidence gap on the training split (:func:`overconfidence_gap`) and the recalibrated-default regret row --
    the held-out regret of the fixed tau = ``default_tau`` against the best swept tau with its 95 % CI on the raw
    (logged) confidences and on the recalibrated ones (:func:`recalibrate`). The cell *returns to the front* iff the
    raw CI lower bound is > 0 and the recalibrated one is <= 0. ``dist_t1`` (descriptive) scores the logged
    distributions at T = 1, i.e. the score the recalibration rescales, so that a difference between the raw and the
    recalibrated row that comes from scoring the distribution instead of the wire confidence is visible. Binary-score
    caveat (A6.22 (c)): a monotone recalibration leaves the attainable front unchanged; this row measures the cost of
    a mis-set fixed threshold only."""
    oc = dict(overconfidence) if overconfidence is not None else overconfidence_gap(train, n_boot, seed, alpha)
    kw = dict(default_tau=default_tau, lam_per_hour=lam_per_hour, delta_c=delta_c, n_boot=n_boot, seed=seed,
              alpha=alpha)
    test_recal, cal = recalibrate(train, test)
    test_t1, _ = recalibrate(train, test, temperatures={"global": 1.0, "per_kind": {}})
    raw = default_regret(test, V, taus, **kw)
    rec = default_regret(test_recal, V, taus, **kw)
    t1 = default_regret(test_t1, V, taus, **kw)
    ret = None
    if "regret_ci" in raw and "regret_ci" in rec:
        ret = bool(raw["regret_ci"][0] > 0 and rec["regret_ci"][0] <= 0)
    return {"V": V, "default_tau": default_tau, "overconfidence": oc, "over_confident": oc.get("over_confident"),
            "temperatures": cal, "raw": raw, "recalibrated": rec, "dist_t1": t1, "returns_to_front": ret,
            "registered": "A6.22 (c) H1 clause 2 (W = $1, lambda = 0 is the registered operating point)"}


# ====================================================================== 8. value of information of the gate
def gate_voi(recs: Sequence[Mapping[str, Any]], tau: float, V: float, lam_per_hour: float = 0.0,
             delta_c: float | None = None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Expected utility of the confidence gate (accept iff c >= tau) minus that of a random gate that accepts
    each (acceptable) decision with the same probability, in USD and in accuracy per decision."""
    a = _arrays(recs, lam_per_hour)
    n = len(a["c"])
    if n == 0 or not V > 0:
        return {"n": n}
    dc = a["dc"] if delta_c is None else np.full(n, float(delta_c))
    acc = ((a["c"] >= tau) & a["ok"]).astype(float)
    okf = a["ok"].astype(float)

    def stats(w: np.ndarray) -> tuple[float, float, float]:
        rate = float((w * acc).sum() / max((w * okf).sum(), 1e-12))
        u_gate = acc * (V * a["s1"]) + (1 - acc) * (V * a["s2"] - dc)
        u_rand = okf * (rate * V * a["s1"] + (1 - rate) * (V * a["s2"] - dc)) + (1 - okf) * (V * a["s2"] - dc)
        c_gate = acc * a["s1"] + (1 - acc) * a["s2"]
        c_rand = okf * (rate * a["s1"] + (1 - rate) * a["s2"]) + (1 - okf) * a["s2"]
        tot = w.sum()
        return rate, float((w * (u_gate - u_rand)).sum() / tot), float((w * (c_gate - c_rand)).sum() / tot)

    rate, voi, dacc = stats(np.ones(n))
    W, tidx = _task_boot(a, n_boot, seed)
    boot = np.array([stats(W[b][tidx])[1:] for b in range(W.shape[0])])
    q = (alpha / 2, 1 - alpha / 2)
    return {"n": n, "tau": tau, "accept_rate": rate, "voi_usd": voi, "voi_over_V": voi / V,
            "voi_ci": [float(np.quantile(boot[:, 0], q[0])), float(np.quantile(boot[:, 0], q[1]))],
            "acc_gain": dacc,
            "acc_gain_ci": [float(np.quantile(boot[:, 1], q[0])), float(np.quantile(boot[:, 1], q[1]))]}


# ====================================================================== gates (docs/META_GOLD_SPEC.md §9.6)
GATE_COST_DPS = {"reflection": "reflection", "routing": "tool_select", "stop_guard": "tool_select"}


def gate_records(trajs: Iterable[Mapping[str, Any]], gate: str, arm: str | None = None,
                 modes: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """(score, y, task) records of a trigger / guard gate on its primary gold, from shadow / trigger / guard logs
    with ``gold_status == "ok"`` (never ``traj.probes``; ``modes`` restricts the logged modes). A trigger's score is
    p_yes and y = gold (need); a guard fires (rejects) on low p_yes, so its score is 1 - p_yes and y = the proposal
    is wrong."""
    from .meta_gold import gate_records as _gr

    out = []
    guard = gate in ("stop_guard", "tool_guard")
    for r in _gr(list(trajs), gate if gate != "tool_guard" else "tool_select", modes=modes):
        if r.get("gold_status") not in (None, "ok") or not isinstance(r.get("gold"), bool):
            continue
        if gate == "tool_guard" and r.get("mode") != "guard":
            continue
        p = p_yes_of(r)
        if p is None:
            continue
        out.append({"arm": arm or r.get("config"), "task_id": str(r["task_id"]), "p_yes": p,
                    "score": 1.0 - p if guard else p, "y": (not r["gold"]) if guard else bool(r["gold"]),
                    "aux": r.get("gold_aux") or {}, "mode": r.get("mode")})
    return out


def gate_cost(trajs: Sequence[Mapping[str, Any]], gate: str) -> float:
    """c_f from the ledger: the mean cost of the System-2 call a firing buys (reflection: the reflection call;
    routing: a large-model step minus a small-model step; stop guard: one retry step; memory: 0 here -- the
    stored item's input-token cost is paid by every later call)."""
    from .meta_gold import _fired_cost

    return _fired_cost(trajs, gate)


def _gate_utility(score: np.ndarray, y: np.ndarray, tau: float, V: float, e: float, h: float, c_f: float) -> float:
    """Per-opportunity expected utility of firing at score >= tau, relative to never firing."""
    fire = score >= tau
    gain = V * (y * e - (1 - y) * h) - c_f
    return float(np.mean(fire * gain)) if len(score) else NAN


def gate_analysis(recs: Sequence[Mapping[str, Any]], V: float, c_f: float = 0.0, e: float | None = None,
                  h: float | None = None, taus: Sequence[float] | None = None, train_frac: float = 0.5,
                  seed: int = 0, n_boot: int = 1000, alpha: float = 0.05, branch: str = "verdict_alt") -> dict[str, Any]:
    """Threshold theory for a gate (exploratory; H1 extended to gates):

    - plug-in rule: fire iff g(s) (e + h) >= c_f / V + h, i.e. s >= s* = g^-1((c_f / V + h) / (e + h)), with g the
      isotonic calibration map of the score s against the gold (score = p_yes for triggers, 1 - p_yes for guards);
    - e, h: the counterfactual efficacies (``metrics.cf_efficacy`` on the records, net of the placebo) unless given;
    - the simulated gate frontier U(tau) = E[1(s >= tau) (V (y e - (1 - y) h) - c_f)] and its VoI against a random
      gate at the matched rate (the item-8 analogue), analytically;
    - regret of s* (fitted on the training tasks) and of tau = 0.5 / 0.8 against the best swept tau on held-out tasks,
      with a task-bootstrap CI."""
    taus = list(taus) if taus is not None else [round(x, 2) for x in np.arange(0.0, 1.0001, 0.01)]
    out: dict[str, Any] = {"n": len(recs), "V": V, "c_f": c_f}
    if not recs or not V > 0:
        return out
    if e is None or h is None:
        eff = cf_efficacy([{"gold_aux": r["aux"]} for r in recs], branch)
        out["efficacy"] = eff
        e = eff.get("e", NAN) if e is None else e
        h = eff.get("h", NAN) if h is None else h
    out.update({"e": e, "h": h})
    if e is None or h is None or math.isnan(e) or math.isnan(h):
        out["note"] = "no efficacy (e, h): pass --e / --h or measure counterfactual branches"
        return out
    s = np.array([r["score"] for r in recs], float)
    y = np.array([float(r["y"]) for r in recs])
    target = (c_f / V + h) / (e + h) if (e + h) > 0 else INF

    def s_star(ss: np.ndarray, yy: np.ndarray) -> float:
        if not len(ss) or target == INF:
            return INF
        gx, gy = isotonic_fit(ss, yy, True)
        ok = gy >= target - 1e-12
        return float(gx[np.argmax(ok)]) if ok.any() else INF

    pi = float(y.mean())
    out.update({"pi": pi, "auroc": auroc(s, y), "target_g": target, "s_star": s_star(s, y),
                "frontier": []})
    for tau in taus:
        u = _gate_utility(s, y, tau, V, e, h, c_f)
        r = float((s >= tau).mean())
        u_rand = r * float(np.mean(V * (y * e - (1 - y) * h) - c_f))
        tpr = float((s[y == 1] >= tau).mean()) if (y == 1).any() else NAN
        fpr = float((s[y == 0] >= tau).mean()) if (y == 0).any() else NAN
        out["frontier"].append({"tau": tau, "fire_rate": r, "tpr": tpr, "fpr": fpr, "J": tpr - fpr, "utility": u,
                                "voi": u - u_rand})
    tr, te = split_tasks([{**r, "task_id": r["task_id"]} for r in recs], train_frac, seed)
    if tr and te:
        st = np.array([r["score"] for r in tr], float)
        yt = np.array([float(r["y"]) for r in tr])
        se = np.array([r["score"] for r in te], float)
        ye = np.array([float(r["y"]) for r in te])
        cand = {"s_star": s_star(st, yt), "tau_0.5": 0.5, "tau_0.8": 0.8}
        Ls = [_gate_utility(se, ye, t, V, e, h, c_f) for t in taus]
        best = int(np.argmax(Ls))
        tasks = np.array([r["task_id"] for r in te])
        uniq, tidx = np.unique(tasks, return_inverse=True)
        rng = np.random.default_rng(seed)
        boot = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        out["regret"] = {"n_train": len(tr), "n_test": len(te), "tau_best_test": taus[best], "candidates": {}}
        for name, t in cand.items():
            reg = Ls[best] - _gate_utility(se, ye, t, V, e, h, c_f)
            bs = []
            for b in boot:
                w = np.bincount(b, minlength=len(uniq))[tidx].astype(float)
                gain = V * (ye * e - (1 - ye) * h) - c_f
                bs.append(float(np.sum(w * ((se >= taus[best]).astype(float) - (se >= t).astype(float)) * gain)
                                 / max(w.sum(), 1e-12)))
            out["regret"]["candidates"][name] = {"tau": t, "regret": reg,
                                                 "ci": [float(np.quantile(bs, alpha / 2)),
                                                        float(np.quantile(bs, 1 - alpha / 2))]}
    return out


def default_gate_arms(runs: Mapping[str, Sequence[Mapping[str, Any]]], gate: str,
                      family: str | None = None) -> dict[str, Sequence[Mapping[str, Any]]]:
    """The arms a gate analysis pools by default: MEASUREMENT arms (every decision point off / shadow) with shadow
    records of the gate, of ONE System-2 family (round-3 review: pooling the run directory mixed the measurement arms'
    shadow golds with live trigger arms' selectively labelled records, rate / heuristic control backends and both
    System-2 families). Several families without ``family`` -> SystemExit listing them."""
    from .meta_gold import gate_records as _gr
    from .scorecard import _shadow_only, arm_s2_model

    if gate == "tool_guard":
        raise SystemExit("tool_guard has no shadow records (its measurement is the probe table): pass --arm with the "
                         "live guard arm(s)")
    cand = {a: t for a, t in runs.items() if t and _shadow_only(t) and _gr(list(t), gate, modes=("shadow",))}
    fams: dict[str, list[str]] = {}
    for a, t in cand.items():
        fams.setdefault(str(arm_s2_model(t)), []).append(a)
    if family is not None:
        return {a: cand[a] for a in fams.get(family, [])}
    if len(fams) > 1:
        raise SystemExit(f"measurement arms of several System-2 families have shadow `{gate}` records: "
                         f"{ {f: sorted(v) for f, v in fams.items()} } -- pass --family MODEL or --arm")
    return cand


def analyze_gate_run(run_dir: str | Path, arms: Sequence[str] | None, gate: str, V: float, c_f: float | None = None,
                     e: float | None = None, h: float | None = None, family: str | None = None,
                     cost_arm: str | None = None, **kw: Any) -> dict[str, Any]:
    """Gate analysis of a run directory. Default arms: :func:`default_gate_arms` (shadow records of measurement arms,
    one System-2 family); ``arms`` names arms explicitly (then every logged mode of them is used, e.g. a live guard
    arm). c_f: ``c_f``, else the ledger of ``cost_arm`` -- the live arm whose threshold is analysed (e.g.
    ``reflect_always_s2lite`` / ``jev_reflect_trigger_s2lite``) -- else the analysed arms' own ledger (a measurement
    arm prices a firing from its counterfactual calls of the same prompt, ``meta_gold._fired_cost``)."""
    run_dir = Path(run_dir)
    files = {p.stem: p for p in sorted(run_dir.glob("*.jsonl"))}
    if arms:
        runs = {a: load_trajectories(files[a]) for a in arms if a in files}
        modes = None
    else:
        runs = default_gate_arms({a: load_trajectories(p) for a, p in files.items()}, gate, family)
        modes = ("shadow",)
    recs = [r for a, t in runs.items() for r in gate_records(t, gate, arm=a, modes=modes)]
    if c_f is not None:
        cf, cf_src = c_f, "given"
    elif cost_arm:
        if cost_arm not in files:
            raise SystemExit(f"--cost-arm {cost_arm!r} not in {run_dir}")
        cf, cf_src = gate_cost(load_trajectories(files[cost_arm]), gate), f"ledger of {cost_arm}"
    else:
        cf = float(np.mean([gate_cost(t, gate) for t in runs.values()])) if runs else 0.0
        cf_src = "ledger of the analysed arms"
    branch = {"reflection": "verdict_alt", "routing": "verdict_large", "stop_guard": "verdict_reject"}.get(gate,
                                                                                                         "verdict_alt")
    out = gate_analysis(recs, V, cf, e, h, branch=branch, **kw)
    out["meta"] = {"arms": sorted(runs), "gate": gate, "modes": list(modes) if modes else "all", "c_f_source": cf_src}
    return out


# ====================================================================== driver
def default_taus(step: float = 0.01) -> list[float]:
    """The swept grid: 0, step, ..., 1 and +inf (never accept = defer everything)."""
    return [round(x, 6) for x in np.arange(0.0, 1.0 + step / 2, step)] + [INF]


# PREREGISTRATION §3 H1: regret of tau-hat and of tau = 0.8 against the best swept tau in THIS grid (the live sweep
# arms' thresholds). The CLI's default fine grid (``default_taus``) is a descriptive frontier; run the registered H1
# analysis with ``--taus 0.5 0.7 0.8 0.9 0.95 0.99`` (or ``--taus h1``).
H1_TAUS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def parse_taus(vals: Sequence[str] | None, step: float = 0.01) -> list[float]:
    """CLI ``--taus``: None -> ``default_taus(step)``; ``h1`` -> :data:`H1_TAUS`; else the given floats (``inf`` ok)."""
    if not vals:
        return default_taus(step)
    if len(vals) == 1 and str(vals[0]).lower() == "h1":
        return list(H1_TAUS)
    return [float(v) for v in vals]


def analyze(recs: Sequence[Mapping[str, Any]], Vs: Sequence[float] | None = None, task_value: float = 1.0,
            lam_per_hour: float = 0.0, delta_c: float | None = None, rule: str = "general", n_bins: int = 10,
            train_frac: float = 0.5, seed: int = 0, n_boot: int = 1000, n_perm: int = 2000,
            default_tau: float = 0.8, taus: Sequence[float] | None = None, alpha: float = 0.05) -> dict[str, Any]:
    """Items 1-8 of the module docstring on decision records (:func:`decision_pairs`). Without ``Vs`` the stakes
    are the counterfactual estimate (on the training tasks for tau-hat, on all tasks for the loss)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    taus = list(taus) if taus is not None else default_taus()
    s1 = [r for r in recs if r.get("c") is not None and r.get("s1_ok") is not None]
    ps = paired(recs)
    train, test = split_tasks(recs, train_frac, seed)
    v_all = counterfactual_v(recs, task_value, n_boot, seed, alpha)
    v_train = counterfactual_v(train, task_value, n_boot, seed, alpha)
    out: dict[str, Any] = {
        "n_records": len(recs), "n_s1": len(s1), "n_paired": len(ps),
        "modes": sorted({str(r.get("mode")) for r in recs}),
        "settings": {"task_value": task_value, "lam_per_hour": lam_per_hour, "rule": rule, "n_bins": n_bins,
                     "train_frac": train_frac, "seed": seed, "n_boot": n_boot, "n_perm": n_perm,
                     "default_tau": default_tau, "alpha": alpha, "delta_c_override": delta_c},
        "reliability": reliability([r["c"] for r in s1], [float(r["s1_ok"]) for r in s1], n_bins),
        "q_L": q_l_curve(ps, n_bins, n_perm, seed, alpha),
        "cost": cost_terms(recs, lam_per_hour, delta_c),
        "V_counterfactual": v_all, "V_counterfactual_train": v_train,
        # H1 clause 2 (A6.22 (c)): over-confidence of System-1's paired Choice decisions on the training split
        "overconfidence": overconfidence_gap(train, n_boot, seed, alpha),
    }
    dc = out["cost"]["delta_c"]
    frontier = cascade_frontier(recs, taus, lam_per_hour)
    out["frontier"] = frontier
    if Vs:
        cases = [(float(v), float(v), "user") for v in Vs]
    else:
        cases = [(v_all["V"], v_train["V"], "counterfactual")]
    out["by_V"] = []
    for V, V_plug, src in cases:
        case: dict[str, Any] = {"V": V, "V_source": src, "V_plugin": V_plug}
        if not (V > 0 and V_plug > 0):
            case["note"] = "V is not positive (no wrong-decision episodes, or no effect): tau-hat undefined"
            out["by_V"].append(case)
            continue
        case["thresholds_all"] = plug_in_thresholds(recs, V, dc)
        case["regret"] = regret(train, test, V, V_plug, taus, default_tau, rule, lam_per_hour, delta_c,
                                n_boot, seed, alpha)
        tau_hat = case["regret"].get("thresholds", {}).get(rule, NAN)
        case["voi"] = {"tau_hat": gate_voi(test, tau_hat, V, lam_per_hour, delta_c, n_boot, seed, alpha)
                       if not math.isnan(tau_hat) else {},
                       "tau_default": gate_voi(test, default_tau, V, lam_per_hour, delta_c, n_boot, seed, alpha)}
        case["frontier_loss"] = [{"tau": r["tau"], "loss": float(v)} for r, v in
                                 zip(frontier["rows"], loss_matrix(_arrays(recs, lam_per_hour), taus, V,
                                                                   delta_c).mean(1))] if frontier["n"] else []
        case["h1_clause2"] = h1_clause2(train, test, V, taus, default_tau, lam_per_hour, delta_c, n_boot, seed, alpha,
                                        overconfidence=out["overconfidence"])
        out["by_V"].append(case)
    return out


def analyze_run(run_dir: str | Path, arms: Sequence[str] | None, dp: str, gold: str = "acceptable",
                **kw: Any) -> dict[str, Any]:
    """Load ``RUN_DIR/<arm>.jsonl`` (last record per task), extract the records at ``dp`` and analyze them.
    Default arms: every arm with shadow records at ``dp``."""
    run_dir = Path(run_dir)
    files = {p.stem: p for p in sorted(run_dir.glob("*.jsonl"))}
    runs: dict[str, list[dict[str, Any]]] = {}
    if arms:
        missing = [a for a in arms if a not in files]
        if missing:
            raise FileNotFoundError(f"arm(s) not found in {run_dir}: {missing} (have: {sorted(files)})")
        runs = {a: load_trajectories(files[a]) for a in arms}
    else:
        for a, p in files.items():
            t = load_trajectories(p)
            if any(d.get("dp") == dp and (d.get("shadow") or str(d.get("mode")).startswith("shadow"))
                   for x in t for s in x.get("steps", []) for d in s.get("decisions", [])):
                runs[a] = t
        if not runs:
            raise SystemExit(f"no arm with shadow decisions at {dp!r} in {run_dir}; pass --arm")
    recs = [r for a, t in runs.items() for r in decision_pairs(t, dp, arm=a, gold=gold)]
    out = analyze(recs, **kw)
    if dp in ZERO_DEFER_COST_DPS and out["cost"]["delta_c_source"] == "ledger":
        out["cost"]["delta_c_source"] = f"{dp}: no extra LLM call when deferred, dC = 0"
    try:
        from .scorecard import run_meta
        meta = run_meta(run_dir, runs)
    except Exception:  # noqa: BLE001 - metadata is best effort
        meta = {"run_dir": str(run_dir)}
    out["meta"] = {**meta, "arms": sorted(runs), "dp": dp}
    return out


# ====================================================================== formatting
def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "never" if v > 0 else "-inf"
        if v != 0 and abs(v) < 10 ** -nd:
            return f"{v:.2e}"
        return f"{v:.{nd}f}"
    return str(v)


def _ci(ci: Sequence[float] | None, nd: int = 3) -> str:
    return f"[{_fmt(ci[0], nd)}, {_fmt(ci[1], nd)}]" if ci else "-"


def to_markdown(res: Mapping[str, Any]) -> str:
    m = res.get("meta", {})
    L: list[str] = []
    if m.get("simulated"):
        L.append("> **SIMULATED RUN** -- pipeline test / labelled simulation study only, never headline numbers.\n")
    L.append(f"## Threshold theory: `{m.get('dp', '?')}` in {', '.join(m.get('arms', [])) or '?'}")
    L.append(f"records {res['n_records']} (System-1 scored {res['n_s1']}, paired with the LLM {res['n_paired']}); "
             f"modes {', '.join(res['modes'])}\n")
    rel = res["reliability"]
    if rel.get("n"):
        L.append("### 1. Reliability of System-1 confidence")
        L.append(f"acc {_fmt(rel['acc'])}, mean conf {_fmt(rel['mean_conf'])}, ECE {_fmt(rel['ece'])}, "
                 f"ACE {_fmt(rel['ace'])}, Brier {_fmt(rel['brier'])}, AUROC {_fmt(rel['auroc'])}, "
                 f"AURC {_fmt(rel['aurc'])}\n")
        L.append("| bin | n | conf | acc |\n|---|---|---|---|")
        L += [f"| ({b['lo']:.1f}, {b['hi']:.1f}] | {b['n']} | {_fmt(b['conf'])} | {_fmt(b['acc'])} |"
              for b in rel["bins"]]
        L.append("")
    q = res["q_L"]
    if q.get("n"):
        a2 = q["a2"]
        sl = a2["logit_slope"]
        L.append("### 2. q_L(c): LLM accuracy on the same decision vs. System-1 confidence")
        L.append(f"q_L {_fmt(q['q_L'])} (System-1 {_fmt(q['s1_acc'])}); monotone fit {q['fit']['direction']}; "
                 f"A2 test: logit slope {_fmt(sl['slope'])} {_ci([sl['ci_lo'], sl['ci_hi']])}, "
                 f"permutation r {_fmt(a2['perm']['r'])} (p {_fmt(a2['perm']['p'])}) -> A2 "
                 f"{'**rejected**' if a2['rejected'] else 'not rejected'} at alpha {a2['alpha']}; "
                 f"P(LLM wrong | S1 wrong) {_fmt(q['overlap']['p_llm_wrong_given_s1_wrong'])} vs "
                 f"| S1 right {_fmt(q['overlap']['p_llm_wrong_given_s1_right'])}\n")
        L.append("| bin | n | conf | g (S1 acc) | q_L | q_L fit |\n|---|---|---|---|---|---|")
        L += [f"| ({b['lo']:.1f}, {b['hi']:.1f}] | {b['n']} | {_fmt(b['conf'])} | {_fmt(b['g'])} | {_fmt(b['q_L'])} | "
              f"{_fmt(b['q_L_fit'])} |" for b in q["bins"]]
        L.append("")
    c = res["cost"]
    L.append("### 3. Cost of deferring")
    L.append(f"dC {_fmt(c['delta_c'], 6)} USD/decision ({c['delta_c_source']}; LLM {_fmt(c['llm_usd'], 6)} USD, "
             f"{_fmt(c['llm_latency_ms'], 1)} ms, lambda {c['lam_per_hour']} USD/h); kappa (System-1) "
             f"{_fmt(c['kappa'], 6)} USD/decision ({_fmt(c['s1_latency_ms'], 1)} ms)\n")
    v = res["V_counterfactual"]
    L.append("### 4. Stakes V (counterfactual approximation)")
    L.append(f"W = {v['task_value']}: success {_fmt(v['success_right'])} over {v['n_right']} episodes with every "
             f"executed decision right vs {_fmt(v['success_wrong'])} over {v['n_wrong']} with >= 1 wrong -> V-hat "
             f"{_fmt(v['V'])} {_ci(v['ci'])} (train split: {_fmt(res['V_counterfactual_train']['V'])})\n")
    fr = res["frontier"]
    if fr.get("n"):
        L.append("### Simulated cascade frontier (item 6; all paired decisions)")
        L.append("| tau | acc | delegation | LLM calls | cost (USD) | latency (ms) |\n|---|---|---|---|---|---|")
        show = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, INF}
        for r in fr["rows"]:
            if r["tau"] in show:
                L.append(f"| {_fmt(r['tau'], 2)} | {_fmt(r['acc'])} | {_fmt(r['delegation'])} | "
                         f"{_fmt(r['llm_calls'])} | "
                         f"{_fmt(r['cost_usd'], 6)} | {_fmt(r['latency_ms'], 1)} |")
        ref = fr["references"]
        L.append(f"\nreferences: LLM only acc {_fmt(ref['llm_only']['acc'])}, "
                 f"cost {_fmt(ref['llm_only']['cost_usd'], 6)}; "
                 f"System-1 only acc {_fmt(ref['s1_only']['acc'])}, cost {_fmt(ref['s1_only']['cost_usd'], 6)}\n")
    for case in res["by_V"]:
        L.append(f"### Plug-in threshold, regret and VOI (items 5, 7, 8) at V = {_fmt(case['V'])} ({case['V_source']})")
        if "note" in case:
            L.append(case["note"] + "\n")
            continue
        rg = case["regret"]
        th = rg.get("thresholds", {})
        L.append(f"plug-in tau-hat on {rg.get('n_train', 0)} training decisions (dC/V = {_fmt(th.get('k'))}): "
                 + ", ".join(f"{r} {_fmt(th.get(r))}" for r in RULES)
                 + f" (primary: {rg.get('rule', '')}; "
                 f"threshold-shaped: {_fmt(th.get(rg.get('rule', '') + '_interval'))})")
        if "candidates" in rg:
            L.append(f"\nheld-out ({rg['n_test']} decisions): best swept tau {_fmt(rg['tau_best_test'])} "
                     f"(loss {_fmt(rg['loss_best_test'], 5)} USD/decision)\n")
            L.append("| threshold | tau | loss | regret | regret / V | regret CI |\n|---|---|---|---|---|---|")
            for name, r in rg["candidates"].items():
                L.append(f"| {name} | {_fmt(r['tau'])} | {_fmt(r['loss'], 5)} | {_fmt(r['regret'], 5)} | "
                         f"{_fmt(r['regret_over_V'], 4)} | {_ci(r['regret_ci'], 5)} |")
            d = rg["default_minus_hat"]
            L.append(f"\nregret(0.8) - regret(tau-hat) = {_fmt(d['diff'], 5)} {_ci(d['ci'], 5)}\n")
        c2 = case.get("h1_clause2") or {}
        if c2:
            oc, tm = c2.get("overconfidence") or {}, c2.get("temperatures") or {}

            def _rg(x: Mapping[str, Any]) -> str:
                return f"{_fmt(x.get('regret'), 5)} {_ci(x.get('regret_ci'), 5)} (best swept {_fmt(x.get('tau_best_test'))})"
            L.append(f"H1 clause 2 (A6.22 (c)): over-confidence gap on the training split {_fmt(oc.get('gap'))} "
                     f"(mean conf {_fmt(oc.get('mean_conf'))} - acc {_fmt(oc.get('acc'))}; one-sided 95 % LB "
                     f"{_fmt(oc.get('lb'))}, n {oc.get('n')}) -> over-confident: {_fmt(c2.get('over_confident'))}; "
                     f"held-out regret of tau = {_fmt(c2.get('default_tau'), 2)} vs the best swept tau: raw "
                     f"{_rg(c2.get('raw') or {})}, recalibrated (T {_fmt(tm.get('global'))}, per kind "
                     f"{ {k: round(v, 3) for k, v in (tm.get('per_kind') or {}).items()} }) {_rg(c2.get('recalibrated') or {})}, "
                     f"logged distribution at T = 1 {_rg(c2.get('dist_t1') or {})}; returns to the front: "
                     f"{_fmt(c2.get('returns_to_front'))}\n")
        L.append("| gate (held-out) | tau | accept rate | VOI (USD) | VOI CI | acc gain | acc gain CI |")
        L.append("|---|---|---|---|---|---|---|")
        for name, g in case["voi"].items():
            if g.get("n"):
                L.append(f"| {name} | {_fmt(g['tau'])} | {_fmt(g['accept_rate'])} | {_fmt(g['voi_usd'], 5)} | "
                         f"{_ci(g['voi_ci'], 5)} | {_fmt(g['acc_gain'])} | {_ci(g['acc_gain_ci'])} |")
        L.append("")
    return "\n".join(L)


def to_latex(res: Mapping[str, Any]) -> str:
    m = res.get("meta", {})
    head = "% SIMULATED RUN -- not a headline result\n" if m.get("simulated") else ""
    rows = []
    for case in res["by_V"]:
        rg = case.get("regret", {})
        if "candidates" not in rg:
            continue
        ch, cd = rg["candidates"]["tau_hat"], rg["candidates"]["tau_default"]
        g = case["voi"].get("tau_hat") or {}
        rows.append(f"{_fmt(case['V'], 2)} & {_fmt(ch['tau'], 2)} & {_fmt(rg['tau_best_test'], 2)} & "
                    f"{_fmt(ch['regret_over_V'], 3)} {_ci([x / case['V'] for x in ch['regret_ci']], 3)} & "
                    f"{_fmt(cd['regret_over_V'], 3)} {_ci([x / case['V'] for x in cd['regret_ci']], 3)} & "
                    f"{_fmt(g.get('acc_gain'), 3)} {_ci(g.get('acc_gain_ci'), 3)} \\\\")
    a2 = res["q_L"].get("a2", {}).get("logit_slope", {})
    dp = str(m.get("dp", "")).replace("_", r"\_")
    return (head + "\\begin{tabular}{rrrlll}\n\\toprule\n"
            "$V$ & $\\hat\\tau^\\star$ & $\\tau_{\\text{best}}$ & regret$/V$ ($\\hat\\tau^\\star$) & "
            "regret$/V$ ($\\tau{=}0.8$) & VOI (acc.) \\\\\n\\midrule\n" + "\n".join(rows)
            + f"\n\\bottomrule\n\\end{{tabular}}\n% {dp}: A2 logit slope {_fmt(a2.get('slope'))} "
            f"{_ci([a2.get('ci_lo', NAN), a2.get('ci_hi', NAN)])}\n")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser("jevharness.bench.threshold_theory", description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--arm", action="append", default=None,
                    help="arm(s) to pool (default: arms with shadow logs; --gate: measurement arms of one System-2 "
                         "family)")
    ap.add_argument("--dp", default="tool_select")
    ap.add_argument("--V", type=float, nargs="+", default=None, help="stakes of one decision in USD (several ok)")
    ap.add_argument("--task-value", type=float, default=1.0, help="W: USD value of one task success (for V-hat)")
    ap.add_argument("--lam-per-hour", type=float, default=0.0, help="price of critical-path latency, USD/hour")
    ap.add_argument("--delta-c", type=float, default=None, help="override the ledger estimate of dC (USD)")
    ap.add_argument("--rule", choices=RULES, default="general")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--default-tau", type=float, default=0.8)
    ap.add_argument("--tau-step", type=float, default=0.01)
    ap.add_argument("--taus", nargs="+", default=None,
                    help="swept thresholds for the frontier / regret (default: 0..1 by --tau-step, and inf); 'h1' = the "
                         "registered H1 grid 0.5 0.7 0.8 0.9 0.95 0.99 (PREREGISTRATION §3)")
    ap.add_argument("--format", choices=["md", "json", "latex"], default="md")
    ap.add_argument("--out", default=None, help="also write the full result as JSON to this file")
    ap.add_argument("--gold", choices=["acceptable", "canonical"], default="acceptable",
                    help="tool_select gold: the acceptable tool set AT(s) (default) or the canonical next tool")
    ap.add_argument("--gate", default=None,
                    choices=["reflection", "routing", "stop_guard", "stop_trigger", "memory_write", "tool_guard"],
                    help="gate analysis instead of the cascade analysis: plug-in rule g^-1((c_f/V + h)/(e + h)), "
                         "simulated gate frontier, VoI vs. the matched random gate, held-out regret")
    ap.add_argument("--c-f", type=float, default=None, help="gate: cost of one firing in USD (default: the ledger)")
    ap.add_argument("--e", type=float, default=None, help="gate: efficacy at a needed state (default: CF branches)")
    ap.add_argument("--h", type=float, default=None, help="gate: harm at a non-needed state (default: CF branches)")
    ap.add_argument("--family", default=None,
                    help="gate: System-2 model of the default measurement arms (needed when several families ran)")
    ap.add_argument("--cost-arm", default=None,
                    help="gate: the live arm whose firing cost c_f is read from the ledger (default: the analysed arms)")
    ap.add_argument("--calibration-arm", action="store_true",
                    help="A6 / P0-6: fit tau-hat* from the calibration shadow arm(s) (--arm, seeds 5000-5149) and "
                         "print the scalar-threshold projection at W = --task-value, lambda = --lam-per-hour with "
                         "the W / lambda sensitivity rows")
    ap.add_argument("--w-grid", type=float, nargs="+", default=[0.1, 10.0], help="calibration arm: W sensitivity rows")
    ap.add_argument("--lam-grid", type=float, nargs="+", default=[0.5, 5.0],
                    help="calibration arm: lambda ($/h) sensitivity rows")
    ap.add_argument("--min-bin", type=int, default=20, help="calibration arm: minimum decisions per bin")
    ap.add_argument("--v-fallback", default="W",
                    help="calibration arm: V when the counterfactual estimate is undefined (A6.11): 'W' (default), "
                         "'fatal' (W x P(fatal | System-1 wrong) over the strict-JevGym fatal tools), 'none' (tau-hat "
                         "stays undefined) or a number (V at W = --task-value, scaled with W)")
    a = ap.parse_args(argv)
    if a.calibration_arm:
        if not a.arm:
            raise SystemExit("--calibration-arm needs --arm (the calibration shadow arm, e.g. jev_shadow_cal)")
        vf: str | float | None = a.v_fallback
        if isinstance(vf, str):
            vf = (None if vf.lower() == "none" else "W" if vf.upper() == "W"
                  else "fatal" if vf.lower() == "fatal" else float(vf))
        res = calibration_arm_run(a.run_dir, a.arm, dp=a.dp, gold=a.gold, task_value=a.task_value,
                                  lam_per_hour=a.lam_per_hour, w_grid=a.w_grid, lam_grid=a.lam_grid, n_bins=a.n_bins,
                                  min_count=a.min_bin, n_boot=a.n_boot, seed=a.seed, delta_c=a.delta_c, v_fallback=vf)
        res = _jsonable(res)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str) if a.format == "json" else calibration_markdown(res))
        return
    if a.gate:
        res = {"by_V": [analyze_gate_run(a.run_dir, a.arm, a.gate, V, a.c_f, a.e, a.h, family=a.family,
                                         cost_arm=a.cost_arm, train_frac=a.train_frac, seed=a.seed, n_boot=a.n_boot)
                        for V in (a.V or [1.0])]}
        print(json.dumps(_jsonable(res), indent=1, default=str))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(_jsonable(res), indent=1, default=str))
        return
    res = analyze_run(a.run_dir, a.arm, a.dp, gold=a.gold, Vs=a.V, task_value=a.task_value,
                      lam_per_hour=a.lam_per_hour, delta_c=a.delta_c, rule=a.rule, n_bins=a.n_bins,
                      train_frac=a.train_frac, seed=a.seed, n_boot=a.n_boot, n_perm=a.n_perm,
                      default_tau=a.default_tau, taus=parse_taus(a.taus, a.tau_step))
    res = _jsonable(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    if a.format == "json":
        print(json.dumps(res, indent=1, default=str))
    elif a.format == "latex":
        print(to_latex(res))
    else:
        print(to_markdown(res))


if __name__ == "__main__":
    main(sys.argv[1:])
