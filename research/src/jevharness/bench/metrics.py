"""Metrics for JevAgentBench.

Episode level     success rate, mean score, cost/episode, latency/episode, #LLM calls, steps
Efficiency        cost-normalised success, success per dollar, Pareto front over (cost, success),
                  cost-/latency-of-pass (Erol et al. 2025), effectiveness under a budget (SWE-Effi)
Reliability       pass^k over repeated trials (tau-bench; Yao et al. 2025), pass@k (Chen et al. 2021)
Routing-style     PGR / APGR / CPT(x%) (RouteLLM; Ong et al. 2025), AIQ over the non-decreasing convex
                  hull (RouterBench; Hu et al. 2024)
Multi-objective   dominance counts, 2-D hypervolume and exclusive contributions (Zitzler & Thiele 1998/99),
                  iso-quality cost savings, quality at iso-cost (cost-controlled comparison; Kapoor et al.)
Decision level    accuracy vs. gold, deferral rate, accuracy-when-accepted (selective accuracy),
                  delegation ratio, harmful-delegation rate, guard FAR/FRR, rescue rate
Calibration       ECE (Naeini et al. 2015; Guo et al. 2017), adaptive ECE (Nixon et al. 2019), Brier
                  score (Brier 1950), AUROC, risk-coverage curve, AURC / E-AURC and selective accuracy at
                  a coverage (Geifman & El-Yaniv 2017; Geifman et al. 2019)
Statistics        bootstrap CIs, paired bootstrap tests (Efron & Tibshirani 1993; Koehn 2004;
                  Berg-Kirkpatrick et al. 2012), McNemar's test (McNemar 1947; Dietterich 1998),
                  Holm (1979) and Benjamini-Hochberg (1995) corrections (see Dror et al. 2018)

Exact definitions, the statistical protocol and full references: ``docs/BENCHMARK_SPEC.md``. The
per-arm score card built from these functions lives in ``jevharness.bench.scorecard``.
"""

from __future__ import annotations

import functools
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

NAN = float("nan")


# ------------------------------------------------------------------ episode level
def episode_table(trajs: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per-configuration episode means: success (with a percentile-bootstrap 95% CI over episodes), cost
    (total / System-1 / System-2), critical-path latency (sum of call latencies), LLM calls and input
    tokens, steps, errors, success per dollar, cost-of-pass and latency-of-pass. Success is always reported
    together with cost and latency (Kapoor et al., 2025, "AI Agents That Matter")."""
    n = len(trajs)
    if n == 0:
        return {}
    succ = [float(bool(t["result"].get("success"))) for t in trajs]
    cost = [t["usage"]["total"]["cost_usd"] for t in trajs]
    s1cost = [t["usage"].get("s1", {}).get("cost_usd", 0.0) for t in trajs]
    s2cost = [t["usage"].get("s2", {}).get("cost_usd", 0.0) for t in trajs]
    lat = [t["usage"]["total"]["latency_ms"] for t in trajs]
    s2calls = [t["usage"].get("s2", {}).get("calls", 0) for t in trajs]
    s1calls = [t["usage"].get("s1", {}).get("calls", 0) for t in trajs]
    s2in = [t["usage"].get("s2", {}).get("input_tokens", 0) for t in trajs]
    steps = [len(t["steps"]) for t in trajs]
    errors = sum(1 for t in trajs if t.get("error"))
    lo, hi = bootstrap_ci(succ)
    return {
        "n": n,
        "success": float(np.mean(succ)), "success_ci_lo": lo, "success_ci_hi": hi,
        "cost_usd": float(np.mean(cost)), "s1_cost_usd": float(np.mean(s1cost)), "s2_cost_usd": float(np.mean(s2cost)),
        "latency_s": float(np.mean(lat)) / 1000, "llm_calls": float(np.mean(s2calls)), "s1_calls": float(np.mean(s1calls)),
        "llm_input_tokens": float(np.mean(s2in)), "steps": float(np.mean(steps)), "errors": errors,
        "success_per_dollar": float(np.sum(succ) / max(np.sum(cost), 1e-9)),
        "cost_of_pass": cost_of_pass(cost, succ), "latency_of_pass": latency_of_pass([x / 1000 for x in lat], succ),
    }


def pareto_front(points: Sequence[tuple[str, float, float]]) -> list[str]:
    """points: (name, cost, success). Returns the names on the cost-success Pareto front (no other
    configuration is at most as expensive and at least as successful, strictly better in one) -- the
    joint cost-accuracy view of Kapoor et al. (2025, "AI Agents That Matter", arXiv:2407.01502)."""
    front = []
    for name, c, s in points:
        dominated = any((c2 <= c and s2 >= s) and (c2 < c or s2 > s) for _, c2, s2 in points)
        if not dominated:
            front.append(name)
    return front


# ------------------------------------------------------------------ cost- and latency-normalised success
def cost_of_pass(costs: Sequence[float], successes: Sequence[float]) -> float:
    """Cost-of-pass (Erol et al., 2025, arXiv:2504.13359): the expected monetary cost of obtaining one
    *correct* outcome, ``E[cost] / E[success]`` (USD per success; ``inf`` if nothing succeeds).

    Erol et al. define it per problem as ``C_m(p) / R_m(p)`` (expected cost of an attempt over its
    success probability); over a set of episodes the ratio of means is the plug-in estimator of the
    dataset-level quantity, as used for agents by Wang et al. (2025, "Efficient Agents"). For the
    per-task estimator and the frontier over configurations see :func:`frontier_cost_of_pass`."""
    c, s = np.asarray(costs, float), np.asarray(successes, float)
    if len(c) == 0:
        return float("nan")
    rate = float(s.mean())
    return float(c.mean()) / rate if rate > 0 else float("inf")


def latency_of_pass(latencies: Sequence[float], successes: Sequence[float]) -> float:
    """Latency-of-pass: the cost-of-pass construction (Erol et al., 2025) with wall-clock / critical-path
    latency in place of dollars, ``E[latency] / E[success]`` -- expected seconds per *successful* episode
    (``inf`` if nothing succeeds). Reported next to cost-of-pass because System-1 models mainly buy time."""
    return cost_of_pass(latencies, successes)


def frontier_cost_of_pass(per_arm: Mapping[str, Mapping[str, Sequence[tuple[float, float]]]],
                          baseline_cop: float | None = None) -> dict[str, Any]:
    """Frontier cost-of-pass (Erol et al., 2025): for every task the minimum cost-of-pass over the
    available configurations (optionally including a human-expert cost ``baseline_cop``), averaged over
    tasks -- exactly the estimator of the authors' reference implementation (``FrontierCostofPass``).

    per_arm: arm -> task -> [(cost, success), ...] (one entry per trial).
    Returns ``{"value", "per_task", "argmin_counts", "n_unsolved"}``; ``value`` is ``inf`` when some task is
    solved by no configuration (and no baseline is given)."""
    per_task: dict[str, dict[str, float]] = defaultdict(dict)
    for arm, tasks in per_arm.items():
        for task, trials in tasks.items():
            if trials:
                per_task[task][arm] = cost_of_pass([c for c, _ in trials], [s for _, s in trials])
    if not per_task:
        return {"value": float("nan"), "per_task": {}, "argmin_counts": {}, "n_unsolved": 0}
    frontier: dict[str, float] = {}
    counts: dict[str, int] = defaultdict(int)
    for task, cops in per_task.items():
        if baseline_cop is not None:
            cops = {**cops, "baseline": float(baseline_cop)}
        arm = min(cops, key=lambda a: cops[a])
        frontier[task] = cops[arm]
        if math.isfinite(cops[arm]):
            counts[arm] += 1
    n_unsolved = sum(1 for v in frontier.values() if not math.isfinite(v))
    value = float("inf") if n_unsolved else float(np.mean(list(frontier.values())))
    return {"value": value, "per_task": frontier, "argmin_counts": dict(counts), "n_unsolved": n_unsolved}


def effectiveness_under_budget(resources: Sequence[float], successes: Sequence[float], budget: float) -> float:
    """Effectiveness under a budget (SWE-Effi; Fan et al., 2025, arXiv:2509.09853): the normalised area
    under the "success within budget b" curve for b in [0, B],

        R(b) = (1/N) sum_i y_i 1[r_i <= b],      EuB = (1/B) int_0^B R(b) db = (1/(N B)) sum_i y_i max(0, B - r_i)

    where r_i is the resource (USD, tokens, seconds) episode i consumed. Cheap successes count almost
    fully, successes that need close to the whole budget count little, anything over budget counts zero
    (so "expensive failures" and expensive successes are both penalised)."""
    r, y = np.asarray(resources, float), np.asarray(successes, float)
    if len(r) == 0 or not budget > 0:
        return float("nan")
    return float(np.sum(y * np.clip(budget - r, 0.0, None)) / (len(r) * budget))


# ------------------------------------------------------------------ reliability over repeated trials
def pass_hat_k(n: int, c: int, k: int) -> float:
    """tau-bench pass^k for one task (Yao et al., 2025, arXiv:2406.12045): the probability that *all* k
    i.i.d. trials succeed, estimated without bias from n trials with c successes as C(c, k) / C(n, k).
    Averaged over tasks it measures consistency; pass^1 is the success rate."""
    if not 0 < k <= n or not 0 <= c <= n:
        raise ValueError(f"need 0 < k <= n and 0 <= c <= n (got n={n}, c={c}, k={k})")
    return math.comb(c, k) / math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k for one task (Chen et al., 2021, arXiv:2107.03374): the probability that *at least one* of k
    samples succeeds, unbiased estimate 1 - C(n - c, k) / C(n, k). Shown next to pass^k for contrast."""
    if not 0 < k <= n or not 0 <= c <= n:
        raise ValueError(f"need 0 < k <= n and 0 <= c <= n (got n={n}, c={c}, k={k})")
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k_curve(trials: Mapping[str, Sequence[float | bool]], ks: Iterable[int] | None = None,
                     estimator: str = "hat") -> dict[int, float]:
    """Task-averaged pass^k (``estimator="hat"``) or pass@k (``"at"``) from task -> per-trial outcomes.
    ``ks`` defaults to 1..min #trials; values of k above a task's #trials are skipped."""
    per = {t: [bool(x) for x in v] for t, v in trials.items() if len(v)}
    if not per:
        return {}
    n_min = min(len(v) for v in per.values())
    fn = pass_hat_k if estimator == "hat" else pass_at_k
    return {k: float(np.mean([fn(len(v), sum(v), k) for v in per.values()]))
            for k in (ks or range(1, n_min + 1)) if 0 < k <= n_min}


# ------------------------------------------------------------------ routing-style metrics
def _trapezoid(xs: Sequence[float], ys: Sequence[float]) -> float:
    return float(sum((x1 - x0) * (y0 + y1) / 2 for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:])))


def pgr(quality: float, weak: float, strong: float) -> float:
    """Performance gap recovered (RouteLLM; Ong et al., 2025, arXiv:2406.18665):
    PGR = (q - q_weak) / (q_strong - q_weak). Here "strong" is the LLM-only harness and "weak" the
    System-1-only configuration (every delegable decision taken by System-1). nan if the gap is 0."""
    gap = strong - weak
    return (quality - weak) / gap if abs(gap) > 1e-12 else float("nan")


def _pgr_curve(points: Sequence[tuple[float, float]], weak: float, strong: float, anchors: bool,
               weak_x: float, strong_x: float) -> list[tuple[float, float]]:
    pts = [(float(x), pgr(q, weak, strong)) for x, q in points]
    if anchors:
        pts += [(float(weak_x), 0.0), (float(strong_x), 1.0)]
    return sorted(p for p in pts if not math.isnan(p[1]))


def apgr(points: Sequence[tuple[float, float]], weak: float, strong: float, anchors: bool = True,
         weak_x: float = 0.0, strong_x: float = 1.0) -> float:
    """Average performance gap recovered (RouteLLM; Ong et al., 2025): the area under PGR as a function of
    the fraction x of calls that go to the strong model, APGR = int PGR(x) dx over the swept range,
    normalised by its width. Trapezoid rule over the configurations sorted by x, exactly as in the
    authors' reference implementation (``routellm/evals/evaluate.py::apgr_metric``).

    points: (strong-call fraction in [0, 1], quality) of one family of configurations (e.g. a cascade
    tau sweep). ``anchors`` adds the weak reference at (weak_x, PGR=0) and the strong one at (strong_x, 1),
    so a family that is no better than interpolating the two references scores 0.5."""
    pts = _pgr_curve(points, weak, strong, anchors, weak_x, strong_x)
    if len(pts) < 2 or pts[-1][0] - pts[0][0] <= 0:
        return float("nan")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return _trapezoid(xs, ys) / (xs[-1] - xs[0])


def cpt(points: Sequence[tuple[float, float]], weak: float, strong: float, target: float = 0.5,
        anchors: bool = True, weak_x: float = 0.0, strong_x: float = 1.0) -> float:
    """Call-performance threshold CPT(x%) (RouteLLM; Ong et al., 2025): the minimum fraction of strong-model
    calls needed to reach a PGR of ``target`` (0.5 = CPT(50%), 0.8 = CPT(80%)), read off the piecewise-linear
    PGR curve (see :func:`apgr`). nan if the family never reaches the target."""
    pts = _pgr_curve(points, weak, strong, anchors, weak_x, strong_x)
    if not pts:
        return float("nan")
    if pts[0][1] >= target:
        return pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 < target <= y1:
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return float("nan")


def nondecreasing_convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-decreasing convex hull of (cost, quality) points (RouterBench; Hu et al., 2024,
    arXiv:2403.12031): the upper concave envelope -- every point on a segment between two
    configurations is achievable by randomly mixing them per episode -- truncated at the quality maximum,
    so that a larger budget never lowers quality. Returned sorted by cost."""
    best: dict[float, float] = {}
    for x, y in points:
        x, y = float(x), float(y)
        if math.isnan(x) or math.isnan(y):
            continue
        best[x] = max(y, best.get(x, -math.inf))
    hull: list[tuple[float, float]] = []
    for p in sorted(best.items()):
        while len(hull) >= 2:
            (ox, oy), (ax, ay) = hull[-2], hull[-1]
            if (ax - ox) * (p[1] - oy) - (ay - oy) * (p[0] - ox) >= 0:  # not a right turn -> not on upper hull
                hull.pop()
            else:
                break
        hull.append(p)
    if not hull:
        return []
    top = max(range(len(hull)), key=lambda i: (hull[i][1], -i))
    return hull[: top + 1]


def _hull_value(hull: Sequence[tuple[float, float]], x: float, floor: float) -> float:
    if x < hull[0][0]:
        return floor
    if x >= hull[-1][0]:
        return hull[-1][1]
    for (x0, y0), (x1, y1) in zip(hull[:-1], hull[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else max(y0, y1)
    return hull[-1][1]


def _log10c(c: float, floor: float = 1e-6) -> float:
    return math.log10(max(float(c), floor))


def aiq(points: Sequence[tuple[float, float]], x_range: tuple[float, float] | None = None, log_x: bool = False,
        floor: float = 0.0) -> float:
    """Average improvement in quality (RouterBench; Hu et al., 2024): the normalised area under a family's
    non-decreasing convex hull Q~(c) over a cost range [c_lo, c_hi],

        AIQ = 1 / (c_hi - c_lo) * int_{c_lo}^{c_hi} Q~(c) dc,

    with Q~ extended flat beyond the most expensive configuration (as RouterBench's
    ``evaluation/AIQ.py`` does) and equal to ``floor`` below the cheapest one (that budget is infeasible
    for the family). Families must share ``x_range`` to be comparable. ``log_x`` integrates over log10
    cost -- easier to read when costs span orders of magnitude, but randomised mixing is linear in cost,
    so the hull is only exactly achievable with ``log_x=False`` (the paper's convention)."""
    tf = _log10c if log_x else float
    pts = [(tf(x), float(y)) for x, y in points]
    hull = nondecreasing_convex_hull(pts)
    if not hull:
        return float("nan")
    lo, hi = (tf(x_range[0]), tf(x_range[1])) if x_range else (min(p[0] for p in pts), max(p[0] for p in pts))
    if hi <= lo:
        return float(hull[-1][1]) if lo >= hull[0][0] else float(floor)
    xs = sorted({lo, hi} | {x for x, _ in hull if lo < x < hi})
    area = 0.0
    for a, b in zip(xs[:-1], xs[1:]):
        if b <= hull[0][0]:
            area += floor * (b - a)
        else:
            area += (b - a) * (_hull_value(hull, a, floor) + _hull_value(hull, b, floor)) / 2
    return area / (hi - lo)


# ------------------------------------------------------------------ Pareto / multi-objective
def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """(cost, success): a dominates b iff it is no more expensive, no less successful, and strictly better
    in at least one of the two."""
    return a[0] <= b[0] and a[1] >= b[1] and (a[0] < b[0] or a[1] > b[1])


def dominance_counts(points: Mapping[str, tuple[float, float]]) -> dict[str, dict[str, int]]:
    """For every configuration: how many others it dominates (the "strength" of SPEA; Zitzler & Thiele,
    1999) and by how many it is dominated (0 = on the Pareto front)."""
    return {n: {"dominates": sum(dominates(p, q) for m, q in points.items() if m != n),
                "dominated_by": sum(dominates(q, p) for m, q in points.items() if m != n)}
            for n, p in points.items()}


def hypervolume_2d(points: Sequence[tuple[float, float]], ref: tuple[float, float], log_cost: bool = False,
                   normalize: bool = False) -> float:
    """Hypervolume indicator (Zitzler & Thiele, 1998; 1999) for (cost -> min, success -> max): the area of
    the region dominated by the point set and bounded by the reference point ``ref = (cost_ref,
    success_ref)``. Points that are not strictly better than ``ref`` in both objectives add nothing; the
    indicator is strictly monotone in Pareto dominance. ``log_cost`` measures the cost axis in log10(USD)
    (the scale of our Pareto plots); ``normalize`` divides by the area of the box between ``ref`` and the
    ideal point (cheapest cost among the points, success 1)."""
    tf = _log10c if log_cost else float
    rc, rs = tf(ref[0]), float(ref[1])
    pts = sorted((tf(c), float(s)) for c, s in points if tf(c) < rc and s > rs)
    area, best = 0.0, rs
    for i, (c, s) in enumerate(pts):
        best = max(best, s)
        nxt = pts[i + 1][0] if i + 1 < len(pts) else rc
        area += (nxt - c) * (best - rs)
    if normalize:
        if not pts:
            return 0.0
        box = (rc - pts[0][0]) * (1.0 - rs)
        return area / box if box > 0 else float("nan")
    return area


def hypervolume_contributions(points: Mapping[str, tuple[float, float]], ref: tuple[float, float],
                              log_cost: bool = False) -> dict[str, float]:
    """Exclusive hypervolume contribution of each configuration, HV(P) - HV(P minus {p}): how much of the
    attainable cost-success region is lost if that configuration is removed (0 for dominated ones; clipped at
    0 against floating-point noise)."""
    total = hypervolume_2d(list(points.values()), ref, log_cost)
    return {n: max(0.0, total - hypervolume_2d([q for m, q in points.items() if m != n], ref, log_cost))
            for n in points}


# ------------------------------------------------------------------ statistics
def bootstrap_ci(xs: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile-bootstrap (1 - alpha) CI of the mean (Efron & Tibshirani, 1993), ``n`` resamples."""
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, float)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n: int = 5000, seed: int = 0) -> dict[str, float]:
    """Legacy paired bootstrap (Koehn, 2004). a, b: task_id -> score. Returns the mean difference (b - a) and
    the one-sided bootstrap p-value P*(mean diff <= 0) (small = b better). Prefer :func:`paired_bootstrap_diff`
    (vectorised, with CIs and both one-sided p-values)."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "diff": float("nan"), "p": float("nan")}
    d = np.array([b[k] - a[k] for k in keys])
    rng = random.Random(seed)
    worse = 0
    for _ in range(n):
        s = sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d)
        worse += s <= 0
    return {"n": len(keys), "diff": float(d.mean()), "p": worse / n}


def paired_bootstrap_diff(base: Mapping[str, float], other: Mapping[str, float], n: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> dict[str, Any]:
    """Paired bootstrap over tasks (Efron & Tibshirani, 1993; Koehn, 2004; Berg-Kirkpatrick et al., 2012)
    for Delta = mean_t(other_t - base_t) on the tasks both configurations ran (task = resampling unit, so
    several trials per task are handled as a cluster when the inputs are per-task means).

    Returns ``diff``, a two-sided (1 - alpha) percentile CI (``ci_lo``, ``ci_hi``), the one-sided lower
    bound ``lcb`` (alpha quantile; non-inferiority at margin m holds iff lcb > -m), ``p_worse`` = P*(Delta* >= 0)
    -- a small value is evidence that ``other`` is *worse* than ``base`` -- and ``p_better`` =
    P*(Delta* <= 0), the one-sided p-value of :func:`paired_bootstrap`. Vectorised (numpy)."""
    keys, d, boot = _paired_boot(base, other, n, seed)
    if not keys:
        nan = float("nan")
        return {"n": 0, "diff": nan, "ci_lo": nan, "ci_hi": nan, "lcb": nan, "p_worse": nan, "p_better": nan}
    return {"n": len(keys), "diff": float(d.mean()),
            "ci_lo": float(np.quantile(boot, alpha / 2)), "ci_hi": float(np.quantile(boot, 1 - alpha / 2)),
            "lcb": float(np.quantile(boot, alpha)),
            "p_worse": float(np.mean(boot >= 0)), "p_better": float(np.mean(boot <= 0))}


def _paired_boot(base: Mapping[str, float], other: Mapping[str, float], n: int,
                 seed: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The resampling behind :func:`paired_bootstrap_diff` and :func:`ni_pvalue` (one place, so the registered
    NI test and the score card's Delta rows share it exactly): (common task keys, per-task differences other - base,
    the ``n`` bootstrap means of Delta* over tasks resampled with replacement, ``numpy.random.default_rng(seed)``)."""
    keys = sorted(set(base) & set(other))
    if not keys:
        return [], np.zeros(0), np.zeros(0)
    d = np.array([float(other[k]) - float(base[k]) for k in keys])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return keys, d, boot


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P[X <= k] for X ~ Bin(n, p), summed in log space (log-gamma binomial coefficients, so n in the thousands does
    not overflow a float the way math.comb(n, k) * p**k does); exact to floating-point rounding."""
    k, n = int(k), int(n)
    if n <= 0 or k >= n:
        return 1.0
    if k < 0:
        return 0.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    i = np.arange(0, k + 1)
    lf = np.array([math.lgamma(x + 1) for x in range(0, n + 1)], float)      # lf[x] = log x!
    lp = lf[n] - lf[i] - lf[n - i] + i * math.log(p) + (n - i) * math.log1p(-p)
    return float(min(1.0, np.exp(lp).sum()))


def mcnemar(a: Mapping[str, float | bool], b: Mapping[str, float | bool], exact: bool | None = None) -> dict[str, Any]:
    """McNemar's (1947) test for paired binary outcomes of two configurations on the same episodes (keys),
    the test Dietterich (1998) recommends for comparing two classifiers on one test set.

    n01 = #(a succeeds, b fails), n10 = #(a fails, b succeeds). With ``exact`` (default when
    n01 + n10 < 25) the p-values are exact binomial (Bin(n01 + n10, 1/2)); otherwise the chi-square
    statistic with continuity correction (|n01 - n10| - 1)^2 / (n01 + n10) with 1 d.o.f. is used.
    Returns ``p`` (two-sided), ``p_b_worse`` (one-sided: b fails more often than a) and ``p_b_better``."""
    keys = sorted(set(a) & set(b))
    n01 = sum(1 for k in keys if bool(a[k]) and not bool(b[k]))
    n10 = sum(1 for k in keys if not bool(a[k]) and bool(b[k]))
    m = n01 + n10
    out: dict[str, Any] = {"n": len(keys), "n01": n01, "n10": n10}
    if m == 0:
        return {**out, "stat": 0.0, "p": 1.0, "p_b_worse": 1.0, "p_b_better": 1.0, "exact": True}
    if exact is None:
        exact = m < 25
    if exact:
        p_worse = 1.0 - _binom_cdf(n01 - 1, m) if n01 > 0 else 1.0     # P(X >= n01)
        p_better = 1.0 - _binom_cdf(n10 - 1, m) if n10 > 0 else 1.0    # P(X >= n10)
        p = min(1.0, 2.0 * _binom_cdf(min(n01, n10), m))
        return {**out, "stat": float(min(n01, n10)), "p": p, "p_b_worse": min(1.0, p_worse),
                "p_b_better": min(1.0, p_better), "exact": True}
    stat = (abs(n01 - n10) - 1) ** 2 / m
    p = math.erfc(math.sqrt(stat / 2))                              # chi2(1) survival function
    z = (n01 - n10) / math.sqrt(m)                                  # normal approximation, one-sided
    return {**out, "stat": stat, "p": p, "p_b_worse": 0.5 * math.erfc(z / math.sqrt(2)),
            "p_b_better": 0.5 * math.erfc(-z / math.sqrt(2)), "exact": False}


def holm(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm (1979) step-down adjusted p-values, controlling the family-wise error rate over the m tests
    (e.g. every configuration vs. the LLM-only baseline). nan p-values are passed through."""
    valid = sorted(((p, k) for k, p in pvals.items() if p is not None and not math.isnan(p)))
    m, out, running = len(valid), {k: float("nan") for k in pvals}, 0.0
    for i, (p, k) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def benjamini_hochberg(pvals: Mapping[str, float]) -> dict[str, float]:
    """Benjamini & Hochberg (1995) adjusted p-values (false-discovery-rate control), for exploratory
    families of tests such as threshold sweeps. nan p-values are passed through."""
    valid = sorted(((p, k) for k, p in pvals.items() if p is not None and not math.isnan(p)))
    m, out, running = len(valid), {k: float("nan") for k in pvals}, 1.0
    for i in range(m - 1, -1, -1):
        p, k = valid[i]
        running = min(running, m * p / (i + 1))
        out[k] = min(1.0, running)
    return out


# ------------------------------------------------------------------ non-inferiority procedure (A6 / P0-1; docs/METRICS_A6.md)
def _z(p: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ni_exact_check(wins: int, losses: int, n: int, margin: float, alpha: float = 0.05) -> dict[str, Any]:
    """Exact one-sided binomial check of non-inferiority on ``n`` paired tasks with few discordant ones (A6 / P0-1;
    reported next to the bootstrap bound when fewer than 5 tasks are discordant, where the task bootstrap is near
    degenerate). H0: the arm's per-task loss rate is >= the margin m; under H0 the number of losses L ~ Bin(n, m) at the
    boundary, so ``p`` = P[L <= losses | n, m] (with zero discordant tasks and m = 0.03, n = 500: 0.97^500 = 2.4e-7, so NI
    holds). ``loss_rate_ub`` is the exact one-sided (1 - alpha) Clopper-Pearson upper bound on the loss rate, which for
    zero losses is 1 - alpha^(1/n) (0.6 pt at n = 500, alpha = 0.05) -- what the paper reports in place of a zero-width
    bootstrap interval. ``ni`` = p < alpha. Wins are reported only (NI asks about losses)."""
    if n <= 0:
        return {"n": n, "wins": wins, "losses": losses, "margin": margin, "alpha": alpha, "p": float("nan"),
                "loss_rate_ub": float("nan"), "ni": False, "n_discordant": wins + losses}
    pv = _binom_cdf(losses, n, margin)
    if losses == 0:
        ub = 1.0 - alpha ** (1.0 / n)
    else:
        lo, hi = losses / n, 1.0                      # smallest q with P[L <= losses | q] <= alpha
        for _ in range(100):
            mid = (lo + hi) / 2
            if _binom_cdf(losses, n, mid) > alpha:
                lo = mid
            else:
                hi = mid
        ub = hi
    return {"n": n, "wins": wins, "losses": losses, "n_discordant": wins + losses, "margin": margin, "alpha": alpha,
            "p": pv, "loss_rate_ub": ub, "ni": bool(pv < alpha)}


def ni_sample_size(discordance: float, margin: float, alpha: float = 0.05, delta: float = 0.0,
                   power: float = 0.8) -> float:
    """Tasks needed for ``power`` in the one-sided paired NI test H0: Delta <= -m at a true Delta = ``delta`` with
    discordance d = P(win) + P(loss) (Wald / score approximation of results/reviews/2026-10-01b/ni_power.py):

        n = (z_{1-alpha} sigma_0 + z_{power} sigma_1)^2 / (m + delta)^2,  sigma_0^2 = d - m^2,  sigma_1^2 = d - delta^2.

    3-pt margin at 5 % discordance and Delta = 0: 339 tasks at alpha = 0.05, 611 at 0.05/8, 652 at the 11-test Holm
    floor 0.05/11 (REVIEW_ROUND5 §5a)."""
    s0, s1 = math.sqrt(max(discordance - margin ** 2, 0.0)), math.sqrt(max(discordance - delta ** 2, 0.0))
    return (_z(1 - alpha) * s0 + _z(power) * s1) ** 2 / (margin + delta) ** 2


def _power_wald(n: float, margin: float, discordance: float, delta: float, alpha: float) -> float:
    s0, s1 = math.sqrt(max(discordance - margin ** 2, 1e-18)), math.sqrt(max(discordance - delta ** 2, 1e-18))
    return _phi(((margin + delta) * math.sqrt(n) - _z(1 - alpha) * s0) / s1)


def ni_power(n: int, discordance: float, margin: float, alpha: float = 0.05, delta: float = 0.0, sims: int = 0,
             n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Power of the registered NI rule at ``n`` paired tasks (A6 / P0-2). Analytic (``power``): the Wald / score
    approximation Phi(((m + delta) sqrt(n) - z_{1-alpha} sigma_0) / sigma_1) of ni_power.py (3 pt, d = 5 %, Delta = 0:
    0.91 at n = 500 and alpha = 0.05, 0.66 at 0.05/11, 0.50 at n = 149). With ``sims`` > 0 the task-bootstrap rule
    itself is simulated (``n_boot`` resamples per replication; outcomes multinomial over win / loss / tie with
    P(win) = (d + Delta) / 2, P(loss) = (d - Delta) / 2): ``type1_boot`` is its rejection rate at the boundary
    Delta = -m and ``power_boot`` at ``delta``; ``type1_score`` / ``power_score`` are the Wald test with the H0 variance
    for comparison (the review's simulation: size 0.051-0.060 at alpha = 0.05 and 0.007-0.013 at 0.05/11, n = 500)."""
    out: dict[str, Any] = {"n": n, "discordance": discordance, "margin": margin, "alpha": alpha, "delta": delta,
                           "power": _power_wald(n, margin, discordance, delta, alpha),
                           "n_for_80": ni_sample_size(discordance, margin, alpha, delta, 0.8)}
    if sims <= 0:
        return out
    rng = np.random.default_rng(seed)
    z = _z(1 - alpha)
    for name, dl in (("type1", -margin), ("power", delta)):
        p_win, p_loss = (discordance + dl) / 2, (discordance - dl) / 2
        if min(p_win, p_loss) < 0 or p_win + p_loss > 1:
            out[f"{name}_boot"] = out[f"{name}_score"] = float("nan")
            continue
        rej_b = rej_s = 0
        for _ in range(sims):
            w, l, _t = rng.multinomial(n, [p_win, p_loss, 1 - discordance])
            bs = rng.multinomial(n, [w / n, l / n, 1 - (w + l) / n], size=n_boot)
            lcb = float(np.quantile((bs[:, 0] - bs[:, 1]) / n, alpha))
            rej_b += lcb > -margin
            dh = (w + l) / n
            se0 = math.sqrt(max(dh - margin ** 2, 1e-12) / n)
            rej_s += ((w - l) / n + margin) / se0 > z
        out[f"{name}_boot"], out[f"{name}_score"] = rej_b / sims, rej_s / sims
    out["sims"], out["n_boot"] = sims, n_boot
    return out


def ni_mde(n: int, discordance: float, margin: float, alpha: float = 0.05, power: float = 0.8) -> dict[str, Any]:
    """Minimal detectable effect of the NI test at ``n`` tasks and the achieved discordance (A6 / P0-2: printed next
    to every primary). ``delta_mde``: the smallest true Delta (arm - reference; negative = a loss) at which NI at
    margin m is established with the given power, solving (m + Delta) sqrt(n) = z_{1-alpha} sigma_0 + z_{power}
    sigma_1(Delta); nan when even Delta = +sqrt(d) (every discordant task a win) is under-powered. ``margin_mde``:
    the smallest margin that reaches the power at Delta = 0 (the margin the sample can actually support)."""
    out: dict[str, Any] = {"n": n, "discordance": discordance, "margin": margin, "alpha": alpha, "power": power}
    hi = math.sqrt(max(discordance, 0.0))
    f = lambda dl: _power_wald(n, margin, discordance, dl, alpha) - power  # noqa: E731
    if discordance <= 0 or f(hi) < 0:
        out["delta_mde"] = float("nan")
    else:
        lo = -margin
        if f(lo) >= 0:
            out["delta_mde"] = -margin
        else:
            for _ in range(100):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if f(mid) >= 0 else (mid, hi)
            out["delta_mde"] = hi
    g = lambda m: _power_wald(n, m, discordance, 0.0, alpha) - power  # noqa: E731
    mhi = math.sqrt(max(discordance, 0.0))
    if discordance <= 0 or g(mhi) < 0:
        out["margin_mde"] = float("nan")
    else:
        mlo = 1e-9
        for _ in range(100):
            mid = (mlo + mhi) / 2
            mlo, mhi = (mlo, mid) if g(mid) >= 0 else (mid, mhi)
        out["margin_mde"] = mhi
    return out


def ni_pvalue(base: Mapping[str, float], arm: Mapping[str, float], margin: float, n_boot: int = 10000, seed: int = 0,
              alpha: float = 0.05, alpha_adj: float | None = None, exact_below: int = 5,
              unconditional: bool = True, rule: str | None = None) -> dict[str, Any]:
    """The registered NI test as ONE procedure (A6 / P0-1). Statistic: the paired task-bootstrap distribution of
    Delta* = mean_t(arm_t - base_t) (B = ``n_boot``, task = resampling unit, trials of a task stay together: the
    inputs are per-task means; :func:`_paired_boot`, the resampling of :func:`paired_bootstrap_diff`).

    * ``p_ni`` = P*(Delta* <= -m): enters Holm with the superiority p-values ``p_sup`` = P*(Delta* <= 0) of the
      other primaries (= ``paired_bootstrap_diff``'s ``p_better``); the test rejects iff p_ni <= alpha_adj, the
      Holm-adjusted level, equivalently iff ``lcb_adj``, the one-sided (1 - alpha_adj) bootstrap lower bound, is above
      -m (``ni_adj``; ``alpha_adj`` defaults to ``alpha``).
    * ``lcb95`` (the one-sided (1 - alpha) bound, reported descriptively) and ``ni_95``; ``ci95`` two-sided.
    * ``inconclusive``: NI fails at the adjusted level while the two-sided (1 - alpha) CI covers 0 -- reported as
      "inconclusive at this n", not as inferiority (P0-2).
    * **Decision rule (A6.21; ``rule`` = "bb_ordered", the default).** ``p_bb`` = :func:`ni_certified_p`: one
      ordering of every (wins, losses) outcome (the monotone hull of the A6.14 rule's nominal p), its tail maximised
      over the Berger-Boos nuisance set with a certified branch-and-bound upper bound, plus gamma = 1e-4; a valid
      p-value at every n, level and null point. ``p_decisive`` = ``p_bb`` enters Holm; ``ni_adj`` / ``ni_95`` are
      p_bb <= alpha_adj / alpha. ``bb`` holds the certificate (``ub`` >= sup, ``lb``, nuisance set, evaluations).
    * Comparison rules (superseded, reported): ``p_tango`` = 1 - Phi(Z(-m)), Tango's (1998) one-sided score p on the
      paired counts (``tango_lcb_adj`` the matching (1 - alpha_adj) lower bound); ``rule="tango_uncond"`` decides as
      A6.17 (``p_uncond``, :func:`ni_unconditional_p`, grid maximum on the score branch), ``rule="tango"`` (or
      ``unconditional=False``) as A6.14 (asymptotic p_T); under both, ``wins == 0`` or fewer than ``exact_below``
      discordant tasks switch to :func:`ni_exact_check` (``rule`` = "exact"). Whatever the rule, ``exact`` holds
      :func:`ni_exact_check` in that region (the loss-rate upper bound reported at zero discordance).
      ``p_ni`` / ``lcb_adj`` / ``ni_boot_adj`` (the pre-A6.14 bootstrap rule) stay descriptive: its size at the Holm
      floor is up to 2.3x alpha (results/reviews/2026-10-01c/ni_size_sim.md).
    * ``mde``: :func:`ni_mde` at the achieved discordance.
    For two System-1 families combine with :func:`holm_conjunction` before Holm."""
    rule = rule or ("bb_ordered" if unconditional else "tango")
    if rule not in ("bb_ordered", "tango_uncond", "tango"):
        raise ValueError(f"unknown NI rule {rule!r}")
    a_adj = alpha if alpha_adj is None else float(alpha_adj)
    keys, d, boot = _paired_boot(base, arm, n_boot, seed)
    nan = float("nan")
    if not keys:
        return {"n": 0, "diff": nan, "margin": margin, "p_ni": nan, "p_sup": nan, "alpha": alpha, "alpha_adj": a_adj,
                "lcb95": nan, "lcb_adj": nan, "ci95": [nan, nan], "ni_95": False, "ni_adj": False,
                "inconclusive": False, "n_discordant": 0, "wins": 0, "losses": 0, "exact": None, "mde": None}
    n = len(keys)
    wins, losses = int(np.sum(d > 1e-12)), int(np.sum(d < -1e-12))
    out: dict[str, Any] = {
        "n": n, "diff": float(d.mean()), "margin": margin, "n_boot": n_boot, "seed": seed,
        "p_ni": float(np.mean(boot <= -margin)), "p_sup": float(np.mean(boot <= 0)),
        "alpha": alpha, "alpha_adj": a_adj,
        "lcb95": float(np.quantile(boot, alpha)), "lcb_adj": float(np.quantile(boot, a_adj)),
        "ci95": [float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))],
        "n_discordant": wins + losses, "wins": wins, "losses": losses, "exact": None,
        "mde": ni_mde(n, (wins + losses) / n, margin, a_adj)}
    # --- the decision rule (A6.14, review round 6): Tango's one-sided score test of H0: Delta <= -m on the paired
    # (wins, losses, n) counts, p_T = 1 - Phi(Z(-m)); the exact one-sided binomial test on the losses replaces it
    # whenever wins == 0 or fewer than ``exact_below`` tasks are discordant. The bootstrap p_ni / lcb stay
    # descriptive (its size at the Holm floor is inflated: results/reviews/2026-10-01c/ni_size_sim.md).
    both = int(np.sum((np.asarray([base[k] for k in keys], float) > 0.5) & (np.asarray([arm[k] for k in keys], float) > 0.5)))
    neither = n - both - wins - losses
    z = _tango_z(-margin, wins, losses, n)
    out["p_tango"] = float(_norm_sf(z))
    out["tango_lcb_adj"] = float(tango_interval(both, wins, losses, max(neither, 0), alpha=a_adj, one_sided=True)["ci"][0])
    out["tango_lcb95"] = float(tango_interval(both, wins, losses, max(neither, 0), alpha=alpha, one_sided=True)["ci"][0])
    switched = wins == 0 or wins + losses < exact_below
    if switched:          # the A6.14 switch region: the exact loss-count check (decisive under the superseded rules;
        ex = ni_exact_check(wins, losses, n, margin, a_adj)       # its loss-rate upper bound is reported under all)
        out["exact"] = {**ex, "p_95": ni_exact_check(wins, losses, n, margin, alpha)["p"]}
    if rule == "bb_ordered":
        # A6.21 (review round 8): one ordering of the whole outcome space, certified Berger-Boos supremum; valid at
        # every n, level and null point (proof: the comment block above NI_BB_GAMMA; docs/PREREGISTRATION.md A6.21 (a))
        bb = ni_certified_p(wins, losses, n, margin, exact_below=exact_below)
        out["p_bb"] = bb["p"]
        out["bb"] = {k: bb[k] for k in ("ub", "lb", "p_lb", "w_lb", "w_set", "gamma", "s_obs", "s_nominal", "n_evals",
                                        "converged", "rel_tol", "abs_tol")}
        if switched:                                  # A6.17's p_U, descriptive (it is undefined on the switch region)
            out["p_uncond"], out["w_max"] = float("nan"), float("nan")
        else:
            u = ni_unconditional_p(wins, losses, n, margin, exact_below=exact_below)
            out["p_uncond"], out["w_max"] = u["p"], u["w_max"]
        out["p_decisive"], out["rule"] = bb["p"], "bb_ordered"
        out["ni_95"], out["ni_adj"] = bool(bb["p"] <= alpha), bool(bb["p"] <= a_adj)
    else:
        # A6.17 (rule tango_uncond): the Tango branch decides by its grid-maximised unconditional p_U; A6.14 (rule
        # tango): by the asymptotic p_T; both switch to the exact loss-count test (superseded by A6.21)
        if rule == "tango_uncond" and not switched:
            u = ni_unconditional_p(wins, losses, n, margin, exact_below=exact_below)
            out["p_uncond"], out["w_max"] = u["p"], u["w_max"]
        else:
            out["p_uncond"], out["w_max"] = out["p_tango"], float("nan")
        out["ni_95"] = bool(out["p_uncond"] <= alpha)
        out["ni_adj"] = bool(out["p_uncond"] <= a_adj)
        out["p_decisive"], out["rule"] = out["p_uncond"], rule
        if switched:
            out["ni_adj"], out["ni_95"] = out["exact"]["ni"], bool(out["exact"]["p_95"] < alpha)
            out["p_decisive"], out["rule"] = out["exact"]["p"], "exact"
    out["ni_boot_adj"] = bool(out["p_ni"] <= a_adj)   # the pre-A6.14 bootstrap decision, descriptive
    out["inconclusive"] = bool(not out["ni_adj"] and out["ci95"][0] <= 0 <= out["ci95"][1])
    return out


def holm_conjunction(p_jev: float, p_intern: float) -> float:
    """The two-family conjunction of C1 / C2 / C6 (A6 / P0-1): a claim confirmed for both System-1 families is an
    intersection-union test, so its p-value is max(p_Jev, p_Intern) and enters Holm as one test (no further
    correction). nan if either is nan (a family that did not run or was demoted by the competence gate: the claim
    is then "Jev-specific", tested on p_Jev alone)."""
    return max(float(p_jev), float(p_intern))


def ni_holm_levels(pvals: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """Holm step-down levels of a family: for each test its adjusted p-value (:func:`holm`), the level alpha_adj =
    alpha / (m - i) at its step (i = rank of its raw p, m = family size; nan p-values are left out) and whether it
    rejects (Holm stops at the first non-rejection, so a later test with p <= alpha_adj can still be "retained").
    ``lcb_adj`` of :func:`ni_pvalue` at ``alpha_adj`` is the bound the A6 rule compares with -m."""
    valid = sorted(((p, k) for k, p in pvals.items() if p is not None and not math.isnan(p)))
    m = len(valid)
    adj = holm(pvals)
    out: dict[str, dict[str, float]] = {k: {"p": float("nan"), "p_holm": float("nan"), "alpha_adj": float("nan"),
                                            "reject": False, "rank": float("nan")} for k in pvals}
    stopped = False
    for i, (p, k) in enumerate(valid):
        a_i = alpha / (m - i)
        rej = (not stopped) and p <= a_i
        if not rej:
            stopped = True
        out[k] = {"p": p, "p_holm": adj[k], "alpha_adj": a_i, "reject": bool(rej), "rank": i + 1, "m": m}
    return out


def newcombe_interval(both: int, only_a: int, only_b: int, neither: int, alpha: float = 0.05) -> dict[str, Any]:
    """Newcombe (1998, Stat. Med. 17:2635-2650) method 10: a score-type (1 - alpha) interval for the difference of
    paired proportions p_a - p_b, descriptive for single-trial binary endpoints (A6 / P0-1). Cells: ``both`` (a and b
    succeed), ``only_a``, ``only_b``, ``neither``; p_a = (both + only_a) / n, p_b = (both + only_b) / n, d = p_a - p_b
    = (only_a - only_b) / n. With the Wilson limits (l_a, u_a), (l_b, u_b) of each marginal and the corrected
    phi-coefficient phi = (A - B) / sqrt(prod of marginals), A = both * neither, B = only_a * only_b, where A - B is
    replaced by max(A - B - n/2, 0) when A > B (and phi = 0 when a marginal is empty):
        lower = d - sqrt((p_a - l_a)^2 - 2 phi (p_a - l_a)(u_b - p_b) + (u_b - p_b)^2),
        upper = d + sqrt((u_a - p_a)^2 - 2 phi (u_a - p_a)(p_b - l_b) + (p_b - l_b)^2).
    With no discordant pair at all (both = n) the interval is +-z^2 / (n + z^2) (not zero-width)."""
    n = both + only_a + only_b + neither
    nan = float("nan")
    if n <= 0:
        return {"n": 0, "diff": nan, "ci": [nan, nan], "alpha": alpha}
    pa, pb = (both + only_a) / n, (both + only_b) / n
    la, ua = wilson_ci(both + only_a, n, alpha)
    lb, ub = wilson_ci(both + only_b, n, alpha)
    marg = (both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither)
    if marg <= 0:
        phi = 0.0
    else:
        A, B = both * neither, only_a * only_b
        num = max(A - B - n / 2, 0.0) if A > B else float(A - B)
        phi = num / math.sqrt(marg)
    d = pa - pb
    lo = d - math.sqrt(max((pa - la) ** 2 - 2 * phi * (pa - la) * (ub - pb) + (ub - pb) ** 2, 0.0))
    hi = d + math.sqrt(max((ua - pa) ** 2 - 2 * phi * (ua - pa) * (pb - lb) + (pb - lb) ** 2, 0.0))
    return {"n": n, "diff": d, "ci": [max(-1.0, lo), min(1.0, hi)], "alpha": alpha, "phi": phi, "method": "Newcombe 1998 (10)"}


def _norm_sf(z: float) -> float:
    """1 - Phi(z) (upper tail of the standard normal), inf-safe."""
    if math.isinf(z):
        return 0.0 if z > 0 else 1.0
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _tango_z(delta: float, b: int, c: int, n: int) -> float:
    """Tango's (1998) score statistic for H0: p_a - p_b = delta on paired binary data (b = a-only successes,
    c = b-only successes): Z = (b - c - n delta) / sqrt(n (2 p~_c + delta (1 - delta))), p~_c the restricted MLE of
    P(b-only) under H0, the root of 2n p^2 - (b + c + delta (b - c - 2n)) p - c delta (1 - delta) = 0."""
    A = 2.0 * n
    B = -(b + c) - delta * (b - c - 2 * n)
    C = -c * delta * (1 - delta)
    disc = max(B * B - 4 * A * C, 0.0)
    p = (-B + math.sqrt(disc)) / (2 * A)
    var = n * (2 * p + delta * (1 - delta))
    if var <= 1e-15:
        return math.copysign(float("inf"), b - c - n * delta) if abs(b - c - n * delta) > 1e-12 else 0.0
    return (b - c - n * delta) / math.sqrt(var)


def _tango_z_vec(delta: float, b: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """:func:`_tango_z` over arrays of (b, c) counts (same closed form; zero-variance cells get +-inf / 0)."""
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    A = 2.0 * n
    B = -(b + c) - delta * (b - c - 2 * n)
    C = -c * delta * (1 - delta)
    disc = np.maximum(B * B - 4 * A * C, 0.0)
    p = (-B + np.sqrt(disc)) / (2 * A)
    var = n * (2 * p + delta * (1 - delta))
    num = b - c - n * delta
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 1e-15, num / np.sqrt(np.maximum(var, 1e-300)),
                     np.where(np.abs(num) > 1e-12, np.copysign(np.inf, num), 0.0))
    return z


def _lgamma_table(n: int) -> np.ndarray:
    """lgamma(k + 1) = log k! for k = 0..n."""
    return np.array([math.lgamma(k + 1) for k in range(n + 1)], float)


def _trinomial_logpmf(w: float, margin: float, W: np.ndarray, L: np.ndarray, n: int,
                      lgt: np.ndarray | None = None) -> np.ndarray:
    """log Multinomial(n; w, w + margin, 1 - 2w - margin)(W, L): the paired (win, loss, tie) counts at the NI
    boundary Delta = -margin with per-task win probability ``w`` (the nuisance parameter)."""
    T = n - W - L
    lgt = _lgamma_table(n) if lgt is None else lgt
    base = lgt[n] - lgt[W] - lgt[L] - lgt[T]
    lw = W * math.log(w) if w > 0 else np.where(W > 0, -np.inf, 0.0)
    ll = L * math.log(w + margin) if w + margin > 0 else np.where(L > 0, -np.inf, 0.0)
    pt = 1.0 - 2.0 * w - margin
    lt = T * math.log(pt) if pt > 1e-15 else np.where(T > 0, -np.inf, 0.0)
    return base + lw + ll + lt


def ni_unconditional_w_grid(margin: float, step: float = 0.0025) -> np.ndarray:
    """The nuisance grid over which :func:`ni_unconditional_p` maximises: w = 0, 0.0005, 0.001, then ``step`` up to
    (1 - margin) / 2 (every tie probability from 1 - margin down to 0), PREREGISTRATION A6.17."""
    hi = (1.0 - margin) / 2.0
    g = [0.0, 0.0005, 0.001] + [round(k * step, 6) for k in range(1, int(hi / step) + 1)]
    return np.array(sorted(set(v for v in g if 0.0 <= v <= hi + 1e-12)), float)


def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) Clopper-Pearson interval for a binomial proportion (bisection on the binomial
    cdf, no scipy): lower = 0 when k = 0, upper = 1 when k = n."""
    if n <= 0:
        return 0.0, 1.0
    half = alpha / 2.0

    def solve(cond, lo: float, hi: float) -> float:
        for _ in range(60):
            mid = (lo + hi) / 2
            if cond(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda q: 1.0 - _binom_cdf(k - 1, n, q) < half, 0.0, k / n)   # P[X >= k] = half
    upper = 1.0 if k == n else solve(lambda q: _binom_cdf(k, n, q) > half, k / n, 1.0)            # P[X <= k] = half
    return lower, upper


def ni_nuisance_set(wins: int, losses: int, n: int, margin: float, gamma: float = 1e-4) -> tuple[float, float]:
    """Berger-Boos (1 - gamma) confidence set for the nuisance win probability w at the NI boundary: the discordance
    rate rho = 2w + m is Bin(n, rho) / n there, so the Clopper-Pearson (1 - gamma) interval of rho from D = W + L,
    mapped by w = (rho - m) / 2 and clipped to [0, (1 - m) / 2]."""
    lo, hi = clopper_pearson(int(wins) + int(losses), int(n), gamma)
    w_hi = (1.0 - margin) / 2.0
    return (min(max((lo - margin) / 2.0, 0.0), w_hi), min(max((hi - margin) / 2.0, 0.0), w_hi))


def ni_unconditional_p(wins: int, losses: int, n: int, margin: float, step: float = 0.0025,
                       prune: float = 1e-15, exact_below: int = 5, gamma: float = 1e-4) -> dict[str, Any]:
    """Exact unconditional one-sided p-value of Tango's score statistic for H0: Delta <= -margin on paired binary
    data, for the Tango branch of the registered rule (PREREGISTRATION A6.17; Suissa & Shuster 1985; Berger & Boos
    1994): with the per-task (win, loss, tie) counts trinomial at the boundary Delta = -m and win probability w
    (nuisance),

        p_U = max_{w in C_gamma} P_w[ Z(-m) >= z_obs  and  W >= 1  and  W + L >= exact_below ]  +  gamma,

    the tail restricted to the region of the sample space on which the Tango branch decides (W = 0 or fewer than
    ``exact_below`` discordant tasks go to the exact loss-count test, A6.14) and the maximum taken over the points
    of :func:`ni_unconditional_w_grid` inside the Berger-Boos (1 - gamma) confidence set C_gamma for w
    (:func:`ni_nuisance_set`, from the observed discordant count), its two end points and the point estimate.
    SUPERSEDED by A6.21 (:func:`ni_certified_p`, rule ``bb_ordered``; kept for comparison): a maximum over grid
    points is a lower approximation of the Berger-Boos supremum, not a certified upper bound, and the two-branch rule
    built on it is a union of two tests (review round 8, W1). Berger & Boos (1994, JASA 89:1012-1016) give validity
    for the exact supremum: the confidence-set restriction keeps the test from being calibrated on win rates the data
    exclude (the plain
    maximum over all w sits at w -> (1 - m) / 2, every task discordant, and costs 10-30 % of the rejection
    probability near the boundary). The size of the whole two-branch rule is verified by exact summation in
    ``scripts/ni_exact_size.py`` (rule ``registered_u``). ``p_at_w`` records the tail probability at each
    evaluated w; ``w_max`` the maximiser; ``w_set`` the confidence set. Exact enumeration of all (W, L) with
    W + L <= n (O(n^2) pairs, pmf below ``prune`` dropped per w; at n = 1000 about a second, memoised)."""
    nan = float("nan")
    if n <= 0:
        return {"n": n, "wins": wins, "losses": losses, "margin": margin, "p": nan, "p_tango": nan, "z": nan,
                "w_max": nan, "n_grid": 0, "gamma": gamma}
    wins, losses = int(wins), int(losses)
    z_obs, ws, p_at, w_set = _ni_uncond_cached(wins, losses, int(n), float(margin), float(step), float(prune),
                                               int(exact_below), float(gamma))
    k = int(np.argmax(p_at))
    return {"n": n, "wins": wins, "losses": losses, "margin": margin, "p": float(min(1.0, p_at[k] + gamma)),
            "p_max_tail": float(p_at[k]), "gamma": gamma, "w_set": [float(w_set[0]), float(w_set[1])],
            "p_tango": float(_norm_sf(z_obs)), "z": float(z_obs), "w_max": float(ws[k]), "n_grid": int(len(ws)),
            "p_at_w": {float(w): float(p) for w, p in zip(ws, p_at)}, "step": step, "exact_below": exact_below}


@functools.lru_cache(maxsize=4096)
def _ni_uncond_cached(wins: int, losses: int, n: int, margin: float, step: float, prune: float, exact_below: int,
                      gamma: float):
    """The enumeration behind :func:`ni_unconditional_p`, memoised per (wins, losses, n, margin, grid): the score
    card recomputes the test at each Holm level and the p-value does not depend on the level."""
    z_obs = _tango_z(-margin, wins, losses, n)
    W, L = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")   # all (W, L) with W + L <= n
    W, L = W.ravel(), L.ravel()
    keep = W + L <= n
    W, L = W[keep], L[keep]
    z = _tango_z_vec(-margin, W, L, n)
    tail = (z >= z_obs - 1e-12) & (W >= 1) & (W + L >= exact_below)
    Wt, Lt = W[tail], L[tail]
    lgt = _lgamma_table(n)
    w_lo, w_hi = ni_nuisance_set(wins, losses, n, margin, gamma) if gamma > 0 else (0.0, (1.0 - margin) / 2.0)
    grid = ni_unconditional_w_grid(margin, step)
    w_hat = min(max(((wins + losses) / n - margin) / 2.0, 0.0), (1.0 - margin) / 2.0)
    pts = {round(float(w), 10) for w in grid if w_lo - 1e-12 <= w <= w_hi + 1e-12}
    pts |= {round(w_lo, 10), round(w_hi, 10), round(min(max(w_hat, w_lo), w_hi), 10)}
    ws = np.array(sorted(pts), float)
    p_at = np.empty(len(ws), float)
    for i, w in enumerate(ws):
        pm = np.exp(_trinomial_logpmf(float(w), margin, Wt, Lt, n, lgt))
        p_at[i] = float(np.sum(pm[pm >= prune]))
    return z_obs, ws, p_at, (w_lo, w_hi)


# ------------------------------------------------------------------ A6.21: one ordering, certified Berger-Boos supremum
# The decisive NI p-value from A6.21 on (review round 8, W1). One statistic orders the WHOLE outcome space
# {(W, L): W, L >= 0, W + L <= n}; its tail probability is maximised over the boundary nuisance with a certified upper
# bound. Proof sketch (docs/PREREGISTRATION.md A6.21 (a) has the full argument):
#   1. s(W, L) = log of the nominal p of the A6.14 switching rule (asymptotic Tango p on W >= 1, W + L >= 5; exact
#      loss-count p P[Bin(n, m) <= L] elsewhere); s~ = its monotone hull, s~(W, L) = min s(W', L') over W' <= W,
#      L' >= L, W' + L' <= n. A point with more wins and fewer losses is at least as significant, so every tail
#      R_t = {s~ <= t} is an upper set of the order "more wins, fewer losses".
#   2. Coupling: draw each trial from one uniform U_i (loss if U_i < P(loss), win if U_i > 1 - P(win), tie else).
#      If theta* has P(win) no smaller and P(loss) no larger than theta, every trial moves up (loss < tie < win), so
#      (W, L) moves up and P_theta(R_t) <= P_theta*(R_t). A null point theta (P(loss) - P(win) >= m) with discordance
#      rho is dominated so by the boundary point with the same discordance, w* = (rho - m) / 2.
#   3. D = W + L ~ Bin(n, rho) under EVERY theta, so the Clopper-Pearson (1 - gamma) set for rho mapped by
#      w = (rho - m) / 2 covers w* with probability >= 1 - gamma; Berger & Boos (1994): p = gamma + U(x), with U(x)
#      >= sup over that set of P_w[s~(X) <= s~(x)], gives P_theta(p <= alpha) <= gamma + (alpha - gamma) = alpha for
#      every theta in the null, every n and every alpha.
#   4. U(x) is certified by branch-and-bound over the set: f(w) = P_w(R) satisfies, for any q > 1 (Hoelder),
#      f(x) <= f(a)^(1 - 1/q) M_q(a, x)^(n/q), M_q(a, x) = sum_i p_i(x)^q p_i(a)^(1 - q) (the per-trial Renyi
#      moment, exact in closed form); M_q(a, .) is convex along the boundary path with minimum 1 at a, so its value at
#      the far end of a subinterval bounds the whole subinterval. U is the largest such subinterval bound, refined
#      until U - L <= max(abs_tol, rel_tol * L) (L the best evaluated value); U (+ 2n * 1e-15 for floating-point end
#      points) is what is returned.
NI_BB_GAMMA = 1e-4           # Berger-Boos confidence-set level (as A6.17)
NI_BB_REL_TOL = 1e-3         # branch-and-bound tolerance: U - L <= max(NI_BB_ABS_TOL, NI_BB_REL_TOL * L)
NI_BB_ABS_TOL = 1e-7
NI_BB_TIE = 1e-9             # tail of x = {s~ <= s~(x) + NI_BB_TIE} (log-p scale: ties within 1e-9 relative count as ties)
NI_BB_MAX_EVALS = 20000
_NI_Q_GRID = 1.0 + 10.0 ** np.linspace(-2.0, 7.0, 181)   # Hoelder exponents tried (any q > 1 gives a valid bound)
_LOG_2PI = math.log(2.0 * math.pi)
_erfc_u = np.frompyfunc(math.erfc, 1, 1)


def _log_norm_sf_vec(z: np.ndarray) -> np.ndarray:
    """log(1 - Phi(z)) elementwise: math.erfc up to z = 30, the asymptotic series beyond (relative error < 1e-12)."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    big = z > 30.0
    sm = ~big
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out[sm] = np.log(0.5 * _erfc_u(z[sm] / math.sqrt(2.0)).astype(float))
        zb = z[big]
        r = 1.0 / (zb * zb)
        out[big] = -0.5 * zb * zb - np.log(zb) - 0.5 * _LOG_2PI + np.log1p(r * (-1.0 + r * (3.0 + r * (-15.0 + 105.0 * r))))
    return out


@functools.lru_cache(maxsize=6)
def _ni_order_tables(n: int, margin: float, exact_below: int) -> tuple[np.ndarray, np.ndarray]:
    """(s, s~) over the (n + 1) x (n + 1) square, +inf off the triangle W + L <= n (read-only, memoised)."""
    W, L = np.indices((n + 1, n + 1))
    tri = W + L <= n
    lf = _lgamma_table(n)
    k = np.arange(n + 1)
    lpmf = lf[n] - lf[k] - lf[n - k] + k * math.log(margin) + (n - k) * math.log1p(-margin)
    logcdf = np.minimum(np.logaddexp.accumulate(lpmf), 0.0)          # log P[Bin(n, m) <= l]
    ex = tri & ((W == 0) | (W + L < exact_below))
    sc = tri & ~ex
    s = np.full((n + 1, n + 1), np.inf)
    s[ex] = logcdf[L[ex]]
    s[sc] = _log_norm_sf_vec(_tango_z_vec(-margin, W[sc], L[sc], n))
    st = np.minimum.accumulate(s, axis=0)                              # min over W' <= W
    st = np.minimum.accumulate(st[:, ::-1], axis=1)[:, ::-1]           # then over L' >= L
    st = np.where(tri, st, np.inf)
    s.setflags(write=False)
    st.setflags(write=False)
    return s, st


def ni_order_table(n: int, margin: float, exact_below: int = 5, hull: bool = True) -> np.ndarray:
    """The A6.21 ordering statistic on the whole outcome space, as an (n + 1) x (n + 1) array indexed [W, L] (+inf off
    the triangle W + L <= n; smaller = more significant). ``hull=False``: s, the log nominal p of the A6.14 switching
    rule (log of Tango's asymptotic one-sided p where W >= 1 and W + L >= ``exact_below``, log P[Bin(n, m) <= L]
    elsewhere). ``hull=True``: its monotone hull s~(W, L) = min s(W', L') over W' <= W, L' >= L, W' + L' <= n, which
    is non-increasing in W and non-decreasing in L (two cumulative minima). Memoised per (n, margin, exact_below)."""
    s, st = _ni_order_tables(int(n), float(margin), int(exact_below))
    return st if hull else s


def _binom_logpmf_rows(n: int, ps: Sequence[float] | np.ndarray, lf: np.ndarray) -> np.ndarray:
    """(len(ps), n + 1) array of log Bin(n, p)(k), p in [0, 1] (0 * log 0 = 0)."""
    ps = np.asarray(ps, float)
    k = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(ps[:, None] > 0, k[None, :] * np.log(np.where(ps > 0, ps, 1.0))[:, None],
                      np.where(k[None, :] == 0, 0.0, -np.inf))
        lq = np.where(ps[:, None] < 1, (n - k)[None, :] * np.log1p(-np.where(ps < 1, ps, 0.0))[:, None],
                      np.where(k[None, :] == n, 0.0, -np.inf))
    return lf[n] - lf[k][None, :] - lf[n - k][None, :] + lp + lq


def _binom_window(n: int, p_lo: float, p_hi: float, lf: np.ndarray, eps: float) -> tuple[int, int]:
    """[a, b] with P[X < a] <= eps at p_lo and P[X > b] <= eps at p_hi, X ~ Bin(n, p) (so for every p in [p_lo, p_hi],
    Bin(n, p) being stochastically increasing in p). Only a summation window: the mass outside it is bounded exactly
    per evaluated p and added to the certified value."""
    rows = np.exp(_binom_logpmf_rows(n, [p_lo, min(p_hi, 1.0)], lf))
    left = np.cumsum(rows[0])
    right = np.cumsum(rows[1][::-1])[::-1]
    a = int(np.argmax(left > eps)) if np.any(left > eps) else 0
    idx = np.nonzero(right > eps)[0]
    return a, (int(idx[-1]) if len(idx) else n)


def ni_holder_logbound(logF: Any, a: Any, x: Any, n: int, margin: float, qs: np.ndarray | None = None) -> np.ndarray:
    """Certified bound on P_x(R) from P_a(R) along the NI boundary path p(w) = (w, w + m, 1 - 2w - m), for ANY event R
    of n i.i.d. trials (A6.21): by Hoelder's inequality with exponents (q / (q - 1), q) under P_a,

        P_x(R) = E_a[1_R LR] <= P_a(R)^(1 - 1/q) E_a[LR^q]^(1/q) = P_a(R)^(1 - 1/q) M_q(a, x)^(n/q),

    LR = dP_x / dP_a, M_q(a, x) = sum_i p_i(x)^q p_i(a)^(1 - q) (closed form; log M_q computed as
    log1p(sum_i p_i(a) expm1(q log1p((p_i(x) - p_i(a)) / p_i(a)))) for accuracy). Returns log of the minimum over
    the exponents ``qs`` (default 181 values in (1.01, 1e7]) given ``logF`` = log of an UPPER bound on P_a(R); +inf
    where P_x is not absolutely continuous w.r.t. P_a (a = 0 or a = (1 - m)/2 with x != a). Broadcasts over its
    arguments. Because M_q(a, .) is convex along the path with minimum 1 at a, the bound at x also holds at every
    point between a and x."""
    qs = _NI_Q_GRID if qs is None else np.asarray(qs, float)
    logF, a, x = np.broadcast_arrays(np.atleast_1d(np.asarray(logF, float)), np.atleast_1d(np.asarray(a, float)),
                                     np.atleast_1d(np.asarray(x, float)))
    bad, log_m = _ni_holder_logm(a, x, margin, qs)
    with np.errstate(invalid="ignore"):
        val = (1.0 - 1.0 / qs) * logF[..., None] + (n / qs) * log_m
    val = np.where(np.isnan(val), np.inf, val)
    return np.where(bad, np.inf, val.min(axis=-1))


def _ni_holder_logm(a: np.ndarray, x: np.ndarray, margin: float, qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bad, log M_q(a, x)) for the boundary path, shape (..., len(qs)); ``bad`` where P_x is not << P_a."""
    m = float(margin)
    pa = np.stack([a, a + m, 1.0 - 2.0 * a - m], -1)
    dx = (x - a)[..., None] * np.array([1.0, 1.0, -2.0])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rel = np.where(pa > 0, dx / np.where(pa > 0, pa, 1.0), 0.0)
        bad = np.any((pa <= 0) & (dx != 0), axis=-1)
        delta = np.log1p(np.maximum(rel, -1.0))
        terms = np.where(pa[..., None] > 0, pa[..., None] * np.expm1(qs * delta[..., None]), 0.0)
        log_m = np.log1p(terms.sum(axis=-2))
    return bad, np.where(np.isnan(log_m), np.inf, log_m)


def ni_tail_sup(order: np.ndarray, thresh: float, n: int, margin: float, w_lo: float, w_hi: float,
                rel_tol: float = NI_BB_REL_TOL, abs_tol: float = NI_BB_ABS_TOL, max_evals: int = NI_BB_MAX_EVALS,
                k0: int = 16, eps_out: float = 1e-14) -> dict[str, Any]:
    """Certified supremum over w in [w_lo, w_hi] of f(w) = P_w[order(W, L) <= thresh] under the NI boundary trinomial
    (P(win) = w, P(loss) = w + m, P(tie) = 1 - 2w - m), by branch-and-bound (A6.21).

    * Evaluation: f is summed over the cells of the region inside a (W, L) window that holds all but <= 4 * eps_out of
      the mass for every w in [w_lo, w_hi]; the certified value at a node is f_hi = f_lo (1 + 1e-9) + (exact binomial
      mass outside the window) (1 + 1e-6) + 1e-15 >= f(w), f_lo <= f(w) the windowed sum (the slack covers
      floating-point error of the log-gamma pmf, < 1e-11 relative).
    * Bound per subinterval [a, b] (midpoint c): min(max(B(a -> c), B(b -> c)), B(a -> b), B(b -> a)) with B(a -> x) the
      :func:`ni_holder_logbound` from the certified value at a (valid on the whole segment from a to x), times
      (1 + 1e-9).
    * Refinement: bisect every subinterval whose bound exceeds L + tol, L = max f_lo over the nodes, tol =
      max(abs_tol, rel_tol * L); stop when U = max(subinterval bounds, node values) <= L + tol or after ``max_evals``
      evaluations (``converged`` False; U is still a valid upper bound).
    Returns ``ub`` (= U, certified >= sup f), ``lb`` (= L, attained at ``w_lb``), ``n_evals``, ``converged``."""
    m = float(margin)
    lf = _lgamma_table(n)
    w_lo, w_hi = float(w_lo), float(w_hi)
    Wa, Wb = _binom_window(n, w_lo, w_hi, lf, eps_out)
    La, Lb = _binom_window(n, w_lo + m, w_hi + m, lf, eps_out)
    Wi, Li = np.nonzero(np.asarray(order)[Wa:Wb + 1, La:Lb + 1] <= thresh)
    Wc, Lc = Wi + Wa, Li + La
    keep = Wc + Lc <= n
    Wc, Lc = Wc[keep], Lc[keep]
    Tc = n - Wc - Lc
    base = lf[n] - lf[Wc] - lf[Lc] - lf[Tc]
    A = np.stack([Wc, Lc, Tc], 1).astype(float)
    C = len(base)
    n_evals = 0

    def evaluate(ws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal n_evals
        ws = np.asarray(ws, float)
        n_evals += len(ws)
        with np.errstate(divide="ignore"):
            B = np.stack([np.log(ws), np.log(ws + m), np.log(np.maximum(1.0 - 2.0 * ws - m, 0.0))], 0)
        B = np.where(np.isinf(B), -1e300, B)                      # 0 * log 0 = 0; positive counts -> pmf 0
        flo = np.zeros(len(ws))
        if C:
            g = max(1, 2_000_000 // C)
            for i in range(0, len(ws), g):
                Bi = B[:, i:i + g]
                flo[i:i + g] = np.exp(base[:, None] + np.outer(A[:, 0], Bi[0]) + np.outer(A[:, 1], Bi[1])
                                      + np.outer(A[:, 2], Bi[2])).sum(0)
        pw = np.exp(_binom_logpmf_rows(n, ws, lf))
        pl = np.exp(_binom_logpmf_rows(n, np.minimum(ws + m, 1.0), lf))
        out = pw[:, :Wa].sum(1) + pw[:, Wb + 1:].sum(1) + pl[:, :La].sum(1) + pl[:, Lb + 1:].sum(1)
        return flo, np.minimum(1.0, flo * (1.0 + 1e-9) + out * (1.0 + 1e-6) + 1e-15)

    if w_hi <= w_lo:
        flo, fhi = evaluate(np.array([w_lo]))
        return {"ub": float(min(1.0, fhi[0] + 2.0 * n * 1e-15)), "lb": float(flo[0]), "w_lb": w_lo, "n_evals": n_evals,
                "converged": True, "cells": C, "w_set": [w_lo, w_hi]}
    nodes = np.linspace(w_lo, w_hi, k0 + 1)
    flo, fhi = evaluate(nodes)
    ub = np.full(len(nodes), np.nan)          # ub[j]: bound on the subinterval [nodes[j], nodes[j + 1]] (nan: to compute)
    converged = False
    while True:
        todo = np.nonzero(np.isnan(ub[:-1]))[0]
        if len(todo):
            a, b = nodes[todo], nodes[todo + 1]
            c = 0.5 * (a + b)
            la, lb_ = np.log(fhi[todo]), np.log(fhi[todo + 1])
            split = np.maximum(ni_holder_logbound(la, a, c, n, m), ni_holder_logbound(lb_, b, c, n, m))
            whole = np.minimum(ni_holder_logbound(la, a, b, n, m), ni_holder_logbound(lb_, b, a, n, m))
            ub[todo] = np.minimum(1.0, np.exp(np.minimum(split, whole) + 1e-9))
        ub_int = ub[:-1]
        L_best = float(flo.max())
        U = float(max(ub_int.max(), fhi.max()))
        tol = max(abs_tol, rel_tol * L_best)
        active = ub_int > L_best + tol
        if U <= L_best + tol or not active.any():
            converged = True
            break
        if n_evals >= max_evals:
            break
        new = 0.5 * (nodes[:-1][active] + nodes[1:][active])
        nflo, nfhi = evaluate(new)
        ub[:-1][active] = np.nan                                   # both halves of a split interval are recomputed
        nodes, flo, fhi = np.concatenate([nodes, new]), np.concatenate([flo, nflo]), np.concatenate([fhi, nfhi])
        ub = np.concatenate([ub, np.full(len(new), np.nan)])
        o = np.argsort(nodes, kind="stable")
        nodes, flo, fhi, ub = nodes[o], flo[o], fhi[o], ub[o]
    j = int(np.argmax(flo))
    # floating-point end points: the set's ends may sit up to ~1e-16 inside the real ones (e.g. (1 - m)/2); the total-
    # variation bound |f(w) - f(w')| <= 2 n |w - w'| covers such a sliver with 2 n * 1e-15
    U = min(1.0, U + 2.0 * n * 1e-15)
    return {"ub": U, "lb": L_best, "w_lb": float(nodes[j]), "n_evals": n_evals, "converged": converged, "cells": C,
            "w_set": [w_lo, w_hi]}


def ni_bb_nuisance_set(wins: int, losses: int, n: int, margin: float, gamma: float = NI_BB_GAMMA) -> tuple[float, float]:
    """The A6.21 nuisance set: :func:`ni_nuisance_set` (Clopper-Pearson (1 - gamma) set for the discordance rate from
    D = W + L, mapped by w = (rho - m)/2 and clipped to [0, (1 - m)/2]) widened by 1e-9 on each side against
    bisection and rounding error (widening can only raise the p-value). It contains, with probability >= 1 - gamma
    under ANY null point, the boundary point with that point's discordance (step 3 of the proof); when the whole set
    lies below rho = m (no null point is compatible with D) the clipping leaves {0}, a superset of the empty set."""
    lo, hi = ni_nuisance_set(wins, losses, n, margin, gamma)
    return max(0.0, lo - 1e-9), min((1.0 - margin) / 2.0, hi + 1e-9)


def ni_certified_p(wins: int, losses: int, n: int, margin: float, gamma: float = NI_BB_GAMMA, exact_below: int = 5,
                   rel_tol: float = NI_BB_REL_TOL, abs_tol: float = NI_BB_ABS_TOL,
                   max_evals: int = NI_BB_MAX_EVALS) -> dict[str, Any]:
    """The A6.21 decisive p-value of the paired NI test H0: P(loss) - P(win) >= m (rule ``bb_ordered``):

        p = min{1, gamma + U},   U >= sup_{w in C_gamma(D)} P_w[ s~(W', L') <= s~(W, L) + 1e-9 ],

    s~ = :func:`ni_order_table` (one ordering of every outcome, including those where the A6.14 rule switched to the
    loss-count test), C_gamma(D) = :func:`ni_bb_nuisance_set`, U the certified branch-and-bound supremum of
    :func:`ni_tail_sup` (stops when U - L <= max(1e-7, 1e-3 L)). Valid for every n, level and null point (proof in
    the comment block above; docs/PREREGISTRATION.md A6.21 (a)): P_theta(p <= alpha) <= alpha. ``p_lb`` = min(1, gamma + L)
    is the uncertified value (descriptive). Memoised per (wins, losses, n, margin, settings)."""
    if n <= 0 or not 0 < margin < 1:              # undefined (no task, or no positive margin): p = nan, never NI
        nan = float("nan")
        return {"n": n, "wins": wins, "losses": losses, "margin": margin, "rule": "bb_ordered", "p": nan, "p_lb": nan,
                "ub": nan, "lb": nan, "w_lb": nan, "w_set": [nan, nan], "gamma": gamma, "s_obs": nan, "s_nominal": nan,
                "n_evals": 0, "converged": False, "rel_tol": rel_tol, "abs_tol": abs_tol, "exact_below": exact_below}
    r = _ni_certified_cached(int(wins), int(losses), int(n), float(margin), float(gamma), int(exact_below),
                             float(rel_tol), float(abs_tol), int(max_evals))
    return dict(r)


@functools.lru_cache(maxsize=4096)
def _ni_certified_cached(wins: int, losses: int, n: int, margin: float, gamma: float, exact_below: int,
                         rel_tol: float, abs_tol: float, max_evals: int) -> dict[str, Any]:
    if wins < 0 or losses < 0 or wins + losses > n:
        raise ValueError(f"(wins, losses) = ({wins}, {losses}) outside the outcome space of n = {n}")
    st = ni_order_table(n, margin, exact_below)
    s_obs = float(st[wins, losses])
    w_lo, w_hi = ni_bb_nuisance_set(wins, losses, n, margin, gamma)
    sup = ni_tail_sup(st, s_obs + NI_BB_TIE, n, margin, w_lo, w_hi, rel_tol=rel_tol, abs_tol=abs_tol,
                      max_evals=max_evals)
    return {"n": n, "wins": wins, "losses": losses, "margin": margin, "rule": "bb_ordered",
            "p": float(min(1.0, gamma + sup["ub"])), "p_lb": float(min(1.0, gamma + sup["lb"])),
            "ub": sup["ub"], "lb": sup["lb"], "w_lb": sup["w_lb"], "w_set": [w_lo, w_hi], "gamma": gamma,
            "s_obs": s_obs, "s_nominal": float(ni_order_table(n, margin, exact_below, hull=False)[wins, losses]),
            "n_evals": sup["n_evals"], "converged": sup["converged"], "rel_tol": rel_tol, "abs_tol": abs_tol,
            "exact_below": exact_below}


def tango_interval(both: int, only_a: int, only_b: int, neither: int, alpha: float = 0.05,
                   one_sided: bool = False) -> dict[str, Any]:
    """Tango (1998, Stat. Med. 17:891-908) score interval for the difference of paired proportions p_a - p_b (same
    cells as :func:`newcombe_interval`): the set of delta with |Z(delta)| <= z, Z the score statistic of
    :func:`_tango_z` with the restricted MLE of the discordant probability under H0, found by bisection on each side
    of the estimate (only_a - only_b) / n. ``one_sided`` returns the (1 - alpha) lower bound (z = z_{1-alpha}) in
    ``ci[0]`` with ``ci[1]`` = 1. With zero discordant pairs the lower bound is -z^2 / (n + z^2) by hand (0.54 pt at n =
    500, one-sided 95 %), next to the exact 1 - 0.05^(1/n) of :func:`ni_exact_check`. Descriptive (A6 / P0-1)."""
    n = both + only_a + only_b + neither
    nan = float("nan")
    if n <= 0:
        return {"n": 0, "diff": nan, "ci": [nan, nan], "alpha": alpha}
    b, c = only_a, only_b
    d = (b - c) / n
    z = _z(1 - alpha) if one_sided else _z(1 - alpha / 2)

    def solve(lo: float, hi: float, target: float) -> float:
        # Z is decreasing in delta: find delta with Z(delta) = target between lo and hi
        for _ in range(200):
            mid = (lo + hi) / 2
            if _tango_z(mid, b, c, n) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = solve(-1.0, d, z) if _tango_z(-1.0, b, c, n) > z else -1.0
    upper = 1.0 if one_sided else (solve(d, 1.0, -z) if _tango_z(1.0, b, c, n) < -z else 1.0)
    return {"n": n, "diff": d, "ci": [lower, upper], "alpha": alpha, "one_sided": one_sided, "method": "Tango 1998"}


def ni_cells(base: Mapping[str, float], arm: Mapping[str, float]) -> dict[str, int]:
    """Paired 2x2 cells of per-task binary outcomes (task means >= 1 count as success) for the score intervals."""
    keys = sorted(set(base) & set(arm))
    cells = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0}
    for k in keys:
        a, b = float(arm[k]) >= 1.0, float(base[k]) >= 1.0
        cells["both" if a and b else "only_a" if a else "only_b" if b else "neither"] += 1
    return cells


# ------------------------------------------------------------------ H5 readout (hard tasks; pre-registration A5.2)
def sign_test_exact(wins: int, losses: int) -> dict[str, Any]:
    """Exact sign test on paired per-task outcomes, ties dropped: under H0 wins ~ Bin(n, 1/2) with n = wins + losses.
    Returns ``p`` (two-sided: twice the smaller tail, capped at 1), ``p_wins`` = P(X >= wins) (one-sided: the arm
    wins more often than it loses) and ``p_losses`` = P(X >= losses). n = 0 gives every p = 1."""
    wins, losses = int(wins), int(losses)
    n = wins + losses
    if n == 0:
        return {"n": 0, "wins": wins, "losses": losses, "p": 1.0, "p_wins": 1.0, "p_losses": 1.0}
    p_wins = 1.0 - _binom_cdf(wins - 1, n) if wins > 0 else 1.0
    p_losses = 1.0 - _binom_cdf(losses - 1, n) if losses > 0 else 1.0
    return {"n": n, "wins": wins, "losses": losses, "p": min(1.0, 2.0 * _binom_cdf(min(wins, losses), n)),
            "p_wins": min(1.0, p_wins), "p_losses": min(1.0, p_losses)}


def sign_flip_test(d: Sequence[float], n_perm: int = 20000, seed: int = 0, exact_max_n: int = 16) -> dict[str, Any]:
    """One-sided sign-flip (permutation) test of paired differences ``d`` (e.g. per-task Delta U of a gate arm minus
    its matched-rate random gate, PREREGISTRATION A6.16 for C5). Exact when every task's difference is symmetric
    about zero (within-task exchangeability of the two arms), whatever the skew of |d| or the share of zeros. Because
    sum d_i^2 is invariant under sign flips, it is the studentized (paired-t) sign-flip test; for the weak null
    E[d] = 0 with finite variance it is only asymptotically of level alpha (Romano 1990, JASA 85:686-692; Lehmann &
    Romano 2022, Example 17.2.4), and at n = 500 its size is inflated when a small systematic cost difference offsets
    a success deficit (results/reviews/2026-10-02/c5_signflip_size.md; A6.21 (b)). ``p_greater`` = P(mean* >=
    mean_obs) (the arm's utility is higher), ``p_less`` the other side. Zeros contribute nothing to the statistic and
    are kept. Exact enumeration of the 2^n sign patterns for n <= ``exact_max_n``, else ``n_perm`` Monte-Carlo sign
    draws with a fixed seed (the p-value includes the observed pattern: (k + 1) / (B + 1))."""
    x = np.asarray([float(v) for v in d], float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": NAN, "p_greater": 1.0, "p_less": 1.0, "method": "none"}
    obs = float(x.mean())
    if n <= exact_max_n:
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * n), indexing="ij")).reshape(n, -1).T
        means = signs @ x / n
        return {"n": n, "mean": obs, "p_greater": float(np.mean(means >= obs - 1e-12)),
                "p_less": float(np.mean(means <= obs + 1e-12)), "method": "exact", "n_perm": int(len(means))}
    rng = np.random.default_rng(seed)
    ge = le = 0
    chunk = 2000
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        signs = rng.choice([-1.0, 1.0], size=(b, n))
        means = signs @ x / n
        ge += int(np.sum(means >= obs - 1e-12))
        le += int(np.sum(means <= obs + 1e-12))
        done += b
    return {"n": n, "mean": obs, "p_greater": (ge + 1) / (n_perm + 1), "p_less": (le + 1) / (n_perm + 1),
            "method": "monte-carlo", "n_perm": n_perm, "seed": seed}


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta function (modified Lentz; Numerical Recipes 6.4)."""
    fpmin, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > fpmin else fpmin)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > fpmin else fpmin)
            c = 1.0 + aa / c
            c = c if abs(c) > fpmin else fpmin
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def student_t_sf(t: float, df: float) -> float:
    """P(T >= t) for Student's t with ``df`` degrees of freedom (regularized incomplete beta; no scipy)."""
    if math.isnan(t) or df <= 0:
        return NAN
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    if x <= 0.0:
        ib = 0.0
    else:
        bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)) \
            if x < 1.0 else 0.0
        ib = bt * _betacf(a, b, x) / a if x < (a + 1.0) / (a + b + 2.0) else 1.0 - bt * _betacf(b, a, 1.0 - x) / b
    half = 0.5 * ib                                    # P(|T| >= |t|) / 2
    return half if t >= 0 else 1.0 - half


def paired_t_test(d: Sequence[float]) -> dict[str, Any]:
    """One-sided paired t-test of mean(d) > 0 (Student t, n - 1 df): the studentized paired-mean test proposed for
    C5 in A6.21 (b) (asymptotically level alpha for the weak null E[d] = 0 with finite variance, like the sign-flip
    test, but compared with a fixed critical value, so a small systematic component of d cannot break ties on the
    lattice of the success flips; results/reviews/2026-10-02/c5_signflip_size.md). ``p_greater`` / ``p_less``; a
    sample with zero variance gives p = 0.5 when its mean is 0, else 0 or 1 in the direction of the mean."""
    x = np.asarray([float(v) for v in d], float)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else NAN, "t": NAN, "df": n - 1, "p_greater": 1.0, "p_less": 1.0}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd <= 1e-15 * max(1.0, abs(mean)):
        pg = 0.5 if abs(mean) <= 1e-15 else (0.0 if mean > 0 else 1.0)
        return {"n": n, "mean": mean, "t": math.copysign(math.inf, mean) if pg != 0.5 else 0.0, "df": n - 1,
                "p_greater": pg, "p_less": 1.0 - pg if pg != 0.5 else 0.5}
    t = mean / (sd / math.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "t": t, "df": n - 1, "p_greater": student_t_sf(t, n - 1),
            "p_less": student_t_sf(-t, n - 1)}


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """1-based ranks of ``a`` with ties given their average rank."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), float)
    s, i = a[order], 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def wilcoxon_one_sided(d: Sequence[float], exact_max_n: int = 200) -> dict[str, Any]:
    """Wilcoxon (1945) signed-rank test of paired differences ``d`` (e.g. arm time - baseline time per task) against
    a null symmetric about zero. Zeros are dropped (Wilcoxon's rule), tied |d| get average ranks and W+ is the rank
    sum of the positive differences. ``p_less`` = P(W+ <= W+_obs) is the one-sided p-value for *negative* differences
    (the arm is faster), ``p_greater`` = P(W+ >= W+_obs) the other side. For n <= ``exact_max_n`` the p-values come
    from the exact conditional distribution (the 2^n sign assignments of the observed ranks, enumerated by dynamic
    programming over doubled ranks, so ties are handled exactly); beyond it from the normal approximation with the
    tie correction and a continuity correction. ``median`` is the median of ``d`` including zeros (the A5.2 "median
    paired change" criterion)."""
    raw = np.asarray([float(v) for v in d], float)
    nan = float("nan")
    x = raw[raw != 0]
    n = int(len(x))
    med = float(np.median(raw)) if len(raw) else nan
    if n == 0:
        return {"n": 0, "n_zero": int(len(raw)), "w_plus": 0.0, "w_minus": 0.0, "median": med, "p_less": 1.0,
                "p_greater": 1.0, "exact": True}
    ranks = _average_ranks(np.abs(x))
    w_plus, w_minus = float(ranks[x > 0].sum()), float(ranks[x < 0].sum())
    if n <= exact_max_n:
        r2 = np.rint(ranks * 2).astype(int)                     # average ranks are half-integers: double them
        total = int(r2.sum())
        dist = np.zeros(total + 1, float)
        dist[0] = 1.0
        for r in r2:
            nxt = dist.copy()
            nxt[r:] += dist[: total + 1 - r]
            dist = nxt
        dist /= dist.sum()
        w2 = int(round(w_plus * 2))
        p_less, p_greater, exact = float(dist[: w2 + 1].sum()), float(dist[w2:].sum()), True
    else:
        _, counts = np.unique(np.abs(x), return_counts=True)
        mean = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0 - float((counts.astype(float) ** 3 - counts).sum()) / 48.0
        sd = math.sqrt(var) if var > 0 else nan
        if not sd or math.isnan(sd):
            p_less = p_greater = 1.0
        else:
            p_less = 0.5 * math.erfc(-((w_plus + 0.5 - mean) / sd) / math.sqrt(2))      # Phi(z)
            p_greater = 0.5 * math.erfc(((w_plus - 0.5 - mean) / sd) / math.sqrt(2))    # 1 - Phi(z)
        exact = False
    return {"n": n, "n_zero": int(len(raw) - n), "w_plus": w_plus, "w_minus": w_minus, "median": med,
            "p_less": min(1.0, p_less), "p_greater": min(1.0, p_greater), "exact": exact}


def paired_success_lb(base: Mapping[str, float], other: Mapping[str, float], alpha: float = 0.10,
                      n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    """One-sided (1 - alpha) task-bootstrap lower bound of Delta success = mean_t(other_t - base_t) over the tasks
    both arms ran (``base`` / ``other``: task -> per-task success, attempts averaged), with the per-task wins (other
    higher), losses and ties. A5.2 *better*: ``lb > -0.05`` and ``wins >= losses``."""
    keys = sorted(set(base) & set(other))
    d = [float(other[k]) - float(base[k]) for k in keys]
    r = paired_bootstrap_diff(base, other, n=n_boot, alpha=alpha, seed=seed)
    return {"n": len(keys), "diff": r["diff"], "lb": r["lcb"], "alpha": alpha,
            "wins": sum(1 for v in d if v > 0), "losses": sum(1 for v in d if v < 0),
            "ties": sum(1 for v in d if v == 0)}


def h5_verdict(base_success: Mapping[str, float], arm_success: Mapping[str, float],
               base_time: Mapping[str, float], arm_time: Mapping[str, float],
               base_time_solved: Mapping[str, float] | None = None,
               arm_time_solved: Mapping[str, float] | None = None, margin: float = 0.05,
               alpha_better: float = 0.10, alpha_faster: float = 0.05, alpha_sign: float = 0.05,
               n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Pre-registered H5 readout on hard tasks (A5.2, renamed by A6.6, bounded by A6.16 (j)) of one arm against the
    unmodified harness. Exploratory, outside the Holm family.

    Inputs are per-task attempt means: ``*_success`` in [0, 1]; ``*_time`` = agent-run seconds over all attempts with
    timeouts at the cap; ``*_time_solved`` = mean agent-run seconds of the *solved* attempts, present only for tasks
    the arm solved at least once (default: ``*_time`` on tasks with success == 1, exact for k = 1).

    success_preserving (the *exploratory success screen*; ``better`` is the same flag, kept for callers)
        <=> one-sided (1 - alpha_better) task-bootstrap lower bound of Delta success > -margin and wins >= losses.
        Not a size-controlled NI test (A6.16 (j)): with a ceiling reference and loss probability 0.05, zero observed
        losses has probability ~0.36 at n = 20; displayed as "passes the exploratory success screen" (A6.22 (d)).
    strictly_better (A6.6's *better* tier) <=> that lower bound > 0 and the exact one-sided sign test on the
        success wins / losses has p_wins < alpha_sign.
    faster (*faster on co-solved tasks*) <=> on the tasks both arms solve, the median paired Delta time < 0 and the
        exact one-sided sign test on the per-task time differences (zeros dropped) has p < alpha_faster
        (A6.16 (j); A5.2's Wilcoxon signed-rank p is reported descriptively, it is not distribution-free without
        symmetry).
    success_preserving_and_faster (the registered verdict) <=> screen and faster;
    faster_and_better (the stricter tier) <=> strictly_better and faster.
    The all-tasks median Delta time (timeouts at the cap) is reported, never used."""
    if base_time_solved is None:
        base_time_solved = {t: base_time[t] for t in base_time if t in base_success and float(base_success[t]) >= 1.0}
    if arm_time_solved is None:
        arm_time_solved = {t: arm_time[t] for t in arm_time if t in arm_success and float(arm_success[t]) >= 1.0}
    succ = paired_success_lb(base_success, arm_success, alpha=alpha_better, n_boot=n_boot, seed=seed)
    sign = sign_test_exact(succ["wins"], succ["losses"])
    common = sorted(set(base_success) & set(arm_success))
    d_all = [float(arm_time[t]) - float(base_time[t]) for t in common if t in arm_time and t in base_time]
    both = [t for t in common if t in base_time_solved and t in arm_time_solved]
    d_solved = [float(arm_time_solved[t]) - float(base_time_solved[t]) for t in both]
    wil = wilcoxon_one_sided(d_solved)                                   # descriptive since A6.16 (j)
    t_faster = sum(1 for d in d_solved if d < -1e-9)
    t_slower = sum(1 for d in d_solved if d > 1e-9)
    tsign = sign_test_exact(t_faster, t_slower)                          # p_wins = P(X >= t_faster | Bin(n, 1/2))
    nan = float("nan")
    have = bool(succ["n"] > 0 and not math.isnan(succ["lb"]))
    screen = bool(have and succ["lb"] > -margin and succ["wins"] >= succ["losses"])
    strictly = bool(have and succ["lb"] > 0 and sign["p_wins"] < alpha_sign)
    faster = bool(both and wil["median"] < 0 and tsign["p_wins"] < alpha_faster)
    return {"n_tasks": succ["n"], "d_success": succ["diff"], "d_success_lb": succ["lb"], "alpha_better": alpha_better,
            "margin": margin, "wins": succ["wins"], "losses": succ["losses"], "ties": succ["ties"],
            "sign_p": sign["p"], "sign_p_wins": sign["p_wins"], "alpha_sign": alpha_sign,
            "d_time_median_all": float(np.median(d_all)) if d_all else nan,
            "n_both_solved": len(both), "d_time_median_solved": wil["median"],
            "time_faster_tasks": t_faster, "time_slower_tasks": t_slower, "time_sign_p": tsign["p_wins"],
            "wilcoxon_p": wil["p_less"], "wilcoxon_exact": wil["exact"], "alpha_faster": alpha_faster,
            "success_preserving": screen, "better": screen,
            "screen_label": ("passes the exploratory success screen" if screen else
                             "fails the exploratory success screen"),
            "strictly_better": strictly, "faster": faster,
            "success_preserving_and_faster": bool(screen and faster),
            "faster_and_better": bool(strictly and faster),
            # A6.22 (d) 1: displayed labels (the rule and the field names above are unchanged)
            "verdict_label": ("faster and better (faster among jointly solved tasks)" if strictly and faster else
                              "passes the exploratory success screen; faster among jointly solved tasks"
                              if screen and faster else screen_and_time_label(screen, faster))}


def screen_and_time_label(screen: bool, faster: bool) -> str:
    """H5 display label below the registered verdict (A6.22 (d) 1): each part worded on its own."""
    return (("passes" if screen else "fails") + " the exploratory success screen; "
            + ("faster" if faster else "not faster") + " among jointly solved tasks")


# ------------------------------------------------------------------ iso-quality / iso-cost comparisons
def _task_means(xs: Mapping[str, float], keys: Sequence[str]) -> np.ndarray:
    return np.array([float(xs[k]) for k in keys])


def iso_quality_savings(success: Mapping[str, Mapping[str, float]], cost: Mapping[str, Mapping[str, float]],
                        baseline: str, alpha: float = 0.05, margin: float = 0.0, criterion: str = "not_worse",
                        n_boot: int = 2000, seed: int = 0, candidates: Iterable[str] | None = None) -> dict[str, Any]:
    """Iso-quality cost savings: the largest relative cost reduction ``1 - C_a / C_base`` over
    configurations a whose success is statistically indistinguishable from -- or non-inferior to -- the
    baseline (the LLM-only harness), in the spirit of the cost-controlled comparisons of Kapoor et al.
    (2025) and of routing work that reports cost saved at matched quality (Ong et al., 2025).

    success, cost: arm -> task -> per-task mean. Paired over the tasks both arms ran.
    criterion="not_worse"   a qualifies unless a one-sided paired bootstrap finds it worse:
                            p_worse >= alpha (lenient: absence of evidence; report the CI of Delta too).
    criterion="noninferior" a qualifies iff the registered NI decision statistic rejects at the unadjusted level
                            alpha and margin (PREREGISTRATION A6.14 / A6.17 / A6.18 (c): :func:`ni_pvalue`'s ``ni_95``,
                            the Berger-Boos exact unconditional p of Tango's score statistic or the exact loss-count
                            p); the bootstrap lower bound ``lcb`` is reported next to it as descriptive.
    The savings CI is a percentile bootstrap over tasks for the selected arm (conditional on selection)."""
    if criterion not in ("not_worse", "noninferior"):
        raise ValueError("criterion must be 'not_worse' or 'noninferior'")
    per_arm: dict[str, dict[str, Any]] = {}
    arms = [a for a in (candidates or success) if a != baseline and a in success and a in cost]
    for a in arms:
        keys = sorted(set(success[a]) & set(success[baseline]) & set(cost[a]) & set(cost[baseline]))
        if not keys:
            continue
        test = paired_bootstrap_diff({k: success[baseline][k] for k in keys}, {k: success[a][k] for k in keys},
                                     n=n_boot, alpha=alpha, seed=seed)
        ca, cb = _task_means(cost[a], keys), _task_means(cost[baseline], keys)
        savings = 1.0 - ca.mean() / cb.mean() if cb.mean() > 0 else float("nan")
        if criterion == "not_worse":
            ok, ni = test["p_worse"] >= alpha, None
        else:
            ni = ni_pvalue({k: success[baseline][k] for k in keys}, {k: success[a][k] for k in keys}, margin,
                           n_boot=min(n_boot, 200), seed=seed, alpha=alpha)
            ok = bool(ni["ni_95"])
        per_arm[a] = {"n": len(keys), "delta_success": test["diff"], "delta_ci": [test["ci_lo"], test["ci_hi"]],
                      "lcb": test["lcb"], "p_worse": test["p_worse"], "cost_savings": float(savings),
                      "qualifies": bool(ok)}
        if ni is not None:
            per_arm[a].update({"p_decisive": ni["p_decisive"], "rule": ni["rule"]})
    qualified = [a for a, r in per_arm.items() if r["qualifies"] and not math.isnan(r["cost_savings"])]
    out: dict[str, Any] = {"baseline": baseline, "criterion": criterion, "alpha": alpha, "margin": margin,
                           "qualified": sorted(qualified), "per_arm": per_arm, "arm": None,
                           "cost_savings": float("nan"), "cost_savings_ci": [float("nan"), float("nan")]}
    if not qualified:
        return out
    best = max(qualified, key=lambda a: per_arm[a]["cost_savings"])
    keys = sorted(set(success[best]) & set(success[baseline]) & set(cost[best]) & set(cost[baseline]))
    ca, cb = _task_means(cost[best], keys), _task_means(cost[baseline], keys)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = 1.0 - ca[idx].mean(axis=1) / np.maximum(cb[idx].mean(axis=1), 1e-300)
    out.update({"arm": best, "cost_savings": per_arm[best]["cost_savings"],
                "cost_savings_ci": [float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))],
                "delta_success": per_arm[best]["delta_success"], "delta_ci": per_arm[best]["delta_ci"],
                "p_worse": per_arm[best]["p_worse"]})
    return out


def quality_at_iso_cost(success: Mapping[str, Mapping[str, float]], cost: Mapping[str, Mapping[str, float]],
                        baseline: str, budget: float | None = None, tol: float = 0.0, alpha: float = 0.05,
                        n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Quality at iso-cost: the best success among configurations whose mean cost per episode is at most
    the budget (default: the baseline's cost, times 1 + tol), with a paired bootstrap of its difference
    to the baseline, plus the value of the non-decreasing convex hull (RouterBench) at the budget --
    the success reachable by randomly mixing configurations at that spend."""
    mean_c = {a: float(np.mean(list(v.values()))) for a, v in cost.items() if v}
    mean_s = {a: float(np.mean(list(v.values()))) for a, v in success.items() if v}
    arms = [a for a in mean_s if a in mean_c]
    if baseline not in arms:
        raise KeyError(f"baseline {baseline!r} not among the configurations")
    b = mean_c[baseline] if budget is None else float(budget)
    feasible = [a for a in arms if mean_c[a] <= b * (1 + tol) + 1e-15]
    best = max(feasible, key=lambda a: (mean_s[a], -mean_c[a])) if feasible else None
    hull = nondecreasing_convex_hull([(mean_c[a], mean_s[a]) for a in arms])
    out: dict[str, Any] = {"budget": b, "arm": best, "success": mean_s.get(best, float("nan")) if best else float("nan"),
                           "hull_success": _hull_value(hull, b, float("nan")) if hull else float("nan"),
                           "baseline_success": mean_s[baseline]}
    if best is not None and best != baseline:
        t = paired_bootstrap_diff(success[baseline], success[best], n=n_boot, alpha=alpha, seed=seed)
        out.update({"delta_success": t["diff"], "delta_ci": [t["ci_lo"], t["ci_hi"]], "p_better": t["p_better"]})
    else:
        out.update({"delta_success": 0.0 if best == baseline else float("nan"), "delta_ci": [float("nan")] * 2,
                    "p_better": float("nan")})
    return out


# ------------------------------------------------------------------ decision level
def decision_records(trajs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the ``DecisionLog`` entries of trajectories into records carrying ``task_id``, ``config``,
    ``step`` and ``episode_success``."""
    out = []
    for t in trajs:
        for st in t["steps"]:
            for d in st["decisions"]:
                out.append({**d, "task_id": t["task_id"], "config": t["config"], "step": st["step"],
                            "episode_success": bool(t["result"].get("success"))})
    return out


def _norm(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v).strip().lower()


def p_yes_of(r: Mapping[str, Any]) -> float | None:
    """P(yes) of a logged ``Noul`` decision (``s1_dist["yes"]``), or None when it was not logged."""
    d = r.get("s1_dist")
    if isinstance(d, Mapping) and d.get("yes") is not None:
        return float(d["yes"])
    return None


def is_noul_record(r: Mapping[str, Any]) -> bool:
    """A System-1 yes/no decision with a binary gold label (guard / trigger decision points)."""
    return not r.get("options") and isinstance(r.get("gold"), bool) and isinstance(r.get("s1_value"), bool)


def acceptable_of(r: Mapping[str, Any]) -> list[str] | None:
    """The information-feasible acceptable tool set AT(s) logged with a tool_select decision (META_GOLD_SPEC §3.5),
    or None for logs without it (other suites, older runs)."""
    at = (r.get("gold_aux") or {}).get("acceptable_tools")
    return list(at) if isinstance(at, list) else None


def has_gold(r: Mapping[str, Any], gold: str = "acceptable") -> bool:
    """A record carries a usable gold: not doomed / unobserved, and a gold value or (tool_select) an AT set."""
    if r.get("gold_status") in ("doomed", "unobserved", "cf_error"):
        return False
    if gold != "canonical" and r.get("dp") == "tool_select" and acceptable_of(r) is not None:
        return True
    return r.get("gold") is not None


def is_correct(r: Mapping[str, Any], value: Any, gold: str = "acceptable") -> bool | None:
    """Is ``value`` right for record ``r``? tool_select with a logged AT(s): membership in AT (primary; ``gold=
    "canonical"`` uses the canonical next tool instead); a list gold (memory_read): membership; otherwise normalised
    equality. None without a usable gold."""
    if value is None or not has_gold(r, gold):
        return None
    if gold != "canonical" and r.get("dp") == "tool_select":
        at = acceptable_of(r)
        if at is not None:
            return _norm(value) in {_norm(x) for x in at}
    g = r.get("gold")
    if g is None:
        return None
    if isinstance(g, (list, tuple)):
        return _norm(value) in {_norm(x) for x in g}
    return _norm(value) == _norm(g)


def choice_calibration_data(records: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    """(top-label confidence, correct) of System-1 ``Choice`` decisions with a gold label; the executed value is
    the argmax, so the logged confidence refers to it. Correct = :func:`is_correct` (AT(s) for tool selection)."""
    rs = [r for r in records if r.get("options") and r.get("s1_value") is not None and has_gold(r)]
    return ([float(r["s1_conf"]) if r.get("s1_conf") is not None else 1.0 for r in rs],
            [float(bool(is_correct(r, r["s1_value"]))) for r in rs])


def noul_calibration_data(records: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    """(P(yes), 1[gold = yes]) of System-1 ``Noul`` decisions with a binary gold label and a logged
    distribution. Guard / trigger decisions execute the *thresholded* value ``p_yes >= tau``, while the logged
    ``s1_conf`` is ``max(p_yes, 1 - p_yes)``; for tau != 0.5 the two do not refer to the same event, so
    calibration must use P(yes) as a probabilistic forecast of the binary gold instead. Records without a
    logged ``s1_dist`` (older logs) are skipped."""
    rs = [(p_yes_of(r), r) for r in records if is_noul_record(r)]
    rs = [(p, r) for p, r in rs if p is not None]
    return [p for p, _ in rs], [float(bool(r["gold"])) for _, r in rs]


def binary_calibration(p_yes: Sequence[float], gold_yes: Sequence[float], n_bins: int = 10) -> dict[str, float]:
    """Calibration of yes/no forecasts p = P(yes) against the binary gold y: ECE of p vs. y (reliability of
    P(yes), Naeini et al. 2015), Brier score mean (p - y)^2 (Brier 1950), AUROC of p for y (discrimination),
    and the AURC (Geifman et al. 2019) of the argmax answer [p >= 1/2] ranked by max(p, 1 - p)."""
    p, y = np.asarray(p_yes, float), np.asarray(gold_yes, float)
    if len(p) == 0:
        return {"n": 0}
    argmax_ok = ((p >= 0.5) == (y == 1)).astype(float)
    return {"n": len(p), "ece": ece(p, y, n_bins), "brier": brier(p, y), "auroc": auroc(p, y),
            "aurc": aurc(np.maximum(p, 1 - p), argmax_ok), "argmax_acc": float(argmax_ok.mean())}


def decision_table(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per decision point / mode: System-1 accuracy of the *executed* value vs. gold, deferral rate,
    accepted-decision (selective) accuracy, LLM and final accuracy, and calibration (``cal`` names the kind):

    ``top-label``  ``Choice`` decisions: ECE / Brier / AUROC / AURC of the top-label confidence (argmax executed);
    ``p_yes``      ``Noul`` guard / trigger decisions: :func:`binary_calibration` of the logged P(yes) against the
                   binary gold (the executed value is thresholded at tau, so the logged ``s1_conf`` must not be
                   paired with it). Omitted for logs without ``s1_dist``."""
    by_dp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_dp[f"{r['dp']}/{r['mode']}"].append(r)
    table = {}
    for key, rs in sorted(by_dp.items()):
        s1 = [r for r in rs if r.get("s1_value") is not None and has_gold(r)]
        row: dict[str, Any] = {"n": len(rs), "n_gold": len(s1)}
        st = Counter(str(r.get("gold_status")) for r in rs if r.get("gold_status") is not None)
        if st:
            row["gold_status"] = dict(st)
        if s1:
            correct = [float(bool(is_correct(r, r["s1_value"]))) for r in s1]
            row["s1_acc"] = float(np.mean(correct))
            if any(acceptable_of(r) is not None for r in s1):     # canonical-gold column next to the AT one
                can = [float(bool(is_correct(r, r["s1_value"], "canonical"))) for r in s1 if r.get("gold") is not None]
                row["s1_acc_canonical"] = float(np.mean(can)) if can else float("nan")
            conf, corr = choice_calibration_data(s1)
            pys, ys = noul_calibration_data(s1)
            if conf:
                row.update({"cal": "top-label", "ece": ece(conf, corr), "brier": brier(conf, corr),
                            "auroc": auroc(conf, corr), "aurc": aurc(conf, corr)})
            elif pys:
                bc = binary_calibration(pys, ys)
                row.update({"cal": "p_yes", "n_cal": bc["n"], "ece": bc["ece"], "brier": bc["brier"],
                            "auroc": bc["auroc"], "aurc": bc["aurc"]})
            acc_accept = [c for r, c in zip(s1, correct) if not r.get("deferred")]
            row["accepted_acc"] = float(np.mean(acc_accept)) if acc_accept else float("nan")
        row["defer_rate"] = float(np.mean([bool(r.get("deferred")) for r in rs]))
        s2 = [r for r in rs if r.get("s2_value") is not None and has_gold(r) and not isinstance(r.get("gold"), bool)]
        if s2:
            row["s2_acc"] = float(np.mean([bool(is_correct(r, r["s2_value"])) for r in s2]))
        fin = [r for r in rs if r.get("final_value") is not None and has_gold(r)
               and not isinstance(r.get("gold"), bool) and not isinstance(r.get("final_value"), list)
               and r["mode"] not in ("guard", "trigger")]
        if fin:
            row["final_acc"] = float(np.mean([bool(is_correct(r, r["final_value"])) for r in fin]))
        table[key] = row
    return table


OBJECT_DPS = ("tool_select", "tool_params", "action")   # decisions that change the environment
DELEGATING_MODES = ("replace", "cascade")               # modes in which System-1 can act without the LLM


def is_delegated(r: Mapping[str, Any]) -> bool:
    """A decision was *delegated* when System-1's answer was executed without consulting the LLM:
    ``replace`` mode, or ``cascade`` above the threshold (not deferred), at an object-level decision point."""
    return (r.get("dp") in OBJECT_DPS and r.get("mode") in DELEGATING_MODES and not r.get("deferred")
            and r.get("s1_value") is not None)


def delegation_ratio(records: Sequence[Mapping[str, Any]], dps: Sequence[str] = OBJECT_DPS) -> dict[str, Any]:
    """Delegation ratio: the fraction of object-level decisions (tool choice, tool arguments, discrete
    actions) taken by System-1 without the LLM (:func:`is_delegated`). LLM-made decisions count in the
    denominator: ``off`` and ``assist`` records, and each guarded proposal once (retries of one step are
    one decision). Meta-level decisions (trigger / always / llm modes) are excluded.
    Returns ``{"ratio", "n", "n_delegated", "by_dp": {dp: ratio}}``."""
    seen_guard: set[tuple[Any, Any, Any, Any]] = set()
    num: dict[str, int] = defaultdict(int)
    den: dict[str, int] = defaultdict(int)
    for r in records:
        dp, mode = r.get("dp"), r.get("mode")
        # ``generate``: a free-text argument System-2 filled (never delegable; META_GOLD_SPEC §7.3)
        if dp not in dps or mode not in ("off", "shadow", "replace", "cascade", "assist", "guard", "generate"):
            continue
        if mode == "guard":
            key = (r.get("config"), r.get("task_id"), r.get("step"), dp)
            if key in seen_guard:
                continue
            seen_guard.add(key)
        den[dp] += 1
        num[dp] += int(is_delegated(r))
    n, k = sum(den.values()), sum(num.values())
    return {"ratio": k / n if n else float("nan"), "n": n, "n_delegated": k,
            "by_dp": {dp: num[dp] / den[dp] for dp in sorted(den)}}


def llm_free_step_ratio(trajs: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of agent steps in which no System-2 (LLM) call was made at all -- the step-level
    counterpart of the delegation ratio (a delegated tool choice whose free-text argument still needs
    the LLM is not an LLM-free step)."""
    steps = [s for t in trajs for s in t.get("steps", [])]
    return float(np.mean([int(s.get("llm_calls", 0)) == 0 for s in steps])) if steps else float("nan")


def harmful_delegation(records: Sequence[Mapping[str, Any]],
                       episodes: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """Harmful-delegation rate: delegated decisions (:func:`is_delegated`) that were wrong vs. gold *and*
    occurred in an episode that failed, per delegated decision with a gold label. Also returns the plain
    wrong-delegation rate (1 - accepted accuracy) and, given ``episodes`` (task_id -> success), the
    fraction of all episodes / of failed episodes that contain at least one wrong delegated decision
    (an upper bound on the failures attributable to delegation)."""
    deleg = [r for r in records if is_delegated(r) and has_gold(r)]
    wrong = [r for r in deleg if not is_correct(r, r.get("s1_value"))]
    harmful = [r for r in wrong if not r.get("episode_success")]
    out: dict[str, Any] = {"n_delegated": len(deleg), "n_wrong": len(wrong), "n_harmful": len(harmful),
                           "wrong_rate": len(wrong) / len(deleg) if deleg else float("nan"),
                           "harmful_rate": len(harmful) / len(deleg) if deleg else float("nan")}
    if episodes is not None:
        bad = {r.get("task_id") for r in harmful}
        failed = [k for k, ok in episodes.items() if not ok]
        out["harmful_episode_rate"] = len(bad) / len(episodes) if episodes else float("nan")
        out["failure_attribution"] = len(bad & set(failed)) / len(failed) if failed else float("nan")
    return out


def opportunity_key(r: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """The opportunity a yes/no record answers, for counting each one once (round-3 review: a rejected finish used to
    leave System-1's state unchanged, so a deterministic gate re-answered the identical question at every later step
    and one false rejection became ~25 records). Stop-guard records: the proposal (task + question, which quotes the
    answer) -- a resubmission of the same answer is a retry of that proposal; tool-guard records: task + step +
    question; every other record: task + step + decision point + mode + question + state. None (always a distinct
    opportunity) when the record does not log the question / state that identify it."""
    tid, dp, mode, q = r.get("task_id"), r.get("dp"), r.get("mode"), r.get("question")
    if not q:
        return None
    if mode == "guard":
        return (tid, dp, mode, q) if dp == "stop" else (tid, r.get("step"), dp, mode, q)
    st = r.get("state")
    if not isinstance(st, str):
        return None
    return (tid, r.get("step"), dp, mode, q, st)


def first_opportunities(records: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int]:
    """(the first record of every opportunity, in order; the number of retries dropped) -- see
    :func:`opportunity_key`. Rates of the matched-rate protocol, guard FAR / FRR and gate fire rates are computed per
    opportunity; retries are reported separately."""
    seen: set[tuple[Any, ...]] = set()
    out: list[Mapping[str, Any]] = []
    for r in records:
        k = opportunity_key(r)
        if k is not None and k in seen:
            continue
        if k is not None:
            seen.add(k)
        out.append(r)
    return out, len(records) - len(out)


def _guard_gold(r: Mapping[str, Any], gold: str = "progress") -> Any:
    """The guard gold of a record: the logged one (``proposal_progress`` / ``finish_ok`` under META_GOLD_SPEC), or
    with ``gold="canonical"`` the legacy canonical verdict kept in ``gold_aux`` (``canonical`` for the tool guard,
    ``canonical_done`` for the stop guard; pre-spec logs carry only that one as ``gold``)."""
    if gold == "canonical":
        aux = r.get("gold_aux") or {}
        if "canonical" in aux:
            return aux["canonical"]
        if "canonical_done" in aux:
            return aux["canonical_done"]
    return r.get("gold")


def guard_kind(r: Mapping[str, Any]) -> str | None:
    """The stratum of a guarded proposal: the verdict (tool guard: progress / detour / fatal) or the finish kind
    (stop guard: ok / premature / wrong_answer)."""
    aux = r.get("gold_aux") or {}
    return aux.get("kind") or aux.get("verdict")


def _guard_rows(records: Sequence[Mapping[str, Any]], gold: str = "progress") -> dict[str, list[Mapping[str, Any]]]:
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("mode") == "guard" and isinstance(_guard_gold(r, gold), bool) and r.get("s1_value") is not None:
            rows[str(r.get("dp"))].append(r)
    if rows:
        rows["all"] = [r for v in list(rows.values()) for r in v]
    return rows


def _rate(k: int, n: int) -> float:
    return k / n if n else float("nan")


def guard_error_rates(records: Sequence[Mapping[str, Any]], gold: str = "progress") -> dict[str, dict[str, Any]]:
    """Verifier (guard mode) error rates vs. gold at the operating threshold, per decision point and
    pooled ("all"), with the usual verification definitions:
    FAR = P(accept | proposal wrong)  -- false accepts; 1 - FAR is the catch rate,
    FRR = P(reject | proposal right)  -- false rejects, each costing an extra LLM call.
    The gold label is the oracle's verdict on the LLM's proposal (``DecisionLog.gold``: the proposal is progress /
    finishing now with this answer succeeds); ``gold="canonical"`` uses the legacy canonical verdict instead.
    ``by_kind`` gives the accept rate per stratum (verdict / finish kind) and ``far_fatal`` the FAR on fatal
    proposals (the safety-relevant catch rate). Each proposal counts once (:func:`first_opportunities`: a stop-guard
    resubmission of the same answer is a retry, not a new proposal); ``n_retries`` counts the dropped re-queries."""
    out: dict[str, dict[str, Any]] = {}
    for dp, rs_all in _guard_rows(records, gold).items():
        rs, n_retries = first_opportunities(rs_all)
        wrong = [r for r in rs if not _guard_gold(r, gold)]
        right = [r for r in rs if _guard_gold(r, gold)]
        fa = sum(1 for r in wrong if bool(r["s1_value"]))
        fr = sum(1 for r in right if not bool(r["s1_value"]))
        row: dict[str, Any] = {"n": len(rs), "n_wrong": len(wrong), "n_right": len(right),
                               "far": _rate(fa, len(wrong)), "frr": _rate(fr, len(right)), "n_retries": n_retries}
        kinds: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for r in rs:
            if guard_kind(r):
                kinds[str(guard_kind(r))].append(r)
        if kinds:
            row["by_kind"] = {k: {"n": len(v), "accept_rate": _rate(sum(1 for r in v if bool(r["s1_value"])), len(v))}
                              for k, v in sorted(kinds.items())}
            fatal = kinds.get("fatal", [])
            row["far_fatal"] = _rate(sum(1 for r in fatal if bool(r["s1_value"])), len(fatal))
        out[dp] = row
    return out


def rescue_rate(records: Sequence[Mapping[str, Any]], gold: str = "progress") -> dict[str, dict[str, Any]]:
    """Rescue rate of a guard: the fraction of *wrong* LLM proposals that the guard rejected in an episode
    that then succeeded (caught and recovered), per decision point and pooled. Also returns the catch rate
    (= 1 - FAR) and the rescue rate given a catch (did the retry lead to success?); ``by_kind`` per stratum."""
    out: dict[str, dict[str, Any]] = {}
    for dp, rs_all in _guard_rows(records, gold).items():
        rs, _ = first_opportunities(rs_all)          # one row per proposal (retries of a proposal dropped)
        wrong = [r for r in rs if not _guard_gold(r, gold)]
        caught = [r for r in wrong if not bool(r["s1_value"])]
        rescued = [r for r in caught if r.get("episode_success")]
        row: dict[str, Any] = {"n_wrong": len(wrong), "n_caught": len(caught), "n_rescued": len(rescued),
                               "catch_rate": _rate(len(caught), len(wrong)),
                               "rescue_rate": _rate(len(rescued), len(wrong)),
                               "rescue_given_catch": _rate(len(rescued), len(caught))}
        kinds = sorted({str(guard_kind(r)) for r in wrong if guard_kind(r)})
        if kinds:
            row["by_kind"] = {}
            for k in kinds:
                w = [r for r in wrong if str(guard_kind(r)) == k]
                c = [r for r in w if not bool(r["s1_value"])]
                row["by_kind"][k] = {"n_wrong": len(w), "catch_rate": _rate(len(c), len(w)),
                                     "rescue_rate": _rate(sum(1 for r in c if r.get("episode_success")), len(w))}
        out[dp] = row
    return out


# ------------------------------------------------------------------ meta-decision golds (docs/META_GOLD_SPEC.md §9)
FAILING_VERDICTS = frozenset({"fatal", "premature", "wrong_answer"})
NONE_MEMORY = "(none of these memories is relevant)"        # == harness.memory.NONE_MEMORY
GATE_DPS = ("reflection", "routing", "stop", "memory_write")
MIN_POS_NEG = 30                                            # H2 estimability: >= 30 positive and >= 30 negative golds


def wilson_ci(k: float, n: float, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson (1927) score interval of a binomial proportion k / n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def gold_coverage(trajs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per decision point / mode: opportunities with a gold record, labelled ones (``gold_status == "ok"``) and the
    ``gold_status`` counts (doomed, unobserved, cf_error, no_oracle) -- the score card's coverage block (§9.7)."""
    out: dict[str, dict[str, Any]] = {}
    for t in trajs:
        for st in t.get("steps") or []:
            for d in st.get("decisions") or []:
                if d.get("gold_status") is None:
                    continue
                row = out.setdefault(f"{d['dp']}/{d['mode']}", {"n": 0, "n_labelled": 0, "status": Counter()})
                row["n"] += 1
                row["n_labelled"] += int(d.get("gold_status") == "ok" and d.get("gold") is not None)
                row["status"][str(d.get("gold_status"))] += 1
    for row in out.values():
        row["status"] = dict(row["status"])
        row["coverage"] = row["n_labelled"] / row["n"] if row["n"] else float("nan")
    return out


def gate_stats(records: Sequence[Mapping[str, Any]], tau: float = 0.5, guard: bool = False,
               min_pos_neg: int = MIN_POS_NEG) -> dict[str, Any]:
    """H2 regressors of a gate on its binary primary gold (§9.1-9.2): base rate pi, AUROC of the firing score, and at
    the operating tau the TPR / FPR, Youden J = TPR - FPR and accuracy; ECE of the score. A trigger fires iff
    p_yes >= tau and its positive class is gold = True (need); a guard *fires* (rejects) iff p_yes < tau and its
    positive class is a wrong proposal (gold = False), so J = 1 - FAR - FRR. ``estimable`` = at least
    ``min_pos_neg`` positives and negatives (else the cell is "no opportunity")."""
    rs = [(p_yes_of(r), bool(r["gold"])) for r in records
          if isinstance(r.get("gold"), bool) and p_yes_of(r) is not None]
    if not rs:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "estimable": False}
    p = np.array([x for x, _ in rs], float)
    g = np.array([y for _, y in rs], bool)
    score, y = (1.0 - p, ~g) if guard else (p, g)
    fire = (p < tau) if guard else (p >= tau)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    tpr = float(fire[y].mean()) if n_pos else float("nan")
    fpr = float(fire[~y].mean()) if n_neg else float("nan")
    return {"n": len(rs), "n_pos": n_pos, "n_neg": n_neg, "pi": float(y.mean()), "tau": tau, "guard": guard,
            "auroc": auroc(score, y.astype(float)), "tpr": tpr, "fpr": fpr, "J": tpr - fpr,
            "accuracy": float((fire == y).mean()), "fire_rate": float(fire.mean()),
            "ece": ece(score, y.astype(float)), "estimable": n_pos >= min_pos_neg and n_neg >= min_pos_neg}


def cf_efficacy(records: Sequence[Mapping[str, Any]], branch: str = "verdict_alt") -> dict[str, Any]:
    """Counterfactual efficacy of an intervention (§4), from records whose ``gold_aux`` carries the branch verdict
    (``verdict_alt`` reflection, ``verdict_large`` routing, ``verdict_reject`` stop guard), the default action's
    verdict ``verdict`` (v0) and the placebo resample ``verdict_placebo``, weighted by ``cf_weight`` (IPW for the
    sampled negatives):
        e = Pw(v_alt = progress | v0 failing) - Pw(v_placebo = progress | v0 failing)
        h = Pw(v_alt failing | v0 = progress) - Pw(v_placebo failing | v0 = progress)
        delta = Pw(v_placebo != v0)  (the label-noise floor).
    e / h are NaN when a placebo term is missing (``net = False``) or the stratum is empty; ``e_raw`` / ``h_raw`` are
    the gross branch rates.
    ``final_share`` = share of positives whose failing action is a finish (the one-step label = episode outcome)."""
    rows = []
    for r in records:
        a = r.get("gold_aux") or {}
        if a.get("cf_status") == "ok" and a.get(branch) is not None and a.get("verdict") is not None:
            rows.append((a["verdict"], a[branch], a.get("verdict_placebo"), float(a.get("cf_weight", 1.0) or 1.0)))
    out: dict[str, Any] = {"branch": branch, "n": len(rows)}
    if not rows:
        return out

    def wmean(sel: list[tuple[float, bool]]) -> float:
        w = sum(x for x, _ in sel)
        return sum(x for x, b in sel if b) / w if w > 0 else float("nan")

    pos = [x for x in rows if x[0] in FAILING_VERDICTS]
    neg = [x for x in rows if x[0] == "progress"]
    alt_p = wmean([(w, v == "progress") for _, v, _, w in pos])
    alt_h = wmean([(w, v in FAILING_VERDICTS) for _, v, _, w in neg])
    pl_p = wmean([(w, p == "progress") for _, _, p, w in pos if p is not None])
    pl_h = wmean([(w, p in FAILING_VERDICTS) for _, _, p, w in neg if p is not None])
    # e / h are NET of the placebo resample by definition: without a placebo verdict (placebo disabled, or every
    # placebo call failed) they are undefined (NaN) -- the raw branch rates stay available as e_raw / h_raw, labelled
    # as such, and ``net`` says which one a consumer got (round-3 review: NaN used to be replaced by 0, so the VoI
    # prediction and the ceiling silently included the resampling effect the placebo removes)
    net = (math.isnan(alt_p) or not math.isnan(pl_p)) and (math.isnan(alt_h) or not math.isnan(pl_h))
    out.update({"n_pos": len(pos), "n_neg": len(neg), "alt_progress_given_pos": alt_p, "alt_fail_given_neg": alt_h,
                "placebo_progress_given_pos": pl_p, "placebo_fail_given_neg": pl_h,
                "e": alt_p - pl_p, "h": alt_h - pl_h, "net": net,
                "e_raw": alt_p, "h_raw": alt_h,
                "delta": wmean([(w, p != v0) for v0, _, p, w in rows if p is not None]),
                "final_share": (sum(1 for x in pos if x[0] in ("premature", "wrong_answer")) / len(pos)
                                if pos else float("nan"))})
    return out


def voi_identity(n: float, W: float, e: float, h: float, pi: float, J: float, kappa: float = 0.0,
                 dr: float = 0.0, c_f: float = 0.0) -> dict[str, float]:
    """Gate VoI against a random gate at the matched rate (§9.1):
        E[U_gate - U_rand] = n [W (e + h) pi (1 - pi) J - kappa],
    and the realised-rate correction when the random arm fires at r + dr (Corollary 3):
        -n dr [W (pi e - (1 - pi) h) - c_f].  ``gross`` leaves out System-1's own price kappa."""
    gross = n * W * (e + h) * pi * (1 - pi) * J
    corr = -n * dr * (W * (pi * e - (1 - pi) * h) - c_f)
    return {"delta_u": gross - n * kappa, "delta_u_gross": gross, "rate_correction": corr,
            "delta_u_corrected": gross - n * kappa + corr}


def _d_bucket(d: Any) -> str:
    if d is None:
        return "na"
    d = int(d)
    return "1" if d <= 1 else ("2" if d == 2 else ("3-4" if d <= 4 else "5+"))


def cluster_bootstrap_prop(rows: Sequence[tuple[Any, float]], n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0) -> list[float]:
    """Task-clustered percentile-bootstrap (1 - alpha) CI of a proportion: rows are (task, 0/1); tasks are resampled
    with replacement and the proportion is the ratio of the resampled sums (several correlated probes per task --
    forced, transplants, T5 at 25% of steps, T6, natural -- make record-level intervals too narrow). NaN without
    rows or fewer than two tasks (one cluster has no between-task variance to resample: use the Wilson interval
    next to it and say so)."""
    if not rows or len({t for t, _ in rows}) < 2:
        return [float("nan"), float("nan")]
    by: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for t, v in rows:
        c = by[t]
        c[0] += float(v)
        c[1] += 1.0
    k = np.array([c[0] for c in by.values()])
    n = np.array([c[1] for c in by.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), size=(n_boot, len(k)))
    p = k[idx].sum(axis=1) / n[idx].sum(axis=1)
    return [float(np.quantile(p, alpha / 2)), float(np.quantile(p, 1 - alpha / 2))]


def _probe_rates(rows: Sequence[tuple[Any, ...]], alpha: float = 0.05, n_boot: int = 2000,
                 seed: int = 0) -> dict[str, Any]:
    """rows: (gold right?, accepted?[, task]) -> FAR / FRR with **task-clustered bootstrap** 95% CIs (``far_ci`` /
    ``frr_ci``, the pre-registered interval of META_GOLD_SPEC §6.3; rows without a task are their own cluster) and
    the record-level Wilson intervals next to them (``far_ci_wilson`` / ``frr_ci_wilson``, too narrow when several
    probes share a task)."""
    rs = [(r[0], r[1], r[2] if len(r) > 2 else ("row", i)) for i, r in enumerate(rows)]
    wrong = [(t, a) for g, a, t in rs if not g]
    right = [(t, a) for g, a, t in rs if g]
    fa, fr = sum(bool(a) for _, a in wrong), sum(1 for _, a in right if not a)
    return {"n": len(rs), "n_wrong": len(wrong), "n_right": len(right),
            "n_tasks_wrong": len({t for t, _ in wrong}),
            "far": _rate(fa, len(wrong)),
            "far_ci": cluster_bootstrap_prop([(t, float(bool(a))) for t, a in wrong], n_boot, alpha, seed),
            "far_ci_wilson": list(wilson_ci(fa, len(wrong), alpha)),
            "frr": _rate(fr, len(right)),
            "frr_ci": cluster_bootstrap_prop([(t, float(not a)) for t, a in right], n_boot, alpha, seed),
            "frr_ci_wilson": list(wilson_ci(fr, len(right), alpha)), "ci": "task-clustered bootstrap"}


def tost_paired(base: Mapping[str, float], other: Mapping[str, float], margin: float, alpha: float = 0.05,
                n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Equivalence (TOST, Schuirmann 1987) of Delta = mean_t(other_t - base_t) within +-margin by a paired task
    bootstrap: equivalent iff the (1 - 2 alpha) percentile CI lies inside (-margin, margin); ``p`` = max of the two
    one-sided bootstrap p-values P*(Delta* <= -margin), P*(Delta* >= margin). With fewer than 5 discordant tasks the
    bootstrap is degenerate, so equivalence then rests on the upper bound of the discordance rate (|Delta| <= it).
    The pre-registered H2 TOST (±2 pt of success, System-1 gate vs. its matched random gate) is
    ``tost_paired(control_success, gate_success, 0.02)``."""
    keys = sorted(set(base) & set(other))
    nan = float("nan")
    if not keys:
        return {"n": 0, "diff": nan, "ci90": [nan, nan], "margin": margin, "p": nan, "equivalent": False}
    d = np.array([float(other[k]) - float(base[k]) for k in keys])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    p = max(float(np.mean(boot <= -margin)), float(np.mean(boot >= margin)))
    k = int(np.sum(np.abs(d) > 1e-12))
    out = {"n": len(keys), "diff": float(d.mean()), "ci90": [lo, hi], "margin": margin, "alpha": alpha, "p": p,
           "n_discordant": k}
    if k < 5:
        # few discordant tasks: the bootstrap is (near) degenerate -- all-concordant arms give a [0, 0] "CI" at any n.
        # |Delta| <= discordance rate, so equivalence needs its one-sided upper bound below the margin (rule of three:
        # with 0 discordant tasks, n >= 150 for ±2 pt)
        ub = wilson_ci(k, len(keys), 2 * alpha)[1]
        out.update({"discordance_ub": ub, "equivalent": bool(ub < margin), "p": nan})
        return out
    out["equivalent"] = bool(-margin < lo and hi < margin)
    return out


def tost_far_nat_model(rows: Sequence[Mapping[str, Any]], margin: float = 0.10, alpha: float = 0.05,
                       n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The representativeness TOST of META_GOLD_SPEC §6.3: FAR_nat - FAR_model within +-margin (0.10), both computed
    on every task-bootstrap resample (task-clustered; FAR_nat's wrong finishes are a subset of FAR_model's, so the two
    are resampled jointly). ``rows``: :func:`probe_records` stop rows."""
    model_src = ("natural", "forced_premature", "transplant")
    by: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])   # fa_nat, n_nat, fa_model, n_model
    for r in rows:
        if r["dp"] != "stop" or r["gold"] or r["source"] not in model_src:
            continue
        c = by[r["task_id"]]
        c[2] += float(r["accept"])
        c[3] += 1.0
        if r["source"] == "natural":
            c[0] += float(r["accept"])
            c[1] += 1.0
    nan = float("nan")
    a = np.array(list(by.values())) if by else np.zeros((0, 4))
    out: dict[str, Any] = {"margin": margin, "alpha": alpha, "n_nat": int(a[:, 1].sum()) if len(a) else 0,
                           "n_model": int(a[:, 3].sum()) if len(a) else 0}
    if not len(a) or a[:, 1].sum() == 0 or a[:, 3].sum() == 0:
        return {**out, "diff": nan, "ci90": [nan, nan], "p": nan, "equivalent": False,
                "note": "no natural wrong finishes" if len(a) else "no wrong finishes"}
    diff = a[:, 0].sum() / a[:, 1].sum() - a[:, 2].sum() / a[:, 3].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    s = a[idx].sum(axis=1)
    ok = (s[:, 1] > 0) & (s[:, 3] > 0)
    boot = s[ok, 0] / s[ok, 1] - s[ok, 2] / s[ok, 3]
    lo, hi = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    p = max(float(np.mean(boot <= -margin)), float(np.mean(boot >= margin)))
    return {**out, "diff": float(diff), "ci90": [lo, hi], "p": p, "equivalent": bool(-margin < lo and hi < margin)}


def _tau_ref(t: Mapping[str, Any]) -> float:
    """The probes' reference tau of an arm (``meta.config.measure.probes.tau_ref``), default 0.5."""
    pr = ((((t.get("meta") or {}).get("config") or {}).get("measure") or {}).get("probes") or {})
    try:
        return float(pr.get("tau_ref", 0.5))
    except (TypeError, ValueError):
        return 0.5


def probe_records(trajs: Sequence[Mapping[str, Any]], tau: float | None = None,
                  natural_stop: bool = True) -> list[dict[str, Any]]:
    """Flatten the guard probes of measurement arms (``traj.probes``, mode ``guard_probe``) plus -- for the stop guard
    -- the natural finish proposals logged in ``steps[]`` (shadow / guard stop records with a finish ``kind``), as
    rows {dp, source, operator, kind, d_bucket, gold, accept, p_yes, task_id}. ``accept`` = p_yes >= tau; without
    ``tau`` probes AND natural shadow records use the arm's probe ``tau_ref`` (one operating point for FAR_model,
    which pools them); live guard records use their executed decision. A live guard's resubmission of a rejected
    answer is a retry of the same proposal and counts once (:func:`first_opportunities`)."""
    out = []
    for t in trajs:
        tid = t.get("task_id")
        tref = _tau_ref(t)
        for p in t.get("probes") or []:
            if p.get("mode") != "guard_probe" or not isinstance(p.get("gold"), bool):
                continue
            a = p.get("gold_aux") or {}
            py = p_yes_of(p)
            thr = tau if tau is not None else float(a.get("tau_ref", tref))
            acc = (py >= thr) if py is not None else bool(p.get("s1_value"))
            out.append({"dp": p["dp"], "source": a.get("probe_source"), "operator": a.get("operator"),
                        "kind": a.get("kind"), "d_bucket": _d_bucket(a.get("d_before")), "gold": p["gold"],
                        "accept": bool(acc), "p_yes": py, "task_id": tid})
        if not natural_stop:
            continue
        nat = []
        for st in t.get("steps") or []:
            for d in st.get("decisions") or []:
                a = d.get("gold_aux") or {}
                if d.get("dp") != "stop" or d.get("mode") not in ("shadow", "guard") or not a.get("kind") \
                        or not isinstance(d.get("gold"), bool):
                    continue
                nat.append({**d, "task_id": tid, "step": st.get("step")})
        for d in first_opportunities(nat)[0]:
            a = d.get("gold_aux") or {}
            py = p_yes_of(d)
            if d.get("mode") == "guard":
                acc = bool(d.get("s1_value"))
            else:
                acc = (py >= (tau if tau is not None else tref)) if py is not None else bool(d.get("s1_value"))
            out.append({"dp": "stop", "source": "natural", "operator": None, "kind": a.get("kind"),
                        "d_bucket": _d_bucket(a.get("d_before")), "gold": d["gold"], "accept": acc,
                        "p_yes": py, "task_id": tid})
    return out


def probe_error_rates(trajs: Sequence[Mapping[str, Any]], tau: float | None = None,
                      kind_weights: Mapping[str, float] | None = None, alpha: float = 0.05,
                      taus: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9), n_boot: int = 2000,
                      seed: int = 0) -> dict[str, Any]:
    """Guard FAR / FRR from the probe protocol (§6.3), per decision point:

    stop         ``far_model`` (the C6 readout: natural + forced_premature + transplant wrong finishes),
                 ``far_nat`` (natural wrong only), ``frr`` (natural right), ``far_edit`` per operator (T5 / T6,
                 robustness, never pooled), strata by source / kind / D bucket, the style check |FRR(transplant
                 right) - FRR(natural right)| <= 0.05, the representativeness TOST of FAR_nat - FAR_model (±0.10,
                 task-clustered, :func:`tost_far_nat_model`), FAR_model reweighted to the natural kind mix
                 (``kind_weights``, default: the natural wrong finishes' kinds), FAR(tau) / FRR(tau) curves and the
                 AUROC of p_yes;
    tool_select  FAR / FRR against progress over natural + transplant proposals, ``far_fatal`` (every fatal probe:
                 T1-T4 edits and fatal transplants), per operator and per verdict stratum.

    CIs: task-clustered percentile bootstrap (``far_ci`` / ``frr_ci``; pre-registered), Wilson next to them."""
    rows = probe_records(trajs, tau)

    def rates(sel: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return _probe_rates([(r["gold"], r["accept"], r["task_id"]) for r in sel], alpha, n_boot, seed)

    out: dict[str, Any] = {}
    for dp in sorted({r["dp"] for r in rows}):
        R = [r for r in rows if r["dp"] == dp]
        blk: dict[str, Any] = {"n": len(R)}
        blk["by_source"] = {s: rates([r for r in R if r["source"] == s]) for s in sorted({str(r["source"]) for r in R})}
        blk["by_operator"] = {o: rates([r for r in R if r["operator"] == o])
                              for o in sorted({r["operator"] for r in R if r["operator"]})}
        blk["by_kind"] = {k: rates([r for r in R if r["kind"] == k]) for k in sorted({str(r["kind"]) for r in R if r["kind"]})}
        blk["by_d"] = {b: rates([r for r in R if r["d_bucket"] == b and not r["gold"]])
                       for b in sorted({r["d_bucket"] for r in R})}
        model_src = ("natural", "forced_premature", "transplant")
        if dp == "stop":
            model = [r for r in R if r["source"] in model_src]
            blk["far_model"] = rates([r for r in model if not r["gold"]])
            blk["far_nat"] = rates([r for r in R if r["source"] == "natural" and not r["gold"]])
            blk["frr"] = rates([r for r in R if r["source"] == "natural" and r["gold"]])
            blk["far_edit"] = {o: rates([r for r in R if r["operator"] == o])
                               for o in ("T5", "T6") if any(r["operator"] == o for r in R)}
            blk["representativeness"] = tost_far_nat_model(R, 0.10, alpha, n_boot, seed)
            nat_kinds = Counter(r["kind"] for r in R if r["source"] == "natural" and not r["gold"] and r["kind"])
            w = dict(kind_weights) if kind_weights else {k: v / sum(nat_kinds.values()) for k, v in nat_kinds.items()}
            per = {k: rates([r for r in model if r["kind"] == k])["far"] for k in w}
            usable = {k: v for k, v in per.items() if not math.isnan(v)}
            z = sum(w[k] for k in usable)
            blk["far_reweighted"] = sum(w[k] * usable[k] for k in usable) / z if z > 0 else float("nan")
            blk["kind_weights"] = w
        else:
            main = [r for r in R if r["source"] in ("natural", "transplant")]
            blk["headline"] = rates(main)
            fatal = [r for r in R if r["kind"] == "fatal"]
            blk["far_fatal"] = rates(fatal)
        tr = [r for r in R if r["source"] == "transplant" and r["gold"]]
        nr = [r for r in R if r["source"] == "natural" and r["gold"]]
        f_t = _rate(sum(1 for r in tr if not r["accept"]), len(tr))
        f_n = _rate(sum(1 for r in nr if not r["accept"]), len(nr))
        blk["style_check"] = {"frr_transplant_right": f_t, "frr_natural_right": f_n, "diff": f_t - f_n,
                              "ok": (abs(f_t - f_n) <= 0.05) if not (math.isnan(f_t) or math.isnan(f_n)) else None}
        scored = [r for r in R if r["p_yes"] is not None and r["source"] in model_src + ("edit",)]
        if scored:
            py = np.array([r["p_yes"] for r in scored], float)
            g = np.array([r["gold"] for r in scored], bool)
            blk["auroc_p_yes"] = auroc(py, g.astype(float))
            blk["curve"] = [{"tau": t, "far": _rate(int(((py >= t) & ~g).sum()), int((~g).sum())),
                             "frr": _rate(int(((py < t) & g).sum()), int(g.sum()))} for t in taus]
        out[dp] = blk
    return out


def memory_read_hits(records: Sequence[Mapping[str, Any]], ks: Sequence[int] = (1, 3)) -> dict[str, Any]:
    """memory_read decisions with a list gold (§3.7): hit@k (a gold label among the k best-ranked memories, the
    "none" option excluded), MRR of the first gold label in System-1's ranking, the recall hit rate (a gold item
    among the items actually recalled), and none-accuracy (gold "none": the recall came back empty)."""
    rs = [r for r in records if r.get("dp") == "memory_read" and isinstance(r.get("gold"), list) and r.get("s1_dist")
          and r.get("gold_status") in (None, "ok")]
    some = [r for r in rs if r["gold"] != [NONE_MEMORY]]
    none = [r for r in rs if r["gold"] == [NONE_MEMORY]]
    out: dict[str, Any] = {"n": len(rs), "n_relevant": len(some), "n_none": len(none)}
    if some:
        ranks = []
        for r in some:
            order = [o for o, _ in sorted(r["s1_dist"].items(), key=lambda kv: -kv[1]) if o != NONE_MEMORY]
            g = set(r["gold"])
            ranks.append(next((i + 1 for i, o in enumerate(order) if o in g), None))
        for k in ks:
            out[f"hit@{k}"] = float(np.mean([rk is not None and rk <= k for rk in ranks]))
        out["mrr"] = float(np.mean([1.0 / rk if rk else 0.0 for rk in ranks]))
        out["recall_hit"] = float(np.mean([bool(set(r.get("final_value") or []) & set(r["gold"])) for r in some]))
    if none:
        out["none_acc"] = float(np.mean([not (r.get("final_value") or []) for r in none]))
    return out


def s1_ranking(r: Mapping[str, Any]) -> list[str]:
    """System-1's ranking of a Choice record's options: the logged full ranking (``s1_rank``, tournaments) when
    present, else ``s1_dist`` by probability."""
    if isinstance(r.get("s1_rank"), list) and r["s1_rank"]:
        return list(r["s1_rank"])
    return [o for o, _ in sorted((r.get("s1_dist") or {}).items(), key=lambda kv: -kv[1])]


def _ranking_complete(r: Mapping[str, Any], k: int) -> bool:
    """Is the top-k of :func:`s1_ranking` defined? A tournament without a logged ``s1_rank`` keeps only its finalists
    in ``s1_dist``: recall@k with k >= |finalists| would be recall over the finalists, not System-1's top-k."""
    if isinstance(r.get("s1_rank"), list) and r["s1_rank"]:
        return True
    n_opt = len(r.get("options") or [])
    n_dist = len(r.get("s1_dist") or {})
    return n_dist >= n_opt or k < n_dist


def recall_at_k(records: Sequence[Mapping[str, Any]], ks: Sequence[int] = (1, 3, 5), dp: str = "tool_select",
                modes: Sequence[str] | None = None) -> dict[str, Any]:
    """C3 recall@k (§9.5): the share of tool decisions whose top-k of System-1's ranking (:func:`s1_ranking`: the
    logged tournament ranking ``s1_rank``, else ``s1_dist``) contains an acceptable tool, AT(s) (primary), or the
    canonical tool (secondary). Records whose ranking is incomplete at k (an older tournament log with only its
    finalists) are left out of recall@k and counted in ``n_undefined@k`` (round-3 review: with 4 finalists the
    stand-in's "recall@5" was recall over 4 tools). ``offered`` = an acceptable tool among the tools an assist arm
    actually offered System-2 (top-k + ``finish``), with ``offered_k`` its mean size -- report it next to C3.
    Recall@k against AT is the lexical / embedding control's readout next to in-episode success."""
    rs = [r for r in records if r.get("dp") == dp and r.get("s1_dist") and (modes is None or r.get("mode") in modes)
          and r.get("gold_status") not in ("doomed", "unobserved")]
    out: dict[str, Any] = {"n": len(rs)}
    if not rs:
        return out
    at_rs = [r for r in rs if acceptable_of(r) is not None]
    can_rs = [r for r in rs if r.get("gold") is not None]
    for k in ks:
        a_k = [r for r in at_rs if _ranking_complete(r, k)]
        c_k = [r for r in can_rs if _ranking_complete(r, k)]
        if at_rs:
            out[f"recall@{k}"] = (float(np.mean([bool(set(s1_ranking(r)[:k]) & set(acceptable_of(r) or []))
                                                 for r in a_k])) if a_k else float("nan"))
        if can_rs:
            out[f"recall@{k}_canonical"] = (float(np.mean([r["gold"] in s1_ranking(r)[:k] for r in c_k]))
                                            if c_k else float("nan"))
        n_und = len(at_rs or can_rs) - len(a_k if at_rs else c_k)
        if n_und:
            out[f"n_undefined@{k}"] = n_und
    off = [r for r in at_rs if (r.get("gold_aux") or {}).get("offered")]
    if off:
        out["offered_recall"] = float(np.mean([bool(set(r["gold_aux"]["offered"]) & set(acceptable_of(r) or []))
                                               for r in off]))
        out["offered_k"] = float(np.mean([len(r["gold_aux"]["offered"]) for r in off]))
    out["n_acceptable"] = len(at_rs)
    return out


def param_kind_composition(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H3 composition (§7.4): tool_params decisions per fill kind (bounded / copy / free) with the delegation ratio
    and the harmful-delegation rate per kind."""
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("dp") == "tool_params":
            by[str((r.get("gold_aux") or {}).get("param_kind") or "unknown")].append(r)
    out: dict[str, Any] = {}
    for k, rs in sorted(by.items()):
        deleg = [r for r in rs if is_delegated(r)]
        h = harmful_delegation(rs)
        out[k] = {"n": len(rs), "delegated": len(deleg), "delegation_ratio": _rate(len(deleg), len(rs)),
                  "harmful_rate": h["harmful_rate"]}
    return out


# ------------------------------------------------------------------ calibration
def ece(conf: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> float:
    """Expected calibration error (Naeini et al., 2015; Guo et al., 2017) of the top-label confidence with
    ``n_bins`` equal-width bins: sum_m |B_m| / N * |acc(B_m) - conf(B_m)|."""
    conf_a, corr_a = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf_a) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf_a > lo) & (conf_a <= hi) if lo > 0 else (conf_a >= lo) & (conf_a <= hi)
        if m.any():
            total += m.mean() * abs(conf_a[m].mean() - corr_a[m].mean())
    return float(total)


def ece_adaptive(conf: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> float:
    """Top-label ECE with adaptive (equal-mass) bins -- the binning scheme of Nixon et al.'s (2019) ACE, which
    additionally averages over classes: as :func:`ece`, but each bin holds (almost) the same number of
    decisions, avoiding empty / tiny bins when confidences cluster near 1."""
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    if len(c) == 0:
        return float("nan")
    order = np.argsort(c, kind="stable")
    total = 0.0
    for idx in np.array_split(order, min(n_bins, len(c))):
        if len(idx):
            total += len(idx) / len(c) * abs(c[idx].mean() - y[idx].mean())
    return float(total)


def brier(conf: Sequence[float], correct: Sequence[float]) -> float:
    """Brier (1950) score of the confidence as a forecast of correctness, mean_i (c_i - y_i)^2 with y_i = 1 iff
    the decision is right -- the binary ("top-label") Brier score; lower is better, 0.25 = constant 0.5."""
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((c - y) ** 2)) if len(c) else float("nan")


def auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Area under the ROC curve of ``scores`` for binary ``labels`` via the Mann-Whitney U statistic (ties
    count 1/2): the probability that a random correct decision gets a higher confidence than a random wrong
    one (0.5 = uninformative confidence). nan if one class is empty."""
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U with ties counted as 0.5
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def risk_coverage(conf: Sequence[float], correct: Sequence[float]) -> list[tuple[float, float]]:
    """Risk-coverage curve (Geifman & El-Yaniv, 2017): accept the i most confident decisions, i = 1..N, and
    report (coverage i/N, selective risk). Ties are broken by input order; :func:`aurc` and
    :func:`selective_accuracy` resolve ties in expectation instead."""
    order = np.argsort(-np.asarray(conf, float))
    y = np.asarray(correct, float)[order]
    out = []
    for i in range(1, len(y) + 1):
        out.append((i / len(y), float(1 - y[:i].mean())))
    return out


def _risk_curve(conf: Sequence[float], correct: Sequence[float]) -> np.ndarray:
    """Selective risk after accepting the i most confident decisions, i = 1..N (Geifman & El-Yaniv, 2017),
    with ties in confidence resolved in expectation (a tie group is accepted in random order)."""
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-c, kind="stable")
    cs, ys = c[order], y[order]
    _, starts, counts = np.unique(-cs, return_index=True, return_counts=True)   # tie groups, most confident first
    risks = np.empty(len(c))
    errors_before = 0.0
    for s, n in zip(starts, counts):
        errors = n - float(ys[s:s + n].sum())
        j = np.arange(1, n + 1)
        risks[s:s + n] = (errors_before + j * errors / n) / (s + j)
        errors_before += errors
    return risks


def aurc(conf: Sequence[float], correct: Sequence[float]) -> float:
    """Area under the risk-coverage curve (Geifman et al., 2019, arXiv:1805.08206):
    AURC = (1/N) sum_{i=1..N} risk(i), risk(i) = error rate of the i most confident decisions
    (Geifman & El-Yaniv, 2017). Lower is better; it mixes accuracy and ranking quality."""
    if len(conf) == 0:
        return float("nan")
    return float(_risk_curve(conf, correct).mean())


def eaurc(conf: Sequence[float], correct: Sequence[float]) -> float:
    """Excess AURC (Geifman et al., 2019): AURC minus the AURC of an oracle ranking that accepts every
    correct decision first -- the part of AURC due to the confidence ranking, not to accuracy."""
    if len(conf) == 0:
        return float("nan")
    y = np.asarray(correct, float)
    ideal = np.sort(y)[::-1]
    return aurc(conf, correct) - float(np.mean(1.0 - np.cumsum(ideal) / np.arange(1, len(y) + 1)))


def selective_accuracy(conf: Sequence[float], correct: Sequence[float], coverage: float) -> float:
    """Selective accuracy at a coverage (Geifman & El-Yaniv, 2017): accuracy on the ceil(coverage * N) most
    confident decisions -- e.g. the accuracy a cascade gets if it keeps 80% of decisions for System-1."""
    if len(conf) == 0 or not 0 < coverage <= 1:
        return float("nan")
    i = max(1, math.ceil(coverage * len(conf) - 1e-9))
    return 1.0 - float(_risk_curve(conf, correct)[i - 1])


def coverage_at_accuracy(conf: Sequence[float], correct: Sequence[float], target: float) -> float:
    """The largest coverage whose selective accuracy is at least ``target`` (0 if even the single most
    confident decision misses it): how much System-1 can decide alone at a required accuracy."""
    if len(conf) == 0:
        return float("nan")
    acc = 1.0 - _risk_curve(conf, correct)
    ok = np.nonzero(acc >= target - 1e-12)[0]
    return float((ok.max() + 1) / len(acc)) if len(ok) else 0.0


def fmt_row(d: dict[str, float], keys: Sequence[str]) -> str:
    def f(v: Any) -> str:
        if isinstance(v, float):
            return "nan" if math.isnan(v) else (f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
        return str(v)
    return " | ".join(f(d.get(k, float("nan"))) for k in keys)
