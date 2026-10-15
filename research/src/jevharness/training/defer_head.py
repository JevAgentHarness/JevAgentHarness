"""Exploratory row E18 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E17 and E18: read-out ablation and a
deferral-aware objective", (c)): the deferral target and the small defer head on top of a frozen scoring head.

Labelled states are the decisions of a calibration shadow arm (the LLM acts, the gold is logged) on which a reference
decision model has answered as well. The target of a labelled question is the option that minimises the plug-in
expected loss of the harness's accept-or-defer rule (Proposition 1), in the style of Mozannar & Sontag's learning to
defer: defer (d = 1) iff the LLM is right and the reference head is wrong, or the head's margin (top minus second
probability) is below the cost margin k of the decision point; else answer (d = 0).

The defer head is a logistic regression P(d = 1 | x) = sigmoid(w . f(x)) on
f(x) = [1, log10(1 - p1) clipped below at -15, 1[tool choice], ln(number of options)] (``clef_engine.head_features``),
fitted by Newton's method on the L2-penalised negative log-likelihood (lambda = 1 on the non-intercept weights). The
rule is "defer iff P(d = 1 | x) > k"; served as confidence = 1 - P with tau = 1 - k. numpy only; no model call.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..bench.metrics import has_gold, is_correct
from .clef_engine import head_features

DPS = ("tool_select", "tool_params")
# the cost margins k = Delta c / V (V = W = $1) of the frozen 4B's calibration fits (results/gpu_round5/cal/)
K_MARGIN = {"tool_select": 0.0058322, "tool_params": 0.0034651068524096384}
LAMBDA = 1.0


def key_of(task_id: str, step: Any, i: int) -> str:
    return f"{task_id}|{step}|{i}"


def _jsonl(p: Path) -> list[dict[str, Any]]:
    import gzip

    op = gzip.open if str(p).endswith(".gz") else open
    out = []
    with op(p, "rt") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    for r in _jsonl(path):
        if isinstance(r, dict) and "task_id" in r:
            last[str(r["task_id"])] = r
    return last


def margin_of(dist: Mapping[str, float] | None) -> float:
    ps = sorted((float(v) for v in (dist or {}).values()), reverse=True)
    return (ps[0] - (ps[1] if len(ps) > 1 else 0.0)) if ps else 0.0


def defer_target(llm_ok: bool, ref_ok: bool, ref_margin: float, k: float) -> int:
    """d = 1 iff the LLM is right and (the reference head is wrong or its margin is below the cost margin)."""
    return int(bool(llm_ok) and (not ref_ok or ref_margin < k))


def labelled_questions(shadow_arm: Path, ref_answers: Path, k: Mapping[str, float] = K_MARGIN
                       ) -> list[dict[str, Any]]:
    """One row per tool_select / tool_params decision of the shadow arm with a usable gold, the LLM's answer and the
    reference model's answer (keyed ``task|step|i`` as ``scripts/e5_cal_replay.py``): the correctness of both, the
    reference margin, the features and the target."""
    ans = {r["key"]: r for r in _jsonl(ref_answers)}
    rows: list[dict[str, Any]] = []
    for tid, r in sorted(load_latest(shadow_arm).items()):
        if r.get("error"):
            continue
        for s in r.get("steps") or []:
            for i, d in enumerate(s.get("decisions") or []):
                dp = d.get("dp")
                if dp not in DPS or d.get("s1_value") is None or not d.get("options"):
                    continue
                a = ans.get(key_of(tid, s.get("step"), i))
                if not has_gold(d) or d.get("s2_value") is None or a is None or a.get("error") or a.get("value") is None:
                    continue
                llm_ok = bool(is_correct(d, d["s2_value"]))
                ref_ok = bool(is_correct(d, a["value"]))
                dist = a.get("dist") or {}
                p1 = max((float(v) for v in dist.values()), default=float(a.get("conf") or 0.0))
                mg = margin_of(dist)
                rows.append({"key": key_of(tid, s.get("step"), i), "task_id": tid, "dp": dp, "n_options": len(d["options"]),
                             "gold": d.get("gold"), "llm_value": d["s2_value"], "ref_value": a["value"],
                             "llm_ok": llm_ok, "ref_ok": ref_ok, "ref_p1": p1, "ref_margin": mg,
                             "features": head_features(p1, dp == "tool_select", len(d["options"])),
                             "d": defer_target(llm_ok, ref_ok, mg, float(k[dp]))})
    return rows


def fit_logistic(F: np.ndarray, d: np.ndarray, lam: float = LAMBDA, iters: int = 100, tol: float = 1e-10
                 ) -> dict[str, Any]:
    """Newton's method on L(w) = -sum[d log p + (1 - d) log(1 - p)] + lam / 2 * ||w[1:]||^2 (the intercept free)."""
    F = np.asarray(F, dtype=float)
    d = np.asarray(d, dtype=float)
    n, m = F.shape
    w = np.zeros(m)
    reg = np.full(m, float(lam))
    reg[0] = 0.0

    def loss(w: np.ndarray) -> float:
        z = F @ w
        return float(np.sum(np.logaddexp(0.0, z) - d * z) + 0.5 * np.sum(reg * w * w))

    prev = loss(w)
    for it in range(iters):
        z = F @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = F.T @ (p - d) + reg * w
        H = (F * (p * (1 - p))[:, None]).T @ F + np.diag(reg)
        step = np.linalg.solve(H + 1e-9 * np.eye(m), g)
        t = 1.0
        while t > 1e-6:   # backtracking keeps Newton monotone on the few positives
            cand = w - t * step
            if loss(cand) <= prev + 1e-12:
                break
            t /= 2
        w = cand
        cur = loss(w)
        if abs(prev - cur) < tol:
            prev = cur
            break
        prev = cur
    p = 1.0 / (1.0 + np.exp(-(F @ w)))
    return {"weights": [float(x) for x in w], "lambda": float(lam), "iterations": it + 1, "loss": prev,
            "log_loss_mean": float(np.mean(-(d * np.log(np.clip(p, 1e-15, 1)) + (1 - d) * np.log(np.clip(1 - p, 1e-15, 1))))),
            "n": int(n), "positives": int(d.sum()), "p_mean": float(p.mean()), "p_positives_mean": float(p[d == 1].mean()) if d.sum() else None}


def p_defer(weights: Sequence[float], features: Sequence[float]) -> float:
    z = sum(float(w) * float(x) for w, x in zip(weights, features))
    return 1.0 / (1.0 + math.exp(-z))


def fit_head(rows: Sequence[Mapping[str, Any]], k: Mapping[str, float] = K_MARGIN, lam: float = LAMBDA) -> dict[str, Any]:
    """The head record (what the server loads): weights, the data counts, the margins, and the rule's in-sample
    deferral rate per decision point."""
    F = np.array([r["features"] for r in rows], dtype=float)
    d = np.array([r["d"] for r in rows], dtype=float)
    fit = fit_logistic(F, d, lam)
    by_dp: dict[str, Any] = {}
    for dp in DPS:
        sub = [r for r in rows if r["dp"] == dp]
        if not sub:
            continue
        ps = [p_defer(fit["weights"], r["features"]) for r in sub]
        defer = [p > k[dp] for p in ps]
        by_dp[dp] = {"n": len(sub), "positives": int(sum(r["d"] for r in sub)), "k": float(k[dp]),
                     "llm_right": int(sum(r["llm_ok"] for r in sub)), "ref_right": int(sum(r["ref_ok"] for r in sub)),
                     "deferred_in_sample": int(sum(defer)), "coverage_in_sample": 1.0 - sum(defer) / len(sub),
                     "p_quantiles": {q: float(np.quantile(ps, float(q))) for q in ("0.05", "0.5", "0.95", "0.99")},
                     "positives_deferred": int(sum(1 for r, f in zip(sub, defer) if r["d"] and f))}
    return {"registered": "PREREG_A7_STAGE2 Deviations 2026-10-08 'E17 and E18' (c) 3", "features":
            ["1", "log10(1 - p1) clipped at -15", "1[tool choice]", "ln(n options)"], "rule": "defer iff P(d=1|x) > k_dp; "
            "served confidence = 1 - P, tau = 1 - k", "k": {dp: float(k[dp]) for dp in DPS}, "fit": fit,
            "weights": fit["weights"], "by_dp": by_dp}


def head_coverage(answers: Iterable[Mapping[str, Any]], weights: Sequence[float], k: float, tool_select: bool = True
                  ) -> list[dict[str, Any]]:
    """The rule applied to logged replay answers (``confidence`` = the served top probability, ``n_options``):
    each answer with its P(defer) and whether the rule accepts it."""
    out = []
    for a in answers:
        if a.get("error") or a.get("confidence") is None:
            out.append({**a, "p_defer": None, "accept": False})
            continue
        p = p_defer(weights, head_features(float(a["confidence"]), tool_select, int(a.get("n_options") or 1)))
        out.append({**a, "p_defer": p, "accept": p <= k})
    return out
