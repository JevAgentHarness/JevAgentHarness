"""threshold_theory, PREREGISTRATION A6.22: (b) 5 -- A6.5's sample-size switch of the calibration fit (projection with
>= 30 paired decisions, the marginal prop1 value below 30, tau = 0.9 below 10; the fit's n_paired decides, the choice
is written next to the threshold) -- and (c) H1 clause 2 -- the over-confidence gap on the training split with its
task-bootstrap lower bound and the recalibrated-default regret row. Offline, simulated decision logs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from pytest import approx

from jevharness.bench import threshold_theory as T


def _rec(task: str, step: int, c: float, s1: bool, s2: bool, dc: float = 0.002, **kw: Any) -> dict[str, Any]:
    return {"arm": "a", "task_id": task, "step": step, "mode": "shadow", "shadow": True, "c": c, "s1_ok": s1,
            "s2_ok": s2, "acceptable": True, "final_ok": s2, "s1_cost": 1e-5, "s1_lat_ms": 100.0, "s2_cost": dc,
            "s2_lat_ms": 1000.0, "episode_success": True, **kw}


def _calibrated(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = float(rng.uniform(0.3, 1.0))
        out.append(_rec(f"jevgym-clarify-{5000 + i // 2}", i % 2, c, bool(rng.random() < c), bool(rng.random() < 0.9)))
    return out


# ====================================================================== (b) 5: A6.5's sample-size switch
@pytest.mark.parametrize("n,rule", [(0, "fixed"), (9, "fixed"), (10, "prop1"), (29, "prop1"), (30, "projection"),
                                    (1125, "projection")])
def test_sample_size_rule_thresholds(n, rule):
    assert T.sample_size_rule(n) == rule


def test_fit_keeps_the_projection_with_30_or_more_paired_decisions():
    res = T.fit_calibration_arm(_calibrated(400), n_boot=50, cal_seeds=None)
    assert res["n_paired"] == 400 and res["tau_rule"]["rule"] == "projection"
    assert res["tau_hat"] == res["primary"]["projection"]["tau_hat"] and "tau_hat_projection" not in res["primary"]
    assert all(r["tau_rule"] == "projection" for r in [res["primary"], *res["sensitivity"]])


def test_fit_uses_the_marginal_prop1_value_below_30_paired_decisions():
    res = T.fit_calibration_arm(_calibrated(25), n_boot=50, cal_seeds=None, min_count=5)
    assert res["n_paired"] == 25 and res["tau_rule"]["rule"] == "prop1" and res["tau_rule"]["n_paired"] == 25
    p = res["primary"]
    assert res["tau_hat"] == p["tau_hat"] == approx(p["plug_in"]["prop1"])
    assert p["tau_hat_projection"] == p["projection"]["tau_hat"]          # the replaced value is kept next to it
    for r in res["sensitivity"]:                                           # every row follows the same n_paired
        assert r["tau_rule"] == "prop1" and r["tau_hat"] == approx(r["plug_in"]["prop1"])
    assert "A6.5 sample-size rule" in T.calibration_markdown(res) and "marginal prop1" in T.calibration_markdown(res)


def test_fit_keeps_tau_0_9_below_10_paired_decisions():
    recs = _calibrated(9) + [_rec("jevgym-multi-5100", 0, 0.8, True, None)]    # unpaired: does not count
    res = T.fit_calibration_arm(recs, n_boot=50, cal_seeds=None, min_count=2)
    assert res["n_records"] == 10 and res["n_paired"] == 9
    assert res["tau_rule"]["rule"] == "fixed" and res["tau_hat"] == 0.9 and res["primary"]["tau_hat"] == 0.9
    assert all(r["tau_hat"] == 0.9 for r in res["sensitivity"])


# ====================================================================== (c) H1 clause 2
def _choice_logs(n_tasks: int = 600, per: int = 6, t0: float = 3.0, seed: int = 0, noul: bool = False) -> list[dict]:
    """Two-option Choice decisions: System-1's top label has logit z ~ U(0, 9) (top probability p = sigmoid(z), the
    logged and wire confidence) and is right with probability sigmoid(z / t0) -- over-confident by a temperature t0;
    the LLM is right with q_L = 0.9 and dC / V = 0.1 at V = 0.02, so on calibrated scores Prop. 1's tau* = 0.8."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_tasks):
        for k in range(per):
            z = float(rng.uniform(0.0, 9.0))
            p = 1.0 / (1.0 + math.exp(-z))
            ok = bool(rng.random() < 1.0 / (1.0 + math.exp(-z / t0)))
            out.append(_rec(f"t{t}", k, p, ok, bool(rng.random() < 0.9), s1_value="x", kind="choice",
                            s1_dist={"x": p, "y": 1.0 - p}, gold="x" if ok else "y"))
            if noul:   # yes/no decisions with a huge gap: not Choice decisions, never in the clause-2 gap
                out.append(_rec(f"t{t}", per + k, 0.99, False, True, kind="noul", s1_dist={"yes": 0.99, "no": 0.01},
                                gold=False))
    return out


def test_overconfidence_gap_counts_paired_choice_decisions_only():
    recs = _choice_logs(300, noul=True)
    oc = T.overconfidence_gap(recs, n_boot=300)
    choice = [r for r in recs if r["kind"] == "choice"]
    assert oc["n"] == len(choice) and oc["n_tasks"] == 300
    assert oc["gap"] == approx(np.mean([r["c"] for r in choice]) - np.mean([r["s1_ok"] for r in choice]))
    assert oc["gap"] > 0.1 and oc["lb"] > 0 and oc["over_confident"] is True
    assert oc["ci"][0] <= oc["lb"] <= oc["gap"] <= oc["ci"][1]          # one-sided 95 % LB inside the 95 % CI
    cal = T.overconfidence_gap(_calibrated(2000, seed=4), n_boot=300)
    assert abs(cal["gap"]) < 0.03 and cal["lb"] < 0 and cal["over_confident"] is False
    assert T.overconfidence_gap([], n_boot=10)["over_confident"] is None


def test_recalibration_rescales_the_logged_distribution_and_keeps_the_answer():
    recs = _choice_logs(400)
    train, test = T.split_tasks(recs, 0.3, seed=0)
    out, cal = T.recalibrate(train, test)
    assert cal["per_kind"]["choice"] == approx(3.0, rel=0.15) and cal["n_test_rescaled"] == len(test)
    for r, o in zip(test, out):
        assert o["c_raw"] == r["c"] and o["c"] <= r["c"] + 1e-12 and o["s1_ok"] == r["s1_ok"]
    same, _ = T.recalibrate(train, test, temperatures={"global": 1.0, "per_kind": {}})
    assert [o["c"] for o in same] == approx([r["c"] for r in test])     # T = 1: the logged top probability
    nodist = [{**r, "s1_dist": None} for r in test[:5]]
    kept, cal2 = T.recalibrate(train, nodist)
    assert [o["c"] for o in kept] == [r["c"] for r in nodist] and cal2["n_kept_raw"] == 5


def test_recalibrated_default_returns_to_the_front_for_an_over_confident_system_1():
    res = T.analyze(_choice_logs(), Vs=[0.02], train_frac=0.3, seed=0, n_boot=300, n_perm=50, taus=list(T.H1_TAUS))
    assert res["overconfidence"]["over_confident"] is True
    case = res["by_V"][0]
    c2 = case["h1_clause2"]
    # the default row on raw confidences is exactly the tau_default row of the H1 regret block
    d = case["regret"]["candidates"]["tau_default"]
    assert c2["raw"]["regret"] == approx(d["regret"]) and c2["raw"]["regret_ci"] == approx(d["regret_ci"])
    assert c2["raw"]["tau_best_test"] == case["regret"]["tau_best_test"]
    assert c2["raw"]["regret_ci"][0] > 0                                    # tau = 0.8 is mis-set on raw scores
    assert c2["recalibrated"]["regret_ci"][0] <= 0                          # ... and fine after recalibration
    assert c2["recalibrated"]["tau_best_test"] in (0.7, 0.8, 0.9)
    assert c2["returns_to_front"] is True and c2["over_confident"] is True
    assert c2["dist_t1"]["regret"] == approx(c2["raw"]["regret"])           # wire conf = logged top probability here
    md = T.to_markdown({**res, "meta": {"dp": "tool_select", "arms": ["a"]}})
    assert "H1 clause 2 (A6.22 (c))" in md and "returns to the front: yes" in md


def test_clause2_on_a_calibrated_system_1_is_not_over_confident():
    recs = []
    rng = np.random.default_rng(2)
    for t in range(500):
        for k in range(4):
            z = float(rng.uniform(0.0, 4.0))
            p = 1.0 / (1.0 + math.exp(-z))
            ok = bool(rng.random() < p)
            recs.append(_rec(f"t{t}", k, p, ok, bool(rng.random() < 0.9), s1_dist={"x": p, "y": 1 - p},
                             gold="x" if ok else "y", kind="choice"))
    res = T.analyze(recs, Vs=[0.02], train_frac=0.3, n_boot=200, n_perm=50, taus=list(T.H1_TAUS))
    c2 = res["by_V"][0]["h1_clause2"]
    assert c2["over_confident"] is False and c2["temperatures"]["per_kind"]["choice"] == approx(1.0, abs=0.25)


def test_decision_pairs_carry_the_distribution_gold_and_kind():
    traj = {"task_id": "jevgym-clarify-1", "config": "s", "result": {"success": True}, "calls": [], "steps": [
        {"step": 0, "tool": "get_orders", "decisions": [
            {"dp": "tool_select", "mode": "shadow", "options": ["get_orders", "finish"], "s1_value": "get_orders",
             "s1_conf": 0.9, "s1_dist": {"get_orders": 0.9, "finish": 0.1}, "s2_value": "get_orders",
             "s2_correct": True, "final_value": "get_orders", "gold": "get_orders", "gold_status": "ok",
             "s1_cost_usd": 1e-5, "s2_cost_usd": 1e-3, "shadow": True},
            {"dp": "stop", "mode": "shadow", "options": None, "s1_value": False, "s1_conf": 0.8,
             "s1_dist": {"yes": 0.2, "no": 0.8}, "s2_value": False, "s2_correct": True, "final_value": False,
             "gold": False, "gold_status": "ok", "shadow": True}]}]}
    (ts,) = T.decision_pairs([traj], "tool_select")
    assert ts["kind"] == "choice" and ts["s1_dist"] == {"get_orders": 0.9, "finish": 0.1} and ts["gold"] == "get_orders"
    (st,) = T.decision_pairs([traj], "stop")
    assert st["kind"] == "noul" and st["gold"] is False
