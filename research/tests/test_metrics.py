"""Hand-computed checks for every JevAgentBench metric, the score-card CLI on a tiny simulated run, and
JevGym v2 (near-duplicate distractors, the ``clarify`` and ``multi`` families). Offline and fast."""

from __future__ import annotations

import json
import math

import pytest

from jevharness.bench import metrics as M
from jevharness.bench.scorecard import compute_scorecard, detect_sweeps, main as scorecard_main, task_key
from jevharness.bench.suites import load_suite
from jevharness.bench.suites.jevgym import (
    ALL_FAMILIES, DISTRACTOR_NAMES, NEAR_TOOLS, JevGymEnv, make_distractors,
)
from jevharness.decision.simulated import OracleBackend
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.llm.simulated import SimulatedLLM

approx = pytest.approx


# ====================================================================== efficiency
def test_cost_and_latency_of_pass():
    assert M.cost_of_pass([1, 2, 3, 4], [1, 0, 1, 0]) == approx(5.0)          # E[c] / E[s] = 2.5 / 0.5
    assert M.cost_of_pass([1, 1], [0, 0]) == math.inf
    assert math.isnan(M.cost_of_pass([], []))
    assert M.latency_of_pass([10, 20], [1, 0]) == approx(30.0)                # 15 s / 0.5


def test_frontier_cost_of_pass():
    per_arm = {"A": {"t1": [(1, 1), (1, 0)], "t2": [(1, 0)]},                # CoP: t1 = 2, t2 = inf
               "B": {"t1": [(3, 1)], "t2": [(2, 1), (2, 1)]}}                  # CoP: t1 = 3, t2 = 2
    r = M.frontier_cost_of_pass(per_arm)
    assert r["value"] == approx(2.0) and r["per_task"] == {"t1": 2.0, "t2": 2.0}
    assert r["argmin_counts"] == {"A": 1, "B": 1} and r["n_unsolved"] == 0
    r = M.frontier_cost_of_pass({"A": {"t1": [(1, 1)], "t2": [(1, 0)]}})
    assert r["value"] == math.inf and r["n_unsolved"] == 1
    r = M.frontier_cost_of_pass({"A": {"t1": [(1, 1)], "t2": [(1, 0)]}}, baseline_cop=10.0)   # human expert
    assert r["value"] == approx((1 + 10) / 2)


def test_effectiveness_under_budget():
    # (1 / (N B)) sum_i y_i max(0, B - r_i) = (1.5 + 1.0 + 0) / (3 * 2)
    assert M.effectiveness_under_budget([0.5, 1.0, 3.0], [1, 1, 1], 2.0) == approx(2.5 / 6)
    assert M.effectiveness_under_budget([0.5], [0], 2.0) == 0.0


# ====================================================================== reliability
def test_pass_hat_k_and_pass_at_k():
    assert M.pass_hat_k(4, 2, 2) == approx(1 / 6)                             # C(2,2) / C(4,2)
    assert M.pass_hat_k(4, 4, 3) == 1.0 and M.pass_hat_k(4, 2, 3) == 0.0
    assert M.pass_at_k(4, 2, 2) == approx(5 / 6)                              # 1 - C(2,2) / C(4,2)
    assert M.pass_hat_k(5, 3, 1) == approx(0.6) == M.pass_at_k(5, 3, 1)       # k = 1: success rate
    with pytest.raises(ValueError):
        M.pass_hat_k(2, 1, 3)
    curve = M.pass_hat_k_curve({"t1": [1, 1, 0, 0], "t2": [1, 1, 1, 1]})
    assert curve[1] == approx(0.75) and curve[2] == approx((1 / 6 + 1) / 2) and curve[4] == approx(0.5)
    assert M.pass_hat_k_curve({"t1": [1, 0]}, estimator="at")[2] == 1.0


# ====================================================================== routing (RouteLLM / RouterBench)
def test_pgr_apgr_cpt():
    assert M.pgr(0.8, 0.6, 1.0) == approx(0.5)
    assert math.isnan(M.pgr(0.8, 0.7, 0.7))
    # one configuration at 50% strong calls recovering half the gap: the interpolation line -> APGR 0.5
    assert M.apgr([(0.5, 0.8)], weak=0.6, strong=1.0) == approx(0.5)
    # full recovery with 20% strong calls: area under (0,0)-(0.2,1)-(1,1) = 0.1 + 0.8
    assert M.apgr([(0.2, 1.0)], weak=0.6, strong=1.0) == approx(0.9)
    # ... identical to RouteLLM's (AUC_router - AUC_weak) / (AUC_strong - AUC_weak) over the same grid
    auc_router = 0.2 * (0.6 + 1.0) / 2 + 0.8 * 1.0
    assert M.apgr([(0.2, 1.0)], 0.6, 1.0) == approx((auc_router - 0.6) / (1.0 - 0.6))
    assert M.cpt([(0.2, 1.0)], 0.6, 1.0, 0.5) == approx(0.1)
    assert M.cpt([(0.2, 1.0)], 0.6, 1.0, 0.8) == approx(0.16)
    assert M.cpt([(0.5, 0.8)], 0.6, 1.0, 0.8) == approx(0.8)
    assert math.isnan(M.cpt([(0.5, 0.7)], 0.6, 1.0, 0.5, anchors=False))      # PGR 0.25 never reaches 0.5


def test_nondecreasing_convex_hull_and_aiq():
    pts = [(1, 0.5), (2, 0.9), (3, 0.8), (4, 0.95), (1.5, 0.55), (5, 0.9)]
    hull = M.nondecreasing_convex_hull(pts)
    assert hull == [(1.0, 0.5), (2.0, 0.9), (4.0, 0.95)]       # (1.5,.55), (3,.8) under the hull; (5,.9) cut
    # area 1..2: 0.7, 2..4: 1.85, 4..5 flat: 0.95 -> 3.5 / 4
    assert M.aiq(pts, x_range=(1, 5)) == approx(0.875)
    assert M.aiq(pts, x_range=(0, 5)) == approx(3.5 / 5)       # budgets below the cheapest config score 0
    # zero router: random mixing of two configurations = the straight line between them
    assert M.aiq([(0, 0.6), (1, 1.0)], x_range=(0, 1)) == approx(0.8)


# ====================================================================== Pareto / hypervolume
def test_dominance_pareto_hypervolume():
    pts = {"A": (1, 0.5), "B": (2, 0.9), "C": (3, 0.8), "D": (1, 0.5)}
    assert M.dominates(pts["B"], pts["C"]) and not M.dominates(pts["A"], pts["D"])
    assert sorted(M.pareto_front([(k, c, s) for k, (c, s) in pts.items()])) == ["A", "B", "D"]
    dc = M.dominance_counts(pts)
    assert dc["C"] == {"dominates": 0, "dominated_by": 1} and dc["B"] == {"dominates": 1, "dominated_by": 0}
    ref = (4, 0.0)
    assert M.hypervolume_2d([(1, 0.5), (2, 0.9)], ref) == approx(1 * 0.5 + 2 * 0.9)
    assert M.hypervolume_2d([(1, 0.5), (2, 0.9)], ref, normalize=True) == approx(2.3 / 3)
    assert M.hypervolume_2d([(5, 1.0)], ref) == 0.0                           # not better than the reference
    contrib = M.hypervolume_contributions({"A": (1, 0.5), "B": (2, 0.9), "C": (3, 0.8)}, ref)
    # without A: B alone = 1.8; without B: A (1..3 at .5) + C (3..4 at .8) = 1.8; C is dominated
    assert contrib["A"] == approx(2.3 - 1.8) and contrib["B"] == approx(2.3 - 1.8) and contrib["C"] == 0.0
    contrib = M.hypervolume_contributions({"A": (1, 0.5), "B": (2, 0.9)}, ref)
    assert contrib["A"] == approx(2.3 - 1.8) and contrib["B"] == approx(2.3 - 1.5)
    # log-cost axis: decades instead of dollars
    assert M.hypervolume_2d([(0.01, 1.0)], (1.0, 0.0), log_cost=True) == approx(2.0)


# ====================================================================== statistics
def test_bootstrap_and_paired_bootstrap():
    assert M.bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    base = {f"t{i}": 1.0 for i in range(10)}
    worse = {k: 0.0 for k in base}
    r = M.paired_bootstrap_diff(base, worse)
    assert r["diff"] == -1.0 and r["ci_lo"] == r["ci_hi"] == -1.0 and r["p_worse"] == 0.0 and r["p_better"] == 1.0
    r = M.paired_bootstrap_diff(base, dict(base))
    assert r["diff"] == 0.0 and r["p_worse"] == 1.0
    assert M.paired_bootstrap(base, worse)["diff"] == -1.0


def test_mcnemar_exact_and_chi2():
    a = {f"e{i}": True for i in range(10)}
    b = {k: i >= 6 for i, k in enumerate(a)}                   # b fails 6 episodes a solved: n01=6, n10=0
    r = M.mcnemar(a, b)
    assert (r["n01"], r["n10"], r["exact"]) == (6, 0, True)
    assert r["p"] == approx(2 / 64) and r["p_b_worse"] == approx(1 / 64) and r["p_b_better"] == 1.0
    a2 = {f"e{i}": i < 20 for i in range(25)}                  # n01 = 20, n10 = 5 -> chi2 = (15-1)^2/25
    b2 = {f"e{i}": i >= 20 for i in range(25)}
    r = M.mcnemar(a2, b2)
    assert not r["exact"] and r["stat"] == approx(7.84) and r["p"] == approx(0.005110, abs=2e-5)
    assert M.mcnemar(a, dict(a))["p"] == 1.0


def test_holm_and_benjamini_hochberg():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")}
    h = M.holm(p)
    assert h["a"] == approx(0.03) and h["c"] == approx(0.06) and h["b"] == approx(0.06) and math.isnan(h["d"])
    bh = M.benjamini_hochberg(p)
    assert bh["a"] == approx(0.03) and bh["c"] == approx(0.04) and bh["b"] == approx(0.04)


# ====================================================================== iso-quality / iso-cost
def _per_task(v: float, n: int = 40) -> dict[str, float]:
    return {f"t{i}": v for i in range(n)}


def test_iso_quality_savings():
    succ = {"llm": _per_task(1.0), "cheap_same": _per_task(1.0), "pricier": _per_task(1.0),
            "cheaper_worse": {f"t{i}": float(i >= 20) for i in range(40)}}
    cost = {"llm": _per_task(1.0), "cheap_same": _per_task(0.25), "pricier": _per_task(2.0),
            "cheaper_worse": _per_task(0.1)}
    r = M.iso_quality_savings(succ, cost, "llm")
    assert r["arm"] == "cheap_same" and r["cost_savings"] == approx(0.75)
    assert r["cost_savings_ci"] == approx([0.75, 0.75]) and "cheaper_worse" not in r["qualified"]
    assert set(r["qualified"]) == {"cheap_same", "pricier"} and r["per_arm"]["pricier"]["cost_savings"] == approx(-1)
    # A6.18 (c): `noninferior` is the registered decision statistic at the unadjusted level, not the bootstrap LCB: with
    # zero discordance the exact loss-count test needs (1 - m)^n < 0.05, i.e. n >= 59 at m = 0.05 -- at n = 40 a
    # zero-width bootstrap interval would have qualified the arm, the registered rule does not
    r = M.iso_quality_savings(succ, cost, "llm", criterion="noninferior", margin=0.05)
    assert r["arm"] is None and r["per_arm"]["cheap_same"]["rule"] == "bb_ordered"     # A6.21 decides every outcome
    succ80 = {k: _per_task(1.0, 80) for k in ("llm", "cheap_same", "pricier")}
    succ80["cheaper_worse"] = {f"t{i}": float(i >= 40) for i in range(80)}
    cost80 = {"llm": _per_task(1.0, 80), "cheap_same": _per_task(0.25, 80), "pricier": _per_task(2.0, 80),
              "cheaper_worse": _per_task(0.1, 80)}
    r = M.iso_quality_savings(succ80, cost80, "llm", criterion="noninferior", margin=0.05)
    # A6.21: zero discordance -> the certified p, at least the loss-count tail (1 - m)^n plus gamma
    p80 = r["per_arm"]["cheap_same"]["p_decisive"]
    assert r["arm"] == "cheap_same" and p80 == approx(M.ni_certified_p(0, 0, 80, 0.05)["p"]) and p80 >= 0.95 ** 80 + 1e-4
    assert "cheaper_worse" not in r["qualified"]
    assert M.iso_quality_savings(succ80, cost80, "llm", criterion="noninferior", margin=0.0)["arm"] is None


def test_quality_at_iso_cost():
    succ = {"base": _per_task(0.5, 4), "X": _per_task(0.8, 4), "Y": _per_task(0.9, 4)}
    cost = {"base": _per_task(1.0, 4), "X": _per_task(0.5, 4), "Y": _per_task(2.0, 4)}
    r = M.quality_at_iso_cost(succ, cost, "base")
    assert r["arm"] == "X" and r["success"] == approx(0.8) and r["delta_success"] == approx(0.3)
    assert r["hull_success"] == approx(0.8 + 0.1 * 0.5 / 1.5)                # mix of X and Y at $1
    assert M.quality_at_iso_cost(succ, cost, "base", budget=3.0)["arm"] == "Y"


# ====================================================================== calibration / selective prediction
def test_ece_brier_auroc():
    conf, corr = [0.9, 0.9, 0.6, 0.6], [1, 0, 1, 1]
    assert M.ece(conf, corr) == approx(0.5 * 0.4 + 0.5 * 0.4)
    assert M.ece_adaptive(conf, corr, n_bins=2) == approx(0.4)
    assert M.brier(conf, corr) == approx((0.01 + 0.81 + 0.16 + 0.16) / 4)
    assert M.auroc(conf, corr) == approx(0.5 / 3)                            # one tie, two losses
    assert M.auroc([0.9, 0.1], [1, 0]) == 1.0 and math.isnan(M.auroc([0.5], [1]))


def test_aurc_selective_accuracy_coverage():
    conf, corr = [0.9, 0.8, 0.7, 0.6], [1, 0, 1, 1]
    # selective risks at coverage 1/4..4/4: 0, 1/2, 1/3, 1/4
    assert M.aurc(conf, corr) == approx((0 + 0.5 + 1 / 3 + 0.25) / 4)
    assert M.eaurc(conf, corr) == approx((0 + 0.5 + 1 / 3 + 0.25) / 4 - 0.25 / 4)
    assert M.selective_accuracy(conf, corr, 0.5) == approx(0.5)
    assert M.selective_accuracy(conf, corr, 0.75) == approx(2 / 3)
    assert M.coverage_at_accuracy(conf, corr, 0.75) == 1.0
    assert M.coverage_at_accuracy(conf, corr, 0.8) == 0.25
    assert M.aurc([0.5] * 4, corr) == approx(0.25)                          # ties resolved in expectation
    assert M.risk_coverage(conf, corr)[1] == (0.5, 0.5)


# ====================================================================== decision level
def _rec(dp, mode, s1=None, gold=None, deferred=False, task="t1", step=0, ok=True, options=None):
    return {"dp": dp, "mode": mode, "s1_value": s1, "gold": gold, "deferred": deferred, "task_id": task,
            "config": "arm", "step": step, "episode_success": ok, "options": options, "s1_conf": 0.9}


def test_delegation_ratio_and_harmful_delegation():
    recs = [_rec("tool_select", "replace", "a", "a", step=0),                 # delegated
            _rec("tool_select", "cascade", "a", "a", deferred=True, step=1),  # deferred -> LLM
            _rec("tool_params", "cascade", "x", "x", step=1),                 # delegated
            _rec("tool_select", "off", None, "a", step=2),                    # LLM
            _rec("tool_select", "guard", True, True, step=3), _rec("tool_select", "guard", False, True, step=3),
            _rec("memory_write", "trigger", True, None, step=3)]              # meta-level: excluded
    d = M.delegation_ratio(recs)
    assert (d["n"], d["n_delegated"]) == (5, 2) and d["ratio"] == approx(0.4)
    assert d["by_dp"] == {"tool_params": 1.0, "tool_select": 0.25}
    recs = [_rec("tool_select", "replace", "b", "a", task="t1", ok=False),    # wrong, failed -> harmful
            _rec("tool_select", "replace", "b", "a", task="t2", ok=True),     # wrong but recovered
            _rec("tool_select", "replace", "a", "a", task="t3", ok=False)]    # right
    h = M.harmful_delegation(recs, {"t1": False, "t2": True, "t3": False})
    assert (h["n_delegated"], h["n_wrong"], h["n_harmful"]) == (3, 2, 1)
    assert h["harmful_rate"] == approx(1 / 3) and h["wrong_rate"] == approx(2 / 3)
    assert h["harmful_episode_rate"] == approx(1 / 3) and h["failure_attribution"] == approx(0.5)


def test_guard_far_frr_and_rescue():
    recs = [_rec("tool_select", "guard", True, False, task="t1", ok=False),   # false accept
            _rec("tool_select", "guard", False, False, task="t2", ok=True),   # caught, episode then succeeded
            _rec("tool_select", "guard", False, True, task="t3"),             # false reject
            *[_rec("tool_select", "guard", True, True, task=f"u{i}") for i in range(3)],
            _rec("stop", "guard", True, True, task="t4")]
    g = M.guard_error_rates(recs)
    assert g["tool_select"]["far"] == approx(0.5) and g["tool_select"]["frr"] == approx(0.25)
    assert g["stop"]["frr"] == 0.0 and math.isnan(g["stop"]["far"]) and g["all"]["n"] == 7
    r = M.rescue_rate(recs)["tool_select"]
    assert r["catch_rate"] == approx(0.5) and r["rescue_rate"] == approx(0.5) and r["rescue_given_catch"] == 1.0


def test_decision_table_calibrates_choice_decisions_only():
    recs = [_rec("tool_select", "cascade", "a", "a", options=["a", "b"]),
            _rec("tool_select", "cascade", "b", "a", options=["a", "b"], deferred=True),
            _rec("stop", "guard", True, True), _rec("stop", "guard", False, True)]
    t = M.decision_table(recs)
    assert t["tool_select/cascade"]["s1_acc"] == 0.5 and t["tool_select/cascade"]["accepted_acc"] == 1.0
    assert "ece" in t["tool_select/cascade"] and t["tool_select/cascade"]["defer_rate"] == 0.5
    assert t["stop/guard"]["s1_acc"] == 0.5 and "ece" not in t["stop/guard"]   # thresholded Noul: no ECE


def _noul(p_yes: float, gold: bool, tau: float, dist: bool = True) -> dict:
    """A logged guard / trigger decision: the executed value is thresholded at tau, the logged confidence is
    max(p_yes, 1 - p_yes) (what the harness writes)."""
    r = _rec("stop", "trigger", p_yes >= tau, gold)
    r["s1_conf"] = max(p_yes, 1 - p_yes)
    r["s1_dist"] = {"yes": p_yes, "no": 1 - p_yes} if dist else None
    return r


def test_noul_calibration_uses_p_yes_not_thresholded_value():
    """Regression (code review): at tau = 0.9, a trigger with P(yes) = 0.7 and gold "yes" executes "no" while
    logging confidence 0.7 -- pairing the two scored this well-calibrated forecast as a confident error."""
    recs = [_noul(0.7, True, tau=0.9), _noul(0.2, False, tau=0.9)]
    row = M.decision_table(recs)["stop/trigger"]
    assert row["s1_acc"] == 0.5                                               # executed values: no, no
    assert row["cal"] == "p_yes" and row["n_cal"] == 2
    assert row["ece"] == approx(0.5 * 0.3 + 0.5 * 0.2)                      # |1 - .7|, |0 - .2|
    assert row["brier"] == approx((0.3 ** 2 + 0.2 ** 2) / 2) and row["auroc"] == 1.0
    buggy = M.ece([r["s1_conf"] for r in recs], [float(r["s1_value"] == r["gold"]) for r in recs])
    assert buggy == approx(0.45) and row["ece"] != approx(buggy)
    bc = M.binary_calibration([0.7, 0.2], [1, 0])
    assert bc["argmax_acc"] == 1.0 and bc["aurc"] == 0.0
    assert "ece" not in M.decision_table([_noul(0.7, True, 0.9, dist=False)])["stop/trigger"]   # old logs


def test_noul_calibration_end_to_end_with_trigger_tau_09():
    from jevharness.bench.scorecard import decision_block
    from jevharness.decision.simulated import SimulatedBackend

    cfg = HarnessConfig(name="stop_trigger", stop=DPConfig(mode=Mode.TRIGGER, tau=0.9))
    agent = JevAgent(SimulatedLLM("scripted", error_rate=0.3, simulated_latency_ms=10), cfg,
                     s1=SimulatedBackend(accuracy=0.8, seed=5))
    trajs = [agent.run(JevGymEnv(seed=s)).to_dict() for s in range(30)]
    # (non-strict v1 tasks: a wrong write makes the state unrecoverable -- doomed, gold None, excluded from rates)
    recs = [r for r in M.decision_records(trajs) if r["dp"] == "stop" and r.get("gold_status") != "doomed"]
    assert recs and all(r["s1_dist"] and isinstance(r["gold"], bool) for r in recs)
    p = [r["s1_dist"]["yes"] for r in recs]
    y = [float(r["gold"]) for r in recs]
    assert any((pi >= 0.5) != (r["s1_value"]) for pi, r in zip(p, recs))   # tau = 0.9 changes decisions
    row = M.decision_table(M.decision_records(trajs))["stop/trigger"]
    assert row["cal"] == "p_yes" and row["ece"] == approx(M.ece(p, y)) and row["brier"] == approx(M.brier(p, y))
    noul = decision_block(trajs)["noul"]
    assert noul["n"] == len(recs) and noul["ece"] == approx(M.ece(p, y))
    assert noul["auroc"] == approx(M.auroc(p, y), nan_ok=True)


# ====================================================================== score card
def _sim_exp(n_tasks: int = 6) -> dict:
    return {"name": "tiny_sim", "suite": {"name": "jevgym", "n": n_tasks, "n_distractors": 8},
            "llm": {"provider": "simulated", "model": "scripted", "error_rate": 0.2, "simulated_latency_ms": 1000},
            "s1": {"type": "simulated", "accuracy": 0.8, "latency_ms": 50, "price_per_m_input": 0.042}, "workers": 1,
            "arms": [{"name": "llm_only"},
                     {"name": "ts_params_replace", "tool_select": {"mode": "replace"},
                      "tool_params": {"mode": "replace"}},
                     {"name": "ts_guard", "tool_select": {"mode": "guard", "tau": 0.5}}],
            "sweeps": [{"base": {"name": "ts_cascade", "tool_select": {"mode": "cascade"}},
                        "set": "tool_select.tau", "values": [0.5, 0.8, 0.95]}]}


def test_scorecard_cli_on_tiny_simulated_run(tmp_path, capsys):
    from jevharness.bench.runner import run_experiment

    run_experiment(_sim_exp(), tmp_path, progress=False)
    run_dir = tmp_path / "tiny_sim"
    out = tmp_path / "card.json"
    scorecard_main([str(run_dir), "--baseline", "llm_only", "--weak", "ts_params_replace", "--format", "json",
                    "--n-boot", "300", "--out", str(out)])
    card = json.loads(out.read_text())
    assert card["meta"]["simulated"] is True and card["meta"]["baseline"] == "llm_only"
    arms = card["arms"]
    assert set(arms) == {"llm_only", "ts_params_replace", "ts_guard", "ts_cascade_tau0.5", "ts_cascade_tau0.8",
                         "ts_cascade_tau0.95"}
    for a, r in arms.items():
        assert r["n_tasks"] == 6 and 0.0 <= r["success"] <= 1.0 and r["cost_usd"] >= 0.0
        assert ("vs_baseline" in r) == (a != "llm_only")
    assert arms["ts_params_replace"]["decisions"]["delegation_ratio"] > 0.5
    assert arms["llm_only"]["decisions"]["delegation_ratio"] == 0.0
    assert "guard" in arms["ts_guard"]["decisions"]
    sw = card["headline"]["sweeps"]["ts_cascade_tau"]
    assert sw["values"] == [0.5, 0.8, 0.95] and "apgr" in sw and "aiq_gain" in sw
    assert sw["aiq"] >= sw["aiq_zero_router"] - 1e-9          # the sweep's hull contains the references
    assert {"iso_quality", "iso_cost", "hypervolume", "frontier_cost_of_pass"} <= set(card["headline"])
    scorecard_main([str(run_dir), "--weak", "ts_params_replace"])
    md = capsys.readouterr().out
    assert "SIMULATED RUN" in md and "| llm_only |" in md and "### Sweeps" in md
    scorecard_main([str(run_dir), "--format", "latex"])
    tex = capsys.readouterr().out
    assert r"\toprule" in tex and r"llm\_only" in tex and tex.startswith("% SIMULATED RUN")
    from jevharness.bench.report import report

    assert "## Score card" in report(run_dir)


def _traj(task: str, ok: bool, cost: float, cfg: str = "a", llm_calls: int = 2) -> dict:
    return {"task_id": task, "config": cfg, "backend": None, "llm": "big", "steps": [], "final_answer": None,
            "result": {"success": ok}, "error": None, "meta": {},
            "usage": {"total": {"cost_usd": cost, "latency_ms": 1000.0}, "s2": {"calls": llm_calls}},
            "calls": [{"component": "s2", "model": "big"}] * llm_calls}


def test_scorecard_trials_pass_k_dedupe_and_strong_calls():
    base = [_traj(f"t{i}#{k}", True, 1.0) for i in range(4) for k in range(3)]
    flaky = [_traj(f"t{i}#{k}", k != 0 or i >= 2, 0.5) for i in range(4) for k in range(3)]
    routed = [dict(_traj(f"t{i}#{k}", True, 0.3), calls=[{"component": "s2", "model": "small"}] * 2)
              for i in range(4) for k in range(3)]
    card = compute_scorecard({"llm_only": base, "flaky": flaky + [flaky[-1]], "routed": routed}, n_boot=200)
    assert task_key({"task_id": "t1#2"}) == "t1" and card["meta"]["min_trials_per_task"] == 3
    fl = card["arms"]["flaky"]
    assert fl["n_tasks"] == 4 and fl["n_episodes"] == 12                    # duplicate record dropped
    assert any("duplicate" in n for n in card["meta"]["notes"])
    assert fl["pass_hat_k"]["1"] == approx(10 / 12) and fl["pass_hat_k"]["3"] == approx(0.5)
    assert fl["cost_of_pass"] == approx(0.5 / (10 / 12))
    assert card["arms"]["routed"]["llm_call_ratio"] == 1.0 and card["arms"]["routed"]["strong_call_ratio"] == 0.0
    assert card["headline"]["iso_quality"]["arm"] == "routed"               # 70% cheaper, not worse


def test_detect_sweeps():
    names = ["llm_only", "ts_cascade_tau0.5", "ts_cascade_tau0.9", "ts_cascade_tau0.95", "x_acc70", "x_acc95"]
    assert list(detect_sweeps(names)) == ["ts_cascade_tau"]                  # >= 3 arms when auto-detected
    assert set(detect_sweeps(names, only=["x_acc"])) == {"x_acc"}


# ====================================================================== JevGym v2
def _names(env: JevGymEnv) -> set[str]:
    return {t["function"]["name"] for t in env.tools()}


def test_jevgym_v1_defaults_unchanged():
    env = JevGymEnv(seed=3)
    assert env.family == JevGymEnv.FAMILIES[3] and env.task_id == "jevgym-ship-3-d10"
    assert "ask_user" not in _names(env) and not _names(env) & {t.name for t in NEAR_TOOLS}
    assert env.reset() == "You have access to the customer-operations tools."
    assert set(env.evaluate(None)) == {"success", "score", "n_calls", "optimal_calls"}
    assert [f().family for f in load_suite("jevgym", n=5)] == JevGymEnv.FAMILIES


def test_jevgym_near_and_mixed_distractors():
    env = JevGymEnv(seed=0, distractor_style="near", n_distractors=16)
    names = _names(env)
    assert {"get_order", "find_customer_by_email", "cancel_subscription", "refund_shipping_fee",
            "check_stock_warehouse", "ship_order_international"} <= names
    assert not names & {n for n, _ in DISTRACTOR_NAMES} and env.task_id.endswith("-d16-near")
    near = {t.name: t.shadows for t in NEAR_TOOLS}
    shadows = {near[n] for n in names if n in near}
    assert {"find_user", "get_orders", "cancel_order", "refund", "check_stock", "ship_order"} <= shadows
    mixed = [t["function"]["name"] for t in make_distractors(10, "mixed")]
    assert sum(n in near for n in mixed) == 5 and len(set(mixed)) == 10
    big = make_distractors(len(NEAR_TOOLS) + 5, "near")                        # beyond the pool: random fill
    assert len({t["function"]["name"] for t in big}) == len(NEAR_TOOLS) + 5
    with pytest.raises(ValueError):
        JevGymEnv(seed=0, distractor_style="nearest")


def test_jevgym_near_distractor_dynamics():
    env = JevGymEnv(seed=1, family="refund", distractor_style="near", n_distractors=len(NEAR_TOOLS))
    env.reset()
    for tool, args in env._plan()[:-1]:
        env.step(tool, args)
    assert env.evaluate("done")["success"]
    env.step("refund_shipping_fee", {"order_id": env.delivered_order["order_id"], "method": "store_credit"})
    res = env.evaluate("done")
    assert not res["success"] and res["side_effects"] == 1                  # harmful side effect
    strict = JevGymEnv(seed=1, family="refund", distractor_style="near", n_distractors=len(NEAR_TOOLS), strict=True)
    strict.reset()
    assert strict.step("issue_store_credit", {"user_id": strict.user["user_id"], "amount": 10}).done
    assert strict.evaluate("done")["irreversible_error"]
    stock = JevGymEnv(seed=2, family="stock", distractor_style="near", n_distractors=len(NEAR_TOOLS))
    stock.reset()
    split = stock._warehouse_split(stock.target_item)                         # tempting partial count
    assert sum(split.values()) == stock.world.stock[stock.target_item]
    obs = stock.step("check_stock_warehouse", {"item": stock.target_item, "warehouse": "west"}).observation
    assert obs.endswith(f"{split['west']} units")
    assert stock.step("get_order", {"order_id": "ORD-0"}).observation.startswith("Order not found")


def test_load_suite_plumbs_jevgym_options():
    envs = [f() for f in load_suite("jevgym", n=14, families="all", distractor_style="mixed", strict=True,
                                    n_distractors=12, seed0=100)]
    assert {e.family for e in envs} == set(ALL_FAMILIES)
    assert all(e.strict and e.distractor_style == "mixed" and e.ask_tool and e.n_distractors == 12 for e in envs)
    assert len({e.task_id for e in envs}) == 14
    assert {f().family for f in load_suite("jevgym", n=4, families=["clarify", "multi"])} == {"clarify", "multi"}
    assert {f().family for f in load_suite("jevgym", n=4, families="v2")} == {"clarify", "multi"}
    with pytest.raises(ValueError):
        load_suite("jevgym", n=2, distractor_stlye="near")                   # typo -> error, not silent default
    with pytest.raises(ValueError):
        load_suite("jevgym", n=2, families="v3")


@pytest.mark.parametrize("env_kw", [{}, {"distractor_style": "near", "n_distractors": 30, "strict": True},
                                    {"distractor_style": "mixed", "n_distractors": 60, "noise_lines": 3}])
def test_new_families_solved_by_perfect_llm(env_kw):
    agent = JevAgent(SimulatedLLM("scripted", error_rate=0.0, simulated_latency_ms=10), HarnessConfig())
    for family in ("clarify", "multi"):
        for seed in range(8):                                                 # all four clarify variants twice
            env = JevGymEnv(seed=seed, family=family, **env_kw)
            t = agent.run(env).to_dict()
            assert t["error"] is None and t["result"]["success"], (env.task_id, env.variant, t["result"])
            assert t["result"]["n_calls"] == t["result"]["optimal_calls"]
    variants = {JevGymEnv(seed=s, family="clarify").variant for s in range(4)}
    assert variants == set(JevGymEnv.CLARIFY_VARIANTS)


def test_new_families_oracle_gold_is_exact():
    """An oracle System-1 answering from the logged gold labels solves every task in replace mode, so the
    per-state gold (tool and arguments, including ``ask_user.missing``) is consistent with the evaluator."""
    cfg = HarnessConfig(name="oracle_replace", tool_select=DPConfig(mode=Mode.REPLACE),
                        tool_params=DPConfig(mode=Mode.REPLACE))
    agent = JevAgent(SimulatedLLM("scripted", error_rate=0.0, simulated_latency_ms=10), cfg, s1=OracleBackend())
    trajs = [agent.run(JevGymEnv(seed=s, families="v2", distractor_style="near", n_distractors=20)).to_dict()
             for s in range(12)]
    assert all(t["result"]["success"] for t in trajs), [t["result"] for t in trajs]
    table = M.decision_table(M.decision_records(trajs))
    assert table["tool_select/replace"]["s1_acc"] == 1.0
    asks = [d for t in trajs for s in t["steps"] for d in s["decisions"] if d["gold"] == "ask_user"]
    assert asks and all(d["final_value"] == "ask_user" for d in asks)


def test_clarify_requires_asking_before_acting():
    env = JevGymEnv(seed=0, family="clarify")                                 # variant "order"
    assert env.variant == "order" and env.missing == "order_id" and env.ask_tool
    env.reset()
    u, target = env.user, env.pending_order["order_id"]
    pending = [o for o in env.world.orders.values() if o["user_id"] == u["user_id"] and o["status"] == "pending"]
    assert len(pending) >= 2                                                  # genuinely ambiguous
    env.step("find_user", {"name": u["name"]})
    env.step("get_orders", {"user_id": u["user_id"], "status": "pending"})
    assert env.oracle_action() == ("ask_user", {"missing": "order_id"})
    env.step("cancel_order", {"order_id": target, "reason": "customer_request"})   # a lucky guess
    assert not env.evaluate("done")["success"]
    env = JevGymEnv(seed=0, family="clarify")
    env.reset()
    assert "already gave you" in env.step("ask_user", {"missing": "shipping_speed"}).observation
    assert not env.asked                                                      # wrong slot does not count
    assert target in env.step("ask_user", {"question": "Which order should I cancel?"}).observation
    for tool, args in env._plan():
        if tool not in ("ask_user", "finish"):
            env.step(tool, args)
    res = env.evaluate("done")
    assert res["success"] and res["asked"] and res["n_asks"] == 2
    strict = JevGymEnv(seed=0, family="clarify", strict=True)
    strict.reset()
    assert strict.step("cancel_order", {"order_id": target, "reason": "customer_request"}).done
    # an unnecessary question in a v1 family is harmless (but counted)
    v1 = JevGymEnv(seed=0, families="all")
    v1.reset()
    assert v1.family == "cancel" and v1.step("ask_user", {"missing": "order_id"}).observation.startswith("Customer")
    for tool, args in v1._plan()[:-1]:
        v1.step(tool, args)
    assert v1.evaluate("done") == {"success": True, "score": 1.0, "n_calls": 4, "optimal_calls": 3,
                                   "side_effects": 0, "n_asks": 1}


def test_oracle_accepts_checks_arguments_not_only_the_tool():
    env = JevGymEnv(seed=1, family="refund")
    env.reset()
    for tool, args in env._plan()[:2]:
        env.step(tool, args)
    tool, gold = env.oracle_action()
    assert tool == "refund" and env.oracle_accepts(tool, gold)
    assert env.oracle_accepts("refund", {**gold, "amount": gold["amount"] + 0.004})   # within 0.01
    assert not env.oracle_accepts("refund", {**gold, "method": "original_payment"})   # right tool, wrong arg
    assert not env.oracle_accepts("cancel_order", {"order_id": gold["order_id"], "reason": "fraud"})
    total = JevGymEnv(seed=4, family="total")
    total.reset()
    for tool, args in total._plan()[:-1]:
        total.step(tool, args)
    answer = total._plan()[-1][1]["answer"]
    assert total.oracle_accepts("finish", {"answer": f"It is {answer}"})
    assert not total.oracle_accepts("finish", {"answer": "done"})
    ask = JevGymEnv(seed=1, family="clarify")                                # variant "method": ask first
    ask.reset()
    assert ask.oracle_accepts("ask_user", {"missing": "refund_method", "question": "How to refund?"})
    assert not ask.oracle_accepts("ask_user", {"missing": "order_id"})


def test_multi_family_long_horizon_and_memory():
    env = JevGymEnv(seed=0, family="multi", multi_customers=4)
    env.reset()
    plan = env._plan()
    assert len(env.jobs) == 4 and len(plan) == 3 * 4 + 1 and plan[-1][0] == "finish"
    assert len({u["user_id"] for _, u, _ in env.jobs}) == 4
    total = plan[-1][1]["answer"]
    assert total == f"{sum(o['amount'] for _, _, o in env.jobs):.2f}"
    for tool, args in plan[:-1]:
        env.step(tool, args)
    assert env.evaluate(f"The total is {total}.")["success"]
    assert not env.evaluate("I do not remember the amounts.")["success"]     # needs the earlier observations
    env2 = JevGymEnv(seed=0, family="multi", multi_customers=4)
    env2.reset()
    for tool, args in plan[:-2]:                                              # skip the last write
        env2.step(tool, args)
    assert not env2.evaluate(total)["success"]


# ====================================================================== H5 readout (pre-registration A5.2)
def test_sign_test_exact_hand_cases():
    r = M.sign_test_exact(5, 0)                                   # P(X >= 5 | n = 5) = 1/32; two-sided 2/32
    assert r["p_wins"] == approx(1 / 32) and r["p"] == approx(2 / 32) and r["p_losses"] == 1.0
    r = M.sign_test_exact(3, 3)
    assert r["p"] == 1.0 and r["p_wins"] == approx(42 / 64)       # P(X >= 3 | n = 6) = (20 + 15 + 6 + 1) / 64
    assert M.sign_test_exact(0, 0) == {"n": 0, "wins": 0, "losses": 0, "p": 1.0, "p_wins": 1.0, "p_losses": 1.0}
    assert M.sign_test_exact(1, 4)["p_losses"] == approx(5 / 32 + 1 / 32)


def test_wilcoxon_one_sided_exact_ties_zeros_and_normal():
    r = M.wilcoxon_one_sided([-1, -2, -3, -4, -5])                 # all negative: W+ = 0, P = 1 / 2^5
    assert (r["n"], r["w_plus"], r["w_minus"], r["exact"]) == (5, 0.0, 15.0, True)
    assert r["p_less"] == approx(1 / 32) and r["p_greater"] == 1.0 and r["median"] == -3
    r = M.wilcoxon_one_sided([-3, -1, 2])                          # ranks 3, 1, 2: W+ = 2; subset sums of {1,2,3}
    assert r["w_plus"] == 2 and r["p_less"] == approx(3 / 8) and r["p_greater"] == approx(6 / 8)
    r = M.wilcoxon_one_sided([-2, -2, 1])                          # tie: ranks 2.5, 2.5, 1 -> doubled {5, 5, 2}
    assert r["w_plus"] == 1 and r["p_less"] == approx(2 / 8) and r["p_greater"] == approx(7 / 8)
    r = M.wilcoxon_one_sided([0, -1, -2])                          # zeros dropped
    assert (r["n"], r["n_zero"]) == (2, 1) and r["p_less"] == approx(1 / 4)
    assert M.wilcoxon_one_sided([0.0, 0.0])["p_less"] == 1.0 and M.wilcoxon_one_sided([])["n"] == 0
    d = [-(i + 1) for i in range(12)] + [3.5, 6.5, 9.5]            # moderate tail: exact vs normal agree to ~1e-3
    e, a = M.wilcoxon_one_sided(d), M.wilcoxon_one_sided(d, exact_max_n=10)
    assert e["exact"] and not a["exact"] and e["p_less"] < 0.05 and a["p_less"] == approx(e["p_less"], abs=2e-3)
    assert e["p_less"] + e["p_greater"] >= 1.0                     # the two tails share the observed value


def test_paired_success_lb_hand_case():
    base = {f"t{i}": 1.0 for i in range(20)}
    assert M.paired_success_lb(base, dict(base))["lb"] == 0.0
    one = dict(base)
    one["t0"] = 2 / 3                                              # one failed attempt of 3 on one of 20 tasks
    r = M.paired_success_lb(base, one)
    # resampled Delta = -(#copies of t0) / 60, copies ~ Bin(20, 0.05): P(>= 3) = 0.075 < 0.10 <= P(>= 2) = 0.264,
    # so the 10% quantile sits on -2/60
    assert r["diff"] == approx(-1 / 60) and r["lb"] == approx(-2 / 60) and (r["wins"], r["losses"], r["ties"]) == (0, 1, 19)
    assert M.paired_success_lb(base, {k: 0.0 for k in base})["lb"] == -1.0


def _h5_case(flip: tuple[str, int] | None = None, k: int = 3, cap: float = 3600.0):
    """20 tasks x k attempts. Baseline solves tasks 0-13 (1000 s per attempt), fails 14-19 (at the cap). The arm
    solves 0-17 in every attempt and 18-19 in k - 1 attempts (700 s per solved attempt). ``flip`` turns one solved
    arm attempt into a timeout."""
    tasks = [f"t{i:02d}" for i in range(20)]
    base_att = {t: [(1.0, 1000.0)] * k if i < 14 else [(0.0, cap)] * k for i, t in enumerate(tasks)}
    arm_att = {t: [(1.0, 700.0)] * k if i < 18 else [(1.0, 700.0)] * (k - 1) + [(0.0, cap)]
               for i, t in enumerate(tasks)}
    if flip:
        t, j = flip
        att = list(arm_att[t])
        assert att[j][0] == 1.0
        att[j] = (0.0, cap)
        arm_att[t] = att

    def agg(att):
        succ = {t: sum(s for s, _ in a) / len(a) for t, a in att.items()}
        time = {t: sum(x for _, x in a) / len(a) for t, a in att.items()}
        solved = {t: sum(x for s, x in a if s) / sum(s for s, _ in a) for t, a in att.items() if any(s for s, _ in a)}
        return succ, time, solved
    bs, bt, bsol = agg(base_att)
    as_, at, asol = agg(arm_att)
    return M.h5_verdict(bs, as_, bt, at, bsol, asol, n_boot=4000)


def test_h5_verdict_components():
    v = _h5_case()
    assert v["n_tasks"] == 20 and v["d_success"] == approx((4 + 4 / 3) / 20) and v["d_success_lb"] > 0
    assert (v["wins"], v["losses"], v["ties"]) == (6, 0, 14) and v["sign_p_wins"] == approx(1 / 64)
    assert v["n_both_solved"] == 14 and v["d_time_median_solved"] == -300 and v["wilcoxon_p"] == approx(1 / 2 ** 14)
    assert (v["time_faster_tasks"], v["time_slower_tasks"]) == (14, 0) and v["time_sign_p"] == approx(1 / 2 ** 14)
    assert v["d_time_median_all"] == -300 and v["better"] and v["faster"] and v["faster_and_better"]
    assert v["success_preserving"] and v["strictly_better"] and v["success_preserving_and_faster"]
    assert v["screen_label"].startswith("passes the exploratory")
    # the opposite arm: slower on the solved tasks and losing on success
    base = {f"t{i}": 1.0 if i < 10 else 0.0 for i in range(20)}
    arm = {f"t{i}": 1.0 if i < 6 else 0.0 for i in range(20)}
    bt = {k: 500.0 for k in base}
    at = {k: 900.0 for k in base}
    v = M.h5_verdict(base, arm, bt, at, n_boot=2000)
    assert (v["wins"], v["losses"]) == (0, 4) and not v["better"] and not v["faster"] and v["n_both_solved"] == 6
    assert v["d_time_median_solved"] == 400 and v["wilcoxon_p"] == 1.0 and not v["faster_and_better"]
    # k = 1 default: tasks with success == 1 are the solved ones, timed by their only attempt
    v = M.h5_verdict(base, dict(base), bt, {k: 400.0 for k in base}, n_boot=2000)
    assert v["n_both_solved"] == 10 and v["faster"] and v["better"] and v["d_success_lb"] == 0.0
    assert v["success_preserving_and_faster"] and not v["strictly_better"] and not v["faster_and_better"]   # A6.6 tiers
    # empty intersection: nothing can be claimed
    v = M.h5_verdict({"a": 1.0}, {"b": 1.0}, {"a": 1.0}, {"b": 1.0}, n_boot=100)
    assert v["n_tasks"] == 0 and not v["better"] and not v["faster"]


def test_h5_one_timeout_cannot_flip_the_verdict():
    """A5.2: at the planned sizes (20 tasks x 3 attempts) one capped attempt moves Delta success by 1/60 and one task
    of the medians; whichever attempt of a faster-and-better arm times out, the verdict stands."""
    ref = _h5_case()
    assert ref["faster_and_better"] and ref["success_preserving_and_faster"]
    for i in range(20):
        for j in range(3):
            if i >= 18 and j == 2:
                continue                                           # already a timeout in the reference
            v = _h5_case(flip=(f"t{i:02d}", j))
            assert v["success_preserving_and_faster"], (i, j, v)   # the registered verdict (A6.6, A6.16 (j))
            assert v["d_success"] == approx(ref["d_success"] - 1 / 60)
            assert v["wins"] >= v["losses"] and v["d_success_lb"] > -0.05
