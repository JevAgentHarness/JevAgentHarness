"""Numerical checks of the delegation theory (paper Sec. 3.4, App. A) via ``scripts/check_propositions.py``:
Prop. 1 and its general form, the comparative statics (including the corrected correlated-errors statement), the
expected gain, the guard rule with a damage term, the trigger rule, Prop. 2 (gate value of information against a
matched-rate random gate, with need-dependent firing costs, the matched-rate accuracy gap, the rate correction and
the guard-with-damage version), the AUROC bound, and the repository's estimators against the closed forms. Synthetic
populations, fixed seeds, offline, a few seconds."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_propositions", ROOT / "scripts" / "check_propositions.py")
CP = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(CP)

N_POP, N_MC = 12, 12000


def rng(i: int) -> np.random.Generator:
    return np.random.default_rng([12345, i])


def test_prop1_threshold_minimises_loss_over_all_rules_that_see_only_c():
    r = CP.check_prop1(rng(0), N_POP, N_MC)
    assert r["ok"], r
    assert r["exact_gap_vs_best_threshold"] <= 1e-10 and r["exact_gap_vs_random_rules"] <= 1e-10


def test_prop1_check_detects_a_wrong_closed_form():
    """Power: halving the cost term (a plausible slip) has strictly positive exact regret on some population."""
    g = rng(1)
    worst = 0.0
    for _ in range(20):
        c, w = CP.beta_support(g)
        q = float(g.uniform(0.5, 1.0))
        dC, V = CP.costs(g)
        qv = np.full_like(c, q)
        right = CP.loss_threshold(min(max(q - dC / V, 0), 1), c, w, c, qv, dC, V)
        wrong = CP.loss_threshold(min(max(q - dC / (2 * V), 0), 1), c, w, c, qv, dC, V)
        assert wrong >= right - 1e-12
        worst = max(worst, wrong - right)
    assert worst > 1e-4


def test_general_rule_is_pointwise_and_a_threshold_iff_the_accept_set_is_an_upper_set():
    r = CP.check_general(rng(2), 3 * N_POP, N_MC)
    assert r["ok"], r
    assert r["n_upper_set"] > 0 and r["n_not_upper_set"] > 0      # both cases exercised
    assert r["monotone_a_but_not_threshold"] == 0


def test_comparative_statics_and_the_corrected_correlated_errors_claim():
    r = CP.check_statics(rng(3), N_POP)
    assert r["ok"], r
    ce = r["correlated_errors"]
    # q_L(c) rising with c, at a fixed marginal q_L, moves the optimal threshold in BOTH directions
    assert ce["threshold_above_prop1"] > 0 and ce["threshold_below_prop1"] > 0
    assert ce["sign_rule_failures"] == 0 and ce["linear_formula_failures"] == 0
    assert r["monotone_statics_violations"] == 0


def test_correlated_errors_linear_example_raises_the_threshold_when_tau1_exceeds_mean_confidence():
    # g(c) = c, q_L(c) = alpha + s c, c uniform on (0, 1): E[c] = 1/2
    alpha, s, k = 0.8, 0.2, 0.0
    qbar = alpha + s * 0.5
    tau1 = qbar - k
    tau_g = CP.crossing_continuous(alpha, s, k)
    assert tau1 > 0.5 and tau_g > tau1
    assert math.isclose(tau_g - tau1, s * (tau1 - 0.5) / (1 - s))
    # and the opposite direction when the Prop. 1 threshold is below the mean confidence
    alpha2 = 0.3
    tau1b = alpha2 + s * 0.5 - k
    assert tau1b < 0.5 and CP.crossing_continuous(alpha2, s, k) < tau1b


def test_expected_gain_of_the_optimal_cascade():
    r = CP.check_gain(rng(4), N_POP, N_MC)
    assert r["ok"], r


def test_guard_rule_with_damage_term():
    r = CP.check_guard(rng(5), N_POP, N_MC)
    assert r["ok"], r
    assert r["delta0_special_case_failures"] == 0 and r["guard_as_trigger_failures"] == 0
    assert r["max_regret_of_ignoring_damage"] > 0          # the delta = 0 formula is wrong when retries can damage


def test_guard_threshold_closed_form_special_cases():
    dC, V = 0.1, 1.0
    assert math.isclose(CP.tau_guard(dC, V, 0.5, 0.0), 1 - dC / (0.5 * V))
    assert CP.tau_guard(dC, V, 0.05, 0.3) < 0                # eta <= dC / V: never reject
    assert CP.tau_guard(dC, V, 0.5, 0.2) < CP.tau_guard(dC, V, 0.5, 0.0)   # damage lowers the rejection threshold
    assert math.isclose(1 - CP.tau_trigger(0.5 * V, 0.2 * V, dC), CP.tau_guard(dC, V, 0.5, 0.2))


def test_trigger_rule():
    r = CP.check_trigger(rng(6), N_POP, N_MC)
    assert r["ok"], r


def test_prop2_gate_voi_against_a_matched_rate_random_gate():
    r = CP.check_prop2(rng(7), N_POP, N_MC)
    assert r["ok"], r
    assert r["n_need_dependent_cost"] > 0                       # need-dependent firing costs are exercised
    assert r["exact_max_error"] <= 1e-10 and r["constant_cost_cancels_max_error"] <= 1e-10
    assert r["need_dependent_bias_failures"] == 0
    assert r["accuracy_gap_exact_max_error"] <= 1e-12
    # power: the simulation rejects the constant-cost formula when the firing cost depends on need
    assert r["power_pop"]["z_constant_cost_formula"] > 2 * CP.Z_MC
    assert r["power_pop"]["z_need_dependent_formula"] <= CP.Z_MC


def test_prop2_firing_costs_cancel_only_when_they_do_not_depend_on_need():
    """Review round 8 (astra W3): matching the firing rate cancels a constant firing cost, not a need-dependent one.
    With cost C1 at needed and C0 at unneeded opportunities the coefficient is B + H - (C1 - C0)."""
    n, pi, tpr, fpr, B, H, kappa = 1, 0.5, 0.8, 0.2, 1.0, 0.0, 0.0
    J = tpr - fpr
    # constant cost: any value gives the cost-free closed form
    for C in (0.0, 0.3, 0.9):
        assert math.isclose(CP.voi_exact(n, pi, tpr, fpr, B, H, C, C, kappa),
                            CP.voi_closed(n, pi, J, B, H, kappa), abs_tol=1e-12)
    # need-dependent cost: 0.075 = (1 + 0 - 0.5) * 0.25 * 0.6, not the constant-cost 0.15
    exact = CP.voi_exact(n, pi, tpr, fpr, B, H, 0.5, 0.0, kappa)
    assert math.isclose(exact, CP.voi_closed(n, pi, J, B, H, kappa, dC=0.5), abs_tol=1e-12)
    assert math.isclose(exact, 0.075, abs_tol=1e-12)
    assert not math.isclose(exact, CP.voi_closed(n, pi, J, B, H, kappa), abs_tol=1e-6)
    # a module that costs more when it is NOT needed gains value at the matched rate
    assert CP.voi_exact(n, pi, tpr, fpr, B, H, 0.0, 0.5, kappa) > CP.voi_closed(n, pi, J, B, H, kappa)


def test_prop2_matched_rate_accuracy_gap_is_two_pi_one_minus_pi_j():
    """At a matched firing rate the gate's accuracy exceeds the random gate's by 2 pi (1 - pi) J, so within that
    comparison accuracy and J carry the same contrast; raw accuracy across base rates does not (always-no gate)."""
    g = rng(11)
    for _ in range(200):
        pi, tpr, fpr = (float(x) for x in g.uniform(0, 1, 3))
        assert math.isclose(CP.accuracy_gap(pi, tpr, fpr), 2 * pi * (1 - pi) * (tpr - fpr), abs_tol=1e-12)
    # an always-no gate (TPR = FPR = 0) is 1 - pi accurate but has J = 0: no gain over its random gate, minus kappa
    pi, kappa = 0.1, 0.01
    assert math.isclose(CP.accuracy_gap(pi, 0.0, 0.0), 0.0, abs_tol=1e-15)
    assert math.isclose(CP.voi_exact(10, pi, 0.0, 0.0, 1.0, 0.5, 0.2, 0.2, kappa), -10 * kappa, abs_tol=1e-12)


def test_auroc_bound_needs_a_concave_roc():
    r = CP.check_auroc(rng(8))
    assert r["ok"], r
    nc = r["nonconcave_counterexample"]
    assert nc["max_J"] > 2 * nc["auroc"] - 1


def test_repository_estimators_match_the_closed_forms():
    r = CP.check_code(rng(9), N_POP)
    assert r["ok"], r


def test_cli_writes_a_passing_report(tmp_path):
    out = tmp_path / "props.json"
    code = CP.main(["--n-pop", "4", "--mc", "4000", "--only", "prop1", "guard", "prop2", "code", "--json", str(out)])
    assert code == 0
    rep = json.loads(out.read_text())
    assert rep["ok"] and rep["simulated"] and set(rep) >= {"prop1", "guard", "prop2", "code"}


@pytest.mark.parametrize("seed", [1, 2])
def test_full_run_is_stable_across_seeds(seed):
    res = CP.run(seed=seed, n_pop=8, mc=6000)
    assert res["ok"], {k: v for k, v in res.items() if isinstance(v, dict) and not v.get("ok")}


# ====================================================================== threshold_theory: the registered H1 grid
def _sampled_records(g: np.random.Generator, n_tasks: int = 60, per_task: int = 20, q: float = 0.9,
                     cost: float = 0.05) -> list[dict]:
    """Shadow-style paired records from an A1-A2 population: c on two-decimal levels, System-1 right w.p. c, the
    LLM right w.p. q independently, every deferral costing ``cost``."""
    recs = []
    for t in range(n_tasks):
        for i in range(per_task):
            c = float(np.round(g.beta(5, 1.5), 2))
            recs.append({"arm": "syn", "task_id": f"t{t}", "step": i, "c": c, "s1_ok": bool(g.random() < c),
                         "s2_ok": bool(g.random() < q), "acceptable": True, "final_ok": True,
                         "episode_success": True, "s1_cost": 0.0, "s2_cost": cost, "s1_lat_ms": 0.0,
                         "s2_lat_ms": 0.0})
    return recs


def test_threshold_theory_registered_h1_grid_and_deployed_estimator_row():
    from jevharness.bench import threshold_theory as T

    assert T.parse_taus(["h1"]) == list(T.H1_TAUS) == [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    assert T.parse_taus(None)[-1] == math.inf and T.parse_taus(["0.9", "inf"]) == [0.9, math.inf]
    recs = _sampled_records(rng(10))
    res = T.analyze(recs, Vs=[1.0], taus=list(T.H1_TAUS), train_frac=0.3, n_boot=50, n_perm=50)
    assert [r["tau"] for r in res["frontier"]["rows"]] == list(T.H1_TAUS)
    rg = res["by_V"][0]["regret"]
    assert rg["tau_best_test"] in T.H1_TAUS
    train, _ = T.split_tasks(recs, 0.3, 0)
    want = T.scalar_threshold_projection(train, 1.0, T.cost_terms(train)["delta_c"])["tau_hat"]
    assert rg["candidates"]["tau_hat_projection"]["tau"] == want
    # registered rows are unchanged: tau_hat (the isotonic plug-in) and tau = 0.8 keep their places
    assert list(rg["candidates"])[:2] == ["tau_hat", "tau_default"]
    assert rg["candidates"]["tau_hat"]["tau"] == rg["thresholds"]["general"]
    # the A1-A2 population's Prop. 1 threshold is q - dC / V = 0.85; both estimators land near it
    for name in ("tau_hat", "tau_hat_projection"):
        assert abs(rg["candidates"][name]["tau"] - 0.85) < 0.1, (name, rg["candidates"][name])


def test_threshold_theory_cli_runs_the_registered_h1_grid(tmp_path):
    from jevharness.bench import threshold_theory as T
    from jevharness.bench.runner import run_experiment

    exp = {"name": "tiny_h1", "suite": {"name": "jevgym", "n": 8, "n_distractors": 6, "strict": True},
           "llm": {"provider": "simulated", "model": "scripted", "error_rate": 0.25},
           "s1": {"type": "simulated", "accuracy": 0.8}, "workers": 2,
           "arms": [{"name": "jev_shadow", "tool_select": {"mode": "shadow"}}]}
    run_experiment(exp, tmp_path, progress=False)
    out = tmp_path / "h1.json"
    T.main([str(tmp_path / "tiny_h1"), "--arm", "jev_shadow", "--dp", "tool_select", "--V", "1", "--taus", "h1",
            "--train-frac", "0.3", "--n-boot", "20", "--n-perm", "20", "--format", "json", "--out", str(out)])
    res = json.loads(out.read_text())
    assert [r["tau"] for r in res["frontier"]["rows"]] == list(T.H1_TAUS)
    assert res["meta"]["simulated"] is True
