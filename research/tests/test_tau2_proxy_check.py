"""scripts/tau2_proxy_check.py -- the C7 proxy check of PREREGISTRATION A5.4 (e): Kendall's tau-b between the E2 and
the tau^2-airline rankings with an exact permutation p, the iso-quality ranking value, the registered configuration
list and the tau^2 side (35 airline test tasks x 3 trials, majority pass) on synthetic logs. Offline."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tau2_proxy_check as X  # noqa: E402


def _tau_b_bruteforce(x, y):
    c = d = tx = ty = 0
    n = len(x)
    for i, j in itertools.combinations(range(n), 2):
        a, b = (x[i] > x[j]) - (x[i] < x[j]), (y[i] > y[j]) - (y[i] < y[j])
        if a == 0:
            tx += 1
        if b == 0:
            ty += 1
        if a * b > 0:
            c += 1
        elif a * b < 0:
            d += 1
    n0 = n * (n - 1) // 2
    return (c - d) / math.sqrt((n0 - tx) * (n0 - ty))


@pytest.mark.parametrize("x,y", [([1, 2, 3, 4], [1, 2, 3, 4]), ([1, 2, 3, 4], [4, 3, 2, 1]),
                                 ([0, 0, 1, 2, 3], [0, 1, 1, 3, 2]), ([0.0, -0.2, -0.2, -0.1], [0.0, -0.1, -0.3, -0.1]),
                                 ([5, 1, 4, 2, 3, 3], [2, 1, 6, 5, 4, 4])])
def test_kendall_tau_b_matches_the_definition(x, y):
    assert X.kendall_tau_b(x, y) == pytest.approx(_tau_b_bruteforce(x, y))


def test_kendall_tau_b_is_undefined_for_a_constant_ranking():
    assert math.isnan(X.kendall_tau_b([1, 2, 3], [0, 0, 0]))
    r = X.kendall_permutation([1, 2, 3], [0, 0, 0])
    assert math.isnan(r["tau_b"]) and math.isnan(r["p_greater"]) and "undefined" in r["note"]


def test_exact_permutation_p_of_a_perfect_agreement_and_a_reversal():
    up = X.kendall_permutation([1, 2, 3, 4], [10, 20, 30, 40])
    assert up["tau_b"] == pytest.approx(1.0) and up["method"] == "exact" and up["n_permutations"] == 24
    assert up["p_greater"] == pytest.approx(1 / 24) and up["p_two_sided"] == pytest.approx(2 / 24)
    down = X.kendall_permutation([1, 2, 3, 4], [40, 30, 20, 10])
    assert down["tau_b"] == pytest.approx(-1.0) and down["p_less"] == pytest.approx(1 / 24)
    assert down["p_greater"] == pytest.approx(1.0)


def test_permutation_p_with_ties_counts_every_assignment():
    x, y = [0, 1, 1, 2], [0, 0, 1, 1]
    r = X.kendall_permutation(x, y)
    taus = [X.kendall_tau_b(x, [y[i] for i in p]) for p in itertools.permutations(range(4))]
    assert r["p_greater"] == pytest.approx(sum(t >= r["tau_b"] - 1e-12 for t in taus) / 24)
    assert r["p_two_sided"] == pytest.approx(sum(abs(t) >= abs(r["tau_b"]) - 1e-12 for t in taus) / 24)


def test_monte_carlo_beyond_the_exact_limit_approximates_the_exact_p():
    x, y = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]
    exact = X.kendall_permutation(x, y)
    mc = X.kendall_permutation(x, y, limit=5, n_mc=40000, seed=1)
    assert mc["method"].startswith("monte carlo") and mc["p_greater"] == pytest.approx(exact["p_greater"], abs=0.01)


def _row(name, e2, tau2):
    return {"configuration": name, "e2": e2, "tau2": tau2}


def test_iso_quality_ranks_qualifying_configurations_above_the_rest():
    rows = [_row("LLM-only", {"delta": 0.0, "qualifies": True, "cost_savings": 0.0},
                 {"delta": 0.0, "delta_pass1": 0.0, "qualifies": True, "cost_savings": 0.0}),
            _row("A", {"delta": -0.01, "qualifies": True, "cost_savings": 0.3},
                 {"delta": 0.0, "delta_pass1": 0.01, "qualifies": False, "cost_savings": 0.5}),
            _row("B", {"delta": -0.2, "qualifies": False, "cost_savings": 0.9},
                 {"delta": -0.1, "delta_pass1": -0.1, "qualifies": True, "cost_savings": -0.1})]
    v = X.ranking_vectors(rows)
    e2, t2 = v["iso_quality_savings"]
    assert e2[1] > e2[0] > e2[2]                       # qualifying A (30 %) > LLM-only (0) > non-qualifying B (90 %)
    assert t2[0] > t2[2] > t2[1]                       # non-qualifying A ties at the floor, below B's -10 %
    assert min(e2 + t2) < min(r[s]["cost_savings"] for r in rows for s in ("e2", "tau2"))
    assert v["net_cost_savings"] == ([0.0, 0.3, 0.9], [0.0, 0.5, -0.1])
    assert v["delta_success_pass1"][1] == [0.0, 0.01, -0.1]
    res = X.proxy(rows)
    assert {k: res[k]["role"] for k in res} == {"delta_success": "registered", "iso_quality_savings": "registered",
                                                "delta_success_pass1": "descriptive", "net_cost_savings": "descriptive"}
    assert all(res[k]["n"] == 3 for k in res)


def test_configuration_list_is_a5_4_e_and_the_e2a_sweep():
    names = [c[0] for c in X.CONFIGS]
    assert names[0] == "LLM-only" and "ASSIST k=3" in names and "policy GUARD" in names and len(names) == 9
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "jevgym_tool_select.yaml").read_text())
    sweep = next(s for s in cfg["sweeps"] if s["base"]["name"] == "jev_ts_cascade")
    assert [float(t) for t in X.SWEEP] == [float(v) for v in sweep["values"]]
    assert sweep["base"]["tool_params"]["tau"] == 0.9                    # the stated difference (Deviations (7))
    e2_arms = {c[1] for c in X.CONFIGS}
    assert {f"jev_ts_cascade_tau{t}" for t in X.SWEEP} <= e2_arms and {"jev_ts_assist_k3", "jev_ts_guard"} <= e2_arms
    assert dict((c[1], c[2]) for c in X.CONFIGS)["jev_ts_guard"] == "jev_guard"
    assert all(c[3] == "0.9" and c[4] == c[0].split("=")[1] for c in X.CONFIGS if c[0].startswith("CASCADE"))
    assert [c[2] for c in X.EXTRA] == ["jev_ts_cascade_tauhat_e2"]
    assert "share their argument threshold only at tau = 0.9" in X.STATED_DIFFERENCE


def _tau2_record(task: str, trial: int, success: bool, cost: float) -> dict:
    return {"task_id": f"airline/{task}#{trial}", "config": "x", "llm": "gemini/gemini-3.5-flash", "backend": "jev",
            "steps": [], "result": {"success": success}, "error": None,
            "usage": {"total": {"cost_usd": cost, "latency_ms": 1000.0}, "s1": {"cost_usd": 0.0},
                      "s2": {"input_tokens": 100, "calls": 1}},
            "meta": {"tau2": {"domain": "airline", "task_id": task, "trial": trial, "attempts": 1}}}


def test_tau2_side_uses_the_35_test_tasks_majority_pass_and_pass1(tmp_path):
    split = json.loads((ROOT / "configs" / "tau2" / "airline_task_ids.json").read_text())["split"]
    test, shadow = split["test"], split["shadow"]
    d = tmp_path / "airline"
    d.mkdir()
    ref, arm = [], []
    for i, t in enumerate(test + shadow):
        for k in range(3):
            ref.append(_tau2_record(t, k, True, 0.10))
            # arm: 2 of 3 trials on the first 10 test tasks (majority pass kept, pass^1 lower); shadow tasks all fail
            ok = (t in shadow and False) or (t not in shadow and not (i < 10 and k == 2))
            arm.append(_tau2_record(t, k, ok, 0.08))
    (d / "llm_only.jsonl").write_text("\n".join(json.dumps(r) for r in ref) + "\n")
    (d / "jev_ts_assist_k3.jsonl").write_text("\n".join(json.dumps(r) for r in arm) + "\n")
    cfg = [("LLM-only", "llm_only", "llm_only", "-", "-"), ("ASSIST k=3", "jev_ts_assist_k3", "jev_ts_assist_k3", "-", "-")]
    out = X.tau2_rows(tmp_path, cfg, n_boot=200, seed=0)
    a = out["ASSIST k=3"]
    assert a["n"] == len(test) == 35                                    # the 15 shadow tasks never enter
    assert a["delta"] == pytest.approx(0.0)                             # majority pass unchanged
    assert not a["qualifies"] and a["p_decisive"] > 0.05                # 35 tasks, no discordant one: NI at 5 pt not shown
    assert a["delta_pass1"] == pytest.approx(-(10 / 3) / 35)            # pass^1 drops on 10 tasks by 1/3
    assert a["cost_savings"] == pytest.approx(0.2)
    assert out["LLM-only"]["delta"] == 0.0 and out["_reference"]["n_tasks"] == 35
