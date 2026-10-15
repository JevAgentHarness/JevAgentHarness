"""scripts/benchmark_difficulty.py (offline): the cluster-weighted mean and its paired bootstrap, the minimum-n subset
rule, the score block (n, clusters, spread over the decision models), the first-wrong-decision and error-type
classifiers on synthetic JevGym records, and -- when the S2-3 corpus is present -- the rule-based strata of C and the
reproduction of the registered corpus accuracies."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import benchmark_difficulty as BD  # noqa: E402


def _rec(gold, at):
    return {"dp": "tool_select", "gold": gold, "gold_aux": {"v": 1, "acceptable_tools": at}, "gold_status": "ok"}


def test_cluster_mean_and_bootstrap():
    per = {"t1": [1.0, 0.0], "t2": [1.0], "t3": [0.0, 0.0, 0.0]}
    assert BD.cluster_mean(per) == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    lo, hi = BD.bootstrap_ci([per], n_boot=500)
    assert 0.0 <= lo <= BD.cluster_mean(per) <= hi <= 1.0
    assert BD.bootstrap_ci([per], n_boot=500) == (lo, hi)                      # seeded
    assert BD.bootstrap_ci([{"a": [1.0], "b": [1.0]}], n_boot=100) == (1.0, 1.0)
    d = BD.bootstrap_ci([per, {"t1": [0.0], "t2": [0.0], "t3": [0.0]}], fn=lambda m: m[0] - m[1], n_boot=300)
    assert d[0] <= BD.cluster_mean(per) <= d[1]


def test_minimum_n_rule_and_score_block():
    items = [{"task": f"t{i % 7}", "x": i} for i in range(400)]
    kept, dropped = BD.subsets_by_rule(items, {"all": lambda it: True, "small": lambda it: it["x"] < 10})
    assert list(kept) == ["all"] and len(kept["all"]) == 400 and dropped == {"small": "n = 10 < 300: dropped"}
    blk = BD.score_block(items, {"jev": lambda it: 1.0, "intern": lambda it: float(it["x"] % 2 == 0),
                                 "llm": lambda it: None if it["x"] > 5 else 1.0})
    assert blk["n"] == 400 and blk["clusters"] == 7
    assert blk["models"]["jev"]["acc"] == 1.0 and blk["models"]["llm"]["n"] == 6
    assert blk["spread_8"] == pytest.approx(1.0 - blk["models"]["intern"]["acc"]) and blk["n_models"] == 2
    lo, hi = blk["models"]["intern"]["ci"]
    assert lo <= blk["models"]["intern"]["acc"] <= hi


def test_first_wrong_decision_types():
    near = BD.near_map()
    assert {n for n, _ in near["cancel_order"]} >= {"cancel_shipment", "cancel_order_item"}

    def traj(tool, gold, at, result=None):
        return {"steps": [{"tool": "find_user", "decisions": [dict(_rec("find_user", ["find_user"]))]},
                          {"tool": tool, "decisions": [dict(_rec(gold, at))]}],
                "result": result or {"n_calls": 3, "optimal_calls": 4, "side_effects": 0}}

    assert BD.first_error(traj("cancel_shipment", "cancel_order", ["cancel_order"]), near) == "near-duplicate distractor"
    assert BD.first_error(traj("cancel_order", "ask_user", ["ask_user"]), near) == "changed state before asking"
    assert BD.first_error(traj("finish", "get_orders", ["get_orders"]), near) == "finished too early"
    assert BD.first_error(traj("ask_user", "get_orders", ["get_orders"]), near) == "unneeded question"
    assert BD.first_error(traj("get_weather", "get_orders", ["get_orders"]), near) == "unrelated distractor"
    ok = traj("get_orders", "get_orders", ["get_orders"], {"n_calls": 9, "optimal_calls": 9, "side_effects": 0})
    assert BD.first_error(ok, near) == "every call on plan, final answer wrong"


def test_error_types_on_answers():
    near = BD.near_map()
    it = {"rec": _rec("ship_order", ["ship_order"]), "gold": "ship_order"}
    assert BD.error_type("ship_replacement", it, near) == "look-alike of the gold tool"
    assert BD.error_type("find_customer_by_email", it, near) == "other look-alike"
    assert BD.error_type(None, it, near) == "no valid answer"
    ask = {"rec": _rec("ask_user", ["ask_user", "find_user"]), "gold": "ask_user"}
    assert BD.error_type("cancel_order", ask, near) == "changed state before asking"
    assert BD.error_type("get_orders", ask, near) == "wrong core tool"


@pytest.mark.skipif(not (BD.S23 / "occurrences.jsonl.gz").exists(), reason="S2-3 corpus not present")
def test_corpus_strata_and_registered_accuracies():
    items, _ = BD.c_items()
    assert len(items) == 5314
    n = {k: sum(it["strata"][k] for it in items) for k in BD.C_SUBSETS}
    assert n["after_off_plan"] == 386 and all(v >= BD.MIN_N for v in n.values())
    assert all(it["strata"]["hard"] == (it["strata"]["after_off_plan"] or it["strata"]["ask_gold"]) for it in items)
    from jevharness.bench.stage2_corpus import answer_correct

    for m, registered in (("rl", 0.992), ("jev", 0.915), ("intern", 0.752)):
        a = {r["qhash"]: r for r in BD.read_jsonl(BD.S23 / f"answers_{m}.jsonl")}
        blk = BD.score_block(items, {m: lambda it, a=a: float(answer_correct(it["rec"], a.get(it["qhash"])))}, ci=False)
        assert round(blk["models"][m]["acc"], 3) == registered      # paper/generated/stage2.tex stage2/S2-3/accuracy/*
