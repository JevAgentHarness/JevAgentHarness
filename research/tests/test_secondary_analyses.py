"""Stage-1 secondary analyses (PREREGISTRATION A6.22 (c)): the H1 72-cell verdict rule (scripts/h1_threshold_theory.py),
the H3 class credit at 5 % / front / ranking / verdict (scripts/h3_delegation_classes.py), the H4 matching and decision
rule (scripts/h4_tost.py) and the \\res emitter (scripts/secondary_tex.py), on synthetic inputs whose answer is known;
plus the C1 / tau^2-refit thresholds read from the committed calibration fits. Offline."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import h1_threshold_theory as H1  # noqa: E402
import h3_delegation_classes as H3  # noqa: E402
import h4_tost as H4  # noqa: E402
import secondary_tex as ST  # noqa: E402
from jevharness.harness.tools import OTHER  # noqa: E402


# ====================================================================== H1
def test_h1_grid_is_the_registered_72_cells_with_the_registered_command(tmp_path):
    cs = H1.cells()
    assert len(cs) == 72 and len({c["id"] for c in cs}) == 72
    assert {c["arm"] for c in cs} == {"jev_shadow", "jev_shadow_s2lite", "intern_shadow", "intern_shadow_s2lite"}
    cmd = H1.command(cs[0], tmp_path, tmp_path / "x.json")
    s = " ".join(cmd)
    assert "-m jevharness.bench.threshold_theory" in s and "--arm jev_shadow --dp tool_select" in s
    assert "--train-frac 0.3 --seed 0 --taus h1 --rule general --task-value 0.1 --lam-per-hour 0" in s
    assert "--V" not in cmd and H1.command(cs[0], tmp_path, tmp_path / "x.json", v_w=True)[-6:-4] == ["--V", "0.1"]


def test_h1_v_undefined_detects_the_skipped_cell():
    assert H1.v_undefined({"by_V": [{"V": 0.5, "V_plugin": float("nan"), "note": "V is not positive"}]})
    assert H1.v_undefined({"by_V": [{"V": -0.1, "V_plugin": 0.2}]})
    assert not H1.v_undefined({"by_V": [{"V": 1.0, "V_plugin": 1.0, "regret": {}}]})


def _cell(i: int, d: float, W: float = 0.1, lam: float = 0.0, **kw: Any) -> dict[str, Any]:
    return {"id": f"c{i}", "backend": "jev", "s2": "flash", "dp": "tool_select", "W": W, "lam": lam, "d": d,
            "d_ci": [d - 0.01, d + 0.01], "outcome": H1.outcome(d), "n_test": 100, **kw}


def test_h1_clause1_counts_ties_as_non_wins_and_needs_36_estimable_cells():
    recs = [_cell(i, 0.02) for i in range(36)] + [_cell(36 + i, 0.0) for i in range(36)]
    v = H1.verdict(recs)
    assert v["clause1"]["wins"] == 36 and v["clause1"]["ties"] == 36 and not v["clause1"]["holds"]   # 36 is not > 36
    assert v["verdict"] == "falsified"
    recs[40] = _cell(40, 0.03)
    assert H1.verdict(recs)["clause1"]["holds"]
    few = [_cell(i, 0.02) for i in range(30)] + [{"id": f"a{i}", "W": 0.1, "lam": 0.0, "absent": "arm missing"}
                                                 for i in range(42)]
    v = H1.verdict(few)
    assert v["clause1"]["n_estimable"] == 30 and v["clause1"]["label"] == "not estimable"
    assert v["verdict"] == "not estimable" and len(v["clause1"]["absent"]) == 42


def _c2(over: bool, ret: bool) -> dict[str, Any]:
    return {"over_confident": over, "returns_to_front": ret, "gap": 0.1, "gap_lb": 0.05 if over else -0.01,
            "raw_ci": [0.01, 0.02], "recal_ci": [-0.01, 0.01] if ret else [0.005, 0.02]}


@pytest.mark.parametrize("flags,label,verdict", [
    ([(False, False)] * 4, "not applicable", "supported"),
    ([(True, True), (True, False), (False, False), (False, False)], "holds", "supported"),       # 1 of 2 >= half
    ([(True, False), (True, False), (True, True), (False, False)], "does not hold", "falsified"),  # 1 of 3 < half
])
def test_h1_clause2_and_the_verdict(flags, label, verdict):
    recs = [_cell(i, 0.02, W=1.0, lam=0.0, clause2=_c2(o, r)) for i, (o, r) in enumerate(flags)]
    recs += [_cell(10 + i, 0.02) for i in range(40)]
    v = H1.verdict(recs)
    assert v["clause1"]["holds"] and v["clause2"]["label"] == label and v["verdict"] == verdict


def test_h1_summarise_reads_the_module_output():
    res = {"n_paired": 10, "by_V": [{"V": 1.0, "V_plugin": 1.0, "regret": {
        "n_train": 3, "n_test": 7, "tau_best_test": 0.9,
        "candidates": {"tau_hat": {"tau": 0.95, "loss": 0.01, "regret": -0.001},
                       "tau_default": {"tau": 0.8, "loss": 0.03, "regret": 0.019},
                       "tau_hat_projection": {"tau": 0.99, "loss": 0.04, "regret": 0.029}},
        "default_minus_hat": {"diff": 0.02, "ci": [0.01, 0.03]}}}]}
    s = H1.summarise(res)
    assert s["outcome"] == "win" and s["d"] == pytest.approx(0.02) and s["tau_hat"] == 0.95
    assert s["outcome_projection"] == "loss" and s["d_projection"] == pytest.approx(-0.01)
    assert H1.summarise({"by_V": [{"V": 1.0, "V_plugin": 1.0, "regret": {"n_train": 0, "n_test": 5}}]})["absent"]


# ====================================================================== H3
def _dec(dp: str, mode: str, s1: Any = None, deferred: bool = False, **kw: Any) -> dict[str, Any]:
    return {"dp": dp, "mode": mode, "s1_value": s1, "deferred": deferred, "final_value": s1, **kw}


def _traj(tid: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"task_id": tid, "config": "arm", "result": {"success": True}, "steps": steps,
            "usage": {"s2": {"calls": 1}}, "meta": {}}


def _step(i: int, tool: str, args: dict[str, Any], decs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"step": i, "tool": tool, "args": args, "decisions": decs, "llm_calls": 0}


def test_h3_classify_credits_irreversible_copy_and_bounded_at_5_percent():
    trajs = []
    for t in range(10):
        steps = []
        for i in range(10):
            # task 0, step 0: an accepted cancel_order (1 of 100 tool opportunities = 1 % < 5 %); tasks 0-5: an
            # accepted ship_order at step 1 (6 of 100 = 6 %); every other step deferred to the LLM
            if (t == 0 and i == 0) or (t < 6 and i == 1):
                tool = "cancel_order" if i == 0 else "ship_order"
                d = [_dec("tool_select", "cascade", tool),
                     _dec("tool_params", "cascade", "ORD-1", options=["ORD-1", OTHER], gold_aux={"param_kind": "copy"})]
                steps.append(_step(i, tool, {"order_id": "ORD-1", "reason": "x"}, d))
            else:
                steps.append(_step(i, "get_orders", {"user_id": "U1"}, [_dec("tool_select", "cascade", "x", True)]))
        trajs.append(_traj(f"t{t}", steps))
    c = H3.classify(trajs, H3.JEVGYM_FATAL_TOOLS)
    assert c["tool_opportunities"] == 100 and c["counts"]["irreversible_tool"] == 7
    assert c["shares"]["irreversible_tool"] == pytest.approx(0.07) and c["irreversible_tools"]["ship_order"] == 6
    # argument slots: 7 two-argument calls + 93 one-argument calls = 107; 7 copy values accepted = 6.5 %
    assert c["arg_slots"] == 107 and c["shares"]["copy_arg"] == pytest.approx(7 / 107)
    assert c["credited"] == ["copy-dependent", "irreversible"] and c["riskiest"] == "irreversible"


def test_h3_classify_every_opportunity_modes_and_negligible():
    guard = [_traj("t0", [_step(0, "get_orders", {}, [_dec("tool_select", "guard", True)])])]
    assert H3.classify(guard, H3.JEVGYM_FATAL_TOOLS)["credited"] == ["bounded-recoverable"]
    meta = [_traj("t0", [_step(0, "get_orders", {}, [_dec("tool_select", "off"), _dec("stop", "guard", True)])])]
    assert H3.classify(meta, H3.JEVGYM_FATAL_TOOLS)["credited"] == ["meta"]
    tau2 = [_traj("t0", [_step(0, "cancel_reservation", {"reservation_id": "X"},
                               [_dec("tool_select", "off"), _dec("policy", "guard", True)])])]
    assert H3.classify(tau2, H3.TAU2_IRREVERSIBLE)["credited"] == ["bounded-recoverable"]
    defer = [_traj("t0", [_step(i, "get_orders", {"a": 1}, [_dec("tool_select", "cascade", "x", True)])
                          for i in range(20)])]
    c = H3.classify(defer, H3.JEVGYM_FATAL_TOOLS)
    assert c["negligible"] and c["credited"] == [] and c["riskiest"] is None


def test_h3_copy_class_from_param_kind_or_the_other_option():
    assert H3.is_copy({"gold_aux": {"param_kind": "copy"}, "options": ["a"]})
    assert not H3.is_copy({"gold_aux": {"param_kind": "bounded"}, "options": ["a", OTHER]})
    assert H3.is_copy({"options": ["a", OTHER]}) and not H3.is_copy({"options": ["a", "b"]})


@pytest.mark.parametrize("meta,ok,why", [
    ({"s1": {"describe": {"backend": "jev"}}, "config": {"tool_select": {"mode": "cascade"}}}, True, "Jev"),
    ({"s1": {"describe": {"backend": "intern_decision", "model": "internlm/Intern-Decision-4B"}},
      "config": {"stop": {"mode": "guard"}}}, True, "Intern-Decision-4B"),
    ({"s1": {"describe": {"backend": "intern_decision", "model": "internlm/Intern-Decision-2B"}},
      "config": {"tool_select": {"mode": "replace"}}}, False, "other size"),
    ({"s1": {"describe": {"backend": "jev"}}, "config": {"tool_select": {"mode": "off"}}}, False, "reference"),
    ({"s1": {"describe": {"backend": "jev"}}, "config": {"tool_select": {"mode": "shadow"}}}, False, "shadow"),
    ({"s1": {"describe": {"backend": "jev"}}, "config": {"reflection": {"mode": "always"}}}, False, "LLM control"),
    ({"s1": {"describe": {"backend": "bm25"}}, "config": {"tool_select": {"mode": "assist"}}}, False, "control"),
    ({"config": {"harness": {"stop": {"mode": "guard"}}, "policy_guard": {"mode": "guard"},
                 "s1": {"type": "jev", "provider": "openrouter"}}}, True, "Jev"),          # tau^2's logged config
])
def test_h3_universe(meta, ok, why):
    got, reason = H3.in_universe(meta)
    assert got is ok and why in reason


def _arm(i: str, credited: list[str], ni: bool, sav: float, delta: float, s2: str = "flash", suite: str = "L2",
         run: str = "E2a") -> dict[str, Any]:
    return {"id": i, "suite": suite, "s2": s2, "run": run, "credited": credited, "negligible": not credited,
            "riskiest": max(credited, key=H3.RISK.get) if credited else None, "ni": ni, "cost_savings": sav,
            "delta": delta, "n": 200, "cost_arm": 1 - sav, "success_arm": 0.9 + delta,
            "abs_ref": {"cost": 1.0, "success": 0.9}}


def test_h3_front_ranking_and_verdict():
    rows = [_arm("cascade", ["bounded-recoverable", "irreversible"], True, 0.14, 0.0),
            _arm("guard", ["bounded-recoverable"], True, -0.002, 0.0),
            _arm("stop", ["meta"], True, -0.06, -0.006),
            _arm("replace", ["irreversible"], False, 0.6, -0.2),
            _arm("lite_meta", ["meta"], True, -0.01, 0.03, s2="lite")]
    H3.fronts(rows)
    by = {r["id"]: r for r in rows}
    assert by["cascade"]["front"] and not by["guard"]["front"] and not by["stop"]["front"]   # reference dominates
    assert by["replace"]["front"] is None                                                   # not NI: no membership
    rank = H3.ranking(rows)
    assert rank["L2/flash"]["irreversible"]["median"] == pytest.approx(0.14)
    assert rank["L2/flash"]["irreversible"]["n_credited"] == 2 and rank["L2/flash"]["copy-dependent"]["median"] == 0.0
    assert rank["L2/flash"]["meta"]["median"] == pytest.approx(-0.06)
    kt = H3.taus(rank)
    v = H3.verdict(rows, kt)
    assert v["verdict"] == "falsified" and v["counterexamples"] == ["cascade"]
    rows[0]["ni"] = False
    H3.fronts(rows)
    assert H3.verdict(rows, {"x": {"tau_b": 0.2}})["verdict"] == "no registered counterexample"
    assert H3.verdict(rows, {"x": {"tau_b": -0.2}})["verdict"] == "falsified"


# ====================================================================== H4
def test_h4_matching_rule():
    assert H4.match(0.90, {})["reason"] == "no R4 checkpoint" and not H4.match(0.90, {})["matched"]
    m = H4.match(0.90, {"100": 0.885, "200": 0.909, "300": 0.95})
    assert m["matched"] and m["step"] == "200"
    tie = H4.match(0.90, {"100": 0.895, "200": 0.905})
    assert tie["matched"] and tie["step"] == "200"                         # ties -> the later step
    far = H4.match(0.90, {"100": 0.85})
    assert not far["matched"] and "> 1 pt" in far["reason"]
    res = H4.run(0.90, {}, None, None)
    assert res["verdict"] == "not tested at matched accuracy" and "without an R4 checkpoint" in res["note"]


def test_h4_decision_rule():
    n = 200
    r3 = {f"t{i}": float(i % 2) for i in range(n)}
    better = {k: 1.0 for k in r3}                                           # +50 pt: supported
    assert H4.decide(r3, better)["label"] == "supported"
    same = dict(r3)
    assert H4.decide(r3, same)["label"] == "falsified"                      # no discordant task: equivalence shown
    noisy = {k: (1.0 - v if i < 40 else v) for i, (k, v) in enumerate(r3.items())}   # 40 discordant, Delta 0
    d = H4.decide(r3, noisy)
    assert d["label"] in ("inconclusive at this n", "falsified") and d["n"] == n


# ====================================================================== the \\res emitter
@pytest.mark.parametrize("x,out", [(1.0, "1.00"), (0.99, "0.99"), (0.13, "0.13"), (0.28, "0.28"),
                                   (0.6309452181036016, "0.6309"), (0.8518233437140553, "0.8518"),
                                   (float("inf"), "always defer"), (None, "--")])
def test_threshold_format(x, out):
    assert ST.tau(x) == out


def test_calibration_keys_are_the_registered_fits():
    k = ST.keys_calibration()
    assert k["tau2/refit/tool_select/tauhat"] == "0.28" and k["tau2/refit/tool_params/tauhat"] == "1.00"
    assert k["tau2/refit/tool_select/n_paired"] == "488" and k["tau2/refit/tool_params/n_paired"] == "115"
    assert (k["calibration/c1/jev/flash/tool_select/tauhat"], k["calibration/c1/jev/flash/tool_params/tauhat"]) == \
        ("1.00", "0.99")
    assert (k["calibration/c1/jev/lite/tool_select/tauhat"], k["calibration/c1/jev/lite/tool_params/tauhat"]) == \
        ("always defer", "0.13")
    assert (k["calibration/c1/intern/flash/tool_select/tauhat"], k["calibration/c1/intern/flash/tool_params/tauhat"]) == \
        ("always defer", "0.6309")
    assert (k["calibration/c1/intern/lite/tool_select/tauhat"], k["calibration/c1/intern/lite/tool_params/tauhat"]) == \
        ("always defer", "0.8518")


def test_tex_output_defines_every_key_once():
    tex = ST.to_tex({"secondary/h1/verdict": "falsified", "tau2/refit/tool_select/tauhat": "0.28"}, ["x"])
    assert tex.count("\\defres{") == 2 and "\\defres{secondary/h1/verdict}{falsified}" in tex
    assert tex.rstrip().endswith("\\makeatother")


def test_h3_tau2_irreversible_tools_are_the_airline_write_tools():
    """The hard-coded tau^2 WRITE set equals tau2-bench's airline ToolType.WRITE tools when tau2 is installed."""
    tools = pytest.importorskip("tau2.domains.airline.tools")
    found = {name for name, fn in vars(tools.AirlineTools).items()
             if "WRITE" in str(getattr(fn, "__tool_type__", getattr(fn, "_tool_type", "")))}
    if not found:
        pytest.skip("tau2 tool types not readable in this version")
    assert found == set(H3.TAU2_AIRLINE_WRITE)
    assert H3.TAU2_IRREVERSIBLE == H3.TAU2_AIRLINE_WRITE | {"transfer_to_human_agents"}
