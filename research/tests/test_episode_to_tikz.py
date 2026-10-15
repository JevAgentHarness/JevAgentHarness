"""scripts/episode_to_tikz.py: the deterministic episode- and step-selection rule of Figure 1 on tiny synthetic
trajectories (lowest seed among successful, short episodes with a deferred tool choice and an accepted one that ran
without any LLM call; the lowest-confidence deferral and the first such acceptance after it), the top-k options, the
LaTeX escaping, the tau-hat* consistency check, and that no gold label reaches the generated keys."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import episode_to_tikz as E  # noqa: E402

OPTIONS = ["find_user", "get_orders", "ask_user", "cancel_order", "verify_identity", "finish"]
GOLD = "GOLD_SENTINEL_LABEL"


def _dec(conf: float, top: str, deferred: bool, final: str | None = None, second: str = "ask_user") -> dict:
    dist = {o: 1e-6 for o in OPTIONS}
    dist[top] = conf
    if second != top:
        dist[second] = 1.0 - conf
    return {"dp": "tool_select", "mode": "cascade", "options": list(OPTIONS), "s1_value": top, "s1_conf": conf,
            "s1_dist": dist, "deferred": deferred, "final_value": final or top, "gold": GOLD,
            "gold_aux": {"acceptable_tools": [GOLD]}, "state": "TASK: The customer Ann Lee asked us to cancel their "
            "pending order. Please do it.\n\nCURRENT OBSERVATION: ..."}


def _step(i: int, tool: str, dec: dict, llm_calls: int, extra: list | None = None) -> dict:
    return {"step": i, "tool": tool, "decisions": [dec] + list(extra or []), "llm_calls": llm_calls,
            "oracle_tool": GOLD}


def _calls(steps: list[dict]) -> list[dict]:
    out = []
    for s in steps:
        out.append({"component": "s1", "decision_point": "tool_select", "step": s["step"], "cost_usd": 1e-4,
                    "latency_ms": 200.0})
        if s["decisions"][0]["deferred"]:
            out.append({"component": "s2", "decision_point": "tool_select", "step": s["step"], "cost_usd": 4e-3,
                        "latency_ms": 800.0})
    return out


def _episode(seed: int, steps: list[dict], success: bool = True, tau: float = 1.0) -> dict:
    return {"task_id": f"jevgym-clarify-{seed}-d60-near-strict", "result": {"success": success}, "error": None,
            "steps": steps, "calls": _calls(steps),
            "meta": {"config": {"tool_select": {"mode": "cascade", "tau": tau}}}}


def _good(seed: int) -> dict:
    """find_user (deferred, 0.82) -> get_orders (accepted, but an LLM reflection call) -> ask_user (deferred, 0.62)
    -> cancel_order (accepted, no LLM call) -> finish (deferred, 0.93)."""
    return _episode(seed, [
        _step(0, "find_user", _dec(0.82, "ask_user", True, final="find_user", second="find_user"), 1),
        _step(1, "get_orders", _dec(1.0, "get_orders", False), 1),
        _step(2, "ask_user", _dec(0.62, "ask_user", True, second="cancel_order"), 1),
        _step(3, "cancel_order", _dec(1.0, "cancel_order", False), 0,
              extra=[{"dp": "tool_params", "mode": "cascade", "deferred": False},
                     {"dp": "tool_params", "mode": "cascade", "deferred": False}]),
        _step(4, "finish", _dec(0.93, "finish", True), 1),
    ])


def test_seed_of_and_order():
    assert E.seed_of("jevgym-multi-300011-d60-near-strict") == 300011
    assert E.seed_of("no-seed-here") == sys.maxsize


def test_selection_rule_skips_failed_long_and_llm_accepted_episodes():
    failed = _episode(100, _good(100)["steps"], success=False)
    only_llm_accept = _episode(101, [  # its only accepted tool choice ran with an LLM call in the same step
        _step(0, "find_user", _dec(0.5, "find_user", True), 1),
        _step(1, "get_orders", _dec(1.0, "get_orders", False), 1)])
    no_defer = _episode(102, [_step(0, "find_user", _dec(1.0, "find_user", False), 0)])
    too_long = _episode(103, _good(103)["steps"] + [
        _step(5, "finish", _dec(0.9, "finish", True), 1), _step(6, "finish", _dec(0.9, "finish", True), 1)])
    later, chosen = _good(105), _good(104)
    eps = [later, too_long, no_defer, only_llm_accept, failed, chosen]
    assert not any(E.qualifies(e) for e in (failed, only_llm_accept, no_defer, too_long))
    assert E.select_episode(eps)["task_id"] == chosen["task_id"]          # lowest qualifying seed, input order ignored
    with pytest.raises(SystemExit):
        E.select_episode([failed, no_defer])


def test_step_rule_lowest_confidence_deferral_then_first_acceptance_after_it():
    defer, acc = E.pick_steps(_good(1))
    assert (defer["step"], acc["step"]) == (2, 3)          # 0.62 < 0.82 < 0.93; step 1 had an LLM call
    # no qualifying acceptance after the deferral -> the last one before it
    ep = _episode(2, [_step(0, "get_orders", _dec(1.0, "get_orders", False), 0),
                      _step(1, "cancel_order", _dec(1.0, "cancel_order", False), 0),
                      _step(2, "ask_user", _dec(0.4, "ask_user", True), 1)])
    defer, acc = E.pick_steps(ep)
    assert (defer["step"], acc["step"]) == (2, 1)
    # ties in confidence: the earliest deferral
    ep = _episode(3, [_step(0, "ask_user", _dec(0.5, "ask_user", True), 1),
                      _step(1, "ask_user", _dec(0.5, "ask_user", True), 1),
                      _step(2, "cancel_order", _dec(1.0, "cancel_order", False), 0)])
    assert E.pick_steps(ep)[0]["step"] == 0


def test_top_options_ties_follow_the_offered_order():
    d = _dec(0.63, "ask_user", True, second="cancel_order")
    top = E.top_options(d)
    assert [n for n, _ in top] == ["ask_user", "cancel_order", "find_user"]   # 1e-6 ties: first offered option
    assert E.fmt_p(top[2][1]) == r"$<$0.01" and E.fmt_p(0.63) == "0.63"


def test_figure_keys_escape_count_and_never_show_gold():
    ep = _good(300000)
    keys = E.figure_keys(ep, tauhat=1.0)
    k = {key.split("fig1/", 1)[1]: v for key, v in keys.items()}
    assert k["defer/step"] == "3" and k["accept/step"] == "4"                 # 1-based
    assert k["defer/conf"] == "0.62" and k["accept/conf"] == "1.00" and k["tauhat"] == "1.00"
    assert k["defer/opt/1/name"] == r"ask\_user" and k["defer/opt/2/name"] == r"cancel\_order"
    assert k["defer/chosen"] == r"ask\_user" and k["accept/chosen"] == r"cancel\_order"
    assert k["n_options"] == str(len(OPTIONS)) and k["defer/n_more"] == str(len(OPTIONS) - E.TOP_K)
    assert k["accept/llm_calls"] == "0" and k["accept/n_args_model"] == "2"
    assert [k[f"step/{i}/by"] for i in range(1, 6)] == ["s2", "s1", "s2", "s1", "s2"] and k["n_by_model"] == "2"
    assert k["cost/s1"] == "0.00010" and k["cost/s2"] == "0.0040" and k["latency/s2"] == "800"
    assert k["task"] == "The customer Ann Lee asked us to cancel their pending order."
    tex = E.to_tex(keys, "synthetic", "now")
    assert GOLD not in tex and "\\defres{fig1/defer/opt/1/p}{0.62}" in tex
    with pytest.raises(SystemExit):                                           # profile and episode disagree on tau
        E.figure_keys(ep, tauhat=0.9)


def test_short_task_cuts_at_a_word():
    s = E.short_task("TASK: " + "word " * 30 + ".", max_chars=20)
    assert s.endswith("\u2026") and len(s) <= 21 and " " not in s[-2:]
