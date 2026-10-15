"""scripts/primaries.py, PREREGISTRATION A6.22 (a) 7 and (e) 4-5: the C1 constituents' delegation labels from the two
coverages of A6.22 (a) 5 (tool-decision and argument coverage) with A6.20 (e)'s 5 % rule, the dropped-task listing next
to every full-set row, and the tau^2 attempt counts of the re-run ledger. Offline, synthetic data."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import primaries as P  # noqa: E402

NAN = float("nan")


def _maps(n: int, wins: int, losses: int) -> tuple[dict[str, float], dict[str, float]]:
    ref, arm = {}, {}
    for i in range(n):
        t = f"t{i:04d}"
        ref[t], arm[t] = (0.0, 1.0) if i < wins else (1.0, 0.0) if i < wins + losses else (1.0, 1.0)
    return ref, arm


class _Args:
    def __init__(self, tmp: Path) -> None:
        self.runs, self.run, self.tau2, self.e1 = str(tmp), None, None, None
        self.gate, self.c7_arm, self.c7_selection, self.tau2_secondary = str(P.GATE_FILE), None, None, None
        self.select_c7, self.out, self.n_boot, self.seed = False, str(tmp / "out"), 200, 0
        self.max_tasks, self.allow_simulated, self.allow_dev_tasks = None, True, True
        self.allow_partial_population, self.strict = False, False


def _pair(ref: dict[str, float], arm: dict[str, float], cov: dict[str, Any]) -> dict[str, Any]:
    tab = lambda y: {t: {"status": "ok", "y": v, "cost": 0.01, "s1_cost": 0.0, "tokens": 1.0, "latency": 1.0}  # noqa: E731
                     for t, v in y.items()}
    keys = sorted(ref)
    st = {"ok": len(keys), "errored": 0, "retry_pending": 0, "missing": 0, "incomplete": 0}
    return {"n_expected": len(keys), "n_pairs": len(keys), "n_dropped": 0, "status_ref": st, "status_arm": st,
            "keys": keys, "ref_y": ref, "arm_y": arm, "ref_y_full": ref, "arm_y_full": arm, "arm": "a", "ref": "r",
            "run": "x", "ref_run": "x", "tab_arm": tab(arm), "tab_ref": tab(ref), "s2_arm": "m", "s2_ref": "m",
            "coverage": cov, "dropped": []}


def _cov(ratio: float, tool: float, arg: float) -> dict[str, Any]:
    return {"ratio": ratio, "n": 100, "tool_coverage": tool, "arg_coverage": arg, "arg_coverage_s1_calls": NAN,
            "n_arg_slots": 200, "n_arg_delegated": int(round(200 * arg)) if arg == arg else 0}


def _decided(tmp_path, monkeypatch, jev_cov: dict[str, Any], intern_cov: dict[str, Any], label: str = "C1L"):
    an = P.Analysis(_Args(tmp_path))
    pairs = {"E2a": _pair(*_maps(500, 0, 0), jev_cov), "E2a_intern": _pair(*_maps(500, 30, 5), intern_cov)}
    monkeypatch.setattr(an, "jevgym_pair", lambda where, run_key, *a, **k: pairs[run_key])
    gate = {"co_primary": True}
    C = an.conj_primary(label, gate)
    an._decide(C, {"rank": 1, "m": 11, "alpha_adj": 0.05 / 11, "p_holm": 0.001, "reject": True, "reached": True}, gate)
    return C


# ------------------------------------------------------------------ the 5 % rule (A6.20 (e)) on the two coverages
@pytest.mark.parametrize("tool,arg,cls", [
    (0.0, 0.0, "zero"), (0.0, NAN, "zero"), (NAN, 0.0, "zero"),
    (0.04, 0.0, "negligible"), (0.049, 0.049, "negligible"), (0.0, 0.01, "negligible"),
    (0.05, 0.0, "non-negligible"), (0.0, 0.06, "non-negligible"), (0.3, 0.2, "non-negligible"),
    (NAN, NAN, "unknown"), (None, None, "unknown")])
def test_delegation_class_is_the_5_percent_rule_on_both_coverages(tool, arg, cls):
    assert P.delegation_class(tool, arg) == cls


def test_zero_coverage_label_comes_from_the_two_coverages(tmp_path, monkeypatch):
    C = _decided(tmp_path, monkeypatch, _cov(0.0, 0.0, 0.0), _cov(0.4, 0.5, 0.3))
    assert C["jev"]["delegation"] == "zero" and C["intern"]["delegation"] == "non-negligible"
    assert C["jev"]["tool_coverage"] == 0.0 and C["jev"]["arg_coverage"] == 0.0
    assert C["replication"] == "cross-backend replication of policy NI"
    assert C["decision"].startswith("NI: policy NI with zero coverage for Jev (") and "A6.22 (a)" in C["decision"]


def test_negligible_coverage_reads_as_policy_ni_even_when_the_pooled_ratio_is_large(tmp_path, monkeypatch):
    # pooled delegation ratio 30 % but both registered coverages < 5 %: negligible (A6.20 (e)), not "delegating"
    C = _decided(tmp_path, monkeypatch, _cov(0.30, 0.03, 0.01), _cov(0.4, 0.5, 0.3))
    assert C["jev"]["delegation"] == "negligible"
    assert C["replication"] == "cross-backend replication of policy NI"
    assert C["decision"].startswith("NI: policy NI with negligible coverage (< 5 %) for Jev")


def test_a_zero_pooled_ratio_no_longer_decides_the_label(tmp_path, monkeypatch):
    # the old rule (pooled ratio == 0) would say "policy NI"; the argument coverage of 8 % makes the delegation real
    C = _decided(tmp_path, monkeypatch, _cov(0.0, 0.0, 0.08), _cov(0.4, 0.5, 0.3))
    assert C["jev"]["delegation"] == "non-negligible" and C["intern"]["delegation"] == "non-negligible"
    assert C["replication"] == "cross-backend replication" and C["decision"] == "NI"


def test_both_zero_and_negligible_constituents_are_named(tmp_path, monkeypatch):
    C = _decided(tmp_path, monkeypatch, _cov(0.0, 0.0, 0.0), _cov(0.02, 0.02, 0.0))
    assert C["decision"].startswith("NI: policy NI with zero coverage for Jev; negligible coverage (< 5 %) for "
                                    "Intern-Decision")
    assert C["replication"] == "cross-backend replication of policy NI"


def test_c2_constituents_carry_no_delegation_class(tmp_path, monkeypatch):
    C = _decided(tmp_path, monkeypatch, _cov(0.0, 0.0, 0.0), _cov(0.0, 0.0, 0.0), label="C2L")
    assert "delegation" not in C["jev"] and "tool_coverage" not in C["jev"]
    assert C["replication"] == "cross-backend replication" and C["decision"] == "NI"


# ------------------------------------------------------------------ coverages computed from real trajectories
def _traj(task: str, arm: str, delegate: bool) -> dict[str, Any]:
    """Two agent steps: an executed get_orders(user_id) call, then finish(answer). ``delegate``: System-1's tool choice
    and the user_id argument are accepted (cascade, not deferred); else always-defer (deferred records, no S1 answer)
    and the LLM writes both calls (no tool_params record)."""
    def ts(tool: str) -> dict[str, Any]:
        return {"dp": "tool_select", "mode": "cascade", "question": "next tool", "options": ["get_orders", "finish"],
                "s1_value": tool if delegate else None, "s1_conf": 0.99 if delegate else None,
                "deferred": not delegate, "final_value": tool, "gold": tool}
    steps = [{"step": 0, "tool": "get_orders", "args": {"user_id": "U1"}, "llm_calls": 0 if delegate else 1,
              "decisions": [ts("get_orders")] + ([{"dp": "tool_params", "mode": "cascade", "question": "user_id",
                                                   "options": ["U1", "U2"], "s1_value": "U1", "s1_conf": 0.99,
                                                   "deferred": False, "final_value": "U1", "gold": "U1"}]
                                                 if delegate else [])},
             {"step": 1, "tool": "finish", "args": {"answer": "done"}, "llm_calls": 1, "decisions": [ts("finish")]}]
    u = {"calls": 2, "input_tokens": 100, "output_tokens": 5, "cost_usd": 0.01, "latency_ms": 1000.0}
    return {"task_id": task, "config": arm, "backend": "simulated", "llm": "m", "steps": steps,
            "result": {"success": True}, "usage": {"total": u, "s2": u, "s1": {"cost_usd": 0.0, "calls": 0}},
            "error": None, "meta": {}}


def test_load_arm_reads_the_a6_22_coverages(tmp_path):
    for name, deleg in (("always_defer", False), ("delegating", True)):
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for i in range(3):
                f.write(json.dumps(_traj(f"jevgym-clarify-{i}", name, deleg)) + "\n")
    off = P.load_arm(tmp_path / "always_defer.jsonl")["coverage"]
    assert off["tool_coverage"] == 0.0 and off["arg_coverage"] == 0.0 and off["n_arg_slots"] == 6
    assert P.delegation_class(off["tool_coverage"], off["arg_coverage"]) == "zero"
    on = P.load_arm(tmp_path / "delegating.jsonl")["coverage"]
    assert on["tool_coverage"] == 1.0 and on["arg_coverage"] == pytest.approx(3 / 6)   # user_id of 3 calls / 6 slots
    assert on["ratio"] == pytest.approx(9 / 9) and "by_dp" in on                      # the pooled ratio is kept
    assert P.delegation_class(on["tool_coverage"], on["arg_coverage"]) == "non-negligible"


# ------------------------------------------------------------------ dropped tasks next to the full-set row (A6.22 (e) 5)
def _ep(task: str, success: bool, error: str | None = None, attempts: int = 1, trial: int | None = None,
        term: str | None = None) -> dict[str, Any]:
    return {"episode": task if trial is None else f"{task}#{trial}", "task": task, "trial": trial,
            "success": float(success), "cost": 0.01, "s1_cost": 0.0, "latency": 1.0, "llm_input_tokens": 100.0,
            "error": error, "attempts": attempts, "attempts_logged": True, "llm": "m", "backend": None,
            "termination_reason": term}


def test_jevgym_dropped_tasks_are_listed_with_family_error_class_attempts_and_other_outcome():
    c, m, ok = "jevgym-clarify-200000-d60-near-strict", "jevgym-multi-200001-d60-near-strict", \
        "jevgym-clarify-200002-d60-near-strict"
    err = "APIStatusError: 503 upstream unavailable\nTraceback (most recent call last):\n  File ..."
    arm = {"episodes": {e["episode"]: e for e in [_ep(c, False, err, attempts=3), _ep(m, True), _ep(ok, True)]}}
    ref = {"episodes": {e["episode"]: e for e in [_ep(c, True), _ep(ok, False)]}}           # m missing in the ref
    exp = [c, m, ok]
    p = P.pair_tables(P.task_table(ref, exp), P.task_table(arm, exp), exp, names=("llm_only", "jev_x"))
    assert p["keys"] == [ok] and p["n_dropped"] == 2
    d1, d2 = p["dropped"]
    assert d1 == {"task": c, "family": "clarify", "errored_in": ["jev_x"], "status": {"jev_x": "errored"},
                  "error_class": {"jev_x": "APIStatusError: 503 upstream unavailable"}, "attempts": {"jev_x": 3},
                  "other_outcome": {"llm_only": "success"}}
    assert d2["family"] == "multi" and d2["errored_in"] == ["llm_only"] and d2["status"] == {"llm_only": "missing"}
    assert d2["error_class"] == {"llm_only": None} and d2["other_outcome"] == {"jev_x": "success"}
    lines = P._dropped_lines({"dropped": p["dropped"]})
    assert lines[0].startswith("  - dropped from the pair (2; A6.22 (e) 5)")
    assert f"`{c}` [clarify]: `jev_x` errored (APIStatusError: 503 upstream unavailable; 3 attempt(s)); other arm: " \
           "`llm_only` success" in lines[1]


def test_tau2_dropped_tasks_name_domain_split_trials_and_termination_reason():
    # airline 2 is a test task, airline 0 a shadow task of the committed split; retail has no shadow split
    eps = [_ep("airline/2", True, trial=k) for k in range(3)]
    eps += [_ep("airline/0", True, trial=0), _ep("airline/0", True, trial=1),
            _ep("airline/0", False, "tau2 termination_reason=infrastructure_error", attempts=3, trial=2,
                term="infrastructure_error")]
    eps += [_ep("retail/5", False, "RuntimeError: side-car\n...", attempts=2, trial=1, term="agent_stop")] + \
           [_ep("retail/5", True, trial=k) for k in (0, 2)]
    arm = {"episodes": {e["episode"]: e for e in eps}}
    ref = {"episodes": {e["episode"]: e for t in ("airline/2", "airline/0", "retail/5")
                        for e in [_ep(t, True, trial=k) for k in range(3)]}}
    exp = ["airline/2", "airline/0", "retail/5"]
    ta = P.task_table(arm, exp, trials=3, retry_rule=True)
    tb = P.task_table(ref, exp, trials=3, retry_rule=True)
    assert ta["airline/0"]["status"] == "errored" and ta["retail/5"]["status"] == "retry_pending"
    p = P.pair_tables(tb, ta, exp, names=("llm_only", "c7"))
    by = {d["task"]: d for d in p["dropped"]}
    assert by["airline/0"]["family"] == "airline (shadow)" and by["retail/5"]["family"] == "retail (test)"
    assert by["airline/0"]["error_class"] == {"c7": "termination_reason=infrastructure_error"}
    assert by["airline/0"]["errored_trials"] == {"c7": [2]} and by["airline/0"]["attempts"] == {"c7": 3}
    assert by["retail/5"]["error_class"] == {"c7": "termination_reason=agent_stop; RuntimeError: side-car"}
    assert by["retail/5"]["status"] == {"c7": "retry_pending"} and by["retail/5"]["attempts"] == {"c7": 2}
    assert by["airline/0"]["other_outcome"] == {"llm_only": "success"}


def test_episode_reads_tau2_attempts_from_the_rerun_ledger_stamp():
    base = {"task_id": "airline/3#1", "result": {"success": False, "termination_reason": "agent_error"},
            "error": "tau2 termination_reason=agent_error", "usage": {}}
    e = P._episode({**base, "meta": {"tau2": {"domain": "airline", "task_id": "3", "trial": 1, "attempts": 2}}}, 1)
    assert e["attempts"] == 2 and e["attempts_logged"] and e["termination_reason"] == "agent_error"
    e = P._episode({**base, "meta": {"tau2": {"domain": "airline", "task_id": "3", "trial": 1}}}, 1)
    assert e["attempts"] == 1 and not e["attempts_logged"]          # no ledger: the count is not logged
    e = P._episode({"task_id": "jevgym-multi-1", "result": {"success": True}, "usage": {}, "meta": {}}, 2)
    assert e["attempts"] == 2 and e["attempts_logged"]              # JevGym: the record count is the attempt count


def _tau2_traj(task: str, trial: int, success: bool, attempts: int | None, error: str | None = None) -> dict:
    dom, tid = task.split("/")
    tau = {"domain": dom, "task_id": tid, "trial": trial, **({"attempts": attempts} if attempts is not None else {})}
    u = {"calls": 1, "input_tokens": 10, "output_tokens": 1, "cost_usd": 0.01, "latency_ms": 10.0}
    return {"task_id": f"{task}#{trial}", "config": "x", "llm": "m", "steps": [], "error": error,
            "result": {"success": success, "termination_reason": "agent_error" if error else "user_stop"},
            "usage": {"total": u, "s2": u}, "meta": {"tau2": tau}}


@pytest.mark.parametrize("attempts,status", [(None, "errored"), (1, "retry_pending"), (3, "errored")])
def test_c7_uses_logged_tau2_attempts_for_the_retry_rule(tmp_path, attempts, status):
    """``attempts`` None: no ledger stamp anywhere (every error final, as before A6.22); else every simulation is
    stamped (1 for the clean ones) and retail/0 trial 1 errored after ``attempts`` executions."""
    airline_test = json.loads(P.AIRLINE_SPLIT.read_text())["split"]["test"]
    tasks = [f"airline/{t}" for t in airline_test] + [f"retail/{i}" for i in range(P.TAU2_RETAIL_N)]
    for arm in ("llm_only", "cascade"):
        for dom in ("airline", "retail"):
            (tmp_path / "tau2" / dom).mkdir(parents=True, exist_ok=True)
            with open(tmp_path / "tau2" / dom / f"{arm}.jsonl", "w") as f:
                for t in (x for x in tasks if x.startswith(dom + "/")):
                    for k in range(3):
                        bad = arm == "cascade" and t == "retail/0" and k == 1
                        a = None if attempts is None else (attempts if bad else 1)
                        f.write(json.dumps(_tau2_traj(t, k, True, a, error="tau2 termination_reason=agent_error"
                                                      if bad else None)) + "\n")
    args = _Args(tmp_path)
    args.tau2, args.c7_arm = str(tmp_path / "tau2"), "cascade"
    an = P.Analysis(args)
    c7 = an.c7({"tau2_arm_family": "cascade"})
    blk = c7["jev"]
    assert blk["attempts_logged"] is (attempts is not None)
    assert blk["status_arm"][status] == 1
    (d,) = blk["dropped"]
    assert d["task"] == "retail/0" and d["errored_trials"] == {"cascade": [1]} and d["family"] == "retail (test)"
    assert d["error_class"] == {"cascade": "termination_reason=agent_error"}
    flagged = [f for f in an.flags.items if f["where"] == "C7" and "retries outstanding" in f["msg"]]
    assert bool(flagged) is (status == "retry_pending")
    assert math.isfinite(blk["p_decisive"])
