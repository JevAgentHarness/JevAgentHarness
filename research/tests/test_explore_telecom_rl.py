"""Offline tests of the exploratory E1 scripts (scripts/explore_telecom_rl.py, scripts/explore_common.py; Deviations
entries of 2026-10-05 in docs/PREREG_A7_STAGE2.md): task-id file, rendering, record loading and the go/no-go."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_common as C  # noqa: E402
import explore_telecom_rl as E  # noqa: E402

REGISTERED_IDS_SHA = "ab411862eefd5f7ef3b584796851a0963719af848cdf1b0c802644e652d65b50"


def test_task_id_file_matches_the_registration_and_round_two_uses_validation_ids_6_to_10():
    d = json.loads(E.IDS_OUT.read_text())
    inv = json.loads(E.INVENTORY.read_text())
    val = inv["validation"]["task_ids"]
    assert hashlib.sha256("\n".join(d["task_ids"]).encode()).hexdigest() == d["sha256"] == REGISTERED_IDS_SHA
    assert len(d["task_ids"]) == 150 and d["validation_gonogo"] == val[:5] and d["validation_gonogo_2"] == val[5:10]
    assert not (set(d["task_ids"]) & set(val))
    assert E.GONOGO_ROUNDS[2][0] == "validation_gonogo_2"


def test_committed_rendering_is_the_render_output_with_the_defer_fallback():
    from jevharness.harness import profile as HP
    t2, _ = E.render(HP.load_profile(E.PROFILE))
    assert t2["respond_fallback"] == "defer" and t2["s1"]["name"] == E.SERVED
    assert yaml.safe_load(E.RENDER_OUT.read_text()) == t2


def test_load_latest_strips_the_tau2_trial_suffix(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [{"task_id": "telecom/[x]a[PERSONA:Easy]#0", "error": "boom"},
                                                   {"task_id": "telecom/[x]a[PERSONA:Easy]#0", "error": None},
                                                   {"task_id": "jevgym-clarify-300002-d60-near-strict"}]) + "\n")
    by, n = C.load_latest([p])
    assert set(by) == {"telecom/[x]a[PERSONA:Easy]", "jevgym-clarify-300002-d60-near-strict"}
    assert by["telecom/[x]a[PERSONA:Easy]"]["error"] is None and n["telecom/[x]a[PERSONA:Easy]"] == 2


def _traj(tool: str = "get_details_by_id") -> dict:
    opts = ["get_details_by_id", "respond_to_user"]
    return {"error": None,
            "calls": [{"component": "s1", "model": E.SERVED, "priced": True, "cost_usd": 0.0},
                      {"component": "s2", "model": "gemini-3.5-flash", "priced": True, "cost_usd": 0.01}],
            "usage": {"total": {"cost_usd": 0.01}, "s1": {"cost_usd": 0.0}, "s2": {"cost_usd": 0.01}},
            "steps": [{"tool": tool, "decisions": [
                {"dp": "tool_select", "options": opts, "s1_value": "respond_to_user", "s1_conf": 0.97,
                 "s1_dist": {"get_details_by_id": 0.03, "respond_to_user": 0.97}, "deferred": True,
                 "s2_value": tool, "final_value": tool}]}]}


def _sim(tid: str, tool: str = "get_details_by_id", term: str = "user_stop") -> dict:
    return {"task_id": tid, "termination_reason": term, "agent_cost": 0.01, "user_cost": 0.005,
            "messages": [{"role": "assistant", "tool_calls": [{"name": tool, "arguments": {"id": "L1"}}]},
                         {"role": "tool", "error": tool != "get_details_by_id"}]}


def test_gonogo_passes_clean_records_and_fails_undeclared_tool_loops():
    ids = ["[mms_issue]a[PERSONA:Easy]", "[mms_issue]b[PERSONA:Hard]"]
    trajs = {f"telecom/{t}": _traj() for t in ids}
    r = E.gonogo(trajs, [_sim(t) for t in ids], ids, ref_cost_per_sim=0.16, spent_before=0.03)
    assert r["go"], r["checks"]
    assert r["checks"]["no_crash_or_loop"]["executed_calls_to_unknown_tools"] == 0
    bad = E.gonogo({**trajs, f"telecom/{ids[0]}": _traj("get_line")},
                   [_sim(ids[0], "get_line", "too_many_errors"), _sim(ids[1])], ids, 0.16, 0.03)
    assert not bad["go"] and not bad["checks"]["no_crash_or_loop"]["ok"]
    assert bad["checks"]["no_crash_or_loop"]["executed_calls_to_unknown_tools"] == 1
    missing = E.gonogo({}, [], ids, 0.16, 0.0)
    assert not missing["go"] and missing["checks"]["complete_no_error"]["n_got"] == 0
    over = E.gonogo(trajs, [_sim(t) for t in ids], ids, ref_cost_per_sim=0.6, spent_before=0.03)
    assert not over["go"] and not over["checks"]["projected_spend"]["ok"]


def test_gonogo_tex_keys_cover_every_round_and_the_decision(tmp_path):
    ids = ["[mms_issue]a[PERSONA:Easy]", "[mms_issue]b[PERSONA:Hard]"]
    trajs = {f"telecom/{t}": _traj() for t in ids}
    r1 = E.gonogo({**trajs, f"telecom/{ids[0]}": _traj("get_line")},
                  [_sim(ids[0], "get_line", "too_many_errors"), _sim(ids[1])], ids, 0.16, 0.03)
    r2 = E.gonogo(trajs, [_sim(t) for t in ids], ids, 0.16, 0.06)
    for r in (r1, r2):
        r["task_ids"] = ids
    k = E.gonogo_tex_keys({1: r1, 2: r2})
    assert k["gonogo/decision"] == "go" and k["gonogo/r1/go"] == "no" and k["gonogo/r2/go"] == "yes"
    assert k["gonogo/r1/too_many_errors"] == "1" and k["gonogo/r1/unknown_tool_calls"] == "1"
    assert k["gonogo/n_val_tasks"] == "4" and k["gonogo/spend"] == "0.06" and k["gonogo/r2/max_repeats"] == "0"
    tex = C.write_tex("telecom_rl", k, "test", tmp_path / "explore.tex").read_text()
    assert r"\defres{explore/telecom_rl/gonogo/decision}{go}" in tex


def test_e2_kendall_tau_b_matches_the_definition():
    import explore_startlux_live as L
    assert L.kendall_tau_b([0.8, 2, 4, 9, 27], [1, 2, 3, 4, 5]) == 1.0
    assert L.kendall_tau_b([0.8, 2, 4, 9, 27], [5, 4, 3, 2, 1]) == -1.0
    assert L.kendall_tau_b([1, 2], [1, 2]) is None
    assert abs(L.kendall_tau_b([1, 2, 3], [1, 1, 2]) - 2 / (3 * 2) ** 0.5) < 1e-12
