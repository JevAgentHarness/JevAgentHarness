"""Exploratory row E19 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E19: the rule-randomised model on
DecisionGym-Hard"): the experiment entry, the n_step cost gate, the readout paired with another row's LOGGED reference
arm (E7f's llm_only and rl_tier) on synthetic records, the pause rule's overload test and the hard_ni row. No network,
no model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from jevharness.bench import registry as R
from jevharness.bench.metrics import ni_certified_p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explore_e19_hard_random as E19  # noqa: E402
import explore_e7_hard as E  # noqa: E402
import hard_ni  # noqa: E402

ENTRY = "E19: the rule-randomised model on DecisionGym-Hard"


def _dec(dp, s1, gold, deferred, s2=None, mode="cascade"):
    d = {"dp": dp, "mode": mode, "s1_value": s1, "s1_conf": 0.9, "deferred": deferred, "s2_value": s2,
         "final_value": s2 if deferred else s1, "gold": gold, "shadow": False, "gold_status": "ok", "gold_aux": {},
         "s2_correct": (s2 == gold) if deferred else None}
    if dp == "tool_select":
        d["gold_aux"] = {"acceptable_tools": [gold]}
    return d


def _rec(task_id, success, cost, wall_s, decisions=(), s1_cost=0.0, answer="1.00", error=None):
    return {"task_id": task_id, "config": "x", "backend": None, "llm": "flash", "final_answer": answer,
            "result": {"success": success},
            "usage": {"total": {"cost_usd": cost}, "s1": {"cost_usd": s1_cost, "calls": len(decisions)},
                      "s2": {"cost_usd": cost - s1_cost, "calls": 3}},
            "steps": [{"step": 0, "tool": "find_user", "args": {}, "llm_calls": 1, "decisions": list(decisions)}],
            "calls": [], "wall_ms": wall_s * 1000, "error": error, "meta": {}}


def _write(run: Path, arm: str, recs) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def _tid(seed: int) -> str:
    return f"jevgym-multi-{seed}-d60-near-strict"


def test_e19_entry_pairs_with_e7f_and_prices_its_host():
    X = E.EXPS["e19"]
    assert (X["eval"], X["smoke"], X["cap"], X["n_eval"], X["n_step"], X["prefix"]) == (
        "e19_hard", "e19_hard_smoke", 62.0, 200, 20, "e19")
    assert [p[1] for p in X["eval_parts"]] == ["e19_hard", "e19x_hard"] and X["cap_total"] == 62.0
    assert X["arms"] == ("llm_only", "rl_random") and X["gpu_arms"] == ("rl_random",) and X["diagnose"] == ("rl_random",)
    assert X["ref_runs"] == ("e7f_hard", "e7fx_hard")                     # the reference is E7f's logged arm
    assert X["extra_refs"] == {"vs_tier": ("rl_tier", ("e7f_hard", "e7fx_hard"))}
    assert X["entry"] == ENTRY and ENTRY in (ROOT / "docs/PREREG_A7_STAGE2.md").read_text()
    assert E.HOST_USD_PER_HOUR["g5.2xlarge"] == 1.212
    assert str(X["out"]).endswith("results/explore/e19_hard_random") and str(X["tex"]).endswith("e19_hard_random.tex")
    # E7f's block is the evaluation block (guarded); the smoke seeds are development seeds outside every reserved range
    assert R.overlaps(310600, 200, R.EXPLORE_EVAL_RANGES) and R.overlaps(310600, 200, R.EXPLORE_TRAIN_EXCLUDED)
    assert not R.overlaps(7284, 2, R.EXPLORE_EVAL_RANGES + R.STAGE2_RESERVED_RANGES + R.reserved_ranges())
    assert hard_ni.ROWS["e19"] == ("e19/rl_random/wins", "e19/rl_random/losses", "e19/rl_random/n")


def test_e19_gate_runs_all_200_else_the_largest_multiple_of_20(tmp_path):
    runs = tmp_path / "runs"
    _write(runs / "e19_hard_smoke", "rl_random", [_rec(_tid(7284), True, 0.25, 300.0), _rec(_tid(7285), True, 0.25, 320.0)])
    p = E.projection(runs, exp="e19")
    assert p["basis"] == "e19_hard_smoke" and p["candidates"][:3] == [200, 180, 160] and p["candidates"][-1] == 20
    assert abs(p["projected_200"] - 0.25 * 200 * 1.05) < 1e-9 and p["decision"] == "run all 200"
    assert E.projection(runs, prior=10.0, exp="e19")["decision"] == "run the first 180"    # 10 + 0.2625 n <= 62
    assert E.projection(runs, prior=10.0, exp="e19", cap=56.0)["decision"] == "run the first 160"   # the first part's cap
    assert E.projection(runs, prior=55.0, exp="e19")["decision"] == "run the first 20"
    assert E.projection(runs, prior=61.0, exp="e19")["decision"] == "stop after the smoke"
    _write(runs / "e19_hard_smoke", "rl_random", [_rec(_tid(7284), True, 0.25, 300.0),
                                                   _rec(_tid(7285), False, 0.0, 1.0, error="HTTP 503 high demand")])
    assert E.projection(runs, exp="e19")["decision"].startswith("stop: smoke errored")
    # E7's own gate is unchanged (200, else the first 100)
    assert E.EXPS["e7"].get("n_step") is None


def _e19_fixture(tmp_path: Path):
    runs = tmp_path / "runs"
    ref_tasks = [_tid(310600 + i) for i in range(4)] + [_tid(310680 + i) for i in range(4)]
    good = [_dec("tool_select", "find_user", "find_user", False), _dec("tool_params", "U1", "U1", False),
            _dec("tool_select", "ship_order", "ship_order", False), _dec("reflection", False, False, False, mode="trigger")]
    llm_y = [1, 1, 1, 0, 1, 1, 0, 1]
    tier_y = [1, 0, 1, 0, 1, 1, 1, 1]
    for name, idx in (("e7f_hard", range(0, 4)), ("e7fx_hard", range(4, 8))):
        _write(runs / name, "llm_only", [_rec(ref_tasks[i], bool(llm_y[i]), 0.26, 74.0) for i in idx])
        _write(runs / name, "rl_tier", [_rec(ref_tasks[i], bool(tier_y[i]), 0.23, 327.0, good) for i in idx])
    # the E19 arm ran 6 of the 8 tasks (the gate): three of each part
    own = [ref_tasks[i] for i in (0, 1, 2, 4, 5, 6)]
    own_y = [1, 1, 0, 1, 1, 1]
    _write(runs / "e19_hard", "rl_random", [_rec(t, bool(y), 0.24, 300.0, good) for t, y in zip(own, own_y)])
    (runs / "e19_hard" / "experiment.yaml").write_text(yaml.safe_dump({
        "suite": {"name": "jevgym", "n": 8, "seed0": 310600, "families": "multi", "multi_customers": 12,
                  "multi_rule": "tier", "answer_style": "itemized", "distractor_style": "near", "n_distractors": 60,
                  "noise_lines": 20, "strict": True},
        "harness_defaults": {"max_steps": 45}, "llm": {"provider": "gemini", "model": "gemini-3.5-flash"},
        "workers": 12}))
    window = tmp_path / "eval_window.json"
    window.write_text(json.dumps({"window_start": "2026-10-08T07:00:00Z", "window_end": "2026-10-08T08:00:00Z",
                                  "instance_type": "g5.2xlarge", "workers": 12}))
    return runs, window


def test_e19_readout_pairs_with_the_logged_e7f_arms_on_the_tasks_it_ran(tmp_path):
    runs, window = _e19_fixture(tmp_path)
    res = E.summarize(runs, n_boot=300, seed=0, window=window, exp="e19", light=True)
    m = res["meta"]
    assert m["exp"] == "e19" and m["cap"] == 62.0 and m["ref_runs"] == ["e7f_hard", "e7fx_hard"] and m["n_run"] == 6
    ref = res["arms"]["llm_only"]
    assert ref["n_ok"] == 6 and abs(ref["success_all"] - 5 / 6) < 1e-12      # the reference on the paired tasks only
    k = res["arms"]["rl_random"]["cmp"]
    assert k["n"] == 6 and k["n_dropped"] == 0 and (k["wins"], k["losses"]) == (1, 1) and abs(k["delta"]) < 1e-12
    q = 1.212 * 1.0 / 6
    assert abs(res["tariff"]["q_usd_per_episode"] - q) < 1e-12 and res["tariff"]["instance_type"] == "g5.2xlarge"
    assert abs(k["cost"] - (0.24 + q)) < 1e-12 and abs(k["api"]["saving"] - (1 - 0.24 / 0.26)) < 1e-9
    x = res["arms"]["rl_random"]["cmp_extra"]["vs_tier"]
    assert x["ref_arm"] == "rl_tier" and x["n"] == 6 and (x["wins"], x["losses"]) == (1, 1)
    assert abs(x["delta"]) < 1e-12 and abs(x["ref_wall_s_mean"] - 327.0) < 1e-9
    pts = res["arms"]["rl_random"]["points"]
    assert pts["op"]["n"] == 6 and pts["op"]["coverage"] == 1.0 and pts["op"]["accepted_wrong"] == 0
    keys = E.to_tex_keys(res)
    assert keys["e19/llm_only/success"] == "83.3" and keys["e19/rl_random/n"] == "6" and keys["e19/rl_random/delta"] == "+0.0"
    assert (keys["e19/setting/n"], keys["e19/setting/n_config"]) == ("6", "8")
    assert keys["e19/rl_random/vs_tier/delta"] == "+0.0" and keys["e19/rl_random/vs_tier/wins"] == "1"
    assert keys["e19/rl_random/vs_tier/wall_ref_mean"] == "327.0" and keys["e19/rl_random/wall_ref_mean"] == "74.0"
    assert keys["e19/rl_random/op_coverage"] == "100.0" and keys["e19/tariff/host_usd_h"] == "1.212"
    assert not any(kk.startswith(("e7f/", "e7/")) for kk in keys)
    md = E.to_markdown(res)
    assert "vs `rl_tier` (vs_tier" in md and "`rl_random`" in md
    # the registered reading from the keys: 6 of 200 tasks -> descriptive only
    ni = hard_ni.compute(keys)
    assert ni["e19"]["n"] == 6 and E19.reading(ni)["full"] is False and "descriptive" in E19.reading(ni)["verdict"]


def test_e19_configs_run_the_random_rules_checkpoint_on_e7f_tasks_at_the_fitted_thresholds():
    from jevharness.bench.runner import expand_arms

    load = lambda n: yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text())  # noqa: E731
    ev, sm, evx = load("eval_e19"), load("smoke_e19"), load("eval_e7fx")
    assert (ev["name"], ev["suite"]["seed0"], ev["suite"]["n"], ev["n_tasks"]) == ("e19_hard", 310600, 200, 200)
    # E7f's setting, route and schedule: the config differs from eval_e7fx.yaml only in name, seeds and n, backend and arm
    strip = lambda e: {k: v for k, v in e.items() if k not in ("name", "n_tasks", "suite", "s1_backends", "arms")}  # noqa: E731
    assert strip(ev) == strip(evx)
    assert {k: v for k, v in ev["suite"].items() if k not in ("seed0", "n")} == {
        k: v for k, v in evx["suite"].items() if k not in ("seed0", "n")}
    assert ev["llm"] == {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0, "reasoning_effort": "minimal"}
    assert ev["s1_backends"] == {"agentdecision_random": {
        "type": "jev", "name": "agentdecision-4b-random", "base_url": "http://127.0.0.1:8081",
        "model": "agentdecision-4b-random", "api_key": "", "timeout": 60.0, "cache": None}}
    # the one arm, at the thresholds the registered estimator fitted on this model's calibration answers (unchanged)
    fit = {dp: json.loads((ROOT / f"results/explore/e19_hard_random/box/cal/tauhat_random_{dp}.json").read_text())["tau_hat"]
           for dp in ("tool_select", "tool_params")}
    assert [a["name"] for a in ev["arms"]] == ["rl_random"] and fit["tool_select"] < 1.0 and fit["tool_params"] < 1.0
    arm = ev["arms"][0]
    assert arm["tool_select"] == {"mode": "cascade", "tau": fit["tool_select"]}
    assert arm["tool_params"] == {"mode": "cascade", "tau": fit["tool_params"]}
    assert arm["reflection"] == {"mode": "trigger", "tau": 0.5} and arm["s1"] == "agentdecision_random"
    # everything else as E7f's rl_tier arm
    a = {x.name: x for x in expand_arms(ev)}["rl_random"].model_dump()
    b = {x.name: x for x in expand_arms(evx)}["rl_tier"].model_dump()
    for d in (a, b):
        d.pop("name")
        for dp in ("tool_select", "tool_params"):
            d[dp].pop("tau")
            d[dp]["extra"].pop("calibration", None)
    assert a == b
    # smoke: the same arm and settings on two unused development seeds
    assert (sm["name"], sm["suite"]["seed0"], sm["suite"]["n"]) == ("e19_hard_smoke", 7284, 2)
    assert {k: v for k, v in sm["suite"].items() if k not in ("seed0", "n")} == {
        k: v for k, v in ev["suite"].items() if k not in ("seed0", "n")}
    assert sm["arms"] == ev["arms"] and sm["llm"] == ev["llm"] and sm["s1_backends"] == ev["s1_backends"]
    fits = json.loads((ROOT / "results/explore/e19_hard_random/box/cal/fits_random.json").read_text())
    assert fits["n_tasks"] == 20 and fits["derive"]["replaced"] == 1940 and not fits["tool_select"]["always_defer"]


def test_e19_extension_config_gate_and_pooled_readout(tmp_path):
    # the extension's config differs from eval_e19.yaml only in name, seed0 and n; together they cover E7f's block
    load = lambda n: yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text())  # noqa: E731
    ev, evx = load("eval_e19"), load("eval_e19x")
    assert (evx["name"], evx["suite"]["seed0"], evx["suite"]["n"], evx["n_tasks"]) == ("e19x_hard", 310780, 20, 20)
    strip = lambda e: {**e, "name": None, "n_tasks": None,  # noqa: E731
                       "suite": {k: v for k, v in e["suite"].items() if k not in ("seed0", "n")}}
    assert strip(evx) == strip(ev)
    assert set(range(310600, 310780)) | set(range(310780, 310800)) == set(range(310600, 310800))
    X = E.EXPS["e19x"]
    assert (X["eval"], X["cap"], X["n_eval"], X["n_min"], X["projection_basis"]) == ("e19x_hard", 62.0, 20, 20, "e19_hard")
    assert X["extra_runs"] == ("e19_hard_smoke", "e19_hard") and X["ref_runs"] == ("e7f_hard", "e7fx_hard")
    assert X["stop"] == E.EXPS["e19"]["stop"] == "e19_STOP.json"        # one $ cap over every E19 run
    # the gate: the first part's logged $ per episode x 20 x 1.05 + the spend so far (smoke + first part) <= $62
    runs = tmp_path / "runs"
    _write(runs / "e19_hard_smoke", "rl_random", [_rec(_tid(7284 + i), True, 0.285, 90.0) for i in range(2)])
    _write(runs / "e19_hard", "rl_random", [_rec(_tid(310600 + i), True, 0.30, 300.0) for i in range(180)])
    p = E.projection(runs, exp="e19x")
    assert p["basis"] == "e19_hard" and abs(p["spent_so_far"] - (0.57 + 54.0)) < 1e-9
    assert abs(p["projected_20"] - 0.30 * 20 * 1.05) < 1e-9 and p["decision"] == "run all 20"    # 54.57 + 6.3 <= 62
    _write(runs / "e19_hard", "rl_random", [_rec(_tid(310600 + i), True, 0.31, 300.0) for i in range(180)])
    assert E.projection(runs, exp="e19x")["decision"] == "stop after the smoke"                   # 56.37 + 6.51 > 62
    assert abs(E.total_spend(runs, exp="e19")["total"] - (0.57 + 55.8)) < 1e-9       # E19's accounting: every part
    assert "e19x_hard" in E19.LIVE_RUNS
    # the pooled readout: both parts paired with the logged reference, each part reported
    ref_tasks = [_tid(310600 + i) for i in range(4)] + [_tid(310780 + i) for i in range(2)]
    for name, idx in (("e7f_hard", range(0, 4)), ("e7fx_hard", range(4, 6))):
        _write(runs / name, "llm_only", [_rec(ref_tasks[i], True, 0.26, 74.0) for i in idx])
        _write(runs / name, "rl_tier", [_rec(ref_tasks[i], True, 0.23, 327.0) for i in idx])
    _write(runs / "e19_hard", "rl_random", [_rec(ref_tasks[i], i != 1, 0.28, 300.0) for i in range(4)])
    _write(runs / "e19x_hard", "rl_random", [_rec(ref_tasks[i], True, 0.28, 310.0) for i in range(4, 6)])
    for name, n, seed0 in (("e19_hard", 180, 310600), ("e19x_hard", 20, 310780)):
        (runs / name / "experiment.yaml").write_text(yaml.safe_dump({
            "suite": {"name": "jevgym", "n": n, "seed0": seed0, "families": "multi", "multi_customers": 12,
                      "multi_rule": "tier", "answer_style": "itemized", "distractor_style": "near", "n_distractors": 60,
                      "noise_lines": 20, "strict": True},
            "harness_defaults": {"max_steps": 45}, "llm": {"provider": "gemini", "model": "gemini-3.5-flash"},
            "workers": 12}))
    wins = []
    for name, box in (("e19_hard", "box"), ("e19x_hard", "box_e19x")):
        w = tmp_path / box / "eval_window.json"
        w.parent.mkdir(parents=True, exist_ok=True)
        w.write_text(json.dumps({"window_start": "2026-10-08T06:00:00Z", "window_end": "2026-10-08T06:30:00Z",
                                 "instance_type": "g5.2xlarge", "workers": 12}))
        wins.append(w)
    res = E.summarize(runs, n_boot=200, seed=0, window=wins, exp="e19", light=True)
    k = res["arms"]["rl_random"]["cmp"]
    assert k["n"] == 6 and (k["wins"], k["losses"]) == (0, 1) and res["meta"]["n_run"] == 6
    assert abs(res["tariff"]["q_usd_per_episode"] - 1.212 * 1.0 / 6) < 1e-12      # two windows, both parts' episodes
    assert abs(res["meta"]["spend"]["total"] - (0.57 + 0.28 * 6)) < 1e-9              # the smoke and both parts
    keys = E.to_tex_keys(res)
    assert keys["e19/setting/n"] == "6" and keys["e19/rl_random/n"] == "6"


def test_e19_pause_rule_and_the_ni_row():
    assert E19.overload_error({"error": "HTTP 503: the model is overloaded"})
    assert E19.overload_error({"error": "RESOURCE_EXHAUSTED (429)"}) and not E19.overload_error({"error": "KeyError: x"})
    assert not E19.overload_error({"error": None})
    keys = {"e19/rl_random/wins": "21", "e19/rl_random/losses": "25", "e19/rl_random/n": "200"}
    r = hard_ni.compute(keys)["e19"]
    assert r["n"] == 200 and abs(r["delta"] + 2.0) < 1e-9 and r["ni"] == "no"
    assert abs(r["p"] - ni_certified_p(21, 25, 200, 0.03)["p"]) < 1e-12
    rd = E19.reading({"e19": r})
    assert rd["full"] and "not shown" in rd["verdict"]
    yes = hard_ni.compute({"e19/rl_random/wins": "30", "e19/rl_random/losses": "10", "e19/rl_random/n": "200"})["e19"]
    assert yes["ni"] == "yes" and "non-inferior" in E19.reading({"e19": yes})["verdict"]
