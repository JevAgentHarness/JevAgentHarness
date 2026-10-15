"""Exploratory row E7, JevGym-Hard (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06): the ``multi_rule`` knob of the
JevGym generator (default path unchanged), the evaluation-seed guard, the E7 configs, and the offline readout of
scripts/explore_e7_hard.py on synthetic records (no network, no model)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from jevharness.bench import registry as R
from jevharness.bench.suites import jevgym as J
from jevharness.bench.suites import jevgym_hard as H

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explore_e7_hard as E  # noqa: E402

STAGE2 = dict(families="v2", n_distractors=60, noise_lines=20, strict=True, distractor_style="near")
HARD = dict(families="multi", n_distractors=60, noise_lines=20, strict=True, distractor_style="near")


# ------------------------------------------------------------------------------------------- generator
def _fingerprint(k: int, cls=J.JevGymEnv, **kw) -> str:
    h = hashlib.sha256()
    for seed in range(200):
        e = cls(seed, multi_customers=k, **{**STAGE2, **kw})
        h.update(json.dumps([e.task_id, e.instruction, e._plan()], sort_keys=True).encode())
    return h.hexdigest()[:16]


def test_default_rule_leaves_every_task_unchanged():
    # fingerprints of the frozen generator (git 7097f3c); the "given" rule of the hard env reproduces them
    assert _fingerprint(3) == "6a56009cf7b13249"
    assert _fingerprint(7) == "b9dc293214049ffc"
    assert _fingerprint(3, cls=H.JevGymHardEnv, multi_rule="given") == "6a56009cf7b13249"
    assert _fingerprint(7, cls=H.JevGymHardEnv) == "b9dc293214049ffc"


def test_frozen_generator_is_untouched_and_the_suite_routes_by_option():
    from jevharness.bench.suites import load_suite

    fz = json.loads((ROOT / "configs/eval/jevgym_eval_tasks.json").read_text())
    have = hashlib.sha256((ROOT / "src/jevharness/bench/suites/jevgym.py").read_bytes()).hexdigest()
    assert fz["generator"]["sha256"] == have
    assert type(load_suite("jevgym", n=1, seed0=7000)[0]()) is J.JevGymEnv
    assert type(load_suite("jevgym", n=1, seed0=7000, families="multi", multi_rule="tier")[0]()) is H.JevGymHardEnv


def _play_gold(e: J.JevGymEnv) -> dict:
    e.reset()
    ans = None
    while True:
        a = e.oracle_action()
        if a is None:
            break
        tool, args = a
        if tool == "finish":
            ans = args["answer"]
            break
        e.step(tool, args)
    return e.evaluate(ans)


@pytest.mark.parametrize("rule", ["amount", "tier"])
def test_rules_derive_the_operation_and_keep_the_oracle_exact(rule):
    n_cancel = n_ship = 0
    for seed in range(7000, 7040):
        e = H.JevGymHardEnv(seed, multi_customers=10, multi_rule=rule, **HARD)
        g = J.JevGymEnv(seed, multi_customers=10, **HARD)
        # same world, customers and orders as the "given" task of the seed; only the stated operation differs
        assert [(u["user_id"], o["order_id"]) for _, u, o in e.jobs] == [(u["user_id"], o["order_id"]) for _, u, o in g.jobs]
        for op, u, o in e.jobs:
            if rule == "amount":
                want = o["amount"] > H.MULTI_AMOUNT_CANCEL_ABOVE
            else:
                lim = H.MULTI_TIER_CANCEL_ABOVE.get(u["tier"])
                want = lim is not None and o["amount"] > lim
            assert op == ("cancel" if want else "ship")
            n_cancel += op == "cancel"
            n_ship += op == "ship"
            assert u["name"] in e.instruction
        assert "pending order (the customer requested it)" not in e.instruction   # no per-customer operation stated
        assert len(e._plan()) == 3 * 10 + 1
        assert _play_gold(e)["success"]
    assert n_cancel and n_ship


def test_tier_rule_never_cancels_gold_and_a_wrong_operation_fails_strictly():
    e = H.JevGymHardEnv(7001, multi_customers=4, multi_rule="tier", **HARD)
    assert all(op == "ship" for op, u, _ in e.jobs if u["tier"] == "gold")
    e.reset()
    op, u, o = e.jobs[0]
    e.step("find_user", {"name": u["name"]})
    e.step("get_orders", {"user_id": u["user_id"], "status": "pending"})
    wrong = (("cancel_order", {"order_id": o["order_id"], "reason": "customer_request"}) if op == "ship"
             else ("ship_order", {"order_id": o["order_id"], "speed": "standard"}))
    r = e.step(*wrong)
    assert r.done and not e.evaluate(None)["success"]


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        H.JevGymHardEnv(1, family="multi", multi_rule="sometimes")
    with pytest.raises(ValueError):
        H.task_factories(n=1, seed0=7000, multi_rule="sometimes")
    with pytest.raises(ValueError):
        J.task_factories(n=1, seed0=7000, multi_rule="tier")       # the frozen generator has no such option
    assert len(H.task_factories(n=2, seed0=7000, families="multi", multi_rule="tier")) == 2


# ------------------------------------------------------------------------------------------- seeds and configs
def test_evaluation_seeds_are_refused_without_the_flag(monkeypatch):
    monkeypatch.delenv(R.EVAL_RUNS_ENV, raising=False)
    with pytest.raises(SystemExit):
        R.check_suite_seeds("jevgym", 310000, 200)
    with pytest.raises(SystemExit):
        R.check_suite_seeds("jevgym", 309990, 20)
    R.check_suite_seeds("jevgym", 7000, 300)        # the development range of the pilot
    with pytest.raises(SystemExit):
        R.check_suite_seeds("jevgym", 310200, 200)  # E7b's seeds
    with pytest.raises(SystemExit):
        R.check_suite_seeds("jevgym", 310400, 200)  # E7c's seeds
    with pytest.raises(SystemExit):
        R.check_suite_seeds("jevgym", 310600, 200)  # E7f's block
    with pytest.raises(SystemExit):
        R.check_suite_seeds("jevgym", 310790, 20)   # its last seeds
    R.check_suite_seeds("jevgym", 310800, 10)       # outside the E7 / E7b / E7c / E7f range
    R.check_suite_seeds("jevgym", 5300, 100)        # E7c's calibration seeds
    monkeypatch.setenv(R.EVAL_RUNS_ENV, "1")
    R.check_suite_seeds("jevgym", 310000, 200)
    R.check_suite_seeds("jevgym", 310200, 200)
    R.check_suite_seeds("jevgym", 310400, 200)
    for lo, hi in R.reserved_ranges() + R.STAGE2_RESERVED_RANGES + (R.CALIBRATION_RANGE,):
        assert not R.overlaps(310000, 400, ((lo, hi),))
        assert not R.overlaps(7000, 300, ((lo, hi),))


def test_configs_match_the_registration():
    ev = yaml.safe_load((ROOT / "configs/explore/e7_hard/eval.yaml").read_text())
    sm = yaml.safe_load((ROOT / "configs/explore/e7_hard/smoke.yaml").read_text())
    pi = yaml.safe_load((ROOT / "configs/explore/e7_hard/pilot.yaml").read_text())
    s = ev["suite"]
    assert (s["seed0"], s["n"], ev["n_tasks"]) == (310000, 200, 200)
    assert R.EXPLORE_EVAL_RANGES[0] == (310000, 310799) and R.overlaps(s["seed0"], s["n"], R.EXPLORE_EVAL_RANGES)
    assert [a["name"] for a in ev["arms"]] == list(E.ARMS)
    assert ev["schedule"] == {"mode": "task_major", "seed": 732}
    for k in ("families", "multi_customers", "multi_rule", "distractor_style", "n_distractors", "noise_lines", "strict"):
        assert sm["suite"][k] == s[k]
    assert sm["arms"] == ev["arms"] and sm["llm"] == ev["llm"] and sm["harness_defaults"] == ev["harness_defaults"]
    assert not R.overlaps(sm["suite"]["seed0"], sm["suite"]["n"], R.EXPLORE_EVAL_RANGES + R.STAGE2_RESERVED_RANGES)
    assert 7000 <= pi["suite"]["seed0"] and sm["suite"]["seed0"] + sm["suite"]["n"] - 1 <= 7299
    assert ev["llm"]["temperature"] == 0 and ev["llm"]["reasoning_effort"] == "minimal"
    assert ev["s1_backends"]["jevagent_rl"]["cache"] is None and "s1_cache" not in ev
    from jevharness.bench.suites import load_suite

    assert type(load_suite("jevgym", **{k: v for k, v in s.items() if k != "name"})[0]()) is H.JevGymHardEnv


def test_profile_arms_resolve_to_the_frozen_profile():
    from jevharness.bench.runner import expand_arms

    ev = yaml.safe_load((ROOT / "configs/explore/e7_hard/eval.yaml").read_text())
    arms = {a.name: a for a in expand_arms(ev)}
    assert arms["llm_only"].tool_select.mode == "off"
    jev, rl = arms["jevagent_best"], arms["jevagent_best_rl"]
    assert (jev.tool_select.mode, jev.tool_select.tau) == ("cascade", 1.0)
    assert (jev.tool_params.mode, jev.tool_params.tau) == ("cascade", 0.99)
    assert (jev.reflection.mode, jev.reflection.tau) == ("trigger", 0.5)
    assert rl.tool_select.mode == "cascade" and abs(rl.tool_select.tau - 0.5) < 1e-6
    assert rl.tool_params.mode == "cascade" and rl.tool_params.tau > 0.9999
    assert all(a.max_steps == 45 for a in arms.values())


# ------------------------------------------------------------------------------------------- readout
def _dec(dp, s1, gold, deferred, s2=None, mode="cascade"):
    d = {"dp": dp, "mode": mode, "s1_value": s1, "s1_conf": 0.9, "deferred": deferred, "s2_value": s2,
         "final_value": s2 if deferred else s1, "gold": gold, "shadow": False, "gold_status": "ok", "gold_aux": {},
         "s2_correct": (s2 == gold) if deferred else None}
    if dp == "tool_select":
        d["gold_aux"] = {"acceptable_tools": [gold]}
    return d


def _rec(task_id, success, cost, wall_s, decisions=(), s1_cost=0.0, answer="1.00", error=None, result=None):
    res = {"success": success, **(result or {})}
    return {"task_id": task_id, "config": "x", "backend": None, "llm": "flash", "final_answer": answer,
            "result": res, "usage": {"total": {"cost_usd": cost}, "s1": {"cost_usd": s1_cost, "calls": len(decisions)},
                                     "s2": {"cost_usd": cost - s1_cost, "calls": 3}},
            "steps": [{"step": 0, "tool": "find_user", "args": {}, "llm_calls": 1, "decisions": list(decisions)}],
            "calls": [], "wall_ms": wall_s * 1000, "error": error, "meta": {}}


def _write(run: Path, arm: str, recs) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "runs"
    run = runs / E.RUN_EVAL
    tids = [f"jevgym-multi-{310000 + i}-d60-near-strict" for i in range(10)]
    ref_y = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    jev_y = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
    rl_y = [1, 1, 1, 1, 1, 0, 1, 0, 0, 1]
    good = [_dec("tool_select", "find_user", "find_user", False), _dec("tool_params", "U1", "U1", False),
            _dec("tool_params", "U2", "U1", True, s2="U1"), _dec("reflection", False, False, False, mode="trigger")]
    _write(run, "llm_only", [_rec(t, bool(y), 0.10, 50.0) for t, y in zip(tids, ref_y)])
    _write(run, "jevagent_best", [_rec(t, bool(y), 0.08, 60.0, good, s1_cost=0.001) for t, y in zip(tids, jev_y)])
    rl = [_rec(t, bool(y), 0.07, 40.0, good) for t, y in zip(tids, rl_y)]
    rl.insert(3, _rec(tids[3], False, 0.0, 1.0, error="HTTP 503"))      # an errored attempt, re-run afterwards
    _write(run, "jevagent_best_rl", rl)
    (run / "experiment.yaml").write_text(yaml.safe_dump({
        "suite": {"name": "jevgym", "n": 10, "seed0": 310000, "families": "multi", "multi_customers": 10,
                  "multi_rule": "tier", "distractor_style": "near", "n_distractors": 60, "noise_lines": 20, "strict": True},
        "harness_defaults": {"max_steps": 45}, "llm": {"provider": "openrouter", "model": "google/gemini-3.5-flash"},
        "workers": 12}))
    window = tmp_path / "eval_window.json"
    window.write_text(json.dumps({"window_start": "2026-10-06T16:00:00Z", "window_end": "2026-10-06T17:00:00Z",
                                  "instance_type": "g5.8xlarge", "workers": 12}))
    return runs, window


def test_summary_pairs_tariff_coverage_and_accuracy(tmp_path):
    runs, window = _fixture(tmp_path)
    res = E.summarize(runs, n_boot=500, seed=0, window=window)
    jev, rl = res["arms"]["jevagent_best"]["cmp"], res["arms"]["jevagent_best_rl"]["cmp"]
    assert jev["n"] == 10 and (jev["wins"], jev["losses"]) == (1, 1) and abs(jev["delta"]) < 1e-12
    assert abs(rl["delta"] + 0.1) < 1e-12 and (rl["wins"], rl["losses"]) == (0, 1)
    assert abs(jev["saving"] - 0.2) < 1e-9 and jev["saving_ci"][0] - 1e-9 <= 0.2 <= jev["saving_ci"][1] + 1e-9
    tf = res["tariff"]
    assert tf["episodes"] == 10 and abs(tf["serving_hours"] - 1.0) < 1e-9
    q = 2.448 / 10
    assert abs(tf["q_usd_per_episode"] - q) < 1e-12
    assert abs(rl["cost"] - (0.07 + q)) < 1e-12 and abs(rl["saving"] - (1 - (0.07 + q) / 0.10)) < 1e-9
    assert abs(rl["api"]["saving"] - 0.3) < 1e-9 and abs(rl["sens"]["zero"]["saving"] - 0.3) < 1e-9
    pts = res["arms"]["jevagent_best"]["points"]
    assert abs(pts["tool"]["coverage"] - 2 / 3) < 1e-12           # two of three tool decisions accepted
    assert abs(pts["tool"]["acc"] - 2 / 3) < 1e-12                 # the deferred argument answer was wrong
    assert pts["tool"]["acc_accepted"] == 1.0 and pts["tool_params"]["llm_acc_deferred"] == 1.0
    assert pts["reflection"]["fired"] == 0.0 and pts["reflection"]["acc"] == 1.0
    assert res["arms"]["jevagent_best"]["paired_per_episode"]["wall_s_median"] == 60.0
    assert res["arms"]["llm_only"]["success_all"] == 0.8
    keys = E.to_tex_keys(res)
    for k in ("e7/llm_only/success", "e7/jevagent_best/delta", "e7/jevagent_best/delta_ci/lo", "e7/jevagent_best/saving",
              "e7/jevagent_best/coverage", "e7/jevagent_best/wins", "e7/jevagent_best_rl/losses", "e7/jevagent_best/acc",
              "e7/jevagent_best/wall_mean", "e7/jevagent_best/wall_median", "e7/jevagent_best_rl/cost_k",
              "e7/setting/multi_customers", "e7/setting/rule", "e7/tariff/q", "e7/setting/n"):
        assert k in keys, k
    assert keys["e7/jevagent_best_rl/delta"] == "-10.0" and keys["e7/llm_only/success"] == "80.0"
    md = E.to_markdown(res)
    assert "jevagent_best_rl" in md and "GPU tariff" in md
    tex = E.write_tex(keys, tmp_path / "e7.tex").read_text()
    assert r"\defres{e7/setting/rule}{tier}" in tex and r"\defres{e7/jevagent_best/state}{run}" in tex


def test_failure_types_and_projection(tmp_path):
    suite = {"families": "multi", "multi_customers": 4, "multi_rule": "tier", "distractor_style": "near",
             "n_distractors": 60, "noise_lines": 20, "strict": True}
    tid = "jevgym-multi-7001-d60-near-strict"
    env = H.JevGymHardEnv(7001, **suite)
    gold = f"{env.multi_total:.2f}"
    plan = [{"step": i, "tool": t, "args": a, "decisions": []} for i, (t, a) in enumerate(env._plan())]

    def rec(steps, **kw):
        r = _rec(tid, False, 0, 1, **kw)
        r["steps"] = steps
        return r
    assert E.failure_type(rec(plan, answer="12.00"), suite) == "wrong_total"
    assert E.failure_type(rec(plan[:-4] + plan[-1:], answer=f"total ${gold}"), suite) == "premature_finish"
    assert E.failure_type(rec(plan[:-4] + plan[-1:], answer="12.00"), suite) == "premature_finish"
    assert E.failure_type(rec(plan, answer=f"total ${gold}", result={"n_asks": 1}), suite) == "asked"
    assert E.failure_type(rec(plan, answer=None), suite) == "no_answer"
    assert E.failure_type(rec(plan, result={"irreversible_error": True}), suite) == "strict_stop"
    op, u, o = env.jobs[0]
    wrong = ({"tool": "cancel_order", "args": {"order_id": o["order_id"], "reason": "customer_request"}} if op == "ship"
             else {"tool": "ship_order", "args": {"order_id": o["order_id"], "speed": "standard"}})
    assert E.failure_type(rec(plan[:2] + [wrong] + plan[-1:], answer=gold), suite) == "strict_stop"
    assert {E.FAILURE_CLASS[k] for k in ("wrong_total", "strict_stop", "premature_finish")} == {
        "wrong_total", "wrong_write", "ask_finish"}
    runs = tmp_path / "runs"
    for arm, c in (("llm_only", 0.1), ("jevagent_best", 0.1), ("jevagent_best_rl", 0.1)):
        _write(runs / E.RUN_SMOKE, arm, [_rec(f"jevgym-multi-{7298 + i}-d60-near-strict", True, c, 10) for i in range(2)])
    p = E.projection(runs, prior=12.0)
    assert abs(p["projected_200"] - 0.3 * 200 * 1.5) < 1e-9 and p["decision"] == "run all 200"
    assert E.projection(runs, prior=80.0)["decision"] == "run the first 100"
    assert E.projection(runs, prior=140.0)["decision"] == "stop after the smoke"
    _write(runs / E.RUN_SMOKE, "jevagent_best_rl", [_rec("jevgym-multi-7298-d60-near-strict", False, 0, 1, error="x")])
    assert E.projection(runs)["decision"].startswith("stop: smoke errored")


def test_e7b_configs_change_only_the_customers_and_the_seeds():
    ev, evb = (yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text()) for n in ("eval", "eval_b"))
    sm, smb = (yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text()) for n in ("smoke", "smoke_b"))
    assert (evb["name"], evb["suite"]["seed0"], evb["suite"]["n"], evb["n_tasks"]) == ("e7b_hard", 310200, 200, 200)
    assert evb["suite"]["multi_customers"] == smb["suite"]["multi_customers"] == 12
    for a, b in ((ev, evb), (sm, smb)):
        strip = lambda e: {**e, "name": None, "suite": {k: v for k, v in e["suite"].items()  # noqa: E731
                                                        if k not in ("seed0", "multi_customers")}}
        assert strip(a) == strip(b)
    assert (smb["name"], smb["suite"]["seed0"], smb["suite"]["n"]) == ("e7b_hard_smoke", 7296, 2)
    used = set(range(7000, 7270)) | {7290, 7291, 7298, 7299}      # the pilot, the route check, E7's smoke
    assert not used & set(range(smb["suite"]["seed0"], smb["suite"]["seed0"] + smb["suite"]["n"]))
    assert not set(range(310000, 310200)) & set(range(310200, 310400))
    assert E.EXPS["e7b"]["eval"] == evb["name"] and E.EXPS["e7b"]["smoke"] == smb["name"]


def test_e7b_readout_uses_its_own_runs_cap_and_keys(tmp_path):
    runs, window = _fixture(tmp_path)
    (runs / E.RUN_EVAL).rename(runs / E.EXPS["e7b"]["eval"])
    res = E.summarize(runs, n_boot=200, seed=0, window=window, exp="e7b")
    assert res["meta"]["exp"] == "e7b" and res["meta"]["cap"] == 60.0
    keys = E.to_tex_keys(res)
    assert keys["e7b/jevagent_best_rl/delta"] == "-10.0" and not any(k.startswith("e7/") for k in keys)
    assert E.summarize(runs, n_boot=200, seed=0, window=window)["arms"]["llm_only"] == {"state": "not run"}
    for arm, c in (("llm_only", 0.1), ("jevagent_best", 0.1), ("jevagent_best_rl", 0.05)):
        _write(runs / E.EXPS["e7b"]["smoke"], arm, [_rec(f"jevgym-multi-{7296 + i}-d60-near-strict", True, c, 10)
                                                   for i in range(2)])
    p = E.projection(runs, exp="e7b")              # spend so far: the fixture's E7b run (about $2.5) + the smoke
    assert p["cap"] == 60.0 and abs(p["projected_200"] - 0.25 * 200 * 1.5) < 1e-9
    assert p["decision"] == "run the first 100"    # 200 tasks: $75 > $60; 100 tasks: $37.5 + spent fit


def test_e7c_configs_calibrate_and_evaluate_the_e7b_setting():
    load = lambda n: yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text())  # noqa: E731
    evb, evc, smc, cal = load("eval_b"), load("eval_c"), load("smoke_c"), load("cal_c")
    keys = ("families", "multi_customers", "multi_rule", "distractor_style", "n_distractors", "noise_lines", "strict")
    for c in (evc, smc, cal):
        assert {k: c["suite"][k] for k in keys} == {k: evb["suite"][k] for k in keys}
        assert c["llm"] == evb["llm"] and c["harness_defaults"] == evb["harness_defaults"] and c["s1"] == evb["s1"]
        assert c["s1_backends"] == evb["s1_backends"]
    assert (evc["suite"]["seed0"], evc["suite"]["n"], evc["n_tasks"]) == (310400, 200, 200)
    assert (cal["suite"]["seed0"], cal["suite"]["n"]) == (5300, 100)
    lo, hi = R.CALIBRATION_RANGE
    assert lo <= 5300 and 5399 <= hi and not R.overlaps(5300, 100, ((5000, 5299),))      # earlier calibrations
    assert (smc["suite"]["seed0"], smc["suite"]["n"]) == (7294, 2) and smc["arms"] == evc["arms"]
    used = set(range(7000, 7270)) | {7290, 7291, 7296, 7297, 7298, 7299}
    assert not used & {7294, 7295}
    assert [a["name"] for a in evc["arms"]] == list(E.EXPS["e7c"]["arms"])
    assert {a["name"]: a.get("s1") for a in cal["arms"]} == {"jev_shadow_cal": None, "rl_shadow_cal": "jevagent_rl"}
    for a in cal["arms"]:
        assert a["tool_select"] == {"mode": "shadow"} and a["tool_params"] == {"mode": "shadow"}
    assert set(E.CAL_ARMS) == {a["name"] for a in cal["arms"]} and set(E.CAL_ARMS.values()) == {"jev_refit", "rl_refit"}
    arms = {a["name"]: a for a in evc["arms"]}
    assert arms["rl_refit"]["s1"] == "jevagent_rl" and "s1" not in arms["jev_refit"]
    for name in ("jev_refit", "rl_refit"):
        a = arms[name]
        assert a["tool_select"]["mode"] == a["tool_params"]["mode"] == "cascade"
        assert a["reflection"] == {"mode": "trigger", "tau": 0.5}     # the profile's fixed trigger threshold
        assert set(a) <= {"name", "s1", "tool_select", "tool_params", "reflection"}


def test_e7c_profile_arms_match_the_frozen_profile_but_for_the_thresholds():
    from jevharness.bench.runner import expand_arms

    evc = yaml.safe_load((ROOT / "configs/explore/e7_hard/eval_c.yaml").read_text())
    ref = yaml.safe_load((ROOT / "configs/explore/e7_hard/eval_b.yaml").read_text())
    got = {a.name: a for a in expand_arms(evc)}
    want = {a.name: a for a in expand_arms(ref)}
    for mine, frozen in (("jev_refit", "jevagent_best"), ("rl_refit", "jevagent_best_rl")):
        a, b = got[mine].model_dump(), want[frozen].model_dump()
        for d in (a, b):
            d.pop("name")
            for dp in ("tool_select", "tool_params"):
                d[dp].pop("tau")
                d[dp]["extra"].pop("calibration", None)
                d[dp]["extra"].pop("always_defer", None)
        assert a == b, mine


def test_at_tau_and_e7c_spend_counts_the_calibration(tmp_path):
    ps = [{"c": 0.99, "s1_ok": True, "s2_ok": True}, {"c": 0.5, "s1_ok": False, "s2_ok": True},
          {"c": 0.95, "s1_ok": True, "s2_ok": False}, {"c": 0.2, "s1_ok": False, "s2_ok": False}]
    r = E._at_tau(ps, 0.9)
    assert r["coverage"] == 0.5 and r["s1_acc_taken"] == 1.0 and r["llm_acc_taken"] == 0.5 and r["s1_acc_all"] == 0.5
    assert E._at_tau(ps, float("inf"))["coverage"] == 0.0
    runs = tmp_path / "runs"
    _write(runs / "e7c_cal", "jev_shadow_cal", [_rec("jevgym-multi-5300-d60-near-strict", True, 1.0, 1)])
    _write(runs / "e7c_hard_smoke", "llm_only", [_rec("jevgym-multi-7294-d60-near-strict", True, 0.5, 1)])
    assert abs(E.total_spend(runs, exp="e7c")["total"] - 1.5) < 1e-12
    assert abs(E.total_spend(runs, exp="e7b")["total"]) < 1e-12
    assert E.EXPS["e7c"]["cap"] == 70.0 and E.EXPS["e7c"]["safety"] == 1.2 and E.EXPS["e7"]["safety"] == 1.5


def test_spend_counts_shadow_system1_calls():
    r = _rec("jevgym-multi-5300-d60-near-strict", True, 0.5, 1)
    r["usage"]["shadow"] = {"cost_usd": 0.02, "calls": 30}
    sp = E.episode_spend(r)
    assert abs(sp["total"] - 0.52) < 1e-12 and abs(sp["s1"] - 0.02) < 1e-12 and abs(sp["s2"] - 0.5) < 1e-12


def test_e7c_direct_row_keeps_e7c_tasks_and_rl_arm_on_the_direct_route():
    load = lambda n: yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text())  # noqa: E731
    evc, evd, smd = load("eval_c"), load("eval_cd"), load("smoke_cd")
    assert evd["suite"] == evc["suite"] and evd["n_tasks"] == evc["n_tasks"] == 200
    assert evd["llm"] == {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0,
                          "reasoning_effort": "minimal"}
    arms_c = {a["name"]: a for a in evc["arms"]}
    assert [a["name"] for a in evd["arms"]] == ["llm_only", "rl_refit"] == list(E.EXPS["e7cd"]["arms"])
    assert {a["name"]: a for a in evd["arms"]}["rl_refit"] == arms_c["rl_refit"]
    assert evd["s1_backends"] == evc["s1_backends"] and "s1" not in evd
    assert (smd["suite"]["seed0"], smd["suite"]["n"]) == (7294, 2) and smd["arms"] == evd["arms"]
    assert {k: v for k, v in smd["suite"].items() if k not in ("seed0", "n")} == {
        k: v for k, v in evd["suite"].items() if k not in ("seed0", "n")}
    assert smd["llm"] == evd["llm"] and smd["harness_defaults"] == evd["harness_defaults"]
    assert E.EXPS["e7cd"]["prefix"] == "e7c" and "jev_refit" in E.EXPS["e7cd"]["not_run"]


def test_latency_breakdown_groups_calls_and_attributes_the_tail():
    def ep(calls, wall):
        return {"task_id": "t", "wall_ms": wall * 1000, "calls": [
            {"component": comp, "decision_point": dp, "latency_ms": sec * 1000, "backend": "openrouter", "model": "m",
             "meta": {"served_provider": prov, "n_tools": nt}} for comp, dp, sec, prov, nt in calls]}
    ref = [ep([("s2", "tool_select", 1.0, "Google", 69)] * 10, 12.0)]
    arm = [ep([("s2", "tool_select", 1.0, "Google", 69)] * 5 + [("s2", "tool_params", 6.0, "Google", 1)]
              + [("s1", "tool_select", 0.2, None, None)] * 5, 16.0)]
    lat = {"llm_only": E.latency_breakdown(ref), "jevagent_best": E.latency_breakdown(arm)}
    g = lat["jevagent_best"]["groups"]
    assert g["s2/tool_params/1tool/Google"]["n_over"] == 1 and g["s2/tool_select/Google"]["n_over"] == 0
    assert abs(lat["jevagent_best"]["s1_per_episode"] - 1.0) < 1e-9
    c = E.latency_compare(lat)["jevagent_best"]
    assert abs(c["extra_wall_s"] - 4.0) < 1e-9 and abs(c["extra_s2_tail_s"] - 6.0) < 1e-9
    assert "tail" in E.latency_markdown(lat, {"jevagent_best": c})


def test_calibration_keys_cover_both_models_and_the_cancel_ship_subset():
    at = {"n": 10, "coverage": 0.5, "s1_acc_taken": 1.0, "llm_acc_taken": 1.0, "s1_acc_all": 0.9, "llm_acc_all": 1.0}
    row = lambda tau, ad: {"tau_hat": tau, "always_defer": ad, "at_tau": at, "op_at_tau": at}  # noqa: E731
    res = {"fits": {"jev_shadow_cal": {"n_tasks": 100, "errored": 0, "tool_select": row(0.91, False),
                                       "tool_params": row(0.87, False)},
                    "rl_shadow_cal": {"n_tasks": 100, "errored": 9, "tool_select": row(float("inf"), True),
                                      "tool_params": row(1.0, False)}}, "spend": {"total": 14.97}}
    k = E.cal_tex_keys(res)
    assert k["e7c/cal/rl4b/toolselect/tau"] == "always defer" and k["e7c/cal/jev/toolselect/tau"] == "0.91"
    for m in ("jev", "rl4b"):
        for dp in ("toolselect", "toolparams", "op"):
            for f in ("tau", "n", "coverage", "s1acctaken", "llmacctaken", "s1accall", "llmaccall"):
                assert f"e7c/cal/{m}/{dp}/{f}" in k
    assert k["e7c/cal/rl4b/ntasks"] == "100" and k["e7c/cal/spend"] == "14.97"


def test_e7c_jev_row_config_and_gate(tmp_path):
    load = lambda n: yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text())  # noqa: E731
    evc, evj, smj = load("eval_c"), load("eval_cj"), load("smoke_cj")
    assert (evj["suite"]["seed0"], evj["suite"]["n"], evj["n_tasks"]) == (310400, 100, 100)
    assert {k: v for k, v in evj["suite"].items() if k != "n"} == {k: v for k, v in evc["suite"].items() if k != "n"}
    assert evj["llm"]["provider"] == "gemini" and evj["s1"] == {"type": "jev", "provider": "openrouter"}
    arms_c = {a["name"]: a for a in evc["arms"]}
    assert [a["name"] for a in evj["arms"]] == ["llm_only", "jev_refit"] == list(E.EXPS["e7cj"]["arms"])
    assert {a["name"]: a for a in evj["arms"]}["jev_refit"] == arms_c["jev_refit"]
    assert (smj["suite"]["seed0"], smj["suite"]["n"]) == (7270, 2) and smj["arms"] == evj["arms"]
    used = set(range(7000, 7270)) | {7290, 7291, 7294, 7295, 7296, 7297, 7298, 7299}
    assert not used & {7270, 7271}
    runs = tmp_path / "runs"
    for arm, c in (("llm_only", 0.3), ("jev_refit", 0.1)):
        _write(runs / "e7cj_hard_smoke", arm, [_rec(f"jevgym-multi-{7270 + i}-d60-near-strict", True, c, 10)
                                              for i in range(2)])
    p = E.projection(runs, prior=16.05, exp="e7cj")
    assert p["cap"] == 90.0 and abs(p["projected_100"] - 0.4 * 100 * 1.2) < 1e-9 and p["decision"] == "run all 100"
    assert E.projection(runs, prior=50.0, exp="e7cj")["decision"] == "stop after the smoke"


# ------------------------------------------------------------------------------- final-answer form (E7e (e))
def test_answer_form_classes():
    assert E.answer_form("6263.88") == "number"
    assert E.answer_form(" $6,263.88. ") == "number"
    assert E.answer_form(None) == "none" and E.answer_form("  ") == "none"
    assert E.answer_form("I processed all 12 customers. The total of the processed orders is $5498.56.") == "sentence"
    lines = "\n".join(f"Customer {i}: shipped ORD-{100 + i} (${10 * i + 0.5:.2f})" for i in range(1, 13))
    assert E.answer_form(lines + "\nTotal: $391.00") == "itemized"


def test_answer_forms_shares_and_success():
    trajs = [{"final_answer": "12.00", "result": {"success": False}},
             {"final_answer": "a 1.00\nb 2.00\nc 3.00 total 6.00", "result": {"success": True}},
             {"final_answer": "a 1.00\nb 2.00\nc 3.00 total 6.00", "result": {"success": False}},
             {"final_answer": "The total is 6.00.", "result": {"success": False}}]
    f = E.answer_forms(trajs)
    assert (f["number"]["n"], f["itemized"]["n"], f["sentence"]["n"], f["none"]["n"]) == (1, 2, 1, 0)
    assert f["itemized"]["share"] == 0.5 and f["itemized"]["success"] == 0.5 and f["number"]["success"] == 0.0
    assert f["none"]["share"] == 0.0



# ------------------------------------------------------------------------------- E7f (AgentDecision-4B (tier) live)
def test_e7f_configs_run_the_tier_checkpoint_in_the_e7e_setting_on_fresh_seeds():
    from jevharness.bench.runner import expand_arms

    load = lambda n: yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text())  # noqa: E731
    eve, evf, smf, evb = load("eval_e7e"), load("eval_e7f"), load("smoke_e7f"), load("eval_b")
    assert (evf["name"], evf["suite"]["seed0"], evf["suite"]["n"], evf["n_tasks"]) == ("e7f_hard", 310600, 100, 100)
    assert {k: v for k, v in evf["suite"].items() if k != "seed0"} == {k: v for k, v in eve["suite"].items() if k != "seed0"}
    assert evf["suite"]["answer_style"] == "itemized" and evf["suite"]["multi_customers"] == 12
    assert evf["llm"] == eve["llm"] == {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0,
                                        "reasoning_effort": "minimal"}
    assert evf["harness_defaults"] == eve["harness_defaults"] == {"max_steps": 45} and "s1" not in evf
    assert evf["schedule"] == {"mode": "task_major", "seed": 732}
    # fresh seeds: inside the guarded block, outside every earlier JevGym-Hard row (310000-310599)
    seeds = set(range(310600, 310700))
    assert R.overlaps(310600, 100, R.EXPLORE_EVAL_RANGES) and not seeds & set(range(310000, 310600))
    assert R.in_ranges(310799, R.EXPLORE_EVAL_RANGES) and R.overlaps(310600, 200, R.EXPLORE_TRAIN_EXCLUDED)
    sb = evf["s1_backends"]["agentdecision_tier"]
    assert sb == {"type": "jev", "name": "agentdecision-4b-tier", "base_url": "http://127.0.0.1:8081",
                  "model": "agentdecision-4b-tier", "api_key": "", "timeout": 60.0, "cache": None}
    # thresholds: E7d's fit on the calibration states, unchanged
    fit = {dp: json.loads((ROOT / f"results/explore/e7_hard/box_e7d/cal/tauhat_tier_{dp}.json").read_text())["tau_hat"]
           for dp in ("tool_select", "tool_params")}
    arms = {a["name"]: a for a in evf["arms"]}
    assert list(arms) == ["llm_only", "rl_tier"] == list(E.EXPS["e7f"]["arms"])
    assert arms["rl_tier"]["tool_select"] == {"mode": "cascade", "tau": fit["tool_select"]}
    assert arms["rl_tier"]["tool_params"] == {"mode": "cascade", "tau": fit["tool_params"]}
    assert arms["rl_tier"]["reflection"] == {"mode": "trigger", "tau": 0.5} and arms["rl_tier"]["s1"] == "agentdecision_tier"
    # everything else as the frozen profile's 4B arm
    a = {x.name: x for x in expand_arms(evf)}["rl_tier"].model_dump()
    b = {x.name: x for x in expand_arms(evb)}["jevagent_best_rl"].model_dump()
    for d in (a, b):
        d.pop("name")
        for dp in ("tool_select", "tool_params"):
            d[dp].pop("tau")
            d[dp]["extra"].pop("calibration", None)
    assert a == b
    # smoke: the same arms and settings on two unused development seeds
    assert (smf["name"], smf["suite"]["seed0"], smf["suite"]["n"]) == ("e7f_hard_smoke", 7274, 2)
    assert {k: v for k, v in smf["suite"].items() if k not in ("seed0", "n")} == {
        k: v for k, v in evf["suite"].items() if k not in ("seed0", "n")}
    assert smf["arms"] == evf["arms"] and smf["llm"] == evf["llm"] and smf["s1_backends"] == evf["s1_backends"]
    used = set(range(7000, 7270)) | {7270, 7271, 7272, 7273, 7290, 7291, 7294, 7295, 7296, 7297, 7298, 7299}
    assert not used & {7274, 7275}
    assert not R.overlaps(7274, 2, R.EXPLORE_EVAL_RANGES + R.STAGE2_RESERVED_RANGES + R.reserved_ranges())


def test_e7f_gate_runs_100_else_80_else_stops(tmp_path):
    runs = tmp_path / "runs"
    for arm, c in (("llm_only", 0.265), ("rl_tier", 0.15)):
        _write(runs / "e7f_hard_smoke", arm, [_rec(f"jevgym-multi-{7274 + i}-d60-near-strict", True, c, 10)
                                             for i in range(2)])
    X = E.EXPS["e7f"]
    assert (X["cap"], X["safety"], X["n_eval"], X["n_min"]) == (45.0, 1.05, 100, 80)
    p = E.projection(runs, exp="e7f")       # spend so far: the smoke, 2 x (0.265 + 0.15) = $0.83
    assert abs(p["projected_100"] - 0.415 * 100 * 1.05) < 1e-9 and p["decision"] == "run all 100"
    assert E.projection(runs, prior=2.0, exp="e7f")["decision"] == "run the first 80"
    assert E.projection(runs, prior=20.0, exp="e7f")["decision"] == "stop after the smoke"
    assert X["out"] == ROOT / "results/explore/e7f" and X["tex"] == ROOT / "paper/generated/e7f_hard.tex"


def _hard_episode(seed: int, suite: dict, wrong_at: int | None, tau_ok: bool = True):
    """A logged-like episode of the gold plan on JevGym-Hard: tool-choice cascade decisions carry the System-1 state
    (task, history with the find_user record, the get_orders result as the current observation) and, at the write
    states, both writes among the options; ``wrong_at`` = the index of the write where the model names the other write
    (accepted), which ends the episode as a strict stop."""
    env = H.JevGymHardEnv(seed, **suite)
    env.reset()
    plan = env._plan()
    steps, hist, obs = [], [], ""
    n_w = 0
    for i, (tool, args) in enumerate(plan):
        state = f"TASK: {env.instruction}\n\nHISTORY:\n" + "\n".join(hist[-8:]) + f"\n\nCURRENT OBSERVATION:\n{obs}"
        gold = tool
        s1 = tool
        if tool in E.OP_TOOLS:
            if wrong_at == n_w:
                s1 = "ship_order" if tool == "cancel_order" else "cancel_order"
            n_w += 1
        d = {"dp": "tool_select", "mode": "cascade", "options": ["find_user", "get_orders", "cancel_order", "ship_order",
                                                                  "finish"],
             "s1_value": s1, "s1_conf": 0.9 if tau_ok else 0.4, "deferred": not tau_ok, "final_value": s1,
             "gold": gold, "state": state, "shadow": False}
        steps.append({"step": i, "tool": s1, "args": args, "decisions": [d]})
        if s1 != gold:
            break
        r = env.step(tool, dict(args))
        obs = r.observation
        hist.append(f"[{i}] {tool}({json.dumps(args)}) -> {obs[:400]}")
    failed = wrong_at is not None
    return {"task_id": f"jevgym-multi-{seed}-d60-near-strict", "config": "rl_tier", "final_answer": None if failed else
            f"{env.multi_total:.2f}", "result": {"success": not failed, "irreversible_error": failed},
            "usage": {"total": {"cost_usd": 0.1}}, "steps": steps,
            "calls": [{"component": "s1", "backend": "agentdecision-4b-tier", "model": "agentdecision-4b-tier"}],
            "wall_ms": 1000.0, "error": None,
            "meta": {"s1": {"describe": {"backend": "agentdecision-4b-tier", "base_url": "http://127.0.0.1:8081"}},
                     "config": {"s1_state_max_chars": 12000}}}


def test_diagnose_reads_thresholds_endpoint_prompts_and_failures(tmp_path):
    suite = {"name": "jevgym", "n": 3, "seed0": 7274, "families": "multi", "multi_customers": 3, "multi_rule": "tier",
             "distractor_style": "near", "n_distractors": 60, "noise_lines": 2, "strict": True}
    run = tmp_path / "run"
    run.mkdir()
    (run / "experiment.yaml").write_text(yaml.safe_dump({"suite": suite, "arms": [
        {"name": "rl_tier", "tool_select": {"mode": "cascade", "tau": 0.5}, "tool_params": {"mode": "cascade", "tau": 0.9}}]}))
    kw = {k: v for k, v in suite.items() if k not in ("name", "n", "seed0")}
    eps = [_hard_episode(7274, kw, None), _hard_episode(7275, kw, 1), _hard_episode(7276, kw, None, tau_ok=False)]
    eps[2]["steps"][0]["decisions"][0]["deferred"] = False          # one decision that disagrees with its threshold
    _write(run, "rl_tier", eps)
    d = E.diagnose(run, "rl_tier")
    assert d["thresholds"]["tool_select"]["tau"] == 0.5 and d["n_episodes"] == 3
    th = d["threshold_check"]["tool_select"]
    assert th["violations"] == 1 and th["consistent"] == th["n"] - 1 and th["examples"][0]["step"] == 0
    assert d["s1_calls"] == {"agentdecision-4b-tier / agentdecision-4b-tier": 3}
    ws = d["write_states"]
    n_writes = sum(1 for t in eps for st in t["steps"] if st["decisions"][0]["gold"] in E.OP_TOOLS)
    assert ws["n"] == n_writes > 0 and ws["both_writes_offered"] == ws["n"] == ws["policy_in_task"]
    assert ws["job_found"] == ws["tier_and_amount_seen"] == ws["n"] and ws["state_at_cap"] == 0
    assert sum(c["n"] for c in d["cells"].values()) == ws["n"]
    assert sum(c["model_wrong"] for c in d["cells"].values()) == 1
    assert sum(b["n"] for b in d["cancel_rate"].values()) == ws["n"]
    assert d["failures"] == {"strict_stop": 1}
    (key, n), = d["attribution"].items()
    assert n == 1 and key.startswith("strict_stop: wrong tool") and key.endswith("chosen by the model")
    md = E.diagnosis_markdown(d)
    assert "violations 1" in md and "Cancel rate by tier" in md and "chosen by the model" in md


def test_e7f_summary_writes_its_keys_and_the_diagnosis_view(tmp_path):
    suite = {"name": "jevgym", "n": 4, "seed0": 310600, "families": "multi", "multi_customers": 3, "multi_rule": "tier",
             "answer_style": "itemized", "distractor_style": "near", "n_distractors": 60, "noise_lines": 2,
             "strict": True}
    runs = tmp_path / "runs"
    run = runs / "e7f_hard"
    run.mkdir(parents=True)
    (run / "experiment.yaml").write_text(yaml.safe_dump({"suite": suite, "harness_defaults": {"max_steps": 45},
                                                         "llm": {"provider": "gemini", "model": "gemini-3.5-flash"},
                                                         "workers": 12, "arms": [{"name": "llm_only"}, {
                                                             "name": "rl_tier", "tool_select": {"mode": "cascade",
                                                                                                "tau": 0.5}}]}))
    kw = {k: v for k, v in suite.items() if k not in ("name", "n", "seed0")}
    rl = [_hard_episode(310600 + i, kw, 0 if i == 3 else None) for i in range(4)]
    ref = [_rec(t["task_id"], True, 0.25, 50.0) for t in rl]
    _write(run, "llm_only", ref)
    _write(run, "rl_tier", rl)
    window = tmp_path / "eval_window.json"
    window.write_text(json.dumps({"window_start": "2026-10-07T06:00:00Z", "window_end": "2026-10-07T06:30:00Z",
                                  "instance_type": "g5.8xlarge", "workers": 12}))
    res = E.summarize(runs, n_boot=200, seed=0, window=window, exp="e7f")
    assert res["meta"]["exp"] == "e7f" and res["meta"]["cap"] == 115.0 and "rl_frozen_refit" in res["meta"]["not_run"]
    k = res["arms"]["rl_tier"]["cmp"]
    assert k["n"] == 4 and (k["wins"], k["losses"]) == (0, 1)
    assert abs(res["tariff"]["q_usd_per_episode"] - 2.448 * 0.5 / 4) < 1e-12
    assert res["diagnosis"]["rl_tier"]["failures"] == {"strict_stop": 1}
    keys = E.to_tex_keys(res)
    assert keys["e7f/rl_tier/delta"] == "-25.0" and keys["e7f/setting/answer_style"] == "itemized"
    assert keys["e7f/rl_tier/diag/wrongwritebymodel"] == "1" and keys["e7f/rl_frozen_refit/state"] == "not run"
    assert keys["e7f/setting/n"] == "4" and "e7f/setting/n_config" not in keys
    # the gate ran the first tasks only: setting/n is the number run, the config's n beside it
    (run / "experiment.yaml").write_text((run / "experiment.yaml").read_text().replace("n: 4", "n: 5"))
    res5 = E.summarize(runs, n_boot=200, seed=0, window=window, exp="e7f")
    k5 = E.to_tex_keys(res5)
    assert (k5["e7f/setting/n"], k5["e7f/setting/n_config"]) == ("4", "5")
    assert "310600-310603 (the first 4 of the config's 5" in E.to_markdown(res5)
    assert not any(k.startswith(("e7/", "e7e/")) for k in keys)
    md = E.to_markdown(res)
    assert "Diagnosis view" in md and "answer style `itemized`" in md and "via gemini at the list prices" in md



# ------------------------------------------------------------------------------- E7f extension (the remaining 120 tasks)
def test_e7f_extension_config_changes_only_the_seeds_and_the_name():
    load = lambda n: yaml.safe_load((ROOT / f"configs/explore/e7_hard/{n}.yaml").read_text())  # noqa: E731
    evf, evx = load("eval_e7f"), load("eval_e7fx")
    assert (evx["name"], evx["suite"]["seed0"], evx["suite"]["n"], evx["n_tasks"]) == ("e7fx_hard", 310680, 120, 120)
    strip = lambda e: {**e, "name": None, "n_tasks": None,  # noqa: E731
                       "suite": {k: v for k, v in e["suite"].items() if k not in ("seed0", "n")}}
    assert strip(evx) == strip(evf)
    # the first part ran 310600-310679 (the gate's first 80); together they cover the reserved block exactly
    assert set(range(310600, 310680)) | set(range(310680, 310800)) == set(range(310600, 310800))
    assert R.overlaps(310680, 120, R.EXPLORE_EVAL_RANGES) and R.in_ranges(310799, R.EXPLORE_EVAL_RANGES)
    X = E.EXPS["e7fx"]
    assert (X["eval"], X["cap"], X["n_eval"], X["projection_basis"]) == ("e7fx_hard", 70.0, 120, "e7f_hard")
    assert [p[1] for p in E.EXPS["e7f"]["eval_parts"]] == ["e7f_hard", "e7fx_hard"]


def test_e7f_extension_gate_projects_from_the_first_part(tmp_path):
    runs = tmp_path / "runs"
    for arm, c in (("llm_only", 0.26), ("rl_tier", 0.25)):
        _write(runs / "e7f_hard", arm, [_rec(f"jevgym-multi-{310600 + i}-d60-near-strict", True, c, 10) for i in range(4)])
    p = E.projection(runs, exp="e7fx")
    assert p["basis"] == "e7f_hard" and p["spent_so_far"] == 0.0      # the first part's spend is not the extension's
    assert abs(p["projected_120"] - 0.51 * 120 * 1.05) < 1e-9 and p["decision"] == "run all 120"
    assert E.projection(runs, prior=10.0, exp="e7fx")["decision"] == "stop after the smoke"
    _write(runs / "e7fx_hard", "llm_only", [_rec("jevgym-multi-310680-d60-near-strict", True, 0.3, 10)])
    assert abs(E.total_spend(runs, exp="e7fx")["total"] - 0.3) < 1e-12
    assert abs(E.total_spend(runs, exp="e7f")["total"] - (4 * 0.51 + 0.3)) < 1e-9     # E7f's spend: both parts


def test_e7f_pooled_readout_reports_both_parts(tmp_path):
    suite = {"name": "jevgym", "n": 3, "seed0": 310600, "families": "multi", "multi_customers": 3, "multi_rule": "tier",
             "answer_style": "itemized", "distractor_style": "near", "n_distractors": 60, "noise_lines": 2,
             "strict": True}
    kw = {k: v for k, v in suite.items() if k not in ("name", "n", "seed0")}
    runs = tmp_path / "runs"
    for name, seed0, n, wrong in (("e7f_hard", 310600, 3, {2}), ("e7fx_hard", 310680, 4, {1, 3})):
        run = runs / name
        run.mkdir(parents=True)
        (run / "experiment.yaml").write_text(yaml.safe_dump({
            "suite": {**suite, "seed0": seed0, "n": n}, "harness_defaults": {"max_steps": 45},
            "llm": {"provider": "gemini", "model": "gemini-3.5-flash"}, "workers": 12,
            "arms": [{"name": "llm_only"}, {"name": "rl_tier", "tool_select": {"mode": "cascade", "tau": 0.5}}]}))
        rl = [_hard_episode(seed0 + i, kw, 0 if i in wrong else None) for i in range(n)]
        _write(run, "llm_only", [_rec(t["task_id"], True, 0.25, 50.0) for t in rl])
        _write(run, "rl_tier", rl)
    wins = []
    for i, box in enumerate(("box_a", "box_b")):
        w = tmp_path / box / "eval_window.json"
        w.parent.mkdir()
        w.write_text(json.dumps({"window_start": "2026-10-07T06:00:00Z", "window_end": f"2026-10-07T0{6 + i}:30:00Z",
                                 "instance_type": "g5.8xlarge", "workers": 12}))
        wins.append(w)
    res = E.summarize(runs, n_boot=200, seed=0, window=wins, exp="e7f")
    k = res["arms"]["rl_tier"]["cmp"]
    assert k["n"] == 7 and (k["wins"], k["losses"]) == (0, 3) and res["meta"]["n_run"] == 7
    assert abs(res["tariff"]["serving_hours"] - 2.0) < 1e-9 and res["tariff"]["episodes"] == 7
    assert abs(res["tariff"]["q_usd_per_episode"] - 2.448 * 2.0 / 7) < 1e-12
    assert res["diagnosis"]["rl_tier"]["failures"] == {"strict_stop": 3}
    assert set(res["parts"]) == {"first", "ext"}
    assert res["parts"]["first"]["arms"]["rl_tier"]["cmp"]["n"] == 3 and res["parts"]["ext"]["arms"]["rl_tier"]["cmp"]["n"] == 4
    keys = E.to_tex_keys(res)
    assert keys["e7f/setting/n"] == "7" and "e7f/setting/n_config" not in keys      # pooled: more tasks than the config
    assert keys["e7f/rl_tier/delta"] == E.to_tex_keys(res)["e7f/rl_tier/delta"] == f"{-300 / 7:+.1f}"
    assert keys["e7f/first/rl_tier/delta"] == f"{-100 / 3:+.1f}" and keys["e7f/ext/rl_tier/delta"] == "-50.0"
    assert (keys["e7f/first/n"], keys["e7f/ext/n"], keys["e7f/ext/seed0"]) == ("3", "4", "310680")
    for kk in ("e7f/ext/llm_only/success", "e7f/ext/rl_tier/success", "e7f/ext/rl_tier/saving",
               "e7f/ext/rl_tier/wall_mean", "e7f/ext/rl_tier/delta_ci/lo", "e7f/first/rl_tier/losses"):
        assert kk in keys, kk
    md = E.to_markdown(res)
    assert "## By part" in md and "310600-310606" in md and "pooled parts: first 3 tasks, ext 4 tasks" in md
