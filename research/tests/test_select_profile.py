"""scripts/select_profile.py (docs/PREREG_A7_STAGE2.md A7.1) on synthetic fixture runs: qualification (complete,
delegating, NI by the registered rule, cheaper by the paired t-test), the choice per group (largest saving, 2-pt tie ->
the more conservative mode), the strongest single mode, the degenerate sections, the written profile and its tau^2 /
ALFWorld renderings. The fixture runs carry hand-made episode records only (no model is called)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import select_profile as SP  # noqa: E402
from jevharness.harness import profile as HP  # noqa: E402

N = 200      # --max-tasks: every expected pair holds the first 200 frozen evaluation tasks


def _traj(task: str, arm: str, llm: str, ok: bool, cost: float, lat: float, steps: list[dict], s2_calls: int,
          backend: str | None = "jev") -> dict:
    return {"task_id": task, "config": arm, "llm": llm, "backend": backend, "result": {"success": ok},
            "usage": {"total": {"cost_usd": cost, "latency_ms": lat * 1000.0}, "s1": {"cost_usd": 0.0001},
                      "s2": {"calls": s2_calls, "input_tokens": 1000}},
            "steps": steps, "meta": {"s1": {"id": "x", "describe": {"backend": backend}}} if backend else {},
            "error": None}


def _steps(kind: str) -> list[dict]:
    """Two executed tool steps; the decisions a given arm logs there."""
    out = []
    for i in range(2):
        d: list[dict] = []
        if kind == "cascade":
            d = [{"dp": "tool_select", "mode": "cascade", "s1_value": "lookup", "deferred": i == 1,
                  "final_value": "lookup"}]
        elif kind == "always_defer":
            d = [{"dp": "tool_select", "mode": "cascade", "s1_value": None, "deferred": True, "final_value": "lookup"}]
        elif kind == "replace":
            d = [{"dp": "tool_select", "mode": "replace", "s1_value": "lookup", "deferred": False,
                  "final_value": "lookup"}]
        elif kind in ("assist", "guard"):
            d = [{"dp": "tool_select", "mode": kind, "s1_value": "lookup", "final_value": "lookup",
                  "options": ["lookup", "a", "b", "c", "finish"], "gold_aux": {"offered": ["lookup", "a", "finish"]}}]
        elif kind.startswith("meta:"):
            _, dp, mode = kind.split(":")
            d = [{"dp": dp, "mode": mode, "s1_value": False, "final_value": False}]
        out.append({"step": i, "tool": "lookup", "args": {"id": "1"}, "decisions": d})
    return out


def _write_arm(run: Path, arm: str, llm: str, tasks: list[str], *, losses: int = 0, cost: float = 0.04,
               cost_sd: float = 0.002, kind: str = "off", seed: int = 0, base_fail: int = 0) -> None:
    rng = np.random.default_rng(seed)
    with open(run / f"{arm}.jsonl", "w") as f:
        for i, t in enumerate(tasks):
            ok = not (i < losses or (base_fail and i % base_fail == 0))
            c = max(1e-5, cost + rng.normal(0, cost_sd))
            f.write(json.dumps(_traj(t, arm, llm, ok, c, 10.0 + rng.normal(0, 1), _steps(kind),
                                     0 if kind == "replace" else 2, None if kind == "off" else "jev")) + "\n")


def _fixture(root: Path) -> None:
    tasks = json.loads((ROOT / "configs/eval/jevgym_eval_tasks.json").read_text())["task_ids"][:N]
    F, L = SP.FLASH, SP.LITE
    for key, cfg in SP.CONFIGS.items():
        if key == "E6":
            continue                        # E6 absent: the action group stays with the LLM
        d = root / yaml.safe_load((ROOT / "configs/experiments" / cfg).read_text())["name"]
        d.mkdir(parents=True)
        (d / "experiment.yaml").write_text((ROOT / "configs/experiments" / cfg).read_text())
    e2a, e2b, e2c = (root / "e2a_jevgym_tool_decisions", root / "e2b_jevgym_meta_control", root / "e2c_jevgym_baselines")
    # E2a, Flash: the cascade at tau-hat* saves ~37 %; assist k5 ~12 % (not within 2 pt); assist k3 / replace lose
    _write_arm(e2a, "llm_only", F, tasks, cost=0.040, kind="off", seed=1)
    _write_arm(e2a, "jev_ts_cascade_tauhat", F, tasks, cost=0.025, kind="cascade", seed=2)
    _write_arm(e2a, "jev_ts_assist_k5", F, tasks, cost=0.035, kind="assist", seed=3)
    _write_arm(e2a, "jev_ts_assist_k3", F, tasks, losses=20, cost=0.020, kind="assist", seed=4)
    _write_arm(e2a, "jev_ts_guard", F, tasks, cost=0.041, kind="guard", seed=5)
    _write_arm(e2a, "jev_ts_replace", F, tasks, losses=30, cost=0.015, kind="replace", seed=6)
    _write_arm(e2a, "jev_ts_params_replace", F, tasks, losses=40, cost=0.008, kind="replace", seed=7)
    # E2a, Flash-Lite: the C1L arm is cheaper here but delegates nothing (always-defer); assist loses; guard costs more
    _write_arm(e2a, "llm_only_s2lite", L, tasks, cost=0.010, cost_sd=0.0005, kind="off", seed=8, base_fail=5)
    _write_arm(e2a, "jev_ts_cascade_tau0.9_s2lite", L, tasks, cost=0.008, cost_sd=0.0005, kind="always_defer",
               seed=9, base_fail=5)
    _write_arm(e2a, "jev_ts_assist_k3_s2lite", L, tasks, losses=25, cost=0.005, cost_sd=0.0005, kind="assist",
               seed=10, base_fail=5)
    _write_arm(e2a, "jev_ts_guard_s2lite", L, tasks, cost=0.011, cost_sd=0.0005, kind="guard", seed=11, base_fail=5)
    # E2b, Flash: the stop trigger saves ~2.5 % (qualifies); the stop guard costs more; routing loses; gates cost more
    _write_arm(e2b, "llm_only", F, tasks, cost=0.040, cost_sd=0.001, kind="off", seed=12)
    _write_arm(e2b, "jev_stop_trigger", F, tasks, cost=0.039, cost_sd=0.001, kind="meta:stop:trigger", seed=13)
    _write_arm(e2b, "jev_stop_guard", F, tasks, cost=0.0405, cost_sd=0.001, kind="meta:stop:guard", seed=14)
    _write_arm(e2b, "jev_reflect_trigger", F, tasks, cost=0.043, cost_sd=0.001, kind="meta:reflection:trigger", seed=15)
    _write_arm(e2b, "jev_route_tau0.4", F, tasks, losses=25, cost=0.013, kind="meta:routing:trigger", seed=16)
    _write_arm(e2b, "jev_mem", F, tasks, cost=0.045, cost_sd=0.001, kind="meta:memory_write:trigger", seed=17)
    _write_arm(e2b, "llm_only_s2lite", L, tasks, cost=0.010, cost_sd=0.0005, kind="off", seed=18, base_fail=5)
    _write_arm(e2b, "jev_stop_guard_s2lite", L, tasks, cost=0.013, cost_sd=0.0005, kind="meta:stop:guard", seed=19,
               base_fail=5)
    # E2c, Flash-Lite: the reflection trigger costs the same (not cheaper)
    _write_arm(e2c, "llm_only_s2lite", L, tasks, cost=0.010, cost_sd=0.0005, kind="off", seed=20, base_fail=5)
    # A7.14 controls (Flash): BM25 assist k = 3 is NI and 30 % cheaper (qualifies); TF-IDF assist saves 10 %; the
    # heuristic stop guard vetoes (its fixture decisions reject); Flash-Lite alone (E2b small_only) loses 30 tasks
    _write_arm(e2c, "llm_only", F, tasks, cost=0.040, kind="off", seed=22)
    _write_arm(e2c, "bm25_ts_assist_k3", F, tasks, cost=0.028, kind="assist", seed=23)
    _write_arm(e2c, "tfidf_ts_assist_k3", F, tasks, cost=0.036, kind="assist", seed=24)
    _write_arm(e2c, "heuristic_stop_guard", F, tasks, cost=0.039, cost_sd=0.001, kind="meta:stop:guard", seed=25)
    _write_arm(e2b, "small_only", L, tasks, losses=30, cost=0.010, cost_sd=0.0005, kind="off", seed=26)
    _write_arm(e2c, "jev_reflect_trigger_s2lite", L, tasks, cost=0.010, cost_sd=0.0005,
               kind="meta:reflection:trigger", seed=21, base_fail=5)


def test_choose_tie_rule_prefers_the_conservative_mode():
    q = [{"arm": "cascade", "savings": 0.40, "authority_rank": 4.0, "n_pairs": 500},
         {"arm": "assist_k5", "savings": 0.385, "authority_rank": 2.0, "n_pairs": 200},
         {"arm": "guard", "savings": 0.10, "authority_rank": 1.0, "n_pairs": 200}]
    sel, why = SP.choose(q)
    assert sel["arm"] == "assist_k5" and "within 2 pt" in why
    sel, _ = SP.choose([q[0], q[2]])
    assert sel["arm"] == "cascade"          # the conservative guard saves > 2 pt less: the cascade is selected
    assert SP.choose([])[0] is None


def test_select_profile_on_fixture_runs(tmp_path: Path):
    _fixture(tmp_path / "runs")
    out, prof_p, t2_p = tmp_path / "out", tmp_path / "best.yaml", tmp_path / "tau2" / "best.yaml"
    rc = SP.main(["--runs", str(tmp_path / "runs"), "--out", str(out), "--profile-out", str(prof_p),
                  "--tau2-out", str(t2_p), "--max-tasks", str(N), "--n-boot", "200",
                  "--control-out", str(tmp_path / "best_control.yaml")])
    assert rc == 0
    res = json.loads((out / "profile_selection.json").read_text())
    c = {(x["system2"], x["arm"]): x for x in res["candidates"]}
    F, L = SP.FLASH, SP.LITE
    casc = c[(F, "jev_ts_cascade_tauhat")]
    assert casc["qualifies"] and casc["noninferior"] and casc["cheaper"] and casc["delegating"]
    assert casc["n_pairs"] == N and 0.3 < casc["savings"] < 0.45 and casc["llm_calls_avoided"] == 0
    assert not c[(F, "jev_ts_assist_k3")]["noninferior"] and not c[(F, "jev_ts_guard")]["cheaper"]
    assert c[(F, "jev_stop_trigger")]["qualifies"] and not c[(F, "jev_stop_guard")]["qualifies"]
    assert not c[(F, "jev_route_tau0.4")]["qualifies"]
    lite_c1 = c[(L, "jev_ts_cascade_tau0.9_s2lite")]
    assert lite_c1["cheaper"] and not lite_c1["delegating"] and lite_c1["delegation_class"] == "zero"
    assert "not in the registered config" not in c[(F, "jev_action_replace")]["fate"]
    assert not c[(F, "jev_action_replace")]["qualifies"]          # E6 absent
    # the profile: Flash = tool cascade at tau-hat* + stop trigger; Flash-Lite = LLM-only (skipped)
    prof = HP.load_profile(prof_p)
    assert prof["status"] == "development" and "DEV" in prof["marks"]       # --max-tasks marks a dry run
    flash = prof["system2"][F]
    assert flash["decision_points"] == {"tool_select": {"mode": "cascade", "tau": 1.0},
                                        "tool_params": {"mode": "cascade", "tau": 0.99},
                                        "stop": {"mode": "trigger", "tau": 0.9}}
    assert flash["groups"]["tool"] == "jev_ts_cascade_tauhat" and flash["groups"]["routing"] is None
    assert flash["best_single"]["arm"] == "jev_ts_cascade_tauhat" and "skip" not in flash["best_single"]
    # A7.11 (a): best_single = the profile restricted to g* (tool); the A7.1 (g) 3 candidate is recorded, not run
    assert flash["best_single"]["group"] == "tool"
    assert flash["best_single"]["decision_points"] == {k: v for k, v in flash["decision_points"].items()
                                                       if k in ("tool_select", "tool_params")}
    assert flash["best_single"]["original_candidate"]["arm"] == "jev_ts_cascade_tauhat"
    # A7.11 (c): |G| = 2 -> the row without g* runs; the row without the other group is best_single itself (reused)
    assert flash["logo"]["tool"] is None and "configuration-identical" in flash["logo"]["stop"]
    assert flash["backends"]["jevagent_rl"] == flash["backends"]["startlux_4b"] == {
        "tool_select": {"calibration": "pending"}, "tool_params": {"calibration": "pending"}}   # A7.1 (f), A7.18 (e) 4
    assert "skip" in flash["alfworld"]                                      # no action-loop decision point
    assert "skip" in prof["system2"][L]
    # the arms of configs/experiments/stage2_jevgym.yaml resolve against it
    exp = yaml.safe_load((ROOT / "configs/experiments/stage2_jevgym.yaml").read_text())
    arms = {a["name"]: {**a, "profile": str(prof_p)} if "profile" in a and "default" not in a["profile"] else a
            for a in exp["arms"]}
    best, meta = HP.resolve_arm(arms["jevagent_best"], exp)
    assert meta["skip"] is None and best["stop"] == {"mode": "trigger", "tau": 0.9}
    single, meta = HP.resolve_arm(arms["best_single"], exp)
    assert meta["skip"] is None and "stop" not in single and single["tool_select"]["tau"] == 1.0
    # A7.11 (a) fixture check: the restriction changes the decision points and nothing else (model, prompt, history,
    # step budget, retries come from the same section and experiment)
    dp_names = set(HP.DP_NAMES)
    assert {k: v for k, v in best.items() if k not in dp_names | {"name"}} == \
        {k: v for k, v in single.items() if k not in dp_names | {"name"}}
    minus_tool, meta = HP.resolve_arm(arms["jevagent_minus_tool"], exp)
    assert meta["skip"] is None and minus_tool["stop"] == {"mode": "trigger", "tau": 0.9} and "tool_select" not in minus_tool
    assert "configuration-identical" in HP.skip_reason(arms["jevagent_minus_stop"], exp)
    assert "not active" in HP.skip_reason(arms["jevagent_minus_routing"], exp)
    rl, meta = HP.resolve_arm(arms["jevagent_best_rl"], exp)
    assert rl["tool_select"] == {"mode": "cascade", "tau": 1.0, "calibration": "pending"}   # skipped until fitted
    sl, meta = HP.resolve_arm(arms["jevagent_best_sl4b"], exp)                            # S2-4S (A7.18 (e) 5)
    assert sl["tool_select"]["calibration"] == "pending" and sl["stop"] == {"mode": "trigger", "tau": 0.9}
    assert HP.skip_reason(arms["jevagent_best_s2lite"], exp)
    # tau^2 rendering: the cascade carries over, the stop trigger is dropped (A7.1 (i))
    t2 = yaml.safe_load(t2_p.read_text())
    assert t2["tool_select"] == {"mode": "cascade", "tau": 1.0} and "stop" not in t2 and t2["guard_max_retries"] == 1
    assert "stop trigger -> off" in t2_p.read_text()
    # A7.14: best_control = the qualifying control with the largest saving; its fresh config is written
    ctl = json.loads((out / "control_selection.json").read_text())
    rows = {r["arm"]: r for r in ctl["rows"] if not r.get("duplicate_of")}
    assert ctl["best_control"] == "bm25_ts_assist_k3" and rows["bm25_ts_assist_k3"]["activation"]["changes"] > 0
    assert rows["tfidf_ts_assist_k3"]["qualifies"] and rows["tfidf_ts_assist_k3"]["savings"] < 0.2
    assert rows["heuristic_stop_guard"]["activation"]["statistic"] == "vetoes" and rows["heuristic_stop_guard"]["active"]
    assert not rows["small_only"]["noninferior"] and rows["small_only"]["activation"]["statistic"] == "void"
    bc = yaml.safe_load((tmp_path / "best_control.yaml").read_text())
    assert bc["suite"]["seed0"] == 300000 and bc["n_tasks"] == 500 and bc["s1_backends"]["control"]["type"] == "bm25"
    assert bc["arms"] == [{"name": "best_control", "s1": "control", "tool_select": {"mode": "assist", "k": 3}}]


def test_select_profile_strict_refuses_missing_runs(tmp_path: Path):
    rc = SP.main(["--runs", str(tmp_path / "none"), "--out", str(tmp_path / "o"), "--profile-out",
                  str(tmp_path / "b.yaml"), "--tau2-out", str(tmp_path / "t.yaml"), "--strict"])
    assert rc == 1 and not (tmp_path / "b.yaml").exists()


@pytest.mark.parametrize("arm", [a for _, _, a, _, _, _ in SP.CANDIDATES])
def test_every_candidate_is_registered_with_its_setting(arm: str):
    key = next(k for _, _, a, k, _, _ in SP.CANDIDATES if a == arm)
    setting = SP.registered_setting(yaml.safe_load((ROOT / "configs/experiments" / SP.CONFIGS[key]).read_text()), arm)
    assert setting and all(dp in HP.DP_NAMES for dp in setting)
