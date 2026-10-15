"""The optional closed-family row of docs/PREREG_A7_STAGE2.md A7.12 (i) (gpt-6.1-sol / gpt-6-luna; Deviations
2026-10-05): eligibility from the served-model pin and the malformed share, the cost gate's n against the row's $80
envelope (frozen price + cache-write surcharge), the spend reader, the configs and launch gates, and the descriptive
report scripts/stage2_sol.py on a synthetic run (offline; no API)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sol_eligibility as SE  # noqa: E402


def _call(model: str, served: str | None = None, inp: int = 10000, cached: int = 8000, write: int = 1000,
          out: int = 200, cost: float = 0.0) -> dict:
    return {"component": "s2", "model": model, "input_tokens": inp, "output_tokens": out, "cost_usd": cost,
            "latency_ms": 1000.0,
            "meta": {"served_model": served or model, "service_tier": "default", "cached_input_tokens": cached,
                     "cache_write_tokens": write, "reasoning_tokens": 0}}


def _smoke_ep(i: int, model: str, served: str | None = None, malformed: bool = False, error: str | None = None) -> dict:
    steps = [{"step": 0, "tool": "find_user", "args": {"name": "a"}, "observation": "ok"},
             {"step": 1, "tool": "nope", "args": {}, "observation": "Error: unknown tool 'nope'" if malformed else "ok"}]
    return {"task_id": f"t{i}", "steps": steps, "calls": [_call(model, served)] * 2, "meta": {}, "error": error,
            "result": {"success": True}, "usage": {"total": {"cost_usd": 0.01}}}


def test_member_pin_malformed_and_cost_at_the_frozen_price(tmp_path: Path):
    run = tmp_path / "smoke"
    run.mkdir()
    for arm, model in SE.MEMBERS.items():
        (run / f"{arm}.jsonl").write_text("\n".join(json.dumps(_smoke_ep(i, model)) for i in range(50)) + "\n")
    m = SE.member(run / "llm_only_sol.jsonl", SE.SOL)
    assert m["eligible"] and m["malformed_rate"] == 0.0 and m["served_models"] == {SE.SOL: 100}
    # 2 calls x (2k uncached + 8k cached input, 200 output) at $2 / $0.10 / $10 per M, + 1k cache writes x $0.50 per M
    frozen = 2 * (2000 * 2.0 + 8000 * 0.10 + 200 * 10.0) / 1e6
    assert abs(m["cost_per_episode_frozen"] - frozen) < 1e-12
    assert abs(m["cost_per_episode"] - (frozen + 2 * 1000 * 0.5 / 1e6)) < 1e-12
    assert abs(m["smoke_spend"] - 50 * (0.01 + 2 * 1000 * 0.5 / 1e6)) < 1e-9          # every attempt, + surcharge
    (run / "bad.jsonl").write_text("\n".join(json.dumps(
        _smoke_ep(i, SE.LUNA, served="gpt-6-luna-2026-11-01" if i == 3 else None, malformed=i < 5)) for i in range(50)))
    bad = SE.member(run / "bad.jsonl", SE.LUNA)
    assert bad["malformed_rate"] == 0.05 and not bad["eligible"]                        # a snapshot other than the pin
    (run / "err.jsonl").write_text(json.dumps(_smoke_ep(0, SE.LUNA, error="RateLimit")) + "\n")
    assert not SE.member(run / "err.jsonl", SE.LUNA)["eligible"]                        # an error left


def test_cost_gate_n_by_the_envelope(tmp_path: Path):
    run = tmp_path / "smoke"
    run.mkdir()
    for arm, model in SE.MEMBERS.items():
        (run / f"{arm}.jsonl").write_text("\n".join(json.dumps(_smoke_ep(i, model)) for i in range(50)) + "\n")
    out = tmp_path / "gate.json"
    assert SE.main(["gate", "--run", str(run), "--manifest", str(tmp_path / "none.json"), "--write",
                    "--out", str(out)]) == 0
    g = json.loads(out.read_text())
    assert g["eligible"] and g["n"] == 500 and g["envelope_usd"] == 80.0 and g["predicted_usd"]["500"] < 80
    rows = g["members"]
    # predicted = 1.2 x (smoke + 150 (c_S + Jev_cal) + 2 n c_S + n c_L + n Jev)
    cs, cl = rows["llm_only_sol"]["cost_per_episode"], rows["llm_only_solsmall"]["cost_per_episode"]
    smoke = sum(r["smoke_spend"] for r in rows.values())
    want = 1.2 * (smoke + 150 * (cs + SE.JEV_CAL_PER_EPISODE) + 1000 * cs + 500 * cl + 500 * SE.JEV_PER_EPISODE)
    assert abs(g["predicted_usd"]["500"] - want) < 1e-9
    mid = {k: {**v, "cost_per_episode": 0.06} for k, v in rows.items()}
    assert SE.gate(mid)["n"] == 300                                                       # 500 misses $80, 300 fits
    pricey = {k: {**v, "cost_per_episode": 0.2} for k, v in rows.items()}
    assert SE.gate(pricey)["n"] is None and "resource gate" in SE.gate(pricey)["decision"]
    inel = {**rows, "llm_only_solsmall": {**rows["llm_only_solsmall"], "eligible": False}}
    assert SE.gate(inel)["n"] is None and "ineligible" in SE.gate(inel)["decision"]


def test_spend_counts_shadow_calls_cache_writes_and_the_probe(tmp_path: Path):
    d = tmp_path / "run"
    d.mkdir()
    rec = {"task_id": "t0", "usage": {"total": {"cost_usd": 1.0}, "total_with_shadow": {"cost_usd": 1.5}},
           "calls": [_call(SE.SOL, write=2_000_000)]}
    (d / "a.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps({**rec, "task_id": "t1"}) + "\n")
    (d / "schedule_order.jsonl").write_text(json.dumps({"task_id": "x"}) + "\n")      # not an episode file
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"probe_cost_usd": 0.25}))
    s = SE.spend([d], man)
    assert abs(s["total_usd"] - (2 * (1.5 + 2.0 * 0.5) + 0.25)) < 1e-9


def test_sol_configs_and_launch_gates(monkeypatch, tmp_path: Path):
    import stage2_spec as SS
    from jevharness.bench.runner import arm_profiles, expand_arms

    sol = yaml.safe_load((ROOT / "configs/experiments/stage2_sol.yaml").read_text())
    x = yaml.safe_load((ROOT / "configs/experiments/stage2_x.yaml").read_text())
    cal = yaml.safe_load((ROOT / "configs/experiments/jevgym_shadow_cal_sol.yaml").read_text())
    calx = yaml.safe_load((ROOT / "configs/experiments/jevgym_shadow_cal_x.yaml").read_text())
    smoke = yaml.safe_load((ROOT / "configs/dev/sol_eligibility.yaml").read_text())
    assert [a.name for a in expand_arms(sol)] == ["llm_only_sol", "jevagent_best_sol", "llm_only_solsmall"]
    assert sol["suite"] == x["suite"] and sol["schedule"] == x["schedule"] and sol["n_tasks"] in (300, 500)
    assert cal["suite"] == calx["suite"] and cal["arms"][0] == {**calx["arms"][0], "name": "jev_shadow_cal_sol"}
    assert smoke["suite"]["seed0"] == 0 and smoke["suite"]["n"] == 50                   # development seeds only
    for c in (sol, cal, smoke):
        assert c["llm"] == {"provider": "openai_responses", "model": "gpt-6.1-sol", "reasoning_effort": "low"}
    assert sol["llms"]["solsmall"] == smoke["llms"]["solsmall"] == {
        "provider": "openai_responses", "model": "gpt-6-luna", "reasoning_effort": "none", "temperature": 0}
    p = arm_profiles(sol)["jevagent_best_sol"]
    assert p["s2"] == "gpt-6.1-sol" and (p["status"] == "selected" or "pending" in str(p["skip"]))
    for run in ("jevgym_shadow_cal_sol", "stage2_sol"):
        assert run in SS.CONFIGS and run in SS.GATES
    g = tmp_path / "sol_cost_gate.json"
    monkeypatch.setattr(SS, "SOL_COST_GATE", g)
    assert SS._sol_cost_gate()[0] is False
    g.write_text(json.dumps({"eligible": True, "n": sol["n_tasks"], "decision": "n"}))
    assert SS._sol_cost_gate()[0] is True
    g.write_text(json.dumps({"eligible": False, "n": sol["n_tasks"]}))
    assert SS._sol_cost_gate()[0] is False
    g.write_text(json.dumps({"eligible": True, "n": 300 if sol["n_tasks"] == 500 else 500}))
    assert SS._sol_cost_gate()[0] is False
    assert SS._sol_prices()[0], SS._sol_prices()[1]                                      # the frozen $2 / $10 / $0.10


# ------------------------------------------------------------------------------------- the descriptive report
def _traj(tid: str, model: str, success: bool, cost: float, s2_calls: int, fire: bool = False,
          error: str | None = None) -> dict:
    steps = [{"step": 0, "tool": "find_user", "args": {"name": "a"}, "observation": "ok",
              "decisions": [{"dp": "tool_select", "mode": "cascade", "s1_value": "find_user", "deferred": True,
                             "final_value": "find_user", "shadow": False}]
              + ([{"dp": "reflection", "mode": "trigger", "final_value": fire, "shadow": False}]
                 if model == "gpt-6.1-sol" and cost > 0.011 else [])}]
    return {"task_id": tid, "config": "x", "backend": "jev", "llm": model, "steps": steps, "error": error,
            "result": {"success": success}, "wall_ms": 20000.0,
            "usage": {"total": {"cost_usd": cost, "latency_ms": 10000.0 + 1000 * s2_calls},
                      "s1": {"calls": 1, "cost_usd": 0.0002}, "s2": {"calls": s2_calls, "input_tokens": 9000},
                      "unpriced_models": []},
            "calls": [_call(model, inp=3000, cached=2000, write=500, out=100)] * s2_calls,
            "meta": {"s1": {"describe": {"backend": "jev", "provider": "openrouter"}}}}


def _write_run(tmp_path: Path, n: int = 8) -> tuple[Path, list[str]]:
    import stage2_sol as SS

    fz = json.loads((ROOT / "configs/eval/jevgym_stage2_tasks.json").read_text())
    ids = fz["task_ids"][:n]
    run = tmp_path / "stage2_sol"
    run.mkdir()
    exp = yaml.safe_load((ROOT / "configs/experiments/stage2_sol.yaml").read_text())
    (run / "experiment.yaml").write_text(yaml.safe_dump({**exp, "n_tasks": n}))
    ref = [_traj(t, SS.SOL, i % 4 != 0, 0.010, 5) for i, t in enumerate(ids)]           # 6 / 8 successes
    prof = [_traj(t, SS.SOL, i % 4 != 1, 0.012, 6, fire=(i == 2)) for i, t in enumerate(ids)]
    small = [_traj(t, SS.LUNA, True, 0.001, 5) for t in ids]
    prof.insert(0, _traj(ids[0], SS.SOL, False, 0.003, 1, error="RateLimitError"))      # an attempt then retried
    for name, recs in (("llm_only_sol", ref), ("jevagent_best_sol", prof), ("llm_only_solsmall", small)):
        (run / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return run, ids


def test_stage2_sol_report_on_a_synthetic_run(tmp_path: Path):
    import stage2_sol as SS

    run, ids = _write_run(tmp_path)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"eligible": True, "n": 8, "decision": "n = 8", "envelope_usd": 80.0,
                                "predicted_usd": {"500": 30.0, "300": 20.0}, "members": {}}))
    fits = tmp_path / "cal"
    fits.mkdir()
    (fits / "tauhat_jev_shadow_cal_sol_tool_select.json").write_text(json.dumps({"primary": {"tau_hat": float("inf")}}))
    (fits / "tauhat_jev_shadow_cal_sol_tool_params.json").write_text(json.dumps({"primary": {"tau_hat": 0.98}}))
    prof = tmp_path / "prof.yaml"
    prof.write_text(yaml.safe_dump({"kind": "jevagent_profile", "version": 1, "name": "p", "status": "selected",
                                    "system2": {"gpt-6.1-sol": {"decision_points": {
                                        "tool_select": {"mode": "cascade", "tau": 0.99, "always_defer": True},
                                        "tool_params": {"mode": "cascade", "tau": 0.98},
                                        "reflection": {"mode": "trigger", "tau": 0.5}}}}}))
    out, tex = tmp_path / "out", tmp_path / "stage2_sol.tex"
    args = ["--run", str(run), "--gate", str(gate), "--manifest", str(tmp_path / "none.json"), "--fits-dir", str(fits),
            "--profile", str(prof), "--out", str(out), "--tex", str(tex), "--n-boot", "500", "--no-spend", "--strict"]
    assert SS.main(args) == 0
    res = json.loads((out / "summary.json").read_text())
    assert res["status"] == "complete" and "no test" in res["label"]
    assert res["procedure"]["fits"]["tool_select"]["tau_hat"] is None                   # no accepted level
    assert res["procedure"]["fits"]["tool_params"]["tau_hat"] == 0.98
    a = res["arms"]
    assert a["llm_only_sol"]["n"] == 8 and a["llm_only_sol"]["success"] == 0.75
    assert a["jevagent_best_sol"]["attempts"] == 9 and a["jevagent_best_sol"]["served_models"] == {"gpt-6.1-sol": 48}
    assert abs(a["llm_only_sol"]["cost_usd_incl_cache_writes"] - (0.010 + 5 * 500 * 0.5 / 1e6)) < 1e-12
    p = res["pairs"]["jevagent_best_sol"]
    assert p["n_pairs"] == 8 and p["wins"] == 2 and p["losses"] == 2 and p["delta"] == 0.0
    assert p["delta_ci"][0] <= 0.0 <= p["delta_ci"][1]
    assert abs(p["cost"]["cost_savings"] - (1 - 0.012 / 0.010)) < 1e-9                  # more expensive: negative
    s = res["pairs"]["llm_only_solsmall"]
    assert s["delta"] == 0.25 and abs(s["cost"]["cost_savings"] - 0.9) < 1e-9
    iv = res["intervention"]["jevagent_best_sol"]
    assert iv["trigger_fires"] == 1 and iv["trigger_tasks"] == 1 and iv["tool_coverage"] == 0.0
    assert abs(iv["llm_calls_avoided"] - (5 - 6)) < 1e-9
    t = tex.read_text()
    assert r"\defres{stage2_sol/jevagent_best_sol/delta}{+0.0}" in t
    assert r"\defres{stage2_sol/llm_only_solsmall/delta}{+25.0}" in t
    assert r"\defres{stage2_sol/fit/tool_select/always_defer}{yes}" in t and "SIM/" not in t
    md = (out / "summary.md").read_text()
    assert "no test" in md and "jevagent_best_sol" in md


def test_stage2_sol_report_when_not_run(tmp_path: Path):
    import stage2_sol as SS

    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"eligible": False, "n": None, "decision": "pair ineligible (A7.12 (b))"}))
    out, tex = tmp_path / "out", tmp_path / "t.tex"
    assert SS.main(["--run", str(tmp_path / "missing"), "--gate", str(gate), "--manifest", str(tmp_path / "x.json"),
                    "--fits-dir", str(tmp_path), "--profile", str(tmp_path / "p.yaml"), "--out", str(out),
                    "--tex", str(tex), "--no-spend", "--strict"]) == 0
    res = json.loads((out / "summary.json").read_text())
    assert res["status"].startswith("not run") and not res["arms"]
    assert r"\defres{stage2_sol/status}{not run: pair ineligible (A7.12 (b))}" in tex.read_text()
