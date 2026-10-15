"""Code items of docs/PREREG_A7_STAGE2.md A7.10-A7.14: the best_control candidate table stays in sync with the
registered configs (A7.14 (a)); the derived profile of the second System-2 family (A7.12 (e), (g)); the router
metadata in the ledger (A7.12 (b), (h)); the Stage-2 configs that the amendments add load and resolve."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_control_candidate_table_is_in_sync_with_the_registered_configs():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/control_candidates.py"), "--check"], cwd=ROOT,
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert r.returncode == 0, r.stderr
    t = yaml.safe_load((ROOT / "configs/harness/control_candidates.yaml").read_text())
    rows = {(x["run"], x["arm"]): x for x in t["rows"]}
    assert ("E2b", "small_only") in rows and rows[("E2b", "small_only")]["intervention"] == "llm_swap"
    assert ("E2c", "bm25_ts_assist_k3") in rows and ("E2a", "lite_ts_cascade_tau0.9") in rows   # sweeps included
    assert all(x["system2"] == "gemini-3.5-flash" for k, x in rows.items() if k[1] != "small_only")
    assert not any(str((x.get("s1") or {}).get("type")) in ("jev", "intern_decision") for x in t["rows"])
    assert not any(k[1].endswith("_s2lite") for k in rows)                       # Flash only
    assert rows[("E2c", "lite_ts_assist_k3")]["duplicate_of"] == "E2a:lite_ts_assist_k3"


def _fit(tau):
    return {"primary": {"tau_hat": tau}}


def test_render_profile_x_refits_cascades_and_detects_a_profile_that_delegates_nothing():
    import render_profile_x as RX

    parent = {"status": "selected", "system2": {"gemini-3.5-flash": {"decision_points": {
        "tool_select": {"mode": "cascade", "tau": 1.0}, "tool_params": {"mode": "cascade", "tau": 0.99},
        "stop": {"mode": "trigger", "tau": 0.9}}}}}
    p = RX.render(parent, {"tool_select": _fit(0.97), "tool_params": _fit(float("inf"))}, "deepseek/deepseek-v4-pro")
    sec = p["system2"]["deepseek/deepseek-v4-pro"]
    assert sec["decision_points"]["tool_select"] == {"mode": "cascade", "tau": 0.97}
    assert sec["decision_points"]["tool_params"]["always_defer"] is True
    assert sec["decision_points"]["stop"] == {"mode": "trigger", "tau": 0.9} and "skip" not in sec
    only_tool = {"status": "selected", "system2": {"gemini-3.5-flash": {"decision_points": {
        "tool_select": {"mode": "cascade", "tau": 1.0}, "tool_params": {"mode": "cascade", "tau": 0.99}}}}}
    p = RX.render(only_tool, {"tool_select": _fit(float("inf")), "tool_params": _fit(0.5)}, "x")
    assert "delegates nothing" in p["system2"]["x"]["skip"]                      # tool_params unreachable (A6.22 (a) 4)


def test_router_metadata_lands_in_the_call_meta():
    from jevharness.llm.base import _routed_extras

    raw = SimpleNamespace(model_extra={"provider": "StreamLake"},
                          usage=SimpleNamespace(model_extra={"cost": 1.6e-06}))
    assert _routed_extras(raw) == {"served_provider": "StreamLake", "provider_cost_usd": 1.6e-06}
    assert _routed_extras(None) == {} and _routed_extras(object()) == {}


def test_price_lookup_matches_whole_path_segments_only():
    """A7.12 (h): a suffix match must not lend one model's price to another id. A bare "flash" used to resolve to the
    longest key ending in it ("deepseek/deepseek-v4-flash"); provider prefixes still resolve."""
    from jevharness.accounting import Price, PriceTable

    pt = PriceTable({"deepseek/deepseek-v4-flash": Price(input_per_m=0.028, output_per_m=0.056),
                     "deepseek/deepseek-v4-pro": Price(input_per_m=0.2088, output_per_m=0.4176),
                     "gemini-3.5-flash": Price(input_per_m=1.5, output_per_m=9.0),
                     "gemini-3.5-flash-lite": Price(input_per_m=0.3, output_per_m=2.5),
                     "typesafe/jev-1.13": Price(input_per_m=0.042, output_per_m=0.0),
                     "qwen/qwen3-8b": Price(input_per_m=0.05, output_per_m=0.1)})
    for frag in ("flash", "v4-flash", "4-flash", "pro", "lite", "flash-lite", "3.5-flash", "13", ""):
        assert pt.get(frag) is None, frag
    assert pt.get("deepseek/deepseek-v4-flash") is pt.prices["deepseek/deepseek-v4-flash"]
    assert pt.get("deepseek-v4-flash") is pt.prices["deepseek/deepseek-v4-flash"]            # bare id of a prefixed row
    assert pt.get("openrouter/deepseek/deepseek-v4-pro") is pt.prices["deepseek/deepseek-v4-pro"]
    assert pt.get("gemini/gemini-3.5-flash") is pt.prices["gemini-3.5-flash"]
    assert pt.get("gemini/gemini-3.5-flash-lite") is pt.prices["gemini-3.5-flash-lite"]
    assert pt.get("jev-1.13") is pt.prices["typesafe/jev-1.13"]
    assert pt.get("openrouter/qwen/qwen3-8b") is pt.prices["qwen/qwen3-8b"]
    assert pt.get("deepseek/deepseek-v4-pro-0813") is None and pt.get("typesafe/jev-1.13-20260917") is None
    assert pt.cost("flash", 1_000_000, 0) == 0.0


def test_new_stage2_configs_resolve():
    from jevharness.bench.runner import arm_profiles, expand_arms

    for cfg, n in (("stage2_rl_states.yaml", 3), ("stage2_rl_attribution.yaml", 3), ("stage2_x.yaml", 3),
                   ("jevgym_shadow_cal_x.yaml", 1)):
        exp = yaml.safe_load((ROOT / "configs/experiments" / cfg).read_text())
        assert len(expand_arms(exp)) == n, cfg
    x = yaml.safe_load((ROOT / "configs/experiments/stage2_x.yaml").read_text())
    px = arm_profiles(x)["jevagent_best_x"]
    if px["status"] == "pending":                       # before scripts/render_profile_x.py (A7.12 (e))
        assert "pending" in px["skip"]
    else:                                               # rendered: the X section only, from the committed parent
        assert px["status"] == "selected" and px["s2"] == "deepseek/deepseek-v4-pro"
    assert x["llm"]["extra_body"]["provider"] == {"order": ["StreamLake"], "allow_fallbacks": False}
    cal = yaml.safe_load((ROOT / "configs/experiments/jevgym_shadow_cal_x.yaml").read_text())
    assert cal["suite"]["seed0"] == 5150 and cal["suite"]["n"] == 150                 # A7.12 (d): 5150-5299
    st = yaml.safe_load((ROOT / "configs/experiments/stage2_rl_states.yaml").read_text())
    s2 = yaml.safe_load((ROOT / "configs/experiments/stage2_jevgym.yaml").read_text())
    assert st["suite"] == s2["suite"] and st["n_tasks"] == 200                       # A7.10 (c): 300000-300199
    assert not (ROOT / "configs/experiments/jevgym_tool_select_rl.yaml").exists()      # A7.13 (v)


def test_task_major_scheduler_runs_task_by_task_and_records_the_schedule(tmp_path: Path):
    """A7.16 (a): every arm's episode of task i is submitted before any of task i + 1, arms in a seeded order per task;
    the schedule is recorded and survives a resume; a changed seed is refused."""
    import json

    import pytest

    from jevharness.bench.runner import run_experiment, task_major_order

    exp = {"name": "tm", "suite": {"name": "jevgym", "n": 4, "seed0": 0, "families": "v2", "distractor_style": "near",
                                   "n_distractors": 10, "noise_lines": 0, "strict": True},
           "llm": {"provider": "simulated", "model": "sim", "error_rate": 0.0, "simulated_latency_ms": 1},
           "s1": {"type": "simulated", "accuracy": 0.8, "latency_ms": 1}, "workers": 1,
           "schedule": {"mode": "task_major", "seed": 732},
           "arms": [{"name": "llm_only"}, {"name": "ts_replace", "tool_select": {"mode": "replace"}},
                    {"name": "ts_cascade", "n_tasks": 2, "tool_select": {"mode": "cascade", "tau": 0.9}}]}
    run_experiment(exp, tmp_path, progress=False)
    out = tmp_path / "tm"
    order = [json.loads(x) for x in (out / "schedule_order.jsonl").read_text().splitlines()]
    tasks = [o["task"] for o in order]
    first = list(dict.fromkeys(tasks))
    assert len(order) == 4 + 4 + 2 and tasks == sorted(tasks, key=first.index)      # task-major
    per = {t: [o["arm"] for o in order if o["task"] == t] for t in first}
    assert [len(per[t]) for t in first] == [3, 3, 2, 2]                              # ts_cascade: first 2 tasks only
    assert order == [{"inv": 0, "seq": j, "task": t, "arm": a} for j, (t, a) in
                     enumerate(task_major_order(first, {t: per[t] for t in first}, 732))]
    for arm, n in (("llm_only", 4), ("ts_replace", 4), ("ts_cascade", 2)):
        assert len((out / f"{arm}.jsonl").read_text().splitlines()) == n
    rec = json.loads((out / "schedule.json").read_text())
    assert rec["mode"] == "task_major" and rec["seed"] == 732 and rec["invocations"][0]["n_jobs"] == 10
    run_experiment(exp, tmp_path, progress=False)                                     # resume: nothing left to run
    rec = json.loads((out / "schedule.json").read_text())
    assert [i["n_jobs"] for i in rec["invocations"]] == [10, 0]
    with pytest.raises(ValueError, match="schedule"):
        run_experiment({**exp, "schedule": {"mode": "task_major", "seed": 1}}, tmp_path, progress=False)
    s2 = yaml.safe_load((ROOT / "configs/experiments/stage2_jevgym.yaml").read_text())
    sx = yaml.safe_load((ROOT / "configs/experiments/stage2_x.yaml").read_text())
    assert s2["schedule"] == sx["schedule"] == {"mode": "task_major", "seed": 732}


def test_tariff_is_frozen_once_and_needs_a_zero_per_token_price(tmp_path: Path, monkeypatch):
    """A7.13 (iii): q = host $/h x serving hours / completed development episodes; the served model's per-token entry
    must be an exact 0 (nothing charged twice); the record is the one stage2_analysis reads."""
    import json

    import pytest
    import stage2_tariff as TF

    run = tmp_path / "w"
    run.mkdir()
    eps = [{"task_id": f"t{i}", "error": "RateLimit" if i == 0 else None, "steps": []} for i in range(41)]
    (run / "jevagent_best_rl.jsonl").write_text("\n".join(json.dumps(e) for e in eps) + "\n")
    args = ["--run", str(run), "--host-usd-per-hour", "1.32", "--window-start", "2026-10-05T10:00:00Z",
            "--window-end", "2026-10-05T11:00:00Z", "--sensitivity", "busy_throughput=0.01", "--workload-matches",
            "--out", str(tmp_path / "tariff.json")]
    prices = tmp_path / "prices.yaml"
    prices.write_text("models:\n  jevagent-rl: {input_per_m: 0.042, output_per_m: 0.0}\n")
    monkeypatch.setenv("JEV_PRICE_FILES", str(prices))
    with pytest.raises(SystemExit, match="exact"):
        TF.main(args)
    prices.write_text("models:\n  jevagent-rl: {input_per_m: 0.0, output_per_m: 0.0}\n")
    assert TF.main(args) == 0
    t = json.loads((tmp_path / "tariff.json").read_text())
    assert t["episodes"] == 40 and abs(t["q_usd_per_episode"] - 1.32 / 40) < 1e-12 and t["workload_matches"]
    assert t["sensitivity"] == {"busy_throughput": 0.01}
    with pytest.raises(SystemExit, match="frozen once"):
        TF.main(args)


def _x_ep(i: int, provider: str, malformed: bool = False, error: str | None = None) -> dict:
    steps = [{"step": 0, "tool": "find_user", "args": {"name": "a"}, "observation": "ok"},
             {"step": 1, "tool": "nope", "args": {}, "observation": "Error: unknown tool 'nope'" if malformed else "ok"}]
    calls = [{"component": "s2", "input_tokens": 20000, "output_tokens": 1000,
              "meta": {"served_provider": provider, "cached_input_tokens": 10000, "provider_cost_usd": 0.004}}] * 2
    return {"task_id": f"t{i}", "steps": steps, "calls": calls, "meta": {}, "error": error, "result": {"success": True}}


def test_x_eligibility_and_cost_gate(tmp_path: Path):
    """A7.12 (b), (c): malformed share, pin check from `served_provider`, infrastructure errors; the cost gate's n."""
    import json

    import x_eligibility as XE

    run = tmp_path / "x"
    run.mkdir()
    for arm in XE.MEMBERS:
        (run / f"{arm}.jsonl").write_text("\n".join(json.dumps(_x_ep(i, "StreamLake")) for i in range(50)) + "\n")
    out = tmp_path / "gate.json"
    assert XE.main(["--run", str(run), "--write", "--out", str(out)]) == 0
    g = json.loads(out.read_text())
    m = g["members"]["llm_only_x"]
    assert g["eligible"] and m["malformed_rate"] == 0.0 and m["served_providers"] == {"StreamLake": 100}
    # 2 calls x (10k uncached + 10k cached input, 1k output) at the verified V4-Pro list price
    assert abs(m["cost_per_episode_list"] - 2 * (10000 * 0.2088 + 10000 * 0.0174 + 1000 * 0.4176) / 1e6) < 1e-12
    assert g["n"] == 500 and g["predicted_usd"]["500"] < 25
    (run / "llm_only_xsmall.jsonl").write_text(
        "\n".join(json.dumps(_x_ep(i, "OtherProvider" if i == 3 else "StreamLake", malformed=i < 5)) for i in range(50)))
    bad = XE.member(run / "llm_only_xsmall.jsonl", XE.MEMBERS["llm_only_xsmall"], ["StreamLake"])
    assert bad["malformed_rate"] == 0.05 and not bad["eligible"]                     # unpinned provider seen
    rows = {"llm_only_x": m, "llm_only_xsmall": {**bad, "eligible": True}}
    pricey = {k: {**v, "cost_per_episode_list": 0.02} for k, v in rows.items()}
    assert XE.gate(pricey)["n"] is None and "resource gate" in XE.gate(pricey)["decision"]
