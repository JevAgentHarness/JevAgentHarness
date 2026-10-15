"""scripts/stage2_economics.py on tiny synthetic runs (offline): the frozen-price reproduction of every logged call, of
the episode costs and of the registered means / savings (and a loud failure when any of them differs), the re-pricing
of System-2 tokens only (System-1 cost and the GPU tariff unchanged; fresh, cached and output tokens at the scenario
row), the paired bootstrap (the same draw as scripts/primaries.cost_block), the timing block and the TeX keys."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import stage2_economics as E  # noqa: E402

PRICES = {"models": {"m-flash": {"input_per_m": 1.5, "output_per_m": 9.0, "cached_input_per_m": 0.15},
                     "gpt-6.1-sol": {"input_per_m": 2.0, "output_per_m": 10.0, "cached_input_per_m": 0.10},
                     "gpt-6-astra": {"input_per_m": 10.0, "output_per_m": 50.0, "cached_input_per_m": 1.0}}}
FLASH = E.Price(**PRICES["models"]["m-flash"])
TASKS = [f"jevgym-clarify-{300000 + i}-d60-near-strict" for i in range(6)]
Q = 0.002                                      # the tariffed arm's frozen per-episode GPU tariff


def _s2(inp: int, out: int, cached: int, step: int = 0) -> dict:
    return {"component": "s2", "backend": "gemini", "model": "m-flash", "decision_point": "tool_select",
            "input_tokens": inp, "output_tokens": out, "cost_usd": FLASH.cost(inp, out, cached), "latency_ms": 800.0,
            "step": step, "meta": {"cached_input_tokens": cached, "reasoning_tokens": 0}, "retry_wait_ms": 0.0}


def _s1(cost: float, step: int = 0) -> dict:
    return {"component": "s1", "backend": "jev", "model": "typesafe/jev-1.13", "decision_point": "tool_select",
            "input_tokens": 1000, "output_tokens": 0, "cost_usd": cost, "latency_ms": 200.0, "step": step,
            "meta": {}, "retry_wait_ms": 0.0}


def _rec(task: str, calls: list[dict], wall_s: float, error: str | None = None) -> dict:
    tot = {"calls": len(calls), "cost_usd": sum(c["cost_usd"] for c in calls),
           "latency_ms": sum(c["latency_ms"] for c in calls)}
    s1 = [c for c in calls if c["component"] == "s1"]
    return {"task_id": task, "result": {"success": error is None}, "error": error, "calls": calls,
            "wall_ms": wall_s * 1000.0,
            "usage": {"total": tot, "s1": {"cost_usd": sum(c["cost_usd"] for c in s1)}}}


def _arms() -> dict[str, list[dict]]:
    ref = [_rec(t, [_s2(4000 + 100 * i, 20, 1000), _s2(5000, 30, 2000, 1)], 6.0 + i) for i, t in enumerate(TASKS)]
    arm = [_rec(t, [_s1(1e-4), _s2(3000, 15, 500), _s1(1e-4, 1)], 7.0 + 0.5 * i) for i, t in enumerate(TASKS)]
    # the tariffed arm: System-1 priced at 0 per token, its GPU cost is the frozen tariff Q per episode
    gpu = [_rec(t, [_s1(0.0), _s2(1000, 10, 0)], 9.0) for t in TASKS]
    # a retried task: the errored first record is superseded by the later one (latest record per task decides)
    arm_retry = [_rec(TASKS[0], [_s1(5e-3)], 1.0, error="Timeout")] + arm
    return {"llm_only": ref, "prof": arm_retry, "prof_gpu": gpu}


def _setup(tmp: Path, registered_patch=None) -> tuple[E.Economics, dict]:
    runs = tmp / "runs"
    run = runs / "s2"
    run.mkdir(parents=True)
    recs = _arms()
    for name, rs in recs.items():
        (run / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    (run / "experiment.yaml").write_text(yaml.safe_dump({"n_tasks": len(TASKS), "arms": [{"name": n} for n in recs]}))
    (run / "schedule.json").write_text(json.dumps({"concurrency": 8, "arms": list(recs),
                                                   "invocations": [{"n_jobs": 18, "started": "a", "finished": "b"}]}))
    prices = tmp / "prices.yaml"
    prices.write_text(yaml.safe_dump(PRICES))
    tasks = tmp / "tasks.json"
    tasks.write_text(json.dumps({"task_ids": TASKS}))
    tariff = tmp / "tariff.json"
    tariff.write_text(json.dumps({"q_usd_per_episode": Q}))
    latest = {n: {r["task_id"]: r for r in rs} for n, rs in recs.items()}
    mean = {n: float(np.mean([latest[n][t]["usage"]["total"]["cost_usd"] for t in TASKS])) for n in latest}
    lat = {n: float(np.mean([latest[n][t]["usage"]["total"]["latency_ms"] / 1000 for t in TASKS])) for n in latest}
    ca = np.array([latest["prof"][t]["usage"]["total"]["cost_usd"] for t in TASKS])
    cr = np.array([latest["llm_only"][t]["usage"]["total"]["cost_usd"] for t in TASKS])
    idx = E.boot_indices(len(TASKS), 500, 0)
    sv = E.savings_block(ca, cr, idx)
    registered = {"family": {"T1": {"cost": {"cost_arm": mean["prof"], "cost_ref": mean["llm_only"],
                                             "cost_savings": sv["savings"], "cost_savings_ci": sv["savings_ci"]}},
                             "T2": {"cost": {"cost_arm": mean["prof_gpu"] + Q}}},
                  "descriptive": {"scorecard": {"llm_only": {"cost_usd": mean["llm_only"], "latency_s": lat["llm_only"]},
                                                "prof_gpu": {"cost_usd": mean["prof_gpu"]}}},
                  "provenance": {}}
    if registered_patch:
        registered_patch(registered)
    reg = tmp / "stage2.json"
    reg.write_text(json.dumps(registered))
    arms = (
        {"key": "llm_only", "run": "s2", "family": "flash", "ref": None, "tariff": None, "label": "ref",
         "status": "reference", "checks": [("cost", ("family", "T1", "cost", "cost_ref"), False),
                                           ("cost", ("descriptive", "scorecard", "llm_only", "cost_usd"), False),
                                           ("latency", ("descriptive", "scorecard", "llm_only", "latency_s"))]},
        {"key": "prof", "run": "s2", "family": "flash", "ref": "llm_only", "tariff": None, "label": "profile",
         "status": "T1", "checks": [("cost", ("family", "T1", "cost", "cost_arm"), False),
                                    ("savings", ("family", "T1", "cost"))]},
        {"key": "prof_gpu", "run": "s2", "family": "flash", "ref": "llm_only", "tariff": "gpu", "label": "gpu",
         "status": "T2", "checks": [("cost", ("family", "T2", "cost", "cost_arm"), True),
                                    ("cost", ("descriptive", "scorecard", "prof_gpu", "cost_usd"), False)]},
    )
    eco = E.Economics(runs, reg, prices, tasks, {"gpu": tariff}, n_boot=500, seed=0, arms=arms,
                      run_dirs={"s2": "s2"})
    return eco, latest


def test_reproduces_and_reprices_system2_only(tmp_path: Path) -> None:
    eco, latest = _setup(tmp_path)
    res = eco.run()
    assert all(c["ok"] for c in res["checks"]) and res["n_checks"] >= 6
    a = res["arms"]["prof"]
    # the errored first record of TASKS[0] is superseded: its $0.005 System-1 call never counts
    assert a["n"] == len(TASKS) and a["n_dropped"] == 0
    assert a["cost"]["frozen"]["s1"] == pytest.approx(2e-4)
    # Sol re-pricing by hand: 3000 input of which 500 cached, 15 output
    sol = (2500 * 2.0 + 500 * 0.10 + 15 * 10.0) / 1e6
    assert a["cost"]["sol"]["s2"] == pytest.approx(sol, abs=1e-15)
    assert a["cost"]["sol"]["s1"] == a["cost"]["frozen"]["s1"]                # System-1 as recorded
    astra = (2500 * 10.0 + 500 * 1.0 + 15 * 50.0) / 1e6
    assert a["cost"]["astra"]["mean"] == pytest.approx(2e-4 + astra, abs=1e-15)
    g = res["arms"]["prof_gpu"]
    assert g["tariff_q"] == Q
    for s in ("frozen", "sol", "astra"):                                       # the tariff is charged in every scenario
        assert g["cost"][s]["mean"] == pytest.approx(g["cost"][s]["s1"] + g["cost"][s]["s2"] + Q, abs=1e-15)
    # savings against the reference re-priced the same way
    ref_sol = np.mean([E.episode_cost(latest["llm_only"][t], eco.table, eco.table.get("gpt-6.1-sol"))["total"]
                       for t in TASKS])
    assert a["savings"]["sol"]["savings"] == pytest.approx(1 - a["cost"]["sol"]["mean"] / ref_sol, abs=1e-12)
    assert a["savings"]["sol"]["abs_savings"] == pytest.approx(ref_sol - a["cost"]["sol"]["mean"], abs=1e-15)


def test_registered_mismatch_fails_loudly(tmp_path: Path) -> None:
    def bump(reg: dict) -> None:
        reg["family"]["T1"]["cost"]["cost_arm"] += 1e-6
    eco, _ = _setup(tmp_path, bump)
    with pytest.raises(E.ReproductionError, match="registered"):
        eco.run()


def test_registered_savings_interval_mismatch_fails(tmp_path: Path) -> None:
    def bump(reg: dict) -> None:
        reg["family"]["T1"]["cost"]["cost_savings_ci"][0] -= 1e-4
    eco, _ = _setup(tmp_path, bump)
    with pytest.raises(E.ReproductionError, match="bootstrap"):
        eco.run()


def test_logged_call_cost_must_match_frozen_price(tmp_path: Path) -> None:
    eco, latest = _setup(tmp_path)
    rec = json.loads(json.dumps(latest["llm_only"][TASKS[1]]))
    rec["calls"][0]["cost_usd"] *= 1.01
    with pytest.raises(E.ReproductionError, match="re-priced at the frozen"):
        E.episode_cost(rec, eco.table, None, where="x")


def test_bootstrap_is_primaries_cost_block_draw() -> None:
    import primaries as P

    rng = np.random.default_rng(3)
    ca, cr = rng.uniform(0.01, 0.03, 40), rng.uniform(0.03, 0.05, 40)
    keys = [str(i) for i in range(40)]
    blk = P.cost_block({k: {"cost": c, "tokens": 1.0} for k, c in zip(keys, cr)},
                       {k: {"cost": c, "tokens": 1.0} for k, c in zip(keys, ca)}, keys, 1000, 0)
    sv = E.savings_block(ca, cr, E.boot_indices(40, 1000, 0))
    assert sv["savings"] == pytest.approx(blk["cost_savings"], abs=1e-15)
    assert sv["savings_ci"] == pytest.approx(blk["cost_savings_ci"], abs=1e-15)


def test_timing_block_point_estimates() -> None:
    ta, tr = np.array([1.0, 2.0, 9.0]), np.array([1.0, 1.0, 1.0])
    t = E.timing_block(ta, tr, E.boot_indices(3, 200, 0))
    assert t["mean_diff"] == pytest.approx(4.0 - 1.0)
    assert t["median_diff"] == pytest.approx(2.0 - 1.0)
    assert t["mean_ratio"] == pytest.approx(4.0)
    assert t["mean_diff_ci"][0] <= t["mean_diff_ci"][1]


def test_tex_keys(tmp_path: Path) -> None:
    eco, _ = _setup(tmp_path)
    tex = E.to_tex(eco.run())
    assert r"\defres{econ/prof/savings/sol}{" in tex and r"\defres{econ/prof_gpu/tariff_q}{0.002}" in tex
    assert r"\defres{econ/price/astra/output}{50.00}" in tex
    assert r"\defres{econ/prof/wall/mean_diff_ci/lo}{" in tex and r"\defres{econ/prof/lgcost/frozen}{" in tex
    assert "nan" not in tex.lower()
    md = E.to_markdown(eco.run())
    assert "not modelled" in md and "post hoc" in md
