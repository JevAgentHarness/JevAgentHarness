"""Prompt-cache sensitivity columns of the episode report / score card (results/pilot_jev/CACHE_NOTES.md):
``cost_usd`` stays the billed ledger figure; ``cost_usd_cache_modelled`` re-prices the MEASURED cached tokens,
``cost_usd_no_cache`` is the list price without a cache, ``cached_frac`` the measured cached share. Offline."""

from __future__ import annotations

import json

import pytest

from jevharness.accounting import Price, PriceTable
from jevharness.bench.report import CACHE_KEYS, episode_cache_columns, episode_cache_costs, report
from jevharness.bench.scorecard import compute_scorecard

PRICES = PriceTable({"flash": Price(input_per_m=1.5, output_per_m=9.0, cached_input_per_m=0.15),
                     "nocache-rate": Price(input_per_m=1.0, output_per_m=2.0)})


def _s2(model: str, inp: int, out: int, cached: int, cost: float, shadow: bool = False, step: int = 0) -> dict:
    return {"component": "s2", "backend": "gemini", "model": model, "decision_point": "tool_select",
            "input_tokens": inp, "output_tokens": out, "cost_usd": cost, "latency_ms": 100.0, "cached": False,
            "priced": model != "mystery", "step": step,
            "meta": {"cached_input_tokens": cached, "reasoning_tokens": 0, "n_tools": 69, **({"shadow": True} if shadow else {})}}


def _traj(task: str, config: str, calls: list[dict], success: bool = True, s1_cost: float = 0.0) -> dict:
    live = [c for c in calls if not c["meta"].get("shadow")]
    s2 = [c for c in live if c["component"] == "s2"]
    total = sum(c["cost_usd"] for c in live) + s1_cost
    usage = {"total": {"calls": len(live) + int(s1_cost > 0), "input_tokens": sum(c["input_tokens"] for c in live),
                       "output_tokens": sum(c["output_tokens"] for c in live), "cost_usd": total, "latency_ms": 100.0},
             "s2": {"calls": len(s2), "input_tokens": sum(c["input_tokens"] for c in s2),
                    "output_tokens": sum(c["output_tokens"] for c in s2), "cost_usd": sum(c["cost_usd"] for c in s2),
                    "latency_ms": 100.0},
             "unpriced_models": sorted({c["model"] for c in live if not c["priced"]})}
    if s1_cost:
        usage["s1"] = {"calls": 1, "input_tokens": 10, "output_tokens": 0, "cost_usd": s1_cost, "latency_ms": 1.0}
    return {"task_id": task, "config": config, "backend": None, "llm": "flash", "steps": [{"step": 0, "tool": "finish", "args": {}, "decisions": []}],
            "final_answer": "x", "result": {"success": success}, "usage": usage, "calls": calls, "wall_ms": 1.0,
            "error": None, "meta": {}, "probes": []}


def test_episode_cache_costs_reprices_measured_cached_tokens_only():
    p = PRICES.get("flash")
    billed = p.cost(5392, 28, 1932)                       # what the ledger recorded (cached rate applied)
    t = _traj("t1", "llm_only", [
        _s2("flash", 5097, 18, 0, p.cost(5097, 18, 0), step=0),
        _s2("flash", 5392, 28, 1932, billed, step=1),
        _s2("flash", 9999, 1, 9999, 0.0, shadow=True),   # measurement call: excluded, like usage.total
    ], s1_cost=0.001)
    r = episode_cache_costs(t, PRICES)
    total = t["usage"]["total"]["cost_usd"]
    assert r["cost_usd_cache_modelled"] == pytest.approx(total)          # the ledger already priced the cached tokens
    assert r["cost_usd_no_cache"] == pytest.approx(total + 1932 * (1.5 - 0.15) / 1e6)
    assert r["cached_frac"] == pytest.approx(1932 / (5097 + 5392)) and r["cached_tokens"] == 1932
    # a price table that gained a cached rate after the run: modelled < recorded, no-cache = recorded
    old = PriceTable({"flash": Price(input_per_m=1.5, output_per_m=9.0)})
    t2 = _traj("t1", "llm_only", [_s2("flash", 5392, 28, 1932, old.get("flash").cost(5392, 28, 1932))])
    r2 = episode_cache_costs(t2, PRICES)
    assert r2["cost_usd_no_cache"] == pytest.approx(t2["usage"]["total"]["cost_usd"])
    assert r2["cost_usd_cache_modelled"] == pytest.approx(PRICES.get("flash").cost(5392, 28, 1932))
    # unpriced model: recorded cost kept, cached share still measured
    t3 = _traj("t1", "llm_only", [_s2("mystery", 4000, 10, 2000, 0.0)])
    r3 = episode_cache_costs(t3, PRICES)
    assert r3["cost_usd_cache_modelled"] == 0.0 == r3["cost_usd_no_cache"] and r3["cached_frac"] == 0.5
    # no System-2 calls at all
    assert episode_cache_costs(_traj("t1", "x", []), PRICES)["cached_frac"] == 0.0
    assert episode_cache_columns([], PRICES) == {}
    cols = episode_cache_columns([t, t3], PRICES)
    assert set(CACHE_KEYS) <= set(cols) and cols["cached_frac"] == pytest.approx((r["cached_frac"] + 0.5) / 2)


def test_report_and_scorecard_carry_the_cache_columns(tmp_path):
    # report() prices with configs/prices: a real model id (a bare "flash" matches no row since the whole-segment
    # lookup, A7.12 (h)); the test table's "flash" row has the same rates as gemini-3.5-flash
    p = PRICES.get("flash")
    base = [_traj(f"t{i}", "llm_only", [_s2("gemini-3.5-flash", 5100, 20, 0, p.cost(5100, 20, 0), step=0),
                                        _s2("gemini-3.5-flash", 5400, 20, 1932, p.cost(5400, 20, 1932), step=1)]) for i in range(4)]
    assist = [_traj(f"t{i}", "assist", [_s2("gemini-3.5-flash", 500, 20, 0, p.cost(500, 20, 0))], success=i < 3) for i in range(4)]
    for name, trajs in (("llm_only", base), ("assist", assist)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trajs))
    md = report(tmp_path, scorecard=False)
    head = md.splitlines()[2]
    assert all(k in head for k in CACHE_KEYS)
    doc = json.loads(report(tmp_path, fmt="json", scorecard=False))
    ep = doc["episode"]["llm_only"]
    assert ep["cost_usd_cache_modelled"] == pytest.approx(ep["cost_usd"])     # priced with the same table at run time
    assert ep["cost_usd_no_cache"] > ep["cost_usd"] and ep["cached_frac"] == pytest.approx(1932 / 10500)
    assert doc["episode"]["assist"]["cached_frac"] == 0.0
    card = compute_scorecard({"llm_only": base, "assist": assist}, baseline="llm_only", n_boot=50)
    row = card["arms"]["llm_only"]
    assert row["cost_usd"] == pytest.approx(ep["cost_usd"]) and row["cost_usd_cache_modelled"] == pytest.approx(ep["cost_usd"])
    assert row["cost_usd_no_cache"] == pytest.approx(ep["cost_usd_no_cache"])
    from jevharness.bench.scorecard import to_markdown

    full = to_markdown(card)
    assert "cache-mod $" in full and "cached %" in full and "no-cache $" in full
    assert "cache-mod $" not in to_markdown(card, compact=True)
