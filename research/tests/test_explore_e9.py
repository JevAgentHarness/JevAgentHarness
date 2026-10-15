"""E9 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E9: cost-accuracy frontier of multiple-choice decisions"):
the offline readout code of scripts/explore_e9_frontier.py on synthetic records (frontier, GPU tariff, comparisons,
TeX keys), the check's registered settings, and that every key the figure reads is generated."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explore_e9_frontier as X  # noqa: E402

FAMS = [("tool_select", "bfcl:simple_python"), ("tool_params", "bfcl:multiple"), ("tool_select", "when2call:BFCL")]


def _recs(n_right: int, n: int = 10, cost: float = 0.0, lat: float = 100.0, q: int = 1) -> dict[str, dict]:
    out = {}
    for i in range(n):
        dp, src = FAMS[i % 3]
        out[f"item{i}"] = {"id": f"item{i}", "dp": dp, "source": src, "correct": i < n_right, "cost_usd": cost,
                           "latency_ms": lat, "request_questions": q, "priced": True, "input_tokens": 100}
    return out


SYN_MODELS = {"jev": ("Jev", "-", "dm", "api", True), "rl": ("AgentDecision-4B", "-", "dm", "gpu", True),
              "flash": ("Flash", "-", "llm", "api", True), "lite": ("Flash-Lite", "-", "llm", "api", True),
              "flashfc": ("Flash, own call", "-", "fc", "api", True)}


def _synthetic():
    return {"jev": _recs(8, cost=1e-5), "rl": _recs(7, lat=360.0, q=2),            # GPU: 180 ms per decision
            "flash": _recs(9, cost=1e-3), "lite": _recs(7, cost=2e-4), "flashfc": _recs(6, cost=8e-4)}


def test_frontier_keeps_only_undominated_points_sorted_by_cost():
    pts = {"a": (1.0, 0.70), "b": (2.0, 0.90), "c": (3.0, 0.85), "d": (1.0, 0.60), "e": (5.0, 0.95)}
    assert X.frontier(pts) == ["a", "b", "e"]
    assert X.frontier({"x": (1.0, 0.8), "y": (1.0, 0.8)}) == ["x", "y"]        # ties: neither strictly better


def test_gpu_cost_is_tariff_times_latency_split_over_the_request():
    r = {"latency_ms": 360.0, "request_questions": 2}
    assert X.gpu_cost(r) == pytest.approx(X.GPU_USD_PER_H / 3600 * 0.18)
    assert X.decision_cost({"cost_usd": 0.003, "latency_ms": 1e6}, "api") == 0.003


def test_cheapest_at_least_and_errors_count_as_wrong():
    assert X.cheapest_at_least(0.8, {"a": (2.0, 0.85), "b": (1.0, 0.8), "c": (0.5, 0.7)}) == "b"
    assert X.cheapest_at_least(0.99, {"a": (2.0, 0.85)}) is None
    assert not X.is_correct({"correct": True, "error": "Timeout"})


def test_analyse_e1_on_synthetic_records():
    res = X.analyse_e1(_synthetic(), rates=None, models=SYN_MODELS, n_boot=200)
    m = res["models"]
    assert m["jev"]["acc"] == pytest.approx(0.8) and m["rl"]["usd"] == pytest.approx(X.gpu_cost({"latency_ms": 360,
                                                                                                 "request_questions": 2}))
    assert res["frontier"] == ["jev", "flash"]                    # rl, lite, flashfc are dominated by jev
    assert res["frontier_only_dm_below_top_llm"] is True
    c = res["comparisons"]
    assert c["flash"]["jev_gap_pt"]["est"] == pytest.approx(-10.0)
    assert c["flash"]["jev_cost_ratio"] == pytest.approx(100.0)
    assert c["flash"]["cheapest_dm_at_least_as_accurate"] is None
    assert c["lite"]["cheapest_dm_at_least_as_accurate"] == "jev"
    assert c["lite"]["cheapest_dm_cost_ratio"] == pytest.approx(20.0)
    assert set(m["jev"]["families"]) == {"toolchoice", "args", "w2c"}
    assert res["sensitivity"]["zero"]["frontier"][0] == "rl"         # free GPU: the open model becomes the cheapest


def test_tex_keys_signs_and_magnitudes():
    res = X.analyse_e1(_synthetic(), rates=None, models=SYN_MODELS, n_boot=200)
    k = X.tex_keys(res, {"available": False})
    assert k["e9/jev/vs/flash/gap"] == "-10.0" and k["e9/jev/vs/flash/gapabs"] == "10.0"
    assert k["e9/jev/vs/lite/gap"] == "+10.0" and k["e9/jev/vs/flash/ratio"] == "100"
    assert k["e9/jev/lg"] == "-5.0000" and k["e9/front/first"] == "jev" and k["e9/front/rest"] == "flash"
    assert "e9/cheapest/flash/ratio" not in k and k["e9/cheapest/flash/model"] == "none"
    assert not any(r"\tbd" in v for v in k.values())
    assert X.tikz_name("sl0.8b") == "sl08b"


def test_check_config_is_the_registered_one():
    exp = yaml.safe_load(X.CHECK_CFG.read_text())
    b = exp["backends"]
    assert set(b) == set(X.CHECK_BACKENDS)
    assert b["luna_dist"]["llm"] == {"provider": "openai_responses", "model": "gpt-6-luna"}
    assert b["luna_dist"]["reasoning_effort"] == "none" and b["sol_dist"]["reasoning_effort"] == "low"
    assert b["sol_dist"]["llm"]["model"] == "gpt-6.1-sol"
    assert all(v.get("mode", "distribution") == "distribution" for k, v in b.items() if k != "jev")
    assert b["jev"] == {"type": "jev", "provider": "openrouter"}
    assert exp["batch_by_state"] is True and "exclude_ids_file" not in exp       # --max-items 300 = the dev prefix
    assert exp["s1_cache"] == "cache/s1_e9.sqlite"
    reg = (ROOT / "docs/PREREG_A7_STAGE2.md").read_text()
    assert '"E9: cost-accuracy frontier of multiple-choice decisions"' in reg and "$30 of API for E9" in reg


def test_every_key_the_figure_reads_is_generated():
    tex = (ROOT / "paper/generated/e9_frontier.tex")
    if not tex.exists():
        pytest.skip("paper/generated/e9_frontier.tex not generated")
    have = set(re.findall(r"\\defres\{([^}]+)\}", tex.read_text()))
    fig = (ROOT / "paper/figures/fig_frontier.tex").read_text()
    need = set(re.findall(r"\\res\{(e9/[^}]+)\}", fig))
    guarded = set(re.findall(r"\\ifcsname jevres@e9/([^/}]+)/lg\\endcsname", fig))   # drawn only once generated (E12)
    # drawn only on E30's basis (Figure 4a's cheapest-deployment re-plot; e.g. AgentDecision-0.8B = e30/e28)
    guarded |= set(re.findall(r"\\ifcsname jevres@e30/([^/}]+)/lg\\endcsname", fig))
    for model in re.findall(r"\\efP\{[^}]+\}\{([^}]+)\}", fig):
        if model not in guarded:
            need |= {f"e9/{model}/lg", f"e9/{model}/acc"}
    need -= {"e9/#2/lg", "e9/#2/acc", "e9/#1/#2"}   # macro parameters (\\efP, \\efK), not keys
    assert need and need <= have, sorted(need - have)
    names = set(re.findall(r"\\efP\{([^}]+)\}", fig))
    vals = dict(re.findall(r"\\defres\{(e9/front/(?:first|rest))\}\{([^}]*)\}", tex.read_text()))
    assert {vals["e9/front/first"], *vals["e9/front/rest"].split(",")} <= names
