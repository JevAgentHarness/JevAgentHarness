"""Review P1-4: every System-1-backed DecisionLog carries its pass accounting (calls, sequential rounds, backend-reported
passes) from the decision's ledger tap; a Choice wider than the backend's width runs the harness tournament and logs
two rounds."""
from __future__ import annotations

import json

from jevharness.bench.runner import run_experiment

SUITE = {"families": "v2", "distractor_style": "near", "n_distractors": 60, "noise_lines": 20, "strict": True}


def _decisions(path):
    out = []
    for line in path.read_text().splitlines():
        t = json.loads(line)
        for st in t.get("steps", []):
            out.extend(st.get("decisions", []))
    return out


def test_decision_logs_carry_calls_rounds_and_passes(tmp_path):
    lite = {"provider": "simulated", "model": "sim-lite", "error_rate": 0.0, "simulated_latency_ms": 1}
    exp = {"name": "pc", "suite": {"name": "jevgym", "n": 2, "seed0": 0, **SUITE}, "llm": lite, "workers": 1,
           "s1": {"type": "simulated", "accuracy": 0.9, "latency_ms": 1},
           "arms": [{"name": "assist_guard", "tool_select": {"mode": "assist", "k": 3},
                     "stop": {"mode": "guard", "tau": 0.5}},
                    {"name": "llm_only"}]}
    run_experiment(exp, tmp_path, progress=False)
    dec = _decisions(tmp_path / "pc" / "assist_guard.jsonl")
    ts = [d for d in dec if d["dp"] == "tool_select"]
    # 69 JevGym tools over the 52-option simulated backend: chunks -> finalists -> final read = 2 sequential rounds
    assert ts and all(d["s1_rounds"] == 2 and d["s1_calls"] >= 2 for d in ts)
    stop = [d for d in dec if d["dp"] == "stop"]
    assert stop and all(d["s1_calls"] == 1 and d["s1_rounds"] == 1 for d in stop)
    # simulated backends report no forward passes: s1_passes stays None (never a fabricated count)
    assert all(d["s1_passes"] is None for d in ts + stop)
    assert all(d.get("s1_calls") is None for d in _decisions(tmp_path / "pc" / "llm_only.jsonl"))
