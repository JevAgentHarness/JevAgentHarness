"""Paper figures render from a run directory and read arm profiles from experiment.yaml."""

from __future__ import annotations

import json

import pytest
import yaml

from jevharness.bench import plots

EXP = {"name": "e", "suite": {"name": "jevgym"}, "s1": {"type": "jev"},
       "s1_backends": {"lite": {"type": "llm_generative"}},
       "arms": [{"name": "llm_only"}, {"name": "jev_ts_replace", "tool_select": {"mode": "replace"}},
                {"name": "lite_ts_assist_k3", "s1": "lite", "tool_select": {"mode": "assist", "k": 3}}],
       "sweeps": [{"base": {"name": "jev_ts_cascade", "tool_select": {"mode": "cascade"}},
                   "set": "tool_select.tau", "values": [0.5, 0.9]}]}


def _traj(arm, i, ok, cost):
    return {"task_id": str(i), "config": arm, "result": {"success": ok}, "steps": [], "calls": [],
            "usage": {"total": {"cost_usd": cost, "latency_ms": 1000.0}, "s2": {"calls": 2}, "s1": {"calls": 1}}}


@pytest.fixture()
def run_dir(tmp_path):
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(EXP))
    for arm, p, c in [("llm_only", 1.0, 0.01), ("jev_ts_replace", 0.8, 0.002), ("lite_ts_assist_k3", 0.9, 0.006),
                      ("jev_ts_cascade_tau0.5", 0.85, 0.004), ("jev_ts_cascade_tau0.9", 0.95, 0.007)]:
        rows = [_traj(arm, i, i < 10 * p, c) for i in range(10)]
        (tmp_path / f"{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_arm_profiles_expand_sweeps_and_bindings(run_dir):
    prof = plots.arm_profiles(run_dir)
    assert prof["llm_only"] == {"backend": "", "mode": ""}
    assert prof["jev_ts_replace"] == {"backend": "jev", "mode": "replace"}
    assert prof["lite_ts_assist_k3"] == {"backend": "lite", "mode": "assist"}
    assert prof["jev_ts_cascade_tau0.9"] == {"backend": "jev", "mode": "cascade"}


def test_pareto_and_sweep_render(run_dir, tmp_path):
    pytest.importorskip("matplotlib")
    plots.pareto(run_dir, tmp_path / "out" / "p.pdf")
    plots.sweep(run_dir, "jev_ts_cascade_tau", tmp_path / "out" / "s.pdf")
    assert (tmp_path / "out" / "p.pdf").stat().st_size > 1000
    assert (tmp_path / "out" / "s.pdf").stat().st_size > 1000


def test_shadow_arm_profile_and_pareto_skip(run_dir, tmp_path):
    exp = yaml.safe_load((run_dir / "experiment.yaml").read_text())
    exp["arms"].append({"name": "jev_shadow", "tool_select": {"mode": "shadow"}})
    (run_dir / "experiment.yaml").write_text(yaml.safe_dump(exp))
    (run_dir / "jev_shadow.jsonl").write_text((run_dir / "llm_only.jsonl").read_text())
    assert plots.arm_profiles(run_dir)["jev_shadow"] == {"backend": "", "mode": "shadow"}
    pytest.importorskip("matplotlib")
    plots.pareto(run_dir, tmp_path / "p.pdf")  # renders without the shadow arm
