"""Registered seed namespaces (bench/registry.py, PREREGISTRATION A6.13): the runner refuses evaluation and retired
ranges without JEV_EVAL_RUNS=1; the harvest guard excludes them; the frozen task list reproduces from the configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from jevharness.bench import registry as R

ROOT = Path(__file__).resolve().parents[1]
SUITE = {"families": "v2", "distractor_style": "near", "n_distractors": 60, "noise_lines": 20, "strict": True}


def test_ranges_and_overlap_helpers():
    assert R.in_ranges(200000, R.EVAL_SEED_RANGES) and not R.in_ranges(199999, R.EVAL_SEED_RANGES)
    assert R.overlaps(199990, 20, R.EVAL_SEED_RANGES) == [(200000, 200499)]
    assert R.overlaps(5000, 150, R.reserved_ranges()) == []
    assert R.ranges_spec(R.reserved_ranges()) == "200000-200499,100000-100499"
    assert not set(range(*R.CALIBRATION_RANGE)) & {s for lo, hi in R.reserved_ranges() for s in (lo, hi)}


def test_runner_refuses_reserved_seeds_without_the_env_flag(tmp_path, monkeypatch):
    from jevharness.bench.runner import run_experiment

    monkeypatch.delenv(R.EVAL_RUNS_ENV, raising=False)
    lite = {"provider": "simulated", "model": "sim-lite", "error_rate": 0.0, "simulated_latency_ms": 1}
    base = {"name": "g", "llm": lite, "workers": 1, "arms": [{"name": "llm_only"}]}
    for seed0 in (200000, 100010):
        with pytest.raises(SystemExit, match="A6.13"):
            run_experiment({**base, "suite": {"name": "jevgym", "n": 2, "seed0": seed0, **SUITE}}, tmp_path,
                           progress=False)
    assert not (tmp_path / "g").exists() or not list((tmp_path / "g").glob("*.jsonl"))
    # dev seeds never need the flag; with the flag the evaluation range runs
    run_experiment({**base, "name": "dev", "suite": {"name": "jevgym", "n": 2, "seed0": 0, **SUITE}}, tmp_path,
                   progress=False)
    assert (tmp_path / "dev" / "llm_only.jsonl").exists()
    monkeypatch.setenv(R.EVAL_RUNS_ENV, "1")
    run_experiment({**base, "name": "ev", "suite": {"name": "jevgym", "n": 2, "seed0": 200000, **SUITE}}, tmp_path,
                   progress=False)
    ids = [json.loads(l)["task_id"] for l in (tmp_path / "ev" / "llm_only.jsonl").read_text().splitlines()]
    assert all("-20000" in t for t in ids)


def test_harvest_guard_default_excludes_every_reserved_range():
    from jevharness.training.harvest import seed_allowed, seed_ranges

    ex = seed_ranges(R.ranges_spec(R.reserved_ranges()))
    assert ex == [(200000, 200499), (100000, 100499)]
    assert not seed_allowed(200010, 1000, ex) and not seed_allowed(100000, 1000, ex)
    assert seed_allowed(10000, 1000, ex) and not seed_allowed(999, 1000, ex)


def test_frozen_task_list_matches_the_evaluation_configs():
    from jevharness.bench.suites import load_suite

    fz = json.loads((ROOT / R.EVAL_TASKS_FILE).read_text())
    assert fz["seed_range"] == [200000, 200499] and fz["n"] == 500 and len(set(fz["task_ids"])) == 500
    assert hashlib.sha256("\n".join(fz["task_ids"]).encode()).hexdigest() == fz["sha256"]
    for cfg in ("jevgym_tool_select.yaml", "jevgym_meta_control.yaml", "jevgym_tool_select_intern.yaml",
                "baselines_jevgym.yaml"):
        suite = dict(yaml.safe_load((ROOT / "configs/experiments" / cfg).read_text())["suite"])
        assert suite["seed0"] == 200000 and suite["n"] == 500, cfg
    suite = dict(fz["suite"])
    ids = [f().task_id for f in load_suite(suite.pop("name"), **suite)]
    assert ids == fz["task_ids"]


def test_runner_refuses_arms_with_a_pending_calibration_tau(tmp_path, monkeypatch):
    """A6.16: a cascade arm whose tau is still `calibration: pending` (the Intern-Decision C1 arms before their
    fit) is skipped with a message, never run silently at the placeholder; JEV_ALLOW_PENDING_TAU=1 overrides."""
    from jevharness.bench.runner import run_experiment

    monkeypatch.delenv("JEV_ALLOW_PENDING_TAU", raising=False)
    lite = {"provider": "simulated", "model": "sim-lite", "error_rate": 0.0, "simulated_latency_ms": 1}
    exp = {"name": "pend", "suite": {"name": "jevgym", "n": 2, "seed0": 0, **SUITE}, "llm": lite, "workers": 1,
           "s1": {"type": "simulated", "accuracy": 0.9, "latency_ms": 1},
           "arms": [{"name": "llm_only"},
                    {"name": "c1_pending", "tool_select": {"mode": "cascade", "tau": 0.9, "calibration": "pending"}}]}
    run_experiment(exp, tmp_path, progress=False)
    assert (tmp_path / "pend" / "llm_only.jsonl").exists() and not (tmp_path / "pend" / "c1_pending.jsonl").exists()
    # the matched-rate placeholder override does not lift the calibration guard (GPU round 3 review)
    monkeypatch.setenv("JEV_ALLOW_PLACEHOLDER", "1")
    run_experiment(exp, tmp_path / "ph", progress=False)
    assert not (tmp_path / "ph" / "pend" / "c1_pending.jsonl").exists()
    monkeypatch.delenv("JEV_ALLOW_PLACEHOLDER")
    monkeypatch.setenv("JEV_ALLOW_PENDING_TAU", "1")
    run_experiment(exp, tmp_path / "b", progress=False)
    assert (tmp_path / "b" / "pend" / "c1_pending.jsonl").exists()
    # the Intern-Decision fits landed 2026-10-02 (GPU round 3): the C1 arms carry their fitted values, tool_select
    # always-defer ("never"), and no `calibration: pending` key remains in any evaluation config
    cfg = yaml.safe_load((ROOT / "configs/experiments/jevgym_tool_select_intern.yaml").read_text())
    arms = {a["name"]: a for a in cfg["arms"]}
    for arm in ("intern_ts_cascade_tauhat", "intern_ts_cascade_tau0.9_s2lite"):
        assert arms[arm]["tool_select"].get("always_defer") is True
        assert "calibration" not in arms[arm]["tool_select"] and "calibration" not in arms[arm]["tool_params"]


def test_stage2_block_is_reserved_and_refused_without_the_env_flag(tmp_path, monkeypatch):
    """PREREG_A7_STAGE2 A7.2 (a): the Stage-2 fresh seeds 300000-300499 (and the spare 300500-300999) are refused by the
    runner without JEV_EVAL_RUNS=1, excluded by every training guard, and the Stage-1 `reserved_ranges()` (the source of
    docs/PREREG_EFFECTIVE.md) is unchanged."""
    from jevharness.bench.runner import run_experiment
    from jevharness.training import data_v1
    from jevharness.training.harvest import seed_allowed, seed_ranges

    assert R.STAGE2_EVAL_RANGES == ((300000, 300499),) and R.STAGE2_RESERVED_RANGES == ((300000, 300999),)
    assert R.ranges_spec(R.training_exclusions()) == ("200000-200499,100000-100499,300000-300999,310000-310999,"
                                                      "311000-311999,321000-321999,330000-339999")   # + E7 / E13 / E7d / E21
    assert R.ranges_spec(R.reserved_ranges()) == "200000-200499,100000-100499"
    ex = seed_ranges(R.ranges_spec(R.training_exclusions()))
    for s in (300000, 300499, 300500, 300999):
        assert not seed_allowed(s, 1000, ex) and R.in_ranges(s, data_v1.FORBIDDEN_RANGES)
    assert seed_allowed(30000, 1000, ex) and seed_allowed(301000, 1000, ex)
    monkeypatch.delenv(R.EVAL_RUNS_ENV, raising=False)
    lite = {"provider": "simulated", "model": "sim-lite", "error_rate": 0.0, "simulated_latency_ms": 1}
    base = {"name": "s2", "llm": lite, "workers": 1, "arms": [{"name": "llm_only"}]}
    for seed0, n in ((300000, 2), (300990, 5), (299999, 2)):
        with pytest.raises(SystemExit, match="STAGE-2"):
            run_experiment({**base, "suite": {"name": "jevgym", "n": n, "seed0": seed0, **SUITE}}, tmp_path,
                           progress=False)
    assert not (tmp_path / "s2").exists() or not list((tmp_path / "s2").glob("*.jsonl"))
    monkeypatch.setenv(R.EVAL_RUNS_ENV, "1")
    run_experiment({**base, "name": "s2ok", "suite": {"name": "jevgym", "n": 2, "seed0": 300000, **SUITE}}, tmp_path,
                   progress=False)
    assert (tmp_path / "s2ok" / "llm_only.jsonl").exists()


def test_stage2_frozen_task_list_matches_the_stage2_config():
    from jevharness.bench.suites import load_suite

    fz = json.loads((ROOT / R.STAGE2_TASKS_FILE).read_text())
    assert fz["seed_range"] == [300000, 300499] and fz["n"] == 500 and len(set(fz["task_ids"])) == 500
    assert hashlib.sha256("\n".join(fz["task_ids"]).encode()).hexdigest() == fz["sha256"]
    assert fz["sha256"] == "c167829adccbf843052d6ea8eb207a37eb13e7e1a8281cbddbbac9af976e704e"
    cfg = yaml.safe_load((ROOT / "configs/experiments/stage2_jevgym.yaml").read_text())
    assert cfg["suite"] == fz["suite"]
    stage1 = dict(yaml.safe_load((ROOT / "configs/experiments/jevgym_tool_select.yaml").read_text())["suite"])
    assert {k: v for k, v in cfg["suite"].items() if k != "seed0"} == {k: v for k, v in stage1.items() if k != "seed0"}
    suite = dict(fz["suite"])
    ids = [f().task_id for f in load_suite(suite.pop("name"), **suite)]
    assert ids == fz["task_ids"]
    assert not set(json.loads((ROOT / R.EVAL_TASKS_FILE).read_text())["task_ids"]) & set(ids)
