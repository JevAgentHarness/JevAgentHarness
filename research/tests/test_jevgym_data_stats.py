"""scripts/jevgym_data_stats.py: JevGym plan statistics from task generation only (no episode). On a development-seed
list it reads each task's canonical plan, refuses a list the generator no longer reproduces, writes the
\\defres{data_stats/jevgym_eval/...} keys the paper reads; the committed numbers match a recomputation."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import jevgym_data_stats as DS  # noqa: E402

SUITE = {"name": "jevgym", "n": 6, "seed0": 0, "families": "v2", "distractor_style": "near", "n_distractors": 60,
         "noise_lines": 20, "strict": True}


def _frozen() -> dict:
    from jevharness.bench.suites import load_suite

    s = dict(SUITE)
    ids = [f().task_id for f in load_suite(s.pop("name"), **s)]
    return {"registered": "test", "config": "dev.yaml", "suite": SUITE, "seed_range": [0, 5], "sha256": "x",
            "task_ids": ids}


def test_plan_statistics_on_development_seeds(tmp_path: Path):
    fz = _frozen()
    st = DS.compute(fz)
    assert st["n_tasks"] == 6 and st["n_families"] == 2                 # v2 round-robin: clarify / multi
    assert st["plan_steps_min"] >= 5 and st["plan_steps_max"] <= 10 and st["plan_calls_median"] == st["plan_steps_median"] - 1
    assert 1 <= st["n_distinct_plans"] <= 6 and sum(st["distinct_plans"].values()) == 6
    assert st["tools_offered_median"] > 60                              # 60 distractors + the suite's tools + finish
    p = tmp_path / "list.json"
    p.write_text(json.dumps(fz))
    assert DS.main(["--tasks", str(p), "--out", str(tmp_path / "s.json"), "--tex", str(tmp_path / "s.tex")]) == 0
    keys = dict(re.findall(r"\\defres\{([^}]*)\}\{(.*)\}", (tmp_path / "s.tex").read_text()))
    for k in ("plan_steps_median", "n_distinct_plans", "n_tasks", "population"):
        assert f"data_stats/jevgym_eval/{k}" in keys
    bad = {**fz, "task_ids": fz["task_ids"][::-1]}
    with pytest.raises(SystemExit):
        DS.compute(bad)


def test_committed_stage2_statistics_match_a_recomputation():
    rec = json.loads(DS.OUT_JSON.read_text())
    st = DS.compute(json.loads(DS.TASKS.read_text()))
    assert {k: rec[k] for k in st} == json.loads(json.dumps(st))
