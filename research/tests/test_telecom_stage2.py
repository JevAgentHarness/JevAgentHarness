"""scripts/telecom_stage2.py and scripts/run_stage2_tau2.sh (docs/PREREG_A7_STAGE2.md A7.15) on a fixture pool: the
inventory (ids, splits, issue type, persona, exact-duplicate audit), the 32 seeded validation tasks, the outcome-blind
sample size rule, a sample disjoint from validation that is frozen once, and the run script's dry run (the `full`
split, one attempt, ids with brackets passed intact). No tau2 install, no simulation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import telecom_stage2 as T  # noqa: E402


def _pool(d: Path, n: int = 120) -> list[str]:
    d.mkdir(parents=True)
    tasks, ids = [], []
    for i in range(n):
        issue = ("mms_issue", "mobile_data_issue", "service_issue")[i % 3]
        persona = ("None", "Easy", "Hard")[i % 3]
        tid = f"[{issue}]fault_{i}|data_mode_off[PERSONA:{persona}]"
        ids.append(tid)
        tasks.append({"id": tid, "user_scenario": {"persona": None, "instructions": {"reason_for_call": f"r{i}"}}})
    tasks.append({"id": "[mms_issue]dup[PERSONA:None]", "user_scenario": tasks[5]["user_scenario"]})   # same content
    (d / "tasks_full.json").write_text(json.dumps(tasks))
    (d / "split_tasks.json").write_text(json.dumps({"full": [t["id"] for t in tasks], "base": ids[:10]}))
    return ids


def test_inventory_validation_and_sample(tmp_path: Path):
    ids = _pool(tmp_path / "data")
    inv_p = tmp_path / "inv.json"
    assert T.main(["inventory", "--data", str(tmp_path / "data"), "--out", str(inv_p)]) == 0
    inv = json.loads(inv_p.read_text())
    assert inv["n_pool"] == 121 and inv["n_exact_duplicates"] == 1 and inv["n_eligible"] == 120
    assert inv["issue_types"] == {"mms_issue": 41, "mobile_data_issue": 40, "service_issue": 40}
    row = next(r for r in inv["tasks"] if r["id"] == ids[0])
    assert row["splits"] == ["base", "full"] and row["persona"] == "None" and row["issue_type"] == "mms_issue"
    val = inv["validation"]["task_ids"]
    assert len(val) == 32 == len(set(val)) and set(val) <= set(ids)
    # A7.15 (c): n = min(800, N_eligible - 32, floor((250 - V) / (2 x 1.25 x C)))
    assert T.sample_size(2285, 10.0, 0.15) == 640 and T.sample_size(2285, 10.0, 0.12) == 800
    assert T.sample_size(120, 10.0, 0.15) == 88
    out = tmp_path / "ids.json"
    assert T.main(["sample", "--inventory", str(inv_p), "--validation-spend", "240", "--cost-per-sim", "0.1",
                   "--out", str(out)]) == 0
    s = json.loads(out.read_text())
    assert s["rule"]["n"] == 40 == len(s["task_ids"]) and not set(s["task_ids"]) & set(val)
    assert "[mms_issue]dup[PERSONA:None]" not in s["task_ids"]                      # duplicates are not eligible
    with pytest.raises(SystemExit, match="frozen once"):
        T.main(["sample", "--inventory", str(inv_p), "--validation-spend", "1", "--cost-per-sim", "0.1",
                "--out", str(out)])
    again = T.sample(json.loads(inv_p.read_text()), 240.0, 0.1)
    assert again["task_ids"] == s["task_ids"]                                       # one seeded order (seed 731)


def test_amended_power_design_has_the_five_row_secondary_table_and_the_frozen_telecom_n():
    import stage2_power as SP

    rows = SP.amended_rows(None)
    assert {r[-1] for r in rows} == {"family", "secondary"} and all(len(r) == 8 for r in rows)
    assert set(SP.LEVELS5) == {"0.05/5", "0.05/4", "0.05/3", "0.05/2", "0.05/1"} and SP.LEVELS5["0.05/5"] == 0.01
    assert set(SP.LEVELS6) == set(SP.LEVELS5) | {"0.05/6"}                            # A7.18 (f): six rows
    assert any(r[0].startswith("S2-4S") for r in rows)
    assert any(r[0].startswith("S2-1X") for r in rows) and not any(r[1] == 114 for r in rows)   # A7.15 replaces 114
    frozen = [r for r in SP.amended_rows(700) if "FROZEN" in r[0]]
    assert [r[1] for r in frozen] == [700, 700] and all(r[2] == 0.05 and r[-1] == "secondary" for r in frozen)


def test_run_script_dry_run_passes_the_full_split_and_bracketed_ids(tmp_path: Path):
    _pool(tmp_path / "data")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs/harness").mkdir(parents=True)
    (repo / "configs/tau2/stage2").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/run_stage2_tau2.sh", repo / "scripts/run_stage2_tau2.sh")
    (repo / "configs/harness/jevagent_best.yaml").write_text("status: selected\n")
    (repo / "configs/tau2/stage2/jevagent_best.yaml").write_text("name: jevagent_best\n")
    assert T.main(["inventory", "--data", str(tmp_path / "data"),
                   "--out", str(repo / "configs/tau2/stage2/telecom_inventory.json")]) == 0
    env = {**os.environ, "DRY_RUN": "1", "VALIDATION": "1", "PY": sys.executable}
    r = subprocess.run(["bash", str(repo / "scripts/run_stage2_tau2.sh")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    run_lines = [ln for ln in r.stdout.splitlines() if " run " in ln and "rerun" not in ln]
    assert len(run_lines) == 2 and all("--task-split-name full" in ln and "--num-trials 1" in ln for ln in run_lines)
    val = json.loads((repo / "configs/tau2/stage2/telecom_inventory.json").read_text())["validation"]["task_ids"]
    assert all(ln.count("PERSONA") == 32 for ln in run_lines) and "stage2_tau2_validation" in run_lines[0]
    assert val[0].replace("[", "\\[").replace("]", "\\]").replace("|", "\\|") in run_lines[0]
