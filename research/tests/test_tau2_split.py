"""Offline tests of the PREREGISTRATION A5.4 tooling in ``jevharness.bench.suites.tau2``: the seeded 15/35 airline
shadow/test split (committed in ``configs/tau2/airline_task_ids.json``), its CLI, the split-aware summaries and
``report``, and the ``SPLIT`` / ``TASK_IDS`` knobs of ``scripts/run_tau2.sh`` (``DRY_RUN``). No tau2-bench needed
(the tau2 import in the adapter is optional); with it installed the task list is cross-checked against the dataset.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jevharness.accounting import Ledger
from jevharness.bench.suites import tau2 as jt
from jevharness.harness.trajectory import Trajectory

REPO = Path(__file__).resolve().parents[1]
SHADOW = ["0", "1", "5", "7", "12", "14", "20", "27", "28", "33", "34", "36", "43", "47", "49"]
ALL = [str(i) for i in range(50)]
TEST = [i for i in ALL if i not in SHADOW]


# ------------------------------------------------------------------ the split
def test_airline_split_is_deterministic_disjoint_and_pinned():
    ids = jt.airline_task_ids(check_installed=False)
    assert ids == ALL
    shadow, test = jt.airline_split(ids)
    assert (shadow, test) == (SHADOW, TEST)
    assert len(shadow) == 15 and len(test) == 35 and not set(shadow) & set(test) and set(shadow) | set(test) == set(ids)
    # order- and type-insensitive (the shuffle runs over the SORTED ids), seed-sensitive
    assert jt.airline_split(reversed(ids)) == jt.airline_split(range(50)) == (shadow, test)
    assert jt.airline_split(ids, seed=1) != (shadow, test)
    assert jt.airline_split(ids, n_shadow=0) == ([], ALL)
    with pytest.raises(ValueError):
        jt.airline_split(ids, n_shadow=51)
    # the committed record carries the provenance and the split the function reproduces
    rec = json.loads(jt.AIRLINE_TASK_IDS_FILE.read_text())
    assert rec["source"]["repo"].endswith("sierra-research/tau2-bench") and rec["source"]["commit"] == "5bfa7e3"
    assert rec["source"]["path"] == "data/tau2/domains/airline/tasks.json" and len(rec["source"]["sha256"]) == 64
    assert rec["split"]["seed"] == jt.SPLIT_SEED == 0 and rec["split"]["n_shadow"] == jt.SPLIT_N_SHADOW == 15
    assert (rec["split"]["shadow"], rec["split"]["test"]) == (shadow, test)
    assert jt.airline_task_split() == (shadow, test)   # asserts the record against airline_split()
    assert jt.split_of("airline", "0") == "shadow" and jt.split_of("airline", 2) == "test"
    assert jt.split_of("retail", "0") is None and jt.split_of("airline", "99") is None
    assert jt.split_of(None, "0") is None


def test_split_cli_prints_both_lists(capsys, tmp_path):
    jt.main(["--split", "airline"])        # the documented spelling; cross-checks tau2's dataset when installed
    doc = json.loads(capsys.readouterr().out)
    assert doc == {"domain": "airline", "seed": 0, "n_shadow": 15, "n": 50, "shadow": SHADOW, "test": TEST}
    jt.main(["split", "airline", "--which", "shadow", "--format", "args", "--no-check"])
    assert capsys.readouterr().out.split() == SHADOW
    jt.main(["split", "airline", "--which", "test", "--format", "args", "--no-check"])
    assert capsys.readouterr().out.split() == TEST
    jt.main(["split", "airline", "--format", "lines", "--no-check"])
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["shadow: " + " ".join(SHADOW), "test: " + " ".join(TEST)]
    jt.main(["split", "airline", "--seed", "1", "--no-check"])   # another seed is computed, never the pinned one
    assert json.loads(capsys.readouterr().out)["shadow"] != SHADOW
    with pytest.raises(SystemExit):
        jt.main(["split", "retail"])                              # only airline has a shadow split
    with pytest.raises(SystemExit):
        jt.main(["split", "airline", "--format", "args", "--no-check"])   # args needs --which shadow|test
    # --ids-file: the TASK_IDS=<file> knob of run_tau2.sh
    f = tmp_path / "ids.txt"
    f.write_text("3, 4\n# comment\n7 # trailing\n")
    jt.main(["split", "--ids-file", str(f), "--format", "args"])
    assert capsys.readouterr().out.split() == ["3", "4", "7"]
    jt.main(["split", "--ids-file", str(jt.AIRLINE_TASK_IDS_FILE), "--format", "json"])
    assert json.loads(capsys.readouterr().out) == ALL


def test_read_task_ids_formats(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([1, "2", 3]))
    (tmp_path / "b.json").write_text(json.dumps({"task_ids": ["5", "6"], "other": 1}))
    (tmp_path / "c.txt").write_text("1\n2,3\n\n4 5\n")
    (tmp_path / "d.json").write_text(json.dumps({"nope": 1}))
    assert jt.read_task_ids(tmp_path / "a.json") == ["1", "2", "3"]
    assert jt.read_task_ids(tmp_path / "b.json") == ["5", "6"]
    assert jt.read_task_ids(tmp_path / "c.txt") == ["1", "2", "3", "4", "5"]
    with pytest.raises(ValueError):
        jt.read_task_ids(tmp_path / "d.json")


# ------------------------------------------------------------------ split-aware summaries / report
def _traj(domain: str, task_id: str, trial: int = 0, success: bool = True, config: str = "arm",
          split: str | None = "auto") -> Trajectory:
    t = Trajectory(task_id=f"{domain}/{task_id}#{trial}", config=config, backend=None, llm="llm",
                   result={"success": success, "score": float(success), "reward": float(success)},
                   usage=Ledger().summary(),
                   meta={"tau2": {"domain": domain, "task_id": task_id, "trial": trial, "sidecar": True,
                                  "agent_cost": 0.01, "user_cost": 0.001}})
    if split != "auto":
        t.meta["tau2"]["split"] = split
    return t


def test_filter_split_and_summarize_by_split():
    # airline: 50 tasks, the test tasks succeed and the shadow tasks fail; no split annotation (classified by id)
    air = [_traj("airline", i, success=i in TEST, split=None) for i in ALL]
    assert len(jt.filter_split(air, "all")) == 50
    assert [t.meta["tau2"]["task_id"] for t in jt.filter_split(air, "shadow")] == SHADOW
    assert [t.meta["tau2"]["task_id"] for t in jt.filter_split(air, "test")] == TEST
    by = jt.summarize_by_split(air)
    assert set(by) == {"test", "shadow"} and by["test"]["n"] == 35 and by["shadow"]["n"] == 15
    assert by["test"]["avg_reward"] == 1.0 and by["shadow"]["avg_reward"] == 0.0 and by["test"]["pass^1"] == 1.0
    with pytest.raises(ValueError):
        jt.filter_split(air, "train")
    # retail has no shadow split: every task is a test task, and nothing is reported per split
    ret = [_traj("retail", str(i)) for i in range(4)]
    assert jt.filter_split(ret, "test") == ret and jt.filter_split(ret, "shadow") == [] and jt.summarize_by_split(ret) == {}
    mixed = air + ret
    assert len(jt.filter_split(mixed, "test")) == 39 and jt.summarize_by_split(mixed)["test"]["n"] == 39
    # an explicit annotation (convert writes meta.tau2.split) wins over the id
    odd = _traj("airline", "2", split="shadow")
    assert jt.filter_split([odd], "shadow") == [odd] and jt.filter_split([odd], "test") == []
    assert jt.filter_split([t.to_dict() for t in air], "shadow")[0]["meta"]["tau2"]["task_id"] == "0"  # dicts too


def test_report_split_writes_filtered_dirs(tmp_path, capsys):
    d = tmp_path / "airline"
    d.mkdir()
    for arm in ("llm_only", "jev_x"):
        for trial in range(2):
            for i in ALL:
                _traj("airline", i, trial, success=(arm == "jev_x" or i in TEST), config=arm).dump(d / f"{arm}.jsonl")
    jt.main(["report", str(d), "--split", "test", "--no-report"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["split"] == "test" and set(doc["arms"]) == {"llm_only", "jev_x"}
    row = doc["arms"]["llm_only"]
    assert row["all"]["n"] == 100 and row["test"]["n"] == 70 and row["shadow"]["n"] == 30
    assert row["all"]["pass^2"] == 0.7 and row["test"]["pass^2"] == 1.0 and row["shadow"]["pass^2"] == 0.0
    out = tmp_path / "airline_test"
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["jev_x.jsonl", "llm_only.jsonl"]
    kept = [json.loads(ln) for ln in (out / "llm_only.jsonl").read_text().splitlines()]
    assert len(kept) == 70 and {k["meta"]["tau2"]["task_id"] for k in kept} == set(TEST)
    jt.main(["report", str(d), "--test-only", "--out", str(tmp_path / "custom"), "--no-report"])
    capsys.readouterr()
    assert len((tmp_path / "custom" / "jev_x.jsonl").read_text().splitlines()) == 70
    jt.main(["report", str(d), "--split", "shadow"])         # with `jevbench report` on the 15-task dir
    text = capsys.readouterr().out
    assert "Episode metrics" in text and "llm_only" in text
    shadow = [json.loads(ln) for ln in (tmp_path / "airline_shadow" / "llm_only.jsonl").read_text().splitlines()]
    assert len(shadow) == 30 and {k["meta"]["tau2"]["task_id"] for k in shadow} == set(SHADOW)
    with pytest.raises(SystemExit):
        jt.main(["report", str(tmp_path / "empty"), "--no-report"])


def _task_ids(cmd: list[str]) -> list[str] | None:
    if "--task-ids" not in cmd:
        return None
    i = cmd.index("--task-ids") + 1
    out = []
    while i < len(cmd) and not cmd[i].startswith("--"):
        out.append(cmd[i])
        i += 1
    return out


def test_experiment_yaml_split_key_resolves_to_task_ids():
    base = {"name": "x", "domain": "airline", "agent_llm": "a", "user_llm": "u",
            "arms": [{"name": "llm_only", "config": str(REPO / "configs" / "tau2" / "llm_only.yaml")}]}
    (_, cmd, _), = jt.experiment_commands({**base, "split": "shadow"})
    assert _task_ids(cmd) == SHADOW
    (_, cmd, _), = jt.experiment_commands({**base, "split": "test", "num_tasks": 5})
    assert _task_ids(cmd) == TEST and cmd[cmd.index("--num-tasks") + 1] == "5"
    (_, cmd, _), = jt.experiment_commands({**base, "split": "test", "task_ids": ["3"]})   # explicit ids win
    assert _task_ids(cmd) == ["3"]
    (_, cmd, _), = jt.experiment_commands({**base, "domain": "retail", "split": "test"})   # every retail task
    assert "--task-ids" not in cmd
    with pytest.raises(ValueError):
        jt.experiment_commands({**base, "domain": "retail", "split": "shadow"})
    with pytest.raises(ValueError):
        jt.experiment_commands({**base, "split": "train"})


# ------------------------------------------------------------------ scripts/run_tau2.sh dispatch
def _dry(tmp_path: Path, **env: str) -> str:
    e = {**os.environ, "DRY_RUN": "1", "PY": sys.executable, "OUT": str(tmp_path / "out"), "NUM_TRIALS": "3", **env}
    r = subprocess.run(["bash", str(REPO / "scripts" / "run_tau2.sh")], env=e, capture_output=True, text=True,
                       cwd=REPO, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stderr


def _runs(log: str) -> list[list[str]]:
    return [ln[2:].split() for ln in log.splitlines() if ln.startswith("+ ") and " run " in ln]


def test_run_tau2_dry_run_split_shadow_dispatch(tmp_path):
    log = _dry(tmp_path, SPLIT="shadow", DOMAINS="airline retail", ARMS="jev_shadow")
    runs = _runs(log)
    assert len(runs) == 1, log
    cmd = runs[0]
    assert "configs/tau2/jev_shadow.yaml" in cmd and cmd[cmd.index("--jev-name") + 1] == "jev_shadow"
    assert cmd[cmd.index("--domain") + 1] == "airline" and _task_ids(cmd) == SHADOW
    assert cmd[cmd.index("--task-split-name") + 1] == "base" and "--auto-resume" in cmd and "--num-tasks" not in cmd
    assert "retail: SPLIT=shadow selects no task" in log and "--domain retail" not in log
    # the airline summaries: `jevbench report`, then the A5.4 per-split reports (35 test / 15 shadow)
    assert "jevharness.bench.cli report" in log
    assert f"--split test" in log and "--split shadow --no-report" in log


def test_run_tau2_dry_run_split_test_and_task_ids(tmp_path):
    log = _dry(tmp_path, SPLIT="test", DOMAINS="airline retail", ARMS="llm_only")
    runs = _runs(log)
    assert [c[c.index("--domain") + 1] for c in runs] == ["airline", "retail"]
    assert _task_ids(runs[0]) == TEST and _task_ids(runs[1]) is None     # retail: every task is a test task
    log = _dry(tmp_path, DOMAINS="airline", ARMS="llm_only")               # SPLIT=all (default): no --task-ids
    assert _task_ids(_runs(log)[0]) is None
    # TASK_IDS overrides SPLIT; comma or space separated, or a file
    log = _dry(tmp_path, SPLIT="shadow", TASK_IDS="3,4", DOMAINS="airline", ARMS="jev_ts_cascade", TAUS="0.9")
    (cmd,) = _runs(log)
    assert _task_ids(cmd) == ["3", "4"] and cmd[cmd.index("--jev-name") + 1] == "jev_ts_cascade_tau0.9"
    assert "tool_select.tau=0.9" in cmd and "tool_params.tau=0.9" in cmd
    f = tmp_path / "ids.txt"
    f.write_text("2\n6 # the first two test tasks\n")
    log = _dry(tmp_path, TASK_IDS=str(f), DOMAINS="retail", ARMS="jev_guard", NUM_TASKS="1")
    (cmd,) = _runs(log)
    assert _task_ids(cmd) == ["2", "6"] and cmd[cmd.index("--num-tasks") + 1] == "1"
    r = subprocess.run(["bash", str(REPO / "scripts" / "run_tau2.sh")], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "DRY_RUN": "1", "SPLIT": "bogus", "OUT": str(tmp_path / "o")})
    assert r.returncode == 2 and "SPLIT must be" in r.stderr
