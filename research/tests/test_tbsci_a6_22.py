"""Terminal-Bench-Science, PREREGISTRATION A6.22 (d) 1-2 and (e) 5: the `tbsci.py select` subcommand (validity /
floor / signal / fill / freeze on the Harbor result directories of one calibration pass; it never launches a run),
and in `summarize_harbor.py` the displayed H5 labels, solved-by-t on the verdict row and the complete-case view next
to the full-set view. Offline: fixture Harbor result directories and a synthetic checkout."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
PI = REPO / "adapters" / "pi"
sys.path.insert(0, str(PI))
try:
    import summarize_harbor as sh
    import tbsci
finally:
    sys.path.remove(str(PI))

T0 = datetime(2026, 10, 3, 1, 0, tzinfo=timezone.utc)


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _trial(job: Path, task: str, k: int = 1, reward: float | None = 1.0, run_s: float = 300.0,
           tests: tuple[int, int] | None = None, err: str | None = None, msg: str | None = None,
           start_min: int = 0) -> None:
    """One Harbor trial directory ``<job>/<task>__<k>`` with result.json (+ verifier/ctrf.json)."""
    t = job / f"{task}__{k}"
    (t / "verifier").mkdir(parents=True)
    st = T0 + timedelta(minutes=start_min)
    res = {"task_name": f"terminal-bench-science/{task}", "started_at": _iso(st),
           "finished_at": _iso(st + timedelta(seconds=run_s + 60)), "agent_result": {"cost_usd": 0.1},
           "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
           "exception_info": {"exception_type": err, "exception_message": msg} if err else None,
           "agent_execution": {"started_at": _iso(st), "finished_at": _iso(st + timedelta(seconds=run_s))}}
    (t / "result.json").write_text(json.dumps(res))
    if tests:
        (t / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"summary": {"passed": tests[0],
                                                                                       "tests": tests[1]}}}))


def _valid(cal: Path, task: str, floor: int = 1, total: int = 10) -> None:
    """oracle solves within the cap; nop passes ``floor`` tests."""
    _trial(cal / "oracle", task, reward=1.0, run_s=600, tests=(total, total))
    _trial(cal / "nop", task, reward=0.0, run_s=5, tests=(floor, total))


INFO: dict[str, dict[str, Any]] = {}


def _task(slug: str, domain: str, hours: float) -> str:
    INFO[slug] = {"slug": slug, "domain": f"{domain}-sciences" if domain != "engineering" else "engineering",
                  "expert_hours": hours}
    return slug


def _calibration(cal: Path) -> list[str]:
    pool = []
    # earth: 4 eligible of 8 (one of them after a retried baseline error, one timed out above its floor)
    for slug, h in (("e-solved", 5), ("e-above", 6), ("e-retried", 9), ("e-timeout", 10)):
        pool.append(_task(slug, "earth", h))
        _valid(cal, slug, floor=2)
    _trial(cal / "baseline", "e-solved", reward=1.0, tests=(10, 10))
    _trial(cal / "baseline", "e-above", reward=0.0, tests=(3, 10))
    _trial(cal / "baseline", "e-retried", reward=None, err="EnvironmentStartTimeoutError", msg="build timed out\n...")
    _trial(cal / "baseline", "e-retried", k=2, reward=1.0, tests=(10, 10), start_min=30)
    _trial(cal / "baseline", "e-timeout", reward=0.0, run_s=3600, tests=(5, 10), err="AgentTimeoutError")
    pool.append(_task("e-oracle-fails", "earth", 3))
    _trial(cal / "oracle", "e-oracle-fails", reward=0.0, tests=(9, 10))
    pool.append(_task("e-oracle-slow", "earth", 4))
    _trial(cal / "oracle", "e-oracle-slow", reward=1.0, run_s=1900)
    pool.append(_task("e-build-twice", "earth", 4.5))
    _trial(cal / "oracle", "e-build-twice", reward=1.0)
    _trial(cal / "nop", "e-build-twice", reward=None, err="DockerBuildError", msg="pip: no route")
    _trial(cal / "nop", "e-build-twice", k=2, reward=None, err="DockerBuildError", msg="pip: no route", start_min=9)
    pool.append(_task("e-at-floor", "earth", 2))
    _valid(cal, "e-at-floor", floor=2)
    _trial(cal / "baseline", "e-at-floor", reward=0.0, tests=(2, 10))
    # engineering: one eligible task -> three spare slots
    pool.append(_task("g-1", "engineering", 5))
    _valid(cal, "g-1")
    _trial(cal / "baseline", "g-1", reward=1.0)
    # life: seven eligible, two tied at 2 h (ties by slug); mathematical two; physical six
    for slug, h in (("l-b", 2), ("l-a", 2), ("l-c", 3), ("l-d", 4), ("l-e", 5), ("l-f", 6), ("l-g", 7)):
        pool.append(_task(slug, "life", h))
    for slug, h in (("m-1", 8), ("m-2", 9)):
        pool.append(_task(slug, "mathematical", h))
    for slug, h in (("p-1", 4), ("p-2", 5), ("p-3", 6), ("p-4", 7), ("p-5", 8), ("p-6", 12)):
        pool.append(_task(slug, "physical", h))
    for slug in pool:
        if slug[0] in "lmp":
            _valid(cal, slug)
            _trial(cal / "baseline", slug, reward=1.0)
    return pool


def test_select_applies_validity_floor_signal_and_the_fill_rule(tmp_path):
    cal = tmp_path / "tbsci_cal"
    pool = _calibration(cal)
    sel = tbsci.select_hard(cal, INFO, pool)
    t = sel["tasks"]
    assert sel["pending"] == [] and sel["jobs_found"] == ["oracle", "nop", "baseline"]
    assert t["e-oracle-fails"]["status"] == "dropped" and t["e-oracle-fails"]["stage"] == "validity"
    assert "1900" in t["e-oracle-slow"]["reason"] and t["e-oracle-slow"]["stage"] == "validity"
    assert t["e-build-twice"]["stage"] == "floor" and "infrastructure failed twice" in t["e-build-twice"]["reason"]
    assert t["e-at-floor"]["stage"] == "signal" and t["e-at-floor"]["floor"] == 2
    assert t["e-retried"]["status"] == "eligible"                       # the retry decides
    assert t["e-timeout"]["status"] == "eligible" and t["e-timeout"]["baseline_timed_out"]
    assert t["e-above"]["reason"] == "tests passed above the nop floor"
    # 4 per domain first (earth 4, engineering 1, life 4, mathematical 2, physical 4 = 15), then the 5 spare slots
    # round-robin earth -> physical over the domains with eligible tasks left: life, physical, life, physical, life
    assert sel["by_domain"] == {"earth": ["e-solved", "e-above", "e-retried", "e-timeout"], "engineering": ["g-1"],
                                "life": ["l-a", "l-b", "l-c", "l-d", "l-e", "l-f", "l-g"],
                                "mathematical": ["m-1", "m-2"], "physical": ["p-1", "p-2", "p-3", "p-4", "p-5", "p-6"]}
    assert len(sel["selected"]) == 20 and sel["counts"] == {"eligible": 20, "dropped": 4, "pending": 0}


def test_fill_stops_at_20_and_reports_a_smaller_subset(tmp_path):
    cal = tmp_path / "cal"
    pool = []
    for i in range(30):                                               # 30 eligible physical tasks: 20 taken
        pool.append(_task(f"q-{i:02d}", "physical", 4 + i / 10))
        _valid(cal, f"q-{i:02d}")
        _trial(cal / "baseline", f"q-{i:02d}", reward=1.0)
    sel = tbsci.select_hard(cal, INFO, pool)
    assert sel["selected"] == [f"q-{i:02d}" for i in range(20)]
    sel = tbsci.select_hard(cal, INFO, pool[:3])
    assert len(sel["selected"]) == 3
    text = tbsci.freeze_text(sel, PI / "tbsci_hard_pool.txt", "cal", "abc", "cfff307")
    assert "# N = 3 of 20 (a smaller subset, reported as such)" in text


def test_a_single_errored_trial_is_pending_and_blocks_the_freeze(tmp_path, capsys):
    cal = tmp_path / "cal"
    pool = [_task("w-1", "earth", 5), _task("w-2", "earth", 6)]
    _trial(cal / "oracle", "w-1", reward=None, err="VerifierTimeoutError", msg="verifier exceeded 1800 s")
    _valid(cal, "w-2")                                                 # no baseline trial at all: missing
    sel = tbsci.select_hard(cal, INFO, pool)
    assert sel["pending"] == ["w-1", "w-2"]
    assert "retry needed (VerifierTimeoutError: verifier exceeded 1800 s)" in sel["tasks"]["w-1"]["reason"]
    assert sel["tasks"]["w-2"]["reason"] == "baseline: missing"
    survey = tmp_path / "rows.json"
    survey.write_text(json.dumps(list(INFO.values())))
    pf = tmp_path / "pool.txt"
    pf.write_text("w-1\nw-2\n")
    out = tmp_path / "hard.txt"
    with pytest.raises(SystemExit) as e:
        tbsci.main(["select", str(cal), "--survey", str(survey), "--pool", str(pf), "--allow-other-pool",
                    "--out", str(out)])
    assert e.value.code == 1 and not out.exists() and "pending" in capsys.readouterr().err


def test_select_cli_freezes_the_list_with_its_header_and_never_overwrites(tmp_path):
    cal = tmp_path / "tbsci_cal"
    pool = _calibration(cal)
    survey, pf, out = tmp_path / "rows.json", tmp_path / "pool.txt", tmp_path / "tbsci_hard.txt"
    survey.write_text(json.dumps([{**INFO[s], "set": "main"} for s in pool]))
    pf.write_text("# fixture pool\n" + "\n".join(pool) + "\n")
    cmd = [sys.executable, str(PI / "tbsci.py"), "select", str(cal), "--survey", str(survey), "--pool", str(pf),
           "--out", str(out), "--run-id", "cal-2026-10-03"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "is not the registered pool" in r.stderr     # only the registered pool is accepted
    r = subprocess.run(cmd + ["--allow-other-pool", "--json", str(tmp_path / "sel.json")], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    lines = out.read_text().splitlines()
    head = [ln for ln in lines if ln.startswith("#")]
    assert any(f"sha256 {tbsci.sha256_of(pf)}" in ln for ln in head) and "# calibration run: cal-2026-10-03" in head
    assert any(ln.startswith("# commit: ") for ln in head)
    assert [ln for ln in lines if not ln.startswith("#")] == json.loads((tmp_path / "sel.json").read_text())["selected"]
    assert tbsci.subset_slugs(out) == json.loads((tmp_path / "sel.json").read_text())["selected"]
    assert f"sha256 {tbsci.sha256_of(out)}" in r.stdout and f"{out.stat().st_size} bytes" in r.stdout
    r = subprocess.run(cmd + ["--allow-other-pool"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "exists" in r.stderr                           # frozen once


def test_registered_pool_is_the_committed_one():
    pool = PI / "tbsci_hard_pool.txt"
    assert tbsci.sha256_of(pool) == tbsci.REGISTERED_POOL_SHA256 and pool.stat().st_size == 1389
    assert len(tbsci.subset_slugs(pool)) == 37


# ------------------------------------------------------------------ summarize_harbor.py: labels, views
def test_h5_labels_are_the_a6_22_wording(tmp_path):
    base, arm = tmp_path / "baseline", tmp_path / "jev"
    for i in range(10):                         # 5 co-solved (arm faster), 5 solved by the arm only
        _trial(base, f"t{i}", reward=1.0 if i < 5 else 0.0, run_s=900)
        _trial(arm, f"t{i}", reward=1.0, run_s=300)
    p = sh.paired(sh.summarize([base, arm]), "baseline", n_boot=2000)["jev"]
    assert p["h5_strictly_better"] and p["h5_faster"]
    assert p["h5_verdict"] == "faster and better (faster among jointly solved tasks)" == sh.H5_LABEL_STRICT
    assert sh.H5_LABEL_REGISTERED == "passes the exploratory success screen; faster among jointly solved tasks"
    # solved-by-t on the verdict row (600 / 1800 / 3600 s), both arms
    assert p["solved_by_600s"] == 1.0 and p["solved_by_600s_base"] == 0.0 and p["solved_by_1800s_base"] == 0.5


def test_complete_case_view_drops_tasks_with_an_errored_trial_and_lists_them(tmp_path):
    base, arm = tmp_path / "baseline", tmp_path / "jev"
    for t in ("a", "b", "c"):
        _trial(base, t, k=1, reward=1.0, run_s=600)
        _trial(base, t, k=2, reward=1.0, run_s=600)
    _trial(arm, "a", k=1, reward=None, err="DockerBuildError", msg="image build failed\ntrace", run_s=10)
    _trial(arm, "a", k=2, reward=1.0, run_s=300)
    _trial(arm, "b", k=1, reward=0.0, run_s=3600, err="AgentTimeoutError")   # a timeout is an agent failure
    _trial(arm, "b", k=2, reward=1.0, run_s=300)
    for k in (1, 2):
        _trial(arm, "c", k=k, reward=1.0, run_s=300)
    rows = sh.summarize([base, arm])
    full = sh.paired(rows, "baseline", n_boot=500)["jev"]
    cc = sh.paired(rows, "baseline", n_boot=500, complete_case=True)["jev"]
    assert full["view"] == "full set" and full["tasks"] == 3 and full["n_errored_tasks"] == 1
    assert full["success_arm"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)      # the errored attempt counts as a failure
    assert cc["view"] == "complete case" and cc["tasks"] == 2 and cc["success_arm"] == pytest.approx(0.75)
    assert cc["dropped"] == [{"task": "a", "errored_in": ["jev"], "error_class": {"jev": ["DockerBuildError: image "
                                                                                          "build failed"]},
                              "attempts": {"jev": 2}, "errored_attempts": {"jev": 1},
                              "other_outcome": {"baseline": 1.0}}]
    assert "dropped" not in full


def test_summarize_cli_prints_both_views_and_the_dropped_tasks(tmp_path):
    base, arm = tmp_path / "baseline", tmp_path / "jev"
    _trial(base, "a", reward=1.0, run_s=600)
    _trial(base, "b", reward=1.0, run_s=600)
    _trial(arm, "a", reward=None, err="DockerBuildError", msg="image build failed")
    _trial(arm, "b", reward=1.0, run_s=300)
    out = tmp_path / "s.json"
    r = subprocess.run([sys.executable, str(PI / "summarize_harbor.py"), str(base), str(arm), "--baseline", "baseline",
                        "--json", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "| jev | full set |" in r.stdout and "| jev | complete case |" in r.stdout
    assert "complete case: dropped tasks (A6.22 (e) 5)" in r.stdout and "DockerBuildError: image build failed" in r.stdout
    data = json.loads(out.read_text())
    assert data["paired"]["jev"]["tasks"] == 2 and data["paired_complete_case"]["jev"]["tasks"] == 1
