"""Terminal-Bench-Science integration (adapters/pi: tbsci.py, tbsci_tasks.sh, run_tbsci.sh, summarize_harbor.py).

Offline: a synthetic TB-science checkout in tmp_path, no Docker, no Harbor, no API keys.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PI = REPO / "adapters" / "pi"
sys.path.insert(0, str(PI))
try:
    import summarize_harbor as sh
    import tbsci
finally:
    sys.path.remove(str(PI))

UBUNTU_DF = """# canary
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip curl
RUN python3 -c "import os; \\
from os import path; \\
print(path.sep)"
RUN pip3 install --break-system-packages numpy==1.26.4
"""
DEBIAN_DF = """FROM python:3.11-slim AS base
RUN apt-get update && apt-get install -y gcc
FROM base
RUN pip install numpy
"""
HF_DF = """FROM python:3.12-slim
RUN pip install huggingface_hub && python -c "from huggingface_hub import snapshot_download; \\
snapshot_download(repo_id='harborframework/terminal-bench-science-lfs', repo_type='dataset')"
"""


def _toml(slug: str, *, expert: float = 8.0, gpus: int = 0, cpus: int = 2, verifier: float = 300.0) -> str:
    return f"""schema_version = "1.0"
artifacts = ["/app/out.txt"]
[task]
name = "terminal-bench-science/{slug}"
[metadata]
domain = "physical sciences"
field = "physics"
expert_time_estimate_hours = {expert}
[verifier]
timeout_sec = {verifier}
environment_mode = "separate"
[verifier.environment]
network_mode = "no-network"
[agent]
timeout_sec = 28800.0
[environment]
build_timeout_sec = 600.0
cpus = {cpus}
memory_mb = 4096
storage_mb = 10240
gpus = {gpus}
network_mode = "public"
"""


def _task(root: Path, rel: str, slug: str, env_df: str, test_df: str = UBUNTU_DF, **kw) -> Path:
    d = root / rel / slug
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir(parents=True)
    (d / "task.toml").write_text(_toml(slug, **kw))
    (d / "instruction.md").write_text("Do the science. You have 28800 seconds to complete this task.\n")
    (d / "environment" / "Dockerfile").write_text(env_df)
    (d / "tests" / "Dockerfile").write_text(test_df)
    (d / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return d


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    root = tmp_path / "tbsci"
    _task(root, "tasks/physical-sciences/physics", "ubuntu-task", UBUNTU_DF, expert=12)
    _task(root, "tasks/physical-sciences/physics", "debian-task", DEBIAN_DF, DEBIAN_DF, expert=6)
    _task(root, "tasks/life-sciences/biology", "hf-task", HF_DF, expert=4)
    _task(root, "tasks/life-sciences/biology", "gpu-task", UBUNTU_DF, gpus=1)
    _task(root, "tasks/earth-sciences/geosciences", "epic-task", UBUNTU_DF, expert=80)
    _task(root, "tasks/earth-sciences/geosciences", "unlisted-task", UBUNTU_DF)
    _task(root, "lite", "easy-task", UBUNTU_DF, expert=2)
    listed = ["ubuntu-task", "debian-task", "hf-task", "gpu-task", "epic-task"]
    (root / "tasks" / "dataset.toml").write_text(
        '[dataset]\nname = "terminal-bench-science/terminal-bench-science"\n\n'
        + "".join(f'[[tasks]]\nname = "terminal-bench-science/{s}"\ndigest = "sha256:00"\n\n' for s in listed))
    return root


# ------------------------------------------------------------------ Dockerfile parsing / CA patch
def test_python_from_import_inside_run_is_not_a_from_instruction():
    froms = [ins for _, _, ins in tbsci.instructions(UBUNTU_DF + HF_DF) if re.match(r"FROM\s", ins, re.I)]
    assert froms == ["FROM ubuntu:24.04", "FROM python:3.12-slim"]
    heredoc = "FROM ubuntu:24.04\nRUN <<EOF\nfrom os import path\nEOF\nFROM alpine AS x\n"
    assert [i for _, _, i in tbsci.instructions(heredoc) if i.startswith("FROM")] == ["FROM ubuntu:24.04",
                                                                                     "FROM alpine AS x"]


def test_stages_resolve_aliases_and_detect_debian_apt_and_curl():
    st = tbsci.dockerfile_stages(DEBIAN_DF)
    assert [s["image"] for s in st] == ["python:3.11-slim", "python:3.11-slim"]
    assert [s["family"] for s in st] == ["debian", "debian"] and st[0]["apt"] and not st[1]["apt"]
    (u,) = tbsci.dockerfile_stages(UBUNTU_DF)
    assert u["family"] == "ubuntu" and u["apt"] and u["curl"]
    assert tbsci.image_family("rocker/r-ver:4.3.0") == "ubuntu"
    assert tbsci.image_family("node:22.14.0-bookworm-slim@sha256:ab") == "debian"
    assert tbsci.image_family("ghcr.io/astral-sh/uv:0.11.1") == "other"


def test_patch_inserts_ca_block_after_every_from_and_nothing_else():
    new, n = tbsci.patch_dockerfile_text(DEBIAN_DF + HF_DF)
    assert n == 3
    lines = new.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("FROM"):
            assert lines[i + 1].startswith(tbsci.CA_MARK) and lines[i + 2].startswith("COPY harbor-sandbox-ca.crt")
            assert "PIP_CERT=/usr/local/share/harbor-sandbox-ca.crt" in lines[i + 3]
    block_free = re.sub(re.escape(tbsci.CA_MARK) + r".*\nCOPY .*\nENV .*\n", "", new)
    assert block_free == DEBIAN_DF + HF_DF  # everything else byte-identical
    # a continued FROM line gets the block after its last line; a missing final newline is tolerated
    new, n = tbsci.patch_dockerfile_text("FROM --platform=linux/amd64 \\\n  ubuntu:24.04\nRUN true")
    assert n == 1 and new.splitlines()[2].startswith(tbsci.CA_MARK) and new.endswith("RUN true")


def test_patch_ca_copies_bundle_into_every_build_context_and_is_idempotent(clone, tmp_path):
    task = clone / "tasks/physical-sciences/physics/ubuntu-task"
    sub = task / "environment" / "sidecar"
    sub.mkdir()
    (sub / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN true\n")
    (task / "tests" / ".dockerignore").write_text("__pycache__/\n")
    ca = tmp_path / "ca.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\nX\n-----END CERTIFICATE-----\n")
    instr = (task / "instruction.md").read_text()
    test_sh = (task / "tests" / "test.sh").read_text()
    assert tbsci.patch_ca(task, ca) == 3
    for d in (task / "environment", task / "tests", sub):
        assert (d / tbsci.CA_NAME).read_text() == ca.read_text()
    assert (task / "tests" / ".dockerignore").read_text().endswith(f"!{tbsci.CA_NAME}\n")
    assert (task / "instruction.md").read_text() == instr and (task / "tests" / "test.sh").read_text() == test_sh
    before = (task / "environment" / "Dockerfile").read_text()
    assert tbsci.patch_ca(task, ca) == 0 and (task / "environment" / "Dockerfile").read_text() == before
    assert (task / "tests" / ".dockerignore").read_text().count(f"!{tbsci.CA_NAME}") == 1


# ------------------------------------------------------------------ survey / subsets
def test_survey_rows_and_subset_rules(clone):
    rows = {r["slug"]: r for r in tbsci.survey(clone)}
    assert set(rows) == {"ubuntu-task", "debian-task", "hf-task", "gpu-task", "epic-task", "unlisted-task",
                         "easy-task"}
    u = rows["ubuntu-task"]
    assert (u["domain"], u["field"], u["set"], u["in_manifest"]) == ("physical-sciences", "physics", "main", True)
    assert u["agent_timeout_s"] == 28800.0 and u["verifier_mode"] == "separate" and u["n_artifacts"] == 1
    assert u["verifier_network"] == "no-network" and u["agent_network"] == "public"
    assert u["hard_pool"] and u["sandbox_ok"]
    assert rows["debian-task"]["debian_apt"] and not rows["debian-task"]["sandbox_ok"]
    assert rows["debian-task"]["hard_pool"]  # open egress has Debian mirrors: still a pool task
    assert rows["hf-task"]["hf_data"] and not rows["hf-task"]["sandbox_ok"] and rows["hf-task"]["hard_pool"]
    assert not rows["gpu-task"]["hard_pool"] and not rows["gpu-task"]["sandbox_ok"]
    assert not rows["epic-task"]["hard_pool"]  # 80 expert hours: outside the 4-24 h band
    assert not rows["unlisted-task"]["in_manifest"] and not rows["unlisted-task"]["hard_pool"]
    e = rows["easy-task"]
    assert e["set"] == "lite" and e["domain"] == "physical-sciences" and not e["hard_pool"]
    all_rows = tbsci.survey(clone)
    # path order; the manifest's [dataset] name is not a task, unlisted-task and lite tasks are not in "all"
    assert tbsci.select(all_rows, "all") == ["epic-task", "gpu-task", "hf-task", "debian-task", "ubuntu-task"]
    assert tbsci.select(all_rows, "hard_pool") == ["hf-task", "debian-task", "ubuntu-task"]
    assert tbsci.select(all_rows, "sandbox") == ["epic-task", "ubuntu-task"]
    with pytest.raises(ValueError):
        tbsci.select(all_rows, "nope")


def test_survey_cli_list_and_json(clone, tmp_path):
    out = tmp_path / "rows.json"
    r = subprocess.run([sys.executable, str(PI / "tbsci.py"), "survey", str(clone), "--json", str(out)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "| ubuntu-task |" in r.stdout and "hard_pool 3" in r.stderr
    assert len(json.loads(out.read_text())) == 7
    r = subprocess.run([sys.executable, str(PI / "tbsci.py"), "list", str(clone), "sandbox"], capture_output=True,
                       text=True, timeout=60)
    assert r.stdout.split() == ["epic-task", "ubuntu-task"]
    r = subprocess.run([sys.executable, str(PI / "tbsci.py"), "manifest", str(clone)], capture_output=True,
                       text=True, timeout=60)
    assert r.stdout.split() == ["ubuntu-task", "debian-task", "hf-task", "gpu-task", "epic-task"]


def _subset_slugs(path: Path) -> list[str]:
    return [ln.split("#")[0].strip() for ln in path.read_text().splitlines() if ln.split("#")[0].strip()]


def test_committed_subset_lists_are_well_formed():
    pool, sandbox = _subset_slugs(PI / "tbsci_hard_pool.txt"), _subset_slugs(PI / "tbsci_sandbox.txt")
    for slugs in (pool, sandbox):
        assert slugs and len(slugs) == len(set(slugs))
        assert all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", s) for s in slugs)


@pytest.mark.skipif(not (Path(os.environ.get("TBSCI_CLONE", "<local-path>"))
                         / "tasks" / "dataset.toml").exists(), reason="no terminal-bench-science checkout")
def test_committed_subset_lists_match_the_rules_on_the_pinned_checkout():
    clone = Path(os.environ.get("TBSCI_CLONE", "<local-path>"))
    rows = tbsci.survey(clone)
    assert len(tbsci.select(rows, "all")) == 70
    assert _subset_slugs(PI / "tbsci_hard_pool.txt") == tbsci.select(rows, "hard_pool")
    assert _subset_slugs(PI / "tbsci_sandbox.txt") == tbsci.select(rows, "sandbox")


# ------------------------------------------------------------------ tbsci_tasks.sh
def _run_tasks_sh(clone: Path, out: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k != "PATCH_CA"}
    e.update({"TBSCI_CLONE": str(clone), **env})
    return subprocess.run(["bash", str(PI / "tbsci_tasks.sh"), str(out), *args], env=e, capture_output=True,
                          text=True, timeout=60)


def test_tbsci_tasks_flattens_the_manifest_and_patches_only_on_request(clone, tmp_path):
    out = tmp_path / "flat"
    r = _run_tasks_sh(clone, out)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.iterdir()) == ["debian-task", "epic-task", "gpu-task", "hf-task",
                                                     "ubuntu-task"]  # @all = dataset.toml, not unlisted-task
    src = clone / "tasks/physical-sciences/physics/ubuntu-task"
    for f in ("task.toml", "instruction.md", "environment/Dockerfile", "tests/Dockerfile", "tests/test.sh"):
        assert (out / "ubuntu-task" / f).read_bytes() == (src / f).read_bytes()  # plain copies by default
    ca = tmp_path / "ca.crt"
    ca.write_text("CA\n")
    r = _run_tasks_sh(clone, out, "ubuntu-task", "easy-task", PATCH_CA=str(ca))
    assert r.returncode == 0, r.stderr
    df = (out / "ubuntu-task" / "environment" / "Dockerfile").read_text()
    assert tbsci.CA_MARK in df and (out / "ubuntu-task" / "tests" / tbsci.CA_NAME).read_text() == "CA\n"
    assert (out / "easy-task" / "task.toml").exists()  # lite tasks resolve by slug too
    assert (out / "ubuntu-task" / "instruction.md").read_bytes() == (src / "instruction.md").read_bytes()


def test_tbsci_tasks_rejects_unknown_tasks_and_subsets(clone, tmp_path):
    r = _run_tasks_sh(clone, tmp_path / "o", "no-such-task")
    assert r.returncode == 1 and "0 matches" in r.stderr
    r = _run_tasks_sh(clone, tmp_path / "o", "@no_such_subset")
    assert r.returncode == 1 and "unknown subset" in r.stderr
    r = _run_tasks_sh(clone, tmp_path / "o", "ubuntu-task", PATCH_CA=str(tmp_path / "missing.crt"))
    assert r.returncode == 1 and "no such file" in r.stderr


# ------------------------------------------------------------------ run_tbsci.sh / run_tb2.sh
def _dry(script: str, *arms: str, **env: str) -> list[str]:
    e = {k: v for k, v in os.environ.items() if k not in (
        "DATASET", "TASKS_DIR", "INCLUDE", "SUBSET", "AGENT_TIMEOUT_MULT", "MAX_TURNS", "EXTRA_CA",
        "EXTRA_CA_VERIFIER", "OUT", "N_CONCURRENT", "HARBOR_ARGS")}
    e.update({"DRY_RUN": "1", "HARBOR": "harbor", **env})
    r = subprocess.run(["bash", str(PI / script), *arms], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return [ln for ln in r.stderr.splitlines() if ln.startswith("+ ")]


def test_run_tbsci_defaults_hub_dataset_capped_budget_and_no_verifier_ca():
    (base,) = _dry("run_tbsci.sh", "baseline", EXTRA_CA="/ca.crt")
    assert "-d terminal-bench-science/terminal-bench-science@0.1.0" in base and " -p " not in base
    assert "-n 2 " in base and "--agent-timeout-multiplier 0.125" in base and "runs/tbsci" in base
    assert "--ae NODE_EXTRA_CA_CERTS=/etc/harbor-extra-ca/ca-bundle.crt" in base
    assert "--ve " not in base  # separate no-network verifiers never get the mount
    assert "--ak max_turns=100" in base and " -i " not in base
    (official,) = _dry("run_tbsci.sh", "baseline", AGENT_TIMEOUT_MULT="", MAX_TURNS="")
    assert "--agent-timeout-multiplier" not in official and "max_turns" not in official


def test_run_tbsci_subset_names_tasks_for_hub_and_local_datasets(tmp_path):
    slugs = _subset_slugs(PI / "tbsci_sandbox.txt")
    (hub,) = _dry("run_tbsci.sh", "baseline", SUBSET="sandbox")
    assert re.findall(r"-i (\S+)", hub) == [f"terminal-bench-science/{s}" for s in slugs]
    f = tmp_path / "mine.txt"
    f.write_text("# comment\nalpha  # trailing\n\nbeta\n")
    (loc,) = _dry("run_tbsci.sh", "jev_stop_reflect", SUBSET=str(f), TASKS_DIR=str(tmp_path), MAX_TURNS="30")
    assert re.findall(r"-i (\S+)", loc) == ["alpha", "beta"] and f"-p {tmp_path}" in loc
    assert "--ak max_turns=30" in loc and "--ak jev_extensions=stop-verify,reflect" in loc
    assert "--ae JEV_REFLECT_TAU=0.5" in loc
    r = subprocess.run(["bash", str(PI / "run_tbsci.sh"), "baseline"], capture_output=True, text=True,
                       env={**os.environ, "DRY_RUN": "1", "SUBSET": "no_such_subset"}, timeout=60)
    assert r.returncode == 2 and "unknown SUBSET" in r.stderr


def test_run_tb2_new_arms_and_knobs_keep_tb2_defaults(tmp_path):
    base, refl = _dry("run_tb2.sh", "baseline", "jev_reflect", EXTRA_CA="/ca.crt", TASKS_DIR=str(tmp_path))
    assert "--ve SSL_CERT_FILE=/etc/harbor-extra-ca/ca-bundle.crt" in base  # TB2 default unchanged
    assert "--agent-timeout-multiplier" not in base and "max_turns" not in base
    assert "--ak jev_extensions=reflect " in refl + " " and "JEV_BASE_URL=" in refl
    (nove,) = _dry("run_tb2.sh", "baseline", EXTRA_CA="/ca.crt", EXTRA_CA_VERIFIER="0", AGENT_TIMEOUT_MULT="0.5")
    assert "--ve " not in nove and "--agent-timeout-multiplier 0.5" in nove


def test_oracle_and_nop_arms_use_harbor_agents_without_pi_flags(tmp_path):
    oracle, nop, base = _dry("run_tbsci.sh", "oracle", "nop", "baseline", MAX_TURNS="40", SUBSET="sandbox")
    for line, agent in ((oracle, "oracle"), (nop, "nop")):
        assert f"run -a {agent} -o " in line and f"--job-name {agent}" in line
        assert "harbor_jev_pi" not in line and "--ak" not in line and " -m " not in line
        assert "-i terminal-bench-science/" in line  # same task filter as the pi arms
    assert "-a harbor_jev_pi:JevPi" in base and "--ak max_turns=40" in base


def test_relative_paths_are_made_absolute_from_the_callers_directory(tmp_path):
    e = {k: v for k, v in os.environ.items() if k not in ("TASKS_DIR", "OUT", "EXTRA_CA", "ENV_FILE")}
    e.update(DRY_RUN="1", HARBOR="harbor", OUT="jobs", TASKS_DIR="./tasks", EXTRA_CA="ca.crt", ENV_FILE="none.env")
    r = subprocess.run(["bash", str(PI / "run_tb2.sh"), "baseline"], cwd=tmp_path, env=e, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stderr.splitlines() if ln.startswith("+ "))
    assert f"-o {tmp_path}/jobs " in line and f"-p {tmp_path}/tasks " in line
    assert f'"source":"{tmp_path}/ca.crt"' in line
    assert "--env-file" not in line  # missing env file is skipped, as before


# ------------------------------------------------------------------ summarize_harbor.py
def _trial(job: Path, name: str, task: str, reward: float | None, run_s: float, cost: float,
           steps: int = 3, err: str | None = None, ctrf: tuple[int, int] | None = None,
           decisions: list[dict] | None = None) -> None:
    t = job / name
    (t / "agent").mkdir(parents=True)
    (t / "verifier").mkdir()
    res = {"task_name": task, "agent_result": {"cost_usd": cost, "n_input_tokens": 100},
           "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
           "exception_info": {"exception_type": err} if err else None,
           "started_at": "2026-10-01T01:00:00Z", "finished_at": f"2026-10-01T01:{10 + int(run_s // 60):02d}:00Z",
           "agent_execution": {"started_at": "2026-10-01T01:00:00Z",
                               "finished_at": f"2026-10-01T01:{int(run_s // 60):02d}:{int(run_s % 60):02d}Z"}}
    (t / "result.json").write_text(json.dumps(res))
    steps_l = [{"source": "user"}] + [{"source": "agent", "llm_call_count": 1, "tool_calls": [{"x": 1}]}] * steps
    (t / "agent" / "trajectory.json").write_text(json.dumps({"steps": steps_l}))
    if ctrf:
        (t / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"summary": {"tests": ctrf[1],
                                                                                       "passed": ctrf[0]}}}))
    if decisions:
        (t / "agent" / "jev-decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n")


def test_summarize_rows_have_steps_tests_trial_time_and_reflections(tmp_path):
    job = tmp_path / "jev_stop_reflect"
    decisions = [{"request_id": "a", "model": "jev", "dp": "reflection", "final_value": True, "latency_ms": 300},
                 {"request_id": "b", "model": "jev", "dp": "reflection", "final_value": False, "latency_ms": 500},
                 {"request_id": "c", "model": "jev", "dp": "stop", "mode": "guard", "final_value": "continue"},
                 {"request_id": "d", "model": "jev", "dp": "stop", "mode": "trigger", "final_value": "continue"},
                 {"request_id": None, "model": None, "dp": "stop", "final_value": "stop:nudge-limit"}]
    _trial(job, "x__1", "terminal-bench-science/x", 0.0, 900, 0.3, steps=5, err="AgentTimeoutError", ctrf=(2, 4),
           decisions=decisions)
    (row,) = sh.summarize([job])
    assert row["task"] == "x" and row["timed_out"] and row["reward"] == 0.0
    assert row["agent_run_s"] == 900 and row["trial_s"] == 1500 and row["agent_steps"] == 5 and row["tool_calls"] == 5
    assert (row["tests_passed"], row["tests_total"]) == (2, 4)
    assert row["reflections"] == 1 and row["stop_nudges"] == 1 and row["s1_s_total"] == 0.8


def test_arm_stats_and_paired_comparison(tmp_path):
    base, arm = tmp_path / "baseline", tmp_path / "jev"
    # task a: both solve, arm faster; task b: only the arm solves; task c: both fail (timeout); d: only baseline ran
    _trial(base, "a__1", "terminal-bench-science/a", 1.0, 600, 0.20)
    _trial(base, "b__1", "terminal-bench-science/b", 0.0, 900, 0.50, err="AgentTimeoutError")
    _trial(base, "c__1", "terminal-bench-science/c", 0.0, 900, 0.40, err="AgentTimeoutError")
    _trial(base, "d__1", "terminal-bench-science/d", 1.0, 120, 0.05)
    _trial(arm, "a__1", "a", 1.0, 300, 0.10)  # local run: task named by slug only
    _trial(arm, "b__1", "b", 1.0, 480, 0.30)
    _trial(arm, "c__1", "c", 0.0, 900, 0.45, err="AgentTimeoutError")
    rows = sh.summarize([base, arm])
    st = sh.arm_stats(rows, budgets=[300, 600])
    assert st["baseline"]["success"] == 0.5 and st["baseline"]["timeouts"] == 2
    assert st["baseline"]["agent_run_p50_s"] == 750 and st["jev"]["agent_run_p50_s"] == 480
    assert st["jev"]["agent_run_p90_s"] == pytest.approx(816)
    assert st["jev"]["solved_by_300s"] == pytest.approx(1 / 3) and st["jev"]["solved_by_600s"] == pytest.approx(2 / 3)
    assert st["jev"]["cost_per_solved"] == pytest.approx(0.85 / 2)
    p = sh.paired(rows, "baseline")["jev"]
    assert p["tasks"] == 3 and p["wins"] == 1 and p["losses"] == 0 and p["sign_test_p"] == 1.0
    assert p["success_base"] == pytest.approx(1 / 3) and p["success_arm"] == pytest.approx(2 / 3)
    assert p["d_agent_run_s_median"] == -300 and p["faster_tasks"] == 2
    assert p["n_both_solved"] == 1 and p["d_agent_run_s_median_both_solved"] == -300
    assert p["d_cost_median"] == pytest.approx(-0.1)
    with pytest.raises(ValueError):
        sh.paired(rows, "nope")


def test_sign_test_and_quantile_helpers():
    assert sh.sign_test_p(0, 0) is None and sh.sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert sh.sign_test_p(3, 3) == 1.0
    assert sh.quantile([], 0.5) is None and sh.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert sh.quantile([10, 20, 30, 40, 50], 0.9) == pytest.approx(46)


def test_summarize_cli_prints_arm_and_paired_tables(tmp_path):
    _trial(tmp_path / "baseline", "a__1", "a", 1.0, 600, 0.2)
    _trial(tmp_path / "jev", "a__1", "a", 1.0, 300, 0.1)
    out = tmp_path / "s.json"
    r = subprocess.run([sys.executable, str(PI / "summarize_harbor.py"), str(tmp_path / "baseline"),
                        str(tmp_path / "jev"), "--baseline", "baseline", "--budgets", "450", "--json", str(out)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "paired vs baseline" in r.stdout and "solved_by_450s" in r.stdout
    data = json.loads(out.read_text())
    assert set(data) == {"rows", "arms", "paired", "paired_complete_case"}       # + the A6.22 (e) 5 complete case
    assert data["paired"]["jev"]["d_agent_run_s_median"] == -300


def test_paired_h5_verdict_columns(tmp_path):
    """A5.2: 'better' (LB90 of d_success > -5 pt and wins >= losses) and 'faster' (median d time on tasks both solve
    at least once < 0, Wilcoxon p < 0.05) are computed from per-task attempt means; solved tasks are timed by their
    solved attempts only."""
    base, arm = tmp_path / "baseline", tmp_path / "jev"
    for i in range(6):                                             # 6 tasks x 2 attempts
        for j in (1, 2):
            _trial(base, f"t{i}__{j}", f"terminal-bench-science/t{i}", 1.0 if i < 4 else 0.0, 600 if i < 4 else 900,
                   0.2, err=None if i < 4 else "AgentTimeoutError")
            solved = i < 5 and not (i == 4 and j == 2)             # t4: solved once, once timed out
            _trial(arm, f"t{i}__{j}", f"t{i}", 1.0 if solved else 0.0, 300 if solved else 900, 0.1,
                   err=None if solved else "AgentTimeoutError")
    rows = sh.summarize([base, arm])
    pt = sh.per_task(rows)["jev"]
    assert pt["t4"]["n_solved"] == 1 and pt["t4"]["agent_run_s_solved"] == 300 and pt["t4"]["agent_run_s"] == 600
    assert pt["t5"]["agent_run_s_solved"] is None and pt["t0"]["agent_run_s_solved"] == 300
    p = sh.paired(rows, "baseline", n_boot=2000)["jev"]
    assert set(sh.H5_KEYS) <= set(p) and p["tasks"] == 6
    assert (p["wins"], p["losses"]) == (1, 0) and p["n_both_solved"] == 4           # strict: all attempts solved
    assert p["n_h5_both_solved"] == 4 and p["d_solved_run_s_median"] == -300 and p["wilcoxon_p"] == pytest.approx(1 / 16)
    assert p["d_success_lb90"] > -0.05 and p["sign_p_wins"] == 0.5
    assert p["h5_better"] and not p["h5_faster"] and p["h5_verdict"] == "success-preserving"   # n = 4: min one-sided sign p = 1/16
    assert p["time_sign_p"] == pytest.approx(1 / 16) and not p["h5_strictly_better"]
    # two more solved tasks in both arms make the Wilcoxon reachable (n = 6: p = 1/64)
    for i in (6, 7):
        for j in (1, 2):
            _trial(base, f"t{i}__{j}", f"t{i}", 1.0, 600, 0.2)
            _trial(arm, f"t{i}__{j}", f"t{i}", 1.0, 300, 0.1)
    p = sh.paired(sh.summarize([base, arm]), "baseline", n_boot=2000)["jev"]
    assert p["n_h5_both_solved"] == 6 and p["wilcoxon_p"] == pytest.approx(1 / 64) and p["time_sign_p"] == pytest.approx(1 / 64)
    # the registered verdict, displayed with the A6.22 (d) 1 label; 1 win / 0 losses: the strict tier needs sign p < 0.05
    assert p["h5_verdict"] == "passes the exploratory success screen; faster among jointly solved tasks"


def test_summarize_cli_prints_h5_verdict(tmp_path):
    _trial(tmp_path / "baseline", "a__1", "a", 1.0, 600, 0.2)
    _trial(tmp_path / "jev", "a__1", "a", 1.0, 300, 0.1)
    r = subprocess.run([sys.executable, str(PI / "summarize_harbor.py"), str(tmp_path / "baseline"),
                        str(tmp_path / "jev"), "--baseline", "baseline"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "H5 verdict" in r.stdout and "h5_verdict" in r.stdout and "| success-preserving |" in r.stdout
