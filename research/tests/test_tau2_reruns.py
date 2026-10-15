"""PREREGISTRATION A6.22 (e) 4 (A6.20 (f)): an errored tau^2 simulation -- termination agent_error / user_error /
infrastructure_error, or a JevTau2Agent exception (side-car ``error``) -- is re-run (same task, same trial, hence same
seed) up to twice before the analysis; the attempts are recorded in the save directory's re-run ledger, stamped by
the converter (``meta.tau2.attempts``), and a task with any trial still errored leaves the pair with all its trials.
Offline: synthetic tau2 result files and a scripted runner; one end-to-end test with the real tau2 runner and a
stubbed LLM (skipped without tau2-bench)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from jevharness.bench.suites import tau2 as jt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import primaries as P  # noqa: E402

UNITS = [("1", 0), ("1", 1), ("2", 0), ("2", 1)]
SEED = {0: 300, 1: 301}


def _sim(task: str, trial: int, n: int, term: str = "user_stop", reward: float = 1.0) -> dict[str, Any]:
    return {"id": f"sim-{task}-{trial}-{n}", "task_id": task, "trial": trial, "seed": SEED[trial],
            "termination_reason": term, "reward_info": {"reward": reward}, "messages": [], "duration": 1.0}


def _side(task: str, trial: int, error: str | None = None) -> dict[str, Any]:
    return {"episode": f"ep-{task}-{trial}", "task_id": task, "seed": SEED[trial], "config": "c", "steps": [],
            "usage": {}, "calls": [], "error": error}


class Scripted:
    """A stand-in for `suites.tau2 run --auto-resume`: runs exactly the (task, trial) units missing from results.json
    (as tau2 does), with scripted outcomes per execution round: {unit: [(termination, side-car error), ...]}."""

    def __init__(self, save: Path, script: dict[tuple[str, int], list[tuple[str, str | None]]]) -> None:
        self.save, self.script, self.calls, self.n = save, script, [], 0

    def execute(self) -> None:
        self.n += 1
        meta = json.loads((self.save / "results.json").read_text())
        have = {(s["task_id"], s["trial"]) for s in meta["simulations"]}
        for u in UNITS:
            if u in have:
                continue
            outcomes = self.script.get(u, [])
            k = sum(1 for c in self.calls for x in c["ran"] if x == u)
            term, err = outcomes[k] if k < len(outcomes) else ("user_stop", None)
            meta["simulations"].append(_sim(*u, n=self.n, term=term, reward=0.0 if term.endswith("error") else 1.0))
            with open(self.save / jt.SIDECAR_FILE, "a") as f:
                f.write(json.dumps(_side(*u, error=err)) + "\n")
            self.calls[-1]["ran"].append(u)
        (self.save / "results.json").write_text(json.dumps(meta))

    def __call__(self, argv: list[str]) -> int:
        self.calls.append({"argv": list(argv), "ran": []})
        self.execute()
        return 0


def _first_run(save: Path, script: dict) -> Scripted:
    save.mkdir(parents=True)
    (save / "results.json").write_text(json.dumps({"info": {"environment_info": {"domain_name": "retail"},
                                                            "agent_info": {"llm": "m"}}, "simulations": []}))
    r = Scripted(save, script)
    r.calls.append({"argv": ["first"], "ran": []})
    r.execute()
    return r


SCRIPT = {("1", 1): [("infrastructure_error", None), ("user_stop", None)],             # re-run once, then fine
          ("2", 0): [("user_stop", "ValueError: boom"), ("agent_error", None), ("user_stop", None)],  # twice
          ("2", 1): [("agent_error", None), ("user_error", None), ("agent_error", None)]}  # still errored after 3


def test_errored_simulations_cover_the_a6_22_error_definition(tmp_path):
    save = tmp_path / "raw"
    _first_run(save, SCRIPT)
    errs = {e["key"]: e for e in jt.errored_simulations(save)}
    assert set(errs) == {"1#1", "2#0", "2#1"}                                  # 1#0 ran cleanly
    assert errs["2#0"]["termination_reason"] == "user_stop" and errs["2#0"]["error"] == "ValueError: boom"
    assert errs["1#1"]["termination_reason"] == "infrastructure_error" and errs["1#1"]["attempts"] == 1


def test_rerun_errored_reruns_each_errored_simulation_up_to_twice(tmp_path):
    save = tmp_path / "raw"
    runner = _first_run(save, SCRIPT)
    res = jt.rerun_errored(save, ["--auto-resume", "--save-to", str(save)], runner=runner)
    assert [r["pruned"] for r in res["rounds"]] == [["1#1", "2#0", "2#1"], ["2#0", "2#1"]]
    assert [c["ran"] for c in runner.calls[1:]] == [[("1", 1), ("2", 0), ("2", 1)], [("2", 0), ("2", 1)]]
    assert all(c["argv"] == ["--auto-resume", "--save-to", str(save)] for c in runner.calls[1:])
    assert res["still_errored"] == [{"key": "2#1", "termination_reason": "agent_error", "error": None, "attempts": 3}]
    assert res["retries_left"] == 0
    led = jt.load_attempts(save)
    assert {k: len(v["pruned"]) for k, v in led["sims"].items()} == {"1#1": 1, "2#0": 2, "2#1": 2}
    assert len(list((save / jt.ERRORED_DIR).glob("*.json"))) == 5                 # every pruned execution is kept
    sims = json.loads((save / "results.json").read_text())["simulations"]
    assert sorted((s["task_id"], s["trial"]) for s in sims) == sorted(UNITS)       # one record per unit, same seeds
    assert all(s["seed"] == SEED[s["trial"]] for s in sims)
    # a second invocation never re-runs a simulation that used its three executions
    again = jt.rerun_errored(save, ["--auto-resume"], runner=runner)
    assert again["rounds"] == [{"pruned": []}] and len(runner.calls) == 3

    # the converter stamps the attempts; the analysis drops task 2 (a trial still errored) with all its trials
    trajs = {t.task_id: t for t in jt.convert_results(save / "results.json", arm="c7")}
    assert {k: t.meta["tau2"]["attempts"] for k, t in trajs.items()} == \
        {"retail/1#0": 1, "retail/1#1": 2, "retail/2#0": 3, "retail/2#1": 3}
    assert trajs["retail/2#1"].error and not trajs["retail/2#0"].error and not trajs["retail/1#1"].error
    assert [e["termination_reason"] for e in trajs["retail/2#1"].meta["tau2"]["errored_attempts"]] == \
        ["agent_error", "user_error"]
    eps = {e["episode"]: e for e in (P._episode(t.to_dict(), 1) for t in trajs.values())}
    tab = P.task_table({"episodes": eps}, ["retail/1", "retail/2"], trials=2, retry_rule=True)
    assert tab["retail/1"]["status"] == "ok" and tab["retail/2"]["status"] == "errored"
    assert tab["retail/2"]["attempts"] == 3 and tab["retail/2"]["errored_trials"] == [1]


def test_prune_handles_tau2s_directory_format(tmp_path):
    save = tmp_path / "raw"
    (save / "simulations").mkdir(parents=True)
    sims = [_sim("1", 0, 1), _sim("1", 1, 1, term="user_error")]
    for s in sims:
        (save / "simulations" / f"{s['id']}.json").write_text(json.dumps(s))
    index = [{"id": s["id"], "task_id": s["task_id"], "trial": s["trial"]} for s in sims]
    (save / "results.json").write_text(json.dumps({"info": {}, "simulation_index": index}))
    out = jt.prune_errored(save)
    assert [e["key"] for e in out["pruned"]] == ["1#1"] and out["final"] == []
    assert sorted(p.name for p in (save / "simulations").glob("*.json")) == ["sim-1-0-1.json"]
    assert [e["id"] for e in json.loads((save / "results.json").read_text())["simulation_index"]] == ["sim-1-0-1"]
    assert (save / jt.ERRORED_DIR / "sim-1-1-1.json").exists()
    assert jt.load_attempts(save)["sims"]["1#1"]["pruned"][0]["termination_reason"] == "user_error"


def test_no_error_still_writes_the_ledger_so_attempts_are_logged(tmp_path):
    save = tmp_path / "raw"
    runner = _first_run(save, {})
    res = jt.rerun_errored(save, ["--auto-resume"], runner=runner)
    assert res["rounds"] == [{"pruned": []}] and len(runner.calls) == 1 and (save / jt.ATTEMPTS_FILE).exists()
    assert all(t.meta["tau2"]["attempts"] == 1 for t in jt.convert_results(save))
    # without a ledger (runs made before A6.22) the converter stamps nothing: primaries treats errors as final
    (save / jt.ATTEMPTS_FILE).unlink()
    assert all("attempts" not in t.meta["tau2"] for t in jt.convert_results(save))


def test_rerun_cli_validates_its_arguments(tmp_path, capsys):
    save = tmp_path / "raw"
    _first_run(save, SCRIPT)
    with pytest.raises(SystemExit, match="after `--`"):
        jt.main(["rerun-errored", str(save)])
    with pytest.raises(SystemExit, match="--auto-resume"):
        jt.main(["rerun-errored", str(save), "--", "--domain", "retail"])
    with pytest.raises(SystemExit, match="no tau2 results"):
        jt.main(["rerun-errored", str(tmp_path / "nope"), "--", "--auto-resume"])
    jt.main(["rerun-errored", str(save), "--dry-run"])
    assert {e["key"] for e in json.loads(capsys.readouterr().out)} == {"1#1", "2#0", "2#1"}
    assert not (save / jt.ATTEMPTS_FILE).exists()                                  # a dry run prunes nothing


def test_run_tau2_sh_reruns_between_run_and_convert_with_the_same_arguments():
    env = {**os.environ, "DRY_RUN": "1", "DOMAINS": "retail", "ARMS": "llm_only"}
    r = subprocess.run(["bash", str(REPO / "scripts" / "run_tau2.sh")], env=env, capture_output=True, text=True,
                       timeout=60, cwd=REPO)
    assert r.returncode == 0, r.stderr
    cmds = [ln[2:] for ln in r.stderr.splitlines() if ln.startswith("+ ")]
    run, rerun, conv = cmds[:3]
    assert " run " in run and " rerun-errored " in rerun and " convert " in conv
    save = run.split("--save-to ")[1].split()[0]
    assert rerun.split(" rerun-errored ")[1].startswith(f"{save} --max-attempts 3 -- ")
    assert rerun.split(" -- ", 1)[1] == run.split(".tau2 run ", 1)[1]             # identical `run` arguments


def test_end_to_end_with_the_real_tau2_runner(tmp_path, monkeypatch):
    """An LLM outage (every agent call fails until tau2's own 4 attempts are spent) leaves trial 0 with an
    infrastructure error; ``rerun-errored`` re-runs it with the same seed, the converter records 2 attempts."""
    pytest.importorskip("tau2.registry", reason="tau2-bench not installed (see docs/TAU2.md)")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    sys.path.insert(0, str(REPO / "tests"))
    import tau2.utils.llm_utils as tau2_llm_utils
    import test_tau2_adapter as TA

    class Outage(TA.FakeLLM):
        def __init__(self, fails: int) -> None:
            super().__init__()
            self.fails = fails

        def _agent(self, model, messages, tools, tool_choice):  # noqa: ANN001
            if self.fails > 0:
                self.fails -= 1
                raise RuntimeError("503 upstream unavailable")
            return super()._agent(model, messages, tools, tool_choice)

    fake = Outage(4)
    monkeypatch.setattr(tau2_llm_utils, "completion", fake)
    for k in (jt.ENV_CONFIG, jt.ENV_OVERRIDES, jt.ENV_LOG_DIR):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    save = tmp_path / "raw"
    args = ["--jev-config", str(REPO / "configs" / "tau2" / "jev_ts_cascade.yaml"), "--jev-set", "s1={type: oracle}",
            "--domain", "mock", "--task-ids", "create_task_1", "--agent-llm", TA.AGENT_LLM, "--user-llm",
            TA.USER_LLM, "--num-trials", "2", "--max-concurrency", "1", "--save-to", str(save), "--auto-resume"]
    jt.main(["run", *args])
    first = {s["trial"]: s for s in json.loads((save / "results.json").read_text())["simulations"]}
    assert first[0]["termination_reason"] == "infrastructure_error" and first[1]["termination_reason"] == "user_stop"

    def runner(argv: list[str]) -> int:
        fake.user_turns = 0
        jt.main(["run", *argv])
        return 0

    res = jt.rerun_errored(save, args, runner=runner)
    assert res["rounds"][0]["pruned"] == ["create_task_1#0"] and res["still_errored"] == []
    again = {s["trial"]: s for s in json.loads((save / "results.json").read_text())["simulations"]}
    assert again[0]["seed"] == first[0]["seed"] and again[0]["termination_reason"] == "user_stop"
    trajs = {t.task_id: t for t in jt.convert_results(save / "results.json")}
    assert trajs["mock/create_task_1#0"].meta["tau2"]["attempts"] == 2 and not trajs["mock/create_task_1#0"].error
    assert trajs["mock/create_task_1#1"].meta["tau2"]["attempts"] == 1
