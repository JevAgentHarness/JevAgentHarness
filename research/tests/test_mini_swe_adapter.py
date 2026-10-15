"""Offline tests for the mini-swe-agent adapter (``jevharness.adapters.mini_swe``).

mini-swe-agent's ``DeterministicModel`` / ``DeterministicToolcallModel`` replay scripted LM turns, the
``LocalEnvironment`` runs the bash commands inside ``tmp_path``, and the ``oracle`` / ``simulated``
System-1 backends answer from gold labels supplied by a labeler (rule-based or test-specific).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")  # no mini-swe-agent banner in test output

import pytest
import yaml

pytest.importorskip("minisweagent")

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.test_models import (  # noqa: E402
    DeterministicModel,
    DeterministicToolcallModel,
    make_output,
    make_toolcall_output,
)

from jevharness.adapters.mini_swe import (  # noqa: E402
    DEFAULT_CONFIG,
    JevMiniSweAgent,
    JevMiniSweConfig,
    classify_command,
    collect_run,
    fix_verified,
    is_destructive,
    load_resolved_ids,
    main,
    process_instance,
    to_trajectory,
)
from jevharness.bench.metrics import decision_records, decision_table, episode_table  # noqa: E402
from jevharness.decision.base import DecisionBackend  # noqa: E402
from jevharness.harness.trajectory import Trajectory  # noqa: E402

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo PATCH"
AGENT_KW = dict(system_template="You fix bugs in the repository.", instance_template="Issue: {{task}}",
                step_limit=30, cost_limit=0)  # DeterministicModel charges $1/call: disable the $3 default cap


def _model(turns, name="deterministic", cost=1.0):
    return DeterministicModel(outputs=[make_output(f"THOUGHT: {t}", [{"command": c}], cost=cost) for t, c in turns],
                              model_name=name)


def _agent(model, cwd, jev, **kw):
    return JevMiniSweAgent(model, LocalEnvironment(cwd=str(cwd)), jev=jev, **{**AGENT_KW, **kw})


def _decisions(agent, dp):
    return [d for s in agent.jev_steps for d in s.decisions if d.dp == dp]


def _text(msg):
    c = msg.get("content")
    return c if isinstance(c, str) else json.dumps(c)


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "repo"
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test_core.py").write_text("def test_ok():\n    assert True\n")
    (d / "core.py").write_text("x = 0\n")
    return d


# ------------------------------------------------------------------ 1. action guard
def test_guard_blocks_destructive_command(repo):
    model = _model([("delete the tests", "rm -rf tests"), ("look around", "ls"), ("done", SUBMIT)])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "gold": "heuristic",
                                 "action_guard": {"mode": "guard", "tau": 0.5}})
    info = agent.run("make x equal 1")
    assert (repo / "tests" / "test_core.py").exists(), "destructive command was executed"
    assert info["exit_status"] == "Submitted" and info["submission"].strip() == "PATCH"
    g = _decisions(agent, "tool_select")
    assert [d.mode for d in g] == ["guard", "guard"]
    assert g[0].s1_value is False and g[0].final_value is None and g[0].gold is False
    assert g[1].s1_value is True and g[1].final_value == "ls"
    blocked_obs = [m for m in agent.messages if "was NOT executed" in _text(m)]
    assert len(blocked_obs) == 1 and blocked_obs[0]["extra"]["jev_blocked"] == "tool_select"
    assert agent.jev_counters["blocked"] == 1
    assert agent.ledger.summary()["s1"]["calls"] == 2  # submit command is not guarded (stop hook's job)


def test_guard_override_after_reissue_and_shadow_mode(repo):
    # the same command re-issued after a block runs (guard max_retries=1 -> the LLM may insist once)
    model = _model([("delete", "rm -rf tests"), ("I insist", "rm -rf tests"), ("done", SUBMIT)])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "gold": "heuristic",
                                 "action_guard": {"mode": "guard", "tau": 0.5}})
    agent.run("task")
    assert not (repo / "tests").exists()
    g = _decisions(agent, "tool_select")
    assert g[0].final_value is None and g[1].deferred and g[1].final_value == "rm -rf tests"
    assert "re-issue it unchanged" in _text([m for m in agent.messages if "NOT executed" in _text(m)][0])

    # shadow: S1 is asked and logged, the command still runs
    (repo / "tests").mkdir()
    model = _model([("delete", "rm -rf tests"), ("done", SUBMIT)])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "gold": "heuristic",
                                 "action_guard": {"mode": "shadow", "tau": 0.5}})
    agent.run("task")
    assert not (repo / "tests").exists()
    (d,) = _decisions(agent, "tool_select")
    assert d.mode == "shadow" and d.s1_value is False and d.final_value == "rm -rf tests"


def test_guard_parallel_toolcalls_batched_and_tool_ids_kept(repo):
    calls = [{"id": "call_rm", "type": "function",
              "function": {"name": "bash", "arguments": json.dumps({"command": "rm -rf tests"})}},
             {"id": "call_ls", "type": "function",
              "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})}}]
    sub = [{"id": "call_sub", "type": "function", "function": {"name": "bash",
                                                                "arguments": json.dumps({"command": SUBMIT})}}]
    model = DeterministicToolcallModel(outputs=[
        make_toolcall_output("two commands", calls, [{"command": "rm -rf tests", "tool_call_id": "call_rm"},
                                                     {"command": "ls", "tool_call_id": "call_ls"}]),
        make_toolcall_output("submit", sub, [{"command": SUBMIT, "tool_call_id": "call_sub"}])])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "gold": "heuristic",
                                 "action_guard": {"mode": "guard", "tau": 0.5, "extra": {"kind": "choice"}}})
    info = agent.run("task")
    assert info["exit_status"] == "Submitted" and (repo / "tests").exists()
    tool_msgs = {m["tool_call_id"]: m for m in agent.messages if m.get("role") == "tool"}
    assert "NOT executed" in tool_msgs["call_rm"]["content"]
    assert "core.py" in tool_msgs["call_ls"]["content"]
    s1 = [c for c in agent.ledger.to_list() if c["component"] == "s1"]
    assert len(s1) == 1 and s1[0]["meta"]["n_questions"] == 2  # one batched S1 call for both commands
    g = _decisions(agent, "tool_select")
    assert [d.s1_value for d in g] == [False, True] and set(g[0].s1_dist) == {"on_track", "unproductive",
                                                                              "destructive"}


# ------------------------------------------------------------------ 2. stop guard
def test_stop_guard_nudges_once(repo):
    model = _model([("fix", "echo 'x = 1' > core.py"), ("submit", SUBMIT), ("submit again", SUBMIT)])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "gold": "heuristic",
                                 "stop": {"mode": "guard", "tau": 0.5, "extra": {"max_nudges": 1}}})
    info = agent.run("make x equal 1")
    assert info["exit_status"] == "Submitted" and info["submission"].strip() == "PATCH"
    nudges = [m for m in agent.messages if "Submission deferred" in _text(m)]
    assert len(nudges) == 1
    stop = _decisions(agent, "stop")
    assert len(stop) == 2
    assert stop[0].s1_value is False and stop[0].final_value is False and stop[0].gold is False
    assert stop[1].deferred and stop[1].final_value is True  # nudge budget exhausted -> accepted, no S1 call
    assert agent.ledger.summary()["s1"]["calls"] == 1
    assert agent.n_calls == 3 and agent.jev_counters["nudges"] == 1


def test_stop_guard_accepts_verified_fix(repo):
    model = _model([("fix", "echo 'x = 1' > core.py"), ("verify", "python3 core.py"), ("submit", SUBMIT)])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "gold": "heuristic", "stop": {"mode": "guard"}})
    info = agent.run("task")
    assert info["exit_status"] == "Submitted" and agent.n_calls == 3
    (d,) = _decisions(agent, "stop")
    assert d.s1_value is True and d.gold is True and d.final_value is True


def test_stop_trigger_hints_submission(repo):
    model = _model([("fix", "echo 'x = 1' > core.py"), ("verify", "python3 core.py"), ("submit", SUBMIT)])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "gold": "heuristic",
                                 "stop": {"mode": "trigger", "tau": 0.5, "extra": {"min_step": 0}}})
    agent.run("task")
    hints = [m for m in agent.messages if (m.get("extra") or {}).get("jev_hints") == ["stop_trigger"]]
    assert len(hints) == 1 and hints[0]["extra"]["step"] == 2  # only after the edit has been run
    assert [d.final_value for d in _decisions(agent, "stop")] == [False, True]


# ------------------------------------------------------------------ 3. routing
def test_routing_switches_models(repo):
    strong = _model([("plan", "echo a"), ("finish", SUBMIT)], name="strong-model", cost=1.0)
    cheap = _model([("routine", "echo b")], name="cheap-model", cost=0.1)

    def labeler(hook, ctx):  # "hard" on even steps
        return ctx["step"] % 2 == 0 if hook == "routing" else None

    agent = JevMiniSweAgent(strong, LocalEnvironment(cwd=str(repo)), cheap_model=cheap, labeler=labeler,
                            jev={"s1": {"type": "oracle"}, "routing": {"mode": "trigger", "tau": 0.5}}, **AGENT_KW)
    info = agent.run("task")
    assert info["exit_status"] == "Submitted"
    assert [s.model for s in agent.jev_steps] == ["strong-model", "cheap-model", "strong-model"]
    assert [d.final_value for d in _decisions(agent, "routing")] == ["large", "small", "large"]
    roles = [m["extra"]["jev_model"]["role"] for m in agent.messages if m.get("role") == "assistant"]
    assert roles == ["strong", "cheap", "strong"]
    assert agent.model_stats["strong"]["calls"] == 2 and agent.model_stats["cheap"]["calls"] == 1
    assert agent.cost == pytest.approx(2.1)
    assert agent.model is strong  # restored after every routed query
    traj = to_trajectory(agent, instance_id="demo__demo-1")
    assert traj.usage["s2"]["cost_usd"] == pytest.approx(2.1) and traj.usage["s2"]["calls"] == 3
    assert traj.meta["cheap_llm"] == "cheap-model" and traj.llm == "strong-model"


def test_routing_trigger_requires_cheap_model(repo):
    with pytest.raises(ValueError, match="cheap model"):
        _agent(_model([("x", "ls")]), repo, {"s1": {"type": "oracle"}, "routing": {"mode": "trigger"}})
    with pytest.raises(ValueError, match="no backend"):
        _agent(_model([("x", "ls")]), repo, {"stop": {"mode": "guard"}})


# ------------------------------------------------------------------ 4. reflection
def test_reflection_trigger_after_failed_command(repo):
    model = _model([("read", "cat missing_file.txt"), ("retry", "cat core.py"), ("next", "ls"), ("done", SUBMIT)])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "gold": "heuristic",
                                 "reflection": {"mode": "trigger", "tau": 0.5, "extra": {"cooldown": 0}}})
    agent.run("task")
    refl = [m for m in agent.messages if "[Reflection]" in _text(m)]
    assert len(refl) == 1 and "last command failed" in _text(refl[0])
    (d,) = _decisions(agent, "reflection")  # only asked after the failure event
    assert d.s1_value is True and d.final_value is True and d.gold is True
    assert agent.jev_steps[1].reflection is not None


def test_reflection_always_needs_no_s1(repo):
    model = _model([("x", "echo a"), ("again", "echo a"), ("done", SUBMIT)])
    agent = _agent(model, repo, {"reflection": {"mode": "always"}})
    assert agent.s1 is None
    agent.run("task")
    assert any("repeating a command" in _text(m) for m in agent.messages)
    assert agent.ledger.summary().get("s1") is None


# ------------------------------------------------------------------ 5. action-type hint
def test_action_type_assist_hint(repo):
    model = _model([("search", "grep -rn 'x =' ."), ("done", SUBMIT)])

    def labeler(hook, ctx):
        return "search" if hook == "action_type" else None

    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "action_type": {"mode": "assist", "k": 2}},
                   labeler=labeler)
    agent.run("task")
    hints = [m for m in agent.messages if (m.get("extra") or {}).get("jev_hints") == ["action_type"]]
    assert len(hints) == 2 and "search (p=1.00" in _text(hints[0]) and "explore" not in _text(hints[0])
    a = _decisions(agent, "action")
    assert a[0].s1_value == "search" and a[0].s2_value == "search" and a[1].s2_value == "submit"
    table = decision_table(decision_records([to_trajectory(agent, instance_id="i").to_dict()]))
    assert table["action/assist"]["s1_acc"] == 1.0


# ------------------------------------------------------------------ all hooks, simulated S1
def test_all_hooks_with_simulated_backend(repo):
    cmds = [("look", "ls"), ("read", "cat core.py"), ("oops", "cat nope.txt"), ("fix", "echo 'x = 1' > core.py"),
            ("delete", "rm -rf tests"), ("run", "python3 core.py")] + [("submit", SUBMIT)] * 8
    strong, cheap = _model(cmds, name="strong"), _model(cmds, name="cheap", cost=0.1)
    jev = {"name": "kitchen_sink", "s1": {"type": "simulated", "accuracy": 0.8, "seed": 1}, "gold": "heuristic",
           "action_guard": {"mode": "guard", "tau": 0.5}, "stop": {"mode": "guard", "tau": 0.5},
           "routing": {"mode": "trigger", "tau": 0.5}, "reflection": {"mode": "trigger", "tau": 0.5},
           "action_type": {"mode": "assist", "k": 2}}
    agent = JevMiniSweAgent(strong, LocalEnvironment(cwd=str(repo)), jev=jev, cheap_model=cheap, **AGENT_KW)
    info = agent.run("task")
    assert info["exit_status"] == "Submitted", info
    dps = {d.dp for s in agent.jev_steps for d in s.decisions}
    assert {"tool_select", "stop", "routing", "reflection", "action"} <= dps
    s1_calls = [c for c in agent.ledger.to_list() if c["component"] == "s1"]
    assert any("+" in (c["decision_point"] or "") for c in s1_calls)  # batched pre-query questions
    assert all(c["latency_ms"] == 100.0 for c in s1_calls)
    traj = to_trajectory(agent, instance_id="demo__demo-2")
    assert traj.config == "kitchen_sink" and traj.backend == "simulated" and traj.error is None


# ------------------------------------------------------------------ baseline + robustness
def test_baseline_matches_default_agent(repo):
    turns = [("a", "echo one"), ("b", "cat core.py"), ("done", SUBMIT)]
    ref = DefaultAgent(_model(turns), LocalEnvironment(cwd=str(repo)), **AGENT_KW)
    ref_info = ref.run("task")
    agent = _agent(_model(turns), repo, None)
    info = agent.run("task")
    assert info == {**ref_info} and len(agent.messages) == len(ref.messages)
    assert [m["role"] for m in agent.messages] == [m["role"] for m in ref.messages]
    assert all(not s.decisions for s in agent.jev_steps) and "s1" not in agent.ledger.summary()


class _Broken(DecisionBackend):
    name = "broken"

    def _decide(self, state, questions):
        raise RuntimeError("S1 endpoint down")


def test_s1_failure_is_fail_open(repo):
    model = _model([("delete", "rm -rf tests"), ("done", SUBMIT)])
    agent = _agent(model, repo, {"action_guard": {"mode": "guard"}, "stop": {"mode": "guard"}}, s1=_Broken())
    info = agent.run("task")
    assert info["exit_status"] == "Submitted" and not (repo / "tests").exists()
    assert agent.jev_counters["s1_errors"] == 2
    assert {d.mode for s in agent.jev_steps for d in s.decisions} == {"error"}


def test_limits_exceeded_keeps_logs_consistent(repo):
    model = _model([("a", "echo 1"), ("b", "echo 2"), ("c", "echo 3")])
    agent = _agent(model, repo, {"s1": {"type": "oracle"}, "reflection": {"mode": "shadow"}}, step_limit=2)
    info = agent.run("task")
    assert info["exit_status"] == "LimitsExceeded"
    assert len(agent.jev_steps) == 2 and all(s.llm_calls == 1 for s in agent.jev_steps)


def test_shadow_arm_logs_everything_but_behaves_like_baseline(repo):
    turns = [("fix", "echo 'x = 1' > core.py"), ("delete", "rm -rf tests"), ("oops", "cat nope.txt"),
             ("run", "python3 core.py"), ("done", SUBMIT)]
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())["jev"]
    cfg.update(s1={"type": "oracle"}, s1_cache=None, gold="heuristic")
    jcfg = JevMiniSweConfig.for_arm(cfg, "shadow")
    jcfg.stop.extra["min_step"] = 0
    agent = _agent(_model(turns), repo, jcfg)
    info = agent.run("task")
    ref = DefaultAgent(_model(turns), LocalEnvironment(cwd=str(repo)), **AGENT_KW)
    (repo / "tests").mkdir()
    assert info == ref.run("task") and len(agent.messages) == len(ref.messages)  # no intervention at all
    modes = {f"{d.dp}/{d.mode}" for s in agent.jev_steps for d in s.decisions}
    assert {"tool_select/shadow", "routing/shadow", "reflection/shadow", "action/shadow",
            "stop/shadow_trigger", "stop/shadow_guard"} <= modes
    traj = to_trajectory(agent, instance_id="i", resolved=True, stop_gold_from_eval=True)
    last_stop = [d for s in traj.steps for d in s.decisions if d.mode == "shadow_guard"][-1]
    assert last_stop.gold is True  # heuristic label: edit followed by a run


# ------------------------------------------------------------------ records, conversion, metrics
def test_decisions_logged_and_trajectory_roundtrip(repo, tmp_path):
    model = _model([("delete", "rm -rf tests"), ("fix", "echo 'x = 1' > core.py"), ("submit", SUBMIT),
                    ("run", "python3 core.py"), ("submit", SUBMIT)])
    agent = _agent(model, repo, {"name": "guards", "s1": {"type": "oracle"}, "gold": "heuristic",
                                 "action_guard": {"mode": "guard"},
                                 "stop": {"mode": "guard", "extra": {"max_nudges": 2}}})
    agent.run("task")
    assert [d.s1_value for d in _decisions(agent, "stop")] == [False, True]  # unverified, then verified
    rec = agent.save_jev(tmp_path / "demo.jev.json", instance_id="demo__demo-3")
    assert rec["format"] == "jev-mini-swe-1" and rec["exit_status"] == "Submitted"
    assert rec["usage"]["s1"]["calls"] == 5 and rec["usage"]["s2"]["calls"] == 5
    assert rec["model_stats"]["instance_cost"] == pytest.approx(5.0)
    traj = to_trajectory(tmp_path / "demo.jev.json", resolved=True, stop_gold_from_eval=True)
    assert isinstance(traj, Trajectory) and traj.success and traj.final_answer.strip() == "PATCH"
    assert traj.config == "guards" and traj.backend == "oracle" and traj.meta["gold_source"] == "heuristic"
    d = traj.to_dict()
    json.dumps(d)  # JSON-serialisable
    ep = episode_table([d])
    assert ep["success"] == 1.0 and ep["s1_calls"] == 5 and ep["llm_calls"] == 5
    assert ep["cost_usd"] == pytest.approx(5.0)  # oracle S1 is free; 5 LM calls x $1
    table = decision_table(decision_records([d]))
    assert table["tool_select/guard"]["s1_acc"] == 1.0 and table["tool_select/guard"]["n_gold"] == 3
    assert table["stop/guard"]["n"] == 2 and table["stop/guard"]["s1_acc"] == 1.0
    unresolved = to_trajectory(rec, resolved=False)
    assert not unresolved.success and unresolved.result["resolved"] is False
    assert to_trajectory(rec).result["resolved"] is None


def test_process_instance_and_collect_offline(tmp_path):
    work = tmp_path / "testbed"
    work.mkdir()
    (work / "core.py").write_text("x = 0\n")
    turns = [make_output("fix", [{"command": "echo 'x = 1' > core.py"}]),
             make_output("run", [{"command": "python3 core.py"}]),
             make_output("submit", [{"command": SUBMIT}])]
    config = {"model": {"model_class": "deterministic", "model_name": "det-strong", "outputs": turns},
              "agent": AGENT_KW,
              "jev": {"s1": {"type": "oracle"}, "gold": "heuristic",
                      "cheap_model": {"model_name": "det-cheap", "outputs": turns},
                      "arms": {"guarded": {"stop": {"mode": "guard"}, "routing": {"mode": "trigger"}}}}}
    jcfg = JevMiniSweConfig.for_arm(config["jev"], "guarded")
    from jevharness.decision import make_backend
    out = tmp_path / "run"
    out.mkdir()
    inst = {"instance_id": "demo__demo-4", "problem_statement": "x should be 1"}
    rec = process_instance(inst, out, config, jcfg, make_backend(jcfg.s1),
                           env_factory=lambda cfg, i: LocalEnvironment(cwd=str(work)),
                           labeler=None)
    assert rec["exit_status"] == "Submitted" and rec["models"] == {"strong": "det-strong", "cheap": "det-cheap"}
    preds = json.loads((out / "preds.json").read_text())
    assert preds["demo__demo-4"]["model_patch"].strip() == "PATCH"
    assert preds["demo__demo-4"]["model_name_or_path"] == "det-strong__jev-guarded"
    mini_traj = json.loads((out / "demo__demo-4" / "demo__demo-4.traj.json").read_text())
    assert mini_traj["info"]["exit_status"] == "Submitted" and mini_traj["info"]["jev"]["arm"] == "guarded"
    assert (out / "trajectories.jsonl").read_text().count("\n") == 1

    # env start-up failure still leaves a record
    def boom(cfg, i):
        raise RuntimeError("docker not available")
    rec2 = process_instance({"instance_id": "demo__demo-5", "problem_statement": "p"}, out, config, jcfg, None,
                            env_factory=boom)
    assert rec2["exit_status"] == "RuntimeError" and "docker" in rec2["error"]

    report = tmp_path / "det.jev_guarded.json"  # SWE-bench harness summary format
    report.write_text(json.dumps({"resolved_ids": ["demo__demo-4"], "unresolved_ids": []}))
    per_inst = tmp_path / "report.json"  # per-instance harness format
    per_inst.write_text(json.dumps({"demo__demo-5": {"resolved": False}}))
    assert load_resolved_ids([report, per_inst]) == {"demo__demo-4"}
    trajs = collect_run(out, [report], out=tmp_path / "guarded.jsonl")
    assert [t["task_id"] for t in trajs] == ["demo__demo-4", "demo__demo-5"]
    assert [t["result"]["success"] for t in trajs] == [True, False]
    assert episode_table(trajs)["success"] == 0.5
    assert len((tmp_path / "guarded.jsonl").read_text().splitlines()) == 2


# ------------------------------------------------------------------ config + heuristics
def test_yaml_config_arms_are_valid(capsys):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert {"agent", "environment", "model", "jev"} <= set(cfg)
    for arm in cfg["jev"]["arms"]:
        c = JevMiniSweConfig.for_arm(cfg["jev"], arm)
        assert c.name == arm
    assert JevMiniSweConfig.for_arm(cfg["jev"], "baseline").uses_s1() is False
    assert JevMiniSweConfig.for_arm(cfg["jev"], "full").routing.mode == "trigger"
    assert JevMiniSweConfig(stop={"mode": False}).stop.mode == "off"  # YAML `off` -> False
    assert JevMiniSweConfig(stop={"mode": "log"}).stop.mode == "shadow"  # alias used by adapters.hooks
    with pytest.raises(ValueError, match="not supported"):
        JevMiniSweConfig(routing={"mode": "guard"}).validate_modes()
    with pytest.raises(KeyError):
        JevMiniSweConfig.for_arm(cfg["jev"], "nope")
    main(["arms"])
    assert "stop_guard" in capsys.readouterr().out.split()


def test_heuristics():
    for c in ["rm -rf /", "rm -rf .", "cd /testbed && rm -rf *", "rm -r tests", "rm tests/test_a.py",
              "git reset --hard HEAD", "git checkout -- .", "git clean -fdx", "pip uninstall -y numpy",
              "rm -rf .git", "sed -i 's/a/b/' tests/test_x.py"]:
        assert is_destructive(c), c
    for c in ["rm reproduce.py", "rm -rf build/", "ls -la", "grep -rn foo .", "git diff > patch.txt",
              "sed -i 's/a/b/' src/core.py", "python reproduce.py", "rm -f /tmp/x.txt", SUBMIT]:
        assert not is_destructive(c), c
    assert classify_command(SUBMIT) == "submit"
    assert classify_command("git diff -- core.py > patch.txt") == "submit"
    assert classify_command("sed -i 's/0/1/' core.py") == "edit"
    assert classify_command("cat > reproduce.py <<'EOF'\nprint(1)\nEOF") == "reproduce"
    assert classify_command("python reproduce.py") == "reproduce"
    assert classify_command("python -m pytest tests/test_core.py -q") == "test"
    assert classify_command("grep -rn 'def foo' src") == "search"
    assert classify_command("sed -n '1,40p' core.py") == "explore"
    h = lambda cmds: [{"step": i, "command": c, "returncode": 0, "output": "", "blocked": False}  # noqa: E731
                      for i, c in enumerate(cmds)]
    assert not fix_verified(h(["ls", "cat core.py"]))
    assert not fix_verified(h(["python reproduce.py", "sed -i 's/0/1/' core.py"]))
    assert fix_verified(h(["sed -i 's/0/1/' core.py", "python -m pytest -q"]))
    assert not fix_verified(h(["sed -i 's/0/1/' core.py", "git diff -- core.py > patch.txt", "cat patch.txt"]))
