"""Regression tests for the integration bugs found by the L3 smoke runs (docs/HARNESS_INTEGRATION.md, "L3 smoke").

All offline: no API keys, no Docker. Each test names the smoke finding it pins down.

* tau2 retail: tau2 hard-codes its NL-assertion judge to gpt-4.1 -> ``--jev-nl-judge`` / ``set_nl_judge``.
* tau2 gold: a READ reference action the agent bypassed (retail: auth by e-mail instead of name+zip) stayed
  the "next step" for the rest of the conversation -> ``_Reference.pending`` drops moot READs.
* mini-swe: ``run_swebench.sh`` did not read .env, could not add config overlays (Gemini agent / stand-in S1),
  and could not evaluate against an offline dataset copy (huggingface.co blocked).
* Terminal-Bench: the TB2 verifiers install uv from astral.sh -> ``tb2_sandbox_tasks.sh`` (sandbox-only patch).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # litellm: no remote price-map fetch


# ------------------------------------------------------------------ tau2
@pytest.fixture
def tau2_mod():
    pytest.importorskip("tau2.registry", reason="tau2-bench not installed (see docs/TAU2.md)")
    from jevharness.bench.suites import tau2 as jt

    return jt


@pytest.fixture
def restore_nl_judge(tau2_mod):
    from tau2.evaluator import evaluator_nl_assertions as nl

    saved = (nl.DEFAULT_LLM_NL_ASSERTIONS, dict(nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS))
    yield nl
    nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = saved


def test_set_nl_judge_patches_tau2_evaluator(tau2_mod, restore_nl_judge):
    nl = restore_nl_judge
    assert tau2_mod.nl_judge()["model"] == nl.DEFAULT_LLM_NL_ASSERTIONS  # read-only call
    model, args = tau2_mod.set_nl_judge("gemini/gemini-3.5-flash")
    assert model == nl.DEFAULT_LLM_NL_ASSERTIONS == "gemini/gemini-3.5-flash"
    # tau2 json.loads the raw judge text: ask for a JSON response; minimal thinking on Gemini 3.x
    assert args["response_format"] == {"type": "json_object"} and args["temperature"] == 0.0
    assert args["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS == args
    _, args = tau2_mod.set_nl_judge("openai/judge-x", {"temperature": 0.0})
    assert args == {"temperature": 0.0}  # explicit args win, no Gemini thinking config for others


def test_nl_judge_reaches_the_evaluator_call(tau2_mod, restore_nl_judge, monkeypatch):
    nl = restore_nl_judge
    from tau2.data_model.message import AssistantMessage, UserMessage

    seen = {}

    def fake_generate(model, messages, **kw):
        seen.update(model=model, **kw)
        return AssistantMessage(role="assistant", content=json.dumps({"results": [
            {"expectedOutcome": "says hi", "reasoning": "it did", "metExpectation": True}]}))

    monkeypatch.setattr(nl, "generate", fake_generate)
    tau2_mod.set_nl_judge("gemini/gemini-3.5-flash")
    checks = nl.NLAssertionsEvaluator.evaluate_nl_assertions(
        [UserMessage(role="user", content="hi"), AssistantMessage(role="assistant", content="hi!")], ["says hi"])
    assert checks[0].met and seen["model"] == "gemini/gemini-3.5-flash"
    assert seen["response_format"] == {"type": "json_object"}


def test_run_launcher_sets_nl_judge_and_sidecar_records_it(tau2_mod, restore_nl_judge, monkeypatch, tmp_path):
    nl = restore_nl_judge
    import tau2.cli

    for k in (tau2_mod.ENV_CONFIG, tau2_mod.ENV_OVERRIDES, tau2_mod.ENV_LOG_DIR):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    called = {}
    monkeypatch.setattr(tau2.cli, "main", lambda: called.setdefault("argv", list(sys.argv)))
    tau2_mod.main(["run", "--jev-config", str(REPO / "configs/tau2/llm_only.yaml"),
                   "--jev-nl-judge", "gemini/gemini-3.5-flash", "--jev-nl-judge-args", '{"temperature": 0}',
                   "--domain", "mock", "--save-to", str(tmp_path / "out")])
    assert nl.DEFAULT_LLM_NL_ASSERTIONS == "gemini/gemini-3.5-flash"
    assert nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS == {"temperature": 0}
    assert "--jev-nl-judge" not in " ".join(called["argv"])  # our flags never reach tau2's CLI
    assert tau2_mod.nl_judge() == {"model": "gemini/gemini-3.5-flash", "args": {"temperature": 0}}


def test_experiment_yaml_passes_nl_judge_to_every_arm(tau2_mod):
    exp = {"name": "x", "out": "/tmp/x", "domain": "retail", "agent_llm": "a", "user_llm": "u",
           "nl_judge_llm": "gemini/gemini-3.5-flash", "nl_judge_llm_args": {"temperature": 0},
           "arms": [{"name": "llm_only", "config": str(REPO / "configs/tau2/llm_only.yaml")},
                    {"name": "jev_guard", "config": str(REPO / "configs/tau2/jev_guard.yaml")}]}
    for _, cmd, _ in tau2_mod.experiment_commands(exp):
        i = cmd.index("--jev-nl-judge")
        assert cmd[i + 1] == "gemini/gemini-3.5-flash"
        assert json.loads(cmd[cmd.index("--jev-nl-judge-args") + 1]) == {"temperature": 0}


def test_run_tau2_sh_forwards_nl_judge(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "DOMAINS": "retail", "ARMS": "llm_only", "OUT": str(tmp_path),
           "NL_JUDGE_LLM": "gemini/gemini-3.5-flash", "NL_JUDGE_LLM_ARGS": '{"temperature": 0}'}
    r = subprocess.run(["bash", str(REPO / "scripts/run_tau2.sh")], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    run_line = next(ln for ln in r.stderr.splitlines() if " run " in ln)
    assert "--jev-nl-judge gemini/gemini-3.5-flash" in run_line and "--jev-nl-judge-args" in run_line


def test_gold_drops_read_actions_the_agent_bypassed(tau2_mod):
    """Retail task 1: the reference authenticates with find_user_id_by_name_zip; the agent used the e-mail.
    After it moved on (get_order_details = the reference's next action), the name+zip lookup is moot. Before
    the fix it stayed the gold next step for the rest of the conversation (its arguments appear in tool output)."""
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.data_model.message import ToolCall as T2Call
    from tau2.runner import build_environment, get_tasks

    env = build_environment("retail")
    task = [t for t in get_tasks("retail") if t.id == "1"][0]
    agent = tau2_mod.JevTau2Agent(env.get_tools(), env.get_policy(), llm="gpt-4.1-mini", task=task)
    assert agent.kinds["find_user_id_by_name_zip"] == "read"
    st = agent.get_init_state([AssistantMessage(role="assistant", content="Hi! How can I help you today?")])
    st.messages.append(UserMessage(role="user", content="I'm Yusuf Rossi, zip 19122; exchange order #W2378156."))
    assert agent._view(st).gold_tool == "find_user_id_by_name_zip"  # reference step 1 is takeable now
    st.messages += [
        AssistantMessage(role="assistant", content=None, tool_calls=[
            T2Call(id="c1", name="find_user_id_by_email", arguments={"email": "yusuf.rossi7301@example.com"})]),
        ToolMessage(id="c1", role="tool", content="yusuf_rossi_9620"),
        AssistantMessage(role="assistant", content=None, tool_calls=[
            T2Call(id="c2", name="get_order_details", arguments={"order_id": "#W2378156"})]),
        ToolMessage(id="c2", role="tool", content='{"order_id": "#W2378156", "user_id": "yusuf_rossi_9620", '
                                                  '"address": {"zip": "19122"}, "name": "Yusuf Rossi"}')]
    v = agent._view(st)
    assert "find_user_id_by_name_zip" not in [a.name for a in v.pending]
    assert v.gold_tool != "find_user_id_by_name_zip"
    # WRITE actions are never dropped as moot, even when a later action was executed
    assert "exchange_delivered_order_items" in [a.name for a in v.pending]


# ------------------------------------------------------------------ mini-swe launcher
def _stub(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_run_swebench_overlay_envfile_and_offline_dataset(tmp_path):
    log = tmp_path / "calls.log"
    py = _stub(tmp_path / "py", f'echo "PY $* SMOKE_KEY=${{SMOKE_KEY:-unset}}" >> {log}\n'
                                'case "$*" in *" collect "*|*"collect "*) ;; esac\n')
    evpy = _stub(tmp_path / "evpy", f'echo "EVAL $*" >> {log}\n'  # writes a report like swebench 4.1.0
                                    'while [ $# -gt 0 ]; do [ "$1" = --report_dir ] && echo {{}} > "$2/m.jev_baseline.json"; '
                                    'shift; done\n')
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "test.jsonl").write_text('{"instance_id": "a__b-1"}\n')
    ids = tmp_path / "ids.txt"
    ids.write_text("a__b-1\n")
    out = tmp_path / "out"
    (out / "baseline").mkdir(parents=True)
    (out / "baseline" / "preds.json").write_text("{}")
    envf = tmp_path / "dot.env"
    envf.write_text("SMOKE_KEY=from_env_file\n")
    env = {**os.environ, "PY": str(py), "EVAL": "local", "EVAL_PY": str(evpy), "ENV_FILE": str(envf),
           "SUBSET": str(data), "IDS": str(ids), "OUT": str(out),
           "CONFIG_OVERLAY": "adapters/mini_swe/gemini.yaml adapters/mini_swe/s1_gemini_standin.yaml"}
    env.pop("SMOKE_KEY", None)
    r = subprocess.run(["bash", str(REPO / "adapters/mini_swe/run_swebench.sh"), "baseline"], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    run = next(c for c in calls if "mini_swe run" in c)
    assert "-c adapters/mini_swe/gemini.yaml -c adapters/mini_swe/s1_gemini_standin.yaml" in run
    assert run.index("swebench_jev.yaml") < run.index("gemini.yaml")  # overlays come after the base config
    assert "SMOKE_KEY=from_env_file" in run  # .env exported to the Python entry point
    ev = next(c for c in calls if c.startswith("EVAL"))
    assert f"--dataset_name {data}/test.jsonl" in ev  # the harness needs the file, not the directory
    collect = next(c for c in calls if "mini_swe collect" in c)
    assert f"--report {out}/baseline/eval/m.jev_baseline.json" in collect
    # no report (eval failed / wrote elsewhere): warn and still collect instead of aborting under pipefail
    (out / "baseline" / "eval" / "m.jev_baseline.json").unlink()
    _stub(evpy, 'true\n')
    r = subprocess.run(["bash", str(REPO / "adapters/mini_swe/run_swebench.sh"), "baseline"], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "no evaluation report found" in r.stderr, r.stderr


def test_gemini_overlays_merge_into_a_valid_arm():
    pytest.importorskip("minisweagent")
    from jevharness.adapters.mini_swe import JevMiniSweConfig, load_config

    cfg = load_config([str(REPO / "adapters/mini_swe/swebench_jev.yaml"), str(REPO / "adapters/mini_swe/gemini.yaml"),
                       str(REPO / "adapters/mini_swe/s1_gemini_standin.yaml")])
    assert cfg["model"]["model_name"] == "gemini/gemini-3.5-flash"
    assert cfg["model"]["model_kwargs"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert cfg["model"]["model_kwargs"]["parallel_tool_calls"] is True  # base kwargs survive the merge
    jcfg = JevMiniSweConfig.for_arm(cfg["jev"], "action_guard")
    assert jcfg.s1["type"] == "llm_generative" and jcfg.s1["llm"]["model"] == "gemini-3.5-flash-lite"
    assert jcfg.s1.get("model") != "jev-latest"  # no misleading leftover from the base `s1:` dict
    assert jcfg.action_guard.mode == "guard"
    # without the stand-in overlay the base Jev backend is untouched
    base = load_config([str(REPO / "adapters/mini_swe/swebench_jev.yaml"), str(REPO / "adapters/mini_swe/gemini.yaml")])
    assert base["jev"]["s1"]["type"] == "jev"


# ------------------------------------------------------------------ Terminal-Bench (Harbor)
def test_tb2_sandbox_tasks_patches_only_the_uv_installer(tmp_path):
    clone = tmp_path / "tb2"
    task = clone / "demo-task"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    test_sh = ("#!/bin/bash\napt-get update\napt-get install -y curl\n\n"
               "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n\nsource $HOME/.local/bin/env\n"
               "uvx -p 3.13 -w pytest==8.4.1 pytest /tests/test_outputs.py -rA\n")
    (task / "tests" / "test.sh").write_text(test_sh)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(clone)], check=True)
    subprocess.run([*git, "-C", str(clone), "add", "."], check=True)
    subprocess.run([*git, "-C", str(clone), "commit", "-qm", "x"], check=True)
    out = tmp_path / "out"
    env = {**os.environ, "TB2_CLONE": str(clone), "TB2_COMMIT": "HEAD"}
    r = subprocess.run(["bash", str(REPO / "adapters/pi/tb2_sandbox_tasks.sh"), str(out), "demo-task"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    patched = (out / "demo-task" / "tests" / "test.sh").read_text()
    assert "astral.sh" not in patched.replace("[sandbox patch: astral.sh blocked]", "")
    assert "https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz" in patched
    before, after = test_sh.split("curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n")
    assert patched.startswith(before) and patched.endswith(after)  # everything else byte-identical
    r = subprocess.run(["bash", "-n", str(out / "demo-task" / "tests" / "test.sh")], capture_output=True)
    assert r.returncode == 0
    env["PATCH_UV"] = "0"
    subprocess.run(["bash", str(REPO / "adapters/pi/tb2_sandbox_tasks.sh"), str(out), "demo-task"], env=env,
                   check=True, capture_output=True)
    assert (out / "demo-task" / "tests" / "test.sh").read_text() == test_sh


def test_harbor_host_gateway_overlay():
    overlay = yaml.safe_load((REPO / "adapters/pi/harbor_host_gateway.yaml").read_text())
    assert "host.docker.internal:host-gateway" in overlay["services"]["main"]["extra_hosts"]


def test_summarize_harbor_reads_trials_and_dedupes_s1_requests(tmp_path):
    sys.path.insert(0, str(REPO / "adapters" / "pi"))
    try:
        import summarize_harbor as sh
    finally:
        sys.path.pop(0)
    trial = tmp_path / "jev_gate_stop" / "demo__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({
        "task_name": "demo", "agent_result": {"cost_usd": 0.02, "n_input_tokens": 1000, "n_output_tokens": 10},
        "verifier_result": {"rewards": {"reward": 1.0}},
        "agent_execution": {"started_at": "2026-10-01T01:00:00Z", "finished_at": "2026-10-01T01:00:30Z"}}))
    recs = [{"request_id": "r1", "model": "s1", "input_tokens": 1000, "latency_ms": 500, "final_value": "allow"},
            {"request_id": "r2", "model": "s1", "input_tokens": 3000, "latency_ms": 700, "final_value": "block"},
            {"request_id": None, "model": None, "final_value": "allow:retry-limit"},
            {"request_id": "r3", "model": "s1", "input_tokens": 2000, "latency_ms": 600, "dp": "stop",
             "final_value": "continue"},
            {"request_id": "r3", "model": "s1", "input_tokens": 2000, "latency_ms": 600, "dp": "stop",
             "final_value": "continue"}]  # batched request: one record per question, same request_id
    (trial / "agent" / "jev-decisions.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (row,) = sh.summarize([tmp_path / "jev_gate_stop"], s1_price_per_m=1.0)
    assert row["reward"] == 1.0 and row["agent_run_s"] == 30.0 and row["llm_cost_usd"] == 0.02
    assert row["s1_requests"] == 3 and row["s1_input_tokens"] == 6000 and row["s1_cost_usd_est"] == 0.006
    assert row["blocks"] == 1 and row["stop_nudges"] == 2 and row["s1_decisions"] == 5


def test_run_tb2_sh_dry_run_builds_arm_flags(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "HARBOR": "harbor", "TASKS_DIR": str(tmp_path), "INCLUDE": "a* b",
           "EXTRA_CA": "/ca.crt", "OUT": str(tmp_path / "jobs")}
    r = subprocess.run(["bash", str(REPO / "adapters/pi/run_tb2.sh"), "baseline", "jev_shadow"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    base, shadow = [ln for ln in r.stderr.splitlines() if ln.startswith("+ ")]
    assert "-a harbor_jev_pi:JevPi" in base and f"-p {tmp_path}" in base and "-i a* -i b" in base
    assert "jev_extensions" not in base and "--job-name baseline" in base
    assert "--ak jev_extensions=tool-gate,stop-verify" in shadow and "JEV_GATE_MODE=log" in shadow
    assert "--ae NODE_EXTRA_CA_CERTS=/etc/harbor-extra-ca/ca-bundle.crt" in base
    assert "--ve SSL_CERT_FILE=/etc/harbor-extra-ca/ca-bundle.crt" in base
    assert "harbor_host_gateway.yaml" in base
    r = subprocess.run(["bash", str(REPO / "adapters/pi/run_tb2.sh"), "nope"], env=env, capture_output=True, text=True)
    assert r.returncode == 2
