"""Tests for the harness-agnostic command-hook adapter (``jevharness.adapters.hooks``).

Payload fixtures follow the exact stdin schemas of
* Codex:  codex-rs/hooks/src/schema.rs (vendored JSON schemas in adapters/codex/schema/, validated below),
* dsh:    its command-hook bridge package (src/index.ts (base(), preToolPayload(), ... ; transcript_path is ''),
* generic: a command-hook reference (session_id, transcript_path, cwd, permission_mode, hook_event_name).
Codex outputs are checked against the vendored output schemas *and* the semantic rules of
codex-rs/hooks/src/engine/output_parser.rs. No test touches the network.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from jevharness.adapters import hooks as H
from jevharness.decision import DecisionBackend, DecisionResult, QKind, Question, Usage
from jevharness.decision.jev import JevBackend
from jevharness.decision.types import answer_from_distribution

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "adapters" / "codex" / "schema"
SCHEMA_FILE = {"PreToolUse": "pre-tool-use", "PermissionRequest": "permission-request", "PostToolUse": "post-tool-use",
               "UserPromptSubmit": "user-prompt-submit", "Stop": "stop", "SessionStart": "session-start"}
LOG_FIELDS = {"ts", "harness", "session", "dp", "mode", "question", "options", "s1_value", "s1_conf", "s1_dist",
              "final_value", "latency_ms", "input_tokens", "model", "error"}

# ------------------------------------------------------------------------------------ payloads
CODEX_BASE = {"session_id": "019a7e1c-5f7e-7c10-a1a2-3b4c5d6e7f80", "turn_id": "1",
              "transcript_path": None, "cwd": "/work/repo", "model": "gpt-5.5-codex", "permission_mode": "default"}


def codex(event: str, transcript: str | None = None, **kw: Any) -> dict[str, Any]:
    p = {**CODEX_BASE, "hook_event_name": event, **kw}
    if transcript is not None:
        p["transcript_path"] = transcript
    if event == "SessionStart":  # SessionStartCommandInput has no turn_id
        p.pop("turn_id")
    return p


CODEX_PRE = dict(tool_name="Bash", tool_input={"command": "rm -rf build/ && make"}, tool_use_id="call_Qm1")
CODEX_PERM = dict(tool_name="Bash", tool_input={"command": "git push --force origin main",
                                                "description": "Push the rebased branch"})
CODEX_POST = dict(tool_name="Bash", tool_input={"command": "pytest -q"}, tool_use_id="call_Qm2",
                  tool_response="FAILED tests/test_math.py::test_add - assert 3 == 4\n1 failed, 3 passed")
CODEX_STOP = dict(stop_hook_active=False, last_assistant_message="I fixed add(); all tests pass.")
CODEX_PROMPT = dict(prompt="Fix the failing test in tests/test_math.py")

DSH_BASE = {"session_id": "session-7d2c", "transcript_path": "", "cwd": "/work/repo"}  # the dsh bridge's base()


def dsh(event: str, **kw: Any) -> dict[str, Any]:
    return {**DSH_BASE, "hook_event_name": event, **kw}


def generic(event: str, transcript: str, **kw: Any) -> dict[str, Any]:
    return {"session_id": "3f1c2b7e-generic", "transcript_path": transcript, "cwd": "/work/repo",
            "permission_mode": "default", "hook_event_name": event, **kw}


# ------------------------------------------------------------------------------------ helpers
class FixedBackend(DecisionBackend):
    """Deterministic S1: answers each hook question from constructor values (keyed by meta['hook_q'])."""

    name = "fixed"
    model = "fixed-test"
    batched_questions = True

    def __init__(self, approval: dict[str, float] | None = None, destructive: float = 0.05, stop_done: float = 0.9,
                 memory: float = 0.9, routing: float = 0.9, complexity: str = "4", delay: float = 0.0,
                 fail: bool = False):
        self.approval = approval or {"allow": 0.95, "ask": 0.04, "block": 0.01}
        self.noul = {"destructive": destructive, "stop_done": stop_done, "memory": memory, "routing": routing}
        self.complexity, self.delay, self.fail = complexity, delay, fail
        self.calls: list[tuple[str, list[Question]]] = []

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        self.calls.append((state, questions))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("backend exploded")
        answers = []
        for q in questions:
            if q.kind == QKind.CHOICE:
                dist = self.approval
            elif q.kind == QKind.NOUL:
                p = self.noul[q.meta["hook_q"]]
                dist = {"yes": p, "no": 1 - p}
            else:
                dist = {self.complexity: 1.0}
            answers.append(answer_from_distribution(q, dist))
        return DecisionResult(answers=answers, usage=Usage(backend="fixed", model="fixed-test",
                                                           input_tokens=len(state) // 4, latency_ms=3.0))


ALLOW = {"allow": 0.95, "ask": 0.04, "block": 0.01}
ASK = {"allow": 0.45, "ask": 0.40, "block": 0.15}
BLOCK = {"allow": 0.05, "ask": 0.15, "block": 0.80}


@pytest.fixture
def cfg(tmp_path: Path) -> H.HookConfig:
    return H.HookConfig(s1={"type": "oracle"}, state_dir=str(tmp_path / "state"), log_path=str(tmp_path / "log.jsonl"))


def records(cfg: H.HookConfig) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(cfg.log_path).read_text().splitlines()]


def _schema(event: str, kind: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / f"{SCHEMA_FILE[event]}.command.{kind}.schema.json").read_text())


def assert_codex_accepts(event: str, out: dict[str, Any] | None) -> None:
    """Mirror Codex: serde deny_unknown_fields schema + output_parser.rs semantic checks."""
    if out is None:
        return  # empty stdout == no hook verdict (all Codex event parsers)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(out, _schema(event, "output"))
    hso = out.get("hookSpecificOutput") or {}
    if hso:
        assert hso["hookEventName"] == event
    for k in ("continue", "stopReason", "suppressOutput"):
        assert k not in out, f"{event}: universal field {k} is unsupported/unwanted"
    if event == "PreToolUse":
        assert "decision" not in out and "reason" not in out  # legacy approve is unsupported
        pd = hso.get("permissionDecision")
        assert pd in (None, "deny"), "Codex rejects permissionDecision allow (w/o updatedInput) and ask"
        if pd == "deny":
            assert hso["permissionDecisionReason"].strip()
    if event == "PermissionRequest":
        d = hso["decision"]
        assert "updatedInput" not in d and "updatedPermissions" not in d and not d.get("interrupt", False)
    if event in ("Stop", "PostToolUse", "UserPromptSubmit"):
        if out.get("decision") == "block":
            assert out.get("reason", "").strip()
        else:
            assert "reason" not in out
    if event == "PostToolUse":
        assert "updatedMCPToolOutput" not in hso


def write_codex_rollout(path: Path) -> Path:
    lines = [
        {"type": "session_meta", "payload": {"id": CODEX_BASE["session_id"], "cwd": "/work/repo"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<environment_context>\n  <cwd>/work/repo</cwd>\n</environment_context>"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Fix the failing test in tests/test_math.py"}]}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "Fix the failing test in tests/test_math.py"}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                                              "arguments": "{\"cmd\":\"pytest -q\"}", "call_id": "call_1"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "call_1",
                                              "output": "1 failed, 3 passed"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "add() is off by one; I will fix it."}]}},
    ]
    path.write_text("\n".join(json.dumps({"timestamp": "2026-09-30T12:00:00.000Z", **ln}) for ln in lines) + "\n")
    return path


def write_generic_transcript(path: Path) -> Path:
    lines = [
        {"type": "user", "message": {"role": "user", "content": "Add a --verbose flag to cli.py"}, "uuid": "u1"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Let me read cli.py first."},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "cli.py"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "import argparse\nparser = ..."}]}},
    ]
    path.write_text("\n".join(json.dumps(ln) for ln in lines) + "\n")
    return path


# ------------------------------------------------------------------------------------ fixtures are faithful
@pytest.mark.parametrize("event,kw", [("PreToolUse", CODEX_PRE), ("PermissionRequest", CODEX_PERM),
                                      ("PostToolUse", CODEX_POST), ("Stop", CODEX_STOP),
                                      ("UserPromptSubmit", CODEX_PROMPT), ("SessionStart", {"source": "startup"})])
def test_codex_fixture_payloads_match_codex_input_schema(event: str, kw: dict[str, Any]) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(codex(event, **kw), _schema(event, "input"))


def test_detect_harness() -> None:
    assert H.detect_harness(codex("PreToolUse", **CODEX_PRE)) == "codex"
    assert H.detect_harness(dsh("PreToolUse", tool_name="bash", tool_input={}, tool_use_id="x")) == "deepseek"
    assert H.detect_harness(generic("PreToolUse", "/tmp/t.jsonl", tool_name="Bash", tool_input={})) == "generic"
    assert H.detect_harness(codex("PreToolUse", **CODEX_PRE), "generic") == "generic"  # explicit flag wins


# ------------------------------------------------------------------------------------ PreToolUse
@pytest.mark.parametrize("dist,expected", [(ALLOW, None), (ASK, None), (BLOCK, "deny")])
def test_codex_pretooluse_only_emits_deny(cfg: H.HookConfig, dist: dict[str, float], expected: str | None) -> None:
    cfg.mode = "replace"
    out = H.handle(codex("PreToolUse", **CODEX_PRE), cfg, FixedBackend(approval=dist))
    assert_codex_accepts("PreToolUse", out)
    assert (out["hookSpecificOutput"]["permissionDecision"] if out else None) == expected
    rec = records(cfg)[-1]
    assert rec["final_value"] == {id(ALLOW): "allow", id(ASK): "ask", id(BLOCK): "block"}[id(dist)]
    assert rec["emitted"] == (expected or "none") and rec["harness"] == "codex" and rec["turn_id"] == "1"


@pytest.mark.parametrize("dist,expected", [(ALLOW, "allow"), (ASK, "ask"), (BLOCK, "deny")])
def test_generic_pretooluse_replace(cfg: H.HookConfig, tmp_path: Path, dist: dict[str, float], expected: str) -> None:
    cfg.mode = "replace"
    t = write_generic_transcript(tmp_path / "t.jsonl")
    out = H.handle(generic("PreToolUse", str(t), tool_name="Bash", tool_input={"command": "ls"},
                          tool_use_id="toolu_2"), cfg, FixedBackend(approval=dist))
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse" and hso["permissionDecision"] == expected
    assert hso["permissionDecisionReason"].startswith("[jev]")
    assert set(out) == {"hookSpecificOutput"}


@pytest.mark.parametrize("dist,expected", [(ALLOW, None), (ASK, "ask"), (BLOCK, "deny")])
def test_deepseek_pretooluse_replace(cfg: H.HookConfig, dist: dict[str, float], expected: str | None) -> None:
    cfg.mode = "replace"
    out = H.handle(dsh("PreToolUse", tool_name="bash", tool_input={"command": "ls"}, tool_use_id="c1"), cfg,
                   FixedBackend(approval=dist))
    assert (out["hookSpecificOutput"]["permissionDecision"] if out else None) == expected


@pytest.mark.parametrize("mode,dist,expected", [
    ("log", BLOCK, None),                                    # log-only never blocks
    ("guard", ALLOW, None), ("guard", ASK, "ask"), ("guard", BLOCK, "deny"),
    ("cascade", ALLOW, "allow"), ("cascade", ASK, None), ("cascade", BLOCK, "deny"),
])
def test_modes_generic(cfg: H.HookConfig, tmp_path: Path, mode: str, dist: dict[str, float],
                      expected: str | None) -> None:
    cfg.mode = mode
    out = H.handle(generic("PreToolUse", "", tool_name="Bash", tool_input={"command": "x"}), cfg,
                   FixedBackend(approval=dist))
    assert (out["hookSpecificOutput"]["permissionDecision"] if out else None) == expected
    rec = records(cfg)[-1]
    assert rec["mode"] == mode and rec["deferred"] == (mode == "cascade" and dist is ASK)


def test_off_mode_skips_s1(cfg: H.HookConfig) -> None:
    cfg.mode = "off"
    b = FixedBackend(approval=BLOCK)
    assert H.handle(codex("PreToolUse", **CODEX_PRE), cfg, b) is None
    assert b.calls == [] and records(cfg)[-1]["mode"] == "off"


def test_destructive_noul_downgrades_allow_and_ask_as(cfg: H.HookConfig) -> None:
    cfg.mode = "replace"
    b = FixedBackend(approval=ALLOW, destructive=0.93)
    out = H.handle(dsh("PreToolUse", tool_name="bash", tool_input={"command": "rm -rf ~"}), cfg, b)
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert "destructive" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert len(b.calls) == 1 and len(b.calls[0][1]) == 2  # Choice + Noul answered in one batched S1 call
    cfg.ask_as = "deny"  # e.g. headless Codex runs: turn "ask" into a deny the harness can honour
    out = H.handle(codex("PreToolUse", **CODEX_PRE), cfg, FixedBackend(approval=ASK))
    assert_codex_accepts("PreToolUse", out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    cfg.destructive_check = False
    b = FixedBackend(approval=ALLOW, destructive=0.99)
    assert H.handle(generic("PreToolUse", "", tool_name="Bash", tool_input={}), cfg, b)[
        "hookSpecificOutput"]["permissionDecision"] == "allow"
    assert len(b.calls[0][1]) == 1


# ------------------------------------------------------------------------------------ PermissionRequest
@pytest.mark.parametrize("dist,behavior", [(ALLOW, "allow"), (ASK, None), (BLOCK, "deny")])
def test_codex_permission_request_cascade(cfg: H.HookConfig, dist: dict[str, float], behavior: str | None) -> None:
    cfg.mode = "cascade"  # Jev decides when confident; otherwise Codex falls through to Guardian / the user
    out = H.handle(codex("PermissionRequest", **CODEX_PERM), cfg, FixedBackend(approval=dist))
    assert_codex_accepts("PermissionRequest", out)
    got = out["hookSpecificOutput"]["decision"]["behavior"] if out else None
    assert got == behavior
    if behavior == "deny":
        assert out["hookSpecificOutput"]["decision"]["message"].startswith("[jev]")
    if behavior == "allow":
        assert out["hookSpecificOutput"]["decision"] == {"behavior": "allow"}


def test_state_contains_task_tail_and_action_from_codex_rollout(cfg: H.HookConfig, tmp_path: Path) -> None:
    cfg.mode = "log"
    t = write_codex_rollout(tmp_path / "rollout.jsonl")
    b = FixedBackend()
    H.handle(codex("PermissionRequest", transcript=str(t), **CODEX_PERM), cfg, b)
    state = b.calls[0][0]
    assert "# Task (the user's request)\nFix the failing test in tests/test_math.py" in state
    assert "environment_context" not in state.split("# Recent trajectory")[0]
    assert "[tool_call] exec_command" in state and "[tool_result] 1 failed, 3 passed" in state
    assert "git push --force origin main" in state and "justification: Push the rebased branch" in state
    assert "stage: approval request" in state
    tr = H.load_transcript(str(t))
    assert [r for r, _ in tr.entries].count("user") == 1  # event_msg duplicate collapsed


def test_state_from_generic_transcript(cfg: H.HookConfig, tmp_path: Path) -> None:
    cfg.mode = "log"
    t = write_generic_transcript(tmp_path / "t.jsonl")
    b = FixedBackend()
    H.handle(generic("PreToolUse", str(t), tool_name="Edit", tool_input={"file_path": "cli.py"}), cfg, b)
    state = b.calls[0][0]
    assert "Add a --verbose flag to cli.py" in state and "[tool_call] Read" in state
    assert "[tool_result] import argparse" in state and '"file_path": "cli.py"' in state


def test_transcript_tail_budget(cfg: H.HookConfig, tmp_path: Path) -> None:
    t = write_codex_rollout(tmp_path / "r.jsonl")
    cfg.transcript_tail, cfg.mode = 40, "log"
    b = FixedBackend()
    H.handle(codex("PreToolUse", transcript=str(t), **CODEX_PRE), cfg, b)
    tail = b.calls[0][0].split("# Recent trajectory (oldest first)\n")[1].split("\n\n# ")[0]
    assert len(tail) <= 40 + len("...[truncated] ")
    cfg.transcript_tail = 0
    H.handle(codex("PreToolUse", transcript=str(t), **CODEX_PRE), cfg, b)
    assert "# Recent trajectory" not in b.calls[1][0]


# ------------------------------------------------------------------------------------ Stop
def test_stop_guard_with_loop_guard_on_dsh(cfg: H.HookConfig) -> None:
    """dsh always reports stop_hook_active=false and has no block cap: the counter file is the guard."""
    cfg.mode, cfg.stop_max_blocks = "guard", 2
    b = FixedBackend(stop_done=0.2)
    stop = dsh("Stop", stop_hook_active=False)
    outs = [H.handle(stop, cfg, b) for _ in range(4)]
    assert [o["decision"] if o else None for o in outs] == ["block", "block", None, "block"]
    assert "run the relevant tests" in outs[0]["reason"] and "1/2" in outs[0]["reason"]
    recs = [r for r in records(cfg) if r["dp"] == "stop"]
    assert recs[2]["loop_guard"] is True and recs[2]["final_value"] == "stop"
    assert len(b.calls) == 3  # the loop-guard pass skips the S1 call
    # a new user prompt resets the counter
    H.handle(dsh("Stop", stop_hook_active=False), cfg, b)  # 2nd consecutive block
    H.handle(dsh("UserPromptSubmit", prompt="now also update the docs"), cfg, FixedBackend())
    assert H.handle(stop, cfg, b)["decision"] == "block"


def test_codex_stop_output_and_complete_task(cfg: H.HookConfig) -> None:
    cfg.mode = "guard"
    out = H.handle(codex("Stop", **CODEX_STOP), cfg, FixedBackend(stop_done=0.1))
    assert_codex_accepts("Stop", out)
    assert out["decision"] == "block"
    b = FixedBackend(stop_done=0.95)
    assert H.handle(codex("Stop", **CODEX_STOP), cfg, b) is None
    assert "I fixed add(); all tests pass." in b.calls[0][0]
    assert records(cfg)[-1]["final_value"] == "stop" and records(cfg)[-1]["p_complete"] == 0.95


def test_stop_log_mode_never_blocks(cfg: H.HookConfig) -> None:
    cfg.mode = "log"
    for _ in range(3):
        assert H.handle(codex("Stop", **CODEX_STOP), cfg, FixedBackend(stop_done=0.0)) is None
    assert {r["final_value"] for r in records(cfg)} == {"continue"}
    assert {r["emitted"] for r in records(cfg)} == {"none"}


# ------------------------------------------------------------------------------------ PostToolUse
def test_post_tool_use_memory_write(cfg: H.HookConfig, tmp_path: Path) -> None:
    cfg.mode, cfg.memory_context = "trigger", True
    out = H.handle(codex("PostToolUse", **CODEX_POST), cfg, FixedBackend(memory=0.8))
    assert_codex_accepts("PostToolUse", out)
    assert out["hookSpecificOutput"]["additionalContext"].startswith("[jev-memory]")
    mem = Path(cfg.state_dir) / CODEX_BASE["session_id"] / "memory.jsonl"
    note = json.loads(mem.read_text().splitlines()[0])
    assert note["tool_name"] == "Bash" and "test_add" in note["observation"] and note["p"] == 0.8
    # below tau: nothing written; log mode: nothing written either
    H.handle(codex("PostToolUse", **CODEX_POST), cfg, FixedBackend(memory=0.2))
    cfg.mode = "log"
    assert H.handle(codex("PostToolUse", **CODEX_POST), cfg, FixedBackend(memory=0.99)) is None
    assert len(mem.read_text().splitlines()) == 1
    assert [r["final_value"] for r in records(cfg)] == ["write", "skip", "write"]


def test_dsh_session_store_replaces_missing_transcript(cfg: H.HookConfig) -> None:
    """dsh sends transcript_path '' -> the adapter's own store supplies task + recent events."""
    cfg.mode = "guard"
    H.handle(dsh("UserPromptSubmit", prompt="Rename util.py to helpers.py and update imports"), cfg, FixedBackend())
    H.handle(dsh("PostToolUse", tool_name="bash", tool_input={"command": "grep -rn util ."}, tool_use_id="c1",
                 tool_response="app.py:1:import util"), cfg, FixedBackend(memory=0.1))
    b = FixedBackend()
    H.handle(dsh("PreToolUse", tool_name="bash", tool_input={"command": "git mv util.py helpers.py"}), cfg, b)
    state = b.calls[0][0]
    assert "Rename util.py to helpers.py" in state and "app.py:1:import util" in state
    assert "git mv util.py helpers.py" in state


# ------------------------------------------------------------------------------------ UserPromptSubmit
def test_user_prompt_submit_routing(cfg: H.HookConfig) -> None:
    cfg.mode = "trigger"
    assert H.handle(codex("UserPromptSubmit", **CODEX_PROMPT), cfg, FixedBackend(routing=0.9)) is None
    rec = records(cfg)[-1]
    assert rec["dp"] == "routing" and rec["final_value"] == "large" and rec["complexity"] == 4.0
    cfg.routing_context = True
    out = H.handle(codex("UserPromptSubmit", **CODEX_PROMPT), cfg, FixedBackend(routing=0.1, complexity="1"))
    assert_codex_accepts("UserPromptSubmit", out)
    assert records(cfg)[-1]["final_value"] == "small" and "routine" in out["hookSpecificOutput"]["additionalContext"]
    cfg.mode = "log"
    assert H.handle(codex("UserPromptSubmit", **CODEX_PROMPT), cfg, FixedBackend()) is None


def test_session_start_and_unknown_events_are_silent(cfg: H.HookConfig) -> None:
    assert H.handle(codex("SessionStart", source="startup"), cfg, FixedBackend()) is None
    assert H.handle({"hook_event_name": "PreCompact", "session_id": "x", "trigger": "auto"}, cfg) is None


# ------------------------------------------------------------------------------------ logging
def test_log_record_fields(cfg: H.HookConfig) -> None:
    cfg.mode = "guard"
    H.handle(codex("PreToolUse", **CODEX_PRE), cfg, FixedBackend(approval=BLOCK))
    rec = records(cfg)[-1]
    assert LOG_FIELDS <= set(rec)
    assert rec["options"] == ["allow", "ask", "block"] and rec["s1_value"] == "block"
    assert math.isclose(sum(rec["s1_dist"].values()), 1.0, abs_tol=1e-3)
    assert rec["model"] == "fixed-test" and rec["input_tokens"] > 0 and rec["error"] is None
    assert rec["latency_ms"] >= 0 and rec["backend_latency_ms"] == 3.0 and "state" not in rec
    cfg.log_state = True
    H.handle(codex("PreToolUse", **CODEX_PRE), cfg, FixedBackend())
    assert "rm -rf build/" in records(cfg)[-1]["state"]


def test_summarize(cfg: H.HookConfig) -> None:
    cfg.mode = "guard"
    for d in (ALLOW, BLOCK, BLOCK):
        H.handle(codex("PreToolUse", **CODEX_PRE), cfg, FixedBackend(approval=d))
    s = H.summarize([cfg.log_path])["codex/tool_select/guard"]
    assert s["n"] == 3 and s["final_value"] == {"allow": 1, "block": 2} and s["emitted"] == {"none": 1, "deny": 2}


# ------------------------------------------------------------------------------------ fail open
def test_fail_open_on_backend_error_and_timeout(cfg: H.HookConfig) -> None:
    cfg.mode = "replace"
    assert H.handle(codex("PreToolUse", **CODEX_PRE), cfg, FixedBackend(fail=True)) is None
    assert records(cfg)[-1]["error"].startswith("RuntimeError") and records(cfg)[-1]["fail_open"] is True
    cfg.timeout = 0.2
    t = time.perf_counter()
    assert H.handle(codex("Stop", **CODEX_STOP), cfg, FixedBackend(stop_done=0.0, delay=3.0)) is None
    assert time.perf_counter() - t < 1.5
    assert records(cfg)[-1]["error"].startswith("TimeoutError")


def test_fail_open_unreachable_jev_in_process(cfg: H.HookConfig) -> None:
    cfg.mode, cfg.s1 = "replace", {"type": "jev", "base_url": "http://127.0.0.1:9", "api_key": "test"}
    assert H.handle(codex("PreToolUse", **CODEX_PRE), cfg) is None
    rec = records(cfg)[-1]
    assert rec["error"] and rec["final_value"] == "allow" and rec["backend"] == "jev"


def _run_cli(payload: Any, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    full_env = {**os.environ, **env}
    return subprocess.run([sys.executable, "-m", "jevharness.adapters.hooks", *args], input=stdin, env=full_env,
                          capture_output=True, text=True, timeout=30)


def test_cli_fail_open_unreachable_jev(tmp_path: Path) -> None:
    env = {"JEV_S1": json.dumps({"type": "jev", "base_url": "http://127.0.0.1:9", "api_key": "test"}),
           "JEV_MODE": "replace", "JEV_TIMEOUT": "2", "JEV_STATE_DIR": str(tmp_path), "JEV_LOG": str(tmp_path / "l.jsonl")}
    for event, kw in (("PreToolUse", CODEX_PRE), ("PermissionRequest", CODEX_PERM), ("Stop", CODEX_STOP),
                      ("PostToolUse", CODEX_POST), ("UserPromptSubmit", CODEX_PROMPT)):
        r = _run_cli(codex(event, **kw), env, "--harness", "codex")
        assert r.returncode == 0 and r.stdout == "", (event, r.stdout, r.stderr)
    recs = [json.loads(ln) for ln in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert len(recs) == 5 and all(rec["error"] for rec in recs)


def test_cli_oracle_backend_end_to_end(tmp_path: Path) -> None:
    env = {"JEV_S1": '{"type":"oracle"}', "JEV_MODE": "replace", "JEV_STATE_DIR": str(tmp_path),
           "JEV_LOG": str(tmp_path / "l.jsonl")}
    # oracle w/o gold answers the first label: approval=allow, destructive=yes -> downgraded to ask
    r = _run_cli(generic("PreToolUse", "", tool_name="Bash", tool_input={"command": "ls"}), env, "--harness", "generic")
    assert r.returncode == 0 and json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"
    r = _run_cli(generic("PreToolUse", "", tool_name="Bash", tool_input={"command": "ls"}),
                 {**env, "JEV_DESTRUCTIVE": "0"}, "--harness", "generic")
    assert json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"
    r = _run_cli(codex("Stop", **CODEX_STOP), env)  # auto-detected as codex; p_complete ~1 -> allow stop
    assert r.returncode == 0 and r.stdout == ""
    r = _run_cli(codex("PreToolUse", **CODEX_PRE), {**env, "JEV_S1": '{"type":"simulated","accuracy":0.9}'},
                 "--mode", "log")
    assert r.returncode == 0 and r.stdout == ""
    assert json.loads((tmp_path / "l.jsonl").read_text().splitlines()[-1])["model"].startswith("sim(")


def test_cli_env_file_survives_dsh_env_scrub(tmp_path: Path) -> None:
    """dsh strips *KEY*/*TOKEN* vars from hook processes; JEV_ENV_FILE carries them (never overriding env)."""
    envf = tmp_path / "jev.env"
    envf.write_text('JEV_MODE=replace\nJEV_S1={"type":"oracle"}\nJEV_DESTRUCTIVE=0\nJEV_TIMEOUT=9\n'
                    "TYPESAFE_API_KEY=sk-test\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("JEV_")}
    env.update(JEV_ENV_FILE=str(envf), JEV_STATE_DIR=str(tmp_path), JEV_LOG=str(tmp_path / "l.jsonl"), JEV_TIMEOUT="3")
    r = subprocess.run([sys.executable, "-m", "jevharness.adapters.hooks", "--harness", "generic"],
                       input=json.dumps(generic("PreToolUse", "", tool_name="Bash", tool_input={})),
                       env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert json.loads((tmp_path / "l.jsonl").read_text())["mode"] == "replace"


@pytest.mark.parametrize("stdin,args", [("not json", ()), ("", ()), ("[1,2]", ()),
                                        (json.dumps(CODEX_BASE), ("--no-such-flag",)),
                                        (json.dumps(codex("PreToolUse", **CODEX_PRE)), ("--mode", "bogus"))])
def test_cli_never_exits_2(tmp_path: Path, stdin: str, args: tuple[str, ...]) -> None:
    """Exit code 2 means "block" to every hook protocol; bad input must fail open instead."""
    r = _run_cli(stdin, {"JEV_S1": '{"type":"oracle"}', "JEV_STATE_DIR": str(tmp_path)}, *args)
    assert r.returncode == 0 and r.stdout == ""


# ------------------------------------------------------------------------------------ Jev wire format
def test_jev_wire_request_from_hook(cfg: H.HookConfig) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["body"] = body
        return httpx.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 812, "output_tokens": 0},
                                         "answers": {"q0": {"type": "choice", "choice": "block", "confidence": 0.9,
                                                            "probabilities": {"allow": 0.02, "ask": 0.08, "block": 0.9}},
                                                     "q1": {"type": "noul", "noul": 0.97}}})

    b = JevBackend(model="jev-latest", api_key="k", base_url="https://mock", max_retries=0)
    b._http = httpx.Client(base_url="https://mock", transport=httpx.MockTransport(handler))
    cfg.mode = "guard"
    out = H.handle(codex("PreToolUse", **CODEX_PRE), cfg, b)
    assert_codex_accepts("PreToolUse", out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    qs = seen["body"]["questions"]
    assert qs["q0"]["type"] == "choice" and list(qs["q0"]["criteria"]) == ["allow", "ask", "block"]
    assert qs["q1"] == {"type": "noul", "instructions": H.DESTRUCTIVE_TEXT, "criteria": H.DESTRUCTIVE_CRITERIA}
    assert "rm -rf build/" in seen["body"]["state"]
    rec = records(cfg)[-1]
    assert rec["model"] == "jev-1.13.0" and rec["input_tokens"] == 812 and rec["p_destructive"] == 0.97
    assert "rm -rf build/ && make" in rec["tool_input"]  # joinable with codex exec --json command items


def test_config_from_env() -> None:
    env = {"JEV_S1": '{"type":"openjev","base_url":"http://gpu:8080"}',
           "JEV_MODE": "cascade,stop=guard,routing=log,PreToolUse=off",
           "JEV_TAU": "0.7,stop=0.4", "JEV_TAU_BLOCK": "0.6", "JEV_TRANSCRIPT_TAIL": "123", "JEV_ASK_AS": "abstain",
           "JEV_MEMORY_CONTEXT": "1"}
    c = H.HookConfig.from_env(env)
    assert c.s1 == {"type": "openjev", "base_url": "http://gpu:8080"}
    assert (c.mode_for("tool_select"), c.mode_for("stop"), c.mode_for("routing")) == ("cascade", "guard", "log")
    assert c.mode_for("tool_select", "PreToolUse") == "off"  # event key beats DP key beats default
    assert c.mode_for("tool_select", "PermissionRequest") == "cascade"
    with pytest.raises(ValueError):
        H.HookConfig.from_env({"JEV_MODE": "guard,tool_selct=off"})
    assert (c.tau_for("tool_select"), c.tau_for("stop"), c.tau_block) == (0.7, 0.4, 0.6)
    assert c.transcript_tail == 123 and c.ask_as == "abstain" and c.memory_context and c.destructive_check
    assert H.HookConfig.from_env({"JEV_RUN_TAG": "jev_cascade/task-7"}).run_tag == "jev_cascade/task-7"
    d = H.HookConfig.from_env({})
    assert d.s1 == {"type": "jev"} and d.mode == "guard" and d.timeout == 2.0
    assert (d.tau_for("tool_select"), d.tau_for("stop")) == (0.8, 0.5)
    spec = H.build_backend(H.HookConfig(s1={"type": "jev", "api_key": "k"}, timeout=1.5))
    assert spec.timeout == 1.5 and spec.max_retries == 0


# ------------------------------------------------------------------------------------ shipped configs
def _commands(hooks_map: dict[str, Any]) -> list[tuple[str, str]]:
    return [(ev, h["command"]) for ev, groups in hooks_map.items() for g in groups for h in g["hooks"]]


@pytest.mark.parametrize("path,harness", [("adapters/codex/hooks.json", "codex"),
                                          ("adapters/deepseek/hooks.json", "deepseek")])
def test_shipped_hook_configs_run(tmp_path: Path, path: str, harness: str) -> None:
    cfg_json = json.loads((ROOT / path).read_text())
    if harness == "codex":  # HooksFile is deny_unknown_fields: {description, hooks} (config/src/hook_config.rs:10-17)
        assert set(cfg_json) <= {"description", "hooks"}
    cmds = _commands(cfg_json["hooks"])
    assert {ev for ev, _ in cmds} >= {"PreToolUse", "PostToolUse", "Stop", "UserPromptSubmit"}
    env = {**os.environ, "JEV_S1": '{"type":"oracle"}', "JEV_MODE": "replace", "JEV_DESTRUCTIVE": "0",
           "JEV_STATE_DIR": str(tmp_path), "JEV_LOG": str(tmp_path / "l.jsonl")}
    ev, cmd = next((e, c) for e, c in cmds if e == "PreToolUse")
    assert f"--harness {harness}" in cmd
    cmd = cmd.replace("/ABS/PATH/JevAgentHarness/research/.venv/bin/python", sys.executable)
    cmd = cmd.replace("/ABS/PATH/JevAgentHarness/research", str(ROOT))
    payload = {"codex": codex("PreToolUse", **CODEX_PRE), "deepseek": dsh("PreToolUse", tool_name="bash",
                                                                          tool_input={"command": "ls"}),
               "generic": generic("PreToolUse", "", tool_name="Bash", tool_input={"command": "ls"})}[harness]
    r = subprocess.run(cmd, shell=True, input=json.dumps(payload), env=env, capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 0
    expected = {"codex": "", "deepseek": "", "generic": "allow"}[harness]  # allow is silent on codex / dsh
    got = json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] if r.stdout.strip() else ""
    assert got == expected and json.loads((tmp_path / "l.jsonl").read_text())["harness"] == harness


def test_shipped_codex_toml_and_dsh_patch_parse() -> None:
    tomllib = pytest.importorskip("tomllib")
    import yaml
    toml = tomllib.loads((ROOT / "adapters/codex/config.toml").read_text())
    assert toml["approvals_reviewer"] == "auto_review" and toml["approval_policy"] == "on-request"
    assert {ev for ev, _ in _commands(toml["hooks"])} == {"PermissionRequest", "PreToolUse", "Stop", "PostToolUse",
                                                          "UserPromptSubmit"}
    patch = yaml.safe_load((ROOT / "adapters/deepseek/jev-hooks.cordis.yml").read_text())
    row = patch[0]["insert"][0]
    assert row["config"]["configPath"].endswith("adapters/deepseek/hooks.json")
