"""E29b (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~19:55 UTC, "E29b: step-type routing with a write guard"):
the read-only classifier, the write guard in the proxy, the typed routing state machine, the typed shadow mode, the fit,
the go rule and the readout. Offline: a fake upstream (aiohttp) and a fake decision backend."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jevharness.adapters import route_formats as F  # noqa: E402
from jevharness.adapters import route_proxy as RP  # noqa: E402
from jevharness.decision.types import Answer, DecisionResult, Usage  # noqa: E402

import explore_e29b_route as B  # noqa: E402

PRICES = {"strong": {"input_per_m": 1.5, "output_per_m": 9.0, "cached_input_per_m": 0.15},
          "fast": {"input_per_m": 0.3, "output_per_m": 2.5, "cached_input_per_m": 0.03}}


# ------------------------------------------------------------------------------------------------ classifier
READ_ONLY = [
    "ls -la", "cat a.py", "head -50 a.py", "tail -n 20 log.txt", "wc -l a.py", "pwd", "tree -L 2",
    "cd /testbed && grep -n 'def foo' x.py | head -10", "grep -rn foo . 2>/dev/null | sort | uniq -c",
    "find . -name '*.py' -type f 2>/dev/null | head", "rg -n pattern src", "sed -n '10,20p' a.py",
    "nl -ba a.py | sed -n '1,40p'", "sed -n '/def foo/,/def bar/p' a.py", "sed -ne '1,3p' a.py",
    "awk '{print $1}' f", "git status", "git diff", "git -C /repo log --oneline -5", "git show HEAD:a.py",
    "ls; cat a\ncat b", "cat x >/dev/null", "ls -la 2>&1 | head", "diff a b", "cut -d, -f1 a.csv | sort", "less a",
    "file a.bin", "stat a", "du -sh .", "which python",
]
NOT_READ_ONLY = [
    "", "rm -rf build", "mv a b", "cp a b", "touch x", "mkdir d", "cat a > b", "cat a >> b", "echo hi",
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "python -c 'print(1)'", "python3 repro.py", "pytest -q",
    "make", "pip install x", "npm test", "ls\nrm x", "ls # comment\nrm x", "ls && rm x", "cat x | tee y",
    "grep -l foo . | xargs rm", "sed -i 's/a/b/' f", "sed 's/a/b/' f", "sed -n 's/a/b/w out' f", "sed -n -f s.sed f",
    "find . -name '*.pyc' -delete", "find . -exec rm {} \\;", "awk '{print > \"o\"}' f", "awk 'BEGIN{system(\"rm x\")}'",
    "git checkout a.py", "git apply p.diff", "git diff --output=x", "git -c core.pager=sh log", "echo $(rm x)",
    "cat `which x`", "diff <(ls) b", "cat <<EOF\nx\nEOF", "ls &", "(cd x && ls)", "A=1 ls", "sort -o out in",
    "uniq a b", "tree -o out", "rg foo --pre cat", "cat 'unterminated", "sudo cat /etc/shadow",
    "cat x > /dev/nullx", "apply_patch < p.diff",
]


@pytest.mark.parametrize("cmd", READ_ONLY)
def test_classifier_accepts_the_registered_read_only_commands(cmd: str):
    assert F.read_only_command(cmd) is True


@pytest.mark.parametrize("cmd", NOT_READ_ONLY)
def test_classifier_rejects_everything_else(cmd: str):
    assert F.read_only_command(cmd) is False


def test_tool_level_classification_covers_the_three_harnesses():
    # pi
    assert F.read_only_call("read", {"path": "a.py"}) and F.read_only_call("grep", {"pattern": "x"})
    assert not F.read_only_call("edit", {"path": "a.py"}) and not F.read_only_call("write", {"path": "a.py"})
    assert F.read_only_call("bash", '{"command": "ls"}') and not F.read_only_call("bash", {"command": "rm x"})
    # SWE-agent
    assert F.read_only_call("str_replace_editor", {"command": "view", "path": "/a.py"})
    for sub in ("create", "str_replace", "insert", "undo_edit"):
        assert not F.read_only_call("str_replace_editor", {"command": sub, "path": "/a.py"})
    assert not F.read_only_call("submit", {})
    # a response: every call read-only, and at least one call (a text-only answer ends the episode)
    assert F.read_only_calls([("read", {"path": "a"}), ("bash", {"command": "ls"})])
    assert not F.read_only_calls([("read", {"path": "a"}), ("bash", {"command": "rm a"})])
    assert not F.read_only_calls([])


def test_parsed_responses_carry_every_tool_call_and_the_verdict():
    gem = {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "read", "args": {"path": "a"}}},
        {"functionCall": {"name": "bash", "args": {"command": "python x.py"}}}]}}]}
    p = F.parse_response(F.FORMAT_GEMINI, json.dumps(gem).encode(), streamed=False)
    assert len(p.calls) == 2 and p.key == "read:path:a" and p.read_only is False
    oai = {"choices": [{"message": {"content": "x", "tool_calls": [
        {"id": "1", "function": {"name": "bash", "arguments": '{"command": "cd /testbed && cat a.py"}'}}]}}]}
    p = F.parse_response(F.FORMAT_OPENAI, json.dumps(oai).encode(), streamed=False)
    assert p.read_only is True
    assert F.parse_response(F.FORMAT_OPENAI, b"not json", streamed=False).read_only is None
    none = {"choices": [{"message": {"content": "done"}}]}
    assert F.parse_response(F.FORMAT_OPENAI, json.dumps(none).encode(), streamed=False).read_only is False


# ------------------------------------------------------------------------------------------------ routing policy
def test_typed_routing_needs_the_inspect_answer_and_caps_segments_at_k():
    st = RP.EpisodeState(arm="gated", episode="e")
    tau, k = 0.6, 2
    seq = [(0, 0.9, True), (1, 0.9, True), (2, 0.9, False), (3, 0.8, True), (4, 0.95, True), (5, 0.7, True),
           (6, 0.5, True), (7, None, False)]
    out = [RP.route_step(st, t, c, tau, k, answer_ok=ok) for t, c, ok in seq]
    assert out == [("strong", "first"), ("fast", "enter"), ("strong", "other_answer"), ("fast", "enter"),
                   ("fast", "continue"), ("strong", "cap_k"), ("strong", "below_tau"), ("strong", "fail_open")]


# ------------------------------------------------------------------------------------------------ proxy
class TypedBackend:
    """A fake decision model for the typed question: answers ``value`` with p(inspect) = c."""
    name, model = "fake", "fake-jev"

    def __init__(self, value: str = "inspect", c: float = 0.9):
        self.value, self.c, self.questions = value, c, []

    def decide(self, state, questions, ledger=None, decision_point=None):
        q = questions[0]
        self.questions.append(q)
        rest = (1 - self.c) / 3
        dist = {o: (self.c if o == "inspect" else rest) for o in q.options}
        if self.value != "inspect":
            dist = {o: (0.7 if o == self.value else 0.1) for o in q.options}
            dist["inspect"] = self.c if self.c < 0.7 else 0.1
        return DecisionResult(answers=[Answer(question_id=q.id, kind=q.kind, distribution=dist,
                                              value=max(dist, key=dist.get), confidence=max(dist.values()))],
                              usage=Usage(backend="fake", model="fake-jev", input_tokens=100, cost_usd=1e-5))

    def describe(self):
        return {"backend": "fake"}


def upstream(state: dict[str, Any]) -> web.Application:
    """Gemini: the strong model always answers ``git diff``; the fast model answers ``state['fast_cmd']`` (default
    ``ls``, read-only). OpenAI likewise by the body's model."""
    state.setdefault("seen", [])

    def cmd(model: str) -> str:
        return state.get("fast_cmd", "ls") if "fast" in model else "git diff"

    async def gem(request: web.Request) -> web.StreamResponse:
        model, method = request.match_info["mm"].split(":", 1)
        state["seen"].append({"model": model, "method": method})
        resp = {"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": "bash", "args": {"command": cmd(model)}}, "thoughtSignature": "s"}]},
            "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 20},
            "modelVersion": model}
        if method.startswith("stream"):
            out = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await out.prepare(request)
            await out.write(f"data: {json.dumps(resp)}\r\n\r\n".encode())
            await out.write_eof()
            return out
        return web.json_response(resp)

    async def oai(request: web.Request) -> web.Response:
        body = await request.json()
        state["seen"].append({"model": body["model"]})
        return web.json_response({"model": body["model"], "choices": [{"message": {
            "role": "assistant", "content": "T", "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": "bash",
                                                             "arguments": json.dumps({"command": cmd(body["model"])})}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                      "cost": 0.002 if "strong" in body["model"] else 0.0005}})

    app = web.Application()
    app.router.add_post("/v1beta/models/{mm}", gem)
    app.router.add_post("/api/v1/chat/completions", oai)
    return app


def _gem_body(n: int) -> dict[str, Any]:
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": "Fix the bug in calc.py"}]}]
    for i in range(n):
        contents += [{"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {"command": f"c{i}"}},
                                                  "thoughtSignature": "s"}]},
                     {"role": "user", "parts": [{"functionResponse": {"name": "bash", "response": {"output": "o"}}}]}]
    return {"contents": contents}


async def _run_proxy(fmt: str, arms: dict[str, Any], tmp: Path, backend: Any, script, upstate=None):
    upstate = upstate if upstate is not None else {}
    up = TestServer(upstream(upstate))
    await up.start_server()
    base = str(up.make_url("")).rstrip("/")
    tiers = {"strong": {"model": "m-strong", "prices": PRICES["strong"]}, "fast": {"model": "m-fast",
                                                                                   "prices": PRICES["fast"]}}
    if fmt == "openai":
        tiers = {"strong": {"model": "m-strong"}, "fast": {"model": "m-fast"}}
    cfg = RP.ProxyConfig.from_dict({"format": fmt, "upstream": base + ("/v1beta" if fmt == "gemini" else "/api/v1"),
                                    "key_env": "E29B_TEST_KEY", "tiers": tiers, "arms": arms, "retries": 0,
                                    "decision": {"wait_s": 2.0, "timeout_s": 2.0}, "log_dir": str(tmp / "logs")})
    proxy = RP.RouteProxy(cfg, backend=backend, api_key="SECRET-KEY-456")
    client = TestClient(TestServer(proxy.app()))
    await client.start_server()
    try:
        res = await script(client)
        await proxy.drain()
    finally:
        await client.close()
        await up.close()
    return res, proxy, upstate


def _logs(tmp: Path, name: str) -> list[dict[str, Any]]:
    p = tmp / "logs" / name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


GATED = {"gated": {"mode": "gated", "tau": 0.5, "k": 6, "question": "step_type", "guard": True}}


async def _steps(client, n: int, arm: str = "gated", stream: bool = True) -> list[str]:
    texts = []
    for t in range(n):
        m = "streamGenerateContent?alt=sse" if stream else "generateContent"
        r = await client.post(f"/e29/{arm}/ep/v1beta/models/x:{m}", json=_gem_body(t))
        assert r.status == 200
        texts.append(await r.text())
        await asyncio.sleep(0.05)
    return texts


def test_guard_accepts_read_only_fast_actions_and_returns_them_whole(tmp_path: Path):
    backend = TypedBackend("inspect", 0.9)
    texts, proxy, up = asyncio.run(_run_proxy("gemini", GATED, tmp_path, backend, lambda c: _steps(c, 4)))
    calls = _logs(tmp_path, "calls.jsonl")
    assert [(c["tier"], c.get("guard")) for c in calls] == [("strong", None), ("fast", "accept"), ("fast", "accept"),
                                                            ("fast", "accept")]
    assert all(c["ro"] is True for c in calls)
    assert '"ls"' in texts[1] and texts[1].startswith("data:")          # the buffered SSE body reaches the harness
    assert backend.questions[0].options == ["inspect", "edit", "run", "finish"]
    dec = _logs(tmp_path, "decisions.jsonl")
    assert all(d["question"] == "step_type" and d["value"] == "inspect" and d["c"] == pytest.approx(0.9) for d in dec)
    assert "SECRET-KEY-456" not in (tmp_path / "logs" / "calls.jsonl").read_text()


def test_guard_rejects_a_writing_fast_action_and_the_strong_llm_writes_the_step(tmp_path: Path):
    texts, proxy, up = asyncio.run(_run_proxy("gemini", GATED, tmp_path, TypedBackend("inspect", 0.9),
                                              lambda c: _steps(c, 3), upstate={"fast_cmd": "rm -rf build"}))
    calls = _logs(tmp_path, "calls.jsonl")
    assert [(c["step"], c["tier"], c.get("guard"), c["route_reason"]) for c in calls] == [
        (0, "strong", None, "first"), (1, "fast", "reject", "enter"), (1, "strong", None, "guard_reject"),
        (2, "fast", "reject", "enter"), (2, "strong", None, "guard_reject")]
    assert "rm -rf" not in texts[1] and "git diff" in texts[1]        # the harness never sees the writing action
    assert calls[1]["cost_usd"] > 0                                     # a discarded call still counts
    assert [s["model"] for s in up["seen"]] == ["m-strong", "m-fast", "m-strong", "m-fast", "m-strong"]


def test_other_answers_and_low_confidence_keep_the_strong_llm(tmp_path: Path):
    _, _, up = asyncio.run(_run_proxy("gemini", GATED, tmp_path, TypedBackend("edit", 0.2),
                                      lambda c: _steps(c, 3, stream=False)))
    calls = _logs(tmp_path, "calls.jsonl")
    assert [c["tier"] for c in calls] == ["strong"] * 3
    assert [c["route_reason"] for c in calls][1:] == ["below_tau", "below_tau"]
    tmp2 = tmp_path / "b"
    _, _, _ = asyncio.run(_run_proxy("gemini", {"gated": {**GATED["gated"], "tau": 0.0}}, tmp2,
                                     TypedBackend("run", 0.2), lambda c: _steps(c, 2, stream=False)))
    assert [c["route_reason"] for c in _logs(tmp2, "calls.jsonl")] == ["first", "other_answer"]


def test_sticky_segments_are_capped_at_k_six(tmp_path: Path):
    asyncio.run(_run_proxy("gemini", GATED, tmp_path, TypedBackend("inspect", 0.99),
                           lambda c: _steps(c, 9, stream=False)))
    tiers = [c["tier"] for c in _logs(tmp_path, "calls.jsonl")]
    assert tiers == ["strong"] + ["fast"] * 6 + ["strong", "fast"]


def test_typed_shadow_mode_logs_the_fast_verdict_without_a_second_strong_sample(tmp_path: Path):
    arms = {"shadow": {"mode": "shadow", "question": "step_type", "shadow_roles": ["fast"]}}
    asyncio.run(_run_proxy("gemini", arms, tmp_path, TypedBackend("inspect", 0.8), lambda c: _steps(c, 3, "shadow"),
                           upstate={"fast_cmd": "python x.py"}))
    calls = _logs(tmp_path, "calls.jsonl")
    assert sorted((c["step"], c["role"]) for c in calls) == sorted((t, r) for t in range(3)
                                                                   for r in ("primary", "shadow_fast"))
    by = {(c["step"], c["role"]): c for c in calls}
    assert by[(1, "primary")]["ro"] is True and by[(1, "shadow_fast")]["ro"] is False
    assert sorted(d["step"] for d in _logs(tmp_path, "decisions.jsonl")) == [1, 2, 3]


def test_openai_format_guard_and_e29_arms_unchanged(tmp_path: Path):
    body = {"model": "x", "messages": [{"role": "user", "content": "task"}]}

    async def script(client):
        for t in range(3):
            b = json.loads(json.dumps(body))
            for i in range(t):
                b["messages"] += [{"role": "assistant", "content": "x", "tool_calls": [
                    {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
                    {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"}]
            r = await client.post("/e29/gated/django__django-1/api/v1/chat/completions", json=b)
            assert r.status == 200
            await asyncio.sleep(0.05)

    asyncio.run(_run_proxy("openai", GATED, tmp_path, TypedBackend("inspect", 0.9), script,
                           upstate={"fast_cmd": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}))
    calls = _logs(tmp_path, "calls.jsonl")
    assert [(c["tier"], c.get("guard")) for c in calls] == [("strong", None), ("fast", "reject"), ("strong", None),
                                                            ("fast", "reject"), ("strong", None)]
    cfg = RP.ArmCfg(mode="gated", tau=0.5)
    assert cfg.question == "route" and cfg.guard is False and cfg.shadow_roles == ["fast", "strong"]
    with pytest.raises(ValueError):
        RP.ArmCfg(mode="gated", question="nope")


# ------------------------------------------------------------------------------------------------ fit, go, readout
def _cal(rows: list[tuple[str, int, str, float, bool, bool | None]]) -> tuple[list[dict], list[dict]]:
    calls, decs = [], []
    for ep, t, value, c, ro, fro in rows:
        calls.append({"arm": "shadow", "episode": ep, "step": t, "role": "primary", "status": 200, "key": "k",
                      "ro": ro, "cost_usd": 0.01})
        if fro is not None:
            calls.append({"arm": "shadow", "episode": ep, "step": t, "role": "shadow_fast", "status": 200, "key": "k",
                          "ro": fro, "cost_usd": 0.002})
        decs.append({"arm": "shadow", "episode": ep, "step": t, "c": c, "value": value})
    return calls, decs


def test_fit_takes_the_lowest_threshold_with_read_only_precision_095():
    rows = [("a", i + 1, "inspect", 0.9, True, True) for i in range(19)] + [("a", 30, "inspect", 0.9, False, True)]
    rows += [("b", i + 1, "inspect", 0.6, i % 2 == 0, False) for i in range(10)]        # 50 % precise at 0.6
    rows += [("c", i + 1, "edit", 0.1, False, None) for i in range(10)]
    rows += [("c", 0, "inspect", 0.99, True, True)]                                     # the first step never counts
    fit = B.fit_typed(B.typed_steps(*_cal(rows)))
    assert fit["tau"] == 0.9 and fit["precision"] == pytest.approx(0.95) and fit["n_routed"] == 20
    assert fit["n_steps"] == 40 and fit["implied_share"] == pytest.approx(0.5) and fit["n_eligible"] == 30
    assert fit["guard_accept_sim"] == 1.0 and fit["recall"] == pytest.approx(19 / 24)
    assert fit["sim_llm_saving"] == pytest.approx(20 * 0.008 / 0.4)
    # the precision rule is never met: the gate routes nothing
    bad = B.fit_typed(B.typed_steps(*_cal([("a", i + 1, "inspect", 0.5 + i / 100, i % 3 == 0, None)
                                          for i in range(9)])))
    assert bad["tau"] is None and bad["n_routed"] == 0
    # 'edit' answers are never eligible, whatever their p(inspect)
    only_edit = B.fit_typed(B.typed_steps(*_cal([("a", 1, "edit", 0.4, True, None)])))
    assert only_edit["tau"] is None and only_edit["n_eligible"] == 0


def _pilot(saving: float, base: int, gated: int) -> dict:
    return {"n": 30, "paired": {"saving": saving, "solved_base": base, "solved_arm": gated}}


def test_go_rule_is_the_registered_one():
    assert B.go_rule({"tb": _pilot(0.25, 15, 12), "mswe": _pilot(0.05, 22, 19)})["decision"] == "go"
    assert B.go_rule({"tb": _pilot(0.25, 15, 11), "mswe": _pilot(0.30, 22, 22)})["decision"] == "no-go"  # tb lost 4
    assert B.go_rule({"tb": _pilot(0.02, 15, 12), "mswe": _pilot(0.30, 22, 22)})["decision"] == "go"
    assert B.go_rule({"tb": _pilot(0.25, 15, 11), "mswe": _pilot(0.10, 22, 22)})["decision"] == "no-go"
    assert B.go_rule({"tb": _pilot(0.19, 15, 15), "mswe": _pilot(0.19, 22, 22)})["decision"] == "no-go"
    r = B.go_rule({"tb": {"n": 0, "no_gate": True}, "mswe": _pilot(0.21, 22, 19)})
    assert r["decision"] == "go" and r["per_harness"]["tb"]["ok"] and r["per_harness"]["tb"]["drop"] == 0
    assert B.go_rule({"tb": {"n": 0, "status": "stopped: smoke failed"}, "mswe": _pilot(0.5, 22, 22)})["decision"] \
        == "no-go"
    assert B.go_rule({"mswe": _pilot(0.5, 22, 22)})["decision"] == "no-go"


def test_full_subset_is_the_first_100_of_e4rs_extension_and_hashed():
    sel = json.loads((B.CFG / "selection.json").read_text())
    text = (B.CFG / "mswe_full.txt").read_text()
    assert B.E.sha256_text(text) == sel["files"]["mswe_full.txt"]["sha256"]
    ids = B.E.read_ids(B.CFG / "mswe_full.txt")
    assert ids == B.full_mswe_ids() and len(ids) == 100
    assert ids[:30] == B.E.read_ids(B.E.CFG / "mswe_pilot.txt")


def test_summary_counts_guard_verdicts_and_fast_steps_and_the_readout_writes_e29b_keys(tmp_path: Path):
    run, logs = tmp_path / "run", tmp_path / "proxy"
    logs.mkdir()
    calls, decs = [], []
    for i in range(4):
        task = f"task-{i}"
        for arm in ("baseline", "gated"):
            d = run / arm / f"{task}__abc{i}"
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps({
                "task_name": task, "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_execution": {"started_at": "2026-10-09T10:00:00Z", "finished_at": "2026-10-09T10:02:00Z"}}))
            ep = f"{task}__abc{i}"
            for t in range(4):
                if arm == "gated" and t in (1, 2):
                    calls.append({"arm": arm, "episode": ep, "step": t, "role": "primary", "status": 200,
                                  "tier": "fast", "guard": "accept" if t == 1 else "reject", "cost_usd": 0.01,
                                  "route_reason": "enter"})
                    if t == 2:
                        calls.append({"arm": arm, "episode": ep, "step": t, "role": "primary", "status": 200,
                                      "tier": "strong", "cost_usd": 0.1, "route_reason": "guard_reject"})
                    continue
                calls.append({"arm": arm, "episode": ep, "step": t, "role": "primary", "status": 200,
                              "tier": "strong", "cost_usd": 0.1, "route_reason": "first"})
            if arm == "gated":
                decs += [{"arm": arm, "episode": ep, "step": t, "c": 0.9, "value": "inspect", "cost_usd": 1e-4,
                          "latency_s": 0.1} for t in range(1, 4)]
    (logs / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    (logs / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decs) + "\n")
    s = B.E.summarize("tb", run, [logs], [f"task-{i}" for i in range(4)], k=B.K)
    g = s["arms"]["gated"]
    assert g["guard_accept"] == 4 and g["guard_reject"] == 4 and g["steps"] == 16 and g["fast_steps"] == 4
    assert g["fast_step_share"] == pytest.approx(0.25)
    assert s["paired"]["saving"] == pytest.approx(1 - (0.32 + 3e-4) / 0.4)
    vals = B.tex_values({"pilot": {"tb": s}, "gates": {"tb": {"tau": 0.9, "precision": 0.96, "n_steps": 50}},
                         "go": {"decision": "no-go", "per_harness": {"tb": {"saving": 0.2, "drop": 0}}}})
    assert vals["e29b/pilot/tb/guard_reject"] == "4" and vals["e29b/pilot/tb/guard_reject_share"] == "50.0"
    assert vals["e29b/pilot/tb/fast_share"] == "25.0" and vals["e29b/gate/tb/tau"] == "0.900"
    assert vals["e29b/go/decision"] == "no-go" and not [k for k in vals if k.startswith("e29/")]
    B.write_tex(vals, tmp_path / "x.tex")
    assert "\\defres{e29b/pilot/tb/saving}" in (tmp_path / "x.tex").read_text()


def test_smoke_checks_need_a_guard_verdict_and_typed_answers():
    calls = [{"arm": "smoke_gated", "episode": "e", "step": 0, "role": "primary", "status": 200, "tier": "strong",
              "cost_usd": 0.01},
             {"arm": "smoke_gated", "episode": "e", "step": 1, "role": "primary", "status": 200, "tier": "fast",
              "guard": "accept", "cost_usd": 0.001},
             {"arm": "shadow", "episode": "s", "step": 1, "role": "primary", "status": 200, "ro": True, "cost_usd": 0.01},
             {"arm": "shadow", "episode": "s", "step": 1, "role": "shadow_fast", "status": 200, "ro": False,
              "cost_usd": 0.001}]
    decs = [{"arm": "smoke_gated", "c": 0.7, "value": "inspect"}, {"arm": "shadow", "c": 0.2, "value": "run"}]
    trials = {"smoke_gated": {"t": {"infra_error": False}}, "shadow": {"t": {"infra_error": False}}}
    assert B.smoke_checks("tb", calls, decs, trials, 0)["ok"]
    assert not B.smoke_checks("tb", calls[:1] + calls[2:], decs, trials, 0)["checks"]["4_guard_judged"]
    assert not B.smoke_checks("tb", calls, decs, trials, 1)["ok"]


def test_calibration_descriptives_report_auroc_and_read_only_share_by_answer():
    rows = [("a", 1, "inspect", 0.9, True, True), ("a", 2, "inspect", 0.8, False, False),
            ("a", 3, "edit", 0.1, False, False), ("a", 4, "run", 0.2, True, False)]
    d = B.calib_descriptives(B.typed_steps(*_cal(rows)))
    assert d["n"] == 4 and d["by_answer"]["inspect"] == {"n": 2, "read_only": 0.5}
    assert d["by_answer"]["finish"]["n"] == 0 and d["fast_ro_when_strong_ro"] == 0.5
    assert d["fast_ro_when_strong_not"] == 0.0 and d["auroc_c_inspect"] == pytest.approx(0.75)
    assert d["ro_cost_share"] == pytest.approx(0.5)
