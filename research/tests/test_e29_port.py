"""E29 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09, "E29: the full harness in open-source harnesses"): the routing
proxy (``jevharness.adapters.route_proxy`` / ``route_formats``), the gate fit, the go rule and the readout. Offline:
a fake upstream (aiohttp) and a fake decision backend."""
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
from jevharness.decision.types import Answer, DecisionResult, QKind, Usage  # noqa: E402

GEMINI_PRICES = {"strong": {"input_per_m": 1.5, "output_per_m": 9.0, "cached_input_per_m": 0.15},
                 "fast": {"input_per_m": 0.3, "output_per_m": 2.5, "cached_input_per_m": 0.03}}


# ------------------------------------------------------------------------------------------------ formats
def test_action_keys_follow_the_registered_equivalence_rule():
    assert F.action_key("bash", {"command": "  ls   -la\n"}) == F.action_key("bash", '{"command": "ls -la"}')
    assert F.action_key("bash", {"command": "ls"}) != F.action_key("bash", {"command": "ls -la"})
    assert F.action_key("str_replace_editor", {"command": "view", "path": "/a.py", "view_range": [1, 5]}) == \
        F.action_key("str_replace_editor", {"command": "view", "path": "/a.py"})
    assert F.action_key("str_replace_editor", {"command": "view", "path": "/a.py"}) != \
        F.action_key("str_replace_editor", {"command": "str_replace", "path": "/a.py"})
    assert F.action_key("edit", {"path": "x.py", "oldText": "a", "newText": "b"}) == \
        F.action_key("edit", {"path": "x.py", "oldText": "c", "newText": "d"})
    assert F.action_key("read", {"path": "x.py"}) != F.action_key("edit", {"path": "x.py"})
    assert F.action_key("submit", {}) == "submit:{}"
    assert F.action_key(None, None) == F.NO_TOOL
    assert F.equivalent(F.NO_TOOL, F.NO_TOOL) is True and F.equivalent(None, "x") is None


def _gem_body(n_model_turns: int = 1) -> dict[str, Any]:
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": "Fix the bug in calc.py"}]}]
    for i in range(n_model_turns):
        contents.append({"role": "model", "parts": [{"text": f"step {i}"},
                                                    {"functionCall": {"name": "bash", "args": {"command": f"cmd{i}"}},
                                                     **({"thoughtSignature": "sig"} if i % 2 == 0 else {})}]})
        contents.append({"role": "user", "parts": [{"functionResponse": {"name": "bash",
                                                                          "response": {"output": f"out{i}"}}}]})
    return {"contents": contents, "systemInstruction": {"parts": [{"text": "sys"}]}}


def test_gemini_episode_step_index_and_signature_fix():
    body = _gem_body(3)
    assert F.gemini_step_index(body) == 3
    task, steps = F.gemini_episode(body)
    assert task == "Fix the bug in calc.py"
    assert [s["output"] for s in steps] == ["out0", "out1", "out2"]
    assert steps[1]["action"].startswith('bash({"command": "cmd1"')
    b = json.loads(json.dumps(body))
    assert F.fill_missing_signatures(b) == 1                      # only the unsigned call (i = 1)
    assert b["contents"][3]["parts"][1]["thoughtSignature"] == F.GEMINI_DUMMY_SIGNATURE
    assert b["contents"][1]["parts"][1]["thoughtSignature"] == "sig"
    assert F.fill_missing_signatures(b, replace_all=True) == 2
    assert F.is_signature_error(400, '{"error": "Function call is missing a thought_signature"}')
    assert not F.is_signature_error(429, "thought signature")


def test_sse_merge_usage_and_price():
    chunks = [{"candidates": [{"content": {"parts": [{"text": "I will list."}]}}]},
              {"candidates": [{"content": {"parts": [{"functionCall": {"name": "bash", "args": {"command": "ls"}}}]},
                               "finishReason": "STOP"}],
               "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 800,
                                 "candidatesTokenCount": 50, "thoughtsTokenCount": 10},
               "modelVersion": "gemini-3.5-flash-lite"}]
    raw = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks).encode()
    p = F.parse_response(F.FORMAT_GEMINI, raw, streamed=True)
    assert p.key == "bash:ls" and p.text == "I will list." and p.model_version == "gemini-3.5-flash-lite"
    assert p.usage == {"input_tokens": 1000, "cached_tokens": 800, "output_tokens": 60}
    assert F.price_cost(GEMINI_PRICES["fast"], p.usage) == pytest.approx((200 * 0.3 + 800 * 0.03 + 60 * 2.5) / 1e6)
    assert F.price_cost(None, p.usage) is None


def test_openai_episode_rewrite_and_usage():
    body = {"model": "deepseek/deepseek-v4-pro", "reasoning": {"enabled": True}, "provider": {"order": ["X"]},
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "the task"},
                         {"role": "assistant", "content": "THOUGHT", "tool_calls": [
                             {"id": "c1", "type": "function", "function": {"name": "bash",
                                                                           "arguments": '{"command": "ls"}'}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "a.py"}]}
    assert F.openai_step_index(body) == 1
    task, steps = F.openai_episode(body)
    assert task == "the task" and steps[0]["output"] == "a.py"
    b = F.openai_rewrite(body, "deepseek/deepseek-v4-flash", {"reasoning": {"enabled": False}})
    assert b["model"] == "deepseek/deepseek-v4-flash" and b["reasoning"] == {"enabled": False}
    assert b["provider"] == {"order": ["X"]} and body["reasoning"] == {"enabled": True}    # copy, not in place
    resp = {"choices": [{"message": {"content": "x", "tool_calls": [
        {"id": "c2", "function": {"name": "bash", "arguments": '{"command": "cat a.py"}'}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "cost": 0.001,
                  "prompt_tokens_details": {"cached_tokens": 64}}}
    p = F.parse_response(F.FORMAT_OPENAI, json.dumps(resp).encode(), streamed=False)
    assert p.key == "bash:cat a.py" and p.usage["reported_cost"] == 0.001 and p.usage["cached_tokens"] == 64


def test_render_state_keeps_the_last_steps_and_the_action_in_progress():
    steps = [{"action": f"bash(cmd{i})", "output": "o" * 50} for i in range(20)]
    st = F.render_state("task", steps, 'bash({"command": "pytest"})', note="run the tests")
    assert len(st["history"]) == F.HISTORY_WINDOW and st["history"][0].startswith("[13] ")
    assert st["action_in_progress"].startswith("bash(") and st["assistant_note"] == "run the tests"
    big = [{"action": "a", "output": "x" * 5000} for _ in range(8)]
    st2 = F.render_state("t" * 9000, big, "b")
    assert len(json.dumps(st2)) <= F.STATE_MAX_CHARS + 200


# ------------------------------------------------------------------------------------------------ policy
def test_route_step_policy_gate_stickiness_and_fail_open():
    st = RP.EpisodeState(arm="gated", episode="e")
    tau, k = 0.6, 2
    out = [RP.route_step(st, 0, 0.9, tau, k)]
    out += [RP.route_step(st, t, c, tau, k) for t, c in [(1, 0.9), (2, 0.8), (3, 0.95), (4, 0.7), (5, 0.5),
                                                          (6, None), (7, 0.61)]]
    assert out == [("strong", "first"), ("fast", "enter"), ("fast", "continue"), ("strong", "cap_k"),
                   ("fast", "enter"), ("strong", "below_tau"), ("strong", "fail_open"), ("fast", "enter")]
    st2 = RP.EpisodeState(arm="gated", episode="e")
    assert RP.route_step(st2, 3, 0.99, None, 4) == ("strong", "no_gate")


# ------------------------------------------------------------------------------------------------ proxy (fake upstream)
class FakeBackend:
    name, model = "fake", "fake-jev"

    def __init__(self, c: float | None = 0.9, delay: float = 0.0, fail: bool = False):
        self.c, self.delay, self.fail, self.states = c, delay, fail, []

    def decide(self, state, questions, ledger=None, decision_point=None):
        import time as _t
        self.states.append(state)
        if self.delay:
            _t.sleep(self.delay)
        if self.fail:
            raise RuntimeError("decision model down")
        q = questions[0]
        assert q.kind == QKind.CHOICE and q.options == ["fast", "strong"]
        dist = {"fast": self.c, "strong": 1 - self.c}
        return DecisionResult(answers=[Answer(question_id=q.id, kind=q.kind, distribution=dist,
                                              value=max(dist, key=dist.get), confidence=max(dist.values()))],
                              usage=Usage(backend="fake", model="fake-jev", input_tokens=100, cost_usd=1e-5))

    def describe(self):
        return {"backend": "fake"}


def fake_upstream(state: dict[str, Any]) -> web.Application:
    """Gemini: model ``m-strong`` answers bash 'ls -la', ``m-fast`` bash 'ls' (or the same with ``same=True``).
    OpenAI: the body's ``model`` decides likewise. ``state['fail_fast']`` -> 500 for the fast model;
    ``state['sig_error']`` -> the first request with a non-placeholder signature gets the HTTP 400 of E25's defect."""
    state.setdefault("seen", [])

    def cmd(model: str) -> str:
        return "ls" if ("fast" in model and not state.get("same")) else "ls -la"

    async def gem(request: web.Request) -> web.StreamResponse:
        mm = request.match_info["mm"]
        model, method = mm.split(":", 1)
        body = await request.json()
        state["seen"].append({"model": model, "method": method, "key": request.headers.get("x-goog-api-key"),
                              "query": dict(request.query), "body": body})
        if state.get("fail_fast") and "fast" in model:
            return web.json_response({"error": {"code": 500, "message": "boom"}}, status=500)
        sigs = [p.get("thoughtSignature") for c in body.get("contents", []) for p in c.get("parts", [])
                if "functionCall" in p]
        if state.get("sig_error") and any(s != F.GEMINI_DUMMY_SIGNATURE for s in sigs):
            state["sig_error"] = False
            return web.json_response({"error": {"code": 400, "message":
                                                "Function call is missing a thought_signature"}}, status=400)
        resp = {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "ok"}, {"functionCall": {"name": "bash", "args": {"command": cmd(model)}},
                             "thoughtSignature": f"sig-{model}"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 500, "candidatesTokenCount": 20},
            "modelVersion": model}
        if method.startswith("stream"):
            out = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await out.prepare(request)
            first = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]}
            rest = {**resp, "candidates": [{"content": {"role": "model", "parts": resp["candidates"][0]["content"]
                                                        ["parts"][1:]}, "finishReason": "STOP"}]}
            for ch in (first, rest):
                await out.write(f"data: {json.dumps(ch)}\r\n\r\n".encode())
            await out.write_eof()
            return out
        return web.json_response(resp)

    async def oai(request: web.Request) -> web.Response:
        body = await request.json()
        state["seen"].append({"model": body["model"], "auth": request.headers.get("Authorization"), "body": body})
        if state.get("fail_fast") and "fast" in body["model"]:
            return web.json_response({"error": {"message": "boom"}}, status=502)
        return web.json_response({"model": body["model"], "choices": [{"message": {
            "role": "assistant", "content": "THOUGHT", "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": "bash",
                                                             "arguments": json.dumps({"command": cmd(body["model"])})}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.002 if "strong" in body["model"]
                      else 0.0005}})

    app = web.Application()
    app.router.add_post("/v1beta/models/{mm}", gem)
    app.router.add_post("/api/v1/chat/completions", oai)
    return app


def _cfg(fmt: str, upstream: str, arms: dict[str, Any], tmp: Path, cap: float | None = None) -> RP.ProxyConfig:
    tiers = {"strong": {"model": "m-strong", "prices": GEMINI_PRICES["strong"]},
             "fast": {"model": "m-fast", "prices": GEMINI_PRICES["fast"]}}
    if fmt == "openai":
        tiers = {"strong": {"model": "m-strong", "overrides": {"reasoning": {"enabled": True}}},
                 "fast": {"model": "m-fast", "overrides": {"reasoning": {"enabled": False}}}}
    return RP.ProxyConfig.from_dict({"format": fmt, "upstream": upstream + ("/v1beta" if fmt == "gemini" else "/api/v1"),
                                     "key_env": "E29_TEST_KEY", "tiers": tiers, "arms": arms, "cap_usd": cap,
                                     "decision": {"wait_s": 2.0, "timeout_s": 2.0}, "retries": 1,
                                     "log_dir": str(tmp / "logs")})


def _run(coro):
    return asyncio.run(coro)


async def _with_proxy(fmt, arms, tmp, backend, script, cap=None, upstate=None):
    upstate = upstate if upstate is not None else {}
    up = TestServer(fake_upstream(upstate))
    await up.start_server()
    cfg = _cfg(fmt, str(up.make_url("")).rstrip("/"), arms, tmp, cap)
    proxy = RP.RouteProxy(cfg, backend=backend, api_key="SECRET-KEY-123")
    proxy.cfg.retries = 1
    client = TestClient(TestServer(proxy.app()))
    await client.start_server()
    try:
        result = await script(client, proxy, upstate)
        await proxy.drain()
    finally:
        await client.close()
        await up.close()
    return result, proxy, upstate


def _logs(tmp: Path, name: str) -> list[dict[str, Any]]:
    p = tmp / "logs" / name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_baseline_mode_forwards_streams_and_keeps_the_key_on_the_host(tmp_path: Path):
    async def script(client, proxy, up):
        r = await client.post("/e29/baseline/trial-1/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse&key=dummy",
                              json=_gem_body(0), headers={"x-goog-api-key": "dummy-from-container"})
        text = await r.text()
        r2 = await client.post("/e29/baseline/trial-1//v1beta//models/gemini-3.5-flash:generateContent",
                               json=_gem_body(1))
        return r.status, text, r2.status, await r2.json()

    (st, text, st2, js), proxy, up = _run(_with_proxy("gemini", {"baseline": {"mode": "strong"}}, tmp_path,
                                                         None, script))
    assert st == 200 and "functionCall" in text and st2 == 200 and js["modelVersion"] == "m-strong"
    assert [s["model"] for s in up["seen"]] == ["m-strong", "m-strong"]
    assert all(s["key"] == "SECRET-KEY-123" for s in up["seen"]) and up["seen"][0]["query"] == {"alt": "sse"}
    calls = _logs(tmp_path, "calls.jsonl")
    assert [c["step"] for c in calls] == [0, 1] and all(c["tier"] == "strong" for c in calls)
    assert calls[0]["streamed"] and calls[0]["key"] == "bash:ls -la"
    assert calls[0]["cost_usd"] == pytest.approx((500 * 1.5 + 500 * 0.15 + 20 * 9.0) / 1e6)
    assert not _logs(tmp_path, "decisions.jsonl")
    assert "SECRET-KEY-123" not in (tmp_path / "logs" / "calls.jsonl").read_text()


def test_gated_mode_decides_ahead_routes_fast_and_caps_segments(tmp_path: Path):
    backend = FakeBackend(c=0.9)

    async def script(client, proxy, up):
        tiers = []
        for t in range(6):
            r = await client.post(f"/e29/gated/ep/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
                                  json=_gem_body(t))
            assert r.status == 200
            await r.text()
            await asyncio.sleep(0.05)            # the harness runs the tool: the decision arrives meanwhile
            tiers.append(_logs(tmp_path, "calls.jsonl")[-1]["tier"])
        return tiers

    tiers, proxy, up = _run(_with_proxy("gemini", {"gated": {"mode": "gated", "tau": 0.5, "k": 2}}, tmp_path,
                                        backend, script))
    assert tiers == ["strong", "fast", "fast", "strong", "fast", "fast"]
    calls = _logs(tmp_path, "calls.jsonl")
    assert [c["route_reason"] for c in calls] == ["first", "enter", "continue", "cap_k", "enter", "continue"]
    assert [c["model"] for c in calls][1] == "m-fast" and all(c["waited_s"] < 1.0 for c in calls)
    dec = _logs(tmp_path, "decisions.jsonl")
    assert [d["step"] for d in dec] == [1, 2, 3, 4, 5, 6] and all(d["c"] == 0.9 for d in dec)
    st = backend.states[0]
    assert st["task"] == "Fix the bug in calc.py" and st["action_in_progress"].startswith("bash(")
    # every function call that reached the upstream carried a signature (E25's fix)
    for s in up["seen"]:
        for c in s["body"]["contents"]:
            for p in c["parts"]:
                if "functionCall" in p:
                    assert p.get("thoughtSignature")


def test_gated_mode_fails_open_on_decision_errors_and_escalates_failed_fast_calls(tmp_path: Path):
    async def script(client, proxy, up):
        for t in range(3):
            r = await client.post("/e29/gated/ep/v1beta/models/x:generateContent", json=_gem_body(t))
            assert r.status == 200
            await asyncio.sleep(0.05)
        return None

    _, _, _ = _run(_with_proxy("gemini", {"gated": {"mode": "gated", "tau": 0.5}}, tmp_path,
                               FakeBackend(fail=True), script))
    calls = _logs(tmp_path, "calls.jsonl")
    assert [c["tier"] for c in calls] == ["strong"] * 3
    assert calls[1]["route_reason"].startswith("fail_open") and calls[1]["c"] is None
    assert all(d.get("error") for d in _logs(tmp_path, "decisions.jsonl"))

    tmp2 = tmp_path / "b"
    _, _, up = _run(_with_proxy("gemini", {"gated": {"mode": "gated", "tau": 0.5}}, tmp2, FakeBackend(c=0.99),
                                script, upstate={"fail_fast": True}))
    calls = _logs(tmp2, "calls.jsonl")
    assert [(c["tier"], c["status"]) for c in calls] == [("strong", 200), ("fast", 500), ("strong", 200),
                                                          ("fast", 500), ("strong", 200)]
    assert calls[2]["route_reason"] == "escalate_fast_error"


def test_signature_error_is_retried_once_with_placeholders(tmp_path: Path):
    async def script(client, proxy, up):
        r = await client.post("/e29/baseline/ep/v1beta/models/x:generateContent", json=_gem_body(2))
        return r.status

    status, _, up = _run(_with_proxy("gemini", {"baseline": {"mode": "strong"}}, tmp_path, None, script,
                                     upstate={"sig_error": True}))
    assert status == 200 and len(up["seen"]) == 2
    calls = _logs(tmp_path, "calls.jsonl")
    assert calls[0]["signature_fix"] is True and calls[0]["attempts"] == 2


def test_shadow_mode_logs_fast_and_second_strong_actions(tmp_path: Path):
    async def script(client, proxy, up):
        for t in range(3):
            r = await client.post("/e29/shadow/cal-1/v1beta/models/x:streamGenerateContent?alt=sse", json=_gem_body(t))
            assert r.status == 200
            await r.text()
            await asyncio.sleep(0.05)
        return None

    _, proxy, up = _run(_with_proxy("gemini", {"shadow": {"mode": "shadow"}}, tmp_path, FakeBackend(c=0.3), script))
    calls = _logs(tmp_path, "calls.jsonl")
    roles = sorted((c["step"], c["role"]) for c in calls)
    assert roles == sorted((t, r) for t in range(3) for r in ("primary", "shadow_fast", "shadow_strong"))
    by = {(c["step"], c["role"]): c for c in calls}
    assert by[(1, "primary")]["tier"] == "strong" and by[(1, "shadow_fast")]["key"] == "bash:ls"
    assert by[(1, "shadow_strong")]["key"] == by[(1, "primary")]["key"] == "bash:ls -la"
    assert all(not by[(t, "shadow_fast")]["streamed"] for t in range(3))
    assert sorted(d["step"] for d in _logs(tmp_path, "decisions.jsonl")) == [1, 2, 3]
    assert proxy.spent > 0


def test_openai_format_rewrites_model_and_overrides_and_reports_cost(tmp_path: Path):
    body = {"model": "deepseek/deepseek-v4-pro", "reasoning": {"enabled": True},
            "messages": [{"role": "user", "content": "task"}], "usage": {"include": True}}

    async def script(client, proxy, up):
        out = []
        for t in range(3):
            b = json.loads(json.dumps(body))
            for i in range(t):
                b["messages"] += [{"role": "assistant", "content": "x", "tool_calls": [
                    {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
                    {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"}]
            r = await client.post("/e29/gated/django__django-1/api/v1/chat/completions", json=b,
                                  headers={"Authorization": "Bearer "})
            out.append((r.status, (await r.json())["usage"]["cost"]))
            await asyncio.sleep(0.05)
        return out

    out, proxy, up = _run(_with_proxy("openai", {"gated": {"mode": "gated", "tau": 0.5}}, tmp_path,
                                      FakeBackend(c=0.8), script))
    assert [s for s, _ in out] == [200, 200, 200] and out[0][1] == 0.002 and out[1][1] == 0.0005
    assert [s["model"] for s in up["seen"]] == ["m-strong", "m-fast", "m-fast"]
    assert up["seen"][1]["body"]["reasoning"] == {"enabled": False} and up["seen"][0]["body"]["reasoning"] == {
        "enabled": True}
    assert all(s["auth"] == "Bearer SECRET-KEY-123" for s in up["seen"])
    calls = _logs(tmp_path, "calls.jsonl")
    assert [c["cost_usd"] for c in calls] == [0.002, 0.0005, 0.0005]


def test_cap_refuses_new_requests_and_counts_earlier_logs(tmp_path: Path):
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "calls.jsonl").write_text(json.dumps({"cost_usd": 0.9}) + "\n")

    async def script(client, proxy, up):
        r = await client.post("/e29/baseline/ep/v1beta/models/x:generateContent", json=_gem_body(0))
        return r.status, await r.json()

    (status, js), proxy, up = _run(_with_proxy("gemini", {"baseline": {"mode": "strong"}}, tmp_path, None, script,
                                               cap=1.0))
    assert status == 503 and "cap" in js["error"]["message"] and not up["seen"] and proxy.n_refused == 1


def test_cap_lets_running_episodes_finish_up_to_the_hard_limit(tmp_path: Path):
    async def script(client, proxy, up):
        out = []
        r = await client.post("/e29/baseline/ep1/v1beta/models/x:generateContent", json=_gem_body(0))
        out.append(r.status)                                   # below the cap: served
        proxy.spent = 1.0                                      # the cap is reached
        for path, body in (("ep1", _gem_body(1)), ("ep2", _gem_body(0)), ("ep3", _gem_body(2))):
            r = await client.post(f"/e29/baseline/{path}/v1beta/models/x:generateContent", json=body)
            out.append(r.status)
        proxy.spent = 5.0                                      # past the hard limit (cap + 2)
        r = await client.post("/e29/baseline/ep1/v1beta/models/x:generateContent", json=_gem_body(2))
        out.append(r.status)
        return out

    out, proxy, up = _run(_with_proxy("gemini", {"baseline": {"mode": "strong"}}, tmp_path, None, script, cap=1.0))
    assert out == [200, 200, 503, 503, 503]                   # ep1 goes on; new ep2 / unknown ep3 refused; then all


def test_unknown_arm_and_wrong_format_are_rejected(tmp_path: Path):
    async def script(client, proxy, up):
        a = await client.post("/e29/nope/ep/v1beta/models/x:generateContent", json=_gem_body(0))
        b = await client.post("/e29/baseline/ep/api/v1/chat/completions", json={"messages": []})
        h = await client.get("/health")
        return a.status, b.status, await h.json()

    (a, b, h), _, _ = _run(_with_proxy("gemini", {"baseline": {"mode": "strong"}}, tmp_path, None, script))
    assert a == 404 and b == 404 and h["ok"] and h["tiers"] == {"strong": "m-strong", "fast": "m-fast"}


# ------------------------------------------------------------------------------------------------ gate fit, go rule
import explore_e29_port as E  # noqa: E402


def _cal_logs(rows: list[tuple[str, int, float, str, str, str]]) -> tuple[list[dict], list[dict]]:
    calls, decs = [], []
    for ep, t, c, kp, kf, ks in rows:
        for role, key, cost in (("primary", kp, 0.01), ("shadow_fast", kf, 0.002), ("shadow_strong", ks, 0.01)):
            calls.append({"arm": "shadow", "episode": ep, "step": t, "role": role, "status": 200, "key": key,
                          "cost_usd": cost})
        decs.append({"arm": "shadow", "episode": ep, "step": t, "c": c})
    return calls, decs


def test_fit_gate_compares_fast_agreement_with_the_strong_llms_own_consistency():
    rows = []
    # high-confidence steps: fast agrees as often as strong agrees with itself; low-confidence: fast disagrees
    for i in range(10):
        rows.append(("a", i + 1, 0.9, "k", "k", "k"))
        rows.append(("b", i + 1, 0.2, "k", "x", "k"))
    rows.append(("a", 0, 0.99, "k", "x", "k"))                 # the first step never counts
    calls, decs = _cal_logs(rows)
    steps = E.shadow_steps(calls, decs)
    assert len(steps) == 20 and all(s["step"] >= 1 for s in steps)
    fit = E.fit_gate(steps)
    assert fit["tau"] == 0.9 and fit["n_routed_sim"] == 10 and fit["sim_fast_share"] == 0.5
    assert fit["routed_agree_fast"] == 1.0 and fit["agree_self"] == 1.0
    assert fit["dC_usd"] == pytest.approx(0.008)
    # fast never as consistent as strong itself: no step pays, the gate routes nothing
    calls, decs = _cal_logs([("a", i + 1, 0.5 + i / 100, "k", "x", "k") for i in range(10)])
    assert E.fit_gate(E.shadow_steps(calls, decs))["tau"] is None
    # strong inconsistent where fast agrees: those steps pay even at a lower confidence
    calls, decs = _cal_logs([("a", 1, 0.3, "k", "k", "z"), ("a", 2, 0.8, "k", "x", "k"), ("a", 3, 0.6, "k", "k", "k")])
    assert E.fit_gate(E.shadow_steps(calls, decs))["tau"] == 0.3


def _pilot(saving: float, base: int, gated: int, n: int = 30) -> dict:
    return {"n": n, "paired": {"saving": saving, "solved_base": base, "solved_arm": gated}}


def test_go_rule_is_the_registered_one():
    assert E.go_rule({"tb": _pilot(0.25, 15, 13), "mswe": _pilot(0.30, 22, 21)})["decision"] == "go"
    r = E.go_rule({"tb": _pilot(0.25, 15, 11), "mswe": _pilot(0.30, 22, 21)})     # tb lost 4 of 30: neither clause
    assert r["per_harness"]["tb"]["pass"] is False and r["per_harness"]["tb"]["not_worse"] is False
    assert r["per_harness"]["mswe"]["pass"] is True and r["decision"] == "no-go"
    assert E.go_rule({"tb": _pilot(0.10, 15, 15), "mswe": _pilot(0.30, 22, 22)})["decision"] == "go"   # 10 % >= 9.6 %
    assert E.go_rule({"tb": _pilot(0.09, 15, 15), "mswe": _pilot(0.30, 22, 22)})["decision"] == "no-go"
    assert E.go_rule({"tb": _pilot(0.19, 15, 15), "mswe": _pilot(0.19, 22, 22)})["decision"] == "no-go"
    assert E.go_rule({"tb": {"n": 0, "no_gate": True}, "mswe": _pilot(0.5, 22, 22)})["decision"] == "no-go"
    assert E.go_rule({"mswe": _pilot(0.5, 22, 22)})["decision"] == "no-go"


def test_selection_lists_are_frozen_disjoint_and_hashed():
    sel = json.loads((E.CFG / "selection.json").read_text())
    for name, meta in sel["files"].items():
        text = (E.CFG / name).read_text()
        assert E.sha256_text(text) == meta["sha256"] and len(E.read_ids(E.CFG / name)) == meta["n"]
    ids = {n: set(E.read_ids(E.CFG / f"{n}.txt")) for n in ("tb_cal", "tb_smoke", "tb_pilot", "tb_eval", "mswe_cal",
                                                           "mswe_pilot", "mswe_eval", "swea_cal", "swea_eval")}
    assert len(ids["tb_cal"]) == 10 and len(ids["mswe_cal"]) == 20 and len(ids["swea_cal"]) == 15
    assert ids["tb_pilot"] <= ids["tb_eval"] and ids["mswe_pilot"] <= ids["mswe_eval"]
    evals = ids["tb_eval"] | ids["mswe_eval"] | ids["swea_eval"]
    assert not (ids["tb_cal"] | ids["tb_smoke"] | ids["mswe_cal"] | ids["swea_cal"]) & evals
    for p in E.SWE_EXCLUDE.values():
        assert not set(E.read_ids(ROOT / p)) & (ids["mswe_cal"] | ids["swea_cal"])
    assert E.read_ids(E.CFG / "tb_pilot.txt") == E.read_ids(ROOT / "configs/explore/e25_harnesses/tb2_order.txt")[:30]


def test_proxy_configs_load_and_use_the_registered_prices():
    import yaml
    prices = yaml.safe_load((ROOT / "configs/prices/models.yaml").read_text())
    flat = {}
    for v in prices.values():
        if isinstance(v, dict):
            flat.update({k: x for k, x in v.items() if isinstance(x, dict)})
    for h in ("tb", "swea", "mswe"):
        cfg = RP.load_config(E.CFG / f"proxy_{h}.json", ['arms={"baseline": {"mode": "strong"}}', "cap_usd=1.5"])
        assert cfg.cap_usd == 1.5 and set(cfg.arms) == {"baseline"}
        if cfg.fmt == "gemini":
            for t in cfg.tiers.values():
                ref = flat[t.model]
                assert t.prices == {k: ref[k] for k in ("input_per_m", "output_per_m", "cached_input_per_m")}
        else:
            assert cfg.tiers["fast"].overrides == {"reasoning": {"enabled": False}}
            assert cfg.tiers["strong"].model == "deepseek/deepseek-v4-pro"


def _harbor_trial(root: Path, arm: str, task: str, reward: float, secs: float, err: str | None = None) -> str:
    name = f"{task}__{abs(hash((arm, task))) % 10**6:06d}"
    d = root / arm / name
    d.mkdir(parents=True)
    r = {"task_name": task, "verifier_result": {"rewards": {"reward": reward}},
         "agent_execution": {"started_at": "2026-10-09T10:00:00Z",
                             "finished_at": f"2026-10-09T10:{int(secs // 60):02d}:{int(secs % 60):02d}Z"}}
    if err:
        r["exception_info"] = {"exception_type": err, "exception_message": "x"}
    (d / "result.json").write_text(json.dumps(r))
    return name


def test_summary_joins_harbor_trials_with_proxy_logs_and_pairs_the_arms(tmp_path: Path):
    run, logs = tmp_path / "run", tmp_path / "proxy"
    logs.mkdir()
    calls, decs = [], []
    for i in range(6):
        task = f"task-{i}"
        for arm, reward, secs, cost, fast in (("baseline", 1.0, 120, 0.5, 0), ("gated", float(i != 0), 100, 0.3, 2)):
            ep = _harbor_trial(run, arm, task, reward, secs)
            for t in range(4):
                tier = "fast" if (arm == "gated" and 1 <= t <= fast) else "strong"
                calls.append({"arm": arm, "episode": ep, "step": t, "role": "primary", "status": 200, "tier": tier,
                              "cost_usd": cost / 4, "route_reason": "enter" if tier == "fast" else "first",
                              "waited_s": 0.01, "input_tokens": 1000, "cached_tokens": 800, "output_tokens": 10})
                if arm == "gated" and t >= 1:
                    decs.append({"arm": arm, "episode": ep, "step": t, "c": 0.9, "cost_usd": 1e-4, "latency_s": 0.2})
    (logs / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    (logs / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decs) + "\n")
    s = E.summarize("tb", run, [logs], [f"task-{i}" for i in range(6)])
    p, g = s["paired"], s["arms"]["gated"]
    assert s["n"] == 6 and p["solved_base"] == 6 and p["solved_arm"] == 5 and p["losses"] == 1
    assert p["saving"] == pytest.approx(1 - (0.3 + 3e-4) / 0.5) and p["time_saving"] == pytest.approx(1 / 6)
    assert g["fast_share"] == pytest.approx(0.5) and g["segments_per_task"] == 1 and g["segment_len_mean"] == 2
    assert g["dm_latency_per_task_s"] == pytest.approx(0.6) and g["cached_share"] == pytest.approx(0.8)
    assert 0 <= p["ni_p"] <= 1 and s["missing"] == {"baseline": [], "gated": []}
    vals = E.tex_values({"full": {"tb": s}, "gates": {"tb": {"tau": 0.8, "n_steps": 100}}})
    assert vals["e29/tb/n"] == "6" and vals["e29/tb/ref_success"] == "100.0" and vals["e29/tb/fast_share"] == "50.0"
    assert vals["e29/gate/tb/tau"] == "0.800"
    E.write_tex(vals, tmp_path / "e29.tex")
    assert "\\defres{e29/tb/saving}" in (tmp_path / "e29.tex").read_text()


def test_sweagent_wrapper_adds_the_proxy_api_base_only_to_run():
    import importlib.util as iu
    spec = iu.spec_from_file_location("jev_sweagent", ROOT / "adapters/swe_agent/jev_sweagent.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    url = "http://host.docker.internal:8092/e29/gated/trial-1/v1beta"
    assert m.proxy_args(["run", "--x=1"], url) == ["run", "--x=1", f"--agent.model.api_base={url}"]
    assert m.proxy_args(["run"], "") == ["run"] and m.proxy_args(["other"], url) == ["other"]
    assert m.proxy_args(["run", "--agent.model.api_base=y"], url) == ["run", "--agent.model.api_base=y"]


def test_mini_swe_model_takes_the_proxy_url():
    pytest.importorskip("minisweagent")
    from jevharness.adapters.mini_swe import OpenRouterTimeoutModel
    m = OpenRouterTimeoutModel(model_name="deepseek/deepseek-v4-pro", api_url="http://127.0.0.1:8093/e29/gated/x/api/v1/chat/completions")
    assert m._api_url.endswith("/e29/gated/x/api/v1/chat/completions")
    assert OpenRouterTimeoutModel(model_name="m")._api_url == "https://openrouter.ai/api/v1/chat/completions"


def test_calibration_descriptives_report_agreement_by_tool_and_answer():
    rows = [("a", 1, 0.9, "bash:ls", "bash:ls -la", "bash:ls"), ("a", 2, 0.2, "read:path:x", "read:path:x", "read:path:y"),
            ("b", 1, 0.7, "bash:pytest", "bash:pytest", "bash:pytest")]
    calls, decs = _cal_logs(rows)
    d = E.calib_descriptives(calls, decs)
    assert d["n"] == 3 and d["all"]["agree_fast"] == pytest.approx(2 / 3) and d["all"]["same_tool_fast"] == 1.0
    assert d["by_tool"]["bash"]["n"] == 2 and d["jev_says_strong"]["n"] == 1 and d["fixed_tau_0p5_share"] == pytest.approx(2 / 3)
    assert d["auroc_c_for_fast_agreement"] == pytest.approx(0.0)        # the agreeing steps had the lower c
