from __future__ import annotations

import json
from typing import Any

from conftest import FakeGemini, FixedDecision, call_reply, start, text_reply

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a command",
                                           "parameters": {"type": "object",
                                                          "properties": {"command": {"type": "string"}},
                                                          "required": ["command"]}}}]


def chat_body(model: str, steps: int, stream: bool = False, task: str = "fix the bug") -> dict[str, Any]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "You are a coding agent."},
                                  {"role": "user", "content": task}]
    for i in range(steps):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps(
                {"command": f"cmd{i}"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"output {i}"})
    return {"model": model, "messages": msgs, "tools": TOOLS, "stream": stream}


def models_called(fake: FakeGemini) -> list[str]:
    return [m for m, _, _ in fake.requests]


def test_route_first_step_strong_then_fast(run, tmp_path):
    fake = FakeGemini(lambda model, body: (200, call_reply("bash", {"command": "ls"})))

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.1))
        try:
            r0 = await client.post("/v1/chat/completions", json=chat_body("jah-auto", 0))
            d0 = await r0.json()
            r1 = await client.post("/v1/chat/completions", json=chat_body("jah-auto", 1))
            d1 = await r1.json()
            return d0, d1
        finally:
            await client.close()
            await up.close()

    d0, d1 = run(go())
    assert models_called(fake) == ["gemini-4-argon", "gemini-3.8-flash"]
    call = d1["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "bash" and json.loads(call["function"]["arguments"]) == {"command": "ls"}
    assert d0["choices"][0]["finish_reason"] == "tool_calls"
    steps = [json.loads(x) for x in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [s["tier"] for s in steps] == ["strong", "fast"]
    assert [s["reason"] for s in steps] == ["first_steps", "routine"]
    assert steps[1]["p_hard"] == 0.1 and steps[0]["cost_usd"] > 0


def test_hard_step_escalates_and_thinking_level(run, tmp_path):
    fake = FakeGemini(lambda model, body: (200, call_reply("bash", {"command": "ls"})))

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.9))
        try:
            await client.post("/v1/chat/completions", json=chat_body("jah-auto", 2))
        finally:
            await client.close()
            await up.close()

    run(go())
    model, body, headers = fake.requests[0]
    assert model == "gemini-4-argon"
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
    assert "authorization" not in {k.lower() for k in headers}


def test_pinned_tiers_skip_the_decision_model(run, tmp_path):
    fake = FakeGemini()
    dec = FixedDecision(0.9)

    async def go():
        client, up, proxy = await start(fake, tmp_path, dec)
        try:
            await client.post("/v1/chat/completions", json=chat_body("jah-fast", 3))
            await client.post("/v1/chat/completions", json=chat_body("jah-strong", 3, task="other"))
        finally:
            await client.close()
            await up.close()

    run(go())
    assert models_called(fake) == ["gemini-3.8-flash", "gemini-4-argon"]
    assert dec.states == []


def test_finish_check_redoes_fast_final_answer_on_strong(run, tmp_path):
    def script(model, body):
        return 200, text_reply("done" if model == "gemini-3.8-flash" else "not yet")

    fake = FakeGemini(script)

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.1))
        try:
            r = await client.post("/v1/chat/completions", json=chat_body("jah-auto", 2))
            return await r.json()
        finally:
            await client.close()
            await up.close()

    d = run(go())
    assert models_called(fake) == ["gemini-3.8-flash", "gemini-4-argon"]
    assert d["choices"][0]["message"]["content"] == "not yet"
    step = json.loads((tmp_path / "steps.jsonl").read_text().splitlines()[-1])
    assert step["reason"] == "finish_check" and len(step["calls"]) == 2


def test_decision_failure_fails_safe_to_strong(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(fail=True))
        try:
            await client.post("/v1/chat/completions", json=chat_body("jah-auto", 2))
        finally:
            await client.close()
            await up.close()

    run(go())
    assert models_called(fake) == ["gemini-4-argon"]
    step = json.loads((tmp_path / "steps.jsonl").read_text())
    assert step["reason"] == "decision_error" and "boom" in step["decision"]["error"]


def test_fast_error_goes_to_strong_once(run, tmp_path):
    def script(model, body):
        if model == "gemini-3.8-flash":
            return 400, {"error": {"message": "bad request"}}
        return 200, call_reply("bash", {"command": "ls"})

    fake = FakeGemini(script)

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.1))
        try:
            r = await client.post("/v1/chat/completions", json=chat_body("jah-auto", 2))
            return r.status
        finally:
            await client.close()
            await up.close()

    assert run(go()) == 200
    assert models_called(fake) == ["gemini-3.8-flash", "gemini-4-argon"]


def test_signatures_round_trip_and_cross_model_placeholder(run, tmp_path):
    """The client drops the signature; the proxy restores it for the same model, uses the placeholder otherwise."""
    replies = iter([call_reply("bash", {"command": "ls"}, sig="SIG-ARGON"),
                    call_reply("bash", {"command": "pwd"}, sig="SIG-FLASH"),
                    call_reply("bash", {"command": "cat x"}, sig="SIG-ARGON-2")])
    fake = FakeGemini(lambda m, b: (200, next(replies)))

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision([0.1, 0.9]))
        try:
            body = chat_body("jah-auto", 0)
            d0 = await (await client.post("/v1/chat/completions", json=body)).json()
            c0 = d0["choices"][0]["message"]["tool_calls"][0]
            c0.pop("extra_content")                       # a client that drops unknown fields
            body["messages"] += [{"role": "assistant", "content": None, "tool_calls": [c0]},
                                 {"role": "tool", "tool_call_id": c0["id"], "content": "a b"}]
            d1 = await (await client.post("/v1/chat/completions", json=body)).json()   # fast tier
            c1 = d1["choices"][0]["message"]["tool_calls"][0]
            body["messages"] += [{"role": "assistant", "content": None, "tool_calls": [c1]},
                                 {"role": "tool", "tool_call_id": c1["id"], "content": "/x"}]
            await client.post("/v1/chat/completions", json=body)                       # strong tier
        finally:
            await client.close()
            await up.close()

    run(go())
    (m1, b1, _), (m2, b2, _) = fake.requests[1], fake.requests[2]
    assert m1 == "gemini-3.8-flash"
    model_parts_1 = [p for c in b1["contents"] if c["role"] == "model" for p in c["parts"]]
    assert model_parts_1[0]["thoughtSignature"] == "skip_thought_signature_validator"   # made by argon
    assert m2 == "gemini-4-argon"
    model_parts_2 = [p for c in b2["contents"] if c["role"] == "model" for p in c["parts"]]
    assert model_parts_2[0]["thoughtSignature"] == "SIG-ARGON"
    assert model_parts_2[1]["thoughtSignature"] == "skip_thought_signature_validator"   # made by flash
    fr = [p for c in b2["contents"] if c["role"] == "user" for p in c["parts"] if "functionResponse" in p]
    assert [p["functionResponse"]["name"] for p in fr] == ["bash", "bash"]


def test_chat_streaming(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))

    async def go():
        client, up, proxy = await start(fake, tmp_path)
        try:
            r = await client.post("/v1/chat/completions", json=chat_body("jah-auto", 0, stream=True))
            return r.headers["Content-Type"], await r.text()
        finally:
            await client.close()
            await up.close()

    ctype, text = run(go())
    assert ctype.startswith("text/event-stream")
    events = [json.loads(ln[6:]) for ln in text.splitlines() if ln.startswith("data: {")]
    calls = [tc for e in events for ch in e["choices"] for tc in ch["delta"].get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "bash" and calls[0]["index"] == 0
    assert events[-1]["usage"]["prompt_tokens"] == 100
    assert text.rstrip().endswith("data: [DONE]")


def test_budget_refuses_new_sessions(run, tmp_path):
    fake = FakeGemini()

    async def go():
        client, up, proxy = await start(fake, tmp_path)
        proxy.cfg.budget_usd = 0.0
        try:
            r = await client.post("/v1/chat/completions", json=chat_body("jah-auto", 0))
            return r.status
        finally:
            await client.close()
            await up.close()

    assert run(go()) == 429
    assert fake.requests == []


def test_rules_backend(run, tmp_path):
    from jevagentharness.decision import RulesBackend, StepView
    rb = RulesBackend()
    tb = "Traceback (most recent call last):\n x"
    assert rb.judge(StepView(last_action="a", last_output=tb))[1] == "error_signal"
    assert rb.judge(StepView(last_action="a", prev_action="a"))[1] == "repeated_action"
    assert rb.judge(StepView(last_action="a", last_output="12 passed"))[1] == "no_signal"


def test_models_and_stats(run, tmp_path):
    fake = FakeGemini()

    async def go():
        client, up, proxy = await start(fake, tmp_path)
        try:
            m = await (await client.get("/v1/models")).json()
            s = await (await client.get("/stats")).json()
            return m, s
        finally:
            await client.close()
            await up.close()

    m, s = run(go())
    assert {x["id"] for x in m["data"]} >= {"jah-auto", "jah-strong", "jah-fast"}
    assert s["tiers"]["strong"]["model"] == "gemini-4-argon"


def test_explicit_cache_for_fast_tier(run, tmp_path):
    """The fast tier (cache: explicit) caches the session's history and sends only the new turns afterwards."""
    import asyncio
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"}, sig=None)))
    big = "x" * 40000                                  # a long task prompt (~11k tokens)

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.1))
        try:
            for steps in (1, 2, 3):
                await client.post("/v1/chat/completions", json=chat_body("jah-auto", steps, task=big))
                await asyncio.sleep(0.05)                 # let the background cache write finish
        finally:
            await client.close()
            await up.close()

    run(go())
    bodies = [b for m, b, _ in fake.requests if m == "gemini-3.8-flash"]
    assert "cachedContent" not in bodies[0] and "systemInstruction" in bodies[0]
    assert bodies[1]["cachedContent"] == "cachedContents/c0" and "systemInstruction" not in bodies[1]
    assert len(bodies[1]["contents"]) < len(bodies[0]["contents"]) + 2
    cached = fake.caches["cachedContents/c0"]
    assert cached["model"] == "models/gemini-3.8-flash" and "tools" in cached
    recs = [json.loads(x) for x in (tmp_path / "steps.jsonl").read_text().splitlines()]
    cache_recs = [r for r in recs if r.get("kind") == "cache"]
    assert cache_recs and cache_recs[0]["cost_usd"] > 0


def test_model_name_overrides_tau(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.6))
        proxy.cfg.models["jah-auto-t75"] = {"policy": "route", "tau": 0.75}
        try:
            await client.post("/v1/chat/completions", json=chat_body("jah-auto", 2))
            await client.post("/v1/chat/completions", json=chat_body("jah-auto-t75", 2))
        finally:
            await client.close()
            await up.close()

    run(go())
    assert models_called(fake) == ["gemini-4-argon", "gemini-3.8-flash"]   # p_hard 0.6: hard at 0.5, routine at 0.75


def test_model_name_selects_fast_tier(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.1))
        from jevagentharness.config import Tier
        proxy.cfg.tiers["fast_med"] = Tier(name="fast_med", upstream="gemini", model="gemini-3.8-flash",
                                           effort="medium")
        proxy.cfg.models["jah-auto-m"] = {"policy": "route", "fast": "fast_med"}
        try:
            await client.post("/v1/chat/completions", json=chat_body("jah-auto-m", 2))
        finally:
            await client.close()
            await up.close()

    run(go())
    model, body, _ = fake.requests[0]
    assert model == "gemini-3.8-flash" and body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "medium"
    step = json.loads((tmp_path / "steps.jsonl").read_text())
    assert step["tier"] == "fast_med" and step["reason"] == "routine"


def test_log_states_writes_decision_inputs(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.2))
        proxy.cfg.log_states = True
        try:
            await client.post("/v1/chat/completions", json=chat_body("jah-auto", 2, task="fix the parser bug"))
        finally:
            await client.close()
            await up.close()

    run(go())
    st = [json.loads(x) for x in (tmp_path / "states.jsonl").read_text().splitlines()]
    assert len(st) == 1 and "fix the parser bug" in st[0]["state"] and st[0]["tier"] == "fast"
    steps = (tmp_path / "steps.jsonl").read_text()
    assert "fix the parser bug" not in steps and "_state" not in steps      # the step log stays text-free
