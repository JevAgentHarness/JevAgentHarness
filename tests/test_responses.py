from __future__ import annotations

import json

from conftest import FakeGemini, FixedDecision, call_reply, start, text_reply
from jevagentharness.dialects import responses as R


def codex_body(stream: bool = True, history: bool = False) -> dict:
    items = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are Codex."}]},
             {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix calc.py"}]}]
    if history:
        items += [{"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
                  {"type": "function_call", "call_id": "call_1", "name": "exec_command",
                   "arguments": json.dumps({"cmd": "cat calc.py"})},
                  {"type": "function_call_output", "call_id": "call_1", "output": "def add(a,b): return a-b"},
                  {"type": "function_call", "call_id": "call_2", "name": "spawn", "namespace": "multi_agent_v1",
                   "arguments": "{}"},
                  {"type": "function_call_output", "call_id": "call_2", "output": [{"type": "input_text",
                                                                                     "text": "ok"}]}]
    return {"model": "jah-auto", "input": items, "stream": stream, "store": False, "prompt_cache_key": "sess-1",
            "tools": [{"type": "function", "name": "exec_command", "strict": False, "description": "run",
                       "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                      "required": ["cmd"], "additionalProperties": False}},
                      {"type": "custom", "name": "apply_patch", "description": "patch",
                       "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}},
                      {"type": "namespace", "name": "multi_agent_v1", "description": "agents",
                       "tools": [{"type": "function", "name": "spawn", "parameters": {"type": "object"}}]},
                      {"type": "web_search"}],
            "tool_choice": "auto", "parallel_tool_calls": True, "reasoning": {"effort": "medium"},
            "include": ["reasoning.encrypted_content"]}


def test_parse_request_shapes():
    req = R.parse_request(codex_body(history=True))
    assert req.system == "You are Codex."
    assert [t.name for t in req.tools] == ["exec_command", "apply_patch", "multi_agent_v1__spawn"]
    assert req.tools[1].kind == "custom"
    a = [m for m in req.messages if m.role == "assistant"]
    assert [c.name for m in a for c in m.tool_calls] == ["exec_command", "multi_agent_v1__spawn"]
    assert [m.text for m in req.messages if m.role == "tool"] == ["def add(a,b): return a-b", "ok"]
    assert req.session == "sess-1" and req.step_index() == 2


def events(text: str) -> list[dict]:
    return [json.loads(ln[6:]) for ln in text.splitlines() if ln.startswith("data: ")]


def test_codex_stream_function_call_and_custom_tool(run, tmp_path):
    replies = iter([call_reply("exec_command", {"cmd": "ls"}),
                    call_reply("apply_patch", {"input": "*** Begin Patch\n*** End Patch\n"}),
                    call_reply("multi_agent_v1__spawn", {"task": "x"})])
    fake = FakeGemini(lambda m, b: (200, next(replies)))

    async def go():
        client, up, proxy = await start(fake, tmp_path, FixedDecision(0.1))
        try:
            outs = []
            for hist in (False, True, True):
                r = await client.post("/v1/responses", json=codex_body(history=hist))
                outs.append(await r.text())
            return outs
        finally:
            await client.close()
            await up.close()

    t0, t1, t2 = run(go())
    e0 = events(t0)
    assert e0[0]["type"] == "response.created" and e0[-1]["type"] == "response.completed"
    done = [e["item"] for e in e0 if e["type"] == "response.output_item.done"]
    assert done[0]["type"] == "function_call" and json.loads(done[0]["arguments"]) == {"cmd": "ls"}
    assert e0[-1]["response"]["usage"]["input_tokens"] == 100
    seqs = [e["sequence_number"] for e in e0]
    assert seqs == sorted(seqs)
    d1 = [e["item"] for e in events(t1) if e["type"] == "response.output_item.done"][0]
    assert d1["type"] == "custom_tool_call" and d1["input"].startswith("*** Begin Patch")
    d2 = [e["item"] for e in events(t2) if e["type"] == "response.output_item.done"][0]
    assert d2["name"] == "spawn" and d2["namespace"] == "multi_agent_v1"
    # upstream saw: the freeform tool as a one-string function, hosted web_search dropped, developer -> system
    _, body, _ = fake.requests[0]
    names = [f["name"] for f in body["tools"][0]["functionDeclarations"]]
    assert names == ["exec_command", "apply_patch", "multi_agent_v1__spawn"]
    assert body["systemInstruction"]["parts"][0]["text"] == "You are Codex."


def test_codex_text_answer_non_stream(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, text_reply("All tests pass.")))

    async def go():
        client, up, proxy = await start(fake, tmp_path)
        try:
            r = await client.post("/v1/responses", json=codex_body(stream=False))
            return await r.json()
        finally:
            await client.close()
            await up.close()

    d = run(go())
    assert d["status"] == "completed" and d["output"][0]["content"][0]["text"] == "All tests pass."


def test_codex_stream_error_is_response_failed(run, tmp_path):
    fake = FakeGemini(lambda m, b: (400, {"error": {"message": "bad"}}))

    async def go():
        client, up, proxy = await start(fake, tmp_path)
        try:
            r = await client.post("/v1/responses", json=codex_body())
            return await r.text()
        finally:
            await client.close()
            await up.close()

    ev = events(run(go()))
    assert ev[-1]["type"] == "response.failed"
