"""The OpenAI Responses upstream against a scripted fake ``/v1/responses`` server (offline)."""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from conftest import FixedDecision
from jevagentharness.config import Config
from jevagentharness.prices import PriceTable
from jevagentharness.server import Proxy
from jevagentharness.types import Usage


class FakeResponses:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    async def handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        return web.json_response(self.replies.pop(0))

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/responses", self.handle)
        return app


def reply(model: str, call: tuple[str, str, dict[str, Any]] | None = None, text: str = "", reasoning: bool = False,
          cached: int = 0, written: int = 0) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    if reasoning:
        out.append({"type": "reasoning", "id": f"rs_{model}", "summary": [], "encrypted_content": f"ENC-{model}"})
    if call:
        out.append({"type": "function_call", "call_id": call[0], "name": call[1], "arguments": json.dumps(call[2])})
    if text:
        out.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    return {"id": "resp_1", "model": model, "status": "completed", "output": out,
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": cached,
                                                                     "cache_write_tokens": written},
                      "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 20}}}


def cfg(base: str, tmp: Any) -> Config:
    return Config.from_dict({
        "log_dir": str(tmp),
        "upstreams": {"openai": {"type": "openai_responses", "base_url": base + "/v1", "key_env": "", "retries": 0}},
        "tiers": {"strong": {"upstream": "openai", "model": "gpt-6.1-sol", "effort": "medium"},
                  "fast": {"upstream": "openai", "model": "gpt-6-luna", "effort": "low"}},
        "router": {"decision": {"type": "rules"}},
        "models": {"jah-auto": {"policy": "route"}, "jah-strong": {"tier": "strong"}},
    })


TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}}}}]


def test_chat_client_to_responses_upstream(run, tmp_path):
    """Request shape, reasoning items returned to the same model only, tool results, cost with cache writes."""
    fake = FakeResponses([
        reply("gpt-6.1-sol", ("c1", "bash", {"command": "ls"}), reasoning=True, written=900),
        reply("gpt-6-luna", ("c2", "bash", {"command": "cat a"}), cached=900),
        reply("gpt-6.1-sol", text="done", cached=900),
    ])
    sys_ = {"role": "system", "content": "You are an agent."}
    user = {"role": "user", "content": "fix it"}

    async def go() -> list[dict[str, Any]]:
        dec = FixedDecision([0.1, 0.9])
        outs = []
        up = TestServer(fake.app())
        await up.start_server()
        proxy = Proxy(cfg(str(up.make_url("")).rstrip("/"), tmp_path), backend=dec)
        client = TestClient(TestServer(proxy.app()))
        await client.start_server()
        try:
            msgs = [sys_, user]
            for _ in range(3):
                r = await client.post("/s/sess-1/v1/chat/completions",
                                      json={"model": "jah-auto", "messages": msgs, "tools": TOOLS})
                d = await r.json()
                outs.append(d)
                m = d["choices"][0]["message"]
                if not m.get("tool_calls"):
                    break
                c = m["tool_calls"][0]
                c.pop("extra_content", None)                       # a client that drops unknown fields
                msgs = msgs + [{"role": "assistant", "content": None, "tool_calls": [c]},
                               {"role": "tool", "tool_call_id": c["id"], "content": "output of " + c["id"]}]
        finally:
            await client.close()
            await up.close()
        return outs

    outs = run(go())
    r0, r1, r2 = fake.requests
    # step 0: strong, instructions + tools + reasoning + cache key
    assert r0["model"] == "gpt-6.1-sol" and r0["instructions"] == "You are an agent."
    assert r0["reasoning"] == {"effort": "medium"} and r0["include"] == ["reasoning.encrypted_content"]
    assert r0["store"] is False and r0["prompt_cache_key"].endswith("sess-1")
    assert r0["tools"][0]["type"] == "function" and r0["tools"][0]["name"] == "bash"
    # step 1: routed to luna; sol's reasoning item is NOT sent to a different model
    assert r1["model"] == "gpt-6-luna" and r1["reasoning"] == {"effort": "low"}
    kinds1 = [it["type"] for it in r1["input"]]
    assert "reasoning" not in kinds1 and kinds1[-2:] == ["function_call", "function_call_output"]
    assert r1["input"][-1] == {"type": "function_call_output", "call_id": "c1", "output": "output of c1"}
    # step 2: back on sol; sol's own reasoning item goes back in, before its function call
    assert r2["model"] == "gpt-6.1-sol"
    kinds2 = [it["type"] for it in r2["input"]]
    i = kinds2.index("reasoning")
    assert r2["input"][i]["encrypted_content"] == "ENC-gpt-6.1-sol" and kinds2[i + 1] == "function_call"
    assert r2["input"][i + 1]["call_id"] == "c1"
    assert outs[2]["choices"][0]["message"]["content"] == "done"
    # cost: 1000 input (900 written at the cache-write rate on step 0), 50 output incl. 20 reasoning
    steps = [json.loads(x) for x in (tmp_path / "steps.jsonl").read_text().splitlines()]
    exp0 = (100 * 2.00 + 900 * 2.50 + 50 * 10.00) / 1e6
    exp1 = (100 * 0.10 + 900 * 0.01 + 50 * 0.50) / 1e6
    assert abs(steps[0]["llm_cost_usd"] - exp0) < 1e-9 and abs(steps[1]["llm_cost_usd"] - exp1) < 1e-9
    assert steps[0]["cache_write_tokens"] == 900 and steps[1]["cached_tokens"] == 900
    assert steps[0]["reasoning_tokens"] == 20 and steps[0]["output_tokens"] == 30


def test_price_with_cache_writes():
    p = PriceTable()
    usd, priced = p.cost("gpt-6.1-sol", Usage(input_tokens=10_000, cached_tokens=6_000, cache_write_tokens=3_000,
                                              output_tokens=100, reasoning_tokens=400))
    assert priced and abs(usd - (1_000 * 2.0 + 6_000 * 0.10 + 3_000 * 2.50 + 500 * 10.0) / 1e6) < 1e-12
    # Gemini rows have no cache-write rate and report none: unchanged formula
    usd2, _ = p.cost("gemini-3.8-flash", Usage(input_tokens=1000, cached_tokens=500, output_tokens=10))
    assert abs(usd2 - (500 * 0.75 + 500 * 0.075 + 10 * 3.75) / 1e6) < 1e-12


def test_codex_custom_tool_through_responses_upstream(run, tmp_path):
    """Codex (Responses client) with the freeform apply_patch tool -> OpenAI Responses upstream -> custom call back."""
    fake = FakeResponses([reply("gpt-6.1-sol", ("c9", "apply_patch", {"input": "*** Begin Patch\n*** End Patch\n"}))])

    async def go() -> str:
        up = TestServer(fake.app())
        await up.start_server()
        proxy = Proxy(cfg(str(up.make_url("")).rstrip("/"), tmp_path), backend=FixedDecision(0.1))
        client = TestClient(TestServer(proxy.app()))
        await client.start_server()
        try:
            r = await client.post("/v1/responses", json={
                "model": "jah-strong", "stream": True, "store": False, "prompt_cache_key": "k",
                "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "sys"}]},
                          {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}],
                "tools": [{"type": "custom", "name": "apply_patch", "description": "patch",
                           "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}}]})
            return await r.text()
        finally:
            await client.close()
            await up.close()

    text = run(go())
    sent = fake.requests[0]
    assert sent["tools"][0]["name"] == "apply_patch" and "input" in sent["tools"][0]["parameters"]["properties"]
    items = [json.loads(ln[6:])["item"] for ln in text.splitlines()
             if ln.startswith("data: ") and '"response.output_item.done"' in ln]
    assert items[0]["type"] == "custom_tool_call" and items[0]["input"].startswith("*** Begin Patch")
