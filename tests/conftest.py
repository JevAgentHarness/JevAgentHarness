"""A scripted fake Gemini upstream and helpers to run the proxy against it (offline; no keys, no network)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from jevagentharness.config import load_config
from jevagentharness.decision import Decision, DecisionBackend, Question
from jevagentharness.server import Proxy


class FakeGemini:
    """Records every request; answers with ``script(model, body)`` -> (status, json)."""

    def __init__(self, script: Callable[[str, dict[str, Any]], tuple[int, dict[str, Any]]] | None = None) -> None:
        self.requests: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.script = script or (lambda model, body: (200, text_reply("ok")))
        self.caches: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    async def create_cache(self, request: web.Request) -> web.Response:
        body = await request.json()
        name = f"cachedContents/c{len(self.caches)}"
        self.caches[name] = body
        return web.json_response({"name": name, "usageMetadata": {"totalTokenCount": 5000}})

    async def delete_cache(self, request: web.Request) -> web.Response:
        self.deleted.append("cachedContents/" + request.match_info["cid"])
        return web.json_response({})

    async def handle(self, request: web.Request) -> web.Response:
        model = request.match_info["model"]
        body = await request.json()
        self.requests.append((model, body, dict(request.headers)))
        status, data = self.script(model, body)
        return web.json_response(data, status=status)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1beta/models/{model}:generateContent", self.handle)
        app.router.add_post("/v1beta/cachedContents", self.create_cache)
        app.router.add_delete("/v1beta/cachedContents/{cid}", self.delete_cache)
        return app


def usage(prompt: int = 100, cached: int = 0, out: int = 10, thoughts: int = 0) -> dict[str, Any]:
    return {"promptTokenCount": prompt, "cachedContentTokenCount": cached, "candidatesTokenCount": out,
            "thoughtsTokenCount": thoughts}


def text_reply(text: str, **u: int) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": usage(**u)}


def call_reply(name: str, args: dict[str, Any], sig: str | None = "SIG", **u: int) -> dict[str, Any]:
    part: dict[str, Any] = {"functionCall": {"name": name, "args": args}}
    if sig:
        part["thoughtSignature"] = sig
    return {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}],
            "usageMetadata": usage(**u)}


class FixedDecision(DecisionBackend):
    name = "fixed"

    def __init__(self, p_hard: float | list[float] = 0.1, fail: bool = False) -> None:
        self.p = list(p_hard) if isinstance(p_hard, list) else None
        self.p0 = p_hard if not isinstance(p_hard, list) else 0.1
        self.fail = fail
        self.states: list[str] = []

    async def decide(self, state: str, q: Question) -> Decision:
        self.states.append(state)
        if self.fail:
            raise RuntimeError("boom")
        p = self.p.pop(0) if self.p else self.p0
        return Decision(p={"hard": p, "routine": 1 - p}, model="fixed", cost_usd=0.0001)


def make_cfg(base_url: str, tmp_path: Any, **router: Any) -> Any:
    sets = [f"upstreams.gemini.base_url={base_url}/v1beta", "upstreams.gemini.key_env=FAKE_GEMINI_KEY",
            f"log_dir={tmp_path}", "upstreams.gemini.retries=1",
            # the tests were written for this pair; pin it so a change of the built-in default does not move them
            'tiers.fast={"upstream": "gemini", "model": "gemini-3.8-flash", "effort": "low", "cache": "explicit"}',
            "router.tau=0.5"]
    sets += [f"router.{k}={json.dumps(v)}" for k, v in router.items()]
    return load_config(None, sets)


@pytest.fixture
def run() -> Callable[..., Any]:
    def _run(coro: Any) -> Any:
        return asyncio.run(coro)
    return _run


async def start(fake: FakeGemini, tmp_path: Any, backend: DecisionBackend | None = None, **router: Any
                ) -> tuple[TestClient, TestServer, Proxy]:
    upstream = TestServer(fake.app())
    await upstream.start_server()
    cfg = make_cfg(str(upstream.make_url("")).rstrip("/"), tmp_path, **router)
    proxy = Proxy(cfg, backend=backend or FixedDecision())
    client = TestClient(TestServer(proxy.app()))
    await client.start_server()
    return client, upstream, proxy
