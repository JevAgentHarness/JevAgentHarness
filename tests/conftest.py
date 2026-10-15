"""A scripted fake Gemini upstream and helpers to run the proxy against it (offline; no keys, no network)."""

from __future__ import annotations

import asyncio
import json
import time
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


class ScriptedDecision(DecisionBackend):
    """Answers per question id: ``answers[qid]`` is an option name (probability 1), a {option: p} dict, a list of
    those (consumed in order, the last one repeats) or a callable (state, question) -> one of those. Unscripted:
    routine for the routing question, uniform otherwise. ``fail``: question ids that raise; ``delay``: seconds per
    question id (or "*") before answering."""
    name = "scripted"

    def __init__(self, answers: dict[str, Any] | None = None, fail: set[str] | None = None,
                 delay: dict[str, float] | None = None) -> None:
        self.answers = dict(answers or {})
        self.fail = set(fail or ())
        self.delay = dict(delay or {})
        self.asked: list[tuple[str, str, list[str]]] = []          # (question id, state, options)
        self.events: list[tuple[str, str, float]] = []             # (question id, "start" | "end", time)

    async def decide(self, state: str, q: Question) -> Decision:
        self.asked.append((q.id, state, list(q.options)))
        self.events.append((q.id, "start", time.monotonic()))
        d = self.delay.get(q.id, self.delay.get("*", 0.0))
        if d:
            await asyncio.sleep(d)
        self.events.append((q.id, "end", time.monotonic()))
        if q.id in self.fail:
            raise RuntimeError(f"boom {q.id}")
        a = self.answers.get(q.id)
        if callable(a):
            a = a(state, q)
        if isinstance(a, list):
            a = a.pop(0) if len(a) > 1 else a[0]
        if a is None:
            a = {"routine": 0.9, "hard": 0.1} if q.id == "next_step" else {o: 1 / len(q.options) for o in q.options}
        if isinstance(a, str):
            a = {a: 1.0}
        return Decision(p={o: float(a.get(o, 0.0)) for o in q.options}, model="scripted", cost_usd=0.0001)

    def ids(self) -> list[str]:
        return [q for q, _, _ in self.asked]


def make_cfg(base_url: str, tmp_path: Any, sets: list[str] | None = None, **router: Any) -> Any:
    base = [f"upstreams.gemini.base_url={base_url}/v1beta", "upstreams.gemini.key_env=FAKE_GEMINI_KEY",
            f"log_dir={tmp_path}", "upstreams.gemini.retries=1",
            # the tests were written for this pair; pin it so a change of the built-in default does not move them
            'tiers.fast={"upstream": "gemini", "model": "gemini-3.8-flash", "effort": "low", "cache": "explicit"}',
            "router.tau=0.5"]
    return load_config(None, base + [f"router.{k}={json.dumps(v)}" for k, v in router.items()] + list(sets or []))


@pytest.fixture
def run() -> Callable[..., Any]:
    def _run(coro: Any) -> Any:
        return asyncio.run(coro)
    return _run


async def start(fake: FakeGemini, tmp_path: Any, backend: DecisionBackend | None = None,
                sets: list[str] | None = None, **router: Any) -> tuple[TestClient, TestServer, Proxy]:
    upstream = TestServer(fake.app())
    await upstream.start_server()
    cfg = make_cfg(str(upstream.make_url("")).rstrip("/"), tmp_path, sets, **router)
    proxy = Proxy(cfg, backend=backend or FixedDecision())
    client = TestClient(TestServer(proxy.app()))
    await client.start_server()
    return client, upstream, proxy
