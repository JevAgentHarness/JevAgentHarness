"""Calls to the LLM providers behind the tiers (retries, Gemini's signature fix, cost)."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector

from ..config import Config, Tier
from ..prices import PriceTable
from ..types import Request, Response
from . import gemini as G
from . import openai_chat as O
from . import openai_responses as OR
from .gemini_cache import GeminiCache

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class UpstreamError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"upstream HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


@dataclass
class CallInfo:
    tier: str
    model: str
    latency_s: float
    attempts: int
    cost_usd: float
    priced: bool
    signature_fix: bool = False


class Upstreams:
    def __init__(self, cfg: Config, prices: PriceTable | None = None,
                 cost_sink: Any = None) -> None:
        self.cfg = cfg
        self.prices = prices or PriceTable(cfg.prices)
        self.sigs = G.SignatureStore()
        self.reasoning = OR.ReasoningStore()
        self._client: ClientSession | None = None
        self.caches: dict[str, GeminiCache] = {}
        self.cost_sink = cost_sink          # (session, usd, info) for cache writes; set by the proxy

    def cache_for(self, upstream: str, session: str | None = None) -> GeminiCache:
        """One explicit-cache manager per (upstream, pool key): a context cache lives in the project of its key."""
        up = self.cfg.upstreams[upstream]
        ki = up.key_index(session)
        name = f"{upstream}#{ki}" if ki else upstream
        if name not in self.caches:
            key = up.key(session)

            async def post(path: str, body: dict[str, Any] | None, method: str) -> tuple[int, str]:
                cl = await self.client()
                headers = {"Content-Type": "application/json", "x-goog-api-key": key, **up.headers}
                url = f"{up.base_url.rstrip('/')}{path}"
                async with cl.request(method, url, data=None if body is None else json.dumps(body), headers=headers,
                                      timeout=ClientTimeout(total=120, sock_connect=30)) as r:
                    return r.status, await r.text()

            self.caches[name] = GeminiCache(post, cost_sink=lambda s, usd, info: (
                self.cost_sink(s, usd, info) if self.cost_sink else None))
        return self.caches[name]

    async def client(self) -> ClientSession:
        if self._client is None or self._client.closed:
            ctx: Any = True
            ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
            if ca and Path(ca).exists():
                ctx = ssl.create_default_context(cafile=ca)
            self._client = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30),
                                         connector=TCPConnector(limit=512, ssl=ctx), trust_env=True)
        return self._client

    async def close(self) -> None:
        for c in self.caches.values():
            await c.close()
        if self._client is not None:
            await self._client.close()

    def _wire(self, tier: Tier, req: Request, session: str | None = None) -> tuple[str, dict[str, str], dict[str, Any]]:
        up = self.cfg.upstreams[tier.upstream]
        effort = tier.effort if tier.effort is not None else req.reasoning
        if up.type == "gemini":
            url = f"{up.base_url.rstrip('/')}/models/{tier.model}:generateContent"
            headers = {"Content-Type": "application/json", "x-goog-api-key": up.key(session), **up.headers}
            return url, headers, G.to_gemini(req, tier.model, effort, self.sigs, tier.extra)
        headers = {"Content-Type": "application/json", **up.headers}
        if up.key(session):
            headers["Authorization"] = f"Bearer {up.key(session)}"
        if up.type == "openai_responses":
            return (f"{up.base_url.rstrip('/')}/responses", headers,
                    OR.to_responses(req, tier.model, effort, self.reasoning, session, tier.extra))
        url = f"{up.base_url.rstrip('/')}/chat/completions"
        return url, headers, O.to_chat(req, tier.model, effort, tier.extra, up.effort_field)

    def _parse(self, tier: Tier, data: dict[str, Any], req: Request) -> Response:
        up = self.cfg.upstreams[tier.upstream]
        if up.type == "gemini":
            r = G.from_gemini(data, tier.model, self.sigs)
        elif up.type == "openai_responses":
            r = OR.from_responses(data, tier.model, self.reasoning)
        else:
            r = O.from_chat(data, tier.model, {t.name for t in req.tools if t.kind == "custom"})
        custom = {t.name for t in req.tools if t.kind == "custom"}
        for c in r.tool_calls:                       # freeform tools went out as {"input": str} functions
            if c.name in custom and c.kind != "custom":
                c.kind = "custom"
                try:
                    c.arguments = str(json.loads(c.arguments).get("input", ""))
                except (json.JSONDecodeError, AttributeError):
                    pass
        return r

    async def call(self, tier_name: str, req: Request, session: str | None = None) -> tuple[Response, CallInfo]:
        tier = self.cfg.tiers[tier_name]
        up = self.cfg.upstreams[tier.upstream]
        req = _custom_as_function(req)
        url, headers, full = self._wire(tier, req, session)
        explicit = up.type == "gemini" and tier.cache == "explicit" and bool(session)
        cache = self.cache_for(tier.upstream, session) if explicit else None
        body, entry = cache.apply(session, tier.model, full) if cache is not None else (full, None)
        t0 = time.monotonic()
        delay, status, text, sig_fix = 2.0, 0, "", False
        cl = await self.client()
        for attempt in range(up.retries + 2):
            try:
                async with cl.post(url, data=json.dumps(body), headers=headers,
                                   timeout=ClientTimeout(total=up.timeout_s, sock_connect=30)) as r:
                    status, text = r.status, await r.text()
            except (ClientError, asyncio.TimeoutError, OSError) as e:
                status, text = 599, f"{type(e).__name__}: {e}"
            if status == 200:
                try:
                    resp = self._parse(tier, json.loads(text), req)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    raise UpstreamError(502, f"unparseable upstream reply: {e}: {text[:200]}") from e
                cost, priced = self.prices.cost(tier.model, resp.usage)
                if cache is not None and session and not req.modified:
                    # a request a decision point changed (forced or filtered tools, a note) is a one-off: a cache
                    # made from it would not be a prefix of the next ordinary request
                    cache.after(session, tier.model, full, self.prices)
                return resp, CallInfo(tier=tier_name, model=tier.model, latency_s=time.monotonic() - t0,
                                      attempts=attempt + 1, cost_usd=cost, priced=priced, signature_fix=sig_fix)
            if entry is not None and status in (400, 403, 404) and "cache" in text.lower():
                cache.invalidate(session, tier.model)         # the cache expired or was deleted: send in full
                body, entry = full, None
                continue
            if up.type == "gemini" and not sig_fix and G.is_signature_error(status, text):
                G.strip_signatures(body)
                if body is not full:
                    G.strip_signatures(full)
                sig_fix = True
                continue
            if status in RETRY_STATUS and attempt < up.retries:
                await asyncio.sleep(delay)
                delay *= 2.5
                continue
            break
        raise UpstreamError(status, text)


def _custom_as_function(req: Request) -> Request:
    """Freeform (custom) tools have no schema; providers that only take functions get one string argument."""
    if not any(t.kind == "custom" for t in req.tools):
        return req
    from dataclasses import replace

    from ..types import Tool
    tools = [t if t.kind != "custom" else Tool(
        name=t.name, kind="custom", meta=t.meta,
        description=(t.description + "\n\nPut the complete raw input for this tool in the `input` argument."
                     + (f"\nFormat: {t.meta.get('format')}" if t.meta.get("format") else "")).strip(),
        parameters={"type": "object", "properties": {"input": {"type": "string", "description": "raw tool input"}},
                    "required": ["input"]}) for t in req.tools]
    return replace(req, tools=tools)
