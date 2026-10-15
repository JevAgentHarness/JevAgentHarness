"""Explicit context caching for Gemini tiers (``cache: explicit`` on a tier).

An agent re-sends its whole history at every step. Gemini bills cached input at a tenth of the input price, but its
implicit cache does not hit reliably for every model. For a tier with ``cache: explicit`` the proxy keeps one
``cachedContents`` entry per session: the system instruction, the tools and the conversation up to some turn. A
request whose history starts with exactly that prefix is sent as ``cachedContent`` plus the new turns only. After a
response, once the uncached tail has grown past the refresh point, the proxy caches the request's full history in the
background (the next request extends it), and deletes the cache it replaces.

Caches cannot be extended, so every refresh re-writes the whole history at the full input price. With a cached prefix
of P tokens and a history that grows by g tokens per step, refreshing every k steps costs about P/k + k*g/2 full-price
tokens per step, which is smallest at k = sqrt(2P/g): the refresh point is a tail of sqrt(2*P*g) tokens (at least
``min_refresh_tokens``).

Costs are reported through ``cost_sink(session, usd, info)``: a cache write bills its tokens at the input price plus
storage for the whole TTL (an upper bound; the cache is usually deleted sooner). If a cached request fails because the
cache is gone, the request is re-sent in full.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

MIN_CACHE_TOKENS = 4096          # Gemini's floor for explicit caches is lower; below this caching saves little
CHARS_PER_TOKEN = 3.6


def est_tokens(obj: Any) -> int:
    return int(len(json.dumps(obj, ensure_ascii=False)) / CHARS_PER_TOKEN)


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class CacheEntry:
    name: str
    model: str
    n_contents: int
    prefix_hash: str
    head_hash: str               # systemInstruction + tools + toolConfig
    tokens: int
    expires: float


class GeminiCache:
    def __init__(self, post: Callable[[str, dict[str, Any] | None, str], Awaitable[tuple[int, str]]],
                 ttl_s: int = 600, min_refresh_tokens: int = 8000,
                 cost_sink: Callable[[str, float, dict[str, Any]], None] | None = None) -> None:
        self.post = post          # (path, body, method) -> (status, text); path relative to the upstream base
        self.ttl_s = ttl_s
        self.min_refresh_tokens = min_refresh_tokens
        self.growth: dict[tuple[str, str], tuple[int, int]] = {}    # (tokens at last request, smoothed growth)
        self.cost_sink = cost_sink
        self.entries: dict[tuple[str, str], CacheEntry] = {}
        self.pending: set[tuple[str, str]] = set()
        self._bg: set[asyncio.Task[Any]] = set()

    @staticmethod
    def head(body: dict[str, Any]) -> dict[str, Any]:
        return {k: body[k] for k in ("systemInstruction", "tools", "toolConfig") if k in body}

    def apply(self, session: str, model: str, body: dict[str, Any]) -> tuple[dict[str, Any], CacheEntry | None]:
        """The body to send: with ``cachedContent`` and only the new turns if the session's cache is a prefix."""
        e = self.entries.get((session, model))
        contents = body.get("contents") or []
        if (e is None or e.expires - time.time() < 30 or len(contents) <= e.n_contents
                or _h(self.head(body)) != e.head_hash or _h(contents[:e.n_contents]) != e.prefix_hash):
            return body, None
        out = {k: v for k, v in body.items() if k not in ("systemInstruction", "tools", "toolConfig", "contents")}
        out["cachedContent"] = e.name
        out["contents"] = contents[e.n_contents:]
        return out, e

    def after(self, session: str, model: str, body: dict[str, Any], prices: Any) -> None:
        """Refresh the session's cache in the background when the uncached tail has grown."""
        key = (session, model)
        contents = body.get("contents") or []
        total = est_tokens(body)
        last, g = self.growth.get(key, (total, 2000))
        if total > last:
            g = int(0.7 * g + 0.3 * (total - last))
        self.growth[key] = (total, g)
        e = self.entries.get(key)
        tail = est_tokens(contents[e.n_contents:]) if e else total
        point = max(self.min_refresh_tokens, int((2 * e.tokens * max(g, 500)) ** 0.5)) if e else MIN_CACHE_TOKENS
        if key in self.pending or tail < point:
            return
        if contents and contents[-1].get("role") == "model":
            return
        self.pending.add(key)
        task = asyncio.ensure_future(self._create(session, model, body, prices))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _create(self, session: str, model: str, body: dict[str, Any], prices: Any) -> None:
        key = (session, model)
        try:
            contents = body.get("contents") or []
            req = {"model": f"models/{model}", "contents": contents, "ttl": f"{self.ttl_s}s", **self.head(body)}
            status, text = await self.post("/cachedContents", req, "POST")
            if status != 200:
                if self.cost_sink:
                    self.cost_sink(session, 0.0, {"cache": "create_failed", "status": status, "model": model})
                return
            d = json.loads(text)
            tokens = int((d.get("usageMetadata") or {}).get("totalTokenCount") or est_tokens(req))
            old = self.entries.get(key)
            self.entries[key] = CacheEntry(name=d["name"], model=model, n_contents=len(contents),
                                           prefix_hash=_h(contents), head_hash=_h(self.head(body)), tokens=tokens,
                                           expires=time.time() + self.ttl_s)
            p = prices.get(model) if prices is not None else None
            usd = 0.0
            if p is not None:
                usd = tokens * p.input / 1e6 + tokens * (p.cache_storage_per_hour or 0.0) / 1e6 * self.ttl_s / 3600
            if self.cost_sink:
                self.cost_sink(session, usd, {"cache": "created", "model": model, "tokens": tokens})
            if old is not None:
                await self.post(f"/{old.name}", None, "DELETE")
        except Exception as ex:  # noqa: BLE001 - caching is an optimisation; never fail a request over it
            if self.cost_sink:
                self.cost_sink(session, 0.0, {"cache": "error", "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
        finally:
            self.pending.discard(key)

    def invalidate(self, session: str, model: str) -> None:
        self.entries.pop((session, model), None)

    async def close(self) -> None:
        for task in list(self._bg):
            task.cancel()
        for e in list(self.entries.values()):
            try:
                await self.post(f"/{e.name}", None, "DELETE")
            except Exception:  # noqa: BLE001
                pass
        self.entries.clear()
