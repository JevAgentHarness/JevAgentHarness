"""The proxy server: an OpenAI-compatible endpoint that routes every agent step to the strong or the fast tier.

Endpoints (any of them also under ``/s/<session>/...`` to name the session explicitly):

    POST /v1/chat/completions     OpenAI Chat Completions (streaming and non-streaming)
    POST /v1/responses            OpenAI Responses (Codex)
    GET  /v1/models               the model names clients may use
    GET  /health, GET /stats      liveness; spend, calls and tier shares so far

Keys stay in the proxy's environment: point the harness at the proxy with any dummy key, so no provider key ever
enters the agent's sandbox. Logs (``log_dir``) are JSONL, one record per agent step, and never contain a key, a
header or a message body.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from aiohttp import web

from . import __version__
from .config import Config
from .decision import DecisionBackend, make_backend
from .dialects import chat as chat_dialect
from .dialects import responses as responses_dialect
from .router import Router, Session, session_key
from .types import Request, Response
from .upstreams import CallInfo, UpstreamError, Upstreams

KEEPALIVE_S = 10.0
# router settings a model name may override (models: {jah-auto-t75: {policy: route, tau: 0.75}})
ROUTER_KEYS = ("tau", "first_steps_strong", "stick_strong", "max_fast_streak", "verify_finish", "on_error", "strong",
               "fast")
SESSION_TTL_S = 6 * 3600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Proxy:
    def __init__(self, cfg: Config, backend: DecisionBackend | None = None, upstreams: Upstreams | None = None,
                 log_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.backend = backend if backend is not None else make_backend(cfg.router.decision)
        self.router = Router(cfg.router, self.backend)
        self.up = upstreams or Upstreams(cfg)
        self.up.cost_sink = self._cache_cost
        self.log_dir = Path(log_dir or cfg.log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Session] = {}
        self.spent = self._spent_in_logs()
        self.started = time.time()
        self.counts: Counter[str] = Counter()

    # ------------------------------------------------------------------ bookkeeping
    def _spent_in_logs(self) -> float:
        p = self.log_dir / "steps.jsonl"
        total = 0.0
        if p.exists():
            for ln in p.read_text(errors="replace").splitlines():
                with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError, AttributeError):
                    total += float(json.loads(ln).get("cost_usd") or 0.0)
        return total

    def _log(self, rec: dict[str, Any]) -> None:
        self.spent += float(rec.get("cost_usd") or 0.0)
        with open(self.log_dir / "steps.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _cache_cost(self, session: str, usd: float, info: dict[str, Any]) -> None:
        """A context-cache write (or failure): its own log record, counted in the session's and the proxy's spend."""
        if session in self.sessions:
            self.sessions[session].cost_usd += usd
        self._log({"ts": utc_now(), "kind": "cache", "session": session, "cost_usd": round(usd, 8), **info})

    def over_budget(self) -> bool:
        return self.cfg.budget_usd is not None and self.spent >= self.cfg.budget_usd

    def session(self, key: str) -> Session:
        now = time.time()
        if len(self.sessions) > 5000:
            for k in [k for k, s in self.sessions.items() if now - s.created > SESSION_TTL_S]:
                self.sessions.pop(k, None)
        if key not in self.sessions:
            self.sessions[key] = Session(key=key)
        return self.sessions[key]

    # ------------------------------------------------------------------ one agent step
    async def complete(self, req: Request, session_hint: str | None = None
                       ) -> tuple[Response, dict[str, Any]]:
        """Route and run one request. Returns the response and the step record (also logged)."""
        name, spec = self.cfg.model_spec(req.model)
        # the model name is part of the key: two arms of one benchmark task run side by side as separate sessions
        sess = self.session(f"{name}/{session_key(req, session_hint)}")
        if self.over_budget() and req.step_index() == 0:
            raise UpstreamError(429, f"jah budget of ${self.cfg.budget_usd} reached; no new sessions")
        t0 = time.monotonic()
        routed = "tier" not in spec
        over = {k: v for k, v in spec.items() if k in ROUTER_KEYS}
        rc = replace(self.cfg.router, **over) if over else self.cfg.router     # this model name's router settings
        async with sess.lock:
            if routed:
                rec = await self.router.route(sess, req, rc)
            else:
                rec = {"step": req.step_index(), "tier": spec["tier"], "reason": "pinned", "p_hard": None}
        state = rec.pop("_state", None)
        if state is not None and self.cfg.log_states and not str(rec.get("reason", "")).startswith("repeat"):
            with open(self.log_dir / "states.jsonl", "a") as f:
                f.write(json.dumps({"ts": utc_now(), "session": sess.key, "model_name": name, "step": rec["step"],
                                    "state": state, "p_hard": rec.get("p_hard"), "tier": rec.get("tier"),
                                    "reason": rec.get("reason")}, ensure_ascii=False) + "\n")
        rec = {"ts": utc_now(), "session": sess.key, "model_name": name, **rec}
        calls: list[CallInfo] = []
        tier = rec["tier"]
        try:
            resp, info = await self.up.call(tier, req, sess.key)
            calls.append(info)
        except UpstreamError as e:
            if not (routed and tier == rc.fast):
                self._finish(sess, rec, calls, t0, error=str(e))
                raise
            rec["fast_error"] = str(e)[:200]           # a failed fast call goes to the strong tier once
            tier = rc.strong
            self.router.escalated(sess, rec["step"], "fast_error", rc)
            rec["reason"] = "fast_error"
            resp, info = await self._call_or_log(sess, rec, calls, t0, tier, req)
        if (routed and tier == rc.fast and rc.verify_finish and req.tools
                and not resp.tool_calls):
            rec["finish_check"] = True                 # the fast tier wants to end the turn: the strong tier decides
            tier = rc.strong
            self.router.escalated(sess, rec["step"], "finish_check", rc)
            rec["reason"] = "finish_check"
            resp, info = await self._call_or_log(sess, rec, calls, t0, tier, req)
        rec["tier"] = tier
        self._finish(sess, rec, calls, t0, resp=resp)
        return resp, rec

    async def _call_or_log(self, sess: Session, rec: dict[str, Any], calls: list[CallInfo], t0: float, tier: str,
                           req: Request) -> tuple[Response, CallInfo]:
        try:
            resp, info = await self.up.call(tier, req, sess.key)
        except UpstreamError as e:
            self._finish(sess, rec, calls, t0, error=str(e))
            raise
        calls.append(info)
        return resp, info

    def _finish(self, sess: Session, rec: dict[str, Any], calls: list[CallInfo], t0: float,
                resp: Response | None = None, error: str | None = None) -> None:
        dec = rec.get("decision") or {}
        llm_cost = sum(c.cost_usd for c in calls)
        rec.update({
            "calls": [{"tier": c.tier, "model": c.model, "latency_s": round(c.latency_s, 3), "attempts": c.attempts,
                       "cost_usd": round(c.cost_usd, 8), "priced": c.priced,
                       **({"signature_fix": True} if c.signature_fix else {})} for c in calls],
            "llm_cost_usd": round(llm_cost, 8),
            "decision_cost_usd": round(float(dec.get("cost_usd") or 0.0), 8),
            "cost_usd": round(llm_cost + float(dec.get("cost_usd") or 0.0), 8),
            "latency_s": round(time.monotonic() - t0, 3),
        })
        if resp is not None:
            u = resp.usage
            rec.update({"input_tokens": u.input_tokens, "cached_tokens": u.cached_tokens,
                        "cache_write_tokens": u.cache_write_tokens,
                        "output_tokens": u.output_tokens, "reasoning_tokens": u.reasoning_tokens,
                        "n_tool_calls": len(resp.tool_calls), "finish": resp.finish_reason})
        if error:
            rec["error"] = error[:300]
        sess.cost_usd += rec["cost_usd"]
        sess.calls += 1
        sess.tiers[rec["tier"]] = sess.tiers.get(rec["tier"], 0) + 1
        self.counts[rec["tier"]] += 1
        self._log(rec)

    # ------------------------------------------------------------------ HTTP
    async def _serve(self, request: web.Request, dialect: Any) -> web.StreamResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _error(400, "request body is not JSON")
        try:
            req = dialect.parse_request(body)
        except (KeyError, TypeError, ValueError) as e:
            return _error(400, f"cannot parse request: {e}")
        hint = request.match_info.get("session") or request.headers.get("x-jah-session") \
            or dialect.session_hint(request.headers, body)
        model_out = req.model or self.cfg.default_model
        if not req.stream:
            try:
                resp, rec = await self.complete(req, hint)
            except UpstreamError as e:
                return _error(e.status if 400 <= e.status < 600 and e.status != 599 else 502, str(e))
            return web.json_response(dialect.render_response(resp, model_out, req))

        out = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await out.prepare(request)
        for chunk in dialect.stream_preamble(req, model_out):
            await out.write(chunk)
        task = asyncio.ensure_future(self.complete(req, hint))
        while True:
            done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_S)
            if done:
                break
            with contextlib.suppress(ConnectionError, RuntimeError):
                await out.write(b": keepalive\n\n")
        try:
            resp, _rec = task.result()
            chunks: Iterator[bytes] = dialect.render_stream(resp, model_out, req)
        except UpstreamError as e:
            chunks = dialect.render_stream_error(str(e), e.status, model_out, req)
        with contextlib.suppress(ConnectionError, RuntimeError):
            for chunk in chunks:
                await out.write(chunk)
            await out.write_eof()
        return out

    async def chat(self, request: web.Request) -> web.StreamResponse:
        return await self._serve(request, chat_dialect)

    async def responses(self, request: web.Request) -> web.StreamResponse:
        return await self._serve(request, responses_dialect)

    async def models(self, request: web.Request) -> web.Response:
        names = list(self.cfg.models) + [f"jah-{t}" for t in self.cfg.tiers if f"jah-{t}" not in self.cfg.models]
        return web.json_response({"object": "list", "data": [
            {"id": n, "object": "model", "created": 0, "owned_by": "jevagentharness"} for n in names]})

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "version": __version__})

    async def stats(self, request: web.Request) -> web.Response:
        total = sum(self.counts.values())
        return web.json_response({
            "version": __version__, "uptime_s": round(time.time() - self.started, 1),
            "spent_usd": round(self.spent, 6), "budget_usd": self.cfg.budget_usd, "sessions": len(self.sessions),
            "steps": total, "tier_share": {k: round(v / total, 4) for k, v in self.counts.items()} if total else {},
            "tiers": {k: {"model": t.model, "effort": t.effort} for k, t in self.cfg.tiers.items()},
            "decision": self.backend.describe(), "tau": self.cfg.router.tau})

    async def close(self) -> None:
        await self.up.close()
        await self.backend.close()

    def app(self) -> web.Application:
        app = web.Application(client_max_size=512 * 1024 * 1024)
        r = app.router
        for prefix in ("", "/s/{session}"):
            for path in ("/v1/chat/completions", "/chat/completions", "/api/v1/chat/completions"):
                r.add_post(prefix + path, self.chat)
            for path in ("/v1/responses", "/responses"):
                r.add_post(prefix + path, self.responses)
            for path in ("/v1/models", "/models"):
                r.add_get(prefix + path, self.models)
        r.add_get("/health", self.health)
        r.add_get("/stats", self.stats)

        async def _cleanup(_app: web.Application) -> None:
            await self.close()
        app.on_cleanup.append(_cleanup)
        return app


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": {"code": status, "message": message, "type": "jah_error"}}, status=status)


def run(cfg: Config, on_start: Callable[[Proxy], None] | None = None) -> None:
    proxy = Proxy(cfg)
    if on_start:
        on_start(proxy)
    web.run_app(proxy.app(), host=cfg.host, port=cfg.port, print=None, access_log=None)
