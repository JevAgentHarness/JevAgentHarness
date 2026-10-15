"""The proxy server: an OpenAI-compatible endpoint that routes every agent step to the strong or the fast tier, with
the decision points of points.py (tool choice, stop, reflection, action) around each call.

Endpoints (any of them also under ``/s/<session>/...`` to name the session explicitly):

    POST /v1/chat/completions     OpenAI Chat Completions (streaming and non-streaming)
    POST /v1/responses            OpenAI Responses (Codex)
    GET  /v1/models               the model names clients may use
    GET  /health, GET /stats      liveness; spend, calls and tier shares so far

Keys stay in the proxy's environment: point the harness at the proxy with any dummy key, so no provider key ever
enters the agent's sandbox. Logs (``log_dir``) are JSONL, one record per agent step (steps.jsonl) and one per decision
call (decisions.jsonl), and never contain a key, a header or a message body.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from aiohttp import web

from . import __version__
from . import points as P
from .config import Config, RouterCfg
from .decision import DecisionBackend, StepView, make_backend
from .dialects import chat as chat_dialect
from .dialects import responses as responses_dialect
from .router import HARD, QUESTION, Router, Session, render_state, session_key
from .types import Request, Response
from .upstreams import CallInfo, UpstreamError, Upstreams

KEEPALIVE_S = 10.0
# router settings a model name may override (models: {jah-auto-t75: {policy: route, tau: 0.75}})
ROUTER_KEYS = ("tau", "first_steps_strong", "stick_strong", "max_fast_streak", "verify_finish", "on_error", "strong",
               "fast")
SESSION_TTL_S = 6 * 3600
# the order in which redo reasons are reported when several checks flag one reply
REDO_ORDER = ["action_guard", "tool_guard", "route_guard", "stop_guard", "finish_check"]
# routing reasons whose step the tool-choice cascade may move to its tier (not hard or forced-strong steps)
MOVABLE = {"routine", "guard", "routing_off", "shadow", "pinned"}
MAX_ACTION_ASKS = 4                       # proposed calls checked per reply (more: the first four)


@dataclass
class _Verdict:
    """A post-call check's outcome: redo the step on the strong tier (with a note), and the decision calls made."""
    redo: bool
    reason: str
    note: str | None = None
    asks: list[P.Ask] = field(default_factory=list)


@dataclass
class _Step:
    """Everything one agent step's decision points share."""
    name: str
    sess: Session
    req: Request
    rc: RouterCfg
    dc: P.Decisions
    routed: bool
    step: int
    pinned: str | None = None
    pending: dict[str, asyncio.Future[P.Ask]] = field(default_factory=dict)   # pre-call questions
    route_reason: str = ""
    tier: str = ""
    reason: str = ""
    req_base: Request | None = None        # the request with the harness notes (if any)
    req_call: Request | None = None        # ... and with a forced or shortlisted tool set (if any)
    state: str | None = None
    view: StepView | None = None
    route_ask: P.Ask | None = None
    asks: list[P.Ask] = field(default_factory=list)
    marks: dict[str, Any] = field(default_factory=dict)
    first_tier: str = ""
    first_resp: Response | None = None
    final_tier: str = ""


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
        self.asker = P.Asker(self.backend)
        self.point_counts: Counter[str] = Counter()
        self._filters: dict[P.PointCfg, P.ActionFilter] = {}
        self._submits: dict[tuple[str, ...], Any] = {}

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
        """Decide, run and check one request. Returns the response and the step record (also logged).

        Before the call, the routing question and the other pre-call questions (tool choice, reflection, the stop
        trigger) run concurrently; after it, the post-call checks (finish check / stop guard, tool guard, routing
        guard, action guard) run concurrently on the reply. A check that flags the reply has the strong tier redo the
        step once (a risky action in that redo gets one more look). Every decision call is logged to decisions.jsonl.
        """
        name, spec = self.cfg.model_spec(req.model)
        # the model name is part of the key: two arms of one benchmark task run side by side as separate sessions
        sess = self.session(f"{name}/{session_key(req, session_hint)}")
        if self.over_budget() and req.step_index() == 0:
            raise UpstreamError(429, f"jah budget of ${self.cfg.budget_usd} reached; no new sessions")
        t0 = time.monotonic()
        routed = "tier" not in spec
        over = {k: v for k, v in spec.items() if k in ROUTER_KEYS}
        rc = replace(self.cfg.router, **over) if over else self.cfg.router     # this model name's router settings
        st = _Step(name=name, sess=sess, req=req, rc=rc, dc=self.cfg.decisions_for(spec, rc.tau), routed=routed,
                   step=req.step_index(), pinned=spec.get("tier"))
        async with sess.lock:
            st.pending = self._start_pre(st)
            if routed:
                rec = await self.router.route(sess, req, rc, mode=st.dc.routing.mode, commit=False)
            else:
                rec = {"step": st.step, "tier": spec["tier"], "reason": "pinned", "p_hard": None}
            st.route_reason = str(rec["reason"])
            await self._apply_pre(st, rec)
            if routed and not st.route_reason.startswith("repeat"):
                self.router.settle(sess, st.step, st.tier, st.reason, rc)
            rec["tier"], rec["reason"] = st.tier, st.reason
        state = rec.pop("_state", None)
        if state is not None and self.cfg.log_states and not str(rec.get("reason", "")).startswith("repeat"):
            with open(self.log_dir / "states.jsonl", "a") as f:
                f.write(json.dumps({"ts": utc_now(), "session": sess.key, "model_name": name, "step": rec["step"],
                                    "state": state, "p_hard": rec.get("p_hard"), "tier": rec.get("tier"),
                                    "reason": rec.get("reason")}, ensure_ascii=False) + "\n")
        if routed and "decision" in rec and not st.route_reason.startswith("repeat"):
            st.route_ask = self._route_ask(st, rec, state)
        rec = {"ts": utc_now(), "session": sess.key, "model_name": name, **rec}
        calls: list[CallInfo] = []
        try:
            resp = await self._run_and_check(st, rec, calls, t0)
        except BaseException:
            for task in st.pending.values():
                task.cancel()
            raise
        rec["tier"] = st.final_tier
        await self._settle_pending(st, resp)
        self._finish(sess, rec, calls, t0, resp=resp, st=st)
        return resp, rec

    async def _run_and_check(self, st: _Step, rec: dict[str, Any], calls: list[CallInfo], t0: float) -> Response:
        rc, sess = st.rc, st.sess
        tier = st.tier
        fallback = rc.strong if st.routed else str(st.pinned)
        try:
            resp, info = await self.up.call(tier, st.req_call, sess.key)
            calls.append(info)
        except UpstreamError as e:
            if tier == fallback and st.req_call is st.req_base:      # nothing else to try
                self._finish(sess, rec, calls, t0, error=str(e), st=st)
                raise
            # a failed fast call goes to the strong tier once; a failed forced or shortlisted call is re-sent plain
            rec["fast_error"] = str(e)[:200]
            tier = fallback
            if st.routed:
                self.router.escalated(sess, rec["step"], "fast_error", rc)
            rec["reason"] = "fast_error"
            resp, info = await self._call_or_log(sess, rec, calls, t0, tier, st.req_base, st)
        st.first_tier, st.first_resp = tier, resp
        verdicts = await self._checks(st, resp, tier)
        redo = [v for v in verdicts if v.redo]
        if redo:
            redo.sort(key=lambda v: REDO_ORDER.index(v.reason))
            reasons = [v.reason for v in redo]
            notes = [v.note for v in redo if v.note]
            if "finish_check" in reasons:
                rec["finish_check"] = True             # the fast tier wanted to end the turn: the strong tier decides
            if reasons != ["finish_check"]:
                rec["redo"] = reasons
            tier = rc.strong
            if st.routed:
                self.router.escalated(sess, rec["step"], reasons[0], rc)
            rec["reason"] = reasons[0]
            resp, info = await self._call_or_log(sess, rec, calls, t0, tier, P.with_notes(st.req_base, notes), st)
            self._label_redo(st, verdicts, resp, tier, bool(notes))
            if st.dc.action.on and "action_guard" not in reasons and resp.tool_calls:
                v2 = await self._action_check(st, resp, tier)
                if v2 is not None:
                    st.asks += v2.asks
                    if v2.redo:                         # the redo proposes a risky action: one more look
                        rec["redo"] = rec.get("redo", reasons) + ["action_guard"]
                        rec["reason"] = "action_guard"
                        resp2, _ = await self._call_or_log(sess, rec, calls, t0, tier,
                                                           P.with_notes(st.req_base, [v2.note or ""]), st)
                        self._label_action(v2.asks, resp, resp2, self._action_filter(st.dc.action))
                        resp = resp2
        st.final_tier = tier
        return resp

    async def _call_or_log(self, sess: Session, rec: dict[str, Any], calls: list[CallInfo], t0: float, tier: str,
                           req: Request, st: _Step | None = None) -> tuple[Response, CallInfo]:
        try:
            resp, info = await self.up.call(tier, req, sess.key)
        except UpstreamError as e:
            self._finish(sess, rec, calls, t0, error=str(e), st=st)
            raise
        calls.append(info)
        return resp, info

    # ------------------------------------------------------------------ decision points before the call
    def _start_pre(self, st: _Step) -> dict[str, asyncio.Future[P.Ask]]:
        """Launch the pre-call questions that can change (or, in shadow, describe) this step; skip the others."""
        dc, req, step = st.dc, st.req, st.step
        if step in st.sess.plans:                      # a retried request: this step's decisions are reused
            return {}
        ts = dc.tool_select
        opening = step < st.rc.first_steps_strong
        forced = self.router.forced(st.sess, step, st.rc) if st.routed else None
        opts = P.tool_options(req) if ts.on else None
        ask_tool = bool(opts) and (ts.mode == "shadow" or (not opening and (
            (ts.mode == "assist" and len(req.tools) >= ts.min_tools) or ts.mode == "guard"
            or (ts.mode == "cascade" and forced is None))))
        ask_refl = dc.reflection.mode in ("trigger", "shadow") and step >= 1
        ask_stop = dc.stop.mode == "trigger" and step >= 1
        if not (ask_tool or ask_refl or ask_stop):
            return {}
        state, view = render_state(req)
        st.state, st.view = state, view
        timeout = st.rc.decision_timeout_s
        tasks: dict[str, asyncio.Future[P.Ask]] = {}
        if ask_tool and opts:
            tasks["tool_select"] = asyncio.ensure_future(self.asker.ask_choice(
                "tool_select", ts.mode, state, P.TOOL_TEXT, opts[0], opts[1], timeout, ts.tau, ts.max_options,
                keep=max(ts.k, 3)))
        if ask_refl:
            tasks["reflection"] = asyncio.ensure_future(self.asker.ask(
                "reflection", dc.reflection.mode, state, P.REFLECT, timeout, dc.reflection.tau, "yes", view))
        if ask_stop:
            tasks["stop_trigger"] = asyncio.ensure_future(self.asker.ask(
                "stop", "trigger", state, P.STOP_TRIGGER, timeout, dc.stop.tau, "yes", view))
        return tasks

    async def _apply_pre(self, st: _Step, rec: dict[str, Any]) -> None:
        """Await the pre-call questions that decide this step and apply them: notes, the tier, a forced or
        shortlisted tool set. Sets ``st.tier``, ``st.reason``, ``st.req_base`` (with notes) and ``st.req_call``."""
        dc, rc, req = st.dc, st.rc, st.req
        st.tier, st.reason = str(rec["tier"]), str(rec["reason"])
        st.req_base = st.req_call = req
        plan = st.sess.plans.get(st.step)
        if plan is not None:                           # a retry: the same notes, tier and tool set as the first time
            if not st.routed:
                st.tier, st.reason = plan["tier"], "repeat:" + plan["reason"]
            st.req_base = P.with_notes(req, plan["notes"])
            if st.tier == plan["tier"]:                # (not after an escalation: the redo was sent plain)
                st.req_call = self._shape(st.req_base, plan["tool_choice"], plan["keep"])
            else:
                st.req_call = st.req_base
            return
        blocking = [k for k, task in st.pending.items()
                    if k == "stop_trigger" or (k == "reflection" and dc.reflection.mode == "trigger")
                    or (k == "tool_select" and dc.tool_select.mode in ("assist", "cascade"))]
        if blocking:
            await asyncio.gather(*(st.pending[k] for k in blocking))
        notes: list[str] = []
        fired = None
        if "reflection" in st.pending and dc.reflection.mode == "trigger":
            a = st.pending["reflection"].result()
            if a.ok and a.p_of() >= dc.reflection.tau:
                a.acted, a.effect = True, "reflect_on_strong"
                notes.append(P.REFLECT_NOTE)
                fired = "reflection"
            else:
                a.effect = "error_skip" if not a.ok else "no"
        elif dc.reflection.mode == "always" and st.step >= 1 and st.step % dc.reflection.every == 0:
            notes.append(P.REFLECT_NOTE)
            fired = "reflection"
            st.marks["reflection"] = "always"
        if "stop_trigger" in st.pending:
            a = st.pending["stop_trigger"].result()
            if a.ok and a.p_of() >= dc.stop.tau:
                a.acted, a.effect = True, "finish_note_on_strong"
                notes.append(P.STOP_TRIGGER_NOTE)
                fired = fired or "stop_trigger"
            else:
                a.effect = "error_skip" if not a.ok else "no"
        if fired:
            st.tier, st.reason = rc.strong, fired
        force: str | None = None
        keep: list[str] | None = None
        ts = dc.tool_select
        if "tool_select" in st.pending and ts.mode in ("assist", "cascade"):
            a = st.pending["tool_select"].result()
            if ts.mode == "cascade":
                if not a.ok:
                    a.effect = "error_defer"
                elif a.choice == P.NO_TOOL:
                    a.effect = "defer_no_tool"
                elif (a.confidence or 0.0) < ts.tau:
                    a.effect = "defer_unsure"
                elif fired or st.reason not in MOVABLE:
                    a.effect = f"defer_{st.reason}"            # a hard or forced-strong step stays the strong tier's
                else:
                    a.acted, a.effect = True, "forced" if ts.force else "tier"
                    a.extra["forced_tool"] = a.choice
                    st.tier, st.reason = self._cascade_tier(st), "tool_cascade"
                    force = a.choice if ts.force else None
            elif not a.ok:
                a.effect = "error_all_tools"
            else:
                top = [o for o, _ in sorted(a.p.items(), key=lambda kv: -kv[1]) if o != P.NO_TOOL][:ts.k]
                wanted = set(top) | P.used_tools(req)
                names = [t.name for t in req.tools if t.name in wanted]
                if len(names) < len(req.tools):
                    keep = names
                    a.acted, a.effect = True, "shortlist"
                    a.extra["kept_tools"] = len(names)
                else:
                    a.effect = "all_tools"
        st.req_base = P.with_notes(req, notes)
        st.req_call = self._shape(st.req_base, force, keep)
        if notes or force or keep or st.pending:
            st.sess.plans[st.step] = {"tier": st.tier, "reason": st.reason, "notes": notes, "tool_choice": force,
                                      "keep": keep}
            for old in [s for s in st.sess.plans if s < st.step - 4]:
                st.sess.plans.pop(old, None)

    def _cascade_tier(self, st: _Step) -> str:
        t = st.dc.tool_select.tier
        return st.rc.fast if t in (None, "fast") else st.rc.strong if t == "strong" else str(t)

    @staticmethod
    def _shape(req: Request, force: str | None, keep: list[str] | None) -> Request:
        if force is None and keep is None:
            return req
        tools = req.tools if keep is None else [t for t in req.tools if t.name in set(keep)]
        return replace(req, tools=tools, tool_choice={"name": force} if force else req.tool_choice, modified=True)

    # ------------------------------------------------------------------ decision points after the call
    def _state(self, st: _Step) -> tuple[str, StepView]:
        if st.state is None or st.view is None:
            st.state, st.view = render_state(st.req)
        return st.state, st.view

    async def _checks(self, st: _Step, resp: Response, tier: str) -> list[_Verdict]:
        """The post-call checks that apply to this reply, run concurrently. A check that fails internally accepts the
        reply (the step goes on as without the decision point)."""
        jobs = []
        final = bool(st.req.tools) and P.ends_turn(resp, self._submit(st.dc.stop))
        if final and st.dc.stop.on:
            jobs.append(self._stop_check(st, resp, tier))
        if "tool_select" in st.pending and st.dc.tool_select.mode == "guard":
            jobs.append(self._tool_guard(st, resp, tier))
        # the routing guard is skipped when the finish check re-does this final answer anyway
        redone = final and st.dc.stop.mode in ("strong", "trigger", "shadow")
        if st.dc.routing.mode == "guard" and st.route_reason == "guard" and tier != st.rc.strong and not redone:
            jobs.append(self._route_guard(st, resp, tier))
        if st.dc.action.on and resp.tool_calls:
            jobs.append(self._action_check(st, resp, tier))
        out: list[_Verdict] = []
        for v in await asyncio.gather(*jobs, return_exceptions=True):
            if isinstance(v, _Verdict):
                st.asks += v.asks
                out.append(v)
            elif isinstance(v, BaseException):
                st.marks["check_error"] = f"{type(v).__name__}: {str(v)[:200]}"
        return out

    async def _stop_check(self, st: _Step, resp: Response, tier: str) -> _Verdict | None:
        c, rc = st.dc.stop, st.rc
        on_strong = tier == rc.strong
        if c.mode in ("strong", "trigger"):
            return None if on_strong else _Verdict(True, "finish_check")
        if on_strong and c.scope != "all":
            return None
        state, view = self._state(st)
        a = await self.asker.ask("stop", c.mode, f"{state}\n\nPROPOSED FINAL REPLY: {P.proposal_text(resp)}", P.STOP,
                                 st.rc.decision_timeout_s, c.tau, "done", view)
        a.extra["tier"] = tier
        if c.mode == "shadow":                          # behaves as `strong`; the strong redo labels the answer
            a.effect = "logged"
            return _Verdict(not on_strong, "finish_check", asks=[a])
        if not a.ok:                                    # guard without an answer: the finish check (safe default)
            a.effect = "error_accept" if on_strong else "error_redo"
            return _Verdict(not on_strong, "finish_check", asks=[a])
        if a.p_of() >= c.tau:
            a.effect = "accepted"
            return _Verdict(False, "stop_guard", asks=[a])
        a.acted, a.effect = True, "redo"
        return _Verdict(True, "stop_guard", P.STOP_REJECT_NOTE if on_strong else None, [a])

    async def _tool_guard(self, st: _Step, resp: Response, tier: str) -> _Verdict:
        c = st.dc.tool_select
        a = await st.pending["tool_select"]
        a.label, a.label_tier, a.fit_ok = P.llm_choice(resp), tier, True
        if not a.ok:
            a.effect = "error_accept"
            return _Verdict(False, "tool_guard", asks=[])
        if a.choice != P.NO_TOOL and (a.confidence or 0.0) >= c.tau and a.choice != a.label:
            a.acted, a.effect = True, "redo"
            return _Verdict(True, "tool_guard", P.tool_guard_note(resp), [])
        a.effect = "accepted"
        return _Verdict(False, "tool_guard")

    async def _route_guard(self, st: _Step, resp: Response, tier: str) -> _Verdict:
        c, rc = st.dc.routing, st.rc
        state, view = self._state(st)
        a = await self.asker.ask("routing", "guard", f"{state}\n\nPROPOSED NEXT STEP: {P.proposal_text(resp)}",
                                 P.ROUTE_GUARD, rc.decision_timeout_s, c.tau, "redo", view)
        a.extra["tier"] = tier
        if not a.ok:                                    # no answer: the router's fail-safe tier decides
            redo = rc.on_error == rc.strong
            a.effect = "error_redo" if redo else "error_accept"
            return _Verdict(redo, "route_guard", asks=[a])
        if a.p_of() >= c.tau:
            a.acted, a.effect = True, "redo"
            return _Verdict(True, "route_guard", asks=[a])
        a.effect = "accepted"
        return _Verdict(False, "route_guard", asks=[a])

    def _submit(self, c: P.PointCfg) -> Any:
        if not c.submit:
            return None
        rx = self._submits.get(c.submit)
        if rx is None:
            rx = self._submits[c.submit] = P.submit_re(c)
        return rx

    def _action_filter(self, c: P.PointCfg) -> P.ActionFilter:
        flt = self._filters.get(c)
        if flt is None:
            flt = self._filters[c] = P.ActionFilter(c)
        return flt

    async def _action_check(self, st: _Step, resp: Response, tier: str) -> _Verdict | None:
        c = st.dc.action
        flt = self._action_filter(c)
        cands = flt.candidates(resp.tool_calls)[:MAX_ACTION_ASKS]
        if not cands:
            return None
        state, view = self._state(st)
        asks = list(await asyncio.gather(*[self.asker.ask(
            "action", c.mode, f"{state}\n\nPROPOSED ACTION: {P.call_text(x, P.CHARS_PROPOSAL)}", P.ACTION,
            st.rc.decision_timeout_s, c.tau, "risky", replace(view, risky=flt.is_risky(x))) for x in cands]))
        flagged = []
        for x, a in zip(cands, asks):
            a.extra.update({"tool": x.name, "tier": tier, "pattern": flt.is_risky(x)})
            if not a.ok:
                a.effect = "error_recheck" if c.on_error == "recheck" and c.mode == "guard" else "error_accept"
                if a.effect == "error_recheck":
                    flagged.append(x)
            elif a.p_of() >= c.tau and c.mode == "guard":
                a.acted, a.effect = True, "recheck"
                flagged.append(x)
            else:
                a.effect = "logged" if c.mode == "shadow" else "accepted"
        if flagged:
            return _Verdict(True, "action_guard", P.action_note(flagged), asks)
        return _Verdict(False, "action_guard", asks=asks)

    def _label_redo(self, st: _Step, verdicts: list[_Verdict], resp: Response, tier: str, noted: bool) -> None:
        """What the strong redo says about the checked reply: labels for `jah fit` (stop) and the logs."""
        for v in verdicts:
            for a in v.asks:
                if a.point == "stop" and a.extra.get("tier") != st.rc.strong:
                    a.label = "done" if P.ends_turn(resp, self._submit(st.dc.stop)) else "not_done"
                    a.label_tier, a.fit_ok = tier, not noted
                elif a.point == "action" and a.acted:
                    self._label_action([a], st.first_resp, resp, self._action_filter(st.dc.action))
                elif a.point == "routing" and a.acted:          # the routing guard: did the strong redo differ?
                    a.extra["recheck"] = P.recheck_kind(st.first_resp, resp)
        if "tool_select" in st.pending and st.dc.tool_select.mode == "guard":
            a = st.pending["tool_select"].result()
            a.extra["after"] = P.llm_choice(resp)

    @staticmethod
    def _label_action(asks: list[P.Ask], before: Response | None, after: Response, flt: P.ActionFilter) -> None:
        """What the strong re-check did with a flagged call: ``recheck`` = ``same`` (the same calls, up to formatting),
        ``changed`` (other calls) or ``no_call`` (a reply without a tool call); ``recheck_risky``: whether the reply
        that reaches the harness still makes a call the pre-filter matches."""
        what = P.recheck_kind(before, after)
        risky = any(flt.is_risky(c) for c in after.tool_calls)
        for a in asks:
            if a.acted:
                a.extra["recheck"] = what
                a.extra["recheck_risky"] = risky

    async def _settle_pending(self, st: _Step, resp: Response) -> None:
        """Collect the pre-call questions that did not decide anything (shadow, guard) and label them with the LLM's
        own choice."""
        if not st.pending:
            return
        await asyncio.gather(*st.pending.values(), return_exceptions=True)
        for k, task in st.pending.items():
            if task.cancelled() or task.exception() is not None:
                continue
            a = task.result()
            if k == "tool_select" and st.first_resp is not None:
                if a.label is None:
                    a.label, a.label_tier = P.llm_choice(st.first_resp), st.first_tier
                    a.fit_ok = not a.acted
                if a.mode == "shadow":
                    a.effect = "logged"
                if a.extra.get("forced_tool"):
                    a.extra["followed"] = a.label == a.extra["forced_tool"]
            elif k == "reflection" and a.mode == "shadow":
                a.effect = "logged"
            st.asks.append(a)

    def _route_ask(self, st: _Step, rec: dict[str, Any], state: str | None) -> P.Ask:
        d = rec.get("decision") or {}
        a = P.Ask(point="routing", mode=st.dc.routing.mode, question=QUESTION.id, options=list(QUESTION.options),
                  tau=st.rc.tau, act_option=HARD, model=str(d.get("model") or ""),
                  cost_usd=float(d.get("cost_usd") or 0.0), latency_s=float(d.get("latency_s") or 0.0),
                  error=d.get("error"))
        if rec.get("p_hard") is not None:
            a.p = {"routine": 1.0 - float(rec["p_hard"]), HARD: float(rec["p_hard"])}
        a.acted = st.route_reason in ("hard", "routine")
        a.effect = st.route_reason
        return a

    def _finish(self, sess: Session, rec: dict[str, Any], calls: list[CallInfo], t0: float,
                resp: Response | None = None, error: str | None = None, st: _Step | None = None) -> None:
        dec = rec.get("decision") or {}
        asks = list(st.asks) if st is not None else []
        if st is not None:
            for task in st.pending.values():           # an upstream error: the questions that finished are logged
                if task.done() and not task.cancelled() and task.exception() is None:
                    a = task.result()
                    if all(a is not x for x in asks):
                        asks.append(a)
        llm_cost = sum(c.cost_usd for c in calls)
        dcost = float(dec.get("cost_usd") or 0.0) + sum(a.cost_usd for a in asks)
        rec.update({
            "calls": [{"tier": c.tier, "model": c.model, "latency_s": round(c.latency_s, 3), "attempts": c.attempts,
                       "cost_usd": round(c.cost_usd, 8), "priced": c.priced,
                       **({"signature_fix": True} if c.signature_fix else {})} for c in calls],
            "llm_cost_usd": round(llm_cost, 8),
            "decision_cost_usd": round(dcost, 8),
            "cost_usd": round(llm_cost + dcost, 8),
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
        if st is not None and (asks or st.marks or not st.dc.only_routing()):
            rec["points"] = [{"point": a.point, "mode": a.mode, "choice": a.choice,
                              "p": None if not a.p else round(a.p_of() if a.act_option else a.confidence or 0.0, 4),
                              "acted": a.acted, "effect": a.effect, **({"error": True} if a.error else {})}
                             for a in asks]
            rec.update({f"mark_{k}": v for k, v in st.marks.items()})
        sess.cost_usd += rec["cost_usd"]
        sess.calls += 1
        sess.tiers[rec["tier"]] = sess.tiers.get(rec["tier"], 0) + 1
        self.counts[rec["tier"]] += 1
        self._log(rec)
        if st is not None:
            self._log_decisions(st, rec, asks)

    def _log_decisions(self, st: _Step, rec: dict[str, Any], asks: list[P.Ask]) -> None:
        """decisions.jsonl: one line per decision call (states.jsonl gets the state when log_states is on)."""
        rows = ([st.route_ask] if st.route_ask is not None else []) + asks
        if not rows:
            return
        head = {"ts": rec["ts"], "session": rec["session"], "model_name": rec["model_name"], "step": rec["step"]}
        with open(self.log_dir / "decisions.jsonl", "a") as f:
            for a in rows:
                f.write(json.dumps({**head, **a.record()}, ensure_ascii=False, default=str) + "\n")
                self.point_counts[f"{a.point}:asked"] += 1
                if a.acted:
                    self.point_counts[f"{a.point}:acted"] += 1
        if self.cfg.log_states:
            with open(self.log_dir / "states.jsonl", "a") as f:
                for a in asks:
                    if a.state is not None:
                        f.write(json.dumps({**head, "point": a.point, "question": a.question, "state": a.state,
                                            "p": {k: round(v, 4) for k, v in a.p.items()}, "choice": a.choice},
                                           ensure_ascii=False) + "\n")

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
            "decision": self.backend.describe(), "tau": self.cfg.router.tau,
            "decision_points": self.cfg.decisions_for(self.cfg.models[self.cfg.default_model]).modes(),
            "decision_calls": dict(sorted(self.point_counts.items()))})

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
