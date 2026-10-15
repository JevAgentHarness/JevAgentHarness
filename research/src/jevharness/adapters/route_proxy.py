"""DecisionAgentHarness as a model proxy: per-step routing between a strong and a fast LLM for ANY agent harness.

E29 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09, "E29: the full harness in open-source harnesses"). The proxy runs
on the host, holds the API keys and forwards every LLM request of an unmodified harness (pi, SWE-agent, mini-SWE-agent,
...) to the provider; the harness only gets the proxy's URL and a dummy key, so no key ever enters a task container.
The request path names the arm and the episode::

    POST /e29/<arm>/<episode>/v1beta/models/<model>:generateContent           (Gemini native; also :streamGenerateContent)
    POST /e29/<arm>/<episode>/api/v1/chat/completions                          (OpenAI chat; OpenRouter upstream)
    GET  /health                                                               (spend, cap, arms, episodes)

Modes (one per arm, from the config):

``strong``  every request to the strong LLM, unchanged (the baseline: the harness as it is).
``gated``   the full harness. After the proxy has returned the response of step t it asks the decision model one typed
            multiple-choice question about step t + 1 (:data:`QUESTION`: ``fast`` or ``strong``) on the episode's state
            up to and including step t's action, *while the harness executes that action* (decide-ahead: the call is
            off the critical path). At step t + 1 it waits for the answer at most ``wait_s`` seconds; the confidence
            c = p(fast) routes the step to the fast LLM iff c >= tau (the fitted gate). Sticky: a fast segment goes on
            while c >= tau for at most ``k`` consecutive fast steps, then the strong LLM takes the next step. The first
            request of an episode goes to the strong LLM. Fail open: no answer -> strong; a failed fast request -> the
            strong LLM once.
``shadow``  calibration: the strong LLM acts at every step; after each response the identical request goes once to the
            fast LLM and once more to the strong LLM (non-streaming, never returned to the harness), and the decide-ahead
            question is asked. The logs give the gate's labels (scripts/explore_e29_port.py fit).

E29b (Deviations 2026-10-09, "E29b: step-type routing with a write guard") adds three arm settings, E29's arms keep
their defaults: ``question = "step_type"`` asks :data:`STEP_QUESTION_TEXT` (``inspect`` / ``edit`` / ``run`` /
``finish``) instead of the fast/strong question; a step goes to the fast LLM iff the answer is ``inspect`` with
c = p(inspect) >= tau (sticky segments capped at ``k``). ``guard = true`` buffers every fast-LLM response and returns it
only if it is read-only (:func:`route_formats.read_only_calls`); otherwise the response is discarded (logged with
``guard = reject``) and the strong LLM writes the step from the same request. ``shadow_roles`` picks the shadow calls of
a calibration arm (E29: ``["fast", "strong"]``; E29b: ``["fast"]``). Every call record carries ``ro``, the classifier's
verdict on its response.

Gemini 3.x thought signatures: a function-call part without a signature gets Google's placeholder; an HTTP 400 that
names thought signatures is retried once with every function-call signature replaced by it.

Logs (``--log-dir``; JSONL, never a key, a header or a request body): ``calls.jsonl`` (one record per upstream call:
arm, episode, step, tier, role, model, tokens, cost, latency, status, the response's first-action key and, for primary
calls, the route), ``decisions.jsonl`` (one per decision-model call). The proxy sums the cost in its logs (also across
restarts) and refuses new requests with HTTP 503 once the spend reaches ``cap_usd - cap_margin_usd``.

    python -m jevharness.adapters.route_proxy --config cfg.json --port 8091 --log-dir runs/e29/tb/proxy
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web

from ..accounting import CallRecord, Ledger
from . import route_formats as F

STRONG, FAST = "strong", "fast"
MODES = ("strong", "gated", "shadow")
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
QUESTION_ID = "route"
QUESTION_TEXT = ("Which LLM should write the agent's next step, after the result of the action in progress comes back? "
                 "Choose 'fast' when that step is routine, so a smaller and faster LLM will choose the same action "
                 "(read or search output, run or re-run a command, an edit that follows from the plan); choose 'strong' "
                 "when it needs careful multi-step reasoning, planning, debugging or recovery from an error.")
OPTIONS = ["fast", "strong"]
OPTION_DESCRIPTIONS = ["Routine next step: a smaller, faster LLM will choose the same action as the strong LLM",
                       "Hard next step: needs the strong LLM's careful reasoning, planning or debugging"]
# E29b: the typed step-kind question (registration (b) 1); the fast LLM may write predicted `inspect` steps only
STEP_QUESTION_ID = "step_type"
STEP_QUESTION_TEXT = "Which kind of step comes next, after the result of the action in progress comes back?"
STEP_OPTIONS = ["inspect", "edit", "run", "finish"]
STEP_OPTION_DESCRIPTIONS = [
    "Inspect: read files or run a read-only command (list, view, search, git status / diff / log)",
    "Edit: edit, create or delete a file",
    "Run: run tests, a build or other code",
    "Finish: finish or submit the task"]
QUESTIONS = ("route", "step_type")
ROUTE_OPTION = {"route": "fast", "step_type": "inspect"}     # the answer whose probability is the confidence c


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ----------------------------------------------------------------------------------------------------------- config
@dataclass
class TierCfg:
    model: str
    overrides: dict[str, Any] = field(default_factory=dict)     # OpenAI body fields (e.g. OpenRouter's reasoning)
    prices: dict[str, float] | None = None                      # per-M prices (Gemini; OpenRouter reports usage.cost)


@dataclass
class ArmCfg:
    mode: str
    tau: float | None = None          # gated: None = the gate routes nothing (every step strong, no question asked)
    k: int = 4
    question: str = "route"           # E29: fast/strong; E29b: "step_type" (inspect / edit / run / finish)
    guard: bool = False               # E29b: execute a fast-LLM response only if it is read-only
    shadow_roles: list[str] = field(default_factory=lambda: [FAST, STRONG])   # shadow mode: which shadow calls

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"arm mode must be one of {MODES}, got {self.mode!r}")
        if self.question not in QUESTIONS:
            raise ValueError(f"arm question must be one of {QUESTIONS}, got {self.question!r}")
        if not set(self.shadow_roles) <= {FAST, STRONG}:
            raise ValueError("shadow_roles must be a subset of ['fast', 'strong']")


@dataclass
class ProxyConfig:
    fmt: str
    upstream: str
    key_env: str
    tiers: dict[str, TierCfg]
    arms: dict[str, ArmCfg]
    decision: dict[str, Any] = field(default_factory=dict)      # {"backend": spec, "wait_s": 10, "timeout_s": 30}
    cap_usd: float | None = None
    cap_margin_usd: float = 0.5
    cap_hard_extra_usd: float = 2.0     # past the cap only running episodes go on, up to this much more
    retries: int = 2
    request_timeout_s: float = 900.0
    log_dir: str = "runs/route_proxy"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProxyConfig":
        fmt = d.get("format") or d.get("fmt")
        if fmt not in F.FORMATS:
            raise ValueError(f"format must be one of {F.FORMATS}")
        tiers = {name: TierCfg(**t) for name, t in (d.get("tiers") or {}).items()}
        if set(tiers) != {STRONG, FAST}:
            raise ValueError("tiers must be exactly 'strong' and 'fast'")
        arms = {name: ArmCfg(**a) for name, a in (d.get("arms") or {}).items()}
        if not arms:
            raise ValueError("no arms")
        return cls(fmt=fmt, upstream=str(d["upstream"]).rstrip("/"), key_env=str(d["key_env"]), tiers=tiers,
                   arms=arms, decision=dict(d.get("decision") or {}), cap_usd=d.get("cap_usd"),
                   cap_margin_usd=float(d.get("cap_margin_usd", 0.5)),
                   cap_hard_extra_usd=float(d.get("cap_hard_extra_usd", 2.0)), retries=int(d.get("retries", 2)),
                   request_timeout_s=float(d.get("request_timeout_s", 900.0)),
                   log_dir=str(d.get("log_dir", "runs/route_proxy")))

    def needs_decisions(self) -> bool:
        return any(a.mode == "shadow" or (a.mode == "gated" and a.tau is not None) for a in self.arms.values())


# ----------------------------------------------------------------------------------------------------------- policy
@dataclass
class EpisodeState:
    arm: str
    episode: str
    tier: str = STRONG                # tier of the latest routed step
    seg_len: int = 0                  # consecutive fast steps so far (0 on strong)
    decided: dict[int, tuple[str, str, float | None]] = field(default_factory=dict)   # step -> (tier, reason, c)
    pending: dict[int, "asyncio.Future[dict[str, Any]]"] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ledger: Ledger = field(default_factory=Ledger)


def route_step(st: EpisodeState, step: int, c: float | None, tau: float | None, k: int,
               answer_ok: bool = True) -> tuple[str, str]:
    """The gated policy for one step (pure; updates ``st.tier`` / ``st.seg_len``). Returns (tier, reason).
    ``answer_ok``: E29b's typed question routes only when the decision model's answer is ``inspect`` (E29: always)."""
    if step == 0:
        tier, reason = STRONG, "first"
    elif tau is None:
        tier, reason = STRONG, "no_gate"
    elif c is None:
        tier, reason = STRONG, "fail_open"
    elif c < tau:
        tier, reason = STRONG, "below_tau"
    elif not answer_ok:
        tier, reason = STRONG, "other_answer"
    elif st.tier == FAST and st.seg_len >= k:
        tier, reason = STRONG, "cap_k"
    else:
        tier, reason = FAST, ("continue" if st.tier == FAST else "enter")
    st.seg_len = st.seg_len + 1 if tier == FAST and st.tier == FAST else (1 if tier == FAST else 0)
    st.tier = tier
    return tier, reason


# ----------------------------------------------------------------------------------------------------------- proxy
class UpstreamError(Exception):
    def __init__(self, status: int, body: bytes, msg: str = ""):
        super().__init__(msg or f"HTTP {status}")
        self.status, self.body = status, body


class RouteProxy:
    def __init__(self, cfg: ProxyConfig, backend: Any = None, api_key: str | None = None,
                 log_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.backend = backend
        self._api_key = api_key if api_key is not None else os.environ.get(cfg.key_env, "")
        self.log_dir = Path(log_dir or cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes: dict[tuple[str, str], EpisodeState] = {}
        self.spent = self._spent_in_logs()
        self.spent_at_start = self.spent
        self.n_calls = 0
        self.n_refused = 0
        self._bg: set[asyncio.Task[Any]] = set()
        self._pool = ThreadPoolExecutor(max_workers=int(cfg.decision.get("threads", 32)))
        self._client: ClientSession | None = None
        self.question = None
        self.questions: dict[str, Any] = {}
        if backend is not None:
            from ..decision.types import Choice
            self.question = Choice(QUESTION_TEXT, list(OPTIONS), list(OPTION_DESCRIPTIONS), decision_point="routing")
            self.question.id = QUESTION_ID
            step_q = Choice(STEP_QUESTION_TEXT, list(STEP_OPTIONS), list(STEP_OPTION_DESCRIPTIONS),
                            decision_point="routing")
            step_q.id = STEP_QUESTION_ID
            self.questions = {"route": self.question, "step_type": step_q}

    # ---------------------------------------------------------------- logs and spend
    def _spent_in_logs(self) -> float:
        total = 0.0
        for name in ("calls.jsonl", "decisions.jsonl"):
            p = self.log_dir / name
            if not p.exists():
                continue
            for ln in p.read_text(errors="replace").splitlines():
                try:
                    total += float(json.loads(ln).get("cost_usd") or 0.0)
                except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                    continue
        return total

    def _log(self, name: str, rec: dict[str, Any]) -> None:
        cost = rec.get("cost_usd")
        if isinstance(cost, (int, float)):
            self.spent += float(cost)
        with open(self.log_dir / name, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def capped(self) -> bool:
        """At the cap (minus the margin) no new episode starts; running episodes go on until ``hard_capped``."""
        return self.cfg.cap_usd is not None and self.spent >= float(self.cfg.cap_usd) - self.cfg.cap_margin_usd

    def hard_capped(self) -> bool:
        return self.cfg.cap_usd is not None and self.spent >= float(self.cfg.cap_usd) + self.cfg.cap_hard_extra_usd

    # ---------------------------------------------------------------- upstream
    async def client(self) -> ClientSession:
        if self._client is None or self._client.closed:
            ctx = None
            ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
            if ca and Path(ca).exists():
                ctx = ssl.create_default_context(cafile=ca)
            self._client = ClientSession(timeout=ClientTimeout(total=self.cfg.request_timeout_s, sock_connect=30),
                                         connector=TCPConnector(limit=256, ssl=ctx if ctx is not None else True),
                                         trust_env=True)
        return self._client

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.fmt == F.FORMAT_GEMINI:
            h["x-goog-api-key"] = self._api_key
        else:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _url(self, tier: str, method: str | None, query: str) -> str:
        if self.cfg.fmt == F.FORMAT_GEMINI:
            url = f"{self.cfg.upstream}/models/{self.cfg.tiers[tier].model}:{method}"
            return f"{url}?{query}" if query else url
        return f"{self.cfg.upstream}/chat/completions"

    def _body(self, tier: str, body: Mapping[str, Any], stream: bool | None = None) -> dict[str, Any]:
        if self.cfg.fmt == F.FORMAT_GEMINI:
            b = json.loads(json.dumps(body))
            F.fill_missing_signatures(b)
            return b
        t = self.cfg.tiers[tier]
        return F.openai_rewrite(body, t.model, t.overrides, stream=stream)

    def _cost(self, tier: str, usage: Mapping[str, Any]) -> tuple[float, bool]:
        if "reported_cost" in usage:
            return float(usage["reported_cost"]), True
        c = F.price_cost(self.cfg.tiers[tier].prices, usage)
        return (c, True) if c is not None else (0.0, False)

    async def _post(self, url: str, payload: dict[str, Any]) -> tuple[int, bytes, Any]:
        cl = await self.client()
        resp = await cl.post(url, data=json.dumps(payload).encode(), headers=self._headers())
        if resp.status != 200:
            raw = await resp.read()
            resp.release()
            return resp.status, raw, None
        return 200, b"", resp

    async def call(self, tier: str, method: str | None, query: str, body: Mapping[str, Any],
                   stream_to: web.StreamResponse | None = None, request: web.Request | None = None
                   ) -> tuple[int, bytes, dict[str, Any]]:
        """One upstream call with the proxy's retries and the signature fix. With ``stream_to`` (a prepared-on-demand
        StreamResponse) a 200 body is relayed as it arrives. Returns (status, raw body, info)."""
        payload = self._body(tier, body, stream=None)
        url = self._url(tier, method, query)
        info: dict[str, Any] = {"attempts": 0, "signature_fix": False}
        delay = 2.0
        status, raw = 0, b""
        for attempt in range(self.cfg.retries + 2):
            info["attempts"] = attempt + 1
            try:
                status, raw, resp = await self._post(url, payload)
            except (ClientError, asyncio.TimeoutError) as e:
                status, raw, resp = 599, f"{type(e).__name__}: {e}".encode(), None
            if status == 200 and resp is not None:
                chunks: list[bytes] = []
                try:
                    if stream_to is not None and request is not None:
                        stream_to.content_type = resp.headers.get("Content-Type", "text/event-stream").split(";")[0]
                        await stream_to.prepare(request)
                        info["relayed"] = True
                        async for chunk in resp.content.iter_any():
                            chunks.append(chunk)
                            await stream_to.write(chunk)
                        raw = b"".join(chunks)
                    else:
                        raw = await resp.read()
                except (ClientError, asyncio.TimeoutError, ConnectionError) as e:
                    # the stream broke after the client got part of it: nothing to retry (the harness retries)
                    info["relay_error"] = f"{type(e).__name__}: {e}"[:300]
                    return 599, b"".join(chunks), info
                finally:
                    resp.release()
                return 200, raw, info
            if self.cfg.fmt == F.FORMAT_GEMINI and F.is_signature_error(status, raw.decode("utf-8", "replace")) \
                    and not info["signature_fix"]:
                F.fill_missing_signatures(payload, replace_all=True)
                info["signature_fix"] = True
                continue
            if status in RETRY_STATUS and attempt < self.cfg.retries:
                await asyncio.sleep(delay)
                delay *= 2.5
                continue
            break
        return status, raw, info

    # ---------------------------------------------------------------- decisions
    async def _ask(self, st: EpisodeState, step: int, state: dict[str, Any], qname: str = "route") -> dict[str, Any]:
        t0 = time.monotonic()
        timeout = float(self.cfg.decision.get("timeout_s", 30.0))
        question = self.questions.get(qname, self.question)
        rec: dict[str, Any] = {"ts": utc_now(), "arm": st.arm, "episode": st.episode, "step": step,
                               "state_chars": len(json.dumps(state, ensure_ascii=False)),
                               "state_sha": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]}
        if qname != "route":
            rec["question"] = qname
        try:
            loop = asyncio.get_running_loop()
            res = await asyncio.wait_for(loop.run_in_executor(
                self._pool, lambda: self.backend.decide(state, [question], ledger=st.ledger,
                                                        decision_point="routing")), timeout=timeout)
            ans = res.answers[0]
            dist = {k: float(v) for k, v in (ans.distribution or {}).items()}
            c = dist.get(ROUTE_OPTION[qname])
            rec.update({"c": c, "dist": dist, "value": ans.value, "confidence": float(ans.confidence),
                        "cost_usd": float(res.usage.cost_usd or 0.0), "input_tokens": res.usage.input_tokens,
                        "backend_latency_ms": res.usage.latency_ms, "model": res.usage.model})
        except Exception as e:  # noqa: BLE001 - fail open: any decision-model failure routes the step to strong
            rec.update({"c": None, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        rec["latency_s"] = round(time.monotonic() - t0, 4)
        self._log("decisions.jsonl", rec)
        return rec

    def _decide_ahead(self, st: EpisodeState, body: Mapping[str, Any], parsed: F.Parsed, step: int,
                      qname: str = "route") -> None:
        """Start the decision for step ``step + 1`` (the action of ``step`` is running in the harness)."""
        if self.backend is None or self.question is None:
            return
        task, steps = F.episode(self.cfg.fmt, body)
        state = F.render_state(task, steps, parsed.call or "(no tool call)", parsed.text)
        nxt = step + 1
        if nxt in st.pending:
            return
        fut = asyncio.ensure_future(self._ask(st, nxt, state, qname))
        st.pending[nxt] = fut
        for old in [s for s in st.pending if s < nxt - 2]:
            st.pending.pop(old, None)

    async def _await_decision(self, st: EpisodeState, step: int
                              ) -> tuple[float | None, float, str | None, str | None]:
        """(c, seconds waited, fail-open reason, the decision model's answer)."""
        fut = st.pending.get(step)
        if fut is None:
            return None, 0.0, "no_decision", None
        t0 = time.monotonic()
        try:
            rec = await asyncio.wait_for(asyncio.shield(fut), timeout=float(self.cfg.decision.get("wait_s", 10.0)))
        except asyncio.TimeoutError:
            return None, time.monotonic() - t0, "wait_timeout", None
        return rec.get("c"), time.monotonic() - t0, (rec.get("error") and "decision_error"), rec.get("value")

    # ---------------------------------------------------------------- shadow
    async def _shadow(self, st: EpisodeState, step: int, method: str | None, query: str,
                      body: Mapping[str, Any], roles: list[str] | None = None) -> None:
        sm = "generateContent" if self.cfg.fmt == F.FORMAT_GEMINI else None
        q = "" if self.cfg.fmt == F.FORMAT_GEMINI else query
        for role, tier in (("shadow_fast", FAST), ("shadow_strong", STRONG)):
            if roles is not None and tier not in roles:
                continue
            t0 = time.monotonic()
            b = dict(body)
            if self.cfg.fmt == F.FORMAT_OPENAI:
                b["stream"] = False
            status, raw, info = await self.call(tier, sm, q, b)
            self._record_call(st, step, tier, role, status, raw, False, time.monotonic() - t0, info)

    # ---------------------------------------------------------------- records
    def _record_call(self, st: EpisodeState, step: int, tier: str, role: str, status: int, raw: bytes,
                     streamed: bool, latency: float, info: Mapping[str, Any], route: Mapping[str, Any] | None = None,
                     guard: bool = False) -> F.Parsed:
        parsed = F.parse_response(self.cfg.fmt, raw, streamed) if status == 200 else F.Parsed()
        cost, priced = self._cost(tier, parsed.usage) if status == 200 else (0.0, True)
        model = self.cfg.tiers[tier].model
        rec: dict[str, Any] = {"ts": utc_now(), "arm": st.arm, "episode": st.episode, "step": step, "tier": tier,
                               "role": role, "model": model, "model_version": parsed.model_version, "status": status,
                               "latency_s": round(latency, 4), "streamed": streamed, "cost_usd": cost,
                               "priced": priced, **{k: parsed.usage.get(k) for k in
                                                    ("input_tokens", "cached_tokens", "output_tokens")},
                               "key": parsed.key, "ro": parsed.read_only, "attempts": info.get("attempts"),
                               "signature_fix": info.get("signature_fix") or None}
        if route:
            rec.update(route)
        if guard and status == 200:       # E29b: the write guard's verdict on a fast-LLM response
            rec["guard"] = "accept" if parsed.read_only else "reject"
        if status != 200:
            rec["error"] = raw[:300].decode("utf-8", "replace")
        self.n_calls += 1
        self._log("calls.jsonl", rec)
        st.ledger.add(CallRecord(component="s2", backend=self.cfg.fmt, model=model, decision_point=role,
                                 input_tokens=int(parsed.usage.get("input_tokens") or 0),
                                 output_tokens=int(parsed.usage.get("output_tokens") or 0), cost_usd=cost,
                                 latency_ms=latency * 1000.0, priced=priced, step=step,
                                 meta={"tier": tier, "cached_tokens": parsed.usage.get("cached_tokens")}))
        return parsed

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ---------------------------------------------------------------- request handling
    def _episode(self, arm: str, episode: str) -> EpisodeState:
        key = (arm, episode)
        if key not in self.episodes:
            self.episodes[key] = EpisodeState(arm=arm, episode=episode)
        return self.episodes[key]

    async def handle(self, request: web.Request, arm: str, episode: str, method: str | None) -> web.StreamResponse:
        if arm not in self.cfg.arms:
            return _error(404, f"unknown arm {arm!r}")
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _error(400, "request body is not JSON")
        step = F.step_index(self.cfg.fmt, body)
        # spend cap: no new episode once the cap is reached (the host watchdog then stops the runs, so running
        # episodes are cancelled rather than failed); every request is refused past the hard limit
        if (self.capped() and (step == 0 or (arm, episode) not in self.episodes)) or self.hard_capped():
            self.n_refused += 1
            return _error(503, "E29 spend cap reached: the proxy refuses this request")
        acfg = self.cfg.arms[arm]
        st = self._episode(arm, episode)
        query = urlencode([(k, v) for k, v in request.query.items() if k != "key"])
        streamed = bool(method and method.startswith("streamGenerateContent")) or bool(body.get("stream"))
        if self.cfg.fmt == F.FORMAT_OPENAI and body.get("stream"):
            return _error(400, "streaming chat completions are not supported by this proxy")

        # ---- route
        waited, c, reason = 0.0, None, None
        async with st.lock:
            if acfg.mode != "gated":
                tier, reason = STRONG, acfg.mode
            elif step in st.decided:
                tier, reason, c = st.decided[step]
                reason = f"repeat:{reason}"
            else:
                answer = None
                if step > 0 and acfg.tau is not None:
                    c, waited, why, answer = await self._await_decision(st, step)
                ok = acfg.question == "route" or answer == ROUTE_OPTION[acfg.question]
                tier, reason = route_step(st, step, c, acfg.tau, acfg.k, answer_ok=ok)
                if c is None and step > 0 and acfg.tau is not None:
                    reason = f"fail_open:{why}"
                st.decided[step] = (tier, reason, c)
        route = {"route_reason": reason, "c": c, "waited_s": round(waited, 4), "seg_len": st.seg_len,
                 "tau": acfg.tau}

        # ---- forward (a failed fast request goes to the strong LLM once). E29b's write guard: a fast response is
        # buffered (never relayed while it streams) and returned only if it is read-only; otherwise it is discarded
        # and the strong LLM writes the step from the same request.
        guarded = acfg.mode == "gated" and acfg.guard and tier == FAST
        out = web.StreamResponse() if (streamed and not guarded) else None
        t0 = time.monotonic()
        status, raw, info = await self.call(tier, method, query, body, stream_to=out, request=request)
        parsed = self._record_call(st, step, tier, "primary", status, raw, streamed, time.monotonic() - t0, info,
                                   route, guard=guarded)
        if guarded and status == 200 and not parsed.read_only:
            st.tier, st.seg_len = STRONG, 0
            st.decided[step] = (STRONG, "guard_reject", c)
            tier = STRONG
            out = web.StreamResponse() if streamed else None
            t0 = time.monotonic()
            status, raw, info = await self.call(tier, method, query, body, stream_to=out, request=request)
            parsed = self._record_call(st, step, tier, "primary", status, raw, streamed, time.monotonic() - t0, info,
                                       {**route, "route_reason": "guard_reject", "seg_len": 0})
        elif status != 200 and tier == FAST and not info.get("relayed"):
            st.tier, st.seg_len = STRONG, 0
            st.decided[step] = (STRONG, "escalate_fast_error", c)
            tier = STRONG
            t0 = time.monotonic()
            status, raw, info = await self.call(tier, method, query, body, stream_to=out, request=request)
            parsed = self._record_call(st, step, tier, "primary", status, raw, streamed, time.monotonic() - t0, info,
                                       {**route, "route_reason": "escalate_fast_error"})

        # ---- after the response: decide ahead (gated with a gate, shadow) and shadow calls
        if status == 200:
            if acfg.mode == "shadow" or (acfg.mode == "gated" and acfg.tau is not None):
                self._decide_ahead(st, body, parsed, step, acfg.question)
            if acfg.mode == "shadow":
                self._spawn(self._shadow(st, step, method, query, body, list(acfg.shadow_roles)))
        if out is not None and info.get("relayed"):
            with contextlib.suppress(ConnectionError, RuntimeError):
                await out.write_eof()
            return out
        ctype = "text/event-stream" if status == 200 and streamed else "application/json"
        return web.Response(status=502 if status == 599 else status, body=raw, content_type=ctype)

    async def dispatch(self, request: web.Request) -> web.StreamResponse:
        """``/e29/<arm>/<episode>/<rest>``: ``rest`` = ``[v1beta/]models/<model>:<method>`` (Gemini; repeated slashes
        are tolerated) or ``[api/]v1/chat/completions`` (OpenAI)."""
        arm, ep, rest = request.match_info["arm"], request.match_info["episode"], request.match_info["rest"]
        rest = "/".join(p for p in rest.split("/") if p)
        if self.cfg.fmt == F.FORMAT_OPENAI:
            if not rest.endswith("chat/completions"):
                return _error(404, "this proxy speaks OpenAI chat completions (POST .../chat/completions)")
            return await self.handle(request, arm, ep, None)
        if "models/" not in rest:
            return _error(404, "expected .../models/<model>:<method> (this proxy speaks Gemini's native API)")
        mm = rest.split("models/", 1)[1]
        if ":" not in mm:
            return _error(404, "expected models/<model>:<method>")
        method = mm.split(":", 1)[1]
        if method not in ("generateContent", "streamGenerateContent"):
            return _error(404, f"unsupported method {method!r}")
        return await self.handle(request, arm, ep, method)

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "format": self.cfg.fmt, "spent_usd": round(self.spent, 6),
                                  "spent_at_start_usd": round(self.spent_at_start, 6), "cap_usd": self.cfg.cap_usd,
                                  "capped": self.capped(), "arms": {k: vars(v) for k, v in self.cfg.arms.items()},
                                  "tiers": {k: v.model for k, v in self.cfg.tiers.items()},
                                  "episodes": len(self.episodes), "calls": self.n_calls, "refused": self.n_refused,
                                  "decision_backend": (self.backend.describe() if self.backend is not None else None)})

    async def drain(self) -> None:
        """Wait for background shadow calls and pending decisions (tests, shutdown)."""
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)
        pend = [f for st in self.episodes.values() for f in st.pending.values() if not f.done()]
        if pend:
            await asyncio.gather(*pend, return_exceptions=True)

    async def close(self) -> None:
        await self.drain()
        if self._client is not None:
            await self._client.close()
        self._pool.shutdown(wait=False)

    def app(self) -> web.Application:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_get("/health", self.health)
        app.router.add_post("/e29/{arm}/{episode}/{rest:.*}", self.dispatch)

        async def _cleanup(_app: web.Application) -> None:
            await self.close()
        app.on_cleanup.append(_cleanup)
        return app


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": {"code": status, "message": message}}, status=status)


# ----------------------------------------------------------------------------------------------------------- CLI
def load_config(path: str | Path, sets: list[str] | None = None) -> ProxyConfig:
    d = json.loads(Path(path).read_text())
    for s in sets or []:
        k, _, v = s.partition("=")
        cur: Any = d
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        try:
            cur[parts[-1]] = json.loads(v)
        except json.JSONDecodeError:
            cur[parts[-1]] = v
    return ProxyConfig.from_dict(d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="override a config value: dotted.key=JSON")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--log-dir", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if not os.environ.get(cfg.key_env):
        print(f"refusing: {cfg.key_env} is not set", file=sys.stderr)
        return 2
    backend = None
    if cfg.needs_decisions():
        from ..decision import make_backend
        backend = make_backend(cfg.decision.get("backend") or {"type": "jev", "provider": "openrouter"})
    proxy = RouteProxy(cfg, backend=backend, log_dir=a.log_dir)
    print(f"route proxy: {cfg.fmt} -> {cfg.upstream}; arms {sorted(cfg.arms)}; spent so far ${proxy.spent:.4f}; "
          f"cap {cfg.cap_usd}; logs {proxy.log_dir}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        web.run_app(proxy.app(), host=a.host, port=a.port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
