"""Abstract System-1 decision backend + caching wrapper."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..accounting import CallRecord, Ledger
from .types import DecisionResult, QKind, Question, Usage


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=1, default=str)


class DecisionBackend(ABC):
    """A System-1 model: (state, typed questions) -> calibrated typed answers.

    Subclasses implement :meth:`_decide`. :meth:`decide` adds ledger bookkeeping so the
    harness never has to think about accounting.
    """

    name: str = "base"
    model: str | None = None
    # True when all questions about a shared state are answered by ONE call (one request: Jev/OpenJev, the batched
    # verbalised deciders; one multi-field forward pass: Intern-Decision) -- not a claim about the provider's passes.
    batched_questions: bool = False
    # max options per Choice in a single call (letter-label deciders: 52; hosted Jev: 255). Wider Choices are answered
    # by a multi-pass knockout (decision/tournament.py), inside the backend or by the harness tournament.
    max_options: int = 52

    @abstractmethod
    def _decide(self, state: str, questions: list[Question]) -> DecisionResult: ...

    def decide(self, state: Any, questions: list[Question], ledger: Ledger | None = None,
               decision_point: str | None = None) -> DecisionResult:
        res = self._decide(render_state(state), questions)
        if ledger is not None:
            dp = decision_point or (questions[0].decision_point if questions else None)
            u = res.usage
            meta: dict[str, Any] = {"n_questions": len(questions), "n_calls": u.n_calls}
            # review P1-4: forward passes / sequential rounds, when the backend observes them (decision/tournament.py)
            meta.update({k: v for k, v in (("n_passes", u.n_passes), ("n_rounds", u.n_rounds)) if v is not None})
            if u.server:          # serve_jev's own timing / prefix-cache counters (harness_final diagnosis)
                meta["server"] = dict(u.server)
            ledger.add(CallRecord(component="s1", backend=u.backend, model=u.model, decision_point=dp,
                                  input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                                  cost_usd=u.cost_usd, latency_ms=u.latency_ms, cached=u.cached,
                                  priced=u.priced is not False,
                                  meta=meta,
                                  attempts=u.attempts, retry_wait_ms=u.retry_wait_ms, request_id=u.request_id))
        return res

    def ask(self, state: Any, question: Question, ledger: Ledger | None = None):
        """Convenience: single question -> single Answer."""
        return self.decide(state, [question], ledger=ledger).answers[0]

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model}


class CachedBackend(DecisionBackend):
    """SQLite-backed cache. On a hit we *replay* the original usage (latency, cost) and
    mark it ``cached=True`` so reruns are free but reported efficiency stays faithful."""

    def __init__(self, inner: DecisionBackend, path: str | Path = "cache/decisions.sqlite"):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.batched_questions = inner.batched_questions
        self.max_options = inner.max_options
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def _key(self, state: str, questions: list[Question]) -> str:
        body: dict[str, Any] = {
            "b": self.inner.describe(),
            "s": state,
            "q": [q.model_dump(exclude={"id", "meta", "decision_point"}) for q in questions],
        }
        # ``meta`` stays out of the key (it holds gold labels and bookkeeping), except Noul ``criteria``,
        # which is sent on the Jev wire and changes the answer. Added only when present so existing
        # cache entries stay valid.
        crit = [q.meta.get("criteria") for q in questions]
        if any(c is not None for c in crit):
            body["c"] = crit
        # a Choice wider than one read is a knockout whose full-set distribution depends on the composition method
        # (decision/tournament.py; "finalists" before 2026-10-02): keyed only for those questions, so every other
        # cache entry stays valid
        ko = knockout_method(self.inner)
        if ko is not None and any(q.kind == QKind.CHOICE and len(q.options or []) > self.max_options
                                  for q in questions):
            body["ko"] = ko
        payload = json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        key = self._key(state, questions)
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if row:
            res = DecisionResult.model_validate_json(row[0])
            # re-map answer ids to the fresh question ids (ids are excluded from the key)
            for a, q in zip(res.answers, questions):
                a.question_id = q.id
            res.usage.cached = True
            return res
        res = self.inner._decide(state, questions)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, res.model_dump_json()))
            self._db.commit()
        return res

    def describe(self) -> dict[str, Any]:
        return {**self.inner.describe(), "cached": True}


def knockout_method(backend: Any) -> str | None:
    """The knockout composition method (``knockout_probs``) of a backend or of the backend it wraps, if it has one."""
    seen = 0
    while backend is not None and seen < 8:
        m = getattr(backend, "knockout_probs", None)
        if m is not None:
            return str(m)
        backend, seen = getattr(backend, "inner", None), seen + 1
    return None


def usage_from(backend: str, model: str | None, **kw: Any) -> Usage:
    return Usage(backend=backend, model=model, **kw)


def backend_id(desc: dict[str, Any] | None) -> str | None:
    """A short, stable identity of a System-1 backend *spec*: a hash of its ``describe()`` (backend, model, version,
    mode, rate, seed, ...) without the cache flag, so a cached and an uncached instance of one spec share it while
    two specs with one backend name (``rate(0.1)`` / ``rate(0.3)``, two LLM deciders) do not. Logged per episode
    (``Trajectory.meta["s1"]``) and per replay row; analyses match backends on it, never on the name alone."""
    if not desc:
        return None
    d = {k: v for k, v in desc.items() if k != "cached"}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]
