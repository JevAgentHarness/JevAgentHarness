"""An ensemble of decision models behind ONE System-1 interface (exploratory row E20, docs/PREREG_A7_STAGE2.md,
Deviations 2026-10-08 "E20: decision-model ensembles and a hosted decision API behind the gate").

Every member is a decision model (never an LLM voter); the agent LLM only receives what the cascade defers.

    {"type": "ensemble", "rule": "agree", "name": "agree",
     "members": [{"type": "jev", "provider": "openrouter"}, {"type": "openai_decisions", "model": "gpt-6-luna"}]}

Rule ``agree`` (the registered gate of E20 (b) 2): each question set goes to every member (in parallel); when all
members give the same answer the ensemble returns it with **confidence = the smallest member confidence**; otherwise
(different answers, or a member without an answer: a refusal or an error) it returns the answer with the largest summed
member confidence and **confidence 0**, so a cascade delegates only when the members agree and the combined confidence
reaches its threshold. The returned distribution is the members' mean (logged only); ``raw["ensemble"]`` keeps every
member's answer and confidence. Rule ``weighted`` returns the largest summed confidence with that sum / k as the
confidence (not used live).

Accounting: each member's call goes through the Ledger under the member's own backend name, model and price (the
base class's bookkeeping, unchanged), tagged ``meta["ensemble"]`` = {"name", "group", "members"}; the calls of one
decision share the group id, so the parallel System-1 time of a decision is the slowest member's. The ensemble adds no
record of its own. :func:`combine` is the one place the rule lives: the offline calibration answers of E20 (c) are
computed with it from the members' stored answers.
"""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from ..accounting import CallRecord, Ledger
from .base import DecisionBackend
from .types import Answer, DecisionResult, QKind, Question, Usage

RULES = ("agree", "weighted")


def _no_answer(a: Answer | None) -> bool:
    return a is None or bool((a.raw or {}).get("refusal")) or bool((a.raw or {}).get("error"))


def _key(v: Any) -> Any:
    return v if not isinstance(v, float) else round(v, 9)


def combine(q: Question, answers: Sequence[Answer | None], rule: str = "agree",
            names: Sequence[str] | None = None) -> Answer:
    """The ensemble's answer to one question from its members' answers (None = the member gave no answer)."""
    if rule not in RULES:
        raise ValueError(f"unknown ensemble rule {rule!r}; one of {RULES}")
    names = list(names or [f"m{j}" for j in range(len(answers))])
    labels = q.labels()
    got = [a for a in answers if not _no_answer(a)]
    dist: dict[str, float] = {lab: 0.0 for lab in labels}
    for a in got:
        for lab in labels:
            dist[lab] += float((a.distribution or {}).get(lab, 0.0)) / max(1, len(got))
    if not got:
        dist = {lab: 1.0 / len(labels) for lab in labels}
    wsum: dict[Any, float] = {}
    order: dict[Any, int] = {}
    for j, a in enumerate(answers):
        if _no_answer(a):
            continue
        k = _key(a.value)  # type: ignore[union-attr]
        wsum[k] = wsum.get(k, 0.0) + float(a.confidence)  # type: ignore[union-attr]
        order.setdefault(k, j)
    raw = {"ensemble": {"rule": rule, "members": [
        {"name": n, "value": None if _no_answer(a) else a.value, "confidence": None if _no_answer(a) else a.confidence,
         "refusal": bool(a is not None and (a.raw or {}).get("refusal")), "error": a is None}
        for n, a in zip(names, answers)]}}
    if not wsum:
        value: Any = max(dist, key=dist.get) if q.kind == QKind.CHOICE else (dist.get("yes", 0.5) >= 0.5 if q.kind == QKind.NOUL else None)
        raw["ensemble"]["agree"] = False
        return Answer(question_id=q.id, kind=q.kind, distribution=dist, value=value, confidence=0.0, raw=raw)
    best = min(wsum, key=lambda k: (-wsum[k], order[k]))
    value = next(a.value for a in got if _key(a.value) == best)
    agree = len(got) == len(answers) and len(wsum) == 1
    if rule == "agree":
        conf = min(float(a.confidence) for a in got) if agree else 0.0
    else:
        conf = wsum[best] / len(answers)
    raw["ensemble"]["agree"] = agree
    return Answer(question_id=q.id, kind=q.kind, distribution=dist, value=value, confidence=conf, raw=raw)


class _GroupTag(Ledger):
    """A ledger view that tags every record with the ensemble group and forwards it to the decision's ledger."""

    def __init__(self, inner: Ledger | None, tag: dict[str, Any]):
        super().__init__()
        self.inner = inner
        self.tag = tag
        self.step = getattr(inner, "step", None)

    def add(self, rec: CallRecord) -> None:
        rec.meta = {**(rec.meta or {}), "ensemble": dict(self.tag)}
        if self.inner is not None:
            self.inner.add(rec)
        with self._lock:
            self.records.append(rec)


class EnsembleBackend(DecisionBackend):
    name = "ensemble"
    batched_questions = True

    def __init__(self, members: Sequence[dict[str, Any] | DecisionBackend], rule: str = "agree", name: str | None = None,
                 parallel: bool = True, member_names: Sequence[str] | None = None):
        from . import make_backend

        if rule not in RULES:
            raise ValueError(f"unknown ensemble rule {rule!r}; one of {RULES}")
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two decision models")
        self.members = [m if isinstance(m, DecisionBackend) else make_backend(dict(m)) for m in members]
        self.member_names = list(member_names or [getattr(m, "name", f"m{j}") for j, m in enumerate(self.members)])
        self.rule = rule
        self.parallel = parallel
        if name:
            self.name = name
        self.model = "+".join(str(getattr(m, "model", None) or m.name) for m in self.members)
        self.batched_questions = all(getattr(m, "batched_questions", False) for m in self.members)
        self.max_options = min(int(getattr(m, "max_options", 52)) for m in self.members)
        self._groups = itertools.count()
        self._glock = threading.Lock()

    def _member_results(self, state: str, questions: list[Question], ledger: Ledger | None,
                        decision_point: str | None) -> list[DecisionResult | None]:
        with self._glock:
            gid = next(self._groups)
        tag = {"name": self.name, "group": gid, "members": len(self.members)}

        def one(m: DecisionBackend) -> DecisionResult | None:
            try:
                return m.decide(state, questions, ledger=_GroupTag(ledger, tag), decision_point=decision_point)
            except Exception:  # noqa: BLE001 - a failed member is no answer (the ensemble defers), never a crash
                return None
        if self.parallel:
            with ThreadPoolExecutor(max_workers=len(self.members)) as ex:
                return list(ex.map(one, self.members))
        return [one(m) for m in self.members]

    def _assemble(self, questions: list[Question], results: list[DecisionResult | None]) -> DecisionResult:
        answers = []
        for i, q in enumerate(questions):
            per = [r.answers[i] if r is not None and i < len(r.answers) else None for r in results]
            answers.append(combine(q, per, self.rule, self.member_names))
        ok = [r for r in results if r is not None]
        usage = Usage(backend=self.name, model=self.model, n_calls=sum(r.usage.n_calls for r in ok),
                      input_tokens=sum(r.usage.input_tokens for r in ok), output_tokens=sum(r.usage.output_tokens for r in ok),
                      cost_usd=sum(r.usage.cost_usd for r in ok),
                      latency_ms=max([r.usage.latency_ms for r in ok], default=0.0),
                      cached=bool(ok) and all(r.usage.cached for r in ok),
                      priced=all(r.usage.priced is not False for r in ok))
        if len(ok) < len(results):
            usage.request_id = f"{len(results) - len(ok)} member(s) failed"
        return DecisionResult(answers=answers, usage=usage)

    def decide(self, state: Any, questions: list[Question], ledger: Ledger | None = None,
               decision_point: str | None = None) -> DecisionResult:
        """The members' own ``decide`` puts each member's call into the ledger; the ensemble adds no record."""
        from .base import render_state

        dp = decision_point or (questions[0].decision_point if questions else None)
        return self._assemble(questions, self._member_results(render_state(state), questions, ledger, dp))

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:   # without a ledger (caches, tests)
        return self._assemble(questions, self._member_results(state, questions, None, None))

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "type": "ensemble", "rule": self.rule,
                "members": [m.describe() for m in self.members], "member_names": self.member_names}
