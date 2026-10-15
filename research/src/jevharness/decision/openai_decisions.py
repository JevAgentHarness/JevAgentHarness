"""Client for OpenAI's Decisions API (public beta; https://developers.openai.com/api/docs/guides/decisions).

Exploratory row E15 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E15: more decision-model baselines"). The API is a
hosted decision model (a specialised ``gpt-6-luna`` that scores only the allowed options), the closest hosted rival to
Jev. Wire format (the guide):

    POST https://api.openai.com/v1/decisions        Authorization: Bearer $OPENAI_API_KEY
    {"model": "gpt-6-luna", "input": "<state text>",
     "questions": [{"type": "choice", "name": "q0", "instructions": "...",
                    "choices": [{"value": "cancel_order", "description": "Cancel an order"}, ...]},
                   {"type": "predicate", "name": "q1", "instructions": "..."},
                   {"type": "score", "name": "q2", "instructions": "...",
                    "levels": [{"label": "1", "description": "..."}, ...]}]}
    ->
    {"model": "gpt-6-luna",
     "answers": [{"type": "choice", "name": "q0", "choice": "cancel_order",
                  "probabilities": [{"value": "cancel_order", "probability": 0.97}, ...], "confidence": 0.97},
                 {"type": "predicate", "name": "q1", "probability": 0.87},
                 {"type": "score", "name": "q2", "score": 2.4, "probabilities": [...], "confidence": 0.6},
                 {"type": "refusal", "name": "q3"}],
     "usage": {"input_tokens": 128, "output_tokens": 0}}

``questions`` is an array; ``name`` maps answers back to questions (q0, q1, ... here; answers without a name are taken
in order). Several questions about one state go in one request, as with Jev. Price: $0.10 per 1M input tokens, no output
or cache charge (``configs/prices/models.yaml``: ``openai-decisions/gpt-6-luna``).

A **refusal** (an answer of type ``refusal``) is no answer (registration (b) 1): the returned :class:`Answer` has a flat
distribution, confidence 0 and ``raw["refusal"] = True``; ``decision_eval`` and the corpus replay score it as wrong and
count it, and the harness never executes a refused tool choice in REPLACE mode (it falls through to the agent LLM).
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

from ..accounting import default_prices
from .base import DecisionBackend
from .types import JEV_MAX_OPTIONS, Answer, DecisionResult, QKind, Question, Usage, answer_from_distribution, normalize

DEFAULT_BASE_URL = "https://api.openai.com/v1"
PRICE_MODEL = "openai-decisions/gpt-6-luna"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class OpenAIDecisionsError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"OpenAI Decisions API error {status}: {body[:500]}")
        self.status = status
        self.body = body


def question_to_api(q: Question, name: str, describe_empty: bool = False) -> dict[str, Any]:
    """One harness question in the API's schema. A Choice option without a description is sent as ``{"value": ...}``
    (``describe_empty`` sends ``"description": ""`` instead, should the API require the field)."""
    if q.kind == QKind.CHOICE:
        assert q.options is not None
        descs = q.option_descriptions or [None] * len(q.options)
        choices = []
        for opt, d in zip(q.options, descs):
            c: dict[str, Any] = {"value": opt}
            if d:
                c["description"] = d
            elif describe_empty:
                c["description"] = ""
            choices.append(c)
        return {"type": "choice", "name": name, "instructions": q.text, "choices": choices}
    if q.kind == QKind.NOUL:
        return {"type": "predicate", "name": name, "instructions": q.text}
    lo, hi = q.scale  # type: ignore[misc]
    descs = q.levels or [None] * (hi - lo + 1)
    levels = []
    for v, d in zip(range(lo, hi + 1), descs):
        lv: dict[str, Any] = {"label": str(v)}
        if d:
            lv["description"] = d
        elif describe_empty:
            lv["description"] = ""
        levels.append(lv)
    instr = q.text + (f"\nRubric: {q.rubric}" if q.rubric else "")
    return {"type": "score", "name": name, "instructions": instr, "levels": levels}


def _probs(a: dict[str, Any]) -> dict[str, float]:
    p = a.get("probabilities")
    if isinstance(p, dict):
        return {str(k): float(v) for k, v in p.items()}
    out: dict[str, float] = {}
    for e in p or []:
        if isinstance(e, dict):
            key = e.get("value", e.get("label", e.get("choice")))
            if key is not None and e.get("probability") is not None:
                out[str(key)] = float(e["probability"])
    return out


def refusal_answer(q: Question, raw: dict[str, Any]) -> Answer:
    dist = normalize({lab: 1.0 for lab in q.labels()}, q.labels())
    ans = answer_from_distribution(q, dist, confidence=0.0, raw={**raw, "refusal": True})
    return ans


def answer_from_api(q: Question, a: dict[str, Any]) -> Answer:
    t = a.get("type")
    if t == "refusal":
        return refusal_answer(q, a)
    if q.kind == QKind.CHOICE:
        probs = _probs(a) or ({str(a["choice"]): 1.0} if a.get("choice") is not None else {})
        if not probs:
            raise OpenAIDecisionsError(200, f"choice answer without a choice or probabilities: {str(a)[:300]}")
        return answer_from_distribution(q, probs, confidence=a.get("confidence"), raw=a)
    if q.kind == QKind.NOUL:
        if a.get("probability") is None:
            raise OpenAIDecisionsError(200, f"predicate answer without a probability: {str(a)[:300]}")
        p = float(a["probability"])
        return answer_from_distribution(q, {"yes": p, "no": 1.0 - p}, raw=a)
    lo, _ = q.scale  # type: ignore[misc]
    dist = _probs(a)
    ans = answer_from_distribution(q, dist, confidence=a.get("confidence"), raw=a)
    if a.get("score") is not None and not dist:
        ans.value = float(a["score"])
    return ans


class OpenAIDecisionsBackend(DecisionBackend):
    name = "openai_decisions"
    batched_questions = True
    max_options = JEV_MAX_OPTIONS    # no documented limit; the harness treats it as hosted Jev (one read up to 255)

    def __init__(self, model: str = "gpt-6-luna", api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, max_retries: int = 4, max_questions_per_request: int = 64,
                 describe_empty: bool = False, price_model: str = PRICE_MODEL, name: str | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_q = max(1, int(max_questions_per_request))
        self.describe_empty = describe_empty
        self.price_model = price_model
        if name:
            self.name = name
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)
        self.served_models: set[str] = set()

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, str], int, float]:
        delay, waited = 1.0, 0.0
        for attempt in range(self.max_retries + 1):
            t = time.perf_counter()
            try:
                r = self._http.post("/decisions", json=body)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == self.max_retries:
                    raise
                nap = min(20.0, delay) + random.uniform(0, 0.5)
                time.sleep(nap)
                waited += nap
                delay *= 2
                continue
            ms = (time.perf_counter() - t) * 1000
            if r.status_code == 200:
                return r.json(), ms, dict(r.headers), attempt + 1, waited
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                ra = r.headers.get("retry-after-ms") or r.headers.get("retry-after")
                try:
                    wait = float(ra) / (1000.0 if "retry-after-ms" in r.headers else 1.0) if ra else delay
                except ValueError:
                    wait = delay
                nap = min(20.0, wait) + random.uniform(0, 0.5)
                time.sleep(nap)
                waited += nap
                delay *= 2
                continue
            raise OpenAIDecisionsError(r.status_code, r.text)
        raise RuntimeError("unreachable")

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        answers: list[Answer] = []
        usage = Usage(backend=self.name, model=self.model, n_calls=0, attempts=0, retry_wait_ms=0.0)
        for i in range(0, len(questions), self.max_q):
            chunk = questions[i:i + self.max_q]
            names = [f"q{j}" for j in range(len(chunk))]
            body = {"model": self.model, "input": state,
                    "questions": [question_to_api(q, n, self.describe_empty) for q, n in zip(chunk, names)]}
            data, ms, headers, attempts, waited = self._post(body)
            got = data.get("answers") or []
            by_name = {str(a.get("name")): a for a in got if isinstance(a, dict) and a.get("name") is not None}
            for j, (q, n) in enumerate(zip(chunk, names)):
                a = by_name.get(n)
                if a is None:
                    if len(got) != len(chunk) or not isinstance(got[j], dict):
                        raise OpenAIDecisionsError(200, f"no answer for {n}: {str(data)[:300]}")
                    a = got[j]
                answers.append(answer_from_api(q, a))
            u = data.get("usage") or {}
            served = data.get("model") or self.model
            self.served_models.add(served)
            usage.model = served
            usage.input_tokens += int(u.get("input_tokens", 0) or 0)
            usage.output_tokens += int(u.get("output_tokens", 0) or 0)
            usage.latency_ms += ms
            usage.n_calls += 1
            usage.attempts = (usage.attempts or 0) + attempts
            usage.retry_wait_ms = (usage.retry_wait_ms or 0.0) + waited * 1000
            rid = headers.get("x-request-id") or data.get("id")
            if rid:
                usage.request_id = str(rid) if usage.request_id is None else f"{usage.request_id},{rid}"
        prices = default_prices()
        usage.cost_usd = prices.cost(self.price_model, usage.input_tokens, usage.output_tokens)
        usage.priced = prices.get(self.price_model) is not None
        return DecisionResult(answers=answers, usage=usage)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"backend": self.name, "provider": "openai", "model": self.model, "base_url": self.base_url,
                             "max_questions_per_request": self.max_q}
        if self.describe_empty:
            d["describe_empty"] = True
        return d
