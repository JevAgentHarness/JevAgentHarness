"""Client for TypeSafe's Jev System-One API and any Jev wire-compatible server.

Wire format (TypeSafe OpenAPI 0.2.0, mirrored by the official ``typesafe-sdk`` v0.7.x):

    POST {base}/v1/systemone          Authorization: Bearer $TYPESAFE_API_KEY
    {"state": str|dict|list, "model": "jev-latest",
     "questions": {"<id>": {"type": "choice", "instructions": "...", "criteria": {"label": "desc"|null}},
                   "<id>": {"type": "noul",   "instructions": "...", "criteria": {"true": ..., "false": ...}},
                   "<id>": {"type": "score",  "instructions": "...", "criteria": ["level0", "level1", ...]}}}
    ->
    {"model": "jev-1.13.0",
     "answers": {"<id>": {"type": "choice", "choice": "label", "confidence": 0.93, "probabilities": {...}},
                 "<id>": {"type": "noul", "noul": 0.87},
                 "<id>": {"type": "score", "score": 2.4, "confidence": 0.6, "probabilities": {"0": ..}}},
     "usage": {"input_tokens": 123, "output_tokens": 0}}

Limits (hosted Jev, per public docs): <=255 Choice options, <=10 Score levels, 64k-token
requests. Unknown response fields / answer types are tolerated. Retries follow the SDK
defaults: 408/429/5xx/529 with exponential backoff.

The same class talks to OpenJev (razorback16/openjev, DiffusionGemma), openjev-server
(HF openjev/openjev), and our own trained decision model (``training/serve_jev.py``), which
all implement the same endpoint -- only ``base_url`` / ``model`` change.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from typing import Any

import httpx

from ..accounting import default_prices
from .base import DecisionBackend
from .types import JEV_MAX_OPTIONS, Answer, DecisionResult, QKind, Question, Usage, answer_from_distribution

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}

# Hosts that serve the System One protocol. OpenRouter serves TypeSafe's protocol at
# /api/v1/systemone with model ids like "typesafe/jev-1.13" and reports the billed cost in
# usage.cost (verified against pi-ai's OpenRouter provider + tests, 2026-09-30).
PRESETS: dict[str, dict[str, str]] = {
    "typesafe": {"base_url": "https://api.typesafe.ai", "key_env": "TYPESAFE_API_KEY", "model": "jev-latest",
                 "price_model": "jev-latest"},
    "openrouter": {"base_url": "https://openrouter.ai/api", "key_env": "OPENROUTER_API_KEY",
                   "jev_key_env": "OPENROUTER_API_KEY_JEV",
                   "model": "typesafe/jev-1.13", "price_model": "jev-1.13.0"},
    "codiv": {"base_url": "https://api.codiv.ai", "key_env": "OPENJEV_API_KEY", "model": "openjev-latest",
              "price_model": "codiv-openjev"},
}


SERVER_USAGE_KEYS = ("prefilled_tokens", "cache_hit_tokens", "stateless_tokens", "lock_wait_ms", "compute_ms")


def _server_timing_ms(h: str | None) -> float | None:
    """``model;dur=12.3`` (serve_jev's Server-Timing header) -> 12.3."""
    if not h or "dur=" not in h:
        return None
    try:
        return float(h.split("dur=", 1)[1].split(",")[0].split(";")[0])
    except ValueError:
        return None


def replica_index(state: str, n: int) -> int:
    """The replica that serves a state: a hash of its first block (``TASK: ...``, constant over an episode)."""
    if n <= 1:
        return 0
    key = state.split("\n\n", 1)[0]
    return int(hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()[:8], 16) % n


class JevAPIError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Jev API error {status}: {body[:500]}")
        self.status = status
        self.body = body


def question_to_wire(q: Question) -> dict[str, Any]:
    if q.kind == QKind.CHOICE:
        assert q.options is not None
        descs = q.option_descriptions or [None] * len(q.options)
        return {"type": "choice", "instructions": q.text,
                "criteria": {opt: (d or None) for opt, d in zip(q.options, descs)}}
    if q.kind == QKind.NOUL:
        wire: dict[str, Any] = {"type": "noul", "instructions": q.text}
        crit = q.meta.get("criteria")
        if isinstance(crit, dict):
            wire["criteria"] = {k: v for k, v in crit.items() if k in ("true", "false")}
        return wire
    lo, hi = q.scale  # type: ignore[misc]
    levels = q.levels or [f"{v}" + (" (lowest)" if v == lo else " (highest)" if v == hi else "")
                          for v in range(lo, hi + 1)]
    instr = q.text + (f"\nRubric: {q.rubric}" if q.rubric else "")
    return {"type": "score", "instructions": instr, "criteria": levels}


def answer_from_wire(q: Question, a: dict[str, Any]) -> Answer:
    t = a.get("type")
    if q.kind == QKind.CHOICE:
        probs = a.get("probabilities") or ({a["choice"]: 1.0} if "choice" in a else {})
        return answer_from_distribution(q, probs, confidence=a.get("confidence"), raw=a)
    if q.kind == QKind.NOUL:
        p = float(a.get("noul", 0.5))
        return answer_from_distribution(q, {"yes": p, "no": 1.0 - p}, raw=a)
    lo, _ = q.scale  # type: ignore[misc]
    probs = a.get("probabilities") or {}
    dist = {str(lo + int(k)): float(v) for k, v in probs.items()}
    if not dist and "score" in a:
        dist = {str(lo + int(round(float(a["score"])))): 1.0}
    ans = answer_from_distribution(q, dist, confidence=a.get("confidence"), raw=a)
    if "score" in a:  # keep the server's expected level, shifted onto our scale
        ans.value = lo + float(a["score"])
    if t not in (None, "score"):
        ans.raw = {**(ans.raw or {}), "warning": f"unexpected answer type {t}"}
    return ans


class JevBackend(DecisionBackend):
    name = "jev"
    batched_questions = True
    max_options = JEV_MAX_OPTIONS

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 10.0, max_retries: int = 2, extra_body: dict[str, Any] | None = None,
                 price_model: str | None = None, max_questions_per_request: int = 64, name: str | None = None,
                 provider: str = "typesafe", transport: httpx.BaseTransport | None = None,
                 replicas: list[str] | None = None, max_options: int | None = None):
        """``provider`` picks defaults from PRESETS ("typesafe" | "openrouter" | "codiv"); explicit
        arguments and (for typesafe) TYPESAFE_* env vars override them. ``transport`` replaces the HTTP transport
        (tests: ``httpx.MockTransport`` sees every request byte; tests/test_wire_canary.py). ``replicas``: several
        servers of ONE model (e.g. two ``serve_jev`` processes on one GPU); every request of an episode goes to the same
        replica (:func:`replica_index` of the state's first block, the task), so its session cache keeps hitting.
        ``max_options``: the most options one read of the served model takes (E15: Verdict reads 24), so the harness's
        knockout answers wider Choices; unset keeps hosted Jev's 255 (and every existing cache key)."""
        if provider not in PRESETS:
            raise ValueError(f"unknown Jev provider {provider!r}; choose from {sorted(PRESETS)}")
        pre = PRESETS[provider]
        self.provider = provider
        env_model = os.environ.get("TYPESAFE_DEFAULT_MODEL") if provider == "typesafe" else None
        env_url = os.environ.get("TYPESAFE_BASE_URL") if provider == "typesafe" else None
        self.model = model or env_model or pre["model"]
        self.base_url = (base_url or env_url or pre["base_url"]).rstrip("/")
        # A key reserved for Jev calls (OPENROUTER_API_KEY_JEV, author's key since 2026-10-06) wins over the shared one;
        # System-2 LLM calls through OpenRouter never read it (llm/ uses OPENROUTER_API_KEY only).
        self.api_key = api_key if api_key is not None else (
            (os.environ.get(pre["jev_key_env"]) if pre.get("jev_key_env") else None) or os.environ.get(pre["key_env"]))
        price_model = price_model or (pre["price_model"] if not model else None)
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.price_model = price_model or self.model
        self.max_q = max_questions_per_request
        if name:
            self.name = name
        headers = {"Content-Type": "application/json", "X-TypeSafe-SDK": "jevharness/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)
        self.replicas = [u.rstrip("/") for u in replicas] if replicas else None
        self._replica_http = ([httpx.Client(base_url=u, headers=headers, timeout=timeout, transport=transport)
                               for u in self.replicas] if self.replicas else None)
        self.served_models: set[str] = set()
        self._max_options_set = max_options is not None
        if max_options is not None:
            self.max_options = int(max_options)

    def _client(self, state: Any) -> httpx.Client:
        if not self._replica_http:
            return self._http
        return self._replica_http[replica_index(state if isinstance(state, str) else str(state),
                                                len(self._replica_http))]

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, str]]:
        """(json, ms of the successful attempt, response headers). The headers carry two bookkeeping keys of the
        A6 ruling (e): ``x-jevharness-attempts`` (attempts incl. the successful one) and ``x-jevharness-retry-wait-ms``
        (backoff slept before it); latency excludes both, as before."""
        delay = 0.5
        waited = 0.0
        for attempt in range(self.max_retries + 1):
            t = time.perf_counter()
            try:
                r = self._client(body.get("state")).post("/v1/systemone", json=body,
                                    headers={"X-TypeSafe-Retry-Count": str(attempt)})
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == self.max_retries:
                    raise
                nap = min(5.0, delay) + random.uniform(0, 0.25)
                time.sleep(nap)
                waited += nap
                delay *= 2
                continue
            ms = (time.perf_counter() - t) * 1000
            if r.status_code == 200:
                return r.json(), ms, {**dict(r.headers), "x-jevharness-attempts": str(attempt + 1),
                                      "x-jevharness-retry-wait-ms": f"{waited * 1000:.3f}"}
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                ra = r.headers.get("retry-after-ms") or r.headers.get("retry-after")
                wait = float(ra) / (1000.0 if "retry-after-ms" in r.headers else 1.0) if ra else delay
                nap = min(5.0, wait) + random.uniform(0, 0.25)
                time.sleep(nap)
                waited += nap
                delay *= 2
                continue
            raise JevAPIError(r.status_code, r.text)
        raise RuntimeError("unreachable")

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        answers: list[Answer] = []
        usage = Usage(backend=self.name, model=self.model, n_calls=0, attempts=0, retry_wait_ms=0.0)
        reported_cost, n_reported = 0.0, 0
        # review P1-4: forward passes / rounds are reported only by our own wire servers (training/serve_jev.py,
        # decision/server.py); hosted Jev does not report them, so they stay None (unobservable) there
        passes: int | None = 0
        rounds: int | None = 0
        for i in range(0, len(questions), self.max_q):
            chunk = questions[i:i + self.max_q]
            keys = [f"q{j}" for j in range(len(chunk))]
            body = {"state": state, "model": self.model,
                    "questions": {k: question_to_wire(q) for k, q in zip(keys, chunk)}, **self.extra_body}
            data, ms, headers = self._post(body)
            ans = data.get("answers") or {}
            for k, q in zip(keys, chunk):
                if k not in ans:
                    raise JevAPIError(200, f"missing answer for {k}: {str(data)[:300]}")
                answers.append(answer_from_wire(q, ans[k]))
            u = data.get("usage") or {}
            served = data.get("model") or self.model
            self.served_models.add(served)
            usage.model = served
            usage.input_tokens += int(u.get("input_tokens", 0))
            usage.output_tokens += int(u.get("output_tokens", 0))
            if u.get("cost") is not None:  # OpenRouter reports the billed amount
                reported_cost += float(u["cost"])
                n_reported += 1
            passes = None if passes is None or u.get("n_passes") is None else passes + int(u["n_passes"])
            rounds = None if rounds is None or u.get("n_rounds") is None else rounds + int(u["n_rounds"])  # sequential
            usage.latency_ms += ms  # chunks are sequential
            usage.n_calls += 1
            usage.attempts = (usage.attempts or 0) + int(headers.get("x-jevharness-attempts", 1))
            usage.retry_wait_ms = (usage.retry_wait_ms or 0.0) + float(headers.get("x-jevharness-retry-wait-ms", 0.0))
            rid = headers.get("x-request-id") or headers.get("cf-ray") or data.get("id")
            if rid:
                usage.request_id = str(rid) if usage.request_id is None else f"{usage.request_id},{rid}"
            st = headers.get("server-timing")
            if st:
                answers[-1].raw = {**(answers[-1].raw or {}), "server_timing": st}
            # our own wire servers (training/serve_jev.py) report prefix-cache counters and their time; hosted Jev
            # does not, so its records are unchanged
            srv = {k: float(u[k]) for k in SERVER_USAGE_KEYS if isinstance(u.get(k), (int, float))
                   and not isinstance(u.get(k), bool)}
            sms = _server_timing_ms(st)
            if sms is not None and srv:
                srv["ms"] = sms
            if srv:
                usage.server = {k: (usage.server or {}).get(k, 0.0) + v for k, v in srv.items()}
        if usage.n_calls:
            usage.n_passes, usage.n_rounds = passes, rounds
        if n_reported == usage.n_calls and n_reported > 0:
            usage.cost_usd = reported_cost
            usage.priced = True
        else:  # Jev bills input tokens only
            usage.cost_usd = default_prices().cost(self.price_model, usage.input_tokens, 0)
            usage.priced = default_prices().get(self.price_model) is not None
        return DecisionResult(answers=answers, usage=usage)

    def list_models(self) -> Any:
        r = self._http.get("/v1/models")
        r.raise_for_status()
        return r.json()

    def describe(self) -> dict[str, Any]:
        d = {"backend": self.name, "provider": self.provider, "model": self.model, "base_url": self.base_url,
             "extra_body": self.extra_body}
        if self.replicas:
            d["replicas"] = list(self.replicas)
        if getattr(self, "_max_options_set", False):
            d["max_options"] = self.max_options
        return d


class OpenJevBackend(JevBackend):
    """Self-hosted OpenJev (razorback16/openjev, DiffusionGemma-26B-A4B) or openjev-server.

    Extensions accepted by razorback16/openjev (a request without them behaves like Jev):
    ``samples`` (1-32), ``steps`` (1-8), ``think`` (0-4096 tokens), ``images`` (<=8),
    ``sequential``. Pass them via ``extra_body``.
    """

    name = "openjev"

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 30.0, price_model: str | None = "openjev-selfhost", **kw: Any):
        super().__init__(model=model or os.environ.get("OPENJEV_MODEL", "openjev-latest"),
                         base_url=base_url or os.environ.get("OPENJEV_BASE_URL", "http://localhost:8080"),
                         api_key=api_key if api_key is not None else os.environ.get("OPENJEV_API_KEY", ""),
                         timeout=timeout, price_model=price_model, **kw)
        self.provider = "self-hosted"
