"""Embedding-similarity baseline (dense tool retrieval style) for CHOICE questions.

Score(option) = cos(emb(state + question), emb(option + description)); softmax with a
temperature gives a distribution. Cheap and fast but cannot do NOUL/SCORE questions.

Latency counts the successful embedding requests only: rate-limit retries and their backoff sleeps are
quota artefacts and are excluded, as in the chat clients (``llm/openai_compat.py``). Keys rotate over
``<PROVIDER>_API_KEYS`` like the chat clients. Option embeddings are memoised in-process (a deployed
retriever pre-computes its tool index), so repeated options cost neither tokens nor latency.
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from typing import Any

import numpy as np

from ..accounting import default_prices, estimate_tokens
from .base import DecisionBackend
from .types import DecisionResult, QKind, Question, Usage, answer_from_distribution

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_BATCH = 100  # inputs per embeddings request (Gemini's batch limit)


class EmbeddingBackend(DecisionBackend):
    name = "embedding"

    def __init__(self, model: str = "text-embedding-3-small", provider: str = "openai",
                 provider_base_url: str | None = None, api_key: str | None = None, temperature: float = 0.05,
                 max_retries: int = 6, timeout: float = 60.0):
        """``provider`` reuses the OpenAI-compatible presets (e.g. "gemini" with
        model="gemini-embedding-001"); keys come from the environment, never from configs."""
        from openai import OpenAI

        from ..llm.openai_compat import PROVIDERS, provider_keys

        default_url, key_env = PROVIDERS.get(provider, (None, "OPENAI_API_KEY"))
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        # max_retries=0: the SDK's own retries would sleep inside the timed request
        self.clients = [OpenAI(base_url=provider_base_url or default_url, api_key=k, timeout=timeout, max_retries=0)
                        for k in provider_keys(key_env, api_key)]
        self._rr = itertools.count()
        self._lock = threading.Lock()
        self._cache: dict[str, np.ndarray] = {}
        self.n_retries = 0

    @property
    def client(self) -> Any:
        return self.clients[0]

    @client.setter
    def client(self, c: Any) -> None:
        """Replacing ``client`` (tests, custom transports) replaces the whole key rotation."""
        self.clients = [c]

    def _request(self, texts: list[str]) -> tuple[Any, float]:
        """One embeddings request with retries; returns (response, ms of the successful attempt)."""
        import openai

        delay, last = 1.0, None
        for attempt in range(self.max_retries + 1):
            with self._lock:
                client = self.clients[next(self._rr) % len(self.clients)]
            t = time.perf_counter()
            try:
                r = client.embeddings.create(model=self.model, input=texts)
                return r, (time.perf_counter() - t) * 1000
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last = e
            except openai.APIStatusError as e:
                if e.status_code not in RETRY_STATUS:
                    raise
                last = e
                ra = e.response.headers.get("retry-after") if e.response is not None else None
                try:
                    delay = max(delay, float(ra)) if ra else delay
                except ValueError:
                    pass
            if attempt == self.max_retries:
                break
            self.n_retries += 1
            time.sleep(min(30.0, delay) + random.uniform(0, 0.5))
            delay *= 2
        assert last is not None
        raise last

    def _embed(self, texts: list[str]) -> tuple[list[np.ndarray], int, float, int]:
        """Vectors for ``texts``, tokens billed, API milliseconds (successful attempts only) and the number
        of requests sent (0 when every text is memoised)."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        n_tok, ms = 0, 0.0
        for i in range(0, len(missing), MAX_BATCH):
            chunk = missing[i:i + MAX_BATCH]
            r, dt = self._request(chunk)
            ms += dt
            for t, d in zip(chunk, r.data):
                v = np.asarray(d.embedding, dtype=np.float32)
                self._cache[t] = v / (np.linalg.norm(v) + 1e-9)
            n_tok += int(getattr(getattr(r, "usage", None), "total_tokens", 0) or 0) or sum(estimate_tokens(t) for t in chunk)
        return [self._cache[t] for t in texts], n_tok, ms, -(-len(missing) // MAX_BATCH)

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        # every text of the request (each question's query + its options) goes out in ONE embeddings request,
        # as the batched System-1 deciders answer all questions about a state in one call
        groups: list[list[str]] = []
        for q in questions:
            if q.kind != QKind.CHOICE:
                raise NotImplementedError("EmbeddingBackend only supports CHOICE questions")
            assert q.options is not None
            opts = [o + (f": {q.option_descriptions[i]}" if q.option_descriptions and q.option_descriptions[i] else "")
                    for i, o in enumerate(q.options)]
            groups.append([f"{state}\n{q.text}"] + opts)
        vecs, tokens, api_ms, n_req = self._embed([t for g in groups for t in g])
        t = time.perf_counter()
        answers, k = [], 0
        for q, g in zip(questions, groups):
            qv, ovs = vecs[k], vecs[k + 1:k + len(g)]
            k += len(g)
            sims = np.array([float(qv @ v) for v in ovs])
            logits = sims / self.temperature
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            answers.append(answer_from_distribution(q, dict(zip(q.options or [], probs.tolist()))))
        local = (time.perf_counter() - t) * 1000
        return DecisionResult(answers=answers, usage=Usage(
            backend=self.name, model=self.model, input_tokens=tokens, latency_ms=api_ms + local,
            cost_usd=default_prices().cost(self.model, tokens, 0), n_calls=n_req,
            priced=default_prices().get(self.model) is not None))

    def describe(self) -> dict[str, Any]:
        # "batch": one request per decide() (cache entries of the per-question version replayed its latency)
        return {"backend": self.name, "model": self.model, "temperature": self.temperature, "batch": True}
