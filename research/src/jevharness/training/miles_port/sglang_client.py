"""Option-code log-probabilities from an SGLang ``/generate`` endpoint (the Miles rollout engine or its router).

One request per batch of prompts: ``max_new_tokens = 1``, temperature 1 with no top-k / top-p (so the returned
log-probabilities are the plain log-softmax of the logits at the decision position), ``token_ids_logprob`` = the
option-code ids of each prompt. The generated token itself is ignored: the decider samples from the restricted
softmax over the option codes. A log-softmax over the full vocabulary differs from the logits by one constant per row,
so the restricted softmax, the knockout ranking and every sampler see exactly what they would see from the logits.
Run the engine with ``SGLANG_RETURN_ORIGINAL_LOGPROB=1`` so the values are taken before any sampling transform.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Callable, Sequence

import httpx

NEG_INF = float("-inf")


def build_payload(input_ids: Sequence[Sequence[int]], code_ids: Sequence[Sequence[int]]) -> dict[str, Any]:
    if len(input_ids) != len(code_ids):
        raise ValueError("one code-id list per prompt")
    return {
        "input_ids": [list(map(int, x)) for x in input_ids],
        "sampling_params": {"max_new_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0},
        "return_logprob": True,
        "logprob_start_len": -1,
        "token_ids_logprob": [list(map(int, c)) for c in code_ids],
    }


def _as_list(resp: Any) -> list[dict[str, Any]]:
    if isinstance(resp, dict):
        return [resp]
    if isinstance(resp, list):
        return resp
    raise ValueError(f"unexpected /generate response type {type(resp).__name__}")


def parse_option_logprobs(resp: Any, code_ids: Sequence[Sequence[int]]) -> list[list[float]]:
    """``meta_info.output_token_ids_logprobs[0]`` = [(logprob, token_id, text?), ...] for the requested ids at the first
    generated position -> log-probabilities in the order of ``code_ids`` (missing ids -> -inf)."""
    items = _as_list(resp)
    if len(items) != len(code_ids):
        raise ValueError(f"{len(items)} responses for {len(code_ids)} prompts")
    out: list[list[float]] = []
    for item, codes in zip(items, code_ids):
        meta = item.get("meta_info") or {}
        pos = meta.get("output_token_ids_logprobs")
        if not pos:
            raise ValueError(f"no output_token_ids_logprobs in the response: keys {sorted(meta)}")
        first = pos[0]
        got: dict[int, float] = {}
        for entry in first or []:
            if isinstance(entry, dict):
                lp, tid = entry.get("logprob"), entry.get("token_id")
            else:
                lp, tid = entry[0], entry[1]
            got[int(tid)] = NEG_INF if lp is None else float(lp)
        vals = [got.get(int(c), NEG_INF) for c in codes]
        if all(math.isinf(v) for v in vals):
            raise ValueError("every option code has -inf log-probability (wrong token ids?)")
        out.append(vals)
    return out


def build_single_payload(input_ids: Sequence[int], code_ids: Sequence[int]) -> dict[str, Any]:
    """One prompt per request: Miles' SGLang router (sgl-model-gateway) rejects batched ``token_ids_logprob`` (it
    expects a flat id list), so requests through the router carry one prompt each."""
    p = build_payload([input_ids], [code_ids])
    p["input_ids"], p["token_ids_logprob"] = p["input_ids"][0], p["token_ids_logprob"][0]
    return p


class SGLangClient:
    """Thread-safe client. ``post`` is injectable (tests serve a local model through it). ``batched=False`` (the
    default; required behind the Miles router) sends one request per prompt, concurrently; ``batched=True`` sends one
    batched request (a bare SGLang server accepts it)."""

    def __init__(self, url: str, timeout: float = 300.0, retries: int = 3,
                 post: Callable[[dict[str, Any]], Any] | None = None, batched: bool = False,
                 concurrency: int = 64):
        self.url = url.rstrip("/")
        if not self.url.endswith("/generate"):
            self.url += "/generate"
        self.timeout = timeout
        self.retries = retries
        self.batched = batched
        self.concurrency = concurrency
        self._post = post
        self._client = httpx.Client(timeout=timeout, limits=httpx.Limits(max_connections=4 * concurrency)) \
            if post is None else None
        self._pool: Any = None
        self._lock = threading.Lock()
        self.requests = 0
        self.prompts = 0

    def _send(self, payload: dict[str, Any]) -> Any:
        if self._post is not None:
            return self._post(payload)
        assert self._client is not None
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self._client.post(self.url, json=payload)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, json.JSONDecodeError) as e:  # engine restart / weight sync window
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"SGLang /generate failed after {self.retries} attempts: {last}")

    def option_logprobs(self, input_ids: Sequence[Sequence[int]], code_ids: Sequence[Sequence[int]]
                        ) -> list[list[float]]:
        if not input_ids:
            return []
        if self.batched:
            resp = self._send(build_payload(input_ids, code_ids))
            self.requests += 1
            self.prompts += len(input_ids)
            return parse_option_logprobs(resp, code_ids)
        if len(input_ids) != len(code_ids):
            raise ValueError("one code-id list per prompt")
        with self._lock:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor

                self._pool = ThreadPoolExecutor(max_workers=self.concurrency)
        resps = list(self._pool.map(lambda ic: self._send(build_single_payload(*ic)), zip(input_ids, code_ids)))
        self.requests += len(resps)
        self.prompts += len(resps)
        return parse_option_logprobs(resps, code_ids)
