"""OpenAI **Responses API** client (``POST /v1/responses``), provider name ``openai_responses``.

Why a second OpenAI client: ``openai_compat.py`` speaks Chat Completions, which OpenAI keeps for compatibility
but which no longer carries the current models' full feature set (the GPT-5.x/6 docs: "Chat Completions
supports function calling only with ``reasoning_effort: none``; use Responses for reasoning with tools").
This client speaks the Responses wire format directly over ``httpx`` (no SDK dependency, so a fake transport
can test every request byte offline) and plugs into the same :class:`LLMClient` interface, so it works as

* **System-2** (the agent LLM with native function calling), ``llm: {provider: openai_responses, model: ...}``;
* **System-1** through the existing deciders: ``llm_generative`` / ``s2_self`` (verbalised distributions, any
  model) and ``llm_logprob`` (option-token logprobs, only where the API allows them -- see below).

Responses vs Chat Completions, the differences that matter here (docs/HARNESS_INTEGRATION.md has the long form):

* Input is a list of *items*, not messages: the leading system prompt becomes ``instructions``; a tool call is a
  ``function_call`` item (``call_id``/``name``/``arguments``) and its result a ``function_call_output`` item.
* Reasoning models emit opaque ``reasoning`` items. With ``store: false`` (our default: nothing is retained on
  OpenAI's side) they carry ``encrypted_content`` (``include: ["reasoning.encrypted_content"]``) and the docs
  ask for them to be echoed back untouched between a function call and its output. They are kept in
  ``ToolCall.extra["openai"]["reasoning"]`` (the harness already round-trips ``extra`` as ``extra_content``,
  exactly as it does for Gemini thought signatures) and re-emitted in front of the function call.
* ``reasoning: {effort}`` replaces ``reasoning_effort``. When the effort is not ``none``, the API rejects
  ``temperature``, ``top_p`` and ``top_logprobs`` (and the logprobs ``include``): the client drops the sampling
  parameters (counted in ``stats["dropped_params"]``) and raises :class:`LogprobsUnavailableError` for logprobs.
* Logprobs: ``top_logprobs: n`` + ``include: ["message.output_text.logprobs"]``; returned per output position
  inside the message content part. Available only with reasoning effort ``none`` (GPT-5.x / GPT-6 models that
  support ``none``) or on non-reasoning models (gpt-4.1 / gpt-4o families). ``max_output_tokens`` has a documented
  minimum of 16, so the logprob decider's ``max_tokens=1`` is clamped (the first position is still what is read).
* Usage: ``input_tokens`` (+ ``input_tokens_details.cached_tokens`` / ``cache_write_tokens``), ``output_tokens``
  (reasoning tokens included, detailed in ``output_tokens_details.reasoning_tokens``).
* Server-side state: ``previous_response_id`` lets a turn send only the new items. **Off by default**
  (``server_state: false``): the docs state that all previous input tokens of the chain are billed again on every
  turn, so what it saves is wire bytes and client bookkeeping, not tokens. When enabled, ``store`` is forced on
  and the client records, per call, the estimated tokens actually put on the wire next to the billed input
  tokens (``stats`` / ledger meta ``wire_input_tokens_est`` vs ``input_tokens``), so the "incremental state"
  argument of docs/DECISION_MODEL_LANDSCAPE.md §3 is measured, not assumed. A history that is not a pure
  extension of the last request (a System-1 *replace* of the model's call, a stop-verifier note, the early-stop
  branch) breaks the chain and falls back to a full stateless send (``stats["chain_breaks"]``).

Config keys (all optional): ``api_key`` (else ``OPENAI_API_KEYS`` round-robin or ``OPENAI_API_KEY``),
``base_url`` (else ``OPENAI_BASE_URL``), ``timeout``, ``max_retries``, ``store`` (default false),
``server_state`` (default false), ``strict_tools`` (default false), ``echo_reasoning`` (default true),
``allow_missing_logprobs`` (default false), ``reasoning_summary`` (``auto|concise|detailed``), plus per-call
defaults such as ``reasoning_effort``, ``temperature``, ``max_tokens``, ``service_tier``, ``prompt_cache_key``.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import random
import threading
import time
from typing import Any

import httpx

from ..accounting import CallRecord, Ledger, default_prices, estimate_tokens
from .base import ChatResponse, LLMClient, ToolCall
from .openai_compat import RETRY_STATUS, provider_keys

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
NO_KEY_MESSAGE = ("OPENAI_API_KEY is not set: add a line `OPENAI_API_KEY=sk-...` to .env (gitignored; never commit "
                  "or print the value) or pass `api_key` in the llm block. Live check once it is there: "
                  "bash scripts/smoke_openai.sh")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
# rejected by the API when reasoning effort is not "none" (GPT-5.x / GPT-6 "latest model" guide)
SAMPLING_PARAMS = ("temperature", "top_p", "top_logprobs")
LOGPROBS_INCLUDE = "message.output_text.logprobs"
REASONING_INCLUDE = "reasoning.encrypted_content"
MIN_OUTPUT_TOKENS = 16          # documented minimum of max_output_tokens
# request fields passed through unchanged when given as defaults or per call
PASSTHROUGH = ("parallel_tool_calls", "service_tier", "prompt_cache_key", "prompt_cache_options", "metadata",
               "truncation", "max_tool_calls", "safety_identifier", "text", "background")
# Chat-Completions-only knobs the Responses API has no equivalent for; dropped and counted, never sent
IGNORED = ("n", "seed", "stop", "logit_bias", "presence_penalty", "frequency_penalty")
_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4", "codex")


class OpenAIResponsesError(RuntimeError):
    """Non-retryable API error (status + message from the body)."""

    def __init__(self, status: int, message: str, body: Any = None):
        super().__init__(f"OpenAI Responses API error {status}: {message}")
        self.status = status
        self.message = message
        self.body = body


class LogprobsUnavailableError(RuntimeError):
    """Raised when logprobs were requested but this model / reasoning setting cannot return them."""


def is_reasoning_family(model: str) -> bool:
    m = (model or "").lower().split("/")[-1]
    return m.startswith(_REASONING_PREFIXES)


# ---------------------------------------------------------------------------------------------- request building
def _user_content(content: Any) -> Any:
    """Chat-style content (string or parts) -> Responses input content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if not isinstance(p, dict):
            parts.append({"type": "input_text", "text": str(p)})
        elif p.get("type") == "text":
            parts.append({"type": "input_text", "text": p.get("text", "")})
        elif p.get("type") == "image_url":
            url = p["image_url"]["url"] if isinstance(p.get("image_url"), dict) else p.get("image_url")
            parts.append({"type": "input_image", "image_url": url})
        else:
            parts.append(p)
    return parts


def _reasoning_items(extra: dict[str, Any] | None) -> list[dict[str, Any]]:
    items = ((extra or {}).get("openai") or {}).get("reasoning") or []
    return [it for it in items if isinstance(it, dict)]


def messages_to_input(messages: list[dict[str, Any]],
                      extract_instructions: bool = True) -> tuple[str | None, list[dict[str, Any]]]:
    """Our OpenAI-style message list -> (``instructions``, Responses ``input`` items).

    Leading system/developer messages become ``instructions`` (unless ``extract_instructions`` is off, for the
    delta of a chained call); later ones stay in place as items. Assistant tool calls become ``function_call``
    items, preceded by the reasoning items remembered in ``extra_content`` (echoed verbatim, deduplicated by
    id); ``tool`` messages become ``function_call_output`` items."""
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer") and not items and extract_instructions:
            instructions.append(str(m.get("content") or ""))
            continue
        if role in ("system", "developer"):
            items.append({"role": role, "content": str(m.get("content") or "")})
        elif role == "user":
            items.append({"role": "user", "content": _user_content(m.get("content"))})
        elif role == "tool":
            content = m.get("content")
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"],
                          "output": content if isinstance(content, str) else json.dumps(content)})
        elif role == "assistant":
            seen: set[str] = set()
            reasoning = _reasoning_items(m.get("extra_content"))
            for tc in m.get("tool_calls") or []:
                reasoning += _reasoning_items(tc.get("extra_content"))
            for it in reasoning:
                key = it.get("id") or json.dumps(it, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    items.append({"type": "reasoning", **it})
            if m.get("content"):
                items.append({"role": "assistant", "content": str(m["content"])})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                item = {"type": "function_call", "call_id": tc["id"], "name": fn.get("name", tc.get("name")),
                        "arguments": fn.get("arguments") or "{}"}
                item_id = ((tc.get("extra_content") or {}).get("openai") or {}).get("item_id")
                if item_id:
                    item["id"] = item_id
                items.append(item)
        else:
            raise ValueError(f"unknown message role {role!r}")
    return ("\n\n".join(s for s in instructions if s) or None), items


def tools_to_responses(tools: list[dict[str, Any]] | None, strict: bool | None = None) -> list[dict[str, Any]]:
    out = []
    for t in tools or []:
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            f = t["function"]
            item: dict[str, Any] = {"type": "function", "name": f["name"], "description": f.get("description", ""),
                                    "parameters": f.get("parameters") or {"type": "object", "properties": {}}}
            if strict is not None:
                item["strict"] = bool(strict)
            out.append(item)
        else:  # already a Responses tool (function with top-level name, or a built-in tool)
            out.append(t)
    return out


def tool_choice_to_responses(choice: Any) -> Any:
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        return {"type": "function", "name": choice["function"]["name"]}
    return choice


def _text_format(response_format: Any) -> dict[str, Any] | None:
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    if kind == "json_schema":
        js = response_format.get("json_schema") or {}
        return {"type": "json_schema", "name": js.get("name", "response"), "schema": js.get("schema", {}),
                "strict": bool(js.get("strict", False))}
    if kind in ("json_object", "text"):
        return {"type": kind}
    return None


def _canon(m: dict[str, Any]) -> str:
    return json.dumps(m, sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------------- the client
class OpenAIResponsesClient(LLMClient):
    provider = "openai_responses"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0,
                 max_retries: int = 6, store: bool | None = None, server_state: bool = False,
                 strict_tools: bool | None = None, echo_reasoning: bool = True, allow_missing_logprobs: bool = False,
                 reasoning_summary: str | None = None, transport: httpx.BaseTransport | None = None,
                 **defaults: Any):
        super().__init__(model, **defaults)
        if server_state and store is False:
            raise ValueError("server_state=true needs stored responses (previous_response_id); do not set store=false")
        self.store = bool(store) if store is not None else bool(server_state)
        self.server_state = bool(server_state)
        self.strict_tools = strict_tools
        self.echo_reasoning = echo_reasoning
        self.allow_missing_logprobs = allow_missing_logprobs
        self.reasoning_summary = reasoning_summary
        self.max_retries = max_retries
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.keys = provider_keys("OPENAI_API_KEY", api_key)
        self._rr = itertools.count()
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport,
                                  headers={"Content-Type": "application/json"})
        self._lock = threading.Lock()
        self._chains: dict[int, dict[str, Any]] = {}      # thread id -> last response of that thread's episode
        self._sampling_rejected = False                   # learned from a 400: this model takes no temperature/top_p
        self._reasoning_include_rejected = False
        self.n_retries = 0
        self.stats: dict[str, Any] = {
            "calls": 0, "chained_calls": 0, "full_sends": 0, "chain_breaks": 0, "new_conversations": 0,
            "wire_items": 0, "wire_chars": 0,
            "wire_input_tokens_est": 0, "input_tokens": 0, "cached_input_tokens": 0, "cache_write_tokens": 0,
            "output_tokens": 0, "reasoning_tokens": 0, "dropped_params": {}, "ignored_params": {}}

    # ------------------------------------------------------------------ bookkeeping
    def _count(self, bucket: str, key: str) -> None:
        with self._lock:
            d = self.stats[bucket]
            d[key] = d.get(key, 0) + 1

    def describe(self) -> dict[str, Any]:
        d = {"provider": self.provider, "model": self.model, "api": "responses", "store": self.store,
             "server_state": self.server_state}
        for k in ("reasoning_effort", "temperature", "service_tier"):
            if k in self.defaults:
                d[k] = self.defaults[k]
        return d

    # ------------------------------------------------------------------ request
    def _build(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
               kw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns (request body, flags). ``flags``: ``logprobs`` requested, ``chained`` (previous_response_id)."""
        kw = dict(kw)
        flags: dict[str, Any] = {"logprobs": False, "chained": False, "sampling": []}
        params: dict[str, Any] = {"model": self.model, "store": self.store}
        include: list[str] = list(kw.pop("include", None) or [])

        effort = kw.pop("reasoning_effort", None)
        reasoning = dict(kw.pop("reasoning", None) or {})
        if effort is not None:
            effort = str(effort).lower()
            if effort not in REASONING_EFFORTS:
                raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}, got {effort!r}")
            reasoning["effort"] = effort
        if self.reasoning_summary and "summary" not in reasoning:
            reasoning["summary"] = self.reasoning_summary
        if reasoning:
            params["reasoning"] = reasoning
        effort = reasoning.get("effort")
        reasoning_on = effort not in (None, "none") if effort is not None else is_reasoning_family(self.model)
        sampling_ok = effort == "none" or (effort is None and not is_reasoning_family(self.model))
        if self._sampling_rejected:
            sampling_ok = False

        top_logprobs = kw.pop("top_logprobs", None)
        if top_logprobs:
            if not sampling_ok:
                raise LogprobsUnavailableError(
                    f"{self.model}: logprobs need reasoning effort 'none' (or a non-reasoning model such as "
                    f"gpt-4.1-*); current effort is {effort!r}. Set reasoning_effort: none in the llm block.")
            params["top_logprobs"] = max(0, min(20, int(top_logprobs)))
            include.append(LOGPROBS_INCLUDE)
            flags["logprobs"] = True
        for k in ("temperature", "top_p"):
            v = kw.pop(k, None)
            if v is None:
                continue
            if sampling_ok:
                params[k] = v
                flags["sampling"].append(k)
            else:
                self._count("dropped_params", k)
        max_tokens = kw.pop("max_output_tokens", None) or kw.pop("max_tokens", None)
        if max_tokens is not None:
            params["max_output_tokens"] = max(MIN_OUTPUT_TOKENS, int(max_tokens))
        fmt = _text_format(kw.pop("response_format", None))
        if fmt:
            params["text"] = {**(kw.get("text") or {}), "format": fmt}
            kw.pop("text", None)
        if self.echo_reasoning and reasoning_on and not self._reasoning_include_rejected:
            include.append(REASONING_INCLUDE)
        if include:
            params["include"] = sorted(set(include))

        if tools:
            params["tools"] = tools_to_responses(tools, self.strict_tools)
            if kw.get("tool_choice") is not None:
                params["tool_choice"] = tool_choice_to_responses(kw.pop("tool_choice"))
        kw.pop("tool_choice", None)
        for k in PASSTHROUGH:
            if kw.get(k) is not None:
                params[k] = kw.pop(k)
        for k in IGNORED:
            if kw.pop(k, None) is not None:
                self._count("ignored_params", k)
        extra_body = kw.pop("extra_body", None)
        if isinstance(extra_body, dict):
            params.update(extra_body)

        instructions, items = messages_to_input(messages)
        if instructions:
            params["instructions"] = instructions   # not carried over by previous_response_id: always sent
        chain = self._chains.get(threading.get_ident()) if self.server_state else None
        if chain is not None:
            sent = chain["sent"]
            cur = [_canon(m) for m in messages]
            new_msgs = messages[len(sent):] if len(cur) > len(sent) and cur[:len(sent)] == sent else None
            # the first new message must be the echo of the chained response itself (its calls, or its text):
            # anything else (a replaced call, a dropped call, a branch) means the histories diverged
            echo_ok = False
            if new_msgs:
                first = new_msgs[0]
                if first.get("role") == "assistant":
                    ids = {tc["id"] for tc in first.get("tool_calls") or []}
                    echo_ok = (ids == chain["call_ids"]) if ids else ((first.get("content") or "") == chain["text"])
            if new_msgs and echo_ok:
                _, items = messages_to_input(new_msgs[1:], extract_instructions=False)
                params["previous_response_id"] = chain["id"]
                flags["chained"] = True
            else:
                # same system prompt + first user turn = the same episode diverged (a real break); otherwise the
                # worker thread simply started a new conversation
                same_episode = len(sent) >= 2 and cur[:2] == sent[:2]
                with self._lock:
                    self.stats["chain_breaks" if same_episode else "new_conversations"] += 1
                    self._chains.pop(threading.get_ident(), None)
        params["input"] = items
        return params, flags

    # ------------------------------------------------------------------ transport with retries
    def _post(self, params: dict[str, Any], timing: dict[str, float] | None = None) -> dict[str, Any]:
        """``timing["ms"]`` is the latency of the successful attempt only; 429/5xx backoff is a quota artefact."""
        delay = 1.0
        last: Exception | None = None
        waited = 0.0
        for attempt in range(self.max_retries + 1):
            key = self.keys[next(self._rr) % len(self.keys)]
            if key == "EMPTY":
                raise RuntimeError(NO_KEY_MESSAGE)
            t = time.perf_counter()
            try:
                r = self._http.post("/responses", json=params, headers={"Authorization": f"Bearer {key}"})
            except httpx.TransportError as e:
                last = e
            else:
                if r.status_code < 300:
                    if timing is not None:
                        timing["ms"] = (time.perf_counter() - t) * 1000
                        timing["attempts"] = attempt + 1
                        timing["retry_wait_ms"] = waited * 1000
                    return r.json()
                msg = _error_message(r)
                if r.status_code == 401:
                    raise OpenAIResponsesError(401, f"authentication failed ({msg}). {NO_KEY_MESSAGE}", r.text)
                if r.status_code not in RETRY_STATUS:
                    raise OpenAIResponsesError(r.status_code, msg, r.text)
                last = OpenAIResponsesError(r.status_code, msg, r.text)
                ra = r.headers.get("retry-after")
                if ra:
                    try:
                        delay = max(delay, float(ra))
                    except ValueError:
                        pass
            if attempt == self.max_retries:
                break
            self.n_retries += 1
            nap = min(30.0, delay) + random.uniform(0, 0.5)
            time.sleep(nap)
            waited += nap
            delay *= 2
        assert last is not None
        raise last

    def _post_adaptive(self, params: dict[str, Any], flags: dict[str, Any],
                       timing: dict[str, float]) -> dict[str, Any]:
        """One documented 400 is handled in place: sampling parameters on a reasoning model are dropped (and
        remembered), a rejected reasoning include is dropped; a lost previous_response_id falls back to a full send
        (the caller re-builds)."""
        try:
            return self._post(params, timing)
        except OpenAIResponsesError as e:
            text = (e.message or "").lower()
            if e.status == 400 and any(p in text for p in SAMPLING_PARAMS) and \
                    any(p in params for p in SAMPLING_PARAMS):
                if flags.get("logprobs"):
                    raise LogprobsUnavailableError(
                        f"{self.model} rejected logprobs ({e.message}). Logprobs need reasoning effort 'none' "
                        "or a non-reasoning model (gpt-4.1-*, gpt-4o-*).") from e
                for p in SAMPLING_PARAMS:
                    if params.pop(p, None) is not None:
                        self._count("dropped_params", p)
                self._sampling_rejected = True
                return self._post(params, timing)
            if e.status == 400 and REASONING_INCLUDE in text and REASONING_INCLUDE in (params.get("include") or []):
                params["include"] = [i for i in params["include"] if i != REASONING_INCLUDE] or None
                if params["include"] is None:
                    params.pop("include")
                self._reasoning_include_rejected = True
                return self._post(params, timing)
            raise

    # ------------------------------------------------------------------ the call
    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        params, flags = self._build(messages, tools, kw)
        timing: dict[str, float] = {}
        t0 = time.perf_counter()
        try:
            r = self._post_adaptive(params, flags, timing)
        except OpenAIResponsesError as e:
            if flags["chained"] and e.status in (400, 404) and "previous" in (e.message or "").lower():
                with self._lock:            # the chained response is gone (expired / not stored): resend everything
                    self.stats["chain_breaks"] += 1
                    self._chains.pop(threading.get_ident(), None)
                params, flags = self._build(messages, tools, kw)
                r = self._post_adaptive(params, flags, timing)
            else:
                raise
        latency = timing.get("ms", (time.perf_counter() - t0) * 1000)
        resp = self._parse(r, flags)
        resp.latency_ms = latency
        resp.attempts = int(timing.get("attempts", 1))
        resp.retry_wait_ms = float(timing.get("retry_wait_ms", 0.0))
        resp.request_id = r.get("id")

        wire_text = json.dumps(params.get("input", []), default=str) + (params.get("instructions") or "")
        wire = {"wire_items": len(params.get("input", [])), "wire_chars": len(wire_text),
                "wire_input_tokens_est": estimate_tokens(wire_text) if wire_text else 0,
                "chained": flags["chained"], "response_id": r.get("id"),
                "cache_write_tokens": resp.raw["cache_write_tokens"], "attempts": int(timing.get("attempts", 1))}
        resp.raw = {"response": r, "wire": wire}
        with self._lock:
            s = self.stats
            s["calls"] += 1
            s["chained_calls" if flags["chained"] else "full_sends"] += 1
            for k in ("wire_items", "wire_chars", "wire_input_tokens_est", "cache_write_tokens"):
                s[k] += wire[k]
            s["input_tokens"] += resp.input_tokens
            s["cached_input_tokens"] += resp.cached_input_tokens
            s["output_tokens"] += resp.output_tokens
            s["reasoning_tokens"] += resp.reasoning_tokens
            if self.server_state and r.get("id"):
                self._chains[threading.get_ident()] = {
                    "id": r["id"], "sent": [_canon(m) for m in messages],
                    "call_ids": {tc.id for tc in resp.tool_calls}, "text": resp.text}
        return resp

    def _parse(self, r: dict[str, Any], flags: dict[str, Any]) -> ChatResponse:
        if r.get("status") == "failed" or r.get("error"):
            err = r.get("error") or {}
            raise OpenAIResponsesError(0, f"response failed: {err.get('message') or err}", r)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        reasoning_buf: list[dict[str, Any]] = []
        tlp: list[dict[str, float]] = []
        refusal = False
        for item in r.get("output") or []:
            kind = item.get("type")
            if kind == "reasoning":
                reasoning_buf.append({k: item[k] for k in ("id", "summary", "encrypted_content", "content")
                                      if k in item})
            elif kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        text_parts.append(part.get("text") or "")
                        for pos in part.get("logprobs") or []:
                            top = {alt["token"]: float(alt["logprob"]) for alt in pos.get("top_logprobs") or []}
                            if "token" in pos and pos["token"] not in top:
                                top[pos["token"]] = float(pos.get("logprob", 0.0))
                            tlp.append(top)
                    elif part.get("type") == "refusal":
                        text_parts.append(part.get("refusal") or "")
                        refusal = True
            elif kind == "function_call":
                raw_args = item.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"__raw__": raw_args}
                extra: dict[str, Any] = {"openai": {"item_id": item.get("id"), "reasoning": reasoning_buf}}
                reasoning_buf = []
                calls.append(ToolCall(id=item.get("call_id") or f"call_{random.getrandbits(32):08x}",
                                      name=item.get("name", ""),
                                      arguments=args if isinstance(args, dict) else {"__raw__": args}, extra=extra))
        u = r.get("usage") or {}
        in_details = u.get("input_tokens_details") or {}
        out_details = u.get("output_tokens_details") or {}
        text = "".join(text_parts)
        input_tokens = int(u.get("input_tokens") or 0)
        output_tokens = int(u.get("output_tokens") or 0)
        if not u:  # should not happen; keep the accounting non-zero rather than silently free
            input_tokens = estimate_tokens(json.dumps(r.get("input", "")))
            output_tokens = estimate_tokens(text)
        if flags.get("logprobs") and not tlp and not self.allow_missing_logprobs:
            raise LogprobsUnavailableError(
                f"{self.model} returned no logprobs although top_logprobs was requested (reasoning effort other "
                "than 'none', or a model without logprob support). Use reasoning_effort: none on a model that "
                "supports it, a gpt-4.1-* model, or the verbalised decider (llm_generative).")
        status = r.get("status")
        if status == "incomplete":
            reason = (r.get("incomplete_details") or {}).get("reason") or "incomplete"
            finish = "length" if reason == "max_output_tokens" else reason
        elif refusal:
            finish = "refusal"
        else:
            finish = "tool_calls" if calls else "stop"
        returned = r.get("model") or self.model
        model = self.model if returned.startswith(self.model) else returned   # snapshot id -> configured alias
        resp = ChatResponse(text=text, tool_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens,
                            cached_input_tokens=int(in_details.get("cached_tokens") or 0),
                            reasoning_tokens=int(out_details.get("reasoning_tokens") or 0),
                            model=model, top_logprobs=tlp or None, finish_reason=finish,
                            raw={"cache_write_tokens": int(in_details.get("cache_write_tokens") or 0)})
        if reasoning_buf:   # reasoning of a text-only reply: echoed through the assistant message's extra_content
            resp.extra = {"openai": {"reasoning": reasoning_buf}}
        return resp

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             ledger: Ledger | None = None, decision_point: str | None = None,
             component: str = "s2", **kw: Any) -> ChatResponse:
        """As :meth:`LLMClient.chat`, with the wire/billing split in the ledger meta."""
        params = {**self.defaults, **kw}
        resp = self._chat(messages, tools=tools, **params)
        if ledger is not None:
            price = default_prices().get(resp.model or self.model)
            cost = price.cost(resp.input_tokens, resp.output_tokens, resp.cached_input_tokens) if price else 0.0
            wire = (resp.raw or {}).get("wire", {}) if isinstance(resp.raw, dict) else {}
            # the model id and service tier exactly as the API returned them (the ledger's `model` maps a dated
            # snapshot back to the configured alias), so the served model stays identifiable (A7.12 (b), (i))
            served = ((resp.raw or {}).get("response") or {}) if isinstance(resp.raw, dict) else {}
            ledger.add(CallRecord(component=component, backend=self.provider, model=resp.model or self.model,
                                  decision_point=decision_point, input_tokens=resp.input_tokens,
                                  output_tokens=resp.output_tokens, cost_usd=cost, latency_ms=resp.latency_ms,
                                  priced=price is not None,
                                  meta={"cached_input_tokens": resp.cached_input_tokens,
                                        "reasoning_tokens": resp.reasoning_tokens, "n_tools": len(tools or []),
                                        **{k: wire[k] for k in ("wire_input_tokens_est", "wire_items", "chained",
                                                                "cache_write_tokens", "response_id") if k in wire},
                                        **({"served_model": served["model"]} if served.get("model") else {}),
                                        **({"service_tier": served["service_tier"]} if served.get("service_tier")
                                           else {})},
                                  attempts=resp.attempts, retry_wait_ms=resp.retry_wait_ms,
                                  request_id=resp.request_id))
        return resp

    # ------------------------------------------------------------------ logprob probe
    def option_logprobs(self, messages: list[dict[str, Any]], options: list[str], top_logprobs: int = 20,
                        **kw: Any) -> dict[str, float]:
        """Log-probability of each option label at the FIRST output position (``-inf`` if not among the top-k).
        Raises :class:`LogprobsUnavailableError` for models / reasoning settings that cannot return logprobs."""
        resp = self.chat(messages, max_tokens=1, top_logprobs=top_logprobs, **kw)
        top = resp.top_logprobs[0] if resp.top_logprobs else {}
        out = {o: -math.inf for o in options}
        for tok, lp in top.items():
            key = tok.strip().strip(".:)").strip()
            if key in out:
                out[key] = float(max(out[key], lp)) if out[key] != -math.inf else float(lp)
        return out


def _error_message(r: httpx.Response) -> str:
    try:
        body = r.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            return str(err.get("message") or err)
        return str(err or body)[:500]
    except ValueError:
        return (r.text or "")[:500]
