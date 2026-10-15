"""OpenAI-compatible chat client.

Covers OpenAI, Gemini (OpenAI-compatible endpoint), DeepSeek, OpenRouter, Together, Fireworks, Groq,
and self-hosted vLLM/SGLang servers (anything exposing ``/v1/chat/completions``). Logprobs are
requested on demand so the same client powers the "small LLM as System-1" baseline
(see ``decision/llm_logprob.py``).

Key rotation: if ``<PROVIDER>_API_KEYS`` (comma-separated) is set, e.g. ``GEMINI_API_KEYS=k1,k2``,
requests are spread round-robin over the keys and a rate-limited / overloaded key is retried on the
next one with exponential backoff. Otherwise ``<PROVIDER>_API_KEY`` is used.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import threading
import time
from typing import Any

from .base import ChatResponse, LLMClient, ToolCall

# provider name -> (base_url, api-key env var)
PROVIDERS: dict[str, tuple[str | None, str]] = {
    "openai": (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "vllm": (os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"), "VLLM_API_KEY"),
}

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
# Gemini 3.x rejects function calls in the history that lack a thought signature. Calls that Gemini
# did not generate (e.g. made by the System-1 model) carry this documented placeholder instead.
GEMINI_DUMMY_SIGNATURE = "skip_thought_signature_validator"


def _gemini_sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m = dict(m)
            tcs = []
            for tc in m["tool_calls"]:
                if not (tc.get("extra_content") or {}).get("google", {}).get("thought_signature"):
                    tc = {**tc, "extra_content": {"google": {"thought_signature": GEMINI_DUMMY_SIGNATURE}}}
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def provider_keys(key_env: str, explicit: str | None = None) -> list[str]:
    """Keys for a provider: explicit > <X>_API_KEYS (comma-separated) > <X>_API_KEY."""
    if explicit:
        return [explicit]
    multi = os.environ.get(key_env.replace("_API_KEY", "_API_KEYS"), "")
    keys = [k.strip() for k in multi.split(",") if k.strip()]
    if not keys and os.environ.get(key_env):
        keys = [os.environ[key_env]]
    return keys or ["EMPTY"]


class OpenAICompatClient(LLMClient):
    provider = "openai_compat"

    def __init__(self, model: str, provider: str = "openai", base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 120.0, max_retries: int = 6, **defaults: Any):
        super().__init__(model, **defaults)
        from openai import OpenAI  # local import keeps the package importable without openai

        default_url, key_env = PROVIDERS.get(provider, (None, "OPENAI_API_KEY"))
        self.provider = provider
        self.max_retries = max_retries
        self.clients = [OpenAI(base_url=base_url or default_url, api_key=k, timeout=timeout, max_retries=0)
                        for k in provider_keys(key_env, api_key)]
        self.client = self.clients[0]
        self._rr = itertools.count()
        self._lock = threading.Lock()
        self.n_retries = 0

    def _create(self, params: dict[str, Any], timing: dict[str, float] | None = None) -> Any:
        """``timing["ms"]`` receives the latency of the successful attempt only: failed attempts and
        429/5xx backoff sleeps are quota artefacts, excluded exactly as the Jev client excludes them."""
        import openai

        delay = 1.0
        last: Exception | None = None
        waited = 0.0
        for attempt in range(self.max_retries + 1):
            with self._lock:
                client = self.clients[next(self._rr) % len(self.clients)]
            t = time.perf_counter()
            try:
                r = client.chat.completions.create(**params)
                if timing is not None:
                    timing["ms"] = (time.perf_counter() - t) * 1000
                    timing["attempts"] = attempt + 1
                    timing["retry_wait_ms"] = waited * 1000      # A6 ruling (e): backoff slept before this attempt
                return r
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last = e
            except openai.APIStatusError as e:
                if e.status_code not in RETRY_STATUS:
                    raise
                last = e
                ra = e.response.headers.get("retry-after") if e.response is not None else None
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

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        if self.provider == "gemini":
            messages = _gemini_sanitize(messages)
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            params["tools"] = tools
            if "tool_choice" in kw:
                params["tool_choice"] = kw.pop("tool_choice")
        top_logprobs = kw.pop("top_logprobs", None)
        if top_logprobs:
            params["logprobs"] = True
            params["top_logprobs"] = int(top_logprobs)
        # vLLM / Qwen3: {"chat_template_kwargs": {"enable_thinking": False}} etc.
        extra_body = kw.pop("extra_body", None)
        if extra_body:
            params["extra_body"] = extra_body
        for k in ("temperature", "max_tokens", "max_completion_tokens", "top_p", "seed", "stop",
                  "response_format", "reasoning_effort", "n"):
            if k in kw and kw[k] is not None:
                params[k] = kw[k]
        timing: dict[str, float] = {}
        t = time.perf_counter()
        r = self._create(params, timing=timing)
        latency = timing.get("ms", (time.perf_counter() - t) * 1000)
        choice = r.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__raw__": tc.function.arguments}
            extra = (getattr(tc, "model_extra", None) or {}).get("extra_content")
            calls.append(ToolCall(id=tc.id or f"call_{random.getrandbits(32):08x}", name=tc.function.name,
                                  arguments=args if isinstance(args, dict) else {"__raw__": args},
                                  extra=extra if isinstance(extra, dict) else None))
        u = r.usage
        cached = 0
        reasoning = 0
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
            # DeepSeek reports cache hits separately
            cached = cached or int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
            cdet = getattr(u, "completion_tokens_details", None)
            reasoning = int(getattr(cdet, "reasoning_tokens", 0) or 0) if cdet else 0
        prompt_tokens = int(u.prompt_tokens or 0) if u else 0
        completion_tokens = int(u.completion_tokens or 0) if u else 0
        total = int(getattr(u, "total_tokens", 0) or 0) if u is not None else 0
        if u is not None and not reasoning:
            # Gemini's OpenAI endpoint folds thinking tokens into total_tokens only
            reasoning = max(0, total - prompt_tokens - completion_tokens)
        output_tokens = completion_tokens
        if self.provider == "gemini":
            # Gemini bills thinking as output but reports it either only via total_tokens or as
            # reasoning_tokens (with completion_tokens in- or exclusive of it); total - prompt counts it
            # exactly once in every case.
            output_tokens = max(completion_tokens, total - prompt_tokens) if total else completion_tokens + reasoning
        tlp = None
        if top_logprobs and choice.logprobs and choice.logprobs.content:
            tlp = [{alt.token: alt.logprob for alt in pos.top_logprobs} for pos in choice.logprobs.content]
        # reasoning tokens are billed as output tokens by every provider we use
        return ChatResponse(text=msg.content or "", tool_calls=calls,
                            input_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached, reasoning_tokens=reasoning,
                            latency_ms=latency, model=r.model or self.model, top_logprobs=tlp,
                            finish_reason=choice.finish_reason, raw=r,
                            attempts=int(timing["attempts"]) if "attempts" in timing else None,
                            retry_wait_ms=float(timing["retry_wait_ms"]) if "retry_wait_ms" in timing else None,
                            request_id=(getattr(r, "_request_id", None) or getattr(r, "id", None) or None))
