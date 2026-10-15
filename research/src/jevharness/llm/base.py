"""System-2 (generative LLM) client abstraction.

Internally we use OpenAI-style chat messages and tool specs; provider adapters translate.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..accounting import CallRecord, Ledger, default_prices


def _routed_extras(raw: Any) -> dict[str, Any]:
    """Router metadata of a raw OpenAI-compatible response, for the ledger's call meta (PREREG_A7_STAGE2 A7.12 (b),
    (h)): the provider OpenRouter actually served (``served_provider``, the pin check) and the cost it reports
    (``provider_cost_usd``, logged next to the list-price cost). Empty for every other provider; never raises."""
    out: dict[str, Any] = {}
    try:
        mx = getattr(raw, "model_extra", None) or {}
        if mx.get("provider"):
            out["served_provider"] = str(mx["provider"])
        u = getattr(raw, "usage", None)
        ux = (getattr(u, "model_extra", None) or {}) if u is not None else {}
        if ux.get("cost") is not None:
            out["provider_cost_usd"] = float(ux["cost"])
    except Exception:  # noqa: BLE001 - metadata only
        return {}
    return out


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # provider-specific payload that must be echoed back verbatim, e.g. Gemini 3.x
    # {"google": {"thought_signature": "..."}} (OpenAI-compatible field `extra_content`)
    extra: dict[str, Any] | None = None

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": "function",
                             "function": {"name": self.name, "arguments": json.dumps(self.arguments)}}
        if self.extra:
            d["extra_content"] = self.extra
        return d


@dataclass
class ChatResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    model: str | None = None
    # top-k logprobs for each generated position: list[dict[token, logprob]]
    top_logprobs: list[dict[str, float]] | None = None
    finish_reason: str | None = None
    raw: Any = None
    # provider payload of a text-only reply that must be echoed back with it (e.g. OpenAI Responses reasoning
    # items, ``{"openai": {"reasoning": [...]}}``); tool-call payloads live in ``ToolCall.extra`` instead
    extra: dict[str, Any] | None = None
    # A6 ruling (e): transport bookkeeping copied into the CallRecord (None when the client does not count them)
    attempts: int | None = None
    retry_wait_ms: float | None = None
    request_id: str | None = None

    def assistant_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.extra:
            msg["extra_content"] = self.extra
        return msg


class LLMClient(ABC):
    provider: str = "base"

    def __init__(self, model: str, **defaults: Any):
        self.model = model
        self.defaults = defaults

    @abstractmethod
    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse: ...

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             ledger: Ledger | None = None, decision_point: str | None = None,
             component: str = "s2", **kw: Any) -> ChatResponse:
        params = {**self.defaults, **kw}
        # ``price_model`` (llm block, optional) names the price row to charge instead of the served model id: a dated row
        # for a router endpoint whose list price changed since an earlier experiment priced the same id (E13). It is
        # never sent to the provider.
        price_model = params.pop("price_model", None)
        resp = self._chat(messages, tools=tools, **params)
        if ledger is not None:
            prices = default_prices()
            price = prices.get(price_model or resp.model or self.model)
            cost = price.cost(resp.input_tokens, resp.output_tokens, resp.cached_input_tokens) if price else 0.0
            meta: dict[str, Any] = {"cached_input_tokens": resp.cached_input_tokens,
                                    "reasoning_tokens": resp.reasoning_tokens, "n_tools": len(tools or [])}
            if price_model:
                meta["price_model"] = price_model
            meta.update(_routed_extras(getattr(resp, "raw", None)))
            ledger.add(CallRecord(component=component, backend=self.provider, model=resp.model or self.model,
                                  decision_point=decision_point, input_tokens=resp.input_tokens,
                                  output_tokens=resp.output_tokens, cost_usd=cost,
                                  latency_ms=resp.latency_ms, priced=price is not None,
                                  meta=meta,
                                  attempts=resp.attempts, retry_wait_ms=resp.retry_wait_ms,
                                  request_id=resp.request_id))
        return resp

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model}
