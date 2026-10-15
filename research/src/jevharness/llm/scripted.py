"""Deterministic, offline System-2 stand-in used in tests and dry runs.

``ScriptedLLM`` delegates to a python policy ``fn(messages, tools) -> ChatResponse``. If no
policy is given it calls the first tool with empty arguments, or answers "done".
Token counts are estimated from text length so cost/latency plumbing is exercised offline.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from ..accounting import estimate_tokens
from .base import ChatResponse, LLMClient, ToolCall

Policy = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], ChatResponse]


class ScriptedLLM(LLMClient):
    provider = "scripted"

    def __init__(self, model: str = "scripted", policy: Policy | None = None,
                 simulated_latency_ms: float = 0.0, **defaults: Any):
        super().__init__(model, **defaults)
        self.policy = policy
        self.simulated_latency_ms = simulated_latency_ms
        self.calls = 0

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        self.calls += 1
        t = time.perf_counter()
        if self.policy is not None:
            resp = self.policy(messages, tools)
        elif tools:
            name = tools[0]["function"]["name"]
            resp = ChatResponse(text="", tool_calls=[ToolCall(id=f"call_{self.calls}", name=name, arguments={})])
        else:
            resp = ChatResponse(text="done")
        prompt = json.dumps(messages) + json.dumps(tools or [])
        resp.input_tokens = resp.input_tokens or estimate_tokens(prompt)
        out = resp.text + json.dumps([tc.arguments for tc in resp.tool_calls])
        resp.output_tokens = resp.output_tokens or estimate_tokens(out)
        resp.latency_ms = resp.latency_ms or ((time.perf_counter() - t) * 1000 + self.simulated_latency_ms)
        resp.model = resp.model or self.model
        return resp
