"""Provider-neutral conversation types.

Every client dialect (OpenAI Chat Completions, OpenAI Responses) is parsed into a :class:`Request`, every upstream
(Gemini's native API, any OpenAI-compatible endpoint) turns a :class:`Request` into its own wire body and its reply
into a :class:`Response`. The router only ever sees these types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str                       # JSON text for function tools, raw text for freeform tools
    kind: str = "function"               # "function" | "custom" (freeform: Responses API custom tools)
    signature: str | None = None         # Gemini thought signature (opaque)
    signature_model: str | None = None   # the model that produced the signature

    def args_obj(self) -> Any:
        if self.kind == "custom":
            return {"input": self.arguments}
        try:
            v = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {"input": self.arguments}
        return v if isinstance(v, dict) else {"value": v}


@dataclass
class Part:
    """A content part of a message: text or an inline image (base64 data)."""
    text: str | None = None
    mime: str | None = None
    data: str | None = None              # base64 payload of an image


@dataclass
class Message:
    role: str                            # "system" | "user" | "assistant" | "tool"
    parts: list[Part] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None      # role == "tool"
    name: str | None = None              # role == "tool": the function name, when the client sent it
    reasoning: str | None = None         # assistant reasoning text the client chose to send back (not forwarded)

    @property
    def text(self) -> str:
        return "".join(p.text or "" for p in self.parts if p.text)

    @classmethod
    def of(cls, role: str, text: str, **kw: Any) -> "Message":
        return cls(role=role, parts=[Part(text=text)] if text else [], **kw)


@dataclass
class Tool:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    kind: str = "function"               # "function" | "custom"
    meta: dict[str, Any] = field(default_factory=dict)   # dialect-specific bits (e.g. a custom tool's format)


@dataclass
class Request:
    messages: list[Message]
    tools: list[Tool] = field(default_factory=list)
    tool_choice: Any = None              # None | "auto" | "none" | "required" | {"name": ...}
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    reasoning: str | None = None         # the client's requested effort ("low" | "medium" | "high" | ...)
    model: str = ""                      # the model name the client asked for
    stream: bool = False
    parallel_tool_calls: bool | None = None
    response_format: Any = None
    session: str | None = None           # a client-supplied session key, when there is one

    @property
    def system(self) -> str:
        return "\n\n".join(m.text for m in self.messages if m.role == "system" and m.text)

    def turns(self) -> list[Message]:
        return [m for m in self.messages if m.role != "system"]

    def step_index(self) -> int:
        """The number of assistant turns already in the history (0 = the first request of an episode)."""
        return sum(1 for m in self.messages if m.role == "assistant")

    def tool_name(self, call_id: str | None) -> str | None:
        for m in self.messages:
            for c in m.tool_calls:
                if c.id == call_id:
                    return c.name
        return None


@dataclass
class Usage:
    input_tokens: int = 0                # all prompt tokens, cached included
    cached_tokens: int = 0
    cache_write_tokens: int = 0          # prompt tokens written to the provider's cache (OpenAI bills them extra)
    output_tokens: int = 0               # visible output
    reasoning_tokens: int = 0            # thinking tokens (billed as output)
    reported_cost: float | None = None   # the upstream's billed amount, when it reports one (OpenRouter)


@dataclass
class Response:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"          # "stop" | "tool_calls" | "length" | "content_filter"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    reasoning: str = ""                  # thought summary, when the upstream returns one
    raw_id: str | None = None
