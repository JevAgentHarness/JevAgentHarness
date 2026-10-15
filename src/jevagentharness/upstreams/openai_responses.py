"""The OpenAI Responses API (``POST {base}/responses``) as an upstream.

OpenAI's reasoning models take function tools together with a reasoning effort only on this API. Requests are
stateless (``store: false``): the proxy sends the whole history every time, with ``prompt_cache_key`` set to the
session so OpenAI's automatic prompt cache keeps hitting.

Reasoning items: a reasoning model returns encrypted reasoning before its function calls and works best when it gets
them back on the next turn. Clients rarely keep them, so the proxy remembers them by the call id they precede
(:class:`ReasoningStore`) and puts them back, but only for the model that produced them (the router may switch
models between turns).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Mapping

from ..types import Message, Request, Response, ToolCall, Usage


class ReasoningStore:
    """first call id of a turn -> (reasoning items, model); bounded LRU."""

    def __init__(self, cap: int = 50_000) -> None:
        self.cap = cap
        self._d: OrderedDict[str, tuple[list[dict[str, Any]], str]] = OrderedDict()

    def put(self, call_id: str, items: list[dict[str, Any]], model: str) -> None:
        self._d[call_id] = (items, model)
        self._d.move_to_end(call_id)
        while len(self._d) > self.cap:
            self._d.popitem(last=False)

    def get(self, call_id: str) -> tuple[list[dict[str, Any]], str] | None:
        return self._d.get(call_id)


def _content(m: Message, role: str) -> list[dict[str, Any]]:
    kind = "output_text" if role == "assistant" else "input_text"
    out: list[dict[str, Any]] = []
    for p in m.parts:
        if p.text:
            out.append({"type": kind, "text": p.text})
        elif p.data and p.mime and role != "assistant":
            out.append({"type": "input_image", "image_url": f"data:{p.mime};base64,{p.data}"})
    return out


def to_responses(req: Request, model: str, effort: str | None, store: ReasoningStore | None = None,
                 session: str | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for m in req.turns():
        if m.role == "tool":
            items.append({"type": "function_call_output", "call_id": m.tool_call_id or "", "output": m.text})
        elif m.role == "assistant":
            if m.text:
                items.append({"type": "message", "role": "assistant", "content": _content(m, "assistant")})
            if m.tool_calls and store is not None:
                kept = store.get(m.tool_calls[0].id)
                if kept is not None and kept[1] == model:
                    items.extend(kept[0])
            for c in m.tool_calls:
                args = json.dumps({"input": c.arguments}) if c.kind == "custom" else (c.arguments or "{}")
                items.append({"type": "function_call", "call_id": c.id, "name": c.name, "arguments": args})
        else:
            content = _content(m, "user")
            if content:
                items.append({"type": "message", "role": "user", "content": content})
    body: dict[str, Any] = {"model": model, "input": items, "store": False}
    if req.system:
        body["instructions"] = req.system
    if req.tools:
        body["tools"] = [{"type": "function", "name": t.name, "description": t.description or t.name,
                          "parameters": t.parameters or {"type": "object", "properties": {}}, "strict": False}
                         for t in req.tools]
        tc = req.tool_choice
        if isinstance(tc, dict) and tc.get("name"):
            body["tool_choice"] = {"type": "function", "name": tc["name"]}
        elif tc in ("auto", "none", "required"):
            body["tool_choice"] = tc
        if req.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = req.parallel_tool_calls
    if effort:
        body["reasoning"] = {"effort": effort}
        body["include"] = ["reasoning.encrypted_content"]
    if req.max_tokens:
        body["max_output_tokens"] = int(req.max_tokens)
    if session:
        body["prompt_cache_key"] = session[:64]
    for k, v in (extra or {}).items():
        body[k] = v
    return body


def from_responses(data: Mapping[str, Any], model: str, store: ReasoningStore | None = None) -> Response:
    text: list[str] = []
    calls: list[ToolCall] = []
    reasoning_items: list[dict[str, Any]] = []
    summary: list[str] = []
    for it in data.get("output") or []:
        t = it.get("type")
        if t == "message":
            for c in it.get("content") or []:
                if c.get("type") in ("output_text", "text"):
                    text.append(c.get("text") or "")
        elif t == "function_call":
            calls.append(ToolCall(id=it.get("call_id") or it.get("id") or "", name=it.get("name", ""),
                                  arguments=it.get("arguments") or "{}"))
        elif t == "reasoning":
            keep = {k: it[k] for k in ("type", "id", "summary", "encrypted_content") if k in it}
            keep.setdefault("summary", [])
            reasoning_items.append(keep)
            summary.extend(s.get("text", "") for s in it.get("summary") or [] if isinstance(s, dict))
    if store is not None and calls and reasoning_items:
        store.put(calls[0].id, reasoning_items, model)
    u = data.get("usage") or {}
    itd = u.get("input_tokens_details") or {}
    otd = u.get("output_tokens_details") or {}
    reasoning = int(otd.get("reasoning_tokens") or 0)
    out_total = int(u.get("output_tokens") or 0)
    usage = Usage(input_tokens=int(u.get("input_tokens") or 0), cached_tokens=int(itd.get("cached_tokens") or 0),
                  cache_write_tokens=int(itd.get("cache_write_tokens") or 0),
                  output_tokens=max(0, out_total - reasoning), reasoning_tokens=reasoning)
    status = data.get("status")
    finish = ("tool_calls" if calls else "length" if status == "incomplete" else "stop")
    return Response(text="".join(text), tool_calls=calls, finish_reason=finish, usage=usage,
                    model=str(data.get("model") or model), reasoning="\n".join(summary), raw_id=data.get("id"))
