"""Any OpenAI-compatible Chat Completions endpoint as an upstream (OpenAI, OpenRouter, DeepSeek, vLLM, SGLang, ...)."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..types import Message, Request, Response, ToolCall, Usage


def _content(m: Message) -> Any:
    if not any(p.data for p in m.parts):
        return m.text
    out: list[dict[str, Any]] = []
    for p in m.parts:
        if p.text:
            out.append({"type": "text", "text": p.text})
        elif p.data and p.mime:
            out.append({"type": "image_url", "image_url": {"url": f"data:{p.mime};base64,{p.data}"}})
    return out


def to_chat(req: Request, model: str, effort: str | None, extra: Mapping[str, Any] | None = None,
            effort_field: str = "reasoning_effort") -> dict[str, Any]:
    msgs: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.text})
        elif m.role == "assistant":
            d: dict[str, Any] = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                d["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": (
                                        json.dumps({"input": c.arguments}) if c.kind == "custom" else c.arguments)}}
                                   for c in m.tool_calls]
            msgs.append(d)
        else:
            msgs.append({"role": m.role, "content": _content(m)})
    body: dict[str, Any] = {"model": model, "messages": msgs}
    if req.tools:
        body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                           "parameters": t.parameters}} for t in req.tools]
        if req.tool_choice is not None:
            tc = req.tool_choice
            body["tool_choice"] = ({"type": "function", "function": {"name": tc["name"]}}
                                   if isinstance(tc, dict) else tc)
        if req.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = req.parallel_tool_calls
    if req.max_tokens:
        body["max_tokens"] = int(req.max_tokens)
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop"] = req.stop
    if req.response_format is not None:
        body["response_format"] = req.response_format
    if effort:
        if effort_field == "reasoning":          # OpenRouter's unified field
            body["reasoning"] = {"effort": effort}
        else:
            body[effort_field] = effort
    for k, v in (extra or {}).items():
        body[k] = v
    return body


def from_chat(data: Mapping[str, Any], model: str, custom_tools: set[str] | None = None) -> Response:
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    calls = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments") or "{}"
        kind = "function"
        if custom_tools and name in custom_tools:
            kind = "custom"
            try:
                args = str(json.loads(args).get("input", ""))
            except (json.JSONDecodeError, AttributeError):
                pass
        calls.append(ToolCall(id=c.get("id") or "", name=name, arguments=args, kind=kind))
    u = data.get("usage") or {}
    ptd = u.get("prompt_tokens_details") or {}
    ctd = u.get("completion_tokens_details") or {}
    reasoning = int(ctd.get("reasoning_tokens") or 0)
    completion = int(u.get("completion_tokens") or 0)
    usage = Usage(input_tokens=int(u.get("prompt_tokens") or 0), cached_tokens=int(ptd.get("cached_tokens") or 0),
                  output_tokens=max(0, completion - reasoning), reasoning_tokens=reasoning,
                  reported_cost=(float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else None))
    fr = ch.get("finish_reason") or ("tool_calls" if calls else "stop")
    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
    return Response(text=content or "", tool_calls=calls, finish_reason="tool_calls" if calls else fr, usage=usage,
                    model=str(data.get("model") or model),
                    reasoning=str(msg.get("reasoning_content") or msg.get("reasoning") or ""), raw_id=data.get("id"))
