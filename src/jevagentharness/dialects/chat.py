"""The OpenAI Chat Completions dialect (``POST /v1/chat/completions``), as spoken by most agent harnesses."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator, Mapping

from ..types import Message, Part, Request, Response, Tool, ToolCall

# Gemini's OpenAI-compatible endpoint carries the thought signature here; a client that keeps it sends it back
_SIG_PATH = ("extra_content", "google", "thought_signature")


def _parts(content: Any) -> list[Part]:
    if content is None:
        return []
    if isinstance(content, str):
        return [Part(text=content)] if content else []
    out: list[Part] = []
    for c in content if isinstance(content, list) else [content]:
        if not isinstance(c, dict):
            out.append(Part(text=str(c)))
            continue
        t = c.get("type")
        if t in ("text", "input_text", "output_text"):
            out.append(Part(text=c.get("text", "")))
        elif t == "image_url":
            url = (c.get("image_url") or {}).get("url", "") if isinstance(c.get("image_url"), dict) \
                else str(c.get("image_url") or "")
            if url.startswith("data:") and ";base64," in url:
                mime, data = url[5:].split(";base64,", 1)
                out.append(Part(mime=mime, data=data))
            else:
                out.append(Part(text=f"[image: {url[:200]}]"))
    return out


def _sig(c: Mapping[str, Any]) -> str | None:
    cur: Any = c
    for k in _SIG_PATH:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur if isinstance(cur, str) else None


def parse_request(body: Mapping[str, Any]) -> Request:
    msgs: list[Message] = []
    for m in body.get("messages") or []:
        role = m.get("role", "user")
        if role == "developer":
            role = "system"
        if role == "function":
            role = "tool"
        if role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                args = fn.get("arguments")
                calls.append(ToolCall(id=c.get("id") or f"call_{uuid.uuid4().hex[:24]}", name=fn.get("name", ""),
                                      arguments=args if isinstance(args, str) else json.dumps(args or {}),
                                      signature=_sig(c)))
            msgs.append(Message(role="assistant", parts=_parts(m.get("content")), tool_calls=calls,
                                reasoning=m.get("reasoning_content") or m.get("reasoning")))
        elif role == "tool":
            msgs.append(Message(role="tool", parts=_parts(m.get("content")), tool_call_id=m.get("tool_call_id"),
                                name=m.get("name")))
        else:
            msgs.append(Message(role=role, parts=_parts(m.get("content"))))
    tools = []
    for t in body.get("tools") or []:
        fn = t.get("function") or t
        tools.append(Tool(name=fn.get("name", ""), description=fn.get("description") or "",
                          parameters=fn.get("parameters") or {"type": "object", "properties": {}}))
    if not tools and body.get("functions"):
        tools = [Tool(name=f.get("name", ""), description=f.get("description") or "",
                      parameters=f.get("parameters") or {"type": "object", "properties": {}})
                 for f in body["functions"]]
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        tc = {"name": (tc.get("function") or {}).get("name") or tc.get("name")}
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    stop = body.get("stop")
    return Request(messages=msgs, tools=tools, tool_choice=tc,
                   max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
                   temperature=body.get("temperature"), top_p=body.get("top_p"),
                   stop=[stop] if isinstance(stop, str) else stop, reasoning=effort, model=str(body.get("model", "")),
                   stream=bool(body.get("stream")), parallel_tool_calls=body.get("parallel_tool_calls"),
                   response_format=body.get("response_format"), session=body.get("user") or None)


def _call_json(c: ToolCall, i: int | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
    if i is not None:
        d["index"] = i
    if c.signature:
        d["extra_content"] = {"google": {"thought_signature": c.signature}}
    return d


def _usage(r: Response) -> dict[str, Any]:
    u = r.usage
    return {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens + u.reasoning_tokens,
            "total_tokens": u.input_tokens + u.output_tokens + u.reasoning_tokens,
            "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": u.reasoning_tokens}}


def session_hint(headers: Mapping[str, str], body: Mapping[str, Any]) -> str | None:
    return None


def render_response(r: Response, model: str, req: Request | None = None,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": r.text or None}
    if r.tool_calls:
        msg["tool_calls"] = [_call_json(c) for c in r.tool_calls]
    out = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
           "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": r.finish_reason}],
           "usage": _usage(r)}
    if extra:
        out["jah"] = dict(extra)
    return out


def sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


def stream_preamble(req: Request, model: str) -> Iterator[bytes]:
    return iter(())


def render_stream_error(message: str, status: int, model: str, req: Request | None = None) -> Iterator[bytes]:
    yield sse({"error": {"code": status, "message": message, "type": "jah_error"}})
    yield b"data: [DONE]\n\n"


def render_stream(r: Response, model: str, req: Request | None = None, include_usage: bool = True
                  ) -> Iterator[bytes]:
    """The whole response as Chat Completions chunks (the upstream call is not streamed; the client's is)."""
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield sse(chunk({"role": "assistant", "content": ""}))
    if r.text:
        yield sse(chunk({"content": r.text}))
    for i, c in enumerate(r.tool_calls):
        yield sse(chunk({"tool_calls": [_call_json(c, i)]}))
    yield sse(chunk({}, r.finish_reason))
    if include_usage:
        yield sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [],
                   "usage": _usage(r)})
    yield b"data: [DONE]\n\n"
