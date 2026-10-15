"""The OpenAI Responses dialect (``POST /v1/responses``), as spoken by Codex.

Requests are stateless here (Codex sends ``store: false`` and the whole history in ``input``). Freeform ``custom`` tools
(Codex's ``apply_patch``) go upstream as functions with one string argument and come back as ``custom_tool_call``
items; ``local_shell`` goes upstream as a ``local_shell`` function and comes back as ``local_shell_call``. Hosted tools
the upstream cannot run (``web_search`` and the like) are not offered to it. Reasoning items are dropped from the
history: their encrypted content belongs to OpenAI's models.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator, Mapping

from ..types import Message, Part, Request, Response, Tool, ToolCall

LOCAL_SHELL = "local_shell"
LOCAL_SHELL_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "array", "items": {"type": "string"},
                               "description": "the command and its arguments, e.g. [\"bash\", \"-lc\", \"ls\"]"},
                   "workdir": {"type": "string"}, "timeout_ms": {"type": "integer"}},
    "required": ["command"]}


def _parts(content: Any) -> list[Part]:
    if content is None:
        return []
    if isinstance(content, str):
        return [Part(text=content)] if content else []
    out: list[Part] = []
    for c in content:
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        if t in ("input_text", "output_text", "text", "summary_text", "refusal"):
            out.append(Part(text=c.get("text") or c.get("refusal") or ""))
        elif t == "input_image":
            url = c.get("image_url") or ""
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                mime, data = url[5:].split(";base64,", 1)
                out.append(Part(mime=mime, data=data))
        elif t == "input_file":
            out.append(Part(text=f"[file: {c.get('filename') or c.get('file_id') or ''}]"))
    return out


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(p.text or "" for p in _parts(output))
    if isinstance(output, dict):
        return str(output.get("content") or output.get("output") or json.dumps(output))
    return "" if output is None else str(output)


def parse_request(body: Mapping[str, Any]) -> Request:
    msgs: list[Message] = []
    if body.get("instructions"):
        msgs.append(Message.of("system", str(body["instructions"])))
    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]

    def assistant() -> Message:
        """Consecutive assistant items (text, calls) form one assistant turn."""
        if msgs and msgs[-1].role == "assistant" and not getattr(msgs[-1], "_closed", False):
            return msgs[-1]
        m = Message(role="assistant")
        msgs.append(m)
        return m

    for it in items or []:
        t = it.get("type", "message")
        if t == "message":
            role = it.get("role", "user")
            role = "system" if role == "developer" else role
            if role == "assistant":
                a = assistant()
                a.parts.extend(_parts(it.get("content")))
            else:
                msgs.append(Message(role=role, parts=_parts(it.get("content"))))
        elif t == "function_call":
            assistant().tool_calls.append(ToolCall(id=it.get("call_id") or it.get("id") or "",
                                                   name=_flat(it.get("namespace"), it.get("name", "")),
                                                   arguments=it.get("arguments") or "{}"))
        elif t == "custom_tool_call":
            assistant().tool_calls.append(ToolCall(id=it.get("call_id") or it.get("id") or "",
                                                   name=_flat(it.get("namespace"), it.get("name", "")),
                                                   arguments=it.get("input") or "", kind="custom"))
        elif t == "local_shell_call":
            action = it.get("action") or {}
            args = {k: action[k] for k in ("command", "workdir", "timeout_ms") if k in action}
            assistant().tool_calls.append(ToolCall(id=it.get("call_id") or it.get("id") or "", name=LOCAL_SHELL,
                                                   arguments=json.dumps(args)))
        elif t in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
            msgs.append(Message(role="tool", parts=[Part(text=_output_text(it.get("output")))],
                                tool_call_id=it.get("call_id")))
        # reasoning, web_search_call, compaction summaries without text, ...: not forwarded
    tools: list[Tool] = []

    def add(t: Mapping[str, Any], ns: str | None) -> None:
        kind = t.get("type")
        name = _flat(ns, t.get("name", ""))
        meta: dict[str, Any] = {"namespace": ns, "name": t.get("name", "")} if ns else {}
        if kind == "function":
            tools.append(Tool(name=name, description=t.get("description") or "", meta=meta,
                              parameters=t.get("parameters") or {"type": "object", "properties": {}}))
        elif kind == "custom":
            fmt = t.get("format") or {}
            tools.append(Tool(name=name, description=t.get("description") or "", kind="custom",
                              meta={**meta, "format": fmt.get("definition") or fmt.get("type") or ""}))
        elif kind == "local_shell":
            tools.append(Tool(name=LOCAL_SHELL, description="Run a command on the local machine.",
                              parameters=LOCAL_SHELL_SCHEMA, meta={"local_shell": True}))
        elif kind == "namespace":
            for inner in t.get("tools") or []:
                add(inner, t.get("name"))
        # hosted tools (web_search, tool_search, image_generation, ...) are not offered to the upstream

    for t in body.get("tools") or []:
        add(t, None)
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        tc = {"name": tc.get("name")} if tc.get("name") else "auto"
    reasoning = body.get("reasoning") or {}
    return Request(messages=msgs, tools=tools, tool_choice=tc, max_tokens=body.get("max_output_tokens"),
                   temperature=body.get("temperature"), top_p=body.get("top_p"),
                   reasoning=reasoning.get("effort") if isinstance(reasoning, dict) else None,
                   model=str(body.get("model", "")), stream=bool(body.get("stream")),
                   parallel_tool_calls=body.get("parallel_tool_calls"),
                   session=body.get("prompt_cache_key") or None)


def session_hint(headers: Mapping[str, str], body: Mapping[str, Any]) -> str | None:
    return (body.get("prompt_cache_key") or headers.get("session-id") or headers.get("session_id")
            or headers.get("thread-id") or None)


NS_SEP = "__"


def _flat(ns: str | None, name: str) -> str:
    """Namespaced tools (Codex's MCP and multi-agent tools) go upstream under one flat name."""
    return f"{ns}{NS_SEP}{name}" if ns else name


# ---------------------------------------------------------------------------------------------- output items
def _items(r: Response, req: Request | None) -> list[dict[str, Any]]:
    local = {t.name for t in (req.tools if req else []) if t.meta.get("local_shell")}
    spaced = {t.name: (t.meta["namespace"], t.meta["name"]) for t in (req.tools if req else [])
              if t.meta.get("namespace")}
    out: list[dict[str, Any]] = []
    if r.text:
        out.append({"type": "message", "id": f"msg_{uuid.uuid4().hex[:24]}", "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": r.text, "annotations": []}]})
    for c in r.tool_calls:
        iid = f"fc_{uuid.uuid4().hex[:24]}"
        ns: dict[str, str] = {}
        name = c.name
        if c.name in spaced:
            ns = {"namespace": spaced[c.name][0]}
            name = spaced[c.name][1]
        if c.kind == "custom":
            out.append({"type": "custom_tool_call", "id": iid, "status": "completed", "call_id": c.id,
                        "name": name, "input": c.arguments, **ns})
        elif c.name in local:
            try:
                args = json.loads(c.arguments or "{}")
            except json.JSONDecodeError:
                args = {"command": ["bash", "-lc", c.arguments]}
            cmd = args.get("command")
            if isinstance(cmd, str):
                cmd = ["bash", "-lc", cmd]
            action = {"type": "exec", "command": cmd or [], "env": {}}
            for k in ("workdir", "timeout_ms"):
                if k in args:
                    action[k] = args[k]
            out.append({"type": "local_shell_call", "id": iid, "status": "completed", "call_id": c.id,
                        "action": action})
        else:
            out.append({"type": "function_call", "id": iid, "status": "completed", "call_id": c.id,
                        "name": name, "arguments": c.arguments, **ns})
    return out


def _usage(r: Response) -> dict[str, Any]:
    u = r.usage
    out = u.output_tokens + u.reasoning_tokens
    return {"input_tokens": u.input_tokens, "input_tokens_details": {"cached_tokens": u.cached_tokens},
            "output_tokens": out, "output_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
            "total_tokens": u.input_tokens + out}


def _envelope(rid: str, model: str, status: str, output: list[dict[str, Any]] | None = None,
              usage: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": rid, "object": "response", "created_at": int(time.time()), "status": status, "model": model,
            "output": output or [], "usage": usage, "error": error, "incomplete_details": None}


def render_response(r: Response, model: str, req: Request | None = None) -> dict[str, Any]:
    return _envelope(f"resp_{uuid.uuid4().hex}", model, "completed", _items(r, req), _usage(r))


class _Events:
    def __init__(self) -> None:
        self.seq = 0

    def ev(self, kind: str, **payload: Any) -> bytes:
        d = {"type": kind, "sequence_number": self.seq, **payload}
        self.seq += 1
        return f"event: {kind}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n".encode()


_STREAMS: dict[int, tuple[str, _Events]] = {}


def stream_preamble(req: Request, model: str) -> Iterator[bytes]:
    rid, ev = f"resp_{uuid.uuid4().hex}", _Events()
    _STREAMS[id(req)] = (rid, ev)
    yield ev.ev("response.created", response=_envelope(rid, model, "in_progress"))
    yield ev.ev("response.in_progress", response=_envelope(rid, model, "in_progress"))


def _stream_state(req: Request | None) -> tuple[str, _Events]:
    if req is not None and id(req) in _STREAMS:
        return _STREAMS.pop(id(req))
    return f"resp_{uuid.uuid4().hex}", _Events()


def render_stream(r: Response, model: str, req: Request | None = None) -> Iterator[bytes]:
    rid, ev = _stream_state(req)
    items = _items(r, req)
    for i, item in enumerate(items):
        if item["type"] == "message":
            text = item["content"][0]["text"]
            yield ev.ev("response.output_item.added", output_index=i,
                        item={**item, "status": "in_progress", "content": []})
            yield ev.ev("response.content_part.added", item_id=item["id"], output_index=i, content_index=0,
                        part={"type": "output_text", "text": "", "annotations": []})
            yield ev.ev("response.output_text.delta", item_id=item["id"], output_index=i, content_index=0, delta=text)
            yield ev.ev("response.output_text.done", item_id=item["id"], output_index=i, content_index=0, text=text)
            yield ev.ev("response.content_part.done", item_id=item["id"], output_index=i, content_index=0,
                        part=item["content"][0])
        elif item["type"] == "function_call":
            yield ev.ev("response.output_item.added", output_index=i,
                        item={**item, "status": "in_progress", "arguments": ""})
            yield ev.ev("response.function_call_arguments.delta", item_id=item["id"], output_index=i,
                        delta=item["arguments"])
            yield ev.ev("response.function_call_arguments.done", item_id=item["id"], output_index=i,
                        arguments=item["arguments"])
        elif item["type"] == "custom_tool_call":
            yield ev.ev("response.output_item.added", output_index=i, item={**item, "status": "in_progress",
                                                                             "input": ""})
            yield ev.ev("response.custom_tool_call_input.delta", item_id=item["id"], output_index=i,
                        delta=item["input"])
            yield ev.ev("response.custom_tool_call_input.done", item_id=item["id"], output_index=i,
                        input=item["input"])
        else:
            yield ev.ev("response.output_item.added", output_index=i, item={**item, "status": "in_progress"})
        yield ev.ev("response.output_item.done", output_index=i, item=item)
    yield ev.ev("response.completed", response=_envelope(rid, model, "completed", items, _usage(r)))


def render_stream_error(message: str, status: int, model: str, req: Request | None = None) -> Iterator[bytes]:
    rid, ev = _stream_state(req)
    code = "rate_limit_exceeded" if status == 429 else "server_error"
    yield ev.ev("response.failed", response=_envelope(rid, model, "failed",
                                                       error={"code": code, "message": message}))
