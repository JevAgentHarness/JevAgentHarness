"""Gemini's native API (``POST {base}/models/{model}:generateContent``) as an upstream.

Translation notes:

* ``system`` messages become ``systemInstruction``; ``assistant`` turns become ``model`` turns; consecutive ``tool``
  messages become one ``user`` turn of ``functionResponse`` parts (Gemini wants one response per call, in one turn).
* Tool schemas go out as ``parametersJsonSchema`` (full JSON Schema), so client schemas need no down-conversion.
* Gemini 3 and later attach an opaque *thought signature* to function calls and expect it back. Clients that speak
  OpenAI dialects drop it, so the proxy remembers every signature by tool-call id (:class:`SignatureStore`) and puts it
  back. A call whose signature is unknown, or was produced by a different model (the router switches models), gets
  Google's documented placeholder instead.
"""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from typing import Any, Mapping

from ..types import Message, Request, Response, ToolCall, Usage

DUMMY_SIGNATURE = "skip_thought_signature_validator"
EFFORTS = {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "none": "minimal"}
_SCHEMA_DROP = {"$schema", "$id", "strict"}


class SignatureStore:
    """tool-call id -> (signature, model); bounded LRU so a long-running proxy does not grow without limit."""

    def __init__(self, cap: int = 200_000) -> None:
        self.cap = cap
        self._d: OrderedDict[str, tuple[str, str]] = OrderedDict()

    def put(self, call_id: str, signature: str, model: str) -> None:
        self._d[call_id] = (signature, model)
        self._d.move_to_end(call_id)
        while len(self._d) > self.cap:
            self._d.popitem(last=False)

    def get(self, call_id: str) -> tuple[str, str] | None:
        return self._d.get(call_id)


def clean_schema(s: Any) -> Any:
    if isinstance(s, dict):
        return {k: clean_schema(v) for k, v in s.items() if k not in _SCHEMA_DROP}
    if isinstance(s, list):
        return [clean_schema(v) for v in s]
    return s


def _parts(m: Message) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in m.parts:
        if p.text:
            out.append({"text": p.text})
        elif p.data and p.mime:
            out.append({"inlineData": {"mimeType": p.mime, "data": p.data}})
    return out


def _function_response(m: Message, req: Request) -> dict[str, Any]:
    name = m.name or req.tool_name(m.tool_call_id) or "tool"
    return {"functionResponse": {"name": name, "response": {"output": m.text}}}


def to_gemini(req: Request, model: str, effort: str | None, sigs: SignatureStore | None = None,
              extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    contents: list[dict[str, Any]] = []
    pending_tool: list[dict[str, Any]] = []
    image_parts: list[dict[str, Any]] = []

    def flush_tools() -> None:
        if pending_tool:
            contents.append({"role": "user", "parts": list(pending_tool) + list(image_parts)})
            pending_tool.clear()
            image_parts.clear()

    for m in req.turns():
        if m.role == "tool":
            pending_tool.append(_function_response(m, req))
            image_parts.extend(p for p in _parts(m) if "inlineData" in p)
            continue
        flush_tools()
        if m.role == "assistant":
            parts: list[dict[str, Any]] = []
            if m.text:
                parts.append({"text": m.text})
            for i, c in enumerate(m.tool_calls):
                part: dict[str, Any] = {"functionCall": {"name": c.name, "args": c.args_obj()}}
                sig = None
                known = sigs.get(c.id) if sigs is not None else None
                if known is not None:          # the proxy saw this call: it knows which model signed it
                    sig = known[0] if known[1] == model else None
                elif c.signature and c.signature_model in (None, model):
                    sig = c.signature
                if sig is None and i == 0:
                    sig = DUMMY_SIGNATURE   # Gemini requires a signature on the first call of a step
                if sig is not None:
                    part["thoughtSignature"] = sig
                parts.append(part)
            if not parts:
                parts.append({"text": ""})
            contents.append({"role": "model", "parts": parts})
        else:
            parts = _parts(m) or [{"text": ""}]
            if contents and contents[-1]["role"] == "user" and not any(
                    "functionResponse" in p for p in contents[-1]["parts"]):
                contents[-1]["parts"].extend(parts)          # Gemini wants alternating roles
            else:
                contents.append({"role": "user", "parts": parts})
    flush_tools()

    body: dict[str, Any] = {"contents": contents}
    if req.system:
        body["systemInstruction"] = {"parts": [{"text": req.system}]}
    if req.tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t.name, "description": t.description or t.name,
             "parametersJsonSchema": clean_schema(t.parameters or {"type": "object", "properties": {}})}
            for t in req.tools]}]
        tc = req.tool_choice
        mode = None
        if tc == "none":
            mode = {"mode": "NONE"}
        elif tc == "required":
            mode = {"mode": "ANY"}
        elif isinstance(tc, dict) and tc.get("name"):
            mode = {"mode": "ANY", "allowedFunctionNames": [tc["name"]]}
        if mode:
            body["toolConfig"] = {"functionCallingConfig": mode}
    gen: dict[str, Any] = {}
    if req.max_tokens:
        gen["maxOutputTokens"] = int(req.max_tokens)
    if req.temperature is not None:
        gen["temperature"] = float(req.temperature)
    if req.top_p is not None:
        gen["topP"] = float(req.top_p)
    if req.stop:
        gen["stopSequences"] = list(req.stop)[:5]
    if effort:
        gen["thinkingConfig"] = {"thinkingLevel": EFFORTS.get(effort, effort)}
    if gen:
        body["generationConfig"] = gen
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(body.get(k), dict):
            body[k] = {**body[k], **v}
        else:
            body[k] = v
    return body


def from_gemini(data: Mapping[str, Any], model: str, sigs: SignatureStore | None = None) -> Response:
    cands = data.get("candidates") or []
    cand = cands[0] if cands else {}
    parts = ((cand.get("content") or {}).get("parts")) or []
    text, thought, calls = [], [], []
    for p in parts:
        if "functionCall" in p:
            fc = p["functionCall"] or {}
            cid = fc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
            sig = p.get("thoughtSignature")
            calls.append(ToolCall(id=cid, name=fc.get("name", ""), arguments=json.dumps(fc.get("args") or {}),
                                  signature=sig, signature_model=model if sig else None))
            if sig and sigs is not None:
                sigs.put(cid, sig, model)
        elif p.get("thought"):
            thought.append(p.get("text") or "")
        elif "text" in p:
            text.append(p.get("text") or "")
    u = data.get("usageMetadata") or {}
    usage = Usage(input_tokens=int(u.get("promptTokenCount") or 0) + int(u.get("toolUsePromptTokenCount") or 0),
                  cached_tokens=int(u.get("cachedContentTokenCount") or 0),
                  output_tokens=int(u.get("candidatesTokenCount") or 0),
                  reasoning_tokens=int(u.get("thoughtsTokenCount") or 0))
    fr = str(cand.get("finishReason") or "STOP").upper()
    finish = ("tool_calls" if calls else "length" if fr == "MAX_TOKENS"
              else "content_filter" if fr in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT") else "stop")
    return Response(text="".join(text), tool_calls=calls, finish_reason=finish, usage=usage,
                    model=str(data.get("modelVersion") or model), reasoning="".join(thought),
                    raw_id=data.get("responseId"))


def is_signature_error(status: int, text: str) -> bool:
    t = text.lower()
    return status == 400 and "signature" in t and "thought" in t


def strip_signatures(body: dict[str, Any]) -> None:
    """Replace every signature by the placeholder (the retry after a signature error)."""
    for c in body.get("contents") or []:
        first = True
        for p in c.get("parts") or []:
            if "functionCall" in p:
                if first:
                    p["thoughtSignature"] = DUMMY_SIGNATURE
                    first = False
                else:
                    p.pop("thoughtSignature", None)
            elif "thoughtSignature" in p:
                p.pop("thoughtSignature", None)
