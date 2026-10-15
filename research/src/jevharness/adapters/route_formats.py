"""Wire formats of the routing proxy (:mod:`jevharness.adapters.route_proxy`): Gemini's native API and OpenAI chat.

The proxy sits between an unmodified agent harness and its LLM provider. For each request it needs five things from
the wire format, all here and all pure functions (no I/O), so they are unit-tested offline:

1. the *step index* of a request: the number of model turns already in its history (0 = the episode's first request);
2. the *episode view* of a request: the task (first user text) and the steps so far, each an action with its output;
3. the *first action* of a response (its first tool call, or "no tool call"), keyed for the equivalence rule of E29 (c) 2;
4. *usage and cost* of a response (Gemini: from ``usageMetadata`` at the configured prices; OpenRouter: ``usage.cost``);
5. request rewriting: the target model (Gemini: in the URL path; OpenAI: the body's ``model`` plus per-tier overrides
   such as OpenRouter's ``reasoning``) and Gemini's thought-signature fix.

Formats: ``gemini`` (``POST {base}/models/{model}:generateContent`` or ``:streamGenerateContent?alt=sse``; pi through
``models.json``'s ``baseUrl``, SWE-agent through litellm's ``api_base``) and ``openai`` (``POST
{base}/chat/completions``; mini-SWE-agent's ``OpenRouterModel``).
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

FORMAT_GEMINI = "gemini"
FORMAT_OPENAI = "openai"
FORMATS = (FORMAT_GEMINI, FORMAT_OPENAI)

NO_TOOL = "<no tool call>"
# Google's documented placeholder for a function call whose thought signature the current model did not produce
# (also adapters/pi/jev-router.ts and jevharness.llm.openai_compat)
GEMINI_DUMMY_SIGNATURE = "skip_thought_signature_validator"

# E4's state rendering (jevharness.adapters.mini_swe / adapters/swe_agent/jev_sweagent.py)
HISTORY_WINDOW = 8
OBS_CHARS = 2000
HIST_OBS_CHARS = 400
TASK_CHARS = 4000
ACTION_CHARS = 600
STATE_MAX_CHARS = 12000


# --------------------------------------------------------------------------------------------------- text helpers
def norm_ws(text: Any) -> str:
    """Strip and collapse every run of whitespace to one space (the equivalence rule's normalisation)."""
    return " ".join(str(text if text is not None else "").split())


def trunc(text: Any, n: int) -> str:
    """Head and tail of long text (the harness's ``_trunc``)."""
    s = "" if text is None else str(text)
    if len(s) <= n:
        return s
    half = max(1, n // 2)
    return f"{s[:half]}\n...[truncated]...\n{s[-half:]}"


def _parse_args(args: Any) -> Any:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return args
    return args


def render_call(name: str, args: Any, max_chars: int = 300) -> str:
    args = _parse_args(args)
    try:
        js = json.dumps(args if args is not None else {}, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        js = str(args)
    return f"{name}({js if len(js) <= max_chars else js[:max_chars] + '...'})"


# --------------------------------------------------------------------------------------------------- action keys
def action_key(name: str | None, args: Any) -> str:
    """Key of one tool call under E29's equivalence rule (registration (c) 2): ``bash`` -> the command;
    ``str_replace_editor`` -> (sub-command, path); a tool with a ``path`` argument -> the path; anything else -> the
    name and the canonical JSON of the arguments; every string whitespace-normalised. ``None`` -> "no tool call"."""
    if not name:
        return NO_TOOL
    a = _parse_args(args)
    if not isinstance(a, Mapping):
        return f"{name}:{norm_ws(a)}"
    if name == "bash" and "command" in a:
        return f"bash:{norm_ws(a.get('command'))}"
    if name == "str_replace_editor":
        return f"str_replace_editor:{norm_ws(a.get('command'))}:{norm_ws(a.get('path'))}"
    if "path" in a:
        return f"{name}:path:{norm_ws(a.get('path'))}"
    try:
        js = json.dumps(a, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        js = str(a)
    return f"{name}:{norm_ws(js)}"


def equivalent(key_a: str | None, key_b: str | None) -> bool | None:
    """Two first actions are equivalent iff their keys are equal; ``None`` (no parsed response) -> unknown."""
    if key_a is None or key_b is None:
        return None
    return key_a == key_b


# --------------------------------------------------------------------------------------------------- Gemini
def gemini_step_index(body: Mapping[str, Any]) -> int:
    return sum(1 for c in body.get("contents") or [] if isinstance(c, Mapping) and c.get("role") == "model")


def _gemini_output(fr: Mapping[str, Any]) -> str:
    resp = fr.get("response")
    if isinstance(resp, Mapping):
        for k in ("output", "content", "result", "error"):
            if k in resp:
                v = resp[k]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return json.dumps(resp, ensure_ascii=False)
    return "" if resp is None else str(resp)


def gemini_episode(body: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """(task, steps): the first user text and every model turn as ``{"action", "output"}`` (tool calls paired with the
    function responses that follow them; a text-only model turn is a "said: ..." step)."""
    task = ""
    steps: list[dict[str, str]] = []
    pending: list[int] = []          # indices of steps whose call has no output yet, in order
    for c in body.get("contents") or []:
        if not isinstance(c, Mapping):
            continue
        parts = [p for p in c.get("parts") or [] if isinstance(p, Mapping)]
        if c.get("role") == "model":
            calls = [p["functionCall"] for p in parts if isinstance(p.get("functionCall"), Mapping)]
            text = " ".join(str(p.get("text") or "") for p in parts if p.get("text") and not p.get("thought")).strip()
            if calls:
                for fc in calls:
                    pending.append(len(steps))
                    steps.append({"action": render_call(str(fc.get("name") or "?"), fc.get("args")), "output": ""})
            elif text:
                steps.append({"action": f"said: {trunc(text, 300)}", "output": ""})
        else:
            texts = []
            for p in parts:
                if isinstance(p.get("functionResponse"), Mapping):
                    out = _gemini_output(p["functionResponse"])
                    if pending:
                        steps[pending.pop(0)]["output"] = out
                    else:
                        steps.append({"action": f"{p['functionResponse'].get('name', '?')}(...)", "output": out})
                elif p.get("text"):
                    texts.append(str(p["text"]))
            if texts:
                if not task:
                    task = "\n".join(texts)
                elif steps and not steps[-1]["output"]:
                    steps[-1]["output"] = "\n".join(texts)
    return task, steps


def gemini_first_action(resp: Mapping[str, Any]) -> tuple[str, str]:
    """(key, text) of a (merged) Gemini response: its first function call's key and its visible text."""
    cands = resp.get("candidates") or []
    parts = ((cands[0] or {}).get("content") or {}).get("parts") or [] if cands else []
    key, texts = NO_TOOL, []
    for p in parts:
        if not isinstance(p, Mapping):
            continue
        if isinstance(p.get("functionCall"), Mapping) and key == NO_TOOL:
            fc = p["functionCall"]
            key = action_key(str(fc.get("name") or "?"), fc.get("args"))
        elif p.get("text") and not p.get("thought"):
            texts.append(str(p["text"]))
    return key, " ".join(texts).strip()


def gemini_first_call(resp: Mapping[str, Any]) -> str:
    """The first function call rendered for the decision model's state (``name({...})``), or the text."""
    cands = resp.get("candidates") or []
    parts = ((cands[0] or {}).get("content") or {}).get("parts") or [] if cands else []
    for p in parts:
        if isinstance(p, Mapping) and isinstance(p.get("functionCall"), Mapping):
            fc = p["functionCall"]
            return render_call(str(fc.get("name") or "?"), fc.get("args"), ACTION_CHARS)
    return "(no tool call)"


def parse_sse(raw: bytes | str) -> list[dict[str, Any]]:
    """The JSON objects of an SSE body (``data: {...}`` lines; anything else is ignored)."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def merge_gemini_chunks(chunks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """One response from streamed chunks: the parts of candidate 0 concatenated, the last ``usageMetadata``,
    ``modelVersion`` and ``finishReason``."""
    parts: list[Any] = []
    out: dict[str, Any] = {}
    finish = None
    for ch in chunks:
        for k in ("usageMetadata", "modelVersion", "responseId", "error"):
            if ch.get(k) is not None:
                out[k] = ch[k]
        cands = ch.get("candidates") or []
        if cands:
            c0 = cands[0] or {}
            parts.extend((c0.get("content") or {}).get("parts") or [])
            finish = c0.get("finishReason") or finish
    out["candidates"] = [{"content": {"role": "model", "parts": parts}, **({"finishReason": finish} if finish else {})}]
    return out


def gemini_usage(resp: Mapping[str, Any]) -> dict[str, int]:
    u = resp.get("usageMetadata") or {}
    prompt = int(u.get("promptTokenCount") or 0) + int(u.get("toolUsePromptTokenCount") or 0)
    cached = int(u.get("cachedContentTokenCount") or 0)
    out = int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0)
    return {"input_tokens": prompt, "cached_tokens": min(cached, prompt), "output_tokens": out}


def price_cost(prices: Mapping[str, Any] | None, usage: Mapping[str, int]) -> float | None:
    """USD of one call at per-M prices (``input_per_m``, ``output_per_m``, ``cached_input_per_m``); None if unpriced."""
    if not prices:
        return None
    inp, cached, out = (int(usage.get(k) or 0) for k in ("input_tokens", "cached_tokens", "output_tokens"))
    cin = float(prices.get("cached_input_per_m", prices.get("input_per_m", 0.0)))
    return ((inp - cached) * float(prices.get("input_per_m", 0.0)) + cached * cin
            + out * float(prices.get("output_per_m", 0.0))) / 1e6


def fill_missing_signatures(body: dict[str, Any], replace_all: bool = False) -> int:
    """Give every function-call part without a thought signature (or, with ``replace_all``, every function-call part)
    Google's placeholder, in place. Returns the number of parts changed."""
    n = 0
    for c in body.get("contents") or []:
        for p in (c or {}).get("parts") or [] if isinstance(c, Mapping) else []:
            if isinstance(p, dict) and isinstance(p.get("functionCall"), Mapping):
                if replace_all or not p.get("thoughtSignature"):
                    if p.get("thoughtSignature") != GEMINI_DUMMY_SIGNATURE:
                        p["thoughtSignature"] = GEMINI_DUMMY_SIGNATURE
                        n += 1
    return n


def is_signature_error(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status == 400 and "thought" in t and "signature" in t


# --------------------------------------------------------------------------------------------------- OpenAI chat
def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, Mapping) else str(p) for p in content)
    return "" if content is None else str(content)


def openai_step_index(body: Mapping[str, Any]) -> int:
    return sum(1 for m in body.get("messages") or [] if isinstance(m, Mapping) and m.get("role") == "assistant")


def openai_episode(body: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
    task = ""
    steps: list[dict[str, str]] = []
    by_id: dict[str, int] = {}
    for m in body.get("messages") or []:
        if not isinstance(m, Mapping):
            continue
        role = m.get("role")
        if role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                for tc in calls:
                    fn = (tc or {}).get("function") or {}
                    by_id[str((tc or {}).get("id"))] = len(steps)
                    steps.append({"action": render_call(str(fn.get("name") or "?"), fn.get("arguments")), "output": ""})
            else:
                steps.append({"action": f"said: {trunc(_content_text(m.get('content')), 300)}", "output": ""})
        elif role == "tool":
            i = by_id.get(str(m.get("tool_call_id")))
            out = _content_text(m.get("content"))
            if i is not None:
                steps[i]["output"] = out
            elif steps:
                steps[-1]["output"] = out
        elif role == "user":
            text = _content_text(m.get("content"))
            if not task:
                task = text
            elif steps and not steps[-1]["output"]:
                steps[-1]["output"] = text
    return task, steps


def openai_first_action(resp: Mapping[str, Any]) -> tuple[str, str]:
    choices = resp.get("choices") or []
    msg = (choices[0] or {}).get("message") or {} if choices else {}
    calls = msg.get("tool_calls") or []
    text = _content_text(msg.get("content")).strip()
    if calls:
        fn = (calls[0] or {}).get("function") or {}
        return action_key(str(fn.get("name") or "?"), fn.get("arguments")), text
    return NO_TOOL, text


def openai_first_call(resp: Mapping[str, Any]) -> str:
    choices = resp.get("choices") or []
    msg = (choices[0] or {}).get("message") or {} if choices else {}
    calls = msg.get("tool_calls") or []
    if calls:
        fn = (calls[0] or {}).get("function") or {}
        return render_call(str(fn.get("name") or "?"), fn.get("arguments"), ACTION_CHARS)
    return "(no tool call)"


def openai_usage(resp: Mapping[str, Any]) -> dict[str, Any]:
    u = resp.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    out = {"input_tokens": int(u.get("prompt_tokens") or 0),
           "cached_tokens": int((det or {}).get("cached_tokens") or 0),
           "output_tokens": int(u.get("completion_tokens") or 0)}
    if isinstance(u.get("cost"), (int, float)):
        out["reported_cost"] = float(u["cost"])
    return out


def openai_rewrite(body: Mapping[str, Any], model: str, overrides: Mapping[str, Any] | None = None,
                   stream: bool | None = None) -> dict[str, Any]:
    """A copy of the request with the tier's model and body overrides (e.g. OpenRouter's ``reasoning``)."""
    b = copy.deepcopy(dict(body))
    b["model"] = model
    for k, v in (overrides or {}).items():
        b[k] = copy.deepcopy(v)
    if stream is not None:
        b["stream"] = stream
    return b


# --------------------------------------------------------------------------------------------------- write guard
# E29b (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09, "E29b: step-type routing with a write guard"): a fixed,
# mechanical classifier of the actions an LLM response asks for. A response is READ-ONLY iff it has at least one tool
# call and every tool call is read-only. It is the calibration label (the strong LLM's next action) and the harness-side
# write guard (a fast-LLM response is executed only if it is read-only; otherwise the strong LLM writes the step).
# The registered allow-list: ls, cat, head, tail, less, more, grep (egrep, fgrep), rg, find (no -exec / -delete /
# -fprint...), wc, sed -n (print commands only, no -i), awk (no redirection, pipe, system or getline), git status / diff /
# log / show (no --output, no -c), pwd, tree (no -o), plus cd (navigation), nl (mini-SWE-agent's file viewer) and the
# pure filters sort (no -o), uniq (at most one operand), cut, tr, diff, file, stat, du, which; segments joined by &&,
# ||, ;, newlines or pipes between allowed commands; no redirection except to /dev/null and fd duplication (2>&1), no
# command or process substitution, no here-document, no background &, no subshell or variable assignment; and the
# harnesses' own viewer / search tools (pi read / grep / find / ls, SWE-agent str_replace_editor view and its search
# tools). Everything else (rm, mv, cp, edit, write, apply_patch, python, pytest, make, pip, npm, echo, submit, a response
# without a tool call, ...) is not read-only.
READ_ONLY_TOOLS = frozenset({"read", "ls", "grep", "find", "glob", "view", "open", "goto", "scroll_up", "scroll_down",
                             "find_file", "search_file", "search_dir"})
SHELL_TOOLS = frozenset({"bash", "shell", "execute_bash", "run_shell_command"})
_ALWAYS_RO = frozenset({"cd", "pwd", "ls", "cat", "head", "tail", "less", "more", "wc", "nl", "grep", "egrep", "fgrep",
                        "cut", "tr", "diff", "file", "stat", "du", "which"})
_FIND_WRITES = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"})
_GIT_RO = frozenset({"status", "diff", "log", "show"})
_SEPARATORS = frozenset({"", ";", "&&", "||", "|"})
_DEVNULL = re.compile(r"(?<![\w/&>])(?:[0-9]?>{1,2}|&>)\s*/dev/null(?![\w/.-])|(?<![\w>])[0-9]?>&[0-9](?![\w])")
_SED_ADDR = r"(?:\d+|\$|/(?:[^/\\]|\\.)*/|\+\d+|~\d+)"
_SED_PRINT = re.compile(rf"^\s*(?:{_SED_ADDR}(?:\s*,\s*{_SED_ADDR})?)?\s*[p=]\s*$")


def _shell_tokens(cmd: str) -> list[str] | None:
    lx = shlex.shlex(cmd, posix=True, punctuation_chars="();<>|&\n")
    lx.whitespace = " \t\r"
    lx.whitespace_split = True
    lx.commenters = ""
    try:
        return list(lx)
    except ValueError:            # unbalanced quotes
        return None


def _is_punct(tok: str) -> bool:
    return bool(tok) and all(ch in "();<>|&\n" for ch in tok)


def _sed_read_only(args: list[str]) -> bool:
    quiet, scripts, rest = False, [], []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-n", "--quiet", "--silent"):
            quiet = True
        elif a in ("-e", "--expression"):
            if i + 1 >= len(args):
                return False
            scripts.append(args[i + 1])
            i += 1
        elif a.startswith("--expression="):
            scripts.append(a.split("=", 1)[1])
        elif a.startswith("--"):
            if a.startswith("--in-place") or a.startswith("--file"):
                return False
            if a not in ("--posix", "--regexp-extended", "--null-data", "--separate", "--sandbox"):
                return False
        elif a.startswith("-") and len(a) > 1:
            flags = a[1:]
            if any(f in flags for f in "if"):          # in place / script file
                return False
            if "e" in flags:                            # -ne SCRIPT
                if not flags.endswith("e") or i + 1 >= len(args):
                    return False
                scripts.append(args[i + 1])
                i += 1
            if "n" in flags:
                quiet = True
            if set(flags) - set("nEersz"):
                return False
        else:
            rest.append(a)
        i += 1
    if not scripts:
        if not rest:
            return False
        scripts, rest = [rest[0]], rest[1:]
    if not quiet:
        return False
    for s in scripts:
        for part in s.replace("\n", ";").split(";"):
            if part.strip() and not _SED_PRINT.match(part):
                return False
    return True


def _awk_read_only(args: list[str]) -> bool:
    prog = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-F", "-v"):
            i += 2
            continue
        if a.startswith("-F") or a.startswith("-v"):
            i += 1
            continue
        if a.startswith("-"):
            return False                                # -f progfile, -i inplace, ...
        prog = a
        break
    if prog is None:
        return False
    return not any(x in prog for x in (">", "|", "system", "getline", "close(", "fflush"))


def _git_read_only(args: list[str]) -> bool:
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-C":
            i += 2
            continue
        if a in ("--no-pager", "-P", "--no-optional-locks"):
            i += 1
            continue
        break
    if i >= len(args) or args[i] not in _GIT_RO:
        return False
    return not any(a == "--output" or a.startswith("--output=") or a == "-c" for a in args[i + 1:])


def _segment_read_only(words: list[str]) -> bool:
    if not words:
        return False
    head = words[0]
    if "=" in head.split("/")[-1] and not head.startswith("-"):
        return False                                    # VAR=value prefix
    cmd, args = head.split("/")[-1], words[1:]
    if cmd in _ALWAYS_RO:
        return True
    if cmd == "rg":
        return not any(a == "--pre" or a.startswith("--pre=") for a in args)
    if cmd == "tree":
        return not any(a == "-o" or a.startswith("-o") and len(a) > 2 for a in args)
    if cmd == "find":
        return not any(a in _FIND_WRITES for a in args)
    if cmd == "sed":
        return _sed_read_only(args)
    if cmd in ("awk", "gawk", "mawk"):
        return _awk_read_only(args)
    if cmd == "git":
        return _git_read_only(args)
    if cmd == "sort":
        return not any(a in ("-o", "--output") or a.startswith("--output=") or (a.startswith("-o") and len(a) > 2)
                       for a in args)
    if cmd == "uniq":
        return sum(1 for a in args if not a.startswith("-")) <= 1
    return False


def read_only_command(cmd: Any) -> bool:
    """True iff a shell command only reads (the registered allow-list above). Conservative: anything it cannot parse
    is not read-only."""
    if not isinstance(cmd, str) or not cmd.strip():
        return False
    if "`" in cmd or "$(" in cmd or "<(" in cmd or ">(" in cmd:
        return False
    toks = _shell_tokens(_DEVNULL.sub(" ", cmd))
    if not toks:
        return False
    segments: list[list[str]] = [[]]
    for t in toks:
        if _is_punct(t):
            if t.replace("\n", "") not in _SEPARATORS:
                return False                            # redirection, background &, subshell, here-document
            segments.append([])
        else:
            segments[-1].append(t)
    segments = [s for s in segments if s]
    return bool(segments) and all(_segment_read_only(s) for s in segments)


def read_only_call(name: Any, args: Any) -> bool:
    """One tool call: a read-only harness tool, a shell command on the allow-list or SWE-agent's ``view``."""
    n = str(name or "")
    a = _parse_args(args)
    if n in READ_ONLY_TOOLS:
        return True
    if n == "str_replace_editor":
        return isinstance(a, Mapping) and str(a.get("command") or "") == "view"
    if n in SHELL_TOOLS:
        cmd = a.get("command", a.get("cmd")) if isinstance(a, Mapping) else a
        return read_only_command(cmd)
    return False


def read_only_calls(calls: list[tuple[str, Any]] | None) -> bool:
    """A response is read-only iff it asks for at least one tool call and every one is read-only."""
    return bool(calls) and all(read_only_call(n, a) for n, a in calls)


def gemini_calls(resp: Mapping[str, Any]) -> list[tuple[str, Any]]:
    cands = resp.get("candidates") or []
    parts = ((cands[0] or {}).get("content") or {}).get("parts") or [] if cands else []
    return [(str(p["functionCall"].get("name") or "?"), p["functionCall"].get("args")) for p in parts
            if isinstance(p, Mapping) and isinstance(p.get("functionCall"), Mapping)]


def openai_calls(resp: Mapping[str, Any]) -> list[tuple[str, Any]]:
    choices = resp.get("choices") or []
    msg = (choices[0] or {}).get("message") or {} if choices else {}
    return [(str(((tc or {}).get("function") or {}).get("name") or "?"), ((tc or {}).get("function") or {}).get(
        "arguments")) for tc in msg.get("tool_calls") or []]


# --------------------------------------------------------------------------------------------------- shared
@dataclass
class Parsed:
    """What the proxy keeps of one response."""
    key: str | None = None            # first action's key (None: unparsed)
    text: str = ""
    call: str = ""                    # first call rendered for the decision model
    usage: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = None
    calls: list[tuple[str, Any]] = field(default_factory=list)    # every tool call (E29b's write guard)

    @property
    def read_only(self) -> bool | None:
        """E29b: the response is read-only (None: unparsed)."""
        return None if self.key is None else read_only_calls(self.calls)


def parse_response(fmt: str, raw: bytes, streamed: bool) -> Parsed:
    """Parse a complete upstream response body (SSE for a streamed Gemini call). Never raises."""
    try:
        if fmt == FORMAT_GEMINI:
            resp = merge_gemini_chunks(parse_sse(raw)) if streamed else json.loads(raw or b"{}")
            key, text = gemini_first_action(resp)
            return Parsed(key, text, gemini_first_call(resp), gemini_usage(resp), resp.get("modelVersion"),
                          gemini_calls(resp))
        if streamed:   # OpenAI streaming is not used by the E29 harnesses; parse the chunks' deltas minimally
            return Parsed(None, "", "", {}, None)
        resp = json.loads(raw or b"{}")
        key, text = openai_first_action(resp)
        return Parsed(key, text, openai_first_call(resp), openai_usage(resp), resp.get("model"), openai_calls(resp))
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError, IndexError):
        return Parsed()


def step_index(fmt: str, body: Mapping[str, Any]) -> int:
    return gemini_step_index(body) if fmt == FORMAT_GEMINI else openai_step_index(body)


def episode(fmt: str, body: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
    return gemini_episode(body) if fmt == FORMAT_GEMINI else openai_episode(body)


def render_state(task: str, steps: list[dict[str, str]], action_in_progress: str, note: str = "",
                 history_window: int = HISTORY_WINDOW, max_chars: int = STATE_MAX_CHARS) -> dict[str, Any]:
    """The decide-ahead state (registration (b) 2): the task, the last ``history_window`` finished steps with their
    truncated outputs, the latest output in full (up to ``OBS_CHARS``), and the action now running (with the LLM's note
    that came with it). Oldest history goes first when the state is too long."""
    start = max(0, len(steps) - history_window)
    hist = [f"[{start + i + 1}] {trunc(s['action'], 300)} -> {trunc(s['output'], HIST_OBS_CHARS)}"
            for i, s in enumerate(steps[start:])]
    latest = steps[-1]["output"] if steps else "(no action has run yet)"

    def build(task_chars: int) -> dict[str, Any]:
        st: dict[str, Any] = {"task": trunc(task, task_chars), "history": list(hist),
                              "latest_output": trunc(latest, OBS_CHARS),
                              "action_in_progress": trunc(action_in_progress, ACTION_CHARS)}
        if note:
            st["assistant_note"] = trunc(note, ACTION_CHARS)
        return st

    state = build(TASK_CHARS)
    while len(hist) > 1 and len(json.dumps(state, ensure_ascii=False)) > max_chars:
        hist.pop(0)
        state = build(TASK_CHARS)
    if len(json.dumps(state, ensure_ascii=False)) > max_chars:
        state = build(1000)
    return state
