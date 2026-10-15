"""Tool-spec utilities: which parameters a System-1 model can fill, and candidate extraction.

System-1 models answer *bounded* questions. A tool parameter is S1-fillable when its value set
is bounded: ``enum`` strings, booleans, small integer ranges -- or when the value is a span that
already appears in the state (IDs, file paths, quoted strings, numbers). The latter is the
"copy, don't generate" hypothesis: most agent arguments are copied from context, so a Choice
over extracted candidates can replace generation. Free text (e.g. an email body, a code patch)
still goes to the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..decision.types import MAX_OPTIONS

OTHER = "<OTHER: none of these / needs generation>"

# Free-text parameter policy (docs/META_GOLD_SPEC.md §7): every parameter gets a frozen fill kind, from the JSON
# schema only (never from gold or outcomes), by the first rule that applies:
#   1. an annotation decides: the schema key ``x-jev-fill`` or a suite / config table ``{"tool.param": kind}``;
#   2. bounded -- enum, boolean, or an integer range of at most 20 values;
#   3. copy    -- a number / integer; a string with ``format`` / ``pattern``; a string whose name or description
#                 marks an identifier or a quantity taken from context (``COPY_NAME_RE`` / ``COPY_DESC_RE``);
#   4. free    -- every other string (and arrays / objects): the conservative default.
# System-1 fills only bounded and copy parameters; free ones always go to System-2.
FILL_KINDS = ("bounded", "copy", "free")
COPY_NAME_RE = re.compile(r"^(id|.*_id|.*_ids|email|phone|name|.*_name|path|url|uri|date|time|sku|code|.*_code|"
                          r"amount|units|count|quantity)$", re.I)
COPY_DESC_RE = re.compile(r"\b(id|identifier|e-?mail|phone|name|path|url|date)\b", re.I)


@dataclass
class ParamInfo:
    name: str
    schema: dict[str, Any]
    required: bool
    fill: str | None = None            # rule-1 override from a suite / config table (see :func:`params_of`)

    @property
    def kind(self) -> str:
        """The frozen fill kind: ``bounded`` | ``copy`` | ``free`` (module comment above)."""
        ann = self.fill or self.schema.get("x-jev-fill")
        if ann in FILL_KINDS:
            return str(ann)
        if self.is_bounded():
            return "bounded"
        t = self.type
        if t in ("number", "integer"):
            return "copy"
        if t == "string":
            if "format" in self.schema or "pattern" in self.schema:
                return "copy"
            if COPY_NAME_RE.match(self.name) or COPY_DESC_RE.search(str(self.schema.get("description") or "")):
                return "copy"
        return "free"

    @property
    def type(self) -> str:
        t = self.schema.get("type", "string")
        return t[0] if isinstance(t, list) else t

    def bounded_values(self, max_int_range: int = 20) -> list[Any] | None:
        if "enum" in self.schema:
            return list(self.schema["enum"])
        if self.type == "boolean":
            return [True, False]
        if self.type == "integer":
            lo, hi = self.schema.get("minimum"), self.schema.get("maximum")
            if lo is not None and hi is not None and 0 < hi - lo + 1 <= max_int_range:
                return list(range(int(lo), int(hi) + 1))
        return None

    def is_bounded(self) -> bool:
        return self.bounded_values() is not None

    def is_copyable(self) -> bool:
        """Legacy meaning (kept for ``bench/suites/tau2.py``, which applies its own free-text rule): any string /
        number that is not bounded. The agent harness uses :attr:`kind` instead."""
        return self.type in {"string", "integer", "number"} and not self.is_bounded()


def tool_name(tool: dict[str, Any]) -> str:
    return tool["function"]["name"]


def tool_desc(tool: dict[str, Any]) -> str:
    return (tool["function"].get("description") or "").strip().split("\n")[0][:200]


def params_of(tool: dict[str, Any], kinds: dict[str, str] | None = None) -> list[ParamInfo]:
    """The parameters of a tool spec. ``kinds`` maps ``"tool.param"`` (or ``"*.param"``) to a fill kind -- rule 1 of
    the free-text policy, set by a suite (``Env.param_kinds``) or an experiment (``tool_params.extra.param_kinds``)."""
    p = tool["function"].get("parameters") or {}
    req = set(p.get("required") or [])
    name = tool["function"].get("name", "")
    kinds = kinds or {}
    return [ParamInfo(n, s, n in req, kinds.get(f"{name}.{n}") or kinds.get(f"*.{n}"))
            for n, s in (p.get("properties") or {}).items()]


def param_kind_table(tools: list[dict[str, Any]], kinds: dict[str, str] | None = None) -> dict[str, str]:
    """``{"tool.param": kind}`` for every parameter of ``tools`` (the H3 composition / §7.2 regression table)."""
    return {f"{tool_name(t)}.{p.name}": p.kind for t in tools for p in params_of(t, kinds)}


def finish_tool(name: str = "finish") -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name,
        "description": "Call when the task is complete. Provide the final answer (or a short summary).",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}}


_PATTERNS = [
    r"\"([^\"\n]{1,80})\"",                       # double-quoted
    r"'([^'\n]{1,80})'",                          # single-quoted
    r"\b([A-Za-z][A-Za-z_]*[-_#]?\d[\w\-#]*)\b",   # IDs such as ORD-1234, user_42, #A12, raj_sanchez_7340
    r"((?:/|\./)[\w./\-]+)",                      # paths
    r"\b([\w.+-]+@[\w-]+\.[\w.]+)\b",             # emails
    r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])",      # numbers
    r"\b(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\b",   # dates / datetimes
    r"\b(\d[\w:/\-.]*[A-Za-z%][\w:/\-.%]*)",        # digit-led tokens: 12pm, 5kg, 3rd, 10%
    r"(?<=[a-z,] )([A-Z][a-zA-Z]{2,})\b",            # single capitalised word mid-sentence (Paris)
    r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b",          # multi-word proper nouns (person / place names)
]


# section headers of the harness's System-1 state (``JevAgent._s1_state``) that may follow its TASK section
STATE_SECTIONS = ("MEMORY:", "LATEST REFLECTION:", "HISTORY:", "CURRENT OBSERVATION:")


def task_section_end(text: str) -> int:
    """End offset of the TASK section of a harness state (``TASK: ...`` up to the next section header); 0 when
    the text is not a harness state."""
    if not text.startswith("TASK:"):
        return 0
    ends = [i for i in (text.find(f"\n\n{h}\n") for h in STATE_SECTIONS) if i > 0]
    return min(ends) if ends else len(text)


def extract_candidates(text: str, limit: int = MAX_OPTIONS - 1, task_first: int = 0) -> list[str]:
    """Extract copyable spans from the state, most recent first, de-duplicated.

    ``task_first`` > 0 keeps up to that many spans of the state's TASK section (the instruction: the values the
    user supplied) first, in order of appearance, and fills the remaining slots with the most recent spans.
    Without it a long, noisy history pushes the instruction's values out of the list: in the E2 dev runs (JevGym
    v2 ``multi``, 20 noise lines per observation) the customer names of the task ranked 51st-84th among the
    spans, so System-1 could not copy the next customer's name and replace / cascade looked up the wrong one."""
    found: list[tuple[int, str]] = []
    for pat in _PATTERNS:
        for m in re.finditer(pat, text):
            s = m.group(1).strip().rstrip(".,;:!?")
            if 0 < len(s) <= 80:
                found.append((m.start(), s))
            words = s.split(" ")
            if pat == _PATTERNS[-1] and len(words) > 2:  # "Ship Dana Kim" -> also "Dana Kim", "Ship Dana"
                for i in range(len(words)):
                    for j in range(i + 2, len(words) + 1):
                        if (i, j) != (0, len(words)):
                            found.append((m.start() + i, " ".join(words[i:j])))
    out: list[str] = []
    seen: set[str] = set()
    end = task_section_end(text) if task_first > 0 else 0
    if end:
        for _, s in sorted((x for x in found if x[0] < end), key=lambda x: (x[0], -len(x[1]))):  # longest first
            if len(out) >= min(task_first, limit):
                break
            if s not in seen:
                seen.add(s)
                out.append(s)
    found.sort(key=lambda x: -x[0])  # recency: later in the state = more recent
    for _, s in found:
        if len(out) >= limit:
            break
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def coerce(value: str, info: ParamInfo) -> Any:
    try:
        if info.type == "integer":
            return int(float(value))
        if info.type == "number":
            return float(value)
        if info.type == "boolean":
            return str(value).lower() in {"true", "yes", "1"}
    except ValueError:
        return value
    return value
