"""Static (single-decision) suites: public benchmarks converted into typed System-1 questions.

Each :class:`DecisionItem` is one decision with a gold answer, so System-1 backends and the
System-2 LLM's *native function calling* can be compared head-to-head on accuracy, calibration,
latency and cost -- the decision-level half of JevAgentBench.

* BFCL v4 (Berkeley Function Calling Leaderboard, pinned commit)
    - tool_select: Choice over the offered functions + "none"  (simple/multiple/live_*/irrelevance)
    - tool_params: per-argument Choice, conditional on the gold function
                   (enum/bool -> bounded values; string/number -> candidates copied from the query + OTHER)
* When2Call (NVIDIA, CC-BY-4.0): Choice {call tool, ask for info, cannot answer, answer directly}
    - the System-1 state lists every available tool with its arguments (type, required / optional), since
      "ask for info" items are tool_call items with a required argument's value removed from the question

The native-FC baseline (``bench/decision_eval.py``) gets the same information natively: BFCL's chat
messages (``meta["messages"]``) or the When2Call question (``meta["user_message"]``) plus the full tool
schemas. Argument items record ``omit_ok`` (BFCL accepts leaving the optional argument out).
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...decision.types import JEV_MAX_OPTIONS, Choice, Question
from ...harness.tools import OTHER, extract_candidates

DATA_DIR = Path("data/raw")
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_URL = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/{commit}/berkeley-function-call-leaderboard/"
            "bfcl_eval/data/{path}")
WHEN2CALL_COMMIT = "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc"
WHEN2CALL_URL = "https://raw.githubusercontent.com/NVIDIA/When2Call/{commit}/data/{path}"

NONE_TOOL = "none (no available function fits the request)"
BFCL_TOOL_CATS = ["simple_python", "multiple", "live_simple", "live_multiple", "irrelevance", "live_irrelevance"]


def _attach_gold(q: Question, gold: list[str]) -> Question:
    """Store the gold option (original casing) in ``meta`` for simulated/oracle backends only;
    real backends never see ``Question.meta``."""
    for o in q.options or []:
        if norm(o) in gold:
            q.meta["gold"] = o
            break
    return q


@dataclass
class DecisionItem:
    id: str
    source: str
    dp: str
    state: str
    question: Question
    gold: list[str]                       # acceptable answers (normalised strings)
    tools: list[dict[str, Any]] = field(default_factory=list)   # OpenAI-format tools (for the FC-LLM baseline)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def unit(self) -> str:
        """The benchmark row this decision comes from: a BFCL row's tool_select item and its tool_params
        items share a unit (``<row id>``), so samplers keep them together."""
        if "/param/" in self.id:
            return self.id.split("/param/", 1)[0]
        return self.id[: -len("/tool")] if self.id.endswith("/tool") else self.id


# keys of DecisionItem.meta that are inputs for the FC baseline (or large) and stay out of result records
META_PRIVATE = ("name_map", "messages", "user_message")


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - pinned public URLs
            dest.write_bytes(r.read())
    return dest


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(v: Any) -> str:
    """Normalise a value for matching. Strings follow BFCL's AST checker (`standardize_string`:
    lowercase, drop spaces and , . / - _ * ^) so "San Diego, CA"-style variants match the same way
    BFCL scores them."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).strip()
    try:  # numeric strings compare as numbers ("5.0" == 5)
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        pass
    return re.sub(r"[ ,./\-_*^]", "", s).lower()


# ------------------------------------------------------------------ BFCL
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _openai_schema(s: Any) -> Any:
    """BFCL uses python-ish type names; map them to JSON schema for native function calling."""
    if isinstance(s, dict):
        out = {k: _openai_schema(v) for k, v in s.items()}
        if isinstance(out.get("type"), str):
            out["type"] = _TYPE_MAP.get(out["type"], out["type"])
        return out
    if isinstance(s, list):
        return [_openai_schema(x) for x in s]
    return s


def bfcl_tools(funcs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for f in funcs:
        name = f["name"].replace(".", "_")  # OpenAI tool names disallow dots
        tools.append({"type": "function", "function": {
            "name": name, "description": f.get("description", ""),
            "parameters": _openai_schema(f.get("parameters", {"type": "object", "properties": {}}))}})
    return tools


def _query_text(q: list[list[dict[str, str]]]) -> str:
    turns = [m for turn in q for m in turn]
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in turns)


def _query_messages(q: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """BFCL turns as chat messages, for the native function-calling baseline (BFCL's own FC format)."""
    return [{"role": m["role"], "content": m["content"]} for turn in q for m in turn]


def load_bfcl(categories: list[str] | None = None, max_per_cat: int | None = None, seed: int = 0,
              with_params: bool = True, data_dir: Path = DATA_DIR) -> list[DecisionItem]:
    items: list[DecisionItem] = []
    for cat in categories or BFCL_TOOL_CATS:
        fn = f"BFCL_v4_{cat}.json"
        qpath = _download(BFCL_URL.format(commit=BFCL_COMMIT, path=fn), data_dir / "bfcl" / fn)
        rows = _jsonl(qpath)
        answers: dict[str, Any] = {}
        if "irrelevance" not in cat:
            apath = _download(BFCL_URL.format(commit=BFCL_COMMIT, path=f"possible_answer/{fn}"),
                              data_dir / "bfcl" / "possible_answer" / fn)
            answers = {r["id"]: r["ground_truth"] for r in _jsonl(apath)}
        if max_per_cat:
            rows = random.Random(seed).sample(rows, min(max_per_cat, len(rows)))
        for r in rows:
            items += bfcl_row_items(r, answers.get(r["id"]), cat, with_params=with_params)
    return items


def bfcl_row_items(r: dict[str, Any], gt: Any, cat: str, with_params: bool = True,
                   source: str | None = None) -> list[DecisionItem]:
    """E1's converter for ONE BFCL-format row ``{id, question: [[{role, content}]], function: [...]}`` with its
    ground truth ``gt`` (BFCL's ``possible_answer`` list; None for irrelevance): the tool_select item and, with
    ``with_params``, one tool_params item per scalar argument of the gold call. A category containing "irrelevance"
    makes "none" the gold tool. ``source`` defaults to ``bfcl:<cat>``; other public sets in the BFCL format (E12's
    training items) pass their own, so the items are built exactly as E1 builds its own."""
    items: list[DecisionItem] = []
    src = source or f"bfcl:{cat}"
    funcs = r["function"]
    state = _query_text(r["question"])
    names = [f["name"] for f in funcs]
    # rows with no functions (4 in live_irrelevance) would make a one-option Choice: not a decision
    if not names or len(names) + 1 > JEV_MAX_OPTIONS or len(set(names)) != len(names):
        return items
    if "irrelevance" in cat:
        gold_tool = NONE_TOOL
    elif gt and len(gt) == 1:
        gold_tool = next(iter(gt[0]))
    else:
        return items  # parallel calls are out of scope for single-decision items
    descs = [(f.get("description") or "")[:300] for f in funcs] + ["Use when no function applies."]
    q = Choice("Which function should be called to handle the user's request?", names + [NONE_TOOL],
               descs, decision_point="tool_select")
    tools = bfcl_tools(funcs)
    msgs = _query_messages(r["question"])
    items.append(DecisionItem(f"{r['id']}/tool", src, "tool_select", state,
                              _attach_gold(q, [norm(gold_tool)]), [norm(gold_tool)],
                              tools, {"category": cat, "n_tools": len(funcs), "messages": msgs,
                                      "name_map": {t["function"]["name"]: n for t, n in zip(tools, names)}}))
    if not with_params or gold_tool == NONE_TOOL:
        return items
    func = next(f for f in funcs if f["name"] == gold_tool)
    props = func.get("parameters", {}).get("properties", {})
    cands = extract_candidates(state)
    for pname, accept in gt[0][gold_tool].items():
        schema = props.get(pname, {})
        # BFCL lists "" among the possible answers of an optional argument whose omission is correct
        # (the function then uses its default); a native-FC call that leaves it out is scored correct
        omit_ok = "" in accept
        accept = [a for a in accept if a != ""]
        if not accept or any(isinstance(a, (dict, list)) for a in accept):
            continue
        ptype = schema.get("type", "string")
        if "enum" in schema:
            opts = [str(v) for v in schema["enum"]]
        elif ptype == "boolean":
            opts = ["true", "false"]
        elif ptype in ("string", "integer", "float", "number"):
            opts = [c for c in cands if c != OTHER][: 50] + [OTHER]
        else:
            continue
        opts = list(dict.fromkeys(opts))
        if len(opts) < 2:
            continue
        gold = [norm(a) for a in accept]
        in_opts = any(norm(o) in gold for o in opts)
        qp = Choice(f"The function `{gold_tool}` will be called. Which value should its argument `{pname}` "
                    f"take? ({schema.get('description', '')[:200]})", opts, decision_point="tool_params")
        pgold = gold if in_opts else [norm(OTHER)]
        items.append(DecisionItem(f"{r['id']}/param/{pname}", src, "tool_params", state,
                                  _attach_gold(qp, pgold), pgold, tools,
                                  {"category": cat, "param": pname, "param_type": ptype, "accept": gold,
                                   "gold_in_candidates": in_opts, "gold_tool": gold_tool,
                                   "omit_ok": omit_ok, "messages": msgs}))
    return items


# ------------------------------------------------------------------ When2Call
W2C_OPTIONS = {
    "tool_call": "call one of the available tools",
    "request_for_info": "ask the user for missing required information",
    "cannot_answer": "say it cannot help: no available tool fits and the question needs one",
    "direct": "answer directly without any tool",
}


W2C_MAX_PARAMS = 12  # per tool, in the rendered System-1 state


def w2c_tool_text(t: dict[str, Any]) -> str:
    """One available tool for the System-1 state: name, description and its arguments with types and
    whether each is required. Telling "call the tool" from "ask for missing information" needs to know
    which arguments are required (that is how When2Call builds its request_for_info items: a required
    argument's value is missing from the question), and the native-FC baseline sees the full schema too."""
    lines = [f"- {t.get('name')}: {(t.get('description') or '')[:200]}"]
    params = t.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    for i, (pname, ps) in enumerate(props.items()):
        if i == W2C_MAX_PARAMS:
            lines.append(f"    ({len(props) - W2C_MAX_PARAMS} more optional arguments)")
            break
        ps = ps if isinstance(ps, dict) else {}
        typ = str(ps.get("type", "any"))
        if ps.get("enum"):
            typ += " one of " + "|".join(str(v) for v in ps["enum"][:8])
        flag = "required" if pname in required else "optional"
        desc = str(ps.get("description") or "")[:100]
        lines.append(f"    * {pname} ({typ}, {flag})" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


def load_when2call(split: str = "test", max_items: int | None = None, seed: int = 0,
                   data_dir: Path = DATA_DIR) -> list[DecisionItem]:
    fn = "test/when2call_test_mcq.jsonl" if split == "test" else "train/when2call_train_sft.jsonl"
    path = _download(WHEN2CALL_URL.format(commit=WHEN2CALL_COMMIT, path=fn), data_dir / "when2call" / fn)
    rows = _jsonl(path)
    if max_items:
        rows = random.Random(seed).sample(rows, min(max_items, len(rows)))
    return [when2call_item(r["uuid"], r.get("tools") or [], r["question"], r["correct_answer"],
                           f"when2call:{r.get('source', '')}") for r in rows]


def when2call_item(uid: str, tools: list[Any], question: str, label: str, source: str,
                   id_prefix: str = "w2c/") -> DecisionItem:
    """E1's converter for ONE When2Call-format decision (tools as dicts or JSON strings with ``required`` inside
    ``parameters``, the user's question, the label key of ``W2C_OPTIONS``). Other public sets in this format (E12's
    training items) pass their own ``source`` and ``id_prefix``, so the items are built exactly as E1 builds its own."""
    labels = list(W2C_OPTIONS.values())
    tools = [json.loads(t) if isinstance(t, str) else t for t in (tools or [])]
    tool_lines = "\n".join(w2c_tool_text(t) for t in tools) or "(none)"
    state = f"AVAILABLE TOOLS:\n{tool_lines}\n\nUSER: {question}"
    q = Choice("How should the assistant respond to the user's message?", labels, decision_point="tool_select")
    gold = W2C_OPTIONS[label]
    return DecisionItem(f"{id_prefix}{uid}", source, "tool_select", state,
                        _attach_gold(q, [norm(gold)]), [norm(gold)], bfcl_tools(tools) if tools else [],
                        {"label": label, "n_tools": len(tools), "user_message": question})


def load_static(name: str, **kw: Any) -> list[DecisionItem]:
    if name == "bfcl":
        return load_bfcl(**kw)
    if name == "when2call":
        return load_when2call(**kw)
    raise ValueError(f"unknown static suite {name!r}")
