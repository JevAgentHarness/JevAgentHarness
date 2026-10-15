#!/usr/bin/env python3
"""Every prompt our code sends to a model in the paper's experiments, generated from the code (App. app:prompts-full).

    python scripts/paper_prompts.py
    # writes paper/sections/app_prompts.tex (one \\subsection, one shaded box per prompt) and results/prompts/prompts.md

Nothing is retyped where the code can supply it. Module constants and prompt functions are imported (or, for modules
whose import pulls in a heavy third-party package, read from the source with ``ast``: tau2.py, mini_swe.py); prompts
built inline inside a function are read from that function's source (``template``: the f-string's fixed parts verbatim,
each interpolated expression as a named placeholder); the pi extensions (TypeScript) are read with a small string-
expression reader. The decision-model inputs are recorded from a real, offline run of the reference loop
(``JevAgent``) on one DecisionGym task at DEVELOPMENT seed 600 with the registered evaluation settings, driven by a
scripted agent LLM that only reads what the episode shows and a recording decision model that answers uniformly; the
evaluation blocks (200000-200499, the retired 100000-100499, 300000-300999 and 310000-310599) are refused.

No API call, no GPU, no key. ``Question.meta`` (which holds gold labels) is never rendered: every request is built by
the code's own wire / prompt functions, which do not read it (Noul ``criteria`` aside, which are part of the question).

Placeholders print as <name> in italics (sentinels ``@@name@@`` here), shortenings as an italic [...] note
(``@@SKIP:...@@``). Long lines are wrapped for the two-column page (a grey hook marks a continuation line).
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from jevharness.bench import registry  # noqa: E402
from jevharness.bench.suites import jevgym_hard as JH  # noqa: E402
from jevharness.bench.suites.jevgym import ASK_POLICY, JevGymEnv  # noqa: E402
from jevharness.decision import intern_decision as ID  # noqa: E402
from jevharness.decision import llm_generative as LG  # noqa: E402
from jevharness.decision import llm_logprob as LP  # noqa: E402
from jevharness.decision.base import DecisionBackend  # noqa: E402
from jevharness.decision.jev import PRESETS, question_to_wire  # noqa: E402
from jevharness.decision.types import Choice, DecisionResult, Usage, answer_from_distribution  # noqa: E402
from jevharness.harness import agent as AG  # noqa: E402
from jevharness.harness import memory as MEM  # noqa: E402
from jevharness.harness import prompts as PR  # noqa: E402
from jevharness.harness.config import HarnessConfig  # noqa: E402
from jevharness.harness.tools import OTHER, finish_tool  # noqa: E402
from jevharness.llm.base import ChatResponse, ToolCall  # noqa: E402
from jevharness.llm.scripted import ScriptedLLM  # noqa: E402
from jevharness.training import data as TD  # noqa: E402

OUT_TEX = ROOT / "paper" / "sections" / "app_prompts.tex"
OUT_MD = ROOT / "results" / "prompts" / "prompts.md"
DEV_SEED = 600            # scripts/jevgym_examples.py: a clarify task (two pending orders)
HARD_SEED = 7003          # E7's development range (7000-7299)
EVAL = dict(families="v2", distractor_style="near", n_distractors=60, noise_lines=20, strict=True)
WIDTH = 60                # characters per line of a box (\scriptsize Inconsolata, 3.5 pt, in a 219 pt column)
N_OPTIONS_SHOWN = 2       # tool options shown in the decider examples (plus finish)

SRC = "src/jevharness/"
P_PROMPTS, P_AGENT, P_MEMORY = SRC + "harness/prompts.py", SRC + "harness/agent.py", SRC + "harness/memory.py"
P_TOOLS, P_CONFIG = SRC + "harness/tools.py", SRC + "harness/config.py"
P_JEV, P_INTERN = SRC + "decision/jev.py", SRC + "decision/intern_decision.py"
P_LOGPROB, P_GEN = SRC + "decision/llm_logprob.py", SRC + "decision/llm_generative.py"
P_EMB = SRC + "decision/embedding.py"
P_DATA, P_SERVE = SRC + "training/data.py", SRC + "training/serve_jev.py"
P_GYM, P_HARD = SRC + "bench/suites/jevgym.py", SRC + "bench/suites/jevgym_hard.py"
P_TAU2, P_ALF, P_STATIC = SRC + "bench/suites/tau2.py", SRC + "bench/suites/alfworld.py", SRC + "bench/suites/static.py"
P_DEVAL, P_SWE = SRC + "bench/decision_eval.py", SRC + "adapters/mini_swe.py"
P_PI = "adapters/pi/"
SWE_YAML = "adapters/mini_swe/swebench_jev.yaml"

# the prompts.py names the catalogue shows (tests/test_paper_prompts.py: every public name is covered)
PROMPTS_PY_COVERED: set[str] = set()
# mini_swe.py prompt constants of hooks no run used (the test checks every other one is shown)
SWE_NOT_RUN = {"Q_GUARD_CHOICE", "Q_STOP_TRIGGER", "Q_REFLECTION", "Q_ACTION", "MSG_REFLECTION", "MSG_STOP_HINT",
               "MSG_ACTION_HINT"}


def ph(name: str) -> str:
    """A placeholder sentinel; rendered as <name> in italics."""
    return f"@@{name}@@"


def skip(note: str) -> str:
    """A shortening marker; rendered as an italic [note]."""
    return f"@@SKIP:{note}@@"


def jph(name: str) -> str:
    """A placeholder for a value the code JSON-encodes (``json.dumps``): passed as this string, its JSON quotes are
    removed afterwards (:func:`unquote_json_ph`), so the box shows <name> where the JSON object goes."""
    return f"@@J:{name}@@"


def unquote_json_ph(s: str) -> str:
    return re.sub(r'"@@J:(.+?)@@"', r"@@\1@@", s)


def cover(*names: str) -> None:
    PROMPTS_PY_COVERED.update(names)


def fmt_fields(text: str, names: dict[str, str] | None = None) -> str:
    """A ``str.format`` template (``{cmd}``, ``{p:.2f}``) with every field as a placeholder."""
    out = re.sub(r"(?<!\{)\{(\w+)(?::[^{}]*)?\}(?!\})", lambda m: ph((names or {}).get(m.group(1), m.group(1))), text)
    return out.replace("{{", "{").replace("}}", "}")


# --------------------------------------------------------------------------------------------- reading code
def _parse(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text(), filename=path)


def module_constants(path: str) -> dict[str, Any]:
    """Module-level ``NAME = <str | dict | list expression>`` assignments of a Python file, evaluated from the source
    (string concatenation, f-strings over earlier constants, literals), without importing the module."""
    out: dict[str, Any] = {}
    for node in _parse(path).body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        try:
            v = _eval_const(value, out)
        except (ValueError, TypeError, KeyError):
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out[t.id] = v
    return out


def _eval_const(node: ast.expr, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                parts.append(str(_eval_const(v.value, env)))
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_const(node.left, env) + _eval_const(node.right, env)
    if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
        try:
            return ast.literal_eval(node)
        except ValueError:
            pass
        if isinstance(node, ast.Dict):
            return {_eval_const(k, env): _eval_const(v, env) for k, v in zip(node.keys, node.values)}  # type: ignore
        return [_eval_const(e, env) for e in node.elts]
    raise ValueError(ast.dump(node)[:80])


def _find_function(tree: ast.AST, name: str) -> ast.AST:
    hits = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(hits) != 1:
        raise LookupError(f"function {name!r}: {len(hits)} definitions")
    return hits[0]


def _named(src: str, names: dict[str, str]) -> str | None:
    """``names[src]`` as display text: ``=text`` is literal text, an ``@@...@@`` sentinel or "" is kept as is, any
    other value becomes a placeholder; None when ``src`` is not named."""
    if src not in names:
        return None
    v = names[src]
    if v.startswith("="):
        return v[1:]
    return v if v.startswith("@@") or v == "" else ph(v)


def _render(node: ast.expr, names: dict[str, str], consts: dict[str, Any]) -> str | None:
    """A string expression with its interpolations as placeholders; None when ``node`` is not a string expression."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
                continue
            assert isinstance(v, ast.FormattedValue)
            src = ast.unparse(v.value)
            named = _named(src, names)
            if named is not None:
                out.append(named)
            elif isinstance(v.value, ast.Name) and v.value.id in consts:
                out.append(str(consts[v.value.id]))
            else:
                out.append(ph(src))
        return "".join(out)
    named = _named(ast.unparse(node), names)
    if named is not None:                      # a named call / attribute / name (e.g. ``self.answer_text()``)
        return named
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _render(node.left, names, consts), _render(node.right, names, consts)
        return None if a is None or b is None else a + b
    if isinstance(node, ast.IfExp):            # the variant the experiments asked (the body); notes name the other
        return _render(node.body, names, consts)
    if isinstance(node, ast.Name):
        if node.id in consts and isinstance(consts[node.id], str):
            return consts[node.id]
        return ph(node.id)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):   # `override or (default ...)`: the default
        return _render(node.values[-1], names, consts)
    return None


def template(path: str, func: str, anchor: str, names: dict[str, str] | None = None,
             consts: dict[str, Any] | None = None, without: str | None = None) -> str:
    """The longest string expression inside ``func`` of ``path`` whose rendering contains ``anchor``: the prompt as
    the code builds it, interpolations shown as placeholders (``names`` maps an expression's source to a name);
    renderings that contain ``without`` are skipped."""
    fn = _find_function(_parse(path), func)
    best = None
    for node in ast.walk(fn):
        if isinstance(node, (ast.JoinedStr, ast.Constant, ast.BinOp, ast.IfExp, ast.BoolOp)):
            s = _render(node, names or {}, consts or {})
            if s is not None and anchor in s and (without is None or without not in s) and (
                    best is None or len(s) > len(best)):
                best = s
    if best is None:
        raise LookupError(f"{path}:{func}: no string expression contains {anchor!r}")
    return best


# --- TypeScript (pi extensions): string expressions "..." / '...' / `...${x}...` joined by +
_TS_ESC = {"n": "\n", "t": "\t", '"': '"', "'": "'", "`": "`", "\\": "\\", "$": "$"}


def _ts_string_expr(src: str, i: int, names: dict[str, str]) -> tuple[str | None, int]:
    parts: list[str] = []
    while True:
        while i < len(src) and src[i] in " \t\r\n":
            i += 1
        if i >= len(src) or src[i] not in "\"'`":
            break
        q = src[i]
        i += 1
        buf = []
        while src[i] != q:
            c = src[i]
            if c == "\\":
                buf.append(_TS_ESC.get(src[i + 1], src[i + 1]))
                i += 2
                continue
            if q == "`" and src.startswith("${", i):
                j = src.index("}", i)
                expr = src[i + 2:j].strip()
                buf.append(ph(names.get(expr, expr)))
                i = j + 1
                continue
            buf.append(c)
            i += 1
        parts.append("".join(buf))
        i += 1
        j = i
        while j < len(src) and src[j] in " \t\r\n":
            j += 1
        if j < len(src) and src[j] == "+":
            i = j + 1
            continue
        break
    return ("".join(parts) if parts else None), i


def ts_string(path: str, key: str, anchor: str = "", names: dict[str, str] | None = None) -> str:
    """The string expression after ``key`` (e.g. ``const DEFAULT_NUDGE =`` or ``instructions:``) in a TypeScript
    file whose text contains ``anchor``."""
    src = (ROOT / path).read_text()
    for m in re.finditer(re.escape(key), src):
        s, _ = _ts_string_expr(src, m.end(), names or {})
        if s is not None and anchor in s:
            return s
    raise LookupError(f"{path}: no string after {key!r} contains {anchor!r}")


def ts_criteria(path: str, after: str) -> dict[str, str]:
    """``criteria: { true: "...", false: "..." }`` (or a named criteria object) following the text ``after``."""
    src = (ROOT / path).read_text()
    i = src.index(after)
    m = re.compile(r"(?:criteria|CHOICE_CRITERIA[^=]*=)\s*:?\s*\{").search(src, i)
    assert m is not None
    j, out = m.end(), {}
    while True:
        km = re.compile(r"\s*(\w+)\s*:").match(src, j)
        if not km:
            break
        s, j = _ts_string_expr(src, km.end(), {})
        out[km.group(1)] = s or ""
        j = re.compile(r"\s*,?").match(src, j).end()   # type: ignore[union-attr]
    return out


# --------------------------------------------------------------------------------------------- recorded episode
class Recorder(DecisionBackend):
    """Records every (state, questions) request the harness sends and answers each question uniformly (so every
    cascade defers and the scripted LLM decides). Wire-like: one request per question, up to 255 options."""

    name = "recorder"
    batched_questions = False
    max_options = 255

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any]]] = []

    def _decide(self, state: str, questions: list[Any]) -> DecisionResult:
        self.calls.append((state, [q.model_copy(deep=True) for q in questions]))
        return DecisionResult(answers=[answer_from_distribution(q, {lab: 1.0 for lab in q.labels()})
                                       for q in questions], usage=Usage(backend=self.name))


def refuse_reserved(seed: int) -> None:
    for ranges in (registry.EVAL_SEED_RANGES, registry.RETIRED_EVAL_RANGES, registry.STAGE2_RESERVED_RANGES,
                   registry.EXPLORE_EVAL_RANGES):
        if registry.in_ranges(seed, ranges):
            raise SystemExit(f"seed {seed} is in a reserved evaluation block {ranges}; use a development seed")


@dataclass
class Episode:
    env: JevGymEnv
    s1: list[tuple[str, list[Any]]]
    llm: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]


def recorded_episode(seed: int = DEV_SEED) -> Episode:
    """Three steps of the reference loop on a DecisionGym task, every decision point that has a shadow mode asked in
    shadow (the episode is the LLM-only one; the decision model is asked what a live arm asks, at the same state).
    The scripted agent LLM looks the customer up by the name in the request, lists the pending orders of the user id
    it read, and asks which order is meant (the policy of the first observation)."""
    refuse_reserved(seed)
    env = JevGymEnv(seed=seed, **EVAL)
    llm_calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def policy(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResponse:
        llm_calls.append((json.loads(json.dumps(messages, default=str)), list(tools or [])))
        n = sum(1 for m in messages if m["role"] == "tool")
        if n == 0:
            name = re.search(r"customer (.+?) asked", messages[1]["content"]).group(1)   # type: ignore[union-attr]
            return ChatResponse(text="", tool_calls=[ToolCall(id="c0", name="find_user", arguments={"name": name})])
        if n == 1:
            uid = json.loads(messages[-1]["content"].split("\n")[0])["user_id"]
            return ChatResponse(text="", tool_calls=[ToolCall(id="c1", name="get_orders",
                                                              arguments={"user_id": uid, "status": "pending"})])
        return ChatResponse(text="", tool_calls=[ToolCall(id="c2", name="ask_user", arguments={
            "missing": "order_id", "question": "Which of your pending orders should I cancel?"})])

    shadow = {"mode": "shadow"}
    cfg = HarnessConfig(name="prompt_catalogue", max_steps=3, tool_select=shadow, tool_params=shadow, stop=shadow,
                        reflection=shadow, routing=shadow, memory_write=shadow)
    rec = Recorder()
    traj = AG.JevAgent(ScriptedLLM(policy=policy), cfg, s1=rec).run(env)
    if traj.error:
        raise RuntimeError(traj.error)
    return Episode(env, rec.calls, llm_calls)


def asked(ep: Episode, dp: str, k: int = 0, contains: str = "") -> tuple[str, Any]:
    """The k-th recorded (state, question) at decision point ``dp`` (whose text contains ``contains``)."""
    hits = [(st, q) for st, qs in ep.s1 for q in qs if q.decision_point == dp and contains in q.text]
    return hits[k]


_LOG_LINE = re.compile(r"^(log: heartbeat ok #\d+|.*heartb\w*|\w*beat ok #\d+)$")


def shorten_state(state: str) -> str:
    """Collapse runs of the log-noise lines (kept: the first and the last of a run) for display."""
    lines, out, run = state.split("\n"), [], []

    def flush() -> None:
        if len(run) >= 4:
            out.extend([run[0], skip(f"{len(run) - 2} more log lines"), run[-1]])
        else:
            out.extend(run)
        run.clear()

    for ln in lines:
        if _LOG_LINE.match(ln):
            run.append(ln)
            continue
        flush()
        out.append(ln)
    flush()
    return "\n".join(out)


def subset_choice(q: Any, n: int = N_OPTIONS_SHOWN) -> Any:
    """The same Choice with its first ``n`` options and the last one (finish), for a short example."""
    keep = list(range(n)) + [len(q.options) - 1]
    descs = [q.option_descriptions[i] for i in keep] if q.option_descriptions else None
    return Choice(q.text, [q.options[i] for i in keep], descs, decision_point=q.decision_point)


# --------------------------------------------------------------------------------------------- catalogue
@dataclass
class Box:
    key: str
    title: str             # light markup: {gym} {jev} {ours} {tau2} `code`
    sources: list[str]     # repo paths, optionally "path: symbol"
    text: str              # the prompt, with @@placeholders@@ / @@SKIP:...@@
    note: str = ""         # one or two plain sentences (same markup as the title)


@dataclass
class Group:
    title: str
    intro: str
    boxes: list[Box]


def catalogue() -> tuple[str, list[Group]]:
    ep = recorded_episode()
    env = ep.env
    cfg = HarnessConfig()
    groups: list[Group] = []

    # ------------------------------------------------------------------ decision-model inputs (reference loop)
    st_tool, q_tool = asked(ep, "tool_select", 2)          # step 2: after find_user and get_orders
    state_tpl = "\n\n".join([
        template(P_AGENT, "_s1_state", "TASK: ", {"env.instruction": "task instruction"}),
        template(P_AGENT, "_s1_state", "MEMORY:", {"memory_text": "recalled memories, one per line"}),
        template(P_AGENT, "_s1_state", "LATEST REFLECTION:", {"reflection": "the agent LLM's latest reflection"}),
        "HISTORY:\n" + template(P_AGENT, "_s1_state", "] ", {"start + i": "step", "a": "tool(arguments)",
                                                              "_trunc(o, 400)": "result"}),
        template(P_AGENT, "_s1_state", "CURRENT OBSERVATION:", {"_trunc(obs, c.obs_truncate_chars)":
                                                                "latest tool result"})])
    probe = AG.JevAgent._s1_state(type("C", (), {"cfg": cfg})(), env, [("a", "x" * 5000)], "y" * 5000)
    hist_cap = len(probe.split("HISTORY:\n[0] a -> ", 1)[1].split("\n\nCURRENT OBSERVATION:")[0]) - len(
        "\n...[truncated]...\n")
    caps = (f"MEMORY and LATEST REFLECTION appear only when the arm has them. The history keeps the last "
            f"{cfg.history_window} steps and cuts each result to {hist_cap} characters. The current observation is "
            f"cut to {cfg.obs_truncate_chars:,} characters and the whole state to {cfg.s1_state_max_chars:,}. A cut "
            f"keeps the head and the tail around `...[truncated]...`.")
    q_param = asked(ep, "tool_params", 0, "`name`")[1]
    q_enum = asked(ep, "tool_params", 0, "status")[1]
    st_mem, q_mem = asked(ep, "memory_write", 1)
    store = MEM.MemoryStore(cfg.memory_write, cfg.memory_read)
    mem_q = store._question([MEM.MemoryItem(id=ph("id"), text=ph("memory text"))])     # type: ignore[arg-type]
    mem_label = store.label(MEM.MemoryItem(id=ph("id"), text=ph("its first 60 characters")))  # type: ignore[arg-type]
    mem_cand = st_mem.split("\n\nCANDIDATE MEMORY:\n", 1)
    state_box = Box(
        "state", "Decision-model state (reference loop)", [P_AGENT + ": JevAgent._s1_state", P_CONFIG],
        state_tpl, caps)
    example_box = Box(
        "state_example", "State at development seed " + str(DEV_SEED) + ", step 2", [P_AGENT, P_GYM],
        shorten_state(st_tool),
        "A {gym} clarify task in the evaluation settings (60 look-alike tools, 20 log lines per lookup), after the "
        "agent looked the customer up and listed the pending orders. Our shortening of the log lines is marked "
        "in italics. The model sees all of them.")
    tool_opts = [f"{o} -- {d}" if d else o for o, d in zip(q_tool.options, q_tool.option_descriptions or [])]
    n_tools = len(q_tool.options)
    q_lines = [
        f"[Tool choice: one of {n_tools} options]", q_tool.text,
        *tool_opts[:3], skip(f"{n_tools - 4} more tools"), tool_opts[-1], "",
        "[Tool arguments: one question per argument]",
        AG.JevAgent._param_question({"type": "function", "function": {"name": ph("tool")}},
                                    _PInfo(ph("argument"), ph("description in the schema")), ["x", "y"], None).text,
        "Example: " + q_param.text, "Options: the " + str(len(q_param.options) - 1) + " values copied from the "
        "state (" + ", ".join(q_param.options[:-1]) + "), then", OTHER,
        "Example: " + q_enum.text, "Options: the enum " + ", ".join(q_enum.options), "",
        "[Tool guard: yes/no]", unquote_json_ph(PR.q_tool_guard(ph("tool"), jph("arguments as JSON"))),
    ]
    cover("Q_TOOL_SELECT", "q_tool_guard")
    box_tool_q = Box("questions_tools", "Questions at tool calls", [P_PROMPTS, P_AGENT, P_TOOLS],
                     "\n".join(q_lines),
                     "Options carry the first line of each tool description. The decision model fills an argument "
                     "only when its values are bounded, as with an enum or a boolean, or can be copied from the "
                     "state. The copy options are the spans the code extracts. Free text always goes to the LLM.")
    yn = [
        "[Memory write: the state ends with the candidate]",
        template(P_MEMORY, "offer", "CANDIDATE MEMORY", {"context": "", "text": "candidate"}).strip(),
        "  candidate: " + template(P_AGENT, "run", ") -> ", {"name": "tool", "json.dumps(args)[:200]": "arguments",
                                                              "_trunc(res_obs, 600)": "result"}),
        q_mem.text, "",
        "[Memory read: one option per stored memory]", mem_q.text,
        f"{mem_label} -- {mem_q.option_descriptions[0]}", *mem_q.options[1:], "",
        "[Memory read trigger: the state ends with]",
        template(P_MEMORY, "recall", "STORED MEMORIES:", {"query": "", "len(self.items)": "n"}).strip(),
        PR.Q_MEMORY_READ_TRIGGER, "",
        "[Memory read guard: the state ends with the selection]",
        template(P_MEMORY, "_guarded_recall", "SELECTED MEMORIES:").strip() + "\n"
        + template(P_MEMORY, "_guarded_recall", "- [m", {"m.id": "id", "m.text[:600]": "memory text"}),
        PR.q_memory_read_guard([ph("label"), ph("label")]), "",
        "[Reflection]", PR.Q_REFLECTION, "Earlier wording (measurement arms):", PR.Q_REFLECTION_V1, "",
        "[Stop guard, when the LLM calls finish]", unquote_json_ph(PR.q_stop_guard(jph("answer, JSON-quoted"))),
        "[Stop trigger, at every other step]", PR.Q_STOP_TRIGGER, "",
        "[Routing]", PR.Q_ROUTING, "",
        "[Action-only environments: choice and guard]",
        template(P_AGENT, "_action_step", "Which action"), "Options: the admissible actions",
        PR.q_action_guard(ph("action")), "",
        "[Verification triggers: does the LLM's proposal need a second look?]",
        unquote_json_ph(PR.q_verify_tool(ph("tool"), jph("arguments"))),
        unquote_json_ph(PR.q_verify_args(ph("tool"), jph("arguments"))),
        PR.q_verify_action(ph("action")),
    ]
    cover("Q_MEMORY_WRITE", "Q_MEMORY_READ_TRIGGER", "q_memory_read_guard", "Q_REFLECTION", "Q_REFLECTION_V1",
          "REFLECTION_WORDINGS", "q_stop_guard", "Q_STOP_TRIGGER", "Q_ROUTING", "q_action_guard", "clip_answer",
          "ANSWER_CHARS", "ANSWER_HEAD_CHARS", "q_verify_tool", "q_verify_args", "q_verify_action")
    assert q_mem.text == PR.Q_MEMORY_WRITE and mem_cand[0] == asked(ep, "tool_select", 1)[0]
    box_yn = Box("questions_yesno", "Questions at the other decision points", [P_PROMPTS, P_MEMORY, P_AGENT],
                 "\n".join(yn),
                 f"The stop guard sees the whole proposed answer when it has at most {PR.ANSWER_CHARS} characters, "
                 f"and otherwise its first {PR.ANSWER_HEAD_CHARS} and last {PR.ANSWER_CHARS - PR.ANSWER_HEAD_CHARS}. "
                 "A memory option is labelled with the first 60 characters of the memory, and the full text is its "
                 "description.")
    wire_q = question_to_wire(subset_choice(q_tool))
    stop_q = question_to_wire(asked(ep, "stop", 2)[1])
    body = {"state": skip("the state shown above"), "model": PRESETS["openrouter"]["model"],
            "questions": {"q0": wire_q, "q1": stop_q}}
    wire_box = Box("wire", "{jev} request (wire format)", [P_JEV + ": question_to_wire"], json.dumps(body, indent=1),
                   f"The example keeps {N_OPTIONS_SHOWN} of the {n_tools} tool options and finish. In the reference "
                   "loop each decision point is one request. A Noul (yes/no) question carries `criteria` only when "
                   "the harness gives them, as mini-SWE-agent and pi do. {jev}, OpenJev, StartLux-Decision and our "
                   "trained models all receive this request.")
    groups.append(Group(
        "Decision-model inputs", "Every decision model reads the same state text and one typed question, a choice "
        "among options or a yes/no question. The reference loop builds the state below. The questions come next, "
        "then the request that carries them.",
        [state_box, example_box, box_tool_q, box_yn, wire_box]))

    # ------------------------------------------------------------------ how each decision model reads it
    q_small = subset_choice(q_tool)
    q_stop = asked(ep, "stop", 2)[1]
    st = ph("state")
    comp, _ = ID.compile_request(st, [q_small, q_stop], ["tool_select", "stop"])
    intern_text = "\n".join(f"[{m['role']}]\n{m['content']}" for m in comp.messages)
    intern_box = Box("intern", "Intern-Decision (one pass for all questions)", [P_INTERN + ": compile_row"],
                     intern_text,
                     "The vendor's prompt compiler, copied with its licence. The state is JSON-encoded. Each field's "
                     "answer is read at its `<decision>` placeholder, so k questions cost one forward pass.")
    from jevharness.training.serve_jev import Engine   # noqa: E402  (imports torch)

    def served(q: Any) -> str:
        kind, instr, opts, _ = Engine._plan(question_to_wire(q))
        return TD.render({"state": st, "question": instr, "options": opts})[0]

    trained_text = "\n".join(["[system]", TD.SYSTEM, "[user]", served(q_small), "", "[user, a yes/no question]",
                              served(q_stop)])
    trained_box = Box("trained", "{ours} and the other trained models", [P_DATA + ": SYSTEM, render", P_SERVE],
                      trained_text,
                      "Each question gets its own prompt in the checkpoint's chat template, with thinking off and the "
                      "assistant turn opened. The answer is read from the next-token logits of the option codes. "
                      "Training rows use this exact format.")
    lp_prompt, _ = LP.build_prompt(st, q_small)
    top_lp = inspect.signature(LP.LogprobLLMBackend.__init__).parameters["top_logprobs"].default
    lp_noul, _ = LP.build_prompt(st, q_stop)
    logprob_box = Box("logprob", "LLM decider read through log-probabilities", [P_LOGPROB],
                      "\n".join(["[system]", LP.SYSTEM_PROMPT, "[user]", lp_prompt, "", "[user, a yes/no question]",
                                 lp_noul]),
                      f"The model generates one token at temperature 0 and returns the top {top_lp} "
                      "log-probabilities. The probabilities of tokens that equal an option label are summed per "
                      "label.")
    gen_dist = LG.build_prompt(st, [("Q1", q_small), ("Q2", q_stop)], "distribution")
    gen_label = LG.build_prompt(st, [("Q1", q_small)], "label").rsplit("\n\n", 1)[1]
    gen_box = Box("verbalised", "LLM decider with verbalised probabilities", [P_GEN],
                  "\n".join(["[system]", LG.SYSTEM, "[user]", gen_dist, "", "[last block in the label variant]",
                             gen_label]),
                  "The Gemini stand-in and the control in which the agent LLM answers its own questions use this "
                  "prompt. All questions about one state go in one call.")
    emb_q = template(P_EMB, "_decide", "\n", {"state": "state", "q.text": "question"})
    emb_o = template(P_EMB, "_decide", ": ", {"o": "option", "q.option_descriptions[i]": "description"})
    emb_box = Box("embedding", "Embedding baseline (L1 only)", [P_EMB],
                  f"[query]\n{emb_q}\n[one text per option]\n{emb_o}",
                  "The score of an option is the cosine similarity of the two embeddings.")
    groups.append(Group(
        "How each decision model reads a question", "{jev}, OpenJev and StartLux-Decision receive the request above "
        "and build their prompts inside their own servers {cite:typesafe2026jev,openjev2026,startlux2026decision}. Every other "
        "decision model gets one of the prompts below, where <state> is the state text shown above.",
        [intern_box, trained_box, logprob_box, gen_box, emb_box]))

    # ------------------------------------------------------------------ the agent LLM on DecisionGym
    first = ep.llm[0][0]
    sys_text = first[0]["content"]
    assert sys_text == AG.DEFAULT_SYSTEM
    fin = finish_tool(cfg.final_tool)["function"]
    first_user = template(P_AGENT, "run", "\n\n", {"env.instruction": "task instruction",
                                                     "obs": "first observation"})
    obs0 = env.reset()
    assert obs0.endswith(ASK_POLICY.strip()) and first[1]["content"] == f"{env.instruction}\n\n{obs0}"
    sys_box = Box("agent_system", "Agent LLM: system prompt and first message on {gym}", [P_AGENT, P_GYM, P_TOOLS],
                  "\n".join(["[system]", sys_text, "[user]", first_user, "", "[first observation]", obs0, "",
                             f"[tool `{fin['name']}`, offered with the task's tools]", fin["description"]]),
                  f"The task's tools ({len(env.tools())} at these settings, shuffled per task) are passed as "
                  "function declarations, and the LLM sees every tool result as a tool message.")
    names = {"nm": "customer", "self.target_item": "item", "steps": "numbered requests", "speed": "shipping rule"}
    instr = []
    for fam, anchor in (("cancel", "asked us to cancel"), ("refund", "as store credit"), ("stock", "units of"),
                        ("ship", "Use express shipping if"), ("total", "total amount of all")):
        instr.append(f"[{fam}] " + template(P_GYM, "_build", anchor, names))
    for var, anchor in (("order", "asked us to cancel"), ("method", "(highest order id)."),
                        ("speed", "Ship @@customer@@'s pending order."), ("name", "did not say who")):
        instr.append(f"[clarify: {var} missing] " + template(P_GYM, "_build_clarify", anchor, names))
    instr += ["[multi] " + template(P_GYM, "_build_multi", "Process these requests", names),
              "  numbered requests, joined by \"; \": " + template(P_GYM, "_build_multi", "@@request@@",
                                                                {"i + 1": "i", "p": "request"}),
              "  request: " + template(P_GYM, "_build_multi", "(the customer requested it)",
                                        {"u['name']": "customer"}),
              "       or: " + template(P_GYM, "_build_multi", "ship @@customer@@'s pending order",
                                        {"u['name']": "customer"}),
              "  shipping rule (if any request ships):" + template(P_GYM, "_build_multi", "Use express shipping for"),
              "", "[customer replies to ask_user]"]
    rnames = {"self.pending_order['order_id']": "order id", "self.pending_order['item']": "item",
              "self.ship_speed.capitalize()": "Speed", "nm": "customer"}
    instr += ["Customer: " + template(P_GYM, "_build_clarify", "Sorry for the confusion", rnames),
              "Customer: " + template(P_GYM, "_build_clarify", "store credit."),
              "Customer: " + template(P_GYM, "_build_clarify", "original payment method."),
              "Customer: " + template(P_GYM, "_build_clarify", "shipping, please.", rnames),
              "Customer: " + template(P_GYM, "_build_clarify", "my name is", rnames),
              template(P_GYM, "_ask", "already gave you"), template(P_GYM, "_ask", "nothing to add")]
    tasks_box = Box("gym_tasks", "{gym} task instructions", [P_GYM + ": JevGymEnv._build*, _ask"],
                    "\n".join(instr),
                    "The first five replies give a clarify task's missing detail. A question about another detail "
                    "gets the sixth reply in a clarify task and the seventh in every other task.")
    hard = JH.JevGymHardEnv(seed=HARD_SEED, families="multi", multi_customers=3, multi_rule="tier")
    refuse_reserved(HARD_SEED)
    hard_items = JH.JevGymHardEnv(seed=HARD_SEED, families="multi", multi_customers=3, multi_rule="tier",
                                  answer_style="itemized")
    hard_tpl = template(P_HARD, "_build_multi", "Handle the pending order", {
        "names": "=(1) " + ph("customer 1") + "; (2) " + ph("customer 2") + "; " + skip("one per customer"),
        "self.policy_text()": "rule", "self.answer_text()": "answer sentence"})
    hard_box = Box("gym_hard", "{gym}-Hard instruction", [P_HARD + ": _build_multi, policy_text, answer_text"],
                   "\n".join([hard_tpl, "", "[rule]", hard.policy_text(), "", "[answer sentence: total, first runs]",
                              hard.answer_text(), "[answer sentence: itemized, from E7e]", hard_items.answer_text()]),
                   "The runs used ten or twelve customers per task. Each customer's tier and amount appear only in "
                   "the lookup results. The code also has a rule on the amount alone, which no run used.")
    msg = [
        "[Reflection: asked as a user turn, the reply kept]", PR.REFLECT_PROMPT,
        template(P_AGENT, "_reflect", "[Reflection]", {"r.text": "the LLM's reply"}), "",
        "[Recalled memories, before the LLM's turn]",
        template(P_AGENT, "run", "[Memory]", {"mem_text": "=" + MEM.MemoryStore.render(
            None, [MEM.MemoryItem(id=ph("id"), text=ph("memory text"))] * 2)}), "",   # type: ignore[arg-type]
        "[Stop guard rejected a finish]", PR.STOP_REJECT_NOTE,
        "[Stop trigger fired: a forced finish call]", PR.EARLY_STOP_PROMPT, "",
        "[Tool guard rejected the call]", unquote_json_ph(PR.tool_reject_note(ph("tool"), jph("arguments"))),
        "[Argument guard rejected one argument]",
        unquote_json_ph(PR.arg_reject_note(ph("tool"), jph("arguments"), ph("argument"))), "",
        "[Verification pass at tool choice: every tool offered]",
        unquote_json_ph(PR.verify_tool_note(ph("tool"), jph("arguments"))),
        "[Verification pass at tool arguments: the call forced to the proposed tool]",
        unquote_json_ph(PR.verify_args_note(ph("tool"), jph("arguments"))), "",
        "[Decision model chose the tool and the LLM fills the rest]",
        template(P_AGENT, "_delegated_call", "You must now call", {"name": "tool",
                                                                   "json.dumps(filled)": "the model's arguments"}),
        "[Argument assist: the model's shortlists]",
        unquote_json_ph(PR.assist_args_hint(ph("tool"), [(ph("argument"), [jph("value"), jph("value")], False),
                                                         (ph("argument"), [jph("value")], True)],
                                            {ph("argument"): jph("value")})),
    ]
    cover("REFLECT_PROMPT", "STOP_REJECT_NOTE", "EARLY_STOP_PROMPT", "tool_reject_note", "arg_reject_note",
          "assist_args_hint", "append_note", "verify_tool_note", "verify_args_note")
    msg_box = Box("agent_notes", "Messages the harness adds to the LLM's context", [P_PROMPTS, P_AGENT],
                  "\n".join(msg),
                  "The reflection reply and the stop-guard rejection stay in the context. They are appended to the "
                  "last tool message when one precedes them, because Gemini 3 returns an empty reply after a user "
                  "turn there. The other notes go with one call only. Tool assist adds no text and offers the LLM "
                  "only the model's top tools and finish. Routing adds no text either. It sends the step to the "
                  "cheaper LLM with the same messages.")
    adv_names = {"'yes' if yes else 'no'": "yes|no", "confidence": "p"}
    mem_gate = template(P_MEMORY, "_llm_gate", "Should this be stored", {"state": "state"}, without="@@advice@@")
    mem_gate_adv = template(P_MEMORY, "_llm_gate", "@@advice@@", {"state": "state", "advice": "advice"})
    gate = [
        "[Memory-write gate (LLM decides, or the model defers)]", mem_gate, "",
        "[Reflection and routing gates (the model defers)]", PR.llm_gate_prompt(ph("state"), ph("question")), "",
        "[Advise variants: the advice comes before the question]",
        PR.llm_gate_prompt(ph("state"), ph("question"), ph("advice")),
        template(P_PROMPTS, "advice_note", "[Advisor]", adv_names),
        "[Stop advise, added to the LLM's turn]",
        template(P_PROMPTS, "stop_advice_note", "[Advisor]", adv_names, {"Q_STOP_TRIGGER": PR.Q_STOP_TRIGGER}), "",
        "[Memory-read guard: the LLM selects]",
        template(P_MEMORY, "_llm_select", "STORED MEMORIES:", {
            "query": "state", "note": "rejection note, after a veto",
            "listing": "=" + template(P_MEMORY, "_llm_select", "- [m", {"m.id": "id", "m.text[:max_chars]": "text"})},
            {"MEMORY_SELECT_PROMPT": PR.MEMORY_SELECT_PROMPT}),
        "[rejection note]", PR.MEMORY_REJECT_NOTE,
    ]
    gate_text = "\n".join(gate)
    cover("llm_gate_prompt", "advice_note", "stop_advice_note", "MEMORY_SELECT_PROMPT", "MEMORY_REJECT_NOTE")
    assert "@@advice@@" in mem_gate_adv and mem_gate in mem_gate_adv.replace("@@advice@@\n\n", "")
    gate_box = Box("llm_gates", "Yes/no questions the agent LLM answers itself", [P_MEMORY, P_PROMPTS, P_AGENT],
                   gate_text,
                   "Each gate is one call without tools, read as yes when the reply starts with y. With advice, the "
                   "memory gate puts the advice line between the state and its question.")
    action_box = Box("alfworld", "Action-only environments (ALFWorld)", [P_AGENT, P_ALF],
                     "\n".join(["[system]", AG.ACTION_SYSTEM, "[user]", first_user,
                                "[each turn, appended to the history]",
                                template(P_AGENT, "_action_step", "Admissible actions", {
                                    "listing": "=" + "\n".join([template(P_AGENT, "_action_step", "- @@action@@",
                                                                         {"x": "action"})] * 2)}),
                                "[after each action]",
                                template(P_AGENT, "run", "Observation: ", {"res.observation": "observation"}),
                                "[action guard rejected the proposal]",
                                PR.action_reject_note(ph("action")),
                                "[verification pass, after the admissible list]",
                                PR.verify_action_note(ph("action"))]),
                     "The task sentence and the observations are ALFWorld's own {cite:shridhar2021alfworld}. The "
                     "reply's last `Action:` line is matched to the closest admissible action.")
    cover("action_reject_note", "verify_action_note")
    groups.append(Group(
        "The agent LLM on {gym}", "The agent LLM sees a short system prompt, the task and the tool results, plus "
        "the notes below when a decision point acts on its turn.", [sys_box, tasks_box, hard_box, msg_box, gate_box,
                                                                     action_box]))

    # ------------------------------------------------------------------ other harnesses
    T = module_constants(P_TAU2)
    tview = template(P_TAU2, "_view", "CONVERSATION SO FAR", {"head": "", "conv": "earlier messages",
                                                               "cur": "latest message"})
    thead = template(P_TAU2, "_view", "DOMAIN POLICY", {"policy": "policy excerpt"})
    t_text = template(P_TAU2, "_render_msg", "@@text@@", {"who": "user|agent", "m.content": "text"})
    t_call = template(P_TAU2, "_render_msg", "-> tool]", {"who": "=agent"}) + ph("tool(arguments)") + "; ..."
    t_res = template(P_TAU2, "_render_msg", "@@result@@", {"tag": "tool|tool ERROR",
                                                           "_trunc(m.content, tool_chars)": "result"})
    arg_q = {"p.name": "argument", "p.schema.get('description', '')": "description"}
    assert template(P_TAU2, "_param_question", "Which value", {**arg_q, "name": "tool"}) == \
        template(P_AGENT, "_param_question", "Which value", {**arg_q, "tool_name(tool)": "tool"}), \
        "tau2's argument question differs from the reference loop's: show it in the tau2 box"
    tau_q = [
        "[state]", thead + tview,
        "[each message of the conversation, one line]", t_text, t_call, t_res, "",
        "[Tool choice]", T["TOOL_SELECT_Q"], "Options: the domain's tools, then",
        f"{T['RESPOND']} -- {T['RESPOND_DESC']}",
        "[Tool arguments: the reference loop's question]", "",
        "[Tool guard]", template(P_TAU2, "_check", "Is this the right next step", {"desc": "proposal"}),
        "  proposal: " + template(P_TAU2, "_check", "call `", {"name": "tool", "json.dumps(calls[0].arguments, "
                                                               "default=str)[:500]": "arguments as JSON"}),
        "        or: " + template(P_TAU2, "_check", "reply to the user", {"_trunc(msg.content, 300)": "message"}),
        "[Policy guard, at state-changing calls]",
        template(P_TAU2, "_check", "Does this action comply", {"tc.name": "tool", "args": "arguments as JSON"}),
        "[Transfer guard]", T["STOP_Q"], "[Routing]", T["ROUTING_Q"],
    ]
    tau_msgs = [
        "[Decision model chose the tool and the LLM fills the rest]",
        template(P_TAU2, "_emit_tool", "You must now call", {"name": "tool", "json.dumps(filled)": "the model's "
                                                                                                  "arguments"}),
        "[Transfer guard rejected]", template(P_TAU2, "_check", "Transferring to a human agent"),
        "[Policy guard rejected]", template(P_TAU2, "_check", "[Policy check]", {
            "tc.name": "tool", "args[:300]": "arguments", "lack": "=" + template(P_TAU2, "_check", "or to lack")}),
        "[Tool guard rejected]", template(P_TAU2, "_check", "was judged likely wrong", {"desc[:300]": "proposal"}),
    ]
    tau_box = Box("tau2", "{tau2}: state and questions", [P_TAU2], "\n".join(tau_q),
                  "The policy question drops the confirmation clause when the domain policy does not require "
                  "explicit confirmation. The agent's system prompt and the user simulator are {tau2}'s own "
                  "{cite:barres2025tau2}.")
    tau_msg_box = Box("tau2_notes", "{tau2}: messages to the agent LLM", [P_TAU2], "\n".join(tau_msgs),
                      "A rejected proposal is not executed. The LLM answers again, with the note as a final user "
                      "turn.")
    S = module_constants(P_SWE)
    swe_state = "\n\n".join([
        template(P_SWE, "_s1_state", "TASK (issue to resolve)", {"_trunc(self._task, c.task_max_chars)":
                                                                 "issue text"}),
        "HISTORY (most recent last):\n" + template(P_SWE, "_s1_state", "] $ ", {
            "e['step']": "step", "_trunc(e['command'], 300)": "command", "tag": "= [NOT EXECUTED]",
            "e['returncode']": "exit code", "_trunc(e['output'], c.hist_obs_chars)": "output"}),
        template(P_SWE, "_s1_state", "CURRENT OBSERVATION", {"_trunc(last, c.obs_truncate_chars)": "last output"}),
        template(P_SWE, "_s1_state", "STEP:", {"max(self._step, 0)": "step"}),
        "[guard only] " + template(P_SWE, "_guard", "REASONING FOR THIS STEP", {
            "_trunc(thought, self.jcfg.thought_max_chars)": "the agent's reasoning"}),
        "[stop guard only] " + template(P_SWE, "_stop_guard", "issuing the submit command")])
    swe_q = [
        "[Routing: deep or fast path (E4, E4b and the H5 route)]", S["Q_ROUTING"], "",
        "[Command guard (H5)]", fmt_fields(S["Q_GUARD"], {"cmd": "command"}),
        f"criteria: yes = {S['CRIT_GUARD']['true']}", f"no = {S['CRIT_GUARD']['false']}", "",
        "[Submit guard (H5)]", S["Q_STOP_GUARD"], f"criteria: yes = {S['CRIT_STOP']['true']}",
        f"no = {S['CRIT_STOP']['false']}", "",
        "[blocked command, as its observation]",
        fmt_fields(S["MSG_BLOCK"], {"tau": "threshold"}) + S["MSG_BLOCK_OVERRIDE"], "[deferred submission]",
        S["MSG_STOP_NUDGE"],
    ]
    swe_cfg = yaml.safe_load((ROOT / SWE_YAML).read_text())["jev"]
    swe_box = Box("swe_state", "mini-SWE-agent: decision-model state", [P_SWE + ": _s1_state", SWE_YAML], swe_state,
                  f"The history keeps the last {swe_cfg['history_window']} commands and cuts each output to "
                  f"{swe_cfg['hist_obs_chars']} characters. Only a blocked command carries the `[NOT EXECUTED]` "
                  "tag. The questions of one step share this state and go out in one request.")
    swe_q_box = Box("swe", "mini-SWE-agent: questions and messages", [P_SWE], "\n".join(swe_q),
                    "The agent's prompts are mini-SWE-agent's SWE-bench configuration, unchanged "
                    "{cite:yang2024sweagent}. The reflection, action-type and stop-trigger hooks are implemented "
                    "but were not run.")
    pi_names = {"event.toolName": "tool", "args": "arguments as JSON", "call": "tool(arguments)"}
    gate_q = ts_string(P_PI + "jev-tool-gate.ts", "instructions:", "right next step", pi_names)
    gate_c = ts_criteria(P_PI + "jev-tool-gate.ts", "kind === \"bool\"")
    stop_q = ts_string(P_PI + "jev-stop-verify.ts", "instructions:", "actually complete")
    stop_c = ts_criteria(P_PI + "jev-stop-verify.ts", "const GUARD_QUESTION")
    refl_q = ts_string(P_PI + "jev-reflect.ts", "instructions:", "stuck")
    refl_c = ts_criteria(P_PI + "jev-reflect.ts", "const QUESTION")
    common = (ROOT / P_PI / "jev-common.ts").read_text()
    for key in ("task: taskText", "{ summary: truncate(summary", "history,", "{ current_observation: truncate("):
        assert key in common, f"jev-common.ts state builder changed: {key!r}"
    assert "{ proposed_call: call }" in (ROOT / P_PI / "jev-tool-gate.ts").read_text()
    assert "{ final_response: truncate(" in (ROOT / P_PI / "jev-stop-verify.ts").read_text()
    pi_text = [
        "[state: a JSON object]",
        '{"task": ' + ph("latest user message") + ', "summary": ' + ph("compaction summary, if any") + ',',
        ' "history": ["[' + ph("i") + '] ' + ph("tool(arguments)") + ' -> ' + ph("result") + '", ...],',
        ' "current_observation": ' + ph("latest tool result") + ',',
        ' "proposed_call" (guard) | "final_response" (stop): ' + ph("value") + "}", "",
        "[Tool guard]", gate_q, f"criteria: yes = {gate_c['true']}", f"no = {gate_c['false']}",
        "[rejected: the tool result]", ts_string(P_PI + "jev-tool-gate.ts", "reason:", "Verifier", pi_names), "",
        "[Stop guard, when the agent settles]", stop_q, f"criteria: yes = {stop_c['true']}", f"no = {stop_c['false']}",
        "[rejected: a message, then one more turn]", ts_string(P_PI + "jev-stop-verify.ts", "const DEFAULT_NUDGE ="),
        "", "[Reflection, after a failed or repeated call]", refl_q, f"criteria: yes = {refl_c['true']}",
        f"no = {refl_c['false']}", "[fired: a message]", ts_string(P_PI + "jev-reflect.ts", "const DEFAULT_PROMPT ="),
    ]
    pi_box = Box("pi", "pi on Terminal-Bench-Science (H5)", [P_PI + "jev-tool-gate.ts", P_PI + "jev-stop-verify.ts",
                                                             P_PI + "jev-reflect.ts", P_PI + "jev-common.ts"],
                 "\n".join(pi_text),
                 "The agent's prompts are pi's own {cite:pi2026}, and the tasks are Terminal-Bench-Science's "
                 "{cite:tbscience2026}. The router, memory, tool-pruning and compaction extensions were not run.")
    st_consts = module_constants(P_STATIC)
    bfcl_tool = template(P_STATIC, "bfcl_row_items", "Which function should be called")
    bfcl_arg = template(P_STATIC, "bfcl_row_items", "will be called", {
        "gold_tool": "reference function", "pname": "argument", "schema.get('description', '')[:200]":
        "argument description"})
    w2c_state = template(P_STATIC, "when2call_item", "AVAILABLE TOOLS", {"tool_lines": "tools",
                                                                         "question": "question"})
    w2c_tool = template(P_STATIC, "w2c_tool_text", "@@name@@: ", {"t.get('name')": "name", "(t.get('description') or "
                        "'')[:200]": "description"})
    w2c_par = template(P_STATIC, "w2c_tool_text", "    * ", {"pname": "argument", "typ": "type", "flag":
                                                             "required|optional", "desc": "description"})
    w2c_state = w2c_state.replace(ph("tools"), w2c_tool + "\n" + w2c_par + "\n" + skip("one block per tool"))
    w2c_q = template(P_STATIC, "when2call_item", "How should the assistant")
    bfcl_state = template(P_STATIC, "_query_text", ": ", {"m['role'].upper()": "ROLE", "m['content']": "message"})
    l1 = ["[BFCL state, one line per message]", bfcl_state, "[BFCL tool choice]", bfcl_tool,
          "Options: the functions, then", st_consts["NONE_TOOL"] + " -- "
          + template(P_STATIC, "bfcl_row_items", "no function applies"),
          "[BFCL argument, given the reference function]", bfcl_arg,
          "Options: enum or boolean values, or spans copied from the request, then " + OTHER, "",
          "[When2Call state]", w2c_state, "[When2Call]",
          w2c_q, *[f"- {v}" for v in st_consts["W2C_OPTIONS"].values()], "",
          "[Judge of the native function-calling baseline's text replies]",
          module_constants(P_DEVAL)["JUDGE_PROMPT"].replace("{tools}", ph("tools")).replace(
              "{question}", ph("request")).replace("{reply}", ph("reply")).replace("{{", "{").replace("}}", "}")]
    l1_box = Box("static", "L1 static decisions (BFCL, When2Call)", [P_STATIC, P_DEVAL], "\n".join(l1),
                 "The argument question names the reference function on purpose, so that it measures argument "
                 "filling given the right function. The native function-calling baseline sees each benchmark's own "
                 "messages and tools.")
    groups.append(Group(
        "Other harnesses and the static suites", "Each integration renders its own state and asks its own questions. "
        "We list only the parts we wrote.", [tau_box, tau_msg_box, swe_box, swe_q_box, pi_box, l1_box]))

    intro = ("This subsection lists every prompt our code sends to a model in the experiments, generated from the "
             "code by `scripts/paper_prompts.py`. Examples come from development seed " + str(DEV_SEED) + ", which "
             "no evaluation uses. <name> marks a value filled in at run time, an italic [note] marks our "
             "shortening, and a hook marks a line we wrapped. No prompt contains the label of the decision it asks "
             "about. The answer key and every label stay in the decision log. Training data and teacher labels add "
             "no prompt of their own, since teachers answer the same requests and training rows use the "
             "trained-model format.")
    return intro, groups


class _PInfo:
    """A stand-in tool parameter for rendering the argument question with placeholders."""

    def __init__(self, name: str, desc: str):
        self.name = name
        self.schema = {"description": desc}


# --------------------------------------------------------------------------------------------- rendering
MARKUP_TEX = {"{gym}": r"\gym{}", "{jev}": r"\jev{}", "{ours}": r"\ours{}", "{tau2}": r"$\tau^2$-bench"}
MARKUP_MD = {"{gym}": "DecisionGym", "{jev}": "Jev", "{ours}": "AgentDecision-4B", "{tau2}": "tau2-bench"}
_TOKEN = re.compile(r"@@(SKIP:)?(J:)?(.+?)@@")
_TT_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "%": r"\%",
           "_": r"\_", "^": r"\^{}", "~": r"\textasciitilde{}", "<": "<{}", ">": ">{}", "-": "-{}",
           "'": r"\textquotesingle{}", "`": r"\textasciigrave{}", '"': r"\textquotedbl{}", " ": r"\ "}
_RM_ESC = {k: v for k, v in _TT_ESC.items() if k not in ("'", "-", " ", "<", ">")} | {
    "<": r"\textless{}", ">": r"\textgreater{}", '"': "''"}


def _check_ascii(s: str, where: str) -> None:
    bad = sorted({c for c in s if ord(c) > 126 or (ord(c) < 32 and c != "\n")})
    if bad:
        raise ValueError(f"{where}: characters without a TeX mapping {bad!r}")


def tex_tt(s: str) -> str:
    return "".join(_TT_ESC.get(c, c) for c in s)


def tex_rm(s: str) -> str:
    return "".join(_RM_ESC.get(c, c) for c in s)


def prose_tex(s: str) -> str:
    """Titles / notes: markup tokens, `code`, {cite:key,...}, <name> placeholders; everything else escaped."""
    out = []
    for part in re.split(r"(`[^`]*`|\{cite:[^}]*\}|\{\w+\}|<[\w ,.-]+>)", s):
        if not part:
            continue
        if part in MARKUP_TEX:
            out.append(MARKUP_TEX[part])
        elif part.startswith("{cite:"):
            out.append(r"\citep{" + part[6:-1] + "}")
        elif part.startswith("`"):
            out.append(r"\texttt{" + tex_tt(part[1:-1]).replace(r"\ ", " ") + "}")
        elif part.startswith("<") and part.endswith(">"):
            out.append(r"\textit{$\langle$" + tex_rm(part[1:-1]) + r"$\rangle$}")
        else:
            out.append(tex_rm(part))
    return "".join(out)


def prose_md(s: str) -> str:
    out = []
    for part in re.split(r"(`[^`]*`)", s):
        if not part.startswith("`"):
            for k, v in MARKUP_MD.items():
                part = part.replace(k, v)
            part = re.sub(r"\{cite:([^}]*)\}", lambda m: f"[{m.group(1)}]", part)
            part = re.sub(r"<([\w ,.-]+)>", r"⟨\1⟩", part)
        out.append(part)
    return "".join(out)


def _atoms(line: str) -> list[tuple[str, int, bool]]:
    """(tex, width in characters, breakable-after) atoms of one line of a box."""
    atoms: list[tuple[str, int, bool]] = []
    pos = 0
    for m in _TOKEN.finditer(line):
        for c in line[pos:m.start()]:
            atoms.append((_TT_ESC.get(c, c), 1, c == " "))
        name = m.group(3)
        if m.group(1):
            atoms.append((r"\textrm{\textit{[" + tex_rm(name) + "]}}", len(name) + 2, False))
        else:
            atoms.append((r"\textrm{\textit{$\langle$" + tex_rm(name) + r"$\rangle$}}", len(name) + 2, False))
        pos = m.end()
    for c in line[pos:]:
        atoms.append((_TT_ESC.get(c, c), 1, c == " "))
    return atoms


HOOK = r"\rlap{\textcolor{gray}{$\hookrightarrow$}}\phantom{xx}"   # exactly two characters wide


def wrap_tex(line: str, width: int = WIDTH) -> list[str]:
    """One logical line -> TeX lines of at most ``width`` characters (breaking after the last space when it leaves
    at least a third of the line filled, else hard); continuation lines start with a grey hook (2 characters)."""
    atoms = _atoms(line)
    lead = 0
    while lead < len(atoms) and atoms[lead][0] == r"\ ":
        lead += 1
    out: list[str] = []
    i, first = lead, True
    while i < len(atoms) or first:
        avail = width - (lead if first else 2)
        j, w, brk = i, 0, -1
        while j < len(atoms) and w + atoms[j][1] <= avail:
            w += atoms[j][1]
            if atoms[j][2]:
                brk = j
            j += 1
        if j < len(atoms) and brk > i and (brk - i) >= avail // 3:
            j = brk + 1
        if j == i and i < len(atoms):          # one atom wider than a line: it gets a line of its own
            j = i + 1
        k = j
        while k > i and atoms[k - 1][0] == r"\ " and j < len(atoms):    # no trailing space on a wrapped line
            k -= 1
        body = "".join(a[0] for a in atoms[i:k])
        prefix = (r"\phantom{" + "x" * lead + "}" if lead else "") if first else HOOK
        out.append(prefix + body)
        i, first = j, False
    return out


def box_tex_lines(text: str) -> list[str]:
    lines: list[str] = []
    for ln in text.split("\n"):
        lines.extend(wrap_tex(ln) if ln.strip() else [r"\mbox{}"])
    return lines


# One box line: a shaded strip with a grey bar on the left. Strips of consecutive lines touch (\strut height + depth =
# \scriptsize \baselineskip), so a box reads as one shaded block, and a page or column may break between two lines. An empty
# line of a prompt is a strip of half a line's height (\promptgap): its own \baselineskip is halved and its depth equals
# the strut's, so the strips above and below it still touch.
LINE_MACRO = (r"\providecommand{\promptline}[1]{\noindent\colorbox{gray!7}{\makebox[\columnwidth][l]{"
              r"\textcolor{gray!55}{\vrule width 1pt}\hspace{3pt}\strut#1}}\par}"
              "\n"
              r"\providecommand{\promptgap}{{\setlength{\baselineskip}{0.5\baselineskip}\noindent\colorbox{gray!7}{"
              r"\makebox[\columnwidth][l]{\textcolor{gray!55}{\vrule width 1pt height \dimexpr\baselineskip-\dp\strutbox"
              r"\relax depth \dp\strutbox}}}\par}}")


def render_tex(intro: str, groups: list[Group]) -> str:
    out = ["% Generated by scripts/paper_prompts.py from the code: do not edit by hand (tests/test_paper_prompts.py",
           "% checks it is in sync). Examples come from a recorded offline run at DEVELOPMENT seed "
           f"{DEV_SEED} (JevGym-Hard: {HARD_SEED});",
           "% no evaluation seed (200000-200499, 100000-100499, 300000-300999, 310000-310599) and no gold label "
           "appears.", r"\subsection{Prompts}", r"\label{app:prompts-full}", LINE_MACRO,
           # lineno (review mode) runs the output routine at every \par, and a page of \scriptsize prompt lines holds more
           # than TeX's default 200 paragraphs between two shipouts
           r"\ifnum\maxdeadcycles<1000 \global\maxdeadcycles=1000 \fi", prose_tex(intro), ""]
    n = 0
    for g in groups:
        out += [r"\paragraph{" + prose_tex(g.title) + ".}", prose_tex(g.intro), ""]
        for b in g.boxes:
            n += 1
            lines = box_tex_lines(b.text)
            src = ", ".join(r"\texttt{" + tex_tt(x).replace(r"\ ", " ").replace("/", r"/\allowbreak{}") + "}"
                            for x in b.sources)
            # title, source and note in one small paragraph above the box (kept with the box's first lines)
            out += [r"\par\smallskip{\footnotesize\raggedright\interlinepenalty=10000\noindent"
                    r"\textbf{" + prose_tex(f"P{n}. " + b.title) + r".} Source: " + src + "."
                    + (" " + prose_tex(b.note) if b.note else "") + r"\par}\nobreak\vspace{1pt}\nobreak",
                    r"\begingroup\ttfamily\scriptsize\setlength{\fboxsep}{0pt}\setlength{\parskip}{0pt}%"]
            for i, ln in enumerate(lines):
                # no break after the first line or before the last one
                glue = r"\nobreak" if i == 0 or i >= len(lines) - 2 else ""
                out.append((r"\promptgap" if ln == r"\mbox{}" else r"\promptline{" + ln + "}") + glue)
            out += [r"\endgroup\smallskip", ""]
    return "\n".join(out) + "\n"


def render_md(intro: str, groups: list[Group]) -> str:
    out = ["# Prompts sent to models", "",
           f"Generated by `scripts/paper_prompts.py` from the code (do not edit by hand); rendered for the paper in "
           f"`paper/sections/app_prompts.tex` (App. app:prompts-full). Examples: development seed {DEV_SEED} "
           f"(JevGym-Hard {HARD_SEED}).", "", prose_md(intro).replace("`scripts/paper_prompts.py`", "this script"), ""]
    n = 0
    for g in groups:
        out += [f"## {prose_md(g.title)}", "", prose_md(g.intro), ""]
        for b in g.boxes:
            n += 1
            text = _TOKEN.sub(lambda m: f"[{m.group(3)}]" if m.group(1) else f"⟨{m.group(3)}⟩", b.text)
            out += [f"### P{n}. {prose_md(b.title)}", "", "Source: " + ", ".join(f"`{s}`" for s in b.sources) + "."
                    + (" " + prose_md(b.note) if b.note else ""), "", "```text", text, "```", ""]
    return "\n".join(out)


def build() -> tuple[str, str]:
    intro, groups = catalogue()
    for g in groups:
        for b in g.boxes:
            _check_ascii(_TOKEN.sub("", b.text), b.key)
            _check_ascii(b.title + b.note, b.key)
    tex, md = render_tex(intro, groups), render_md(intro, groups)
    assert "@@" not in tex and "@@" not in md, "a placeholder sentinel was not rendered"
    return tex, md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-tex", type=Path, default=OUT_TEX)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    a = ap.parse_args(argv)
    tex, md = build()
    for p, s in ((a.out_tex, tex), (a.out_md, md)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
        print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p} ({len(s.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
