"""Decision points and their modes: where the decision model acts in the proxy, and how.

The paper's harness asks a decision model at eight decision points (tool choice, tool arguments, memory writes and
reads, reflection, stopping, routing, and the action of an action-only environment) in one of several integration
modes. A proxy sees each request (messages and tools) before the model runs and each reply before the client gets
it, so it offers the points it can act on from there (docs/modes.md has the full map, with the paper's findings):

=============  =====================================  ===========================================================
point          modes                                  question (options)
=============  =====================================  ===========================================================
routing        off, shadow, cascade, guard            next_step (routine, hard); guard: fast_step_ok (keep, redo)
tool_select    off, shadow, assist, cascade, guard    next_tool (the request's tool names, reply_without_tool)
stop           off, shadow, strong, guard, trigger    task_done (done, not_done); trigger: already_done (yes, no)
reflection     off, shadow, trigger, always           needs_reflection (yes, no)
action         off, shadow, guard                     action_risk (safe, risky)
=============  =====================================  ===========================================================

``shadow`` asks the question and logs the answer next to what the agent did, and changes nothing. A decision-model
failure or timeout at a point falls back to that point's LLM-only behaviour, so it never stops the agent.
``tool_params`` and the memory points are not offered: docs/modes.md says why.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .decision import DecisionBackend, Question, RulesBackend, StepView
from .types import Message, Part, Request, Response, ToolCall

POINTS = ("routing", "tool_select", "stop", "reflection", "action")
MODES: dict[str, tuple[str, ...]] = {
    "routing": ("off", "shadow", "cascade", "guard"),
    "tool_select": ("off", "shadow", "assist", "cascade", "guard"),
    "stop": ("off", "shadow", "strong", "guard", "trigger"),
    "reflection": ("off", "shadow", "trigger", "always"),
    "action": ("off", "shadow", "guard"),
}
# names accepted for a mode that equals another one in the proxy (docs/modes.md)
ALIASES = {("routing", "trigger"): "cascade"}
NOT_OFFERED = {
    "tool_params": "the LLM writes a call's arguments in the same reply as the tool; a proxy cannot fill some of them",
    "memory_write": "the proxy has no memory module (the harness owns its context)",
    "memory_read": "the proxy has no memory module (the harness owns its context)",
}
KEYS: dict[str, set[str]] = {
    "routing": {"mode", "tau"},
    "tool_select": {"mode", "tau", "k", "min_tools", "max_options", "tier", "force"},
    "stop": {"mode", "tau", "scope"},
    "reflection": {"mode", "tau", "every"},
    "action": {"mode", "tau", "prefilter", "tools", "patterns", "on_error"},
}
# default thresholds. Tool choice is delegated only when the decision model is (almost) certain until `jah fit` has
# fitted a threshold on your own runs: the paper's fitted tool threshold was 1.00 (Jev, Gemini 3.5 Flash), and the same
# delegation at a fixed 0.5 lost 21.5 points of task success.
TAU = {"tool_select": 0.99, "stop": 0.9, "reflection": 0.8, "action": 0.5}

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "route": {},                                                    # the default: routing cascade + finish check
    "paper": {"tool_select": {"mode": "cascade"}},                  # + the paper's tool-choice cascade (fit tau!)
    "safe": {"action": {"mode": "guard"}},                          # + a second look at risky actions
    "shadow": {"tool_select": {"mode": "shadow"}, "stop": {"mode": "shadow"},
               "reflection": {"mode": "shadow"}, "action": {"mode": "shadow"}},   # log everything, for `jah fit`
}


@dataclass(frozen=True)
class PointCfg:
    point: str
    mode: str = "off"
    tau: float = 0.5
    k: int = 5                      # tool_select assist: keep the k likeliest tools
    min_tools: int = 12             # tool_select assist: only shortlist requests with at least this many tools
    max_options: int = 100          # tool_select: options per question (more tools: a two-round knockout)
    tier: str | None = None         # tool_select cascade: the tier that writes the arguments (default: the fast tier)
    force: bool = True              # tool_select cascade: force the tool (tool_choice); false = only pick the tier
    scope: str = "fast"             # stop guard / shadow: check final answers of the fast tier, or of every tier
    every: int = 5                  # reflection always: every N steps
    prefilter: str = "risky"        # action: risky (pattern match) | command (command-like tools) | all
    tools: tuple[str, ...] = ()     # action: extra regexes for command-like tool names
    patterns: tuple[str, ...] = ()  # action: extra regexes for risky call text
    on_error: str = "accept"        # action guard: what a decision-model failure does (accept | recheck)

    @property
    def on(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class Decisions:
    routing: PointCfg
    tool_select: PointCfg
    stop: PointCfg
    reflection: PointCfg
    action: PointCfg

    def modes(self) -> dict[str, str]:
        return {p: getattr(self, p).mode for p in POINTS}

    def only_routing(self) -> bool:
        """True for the default behaviour (routing + finish check): the step record stays as it always was."""
        return all(getattr(self, p).mode == "off" for p in ("tool_select", "reflection", "action")) \
            and self.stop.mode in ("off", "strong")


def _mode_name(v: Any) -> Any:
    return "off" if v is False else v       # YAML 1.1 reads a bare `off` as false


def _norm_point(point: str, v: Any, where: str) -> dict[str, Any]:
    v = _mode_name(v)
    if isinstance(v, str):
        v = {"mode": v}
    if not isinstance(v, Mapping):
        raise ValueError(f"{where}.{point}: expected a mapping like {{mode: shadow}}, got {v!r}")
    d = dict(v)
    bad = set(d) - KEYS[point]
    if bad:
        raise ValueError(f"{where}.{point}: unknown keys {sorted(bad)} (allowed: {sorted(KEYS[point])})")
    if "mode" in d:
        d["mode"] = _mode_name(d["mode"])
        m = ALIASES.get((point, str(d["mode"])), str(d["mode"]))
        if m not in MODES[point]:
            raise ValueError(f"{where}.{point}.mode must be one of {', '.join(MODES[point])}, got {d['mode']!r} "
                             "(docs/modes.md lists what each point supports)")
        d["mode"] = m
    if "tau" in d and d["tau"] is not None:
        t = float(d["tau"])
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"{where}.{point}.tau must be in [0, 1], got {d['tau']!r}")
        d["tau"] = t
    for k, lo, hi in (("k", 1, None), ("min_tools", 0, None), ("every", 1, None), ("max_options", 2, 255)):
        if k in d:
            try:
                d[k] = int(d[k])
            except (TypeError, ValueError):
                raise ValueError(f"{where}.{point}.{k} must be an integer, got {d[k]!r}") from None
            if d[k] < lo or (hi is not None and d[k] > hi):
                raise ValueError(f"{where}.{point}.{k} must be in [{lo}, {hi or 'inf'}]"
                                 + (" (Jev's limit is 255 options)" if k == "max_options" else ""))
    if "force" in d:
        d["force"] = bool(d["force"])
    if d.get("tier") is not None:
        d["tier"] = str(d["tier"])
    for k, allowed in (("scope", ("fast", "all")), ("prefilter", ("risky", "command", "all")),
                       ("on_error", ("accept", "recheck"))):
        if k in d and d[k] not in allowed:
            raise ValueError(f"{where}.{point}.{k} must be one of {', '.join(allowed)}, got {d[k]!r}")
    for k in ("tools", "patterns"):
        if k in d:
            vals = d[k] if isinstance(d[k], (list, tuple)) else [d[k]]
            for r in vals:
                try:
                    re.compile(str(r))
                except re.error as e:
                    raise ValueError(f"{where}.{point}.{k}: bad regex {r!r}: {e}") from e
            d[k] = tuple(str(r) for r in vals)
    return d


def validate(decisions: Mapping[str, Any] | None, where: str = "decisions") -> dict[str, dict[str, Any]]:
    """Check a ``decisions:`` block; returns it normalized (string shorthands expanded, aliases resolved)."""
    out: dict[str, dict[str, Any]] = {}
    for point, v in (decisions or {}).items():
        if point in NOT_OFFERED:
            raise ValueError(f"{where}.{point} is not offered by the proxy: {NOT_OFFERED[point]} (docs/modes.md)")
        if point not in KEYS:
            raise ValueError(f"{where}: unknown decision point {point!r} (one of {', '.join(POINTS)})")
        out[point] = _norm_point(point, v, where)
    return out


def check_preset(name: Any, where: str = "preset") -> str | None:
    if name in (None, ""):
        return None
    if name not in PRESETS:
        raise ValueError(f"{where}: unknown preset {name!r} (one of {', '.join(PRESETS)})")
    return str(name)


def _merge(base: dict[str, dict[str, Any]], over: Mapping[str, Any]) -> None:
    for point, v in validate(over).items():
        base[point] = {**base[point], **v}


def resolve(router_tau: float, verify_finish: bool = True, preset: str | None = None,
            decisions: Mapping[str, Any] | None = None, spec: Mapping[str, Any] | None = None,
            pinned: bool = False) -> Decisions:
    """The decision points of one model name. Later layers win: the defaults (routing cascade at ``router.tau``; the
    finish check when ``router.verify_finish``), the top-level ``preset``, the top-level ``decisions``, the model's
    legacy ``verify_finish``, the model's ``preset``, the model's ``decisions``. A pinned name (``tier:``) is a
    baseline: only its own ``preset`` / ``decisions`` apply to it."""
    base: dict[str, dict[str, Any]] = {p: {} for p in POINTS}
    if not pinned:
        base["routing"]["mode"] = "cascade"
        base["stop"]["mode"] = "strong" if verify_finish else "off"
        if preset:
            _merge(base, PRESETS[preset])
        _merge(base, decisions or {})
    if spec:
        if not pinned and "verify_finish" in spec:
            base["stop"]["mode"] = "strong" if spec["verify_finish"] else "off"
        if not pinned and "tau" in spec:
            base["routing"]["tau"] = float(spec["tau"])
        if spec.get("preset"):
            _merge(base, PRESETS[str(spec["preset"])])
        _merge(base, spec.get("decisions") or {})
    pts = {}
    for p in POINTS:
        d = dict(base[p])
        if d.get("tau") is None:
            d["tau"] = router_tau if p == "routing" else TAU[p]
        pts[p] = PointCfg(point=p, **d)
    if pinned:
        pts["routing"] = PointCfg(point="routing", mode="off", tau=router_tau)
    return Decisions(**pts)


# ---------------------------------------------------------------------------------------------- questions and notes
_WATCH = "You are watching an AI agent work on the task below, one tool call at a time. "
NO_TOOL = "reply_without_tool"
NO_TOOL_DESC = "Call no tool: reply in text (the final answer, a report, or a question to the user)"
TOOL_TEXT = _WATCH + "Which tool should the agent call next?"

ROUTE_GUARD = Question(
    id="fast_step_ok",
    text=(_WATCH + "A fast, cheap model wrote the agent's next step, shown at the end as PROPOSED NEXT STEP. Decide "
          "whether to keep it. 'keep': a sensible next move that follows from what the agent knows. 'redo': the step "
          "looks wrong, careless or premature (a wrong command or edit, a repeated failed attempt, or a claim that "
          "the task is done without checking), so a stronger model should write this step instead."),
    options=["keep", "redo"],
    descriptions=["The proposed step is a sensible next move", "A stronger model should write this step instead"])
STOP = Question(
    id="task_done",
    text=(_WATCH + "The agent now wants to end its turn with the reply shown at the end as PROPOSED FINAL REPLY, with "
          "no further tool calls. Is ending here right: is the task actually complete and the reply correct (or does "
          "the agent rightly need input from the user)?"),
    options=["done", "not_done"],
    descriptions=["Ending the turn here is right: the task is complete and the reply is correct, or the agent rightly "
                  "asks the user for input",
                  "Work remains, a result was not checked, or the reply is wrong or incomplete"])
STOP_TRIGGER = Question(
    id="already_done",
    text=_WATCH + "Has the task already been fully completed, so the agent should stop now and give its final answer?",
    options=["yes", "no"],
    descriptions=["The task is complete: the agent should stop calling tools and answer",
                  "Work remains"])
REFLECT = Question(
    id="needs_reflection",
    text=(_WATCH + "Should the agent pause and re-check before its next action: is it stuck or off track, repeating "
          "itself, or likely to make a wrong action or give a wrong final answer next?"),
    options=["yes", "no"],
    descriptions=["The agent should pause and re-check its progress and plan before acting",
                  "The agent is making progress; no pause needed"])
ACTION = Question(
    id="action_risk",
    text=(_WATCH + "The agent proposes the action shown at the end as PROPOSED ACTION. Could it be irreversible or "
          "risky: delete or overwrite data that is not easy to restore, rewrite version-control history or "
          "force-push, change systems outside the task's workspace (deploy, publish, send messages, spend money), or "
          "weaken security? Answer 'risky' only if running it without a second look could cause harm that is hard "
          "to undo."),
    options=["safe", "risky"],
    descriptions=["Running it is fine or easy to undo", "It could cause harm that is hard to undo"])

NOTE = "[Note from the harness] "
REFLECT_NOTE = (NOTE + "Before your next action, pause and check your progress: re-read the task, look for a mistake "
                "or a wrong assumption so far, and decide the best next step. Then continue.")
STOP_TRIGGER_NOTE = (NOTE + "The task may already be complete. Check the result against the task. If it is complete, "
                     "stop calling tools and give your final answer now; otherwise continue.")
STOP_REJECT_NOTE = (NOTE + "The task may not be complete yet, or the final reply may be wrong. Re-check the task and "
                    "the results before you end your turn.")
CHARS_PROPOSAL = 1500


def call_text(c: ToolCall, n: int = 300) -> str:
    s = f"{c.name}({c.arguments})"
    return s if len(s) <= n else s[: n - 4] + " ..."


def tool_guard_note(resp: Response) -> str:
    what = call_text(resp.tool_calls[0]) if resp.tool_calls else "a final reply without a tool call"
    return (NOTE + f"A check doubts the proposed next step `{what}`. Re-read the task and the latest results, then "
            "make the right next call (the same one if it is right).")


def action_note(calls: list[ToolCall]) -> str:
    what = "; ".join(f"`{call_text(c)}`" for c in calls)
    return (NOTE + f"The proposed action {what} may be irreversible or risky (it could delete data, rewrite history, "
            "or change systems outside the task). Check that it is needed and safe. If it is, make the same call "
            "again; if not, take a safer next step.")


def with_notes(req: Request, notes: list[str]) -> Request:
    """The request with harness notes appended to its last message (a tool result or a user message), so the
    conversation keeps its shape on every API and a cached prefix stays a prefix. Marked ``modified``."""
    if not notes:
        return req
    text = "\n\n".join(notes)
    msgs = list(req.messages)
    if msgs and msgs[-1].role in ("tool", "user"):
        m = msgs[-1]
        msgs[-1] = replace(m, parts=list(m.parts) + [Part(text=("\n\n" if m.text else "") + text)])
    else:
        msgs.append(Message.of("user", text))
    return replace(req, messages=msgs, modified=True)


def proposal_text(resp: Response) -> str:
    if resp.tool_calls:
        s = "; ".join(f"{c.name}({c.arguments})" for c in resp.tool_calls)
    else:
        s = f"message: {resp.text}"
    s = " ".join(s.split())
    if len(s) > CHARS_PROPOSAL:     # keep the head and the tail: agents end long replies with the answer
        s = s[:300] + " ... " + s[-(CHARS_PROPOSAL - 305):]
    return s


def llm_choice(resp: Response) -> str:
    """What the LLM chose at the tool-choice point: its first tool call, or no tool."""
    return resp.tool_calls[0].name if resp.tool_calls else NO_TOOL


def tool_options(req: Request) -> tuple[list[str], list[str]] | None:
    """The tool-choice question's options for this request, or None when the choice is not open (no tools, the
    client forced a tool or forbade tools, or fewer than two options)."""
    tc = req.tool_choice
    if not req.tools or tc == "none" or isinstance(tc, dict):
        return None
    opts: list[str] = []
    descs: list[str] = []
    for t in req.tools:
        if t.name and t.name not in opts:
            opts.append(t.name)
            first = next((ln.strip() for ln in (t.description or "").splitlines() if ln.strip()), "")
            descs.append(first[:160])
    if tc != "required" and NO_TOOL not in opts:
        opts.append(NO_TOOL)
        descs.append(NO_TOOL_DESC)
    return (opts, descs) if len(opts) >= 2 else None


def used_tools(req: Request) -> set[str]:
    return {c.name for m in req.messages for c in m.tool_calls}


# ---------------------------------------------------------------------------------------------- action pre-filter
COMMAND_TOOLS = [
    r"bash", r"shell", r"exec", r"command", r"\bcmd\b", r"terminal", r"\brun", r"local_shell", r"apply_patch",
    r"write", r"edit", r"create", r"delete", r"remove", r"move", r"rename", r"patch", r"\bgit", r"deploy",
    r"publish", r"send", r"post", r"push", r"upload", r"sql", r"query", r"kubectl", r"docker", r"http", r"request",
    r"transfer", r"pay", r"cancel", r"refund", r"update", r"drop",
]
# a command position: the start of the text or of a JSON string, after ; & | ( ` or an escaped newline, or after
# sudo / xargs / exec (so `grep -rn rm .` is not a deletion)
_CMD = r"""(?:^|["';&|(`]|\\n|\bsudo\s+|\bxargs\s+(?:-\S+\s+)*|\bexec\s+)\s*"""
RISKY_PATTERNS = [
    # files
    _CMD + r"rm(?![\w.-])", _CMD + r"(?:rmdir|unlink|shred)\b", r"\bfind\b[^\n]*-delete\b",
    r"\bfind\b[^\n]*-exec\s+rm\b", r"shutil\.rmtree", r"os\.(remove|unlink|rmdir)\b", r"\*\*\* Delete File:",
    r"\btruncate\s+-s\s*0\b", r">\s*/dev/sd",
    # version control
    r"\bgit\s+push\b", r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\s+-\w*f", r"\bgit\s+checkout\s+(--\s+)?\.(\s|$)",
    r"\bgit\s+restore\s+(--\S+\s+)*\.(\s|$)", r"\bgit\s+branch\s+-D\b", r"\bgit\s+stash\s+(drop|clear)\b",
    r"\bgit\s+rebase\b", r"\bgit\s+filter-(branch|repo)\b", r"\bgit\s+commit\s+[^\n]*--amend\b",
    r"\bgit\s+tag\s+-d\b", r"\bgit\s+update-ref\s+-d\b",
    # databases
    r"(?i:\b(drop|truncate)\s+(table|database|schema|index)\b)", r"(?i:\bdelete\s+from\b)",
    r"(?i:\balter\s+table\b[^\n]*\bdrop\b)", r"(?i:\bflushall\b)", r"(?i:\bdropdatabase\b)",
    # system
    r"\bmkfs(\.\w+)?\b", r"\bdd\s+[^\n]*\bof=", r"\bchmod\s+(-R|777)\b", r"\bchown\s+-R\b", r"\bsudo\b",
    r"\b(shutdown|reboot|halt|poweroff)\b", r"\bkill\s+-9\b", r"\b(pkill|killall)\b", r":\(\)\s*\{",
    r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z)?sh\b",
    # publishing, deploying, cloud and remote systems
    r"\bnpm\s+(publish|unpublish)\b", r"\btwine\s+upload\b", r"\bcargo\s+publish\b", r"\bgem\s+push\b",
    r"\bdocker\s+(push|system\s+prune|volume\s+rm|rm)\b", r"\bkubectl\s+(delete|apply|replace|scale|drain)\b",
    r"\bhelm\s+(uninstall|delete|upgrade)\b", r"\bterraform\s+(apply|destroy)\b",
    r"\baws\s+\S+\s+(delete|rm|terminate|remove)\S*", r"\bgcloud\b[^\n]*\b(delete|destroy)\b",
    r"\baz\b[^\n]*\bdelete\b", r"\bgh\s+(pr\s+merge|release\s+(create|delete)|repo\s+delete)\b",
    r"(?i:\bcurl\b[^\n]*-X\s*(DELETE|PUT|POST|PATCH)\b)",
]
DESTRUCTIVE_TOOL_NAMES = r"(?i)(^|[_\W])(delete|remove|drop|destroy|purge|wipe|cancel|refund|transfer|pay|send|" \
                         r"publish|deploy|merge|force)([_\W]|$)"


class ActionFilter:
    """Cheap deterministic pre-filter: which proposed calls go to the decision model at all."""

    def __init__(self, cfg: PointCfg) -> None:
        self.mode = cfg.prefilter
        self.tools = re.compile("|".join(COMMAND_TOOLS + list(cfg.tools)), re.I)
        self.risky = re.compile("|".join(f"(?:{p})" for p in RISKY_PATTERNS + list(cfg.patterns)))
        self.destructive = re.compile(DESTRUCTIVE_TOOL_NAMES)

    def is_risky(self, c: ToolCall) -> bool:
        return bool(self.destructive.search(c.name) or self.risky.search(c.arguments or ""))

    def candidates(self, calls: list[ToolCall]) -> list[ToolCall]:
        if self.mode == "all":
            return list(calls)
        if self.mode == "command":
            return [c for c in calls if self.tools.search(c.name) or self.is_risky(c)]
        return [c for c in calls if self.is_risky(c)]


# ---------------------------------------------------------------------------------------------- asking
@dataclass
class Ask:
    """One decision call (or a two-round knockout over many tools) and what came of it; one line of decisions.jsonl.
    ``label`` is what the LLM itself chose at this point when that is known, and ``fit_ok`` says whether that choice
    was made without the decision model's influence (so `jah fit` may use it)."""
    point: str
    mode: str
    question: str
    options: list[str]
    tau: float | None = None
    act_option: str | None = None           # the option whose probability is compared with tau
    p: dict[str, float] = field(default_factory=dict)
    model: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0
    rounds: int = 1
    error: str | None = None
    acted: bool = False
    effect: str | None = None
    label: str | None = None
    label_tier: str | None = None
    fit_ok: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    state: str | None = None                # in memory only; written to states.jsonl when log_states is on

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.p)

    @property
    def choice(self) -> str | None:
        return max(self.p, key=lambda k: self.p[k]) if self.p else None

    @property
    def confidence(self) -> float | None:
        return max(self.p.values()) if self.p else None

    def p_of(self, option: str | None = None) -> float:
        return float(self.p.get(option or self.act_option or "", 0.0))

    def record(self) -> dict[str, Any]:
        top = sorted(self.p.items(), key=lambda kv: -kv[1])[:10]
        agree = (self.choice == self.label) if self.p and self.label in self.options else None
        rec: dict[str, Any] = {
            "point": self.point, "mode": self.mode, "question": self.question, "n_options": len(self.options),
            "options": self.options[:300], "p": {k: round(v, 4) for k, v in top}, "choice": self.choice,
            "confidence": None if self.confidence is None else round(self.confidence, 4), "tau": self.tau,
            "act_option": self.act_option, "p_act": round(self.p_of(), 4) if self.act_option and self.p else None,
            "acted": self.acted, "effect": self.effect, "label": self.label, "label_tier": self.label_tier,
            "agree": agree, "fit_ok": self.fit_ok, "latency_s": round(self.latency_s, 4),
            "cost_usd": round(self.cost_usd, 8), "model": self.model or None, "rounds": self.rounds,
            "error": self.error}
        rec.update(self.extra)
        return rec


class Asker:
    """Asks the decision model one question with a time limit and turns any failure into an ``Ask`` with ``error``."""

    def __init__(self, backend: DecisionBackend) -> None:
        self.backend = backend

    async def _decide(self, state: str, q: Question, timeout: float, view: StepView | None) -> Any:
        if isinstance(self.backend, RulesBackend):
            return await self.backend.decide(state, q, view)
        return await asyncio.wait_for(self.backend.decide(state, q), timeout=timeout)

    async def ask(self, point: str, mode: str, state: str, q: Question, timeout: float, tau: float | None = None,
                  act_option: str | None = None, view: StepView | None = None) -> Ask:
        a = Ask(point=point, mode=mode, question=q.id, options=list(q.options), tau=tau, act_option=act_option,
                state=state)
        t0 = time.monotonic()
        try:
            d = await self._decide(state, q, timeout, view)
            a.p = {o: float(d.p.get(o, 0.0)) for o in q.options}
            a.model, a.cost_usd = d.model, float(d.cost_usd or 0.0)
            if not any(a.p.values()):
                raise RuntimeError("the decision model returned no probability for any option")
        except Exception as e:  # noqa: BLE001 - a decision-model failure falls back to the LLM-only behaviour
            a.error = f"{type(e).__name__}: {str(e)[:200]}"
            a.p = {}
        a.latency_s = time.monotonic() - t0
        return a

    async def ask_choice(self, point: str, mode: str, state: str, text: str, options: list[str], descs: list[str],
                         timeout: float, tau: float, max_options: int, keep: int = 5) -> Ask:
        """A choice among options; more than ``max_options`` options -> a knockout: the chunks are asked
        concurrently, the ``keep`` likeliest of each go to one final question. The final question's probabilities
        are logged (renormalized among the finalists)."""
        if len(options) <= max_options:
            return await self.ask(point, mode, state, Question(id="next_tool", text=text, options=options,
                                                               descriptions=descs), timeout, tau)
        keep = max(2, min(keep, max_options - 1))
        n_chunks = -(-len(options) // max_options)
        size = -(-len(options) // n_chunks)
        chunks = [list(range(i, min(i + size, len(options)))) for i in range(0, len(options), size)]
        t0 = time.monotonic()
        firsts = await asyncio.gather(*[self.ask(point, mode, state, Question(
            id="next_tool", text=text, options=[options[i] for i in ch], descriptions=[descs[i] for i in ch]),
            timeout, tau) for ch in chunks])
        bad = next((a for a in firsts if not a.ok), None)
        if bad is not None:
            bad.latency_s, bad.rounds, bad.options = time.monotonic() - t0, 2, list(options)
            bad.cost_usd = sum(a.cost_usd for a in firsts)
            return bad
        finalists: list[str] = []
        for a in firsts:
            finalists += [o for o, _ in sorted(a.p.items(), key=lambda kv: -kv[1])[:keep]]
        if len(finalists) > max_options:      # very many tools: keep the likeliest finalists overall
            pr = {o: p for a in firsts for o, p in a.p.items()}
            finalists = sorted(finalists, key=lambda o: -pr[o])[:max_options]
        if NO_TOOL in options and NO_TOOL not in finalists:
            finalists[-1] = NO_TOOL           # the no-tool option always reaches the final round
        idx = {o: i for i, o in enumerate(options)}
        final = await self.ask(point, mode, state, Question(
            id="next_tool", text=text, options=finalists, descriptions=[descs[idx[o]] for o in finalists]),
            timeout, tau)
        final.options = list(options)
        final.rounds = 2
        final.cost_usd += sum(a.cost_usd for a in firsts)
        final.latency_s = time.monotonic() - t0
        final.extra["finalists"] = len(finalists)
        return final
