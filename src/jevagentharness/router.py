"""The router: one typed decision per agent step picks the tier that writes the step.

Policy (``route``), for step t of a session (t = the number of assistant turns already in the request):

1. the first ``first_steps_strong`` steps go to the strong tier (the agent reads the task and plans);
2. otherwise the decision model reads the session's state (the task, the recent actions and results) and answers
   :data:`QUESTION`: is the next step ``routine`` or ``hard``? The step goes to the strong tier iff p(hard) >= tau;
3. after an escalation the next ``stick_strong`` steps stay on the strong tier; with ``max_fast_streak`` > 0 a strong
   step is forced after that many fast steps in a row;
4. fail-safe: no answer (timeout, error) -> the ``on_error`` tier (strong by default).

``verify_finish`` is applied by the server after the call: a fast-tier response with no tool call ends the agent's
turn, so the strong tier re-does that step from the same request.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from .config import RouterCfg
from .decision import DecisionBackend, Question, RulesBackend, StepView
from .types import Request

QUESTION = Question(
    id="next_step",
    text=("You are watching an AI agent work on the task below, one tool call at a time. Read its latest action and "
          "the result. Decide how hard the agent's NEXT step is. 'routine': the next step follows directly from what "
          "the agent already knows, e.g. read or search more, run a command or test it planned, make an edit it "
          "already worked out. 'hard': the next step needs careful reasoning, e.g. something failed or behaved "
          "unexpectedly, a plan or a fix must be worked out, the agent seems stuck or repeats itself, or it must "
          "judge whether the task is really done."),
    options=["routine", "hard"],
    descriptions=["The next step is routine: a faster, cheaper model will do it as well as a strong one",
                  "The next step needs careful reasoning: debugging, planning, recovering from an error, or "
                  "verifying the result"],
)
HARD = "hard"

TASK_CHARS = 3000
HISTORY_STEPS = 6
ACTION_CHARS = 400
OUTPUT_CHARS = 300
LAST_OUTPUT_CHARS = 2500


def _trunc(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n // 2] + " ... " + s[-(n - n // 2 - 5):]


def _tail(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else "... " + s[-n:]


@dataclass
class Step:
    action: str
    output: str


def episode(req: Request) -> tuple[str, list[Step]]:
    """(task, steps): the first user message, then one step per assistant turn with the tool results after it."""
    turns = req.turns()
    task = next((m.text for m in turns if m.role == "user" and m.text), "")
    steps: list[Step] = []
    for i, m in enumerate(turns):
        if m.role != "assistant":
            continue
        if m.tool_calls:
            action = "; ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
        else:
            action = f"message: {m.text}"
        outs: list[str] = []
        for n in turns[i + 1:]:
            if n.role == "assistant":
                break
            if n.role in ("tool", "user"):   # harnesses that send observations as user messages
                outs.append(n.text)
        steps.append(Step(action=action, output="\n".join(o for o in outs if o)))
    return task, steps


def render_state(req: Request) -> tuple[str, StepView]:
    task, steps = episode(req)
    lines = [f"TASK: {_trunc(task, TASK_CHARS)}", "", f"STEPS SO FAR: {len(steps)}"]
    recent = steps[-HISTORY_STEPS:]
    first = len(steps) - len(recent) + 1
    for j, s in enumerate(recent):
        last = j == len(recent) - 1
        lines.append(f"[{first + j}] ACTION: {_trunc(s.action, ACTION_CHARS)}")
        lines.append(f"    RESULT: {_tail(s.output, LAST_OUTPUT_CHARS) if last else _trunc(s.output, OUTPUT_CHARS)}")
    view = StepView(last_action=steps[-1].action if steps else "", last_output=steps[-1].output if steps else "",
                    prev_action=steps[-2].action if len(steps) > 1 else "")
    return "\n".join(lines), view


def session_key(req: Request, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if req.session:
        return str(req.session)
    task, _ = episode(req)
    h = hashlib.sha256((req.system[:4000] + "\x00" + task[:8000]).encode("utf-8", "replace")).hexdigest()
    return h[:16]


@dataclass
class Session:
    key: str
    created: float = field(default_factory=time.time)
    last_tier: str = ""
    fast_streak: int = 0
    stick_left: int = 0
    decided: dict[int, dict[str, Any]] = field(default_factory=dict)   # step -> route record (reused on a retry)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cost_usd: float = 0.0
    calls: int = 0
    tiers: dict[str, int] = field(default_factory=dict)


class Router:
    def __init__(self, cfg: RouterCfg, backend: DecisionBackend) -> None:
        self.cfg = cfg
        self.backend = backend

    async def route(self, sess: Session, req: Request, cfg: RouterCfg | None = None) -> dict[str, Any]:
        """The route record for this request: {step, tier, reason, p_hard, decision{...}}. ``cfg``: the router
        settings of the client's model name (e.g. another ``tau`` or fast tier); default: the configured router."""
        step = req.step_index()
        if step in sess.decided:
            rec = dict(sess.decided[step])
            rec["reason"] = "repeat:" + str(rec["reason"]).removeprefix("repeat:")
            return rec
        c = cfg or self.cfg
        rec: dict[str, Any] = {"step": step, "p_hard": None}
        if step < c.first_steps_strong:
            tier, reason = c.strong, "first_steps"
        elif sess.stick_left > 0:
            tier, reason = c.strong, "stick"
            sess.stick_left -= 1
        elif c.max_fast_streak > 0 and sess.fast_streak >= c.max_fast_streak:
            tier, reason = c.strong, "fast_streak"
        else:
            state, view = render_state(req)
            rec["_state"] = state                     # kept only in memory; the server logs it when log_states is on
            t0 = time.monotonic()
            try:
                if isinstance(self.backend, RulesBackend):
                    d = await self.backend.decide(state, QUESTION, view)
                else:
                    d = await asyncio.wait_for(self.backend.decide(state, QUESTION), timeout=c.decision_timeout_s)
                p_hard = float(d.p.get(HARD, 0.0))
                rec.update({"p_hard": round(p_hard, 4), "decision": {
                    "model": d.model, "cost_usd": d.cost_usd, "latency_s": round(time.monotonic() - t0, 4),
                    "input_tokens": d.input_tokens, "why": d.why or None, "state_chars": len(state)}})
                if p_hard >= c.tau:
                    tier, reason = c.strong, "hard"
                    sess.stick_left = c.stick_strong
                else:
                    tier, reason = c.fast, "routine"
            except Exception as e:  # noqa: BLE001 - fail safe: a decision-model failure never stops the agent
                tier, reason = c.on_error, "decision_error"
                rec["decision"] = {"error": f"{type(e).__name__}: {str(e)[:200]}",
                                   "latency_s": round(time.monotonic() - t0, 4)}
        self.commit(sess, step, tier, c)
        rec.update({"tier": tier, "reason": reason})
        sess.decided[step] = rec
        for old in [s for s in sess.decided if s < step - 4]:
            sess.decided.pop(old, None)
        return rec

    def commit(self, sess: Session, step: int, tier: str, cfg: RouterCfg | None = None) -> None:
        sess.fast_streak = sess.fast_streak + 1 if tier == (cfg or self.cfg).fast else 0
        sess.last_tier = tier

    def escalated(self, sess: Session, step: int, reason: str, cfg: RouterCfg | None = None) -> None:
        """The server re-did a fast step on the strong tier (finish check, fast-tier error)."""
        sess.fast_streak = 0
        strong = (cfg or self.cfg).strong
        sess.last_tier = strong
        if step in sess.decided:
            sess.decided[step] = {**sess.decided[step], "tier": strong, "reason": reason}
