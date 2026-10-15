"""The JevAgent harness: a standard tool-calling agent loop with a System-1 hook at every
decision point (see ``config.py`` for the WHERE x HOW design space).

With every decision point ``off`` this is a plain function-calling agent -- the LLM-only
baseline. Each paradigm studied in the paper is a config, not a code fork:

  Jev-for-tool-selection   tool_select  in {replace, cascade, assist, guard}
  Jev-for-tool-parameters  tool_params  in {replace, cascade}   (bounded + copy args; free text -> System-2)
  Jev-for-memory           memory_write=trigger, memory_read=assist
  Jev-for-reflection       reflection=trigger
  Jev-as-stop-verifier     stop in {guard, trigger}
  Jev-as-router            routing=trigger   (small vs large LLM per step)
  Jev-action-only          action in {replace, cascade, assist, guard}  (discrete-action envs; guard: E6c)

Exploratory row E6b (2026-10-06; ``config.E6B_CELLS``, results/explore/e6b_modes/cells.md) completes the grid:
tool_params in {assist, guard}, memory_write in {cascade, guard}, memory_read in {cascade, trigger} and a cascade at
reflection, stop and routing. A cascade at a yes / no meta decision uses System-1's answer when its confidence is at
least tau and otherwise lets the LLM decide: the memory LLM gate, an LLM gate asking the same question (reflection;
routing: the cheap LLM), or the LLM's own finish decision (stop).

Exploratory row E6d (``config.E6D_CELLS``, results/explore/e6d_modes/cells.md) adds "advise" assists at the yes / no
points (System-1's answer and confidence become a hint to the LLM, which decides: memory write, reflection, routing via
their LLM gates; stop via a note in the LLM's turn), guards at reflection and routing (the LLM gate proposes, System-1
vetoes a proposal when p_yes < tau) and a memory-read guard (the LLM selects memories, System-1 vetoes once).

Exploratory row E6f (``config.E6F_CELLS``, results/explore/e6f_modes/cells.md) defines the trigger at tool choice, tool
arguments and action: System-1 decides whether the LLM runs one verification call on its own proposal (it re-reads the
state and the proposal and confirms or revises it once); ``always`` (no System-1) verifies every proposal.

``shadow`` (tool_select, tool_params, stop, action, reflection, routing, memory_write) is a measurement mode: the
episode runs exactly as in the LLM-only harness, System-1 is asked the same typed question at the same state and its
answer is logged next to the LLM's own decision and the gold label (``DecisionLog.shadow``), which is what
``bench/threshold_theory.py`` needs to estimate q_L(c) and test Prop. 1. Shadow System-1 calls go through the
ledger tagged ``meta["shadow"]`` and are kept out of the episode cost (see :func:`usage_summary`); a failing
shadow call never affects the episode (fail-open, counted in ``Trajectory.meta["shadow_s1_errors"]``).

Meta-decision golds (docs/META_GOLD_SPEC.md): when the env exposes the oracle hooks (``oracle_verdict``,
``oracle_finish_ok``, ``oracle_mem_gold``, ...; JevGym does), every meta ``DecisionLog`` gets ``gold`` /
``gold_status`` / ``gold_aux`` -- computed on copies of the environment *after* System-1 answered, written only to
the log, never to ``Question.meta``, a rendered state, an LLM message or a memory item. ``HarnessConfig.measure``
adds deferred counterfactual System-2 calls and wrong-proposal probes; they run after ``env.evaluate`` on snapshots
taken at decision time (``harness/measure.py``), so they can never change the episode.
"""

from __future__ import annotations

import copy
import difflib
import json
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..accounting import Ledger
from ..bench.env import Env
from ..decision.base import DecisionBackend, backend_id
from ..decision.types import MAX_OPTIONS, Answer, Choice, Noul, Question
from ..llm.base import ChatResponse, LLMClient
from .config import DP, HarnessConfig, Mode
from .memory import NONE_MEMORY, MemoryStore
from .prompts import (EARLY_STOP_PROMPT, Q_REFLECTION, Q_REFLECTION_V1, Q_ROUTING, Q_STOP_TRIGGER, Q_TOOL_SELECT,
                      REFLECT_PROMPT, REFLECTION_WORDINGS, STOP_REJECT_NOTE, action_reject_note, advice_note,
                      append_note, arg_reject_note, assist_args_hint, llm_gate_prompt, q_action_guard, q_stop_guard,
                      q_tool_guard, q_verify_action, q_verify_args, q_verify_tool, stop_advice_note, tool_reject_note,
                      verify_action_note, verify_args_note, verify_tool_note)
from .tap import Tap, is_shadow, measure_of
from .tools import OTHER, ParamInfo, coerce, extract_candidates, finish_tool, params_of, tool_desc, tool_name
from .trajectory import DecisionLog, StepLog, Trajectory

DEFAULT_SYSTEM = (
    "You are a capable autonomous agent. Solve the user's task by calling the available tools. "
    "Call one tool per turn. When the task is complete, call `finish` with the final answer."
)

ACTION_SYSTEM = (
    "You are an agent acting in a text environment. At each turn choose exactly one action from the "
    "list of admissible actions. Think briefly if needed, then end with a line `Action: <action>` "
    "copied verbatim from the list."
)

# kept for importers of the pre-A4 names (the guard question now carries the proposed answer: ``q_stop_guard``)
Q_STOP_GUARD = q_stop_guard("{answer}")
_Tap = Tap
_is_shadow = is_shadow

GOLD_V = 1                                          # META_GOLD_SPEC version, logged as ``gold_aux["v"]``
FAILING = frozenset({"fatal", "premature", "wrong_answer"})
# retrospective reflection label: the previous observation starts with an error / not-found / empty marker
_RETRO_RE = re.compile(r"^\s*(error\b|irreversible error|no \w+ found|order not found|unknown\b|\[\]\s*$|"
                       r"\(finish rejected|[\w.\-]+: no relevant result)", re.I)


# call id -> provider extras (e.g. Gemini thought signatures) of the calls made in the current episode.
# Episode-local: an episode runs entirely in one runner worker thread, so the map is thread-local and is
# cleared at the start of every run -- extras of calls that never reach the history (finish, guard-rejected
# proposals) can then neither leak memory nor be echoed back under a reused call id in a later episode.
_EXTRAS_TLS = threading.local()


def _call_extras() -> dict[str, dict[str, Any]]:
    d = getattr(_EXTRAS_TLS, "d", None)
    if d is None:
        d = _EXTRAS_TLS.d = {}
    return d


def _shadow_errors() -> list[str]:
    """Episode-local list of failed shadow System-1 calls (thread-local like :func:`_call_extras`)."""
    d = getattr(_EXTRAS_TLS, "shadow_errors", None)
    if d is None:
        d = _EXTRAS_TLS.shadow_errors = []
    return d


def _empty_responses() -> list[str]:
    """Episode-local list of System-2 replies with neither text nor a tool call (thread-local like
    :func:`_call_extras`): the loop reads such a reply as an empty ``finish``, so a provider quirk silently
    ends the episode -- counted in ``Trajectory.meta["empty_llm_responses"]`` to make it visible."""
    d = getattr(_EXTRAS_TLS, "empty", None)
    if d is None:
        d = _EXTRAS_TLS.empty = []
    return d


@dataclass
class StepSnap:
    """What the deferred measurement (``harness/measure.py``) needs about one decision state: an independent copy
    of the env, the System-2 messages before any reflection of the step (a shallow list copy: ``append_note``
    replaces ``messages[-1]``, never mutates it), the S1 state string, the acting LLM, and the step's logs."""
    step: int
    budget: int
    env: Any
    messages: list[dict[str, Any]]
    mem_text: str
    state: str
    llm: Any
    tools: list[dict[str, Any]]
    a0: tuple[str, dict[str, Any]]
    finish: bool = False
    refl: DecisionLog | None = None
    route: DecisionLog | None = None
    stop: DecisionLog | None = None
    stop_gold: dict[str, Any] = field(default_factory=dict)   # {"finish_ok", "world_complete", "d_before", ...}


@dataclass
class _Episode:
    """Per-episode gold / measurement context (thread-local: the runner shares one agent across worker threads)."""
    env: Any = None
    step: int = 0
    budget: int = 0
    verdicts: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    prev: tuple[Any, str] | None = None             # (verdict of the previous executed action, its observation)
    snaps: list[StepSnap] = field(default_factory=list)
    extra_probes: list[tuple[str, DecisionLog, int]] = field(default_factory=list)   # (source, base log, step)
    gold_errors: list[str] = field(default_factory=list)
    cf_errors: list[str] = field(default_factory=list)
    probe_errors: list[str] = field(default_factory=list)
    n_no_valid_probe: int = 0
    n_stop_rejects: int = 0                          # finish proposals the stop guard rejected so far
    delegated: bool = False                          # this step's call was System-1's (``_delegated_call``)
    stop_advice: Any = None                          # E6d stop assist: (question, answer, tap) asked before the step


def _ep() -> _Episode:
    e = getattr(_EXTRAS_TLS, "ep", None)
    if e is None:
        e = _EXTRAS_TLS.ep = _Episode()
    return e


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n // 2] + "\n...[truncated]...\n" + s[-n // 2:]


def _same_value(a: Any, b: Any) -> bool:
    """Loose equality of an argument value and its gold value: numerically when both parse as numbers,
    otherwise case-/whitespace-insensitively as strings (``True`` == ``"true"``)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return str(a).strip().lower() == str(b).strip().lower()
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _parse_action(text: str, actions: list[str]) -> str:
    """The admissible action an action-env LLM reply names: the text after its last ``Action:`` (else the whole
    reply), matched exactly or else to the closest admissible action."""
    if "Action:" in text:
        lines = text.rsplit("Action:", 1)[-1].strip().splitlines()
        guess = lines[0].strip() if lines else ""  # "...Action:" with nothing after it
    else:
        guess = text.strip()
    return guess if guess in actions else (difflib.get_close_matches(guess, actions, n=1, cutoff=0.0) or actions)[0]


def _label(v: Any) -> str | None:
    """An argument value in the option-label space of the tool_params ``Choice`` (JSON for non-strings)."""
    return None if v is None else (v if isinstance(v, str) else json.dumps(v))


def _dnum(d: Any) -> int | None:
    """A cost-to-go for the log: an int, or None for inf / missing (JSON has no inf)."""
    try:
        return None if d is None or d == float("inf") else int(d)
    except (TypeError, ValueError):
        return None


def usage_summary(ledger: Ledger) -> dict[str, Any]:
    """``Ledger.summary()`` of the harness being evaluated. Measurement calls (``meta["shadow"]``: shadow System-1
    questions, deferred counterfactual System-2 calls, probe generation and scoring -- never acted on) stay in the
    per-call log (``Trajectory.calls``) but are kept out of ``total`` / ``s1`` / ``by_decision_point``, so a shadow
    or measurement arm costs exactly what the LLM-only harness costs and every existing cost metric
    (``metrics.episode_table``, the score card) is unchanged. Their overhead is reported as ``usage["shadow"]``
    (the same aggregate plus ``by_decision_point`` and ``by_measure``: shadow / cf / probe), and
    ``usage["total_with_shadow"]`` is what the run actually spent (use it for budgeting)."""
    shadow = [r for r in ledger.records if is_shadow(r)]
    if not shadow:
        return ledger.summary()
    real, sh = Ledger(), Ledger()
    real.records = [r for r in ledger.records if not is_shadow(r)]
    sh.records = shadow
    out = real.summary()
    s = sh.summary()
    by_measure: dict[str, Any] = {}
    for m in sorted({measure_of(r) for r in shadow}):
        sub = Ledger()
        sub.records = [r for r in shadow if measure_of(r) == m]
        by_measure[m] = sub.summary()["total"]
    out["shadow"] = {**s["total"], "by_decision_point": s["by_decision_point"], "by_measure": by_measure}
    out["total_with_shadow"] = ledger.summary()["total"]
    out["unpriced_models"] = sorted(set(out["unpriced_models"]) | set(s["unpriced_models"]))
    return out


class JevAgent:
    def __init__(self, llm: LLMClient, cfg: HarnessConfig, s1: DecisionBackend | None = None,
                 small_llm: LLMClient | None = None, system_prompt: str | None = None,
                 measure_llms: dict[str, LLMClient] | None = None):
        cfg.validate_modes()
        if cfg.uses_s1() and s1 is None:
            raise ValueError(f"config {cfg.name!r} uses System-1 decision points but no backend was given")
        if cfg.routing.mode in (Mode.TRIGGER, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.REPLACE) and small_llm is None:
            raise ValueError(f"routing={cfg.routing.mode.value} needs small_llm")
        self.llm = llm
        self.small_llm = small_llm
        self.s1 = s1
        self.cfg = cfg
        self.system_prompt = system_prompt
        # LLMs of the deferred measurement (``measure.cf.large_llm``, ``measure.probes.transplant_llm``), by name;
        # "main" / "small" are always the arm's own LLM / the routing small LLM
        self.measure_llms = {**({"main": llm} if llm else {}), **({"small": small_llm} if small_llm else {}),
                             **(measure_llms or {})}
        m = cfg.measure
        for nm in ([m.cf.large_llm] if m.cf and m.cf.large_llm else []) + \
                  ([m.probes.transplant_llm] if m.probes and m.probes.transplant_llm else []):
            if nm not in self.measure_llms:
                raise ValueError(f"measure needs LLM {nm!r}; known: {sorted(self.measure_llms)}")

    # ------------------------------------------------------------------ state rendering
    def _s1_state(self, env: Env, history: list[tuple[str, str]], obs: str, memory_text: str = "",
                  reflection: str = "") -> str:
        c = self.cfg
        parts = [f"TASK: {env.instruction}"]
        if memory_text:
            parts.append(f"MEMORY:\n{memory_text}")
        if reflection:  # the LLM keeps its reflections in context; System-1 sees the latest one
            parts.append(f"LATEST REFLECTION:\n{reflection}")
        if history:
            h = history[-c.history_window:]
            start = len(history) - len(h)
            parts.append("HISTORY:\n" + "\n".join(
                f"[{start + i}] {a} -> {_trunc(o, 400)}" for i, (a, o) in enumerate(h)))
        parts.append(f"CURRENT OBSERVATION:\n{_trunc(obs, c.obs_truncate_chars)}")
        return _trunc("\n\n".join(parts), c.s1_state_max_chars)

    # ------------------------------------------------------------------ S1 helpers
    def _ask(self, state: str, q: Question, ledger: Ledger) -> Answer:
        assert self.s1 is not None
        return self.s1.ask(state, q, ledger=ledger)

    def _choice_tournament(self, state: str, text: str, options: list[str], dp: str, ledger: Ledger,
                           descriptions: list[str] | None = None, gold: Any = None, keep: int = 2,
                           ranking: list[str] | None = None) -> Answer:
        """Choice over arbitrarily many options: chunks of <= backend.max_options, the top ``keep`` of each chunk
        re-compete (assist mode passes ``keep = max(2, k)``, so its top-k can come from every chunk: with two
        survivors per chunk the stand-in's 69-tool tournament ended on 4 finalists and "top-5" offered 4 tools).
        When the backend batches (``batched_questions``: Jev, the generative stand-in with ``batch: true``, the free
        controls) all chunk questions of a round go out in ONE request -- they share the state and are answered
        independently -- instead of one sequential call per chunk (E2 dev: the stand-in's 69-tool tournament took 3
        calls, 2.7 s, $0.0029 per decision). ``ranking``, when given, receives System-1's full ranking of
        ``options`` (finalists by the final round, then each earlier round's eliminated options by their score there)
        -- what C3 recall@k ranks; the returned answer's distribution covers the finalists only. Only ``self.s1`` and
        ``self._ask`` are used, so other harnesses (``bench/suites/tau2.py``) can call this unbound."""
        assert self.s1 is not None
        m = self.s1.max_options
        keep = max(1, min(max(2, int(keep)), m - 1))       # < m, so every round shrinks the pool
        if len(options) <= m:
            q = Choice(text, options, descriptions, decision_point=dp, gold=gold)
            return self._ask(state, q, ledger)
        desc = dict(zip(options, descriptions)) if descriptions else {}
        batched = bool(getattr(self.s1, "batched_questions", False))
        pool = list(options)
        eliminated: list[list[str]] = []       # per round, best first
        rnd = 0
        while len(pool) > m:
            chunks = [pool[i:i + m] for i in range(0, len(pool), m)]
            asks: list[tuple[int, Question]] = []
            for ci, chunk in enumerate(chunks):
                if len(chunk) > keep:
                    d = [desc.get(o, "") for o in chunk] if desc else None
                    asks.append((ci, Choice(text, chunk, d, decision_point=dp,
                                            gold=gold if gold in chunk else None)))
            if not asks:       # every chunk already fits the survivors: nothing left to eliminate this way
                break
            n0 = len(ledger.records)
            if batched and len(asks) > 1:
                answers = self.s1.decide(state, [q for _, q in asks], ledger=ledger).answers
            else:
                answers = [self._ask(state, q, ledger) for _, q in asks]
            for rec in ledger.records[n0:]:       # A6 ruling (e): the round of every chunk call
                rec.tournament_round = rnd
            rnd += 1
            top = {ci: a.top_k(keep) for (ci, _), a in zip(asks, answers)}
            out = sorted(((p, o) for (ci, _), a in zip(asks, answers) for o, p in a.distribution.items()
                          if o not in top[ci]), key=lambda x: -x[0])
            seen = {o for _, o in out}
            out_names = [o for _, o in out] + [o for ci, _ in asks for o in chunks[ci]
                                                if o not in top[ci] and o not in seen]
            eliminated.append(out_names)
            pool = [o for ci, chunk in enumerate(chunks) for o in (top[ci] if ci in top else chunk)]
        d = [desc.get(o, "") for o in pool] if desc else None
        n0 = len(ledger.records)
        a = self._ask(state, Choice(text, pool, d, decision_point=dp, gold=gold), ledger)
        for rec in ledger.records[n0:]:
            rec.tournament_round = rnd                 # the final round
        if ranking is not None:
            fin = [o for o in a.top_k(len(pool))] + [o for o in pool if o not in a.distribution]
            ranking[:] = fin + [o for rnd in reversed(eliminated) for o in rnd]
        return a

    def _shadow_ask(self, fn: Any, ledger: Ledger, what: str) -> tuple[Answer | None, Tap]:
        """Run a shadow System-1 query ``fn(tap)`` fail-open: a measurement must never change the episode, so an
        S1 error is recorded (``Trajectory.meta["shadow_s1_errors"]``) and the decision is logged without S1."""
        tap = Tap(ledger, shadow=True)
        try:
            return fn(tap), tap
        except Exception as e:  # noqa: BLE001 - fail-open by design
            _shadow_errors().append(f"{what}: {type(e).__name__}: {str(e)[:200]}")
            return None, tap

    # ------------------------------------------------------------------ measurement helpers
    @staticmethod
    def _s2_spent(ledger: Ledger, n0: int) -> tuple[float, float]:
        """(cost, latency) of the LLM calls logged since record ``n0``: the calls behind one decision."""
        rs = [r for r in ledger.records[n0:] if r.component == "s2" and not is_shadow(r)]
        return sum(r.cost_usd for r in rs), sum(r.latency_ms for r in rs)

    @staticmethod
    def _step_s2_spent(ledger: Ledger, dps: tuple[str, ...]) -> tuple[float, float]:
        """(cost, latency) of this step's LLM calls at the given decision points."""
        rs = [r for r in ledger.records if r.step == ledger.step and r.component == "s2" and not is_shadow(r)
              and r.decision_point in dps]
        return sum(r.cost_usd for r in rs), sum(r.latency_ms for r in rs)

    @staticmethod
    def _measure(dlog: DecisionLog, tap: Tap | None = None, s2: tuple[float, float] | None = None,
                 s2_correct: bool | None = None) -> DecisionLog:
        if tap is not None:
            tap.stamp(dlog)
        if s2 is not None:
            dlog.s2_cost_usd, dlog.s2_latency_ms = s2
        if s2_correct is not None:
            dlog.s2_correct = s2_correct
        return dlog

    @staticmethod
    def _proposal_gold(env: Env, name: str, args: dict[str, Any], gold_tool: str | None) -> bool | None:
        """Legacy verdict on a *proposed* call (what ``Question.meta`` carries for guard decisions, so simulated
        backends and simulation studies are unchanged): ``env.oracle_accepts(tool, args)`` when the env implements
        it (the tool *and* its arguments must be the canonical next step), else whether the tool is the oracle's."""
        acc = getattr(env, "oracle_accepts", None)
        if callable(acc):
            try:
                return acc(name, args)
            except Exception:  # noqa: BLE001 - a gold label must never crash the episode
                pass
        return (name == gold_tool) if gold_tool else None

    @staticmethod
    def _arg_correct(env: Env, name: str, gold_tool: str | None, oracle_args: dict[str, Any] | None,
                     pname: str, value: Any, gold: Any) -> bool | None:
        """Is the LLM's value of argument ``pname`` right? Judged by ``env.oracle_accepts`` on the gold call with
        this one argument replaced (tolerances such as refund amounts, answers containing the gold number),
        else by :func:`_same_value`. None without a gold value; a missing argument is wrong."""
        if gold is None:
            return None
        if value is None:
            return False
        acc = getattr(env, "oracle_accepts", None)
        if callable(acc) and name == gold_tool and oracle_args is not None:
            try:
                r = acc(name, {**oracle_args, pname: value})
                if r is not None:
                    return bool(r)
            except Exception:  # noqa: BLE001
                pass
        return _same_value(value, gold)

    # ------------------------------------------------------------------ gold helpers (META_GOLD_SPEC §3)
    def _env_tool(self, env: Env, name: str) -> str:
        """The env's name of the harness's final tool (the oracle hooks know it as ``env.FINAL_TOOL``)."""
        return str(getattr(env, "FINAL_TOOL", name)) if name == self.cfg.final_tool else name

    def _budget(self) -> int:
        """The remaining step budget B = max_steps - t of the current episode (the full budget outside ``run``)."""
        ep = _ep()
        return ep.budget if ep.env is not None else self.cfg.max_steps

    def _verdict(self, env: Env, name: str, args: Any) -> dict[str, Any] | None:
        """verdict(s_t, a) at the current state (cached per step); None when the env has no oracle."""
        fn = getattr(env, "oracle_verdict", None)
        if not callable(fn):
            return None
        ep = _ep()
        key = (ep.step, name, json.dumps(args, sort_keys=True, default=str))
        if key not in ep.verdicts:
            try:
                ep.verdicts[key] = fn(self._env_tool(env, name), args if isinstance(args, dict) else {},
                                      self._budget())
            except Exception as e:  # noqa: BLE001 - a gold label never crashes the episode
                ep.gold_errors.append(f"verdict: {type(e).__name__}: {str(e)[:200]}")
                ep.verdicts[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        return ep.verdicts[key]

    @staticmethod
    def _need_gold(v: dict[str, Any] | None, fired: bool = False) -> tuple[Any, str, dict[str, Any]]:
        """need_fatal of the System-2 action a0 at this state (reflection, routing; §3.1-3.2)."""
        if v is None:
            return None, "no_oracle", {"v": GOLD_V}
        if "error" in v:
            return None, "no_oracle", {"v": GOLD_V, "error": v["error"]}
        if v.get("verdict") is None:
            return None, "doomed", {"v": GOLD_V, "d_before": None}
        if fired:  # the executed action came after the intervention: a0 is not observed here
            return None, "unobserved", {"v": GOLD_V, "d_before": v.get("d_before"),
                                        "verdict_executed": v.get("verdict")}
        verdict = v["verdict"]
        aux = {"v": GOLD_V, "verdict": verdict, "need_any": verdict != "progress", "d_before": v.get("d_before"),
               "verdict_src": "executed"}
        if v.get("fatal_kind"):      # A6 / P0-8: terminal (proven) vs budget (oracle-relative) need_fatal positives
            aux["fatal_kind"] = v["fatal_kind"]
        return verdict in FAILING, "ok", aux

    def _proposal_verdict(self, env: Env, name: str, args: dict[str, Any],
                          legacy: Any) -> tuple[Any, str, dict[str, Any]]:
        """Tool-guard gold: ``proposal_progress`` = verdict(s, proposal) == progress (§3.5); the legacy canonical
        verdict stays in ``gold_aux.canonical``. Envs without the hook keep the legacy gold."""
        v = self._verdict(env, name, args)
        if v is None or "error" in v:
            aux = {"v": GOLD_V, "src": "oracle_accepts", **({"error": v["error"]} if v else {})}
            return legacy, ("ok" if legacy is not None else "no_oracle"), aux
        aux = {"v": GOLD_V, "verdict": v.get("verdict"), "canonical": legacy, "d_before": v.get("d_before"),
               "d_after": v.get("d_after")}
        if v.get("fatal_kind"):
            aux["fatal_kind"] = v["fatal_kind"]
        if v.get("verdict") is None:
            return None, "doomed", aux
        return v["verdict"] == "progress", "ok", aux

    def _stop_gold(self, env: Env, stopping: bool, answer: Any, legacy: Any) -> tuple[Any, str, dict[str, Any]]:
        """Stop gold (§3.3-3.4): ``finish_ok`` for a finish proposal (finishing now with this answer succeeds),
        ``world_complete`` at other steps (only the answer remains); the legacy plan-order ``oracle_done`` stays in
        ``gold_aux.canonical_done``. Envs without the hooks keep the legacy gold."""
        fok = getattr(env, "oracle_finish_ok", None)
        if not callable(fok):
            return legacy, ("ok" if legacy is not None else "no_oracle"), {"v": GOLD_V, "src": "oracle_done"}
        ep = _ep()
        try:
            d0 = env.oracle_cost_to_go(self._budget())   # type: ignore[attr-defined]
            wc = bool(env.oracle_world_complete())       # type: ignore[attr-defined]
            aux: dict[str, Any] = {"v": GOLD_V, "d_before": _dnum(d0), "canonical_done": legacy,
                                   "world_complete": wc}
            if _dnum(d0) is None:
                return None, "doomed", aux
            if stopping:
                ok = bool(fok("" if answer is None else str(answer)))
                aux["kind"] = "ok" if ok else ("wrong_answer" if wc else "premature")
                return ok, "ok", aux
            aux["cut_losses"] = False      # not recoverable == doomed, excluded above (kept for L3 suites)
            return wc, "ok", aux
        except Exception as e:  # noqa: BLE001
            ep.gold_errors.append(f"stop: {type(e).__name__}: {str(e)[:200]}")
            return None, "no_oracle", {"v": GOLD_V, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    def _choice_gold(self, env: Env, dlog: DecisionLog) -> None:
        """tool_select Choice records keep the canonical tool as ``gold`` (continuity) and get the information-
        feasible acceptable tool set AT(s) (primary for tool-level accuracy, C3 recall@k, H1) in ``gold_aux``."""
        fn = getattr(env, "oracle_acceptable_tool_sets", None)
        aux: dict[str, Any] = {"v": GOLD_V}
        status = "ok" if dlog.gold is not None else "no_oracle"
        if callable(fn):
            ep = _ep()
            try:
                if _dnum(env.oracle_cost_to_go(self._budget())) is None:     # type: ignore[attr-defined]
                    dlog.gold, status = None, "doomed"
                    aux.update(acceptable_tools=[], acceptable_tools_unfiltered=[])
                else:
                    fin = str(getattr(env, "FINAL_TOOL", self.cfg.final_tool))
                    f, u = fn(self._budget())
                    aux["acceptable_tools"] = [self.cfg.final_tool if t == fin else t for t in f]
                    aux["acceptable_tools_unfiltered"] = [self.cfg.final_tool if t == fin else t for t in u]
            except Exception as e:  # noqa: BLE001
                ep.gold_errors.append(f"acceptable_tools: {type(e).__name__}: {str(e)[:200]}")
                aux["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        dlog.gold_aux, dlog.gold_status = aux, status

    def _patch_meta_golds(self, env: Env, slog: StepLog, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Back-patch the reflection / routing golds of this step from the System-2 action a0 = (name, args) at
        s_t (before ``env.step``): ``need_fatal`` = a0 is failing; fired trigger steps are ``unobserved``."""
        ep = _ep()
        gates = [d for d in slog.decisions if d.dp in (DP.REFLECTION.value, DP.ROUTING.value)
                 and d.mode in (Mode.TRIGGER.value, Mode.SHADOW.value, Mode.CASCADE.value, Mode.ASSIST.value,
                                Mode.GUARD.value, Mode.REPLACE.value)]
        need_v = gates or self.cfg.reflection.mode in (Mode.TRIGGER, Mode.SHADOW, Mode.CASCADE, Mode.ASSIST,
                                                       Mode.GUARD, Mode.REPLACE) or self.cfg.measure.active()
        v = self._verdict(env, name, args) if need_v else None
        for d in gates:
            # trigger / cascade: a fired gate (reflected, or routed to the strong LLM) changed a0: unobserved
            fired = d.mode in (Mode.TRIGGER.value, Mode.CASCADE.value, Mode.ASSIST.value, Mode.GUARD.value,
                               Mode.REPLACE.value) and (
                d.final_value is True if d.dp == DP.REFLECTION.value else d.final_value == "large")
            g, st, aux = self._need_gold(v, fired)
            if d.dp == DP.REFLECTION.value and st == "ok":
                pv, pobs = ep.prev or (None, "")
                aux["retro"] = bool((pv is not None and pv.get("verdict") not in (None, "progress"))
                                    or _RETRO_RE.search((pobs or "").splitlines()[0] if pobs else ""))
            d.gold, d.gold_status, d.gold_aux = g, st, aux
        return v

    def _memory_sim_meta(self, env: Env, text: str) -> dict[str, Any] | None:
        """``memory_write.extra.sim_gold``: simulation studies only -- the gate's ``Question.meta`` carries mem_need
        (the env after the step is known at offer time). Reflection / routing have no such option: their gold is a
        verdict on the action System-2 takes AFTER the question is asked."""
        fn = getattr(env, "oracle_mem_gold", None)
        if not self.cfg.memory_write.extra.get("sim_gold") or not callable(fn):
            return None
        try:
            return {"gold": bool(fn(text, ())["mem_need"])}
        except Exception:  # noqa: BLE001
            return None

    def _memory_write_gold(self, env: Env, logs: list[DecisionLog], text: str, known: list[str]) -> None:
        """``mem_need`` of the offered text on the env after the step (§3.6); ``far`` / ``new`` in ``gold_aux``."""
        fn = getattr(env, "oracle_mem_gold", None)
        for d in logs:
            aux = dict(d.gold_aux or {"v": GOLD_V})
            if not callable(fn):
                d.gold_status, d.gold_aux = "no_oracle", aux
                continue
            ep = _ep()
            try:
                if _dnum(env.oracle_cost_to_go(max(self._budget() - 1, 0))) is None:  # type: ignore[attr-defined]
                    d.gold, d.gold_status, d.gold_aux = None, "doomed", aux
                    continue
                g = fn(text, known)
                aux.update(far=g["far"], new=g["new"], atoms=g["atoms"])
                d.gold, d.gold_status, d.gold_aux = bool(g["mem_need"]), "ok", aux
            except Exception as e:  # noqa: BLE001
                ep.gold_errors.append(f"memory_write: {type(e).__name__}: {str(e)[:200]}")
                d.gold_status, d.gold_aux = "no_oracle", {**aux, "error": str(e)[:200]}

    def _memory_read_gold(self, env: Env, memory: MemoryStore, logs: list[DecisionLog],
                          pool: list[Any]) -> None:
        """Labels of the stored items holding an input of the next oracle step, or the "none" option (§3.7)."""
        fn = getattr(env, "oracle_memory_read_gold", None)
        for d in logs:
            aux = dict(d.gold_aux or {"v": GOLD_V})
            if not callable(fn):
                d.gold_status, d.gold_aux = "no_oracle", aux
                continue
            try:
                r = fn([m.text for m in pool])
                strict = sorted(memory.label(pool[i]) for i in r["strict"]) or [NONE_MEMORY]
                lenient = sorted(memory.label(pool[i]) for i in r["lenient"]) or [NONE_MEMORY]
                aux["lenient"] = lenient
                if d.mode == Mode.TRIGGER.value:    # E6b read trigger: does a stored item hold a next-step input?
                    aux["labels"] = strict
                    d.gold, d.gold_status, d.gold_aux = strict != [NONE_MEMORY], "ok", aux
                    continue
                if d.mode == Mode.GUARD.value:      # E6d read guard: does the LLM's selection hold every needed item?
                    aux["labels"] = strict
                    sel = set(d.s2_value or [])
                    d.gold = strict == [NONE_MEMORY] or all(x in sel for x in strict)
                    aux["final_covers"] = strict == [NONE_MEMORY] or all(x in set(d.final_value or []) for x in strict)
                    d.gold_status, d.gold_aux = "ok", aux
                    continue
                d.gold, d.gold_status, d.gold_aux = strict, "ok", aux
            except Exception as e:  # noqa: BLE001
                _ep().gold_errors.append(f"memory_read: {type(e).__name__}: {str(e)[:200]}")
                d.gold_status, d.gold_aux = "no_oracle", {**aux, "error": str(e)[:200]}

    # ------------------------------------------------------------------ LLM helpers
    def _llm_call(self, llm: LLMClient, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                  ledger: Ledger, dp: str, slog: StepLog, **kw: Any) -> ChatResponse:
        slog.llm_calls += 1
        slog.model = llm.model
        return llm.chat(messages, tools=tools, ledger=ledger, decision_point=dp, **kw)

    def _llm_gate(self, llm: LLMClient, state: str, question: str, ledger: Ledger, dp: str,
                  slog: StepLog, advice: str | None = None) -> tuple[bool, tuple[float, float]]:
        """E6b: the LLM answers a yes / no meta question itself on System-1's state (the deferral of a reflection or
        routing cascade; the memory-write gate's pattern): one call, no tools, through the ledger at ``dp``. Returns
        (yes, (cost, latency))."""
        n0 = len(ledger.records)
        r = self._llm_call(llm, [{"role": "user", "content": llm_gate_prompt(state, question, advice)}], None, ledger,
                           dp, slog, max_tokens=16, temperature=0.0)
        return (r.text or "").strip().lower().startswith("y"), self._s2_spent(ledger, n0)

    def _gate_with_s1(self, mode: Mode, q: Question, state: str, gate_llm: LLMClient, tau: float, ledger: Ledger,
                      slog: StepLog) -> tuple[bool, DecisionLog]:
        """E6d at a yes / no meta decision with an LLM gate (reflection, routing). ``assist``: System-1 answers ``q``,
        and the LLM gate answers it with System-1's answer and confidence as advice; the LLM's answer is executed.
        ``guard``: the LLM gate proposes; a yes is checked by System-1 and vetoed when p_yes < tau (not executed); a
        no needs no check. Returns (executed yes?, the decision log; values as booleans)."""
        dp = str(q.decision_point)
        if mode == Mode.ASSIST:
            tap = Tap(ledger)
            a = self._ask(state, q, tap)
            yes, s2 = self._llm_gate(gate_llm, state, q.text, ledger, dp, slog,
                                     advice=advice_note(bool(a.value), a.confidence))
            dlog = self._measure(DecisionLog(dp, "assist", q.text, s1_value=bool(a.value), s1_conf=a.confidence,
                                             s1_dist=a.distribution, deferred=True, s2_value=yes, final_value=yes,
                                             state=state), tap, s2)
            return yes, dlog
        yes, s2 = self._llm_gate(gate_llm, state, q.text, ledger, dp, slog)
        dlog = self._measure(DecisionLog(dp, "guard", q.text, deferred=True, s2_value=yes, state=state), s2=s2)
        if yes:                        # a "no" proposal invokes nothing: there is nothing to veto
            tap = Tap(ledger)
            a = self._ask(state, q, tap)
            yes = a.p_yes >= tau
            dlog.s1_value, dlog.s1_conf, dlog.s1_dist, dlog.retry = yes, a.confidence, a.distribution, 0
            self._measure(dlog, tap)
        dlog.final_value = yes
        return yes, dlog

    @staticmethod
    def _resp_to_call(resp: ChatResponse, final_tool: str) -> tuple[str, dict[str, Any], str]:
        """Returns (tool, args, call_id). Provider extras of the call (e.g. Gemini thought signatures)
        are remembered under the call id so the history can echo them back verbatim."""
        if resp.tool_calls:
            tc = resp.tool_calls[0]
            if tc.extra:
                _call_extras()[tc.id] = tc.extra
            else:  # never inherit a stale entry stored under a reused id
                _call_extras().pop(tc.id, None)
            return tc.name, tc.arguments, tc.id
        if not (resp.text or "").strip():
            _empty_responses().append(str(resp.finish_reason))
        return final_tool, {"answer": resp.text}, f"call_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------ decision points
    def _route(self, state: str, ledger: Ledger, slog: StepLog) -> LLMClient:
        c = self.cfg.routing
        if c.mode not in (Mode.TRIGGER, Mode.SHADOW, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.REPLACE):
            return self.llm
        q = Noul(Q_ROUTING, decision_point=DP.ROUTING.value)
        if c.mode in (Mode.ASSIST, Mode.GUARD):   # E6d: the cheap LLM's own gate, advised or guarded by System-1
            big, dlog = self._gate_with_s1(c.mode, q, state, self.small_llm, c.tau, ledger, slog)  # type: ignore[arg-type]
            dlog.s2_value = "large" if dlog.s2_value else "small"
            dlog.final_value = "large" if big else "small"
            slog.decisions.append(dlog)
            return self.llm if big else self.small_llm  # type: ignore[return-value]
        if c.mode == Mode.SHADOW:  # default resolver: always the main LLM; S1's answer is only logged
            a, tap = self._shadow_ask(lambda led: self._ask(state, q, led), ledger, "routing")
            slog.decisions.append(self._measure(DecisionLog(
                DP.ROUTING.value, "shadow", q.text, s1_value=a.value if a else None,
                s1_conf=a.confidence if a else None, s1_dist=a.distribution if a else None, deferred=True,
                final_value="large", state=state, shadow=True), tap))
            return self.llm
        tap = Tap(ledger)
        a = self._ask(state, q, tap)
        if c.mode == Mode.CASCADE:    # E6b: System-1's call when sure, else the cheap LLM asks itself the question
            dlog = self._measure(DecisionLog(DP.ROUTING.value, "cascade", q.text, s1_value=bool(a.value),
                                             s1_conf=a.confidence, s1_dist=a.distribution,
                                             deferred=a.confidence < c.tau, state=state), tap)
            big = bool(a.value)
            if dlog.deferred:
                big, s2 = self._llm_gate(self.small_llm, state, Q_ROUTING, ledger, DP.ROUTING.value, slog)  # type: ignore[arg-type]
                dlog.s2_value = "large" if big else "small"
                self._measure(dlog, s2=s2)
            dlog.final_value = "large" if big else "small"
            slog.decisions.append(dlog)
            return self.llm if big else self.small_llm  # type: ignore[return-value]
        big = a.p_yes >= (c.tau if c.mode == Mode.TRIGGER else 0.5)   # E6e replace: the argmax (= trigger at 0.5)
        slog.decisions.append(self._measure(DecisionLog(DP.ROUTING.value, c.mode.value, q.text, s1_value=big,
                                                        s1_conf=a.confidence, s1_dist=a.distribution,
                                                        final_value="large" if big else "small", state=state), tap))
        return self.llm if big else self.small_llm  # type: ignore[return-value]

    def _reflect(self, llm: LLMClient, env: Env, state: str, messages: list[dict[str, Any]], ledger: Ledger,
                 slog: StepLog, step: int) -> None:
        c = self.cfg.reflection
        if c.mode == Mode.OFF or step == 0:
            return
        if c.mode == Mode.SHADOW:  # default resolver: never reflect; S1's answer is only logged
            wordings = [w for w in (c.extra.get("wordings") or ["v2"]) if w in REFLECTION_WORDINGS] or ["v2"]
            q = Noul(REFLECTION_WORDINGS[wordings[0]], decision_point=DP.REFLECTION.value)
            a, tap = self._shadow_ask(lambda led: self._ask(state, q, led), ledger, "reflection")
            dlog = self._measure(DecisionLog(
                DP.REFLECTION.value, "shadow", q.text, s1_value=a.value if a else None,
                s1_conf=a.confidence if a else None, s1_dist=a.distribution if a else None, deferred=True,
                final_value=False, state=state, shadow=True), tap)
            slog.decisions.append(dlog)
            for w in wordings[1:]:      # other wordings: probe-list records asked after the episode (§3.1)
                _ep().extra_probes.append((f"wording_{w}", dlog, step))
            return
        fire = c.mode == Mode.ALWAYS
        if c.mode in (Mode.ASSIST, Mode.GUARD):   # E6d: the LLM's own gate, advised or guarded by System-1
            fire, dlog = self._gate_with_s1(c.mode, Noul(Q_REFLECTION, decision_point=DP.REFLECTION.value), state,
                                            llm, c.tau, ledger, slog)
            slog.decisions.append(dlog)
        if c.mode == Mode.CASCADE:    # E6b: System-1's yes / no when sure, else the LLM asks itself the question
            q = Noul(Q_REFLECTION, decision_point=DP.REFLECTION.value)
            tap = Tap(ledger)
            a = self._ask(state, q, tap)
            dlog = self._measure(DecisionLog(DP.REFLECTION.value, "cascade", q.text, s1_value=bool(a.value),
                                             s1_conf=a.confidence, s1_dist=a.distribution,
                                             deferred=a.confidence < c.tau, state=state), tap)
            fire = bool(a.value)
            if dlog.deferred:
                fire, s2 = self._llm_gate(llm, state, Q_REFLECTION, ledger, DP.REFLECTION.value, slog)
                dlog.s2_value = fire
                self._measure(dlog, s2=s2)
            dlog.final_value = fire
            slog.decisions.append(dlog)
        if c.mode in (Mode.TRIGGER, Mode.REPLACE):   # E6e replace: the argmax alone (= the trigger at 0.5)
            q = Noul(Q_REFLECTION, decision_point=DP.REFLECTION.value)
            tap = Tap(ledger)
            a = self._ask(state, q, tap)
            fire = a.p_yes >= (c.tau if c.mode == Mode.TRIGGER else 0.5)
            slog.decisions.append(self._measure(DecisionLog(DP.REFLECTION.value, c.mode.value, q.text,
                                                            s1_value=fire, s1_conf=a.confidence,
                                                            s1_dist=a.distribution, final_value=fire,
                                                            state=state), tap))
        if fire:
            r = self._llm_call(llm, messages + [{"role": "user", "content": REFLECT_PROMPT}],
                               None, ledger, DP.REFLECTION.value, slog, temperature=0.0)
            slog.reflection = r.text
            append_note(messages, f"[Reflection] {r.text}")

    def _param_kinds(self, env: Env | None) -> dict[str, str]:
        """Rule-1 fill-kind overrides (free-text policy): the suite's table (``env.param_kinds``), then the
        experiment's (``tool_params.extra.param_kinds``, e.g. the coverage valve's reclassifications) and the H3
        ablation ``tool_params.extra.answer_as_copy`` (``finish.answer`` copied from the state)."""
        kinds = dict(getattr(env, "param_kinds", None) or {})
        extra = self.cfg.tool_params.extra
        kinds.update(extra.get("param_kinds") or {})
        if extra.get("answer_as_copy"):
            kinds[f"{self.cfg.final_tool}.answer"] = "copy"
        return kinds

    def _param_options(self, p: ParamInfo, state: str, gold: Any,
                       cache: dict[str, Any]) -> tuple[list[str], dict[str, Any], Any] | None:
        """The ``Choice`` System-1 answers for argument ``p``: (option labels, label -> typed value for bounded
        arguments, gold label), or None when ``p`` is not S1-fillable (free text, or nothing to copy)."""
        kind = p.kind
        if kind == "free":
            return None
        bounded = p.bounded_values() if kind == "bounded" else None
        if bounded is not None:
            opts = [json.dumps(v) if not isinstance(v, str) else v for v in bounded]
            gold_s = (json.dumps(gold) if not isinstance(gold, str) else gold) if gold is not None else None
            return opts, dict(zip(opts, bounded)), gold_s  # option label -> original typed value
        if kind == "copy" and self.cfg.tool_params.extra.get("copy", True):
            if "candidates" not in cache:
                # instruction values first (half the slots), then the most recent spans: see extract_candidates
                cache["candidates"] = extract_candidates(
                    state, task_first=int(self.cfg.tool_params.extra.get("task_candidates", (MAX_OPTIONS - 1) // 2)))
            candidates = cache["candidates"]
            if not candidates:
                return None
            opts = candidates[: MAX_OPTIONS - 1] + [OTHER]
            gold_s = str(gold) if gold is not None and str(gold) in opts else (OTHER if gold is not None else None)
            return opts, {}, gold_s
        return None

    @staticmethod
    def _param_question(tool: dict[str, Any], p: ParamInfo, opts: list[str], gold_s: Any) -> Question:
        return Choice(f"Which value should the argument `{p.name}` of tool `{tool_name(tool)}` take? "
                      f"({p.schema.get('description', '')})", opts, decision_point=DP.TOOL_PARAMS.value,
                      gold=gold_s)

    def _fill_params(self, env: Env | None, tool: dict[str, Any], state: str, ledger: Ledger, slog: StepLog,
                     oracle_args: dict[str, Any] | None) -> tuple[dict[str, Any], list[str], dict[str, DecisionLog]]:
        """S1 fills bounded / copy params (never free text). Returns (filled args, names still needing the LLM,
        the decision logs of the arguments S1 deferred -- their final value is the LLM's, filled in by the caller)."""
        c = self.cfg.tool_params
        infos = params_of(tool, self._param_kinds(env))
        if c.mode in (Mode.OFF, Mode.SHADOW):  # shadow: the LLM fills every argument (S1 is logged afterwards)
            return {}, [p.name for p in infos], {}
        filled: dict[str, Any] = {}
        todo: list[str] = []
        open_logs: dict[str, DecisionLog] = {}  # deferred bounded / copy arguments, by parameter name
        cache: dict[str, Any] = {}
        asks: list[tuple[ParamInfo, list[str], dict[str, Any], Any, Question]] = []
        for p in infos:
            gold = (oracle_args or {}).get(p.name)
            po = self._param_options(p, state, gold, cache)
            if po is None or not po[0]:
                todo.append(p.name)
                continue
            opts, value_of, gold_s = po
            if len(opts) < 2:
                filled[p.name] = self._typed(value_of[opts[0]], p) if value_of else opts[0]
                continue
            asks.append((p, opts, value_of, gold_s, self._param_question(tool, p, opts, gold_s)))
        # ``tool_params.extra.batch_s1`` (the exploratory final harness, docs/JEVAGENT_HARNESS.md §8): the argument
        # questions of one call share the state and are answered independently, so a batching backend answers all of
        # them in ONE request (the server prefills the state once and scores every question's suffix in one pass);
        # every answer, threshold test and log is the per-question path's. Off (the registered call graph, A6.22 (a)):
        # one request per question, in parameter order.
        batch = bool(c.extra.get("batch_s1")) and len(asks) > 1 and bool(getattr(self.s1, "batched_questions", False))
        answers: list[tuple[Answer, Tap]] = []
        if batch:
            assert self.s1 is not None
            tap = Tap(ledger)
            res = self.s1.decide(state, [x[4] for x in asks], ledger=tap)
            answers = [(a, tap) for a in res.answers]
        for i, (p, opts, value_of, gold_s, q) in enumerate(asks):
            if batch:
                a, tap = answers[i]
            else:
                tap = Tap(ledger)
                a = self._ask(state, q, tap)
            ok = a.value != OTHER and (c.mode == Mode.REPLACE or a.confidence >= c.tau)
            aux: dict[str, Any] = {"v": GOLD_V, "param_kind": p.kind}
            if batch:
                aux["s1_batch"] = {"size": len(asks), "index": i}
            dlog = self._measure(DecisionLog(
                DP.TOOL_PARAMS.value, c.mode.value, q.text, opts, a.value, a.confidence, a.distribution,
                deferred=not ok, gold=gold_s, final_value=a.value if ok else None, state=state,
                gold_status="ok" if gold_s is not None else None, gold_aux=aux), tap)
            if batch:
                # one request answered the whole batch: each member logs the request's latency (what it waited) and
                # an equal share of its cost, so the decision costs still sum to the ledger's System-1 cost
                dlog.s1_cost_usd = (dlog.s1_cost_usd or 0.0) / len(asks)
            slog.decisions.append(dlog)
            if ok:
                filled[p.name] = self._typed(value_of[a.value], p) if value_of else coerce(a.value, p)
            else:
                todo.append(p.name)
                open_logs[p.name] = dlog
        return filled, todo, open_logs

    def _shadow_params(self, env: Env, tool: dict[str, Any], args: dict[str, Any], state: str, ledger: Ledger,
                       slog: StepLog, oracle: tuple[str, dict[str, Any]] | None, s2: tuple[float, float]) -> None:
        """tool_params=shadow: after the LLM filled the arguments of the executed call, ask System-1 the Choice it
        would answer in cascade / replace mode for every S1-fillable (bounded / copy) argument, and log both
        answers. As in cascade mode, gold arguments exist only when the executed tool is the oracle's tool."""
        name = tool_name(tool)
        gold_tool = oracle[0] if oracle else None
        oracle_args = oracle[1] if oracle and name == gold_tool else None
        cache: dict[str, Any] = {}
        for p in params_of(tool, self._param_kinds(env)):
            gold = (oracle_args or {}).get(p.name)
            po = self._param_options(p, state, gold, cache)
            if po is None or len(po[0]) < 2:
                continue
            opts, _, gold_s = po
            q = self._param_question(tool, p, opts, gold_s)
            a, tap = self._shadow_ask(lambda led, q=q: self._ask(state, q, led), ledger, f"tool_params:{p.name}")
            v = args.get(p.name)
            slog.decisions.append(self._measure(DecisionLog(
                DP.TOOL_PARAMS.value, "shadow", q.text, opts, a.value if a else None, a.confidence if a else None,
                a.distribution if a else None, deferred=True, s2_value=_label(v), final_value=_label(v),
                gold=gold_s, state=state, shadow=True, gold_status="ok" if gold_s is not None else None,
                gold_aux={"v": GOLD_V, "param_kind": p.kind}), tap, s2,
                self._arg_correct(env, name, gold_tool, oracle_args, p.name, v, gold)))

    @staticmethod
    def _typed(v: Any, p: Any) -> Any:
        """A bounded value as the tool expects it: non-string enum members / booleans / ints keep their
        JSON type (mixed enums such as [null, "asc"] included); string members are coerced to the declared
        type as before."""
        return coerce(v, p) if isinstance(v, str) else v

    def _tool_step(self, llm: LLMClient, env: Env, tools: list[dict[str, Any]], messages: list[dict[str, Any]],
                   state: str, ledger: Ledger, slog: StepLog) -> tuple[str, dict[str, Any], str]:
        oracle = env.oracle_action()
        n0 = len(ledger.records)
        _ep().delegated = False
        name, args, cid = self._tool_decision(llm, env, tools, messages, state, ledger, slog, oracle)
        if self.cfg.tool_params.mode == Mode.SHADOW and isinstance(args, dict):
            by_name = {tool_name(t): t for t in tools}
            if name in by_name:
                self._shadow_params(env, by_name[name], args, state, ledger, slog, oracle,
                                    self._s2_spent(ledger, n0))
        elif self.cfg.tool_params.mode == Mode.GUARD and isinstance(args, dict) and not _ep().delegated:
            name, args, cid = self._guard_params(llm, env, tools, messages, state, ledger, slog, oracle,
                                                 name, args, cid)
        elif self.cfg.tool_params.mode in (Mode.TRIGGER, Mode.ALWAYS) and isinstance(args, dict) \
                and not _ep().delegated:    # E6f: the argument trigger / always-verify on the LLM's own call
            name, args, cid = self._verify_call(llm, env, tools, messages, state, ledger, slog, oracle,
                                                name, args, cid, DP.TOOL_PARAMS)
        return name, args, cid

    @staticmethod
    def _value_label(v: Any, opts: list[str]) -> str | None:
        """The option label of an argument value the LLM wrote: its label when it is an option, ``OTHER`` for a copy
        value outside the candidates, None for a missing value or a bounded value outside the value set."""
        lab = _label(v)
        if lab is not None and lab in opts:
            return lab
        return OTHER if v is not None and OTHER in opts else None

    def _guard_params(self, llm: LLMClient, env: Env, tools: list[dict[str, Any]], messages: list[dict[str, Any]],
                      state: str, ledger: Ledger, slog: StepLog, oracle: tuple[str, dict[str, Any]] | None,
                      name: str, args: dict[str, Any], cid: str) -> tuple[str, dict[str, Any], str]:
        """E6b ``tool_params: guard``: the LLM wrote the call; System-1 answers the Choice of every bounded / copy
        argument (the question cascade and shadow ask) and vetoes the call when it is confident (>= tau) of another
        value than the LLM's (never on its ``OTHER`` answer). A veto names the most confidently disputed argument, the
        LLM rewrites the call once (``guard_max_retries``) and the rewritten call is executed unchecked, as the tool
        guard does. Gold: the argument's Choice gold, only when the proposed tool is the oracle's tool."""
        c = self.cfg.tool_params
        by_name = {tool_name(t): t for t in tools}
        gold_tool = oracle[0] if oracle else None
        for retry in range(self.cfg.guard_max_retries):
            if name not in by_name or not isinstance(args, dict):
                break
            tool = by_name[name]
            oracle_args = oracle[1] if oracle and name == gold_tool else None
            cache: dict[str, Any] = {}
            checks: list[tuple[DecisionLog, str, float]] = []
            for p in params_of(tool, self._param_kinds(env)):
                gold = (oracle_args or {}).get(p.name)
                po = self._param_options(p, state, gold, cache)
                if po is None or len(po[0]) < 2:
                    continue
                opts, _, gold_s = po
                v = args.get(p.name)
                lab = self._value_label(v, opts)
                q = self._param_question(tool, p, opts, gold_s)
                tap = Tap(ledger)
                a = self._ask(state, q, tap)
                dispute = a.value != OTHER and a.value != lab and a.confidence >= c.tau
                dlog = self._measure(DecisionLog(
                    DP.TOOL_PARAMS.value, "guard", q.text, opts, a.value, a.confidence, a.distribution,
                    deferred=True, s2_value=lab, final_value=lab, gold=gold_s, state=state,
                    gold_status="ok" if gold_s is not None else None, retry=retry,
                    gold_aux={"v": GOLD_V, "param_kind": p.kind, "dispute": dispute, "veto": False}), tap, None,
                    self._arg_correct(env, name, gold_tool, oracle_args, p.name, v, gold))
                slog.decisions.append(dlog)
                checks.append((dlog, p.name, a.confidence if dispute else -1.0))
            disputed = [x for x in checks if x[2] >= 0]
            if not disputed:
                break
            veto = max(disputed, key=lambda x: x[2])
            veto[0].gold_aux = {**(veto[0].gold_aux or {}), "veto": True}
            n1 = len(ledger.records)
            resp = self._llm_call(llm, messages + [{"role": "user", "content": arg_reject_note(name, args, veto[1])}],
                                  tools, ledger, DP.TOOL_PARAMS.value, slog)
            name2, args2, cid = self._resp_to_call(resp, self.cfg.final_tool)
            for d, pn, _ in checks:    # the executed value of every checked argument (None if the tool changed)
                d.final_value = _label(args2.get(pn)) if name2 == name and isinstance(args2, dict) else None
                d.gold_aux = {**(d.gold_aux or {}), "retry_tool": name2}
            veto[0].s2_cost_usd, veto[0].s2_latency_ms = self._s2_spent(ledger, n1)
            name, args = name2, args2
        return name, args, cid

    def _tool_decision(self, llm: LLMClient, env: Env, tools: list[dict[str, Any]],
                       messages: list[dict[str, Any]], state: str, ledger: Ledger, slog: StepLog,
                       oracle: tuple[str, dict[str, Any]] | None) -> tuple[str, dict[str, Any], str]:
        c = self.cfg.tool_select
        gold_tool = oracle[0] if oracle else None
        names = [tool_name(t) for t in tools]
        by_name = dict(zip(names, tools))
        slog.oracle_tool = gold_tool
        dlog: DecisionLog | None = None

        rank: list[str] = []
        if c.mode == Mode.SHADOW:  # ask exactly what cascade would ask; the LLM decides below
            a, tap = self._shadow_ask(lambda led: self._choice_tournament(
                state, Q_TOOL_SELECT, names, DP.TOOL_SELECT.value, led,
                [tool_desc(t) for t in tools], gold_tool, ranking=rank), ledger, "tool_select")
            dlog = self._measure(DecisionLog(
                DP.TOOL_SELECT.value, "shadow", "next tool", names, a.value if a else None,
                a.confidence if a else None, a.distribution if a else None, deferred=True, gold=gold_tool,
                state=state, shadow=True, s1_rank=list(rank) if a is not None and rank else None), tap)
            self._choice_gold(env, dlog)
            slog.decisions.append(dlog)
        elif c.mode == Mode.CASCADE and c.extra.get("always_defer"):
            # PREREGISTRATION A6.11 (c): the plug-in policy accepts no confidence level at this decision point, so
            # System-1 is not queried (no cost, no latency) and every decision is the LLM's; logged as a deferred
            # cascade decision without a System-1 answer so the arm's records keep their structure
            dlog = DecisionLog(DP.TOOL_SELECT.value, c.mode.value, "next tool", names, None, None, None,
                               deferred=True, gold=gold_tool, state=state)
            self._choice_gold(env, dlog)
            dlog.gold_aux = {**(dlog.gold_aux or {}), "always_defer": True}
            slog.decisions.append(dlog)
        elif c.mode in (Mode.REPLACE, Mode.CASCADE, Mode.ASSIST):
            tap = Tap(ledger)
            # assist keeps max(2, k) survivors per tournament chunk, so its top-k is a genuine top-k
            a = self._choice_tournament(state, Q_TOOL_SELECT, names,
                                        DP.TOOL_SELECT.value, tap, [tool_desc(t) for t in tools], gold_tool,
                                        keep=c.k if c.mode == Mode.ASSIST else 2, ranking=rank)
            dlog = self._measure(DecisionLog(DP.TOOL_SELECT.value, c.mode.value, "next tool", names, a.value,
                                             a.confidence, a.distribution, gold=gold_tool, state=state,
                                             s1_rank=list(rank) or None), tap)
            self._choice_gold(env, dlog)
            slog.decisions.append(dlog)
            if c.mode == Mode.ASSIST:
                keep = a.top_k(c.k)
                if self.cfg.final_tool not in keep:
                    keep.append(self.cfg.final_tool)
                n1 = len(ledger.records)
                resp = self._llm_call(llm, messages, [by_name[n] for n in keep if n in by_name], ledger,
                                      DP.TOOL_SELECT.value, slog)
                name, args, cid = self._resp_to_call(resp, self.cfg.final_tool)
                dlog.s2_value = dlog.final_value = name
                dlog.gold_aux = {**(dlog.gold_aux or {}), "offered": [n for n in keep if n in by_name]}
                self._measure(dlog, s2=self._s2_spent(ledger, n1),
                              s2_correct=(name == gold_tool) if gold_tool else None)
                return name, args, cid
            # E15: a refused question (OpenAI Decisions API, ``raw["refusal"]``) is no answer: never executed, even in
            # replace mode, so the LLM decides (no other backend sets the flag, so no logged row changes)
            refused = bool((a.raw or {}).get("refusal"))
            if not refused and (c.mode == Mode.REPLACE or a.confidence >= c.tau):
                name = a.value
                dlog.final_value = name
                return self._delegated_call(llm, env, by_name, name, messages, state, ledger, slog, oracle)
            dlog.deferred = True  # cascade (or a refusal): fall through to the LLM
            if refused:
                dlog.gold_aux = {**(dlog.gold_aux or {}), "s1_refusal": True}

        n1 = len(ledger.records)
        resp = self._llm_call(llm, messages, tools, ledger, DP.TOOL_SELECT.value, slog)
        name, args, cid = self._resp_to_call(resp, self.cfg.final_tool)
        s2_ok = (name == gold_tool) if gold_tool else None
        if dlog is not None:  # cascade (deferred) or shadow
            dlog.s2_value = dlog.final_value = name
            self._measure(dlog, s2=self._s2_spent(ledger, n1), s2_correct=s2_ok)
        elif c.mode == Mode.OFF:
            off = DecisionLog(DP.TOOL_SELECT.value, "off", "next tool", names, s2_value=name, final_value=name,
                              gold=gold_tool, state=state)
            self._choice_gold(env, off)
            slog.decisions.append(self._measure(off, s2=self._s2_spent(ledger, n1), s2_correct=s2_ok))
        if c.mode == Mode.GUARD:
            for retry in range(self.cfg.guard_max_retries):
                # Question.meta keeps the legacy canonical verdict (simulated backends); the log gets the
                # acceptable-action gold: is the proposed call (tool and arguments) progress? (§3.5)
                legacy = self._proposal_gold(env, name, args, gold_tool)
                q = Noul(q_tool_guard(name, args), decision_point=DP.TOOL_SELECT.value, gold=legacy)
                if c.extra.get("sim_gold"):
                    q.meta["gold"] = self._proposal_verdict(env, name, args, legacy)[0]
                tap = Tap(ledger)
                a = self._ask(state, q, tap)
                ok = a.p_yes >= c.tau
                g, st, aux = self._proposal_verdict(env, name, args, legacy)
                slog.decisions.append(self._measure(DecisionLog(
                    DP.TOOL_SELECT.value, "guard", q.text, None, ok, a.confidence, a.distribution, s2_value=name,
                    final_value=name if ok else None, gold=g, state=state, gold_status=st, gold_aux=aux,
                    retry=retry), tap))
                if ok:
                    break
                resp = self._llm_call(llm, messages + [{"role": "user", "content": tool_reject_note(name, args)}],
                                      tools, ledger, DP.TOOL_SELECT.value, slog)
                name, args, cid = self._resp_to_call(resp, self.cfg.final_tool)
        if c.mode in (Mode.TRIGGER, Mode.ALWAYS):    # E6f: the tool-choice trigger / always-verify
            name, args, cid = self._verify_call(llm, env, tools, messages, state, ledger, slog, oracle,
                                                name, args if isinstance(args, dict) else {}, cid, DP.TOOL_SELECT)
        return name, args, cid

    # ------------------------------------------------------------------ E6f: the verification pass
    def _tool_ok(self, env: Env, name: str, gold_tool: str | None) -> tuple[bool | None, str]:
        """Tool-level gold of a proposed call: its tool is in the information-feasible acceptable set AT(s) (the
        tool-choice gold of ``_choice_gold``), else (envs without the hook) the oracle's tool. (value, status)."""
        fn = getattr(env, "oracle_acceptable_tool_sets", None)
        if callable(fn):
            try:
                if _dnum(env.oracle_cost_to_go(self._budget())) is None:     # type: ignore[attr-defined]
                    return None, "doomed"
                feasible, _ = fn(self._budget())
                return self._env_tool(env, name) in set(feasible), "ok"
            except Exception as e:  # noqa: BLE001 - a gold label never crashes the episode
                _ep().gold_errors.append(f"verify_tool: {type(e).__name__}: {str(e)[:200]}")
                return None, "no_oracle"
        return ((name == gold_tool), "ok") if gold_tool else (None, "no_oracle")

    @staticmethod
    def _verify_outcome(ok0: bool | None, ok1: bool | None, changed: bool, fired: bool) -> str | None:
        """What one verification did to a proposal, judged by the gold: fixed (wrong -> right), broke (right -> wrong),
        confirmed_right / confirmed_wrong (unchanged), revised_right / revised_wrong (changed, same correctness).
        None when it did not run or there is no gold."""
        if not fired or ok0 is None or ok1 is None:
            return None
        if not changed:
            return "confirmed_right" if ok0 else "confirmed_wrong"
        if ok0 != ok1:
            return "fixed" if ok1 else "broke"
        return "revised_right" if ok1 else "revised_wrong"

    def _verify_call(self, llm: LLMClient, env: Env, tools: list[dict[str, Any]], messages: list[dict[str, Any]],
                     state: str, ledger: Ledger, slog: StepLog, oracle: tuple[str, dict[str, Any]] | None,
                     name: str, args: dict[str, Any], cid: str, dp: DP) -> tuple[str, dict[str, Any], str]:
        """E6f, ``tool_select`` / ``tool_params`` in {trigger, always}: the LLM proposed the call (name, args). Trigger:
        System-1 is asked whether the proposal needs a second look (``q_verify_tool`` / ``q_verify_args``) and the
        verification runs iff p_yes >= tau; always: it runs on every proposal (no System-1). The verification is ONE
        LLM call with the proposal named in a check note (``verify_tool_note``: every tool offered;
        ``verify_args_note``: forced to the proposed tool), and the call it returns is executed; a reply without a
        call leaves the proposal standing. The argument point checks calls of a known tool that has parameters.
        Gold (log only, never on the wire): tool choice -- the tool is in AT(s); arguments -- the call is progress
        (``_proposal_verdict``, the tool guard's gold); ``gold`` = the proposal needs a second look (it is wrong);
        ``gold_aux`` carries both levels before and after and the outcome (``_verify_outcome``)."""
        c = self.cfg.dp(dp)
        by_name = {tool_name(t): t for t in tools}
        if dp == DP.TOOL_PARAMS and (name not in by_name or not params_of(by_name[name], self._param_kinds(env))):
            return name, args, cid           # nothing to check: an unknown tool, or a tool without arguments
        gold_tool = oracle[0] if oracle else None
        args = args if isinstance(args, dict) else {}
        qtext = (q_verify_tool if dp == DP.TOOL_SELECT else q_verify_args)(name, args)
        dlog = DecisionLog(dp.value, c.mode.value, qtext, None, deferred=True, state=state,
                           s2_value=name if dp == DP.TOOL_SELECT else json.dumps(args, sort_keys=True, default=str))
        fire = True
        if c.mode == Mode.TRIGGER:
            # Question.meta (simulated backends only, never on the wire): the legacy canonical verdict, negated
            if dp == DP.TOOL_SELECT:
                legacy = (name != gold_tool) if gold_tool else None
            else:
                pg = self._proposal_gold(env, name, args, gold_tool)
                legacy = (not pg) if pg is not None else None
            q = Noul(qtext, decision_point=dp.value, gold=legacy)
            tap = Tap(ledger)
            a = self._ask(state, q, tap)
            fire = a.p_yes >= c.tau
            dlog.s1_value, dlog.s1_conf, dlog.s1_dist = fire, a.confidence, a.distribution
            self._measure(dlog, tap)
        name2, args2, cid2, reply = name, args, cid, None
        if fire:
            n1 = len(ledger.records)
            if dp == DP.TOOL_SELECT:
                resp = self._llm_call(llm, messages + [{"role": "user", "content": verify_tool_note(name, args)}],
                                      tools, ledger, dp.value, slog)
            else:
                resp = self._llm_call(llm, messages + [{"role": "user", "content": verify_args_note(name, args)}],
                                      [by_name[name]], ledger, dp.value, slog,
                                      tool_choice={"type": "function", "function": {"name": name}})
            if resp.tool_calls:
                name2, args2, cid2 = self._resp_to_call(resp, self.cfg.final_tool)
                args2 = args2 if isinstance(args2, dict) else {}
                reply = "call"
            else:                            # no call: the proposal stands (never read as a finish)
                reply = "text" if (resp.text or "").strip() else "empty"
            dlog.s2_cost_usd, dlog.s2_latency_ms = self._s2_spent(ledger, n1)
        dlog.final_value = name2 if dp == DP.TOOL_SELECT else json.dumps(args2, sort_keys=True, default=str)
        # golds at s_t (before env.step), on the proposal and on the executed call
        t0, ts0 = self._tool_ok(env, name, gold_tool)
        p0, ps0, aux0 = self._proposal_verdict(env, name, args, self._proposal_gold(env, name, args, gold_tool))
        same = name2 == name and json.dumps(args2, sort_keys=True, default=str) == json.dumps(args, sort_keys=True,
                                                                                               default=str)
        if same:
            t1, p1, aux1 = t0, p0, aux0
        else:
            t1, _ = self._tool_ok(env, name2, gold_tool)
            p1, _, aux1 = self._proposal_verdict(env, name2, args2, self._proposal_gold(env, name2, args2, gold_tool))
        ok0, st0, ok1 = (t0, ts0, t1) if dp == DP.TOOL_SELECT else (p0, ps0, p1)
        changed = (name2 != name) if dp == DP.TOOL_SELECT else not same
        dlog.gold = (not ok0) if ok0 is not None else None
        dlog.gold_status = st0 if ok0 is not None or st0 != "ok" else "no_oracle"
        dlog.gold_aux = {"v": GOLD_V, "fired": fire, "changed": changed, "changed_call": not same,
                         "proposal_ok": ok0, "final_ok": ok1, "outcome": self._verify_outcome(ok0, ok1, changed, fire),
                         "tool_ok0": t0, "tool_ok1": t1, "call_ok0": p0, "call_ok1": p1,
                         "verdict0": aux0.get("verdict"), "verdict1": aux1.get("verdict"),
                         "outcome_call": self._verify_outcome(p0, p1, not same, fire), "verify_reply": reply,
                         "proposal": [name, args], **({"final": [name2, args2]} if not same else {})}
        slog.decisions.append(dlog)
        return name2, args2, cid2

    def _verify_action(self, llm: LLMClient, actions: list[str], messages: list[dict[str, Any]], listing: str,
                       state: str, ledger: Ledger, slog: StepLog, gold: Any, act: str, text: str) -> tuple[str, str]:
        """E6f, ``action`` in {trigger, always}: the LLM proposed the admissible action ``act``. Trigger: System-1 is
        asked whether it needs a second look (``q_verify_action``) and the verification runs iff p_yes >= tau;
        always: on every proposal. The verification is ONE LLM call: the action prompt with the check note
        (``verify_action_note``) after the admissible list; the action its reply names is executed (an empty reply
        leaves the proposal standing). Gold (log only): the proposal equals the env's single gold action, a lower
        bound of "right" (as the action guard); ``gold`` = the proposal needs a second look (it is not the gold)."""
        c = self.cfg.action
        g0 = _same_value(act, gold) if gold is not None else None
        qtext = q_verify_action(act)
        dlog = DecisionLog(DP.ACTION.value, c.mode.value, qtext, None, deferred=True, s2_value=act, state=state)
        fire = True
        if c.mode == Mode.TRIGGER:
            q = Noul(qtext, decision_point=DP.ACTION.value, gold=(not g0) if g0 is not None else None)
            tap = Tap(ledger)
            a = self._ask(state, q, tap)
            fire = a.p_yes >= c.tau
            dlog.s1_value, dlog.s1_conf, dlog.s1_dist = fire, a.confidence, a.distribution
            self._measure(dlog, tap)
        act2, text2, reply = act, text, None
        if fire:
            n1 = len(ledger.records)
            resp = self._llm_call(llm, messages + [{"role": "user", "content": f"Admissible actions:\n{listing}\n\n"
                                                                                f"{verify_action_note(act)}"}],
                                  None, ledger, DP.ACTION.value, slog, temperature=0.0)
            t2 = resp.text or ""
            if t2.strip():
                act2, text2, reply = _parse_action(t2, actions), t2, "action"
            else:
                reply = "empty"
            dlog.s2_cost_usd, dlog.s2_latency_ms = self._s2_spent(ledger, n1)
        g1 = _same_value(act2, gold) if gold is not None else None
        dlog.final_value = act2
        dlog.gold = (not g0) if g0 is not None else None
        dlog.gold_status = "ok" if g0 is not None else "no_oracle"
        dlog.gold_aux = {"v": GOLD_V, "src": "oracle_action", "gold_action": gold, "fired": fire,
                         "changed": act2 != act, "proposal_ok": g0, "final_ok": g1,
                         "outcome": self._verify_outcome(g0, g1, act2 != act, fire), "verify_reply": reply}
        slog.decisions.append(dlog)
        return act2, text2

    def _delegated_call(self, llm: LLMClient, env: Env, by_name: dict[str, dict[str, Any]], name: str,
                        messages: list[dict[str, Any]], state: str, ledger: Ledger, slog: StepLog,
                        oracle: tuple[str, dict[str, Any]] | None) -> tuple[str, dict[str, Any], str]:
        """System-1 chose ``name`` (replace, or cascade above tau): System-1 fills the bounded / copy arguments;
        System-2 fills every free-text argument -- required or optional -- and every required argument System-1
        left open, in ONE forced call (``tool_choice = name``; System-1's values win on conflict). Free text is
        never filled by System-1 (META_GOLD_SPEC §7: a Choice whose gold lies outside its options has VoI <= 0;
        in the E2 dev run System-1 copied the heartbeat number 899659 as a final answer).

        The forced call offers the one chosen tool by default. ``tool_params.extra.full_tools`` (off by default; no
        registered arm sets it) offers the step's whole tool list in its order instead, still forced to ``name``, so
        the request keeps the prompt prefix of the LLM's own tool-choice calls and a provider's prefix cache can hit
        (E13 diagnosis, 2026-10-07: with GPT-6-Luna the one-tool calls got 0.7 % cache hits against LLM-only's
        95.5 %)."""
        _ep().delegated = True
        if self.cfg.tool_params.mode == Mode.ASSIST:
            return self._assisted_call(llm, env, by_name, name, messages, state, ledger, slog, oracle)
        gold_tool = oracle[0] if oracle else None
        # the oracle's arguments are gold only for the oracle's tool (never for another tool sharing a parameter name)
        filled, todo, open_logs = self._fill_params(env, by_name[name], state, ledger, slog,
                                                    oracle[1] if oracle and name == gold_tool else None)
        infos = params_of(by_name[name], self._param_kinds(env))
        free = [p.name for p in infos if p.kind == "free"]
        need = [p.name for p in infos if p.name in todo and (p.required or p.kind == "free")]
        if not need:
            return name, filled, f"call_{uuid.uuid4().hex[:8]}"
        hint = (f"You must now call `{name}`. Already-determined arguments: {json.dumps(filled)}. "
                "Provide the remaining arguments.")
        n1 = len(ledger.records)
        offer = list(by_name.values()) if self.cfg.tool_params.extra.get("full_tools") else [by_name[name]]
        resp = self._llm_call(llm, messages + [{"role": "user", "content": hint}], offer, ledger,
                              DP.TOOL_PARAMS.value, slog, tool_choice={"type": "function", "function": {"name": name}})
        if resp.tool_calls:
            _, args, cid = self._resp_to_call(resp, self.cfg.final_tool)
        else:  # the LLM ignored the forced call: its text is not an argument of `name` (except a final answer)
            args, cid = ({"answer": resp.text} if name == self.cfg.final_tool and "answer" in free else {},
                         f"call_{uuid.uuid4().hex[:8]}")
        args = {**args, **filled}
        for pn, dlog in open_logs.items():  # deferred below tau: System-2's value is the final value
            dlog.s2_value = dlog.final_value = _label(args.get(pn))
        if self.cfg.tool_params.mode in (Mode.REPLACE, Mode.CASCADE):
            cost = self._s2_spent(ledger, n1)
            for pn in free:                 # free slots: generation, never delegated (delegation ratio: LLM-made)
                v = args.get(pn)
                slog.decisions.append(self._measure(DecisionLog(
                    DP.TOOL_PARAMS.value, "generate", f"free-text argument `{pn}` of tool `{name}`", s1_value=None,
                    deferred=True, s2_value=_label(v), final_value=_label(v), state=state,
                    gold_aux={"v": GOLD_V, "param_kind": "free"}), s2=cost))
        return name, args, cid

    def _assisted_call(self, llm: LLMClient, env: Env, by_name: dict[str, dict[str, Any]], name: str,
                       messages: list[dict[str, Any]], state: str, ledger: Ledger, slog: StepLog,
                       oracle: tuple[str, dict[str, Any]] | None) -> tuple[str, dict[str, Any], str]:
        """E6b ``tool_params: assist``: System-1 chose ``name``; for each bounded / copy argument it answers the
        argument's Choice (the question cascade asks) and its top k values become a shortlist; the LLM fills the
        whole call in ONE forced call (``tool_choice = name``), every argument its own. A string argument's shortlist
        is enforced by an ``enum`` in the offered schema, other types' by the instruction; a shortlist that contains
        ``OTHER`` (or every option) restricts nothing. Single-option bounded arguments are filled by rule, as in
        cascade / replace."""
        c = self.cfg.tool_params
        gold_tool = oracle[0] if oracle else None
        oracle_args = oracle[1] if oracle and name == gold_tool else None
        tool = by_name[name]
        spec = copy.deepcopy(tool)
        props = ((spec["function"].get("parameters") or {}).get("properties") or {})
        infos = params_of(tool, self._param_kinds(env))
        cache: dict[str, Any] = {}
        fixed: dict[str, Any] = {}
        logs: list[tuple[DecisionLog, ParamInfo, Any]] = []
        shortlists: list[tuple[str, list[Any], bool]] = []
        for p in infos:
            gold = (oracle_args or {}).get(p.name)
            po = self._param_options(p, state, gold, cache)
            if po is None or not po[0]:
                continue
            opts, value_of, gold_s = po
            if len(opts) < 2:
                fixed[p.name] = self._typed(value_of[opts[0]], p) if value_of else opts[0]
                continue
            q = self._param_question(tool, p, opts, gold_s)
            tap = Tap(ledger)
            a = self._ask(state, q, tap)
            top = a.top_k(c.k)
            open_ = OTHER in top
            vals = [self._typed(value_of[o], p) if value_of else coerce(o, p) for o in top if o != OTHER]
            binding = not open_ and len(top) < len(opts)
            if vals and (binding or open_):
                shortlists.append((p.name, vals, open_))
                if binding and all(isinstance(v, str) for v in vals) and p.name in props:
                    props[p.name] = {**props[p.name], "enum": vals}
            dlog = self._measure(DecisionLog(
                DP.TOOL_PARAMS.value, "assist", q.text, opts, a.value, a.confidence, a.distribution, deferred=True,
                gold=gold_s, state=state, gold_status="ok" if gold_s is not None else None,
                gold_aux={"v": GOLD_V, "param_kind": p.kind, "offered": top, "binding": binding}), tap)
            slog.decisions.append(dlog)
            logs.append((dlog, p, gold))
        n1 = len(ledger.records)
        resp = self._llm_call(llm, messages + [{"role": "user", "content": assist_args_hint(name, shortlists, fixed)}],
                              [spec], ledger, DP.TOOL_PARAMS.value, slog,
                              tool_choice={"type": "function", "function": {"name": name}})
        free = [p.name for p in infos if p.kind == "free"]
        if resp.tool_calls:
            _, args, cid = self._resp_to_call(resp, self.cfg.final_tool)
        else:  # the LLM ignored the forced call: its text is not an argument of `name` (except a final answer)
            args, cid = ({"answer": resp.text} if name == self.cfg.final_tool and "answer" in free else {},
                         f"call_{uuid.uuid4().hex[:8]}")
        args = {**args, **fixed}
        cost = self._s2_spent(ledger, n1)
        for dlog, p, gold in logs:
            v = args.get(p.name)
            dlog.s2_value = dlog.final_value = _label(v)
            lab = self._value_label(v, list(dlog.options or []))
            dlog.gold_aux = {**(dlog.gold_aux or {}), "in_shortlist": lab is not None and (
                lab in (dlog.gold_aux or {}).get("offered", []) or OTHER in (dlog.gold_aux or {}).get("offered", []))}
            self._measure(dlog, s2=cost, s2_correct=self._arg_correct(env, name, gold_tool, oracle_args, p.name, v, gold))
        for pn in free:                 # free slots: generation, never delegated (as in replace / cascade)
            v = args.get(pn)
            slog.decisions.append(self._measure(DecisionLog(
                DP.TOOL_PARAMS.value, "generate", f"free-text argument `{pn}` of tool `{name}`", s1_value=None,
                deferred=True, s2_value=_label(v), final_value=_label(v), state=state,
                gold_aux={"v": GOLD_V, "param_kind": "free"}), s2=cost))
        return name, args, cid

    def _action_step(self, llm: LLMClient, env: Env, actions: list[str], messages: list[dict[str, Any]],
                     state: str, ledger: Ledger, slog: StepLog) -> tuple[str, str]:
        """Discrete-action envs. Returns (action, raw LLM text or '')."""
        c = self.cfg.action
        oracle = env.oracle_action()
        gold = oracle[1].get("action") if oracle else None
        dlog: DecisionLog | None = None
        if c.mode == Mode.SHADOW:  # ask exactly what cascade would ask; the LLM decides below
            a, tap = self._shadow_ask(lambda led: self._choice_tournament(
                state, "Which action should the agent take next?", actions, DP.ACTION.value, led, gold=gold),
                ledger, "action")
            dlog = self._measure(DecisionLog(
                DP.ACTION.value, "shadow", "next action", list(actions), a.value if a else None,
                a.confidence if a else None, a.distribution if a else None, deferred=True, gold=gold, state=state,
                shadow=True), tap)
            slog.decisions.append(dlog)
        elif c.mode in (Mode.REPLACE, Mode.CASCADE, Mode.ASSIST):
            tap = Tap(ledger)
            a = self._choice_tournament(state, "Which action should the agent take next?", actions,
                                        DP.ACTION.value, tap, gold=gold)
            dlog = self._measure(DecisionLog(DP.ACTION.value, c.mode.value, "next action", actions, a.value,
                                             a.confidence, a.distribution, gold=gold, state=state), tap)
            slog.decisions.append(dlog)
            if c.mode == Mode.REPLACE or (c.mode == Mode.CASCADE and a.confidence >= c.tau):
                dlog.final_value = a.value
                return a.value, ""
            if c.mode == Mode.ASSIST:
                actions = a.top_k(c.k)
            else:
                dlog.deferred = True
        listing = "\n".join(f"- {x}" for x in actions)
        n1 = len(ledger.records)
        resp = self._llm_call(llm, messages + [{"role": "user", "content": f"Admissible actions:\n{listing}"}],
                              None, ledger, DP.ACTION.value, slog, temperature=0.0)
        text = resp.text or ""
        act = _parse_action(text, actions)
        s2_ok = _same_value(act, gold) if gold is not None else None
        if dlog is not None:  # cascade (deferred), assist or shadow
            dlog.s2_value = dlog.final_value = act
            self._measure(dlog, s2=self._s2_spent(ledger, n1), s2_correct=s2_ok)
        elif c.mode == Mode.OFF:
            slog.decisions.append(self._measure(
                DecisionLog(DP.ACTION.value, "off", "next action", list(actions), s2_value=act, final_value=act,
                            gold=gold, state=state), s2=self._s2_spent(ledger, n1), s2_correct=s2_ok))
        if c.mode == Mode.GUARD:
            act, text = self._guard_action(llm, actions, messages, listing, state, ledger, slog, gold, act, text)
        if c.mode in (Mode.TRIGGER, Mode.ALWAYS):    # E6f: the action trigger / always-verify
            act, text = self._verify_action(llm, actions, messages, listing, state, ledger, slog, gold, act, text)
        return act, text

    def _guard_action(self, llm: LLMClient, actions: list[str], messages: list[dict[str, Any]], listing: str,
                      state: str, ledger: Ledger, slog: StepLog, gold: Any, act: str, text: str) -> tuple[str, str]:
        """E6c ``action: guard``, the tool guard at the action point: the LLM proposed ``act``; System-1 is asked the
        tool-guard question about it (``q_action_guard``) and vetoes it when p_yes < tau. A vetoed proposal is
        regenerated once (``guard_max_retries``): the same prompt with the rejection note (``action_reject_note``)
        after the admissible list, and the regenerated action is executed unchecked, as the tool guard does. Gold
        (only when the env exposes ``oracle_action``): the proposal equals the single gold action, a lower bound of
        "right" (many states admit several good actions); ``Question.meta`` carries it as the tool guard's legacy
        gold (never sent on the wire)."""
        c = self.cfg.action
        for retry in range(self.cfg.guard_max_retries):
            g = _same_value(act, gold) if gold is not None else None
            q = Noul(q_action_guard(act), decision_point=DP.ACTION.value, gold=g)
            tap = Tap(ledger)
            a = self._ask(state, q, tap)
            ok = a.p_yes >= c.tau
            dlog = self._measure(DecisionLog(
                DP.ACTION.value, "guard", q.text, None, ok, a.confidence, a.distribution, s2_value=act,
                final_value=act if ok else None, gold=g, state=state,
                gold_status="ok" if g is not None else "no_oracle",
                gold_aux={"v": GOLD_V, "src": "oracle_action", "gold_action": gold}, retry=retry), tap)
            slog.decisions.append(dlog)
            if ok:
                break
            n1 = len(ledger.records)
            note = action_reject_note(act)
            resp = self._llm_call(llm, messages + [{"role": "user",
                                                    "content": f"Admissible actions:\n{listing}\n\n{note}"}],
                                  None, ledger, DP.ACTION.value, slog, temperature=0.0)
            text = resp.text or ""
            act = _parse_action(text, actions)
            dlog.s2_cost_usd, dlog.s2_latency_ms = self._s2_spent(ledger, n1)
            dlog.gold_aux = {**(dlog.gold_aux or {}), "rewrite_same": act == dlog.s2_value,
                             **({"rewrite_gold": _same_value(act, gold)} if gold is not None else {})}
        return act, text

    def _stop_check(self, env: Env, state: str, name: str, ledger: Ledger, slog: StepLog,
                    args: dict[str, Any] | None = None) -> str:
        """Returns 'continue', 'finish' (accepted) or 'reject' (finish blocked by the stop guard).

        The guard question shows the proposed answer; its gold is ``finish_ok`` (finishing now with this answer
        succeeds), the trigger's is ``world_complete`` (only the answer remains) -- META_GOLD_SPEC §3.3-3.4.
        ``Question.meta`` keeps the legacy ``env.oracle_done()`` (``stop.extra.sim_gold: 1`` sends the new gold
        to simulated backends instead)."""
        c = self.cfg.stop
        if c.mode == Mode.OFF:
            return "finish" if name == self.cfg.final_tool else "continue"
        stopping = name == self.cfg.final_tool
        answer = (args or {}).get("answer", "") if stopping else None
        legacy = env.oracle_done()
        if c.mode == Mode.ASSIST:       # E6d: System-1 advised before the step; the LLM's call decided
            adv = _ep().stop_advice
            _ep().stop_advice = None
            if adv is None:
                return "finish" if stopping else "continue"
            q, a, tap = adv
            g, st, aux = self._stop_gold(env, False, None, legacy)   # the advice answered "complete now?"
            aux = {**aux, "llm_stopped": stopping}
            slog.decisions.append(self._measure(DecisionLog(
                DP.STOP.value, "assist", q.text, None, bool(a.value), a.confidence, a.distribution, deferred=True,
                s2_value=stopping, final_value=stopping, gold=g, state=state, gold_status=st, gold_aux=aux), tap,
                self._step_s2_spent(ledger, (DP.TOOL_SELECT.value, DP.TOOL_PARAMS.value))))
            return "finish" if stopping else "continue"
        ask_here = c.mode in (Mode.SHADOW, Mode.CASCADE, Mode.REPLACE) or (stopping and c.mode == Mode.GUARD) or \
            (not stopping and c.mode == Mode.TRIGGER)
        if not ask_here:
            return "finish" if stopping else "continue"
        q = Noul(q_stop_guard(answer) if stopping else Q_STOP_TRIGGER, decision_point=DP.STOP.value, gold=legacy)
        if c.extra.get("sim_gold"):
            q.meta["gold"] = self._stop_gold(env, stopping, answer, legacy)[0]
        if c.mode == Mode.SHADOW:
            # the question guard (agent stops) or trigger (agent continues) would ask here; the harness keeps the
            # LLM's own stop decision (``s2_value``), S1's argmax answer is only logged
            a, tap = self._shadow_ask(lambda led: self._ask(state, q, led), ledger, "stop")
            g, st, aux = self._stop_gold(env, stopping, answer, legacy)
            slog.decisions.append(self._measure(DecisionLog(
                DP.STOP.value, "shadow", q.text, None, a.value if a else None, a.confidence if a else None,
                a.distribution if a else None, deferred=True, s2_value=stopping, final_value=stopping,
                gold=g, state=state, shadow=True, gold_status=st, gold_aux=aux), tap,
                self._step_s2_spent(ledger, (DP.TOOL_SELECT.value, DP.TOOL_PARAMS.value)),
                (stopping == g) if g is not None else None))
            return "finish" if stopping else "continue"
        tap = Tap(ledger)
        a = self._ask(state, q, tap)
        g, st, aux = self._stop_gold(env, stopping, answer, legacy)
        if c.mode == Mode.REPLACE:
            # E6e: System-1's argmax decides stop / continue at every step (= the cascade at tau 0.5): a "no" at a
            # finish proposal rejects it, a "yes" at another step ends the episode early (the LLM writes the answer)
            stop_now = bool(a.value)
            slog.decisions.append(self._measure(DecisionLog(
                DP.STOP.value, "replace", q.text, None, stop_now, a.confidence, a.distribution, deferred=False,
                s2_value=stopping, final_value=stop_now, gold=g, state=state, gold_status=st, gold_aux=aux,
                retry=_ep().n_stop_rejects if stopping else None), tap))
            if stopping:
                return "finish" if stop_now else "reject"
            return "finish" if stop_now else "continue"
        if c.mode == Mode.CASCADE:
            # E6b: System-1 decides stop / continue when its confidence >= tau (the guard question at a finish
            # proposal, the trigger question otherwise); below tau the LLM's own decision stands (finish iff it
            # proposed finish) -- the trigger at tau and the guard at 1 - tau in one arm
            sure = a.confidence >= c.tau
            stop_now = bool(a.value) if sure else stopping
            slog.decisions.append(self._measure(DecisionLog(
                DP.STOP.value, "cascade", q.text, None, bool(a.value), a.confidence, a.distribution,
                deferred=not sure, s2_value=stopping, final_value=stop_now, gold=g, state=state, gold_status=st,
                gold_aux=aux, retry=_ep().n_stop_rejects if stopping else None), tap))
            if stopping:
                return "finish" if stop_now else "reject"
            return "finish" if stop_now else "continue"
        if stopping:  # guard
            ok = a.p_yes >= c.tau
            slog.decisions.append(self._measure(DecisionLog(
                DP.STOP.value, "guard", q.text, None, ok, a.confidence, a.distribution, final_value=ok, gold=g,
                state=state, gold_status=st, gold_aux=aux, retry=_ep().n_stop_rejects), tap))
            return "finish" if ok else "reject"
        done = a.p_yes >= c.tau        # trigger
        slog.decisions.append(self._measure(DecisionLog(
            DP.STOP.value, "trigger", q.text, None, done, a.confidence, a.distribution, final_value=done, gold=g,
            state=state, gold_status=st, gold_aux=aux), tap))
        return "finish" if done else "continue"

    # ------------------------------------------------------------------ main loop
    def run(self, env: Env) -> Trajectory:
        cfg = self.cfg
        ledger = Ledger()
        traj = Trajectory(task_id=env.task_id, config=cfg.name, llm=self.llm.model,
                          backend=(self.s1.describe()["backend"] if self.s1 and cfg.uses_s1() else None))
        if self.s1 is not None and cfg.uses_s1():
            # the full System-1 spec: analyses match backends on its id, never on the name alone (rate(0.1) and
            # rate(0.3) share the name "rate"); the runner refuses to resume an arm whose spec changed
            desc = self.s1.describe()
            traj.meta["s1"] = {"id": backend_id(desc), "describe": desc}
        t0 = time.perf_counter()
        _call_extras().clear()  # provider extras are episode-local
        _shadow_errors().clear()
        _empty_responses().clear()
        ep = _EXTRAS_TLS.ep = _Episode(env=env)
        snapshotting = cfg.measure.active()
        for m in (self.llm, self.small_llm):  # simulation-only hook (see llm/simulated.py)
            if m is not None and hasattr(m, "bind_env"):
                m.bind_env(env)
        memory: MemoryStore | None = None
        try:
            obs = env.reset()
            action_env = env.admissible_actions() is not None
            tools = [] if action_env else env.tools() + [finish_tool(cfg.final_tool)]
            sys_prompt = self.system_prompt or (ACTION_SYSTEM if action_env else DEFAULT_SYSTEM)
            messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt},
                                              {"role": "user", "content": f"{env.instruction}\n\n{obs}"}]
            memory = MemoryStore(cfg.memory_write, cfg.memory_read, s1=self.s1, llm=self.llm, log_decisions=True)
            history: list[tuple[str, str]] = []
            last_reflection = ""
            for step in range(cfg.max_steps):
                ledger.step = step
                ep.step, ep.budget = step, cfg.max_steps - step
                slog = StepLog(step=step)
                traj.steps.append(slog)
                mem_text = ""
                if cfg.memory_read.mode != Mode.OFF:
                    base_state = self._s1_state(env, history, obs)
                    pool = list(memory.items)
                    n_gate = memory.n_llm_calls
                    recalled = memory.recall(base_state, ledger=ledger)
                    slog.llm_calls += memory.n_llm_calls - n_gate     # E6d read guard: the LLM's selection calls
                    new = memory.take_decisions()
                    self._memory_read_gold(env, memory, new, pool)
                    slog.decisions.extend(new)
                    mem_text = memory.render(recalled)
                state = self._s1_state(env, history, obs, mem_text, last_reflection)
                pre_msgs = list(messages)      # the messages before any reflection of this step (CF snapshots)
                llm = self._route(state, ledger, slog)
                self._reflect(llm, env, state, messages, ledger, slog, step)
                if slog.reflection:  # System-1 decisions of this step must see the reflection too
                    last_reflection = slog.reflection
                    state = self._s1_state(env, history, obs, mem_text, last_reflection)
                # built after the reflection so this step's decision is conditioned on the reflection it paid for
                turn_msgs = messages + ([{"role": "user", "content": f"[Memory]\n{mem_text}"}] if mem_text else [])
                if cfg.stop.mode == Mode.ASSIST and not action_env:   # E6d: System-1's advice in the LLM's turn
                    q_adv = Noul(Q_STOP_TRIGGER, decision_point=DP.STOP.value, gold=env.oracle_done())
                    tap_adv = Tap(ledger)
                    a_adv = self._ask(state, q_adv, tap_adv)
                    ep.stop_advice = (q_adv, a_adv, tap_adv)
                    turn_msgs = turn_msgs + [{"role": "user", "content": stop_advice_note(bool(a_adv.value),
                                                                                         a_adv.confidence)}]

                if action_env:
                    actions = env.admissible_actions() or []
                    name, text = self._action_step(llm, env, actions, turn_msgs, state, ledger, slog)
                    args: dict[str, Any] = {"action": name}
                    slog.tool, slog.args = "act", args
                    self._patch_meta_golds(env, slog, "act", args)
                    res = env.step("act", args)
                    messages.append({"role": "assistant", "content": text or f"Action: {name}"})
                    messages.append({"role": "user", "content": f"Observation: {res.observation}"})
                    history.append((name, res.observation))
                    slog.observation = _trunc(res.observation, cfg.obs_truncate_chars)
                    obs = res.observation
                    if cfg.memory_write.mode != Mode.OFF:
                        known = [m.text for m in memory.items]
                        mtext = f"{name} -> {res.observation}"
                        n_gate = memory.n_llm_calls
                        memory.offer(mtext, state, ledger=ledger, step=step, q_meta=self._memory_sim_meta(env, mtext))
                        new = memory.take_decisions()
                        self._memory_write_gold(env, new, mtext, known)
                        slog.decisions.extend(new)
                        slog.llm_calls += memory.n_llm_calls - n_gate  # the LLM write gate (llm / cascade / guard)
                    if res.done:
                        break
                    continue

                name, args, cid = self._tool_step(llm, env, tools, turn_msgs, state, ledger, slog)
                snap: StepSnap | None = None
                if snapshotting:
                    snap = self._snapshot(env, step, pre_msgs, mem_text, state, llm, tools, name, args)
                verdict = self._stop_check(env, state, name, ledger, slog, args)
                v0 = self._patch_meta_golds(env, slog, name, args)
                stop_log = next((d for d in slog.decisions if d.dp == DP.STOP.value), None)
                if stop_log is not None and name == cfg.final_tool and isinstance(stop_log.gold_aux, dict) \
                        and cfg.stop.extra.get("augment_trigger"):
                    ep.extra_probes.append(("augmented_stop_trigger", stop_log, step))
                if snap is not None:
                    self._fill_snapshot(snap, slog, stop_log)
                    ep.snaps.append(snap)
                if verdict == "reject":
                    append_note(messages, STOP_REJECT_NOTE)
                    slog.tool, slog.args = name, args
                    slog.observation = "(finish rejected by stop verifier)"
                    ep.prev = (v0, slog.observation)
                    ep.n_stop_rejects += 1
                    # the rejected proposal enters System-1's history, so the next question is asked at a NEW state:
                    # a resubmission is a fresh decision, not a byte-identical re-query that a deterministic gate
                    # (the rate control's hash, a T=0 cached decider) answers the same way until max_steps
                    history.append((f"{name}({json.dumps(args)[:200]})", slog.observation))
                    continue
                if verdict == "finish" and name != cfg.final_tool:
                    # early-stop trigger fired: ask the LLM for the final answer only
                    r = self._llm_call(llm, messages + [{"role": "user", "content": EARLY_STOP_PROMPT}],
                                       [finish_tool(cfg.final_tool)], ledger, DP.STOP.value, slog,
                                       tool_choice={"type": "function", "function": {"name": cfg.final_tool}})
                    name, args, cid = self._resp_to_call(r, cfg.final_tool)
                slog.tool, slog.args = name, args
                if name == cfg.final_tool:
                    traj.final_answer = str(args.get("answer", ""))
                    break
                call = {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                extra = _call_extras().pop(cid, None)
                if extra:
                    call["extra_content"] = extra
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
                if name not in {tool_name(t) for t in tools}:
                    res_obs, done = f"Error: unknown tool {name!r}", False
                else:
                    res = env.step(name, args)
                    res_obs, done = res.observation, res.done
                slog.observation = _trunc(res_obs, cfg.obs_truncate_chars)
                messages.append({"role": "tool", "tool_call_id": cid, "content": _trunc(res_obs, cfg.obs_truncate_chars)})
                history.append((f"{name}({json.dumps(args)[:200]})", res_obs))
                obs = res_obs
                ep.prev = (v0, res_obs)
                if cfg.memory_write.mode != Mode.OFF:
                    known = [m.text for m in memory.items]
                    mtext = f"{name}({json.dumps(args)[:200]}) -> {_trunc(res_obs, 600)}"
                    n_gate = memory.n_llm_calls
                    memory.offer(mtext, state, ledger=ledger, step=step, q_meta=self._memory_sim_meta(env, mtext))
                    new = memory.take_decisions()
                    self._memory_write_gold(env, new, mtext, known)
                    slog.decisions.extend(new)
                    slog.llm_calls += memory.n_llm_calls - n_gate  # the LLM write gate (llm / cascade / guard)
                if done:
                    break
            traj.result = env.evaluate(traj.final_answer)
            traj.meta["memory"] = {"offered": memory.n_offered, "stored": len(memory.items)}
        except Exception as e:  # keep the run going; failures are data
            traj.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
            try:
                traj.result = env.evaluate(traj.final_answer)
            except Exception:
                traj.result = {"success": False, "score": 0.0}
        traj.wall_ms = (time.perf_counter() - t0) * 1000
        if traj.error is None and (ep.snaps or ep.extra_probes):
            # deferred measurement: after env.evaluate, on snapshots -- never changes the episode (fail-open)
            from .measure import run_deferred

            t1 = time.perf_counter()
            try:
                run_deferred(self, traj, ledger, ep)
            except Exception as e:  # noqa: BLE001 - measurement must never error an episode
                ep.probe_errors.append(f"deferred: {type(e).__name__}: {str(e)[:300]}")
            traj.meta["measure_wall_ms"] = (time.perf_counter() - t1) * 1000
        traj.usage = usage_summary(ledger)
        traj.calls = ledger.to_list()
        errs = list(_shadow_errors()) + (memory.shadow_errors if memory is not None else [])
        if errs:
            traj.meta["shadow_s1_errors"] = errs
        if _empty_responses():
            traj.meta["empty_llm_responses"] = len(_empty_responses())
        gold = self._gold_summary(traj, ep)
        if gold:
            traj.meta["gold"] = gold
        for k, v in (("gold_errors", ep.gold_errors), ("cf_errors", ep.cf_errors),
                     ("probe_errors", ep.probe_errors)):
            if v:
                traj.meta[k] = list(v)
        traj.meta["config"] = cfg.model_dump(mode="json")
        _EXTRAS_TLS.ep = None
        return traj

    # ------------------------------------------------------------------ snapshots / summary
    def _snapshot(self, env: Env, step: int, pre_msgs: list[dict[str, Any]], mem_text: str, state: str,
                  llm: LLMClient, tools: list[dict[str, Any]], name: str, args: dict[str, Any]) -> StepSnap | None:
        """An independent copy of s_t for the deferred measurement (taken before anything executes)."""
        try:
            snap_fn = getattr(env, "oracle_snapshot", None)
            e2 = snap_fn() if callable(snap_fn) else copy.deepcopy(env)
        except Exception as e:  # noqa: BLE001
            _ep().cf_errors.append(f"snapshot: {type(e).__name__}: {str(e)[:200]}")
            return None
        return StepSnap(step=step, budget=self.cfg.max_steps - step, env=e2, messages=list(pre_msgs),
                        mem_text=mem_text, state=state, llm=llm, tools=tools,
                        a0=(name, dict(args) if isinstance(args, dict) else {}), finish=name == self.cfg.final_tool)

    @staticmethod
    def _fill_snapshot(snap: StepSnap, slog: StepLog, stop_log: DecisionLog | None) -> None:
        for d in slog.decisions:
            if d.dp == DP.REFLECTION.value and snap.refl is None:
                snap.refl = d
            elif d.dp == DP.ROUTING.value and snap.route is None:
                snap.route = d
        snap.stop = stop_log
        if stop_log is not None and isinstance(stop_log.gold_aux, dict):
            snap.stop_gold = {"gold": stop_log.gold, "status": stop_log.gold_status, **stop_log.gold_aux}

    @staticmethod
    def _gold_summary(traj: Trajectory, ep: _Episode) -> dict[str, Any]:
        """``Trajectory.meta.gold``: counts of the labelled / excluded opportunities of this episode."""
        recs = [d for s in traj.steps for d in s.decisions if d.gold_aux is not None] + list(traj.probes)
        if not recs:
            return {}
        st = [d.gold_status for d in recs]
        n_cf_err = sum(1 for d in recs if d.gold_status == "cf_error"
                       or (d.gold_aux or {}).get("cf_status") == "cf_error")
        return {"v": GOLD_V, "n_labelled": sum(1 for x in st if x == "ok"),
                "n_doomed": st.count("doomed"), "n_unobserved": st.count("unobserved"),
                "n_no_oracle": st.count("no_oracle"), "n_cf_error": n_cf_err,
                "n_no_valid_probe": ep.n_no_valid_probe}


__all__ = ["JevAgent", "usage_summary", "append_note", "Q_ROUTING", "Q_REFLECTION", "Q_REFLECTION_V1",
           "Q_STOP_GUARD", "Q_STOP_TRIGGER", "q_stop_guard", "StepSnap", "Tap"]
