"""Deferred measurement (docs/META_GOLD_SPEC.md §4, §6): counterfactual efficacy labels and wrong-proposal probes.

Everything here runs AFTER the episode (after ``env.evaluate``, before the trajectory is dumped), on snapshots taken
at decision time (``agent.StepSnap``). It therefore cannot change the trajectory, warm a provider's prompt cache for
later live steps, or add critical-path latency, and it is fail-open: a failed call is counted
(``Trajectory.meta["cf_errors"]`` / ``["probe_errors"]``), never raised. Every call goes through the episode ledger
tagged ``meta.shadow = true`` and ``meta.measure`` in {"cf", "probe"} (``harness/tap.py``), so it stays out of the
episode cost (``usage.shadow.by_measure`` itemises it).

Principles kept here (§1):

- **The oracle labels and selects; it never authors.** The oracle decides where a probe or counterfactual is placed
  and which candidate edits are kept (wrongness filter). Every string a backend sees comes from a model (the arm's
  System-2 or a transplant model), from a fixed template, or from a deterministic edit of a model proposal that uses
  only values visible in the S1 state string.
- **Probes are single-question System-1 calls**, never batched with any other question, and stored in
  ``Trajectory.probes`` (``mode = "guard_probe"``), outside ``steps[]``.

Counterfactual branches (§4), each classified by ``verdict`` on the snapshot:

  verdict_alt       (reflection)  the reflection call (same prompt), ``append_note``, then the step's tool decision
  verdict_large     (routing)     the large model's tool decision on the same turn messages
  verdict_reject    (stop guard)  the rejection note, then the next tool decision (every natural finish)
  verdict_placebo                 the default tool decision again, without intervention (every state with a branch)
  fired steps                     live trigger arms (``cf.fired_steps``): the default decision a0 where the gate fired

Positive states (need_fatal) always get the branches; negatives a seeded share ``cf.sample_negatives`` (hash of
seed, task, step, purpose), with inverse-probability weight ``gold_aux.cf_weight``.

Probe sources (§6.2): ``natural`` (the executed proposal), ``forced_premature`` (the arm's own System-2 forced to
finish at a not-yet-complete state, with the stop trigger's own text), ``transplant`` (a weaker generator on the
same messages; kept whatever the label) and the deterministic edits T1-T6 (kept only when the oracle labels them
wrong; up to ``max_resamples`` tries, then a "no valid probe" count).

Full-episode continuation (review round 4, P1-6; ``scripts/cf_validity.py``): :func:`continue_from_snapshot`
resumes an episode from a decision-time snapshot kept in process (:func:`keep_snapshots`), and
:func:`continue_episode` rebuilds the state of a logged step by replaying the logged action prefix through a fresh
env (JevGym is deterministic given seed + actions, ``bench/meta_gold.replay_golds``); both apply one of the
counterfactual interventions ONCE at that step and then run the LLM-only loop to the end of the episode. It is the
validity check of the one-step labels (``verdict_alt`` / ``verdict_large`` / ``verdict_reject`` against the episode
outcome), never a label itself.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

from ..accounting import Ledger
from ..decision.controls import STATE_CHANGING_VERBS
from ..decision.types import Noul
from .config import DP, Mode
from .prompts import (EARLY_STOP_PROMPT, Q_STOP_TRIGGER, REFLECT_PROMPT, REFLECTION_WORDINGS, STOP_REJECT_NOTE,
                      append_note, q_stop_guard, q_tool_guard)
from .tap import Tap
from .tools import ParamInfo, extract_candidates, finish_tool, params_of, tool_name
from .trajectory import DecisionLog, StepLog, Trajectory

if TYPE_CHECKING:  # pragma: no cover
    from ..llm.base import LLMClient
    from .agent import JevAgent, StepSnap, _Episode

GOLD_V = 1
FAILING = frozenset({"fatal", "premature", "wrong_answer"})
FREE_TEMPLATE = "(as requested above)"          # fixed template for free-text slots of synthetic edits
_NUMBER_ASK = re.compile(r"\b(total|how many|number|amount|sum|count)\b", re.I)
_AMOUNT_KEY = re.compile(r"\"amount\"\s*:\s*(\d+(?:\.\d+)?)")
_TWO_DEC = re.compile(r"(?<![\w.#-])(\d+\.\d{2})(?![\w.])")
_NUM_IN_ANSWER = re.compile(r"(?<![\w.#])(\d+(?:\.\d+)?)(?![\w])")


def _u01(*parts: Any) -> float:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def _call_of(resp: Any, final_tool: str) -> tuple[str, dict[str, Any]]:
    if resp.tool_calls:
        tc = resp.tool_calls[0]
        return tc.name, dict(tc.arguments or {})
    return final_tool, {"answer": resp.text}


def _turn(snap: "StepSnap", msgs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    m = list(snap.messages if msgs is None else msgs)
    return m + ([{"role": "user", "content": f"[Memory]\n{snap.mem_text}"}] if snap.mem_text else [])


def _dnum(d: Any) -> int | None:
    try:
        return None if d is None or d == float("inf") else int(d)
    except (TypeError, ValueError):
        return None


class _Runner:
    def __init__(self, agent: "JevAgent", traj: Trajectory, ledger: Ledger, ep: "_Episode"):
        self.agent, self.traj, self.ledger, self.ep = agent, traj, ledger, ep
        self.cfg = agent.cfg
        self.m = agent.cfg.measure
        self.final = agent.cfg.final_tool
        self.task = traj.task_id

    # ------------------------------------------------------------------ shared helpers
    def verdict(self, snap: "StepSnap", name: str, args: dict[str, Any], budget: int | None = None) -> dict[str, Any]:
        return snap.env.oracle_verdict(self.agent._env_tool(snap.env, name), args,
                                       snap.budget if budget is None else budget)

    def tap(self, snap: "StepSnap", measure: str, tag: str) -> Tap:
        self.ledger.step = snap.step
        key = "cf_of" if measure == "cf" else "probe_of"
        return Tap(self.ledger, measure=measure, extra={key: f"{snap.step}:{tag}"})

    def chat(self, llm: Any, snap: "StepSnap", msgs: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
             measure: str, tag: str, **kw: Any) -> Any:
        if hasattr(llm, "bind_env"):       # simulation-only hook: a simulated LLM follows the snapshot's oracle
            llm.bind_env(snap.env)
        dp = DP.REFLECTION.value if tools is None else (DP.STOP.value if kw.get("tool_choice") else
                                                        DP.TOOL_SELECT.value)
        return llm.chat(msgs, tools=tools, ledger=self.tap(snap, measure, tag), decision_point=dp, **kw)

    def decide(self, snap: "StepSnap", llm: Any, msgs: list[dict[str, Any]], measure: str, tag: str,
               **kw: Any) -> tuple[str, dict[str, Any]]:
        return _call_of(self.chat(llm, snap, msgs, snap.tools, measure, tag, **kw), self.final)

    def ask(self, snap_step: int, state: str, text: str, dp: str, legacy: Any, source: str) -> Any:
        """One System-1 question, alone in its request (probes are never batched)."""
        assert self.agent.s1 is not None
        self.ledger.step = snap_step
        tap = Tap(self.ledger, measure="probe", extra={"probe_of": f"{snap_step}:{source}"})
        # Question.meta carries exactly the legacy label (stop: oracle_done, tool guard: oracle_accepts), else nothing
        q = Noul(text, decision_point=dp, **({"gold": legacy} if legacy is not None else {}))
        a = self.agent.s1.ask(state, q, ledger=tap)
        return a, tap

    # ------------------------------------------------------------------ probe-list records without env (§3.1, §3.4)
    def extra_records(self) -> None:
        for item in self.ep.extra_probes:
            source, base, step = item if len(item) == 3 else (*item, None)   # type: ignore[misc]
            try:
                if source.startswith("wording_"):
                    text, dp = REFLECTION_WORDINGS[source.split("_", 1)[1]], DP.REFLECTION.value
                    gold, status, aux, legacy = base.gold, base.gold_status, dict(base.gold_aux or {}), None
                else:   # augmented_stop_trigger: the trigger question at a natural finish state
                    text, dp = Q_STOP_TRIGGER, DP.STOP.value
                    aux = dict(base.gold_aux or {})
                    gold = aux.get("world_complete") if base.gold_status == "ok" else None
                    status = base.gold_status if base.gold_status != "ok" else ("ok" if gold is not None
                                                                               else "no_oracle")
                    legacy = aux.get("canonical_done")
                    for k in ("kind",):
                        aux.pop(k, None)
                st = step if step is not None else self._step_of(base)
                a, tap = self.ask(st if st is not None else 0, base.state or "", text, dp, legacy, source)
                d = DecisionLog(dp, "shadow", text, None, a.value, a.confidence, a.distribution, deferred=True,
                                gold=gold, state=base.state, shadow=True, gold_status=status,
                                gold_aux={**aux, "v": GOLD_V, "probe_source": source, "step": st})
                d.s1_cost_usd, d.s1_latency_ms = tap.spent()
                self.traj.probes.append(d)
            except Exception as e:  # noqa: BLE001
                self.ep.probe_errors.append(f"{source}: {type(e).__name__}: {str(e)[:200]}")

    def _step_of(self, d: DecisionLog) -> int | None:
        for s in self.traj.steps:
            if any(x is d for x in s.decisions):
                return s.step
        return None

    # ------------------------------------------------------------------ counterfactual branches (§4)
    def weight(self, gold: Any, snap: "StepSnap", purpose: str) -> float | None:
        if gold is True:
            return 1.0
        rate = float(self.m.cf.sample_negatives) if self.m.cf else 0.0
        if rate > 0 and _u01(self.m.seed, self.task, snap.step, purpose) < rate:
            return 1.0 / rate
        return None

    def counterfactuals(self) -> None:
        cf = self.m.cf
        assert cf is not None
        supported = self.cfg.tool_select.mode in (Mode.OFF, Mode.SHADOW)
        for snap in self.ep.snaps:
            if not callable(getattr(snap.env, "oracle_verdict", None)):
                continue
            refl, route, stop = snap.refl, snap.route, snap.stop
            br: dict[str, tuple[DecisionLog, float]] = {}
            if cf.reflection and refl is not None and refl.mode == "shadow" and refl.gold_status == "ok":
                w = self.weight(refl.gold, snap, "reflection")
                if w:
                    br["reflection"] = (refl, w)
            if cf.large_llm and route is not None and route.mode == "shadow" and route.gold_status == "ok":
                w = self.weight(route.gold, snap, "large")
                if w:
                    br["large"] = (route, w)
            if cf.stop_reject and snap.finish and stop is not None and stop.gold_status == "ok":
                br["reject"] = (stop, 1.0)
            if cf.fired_steps:
                if refl is not None and refl.mode == "trigger" and refl.gold_status == "unobserved":
                    br["fired_reflection"] = (refl, 1.0)
                if route is not None and route.mode == "trigger" and route.gold_status == "unobserved":
                    br["fired_route"] = (route, 1.0)
            if not br:
                continue
            if not supported:   # the default decision is not a plain System-2 call: no counterfactual replay
                for d, _ in br.values():
                    d.gold_aux = {**(d.gold_aux or {"v": GOLD_V}), "cf_status": "cf_unsupported"}
                continue
            placebo: str | None = None
            placebo_err = None
            if cf.placebo and br.keys() & {"reflection", "large", "reject"}:
                try:
                    placebo = self.verdict(snap, *self.decide(snap, snap.llm, _turn(snap), "cf", "placebo"))["verdict"]
                except Exception as e:  # noqa: BLE001
                    placebo_err = f"placebo: {type(e).__name__}: {str(e)[:200]}"
                    self.ep.cf_errors.append(placebo_err)
            for b, (d, w) in br.items():
                try:
                    self._branch(snap, b, d, w, placebo, placebo_err)
                except Exception as e:  # noqa: BLE001
                    self.ep.cf_errors.append(f"{b}: {type(e).__name__}: {str(e)[:200]}")
                    d.gold_aux = {**(d.gold_aux or {"v": GOLD_V}), "cf_status": "cf_error"}
                    if b.startswith("fired_"):
                        d.gold_status = "cf_error"

    def _branch(self, snap: "StepSnap", b: str, d: DecisionLog, w: float, placebo: str | None,
                placebo_err: str | None) -> None:
        aux = dict(d.gold_aux or {"v": GOLD_V})
        if b == "reflection":
            r = self.chat(snap.llm, snap, list(snap.messages) + [{"role": "user", "content": REFLECT_PROMPT}], None,
                          "cf", "reflection", temperature=0.0)
            m2 = list(snap.messages)
            append_note(m2, f"[Reflection] {r.text}")
            v_alt = self.verdict(snap, *self.decide(snap, snap.llm, _turn(snap, m2), "cf", "reflection"))["verdict"]
            v0 = aux.get("verdict")
            aux.update(verdict_alt=v_alt, benefit=bool(v0 in FAILING and v_alt == "progress"),
                       harm=bool(v0 == "progress" and v_alt in FAILING))
        elif b == "large":
            large = self.agent.measure_llms[self.m.cf.large_llm]   # type: ignore[union-attr,index]
            v_l = self.verdict(snap, *self.decide(snap, large, _turn(snap), "cf", "large"))["verdict"]
            v0 = aux.get("verdict")
            aux.update(verdict_large=v_l, large_model=getattr(large, "model", None),
                       benefit=bool(v0 in FAILING and v_l == "progress"),
                       harm=bool(v0 == "progress" and v_l in FAILING))
        elif b == "reject":
            m2 = list(snap.messages)
            append_note(m2, STOP_REJECT_NOTE)
            name, args = self.decide(snap, snap.llm, _turn(snap, m2), "cf", "reject")
            v_r = self.verdict(snap, name, args, budget=snap.budget - 1)["verdict"]
            v0 = "progress" if d.gold else aux.get("kind")
            aux.update(verdict=v0, verdict_reject=v_r, benefit=bool(v0 in FAILING and v_r == "progress"),
                       harm=bool(v0 == "progress" and v_r in FAILING))
        elif b in ("fired_reflection", "fired_route"):
            llm = snap.llm if b == "fired_reflection" else self.agent.small_llm
            v = self.verdict(snap, *self.decide(snap, llm, _turn(snap), "cf", b))
            if v.get("verdict") is None:
                d.gold, d.gold_status = None, "doomed"
            else:
                d.gold, d.gold_status = v["verdict"] in FAILING, "ok"
                aux.update(verdict=v["verdict"], need_any=v["verdict"] != "progress", verdict_src="cf",
                           d_before=v.get("d_before"))
                if v.get("fatal_kind"):
                    aux["fatal_kind"] = v["fatal_kind"]
            aux["cf_status"] = "ok"
            d.gold_aux = aux
            return
        aux["cf_weight"] = w
        aux["cf_status"] = "ok"
        if placebo is not None:
            aux["verdict_placebo"] = placebo
        elif placebo_err:
            aux["placebo_status"] = "cf_error"
        d.gold_aux = aux

    # ------------------------------------------------------------------ probes (§6)
    def _score(self, snap: "StepSnap", dp: str, text: str, gold: Any, status: str, aux: dict[str, Any],
               legacy: Any, source: str) -> None:
        P = self.m.probes
        assert P is not None
        sim = (self.cfg.stop if dp == DP.STOP.value else self.cfg.tool_select).extra.get("sim_gold")
        a, tap = self.ask(snap.step, snap.state, text, dp, gold if sim else legacy, source)
        ok = a.p_yes >= P.tau_ref
        d = DecisionLog(dp, "guard_probe", text, None, ok, a.confidence, a.distribution, gold=gold, state=snap.state,
                        shadow=True, gold_status=status, gold_aux={"v": GOLD_V, **aux, "step": snap.step,
                                                                   "tau_ref": P.tau_ref})
        d.s1_cost_usd, d.s1_latency_ms = tap.spent()
        self.traj.probes.append(d)

    def tool_probe(self, snap: "StepSnap", name: str, args: dict[str, Any], source: str, operator: str | None,
                   model: str | None, v: dict[str, Any] | None = None) -> None:
        v = v or self.verdict(snap, name, args)
        gold, status = (None, "doomed") if v.get("verdict") is None else (v["verdict"] == "progress", "ok")
        legacy = None
        acc = getattr(snap.env, "oracle_accepts", None)
        if callable(acc):
            try:
                legacy = acc(self.agent._env_tool(snap.env, name), args)
            except Exception:  # noqa: BLE001
                legacy = None
        aux = {"probe_source": source, "operator": operator, "verdict": v.get("verdict"), "kind": v.get("verdict"),
               "d_before": v.get("d_before"), "canonical": v.get("canonical"), "generator_model": model,
               "proposal": {"tool": name, "args": args}}
        self._score(snap, DP.TOOL_SELECT.value, q_tool_guard(name, args), gold, status, aux, legacy, source)

    def stop_label(self, snap: "StepSnap", answer: Any) -> tuple[Any, str, dict[str, Any]]:
        env = snap.env
        d0 = _dnum(env.oracle_cost_to_go(snap.budget))
        wc = bool(env.oracle_world_complete())
        if d0 is None:
            return None, "doomed", {"verdict": None, "kind": None, "d_before": None, "world_complete": wc}
        ok = bool(env.oracle_finish_ok("" if answer is None else str(answer)))
        kind = "ok" if ok else ("wrong_answer" if wc else "premature")
        return ok, "ok", {"verdict": "progress" if ok else kind, "kind": kind, "d_before": d0, "world_complete": wc}

    def stop_probe(self, snap: "StepSnap", answer: Any, source: str, operator: str | None, model: str | None,
                   label: tuple[Any, str, dict[str, Any]] | None = None) -> None:
        gold, status, aux = label or self.stop_label(snap, answer)
        legacy = snap.env.oracle_done() if callable(getattr(snap.env, "oracle_done", None)) else None
        self._score(snap, DP.STOP.value, q_stop_guard(answer), gold, status,
                    {**aux, "probe_source": source, "operator": operator, "generator_model": model,
                     "answer": "" if answer is None else str(answer)[:300]}, legacy, source)

    def _try(self, what: str, fn: Any) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - fail-open
            self.ep.probe_errors.append(f"{what}: {type(e).__name__}: {str(e)[:200]}")

    def probes(self) -> None:
        P = self.m.probes
        assert P is not None
        snaps = [s for s in self.ep.snaps if callable(getattr(s.env, "oracle_verdict", None))
                 and callable(getattr(s.env, "oracle_finish_ok", None))]
        if not snaps:
            return
        fins = [s for s in snaps if s.finish]
        seed, task = self.m.seed, self.task
        if "natural" in P.tool_select:
            for s in snaps:
                if not s.finish:
                    self._try("natural", lambda s=s: self.tool_probe(s, *s.a0, "natural", None,
                                                                     getattr(s.llm, "model", None)))
        if "natural" in P.stop and self.cfg.stop.mode not in (Mode.SHADOW, Mode.GUARD):
            for s in fins:
                self._try("natural_stop", lambda s=s: self.stop_probe(s, s.a0[1].get("answer", ""), "natural", None,
                                                                      getattr(s.llm, "model", None)))
        if "forced_premature" in P.stop:
            self._try("forced_premature", lambda: self._forced_premature(snaps, fins))
        if "transplant" in P.stop or "transplant" in P.tool_select:
            tl = self.agent.measure_llms.get(P.transplant_llm or "main")
            for s in snaps:
                if s.finish or _u01(seed, task, s.step, "transplant") < P.rho_transplant:
                    self._try("transplant", lambda s=s: self._transplant(s, tl))
        tool_ops = [o for o in ("T1", "T2", "T3", "T4") if o in P.tool_select]
        for s in snaps:
            if tool_ops and not s.finish and _u01(seed, task, s.step, "edit") < P.rho_edit:
                self._try("edit", lambda s=s: self._tool_edit(s, tool_ops))
            if "T5" in P.stop and not s.finish and _u01(seed, task, s.step, "T5") < P.rho_t5:
                self._try("T5", lambda s=s: self._t5(s))
            if "T6" in P.stop and s.finish:
                self._try("T6", lambda s=s: self._t6(s))

    def _forced_premature(self, snaps: list["StepSnap"], fins: list["StepSnap"]) -> None:
        """Once per episode, at a state with complete = False and t >= 1: with probability 1/2 the last such state
        before the natural finish (D = 2, the hardest case), otherwise seeded-uniform; the arm's own System-2 is
        told exactly what a mis-fired stop trigger tells it and forced to call finish."""
        cands = []
        for s in snaps:
            if s.step < 1 or s.finish:
                continue
            if _dnum(s.env.oracle_cost_to_go(s.budget)) is None or s.env.oracle_world_complete():
                continue
            cands.append(s)
        if not cands:
            self.ep.n_no_valid_probe += 1
            return
        fin_step = min((s.step for s in fins), default=None)
        before = [s for s in cands if fin_step is None or s.step < fin_step]
        if fin_step is not None and before and _u01(self.m.seed, self.task, "forced_premature") < 0.5:
            pick = before[-1]
        else:
            pick = cands[min(len(cands) - 1, int(_u01(self.m.seed, self.task, "forced_premature_pick") * len(cands)))]
        resp = self.chat(pick.llm, pick, list(pick.messages) + [{"role": "user", "content": EARLY_STOP_PROMPT}],
                         [finish_tool(self.final)], "probe", "forced_premature",
                         tool_choice={"type": "function", "function": {"name": self.final}})
        _, args = _call_of(resp, self.final)
        self.stop_probe(pick, args.get("answer", ""), "forced_premature", None, getattr(pick.llm, "model", None))

    def _transplant(self, s: "StepSnap", tl: Any) -> None:
        P = self.m.probes
        assert P is not None
        name, args = self.decide(s, tl, _turn(s), "probe", "transplant", temperature=P.transplant_temperature)
        model = getattr(tl, "model", None)
        if name == self.final:
            if "transplant" in P.stop:
                self.stop_probe(s, args.get("answer", ""), "transplant", None, model)
        elif "transplant" in P.tool_select:
            self.tool_probe(s, name, args, "transplant", None, model)

    # ---- synthetic edits (robustness; the only source of fatal tool-guard probes)
    def _tool_edit(self, s: "StepSnap", ops: list[str]) -> None:
        P = self.m.probes
        assert P is not None
        for attempt in range(max(1, P.max_resamples)):
            op = ops[min(len(ops) - 1, int(_u01(self.m.seed, self.task, s.step, "edit_op", attempt) * len(ops)))]
            rng = random.Random(f"{self.m.seed}|{self.task}|{s.step}|{op}|{attempt}")
            prop = EDITS[op](self, s, rng)
            if prop is None:
                continue
            v = self.verdict(s, *prop)
            if v.get("verdict") in (None, "progress"):     # keep only edits the oracle labels wrong
                continue
            self.tool_probe(s, prop[0], prop[1], "edit", op, None, v=v)
            return
        self.ep.n_no_valid_probe += 1

    def _t5(self, s: "StepSnap") -> None:
        """T5: a premature finish -- the sum of the amounts visible in the state when the instruction asks for a
        number, else "done"."""
        amounts = visible_amounts(s.state)
        ans = f"{sum(amounts):.2f}" if amounts and _NUMBER_ASK.search(task_text(s.state)) else "done"
        lab = self.stop_label(s, ans)
        if lab[0] is not False:
            self.ep.n_no_valid_probe += 1
            return
        self.stop_probe(s, ans, "edit", "T5", None, label=lab)

    def _t6(self, s: "StepSnap") -> None:
        """T6: at a finish whose answer has a number, drop or add one visible summand, or apply a +-1.00 / +-0.01 /
        transposed-digit slip."""
        P = self.m.probes
        assert P is not None
        ans0 = str(s.a0[1].get("answer", ""))
        nums = list(_NUM_IN_ANSWER.finditer(ans0))
        if not nums:
            return
        for attempt in range(max(1, P.max_resamples)):
            rng = random.Random(f"{self.m.seed}|{self.task}|{s.step}|T6|{attempt}")
            new = t6_edit(ans0, nums[-1], visible_amounts(s.state), rng)
            if new is None or new == ans0:
                continue
            lab = self.stop_label(s, new)
            if lab[0] is False:
                self.stop_probe(s, new, "edit", "T6", None, label=lab)
                return
        self.ep.n_no_valid_probe += 1


# ---------------------------------------------------------------------- edit operators (state-visible values only)
def task_text(state: str) -> str:
    m = re.match(r"TASK:(.*?)(\n\n[A-Z ]+:\n|$)", state, re.S)
    return m.group(1) if m else state[:2000]


def visible_amounts(state: str) -> list[float]:
    """Amounts visible in the S1 state (JSON ``"amount": x`` fields, else 2-decimal numbers; never ``#`` noise)."""
    vals = [float(x) for x in _AMOUNT_KEY.findall(state)]
    if not vals:
        vals = [float(x) for x in _TWO_DEC.findall(state)]
    out: list[float] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def t6_edit(ans: str, m: re.Match[str], amounts: list[float], rng: random.Random) -> str | None:
    s = m.group(1)
    try:
        x = float(s)
    except ValueError:
        return None
    dec = len(s.split(".")[1]) if "." in s else 0
    ops = ["plus1", "minus1", "plus_cent", "minus_cent"]
    if len(s.replace(".", "")) >= 2:
        ops.append("transpose")
    if amounts:
        ops += ["drop", "add"]
    op = rng.choice(ops)
    if op == "transpose":
        digits = [i for i in range(len(s) - 1) if s[i].isdigit() and s[i + 1].isdigit() and s[i] != s[i + 1]]
        if not digits:
            return None
        i = rng.choice(digits)
        new = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    else:
        y = {"plus1": x + 1.0, "minus1": x - 1.0, "plus_cent": x + 0.01, "minus_cent": x - 0.01,
             "drop": x - rng.choice([a for a in amounts if a <= x] or amounts),
             "add": x + rng.choice(amounts)}[op]
        new = f"{y:.{max(dec, 2 if op in ('plus_cent', 'minus_cent') or dec else 0)}f}"
    return ans[:m.start(1)] + new + ans[m.end(1):]


_SHAPES: list[tuple[re.Pattern[str], re.Pattern[str]]] = [
    (re.compile(r"(^|_)(id|ids)$|^id$", re.I), re.compile(r"^(?=.*\d)[A-Za-z][\w\-#]*$")),
    (re.compile(r"^(amount|units|count|quantity)$", re.I), re.compile(r"^-?\d+(\.\d+)?$")),
    (re.compile(r"date", re.I), re.compile(r"^\d{4}-\d{2}-\d{2}")),
    (re.compile(r"e-?mail", re.I), re.compile(r"@")),
    (re.compile(r"phone", re.I), re.compile(r"^\+?[\d\- ]{6,}$")),
    (re.compile(r"name", re.I), re.compile(r"^[A-Z][a-z]+( [A-Z][a-z]+)+$")),
]


def _shaped(pname: str, cands: list[str]) -> list[str]:
    for npat, vpat in _SHAPES:
        if npat.search(pname):
            return [c for c in cands if vpat.search(c)]
    return list(cands)


def _fill(p: ParamInfo, state: str, rng: random.Random, need_id: bool = False) -> Any:
    """A value for a parameter of a synthetic proposal: bounded -> a seeded member; copy -> a seeded span of the
    state of the parameter's shape; free -> the fixed template. None when nothing of the right shape is visible."""
    kind = p.kind
    if kind == "bounded" and p.bounded_values():
        return rng.choice(p.bounded_values())     # type: ignore[arg-type]
    if kind == "copy":
        c = _shaped(p.name, extract_candidates(state, limit=200))
        if need_id:
            c = [x for x in c if re.search(r"\d", x)]
        if not c:
            return None
        v = rng.choice(c)
        if p.type in ("number", "integer"):
            try:
                return float(v) if p.type == "number" else int(float(v))
            except ValueError:
                return None
        return v
    return FREE_TEMPLATE


def _tools(r: _Runner, s: "StepSnap") -> dict[str, dict[str, Any]]:
    return {tool_name(t): t for t in s.tools if tool_name(t) != r.final}


def edit_t1(r: _Runner, s: "StepSnap", rng: random.Random) -> tuple[str, dict[str, Any]] | None:
    """T1: swap the tool for an offered near-duplicate that shadows it (shared name tokens / parameter names), with
    the arguments mapped by name; other required arguments come from the state (``_fill``)."""
    name, args = s.a0
    tools = _tools(r, s)
    if name not in tools:
        return None
    kinds = r.agent._param_kinds(s.env)
    src_p = {p.name for p in params_of(tools[name], kinds)}
    toks = set(name.split("_"))
    scored = []
    for n, t in tools.items():
        if n == name:
            continue
        pn = {p.name for p in params_of(t, kinds)}
        st = len(toks & set(n.split("_")))
        if not st and not (src_p & pn):
            continue
        scored.append(((st, len(src_p & pn), difflib.SequenceMatcher(None, name, n).ratio()), n))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    target = rng.choice([n for _, n in scored[:3]])
    out: dict[str, Any] = {}
    for p in params_of(tools[target], kinds):
        if p.name in args:
            out[p.name] = args[p.name]
        elif p.required:
            v = _fill(p, s.state, rng)
            if v is None:
                return None
            out[p.name] = v
    return target, out


def edit_t2(r: _Runner, s: "StepSnap", rng: random.Random) -> tuple[str, dict[str, Any]] | None:
    """T2: swap an ID argument for another same-typed span of the state (``extract_candidates``)."""
    name, args = s.a0
    tools = _tools(r, s)
    if name not in tools:
        return None
    ps = [p for p in params_of(tools[name], r.agent._param_kinds(s.env))
          if p.kind == "copy" and isinstance(args.get(p.name), str) and re.search(r"\d", args[p.name])]
    if not ps:
        return None
    p = rng.choice(ps)
    cur = str(args[p.name])
    prefix = re.match(r"^[A-Za-z]*[-_#]?", cur).group(0)     # type: ignore[union-attr]
    cands = [c for c in extract_candidates(s.state, limit=200)
             if c != cur and re.search(r"\d", c) and c.startswith(prefix) and re.fullmatch(r"[\w\-#.]+", c)
             and abs(len(c) - len(cur)) <= 2]
    if not cands:
        return None
    return name, {**args, p.name: rng.choice(cands)}


def edit_t3(r: _Runner, s: "StepSnap", rng: random.Random) -> tuple[str, dict[str, Any]] | None:
    """T3: flip an enum / boolean argument to another member."""
    name, args = s.a0
    tools = _tools(r, s)
    if name not in tools:
        return None
    ps = [p for p in params_of(tools[name], r.agent._param_kinds(s.env)) if p.kind == "bounded"
          and p.bounded_values() and p.name in args]
    if not ps:
        return None
    p = rng.choice(ps)
    others = [v for v in (p.bounded_values() or []) if str(v).strip().lower() != str(args[p.name]).strip().lower()]
    if not others:
        return None
    return name, {**args, p.name: rng.choice(others)}


def edit_t4(r: _Runner, s: "StepSnap", rng: random.Random) -> tuple[str, dict[str, Any]] | None:
    """T4: at a clarify state before the ask (placement by the oracle: its next step is ``ask_user``), propose the
    state-changing core tool implied by the instruction's verb with an ID taken from the state."""
    try:
        nxt = s.env.oracle_action()
    except Exception:  # noqa: BLE001
        return None
    if not nxt or nxt[0] != "ask_user":
        return None
    tools = _tools(r, s)
    verbs = [w for w in re.findall(r"[a-z]+", task_text(s.state).lower()) if w in STATE_CHANGING_VERBS]
    for verb in dict.fromkeys(verbs):
        cands = sorted((n for n in tools if n.split("_")[0] == verb), key=lambda n: (len(n), n))
        if not cands:
            continue
        target = cands[0]
        out: dict[str, Any] = {}
        ok = True
        for p in params_of(tools[target], r.agent._param_kinds(s.env)):
            if not p.required:
                continue
            v = _fill(p, s.state, rng, need_id=p.kind == "copy")
            if v is None:
                ok = False
                break
            out[p.name] = v
        if ok:
            return target, out
    return None


EDITS = {"T1": edit_t1, "T2": edit_t2, "T3": edit_t3, "T4": edit_t4}


# ---------------------------------------------------------------------- full-episode continuation (round 4, P1-6)
FULL_INTERVENTIONS = ("reflection", "large", "reject")
REJECTED_OBS = "(finish rejected by stop verifier)"      # the live stop guard's observation of a rejected finish
_SNAPSHOT_SINK: Callable[[Trajectory, list["StepSnap"]], None] | None = None


@contextmanager
def keep_snapshots() -> Iterator[dict[str, list["StepSnap"]]]:
    """Keep every episode's decision-time snapshots (``agent.StepSnap``: an independent env copy, the System-2
    messages WITH their provider extras such as Gemini thought signatures, the step's action and its gold / CF
    records) by task id while the block runs -- what :func:`continue_from_snapshot` resumes from. Outside the block
    the snapshots are discarded with the episode context, as before. Thread-safe for the runner's worker threads
    (one assignment per episode)."""
    global _SNAPSHOT_SINK
    store: dict[str, list["StepSnap"]] = {}
    prev = _SNAPSHOT_SINK

    def sink(traj: Trajectory, snaps: list["StepSnap"]) -> None:
        store[traj.task_id] = list(snaps)

    _SNAPSHOT_SINK = sink
    try:
        yield store
    finally:
        _SNAPSHOT_SINK = prev


@dataclass
class Rollout:
    """One full-episode continuation (:func:`continue_episode` / :func:`continue_from_snapshot`): the episode
    outcome after a logged state plus (optionally) one intervention at the first continued step. ``actions`` are the
    continuation's own decisions (the prefix excluded; a finish proposal is the last one), ``n_steps`` counts the
    steps the continuation decided (a rejected finish, consumed by the ``reject`` intervention, is not one of
    them); ``source`` is ``replay`` (history rebuilt from the log) or ``snapshot`` (resumed in process)."""
    task_id: str
    start_step: int
    intervention: str | None
    success: bool
    n_steps: int
    final_answer: str | None
    actions: list[tuple[str, dict[str, Any]]]
    prefix_match: bool | None
    cost_usd: float
    latency_ms: float
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    reflection: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    source: str = "replay"


def _obs_prefix_match(a: str, logged: str, k: int = 300) -> bool:
    k = min(len(a), len(logged), k)
    return a[:k] == logged[:k]


def replay_call_id(task_id: str, k: int) -> str:
    """The id of the k-th replayed call: ``call_`` + 6 digits, deterministic in (task, k) -- the shape of the ids
    Gemini's OpenAI-compatible endpoint generates itself (``call_300344``). Note (cf_validity smoke, 2026-10-01,
    gemini-3.5-flash-lite): a rebuilt history whose calls carry only the placeholder thought signature made the
    endpoint answer the next tool-calling request with ``finish_reason = MALFORMED_FUNCTION_CALL`` (no text, no
    call: the loop reads an empty finish) in 11 of 16 samples, whatever the id shape (``call_replay0000`` 4 / 4,
    hex 4 / 6, digits 3 / 6), while a fresh step-0 request and a text-only history never did. On Gemini, resume
    from in-process snapshots (:func:`continue_from_snapshot`), which keep the real signatures."""
    return "call_" + f"{int(hashlib.sha256(f'{task_id}|{k}'.encode()).hexdigest()[:8], 16) % 10 ** 6:06d}"


def _check_continuation(agent: "JevAgent", intervention: str | None, large_llm: Any) -> None:
    cfg = agent.cfg
    live = [dp.value for dp in DP if cfg.dp(dp).mode != Mode.OFF]
    if live or agent.s1 is not None and cfg.uses_s1():
        raise ValueError(f"the continuation needs the LLM-only harness (every decision point off); {cfg.name!r} "
                         f"has {live}")
    if intervention not in (None, *FULL_INTERVENTIONS):
        raise ValueError(f"unknown intervention {intervention!r}; choose from {FULL_INTERVENTIONS}")
    if intervention == "large" and large_llm is None:
        raise ValueError("intervention 'large' needs large_llm")


def _continue(agent: "JevAgent", env: Any, messages: list[dict[str, Any]], history: list[tuple[str, str]],
              obs: str, t0: int, intervention: str | None, ledger: Ledger, large_llm: Any,
              a0: tuple[str, dict[str, Any]] | None, match: bool | None, source: str,
              error: str | None = None) -> Rollout:
    """The LLM-only loop from step ``t0`` on a rebuilt (env, messages, history) -- see :func:`continue_episode`."""
    from .agent import _call_extras, _empty_responses, _Episode, _EXTRAS_TLS, _shadow_errors, _trunc

    cfg = agent.cfg
    final = cfg.final_tool
    n0 = len(ledger.records)
    actions: list[tuple[str, dict[str, Any]]] = []
    final_answer: str | None = None
    reflection: str | None = None
    start = t0
    result: dict[str, Any]
    _call_extras().clear()
    _shadow_errors().clear()
    _empty_responses().clear()
    ep = _EXTRAS_TLS.ep = _Episode(env=env)
    for m in (agent.llm, agent.small_llm, large_llm):      # simulation-only hook (see llm/simulated.py)
        if m is not None and hasattr(m, "bind_env"):
            m.bind_env(env)
    try:
        if error:
            raise ValueError(error)
        tools = env.tools() + [finish_tool(final)]
        names = {tool_name(t) for t in tools}
        if intervention == "reject":
            # the rejected proposal never enters the history; the note does, and the proposal consumed step t
            a0 = a0 or (final, {})
            append_note(messages, STOP_REJECT_NOTE)
            history.append((f"{a0[0]}({json.dumps(a0[1])[:200]})", REJECTED_OBS))
            start = t0 + 1
        for step in range(start, cfg.max_steps):
            ledger.step = step
            ep.step, ep.budget = step, cfg.max_steps - step
            slog = StepLog(step=step)
            state = agent._s1_state(env, history, obs, "", reflection or "")
            llm = agent.llm
            if step == start and intervention == "reflection":
                r = agent._llm_call(llm, messages + [{"role": "user", "content": REFLECT_PROMPT}], None, ledger,
                                    DP.REFLECTION.value, slog, temperature=0.0)
                reflection = r.text
                append_note(messages, f"[Reflection] {r.text}")
                state = agent._s1_state(env, history, obs, "", reflection)
            elif step == start and intervention == "large":
                llm = large_llm
            name, args, cid = agent._tool_step(llm, env, tools, messages, state, ledger, slog)
            agent._stop_check(env, state, name, ledger, slog, args)       # stop is off: finish iff `finish`
            actions.append((name, dict(args) if isinstance(args, dict) else {}))
            if name == final:
                final_answer = str((args or {}).get("answer", ""))
                break
            call = {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            extra = _call_extras().pop(cid, None)
            if extra:
                call["extra_content"] = extra
            messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            if name not in names:
                res_obs, done = f"Error: unknown tool {name!r}", False
            else:
                res = env.step(name, args)
                res_obs, done = res.observation, res.done
            messages.append({"role": "tool", "tool_call_id": cid, "content": _trunc(res_obs, cfg.obs_truncate_chars)})
            history.append((f"{name}({json.dumps(args)[:200]})", res_obs))
            obs = res_obs
            if done:
                break
        result = env.evaluate(final_answer)
    except Exception as e:  # noqa: BLE001 - a failed continuation is data, never a crash of the batch
        error = f"{type(e).__name__}: {str(e)[:300]}"
        try:
            result = env.evaluate(final_answer)
        except Exception:  # noqa: BLE001
            result = {"success": False, "score": 0.0}
    finally:
        _EXTRAS_TLS.ep = None
    recs = ledger.records[n0:]
    return Rollout(task_id=str(getattr(env, "task_id", "")), start_step=start, intervention=intervention,
                   success=bool(result.get("success", False)), n_steps=len(actions), final_answer=final_answer,
                   actions=actions, prefix_match=match, cost_usd=sum(r.cost_usd for r in recs),
                   latency_ms=sum(r.latency_ms for r in recs), result=dict(result), error=error,
                   reflection=reflection, source=source,
                   calls=[{"step": r.step, "model": r.model, "decision_point": r.decision_point,
                           "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "cost_usd": r.cost_usd,
                           "latency_ms": r.latency_ms, "cached_input_tokens": (r.meta or {}).get("cached_input_tokens", 0)}
                          for r in recs])


def continue_episode(agent: "JevAgent", env: Any, prefix: Sequence[tuple[str, dict[str, Any]]],
                     intervention: str | None = None, ledger: Ledger | None = None,
                     large_llm: "LLMClient | None" = None, logged_obs: Sequence[str] | None = None,
                     a0: tuple[str, dict[str, Any]] | None = None) -> Rollout:
    """Run the rest of an episode from a logged state, with one intervention applied ONCE at that state.

    ``agent`` must be the LLM-only harness (every decision point ``off``, no System-1): the one-step counterfactual
    labels (§4) are defined on the measurement arm, which behaves exactly like it. ``prefix`` is the logged
    ``(tool, args)`` of the steps before the snapshot, replayed through ``env`` (fresh, un-reset) to rebuild the
    world, the System-2 history and the heartbeat RNG exactly as the episode had them (``logged_obs``, when given,
    is checked against the replayed observations: ``Rollout.prefix_match``). A logged observation that starts with
    :data:`REJECTED_OBS` (a live stop-guard rejection) is replayed as the rejection note, not as a call. The
    replayed calls carry synthetic ids (:func:`replay_call_id`) and no provider thought signatures (the logs keep
    none; ``llm/openai_compat.py`` injects Gemini's documented placeholder), so a continuation is the model's
    behaviour on the *rebuilt* history, not a bit-identical resumption -- and on Gemini 3.5 that history mostly
    draws ``MALFORMED_FUNCTION_CALL`` (see :func:`replay_call_id`): there, resume from in-process snapshots with
    :func:`continue_from_snapshot` instead.

    ``intervention`` at the first continued step t (the snapshot step), mirroring the one-step branches:

      ``reflection``  the reflection call on the pre-step messages, ``append_note``, then the loop (``verdict_alt``)
      ``large``       the step's tool decision by ``large_llm``; later steps by the arm's own LLM (``verdict_large``)
      ``reject``      the snapshot step was a finish proposal ``a0``: the rejection note is appended and the proposal
                      consumes step t as in the live guard; the loop continues at t + 1 (``verdict_reject``)
      None            the control: the loop simply continues at t (the placebo resample, then the rest)

    Every call goes through ``ledger`` (the rollout's cost); nothing here touches the logged trajectory.
    """
    from .agent import DEFAULT_SYSTEM, _trunc

    _check_continuation(agent, intervention, large_llm)
    if env.admissible_actions() is not None:
        raise ValueError("continue_episode supports tool-calling envs only")
    ledger = ledger if ledger is not None else Ledger()
    cfg = agent.cfg
    final = cfg.final_tool
    match: bool | None = None if logged_obs is None else True
    error: str | None = None
    obs = env.reset()
    names = {tool_name(t) for t in env.tools()} | {final}
    messages: list[dict[str, Any]] = [{"role": "system", "content": agent.system_prompt or DEFAULT_SYSTEM},
                                      {"role": "user", "content": f"{env.instruction}\n\n{obs}"}]
    history: list[tuple[str, str]] = []
    for k, (name, args) in enumerate(prefix):
        args = dict(args or {})
        lo = logged_obs[k] if logged_obs is not None and k < len(logged_obs) else None
        if lo is not None and str(lo).startswith(REJECTED_OBS[:16]):     # a rejected finish of a live guard
            append_note(messages, STOP_REJECT_NOTE)
            history.append((f"{name}({json.dumps(args)[:200]})", REJECTED_OBS))
            continue
        if name == final:
            error = f"prefix step {k} is a finish proposal: the prefix must end before the snapshot"
            break
        cid = replay_call_id(getattr(env, "task_id", ""), k)
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        if name not in names:
            res_obs, done = f"Error: unknown tool {name!r}", False
        else:
            res = env.step(name, args)
            res_obs, done = res.observation, res.done
        if lo is not None and not _obs_prefix_match(res_obs, str(lo)):
            match = False
        messages.append({"role": "tool", "tool_call_id": cid, "content": _trunc(res_obs, cfg.obs_truncate_chars)})
        history.append((f"{name}({json.dumps(args)[:200]})", res_obs))
        obs = res_obs
        if done:
            error = "the prefix already ended the episode"
            break
    return _continue(agent, env, messages, history, obs, len(prefix), intervention, ledger, large_llm, a0, match,
                     "replay", error)


def continue_from_snapshot(agent: "JevAgent", snap: "StepSnap", intervention: str | None = None,
                           ledger: Ledger | None = None, large_llm: "LLMClient | None" = None) -> Rollout:
    """:func:`continue_episode` from an in-process decision-time snapshot (``agent.StepSnap``, kept with
    :func:`keep_snapshots`): the env is copied again (``oracle_snapshot``), so several continuations can start from
    one snapshot and the snapshot itself is never stepped; the System-2 messages are the episode's own, provider
    extras (Gemini thought signatures) included, so this IS the resumption the one-step branches were computed on.
    ``snap.a0`` is the rejected proposal of the ``reject`` intervention."""
    _check_continuation(agent, intervention, large_llm)
    ledger = ledger if ledger is not None else Ledger()
    snap_fn = getattr(snap.env, "oracle_snapshot", None)
    env = snap_fn() if callable(snap_fn) else copy.deepcopy(snap.env)
    messages = list(snap.messages)
    history: list[tuple[str, str]] = []
    obs = messages[1]["content"] if len(messages) > 1 else ""
    pending: str | None = None
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            f = m["tool_calls"][0]["function"]
            pending = f"{f['name']}({str(f.get('arguments', ''))[:200]})"
        elif m.get("role") == "tool":
            history.append((pending or "?", str(m.get("content", ""))))
            obs = str(m.get("content", ""))
            pending = None
    return _continue(agent, env, messages, history, obs, snap.step, intervention, ledger, large_llm, snap.a0, None,
                     "snapshot")


def run_deferred(agent: "JevAgent", traj: Trajectory, ledger: Ledger, ep: "_Episode") -> None:
    """Run the episode's deferred measurement: probe-list records (wording ablation, augmented stop trigger), the
    counterfactual branches (``measure.cf``) and the probes (``measure.probes``)."""
    r = _Runner(agent, traj, ledger, ep)
    if ep.extra_probes:
        r.extra_records()
    if agent.cfg.measure.cf is not None:
        r.counterfactuals()
    if agent.cfg.measure.probes is not None:
        r.probes()
    if _SNAPSHOT_SINK is not None:
        _SNAPSHOT_SINK(traj, ep.snaps)
