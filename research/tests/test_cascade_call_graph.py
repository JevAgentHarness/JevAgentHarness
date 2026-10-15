"""The C1 call graph (PREREGISTRATION A6.22 (a); review round 8, W5): request traces of the tool + argument cascade in
``harness/agent.py`` (``_tool_decision`` -> ``_delegated_call`` -> ``_fill_params``), one test per branch, asserting how
many System-1 requests and System-2 calls each branch makes and which outputs are reused. Offline: a tiny deterministic
tool env, a System-1 stub with fixed confidences (answers the simulation-only ``Question.meta`` gold) and a System-2 stub
that records every request (tools offered, ``tool_choice``) and answers the env's oracle call.

Branches (per agent step):
  * tool deferred (System-1 below tau, or ``always_defer``): the LLM writes the whole call in ONE native call;
    ``tool_params`` is never consulted (no System-1 argument question, no tool_params record, whatever its tau);
  * tool accepted, every argument accepted, no free text: 0 System-2 calls (the native call is avoided);
  * tool accepted, a required bounded / copy argument deferred: ONE forced call (``tool_choice`` = the tool, only that
    tool offered); System-1's accepted values win on conflict; the deferred argument takes the LLM's value;
  * tool accepted with a free-text argument (``finish.answer``, an optional note): ONE forced call; System-1 is never
    asked about free text;
  * a deferred OPTIONAL argument with nothing else left for the LLM: no call, the argument is omitted;
  * a single-option bounded argument is filled by rule: no System-1 and no System-2 request.
System-1 requests per tool decision: 1 when the options fit one read (``max_options``), else one per tournament
round and chunk (one per round when the backend batches). The score-card coverage quantities of A6.22 (a)
(``decisions.coverage``, ``vs_baseline.llm_calls_avoided``) are checked on the same traces."""

from __future__ import annotations

import math
import re
from typing import Any

import pytest

from jevharness.bench.env import Env, StepResult
from jevharness.bench.scorecard import call_coverage, compute_scorecard
from jevharness.decision.base import DecisionBackend
from jevharness.decision.types import DecisionResult, Question, Usage, answer_from_distribution
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.llm.base import ChatResponse, LLMClient, ToolCall


def _fn(name: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": f"{name} tool",
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


# fill kinds (harness/tools.py, META_GOLD_SPEC §7): order_id copy (name *_id), reason / level / urgent bounded (enum,
# small integer range, boolean), channel bounded with ONE value (filled by rule), text free (plain string)
TOOLS = [
    _fn("lookup_order", {"order_id": {"type": "string"}}, ["order_id"]),
    _fn("cancel_order", {"order_id": {"type": "string"}, "reason": {"type": "string", "enum": ["fraud", "late", "other"]}},
        ["order_id", "reason"]),
    _fn("add_note", {"order_id": {"type": "string"}, "text": {"type": "string", "description": "note body"}},
        ["order_id"]),
    _fn("set_priority", {"level": {"type": "integer", "minimum": 1, "maximum": 3}, "urgent": {"type": "boolean"},
                         "channel": {"type": "string", "enum": ["email"]}}, ["level", "channel"]),
]
FINISH = ("finish", {"answer": "done"})
PLANS = {
    "cancel": [("cancel_order", {"order_id": "ORD-1234", "reason": "fraud"}), FINISH],
    "note": [("add_note", {"order_id": "ORD-1234", "text": "customer called"}), FINISH],
    "priority": [("set_priority", {"level": 2, "channel": "email"}), FINISH],
    "lookup_cancel": [("lookup_order", {"order_id": "ORD-1234"}),
                      ("cancel_order", {"order_id": "ORD-1234", "reason": "fraud"}), FINISH],
}


class TinyEnv(Env):
    """Follow a fixed plan; success iff the executed calls equal the plan (the LLM stub follows the oracle)."""

    def __init__(self, plan: str, i: int = 0):
        self.plan = PLANS[plan]
        self.task_id = f"tiny-{plan}-{i}"
        self.instruction = "Customer request: please cancel order ORD-1234 (reason: fraud) and add a note."

    def reset(self) -> str:
        self.i, self.done_calls = 0, []
        return "You have access to the order tools."

    def tools(self) -> list[dict[str, Any]]:
        return TOOLS

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        self.done_calls.append((tool, dict(args)))
        self.i += 1
        return StepResult(f"ok: {tool}")

    def oracle_action(self) -> tuple[str, dict[str, Any]] | None:
        return self.plan[self.i] if self.i < len(self.plan) else None

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        ok = self.done_calls == [(t, a) for t, a in self.plan[:-1]] and final_answer == FINISH[1]["answer"]
        return {"success": ok, "score": float(ok)}


class FixedS1(DecisionBackend):
    """Answers each question with its gold option (simulation-only ``Question.meta['gold']``; never on a wire) or its
    first option, at a confidence fixed per key: ``tool_select:<gold tool>``, ``tool_params:<argument>``, then the
    decision point, else 0.95. Records every request (one ``decide`` = one System-1 request)."""

    name = "fixed"

    def __init__(self, conf: dict[str, float] | None = None, max_options: int = 52, batched: bool = False):
        self.conf = conf or {}
        self.max_options = max_options
        self.batched_questions = batched
        self.requests: list[list[str]] = []

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        self.requests.append([q.text for q in questions])
        out = []
        for q in questions:
            labels = q.labels()
            gold = q.meta.get("gold")
            top = str(gold) if gold is not None and str(gold) in labels else labels[0]
            m = re.search(r"argument `([^`]+)`", q.text)
            key = f"{q.decision_point}:{m.group(1) if m else top}"
            c = self.conf.get(key, self.conf.get(str(q.decision_point), 0.95))
            rest = (1.0 - c) / max(1, len(labels) - 1)
            out.append(answer_from_distribution(q, {lab: (c if lab == top else rest) for lab in labels}))
        return DecisionResult(answers=out, usage=Usage(backend=self.name, model="fixed", input_tokens=10, latency_ms=1.0))


class RecordingLLM(LLMClient):
    """System-2 stub: answers the env's oracle call (the forced tool when ``tool_choice`` names one; then every
    argument value in ``conflict`` is replaced, to show which side wins) and records each request."""

    provider = "scripted"

    def __init__(self, conflict: dict[str, Any] | None = None):
        super().__init__("scripted-s2")
        self.conflict = conflict or {}
        self.requests: list[dict[str, Any]] = []
        self.env: TinyEnv | None = None

    def bind_env(self, env: TinyEnv) -> None:
        self.env = env

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        forced = ((kw.get("tool_choice") or {}).get("function") or {}).get("name")
        self.requests.append({"tools": [t["function"]["name"] for t in tools or []], "tool_choice": forced})
        assert self.env is not None
        name, args = self.env.oracle_action() or FINISH
        args = dict(args)
        if forced:
            args = {**args, **{k: v for k, v in self.conflict.items() if k in args}}
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"c{len(self.requests)}", name=name, arguments=args)],
                            input_tokens=100, output_tokens=10, latency_ms=5.0)


def _cascade(ts_tau: float = 0.9, tp_tau: float = 0.9, always_defer: bool = False, name: str = "c1") -> HarnessConfig:
    ts = DPConfig(mode=Mode.CASCADE, tau=ts_tau, extra={"always_defer": True} if always_defer else {})
    return HarnessConfig(name=name, tool_select=ts, tool_params=DPConfig(mode=Mode.CASCADE, tau=tp_tau))


def _run(cfg: HarnessConfig, plan: str, s1: FixedS1 | None = None, llm: RecordingLLM | None = None,
         n: int = 1) -> tuple[list[dict[str, Any]], FixedS1 | None, RecordingLLM]:
    llm = llm or RecordingLLM()
    agent = JevAgent(llm, cfg, s1=s1)
    trajs = [agent.run(TinyEnv(plan, i)).to_dict() for i in range(n)]
    for t in trajs:
        assert t["error"] is None, t["error"]
    return trajs, s1, llm


def _calls(t: dict[str, Any], step: int, component: str) -> list[dict[str, Any]]:
    return [c for c in t["calls"] if c["component"] == component and c["step"] == step
            and not (c.get("meta") or {}).get("shadow")]


def _decisions(t: dict[str, Any], step: int, dp: str) -> list[dict[str, Any]]:
    return [d for d in t["steps"][step]["decisions"] if d["dp"] == dp]


# ------------------------------------------------------------------ the branches
def test_tool_deferred_the_llm_writes_the_whole_call_and_tool_params_is_never_consulted():
    # System-1 is sure of every argument (and tool_params tau = 0 would accept anything), but below tau on the tool
    s1 = FixedS1({"tool_select": 0.6, "tool_params": 0.99})
    (t,), s1, llm = _run(_cascade(ts_tau=0.9, tp_tau=0.0), "cancel", s1)
    for step in (0, 1):
        assert len(_calls(t, step, "s1")) == 1                    # the tool question only
        s2 = _calls(t, step, "s2")
        assert len(s2) == 1 and s2[0]["decision_point"] == "tool_select"
        assert not _decisions(t, step, "tool_params")           # no argument question, no record
        (ts,) = _decisions(t, step, "tool_select")
        assert ts["deferred"] and ts["final_value"] == ts["s2_value"]
    # the native call: every tool offered, nothing forced; its arguments are executed as the LLM wrote them
    assert llm.requests[0] == {"tools": [f["function"]["name"] for f in TOOLS] + ["finish"], "tool_choice": None}
    assert t["steps"][0]["args"] == PLANS["cancel"][0][1]
    assert all("argument" not in q for r in s1.requests for q in r)
    assert t["steps"][0]["llm_calls"] == 1 and t["result"]["success"]


def test_always_defer_never_queries_system1_and_makes_one_native_call_per_step():
    s1 = FixedS1({"tool_params": 0.99})
    (t,), s1, llm = _run(_cascade(tp_tau=0.0, always_defer=True), "lookup_cancel", s1)
    assert s1.requests == []                                     # A6.11 (c).4: no System-1 cost or latency
    assert [len(_calls(t, s, "s2")) for s in range(3)] == [1, 1, 1]
    assert [r["tool_choice"] for r in llm.requests] == [None, None, None]
    assert not [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "tool_params"]
    cv = call_coverage([t])
    assert cv["tool_coverage"] == 0.0 and cv["arg_coverage"] == 0.0 and math.isnan(cv["arg_coverage_s1_calls"])
    assert cv["llm_free_steps"] == 0.0 and cv["n_arg_slots"] == 4   # lookup 1 + cancel 2 + finish 1


def test_tool_and_every_argument_accepted_makes_no_llm_call():
    s1 = FixedS1({"tool_select": 0.95, "tool_params": 0.95})
    (t,), s1, llm = _run(_cascade(), "cancel", s1)
    # step 0: one tool question + one question per bounded / copy argument (order_id, reason); no System-2 call
    assert len(_calls(t, 0, "s1")) == 3 and _calls(t, 0, "s2") == []
    assert t["steps"][0]["llm_calls"] == 0 and t["steps"][0]["args"] == PLANS["cancel"][0][1]
    tp = _decisions(t, 0, "tool_params")
    assert len(tp) == 2 and all(not d["deferred"] and d["final_value"] == d["s1_value"] for d in tp)
    # step 1: an accepted finish still costs ONE forced call -- its answer is free text, never System-1's
    assert len(_calls(t, 1, "s1")) == 1
    (s2,) = _calls(t, 1, "s2")
    assert s2["decision_point"] == "tool_params"
    assert llm.requests == [{"tools": ["finish"], "tool_choice": "finish"}]
    gen = _decisions(t, 1, "tool_params")
    assert [d["mode"] for d in gen] == ["generate"] and gen[0]["s1_value"] is None
    assert t["result"]["success"]


def test_one_required_argument_deferred_costs_one_forced_call_and_system1_values_win():
    s1 = FixedS1({"tool_select": 0.95, "tool_params": 0.95, "tool_params:reason": 0.6})
    llm = RecordingLLM(conflict={"order_id": "ORD-0000"})          # the forced call disagrees on order_id
    (t,), s1, llm = _run(_cascade(), "cancel", s1, llm)
    assert len(_calls(t, 0, "s1")) == 3                           # tool + order_id + reason
    (s2,) = _calls(t, 0, "s2")
    assert s2["decision_point"] == "tool_params"
    assert llm.requests[0] == {"tools": ["cancel_order"], "tool_choice": "cancel_order"}
    # System-1's accepted order_id is executed, the deferred reason is the LLM's
    assert t["steps"][0]["args"] == {"order_id": "ORD-1234", "reason": "fraud"}
    by = {re.search(r"argument `([^`]+)`", d["question"]).group(1): d for d in _decisions(t, 0, "tool_params")}
    assert not by["order_id"]["deferred"] and by["reason"]["deferred"]
    assert by["reason"]["s2_value"] == by["reason"]["final_value"] == "fraud"


def test_free_text_argument_forces_one_call_and_is_never_asked_of_system1():
    s1 = FixedS1({"tool_select": 0.95, "tool_params": 0.95})
    (t,), s1, llm = _run(_cascade(), "note", s1)
    assert len(_calls(t, 0, "s1")) == 2                           # tool + order_id; `text` is free
    assert len(_calls(t, 0, "s2")) == 1 and llm.requests[0]["tool_choice"] == "add_note"
    assert not any("argument `text`" in q for r in s1.requests for q in r)
    modes = sorted(d["mode"] for d in _decisions(t, 0, "tool_params"))
    assert modes == ["cascade", "generate"]                       # order_id (System-1) + text (generated)
    assert t["steps"][0]["args"] == PLANS["note"][0][1]


def test_deferred_optional_argument_alone_is_omitted_and_single_option_is_filled_by_rule():
    # level (required, bounded) accepted; channel (required) has one value: rule-filled; urgent (optional, bounded)
    # deferred: nothing is left for the LLM, so no call is made and `urgent` is left out of the executed call
    s1 = FixedS1({"tool_select": 0.95, "tool_params": 0.95, "tool_params:urgent": 0.6})
    (t,), s1, llm = _run(_cascade(), "priority", s1)
    assert len(_calls(t, 0, "s1")) == 3 and _calls(t, 0, "s2") == []   # tool + level + urgent; channel not asked
    assert not any("argument `channel`" in q for r in s1.requests for q in r)
    assert t["steps"][0]["args"] == {"level": 2, "channel": "email"}
    by = {re.search(r"argument `([^`]+)`", d["question"]).group(1): d for d in _decisions(t, 0, "tool_params")}
    assert set(by) == {"level", "urgent"} and by["urgent"]["deferred"] and by["urgent"]["final_value"] is None


def test_tool_choice_wider_than_one_read_is_a_tournament_of_system1_requests():
    # 5 options (4 tools + finish), 3 per read, 2 survivors per chunk: round 0 asks chunk 1 only (chunk 2 already fits
    # the survivors), round 1 asks one chunk, then the final read -> 3 System-1 requests; still no LLM call
    s1 = FixedS1({"tool_select": 0.95, "tool_params": 0.95}, max_options=3)
    (t,), s1, llm = _run(_cascade(), "lookup_cancel", s1)
    assert len([c for c in _calls(t, 0, "s1") if c["decision_point"] == "tool_select"]) == 3
    assert _calls(t, 0, "s2") == []


@pytest.mark.parametrize("max_options,batched,expected", [(255, True, 1), (62, True, 2), (62, False, 3)])
def test_requests_per_tool_decision_on_the_registered_suite_width(max_options, batched, expected):
    """The K = 69 options of the registered JevGym suite (68 tools + finish): one request for Jev (255 per read), two
    batched rounds for Intern-Decision (62 per read: chunks of 62 and 7 in one request, then the 4 finalists), three
    requests for a backend that reads 62 options but does not batch (A6.22 (a) 1)."""
    from jevharness.accounting import Ledger

    s1 = FixedS1(max_options=max_options, batched=batched)
    agent = JevAgent(RecordingLLM(), _cascade(), s1=s1)
    ledger = Ledger()
    agent._choice_tournament("TASK: x", "Which tool?", [f"tool_{i:02d}" for i in range(69)], "tool_select", ledger)
    assert len(s1.requests) == expected and len(ledger.records) == expected


# ------------------------------------------------------------------ the A6.22 (a) score-card quantities
def test_scorecard_reports_tool_and_argument_coverage_and_llm_calls_avoided():
    n = 4
    llm_only, _, _ = _run(HarnessConfig(name="llm_only"), "lookup_cancel", None, n=n)
    # lookup deferred (System-1 unsure of that tool), cancel and finish accepted
    s1 = FixedS1({"tool_select:lookup_order": 0.6, "tool_select": 0.95, "tool_params": 0.95})
    c1, _, _ = _run(_cascade(name="c1_cascade"), "lookup_cancel", s1, n=n)
    c1l, _, _ = _run(_cascade(always_defer=True, name="c1_always_defer"), "lookup_cancel", FixedS1(), n=n)
    card = compute_scorecard({"llm_only": llm_only, "c1_cascade": c1, "c1_always_defer": c1l}, baseline="llm_only",
                             n_boot=200)
    cv = card["arms"]["c1_cascade"]["decisions"]["coverage"]
    # tool: cancel + finish of 3 steps; arguments: System-1's order_id + reason of the 4 executed slots (lookup's
    # order_id was written by the LLM after the deferral; finish.answer is free text); inside System-1's calls: 2 / 3
    assert cv["tool_coverage"] == pytest.approx(2 / 3) and cv["arg_coverage"] == pytest.approx(2 / 4)
    assert cv["arg_coverage_s1_calls"] == pytest.approx(2 / 3) and cv["llm_free_steps"] == pytest.approx(1 / 3)
    assert cv["llm_calls_per_episode"] == 2.0 and cv["delegating"]
    vb = card["arms"]["c1_cascade"]["vs_baseline"]
    assert vb["llm_calls_avoided"] == pytest.approx(1.0) and vb["llm_calls_avoided_frac"] == pytest.approx(1 / 3)
    cl = card["arms"]["c1_always_defer"]
    assert cl["decisions"]["coverage"]["tool_coverage"] == 0.0 and cl["decisions"]["coverage"]["arg_coverage"] == 0.0
    assert math.isnan(cl["decisions"]["coverage"]["arg_coverage_s1_calls"])
    assert cl["vs_baseline"]["llm_calls_avoided"] == 0.0
    assert not card["arms"]["llm_only"]["decisions"]["coverage"]["delegating"]
    from jevharness.bench.scorecard import to_markdown

    md = to_markdown(card)
    assert "### Delegation coverage and native LLM calls (A6.22 (a))" in md
    assert "| c1_cascade | 66.7 | 50.0 (8/16) | 66.7 | 33.3 | 2.00 | +1.00 | 33.3 | llm_only |" in md


def test_s1_cache_hits_replay_the_answer_and_are_counted(tmp_path):
    """A6.22 (e) 8: a System-1 cache hit replays the stored answer with its original usage (``cached`` on the call
    record); the score card reports the share of acting System-1 calls served from the cache. System-2 is never
    cached: the second run makes its LLM calls afresh."""
    from jevharness.bench.scorecard import s1_cached_frac
    from jevharness.decision.base import CachedBackend

    inner = FixedS1({"tool_select": 0.95, "tool_params": 0.95})
    s1 = CachedBackend(inner, tmp_path / "s1.sqlite")
    llm = RecordingLLM()
    agent = JevAgent(llm, _cascade(), s1=s1)
    first = agent.run(TinyEnv("cancel")).to_dict()
    n_inner = len(inner.requests)
    second = agent.run(TinyEnv("cancel")).to_dict()              # identical states and questions
    assert len(inner.requests) == n_inner                        # every System-1 answer came from the cache
    assert s1_cached_frac([first]) == 0.0 and s1_cached_frac([second]) == 1.0
    assert s1_cached_frac([first, second]) == pytest.approx(0.5)
    assert [c["cost_usd"] for c in second["calls"] if c["component"] == "s1"] == \
        [c["cost_usd"] for c in first["calls"] if c["component"] == "s1"]
    assert len(llm.requests) == 2                                # one forced finish call per run, both fresh
    assert second["steps"][0]["args"] == first["steps"][0]["args"]
