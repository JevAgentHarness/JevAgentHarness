"""E6b (exploratory; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6b", results/explore/e6b_modes/cells.md):
the cells added to the design-space grid -- tool_params assist / guard, memory_write cascade / guard, memory_read
cascade / trigger, and the reflection / stop / routing cascades. Offline: a tiny tool env with an oracle, a System-1
stub with fixed answers and confidences, a System-2 stub that records every request; plus the JevGym simulation for
every new mode and the fingerprint check that every mode that existed before E6b behaves and calls exactly as before
(``tests/e6b_fingerprint.py``, fixture generated on the pre-E6b code)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import e6b_fingerprint as FP  # noqa: E402

from jevharness.bench.env import Env, StepResult  # noqa: E402
from jevharness.bench.metrics import decision_records, is_delegated  # noqa: E402
from jevharness.bench.scorecard import call_coverage  # noqa: E402
from jevharness.bench.suites.jevgym import JevGymEnv  # noqa: E402
from jevharness.decision.base import DecisionBackend  # noqa: E402
from jevharness.decision.simulated import SimulatedBackend  # noqa: E402
from jevharness.decision.types import DecisionResult, QKind, Question, Usage, answer_from_distribution  # noqa: E402
from jevharness.harness.agent import JevAgent  # noqa: E402
from jevharness.harness.config import (ACTION_PENDING, DP, E6B_CELLS, EQUIVALENT_NOTES, UNDEFINED_CELLS,  # noqa: E402
                                       VALID_MODES, DPConfig, HarnessConfig, Mode)
from jevharness.harness.prompts import EARLY_STOP_PROMPT, REFLECT_PROMPT  # noqa: E402
from jevharness.llm.base import ChatResponse, LLMClient, ToolCall  # noqa: E402
from jevharness.llm.simulated import SimulatedLLM  # noqa: E402

GRID_MODES = (Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER)


# ------------------------------------------------------------------ the grid and the unchanged modes
def test_existing_modes_behave_and_call_exactly_as_before() -> None:
    """Every configuration of the modes that existed before E6b (one per mode, tests/e6b_fingerprint.py) produces the
    same decisions, executed calls, outcomes and ledger records (component, decision point, step, model, tokens,
    cost) as on the pre-E6b code."""
    exp = json.loads(FP.FIXTURE.read_text())["digests"]
    got = FP.all_digests()
    assert set(got) == set(exp)
    assert {k for k in exp if got[k] != exp[k]} == set()


def test_every_grid_cell_is_implemented_undefined_or_pending() -> None:
    cells = {(dp, m) for dp in DP for m in GRID_MODES}
    impl = {(dp, m) for dp, ms in VALID_MODES.items() for m in ms if m in GRID_MODES}
    undef, pending = set(UNDEFINED_CELLS), set(ACTION_PENDING)
    assert impl | undef | pending == cells and len(cells) == 40
    assert not (impl & undef) and not (impl & pending) and not (undef & pending)
    assert E6B_CELLS <= impl
    assert all(kind in ("undefined", "equivalent") and len(why) > 20 for kind, why in UNDEFINED_CELLS.values())
    # the binary meta decisions: replace is the trigger (or, for stop, the cascade) at 0.5, run as its own arm since
    # E6e (EQUIVALENT_NOTES names the cell it equals; tests/test_e6e_modes.py); their assist is E6d's "advise"
    # (tests/test_e6d_modes.py)
    for dp in (DP.MEMORY_WRITE, DP.REFLECTION, DP.ROUTING, DP.STOP):
        assert Mode.REPLACE in VALID_MODES[dp] and (dp, Mode.REPLACE) in EQUIVALENT_NOTES
        assert Mode.ASSIST in VALID_MODES[dp]
    # the argument trigger was undefined until E6f defined it as a verification trigger (tests/test_e6f_modes.py)
    HarnessConfig(tool_params=DPConfig(mode=Mode.TRIGGER)).validate_modes()
    with pytest.raises(ValueError, match="not supported"):
        HarnessConfig(tool_params=DPConfig(mode=Mode.LLM)).validate_modes()
    with pytest.raises(ValueError, match="checks the calls the LLM writes"):
        HarnessConfig(tool_select=DPConfig(mode=Mode.CASCADE), tool_params=DPConfig(mode=Mode.GUARD)).validate_modes()


# ------------------------------------------------------------------ a tiny tool env and stubs
def _fn(name: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": f"{name} tool",
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


TOOLS = [
    _fn("lookup_order", {"order_id": {"type": "string"}}, ["order_id"]),
    _fn("cancel_order", {"order_id": {"type": "string"},
                         "reason": {"type": "string", "enum": ["fraud", "late", "other", "duplicate"]}},
        ["order_id", "reason"]),
    _fn("set_priority", {"level": {"type": "integer", "minimum": 1, "maximum": 5}}, ["level"]),
]
FINISH = ("finish", {"answer": "done"})
PLANS = {
    "cancel": [("cancel_order", {"order_id": "ORD-1234", "reason": "fraud"}), FINISH],
    "lookup_cancel": [("lookup_order", {"order_id": "ORD-1234"}),
                      ("cancel_order", {"order_id": "ORD-1234", "reason": "fraud"}), FINISH],
    "priority": [("set_priority", {"level": 2}), FINISH],
}


class TinyEnv(Env):
    def __init__(self, plan: str, i: int = 0):
        self.plan = PLANS[plan]
        self.task_id = f"tiny-{plan}-{i}"
        self.instruction = ("Customer request: cancel order ORD-1234 (reason: fraud); order ORD-5678 and ORD-9012 are "
                            "fine.")

    def reset(self) -> str:
        self.i, self.done_calls = 0, []
        return "You have access to the order tools."

    def tools(self) -> list[dict[str, Any]]:
        return TOOLS

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        self.done_calls.append((tool, dict(args)))
        self.i += 1
        return StepResult(f"ok: {tool} ORD-1234")

    def oracle_action(self) -> tuple[str, dict[str, Any]] | None:
        return self.plan[self.i] if self.i < len(self.plan) else None

    def oracle_done(self) -> bool | None:
        return self.i >= len(self.plan) - 1

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        ok = self.done_calls == [(t, a) for t, a in self.plan[:-1]] and final_answer == FINISH[1]["answer"]
        return {"success": ok, "score": float(ok)}


class S1Stub(DecisionBackend):
    """System-1 stub. Choice: the option ``pick[key]`` (key ``tool_params:<argument>`` / ``tool_select``) or the
    simulation-only gold, else the first option; Noul: ``yes[dp]`` (default True). Confidence ``conf[key]`` /
    ``conf[dp]`` (default 0.95). Records every question (one ``decide`` = one request)."""

    name = "s1stub"

    def __init__(self, conf: dict[str, float] | None = None, yes: dict[str, bool] | None = None,
                 pick: dict[str, str] | None = None):
        self.conf, self.yes, self.pick = conf or {}, yes or {}, pick or {}
        self.max_options = 52
        self.requests: list[Question] = []

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        out = []
        for q in questions:
            self.requests.append(q)
            m = re.search(r"argument `([^`]+)`", q.text)
            key = f"{q.decision_point}:{m.group(1)}" if m else str(q.decision_point)
            c = self.conf.get(key, self.conf.get(str(q.decision_point), 0.95))
            labels = q.labels()
            if q.kind == QKind.NOUL:
                top = "yes" if self.yes.get(str(q.decision_point), True) else "no"
            else:
                gold = q.meta.get("gold")
                top = self.pick.get(key) or (str(gold) if gold is not None and str(gold) in labels else labels[0])
            rest = (1.0 - c) / max(1, len(labels) - 1)
            out.append(answer_from_distribution(q, {lab: (c if lab == top else rest) for lab in labels}))
        return DecisionResult(answers=out, usage=Usage(backend=self.name, model="stub", input_tokens=10,
                                                       latency_ms=1.0))


class S2Stub(LLMClient):
    """System-2 stub: answers the env's oracle call (the forced tool when ``tool_choice`` names one), with
    ``first_args`` overriding arguments of its first tool call of each step; yes / no gate prompts with ``gate``;
    reflection prompts with a note. Records every request."""

    provider = "scripted"

    def __init__(self, gate: str = "yes", first_args: dict[str, Any] | None = None, model: str = "s2"):
        super().__init__(model)
        self.gate, self.first_args = gate, dict(first_args or {})
        self.requests: list[dict[str, Any]] = []
        self.env: TinyEnv | None = None

    def bind_env(self, env: TinyEnv) -> None:
        self.env = env

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        forced = ((kw.get("tool_choice") or {}).get("function") or {}).get("name")
        last = str(messages[-1].get("content") or "")
        self.requests.append({"tools": [t["function"]["name"] for t in tools or []], "specs": tools,
                              "tool_choice": forced, "last": last, "kind": "call" if tools else "text"})
        if not tools:
            if last.endswith("Answer yes or no."):
                return ChatResponse(text=self.gate, input_tokens=50, output_tokens=1, latency_ms=2.0)
            return ChatResponse(text="Reflection: on track.", input_tokens=80, output_tokens=8, latency_ms=3.0)
        assert self.env is not None
        name, args = self.env.oracle_action() or FINISH
        args = dict(args)
        if forced:
            name = forced
            args = dict(self.env.oracle_action()[1]) if self.env.oracle_action() and \
                self.env.oracle_action()[0] == forced else ({"answer": "done"} if forced == "finish" else {})
        if self.first_args and "[Verifier]" not in last and name == "cancel_order":
            args = {**args, **{k: v for k, v in self.first_args.items() if k in args}}
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"c{len(self.requests)}", name=name, arguments=args)],
                            input_tokens=100, output_tokens=10, latency_ms=5.0)


def _run(cfg: HarnessConfig, plan: str, s1: S1Stub, llm: S2Stub | None = None,
         small: S2Stub | None = None) -> tuple[dict[str, Any], S1Stub, S2Stub]:
    llm = llm or S2Stub()
    agent = JevAgent(llm, cfg, s1=s1, small_llm=small)
    t = agent.run(TinyEnv(plan)).to_dict()
    assert t["error"] is None, t["error"]
    return t, s1, llm


def _calls(t: dict[str, Any], step: int, component: str, dp: str | None = None) -> list[dict[str, Any]]:
    return [c for c in t["calls"] if c["component"] == component and c["step"] == step
            and (dp is None or c["decision_point"] == dp)]


def _dec(t: dict[str, Any], step: int, dp: str) -> list[dict[str, Any]]:
    return [d for d in t["steps"][step]["decisions"] if d["dp"] == dp]


# ------------------------------------------------------------------ tool arguments
def test_tool_params_assist_shortlists_values_and_the_llm_fills_the_call_in_one_forced_call() -> None:
    cfg = HarnessConfig(name="tp_assist", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.9),
                        tool_params=DPConfig(mode=Mode.ASSIST, k=2))
    s1 = S1Stub({"tool_select": 0.95, "tool_params": 0.7})
    t, s1, llm = _run(cfg, "cancel", s1)
    # step 0: tool + order_id + reason questions; ONE forced System-2 call offering only cancel_order
    assert len(_calls(t, 0, "s1")) == 3
    (s2,) = _calls(t, 0, "s2")
    assert s2["decision_point"] == "tool_params"
    req = llm.requests[0]
    assert req["tool_choice"] == "cancel_order" and req["tools"] == ["cancel_order"]
    props = req["specs"][0]["function"]["parameters"]["properties"]
    assert props["reason"]["enum"][0] == "fraud" and len(props["reason"]["enum"]) == 2   # top-2 of 4 reasons
    assert props["order_id"]["enum"][0] == "ORD-1234" and len(props["order_id"]["enum"]) == 2
    assert "shortlists" in req["last"] and "`reason`: one of" in req["last"]
    assert TOOLS[1]["function"]["parameters"]["properties"]["reason"]["enum"] == ["fraud", "late", "other",
                                                                                  "duplicate"]  # not mutated
    tp = _dec(t, 0, "tool_params")
    assert {d["mode"] for d in tp} == {"assist"} and len(tp) == 2
    assert all(d["deferred"] and d["final_value"] == d["s2_value"] and d["gold_aux"]["binding"]
               and d["gold_aux"]["in_shortlist"] and not is_delegated(d) for d in tp)
    assert t["steps"][0]["args"] == {"order_id": "ORD-1234", "reason": "fraud"} and t["result"]["success"]
    # the finish call: answer is free text, no shortlist, one forced call
    assert [d["mode"] for d in _dec(t, 1, "tool_params")] == ["generate"]
    cv = call_coverage([t])
    assert cv["arg_coverage"] == 0.0 and cv["tool_coverage"] == 1.0


def test_tool_params_assist_without_a_binding_shortlist_restricts_nothing() -> None:
    # k = 5 covers all four reasons: the reason is not listed and its enum is the tool's own
    cfg = HarnessConfig(name="tp_assist5", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.9),
                        tool_params=DPConfig(mode=Mode.ASSIST, k=5))
    t, _, llm = _run(cfg, "cancel", S1Stub({"tool_select": 0.95}))
    props = llm.requests[0]["specs"][0]["function"]["parameters"]["properties"]
    assert props["reason"]["enum"] == ["fraud", "late", "other", "duplicate"]
    by = {re.search(r"argument `([^`]+)`", d["question"]).group(1): d for d in _dec(t, 0, "tool_params")}
    assert not by["reason"]["gold_aux"]["binding"] and "`reason`" not in llm.requests[0]["last"]
    assert by["order_id"]["gold_aux"]["binding"] and len(by["order_id"]["options"]) > 5   # 5 of the copy candidates


def test_tool_params_assist_lists_integer_shortlists_in_the_instruction_only() -> None:
    cfg = HarnessConfig(name="tp_assist_int", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.9),
                        tool_params=DPConfig(mode=Mode.ASSIST, k=2))
    t, _, llm = _run(cfg, "priority", S1Stub({"tool_select": 0.95}))
    props = llm.requests[0]["specs"][0]["function"]["parameters"]["properties"]
    assert "enum" not in props["level"] and "`level`: one of [2," in llm.requests[0]["last"]
    assert t["steps"][0]["args"] == {"level": 2}


def test_tool_params_assist_is_not_consulted_when_the_tool_is_deferred() -> None:
    cfg = HarnessConfig(name="tp_assist_def", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.9),
                        tool_params=DPConfig(mode=Mode.ASSIST, k=2))
    t, _, llm = _run(cfg, "cancel", S1Stub({"tool_select": 0.6}))
    assert not _dec(t, 0, "tool_params") and len(_calls(t, 0, "s1")) == 1
    assert llm.requests[0]["tool_choice"] is None and len(llm.requests[0]["tools"]) == 4


def test_tool_params_guard_vetoes_a_confidently_disputed_argument_once() -> None:
    cfg = HarnessConfig(name="tp_guard", tool_params=DPConfig(mode=Mode.GUARD, tau=0.9))
    s1 = S1Stub({"tool_params": 0.95})
    llm = S2Stub(first_args={"reason": "late"})          # the LLM's first proposal gets the reason wrong
    t, s1, llm = _run(cfg, "cancel", s1, llm)
    # step 0: the native call, two argument questions (order_id, reason), the veto, ONE retry at tool_params
    s2 = _calls(t, 0, "s2")
    assert [c["decision_point"] for c in s2] == ["tool_select", "tool_params"]
    assert len(_calls(t, 0, "s1", "tool_params")) == 2
    assert "argument `reason`" in llm.requests[1]["last"] and "[Verifier]" in llm.requests[1]["last"]
    assert "fraud" not in llm.requests[1]["last"].split("[Verifier]")[1].split("was judged")[1]  # no value suggested
    by = {re.search(r"argument `([^`]+)`", d["question"]).group(1): d for d in _dec(t, 0, "tool_params")}
    assert by["reason"]["gold_aux"]["veto"] and not by["order_id"]["gold_aux"]["veto"]
    assert by["reason"]["s2_value"] == "late" and by["reason"]["final_value"] == "fraud"
    assert by["reason"]["s1_value"] == "fraud" and by["reason"]["gold"] == "fraud" and by["reason"]["s2_correct"] is False
    assert t["steps"][0]["args"] == {"order_id": "ORD-1234", "reason": "fraud"} and t["result"]["success"]
    assert t["steps"][0]["llm_calls"] == 2


def test_tool_params_guard_without_confidence_or_dispute_never_vetoes() -> None:
    cfg = HarnessConfig(name="tp_guard", tool_params=DPConfig(mode=Mode.GUARD, tau=0.9))
    # unsure of the disputed value: no veto, the LLM's wrong reason is executed
    t, _, _ = _run(cfg, "cancel", S1Stub({"tool_params": 0.8}), S2Stub(first_args={"reason": "late"}))
    assert len(_calls(t, 0, "s2")) == 1 and t["steps"][0]["args"]["reason"] == "late"
    assert not any(d["gold_aux"]["veto"] for d in _dec(t, 0, "tool_params"))
    # sure and in agreement: no veto either
    t, _, _ = _run(cfg, "cancel", S1Stub({"tool_params": 0.99}))
    assert len(_calls(t, 0, "s2")) == 1 and t["result"]["success"]
    assert all(not d["gold_aux"]["dispute"] for d in _dec(t, 0, "tool_params"))


# ------------------------------------------------------------------ memory
def _mem_cfg(write: DPConfig, read: DPConfig | None = None) -> HarnessConfig:
    return HarnessConfig(name="mem", memory_write=write, memory_read=read or DPConfig(mode=Mode.ALWAYS))


def test_memory_write_cascade_uses_system1_when_sure_and_the_llm_gate_otherwise() -> None:
    t, _, llm = _run(_mem_cfg(DPConfig(mode=Mode.CASCADE, tau=0.9)), "lookup_cancel",
                     S1Stub({"memory_write": 0.95}, yes={"memory_write": False}))
    w = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_write"]
    assert w and all(d["mode"] == "cascade" and not d["deferred"] and d["final_value"] is False for d in w)
    assert not [c for c in t["calls"] if c["component"] == "s2" and c["decision_point"] == "memory_write"]
    assert t["meta"]["memory"]["stored"] == 0
    t, _, llm = _run(_mem_cfg(DPConfig(mode=Mode.CASCADE, tau=0.9)), "lookup_cancel",
                     S1Stub({"memory_write": 0.6}, yes={"memory_write": False}), S2Stub(gate="yes"))
    w = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_write"]
    gates = [c for c in t["calls"] if c["component"] == "s2" and c["decision_point"] == "memory_write"]
    assert len(gates) == len(w) == 2 and all(d["deferred"] and d["s2_value"] is True and d["final_value"] for d in w)
    assert all(d["s2_cost_usd"] is not None for d in w) and t["meta"]["memory"]["stored"] == 2
    assert t["steps"][0]["llm_calls"] == 2           # the step's tool call + the write gate
    # the stored memory reaches the next steps' LLM context (read: always)
    assert sum(r["last"].startswith("[Memory]") for r in llm.requests) == 2


def test_memory_write_guard_checks_only_proposed_writes() -> None:
    t, s1, _ = _run(_mem_cfg(DPConfig(mode=Mode.GUARD, tau=0.5)), "lookup_cancel",
                    S1Stub(yes={"memory_write": False}), S2Stub(gate="yes"))
    w = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_write"]
    assert len(w) == 2 and all(d["mode"] == "guard" and d["s2_value"] is True and d["s1_value"] is False
                               and d["final_value"] is False for d in w)
    assert t["meta"]["memory"]["stored"] == 0 and len([q for q in s1.requests if q.decision_point == "memory_write"]) == 2
    t, s1, _ = _run(_mem_cfg(DPConfig(mode=Mode.GUARD, tau=0.5)), "lookup_cancel", S1Stub(), S2Stub(gate="no"))
    assert not [q for q in s1.requests if q.decision_point == "memory_write"]      # nothing proposed, nothing asked
    w = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_write"]
    assert all(d["s1_value"] is None and d["final_value"] is False for d in w)


def test_memory_read_trigger_reads_the_recent_memories_only_when_it_fires() -> None:
    cfg = _mem_cfg(DPConfig(mode=Mode.ALWAYS), DPConfig(mode=Mode.TRIGGER, tau=0.9))
    t, s1, llm = _run(cfg, "lookup_cancel", S1Stub({"memory_read": 0.95}, yes={"memory_read": True}))
    r = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_read"]
    assert len(r) == 2 and all(d["mode"] == "trigger" and d["final_value"] is True for d in r)  # steps 1, 2
    assert "STORED MEMORIES: 1 earlier" in r[0]["state"]
    assert t["steps"][0]["decisions"][0]["dp"] != "memory_read"                     # empty store: not asked
    assert sum(r["last"].startswith("[Memory]") for r in llm.requests) == 2
    t2, _, llm2 = _run(cfg, "lookup_cancel", S1Stub({"memory_read": 0.95}, yes={"memory_read": False}))
    r2 = [d for s in t2["steps"] for d in s["decisions"] if d["dp"] == "memory_read"]
    assert len(r2) == 2 and all(d["final_value"] is False and d["gold_aux"]["n_injected"] == 0 for d in r2)
    assert sum(r["last"].startswith("[Memory]") for r in llm2.requests) == 0


def test_memory_read_cascade_falls_back_to_every_recent_memory_when_unsure() -> None:
    cfg = _mem_cfg(DPConfig(mode=Mode.ALWAYS), DPConfig(mode=Mode.CASCADE, tau=0.9, k=1))
    # two memories after step 1: the read question has 3 options (2 memories + none) > k
    t, _, _ = _run(cfg, "lookup_cancel", S1Stub({"memory_read": 0.95}))
    r = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_read"]
    assert r and all(d["mode"] == "cascade" and not d["deferred"] and len(d["final_value"]) <= 1 for d in r)
    t, _, _ = _run(cfg, "lookup_cancel", S1Stub({"memory_read": 0.5}))
    r = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_read"]
    assert r and all(d["deferred"] and len(d["final_value"]) == 2 for d in r)


# ------------------------------------------------------------------ reflection, stop, routing
def test_reflection_cascade_defers_to_an_llm_gate_on_the_same_question() -> None:
    cfg = HarnessConfig(name="rf_cascade", reflection=DPConfig(mode=Mode.CASCADE, tau=0.9))
    t, _, llm = _run(cfg, "lookup_cancel", S1Stub({"reflection": 0.95}, yes={"reflection": True}))
    rf = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "reflection"]
    assert len(rf) == 2 and all(not d["deferred"] and d["final_value"] is True for d in rf)    # steps 1, 2
    assert [c["decision_point"] for c in _calls(t, 1, "s2")] == ["reflection", "tool_select"]  # reflect, act
    assert not any(r["last"].endswith("Answer yes or no.") for r in llm.requests)
    t, _, llm = _run(cfg, "lookup_cancel", S1Stub({"reflection": 0.6}), S2Stub(gate="no"))
    rf = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "reflection"]
    assert all(d["deferred"] and d["s2_value"] is False and d["final_value"] is False for d in rf)
    gate = [r for r in llm.requests if r["last"].endswith("Answer yes or no.")]
    assert len(gate) == 2 and gate[0]["last"].startswith("TASK:") and "re-check" in gate[0]["last"]
    assert [c["decision_point"] for c in _calls(t, 1, "s2")] == ["reflection", "tool_select"]   # gate, act
    assert not any(REFLECT_PROMPT in r["last"] for r in llm.requests)
    assert t["steps"][1]["llm_calls"] == 2


def test_stop_cascade_rejects_a_finish_only_when_sure_and_can_stop_early() -> None:
    cfg = HarnessConfig(name="st_cascade", stop=DPConfig(mode=Mode.CASCADE, tau=0.9))
    # unsure at every step: the LLM's own decision stands (it finishes after cancel)
    t, _, _ = _run(cfg, "cancel", S1Stub({"stop": 0.7}, yes={"stop": False}))
    st = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "stop"]
    assert [d["deferred"] for d in st] == [True, True] and [d["final_value"] for d in st] == [False, True]
    assert t["result"]["success"]
    # sure "no" at the finish proposal: rejected (the note is added), the agent proposes finish again
    t, _, _ = _run(HarnessConfig(name="st", max_steps=4, stop=DPConfig(mode=Mode.CASCADE, tau=0.9)), "cancel",
                   S1Stub({"stop": 0.95}, yes={"stop": False}))
    st = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "stop"]
    rejects = [d for d in st if d["s2_value"] is True and d["final_value"] is False]
    assert rejects and rejects[0]["retry"] == 0 and t["steps"][1]["observation"].startswith("(finish rejected")
    assert call_coverage([t])["n_arg_slots"] == 2                 # rejected finishes were never executed
    # sure "yes" at a non-finish step: early stop, the LLM writes the answer in one forced finish call
    t, _, llm = _run(cfg, "lookup_cancel", S1Stub({"stop": 0.95}, yes={"stop": True}))
    assert t["steps"][0]["tool"] == "finish" and llm.requests[-1]["tool_choice"] == "finish"
    assert EARLY_STOP_PROMPT in llm.requests[-1]["last"]


def test_routing_cascade_uses_the_cheap_llm_as_its_fallback_judge() -> None:
    cfg = HarnessConfig(name="rt_cascade", routing=DPConfig(mode=Mode.CASCADE, tau=0.9))
    big, small = S2Stub(model="big"), S2Stub(model="small", gate="yes")
    with pytest.raises(ValueError, match="needs small_llm"):
        JevAgent(big, cfg, s1=S1Stub())
    t, _, _ = _run(cfg, "cancel", S1Stub({"routing": 0.95}, yes={"routing": False}), big, small)
    rt = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "routing"]
    assert all(not d["deferred"] and d["final_value"] == "small" for d in rt)
    assert {c["model"] for c in t["calls"] if c["component"] == "s2"} == {"small"}
    big, small = S2Stub(model="big"), S2Stub(model="small", gate="yes")
    t, _, _ = _run(cfg, "cancel", S1Stub({"routing": 0.6}, yes={"routing": False}), big, small)
    rt = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "routing"]
    assert all(d["deferred"] and d["s2_value"] == "large" and d["final_value"] == "large" for d in rt)
    gates = [c for c in t["calls"] if c["component"] == "s2" and c["decision_point"] == "routing"]
    assert len(gates) == 2 and {c["model"] for c in gates} == {"small"}
    assert {c["model"] for c in t["calls"] if c["component"] == "s2" and c["decision_point"] == "tool_select"} == {"big"}


# ------------------------------------------------------------------ JevGym simulation of every new cell
E6B_CONFIGS = {
    "tp_assist": HarnessConfig(name="tp_assist", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.9),
                               tool_params=DPConfig(mode=Mode.ASSIST, k=3)),
    "tp_guard": HarnessConfig(name="tp_guard", tool_params=DPConfig(mode=Mode.GUARD, tau=0.9)),
    "mw_cascade": HarnessConfig(name="mw_cascade", memory_write=DPConfig(mode=Mode.CASCADE, tau=0.9),
                                memory_read=DPConfig(mode=Mode.ALWAYS)),
    "mw_guard": HarnessConfig(name="mw_guard", memory_write=DPConfig(mode=Mode.GUARD, tau=0.5),
                              memory_read=DPConfig(mode=Mode.ALWAYS)),
    "mr_trigger": HarnessConfig(name="mr_trigger", memory_write=DPConfig(mode=Mode.ALWAYS),
                                memory_read=DPConfig(mode=Mode.TRIGGER, tau=0.9)),
    "mr_cascade": HarnessConfig(name="mr_cascade", memory_write=DPConfig(mode=Mode.ALWAYS),
                                memory_read=DPConfig(mode=Mode.CASCADE, tau=0.9, k=3)),
    "rf_cascade": HarnessConfig(name="rf_cascade", reflection=DPConfig(mode=Mode.CASCADE, tau=0.9)),
    "st_cascade": HarnessConfig(name="st_cascade", stop=DPConfig(mode=Mode.CASCADE, tau=0.9)),
    "rt_cascade": HarnessConfig(name="rt_cascade", routing=DPConfig(mode=Mode.CASCADE, tau=0.9)),
}


@pytest.mark.parametrize("name", sorted(E6B_CONFIGS))
def test_every_e6b_cell_runs_on_jevgym_with_golds_and_one_ledger(name: str) -> None:
    cfg = E6B_CONFIGS[name]
    llm = SimulatedLLM("scripted", error_rate=0.1, simulated_latency_ms=5, seed=0)
    small = SimulatedLLM("scripted-small", error_rate=0.3, simulated_latency_ms=2, seed=1)
    agent = JevAgent(llm, cfg, s1=SimulatedBackend(accuracy=0.8, seed=1), small_llm=small)
    trajs = [agent.run(JevGymEnv(seed=i, families="v2", strict=True, n_distractors=20, distractor_style="near",
                                 noise_lines=5)).to_dict() for i in range(6)]
    assert all(t["error"] is None for t in trajs), [t["error"] for t in trajs]
    recs = decision_records(trajs)
    (dp, mode), = [(dp, m) for dp, m in E6B_CELLS if cfg.dp(dp).mode == m]
    mine = [r for r in recs if r["dp"] == dp.value and r["mode"] == mode.value]
    assert mine, f"{name}: no {dp.value}/{mode.value} decision logged"
    # JevGym exposes the oracles: the new cell's records carry gold labels (status ok / doomed / unobserved)
    assert any(r.get("gold_status") == "ok" for r in mine), {r.get("gold_status") for r in mine}
    # one ledger: the episode's s1 + s2 cost is the sum of its call records; every call is attributed to a point
    for t in trajs:
        tot = sum(c["cost_usd"] for c in t["calls"])
        assert abs(t["usage"]["total"]["cost_usd"] - tot) < 1e-9
        assert all(c["decision_point"] for c in t["calls"])
        assert sum(s["llm_calls"] for s in t["steps"]) == sum(1 for c in t["calls"] if c["component"] == "s2")


E6B_WIRE = {
    "e6b_cascades": dict(tool_select={"mode": "cascade", "tau": 0.5}, tool_params={"mode": "assist", "k": 3},
                         memory_write={"mode": "cascade", "tau": 0.9}, memory_read={"mode": "cascade", "tau": 0.9, "k": 2},
                         reflection={"mode": "cascade", "tau": 0.9}, stop={"mode": "cascade", "tau": 0.9},
                         routing={"mode": "cascade", "tau": 0.9}),
    "e6b_guards": dict(tool_params={"mode": "guard", "tau": 0.5}, memory_write={"mode": "guard", "tau": 0.5},
                       memory_read={"mode": "trigger", "tau": 0.5}),
}


@pytest.mark.parametrize("canary_tool", [True, False])
@pytest.mark.parametrize("arm", list(E6B_WIRE))
def test_new_cells_leak_no_oracle_value_on_the_wire(arm: str, canary_tool: bool) -> None:
    """The request-level canary of tests/test_wire_canary.py on the E6b cells: the full loop on JevGym with an oracle
    whose every answer carries a canary, a Jev-wire System-1 and an OpenAI-compatible System-2 behind mock transports;
    no canary in any outgoing request (and the canaries do reach the gold side of the logs)."""
    import test_wire_canary as W

    s1_all, s2_all, dumps = W.Wire(W.jev_answers), W.Wire(W.jev_answers), []
    for seed in (0, 3):
        real = JevGymEnv(seed=seed, families="v2", distractor_style="near", n_distractors=60, noise_lines=5,
                         strict=True)
        env = W.CanaryOracleEnv(real, canary_tool)
        s1_wire, s2_wire = W.Wire(W.s1_policy(real)), W.Wire(W.s2_policy(real))
        from jevharness.decision.jev import JevBackend
        s1 = JevBackend(provider="openrouter", api_key="test-key", transport=s1_wire.transport())
        llm = W.compat_llm("gemini", "gemini-3.5-flash", s2_wire, temperature=0, reasoning_effort="minimal")
        small = W.compat_llm("gemini", "gemini-3.5-flash-lite", s2_wire, temperature=0)
        cfg = HarnessConfig(name=arm, max_steps=12, **E6B_WIRE[arm])
        traj = JevAgent(llm, cfg, s1=s1, small_llm=small).run(env)
        assert traj.error is None, traj.error
        s1_all.requests += s1_wire.requests
        s2_all.requests += s2_wire.requests
        dumps.append(json.dumps(traj.to_dict(), default=str))
    s1_all.assert_clean(min_requests=2, key_names=False)
    s2_all.assert_clean(min_requests=2, key_names=False)
    assert any(W.CANARY in d for d in dumps)
