"""Regression tests for the correctness review (docs/CODE_REVIEW.md). Each test pins one bug that
could have corrupted cost/latency accounting, decision logs or the agent loop. Offline only."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from jevharness.accounting import Ledger
from jevharness.bench.env import Env, StepResult
from jevharness.decision.base import CachedBackend, DecisionBackend
from jevharness.decision.calibration import records_to_examples
from jevharness.decision.simulated import OracleBackend, SimulatedBackend
from jevharness.decision.types import Choice, DecisionResult, Noul, Usage, answer_from_distribution
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.harness.tools import OTHER
from jevharness.harness.trajectory import Trajectory, load_trajectories
from jevharness.llm.base import ChatResponse, ToolCall
from jevharness.llm.scripted import ScriptedLLM


# ------------------------------------------------------------------ helpers
def _fn(name: str, props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": f"{name} tool", "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


class ToyEnv(Env):
    def __init__(self, tools: list[dict[str, Any]], oracle: tuple[str, dict[str, Any]] | None = None,
                 obs: str = "Customer order ORD-1 is late.", task_id: str = "toy",
                 actions: list[str] | None = None):
        self.task_id = task_id
        self.instruction = "Handle the customer's request."
        self._tools, self._oracle, self._obs, self._actions = tools, oracle, obs, actions
        self.received: list[tuple[str, dict[str, Any]]] = []

    def reset(self) -> str:
        return self._obs

    def tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def admissible_actions(self) -> list[str] | None:
        return self._actions

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        self.received.append((tool, dict(args)))
        return StepResult(f"ok: {tool}")

    def oracle_action(self):
        return self._oracle

    def oracle_done(self):
        return False

    def evaluate(self, final_answer):
        return {"success": final_answer is not None, "score": 0.0}


class ScriptedS1(DecisionBackend):
    """System-1 double: ``fn(question) -> partial distribution``; records every request."""

    name = "scripted_s1"
    model = "scripted-small"

    def __init__(self, fn):
        self.fn = fn
        self.batches: list[list[Any]] = []

    def _decide(self, state, questions):
        self.batches.append(list(questions))
        return DecisionResult(answers=[answer_from_distribution(q, self.fn(q)) for q in questions],
                              usage=Usage(backend=self.name, model=self.model))


def _decisions(traj: Trajectory, dp: str | None = None):
    return [d for s in traj.steps for d in s.decisions if dp is None or d.dp == dp]


def _fake_openai_response(prompt: int, completion: int, total: int, reasoning: int | None = None,
                          model: str = "gemini-3.5-flash"):
    cdet = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total,
                            prompt_tokens_details=None, completion_tokens_details=cdet)
    msg = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop", logprobs=None)],
                           usage=usage, model=model)


REPLACE = DPConfig(mode=Mode.REPLACE)


# ------------------------------------------------------------------ L1 decision eval
def test_decision_eval_never_batches_tool_select_with_tool_params_questions():
    """The tool_params question text names the gold function; batching it with the tool_select question
    of the same state leaked the tool_select answer into batched (e.g. verbalised) System-1 prompts."""
    from jevharness.bench.decision_eval import _s1_records
    from jevharness.bench.suites.static import NONE_TOOL, DecisionItem, norm

    state = "USER: What is the weather in Paris?"
    q_ts = Choice("Which function should be called to handle the user's request?",
                  ["get_weather", "get_time", NONE_TOOL], decision_point="tool_select")
    q_tp = Choice("The function `get_weather` will be called. Which value should its argument `city` take?",
                  ["Paris", OTHER], decision_point="tool_params")
    items = [DecisionItem("1/tool", "bfcl:multiple", "tool_select", state, q_ts, [norm("get_weather")]),
             DecisionItem("1/param/city", "bfcl:multiple", "tool_params", state, q_tp, [norm("Paris")])]
    s1 = ScriptedS1(lambda q: {q.options[0]: 0.9})
    recs = _s1_records(s1, items, batch_by_state=True, workers=1, name="rec")
    assert len(recs) == 2 and all(r["correct"] for r in recs)
    for batch in s1.batches:
        assert len({q.decision_point for q in batch}) == 1, "tool_select batched with tool_params"


def test_decision_eval_reports_errors_instead_of_silently_dropping_them():
    from jevharness.bench.decision_eval import _s1_records, summarize
    from jevharness.bench.suites.static import DecisionItem, norm

    class Boom(DecisionBackend):
        name = "boom"

        def _decide(self, state, questions):
            raise RuntimeError("quota exhausted")

    q = Choice("Which function?", ["f", "g"], decision_point="tool_select")
    items = [DecisionItem("1/tool", "bfcl:multiple", "tool_select", "USER: x", q, [norm("f")])]
    recs = _s1_records(Boom(), items, batch_by_state=True, workers=1, name="boom")
    assert recs[0]["source"] == "bfcl:multiple" and recs[0]["dp"] == "tool_select" and "error" in recs[0]
    ok = {"id": "2/tool", "source": "bfcl:multiple", "dp": "tool_select", "backend": "boom", "kind": "s1",
          "pred": "f", "gold": ["f"], "correct": True, "conf": 0.9, "latency_ms": 10.0}
    row = summarize(recs + [ok])["boom|tool_select|bfcl"]
    assert row["n"] == 1 and row["n_errors"] == 1


# ------------------------------------------------------------------ agent loop
def test_reflection_is_visible_to_the_same_steps_llm_call():
    seen: list[list[dict[str, Any]]] = []

    def policy(messages, tools):
        if tools is None:  # the reflection call
            return ChatResponse(text="REFLECTION-TEXT")
        seen.append(messages)
        n = len(seen)
        name, args = ("lookup", {"q": "x"}) if n < 3 else ("finish", {"answer": "a"})
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"c{n}", name=name, arguments=args)])

    cfg = HarnessConfig(name="reflect", max_steps=5, reflection=DPConfig(mode=Mode.ALWAYS))
    traj = JevAgent(ScriptedLLM("scripted", policy=policy), cfg).run(ToyEnv([_fn("lookup", {"q": {"type": "string"}})]))
    assert traj.error is None and traj.steps[1].reflection == "REFLECTION-TEXT"
    # step 1 paid for a reflection -> the step-1 tool call must be conditioned on it
    assert any("[Reflection]" in str(m.get("content")) for m in seen[1])


def test_reflection_reaches_the_system1_state_when_s1_decides():
    """action=replace + reflection: the reflection must be part of the state System-1 decides from,
    otherwise the reflection arm pays for LLM reflections that cannot influence any action."""
    env = ToyEnv([], actions=["go north", "go south"], obs="You are in a room.")
    s1 = ScriptedS1(lambda q: {"go north": 0.9})
    llm = ScriptedLLM("scripted", policy=lambda m, t: ChatResponse(text="REFLECTION-TEXT"))
    cfg = HarnessConfig(name="a", max_steps=3, action=REPLACE, reflection=DPConfig(mode=Mode.ALWAYS))
    traj = JevAgent(llm, cfg, s1=s1).run(env)
    assert traj.error is None and traj.steps[1].reflection == "REFLECTION-TEXT"
    states = [d.state for d in _decisions(traj, "action")]
    assert "REFLECTION-TEXT" not in states[0]  # no reflection at step 0
    assert "REFLECTION-TEXT" in states[1] and "REFLECTION-TEXT" in states[2]


def test_tool_params_gold_only_when_s1_picked_the_gold_tool():
    enum = {"type": "string", "enum": ["ORD-1", "ORD-2"]}
    tools = [_fn("cancel_order", {"order_id": enum}, ["order_id"]),
             _fn("refund_order", {"order_id": enum}, ["order_id"])]
    cfg = HarnessConfig(name="r", max_steps=1, tool_select=REPLACE, tool_params=REPLACE)
    for picked, want_gold in (("refund_order", None), ("cancel_order", "ORD-1")):
        s1 = ScriptedS1(lambda q, p=picked: {p: 0.9} if q.decision_point == "tool_select" else {"ORD-2": 0.9})
        env = ToyEnv(tools, oracle=("cancel_order", {"order_id": "ORD-1"}))
        traj = JevAgent(ScriptedLLM("scripted"), cfg, s1=s1).run(env)
        assert traj.error is None and env.received == [(picked, {"order_id": "ORD-2"})]
        tp = _decisions(traj, "tool_params")
        assert len(tp) == 1 and tp[0].gold == want_gold
        assert tp[0].s1_dist is not None and set(tp[0].s1_dist) == {"ORD-1", "ORD-2"}
        q_meta = [q.meta.get("gold") for b in s1.batches for q in b if q.decision_point == "tool_params"]
        assert q_meta == [want_gold]


def test_llm_param_fill_without_tool_call_does_not_inject_answer_arg():
    tools = [_fn("send_note", {"body": {"type": "string"}}, ["body"])]
    env = ToyEnv(tools, obs="Please send a note.")
    llm = ScriptedLLM("scripted", policy=lambda m, t: ChatResponse(text="I would write something nice."))
    cfg = HarnessConfig(name="r", max_steps=1, tool_select=REPLACE)
    traj = JevAgent(llm, cfg, s1=ScriptedS1(lambda q: {"send_note": 0.99})).run(env)
    assert traj.error is None and env.received and env.received[0][0] == "send_note"
    assert "answer" not in env.received[0][1]


def test_action_step_tolerates_an_empty_action_line():
    env = ToyEnv([], actions=["go north", "go south"], obs="You are in a room.")
    llm = ScriptedLLM("scripted", policy=lambda m, t: ChatResponse(text="Let me think.\nAction:"))
    traj = JevAgent(llm, HarnessConfig(name="a", max_steps=1)).run(env)
    assert traj.error is None, traj.error
    assert env.received and env.received[0][1]["action"] in {"go north", "go south"}


def test_bounded_param_values_keep_their_original_types():
    tools = [_fn("list_orders", {"order": {"enum": [None, "asc", "desc"]}, "limit": {"enum": [5]}},
                 ["order", "limit"])]
    env = ToyEnv(tools)
    s1 = ScriptedS1(lambda q: {"list_orders": 0.99} if q.decision_point == "tool_select" else {"asc": 0.9})
    cfg = HarnessConfig(name="r", max_steps=1, tool_select=REPLACE, tool_params=REPLACE)
    traj = JevAgent(ScriptedLLM("scripted"), cfg, s1=s1).run(env)
    assert traj.error is None, traj.error
    assert env.received == [("list_orders", {"order": "asc", "limit": 5})]


def test_thought_signature_extras_do_not_leak_across_episodes():
    phase = {"ep": 1}
    seen: list[list[dict[str, Any]]] = []

    def policy(messages, tools):
        seen.append(messages)
        if phase["ep"] == 1:  # finishes immediately; its extras are never echoed into a history
            return ChatResponse(text="", tool_calls=[ToolCall(id="call_1", name="finish", arguments={"answer": "a"},
                                                              extra={"google": {"thought_signature": "SIG-EP1"}})])
        if not any(m.get("role") == "tool" for m in messages):  # same id, no extras this time
            extra = {"google": {"thought_signature": "SIG-EP3"}} if phase["ep"] == 3 else None
            return ChatResponse(text="", tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"q": "x"},
                                                              extra=extra)])
        return ChatResponse(text="", tool_calls=[ToolCall(id="call_2", name="finish", arguments={"answer": "b"})])

    agent = JevAgent(ScriptedLLM("scripted", policy=policy), HarnessConfig(name="llm_only", max_steps=3))
    tools = [_fn("lookup", {"q": {"type": "string"}})]
    agent.run(ToyEnv(tools, task_id="ep1"))
    phase["ep"] = 2
    agent.run(ToyEnv(tools, task_id="ep2"))
    calls = [tc for m in seen[-1] if m.get("role") == "assistant" for tc in m.get("tool_calls") or []]
    assert calls and all("extra_content" not in tc for tc in calls), calls
    phase["ep"] = 3  # a genuine signature is still echoed back verbatim
    agent.run(ToyEnv(tools, task_id="ep3"))
    calls = [tc for m in seen[-1] if m.get("role") == "assistant" for tc in m.get("tool_calls") or []]
    assert calls[0]["extra_content"] == {"google": {"thought_signature": "SIG-EP3"}}


def test_noul_decisions_log_s1_dist_so_noul_calibration_can_be_fitted():
    def policy(messages, tools):
        done = any(m.get("role") == "tool" for m in messages)
        name, args = ("finish", {"answer": "a"}) if done else ("lookup", {"q": "x"})
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"c{len(messages)}", name=name, arguments=args)])

    cfg = HarnessConfig(name="g", max_steps=2, tool_select=DPConfig(mode=Mode.GUARD, tau=0.5),
                        stop=DPConfig(mode=Mode.GUARD, tau=0.5), reflection=DPConfig(mode=Mode.TRIGGER, tau=0.9),
                        routing=DPConfig(mode=Mode.TRIGGER, tau=0.5))
    s1 = ScriptedS1(lambda q: {"yes": 0.7, "no": 0.3})
    agent = JevAgent(ScriptedLLM("scripted", policy=policy), cfg, s1=s1, small_llm=ScriptedLLM("scripted-small",
                                                                                              policy=policy))
    traj = agent.run(ToyEnv([_fn("lookup", {"q": {"type": "string"}})], oracle=("lookup", {"q": "x"})))
    assert traj.error is None
    s1_logs = [d for d in _decisions(traj) if d.s1_value is not None]
    assert {d.dp for d in s1_logs} >= {"tool_select", "stop", "reflection", "routing"}
    for d in s1_logs:
        assert d.s1_dist is not None and abs(d.s1_dist["yes"] - 0.7) < 1e-6, d
    ex = records_to_examples([d.__dict__ for d in s1_logs])
    assert "noul" in ex and len(ex["noul"][0]) >= 2


def test_llm_only_decision_logs_carry_options_for_s2_teacher_harvest(tmp_path):
    from jevharness.training.data import harvest_trajectories

    def policy(messages, tools):
        done = any(m.get("role") == "tool" for m in messages)
        name, args = ("finish", {"answer": "a"}) if done else ("lookup", {"q": "x"})
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"c{len(messages)}", name=name, arguments=args)])

    traj = JevAgent(ScriptedLLM("scripted", policy=policy), HarnessConfig(name="llm_only")).run(
        ToyEnv([_fn("lookup", {"q": {"type": "string"}})]))
    d = _decisions(traj, "tool_select")[0]
    assert d.options == ["lookup", "finish"] and d.s2_value == "lookup"
    traj.dump(tmp_path / "llm_only.jsonl")
    ex = harvest_trajectories([tmp_path / "llm_only.jsonl"], label_source="s2")
    assert len(ex) == 2 and ex[0]["target"]["lookup"] == 1.0


def test_memory_llm_gate_counts_as_llm_call_in_step_log():
    def policy(messages, tools):
        if tools is None:
            return ChatResponse(text="yes")
        done = any(m.get("role") == "tool" for m in messages)
        name, args = ("finish", {"answer": "a"}) if done else ("lookup", {"q": "x"})
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"c{len(messages)}", name=name, arguments=args)])

    cfg = HarnessConfig(name="m", max_steps=3, memory_write=DPConfig(mode=Mode.LLM))
    traj = JevAgent(ScriptedLLM("scripted", policy=policy), cfg).run(ToyEnv([_fn("lookup", {"q": {"type": "string"}})]))
    assert traj.error is None and traj.usage["s2"]["calls"] == 3
    assert sum(s.llm_calls for s in traj.steps) == traj.usage["s2"]["calls"]


# ------------------------------------------------------------------ accounting
def test_unpriced_s1_model_is_flagged_in_ledger():
    from jevharness.decision.jev import JevBackend
    from jevharness.decision.llm_generative import GenerativeLLMBackend

    reply = ChatResponse(text=json.dumps({"yes": 0.9, "no": 0.1}))
    led = Ledger()
    GenerativeLLMBackend(ScriptedLLM("no-such-model-xyz", policy=lambda m, t: reply)).ask("S", Noul("ok?"), ledger=led)
    GenerativeLLMBackend(ScriptedLLM("scripted-small", policy=lambda m, t: reply)).ask("S", Noul("ok?"), ledger=led)
    assert led.summary()["unpriced_models"] == ["no-such-model-xyz"]

    def handler(cost):
        def h(request):
            usage = {"input_tokens": 100, **({"cost": cost} if cost is not None else {})}
            return httpx.Response(200, json={"model": "mystery-jev", "usage": usage,
                                             "answers": {"q0": {"type": "noul", "noul": 0.8}}})
        return h

    for cost, flagged in ((None, True), (0.00001, False)):
        b = JevBackend(model="mystery-jev", base_url="http://x", api_key="")
        b._http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler(cost)))
        led = Ledger()
        b.ask("S", Noul("ok?"), ledger=led)
        assert (led.summary()["unpriced_models"] == ["mystery-jev"]) is flagged


def test_openai_compat_latency_excludes_retry_backoff(monkeypatch):
    openai = pytest.importorskip("openai")
    from jevharness.llm.openai_compat import OpenAICompatClient

    c = OpenAICompatClient("gemini-3.5-flash", provider="openai", api_key="k", max_retries=2)
    n = {"calls": 0}

    class Fake:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **params):
            n["calls"] += 1
            if n["calls"] == 1:
                req = httpx.Request("POST", "https://x")
                raise openai.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)
            return _fake_openai_response(10, 2, 12, model="gemini-3.5-flash")

    c.clients = [Fake()]
    real_sleep = time.sleep
    monkeypatch.setattr("time.sleep", lambda s: real_sleep(0.3))
    r = c.chat([{"role": "user", "content": "x"}])
    assert n["calls"] == 2 and r.latency_ms < 250, r.latency_ms


def test_gemini_reasoning_tokens_are_billed_exactly_once():
    pytest.importorskip("openai")
    from jevharness.llm.openai_compat import OpenAICompatClient

    c = OpenAICompatClient("gemini-3.5-flash", provider="gemini", api_key="k")
    cases = [((100, 50, 150, 40), 50),   # completion already includes the 40 reasoning tokens
             ((100, 10, 150, None), 50),  # reasoning only visible through total_tokens
             ((100, 10, 150, 40), 50)]    # reasoning reported separately, excluded from completion
    for args, want in cases:
        resp = _fake_openai_response(*args)
        c._create = lambda params, resp=resp, **kw: resp  # type: ignore[method-assign]
        out = c._chat([{"role": "user", "content": "x"}])
        assert out.output_tokens == want, (args, out.output_tokens)


def test_logprob_permutation_latency_matches_sequential_execution():
    from jevharness.decision.llm_logprob import LogprobLLMBackend

    llm = ScriptedLLM("scripted-small", policy=lambda m, t: ChatResponse(text="A", latency_ms=100.0))
    r = LogprobLLMBackend(llm, n_permutations=2).decide("S", [Choice("x?", ["a", "b"])])
    assert r.usage.n_calls == 2 and r.usage.latency_ms == pytest.approx(200.0)


# ------------------------------------------------------------------ System-1 plumbing
def test_generic_server_handles_concurrent_requests():
    from jevharness.decision.server import build_app

    barrier = threading.Barrier(2, timeout=3)

    class NeedsTwo(OracleBackend):
        def _decide(self, state, questions):
            barrier.wait()  # only passes if two requests are inside the backend at the same time
            return super()._decide(state, questions)

    app = build_app(NeedsTwo(), "two")
    body = {"state": "S", "questions": {"q0": {"type": "noul", "instructions": "ok?"}}}

    async def main():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as cl:
            rs = await asyncio.gather(cl.post("/v1/systemone", json=body), cl.post("/v1/systemone", json=body))
        return [r.status_code for r in rs]

    assert asyncio.run(main()) == [200, 200]


def test_cached_backend_key_includes_noul_criteria(tmp_path):
    inner = ScriptedS1(lambda q: {"yes": 0.9})
    cb = CachedBackend(inner, tmp_path / "c.sqlite")
    cb.ask("S", Noul("done?", criteria={"true": "all tests pass", "false": "otherwise"}))
    cb.ask("S", Noul("done?", criteria={"true": "the file is saved", "false": "otherwise"}))
    assert len(inner.batches) == 2
    cb.ask("S", Noul("done?", criteria={"true": "the file is saved", "false": "otherwise"}, gold=True))
    assert len(inner.batches) == 2  # a true repeat (gold is not part of the request) is still a hit


def test_embedding_backend_does_not_embed_missing_descriptions_as_none():
    pytest.importorskip("openai")
    from jevharness.decision.embedding import EmbeddingBackend

    b = EmbeddingBackend(api_key="k")
    seen: list[str] = []

    class FakeEmb:
        def create(self, model, input):
            seen.extend(input)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, float(i)]) for i, _ in enumerate(input)],
                                   usage=SimpleNamespace(total_tokens=5))

    b.client = SimpleNamespace(embeddings=FakeEmb())
    b.decide("S", [Choice("which?", ["a", "b"], ["desc a", None])])
    assert "a: desc a" in seen and "b" in seen and not any("None" in s for s in seen)


def test_simulated_backend_answers_do_not_depend_on_unrelated_calls():
    """The runner shares one backend across worker threads; answers must not depend on which other
    questions happened to be drawn first (thread scheduling) or simulation studies are not reproducible."""
    qa = Choice("pick a", ["x", "y", "z"], gold="x")
    qb = Choice("pick b", ["x", "y", "z"], gold="y")
    b1 = SimulatedBackend(accuracy=0.6, seed=1)
    b1.ask("S", qa)
    d1 = b1.ask("S", qb).distribution
    d2 = SimulatedBackend(accuracy=0.6, seed=1).ask("S", qb).distribution
    assert d1 == d2


# ------------------------------------------------------------------ runner
def test_trajectory_dump_recovers_from_a_partial_trailing_line(tmp_path):
    from jevharness.bench.runner import _done_keys

    p = tmp_path / "arm.jsonl"
    p.write_text('{"task_id": "a", "result": {"success": tr')  # process killed mid-write
    Trajectory(task_id="b", config="arm", backend=None, llm="x").dump(p)
    assert _done_keys(p) == {"b"}
    with pytest.warns(UserWarning):
        recs = load_trajectories(p)
    assert [r["task_id"] for r in recs] == ["b"]


def test_expand_arms_rejects_duplicate_arm_names():
    from jevharness.bench.runner import expand_arms

    with pytest.raises(ValueError, match="duplicate"):
        expand_arms({"arms": [{"name": "a"}, {"name": "a", "stop": {"mode": "guard"}}]})
    sweeps = [{"base": {"name": "x", "tool_select": {"mode": "cascade"}}, "set": "tool_select.tau", "values": [0.5]},
              {"base": {"name": "x", "stop": {"mode": "guard"}}, "set": "stop.tau", "values": [0.5]}]
    with pytest.raises(ValueError, match="duplicate"):
        expand_arms({"sweeps": sweeps})
