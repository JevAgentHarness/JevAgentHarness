"""Offline tests of the tau2-bench adapter (``jevharness.bench.suites.tau2``).

Runs the real tau2 orchestrator + ``UserSimulator`` + mock domain (DB, tools, evaluator) with every
LLM call stubbed at ``tau2.utils.llm_utils.completion`` (the litellm entry point used by tau2's
``generate``), and the ``oracle`` / ``simulated`` / constant System-1 backends. No network needed.
Skipped when tau2-bench is not installed (see docs/TAU2.md).
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # litellm: no remote price-map fetch
pytest.importorskip("tau2.registry", reason="tau2-bench not installed (see docs/TAU2.md)")

import tau2.utils.llm_utils as tau2_llm_utils  # noqa: E402
from litellm import ModelResponse  # noqa: E402
from tau2.orchestrator.orchestrator import Orchestrator  # noqa: E402
from tau2.runner import build_environment, build_user, get_tasks, run_simulation  # noqa: E402

from jevharness.bench import threshold_theory as TT  # noqa: E402
from jevharness.bench.metrics import decision_records, decision_table, episode_table  # noqa: E402
from jevharness.bench.report import report  # noqa: E402
from jevharness.bench.suites import tau2 as jt  # noqa: E402
from jevharness.decision.base import DecisionBackend  # noqa: E402
from jevharness.decision.simulated import OracleBackend, SimulatedBackend  # noqa: E402
from jevharness.decision.types import DecisionResult, QKind, Usage, answer_from_distribution  # noqa: E402
from jevharness.harness.trajectory import load_trajectories  # noqa: E402

AGENT_LLM, USER_LLM, SMALL_LLM = "gpt-4.1-mini", "gpt-4.1-mini", "gpt-4.1-nano"
GOOD = {"user_id": "user_1", "title": "Important Meeting"}
REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ stubs
def _response(model: str, content: str | None = None, call: tuple[str, dict] | None = None) -> ModelResponse:
    msg: dict = {"role": "assistant", "content": content}
    if call is not None:
        msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                              "function": {"name": call[0], "arguments": json.dumps(call[1])}}]
    return ModelResponse(model=model, choices=[{"index": 0, "message": msg,
                                                "finish_reason": "tool_calls" if call else "stop"}],
                         usage={"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230})


class FakeLLM:
    """Scripted stand-in for litellm.completion, for mock task ``create_task_1`` (user_1 wants a task
    titled 'Important Meeting'). Serves the agent (system prompt has <policy>) and the user simulator
    (system prompt has <scenario>).

    agent scripts: good | wrong_first (wrong title until a verifier note) | transfer_first | ignore_none (under
    tool_choice="none", before the task exists, a text-less call to the undeclared tool `get_line`)
    """

    def __init__(self, script: str = "good"):
        self.script = script
        self.agent_calls: list[dict] = []
        self.user_turns = 0

    def __call__(self, model, messages, tools=None, tool_choice=None, **kw):
        if "<scenario>" in (messages[0].get("content") or ""):
            return self._user(model, messages)
        return self._agent(model, messages, tools, tool_choice)

    def _agent(self, model, messages, tools, tool_choice):
        names = [t["function"]["name"] for t in tools or []]
        last = messages[-1]
        self.agent_calls.append({"model": model, "tool_choice": tool_choice, "tools": names,
                                 "last": last.get("content")})
        if isinstance(tool_choice, dict):  # forced by the harness (tool_params hint)
            fn = tool_choice["function"]["name"]
            return _response(model, call=(fn, GOOD if fn == "create_task" else {"summary": "x"}))
        created = any(m["role"] == "tool" for m in messages)
        if created:
            return _response(model, "Done: the task 'Important Meeting' has been created for user_1.")
        if self.script == "ignore_none" and tool_choice == "none":  # no text, a call to an undeclared tool
            return _response(model, call=("get_line", {"line_id": "L1"}))
        if tool_choice == "none" or "create_task" not in names:
            return _response(model, "Sure. Shall I create the task 'Important Meeting' for user_1?")
        noted = last["role"] == "user" and str(last.get("content") or "").startswith("[")
        if self.script == "wrong_first" and not noted:
            return _response(model, call=("create_task", {"user_id": "user_1", "title": "Meeting"}))
        if self.script == "transfer_first" and not noted:
            return _response(model, call=("transfer_to_human_agents", {"summary": "wants a task"}))
        return _response(model, call=("create_task", GOOD))

    def _user(self, model, messages):
        self.user_turns += 1
        last = str(messages[-1].get("content") or "").lower()
        if self.user_turns > 8 or "created" in last:
            return _response(model, "Thanks! ###STOP###")
        if "?" in last and "create" in last:
            return _response(model, "Yes, please go ahead.")
        return _response(model, "Hi! Please create a new task called 'Important Meeting' for user_1.")


class ConstBackend(DecisionBackend):
    """Noul -> p_yes; Choice -> uniform. Deterministic S1 for routing tests."""

    name = "const"

    def __init__(self, p_yes: float):
        self.p_yes = p_yes

    def _decide(self, state, questions):
        out = []
        for q in questions:
            labs = q.labels()
            dist = {"yes": self.p_yes, "no": 1 - self.p_yes} if q.kind == QKind.NOUL else \
                {lab: 1 / len(labs) for lab in labs}
            out.append(answer_from_distribution(q, dist))
        return DecisionResult(answers=out, usage=Usage(backend="const", input_tokens=100, cost_usd=1e-6,
                                                       latency_ms=2.0))


class RespondBackend(DecisionBackend):
    """tool_select -> respond_to_user at 0.97 (as JevAgentRL-4B on tau^2 telecom, E1 go/no-go round 1); else uniform."""

    name = "respond"

    def _decide(self, state, questions):
        out = []
        for q in questions:
            labs = q.labels()
            if q.kind == QKind.NOUL:
                dist = {"yes": 0.5, "no": 0.5}
            elif jt.RESPOND in labs:
                dist = {lab: 0.97 if lab == jt.RESPOND else 0.03 / (len(labs) - 1) for lab in labs}
            else:
                dist = {lab: 1 / len(labs) for lab in labs}
            out.append(answer_from_distribution(q, dist))
        return DecisionResult(answers=out, usage=Usage(backend="respond", input_tokens=10, cost_usd=0.0,
                                                       latency_ms=1.0))


class PricedOracle(OracleBackend):
    """The oracle with a price and a latency, so the shadow cost accounting can be asserted."""

    def __init__(self):
        super().__init__()
        self.price, self.latency_ms = 0.042, 5.0


class DownBackend(DecisionBackend):
    """A System-1 that always fails (shadow mode must be fail-open)."""

    name = "down"

    def _decide(self, state, questions):
        raise RuntimeError("S1 is down")


SHADOW_CFG = dict(tool_select={"mode": "shadow"}, tool_params={"mode": "shadow"}, stop={"mode": "shadow"},
                  routing={"mode": "shadow"}, policy_guard={"mode": "shadow"}, s1={"type": "oracle"})


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setattr(tau2_llm_utils, "completion", f)
    return f


@pytest.fixture
def clean_env(monkeypatch):
    # "" == unset for the adapter; setenv records the original value so direct writes get undone
    for k in (jt.ENV_CONFIG, jt.ENV_OVERRIDES, jt.ENV_LOG_DIR):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(sys, "argv", list(sys.argv))


def _cfg(**kw) -> jt.Tau2JevConfig:
    return jt.Tau2JevConfig.from_dict({"name": "test", **kw})


def run_episode(cfg, s1=None, log_dir=None, seed=7, task_id="create_task_1"):
    task = get_tasks("mock", task_ids=[task_id])[0]
    env = build_environment("mock")
    agent = jt.JevTau2Agent(env.get_tools(), env.get_policy(), llm=AGENT_LLM, llm_args={"temperature": 0.0},
                            jev_config=cfg, task=task, s1=s1, log_dir=log_dir)
    user = build_user("user_simulator", env, task, llm=USER_LLM, llm_args={"temperature": 0.0})
    orch = Orchestrator(domain="mock", agent=agent, user=user, environment=env, task=task, max_steps=40,
                        max_errors=5, seed=seed)
    return run_simulation(orch), agent


def _tool_calls(sim) -> list:
    return [tc for m in sim.messages if m.role == "assistant" for tc in (m.tool_calls or [])]


def _decisions(agent, dp=None) -> list:
    return [d for s in agent.steps for d in s.decisions if dp is None or d.dp == dp]


# ------------------------------------------------------------------ config
def test_config_loading_overrides_and_validation(tmp_path, monkeypatch, clean_env):
    p = tmp_path / "arm.yaml"
    p.write_text("name: jev_ts_cascade\ntool_select: {mode: cascade, tau: 0.8}\ns1: {type: oracle}\n")
    monkeypatch.setenv(jt.ENV_CONFIG, str(p))
    monkeypatch.setenv(jt.ENV_OVERRIDES, json.dumps({"tool_select.tau": 0.9}))
    cfg = jt.load_config()
    assert cfg.harness.tool_select.tau == 0.9 and cfg.name == "jev_ts_cascade_tau0.9"
    assert jt.parse_overrides(["s1={type: simulated, accuracy: 0.8}", "k=3"]) == \
        {"s1": {"type": "simulated", "accuracy": 0.8}, "k": 3}
    for bad in ({"reflection": {"mode": "trigger"}, "s1": "oracle"},   # not implemented for tau2
                {"reflection": {"mode": "shadow"}, "s1": "oracle"},
                {"stop": {"mode": "trigger"}, "s1": "oracle"},         # the user simulator ends conversations
                {"routing": {"mode": "trigger"}, "s1": "oracle"},      # no small_llm
                {"tool_select": {"mode": "replace"}},                  # S1 used but no backend
                {"tool_select": {"mode": "shadow"}},                   # shadow needs a backend too
                {"policy_guard": {"mode": "guard"}},
                {"policy_guard": {"mode": "trigger"}, "s1": "oracle"}):
        with pytest.raises(ValueError):
            _cfg(**bad)
    # shadow mode (A5.4) at tool_select / tool_params / stop / routing / policy; routing shadow needs no small_llm
    sh = _cfg(**SHADOW_CFG)
    assert sh.uses_s1() and sh.shadow_points() == ["tool_select", "tool_params", "stop", "routing", "policy"]
    assert _cfg(name="llm_only").shadow_points() == []
    # every shipped arm config must load
    shipped = sorted((REPO / "configs" / "tau2").glob("*.yaml"))
    assert {q.stem for q in shipped} >= {"llm_only", "jev_ts_cascade", "jev_ts_assist_k3", "jev_guard",
                                         "jev_route", "jev_shadow"}
    for q in shipped:
        jt.Tau2JevConfig.load(q)
    js = jt.Tau2JevConfig.load(REPO / "configs" / "tau2" / "jev_shadow.yaml")
    assert js.name == "jev_shadow" and js.shadow_points() == ["tool_select", "tool_params", "stop"]


def test_airline_gold_heuristic_and_value_candidates():
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.data_model.message import ToolCall as T2Call

    env = build_environment("airline")
    task = [t for t in get_tasks("airline") if t.id == "7"][0]  # reference: 2 lookups, a change, 2 cancels
    agent = jt.JevTau2Agent(env.get_tools(), env.get_policy(), llm=AGENT_LLM, task=task)
    assert agent.kinds["cancel_reservation"] == "write" and agent.ref.requires_confirmation
    st = agent.get_init_state([AssistantMessage(role="assistant", content="Hi! How can I help you today?")])
    st.messages.append(UserMessage(role="user", content="I need help with my flights."))
    assert agent._view(st).gold_tool == jt.RESPOND  # no reservation id known yet -> ask
    st.messages.append(UserMessage(role="user", content="Please cancel reservation 59XX6W."))
    v = agent._view(st)
    assert (v.gold_tool, v.gold_args) == ("get_reservation_details", {"reservation_id": "59XX6W"})
    st.messages += [AssistantMessage(role="assistant", content=None, tool_calls=[
        T2Call(id="c1", name="get_reservation_details", arguments={"reservation_id": "59XX6W"})]),
        ToolMessage(id="c1", role="tool", content='{"reservation_id": "59XX6W"}')]
    assert agent._view(st).gold_tool == jt.RESPOND  # a WRITE needs an explicit "yes" first
    st.messages += [AssistantMessage(role="assistant", content="Cancel 59XX6W - confirm?"),
                    UserMessage(role="user", content="Yes, go ahead.")]
    v = agent._view(st)
    assert v.confirmed and v.gold_tool == "cancel_reservation"
    assert agent.ref.matches_pending(v.pending, "cancel_reservation", {"reservation_id": "59XX6W"})
    assert not agent.ref.matches_pending(v.pending, "cancel_reservation", {"reservation_id": "NOPE00"})
    tool_out = json.dumps({"user_id": "sara_doe_496",
                           "payment_methods": {"credit_card_7815826": {"source": "credit_card"}}})
    cands = jt._value_candidates(f"[tool] {tool_out}")
    assert {"sara_doe_496", "credit_card_7815826"} <= set(cands) and "user_id" not in cands


def test_value_candidates_are_param_aware():
    """Pilot regressions (docs/PILOT_RESULTS.md): ids given in free text were missed, date fragments and
    prices crowded the list, and the value recorded under the argument's own JSON key was not preferred."""
    conv = "\n".join([
        "[user] Hi, my user ID is raj_sanchez_7340 and I fly on 2024-05-17.",
        "[agent -> tool] get_user_details({\"user_id\": \"raj_sanchez_7340\"})",
        '[tool] {"user_id": "raj_sanchez_7340", "reservations": ["MZDDS4", "Q69X3R"], '
        '"payment_methods": {"gift_card_3226531": {"amount": 1167}}}',
        "[agent] **Date:** 2024-05-17, **Total:** 430 USD. Which reservation?",
        "[user] The one from Philadelphia.",
    ])
    free = jt._value_candidates(conv)
    assert "raj_sanchez_7340" in free and "gift_card_3226531" in free
    assert not {"2024", "05", "17"} & set(free) and "2024-05-17" in free   # date fragments dropped
    res = jt._value_candidates(conv, param="reservation_id", param_type="string")
    assert res[:2] == ["Q69X3R", "MZDDS4"]                                   # "reservations": [...] first
    assert jt._value_candidates(conv, param="user_id", param_type="string")[0] == "raj_sanchez_7340"
    s = jt._value_candidates(conv, param="origin", param_type="string")
    assert s.index("1167") > s.index("Philadelphia") and s.index("430") > s.index("Philadelphia")
    assert len(jt._value_candidates(conv, limit=3)) == 3
    assert jt._FREE_TEXT_PARAM.match("summary") and not jt._FREE_TEXT_PARAM.match("title")


# ------------------------------------------------------------------ episodes on the real mock domain
def test_llm_only_episode_logs_like_llm_agent(tmp_path, fake):
    sim, agent = run_episode(_cfg(name="llm_only"), log_dir=tmp_path)
    assert sim.reward_info.reward == 1.0, sim.reward_info
    assert agent.s1 is None and fake.agent_calls and all(c["tool_choice"] == "auto" for c in fake.agent_calls)
    (rec,) = jt.load_sidecar(tmp_path / jt.SIDECAR_FILE)
    assert rec["config"] == "llm_only" and rec["seed"] == 7 and rec["task_id"] == "create_task_1"
    assert rec["usage"]["s2"]["calls"] == len(fake.agent_calls) and "s1" not in rec["usage"]
    offs = [d for s in rec["steps"] for d in s["decisions"]]
    assert {(d["dp"], d["mode"]) for d in offs} == {("tool_select", "off")}
    assert all(jt.RESPOND in d["options"] and d["state"] and d["s2_value"] in d["options"] for d in offs)
    produced = [m for m in sim.messages if m.role == "assistant" and m.raw_data]
    assert produced and all(m.raw_data["jev"]["episode"] == agent.episode_id for m in produced)
    assert sim.agent_cost and sim.agent_cost > 0  # litellm-priced, tau2-native accounting still works
    assert agent.steps[0].oracle_tool == "create_task" and agent.steps[0].tool == "create_task"


def test_replace_with_oracle_s1_emits_tool_call_without_llm(fake):
    cfg = _cfg(tool_select={"mode": "replace"}, tool_params={"mode": "replace"}, s1={"type": "oracle"})
    sim, agent = run_episode(cfg, s1=OracleBackend())
    assert sim.reward_info.reward == 1.0
    first = agent.steps[0]
    assert first.tool == "create_task" and first.args == GOOD and first.llm_calls == 0
    ts = [d for d in first.decisions if d.dp == "tool_select"][0]
    assert ts.s1_value == ts.gold == ts.final_value == "create_task"
    assert {d.final_value for d in first.decisions if d.dp == "tool_params"} == set(GOOD.values())
    # after the write, S1 picks respond_to_user and the LLM only writes the message (tool_choice none)
    assert agent.steps[1].tool == jt.RESPOND and fake.agent_calls[-1]["tool_choice"] == "none"
    emitted = [m for m in sim.messages if m.role == "assistant" and m.tool_calls][0]
    assert emitted.raw_data["jev"]["llm_calls"] == 0 and emitted.cost == 0.0
    usage = agent.ledger.summary()
    assert usage["s1"]["calls"] >= 3 and usage["s2"]["calls"] == len(fake.agent_calls) == 1


def test_cascade_defers_to_llm_and_assist_prunes_tools(fake):
    cfg = _cfg(tool_select={"mode": "cascade", "tau": 1.0}, s1={"type": "simulated"})
    sim, agent = run_episode(cfg, s1=SimulatedBackend(accuracy=0.7, seed=0))
    assert sim.reward_info.reward == 1.0
    ts = _decisions(agent, "tool_select")
    assert ts and all(d.deferred and d.s2_value == d.final_value for d in ts)
    fake.agent_calls.clear()
    cfg = _cfg(tool_select={"mode": "assist", "k": 2}, s1={"type": "oracle"})
    sim, agent = run_episode(cfg, s1=OracleBackend())
    assert sim.reward_info.reward == 1.0
    assert fake.agent_calls and all(len(c["tools"]) <= 2 for c in fake.agent_calls)
    assert "create_task" in fake.agent_calls[0]["tools"]


def test_respond_fallback_defer_reasks_the_llm_with_its_tools(fake):
    """S1 says respond_to_user; the LLM, called under tool_choice="none", returns only a call to an undeclared tool.
    The registered fallback ("execute") runs that call; "defer" (E1, Deviations 2026-10-05) re-asks the LLM with its
    tools, as a cascade deferral, so no undeclared tool is ever executed."""
    fake.script = "ignore_none"
    cfg = _cfg(tool_select={"mode": "cascade", "tau": 0.5}, s1={"type": "simulated"})
    assert cfg.respond_fallback == "execute"
    sim, agent = run_episode(cfg, s1=RespondBackend())
    first = _decisions(agent, "tool_select")[0]
    assert first.s1_value == jt.RESPOND and first.deferred and agent.steps[0].tool == "get_line"
    assert agent.n_respond_overrides >= 1
    fake.agent_calls.clear()
    cfg = _cfg(tool_select={"mode": "cascade", "tau": 0.5}, s1={"type": "simulated"}, respond_fallback="defer")
    sim, agent = run_episode(cfg, s1=RespondBackend())
    assert sim.reward_info.reward == 1.0
    first = _decisions(agent, "tool_select")[0]
    assert first.s1_value == jt.RESPOND and first.deferred and first.s2_value == first.final_value == "create_task"
    assert agent.steps[0].tool == "create_task" and agent.steps[0].llm_calls == 2
    assert fake.agent_calls[0]["tool_choice"] == "none" and fake.agent_calls[1]["tool_choice"] in (None, "auto")
    assert "create_task" in fake.agent_calls[1]["tools"]
    assert not any(tc.name == "get_line" for tc in _tool_calls(sim))
    assert agent.sidecar_record()["config_dump"]["respond_fallback"] == "defer"
    with pytest.raises(ValueError, match="respond_fallback"):
        _cfg(respond_fallback="maybe")


def test_cascade_always_defer_never_queries_system1(fake):
    """A6.11 (c).4 / A6.20: a tau^2 refit that accepts no level deploys always-defer on that decision point: every
    decision is logged as deferred with no System-1 answer, and System-1 is never called."""
    class Counting(SimulatedBackend):
        n = 0

        def decide(self, *a, **k):
            Counting.n += 1
            return super().decide(*a, **k)

    cfg = _cfg(tool_select={"mode": "cascade", "tau": 0.99, "always_defer": True},
               tool_params={"mode": "cascade", "tau": 0.99, "always_defer": True}, s1={"type": "simulated"})
    sim, agent = run_episode(cfg, s1=Counting(accuracy=0.9, seed=0))
    assert sim.reward_info.reward == 1.0 and Counting.n == 0
    assert not any(r.component == "s1" for r in agent.ledger.records)
    ts = _decisions(agent, "tool_select")
    assert ts and all(d.deferred and d.s1_value is None and d.gold_aux.get("always_defer") for d in ts)
    assert all(d.s2_value == d.final_value for d in ts)
    tp = _decisions(agent, "tool_params")
    assert all(d.deferred and d.s1_value is None for d in tp)


def test_log_params_harvests_the_llm_argument_choices_without_system1(tmp_path, fake):
    """E22 (Deviations 2026-10-08): ``log_params`` logs, at every executed LLM tool call, the argument Choices the
    cascade would ask (options as ``_param_options`` builds them) with the LLM's value; no System-1 call, no change."""
    sim, agent = run_episode(_cfg(name="harvest", log_params=True), log_dir=tmp_path)
    assert sim.reward_info.reward == 1.0 and agent.s1 is None
    assert all(c["tool_choice"] == "auto" for c in fake.agent_calls)
    tp = _decisions(agent, "tool_params")
    assert tp and all(d.mode == "off" and d.s1_value is None and d.state is None for d in tp)
    by_param = {d.gold_aux["param"]: d for d in tp}
    assert by_param["user_id"].s2_value == "user_1" and "user_1" in by_param["user_id"].options
    assert all(d.options[-1] == jt.OTHER for d in tp if d.gold_aux["param"] == "user_id")
    assert all(d.gold_aux["tool"] == "create_task" for d in tp)
    (rec,) = jt.load_sidecar(tmp_path / jt.SIDECAR_FILE)
    assert "s1" not in rec["usage"]
    with pytest.raises(ValueError, match="log_params"):
        _cfg(log_params=True, tool_params={"mode": "cascade"}, s1={"type": "oracle"})


def test_s1_act_complete_acts_only_on_complete_calls(fake):
    """E22: ``s1_act: complete`` emits System-1's tool call only when System-1 decides every required argument (no LLM
    call); respond_to_user and incomplete calls go to the LLM-only turn (full tool list, tool_choice auto)."""
    cfg = _cfg(tool_select={"mode": "cascade", "tau": 0.0}, tool_params={"mode": "cascade", "tau": 0.0},
               s1={"type": "oracle"}, s1_act="complete")
    sim, agent = run_episode(cfg, s1=OracleBackend())
    assert sim.reward_info.reward == 1.0
    first = agent.steps[0]
    assert first.tool == "create_task" and first.args == GOOD and first.llm_calls == 0
    # S1 then answers respond_to_user: not acted on; the LLM gets the request llm_only sends
    ts = _decisions(agent, "tool_select")[1]
    assert ts.s1_value == jt.RESPOND and ts.deferred and ts.gold_aux["not_acted"] == "respond"
    assert fake.agent_calls and all(c["tool_choice"] in (None, "auto") for c in fake.agent_calls)
    assert all("create_task" in c["tools"] for c in fake.agent_calls)
    fake.agent_calls.clear()
    cfg = _cfg(tool_select={"mode": "cascade", "tau": 0.0}, tool_params={"mode": "cascade", "tau": 1.01},
               s1={"type": "oracle"}, s1_act="complete")
    sim, agent = run_episode(cfg, s1=OracleBackend())
    assert sim.reward_info.reward == 1.0
    first = agent.steps[0]
    ts = _decisions(agent, "tool_select")[0]
    assert ts.s1_value == "create_task" and ts.deferred and ts.gold_aux["not_acted"] == "incomplete"
    assert first.llm_calls == 1 and ts.s2_value == ts.final_value == "create_task"
    tp = [d for d in first.decisions if d.dp == "tool_params"]
    assert tp and all(d.deferred and d.final_value is None for d in tp)
    assert fake.agent_calls[0]["tool_choice"] in (None, "auto") and len(fake.agent_calls[0]["tools"]) > 1
    with pytest.raises(ValueError, match="s1_act"):
        _cfg(s1_act="some")


def test_policy_guard_blocks_noncompliant_write(fake):
    fake.script = "wrong_first"
    cfg = _cfg(policy_guard={"mode": "guard", "tau": 0.5}, s1={"type": "oracle"})
    sim, agent = run_episode(cfg, s1=OracleBackend())
    pol = _decisions(agent, "policy")
    assert pol[0].s1_value is False and pol[0].gold is False  # wrong title: not a reference action
    created = [tc for tc in _tool_calls(sim) if tc.name == "create_task"]
    assert created and all(tc.arguments["title"] == "Important Meeting" for tc in created)
    assert sim.reward_info.reward == 1.0
    assert any(c.decision_point == "policy" and c.component == "s2" for c in agent.ledger.records)
    assert any(str(c["last"]).startswith("[Policy check]") for c in fake.agent_calls)


def test_stop_guard_blocks_unjustified_transfer(fake):
    fake.script = "transfer_first"
    cfg = _cfg(stop={"mode": "guard", "tau": 0.5}, s1={"type": "oracle"})
    sim, agent = run_episode(cfg, s1=OracleBackend())
    stop = _decisions(agent, "stop")
    assert stop and stop[0].s1_value is False and stop[0].gold is False
    assert not any(tc.name == "transfer_to_human_agents" for tc in _tool_calls(sim))
    assert sim.reward_info.reward == 1.0


def test_routing_trigger_sends_turns_to_small_llm(fake):
    cfg = _cfg(routing={"mode": "trigger", "tau": 0.5}, small_llm=SMALL_LLM, s1={"type": "const"})
    sim, agent = run_episode(cfg, s1=ConstBackend(p_yes=0.2))
    assert sim.reward_info.reward == 1.0
    assert {c["model"] for c in fake.agent_calls} == {SMALL_LLM}
    assert {d.final_value for d in _decisions(agent, "routing")} == {"small"}
    assert all(s.model == SMALL_LLM for s in agent.steps if s.llm_calls)
    fake.agent_calls.clear()
    run_episode(cfg, s1=ConstBackend(p_yes=0.9))
    assert {c["model"] for c in fake.agent_calls} == {AGENT_LLM}


# ------------------------------------------------------------------ shadow mode (PREREGISTRATION A5.4)
def _executed(agent) -> list[tuple]:
    return [(s.tool, s.args, s.llm_calls, s.model) for s in agent.steps]


def test_shadow_mode_executes_the_llm_action_and_logs_s1(tmp_path, fake):
    sim0, base = run_episode(_cfg(name="llm_only"), log_dir=tmp_path / "base")
    n_llm = len(fake.agent_calls)
    fake.agent_calls.clear()
    sim, agent = run_episode(_cfg(name="jev_shadow", **SHADOW_CFG), s1=PricedOracle(), log_dir=tmp_path / "sh")
    assert sim.reward_info.reward == 1.0 and agent.error is None
    # the executed actions, the LLM calls (full tool list, tool_choice auto) and tau2's agent_cost are llm_only's
    assert _executed(agent) == _executed(base) and len(fake.agent_calls) == n_llm
    assert all(c["tool_choice"] == "auto" and len(c["tools"]) == len(agent.tools) for c in fake.agent_calls)
    assert sim.agent_cost == sim0.agent_cost
    ds = _decisions(agent)
    assert ds and all(d.mode == "shadow" and d.shadow and d.deferred for d in ds)
    assert {d.dp for d in ds} == {"routing", "tool_select", "policy", "tool_params"}  # stop: only on a transfer
    ts = [d for d in ds if d.dp == "tool_select"]
    assert ts[0].question == jt.TOOL_SELECT_Q and jt.RESPOND in ts[0].options and "create_task" in ts[0].options
    assert ts[0].s1_value == ts[0].gold == ts[0].s2_value == ts[0].final_value == "create_task"
    assert ts[0].s2_correct is True and ts[0].gold_status == "ok" and 0.5 < ts[0].s1_conf <= 1.0 and ts[0].s1_dist
    assert ts[0].s1_cost_usd > 0 and ts[0].s1_latency_ms > 0 and ts[0].s2_cost_usd > 0  # the measurement fields
    assert ts[-1].s1_value == ts[-1].gold == ts[-1].s2_value == jt.RESPOND
    tp = [d for d in ds if d.dp == "tool_params"]
    assert {d.final_value for d in tp} == set(GOOD.values())
    assert all(d.s1_value == d.gold == d.s2_value == d.final_value and d.s2_correct and d.gold_status == "ok"
               and d.question.startswith("Which value should the argument") and d.s2_cost_usd > 0 for d in tp)
    pol = [d for d in ds if d.dp == "policy"]
    assert pol and pol[0].gold is True and pol[0].s1_value is True and pol[0].s2_value is pol[0].final_value is True
    rt = [d for d in ds if d.dp == "routing"]
    assert rt and all(d.final_value == "large" and d.gold is None and d.gold_status == "no_oracle" for d in rt)
    assert all(s.model == AGENT_LLM for s in agent.steps if s.llm_calls)
    # the questions are exactly the live arms' questions (cascade at tau=1 defers every turn: same LLM path)
    fake.agent_calls.clear()
    _, casc = run_episode(_cfg(tool_select={"mode": "cascade", "tau": 1.0}, s1={"type": "oracle"}), s1=OracleBackend())
    live = [(d.question, d.options) for d in _decisions(casc, "tool_select")]
    assert live == [(d.question, d.options) for d in ts] and all(d.deferred for d in _decisions(casc, "tool_select"))
    fake.agent_calls.clear()
    _, repl = run_episode(_cfg(tool_select={"mode": "replace"}, tool_params={"mode": "replace"}, s1={"type": "oracle"}),
                          s1=OracleBackend())
    assert {(d.question, tuple(d.options)) for d in _decisions(repl, "tool_params")} == \
        {(d.question, tuple(d.options)) for d in tp}
    # the Ledger: every S1 call is a tagged shadow call, kept out of the episode cost (harness usage_summary)
    (rec,) = jt.load_sidecar(tmp_path / "sh" / jt.SIDECAR_FILE)
    (rec0,) = jt.load_sidecar(tmp_path / "base" / jt.SIDECAR_FILE)
    u = rec["usage"]
    s1_calls = [c for c in rec["calls"] if c["component"] == "s1"]
    assert s1_calls and all(c["meta"]["shadow"] is True and c["meta"]["measure"] == "shadow" for c in s1_calls)
    assert "s1" not in u and u["shadow"]["calls"] == len(s1_calls) == len(ds) and u["shadow"]["cost_usd"] > 0
    assert u["shadow"]["by_measure"] == {"shadow": {k: v for k, v in u["shadow"].items()
                                                    if k not in ("by_decision_point", "by_measure")}}
    assert set(u["shadow"]["by_decision_point"]) == {"routing", "tool_select", "policy", "tool_params"}
    assert u["total"]["cost_usd"] == pytest.approx(u["s2"]["cost_usd"])
    assert u["total_with_shadow"]["cost_usd"] == pytest.approx(u["total"]["cost_usd"] + u["shadow"]["cost_usd"])
    for k in ("calls", "input_tokens", "output_tokens", "cost_usd"):
        assert u["total"][k] == rec0["usage"]["total"][k]
    assert rec["shadow_points"] == ["tool_select", "tool_params", "stop", "routing", "policy"]
    assert rec["shadow_s1_errors"] == [] and rec["backend"] == "oracle"
    jev = [m.raw_data["jev"] for m in sim.messages if m.role == "assistant" and m.raw_data]
    assert all(j["s1_calls"] == 0 and j["s1_cost_usd"] == 0 and j["shadow_calls"] > 0 and j["shadow_cost_usd"] > 0
               and all(d["shadow"] for d in j["decisions"]) for j in jev)


def test_shadow_stop_logs_the_guard_question_but_lets_the_transfer_through(fake):
    fake.script = "transfer_first"
    sim, agent = run_episode(_cfg(stop={"mode": "shadow"}, s1={"type": "oracle"}), s1=OracleBackend())
    (st,) = _decisions(agent, "stop")
    assert st.mode == "shadow" and st.shadow and st.deferred and st.question == jt.STOP_Q
    assert st.gold is False and st.s1_value is False and st.s2_value is st.final_value is True and st.s2_correct is False
    assert any(tc.name == jt.TRANSFER_TOOL for tc in _tool_calls(sim))   # executed (guard mode would block it)
    assert all((c.meta or {}).get("shadow") for c in agent.ledger.records if c.component == "s1")
    fake.script, fake.agent_calls[:] = "transfer_first", []
    _, guard = run_episode(_cfg(stop={"mode": "guard", "tau": 0.5}, s1={"type": "oracle"}), s1=OracleBackend())
    assert _decisions(guard, "stop")[0].question == st.question


def test_shadow_s1_failure_is_fail_open(tmp_path, fake):
    sim0, base = run_episode(_cfg(name="llm_only"))
    fake.agent_calls.clear()
    sim, agent = run_episode(_cfg(name="sh", **SHADOW_CFG), s1=DownBackend(), log_dir=tmp_path)
    assert sim.reward_info.reward == 1.0 and agent.error is None and _executed(agent) == _executed(base)
    ds = _decisions(agent)
    assert ds and all(d.shadow and d.s1_value is None and d.s1_conf is None and d.s1_dist is None for d in ds)
    assert [d for d in ds if d.dp == "tool_select"][0].s2_value == "create_task"   # the LLM's decision is still logged
    (rec,) = jt.load_sidecar(tmp_path / jt.SIDECAR_FILE)
    assert rec["shadow_s1_errors"] and all("RuntimeError: S1 is down" in e for e in rec["shadow_s1_errors"])
    assert any(e.startswith("tool_select:") for e in rec["shadow_s1_errors"]) and "s1" not in rec["usage"]


def test_jev_shadow_arm_end_to_end_feeds_threshold_theory(tmp_path, monkeypatch, fake, clean_env, capsys):
    """The A5.4 refit path: the shipped jev_shadow arm through the launcher, the converter, and
    ``threshold_theory`` reading the shadow records exactly as it reads the JevGym jev_shadow arm."""
    out = tmp_path / "raw"
    jt.main(["run", "--jev-config", str(REPO / "configs" / "tau2" / "jev_shadow.yaml"),
             "--jev-set", "s1={type: simulated, accuracy: 0.7, price_per_m_input: 0.042, seed: 1}",
             "--domain", "mock", "--task-ids", "create_task_1", "--agent-llm", AGENT_LLM, "--user-llm", USER_LLM,
             "--num-trials", "3", "--max-concurrency", "1", "--save-to", str(out), "--auto-resume"])
    recs = jt.load_sidecar(out / jt.SIDECAR_FILE)
    assert len(recs) == 3 and all(r["config"] == "jev_shadow" and r["backend"] == "simulated" for r in recs)
    assert all(r["shadow_points"] == ["tool_select", "tool_params", "stop"] for r in recs)
    assert all(r["usage"]["shadow"]["cost_usd"] > 0 and "s1" not in r["usage"] for r in recs)
    capsys.readouterr()
    jt.main(["convert", str(out / "results.json"), "--arm", "jev_shadow",
             "--out", str(tmp_path / "rep" / "jev_shadow.jsonl")])
    summary = json.loads(capsys.readouterr().out)
    assert summary["n"] == 3 and summary["pass^3"] == 1.0 and summary["llm_calls_per_sim"] == 2
    assert summary["s1_cost_usd_per_sim"] == 0 and summary["shadow_cost_usd_per_sim"] > 0
    assert summary["shadow_s1_errors"] == 0 and "by_split" not in summary   # mock: no A5.4 split
    dicts = load_trajectories(tmp_path / "rep" / "jev_shadow.jsonl")
    assert all(t["meta"]["tau2"]["split"] is None and t["meta"]["shadow_s1_errors"] == [] for t in dicts)
    assert all(t["usage"]["total"]["cost_usd"] == pytest.approx(t["usage"]["s2"]["cost_usd"]) for t in dicts)
    assert episode_table(dicts)["s1_cost_usd"] == 0 and decision_table(decision_records(dicts))["tool_select/shadow"]
    # the records threshold_theory fits tau-hat* from: confidence c, S1 / LLM correctness, per-decision costs
    pairs = TT.decision_pairs(dicts, "tool_select")
    assert len(pairs) == 6 and all(p["shadow"] and p["mode"] == "shadow" and p["c"] is not None
                                   and p["s1_ok"] is not None and p["s2_ok"] is True and p["final_ok"] is True
                                   and p["s1_cost"] > 0 and p["s2_cost"] > 0 and p["acceptable"] for p in pairs)
    tp = TT.decision_pairs(dicts, "tool_params")
    assert tp and all(p["shadow"] and p["c"] is not None and p["s2_ok"] is True for p in tp)
    TT.main([str(tmp_path / "rep"), "--arm", "jev_shadow", "--dp", "tool_select", "--V", "1", "--n-boot", "20",
             "--n-perm", "20", "--format", "json", "--out", str(tmp_path / "tt.json")])
    res = json.loads((tmp_path / "tt.json").read_text())
    th = res["by_V"][0]["thresholds_all"]
    assert res["meta"]["arms"] == ["jev_shadow"] and th["n"] == 6 and set(th) >= {"prop1", "qL", "general"}
    capsys.readouterr()
    jt.main(["report", str(tmp_path / "rep"), "--no-report"])
    assert json.loads(capsys.readouterr().out)["arms"]["jev_shadow"]["all"]["n"] == 3


# ------------------------------------------------------------------ registry + batch runner + converter
def test_run_domain_registry_and_convert_end_to_end(tmp_path, monkeypatch, fake, clean_env):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_domain

    cfg_path = tmp_path / "arm.yaml"
    cfg_path.write_text("name: jev_e2e\ntool_select: {mode: cascade, tau: 0.5}\n"
                        "tool_params: {mode: replace}\npolicy_guard: {mode: guard, tau: 0.5}\n"
                        "s1: {type: oracle}\n")
    monkeypatch.setenv(jt.ENV_CONFIG, str(cfg_path))
    monkeypatch.setenv(jt.ENV_LOG_DIR, str(tmp_path / "run"))
    assert jt.register() == jt.AGENT_NAME and jt.register() == jt.AGENT_NAME  # idempotent
    common = dict(domain="mock", llm_agent=AGENT_LLM, llm_user=USER_LLM, task_ids=["create_task_1"],
                  num_trials=2, max_concurrency=1, auto_resume=True)
    run_domain(TextRunConfig(agent=jt.AGENT_NAME, save_to=str(tmp_path / "run"), **common))
    trajs = jt.convert_results(tmp_path / "run" / "results.json", out=tmp_path / "report" / "jev_e2e.jsonl")
    assert len(trajs) == 2 and all(t.success and t.meta["tau2"]["sidecar"] for t in trajs)
    assert {t.task_id for t in trajs} == {"mock/create_task_1#0", "mock/create_task_1#1"}
    assert all(t.config == "jev_e2e" and t.backend == "oracle" and t.steps for t in trajs)
    assert jt.pass_hat_k(trajs) == {"pass^1": 1.0, "pass^2": 1.0}
    dicts = [t.to_dict() for t in trajs]
    assert episode_table(dicts)["success"] == 1.0
    table = decision_table(decision_records(dicts))
    assert table["tool_select/cascade"]["s1_acc"] == 1.0 and "policy/guard" in table
    md = report(tmp_path / "report")
    assert "jev_e2e" in md and "tool_select/cascade" in md
    from jevharness.training.data import harvest_trajectories  # S1 training data from tau2 runs

    ex = harvest_trajectories([tmp_path / "report" / "jev_e2e.jsonl"], label_source="gold")
    assert ex and {e["dp"] for e in ex} >= {"tool_select", "tool_params"}
    assert all(e["state"].startswith("DOMAIN POLICY") for e in ex)

    # tau2's stock llm_agent (no side-car) converts from the messages alone
    monkeypatch.setenv(jt.ENV_LOG_DIR, "")
    run_domain(TextRunConfig(agent="llm_agent", save_to=str(tmp_path / "stock"), **common))
    stock = jt.convert_results(tmp_path / "stock", arm="stock_llm_agent",
                               out=tmp_path / "report" / "stock.jsonl")
    assert len(stock) == 2 and all(t.success and not t.meta["tau2"]["sidecar"] for t in stock)
    assert all(t.usage["s2"]["calls"] >= 2 and t.steps for t in stock)
    assert all(t.config == "stock_llm_agent" for t in stock)
    assert "| stock" in report(tmp_path / "report")  # report rows are keyed by file stem


def test_cli_launcher_run_and_convert(tmp_path, monkeypatch, fake, clean_env, capsys):
    cfg = REPO / "configs" / "tau2" / "jev_ts_cascade.yaml"
    out = tmp_path / "cli"
    args = ["run", "--jev-config", str(cfg), "--jev-set", "tool_select.tau=0.9",
            "--jev-set", "s1={type: oracle}",
            "--domain", "mock", "--task-ids", "create_task_1", "--agent-llm", AGENT_LLM,
            "--user-llm", USER_LLM, "--num-trials", "1", "--max-concurrency", "1", "--save-to", str(out),
            "--auto-resume"]
    jt.main(args)
    (rec,) = jt.load_sidecar(out / jt.SIDECAR_FILE)
    assert rec["config"] == "jev_ts_cascade_tau0.9"  # dict-valued overrides do not rename the arm
    assert rec["config_dump"]["harness"]["tool_select"]["tau"] == 0.9 and rec["backend"] == "oracle"
    capsys.readouterr()
    jt.main(["convert", str(out / "results.json"), "--arm", "cascade_tau0.9",
             "--out", str(tmp_path / "c.jsonl")])
    summary = json.loads(capsys.readouterr().out)
    assert summary["n"] == 1 and summary["pass^1"] == 1.0 and summary["with_sidecar"] == 1
    with pytest.raises(SystemExit):
        jt.main(["run", "--jev-config", str(cfg), "--domain", "mock", "--workers", "2"])
