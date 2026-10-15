"""Offline tests for the non-learned baselines / controls (docs/BASELINES.md)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from jevharness.bench.metrics import decision_records, decision_table
from jevharness.bench.suites.jevgym import JevGymEnv
from jevharness.decision import Choice, Noul, Score, make_backend
from jevharness.decision.base import CachedBackend, DecisionBackend
from jevharness.decision.controls import (HeuristicBackend, PriorBackend, RateBackend, fit_prior,
                                          fit_prior_from_trajectories, infer_dp, parse_state, trigger_rates)
from jevharness.decision.lexical import BM25Backend, TfidfBackend, bm25_scores, tfidf_cosine, tokenize
from jevharness.decision.self_llm import SelfLLMBackend, make_self_backend
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.llm.base import ChatResponse
from jevharness.llm.scripted import ScriptedLLM
from jevharness.llm.simulated import SimulatedLLM

TOOLS = ["get_weather", "cancelOrder", "find_user", "list-warehouses", "finish"]
DESCS = ["Current weather for a city.", "Cancel an order.", "Look up a customer by full name.",
         "List warehouse locations.", "Call when the task is complete."]


def _tool_q(**meta):
    return Choice("Which tool should the agent call next?", TOOLS, DESCS, decision_point="tool_select", **meta)


def _state(task, history=(), obs="", memory=""):
    """Render a state exactly as the harness does (keeps the parser tests in sync with agent.py)."""
    agent = JevAgent(SimulatedLLM(), HarnessConfig(name="llm_only"))
    return agent._s1_state(SimpleNamespace(instruction=task), list(history), obs, memory)


# ------------------------------------------------------------------ lexical
def test_tokenize_handles_tool_name_styles():
    assert tokenize("getUserOrders") == tokenize("get_user_orders") == tokenize("get-user.orders") \
        == ["get", "user", "order"]
    assert tokenize("HTTPServer") == ["http", "server"]
    assert tokenize("cancelled shipping ORD-1234") == ["cancel", "ship", "ord", "1234"]
    assert tokenize("the of and") == []


def test_bm25_and_tfidf_scoring_functions():
    docs = [["cancel", "order"], ["weather", "city"], ["order", "list", "customer"]]
    s = bm25_scores(["cancel", "order"], docs)
    assert s[0] > s[2] > s[1] == 0.0
    assert (bm25_scores(["order"], docs) >= 0).all()  # non-negative IDF even for frequent terms
    c = tfidf_cosine(["cancel", "order"], docs)
    assert np.argmax(c) == 0 and c[1] == 0.0 and 0 < c[0] <= 1.0 + 1e-9


@pytest.mark.parametrize("cls", [BM25Backend, TfidfBackend])
def test_lexical_ranks_matching_tool_first(cls):
    b = cls()
    st = _state("Please cancel the pending order ORD-1 for Dana Kim.", obs="You have access to the tools.")
    a = b.ask(st, _tool_q())
    assert a.value == "cancelOrder" and a.top_k(1) == ["cancelOrder"]
    assert abs(sum(a.distribution.values()) - 1) < 1e-6
    # camelCase / kebab-case names are matched through tokenisation, descriptions matter
    a2 = b.ask("show me the warehouse list", _tool_q())
    assert a2.value == "list-warehouses"
    a3 = b.ask("what is the weather in Paris", _tool_q())
    assert a3.value == "get_weather"


def test_lexical_temperature_query_mode_and_abstain():
    q = Choice("Which function?", ["get_weather", "none (no available function fits the request)"],
               ["Current weather for a city.", "Use when no function applies."])
    sharp, flat = BM25Backend(temperature=0.1), BM25Backend(temperature=10.0)
    assert sharp.ask("weather in Rome", q).confidence > flat.ask("weather in Rome", q).confidence
    assert BM25Backend(abstain_score=5.0).ask("translate this", q).value == q.options[1]
    # task_obs ignores history text
    st = _state("Tell me the weather.", history=[("cancelOrder({})", "cancel cancel order")], obs="ok")
    full = BM25Backend().ask(st, _tool_q())
    task = BM25Backend(query_mode="task_obs").ask(st, _tool_q())
    assert full.value == "cancelOrder" and task.value == "get_weather"


def test_lexical_noul_needs_fallback():
    with pytest.raises(NotImplementedError):
        BM25Backend().decide("s", [Noul("ok?")])
    b = BM25Backend(fallback={"type": "rate", "rate": 1.0})
    r = b.decide("please cancel", [_tool_q(), Noul("Is the agent stuck?")])
    assert r.answers[0].value == "cancelOrder" and r.answers[1].value is True
    assert r.usage.cost_usd == 0.0 and r.usage.n_calls == 2


# ------------------------------------------------------------------ rate control
@pytest.mark.parametrize("r", [0.1, 0.3, 0.5])
def test_rate_backend_fires_at_r(r):
    b = RateBackend(rate=r, seed=1)
    q = Noul("Is the agent stuck, repeating itself, or did the last action fail?")
    fires = [b.ask(f"state {i}", q).value for i in range(4000)]
    assert abs(np.mean(fires) - r) < 0.03
    # deterministic (hash-based): same state -> same answer; seed changes draws; tau-independent p_yes
    assert b.ask("state 7", q).p_yes == b.ask("state 7", q).p_yes
    assert b.ask("state 7", q).p_yes in (pytest.approx(0.999), pytest.approx(0.001))
    other = RateBackend(rate=r, seed=2)
    assert any(other.ask(f"state {i}", q).value != fires[i] for i in range(200))


def test_rate_backend_per_dp_uniform_score_and_random_choice():
    b = RateBackend(rate=0.0, rates={"reflection": 1.0})
    assert b.ask("s", Noul("Is the agent stuck?")).value is True           # dp inferred from the text
    assert b.ask("s", Noul("Has the task already been fully completed, so the agent should stop now?")).value is False
    u = RateBackend(rate=None)
    ps = np.array([u.ask(f"s{i}", Noul("q")).p_yes for i in range(4000)])
    assert abs((ps >= 0.7).mean() - 0.3) < 0.03                            # rate = 1 - tau
    picks = [RateBackend(seed=0).ask(f"s{i}", _tool_q()) for i in range(2000)]
    counts = np.bincount([TOOLS.index(a.value) for a in picks], minlength=len(TOOLS))
    assert counts.min() > 300 and all(a.confidence < 0.21 for a in picks)  # uniform, low confidence
    assert len({tuple(a.top_k(3)) for a in picks[:50]}) > 30              # random top-k subsets
    sc = RateBackend().ask("s", Score("How good?", scale=(1, 5)))
    assert 1 <= sc.value <= 5


# ------------------------------------------------------------------ prior control
def test_prior_backend_rules_and_fitted_prior(tmp_path):
    q = _tool_q()
    assert PriorBackend(rule="first").ask("s", q).value == TOOLS[0]
    assert PriorBackend(rule="last").ask("s", q).value == "finish"
    assert PriorBackend(rule="last").ask("s", q).top_k(2) == ["finish", "list-warehouses"]
    maj = PriorBackend(prior={"find_user": 5, "finish": 3})
    assert maj.ask("s", q).value == "find_user" and maj.ask("s", q).top_k(2) == ["find_user", "finish"]
    assert PriorBackend().ask("s", q).value == TOOLS[0]                    # no prior -> fallback rule
    per_dp = {"tool_select": {"finish": 1.0}, "reflection": {"yes": 1, "no": 3}}
    f = tmp_path / "prior.json"
    f.write_text(json.dumps(per_dp))
    pb = PriorBackend(prior_file=str(f))
    assert pb.ask("s", q).value == "finish"
    assert pb.ask("s", Noul("stuck?", decision_point="reflection")).p_yes == pytest.approx(0.25)
    assert PriorBackend(p_yes={"stop": 0.9}).ask("s", Noul("x", decision_point="stop")).value is True
    # "recent": the most recently *useful* tool
    st = _state("t", history=[("find_user({})", '{"user_id": "U1"}'), ("cancelOrder({})", "Error: not found")],
                obs="Error: not found")
    assert PriorBackend(rule="recent").ask(st, q).top_k(2) == ["find_user", "cancelOrder"]


def test_fit_prior_helpers_and_trigger_rates():
    assert fit_prior(["a", "b", "a", True, None]) == {"a": 0.5, "b": 0.25, "yes": 0.25}
    trajs = [{"steps": [{"decisions": [
        {"dp": "reflection", "mode": "trigger", "s1_value": True, "gold": None},
        {"dp": "tool_select", "mode": "replace", "s1_value": "x", "gold": "x"}]},
        {"decisions": [{"dp": "reflection", "mode": "trigger", "s1_value": False, "gold": None},
                       {"dp": "stop", "mode": "guard", "s1_value": True, "gold": True}]}],
        "meta": {"memory": {"offered": 4, "stored": 1}, "config": {"memory_write": {"mode": "trigger"}}}}]
    rates = trigger_rates(trajs)
    assert rates["reflection/trigger"] == {"n": 2, "rate": 0.5}
    assert rates["stop/guard"]["rate"] == 1.0 and rates["memory_write/trigger"]["rate"] == 0.25
    assert fit_prior_from_trajectories(trajs) == {"stop": {"yes": 1.0}, "tool_select": {"x": 1.0}}


# ------------------------------------------------------------------ heuristic control
def _noul(dp, text):
    return Noul(text, decision_point=dp)


REFLECT = "Is the agent stuck, repeating itself, or did the last action fail or move away from the goal?"
STOP_T = "Has the task already been fully completed, so the agent should stop now?"
STOP_G = "The agent wants to stop and submit. Is the task actually complete?"
ROUTE = ("Does the next step require careful multi-step reasoning, planning, or recovery from an error "
         "(i.e., a strong model) rather than a routine action?")
MEMW = "Will this information likely be needed later to complete the task (facts, IDs, constraints, results)?"


def test_parse_state_roundtrip():
    st = parse_state(_state("Cancel it.", [("find_user({\"name\": \"A B\"})", '{"user_id": "U1"}\nlog: ok')],
                            '{"user_id": "U1"}', memory="- [m0] x"))
    assert st.structured and st.task == "Cancel it." and st.memory == "- [m0] x"
    assert st.history == [('find_user({"name": "A B"})', '{"user_id": "U1"}\nlog: ok')]
    assert st.observation == '{"user_id": "U1"}'
    assert not parse_state("free text state").structured


def test_heuristic_meta_rules_on_crafted_states():
    h = HeuristicBackend()
    task = "The customer Dana Kim asked us to cancel their pending order. Please do it."
    ok = _state(task, [("find_user({\"name\": \"Dana Kim\"})", '{"user_id": "U1", "tier": "gold"}')],
                '{"user_id": "U1", "tier": "gold"}')
    err = _state(task, [("get_orders({\"user_id\": \"U9\"})", "Error: order not found")], "Error: order not found")
    rep = _state(task, [("get_orders({})", "[{}]"), ("find_user({})", "{}x"), ("get_orders({})", "[{}]")], "[{}]")
    cancelled = _state(task, [("get_orders({})", '[{"order_id": "ORD-1"}]'), ("cancel_order({\"order_id\": \"ORD-1\"})",
                              "Order ORD-1 cancelled.")], "Order ORD-1 cancelled.")
    for dp, text in (("reflection", REFLECT), ("routing", ROUTE)):
        assert h.ask(ok, _noul(dp, text)).value is False
        assert h.ask(err, _noul(dp, text)).value is True
        assert h.ask(rep, _noul(dp, text)).value is True                   # repeated action
    assert h.ask(cancelled, _noul("stop", STOP_T)).value is True           # state change succeeded
    assert h.ask(ok, _noul("stop", STOP_T)).value is False
    assert h.ask(err, _noul("stop", STOP_T)).value is False
    assert h.ask(ok, _noul("stop", STOP_G)).value is False                 # action task, nothing changed yet
    assert h.ask(cancelled, _noul("stop", STOP_G)).value is True
    info = _state("How many units of 'laptop' do we have in stock?", [("check_stock({})", "laptop: 42 units")],
                  "laptop: 42 units")
    assert h.ask(info, _noul("stop", STOP_G)).value is True               # information task answered
    # memory write gate: keep observations carrying IDs / numbers, drop chatter and errors
    mem = lambda text: f"{ok}\n\nCANDIDATE MEMORY:\n{text}"  # noqa: E731
    assert h.ask(mem('get_orders({}) -> [{"order_id": "ORD-1234", "amount": 12.5}]'), _noul("memory_write", MEMW)).value
    assert not h.ask(mem("send_message({}) -> Message sent."), _noul("memory_write", MEMW)).value
    assert not h.ask(mem("get_orders({}) -> Error: user U1 not found"), _noul("memory_write", MEMW)).value
    # tool_select guard: reject an exact repeat of an earlier call
    g = Noul('The agent proposes to call `find_user` with arguments {"name": "Dana Kim"}. Is this the right next '
             "step toward completing the task?", decision_point="tool_select")
    assert h.ask(ok, g).value is False
    g2 = Noul('The agent proposes to call `get_orders` with arguments {"user_id": "U1"}. Is this the right next step '
              "toward completing the task?", decision_point="tool_select")
    assert h.ask(ok, g2).value is True
    # confidence makes trigger outcomes tau-independent
    assert h.ask(err, _noul("reflection", REFLECT)).p_yes == pytest.approx(0.99)


def test_heuristic_choice_rules():
    h = HeuristicBackend()
    task = "The customer Dana Kim asked us to cancel their pending order."
    # tool_select: look before you leap (state-changing tools wait for a successful read-only call), finish
    # only after the task looks complete, never re-call a tool that already succeeded, retry failed calls
    q = _tool_q()
    s0 = _state(task, obs="You have access to the customer-operations tools.")
    assert h.ask(s0, q).value == "find_user" and h.ask(s0, q).top_k(len(TOOLS))[-1] == "finish"
    assert h.ask(s0, q).top_k(len(TOOLS))[-2] == "cancelOrder"
    looked = _state(task, [("find_user({})", '{"user_id": "U1"}')], '{"user_id": "U1"}')
    assert h.ask(looked, q).value == "cancelOrder"
    ok_obs = "Order ORD-1 cancelled."
    done = _state(task, [("find_user({})", '{"user_id": "U1"}'), ("cancelOrder({\"order_id\": \"ORD-1\"})", ok_obs)],
                  ok_obs)
    assert h.ask(done, q).value == "finish"
    failed = _state(task, [("find_user({})", '{"user_id": "U1"}'), ("cancelOrder({})", "Error: not pending")],
                    "Error: not pending")
    assert h.ask(failed, q).value == "cancelOrder"
    # dead ends (empty results, or two errors) go last even when they match the task best
    empty = _state("show me the warehouse list", [("list-warehouses({})", "list-warehouses: no relevant result.")],
                   "list-warehouses: no relevant result.")
    assert h.ask(empty, q).value != "list-warehouses" and "list-warehouses" in h.ask(empty, q).top_k(5)[-2:]
    twice = _state(task, [("find_user({})", '{"user_id": "U1"}'), ("cancelOrder({})", "Error: x"),
                          ("cancelOrder({})", "Error: x")], "Error: x")
    assert h.ask(twice, q).value != "cancelOrder"
    # tool_params: key:value copy, typed spans, OTHER when nothing fits
    obs = '[{"order_id": "ORD-1234", "user_id": "U5551", "amount": 99.5}]'
    st = _state(task, [("get_orders({\"user_id\": \"U5551\"})", obs)], obs)
    cands = ["99.5", "U5551", "ORD-1234", "Dana Kim", "Kim", "<OTHER: none of these / needs generation>"]
    pq = Choice("Which value should the argument `order_id` of tool `cancel_order` take? ()", cands,
                decision_point="tool_params")
    assert h.ask(st, pq).value == "ORD-1234"
    nq = Choice("Which value should the argument `name` of tool `find_user` take? (customer full name)",
                ["Kim", "Dana Kim", "12", "<OTHER: none of these / needs generation>"], decision_point="tool_params")
    assert h.ask(_state(task, obs="ok"), nq).value == "Dana Kim"
    fq = Choice("Which value should the argument `email` of tool `notify` take? ()",
                ["Kim", "12", "<OTHER: none of these / needs generation>"], decision_point="tool_params")
    assert h.ask(s0, fq).value.startswith("<OTHER")
    eq = Choice("Which value should the argument `method` of tool `refund` take? ()",
                ["original_payment", "store_credit"], decision_point="tool_params")
    assert h.ask(_state("Refund it as store credit.", obs="ok"), eq).value == "store_credit"
    # memory_read: recency, "none" last
    mq = Choice("Which stored memory is most relevant to the current step?",
                ["[m0] a", "[m7] b", "[m3] c", "(none of these memories is relevant)"], decision_point="memory_read")
    assert h.ask(s0, mq).top_k(4) == ["[m7] b", "[m3] c", "[m0] a", "(none of these memories is relevant)"]


def test_dp_inferred_for_wire_questions_and_no_gold_leak():
    assert infer_dp(Noul(REFLECT)) == "reflection" and infer_dp(Noul(STOP_T)) == "stop"
    assert infer_dp(Noul(ROUTE)) == "routing" and infer_dp(Noul(MEMW)) == "memory_write"
    assert infer_dp(Choice("Which stored memory is most relevant to the current step?", ["a", "b"])) == "memory_read"
    assert infer_dp(Choice("Which tool should the agent call next?", ["a", "b"])) == "tool_select"
    err = _state("Cancel the order.", [("x({})", "Error: failed")], "Error: failed")
    assert HeuristicBackend().ask(err, Noul(REFLECT)).value is True            # no decision_point attached
    # gold in Question.meta must never change an answer
    for spec in ({"type": "bm25"}, {"type": "tfidf"}, {"type": "rate"}, {"type": "prior", "rule": "recent"},
                 {"type": "heuristic"}):
        b = make_backend(spec)
        a1, a2 = b.ask(err, _tool_q(gold="get_weather")), b.ask(err, _tool_q(gold="finish"))
        assert a1.distribution == a2.distribution, spec
        if spec["type"] not in ("bm25", "tfidf"):
            n1 = b.ask(err, Noul(REFLECT, gold=True))
            n2 = b.ask(err, Noul(REFLECT, gold=False))
            assert n1.distribution == n2.distribution, spec


# ------------------------------------------------------------------ registration + s2_self
def test_all_baselines_register_via_make_backend(tmp_path):
    for kind, cls in (("bm25", BM25Backend), ("tfidf", TfidfBackend), ("rate", RateBackend),
                      ("prior", PriorBackend), ("heuristic", HeuristicBackend)):
        b = make_backend({"type": kind})
        assert isinstance(b, cls) and isinstance(b, DecisionBackend) and b.describe()["backend"] == kind
        cached = make_backend(kind, cache=str(tmp_path / "c.sqlite"))
        assert isinstance(cached, CachedBackend)
        r1 = cached.decide("please cancel the order", [_tool_q()])
        r2 = cached.decide("please cancel the order", [_tool_q()])
        assert r1.answers[0].value == r2.answers[0].value and r2.usage.cached
        assert r1.usage.cost_usd == 0.0
    s2 = make_backend({"type": "s2_self", "llm": {"provider": "scripted"}})
    assert isinstance(s2, SelfLLMBackend) and s2.describe()["backend"] == "s2_self"


def test_s2_self_backend_answers_typed_questions():
    reply = '{"Q1": {"A": 0.1, "B": 0.8, "C": 0.1, "D": 0.0, "E": 0.0}, "Q2": {"yes": 0.7, "no": 0.3}}'
    llm = ScriptedLLM("big", policy=lambda m, t: ChatResponse(text=reply))
    b = make_self_backend(llm, reasoning_effort="minimal")
    r = b.decide("s", [_tool_q(), Noul("stuck?")])
    assert r.answers[0].value == "cancelOrder" and r.answers[1].p_yes == pytest.approx(0.7)
    assert r.usage.backend == "s2_self" and r.usage.n_calls == 1


# ------------------------------------------------------------------ in-harness: every backend x mode
BACKENDS = {
    "bm25": {"type": "bm25", "fallback": {"type": "heuristic"}},
    "tfidf": {"type": "tfidf", "fallback": {"type": "rate", "rate": 0.3}},
    "rate": {"type": "rate", "rate": 0.3},
    "prior": {"type": "prior", "rule": "recent", "p_yes": 0.5},
    "heuristic": {"type": "heuristic"},
}
MODES = {
    "replace": dict(tool_select=DPConfig(mode=Mode.REPLACE), tool_params=DPConfig(mode=Mode.REPLACE)),
    "cascade": dict(tool_select=DPConfig(mode=Mode.CASCADE, tau=0.5), tool_params=DPConfig(mode=Mode.CASCADE, tau=0.5)),
    "assist": dict(tool_select=DPConfig(mode=Mode.ASSIST, k=3), memory_write=DPConfig(mode=Mode.ALWAYS),
                   memory_read=DPConfig(mode=Mode.ASSIST, k=2)),
    "trigger": dict(reflection=DPConfig(mode=Mode.TRIGGER, tau=0.5), stop=DPConfig(mode=Mode.TRIGGER, tau=0.9),
                    routing=DPConfig(mode=Mode.TRIGGER, tau=0.5), memory_write=DPConfig(mode=Mode.TRIGGER, tau=0.5),
                    memory_read=DPConfig(mode=Mode.ASSIST, k=2)),
    "guard": dict(tool_select=DPConfig(mode=Mode.GUARD, tau=0.5), stop=DPConfig(mode=Mode.GUARD, tau=0.5)),
}


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("backend", list(BACKENDS))
def test_jevgym_episode_runs_with_each_backend_and_mode(backend, mode):
    s1 = make_backend(BACKENDS[backend])
    cfg = HarnessConfig(name=f"{backend}_{mode}", max_steps=12, **MODES[mode])
    llm = SimulatedLLM("scripted", error_rate=0.1, simulated_latency_ms=100)
    small = SimulatedLLM("scripted-small", error_rate=0.3, simulated_latency_ms=20)
    agent = JevAgent(llm, cfg, s1=s1, small_llm=small)
    trajs = [agent.run(JevGymEnv(seed=i, n_distractors=10, noise_lines=2)).to_dict() for i in range(5)]
    assert all(t["error"] is None for t in trajs), [t["error"] for t in trajs if t["error"]]
    assert all(t["usage"]["s1"]["calls"] > 0 and t["usage"]["s1"]["cost_usd"] == 0.0 for t in trajs)
    dps = {d["dp"] for t in trajs for s in t["steps"] for d in s["decisions"]}
    if mode in ("replace", "cascade", "assist", "guard"):
        assert "tool_select" in dps
    if mode == "trigger":
        assert {"reflection", "routing"} <= dps
    if mode == "replace" and backend == "heuristic":
        assert decision_table(decision_records(trajs))["tool_select/replace"]["n"] > 0


def test_rate_control_matches_target_rate_in_harness():
    cfg = HarnessConfig(name="rate_reflect", reflection=DPConfig(mode=Mode.TRIGGER, tau=0.5),
                        routing=DPConfig(mode=Mode.TRIGGER, tau=0.5))
    s1 = RateBackend(rate=0.3, rates={"routing": 0.6})
    agent = JevAgent(SimulatedLLM("scripted", error_rate=0.2, simulated_latency_ms=1), cfg, s1=s1,
                     small_llm=SimulatedLLM("scripted-small", error_rate=0.2, simulated_latency_ms=1))
    trajs = [agent.run(JevGymEnv(seed=i)).to_dict() for i in range(60)]
    rates = trigger_rates(trajs)
    assert rates["reflection/trigger"]["n"] > 80 and abs(rates["reflection/trigger"]["rate"] - 0.3) < 0.12
    assert abs(rates["routing/trigger"]["rate"] - 0.6) < 0.12
