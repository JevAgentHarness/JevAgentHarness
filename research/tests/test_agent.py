from jevharness.bench.metrics import decision_records, decision_table, episode_table
from jevharness.bench.suites.jevgym import JevGymEnv
from jevharness.decision.simulated import OracleBackend, SimulatedBackend
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.llm.simulated import SimulatedLLM


def _run(cfg, s1=None, llm_err=0.0, n=15, small=None, **env_kw):
    llm = SimulatedLLM("scripted", error_rate=llm_err, simulated_latency_ms=1000)
    agent = JevAgent(llm, cfg, s1=s1, small_llm=small)
    return [agent.run(JevGymEnv(seed=i, **env_kw)).to_dict() for i in range(n)]


def test_oracle_env_solvable_by_perfect_llm():
    trajs = _run(HarnessConfig(name="llm_only"))
    assert all(t["result"]["success"] for t in trajs), [t["error"] for t in trajs]
    ep = episode_table(trajs)
    assert ep["success"] == 1.0 and ep["llm_calls"] > 1 and ep["s1_calls"] == 0


def test_replace_mode_with_oracle_s1_needs_no_llm_for_bounded_args():
    cfg = HarnessConfig(name="s1_replace", tool_select=DPConfig(mode=Mode.REPLACE),
                        tool_params=DPConfig(mode=Mode.REPLACE))
    trajs = _run(cfg, s1=OracleBackend())
    ep = episode_table(trajs)
    assert ep["success"] == 1.0, [t["error"] for t in trajs if t["error"]]
    # free-text policy (docs/META_GOLD_SPEC.md §7): System-1 never fills `finish.answer`, so the only LLM call of an
    # episode is the forced System-2 fill of the final answer (v1 families: every other argument is bounded / copy)
    assert ep["llm_calls"] == 1.0
    table = decision_table(decision_records(trajs))
    assert table["tool_select/replace"]["s1_acc"] == 1.0


def test_cascade_defers_when_uncertain():
    cfg = HarnessConfig(name="cascade", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.95))
    trajs = _run(cfg, s1=SimulatedBackend(accuracy=0.7, seed=3), llm_err=0.0)
    table = decision_table(decision_records(trajs))
    row = table["tool_select/cascade"]
    assert 0.0 < row["defer_rate"] <= 1.0
    assert episode_table(trajs)["success"] > 0.5


def test_assist_guard_stop_memory_reflection_routing_modes_run():
    small = SimulatedLLM("scripted-small", error_rate=0.3, simulated_latency_ms=200)
    cfg = HarnessConfig(name="kitchen_sink", tool_select=DPConfig(mode=Mode.ASSIST, k=3),
                        memory_write=DPConfig(mode=Mode.TRIGGER, tau=0.5),
                        memory_read=DPConfig(mode=Mode.ASSIST, k=2),
                        reflection=DPConfig(mode=Mode.TRIGGER, tau=0.9),
                        stop=DPConfig(mode=Mode.GUARD, tau=0.5),
                        routing=DPConfig(mode=Mode.TRIGGER, tau=0.5))
    trajs = _run(cfg, s1=SimulatedBackend(accuracy=0.85, seed=0), llm_err=0.05, n=10, small=small,
                 n_distractors=60)
    assert all(t["error"] is None for t in trajs), [t["error"] for t in trajs]
    dps = {d["dp"] for t in trajs for s in t["steps"] for d in s["decisions"]}
    assert {"tool_select", "routing", "stop"} <= dps
    guard = HarnessConfig(name="guard", tool_select=DPConfig(mode=Mode.GUARD, tau=0.5))
    trajs = _run(guard, s1=SimulatedBackend(accuracy=0.9, seed=0), llm_err=0.2, n=10)
    assert all(t["error"] is None for t in trajs)


def test_large_toolset_tournament():
    cfg = HarnessConfig(name="replace_big", tool_select=DPConfig(mode=Mode.REPLACE),
                        tool_params=DPConfig(mode=Mode.REPLACE))
    trajs = _run(cfg, s1=OracleBackend(), n=5, n_distractors=120)  # > 52 options -> tournament
    assert episode_table(trajs)["success"] == 1.0


def test_strict_mode_makes_wrong_state_changes_irreversible():
    env = JevGymEnv(seed=0, family="cancel", strict=True)
    env.reset()
    r = env.step("cancel_order", {"order_id": "ORD-0", "reason": "fraud"})
    assert r.done and env.evaluate(None)["success"] is False
    trajs = _run(HarnessConfig(name="llm_only"), n=10, strict=True)
    assert all(t["result"]["success"] for t in trajs)  # a perfect LLM is unaffected


def test_memory_ids_unique_after_eviction_and_empty_recall():
    from jevharness.decision.types import answer_from_distribution
    from jevharness.harness.memory import MemoryStore

    class NoneFirst(SimulatedBackend):
        def _decide(self, state, questions):
            res = super()._decide(state, questions)
            q = questions[0]
            res.answers[0] = answer_from_distribution(q, {o: (0.9 if o.startswith("(none") else 0.1 / len(q.options))
                                                          for o in q.options})
            return res

    ms = MemoryStore(DPConfig(mode=Mode.ALWAYS), DPConfig(mode=Mode.ASSIST, k=2), s1=NoneFirst(), max_items=3)
    for i in range(6):
        ms.offer(f"note {i}", "ctx")
    assert [m.id for m in ms.items] == [3, 4, 5]
    assert ms.recall("query") == []  # S1 says none is relevant


def test_cascade_tool_params_deferred_arguments_log_the_forced_call_value():
    """A bounded argument System-1 defers (cascade below tau) is filled by System-2's forced call; the decision log
    must carry that value as ``s2_value`` / ``final_value`` (n=50 pilot: 79 deferred tool_params logs had
    ``final_value: null`` although the call carried the argument, so cascade final accuracy was undercounted)."""
    # tool_select=replace so System-1 owns every call (the delegated-call path); its arguments cascade at tau=0.95
    cfg = HarnessConfig(name="cascade_params", tool_select=DPConfig(mode=Mode.REPLACE),
                        tool_params=DPConfig(mode=Mode.CASCADE, tau=0.95))
    trajs = _run(cfg, s1=SimulatedBackend(accuracy=0.6, seed=5), n=12)
    deferred = [d for t in trajs for s in t["steps"] for d in s["decisions"]
                if d["dp"] == "tool_params" and d["mode"] == "cascade" and d["deferred"]]
    assert deferred, "the simulated backend at 0.6 accuracy should defer some arguments below tau=0.95"
    filled = [d for d in deferred if d["final_value"] is not None]
    assert filled and all(d["s2_value"] == d["final_value"] for d in filled)
    # the scripted LLM answers from the oracle, so the forced call's value is the gold one
    assert all(d["final_value"] == d["gold"] for d in filled if d["gold"] is not None)
    # arguments System-1 did fill keep their own value as the final one
    kept = [d for t in trajs for s in t["steps"] for d in s["decisions"]
            if d["dp"] == "tool_params" and d["mode"] == "cascade" and not d["deferred"]]
    assert kept and all(d["final_value"] == d["s1_value"] for d in kept)


def test_cascade_always_defer_never_queries_system1_and_logs_deferred_decisions():
    """PREREGISTRATION A6.11 (c): a cascade whose plug-in policy accepts no confidence level ("never") runs with
    `always_defer: true`: System-1 is not queried at that decision point, every tool choice is the LLM's, and the
    decision log keeps a deferred cascade record per step with no System-1 answer."""
    cfg = HarnessConfig(name="always_defer", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.99,
                                                                   extra={"always_defer": True}),
                        tool_params=DPConfig(mode=Mode.CASCADE, tau=0.13))
    llm = SimulatedLLM("scripted", error_rate=0.0, simulated_latency_ms=1)
    agent = JevAgent(llm, cfg, s1=OracleBackend())
    trajs = [agent.run(JevGymEnv(seed=i)).to_dict() for i in range(4)]
    for t in trajs:
        assert not [c for c in (t.get("calls") or []) if c.get("component") == "s1"], "System-1 must not be called"
    ep = episode_table(trajs)
    assert ep["success"] == 1.0 and ep["s1_calls"] == 0 and ep["s1_cost_usd"] == 0
    ds = [d for t in trajs for s in t["steps"] for d in s["decisions"] if d["dp"] == "tool_select"]
    assert ds and all(d["mode"] == "cascade" and d["deferred"] and d["s1_value"] is None for d in ds)
    assert all(d["final_value"] == d["s2_value"] and d["gold_aux"].get("always_defer") for d in ds)
    assert not [d for t in trajs for s in t["steps"] for d in s["decisions"] if d["dp"] == "tool_params"]
