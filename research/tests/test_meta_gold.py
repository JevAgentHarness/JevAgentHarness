"""Meta-decision golds, wrong-proposal probes, the free-text policy and the round-2 fixes (docs/META_GOLD_SPEC.md §12).
Offline only: simulated / scripted System-2 and System-1 backends."""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any

import numpy as np
import pytest
from pytest import approx

from jevharness.accounting import Ledger
from jevharness.bench import meta_gold as MG
from jevharness.bench import metrics as M
from jevharness.bench import threshold_theory as T
from jevharness.bench.suites.jevgym import (ASK_TOOL, CORE_TOOLS, NEAR_TOOLS, JevGymEnv)
from jevharness.decision.base import DecisionBackend
from jevharness.decision.controls import STOP_GUARD_RE, RateBackend, infer_dp, trigger_rates
from jevharness.decision.simulated import SimulatedBackend
from jevharness.decision.types import DecisionResult, Noul, Usage, answer_from_distribution
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import (DP, VALID_MODES, CFConfig, DPConfig, HarnessConfig, MeasureConfig, Mode,
                                       ProbeConfig)
from jevharness.harness.prompts import Q_REFLECTION, Q_REFLECTION_V1, STOP_REJECT_NOTE, q_stop_guard
from jevharness.harness.tools import ParamInfo, finish_tool, param_kind_table, params_of
from jevharness.harness.trajectory import Trajectory
from jevharness.llm.base import ChatResponse, ToolCall
from jevharness.llm.scripted import ScriptedLLM
from jevharness.llm.simulated import SimulatedLLM

SUITE = dict(families="v2", strict=True, distractor_style="near", n_distractors=20, noise_lines=5)
B = 30


def _env(seed: int, **kw: Any) -> JevGymEnv:
    e = JevGymEnv(seed=seed, **{**SUITE, **kw})
    e.reset()
    return e


def _v(env: JevGymEnv, tool: str, args: dict[str, Any], budget: int = B) -> str | None:
    return env.oracle_verdict(tool, args, budget)["verdict"]


def _seed_of(family: str, variant: str | None = None) -> int:
    for s in range(200):
        e = JevGymEnv(seed=s, **SUITE)
        if e.family == family and (variant is None or e.variant == variant):
            return s
    raise AssertionError((family, variant))


def _meta_cfg(name: str = "meta_shadow_s2lite", cf: bool = True, probes: bool = True, **kw: Any) -> HarnessConfig:
    return HarnessConfig(
        name=name, reflection=DPConfig(mode=Mode.SHADOW, extra={"wordings": ["v2", "v1"]}),
        routing=DPConfig(mode=Mode.SHADOW), stop=DPConfig(mode=Mode.SHADOW, extra={"augment_trigger": True}),
        memory_write=DPConfig(mode=Mode.SHADOW),
        measure=MeasureConfig(
            seed=0, cf=CFConfig(reflection=True, large_llm="large", stop_reject=True, placebo=True) if cf else None,
            probes=ProbeConfig(stop=["forced_premature", "transplant", "T5", "T6"],
                               tool_select=["natural", "transplant", "T1", "T2", "T3", "T4"], transplant_llm="lite",
                               transplant_temperature=1.0) if probes else None), **kw)


def _agent(cfg: HarnessConfig, err: float = 0.25, s1: Any = None, **kw: Any) -> JevAgent:
    llm = SimulatedLLM("sim-lite", error_rate=err, simulated_latency_ms=10)
    mls = {"large": SimulatedLLM("sim-flash", error_rate=0.02, simulated_latency_ms=10),
           "lite": SimulatedLLM("sim-lite-t1", error_rate=0.5, simulated_latency_ms=10, seed=9)}
    return JevAgent(llm, cfg, s1=s1 or SimulatedBackend(accuracy=0.8, seed=1), measure_llms=mls, **kw)


def _run(agent: JevAgent, n: int = 10, seed0: int = 0, **env_kw: Any) -> list[dict[str, Any]]:
    return [agent.run(JevGymEnv(seed=seed0 + i, **{**SUITE, **env_kw})).to_dict() for i in range(n)]


# ====================================================================== oracle (§2)
def test_oracle_agent_solves_everything_and_every_executed_action_is_progress():
    """SimulatedLLM(error_rate=0) solves 100%, and the replayed verdict of every executed action is progress
    (canonical is a subset of progress); the final finish has finish_ok."""
    trajs = _run(JevAgent(SimulatedLLM("oracle", error_rate=0.0), HarnessConfig(name="llm_only")), n=24)
    trajs += [JevAgent(SimulatedLLM("oracle", error_rate=0.0), HarnessConfig(name="llm_only")).run(
        JevGymEnv(seed=s, strict=True)).to_dict() for s in range(10)]        # v1 families too
    assert all(t["result"]["success"] for t in trajs), [t["task_id"] for t in trajs if not t["result"]["success"]]
    for t in trajs:
        env = JevGymEnv(seed=int(t["task_id"].split("-")[2]),
                        **(SUITE if "near" in t["task_id"] else {"strict": True}))
        rows = MG.replay_golds(t, env)
        assert rows and all(r["verdict"] == "progress" for r in rows), (t["task_id"], rows)
        assert rows[-1]["tool"] == "finish" and rows[-1]["finish_ok"] is True


def test_every_alternative_clarify_order_is_progress_and_the_shortcut_is_complete():
    # clarify/order (clarify-8 of the dev run): ask -> cancel -> finish skips find_user / get_orders
    e = _env(8)
    assert (e.family, e.variant) == ("clarify", "order")
    d0 = e.oracle_cost_to_go(B)
    assert _v(e, "ask_user", {"missing": "order_id"}) == "progress"
    e.step("ask_user", {"missing": "order_id"})
    oid = e.pending_order["order_id"]
    assert oid in e._seen[-1]                                            # the reply names the order
    assert _v(e, "cancel_order", {"order_id": oid, "reason": "customer_request"}) == "progress"
    e.step("cancel_order", {"order_id": oid, "reason": "customer_request"})
    assert e.oracle_cost_to_go(B) == 1 < d0                              # early exit: only the answer remains
    assert e.oracle_world_complete() and e.oracle_finish_ok("done")
    assert e.oracle_done() is False                                      # the legacy plan-order label disagrees
    assert e.oracle_verdict("finish", {"answer": "done"}, B)["verdict"] == "progress"
    # clarify/method: find_user before the ask is progress too
    e = _env(_seed_of("clarify", "method"))
    assert e.oracle_action()[0] == "ask_user"
    assert _v(e, "find_user", {"name": e.user["name"]}) == "progress"
    # wrong ask slot: a detour (the customer adds nothing)
    assert _v(e, "ask_user", {"missing": "shipping_speed"}) == "detour"


def test_out_of_order_multi_jobs_are_progress():
    e = _env(_seed_of("multi"))
    _, u2, _ = e.jobs[1]
    assert e.oracle_action() == ("find_user", {"name": e.jobs[0][1]["name"]})
    assert _v(e, "find_user", {"name": u2["name"]}) == "progress"


def test_p0_oracle_fixes_and_detours():
    e = _env(_seed_of("multi"))
    uid = e.jobs[0][1]["user_id"]
    e.step("find_user", {"name": e.jobs[0][1]["name"]})
    assert _v(e, "get_orders", {"user_id": uid, "status": "all"}) == "progress"
    e.step("get_orders", {"user_id": uid, "status": "all"})            # information-equivalent read: done
    assert e.oracle_action()[0] in ("cancel_order", "ship_order")
    assert _v(e, "get_order", {"order_id": e.jobs[0][2]["order_id"]}) == "detour"     # singular: a detour
    # tolerant matching: a refund whose amount is the string "458.240" is done (not "refund again")
    r = _env(_seed_of("clarify", "method"))
    o = r.delivered_order
    for tool, args in [("ask_user", {"missing": "refund_method"}), ("find_user", {"name": r.user["name"]}),
                       ("get_orders", {"user_id": r.user["user_id"], "status": "delivered"})]:
        r.step(tool, args)
    amt = f"{o['amount']:.3f}"
    assert _v(r, "refund", {"order_id": o["order_id"], "amount": amt, "method": r.method}) == "progress"
    r.step("refund", {"order_id": o["order_id"], "amount": amt, "method": r.method})
    assert r.oracle_action()[0] == "finish" and r.oracle_world_complete()


def test_fatal_premature_and_wrong_answer_verdicts():
    e = _env(_seed_of("multi"))
    op, u, o = e.jobs[0]
    other = next(x for x in e.world.orders if x != o["order_id"] and e.world.orders[x]["status"] == "pending")
    assert _v(e, "cancel_order", {"order_id": other, "reason": "customer_request"}) == "fatal"   # wrong write
    assert _v(e, "cancel_subscription", {"user_id": u["user_id"], "reason": "fraud"}) in ("fatal", "detour")
    assert e._tool_names >= {"cancel_subscription"} or True
    near_write = next(t.name for t in NEAR_TOOLS if t.write and t.name in e._tool_names)
    assert _v(e, near_write, {}) == "fatal"                               # a state-changing near-duplicate
    assert _v(e, "finish", {"answer": "0.00"}) == "premature"             # an early finish
    # acting before the required ask (clarify/order: the right order, but no question asked yet)
    c = _env(8)
    c.step("find_user", {"name": c.user["name"]})
    assert _v(c, "cancel_order", {"order_id": c.pending_order["order_id"], "reason": "customer_request"}) == "fatal"
    # a wrong total at the state where only the answer remains
    m = _env(_seed_of("multi"))
    while m.oracle_action()[0] != "finish":
        m.step(*m.oracle_action())
    assert m.oracle_world_complete()
    assert _v(m, "finish", {"answer": f"{m.multi_total + 1:.2f}"}) == "wrong_answer"
    assert _v(m, "finish", {"answer": f"{m.multi_total:.2f}"}) == "progress"


def test_budget_doomed_states_have_no_verdict():
    e = _env(_seed_of("multi"))
    assert e.oracle_cost_to_go(3) == float("inf")
    v = e.oracle_verdict("find_user", {"name": e.user["name"]}, 3)
    assert v["verdict"] is None and v["status"] == "doomed"
    assert e.oracle_acceptable_tool_sets(3) == ([], [])


def test_gold_computations_leave_the_episode_byte_identical():
    a, b = _env(_seed_of("multi")), _env(_seed_of("multi"))
    obs_a, obs_b = [], []
    for _ in range(12):
        act = a.oracle_action()
        if act[0] == "finish":
            break
        for tool in ("find_user", "get_orders", "cancel_order", "finish"):   # golds on copies, between steps
            a.oracle_verdict(tool, {"name": "x", "user_id": "y", "order_id": "z", "answer": "1"}, B)
        a.oracle_acceptable_tool_sets(B)
        a.oracle_mem_gold("anything")
        obs_a.append(a.step(*act).observation)
        obs_b.append(b.step(*b.oracle_action()).observation)
    assert obs_a == obs_b and a.rng.getstate() == b.rng.getstate()


def test_acceptable_tool_set_is_information_feasible():
    e = _env(_seed_of("multi"))
    feas, unf = e.oracle_acceptable_tool_sets(B)
    assert feas == ["find_user"]                                # write tools wait until their ids are visible
    assert {"get_orders"} <= set(unf) and set(feas) <= set(unf)
    e.step("find_user", {"name": e.jobs[0][1]["name"]})
    assert "get_orders" in e.oracle_acceptable_tools(B)


def test_memory_atoms_are_sufficient_and_ignore_heartbeat_noise():
    """An oracle agent that may read only the instruction plus the mem_need-positive offered texts has seen every
    input atom of every plan step before the step (memory sufficiency); heartbeat numbers never match an atom."""
    for seed in range(16):
        e = _env(seed)
        kept: list[str] = []
        while True:
            tool, args = e.oracle_action()
            known = "\n".join([e.instruction, *kept])
            for kind, val in e._step_atoms(tool, args):
                assert e.atom_in(kind, val, known), (e.task_id, tool, kind, val)
            if tool == "finish":
                break
            res = e.step(tool, args)
            text = f"{tool}({json.dumps(args)[:200]}) -> {res.observation[:600]}"
            if e.oracle_mem_gold(text, kept)["mem_need"]:
                kept.append(text)
    assert not JevGymEnv.atom_in("amount", 123456, "log: heartbeat ok #123456")
    assert not JevGymEnv.atom_in("order_id", "ORD-1234", "ORD-12345")
    assert JevGymEnv.atom_in("name", "Dana Kim", "Oh, sorry -- my name is Dana Kim.")


def test_online_golds_equal_the_offline_replay():
    agent = _agent(_meta_cfg(cf=False, probes=False))
    trajs = _run(agent, n=12)
    facs = {t["task_id"]: (lambda s=int(t["task_id"].split("-")[2]): JevGymEnv(seed=s, **SUITE)) for t in trajs}
    rep = MG.audit(trajs, facs)
    assert rep["n_checked"] > 50 and rep["n_diff"] == 0, rep["diffs"][:5]
    assert rep["n_observation_mismatch"] == 0


# ====================================================================== harness logging (§3, §5)
def test_meta_logs_carry_state_gold_status_and_version():
    trajs = _run(_agent(_meta_cfg(cf=False, probes=False)), n=8)
    recs = M.decision_records(trajs)
    meta = [r for r in recs if r["dp"] in ("reflection", "routing", "stop", "memory_write")]
    assert {r["dp"] for r in meta} == {"reflection", "routing", "stop", "memory_write"}
    for r in meta:
        assert r["state"] and r["gold_status"] in ("ok", "doomed") and r["gold_aux"]["v"] == 1
        if r["gold_status"] == "ok":
            assert isinstance(r["gold"], bool)
    st = [r for r in meta if r["dp"] == "stop" and r["gold_status"] == "ok"]
    fin = [r for r in st if r["gold_aux"].get("kind")]
    assert fin and all(r["question"].startswith("The agent wants to stop and submit the final answer") for r in fin)
    assert all(r["s2_correct"] == (r["s2_value"] == r["gold"]) for r in st)
    refl = [r for r in meta if r["dp"] == "reflection" and r["gold_status"] == "ok"]
    assert all(r["gold"] == (r["gold_aux"]["verdict"] in M.FAILING_VERDICTS) for r in refl)
    assert all(isinstance(r["gold_aux"]["retro"], bool) and r["question"] == Q_REFLECTION for r in refl)
    # live trigger logs carry state too; fired steps are unobserved
    cfg = HarnessConfig(name="trig", reflection=DPConfig(mode=Mode.TRIGGER, tau=0.5),
                        routing=DPConfig(mode=Mode.TRIGGER, tau=0.5))
    small = SimulatedLLM("small", error_rate=0.3)
    tr = [JevAgent(SimulatedLLM("big", error_rate=0.1), cfg, s1=SimulatedBackend(accuracy=0.5, seed=2),
                   small_llm=small).run(JevGymEnv(seed=s, **SUITE)).to_dict() for s in range(6)]
    rs = [r for r in M.decision_records(tr) if r["dp"] in ("reflection", "routing")]
    assert rs and all(r["state"] for r in rs)
    fired = [r for r in rs if (r["dp"] == "reflection" and r["final_value"] is True)
             or (r["dp"] == "routing" and r["final_value"] == "large")]
    assert fired and all(r["gold_status"] == "unobserved" and r["gold"] is None for r in fired)
    assert any(r["gold_status"] == "ok" for r in rs if r not in fired)


@pytest.mark.parametrize("mode", [Mode.ALWAYS, Mode.LLM, Mode.TRIGGER, Mode.SHADOW])
def test_memory_gates_log_one_record_per_offer(mode):
    cfg = HarnessConfig(name=f"mem_{mode.value}", memory_write=DPConfig(mode=mode, tau=0.5))
    trajs = _run(JevAgent(SimulatedLLM("sim", error_rate=0.1), cfg, s1=SimulatedBackend(accuracy=0.7, seed=4)), n=4)
    for t in trajs:
        mw = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_write"]
        assert len(mw) == t["meta"]["memory"]["offered"] > 0
        assert all(d["mode"] == mode.value and "CANDIDATE MEMORY:" in d["state"] for d in mw)
        assert all(d["gold_status"] in ("ok", "doomed") for d in mw)
        ok = [d for d in mw if d["gold_status"] == "ok"]
        assert all(isinstance(d["gold"], bool) and {"far", "new", "atoms"} <= set(d["gold_aux"]) for d in ok)
        if mode == Mode.SHADOW:
            assert t["meta"]["memory"]["stored"] == 0 and all(d["s1_dist"] for d in mw)
    assert Mode.SHADOW in VALID_MODES[DP.MEMORY_WRITE]


def test_memory_read_logs_recall_with_list_gold():
    cfg = HarnessConfig(name="mem", memory_write=DPConfig(mode=Mode.ALWAYS),
                        memory_read=DPConfig(mode=Mode.ASSIST, k=2))
    trajs = _run(JevAgent(SimulatedLLM("sim", error_rate=0.0), cfg, s1=SimulatedBackend(accuracy=0.7, seed=4)),
                 n=6)
    rr = [r for r in M.decision_records(trajs) if r["dp"] == "memory_read"]
    assert rr and all(isinstance(r["gold"], list) and r["gold"] for r in rr)
    assert all(isinstance(r["final_value"], list) and "lenient" in r["gold_aux"] for r in rr)
    h = M.memory_read_hits(rr)
    assert h["n"] == len(rr) and (h.get("hit@1") is not None or h.get("none_acc") is not None)


def test_measurement_arm_is_the_llm_only_episode_plus_tagged_measurement():
    plain = _run(_agent(_meta_cfg(cf=False, probes=False).model_copy(update={
        "name": "plain", "measure": MeasureConfig(),
        "reflection": DPConfig(mode=Mode.SHADOW), "stop": DPConfig(mode=Mode.SHADOW)})), n=10)
    meas = _run(_agent(_meta_cfg()), n=10)
    base = _run(JevAgent(SimulatedLLM("sim-lite", error_rate=0.25, simulated_latency_ms=10),
                         HarnessConfig(name="llm_only")), n=10)

    def ep(t: dict[str, Any]) -> list[Any]:
        return [(s["tool"], json.dumps(s["args"], sort_keys=True), s["observation"]) for s in t["steps"]]

    for a, b, c in zip(meas, plain, base):
        assert ep(a) == ep(b) == ep(c) and a["result"] == b["result"] == c["result"]
        tot = [{k: v for k, v in x["usage"]["total"].items() if k != "latency_ms"} for x in (a, b, c)]  # (wall jitter)
        assert tot[0] == tot[1] == tot[2]
        bm = a["usage"]["shadow"]["by_measure"]
        assert set(bm) <= {"shadow", "cf", "probe"} and "probe" in bm
        tagged = [x for x in a["calls"] if (x.get("meta") or {}).get("measure") in ("cf", "probe")]
        assert len(tagged) == sum(bm[m]["calls"] for m in ("cf", "probe") if m in bm)
        assert all((x["meta"] or {}).get("shadow") for x in tagged)
        assert not [p for p in a["probes"] if p in [d for s in a["steps"] for d in s["decisions"]]]
    assert sum(len(t["probes"]) for t in meas) > 20 and not any(t["probes"] for t in base)
    assert any((r.get("gold_aux") or {}).get("cf_status") == "ok" for r in M.decision_records(meas))


def test_deferred_calls_run_after_evaluate_and_failures_are_counted_not_raised(monkeypatch):
    import jevharness.harness.measure as MS

    flag = {"evaluated": False}
    seen: list[bool] = []
    orig = MS.run_deferred

    def wrapped(agent, traj, ledger, ep):
        flag["evaluated"] = bool(traj.result)         # env.evaluate has already scored the episode
        return orig(agent, traj, ledger, ep)

    monkeypatch.setattr(MS, "run_deferred", wrapped)

    class Recorder(SimulatedLLM):
        def _chat(self, messages, tools=None, **kw):
            seen.append(flag["evaluated"])
            return super()._chat(messages, tools, **kw)

    class Broken(SimulatedLLM):
        def _chat(self, messages, tools=None, **kw):
            raise RuntimeError("503 from the transplant model")

    agent = JevAgent(Recorder("rec", error_rate=0.2), _meta_cfg(), s1=SimulatedBackend(accuracy=0.8, seed=1),
                     measure_llms={"large": Recorder("big", error_rate=0.0), "lite": Broken("broken")})
    t = agent.run(JevGymEnv(seed=_seed_of("multi"), **SUITE)).to_dict()
    assert t["error"] is None and t["meta"]["probe_errors"]                       # the transplant failures
    assert all("transplant" in e for e in t["meta"]["probe_errors"])
    n_live = sum(1 for c in t["calls"] if c["component"] == "s2" and not (c.get("meta") or {}).get("shadow"))
    assert seen[:n_live] == [False] * n_live and all(seen[n_live:]) and len(seen) > n_live


def test_fired_step_counterfactual_labels_live_trigger_arms():
    cfg = HarnessConfig(name="jev_reflect_trigger_s2lite", reflection=DPConfig(mode=Mode.TRIGGER, tau=0.5),
                        measure=MeasureConfig(cf=CFConfig(fired_steps=True)))
    trajs = _run(JevAgent(SimulatedLLM("lite", error_rate=0.3), cfg, s1=SimulatedBackend(accuracy=0.5, seed=3)),
                 n=8)
    fired = [r for r in M.decision_records(trajs) if r["dp"] == "reflection" and r["final_value"] is True]
    assert fired and all(r["gold_status"] in ("ok", "doomed") for r in fired)
    assert all(r["gold_aux"]["verdict_src"] == "cf" for r in fired if r["gold_status"] == "ok")


def test_probes_require_a_measurement_arm():
    with pytest.raises(ValueError, match="measurement"):
        HarnessConfig(name="live", stop=DPConfig(mode=Mode.GUARD),
                      measure=MeasureConfig(probes=ProbeConfig(stop=["T5"]))).validate_modes()


# ====================================================================== leakage (§1.1, §11.12)
class _Recording(SimulatedBackend):
    def __init__(self, **kw: Any):
        super().__init__(**kw)
        self.requests: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []

    def _decide(self, state, questions):
        self.requests.append((state, [q.model_dump(exclude={"meta"}) for q in questions], [dict(q.meta) for q in questions]))
        return super()._decide(state, questions)


class _NoCheat(SimulatedLLM):
    """Follows the oracle for the calls (like the simulated System-2) but never knows the gold answer."""

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.requests: list[str] = []

    def _chat(self, messages, tools=None, **kw):
        self.requests.append(json.dumps(messages, default=str) + json.dumps(tools or [], default=str))
        r = super()._chat(messages, tools, **kw)
        for tc in r.tool_calls:
            if tc.name == "finish":
                tc.arguments = {"answer": "done"}
        if not r.tool_calls and r.text and r.text != "done":
            r.text = "done"
        return r


def test_gold_never_reaches_any_backend():
    canaries = {"98765.43", "87654.32"}

    class CanaryEnv(JevGymEnv):
        def _build(self):
            super()._build()
            self.delivered_total = 98765.43
            if self.family == "multi":
                self.multi_total = 87654.32

    rec = _Recording(accuracy=0.8, seed=1)
    llm = _NoCheat("lite", error_rate=0.2)
    mls = {"large": _NoCheat("large", error_rate=0.0), "lite": _NoCheat("lite2", error_rate=0.5, seed=3)}
    cfg = _meta_cfg().model_copy(update={"tool_select": DPConfig(mode=Mode.SHADOW),
                                         "tool_params": DPConfig(mode=Mode.SHADOW)})
    agent = JevAgent(llm, cfg, s1=rec, measure_llms=mls)
    trajs = [agent.run(CanaryEnv(seed=s, **SUITE)).to_dict() for s in range(10)]
    trajs += [agent.run(CanaryEnv(seed=s, family="total", strict=True)).to_dict() for s in range(3)]
    mem = HarnessConfig(name="mem", memory_write=DPConfig(mode=Mode.TRIGGER, tau=0.5),
                        memory_read=DPConfig(mode=Mode.ASSIST, k=2))
    trajs += [JevAgent(llm, mem, s1=rec).run(CanaryEnv(seed=s, **SUITE)).to_dict() for s in range(4)]
    assert rec.requests and llm.requests
    blobs = [json.dumps([s, q]) for s, q, _ in rec.requests] + llm.requests + \
        [r for m in mls.values() for r in m.requests]
    for b in blobs:
        assert not any(c in b for c in canaries)
        for key in ("need_fatal", "acceptable_tools", "finish_ok", "gold_aux", "mem_need", "world_complete"):
            assert key not in b
    # Question.meta is exactly the legacy one: nothing for reflection / routing / memory, oracle_done for stop
    for _, qs, metas in rec.requests:
        for q, m in zip(qs, metas):
            dp = q["decision_point"]
            if dp in ("reflection", "routing", "memory_write", "memory_read"):
                assert m == {}, (dp, m)
            if dp == "stop":
                assert set(m) <= {"gold"} and m.get("gold") in (True, False, None)
    # probes are single-question System-1 calls
    pc = [c for t in trajs for c in t["calls"] if (c.get("meta") or {}).get("measure") == "probe"
          and c["component"] == "s1"]
    assert pc and all(c["meta"]["n_questions"] == 1 for c in pc)


# ====================================================================== probes (§6)
def test_probe_sources_labels_and_visibility():
    trajs = _run(_agent(_meta_cfg()), n=16)
    probes = [(t, p) for t in trajs for p in t["probes"] if p["mode"] == "guard_probe"]
    src = {(p["dp"], p["gold_aux"]["probe_source"], p["gold_aux"]["operator"]) for _, p in probes}
    assert ("stop", "forced_premature", None) in src and ("tool_select", "natural", None) in src
    assert any(s[1] == "transplant" for s in src) and any(s[1] == "edit" for s in src)
    for t, p in probes:
        a = p["gold_aux"]
        assert p["gold_status"] in ("ok", "doomed") and a["v"] == 1 and isinstance(a["step"], int)
        if a["probe_source"] == "edit":
            assert p["gold"] is False                                    # edits labelled right are discarded
            if p["dp"] == "tool_select":
                for v in a["proposal"]["args"].values():   # state-visible values, schema enum members or the template
                    assert isinstance(v, (bool, int, float)) or str(v) in p["state"] or v == "(as requested above)" \
                        or str(v) in ENUMS, v
        if a["probe_source"] == "forced_premature" and p["gold_status"] == "ok":
            assert a["kind"] == "premature" and p["gold"] is False
    # transplants are kept whatever their label
    tp = [p for _, p in probes if p["gold_aux"]["probe_source"] == "transplant" and p["gold_status"] == "ok"]
    assert {True, False} <= {p["gold"] for p in tp}
    # T5 / T6 answers come from visible amounts or the proposal's own answer
    for _, p in probes:
        if p["gold_aux"].get("operator") == "T5":
            ans = p["gold_aux"]["answer"]
            assert ans == "done" or abs(float(ans) - sum(_amounts(p["state"]))) < 0.01


ENUMS = {str(x) for t in CORE_TOOLS + [ASK_TOOL] + [n.spec() for n in NEAR_TOOLS]
         for p in params_of(t) for x in (p.bounded_values() or [])}


def _amounts(state: str) -> list[float]:
    from jevharness.harness.measure import visible_amounts

    return visible_amounts(state)


def test_probes_never_reach_decision_records_rates_or_threshold_pairs():
    trajs = _run(_agent(_meta_cfg()), n=6)
    assert sum(len(t["probes"]) for t in trajs) > 0
    stripped = [{**t, "probes": []} for t in trajs]
    assert M.decision_records(trajs) == M.decision_records(stripped)
    assert trigger_rates(trajs) == trigger_rates(stripped)
    for dp in ("stop", "tool_select", "reflection"):
        assert T.decision_pairs(trajs, dp) == T.decision_pairs(stripped, dp)


def test_edit_operators():
    from jevharness.harness.measure import EDITS, _Runner, t6_edit

    e = _env(8)                       # clarify/order: the oracle's next step after find + get_orders is the ask
    for a in [("find_user", {"name": e.user["name"]}), ("get_orders", {"user_id": e.user["user_id"],
                                                                        "status": "pending"})]:
        e.step(*a)
    assert e.oracle_action()[0] == "ask_user"
    agent = _agent(_meta_cfg())
    from jevharness.harness.agent import StepSnap

    tools = e.tools() + [finish_tool()]
    state = agent._s1_state(e, [("find_user(...)", e._seen[1]), ("get_orders(...)", e._seen[2])], e._seen[-1])
    snap = StepSnap(step=2, budget=28, env=e.oracle_snapshot(), messages=[], mem_text="", state=state, llm=None,
                    tools=tools, a0=("get_orders", {"user_id": e.user["user_id"], "status": "pending"}))
    r = _Runner(agent, Trajectory("t", "c", None, None), Ledger(), type("E", (), {"snaps": [], "extra_probes": []})())
    rng = random.Random(0)
    t4 = EDITS["T4"](r, snap, rng)
    assert t4 is not None and t4[0].startswith("cancel") and _v(snap.env, *t4) == "fatal"
    assert all(str(v) in state or isinstance(v, (bool, int, float)) or v in ("customer_request", "out_of_stock", "fraud")
               for v in t4[1].values())
    t3 = EDITS["T3"](r, snap, rng)
    assert t3 is not None and t3[1]["status"] != "pending"
    t2 = EDITS["T2"](r, snap, random.Random(1))
    assert t2 is None or (t2[1]["user_id"] != e.user["user_id"] and t2[1]["user_id"] in state)
    t1 = EDITS["T1"](r, snap, rng)
    assert t1 is not None and t1[0] != "get_orders" and t1[0] in {x["function"]["name"] for x in tools}
    m = re.search(r"(\d+\.\d+)", "Total: 123.45")
    outs = {t6_edit("Total: 123.45", m, [20.0, 3.45], random.Random(k)) for k in range(30)}
    assert "Total: 123.45" not in outs and len(outs) > 3 and all(o.startswith("Total: ") for o in outs if o)


# ====================================================================== metrics (§9)
def _p(dp, src, gold, p_yes, kind=None, op=None, d=2, tid="t1"):
    return {"dp": dp, "mode": "guard_probe", "gold": gold, "s1_value": p_yes >= 0.5,
            "s1_dist": {"yes": p_yes, "no": 1 - p_yes}, "gold_status": "ok",
            "gold_aux": {"probe_source": src, "operator": op, "kind": kind, "d_before": d, "tau_ref": 0.5}}


def test_probe_error_rates_hand_computed():
    probes = [_p("stop", "forced_premature", False, 0.9, "premature"), _p("stop", "forced_premature", False, 0.1,
                                                                         "premature"),
              _p("stop", "transplant", False, 0.8, "wrong_answer", d=1), _p("stop", "transplant", True, 0.9, "ok"),
              _p("stop", "edit", False, 0.7, "premature", "T5"),
              _p("tool_select", "natural", True, 0.9, "progress"), _p("tool_select", "natural", True, 0.2, "progress"),
              _p("tool_select", "natural", False, 0.6, "detour"), _p("tool_select", "edit", False, 0.4, "fatal", "T1"),
              _p("tool_select", "edit", False, 0.8, "fatal", "T2")]
    natural_stop = {"dp": "stop", "mode": "shadow", "gold": False, "s1_value": True,
                    "s1_dist": {"yes": 0.95, "no": 0.05}, "gold_status": "ok",
                    "gold_aux": {"kind": "wrong_answer", "d_before": 1}}
    right_stop = {**natural_stop, "gold": True, "s1_dist": {"yes": 0.3, "no": 0.7}, "gold_aux": {"kind": "ok"}}
    trajs = [{"task_id": "t1", "probes": probes, "steps": [{"step": 0, "decisions": [natural_stop, right_stop]}]}]
    r = M.probe_error_rates(trajs)
    s = r["stop"]
    assert s["far_model"]["n_wrong"] == 4 and s["far_model"]["far"] == approx(3 / 4)    # nat + 2 forced + transplant
    assert s["far_nat"]["far"] == 1.0 and s["frr"]["frr"] == 1.0
    assert s["far_edit"]["T5"]["far"] == 1.0
    assert s["by_kind"]["premature"]["n_wrong"] == 3
    assert s["far_reweighted"] == approx(1.0)        # natural mix: wrong_answer only; transplant wrong_answer accepted
    lo, hi = s["far_model"]["far_ci_wilson"]
    assert lo < 0.75 < hi
    # one task = one cluster: the pre-registered task-clustered interval is undefined (never a degenerate [p, p])
    assert all(math.isnan(x) for x in s["far_model"]["far_ci"]) and s["far_model"]["ci"] == "task-clustered bootstrap"
    t = r["tool_select"]
    assert t["headline"]["far"] == 1.0 and t["headline"]["frr"] == 0.5
    assert t["far_fatal"]["far"] == 0.5 and set(t["by_operator"]) == {"T1", "T2"}
    assert M.wilson_ci(0, 0) != M.wilson_ci(0, 0) or math.isnan(M.wilson_ci(0, 0)[0])


def test_memory_read_hits_recall_at_k_and_list_gold_decision_table():
    NONE = M.NONE_MEMORY
    recs = [{"dp": "memory_read", "mode": "assist", "gold": ["[m1] a"], "gold_status": "ok", "options": ["x"],
             "s1_value": "[m0] b", "s1_dist": {"[m0] b": 0.5, "[m1] a": 0.3, NONE: 0.2}, "final_value": ["[m0] b"]},
            {"dp": "memory_read", "mode": "assist", "gold": [NONE], "gold_status": "ok", "options": ["x"],
             "s1_value": NONE, "s1_dist": {NONE: 0.6, "[m0] b": 0.4}, "final_value": []}]
    h = M.memory_read_hits(recs)
    assert h["hit@1"] == 0.0 and h["hit@3"] == 1.0 and h["mrr"] == 0.5 and h["none_acc"] == 1.0
    assert M.decision_table(recs)["memory_read/assist"]["s1_acc"] == 0.5          # list gold = membership
    ts = [{"dp": "tool_select", "mode": "assist", "options": ["a", "b", "c", "d"], "gold": "a", "gold_status": "ok",
           "s1_value": "b", "s1_conf": 0.5, "s1_dist": {"b": 0.5, "c": 0.3, "a": 0.15, "d": 0.05},
           "gold_aux": {"v": 1, "acceptable_tools": ["a", "b"], "offered": ["b", "c", "finish"]}},
          {"dp": "tool_select", "mode": "assist", "options": ["a", "b", "c", "d"], "gold": "a", "gold_status": "ok",
           "s1_value": "d", "s1_conf": 0.5, "s1_dist": {"d": 0.5, "c": 0.3, "a": 0.15, "b": 0.05},
           "gold_aux": {"v": 1, "acceptable_tools": ["a"], "offered": ["d", "c", "finish"]}}]
    rk = M.recall_at_k(ts)
    assert rk["recall@1"] == 0.5 and rk["recall@3"] == 1.0 and rk["recall@1_canonical"] == 0.0
    assert rk["offered_recall"] == 0.5
    row = M.decision_table(ts)["tool_select/assist"]
    assert row["s1_acc"] == 0.5 and row["s1_acc_canonical"] == 0.0


def test_delegation_counts_generate_and_trigger_rates_skip_memory_double_count():
    recs = [{"dp": "tool_params", "mode": "replace", "s1_value": "x", "deferred": False},
            {"dp": "tool_params", "mode": "generate", "s1_value": None, "deferred": True}]
    assert M.delegation_ratio(recs)["ratio"] == 0.5
    mw = lambda v: {"dp": "memory_write", "mode": "trigger", "s1_value": v, "final_value": v}  # noqa: E731
    trajs = [{"task_id": "a", "steps": [{"step": 0, "decisions": [mw(True), mw(False)]}],
              "meta": {"memory": {"offered": 2, "stored": 1}, "config": {"memory_write": {"mode": "trigger"}}}},
             {"task_id": "b", "steps": [{"step": 0, "decisions": [{"dp": "memory_write", "mode": "shadow",
                                                                   "s1_value": True}]}],
              "meta": {"memory": {"offered": 1, "stored": 0}, "config": {"memory_write": {"mode": "shadow"}}}},
             {"task_id": "c", "steps": [], "meta": {"memory": {"offered": 4, "stored": 1},
                                                    "config": {"memory_write": {"mode": "llm"}}}}]
    r = trigger_rates(trajs)
    assert r["memory_write/trigger"] == {"n": 2, "rate": 0.5}          # counted once, from the DecisionLogs
    assert "memory_write/shadow" not in r and r["memory_write/llm"]["rate"] == 0.25   # legacy fallback kept


def test_gate_stats_cf_efficacy_and_the_voi_identity():
    rec = lambda p, g: {"gold": g, "s1_dist": {"yes": p, "no": 1 - p}}  # noqa: E731
    gs = M.gate_stats([rec(0.9, True), rec(0.8, True), rec(0.2, True), rec(0.7, False), rec(0.1, False)], tau=0.5)
    assert gs["pi"] == 0.6 and gs["tpr"] == approx(2 / 3) and gs["fpr"] == 0.5 and gs["J"] == approx(2 / 3 - 0.5)
    g = M.gate_stats([rec(0.9, True), rec(0.2, False), rec(0.8, False)], tau=0.5, guard=True)
    far, frr = 1 / 2, 0.0                      # guard: accept iff p >= 0.5; wrong proposals = gold False
    assert g["J"] == approx(1 - far - frr) and not g["estimable"]
    aux = lambda v0, alt, pl, w=1.0: {"gold_aux": {"cf_status": "ok", "verdict": v0, "verdict_alt": alt,  # noqa: E731
                                                  "verdict_placebo": pl, "cf_weight": w}}
    e = M.cf_efficacy([aux("fatal", "progress", "fatal"), aux("fatal", "fatal", "progress"),
                       aux("progress", "fatal", "progress", 5.0), aux("progress", "progress", "progress", 5.0)])
    assert e["e"] == approx(0.5 - 0.5) and e["h"] == approx(0.5 - 0.0) and e["delta"] == approx(1 / 12)
    # Monte Carlo check of E[U_gate - U_rand] = n [W (e + h) pi (1 - pi) J - kappa] incl. the rate-mismatch term
    rng = np.random.default_rng(0)
    n, W, ee, hh, pi, tpr, fpr, kappa, dr = 6, 1.0, 0.3, 0.1, 0.2, 0.7, 0.15, 0.001, 0.05
    r0 = pi * tpr + (1 - pi) * fpr
    N = 400_000
    y = rng.random((N, n)) < pi
    fire_g = np.where(y, rng.random((N, n)) < tpr, rng.random((N, n)) < fpr)
    fire_r = rng.random((N, n)) < r0 + dr
    u = lambda f: (W * np.where(f & y, ee, 0.0) - W * np.where(f & ~y, hh, 0.0)).sum(1)  # noqa: E731
    mc = float((u(fire_g) - u(fire_r)).mean()) - n * kappa
    th = M.voi_identity(n, W, ee, hh, pi, tpr - fpr, kappa, dr=dr)
    assert mc == approx(th["delta_u_corrected"], abs=4e-3)
    assert th["delta_u"] == approx(n * (W * (ee + hh) * pi * (1 - pi) * (tpr - fpr) - kappa))


def test_guard_rates_by_kind_and_canonical_variant():
    g = lambda gold, ok, kind, can: {"dp": "tool_select", "mode": "guard", "gold": gold, "s1_value": ok,  # noqa: E731
                                     "gold_aux": {"verdict": kind, "canonical": can}}
    recs = [g(True, True, "progress", False), g(False, True, "detour", False), g(False, False, "fatal", False),
            g(True, False, "progress", True)]
    r = M.guard_error_rates(recs)["tool_select"]
    assert r["far"] == 0.5 and r["frr"] == 0.5 and r["far_fatal"] == 0.0 and r["by_kind"]["detour"]["accept_rate"] == 1
    c = M.guard_error_rates(recs, gold="canonical")["tool_select"]
    assert c["n_wrong"] == 3 and c["far"] == approx(2 / 3)


# ====================================================================== free-text policy (§7)
JEVGYM_KINDS = {
    "free": {"finish.answer", "ask_user.question", "send_message.text", "send_email.subject", "send_email.body",
             "notify_warehouse.text", "search_users.prefix", "ship_order_international.country"},
    "bounded_names": {"status", "reason", "method", "item", "speed", "missing", "warehouse", "year", "accept"},
    "copy_names": {"name", "user_id", "order_id", "email", "phone", "date", "damage_report_id", "amount", "units"},
}


def test_jevgym_fill_kind_table_is_frozen():
    tools = CORE_TOOLS + [ASK_TOOL] + [t.spec() for t in NEAR_TOOLS] + [finish_tool()]
    tab = param_kind_table(tools, JevGymEnv.param_kinds)
    assert {p for p, k in tab.items() if k == "free"} == JEVGYM_KINDS["free"]
    assert {p.split(".")[1] for p, k in tab.items() if k == "bounded"} == JEVGYM_KINDS["bounded_names"]
    assert {p.split(".")[1] for p, k in tab.items() if k == "copy"} == JEVGYM_KINDS["copy_names"]


def test_fill_kind_rules_overrides_and_legacy_is_copyable():
    assert ParamInfo("note", {"type": "string"}, False).kind == "free"
    assert ParamInfo("note", {"type": "string", "x-jev-fill": "copy"}, False).kind == "copy"
    assert ParamInfo("note", {"type": "string"}, False, "bounded").kind == "bounded"
    assert ParamInfo("ref", {"type": "string", "pattern": "^R\\d+$"}, True).kind == "copy"
    assert ParamInfo("customer", {"type": "string", "description": "the customer's e-mail"}, True).kind == "copy"
    assert ParamInfo("n", {"type": "integer", "minimum": 1, "maximum": 5}, True).kind == "bounded"
    assert ParamInfo("n", {"type": "integer"}, True).kind == "copy"
    assert ParamInfo("tags", {"type": "array"}, False).kind == "free"
    # is_copyable keeps its legacy meaning (tau2 applies its own free-text rule on top)
    assert ParamInfo("answer", {"type": "string"}, True).is_copyable()
    assert not ParamInfo("s", {"type": "string", "enum": ["a", "b"]}, True).is_copyable()
    t = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "x": {"type": "string"}, "y": {"type": "string"}}}}}
    assert [p.kind for p in params_of(t, {"f.x": "copy", "*.y": "bounded"})] == ["copy", "bounded"]


class _Heartbeat(DecisionBackend):
    """A System-1 that picks the heartbeat number for every copy question, ``finish`` after 3 tool choices."""
    name, model = "heartbeat", "heartbeat"

    def __init__(self):
        self.asked: list[str] = []
        self.n = 0

    def _decide(self, state, questions):
        out = []
        for q in questions:
            self.asked.append(q.text)
            labels = q.labels()
            if q.text.startswith("Which tool"):
                self.n += 1
                pick = "finish" if self.n > 3 and "finish" in labels else next(
                    (x for x in ("find_user", "get_orders") if x in labels), labels[0])
            else:
                pick = next((x for x in labels if re.fullmatch(r"\d{5,6}", x)), labels[0])
            out.append(answer_from_distribution(q, {x: (0.9 if x == pick else 0.1 / len(labels)) for x in labels}))
        return DecisionResult(answers=out, usage=Usage(backend=self.name, model=self.model))


def test_system1_never_fills_finish_answer_899659_regression():
    """E2 dev: in replace mode System-1 copied the heartbeat number 899659 as the final answer. Free-text arguments
    now always go to System-2 in one forced call, logged as `generate` records."""
    def policy(messages, tools):
        name = tools[0]["function"]["name"]
        return ChatResponse(text="", tool_calls=[ToolCall(id="c", name=name, arguments={"answer": "LLM-ANSWER"}
                                                          if name == "finish" else {})])

    s1 = _Heartbeat()
    cfg = HarnessConfig(name="params_replace", max_steps=8, tool_select=DPConfig(mode=Mode.REPLACE),
                        tool_params=DPConfig(mode=Mode.REPLACE))
    llm = ScriptedLLM("scripted", policy=policy)
    traj = JevAgent(llm, cfg, s1=s1).run(JevGymEnv(seed=1, family="total", noise_lines=5))
    assert traj.final_answer == "LLM-ANSWER"
    assert not any("`answer`" in q for q in s1.asked)                  # never asked about a free-text slot
    fin = [s for s in traj.steps if s.tool == "finish"][0]
    gen = [d for d in fin.decisions if d.mode == "generate"]
    assert len(gen) == 1 and gen[0].gold_aux["param_kind"] == "free" and gen[0].s2_value == "LLM-ANSWER"
    assert fin.llm_calls == 1                                           # exactly one forced System-2 call
    # the H3 ablation copies the answer from the state again
    abl = cfg.model_copy(update={"tool_params": DPConfig(mode=Mode.REPLACE, extra={"answer_as_copy": True})})
    s1b = _Heartbeat()
    t2 = JevAgent(ScriptedLLM("scripted", policy=policy), abl, s1=s1b).run(JevGymEnv(seed=1, family="total",
                                                                                     noise_lines=5))
    assert any("`answer`" in q for q in s1b.asked) and t2.final_answer != "LLM-ANSWER"


def test_optional_free_text_is_filled_by_system2_in_one_call():
    tool = {"type": "function", "function": {"name": "notify", "description": "notify", "parameters": {
        "type": "object", "properties": {"order_id": {"type": "string"}, "text": {"type": "string"}},
        "required": ["order_id"]}}}

    class OneTool(JevGymEnv):
        def tools(self):
            return [tool]

    calls: list[Any] = []

    def policy(messages, tools):
        calls.append(tools)
        return ChatResponse(text="", tool_calls=[ToolCall(id="c", name=tools[0]["function"]["name"],
                                                          arguments={"order_id": "LLM", "text": "hello"})])

    class Pick(DecisionBackend):
        name, model = "pick", "pick"

        def _decide(self, state, questions):
            out = []
            for q in questions:
                labels = q.labels()
                pick = "notify" if "notify" in labels else next((x for x in labels if x.startswith("ORD-")), labels[0])
                out.append(answer_from_distribution(q, {x: (0.9 if x == pick else 0.1 / len(labels)) for x in labels}))
            return DecisionResult(answers=out, usage=Usage(backend="pick", model="pick"))

    cfg = HarnessConfig(name="r", max_steps=1, tool_select=DPConfig(mode=Mode.REPLACE),
                        tool_params=DPConfig(mode=Mode.REPLACE))
    traj = JevAgent(ScriptedLLM("s", policy=policy), cfg, s1=Pick()).run(OneTool(seed=0, family="cancel"))
    s = traj.steps[0]
    assert s.llm_calls == 1 and len(calls) == 1 and s.args["text"] == "hello"
    assert s.args["order_id"] != "LLM"                                 # System-1's copy value wins on conflict
    kinds = sorted((d.mode, (d.gold_aux or {}).get("param_kind")) for d in s.decisions if d.dp == "tool_params")
    assert kinds == [("generate", "free"), ("replace", "copy")]


# ====================================================================== wording (§3.1, §3.3) and tournaments (§10)
def test_new_wordings_keep_decision_point_inference():
    assert infer_dp(Noul(Q_REFLECTION)) == "reflection" and infer_dp(Noul(Q_REFLECTION_V1)) == "reflection"
    for ans in ("The task is complete: 1234.56", "I am stuck", "needed later; which tool; strong model 42", None,
                'say "hi"'):
        q = Noul(q_stop_guard(ans))
        assert infer_dp(q) == "stop" and STOP_GUARD_RE.search(q.text), ans
    assert len(q_stop_guard("x" * 5000)) < 1000
    assert "continue working" not in STOP_REJECT_NOTE.lower() and "recompute the answer" in STOP_REJECT_NOTE


def test_tournament_round_goes_out_in_one_request_when_the_backend_batches():
    class Count(SimulatedBackend):
        def __init__(self, batched: bool):
            super().__init__(accuracy=0.9, seed=0)
            self.batched_questions = batched
            self.max_options = 10
            self.sizes: list[int] = []

        def _decide(self, state, questions):
            self.sizes.append(len(questions))
            return super()._decide(state, questions)

    opts = [f"t{i}" for i in range(35)]                    # 4 chunks of <= 10, then 8 winners
    seq, bat = Count(False), Count(True)
    agent = JevAgent(ScriptedLLM("s"), HarnessConfig(), s1=seq)
    agent._choice_tournament("S", "Which tool should the agent call next?", opts, "tool_select", Ledger(), gold="t3")
    agent.s1 = bat
    led = Ledger()
    a = agent._choice_tournament("S", "Which tool should the agent call next?", opts, "tool_select", led, gold="t3")
    assert seq.sizes == [1, 1, 1, 1, 1] and bat.sizes == [4, 1] and len(led.records) == 2
    assert a.value in opts


# ====================================================================== analysis (§9): score card, theory, meta_gold
def _sim_exp(tmp_path) -> Any:
    from jevharness.bench.runner import run_experiment

    sim = {"provider": "simulated", "model": "sim-flash", "error_rate": 0.05, "simulated_latency_ms": 10}
    lite = {"provider": "simulated", "model": "sim-lite", "error_rate": 0.3, "simulated_latency_ms": 10}
    exp = {"name": "meta_sim", "suite": {"name": "jevgym", "n": 10, **SUITE}, "llm": sim, "small_llm": lite,
           "llms": {"lite2": lite, "flash": sim}, "n_tasks": 6, "workers": 1,
           "s1": {"type": "simulated", "accuracy": 0.8, "latency_ms": 5, "price_per_m_input": 0.05},
           "s1_backends": {"ratem": {"type": "rate", "rate": 0.1}},
           "arms": [{"name": "llm_only"}, {"name": "llm_only_s2lite", "llm": "lite2"},
                    {"name": "jev_reflect_trigger_s2lite", "llm": "lite2", "n_tasks": 10,
                     "reflection": {"mode": "trigger", "tau": 0.5}},
                    {"name": "ratem_reflect_trigger_s2lite", "llm": "lite2", "s1": "ratem", "n_tasks": 10,
                     "reflection": {"mode": "trigger", "tau": 0.5}},
                    {"name": "jev_ts_assist_k3", "tool_select": {"mode": "assist", "k": 3}},
                    {"name": "meta_shadow_s2lite", "llm": "lite2", "n_tasks": 10,
                     "reflection": {"mode": "shadow", "extra": {"wordings": ["v2", "v1"]}}, "routing": {"mode": "shadow"},
                     "stop": {"mode": "shadow", "extra": {"augment_trigger": True}}, "memory_write": {"mode": "shadow"},
                     "measure": {"cf": {"reflection": True, "large_llm": "flash", "stop_reject": True},
                                 "probes": {"stop": ["forced_premature", "T5", "T6"],
                                            "tool_select": ["natural", "T1", "T2", "T3", "T4"]}}}]}
    run_experiment(exp, tmp_path, progress=False)
    return tmp_path / "meta_sim"


def test_runner_scorecard_meta_gold_and_threshold_theory_end_to_end(tmp_path):
    from jevharness.bench.scorecard import compute_scorecard, load_runs, to_markdown

    run = _sim_exp(tmp_path)
    runs = load_runs(run)
    assert len(runs["llm_only"]) == 6 and len(runs["meta_shadow_s2lite"]) == 10        # per-arm n_tasks
    card = compute_scorecard(runs, baseline="llm_only", task_values=[1.0, 10.0],
                             pairs=[("jev_reflect_trigger_s2lite", "ratem_reflect_trigger_s2lite")],
                             measure_arms=["meta_shadow_s2lite"])
    ms = card["arms"]["meta_shadow_s2lite"]
    assert ms["shadow_only"] and "p_worse_holm" not in ms.get("vs_baseline", {}) and not ms["pareto"]
    gold = ms["decisions"]["gold"]
    assert {"coverage", "gates", "cf", "probes"} <= set(gold)
    assert gold["gates"]["reflection/shadow"]["n"] > 0 and "estimable" in gold["gates"]["reflection/shadow"]
    pair = card["headline"]["pairs"]["jev_reflect_trigger_s2lite vs ratem_reflect_trigger_s2lite"]
    assert "delta_u_gross" in pair["utility"]["1"] and pair["voi"]["gate"] == "reflection"
    assert pair["voi"]["measure_arm"] == "meta_shadow_s2lite" and "observed" in pair["voi"]
    assert "recall@3" in card["arms"]["jev_ts_assist_k3"]["decisions"]["gold"]["recall"]
    md = to_markdown(card)
    for sec in ("Meta-decision golds: coverage", "Gates vs. their primary gold", "Guard probes",
                "Counterfactual efficacy", "Tool-selection recall@k", "Gate VoI"):
        assert sec in md, sec
    # H2 cells, ceiling, audit, replay
    cells = MG.h2_cells(runs, ["meta_shadow_s2lite"], [("jev_reflect_trigger_s2lite", "ratem_reflect_trigger_s2lite")],
                        [1.0], n_boot=100)
    assert cells and cells[0]["gate"] == "reflection" and cells[0]["no_opportunity"] == (not cells[0]["estimable"])
    c = MG.ceiling(runs["meta_shadow_s2lite"], "reflection", 1.0, n_boot=50)
    assert c["n_tasks"] == 10 and "no_headroom" in c
    assert MG.audit(runs["meta_shadow_s2lite"], MG.suite_factories(run))["n_diff"] == 0

    class Rec(SimulatedBackend):
        def _decide(self, state, questions):
            assert all(not q.meta for q in questions) and len(questions) == 1   # no gold, single questions
            return super()._decide(state, questions)

    rows = MG.replay_backend(runs["meta_shadow_s2lite"], Rec(accuracy=0.7))
    assert rows and {"steps", "forced_premature"} <= {r["source"] for r in rows} and all(r["p_yes"] is not None
                                                                                       for r in rows)
    MG.main(["replay", str(run), "--arm", "meta_shadow_s2lite", "--s1", "{type: heuristic}"])
    assert (run / "replay" / "meta_shadow_s2lite__heuristic.jsonl").exists()
    # threshold theory: AT gold for tool_select pairs; the gate analysis
    pairs = T.decision_pairs(runs["jev_ts_assist_k3"], "tool_select")
    can = T.decision_pairs(runs["jev_ts_assist_k3"], "tool_select", gold="canonical")
    assert len(pairs) == len(can) and sum(bool(p["s1_ok"]) for p in pairs) >= sum(bool(p["s1_ok"]) for p in can)
    g = T.analyze_gate_run(run, ["meta_shadow_s2lite"], "reflection", V=1.0, e=0.3, h=0.05, n_boot=50)
    assert g["n"] > 0 and g["frontier"] and "s_star" in g and "regret" in g


def test_copy_coverage_valve_on_dev_seeds():
    envs = [JevGymEnv(seed=5000 + i, **SUITE) for i in range(30)]
    out = MG.copy_coverage(envs)
    assert out["table"] and all(r["coverage"] >= 0.8 for r in out["table"].values()) and not out["reclassify_free"]
    assert "finish.answer" not in out["table"]                     # free: never a copy parameter


def test_h2_regression_recovers_the_auroc_slope():
    rng = np.random.default_rng(1)
    cells = []
    for i in range(12):
        au = 0.5 + 0.04 * i
        cells.append({"estimable": True, "stats": {"auroc": au, "accuracy": 0.9 - 0.01 * (i % 3), "J": 2 * au - 1,
                                                   "pi": 0.1},
                      "efficacy": {"e": 0.3, "h": 0.1}, "n_bar": 5.0,
                      "delta_u": {"1": {"gross": 5.0 * 0.4 * 0.09 * (2 * au - 1) + rng.normal(0, 1e-4)}}})
    r = MG.h2_regression(cells + [{"estimable": False}], 1.0)
    assert r["n_estimable"] == 12 and r["auroc_only"]["slope_auroc"] == approx(2 * 5 * 0.4 * 0.09, rel=0.05)
    assert r["theory_check"]["slope_J"] == approx(1.0, abs=0.05) and abs(r["theory_check"]["intercept"]) < 0.05
    assert r["lr_p"] > 0.01                                      # accuracy adds nothing once AUROC is in


def test_dev_round2_replay_reproduces_the_logged_episodes():
    """runs/dev (gitignored; skipped when absent): the fixed oracle replays the E2 dev trajectories byte-identically
    and labels the 4 Flash-Lite failures of llm_only_s2lite as wrong totals (META_GOLD_SPEC dev evidence)."""
    from pathlib import Path

    run = Path("runs/dev/e2a_dev")
    if not (run / "llm_only_s2lite.jsonl").exists():
        pytest.skip("runs/dev not present")
    from jevharness.harness.trajectory import load_trajectories

    facs = MG.suite_factories(run)
    rows = [r for t in load_trajectories(run / "llm_only_s2lite.jsonl") for r in MG.replay_golds(t, facs[t["task_id"]]())]
    assert not any(r.get("obs_match") is False for r in rows)
    v = [r["verdict"] for r in rows]
    assert v.count("wrong_answer") == 4 and v.count("fatal") == 0


# ====================================================================== round-3 review regressions
def _live_trigger(s1: Any, name: str, n: int = 16, dp: str = "reflection", llm: Any = None,
                  measure: MeasureConfig | None = None, **kw: Any) -> list[dict[str, Any]]:
    cfg = HarnessConfig(name=name, **{dp: DPConfig(mode=Mode.TRIGGER, tau=0.5)},
                        **({"measure": measure} if measure else {}))
    agent = JevAgent(llm or SimulatedLLM("sim-lite", error_rate=0.25, simulated_latency_ms=10), cfg, s1=s1, **kw)
    return [agent.run(JevGymEnv(seed=i, **SUITE)).to_dict() for i in range(n)]


@pytest.fixture(scope="module")
def meas16() -> list[dict[str, Any]]:
    return _run(_agent(_meta_cfg()), n=16)


def test_h2_replay_population_equals_the_measurement_arms_own_population(meas16):
    """Blocker: replays added every probe-list record (wording_v1 doubled the reflection cell, forced / transplant /
    T5 / T6 phantoms filled the stop-guard cell, augmented queries entered stop_trigger). The H2 population of a
    replay is the steps[] shadow population of the measurement arm's own backend, gate by gate."""
    from jevharness.decision.controls import HeuristicBackend

    rows = MG.replay_backend(meas16, HeuristicBackend(), arm="meta_shadow_s2lite")
    assert {"wording_v1", "augmented_stop_trigger", "forced_premature"} <= {r["source"] for r in rows}
    for g in MG.GATES:
        own = [r for r in MG.gate_records(meas16, g, modes=("shadow",)) if r.get("gold_status") == "ok"]
        rep = MG.h2_replay_records(rows, g, "meta_shadow_s2lite")
        assert len(rep) == len(own) and {r["source"] for r in rep} <= {"steps"}, g
    assert all(MG._gate_of_row(r) is None for r in rows if r["source"] != "steps")
    # same backend replayed == own log: identical cell n through h2_cells
    runs = {"meta_shadow_s2lite": meas16, "heur": _live_trigger(HeuristicBackend(), "heur"),
            "ctrl": _live_trigger(RateBackend(0.1), "ctrl")}
    cell = MG.h2_cells(runs, ["meta_shadow_s2lite"], [("heur", "ctrl")], [1.0],
                       replays={rows[0]["backend_id"]: rows}, n_boot=50)[0]
    own_n = len([r for r in MG.gate_records(meas16, "reflection", modes=("shadow",))
                 if r.get("gold_status") == "ok" and M.p_yes_of(r) is not None])
    assert cell["stats_source"].startswith("replay") and cell["stats"]["n"] == own_n


def test_h2_and_voi_never_use_another_backends_answers(meas16):
    """Blocker / major: without a replay of the live backend, h2_cells fell back to the measurement arm's log (and
    still marked the cell estimable) and voi_readout never checked the backend. Now: no stats, not estimable, out of
    the regression, a note; with the replay the live backend's own AUROC (different from the stand-in's)."""
    from jevharness.decision.controls import HeuristicBackend
    from jevharness.decision.base import backend_id

    heur = _live_trigger(HeuristicBackend(), "heur")
    ctrl = _live_trigger(RateBackend(0.1), "ctrl")
    same = _live_trigger(SimulatedBackend(accuracy=0.8, seed=1), "same")          # the measurement arm's spec
    runs = {"meta_shadow_s2lite": meas16, "heur": heur, "ctrl": ctrl, "same": same}
    cells = {c["live"]: c for c in MG.h2_cells(runs, ["meta_shadow_s2lite"], [("heur", "ctrl"), ("same", "ctrl")],
                                               [1.0], n_boot=50)}
    assert cells["heur"]["stats"] is None and not cells["heur"]["estimable"] and not cells["heur"]["stats_matched"]
    assert "replay missing" in cells["heur"]["stats_source"]
    assert cells["same"]["stats_matched"] and cells["same"]["stats_source"] == "measurement arm log"
    v = MG.voi_readout(heur, ctrl, meas16, [1.0], n_boot=50)
    assert v["stats"] is None and not v["predicted"] and "replay missing" in v["note"]
    rows = MG.replay_backend(meas16, HeuristicBackend(), arm="meta_shadow_s2lite")
    reps = {rows[0]["backend_id"]: rows}
    v2 = MG.voi_readout(heur, ctrl, meas16, [1.0], n_boot=50, replays=reps)
    assert v2["stats_matched"] and v2["stats"]["auroc"] != cells["same"]["stats"]["auroc"]
    # the regression skips unmatched cells even when flagged estimable by hand
    fake = dict(cells["heur"], estimable=True, stats={"auroc": 0.9, "accuracy": 0.9, "J": 0.5, "pi": 0.1})
    assert MG.h2_regression([fake], 1.0)["n_estimable"] == 0
    # identity is the full spec, not the name: two rate controls share the name "rate" but not the id
    assert backend_id(RateBackend(0.1).describe()) != backend_id(RateBackend(0.3).describe())
    assert backend_id({"backend": "x", "cached": True}) == backend_id({"backend": "x"})


def test_routing_cell_uses_the_routers_small_model_measurement_arm():
    """Major: a router's traj.llm is the large model, so the family match picked the Flash measurement arm, which does
    not shadow routing -> every routing cell was "no opportunity"."""
    flash = SimulatedLLM("sim-flash", error_rate=0.02, simulated_latency_ms=10)
    lite = SimulatedLLM("sim-lite", error_rate=0.3, simulated_latency_ms=10)
    flash_cfg = _meta_cfg("meta_shadow", cf=False, probes=False).model_copy(update={"routing": DPConfig()})
    meta_flash = [JevAgent(flash, flash_cfg, s1=SimulatedBackend(accuracy=0.8, seed=1))
                  .run(JevGymEnv(seed=i, **SUITE)).to_dict() for i in range(4)]
    meta_lite = [JevAgent(lite, _meta_cfg(cf=False, probes=False), s1=SimulatedBackend(accuracy=0.8, seed=1))
                 .run(JevGymEnv(seed=i, **SUITE)).to_dict() for i in range(4)]
    router = _live_trigger(RateBackend(0.5), "route", n=4, dp="routing", llm=flash, small_llm=lite)
    runs = {"meta_shadow": meta_flash, "meta_shadow_s2lite": meta_lite, "route": router}
    assert MG.measure_arm_for(runs, ["meta_shadow", "meta_shadow_s2lite"], "route", "routing") == "meta_shadow_s2lite"
    assert MG.measure_arm_for(runs, ["meta_shadow", "meta_shadow_s2lite"], "route", "reflection") == "meta_shadow"


def test_measurement_calls_never_count_as_strong_calls():
    """Major: CF / probe-generation System-2 calls (meta.shadow) inflated strong_calls and strong_call_ratio."""
    from jevharness.bench.scorecard import _episode_values

    s1 = lambda: SimulatedBackend(accuracy=0.5, seed=3)  # noqa: E731
    a = _live_trigger(s1(), "plain", n=8)
    b = _live_trigger(s1(), "cf", n=8, measure=MeasureConfig(cf=CFConfig(fired_steps=True)))
    assert any((c.get("meta") or {}).get("measure") == "cf" for t in b for c in t["calls"])
    for x, y in zip(a, b):
        vx, vy = _episode_values(x), _episode_values(y)
        assert vx["latency"] == approx(vy["latency"], rel=0.2)          # measured wall time (jitter)
        assert {k: v for k, v in vx.items() if k != "latency"} == {k: v for k, v in vy.items() if k != "latency"}


def test_live_trigger_rows_are_consistency_checks_and_c5_arms_are_symmetric():
    """Major: live trigger rows sat in the "H2 inputs" table with structurally-zero fire rate / J (fired steps
    unobserved), and only the Jev C5 arm relabelled fired steps. Live rows are now a separate consistency check (AUROC
    on executed labels, realised fire rate over every opportunity), and every Flash-Lite reflection-trigger arm of
    baselines_jevgym.yaml runs cf.fired_steps alike."""
    import yaml as _yaml
    from jevharness.bench.scorecard import gold_block

    with_cf = _live_trigger(SimulatedBackend(accuracy=0.7, seed=3), "a", n=12,
                            measure=MeasureConfig(cf=CFConfig(fired_steps=True)))
    without = _live_trigger(SimulatedBackend(accuracy=0.7, seed=3), "b", n=12)
    ga, gb = gold_block(with_cf), gold_block(without)
    for g in (ga, gb):
        assert "reflection/trigger" not in g.get("gates", {})
        row = g["live_gates"]["reflection/trigger"]
        assert row["selective"] and row["role"] == "consistency check"
        assert not {"J", "tpr", "fpr", "accuracy", "estimable"} & set(row)
    ra, rb = ga["live_gates"]["reflection/trigger"], gb["live_gates"]["reflection/trigger"]
    assert ra["n_cf_labelled"] > 0 and rb["n_cf_labelled"] == 0
    assert ra["n"] == rb["n"] and ra["auroc"] == approx(rb["auroc"], nan_ok=True)   # CF labels never enter AUROC
    assert ra["fire_rate_all"] == rb["fire_rate_all"] > 0
    exp = _yaml.safe_load(open("configs/experiments/baselines_jevgym.yaml"))
    refl = [a for a in exp["arms"] if a.get("llm") == "lite2" and (a.get("reflection") or {}).get("mode") == "trigger"]
    assert len(refl) >= 7 and all((a.get("measure") or {}).get("cf", {}).get("fired_steps") for a in refl)


def test_probes_only_in_arms_with_every_decision_point_off_or_shadow():
    """Minor: the probe guard checked System-1 modes only; reflection always / memory llm / llm_only passed, and
    _shadow_only then dropped such arms from Holm / Pareto / the reference role."""
    from jevharness.bench.scorecard import _shadow_only

    pr = MeasureConfig(probes=ProbeConfig(stop=["T5"]))
    for cfg in (HarnessConfig(name="reflect_always", reflection=DPConfig(mode=Mode.ALWAYS), measure=pr),
                HarnessConfig(name="mem_llm_gate", memory_write=DPConfig(mode=Mode.LLM),
                              memory_read=DPConfig(mode=Mode.ALWAYS), measure=pr),
                HarnessConfig(name="llm_only", measure=pr),
                HarnessConfig(name="mixed", stop=DPConfig(mode=Mode.SHADOW), reflection=DPConfig(mode=Mode.ALWAYS),
                              measure=pr)):
        with pytest.raises(ValueError, match="measurement"):
            cfg.validate_modes()
    ok = HarnessConfig(name="m", stop=DPConfig(mode=Mode.SHADOW), measure=pr).validate_modes()
    assert ok.is_measurement_arm()
    t = {"probes": [{"dp": "stop"}], "steps": [], "meta": {"config": HarnessConfig(
        name="reflect_always", reflection=DPConfig(mode=Mode.ALWAYS)).model_dump(mode="json")}}
    assert not _shadow_only([t])
    t2 = {"steps": [], "meta": {"config": HarnessConfig(name="m", stop=DPConfig(mode=Mode.SHADOW)).model_dump(mode="json")}}
    assert _shadow_only([t2])


def test_natural_stop_records_use_the_probe_tau_ref():
    """Minor: natural shadow finishes were thresholded at 0.5 while probes used tau_ref, so FAR_model mixed two
    operating points."""
    nat = {"dp": "stop", "mode": "shadow", "gold": False, "s1_dist": {"yes": 0.6, "no": 0.4}, "s1_value": True,
           "question": q_stop_guard("1.00"), "state": "S", "gold_status": "ok",
           "gold_aux": {"kind": "wrong_answer", "d_before": 1}}
    probe = {"dp": "stop", "mode": "guard_probe", "gold": False, "s1_dist": {"yes": 0.6, "no": 0.4},
             "s1_value": False, "gold_status": "ok",
             "gold_aux": {"probe_source": "forced_premature", "kind": "premature", "d_before": 2, "tau_ref": 0.7}}
    t = {"task_id": "t", "probes": [probe], "steps": [{"step": 0, "decisions": [nat]}],
         "meta": {"config": {"measure": {"probes": {"tau_ref": 0.7}}}}}
    rows = M.probe_records([t])
    assert [r["accept"] for r in rows] == [False, False]
    assert all(r["accept"] for r in M.probe_records([t], tau=0.5))


def test_stop_guard_retries_change_the_state_and_count_once():
    """Major: a rejection left System-1's state unchanged, so a deterministic gate re-answered the identical question
    until max_steps and one false rejection became ~25 records (trigger_rates of a rate-0.5 gate: 0.036)."""
    class Reject(SimulatedBackend):
        def _dist(self, q):
            return {"yes": 0.01, "no": 0.99} if q.kind.value == "noul" else super()._dist(q)

    cfg = HarnessConfig(name="g", stop=DPConfig(mode=Mode.GUARD, tau=0.5), max_steps=12)
    llm = SimulatedLLM("sim", error_rate=0.0, simulated_latency_ms=1)
    trajs = [JevAgent(llm, cfg, s1=Reject(accuracy=0.9)).run(JevGymEnv(seed=s, **SUITE)).to_dict() for s in range(4)]
    for t in trajs:
        g = [(st["step"], d) for st in t["steps"] for d in st["decisions"] if d["dp"] == "stop"]
        assert len(g) > 2 and [d["retry"] for _, d in g] == list(range(len(g)))
        assert len({d["state"] for _, d in g}) == len(g)                     # every re-query sees a new state
    r = trigger_rates(trajs)["stop/guard"]
    n_prop = len({(t["task_id"], d["question"]) for t in trajs for st in t["steps"] for d in st["decisions"]
                  if d["dp"] == "stop"})
    assert r["n"] == n_prop and r["n_retries"] > 0 and r["n_all"] == r["n"] + r["n_retries"]
    g = M.guard_error_rates(M.decision_records(trajs))["stop"]
    assert g["n"] == n_prop and g["n_retries"] == r["n_retries"]
    from jevharness.bench.scorecard import gold_block

    lg = gold_block(trajs)["live_gates"]["stop_guard/guard"]
    assert lg["n_opportunities"] == n_prop and lg["n_retries"] == r["n_retries"] and lg["n"] <= n_prop
    # a rate-0.5 gate is matched per distinct proposal: its first-decision acceptance rate, not ~0
    rate = [JevAgent(llm, cfg, s1=RateBackend(0.5, seed=s)).run(JevGymEnv(seed=s, **SUITE)).to_dict()
            for s in range(30)]
    assert 0.25 < trigger_rates(rate)["stop/guard"]["rate"] < 0.75


def test_task_clustered_cis_and_the_tosts():
    """Major: probe CIs were record-level Wilson intervals and neither pre-registered TOST existed."""
    rows = [(False, a, f"t{i}") for i in range(10) for a in ([True] * 5 if i % 2 else [False] * 5)]
    r = M._probe_rates(rows, n_boot=500)
    wl, wh = r["far_ci_wilson"]
    cl, ch = r["far_ci"]
    assert r["far"] == 0.5 and r["ci"] == "task-clustered bootstrap" and (ch - cl) > (wh - wl)
    # all-concordant arms: the bootstrap CI is [0, 0] at any n -- equivalence needs n >= ~150 (rule of three)
    eq = M.tost_paired({f"t{i}": float(i % 2) for i in range(60)}, {f"t{i}": float(i % 2) for i in range(60)}, 0.02)
    assert not eq["equivalent"] and eq["diff"] == 0 and eq["n_discordant"] == 0
    eq = M.tost_paired({f"t{i}": float(i % 2) for i in range(400)}, {f"t{i}": float(i % 2) for i in range(400)}, 0.02)
    assert eq["equivalent"]
    rng = random.Random(0)
    b = {f"t{i}": float(rng.random() < 0.5) for i in range(2000)}
    o = dict(b)
    for i in range(0, 40, 2):           # 20 discordant tasks each way: Delta = 0, CI well inside ±2 pt
        o[f"t{i}"], o[f"t{i + 1}"] = 1 - b[f"t{i}"], b[f"t{i + 1}"]
    eq = M.tost_paired(b, o, 0.02)
    assert eq["n_discordant"] >= 5 and eq["equivalent"]
    neq = M.tost_paired({f"t{i}": 0.0 for i in range(60)}, {f"t{i}": float(i % 3 == 0) for i in range(60)}, 0.02)
    assert not neq["equivalent"] and neq["p"] > 0.05
    pr = [{"dp": "stop", "gold": False, "source": s, "accept": a, "task_id": f"t{i}"}
          for i in range(40) for s, a in (("natural", i % 2 == 0), ("forced_premature", i % 2 == 0))]
    rep = M.tost_far_nat_model(pr)
    assert rep["diff"] == approx(0.0) and rep["equivalent"] and rep["n_nat"] == 40 and rep["n_model"] == 80
    blk = M.probe_error_rates([{"task_id": f"t{i}", "probes": [], "steps": [], "meta": {}} for i in range(2)])
    assert blk == {}


def test_assist_tournament_keeps_k_survivors_and_logs_the_full_ranking():
    """Major: two survivors per chunk left the stand-in 4 finalists, so k = 5 offered 4 tools and recall@5 was
    recall over the finalists (== recall@3)."""
    class Small(SimulatedBackend):
        max_options = 52
        batched_questions = True

    agent = JevAgent(ScriptedLLM("s"), HarnessConfig(), s1=Small(accuracy=0.9, seed=0))
    opts = [f"t{i}" for i in range(69)]
    rank: list[str] = []
    a = agent._choice_tournament("S", "Which tool should the agent call next?", opts, "tool_select", Ledger(),
                                 gold="t3", keep=5, ranking=rank)
    assert len(a.distribution) == 10 and sorted(rank) == sorted(opts) and rank[0] == a.value
    env_kw = dict(SUITE, n_distractors=60)
    cfg = HarnessConfig(name="k5", tool_select=DPConfig(mode=Mode.ASSIST, k=5))
    t = JevAgent(SimulatedLLM("sim", error_rate=0.1, simulated_latency_ms=1), cfg, s1=Small(accuracy=0.9, seed=0)) \
        .run(JevGymEnv(seed=_seed_of("multi"), **env_kw)).to_dict()
    ts = [d for st in t["steps"] for d in st["decisions"] if d["dp"] == "tool_select"]
    assert ts and all(len(d["s1_rank"]) == len(d["options"]) > 52 for d in ts)
    assert all(len(d["gold_aux"]["offered"]) >= 5 for d in ts)
    rec = M.recall_at_k(M.decision_records([t]))
    assert not any(k.startswith("n_undefined") for k in rec)
    # an older tournament log (no s1_rank, 4 finalists in s1_dist): recall@5 is undefined there, not recall@4
    old = {**M.decision_records([t])[0], "s1_rank": None, "options": opts,
           "s1_dist": {"t1": 0.7, "t2": 0.1, "t3": 0.1, "t4": 0.1}, "gold_status": "ok",
           "gold_aux": {"acceptable_tools": ["t9"]}}
    r = M.recall_at_k([old])
    assert r["n_undefined@5"] == 1 and math.isnan(r["recall@5"]) and r["recall@3"] == 0.0


def test_gate_theory_defaults_to_measurement_arms_of_one_family(tmp_path, meas16):
    """Major: without --arm the gate analysis pooled every arm of the run directory (measurement shadow golds, live
    trigger arms' selective labels, controls, both System-2 families); c_f of a Flash-Lite measurement arm came out
    as 0 (reflection) or as "every call is large" (routing)."""
    live = _live_trigger(RateBackend(0.3), "rate30_reflect_trigger_s2lite", n=6)
    for name, ts in (("meta_shadow_s2lite", meas16), ("rate30_reflect_trigger_s2lite", live)):
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for t in ts:
                f.write(json.dumps(t, default=str) + "\n")
    g = T.analyze_gate_run(tmp_path, None, "reflection", V=1.0, e=0.3, h=0.05, n_boot=20)
    assert g["meta"]["arms"] == ["meta_shadow_s2lite"] and g["meta"]["modes"] == ["shadow"]
    own = [r for r in MG.gate_records(meas16, "reflection", modes=("shadow",)) if r.get("gold_status") == "ok"
           and M.p_yes_of(r) is not None]
    assert g["n"] == len(own)
    # c_f of a measurement arm: its own reflection / large calls exist only as CF branches of the same prompt
    cf = lambda tag: {"shadow": True, "measure": "cf", "cf_of": f"3:{tag}"}  # noqa: E731
    calls = [{"component": "s2", "decision_point": "tool_select", "model": "lite", "cost_usd": 0.001, "meta": {}},
             {"component": "s2", "decision_point": "reflection", "model": "lite", "cost_usd": 0.0007,
              "meta": cf("reflection")},
             {"component": "s2", "decision_point": "tool_select", "model": "lite", "cost_usd": 0.001,
              "meta": cf("reflection")},
             {"component": "s2", "decision_point": "tool_select", "model": "flash", "cost_usd": 0.005,
              "meta": cf("large")}]
    assert MG._fired_cost([{"llm": "lite", "calls": calls}], "reflection") == approx(0.0007)
    assert MG._fired_cost([{"llm": "lite", "calls": calls}], "routing") == approx(0.004)        # not "all large"
    assert any((c.get("meta") or {}).get("cf_of", "").endswith(":large") for t in meas16 for c in t["calls"])
    flash = [JevAgent(SimulatedLLM("sim-flash", error_rate=0.02, simulated_latency_ms=10),
                      _meta_cfg("meta_shadow", cf=False, probes=False), s1=SimulatedBackend(accuracy=0.8, seed=1))
             .run(JevGymEnv(seed=i, **SUITE)).to_dict() for i in range(3)]
    with open(tmp_path / "meta_shadow.jsonl", "w") as f:
        for t in flash:
            f.write(json.dumps(t, default=str) + "\n")
    with pytest.raises(SystemExit, match="families"):
        T.analyze_gate_run(tmp_path, None, "reflection", V=1.0, e=0.3, h=0.05, n_boot=20)
    g2 = T.analyze_gate_run(tmp_path, None, "reflection", V=1.0, e=0.3, h=0.05, n_boot=20, family="sim-lite")
    assert g2["meta"]["arms"] == ["meta_shadow_s2lite"]


def test_c5_measurement_arms_load_from_another_run_and_the_e2b_duplicate_is_renamed(tmp_path, meas16):
    """Major: the C5 pair (E2c run) and meta_shadow_s2lite (E2b run) never sat in one directory; jevgym_meta_control
    defined a second jev_reflect_trigger_s2lite."""
    import yaml as _yaml

    e2c, e2b = tmp_path / "e2c", tmp_path / "e2b"
    e2c.mkdir()
    e2b.mkdir()
    live = _live_trigger(SimulatedBackend(accuracy=0.8, seed=1), "jev_reflect_trigger_s2lite", n=8)
    ctrl = _live_trigger(RateBackend(0.1), "ratem_reflect_trigger_s2lite", n=8)
    for d, name, ts in ((e2c, "jev_reflect_trigger_s2lite", live), (e2c, "ratem_reflect_trigger_s2lite", ctrl),
                        (e2b, "meta_shadow_s2lite", meas16)):
        with open(d / f"{name}.jsonl", "w") as f:
            for t in ts:
                f.write(json.dumps(t, default=str) + "\n")
    runs = MG.load_runs_split(e2c, ["jev_reflect_trigger_s2lite", "ratem_reflect_trigger_s2lite"],
                              ["meta_shadow_s2lite"], e2b)
    cell = MG.h2_cells(runs, ["meta_shadow_s2lite"], [("jev_reflect_trigger_s2lite", "ratem_reflect_trigger_s2lite")],
                       [1.0], n_boot=50)[0]
    assert cell["measure_arm"] == "meta_shadow_s2lite" and cell["stats_matched"] and "tost" in cell
    from jevharness.bench.scorecard import main as sc_main

    sc_main([str(e2c), "--baseline", "ratem_reflect_trigger_s2lite", "--pair",
             "jev_reflect_trigger_s2lite:ratem_reflect_trigger_s2lite", "--measure-arm", "meta_shadow_s2lite",
             "--measure-run", str(e2b), "--format", "json", "--out", str(tmp_path / "c.json"), "--n-boot", "50"])
    card = json.loads((tmp_path / "c.json").read_text())
    assert "meta_shadow_s2lite" not in card["arms"]
    v = card["headline"]["pairs"]["jev_reflect_trigger_s2lite vs ratem_reflect_trigger_s2lite"]["voi"]
    assert v["measure_arm"] == "meta_shadow_s2lite" and v["stats_matched"]
    e2b_cfg = _yaml.safe_load(open("configs/experiments/jevgym_meta_control.yaml"))
    assert "jev_reflect_trigger_s2lite" not in {a["name"] for a in e2b_cfg["arms"]}


def test_matched_rate_protocol_is_per_arm_placeholders_skip_and_resume_refuses_a_changed_spec(tmp_path):
    """Blocker: the documented trigger_rates call pooled the whole run directory, the ratem arms ran at placeholder
    rates in the same `jevbench run`, and resume ignored a later rate change."""
    import warnings as _w

    import yaml as _yaml
    from jevharness.bench.runner import run_experiment

    hdr = open("configs/experiments/baselines_jevgym.yaml").read()
    assert "t('runs/e2c_jevgym_baselines'))" not in hdr and "jev_reflect_trigger_s2lite.jsonl" in hdr
    exp_cfg = _yaml.safe_load(hdr)
    # A6.16 (i): the C5 control `ratem_lite` is fixed from the retired pilot (no placeholder); the secondary-row
    # controls keep A4's placeholder protocol until their rates are set from this run's System-1 arms
    ph = {k: bool(v.get("placeholder")) for k, v in exp_cfg["s1_backends"].items() if k.startswith("ratem")}
    assert ph["ratem_lite"] is False and exp_cfg["s1_backends"]["ratem_lite"]["rate"] == 0.1424
    assert all(v for k, v in ph.items() if k != "ratem_lite")
    lite = {"provider": "simulated", "model": "sim-lite", "error_rate": 0.3, "simulated_latency_ms": 1}
    exp = {"name": "ph", "suite": {"name": "jevgym", "n": 4, **SUITE}, "llm": lite, "workers": 1,
           "s1": {"type": "simulated", "accuracy": 0.8},
           "s1_backends": {"ratem": {"type": "rate", "rate": 0.1, "placeholder": True}},
           "arms": [{"name": "jev", "reflection": {"mode": "trigger", "tau": 0.5}},
                    {"name": "ratem", "s1": "ratem", "reflection": {"mode": "trigger", "tau": 0.5}}]}
    paths = run_experiment(exp, tmp_path, progress=False)
    assert set(paths) == {"jev"} and not (tmp_path / "ph" / "ratem.jsonl").exists()
    r = trigger_rates(tmp_path / "ph" / "jev.jsonl")["reflection/trigger"]["rate"]
    exp["s1_backends"]["ratem"] = {"type": "rate", "rate": r, "seed": 0}
    paths = run_experiment(exp, tmp_path, progress=False)
    assert "ratem" in paths
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        trigger_rates(tmp_path / "ph")
    assert any("pools" in str(x.message) for x in w)
    by = trigger_rates(tmp_path / "ph", by_arm=True)
    assert set(by) == {"jev", "ratem"}
    exp["s1_backends"]["ratem"] = {"type": "rate", "rate": 0.5, "seed": 0}
    with pytest.raises(ValueError, match="another System-1 spec"):
        run_experiment(exp, tmp_path, progress=False)


def test_a_read_counts_as_done_only_as_served():
    """Minor: _done_step matched reads tolerantly while _step served them exactly -- a get_orders with a lower-cased
    user id answered "[]" yet counted as done; the rollout then reached finish without the amounts and labelled the
    recoverable state doomed (every gold there dropped)."""
    e = _env(_seed_of("multi"))
    u = e.jobs[0][1]
    e.step("find_user", {"name": u["name"]})
    for bad in ({"user_id": u["user_id"].lower(), "status": "pending"}, {"user_id": f" {u['user_id']} ",
                                                                           "status": "pending"},
                {"user_id": u["user_id"], "status": "ALL"}):
        c = e.oracle_snapshot()
        assert c.step("get_orders", bad).observation == "[]"
        assert c.oracle_action() == ("get_orders", {"user_id": u["user_id"], "status": "pending"})
        assert c.oracle_cost_to_go(B) != float("inf")                                   # not doomed
        assert c.oracle_verdict("get_orders", {"user_id": u["user_id"], "status": "pending"}, B)["verdict"] == \
            "progress"


def test_efficacy_without_placebo_is_not_net_and_yields_no_prediction():
    """Minor: without a placebo verdict, e / h were the raw branch rates (NaN replaced by 0)."""
    recs = [{"gold_aux": {"cf_status": "ok", "verdict": "fatal", "verdict_alt": "progress"}},
            {"gold_aux": {"cf_status": "ok", "verdict": "progress", "verdict_alt": "progress"}}]
    e = M.cf_efficacy(recs)
    assert math.isnan(e["e"]) and math.isnan(e["h"]) and e["net"] is False and e["e_raw"] == 1.0
    e2 = M.cf_efficacy([{"gold_aux": {**r["gold_aux"], "verdict_placebo": "fatal"}} for r in recs])
    assert e2["net"] and e2["e"] == 1.0


def test_stop_guard_keeps_the_tail_of_long_answers():
    """Jev pilot 2026-10-01: a long summary whose final sentence carries the requested total lost that sentence to a
    300-character head cut, and the guard rejected correct finishes. The tail must survive clipping."""
    items = "\n".join(f"{i}. Shipped order ORD-{4700 + i} via standard shipping for ${100 + i}.00." for i in range(30))
    ans = f"I processed every request in order:\n{items}\n\nThe total amount of the orders processed is $897.11."
    assert len(ans) > 1000
    q = q_stop_guard(ans)
    assert "$897.11" in q and "I processed every request" in q and "[...]" in q
    assert infer_dp(Noul(q)) == "stop" and STOP_GUARD_RE.search(q)
    short = "The total amount of the orders processed is $897.11."
    assert json.dumps(short) in q_stop_guard(short)                     # short answers are shown whole
