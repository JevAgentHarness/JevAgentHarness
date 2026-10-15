"""E6d (exploratory; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6d", results/explore/e6d_modes/cells.md):
"advise" assists at the four yes / no points (memory write, reflection, stop, routing), guards at reflection and
routing, and the memory-read guard. Offline, with the stubs of tests/test_e6b_modes.py; plus the JevGym simulation of
every new cell, the wire canary, and the fingerprint check that every mode added by E6b and E6c behaves and calls
exactly as on the pre-E6d code (``tests/e6b_fingerprint.py e6``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e6b_fingerprint as FP  # noqa: E402
import test_e6b_modes as B  # noqa: E402

from jevharness.bench.metrics import decision_records  # noqa: E402
from jevharness.bench.suites.jevgym import JevGymEnv  # noqa: E402
from jevharness.decision.simulated import SimulatedBackend  # noqa: E402
from jevharness.harness.agent import JevAgent  # noqa: E402
from jevharness.harness.config import (DP, E6D_CELLS, EQUIVALENT_NOTES, UNDEFINED_CELLS, VALID_MODES,  # noqa: E402
                                       DPConfig, HarnessConfig, Mode)
from jevharness.harness.prompts import MEMORY_REJECT_NOTE, MEMORY_SELECT_PROMPT, REFLECT_PROMPT  # noqa: E402
from jevharness.llm.base import ChatResponse  # noqa: E402
from jevharness.llm.simulated import SimulatedLLM  # noqa: E402


def test_e6b_and_e6c_modes_behave_and_call_exactly_as_before() -> None:
    exp = json.loads(FP.FIXTURE_E6.read_text())["digests"]
    got = FP.prior_e6_digests()
    assert set(got) == set(exp) and {k for k in exp if got[k] != exp[k]} == set()


def test_grid_after_e6d_three_undefined() -> None:
    assert E6D_CELLS <= {(dp, m) for dp, ms in VALID_MODES.items() for m in ms}
    # E6d left three cells undefined (the triggers at tool choice, tool arguments and action); E6f defined them as
    # verification triggers, so none is left (tests/test_e6f_modes.py)
    undef = sorted((dp.value, m.value) for (dp, m), (k, _) in UNDEFINED_CELLS.items() if k == "undefined")
    assert undef == []
    # the five cells E6d left "equivalent" are implemented since E6e (their equal cell: EQUIVALENT_NOTES)
    assert sum(k == "equivalent" for k, _ in UNDEFINED_CELLS.values()) == 0 and len(EQUIVALENT_NOTES) == 5


class S2(B.S2Stub):
    """The E6b System-2 stub, plus memory selections: ``select`` (then ``reselect`` after a rejection note)."""

    def __init__(self, select: str = "none", reselect: str | None = None, **kw: Any):
        super().__init__(**kw)
        self.select, self.reselect = select, reselect

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        last = str(messages[-1].get("content") or "")
        if not tools and last.endswith(MEMORY_SELECT_PROMPT):
            self.requests.append({"tools": [], "specs": None, "tool_choice": None, "last": last, "kind": "select"})
            ans = self.reselect if (MEMORY_REJECT_NOTE in last and self.reselect is not None) else self.select
            return ChatResponse(text=ans, input_tokens=60, output_tokens=3, latency_ms=2.0)
        return super()._chat(messages, tools, **kw)


def _gates(llm: B.S2Stub) -> list[dict[str, Any]]:
    return [r for r in llm.requests if r["last"].endswith("Answer yes or no.")]


# ------------------------------------------------------------------ assists ("advise")
def test_memory_write_assist_advises_the_llm_gate_which_decides() -> None:
    cfg = HarnessConfig(name="mw_assist", memory_write=DPConfig(mode=Mode.ASSIST), memory_read=DPConfig(mode=Mode.ALWAYS))
    t, s1, llm = B._run(cfg, "lookup_cancel", B.S1Stub({"memory_write": 0.83}, yes={"memory_write": False}),
                        S2(gate="yes"))
    w = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_write"]
    assert len(w) == 2 and all(d["mode"] == "assist" and d["s1_value"] is False and d["s2_value"] is True
                               and d["final_value"] is True for d in w)           # the LLM overrode the advice
    g = _gates(llm)
    assert len(g) == 2 and "[Advisor] A fast decision model answered the question below: no (confidence 0.83)" in g[0]["last"]
    assert g[0]["last"].index("[Advisor]") < g[0]["last"].index("Should this be stored")
    assert t["meta"]["memory"]["stored"] == 2 and t["steps"][0]["llm_calls"] == 2
    assert len([q for q in s1.requests if q.decision_point == "memory_write"]) == 2


def test_reflection_and_routing_assist_put_the_advice_in_their_gates() -> None:
    cfg = HarnessConfig(name="rf_assist", reflection=DPConfig(mode=Mode.ASSIST))
    t, _, llm = B._run(cfg, "lookup_cancel", B.S1Stub({"reflection": 0.9}, yes={"reflection": True}), S2(gate="no"))
    rf = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "reflection"]
    assert len(rf) == 2 and all(d["s1_value"] is True and d["final_value"] is False for d in rf)
    assert all("answered the question below: yes (confidence 0.90)" in r["last"] for r in _gates(llm))
    assert not any(REFLECT_PROMPT in r["last"] for r in llm.requests)
    big, small = S2(model="big"), S2(model="small", gate="yes")
    cfg = HarnessConfig(name="rt_assist", routing=DPConfig(mode=Mode.ASSIST))
    t, _, _ = B._run(cfg, "cancel", B.S1Stub(yes={"routing": False}), big, small)
    rt = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "routing"]
    assert all(d["mode"] == "assist" and d["s1_value"] is False and d["final_value"] == "large" for d in rt)
    assert len(_gates(small)) == 2 and not _gates(big) and "[Advisor]" in _gates(small)[0]["last"]


def test_stop_assist_puts_the_advice_in_the_llm_turn_and_never_overrides() -> None:
    cfg = HarnessConfig(name="st_assist", stop=DPConfig(mode=Mode.ASSIST))
    t, s1, llm = B._run(cfg, "cancel", B.S1Stub({"stop": 0.97}, yes={"stop": True}))
    st = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "stop"]
    assert [d["s2_value"] for d in st] == [False, True] and all(d["final_value"] == d["s2_value"] for d in st)
    assert all(d["mode"] == "assist" and d["s1_value"] is True for d in st)
    calls = [r for r in llm.requests if r["kind"] == "call"]
    assert all('"Has the task already been fully completed' in r["last"] and "yes (confidence 0.97)" in r["last"]
               for r in calls)
    assert t["steps"][0]["tool"] == "cancel_order" and t["result"]["success"]   # the LLM decided, not the advice
    assert len([q for q in s1.requests if q.decision_point == "stop"]) == 2


# ------------------------------------------------------------------ guards
def test_reflection_guard_checks_only_proposed_reflections_and_vetoes_below_tau() -> None:
    cfg = HarnessConfig(name="rf_guard", reflection=DPConfig(mode=Mode.GUARD, tau=0.5))
    t, s1, llm = B._run(cfg, "lookup_cancel", B.S1Stub(yes={"reflection": False}), S2(gate="yes"))
    rf = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "reflection"]
    assert len(rf) == 2 and all(d["s2_value"] is True and d["s1_value"] is False and d["final_value"] is False
                                for d in rf)
    assert not any(REFLECT_PROMPT in r["last"] for r in llm.requests)              # both proposals vetoed
    assert not any("[Advisor]" in r["last"] for r in _gates(llm))                  # a guard gives no advice
    t, s1, llm = B._run(cfg, "lookup_cancel", B.S1Stub(yes={"reflection": False}), S2(gate="no"))
    assert not [q for q in s1.requests if q.decision_point == "reflection"]        # nothing proposed, nothing asked
    t, _, llm = B._run(cfg, "lookup_cancel", B.S1Stub(yes={"reflection": True}), S2(gate="yes"))
    assert sum(REFLECT_PROMPT in r["last"] for r in llm.requests) == 2              # accepted proposals reflect


def test_routing_guard_vetoes_an_escalation() -> None:
    cfg = HarnessConfig(name="rt_guard", routing=DPConfig(mode=Mode.GUARD, tau=0.5))
    big, small = S2(model="big"), S2(model="small", gate="yes")
    t, _, _ = B._run(cfg, "cancel", B.S1Stub(yes={"routing": False}), big, small)
    rt = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "routing"]
    assert all(d["s2_value"] == "large" and d["s1_value"] is False and d["final_value"] == "small" for d in rt)
    assert {c["model"] for c in t["calls"] if c["component"] == "s2"} == {"small"}


def test_memory_read_guard_lets_the_llm_select_and_vetoes_once() -> None:
    cfg = HarnessConfig(name="mr_guard", memory_write=DPConfig(mode=Mode.ALWAYS),
                        memory_read=DPConfig(mode=Mode.GUARD, tau=0.5))
    # accepted: the LLM's selection (m0) is injected
    t, s1, llm = B._run(cfg, "lookup_cancel", B.S1Stub(yes={"memory_read": True}), S2(select="m0, m99"))
    r = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_read"]
    assert len(r) == 2 and all(d["mode"] == "guard" and d["s1_value"] is True for d in r)
    assert all(d["s2_value"] == d["final_value"] and len(d["final_value"]) == 1 for d in r)   # m99 does not exist
    assert len([x for x in llm.requests if x["kind"] == "select"]) == 2 and t["steps"][1]["llm_calls"] == 2
    assert all(r_["last"].startswith("[Memory]\n- [m0]") for r_ in llm.requests if r_["kind"] == "call"
               and r_["last"].startswith("[Memory]"))
    # vetoed: the LLM selects once more with the note, and that selection is injected unchecked
    t, s1, llm = B._run(cfg, "lookup_cancel", B.S1Stub(yes={"memory_read": False}), S2(select="m0", reselect="none"))
    r = [d for s in t["steps"] for d in s["decisions"] if d["dp"] == "memory_read"]
    assert all(d["s1_value"] is False and d["gold_aux"]["veto"] and d["final_value"] == [] for d in r)
    assert len([x for x in llm.requests if x["kind"] == "select"]) == 4
    assert len([q for q in s1.requests if q.decision_point == "memory_read"]) == 2
    assert "STORED MEMORIES:" in [x for x in llm.requests if x["kind"] == "select"][0]["last"]


# ------------------------------------------------------------------ JevGym simulation, wire canary
E6D_CONFIGS = {
    "mw_assist": HarnessConfig(name="mw_assist", memory_write=DPConfig(mode=Mode.ASSIST),
                               memory_read=DPConfig(mode=Mode.ALWAYS)),
    "rf_assist": HarnessConfig(name="rf_assist", reflection=DPConfig(mode=Mode.ASSIST)),
    "st_assist": HarnessConfig(name="st_assist", stop=DPConfig(mode=Mode.ASSIST)),
    "rt_assist": HarnessConfig(name="rt_assist", routing=DPConfig(mode=Mode.ASSIST)),
    "rf_guard": HarnessConfig(name="rf_guard", reflection=DPConfig(mode=Mode.GUARD, tau=0.5)),
    "rt_guard": HarnessConfig(name="rt_guard", routing=DPConfig(mode=Mode.GUARD, tau=0.5)),
    "mr_guard": HarnessConfig(name="mr_guard", memory_write=DPConfig(mode=Mode.ALWAYS),
                              memory_read=DPConfig(mode=Mode.GUARD, tau=0.5)),
}


class SimGate(SimulatedLLM):
    """SimulatedLLM answering gate prompts (alternating yes / no) and memory selections (the two newest ids)."""

    def _chat(self, messages, tools=None, **kw):  # type: ignore[no-untyped-def]
        last = str(messages[-1].get("content") or "")
        if not tools and last.rstrip().endswith("Answer yes or no."):
            self._n = getattr(self, "_n", 0) + 1
            return ChatResponse(text="yes" if self._n % 2 else "no", input_tokens=40, output_tokens=1, latency_ms=1.0)
        if not tools and last.endswith(MEMORY_SELECT_PROMPT):
            import re
            ids = re.findall(r"\[m(\d+)\]", last)[-2:]
            return ChatResponse(text=", ".join(f"m{i}" for i in ids) or "none", input_tokens=80, output_tokens=4,
                                latency_ms=1.0)
        return super()._chat(messages, tools, **kw)


@pytest.mark.parametrize("name", sorted(E6D_CONFIGS))
def test_every_e6d_cell_runs_on_jevgym_with_golds_and_one_ledger(name: str) -> None:
    cfg = E6D_CONFIGS[name]
    agent = JevAgent(SimGate("scripted", error_rate=0.1, simulated_latency_ms=5, seed=0), cfg,
                     s1=SimulatedBackend(accuracy=0.8, seed=1),
                     small_llm=SimGate("scripted-small", error_rate=0.3, simulated_latency_ms=2, seed=1))
    trajs = [agent.run(JevGymEnv(seed=i, families="v2", strict=True, n_distractors=20, distractor_style="near",
                                 noise_lines=5)).to_dict() for i in range(6)]
    assert all(t["error"] is None for t in trajs), [t["error"] for t in trajs]
    (dp, mode), = [(dp, m) for dp, m in E6D_CELLS if cfg.dp(dp).mode == m]
    mine = [r for r in decision_records(trajs) if r["dp"] == dp.value and r["mode"] == mode.value]
    assert mine and any(r.get("gold_status") == "ok" for r in mine), {r.get("gold_status") for r in mine}
    for t in trajs:
        assert abs(t["usage"]["total"]["cost_usd"] - sum(c["cost_usd"] for c in t["calls"])) < 1e-9
        assert all(c["decision_point"] for c in t["calls"])
        assert sum(s["llm_calls"] for s in t["steps"]) == sum(1 for c in t["calls"] if c["component"] == "s2")


E6D_WIRE = {
    "e6d_assists": dict(memory_write={"mode": "assist"}, memory_read={"mode": "always"}, reflection={"mode": "assist"},
                        stop={"mode": "assist"}, routing={"mode": "assist"}),
    "e6d_guards": dict(memory_write={"mode": "always"}, memory_read={"mode": "guard", "tau": 0.5},
                       reflection={"mode": "guard", "tau": 0.5}, routing={"mode": "guard", "tau": 0.5}),
}


@pytest.mark.parametrize("canary_tool", [True, False])
@pytest.mark.parametrize("arm", list(E6D_WIRE))
def test_new_cells_leak_no_oracle_value_on_the_wire(arm: str, canary_tool: bool) -> None:
    import test_wire_canary as W
    from jevharness.decision.jev import JevBackend

    s1_all, s2_all, dumps = W.Wire(W.jev_answers), W.Wire(W.jev_answers), []
    for seed in (0, 3):
        real = JevGymEnv(seed=seed, families="v2", distractor_style="near", n_distractors=60, noise_lines=5,
                         strict=True)
        env = W.CanaryOracleEnv(real, canary_tool)
        s1_wire, s2_wire = W.Wire(W.s1_policy(real)), W.Wire(W.s2_policy(real))
        s1 = JevBackend(provider="openrouter", api_key="test-key", transport=s1_wire.transport())
        llm = W.compat_llm("gemini", "gemini-3.5-flash", s2_wire, temperature=0, reasoning_effort="minimal")
        small = W.compat_llm("gemini", "gemini-3.5-flash-lite", s2_wire, temperature=0)
        traj = JevAgent(llm, HarnessConfig(name=arm, max_steps=12, **E6D_WIRE[arm]), s1=s1, small_llm=small).run(env)
        assert traj.error is None, traj.error
        s1_all.requests += s1_wire.requests
        s2_all.requests += s2_wire.requests
        dumps.append(json.dumps(traj.to_dict(), default=str))
    s1_all.assert_clean(min_requests=2, key_names=False)
    s2_all.assert_clean(min_requests=2, key_names=False)
    assert any(W.CANARY in d for d in dumps)


def test_e6d_configs_and_readout_row() -> None:
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import explore_e6b_modes as E

    from jevharness.bench.runner import expand_arms

    root = Path(__file__).resolve().parents[1]
    exp = yaml.safe_load((root / "configs/explore/e6d_modes/eval.yaml").read_text())
    cfgs = {c.name: c.validate_modes() for c in expand_arms(exp)}
    cells = E.ROWS["e6d"]["CELLS"]
    assert set(cfgs) == {"llm_only", "small_only", *cells}
    assert {(DP(dp), Mode(m)) for dp, m in cells.values()} == set(E6D_CELLS)
    assert all(cfgs[a].dp(DP(dp)).mode == Mode(m) for a, (dp, m) in cells.items())
    assert all(cfgs[a].dp(DP(dp)).tau == 0.5 for a, (dp, m) in cells.items() if m == "guard")
    assert exp["suite"] == {"name": "jevgym", "n": 200, "seed0": 301000, "families": "v2", "distractor_style": "near",
                            "n_distractors": 60, "noise_lines": 20, "strict": True}
    assert exp["workers"] == 24 and exp["schedule"] == {"mode": "task_major", "seed": 732}
    sm = yaml.safe_load((root / "configs/explore/e6d_modes/smoke.yaml").read_text())
    assert sm["suite"]["seed0"] == 900 and sm["n_tasks"] == 2 and sm["arms"] == exp["arms"]
    assert E.ROWS["e6d"]["CAP_USD"] == 120.0
