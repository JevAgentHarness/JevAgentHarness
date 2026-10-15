"""E6e (exploratory; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6e", results/explore/e6e_modes/cells.md):
the five REPLACE cells at memory write, memory read, reflection, stop and routing, run as their own arms. System-1's
argmax decides alone (the LLM is never asked at the point; tau is ignored). Each behaves exactly as the cell named in
``EQUIVALENT_NOTES`` (trigger at 0.5; stop: cascade at 0.5; memory read: assist with k = 1), which the JevGym
simulation below checks decision by decision. Plus: the stub tests of each cell, the wire canary, the configs and the
fingerprint check that every mode added by E6d behaves and calls exactly as on the pre-E6e code
(``tests/e6b_fingerprint.py e6d``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e6b_fingerprint as FP  # noqa: E402
import test_e6b_modes as B  # noqa: E402

from jevharness.bench.scorecard import call_coverage  # noqa: E402
from jevharness.bench.suites.jevgym import JevGymEnv  # noqa: E402
from jevharness.decision.simulated import SimulatedBackend  # noqa: E402
from jevharness.harness.agent import JevAgent  # noqa: E402
from jevharness.harness.config import (DP, E6E_CELLS, EQUIVALENT_NOTES, UNDEFINED_CELLS, VALID_MODES,  # noqa: E402
                                       DPConfig, HarnessConfig, Mode)
from jevharness.harness.prompts import EARLY_STOP_PROMPT, REFLECT_PROMPT  # noqa: E402

D = DPConfig


def test_e6d_modes_behave_and_call_exactly_as_before() -> None:
    exp = json.loads(FP.FIXTURE_E6D.read_text())["digests"]
    got = FP.prior_e6d_digests()
    assert set(got) == set(exp) and {k for k in exp if got[k] != exp[k]} == set()


def test_grid_after_e6e_every_meaningful_cell_is_implemented() -> None:
    impl = {(dp, m) for dp, ms in VALID_MODES.items() for m in ms}
    assert E6E_CELLS <= impl and set(EQUIVALENT_NOTES) == set(E6E_CELLS)
    assert all(m == Mode.REPLACE for _, m in E6E_CELLS)
    assert sorted((dp.value, m.value) for dp, m in UNDEFINED_CELLS) == []   # E6f defined the last three
    assert {EQUIVALENT_NOTES[(dp, Mode.REPLACE)].split(":")[0] for dp in (DP.MEMORY_WRITE, DP.REFLECTION,
                                                                          DP.ROUTING)} == {"trigger at tau = 0.5"}
    assert EQUIVALENT_NOTES[(DP.STOP, Mode.REPLACE)].startswith("cascade at tau = 0.5")
    assert EQUIVALENT_NOTES[(DP.MEMORY_READ, Mode.REPLACE)].startswith("assist with k = 1")


def _dps(t: dict[str, Any], dp: str) -> list[dict[str, Any]]:
    return [d for s in t["steps"] for d in s["decisions"] if d["dp"] == dp]


def _s2_at(t: dict[str, Any], dp: str) -> list[dict[str, Any]]:
    return [c for c in t["calls"] if c["component"] == "s2" and c["decision_point"] == dp]


# ------------------------------------------------------------------ each cell with the stubs of test_e6b_modes
def test_memory_write_replace_stores_on_the_argmax_alone_and_ignores_tau() -> None:
    cfg = HarnessConfig(name="mw_replace", memory_write=D(mode=Mode.REPLACE, tau=0.9), memory_read=D(mode=Mode.ALWAYS))
    t, s1, llm = B._run(cfg, "lookup_cancel", B.S1Stub({"memory_write": 0.6}, yes={"memory_write": True}))
    w = _dps(t, "memory_write")
    assert len(w) == 2 and all(d["mode"] == "replace" and d["s1_value"] is True and d["final_value"] is True
                               and d["s2_value"] is None for d in w)            # p_yes 0.6 >= 0.5 (tau 0.9 ignored)
    assert t["meta"]["memory"]["stored"] == 2 and not _s2_at(t, "memory_write")
    assert not any(r["last"].endswith("Answer yes or no.") for r in llm.requests)
    t, _, _ = B._run(cfg, "lookup_cancel", B.S1Stub({"memory_write": 0.6}, yes={"memory_write": False}))
    assert t["meta"]["memory"]["stored"] == 0 and all(d["final_value"] is False for d in _dps(t, "memory_write"))


def test_memory_read_replace_injects_the_single_top_memory_whatever_k() -> None:
    cfg = HarnessConfig(name="mr_replace", memory_write=D(mode=Mode.ALWAYS), memory_read=D(mode=Mode.REPLACE, k=3))
    t, s1, llm = B._run(cfg, "lookup_cancel", B.S1Stub({"memory_read": 0.95}))
    r = _dps(t, "memory_read")
    # step 1: a single stored memory is injected without a question (as the assist with k = 1); step 2: two memories
    # and "none" -> the model's top memory alone, although k = 3
    assert len(r) == 1 and r[0]["mode"] == "replace" and not r[0]["deferred"] and len(r[0]["options"]) == 3
    assert r[0]["final_value"] == [r[0]["s1_value"]]
    assert not _s2_at(t, "memory_read")
    mem = [x["last"] for x in llm.requests if x["last"].startswith("[Memory]")]
    assert len(mem) == 2 and all(m.count("- [m") == 1 for m in mem)


def test_reflection_replace_reflects_on_the_argmax_without_an_llm_gate() -> None:
    cfg = HarnessConfig(name="rf_replace", reflection=D(mode=Mode.REPLACE, tau=0.9))
    t, _, llm = B._run(cfg, "lookup_cancel", B.S1Stub({"reflection": 0.6}, yes={"reflection": True}))
    rf = _dps(t, "reflection")
    assert len(rf) == 2 and all(d["mode"] == "replace" and d["final_value"] is True for d in rf)
    assert sum(REFLECT_PROMPT in x["last"] for x in llm.requests) == 2
    assert not any(x["last"].endswith("Answer yes or no.") for x in llm.requests)
    t, _, llm = B._run(cfg, "lookup_cancel", B.S1Stub({"reflection": 0.6}, yes={"reflection": False}))
    assert not any(REFLECT_PROMPT in x["last"] for x in llm.requests) and not _s2_at(t, "reflection")


def test_stop_replace_rejects_finishes_and_stops_early_on_the_argmax() -> None:
    cfg = HarnessConfig(name="st_replace", max_steps=4, stop=D(mode=Mode.REPLACE, tau=0.9))
    # an unsure "no" (0.55; a cascade at tau 0.9 would defer) still rejects the finish proposal
    t, _, _ = B._run(cfg, "cancel", B.S1Stub({"stop": 0.55}, yes={"stop": False}))
    st = _dps(t, "stop")
    rejects = [d for d in st if d["s2_value"] is True and d["final_value"] is False]
    assert rejects and rejects[0]["retry"] == 0 and all(d["mode"] == "replace" and not d["deferred"] for d in st)
    assert t["steps"][1]["observation"].startswith("(finish rejected") and not t["result"]["success"]
    assert call_coverage([t])["n_arg_slots"] == 2                 # rejected finishes were never executed
    # an unsure "yes" at a non-finish step ends the episode early: the LLM writes the answer in one forced call
    t, _, llm = B._run(cfg, "lookup_cancel", B.S1Stub({"stop": 0.55}, yes={"stop": True}))
    assert t["steps"][0]["tool"] == "finish" and llm.requests[-1]["tool_choice"] == "finish"
    assert EARLY_STOP_PROMPT in llm.requests[-1]["last"]
    assert [(d["s2_value"], d["final_value"]) for d in _dps(t, "stop")] == [(False, True)]


def test_routing_replace_routes_on_the_argmax_without_a_gate() -> None:
    cfg = HarnessConfig(name="rt_replace", routing=D(mode=Mode.REPLACE, tau=0.9))
    with pytest.raises(ValueError, match="needs small_llm"):
        JevAgent(B.S2Stub(), cfg, s1=B.S1Stub())
    big, small = B.S2Stub(model="big"), B.S2Stub(model="small")
    t, _, _ = B._run(cfg, "cancel", B.S1Stub({"routing": 0.6}, yes={"routing": True}), big, small)
    rt = _dps(t, "routing")
    assert rt and all(d["mode"] == "replace" and d["s1_value"] is True and d["final_value"] == "large" for d in rt)
    assert {c["model"] for c in t["calls"] if c["component"] == "s2"} == {"big"} and not _s2_at(t, "routing")
    big, small = B.S2Stub(model="big"), B.S2Stub(model="small")
    t, _, _ = B._run(cfg, "cancel", B.S1Stub({"routing": 0.6}, yes={"routing": False}), big, small)
    assert {c["model"] for c in t["calls"] if c["component"] == "s2"} == {"small"}


# ------------------------------------------------------------------ JevGym: each replace = the cell it equals
# (replace arm with a tau / k that the replace ignores, the equal cell at its fixed setting, the mode it logs)
PAIRS: dict[str, tuple[HarnessConfig, HarnessConfig, str]] = {
    "mw_replace": (HarnessConfig(name="mw_replace", memory_write=D(mode=Mode.REPLACE, tau=0.9),
                                 memory_read=D(mode=Mode.ALWAYS)),
                   HarnessConfig(name="mw_trigger", memory_write=D(mode=Mode.TRIGGER, tau=0.5),
                                 memory_read=D(mode=Mode.ALWAYS)), "trigger"),
    "mr_replace": (HarnessConfig(name="mr_replace", memory_write=D(mode=Mode.ALWAYS), memory_read=D(mode=Mode.REPLACE, k=3)),
                   HarnessConfig(name="mr_assist1", memory_write=D(mode=Mode.ALWAYS), memory_read=D(mode=Mode.ASSIST, k=1)),
                   "assist"),
    "rf_replace": (HarnessConfig(name="rf_replace", reflection=D(mode=Mode.REPLACE, tau=0.9)),
                   HarnessConfig(name="rf_trigger", reflection=D(mode=Mode.TRIGGER, tau=0.5)), "trigger"),
    "st_replace": (HarnessConfig(name="st_replace", stop=D(mode=Mode.REPLACE, tau=0.9)),
                   HarnessConfig(name="st_cascade", stop=D(mode=Mode.CASCADE, tau=0.5)), "cascade"),
    "rt_replace": (HarnessConfig(name="rt_replace", routing=D(mode=Mode.REPLACE, tau=0.9)),
                   HarnessConfig(name="rt_trigger", routing=D(mode=Mode.TRIGGER, tau=0.5)), "trigger"),
}


def _jevgym(cfg: HarnessConfig, n: int = 6) -> list[dict[str, Any]]:
    agent = JevAgent(FP.GateSimLLM("scripted", error_rate=0.1, simulated_latency_ms=5, seed=0), cfg,
                     s1=SimulatedBackend(accuracy=0.8, seed=1),
                     small_llm=FP.GateSimLLM("scripted-small", error_rate=0.3, simulated_latency_ms=2, seed=1))
    return [agent.run(JevGymEnv(seed=i, families="v2", strict=True, n_distractors=20, distractor_style="near",
                                noise_lines=5)).to_dict() for i in range(n)]


@pytest.mark.parametrize("arm", sorted(PAIRS))
def test_each_replace_arm_equals_its_named_cell_on_jevgym(arm: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import explore_e6b_modes as E

    rep_cfg, eq_cfg, eq_mode = PAIRS[arm]
    (dp, _), = [(dp, m) for dp, m in E6E_CELLS if rep_cfg.dp(dp).mode == Mode.REPLACE]
    rep, eq = _jevgym(rep_cfg), _jevgym(eq_cfg)
    assert all(t["error"] is None for t in rep + eq), [t["error"] for t in rep + eq]
    mine = [d for t in rep for d in _dps(t, dp.value)]
    assert mine and all(d["mode"] == "replace" for d in mine)
    assert any(d.get("gold_status") == "ok" for d in mine), {d.get("gold_status") for d in mine}
    for t in rep:     # one ledger; no LLM gate at the point (reflection: its calls are the reflections it fired;
        # stop: its forced early-stop finish is the agent's turn)
        assert abs(t["usage"]["total"]["cost_usd"] - sum(c["cost_usd"] for c in t["calls"])) < 1e-9
        assert all(c["decision_point"] for c in t["calls"])
        assert sum(s["llm_calls"] for s in t["steps"]) == sum(1 for c in t["calls"] if c["component"] == "s2")
        if dp == DP.REFLECTION:
            assert len(_s2_at(t, dp.value)) == sum(d["final_value"] is True for d in _dps(t, dp.value))
        elif dp != DP.STOP:
            assert not _s2_at(t, dp.value)
    # behaviour and calls: identical to the named cell once the logged mode is set aside
    fr = [FP.episode_fingerprint(t) for t in rep]
    for f in fr:
        for s in f["steps"]:
            for d in s["decisions"]:
                if d["mode"] == "replace":
                    d["mode"] = eq_mode
    assert fr == [FP.episode_fingerprint(t) for t in eq]
    # the readout's replace metrics (scripts/explore_e6b_modes.py cell_metrics)
    pm = E.cell_metrics(rep, dp.value, "replace")
    assert pm["coverage"] == 1.0 and pm["acted"] == 1.0 and pm["acted_kind"] == "decided"
    assert 0.0 <= pm["acc"] <= 1.0 and pm["n_decisions"] == len(mine)
    if dp == DP.STOP:
        assert {"finish_rejected", "early_stops", "overrode"} <= set(pm)
    if dp == DP.MEMORY_READ:
        assert pm["mean_injected"] <= 1.0 and 0.0 <= pm["final_covers"] <= 1.0


# ------------------------------------------------------------------ wire canary
E6E_WIRE = {"e6e_replace": dict(memory_write={"mode": "replace"}, memory_read={"mode": "replace"},
                                reflection={"mode": "replace"}, stop={"mode": "replace"}, routing={"mode": "replace"})}


@pytest.mark.parametrize("canary_tool", [True, False])
@pytest.mark.parametrize("arm", list(E6E_WIRE))
def test_replace_cells_leak_no_oracle_value_on_the_wire(arm: str, canary_tool: bool) -> None:
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
        traj = JevAgent(llm, HarnessConfig(name=arm, max_steps=12, **E6E_WIRE[arm]), s1=s1, small_llm=small).run(env)
        assert traj.error is None, traj.error
        s1_all.requests += s1_wire.requests
        s2_all.requests += s2_wire.requests
        dumps.append(json.dumps(traj.to_dict(), default=str))
    s1_all.assert_clean(min_requests=2, key_names=False)
    s2_all.assert_clean(min_requests=2, key_names=False)
    assert any(W.CANARY in d for d in dumps)


# ------------------------------------------------------------------ configs and the readout row
def test_e6e_configs_and_readout_row() -> None:
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import explore_e6b_modes as E

    from jevharness.bench.runner import expand_arms

    root = Path(__file__).resolve().parents[1]
    exp = yaml.safe_load((root / "configs/explore/e6e_modes/eval.yaml").read_text())
    cfgs = {c.name: c.validate_modes() for c in expand_arms(exp)}
    cells = E.ROWS["e6e"]["CELLS"]
    assert set(cfgs) == {"llm_only", "small_only", *cells}
    assert {(DP(dp), Mode(m)) for dp, m in cells.values()} == set(E6E_CELLS)
    assert all(cfgs[a].dp(DP(dp)).mode == Mode(m) for a, (dp, m) in cells.items())
    assert cfgs["mw_replace"].memory_read.mode == Mode.ALWAYS and cfgs["mr_replace"].memory_write.mode == Mode.ALWAYS
    assert all(sum(cfgs[a].dp(d).mode not in (Mode.OFF, Mode.ALWAYS) for d in DP) == 1 for a in cells)
    assert exp["suite"] == {"name": "jevgym", "n": 200, "seed0": 301400, "families": "v2", "distractor_style": "near",
                            "n_distractors": 60, "noise_lines": 20, "strict": True}
    assert exp["workers"] == 24 and exp["schedule"] == {"mode": "task_major", "seed": 732}
    assert exp["llm"]["provider"] == "openrouter" and exp["small_llm"]["provider"] == "gemini"
    assert exp["llms"]["lite"]["provider"] == "gemini"
    sm = yaml.safe_load((root / "configs/explore/e6e_modes/smoke.yaml").read_text())
    assert sm["suite"]["seed0"] == 900 and sm["n_tasks"] == 2 and sm["arms"] == exp["arms"]
    assert {k: sm[k] for k in ("llm", "small_llm", "llms", "s1")} == {k: exp[k] for k in ("llm", "small_llm", "llms", "s1")}
    assert E.ROWS["e6e"]["CAP_USD"] == 60.0 and E.ROWS["e6e"]["RUN_EVAL"] == "e6e_modes"
