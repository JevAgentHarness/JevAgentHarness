"""One-step counterfactual validity check (review round 4, P1-6): the full-episode continuation from a logged
state (``harness.measure.continue_episode``), the opportunity listing / stratified subset / agreement table
(``bench.meta_gold``) and the script (``scripts/cf_validity.py``). Offline only: simulated System-2."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from jevharness.accounting import Ledger
from jevharness.bench import meta_gold as MG
from jevharness.bench.suites.jevgym import JevGymEnv
from jevharness.decision.simulated import SimulatedBackend
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import CFConfig, DPConfig, HarnessConfig, MeasureConfig, Mode
from jevharness.harness.measure import FULL_INTERVENTIONS, REJECTED_OBS, Rollout, continue_episode
from jevharness.harness.prompts import REFLECT_PROMPT, STOP_REJECT_NOTE
from jevharness.llm.simulated import SimulatedLLM

SUITE = dict(families="v2", strict=True, distractor_style="near", n_distractors=20, noise_lines=5)
ROOT = Path(__file__).resolve().parents[1]


def _env(seed: int) -> JevGymEnv:
    return JevGymEnv(seed=seed, **SUITE)


def _seed_of(family: str) -> int:
    for s in range(100):
        if JevGymEnv(seed=s, **SUITE).family == family:
            return s
    raise AssertionError(family)


class Recording(SimulatedLLM):
    """A simulated System-2 that keeps the messages of every call."""

    def __init__(self, *a: Any, **k: Any):
        super().__init__(*a, **k)
        self.seen: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def _chat(self, messages, tools=None, **kw):  # noqa: ANN001
        self.seen.append(([dict(m) for m in messages], tools))
        return super()._chat(messages, tools, **kw)


def _agent(err: float = 0.25, seed: int = 0, **kw: Any) -> JevAgent:
    return JevAgent(Recording("sim-lite", error_rate=err, simulated_latency_ms=1, seed=seed),
                    HarnessConfig(name="llm_only"), **kw)


# ---------------------------------------------------------------- continuation fidelity
def test_empty_prefix_without_intervention_reproduces_the_llm_only_run():
    for seed in range(6):
        agent = _agent(err=0.3, seed=seed)
        full = agent.run(_env(seed))
        ro = continue_episode(_agent(err=0.3, seed=seed), _env(seed), [], None, ledger=Ledger())
        assert isinstance(ro, Rollout) and ro.error is None and ro.start_step == 0
        assert ro.success == full.success and ro.result["success"] == full.result["success"]
        assert [n for n, _ in ro.actions] == [s.tool for s in full.steps]
        assert ro.n_steps == len(full.steps)
        assert ro.cost_usd >= 0 and len(ro.calls) == sum(s.llm_calls for s in full.steps)


def test_prefix_replay_continues_an_oracle_episode_and_checks_the_logged_observations():
    seed = _seed_of("multi")
    oracle = JevAgent(SimulatedLLM("oracle", error_rate=0.0, simulated_latency_ms=1), HarnessConfig(name="llm_only"))
    full = oracle.run(_env(seed))
    assert full.success and len(full.steps) >= 4
    k = len(full.steps) // 2
    prefix = [(s.tool, s.args) for s in full.steps[:k]]
    obs = [s.observation for s in full.steps[:k]]
    ro = continue_episode(oracle, _env(seed), prefix, None, ledger=Ledger(), logged_obs=obs)
    assert ro.error is None and ro.success and ro.start_step == k and ro.prefix_match is True
    assert [n for n, _ in ro.actions] == [s.tool for s in full.steps[k:]]
    assert ro.n_steps == len(full.steps) - k
    # a wrong logged observation is detected, the continuation still runs
    bad = list(obs)
    bad[0] = "something else entirely"
    ro2 = continue_episode(oracle, _env(seed), prefix, None, ledger=Ledger(), logged_obs=bad)
    assert ro2.prefix_match is False and ro2.success
    # no logged observations: nothing to check
    assert continue_episode(oracle, _env(seed), prefix, None).prefix_match is None


def test_interventions_are_applied_once_at_the_first_continued_step():
    seed = _seed_of("multi")
    oracle = JevAgent(SimulatedLLM("oracle", error_rate=0.0, simulated_latency_ms=1), HarnessConfig(name="llm_only"))
    full = oracle.run(_env(seed))
    k = 3
    prefix = [(s.tool, s.args) for s in full.steps[:k]]
    # reflection: one text call with REFLECT_PROMPT, then the note is in every later request
    agent = _agent(err=0.0)
    ro = continue_episode(agent, _env(seed), prefix, "reflection", ledger=Ledger())
    rec: Recording = agent.llm  # type: ignore[assignment]
    assert ro.error is None and ro.success and ro.reflection is not None and ro.start_step == k
    first_msgs, first_tools = rec.seen[0]
    assert first_tools is None and first_msgs[-1]["content"] == REFLECT_PROMPT
    assert first_msgs[-2]["role"] == "tool"            # the prefix history precedes the reflection prompt
    for msgs, tools in rec.seen[1:]:
        assert tools is not None and any("[Reflection]" in str(m.get("content")) for m in msgs)
    assert ro.calls[0]["decision_point"] == "reflection" and all(c["decision_point"] == "tool_select"
                                                                 for c in ro.calls[1:])
    assert ro.n_steps == len(full.steps) - k
    # large: the first tool decision is the large model's, the rest the arm's own
    agent = _agent(err=0.0)
    large = Recording("sim-flash", error_rate=0.0, simulated_latency_ms=1)
    ro = continue_episode(agent, _env(seed), prefix, "large", ledger=Ledger(), large_llm=large)
    assert ro.error is None and ro.success
    assert [c["model"] for c in ro.calls] == ["sim-flash"] + ["sim-lite"] * (len(ro.calls) - 1)
    assert len(large.seen) == 1 and len(agent.llm.seen) == len(ro.calls) - 1  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="large_llm"):
        continue_episode(agent, _env(seed), prefix, "large")
    # reject: the snapshot step is a finish proposal; the note is appended, the proposal consumes the step
    pre_fin = [(s.tool, s.args) for s in full.steps[:-1]]
    a0 = (full.steps[-1].tool, full.steps[-1].args)
    assert a0[0] == "finish"
    agent = _agent(err=0.0)
    ro = continue_episode(agent, _env(seed), pre_fin, "reject", ledger=Ledger(), a0=a0)
    assert ro.error is None and ro.start_step == len(pre_fin) + 1 and ro.success
    msgs, _ = agent.llm.seen[0]  # type: ignore[attr-defined]
    assert msgs[-1]["role"] == "tool" and STOP_REJECT_NOTE in msgs[-1]["content"]
    assert not any(m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "finish" for m in msgs)
    # control from the same state: the loop just continues at the snapshot step
    ro0 = continue_episode(_agent(err=0.0), _env(seed), pre_fin, None, ledger=Ledger(), a0=a0)
    assert ro0.start_step == len(pre_fin) and ro0.actions[0][0] == "finish" and ro0.success


def test_continuation_refuses_live_decision_points_and_bad_prefixes():
    seed = _seed_of("clarify")
    live = JevAgent(SimulatedLLM("sim", error_rate=0.0), HarnessConfig(name="x", stop=DPConfig(mode=Mode.GUARD)),
                    s1=SimulatedBackend(accuracy=0.9))
    with pytest.raises(ValueError, match="LLM-only"):
        continue_episode(live, _env(seed), [], None)
    shadow = JevAgent(SimulatedLLM("sim", error_rate=0.0), HarnessConfig(name="m", reflection=DPConfig(mode=Mode.SHADOW)),
                      s1=SimulatedBackend(accuracy=0.9))
    with pytest.raises(ValueError, match="LLM-only"):
        continue_episode(shadow, _env(seed), [], None)
    ok = JevAgent(SimulatedLLM("sim", error_rate=0.0), HarnessConfig(name="llm_only"))
    with pytest.raises(ValueError, match="intervention"):
        continue_episode(ok, _env(seed), [], "nonsense")
    # a prefix containing the finish proposal is data, not a crash: the rollout carries the error
    ro = continue_episode(ok, _env(seed), [("finish", {"answer": "x"})], None)
    assert ro.error and "finish proposal" in ro.error and ro.success is False
    assert set(FULL_INTERVENTIONS) == {"reflection", "large", "reject"}


def test_replayed_call_ids_have_the_shape_gemini_accepts():
    """``call_`` + 6 digits (Gemini's own shape), deterministic: Gemini's OpenAI endpoint mostly returns
    MALFORMED_FUNCTION_CALL on a history whose call ids look otherwise (``call_replay0000``, hex); see
    ``measure.replay_call_id``."""
    import re

    from jevharness.harness.measure import replay_call_id

    ids = [replay_call_id("jevgym-multi-19-d60-near-strict", k) for k in range(5)]
    assert all(re.fullmatch(r"call_[0-9]{6}", i) for i in ids) and len(set(ids)) == 5
    assert ids == [replay_call_id("jevgym-multi-19-d60-near-strict", k) for k in range(5)]
    assert replay_call_id("other", 0) != ids[0]
    seed = _seed_of("clarify")
    oracle = JevAgent(SimulatedLLM("oracle", error_rate=0.0, simulated_latency_ms=1), HarnessConfig(name="llm_only"))
    full = oracle.run(_env(seed))
    agent = _agent(err=0.0)
    continue_episode(agent, _env(seed), [(s.tool, s.args) for s in full.steps[:2]], None, ledger=Ledger())
    msgs, _ = agent.llm.seen[0]  # type: ignore[attr-defined]
    calls = [m["tool_calls"][0]["id"] for m in msgs if m.get("tool_calls")]
    assert len(calls) == 2 and all(re.fullmatch(r"call_[0-9]{6}", c) for c in calls)
    assert [m["tool_call_id"] for m in msgs if m.get("role") == "tool"] == calls


def test_rejected_finishes_in_the_prefix_replay_as_the_note():
    seed = _seed_of("clarify")
    oracle = JevAgent(SimulatedLLM("oracle", error_rate=0.0, simulated_latency_ms=1), HarnessConfig(name="llm_only"))
    full = oracle.run(_env(seed))
    k = 2
    prefix = [(s.tool, s.args) for s in full.steps[:k]] + [("finish", {"answer": "early"})]
    obs = [s.observation for s in full.steps[:k]] + [REJECTED_OBS]
    agent = _agent(err=0.0)
    ro = continue_episode(agent, _env(seed), prefix, None, ledger=Ledger(), logged_obs=obs)
    assert ro.error is None and ro.success and ro.start_step == k + 1
    msgs, _ = agent.llm.seen[0]  # type: ignore[attr-defined]
    assert STOP_REJECT_NOTE in msgs[-1]["content"] and sum(1 for m in msgs if m.get("tool_calls")) == k


# ---------------------------------------------------------------- opportunities, subset, table
def _cf_traj(task: str, labels: list[tuple[int, str, str, str, str | None]], n_steps: int = 4,
             success: bool = False) -> dict[str, Any]:
    """A measurement-arm trajectory dict with one-step CF labels: (step, dp, v0, v_branch, v_placebo)."""
    steps = [{"step": i, "tool": "get_orders" if i < n_steps - 1 else "finish", "args": {"user_id": "U1"},
              "observation": f"obs {i}", "decisions": []} for i in range(n_steps)]
    for step, dp, v0, vb, vp in labels:
        br = MG.CF_BRANCHES[dp][0]
        aux = {"v": 1, "verdict": v0, br: vb, "cf_status": "ok", "cf_weight": 1.0,
               "benefit": v0 in MG.FAILING_VERDICTS and vb == "progress",
               "harm": v0 == "progress" and vb in MG.FAILING_VERDICTS}
        if vp is not None:
            aux["verdict_placebo"] = vp
        if dp == "stop":
            aux["kind"] = "ok" if v0 == "progress" else v0
        steps[step]["decisions"].append({"dp": dp, "mode": "shadow", "question": "q", "gold": v0 != "progress",
                                         "gold_status": "ok", "gold_aux": aux, "state": "s"})
    return {"task_id": task, "config": "meta", "steps": steps, "result": {"success": success}, "meta": {}}


def test_cf_opportunities_list_labels_prefixes_and_skip_fired_records():
    t = _cf_traj("jevgym-multi-1", [(1, "reflection", "progress", "progress", "progress"),
                                    (3, "reflection", "wrong_answer", "progress", "wrong_answer"),
                                    (3, "routing", "wrong_answer", "wrong_answer", "wrong_answer"),
                                    (3, "stop", "progress", "wrong_answer", "progress")])
    # a fired-step record (no branch verdict) and an errored CF are not opportunities
    t["steps"][2]["decisions"].append({"dp": "reflection", "mode": "trigger", "gold": True, "gold_status": "ok",
                                       "gold_aux": {"verdict": "fatal", "verdict_src": "cf", "cf_status": "ok"}})
    t["steps"][2]["decisions"].append({"dp": "routing", "mode": "shadow", "gold": None, "gold_status": "ok",
                                       "gold_aux": {"verdict": "progress", "cf_status": "cf_error"}})
    opps = MG.cf_opportunities([t])
    assert [(o["step"], o["dp"], o["label"], o["intervention"], o["final_step"]) for o in opps] == [
        (1, "reflection", "neutral", "reflection", False), (3, "reflection", "helps", "reflection", True),
        (3, "routing", "neutral", "large", True), (3, "stop", "harms", "reject", True)]
    o = opps[1]
    assert o["prefix"] == [("get_orders", {"user_id": "U1"})] * 3 and o["logged_obs"] == ["obs 0", "obs 1", "obs 2"]
    assert o["a0"] == ("finish", {"user_id": "U1"}) and o["v_placebo"] == "wrong_answer" and o["gold"] is True
    assert opps[0]["prefix"] == [("get_orders", {"user_id": "U1"})]


def test_subset_is_stratified_balanced_across_dps_interleaved_and_seeded():
    trajs = []
    for i in range(12):
        labs = [(1, "reflection", "progress", "progress", "progress"), (2, "routing", "progress", "progress", "progress"),
                (3, "stop", "progress", "progress", "progress")]
        if i % 2 == 0:
            labs += [(3, "reflection", "wrong_answer", "progress", "wrong_answer"),
                     (3, "routing", "wrong_answer", "progress", "wrong_answer")]
        if i % 3 == 0:
            labs[-1] = (3, "stop", "wrong_answer", "progress", "wrong_answer")
        trajs.append(_cf_traj(f"t{i}", labs))
    opps = MG.cf_opportunities(trajs)
    sub = MG.select_cf_validity_subset(opps, n=30, min_helps=10, min_not=10, seed=0)
    assert len(sub) == 30 and len({(o["task_id"], o["step"], o["dp"]) for o in sub}) == 30
    strata = [o["stratum"] for o in sub]
    assert strata.count("helps") == 10 and strata.count("not_helps") == 10 and strata.count("random") == 10
    assert all(o["label"] == "helps" for o in sub if o["stratum"] == "helps")
    assert all(o["label"] != "helps" for o in sub if o["stratum"] == "not_helps")
    # round-robin over decision points inside a stratum: 10 helps over 3 dps -> 4 / 3 / 3
    assert sorted(sum(1 for o in sub if o["stratum"] == "helps" and o["dp"] == dp)
                  for dp in ("reflection", "routing", "stop")) == [3, 3, 4]
    assert strata[:3] == ["helps", "not_helps", "random"]       # interleaved for budget stops
    assert MG.select_cf_validity_subset(opps, n=30, seed=0) == sub
    assert MG.select_cf_validity_subset(opps, n=30, seed=1) != sub
    # a stratum smaller than its minimum gives everything it has; the minima win over n (no random filler then)
    n_helps = sum(1 for o in opps if o["label"] == "helps")
    few = MG.select_cf_validity_subset(opps, n=8, min_helps=20, min_not=2, seed=0)
    assert len(few) == n_helps + 2 and sum(1 for o in few if o["stratum"] == "helps") == n_helps
    assert not any(o["stratum"] == "random" for o in few)
    small = MG.select_cf_validity_subset(opps[:3], n=30)
    assert len(small) == 3
    # min_harms: the rare 'harms' labels are forced into the not-helps stratum first; 0 keeps the selection above
    trajs.append(_cf_traj("harm1", [(3, "stop", "progress", "wrong_answer", "progress")]))
    trajs.append(_cf_traj("harm2", [(3, "stop", "progress", "wrong_answer", "progress"),
                                    (2, "reflection", "progress", "fatal", "progress")]))
    opps2 = MG.cf_opportunities(trajs)
    assert sum(1 for o in opps2 if o["label"] == "harms") == 3
    sub2 = MG.select_cf_validity_subset(opps2, n=30, min_helps=10, min_not=10, seed=0, min_harms=2)
    nots2 = [o for o in sub2 if o["stratum"] == "not_helps"]
    assert len(nots2) == 10 and sum(1 for o in nots2 if o["label"] == "harms") == 2
    assert len({(o["task_id"], o["step"], o["dp"]) for o in sub2}) == 30
    assert [o["label"] for o in MG.select_cf_validity_subset(opps2, n=30, seed=0, min_harms=0)
            if o["stratum"] == "not_helps"].count("harms") <= 2


def test_cf_validity_table_hand_computed():
    rows = [
        {"dp": "reflection", "label": "helps", "delta": 1.0, "final_step": True, "control_success": False, "episode_success": False},
        {"dp": "reflection", "label": "helps", "delta": 0.5, "final_step": True, "control_success": False, "episode_success": False},
        {"dp": "reflection", "label": "helps", "delta": 0.0, "final_step": True, "control_success": False, "episode_success": False},
        {"dp": "routing", "label": "neutral", "delta": 0.0, "final_step": False, "control_success": True, "episode_success": True},
        {"dp": "routing", "label": "neutral", "delta": -0.5, "final_step": False, "control_success": True, "episode_success": False},
        {"dp": "stop", "label": "harms", "delta": -1.0, "final_step": True, "control_success": True, "episode_success": True},
        {"dp": "stop", "label": "neutral", "delta": 0.5, "final_step": True, "control_success": True, "episode_success": True},
    ]
    t = MG.cf_validity_table(rows, min_n=3)
    o = t["overall"]
    assert o["n"] == 7
    assert o["confusion"] == {"helps": {"+": 2, "0": 1, "-": 0}, "neutral": {"+": 1, "0": 1, "-": 1},
                              "harms": {"+": 0, "0": 0, "-": 1}}
    # binary: helps<->+ agree on rows 1,2 (helps,+), 4,5,6 (not,not+); disagree on 3 (helps,0) and 7 (neutral,+)
    assert o["agreement"] == pytest.approx(5 / 7)
    lo, hi = o["agreement_ci"]
    assert 0.35 < lo < 5 / 7 < hi < 0.95
    # kappa binary: po = 5/7; marginals: label helps 3/7, rollout + 3/7 -> pe = (3*3 + 4*4)/49 = 25/49
    assert o["kappa_binary"] == pytest.approx((5 / 7 - 25 / 49) / (1 - 25 / 49))
    # 3-class: po = 4/7 (helps+ x2, neutral0, harms-); pe = (3*3 + 3*2 + 1*2)/49 = 17/49
    assert o["kappa_3class"] == pytest.approx((4 / 7 - 17 / 49) / (1 - 17 / 49))
    assert o["helps_confirmed"] == pytest.approx(2 / 3) and o["n_helps"] == 3
    assert o["not_helps_confirmed"] == pytest.approx(3 / 4) and o["harms_confirmed"] == 1.0
    assert o["mean_delta_helps"] == pytest.approx(0.5) and o["control_reproduces_log"] == pytest.approx(6 / 7)
    assert set(t["by_dp"]) == {"reflection", "routing", "stop"} and t["by_dp"]["reflection"]["n"] == 3
    assert "note" in t["by_dp"]["routing"] and t["by_dp"]["routing"]["n"] == 2
    assert t["by_final_step"]["final"]["n"] == 5 and "note" in t["by_final_step"]["non_final"]
    assert math.isnan(MG.cohen_kappa([])) and math.isnan(MG.cohen_kappa([(1, 1), (1, 1)]))
    assert MG.cohen_kappa([(1, 1), (0, 0)]) == 1.0 and MG.cohen_kappa([(1, 0), (0, 1)]) == -1.0
    assert MG.rollout_sign(0.5) == "+" and MG.rollout_sign(-1e-12) == "0" and MG.rollout_sign(-0.5) == "-"


# ---------------------------------------------------------------- the script, end to end, offline
def _measurement_run(tmp: Path, n: int = 8) -> Path:
    """A simulated measurement-arm run directory (experiment.yaml + meta_shadow_s2lite.jsonl) with CF labels."""
    cfg = HarnessConfig(name="meta_shadow_s2lite", reflection=DPConfig(mode=Mode.SHADOW),
                        routing=DPConfig(mode=Mode.SHADOW), stop=DPConfig(mode=Mode.SHADOW),
                        measure=MeasureConfig(seed=0, cf=CFConfig(reflection=True, large_llm="flash",
                                                                   stop_reject=True, placebo=True,
                                                                   sample_negatives=0.5)))
    llm = SimulatedLLM("sim-lite", error_rate=0.3, simulated_latency_ms=1)
    agent = JevAgent(llm, cfg, s1=SimulatedBackend(accuracy=0.8, seed=1),
                     measure_llms={"flash": SimulatedLLM("sim-flash", error_rate=0.0, simulated_latency_ms=1)})
    run = tmp / "e2b_sim"
    run.mkdir()
    exp = {"name": "e2b_sim", "suite": {"name": "jevgym", "n": n, **SUITE},
           "llm": {"provider": "simulated", "model": "sim-lite", "error_rate": 0.3, "simulated_latency_ms": 1},
           "llms": {"small_only": {"provider": "simulated", "model": "sim-lite", "error_rate": 0.3,
                                   "simulated_latency_ms": 1},
                    "flash": {"provider": "simulated", "model": "sim-flash", "error_rate": 0.0,
                              "simulated_latency_ms": 1}},
           "s1": {"type": "simulated", "accuracy": 0.8},
           "arms": [{"name": "meta_shadow_s2lite", "llm": "small_only", "reflection": {"mode": "shadow"},
                     "routing": {"mode": "shadow"}, "stop": {"mode": "shadow"},
                     "measure": {"seed": 0, "cf": {"reflection": True, "large_llm": "flash", "stop_reject": True,
                                                   "placebo": True, "sample_negatives": 0.5}}}]}
    (run / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    for i in range(n):
        agent.run(JevGymEnv(seed=i, **SUITE)).dump(run / "meta_shadow_s2lite.jsonl")
    return run


def test_script_end_to_end_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cf_validity as CV

    run = _measurement_run(tmp_path)
    opps = MG.cf_opportunities(MG._load(run, ["meta_shadow_s2lite"])["meta_shadow_s2lite"])
    assert opps, "the simulated measurement arm logged no one-step labels"
    out = tmp_path / "cfv"
    # dry run: the subset is written, nothing runs
    CV.main(["--run", str(run), "--mode", "replay", "--n", "6", "--min-helps", "2", "--min-not", "2", "--reps", "2",
             "--out", str(out), "--dry-run"])
    sub = json.loads((out / "subset.json").read_text())
    assert len(sub["subset"]) == min(6, len(opps)) and not (out / "rollouts.jsonl").exists()
    # the run (simulated System-2 from the experiment's own specs)
    CV.main(["--run", str(run), "--mode", "replay", "--n", "6", "--min-helps", "2", "--min-not", "2", "--reps", "2",
             "--out", str(out), "--workers", "2", "--budget-usd", "100"])
    recs = [json.loads(ln) for ln in (out / "rollouts.jsonl").read_text().splitlines() if ln.strip()]
    assert len(recs) == 4 * len(sub["subset"]) and all(r["simulated"] and r["source"] == "replay" for r in recs)
    assert all(r["error"] is None for r in recs) and all(r["prefix_match"] is True for r in recs)
    assert {r["arm"] for r in recs} == {"with", "without"} and {r["rep"] for r in recs} == {0, 1}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["simulated"] and summary["n_opportunities"] == len(sub["subset"]) and not summary["incomplete"]
    assert summary["table"]["overall"]["n"] == len(sub["subset"])
    assert summary["run"]["n_run"] == len(recs) and not summary["run"]["stopped_at_budget"]
    text = capsys.readouterr().out
    assert "SIMULATED" in text and "one-step label" in text
    # resume: nothing left to run; the table is recomputed
    CV.main(["--run", str(run), "--mode", "replay", "--n", "6", "--min-helps", "2", "--min-not", "2", "--reps", "2",
             "--out", str(out), "--budget-usd", "100"])
    assert json.loads((out / "summary.json").read_text())["run"]["n_run"] == 0
    assert len((out / "rollouts.jsonl").read_text().splitlines()) == len(recs)
    # --analyse only reads; several directories pool their rollouts
    CV.main(["--analyse", str(out)])
    assert json.loads((out / "summary.json").read_text())["n_rollouts"] == len(recs)
    import shutil

    out2 = tmp_path / "cfv_copy"
    shutil.copytree(out, out2)
    CV.main(["--analyse", str(out), str(out2)])
    pooled = json.loads((out / "summary_pooled.json").read_text())
    assert pooled["n_rollouts"] == 2 * len(recs) and pooled["n_opportunities"] == 2 * len(sub["subset"])
    assert pooled["pooled_from"] == [str(out), str(out2)]
    assert pooled["table"]["overall"]["n"] == 2 * len(sub["subset"])


def test_script_stops_at_the_budget(tmp_path: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cf_validity as CV

    run = _measurement_run(tmp_path, n=6)
    out = tmp_path / "cfv"
    # the simulated LLM is unpriced (cost 0): a zero budget stops before the first submission
    CV.main(["--run", str(run), "--mode", "replay", "--n", "4", "--min-helps", "1", "--min-not", "1", "--reps", "1",
             "--out", str(out), "--workers", "1", "--budget-usd", "0"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["stopped_at_budget"] and summary["run"]["n_run"] == 0
    assert summary["n_opportunities"] == 0 and summary["table"] == {}


def test_llm_only_config_from_a_logged_measurement_config():
    sys.path.insert(0, str(ROOT / "scripts"))
    import cf_validity as CV

    logged = {"name": "meta_shadow_s2lite", "max_steps": 12, "obs_truncate_chars": 500, "final_tool": "finish",
              "reflection": {"mode": "shadow"}, "stop": {"mode": "shadow", "extra": {"augment_trigger": True}},
              "measure": {"seed": 0, "cf": {"reflection": True}}}
    cfg = CV.llm_only_config(logged)
    assert cfg.short() == "llm_only" and cfg.max_steps == 12 and cfg.obs_truncate_chars == 500
    assert not cfg.measure.active() and cfg.name == "llm_only_cf_full"
    exp = {"llm": {"provider": "gemini", "model": "flash"}, "small_llm": {"provider": "gemini", "model": "lite"},
           "llms": {"small_only": {"provider": "gemini", "model": "lite"}, "flash": {"provider": "gemini", "model": "flash"}},
           "arms": [{"name": "m", "llm": "small_only", "measure": {"cf": {"large_llm": "flash"}}}, {"name": "plain"}]}
    assert CV.llm_specs(exp, "m") == ({"provider": "gemini", "model": "lite"}, {"provider": "gemini", "model": "flash"})
    assert CV.llm_specs(exp, "plain") == ({"provider": "gemini", "model": "flash"}, None)
    with pytest.raises(SystemExit):
        CV.llm_specs(exp, "nope")


# ---------------------------------------------------------------- in-process snapshots (the live mode)
def _measure_agent(err: float = 0.3) -> JevAgent:
    cfg = HarnessConfig(name="meta_shadow_cfv", reflection=DPConfig(mode=Mode.SHADOW),
                        routing=DPConfig(mode=Mode.SHADOW), stop=DPConfig(mode=Mode.SHADOW),
                        measure=MeasureConfig(seed=0, cf=CFConfig(reflection=True, large_llm="large",
                                                                   stop_reject=True, placebo=True,
                                                                   sample_negatives=1.0)))
    return JevAgent(SimulatedLLM("sim-lite", error_rate=err, simulated_latency_ms=1), cfg,
                    s1=SimulatedBackend(accuracy=0.8, seed=1),
                    measure_llms={"large": SimulatedLLM("sim-flash", error_rate=0.0, simulated_latency_ms=1)})


def test_keep_snapshots_collects_every_episode_and_leaves_the_snapshot_untouched():
    from jevharness.harness import measure as MS
    from jevharness.harness.measure import continue_from_snapshot, keep_snapshots

    agent = _measure_agent()
    assert MS._SNAPSHOT_SINK is None
    with keep_snapshots() as store:
        trajs = [agent.run(_env(s)).to_dict() for s in range(4)]
        assert MS._SNAPSHOT_SINK is not None
    assert MS._SNAPSHOT_SINK is None
    assert set(store) == {t["task_id"] for t in trajs}
    for t in trajs:
        snaps = store[t["task_id"]]
        assert [s.step for s in snaps] == [st["step"] for st in t["steps"]]
        for s, st in zip(snaps, t["steps"]):
            assert s.a0[0] == st["tool"] and s.route is not None and s.stop is not None
            assert (s.refl is not None) == (s.step > 0)          # the reflection gate is not asked at step 0
            # the snapshot's records ARE the logged decision records (they carry the one-step labels)
            if s.refl is not None:
                assert s.refl.gold_aux == next(d for d in st["decisions"] if d["dp"] == "reflection")["gold_aux"]
    # continuing from a snapshot steps a copy: the snapshot env keeps its state, two continuations are independent
    t = max(trajs, key=lambda x: len(x["steps"]))
    snap = store[t["task_id"]][len(t["steps"]) // 2]
    n_calls = len(snap.env.calls)
    oracle = JevAgent(SimulatedLLM("oracle", error_rate=0.0, simulated_latency_ms=1), HarnessConfig(name="llm_only"))
    ro1 = continue_from_snapshot(oracle, snap, None, ledger=Ledger())
    ro2 = continue_from_snapshot(oracle, snap, None, ledger=Ledger())
    assert ro1.error is None and ro1.source == "snapshot" and ro1.start_step == snap.step and ro1.prefix_match is None
    assert ro1.success and ro2.success and ro1.actions == ro2.actions and len(snap.env.calls) == n_calls
    assert len(snap.messages) == len(snap.messages)       # the message list is copied, never extended
    # the oracle from the step-0 snapshot reproduces the oracle's full episode
    s0 = store[t["task_id"]][0]
    full = oracle.run(_env(int(t["task_id"].split("-")[2])))
    ro0 = continue_from_snapshot(oracle, s0, None, ledger=Ledger())
    assert [n for n, _ in ro0.actions] == [s.tool for s in full.steps] and ro0.success == full.success
    # a prefix replay of the same logged steps lands on the same state as the snapshot (JevGym determinism)
    k = snap.step
    rep = continue_episode(oracle, _env(int(t["task_id"].split("-")[2])), [(s["tool"], s["args"]) for s in t["steps"][:k]],
                           None, ledger=Ledger(), logged_obs=[s["observation"] for s in t["steps"][:k]])
    assert rep.prefix_match is True and rep.actions == ro1.actions


def test_snapshot_interventions_mirror_the_one_step_branches():
    from jevharness.harness.measure import continue_from_snapshot, keep_snapshots

    agent = _measure_agent(err=0.0)
    seed = _seed_of("multi")
    with keep_snapshots() as store:
        t = agent.run(_env(seed)).to_dict()
    snaps = store[t["task_id"]]
    fin = snaps[-1]
    assert fin.finish and fin.a0[0] == "finish"
    cont = _agent(err=0.0)
    # reject at the finish snapshot: the note goes into the history, the step is consumed, the episode still succeeds
    ro = continue_from_snapshot(cont, fin, "reject", ledger=Ledger())
    assert ro.error is None and ro.start_step == fin.step + 1 and ro.success
    msgs, _ = cont.llm.seen[0]  # type: ignore[attr-defined]
    assert STOP_REJECT_NOTE in msgs[-1]["content"] and msgs[-1]["role"] == "tool"
    assert all(m.get("extra_content") is None for m in msgs)     # simulated calls carry no provider extras
    # reflection at a middle snapshot
    mid = snaps[len(snaps) // 2]
    cont = _agent(err=0.0)
    ro = continue_from_snapshot(cont, mid, "reflection", ledger=Ledger())
    assert ro.error is None and ro.reflection is not None and ro.success
    assert cont.llm.seen[0][0][-1]["content"] == REFLECT_PROMPT  # type: ignore[attr-defined]
    # large at a middle snapshot: the first decision is the large model's
    large = Recording("sim-flash", error_rate=0.0, simulated_latency_ms=1)
    ro = continue_from_snapshot(_agent(err=0.0), mid, "large", ledger=Ledger(), large_llm=large)
    assert ro.error is None and ro.calls[0]["model"] == "sim-flash" and len(large.seen) == 1 and ro.success
    with pytest.raises(ValueError, match="LLM-only"):
        continue_from_snapshot(agent, mid, None)


def test_script_live_mode_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cf_validity as CV

    run = _measurement_run(tmp_path, n=6)
    out = tmp_path / "cfv_live"
    CV.main(["--run", str(run), "--mode", "live", "--n-tasks", "5", "--task-offset", "1", "--n", "6", "--min-helps", "2",
             "--min-not", "2", "--reps", "2", "--out", str(out), "--workers", "2", "--budget-usd", "100",
             "--s1", '{"type": "simulated", "accuracy": 0.8}', "--sample-negatives", "1.0"])
    meas = [json.loads(ln) for ln in (out / "measure_arm.jsonl").read_text().splitlines() if ln.strip()]
    assert len(meas) == 5 and all(m["config"] == "meta_shadow_cfv" for m in meas)
    assert sorted(int(m["task_id"].split("-")[2]) for m in meas) == [1, 2, 3, 4, 5]      # --task-offset 1
    assert all(m["meta"]["config"]["measure"]["cf"]["placebo"] for m in meas)
    recs = [json.loads(ln) for ln in (out / "rollouts.jsonl").read_text().splitlines() if ln.strip()]
    sub = json.loads((out / "subset.json").read_text())
    assert sub["mode"] == "live" and len(recs) == 4 * len(sub["subset"]) and len(sub["subset"]) > 0
    assert all(r["source"] == "snapshot" and r["error"] is None and r["prefix_match"] is None for r in recs)
    # every opportunity comes from the live measurement arm, not from the source log
    assert {r["task_id"] for r in recs} <= {m["task_id"] for m in meas}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["simulated"] and summary["sources"] == ["snapshot"] and summary["run"]["mode"] == "live"
    assert summary["run"]["n_tasks"] == 5 and summary["run"]["task_offset"] == 1 and summary["run"]["measurement_usd"] == 0.0
    assert summary["table"]["overall"]["n"] == len(sub["subset"])
    assert "SIMULATED" in capsys.readouterr().out
    # measurement_config: the arm's loop parameters, three shadow gates, CF with placebo, no probes
    cfg = CV.measurement_config({"max_steps": 9, "obs_truncate_chars": 700, "memory_write": {"mode": "shadow"},
                                 "measure": {"probes": {"stop": ["T5"]}}})
    assert cfg.max_steps == 9 and cfg.obs_truncate_chars == 700 and cfg.is_measurement_arm()
    assert cfg.shadow_points() == [DP for DP in cfg.shadow_points()] and len(cfg.shadow_points()) == 3
    assert cfg.measure.probes is None and cfg.measure.cf is not None and cfg.measure.cf.placebo
