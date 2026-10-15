"""Shadow mode, threshold theory (Prop. 1 tests) and resume-retries-errors. Offline: simulated LLM / System-1."""

from __future__ import annotations

import glob
import json
import math
from typing import Any

import numpy as np
import pytest
import yaml
from pytest import approx

from jevharness.accounting import Ledger
from jevharness.bench import threshold_theory as T
from jevharness.bench.env import Env, StepResult
from jevharness.bench.metrics import decision_records, ece, episode_table
from jevharness.bench.runner import _done_keys, expand_arms, run_experiment
from jevharness.bench.suites.jevgym import JevGymEnv
from jevharness.decision.simulated import OracleBackend, SimulatedBackend
from jevharness.harness.agent import JevAgent, usage_summary
from jevharness.harness.config import DP, VALID_MODES, DPConfig, HarnessConfig, Mode
from jevharness.harness.trajectory import Trajectory, load_trajectories
from jevharness.llm.base import ChatResponse, ToolCall
from jevharness.llm.scripted import ScriptedLLM
from jevharness.llm.simulated import SimulatedLLM

SHADOW_ALL = dict(tool_select=DPConfig(mode=Mode.SHADOW), tool_params=DPConfig(mode=Mode.SHADOW),
                  stop=DPConfig(mode=Mode.SHADOW), reflection=DPConfig(mode=Mode.SHADOW),
                  routing=DPConfig(mode=Mode.SHADOW))


def _run(cfg: HarnessConfig, n: int = 12, s1: Any = None, err: float = 0.25, **env_kw: Any) -> list[dict]:
    llm = SimulatedLLM("scripted", error_rate=err, simulated_latency_ms=1000)
    if s1 is None and cfg.uses_s1():
        s1 = SimulatedBackend(accuracy=0.8, seed=3, price_per_m_input=0.042)
    agent = JevAgent(llm, cfg, s1=s1)
    return [agent.run(JevGymEnv(seed=i, strict=True, **env_kw)).to_dict() for i in range(n)]


def _llm_decisions(t: dict) -> list[tuple]:
    return [(s["tool"], json.dumps(s["args"], sort_keys=True), s["llm_calls"]) for s in t["steps"]]


# ====================================================================== shadow mode
def test_shadow_modes_are_valid_where_meaningful():
    for dp in (DP.TOOL_SELECT, DP.TOOL_PARAMS, DP.STOP, DP.ACTION, DP.REFLECTION, DP.ROUTING):
        assert Mode.SHADOW in VALID_MODES[dp]
    # memory_write: shadow asks the gate, logs it and stores nothing (META_GOLD_SPEC §3.6); memory_read has no shadow
    assert Mode.SHADOW in VALID_MODES[DP.MEMORY_WRITE]
    with pytest.raises(ValueError):
        HarnessConfig(memory_read=DPConfig(mode=Mode.SHADOW)).validate_modes()
    cfg = HarnessConfig(name="s", **SHADOW_ALL)
    assert cfg.uses_s1() and "tool_select:shadow" in cfg.short()
    with pytest.raises(ValueError, match="no backend"):
        JevAgent(SimulatedLLM("scripted"), cfg)          # shadow needs a System-1 backend
    JevAgent(SimulatedLLM("scripted"), cfg, s1=OracleBackend())   # routing shadow needs no small LLM


def test_shadow_arm_is_llm_only_plus_logged_s1_answers():
    base = _run(HarnessConfig(name="llm_only"))
    sh = _run(HarnessConfig(name="jev_shadow", **SHADOW_ALL))
    # identical episodes: success, executed tool calls, number of LLM calls per step
    assert [t["result"]["success"] for t in sh] == [t["result"]["success"] for t in base]
    assert 0 < sum(t["result"]["success"] for t in base) < len(base)       # the LLM errs: a non-trivial test
    assert [_llm_decisions(t) for t in sh] == [_llm_decisions(t) for t in base]
    assert all(t["error"] is None for t in sh)
    # cost: the shadow arm costs exactly what the LLM-only harness costs; the overhead is itemised
    eb, es = episode_table(base), episode_table(sh)
    assert es["cost_usd"] == approx(eb["cost_usd"]) and es["s1_cost_usd"] == 0.0 and es["s1_calls"] == 0.0
    for t in sh:
        u = t["usage"]
        assert u["shadow"]["cost_usd"] > 0 and u["shadow"]["calls"] > 0
        assert u["total_with_shadow"]["cost_usd"] == approx(u["total"]["cost_usd"] + u["shadow"]["cost_usd"])
        tagged = [c for c in t["calls"] if (c.get("meta") or {}).get("shadow")]
        assert len(tagged) == u["shadow"]["calls"] and all(c["component"] == "s1" for c in tagged)
        assert sum(c["cost_usd"] for c in t["calls"]) == approx(u["total_with_shadow"]["cost_usd"])
        assert "s1" not in u and set(u["by_decision_point"]) <= {"tool_select"}
    # decision logs: both answers, gold, measured costs
    recs = decision_records(sh)
    by = {dp: [r for r in recs if r["dp"] == dp] for dp in ("tool_select", "tool_params", "stop", "reflection",
                                                               "routing")}
    assert all(by.values()) and all(r["mode"] == "shadow" and r["shadow"] and r["deferred"] for r in recs)
    for r in by["tool_select"]:
        assert r["s1_value"] in r["options"] and 0 < r["s1_conf"] <= 1 and r["s1_dist"]
        assert r["s2_value"] == r["final_value"] and r["gold"] is not None
        assert r["s2_correct"] == (r["s2_value"] == r["gold"])
        assert r["s1_cost_usd"] > 0 and r["s1_latency_ms"] > 0 and r["s2_cost_usd"] > 0 and r["s2_latency_ms"] > 0
    assert any(not r["s2_correct"] for r in by["tool_select"])              # LLM errors are visible
    assert any(r["s1_value"] != r["gold"] for r in by["tool_select"])      # and System-1 errors
    for r in by["stop"]:
        assert isinstance(r["s1_value"], bool) and isinstance(r["gold"], bool)
        assert r["s1_value"] == (r["s1_dist"]["yes"] >= 0.5)               # argmax, so s1_conf refers to it
        assert r["s2_correct"] == (r["s2_value"] == r["gold"])
    assert any(r["gold"] is not None and r["s2_correct"] is not None for r in by["tool_params"])
    assert all(r["final_value"] is False for r in by["reflection"])
    assert all(r["final_value"] == "large" for r in by["routing"])


def test_shadow_system1_failure_never_changes_the_episode():
    class Down(SimulatedBackend):
        def _decide(self, state, questions):
            raise RuntimeError("503 Service Unavailable")

    base = _run(HarnessConfig(name="llm_only"), n=4)
    sh = _run(HarnessConfig(name="shadow", **SHADOW_ALL), n=4, s1=Down())
    assert [t["result"]["success"] for t in sh] == [t["result"]["success"] for t in base]
    assert all(t["error"] is None and t["meta"]["shadow_s1_errors"] for t in sh)
    assert all(d["s1_value"] is None for t in sh for s in t["steps"] for d in s["decisions"])


class Corridor(Env):
    """Walk right to cell n; the scripted LLM sometimes waits (a wrong action)."""

    def __init__(self, seed: int = 0, n: int = 4):
        self.n, self.task_id, self.instruction = n, f"corridor-{seed}", f"Walk to cell {n}."
        self.pos = 0

    def reset(self) -> str:
        self.pos = 0
        return "You are at cell 0."

    def admissible_actions(self) -> list[str]:
        return ["wait", "step right"] + (["step left"] if self.pos > 0 else [])

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        self.pos += {"step right": 1, "step left": -1}.get(args.get("action"), 0)
        return StepResult(f"You are at cell {self.pos}.", done=self.pos >= self.n)

    def oracle_action(self):
        return ("act", {"action": "step right"})

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        return {"success": self.pos >= self.n, "score": float(self.pos >= self.n)}


def test_action_shadow_matches_llm_only():
    def llm() -> ScriptedLLM:
        seq = iter(["wait", "step right", "step right", "wait", "step right", "step right"] * 5)
        return ScriptedLLM("scripted", policy=lambda m, t: ChatResponse(text=f"Action: {next(seq)}"))

    cfg = HarnessConfig(name="act_shadow", max_steps=8, action=DPConfig(mode=Mode.SHADOW))
    base = JevAgent(llm(), HarnessConfig(name="llm_only", max_steps=8)).run(Corridor()).to_dict()
    sh = JevAgent(llm(), cfg, s1=SimulatedBackend(accuracy=0.7, seed=1)).run(Corridor()).to_dict()
    assert [s["tool"] for s in sh["steps"]] == [s["tool"] for s in base["steps"]]
    assert [s["args"] for s in sh["steps"]] == [s["args"] for s in base["steps"]]
    assert sh["result"] == base["result"]
    ds = [d for s in sh["steps"] for d in s["decisions"]]
    assert ds and all(d["mode"] == "shadow" and d["gold"] == "step right" and d["s1_value"] is not None for d in ds)
    assert [d["s2_correct"] for d in ds][:2] == [False, True]


def test_usage_summary_without_shadow_is_the_ledger_summary():
    led = Ledger()
    assert usage_summary(led) == led.summary()


# ====================================================================== guard gold (coordinator request)
def test_guard_gold_judges_arguments_not_only_the_tool():
    """A proposal with the right tool but a wrong argument must get gold=False (env.oracle_accepts)."""
    env = JevGymEnv(seed=1, family="refund", strict=False)
    env.reset()
    gold_tool, gold_args = env.oracle_action()
    assert gold_args, "need a gold call with arguments"
    k = sorted(gold_args)[0]
    bad = {**gold_args, k: "WRONG"}
    assert JevAgent._proposal_gold(env, gold_tool, bad, gold_tool) is False
    assert JevAgent._proposal_gold(env, gold_tool, dict(gold_args), gold_tool) is True

    class NoAccepts(Env):          # envs without oracle_accepts keep the tool-name comparison
        def reset(self):
            return ""

        def step(self, tool, args):
            return StepResult("")

        def evaluate(self, final_answer):
            return {"success": False, "score": 0.0}

    assert JevAgent._proposal_gold(NoAccepts(), "t", {"x": "WRONG"}, "t") is True

    # in the harness: the LLM proposes the gold tool with a corrupted argument
    def policy(messages, tools):
        tool, args = env.oracle_action()
        return ChatResponse(text="", tool_calls=[ToolCall(id="c1", name=tool, arguments={**args, k: "WRONG"})])

    # (a step budget too small to finish makes every state doomed -- gold None -- so give the episode room)
    cfg = HarnessConfig(name="guard", max_steps=6, tool_select=DPConfig(mode=Mode.GUARD, tau=0.5))
    traj = JevAgent(ScriptedLLM("scripted", policy=policy), cfg, s1=OracleBackend()).run(env).to_dict()
    guards = [d for d in traj["steps"][0]["decisions"] if d["mode"] == "guard"]
    assert guards and guards[0]["gold"] is False and guards[0]["s1_value"] is False   # the oracle S1 sees it too
    assert guards[0]["gold_aux"]["canonical"] is False and guards[0]["gold_aux"]["verdict"] == "detour"


# ====================================================================== threshold theory
def _synth(n_tasks: int = 1500, per: int = 10, q_l=lambda c: 0.9, dc: float = 0.002, lo: float = 0.3,
           hi: float = 1.0, informative: bool = True, seed: int = 1) -> list[dict[str, Any]]:
    """Calibrated System-1 (P(correct | c) = c) unless ``informative`` is False (then c says nothing)."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_tasks):
        for k in range(per):
            c = float(rng.uniform(lo, hi))
            s1 = bool(rng.random() < (c if informative else 0.75))
            s2 = bool(rng.random() < q_l(c))
            out.append({"arm": "a", "task_id": f"t{t}", "step": k, "mode": "shadow", "shadow": True, "c": c,
                        "s1_ok": s1, "s2_ok": s2, "acceptable": True, "final_ok": s2, "s1_cost": 1e-5,
                        "s1_lat_ms": 100.0, "s2_cost": dc, "s2_lat_ms": 1000.0, "episode_success": True})
    return out


def test_theory_recovers_known_tau_star_and_regret_is_zero_when_calibrated():
    # q_L = 0.9, dC = 0.002, V = 0.02 -> tau* = 0.9 - 0.1 = 0.8 (Prop. 1)
    res = T.analyze(_synth(), Vs=[0.02], n_boot=200, n_perm=300)
    case = res["by_V"][0]
    th = case["thresholds_all"]
    assert th["k"] == approx(0.1)
    for rule in T.RULES:
        assert th[rule] == approx(0.8, abs=0.03), rule
        assert rule == "prop1" or th[f"{rule}_crossing"] == approx(0.8, abs=0.03)
    assert res["cost"]["delta_c"] == approx(0.002) and res["cost"]["kappa"] == approx(1e-5)
    assert not res["q_L"]["a2"]["rejected"]                        # q_L does not depend on c
    assert res["reliability"]["ece"] < 0.03                         # calibrated by construction
    rg = case["regret"]
    assert rg["n_train"] > 0 and rg["n_test"] > 0
    assert rg["candidates"]["tau_hat"]["regret_over_V"] < 0.01      # ~0 regret on held-out tasks
    lo, hi = rg["candidates"]["tau_hat"]["regret_ci"]
    assert lo >= -1e-3 and hi / 0.02 < 0.01                         # even the CI is < 1 point of V
    # an informative confidence beats a random gate at the same rate
    assert case["voi"]["tau_hat"]["acc_gain_ci"][0] > 0


def test_theory_general_rule_when_q_l_depends_on_c():
    # q_L(c) = 0.6 + 0.35 c (correlated errors), dC/V = 0.1: accept iff c >= q_L(c) - 0.1 <=> c >= 0.5 / 0.65
    res = T.analyze(_synth(q_l=lambda c: 0.6 + 0.35 * c), Vs=[0.02], n_boot=200, n_perm=300)
    th = res["by_V"][0]["thresholds_all"]
    star = 0.5 / 0.65
    assert res["q_L"]["a2"]["rejected"] and res["q_L"]["a2"]["logit_slope"]["ci_lo"] > 0
    assert res["q_L"]["fit"]["direction"] == "increasing"
    assert th["general"] == approx(star, abs=0.035) and th["qL"] == approx(star, abs=0.035)
    assert th["prop1"] == approx(res["q_L"]["q_L"] - 0.1)          # marginal q_L: biased low here
    assert abs(th["general"] - star) < abs(th["prop1"] - star)


def test_theory_random_gate_voi_is_zero_when_confidence_is_uninformative():
    res = T.analyze(_synth(lo=0.5, informative=False), Vs=[0.02], n_boot=300, n_perm=300)
    assert res["reliability"]["auroc"] == approx(0.5, abs=0.03)
    case = res["by_V"][0]
    for name in ("tau_hat", "tau_default"):
        g = case["voi"][name]
        assert abs(g["acc_gain"]) < 0.01 and g["acc_gain_ci"][0] <= 0 <= g["acc_gain_ci"][1], name
    # S1 (75%) is worse than deferring (q_L - dC/V = 80%) everywhere: the general rule (almost) never accepts
    assert case["thresholds_all"]["general"] > 0.95


def test_theory_helpers_agree_with_metrics_and_definitions():
    rng = np.random.default_rng(0)
    c = rng.uniform(0, 1, 500)
    y = (rng.random(500) < c).astype(float)
    rel = T.reliability(c, y, 10)
    assert sum(b["n"] / 500 * abs(b["acc"] - b["conf"]) for b in rel["bins"]) == approx(ece(c, y, 10))
    ux, fy = T.isotonic_fit([0.1, 0.2, 0.3, 0.4], [1, 0, 1, 1])
    assert list(fy) == approx([0.5, 0.5, 1, 1]) and np.all(np.diff(fy) >= 0)
    recs = _synth(n_tasks=50, per=4)
    tr, te = T.split_tasks(recs, 0.5, seed=3)
    assert {r["task_id"] for r in tr}.isdisjoint({r["task_id"] for r in te}) and len(tr) + len(te) == len(recs)
    assert T.split_tasks(recs, 0.5, seed=3)[0] == tr
    fr = T.cascade_frontier(recs, [0.0, 0.8, math.inf])
    rows = {r["tau"]: r for r in fr["rows"]}
    a = np.array([r["c"] for r in recs])
    s1 = np.array([r["s1_ok"] for r in recs], float)
    s2 = np.array([r["s2_ok"] for r in recs], float)
    assert rows[0.0]["acc"] == approx(s1.mean()) and rows[math.inf]["acc"] == approx(s2.mean())
    assert rows[0.8]["acc"] == approx(np.where(a >= 0.8, s1, s2).mean())
    assert rows[0.8]["llm_calls"] == approx((a < 0.8).mean())
    assert rows[math.inf]["cost_usd"] == approx(0.002 + 1e-5)
    # counterfactual V: every episode with a wrong executed decision fails, every other one succeeds
    eps = [{"arm": "a", "task_id": f"e{i}", "final_ok": i % 3 != 0, "episode_success": i % 3 != 0} for i in range(30)]
    v = T.counterfactual_v(eps, task_value=2.0, n_boot=100)
    assert v["V"] == approx(2.0) and v["n_wrong"] == 10


def test_theory_cli_on_a_tiny_simulated_run(tmp_path, capsys):
    exp = {"name": "tiny_shadow", "suite": {"name": "jevgym", "n": 16, "n_distractors": 8, "strict": True},
           "llm": {"provider": "simulated", "model": "scripted", "error_rate": 0.25, "simulated_latency_ms": 1000},
           "s1": {"type": "simulated", "accuracy": 0.8, "latency_ms": 50, "price_per_m_input": 0.042}, "workers": 2,
           "arms": [{"name": "llm_only"},
                    {"name": "jev_shadow", "tool_select": {"mode": "shadow"}, "tool_params": {"mode": "shadow"},
                     "stop": {"mode": "shadow"}}]}
    run_experiment(exp, tmp_path, progress=False)
    run_dir = tmp_path / "tiny_shadow"
    base, sh = load_trajectories(run_dir / "llm_only.jsonl"), load_trajectories(run_dir / "jev_shadow.jsonl")
    succ = lambda ts: {t["task_id"]: t["result"]["success"] for t in ts}   # noqa: E731
    assert succ(sh) == succ(base) and episode_table(sh)["cost_usd"] == approx(episode_table(base)["cost_usd"])
    out = tmp_path / "theory.json"
    T.main([str(run_dir), "--arm", "jev_shadow", "--dp", "tool_select", "--V", "0.1", "1", "--n-boot", "100",
            "--n-perm", "100", "--format", "json", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["meta"]["simulated"] is True and res["meta"]["arms"] == ["jev_shadow"]
    assert res["n_paired"] > 20 and res["q_L"]["n"] == res["n_paired"] and len(res["by_V"]) == 2
    assert res["cost"]["delta_c"] > 0 and res["cost"]["kappa"] > 0
    assert {"tau_hat", "tau_default", "tau_sweep_train"} <= set(res["by_V"][0]["regret"]["candidates"])
    capsys.readouterr()
    T.main([str(run_dir), "--dp", "stop", "--n-boot", "50", "--n-perm", "50"])     # default arm: the shadow one
    md = capsys.readouterr().out
    assert "SIMULATED RUN" in md and "q_L(c)" in md and "jev_shadow" in md
    T.main([str(run_dir), "--arm", "jev_shadow", "--dp", "tool_params", "--V", "1", "--n-boot", "50",
            "--n-perm", "50", "--format", "latex"])
    assert r"\toprule" in capsys.readouterr().out


# ====================================================================== resume retries errored episodes
class _Flaky(SimulatedLLM):
    """Raises a 429 on every call while ``down`` is set."""

    down = True

    def _chat(self, messages, tools=None, **kw):
        if self.down:
            raise RuntimeError("429 Too Many Requests")
        return super()._chat(messages, tools, **kw)


def _resume_exp() -> dict[str, Any]:
    return {"name": "resume", "suite": {"name": "jevgym", "n": 4, "n_distractors": 4},
            "llm": {"provider": "simulated", "model": "scripted", "error_rate": 0.0}, "workers": 1,
            "arms": [{"name": "llm_only"}]}


def test_resume_reruns_errored_episodes_and_loading_keeps_the_last_record(tmp_path):
    llm = _Flaky("scripted", error_rate=0.0, simulated_latency_ms=10)
    path = tmp_path / "resume" / "llm_only.jsonl"
    run_experiment(_resume_exp(), tmp_path, llm=llm, progress=False)
    first = load_trajectories(path)
    assert len(first) == 4 and all(t["error"] and "429" in t["error"] for t in first)
    assert _done_keys(path) == set() and len(_done_keys(path, retry_errors=False)) == 4
    llm.down = False
    run_experiment(_resume_exp(), tmp_path, llm=llm, progress=False)
    assert len(load_trajectories(path, dedupe=False)) == 8                  # re-run records were appended
    recs = load_trajectories(path)
    assert len(recs) == 4 and all(t["error"] is None and t["result"]["success"] for t in recs)
    assert [t["task_id"] for t in recs] == [t["task_id"] for t in first]   # order of first appearance
    assert episode_table(recs)["n"] == 4 and episode_table(recs)["errors"] == 0
    run_experiment(_resume_exp(), tmp_path, llm=llm, progress=False)       # nothing left to do
    assert len(load_trajectories(path, dedupe=False)) == 8


def test_resume_retry_can_be_disabled(tmp_path, monkeypatch):
    llm = _Flaky("scripted", error_rate=0.0, simulated_latency_ms=10)
    path = tmp_path / "resume" / "llm_only.jsonl"
    run_experiment(_resume_exp(), tmp_path, llm=llm, progress=False)
    llm.down = False
    run_experiment(_resume_exp(), tmp_path, llm=llm, progress=False, retry_errors=False)
    assert len(load_trajectories(path, dedupe=False)) == 4
    monkeypatch.setenv("JEV_RETRY_ERRORS", "0")
    run_experiment(_resume_exp(), tmp_path, llm=llm, progress=False)
    assert len(load_trajectories(path, dedupe=False)) == 4
    run_experiment({**_resume_exp(), "retry_errors": True}, tmp_path, llm=llm, progress=False)   # YAML wins
    assert len(load_trajectories(path, dedupe=False)) == 8
    from jevharness.bench.runner import main as runner_main

    cfg = tmp_path / "exp.yaml"
    cfg.write_text(yaml.safe_dump({**_resume_exp(), "llm": {"provider": "simulated", "model": "scripted"}}))
    monkeypatch.delenv("JEV_RETRY_ERRORS")
    runner_main([str(cfg), "--out", str(tmp_path / "cli"), "--no-retry-errors"])
    assert len(load_trajectories(tmp_path / "cli" / "resume" / "llm_only.jsonl")) == 4


def test_done_keys_uses_the_latest_record_of_each_task(tmp_path):
    p = tmp_path / "arm.jsonl"
    Trajectory(task_id="a", config="arm", backend=None, llm="x", error="RuntimeError: 500").dump(p)
    Trajectory(task_id="b", config="arm", backend=None, llm="x").dump(p)
    assert _done_keys(p) == {"b"}
    Trajectory(task_id="a", config="arm", backend=None, llm="x").dump(p)      # the retry succeeded
    Trajectory(task_id="b", config="arm", backend=None, llm="x", error="RuntimeError: 500").dump(p)
    assert _done_keys(p) == {"a"}
    assert [t["error"] for t in load_trajectories(p)] == [None, "RuntimeError: 500"]


# ====================================================================== configs
def test_every_experiment_config_validates_and_the_default_run_has_a_shadow_arm():
    n = 0
    for f in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
        exp = yaml.safe_load(open(f)) or {}
        if isinstance(exp, dict) and (exp.get("arms") or exp.get("sweeps")):
            expand_arms(exp)
            n += 1
    assert n >= 5
    arms = {a.name: a for a in expand_arms(yaml.safe_load(open("configs/experiments/jevgym_tool_select.yaml")))}
    sh = arms["jev_shadow"]
    assert set(sh.shadow_points()) == {DP.TOOL_SELECT, DP.TOOL_PARAMS, DP.STOP} and sh.uses_s1()
