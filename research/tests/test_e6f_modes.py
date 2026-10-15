"""E6f (exploratory; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6f", results/explore/e6f_modes/cells.md):
the trigger cells at tool choice, tool arguments and action. TRIGGER = System-1 decides whether the LLM runs one
verification call on its own proposal (``agent.py`` ``_verify_call`` / ``_verify_action``); ``always`` verifies every
proposal. Offline: the tiny tool env and stubs of ``tests/test_e6b_modes.py``, the tiny action env of
``tests/e6b_fingerprint.py``, the simulated backends on JevGym, the request-level wire canary, the configs, the readout
and the module map's tiles. Every mode that existed before E6f behaves and calls exactly as before: the four fingerprint
fixtures (``tests/test_e6b_modes.py``, ``test_e6d_modes.py``, ``test_e6e_modes.py`` and the E6e modes below)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "scripts"))

import e6b_fingerprint as FP  # noqa: E402
import test_e6b_modes as B  # noqa: E402
import test_e6c_action_guard as A  # noqa: E402

from jevharness.bench.suites.jevgym import JevGymEnv  # noqa: E402
from jevharness.decision.simulated import SimulatedBackend  # noqa: E402
from jevharness.harness.agent import JevAgent  # noqa: E402
from jevharness.harness.config import (DP, E6F_CELLS, UNDEFINED_CELLS, VALID_MODES, DPConfig,  # noqa: E402
                                       HarnessConfig, Mode)
from jevharness.harness.prompts import q_verify_action, q_verify_args, q_verify_tool  # noqa: E402
from jevharness.llm.base import ChatResponse, LLMClient, ToolCall  # noqa: E402

D = DPConfig
WRONG_TOOL = ("set_priority", {"level": 2})


# ------------------------------------------------------------------ the grid and the unchanged modes
def test_e6e_modes_behave_and_call_exactly_as_before() -> None:
    exp = json.loads(FP.FIXTURE_E6E.read_text())["digests"]
    got = FP.prior_e6e_digests()
    assert set(got) == set(exp) and {k for k in exp if got[k] != exp[k]} == set()


def test_grid_after_e6f_every_cell_is_defined() -> None:
    assert E6F_CELLS == {(DP.TOOL_SELECT, Mode.TRIGGER), (DP.TOOL_PARAMS, Mode.TRIGGER), (DP.ACTION, Mode.TRIGGER)}
    assert all(m in VALID_MODES[dp] for dp, m in E6F_CELLS) and UNDEFINED_CELLS == {}
    assert all(Mode.ALWAYS in VALID_MODES[dp] for dp in (DP.TOOL_SELECT, DP.TOOL_PARAMS, DP.ACTION))
    with pytest.raises(ValueError, match="verifies the calls the LLM writes"):
        HarnessConfig(tool_select=D(mode=Mode.CASCADE), tool_params=D(mode=Mode.TRIGGER)).validate_modes()
    HarnessConfig(tool_select=D(mode=Mode.TRIGGER), tool_params=D(mode=Mode.ALWAYS)).validate_modes()
    assert HarnessConfig(tool_params=D(mode=Mode.TRIGGER)).uses_s1()
    assert not HarnessConfig(action=D(mode=Mode.ALWAYS)).uses_s1()


# ------------------------------------------------------------------ stubs
class AcceptsEnv(B.TinyEnv):
    """The tiny tool env with ``oracle_accepts`` (the call is the plan's next call), so the call-level gold exists."""

    def oracle_accepts(self, tool: str, args: dict[str, Any]) -> bool | None:
        nxt = self.oracle_action()
        return None if nxt is None else (tool, dict(args)) == (nxt[0], dict(nxt[1]))


class VerifyS2(LLMClient):
    """System-2 stub. A first proposal is ``propose[step]`` (else the oracle call); on a "[Check]" note it ``fix``es
    (the oracle call; when forced, the oracle's arguments for the forced tool), ``keep``s the proposal, ``break``s it
    (the wrong tool, or for a forced call a wrong reason) or replies with ``text`` and no call. Records requests."""

    provider = "scripted"

    def __init__(self, propose: dict[int, tuple[str, dict[str, Any]]] | None = None, check: str = "fix",
                 model: str = "s2"):
        super().__init__(model)
        self.propose, self.check = dict(propose or {}), check
        self.requests: list[dict[str, Any]] = []
        self.env: Any = None
        self.last: tuple[str, dict[str, Any]] | None = None

    def bind_env(self, env: Any) -> None:
        self.env = env

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        forced = ((kw.get("tool_choice") or {}).get("function") or {}).get("name")
        last = str(messages[-1].get("content") or "")
        self.requests.append({"tools": [t["function"]["name"] for t in tools or []], "tool_choice": forced,
                              "last": last})
        oracle = self.env.oracle_action() or B.FINISH
        if last.startswith("[Check]"):
            if self.check == "text":
                return ChatResponse(text="The proposed call is right.", input_tokens=100, output_tokens=5)
            if self.check == "keep":
                name, args = self.last
            elif self.check == "break":
                name, args = (forced, {**self.last[1], "reason": "late"}) if forced else WRONG_TOOL
            else:
                name, args = oracle
                if forced:
                    name, args = forced, (dict(oracle[1]) if oracle[0] == forced else dict(self.last[1]))
        else:
            name, args = self.propose.get(self.env.i, oracle)
            self.last = (name, dict(args))
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"c{len(self.requests)}", name=name, arguments=dict(args))],
                            input_tokens=100, output_tokens=10, latency_ms=5.0)


def _run(cfg: HarnessConfig, s1: Any, llm: VerifyS2, plan: str = "lookup_cancel") -> dict[str, Any]:
    t = JevAgent(llm, cfg.validate_modes(), s1=s1).run(AcceptsEnv(plan)).to_dict()
    assert t["error"] is None, t["error"]
    return t


def _dec(t: dict[str, Any], dp: str) -> list[dict[str, Any]]:
    return [d for s in t["steps"] for d in s["decisions"] if d["dp"] == dp and d["mode"] in ("trigger", "always")]


def _calls(t: dict[str, Any], component: str, dp: str | None = None, step: int | None = None) -> list[dict[str, Any]]:
    return [c for c in t["calls"] if c["component"] == component and (dp is None or c["decision_point"] == dp)
            and (step is None or c["step"] == step)]


# ------------------------------------------------------------------ tool choice
def test_tool_trigger_fires_and_the_llm_revises_its_proposal_once() -> None:
    cfg = HarnessConfig(name="ts_trigger", tool_select=D(mode=Mode.TRIGGER, tau=0.5))
    llm = VerifyS2(propose={1: WRONG_TOOL})
    t = _run(cfg, B.S1Stub(), llm)          # System-1 says yes at 0.95: every proposal is verified
    assert t["result"]["success"] and [s["tool"] for s in t["steps"]] == ["lookup_order", "cancel_order", "finish"]
    d = _dec(t, "tool_select")
    assert [x["mode"] for x in d] == ["trigger"] * 3 and all(x["s1_value"] is True for x in d)
    assert d[1]["question"] == q_verify_tool(*WRONG_TOOL)
    assert d[1]["s2_value"] == "set_priority" and d[1]["final_value"] == "cancel_order"
    assert d[1]["gold"] is True and d[1]["gold_aux"]["outcome"] == "fixed" and d[1]["gold_aux"]["changed"]
    assert d[0]["gold"] is False and d[0]["gold_aux"]["outcome"] == "confirmed_right"
    assert d[1]["gold_aux"]["final"][0] == "cancel_order" and d[1]["gold_aux"]["outcome_call"] == "fixed"
    # one System-1 question and two LLM calls (proposal + verification) per step, all at tool_select
    assert all(len(_calls(t, "s1", "tool_select", s)) == 1 and len(_calls(t, "s2", "tool_select", s)) == 2
               for s in range(3))
    checks = [r for r in llm.requests if r["last"].startswith("[Check]")]
    assert len(checks) == 3 and all(len(r["tools"]) == 4 and r["tool_choice"] is None for r in checks)
    assert "`set_priority(" in checks[1]["last"] and d[1]["s2_cost_usd"] is not None


def test_tool_trigger_below_tau_runs_the_proposal_unchecked() -> None:
    cfg = HarnessConfig(name="ts_trigger", tool_select=D(mode=Mode.TRIGGER, tau=0.5))
    llm = VerifyS2(propose={1: WRONG_TOOL})
    t = _run(cfg, B.S1Stub(yes={"tool_select": False}), llm)       # p_yes 0.05 < 0.5
    d = _dec(t, "tool_select")
    assert all(x["s1_value"] is False and x["gold_aux"]["fired"] is False and x["gold_aux"]["outcome"] is None
               for x in d)
    assert d[1]["final_value"] == "set_priority" and d[1]["gold"] is True
    assert not any(r["last"].startswith("[Check]") for r in llm.requests) and not t["result"]["success"]
    assert all(len(_calls(t, "s2", "tool_select", s["step"])) == 1 for s in t["steps"])


def test_tool_always_verifies_every_proposal_without_system1() -> None:
    cfg = HarnessConfig(name="ts_verify_all", tool_select=D(mode=Mode.ALWAYS))
    llm = VerifyS2(check="break")
    t = JevAgent(llm, cfg.validate_modes()).run(AcceptsEnv("lookup_cancel")).to_dict()
    assert t["error"] is None and not _calls(t, "s1")
    d = _dec(t, "tool_select")
    assert d[0]["mode"] == "always" and d[0]["s1_value"] is None and d[0]["gold_aux"]["fired"]
    assert d[0]["gold_aux"]["outcome"] == "broke" and d[0]["final_value"] == "set_priority"
    assert len(_calls(t, "s2", "tool_select", 0)) == 2


def test_a_verification_reply_without_a_call_leaves_the_proposal_standing() -> None:
    cfg = HarnessConfig(name="ts_trigger", tool_select=D(mode=Mode.TRIGGER, tau=0.5))
    t = _run(cfg, B.S1Stub(), VerifyS2(check="text"))
    d = _dec(t, "tool_select")
    assert all(x["gold_aux"]["verify_reply"] == "text" and x["gold_aux"]["outcome"] == "confirmed_right" for x in d)
    assert t["result"]["success"] and "empty_llm_responses" not in t["meta"]     # never read as an empty finish


# ------------------------------------------------------------------ tool arguments
def test_argument_trigger_forces_the_proposed_tool_and_fixes_the_arguments() -> None:
    cfg = HarnessConfig(name="tp_trigger", tool_params=D(mode=Mode.TRIGGER, tau=0.9))
    wrong = ("cancel_order", {"order_id": "ORD-1234", "reason": "late"})
    llm = VerifyS2(propose={1: wrong})
    t = _run(cfg, B.S1Stub({"tool_params": 0.95}), llm)
    assert t["result"]["success"] and t["steps"][1]["args"] == {"order_id": "ORD-1234", "reason": "fraud"}
    d = _dec(t, "tool_params")
    assert len(d) == 3 and d[1]["question"] == q_verify_args(*wrong)
    assert json.loads(d[1]["s2_value"]) == wrong[1] and json.loads(d[1]["final_value"])["reason"] == "fraud"
    assert d[1]["gold"] is True and d[1]["gold_aux"]["outcome"] == "fixed" and d[0]["gold"] is False
    checks = [r for r in llm.requests if r["last"].startswith("[Check]")]
    assert [r["tool_choice"] for r in checks] == ["lookup_order", "cancel_order", "finish"]
    assert all(len(r["tools"]) == 1 for r in checks)
    assert all(len(_calls(t, "s2", "tool_params", s)) == 1 and len(_calls(t, "s1", "tool_params", s)) == 1
               for s in range(3))
    # the trigger is on p_yes: 0.85 < 0.9 never fires
    t2 = _run(cfg, B.S1Stub({"tool_params": 0.85}), VerifyS2(propose={1: wrong}))
    assert not any(x["gold_aux"]["fired"] for x in _dec(t2, "tool_params")) and not t2["result"]["success"]


def test_argument_always_can_break_a_right_call() -> None:
    cfg = HarnessConfig(name="tp_verify_all", tool_params=D(mode=Mode.ALWAYS))
    t = JevAgent(VerifyS2(check="break"), cfg.validate_modes()).run(AcceptsEnv("cancel")).to_dict()
    d = _dec(t, "tool_params")
    assert d[0]["mode"] == "always" and d[0]["gold_aux"]["outcome"] == "broke" and not t["result"]["success"]


# ------------------------------------------------------------------ action
class ActVerifyS2(A.ActS2):
    """The E6c action stub with a "[Check]" branch: ``check_answer`` (else the plan's action); ``""`` = empty reply."""

    def __init__(self, first: dict[int, str] | None = None, check_answer: str | None = None):
        super().__init__(first=first)
        self.check_answer = check_answer

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        last = str(messages[-1].get("content") or "")
        if "[Check]" in last:
            self.requests.append(last)
            a = A.PLAN[min(self.env.t, 2)] if self.check_answer is None else self.check_answer
            return ChatResponse(text=f"Checked.\nAction: {a}" if a else "", input_tokens=100, output_tokens=10)
        return super()._chat(messages, tools, **kw)


def test_action_trigger_fires_on_a_doubtful_proposal_and_the_llm_fixes_it() -> None:
    cfg = HarnessConfig(name="jev_action_trigger", max_steps=8, action=D(mode=Mode.TRIGGER, tau=0.5))
    s1 = A.GuardS1(lambda a: 0.9 if a == "look" else 0.1)        # "needs a second look" only for `look`
    llm = ActVerifyS2(first={1: "look"})
    t = JevAgent(llm, cfg.validate_modes(), s1=s1).run(FP.TinyActionEnv(0)).to_dict()
    assert t["error"] is None and t["result"]["success"]
    d = _dec(t, "action")
    assert [x["s1_value"] for x in d] == [False, True, False] and d[1]["question"] == q_verify_action("look")
    assert d[1]["s2_value"] == "look" and d[1]["final_value"] == A.PLAN[1] and d[1]["gold_aux"]["outcome"] == "fixed"
    assert d[0]["gold"] is False and d[1]["gold"] is True and d[0]["gold_aux"]["outcome"] is None
    assert len([c for c in t["calls"] if c["component"] == "s2" and c["step"] == 1]) == 2
    assert sum("[Check]" in r for r in llm.requests) == 1 and "Admissible actions:" in llm.requests[1]


def test_action_always_and_an_empty_check_reply() -> None:
    cfg = HarnessConfig(name="action_verify_all", max_steps=8, action=D(mode=Mode.ALWAYS))
    t = JevAgent(ActVerifyS2(check_answer=""), cfg.validate_modes()).run(FP.TinyActionEnv(0)).to_dict()
    d = _dec(t, "action")
    assert t["result"]["success"] and len(d) == 3 and all(x["gold_aux"]["verify_reply"] == "empty" for x in d)
    assert all(x["final_value"] == x["s2_value"] and x["gold_aux"]["outcome"] == "confirmed_right" for x in d)
    t = JevAgent(ActVerifyS2(check_answer="look"), cfg.validate_modes()).run(FP.TinyActionEnv(0)).to_dict()
    assert _dec(t, "action")[0]["gold_aux"]["outcome"] == "broke" and not t["result"]["success"]


# ------------------------------------------------------------------ simulated backends on JevGym, golds, one ledger
ARMS = {"ts_trigger": dict(tool_select=D(mode=Mode.TRIGGER, tau=0.5)), "ts_verify_all": dict(tool_select=D(mode=Mode.ALWAYS)),
        "tp_trigger": dict(tool_params=D(mode=Mode.TRIGGER, tau=0.9)), "tp_verify_all": dict(tool_params=D(mode=Mode.ALWAYS))}


def _sim_runs(n: int = 4) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for arm, kw in {"llm_only": {}, **ARMS}.items():
        agent = JevAgent(FP.GateSimLLM("scripted", error_rate=0.2, simulated_latency_ms=1, seed=0),
                         HarnessConfig(name=arm, **kw), s1=SimulatedBackend(accuracy=0.8, seed=1))
        out[arm] = [agent.run(JevGymEnv(seed=i, families="v2", strict=True, n_distractors=20, distractor_style="near",
                                         noise_lines=5)).to_dict() for i in range(n)]
    return out


def test_jevgym_simulation_logs_golds_and_every_call_in_the_ledger() -> None:
    runs = _sim_runs()
    for arm, ts in runs.items():
        assert all(t["error"] is None for t in ts)
        if arm == "llm_only":
            continue
        dp = "tool_select" if arm.startswith("ts") else "tool_params"
        d = [x for t in ts for x in _dec(t, dp)]
        assert d and all(x["gold_status"] == "ok" and isinstance(x["gold"], bool) for x in d)
        fired = [x for x in d if x["gold_aux"]["fired"]]
        assert all(x["gold_aux"]["outcome"] in ("fixed", "broke", "confirmed_right", "confirmed_wrong", "revised_right",
                                                "revised_wrong") for x in fired)
        # one verification call per fired decision, at the point, in the ledger (beside the proposals)
        assert sum(len(_calls(t, "s2", dp)) for t in ts) >= len(fired) + (len(d) if dp == "tool_select" else 0)
        if arm.endswith("trigger"):
            assert sum(len(_calls(t, "s1", dp)) for t in ts) == len(d)
        else:
            assert not any(_calls(t, "s1") for t in ts) and len(fired) == len(d)
        for t in ts:
            u = t["usage"]["total"]["cost_usd"]
            assert u == pytest.approx(sum(c["cost_usd"] for c in t["calls"]))


# ------------------------------------------------------------------ no gold on the wire
@pytest.mark.parametrize("canary_tool", [True, False])
def test_verification_triggers_leak_no_oracle_value_on_the_wire(canary_tool: bool) -> None:
    import test_wire_canary as W
    from jevharness.decision.jev import JevBackend

    for kw in (dict(tool_select={"mode": "trigger", "tau": 0.5}), dict(tool_params={"mode": "trigger", "tau": 0.5}),
               dict(tool_select={"mode": "always"}, tool_params={"mode": "always"})):
        s1_all, s2_all, dumps = W.Wire(W.jev_answers), W.Wire(W.jev_answers), []
        for seed in (0, 3):
            real = JevGymEnv(seed=seed, families="v2", distractor_style="near", n_distractors=60, noise_lines=5,
                             strict=True)
            env = W.CanaryOracleEnv(real, canary_tool)
            s1_wire, s2_wire = W.Wire(W.s1_policy(real)), W.Wire(W.s2_policy(real))
            s1 = JevBackend(provider="openrouter", api_key="test-key", transport=s1_wire.transport())
            llm = W.compat_llm("gemini", "gemini-3.5-flash", s2_wire, temperature=0, reasoning_effort="minimal")
            cfg = HarnessConfig(name="e6f", max_steps=12, **kw)
            traj = JevAgent(llm, cfg, s1=s1 if cfg.uses_s1() else None).run(env)
            assert traj.error is None, traj.error
            s1_all.requests += s1_wire.requests
            s2_all.requests += s2_wire.requests
            dumps.append(json.dumps(traj.to_dict(), default=str))
        if HarnessConfig(**kw).uses_s1():
            s1_all.assert_clean(min_requests=2, key_names=False)
        s2_all.assert_clean(min_requests=4, key_names=False)
        # the golds were computed from the canaried oracle (the canary tool name reaches the log's oracle_tool)
        assert any(W.CANARY in d for d in dumps) or not canary_tool
        assert any('"gold_status": "ok"' in d and '"outcome"' in d for d in dumps)
        assert sum("[Check]" in json.dumps(r, default=str) for r in s2_all.requests) >= 2


def test_action_trigger_leaks_no_oracle_value_on_the_wire() -> None:
    import test_wire_canary as W
    from jevharness.decision.jev import JevBackend

    class CanaryActionEnv(FP.TinyActionEnv):
        def oracle_action(self) -> Any:
            return ("act", {"action": W._c("action")}) if self.t < 3 else None

    env = CanaryActionEnv(1)

    def s2(request: Any, body: dict[str, Any]) -> dict[str, Any]:
        return W.chat_completion(f"Thinking.\nAction: {A.PLAN[min(env.t, 2)]}")

    s1_wire, s2_wire = W.Wire(W.jev_answers), W.Wire(s2)
    s1 = JevBackend(provider="openrouter", api_key="test-key", transport=s1_wire.transport())
    llm = W.compat_llm("gemini", "gemini-3.5-flash", s2_wire, temperature=0, reasoning_effort="minimal")
    cfg = HarnessConfig(name="jev_action_trigger", max_steps=8, action=D(mode=Mode.TRIGGER, tau=0.5))
    traj = JevAgent(llm, cfg, s1=s1).run(env)        # Jev mock answers p_yes 0.7: every proposal is verified
    assert traj.error is None, traj.error
    s1_wire.assert_clean(min_requests=3)
    s2_wire.assert_clean(min_requests=6)
    assert W.CANARY in json.dumps(traj.to_dict(), default=str)


# ------------------------------------------------------------------ configs, readout, cells.md, the module map
def test_e6f_configs() -> None:
    import yaml

    import explore_e6f_modes as X
    from jevharness.bench.runner import expand_arms

    root = ROOT.parent
    exp = yaml.safe_load((root / "configs/explore/e6f_modes/eval.yaml").read_text())
    cfgs = {c.name: c.validate_modes() for c in expand_arms(exp)}
    assert set(cfgs) == set(X.SETTINGS["jevgym"]["arms"])
    assert cfgs["ts_trigger"].tool_select.mode == Mode.TRIGGER and cfgs["ts_trigger"].tool_select.tau == 0.5
    assert cfgs["tp_trigger"].tool_params.mode == Mode.TRIGGER and cfgs["tp_trigger"].tool_params.tau == 0.9
    assert cfgs["ts_verify_all"].tool_select.mode == Mode.ALWAYS and cfgs["tp_verify_all"].tool_params.mode == Mode.ALWAYS
    assert all(sum(cfgs[a].dp(d).mode != Mode.OFF for d in DP) == 1 for a in cfgs if a != "llm_only")
    assert exp["suite"] == {"name": "jevgym", "n": 200, "seed0": 301600, "families": "v2", "distractor_style": "near",
                            "n_distractors": 60, "noise_lines": 20, "strict": True}
    assert exp["llm"] == {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0,
                          "reasoning_effort": "minimal"} and exp["s1"] == {"type": "jev", "provider": "openrouter"}
    assert exp["schedule"] == {"mode": "task_major", "seed": 732} and "s1_cache" not in exp
    sm = yaml.safe_load((root / "configs/explore/e6f_modes/smoke.yaml").read_text())
    assert sm["suite"]["seed0"] == 900 and sm["n_tasks"] == 2 and sm["arms"] == exp["arms"]
    assert {k: sm[k] for k in ("llm", "s1", "workers")} == {k: exp[k] for k in ("llm", "s1", "workers")}
    al = yaml.safe_load((root / "configs/explore/e6f_alfworld/eval.yaml").read_text())
    e6c = yaml.safe_load((root / "configs/explore/e6c_alfworld/eval.yaml").read_text())
    assert {k: al[k] for k in ("suite", "n_tasks", "llm", "s1", "harness_defaults", "schedule")} == \
        {k: e6c[k] for k in ("suite", "n_tasks", "llm", "s1", "harness_defaults", "schedule")}
    acfg = {c.name: c.validate_modes() for c in expand_arms(al)}
    assert set(acfg) == set(X.SETTINGS["alfworld"]["arms"])
    assert acfg["jev_action_trigger"].action.mode == Mode.TRIGGER and acfg["jev_action_trigger"].action.tau == 0.5
    assert acfg["action_verify_all"].action.mode == Mode.ALWAYS
    asm = yaml.safe_load((root / "configs/explore/e6f_alfworld/smoke.yaml").read_text())
    assert asm["suite"]["split"] == "train" and asm["n_tasks"] == 2 and asm["arms"] == al["arms"]
    # the cells and thresholds of the readout are the configs'
    assert {(DP(dp), Mode(m)) for dp, m, _, _ in X.CELLS.values()} == set(E6F_CELLS)
    assert X.TAU == {"tool_select": 0.5, "tool_params": 0.9, "action": 0.5}
    assert (X.GEMINI_CAP, X.JEV_CAP, X.SAFETY) == (60.0, 2.0, 1.5)
    # the seed block is E6f's alone: no other config names it
    others = [p for p in (root / "configs").rglob("*.yaml") if "e6f_" not in str(p) and "seed0: 3016" in p.read_text()]
    assert others == []


def test_cells_md_is_generated_from_the_code() -> None:
    import explore_e6f_modes as X
    assert (ROOT.parent / "results/explore/e6f_modes/cells.md").read_text() == X.cells_markdown()


def _write_runs(tmp: Path) -> Path:
    runs = tmp / "runs"
    jg = runs / "e6f_modes"
    jg.mkdir(parents=True)
    for arm, ts in _sim_runs(6).items():
        (jg / f"{arm}.jsonl").write_text("".join(json.dumps(t, default=str) + "\n" for t in ts))
    al = runs / "e6f_alfworld_eval"
    al.mkdir()
    for arm, kw in (("llm_only", {}), ("jev_action_trigger", dict(action=D(mode=Mode.TRIGGER, tau=0.5))),
                    ("action_verify_all", dict(action=D(mode=Mode.ALWAYS)))):
        agent = JevAgent(FP.ActionLLM("scripted", error_rate=0.3, simulated_latency_ms=1, seed=0),
                         HarnessConfig(name=arm, max_steps=8, **kw), s1=SimulatedBackend(accuracy=0.8, seed=1))
        ts = [agent.run(FP.TinyActionEnv(i)).to_dict() for i in range(6)]
        import gzip
        (al / f"{arm}.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(t) + "\n" for t in ts).encode()))
    return runs


def test_e6f_readout_and_the_module_map_tiles(tmp_path: Path) -> None:
    import explore_e6f_modes as X
    import module_map as MM

    runs = _write_runs(tmp_path)
    res = X.summarize(runs, n_boot=200, seed=0)
    assert set(res["cells"]) == set(X.CELLS)
    for arm, c in res["cells"].items():
        assert c["state"] == "exploratory" and c["cmp"]["n"] == 6 and c["verdict"] in MM.VCODE
        v, av = c["verify"], c["always"]["verify"]
        assert v["n_proposals"] > 0 and 0.0 <= v["verified"] <= 1.0 and av["verified"] == 1.0
        assert v["n_verified"] >= sum(v[o] for o in X.OUTCOMES)        # outcomes need a gold
        assert c["points"]["acted"] == pytest.approx(v["verified"])     # the module map's "acted" = fired share
    assert res["references"]["alfworld"]["n_ok"] == 6 and res["references"]["jevgym"]["n_ok"] == 6
    md = X.to_markdown(res)
    assert "always-verify reference" in md and "jev_action_trigger" in md
    keys = X.to_tex_keys(res)
    assert keys["e6f/ts_trigger/state"] == "exploratory" and "e6f/tp_trigger/always/fixed" in keys
    assert "e6f/jev_action_trigger/verified" in keys and keys["e6f/count/run"] == "3"
    # the module map reads the summary into the exploratory tiles of the three cells (Flash map only)
    summ = tmp_path / "summary.json"
    summ.write_text(json.dumps(MM._clean(res), default=MM._json_default))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"task_ids": ["jevgym-x-200000-d60-near-strict"]}))
    mm = MM.ModuleMap(tmp_path / "none", n_boot=50, stage1=(), stage2=None, quiet=True, frozen=frozen,
                      primaries=tmp_path / "none.json", explore=(), e6b=None, e6d=None, e6e=None, e6f=summ).build()
    x = mm["explore_e6b"]["cells"]
    assert set(x) == {"flash/tool_select/trigger", "flash/tool_params/trigger", "flash/action/trigger"}
    assert all(c["explore"] == "E6f" for c in x.values())
    tex = MM.to_tex(mm)
    for dp in ("tool_select", "tool_params", "action"):
        assert f"\\defres{{mm/cell/flash/{dp}/trigger/state}}{{not run}}" in tex
        assert f"\\defres{{mm/xcell/flash/{dp}/trigger/explore}}{{E6f}}" in tex
        assert re.search(rf"\\defres{{mm/xcell/flash/{dp}/trigger/fig_saving}}{{-?\d+}}", tex)
    assert mm["counts"]["flash"]["undefined"] == 0 and mm["counts"]["flash"]["valid"] == 40
