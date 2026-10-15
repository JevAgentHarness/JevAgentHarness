"""E6c (exploratory; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6c"): the action guard, the last
meaningful cell of the design-space grid (``harness/config.py`` ``E6C_CELLS``; ``agent.py`` ``_guard_action``). Offline:
the tiny action-only env of ``tests/e6b_fingerprint.py``, a scripted System-2 that records every request, a System-1
stub with a fixed p_yes per proposed action, the simulated backends, and the request-level canary on mock
transports. The fingerprint test of ``tests/test_e6b_modes.py`` shows that the other action modes are unchanged."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "scripts"))

import e6b_fingerprint as FP  # noqa: E402

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.decision.base import DecisionBackend  # noqa: E402
from jevharness.decision.simulated import SimulatedBackend  # noqa: E402
from jevharness.decision.types import DecisionResult, Question, Usage, answer_from_distribution  # noqa: E402
from jevharness.harness.agent import JevAgent  # noqa: E402
from jevharness.harness.config import (ACTION_PENDING, DP, E6C_CELLS, UNDEFINED_CELLS, VALID_MODES,  # noqa: E402
                                       DPConfig, HarnessConfig, Mode)
from jevharness.harness.prompts import action_reject_note, q_action_guard  # noqa: E402
from jevharness.llm.base import ChatResponse, LLMClient  # noqa: E402

PLAN = FP.TinyActionEnv.PLAN


class GuardS1(DecisionBackend):
    """System-1 stub for the guard question: p_yes = ``p_yes(proposed action)``; records every question."""

    name = "s1guard"

    def __init__(self, p_yes: Callable[[str], float]):
        self.p_yes = p_yes
        self.max_options = 52
        self.requests: list[Question] = []

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        out = []
        for q in questions:
            self.requests.append(q)
            m = re.search(r"the action `([^`]*)`", q.text)
            p = float(self.p_yes(m.group(1) if m else ""))
            out.append(answer_from_distribution(q, {"yes": p, "no": 1.0 - p}))
        return DecisionResult(answers=out, usage=Usage(backend=self.name, model="stub", input_tokens=10,
                                                       latency_ms=1.0))


class ActS2(LLMClient):
    """System-2 stub for the action env: proposes ``first[t]`` (else the plan's action) at plan position t; after a
    rejection note it answers ``retry`` (else the plan's action). Records every request's last message."""

    provider = "scripted"

    def __init__(self, first: dict[int, str] | None = None, retry: str | None = None, model: str = "s2"):
        super().__init__(model)
        self.first, self.retry = dict(first or {}), retry
        self.requests: list[str] = []
        self.env: FP.TinyActionEnv | None = None

    def bind_env(self, env: FP.TinyActionEnv) -> None:
        self.env = env

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        assert self.env is not None and not tools
        last = str(messages[-1].get("content") or "")
        self.requests.append(last)
        right = PLAN[min(self.env.t, 2)]
        a = (self.retry or right) if "[Verifier]" in last else self.first.get(self.env.t, right)
        return ChatResponse(text=f"Thinking.\nAction: {a}", input_tokens=100, output_tokens=10, latency_ms=5.0)


def _guard(tau: float = 0.5) -> HarnessConfig:
    return HarnessConfig(name="jev_action_guard", max_steps=8, action=DPConfig(mode=Mode.GUARD, tau=tau))


def _run(cfg: HarnessConfig, s1: DecisionBackend, llm: LLMClient, i: int = 0) -> dict[str, Any]:
    t = JevAgent(llm, cfg.validate_modes(), s1=s1).run(FP.TinyActionEnv(i)).to_dict()
    assert t["error"] is None, t["error"]
    return t


def _guards(t: dict[str, Any], step: int) -> list[dict[str, Any]]:
    return [d for d in t["steps"][step]["decisions"] if d["dp"] == "action" and d["mode"] == "guard"]


def _calls(t: dict[str, Any], step: int, component: str) -> list[dict[str, Any]]:
    return [c for c in t["calls"] if c["step"] == step and c["component"] == component]


# ------------------------------------------------------------------ the grid
def test_action_guard_is_implemented_and_the_grid_is_complete() -> None:
    assert E6C_CELLS == {(DP.ACTION, Mode.GUARD)} and Mode.GUARD in VALID_MODES[DP.ACTION]
    assert ACTION_PENDING == {} and (DP.ACTION, Mode.GUARD) not in UNDEFINED_CELLS
    _guard().validate_modes()                                   # accepted (was "not implemented")
    HarnessConfig(action=DPConfig(mode=Mode.TRIGGER)).validate_modes()   # defined by E6f (tests/test_e6f_modes.py)
    with pytest.raises(ValueError, match="not supported"):
        HarnessConfig(action=DPConfig(mode=Mode.LLM)).validate_modes()


# ------------------------------------------------------------------ behaviour
def test_action_guard_vetoes_a_proposal_once_and_the_llm_regenerates() -> None:
    s1 = GuardS1(lambda a: 0.1 if a == "look" else 0.9)
    llm = ActS2(first={0: "look"})
    t = _run(_guard(0.5), s1, llm)
    assert t["result"]["success"] is True
    assert [s["args"]["action"] for s in t["steps"]] == PLAN               # the rewrite was executed
    (g0,) = _guards(t, 0)
    assert g0["question"] == q_action_guard("look") and g0["options"] is None
    assert g0["s1_value"] is False and g0["s2_value"] == "look" and g0["final_value"] is None and g0["retry"] == 0
    assert g0["gold"] is False and g0["gold_status"] == "ok"
    assert g0["gold_aux"]["gold_action"] == PLAN[0] and g0["gold_aux"]["rewrite_gold"] is True
    assert g0["gold_aux"]["rewrite_same"] is False and g0["s2_cost_usd"] is not None
    # the regeneration: the same prompt with the admissible list and the rejection note, at the action point
    assert len(llm.requests) == 4 and llm.requests[0].startswith("Admissible actions:")
    assert llm.requests[1].startswith("Admissible actions:") and llm.requests[1].endswith(action_reject_note("look"))
    assert t["steps"][0]["llm_calls"] == 2
    assert [c["decision_point"] for c in _calls(t, 0, "s2")] == ["action", "action"]
    assert [c["decision_point"] for c in _calls(t, 0, "s1")] == ["action"]
    for st in (1, 2):                                                       # accepted proposals: one call each
        (g,) = _guards(t, st)
        assert g["s1_value"] is True and g["final_value"] == PLAN[st] and g["gold"] is True
        assert "rewrite_same" not in g["gold_aux"] and t["steps"][st]["llm_calls"] == 1
    assert [q.decision_point for q in s1.requests] == ["action"] * 3
    assert all("gold" not in json.dumps(q.text) for q in s1.requests)
    # no other decision log at the action point (the guard replaces the "off" record, as the tool guard)
    assert {d["mode"] for s in t["steps"] for d in s["decisions"]} == {"guard"}


def test_action_guard_executes_the_rewrite_unchecked() -> None:
    s1 = GuardS1(lambda a: 0.0 if a == "look" else 1.0)
    t = _run(_guard(0.5), s1, ActS2(first={0: "look"}, retry="look"))
    assert t["steps"][0]["args"]["action"] == "look"                        # vetoed, regenerated, executed anyway
    assert len(_guards(t, 0)) == 1 and _guards(t, 0)[0]["gold_aux"]["rewrite_same"] is True
    assert _guards(t, 0)[0]["gold_aux"]["rewrite_gold"] is False
    assert t["steps"][0]["llm_calls"] == 2 and len(_calls(t, 0, "s1")) == 1


@pytest.mark.parametrize("tau,vetoes", [(0.6, 0), (0.61, 3), (0.0, 0), (1.0, 3)])
def test_action_guard_threshold_is_on_p_yes(tau: float, vetoes: int) -> None:
    """ok = p_yes >= tau, the tool guard's rule (p_yes 0.6 at every proposal)."""
    llm = ActS2()
    t = _run(_guard(tau), GuardS1(lambda a: 0.6), llm)
    gs = [g for st in range(len(t["steps"])) for g in _guards(t, st)]
    assert len(gs) == 3 and sum(g["s1_value"] is False for g in gs) == vetoes
    assert len(llm.requests) == 3 + vetoes and t["result"]["success"] is True


def test_action_guard_without_an_oracle_logs_no_gold() -> None:
    class NoOracle(FP.TinyActionEnv):
        def oracle_action(self) -> None:
            return None

    agent = JevAgent(ActS2(), _guard(0.5), s1=GuardS1(lambda a: 0.9))
    t = agent.run(NoOracle(0)).to_dict()
    assert t["error"] is None
    g = _guards(t, 0)[0]
    assert g["gold"] is None and g["gold_status"] == "no_oracle" and g["gold_aux"]["gold_action"] is None


# ------------------------------------------------------------------ simulated backends: golds, one ledger, readouts
def test_action_guard_runs_with_simulated_backends_golds_and_one_ledger() -> None:
    import module_map as MM

    llm = FP.ActionLLM("scripted", error_rate=0.3, simulated_latency_ms=5, seed=0)
    agent = JevAgent(llm, _guard(0.5), s1=SimulatedBackend(accuracy=0.8, seed=1))
    trajs = [agent.run(FP.TinyActionEnv(i)).to_dict() for i in range(6)]
    assert all(t["error"] is None for t in trajs), [t["error"] for t in trajs]
    recs = [r for r in M.decision_records(trajs) if r["dp"] == "action"]
    assert recs and {r["mode"] for r in recs} == {"guard"}
    assert all(r["gold_status"] == "ok" and isinstance(r["gold"], bool) for r in recs)
    assert any(r["s1_value"] is False for r in recs) and any(r["s1_value"] is True for r in recs)
    for t in trajs:   # every call in the episode's ledger, attributed to a decision point; LLM calls counted per step
        assert abs(t["usage"]["total"]["cost_usd"] - sum(c["cost_usd"] for c in t["calls"])) < 1e-9
        assert all(c["decision_point"] == "action" for c in t["calls"])
        assert sum(s["llm_calls"] for s in t["steps"]) == sum(c["component"] == "s2" for c in t["calls"])
        assert sum(c["component"] == "s1" for c in t["calls"]) == sum(
            d["mode"] == "guard" for s in t["steps"] for d in s["decisions"])
    # the readouts: the guard executes the LLM's action (coverage 0); activity = share of proposals vetoed
    assert M.delegation_ratio(recs, dps=("action",))["ratio"] == 0.0
    pm = MM.point_metrics(trajs, "action", "guard", {"mode": "guard", "tau": 0.5}, True)
    assert pm["acted_kind"] == "vetoed"
    assert pm["acted"] == pytest.approx(sum(r["s1_value"] is False for r in recs) / len(recs))
    assert pm["n_gold"] == len(recs) and 0.0 <= pm["acc"] <= 1.0


# ------------------------------------------------------------------ no gold on the wire
@pytest.mark.parametrize("tau", [0.5, 0.8])   # Jev mock answers p_yes 0.7: accept everything / veto everything
def test_action_guard_leaks_no_oracle_value_on_the_wire(tau: float) -> None:
    """The request-level canary of tests/test_wire_canary.py on the action guard: the oracle's gold action is a
    canary, System-1 speaks the Jev wire format and System-2 the OpenAI-compatible one behind mock transports; no
    canary in any outgoing request, and the canary does reach the gold side of the log."""
    import test_wire_canary as W
    from jevharness.decision.jev import JevBackend

    class CanaryActionEnv(FP.TinyActionEnv):
        def oracle_action(self) -> Any:
            return ("act", {"action": W._c("action")}) if self.t < 3 else None

    env = CanaryActionEnv(1)

    def s2(request: Any, body: dict[str, Any]) -> dict[str, Any]:
        return W.chat_completion(f"Thinking.\nAction: {PLAN[min(env.t, 2)]}")

    s1_wire, s2_wire = W.Wire(W.jev_answers), W.Wire(s2)
    s1 = JevBackend(provider="openrouter", api_key="test-key", transport=s1_wire.transport())
    llm = W.compat_llm("gemini", "gemini-3.5-flash", s2_wire, temperature=0, reasoning_effort="minimal")
    traj = JevAgent(llm, _guard(tau), s1=s1).run(env)
    assert traj.error is None, traj.error
    s1_wire.assert_clean(min_requests=3)
    s2_wire.assert_clean(min_requests=3 if tau < 0.7 else 6)
    assert W.CANARY in json.dumps(traj.to_dict(), default=str)


# ------------------------------------------------------------------ the E6c readout and the module map's tile
def _gdec(proposal: str, ok: bool, gold: str, rewrite: str | None = None) -> dict[str, Any]:
    aux: dict[str, Any] = {"v": 1, "src": "oracle_action", "gold_action": gold}
    if not ok:
        aux.update({"rewrite_same": rewrite == proposal, "rewrite_gold": rewrite == gold})
    p = 0.8 if ok else 0.2
    return {"dp": "action", "mode": "guard", "question": q_action_guard(proposal), "options": None, "s1_value": ok,
            "s1_conf": max(p, 1 - p), "s1_dist": {"yes": p, "no": 1 - p}, "deferred": False, "s2_value": proposal,
            "final_value": proposal if ok else None, "gold": proposal == gold, "gold_status": "ok", "gold_aux": aux,
            "retry": 0, "shadow": False}


def _rec(tid: str, arm: str, ok: bool, cost: float, decs: list[dict[str, Any]], wall: float = 10_000.0,
         s1_cost: float = 0.0) -> dict[str, Any]:
    import module_map as MM
    cfg = {dp: {"mode": "off", "tau": 0.8, "k": 3, "extra": {}} for dp in MM.DPS}
    if arm != "llm_only":
        cfg["action"] = {"mode": "guard", "tau": 0.5, "k": 3, "extra": {}}
    return {"task_id": tid, "config": arm, "backend": "jev" if arm != "llm_only" else None, "llm": MM.FLASH,
            "error": None, "wall_ms": wall,
            "steps": [{"step": i, "tool": "act", "args": {"action": d.get("final_value")}, "llm_calls": 1,
                       "decisions": [d]} for i, d in enumerate(decs)],
            "result": {"success": ok}, "final_answer": None,
            "usage": {"total": {"cost_usd": cost, "calls": 2}, "s1": {"cost_usd": s1_cost, "calls": len(decs)},
                      "s2": {"cost_usd": cost - s1_cost, "calls": len(decs)}},
            "meta": {"config": {"name": arm, **cfg},
                     "s1": {"describe": {"backend": "jev"} if arm != "llm_only" else None}}}


@pytest.fixture()
def e6c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import yaml

    import explore_e6c_alfworld as X
    runs = tmp_path / "runs"
    d = runs / X.RUN_EVAL
    d.mkdir(parents=True)
    tasks = [f"alfworld/valid_seen/pick_and_place_simple-Book-None-Desk-{i}/trial_T{i}" for i in range(10)]
    (d / "experiment.yaml").write_text(yaml.safe_dump({
        "name": X.RUN_EVAL, "n_tasks": 10, "suite": {"name": "alfworld", "n": 10},
        "llm": {"provider": "gemini", "model": "gemini-3.5-flash"}, "s1": {"type": "jev", "provider": "openrouter"},
        "arms": [{"name": "llm_only"}, {"name": X.ARM, "action": {"mode": "guard", "tau": 0.5}}]}))
    off = {"dp": "action", "mode": "off", "question": "next action", "options": PLAN, "s1_value": None,
           "s2_value": PLAN[0], "final_value": PLAN[0], "gold": PLAN[0], "deferred": False, "shadow": False}
    (d / "llm_only.jsonl").write_text("".join(json.dumps(_rec(t, "llm_only", i != 9, 0.02, [off], wall=8000.0)) + "\n"
                                              for i, t in enumerate(tasks)))
    # guard: one veto per episode on tasks 0-3 (a wrong proposal; rewrites right on 0-2), accepts elsewhere;
    # wins task 9, loses task 0; 10 % dearer
    eps = []
    for i, t in enumerate(tasks):
        decs = [_gdec("look", False, PLAN[0], PLAN[0] if i < 3 else "look")] if i < 4 else []
        decs.append(_gdec(PLAN[1], True, PLAN[1]))
        eps.append(_rec(t, X.ARM, i != 0, 0.022, decs, s1_cost=0.0005))
    (d / f"{X.ARM}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "eval_games.json").write_text(json.dumps({"n": 10, "task_ids": tasks}))
    monkeypatch.setattr(X, "EVAL_LIST", cfg / "eval_games.json")
    monkeypatch.setattr(X, "E6A_SUMMARY", cfg / "missing.json")
    return runs


def test_e6c_readout_tex_and_projection(e6c: Path) -> None:
    import module_map as MM

    import explore_e6c_alfworld as X
    r = X.summarize(e6c)
    a = r["arm"]
    assert a["n"] == 10 and a["delta_pt"] == 0.0 and (a["wins"], a["losses"]) == (1, 1)
    assert a["saving"] == pytest.approx(-0.1) and a["verdict"] == MM.NOSAVE
    d = a["decisions"]
    assert d["n_decisions"] == 14 and d["n_vetoed"] == 4 and d["vetoed"] == pytest.approx(4 / 14)
    assert d["coverage"] == d["vetoed"] and d["executed_from_model"] == 0.0
    assert d["acc"] == 1.0 and d["proposal_acc"] == pytest.approx(10 / 14) and d["veto_precision"] == 1.0
    assert d["rewrite_acc"] == pytest.approx(3 / 4) and d["rewrite_same"] == pytest.approx(1 / 4)
    assert d["executed_acc"] == pytest.approx(13 / 14) and d["vetoes_per_episode"] == pytest.approx(0.4)
    tex = X.to_tex(r)
    assert r"\defres{e6c/guard/vetoed}{28.6}" in tex and r"\defres{e6c/guard/verdict}{no saving}" in tex
    assert r"\defres{e6c/tau}{0.5}" in tex and "nan" not in tex.lower()
    md = X.to_markdown(r)
    assert "not the registered S2-L3b" in md and "| guard (`jev_action_guard`, tau 0.5) | 10 |" in md
    assert X.total_spend(e6c)["records"] == 20
    p = X.projection(e6c)                                       # no smoke here: nothing projected, the eval is spent
    assert p["smoke_n"] == {"llm_only": 0, X.ARM: 0} and p["spent_so_far"] == pytest.approx(10 * 0.02 + 10 * 0.022)
    assert X.projection(e6c, cap=0.1)["decision"] == "stop after the smoke"


def test_module_map_draws_the_e6c_guard_tile_apart_from_stage1(e6c: Path, tmp_path: Path) -> None:
    import module_map as MM

    import explore_e6c_alfworld as X
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"task_ids": ["jevgym-x-200000-d60-near-strict"]}))
    mm = MM.ModuleMap(e6c, n_boot=500, seed=0, primaries=tmp_path / "none.json", frozen=frozen, stage1=(),
                      stage2=None, quiet=True, e6b=None)   # default explore: E6a (absent here) and E6c
    res = mm.build()
    assert res["rows"] == [] and res["cells"]["flash/action/guard"]["state"] == "not run"
    x = res["explore_action"]["cells"]
    assert set(x) == {"flash/action/guard"} and res["explore_action"]["run"] == "E6c"
    g = x["flash/action/guard"]
    assert g["state"] == "exploratory" and g["explore"] == "E6c" and g["verdict"] == MM.NOSAVE
    assert g["acted"] == pytest.approx(4 / 14) and g["n"] == 10 and (g["wins"], g["losses"]) == (1, 1)
    assert g["acted"] == pytest.approx(X.summarize(e6c)["arm"]["decisions"]["vetoed"])   # one coverage definition
    tex = MM.to_tex(res)
    assert r"\defres{mm/xcell/flash/action/guard/vcode}{3}" in tex
    assert r"\defres{mm/xcell/flash/action/guard/explore}{E6c}" in tex
    assert "action/guard (E6c)" in MM.to_markdown(res)
