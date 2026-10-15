"""E20 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E20: decision-model ensembles and a hosted decision API behind
the gate"): the ensemble backend's agreement rule and accounting, and Part A's votes, gates, equal-coverage
comparison, split and DecisionGym-Hard slice, on tiny synthetic fixtures (offline, no model call)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e20_ensembles as E  # noqa: E402
from jevharness.accounting import Ledger  # noqa: E402
from jevharness.decision import make_backend  # noqa: E402
from jevharness.decision.base import DecisionBackend  # noqa: E402
from jevharness.decision.ensemble import EnsembleBackend, combine  # noqa: E402
from jevharness.decision.types import Answer, Choice, DecisionResult, Noul, Usage, answer_from_distribution  # noqa: E402


# ------------------------------------------------------------------ the ensemble backend
class Fixed(DecisionBackend):
    """A decision model that answers every Choice with a fixed distribution (and logs a priced call)."""

    def __init__(self, name: str, dist: dict[str, float], cost: float = 0.001, latency: float = 100.0,
                 refuse: bool = False, fail: bool = False):
        self.name = name
        self.model = name
        self.dist = dist
        self.cost = cost
        self.latency = latency
        self.refuse = refuse
        self.fail = fail
        self.batched_questions = True
        self.max_options = 255

    def _decide(self, state: str, questions: list) -> DecisionResult:
        if self.fail:
            raise RuntimeError("boom")
        ans = []
        for q in questions:
            a = answer_from_distribution(q, {k: v for k, v in self.dist.items() if k in q.labels()} or
                                         {q.labels()[0]: 1.0}, raw={"refusal": True} if self.refuse else None)
            if self.refuse:
                a.confidence = 0.0
            ans.append(a)
        return DecisionResult(answers=ans, usage=Usage(backend=self.name, model=self.name, cost_usd=self.cost,
                                                       latency_ms=self.latency))


Q = Choice("next tool", ["a", "b", "c"], decision_point="tool_select")


def _ans(v: str, c: float, refusal: bool = False) -> Answer:
    d = {"a": 0.0, "b": 0.0, "c": 0.0}
    d[v] = c
    rest = [k for k in d if k != v]
    for k in rest:
        d[k] = (1 - c) / len(rest)
    return Answer(question_id=Q.id, kind=Q.kind, distribution=d, value=v, confidence=c,
                  raw={"refusal": True} if refusal else None)


def test_combine_agree_takes_the_smaller_confidence_and_defers_on_any_disagreement():
    a = combine(Q, [_ans("a", 0.9), _ans("a", 0.7)], "agree", ["jev", "oaidec"])
    assert a.value == "a" and a.confidence == pytest.approx(0.7) and a.raw["ensemble"]["agree"] is True
    assert abs(sum(a.distribution.values()) - 1) < 1e-9
    d = combine(Q, [_ans("a", 0.9), _ans("b", 0.95)], "agree")
    assert d.confidence == 0.0 and d.value == "b" and d.raw["ensemble"]["agree"] is False   # larger summed conf
    r = combine(Q, [_ans("a", 0.9), _ans("a", 0.0, refusal=True)], "agree")
    assert r.confidence == 0.0 and r.raw["ensemble"]["members"][1]["refusal"] is True        # a refusal: no answer
    n = combine(Q, [_ans("a", 0.9), None], "agree")
    assert n.confidence == 0.0 and n.value == "a"                                            # a failed member
    w = combine(Q, [_ans("a", 0.6), _ans("b", 0.9), _ans("a", 0.5)], "weighted")
    assert w.value == "a" and w.confidence == pytest.approx((0.6 + 0.5) / 3)
    yn = Noul("stop?", decision_point="stop")
    y1 = answer_from_distribution(yn, {"yes": 0.8, "no": 0.2})
    y2 = answer_from_distribution(yn, {"yes": 0.6, "no": 0.4})
    assert combine(yn, [y1, y2]).confidence == pytest.approx(0.6)
    with pytest.raises(ValueError):
        combine(Q, [_ans("a", 0.9), _ans("a", 0.9)], "median")


def test_ensemble_backend_logs_each_member_call_once_with_one_group_and_no_record_of_its_own():
    ens = EnsembleBackend([Fixed("jev", {"a": 0.9, "b": 0.1}, cost=0.001, latency=100.0),
                           Fixed("openai_decisions", {"a": 0.8, "b": 0.2}, cost=0.002, latency=300.0)],
                          rule="agree", name="agree", member_names=["jev", "oaidec"])
    led = Ledger()
    res = ens.decide("STATE", [Q, Choice("arg", ["a", "b"], decision_point="tool_params")], ledger=led)
    assert [a.value for a in res.answers] == ["a", "a"]
    assert res.answers[0].confidence == pytest.approx(0.8)
    assert sorted(r.backend for r in led.records) == ["jev", "openai_decisions"]
    groups = {r.meta["ensemble"]["group"] for r in led.records}
    assert len(groups) == 1 and all(r.meta["ensemble"]["name"] == "agree" for r in led.records)
    assert sum(r.cost_usd for r in led.records) == pytest.approx(0.003)
    assert res.usage.cost_usd == pytest.approx(0.003) and res.usage.latency_ms == pytest.approx(300.0)
    led2 = Ledger()
    ens.decide("STATE", [Q], ledger=led2)
    assert {r.meta["ensemble"]["group"] for r in led2.records}.isdisjoint(groups)            # a new group per decision
    # a failing member: the ensemble defers (confidence 0) and never crashes
    bad = EnsembleBackend([Fixed("jev", {"a": 1.0}), Fixed("x", {"a": 1.0}, fail=True)], rule="agree")
    out = bad.decide("S", [Q], ledger=Ledger())
    assert out.answers[0].confidence == 0.0
    with pytest.raises(ValueError):
        EnsembleBackend([Fixed("jev", {"a": 1.0})])
    d = ens.describe()
    assert d["type"] == "ensemble" and len(d["members"]) == 2 and "api_key" not in json.dumps(d)


def test_make_backend_builds_an_ensemble_of_member_specs():
    b = make_backend({"type": "ensemble", "rule": "agree",
                      "members": [{"type": "simulated", "accuracy": 0.9}, {"type": "simulated", "accuracy": 0.8}]})
    assert isinstance(b, EnsembleBackend) and len(b.members) == 2


def test_the_cascade_delegates_only_when_the_members_agree():
    from jevharness.harness.agent import JevAgent
    from jevharness.harness.config import DPConfig, HarnessConfig, Mode

    agree = EnsembleBackend([Fixed("jev", {"a": 0.99}), Fixed("o", {"a": 0.98})], rule="agree")
    split = EnsembleBackend([Fixed("jev", {"a": 0.99}), Fixed("o", {"b": 0.99})], rule="agree")
    for be, want in ((agree, True), (split, False)):
        shell = JevAgent(llm=None, cfg=HarnessConfig(name="t", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.9)), s1=be)
        a = shell._choice_tournament("S", "next tool", ["a", "b", "c"], "tool_select", Ledger(), None)
        assert (a.confidence >= 0.9) is want


# ------------------------------------------------------------------ Part A on a synthetic level
def _level(answers: dict[str, list[tuple[str | None, float]]], gold: list[str]) -> E.Level:
    units = [{"i": i, "task": f"t{i // 2}", "source": "llm_only", "step": i % 2, "qhash": f"h{i}", "gold": g,
              "gold_aux": {"acceptable_tools": [g]}, "gold_status": "ok"} for i, g in enumerate(gold)]
    return E.Level("c", units, answers, lambda u, v: v == u["gold"])


def test_votes_gates_and_equal_coverage():
    assert E.vote([("a", 0.9), ("b", 0.6), ("b", 0.5)], "majority") == ("b", 2, pytest.approx(1.1))
    assert E.vote([("a", 0.9), ("b", 0.6), ("b", 0.2)], "weighted")[0] == "a"      # 0.9 against 0.8
    assert E.vote([("a", 0.7), ("b", 0.6), ("b", 0.2)], "weighted")[0] == "b"      # 0.7 against 0.8
    assert E.vote([(None, 0.0), (None, 0.0)], "majority") == (None, 0, 0.0)
    assert E.vote([("a", 0.5), ("b", 0.5)], "majority")[0] == "a"                    # tie: member order
    gold = ["a", "b", "a", "b"]
    lv = _level({"m1": [("a", 0.9), ("b", 0.8), ("b", 0.99), ("b", 0.6)],
                 "m2": [("a", 0.7), ("a", 0.9), ("b", 0.95), ("b", 0.7)],
                 "m3": [("a", 0.6), ("b", 0.5), ("a", 0.4), (None, 0.0)]}, gold)
    idx = list(range(4))
    ens = E.ensemble_answers(lv, ["m1", "m2", "m3"], "majority")
    assert [v for v, _, _ in ens] == ["a", "b", "b", "b"]
    g = E.gate_stats(lv, ens, 3, idx)
    assert g["accepted"] == 1 and g["accepted_errors"] == 0                    # only unit 0 is unanimous
    g2 = E.gate_stats(lv, ens, 2, idx)
    assert g2["accepted"] == 4 and g2["accepted_errors"] == 1                  # unit 2: two wrong members agree
    # m1's own confidence at the gate's coverage (accept its 3 most confident: 0.99 wrong, 0.9, 0.8)
    e = E.errors_at_coverage(lv, "m1", 3, idx)
    assert e["accepted_errors"] == pytest.approx(1.0)
    # ties at the boundary count in expectation
    lv2 = _level({"m": [("a", 0.5), ("a", 0.5), ("b", 0.5), ("b", 0.9)]}, gold)
    assert E.errors_at_coverage(lv2, "m", 2, idx)["accepted_errors"] == pytest.approx(2 / 3)
    assert E.accuracy(lv, [v for v, _, _ in ens], idx) == pytest.approx(0.75)  # task means: t0 1.0, t1 0.5
    st = E.threshold_stats(lv, "m2", 0.8, idx)
    assert st["accepted"] == 2 and st["accepted_errors"] == 2


def test_choose_level_picks_members_on_the_dev_units_only():
    gold = ["a"] * 6
    lv = _level({"good": [("a", 0.9)] * 6, "mid": [("a", 0.8)] * 4 + [("b", 0.9)] * 2,
                 "bad": [("b", 0.9)] * 6}, gold)
    E.FAMILIES["tiny"] = ("good", "mid", "bad")
    try:
        res = E.choose_level(lv, "tiny", [0, 1, 2, 3])
    finally:
        del E.FAMILIES["tiny"]
    assert res["best_single"] in ("good", "mid")
    assert res["gate"]["dev"]["accepted_errors"] == 0


def test_split_is_fixed_and_when2call_items_are_whole_ids():
    assert E.split_of("jevgym-clarify-300000-d60-near-strict") == E.split_of("jevgym-clarify-300000-d60-near-strict")
    assert E.l1_item("simple_python_197/param/location") == "simple_python_197"
    assert E.l1_item("w2c/0a414308-70d2") == "w2c/0a414308-70d2"
    assert {E.split_of(f"w2c/{i}") for i in range(50)} == {"dev", "test"}


def test_hard_slice_reads_the_common_states_of_two_shadow_arms(tmp_path: Path):
    def traj(task: str, steps: list[tuple[str, str, float, str]]) -> dict:
        return {"task_id": task, "steps": [{"step": k, "decisions": [
            {"dp": "tool_select", "mode": "shadow", "state": st, "options": ["cancel_order", "ship_order", "find_user"],
             "s1_value": v, "s1_conf": c, "gold": g, "gold_aux": {"acceptable_tools": [g]}, "gold_status": "ok"}]}
            for k, (st, v, c, g) in enumerate(steps)]}
    jev = [traj("t1", [("s0", "find_user", 0.9, "find_user"), ("s1", "ship_order", 0.99, "ship_order")])]
    rl = [traj("t1", [("s0", "find_user", 1.0, "find_user"), ("s1", "cancel_order", 1.0, "ship_order")]),
          traj("t2", [("x", "find_user", 1.0, "find_user")])]
    (tmp_path / "jev_shadow_cal.jsonl").write_text("".join(json.dumps(t) + "\n" for t in jev))
    (tmp_path / "rl_shadow_cal.jsonl").write_text("".join(json.dumps(t) + "\n" for t in rl))
    h = E.hard_slice(tmp_path)
    assert h["common"] == 2
    assert h["tier"]["n"] == 1 and h["tier"]["jev"] == 1.0 and h["tier"]["rl"] == 0.0
    assert h["tier"]["vote"] == 0.0                         # the 4B's more confident wrong answer wins the vote
    assert h["tier"]["gate_coverage"] == 0.0 and h["tier"]["gate_accepted_errors"] == 0   # disagreement: defer
    assert h["other"]["gate_coverage"] == 1.0


# ------------------------------------------------------------------ Part B accounting
def test_episode_extra_counts_parallel_s1_time_and_providers():
    calls = [{"component": "s1", "backend": "jev", "cost_usd": 0.001, "latency_ms": 200.0,
              "meta": {"ensemble": {"group": 0}}},
             {"component": "s1", "backend": "openai_decisions", "cost_usd": 0.002, "latency_ms": 500.0,
              "meta": {"ensemble": {"group": 0}}},
             {"component": "s1", "backend": "openai_decisions", "cost_usd": 0.002, "latency_ms": 400.0, "meta": {}},
             {"component": "s2", "backend": "gemini", "cost_usd": 0.01, "latency_ms": 1500.0}]
    dec = [{"dp": "tool_select", "mode": "cascade", "s1_value": "a", "s1_conf": 0.99, "deferred": False, "gold": "b",
            "gold_aux": {"acceptable_tools": ["b"]}, "gold_status": "ok"},
           {"dp": "tool_select", "mode": "cascade", "s1_value": "a", "s1_conf": 0.0, "deferred": True, "gold": "a",
            "gold_aux": {"acceptable_tools": ["a"]}, "gold_status": "ok"}]
    x = E.episode_extra({"calls": calls, "steps": [{"decisions": dec}]})
    assert x["s1_par_s"] == pytest.approx(0.9)               # max(0.2, 0.5) + 0.4
    assert x["prov"] == {"gemini": pytest.approx(0.01), "openai": pytest.approx(0.004), "jev": pytest.approx(0.001)}
    assert x["tool_select"]["accepted"] == 1 and x["tool_select"]["accepted_errors"] == 1
    assert x["tool_select"]["zero_conf"] == 1
    assert E.provider_of({"component": "s2", "backend": "something"}) == "gemini"


def test_live_configs_name_the_registered_rows_and_no_key():
    import yaml

    for row, arm in E.ROW_ARM.items():
        for kind in ("live", "live_smoke"):
            p = ROOT / f"configs/explore/e20_ensembles/{kind}_{row}.yaml"
            cfg = yaml.safe_load(p.read_text())
            assert cfg["arms"][0]["name"] == arm and cfg["workers"] == 1
            assert cfg["arms"][0]["profile"] == "configs/harness/explore_e20_final.yaml"
            assert cfg["llm"]["provider"] == "gemini" and cfg["llm"]["model"] == "gemini-3.5-flash"
            assert "api_key" not in cfg["s1_backends"][row] and not re.search(r"sk-[A-Za-z0-9_-]{12,}", p.read_text())
            if kind == "live":
                assert (cfg["suite"]["seed0"], cfg["n_tasks"]) == (300500, 200)
            else:
                assert cfg["n_tasks"] == 2 and cfg["suite"]["seed0"] in (912, 914)


def test_with_the_tool_choice_always_deferred_the_argument_cascade_is_never_reached():
    """E20 amendment "the fitted thresholds": row (ii)'s fit always defers at the tool choice; System-1 fills arguments
    only for a tool System-1 chose (``JevAgent._delegated_call``), so the row makes no System-1 request at all."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_cascade_call_graph import FixedS1, RecordingLLM, TinyEnv

    from jevharness.harness.agent import JevAgent
    from jevharness.harness.config import HarnessConfig

    dps = {"tool_select": {"mode": "cascade", "tau": 1.0, "always_defer": True},
           "tool_params": {"mode": "cascade", "tau": 0.99, "batch_s1": True}}
    s1 = FixedS1({"tool_params": 1.0, "tool_select": 1.0}, batched=True)
    t = JevAgent(RecordingLLM(), HarnessConfig(name="e20", **dps).validate_modes(), s1=s1).run(TinyEnv("cancel")).to_dict()
    assert t["error"] is None and s1.requests == []
    # control: the same backend at a reachable tool-choice threshold is asked the argument questions
    s1b = FixedS1({"tool_params": 1.0, "tool_select": 1.0}, batched=True)
    dps2 = {**dps, "tool_select": {"mode": "cascade", "tau": 0.5}}
    JevAgent(RecordingLLM(), HarnessConfig(name="e20", **dps2).validate_modes(), s1=s1b).run(TinyEnv("cancel"))
    assert any("argument" in q for req in s1b.requests for q in req)


def test_the_e20_profile_runs_no_row_whose_fit_delegates_nothing_reachable():
    from jevharness.harness.profile import decision_points, load_profile

    prof = load_profile("configs/harness/explore_e20_final.yaml")
    for row in E.ROWS:
        dps, _, skip = decision_points(prof, "gemini-3.5-flash", backend=row)
        blk = prof["system2"]["gemini-3.5-flash"]["backends"][row]
        if blk.get("tool_select", {}).get("always_defer") or "tool_select" not in blk:
            assert skip, row
