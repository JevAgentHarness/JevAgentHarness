"""E8 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "Exploratory row E8"): the offline metric code of
scripts/explore_e8_gates.py on synthetic data, the stratified sample, the reply parser, and the canary that no gate
prompt can carry the answer key."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explore_e8_gates as X  # noqa: E402

DESCS = {"find_user": "Look up a customer by full name.", "ask_user": "Ask the customer.", "finish": "Done."}


def test_net_saving_free_perfect_gate_accepts_exactly_the_right_answers():
    g = np.array([1, 1, 0, 1, 0], float)
    q = np.ones(5)
    dc = np.full(5, 0.01)
    k = np.full(5, 0.001)
    e = X.net_saving(g, g, q, dc, k, np.zeros(5), V=1.0)
    assert e["accepted"].tolist() == [True, True, False, True, False]
    assert e["usd"] == pytest.approx((3 * 0.01 - 5 * 0.001) / 5)
    assert e["threshold"] == 1.0 and e["coverage"] == pytest.approx(0.6)


def test_net_saving_accepts_none_when_errors_cost_more_than_the_saving():
    s = np.array([0.9, 0.8, 0.7])
    g = np.array([0, 1, 1], float)          # the most confident answer is wrong
    e = X.net_saving(s, g, np.ones(3), np.full(3, 0.01), np.zeros(3), np.zeros(3), V=1.0)
    # accepting the top answer loses 0.99; accepting all three loses 0.97: the best is to accept none
    assert e["threshold"] == float("inf") and e["usd"] == 0.0 and not e["accepted"].any()


def test_a_gate_that_costs_the_llm_call_never_saves_when_the_llm_is_always_right():
    rng = np.random.default_rng(1)
    n = 200
    g = (rng.random(n) < 0.9).astype(float)
    dc = rng.uniform(0.002, 0.008, n)
    k = np.full(n, 1e-4)
    for s in (g, rng.random(n), g + rng.normal(0, 0.3, n)):
        e = X.net_saving(s, g, np.ones(n), dc, k, dc, V=1.0)      # gate cost = the decision's LLM cost
        assert e["usd"] <= -k.mean() + 1e-12


def test_agreement_gate_saves_nothing_and_pays_the_model_call():
    g = np.array([1, 1, 0, 1], float)
    agree = g.copy()                       # the LLM is always right: agreement = the model is right
    e = X.net_saving(agree, g, np.ones(4), np.full(4, 0.01), np.full(4, 0.001), np.zeros(4), V=1.0, saves=False)
    assert e["usd"] == pytest.approx(-0.001)


def test_two_stage_charges_the_gate_only_below_tau():
    c = np.array([1.0, 1.0, 0.5, 0.4])
    gate = np.array([0.0, 0.0, 0.9, 0.1])
    g = np.array([1, 1, 1, 0], float)
    hi = c >= 1.0
    e = X.net_saving(gate, g, np.ones(4), np.full(4, 0.01), np.zeros(4), np.full(4, 0.002), V=1.0,
                     consult=~hi, pre=hi)
    # pre-accepts two (0.02), the gate is paid twice (0.004) and accepts the 0.9 one (0.01)
    assert e["usd"] == pytest.approx((0.02 + 0.01 - 0.004) / 4)
    assert e["accepted"].tolist() == [True, True, True, False]
    assert X.latency_change(e["accepted"], np.full(4, 800.0), np.full(4, 200.0), np.full(4, 500.0), ~hi) \
        == pytest.approx((3 * 800 - 4 * 200 - 2 * 500) / 4)


def test_auroc_and_selective_accuracy_with_ties():
    g = np.array([1, 1, 1, 0, 0], float)
    assert X.M.auroc(np.array([.9, .8, .7, .2, .1]), g) == 1.0
    assert X.M.auroc(np.ones(5), g) == 0.5
    # a tie group straddling the cut is taken in expectation
    assert X.M.selective_accuracy(np.array([1, 1, .5, .5, .1]), np.array([1, 0, 1, 1, 0], float), 0.4) \
        == pytest.approx(0.5)
    b = X.boot_auroc({"g1": np.array([.9, .8, .7, .2, .1, .6]), "g2": np.ones(6)},
                     np.array([1, 1, 1, 0, 0, 1], float), ["a", "a", "b", "b", "c", "c"], n_boot=50)
    assert b["g2"]["diff"] == pytest.approx(-0.5) and b["g1"]["diff"] == 0.0


def test_parse_g4():
    assert X.parse_g4("VERDICT: yes\nCONFIDENCE: 85") == (0.85, "yes", 85.0)
    assert X.parse_g4("**VERDICT:** no\n**CONFIDENCE:** 10") == (0.10, "no", 10.0)
    assert X.parse_g4("VERDICT: yes")[0] == 1.0
    assert X.parse_g4("no")[0] == 0.0
    assert X.parse_g4("CONFIDENCE: 140")[0] == 1.0
    assert X.parse_g4("")[0] is None


def test_top2_margin():
    assert X.top2_margin({"a": 0.7, "b": 0.2, "c": 0.1}) == pytest.approx(0.5)
    assert X.top2_margin({"a": 1.0}) == 1.0


def test_sample_is_stratified_in_proportion_and_reproducible():
    recs = []
    for dp in X.DPS:
        for i in range(100):
            recs.append({"id": f"t{i % 10}|{i}|{dp}|0", "dp": dp, "c": 0.9, "s1_ok": i % 10 != 0, "s2_ok": True,
                         "acceptable": True})
    a = X.draw_sample(recs, n_per_dp=50, seed=0)
    assert a == X.draw_sample(recs, n_per_dp=50, seed=0)
    for dp in X.DPS:
        assert a["strata"][dp] == {"pool": 100, "pool_wrong": 10, "n": 50, "wrong": 5, "right": 45}
    assert len(set(a["ids"])) == 100


@pytest.mark.parametrize("dp,question,value", [
    ("tool_select", "next tool", "find_user"),
    ("tool_params", "Which value should the argument `name` of tool `find_user` take? (customer full name)", "Hana"),
])
def test_no_gate_prompt_can_carry_the_answer_key(dp, question, value):
    """The builders take the state, the options, the question and the proposal only; a record's gold fields (the
    answer key, the acceptable set, the LLM's correctness) cannot reach a prompt."""
    state = "TASK: Ship Hana Smith's pending order.\n\nCURRENT OBSERVATION:\nok"
    opts = ["find_user", "ask_user", "finish"] if dp == "tool_select" else ["Hana", "Smith", "<OTHER>"]
    rec = {"dp": dp, "state": state, "options": opts, "question": question, "s1_value": value,
           "gold": "GOLD_SENTINEL", "gold_aux": {"acceptable_tools": ["GOLD_SENTINEL"]}, "s2_correct": True}
    q3 = X.g3_question(rec["dp"], rec["question"], rec["s1_value"], DESCS)
    p4 = X.g4_prompt(rec["dp"], rec["state"], rec["options"], rec["question"], rec["s1_value"], DESCS)
    for text in (q3, p4):
        assert "GOLD_SENTINEL" not in text and value in text
    assert p4.startswith(state)
    import inspect
    for fn in (X.g3_question, X.g4_prompt):
        assert not ({"gold", "rec", "record", "s1_ok", "s2_ok"} & set(inspect.signature(fn).parameters))


def test_noul_question_carries_no_meta(monkeypatch):
    """G3 sends a Noul without meta (``question_to_wire`` sends only ``criteria`` from meta)."""
    from jevharness.decision.jev import question_to_wire

    seen = {}

    class Fake:
        def ask(self, state, q, ledger=None):
            seen["wire"] = question_to_wire(q)
            seen["meta"] = dict(q.meta)

            class A:
                p_yes, raw = 0.8, {"noul": 0.8}
            return A()

    out = X.call_g3(Fake(), {"dp": "tool_select", "question": "next tool", "s1_value": "find_user",
                             "state": "TASK: x"}, DESCS)
    assert out["score"] == 0.8 and seen["meta"] == {} and set(seen["wire"]) == {"type", "instructions"}


def test_saving_at_reproduces_the_optimum_and_crossfit_is_held_out():
    rng = np.random.default_rng(3)
    n = 120
    g = (rng.random(n) < 0.9).astype(float)
    s = g * 0.5 + rng.random(n) * 0.6
    q, dc, k, gc = np.ones(n), np.full(n, 0.01), np.full(n, 1e-4), np.zeros(n)
    e = X.net_saving(s, g, q, dc, k, gc, V=0.05)
    assert X.saving_at(e["threshold"], s, g, q, dc, k, gc, V=0.05) / n == pytest.approx(e["usd"])
    tasks = [f"t{i // 4}" for i in range(n)]
    cv = X.crossfit_saving(tasks, s, g, q, dc, k, gc, V=0.05, n_splits=5)
    gain = dc - 0.05 * (q - g)                    # no acceptance set beats accepting exactly the positive gains
    assert cv["min"] <= cv["usd"] <= cv["max"] <= (gain[gain > 0].sum() - k.sum()) / n + 1e-12
    perfect = X.crossfit_saving(tasks, g, g, q, dc, k, gc, V=0.05, n_splits=3)
    assert perfect["usd"] == pytest.approx(X.net_saving(g, g, q, dc, k, gc, V=0.05)["usd"])
