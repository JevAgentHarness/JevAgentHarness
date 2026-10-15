"""Exploratory row E14: decision rows as Clef records (training/clef_data.py) and the Clef backend of serve_jev
(training/clef_engine.py). Offline: a stand-in model; the vendored Clef head is used only where Unsloth is installed."""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from jevharness.decision.jev import answer_from_wire, question_to_wire
from jevharness.decision.types import QKind, Question
from jevharness.training import clef_data as CD
from jevharness.training.clef_engine import ClefEngine, finish, validate_question

ROOT = Path(__file__).resolve().parents[1]


def _row(i, dp="tool_select", kind="choice", options=None, target=None, rid=None, state="S", q="Which tool?"):
    options = options or ["a -- tool a", "b -- tool b", "c -- tool c"]
    target = target or {o: (1.0 if k == 1 else 0.0) for k, o in enumerate(options)}
    return {"id": rid or f"task-{i}/1/{dp}", "dp": dp, "kind": kind, "state": state, "question": q, "options": options,
            "target": target, "meta": {"after_error": 0, "jevgym_seed": 30000 + i}, "source": "traj:x"}


def test_options_to_criteria_splits_only_when_every_option_has_a_description():
    assert CD.options_to_criteria(["get -- Get one -- order.", "list -- List."]) == {"get": "Get one -- order.",
                                                                                      "list": "List."}
    assert CD.options_to_criteria(["U1", "ORD-2 -- x"]) == {"U1": None, "ORD-2 -- x": None}
    assert CD.options_to_criteria(["call one of the available tools", "ask"]) == {
        "call one of the available tools": None, "ask": None}
    with pytest.raises(ValueError):
        CD.options_to_criteria(["a -- x", "a -- y"])


def test_choice_row_matches_the_wire_question_the_harness_sends():
    q = Question(kind=QKind.CHOICE, text="Which tool should the agent call next?", options=["get", "list"],
                 option_descriptions=["Get one order.", "List orders."])
    wire = question_to_wire(q)
    row = _row(0, options=["get -- Get one order.", "list -- List orders."], q=q.text,
               target={"get -- Get one order.": 0.0, "list -- List orders.": 1.0})
    got, keymap = CD.row_question(row)
    assert got == wire
    assert CD.row_gold(row, keymap) == {"label": "list", "probabilities": {"get": 0.0, "list": 1.0}}
    args = Question(kind=QKind.CHOICE, text="Which value should the argument `x` take?", options=["U1", "U2"])
    arow = _row(1, dp="tool_params", options=["U1", "U2"], q=args.text)
    assert CD.row_question(arow)[0] == question_to_wire(args)


def test_noul_row_maps_yes_no_to_true_false_with_soft_targets():
    row = _row(0, dp="stop", kind="noul", options=["yes", "no"], target={"yes": 0.25, "no": 0.75}, q="Stop now?")
    q, keymap = CD.row_question(row)
    assert q == {"type": "noul", "instructions": "Stop now?"}
    assert CD.row_gold(row, keymap) == {"label": "false", "probabilities": {"true": 0.25, "false": 0.75}}


def test_argument_rows_of_one_step_form_one_record_in_logged_order_even_when_shuffled():
    base = "jevgym-multi-30247-d60-near-strict/lite_shadow:4/tool_params"
    a0 = _row(0, dp="tool_params", options=["U1", "U2"], rid=base, state="ST", q="arg user_id")
    a1 = _row(1, dp="tool_params", options=["pending", "all"], rid=base + "#1", state="ST", q="arg status")
    a2 = _row(2, dp="tool_params", options=["x", "y"], rid=base + "#2", state="ST", q="arg z")
    other = _row(3)
    recs = CD.rows_to_records([a2, other, a0, a1])
    assert [r["id"] for r in recs] == [base, other["id"]]   # ordered by each record's first row; id = its #0 row
    step = recs[0]
    assert list(step["questions"]) == ["q0", "q1", "q2"]
    assert [step["questions"][k]["instructions"] for k in ("q0", "q1", "q2")] == ["arg user_id", "arg status", "arg z"]
    assert step["meta"]["row_ids"] == [base, base + "#1", base + "#2"]
    assert step["gold"]["q1"]["label"] == "all"
    wire = CD.wire_request(step)
    assert set(wire) == {"model", "state", "questions"} and "gold" not in json.dumps(wire)
    # a different state never joins the step
    a3 = _row(4, dp="tool_params", options=["p", "q"], rid=base + "#3", state="OTHER", q="arg w")
    assert len(CD.rows_to_records([a0, a3])) == 2
    assert len(CD.rows_to_records([a0, a1], group_args=False)) == 2


def test_sft_v1_dev_file_converts_completely():
    rows = [json.loads(x) for x in gzip.open(ROOT / "data/train_v1/dev.jsonl.gz", "rt")]
    recs = CD.rows_to_records(rows)
    st = CD.record_stats(recs)
    assert st["questions"] == len(rows) == 2377
    assert st["records"] == 2065 and st["questions_by_dp"] == {"tool_select": 884, "stop": 669, "tool_params": 824}
    for r in recs:
        for qid, q in r["questions"].items():
            g = r["gold"][qid]
            keys = ["true", "false"] if q["type"] == "noul" else list(q["criteria"])
            assert g["label"] in keys and set(g["probabilities"]) == set(keys)
            assert math.isclose(sum(g["probabilities"].values()), 1.0, rel_tol=1e-9)


# ------------------------------------------------------------------ the engine with a stand-in model
class _Q:
    def __init__(self, qid, option_ids):
        self.question_id, self.option_ids = qid, tuple(option_ids)


class _Rec:
    def __init__(self, state, questions):
        self.input_ids = list(range(10 + len(str(state))))
        self.questions = []
        for qid, q in questions.items():   # Clef's option order: choice labels sorted, noul true / false
            ids = ["true", "false"] if q["type"] == "noul" else (
                sorted(str(k) for k in q["criteria"]) if q["type"] == "choice" else
                [str(i) for i in range(len(q["criteria"]))])
            self.questions.append(_Q(qid, ids))


def _fake_engine(temp=2.0, **kw):
    def encode(state, questions):
        return _Rec(state, questions)

    def forward(records):   # logit = position of the option in Clef's order (deterministic)
        return [[float(i) for i in range(len(q.option_ids))] for r in records for q in r.questions]

    def scales(records, questions, logits):
        return [temp] * len(logits)

    return ClefEngine(encode, forward, scales, tokenizer=None, name="e14-test", **kw)


def test_engine_answers_in_the_wire_format_unrounded_with_temperatures():
    eng = _fake_engine(temp=2.0)
    qs = {"q0": {"type": "choice", "instructions": "tool?", "criteria": {"zeta": "z", "alpha": None, "mid": "m"}},
          "q1": {"type": "noul", "instructions": "stop?"}}
    ans, usage = eng.answer_all("state", qs)
    z = [math.exp(i / 2.0) for i in range(3)]
    p = [v / sum(z) for v in z]                         # sorted ids: alpha 0, mid 1, zeta 2
    a0 = ans["q0"]
    assert a0["type"] == "choice" and a0["choice"] == "zeta" and a0["confidence"] == pytest.approx(p[2])
    assert list(a0["probabilities"]) == ["zeta", "alpha", "mid"]          # the wire's criteria order
    assert a0["probabilities"]["alpha"] == pytest.approx(p[0])            # not rounded
    assert ans["q1"] == {"type": "noul", "noul": pytest.approx(1 / (1 + math.exp(0.5)))}   # true is option 0
    assert usage["questions"] == 2 and usage["n_passes"] == 1 and "compute_ms" in usage
    assert eng.stats["requests"] == 1 and eng.stats["questions"] == 2
    q = Question(kind=QKind.CHOICE, text="tool?", options=["zeta", "alpha", "mid"])
    assert answer_from_wire(q, a0).value == "zeta"


def test_engine_validates_questions_like_clef():
    for bad in ({"type": "pick", "criteria": {"a": None}}, {"type": "choice", "criteria": {}},
                {"type": "noul", "criteria": {"maybe": "x"}}):
        with pytest.raises(ValueError):
            validate_question("q0", bad)
    eng = _fake_engine()
    with pytest.raises(ValueError):
        eng.answer_all("s", {"q0": {"type": "score"}})


def test_engine_batches_by_tokens_and_keeps_request_order():
    eng = _fake_engine(max_batch_tokens=40, max_batch=2)
    reqs = [("x" * n, {"q0": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}}})
            for n in (30, 1, 5, 2)]
    out = eng.answer_batch(reqs)
    assert len(out) == 4 and all(a["q0"]["choice"] == "b" for a, _ in out)
    assert eng.stats["batches"] >= 3


def test_engine_scores_rendered_rows_for_eval_decisions():
    from jevharness.training.eval_decisions import evaluate

    eng = _fake_engine(temp=1.0)
    rows = [_row(0, options=["b -- B", "a -- A"], target={"b -- B": 1.0, "a -- A": 0.0}),
            _row(1, dp="stop", kind="noul", options=["yes", "no"], target={"yes": 0.0, "no": 1.0})]
    res = evaluate(eng, rows, n_perm=0)
    assert res["overall"]["n"] == 2 and res["overall"]["acc"] == 1.0     # b sorts after a; false (no) is option 1
    assert set(res["by_dp"]) == {"tool_select", "stop"}


def test_served_through_serve_jev_build_app():
    from fastapi.testclient import TestClient

    from jevharness.training.serve_jev import build_app

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return type("E", (), {"input_ids": list(range(len(text.split())))})()

    eng = _fake_engine()
    eng.tok = Tok()
    c = TestClient(build_app(eng))
    assert c.get("/v1/models").json()["data"][0]["id"] == "e14-test"
    r = c.post("/v1/systemone", json={"model": "e14-test", "state": "a b c",
                                      "questions": {"q0": {"type": "noul", "instructions": "ok?"}}})
    assert r.status_code == 200 and r.json()["answers"]["q0"]["type"] == "noul"
    assert c.post("/v1/systemone", json={"state": "s", "questions": {"q0": {"type": "bad"}}}).status_code == 400


def test_finish_score_question():
    out = finish({"type": "score", "criteria": ["low", "high"]}, ["0", "1"], [0.25, 0.75])
    assert out["score"] == pytest.approx(0.75) and out["confidence"] == 0.75 and out["legend"] == {"0": "low", "1": "high"}


# ------------------------------------------------------------------ Cloudflare's head, only where Unsloth is installed
def _vendored_clef():
    spec = importlib.util.find_spec("unsloth")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / "_vendor" / "clef" / "joint_schema_model.py"
    if not path.exists():
        return None
    import sys

    s = importlib.util.spec_from_file_location("_e14_clef_ref", path)
    mod = importlib.util.module_from_spec(s)
    sys.modules["_e14_clef_ref"] = mod   # dataclasses resolve their module while the file executes
    s.loader.exec_module(mod)
    return mod


def test_engine_with_a_tiny_random_init_clef_head():
    mod = _vendored_clef()
    if mod is None:
        pytest.skip("unsloth (with the vendored Clef head) is not installed")
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    class Tok:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return type("E", (), {"input_ids": [1 + (ord(ch) % 250) for ch in text]})()

    torch.manual_seed(0)
    lm = AutoModelForCausalLM.from_config(Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                                                      num_hidden_layers=1, num_attention_heads=4,
                                                      num_key_value_heads=2, max_position_embeddings=4096))
    head = mod.JointSchemaHead(hidden_size=32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    model = mod.ClefModel(lm, head).eval()
    tok = Tok()

    def encode(state, questions):
        return mod.encode_record(tok, {"state": state, "questions": questions}, max_length=4096)

    @torch.no_grad()
    def forward(records):
        out = model(mod.collate_records(records, 0, torch.device("cpu")))
        return [z.tolist() for rec in out for z in rec]

    eng = ClefEngine(encode, forward, lambda r, q, z: [1.0] * len(z), tokenizer=tok, name="tiny")
    qs = {"q0": {"type": "choice", "instructions": "tool?", "criteria": {"b": "B", "a": None}},
          "q1": {"type": "noul", "instructions": "stop?"}}
    ans, _ = eng.answer_all("STATE text", qs)
    ref = mod.systemone(model, type("P", (), {"tokenizer": tok})(), {"model": "m", "state": "STATE text",
                                                                       "questions": qs})
    assert ans["q0"]["choice"] == ref["answers"]["q0"]["choice"]
    for k, v in ans["q0"]["probabilities"].items():
        assert v == pytest.approx(ref["answers"]["q0"]["probabilities"][k], abs=1e-4)
    assert ans["q1"]["noul"] == pytest.approx(ref["answers"]["q1"]["noul"], abs=1e-4)
    random.seed(0)


def test_engine_v2_coalesces_queued_requests_and_keeps_answers():
    import threading
    import time

    calls = []

    def encode(state, questions):
        return _Rec(state, questions)

    def forward(records):   # slow, so concurrent requests queue behind the running pass
        calls.append(len(records))
        time.sleep(0.05)
        return [[float(i) for i in range(len(q.option_ids))] for r in records for q in r.questions]

    v1 = ClefEngine(encode, forward, lambda r, q, z: [1.0] * len(z), name="v1")
    v2 = ClefEngine(encode, forward, lambda r, q, z: [1.0] * len(z), name="v2", coalesce=True)
    reqs = [(f"state {k}", {"q0": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, f"c{k}": None}},
                            "q1": {"type": "noul", "instructions": "stop?"}}) for k in range(8)]
    ref = [v1.answer_all(s, q)[0] for s, q in reqs]
    calls.clear()
    out: dict[int, Any] = {}

    def go(k):
        out[k] = v2.answer_all(*reqs[k])

    th = [threading.Thread(target=go, args=(k,)) for k in range(8)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert [out[k][0] for k in range(8)] == ref                     # same answers as one request per pass
    assert sum(calls) == 8 and max(calls) > 1 and len(calls) < 8      # some requests shared a pass
    assert max(u["coalesced"] for _, u in out.values()) > 1
    assert v2.describe()["engine"] == "v2" and v1.describe()["engine"] == "v1"
    with pytest.raises(ValueError):
        v2.answer_all("s", {"q0": {"type": "pick"}})
    w = v2.warmup(1, lengths=(3, 5))
    assert w["rounds"] == 1 and v2.stats["requests"] == 8
