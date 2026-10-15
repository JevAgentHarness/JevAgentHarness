"""E15 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E15: more decision-model baselines"): the OpenAI Decisions API
backend (mocked HTTP), refusals as no answer, the Jev backend's read width, and the E15 configs and script."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from jevharness.accounting import Ledger
from jevharness.decision import make_backend
from jevharness.decision.jev import JevBackend
from jevharness.decision.openai_decisions import OpenAIDecisionsBackend, question_to_api
from jevharness.decision.types import Choice, Noul, Score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _mock(handler):
    seen: list[dict] = []

    def h(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append({"url": str(req.url), "auth": req.headers.get("authorization"), "body": body})
        return handler(body)
    return httpx.MockTransport(h), seen


def _answers_in_reverse(body):
    out = []
    for q in body["questions"]:
        if q["type"] == "choice":
            vals = [c["value"] for c in q["choices"]]
            probs = [{"value": v, "probability": (0.9 if i == 1 else 0.1 / (len(vals) - 1))} for i, v in enumerate(vals)]
            out.append({"type": "choice", "name": q["name"], "choice": vals[1], "probabilities": probs, "confidence": 0.9})
        elif q["type"] == "predicate":
            out.append({"type": "predicate", "name": q["name"], "probability": 0.8})
        else:
            out.append({"type": "score", "name": q["name"], "score": 2.0, "confidence": 0.7,
                        "probabilities": [{"label": lv["label"], "probability": p}
                                          for lv, p in zip(q["levels"], (0.1, 0.7, 0.2))]})
    return httpx.Response(200, json={"model": "gpt-6-luna", "answers": list(reversed(out)),
                                     "usage": {"input_tokens": 1000, "output_tokens": 0}})


def test_request_schema_and_answers_by_name():
    tr, seen = _mock(_answers_in_reverse)
    b = OpenAIDecisionsBackend(api_key="sk-test-dummy", transport=tr)
    qs = [Choice("Which tool next?", ["cancel_order", "ship_order", "ask_user"],
                 ["Cancel an order", None, "Ask the user"], gold="ship_order"),
          Noul("Is the task done?", gold=True),
          Score("How urgent?", levels=["low", "mid", "high"])]
    led = Ledger()
    res = b.decide("STATE: order 7", qs, ledger=led)
    assert len(seen) == 1                                   # every question about one state in one request
    body = seen[0]["body"]
    assert seen[0]["url"].endswith("/v1/decisions") and seen[0]["auth"] == "Bearer sk-test-dummy"
    assert body["model"] == "gpt-6-luna" and body["input"] == "STATE: order 7"
    assert isinstance(body["questions"], list) and [q["name"] for q in body["questions"]] == ["q0", "q1", "q2"]
    c = body["questions"][0]
    assert c["type"] == "choice" and "id" not in c
    assert c["choices"][0] == {"value": "cancel_order", "description": "Cancel an order"}
    assert c["choices"][1] == {"value": "ship_order"}       # no description -> no field
    assert body["questions"][1]["type"] == "predicate"
    assert [lv["label"] for lv in body["questions"][2]["levels"]] == ["1", "2", "3"]
    assert "gold" not in json.dumps(body) and "ship_order\"" not in json.dumps(body["questions"][1])
    a0, a1, a2 = res.answers                                # matched by name although returned in reverse order
    assert a0.value == "ship_order" and abs(a0.confidence - 0.9) < 1e-9
    assert a1.value is True and abs(a1.p_yes - 0.8) < 1e-6
    assert abs(a2.distribution["2"] - 0.7) < 1e-6
    assert abs(res.usage.cost_usd - 1000 * 0.10 / 1e6) < 1e-12 and res.usage.priced
    assert led.records[0].component == "s1" and abs(led.records[0].cost_usd - 1e-4) < 1e-12


def test_describe_empty_option():
    q = Choice("x?", ["a", "b"])
    assert question_to_api(q, "q0", describe_empty=True)["choices"][0] == {"value": "a", "description": ""}


def test_refusal_is_no_answer_and_counted():
    def h(body):
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": [{"type": "refusal", "name": "q0"}],
                                         "usage": {"input_tokens": 10}})
    tr, _ = _mock(h)
    b = OpenAIDecisionsBackend(api_key="k", transport=tr)
    a = b.decide("s", [Choice("which?", ["a", "b"])]).answers[0]
    assert a.raw["refusal"] is True and a.confidence == 0.0


def test_decision_eval_and_replay_score_a_refusal_wrong(monkeypatch):
    from jevharness.bench import decision_eval as DE
    from jevharness.bench import stage2_corpus as SC

    def h(body):
        return httpx.Response(200, json={"answers": [{"type": "refusal", "name": q["name"]} for q in body["questions"]],
                                         "usage": {"input_tokens": 5}})
    tr, _ = _mock(h)
    b = OpenAIDecisionsBackend(api_key="k", transport=tr)

    class It:
        def __init__(self, i):
            self.id, self.source, self.dp, self.state = f"i{i}", "bfcl:simple", "tool_select", "S"
            self.question = Choice("which?", ["a", "b"])
            self.gold = ["a"]
            self.meta = {}
    monkeypatch.setattr(DE, "_pub_meta", lambda it: {})
    recs = DE._s1_records(b, [It(0), It(1)], True, 1, "oaidec")
    assert all(r["refusal"] and r["correct"] is False and r["pred"] == "<refusal>" for r in recs)
    assert not DE._rescore(recs[0])
    out = SC.replay([{"qhash": "h1", "state": "S", "options": ["a", "b"], "descriptions": []}], b, retries=0)
    assert out[0]["refusal"] and out[0]["valid"] is False and out[0]["answer"] is None
    assert not SC.answer_correct({"gold": "a"}, out[0])


def test_retry_then_error():
    calls = {"n": 0}

    def h(body):
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad"}})
    tr, _ = _mock(h)
    b = OpenAIDecisionsBackend(api_key="k", transport=tr, max_retries=2)
    with pytest.raises(Exception):
        b.decide("s", [Choice("which?", ["a", "b"])])
    assert calls["n"] == 1                                  # a 400 is not retried


def test_make_backend_and_jev_read_width():
    b = make_backend({"type": "openai_decisions", "model": "gpt-6-luna", "api_key": "k"})
    assert isinstance(b, OpenAIDecisionsBackend) and b.max_options == 255 and b.batched_questions
    j = JevBackend(base_url="http://127.0.0.1:1", api_key="", model="verdict-1.4", max_options=24)
    assert j.max_options == 24 and j.describe()["max_options"] == 24
    j2 = JevBackend(base_url="http://127.0.0.1:1", api_key="", model="m")
    assert j2.max_options == 255 and "max_options" not in j2.describe()    # unset: every existing cache key unchanged


def test_e15_configs_match_e1_population():
    e1 = yaml.safe_load((ROOT / "configs/explore/e14_small/e1_e14.yaml").read_text())
    e15 = yaml.safe_load((ROOT / "configs/explore/e15_baselines/e1_e15.yaml").read_text())
    for k in ("suites", "exclude_ids_file", "batch_by_state", "workers"):
        assert e15[k] == e1[k]
    dev = yaml.safe_load((ROOT / "configs/explore/e15_baselines/e1dev_e15.yaml").read_text())
    chk = yaml.safe_load((ROOT / "configs/explore/e9_frontier/e1_dev_check.yaml").read_text())
    assert dev["suites"] == chk["suites"] and "exclude_ids_file" not in dev
    assert set(e15["backends"]) == set(dev["backends"])
    import explore_e15_baselines as X
    assert set(X.MODELS) == set(e15["backends"])
    for slug, spec in e15["backends"].items():
        if spec["type"] == "jev":
            assert spec["base_url"] == "http://127.0.0.1:8090" and spec["price_model"] == "e15-selfhost"
            assert "openrouter" not in json.dumps(spec)
    assert e15["backends"]["verdict"]["max_options"] == 24
    assert e15["s1_cache"] != e1["s1_cache"] and dev["s1_cache"] != e15["s1_cache"]


def test_kendall_block_reading_rule():
    import explore_e15_baselines as X
    rows = {"a": {"e1_acc": .9, "corpus_acc": .9, "live_success": .9}, "b": {"e1_acc": .8, "corpus_acc": .5, "live_success": .2},
            "c": {"e1_acc": .95, "corpus_acc": .6, "live_success": .4}, "d": {"e1_acc": .85, "corpus_acc": .7, "live_success": .6}}
    k = X.kendall_block(rows)
    assert k["corpus_vs_live_all_rows"] == pytest.approx(1.0)
    assert k["e1_vs_live_all_rows"] < 0.5 and k["holds"] is True


def test_wrap_server_app_answers_the_jev_wire():
    """The wrapper's app (stand-in answer function, no vendor code): a Jev wire request is answered, the body reaches
    the function unchanged, and the Jev backend reads the reply."""
    import importlib.util

    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location("e15_wrap", ROOT / "scripts/aws/e15_wrap_server.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    seen = []

    def answer(body):
        if not body.get("questions"):
            raise ValueError("at least one question is required")
        seen.append(body)
        return {"answers": {k: {"type": "choice", "choice": list(q["criteria"])[1], "confidence": 0.7,
                                "probabilities": {o: (0.7 if i == 1 else 0.3) for i, o in enumerate(q["criteria"])}}
                            for k, q in body["questions"].items()}, "usage": {"input_tokens": 12, "output_tokens": 0}}
    client = TestClient(w.build_app(answer, "stand-in", "clef"))
    fwd = httpx.MockTransport(lambda r: (lambda x: httpx.Response(x.status_code, content=x.content, headers=x.headers))(
        client.request(r.method, r.url.path, content=r.content, headers={"content-type": "application/json"})))
    j = JevBackend(base_url="http://testserver", api_key="", model="stand-in", transport=fwd)
    res = j.decide("The user asks to ship order 7.", [Choice("Which tool next?", ["cancel_order", "ship_order"])])
    assert res.answers[0].value == "ship_order" and res.usage.input_tokens == 12
    assert seen[0]["questions"]["q0"]["criteria"] == {"cancel_order": None, "ship_order": None}
    assert client.get("/health").json()["requests"] == 1
    bad = client.post("/v1/systemone", json={"model": "x", "state": "s", "questions": {}})
    assert bad.status_code == 400


def test_rank_selects_complete_l2_by_accuracy(tmp_path):
    """Registration (c) 4: the GPU models with a complete L2, by L2 accuracy (task means), ties by L1 accuracy."""
    import gzip
    import types

    import explore_e15_baselines as X
    occ = [{"task": t, "qhash": f"{t}{i}", "gold": "a", "gold_status": "ok"} for t in ("t1", "t2") for i in range(2)]
    with gzip.open(tmp_path / "occurrences.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(o) for o in occ) + "\n")
    (tmp_path / "corpus.json").write_text(json.dumps({"eligible_tasks": ["t1", "t2"]}))

    def answers(slug, right, n=4):
        rows = [{"qhash": o["qhash"], "answer": "a" if i < right else "b", "valid": True, "confidence": 0.9,
                 "error": None} for i, o in enumerate(occ[:n])]
        (tmp_path / f"answers_{slug}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    answers("jevk5", 2)
    answers("decider", 2)          # ties jevk5 on L2; wins on L1 below
    answers("imajev", 3)
    answers("openjev", 4, n=3)     # best accuracy but incomplete: left out
    answers("oaidec", 4)           # not a GPU model: never ranked here
    l1 = tmp_path / "l1"
    l1.mkdir()
    for slug, c in (("jevk5", 0.5), ("decider", 0.9), ("imajev", 0.1)):
        recs = [{"correct": i < 10 * c, "latency_ms": 100, "request_questions": 1} for i in range(10)]
        (l1 / f"s1_{slug}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = X.cmd_rank(types.SimpleNamespace(corpus=str(tmp_path), l1=str(l1), top=2, out=str(tmp_path / "sel.json")))
    assert rc == 0 and buf.getvalue().split() == ["imajev", "decider"]
    sel = json.loads((tmp_path / "sel.json").read_text())
    assert sel["excluded_incomplete"] == ["openjev"] and "oaidec" not in sel["rows"]
