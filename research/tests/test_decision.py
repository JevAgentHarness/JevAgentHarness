import json
import math

import httpx
import pytest

from jevharness.decision import Choice, Noul, QKind, Score
from jevharness.decision.jev import JevBackend, answer_from_wire, question_to_wire
from jevharness.decision.llm_logprob import LogprobLLMBackend, build_prompt, dist_from_top_logprobs
from jevharness.decision.simulated import OracleBackend, SimulatedBackend
from jevharness.decision.types import answer_from_distribution
from jevharness.llm.base import ChatResponse
from jevharness.llm.scripted import ScriptedLLM


def test_question_validation():
    with pytest.raises(ValueError):
        Choice("x", ["only"])
    with pytest.raises(ValueError):
        Choice("x", ["a", "a"])
    with pytest.raises(ValueError):
        Score("x", scale=(0, 10))  # 11 levels > Jev's 10
    q = Score("x", levels=["bad", "ok", "good"])
    assert q.scale == (1, 3) and q.labels() == ["1", "2", "3"]


def test_answer_from_distribution():
    q = Choice("pick", ["a", "b", "c"])
    a = answer_from_distribution(q, {"a": 0.2, "b": 0.7})
    assert a.value == "b" and math.isclose(sum(a.distribution.values()), 1.0)
    n = answer_from_distribution(Noul("?"), {"yes": 0.8, "no": 0.2})
    assert n.value is True and math.isclose(n.p_yes, 0.8)
    s = answer_from_distribution(Score("?", scale=(1, 3)), {"1": 0.0, "2": 0.5, "3": 0.5})
    assert math.isclose(s.value, 2.5, abs_tol=1e-3)


def test_jev_wire_roundtrip_with_mock_server():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["body"] = body
        assert request.headers["authorization"] == "Bearer test-key"
        qs = body["questions"]
        answers = {}
        for k, q in qs.items():
            if q["type"] == "choice":
                labels = list(q["criteria"])
                answers[k] = {"type": "choice", "choice": labels[1], "confidence": 0.9,
                              "probabilities": {lab: (0.9 if i == 1 else 0.1 / (len(labels) - 1))
                                                for i, lab in enumerate(labels)}, "stats": {}}
            elif q["type"] == "noul":
                answers[k] = {"type": "noul", "noul": 0.25}
            else:
                n = len(q["criteria"])
                answers[k] = {"type": "score", "score": 1.0, "confidence": 0.8,
                              "probabilities": {str(i): (0.8 if i == 1 else 0.2 / (n - 1)) for i in range(n)}}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                         "usage": {"input_tokens": 1000, "output_tokens": 0}, "assets_used": None})

    b = JevBackend(model="jev-latest", api_key="test-key", base_url="https://mock")
    b._http = httpx.Client(base_url="https://mock", transport=httpx.MockTransport(handler),
                           headers={"Authorization": "Bearer test-key"})
    qs = [Choice("route?", ["billing", "tech", "other"], ["money", None, None]), Noul("upset?"),
          Score("urgency", scale=(1, 5))]
    r = b.decide("state text", qs)
    assert seen["body"]["questions"]["q0"]["criteria"] == {"billing": "money", "tech": None, "other": None}
    assert seen["body"]["questions"]["q2"]["type"] == "score" and len(seen["body"]["questions"]["q2"]["criteria"]) == 5
    assert r.answers[0].value == "tech"
    assert r.answers[1].value is False and math.isclose(r.answers[1].p_yes, 0.25)
    assert math.isclose(r.answers[2].value, 2.0)  # level index 1 on a 1..5 scale
    assert r.usage.model == "jev-1.13.0" and r.usage.input_tokens == 1000
    assert math.isclose(r.usage.cost_usd, 1000 * 0.042 / 1e6)


def test_wire_helpers():
    w = question_to_wire(Noul("done?", criteria={"true": "all tests pass", "false": "otherwise", "junk": 1}))
    assert w == {"type": "noul", "instructions": "done?", "criteria": {"true": "all tests pass", "false": "otherwise"}}
    a = answer_from_wire(Choice("c", ["x", "y"]), {"type": "choice", "choice": "y", "confidence": 0.6})
    assert a.value == "y"


def test_logprob_parsing_and_backend():
    q = Choice("pick", ["alpha", "beta", "gamma"])
    prompt, mapping = build_prompt("S", q)
    assert "A. alpha" in prompt and mapping["B"] == "beta"
    dist, cov = dist_from_top_logprobs({" B": math.log(0.6), "A": math.log(0.3), "zzz": math.log(0.1)}, mapping)
    assert math.isclose(dist["beta"], 0.6) and math.isclose(cov, 0.9)

    def policy(messages, tools):
        return ChatResponse(text="B", top_logprobs=[{"B": math.log(0.7), "C": math.log(0.2), "Yes": math.log(0.05)}])

    llm = ScriptedLLM("scripted-small", policy=policy)
    b = LogprobLLMBackend(llm)
    r = b.decide("state", [q, Noul("ok?")])
    assert r.answers[0].value == "beta"
    assert r.usage.n_calls == 2 and r.usage.cost_usd > 0


def test_simulated_backends():
    q = Choice("pick", ["a", "b", "c"], gold="c")
    assert OracleBackend().ask("s", q).value == "c"
    sim = SimulatedBackend(accuracy=0.8, seed=1)
    hits = sum(sim.ask("s", Choice("pick", ["a", "b", "c"], gold="a")).value == "a" for _ in range(500))
    assert 0.72 < hits / 500 < 0.88
    assert q.kind == QKind.CHOICE
