"""Offline tests for provider plumbing: key rotation, Gemini thought signatures, verbalised deciders,
the generic Jev-wire server and the OpenRouter Jev preset. No network."""

import json
import math

import httpx
import pytest

from jevharness.decision import Choice, Noul, Score
from jevharness.decision.jev import JevBackend
from jevharness.decision.llm_generative import GenerativeLLMBackend, build_prompt, parse_answer
from jevharness.decision.server import answer_to_wire, build_app, wire_to_question
from jevharness.decision.simulated import OracleBackend
from jevharness.llm.base import ChatResponse
from jevharness.llm.openai_compat import GEMINI_DUMMY_SIGNATURE, _gemini_sanitize, provider_keys
from jevharness.llm.scripted import ScriptedLLM


def test_provider_keys_prefers_multi_key_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "k1, k2,,k3")
    monkeypatch.setenv("GEMINI_API_KEY", "single")
    assert provider_keys("GEMINI_API_KEY") == ["k1", "k2", "k3"]
    assert provider_keys("GEMINI_API_KEY", explicit="x") == ["x"]
    monkeypatch.delenv("GEMINI_API_KEYS")
    assert provider_keys("GEMINI_API_KEY") == ["single"]


def test_openai_compat_rotates_keys_on_429(monkeypatch):
    openai = pytest.importorskip("openai")
    from jevharness.llm.openai_compat import OpenAICompatClient

    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    c = OpenAICompatClient("gemini-x", provider="gemini", max_retries=3)
    used = []

    class Fake:
        def __init__(self, key):
            self.key = key
            self.chat = self
            self.completions = self

        def create(self, **params):
            used.append(self.key)
            if self.key == "k1":
                req = httpx.Request("POST", "https://x")
                raise openai.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)
            return "ok"

    c.clients = [Fake("k1"), Fake("k2")]
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert c._create({"model": "m", "messages": []}) == "ok"
    assert used[-1] == "k2" and "k1" in used and c.n_retries >= 1


def test_gemini_sanitize_adds_dummy_signature_only_when_missing():
    real = {"google": {"thought_signature": "abc"}}
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}, "extra_content": real},
                {"id": "2", "type": "function", "function": {"name": "g", "arguments": "{}"}}]}]
    out = _gemini_sanitize(msgs)
    tcs = out[1]["tool_calls"]
    assert tcs[0]["extra_content"] == real
    assert tcs[1]["extra_content"]["google"]["thought_signature"] == GEMINI_DUMMY_SIGNATURE
    assert "extra_content" not in msgs[1]["tool_calls"][1]  # input not mutated


def test_verbalised_distribution_parsing_and_batching():
    qs = [Choice("which?", ["alpha", "beta", "gamma"]), Noul("ok?"), Score("urgency", scale=(1, 3))]
    prompt = build_prompt("S", [(f"Q{i + 1}", q) for i, q in enumerate(qs)], "distribution")
    assert "Q1" in prompt and "Q3" in prompt and '"A": "<prob>"' in prompt
    reply = json.dumps({"Q1": {"A": 0.1, "B": "70%", "C": 0.2}, "Q2": {"yes": 0.9, "no": 0.1},
                        "Q3": {"1": 0.2, "2": 0.5, "3": 0.3}})

    def policy(messages, tools):
        return ChatResponse(text=f"```json\n{reply}\n```")

    llm = ScriptedLLM("scripted-small", policy=policy)
    b = GenerativeLLMBackend(llm, mode="distribution", batch=True)
    r = b.decide("S", qs)
    assert r.usage.n_calls == 1  # one call for three questions
    assert r.answers[0].value == "beta" and r.answers[1].value is True
    assert math.isclose(r.answers[2].value, 2.1, abs_tol=1e-6)
    assert all(a.raw["parse_ok"] for a in r.answers)
    # label mode + unparseable answer -> uniform, flagged
    dist, conf = parse_answer(Choice("x", ["a", "b"]), {"answer": "B", "confidence": 0.8}, "label")
    assert dist["b"] == 0.8 and conf == 0.8
    assert parse_answer(Choice("x", ["a", "b"]), "garbage", "distribution") == (None, None)


def test_generic_server_roundtrip_with_wire_client():
    from fastapi.testclient import TestClient

    app = build_app(OracleBackend(), name="oracle")
    b = JevBackend(model="oracle", base_url="http://testserver", api_key="")
    b._http = TestClient(app)
    r = b.decide("S", [Choice("which?", ["a", "b", "c"]), Noul("ok?"), Score("u", scale=(1, 5))])
    assert r.answers[0].value in {"a", "b", "c"} and 0.0 <= r.answers[1].p_yes <= 1.0
    q = wire_to_question("q0", {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]})
    assert q.scale == (0, 2) and q.levels == ["low", "mid", "high"]
    w = answer_to_wire(q, OracleBackend().ask("S", q))
    assert w["type"] == "score" and set(w["probabilities"]) == {"0", "1", "2"}


def test_openrouter_preset_path_and_reported_cost():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = json.loads(request.content)
        seen["model"] = body["model"]
        return httpx.Response(200, json={"id": "gen-1", "provider": "TypeSafe", "answers": {
            "q0": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 308, "output_tokens": 23,
                                                             "cost": 0.000012936}})

    b = JevBackend(provider="openrouter", api_key="sk-test")
    assert b.base_url == "https://openrouter.ai/api" and b.model == "typesafe/jev-1.13"
    b._http = httpx.Client(base_url=b.base_url, transport=httpx.MockTransport(handler),
                           headers={"Authorization": "Bearer sk-test"})
    r = b.decide("S", [Noul("done?")])
    assert seen["url"] == "https://openrouter.ai/api/v1/systemone" and seen["model"] == "typesafe/jev-1.13"
    assert math.isclose(r.usage.cost_usd, 0.000012936) and r.answers[0].value is True
    with pytest.raises(ValueError):
        JevBackend(provider="nope")


def test_temperature_scaling_backend_and_fit():
    import numpy as np

    from jevharness.decision import make_backend
    from jevharness.decision.calibration import fit_from_records, scale_distribution

    d = scale_distribution({"a": 0.9, "b": 0.1}, 2.0)
    assert 0.5 < d["a"] < 0.9 and math.isclose(sum(d.values()), 1.0)
    b = make_backend({"type": "calibrated", "inner": {"type": "simulated", "accuracy": 0.9}, "temperature": 3.0})
    a = b.ask("S", Choice("x", ["a", "b", "c"], gold="a"))
    assert max(a.distribution.values()) < 0.9 and "t" in a.raw
    # over-confident records (conf 0.99 but 70% correct) -> fitted T > 1
    rng = np.random.default_rng(0)
    recs = []
    for i in range(400):
        correct = rng.random() < 0.7
        recs.append({"s1_dist": {"a": 0.99, "b": 0.01}, "gold": "a" if correct else "b"})
    cal = fit_from_records(recs, min_per_kind=10)
    assert cal["global"] > 1.5 and cal["per_kind"]["choice"] > 1.5


def test_openai_compat_keeps_reported_cached_tokens_in_ledger(monkeypatch):
    """Gemini's OpenAI-compatible endpoint reports implicit-cache hits as ``usage.prompt_tokens_details.cached_tokens``
    (pilot ledger: 5392 prompt / 1932 cached on step 1 of an episode). The client must keep the figure, the ledger
    record its ``meta.cached_input_tokens``, and the cost must price those tokens at ``cached_input_per_m``."""
    pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion

    from jevharness.accounting import Ledger, default_prices
    from jevharness.llm.openai_compat import OpenAICompatClient

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    c = OpenAICompatClient("gemini-3.5-flash", provider="gemini")
    payload = {"id": "x", "object": "chat.completion", "created": 0, "model": "gemini-3.5-flash",
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"}}],
               "usage": {"prompt_tokens": 5392, "completion_tokens": 28, "total_tokens": 5420,
                         "prompt_tokens_details": {"cached_tokens": 1932}}}
    c._create = lambda params, timing=None: ChatCompletion.model_validate(payload)  # type: ignore[method-assign]
    ledger = Ledger()
    resp = c.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "f"}}],
                  ledger=ledger)
    assert resp.input_tokens == 5392 and resp.cached_input_tokens == 1932 and resp.output_tokens == 28
    rec = ledger.records[-1]
    assert rec.meta["cached_input_tokens"] == 1932 and rec.meta["n_tools"] == 1 and rec.input_tokens == 5392
    price = default_prices().get("gemini-3.5-flash")
    if price is not None and price.cached_input_per_m is not None:
        assert rec.cost_usd == pytest.approx(price.cost(5392, 28, 1932))
        assert rec.cost_usd < price.cost(5392, 28, 0)          # the cached rate was applied
    # no details -> 0, never None; DeepSeek's separate field is read too
    payload["usage"] = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    assert c.chat([{"role": "user", "content": "hi"}]).cached_input_tokens == 0
    import types
    fake = types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop", logprobs=None)],
        usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=1, total_tokens=101,
                                    prompt_tokens_details=None, completion_tokens_details=None,
                                    prompt_cache_hit_tokens=64), model="deepseek-chat")
    c._create = lambda params, timing=None: fake  # type: ignore[method-assign]
    assert c.chat([{"role": "user", "content": "hi"}]).cached_input_tokens == 64
