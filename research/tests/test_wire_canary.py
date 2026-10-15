"""Request-level canary tests (review P1-1, results/reviews/2026-10-01d/gpt-6-astra.md §6 P1.1): simulation-only
metadata -- ``Question.meta`` (gold, accepted values, oracle notes), ``DecisionLog.gold`` / ``gold_aux`` and every
value an environment oracle returns -- must never be serialised into a request to a real backend.

Every question / state below carries a unique canary string in each gold-bearing field. Each backend or client that
serialises a request is driven through its real code down to the transport, where the outgoing request is captured:
``httpx.MockTransport`` under the Jev client, the OpenAI SDK clients (``http_client=``) and the Responses
client; the tokenizer input and the model's forward-pass ``input_ids`` for the in-process deciders (Intern-Decision,
``training/serve_jev.py``). The canary must appear nowhere in the URL, headers or body bytes, or in the prompt text.
Each test also checks that requests were actually captured and that the canary was really present on the input side
(no vacuous pass). The harness test runs the full agent loop on JevGym with an oracle whose every answer carries a
canary, a Jev-wire System-1 and an OpenAI-compatible System-2 behind mock transports.

Whitelisted ``Question.meta`` keys that ARE sent by design: Noul ``criteria`` (the Jev wire field; ``jev.py``) and
``wire_id`` (the client's own field name, set by ``decision/server.py``). They never hold gold labels.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any

import httpx
import pytest

from jevharness.accounting import Ledger
from jevharness.decision import CachedBackend, Choice, Noul, Score, make_backend
from jevharness.decision.base import DecisionBackend
from jevharness.decision.types import DecisionResult, Question, Usage, answer_from_distribution

CANARY = "CANARY-9d41e7-GOLD"
TOOLS = ["find_user", "get_orders", "cancel_order", "refund", "finish"]


def _c(tag: str) -> str:
    return f"{CANARY}-{tag}"


def canary_questions(n_tools: int = 5) -> list[Question]:
    """Every question kind, with canaries in every meta key a harness / benchmark puts gold into."""
    tools = TOOLS if n_tools <= len(TOOLS) else TOOLS + [f"distractor_{i:03d}" for i in range(n_tools - len(TOOLS))]
    gold_meta = {"gold": _c("gold"), "accept": [_c("accept")], "gold_aux": {"v": 1, "note": _c("aux")},
                 "oracle": {"verdict": _c("verdict"), "args": {"order_id": _c("arg")}}, "answer": _c("answer"),
                 "sim_gold": True}
    qs = [Choice("Which tool should the agent call next?", tools[:n_tools],
                 [f"Tool {t}." for t in tools[:n_tools]], decision_point="tool_select", **gold_meta),
          Noul("Is the task complete?", decision_point="stop", **{**gold_meta, "gold": True}),
          Noul("Should the agent reflect now?", decision_point="reflection",
               criteria={"true": "the last step failed", "false": "progress is fine"}, gold=_c("noul-gold")),
          Score("How risky is the next action?", levels=["none", "low", "high"], scale=(0, 2), decision_point="risk",
                gold=_c("score-gold"), notes=[_c("note")])]
    for q in qs:
        assert CANARY in json.dumps(q.meta, default=str)  # the canary IS on the input side
    return qs


STATE = "TASK: Cancel Maya Chen's pending order.\n\nCURRENT OBSERVATION:\nYou have access to the tools."


class Wire:
    """Records every outgoing request at the transport layer and answers it with ``respond(request) -> json``."""

    def __init__(self, respond: Any):
        self.respond = respond
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request.read()
        self.requests.append({"method": request.method, "url": str(request.url),
                              "headers": dict(request.headers), "body": body})
        out = self.respond(request, json.loads(body) if body else {})
        if isinstance(out, httpx.Response):
            return out
        return httpx.Response(200, json=out, headers={"x-request-id": "req_test"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def transport2(self) -> Any:
        """The same recorder for SDKs built on ``httpx2``."""
        import httpx2

        def handle(request: Any) -> Any:
            body = request.read()
            self.requests.append({"method": request.method, "url": str(request.url),
                                  "headers": dict(request.headers), "body": body})
            return httpx2.Response(200, json=self.respond(request, json.loads(body) if body else {}))

        return httpx2.MockTransport(handle)

    def blob(self) -> str:
        return "\n".join(f"{r['method']} {r['url']}\n{json.dumps(r['headers'])}\n{r['body'].decode('utf-8', 'replace')}"
                         for r in self.requests)

    def assert_clean(self, min_requests: int = 1, key_names: bool = True) -> None:
        """No canary anywhere; with ``key_names`` also no gold-bearing meta key name (off for JevGym states, whose
        customers have a "gold" tier)."""
        assert len(self.requests) >= min_requests, "no request captured: the test would pass vacuously"
        blob = self.blob()
        assert CANARY not in blob, blob[max(0, blob.find(CANARY) - 300): blob.find(CANARY) + 100]
        if key_names:
            assert not any(k in blob for k in ('"gold"', "gold_aux", "sim_gold")), "a meta key name leaked"


# ------------------------------------------------------------------ provider response fakes
def jev_answers(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any]:
    ans: dict[str, Any] = {}
    for k, q in (body.get("questions") or {}).items():
        if q["type"] == "choice":
            labels = list(q["criteria"])
            p = {lab: (0.9 if i == 0 else 0.1 / max(1, len(labels) - 1)) for i, lab in enumerate(labels)}
            ans[k] = {"type": "choice", "choice": labels[0], "confidence": 0.9, "probabilities": p}
        elif q["type"] == "noul":
            ans[k] = {"type": "noul", "noul": 0.7}
        else:
            n = len(q["criteria"])
            ans[k] = {"type": "score", "score": 1.0, "confidence": 0.5,
                      "probabilities": {str(i): 1.0 / n for i in range(n)}}
    return {"model": "jev-1.13.0", "answers": ans, "usage": {"input_tokens": 100, "output_tokens": 0}}


def chat_completion(content: str = "{}", tool_calls: list[dict[str, Any]] | None = None,
                    logprobs: dict[str, float] | None = None, model: str = "gemini-3.5-flash-lite") -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    choice: dict[str, Any] = {"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}
    if logprobs:
        top = [{"token": t, "logprob": lp, "bytes": None} for t, lp in logprobs.items()]
        choice["logprobs"] = {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "bytes": None,
                                           "top_logprobs": top}]}
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": model, "choices": [choice],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}}


def openai_sdk(wire: Wire, base_url: str) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key="test-key", max_retries=0,
                  http_client=httpx.Client(transport=wire.transport()))


def compat_llm(provider: str, model: str, wire: Wire, **kw: Any) -> Any:
    from jevharness.llm.openai_compat import PROVIDERS, OpenAICompatClient

    llm = OpenAICompatClient(model, provider=provider, api_key="test-key", **kw)
    llm.clients = [openai_sdk(wire, PROVIDERS[provider][0] or "https://api.openai.com/v1")]
    llm.client = llm.clients[0]
    return llm


def _decide_all(backend: DecisionBackend, qs: list[Question] | None = None) -> DecisionResult:
    qs = qs if qs is not None else canary_questions()
    return backend.decide(STATE, qs, ledger=Ledger())


# ------------------------------------------------------------------ Jev wire: hosted (typesafe / OpenRouter / codiv)
@pytest.mark.parametrize("provider,path", [("typesafe", "/v1/systemone"), ("openrouter", "/api/v1/systemone"),
                                           ("codiv", "/v1/systemone")])
def test_jev_wire_requests_carry_no_canary(provider, path):
    from jevharness.decision.jev import JevBackend, question_to_wire

    for q in canary_questions():
        assert CANARY not in json.dumps(question_to_wire(q))
    wire = Wire(jev_answers)
    b = JevBackend(provider=provider, api_key="test-key", transport=wire.transport())
    res = _decide_all(b)
    assert len(res.answers) == 4 and wire.requests[0]["url"].endswith(path)
    assert wire.requests[0]["headers"]["authorization"] == "Bearer test-key"
    sent = json.loads(wire.requests[0]["body"])
    assert sent["questions"]["q2"]["criteria"] == {"true": "the last step failed", "false": "progress is fine"}
    wire.assert_clean()
    # a 255-option Choice and > max_questions_per_request questions (several requests) stay clean too
    b2 = JevBackend(provider=provider, api_key="test-key", transport=wire.transport(), max_questions_per_request=2)
    _decide_all(b2, canary_questions(255) + canary_questions())
    wire.assert_clean(min_requests=5)


def test_openjev_calibrated_and_cached_wrappers_carry_no_canary(tmp_path):
    from jevharness.decision.calibration import TemperatureScaledBackend
    from jevharness.decision.jev import OpenJevBackend

    wire = Wire(jev_answers)
    inner = OpenJevBackend(base_url="http://localhost:8080", api_key="", transport=wire.transport())
    cached = CachedBackend(TemperatureScaledBackend(inner, temperature=1.3), tmp_path / "s1.sqlite")
    _decide_all(cached)
    _decide_all(cached)  # replayed from the cache: no new request
    wire.assert_clean()
    assert len(wire.requests) == 1
    assert CANARY.encode() not in (tmp_path / "s1.sqlite").read_bytes()  # nor in the on-disk cache


# ------------------------------------------------------------------ our wire servers (client side + server side)
class Recorder(DecisionBackend):
    """What a backend behind ``decision/server.py`` receives."""
    name = "recorder"
    batched_questions = True

    def __init__(self) -> None:
        self.seen: list[str] = []

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        self.seen.append(state + json.dumps([q.model_dump(mode="json") for q in questions]))
        for q in questions:
            assert set(q.meta) <= {"wire_id", "criteria"}, q.meta
        ans = [answer_from_distribution(q, {lab: 1.0 for lab in q.labels()}) for q in questions]
        return DecisionResult(answers=ans, usage=Usage(backend=self.name, n_calls=1))


def _testclient_wire(app: Any, wire: Wire) -> Any:
    """A TestClient for ``app`` whose outgoing requests are recorded byte for byte by ``wire``."""
    from fastapi.testclient import TestClient

    tc = TestClient(app)

    def hook(request: httpx.Request) -> None:
        body = request.read()
        wire.requests.append({"method": request.method, "url": str(request.url), "headers": dict(request.headers),
                              "body": body})

    tc.event_hooks = {"request": [hook], "response": []}
    return tc


def test_generic_wire_server_and_its_upstream_carry_no_canary():
    pytest.importorskip("fastapi")
    from jevharness.decision.jev import JevBackend
    from jevharness.decision.llm_generative import GenerativeLLMBackend
    from jevharness.decision.server import build_app

    # client -> decision/server.py -> a recording backend
    rec = Recorder()
    wire = Wire(jev_answers)
    client = JevBackend(model="s1", base_url="http://testserver", api_key="")
    client._http = _testclient_wire(build_app(rec, name="s1"), wire)
    res = _decide_all(client)
    assert len(res.answers) == 4 and rec.seen and CANARY not in "".join(rec.seen)
    wire.assert_clean()
    # client -> decision/server.py -> a Gemini verbalised decider -> the (mock) Gemini endpoint
    upstream = Wire(lambda r, b: chat_completion('{"Q1": {"A": 0.6, "B": 0.4}}'))
    gen = GenerativeLLMBackend(compat_llm("gemini", "gemini-3.5-flash-lite", upstream), reasoning_effort="minimal")
    wire2 = Wire(jev_answers)
    client._http = _testclient_wire(build_app(gen, name="gemini-s1"), wire2)
    _decide_all(client)
    wire2.assert_clean()
    upstream.assert_clean()


def test_serve_jev_client_server_and_prompts_carry_no_canary(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    from jevharness.decision.jev import JevBackend
    from jevharness.training import serve_jev

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
    from make_tiny_model import make_tiny_model

    eng = serve_jev.Engine(str(make_tiny_model(tmp_path / "tiny")), name="ours", session_cache=True)
    prompts: list[str] = []
    real_render = serve_jev.render

    def spy_render(ex: dict[str, Any]) -> Any:
        out = real_render(ex)
        prompts.append(out[0])
        return out

    monkeypatch.setattr(serve_jev, "render", spy_render)
    fed: list[str] = []

    def pre_hook(module: Any, args: Any, kwargs: Any) -> None:
        ids = kwargs.get("input_ids") if kwargs.get("input_ids") is not None else (args[0] if args else None)
        if ids is not None:
            fed.extend(eng.tok.batch_decode(ids.reshape(-1, ids.shape[-1])))

    h = eng.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    wire = Wire(jev_answers)
    client = JevBackend(model="ours", base_url="http://testserver", api_key="")
    client._http = _testclient_wire(serve_jev.build_app(eng), wire)
    try:
        res = _decide_all(client, canary_questions() + canary_questions(60))  # 60 options: knockout passes too
    finally:
        h.remove()
    assert len(res.answers) == 8 and res.usage.n_passes and res.usage.n_rounds == 2
    wire.assert_clean()
    assert prompts and fed and all(CANARY not in p for p in prompts + fed)
    assert any("Which tool should the agent call next?" in p for p in prompts)


# ------------------------------------------------------------------ LLM deciders: Gemini / vLLM / OpenAI
@pytest.mark.parametrize("batch", [True, False])
def test_gemini_generative_and_s2_self_requests_carry_no_canary(batch):
    from jevharness.decision.self_llm import SelfLLMBackend
    from jevharness.decision.llm_generative import GenerativeLLMBackend

    wire = Wire(lambda r, b: chat_completion('{"Q1": {"A": 0.7, "B": 0.3}}'))
    for cls in (GenerativeLLMBackend, SelfLLMBackend):
        b = cls(compat_llm("gemini", "gemini-3.5-flash-lite", wire), batch=batch, reasoning_effort="minimal")
        _decide_all(b)
    assert all("generativelanguage.googleapis.com" in r["url"] for r in wire.requests)
    wire.assert_clean(min_requests=2 if batch else 8)


def test_vllm_logprob_requests_carry_no_canary():
    from jevharness.decision.llm_logprob import LogprobLLMBackend

    wire = Wire(lambda r, b: chat_completion("A", logprobs={"A": -0.2, "B": -1.8, "Yes": -0.4, "0": -1.0},
                                             model="Qwen/Qwen3-4B"))
    b = LogprobLLMBackend(compat_llm("vllm", "Qwen/Qwen3-4B", wire), n_permutations=2,
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    _decide_all(b)
    wire.assert_clean(min_requests=4)


def test_openai_responses_deciders_carry_no_canary():
    from jevharness.decision.llm_generative import GenerativeLLMBackend
    from jevharness.decision.llm_logprob import LogprobLLMBackend
    from jevharness.decision.self_llm import SelfLLMBackend
    from jevharness.llm.openai_responses import OpenAIResponsesClient

    def respond(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any]:
        part: dict[str, Any] = {"type": "output_text", "text": '{"Q1": {"A": 0.6, "B": 0.4}}', "annotations": []}
        if body.get("top_logprobs"):
            part = {"type": "output_text", "text": "A", "annotations": [],
                    "logprobs": [{"token": "A", "logprob": -0.1,
                                  "top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.4}]}]}
        return {"id": "resp_test", "object": "response", "status": "completed", "model": body["model"],
                "output": [{"type": "message", "id": "msg_1", "role": "assistant", "content": [part]}],
                "usage": {"input_tokens": 40, "output_tokens": 4}}

    wire = Wire(respond)
    gen = OpenAIResponsesClient("gpt-6-luna", api_key="test-key", transport=wire.transport(), reasoning_effort="low")
    _decide_all(GenerativeLLMBackend(gen))
    _decide_all(SelfLLMBackend(gen, batch=False))
    lp = OpenAIResponsesClient("gpt-6-luna", api_key="test-key", transport=wire.transport(), reasoning_effort="none")
    _decide_all(LogprobLLMBackend(lp))
    assert all(r["url"].endswith("/responses") for r in wire.requests)
    wire.assert_clean(min_requests=1 + 4 + 4)




@pytest.mark.parametrize("provider,model", [("openai", "text-embedding-3-small"), ("gemini", "gemini-embedding-001")])
def test_embedding_requests_carry_no_canary(provider, model):
    from jevharness.decision.embedding import EmbeddingBackend
    from jevharness.llm.openai_compat import PROVIDERS

    def respond(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any]:
        data = [{"object": "embedding", "index": i, "embedding": [float(len(t) % 7), 1.0, float(i % 3)]}
                for i, t in enumerate(body["input"])]
        return {"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": 9, "total_tokens": 9}}

    wire = Wire(respond)
    b = EmbeddingBackend(model=model, provider=provider, api_key="test-key")
    b.client = openai_sdk(wire, PROVIDERS[provider][0] or "https://api.openai.com/v1")
    for q in (canary_questions()[0], canary_questions(60)[0]):
        b.decide(STATE, [q], ledger=Ledger())
    wire.assert_clean(min_requests=2)


# ------------------------------------------------------------------ local baselines: no request at all
def test_local_baselines_open_no_connection_and_ignore_meta(monkeypatch):
    def no_network(*a: Any, **k: Any) -> None:
        raise AssertionError("a local baseline opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    for spec in ({"type": "bm25"}, {"type": "tfidf"}, {"type": "rate", "rate": 0.3},
                 {"type": "prior", "rule": "recent"},
                 {"type": "heuristic"}):
        b = make_backend(spec)
        qs = canary_questions()[:1] if spec["type"] in ("bm25", "tfidf") else canary_questions()[:3]
        plain = [q.model_copy(update={"meta": {}}) for q in qs]
        r1, r2 = b.decide(STATE, qs), b.decide(STATE, plain)
        assert [a.distribution for a in r1.answers] == [a.distribution for a in r2.answers], spec


# ------------------------------------------------------------------ Intern-Decision: tokenizer input + forward ids
from test_intern_decision import tiny_ckpt  # noqa: E402,F401  (module-scoped tiny Qwen3.5 checkpoint fixture)


class TokSpy:
    def __init__(self, tok: Any, seen: list[str]):
        self._tok, self._seen = tok, seen

    def __getattr__(self, k: str) -> Any:
        return getattr(self._tok, k)

    def __call__(self, text: Any, *a: Any, **k: Any) -> Any:
        self._seen.append(str(text))
        return self._tok(text, *a, **k)

    def apply_chat_template(self, messages: Any, *a: Any, **k: Any) -> Any:
        self._seen.append(json.dumps(messages))
        return self._tok.apply_chat_template(messages, *a, **k)


def test_intern_decision_prompt_and_forward_ids_carry_no_canary(tiny_ckpt, monkeypatch):  # noqa: F811
    for opts in ({}, {"max_options": 3}):  # plain fields, then a knockout over 5 options (several passes)
        b = make_backend({"type": "intern_decision", "model": str(tiny_ckpt), "device": "cpu", **opts})
        seen: list[str] = []
        monkeypatch.setattr(b, "tok", TokSpy(b.tok, seen))
        real_forward = b._forward

        def spy(ids: Any, positions: Any, b: Any = b, real_forward: Any = real_forward) -> Any:
            seen.append(b.tok.decode(ids[0]))
            return real_forward(ids, positions)

        monkeypatch.setattr(b, "_forward", spy)
        res = _decide_all(b)
        assert len(res.answers) == 4 and len(seen) >= 3
        assert all(CANARY not in s for s in seen) and any("Which tool should the agent call next?" in s for s in seen)
        assert "gold" not in "".join(seen).lower()


# ------------------------------------------------------------------ the harness: full agent loop, canary oracle
class CanaryOracleEnv:
    """A JevGym episode seen through a proxy whose every oracle answer carries a canary (gold label, oracle
    arguments, acceptable-tool sets, verdicts, memory golds); ``reset`` / ``step`` / ``evaluate`` are the real env's.
    ``canary_tool`` also replaces the oracle's tool name (else only its arguments and the other golds are canaries)."""

    def __init__(self, env: Any, canary_tool: bool):
        self._env, self._canary_tool = env, canary_tool

    def __getattr__(self, k: str) -> Any:
        return getattr(self._env, k)

    def oracle_action(self) -> Any:
        a = self._env.oracle_action()
        if a is None:
            return None
        tool, args = a
        return (_c("tool") if self._canary_tool else tool), {k: _c(f"arg-{k}") for k in args} or {"x": _c("arg")}

    def oracle_acceptable_tool_sets(self, budget: int = 30) -> Any:
        f, u = self._env.oracle_acceptable_tool_sets(budget)
        return f + [_c("AT")], u + [_c("AT")]

    def oracle_verdict(self, *a: Any, **k: Any) -> Any:
        return {**self._env.oracle_verdict(*a, **k), "note": _c("verdict")}

    def oracle_mem_gold(self, *a: Any, **k: Any) -> Any:
        return {**self._env.oracle_mem_gold(*a, **k), "note": _c("mem")}

    def oracle_memory_read_gold(self, *a: Any, **k: Any) -> Any:
        return {**self._env.oracle_memory_read_gold(*a, **k), "note": _c("memread")}

    def oracle_snapshot(self) -> Any:
        return CanaryOracleEnv(self._env.oracle_snapshot(), self._canary_tool)


def s2_policy(env: Any):  # noqa: ANN201
    """The mock System-2 endpoint: a competent LLM (it follows the TRUE plan of the real env; test-side knowledge,
    never a request field). Replies hold no canary, so anything found in a later request leaked from the harness."""

    def respond(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any]:
        tools = [t["function"]["name"] for t in body.get("tools") or []]
        if not tools:
            return chat_completion("Reflection: the plan is on track.")
        forced = ((body.get("tool_choice") or {}).get("function") or {}).get("name") if isinstance(
            body.get("tool_choice"), dict) else None
        a = env.oracle_action()
        name, args = a if a is not None else ("finish", {"answer": "done"})
        if forced:
            args = args if forced == name else ({"answer": "done"} if forced == "finish" else {})
            name = forced
        elif name not in tools:
            name, args = tools[0], {}
        return chat_completion("", tool_calls=[{"id": f"call_{time.perf_counter_ns() % 10**9}", "type": "function",
                                                "function": {"name": name, "arguments": json.dumps(args)}}])

    return respond


def s1_policy(env: Any):  # noqa: ANN201
    """The mock Jev endpoint: confident on the true next tool / argument value when it is offered (test-side
    knowledge of the real env, like ``s2_policy``), 0.7 on every Noul."""

    def respond(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any]:
        out = jev_answers(request, body)
        a = env.oracle_action()
        right = {a[0], *(json.dumps(v) if not isinstance(v, str) else v for v in a[1].values())} if a else set()
        for k, q in body["questions"].items():
            hit = [lab for lab in (q.get("criteria") or {}) if lab in right] if q["type"] == "choice" else []
            if hit:
                labels = list(q["criteria"])
                p = {lab: (0.95 if lab == hit[0] else 0.05 / (len(labels) - 1)) for lab in labels}
                out["answers"][k] = {"type": "choice", "choice": hit[0], "confidence": 0.95, "probabilities": p}
        return out

    return respond


HARNESS_CONFIGS = {
    "live": dict(tool_select={"mode": "cascade", "tau": 0.9}, tool_params={"mode": "cascade", "tau": 0.5},
                 memory_write={"mode": "trigger", "tau": 0.5}, memory_read={"mode": "assist", "k": 2},
                 reflection={"mode": "trigger", "tau": 0.5}, stop={"mode": "guard", "tau": 0.5},
                 routing={"mode": "trigger", "tau": 0.5}),
    "assist_guard": dict(tool_select={"mode": "assist", "k": 3}, stop={"mode": "trigger", "tau": 0.95},
                         memory_write={"mode": "shadow"}),
    "guard": dict(tool_select={"mode": "guard", "tau": 0.5}, tool_params={"mode": "replace"}),
    "shadow_measure": dict(tool_select={"mode": "shadow"}, tool_params={"mode": "shadow"}, stop={"mode": "shadow"},
                           reflection={"mode": "shadow", "wordings": ["v2", "v1"]}, routing={"mode": "shadow"},
                           measure={"cf": {"reflection": True, "placebo": True, "stop_reject": True,
                                           "sample_negatives": 1.0},
                                    "probes": {"stop": ["natural", "forced_premature", "T5", "T6"],
                                               "tool_select": ["natural", "T1", "T2", "T3", "T4"],
                                               "rho_edit": 1.0, "rho_t5": 1.0}}),
}


@pytest.mark.parametrize("canary_tool", [True, False])
@pytest.mark.parametrize("arm", list(HARNESS_CONFIGS))
@pytest.mark.parametrize("s1_width", [None, 20])  # None: Jev's 255 options; 20: the harness tournament runs
def test_harness_agent_loop_leaks_no_oracle_value(arm, canary_tool, s1_width):
    from jevharness.bench.suites.jevgym import JevGymEnv
    from jevharness.decision.jev import JevBackend
    from jevharness.harness.agent import JevAgent
    from jevharness.harness.config import HarnessConfig

    s1_wire_all, s2_wire_all, dumps = Wire(jev_answers), Wire(jev_answers), []
    for seed in (0, 3):
        real = JevGymEnv(seed=seed, families="v2", distractor_style="near", n_distractors=60, noise_lines=5,
                         strict=True)
        env = CanaryOracleEnv(real, canary_tool)
        s1_wire, s2_wire = Wire(s1_policy(real)), Wire(s2_policy(real))
        s1 = JevBackend(provider="openrouter", api_key="test-key", transport=s1_wire.transport())
        if s1_width:
            s1.max_options = s1_width
        llm = compat_llm("gemini", "gemini-3.5-flash", s2_wire, temperature=0, reasoning_effort="minimal")
        small = compat_llm("gemini", "gemini-3.5-flash-lite", s2_wire, temperature=0)
        cfg = HarnessConfig(name=arm, max_steps=12, **HARNESS_CONFIGS[arm])
        traj = JevAgent(llm, cfg, s1=s1, small_llm=small, measure_llms={"lite": small}).run(env)
        assert traj.error is None, traj.error
        s1_wire_all.requests += s1_wire.requests
        s2_wire_all.requests += s2_wire.requests
        dumps.append(json.dumps(traj.to_dict(), default=str))
    s1_wire_all.assert_clean(min_requests=2, key_names=False)
    s2_wire_all.assert_clean(min_requests=2, key_names=False)
    # positive control: the canaries did reach the gold side of the logs (the tool guard alone logs only boolean
    # verdicts and the oracle's tool name)
    assert any(CANARY in d for d in dumps) or (arm == "guard" and not canary_tool)
    if s1_width and arm != "guard":  # every arm but the guard asks the 69-tool Choice: 20-wide tournament chunks
        bodies = [json.loads(r["body"]) for r in s1_wire_all.requests]
        assert any(len(q.get("criteria") or {}) == s1_width for b in bodies for q in b["questions"].values())
