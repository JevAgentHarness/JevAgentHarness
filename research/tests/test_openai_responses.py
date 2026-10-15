"""Offline tests for the OpenAI Responses API provider (``llm/openai_responses.py``): request building, response
parsing, usage/cost, previous_response_id chaining, retries, logprobs and the no-key message. A scripted
``httpx.MockTransport`` stands in for api.openai.com; no network, no SDK."""

import json
import math
import time

import httpx
import pytest

from jevharness.accounting import Ledger
from jevharness.decision import Choice, Noul
from jevharness.decision.llm_generative import GenerativeLLMBackend
from jevharness.decision.llm_logprob import LogprobLLMBackend
from jevharness.llm import make_llm
from jevharness.llm.openai_responses import (LogprobsUnavailableError, OpenAIResponsesClient, OpenAIResponsesError,
                                             messages_to_input, tools_to_responses)

TOOLS = [{"type": "function", "function": {"name": "search", "description": "d",
                                           "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]


def _usage(inp=100, out=10, cached=0, writes=0, reasoning=0):
    return {"input_tokens": inp, "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": writes},
            "output_tokens": out, "output_tokens_details": {"reasoning_tokens": reasoning}, "total_tokens": inp + out}


def _resp(output, model="gpt-6-luna", usage=None, rid="resp_1", status="completed", **extra):
    return {"id": rid, "object": "response", "status": status, "model": model, "output": output,
            "usage": usage or _usage(), **extra}


def _fc(call_id, name, args, item_id="fc_1"):
    return {"type": "function_call", "id": item_id, "call_id": call_id, "name": name,
            "arguments": json.dumps(args), "status": "completed"}


def _msg(text, logprobs=None):
    part = {"type": "output_text", "text": text, "annotations": []}
    if logprobs is not None:
        part["logprobs"] = logprobs
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [part]}


def _reasoning(rid="rs_1", enc="ENC"):
    return {"type": "reasoning", "id": rid, "summary": [], "encrypted_content": enc}


class Server:
    """Scripted fake api.openai.com: records every request body, answers from a queue (dict, Response or callable)."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.headers: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        assert self.responses, "unexpected extra request"
        nxt = self.responses.pop(0)
        if isinstance(nxt, httpx.Response):
            return nxt
        if callable(nxt):
            return nxt(request)
        return httpx.Response(200, json=nxt)


def client(server, model="gpt-6-luna", **kw):
    return OpenAIResponsesClient(model, api_key="sk-test", transport=httpx.MockTransport(server), **kw)


# ------------------------------------------------------------------ request building
def test_request_building_system_tools_and_history_round_trip():
    srv = Server([_resp([_fc("call_1", "search", {"q": "x"})])])
    c = client(srv, reasoning_effort="none", temperature=0)
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "lookup", "arguments": '{"k": 1}'},
                 "extra_content": {"openai": {"item_id": "fc_0", "reasoning": [
                     {"id": "rs_0", "summary": [], "encrypted_content": "E0"}]}}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": "found"}]
    r = c.chat(msgs, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "search"}}, max_tokens=5)
    body = srv.requests[0]
    assert srv.headers[0]["authorization"] == "Bearer sk-test"
    assert body["model"] == "gpt-6-luna" and body["store"] is False and body["instructions"] == "SYS"
    assert body["reasoning"] == {"effort": "none"} and body["temperature"] == 0
    assert body["input"] == [
        {"role": "user", "content": "hi"},
        {"type": "reasoning", "id": "rs_0", "summary": [], "encrypted_content": "E0"},
        {"type": "function_call", "id": "fc_0", "call_id": "call_0", "name": "lookup", "arguments": '{"k": 1}'},
        {"type": "function_call_output", "call_id": "call_0", "output": "found"}]
    assert body["tools"] == [{"type": "function", "name": "search", "description": "d",
                              "parameters": TOOLS[0]["function"]["parameters"]}]
    assert body["tool_choice"] == {"type": "function", "name": "search"}
    assert body["max_output_tokens"] == 16          # documented minimum
    assert "include" not in body and "previous_response_id" not in body
    assert r.tool_calls[0].id == "call_1" and r.tool_calls[0].name == "search"
    assert r.tool_calls[0].arguments == {"q": "x"} and r.finish_reason == "tool_calls"


def test_messages_to_input_edge_cases_and_strict_tools():
    _, items = messages_to_input([{"role": "user", "content": [{"type": "text", "text": "look"},
                                                               {"type": "image_url", "image_url": {"url": "u"}}]},
                                  {"role": "assistant", "content": "ok"},
                                  {"role": "system", "content": "note"},   # mid-conversation note stays in place
                                  {"role": "tool", "tool_call_id": "c", "content": {"a": 1}}])
    assert items[0] == {"role": "user", "content": [{"type": "input_text", "text": "look"},
                                                    {"type": "input_image", "image_url": "u"}]}
    assert items[1] == {"role": "assistant", "content": "ok"} and items[2] == {"role": "system", "content": "note"}
    assert items[3] == {"type": "function_call_output", "call_id": "c", "output": '{"a": 1}'}
    ins, items = messages_to_input([{"role": "system", "content": "a"}, {"role": "developer", "content": "b"}])
    assert ins == "a\n\nb" and items == []
    assert tools_to_responses(TOOLS, strict=True)[0]["strict"] is True
    with pytest.raises(ValueError):
        messages_to_input([{"role": "bogus", "content": ""}])


def test_reasoning_effort_drops_sampling_params_and_echoes_reasoning_items():
    srv = Server([_resp([_reasoning(), _fc("call_1", "search", {"q": "x"})], model="gpt-6.1-sol"),
                  _resp([_msg("done")], model="gpt-6.1-sol")])
    c = client(srv, model="gpt-6.1-sol", reasoning_effort="low", temperature=0)
    r = c.chat([{"role": "user", "content": "x"}], tools=TOOLS)
    body = srv.requests[0]
    assert "temperature" not in body and body["reasoning"] == {"effort": "low"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert c.stats["dropped_params"] == {"temperature": 1}
    tc = r.tool_calls[0]
    assert tc.extra == {"openai": {"item_id": "fc_1",
                                   "reasoning": [{"id": "rs_1", "summary": [], "encrypted_content": "ENC"}]}}
    # the harness echoes ToolCall.extra as extra_content; the next request replays the reasoning item untouched
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [tc.to_openai()]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"}]
    r2 = c.chat(msgs, tools=TOOLS)
    inp = srv.requests[1]["input"]
    assert inp[1] == {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}
    assert inp[2]["type"] == "function_call" and inp[2]["id"] == "fc_1" and inp[2]["call_id"] == "call_1"
    assert inp[3] == {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    assert r2.text == "done" and r2.finish_reason == "stop" and r2.model == "gpt-6.1-sol"
    with pytest.raises(ValueError):
        c.chat([{"role": "user", "content": "x"}], reasoning_effort="turbo")


def test_text_only_reply_keeps_reasoning_in_response_extra():
    srv = Server([_resp([_reasoning("rs_9", "E9"), _msg("answer")], model="gpt-5.4-nano"),
                  _resp([_msg("more")], model="gpt-5.4-nano")])
    c = client(srv, model="gpt-5.4-nano")   # reasoning family, effort unset: include + no sampling params
    r = c.chat([{"role": "user", "content": "q"}], temperature=0)
    assert srv.requests[0]["include"] == ["reasoning.encrypted_content"] and "temperature" not in srv.requests[0]
    assert r.extra == {"openai": {"reasoning": [{"id": "rs_9", "summary": [], "encrypted_content": "E9"}]}}
    msg = r.assistant_message()
    assert msg["extra_content"] == r.extra and msg["content"] == "answer"
    c.chat([{"role": "user", "content": "q"}, msg, {"role": "user", "content": "again"}])
    inp = srv.requests[1]["input"]
    assert inp[1] == {"type": "reasoning", "id": "rs_9", "summary": [], "encrypted_content": "E9"}
    assert inp[2] == {"role": "assistant", "content": "answer"} and inp[3] == {"role": "user", "content": "again"}


def test_unexpected_400_on_temperature_is_retried_without_it_once_and_remembered():
    def reject(request):
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature' is not "
                                                              "supported with this model.", "param": "temperature"}})

    srv = Server([reject, _resp([_msg("ok")], model="gpt-4.1-mini"), _resp([_msg("ok")], model="gpt-4.1-mini")])
    c = client(srv, model="gpt-4.1-mini", temperature=0)
    r = c.chat([{"role": "user", "content": "x"}])
    assert r.text == "ok" and "temperature" in srv.requests[0] and "temperature" not in srv.requests[1]
    assert c.stats["dropped_params"] == {"temperature": 1} and c._sampling_rejected
    c.chat([{"role": "user", "content": "x"}])
    assert len(srv.requests) == 3 and "temperature" not in srv.requests[2]   # not even tried again


def test_other_400s_and_failed_responses_raise_with_the_api_message():
    srv = Server([httpx.Response(400, json={"error": {"message": "Invalid schema for function 'search'"}})])
    with pytest.raises(OpenAIResponsesError, match="Invalid schema") as ei:
        client(srv).chat([{"role": "user", "content": "x"}], tools=TOOLS)
    assert ei.value.status == 400
    srv = Server([_resp([], status="failed", error={"message": "server exploded"})])
    with pytest.raises(OpenAIResponsesError, match="server exploded"):
        client(srv).chat([{"role": "user", "content": "x"}])
    srv = Server([_resp([_msg("trunc")], status="incomplete", incomplete_details={"reason": "max_output_tokens"})])
    assert client(srv).chat([{"role": "user", "content": "x"}]).finish_reason == "length"


# ------------------------------------------------------------------ usage, cost, ledger
def test_usage_and_cost_mapping_with_cached_tokens_and_snapshot_ids():
    srv = Server([_resp([_msg("hi")], model="gpt-6-luna-2026-09-01",
                        usage=_usage(inp=1000, out=50, cached=400, writes=100, reasoning=20))])
    c = client(srv)
    led = Ledger()
    r = c.chat([{"role": "user", "content": "x"}], ledger=led, decision_point="tool_select")
    assert (r.input_tokens, r.cached_input_tokens, r.output_tokens, r.reasoning_tokens) == (1000, 400, 50, 20)
    assert r.model == "gpt-6-luna"                      # dated snapshot id -> configured alias (priced)
    rec = led.records[0]
    assert rec.backend == "openai_responses" and rec.model == "gpt-6-luna" and rec.priced
    assert math.isclose(rec.cost_usd, (600 * 0.10 + 400 * 0.01 + 50 * 0.50) / 1e6)   # cached price applied
    assert rec.meta["cached_input_tokens"] == 400 and rec.meta["cache_write_tokens"] == 100
    assert rec.meta["chained"] is False and rec.meta["wire_input_tokens_est"] > 0 and rec.meta["response_id"] == "resp_1"
    assert led.summary()["unpriced_models"] == []
    assert c.stats["input_tokens"] == 1000 and c.stats["cached_input_tokens"] == 400
    # an unknown model is flagged, not silently free
    srv = Server([_resp([_msg("hi")], model="gpt-99-mystery")])
    led = Ledger()
    client(srv, model="gpt-99-mystery").chat([{"role": "user", "content": "x"}], ledger=led)
    assert led.summary()["unpriced_models"] == ["gpt-99-mystery"]


# ------------------------------------------------------------------ previous_response_id chaining
def test_server_state_sends_only_new_items_and_breaks_on_divergence():
    srv = Server([_resp([_fc("call_1", "search", {"a": 1})], rid="resp_1"), _resp([_msg("final")], rid="resp_2"),
                  _resp([_msg("more")], rid="resp_3"), _resp([_msg("again")], rid="resp_4")])
    c = client(srv, server_state=True, reasoning_effort="none")
    assert c.store is True
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "task"}]
    r1 = c.chat(msgs, tools=TOOLS)
    assert srv.requests[0]["store"] is True and "previous_response_id" not in srv.requests[0]
    # turn 2: history = sent prefix + echo of the model's own call + the tool result -> only the delta goes out
    msgs = msgs + [{"role": "assistant", "content": None, "tool_calls": [r1.tool_calls[0].to_openai()]},
                   {"role": "tool", "tool_call_id": "call_1", "content": "result"}]
    r2 = c.chat(msgs, tools=TOOLS)
    b = srv.requests[1]
    assert b["previous_response_id"] == "resp_1" and b["instructions"] == "SYS"
    assert b["input"] == [{"type": "function_call_output", "call_id": "call_1", "output": "result"}]
    assert r2.raw["wire"]["chained"] is True and r2.raw["wire"]["wire_items"] == 1
    # turn 3: the echoed text reply plus a new user message chains as well
    msgs = msgs + [{"role": "assistant", "content": "final"}, {"role": "user", "content": "more?"}]
    c.chat(msgs, tools=TOOLS)
    b = srv.requests[2]
    assert b["previous_response_id"] == "resp_2" and b["input"] == [{"role": "user", "content": "more?"}]
    assert c.stats["chained_calls"] == 2 and c.stats["full_sends"] == 1 and c.stats["chain_breaks"] == 0
    # divergence (System-1 replaced the model's call): full stateless send, chain break counted
    diverged = msgs[:2] + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_s1", "type": "function", "function": {"name": "other", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_s1", "content": "r"}]
    c.chat(diverged, tools=TOOLS)
    b = srv.requests[3]
    assert "previous_response_id" not in b and len(b["input"]) == 3 and c.stats["chain_breaks"] == 1
    assert c.stats["full_sends"] == 2
    # a new episode on the same worker thread is not a break: it is a new conversation
    srv.responses.append(_resp([_msg("fresh")], rid="resp_5"))
    c.chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "another task"}])
    assert "previous_response_id" not in srv.requests[4]
    assert c.stats["chain_breaks"] == 1 and c.stats["new_conversations"] == 1 and c.stats["full_sends"] == 3
    with pytest.raises(ValueError):
        OpenAIResponsesClient("m", api_key="k", server_state=True, store=False)


def test_server_state_lost_previous_response_falls_back_to_a_full_send():
    def gone(request):
        return httpx.Response(404, json={"error": {"message": "Previous response with id 'resp_1' not found."}})

    srv = Server([_resp([_msg("a")], rid="resp_1"), gone, _resp([_msg("b")], rid="resp_2")])
    c = client(srv, server_state=True)
    msgs = [{"role": "user", "content": "x"}]
    c.chat(msgs)
    msgs = msgs + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "y"}]
    r = c.chat(msgs)
    assert r.text == "b" and srv.requests[1]["previous_response_id"] == "resp_1"
    assert "previous_response_id" not in srv.requests[2] and len(srv.requests[2]["input"]) == 3
    assert c.stats["chain_breaks"] == 1


# ------------------------------------------------------------------ retries
def test_retry_on_429_with_retry_after_and_latency_excludes_backoff(monkeypatch):
    n = {"calls": 0}

    def flaky(request):
        n["calls"] += 1
        if n["calls"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}}, headers={"retry-after": "0"})
        return httpx.Response(200, json=_resp([_msg("ok")]))

    srv = Server([flaky, flaky])
    c = client(srv, max_retries=2)
    real_sleep = time.sleep
    monkeypatch.setattr("time.sleep", lambda s: real_sleep(0.2))
    r = c.chat([{"role": "user", "content": "x"}])
    assert n["calls"] == 2 and c.n_retries == 1 and r.text == "ok"
    assert r.latency_ms < 150, r.latency_ms and r.raw["wire"]["attempts"] == 2
    # exhausted retries surface the last error
    srv = Server([httpx.Response(503, json={"error": {"message": "overloaded"}})] * 2)
    with pytest.raises(OpenAIResponsesError, match="overloaded"):
        client(srv, max_retries=1).chat([{"role": "user", "content": "x"}])


# ------------------------------------------------------------------ logprobs (System-1 via llm_logprob)
LP = [{"token": "B", "logprob": -0.1, "top_logprobs": [{"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -2.5},
                                                        {"token": " C", "logprob": -4.0}]},
      {"token": ".", "logprob": -0.01, "top_logprobs": []}]


def test_logprob_extraction_when_present():
    srv = Server([_resp([_msg("B.", logprobs=LP)])])
    c = client(srv, reasoning_effort="none")
    r = c.chat([{"role": "user", "content": "pick"}], top_logprobs=5, max_tokens=1, temperature=0)
    b = srv.requests[0]
    assert b["top_logprobs"] == 5 and b["include"] == ["message.output_text.logprobs"]
    assert b["max_output_tokens"] == 16 and b["temperature"] == 0
    assert r.top_logprobs == [{"B": -0.1, "A": -2.5, " C": -4.0}, {".": -0.01}]
    srv = Server([_resp([_msg("B.", logprobs=LP)])])
    d = client(srv, reasoning_effort="none").option_logprobs([{"role": "user", "content": "pick"}], ["A", "B", "C", "D"])
    assert d == {"A": -2.5, "B": -0.1, "C": -4.0, "D": -math.inf}


def test_logprobs_absent_or_unsupported_raise_a_clear_error():
    # the API silently returns no logprobs
    c = client(Server([_resp([_msg("B")])]), reasoning_effort="none")
    with pytest.raises(LogprobsUnavailableError, match="no logprobs"):
        c.chat([{"role": "user", "content": "pick"}], top_logprobs=5)
    # ... unless the caller opts into the sampled-token fallback of the decider
    c = client(Server([_resp([_msg("B")])]), reasoning_effort="none", allow_missing_logprobs=True)
    assert c.chat([{"role": "user", "content": "pick"}], top_logprobs=5).top_logprobs is None
    # reasoning effort other than none: refused before any request is sent
    srv = Server([])
    c = client(srv, model="gpt-6.1-sol", reasoning_effort="low")
    with pytest.raises(LogprobsUnavailableError, match="reasoning effort 'none'"):
        c.chat([{"role": "user", "content": "pick"}], top_logprobs=5)
    assert srv.requests == []
    # the API rejects top_logprobs for this model
    def reject(request):
        return httpx.Response(400, json={"error": {"message": "top_logprobs is not supported with this model"}})
    c = client(Server([reject]), model="gpt-4.1-mini")
    with pytest.raises(LogprobsUnavailableError, match="rejected logprobs"):
        c.chat([{"role": "user", "content": "pick"}], top_logprobs=5)


def test_logprob_and_verbalised_deciders_run_on_the_responses_client():
    lp = [{"token": "B", "logprob": math.log(0.7), "top_logprobs": [{"token": "B", "logprob": math.log(0.7)},
                                                                     {"token": "A", "logprob": math.log(0.3)}]}]
    c = client(Server([_resp([_msg("B", logprobs=lp)])]), reasoning_effort="none")
    r = LogprobLLMBackend(c).decide("S", [Choice("which?", ["alpha", "beta"])])
    assert r.answers[0].value == "beta" and r.usage.n_calls == 1 and r.usage.priced
    assert math.isclose(r.answers[0].distribution["beta"], 0.7, abs_tol=1e-6)
    reply = json.dumps({"Q1": {"A": 0.2, "B": 0.8}, "Q2": {"yes": 0.9, "no": 0.1}})
    srv = Server([_resp([_msg(reply)], model="gpt-6.1-sol")])
    b = GenerativeLLMBackend(client(srv, model="gpt-6.1-sol"), reasoning_effort="low")
    r = b.decide("S", [Choice("which?", ["alpha", "beta"]), Noul("ok?")])
    assert r.answers[0].value == "beta" and r.answers[1].value is True and r.usage.n_calls == 1
    assert srv.requests[0]["reasoning"] == {"effort": "low"} and "temperature" not in srv.requests[0]


# ------------------------------------------------------------------ keys and registration
def test_missing_key_message_points_at_dotenv(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    srv = Server([])
    c = OpenAIResponsesClient("gpt-6-luna", transport=httpx.MockTransport(srv))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY") as ei:
        c.chat([{"role": "user", "content": "x"}])
    assert ".env" in str(ei.value) and srv.requests == []
    srv = Server([httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})])
    with pytest.raises(OpenAIResponsesError, match="OPENAI_API_KEY") as ei:
        client(srv).chat([{"role": "user", "content": "x"}])
    assert ei.value.status == 401 and ".env" in str(ei.value)
    monkeypatch.setenv("OPENAI_API_KEYS", "k1,k2")
    assert OpenAIResponsesClient("m", transport=httpx.MockTransport(srv)).keys == ["k1", "k2"]


def test_make_llm_registers_openai_responses_and_keeps_openai_on_chat_completions():
    c = make_llm("openai_responses:gpt-6-luna")
    assert isinstance(c, OpenAIResponsesClient) and c.model == "gpt-6-luna" and c.provider == "openai_responses"
    c = make_llm({"provider": "openai_responses", "model": "gpt-6-luna", "reasoning_effort": "none",
                  "server_state": True, "api_key": "k", "base_url": "https://proxy.example/v1/"})
    assert c.server_state and c.store and c.base_url == "https://proxy.example/v1"
    assert c.describe() == {"provider": "openai_responses", "model": "gpt-6-luna", "api": "responses", "store": True,
                            "server_state": True, "reasoning_effort": "none"}
    pytest.importorskip("openai")
    from jevharness.llm.openai_compat import OpenAICompatClient
    assert isinstance(make_llm({"provider": "openai", "model": "gpt-4.1-nano", "api_key": "k"}), OpenAICompatClient)
