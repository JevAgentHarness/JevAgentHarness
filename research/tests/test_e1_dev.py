"""Regression tests for the E1 (Level-1 static decisions) pipeline found in the dev shakedown
(configs/dev/e1_dev.yaml). Offline only: scripted LLMs and fake API clients."""

from __future__ import annotations

import json
from pathlib import Path as _P

ROOT = _P(__file__).resolve().parents[1]
from collections import Counter
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
import yaml

from jevharness.accounting import Ledger
from jevharness.bench import decision_eval as de
from jevharness.bench.suites import static
from jevharness.bench.suites.static import NONE_TOOL, W2C_OPTIONS, DecisionItem, norm
from jevharness.decision.llm_generative import GenerativeLLMBackend
from jevharness.decision.types import Choice
from jevharness.harness.tools import OTHER
from jevharness.llm.base import ChatResponse, ToolCall
from jevharness.llm.scripted import ScriptedLLM


def _tool(name: str, props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": f"{name} tool",
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required or []}}}


def _bfcl_row(rid: str, cat: str, n_params: int = 2) -> list[DecisionItem]:
    state = f"USER: request {rid}"
    q = Choice("Which function?", ["f", "g", NONE_TOOL], decision_point="tool_select")
    out = [DecisionItem(f"{rid}/tool", f"bfcl:{cat}", "tool_select", state, q, [norm("f")])]
    for i in range(n_params):
        qp = Choice(f"The function `f` will be called. Which value should its argument `p{i}` take?",
                    [f"v{i}", OTHER], decision_point="tool_params")
        out.append(DecisionItem(f"{rid}/param/p{i}", f"bfcl:{cat}", "tool_params", state, qp, [norm(f"v{i}")],
                                meta={"param": f"p{i}", "gold_tool": "f", "accept": [norm(f"v{i}")],
                                      "gold_in_candidates": True}))
    return out


# ------------------------------------------------------------------ sampling
def test_stratified_sample_keeps_rows_whole_proportional_and_prefix_consistent():
    items: list[DecisionItem] = []
    for cat, n in (("simple_python", 20), ("irrelevance", 20)):
        for k in range(n):
            items += _bfcl_row(f"{cat}_{k}", cat, n_params=0 if cat == "irrelevance" else 2)
    for k in range(80):
        q = Choice("How should the assistant respond?", list(W2C_OPTIONS.values()), decision_point="tool_select")
        items.append(DecisionItem(f"w2c/{k}", "when2call:BFCL v2 Live Multiple", "tool_select", "s", q, ["x"]))
    small, big = de.stratified_sample(items, 40), de.stratified_sample(items, 120)
    assert len(small) <= 40 and len(big) <= 120
    assert {i.id for i in small} <= {i.id for i in big}, "a smaller dev sample must be a subset of a larger one"
    assert de.stratified_sample(items, 40) == small  # deterministic
    # rows stay whole: a sampled BFCL row has its tool_select item and all its argument items
    for it in big:
        if it.source == "bfcl:simple_python":
            assert sum(1 for j in big if j.unit == it.unit) == 3
    # sources in proportion to the pool (rows: 20 / 20 / 80)
    rows = Counter(i.source for i in big if i.dp == "tool_select")
    assert set(rows) == {"bfcl:simple_python", "bfcl:irrelevance", "when2call:BFCL v2 Live Multiple"}
    assert rows["when2call:BFCL v2 Live Multiple"] > 2 * rows["bfcl:simple_python"]
    assert abs(rows["bfcl:simple_python"] - rows["bfcl:irrelevance"]) <= 1
    assert de.stratified_sample(items, None) == items


# ------------------------------------------------------------------ native FC baseline
def _fc_param_items(omit_ok: bool) -> list[DecisionItem]:
    tools = [_tool("f", {"p0": {"type": "string"}, "p1": {"type": "string", "default": "d"},
                         "p2": {"type": "string"}})]
    items = []
    for i in range(3):
        qp = Choice(f"Which value for p{i}?", [f"v{i}", OTHER], decision_point="tool_params")
        items.append(DecisionItem(f"r1/param/p{i}", "bfcl:simple_python", "tool_params", "USER: x", qp,
                                  [norm(f"v{i}")], tools,
                                  {"param": f"p{i}", "gold_tool": "f", "accept": [norm(f"v{i}")],
                                   "gold_in_candidates": True, "omit_ok": omit_ok and i == 1,
                                   "messages": [{"role": "user", "content": "x"}]}))
    return items


def test_fc_omitting_an_optional_argument_is_correct_when_bfcl_accepts_omission():
    """BFCL lists "" among an optional argument's possible answers when leaving it out is correct; the
    dev shakedown found 36 of 45 native-FC argument 'errors' in the P1 pilot were such omissions."""
    llm = ScriptedLLM(policy=lambda m, t: ChatResponse(text="", tool_calls=[
        ToolCall(id="c", name="f", arguments={"p0": "v0"})]))      # p1 (omit_ok) and p2 (required) omitted
    recs = de._fc_records(llm, _fc_param_items(omit_ok=True), workers=1, name="fc")
    by = {r["id"].rsplit("/", 1)[1]: r for r in recs}
    assert by["p0"]["correct"] and by["p1"]["correct"] and not by["p2"]["correct"]
    assert by["p1"]["pred"] == "" and by["p2"]["pred"] == ""
    # re-scoring from the records (resummarize) agrees, and the omission rate is reported
    row = de.summarize(recs)["fc|tool_params|bfcl"]
    assert row["acc"] == pytest.approx(2 / 3) and row["omit_rate"] == pytest.approx(2 / 3)
    # without omit_ok the omission stays wrong
    recs2 = de._fc_records(llm, _fc_param_items(omit_ok=False), workers=1, name="fc")
    assert [r["correct"] for r in sorted(recs2, key=lambda r: r["id"])] == [True, False, False]


def test_fc_arguments_one_forced_call_per_row_with_cost_split_over_its_arguments():
    llm = ScriptedLLM(policy=lambda m, t: ChatResponse(
        text="", tool_calls=[ToolCall(id="c", name="f", arguments={"p0": "v0", "p1": "v1", "p2": "v2"})],
        input_tokens=3000, output_tokens=30))
    recs = de._fc_records(llm, _fc_param_items(omit_ok=False), workers=2, name="fc")
    assert llm.calls == 1, "one forced call yields every argument of the row"
    assert all(r["correct"] for r in recs) and all(r["request_questions"] == 3 for r in recs)
    assert all(r["input_tokens"] == pytest.approx(1000) for r in recs)
    assert len({r["cost_usd"] for r in recs}) == 1


def test_fc_gets_native_chat_messages_not_the_rendered_state():
    seen: list[list[dict[str, Any]]] = []

    def policy(messages, tools):
        seen.append(messages)
        return ChatResponse(text="no tool needed")

    msgs = [{"role": "system", "content": "You are a bot."}, {"role": "user", "content": "hi"}]
    q = Choice("Which function?", ["f", NONE_TOOL], decision_point="tool_select")
    it = DecisionItem("r/tool", "bfcl:irrelevance", "tool_select", "SYSTEM: You are a bot.\nUSER: hi", q,
                      [norm(NONE_TOOL)], [_tool("f", {})], {"messages": msgs, "name_map": {"f": "f"}})
    (rec,) = de._fc_records(ScriptedLLM(policy=policy), [it], workers=1, name="fc")
    assert seen == [msgs] and rec["correct"] and rec["pred"] == norm(NONE_TOOL)
    assert "messages" not in rec["meta"] and "name_map" not in rec["meta"]


def _w2c_item(label: str, tools: list[dict[str, Any]]) -> DecisionItem:
    q = Choice("How should the assistant respond?", list(W2C_OPTIONS.values()), decision_point="tool_select")
    return DecisionItem(f"w2c/{label}", "when2call:BFCL v2 Live Simple", "tool_select", "AVAILABLE TOOLS: ...", q,
                        [norm(W2C_OPTIONS[label])], tools, {"label": label, "user_message": "Book me a ride"})


def test_fc_when2call_text_replies_are_classified_by_the_judge_and_judge_cost_kept_apart():
    tools = [_tool("ride", {"loc": {"type": "string"}}, ["loc"])]
    fc = ScriptedLLM(policy=lambda m, t: ChatResponse(text="Where should I pick you up?"))
    judge = ScriptedLLM(model="gemini-3.5-flash-lite",
                        policy=lambda m, t: ChatResponse(text='{"label": "request_for_info"}'))
    (rec,) = de._fc_records(fc, [_w2c_item("request_for_info", tools)], workers=1, name="fc", judge=judge)
    assert rec["correct"] and rec["fc_label"] == "request_for_info" and rec["judge_cost_usd"] > 0
    assert rec["cost_usd"] == pytest.approx(de._fc_cost(SimpleNamespace(
        model="scripted", input_tokens=rec["input_tokens"], output_tokens=rec["output_tokens"],
        cached_input_tokens=0), fc)[0]), "judge cost must not be booked as decision cost"
    # a tool call is "call a tool" without asking the judge
    caller = ScriptedLLM()  # default policy: calls the first tool
    (rec2,) = de._fc_records(caller, [_w2c_item("request_for_info", tools)], workers=1, name="fc", judge=judge)
    assert rec2["fc_label"] == "tool_call" and not rec2["correct"] and judge.calls == 1
    # no judge configured -> skipped (not scored, not an error)
    (rec3,) = de._fc_records(fc, [_w2c_item("cannot_answer", tools)], workers=1, name="fc")
    assert "skipped" in rec3 and "correct" not in rec3


# ------------------------------------------------------------------ static suites
def test_when2call_state_shows_arguments_and_whether_they_are_required(tmp_path, monkeypatch):
    """request_for_info items are built by removing a REQUIRED argument's value from the question; a
    decider that sees only tool names and descriptions cannot tell them from tool_call items."""
    row = {"uuid": "u1", "source": "BFCL v2 Live Simple", "question": "I need an Uber Plus ride",
           "correct_answer": "request_for_info", "answers": {},
           "tools": [json.dumps({"name": "uber.ride", "description": "Find a ride", "parameters": {
               "type": "dict", "required": ["loc", "type"],
               "properties": {"loc": {"type": "string", "description": "Start address"},
                              "type": {"type": "string", "enum": ["plus", "comfort"]},
                              "time": {"type": "integer"}}}})]}
    p = tmp_path / "when2call" / "test" / "when2call_test_mcq.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(static, "_download", lambda url, dest: dest)
    (it,) = static.load_when2call(data_dir=tmp_path)
    assert "loc (string, required): Start address" in it.state
    assert "type (string one of plus|comfort, required)" in it.state and "time (integer, optional)" in it.state
    assert it.meta["user_message"] == "I need an Uber Plus ride" and it.gold == [norm(W2C_OPTIONS["request_for_info"])]
    assert it.tools[0]["function"]["parameters"]["type"] == "object"  # FC sees the full schema


def test_bfcl_argument_items_record_whether_omission_is_accepted(tmp_path, monkeypatch):
    row = {"id": "simple_python_1", "question": [[{"role": "user", "content": "Weather in Paris in celsius?"}]],
           "function": [{"name": "get_weather", "description": "d", "parameters": {"type": "dict", "properties": {
               "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}]}
    ans = {"id": "simple_python_1", "ground_truth": [{"get_weather": {"city": ["Paris"], "unit": ["", "celsius"]}}]}
    d = tmp_path / "bfcl"
    (d / "possible_answer").mkdir(parents=True)
    (d / "BFCL_v4_simple_python.json").write_text(json.dumps(row) + "\n")
    (d / "possible_answer" / "BFCL_v4_simple_python.json").write_text(json.dumps(ans) + "\n")
    monkeypatch.setattr(static, "_download", lambda url, dest: dest)
    items = {i.id: i for i in static.load_bfcl(categories=["simple_python"], data_dir=tmp_path)}
    assert items["simple_python_1/param/city"].meta["omit_ok"] is False
    unit = items["simple_python_1/param/unit"]
    assert unit.meta["omit_ok"] is True and unit.meta["accept"] == ["celsius"]
    assert unit.meta["messages"] == [{"role": "user", "content": "Weather in Paris in celsius?"}]
    assert {i.unit for i in items.values()} == {"simple_python_1"}


# ------------------------------------------------------------------ leakage
def test_question_meta_never_reaches_a_generative_decider():
    seen: list[str] = []

    def policy(messages, tools):
        seen.append(json.dumps(messages))
        return ChatResponse(text='{"Q1": {"A": 0.9, "B": 0.1}}')

    q = Choice("Which function?", ["f", "g"], decision_point="tool_select", secret="SENTINEL-GOLD-42")
    q.meta["gold"] = "SENTINEL-GOLD-43"
    a = GenerativeLLMBackend(ScriptedLLM(policy=policy)).decide("USER: x", [q]).answers[0]
    assert a.value == "f" and seen and not any("SENTINEL" in s for s in seen)


# ------------------------------------------------------------------ summary / accounting
def test_summary_reports_coverage_parse_failures_and_per_category_errors():
    ok = {"source": "bfcl:multiple", "dp": "tool_select", "backend": "b", "kind": "s1", "pred": "f",
          "gold": ["f"], "correct": True, "conf": 0.9, "latency_ms": 5.0}
    pf = {**ok, "pred": "g", "correct": False, "conf": 0.5, "parse_fail": True}
    err = {"source": "bfcl:multiple", "dp": "tool_select", "backend": "b", "kind": "s1", "error": "429"}
    s = de.summarize([ok, pf, err])
    row = s["b|tool_select|bfcl"]
    assert row["n"] == 2 and row["n_errors"] == 1 and row["coverage"] == pytest.approx(2 / 3)
    assert row["n_parse_fail"] == 1
    assert s["b|tool_select|bfcl:multiple"]["n_errors"] == 1  # per-category rows see their errors too


def test_generative_parse_failure_keeps_the_reply_for_inspection():
    b = GenerativeLLMBackend(ScriptedLLM(policy=lambda m, t: ChatResponse(text="I think it is f", finish_reason="stop")))
    q = Choice("Which function?", ["f", "g"], decision_point="tool_select")
    (rec,) = de._s1_records(b, [DecisionItem("1/tool", "bfcl:multiple", "tool_select", "USER: x", q, ["f"])],
                            batch_by_state=True, workers=1, name="gen")
    assert rec["parse_fail"] and rec["raw_reply"] == "I think it is f" and rec["conf"] == pytest.approx(0.5)


def test_run_decision_eval_eval_options_ledger_and_only(tmp_path, monkeypatch):
    items = [it for k in range(6) for it in _bfcl_row(f"multiple_{k}", "multiple", n_params=1)]
    for it in items:  # simulated backends read gold from Question.meta (pipeline test only)
        it.question.meta["gold"] = it.question.options[0]
    monkeypatch.setattr(de, "load_static", lambda name, **kw: list(items))
    exp = {"name": "t", "suites": [{"name": "bfcl"}], "workers": 2,
           "backends": {"sim": {"type": "simulated", "accuracy": 1.0},
                        "sub": {"type": "simulated", "accuracy": 1.0, "eval": {"max_items": 4, "workers": 1}}},
           "fc_llms": {"fc": {"provider": "scripted", "model": "scripted"}}}
    summ = de.run_decision_eval(exp, tmp_path)
    out = tmp_path / "t"
    sub_ids = {json.loads(line)["id"] for line in (out / "s1_sub.jsonl").read_text().splitlines()}
    all_ids = {json.loads(line)["id"] for line in (out / "s1_sim.jsonl").read_text().splitlines()}
    assert len(sub_ids) == 4 and sub_ids < all_ids  # eval.max_items = stratified prefix, never passed to make_backend
    led = json.loads((out / "ledger.json").read_text())
    assert set(led["backends"]) == {"sim", "sub", "fc", "fc.judge"} and led["fresh_cost_usd"] > 0
    assert summ["sim|tool_select|bfcl"]["n"] == 6 and summ["sub|tool_select|bfcl"]["n"] == 2
    # --only re-runs one backend and keeps the others' records; stale files of removed backends are ignored
    (out / "s1_removed.jsonl").write_text(json.dumps({**json.loads((out / "s1_sim.jsonl").read_text().splitlines()[0]),
                                                       "backend": "removed"}) + "\n")
    summ2 = de.run_decision_eval(exp, tmp_path, only=["sub"])
    assert "sim|tool_select|bfcl" in summ2 and not any(k.startswith("removed|") for k in summ2)
    assert set(json.loads((out / "ledger.json").read_text())["backends"]) == {"sim", "sub", "fc", "fc.judge"}
    assert yaml.safe_load((out / "experiment.yaml").read_text())["name"] == "t"


def test_embedding_latency_excludes_rate_limit_retries(monkeypatch):
    """The embedding baseline used the OpenAI SDK's built-in retries inside its timer, so a 429 + backoff
    was counted as decision latency (and keys did not rotate)."""
    from jevharness.decision import embedding as emb

    clock = {"t": 0.0}
    fake_time = SimpleNamespace(perf_counter=lambda: clock["t"],
                                sleep=lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(emb, "time", fake_time)
    calls = {"n": 0}

    def create(model, input):
        calls["n"] += 1
        if calls["n"] == 1:
            clock["t"] += 0.5  # failed attempt
            req = httpx.Request("POST", "https://example.invalid/embeddings")
            raise openai.RateLimitError("429", response=httpx.Response(429, request=req), body=None)
        clock["t"] += 0.02  # successful attempt: 20 ms
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, float(i)]) for i in range(len(input))],
                               usage=SimpleNamespace(total_tokens=10))

    b = emb.EmbeddingBackend(model="gemini-embedding-001", provider="gemini", api_key="test-key")
    b.clients = [SimpleNamespace(embeddings=SimpleNamespace(create=create))]
    res = b.decide("USER: weather", [Choice("Which function?", ["f", "g"], decision_point="tool_select")])
    assert calls["n"] == 2 and b.n_retries == 1
    assert res.usage.latency_ms == pytest.approx(20.0, abs=1e-6), "retry + backoff must not count as latency"
    assert res.usage.input_tokens == 10 and res.usage.cost_usd > 0


# ------------------------------------------------------------------ lexical controls
def test_lexical_abstain_option_is_not_scored_on_question_boilerplate():
    """v1 BM25 / TF-IDF scored NONE_TOOL lexically, and the constant BFCL question ("Which function should be
    called to handle the user's request?") shares "function" / "request" with NONE_TOOL's text: they answered
    "none" on 45% of the dev rows that need a tool. Abstain options now score 0 (no lexical evidence)."""
    from jevharness.decision.lexical import VERSION, BM25Backend, TfidfBackend

    q = Choice("Which function should be called to handle the user's request?", ["math_gcd", NONE_TOOL],
               ["Calculates the greatest common divisor of two numbers.", "Use when no function applies."],
               decision_point="tool_select")
    state = "USER: What is the GCD of 12 and 18?"
    for cls in (BM25Backend, TfidfBackend):
        assert cls(abstain_score="lexical").ask(state, q).value == NONE_TOOL  # the v1 failure, kept for ablation
        a = cls().ask(state, q)
        assert a.value == "math_gcd" and a.raw["scores"][1] == 0.0
        assert cls(abstain_score=50.0).ask(state, q).value == NONE_TOOL  # an explicit threshold still abstains
        assert cls().describe()["version"] == VERSION >= 2  # new cache keys: no v1 answers replayed
    # nothing matches: the real option and "none" tie at 0 -> low confidence, the real (first) option
    nothing = BM25Backend().ask("USER: How many calories are in an avocado?", q)
    assert nothing.value == "math_gcd" and nothing.confidence == pytest.approx(0.5, abs=1e-3)
    with pytest.raises(ValueError):
        BM25Backend(abstain_score="sometimes")


# ------------------------------------------------------------------ embedding
def _fake_embedding_backend(monkeypatch, seen: list[list[str]]):
    from jevharness.decision import embedding as emb

    def create(model, input):
        seen.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, float(len(t) % 7), float(len(t) % 3)])
                                     for t in input], usage=SimpleNamespace(total_tokens=10 * len(input)))

    b = emb.EmbeddingBackend(model="gemini-embedding-001", provider="gemini", api_key="test-key")
    b.clients = [SimpleNamespace(embeddings=SimpleNamespace(create=create))]
    return b


def test_embedding_sends_all_questions_of_a_request_in_one_embeddings_call(monkeypatch):
    """The per-question loop sent one embeddings request per question in sequence, so a 9-argument BFCL row
    paid ~9 round trips of latency while every other batched decider pays one."""
    seen: list[list[str]] = []
    b = _fake_embedding_backend(monkeypatch, seen)
    qs = [Choice(f"Which value should argument p{i} take?", ["a", "b", OTHER], decision_point="tool_params")
          for i in range(3)]
    res = b.decide("USER: set a and b", qs)
    assert len(seen) == 1 and res.usage.n_calls == 1
    assert len(seen[0]) == 3 + 3  # three queries + the shared options, deduplicated
    assert len(res.answers) == 3 and res.usage.input_tokens == 60
    # options are memoised: a second state only embeds its queries
    res2 = b.decide("USER: other state", qs)
    assert len(seen) == 2 and len(seen[1]) == 3 and res2.usage.n_calls == 1
    assert b.describe()["batch"] is True


# ------------------------------------------------------------------ leakage
SENTINEL = "SENTINEL-GOLD-7731"


def test_no_e1_backend_sees_gold_and_answers_do_not_depend_on_it(monkeypatch):
    """Question.meta (gold) and DecisionItem.meta["accept"] are never on the wire: requests contain no
    sentinel and every backend's answer is identical whichever option is marked gold."""
    from jevharness.decision.lexical import BM25Backend, TfidfBackend
    from jevharness.decision.self_llm import SelfLLMBackend

    sent: list[str] = []

    def policy(messages, tools):
        sent.append(json.dumps(messages) + json.dumps(tools or []))
        return ChatResponse(text='{"Q1": {"A": 0.7, "B": 0.2, "C": 0.1}}')

    emb_seen: list[list[str]] = []
    backends = {"bm25": BM25Backend(), "tfidf": TfidfBackend(),
                "embedding": _fake_embedding_backend(monkeypatch, emb_seen),
                "llm_generative": GenerativeLLMBackend(ScriptedLLM(policy=policy)),
                "s2_self": SelfLLMBackend(ScriptedLLM(policy=policy))}
    state = "USER: What's the weather in Paris?"
    for name, b in backends.items():
        dists = []
        for gold in ("get_weather", "translate", SENTINEL):
            q = Choice("Which function should be called to handle the user's request?",
                       ["get_weather", "translate", NONE_TOOL], ["Current weather.", "Translate text.", "None."],
                       decision_point="tool_select")
            q.meta.update({"gold": gold, "secret": SENTINEL})
            dists.append(b.decide(state, [q]).answers[0].distribution)
        assert dists[0] == dists[1] == dists[2], name
    assert sent and not any(SENTINEL in s for s in sent)
    assert emb_seen and not any(SENTINEL in t for batch in emb_seen for t in batch)
    # native FC: messages + tool schemas only, never Question.meta or the accepted values
    fc_sent: list[str] = []

    def fc_policy(messages, tools):
        fc_sent.append(json.dumps(messages) + json.dumps(tools or []))
        return ChatResponse(text="", tool_calls=[ToolCall(id="c", name="f", arguments={"p0": "v0"})])

    items = _fc_param_items(omit_ok=False)
    for it in items:
        it.question.meta["gold"] = SENTINEL
        it.meta["accept"] = [SENTINEL]
    de._fc_records(ScriptedLLM(policy=fc_policy), items, workers=1, name="fc")
    assert fc_sent and not any(SENTINEL in s for s in fc_sent)


# ------------------------------------------------------------------ generative parsing
def test_generative_salvages_well_formed_answers_from_a_malformed_batched_reply():
    reply = ('{"Q1": {"A": 0.01, "B": 0.01, "C": 0.98}, "Q2": {"A": 0.1, "B": 0.9), '
             '"Q3": {"A": 0.01, "B": 0.01, "C": 0.98}}')  # verbatim gemini-3.5-flash reply (E1 dev)
    b = GenerativeLLMBackend(ScriptedLLM(policy=lambda m, t: ChatResponse(text=reply)))
    qs = [Choice(f"Which value for p{i}?", ["x", "y", "z"] if i != 1 else ["x", "y"], decision_point="tool_params")
          for i in range(3)]
    items = [DecisionItem(f"r/param/p{i}", "bfcl:live_simple", "tool_params", "USER: s", q, [norm("z")])
             for i, q in enumerate(qs)]
    recs = {r["id"]: r for r in de._s1_records(b, items, batch_by_state=True, workers=1, name="gen")}
    assert recs["r/param/p0"]["pred"] == "z" and recs["r/param/p0"]["salvaged"]
    assert recs["r/param/p2"]["pred"] == "z" and recs["r/param/p2"]["salvaged"]
    assert recs["r/param/p1"]["parse_fail"] and "salvaged" not in recs["r/param/p1"]
    # a well-formed reply is untouched
    ok = GenerativeLLMBackend(ScriptedLLM(policy=lambda m, t: ChatResponse(text='{"Q1": {"A": 0.2, "B": 0.8}}')))
    a = ok.decide("s", [Choice("?", ["x", "y"])]).answers[0]
    assert a.value == "y" and "salvaged" not in a.raw


# ------------------------------------------------------------------ FC response cache (resume)
def test_fc_cache_replays_responses_books_them_cached_and_keys_on_the_request(tmp_path):
    llm = ScriptedLLM(policy=lambda m, t: ChatResponse(
        text="", tool_calls=[ToolCall(id="c", name="f", arguments={"p0": "v0", "p1": "v1", "p2": "v2"},
                                      extra={"google": {"thought_signature": "sig"}})],
        input_tokens=3000, output_tokens=30, latency_ms=850.0))
    spec = {"provider": "scripted", "model": "gemini-3.5-flash", "temperature": 0}
    path = tmp_path / "fc.sqlite"
    led1, led2 = Ledger(), Ledger()
    r1 = de._fc_records(de.CachedLLM(llm, spec, path), _fc_param_items(False), workers=2, name="fc", ledger=led1)
    r2 = de._fc_records(de.CachedLLM(llm, spec, path), _fc_param_items(False), workers=2, name="fc", ledger=led2)
    assert llm.calls == 1, "the second run is served from the cache"
    assert not any(r["cached"] for r in r1) and all(r["cached"] for r in r2)
    strip = [{k: v for k, v in r.items() if k != "cached"} for r in sorted(r1, key=lambda r: r["id"])]
    assert strip == [{k: v for k, v in r.items() if k != "cached"} for r in sorted(r2, key=lambda r: r["id"])]
    assert r2[0]["latency_ms"] == 850.0 and r2[0]["cost_usd"] > 0  # original efficiency replayed
    assert de._ledger_entry(led1)["fresh_cost_usd"] > 0 and de._ledger_entry(led2)["fresh_cost_usd"] == 0
    assert all(rec.cached for rec in led2.records)
    # a different model / sampling spec is a different request
    de._fc_records(de.CachedLLM(llm, {**spec, "temperature": 0.7}, path), _fc_param_items(False), workers=1,
                   name="fc")
    assert llm.calls == 2
    # tool-call extras (Gemini thought signatures) survive the round trip
    cached = de.CachedLLM(llm, spec, path).chat([{"role": "user", "content": "x"}], tools=[_tool("f", {})])
    again = de.CachedLLM(llm, spec, path).chat([{"role": "user", "content": "x"}], tools=[_tool("f", {})])
    assert again.tool_calls[0].extra == cached.tool_calls[0].extra == {"google": {"thought_signature": "sig"}}


def test_run_decision_eval_fc_cache_defaults_next_to_s1_cache(tmp_path, monkeypatch):
    items = [it for k in range(3) for it in _bfcl_row(f"multiple_{k}", "multiple", n_params=1)]
    for it in items:
        it.question.meta["gold"] = it.question.options[0]
        it.tools = [_tool("f", {"p0": {"type": "string"}})]
        it.meta.setdefault("name_map", {"f": "f"})
    monkeypatch.setattr(de, "load_static", lambda name, **kw: list(items))
    exp = {"name": "t", "suites": [{"name": "bfcl"}], "workers": 1, "s1_cache": str(tmp_path / "c" / "s1.sqlite"),
           "backends": {"sim": {"type": "simulated", "accuracy": 1.0}},
           "fc_llms": {"fc": {"provider": "scripted", "model": "scripted"}}}
    de.run_decision_eval(exp, tmp_path)
    assert (tmp_path / "c" / "fc.sqlite").exists()
    de.run_decision_eval(exp, tmp_path)
    led = json.loads((tmp_path / "t" / "ledger.json").read_text())
    assert led["backends"]["fc"]["fresh_cost_usd"] == 0 and led["backends"]["fc"]["cached_calls"] > 0
    assert all(json.loads(line)["cached"] for line in (tmp_path / "t" / "fc_fc.jsonl").read_text().splitlines())
    assert de._fc_cache_path({**exp, "fc_cache": None}) is None and de._fc_cache_path({}) is None


# ------------------------------------------------------------------ reliability figure
def test_reliability_diagram_keeps_sparse_bins_and_reports_counts(tmp_path):
    """Verbalised confidences pile up near 1 (95-99% above 0.9 in the E1 dev run): the old diagram silently
    dropped every bin with < 5 decisions, so a 3-backend figure showed one or two points per backend."""
    from jevharness.bench import plots

    recs = ([{"dp": "tool_select", "conf": 0.97, "correct": True}] * 40 + [{"dp": "tool_select", "conf": 0.95,
                                                                            "correct": False}] * 4
            + [{"dp": "tool_select", "conf": 0.55, "correct": False}] * 2
            + [{"dp": "tool_params", "conf": 0.0, "correct": False}])
    (tmp_path / "s1_a.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    table = plots.reliability(tmp_path, ["a"], tmp_path / "rel.pdf")
    assert (tmp_path / "rel.pdf").stat().st_size > 0
    bins = {round(b["lo"], 1): b for b in table["a"]}
    assert bins[0.9]["n"] == 44 and bins[0.9]["acc"] == pytest.approx(40 / 44)
    assert bins[0.5]["n"] == 2  # sparse bin kept (drawn hollow)
    assert bins[0.0]["n"] == 1  # confidence 0 lands in the first bin instead of vanishing
    assert sum(b["n"] for b in plots.reliability_bins(tmp_path, "a", dp="tool_select")) == 46


def test_run_decision_eval_excludes_the_listed_ids(tmp_path, monkeypatch):
    """A6.13: the exposed E1 dev-prefix items (configs/eval/e1_dev_prefix_ids.json) leave the C8 population."""
    items = [it for k in range(6) for it in _bfcl_row(f"multiple_{k}", "multiple", n_params=1)]
    for it in items:
        it.question.meta["gold"] = it.question.options[0]
    monkeypatch.setattr(de, "load_static", lambda name, **kw: list(items))
    drop = sorted({it.id for it in items})[:2]
    f = tmp_path / "excl.json"
    f.write_text(json.dumps({"ids": drop}))
    exp = {"name": "x", "suites": [{"name": "bfcl"}], "workers": 1, "exclude_ids_file": str(f),
           "backends": {"sim": {"type": "simulated", "accuracy": 1.0}}}
    de.run_decision_eval(exp, tmp_path)
    recs = [json.loads(l) for l in (tmp_path / "x" / "s1_sim.jsonl").read_text().splitlines()]
    assert recs and not {r["id"] for r in recs} & set(drop) and len(recs) == len(items) - 2
    real = json.loads((ROOT / "configs/eval/e1_dev_prefix_ids.json").read_text())
    assert real["n"] == 299 == len(set(real["ids"]))
