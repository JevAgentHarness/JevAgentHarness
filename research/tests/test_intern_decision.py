"""Offline tests for the Intern-Decision System-1 backend (decision/intern_decision.py) and the text-only loader.

Golden strings below were produced by the checkpoint's own ``inference.py`` (internlm/Intern-Decision-0.8B, rev
85a0cc5) + its tokenizer's chat template on the model card's example request (2026-10-01). The chat template is
vendored in tests/fixtures/qwen3_5_chat_template.jinja (Qwen3.5, Apache-2.0) so the rendering is tested without
downloading the checkpoint; the tiny Qwen3.5 checkpoint built here is random, so only mechanics are asserted."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from jevharness.accounting import Ledger, default_prices
from jevharness.decision import Choice, Noul, QKind, Score, make_backend
from jevharness.decision.intern_decision import (ANSWER_SYMBOLS, DECISION_TOKEN, KNOWN_TEMPERATURES,
                                                 InternDecisionBackend, answer_from_intern, argmax,
                                                 checkpoint_temperature, compile_request, compile_row, field_names,
                                                 question_to_intern, read_fields, scale_probabilities)

FIXTURES = Path(__file__).parent / "fixtures"

STATE = "The customer was charged twice and asks for the extra payment back."
REQUEST = {"state": STATE, "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "Payments and refunds", "delivery": "Shipping and delivery"}},
    "urgency": {"type": "score", "instructions": "Rate the priority.", "criteria": ["Low", "Medium", "High"]},
    "refund_requested": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}}
GOLDEN_SYSTEM = ("You are a careful decision assistant. Use the state and decision schema in the user message to make "
                 "the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from "
                 "its listed options and return one valid JSON object mapping each field name to its chosen symbol. "
                 "Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.")
GOLDEN_USER = ('Return one answer for every field using the supplied answer symbols.\n\n## State\n'
               '"The customer was charged twice and asks for the extra payment back."\n## Decision schema\n'
               'team: Which team should handle this request?\n    A = billing: Payments and refunds\n'
               '    B = delivery: Shipping and delivery\nurgency: Rate the priority.\n    A = 0: Low\n    B = 1: Medium\n'
               '    C = 2: High\nrefund_requested: Is the customer asking for a refund?\n'
               '    A = no: The answer is no (negative, or disagree with the claim).\n'
               '    B = yes: The answer is yes (affirmative, or align with the claim).')
GOLDEN_SKELETON = '{\n    "team": "<decision>",\n    "urgency": "<decision>",\n    "refund_requested": "<decision>"\n}'
GOLDEN_TEXT = (f"<|im_start|>system\n{GOLDEN_SYSTEM}<|im_end|>\n<|im_start|>user\n{GOLDEN_USER}<|im_end|>\n"
               f"<|im_start|>assistant\n<think>\n\n</think>\n\n{GOLDEN_SKELETON}<|im_end|>\n")


def our_questions():
    return [Choice("Which team should handle this request?", ["billing", "delivery"],
                   ["Payments and refunds", "Shipping and delivery"], decision_point="team", gold="billing"),
            Score("Rate the priority.", levels=["Low", "Medium", "High"], scale=(0, 2), decision_point="urgency"),
            Noul("Is the customer asking for a refund?", decision_point="refund_requested")]


# ------------------------------------------------------------------ pure prompt / probability logic
def test_vendored_compiler_reproduces_reference_golden():
    c = compile_row(REQUEST)
    assert c.fields == ("team", "urgency", "refund_requested")
    assert c.symbols == {"team": ("A", "B"), "urgency": ("A", "B", "C"), "refund_requested": ("A", "B")}
    assert c.messages == [{"role": "system", "content": GOLDEN_SYSTEM}, {"role": "user", "content": GOLDEN_USER},
                          {"role": "assistant", "content": GOLDEN_SKELETON}]
    # our typed questions compile to the identical prompt (field names from the decision points)
    ours, qs = compile_request(STATE, our_questions())
    assert ours == c and list(qs) == list(c.fields)
    assert qs["urgency"]["criteria"] == {"0": "Low", "1": "Medium", "2": "High"}  # object form, our scale values
    # a 1..3 scale keeps its own values as the option values, so the expected score is on our scale
    one, _ = compile_request("s", [Score("q", scale=(1, 3), levels=["bad", "ok", "good"], decision_point="s")])
    assert "s: q\n    A = 1: bad\n    B = 2: ok\n    C = 3: good" in one.messages[1]["content"]
    # dict states are JSON-rendered; raw_state embeds a string state verbatim
    assert '## State\n{\n  "k": "v"\n}\n' in compile_row({"state": {"k": "v"}, "questions": REQUEST["questions"]}).messages[1]["content"]
    assert "## State\nline1\nline2\n## Decision" in compile_request("line1\nline2", our_questions(), raw_state=True)[0].messages[1]["content"]
    with pytest.raises(ValueError):
        compile_row({"state": f"x {DECISION_TOKEN} y", "questions": REQUEST["questions"]})


def test_question_mapping_and_field_names():
    w = question_to_intern(Choice("pick", ["a", "b"]))
    assert w == {"type": "choice", "instructions": "pick", "criteria": {"a": "", "b": ""}}
    n = question_to_intern(Noul("ok?", criteria={"true": "T", "false": "F"}))
    assert n["criteria"] == {"true": "T", "false": "F"}
    s = question_to_intern(Score("rate", rubric="R", scale=(1, 3)))
    assert s["instructions"] == "rate\nRubric: R" and list(s["criteria"]) == ["1", "2", "3"]
    qs = [Choice("a", ["x", "y"], wire_id="tool"), Choice("b", ["x", "y"], decision_point="tool_params"),
          Choice("c", ["x", "y"], decision_point="tool_params"), Noul("d")]
    assert field_names(qs) == ["tool", "tool_params", "tool_params_2", "q3"]


def test_scale_probabilities_matches_reference_and_keeps_argmax():
    p = scale_probabilities({"a": 0.7, "b": 0.2, "c": 0.1}, 2.747760550703)  # reference output for the 0.8B T
    assert p == pytest.approx({"a": 0.47027768887576965, "b": 0.2980919213435492, "c": 0.23163038978068112})
    assert scale_probabilities({"a": 0.6, "b": 0.4}, 1.0) == {"a": 0.6, "b": 0.4}
    assert argmax({"b": 0.5, "a": 0.5}) == "a"  # lexical tie-break
    with pytest.raises(ValueError):
        scale_probabilities({"a": 0.5, "b": 0.4}, 2.0)
    with pytest.raises(ValueError):
        scale_probabilities({"a": 1.0}, 0.0)


def test_read_fields_score_expected_value_and_noul_p_yes():
    torch = pytest.importorskip("torch")
    compiled, qs = compile_request(STATE, our_questions())
    symbol_ids = {s: 10 + i for i, s in enumerate(ANSWER_SYMBOLS)}
    logits = torch.full((3, 100), -30.0)
    logits[0, symbol_ids["A"]], logits[0, symbol_ids["B"]] = math.log(0.8), math.log(0.2)      # billing 0.8
    logits[1, symbol_ids["A"]], logits[1, symbol_ids["B"]], logits[1, symbol_ids["C"]] = math.log(0.1), math.log(0.2), math.log(0.7)
    logits[2, symbol_ids["A"]], logits[2, symbol_ids["B"]] = math.log(0.3), math.log(0.7)      # no 0.3 / yes 0.7
    out = read_fields(logits, compiled, qs, symbol_ids, 1.0)
    assert out["team"]["choice"] == "billing" and out["team"]["confidence"] == pytest.approx(0.8, abs=1e-6)
    assert out["urgency"]["score"] == pytest.approx(0.1 * 0 + 0.2 * 1 + 0.7 * 2, abs=1e-6)
    assert out["urgency"]["legend"] == {"0": "Low", "1": "Medium", "2": "High"}
    assert out["refund_requested"]["noul"] == pytest.approx(0.7, abs=1e-6)
    assert out["refund_requested"]["probabilities"] == pytest.approx({"no": 0.3, "yes": 0.7}, abs=1e-6)
    # calibration flattens but never flips
    cal = read_fields(logits, compiled, qs, symbol_ids, 2.0)
    assert cal["team"]["choice"] == "billing" and 0.5 < cal["team"]["confidence"] < 0.8
    assert cal["urgency"]["decision"] == "2" and cal["urgency"]["score"] < out["urgency"]["score"]
    # -> our Answer objects
    team, urg, ref = (answer_from_intern(q, out[f]) for q, f in zip(our_questions(), compiled.fields))
    assert team.kind == QKind.CHOICE and team.value == "billing" and team.confidence == pytest.approx(0.8, abs=1e-6)
    assert urg.kind == QKind.SCORE and urg.value == pytest.approx(1.6, abs=1e-6) and set(urg.distribution) == {"0", "1", "2"}
    assert ref.kind == QKind.NOUL and ref.value is True and ref.p_yes == pytest.approx(0.7, abs=1e-5)
    assert team.raw["probabilities"] == out["team"]["probabilities"] and "legend" not in urg.raw


def test_temperature_lookup_prefers_checkpoint_file(tmp_path):
    assert KNOWN_TEMPERATURES["internlm/Intern-Decision-0.8B"] == 2.747760550703
    assert checkpoint_temperature("/nonexistent/Intern-Decision-2B") == (2.100509348278, "vendored table")
    (tmp_path / "inference.py").write_text("MODEL_NAME = 'x'\nDEFAULT_TEMPERATURE = 1.75\n")
    assert checkpoint_temperature(str(tmp_path)) == (1.75, "checkpoint inference.py")
    assert checkpoint_temperature(str(tmp_path / "missing-dir")) == (None, None)


def test_price_entries_present():
    for size in ("0.8B", "2B", "4B"):
        p = default_prices().get(f"internlm/Intern-Decision-{size}")
        assert p is not None and p.input_per_m > 0 and p.output_per_m == 0.0


# ------------------------------------------------------------------ tiny Qwen3.5-style checkpoint
@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=420, special_tokens=["<unk>", "<|endoftext|>", "<|im_end|>"],
                                             initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([GOLDEN_SYSTEM, GOLDEN_USER, GOLDEN_SKELETON] * 5, trainer)
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<|endoftext|>",
                                 eos_token="<|im_end|>")
    specials = ["<|im_start|>", "<think>", "</think>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
                "<|video_pad|>", DECISION_TOKEN]
    ft.add_special_tokens({"additional_special_tokens": specials})
    ft.chat_template = (FIXTURES / "qwen3_5_chat_template.jinja").read_text()
    tid = ft.convert_tokens_to_ids
    text = dict(vocab_size=len(ft), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, layer_types=["linear_attention"] * 3 + ["full_attention"],
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_conv_kernel_dim=4, tie_word_embeddings=False, mtp_num_hidden_layers=0,
                eos_token_id=tid("<|im_end|>"))
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32, patch_size=4,
                  spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=16, deepstack_visual_indexes=[])
    cfg = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=tid("<|image_pad|>"),
                        video_token_id=tid("<|video_pad|>"), vision_start_token_id=tid("<|vision_start|>"),
                        vision_end_token_id=tid("<|vision_end|>"), tie_word_embeddings=False)
    d = tmp_path_factory.mktemp("intern_tiny")
    Qwen3_5ForConditionalGeneration(cfg).save_pretrained(d)
    ft.save_pretrained(d)
    (d / "inference.py").write_text("MODEL_NAME = 'tiny'\nDEFAULT_TEMPERATURE = 1.5\n")
    return d


def test_chat_template_renders_empty_think_block_and_marker_positions(tiny_ckpt):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_ckpt)
    compiled, _ = compile_request(STATE, our_questions())
    text = tok.apply_chat_template(compiled.messages, tokenize=False, add_generation_prompt=False,
                                   enable_thinking=False, add_vision_id=True)
    assert text == GOLDEN_TEXT
    ids = tok(text, add_special_tokens=False)["input_ids"]
    marker = tok.convert_tokens_to_ids(DECISION_TOKEN)
    pos = [i - 1 for i, t in enumerate(ids) if t == marker]
    assert len(pos) == 3 and all(p >= 0 for p in pos)
    assert all(tok.decode(ids[p]).endswith('"') for p in pos)  # the token before each placeholder is the opening quote
    assert DECISION_TOKEN in tok.get_added_vocab()
    assert all(len(tok.encode(s, add_special_tokens=False)) == 1 for s in ANSWER_SYMBOLS)


def test_load_causal_lm_is_text_only_for_qwen3_5_composite_checkpoints(tiny_ckpt, monkeypatch):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

    from jevharness.training.decision_model import code_token_ids
    from jevharness.training.train_decision import load_causal_lm

    assert json.loads((tiny_ckpt / "config.json").read_text())["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    m = load_causal_lm(str(tiny_ckpt), torch.float32).eval()
    assert type(m).__name__ == "Qwen3_5ForCausalLM" and not hasattr(m.model, "visual")
    full = Qwen3_5ForConditionalGeneration.from_pretrained(tiny_ckpt, dtype=torch.float32).eval()
    assert sum(p.numel() for p in m.parameters()) < sum(p.numel() for p in full.parameters())
    ids = torch.randint(5, 200, (1, 11))
    with torch.no_grad():
        ours = m(input_ids=ids, logits_to_keep=torch.tensor([2, 9]), use_cache=False).logits
        ref = full(input_ids=ids, use_cache=False).logits[:, [2, 9]]
    assert ours.shape == (1, 2, m.config.vocab_size) and torch.allclose(ours, ref, atol=1e-5)
    assert len(code_token_ids(AutoTokenizer.from_pretrained(tiny_ckpt))) == 52  # option codes stay single tokens
    # older transformers: no causal-LM mapping for the composite config -> multimodal class, text-only forward
    real = AutoModelForCausalLM.from_pretrained

    def unrecognized(name, **kw):
        raise ValueError("Unrecognized configuration class <class 'Qwen3_5Config'> for this kind of AutoModel")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", unrecognized)
    fb = load_causal_lm(str(tiny_ckpt), torch.float32).eval()
    assert type(fb).__name__ == "Qwen3_5ForConditionalGeneration"
    with torch.no_grad():
        assert torch.allclose(fb(input_ids=ids, logits_to_keep=torch.tensor([2, 9]), use_cache=False).logits, ref, atol=1e-5)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", real)
    with pytest.raises(ValueError, match="other error"):
        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: (_ for _ in ()).throw(ValueError("other error")))
        load_causal_lm(str(tiny_ckpt), torch.float32)


@pytest.fixture(scope="module")
def backend(tiny_ckpt):
    b = make_backend({"type": "intern_decision", "model": str(tiny_ckpt), "device": "cpu"})
    assert isinstance(b, InternDecisionBackend)
    return b


def test_backend_registration_describe_and_ledger(backend, tiny_ckpt):
    d = backend.describe()
    assert d["backend"] == "intern_decision" and d["model"] == str(tiny_ckpt) and d["batch"] is False
    assert d["dtype"] == "float32" and d["device"] == "cpu" and d["max_length"] == 8192
    assert d["temperature"] == 1.5 and backend.temperature_source == "checkpoint inference.py"  # from inference.py
    assert backend.batched_questions and backend.max_options == len(ANSWER_SYMBOLS)
    led = Ledger()
    res = backend.decide(STATE, our_questions(), ledger=led)
    assert [a.kind for a in res.answers] == [QKind.CHOICE, QKind.SCORE, QKind.NOUL]
    assert res.answers[0].value in {"billing", "delivery"} and 0.0 <= res.answers[1].value <= 2.0
    assert isinstance(res.answers[2].value, bool) and 0.0 <= res.answers[2].p_yes <= 1.0
    for a in res.answers:
        assert math.isclose(sum(a.distribution.values()), 1.0, abs_tol=1e-6)
        assert a.raw["calibration"] == {"method": "temperature-scaling", "temperature": 1.5}
    u = res.usage
    assert u.n_calls == 1 and u.output_tokens == 3 and u.input_tokens > 50 and u.latency_ms > 0
    assert u.priced is False and u.cost_usd == 0.0  # the tiny dir has no price: flagged, not silently $0
    rec = led.records[-1]
    assert rec.component == "s1" and rec.backend == "intern_decision" and rec.decision_point == "team"
    assert rec.input_tokens == u.input_tokens and rec.priced is False and led.summary()["unpriced_models"] == [str(tiny_ckpt)]
    # a priced model name books the self-hosted price
    priced = make_backend({"type": "intern_decision", "model": str(tiny_ckpt), "device": "cpu",
                           "price_model": "internlm/Intern-Decision-0.8B", "temperature": 1.0})
    r2 = priced.decide(STATE, our_questions())
    assert r2.usage.priced is True and r2.usage.cost_usd == pytest.approx(r2.usage.input_tokens * 0.008 / 1e6)
    assert priced.describe()["temperature"] == 1.0


def test_backend_batches_fields_knockout_and_truncation(backend):
    # 17 questions -> two forward passes (<= 16 fields each)
    qs = [Noul(f"q{i}?", decision_point="stop") for i in range(17)]
    r = backend.decide("S", qs)
    assert len(r.answers) == 17 and r.usage.n_calls == 2 and r.usage.output_tokens == 17
    assert [a.raw["field"] for a in r.answers[:3]] == ["stop", "stop_2", "stop_3"]
    assert r.usage.n_passes == 2 and r.usage.n_rounds == 1
    # Choice beyond max_options: knockout like serve_jev (chunks, top-2 advance), reads linked into a full-set
    # distribution (decision/tournament.py); 9 options / width 4 -> chunks 4, 4, 1 -> 5 -> chunks 4, 1 -> 3 finalists
    b = make_backend({"type": "intern_decision", "model": backend.model, "device": "cpu", "max_options": 4})
    opts = [f"tool_{i}" for i in range(9)]
    r = b.decide("S", [Choice("which?", opts, decision_point="tool_select")])
    a = r.answers[0]
    ko = a.raw["knockout"]
    assert ko["rounds"] == 3 and ko["reads"] == 4 and ko["passes"] == 3 and ko["full_set"] == "linked"
    assert 2 <= len(a.raw["finalists"]) <= 4 and a.value in a.raw["finalists"] and a.raw["finalists"] == ko["finalists"]
    assert set(a.distribution) == set(opts) and r.usage.n_calls == 3 and r.usage.output_tokens >= 4
    assert r.usage.n_passes == 3 and r.usage.n_rounds == 3
    assert all(a.distribution[o] > 1e-5 for o in opts) and a.value == max(a.distribution, key=a.distribution.get)
    assert a.confidence == pytest.approx(a.distribution[a.value], abs=1e-5)
    legacy = make_backend({"type": "intern_decision", "model": backend.model, "device": "cpu", "max_options": 4,
                           "knockout_probs": "finalists"})
    la = legacy.decide("S", [Choice("which?", opts, decision_point="tool_select")]).answers[0]
    assert all(la.distribution[o] < 1e-5 for o in opts if o not in la.raw["finalists"]) and la.value == a.value
    assert legacy.describe()["knockout_probs"] == "finalists" and "knockout_probs" not in b.describe()
    # too-long prompts drop the oldest part of the state instead of failing
    long_state = "\n".join(f"step {i}: observation {i}" for i in range(300))
    short = make_backend({"type": "intern_decision", "model": backend.model, "device": "cpu", "max_length": 400})
    r = short.decide(long_state, [Noul("done?", decision_point="stop")])
    assert r.usage.input_tokens <= 400 and r.answers[0].raw["truncated_state"] is True and short.n_truncated == 1
    strict = make_backend({"type": "intern_decision", "model": backend.model, "device": "cpu", "max_length": 400,
                           "truncate_state": False})
    with pytest.raises(ValueError, match="truncation is forbidden"):
        strict.decide(long_state, [Noul("done?")])
    assert strict.describe()["truncate_state"] is False


def test_question_meta_never_reaches_the_prompt(backend, monkeypatch):
    """Gold labels live in ``Question.meta``; only the whitelisted ``wire_id`` / Noul ``criteria`` keys are read."""
    qs = [Choice("Which tool?", ["get_weather", "get_time"], decision_point="tool_select", gold="get_weather",
                 answer="SECRET-GOLD", notes={"leak": "SECRET-NOTE"}),
          Noul("Stop?", decision_point="stop", gold=True, hint="SECRET-HINT")]
    compiled, _ = compile_request("state", qs)
    blob = json.dumps(compiled.messages)
    assert "SECRET" not in blob and "gold" not in blob
    seen = {}
    orig = backend._forward

    def spy(ids, positions):
        seen["text"] = backend.tok.decode(ids[0])
        return orig(ids, positions)

    monkeypatch.setattr(backend, "_forward", spy)
    backend.decide("state", qs)
    assert "SECRET" not in seen["text"] and "gold" not in seen["text"] and "tool_select" in seen["text"]


def test_intern_backend_behind_the_jev_wire_server(backend):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from jevharness.decision.jev import JevBackend
    from jevharness.decision.server import build_app

    client = JevBackend(model="intern", base_url="http://testserver", api_key="")
    client._http = TestClient(build_app(backend, name="intern"))
    r = client.decide("S", [Choice("which?", ["a", "b", "c"], decision_point="tool_select"), Noul("ok?"),
                            Score("u", scale=(1, 5))])
    assert r.answers[0].value in {"a", "b", "c"} and set(r.answers[0].distribution) == {"a", "b", "c"}
    assert 0.0 <= r.answers[1].p_yes <= 1.0 and 1.0 <= r.answers[2].value <= 5.0
    assert r.usage.input_tokens > 0 and r.usage.model == "intern"
    models = client.list_models()["data"][0]
    assert models["backend"]["backend"] == "intern_decision" and models["backend"]["batch"] is False


# ------------------------------------------------------------------ GPU round 2 regressions: memory on one card
def test_instances_of_one_checkpoint_share_weights_and_a_device_lock(tiny_ckpt):
    """Experiments bind several System-1 specs to one checkpoint (calibrated + ``temperature: 1`` uncalibrated);
    loading it once per spec is what pushed an L4 into CUDA OOM in GPU round 2."""
    cal = make_backend({"type": "intern_decision", "model": str(tiny_ckpt), "device": "cpu"})
    uncal = make_backend({"type": "intern_decision", "model": str(tiny_ckpt), "device": "cpu", "temperature": 1.0})
    assert cal._model is uncal._model and uncal.shared_weights is True and uncal._lock is cal._lock
    assert cal.temperature == 1.5 and uncal.temperature == 1.0  # per-instance calibration on shared weights
    assert uncal.describe()["shared_weights"] is True
    r1, r2 = cal.decide(STATE, our_questions()), uncal.decide(STATE, our_questions())
    assert argmax(r1.answers[0].distribution) == argmax(r2.answers[0].distribution)  # same logits, other scaling
    assert r1.answers[0].distribution != r2.answers[0].distribution


def test_forward_retries_once_after_an_out_of_memory_error(tiny_ckpt, monkeypatch):
    import torch

    from jevharness.decision.intern_decision import _oom_error

    b = make_backend({"type": "intern_decision", "model": str(tiny_ckpt), "device": "cpu"})
    real = b._model.forward
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _oom_error()("CUDA out of memory (simulated)")
        return real(*a, **k)

    monkeypatch.setattr(b._model, "forward", flaky)
    res = b.decide(STATE, our_questions())
    assert len(res.answers) == 3 and calls["n"] == 2 and b.n_oom_retries == 1
    monkeypatch.setattr(b._model, "forward", lambda *a, **k: (_ for _ in ()).throw(_oom_error()("still OOM")))
    with pytest.raises(_oom_error()):  # a second failure is real
        b.decide(STATE, our_questions())
    assert isinstance(torch.zeros(1), torch.Tensor)
