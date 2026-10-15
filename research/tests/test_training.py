import math
import random

import pytest

from jevharness.training.data import mc_q_estimate, mc_target, permuted, render, split_by_task


def _ex(i=0):
    return {"id": f"task{i}/0/tool_select", "state": "S", "question": "Q?", "options": ["a", "b", "c"],
            "target": {"a": 0.0, "b": 1.0, "c": 0.0}}


def test_render_and_permute_keep_target_aligned():
    prompt, codes = render(_ex())
    assert "A. a" in prompt and codes == ["A", "B", "C"]
    rng = random.Random(3)
    for _ in range(20):
        p, tgt, idx = permuted(_ex(), rng)
        assert math.isclose(sum(tgt), 1.0)
        pos = tgt.index(1.0)
        assert _ex()["options"][idx[pos]] == "b"
        assert f"{'ABC'[pos]}. b" in p


def test_mc_target_prefers_high_q_and_respects_reference():
    t = mc_target({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}, beta=0.2)
    assert t["a"] > 0.95
    t2 = mc_target({"a": 0.01, "b": 0.99}, {"a": 0.6, "b": 0.5}, beta=1.0)
    assert t2["b"] > t2["a"]  # weak Q evidence does not overturn a strong reference
    assert mc_q_estimate(0, 4) == pytest.approx(1 / 6)


def test_split_by_task_has_no_leakage():
    ex = [dict(_ex(i), id=f"task{i}/{s}/x") for i in range(20) for s in range(3)]
    tr, dv = split_by_task(ex, 0.2)
    assert {e["id"].split("/")[0] for e in tr}.isdisjoint({e["id"].split("/")[0] for e in dv})


def test_tiny_model_train_calibrate_serve(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("fastapi")
    tokenizers = pytest.importorskip("tokenizers")
    from fastapi.testclient import TestClient
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from jevharness.decision import Choice, Noul
    from jevharness.decision.jev import JevBackend
    from jevharness.training.data import write_jsonl
    from jevharness.training.serve_jev import Engine, build_app
    from jevharness.training.train_decision import main as train_main

    tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<pad>", "<|im_end|>"],
                                             initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["STATE QUESTION OPTIONS Answer with the option label only."] * 10, trainer)
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>")
    ft.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                        "{% if add_generation_prompt %}assistant: {% endif %}")
    cfg = Qwen3Config(vocab_size=len(ft), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, tie_word_embeddings=True)
    mdir = tmp_path / "tiny"
    Qwen3ForCausalLM(cfg).save_pretrained(mdir)
    ft.save_pretrained(mdir)
    data = [dict(_ex(i), id=f"t{i}/0/x") for i in range(24)]
    write_jsonl(data, tmp_path / "train.jsonl")
    out = tmp_path / "ckpt"
    train_main(["--model", str(mdir), "--train", str(tmp_path / "train.jsonl"), "--out", str(out),
                "--lora-r", "4", "--lora-alpha", "8", "--lr", "1e-2", "--epochs", "4", "--batch-size", "8",
                "--grad-accum", "1", "--max-len", "256", "--no-bf16", "--no-permute"])
    eng = Engine(str(mdir), str(out), name="tiny")
    client = TestClient(build_app(eng))
    b = JevBackend(model="tiny", base_url="http://testserver", api_key="")
    b._http = client
    r = b.decide("S", [Choice("Q?", ["a", "b", "c"]), Noul("ok?")])
    assert r.answers[0].value in {"a", "b", "c"} and 0 <= r.answers[1].p_yes <= 1
    assert r.answers[0].value == "b"  # fixed order + constant target: learnable by a 1-layer model
    _ = torch
