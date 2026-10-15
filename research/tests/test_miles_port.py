"""JevAgentRL-v2 (the Miles port, ``training/miles_port``): CPU tests with a tiny Qwen3-style model, without Miles.

The SGLang engine is replaced by a function that serves the tiny model's full-vocabulary log-softmax at the decision
position (what ``SGLangClient`` asks the engine for), and Miles' packed micro-batch by the concatenation of per-sample
logits (what a correct packed forward returns). Checked:
  * the SGLang decider reproduces v1's TorchDecider exactly (same events, same stored behaviour log-probabilities) for
    replace / cascade / assist / guard and the > 52-option knockout (A7.10 (f): stored = replayed);
  * the custom loss slices the decision row of thd and bshd layouts, restricts it to the option codes and equals v1's
    SFT / OPD / GRPO objectives; before any update the GRPO ratio is 1; masked placeholders contribute nothing;
  * SFT / OPD / GRPO rollout functions build well-formed samples (fixed counts, siblings per episode, metadata);
  * the trainer key patch, the launcher's Miles command line, the seed guard and the strict DCP -> HF export.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from jevharness.training.miles_port.sglang_client import build_payload, parse_option_logprobs  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<pad>", "<|im_end|>"],
                                             initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["STATE QUESTION OPTIONS Answer with the option label only. TASK HISTORY"] * 10, trainer)
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>")
    ft.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                        "{% if add_generation_prompt %}assistant: {% endif %}")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(ft), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, tie_word_embeddings=True,
                      eos_token_id=ft.eos_token_id, pad_token_id=ft.pad_token_id)
    m = Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        m.model.embed_tokens.weight.mul_(6.0)
    d = tmp_path_factory.mktemp("tiny_miles")
    m.save_pretrained(d)
    ft.save_pretrained(d)
    return d


def _load(tiny: Path):  # noqa: ANN202
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny)
    tok.padding_side = "left"
    return AutoModelForCausalLM.from_pretrained(tiny).float().eval(), tok


def _fake_engine(net):  # noqa: ANN001, ANN202
    """``SGLangClient(post=...)``: the engine's answer for a /generate payload = log-softmax at the last position."""
    calls = []

    def post(payload):  # noqa: ANN001, ANN202
        calls.append(payload)
        single = bool(payload["input_ids"]) and isinstance(payload["input_ids"][0], int)   # the router's format
        ids_l = [payload["input_ids"]] if single else payload["input_ids"]
        codes_l = [payload["token_ids_logprob"]] if single else payload["token_ids_logprob"]
        out = []
        for ids, codes in zip(ids_l, codes_l):
            with torch.no_grad():
                lg = net(input_ids=torch.tensor([ids])).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
            out.append({"text": "x", "meta_info": {"output_token_ids_logprobs": [[[float(lp[c]), int(c), None]
                                                                                  for c in codes]]}})
        return out[0] if single else out

    post.calls = calls  # type: ignore[attr-defined]
    return post


# ------------------------------------------------------------------ SGLang client
def test_payload_and_response_parsing():
    p = build_payload([[1, 2, 3], [4]], [[7, 8], [7, 8, 9]])
    assert p["sampling_params"]["max_new_tokens"] == 1 and p["sampling_params"]["temperature"] == 1.0
    assert p["token_ids_logprob"] == [[7, 8], [7, 8, 9]] and p["return_logprob"] and p["logprob_start_len"] == -1
    resp = [{"meta_info": {"output_token_ids_logprobs": [[[-0.1, 7, None], [-2.0, 8, None]]]}},
            {"meta_info": {"output_token_ids_logprobs": [[{"logprob": -1.0, "token_id": 9}, [-3.0, 7, "A"]]]}}]
    got = parse_option_logprobs(resp, [[7, 8], [7, 8, 9]])
    assert got[0] == [-0.1, -2.0] and got[1][0] == -3.0 and got[1][2] == -1.0 and math.isinf(got[1][1])
    with pytest.raises(ValueError):
        parse_option_logprobs(resp[:1], [[7, 8], [7]])


def test_router_format_one_prompt_per_request():
    """Miles' router rejects batched ``token_ids_logprob``: the default client sends flat single-prompt payloads,
    one per prompt, and returns the answers in prompt order; ``batched=True`` keeps the one-request form."""
    from jevharness.training.miles_port.sglang_client import SGLangClient, build_single_payload

    p = build_single_payload([1, 2, 3], [7, 8])
    assert p["input_ids"] == [1, 2, 3] and p["token_ids_logprob"] == [7, 8] and p["return_logprob"]
    seen = []

    def post(payload):  # noqa: ANN001, ANN202
        seen.append(payload)
        if isinstance(payload["input_ids"][0], list):
            return [{"meta_info": {"output_token_ids_logprobs": [[[-float(len(i)), c[0], None]]]}}
                    for i, c in zip(payload["input_ids"], payload["token_ids_logprob"])]
        assert all(isinstance(t, int) for t in payload["token_ids_logprob"])
        return {"meta_info": {"output_token_ids_logprobs": [[[-float(len(payload["input_ids"])),
                                                             payload["token_ids_logprob"][0], None]]]}}

    cl = SGLangClient("http://x", post=post)
    got = cl.option_logprobs([[1], [1, 2, 3], [1, 2]], [[5], [6], [7]])
    assert got == [[-1.0], [-3.0], [-2.0]] and len(seen) == 3 and cl.requests == 3
    seen.clear()
    assert SGLangClient("http://x", post=post, batched=True).option_logprobs([[1], [1, 2]], [[5], [6]]) == \
        [[-1.0], [-2.0]] and len(seen) == 1


# ------------------------------------------------------------------ decider parity with v1
@pytest.mark.parametrize("mode", ["replace", "cascade", "assist", "guard", "knockout"])
def test_sglang_decider_reproduces_torch_decider(tiny: Path, mode: str):
    from jevharness.decision import Choice, Noul
    from jevharness.harness.config import HarnessConfig
    from jevharness.training.grpo_online import TorchDecider
    from jevharness.training.miles_port.sglang_client import SGLangClient
    from jevharness.training.miles_port.sglang_decider import SGLangDecider

    net, tok = _load(tiny)
    dpc = {"replace": {"mode": "replace"}, "cascade": {"mode": "cascade", "tau": 0.3},
           "assist": {"mode": "assist", "k": 2}, "guard": {"mode": "guard", "tau": 0.5},
           "knockout": {"mode": "replace"}}[mode]
    arm = HarnessConfig(name=mode, tool_select=dpc)
    opts = [f"tool_{i}" for i in range(60)] if mode == "knockout" else ["alpha", "beta", "gamma"]
    q = (Noul("is the proposed call right?", decision_point="tool_select") if mode == "guard"
         else Choice("which tool next?", opts, decision_point="tool_select", gold=opts[1]))
    ref = TorchDecider(net, tok, arm, temperature=0.8, max_len=2048)
    sg = SGLangDecider(tok, "http://unused", arm, client=SGLangClient("http://x", post=_fake_engine(net)),
                       temperature=0.8, max_len=2048)
    for s in range(40):
        recs = []
        for d in (ref, sg):
            d.begin_episode(s, arm)
            d.decide("STATE: s", [q])
            recs += d.end_episode()
        a, b = recs
        assert (a.ev, a.event, a.sampled, a.greedy, a.n, a.train, a.prompt) == \
               (b.ev, b.event, b.sampled, b.greedy, b.n, b.train, b.prompt)
        assert abs(a.old_logp - b.old_logp) < 1e-5 and abs(a.conf - b.conf) < 1e-5


# ------------------------------------------------------------------ the custom loss
def _packed_logits(net, samples, layout: str):  # noqa: ANN001, ANN202
    """What Miles hands the loss: thd = per-sample logits concatenated ([1, T, V]); bshd = right-padded [B, S, V]."""
    outs = [net(input_ids=torch.tensor([s.tokens])).logits[0] for s in samples]
    if layout == "thd":
        return torch.cat(outs, 0)[None], None
    S = max(o.size(0) for o in outs)
    pad = [torch.cat([o, o.new_zeros(S - o.size(0), o.size(1))], 0) for o in outs]
    return torch.stack(pad, 0), [S] * len(outs)


def _batch(samples, max_seq_lens=None):  # noqa: ANN001, ANN202
    return {"metadata": [s.train_metadata for s in samples], "total_lengths": [len(s.tokens) for s in samples],
            "response_lengths": [1] * len(samples), "loss_masks": [torch.tensor(s.loss_mask) for s in samples],
            "max_seq_lens": max_seq_lens}


def _mean_reducer(samples):  # noqa: ANN001, ANN202
    masks = torch.tensor([float(s.loss_mask[0]) for s in samples])
    return lambda x: (x * masks).sum()


class _Args:
    qkv_format = "thd"


@pytest.mark.parametrize("layout", ["thd", "bshd"])
def test_sft_and_opd_losses_equal_v1(tiny: Path, layout: str):
    from jevharness.training.decision_model import DecisionCollator, decision_loss, option_logits
    from jevharness.training.miles_port.loss import jev_decision_loss
    from jevharness.training.miles_port.rollout import make_sample
    from jevharness.training.opd import opd_loss

    net, tok = _load(tiny)
    col = DecisionCollator(tok, max_len=512)
    rows = [{"prompt": "STATE: a\nQUESTION: q", "target": [0.7, 0.2, 0.1]},
            {"prompt": "STATE: bb\nQUESTION: qq", "target": [0.0, 1.0]},
            {"prompt": "STATE: c", "target": [0.1, 0.1, 0.1, 0.7]}]
    for stage in ("sft", "opd"):
        samples = []
        for i, r in enumerate(rows):
            codes = col.codes[:len(r["target"])]
            meta = {"stage": stage, "codes": codes, "target": r["target"], "brier": 0.5, "loss": "forward_kl"}
            samples.append(make_sample(col.encode(r["prompt"]), codes[0], meta, index=i, group_index=i, rollout_id=i))
        logits, msl = _packed_logits(net, samples, layout)
        args = _Args()
        args.qkv_format = layout
        loss, st = jev_decision_loss(args, _batch(samples, msl), logits, _mean_reducer(samples))
        enc = col(rows)
        z = option_logits(net, enc["input_ids"], enc["attention_mask"], enc["code_ids"])
        want = (decision_loss(z, enc, "soft_ce_brier", brier_weight=0.5) * len(rows) if stage == "sft"
                else opd_loss(z, enc["target"], "forward_kl").sum())
        assert abs(float(loss) - float(want)) < 1e-4, (stage, float(loss), float(want))
        assert set(st) >= {"loss"}


def test_rows_only_logits_match_the_packed_path(tiny: Path):
    """keys_patch.patch_fsdp_actor: the LM head runs at the decision positions only (logits_to_keep on the packed
    sequence); the loss then reads one row per sample and gets the same value as from full packed logits."""
    import types

    from jevharness.training.decision_model import DecisionCollator
    from jevharness.training.miles_port import keys_patch
    from jevharness.training.miles_port.loss import jev_decision_loss
    from jevharness.training.miles_port.rollout import make_sample

    net, tok = _load(tiny)
    col = DecisionCollator(tok, max_len=512)
    samples = []
    for i, t in enumerate(([0.6, 0.4], [0.1, 0.2, 0.7], [1.0, 0.0, 0.0, 0.0])):
        codes = col.codes[:len(t)]
        samples.append(make_sample(col.encode(f"STATE: {i} x{i}"), codes[0], {"stage": "sft", "codes": codes,
                                   "target": list(t), "brier": 0.5}, index=i, group_index=i, rollout_id=i))
    full, _ = _packed_logits(net, samples, "thd")
    loss_full, _ = jev_decision_loss(_Args(), _batch(samples), full, _mean_reducer(samples))

    class Actor:  # the patched method's contract: orig(self, batch) -> model kwargs
        def __init__(self):
            self.args = types.SimpleNamespace(loss_type="custom_loss", qkv_format="thd")

        def _get_model_inputs_args(self, batch):
            return {"input_ids": batch["tokens"]}

    mod = types.SimpleNamespace(FSDPTrainRayActor=Actor)
    keys_patch.patch_fsdp_actor(mod)
    packed = torch.tensor([sum((s.tokens for s in samples), [])])
    batch = {**_batch(samples), "tokens": packed}
    kw = Actor()._get_model_inputs_args(batch)
    assert batch["_jev_rows_only"] and kw["logits_to_keep"].tolist() == keys_patch.decision_positions(
        batch["total_lengths"], batch["response_lengths"])
    rows = full[:, kw["logits_to_keep"], :]
    loss_rows, _ = jev_decision_loss(_Args(), batch, rows, _mean_reducer(samples))
    assert abs(float(loss_rows) - float(loss_full)) < 1e-6


def test_grpo_loss_replays_stored_behaviour_and_masks_placeholders(tiny: Path):
    from jevharness.decision import Choice
    from jevharness.harness.config import HarnessConfig
    from jevharness.training.grpo_online import event_logp
    from jevharness.training.miles_port.loss import jev_decision_loss
    from jevharness.training.miles_port.rollout import make_sample
    from jevharness.training.miles_port.sglang_client import SGLangClient
    from jevharness.training.miles_port.sglang_decider import SGLangDecider

    net, tok = _load(tiny)
    samples = []
    for mode, dpc in (("r", {"mode": "replace"}), ("c", {"mode": "cascade", "tau": 0.3}),
                      ("a", {"mode": "assist", "k": 2})):
        arm = HarnessConfig(name=mode, tool_select=dpc)
        d = SGLangDecider(tok, "http://unused", arm, client=SGLangClient("http://x", post=_fake_engine(net)),
                          temperature=0.7, max_len=512)
        for s in range(6):
            d.begin_episode(s, arm)
            d.decide(f"STATE: {mode}{s}", [Choice("which?", ["a", "b", "c", "d"], decision_point="tool_select")])
            for r in d.end_episode():
                if not r.train:
                    continue
                codes = d.collator.codes[:r.n]
                meta = {"stage": "grpo", "codes": codes, "event": r.event, "ev": r.ev, "old_logp": r.old_logp,
                        "adv": 1.0 if s % 2 else -0.5, "temperature": 0.7, "clip": 0.2, "clip_high": 0.28}
                samples.append(make_sample(d.collator.encode(r.prompt), codes[r.sampled], meta, index=len(samples),
                                           group_index=0, rollout_id=s))
    ph_meta = {"stage": "grpo", "codes": samples[0].train_metadata["codes"][:2], "event": [0], "ev": "choice",
               "old_logp": math.log(0.5), "adv": 0.0, "temperature": 1.0, "clip": 0.2, "clip_high": 0.2}
    samples.append(make_sample([1, 2, 3], samples[0].train_metadata["codes"][0], ph_meta, index=999, group_index=0,
                               rollout_id=99, train=False))
    logits, _ = _packed_logits(net, samples, "thd")
    logits = logits.detach().requires_grad_(True)
    loss, st = jev_decision_loss(_Args(), _batch(samples), logits, _mean_reducer(samples))
    n = len(samples) - 1
    assert float(st["logp_gap"]) / n < 1e-4                          # stored behaviour == replayed (ratio 1)
    assert abs(float(st["ratio"]) / n - 1.0) < 1e-4 and float(st["clipped"]) == 0.0
    want = -sum(s.train_metadata["adv"] for s in samples[:-1])        # ratio 1: pg = -A
    assert abs(float(loss) - want) < 1e-3
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    # the placeholder row receives no gradient
    last = sum(len(s.tokens) for s in samples) - 2
    assert float(logits.grad[0, last].abs().sum()) == 0.0
    # after an update the ratio is exp(current - stored) on the restricted event probability
    s0 = samples[0]
    with torch.no_grad():
        z = net(input_ids=torch.tensor([s0.tokens])).logits[0, -2].float()
    zr = z.index_select(0, torch.tensor(s0.train_metadata["codes"]))
    lp = torch.log_softmax(zr / 0.7, -1)
    assert abs(float(event_logp(lp, s0.train_metadata["event"], s0.train_metadata["ev"])) -
               s0.train_metadata["old_logp"]) < 1e-4


# ------------------------------------------------------------------ rollout functions
def _write_stage(tmp_path: Path, tiny: Path, stage: str, **kw) -> Path:  # noqa: ANN003
    import yaml

    base = {"name": f"t_{stage}", "stage": stage, "base": str(tiny), "out": str(tmp_path / stage), "max_len": 512,
            "seed": 3, "miles": {"actor_gpus": 1, "rollout_gpus": 1, "global_batch_size": 4}}
    base.update(kw)
    p = tmp_path / f"{stage}.yaml"
    p.write_text(yaml.safe_dump(base))
    return p


def test_sft_rollout_is_deterministic_and_permutes(tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from jevharness.training.data import write_jsonl
    from jevharness.training.miles_port import rollout as R
    from jevharness.training.miles_port.config import load_stage_config

    rows = [{"id": f"r{i}", "source": "t", "dp": "tool_select", "kind": "choice", "state": f"s{i}", "question": "q",
             "options": ["a", "b", "c"], "target": {"a": 0.0, "b": 1.0, "c": 0.0}, "q_values": None, "meta": {}}
            for i in range(10)]
    write_jsonl(rows, tmp_path / "train.jsonl")
    cfgp = _write_stage(tmp_path, tiny, "sft", sft={"train": [str(tmp_path / "train.jsonl")], "epochs": 2,
                                                     "batch_rows": 4})
    monkeypatch.setenv("JEV_MILES_CONFIG", str(cfgp))
    R._SFT.clear()
    cfg = load_stage_config()
    assert R.sft_num_rollouts(10, 2, 4) == 5
    g1 = R.sft_rollout(None, 1)
    R._SFT.clear()
    g2 = R.sft_rollout(None, 1)
    assert [s[0].tokens for s in g1] == [s[0].tokens for s in g2] and len(g1) == cfg.sft.batch_rows
    seen = set()
    for (s,) in g1 + R.sft_rollout(None, 0) + R.sft_rollout(None, 2):
        m = s.train_metadata
        assert m["stage"] == "sft" and len(m["codes"]) == 3 and s.response_length == 1
        assert s.tokens[-1] == m["codes"][max(range(3), key=lambda i: m["target"][i])]   # gold code appended
        seen.add(tuple(m["target"]))
    assert len(seen) > 1                                                                    # options permuted


def test_harness_rollouts_build_opd_and_grpo_samples(tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from jevharness.llm import make_llm
    from jevharness.training.miles_port import rollout as R
    from jevharness.training.miles_port.config import load_stage_config
    from jevharness.training.miles_port.sglang_client import SGLangClient

    net, _ = _load(tiny)
    common = {"rollout": {"suite": {"families": "v2", "n_distractors": 4, "strict": True}, "seed0": 60000,
                          "n_tasks": 8, "tasks_per_iter": 2, "group_size": 2, "iterations": 2, "workers": 1,
                          "arms": [{"name": "rep", "tool_select": {"mode": "replace"},
                                    "tool_params": {"mode": "replace"}},
                                   {"name": "cas", "tool_select": {"mode": "cascade", "tau": 0.9}}],
                          "executor": {"provider": "simulated", "model": "scripted", "error_rate": 0.1},
                          "max_steps": 6, "decider_batch": 4}}
    ex = make_llm(common["rollout"]["executor"])
    # GRPO
    cfgp = _write_stage(tmp_path, tiny, "grpo", grpo={"reward": {"kind": "gated"}}, **common)
    monkeypatch.setenv("JEV_MILES_CONFIG", str(cfgp))
    cfg = load_stage_config()
    h = R.Harness(cfg, "http://unused", client=SGLangClient("http://x", post=_fake_engine(net)), executor=ex)
    eps, row = h.run(0)
    groups = R.grpo_samples(h, eps, 0, row)
    assert len(groups) == 4 and row["episodes"] == 4                       # tasks_per_iter x group_size rollouts
    for gi, sibs in enumerate(groups):
        assert len({s.rollout_id for s in sibs}) == 1 and sibs[0].rollout_id == gi
        for s in sibs:
            m = s.train_metadata
            assert m["stage"] == "grpo" and s.response_length == 1 and s.tokens[-1] in m["codes"]
            assert s.loss_mask[0] in (0, 1) and (s.loss_mask[0] == 0 or m["ev"] in ("choice", "commit", "defer"))
    assert h.task_ids(0) != h.task_ids(1)
    # OPD (simulated teacher: the only kind that may see the oracle gold)
    cfgp = _write_stage(tmp_path, tiny, "opd", opd={"teacher": '{"type": "simulated", "accuracy": 0.9, "seed": 1, '
                                                               '"latency_ms": 0}', "fresh_rows": 6, "replay_rows": 2},
                        **{**common, "rollout": {**common["rollout"], "group_size": 1}})
    monkeypatch.setenv("JEV_MILES_CONFIG", str(cfgp))
    cfg = load_stage_config()
    h = R.Harness(cfg, "http://unused", client=SGLangClient("http://x", post=_fake_engine(net)), executor=ex)
    for it in range(2):
        eps, row = h.run(it)
        groups = R.opd_samples(h, eps, it, row)
        assert len(groups) == 8 and all(len(g) == 1 for g in groups)       # fixed count: fresh + replay (+ resampled)
        for (s,) in groups:
            m = s.train_metadata
            assert m["stage"] == "opd" and abs(sum(m["target"]) - 1.0) < 1e-6 and len(m["codes"]) == len(m["target"])
        assert (Path(cfg.out) / "labelled" / f"iter-{it:05d}.jsonl").exists()
    assert row["replay_n"] > 0


# ------------------------------------------------------------------ plumbing
FAKE_PACKER = '''
def _package_shards(args, data, partitions):
    shards = []
    for part in partitions:
        shards.append({"partition": part, "tokens": [data["tokens"][j] for j in part]})
    return shards
'''


FAKE_ACTOR = '''
from types import SimpleNamespace
TRAIN_KEYS = ["tokens", "loss_masks"]


def get_parallel_state():
    return SimpleNamespace(intra_dp=SimpleNamespace(size=8), cp=SimpleNamespace(size=1), vpp_size=None,
                           microbatch_group_size_per_vp_stage=None)


class TorchNativeTrainRayActor:
    @property
    def train_parallel_config(self):
        return {"dp_size": get_parallel_state().intra_dp.size}
'''


def _fake_miles(root: Path) -> None:
    """The two Miles modules the key patch touches, reduced to the patched names."""
    actor = root / "miles" / "backends" / "training_utils" / "torch_native"
    conv = root / "miles" / "ray" / "rollout"
    actor.mkdir(parents=True)
    conv.mkdir(parents=True)
    for d in (root / "miles", root / "miles" / "backends", root / "miles" / "backends" / "training_utils", actor,
              root / "miles" / "ray", conv):
        (d / "__init__.py").write_text("")
    (actor / "actor.py").write_text(FAKE_ACTOR)
    (conv / "train_data_conversion.py").write_text(FAKE_PACKER)


def test_keys_patch_carries_metadata_to_the_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from jevharness.training.miles_port import keys_patch

    _fake_miles(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    for k in [k for k in sys.modules if k == "miles" or k.startswith("miles.")]:
        monkeypatch.delitem(sys.modules, k)
    keys_patch.install()
    try:
        import importlib

        mod = importlib.import_module(keys_patch.TARGET)
        assert mod.TRAIN_KEYS == ["tokens", "loss_masks", "metadata", "rollout_mask_sums"]
        # the FSDP backend advertises the full schedule config (compact rollouts are planned on the rollout side)
        assert mod.TorchNativeTrainRayActor().train_parallel_config == {
            "dp_size": 8, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}
        conv = importlib.import_module(keys_patch.CONVERSION)
        data = {"tokens": [[1], [2], [3], [4]], "metadata": [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]}
        shards = conv._package_shards(None, data, [[0, 2], [1, 3]])
        assert [s["metadata"] for s in shards] == [[{"i": 0}, {"i": 2}], [{"i": 1}, {"i": 3}]]
        assert [s["tokens"] for s in shards] == [[[1], [3]], [[2], [4]]]
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, keys_patch._Finder)]
        for k in [k for k in sys.modules if k == "miles" or k.startswith("miles.")]:
            sys.modules.pop(k, None)


def test_startup_sitecustomize_patches_a_fresh_interpreter(tmp_path: Path):
    """The route Ray workers take: a NEW interpreter with the startup dir on PYTHONPATH gets the patched keys."""
    import os
    import subprocess

    from jevharness.training.miles_port.launch import STARTUP_DIR

    _fake_miles(tmp_path)
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(STARTUP_DIR), str(tmp_path), str(src)])}
    code = ("import miles.backends.training_utils.torch_native.actor as a, miles.ray.rollout.train_data_conversion as c; "
            "print(a.TRAIN_KEYS, c._package_shards(None, {'tokens': [[1]], 'metadata': [{'x': 1}]}, [[0]])[0]['metadata'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert out.stdout.strip() == "['tokens', 'loss_masks', 'metadata', 'rollout_mask_sums'] [{'x': 1}]", \
        out.stdout + out.stderr


def test_launcher_command_line_and_seed_guard(tiny: Path, tmp_path: Path):
    from jevharness.training.miles_port.config import load_stage_config
    from jevharness.training.miles_port.launch import miles_args, resolve

    roll = {"suite": {}, "seed0": 70000, "n_tasks": 100, "tasks_per_iter": 8, "group_size": 8, "iterations": 10,
            "arms": [{"name": "r", "tool_select": {"mode": "replace"}}], "executor": {"provider": "simulated"}}
    cfgp = _write_stage(tmp_path, tiny, "grpo", grpo={}, rollout=roll,
                        miles={"actor_gpus": 4, "rollout_gpus": 1, "global_batch_size": 16})
    cfg = resolve(load_stage_config(cfgp))
    a = miles_args(cfg, 10)
    s = " ".join(a)
    for frag in ("--train-backend fsdp", "--loss-type custom_loss", "--rollout-batch-size 8",
                 "--n-samples-per-prompt 8", "--global-batch-size 16", "--actor-num-gpus-per-node 4",
                 "--rollout-num-gpus 1", "--num-gpus-per-node 5", "--disable-compute-advantages-and-returns",
                 "miles_port.loss.jev_decision_loss", "miles_port.rollout.harness_rollout"):
        assert frag in s, frag
    assert "--debug-train-only" not in a and "--colocate" not in a and cfg.init == str(tiny)
    assert a[a.index("--save-interval") + 1] == "10"                      # no interval set: save at the end only
    col = _write_stage(tmp_path, tiny, "grpo", grpo={}, rollout=roll,
                       miles={"actor_gpus": 8, "rollout_gpus": 0, "colocate": True, "global_batch_size": 16})
    a = miles_args(resolve(load_stage_config(col)), 10)
    assert "--colocate" in a and "--rollout-num-gpus" not in a and a[a.index("--num-gpus-per-node") + 1] == "8"
    bad = _write_stage(tmp_path, tiny, "grpo", grpo={}, rollout=roll,
                       miles={"actor_gpus": 3, "rollout_gpus": 1, "global_batch_size": 16})
    with pytest.raises(ValueError):                                         # 16 episodes over 3 ranks
        miles_args(resolve(load_stage_config(bad)), 10)
    for bad in (200000, 300100, 5000, 100050, 9000):
        p = _write_stage(tmp_path, tiny, "grpo", grpo={}, rollout={**roll, "seed0": bad, "n_tasks": 50})
        with pytest.raises(ValueError):
            load_stage_config(p)
    p = _write_stage(tmp_path, tiny, "grpo", grpo={}, rollout={**roll, "n_tasks": 50})
    with pytest.raises(ValueError):                                         # 8 x 10 tasks > 50: no task reuse
        load_stage_config(p)


def test_launcher_env_overrides_and_allocator(tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from jevharness.training.miles_port.config import load_stage_config
    from jevharness.training.miles_port.launch import alloc_env, apply_overrides, miles_args, resolve

    roll = {"suite": {}, "seed0": 70000, "n_tasks": 100, "tasks_per_iter": 8, "group_size": 8, "iterations": 10,
            "arms": [{"name": "r", "tool_select": {"mode": "replace"}}], "executor": {"provider": "simulated"}}
    col = resolve(load_stage_config(_write_stage(tmp_path, tiny, "grpo", grpo={}, rollout=roll, miles={
        "actor_gpus": 8, "colocate": True, "global_batch_size": 16, "max_tokens_per_gpu": 16384})))
    monkeypatch.delenv("JEV_MILES_MAX_TOKENS", raising=False)
    assert apply_overrides(col).miles.max_tokens_per_gpu == 16384
    monkeypatch.setenv("JEV_MILES_MAX_TOKENS", "8192")
    over = apply_overrides(col)
    a = miles_args(over, 10)
    assert a[a.index("--max-tokens-per-gpu") + 1] == "8192" and over.miles.lr == col.miles.lr
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    alloc_env(col, env)                      # colocated: TorchMemorySaver refuses expandable segments
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    raw = col.model_dump()
    raw["miles"]["colocate"], raw["miles"]["rollout_gpus"] = False, 1
    env = {}
    alloc_env(type(col).model_validate(raw), env)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_strict_export_round_trip(tiny: Path, tmp_path: Path):
    import torch.distributed.checkpoint as dcp

    from jevharness.training.miles_port.export import export

    net, _ = _load(tiny)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.01)                                                    # "trained" weights differ from the origin
    it = tmp_path / "dcp" / "iter_0000003"
    dcp.save({"model_state": {"model": {k: v.clone() for k, v in net.state_dict().items()}}},
             checkpoint_id=str(it / "model"))
    (tmp_path / "dcp" / "latest_checkpointed_iteration.txt").write_text("3")
    hf = tmp_path / "hf"                       # the stage directory: records stay, stale weights go
    hf.mkdir()
    (hf / "rollout_metrics.jsonl").write_text('{"rollout_id": 0}\n')
    (hf / "model-00002-of-00002.safetensors").write_bytes(b"stale")
    info = export(str(tmp_path / "dcp"), str(tiny), str(hf))
    assert info["source"].endswith("iter_0000003") and info["weights_sha256"]
    assert (hf / "rollout_metrics.jsonl").read_text() == '{"rollout_id": 0}\n'
    assert not (hf / "model-00002-of-00002.safetensors").exists()
    from transformers import AutoModelForCausalLM

    got = AutoModelForCausalLM.from_pretrained(tmp_path / "hf").float()
    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        a = got(input_ids=ids).logits
        b = net.to(torch.bfloat16).float()(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-2)
    bad = tmp_path / "dcp_bad" / "iter_0000001"
    dcp.save({"model_state": {"model": {"nonsense.weight": torch.zeros(2)}}}, checkpoint_id=str(bad / "model"))
    with pytest.raises(RuntimeError):
        export(str(tmp_path / "dcp_bad"), str(tiny), str(tmp_path / "hf_bad"))
    assert json.loads((tmp_path / "hf" / "export.json").read_text())["class"]


def test_committed_stage_configs_load_and_use_disjoint_training_pools():
    import yaml

    from jevharness.training.miles_port.config import load_stage_config
    from jevharness.training.seed_guard import check_dev_pool

    root = Path(__file__).resolve().parents[1]
    cfgs = {s: load_stage_config(root / f"configs/train/miles_{s}_v2.yaml") for s in ("sft", "opd", "grpo")}
    assert [c.stage for c in cfgs.values()] == ["sft", "opd", "grpo"]
    assert all(c.miles.global_batch_size % c.miles.actor_gpus == 0 for c in cfgs.values())
    pools = [(c.rollout.seed0, c.rollout.seed0 + c.rollout.n_tasks - 1) for c in cfgs.values() if c.rollout]
    used = [(10000, 12059), (30000, 30599), (40000, 43999), (50000, 52999)]   # data v0, data v1, v1 OPD, v1 GRPO
    spans = pools + used
    for i, (a0, a1) in enumerate(spans):
        for b0, b1 in spans[i + 1:]:
            assert a1 < b0 or b1 < a0, (a0, a1, b0, b1)
    g = cfgs["grpo"]
    assert g.grpo.algo.kl_beta == 0.0 and g.grpo.reward.kind == "gated" and g.grpo.reward.lambda_gate == 0.5
    assert (g.grpo.reward.cost_unit, g.grpo.reward.cost_scale, g.grpo.reward.irreversible_penalty) == ("calls", 10.0, 0.2)
    assert (g.rollout.tasks_per_iter * g.rollout.group_size) % g.miles.global_batch_size == 0
    o = cfgs["opd"]
    assert (o.opd.fresh_rows + o.opd.replay_rows) % o.miles.global_batch_size == 0
    stress = yaml.safe_load((root / "configs/experiments/jevgym_stress_dev_miles.yaml").read_text())
    s = stress["suite"]
    assert (s["seed0"], s["n"], s["n_distractors"], s["noise_lines"]) == (2000, 200, 120, 40)
    check_dev_pool(s["seed0"], s["n"], "stress")
