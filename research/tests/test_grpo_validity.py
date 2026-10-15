"""GRPO validity gate (PREREGISTRATION A7.10 (c)): for every decision point mode that in-harness GRPO trains
(`training/grpo_online.py`), the sampled action space is a stated stochastic rule, the stored behaviour log-probability
of the realised event equals the one replayed through the update path, the behaviour distribution is normalised, and
the KL term uses the same option distributions. Sampled action spaces (behaviour policy pi_T = softmax(z / T) over the
scored options; pi_1 at T = 1):

  replace   categorical a ~ pi_T                                                  log P = log pi_T(a)
  cascade   a ~ pi_T; commit a iff pi_1(a) >= tau (and a != OTHER), else defer;   log P(commit) = log pi_T(a)
            the deferral is the event D = {a' : a' fails the gate}                log P(defer) = log sum_{D} pi_T
  assist    ordered top-k by Gumbel-top-k = Plackett-Luce sampling without replacement from pi_T
                                                                                  log P = sum_i log pi_T(s_i) - log(1 - sum_{j<i} pi_T(s_j))
  gates     yes / no ~ pi_T over the Noul's two options; the harness gate follows the sample
                                                                                  log P = log pi_T(sample)

Knockout rounds (> 52 options) are deterministic top-2 cuts and receive no gradient: the trained event is the final
round's categorical choice given the surviving pool. Greedy (evaluation) decisions and assist decisions that keep every
option carry no event and are never trained.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1-layer Qwen3 causal LM + byte-level BPE tokenizer (every option code A-Z/a-z is one token)."""
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
    with torch.no_grad():  # spread the option-code logits so the sampled distributions are far from uniform
        m.lm_head.weight.mul_(40.0) if not cfg.tie_word_embeddings else m.model.embed_tokens.weight.mul_(6.0)
    d = tmp_path_factory.mktemp("tiny_validity")
    m.save_pretrained(d)
    ft.save_pretrained(d)
    return d


def _load(tiny: Path):  # noqa: ANN202
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny)
    tok.padding_side = "left"
    return AutoModelForCausalLM.from_pretrained(tiny).float().eval(), tok


def _replay(net, tok, rec, temperature: float):  # noqa: ANN001, ANN202
    """The update path: the record's stored prompt through the training collator and option_logits."""
    from jevharness.training.decision_model import DecisionCollator, option_logits

    col = DecisionCollator(tok, max_len=512)
    enc = col([{"prompt": rec.prompt, "target": [0.0] * rec.n}])
    with torch.no_grad():
        z = option_logits(net, enc["input_ids"], enc["attention_mask"], enc["code_ids"])[0, :rec.n].double()
    return z, torch.log_softmax(z / temperature, -1)


@pytest.mark.parametrize("mode", ["replace", "cascade", "assist", "guard"])
def test_sampled_events_replay_normalise_and_follow_the_stated_rule(tiny: Path, mode: str):
    from jevharness.decision import Choice, Noul
    from jevharness.harness.config import HarnessConfig
    from jevharness.training.grpo_online import TorchDecider, event_logp

    net, tok = _load(tiny)
    T = 0.7
    dpc = {"replace": {"mode": "replace"}, "cascade": {"mode": "cascade", "tau": 0.3},
           "assist": {"mode": "assist", "k": 2}, "guard": {"mode": "guard", "tau": 0.5}}[mode]
    arm = HarnessConfig(name=mode, tool_select=dpc)
    d = TorchDecider(net, tok, arm, temperature=T, max_len=512)
    q = (Noul("is the proposed call right?", decision_point="tool_select") if mode == "guard"
         else Choice("which tool next?", ["alpha", "beta", "gamma"], decision_point="tool_select", gold="beta"))
    counts: dict = {}
    recs = []
    n = 1200
    for s in range(n):
        d.begin_episode(s, arm)
        d.decide("STATE: s", [q])
        (r,) = d.end_episode()
        recs.append(r)
        counts[(r.ev, tuple(r.event))] = counts.get((r.ev, tuple(r.event)), 0) + 1
    z, lpT = _replay(net, tok, recs[0], T)
    assert abs(float(lpT.exp().sum()) - 1.0) < 1e-9                                     # normalised
    for r in recs[:100]:
        assert r.prompt == recs[0].prompt
        if r.train:
            assert abs(float(event_logp(lpT, r.event, r.ev)) - r.old_logp) < 1e-4       # stored == replayed
    p1, pT = torch.softmax(z, -1), lpT.exp()
    if mode in ("replace", "guard"):
        ev = "gate" if mode == "guard" else "choice"
        assert all(r.train and r.ev == ev for r in recs)
        for i in range(len(pT)):
            assert abs(counts.get((ev, (i,)), 0) / n - float(pT[i])) < 0.05
    elif mode == "cascade":
        below = tuple(i for i in range(3) if float(p1[i]) < 0.3)
        p_defer = float(pT[list(below)].sum()) if below else 0.0
        assert abs(sum(c for (ev, _), c in counts.items() if ev == "defer") / n - p_defer) < 0.05
        for (ev, e) in counts:
            assert (ev == "defer" and e == below) or (ev == "commit" and len(e) == 1 and float(p1[e[0]]) >= 0.3)
        assert all(r.train == (r.ev == "commit" or 0 < len(r.event) < r.n) for r in recs)
    else:  # Plackett-Luce over ordered pairs: normalised, and the Gumbel-top-k sampler's frequencies match it
        tot = 0.0
        for e in itertools.permutations(range(3), 2):
            pe = float(event_logp(lpT, list(e), "subset").exp())
            tot += pe
            assert abs(counts.get(("subset", e), 0) / n - pe) < 0.05
        assert abs(tot - 1.0) < 1e-9


def test_kl_term_uses_the_same_option_distributions(tiny: Path):
    from peft import LoraConfig, get_peft_model

    from jevharness.decision import Choice
    from jevharness.harness.config import HarnessConfig
    from jevharness.training.decision_model import DecisionCollator, option_logits
    from jevharness.training.grpo_online import TorchDecider, fill_reference, grpo_decision_loss

    base, tok = _load(tiny)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(0.2 * torch.randn_like(p))   # the policy differs from the reference (adapter disabled)
    model.eval()
    arm = HarnessConfig(name="r", tool_select={"mode": "replace"})
    d = TorchDecider(model, tok, arm, temperature=1.0, max_len=512)
    d.begin_episode(0, arm)
    d.decide("STATE: s", [Choice("which?", ["a", "b", "c", "d"], decision_point="tool_select")])
    (r,) = d.end_episode()
    col = DecisionCollator(tok, max_len=512)
    fill_reference(model, col, [r], "base", 4)
    enc = col([{"prompt": r.prompt, "target": [0.0] * r.n}])
    model.eval()
    with torch.no_grad():
        z = option_logits(model, enc["input_ids"], enc["attention_mask"], enc["code_ids"])
        _, st = grpo_decision_loss(z, [r.event], [r.ev], torch.tensor([r.old_logp]), torch.tensor([1.0]),
                                   torch.tensor([r.ref]), temperature=1.0, kl_beta=0.1)
        with model.disable_adapter():
            zr = option_logits(model, enc["input_ids"], enc["attention_mask"], enc["code_ids"])
    p = torch.softmax(z[0, :r.n].double(), -1)
    pr = torch.softmax(zr[0, :r.n].double(), -1)
    assert abs(float(torch.tensor(r.ref).double().exp().sum()) - 1.0) < 1e-5            # reference normalised
    manual = float((p * (p.log() - pr.log())).sum())
    assert manual > 1e-6 and abs(float(st["kl"][0]) - manual) < 1e-4                    # same distributions
    assert abs(float(st["ratio"][0]) - 1.0) < 1e-4                                      # behaviour == policy at start


def test_ratio_after_update_uses_current_policy_against_stored_behaviour(tiny: Path):
    """After the policy moves, the PPO ratio is exp(log pi_current(event) - stored behaviour log-prob), computed on the
    same masked option set (mixed option counts in one micro-batch: padded codes are -inf and get no mass)."""
    from peft import LoraConfig, get_peft_model

    from jevharness.decision import Choice
    from jevharness.harness.config import HarnessConfig
    from jevharness.training.decision_model import DecisionCollator, option_logits
    from jevharness.training.grpo_online import TorchDecider, event_logp, grpo_decision_loss

    base, tok = _load(tiny)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.eval()
    arm = HarnessConfig(name="c", tool_select={"mode": "cascade", "tau": 0.3})
    d = TorchDecider(model, tok, arm, temperature=0.8, max_len=512)
    recs = []
    for s, opts in enumerate((["a", "b", "c"], ["a", "b", "c", "d", "e"])):
        d.begin_episode(s, arm)
        d.decide(f"STATE: {s}", [Choice("which?", opts, decision_point="tool_select")])
        recs += d.end_episode()
    torch.manual_seed(2)
    with torch.no_grad():  # an "update": the current policy now differs from the behaviour snapshot
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(0.3 * torch.randn_like(p))
    col = DecisionCollator(tok, max_len=512)
    enc = col([{"prompt": r.prompt, "target": [0.0] * r.n} for r in recs])
    with torch.no_grad():
        z = option_logits(model, enc["input_ids"], enc["attention_mask"], enc["code_ids"])
        _, st = grpo_decision_loss(z, [r.event for r in recs], [r.ev for r in recs],
                                   torch.tensor([r.old_logp for r in recs]), torch.ones(len(recs)),
                                   None, temperature=0.8)
    assert torch.isinf(z[0, 3:]).all() and torch.isfinite(z[0, :3]).all()          # masking of padded codes
    for i, r in enumerate(recs):
        lp = torch.log_softmax(z[i, :r.n].double() / 0.8, -1)
        assert abs(float(lp.exp().sum()) - 1.0) < 1e-9
        if not r.train:
            continue
        want = float(torch.exp(event_logp(lp, r.event, r.ev) - r.old_logp))
        assert abs(float(st["ratio"][i]) - want) < 1e-3 * max(1.0, want)
        assert abs(want - 1.0) > 1e-5                                                 # the policy really moved


def test_knockout_trains_only_the_final_round_choice(tiny: Path):
    """> 52 options: deterministic top-2 knockout rounds (no probability, no gradient); the stored event and its
    behaviour log-prob refer to the final round's categorical choice over the surviving pool, and replay through the
    update path reproduces it."""
    from jevharness.decision import Choice
    from jevharness.harness.config import HarnessConfig
    from jevharness.training.grpo_online import MAX_PASS, TorchDecider, event_logp

    net, tok = _load(tiny)
    arm = HarnessConfig(name="r", tool_select={"mode": "replace"})
    d = TorchDecider(net, tok, arm, temperature=1.0, max_len=2048)
    opts = [f"tool_{i}" for i in range(60)]
    d.begin_episode(0, arm)
    ans = d.decide("STATE: s", [Choice("which tool?", opts, decision_point="tool_select")]).answers[0]
    (r,) = d.end_episode()
    assert r.n <= MAX_PASS < len(opts) and r.n == 4          # 52 + 8 -> top-2 + top-2 -> final pool of 4
    assert r.ev == "choice" and r.train and 0 <= r.event[0] < r.n
    assert sum(1 for v in ans.distribution.values() if v > 0) <= r.n   # eliminated options get probability 0
    _, lp = _replay(net, tok, r, 1.0)
    assert abs(float(event_logp(lp, r.event, r.ev)) - r.old_logp) < 1e-4
