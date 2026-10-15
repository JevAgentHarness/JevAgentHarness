"""CPU tests for the RL / teacher-labelling part of the training program (tiny local Qwen3-style model, simulated
System-2 executor, simulated / oracle teachers). No network, no GPU; ~1 min in total."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
SIM_EXECUTOR = {"provider": "simulated", "model": "scripted", "error_rate": 0.3, "simulated_latency_ms": 0}


# ------------------------------------------------------------------ fixtures
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
    d = tmp_path_factory.mktemp("tiny")
    Qwen3ForCausalLM(cfg).save_pretrained(d)
    ft.save_pretrained(d)
    return d


def _examples(n: int = 16) -> list[dict]:
    out = []
    for i in range(n):
        opts = ["alpha", "beta", "gamma"] + (["delta"] if i % 2 else [])
        q = {o: (0.9 if o == "beta" else 0.1) for o in opts}
        out.append({"id": f"t{i}/0/x", "source": "test", "dp": "tool_select", "kind": "choice", "state": f"S{i}",
                    "question": "Q?", "options": opts, "target": {o: float(o == "beta") for o in opts},
                    "q_values": q, "meta": {}})
    return out


def _lora_b_abs_max(path: Path) -> float:
    from safetensors.torch import load_file

    w = load_file(str(path / "adapter_model.safetensors"))
    return max(float(t.abs().max()) for k, t in w.items() if "lora_B" in k)


# ------------------------------------------------------------------ grpo_online: pure functions
def test_event_logp_subset_is_plackett_luce():
    from itertools import permutations

    from jevharness.training.grpo_online import event_logp

    logp = torch.log_softmax(torch.tensor([0.3, -1.0, 2.0, 0.5]), -1)
    total = sum(math.exp(float(event_logp(logp, list(s), "subset"))) for s in permutations(range(4), 2))
    assert total == pytest.approx(1.0, abs=1e-6)  # ordered 2-subsets form a distribution
    assert float(event_logp(logp, [0, 2], "defer")) == pytest.approx(math.log(logp.exp()[[0, 2]].sum()), abs=1e-6)


def test_advantages_groups_zero_std_errors_and_gigpo():
    from jevharness.training.grpo_online import AlgoCfg, DecisionRecord, Episode, compute_advantages

    def rec(key: str, step: int) -> DecisionRecord:
        return DecisionRecord(prompt=key, n=2, event=[0], ev="choice", old_logp=-0.7, dp="tool_select", step=step,
                              key=key, sampled=0, greedy=0, gold=None, conf=0.5, entropy=0.69)

    def ep(group: int, reward: float, n_steps: int, error: str | None = None) -> Episode:
        return Episode(task_id=f"t{group}", group=group, rollout=0, arm="a", reward=reward, success=reward > 0,
                       cost_usd=0.0, llm_calls=0, s1_calls=1, n_steps=n_steps, irreversible=False, error=error,
                       records=[rec("s0", 0), rec(f"x{group}{reward}{n_steps}", 1)])

    eps = [ep(0, 1.0, 2), ep(0, 0.0, 2), ep(1, 1.0, 2), ep(1, 1.0, 5), ep(2, 1.0, 2, error="boom"), ep(2, 0.0, 2)]
    st = compute_advantages(eps, AlgoCfg(gigpo={"enabled": False}))
    assert eps[0].adv > 0 > eps[1].adv and eps[0].adv == pytest.approx(-eps[1].adv)
    assert eps[2].adv == eps[3].adv == 0.0 and not any(r.train for r in eps[2].records + eps[3].records)
    assert st["error_episodes"] == 1 and not any(r.train for r in eps[4].records)
    # GiGPO: group 1 has identical episode rewards but different step returns at the shared anchor "s0"
    eps = [ep(0, 1.0, 2), ep(0, 0.0, 2), ep(1, 1.0, 2), ep(1, 1.0, 5)]
    compute_advantages(eps, AlgoCfg(gigpo={"enabled": True, "gamma": 0.5}))
    s0_fast, s0_slow = eps[2].records[0], eps[3].records[0]
    assert s0_fast.train and s0_slow.train and s0_fast.adv > 0 > s0_slow.adv  # reaching success sooner is better
    assert not eps[2].records[1].train  # unique state, zero-std episode group -> no signal -> dropped


def test_rewards_linear_vs_gated():
    from types import SimpleNamespace

    from jevharness.training.grpo_online import RewardCfg, episode_reward

    def traj(success: bool, calls: int, irr: bool = False) -> SimpleNamespace:
        return SimpleNamespace(success=success, usage={"total": {"cost_usd": 0.01 * calls}, "s2": {"calls": calls}},
                               result={"irreversible_error": irr})

    lin = RewardCfg(kind="linear", lambda_calls=0.1)
    assert episode_reward(traj(False, 0), lin) > episode_reward(traj(False, 3), lin)  # linear rewards cheap failure
    gated = RewardCfg(kind="gated", lambda_gate=0.5, cost_scale=10)
    assert episode_reward(traj(False, 0), gated) == episode_reward(traj(False, 9), gated) == 0.0  # ... gated does not
    assert episode_reward(traj(True, 2), gated) > episode_reward(traj(True, 8), gated) > 0
    assert episode_reward(traj(False, 1, irr=True), RewardCfg(irreversible_penalty=0.2)) == pytest.approx(-0.2)


def test_grpo_loss_gradient_direction():
    from jevharness.training.grpo_online import grpo_decision_loss

    z = torch.zeros(2, 3, requires_grad=True)
    old = torch.log_softmax(torch.zeros(2, 3), -1)[:, 0]
    loss, st = grpo_decision_loss(z, [[0], [0]], ["choice", "choice"], old, torch.tensor([1.0, -1.0]),
                                  torch.log_softmax(torch.zeros(2, 3), -1), kl_beta=0.1)
    loss.sum().backward()
    assert z.grad[0, 0] < 0 < z.grad[1, 0]  # positive advantage raises the logit, negative lowers it
    assert float(st["kl"].abs().max()) < 1e-6 and float(st["ratio"].mean()) == pytest.approx(1.0)


def test_config_dotted_overrides_merge_into_defaults():
    from jevharness.training.grpo_online import load_config

    cfg = load_config(None, ["suite.n=4", "grpo.gigpo.enabled=true", "reward.kind=linear"], model="m")
    assert cfg.suite["name"] == "jevgym" and cfg.suite["n"] == 4 and cfg.grpo.gigpo.enabled
    y = load_config(ROOT / "configs/train/grpo_online_jevgym.yaml", ["executor.provider=simulated"])
    assert y.executor["provider"] == "simulated" and y.suite["strict"] and len(y.harness_arms()) == 3
    assert y.suite["seed0"] >= 200 and y.eval_suite["seed0"] >= 200  # disjoint from the E2 evaluation seeds


def test_question_to_example_matches_serve_jev():
    from jevharness.decision import Choice, Noul, Score
    from jevharness.decision.jev import question_to_wire
    from jevharness.training.grpo_online import question_to_example

    class Spy:
        def __init__(self) -> None:
            self.items: list = []

        def choice_probs(self, state, question, options, kind):  # noqa: ANN001, ANN201
            self.items.append((state, question, options, kind))
            return [1.0 / len(options)] * len(options)

        def _probs(self, items):  # noqa: ANN001, ANN201
            self.items += items
            return [[1.0 / len(o)] * len(o) for _, _, o, _ in items]

    from jevharness.training.serve_jev import Engine

    spy = Spy()
    for q in [Choice("pick", ["a", "b"], ["first", None]), Noul("ok?"), Score("rate", scale=(1, 3))]:
        Engine.answer(spy, "S", question_to_wire(q))  # type: ignore[arg-type]
        ex = question_to_example("S", q)
        assert spy.items[-1] == ("S", ex["question"], ex["options"], ex["kind"])


# ------------------------------------------------------------------ grpo_online: end to end
def test_grpo_online_tiny_end_to_end(tiny: Path, tmp_path: Path):
    from jevharness.training.grpo_online import load_config, train

    out = tmp_path / "r8"
    sets = ["suite.n=2", "suite.n_distractors=3", "eval_suite.n=1", "eval_suite.n_distractors=3",
            "grpo.tasks_per_iter=2", "grpo.group_size=2", "grpo.minibatch_size=16", "grpo.micro_batch_size=8",
            "grpo.gigpo.enabled=true", "lora.r=4", "lora.alpha=8", "decider.max_len=512", "workers=1",
            "eval_every=1", "save_every=1", "log_rollouts=true", "optim.lr=0.01", "reward.kind=linear",
            "reward.lambda_calls=0.1", "harness_defaults={max_steps: 6}",
            "arms=[{name: gates, tool_select: {mode: replace}, reflection: {mode: trigger, tau: 0.5}, "
            "stop: {mode: trigger, tau: 0.5}}]"]
    cfg = load_config(None, sets, model=str(tiny), out=str(out), executor=SIM_EXECUTOR, **{"grpo.iterations": 2})
    train(cfg)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    it_rows = [r for r in rows if r["iter"] > 0]
    assert len(it_rows) == 2 and "eval/gates/success" in rows[0]
    assert all(r["episodes"] == 4 and r["decisions"] > 0 for r in it_rows)
    upd = [r for r in it_rows if r.get("updated_decisions", 0) > 0]
    assert upd, "no iteration produced a learning signal"
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["kl"]) and r["optimizer_steps"] > 0 for r in upd)
    assert _lora_b_abs_max(out) > 0  # LoRA B starts at 0: non-zero means the policy moved
    assert (out / "ckpt-00001" / "trainer_state.pt").exists() and (out / "rollouts.jsonl").exists()
    assert json.loads((out / "decision_config.json").read_text())["loss"] == "grpo_online"
    # resume from the iteration-1 checkpoint and run one more iteration
    cfg3 = load_config(None, sets, model=str(tiny), out=str(tmp_path / "r8b"), executor=SIM_EXECUTOR,
                       **{"grpo.iterations": 2, "eval_every": 0})
    train(cfg3, resume=str(out / "ckpt-00001"))
    rows_b = [json.loads(line) for line in (tmp_path / "r8b" / "metrics.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in rows_b] == [2]


def test_torch_decider_cascade_events(tiny: Path):
    """Cascade semantics: the realised event is 'commit a' (p(a) >= tau) or 'defer' = every sub-tau option."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jevharness.decision import Choice
    from jevharness.harness.config import HarnessConfig
    from jevharness.training.grpo_online import TorchDecider

    tok = AutoTokenizer.from_pretrained(tiny)
    tok.padding_side = "left"
    net = AutoModelForCausalLM.from_pretrained(tiny)
    for tau, expect in [(0.0, "commit"), (1.0, "defer")]:
        arm = HarnessConfig(name="c", tool_select={"mode": "cascade", "tau": tau})
        d = TorchDecider(net, tok, arm, max_len=256)
        d.begin_episode(0, arm)
        a = d.decide("S", [Choice("which?", ["x", "y", "z"], decision_point="tool_select", gold="y")]).answers[0]
        (r,) = d.end_episode()
        assert r.ev == expect and a.value in {"x", "y", "z"} and r.gold == 1
        assert (r.event == [r.sampled]) if expect == "commit" else (sorted(r.event) == [0, 1, 2] and not r.train)


# ------------------------------------------------------------------ grpo_trl
def test_grpo_trl_torch_impl_with_init(tiny: Path, tmp_path: Path):
    from jevharness.training.data import write_jsonl
    from jevharness.training.grpo_trl import main

    write_jsonl(_examples(), tmp_path / "d.jsonl")
    init = tmp_path / "init"
    main(["--impl", "torch", "--model", str(tiny), "--train", str(tmp_path / "d.jsonl"), "--out", str(init),
          "--lora-r", "4", "--lora-alpha", "8", "--num-generations", "4", "--batch-size", "8", "--grad-accum", "1",
          "--max-steps", "3", "--lr", "0.05", "--max-len", "256", "--rewards", "correct,soft", "--no-bf16",
          "--logging-steps", "1"])
    rows = [json.loads(line) for line in (init / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(math.isfinite(r["loss"]) for r in rows)
    assert _lora_b_abs_max(init) > 0
    out = tmp_path / "cont"  # continue from an adapter: the reference is the frozen init copy (KL starts at 0)
    res = main(["--impl", "torch", "--model", str(tiny), "--init", str(init), "--train", str(tmp_path / "d.jsonl"),
                "--dev", str(tmp_path / "d.jsonl"), "--out", str(out), "--num-generations", "4", "--batch-size", "8",
                "--grad-accum", "1", "--max-steps", "2", "--lr", "0.05", "--max-len", "256", "--rewards",
                "qvalue", "--no-bf16"])
    first = json.loads((out / "metrics.jsonl").read_text().splitlines()[0])
    assert first["kl"] == pytest.approx(0.0, abs=1e-6) and res["dev"]["n"] == 16
    assert not (out / "ref").exists()  # only the trained adapter is saved


def test_grpo_trl_trl_impl_smoke(tiny: Path, tmp_path: Path):
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    from jevharness.training.data import write_jsonl
    from jevharness.training.grpo_trl import main

    write_jsonl(_examples(8), tmp_path / "d.jsonl")
    out = tmp_path / "trl"
    main(["--impl", "trl", "--model", str(tiny), "--train", str(tmp_path / "d.jsonl"), "--out", str(out),
          "--lora-r", "4", "--lora-alpha", "8", "--num-generations", "2", "--batch-size", "4", "--grad-accum", "1",
          "--max-steps", "2", "--max-len", "256", "--rewards", "correct,format", "--no-bf16", "--logging-steps", "1"])
    hist = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert any("reward" in h for h in hist) and (out / "adapter_model.safetensors").exists()


def test_grpo_trl_reward_functions(tiny: Path):
    from transformers import AutoTokenizer

    from jevharness.training.decision_model import code_token_ids
    from jevharness.training.grpo_trl import make_reward_funcs

    tok = AutoTokenizer.from_pretrained(tiny)
    codes = code_token_ids(tok)
    correct, fmt, soft, qv = make_reward_funcs(tok, ["correct", "format", "soft", "qvalue"])
    comp = [[codes[1]], [codes[0]], [tok.eos_token_id], [codes[5]]]
    kw = {"n_options": [3, 3, 3, 3], "gold": [1, 1, 1, 1], "target": [[0.1, 0.8, 0.1]] * 4,
          "q": [[0.2, 0.9, float("nan")]] * 4}
    assert correct(completion_ids=comp, **kw) == [1.0, 0.0, 0.0, 0.0]
    assert fmt(completion_ids=comp, **kw) == [0.0, 0.0, -1.0, -1.0]  # code beyond n_options is a format error
    assert soft(completion_ids=comp, **kw) == pytest.approx([0.8, 0.1, 0.0, 0.0])
    assert qv(completion_ids=comp, **kw)[:2] == pytest.approx([0.9, 0.2])


def test_train_decision_dpo_uses_init_as_reference(tiny: Path, tmp_path: Path, capsys: pytest.CaptureFixture):
    from jevharness.training.data import write_jsonl
    from jevharness.training.train_decision import main as train_main

    write_jsonl(_examples(8), tmp_path / "d.jsonl")
    base_args = ["--model", str(tiny), "--train", str(tmp_path / "d.jsonl"), "--lora-r", "4", "--lora-alpha", "8",
                 "--batch-size", "4", "--grad-accum", "1", "--max-len", "256", "--no-bf16", "--no-permute"]
    train_main(base_args + ["--out", str(tmp_path / "sft"), "--lr", "0.05", "--epochs", "2"])
    capsys.readouterr()
    out = tmp_path / "dpo"
    train_main(base_args + ["--out", str(out), "--init", str(tmp_path / "sft"), "--loss", "dpo", "--lr", "1e-3",
                            "--max-steps", "2", "--min-q-gap", "0.5"])
    cfg = json.loads((out / "decision_config.json").read_text())
    assert cfg["ref"] == "init" and not (out / "ref").exists()
    import re

    losses = re.findall(r"'train_loss': '?([0-9.eE+-]+)", capsys.readouterr().out)
    # the policy starts equal to its frozen init reference, so DPO starts at -log sigmoid(0) = log 2
    # (with the base model as reference -- the old behaviour -- it would not, the SFT moved the policy)
    assert losses and float(losses[-1]) == pytest.approx(math.log(2), abs=0.02)


# ------------------------------------------------------------------ teacher labelling
def _mixed_kind_examples() -> list[dict]:
    ex = _examples(6)
    ex.append({"id": "n0/0/stop", "source": "t", "dp": "stop", "kind": "noul", "state": "S0", "question": "Done?",
               "options": ["yes", "no"], "target": {"yes": 0.0, "no": 1.0}, "q_values": None, "meta": {}})
    ex.append({"id": "s0/0/score", "source": "t", "dp": "reflection", "kind": "score", "state": "S1",
               "question": "Progress?", "options": ["none", "some", "done"],
               "target": {"none": 0.0, "some": 0.0, "done": 1.0}, "q_values": None, "meta": {}})
    return ex


def test_teacher_label_oracle_and_simulated(tmp_path: Path):
    from jevharness.training.data import read_jsonl, write_jsonl
    from jevharness.training.teacher_label import main

    write_jsonl(_mixed_kind_examples(), tmp_path / "in.jsonl")
    s = main(["--in", str(tmp_path / "in.jsonl"), "--out", str(tmp_path / "o.jsonl"), "--teacher", "oracle",
              "--teacher", 'sim={"type": "simulated", "accuracy": 0.5, "seed": 3}', "--alpha", "0.25",
              "--min-agree", "2", "--cache", str(tmp_path / "c.sqlite"), "--workers", "2"])
    out = read_jsonl(tmp_path / "o.jsonl")
    assert s["teachers"]["oracle"]["acc"] == 1.0 and s["teachers"]["oracle"]["answered"] == 8
    assert s["outcomes"].get("kept", 0) == len(out) and s["n_in"] == 8
    for e in out:
        assert math.isclose(sum(e["target"].values()), 1.0, rel_tol=1e-6)
        gold = e["meta"]["gold"]
        assert e["target"][gold] >= 0.25 and max(e["target"], key=e["target"].get) == gold
        assert e["meta"]["teacher_agree"] == 2
    assert s["teachers"]["oracle"]["ledger"]["calls"] > 0 and s["teachers"]["oracle"]["ledger"]["cost_usd"] == 0.0
    # second run: everything replayed from the SQLite cache
    s2 = main(["--in", str(tmp_path / "in.jsonl"), "--out", str(tmp_path / "o2.jsonl"), "--teacher", "oracle",
               "--cache", str(tmp_path / "c.sqlite")])
    assert s2["teachers"]["oracle"]["ledger"]["calls_fresh"] == 0 and s2["n_out"] == 8


def test_teacher_label_pooling_filters_and_mc(tmp_path: Path):
    from jevharness.training.teacher_label import label_example, pool

    assert pool([[0.9, 0.1], [0.5, 0.5]], [1, 1], "mean") == pytest.approx([0.7, 0.3])
    g = pool([[0.9, 0.1], [0.5, 0.5]], [1, 1], "geo")
    assert g[0] == pytest.approx(0.75, abs=1e-6)
    ex = _examples(1)[0]  # gold = beta (index 1), q_values favour beta
    kw = dict(alpha=0.5, min_agree=2, min_conf=0.0, how="mean", temperature=1.0, gold_filter="drop",
              on_disagree="drop")
    agree = {"a": [0.1, 0.8, 0.1], "b": [0.2, 0.6, 0.2]}
    lab, outcome = label_example(ex, agree, {}, **kw)
    assert outcome == "kept" and lab["target"]["beta"] == pytest.approx(0.5 + 0.5 * 0.7)
    _, outcome = label_example(ex, {"a": [0.8, 0.1, 0.1], "b": [0.1, 0.8, 0.1]}, {}, **kw)
    assert outcome == "drop_disagree"
    lab, outcome = label_example(ex, {"a": [0.8, 0.1, 0.1]}, {}, **{**kw, "min_agree": 1, "gold_filter": "gold"})
    assert outcome == "gold_only" and lab["target"]["beta"] == 1.0
    lab, outcome = label_example(ex, {"a": [0.8, 0.1, 0.1]}, {}, **{**kw, "min_agree": 1, "mc_beta": 0.1})
    assert outcome == "mc" and max(lab["target"], key=lab["target"].get) == "beta"  # Q overrides a wrong pi_ref


def test_teacher_label_never_shows_gold_to_real_backends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from jevharness.decision.simulated import SimulatedBackend
    from jevharness.training import teacher_label as tl

    seen: list = []

    class Recorder(SimulatedBackend):
        batched_questions = True

        def _decide(self, state, questions):  # noqa: ANN001, ANN202
            seen.extend(q.meta for q in questions)
            return super()._decide(state, questions)

    monkeypatch.setattr(tl, "make_teacher", lambda spec, cache: Recorder())
    out, s = tl.label_file(_mixed_kind_examples(), [("real", {"type": "jev", "provider": "openrouter"})],
                           min_agree=1, gold_filter="keep", workers=1)
    assert seen and all("gold" not in m for m in seen)
    assert s["teachers"]["real"]["answered"] == 8 and len(out) == 8


def test_teacher_label_local_teacher(tiny: Path, tmp_path: Path):
    from jevharness.training.data import read_jsonl, write_jsonl
    from jevharness.training.teacher_label import main

    write_jsonl(_examples(4), tmp_path / "in.jsonl")
    spec = json.dumps({"type": "local", "model": str(tiny), "max_len": 256, "batch_size": 2})
    s = main(["--in", str(tmp_path / "in.jsonl"), "--out", str(tmp_path / "o.jsonl"), "--teacher", f"t={spec}",
              "--alpha", "0.0", "--gold-filter", "keep", "--cache", ""])
    out = read_jsonl(tmp_path / "o.jsonl")
    assert len(out) == 4 and s["teachers"]["t"]["ledger"]["input_tokens"] > 0
    assert all(math.isclose(sum(e["target"].values()), 1.0, rel_tol=1e-5) for e in out)


# ------------------------------------------------------------------ data utilities, prompt, scripts
def test_harvest_includes_noul_gates(tmp_path: Path):
    from jevharness.training.data import harvest_trajectories

    traj = {"task_id": "t1", "config": "c", "result": {"success": True}, "steps": [{"step": 0, "decisions": [
        {"dp": "tool_select", "mode": "replace", "question": "next tool", "options": ["a", "b"], "gold": "b",
         "state": "S"},
        {"dp": "stop", "mode": "guard", "question": "Is the task complete?", "options": None, "gold": False,
         "state": "S"},
        {"dp": "reflection", "mode": "trigger", "question": "stuck?", "options": None, "gold": None, "state": "S"}]}]}
    (tmp_path / "r.jsonl").write_text(json.dumps(traj) + "\n")
    ex = harvest_trajectories([tmp_path / "r.jsonl"])
    assert [(e["kind"], e["dp"]) for e in ex] == [("choice", "tool_select"), ("noul", "stop")]
    assert ex[1]["options"] == ["yes", "no"] and ex[1]["target"] == {"yes": 0.0, "no": 1.0}


def test_data_mix_and_stats(tmp_path: Path):
    from jevharness.training.data import mix, stats, write_jsonl

    write_jsonl(_examples(10), tmp_path / "a.jsonl")
    write_jsonl(_examples(4), tmp_path / "b.jsonl")
    m = mix([(tmp_path / "a.jsonl", 0.5), (tmp_path / "b.jsonl", 2.5)], seed=1)
    assert len(m) == 5 + 10
    st = stats(m)
    assert st["n"] == 15 and st["with_q_values"] == 15 and st["by_options"]["3-5"] == 15


def test_chat_text_disables_thinking_from_template_not_path():
    from jevharness.training.decision_model import chat_text

    class Tok:
        name_or_path = "checkpoints/r4"  # adapter dir: no "qwen3" in the name
        chat_template = "{% if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}"

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):  # noqa: ANN001, ANN201
            return json.dumps(kw)

    assert json.loads(chat_text(Tok(), "p")) == {"enable_thinking": False}


def test_derive_eval_configs(tmp_path: Path):
    import importlib.util

    spec_ = importlib.util.spec_from_file_location("derive", ROOT / "scripts/train/_derive_eval_configs.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)  # type: ignore[union-attr]
    import yaml

    spec = {"type": "jev", "name": "ours", "base_url": "http://localhost:8081", "model": "ours", "api_key": ""}
    run = mod.derive_run(yaml.safe_load((ROOT / "configs/experiments/jevgym_tool_select.yaml").read_text()), spec,
                         "ours")
    assert run["s1"] == spec and run["s1_backends"] == {} and all("s1" not in a for a in run["arms"])
    assert any(a["name"] == "llm_only" for a in run["arms"]) and "s1_cache" not in run
    dec = mod.derive_decide(yaml.safe_load((ROOT / "configs/decision/bfcl_when2call.yaml").read_text()), spec, "ours")
    assert dec["backends"] == {"ours": spec} and "fc_llms" not in dec


SCRIPTS = sorted(p.name for p in (ROOT / "scripts/train").glob("[0-9]*.sh"))


def test_train_scripts_exist():
    assert SCRIPTS == ["00_harvest.sh", "10_sft.sh", "12_sft_v1.sh", "20_teacher_soft.sh", "30_mc_rollouts.sh",
                       "40_kl_to_mc.sh", "50_dpo.sh", "55_pg_exact.sh", "60_grpo_trl.sh", "65_opd.sh", "70_grpo_online.sh",
                       "80_distill_small.sh", "90_calibrate_serve_eval.sh"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_train_scripts_syntax_and_dry_run(script: str):
    path = ROOT / "scripts/train" / script
    text = path.read_text()
    assert "set -euo pipefail" in (ROOT / "scripts/train/_common.sh").read_text() and "_common.sh" in text
    subprocess.run(["bash", "-n", str(path)], check=True)
    env = {**os.environ, "DRY_RUN": "1", "GPUS": "8"}
    r = subprocess.run(["bash", str(path)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    cmds = [line for line in r.stdout.splitlines() if line.startswith("+ ")]
    assert cmds and any("jevharness" in c for c in cmds)
    if script in ("10_sft.sh", "50_dpo.sh", "55_pg_exact.sh", "60_grpo_trl.sh"):
        assert any(c.startswith("+ accelerate launch") and "--num_processes 8" in c for c in cmds)


def test_seed_determinism_helper():
    from jevharness.training.grpo_online import _seed

    assert _seed(0, 1, 2) == _seed(0, 1, 2) != _seed(0, 1, 3)
    assert 0 <= random.Random(_seed("x")).random() < 1


def test_harvest_seed_guard_excludes_evaluation_range():
    from jevharness.training.harvest import seed_allowed, seed_ranges
    excl = seed_ranges("100000-100499")
    assert seed_ranges("") == [] and seed_ranges("5000-5999,7") == [(5000, 5999), (7, 7)]
    assert seed_allowed(None, 1000, excl)                 # non-JevGym rows are never dropped by the guard
    assert not seed_allowed(999, 1000, excl) and seed_allowed(1000, 1000, excl)
    assert seed_allowed(99999, 1000, excl) and not seed_allowed(100000, 1000, excl)
    assert not seed_allowed(100499, 1000, excl) and seed_allowed(100500, 1000, excl)


def test_token_budget_batches_and_sft_with_budget(tiny: Path, tmp_path: Path, capsys: pytest.CaptureFixture):
    from jevharness.training.data import write_jsonl
    from jevharness.training.decision_model import token_budget_batches
    from jevharness.training.train_decision import TokenBudgetBatchSampler
    from jevharness.training.train_decision import main as train_main

    lengths = [10, 300, 50, 120, 400, 400, 30, 900]
    bs = token_budget_batches(lengths, 800)
    assert sorted(i for b in bs for i in b) == list(range(len(lengths)))
    assert all(len(b) * max(lengths[i] for i in b) <= 800 or len(b) == 1 for b in bs)
    s = TokenBudgetBatchSampler(lengths, 800, seed=1)
    e0 = [sorted(b) for b in s]
    e1 = [sorted(b) for b in s]
    assert sorted(i for b in e0 for i in b) == list(range(8)) and e0 != e1 or len(e0) == 1
    write_jsonl(_examples(12), tmp_path / "d.jsonl")
    train_main(["--model", str(tiny), "--train", str(tmp_path / "d.jsonl"), "--lora-r", "4", "--lora-alpha", "8",
                "--max-len", "256", "--no-bf16", "--out", str(tmp_path / "o"), "--lr", "0.05", "--epochs", "1",
                "--max-batch-tokens", "200", "--eff-batch", "4", "--no-gradient-checkpointing", "--group-by-length"])
    out = capsys.readouterr().out
    assert "token budget 200" in out and (tmp_path / "o" / "adapter_config.json").exists()


def test_grpo_round5_config_runs_with_simulated_executors(tiny: Path, tmp_path: Path):
    """configs/train/grpo_online_round5.yaml: strict seeds, named per-arm executor, eval-arm subset, token budget."""
    from jevharness.training.grpo_online import load_config, train

    sim = "{provider: simulated, model: scripted, error_rate: 0.1, simulated_latency_ms: 0}"
    sets = ["init=null", "ref=base", f"executor={sim}", f"executors={{flash: {sim}}}", "suite.n=4",
            "suite.n_distractors=3", "suite.families=all", "eval_suite.n=1", "eval_suite.n_distractors=3",
            "eval_suite.families=all", "grpo.iterations=1", "grpo.tasks_per_iter=4", "grpo.group_size=2",
            "grpo.max_batch_tokens=300", "lora.r=4", "lora.alpha=8", "decider.max_len=512", "workers=2",
            "eval_every=1", "harness_defaults={max_steps: 5}", "optim.lr=0.01", "bf16=false"]
    cfg = load_config(ROOT / "configs/train/grpo_online_round5.yaml", sets, model=str(tiny), out=str(tmp_path / "g"))
    assert cfg.seed_policy == "strict" and cfg.suite["seed0"] >= 10000 and cfg.eval_suite["seed0"] == 1000
    cfg.check_seeds()
    train(cfg)
    rows = [json.loads(line) for line in (tmp_path / "g" / "metrics.jsonl").read_text().splitlines()]
    assert "eval/rl_ts_params_replace/success" in rows[0] and not any("eval/rl_cascade/" in k for k in rows[0])
    assert any(k.startswith("arm/rl_guard_flash/") for k in rows[1])
