"""Decision-level GRPO (recipe R6b): RL with verifiable rewards on single typed decisions.

Input: decision examples in the ``training/data.py`` format (gold one-hot, teacher-soft or MC targets).
Two implementations of the same objective:

  --impl trl    TRL ``GRPOTrainer`` at the token level. Each prompt is the chat-templated decision prompt
                ending at the answer position (``data.render`` + ``decision_model.chat_text``: byte-identical
                to what ``serve_jev.py`` scores), options in a random order per row. G completions of
                ``--max-completion-length`` (1-2) tokens are sampled from the FULL vocabulary; the first
                completion token is mapped to an option through the option-code token ids. Rewards
                (summed with ``--reward-weights``):
                  correct   1 if the chosen option is the argmax of the target, else 0
                  format    0 if the first token is a valid option code (and, with --strict-eos, is followed
                            by EOS), else -1   (correct + format: right 1 / wrong 0 / garbage -1)
                  soft      target probability of the chosen option (teacher-soft / MC targets p*)
                  qvalue    MC success estimate Q(s, a) of the chosen option (examples with q_values)
                LoRA via ``peft_config``; reference = adapter disabled (fresh LoRA) or, with ``--init``, a
                frozen copy of the init adapter (TRL's "ref" adapter).
  --impl torch  the same sampled-group GRPO directly on the restricted option softmax (no format failures,
                one prefill per prompt instead of G generations); PPO-clip + exact KL shared with
                ``grpo_online.py``. Use it when TRL is unavailable or to ablate "free generation vs
                restricted readout".

Verified against TRL 1.14.1 + transformers 5.18 + peft 0.21 (``GRPOConfig``: num_generations,
max_completion_length, temperature, beta, loss_type, scale_rewards, epsilon, epsilon_high,
num_iterations, reward_weights; ``GRPOTrainer(model, reward_funcs, args, train_dataset,
processing_class, peft_config)``). Prompts are passed pre-rendered (standard, non-conversational
format), so TRL tokenizes them without re-applying a chat template. In TRL, per_device_train_batch_size
counts completions: prompts per optimizer step = batch * grad_accum * processes / num_generations.
CPU-tested end to end with a tiny Qwen3 model (``tests/test_training_rl.py``), both implementations.

Choosing the implementation. ``trl`` needs an init that already emits option codes (an SFT adapter): with a
cold model most samples are non-code tokens, every group has format = -1 and zero reward variance, so no update
happens (watch ``frac_reward_zero_std`` / ``rewards/format/mean``). ``torch`` never has that problem, needs no
generation, and is the fallback when TRL is missing or its API changes (it uses only torch + peft); it is single
process (run seeds / ablations one per GPU). Both use a frozen copy of ``--init`` as the KL reference.

  accelerate launch -m jevharness.training.grpo_trl --model Qwen/Qwen3-4B --init checkpoints/r3 \\
      --train data/decisions/train.jsonl --dev data/decisions/dev.jsonl --rewards correct,format \\
      --num-generations 8 --beta 0.02 --lr 1e-5 --out checkpoints/r6b
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import torch

from .data import SYSTEM, permuted, read_jsonl, render  # noqa: F401  (SYSTEM re-exported for docs)

REWARDS = ("correct", "format", "soft", "qvalue")


# ------------------------------------------------------------------ data
def fit_prompt(tok: Any, ex: dict[str, Any], order: list[int], max_len: int) -> tuple[str, str]:
    """(chat text, user prompt) for ``ex`` with options in ``order``, trimming the middle of the state
    until the chat text fits in ``max_len`` tokens (keeps the task head and the latest observation)."""
    from .decision_model import chat_text

    def build(state: str) -> tuple[str, str]:
        user, _ = render({**ex, "state": state}, order)
        return chat_text(tok, user), user

    text, user = build(ex["state"])
    if len(tok(text, add_special_tokens=False).input_ids) <= max_len:
        return text, user
    state = ex["state"]
    lo, hi = 0, len(state)
    best = build("[...]")
    while lo <= hi:  # largest kept-character budget that fits
        mid = (lo + hi) // 2
        cut = state[: mid // 2] + "\n...[truncated]...\n" + state[len(state) - (mid - mid // 2):]
        cand = build(cut)
        if len(tok(cand[0], add_special_tokens=False).input_ids) <= max_len:
            best, lo = cand, mid + 1
        else:
            hi = mid - 1
    return best


def build_rows(examples: list[dict[str, Any]], tok: Any, n_permutations: int = 1, seed: int = 0,
               max_len: int = 4096, permute: bool = True) -> list[dict[str, Any]]:
    """Decision examples -> GRPO rows (one per example and permutation)."""
    rng = random.Random(seed)
    rows = []
    for ex in examples:
        n = len(ex["options"])
        if not 2 <= n <= 52:
            continue
        for _ in range(max(1, n_permutations)):
            _, tgt, idx = permuted(ex, rng if permute else None)
            if max(tgt) <= 0:
                continue
            text, user = fit_prompt(tok, ex, idx, max_len)
            qv = ex.get("q_values") or {}
            q = [float(qv[ex["options"][j]]) if qv.get(ex["options"][j]) is not None else float("nan")
                 for j in idx] if qv else []
            rows.append({"prompt": text, "user_prompt": user, "n_options": n,
                         "gold": int(max(range(n), key=lambda k: tgt[k])), "target": tgt, "q": q,
                         "id": str(ex.get("id", len(rows)))})
    return rows


# ------------------------------------------------------------------ rewards
def make_reward_funcs(tok: Any, names: list[str], strict_eos: bool = False) -> list[Callable[..., list[Any]]]:
    """TRL reward functions: ``f(prompts, completions, completion_ids, **dataset_columns) -> list[float]``.
    The chosen option is read from the FIRST completion token id (no string parsing)."""
    from .decision_model import code_token_ids

    pos = {t: i for i, t in enumerate(code_token_ids(tok))}
    eos = tok.eos_token_id

    def pick(ids: list[int], n: int) -> int | None:
        if not ids:
            return None
        i = pos.get(int(ids[0]))
        if i is None or i >= n:
            return None
        if strict_eos and len(ids) > 1 and int(ids[1]) != eos:
            return None
        return i

    def correct(completion_ids: list[list[int]], n_options: list[int], gold: list[int], **_: Any) -> list[float]:
        return [1.0 if (k := pick(c, n)) is not None and k == g else 0.0
                for c, n, g in zip(completion_ids, n_options, gold)]

    def format_(completion_ids: list[list[int]], n_options: list[int], **_: Any) -> list[float]:
        return [0.0 if pick(c, n) is not None else -1.0 for c, n in zip(completion_ids, n_options)]

    def soft(completion_ids: list[list[int]], n_options: list[int], target: list[list[float]],
             **_: Any) -> list[float]:
        return [float(t[k]) if (k := pick(c, n)) is not None else 0.0
                for c, n, t in zip(completion_ids, n_options, target)]

    def qvalue(completion_ids: list[list[int]], n_options: list[int], q: list[list[float]],
               **_: Any) -> list[float | None]:
        out: list[float | None] = []
        for c, n, qq in zip(completion_ids, n_options, q):
            k = pick(c, n)
            if k is None:
                out.append(0.0)
            elif not qq or math.isnan(qq[k]):
                out.append(None)  # no MC estimate for this option: TRL ignores the value
            else:
                out.append(float(qq[k]))
        return out

    table = {"correct": correct, "format": format_, "soft": soft, "qvalue": qvalue}
    funcs = []
    for n in names:
        if n not in table:
            raise ValueError(f"unknown reward {n!r}; choose from {REWARDS}")
        f = table[n]
        f.__name__ = n
        funcs.append(f)
    return funcs


def option_reward(names: list[str], weights: list[float], row: dict[str, Any], k: int) -> float:
    """Restricted-readout reward of option k (``--impl torch``: every sample is a valid option)."""
    r = 0.0
    for n, w in zip(names, weights):
        if n == "correct":
            r += w * float(k == row["gold"])
        elif n == "soft":
            r += w * float(row["target"][k])
        elif n == "qvalue" and row["q"] and not math.isnan(row["q"][k]):
            r += w * float(row["q"][k])
    return r


# ------------------------------------------------------------------ evaluation (restricted readout)
@torch.no_grad()
def evaluate_decisions(model: Any, tok: Any, examples: list[dict[str, Any]], max_len: int = 4096,
                       batch_size: int = 16) -> dict[str, float]:
    """Accuracy / NLL / Brier / ECE of the option readout on held-out examples (fixed option order)."""
    from ..bench.metrics import brier, ece
    from .decision_model import DecisionCollator, option_logits

    col = DecisionCollator(tok, max_len=max_len)
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ex = [e for e in examples if 2 <= len(e["options"]) <= 52]
    acc, nll, conf, corr = [], [], [], []
    for s in range(0, len(ex), batch_size):
        chunk = ex[s:s + batch_size]
        items = []
        for e in chunk:
            prompt, tgt, _ = permuted(e, None)
            items.append({"prompt": prompt, "target": tgt})
        enc = col(items)
        z = option_logits(model, enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                          enc["code_ids"].to(dev)).float().cpu()
        lp = torch.log_softmax(z, -1)
        for i, it in enumerate(items):
            n = len(it["target"])
            g = max(range(n), key=lambda k: it["target"][k])
            pred = int(torch.argmax(lp[i, :n]))
            acc.append(float(pred == g))
            nll.append(-float(sum(t * lp[i, k] for k, t in enumerate(it["target"]) if t > 0)))
            conf.append(float(lp[i, pred].exp()))
            corr.append(float(pred == g))
    if was_training:
        model.train()
    n = max(1, len(acc))
    return {"n": len(acc), "acc": sum(acc) / n, "nll": sum(nll) / n, "brier_top1": brier(conf, corr),
            "ece": ece(conf, corr)}


# ------------------------------------------------------------------ model
def _load(a: argparse.Namespace, dtype: torch.dtype) -> tuple[Any, Any, Any]:
    """(model, tokenizer, peft_config): a PeftModel when --init is given, else base model + LoraConfig."""
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer

    from .train_decision import load_causal_lm

    tok = AutoTokenizer.from_pretrained(a.init or a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_causal_lm(a.model, dtype)
    peft_config = None
    if a.init:
        model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
    elif a.lora_r > 0:
        peft_config = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, target_modules="all-linear",
                                 task_type="CAUSAL_LM")
    return model, tok, peft_config


def _save(model: Any, tok: Any, a: argparse.Namespace, extra: dict[str, Any]) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "peft_config"):
        model.save_pretrained(str(out), selected_adapters=["default"])
    else:
        model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    (out / "decision_config.json").write_text(json.dumps(
        {"base_model": a.model, "loss": f"grpo_{a.impl}", "lora": hasattr(model, "peft_config"),
         "max_len": a.max_len, "init": a.init, "rewards": a.rewards, **extra}, indent=1))


# ------------------------------------------------------------------ TRL implementation
def train_trl(a: argparse.Namespace, rows: list[dict[str, Any]], names: list[str], weights: list[float],
              bf16: bool) -> tuple[Any, Any, Any]:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model, tok, peft_config = _load(a, torch.bfloat16 if bf16 else torch.float32)
    if a.lora_r <= 0 and not a.init:
        print("[grpo_trl] full fine-tuning: TRL loads a separate reference model when beta > 0")
    probe = rows[0]["prompt"]
    if tok(text=[probe])["input_ids"][0] != tok(probe, add_special_tokens=False).input_ids:
        print("[grpo_trl] WARNING: this tokenizer adds special tokens to raw text (e.g. a BOS), so TRL's prompt "
              "ids differ from the serve_jev/train_decision encoding")
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "n_options", "gold", "target", "q", "id")} for r in rows])
    args = GRPOConfig(
        output_dir=a.out, per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        num_generations=a.num_generations, max_completion_length=a.max_completion_length,
        temperature=a.temperature, top_p=1.0, top_k=0, beta=a.beta, loss_type=a.loss_type,
        scale_rewards=a.scale_rewards, epsilon=a.epsilon, epsilon_high=a.epsilon_high,
        num_iterations=a.num_iterations, reward_weights=weights, learning_rate=a.lr,
        lr_scheduler_type=a.lr_scheduler, weight_decay=0.0, max_grad_norm=1.0, num_train_epochs=a.epochs,
        max_steps=a.max_steps, logging_steps=a.logging_steps, save_strategy="steps" if a.save_steps else "no",
        save_steps=a.save_steps or 500, bf16=bf16, gradient_checkpointing=a.gradient_checkpointing,
        disable_dropout=True, report_to=a.report_to, seed=a.seed, dataloader_num_workers=0,
        use_vllm=a.use_vllm, vllm_mode=a.vllm_mode, log_completions=False)
    trainer = GRPOTrainer(model=model, reward_funcs=make_reward_funcs(tok, names, a.strict_eos), args=args,
                          train_dataset=ds, processing_class=tok, peft_config=peft_config)
    trainer.train()
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    if trainer.is_world_process_zero():
        _save(unwrapped, tok, a, {"trl_log": [h for h in trainer.state.log_history if "reward" in h][-1:]})
        with open(Path(a.out) / "metrics.jsonl", "w") as f:
            for h in trainer.state.log_history:
                f.write(json.dumps(h) + "\n")
    return unwrapped, tok, trainer


# ------------------------------------------------------------------ restricted-readout implementation
def train_torch(a: argparse.Namespace, rows: list[dict[str, Any]], names: list[str], weights: list[float],
                bf16: bool) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Sampled-group GRPO on the restricted option softmax (single process)."""
    from peft import get_peft_model

    from .decision_model import DecisionCollator, option_logits
    from .grpo_online import grpo_decision_loss

    if a.lora_r <= 0 and not a.init:
        raise SystemExit("--impl torch needs LoRA (reference = adapter disabled)")
    model, tok, peft_config = _load(a, torch.bfloat16 if bf16 else torch.float32)
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    ref_ctx: Callable[[], Any]
    from .decision_model import add_reference_adapter, reference_context

    ref_mode = "init" if a.init else "base"  # reference = frozen copy of the init adapter (same convention as TRL)
    if ref_mode == "init":
        add_reference_adapter(model, a.init)

    def ref_ctx() -> Any:
        return reference_context(model, ref_mode)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    col = DecisionCollator(tok, max_len=a.max_len)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    G = a.num_generations
    P = max(1, a.batch_size * a.grad_accum // G)  # prompts per optimizer step (same semantics as TRL)
    steps = a.max_steps if a.max_steps > 0 else math.ceil(a.epochs * len(rows) / P)
    rng = random.Random(a.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cur, log = 0, []
    micro = max(1, a.batch_size // G)  # prompts per forward/backward

    def logits(batch_rows: list[dict[str, Any]], grad: bool) -> torch.Tensor:
        enc = col([{"prompt": r["user_prompt"], "target": [0.0] * r["n_options"]} for r in batch_rows])
        with torch.set_grad_enabled(grad):
            return option_logits(model, enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                                 enc["code_ids"].to(dev))

    for step in range(1, steps + 1):
        batch = []
        for _ in range(P):
            if cur >= len(order):
                rng.shuffle(order)
                cur = 0
            batch.append(rows[order[cur]])
            cur += 1
        model.eval()
        z_old: list[torch.Tensor] = []   # per row: behaviour logits over its own options
        ref_lp: list[torch.Tensor] = []  # per row: reference log-probs over its own options
        with torch.no_grad():
            for s in range(0, P, micro):
                rs = batch[s:s + micro]
                z = logits(rs, False).float().cpu()
                z_old += [z[j, :r["n_options"]] for j, r in enumerate(rs)]
                with ref_ctx():
                    zr = logits(rs, False).float().cpu()
                ref_lp += [torch.log_softmax(zr[j, :r["n_options"]], -1) for j, r in enumerate(rs)]
        samples, old, rew = [], [], []
        for i, r in enumerate(batch):
            lpT = torch.log_softmax(z_old[i] / a.temperature, -1)
            acts = rng.choices(range(len(lpT)), weights=lpT.exp().tolist(), k=G)
            samples.append(acts)
            old.append([float(lpT[k]) for k in acts])
            rew.append([option_reward(names, weights, r, k) for k in acts])
        flat = [x for rr in rew for x in rr]
        bstd = torch.tensor(flat).std().item() if len(flat) > 1 else 0.0
        adv = []
        n_zero = 0
        for rr in rew:
            t = torch.tensor(rr)
            mu, sd = t.mean().item(), (t.std().item() if len(rr) > 1 else 0.0)
            n_zero += int(sd < 1e-8)
            scale = {"group": sd, "batch": bstd}.get(a.scale_rewards, None)
            adv.append([(x - mu) / (scale + 1e-4) if scale is not None else x - mu for x in rr])
        stats: dict[str, list[float]] = {"loss": [], "kl": [], "entropy": [], "clipped": []}
        gnorms: list[float] = []
        model.train()
        for _ in range(a.num_iterations):
            opt.zero_grad(set_to_none=True)
            for s in range(0, P, micro):
                rs = batch[s:s + micro]
                z = logits(rs, True)
                K = z.size(1)
                zz = z.repeat_interleave(G, 0)
                ev = [[samples[s + j][g]] for j in range(len(rs)) for g in range(G)]
                o = torch.tensor([old[s + j][g] for j in range(len(rs)) for g in range(G)], device=z.device)
                ad = torch.tensor([adv[s + j][g] for j in range(len(rs)) for g in range(G)], device=z.device)
                rf = torch.zeros((len(rs), K), device=z.device)  # padded options are masked inside the loss
                for j in range(len(rs)):
                    rf[j, :ref_lp[s + j].numel()] = ref_lp[s + j].to(z.device)
                rf = rf.repeat_interleave(G, 0)
                loss, st = grpo_decision_loss(zz, ev, ["choice"] * len(ev), o, ad, rf, temperature=a.temperature,
                                              clip_eps=a.epsilon, clip_eps_high=a.epsilon_high, kl_beta=a.beta)
                (loss.sum() / (P * G)).backward()
                stats["loss"] += loss.detach().float().cpu().tolist()
                for k in ("kl", "entropy", "clipped"):
                    stats[k] += st[k].float().cpu().tolist()
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
            opt.step()
        row = {"step": step, "reward": sum(flat) / len(flat),
               "acc_sampled": sum(float(k == r["gold"]) for r, acts in zip(batch, samples) for k in acts) / len(flat),
               "frac_zero_std_groups": n_zero / max(1, len(rew)), "grad_norm": sum(gnorms) / max(1, len(gnorms)),
               **{k: sum(v) / max(1, len(v)) for k, v in stats.items()}}
        log.append(row)
        if step % a.logging_steps == 0 or step == steps:
            print("[grpo_trl:torch] " + json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                                    for k, v in row.items()}), flush=True)
    _save(model, tok, a, {"steps": steps})
    with open(Path(a.out) / "metrics.jsonl", "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    return model, tok, log


# ------------------------------------------------------------------ CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Decision-level GRPO (R6b)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--init", default=None, help="LoRA adapter to continue from (e.g. the SFT checkpoint)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None, help="held-out examples: restricted-readout acc/NLL/ECE at the end")
    ap.add_argument("--out", required=True)
    ap.add_argument("--impl", choices=["trl", "torch"], default="trl")
    ap.add_argument("--rewards", default="correct,format", help=f"comma list from {REWARDS}")
    ap.add_argument("--reward-weights", default=None, help="comma list aligned with --rewards (default all 1)")
    ap.add_argument("--strict-eos", action="store_true", help="format reward also requires EOS after the code")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=1, help="1 = the option token only; 2 = + EOS")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02, help="KL coefficient (TRL default 0 = no reference)")
    ap.add_argument("--loss-type", default="dapo", help="TRL loss_type: grpo | dr_grpo | dapo | bnpo | cispo | ...")
    ap.add_argument("--scale-rewards", default="group", choices=["group", "batch", "none"])
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--epsilon-high", type=float, default=0.28)
    ap.add_argument("--num-iterations", type=int, default=1, help="mu: updates per generated batch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-scheduler", default="constant")
    ap.add_argument("--batch-size", type=int, default=16, help="per-device COMPLETIONS per micro-step (TRL)")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-r", type=int, default=32, help="0 = full fine-tuning (trl impl only)")
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=4096, help="prompt token budget (state trimmed in the middle)")
    ap.add_argument("--n-permutations", type=int, default=1, help="rows per example with fresh option orders")
    ap.add_argument("--no-permute", action="store_true")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use-vllm", action="store_true", help="TRL: generate with vLLM (not needed for 1 token)")
    ap.add_argument("--vllm-mode", default="colocate")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    a = parse_args(argv)
    names = [n.strip() for n in a.rewards.split(",") if n.strip()]
    weights = [float(w) for w in a.reward_weights.split(",")] if a.reward_weights else [1.0] * len(names)
    if len(weights) != len(names):
        raise SystemExit("--reward-weights must align with --rewards")
    bf16 = bool(a.bf16 and torch.cuda.is_available())
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.init or a.model)
    examples = read_jsonl(a.train)
    if a.max_examples:
        examples = examples[: a.max_examples]
    rows = build_rows(examples, tok, a.n_permutations, a.seed, a.max_len, permute=not a.no_permute)
    if not rows:
        raise SystemExit("no usable examples (need 2-52 options and a non-zero target)")
    if "qvalue" in names and not any(r["q"] for r in rows):
        raise SystemExit("--rewards qvalue needs examples with q_values (training/mc_rollouts.py)")
    print(f"[grpo_trl] {len(rows)} rows from {len(examples)} examples; impl={a.impl}; rewards={names} {weights}")
    if a.impl == "trl":
        model, tok, _ = train_trl(a, rows, names, weights, bf16)
    else:
        model, tok, _ = train_torch(a, rows, names, weights, bf16)
    res: dict[str, Any] = {"out": a.out, "rows": len(rows)}
    if a.dev and (a.impl == "torch" or _is_main()):
        ev = evaluate_decisions(model, tok, read_jsonl(a.dev), a.max_len)
        Path(a.out, "eval.json").write_text(json.dumps(ev, indent=1))
        res["dev"] = ev
    print(json.dumps(res, indent=1))
    return res


def _is_main() -> bool:
    import os

    return os.environ.get("RANK", "0") == "0"


if __name__ == "__main__":
    main()
