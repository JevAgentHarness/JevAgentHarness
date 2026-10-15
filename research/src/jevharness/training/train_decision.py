"""Fine-tune a causal LM into a single-pass typed decision model.

  # SFT (soft CE + Brier), LoRA on Qwen3-4B, option permutation every example
  accelerate launch -m jevharness.training.train_decision \
      --model Qwen/Qwen3-4B --train data/decisions/train.jsonl --dev data/decisions/dev.jsonl \
      --loss soft_ce_brier --lora-r 32 --lr 1e-4 --epochs 2 --out checkpoints/jev4b-sft

  # KL to MC targets p* ∝ pi_ref exp(Q/beta): just build targets with training/data.mc_target and use soft_ce
  # DPO / exact policy gradient on counterfactual Q-values (reference = frozen copy of --init; --ref base = LoRA off)
  ... --init checkpoints/jev4b-sft --loss dpo --beta 0.1 --lr 5e-6
  ... --init checkpoints/jev4b-sft --loss pg  --kl-weight 0.02 --lr 1e-5

Data format: see ``training/data.py``. Examples with a ``q_values`` dict feed dpo/pg.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .data import permuted, read_jsonl


def load_causal_lm(name: str, dtype: torch.dtype, **kw: Any) -> Any:
    """Load ``name`` as a TEXT-ONLY causal LM (``logits_to_keep`` works; ``decision_model.code_token_ids`` still
    checks the option codes on the tokenizer).

    Vision-language checkpoints of the Qwen3.5 family -- ``Qwen/Qwen3.5-*`` and ``internlm/Intern-Decision-*``:
    ``architectures = ["Qwen3_5ForConditionalGeneration"]``, weights ``model.language_model.*`` + ``model.visual.*``
    -- are loaded through the text-only class. transformers >= 5 maps model_type ``qwen3_5`` to
    ``Qwen3_5ForCausalLM``, which takes ``config.text_config``, renames ``model.language_model.*`` -> ``model.*``
    (``conversion_mapping``: ``qwen3_5_text``) and drops ``model.visual.*`` / ``mtp.*`` on load, so no vision tower,
    processor or Pillow is needed (verified on 5.18.0; the checkpoints' own ``requirements.txt`` pins 5.14.1).
    A transformers without that mapping (the composite config is "unrecognized" for ``AutoModelForCausalLM``)
    falls back to the image-text-to-text class: its forward takes text-only inputs with ``logits_to_keep``, so
    serving still works, at the cost of holding the vision tower in memory.
    """
    from transformers import AutoModelForCausalLM

    def _load(cls: Any) -> Any:
        try:  # transformers>=5 renamed torch_dtype -> dtype
            return cls.from_pretrained(name, dtype=dtype, **kw)
        except TypeError:
            return cls.from_pretrained(name, torch_dtype=dtype, **kw)

    try:
        return _load(AutoModelForCausalLM)
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText

        print(f"[train_decision] {name}: no text-only causal-LM class for this transformers version; loading the "
              "multimodal class (vision tower kept, text-only forward)", file=sys.stderr)
        return _load(AutoModelForImageTextToText)


class DecisionDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], permute: bool = True, seed: int = 0):
        self.ex = [e for e in examples if 2 <= len(e["options"]) <= 52]
        self.permute = permute
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.ex)

    def lengths(self) -> list[int]:
        """Rendered-prompt length proxy (characters of state, question and options) for length-grouped batching."""
        return [len(e.get("state") or "") + len(e.get("question") or "") + sum(len(o) + 4 for o in e["options"])
                for e in self.ex]

    def __getitem__(self, i: int) -> dict[str, Any]:
        e = self.ex[i]
        prompt, tgt, idx = permuted(e, self.rng if self.permute else None)
        qv = e.get("q_values")
        q = [qv.get(e["options"][j]) for j in idx] if qv else None
        return {"prompt": prompt, "target": tgt, "q": q, "ref_logits": None}


class TokenBudgetBatchSampler:
    """Micro-batches of row indices with a padded-token budget (``decision_model.token_budget_batches``): every epoch
    the rows are shuffled, cut into megabatches of ``mega`` rows, each megabatch is sorted by length and split by the
    budget, and the micro-batches are shuffled. Deterministic in (seed, epoch)."""

    def __init__(self, lengths: list[int], max_tokens: int, max_batch: int = 64, seed: int = 0, mega: int = 1024):
        self.lengths, self.max_tokens, self.max_batch, self.seed, self.mega = lengths, max_tokens, max_batch, seed, mega
        self.epoch = 0
        self._batches = self._make(0)

    def _make(self, epoch: int) -> list[list[int]]:
        from .decision_model import token_budget_batches

        rng = random.Random(self.seed * 1000003 + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        out: list[list[int]] = []
        for s in range(0, len(idx), self.mega):
            mega = sorted(idx[s:s + self.mega], key=lambda i: -self.lengths[i])
            out += token_budget_batches(self.lengths, self.max_tokens, self.max_batch, mega)
        rng.shuffle(out)
        return out

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._make(epoch)

    def mean_rows(self) -> float:
        return len(self.lengths) / max(1, len(self._batches))

    def padding_frac(self) -> float:
        padded = sum(len(b) * max(self.lengths[i] for i in b) for b in self._batches)
        return 1.0 - sum(self.lengths) / max(1, padded)

    def __iter__(self):  # noqa: ANN204
        batches = self._batches
        self.epoch += 1  # the next pass differs even when nobody calls set_epoch
        self._batches = self._make(self.epoch)
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches)


def q_gap(e: dict[str, Any]) -> float:
    """max - min counterfactual Q over the example's options (0 without q_values)."""
    q = [v for v in (e.get("q_values") or {}).values() if v is not None]
    return max(q) - min(q) if len(q) >= 2 else 0.0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--init", default=None, help="LoRA adapter / checkpoint to continue from")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss", default="soft_ce_brier", choices=["soft_ce", "soft_ce_brier", "dpo", "pg"])
    ap.add_argument("--brier-weight", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--kl-weight", type=float, default=0.02)
    ap.add_argument("--ref", choices=["auto", "init", "base"], default="auto",
                    help="dpo/pg reference: frozen copy of --init (auto when --init is given) or the base model")
    ap.add_argument("--min-q-gap", type=float, default=0.0,
                    help="dpo/pg: keep only examples with q_values whose max-min Q gap is >= this (e.g. 0.3)")
    ap.add_argument("--lora-r", type=int, default=32, help="0 = full fine-tuning")
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--no-permute", action="store_true")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--max-steps", type=int, default=-1, help="> 0 overrides --epochs (smoke tests)")
    ap.add_argument("--group-by-length", action="store_true",
                    help="length-grouped batches (megabatches of 50 x batch sorted by prompt length): far less padding "
                         "for mixed 2-option / 52-option rows; gradient accumulation still mixes lengths")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="non-reentrant activation checkpointing (off = faster when the GPU memory allows it)")
    ap.add_argument("--max-batch-tokens", type=int, default=0,
                    help="token-budget micro-batches instead of --batch-size rows: rows sorted by tokenized length "
                         "inside shuffled megabatches, each micro-batch <= N padded tokens (0 = off). Pair with "
                         "--no-gradient-checkpointing; the loss of a micro-batch is weighted by its row count so every "
                         "row counts equally, and --eff-batch sets the rows per optimizer step")
    ap.add_argument("--eff-batch", type=int, default=64,
                    help="with --max-batch-tokens: rows per optimizer step (gradient accumulation is derived)")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer, Trainer, TrainingArguments

    from .decision_model import DecisionCollator, decision_loss, option_logits, reference_context

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if a.bf16 else torch.float32
    model = load_causal_lm(a.model, dtype)
    use_lora = a.lora_r > 0
    if use_lora:
        from peft import LoraConfig, PeftModel, get_peft_model
        if a.init:
            model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0,
                                                     target_modules="all-linear", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    elif a.loss in ("dpo", "pg"):
        raise SystemExit("dpo/pg need a reference policy: use LoRA (reference = adapter disabled or --init copy)")
    ref_mode = a.ref if a.ref != "auto" else ("init" if a.init else "base")
    if a.loss in ("dpo", "pg") and ref_mode == "init":
        if not a.init:
            raise SystemExit("--ref init needs --init")
        from .decision_model import add_reference_adapter

        add_reference_adapter(model, a.init)  # frozen copy of the starting (SFT) policy = DPO / KL anchor

    train_ex = read_jsonl(a.train)
    if a.loss in ("dpo", "pg") and a.min_q_gap > 0:
        n0 = len(train_ex)
        train_ex = [e for e in train_ex if q_gap(e) >= a.min_q_gap]
        print(f"[train_decision] --min-q-gap {a.min_q_gap}: kept {len(train_ex)}/{n0} examples")
        if not train_ex:
            raise SystemExit("no example passes --min-q-gap (need q_values: training/mc_rollouts.py)")
    train = DecisionDataset(train_ex, permute=not a.no_permute, seed=a.seed)
    dev = DecisionDataset(read_jsonl(a.dev), permute=False) if a.dev else None
    collator = DecisionCollator(tok, max_len=a.max_len)

    budget = None
    if a.max_batch_tokens > 0:
        lengths = [len(collator.encode(permuted(e, None)[0])) for e in train.ex]
        budget = TokenBudgetBatchSampler(lengths, a.max_batch_tokens, seed=a.seed)
        mean_b = len(train) / max(1, len(budget))
        a.grad_accum = max(1, round(a.eff_batch / mean_b))
        print(f"[train_decision] token budget {a.max_batch_tokens}: {len(budget)} micro-batches/epoch, "
              f"{mean_b:.2f} rows each on average, grad-accum {a.grad_accum} (~{a.grad_accum * mean_b:.0f} rows/step), "
              f"mean length {sum(lengths) / len(lengths):.0f} tokens, padding "
              f"{budget.padding_frac():.3f}", flush=True)

    class DecisionTrainer(Trainer):
        def get_train_dataloader(self):  # noqa: ANN202
            if budget is None:
                return super().get_train_dataloader()
            from torch.utils.data import DataLoader

            dl = DataLoader(train, batch_sampler=budget, collate_fn=collator, num_workers=2, pin_memory=True)
            return self.accelerator.prepare(dl)

        def _get_train_sampler(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            if not a.group_by_length:
                return super()._get_train_sampler(*args, **kwargs)
            from transformers.trainer_pt_utils import LengthGroupedSampler

            g = torch.Generator()
            g.manual_seed(a.seed)
            return LengthGroupedSampler(a.batch_size, lengths=train.lengths(), generator=g)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001
            if a.loss in ("dpo", "pg"):  # reference first: adapter switching must not straddle the policy graph
                base = self.accelerator.unwrap_model(model)  # DDP-wrapped models have no adapter API
                with torch.no_grad(), reference_context(base, ref_mode):
                    inputs["ref_logits"] = option_logits(base, inputs["input_ids"], inputs["attention_mask"],
                                                         inputs["code_ids"]).nan_to_num(neginf=-1e4)
            z = option_logits(model, inputs["input_ids"], inputs["attention_mask"], inputs["code_ids"])
            loss = decision_loss(z, inputs, a.loss, a.brier_weight, a.beta, a.kl_weight)
            if budget is not None and model.training:  # equal weight per row across unequal micro-batches
                loss = loss * (z.size(0) / budget.mean_rows())
            return (loss, {"logits": z}) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):  # noqa: ANN001
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                z = option_logits(model, inputs["input_ids"], inputs["attention_mask"], inputs["code_ids"])
                loss = decision_loss(z, inputs, "soft_ce")
            return loss, z.nan_to_num(neginf=-1e4), inputs["target"]

    def metrics(ev: Any) -> dict[str, float]:
        z, tgt = torch.tensor(ev.predictions), torch.tensor(ev.label_ids)
        p = torch.softmax(z, -1)
        acc = (p.argmax(-1) == tgt.argmax(-1)).float().mean().item()
        conf = p.max(-1).values
        corr = (p.argmax(-1) == tgt.argmax(-1)).float()
        bins = torch.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += m.float().mean().item() * abs(conf[m].mean().item() - corr[m].mean().item())
        return {"acc": acc, "ece": ece}

    import inspect
    warm = ({"warmup_ratio": 0.03} if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters
            else {"warmup_steps": 0.03})  # transformers>=5: float warmup_steps = ratio
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs,
        lr_scheduler_type="cosine", weight_decay=0.0, max_grad_norm=1.0, **warm,
        logging_steps=10, save_strategy="epoch", eval_strategy="epoch" if dev else "no",
        bf16=a.bf16, gradient_checkpointing=a.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.gradient_checkpointing else None,
        ddp_find_unused_parameters=False, remove_unused_columns=False, report_to=a.report_to,
        seed=a.seed, dataloader_num_workers=2, **({"max_steps": a.max_steps} if a.max_steps > 0 else {}))
    if args.gradient_checkpointing and use_lora:
        model.enable_input_require_grads()
    trainer = DecisionTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                              data_collator=collator, compute_metrics=metrics if dev else None)
    trainer.train()
    final = trainer.accelerator.unwrap_model(trainer.model)
    if "ref" in (getattr(final, "peft_config", None) or {}):
        final.delete_adapter("ref")  # the saved checkpoint holds the trained adapter only
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    Path(a.out, "decision_config.json").write_text(json.dumps({"base_model": a.model, "loss": a.loss, "init": a.init,
                                                               "ref": ref_mode if a.loss in ("dpo", "pg") else None,
                                                               "lora": use_lora, "max_len": a.max_len}, indent=1))
    if dev:
        print(json.dumps(trainer.evaluate(), indent=1))


if __name__ == "__main__":
    main()
