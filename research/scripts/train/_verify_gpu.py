"""Verify the training stack on this host; called by ``scripts/train/setup_gpu.sh`` (step 5).

  python scripts/train/_verify_gpu.py --model Qwen/Qwen3-0.6B --out runs/gpu_setup/env_report.json
  python scripts/train/_verify_gpu.py --model /tmp/tiny --allow-cpu      # CPU check with make_tiny_model.py

Checks (each recorded as {name, ok, severity, detail}; exit 1 when an ``error`` check fails):
  versions   installed torch / transformers / trl / peft / accelerate / datasets vs configs/train/constraints-gpu.txt
             (warn: a drift is allowed but must be reported)
  cuda       CUDA visible, device names / memory / compute capability, bf16 support            (error without --allow-cpu)
  trl        every GRPOConfig field that training/grpo_trl.py passes exists                   (error)
  codes      option codes A-Z / a-z are single tokens of the model's tokenizer                  (error)
  template   the chat prompt renders; Qwen3 templates get enable_thinking=False                  (error)
  step       one forward + backward of the R2 decision loss (soft CE + Brier) with LoRA r=8 and gradient checkpointing
             on two data/train_v0 decisions: loss finite, LoRA grads finite and non-zero, peak memory      (error)
  hf_gated   (--check-hf) access to the gated xLAM dataset                                        (warn)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PKGS = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
GRPO_FIELDS = (  # == the keyword arguments of the GRPOConfig(...) call in training/grpo_trl.py
    "output_dir", "per_device_train_batch_size", "gradient_accumulation_steps", "num_generations",
    "max_completion_length", "temperature", "top_p", "top_k", "beta", "loss_type", "scale_rewards", "epsilon",
    "epsilon_high", "num_iterations", "reward_weights", "learning_rate", "lr_scheduler_type", "weight_decay",
    "max_grad_norm", "num_train_epochs", "max_steps", "logging_steps", "save_strategy", "save_steps", "bf16",
    "gradient_checkpointing", "disable_dropout", "report_to", "seed", "dataloader_num_workers", "use_vllm",
    "vllm_mode", "log_completions")


def pins(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            k, v = line.split("==", 1)
            out[k.strip().lower().replace("_", "-")] = v.strip()
    return out


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {"python": platform.python_version(), "host": platform.node()}

    def add(self, name: str, ok: bool, detail: Any = None, severity: str = "error") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "severity": severity, "detail": detail})
        print(f"[{'ok' if ok else ('WARN' if severity == 'warn' else 'FAIL'):>4}] {name}: {detail}", flush=True)
        return ok

    def run(self, name: str, fn: Any, severity: str = "error") -> Any:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - every check reports, none aborts the others
            ok, detail = False, f"{type(e).__name__}: {str(e)[:500]}"
        self.add(name, ok, detail, severity)
        return ok

    @property
    def failed(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["ok"] and c["severity"] == "error"]


def check_versions(constraints: Path) -> tuple[bool, Any]:
    import importlib

    want = pins(constraints)
    got, drift = {}, {}
    for p in PKGS:
        v = getattr(importlib.import_module(p), "__version__", "?")
        got[p] = v
        if p in want and v.split("+")[0] != want[p]:
            drift[p] = {"installed": v, "pinned": want[p]}
    return not drift, {"installed": got, "drift": drift}


def check_cuda(allow_cpu: bool) -> tuple[bool, Any]:
    import torch

    if not torch.cuda.is_available():
        return allow_cpu, {"cuda": False, "torch_cuda_build": torch.version.cuda,
                           "hint": "driver too old for this torch build? see setup_gpu.sh TORCH_INDEX"}
    devs = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devs.append({"name": p.name, "mem_gb": round(p.total_memory / 2**30, 1), "cc": f"{p.major}.{p.minor}"})
    return torch.cuda.is_bf16_supported(), {"devices": devs, "bf16": torch.cuda.is_bf16_supported(),
                                            "torch_cuda_build": torch.version.cuda,
                                            "cudnn": torch.backends.cudnn.version()}


def check_trl() -> tuple[bool, Any]:
    from trl import GRPOConfig

    names = {f.name for f in dataclasses.fields(GRPOConfig)}
    missing = [f for f in GRPO_FIELDS if f not in names]
    return not missing, {"missing_fields": missing} if missing else "all GRPOConfig fields used by grpo_trl.py exist"


def examples(data: Path, n: int = 2) -> list[dict[str, Any]]:
    from jevharness.training.data import read_jsonl

    if data.exists():
        ex = [e for e in read_jsonl(data) if 2 <= len(e["options"]) <= 52]
        if ex:  # one long tool_select Choice + one short gate when available
            ex.sort(key=lambda e: -len(e["options"]))
            return [ex[0], ex[-1]][:n]
    return [{"id": f"t{i}/0/x", "state": "TASK: refund order ORD-1", "question": "Which tool next?",
             "options": ["find_user", "get_orders", "refund"], "target": {"find_user": 1.0, "get_orders": 0.0,
                                                                          "refund": 0.0}} for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--constraints", default=str(ROOT / "configs/train/constraints-gpu.txt"))
    ap.add_argument("--data", default=str(ROOT / "data/train_v0/train.jsonl"))
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default=str(ROOT / "runs/gpu_setup/env_report.json"))
    ap.add_argument("--allow-cpu", action="store_true", help="CPU-only host (pipeline checks with a tiny model)")
    ap.add_argument("--check-hf", action="store_true", help="also check access to gated HF datasets (xLAM)")
    a = ap.parse_args(argv)
    rep = Report()
    rep.run("versions", lambda: check_versions(Path(a.constraints)), severity="warn")
    rep.run("cuda", lambda: check_cuda(a.allow_cpu))
    rep.run("trl", check_trl)

    import torch
    from transformers import AutoTokenizer

    from jevharness.training.decision_model import DecisionCollator, chat_text, code_token_ids

    state: dict[str, Any] = {}

    def load_tok() -> tuple[bool, Any]:
        state["tok"] = tok = AutoTokenizer.from_pretrained(a.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        ids = code_token_ids(tok)
        return len(set(ids)) == len(ids), f"{len(ids)} single-token codes"

    def template() -> tuple[bool, Any]:
        text = chat_text(state["tok"], "STATE:\nS\n\nQUESTION: Q?\nOPTIONS:\nA. x\nB. y")
        qwen3 = "enable_thinking" in str(getattr(state["tok"], "chat_template", "") or "")
        ok = "OPTIONS" in text and (not qwen3 or "<think>" not in text.split("OPTIONS")[-1]
                                    or "</think>" in text.split("OPTIONS")[-1])
        return ok, {"qwen3_hybrid_template": qwen3, "prompt_tail": text[-80:]}

    def step() -> tuple[bool, Any]:
        from peft import LoraConfig, get_peft_model

        from jevharness.training.data import permuted
        from jevharness.training.decision_model import decision_loss, option_logits
        from jevharness.training.train_decision import load_causal_lm

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        t0 = time.perf_counter()
        model = load_causal_lm(a.model, dtype)
        load_s = time.perf_counter() - t0
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear",
                                                 task_type="CAUSAL_LM")).to(dev)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        exs = examples(Path(a.data))
        batch = []
        for e in exs:
            prompt, tgt, _ = permuted(e, None)
            batch.append({"prompt": prompt, "target": tgt, "q": None, "ref_logits": None})
        enc = DecisionCollator(state["tok"], max_len=a.max_len)(batch)
        enc = {k: v.to(dev) for k, v in enc.items()}
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        z = option_logits(model, enc["input_ids"], enc["attention_mask"], enc["code_ids"])
        loss = decision_loss(z, enc, "soft_ce_brier", 0.5)
        loss.backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        fb_s = time.perf_counter() - t1
        grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
        finite = bool(grads) and all(torch.isfinite(g).all() for g in grads)
        nonzero = any(g.abs().max() > 0 for g in grads)
        detail = {"device": dev, "dtype": str(dtype), "loss": float(loss.detach()), "seq_len": int(enc["input_ids"].shape[1]),
                  "n_options": [len(e["options"]) for e in exs], "lora_tensors_with_grad": len(grads),
                  "load_s": round(load_s, 1), "fwd_bwd_s": round(fb_s, 3),
                  "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None}
        return bool(torch.isfinite(loss)) and finite and nonzero, detail

    if rep.run("codes", load_tok):
        rep.run("template", template)
        rep.run("step", step)
    if a.check_hf:
        def hf() -> tuple[bool, Any]:
            from huggingface_hub import HfApi

            HfApi().dataset_info("Salesforce/xlam-function-calling-60k")
            return True, "xLAM accessible"
        rep.run("hf_gated", hf, severity="warn")
    rep.info["model"] = a.model
    out = {"ok": not rep.failed, "failed": rep.failed, "info": rep.info, "checks": rep.checks}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"ok": out["ok"], "failed": rep.failed, "report": a.out}))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
