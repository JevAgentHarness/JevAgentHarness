#!/usr/bin/env python3
"""Exploratory row E14 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E14: a sub-1B decision model with a scoring
head"; descriptive, outside every Holm table): Qwen3.5-0.8B with a fresh Clef joint schema head, trained with Unsloth's
from-LM recipe and defaults, on the agent's states (arm A), on Unsloth's generic mix (arm G) or on both (AG).

Box side (the E14 host, Unsloth 2026.10.2 in its own venv; scripts/aws/e14_small.sh):

    python scripts/train/e14_clef.py build-a --out D        # CPU: SFT v1's exact training input -> Clef records
    python scripts/train/e14_clef.py build-g --out D        # CPU + downloads: Unsloth's 12-source mix + eval sets
    python scripts/train/e14_clef.py train --arm A --data D --out CK [--hours-used H]   # GPU
    python scripts/train/e14_clef.py evalsets --ckpt CK/A/final --data D --out R/evalsets_A.json   # GPU

Recipe (registration (b)): Unsloth's ``scripts/train_decision_from_lm.py`` defaults -- LoRA r 64 / alpha 64 / dropout
0 (Unsloth's Clef LoRA), head lr 3e-4, LoRA lr 1e-4, cosine with 5 % warm-up, weight decay 0.01, 8 records x 2
accumulation, max length 4,096, bf16, seed 3407, no head warm-up, no 4-bit; Unsloth's Clef loss. A: epoch 1, then a
second pass with the schedule restarted only if the GPU projection allows (registration (d)); the epoch with the higher
development accuracy is kept (tie: epoch 1). Temperatures: Unsloth's ``calibrate`` on the development records.
Only on a CUDA out-of-memory error: 4 records x 4 accumulation (the same 16 records per step), then 2 x 8.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BASE = "Qwen/Qwen3.5-0.8B"
BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"
SEED = 3407
MIX_SHA = "9fe1088d8e5be533f38a2115fe369b9b238018367bbb6b4e729c19f9f94ebe37"   # SFT v1's train_mix.jsonl
V1 = {"dev": "data/train_v1/dev.jsonl.gz", "rec": "data/train_v1/dev_recovery.jsonl.gz",
      "ko": "data/train_v1/dev_knockout.jsonl.gz"}
G_SOURCES = ("banking77", "clinc150", "mnli", "snli", "wanli", "boolq", "ag_news", "sst5", "mmlu", "commonsense_qa",
             "arc", "prompt_injections", "xlam")   # the from-LM script's DEFAULT_SOURCES (xlam is gated -> skipped)
G_ROWS, G_EVAL_ROWS = 12000, 500
RECIPE = {"lora_r": 64, "lora_alpha": 64, "lora_dropout": 0.0, "lr": 1e-4, "head_lr": 3e-4, "batch": 8,
          "grad_accum": 2, "max_len": 4096, "scheduler": "cosine", "warmup_frac": 0.05, "weight_decay": 0.01,
          "seed": SEED, "head_warmup_steps": 0, "load_in_4bit": False}
RESERVE_H = 3.5     # registration (d): hours kept for evaluation, serving and live before a second epoch
MEM_FALLBACK = ((8, 2), (4, 4), (2, 8))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jl(rows: Iterable[dict[str, Any]], p: Path) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def log(*a: Any) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ================================================================================================ data
def cmd_build_a(a: argparse.Namespace) -> int:
    """SFT v1's training input rebuilt exactly (sha256 checked), seed-guarded, as Clef records; the v1 dev files."""
    from jevharness.bench import registry as R
    from jevharness.training import clef_data as CD
    from jevharness.training.data import mix
    from jevharness.training.data_v1 import jevgym_seed
    from jevharness.training.seed_guard import TRAIN_FORBIDDEN

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = mix([(str(ROOT / "data/train_v1/train.jsonl.gz"), 1.0),
                (str(ROOT / "data/train_v1/replay_when2call.jsonl.gz"), 0.45)], seed=0)
    raw = out / "train_mix.jsonl"
    with open(raw, "w") as f:   # byte-identical to `python -m jevharness.training.data mix ... --out` (write_jsonl)
        for r in rows:
            f.write(json.dumps(r) + "\n")
    got = sha256(raw)
    if got != MIX_SHA:
        raise SystemExit(f"train_mix sha256 {got} != SFT v1's {MIX_SHA}")
    bad = []
    for r in rows:
        s = jevgym_seed(r["id"])
        if s is None:
            continue
        if (s < R.TRAIN_MIN_SEED or R.in_ranges(s, R.training_exclusions() + (R.CALIBRATION_RANGE,))
                or any(lo <= s <= hi for lo, hi in TRAIN_FORBIDDEN)):
            bad.append(r["id"])
    if bad:
        raise SystemExit(f"rows on guarded seeds: {bad[:5]}")
    info: dict[str, Any] = {"train_mix_sha256": got, "rows": len(rows), "seed_guard": "ok", "files": {}}
    recs = CD.rows_to_records(rows)
    info["files"]["train_a"] = {"n": write_jl(recs, out / "train_a.jsonl.gz"), **CD.record_stats(recs)}
    for k, rel in V1.items():
        dv = CD.rows_to_records(read_jl(ROOT / rel))
        name = {"dev": "dev_a", "rec": "dev_recovery_a", "ko": "dev_knockout_a"}[k]
        info["files"][name] = {"source": rel, "source_sha256": sha256(ROOT / rel),
                               "n": write_jl(dv, out / f"{name}.jsonl.gz"), **CD.record_stats(dv)}
    raw.unlink()
    for f in sorted(out.glob("*.jsonl.gz")):
        info.setdefault("sha256", {})[f.name] = sha256(f)
    (out / "build_a.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=1))
    return 0


def cmd_build_g(a: argparse.Namespace) -> int:
    """Unsloth's from-LM data, as its script builds it with its defaults (rows 12,000, typed-decisions included)."""
    from unsloth.models import decision_datasets as dd

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"recipe": "unsloth scripts/train_decision_from_lm.py defaults", "rows_arg": G_ROWS,
                            "seed": SEED, "skipped": {}, "eval": {}}
    td_test = dd.load_source("typed_decisions", "test")
    evals = {"typed_decisions": td_test}
    for name in ("banking77", "clinc150"):
        evals[name] = [dd.canonical_row(r) for r in dd.load_source(name, "test", limit=G_EVAL_ROWS, seed=0)]
    eval_texts = [r["state"] for rows in evals.values() for r in rows]
    pools = {}
    for name in G_SOURCES:
        try:
            pools[name] = dd.load_source(name, "train", limit=1, seed=SEED)
        except Exception as exc:  # noqa: BLE001 -- gated or unavailable sources are reported and skipped (as the script)
            info["skipped"][name] = f"{type(exc).__name__}: {str(exc)[:200]}"
            log("skipping", name, info["skipped"][name])
    per_source = max(1, -(-G_ROWS // max(1, len(pools))))
    pools = {name: dd.load_source(name, "train", limit=2 * per_source, seed=SEED) for name in pools}
    rows = dd.build_decision_mixture(pools, n_rows=G_ROWS, seed=SEED, decontaminate_against=eval_texts) if pools else []
    rng = random.Random(SEED)
    cleaner = dd.Decontaminator(eval_texts)
    td = [dd.augment_row(r, rng) for r in dd.load_source("typed_decisions", "train") if not cleaner.contaminated(r)]
    rows += td
    random.Random(SEED).shuffle(rows)
    info["sources"] = sorted(pools) + ["typed_decisions"]
    info["rows"] = len(rows)
    info["typed_decisions_train_rows"] = len(td)
    from collections import Counter
    info["rows_by_source"] = dict(Counter(str(r.get("source")) for r in rows))
    info["files"] = {"train_g": write_jl(rows, out / "train_g.jsonl.gz")}
    for name, ev in evals.items():
        info["eval"][name] = write_jl(ev, out / f"eval_{name}.jsonl.gz")
    for f in sorted(out.glob("*_g.jsonl.gz")) + sorted(out.glob("eval_*.jsonl.gz")):
        info.setdefault("sha256", {})[f.name] = sha256(f)
    (out / "build_g.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k not in ("sha256",)}, indent=1))
    return 0


def clef_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clef records (clef_data) -> Unsloth build_dataset rows (state, questions, gold)."""
    return [{"state": r["state"], "questions": r["questions"], "gold": r["gold"]} for r in records]


# ================================================================================================ model
def load_model(max_len: int, base: str = BASE, rev: str | None = BASE_REV):
    from unsloth import FastDecisionModel

    return FastDecisionModel.from_pretrained(base, revision=rev, decision_head="clef", max_seq_length=max_len,
                                             load_in_4bit=False, random_state=SEED)


def metrics(model: Any, proc: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    from unsloth import FastDecisionModel

    m = FastDecisionModel.evaluate(model, proc, items)
    return {"decisions": sum(len(it.get("labels", [it.get("label")])) for it in items),
            **{k: round(float(m[k]), 5) for k in ("accuracy", "ece", "record_accuracy", "loss") if k in m}}


def trainer(model: Any, proc: Any, train: list[dict[str, Any]], out: Path, batch: int, accum: int, epochs: float = 1.0):
    import torch
    from transformers import TrainingArguments
    from unsloth import DecisionTrainer

    steps = epochs * -(-len(train) // (batch * accum))
    return DecisionTrainer(
        model=model, tokenizer=proc, train_dataset=train, head_learning_rate=RECIPE["head_lr"],
        args=TrainingArguments(
            output_dir=str(out / "_run"), per_device_train_batch_size=batch, gradient_accumulation_steps=accum,
            learning_rate=RECIPE["lr"], lr_scheduler_type=RECIPE["scheduler"],
            warmup_steps=max(1, round(RECIPE["warmup_frac"] * steps)), weight_decay=RECIPE["weight_decay"],
            max_steps=-1, num_train_epochs=epochs, bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            logging_steps=10, report_to="none", save_strategy="no", seed=SEED))


def snapshot(model: Any) -> dict[str, Any]:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


def restore(model: Any, snap: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n].to(p.device, p.dtype))


def is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def run_epoch(model: Any, proc: Any, train: list[dict[str, Any]], out: Path, rec: dict[str, Any], tag: str) -> dict:
    """One epoch with the recipe's schedule (restarted for a second pass); memory fallback on OOM only."""
    import torch

    start = snapshot(model)
    for batch, accum in MEM_FALLBACK[rec.get("_mem_idx", 0):]:
        t0 = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            t = trainer(model, proc, train, out, batch, accum)
            res = t.train()
            losses = [h["loss"] for h in t.state.log_history if "loss" in h]
            ep = {"tag": tag, "batch": batch, "grad_accum": accum, "steps": t.state.global_step,
                  "minutes": round((time.time() - t0) / 60, 2),
                  "s_per_step": round(res.metrics["train_runtime"] / max(1, t.state.global_step), 3),
                  "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)] if losses else None,
                  "loss_log": [round(x, 4) for x in losses],
                  "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            del t
            torch.cuda.empty_cache()
            return ep
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            log(f"{tag}: out of memory at {batch} x {accum}; next setting")
            rec.setdefault("oom", []).append({"tag": tag, "batch": batch, "grad_accum": accum})
            rec["_mem_idx"] = rec.get("_mem_idx", 0) + 1
            restore(model, start)
            torch.cuda.empty_cache()
    raise SystemExit("every memory setting ran out of memory")


def build_items(model: Any, proc: Any, rows: list[dict[str, Any]], what: str, rec: dict[str, Any]) -> list[dict]:
    from unsloth import FastDecisionModel

    items, report = FastDecisionModel.build_dataset(rows, proc, model)
    rec.setdefault("build", {})[what] = {k: report.get(k) for k in ("total", "skipped", "reason", "truncated")} | \
        {"items": len(items)}
    log(what, rec["build"][what])
    return items


def cmd_train(a: argparse.Namespace) -> int:
    import torch
    from unsloth import FastDecisionModel

    arm, data, out = a.arm.upper(), Path(a.data), Path(a.out) / a.arm.upper()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rec: dict[str, Any] = {"arm": arm, "base": BASE, "base_revision": BASE_REV, "recipe": RECIPE,
                           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hours_used_at_start":
                           a.hours_used, "cap_h": a.cap_h, "reserve_h": RESERVE_H, "gpu": torch.cuda.get_device_name()}
    model, proc = load_model(RECIPE["max_len"])
    rec["head"] = dict(getattr(model.head, "config", {}) or {})
    rec["head_params_m"] = round(sum(p.numel() for p in model.head.parameters()) / 1e6, 2)
    train_rows: list[dict[str, Any]] = []
    dev_items: list[dict[str, Any]] = []
    if arm in ("A", "AG"):
        train_rows += clef_rows(read_jl(data / "train_a.jsonl.gz"))
    if arm in ("G", "AG"):
        train_rows += read_jl(data / "train_g.jsonl.gz")
    if arm == "AG":
        random.Random(SEED).shuffle(train_rows)
    items = build_items(model, proc, train_rows, "train", rec)
    if arm == "G":   # G's held-out split, as the from-LM script (it calibrates there)
        items, dev_items = FastDecisionModel.split_holdout(items, SEED)
        rec["dev"] = {"kind": "G held-out split (split_holdout, seed 3407)", "records": len(dev_items)}
    else:            # A / AG: SFT v1's held-out rows (never trained on)
        dev_items = build_items(model, proc, clef_rows(read_jl(data / "dev_a.jsonl.gz")), "dev_a", rec)
        rec["dev"] = {"kind": "SFT v1 dev (data/train_v1/dev.jsonl.gz)", "records": len(dev_items)}
    rec["train_records"] = len(items)
    rec["base_dev"] = metrics(model, proc, dev_items)
    log("base dev", rec["base_dev"])
    model = FastDecisionModel.get_peft_model(model, r=RECIPE["lora_r"], lora_alpha=RECIPE["lora_alpha"],
                                             lora_dropout=RECIPE["lora_dropout"], random_state=SEED)
    rec["epochs"] = []
    ep1 = run_epoch(model, proc, items, out, rec, "epoch1")
    ep1["dev"] = metrics(model, proc, dev_items)
    rec["epochs"].append(ep1)
    log("epoch 1", {k: ep1[k] for k in ("minutes", "s_per_step", "loss_first_last", "dev")})
    model.save_pretrained(str(out / "ep1"))
    (out / "train_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    snap1 = snapshot(model)
    used_h = a.hours_used + (time.time() - t_start) / 3600
    proj = used_h + ep1["minutes"] / 60 + RESERVE_H
    want2 = arm == "A" or (arm == "AG" and a.epochs == 2)
    rec["second_epoch"] = {"considered": want2, "hours_used": round(used_h, 3), "projection_h": round(proj, 3),
                           "cap_h": a.cap_h, "allowed": want2 and proj <= a.cap_h and not a.no_second}
    selected = 1
    if rec["second_epoch"]["allowed"]:
        ep2 = run_epoch(model, proc, items, out, rec, "epoch2")
        ep2["dev"] = metrics(model, proc, dev_items)
        rec["epochs"].append(ep2)
        log("epoch 2", {k: ep2[k] for k in ("minutes", "s_per_step", "loss_first_last", "dev")})
        model.save_pretrained(str(out / "ep2"))
        if ep2["dev"]["accuracy"] > ep1["dev"]["accuracy"]:
            selected = 2
        else:
            restore(model, snap1)
    rec["selected_epoch"] = selected
    rec["selection_rule"] = "higher development accuracy; a tie keeps epoch 1 (registration (d))"
    cal = FastDecisionModel.calibrate(model, proc, dev_items)
    rec["calibration_dev"] = {k: (round(float(v), 5) if isinstance(v, float) else v) for k, v in cal.items()}
    rec["temperature"] = model.decision_config.get("temperature")
    rec["head_temperature"] = model.decision_config.get("head_temperature")
    model.decision_config["training"] = {k: rec.get(k) for k in ("arm", "base", "base_revision", "selected_epoch")}
    model.save_pretrained_merged(str(out / "final"), proc)
    rec["final_dir"] = str(out / "final")
    rec["minutes_total"] = round((time.time() - t_start) / 60, 2)
    rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.pop("_mem_idx", None)
    (out / "train_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    # development read-outs in eval_decisions' format (the served temperatures, as the harness sees them)
    if arm in ("A", "AG", "G") and a.v1_out:
        v1_eval(model, proc, Path(a.v1_out))
    log("done", out / "final")
    return 0


def v1_eval(model: Any, proc: Any, path: Path) -> dict[str, Any]:
    from jevharness.training.clef_engine import ClefEngine
    from jevharness.training.data import read_jsonl
    from jevharness.training.eval_decisions import evaluate

    eng = ClefEngine.from_model(model, proc, name="eval", max_len=16384)
    res: dict[str, Any] = {"model": "e14 clef", "n_perm": 0, "files": {}}
    for rel in V1.values():
        res["files"][rel] = evaluate(eng, read_jsonl(str(ROOT / rel)), n_perm=0)
        log(rel, res["files"][rel]["overall"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    return res


def cmd_evalsets(a: argparse.Namespace) -> int:
    """Unsloth's three evaluation sets (typed-decisions test, BANKING77 / CLINC150 test) for a saved checkpoint, or for
    the fresh-head base model (--base)."""
    from unsloth import FastDecisionModel

    data = Path(a.data)
    if a.base:
        model, proc = load_model(RECIPE["max_len"])
    else:
        model, proc = FastDecisionModel.from_pretrained(a.ckpt, max_seq_length=RECIPE["max_len"])
    res: dict[str, Any] = {"ckpt": None if a.base else a.ckpt, "base": bool(a.base), "sets": {}}
    for name in ("typed_decisions", "banking77", "clinc150"):
        rows = read_jl(data / f"eval_{name}.jsonl.gz")
        items, report = FastDecisionModel.build_dataset(rows, proc, model)
        res["sets"][name] = {**metrics(model, proc, items), "skipped": report.get("skipped")}
        log(name, res["sets"][name])
    accs = [v["accuracy"] for v in res["sets"].values()]
    res["mean_accuracy"] = sum(accs) / len(accs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0


def cmd_v1(a: argparse.Namespace) -> int:
    from unsloth import FastDecisionModel

    model, proc = FastDecisionModel.from_pretrained(a.ckpt, max_seq_length=16384)
    v1_eval(model, proc, Path(a.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-a")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build_a)
    g = sub.add_parser("build-g")
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_build_g)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=["A", "G", "AG", "a", "g", "ag"])
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--hours-used", type=float, default=0.0, help="GPU host hours used before this step")
    t.add_argument("--cap-h", type=float, default=10.0)
    t.add_argument("--epochs", type=int, default=1, help="AG: A's selected epoch count")
    t.add_argument("--no-second", action="store_true")
    t.add_argument("--v1-out", default=None, help="development read-outs (eval_decisions format)")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("evalsets")
    e.add_argument("--ckpt", default=None)
    e.add_argument("--base", action="store_true")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_evalsets)
    v = sub.add_parser("v1")
    v.add_argument("--ckpt", required=True)
    v.add_argument("--out", required=True)
    v.set_defaults(fn=cmd_v1)
    a = ap.parse_args(argv)
    if a.cmd in ("train", "evalsets", "v1", "build-g"):
        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
        import unsloth  # noqa: F401  -- Unsloth before transformers
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
