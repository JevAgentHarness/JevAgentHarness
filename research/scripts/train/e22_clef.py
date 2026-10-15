#!/usr/bin/env python3
"""Exploratory row E22 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E22: the training recipe on tau^2-bench" (e);
descriptive): E14 arm A's merged checkpoint (Qwen3.5-0.8B + Clef joint schema head) continued with a fresh LoRA under
E14's recipe on the tau^2 fit-task records (the agent's own states, labels of (d)); dev-task records choose the epoch
and fit the temperatures. Box side, Unsloth 2026.10.2 in its own venv (scripts/aws/e22_tau2.sh):

    python scripts/train/e22_clef.py train --data D --init CK_A --out CK [--hours-used H] [--cap-h 9.5]

Recipe = E14's (scripts/train/e14_clef.py RECIPE) with max length 8,192 (registration (e)); memory fallback 4 x 4,
then 2 x 8, on an out-of-memory error only. Epoch 1, then a second pass with the schedule restarted only if the host
hours used + epoch 1's duration + 3.0 h stay within --cap-h; the epoch with the higher dev accuracy is kept (tie: 1).
If the start checkpoint cannot be continued (an error before the first optimizer step), the fresh backbone + head.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

MAX_LEN = 8192
RESERVE_H = 3.0


def cmd_train(a: argparse.Namespace) -> int:
    import torch
    from unsloth import FastDecisionModel

    import e14_clef as E

    E.RECIPE["max_len"] = MAX_LEN
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rec: dict[str, Any] = {"row": "E22", "init": a.init, "recipe": dict(E.RECIPE), "started_utc":
                           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hours_used_at_start": a.hours_used,
                           "cap_h": a.cap_h, "reserve_h": RESERVE_H, "gpu": torch.cuda.get_device_name()}
    train_rows = E.clef_rows(E.read_jl(data / "fit.jsonl.gz"))
    dev_rows = E.clef_rows(E.read_jl(data / "dev.jsonl.gz"))
    model = proc = None
    try:
        model, proc = FastDecisionModel.from_pretrained(a.init, max_seq_length=MAX_LEN)
        rec["start"] = "e14_arm_a"
    except Exception as e:  # noqa: BLE001 - registration (e): fall back to the fresh backbone + head
        rec["start"] = "fresh (arm A could not be loaded)"
        rec["start_error"] = f"{type(e).__name__}: {e}"[:2000]
        traceback.print_exc()
        model, proc = E.load_model(MAX_LEN)
    rec["head"] = dict(getattr(model.head, "config", {}) or {})
    items = E.build_items(model, proc, train_rows, "train", rec)
    dev_items = E.build_items(model, proc, dev_rows, "dev", rec)
    rec["train_records"], rec["dev_records"] = len(items), len(dev_items)
    rec["base_dev"] = E.metrics(model, proc, dev_items)   # arm A before any tau^2 training
    E.log("base dev", rec["base_dev"])
    (out / "train_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    try:
        model = FastDecisionModel.get_peft_model(model, r=E.RECIPE["lora_r"], lora_alpha=E.RECIPE["lora_alpha"],
                                                 lora_dropout=E.RECIPE["lora_dropout"], random_state=E.SEED)
    except Exception as e:  # noqa: BLE001 - still before the first optimizer step
        if rec["start"] != "e14_arm_a":
            raise
        rec["start"] = "fresh (LoRA on arm A failed)"
        rec["start_error"] = f"{type(e).__name__}: {e}"[:2000]
        traceback.print_exc()
        model, proc = E.load_model(MAX_LEN)
        items = E.build_items(model, proc, train_rows, "train_fresh", rec)
        dev_items = E.build_items(model, proc, dev_rows, "dev_fresh", rec)
        rec["base_dev_fresh"] = E.metrics(model, proc, dev_items)
        model = FastDecisionModel.get_peft_model(model, r=E.RECIPE["lora_r"], lora_alpha=E.RECIPE["lora_alpha"],
                                                 lora_dropout=E.RECIPE["lora_dropout"], random_state=E.SEED)
    rec["epochs"] = []
    ep1 = E.run_epoch(model, proc, items, out, rec, "epoch1")
    ep1["dev"] = E.metrics(model, proc, dev_items)
    rec["epochs"].append(ep1)
    E.log("epoch 1", {k: ep1[k] for k in ("minutes", "s_per_step", "loss_first_last", "dev")})
    (out / "train_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    snap1 = E.snapshot(model)
    used_h = a.hours_used + (time.time() - t_start) / 3600
    proj = used_h + ep1["minutes"] / 60 + RESERVE_H
    rec["second_epoch"] = {"hours_used": round(used_h, 3), "projection_h": round(proj, 3), "cap_h": a.cap_h,
                           "allowed": proj <= a.cap_h and not a.no_second}
    selected = 1
    if rec["second_epoch"]["allowed"]:
        ep2 = E.run_epoch(model, proc, items, out, rec, "epoch2")
        ep2["dev"] = E.metrics(model, proc, dev_items)
        rec["epochs"].append(ep2)
        E.log("epoch 2", {k: ep2[k] for k in ("minutes", "s_per_step", "loss_first_last", "dev")})
        if ep2["dev"]["accuracy"] > ep1["dev"]["accuracy"]:
            selected = 2
        else:
            E.restore(model, snap1)
    rec["selected_epoch"] = selected
    rec["selection_rule"] = "higher dev accuracy; a tie keeps epoch 1 (registration (e))"
    cal = FastDecisionModel.calibrate(model, proc, dev_items)
    rec["calibration_dev"] = {k: (round(float(v), 5) if isinstance(v, float) else v) for k, v in cal.items()}
    rec["dev_calibrated"] = E.metrics(model, proc, dev_items)
    rec["temperature"] = model.decision_config.get("temperature")
    rec["head_temperature"] = model.decision_config.get("head_temperature")
    model.decision_config["training"] = {"row": "E22", "init": rec["start"], "selected_epoch": selected}
    model.save_pretrained_merged(str(out / "final"), proc)
    rec["final_dir"] = str(out / "final")
    rec["minutes_total"] = round((time.time() - t_start) / 60, 2)
    rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.pop("_mem_idx", None)
    (out / "train_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    E.log("done", out / "final")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--init", required=True, help="E14 arm A's merged checkpoint")
    t.add_argument("--out", required=True)
    t.add_argument("--hours-used", type=float, default=0.0)
    t.add_argument("--cap-h", type=float, default=9.5)
    t.add_argument("--no-second", action="store_true")
    t.set_defaults(fn=cmd_train)
    a = ap.parse_args(argv)
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
    import unsloth  # noqa: F401  -- Unsloth before transformers
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
