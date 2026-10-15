#!/usr/bin/env python3
"""Exploratory row E18-F (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E17 and E18: read-out ablation and a
deferral-aware objective", (c) 4): the deferral target trained INTO a new scoring-head checkpoint as a (K + 1)-th option
``__defer__`` of every choice question, with E14 arm A's recipe (scripts/train/e14_clef.py). Optional, offline only.

    python scripts/train/e18_defer.py build --data D --shadow runs/e2a_jevgym_shadow_cal_rl/rl_shadow_cal.jsonl \
        --ref-answers results/explore/e14_small/records/cal/answers_e14a.jsonl.gz \
        --requests runs/e5_small_inputs/cal_requests.jsonl.gz --out D18          # CPU: records with the option
    python scripts/train/e18_defer.py train --data D18 --out CK --hours-used H --cap-h 8   # GPU (Unsloth venv)
    python scripts/train/e18_defer.py devread --ckpt CK/F/final --data D18 --out R/e18f_dev.json   # GPU

``build`` takes A's records (``train_a.jsonl.gz`` / ``dev_a.jsonl.gz`` of ``e14_clef.py build-a``) and adds the option
with target 0 (the plug-in target is the answer), and turns the 2,453 labelled calibration-state questions
(``training/defer_head.labelled_questions``: gold, the LLM's answer, the reference head's answer) into records with
the harvest's option cap of 52 (gold plus a seeded sample), target ``__defer__`` where d = 1, else one-hot on the
gold; the arguments of one step form one record. Training: Unsloth's from-LM defaults as A, ONE epoch (A's selected
count), no second epoch, temperatures by Unsloth's ``calibrate`` on A's development records carrying the option.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

MAX_OPTIONS = 52
SEED = 0


def log(*a: Any) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _requests(path: Path) -> dict[str, dict[str, Any]]:
    from jevharness.training.defer_head import _jsonl

    return {q["key"]: q for q in _jsonl(path)}


def cal_rows(rows: list[dict[str, Any]], requests: dict[str, dict[str, Any]], max_options: int = MAX_OPTIONS,
             seed: int = SEED) -> list[dict[str, Any]]:
    """Labelled calibration-state questions -> training rows (``training/data.py`` format) with the harvest's cap and
    rendering (tool choices ``name -- description``), the gold as a one-hot target and ``meta.defer`` = d."""
    from jevharness.harness.prompts import Q_TOOL_SELECT
    from jevharness.training.data import cap_options

    out = []
    ordinal: dict[tuple[str, str], int] = {}
    for r in rows:
        q = requests[r["key"]]
        gold = str(r["gold"])
        opts = list(q["options"])
        if gold not in opts:
            continue
        rng = random.Random(f"{seed}:{r['key']}")
        kept = cap_options(opts, gold, max_options, rng)
        if r["dp"] == "tool_select":
            desc = dict(zip(opts, q.get("descriptions") or [""] * len(opts)))
            text = {o: f"{o} -- {desc.get(o) or ''}".rstrip() for o in kept}
            question = q.get("text") or Q_TOOL_SELECT
        else:
            text = {o: o for o in kept}
            question = q["text"]
        rendered = [text[o] for o in kept]
        task, step, _ = r["key"].split("|")
        k = ordinal.get((task, step), 0)          # the harvest's ids: the step's first argument bare, then #1, #2, ...
        if r["dp"] == "tool_params":
            ordinal[(task, step)] = k + 1
        rid = f"{task}/{step}/{r['dp']}" + ("" if r["dp"] == "tool_select" or k == 0 else f"#{k}")
        out.append({"id": rid, "source": "traj:rl_shadow_cal", "dp": r["dp"], "kind": "choice", "state": q["state"],
                    "question": question, "options": rendered,
                    "target": {o: (1.0 if o == text[gold] else 0.0) for o in rendered}, "q_values": None,
                    "meta": {"label_source": "gold", "mode": "shadow", "defer": int(r["d"]), "llm_ok": r["llm_ok"],
                             "ref_ok": r["ref_ok"], "key": r["key"], "jevgym_seed": None}})
    return out


def _records_with_option(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from jevharness.training import clef_data as CD

    recs = CD.rows_to_records(rows)
    by_id = {r["id"]: r for r in rows}
    out = []
    for rec in recs:
        defer = [f"q{k}" for k, rid in enumerate(rec["meta"]["row_ids"]) if (by_id[rid].get("meta") or {}).get("defer")]
        out.append(CD.add_defer_option(rec, defer))
    return out


def cmd_build(a: argparse.Namespace) -> int:
    from e14_clef import read_jl, sha256, write_jl

    from jevharness.training import clef_data as CD
    from jevharness.training.defer_head import labelled_questions

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"registered": "PREREG_A7_STAGE2 Deviations 2026-10-08 'E17 and E18' (c) 4", "files": {},
                            "sources": {"train_a": str(data / "train_a.jsonl.gz"), "shadow": a.shadow,
                                        "ref_answers": a.ref_answers, "requests": a.requests}}
    rows = labelled_questions(Path(a.shadow), Path(a.ref_answers))
    reqs = _requests(Path(a.requests))
    crow = cal_rows(rows, reqs)
    crec = _records_with_option(crow)
    info["labelled"] = {"questions": len(rows), "positives": sum(r["d"] for r in rows),
                        "by_dp": {dp: {"n": sum(1 for r in rows if r["dp"] == dp), "positives": sum(r["d"] for r in rows if r["dp"] == dp)}
                                  for dp in ("tool_select", "tool_params")}, "rows_kept": len(crow), "records": len(crec),
                        "records_with_defer": sum(1 for r in crec if r["meta"].get("defer_qids"))}
    a_recs = [CD.add_defer_option(r) for r in read_jl(data / "train_a.jsonl.gz")]
    train = a_recs + crec
    random.Random(SEED).shuffle(train)
    info["files"]["train_f"] = {"n": write_jl(train, out / "train_f.jsonl.gz"), "a_records": len(a_recs),
                                "cal_records": len(crec), **CD.record_stats(train)}
    dev = [CD.add_defer_option(r) for r in read_jl(data / "dev_a.jsonl.gz")]
    info["files"]["dev_f"] = {"n": write_jl(dev, out / "dev_f.jsonl.gz"), **CD.record_stats(dev)}
    info["files"]["cal_f"] = {"n": write_jl(crec, out / "cal_f.jsonl.gz")}
    for f in sorted(out.glob("*.jsonl.gz")):
        info.setdefault("sha256", {})[f.name] = sha256(f)
    (out / "build_f.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "sha256"}, indent=1))
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    import torch
    from unsloth import FastDecisionModel

    import e14_clef as E

    data, out = Path(a.data), Path(a.out) / "F"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rec: dict[str, Any] = {"arm": "F", "base": E.BASE, "base_revision": E.BASE_REV, "recipe": E.RECIPE,
                           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "hours_used_at_start": a.hours_used, "cap_h": a.cap_h, "gpu": torch.cuda.get_device_name(),
                           "option": "__defer__ on every choice question (E18 (c) 4)"}
    model, proc = E.load_model(E.RECIPE["max_len"])
    rec["head"] = dict(getattr(model.head, "config", {}) or {})
    train_rows = E.clef_rows(E.read_jl(data / "train_f.jsonl.gz"))
    items = E.build_items(model, proc, train_rows, "train", rec)
    dev_items = E.build_items(model, proc, E.clef_rows(E.read_jl(data / "dev_f.jsonl.gz")), "dev_f", rec)
    rec["dev"] = {"kind": "SFT v1 dev (data/train_v1/dev.jsonl.gz) with the option", "records": len(dev_items)}
    rec["train_records"] = len(items)
    rec["base_dev"] = E.metrics(model, proc, dev_items)
    model = FastDecisionModel.get_peft_model(model, r=E.RECIPE["lora_r"], lora_alpha=E.RECIPE["lora_alpha"],
                                             lora_dropout=E.RECIPE["lora_dropout"], random_state=E.SEED)
    rec["epochs"] = []
    ep1 = E.run_epoch(model, proc, items, out, rec, "epoch1")
    ep1["dev"] = E.metrics(model, proc, dev_items)
    rec["epochs"].append(ep1)
    log("epoch 1", {k: ep1[k] for k in ("minutes", "s_per_step", "loss_first_last", "dev")})
    rec["selected_epoch"] = 1
    rec["selection_rule"] = "one epoch, A's selected count; no second epoch (E18 (c) 4)"
    cal = FastDecisionModel.calibrate(model, proc, dev_items)
    rec["calibration_dev"] = {k: (round(float(v), 5) if isinstance(v, float) else v) for k, v in cal.items()}
    rec["temperature"] = model.decision_config.get("temperature")
    rec["head_temperature"] = model.decision_config.get("head_temperature")
    model.decision_config["training"] = {"arm": "E18-F", "base": E.BASE, "base_revision": E.BASE_REV, "selected_epoch": 1}
    model.save_pretrained_merged(str(out / "final"), proc)
    rec["final_dir"] = str(out / "final")
    rec["minutes_total"] = round((time.time() - t_start) / 60, 2)
    rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.pop("_mem_idx", None)
    (out / "train_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    if a.v1_out:   # the development read-outs without the option (like-for-like with A)
        E.v1_eval(model, proc, Path(a.v1_out))
    log("done", out / "final")
    return 0


def devread(engine: Any, records: list[dict[str, Any]], k: dict[str, float]) -> dict[str, Any]:
    """The option's read-out on records: per decision point the coverage of the rule (p_defer <= k), the accuracy
    of the accepted and of the deferred questions, and on questions with a logged LLM answer (``meta.logged``) the
    share of the deferred ones the LLM got right."""
    from jevharness.training.clef_engine import DEFER_LABEL

    per: dict[str, dict[str, Any]] = {}
    for rec in records:
        qs = {qid: {kk: vv for kk, vv in q.items()} for qid, q in rec["questions"].items()
              if q["type"] == "choice"}
        if not qs:
            continue
        plain = {qid: {**q, "criteria": {kk: vv for kk, vv in q["criteria"].items() if kk != DEFER_LABEL}} for qid, q in qs.items()}
        scored = engine.score([(rec["state"], plain)])[0]
        dp = str(rec["meta"].get("dp"))
        kk = float(k.get(dp, 0.0))
        for qid, (ids, probs) in scored.items():
            p = dict(zip(ids, probs))
            pd = p.get(DEFER_LABEL, 0.0)
            real = {o: v for o, v in p.items() if o != DEFER_LABEL}
            best = max(real, key=real.get)
            g = rec["gold"][qid]
            gold_label = max((o for o in g["probabilities"] if o != DEFER_LABEL), key=lambda o: g["probabilities"][o])
            ok = best == gold_label
            r = per.setdefault(dp, {"n": 0, "accepted": 0, "accepted_right": 0, "deferred_right": 0, "p_defer": []})
            r["n"] += 1
            r["p_defer"].append(pd)
            if pd <= kk:
                r["accepted"] += 1
                r["accepted_right"] += int(ok)
            else:
                r["deferred_right"] += int(ok)
    out = {}
    for dp, r in per.items():
        n, acc, dr = r["n"], r["accepted"], r["deferred_right"]
        ps = sorted(r["p_defer"])
        out[dp] = {"n": n, "k": float(k.get(dp, 0.0)), "coverage": acc / n if n else None,
                   "accepted_accuracy": r["accepted_right"] / acc if acc else None,
                   "deferred_accuracy": dr / (n - acc) if n - acc else None,
                   "accuracy_all": (r["accepted_right"] + dr) / n if n else None,
                   "p_defer_quantiles": {q: ps[min(len(ps) - 1, int(q * len(ps)))] for q in (0.05, 0.5, 0.95, 0.99)} if ps else None}
    return out


def cmd_devread(a: argparse.Namespace) -> int:
    from unsloth import FastDecisionModel

    import e14_clef as E

    from jevharness.training.clef_engine import ClefEngine, DeferOption
    from jevharness.training.defer_head import K_MARGIN

    model, proc = FastDecisionModel.from_pretrained(a.ckpt, max_seq_length=16384)
    FastDecisionModel.for_inference(model)
    eng = ClefEngine.from_model(model, proc, name="e18f", max_len=16384, defer=DeferOption(mode="option"))
    res = {"ckpt": a.ckpt, "k": K_MARGIN, "files": {}}
    for name in ("dev_f", "cal_f"):
        p = Path(a.data) / f"{name}.jsonl.gz"
        if p.exists():
            res["files"][name] = devread(eng, E.read_jl(p), K_MARGIN)
            log(name, res["files"][name])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True, help="e14_clef.py build-a output (train_a / dev_a records)")
    b.add_argument("--shadow", default="runs/e2a_jevgym_shadow_cal_rl/rl_shadow_cal.jsonl")
    b.add_argument("--ref-answers", default="results/explore/e14_small/records/cal/answers_e14a.jsonl.gz")
    b.add_argument("--requests", default="runs/e5_small_inputs/cal_requests.jsonl.gz")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--hours-used", type=float, default=0.0)
    t.add_argument("--cap-h", type=float, default=8.0)
    t.add_argument("--v1-out", default=None)
    t.set_defaults(fn=cmd_train)
    d = sub.add_parser("devread")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--data", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_devread)
    a = ap.parse_args(argv)
    if a.cmd in ("train", "devread"):
        import os

        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
        import unsloth  # noqa: F401  -- Unsloth before transformers
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
