#!/usr/bin/env python3
"""Exploratory row E27 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E27: a 0.8B decision model that holds under
shift"; descriptive, outside every Holm table): E14 arm A's recipe (Qwen3.5-0.8B + a fresh Clef joint schema head,
Unsloth 2026.10.2) on surface-augmented agent states (arm S) and on S plus E12's public tool-use items (arm SP).

    python scripts/train/e27_clef.py vocab-check [--corpus results/explore/e21_shift/c] --out FILE   # CPU; exit 1 on overlap
    python scripts/train/e27_clef.py build --out D --e12-mix E12/train_mix.jsonl.gz [--corpus DIR]    # CPU
    python scripts/train/e27_clef.py train --arm S --data D --out CK [--hours-used H] [--v1-out F]     # GPU

Registration (b)-(c): one epoch per arm (arm A's selected count); temperatures by Unsloth's ``calibrate`` on arm A's
development split (SFT v1's dev rows, unaugmented); the model, trainer, epoch and read-out functions are E14's
(``scripts/train/e14_clef.py``), unchanged. The build refuses to write a training file unless the vocabulary check of
(c) 3 finds no overlap.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import e14_clef as E14  # noqa: E402  (E14's recipe, data helpers and read-outs, unchanged)

SEED = E14.SEED                                   # 3407
E12_MIX_SHA = "fcdb3b9a9a1f1194ebf2335251d500a75c37dc75ad5d3cafa1229ce71e539521"
E21_CORPUS = ROOT / "results/explore/e21_shift/c"
E21_MODULES = (ROOT / "src/jevharness/bench/suites/jevgym_shift.py", ROOT / "src/jevharness/bench/suites/jevgym.py")
ARMS = ("S", "SP")


def vocab_check(corpus: Path | None) -> dict[str, Any]:
    from jevharness.training import surface_aug as SA

    aug = SA.aug_vocabulary()
    e21, info = SA.e21_vocabulary(E21_MODULES, corpus)
    res = SA.disjointness(aug, e21)
    res["e21_sources"] = {"modules": [{**m, "path": str(Path(m["path"]).relative_to(ROOT))} for m in info["modules"]],
                          "corpus_files": [{**c, "path": str(Path(c["path"]).relative_to(ROOT))}
                                           for c in info.get("corpus_files", [])],
                          "corpus_texts": info.get("corpus_texts")}
    res["aug_module_sha256"] = E14.sha256(ROOT / "src/jevharness/training/surface_aug.py")
    res["function_words"] = len(SA.FUNCTION_WORDS)
    res["rule"] = ("PREREG_A7_STAGE2 Deviations 2026-10-09 'E27' (c) 3: word tokens (lowercase letter runs) of the "
                   "augmentation's word lists and template literals (fields removed) vs every string literal of "
                   "jevgym_shift.py and jevgym.py plus E21's L2 corpus texts (all renderings), both minus the function "
                   "words; overlap on tokens and on light stems must be 0")
    res["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return res


def cmd_vocab_check(a: argparse.Namespace) -> int:
    res = vocab_check(Path(a.corpus) if a.corpus else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("aug_words", "e21_words", "overlap_count", "aug_sha256", "e21_sha256", "ok")}))
    return 0 if res["ok"] else 1


def seed_guard(rows: list[dict[str, Any]]) -> list[str]:
    """E14's check: every JevGym row's seed is a training seed (training_exclusions incl. 330000-339999, TRAIN_FORBIDDEN)."""
    from jevharness.bench import registry as R
    from jevharness.training.data_v1 import jevgym_seed
    from jevharness.training.seed_guard import TRAIN_FORBIDDEN

    bad = []
    for r in rows:
        s = jevgym_seed(r["id"])
        if s is None:
            continue
        if (s < R.TRAIN_MIN_SEED or R.in_ranges(s, R.training_exclusions() + (R.CALIBRATION_RANGE,))
                or any(lo <= s <= hi for lo, hi in TRAIN_FORBIDDEN)):
            bad.append(r["id"])
    return bad


def public_rows(path: Path, n_target: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Registration (c) 4: E12's 7,000 public items at E12's 1 : 1 ratio to the agent-state rows (two full passes plus a
    seeded sample without replacement of the remainder)."""
    got = E14.sha256(path)
    if got != E12_MIX_SHA:
        raise SystemExit(f"E12 train_mix sha256 {got} != {E12_MIX_SHA}")
    rows = [r for r in E14.read_jl(path) if (r.get("meta") or {}).get("e12") not in (None, "replay_v1")]
    rng = random.Random(SEED)
    k, rem = divmod(n_target, len(rows))
    # each copy gets its own id (a pass suffix), so repeated argument rows never merge into one Clef record
    out = [{**r, "id": f"{r['id']}@e27p{i}"} for i in range(k) for r in rows]
    out += [{**r, "id": f"{r['id']}@e27p{k}"} for r in rng.sample(rows, rem)]
    return out, {"source": str(path), "sha256": got, "public_items": len(rows), "target_rows": n_target,
                 "full_passes": k, "sampled": rem, "by_family": dict(Counter(r["meta"]["e12"] for r in rows))}


def to_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """rows_to_records row by row group; a group that does not convert (e.g. duplicate option labels) is dropped."""
    from jevharness.training import clef_data as CD

    try:
        return CD.rows_to_records(rows), 0
    except ValueError:
        pass
    out, bad = [], 0
    for r in rows:
        try:
            out += CD.rows_to_records([r])
        except ValueError:
            bad += 1
    return out, bad


def cmd_build(a: argparse.Namespace) -> int:
    from jevharness.training import clef_data as CD
    from jevharness.training import surface_aug as SA
    from jevharness.training.data import mix

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vc = vocab_check(Path(a.corpus) if a.corpus else E21_CORPUS)
    (out / "vocab_check.json").write_text(json.dumps(vc, indent=1) + "\n")
    if not vc["ok"]:
        raise SystemExit(f"vocabulary overlap {vc['overlap_count']}: no training file is written")
    rows = mix([(str(ROOT / "data/train_v1/train.jsonl.gz"), 1.0),
                (str(ROOT / "data/train_v1/replay_when2call.jsonl.gz"), 0.45)], seed=0)
    raw = out / "train_mix.jsonl"
    with open(raw, "w") as f:   # E14's build-a check: byte-identical to SFT v1's mix
        for r in rows:
            f.write(json.dumps(r) + "\n")
    got = E14.sha256(raw)
    raw.unlink()
    if got != E14.MIX_SHA:
        raise SystemExit(f"train_mix sha256 {got} != SFT v1's {E14.MIX_SHA}")
    bad = seed_guard(rows)
    if bad:
        raise SystemExit(f"rows on guarded seeds: {bad[:5]}")
    aug_rows, st = SA.augment_rows(rows)
    if seed_guard(aug_rows):
        raise SystemExit("augmented rows on guarded seeds")
    info: dict[str, Any] = {"train_mix_sha256": got, "rows": len(rows), "seed_guard": "ok", "augmentation": st,
                            "vocab_check": {k: vc[k] for k in ("aug_words", "e21_words", "overlap_count", "aug_sha256",
                                                               "e21_sha256", "ok")}, "files": {}}
    rec_s = CD.rows_to_records(aug_rows)
    info["files"]["train_s"] = {"n": E14.write_jl(rec_s, out / "train_s.jsonl.gz"), **CD.record_stats(rec_s)}
    pub, pinfo = public_rows(Path(a.e12_mix), len(rows))
    rec_p, n_bad = to_records(pub)
    pinfo["records"] = len(rec_p)
    pinfo["dropped_not_convertible"] = n_bad
    info["public"] = pinfo
    rec_sp = rec_s + rec_p
    random.Random(SEED).shuffle(rec_sp)
    info["files"]["train_sp"] = {"n": E14.write_jl(rec_sp, out / "train_sp.jsonl.gz"), **CD.record_stats(rec_sp)}
    dv = CD.rows_to_records(E14.read_jl(ROOT / E14.V1["dev"]))
    info["files"]["dev_a"] = {"source": E14.V1["dev"], "source_sha256": E14.sha256(ROOT / E14.V1["dev"]),
                              "n": E14.write_jl(dv, out / "dev_a.jsonl.gz"), **CD.record_stats(dv)}
    for f in sorted(out.glob("*.jsonl.gz")):
        info.setdefault("sha256", {})[f.name] = E14.sha256(f)
    info["built"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "build_e27.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k not in ("files", "sha256")}, indent=1))
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    """E14's cmd_train for one epoch (registration (b)): S or SP records, temperatures on arm A's dev split."""
    import torch
    from unsloth import FastDecisionModel

    arm, data, out = a.arm.upper(), Path(a.data), Path(a.out) / a.arm.upper()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rec: dict[str, Any] = {"arm": arm, "base": E14.BASE, "base_revision": E14.BASE_REV, "recipe": E14.RECIPE,
                           "epochs_registered": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "hours_used_at_start": a.hours_used, "gpu": torch.cuda.get_device_name(),
                           "registered": "PREREG_A7_STAGE2 Deviations 2026-10-09 'E27' (b)-(c)"}
    model, proc = E14.load_model(E14.RECIPE["max_len"])
    rec["head"] = dict(getattr(model.head, "config", {}) or {})
    rec["head_params_m"] = round(sum(p.numel() for p in model.head.parameters()) / 1e6, 2)
    train_rows = E14.clef_rows(E14.read_jl(data / f"train_{arm.lower()}.jsonl.gz"))
    items = E14.build_items(model, proc, train_rows, "train", rec)
    dev_items = E14.build_items(model, proc, E14.clef_rows(E14.read_jl(data / "dev_a.jsonl.gz")), "dev_a", rec)
    rec["dev"] = {"kind": "SFT v1 dev (data/train_v1/dev.jsonl.gz), unaugmented", "records": len(dev_items)}
    rec["train_records"] = len(items)
    rec["base_dev"] = E14.metrics(model, proc, dev_items)
    E14.log("base dev", rec["base_dev"])
    model = FastDecisionModel.get_peft_model(model, r=E14.RECIPE["lora_r"], lora_alpha=E14.RECIPE["lora_alpha"],
                                             lora_dropout=E14.RECIPE["lora_dropout"], random_state=SEED)
    ep1 = E14.run_epoch(model, proc, items, out, rec, "epoch1")
    ep1["dev"] = E14.metrics(model, proc, dev_items)
    rec["epochs"] = [ep1]
    E14.log("epoch 1", {k: ep1[k] for k in ("minutes", "s_per_step", "loss_first_last", "dev")})
    rec["selected_epoch"] = 1
    rec["selection_rule"] = "one epoch (registration (b): arm A's selected epoch count)"
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
    if a.v1_out:
        E14.v1_eval(model, proc, Path(a.v1_out))
    E14.log("done", out / "final")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vocab-check")
    v.add_argument("--corpus", default=str(E21_CORPUS))
    v.add_argument("--out", required=True)
    v.set_defaults(fn=cmd_vocab_check)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--e12-mix", required=True)
    b.add_argument("--corpus", default=str(E21_CORPUS))
    b.set_defaults(fn=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=["S", "SP", "s", "sp"])
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--hours-used", type=float, default=0.0)
    t.add_argument("--v1-out", default=None)
    t.set_defaults(fn=cmd_train)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
        import unsloth  # noqa: F401  -- Unsloth before transformers
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
