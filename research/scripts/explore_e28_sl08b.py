#!/usr/bin/env python3
r"""E28 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E28: our recipe on
StartLux-Decision-0.8B"): E12c's recipe (SFT, fresh LoRA) on StartLux-Decision-0.8B, trained on E12c's mix with E27's
surface augmentation on a seeded half of its agent states and E12's public items at 1 : 1; L1, L2 in-domain, L2 shift,
L3 shift, L3 in-domain; the untrained StartLux-0.8B on the shift (L2, L3) as reference; in-domain thresholds and the
gated final harness (in-domain and, caps allowing, on the shift). No test; outside every Holm table.

    # host operations (scripts/aws/e28_sl08b.sh)
    python scripts/explore_e28_sl08b.py build --e12-mix MIX --out DIR                    # (d): CPU; refuses on any overlap
    python scripts/explore_e28_sl08b.py probe --log FILE | spend --runs runs --log FILE | errored --run DIR
    python scripts/explore_e28_sl08b.py watch --pid PID --runs runs --cap USD --stop FILE --log FILE [--no-pause]
    python scripts/explore_e28_sl08b.py gate --smoke DIR --runs runs --log FILE --n 200 --cap USD
    # local
    python scripts/explore_e28_sl08b.py fetch                                            # S3 -> box records + runs
    python scripts/explore_e28_sl08b.py cal-fit                                          # (f) 7: fit + profile
    python scripts/explore_e28_sl08b.py archive                                          # slim copies of the live runs
    python scripts/explore_e28_sl08b.py summarize                                        # summary.{md,json} + TeX keys
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import explore_e21_shift as E21  # noqa: E402  (E21's corpus metrics, replay, spend and pause tools)

OUT = ROOT / "results/explore/e28_sl08b"
BOX = OUT / "box"
LIVE = OUT / "live"
CAL_OUT = OUT / "cal"
TEX = ROOT / "paper/generated/e28_sl08b.tex"
RUNS = ROOT / "runs"
S3 = "s3://<bucket>/e28_sl08b"
LABEL = ("exploratory (E28), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-09 'E28: our recipe on "
         "StartLux-Decision-0.8B'; outside every Holm table")
KEY, BASE_KEY = "e28", "sl08b"
NAMES = {"e28": "StartLux-0.8B (mixed), E28", "sl08b": "StartLux-0.8B (untrained)", "e27sp": "E27's SP (Qwen3.5-0.8B head)",
         "e12": "AgentDecision-4B (E12)", "e12c": "StartLux-4B (mixed), E12c", "rl": "Checkpoint before public data",
         "e14a": "0.8B head (E14 arm A)", "sl4b": "StartLux-4B (untrained)", "llm_only": "Agent LLM alone (Flash)",
         "jev": "Jev"}
PREFIX = "e28_"
SEED = 0                                    # E12c's trainer seed and E12's mix shuffle
BASE_REPO = "startlux-models/StartLux-Decision-0.8B"
BASE_REV = "bd4f76a600e23227547fee7bfc1825e12a32764c"
S23 = ROOT / "results/stage2/s23"
E21_C = ROOT / "results/explore/e21_shift/c"
E21_LIVE = ROOT / "results/explore/e21_shift/live"
E27_DIR = ROOT / "results/explore/e27_small_shift"
E21_LIVE_REF = {"llm_only": ("e21_live_local", "llm_only"), "e14a": ("e21_live_e14a", "e14a_ts_replace_f"),
                "e12": ("e21_live_e12", "e12_ts_replace_f"), "rl": ("e21_live_rl", "rl_ts_replace_f"),
                "e12c": ("e21_live_e12c", "e12c_ts_replace_f"), "sl4b": ("e21_live_sl4b", "sl4b_ts_replace_f")}
E27_LIVE_REF = {"e27sp": ("e27_live_shift_sp", "e27sp_ts_replace_f")}
E27_IN_REF = ("e27_live_in", "e27sel_ts_replace_f")          # E27's selected arm (SP) in-domain REPLACE
E27_FINAL_REF = {"final_in": ("e27_final_in", "e27_final_in"), "final_shift": ("e27_final_shift", "e27_final_shift")}
E1_RUN = "e1_bfcl_when2call_e28"
E1_VENDOR_RUN = "e1_bfcl_when2call_e28_vendor"   # amendment "E28: L1 cost on the untrained model's serving engine"
E1_REFS = {"e27sp": "e1_bfcl_when2call_e27/s1_e27sp.jsonl", "e12": "e1_bfcl_when2call_e12/s1_e12_mixed.jsonl",
           "e12c": "e1_bfcl_when2call_e12c/s1_e12c_sl4b.jsonl"}
C_REFS = {"rl": S23 / "answers_rl.jsonl", "sl0.8b": S23 / "answers_sl0.8b.jsonl",
          "e12c": ROOT / "results/explore/e12/box_c/s23/answers_e12c_sl4b.jsonl",
          "e27sp": E27_DIR / "box/s23/answers_e27sp.jsonl"}
CAL_SRC = RUNS / "e2a_jevgym_shadow_cal_rl"
CAL_ARM = "rl_shadow_cal"
CAL_WORK = RUNS / "e28_cal"
CAL_NAME = "e28cal"
CAL_REQ = RUNS / "e21b_cal" / "cal_requests.jsonl.gz"
CAL_REQ_SHA = "d46701e8e9bdc69e6a9e047227eabb1e52578ddc527342dda10b33f7dc5fff5e"
PROFILE = ROOT / "configs/harness/explore_e28_final.yaml"
FINAL_PROFILE = ROOT / "configs/harness/jevagent_final.yaml"
DPS = ("tool_select", "tool_params")
IN_REF = (RUNS / "harness_final_dedicated", "llm_only")
TARIFF_USD_H = 1.624                 # the in-domain final-harness convention (one A10G, g5.4xlarge on demand)
HOST_USD_H = {"g5.2xlarge": 1.212, "g5.4xlarge": 1.624, "g6e.2xlarge": 2.24208, "g6e.4xlarge": 3.00424}
N_BOOT, KEEP_LO = 2000, -0.010
HOLD_L2_MARGIN, HOLD_L3_LO = 0.03, -5.0     # registration (i) = E27 (i)
LIVE_ROWS = {"shift": ("e28_live_shift", "e28_ts_replace_f"), "shift_base": ("e28_live_shift_base", "sl08b_ts_replace_f"),
             "live_in": ("e28_live_in", "e28in_ts_replace_f"), "final_in": ("e28_final_in", "e28_final_in"),
             "final_shift": ("e28_final_shift", "e28_final_shift")}
GEMINI_CAP = 30.0


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p: Path) -> list[dict[str, Any]]:
    return E21.jl(p) if p.exists() else []


def read_jl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jl_gz(rows: Sequence[Mapping[str, Any]], p: Path) -> int:
    """Deterministic gzip (mtime 0), so the sha256 of a rebuilt file is the same."""
    with open(p, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r) + "\n").encode())
    return len(rows)


# ================================================================================================ data (d)
def split_mix(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """E12's mix: the SFT v1 replay rows (E12c's agent states) and the public items (meta.e12 a family name)."""
    agent = [dict(r) for r in rows if (r.get("meta") or {}).get("e12") == "replay_v1"]
    pub = [dict(r) for r in rows if (r.get("meta") or {}).get("e12") not in (None, "replay_v1")]
    if len(agent) + len(pub) != len(rows):
        raise SystemExit("E12's mix holds rows that are neither replay nor public items")
    return agent, pub


def build(e12_mix: Path, out: Path, corpus: Path = E21_C) -> dict[str, Any]:
    """Registration (d): E27's vocabulary check and seed guard (imported unchanged), E27's surface augmentation on the
    agent states (a seeded half), E12's public items unchanged, 1 : 1, shuffled with seed 0."""
    import e27_clef as T
    from jevharness.training import surface_aug as SA

    out.mkdir(parents=True, exist_ok=True)
    vc = T.vocab_check(corpus)
    (out / "vocab_check.json").write_text(json.dumps(vc, indent=1) + "\n")
    if not vc["ok"]:
        raise SystemExit(f"vocabulary overlap {vc['overlap_count']}: no training file is written")
    got = sha256_file(e12_mix)
    if got != T.E12_MIX_SHA:
        raise SystemExit(f"E12 train_mix sha256 {got} != {T.E12_MIX_SHA}")
    rows = read_jl(e12_mix)
    agent, pub = split_mix(rows)
    bad = T.seed_guard(agent)
    if bad:
        raise SystemExit(f"agent rows on guarded seeds: {bad[:5]}")
    aug, st = SA.augment_rows(agent)
    if T.seed_guard(aug):
        raise SystemExit("augmented rows on guarded seeds")
    if [r["id"] for r in aug] != [r["id"] for r in agent]:
        raise SystemExit("the augmentation changed the row ids")
    mix = aug + pub
    random.Random(SEED).shuffle(mix)
    n = write_jl_gz(mix, out / "train_mix_e28.jsonl.gz")
    info: dict[str, Any] = {
        "label": LABEL, "registered": "PREREG_A7_STAGE2 Deviations 2026-10-09 'E28' (d)",
        "e12_mix": {"sha256": got, "rows": len(rows)}, "agent_rows": len(agent), "public_rows": len(pub),
        "ratio_public_to_agent": len(pub) / len(agent) if agent else None,
        "public_by_family": dict(Counter(r["meta"]["e12"] for r in pub)),
        "agent_by_dp": dict(Counter(r.get("dp") for r in agent)), "seed_guard": "ok", "augmentation": st,
        "vocab_check": {k: vc[k] for k in ("aug_words", "e21_words", "overlap_count", "aug_sha256", "e21_sha256", "ok")},
        "mix": {"rows": n, "shuffle_seed": SEED, "file": "train_mix_e28.jsonl.gz",
                "sha256": sha256_file(out / "train_mix_e28.jsonl.gz")},
        "augmentation_module_sha256": sha256_file(ROOT / "src/jevharness/training/surface_aug.py"),
        "built": _now()}
    (out / "build_e28.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def cmd_build(a: argparse.Namespace) -> int:
    info = build(Path(a.e12_mix), Path(a.out), Path(a.corpus))
    print(json.dumps({k: info[k] for k in ("agent_rows", "public_rows", "vocab_check", "mix")}, indent=1))
    print(json.dumps({k: info["augmentation"][k] for k in ("states", "augmented_states", "augmented_rows")}))
    return 0


# ================================================================================================ vendor-engine timing
def merge_vendor(base: Path, adapter: Path, out: Path) -> dict[str, Any]:
    """Amendment "E28: L1 cost on the untrained model's serving engine" (c) 1: the LoRA merged into the base's ORIGINAL
    checkpoint layout (the vendor's server loads it as it loads the base). Every base tensor is kept; each targeted weight
    W becomes W + scale * B A (scale alpha / r, or alpha / sqrt(r) with rsLoRA), in float32, stored in W's dtype. Adapter
    modules `model.layers.*` map to `model.language_model.layers.*`; every module must map to exactly one base weight of
    the same shape and every base file is rewritten; everything else of the base directory is copied unchanged."""
    import re
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("modules_to_save"):
        raise SystemExit("DoRA / modules_to_save adapters are not merged here")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    ad = load_file(str(adapter / "adapter_model.safetensors"))
    mods: dict[str, dict[str, Any]] = {}
    for k, v in ad.items():
        m = re.match(r"^(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
        if not m:
            raise SystemExit(f"unexpected adapter tensor {k}")
        mods.setdefault(m.group(1), {})[m.group(2)] = v
    idx = base / "model.safetensors.index.json"
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else \
        sorted(f.name for f in base.glob("*.safetensors"))
    tensors = {f: load_file(str(base / f)) for f in files}
    where = {k: f for f, t in tensors.items() for k in t}
    merged = 0
    for mod, ab in sorted(mods.items()):
        if set(ab) != {"A", "B"}:
            raise SystemExit(f"{mod}: A and B expected")
        cands = [c for c in (f"{mod}.weight", mod.replace("model.", "model.language_model.", 1) + ".weight") if c in where]
        if len(cands) != 1:
            raise SystemExit(f"{mod}: {len(cands)} base weights match")
        key = cands[0]
        w = tensors[where[key]][key]
        delta = (ab["B"].float() @ ab["A"].float()) * scale
        if tuple(delta.shape) != tuple(w.shape):
            raise SystemExit(f"{mod}: shape {tuple(delta.shape)} != {tuple(w.shape)}")
        tensors[where[key]][key] = (w.float() + delta).to(w.dtype).contiguous()
        merged += 1
    if merged != len(mods):
        raise SystemExit("not every adapter module was merged")
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.name.endswith(".safetensors") or f.name == ".cache":
            continue
        dst = out / f.name
        if f.is_dir():
            shutil.copytree(f, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(f, dst)
    sha = {}
    for f, t in tensors.items():
        save_file(t, str(out / f), metadata={"format": "pt"})
        sha[f] = sha256_file(out / f)
    rec = {"base": str(base), "adapter": str(adapter), "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
           "r": r, "alpha": alpha, "rslora": bool(cfg.get("use_rslora")), "scale": scale, "modules_merged": merged,
           "base_tensors": len(where), "merged_sha256": sha, "at": _now(),
           "rule": "amendment 'E28: L1 cost on the untrained model's serving engine' (c) 1"}
    (out / "e28_merge.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def cmd_merge_vendor(a: argparse.Namespace) -> int:
    rec = merge_vendor(Path(a.base), Path(a.adapter), Path(a.out))
    print(json.dumps({k: rec[k] for k in ("scale", "modules_merged", "base_tensors", "merged_sha256")}))
    return 0


# ================================================================================================ live operations
def probe_spend(log: Path | None) -> float:
    return E21._probe_usd(Path(log)) if log else 0.0


def _e21_args(**kw: Any) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def cmd_probe(a: argparse.Namespace) -> int:
    E21.PREFIX = PREFIX
    return E21.cmd_probe(_e21_args(log=a.log, dotenv=a.dotenv))


def cmd_spend(a: argparse.Namespace) -> int:
    E21.PREFIX = PREFIX
    s = E21.total_spend(Path(a.runs))
    pr = probe_spend(Path(a.log) if a.log else None)
    if a.short:
        print(f"E28 runs: Gemini ${s['s2']:.3f} (+ probes ${pr:.4f}), System-1 ${s['s1']:.3f} in {s['records']} records")
    else:
        print(json.dumps({**s, "probes": pr}, indent=1))
    return 0


def cmd_errored(a: argparse.Namespace) -> int:
    return E21.cmd_errored(a)


def cmd_watch(a: argparse.Namespace) -> int:
    """E21's watchdog over the e28_* runs: the Gemini cap (probes included) and the overload pause."""
    E21.PREFIX = PREFIX
    return E21.cmd_watch(_e21_args(pid=a.pid, runs=a.runs, cap=a.cap, extra=probe_spend(Path(a.log) if a.log else None),
                                   s1_cap=None, s1_extra=0.0, s1_arm=None, stop=a.stop, every=a.every,
                                   no_pause=a.no_pause))


def cmd_gate(a: argparse.Namespace) -> int:
    """Registration (g): spend so far (probes included) + 1.2 x the smoke's $ per episode x n <= the cap, else the first
    k tasks that fit."""
    E21.PREFIX = PREFIX
    return E21.cmd_gate(_e21_args(smoke=a.smoke, runs=a.runs, n=a.n, arms=1, cap=a.cap,
                                  extra=probe_spend(Path(a.log) if a.log else None)))


# ================================================================================================ thresholds (f) 7
def write_profile(fits: Mapping[str, Mapping[str, Any]], fit_paths: Mapping[str, Path]) -> dict[str, Any]:
    """jevagent_final.yaml + backends.e28 with the fitted thresholds; an 'always defer' point is not delegated; if the
    tool choice always defers, the final-harness rows equal LLM-only by construction and are not run (registration (f) 7)."""
    import yaml

    prof = yaml.safe_load(FINAL_PROFILE.read_text())
    prof["name"] = "explore_e28_final"
    prof["status"] = "development"
    sec = prof["system2"]["gemini-3.5-flash"]
    defer = {dp: not (isinstance(fits[dp].get("tau_hat"), (int, float)) and math.isfinite(fits[dp]["tau_hat"])
                      and fits[dp]["tau_hat"] <= 1.0) for dp in DPS}
    blk: dict[str, Any] = {}
    for dp in DPS:
        blk[dp] = {"tau": 1.0, "always_defer": True} if defer[dp] else {"tau": float(fits[dp]["tau_hat"])}
    rule = "E28 (f) 7: StartLux-0.8B (mixed) fitted in-domain by the registered command on rl_shadow_cal (seeds 5000-5149)"
    if defer["tool_select"]:
        blk["final_rows"] = "not run: the fit always defers at the tool choice (equals LLM-only by construction)"
    blk["fitted_from"] = {dp: {"path": str(fit_paths[dp].relative_to(ROOT)), "sha256": sha256_file(fit_paths[dp])}
                          for dp in DPS}
    blk["notes"] = rule
    sec["backends"]["e28"] = blk
    prof["description"] = ("Exploratory row E28 (StartLux-0.8B (mixed), PREREG_A7_STAGE2 Deviations 2026-10-09 'E28' (f) "
                           "7-9): a copy of configs/harness/jevagent_final.yaml (tool + argument cascade, reflection off, "
                           "batch_s1) with backends.e28 = its in-domain thresholds. " + rule + ".")
    head = ("# Exploratory row E28's final-harness profile (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 \"E28: our recipe on\n"
            "# StartLux-Decision-0.8B\" (f) 7-9); written by `scripts/explore_e28_sl08b.py cal-fit` from\n"
            "# results/explore/e28_sl08b/cal/fits.json before any E28 final-harness episode: a copy of\n"
            "# configs/harness/jevagent_final.yaml plus backends.e28 (in-domain fitted thresholds; no shifted state used).\n")
    PROFILE.write_text(head + yaml.safe_dump(prof, sort_keys=False, width=118))
    return {"backend": blk, "rule": rule}


def cmd_cal_fit(a: argparse.Namespace) -> int:
    import e5_cal_replay as CR
    import explore_e21b_shift_cascade as E21B

    box = Path(a.box)
    ans = box / "cal" / "answers_cal.jsonl"
    if sha256_file(CAL_REQ) != CAL_REQ_SHA:
        raise SystemExit("the calibration questions differ from the registered file")
    answers = jl(ans)
    keys = {q["key"] for q in jl(CAL_REQ)}
    got = {r["key"] for r in answers if r.get("error") is None}
    d = CR.derive(CAL_SRC, CAL_ARM, ans, CAL_WORK, CAL_NAME, model="startlux-0.8b-mixed")
    CAL_OUT.mkdir(parents=True, exist_ok=True)
    f = CR.fit(CAL_WORK, CAL_NAME, CAL_OUT)
    paths = {dp: Path(f[dp]["path"]) for dp in DPS}
    for dp in DPS:
        f[dp]["path"] = str(paths[dp].relative_to(ROOT))
    cov = E21B.cal_coverage(CAL_WORK / f"{CAL_NAME}.jsonl", f)
    prof = write_profile(f, paths)
    rec = CAL_OUT / "answers_cal.jsonl.gz"
    with open(rec, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(ans.read_bytes())
    res = {"label": LABEL, "at": _now(), "questions": len(keys), "answered": len(got & keys),
           "errors": sum(1 for r in answers if r.get("error")), "derive": d, "fits": f, "coverage_at_tau": cov,
           "profile": {"path": str(PROFILE.relative_to(ROOT)), "sha256": sha256_file(PROFILE), **prof},
           "requests_sha256": CAL_REQ_SHA, "answers": {"path": str(rec.relative_to(ROOT)), "sha256_raw": sha256_file(ans)}}
    (CAL_OUT / "fits.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("questions", "answered", "errors", "fits", "coverage_at_tau")}, indent=1,
                     default=str))
    return 0


# ================================================================================================ records
def cmd_fetch(a: argparse.Namespace) -> int:
    aws = ["env", "-u", "AWS_ACCESS_KEY_ID", "-u", "AWS_SECRET_ACCESS_KEY", "aws", "--region", "us-east-1"]
    env = {**os.environ, "AWS_PAGER": ""}
    subprocess.run(aws + ["s3", "sync", f"{S3}/box", str(BOX), "--only-show-errors", "--exclude", "logs/smi_*",
                          "--exclude", "data/*"], check=True, env=env)
    subprocess.run(aws + ["s3", "sync", f"{S3}/runs", str(RUNS), "--only-show-errors"], check=True, env=env)
    print(f"-> {BOX}, {RUNS}/e28_*")
    return 0


def cmd_archive(a: argparse.Namespace) -> int:
    done = []
    for p in sorted(RUNS.glob("e28_*")):
        if not p.is_dir() or not any(p.glob("*.jsonl")) or p.name in ("e28_cal", "e28_l2view"):
            continue
        dst = LIVE / f"{p.name}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"run": p.name, "arms": E21.slim_live(p), "spend": E21.run_spend(p), "written": _now()},
                                  indent=1) + "\n")
        done.append(dst)
    print("\n".join(str(x.relative_to(ROOT)) for x in done))
    return 0


# ================================================================================================ readout
def _boot(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    import ablation_score as AS

    r = AS.paired_boot(a, b, N_BOOT, 0)
    keeps, loses = r["lo"] >= KEEP_LO, r["hi"] < 0
    r["reading"] = ("keeps the reference's accuracy" if keeps else "loses accuracy" if loses else "inconclusive")
    return r


def training(box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in ("build_e28.json", "vocab_check.json", "base_choice.json", "host.json"):
        if (box / f).exists():
            out[f.split(".")[0]] = json.loads((box / f).read_text())
    sel = box / "sel" / "selection.json"
    out["selection"] = json.loads(sel.read_text()) if sel.exists() else None
    tr = box / "train_runs.jsonl"
    out["train_runs"] = jl(tr)
    for f in ("startlux_08b_sha256.txt", "selected_sha256.txt", "e28_sha256.txt"):
        if (box / f).exists():
            out[f.split(".")[0]] = (box / f).read_text().strip().splitlines()
    cal = box / "e28_calibration.json"
    out["temperatures"] = json.loads(cal.read_text()) if cal.exists() else None
    return out


def level1(runs: Path) -> dict[str, Any]:
    import explore_e9_frontier as E9
    import explore_startlux_live as SL

    data = E9.load_e1()
    e1 = E9.analyse_e1(data, E9.token_rates(), n_boot=N_BOOT)
    ids = sorted(next(iter(data.values())))
    rel = f"{E1_RUN}/s1_{KEY}.jsonl"
    E9.EXTRA[KEY] = (NAMES[KEY], rel, None)
    x = E9._extra_point(KEY, e1, data, runs)
    if x.get("available"):
        recs = {r["id"]: r for r in E9.read_jsonl(runs / rel)}
        items = SL.e1_items(runs / rel)
        x["bfcl_tool_choice_acc"] = float(sum(items.values()) / len(items)) if items else None
        mine = [E9.is_correct(recs[i]) for i in ids]
        gaps = {}
        if "sl0.8b" in data and set(data["sl0.8b"]) >= set(ids):
            gaps["sl08b"] = E9.paired_gap(mine, [E9.is_correct(data["sl0.8b"][i]) for i in ids])
        for ref, f in E1_REFS.items():
            if (runs / f).exists():
                rr = {r["id"]: r for r in E9.read_jsonl(runs / f)}
                if set(rr) >= set(ids):
                    gaps[ref] = E9.paired_gap(mine, [E9.is_correct(rr[i]) for i in ids])
        x["gaps_pt"] = gaps
    out: dict[str, Any] = {KEY: x}
    pv = runs / E1_VENDOR_RUN / f"s1_{KEY}v.jsonl"
    if pv.exists():
        recs = {r["id"]: r for r in E9.read_jsonl(pv)}
        if set(recs) >= set(ids):
            s = E9.model_stats([recs[i] for i in ids], "gpu")
            out["vendor"] = {"available": True, "stats": s, "errors": sum(1 for i in ids if recs[i].get("error")),
                             "basis": "the vendor's server (pinned commit) on the merged checkpoint, E9's rule, as "
                                      "e9/sl0.8b; accuracy only a check (the vendor's prompt is not the trained one)"}
        else:
            out["vendor"] = {"available": False, "reason": f"{len(set(recs) & set(ids))} of {len(ids)} answered"}
    else:
        out["vendor"] = {"available": False, "reason": "not run"}
    refs: dict[str, Any] = {m: {"acc": e1["models"][m]["acc"], "usd": e1["models"][m]["usd"],
                                "lat_ms_median": (e1["models"][m].get("lat_ms_median"))}
                            for m in ("rl", "jev", "sl4b", "sl0.8b") if m in e1["models"]}
    for ref, f in E1_REFS.items():
        if (runs / f).exists():
            recs = [r for r in E9.read_jsonl(runs / f)]
            byid = {r["id"]: r for r in recs}
            if set(byid) >= set(ids):
                s = E9.model_stats([byid[i] for i in ids], "gpu")
                items = SL.e1_items(runs / f)
                refs[ref] = {"acc": s["acc"], "usd": s["usd"], "lat_ms_median": s.get("lat_ms_median"),
                             "bfcl_tool_choice_acc": float(sum(items.values()) / len(items)) if items else None}
    if "sl0.8b" in refs:
        items = SL.e1_items(runs / E9.MODELS["sl0.8b"][1])
        refs["sl0.8b"]["bfcl_tool_choice_acc"] = float(sum(items.values()) / len(items)) if items else None
    out["refs"] = refs
    return out


def level2_in(box: Path) -> dict[str, Any]:
    import ablation_score as AS
    import e5_analysis as E5A

    occ, eligible = AS.load_corpus(S23)
    rows: dict[str, Any] = {}
    per: dict[str, Any] = {}
    for name, p in [(KEY, box / "s23" / f"answers_{KEY}.jsonl"), *C_REFS.items()]:
        ans = {r["qhash"]: r for r in jl(p)}
        if not ans:
            continue
        m, pt = E5A.c_rows(occ, eligible, ans)
        rows[name], per[name] = m, pt
    con = {}
    for r in C_REFS:
        if KEY in per and r in per:
            con[f"{KEY}_vs_{r}"] = {st: _boot(per[KEY][st], per[r][st]) for st in ("all", "offplan")}
    return {"rows": rows, "contrasts": con}


def l2_view(box: Path) -> Path:
    """E21's corpus with E21's logged answers of the reference rows, E27's SP answers and E28's answers, as links."""
    v = RUNS / "e28_l2view"
    v.mkdir(parents=True, exist_ok=True)
    for f in v.iterdir():
        f.unlink()
    for f in ("corpus.json", "occurrences.jsonl.gz", "questions.jsonl.gz", "questions_nosurf.jsonl.gz"):
        (v / f).symlink_to(E21_C / f)
    for slug in ("rl", "e12", "e12c", "e14a", "jev", "sl4b"):
        for suf in ("", "__nosurf"):
            p = E21_C / f"answers_{slug}{suf}.jsonl.gz"
            if p.exists():
                (v / p.name).symlink_to(p)
    for src, slug in ((E27_DIR / "box" / "c_shift", "e27sp"), (box / "c_shift", KEY), (box / "c_shift", BASE_KEY)):
        for suf in ("", "__nosurf"):
            p = src / f"answers_{slug}{suf}.jsonl"
            if p.exists():
                (v / p.name).symlink_to(p.resolve())
    return v


def level2_shift(box: Path) -> dict[str, Any]:
    import explore_common as EC

    v = l2_view(box)
    l2 = E21.l2_rows(v)
    out: dict[str, Any] = {"corpus": l2["corpus"], "rows": {}}
    ta = l2["taskacc"]
    for key in (KEY, BASE_KEY, "e27sp", "e12", "e12c", "e14a", "rl", "sl4b"):
        m = (l2["rows"].get(key) or {}).get("main")
        if not m:
            continue
        row = {s: (m.get(s) or {}).get("acc") for s in E21.SUBSETS}
        row["n_tasks"] = (m.get("all") or {}).get("tasks")
        row["errors"], row["answers"] = m.get("errors"), m.get("answers")
        row["latency_ms_median"] = m.get("latency_ms_median")
        row["surface_cost"] = (m.get("cost_surface") or {}).get("delta")
        row["surface_cost_ci"] = [(m.get("cost_surface") or {}).get("lo"), (m.get("cost_surface") or {}).get("hi")]
        if key in (KEY, BASE_KEY):
            for ref in ("e12", "e12c", "e27sp", "sl4b", BASE_KEY):
                if ref == key:
                    continue
                for s in ("all", "mustask"):
                    a, b = ta.get(f"{key}|main|{s}"), ta.get(f"{ref}|main|{s}")
                    if a and b:
                        common = sorted(set(a) & set(b))
                        r = EC.paired_boot([a[t] for t in common], [b[t] for t in common], n_boot=N_BOOT, seed=0)
                        row[f"vs_{ref}_{s}"] = {"delta": r["est"], "ci": r["ci"], "n_tasks": len(common)}
        out["rows"][key] = row
    return out


def _slim(run: str, arm: str, base: Path = LIVE) -> dict[str, Any] | None:
    p = base / f"{run}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["arms"].get(arm)


def live_block(d: Mapping[str, Any], refs: Mapping[str, Mapping[str, Any] | None], kind_of: Any = None) -> dict[str, Any]:
    import explore_common as EC
    import numpy as np

    ok = sorted(k for k, v in d.items() if not v["error"])
    y = {k: float(bool(d[k]["success"])) for k in ok}
    own = [d[k]["own_acc"] for k in ok if d[k].get("own_acc") is not None]
    blk: dict[str, Any] = {"n": len(ok), "errored": len(d) - len(ok), "success": float(np.mean(list(y.values()))) if y else None,
                           "own_acc": float(np.mean(own)) if own else None,
                           "api_usd": float(np.mean([d[k]["cost"] for k in ok])) if ok else None,
                           "wall_s": float(np.mean([d[k]["wall_s"] for k in ok])) if ok else None,
                           "steps": float(np.mean([d[k]["steps"] for k in ok])) if ok else None}
    if kind_of is not None and ok:
        by: dict[str, list[float]] = defaultdict(list)
        for k in ok:
            by[kind_of(k)].append(y[k])
        blk["by_kind"] = {k: {"n": len(v), "success": float(np.mean(v))} for k, v in sorted(by.items())}
    for name, ref in refs.items():
        if not ref:
            continue
        ry = {k: float(bool(v["success"])) for k, v in ref.items() if not v["error"]}
        common = sorted(set(y) & set(ry))
        if common:
            blk[f"vs_{name}"] = {**EC.newcombe([y[k] for k in common], [ry[k] for k in common]), "n": len(common),
                                 "ref_success": float(np.mean([ry[k] for k in common]))}
    return blk


def final_block(path: Path, ref_eps: Mapping[str, Mapping[str, Any]], win: Mapping[str, Any]) -> dict[str, Any]:
    """E21b's / E27's per-arm block: success, NI at 3 points (A6.21), cost with the GPU tariff, saving, wall time,
    coverage."""
    import numpy as np

    import explore_common as C
    import explore_e21b_shift_cascade as E21B

    recs, nerr, nlat = E21B.arm_records(path)
    eps = E21B.episodes(recs)
    common = sorted(set(eps) & set(ref_eps))
    if not common:
        return {"state": "no common tasks", "n_latest": nlat}
    a = {k: np.array([eps[t][k] for t in common], float) for k in ("success", "cost", "wall_s", "steps", "s2_calls",
                                                                    "s1_requests")}
    b = {k: np.array([ref_eps[t][k] for t in common], float) for k in ("success", "cost", "wall_s")}
    q = TARIFF_USD_H / 3600 * a["wall_s"]
    host_h = HOST_USD_H.get(str(win.get("instance_type") or ""), float("nan"))
    qh = host_h / 3600 * a["wall_s"]
    tot = {x: sum(eps[t][x] for t in common) for x in ("tool_decisions", "tool_by_s1", "arg_decisions", "arg_by_s1")}
    nc = C.newcombe(a["success"], b["success"])
    wins, losses = nc["cells"]["only_arm"], nc["cells"]["only_ref"]
    return {
        "state": "ok", "n": len(common), "n_latest": nlat, "errored": nerr, "window": dict(win),
        "success": float(a["success"].mean()), "ref_success": float(b["success"].mean()),
        "delta": nc["diff_pt"] / 100, "newcombe_pt": nc["ci_pt"], "wins": wins, "losses": losses,
        "ni": E21B.ni_block(wins, losses, len(common)),
        "api_usd": float(a["cost"].mean()), "q_usd": float(q.mean()), "usd_incl_gpu": float((a["cost"] + q).mean()),
        "ref_usd": float(b["cost"].mean()),
        "saving_incl_gpu": C.paired_boot(a["cost"] + q, b["cost"], ratio=True),
        "saving_api": C.paired_boot(a["cost"], b["cost"], ratio=True),
        "saving_host_tariff": C.paired_boot(a["cost"] + qh, b["cost"], ratio=True) if math.isfinite(host_h) else None,
        "host_usd_h": host_h if math.isfinite(host_h) else None,
        "wall_s_mean": float(a["wall_s"].mean()), "wall_s_median": float(np.median(a["wall_s"])),
        "ref_wall_s_mean": float(b["wall_s"].mean()), "ref_wall_s_median": float(np.median(b["wall_s"])),
        "wall_diff": C.paired_boot(a["wall_s"], b["wall_s"]),
        "steps": float(a["steps"].mean()), "s2_calls": float(a["s2_calls"].mean()), "s1_requests": float(a["s1_requests"].mean()),
        "tool_coverage": tot["tool_by_s1"] / tot["tool_decisions"] if tot["tool_decisions"] else None,
        "arg_coverage": tot["arg_by_s1"] / tot["arg_decisions"] if tot["arg_decisions"] else None,
        "decisions": E21B.decision_stats([recs[t] for t in common])}


def time_split(path: Path) -> dict[str, Any] | None:
    """Descriptive: where an episode's wall time went (the latest record per task): System-1 (the decision model) and
    System-2 (the agent LLM) seconds per episode, LLM calls and seconds per LLM call. The LLM's own latency drifts with
    the API's load, so wall times of rows run at different times are compared with this split beside them."""
    import numpy as np

    if not path.exists():
        return None
    last: dict[str, dict[str, Any]] = {}
    for r in jl(path):
        if isinstance(r, dict) and "task_id" in r:
            last[r["task_id"]] = r
    rows = [r for r in last.values() if not r.get("error")]
    if not rows:
        return None
    s1 = [sum(float(c.get("latency_ms") or 0) for c in r.get("calls") or [] if c.get("component") == "s1") / 1000 for r in rows]
    s2c = [[float(c.get("latency_ms") or 0) for c in r.get("calls") or [] if c.get("component") != "s1"] for r in rows]
    s2 = [sum(x) / 1000 for x in s2c]
    n2 = sum(len(x) for x in s2c)
    return {"n": len(rows), "s1_s": float(np.mean(s1)), "s2_s": float(np.mean(s2)),
            "s2_calls": n2 / len(rows), "s2_s_per_call": (sum(s2) / n2) if n2 else None,
            "wall_s": float(np.mean([float(r.get("wall_ms") or 0) / 1000 for r in rows]))}


def in_ref_episodes() -> dict[str, Mapping[str, Any]]:
    import explore_e21b_shift_cascade as E21B

    p = IN_REF[0] / f"{IN_REF[1]}.jsonl"
    if not p.exists():
        return {}
    ok, _, _ = E21B.arm_records(p)
    return E21B.episodes(ok)


def logged_refs() -> dict[str, Any]:
    """The logged in-domain numbers of the reference rows: the untrained StartLux-0.8B's live row (the E2 ladder,
    `results/explore/startlux_live`), E12c and AgentDecision-4B (E12), from their summaries."""
    out: dict[str, Any] = {}
    sl = json.loads((ROOT / "results/explore/startlux_live/summary.json").read_text())
    row = (sl.get("rows") or {}).get("sl0.8b") or {}
    out["sl08b"] = {"live": row.get("live_success"), "live_n": row.get("live_n"), "c": row.get("corpus_acc"),
                    "l1": row.get("e1_acc"), "source": "the E2 ladder's summary (results/explore/startlux_live)"}
    e12 = json.loads((ROOT / "results/explore/e12/summary.json").read_text())
    out["e12"] = {"l1": e12["e1"]["models"]["e12"]["acc"], "c": e12["c"]["e12"]["c_acc"],
                  "live": e12["live"]["e12"]["live_success"], "source": "E12"}
    x = e12.get("e12c") or {}
    out["e12c"] = {"l1": (((x.get("e1") or {}).get("models") or {}).get("e12c") or {}).get("acc"),
                   "c": ((x.get("c") or {}).get("e12c") or {}).get("c_acc"),
                   "live": ((x.get("live") or {}).get("e12c") or {}).get("live_success"), "source": "E12c (E12's summary)"}
    return out


def host_hours(box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in ("host.json", "host_final.json"):
        if (box / f).exists():
            out[f] = json.loads((box / f).read_text())
    st = box / "stage_times.jsonl"
    out["stages"] = jl(st)
    return out


def spend(box: Path) -> dict[str, Any]:
    runs = {}
    tot = 0.0
    for p in sorted(LIVE.glob("e28_*.json")):
        d = json.loads(p.read_text())
        sp = d.get("spend") or {}
        runs[d["run"]] = {"gemini": sp.get("s2", 0.0), "records": sp.get("records")}
        tot += sp.get("s2", 0.0)
    pr = probe_spend(box / "api_probe_log.txt")
    return {"gemini_runs": tot, "gemini_probes": pr, "gemini": tot + pr, "cap": GEMINI_CAP, "runs": runs}


def readings(r: Mapping[str, Any]) -> dict[str, Any]:
    """Registration (i); a reading whose inputs are missing is None (not run), never "no"."""
    out: dict[str, Any] = {}
    l2 = ((r.get("l2_shift") or {}).get("rows") or {})
    e12 = (l2.get("e12") or {}).get("all")
    mine = (l2.get(KEY) or {}).get("all")
    out["l2_holds"] = None if mine is None or e12 is None else bool(mine >= e12 - HOLD_L2_MARGIN)
    l3 = ((r.get("l3_shift") or {}).get(KEY) or {}).get("vs_e12")
    out["l3_holds"] = None if l3 is None else bool(l3["ci_pt"][0] >= HOLD_L3_LO)
    out["holds_under_shift"] = (None if out["l2_holds"] is None or out["l3_holds"] is None
                                else bool(out["l2_holds"] and out["l3_holds"]))
    c = (((r.get("l2_in") or {}).get("contrasts") or {}).get(f"{KEY}_vs_rl") or {}).get("all")
    out["keeps_c"] = None if c is None else bool(c["lo"] >= KEEP_LO)
    fi = (r.get("final_in") or {})
    out["final_in_ni"] = (fi.get("ni") or {}).get("verdict") if fi.get("state") == "ok" else None
    out["keeps_in_domain"] = (None if out["keeps_c"] is None
                              else bool(out["keeps_c"] and (out["final_in_ni"] in (None, "non-inferior"))))
    out["rule"] = ("registration (i) (= E27 (i)): holds under shift iff L2-shift accuracy >= AgentDecision-4B's (E12) - "
                   "3 points and the lower end of the paired Newcombe interval of L3-shift success minus E12's REPLACE "
                   "row >= -5 points; keeps its in-domain level iff C keeps the frozen 4B's by E5's rule and (if run) "
                   "the in-domain final harness is non-inferior at 3 points")
    return out


def summarize(box: Path = BOX, runs: Path = RUNS) -> dict[str, Any]:
    import explore_e21b_shift_cascade as E21B

    res: dict[str, Any] = {"label": LABEL, "generated": _now(), "git": (E21B.git_head() or "?")[:12],
                           "base": {"repo": BASE_REPO, "revision": BASE_REV, "licence": "CC BY-NC 4.0"}}
    res["training"] = training(box)
    res["logged_refs"] = logged_refs()
    res["l1"] = level1(runs)
    res["l2_in"] = level2_in(box)
    res["l2_shift"] = level2_shift(box)
    refs = {name: _slim(run, arm, E21_LIVE) for name, (run, arm) in E21_LIVE_REF.items()}
    refs.update({name: _slim(run, arm, E27_DIR / "live") for name, (run, arm) in E27_LIVE_REF.items()})
    res["l3_shift"] = {}
    for key, tag in ((KEY, "shift"), (BASE_KEY, "shift_base")):
        run, a = LIVE_ROWS[tag]
        d = _slim(run, a)
        if d:
            res["l3_shift"][key] = live_block(d, refs, E21B.kind_of)
            w = box / f"window_{tag}.json"
            res["l3_shift"][key]["window"] = json.loads(w.read_text()) if w.exists() else None
    res["l3_shift_refs"] = {name: live_block(d, {}, E21B.kind_of) for name, d in refs.items() if d}
    run, a = LIVE_ROWS["live_in"]
    d = _slim(run, a)
    res["l3_in"] = live_block(d, {}) if d else None
    d27 = _slim(*E27_IN_REF, E27_DIR / "live")
    res["l3_in_refs"] = {"e27sp": live_block(d27, {})} if d27 else {}
    res["cal"] = json.loads((CAL_OUT / "fits.json").read_text()) if (CAL_OUT / "fits.json").exists() else None
    for tag in ("final_in", "final_shift"):
        run, a = LIVE_ROWS[tag]
        p = runs / run / f"{a}.jsonl"
        win = json.loads((box / f"window_{tag}.json").read_text()) if (box / f"window_{tag}.json").exists() else {}
        if not p.exists():
            res[tag] = {"state": "not run"}
            continue
        ref = in_ref_episodes() if tag == "final_in" else E21B.ref_episodes()[0]
        res[tag] = final_block(p, ref, win)
        res[tag]["time_split"] = time_split(p)
        res[tag]["ref_time_split"] = time_split(IN_REF[0] / f"{IN_REF[1]}.jsonl" if tag == "final_in" else E21B.E21_REF_RUN)
    res["final_refs"] = {}
    for tag, (run, a) in E27_FINAL_REF.items():           # E27's SP in the same harness (descriptive, its own records)
        p = runs / run / f"{a}.jsonl"
        w = E27_DIR / "box" / f"window_{tag}.json"
        if p.exists():
            ref = in_ref_episodes() if tag == "final_in" else E21B.ref_episodes()[0]
            res["final_refs"][f"e27sp/{tag}"] = final_block(p, ref, json.loads(w.read_text()) if w.exists() else {})
    res["host"] = host_hours(box)
    res["spend"] = spend(box)
    res["reading"] = readings(res)
    return res


def _pc(x: Any, d: int = 1) -> str:
    return "–" if x is None else f"{100 * float(x):.{d}f}"


def to_markdown(r: Mapping[str, Any]) -> str:
    L = ["# E28: our recipe on StartLux-Decision-0.8B", "", f"_{r['label']}_", "",
         f"Generated {r['generated']} (git {r['git']}). Base: {r['base']['repo']} @ {r['base']['revision'][:12]} "
         f"({r['base']['licence']}; the trained adapter is a derivative under the same terms).", ""]
    tr = r.get("training") or {}
    vc = tr.get("vocab_check") or {}
    if vc:
        L += [f"Vocabulary disjointness: overlap {vc.get('overlap_count')} (augmentation {vc.get('aug_words')} words, "
              f"sha256 {str(vc.get('aug_sha256'))[:16]}; E21 {vc.get('e21_words')} words, sha256 {str(vc.get('e21_sha256'))[:16]}).", ""]
    b = tr.get("build_e28") or {}
    if b:
        ag = b.get("augmentation") or {}
        L += [f"Data: {b.get('agent_rows')} agent-state rows ({ag.get('augmented_rows')} augmented on "
              f"{ag.get('augmented_states')} of {ag.get('states')} states) + {b.get('public_rows')} public items = "
              f"{(b.get('mix') or {}).get('rows')} rows (sha256 {str((b.get('mix') or {}).get('sha256'))[:16]}).", ""]
    s = tr.get("selection") or {}
    if s:
        L += [f"Checkpoint selection (development data only): {json.dumps(s.get('selected'))}.", ""]
    L += ["| row | L1 | L1 tool choice | $ / 1k (E9, own stack) | L2 C | L2 shift all | must-ask | surface cost | L3 shift (n) |",
          "|---|---|---|---|---|---|---|---|---|"]
    l1 = r.get("l1") or {}
    for key in (KEY, BASE_KEY, "e27sp", "e12c", "e12"):
        if key == KEY:
            st = (l1.get(KEY) or {}).get("stats") or {}
            a1, tc, usd = st.get("acc"), (l1.get(KEY) or {}).get("bfcl_tool_choice_acc"), st.get("usd")
        else:
            rr = (l1.get("refs") or {}).get("sl0.8b" if key == BASE_KEY else key) or {}
            a1, tc, usd = rr.get("acc"), rr.get("bfcl_tool_choice_acc"), rr.get("usd")
        c = (((r.get("l2_in") or {}).get("rows") or {}).get("sl0.8b" if key == BASE_KEY else key) or {}).get("c_acc")
        sh = ((r.get("l2_shift") or {}).get("rows") or {}).get(key) or {}
        l3 = (r.get("l3_shift") or {}).get(key) or (r.get("l3_shift_refs") or {}).get(key) or {}
        L.append(f"| {NAMES.get(key, key)} | {_pc(a1)} | {_pc(tc)} | {'–' if usd is None else f'{1000 * usd:.4f}'} | "
                 f"{_pc(c)} | {_pc(sh.get('all'))} | {_pc(sh.get('mustask'))} | {_pc(sh.get('surface_cost'))} | "
                 f"{_pc(l3.get('success'))} ({l3.get('n', '–')}) |")
    vd = l1.get("vendor") or {}
    own = ((l1.get(KEY) or {}).get("stats") or {})
    if vd.get("available"):
        vs = vd["stats"]
        L += ["", f"L1 cost bases: **e28/l1/usdk and e9/e28/usdk = the vendor's server on the merged checkpoint** "
              f"(the basis of e9/sl0.8b): ${1000 * vs['usd']:.4f} per 1,000 decisions, median {vs['lat_ms_median']:.0f} ms "
              f"(accuracy of that run, a check only: {_pc(vs['acc'])} %); e28/l1/usdk_own = E28's own stack (serve_jev): "
              f"${1000 * own.get('usd', float('nan')):.4f}, median {own.get('lat_ms_median', float('nan')):.0f} ms. "
              "L1 accuracy (e28/l1/acc, e9/e28/acc) is the own stack's."]
    else:
        L += ["", f"L1 cost basis: the vendor-engine timing did not run ({vd.get('reason')}); e28/l1/usdk is not "
              "written and the own-stack cost is e28/l1/usdk_own / e9/e28/usdk (basis key e9/e28/basis = own)."]
    L += ["", "## StartLux-0.8B (mixed) in the agent", ""]
    l3i = r.get("l3_in") or {}
    L.append(f"- L3 in-domain REPLACE: {_pc(l3i.get('success'))} % (n {l3i.get('n', '–')})")
    for tag in ("final_in", "final_shift"):
        b = r.get(tag) or {}
        if b.get("state") != "ok":
            L.append(f"- {tag}: {b.get('state', 'not run')}")
            continue
        L.append(f"- {tag}: {_pc(b['success'])} % vs {_pc(b['ref_success'])} % LLM-only (n {b['n']}), NI {b['ni']['verdict']} "
                 f"(p {b['ni']['p']:.3f}); saving incl. GPU {_pc(b['saving_incl_gpu']['est'])} %; wall {b['wall_s_mean']:.1f} s "
                 f"vs {b['ref_wall_s_mean']:.1f} s; tool coverage {_pc(b['tool_coverage'])} %, args {_pc(b['arg_coverage'])} %")
        ts, rts = b.get("time_split") or {}, b.get("ref_time_split") or {}
        if ts and rts:
            L.append(f"  - {tag} time split (s per episode): model {ts['s1_s']:.1f}, LLM {ts['s2_s']:.1f} "
                     f"({ts['s2_calls']:.1f} calls, {ts['s2_s_per_call']:.2f} s each); LLM-only reference: LLM "
                     f"{rts['s2_s']:.1f} ({rts['s2_calls']:.1f} calls, {rts['s2_s_per_call']:.2f} s each)")
    rd = r.get("reading") or {}
    L += ["", "## Readings (registration (i))", "", "```", json.dumps(rd, indent=1), "```", "",
          f"Spend: {json.dumps(r.get('spend'), default=str)}", ""]
    return "\n".join(L) + "\n"


def _k(v: Any, d: int = 1, pct: bool = True, sign: bool = False) -> str | None:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    x = 100 * float(v) if pct else float(v)
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def _n(v: Any) -> str | None:
    return None if v is None else f"{int(v):,}".replace(",", "{,}")


def _kind_split(l3: Mapping[str, Any]) -> dict[str, str | None]:
    """L3 success on E21's three new task kinds and on the generator's kinds (from the per-kind block), as E27."""
    import explore_e21b_shift_cascade as E21B

    out: dict[str, str | None] = {}
    bk = l3.get("by_kind") or {}
    for tag, sel in (("new", lambda x: x in E21B.NEW_KINDS), ("gen", lambda x: x not in E21B.NEW_KINDS)):
        rows = [b for kind, b in bk.items() if sel(kind)]
        n = sum(b["n"] for b in rows)
        out[f"l3/{tag}kinds"] = _k(sum(b["n"] * b["success"] for b in rows) / n) if n else None
        out[f"l3/{tag}kinds_n"] = str(n) if n else None
    return out


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, Any] = {}
    tr = r.get("training") or {}
    vc = tr.get("vocab_check") or {}
    if vc:
        k["vocab/overlap"] = str(vc.get("overlap_count"))
        k["vocab/augwords"] = str(vc.get("aug_words"))
        k["vocab/e21words"] = str(vc.get("e21_words"))
    b = tr.get("build_e28") or {}
    if b:
        ag = b.get("augmentation") or {}
        k["data/states"] = _n(ag.get("states"))
        k["data/augstates"] = _n(ag.get("augmented_states"))
        k["data/agentrows"] = _n(b.get("agent_rows"))
        k["data/augrows"] = _n(ag.get("augmented_rows"))
        k["data/publicrows"] = _n(b.get("public_rows"))
        k["data/rows"] = _n((b.get("mix") or {}).get("rows"))
    sel = (tr.get("selection") or {}).get("selected") or {}
    k["train/epoch"] = str(sel["epoch"]) if sel.get("epoch") is not None else None
    runs = [x for x in tr.get("train_runs") or [] if x.get("exit") == 0]
    k["train/minutes"] = str(runs[-1]["minutes"]) if runs else None
    k["train/setting"] = runs[-1]["setting"].replace("_", r"\_") if runs else None
    # L1
    x = (r.get("l1") or {}).get(KEY) or {}
    st = x.get("stats") or {}
    k["l1/acc"] = _k(st.get("acc"))
    k["l1/tool"] = _k(x.get("bfcl_tool_choice_acc"))
    k["l1/usdk_own"] = _k(1000 * st["usd"], 3, pct=False) if st.get("usd") else None
    k["l1/lat_own"] = _k(st.get("lat_ms_median"), 0, pct=False)
    vd = ((r.get("l1") or {}).get("vendor") or {})
    if vd.get("available"):
        vs = vd["stats"]
        k["l1/usdk"] = _k(1000 * vs["usd"], 3, pct=False) if vs.get("usd") else None
        k["l1/lat"] = _k(vs.get("lat_ms_median"), 0, pct=False)
        k["l1/acc_vendor"] = _k(vs.get("acc"))
        k["l1/usdk_basis"] = "vendor"
    else:
        k["l1/usdk_basis"] = "own"
    for f, a in (st.get("families") or {}).items():
        k[f"l1/acc/{f}"] = _k(a.get("acc"))
    for ref, g in (x.get("gaps_pt") or {}).items():
        ref = "base" if ref == BASE_KEY else ref
        k[f"l1/vs_{ref}"] = _k(g.get("est"), pct=False, sign=True)
        ci = g.get("ci") or [None, None]
        k[f"l1/vs_{ref}_lo"] = _k(ci[0], pct=False, sign=True)
        k[f"l1/vs_{ref}_hi"] = _k(ci[1], pct=False, sign=True)
    for ref, rr in ((r.get("l1") or {}).get("refs") or {}).items():
        slug = "base" if ref == "sl0.8b" else f"ref/{ref}"
        k[f"{slug}/l1/acc"] = _k(rr.get("acc"))
        k[f"{slug}/l1/tool"] = _k(rr.get("bfcl_tool_choice_acc"))
        k[f"{slug}/l1/usdk"] = _k(1000 * rr["usd"], 3, pct=False) if rr.get("usd") else None
        k[f"{slug}/l1/lat"] = _k(rr.get("lat_ms_median"), 0, pct=False)
    # L2 in-domain
    l2in = r.get("l2_in") or {}
    for name, slug in ((KEY, ""), ("sl0.8b", "base/"), ("rl", "ref/rl/"), ("e12c", "ref/e12c/"), ("e27sp", "ref/e27sp/")):
        c = (l2in.get("rows") or {}).get(name) or {}
        k[f"{slug}l2in/all"] = _k(c.get("c_acc"))
        k[f"{slug}l2in/offplan"] = _k(c.get("c_acc_offplan"))
    for ref in C_REFS:
        con = ((l2in.get("contrasts") or {}).get(f"{KEY}_vs_{ref}") or {}).get("all") or {}
        if con:
            slug = "base" if ref == "sl0.8b" else ref
            k[f"l2in/vs_{slug}"] = _k(con.get("delta"), sign=True)
            k[f"l2in/vs_{slug}_lo"] = _k(con.get("lo"), sign=True)
            k[f"l2in/vs_{slug}_hi"] = _k(con.get("hi"), sign=True)
    # L2 shift
    l2 = (r.get("l2_shift") or {}).get("rows") or {}
    for name, slug in ((KEY, ""), (BASE_KEY, "base/"), ("e27sp", "ref/e27sp/"), ("e12", "ref/e12/"),
                       ("e12c", "ref/e12c/"), ("sl4b", "ref/sl4b/"), ("e14a", "ref/e14a/"), ("rl", "ref/rl/")):
        s = l2.get(name) or {}
        for sub, kk in (("all", "all"), ("mustask", "mustask"), ("offplan", "offplan"), ("renamed", "renamed"),
                        ("newkind", "newkind"), ("genkind", "genkind")):
            k[f"{slug}l2/{kk}"] = _k(s.get(sub))
        k[f"{slug}l2/surfcost"] = _k(s.get("surface_cost"), sign=True)
        if name in (KEY, BASE_KEY):
            for ref in ("e12", "e12c", "e27sp", "sl4b", BASE_KEY):
                for sub in ("all", "mustask"):
                    v = s.get(f"vs_{ref}_{sub}")
                    if v:
                        k[f"{slug}l2/vs_{ref}_{sub}"] = _k(v["delta"], sign=True)
    # L3 shift
    for name, slug in ((KEY, ""), (BASE_KEY, "base/")):
        l3 = (r.get("l3_shift") or {}).get(name) or {}
        k[f"{slug}l3/success"] = _k(l3.get("success"))
        k[f"{slug}l3/n"] = str(l3["n"]) if l3.get("n") is not None else None
        k[f"{slug}l3/errored"] = str(l3["errored"]) if l3.get("errored") is not None else None
        k[f"{slug}l3/own"] = _k(l3.get("own_acc"))
        k[f"{slug}l3/usd_k"] = _k(1000 * l3["api_usd"], 2, pct=False) if l3.get("api_usd") is not None else None
        k[f"{slug}l3/wall"] = _k(l3.get("wall_s"), pct=False)
        k.update({f"{slug}{kk}": v for kk, v in _kind_split(l3).items()})
        for ref in ("e12", "e12c", "e27sp", "llm_only", "sl4b", "e14a"):
            v = l3.get(f"vs_{ref}")
            if v:
                k[f"{slug}l3/vs_{ref}"] = _k(v["diff_pt"] / 100, sign=True)
                k[f"{slug}l3/vs_{ref}_lo"] = _k(v["ci_pt"][0], pct=False, sign=True)
                k[f"{slug}l3/vs_{ref}_hi"] = _k(v["ci_pt"][1], pct=False, sign=True)
    for name, l3 in (r.get("l3_shift_refs") or {}).items():
        k[f"ref/{name}/l3/success"] = _k(l3.get("success"))
        k[f"ref/{name}/l3/n"] = str(l3["n"]) if l3.get("n") is not None else None
    # L3 in-domain and the logged in-domain references
    l3i = r.get("l3_in") or {}
    k["l3in/success"] = _k(l3i.get("success"))
    k["l3in/n"] = str(l3i["n"]) if l3i.get("n") is not None else None
    k["l3in/own"] = _k(l3i.get("own_acc"))
    for name, ref in (r.get("logged_refs") or {}).items():
        slug = "base" if name == "sl08b" else f"ref/{name}"
        k[f"{slug}/l3in/success"] = _k(ref.get("live"))
        if name == "e12":
            k["ref/e12/l2in/all"] = _k(ref.get("c"))
        if ref.get("live_n") is not None:
            k[f"{slug}/l3in/n"] = str(ref["live_n"])
    # thresholds and the final harness
    cal = r.get("cal") or {}
    for dp, short in (("tool_select", "tool"), ("tool_params", "args")):
        cv = (cal.get("coverage_at_tau") or {}).get(dp) or {}
        k[f"cal/{short}cov"] = _k(cv.get("coverage"))
    for tag in ("final_in", "final_shift"):
        b = r.get(tag) or {}
        if b.get("state") != "ok":
            k[f"{tag}/state"] = "not run" if b.get("state") in (None, "not run") else str(b.get("state"))
            continue
        k[f"{tag}/success"] = _k(b["success"])
        k[f"{tag}/ref"] = _k(b["ref_success"])
        k[f"{tag}/n"] = str(b["n"])
        k[f"{tag}/delta"] = _k(b["delta"], sign=True)
        k[f"{tag}/ni"] = (b.get("ni") or {}).get("verdict")
        k[f"{tag}/ni_p"] = f"{b['ni']['p']:.3f}"
        k[f"{tag}/saving"] = _k(b["saving_incl_gpu"]["est"])
        k[f"{tag}/saving_api"] = _k((b.get("saving_api") or {}).get("est"))
        k[f"{tag}/wall"] = _k(b["wall_s_mean"], pct=False)
        k[f"{tag}/refwall"] = _k(b["ref_wall_s_mean"], pct=False)
        k[f"{tag}/toolcov"] = _k(b["tool_coverage"])
        k[f"{tag}/argcov"] = _k(b["arg_coverage"])
        k[f"{tag}/usd_k"] = _k(1000 * b["usd_incl_gpu"], 2, pct=False)
        k[f"{tag}/refusd_k"] = _k(1000 * b["ref_usd"], 2, pct=False)
        k[f"{tag}/errored"] = str(b.get("errored", 0))
        for nm, ts in (("", b.get("time_split")), ("ref", b.get("ref_time_split"))):
            if ts:
                k[f"{tag}/{nm}s1s"] = _k(ts["s1_s"], pct=False)
                k[f"{tag}/{nm}s2s"] = _k(ts["s2_s"], pct=False)
                k[f"{tag}/{nm}s2call"] = _k(ts["s2_s_per_call"], 2, pct=False)
        k[f"{tag}/wins"], k[f"{tag}/losses"] = str(b["wins"]), str(b["losses"])
        dec = (b.get("decisions") or {})
        for dp, short in (("tool_select", "tool"), ("tool_params", "args")):
            d = dec.get(dp) or {}
            k[f"{tag}/{short}wrong"] = str(d["accepted_wrong"]) if d.get("accepted_wrong") is not None else None
    for name, l3 in (r.get("l3_in_refs") or {}).items():
        k[f"ref/{name}/l3in/success"] = _k(l3.get("success"))
        k[f"ref/{name}/l3in/n"] = str(l3["n"]) if l3.get("n") is not None else None
    for name, b in (r.get("final_refs") or {}).items():
        if b.get("state") != "ok":
            continue
        k[f"ref/{name}/success"] = _k(b["success"])
        k[f"ref/{name}/ref"] = _k(b["ref_success"])
        k[f"ref/{name}/ni_p"] = f"{b['ni']['p']:.3f}"
        k[f"ref/{name}/saving"] = _k(b["saving_incl_gpu"]["est"])
        k[f"ref/{name}/wall"] = _k(b["wall_s_mean"], pct=False)
        k[f"ref/{name}/toolcov"] = _k(b["tool_coverage"])
    rd = r.get("reading") or {}
    for f in ("l2_holds", "l3_holds", "holds_under_shift", "keeps_c", "keeps_in_domain"):
        if isinstance(rd.get(f), bool):
            k[f"reading/{f.replace('_', '')}"] = "yes" if rd[f] else "no"
    sp = r.get("spend") or {}
    k["spend/gemini"] = _k(sp.get("gemini"), 2, pct=False)
    hf = ((r.get("host") or {}).get("host_final.json") or {})
    k["host/hours"] = _k(hf.get("hours_used"), 2, pct=False)
    return {kk: v for kk, v in k.items() if v is not None}


def e9_keys(r: Mapping[str, Any]) -> dict[str, str]:
    """E9-style L1 point for Figure 4a: e9/e28/acc = the pooled accuracy of E9's 3,991 questions (own serving stack, the
    registered read-out); e9/e28/{usdk,lg,lat} = E9's rule on the vendor-engine run (the basis of e9/sl0.8b) when it
    ran, with e9/e28/{usdk_own,lg_own,lat_own} on the own stack; e9/e28/basis names the cost basis used."""
    import explore_e9_frontier as E9

    x = (r.get("l1") or {}).get(KEY) or {}
    if not x.get("available"):
        return {}
    s = x["stats"]
    out = {f"e9/{KEY}/acc": f"{100 * s['acc']:.1f}", f"e9/{KEY}/usdk_own": E9._sig(1000 * s["usd"]),
           f"e9/{KEY}/lat_own": str(int(round(s["lat_ms_median"])))}
    if s["usd"] > 0:
        out[f"e9/{KEY}/lg_own"] = f"{math.log10(s['usd']):.4f}"
    for f, a in s["families"].items():
        out[f"e9/{KEY}/acc/{f}"] = f"{100 * a['acc']:.1f}"
    vd = (r.get("l1") or {}).get("vendor") or {}
    src = vd["stats"] if vd.get("available") else s
    out[f"e9/{KEY}/basis"] = "vendor" if vd.get("available") else "own"
    out[f"e9/{KEY}/usdk"] = E9._sig(1000 * src["usd"])
    out[f"e9/{KEY}/lat"] = str(int(round(src["lat_ms_median"])))
    if src["usd"] > 0:
        out[f"e9/{KEY}/lg"] = f"{math.log10(src['usd']):.4f}"
    return out


def to_tex(r: Mapping[str, Any]) -> str:
    head = ["% AUTO-GENERATED by scripts/explore_e28_sl08b.py -- do not edit by hand.",
            "% E28: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 'E28').",
            f"% generated {r['generated']} at git {r['git']}", r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{e28/{kk}}}{{{v}}}" for kk, v in sorted(tex_keys(r).items())]
    body += [f"\\defres{{{kk}}}{{{v}}}" for kk, v in sorted(e9_keys(r).items())]
    return "\n".join(head + body + [r"\makeatother", ""])


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if hasattr(o, "item"):
        return o.item()
    return o


def cmd_summarize(a: argparse.Namespace) -> int:
    r = _jsonable(summarize())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (OUT / "summary.md").write_text(to_markdown(r))
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(to_tex(r))
    print(f"-> {OUT / 'summary.md'}, {TEX}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("--e12-mix", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--corpus", default=str(E21_C))
    p.set_defaults(fn=cmd_build)
    p = sp.add_parser("merge-vendor")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge_vendor)
    p = sp.add_parser("probe")
    p.add_argument("--log", required=True)
    p.add_argument("--dotenv", action="store_true")
    p.set_defaults(fn=cmd_probe)
    p = sp.add_parser("spend")
    p.add_argument("--runs", default="runs")
    p.add_argument("--log", default=None)
    p.add_argument("--short", action="store_true")
    p.set_defaults(fn=cmd_spend)
    p = sp.add_parser("errored")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_errored)
    p = sp.add_parser("watch")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--stop", default="runs/e28_STOP.json")
    p.add_argument("--log", default=None)
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--no-pause", action="store_true")
    p.set_defaults(fn=cmd_watch)
    p = sp.add_parser("gate")
    p.add_argument("--smoke", required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--log", default=None)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--cap", type=float, required=True)
    p.set_defaults(fn=cmd_gate)
    p = sp.add_parser("cal-fit")
    p.add_argument("--box", default=str(BOX))
    p.set_defaults(fn=cmd_cal_fit)
    sp.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sp.add_parser("archive").set_defaults(fn=cmd_archive)
    sp.add_parser("summarize").set_defaults(fn=cmd_summarize)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
