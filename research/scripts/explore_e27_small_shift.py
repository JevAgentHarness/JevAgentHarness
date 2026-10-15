#!/usr/bin/env python3
r"""E27 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E27: a 0.8B decision model that
holds under shift"): E14 arm A's 0.8B recipe trained on surface-augmented agent states (S) and on S + E12's public items
(SP); L1, L2 in-domain, L2 shift and L3 shift per arm; the registered selection rule; AgentDecision-0.8B's L3
in-domain, in-domain thresholds and gated final harness (in-domain and on the shift). No test; outside every Holm table.

    # host operations (scripts/aws/e27_small_shift.sh)
    python scripts/explore_e27_small_shift.py probe --log FILE | spend --runs runs --log FILE | errored --run DIR
    python scripts/explore_e27_small_shift.py watch --pid PID --runs runs --cap USD --stop FILE --log FILE [--no-pause]
    python scripts/explore_e27_small_shift.py gate --smoke DIR --runs runs --log FILE --n 200 --cap USD
    python scripts/explore_e27_small_shift.py select --box DIR --runs runs --out FILE      # registration (f)
    # local
    python scripts/explore_e27_small_shift.py fetch                                        # S3 -> box records + runs
    python scripts/explore_e27_small_shift.py cal-fit                                      # (f) 6: fit + profile
    python scripts/explore_e27_small_shift.py archive                                      # slim copies of the live runs
    python scripts/explore_e27_small_shift.py summarize                                    # summary.{md,json} + TeX keys
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import explore_e21_shift as E21  # noqa: E402  (E21's corpus metrics, replay, spend and pause tools)

OUT = ROOT / "results/explore/e27_small_shift"
BOX = OUT / "box"
LIVE = OUT / "live"
CAL_OUT = OUT / "cal"
TEX = ROOT / "paper/generated/e27_small_shift.tex"
RUNS = ROOT / "runs"
S3 = "s3://<bucket>/e27_small_shift"
LABEL = ("exploratory (E27), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-09 'E27: a 0.8B decision model that holds "
         "under shift'; outside every Holm table")
ARMS = {"S": "e27s", "SP": "e27sp"}
NAMES = {"e27s": "0.8B + surface augmentation (S)", "e27sp": "0.8B + surface augmentation + public data (SP)",
         "e14a": "0.8B head (E14 arm A)", "e12": "AgentDecision-4B (E12)", "rl": "Checkpoint before public data",
         "llm_only": "Agent LLM alone (Flash)"}
PREFIX = "e27_"
S23 = ROOT / "results/stage2/s23"
E21_C = ROOT / "results/explore/e21_shift/c"
E21_LIVE = ROOT / "results/explore/e21_shift/live"
E21_LIVE_REF = {"llm_only": ("e21_live_local", "llm_only"), "e14a": ("e21_live_e14a", "e14a_ts_replace_f"),
                "e12": ("e21_live_e12", "e12_ts_replace_f"), "rl": ("e21_live_rl", "rl_ts_replace_f")}
E14_C_ANSWERS = ROOT / "results/explore/e14_small/records/s23/answers_e14a.jsonl.gz"
E1_RUN = "e1_bfcl_when2call_e27"
CAL_SRC = RUNS / "e2a_jevgym_shadow_cal_rl"
CAL_ARM = "rl_shadow_cal"
CAL_WORK = RUNS / "e27_cal"
CAL_NAME = "e27cal_sel"
CAL_REQ_SHA = "d46701e8e9bdc69e6a9e047227eabb1e52578ddc527342dda10b33f7dc5fff5e"
PROFILE = ROOT / "configs/harness/explore_e27_final.yaml"
FINAL_PROFILE = ROOT / "configs/harness/jevagent_final.yaml"
DPS = ("tool_select", "tool_params")
IN_REF = (RUNS / "harness_final_dedicated", "llm_only")
TARIFF_USD_H = 1.624                 # the in-domain final-harness convention (one A10G, g5.4xlarge on demand)
HOST_USD_H = {"g5.2xlarge": 1.212, "g5.4xlarge": 1.624, "g6e.2xlarge": 2.24208, "g6e.4xlarge": 3.00424}
N_BOOT, KEEP_LO = 2000, -0.010
HOLD_L2_MARGIN, HOLD_L3_LO = 0.03, -5.0     # registration (i)
LIVE_ROWS = {"shift_s": ("e27_live_shift_s", "e27s_ts_replace_f"), "shift_sp": ("e27_live_shift_sp", "e27sp_ts_replace_f"),
             "live_in": ("e27_live_in", "e27sel_ts_replace_f"), "final_in": ("e27_final_in", "e27_final_in"),
             "final_shift": ("e27_final_shift", "e27_final_shift")}


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
        print(f"E27 runs: Gemini ${s['s2']:.3f} (+ probes ${pr:.4f}), System-1 ${s['s1']:.3f} in {s['records']} records")
    else:
        print(json.dumps({**s, "probes": pr}, indent=1))
    return 0


def cmd_errored(a: argparse.Namespace) -> int:
    return E21.cmd_errored(a)


def cmd_watch(a: argparse.Namespace) -> int:
    """E21's watchdog over the e27_* runs: the Gemini cap (probes included) and the overload pause."""
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


# ================================================================================================ selection (f)
def l1_records(runs: Path, key: str) -> dict[str, dict[str, Any]]:
    p = runs / E1_RUN / f"s1_{key}.jsonl"
    return {r["id"]: r for r in jl(p)}


def l1_acc(runs: Path, key: str) -> float | None:
    import explore_e9_frontier as E9

    recs = l1_records(runs, key)
    return (sum(float(E9.is_correct(r)) for r in recs.values()) / len(recs)) if recs else None


def l2_shift(cdir: Path) -> dict[str, Any]:
    if not (cdir / "corpus.json").exists():
        return {"rows": {}, "taskacc": {}}
    return E21.l2_rows(cdir)


def select(box: Path, runs: Path) -> dict[str, Any]:
    """The registered rule: higher L2-shift accuracy (task mean, all, main rendering); an exact tie -> higher L1 pooled
    accuracy; then S."""
    l2 = l2_shift(box / "c_shift")
    acc = {arm: (((l2["rows"].get(k) or {}).get("main") or {}).get("all") or {}).get("acc") for arm, k in ARMS.items()}
    l1 = {arm: l1_acc(runs, k) for arm, k in ARMS.items()}
    have = [arm for arm in ARMS if acc[arm] is not None]
    if not have:
        raise SystemExit("no L2-shift answers for either arm")
    best = max(acc[arm] for arm in have)
    tied = [arm for arm in have if acc[arm] == best]
    by = "L2 shift"
    if len(tied) > 1:
        by = "L1 (tie on L2 shift)"
        b1 = max((l1[arm] or -1.0) for arm in tied)
        tied = [arm for arm in tied if (l1[arm] or -1.0) == b1]
        if len(tied) > 1:
            by = "S (tie on L2 shift and L1)"
    sel = "S" if "S" in tied else tied[0]
    return {"selected": sel, "name": "AgentDecision-0.8B", "by": by, "l2_shift_all": acc, "l1_pooled": l1,
            "answers": {arm: ((l2["rows"].get(k) or {}).get("main") or {}).get("answers") for arm, k in ARMS.items()},
            "rule": "PREREG_A7_STAGE2 Deviations 2026-10-09 'E27' (f): higher L2-shift accuracy (task mean, all "
                    "occurrences, main rendering); exact tie -> higher L1 pooled accuracy, then S", "at": _now()}


def cmd_select(a: argparse.Namespace) -> int:
    r = select(Path(a.box), Path(a.runs))
    Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: r[k] for k in ("selected", "by", "l2_shift_all", "l1_pooled")}))
    return 0


# ================================================================================================ thresholds (f) 6
def cal_coverage(path: Path, fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    import explore_e21b_shift_cascade as E21B

    return E21B.cal_coverage(path, fits)


def write_profile(fits: Mapping[str, Mapping[str, Any]], fit_paths: Mapping[str, Path], arm: str) -> dict[str, Any]:
    """jevagent_final.yaml + backends.e27 with the fitted thresholds; an 'always defer' point is not delegated; if the
    tool choice always defers, the final-harness rows equal LLM-only by construction and are not run (registration (f) 6)."""
    import yaml

    prof = yaml.safe_load(FINAL_PROFILE.read_text())
    prof["name"] = "explore_e27_final"
    prof["status"] = "development"
    sec = prof["system2"]["gemini-3.5-flash"]
    defer = {dp: not (isinstance(fits[dp].get("tau_hat"), (int, float)) and math.isfinite(fits[dp]["tau_hat"])
                      and fits[dp]["tau_hat"] <= 1.0) for dp in DPS}
    blk: dict[str, Any] = {}
    for dp in DPS:
        blk[dp] = {"tau": 1.0, "always_defer": True} if defer[dp] else {"tau": float(fits[dp]["tau_hat"])}
    rule = f"E27 (f) 6: AgentDecision-0.8B (arm {arm}) fitted in-domain by the registered command on rl_shadow_cal (seeds 5000-5149)"
    if defer["tool_select"]:
        blk["final_rows"] = "not run: the fit always defers at the tool choice (equals LLM-only by construction)"
    blk["fitted_from"] = {dp: {"path": str(fit_paths[dp].relative_to(ROOT)), "sha256": sha256_file(fit_paths[dp])}
                          for dp in DPS}
    blk["notes"] = rule
    sec["backends"]["e27"] = blk
    prof["description"] = ("Exploratory row E27 (AgentDecision-0.8B, PREREG_A7_STAGE2 Deviations 2026-10-09 'E27' (f) "
                           "6-8): a copy of configs/harness/jevagent_final.yaml (tool + argument cascade, reflection off, "
                           "batch_s1) with backends.e27 = the selected arm's in-domain thresholds. " + rule + ".")
    head = ("# Exploratory row E27's final-harness profile (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 \"E27: a 0.8B decision\n"
            "# model that holds under shift\" (f) 6-8); written by `scripts/explore_e27_small_shift.py cal-fit` from\n"
            "# results/explore/e27_small_shift/cal/fits.json before any E27 final-harness episode: a copy of\n"
            "# configs/harness/jevagent_final.yaml plus backends.e27 (in-domain fitted thresholds; no shifted state used).\n")
    PROFILE.write_text(head + yaml.safe_dump(prof, sort_keys=False, width=118))
    return {"backend": blk, "rule": rule}


def cmd_cal_fit(a: argparse.Namespace) -> int:
    import e5_cal_replay as CR

    box = Path(a.box)
    sel = json.loads((box / "selected.json").read_text())
    ans = box / "cal" / "answers_sel_cal.jsonl"
    req = RUNS / "e21b_cal" / "cal_requests.jsonl.gz"
    if sha256_file(req) != CAL_REQ_SHA:
        raise SystemExit("the calibration questions differ from the registered file")
    answers = jl(ans)
    keys = {q["key"] for q in jl(req)}
    got = {r["key"] for r in answers if r.get("error") is None}
    d = CR.derive(CAL_SRC, CAL_ARM, ans, CAL_WORK, CAL_NAME, model="e27-sel")
    CAL_OUT.mkdir(parents=True, exist_ok=True)
    f = CR.fit(CAL_WORK, CAL_NAME, CAL_OUT)
    paths = {dp: Path(f[dp]["path"]) for dp in DPS}
    for dp in DPS:
        f[dp]["path"] = str(paths[dp].relative_to(ROOT))
    cov = cal_coverage(CAL_WORK / f"{CAL_NAME}.jsonl", f)
    prof = write_profile(f, paths, sel["selected"])
    rec = CAL_OUT / "answers_sel_cal.jsonl.gz"
    with open(rec, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(ans.read_bytes())
    res = {"label": LABEL, "at": _now(), "selected": sel["selected"], "questions": len(keys), "answered": len(got & keys),
           "errors": sum(1 for r in answers if r.get("error")), "derive": d, "fits": f, "coverage_at_tau": cov,
           "profile": {"path": str(PROFILE.relative_to(ROOT)), "sha256": sha256_file(PROFILE), **prof},
           "requests_sha256": CAL_REQ_SHA, "answers": {"path": str(rec.relative_to(ROOT)), "sha256_raw": sha256_file(ans)}}
    (CAL_OUT / "fits.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("selected", "questions", "answered", "errors", "fits", "coverage_at_tau")},
                     indent=1, default=str))
    return 0


# ================================================================================================ records
def cmd_fetch(a: argparse.Namespace) -> int:
    aws = ["env", "-u", "AWS_ACCESS_KEY_ID", "-u", "AWS_SECRET_ACCESS_KEY", "aws", "--region", "us-east-1"]
    env = {**os.environ, "AWS_PAGER": ""}
    subprocess.run(aws + ["s3", "sync", f"{S3}/box", str(BOX), "--only-show-errors", "--exclude", "logs/smi_*"],
                   check=True, env=env)
    subprocess.run(aws + ["s3", "sync", f"{S3}/runs", str(RUNS), "--only-show-errors"], check=True, env=env)
    print(f"-> {BOX}, {RUNS}/e27_*")
    return 0


def cmd_archive(a: argparse.Namespace) -> int:
    done = []
    for p in sorted(RUNS.glob("e27_*")):
        if not p.is_dir() or not any(p.glob("*.jsonl")) or p.name in ("e27_cal", "e27_l2view"):
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
    for arm in ARMS:
        p = box / f"train_record_{arm}.json"
        if p.exists():
            r = json.loads(p.read_text())
            out[arm] = {k: r.get(k) for k in ("build", "train_records", "dev", "base_dev", "calibration_dev",
                                              "minutes_total", "peak_reserved_gb", "gpu", "oom")}
            out[arm]["epochs"] = [{k: e.get(k) for k in ("tag", "minutes", "s_per_step", "steps", "loss_first_last", "dev",
                                                         "peak_alloc_gb")} for e in r.get("epochs", [])]
    for f in ("build_e27.json", "vocab_check.json"):
        if (box / f).exists():
            out[f.split(".")[0]] = json.loads((box / f).read_text())
    return out


def level1(runs: Path) -> dict[str, Any]:
    import explore_e9_frontier as E9
    import explore_startlux_live as SL

    data = E9.load_e1()
    e1 = E9.analyse_e1(data, E9.token_rates(), n_boot=N_BOOT)
    ids = sorted(next(iter(data.values())))
    ref14 = {r["id"]: r for r in E9.read_jsonl(runs / "e1_bfcl_when2call_e14/s1_e14a.jsonl")} \
        if (runs / "e1_bfcl_when2call_e14/s1_e14a.jsonl").exists() else {}
    out: dict[str, Any] = {}
    for arm, key in ARMS.items():
        rel = f"{E1_RUN}/s1_{key}.jsonl"
        E9.EXTRA[key] = (NAMES[key], rel, None)
        x = E9._extra_point(key, e1, data, runs)
        if x.get("available"):
            recs = {r["id"]: r for r in E9.read_jsonl(runs / rel)}
            items = SL.e1_items(runs / rel)
            x["bfcl_tool_choice_acc"] = float(sum(items.values()) / len(items)) if items else None
            if ref14 and set(ref14) >= set(ids):
                x["gap_to_e14a_pt"] = E9.paired_gap([E9.is_correct(recs[i]) for i in ids],
                                                    [E9.is_correct(ref14[i]) for i in ids])
        out[key] = x
    out["refs"] = {m: {"acc": e1["models"][m]["acc"], "usd": e1["models"][m]["usd"]} for m in ("rl", "jev", "sl4b")
                   if m in e1["models"]}
    return out


def level2_in(box: Path) -> dict[str, Any]:
    import ablation_score as AS
    import e5_analysis as E5A

    occ, eligible = AS.load_corpus(S23)
    refs = {"e14a": E14_C_ANSWERS, "rl": S23 / "answers_rl.jsonl"}
    rows: dict[str, Any] = {}
    per: dict[str, Any] = {}
    for name, p in [*((k, box / "s23" / f"answers_{k}.jsonl") for k in ARMS.values()), *refs.items()]:
        ans = {r["qhash"]: r for r in jl(p)}
        if not ans:
            continue
        m, pt = E5A.c_rows(occ, eligible, ans)
        rows[name], per[name] = m, pt
    con = {}
    for k in ARMS.values():
        for r in refs:
            if k in per and r in per:
                con[f"{k}_vs_{r}"] = {st: _boot(per[k][st], per[r][st]) for st in ("all", "offplan")}
    return {"rows": rows, "contrasts": con}


def l2_view(box: Path) -> Path:
    """E21's corpus with E21's logged answers of the reference rows and E27's answers, as links (runs/, untracked)."""
    v = RUNS / "e27_l2view"
    v.mkdir(parents=True, exist_ok=True)
    for f in v.iterdir():
        f.unlink()
    for f in ("corpus.json", "occurrences.jsonl.gz", "questions.jsonl.gz", "questions_nosurf.jsonl.gz"):
        (v / f).symlink_to(E21_C / f)
    for slug in ("rl", "e12", "e14a", "jev"):
        for suf in ("", "__nosurf"):
            p = E21_C / f"answers_{slug}{suf}.jsonl.gz"
            if p.exists():
                (v / p.name).symlink_to(p)
    for key in ARMS.values():
        for suf in ("", "__nosurf"):
            p = box / "c_shift" / f"answers_{key}{suf}.jsonl"
            if p.exists():
                (v / p.name).symlink_to(p.resolve())
    return v


def level2_shift(box: Path) -> dict[str, Any]:
    import explore_common as EC

    v = l2_view(box)
    l2 = E21.l2_rows(v)
    out: dict[str, Any] = {"corpus": l2["corpus"], "rows": {}}
    ta = l2["taskacc"]
    for key in (*ARMS.values(), "e14a", "e12", "rl"):
        m = (l2["rows"].get(key) or {}).get("main")
        if not m:
            continue
        row = {s: (m.get(s) or {}).get("acc") for s in E21.SUBSETS}
        row["n_tasks"] = (m.get("all") or {}).get("tasks")
        row["errors"], row["answers"] = m.get("errors"), m.get("answers")
        row["surface_cost"] = (m.get("cost_surface") or {}).get("delta")
        row["surface_cost_ci"] = [(m.get("cost_surface") or {}).get("lo"), (m.get("cost_surface") or {}).get("hi")]
        row["errors_by_kind"] = m.get("errors_by_kind")
        if key in ARMS.values():
            for ref in ("e14a", "e12", "rl"):
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
    """E21b's per-arm block: success, NI at 3 points (A6.21), cost with the GPU tariff, saving, wall time, coverage."""
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
    blk: dict[str, Any] = {
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
    return blk


def in_ref_episodes() -> dict[str, Mapping[str, Any]]:
    import explore_e21b_shift_cascade as E21B

    p = IN_REF[0] / f"{IN_REF[1]}.jsonl"
    if not p.exists():
        return {}
    ok, _, _ = E21B.arm_records(p)
    return E21B.episodes(ok)


def latency(box: Path) -> dict[str, Any]:
    import e5_analysis as E5A

    labels = [k for k in ("e27sel", "4b") if (box / "latency" / "bench" / f"bench_{k}_new_c1.json").exists()]
    if not labels:
        return {}
    return E5A.latency(box, labels)


def peak_mib(p: Path) -> float | None:
    if not p.exists():
        return None
    vals = []
    for line in p.read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            try:
                vals.append(float(parts[2]))
            except ValueError:
                pass
    return max(vals) if vals else None


def host_hours(box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in ("host.json", "host_final.json"):
        if (box / f).exists():
            out[f] = json.loads((box / f).read_text())
    return out


def spend(box: Path) -> dict[str, Any]:
    runs = {}
    tot = 0.0
    for p in sorted(LIVE.glob("e27_*.json")):
        d = json.loads(p.read_text())
        sp = d.get("spend") or {}
        runs[d["run"]] = {"gemini": sp.get("s2", 0.0), "records": sp.get("records")}
        tot += sp.get("s2", 0.0)
    pr = probe_spend(box / "api_probe_log.txt")
    return {"gemini_runs": tot, "gemini_probes": pr, "gemini": tot + pr, "cap": 30.0, "runs": runs}


def readings(r: Mapping[str, Any]) -> dict[str, Any]:
    sel = (r.get("selection") or {}).get("selected")
    key = ARMS.get(sel or "", None)
    out: dict[str, Any] = {"selected": sel}
    if not key:
        return out
    l2 = ((r.get("l2_shift") or {}).get("rows") or {})
    e12 = (l2.get("e12") or {}).get("all")
    mine = (l2.get(key) or {}).get("all")
    out["l2_holds"] = (mine is not None and e12 is not None and mine >= e12 - HOLD_L2_MARGIN)
    l3 = ((r.get("l3_shift") or {}).get(key) or {}).get("vs_e12")
    out["l3_holds"] = (l3 is not None and l3["ci_pt"][0] >= HOLD_L3_LO)
    out["holds_under_shift"] = bool(out["l2_holds"] and out["l3_holds"])
    c = (((r.get("l2_in") or {}).get("contrasts") or {}).get(f"{key}_vs_e14a") or {}).get("all")
    out["keeps_c"] = (c is not None and c["lo"] >= KEEP_LO)
    fi = (r.get("final_in") or {})
    out["final_in_ni"] = (fi.get("ni") or {}).get("verdict") if fi.get("state") == "ok" else None
    out["keeps_in_domain"] = bool(out["keeps_c"] and (out["final_in_ni"] in (None, "non-inferior")))
    out["rule"] = ("registration (i): holds under shift iff L2-shift accuracy >= AgentDecision-4B's (E12) - 3 points and "
                   "the lower end of the paired Newcombe interval of L3-shift success minus E12's REPLACE row >= -5 "
                   "points; keeps its in-domain level iff C keeps E14 arm A's by E5's rule and (if run) the in-domain "
                   "final harness is non-inferior at 3 points")
    return out


def logged_refs() -> dict[str, Any]:
    """The reference rows' logged in-domain numbers (L1 = E1 pooled, C, live REPLACE on the first 200 Stage-2 tasks):
    E14 arm A (E14), AgentDecision-4B (E12) and the checkpoint before public data (E21's in-domain table)."""
    out: dict[str, Any] = {}
    e14 = json.loads((ROOT / "results/explore/e14_small/summary.json").read_text())
    out["e14a"] = {"l1": e14["static"]["points"]["e14a"]["stats"]["acc"], "c": e14["corpus"]["rows"]["e14a"]["c_acc"],
                   "live": e14["live_a"]["e14a"]["live_success"], "source": "E14"}
    e12 = json.loads((ROOT / "results/explore/e12/summary.json").read_text())
    out["e12"] = {"l1": e12["e1"]["models"]["e12"]["acc"], "c": e12["c"]["e12"]["c_acc"],
                  "live": e12["live"]["e12"]["live_success"], "source": "E12"}
    out["rl"] = {"l1": e12["e1"]["models"]["frozen"]["acc"], "source": "E12's E1 record of the frozen checkpoint"}
    e21 = json.loads((ROOT / "results/explore/e21_shift/summary.json").read_text())
    rl_in = (e21.get("in_domain") or {}).get("rl") or {}
    out["rl"].update({"c": rl_in.get("c"), "live": rl_in.get("live")})
    return out


def summarize(box: Path = BOX, runs: Path = RUNS) -> dict[str, Any]:
    import explore_e21b_shift_cascade as E21B

    res: dict[str, Any] = {"label": LABEL, "generated": _now(), "git": (E21B.git_head() or "?")[:12]}
    res["training"] = training(box)
    res["logged_refs"] = logged_refs()
    res["selection"] = json.loads((box / "selected.json").read_text()) if (box / "selected.json").exists() else None
    res["l1"] = level1(runs)
    res["l2_in"] = level2_in(box)
    res["l2_shift"] = level2_shift(box)
    refs = {name: _slim(run, arm, E21_LIVE) for name, (run, arm) in E21_LIVE_REF.items()}
    res["l3_shift"] = {}
    for arm, key in ARMS.items():
        run, a = LIVE_ROWS[f"shift_{arm.lower()}"]
        d = _slim(run, a)
        if d:
            res["l3_shift"][key] = live_block(d, refs, E21B.kind_of)
            res["l3_shift"][key]["window"] = json.loads((box / f"window_shift_{arm.lower()}.json").read_text()) \
                if (box / f"window_shift_{arm.lower()}.json").exists() else None
    res["l3_shift_refs"] = {name: live_block(d, {}, E21B.kind_of) for name, d in refs.items() if d}
    run, a = LIVE_ROWS["live_in"]
    d = _slim(run, a)
    res["l3_in"] = live_block(d, {}) if d else None
    if res["l3_in"] is not None:
        e14 = json.loads((ROOT / "results/explore/e14_small/summary.json").read_text())
        res["l3_in"]["e14a_logged"] = (e14.get("live_a") or {}).get("e14a")
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
    res["latency"] = latency(box)
    res["memory_mib"] = {"train_" + arm: peak_mib(box / "logs" / f"train_smi_{arm}.csv") for arm in ARMS}
    res["memory_mib"].update({f"serve_{k}": peak_mib(box / "latency" / f"smi_{k}_c1.csv") for k in ("e27sel", "4b")})
    res["host"] = host_hours(box)
    res["spend"] = spend(box)
    res["reading"] = readings(res)
    return res


def _pc(x: Any, d: int = 1) -> str:
    return "–" if x is None else f"{100 * float(x):.{d}f}"


def to_markdown(r: Mapping[str, Any]) -> str:
    L = [f"# E27: a 0.8B decision model that holds under shift", "", f"_{r['label']}_", "",
         f"Generated {r['generated']} (git {r['git']}).", ""]
    tr = r.get("training") or {}
    vc = tr.get("vocab_check") or {}
    if vc:
        L += [f"Vocabulary disjointness: overlap {vc.get('overlap_count')} (augmentation {vc.get('aug_words')} words, "
              f"sha256 {str(vc.get('aug_sha256'))[:16]}; E21 {vc.get('e21_words')} words, sha256 {str(vc.get('e21_sha256'))[:16]}).", ""]
    sel = r.get("selection") or {}
    L += [f"Selected: **{sel.get('selected')}** ({sel.get('by')}).", "",
          "| row | L1 | L2 C | L2 shift all | must-ask | surface cost | L3 shift (n) |", "|---|---|---|---|---|---|---|"]
    for key in (*ARMS.values(), "e14a", "e12", "rl"):
        ref = (r.get("logged_refs") or {}).get(key) or {}
        l1 = ((r.get("l1") or {}).get(key) or {}).get("stats", {}).get("acc", ref.get("l1"))
        c = (((r.get("l2_in") or {}).get("rows") or {}).get(key) or {}).get("c_acc", ref.get("c"))
        s = ((r.get("l2_shift") or {}).get("rows") or {}).get(key) or {}
        l3 = (r.get("l3_shift") or {}).get(key) or (r.get("l3_shift_refs") or {}).get(key) or {}
        L.append(f"| {NAMES.get(key, key)} | {_pc(l1)} | {_pc(c)} | {_pc(s.get('all'))} | {_pc(s.get('mustask'))} | "
                 f"{_pc(s.get('surface_cost'))} | {_pc(l3.get('success'))} ({l3.get('n', '–')}) |")
    L += ["", "## AgentDecision-0.8B", ""]
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


def _kind_split(l3: Mapping[str, Any]) -> dict[str, str | None]:
    """L3 success on E21's three new task kinds and on the generator's kinds (from the per-kind block)."""
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
    b = tr.get("build_e27") or {}
    if b:
        ag = b.get("augmentation") or {}
        k["data/states"] = f"{ag.get('states'):,}".replace(",", "{,}") if ag.get("states") else None
        k["data/augstates"] = f"{ag.get('augmented_states'):,}".replace(",", "{,}") if ag.get("augmented_states") else None
        k["data/s_records"] = f"{b['files']['train_s']['n']:,}".replace(",", "{,}")
        k["data/sp_records"] = f"{b['files']['train_sp']['n']:,}".replace(",", "{,}")
    sel = (r.get("selection") or {}).get("selected")
    for arm, key in ARMS.items():
        slugs = [arm.lower()] + (["sel"] if arm == sel else [])
        l1 = ((r.get("l1") or {}).get(key) or {})
        c = (((r.get("l2_in") or {}).get("rows") or {}).get(key) or {})
        s = ((r.get("l2_shift") or {}).get("rows") or {}).get(key) or {}
        l3 = (r.get("l3_shift") or {}).get(key) or {}
        trn = tr.get(arm) or {}
        st1 = l1.get("stats") or {}
        vals = {"l1/acc": _k(st1.get("acc")), "l1/tool": _k(l1.get("bfcl_tool_choice_acc")),
                "l1/usdk": _k(1000 * st1["usd"], 3, pct=False) if st1.get("usd") else None,
                "l1/lat": _k(st1.get("lat_ms_median"), 0, pct=False),
                "l1/vs_e14a": _k(((l1.get("gap_to_e14a_pt") or {}).get("est")), pct=False, sign=True)
                if l1.get("gap_to_e14a_pt") else None,
                "l2in/vs_e14a_lo": _k(((((r.get("l2_in") or {}).get("contrasts") or {}).get(f"{key}_vs_e14a") or {})
                                       .get("all") or {}).get("lo"), sign=True),
                "l2/mustask_vs_e12": _k((s.get("vs_e12_mustask") or {}).get("delta"), sign=True),
                "l3/own": _k(l3.get("own_acc")), "l3/usd_k": _k(1000 * l3["api_usd"], 2, pct=False)
                if l3.get("api_usd") is not None else None,
                "l3/wall": _k(l3.get("wall_s"), pct=False),
                "train/mem": _k((r.get("memory_mib") or {}).get(f"train_{arm}", 0) / 1024, 1, pct=False)
                if (r.get("memory_mib") or {}).get(f"train_{arm}") else None,
                "l2in/all": _k(c.get("c_acc")), "l2in/offplan": _k(c.get("c_acc_offplan")),
                "l2/all": _k(s.get("all")), "l2/mustask": _k(s.get("mustask")), "l2/offplan": _k(s.get("offplan")),
                "l2/renamed": _k(s.get("renamed")), "l2/newkind": _k(s.get("newkind")),
                "l2/surfcost": _k(s.get("surface_cost"), sign=True),
                "l2/surfcost_abs": _k(abs(s["surface_cost"])) if s.get("surface_cost") is not None else None,
                "l2/vs_e14a": _k((s.get("vs_e14a_all") or {}).get("delta"), sign=True),
                "l3/success": _k(l3.get("success")), "l3/n": str(l3["n"]) if l3.get("n") is not None else None,
                "l3/vs_e12": _k(((l3.get("vs_e12") or {}).get("diff_pt") or 0) / 100, sign=True) if l3.get("vs_e12") else None,
                "l3/vs_e12_lo": _k(((l3.get("vs_e12") or {}).get("ci_pt") or [None])[0], pct=False, sign=True)
                if l3.get("vs_e12") else None,
                "train/minutes": _k(trn.get("minutes_total"), 0, pct=False),
                "train/records": f"{trn['train_records']:,}".replace(",", "{,}") if trn.get("train_records") else None}
        for kind, b in (l3.get("by_kind") or {}).items():
            vals[f"l3/kind/{kind.replace('.', '_')}"] = _k(b.get("success"))
        vals.update(_kind_split(l3))
        for sl in slugs:
            for kk, v in vals.items():
                k[f"{sl}/{kk}"] = v
    for name, ref in (r.get("logged_refs") or {}).items():
        k[f"ref/{name}/l1"] = _k(ref.get("l1"))
        k[f"ref/{name}/c"] = _k(ref.get("c"))
        k[f"ref/{name}/l3in"] = _k(ref.get("live"))
    for name in ("e14a", "e12", "rl", "llm_only"):
        l3 = (r.get("l3_shift_refs") or {}).get(name) or {}
        k[f"ref/{name}/l3"] = _k(l3.get("success"))
        k.update({f"ref/{name}/{kk.replace('l3/', 'l3')}": v for kk, v in _kind_split(l3).items()})
        s = ((r.get("l2_shift") or {}).get("rows") or {}).get(name) or {}
        k[f"ref/{name}/l2"] = _k(s.get("all"))
        k[f"ref/{name}/l2mustask"] = _k(s.get("mustask"))
        k[f"ref/{name}/surfcost"] = _k(s.get("surface_cost"), sign=True)
        k[f"ref/{name}/surfcost_abs"] = _k(abs(s["surface_cost"])) if s.get("surface_cost") is not None else None
    l3i = r.get("l3_in") or {}
    k["sel/l3in/success"] = _k(l3i.get("success"))
    k["sel/l3in/n"] = str(l3i["n"]) if l3i.get("n") is not None else None
    cal = r.get("cal") or {}
    for dp, short in (("tool_select", "tool"), ("tool_params", "args")):
        cv = (cal.get("coverage_at_tau") or {}).get(dp) or {}
        k[f"sel/cal/{short}cov"] = _k(cv.get("coverage"))
    for tag in ("final_in", "final_shift"):
        b = r.get(tag) or {}
        if b.get("state") != "ok":
            continue
        k[f"sel/{tag}/success"] = _k(b["success"])
        k[f"sel/{tag}/ref"] = _k(b["ref_success"])
        k[f"sel/{tag}/n"] = str(b["n"])
        k[f"sel/{tag}/delta"] = _k(b["delta"], sign=True)
        k[f"sel/{tag}/ni_p"] = f"{b['ni']['p']:.3f}"
        k[f"sel/{tag}/saving"] = _k(b["saving_incl_gpu"]["est"])
        k[f"sel/{tag}/wall"] = _k(b["wall_s_mean"], pct=False)
        k[f"sel/{tag}/refwall"] = _k(b["ref_wall_s_mean"], pct=False)
        k[f"sel/{tag}/toolcov"] = _k(b["tool_coverage"])
        k[f"sel/{tag}/argcov"] = _k(b["arg_coverage"])
        k[f"sel/{tag}/usd_k"] = _k(1000 * b["usd_incl_gpu"], 2, pct=False)
        k[f"sel/{tag}/refusd_k"] = _k(1000 * b["ref_usd"], 2, pct=False)
        k[f"sel/{tag}/handed"] = _k(1 - b["tool_coverage"]) if b.get("tool_coverage") is not None else None
        k[f"sel/{tag}/saving_api"] = _k((b.get("saving_api") or {}).get("est"))
        k[f"sel/{tag}/ni"] = (b.get("ni") or {}).get("verdict")
        k[f"sel/{tag}/errored"] = str(b.get("errored", 0))
        k[f"sel/{tag}/wins"], k[f"sel/{tag}/losses"] = str(b["wins"]), str(b["losses"])
        k[f"sel/{tag}/wall_median"] = _k(b.get("wall_s_median"), pct=False)
        k[f"sel/{tag}/refwall_median"] = _k(b.get("ref_wall_s_median"), pct=False)
        dec = (b.get("decisions") or {})
        for dp, short in (("tool_select", "tool"), ("tool_params", "args")):
            d = dec.get(dp) or {}
            k[f"sel/{tag}/{short}wrong"] = str(d["accepted_wrong"]) if d.get("accepted_wrong") is not None else None
            k[f"sel/{tag}/{short}taken"] = f"{d['taken']:,}".replace(",", "{,}") if d.get("taken") is not None else None
        ma = (dec.get("tool_select") or {}).get("mustask") or {}
        k[f"sel/{tag}/mustask_taken"] = str(ma["taken"]) if ma.get("taken") is not None else None
        k[f"sel/{tag}/mustask_wrong"] = str(ma["accepted_wrong"]) if ma.get("accepted_wrong") is not None else None
    rd = r.get("reading") or {}
    for f in ("l2_holds", "l3_holds", "holds_under_shift", "keeps_c", "keeps_in_domain"):
        if f in rd:
            k[f"sel/reading/{f.replace('_', '')}"] = "yes" if rd[f] else "no"
    for lab, x in (r.get("latency") or {}).items():
        if not isinstance(x, Mapping) or "bench" not in x:
            continue
        a = {"e27sel": "sel", "4b_end": "4bend"}.get(lab, lab)
        for c in ("c1", "c4"):
            b = ((x.get("bench") or {}).get(c)) or {}
            k[f"lat/{a}/{c}/p50"] = _k(b.get("p50_ms"), 0, pct=False)
            k[f"lat/{a}/{c}/p90"] = _k(b.get("p90_ms"), 0, pct=False)
            k[f"lat/{a}/{c}/step"] = _k(b.get("s1_ms_per_step_p50"), 0, pct=False)
            k[f"lat/{a}/{c}/speedup"] = _k((x.get("speedup_vs_4b") or {}).get(f"{c}_p50"), 1, pct=False)
        k[f"lat/{a}/mem"] = _k(x["serve_peak_mib"] / 1024, 1, pct=False) if x.get("serve_peak_mib") else None
    sp = r.get("spend") or {}
    k["spend/gemini"] = _k(sp.get("gemini"), 2, pct=False)
    return {kk: v for kk, v in k.items() if v is not None}


def to_tex(r: Mapping[str, Any]) -> str:
    head = ["% AUTO-GENERATED by scripts/explore_e27_small_shift.py -- do not edit by hand.",
            "% E27: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 'E27').",
            f"% generated {r['generated']} at git {r['git']}", r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{e27/{kk}}}{{{v}}}" for kk, v in sorted(tex_keys(r).items())]
    body += [f"\\defres{{{kk}}}{{{v}}}" for kk, v in sorted(e9_keys(r).items())]
    return "\n".join(head + body + [r"\makeatother", ""])


def e9_keys(r: Mapping[str, Any]) -> dict[str, str]:
    """E9-style L1 points (Figure 4a's coordinates, E9's cost rule): e9/e27s/..., e9/e27sp/... and, for the selected arm,
    e9/e27sel/... (the same values), written as E14 writes e9/e14a/..."""
    import explore_e9_frontier as E9

    sel = (r.get("selection") or {}).get("selected")
    out: dict[str, str] = {}
    for arm, key in ARMS.items():
        x = (r.get("l1") or {}).get(key) or {}
        if not x.get("available"):
            continue
        v = E9._point_keys(key, x)
        out.update(v)
        if arm == sel:
            out.update({kk.replace(f"e9/{key}/", "e9/e27sel/", 1): vv for kk, vv in v.items()})
    return out


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
    p.add_argument("--stop", default="runs/e27_STOP.json")
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
    p = sp.add_parser("select")
    p.add_argument("--box", default=str(BOX))
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_select)
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
