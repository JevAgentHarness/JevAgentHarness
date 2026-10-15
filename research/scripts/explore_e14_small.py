#!/usr/bin/env python3
"""Exploratory row E14 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E14: a sub-1B decision model with a scoring
head" and "E14: pause and resume on API overload"; descriptive, outside every Holm table).

Operations (the E14 host, scripts/aws/e14_small.sh; the live stages hold the Gemini key in their environment only):

    python scripts/explore_e14_small.py probe --log results/explore/e14_small/api_probe_log.txt   # exit 0 iff 6/6
    python scripts/explore_e14_small.py watch --pid PID --cap 40 --runs runs --stop runs/e14_STOP.json
    python scripts/explore_e14_small.py errored --run runs/e14_live_a      # tasks whose latest record errored
    python scripts/explore_e14_small.py spend --runs runs [--short]
    python scripts/explore_e14_small.py gate --runs runs --smoke runs/e14_live_smoke_a --n 200 --cap 40

Analysis (local):

    python scripts/explore_e14_small.py fit --arm A          # tau-hat from the calibration-state answers (E5's (g))
    python scripts/explore_e14_small.py prepare-live --arm A # writes configs/harness/explore_e14_final.yaml
    python scripts/explore_e14_small.py summarize            # -> results/explore/e14_small/summary.{md,json} + keys
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

OUT = ROOT / "results/explore/e14_small"
TEX = ROOT / "paper/generated/e14_small.tex"
LABEL = ("exploratory (E14), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-07 'E14: a sub-1B decision model with a "
         "scoring head'; outside every Holm table")
LIVE_RUNS = ("e14_live_smoke_a", "e14_live_a", "e14_live_smoke_b", "e14_live_b", "e14_live_smoke_g", "e14_live_a_g")
OVERLOAD = ("503", "overload", "high demand", "unavailable", "resource_exhausted", "429")
PROBE_N = 6
FLASH = {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0, "reasoning_effort": "minimal"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _jl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    out = []
    with op(p, "rt") as f:
        for x in f:
            if x.strip():
                try:
                    out.append(json.loads(x))
                except json.JSONDecodeError:
                    continue
    return out


# ================================================================================================ operations
def cmd_probe(a: argparse.Namespace) -> int:
    """One round: six one-word Flash calls on the direct API through the harness client (its retries included)."""
    from jevharness.accounting import Ledger
    from jevharness.llm import make_llm

    llm = make_llm(dict(FLASH))
    led = Ledger()
    ok, lat, errs = 0, [], []
    for i in range(PROBE_N):
        t0 = time.perf_counter()
        try:
            llm.chat([{"role": "user", "content": f"Reply with one word: ready ({i})"}], ledger=led,
                     decision_point="probe", max_tokens=8)
            ok += 1
        except Exception as e:  # noqa: BLE001 -- every failure counts against the round
            errs.append(f"{type(e).__name__}: {str(e)[:120]}")
        lat.append(time.perf_counter() - t0)
    lat.sort()
    line = (f"{_now()} direct Gemini 3.5 Flash: {ok}/{PROBE_N} ok, median {lat[len(lat) // 2]:.1f} s, max {lat[-1]:.1f} s"
            + (f"; errors: {errs[0]}" if errs else "") + ("  CLEAN" if ok == PROBE_N else "  UNHEALTHY"))
    p = Path(a.log)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# E14: direct Gemini API probes (Deviations 2026-10-07 'E14: pause and resume on API overload'). "
                     "One round = six one-word Gemini 3.5 Flash calls; start / resume after two consecutive 6/6 rounds.\n")
    with open(p, "a") as f:
        f.write(line.replace("\n", " ") + f"  (logged ${led.summary()['total']['cost_usd']:.5f})\n")
    print(line)
    return 0 if ok == PROBE_N else 1


def episode_cost(t: Mapping[str, Any]) -> float:
    return float(((t.get("usage") or {}).get("total") or {}).get("cost_usd") or 0.0)


def run_spend(run: Path) -> dict[str, Any]:
    """Every record of every arm (errored attempts and re-runs included: all were paid)."""
    out = {"total": 0.0, "records": 0}
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        for t in _jl(p):
            if isinstance(t, dict) and "task_id" in t:
                out["total"] += episode_cost(t)
                out["records"] += 1
    return out


def total_spend(runs: Path) -> dict[str, Any]:
    per = {name: run_spend(runs / name) for name in LIVE_RUNS if (runs / name).is_dir()}
    return {"by_run": per, "total": sum(v["total"] for v in per.values()),
            "records": sum(v["records"] for v in per.values())}


def latest(run: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """arm -> task -> latest record."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        d: dict[str, dict[str, Any]] = {}
        for t in _jl(p):
            if isinstance(t, dict) and "task_id" in t:
                d[str(t["task_id"])] = t
        out[p.stem] = d
    return out


def overload_error(t: Mapping[str, Any]) -> bool:
    e = str(t.get("error") or "").lower()
    return bool(e) and any(k in e for k in OVERLOAD)


def cmd_errored(a: argparse.Namespace) -> int:
    n = sum(1 for d in latest(Path(a.run)).values() for t in d.values() if t.get("error"))
    print(n)
    return 0


def cmd_spend(a: argparse.Namespace) -> int:
    s = total_spend(Path(a.runs))
    print(f"Gemini ${s['total']:.3f} in {s['records']} records" if a.short else json.dumps(s, indent=1))
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    """The smoke's mean $ per episode x n x 1.5 + the spend so far must fit the cap (registration (g))."""
    sm = run_spend(Path(a.smoke))
    tot = total_spend(Path(a.runs))
    per = sm["total"] / sm["records"] if sm["records"] else float("inf")
    proj = per * a.n * 1.5 + tot["total"]
    res = {"smoke": str(a.smoke), "smoke_records": sm["records"], "usd_per_episode": per, "n": a.n,
           "projection_usd": proj, "spent_usd": tot["total"], "cap_usd": a.cap, "go": proj <= a.cap, "at": _now()}
    print(json.dumps(res))
    return 0 if res["go"] else 1


def _kill(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        time.sleep(10)


def cmd_watch(a: argparse.Namespace) -> int:
    """Spend cap (-> STOP file) and, unless --no-pause, a pause on a new episode that errored with 503 / overload
    (-> <run>/PAUSED_overload.json); the runner's process group is stopped either way; records are kept."""
    runs = Path(a.runs)
    seen = {name: {(arm, k) for arm, d in latest(runs / name).items() for k, t in d.items() if t.get("error")}
            for name in LIVE_RUNS if (runs / name).is_dir()}
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{_now()} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(runs)
        if s["total"] >= a.cap:
            Path(a.stop).write_text(json.dumps({"at": _now(), "spend": s, "cap": a.cap}, indent=1))
            _kill(a.pid)
            print(f"CAP REACHED: ${s['total']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        if not a.no_pause:
            for name in LIVE_RUNS:
                if not (runs / name).is_dir():
                    continue
                new = [(arm, k, t) for arm, d in latest(runs / name).items() for k, t in d.items()
                       if t.get("error") and (arm, k) not in seen.get(name, set()) and overload_error(t)]
                if new:
                    (runs / name / "PAUSED_overload.json").write_text(json.dumps(
                        {"at": _now(), "episodes": [{"arm": arm, "task": k, "error": str(t.get("error"))[:300]}
                                                    for arm, k, t in new]}, indent=1))
                    _kill(a.pid)
                    print(f"{_now()} PAUSED: {len(new)} overload error(s) in {name}; runner stopped", flush=True)
                    return 2
        print(f"{_now()} spend ${s['total']:.3f} ({s['records']} records)", flush=True)
        time.sleep(a.every)


# ================================================================================================ analysis (local)
BOX = ROOT / "runs/e14_small/box"                 # s3://.../e14_small/box synced here (untracked)
RUNS = ROOT / "runs"
S23 = ROOT / "results/stage2/s23"
CAL_SRC = ROOT / "runs/e2a_jevgym_shadow_cal_rl"
CAL_WORK = ROOT / "runs/e14_small_cal"
CAL_OUT = OUT / "cal"
ARMS = {"A": "e14a", "G": "e14g", "AG": "e14ag"}
ARM_LABEL = {"e14a": "Qwen3.5-0.8B + Clef head, agent states (A)", "e14g": "Qwen3.5-0.8B + Clef head, generic mix (G)",
             "e14ag": "Qwen3.5-0.8B + Clef head, both (AG)"}
PROFILE = ROOT / "configs/harness/explore_e14_final.yaml"
FINAL_PROFILE = ROOT / "configs/harness/jevagent_final.yaml"
DPS = ("tool_select", "tool_params")
N_BOOT = 2000
KEEP_LO = -0.010
GUIDE = {"typed_decisions": (0.36, 0.73), "banking77": (0.07, 0.74), "clinc150": (0.19, 0.76), "mean": (0.207, 0.743)}
REPRO_MIN = 0.693                                   # registration (d): 74.3 % minus 5 points
TARIFF_DED = ROOT / "results/harness_final/tariff_dedicated.json"
E9_USD_H = 2.24208
OWN_HOST_USD_H = 1.212                              # g5.2xlarge on-demand, us-east-1 (AWS price list): the E14 host
LOGGED_B = ROOT / "runs/harness_final_dedicated"   # the logged dedicated final-harness rows (seeds 300500-300699)


def _load_answers(p: Path) -> dict[str, dict[str, Any]]:
    return {r["qhash"]: r for r in _jl(p)} if p.exists() else {}


def _fits(key: str, d: Path | None = None) -> dict[str, Any]:
    d = d or CAL_OUT
    out = {}
    for dp in DPS:
        fp = d / f"tauhat_e14cal_{key}_{dp}.json"
        if fp.exists():
            f = json.loads(fp.read_text())
            t = f.get("tau_hat")
            out[dp] = {"tau_hat": t, "n_paired": f.get("n_paired"), "tau_rule": (f.get("tau_rule") or {}).get("rule"),
                       "accepts": t is not None and isinstance(t, (int, float)) and math.isfinite(t) and t <= 1.0}
    return out


def cmd_fit(a: argparse.Namespace) -> int:
    """Registration (f) 3 = E5's entry (g): the calibration-state answers written into a copy of the 4B's calibration
    shadow arm, then the registered threshold command per decision point."""
    import e5_cal_replay as CR

    key = ARMS[a.arm.upper()]
    ans = Path(a.box) / "cal" / f"answers_{key}.jsonl"
    if not ans.exists():
        raise SystemExit(f"no {ans}")
    d = CR.derive(CAL_SRC, "rl_shadow_cal", ans, CAL_WORK, f"e14cal_{key}", model=f"e14-{a.arm.lower()}")
    f = CR.fit(CAL_WORK, f"e14cal_{key}", CAL_OUT)
    print(json.dumps({"derive": d, "fit": f}, indent=1, default=str))
    return 0


def cal_coverage(key: str, taus: Mapping[str, float] | None = None, path: Path | None = None) -> dict[str, Any]:
    """Share of the calibration states' decisions the cascade takes at tau (confidence >= tau and not OTHER), System-1's
    and the LLM's accuracy on them, and the cascade's accuracy over all paired decisions (as E5's table)."""
    from jevharness.bench.threshold_theory import decision_pairs, load_trajectories, paired

    path = path or (CAL_WORK / f"e14cal_{key}.jsonl")
    if not path.exists():
        return {}
    trajs = list(load_trajectories(path))
    fits = _fits(key)
    r: dict[str, Any] = {}
    for dp in DPS:
        if taus is not None:
            tau = taus[dp]
        elif dp in fits:
            t = fits[dp]["tau_hat"]
            tau = float(t) if fits[dp]["accepts"] else float("inf")
        else:
            continue
        ps = paired(decision_pairs(trajs, dp))
        if not ps:
            continue
        acc = [x for x in ps if x["c"] is not None and x["c"] >= tau and x["acceptable"]]
        n = len(ps)
        casc = (sum(float(x["s1_ok"]) for x in acc) + sum(float(x["s2_ok"]) for x in ps if x not in acc)) / n
        r[dp] = {"tau": tau, "n_paired": n, "accepted": len(acc), "coverage": len(acc) / n,
                 "s1_acc_accepted": (sum(float(x["s1_ok"]) for x in acc) / len(acc)) if acc else None,
                 "s2_acc_accepted": (sum(float(x["s2_ok"]) for x in acc) / len(acc)) if acc else None,
                 "s1_acc_all": sum(float(x["s1_ok"]) for x in ps) / n, "s2_acc_all": sum(float(x["s2_ok"]) for x in ps) / n,
                 "cascade_acc": casc, "conf_quantiles": _quantiles([x["c"] for x in ps if x["c"] is not None])}
    return r


def _quantiles(xs: Sequence[float]) -> dict[str, float] | None:
    import numpy as np

    if not xs:
        return None
    return {f"q{int(q * 100)}": float(np.quantile(xs, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}


def cmd_prepare_live(a: argparse.Namespace) -> int:
    """Arm A's offline-fitted thresholds into backends.e14a of a copy of the final-harness profile (registration
    (g) (b)); a fit that accepts no level -> always defer at that point; neither point -> (b) is not run."""
    import hashlib

    import yaml

    from fill_backend_tau import fill
    from jevharness.harness import profile as HP

    key = ARMS[a.arm.upper()]
    fits, parents = {}, {}
    for dp in DPS:
        fp = CAL_OUT / f"tauhat_e14cal_{key}_{dp}.json"
        if not fp.exists():
            raise SystemExit(f"missing fit {fp} (run fit first)")
        fits[dp] = json.loads(fp.read_text())
        parents[dp] = {"path": str(fp.relative_to(ROOT)), "sha256": hashlib.sha256(fp.read_bytes()).hexdigest()}
    prof = HP.load_profile(FINAL_PROFILE)
    prof["name"] = "explore_e14_final"
    prof["description"] = ("Exploratory row E14's live profile (PREREG_A7_STAGE2 Deviations 2026-10-07 'E14' (g) (b)): "
                           "a copy of configs/harness/jevagent_final.yaml (tool + argument cascade, reflection off, "
                           "batch_s1; the frozen JevAgentRL-4B block unchanged) with backends.e14a, arm A's offline-"
                           "fitted thresholds.")
    sec = prof["system2"]["gemini-3.5-flash"]
    sec["backends"]["e14a"] = {"tool_select": {"calibration": "pending"}, "tool_params": {"calibration": "pending"}}
    out = fill(prof, "e14a", fits)
    blk = out["system2"]["gemini-3.5-flash"]["backends"]["e14a"]
    notes = blk.pop("notes")
    blk["fitted_from"] = {**parents, "fit_arm": f"e14cal_{key}", "model": "e14-a",
                          "rule": "PREREG_A7_STAGE2 Deviations 2026-10-07 'E14' (f) 3: offline fit on rl_shadow_cal's "
                                  "states (E5's procedure, entry (g))"}
    for k in [k for k in out if k.startswith("_")]:
        out.pop(k)
    head = ["# Exploratory row E14's live profile (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 'E14: a sub-1B decision",
            "# model with a scoring head' (g) (b)): configs/harness/jevagent_final.yaml copied unchanged (tool + argument",
            "# cascade, reflection off, batch_s1; the frozen JevAgentRL-4B block unchanged) plus backends.e14a, filled from",
            "# arm A's offline fit (scripts/explore_e14_small.py fit -> threshold_theory --calibration-arm -> fill_backend_tau).",
            f"# backends.e14a filled by scripts/explore_e14_small.py prepare-live: {'; '.join(notes)}"]
    PROFILE.write_text("\n".join(head) + "\n" + yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    HP.load_profile(PROFILE)
    print(json.dumps({"taus": {dp: blk.get(dp) for dp in DPS}, "skip": blk.get("skip"), "notes": notes}, indent=1,
                     default=str))
    return 0


# ------------------------------------------------------------------ summarize
def _boot(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    import ablation_score as AS

    r = AS.paired_boot(a, b, N_BOOT, 0)
    keeps, loses = r["lo"] >= KEEP_LO, r["hi"] < 0
    r["reading"] = ("keeps the reference's accuracy within 1 point (small loss: interval below 0)" if keeps and loses
                    else "keeps the reference's accuracy" if keeps else "loses accuracy" if loses else "inconclusive")
    return r


def training(box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, key in ARMS.items():
        p = box / f"train_record_{arm}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        out[key] = {k: r.get(k) for k in ("build", "train_records", "dev", "base_dev", "selected_epoch", "second_epoch",
                                          "calibration_dev", "temperature", "head_temperature", "head", "head_params_m",
                                          "minutes_total", "peak_reserved_gb", "gpu", "oom", "recipe")}
        out[key]["epochs"] = [{k: e.get(k) for k in ("tag", "minutes", "s_per_step", "steps", "loss_first_last", "dev",
                                                     "batch", "grad_accum", "peak_alloc_gb")} for e in r.get("epochs", [])]
    return out


def unsloth_sets(box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("base", *ARMS):
        p = box / "eval" / f"evalsets_{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            out[name if name == "base" else ARMS[name]] = {
                "sets": {k: {kk: v.get(kk) for kk in ("accuracy", "record_accuracy", "ece", "decisions")}
                         for k, v in d["sets"].items()}, "mean_accuracy": d.get("mean_accuracy")}
    g = out.get("e14g")
    out["reproduction"] = {"guide_0.8b": GUIDE, "min_mean": REPRO_MIN,
                           "g_mean": g["mean_accuracy"] if g else None,
                           "reproduced": (g["mean_accuracy"] >= REPRO_MIN) if g else None}
    return out


def v1(box: Path) -> dict[str, Any]:
    import ablation_score as AS

    out = {}
    for key in ARMS.values():
        p = box / "eval" / f"{key}_v1.json"
        if p.exists():
            out[key] = AS.v1_metrics(p)
    e5 = json.loads((ROOT / "results/explore/e5_small/summary.json").read_text())
    out["refs"] = {k: {kk: v.get(kk) for kk in ("v1_acc", "v1_tool_select", "v1_tool_params", "v1_stop", "v1_recovery",
                                                "v1_knockout", "v1_ece", "label")}
                   for k, v in {**e5.get("references", {}), **{f"e5_{m}": e5["models"][m] for m in e5["models"]}}.items()}
    return out


def corpus(box: Path) -> dict[str, Any]:
    import ablation_score as AS
    import e5_analysis as E5A

    occ, eligible = AS.load_corpus(S23)
    refs = {"rl": S23 / "answers_rl.jsonl", "sft_v1": S23 / "answers_sft_v1.jsonl",
            "sl0.8b": S23 / "answers_sl0.8b.jsonl",
            "e5_q06": ROOT / "results/explore/e5_small/records/s23/answers_e5_e5_q06.jsonl.gz",
            "e5_mc1": ROOT / "results/explore/e5_small/records/s23/answers_e5_e5_mc1.jsonl.gz"}
    rows: dict[str, Any] = {}
    per: dict[str, Any] = {}
    for name, p in [*((k, box / "s23" / f"answers_{k}.jsonl") for k in ARMS.values()), *refs.items()]:
        ans = _load_answers(p)
        if not ans:
            continue
        m, pt = E5A.c_rows(occ, eligible, ans)
        rows[name] = m
        per[name] = pt
    con: dict[str, Any] = {}
    for k in ARMS.values():
        if k not in per:
            continue
        for r in refs:
            if r in per:
                con[f"{k}_vs_{r}"] = {st: _boot(per[k][st], per[r][st]) for st in ("all", "offplan", "ask")}
    if "e14a" in per and "e14g" in per:
        con["e14a_vs_e14g"] = {st: _boot(per["e14a"][st], per["e14g"][st]) for st in ("all", "offplan", "ask")}
    return {"rows": rows, "contrasts": con}


def static(runs: Path = RUNS) -> dict[str, Any]:
    import explore_e9_frontier as E9
    import explore_startlux_live as SL

    data = E9.load_e1()
    e1 = E9.analyse_e1(data, E9.token_rates(), n_boot=N_BOOT)
    ids = sorted(next(iter(data.values())))
    pts: dict[str, Any] = {}
    have = {}
    for key in ARMS.values():
        rel = f"e1_bfcl_when2call_e14/s1_{key}.jsonl"
        E9.EXTRA[key] = (ARM_LABEL[key], rel, None)
        x = E9._extra_point(key, e1, data, runs)
        if x.get("available"):
            recs = {r["id"]: r for r in E9.read_jsonl(runs / rel)}
            x["gap_to_sl08_pt"] = E9.paired_gap([E9.is_correct(recs[i]) for i in ids],
                                                [E9.is_correct(data["sl0.8b"][i]) for i in ids])
            items = SL.e1_items(runs / rel)
            x["bfcl_tool_choice_acc"] = float(sum(items.values()) / len(items)) if items else None
            x["bfcl_tool_choice_n"] = len(items or {})
            have[key] = (x["stats"]["usd"], x["stats"]["acc"])
        pts[key] = x
    drawn = {m: (v["usd"], v["acc"]) for m, v in e1["models"].items() if v["drawn"]}
    out = {"points": pts, "frontier_e9": list(e1["frontier"]),
           "refs": {m: {"acc": e1["models"][m]["acc"], "usd": e1["models"][m]["usd"],
                        "families": e1["models"][m]["families"], "lat_ms_median": e1["models"][m]["lat_ms_median"]}
                    for m in ("rl", "sl0.8b", "sl2b", "sl4b", "jev", "flash", "lite", "intern")}}
    if have:
        fw = E9.frontier({**drawn, **have})
        out["frontier_with_e14"] = fw
        for k in have:
            pts[k]["on_frontier_all"] = k in fw
    if "e14a" in have and "e14g" in have:
        ra = {r["id"]: r for r in E9.read_jsonl(runs / "e1_bfcl_when2call_e14/s1_e14a.jsonl")}
        rg = {r["id"]: r for r in E9.read_jsonl(runs / "e1_bfcl_when2call_e14/s1_e14g.jsonl")}
        out["a_minus_g_pt"] = E9.paired_gap([E9.is_correct(ra[i]) for i in ids], [E9.is_correct(rg[i]) for i in ids])
        out["a_minus_g_family_pt"] = {}
        for fam in E9.FAMILIES:   # descriptive: per family (Figure 4b's static column is BFCL tool choice)
            fi = [i for i in ids if E9.family_of(ra[i]) == fam]
            out["a_minus_g_family_pt"][fam] = E9.paired_gap([E9.is_correct(ra[i]) for i in fi],
                                                            [E9.is_correct(rg[i]) for i in fi])
    return out


def _e1_elapsed_s(box: Path, key: str) -> float | None:
    """The E1 run's wall-clock seconds: the decision_eval progress bar's final elapsed time (engine v1 runs), or the
    window the driver recorded around the v2 re-timing."""
    import re

    if key.endswith("_v2"):
        p = box / "e1_v2_window.json"
        if p.exists():
            w = json.loads(p.read_text())
            return float(w["end_epoch"]) - float(w["start_epoch"])
        return None
    p = box / "logs" / f"e1_{key}_1.log"
    if not p.exists():
        return None
    last = None
    for m in re.finditer(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<", p.read_text(errors="replace").replace("\r", "\n")):
        if m.group(1) == m.group(2):
            h, mi, se = int(m.group(3) or 0), int(m.group(4)), int(m.group(5))
            last = 3600 * h + 60 * mi + se
    return float(last) if last is not None else None


def e1_engines(box: Path, runs: Path = RUNS) -> dict[str, Any]:
    """L1 cost of arm A under each engine (registration "serving-engine fix" (d)): E9's rule (request latency / questions,
    $2.24208 per GPU-hour), the wall-clock rule (E1's wall time / 3,991 at the same tariff, as E12c (h)) and, for v2,
    the agreement of its answers with v1's (v1's answers stay the L1 accuracy of record)."""
    import explore_e9_frontier as E9

    out: dict[str, Any] = {}
    base = runs / "e1_bfcl_when2call_e14"
    v1 = {r["id"]: r for r in E9.read_jsonl(base / "s1_e14a.jsonl")} if (base / "s1_e14a.jsonl").exists() else {}
    for key in ("e14a", "e14a_v2", "e14g", "e14ag"):
        f = base / f"s1_{key}.jsonl"
        if not f.exists():
            continue
        recs = {r["id"]: r for r in E9.read_jsonl(f)}
        st = E9.model_stats(list(recs.values()), "gpu")
        el = _e1_elapsed_s(box, key)
        row = {"engine": "v2" if key.endswith("_v2") else "v1", "n": st["n"], "acc": st["acc"],
               "usd_per_1k_e9_rule": 1000 * st["usd"], "lat_ms_median": st["lat_ms_median"], "lat_ms_p90": st["lat_ms_p90"],
               "n_over_5s": sum(1 for r in recs.values() if float(r.get("latency_ms") or 0) > 5000),
               "wall_s": el, "usd_per_1k_wall": (1000 * E9_USD_H / 3600 * el / st["n"]) if el else None}
        if key == "e14a_v2" and v1:
            common = sorted(set(v1) & set(recs))
            row["pred_agree_with_v1"] = sum(v1[i].get("pred") == recs[i].get("pred") for i in common) / len(common)
            row["conf_max_abs_vs_v1"] = max(abs(float(v1[i].get("conf") or 0) - float(recs[i].get("conf") or 0))
                                            for i in common)
        out[key] = row
    chk = box / "engine_check.json"
    if chk.exists():
        out["engine_check"] = json.loads(chk.read_text())
        for eng in ("v1", "v2"):   # wall seconds for the 2,065 SFT v1 dev records at 4 concurrent requests
            lp = box / "logs" / f"devcheck_{eng}.log"
            if lp.exists():
                try:
                    out["engine_check"][f"devcheck_{eng}_s"] = json.loads(lp.read_text().strip().splitlines()[-1])["seconds"]
                except (ValueError, KeyError, IndexError):
                    pass
        kl = box / "logs" / "kernels.log"
        out["engine_check"]["causal_conv1d"] = ("installed" if kl.exists() and "causal_conv1d " in kl.read_text()
                                                else "not installed (no published wheel)")
    return out


def latency(box: Path) -> dict[str, Any]:
    import e5_analysis as E5A

    labels = ["4b"] + [k for k in (*ARMS.values(), "e14a_v2") if (box / "latency" / "bench" / f"bench_{k}_new_c1.json").exists()
                       or (box / "latency" / "bench_compact" / f"bench_{k}_new_c1.json").exists()]
    if not (box / "latency").exists():
        return {}
    return E5A.latency(box, labels)


def train_peaks(box: Path) -> dict[str, Any]:
    import e5_box as EB

    return {arm: EB.peak_mib(box / "logs" / f"train_smi_{arm}.csv") for arm in ARMS
            if (box / "logs" / f"train_smi_{arm}.csv").exists()}


def live_a(runs: Path = RUNS) -> dict[str, Any]:
    import explore_common as C
    import explore_startlux_live as SL

    tasks = set(json.loads((ROOT / "configs/eval/jevgym_stage2_tasks.json").read_text())["task_ids"][:SL.LIVE_N])
    out: dict[str, Any] = {}
    ys: dict[str, dict[str, float]] = {}
    rows = {"e14a": runs / "e14_live_a" / "e14a_ts_replace_f.jsonl", "e14g": runs / "e14_live_a_g" / "e14g_ts_replace_f.jsonl",
            "frozen": runs / "stage2_rl_states" / "rl_ts_replace_f.jsonl", "jev": runs / "stage2_rl_states" / "jev_ts_replace_f.jsonl",
            "sl0.8b": runs / "explore_startlux_live" / "sl0.8b_ts_replace_f.jsonl"}
    for k, p in rows.items():
        r, y = SL.live_row(p, tasks)
        if r:
            out[k] = r
            ys[k] = y
    for k in ("e14a", "e14g"):
        if k not in ys:
            continue
        for ref in ("frozen", "jev", "sl0.8b", "e14a"):
            if ref == k or ref not in ys:
                continue
            common = sorted(set(ys[k]) & set(ys[ref]))
            if common:
                out[k][f"vs_{ref}"] = C.newcombe([ys[k][t] for t in common], [ys[ref][t] for t in common])
    return out


def live_b(runs: Path = RUNS) -> dict[str, Any]:
    """The final harness with arm A against the logged dedicated rows on the same 200 tasks (registration (g) (b))."""
    import numpy as np

    import explore_common as C

    p = runs / "e14_live_b" / "e14_final.jsonl"
    if not p.exists():
        return {}
    tar = json.loads(TARIFF_DED.read_text())
    host_h = float(tar["host_usd_per_hour"])
    arms = {"e14_final": p, "llm_only": LOGGED_B / "llm_only.jsonl", "harness_final": LOGGED_B / "harness_final.jsonl"}
    lat: dict[str, dict[str, dict[str, Any]]] = {}
    for k, f in arms.items():
        latest_, _ = C.load_latest([f])
        lat[k] = {t: r for t, r in latest_.items() if not r.get("error")}
    common = sorted(set.intersection(*(set(v) for v in lat.values())))
    res: dict[str, Any] = {"n_common": len(common), "host_usd_per_hour": host_h, "e9_usd_per_hour": E9_USD_H,
                           "logged_tariff_q_4b": tar["q_usd_per_episode"], "errored": {}, "arms": {}}
    for k in arms:
        res["errored"][k] = sum(1 for t, r in C.load_latest([arms[k]])[0].items() if r.get("error"))
    import harness_final as HF

    per: dict[str, dict[str, Any]] = {}
    for k, d in lat.items():
        eps = [HF._episode_live(d[t]) for t in common]
        wall = np.array([e["wall_s"] for e in eps])
        api = np.array([e["cost"] for e in eps])
        succ = np.array([e["success"] for e in eps])
        gpu = k != "llm_only"
        q = host_h / 3600 * wall if gpu else np.zeros_like(wall)
        q9 = E9_USD_H / 3600 * wall if gpu else np.zeros_like(wall)
        qown = OWN_HOST_USD_H / 3600 * wall if gpu else np.zeros_like(wall)
        per[k] = {"succ": succ, "api": api, "tot": api + q, "tot9": api + q9, "wall": wall}
        if k == "harness_final":   # the logged row's own tariff file (a separate dedicated-GPU window, same host type)
            per[k]["tot_tariff"] = api + float(tar["q_usd_per_episode"])
        tot = {x: sum(e[x] for e in eps) for x in ("tool_decisions", "tool_by_s1", "arg_decisions", "arg_by_s1")}
        res["arms"][k] = {"success": float(succ.mean()), "api_usd": float(api.mean()), "q_usd": float(q.mean()),
                          "q_usd_e9": float(q9.mean()), "usd_incl_gpu": float((api + q).mean()),
                          "usd_incl_gpu_e9": float((api + q9).mean()),
                          "usd_incl_gpu_ownhost": float((api + qown).mean()),
                          "usd_incl_gpu_logged_tariff": (float(per[k]["tot_tariff"].mean()) if "tot_tariff" in per[k]
                                                         else None), "wall_s_mean": float(wall.mean()),
                          "wall_s_median": float(np.median(wall)),
                          "s1_s": float(np.mean([e["s1_s"] for e in eps])), "s2_s": float(np.mean([e["s2_s"] for e in eps])),
                          "s2_calls": float(np.mean([e["s2_calls"] for e in eps])),
                          "s1_requests": float(np.mean([e["s1_requests"] for e in eps])),
                          "steps": float(np.mean([e["steps"] for e in eps])),
                          "tool_coverage": tot["tool_by_s1"] / tot["tool_decisions"] if tot["tool_decisions"] else None,
                          "arg_coverage": tot["arg_by_s1"] / tot["arg_decisions"] if tot["arg_decisions"] else None}
    con: dict[str, Any] = {}
    for ref in ("llm_only", "harness_final"):
        a, b = per["e14_final"], per[ref]
        con[ref] = {"success": C.newcombe(a["succ"], b["succ"]),
                    "success_boot": C.paired_boot(a["succ"], b["succ"]),
                    "saving_incl_gpu": C.paired_boot(a["tot"], b["tot"], ratio=True),
                    "saving_incl_gpu_e9": C.paired_boot(a["tot9"], b["tot9"], ratio=True),
                    "saving_api": C.paired_boot(a["api"], b["api"], ratio=True),
                    **({"saving_vs_logged_tariff": C.paired_boot(a["tot"], b["tot_tariff"], ratio=True)}
                       if "tot_tariff" in b else {}),
                    "wall_s": C.paired_boot(a["wall"], b["wall"])}
    res["contrasts"] = con
    return res


def summarize(box: Path = BOX) -> dict[str, Any]:
    res: dict[str, Any] = {"label": LABEL, "generated": _now(), "box": str(box.relative_to(ROOT)) if box.is_relative_to(ROOT)
                           else str(box)}
    for name, fn in (("training", lambda: training(box)), ("v1", lambda: v1(box)),
                     ("unsloth_sets", lambda: unsloth_sets(box)), ("corpus", lambda: corpus(box)),
                     ("static", static), ("e1_engines", lambda: e1_engines(box)), ("latency", lambda: latency(box)), ("train_peak_mib", lambda: train_peaks(box)),
                     ("live_a", live_a), ("live_b", live_b)):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001 -- a missing block is reported, the rest still summarized
            res[name] = {"error": f"{type(e).__name__}: {e}"}
    res["calibration"] = {k: {"fits": _fits(k), "coverage": cal_coverage(k)} for k in ARMS.values() if _fits(k)}
    res["calibration"]["4b_logged"] = {"coverage": cal_coverage("4b", taus={"tool_select": 0.4999999999978404,
                                                                             "tool_params": 0.9999998807907104},
                                                                path=CAL_SRC / "rl_shadow_cal.jsonl")}
    for k in ("host.json", "host_final.json", "stage_times.jsonl", "build_a.json", "build_g.json", "env_e14venv.txt"):
        p = box / k
        if p.exists():
            res.setdefault("host", {})[k] = (p.read_text() if k.endswith((".txt", ".jsonl")) else json.loads(p.read_text()))
    res["readings"] = readings(res)
    res["gpu_hours"] = _gpu_hours(box)
    res["spend"] = total_spend(RUNS)
    return res


def readings(r: Mapping[str, Any]) -> dict[str, Any]:
    """Registration (j), applied to the records (descriptive)."""
    out: dict[str, Any] = {}
    cal = r.get("calibration") or {}
    for key in ARMS.values():
        cov = (cal.get(key) or {}).get("coverage") or {}
        if len(cov) == 2:
            taus = {dp: c.get("tau") for dp, c in cov.items()}
            ok = all(t is not None and t <= 0.99 for t in taus.values()) and all(c["coverage"] >= 0.5 for c in cov.values())
            out[f"i_usable_confidences/{key}"] = {"holds": ok, "tau": taus,
                                                  "coverage": {dp: c["coverage"] for dp, c in cov.items()}}
    co = (r.get("corpus") or {}).get("contrasts") or {}
    lat = r.get("latency") or {}
    ee = r.get("e1_engines") or {}
    st = r.get("static") or {}
    if "e14a_vs_rl" in co and "e14a" in lat and "4b" in lat:
        c_ok = co["e14a_vs_rl"]["all"]["lo"] >= KEEP_LO
        p_a, p_4 = _g(lat, "e14a", "bench", "c1", "p50_ms"), _g(lat, "4b", "bench", "c1", "p50_ms")
        cost_a = _g(ee, "e14a", "usd_per_1k_e9_rule")
        cost_4 = 1000 * _g(st, "refs", "rl", "usd") if _g(st, "refs", "rl", "usd") is not None else None
        out["ii_near_4b_at_fraction_of_cost/e14a"] = {
            "c_within_1pt": c_ok, "c_lo": co["e14a_vs_rl"]["all"]["lo"], "c1_p50_below_4b": p_a < p_4,
            "c1_p50": [p_a, p_4], "e9_cost_below_4b": (cost_a < cost_4) if cost_a and cost_4 else None,
            "e9_cost_per_1k": [cost_a, cost_4],
            "holds": bool(c_ok and p_a < p_4 and cost_a is not None and cost_4 is not None and cost_a < cost_4)}
    amg = st.get("a_minus_g_pt")
    cg = co.get("e14a_vs_e14g")
    if amg and cg:
        g_ahead_l1 = amg["ci"][1] < 0
        a_ahead_l2 = cg["all"]["lo"] > 0
        out["iii_fixed_questions_misrank/A_vs_G"] = {"g_ahead_on_l1": g_ahead_l1, "a_ahead_on_l2": a_ahead_l2,
                                                     "l1_a_minus_g": amg, "l2_a_minus_g": cg["all"],
                                                     "holds": bool(g_ahead_l1 and a_ahead_l2)}
    return out


def _gpu_hours(box: Path) -> float | None:
    """Host hours from launch to stop: the driver's count at the final stage plus the power-off delay it announced
    (the host stops itself that many minutes later)."""
    p = box / "host_final.json"
    if not p.exists():
        return None
    h = float(json.loads(p.read_text()).get("hours_used") or 0)
    st = box / "status.txt"
    if h and st.exists():
        m = re.findall(r"power-off in (\d+) min", st.read_text())
        if m:
            h += int(m[-1]) / 60
    return h or None


def _jsonable(o: Any) -> Any:
    import numpy as np

    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def cmd_summarize(a: argparse.Namespace) -> int:
    res = _jsonable(summarize(Path(a.box)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "summary.md").write_text(markdown(res) + "\n")
    keys = tex_keys(res)
    write_tex(keys)
    print(f"{len(keys)} keys -> {TEX.relative_to(ROOT)}; summary -> {OUT.relative_to(ROOT)}")
    return 0


# ------------------------------------------------------------------ outputs
def _pc(x: Any, nd: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{nd}f}"


def _nm(x: Any, nd: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _g(d: Any, *ks: Any) -> Any:
    for k in ks:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def markdown(r: Mapping[str, Any]) -> str:
    L = ["# E14: a sub-1B decision model with a scoring head (exploratory; descriptive)", "", r["label"] + ".", "",
         f"Generated {r['generated']} from `{r['box']}`; GPU host hours used {r.get('gpu_hours')}; Gemini spend logged "
         f"${_nm(_g(r, 'spend', 'total'), 2)}. Qwen/Qwen3.5-0.8B (revision 2fc06364) + a fresh Clef joint schema head "
         "(Unsloth 2026.10.2, its from-LM recipe and defaults); one seed per arm.", ""]
    tr = r.get("training") or {}
    if tr and "error" not in tr:
        L += ["## Training (development data only)", "",
              "| arm | records | epoch | minutes | s/step | loss first / last | dev acc | selected | dev acc calibrated | dev ECE calibrated | head T |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, t in tr.items():
            for e in t.get("epochs") or []:
                L.append(f"| {k} | {t.get('train_records')} | {e['tag']} | {e.get('minutes')} | {e.get('s_per_step')} | "
                         f"{e.get('loss_first_last')} | {_pc(_g(e, 'dev', 'accuracy'), 2)} | {t.get('selected_epoch')} | "
                         f"{_pc(_g(t, 'calibration_dev', 'accuracy'), 2)} | {_nm(_g(t, 'calibration_dev', 'ece'), 3)} | "
                         f"{_nm(t.get('head_temperature'), 3)} |")
            L.append(f"| {k} (fresh head, before training) | | | | | | {_pc(_g(t, 'base_dev', 'accuracy'), 2)} | | | | |")
        L += ["", "Second epoch rule (registration (d)): " + "; ".join(
            f"{k}: {json.dumps(t.get('second_epoch'))}" for k, t in tr.items()), ""]
    v = r.get("v1") or {}
    if v and "error" not in v:
        L += ["## Development read-outs (SFT v1 dev files; the served temperatures)", "",
              "| model | v1 acc | tool choice | arguments | stop | after off-plan | knockout | ECE |", "|---|---|---|---|---|---|---|---|"]
        for k, m in [*((k, v[k]) for k in ARMS.values() if k in v), *(v.get("refs") or {}).items()]:
            L.append(f"| {m.get('label') or k} | {_pc(m.get('v1_acc'), 2)} | {_pc(m.get('v1_tool_select'), 2)} | "
                     f"{_pc(m.get('v1_tool_params'), 2)} | {_pc(m.get('v1_stop'), 2)} | {_pc(m.get('v1_recovery'), 2)} | "
                     f"{_pc(m.get('v1_knockout'), 2)} | {_nm(m.get('v1_ece'), 3)} |")
        L.append("")
    us = r.get("unsloth_sets") or {}
    if us and "error" not in us:
        L += ["## Unsloth's evaluation sets (G's reproduction check)", "",
              "| model | typed-decisions | BANKING77 | CLINC150 | mean |", "|---|---|---|---|---|"]
        for k in ("base", *ARMS.values()):
            if k in us:
                ss = us[k]["sets"]
                L.append(f"| {k} | {_pc(_g(ss, 'typed_decisions', 'accuracy'))} | {_pc(_g(ss, 'banking77', 'accuracy'))} | "
                         f"{_pc(_g(ss, 'clinc150', 'accuracy'))} | {_pc(us[k]['mean_accuracy'])} |")
        L.append("| Unsloth's guide, Qwen3.5-0.8B (before -> after) | 36 -> 73 | 7 -> 74 | 19 -> 76 | 20.7 -> 74.3 |")
        L += ["", f"G reproduced (mean >= {100 * REPRO_MIN:.1f} %): {_g(us, 'reproduction', 'reproduced')}.", ""]
    st = r.get("static") or {}
    if st and "error" not in st:
        L += ["## L1: E1's 3,991 fixed decisions (E9's cost rule)", "",
              "| model | acc | BFCL tool choice | BFCL args | When2Call | $ / 1k decisions | median latency ms | vs AgentDecision-4B (pt) | vs StartLux-0.8B (pt) | on E9's frontier |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, x in (st.get("points") or {}).items():
            if not x.get("available"):
                L.append(f"| {k} | not available: {x.get('reason')} |||||||||")
                continue
            s_ = x["stats"]
            fam = s_["families"]
            L.append(f"| {ARM_LABEL[k]} | {_pc(s_['acc'])} | {_pc(_g(fam, 'toolchoice', 'acc'))} | {_pc(_g(fam, 'args', 'acc'))} | "
                     f"{_pc(_g(fam, 'w2c', 'acc'))} | {_nm(1000 * s_['usd'], 4)} | {_nm(s_['lat_ms_median'], 0)} | "
                     f"{_nm(_g(x, 'gap_to_frozen_pt', 'est'), 1)} {_g(x, 'gap_to_frozen_pt', 'ci')} | "
                     f"{_nm(_g(x, 'gap_to_sl08_pt', 'est'), 1)} {_g(x, 'gap_to_sl08_pt', 'ci')} | {x.get('on_frontier_all')} |")
        for m, x in (st.get("refs") or {}).items():
            fam = x["families"]
            L.append(f"| {m} (E9) | {_pc(x['acc'])} | {_pc(_g(fam, 'toolchoice', 'acc'))} | {_pc(_g(fam, 'args', 'acc'))} | "
                     f"{_pc(_g(fam, 'w2c', 'acc'))} | {_nm(1000 * x['usd'], 4)} | {_nm(x['lat_ms_median'], 0)} | | | |")
        L += ["", f"E9's frontier: {st.get('frontier_e9')}; with the E14 points: {st.get('frontier_with_e14')}. "
              f"A minus G on E1 (pt): {json.dumps(st.get('a_minus_g_pt'))}; per family: "
              f"{json.dumps({f: (round(g['est'], 1), [round(x, 1) for x in g['ci']]) for f, g in (st.get('a_minus_g_family_pt') or {}).items()})}.", ""]
    ee = r.get("e1_engines") or {}
    if ee and "error" not in ee:
        L += ["### L1 cost by serving engine (arm A; v1 = one request per pass, the original engine; v2 = queued requests "
              "share a pass, warm-up at load, the causal_conv1d kernel if installed)", "",
              "| run | engine | acc | $ / 1k (E9 rule) | $ / 1k (wall clock) | wall s | median ms | p90 ms | requests > 5 s | agrees with v1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, x in ee.items():
            if k == "engine_check":
                continue
            L.append(f"| {k} | {x['engine']} | {_pc(x['acc'])} | {_nm(x['usd_per_1k_e9_rule'], 4)} | {_nm(x.get('usd_per_1k_wall'), 4)} | "
                     f"{_nm(x.get('wall_s'), 0)} | {_nm(x['lat_ms_median'], 0)} | {_nm(x['lat_ms_p90'], 0)} | {x['n_over_5s']} | "
                     f"{_pc(x.get('pred_agree_with_v1'), 2)} |")
        if ee.get("engine_check"):
            c = ee["engine_check"]
            L += ["", f"Engine check on development items (registration 'serving-engine fix' (c)): pass = {c.get('pass')}; "
                  f"SFT v1 dev: {json.dumps({k: v for k, v in (c.get('sft_v1_dev') or {}).items() if k != 'examples'})}; "
                  f"E1 dev prefix: {json.dumps({k: v for k, v in (c.get('e1_dev_prefix') or {}).items() if k != 'examples'})}."]
        L.append("")
    co = r.get("corpus") or {}
    if co and "error" not in co:
        L += ["## L2: the common states C", "", "| model | C acc | after off-plan | ask-vs-act | ECE |", "|---|---|---|---|---|"]
        for k, m in (co.get("rows") or {}).items():
            L.append(f"| {k} | {_pc(m.get('c_acc'))} | {_pc(m.get('c_acc_offplan'))} | {_pc(m.get('c_acc_ask'))} | "
                     f"{_nm(m.get('c_ece'), 3)} |")
        L += ["", "Paired task bootstrap (B = 2,000, seed 0), model - reference, points; E5's reading rule.", "",
              "| contrast | all states | after off-plan | ask-vs-act |", "|---|---|---|---|"]
        for c, x in (co.get("contrasts") or {}).items():
            L.append("| " + c + " | " + " | ".join(
                f"{100 * x[stt]['delta']:+.1f} [{100 * x[stt]['lo']:+.1f}, {100 * x[stt]['hi']:+.1f}] ({x[stt]['reading']})"
                for stt in ("all", "offplan", "ask")) + " |")
        L.append("")
    ca = r.get("calibration") or {}
    if ca:
        L += ["## Calibration states (E5's offline procedure): tau-hat and coverage", "",
              "| System-1 | decision point | tau | paired | coverage | S1 acc (taken) | LLM acc (taken) | cascade acc | LLM acc | confidence quantiles |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, x in ca.items():
            for dp, c in (x.get("coverage") or {}).items():
                tau = "always defers" if c.get("tau") is None else f"{c['tau']:.12g}"
                L.append(f"| {k} | {dp} | {tau} | {c['n_paired']} | {_pc(c['coverage'])} | {_pc(c['s1_acc_accepted'], 2)} | "
                         f"{_pc(c['s2_acc_accepted'], 2)} | {_pc(c['cascade_acc'], 2)} | {_pc(c['s2_acc_all'], 2)} | "
                         f"{json.dumps({q: round(v, 4) for q, v in (c.get('conf_quantiles') or {}).items()})} |")
        L.append("")
    la = r.get("latency") or {}
    if la and "error" not in la:
        L += ["## Serving: the final harness's request stream, one session on the E14 host's A10G", "",
              "| served | c=1 p50 | c=1 p90 | c=4 p50 | c=4 p90 | S1/step c=1 p50 | serve peak MiB | speed-up vs 4B c=1 / c=4 (p50) |",
              "|---|---|---|---|---|---|---|---|"]
        for k, x in la.items():
            b1, b4 = _g(x, "bench", "c1") or {}, _g(x, "bench", "c4") or {}
            sp = x.get("speedup_vs_4b") or {}
            L.append(f"| {k} | {_nm(b1.get('p50_ms'), 0)} | {_nm(b1.get('p90_ms'), 0)} | {_nm(b4.get('p50_ms'), 0)} | "
                     f"{_nm(b4.get('p90_ms'), 0)} | {_nm(b1.get('s1_ms_per_step_p50'), 0)} | {_nm(x.get('serve_peak_mib'), 0)} | "
                     f"{_nm(sp.get('c1_p50'))} / {_nm(sp.get('c4_p50'))} |")
        L += ["", "Engines and settings: the E14 arms are served by `serve_jev --backend clef` engine v1 (Unsloth 2026.10.2 "
              "loader, the merged bf16 checkpoint, every question of a request in one forward pass, one request per pass "
              "under a lock, no prefix cache, flash-linear-attention 0.5.2, the causal_conv1d reference PyTorch path; the "
              "engine-v2 check failed, so no v2 timing is used); AgentDecision-4B by `serve_jev --fast` (its session "
              "prefix cache, fused multi-question passes, read-out gather); both alone on the same A10G (g5.2xlarge) in "
              "one session, `scripts/harness_final.py bench --path new` (System-2 time emulated from the logged workload). "
              "E1's timings (L1 above) are engine v1 under E1's 4 workers.",
              "", f"Training peak memory (nvidia-smi, MiB): {json.dumps(r.get('train_peak_mib'))}.", ""]
    lv = r.get("live_a") or {}
    if lv and "error" not in lv:
        L += ["## L3 (a): REPLACE at the tool choice, the first 200 fresh Stage-2 tasks", "",
              "| row | n | success | own-state acc | steps | API $ / ep | wall s / ep | vs frozen (pt) | vs Jev (pt) | vs StartLux-0.8B (pt) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, x in lv.items():
            def nc(ref: str) -> str:
                y = x.get(f"vs_{ref}")
                return f"{y['diff_pt']:+.1f} [{y['ci_pt'][0]:+.1f}, {y['ci_pt'][1]:+.1f}]" if y else ""
            L.append(f"| {k} | {x.get('live_n')} | {_pc(x.get('live_success'))} | {_pc(x.get('own_acc'))} | {_nm(x.get('steps'), 1)} | "
                     f"{_nm(x.get('api_cost'), 4)} | {_nm(x.get('wall_s'), 1)} | {nc('frozen')} | {nc('jev')} | {nc('sl0.8b')} |")
        L.append("")
    lb = r.get("live_b") or {}
    if lb and "error" not in lb:
        L += ["## L3 (b): the final harness with arm A against the logged dedicated rows (seeds 300500-300699)", "",
              f"{lb['n_common']} common tasks; errored latest records {lb['errored']}. GPU tariff q = ${lb['host_usd_per_hour']}/h "
              f"(the logged rows' tariff host, g5.4xlarge) x the episode's wall-clock for the arms with a System-1 model "
              f"(E9's ${E9_USD_H}/h as sensitivity); the logged 4B row's own tariff file gives q = ${lb['logged_tariff_q_4b']:.5f}.", "",
              "| arm | success | API $ / ep | q $ / ep | $ / ep incl. GPU | $ incl. GPU (E9 rate) | wall s / ep mean (median) | S1 s / ep | S2 s / ep | S2 calls | tool cov. | arg cov. |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, x in lb["arms"].items():
            L.append(f"| {k} | {_pc(x['success'])} | {_nm(x['api_usd'], 4)} | {_nm(x['q_usd'], 5)} | {_nm(x['usd_incl_gpu'], 4)} | "
                     f"{_nm(x['usd_incl_gpu_e9'], 4)} | {_nm(x['wall_s_mean'])} ({_nm(x['wall_s_median'])}) | {_nm(x['s1_s'])} | "
                     f"{_nm(x['s2_s'])} | {_nm(x['s2_calls'])} | {_pc(x.get('tool_coverage'))} | {_pc(x.get('arg_coverage'))} |")
        L += ["", "| contrast (e14_final minus) | success pt [Newcombe] | saving incl. GPU [boot] | saving incl. GPU, E9 rate | wall s [boot] |",
              "|---|---|---|---|---|"]
        for ref, c in lb["contrasts"].items():
            L.append(f"| {ref} | {c['success']['diff_pt']:+.1f} [{c['success']['ci_pt'][0]:+.1f}, {c['success']['ci_pt'][1]:+.1f}] | "
                     f"{_pc(c['saving_incl_gpu']['est'])} % [{_pc(c['saving_incl_gpu']['ci'][0])}, {_pc(c['saving_incl_gpu']['ci'][1])}] | "
                     f"{_pc(c['saving_incl_gpu_e9']['est'])} % | {c['wall_s']['est']:+.2f} [{c['wall_s']['ci'][0]:+.2f}, {c['wall_s']['ci'][1]:+.2f}] |")
        L.append("")
    rd = r.get("readings") or {}
    if rd:
        L += ["## Registered reading rules (registration (j); descriptive)", ""]
        for k, x in rd.items():
            L.append(f"- {k}: **{'holds' if x.get('holds') else 'does not hold'}** "
                     f"({json.dumps({kk: v for kk, v in x.items() if kk != 'holds'}, default=str)[:400]})")
        L.append("")
    for blk in ("training", "v1", "unsloth_sets", "corpus", "static", "latency", "live_a", "live_b"):
        if isinstance(r.get(blk), Mapping) and r[blk].get("error"):
            L.append(f"- {blk}: not available ({r[blk]['error'][:200]})")
    return "\n".join(L)


def _k(v: Any, nd: int = 1, pct: bool = True) -> str | None:
    import explore_common as C

    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return C.fmt(100 * v if pct else v, nd)


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    import explore_e9_frontier as E9

    k: dict[str, str | None] = {}
    short = {"e14a": "a", "e14g": "g", "e14ag": "ag"}
    for key, t in (r.get("training") or {}).items():
        if not isinstance(t, Mapping) or key not in short:
            continue
        a = short[key]
        k[f"e14/{a}/train/records"] = _k(t.get("train_records"), 0, False) if t.get("train_records") else None
        k[f"e14/{a}/train/minutes"] = _k(t.get("minutes_total"), 0, False)
        k[f"e14/{a}/train/epochs"] = str(len(t.get("epochs") or []))
        k[f"e14/{a}/train/selected"] = str(t.get("selected_epoch"))
        k[f"e14/{a}/train/devbase"] = _k(_g(t, "base_dev", "accuracy"))
        for e in t.get("epochs") or []:
            k[f"e14/{a}/train/{e['tag']}/dev"] = _k(_g(e, "dev", "accuracy"))
            k[f"e14/{a}/train/{e['tag']}/minutes"] = _k(e.get("minutes"), 0, False)
        k[f"e14/{a}/train/devcal"] = _k(_g(t, "calibration_dev", "accuracy"))
        k[f"e14/{a}/train/devece"] = _k(_g(t, "calibration_dev", "ece"), 3, False)
    for key, m in (r.get("v1") or {}).items():
        if key in short and isinstance(m, Mapping):
            a = short[key]
            for f, n in (("v1_acc", "acc"), ("v1_tool_select", "ts"), ("v1_tool_params", "tp"), ("v1_stop", "stop"),
                         ("v1_recovery", "rec"), ("v1_knockout", "ko")):
                k[f"e14/{a}/v1/{n}"] = _k(m.get(f))
            k[f"e14/{a}/v1/ece"] = _k(m.get("v1_ece"), 3, False)
    us = r.get("unsloth_sets") or {}
    for key, x in us.items():
        if key == "reproduction" or not isinstance(x, Mapping) or "sets" not in x:
            continue
        a = short.get(key, key)
        for sname, n in (("typed_decisions", "td"), ("banking77", "bank"), ("clinc150", "clinc")):
            k[f"e14/{a}/sets/{n}"] = _k(_g(x, "sets", sname, "accuracy"))
        k[f"e14/{a}/sets/mean"] = _k(x.get("mean_accuracy"))
    if _g(us, "reproduction", "reproduced") is not None:
        k["e14/g/reproduced"] = "yes" if us["reproduction"]["reproduced"] else "no"
    co = r.get("corpus") or {}
    for key, m in (co.get("rows") or {}).items():
        if key in short:
            a = short[key]
            k[f"e14/{a}/c/acc"] = _k(m.get("c_acc"))
            k[f"e14/{a}/c/offplan"] = _k(m.get("c_acc_offplan"))
            k[f"e14/{a}/c/ask"] = _k(m.get("c_acc_ask"))
            k[f"e14/{a}/c/ece"] = _k(m.get("c_ece"), 3, False)
            k[f"e14/{a}/fig4b/corpus_acc"] = _k(m.get("c_acc"))
    for c, x in (co.get("contrasts") or {}).items():
        a, ref = c.split("_vs_")
        a = short.get(a, a)
        ref = {"sl0.8b": "sl08", "sft_v1": "sftv1"}.get(ref, ref).replace("_", "")
        for stt in ("all", "offplan", "ask"):
            k[f"e14/{a}/c/vs{ref}/{stt}"] = _k(x[stt]["delta"])
            k[f"e14/{a}/c/vs{ref}/{stt}/lo"] = _k(x[stt]["lo"])
            k[f"e14/{a}/c/vs{ref}/{stt}/hi"] = _k(x[stt]["hi"])
    st = r.get("static") or {}
    for key, x in (st.get("points") or {}).items():
        if not isinstance(x, Mapping) or not x.get("available"):
            continue
        a = short[key]
        for kk, vv in E9._point_keys(key, x).items():
            k[kk] = vv
        s_ = x["stats"]
        k[f"e14/{a}/e1/acc"] = _k(s_["acc"])
        for f in ("toolchoice", "args", "w2c"):
            k[f"e14/{a}/e1/{f}"] = _k(_g(s_, "families", f, "acc"))
        k[f"e14/{a}/e1/usdk"] = E9._sig(1000 * s_["usd"])
        k[f"e14/{a}/e1/lat"] = _k(s_["lat_ms_median"], 0, False)
        k[f"e14/{a}/e1/vsrl"] = _k(_g(x, "gap_to_frozen_pt", "est"), 1, False)
        k[f"e14/{a}/e1/vssl08"] = _k(_g(x, "gap_to_sl08_pt", "est"), 1, False)
        k[f"e14/{a}/e1/front"] = "yes" if x.get("on_frontier_all") else "no"
        k[f"e14/{a}/fig4b/e1_acc"] = _k(x.get("bfcl_tool_choice_acc"))
    refs = st.get("refs") or {}
    if refs:   # reference values re-exported under e14/ref/ so the appendix reads only this file's keys
        k["e14/ref/e1n"] = _k(3991, 0, False) if (st.get("points") or {}) else None
        for m, a in (("rl", "rl"), ("sl0.8b", "sl08"), ("jev", "jev"), ("flash", "flash")):
            if m in refs:
                k[f"e14/ref/{a}/e1acc"] = _k(refs[m]["acc"])
                k[f"e14/ref/{a}/e1usdk"] = E9._sig(1000 * refs[m]["usd"])
                k[f"e14/ref/{a}/e1toolchoice"] = _k(_g(refs[m], "families", "toolchoice", "acc"))
    rows = co.get("rows") or {}
    for m, a in (("rl", "rl"), ("sft_v1", "sftv1"), ("e5_q06", "q06"), ("e5_mc1", "mc1"), ("sl0.8b", "sl08")):
        if m in rows:
            k[f"e14/ref/{a}/c/acc"] = _k(rows[m].get("c_acc"))
            k[f"e14/ref/{a}/c/offplan"] = _k(rows[m].get("c_acc_offplan"))
            k[f"e14/ref/{a}/c/ask"] = _k(rows[m].get("c_acc_ask"))
    for m, a in (("frozen", "rl"), ("jev", "jev"), ("sl0.8b", "sl08")):
        x = (r.get("live_a") or {}).get(m)
        if isinstance(x, Mapping) and x.get("live_success") is not None:
            k[f"e14/ref/{a}/live"] = _k(x["live_success"])
            k[f"e14/ref/{a}/livewall"] = _k(x.get("wall_s"), 1, False)
    if "e5_q06" in rows and "rl" in rows:
        k["e14/ref/q06/offplan_vs_rl"] = _k(rows["e5_q06"]["c_acc_offplan"] - rows["rl"]["c_acc_offplan"])
    if st.get("frontier_with_e14"):
        fw = st["frontier_with_e14"]
        k["e9/front14/first"] = E9.tikz_name(fw[0])
        k["e9/front14/rest"] = ",".join(E9.tikz_name(m) for m in fw[1:])
        k["e9/front14/n"] = str(len(fw))
    if st.get("a_minus_g_pt"):
        k["e14/e1/amg"] = _k(st["a_minus_g_pt"]["est"], 1, False)
        k["e14/e1/amg/lo"] = _k(st["a_minus_g_pt"]["ci"][0], 1, False)
        k["e14/e1/amg/hi"] = _k(st["a_minus_g_pt"]["ci"][1], 1, False)
        for fam, g in (st.get("a_minus_g_family_pt") or {}).items():
            k[f"e14/e1/amg/{fam}"] = _k(g["est"], 1, False)
            k[f"e14/e1/amg/{fam}/lo"] = _k(g["ci"][0], 1, False)
            k[f"e14/e1/amg/{fam}/hi"] = _k(g["ci"][1], 1, False)
    for key, x in (r.get("e1_engines") or {}).items():
        if key == "engine_check":
            k["e14/engine/check"] = "passed" if x.get("pass") else "failed"
            k["e14/engine/dev/maxdp"] = _k(_g(x, "sft_v1_dev", "max_abs_prob_diff"), 3, False)
            k["e14/engine/dev/same"] = _k(_g(x, "sft_v1_dev", "same_argmax"), 0, False)
            k["e14/engine/dev/n"] = _k(_g(x, "sft_v1_dev", "n"), 0, False)
            k["e14/engine/e1dev/flips"] = str(_g(x, "e1_dev_prefix", "differ"))
            k["e14/engine/e1dev/n"] = str(_g(x, "e1_dev_prefix", "n"))
            for eng in ("v1", "v2"):
                if x.get(f"devcheck_{eng}_s") is not None:
                    k[f"e14/engine/dev/{eng}s"] = _k(x[f"devcheck_{eng}_s"], 0, False)
            continue
        kk = {"e14a": "a/v1", "e14a_v2": "a/v2", "e14g": "g/v1", "e14ag": "ag/v1"}[key]
        k[f"e14/e1cost/{kk}/e9"] = E9._sig(x["usd_per_1k_e9_rule"])
        if x.get("usd_per_1k_wall"):
            k[f"e14/e1cost/{kk}/wall"] = E9._sig(x["usd_per_1k_wall"])
            k[f"e14/e1cost/{kk}/walls"] = _k(x["wall_s"], 0, False)
        k[f"e14/e1cost/{kk}/lat"] = _k(x["lat_ms_median"], 0, False)
        if x.get("pred_agree_with_v1") is not None:
            k[f"e14/e1cost/{kk}/agree"] = _k(x["pred_agree_with_v1"])
    for key, x in (r.get("calibration") or {}).items():
        a = short.get(key, "4b" if key == "4b_logged" else key)
        for dp, c in (x.get("coverage") or {}).items():
            d = {"tool_select": "ts", "tool_params": "tp"}[dp]
            k[f"e14/{a}/cal/{d}/tau"] = (f"{c['tau']:.4g}" if c.get("tau") is not None and math.isfinite(c["tau"])
                                         else "always defers")
            if c.get("tau") is not None and math.isfinite(c["tau"]) and c["tau"] < 1:
                k[f"e14/{a}/cal/{d}/taugap"] = f"{1 - c['tau']:.1e}".replace("e-0", "e-")   # distance to 1
            k[f"e14/{a}/cal/{d}/cov"] = _k(c["coverage"])
            k[f"e14/{a}/cal/{d}/s1acc"] = _k(c["s1_acc_accepted"])
            k[f"e14/{a}/cal/{d}/casc"] = _k(c["cascade_acc"])
            k[f"e14/{a}/cal/{d}/llm"] = _k(c["s2_acc_all"])
    for lab, x in (r.get("latency") or {}).items():
        if not isinstance(x, Mapping) or "bench" not in x:
            continue
        a = {**short, "e14a_v2": "av2", "4b_end": "4bend"}.get(lab, lab.replace("_", ""))
        for c in ("c1", "c4"):
            b = _g(x, "bench", c) or {}
            k[f"e14/lat/{a}/{c}/p50"] = _k(b.get("p50_ms"), 0, False)
            k[f"e14/lat/{a}/{c}/p90"] = _k(b.get("p90_ms"), 0, False)
            k[f"e14/lat/{a}/{c}/step"] = _k(b.get("s1_ms_per_step_p50"), 0, False)
        k[f"e14/lat/{a}/mem"] = _k((x.get("serve_peak_mib") or 0) / 1024, 1, False) if x.get("serve_peak_mib") else None
        sp = x.get("speedup_vs_4b") or {}
        for c in ("c1", "c4"):
            k[f"e14/lat/{a}/{c}/speedup"] = _k(sp.get(f"{c}_p50"), 1, False)
    for arm, mib in (r.get("train_peak_mib") or {}).items():
        k[f"e14/{short.get(ARMS.get(arm, ''), arm.lower())}/train/mem"] = _k(mib / 1024, 1, False) if mib else None
    for key, x in (r.get("live_a") or {}).items():
        if key not in short or not isinstance(x, Mapping):
            continue
        a = short[key]
        k[f"e14/{a}/live/success"] = _k(x.get("live_success"))
        k[f"e14/{a}/fig4b/live_success"] = _k(x.get("live_success"))
        k[f"e14/{a}/live/own"] = _k(x.get("own_acc"))
        k[f"e14/{a}/live/n"] = str(x.get("live_n"))
        k[f"e14/{a}/live/steps"] = _k(x.get("steps"), 1, False)
        k[f"e14/{a}/live/api"] = _k(x.get("api_cost"), 4, False)
        for ref in ("frozen", "jev", "sl0.8b", "e14a"):
            y = x.get(f"vs_{ref}")
            if y:
                rr = ref.replace(".", "")
                k[f"e14/{a}/live/vs{rr}"] = _k(y["diff_pt"], 1, False)
                k[f"e14/{a}/live/vs{rr}/lo"] = _k(y["ci_pt"][0], 1, False)
                k[f"e14/{a}/live/vs{rr}/hi"] = _k(y["ci_pt"][1], 1, False)
    lb = r.get("live_b") or {}
    if lb.get("arms"):
        k["e14/b/n"] = str(lb["n_common"])
        for arm, x in lb["arms"].items():
            a = {"e14_final": "a", "llm_only": "llm", "harness_final": "hf4b"}[arm]
            k[f"e14/b/{a}/success"] = _k(x["success"])
            k[f"e14/b/{a}/api"] = _k(1000 * x["api_usd"], 2, False)
            k[f"e14/b/{a}/q"] = _k(1000 * x["q_usd"], 2, False)
            k[f"e14/b/{a}/tot"] = _k(1000 * x["usd_incl_gpu"], 2, False)
            k[f"e14/b/{a}/tot9"] = _k(1000 * x["usd_incl_gpu_e9"], 2, False)
            k[f"e14/b/{a}/totown"] = _k(1000 * x["usd_incl_gpu_ownhost"], 2, False)
            if x.get("usd_incl_gpu_logged_tariff") is not None:
                k[f"e14/b/{a}/tottariff"] = _k(1000 * x["usd_incl_gpu_logged_tariff"], 2, False)
            k[f"e14/b/{a}/wall"] = _k(x["wall_s_mean"], 1, False)
            k[f"e14/b/{a}/s1s"] = _k(x["s1_s"], 1, False)
            k[f"e14/b/{a}/s2s"] = _k(x["s2_s"], 1, False)
            if x.get("s2_calls"):
                k[f"e14/b/{a}/s2call"] = _k(x["s2_s"] / x["s2_calls"], 2, False)
            k[f"e14/b/{a}/toolcov"] = _k(x.get("tool_coverage"))
            k[f"e14/b/{a}/argcov"] = _k(x.get("arg_coverage"))
        for ref, c in lb["contrasts"].items():
            a = {"llm_only": "llm", "harness_final": "hf4b"}[ref]
            k[f"e14/b/vs{a}/success"] = _k(c["success"]["diff_pt"], 1, False)
            k[f"e14/b/vs{a}/success/lo"] = _k(c["success"]["ci_pt"][0], 1, False)
            k[f"e14/b/vs{a}/success/hi"] = _k(c["success"]["ci_pt"][1], 1, False)
            k[f"e14/b/vs{a}/saving"] = _k(c["saving_incl_gpu"]["est"])
            k[f"e14/b/vs{a}/saving/lo"] = _k(c["saving_incl_gpu"]["ci"][0])
            k[f"e14/b/vs{a}/saving/hi"] = _k(c["saving_incl_gpu"]["ci"][1])
            k[f"e14/b/vs{a}/wall"] = _k(c["wall_s"]["est"], 1, False)
            k[f"e14/b/vs{a}/saving_api"] = _k(c["saving_api"]["est"])
            if c.get("saving_vs_logged_tariff"):
                k[f"e14/b/vs{a}/saving_tariff"] = _k(c["saving_vs_logged_tariff"]["est"])
            if a == "llm":
                k["e14/a/saved"] = _k(c["saving_incl_gpu"]["est"])   # Table tab:train's "Saved" for arm A
    # Table tab:train rows in the shapes of the existing rows (coordinator, 2026-10-07): fixed = e9/<row>/acc (written
    # above), common = stage2_rl/<row>/corpus_acc, ask = cov/ask/<row>/ask_now/acc, live =
    # explore/startlux_live/<row>/live_success; rows e14a, e14g (and e14ag, which has no live row)
    for key in ("e14a", "e14g", "e14ag"):
        m = (co.get("rows") or {}).get(key)
        if m:
            k[f"stage2_rl/{key}/corpus_acc"] = _k(m.get("c_acc"))
            k[f"cov/ask/{key}/ask_now/acc"] = _k(m.get("c_acc_ask"))
        x = (r.get("live_a") or {}).get(key)
        if isinstance(x, Mapping) and x.get("live_success") is not None:
            k[f"explore/startlux_live/{key}/live_success"] = _k(x["live_success"])
            k[f"explore/startlux_live/{key}/live_n"] = str(x.get("live_n"))
    for key, a in (("e14a", "a"), ("e14g", "g"), ("e14ag", "ag")):   # E1 cost by wall clock (beside E9's rule)
        x = (r.get("e1_engines") or {}).get(key)
        if isinstance(x, Mapping) and x.get("usd_per_1k_wall"):
            k[f"e14/{a}/l1/usdk_wall"] = E9._sig(x["usd_per_1k_wall"])
            k[f"e14/{a}/l1/usdk"] = E9._sig(x["usd_per_1k_e9_rule"])
    for name, x in (r.get("readings") or {}).items():
        k[f"e14/reading/{name.split('/')[0]}/{name.split('/')[1].replace('_', '')}"] = (
            "held" if x.get("holds") else "did not hold")
    if r.get("gpu_hours"):
        k["e14/gpuh"] = _k(r["gpu_hours"], 1, False)
    if _g(r, "spend", "total") is not None:
        k["e14/spend"] = _k(r["spend"]["total"], 2, False)
    return {kk: vv for kk, vv in k.items() if vv is not None}


def write_tex(keys: Mapping[str, str], path: Path = TEX) -> Path:
    import explore_common as C

    head = ["% AUTO-GENERATED by scripts/explore_e14_small.py summarize -- do not edit by hand.",
            "% Exploratory row E14 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 'E14: a sub-1B decision model with a",
            "% scoring head'); descriptive, outside every Holm table. Keys e14/... and E9-style points e9/e14a/..., e9/e14g/...",
            f"% last written {_now()} at git {(C.git_head() or '?')[:12]}",
            r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(keys.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + body + [r"\makeatother", ""]))
    return path


# ------------------------------------------------------------------ records to commit (slimmed)
REC = OUT / "records"


def cmd_collect(a: argparse.Namespace) -> int:
    """The box records the readout needs, slimmed and gzipped into results/explore/e14_small/records/ (raw files stay
    in runs/ and on S3): training / evaluation / build records, C and calibration answers, latency statistics, the live
    runs (explore_e13_llms.slim_run), E1's answer records, the probe log."""
    import shutil

    import e5_analysis as E5A
    import explore_e13_llms as E13

    box = Path(a.box)
    REC.mkdir(parents=True, exist_ok=True)
    n = 0
    for pat in ("train_record_*.json", "eval/*.json", "build_*.json", "host*.json", "stage_times.jsonl", "env_*.txt",
                "e14venv_freeze.txt", "serve_windows.jsonl", "live_b_window.json", "frozen_verify.json",
                "cal_requests.sha256", "logs/gate_*.json", "train_runs.jsonl", "status.txt", "api_probe_log.txt"):
        for f in box.glob(pat):
            dst = REC / f.relative_to(box)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    for sub in ("s23", "cal"):
        for f in sorted((box / sub).glob("answers_*.jsonl")):
            dst = REC / sub / (f.name + ".gz")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(gzip.compress(f.read_bytes(), 9))
            n += 1
    lat = box / "latency" / "bench"
    if lat.exists():
        comp = {f.name: E5A.bench_stats(f) for f in sorted(lat.glob("bench_*.json"))}
        (REC / "latency").mkdir(parents=True, exist_ok=True)
        (REC / "latency" / "bench_stats.json").write_text(json.dumps(comp, indent=1) + "\n")
        for f in (box / "latency").glob("models_*.json"):
            shutil.copy2(f, REC / "latency" / f.name)
        n += 1
    for run in [*LIVE_RUNS]:
        if (RUNS / run).is_dir():
            E13.slim_run(RUNS / run, REC / "live" / run)
            n += 1
    e1 = RUNS / "e1_bfcl_when2call_e14"
    if e1.is_dir():
        (REC / "e1").mkdir(parents=True, exist_ok=True)
        for f in sorted(e1.glob("s1_*.jsonl")):
            (REC / "e1" / (f.name + ".gz")).write_bytes(gzip.compress(f.read_bytes(), 9))
            n += 1
        for name in ("experiment.yaml", "ledger.json", "summary.json"):
            if (e1 / name).exists():
                shutil.copy2(e1 / name, REC / "e1" / name)
    print(f"{n} record files -> {REC.relative_to(ROOT)}")
    return 0


# ------------------------------------------------------------------ the serving-engine check (Deviations "serving-engine fix")
def cmd_devcheck(a: argparse.Namespace) -> int:
    """SFT v1's dev records (wire requests only: state and questions, never the gold) through a served engine, 4 at a
    time; answers by record id -> JSONL (resumes)."""
    import concurrent.futures as cf

    import httpx

    recs = _jl(Path(a.data))
    out = Path(a.out)
    done = {r["id"] for r in _jl(out)} if out.exists() else set()
    todo = [r for r in recs if r["id"] not in done]
    cl = httpx.Client(base_url=a.url, timeout=120.0)

    def one(r: Mapping[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        resp = cl.post("/v1/systemone", json={"model": "devcheck", "state": r["state"], "questions": r["questions"]})
        resp.raise_for_status()
        d = resp.json()
        return {"id": r["id"], "answers": d["answers"], "ms": (time.perf_counter() - t0) * 1000,
                "coalesced": (d.get("usage") or {}).get("coalesced")}

    t0 = time.perf_counter()
    with open(out, "a") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(one, todo):
            f.write(json.dumps(res) + "\n")
    print(json.dumps({"records": len(recs), "answered_now": len(todo), "seconds": round(time.perf_counter() - t0, 1)}))
    return 0


def _probs_of(a: Mapping[str, Any]) -> dict[str, float]:
    if a.get("type") == "noul":
        return {"true": float(a["noul"]), "false": 1.0 - float(a["noul"])}
    return {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}


def compare_answers(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], tie: float = 1e-3,
                    tol: float = 1e-3) -> dict[str, Any]:
    """(v1 answer, v2 answer) per question -> argmax agreement (near-ties excepted), max |dp|."""
    n = same = ties = 0
    worst = 0.0
    bad = []
    for x, y in pairs:
        px, py = _probs_of(x), _probs_of(y)
        n += 1
        ax, ay = max(px, key=px.get), max(py, key=py.get)
        srt = sorted(px.values(), reverse=True)
        near = len(srt) > 1 and srt[0] - srt[1] < tie
        d = max(abs(px[k] - py.get(k, 0.0)) for k in px)
        worst = max(worst, d)
        if ax == ay:
            same += 1
        elif near:
            ties += 1
        else:
            bad.append({"v1": ax, "v2": ay, "max_abs": d})
    return {"n": n, "same_argmax": same, "differ_near_tie": ties, "differ": len(bad), "max_abs_prob_diff": worst,
            "pass": not bad and worst <= tol, "examples": bad[:5], "tie": tie, "tol": tol}


def cmd_compare_engines(a: argparse.Namespace) -> int:
    """Registration "serving-engine fix" (c): SFT v1 dev records and E1's dev prefix, engine v1 against engine v2."""
    res: dict[str, Any] = {"at": _now()}
    v1 = {r["id"]: r["answers"] for r in _jl(Path(a.dev_v1))}
    v2 = {r["id"]: r["answers"] for r in _jl(Path(a.dev_v2))}
    ids = sorted(set(v1) & set(v2))
    res["sft_v1_dev"] = compare_answers([(v1[i][q], v2[i][q]) for i in ids for q in v1[i]])
    res["sft_v1_dev"]["records"] = len(ids)
    e1a = {r["id"]: r for r in _jl(Path(a.e1_v1))}
    e1b = {r["id"]: r for r in _jl(Path(a.e1_v2))}
    common = sorted(set(e1a) & set(e1b))
    pairs = []
    for i in common:   # decision_eval records: the predicted option and its confidence (+ dist when logged)
        ra, rb = e1a[i], e1b[i]
        da = ra.get("dist") or {str(ra.get("pred")): float(ra.get("conf") or 0)}
        db = rb.get("dist") or {str(rb.get("pred")): float(rb.get("conf") or 0)}
        pairs.append(({"type": "choice", "probabilities": da}, {"type": "choice", "probabilities": db}))
    res["e1_dev_prefix"] = {**compare_answers(pairs), "items": len(common),
                            "pred_agree": sum(e1a[i].get("pred") == e1b[i].get("pred") for i in common),
                            "conf_max_abs": max((abs(float(e1a[i].get("conf") or 0) - float(e1b[i].get("conf") or 0))
                                                 for i in common), default=None)}
    res["pass"] = bool(res["sft_v1_dev"]["pass"] and res["e1_dev_prefix"]["differ"] == 0
                       and (res["e1_dev_prefix"]["conf_max_abs"] or 0) <= 1e-3)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "examples"})
                      for k, v in res.items()}))
    return 0 if res["pass"] else 1


# ================================================================================================ CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--log", default=str(OUT / "api_probe_log.txt"))
    p.set_defaults(fn=cmd_probe)
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--cap", type=float, default=40.0)
    w.add_argument("--runs", default="runs")
    w.add_argument("--stop", default="runs/e14_STOP.json")
    w.add_argument("--every", type=int, default=30)
    w.add_argument("--no-pause", action="store_true")
    w.set_defaults(fn=cmd_watch)
    e = sub.add_parser("errored")
    e.add_argument("--run", required=True)
    e.set_defaults(fn=cmd_errored)
    s = sub.add_parser("spend")
    s.add_argument("--runs", default="runs")
    s.add_argument("--short", action="store_true")
    s.set_defaults(fn=cmd_spend)
    g = sub.add_parser("gate")
    g.add_argument("--runs", default="runs")
    g.add_argument("--smoke", required=True)
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--cap", type=float, default=40.0)
    g.set_defaults(fn=cmd_gate)
    f = sub.add_parser("fit")
    f.add_argument("--arm", default="A")
    f.add_argument("--box", default=str(BOX))
    f.set_defaults(fn=cmd_fit)
    pl = sub.add_parser("prepare-live")
    pl.add_argument("--arm", default="A")
    pl.set_defaults(fn=cmd_prepare_live)
    dc = sub.add_parser("devcheck")
    dc.add_argument("--url", required=True)
    dc.add_argument("--data", required=True)
    dc.add_argument("--out", required=True)
    dc.add_argument("--workers", type=int, default=4)
    dc.set_defaults(fn=cmd_devcheck)
    ce = sub.add_parser("compare-engines")
    ce.add_argument("--dev-v1", required=True)
    ce.add_argument("--dev-v2", required=True)
    ce.add_argument("--e1-v1", required=True)
    ce.add_argument("--e1-v2", required=True)
    ce.add_argument("--out", required=True)
    ce.set_defaults(fn=cmd_compare_engines)
    co = sub.add_parser("collect")
    co.add_argument("--box", default=str(BOX))
    co.set_defaults(fn=cmd_collect)
    sm = sub.add_parser("summarize")
    sm.add_argument("--box", default=str(BOX))
    sm.set_defaults(fn=cmd_summarize)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
