#!/usr/bin/env python3
r"""E7 (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry
"Exploratory row E7 (JevGym-Hard)" of 2026-10-06): JevGym-Hard, a harder JevGym setting tuned on development seeds so
that Flash alone succeeds 60-80 %, and the paper's main configurations on it (the Stage-2 Jev profile and the frozen
JevAgentRL-4B profile) beside LLM-only, on fresh seeds 310000-310199. No test; outside every Holm table; nothing here
revises a registered result.

    python scripts/explore_e7_hard.py pilot                        # the difficulty pilot -> results/explore/e7_hard/pilot.{json,md}
    python scripts/explore_e7_hard.py spend [--prior USD]          # logged $ of the E7 runs on this machine (+ the pilot's)
    python scripts/explore_e7_hard.py watch --pid PID --cap 150 --prior USD   # watchdog: stops PID's group at the cap
    python scripts/explore_e7_hard.py project --prior USD          # smoke -> evaluation projection (the registered gate)
    python scripts/explore_e7_hard.py summarize                    # results/explore/e7_hard/summary.{json,md} + e7_hard.tex
    python scripts/explore_e7_hard.py --exp e7b summarize          # E7b: summary_e7b.{json,md} + e7b_hard.tex

Readout (fixed in the Deviations entry before any evaluation episode): paired complete case per arm against
``llm_only`` over the evaluation tasks; success and Delta in points with the paired task-bootstrap 95 % interval
(B = 10,000, seed 0) and the Newcombe interval; paired wins and losses; cost per episode (API as logged; the
JevAgentRL arm plus the provisioned GPU tariff q = host $/h x evaluation window / completed JevAgentRL episodes, as
S2-4's convention, with q = S2-4's frozen tariff and q = 0 beside it) and the saving 1 - C_arm / C_ref (task
bootstrap); coverage (share of the tool decisions -- tool choice and arguments -- executed from the decision model's
answer) per decision point and pooled, and the reflection trigger's firing rate; mean and median wall time per
episode; the decision model's accuracy against the gold label on the run's own states (all its tool answers, and
the accepted ones) and, separately, on the cancel-vs-ship decisions of the tier rule (tool choice where the gold next
tool is a write; the rule is new to every decision model); failures by type, grouped as wrong final total / wrong
state-changing call / ask-or-finish error."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import re
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402
import module_map as MM  # noqa: E402

from jevharness.bench import metrics as M  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "results/explore/e7_hard"
BOX_DIR = OUT_DIR / "box"
TEX = ROOT / "paper/generated/e7_hard.tex"
RUN_SMOKE, RUN_EVAL, PILOT_GLOB = "e7_hard_smoke", "e7_hard", "e7_pilot_*"
CAP_USD, SAFETY, N_EVAL, N_MIN = 150.0, 1.5, 200, 100
REF = "llm_only"
ARMS = (REF, "jevagent_best", "jevagent_best_rl")
GPU_ARMS = ("jevagent_best_rl",)
# E7 and its addendum E7b (12 customers, fresh seeds 310200-310399; Deviations 2026-10-06 "E7b"): run names, records,
# outputs, the $ cap of each and its TeX key prefix
EXPS: dict[str, dict[str, Any]] = {
    "e7": {"smoke": RUN_SMOKE, "eval": RUN_EVAL, "box": BOX_DIR, "tex": TEX, "summary": "summary", "cap": CAP_USD,
           "stop": "e7_STOP.json", "prefix": "e7", "title": "E7: JevGym-Hard", "entry": "Exploratory row E7 (JevGym-Hard)"},
    "e7b": {"smoke": "e7b_hard_smoke", "eval": "e7b_hard", "box": OUT_DIR / "box_e7b",
            "tex": ROOT / "paper/generated/e7b_hard.tex", "summary": "summary_e7b", "cap": 60.0, "stop": "e7b_STOP.json",
            "prefix": "e7b", "title": "E7b: JevGym-Hard with 12 customers", "entry": "E7b (JevGym-Hard, 12 customers)"},
    # E7c: E7b's setting with the cascade thresholds refitted on the setting's calibration arm (e7c_cal, counted in
    # its $ cap); fresh seeds 310400-310599 (Deviations 2026-10-06 "E7c (JevGym-Hard, refitted thresholds)")
    "e7c": {"smoke": "e7c_hard_smoke", "eval": "e7c_hard", "box": OUT_DIR / "box_e7c",
            "tex": ROOT / "paper/generated/e7c_hard.tex", "summary": "summary_e7c", "cap": 70.0, "stop": "e7c_STOP.json",
            "prefix": "e7c", "title": "E7c: JevGym-Hard (12 customers) with refitted thresholds",
            "entry": "E7c (JevGym-Hard, refitted thresholds)", "arms": ("llm_only", "jev_refit", "rl_refit"),
            "gpu_arms": ("rl_refit",), "extra_runs": ("e7c_cal",), "safety": 1.2},
    # E7c's direct-API row (Deviations 2026-10-06 "E7c direct-API row"): OpenRouter's org budget exhausted, so Flash on
    # the direct Gemini API for llm_only and rl_refit; jev_refit not run (Jev is only on OpenRouter). Keys e7c/...
    "e7cd": {"smoke": "e7cd_hard_smoke", "eval": "e7cd_hard", "box": OUT_DIR / "box_e7cd",
             "tex": ROOT / "paper/generated/e7c_hard.tex", "summary": "summary_e7c", "cap": 70.0, "stop": "e7cd_STOP.json",
             "prefix": "e7c", "title": "E7c: JevGym-Hard (12 customers) with refitted thresholds, direct-API row",
             "entry": "E7c direct-API row", "arms": ("llm_only", "rl_refit"), "gpu_arms": ("rl_refit",),
             "extra_runs": (), "safety": 1.2, "not_run": {"jev_refit": "Jev is reachable only through "
             "OpenRouter, whose org budget is exhausted (HTTP 403 since about 19:05 UTC)"},
             "note": "Flash on the direct Gemini API in both arms (the registered E7c route, OpenRouter, was exhausted): "
                     "success and cost compare within this row only; the thresholds were fitted on E7c's calibration "
                     "arm, which ran on the OpenRouter route"},
    # E7c's Jev row, local (Deviations 2026-10-06 "E7c Jev row, local"): llm_only and jev_refit on E7c's first 100
    # tasks in the development container; Flash on the direct API, Jev on OpenRouter with the author's Jev-only key.
    # Keys e7c/...; E7c's spend so far ($16.05) comes in as the prior; cap $90 for E7c in total
    "e7cj": {"smoke": "e7cj_hard_smoke", "eval": "e7cj_hard", "box": OUT_DIR / "box_e7cj",
             "tex": ROOT / "paper/generated/e7c_hard.tex", "summary": "summary_e7c", "cap": 90.0, "stop": "e7cj_STOP.json",
             "prefix": "e7c", "title": "E7c: JevGym-Hard (12 customers) with refitted thresholds, Jev row (local)",
             "entry": "E7c Jev row, local", "arms": ("llm_only", "jev_refit"), "gpu_arms": (), "extra_runs": (),
             "safety": 1.2, "n_eval": 100, "n_min": 100,
             "not_run": {"rl_refit": "not run live: its calibration turns its tool choice off (the reported result)"},
             "note": "Flash on the direct Gemini API in both arms, Jev on OpenRouter with the author's Jev-only key "
                     "(the org's OpenRouter budget is exhausted): dollars compare within this row only; the "
                     "thresholds were fitted on E7c's calibration arm (OpenRouter route)"},
    # E7e (Deviations 2026-10-06 "E7e: itemized final answer"): E7c's Jev row on E7c's other 100 registered tasks
    # (310500-310599, never run) with the task's last sentence asking for one line per customer before the total
    # (suite option answer_style: itemized, jevgym_hard.py), in both arms; same thresholds, route and key as E7c's Jev
    # row. Motivated by E7c's failure analysis (every failure a wrong final total; bare-number answers almost never
    # matched). Keys e7e/...; its own cap $60
    "e7e": {"smoke": "e7e_hard_smoke", "eval": "e7e_hard", "box": OUT_DIR / "box_e7e",
            "tex": ROOT / "paper/generated/e7e_hard.tex", "summary": "summary_e7e", "cap": 60.0, "stop": "e7e_STOP.json",
            "prefix": "e7e", "title": "E7e: JevGym-Hard (12 customers), refitted Jev cascade, itemized final answer",
            "entry": "E7e: itemized final answer", "arms": ("llm_only", "jev_refit"), "gpu_arms": (), "extra_runs": (),
            "safety": 1.2, "n_eval": 100, "n_min": 100,
            "not_run": {"rl_refit": "not run live: its calibration turns its tool choice off (E7c)"},
            "note": "Flash on the direct Gemini API in both arms, Jev on OpenRouter with the author's Jev-only key: "
                    "dollars compare within this row only; thresholds from E7c's calibration arm; the task asks for "
                    "one line per customer before the total in both arms"},
    # E7f (Deviations 2026-10-07 "E7f: AgentDecision-4B (tier) live"): E7d's continued checkpoint run live at E7d's fitted
    # thresholds beside llm_only, in E7e's setting (12 customers, itemized answer), on fresh seeds 310600-310699 (n = 100;
    # 310600-310799 reserved), Flash on the direct Gemini API, the checkpoint served on the E7 host. Keys e7f/...;
    # outputs in results/explore/e7f/; cap $45; gate: 100 tasks, else the first 80, else stop after the smoke
    "e7f": {"smoke": "e7f_hard_smoke", "eval": "e7f_hard", "box": OUT_DIR / "box_e7f",
            "tex": ROOT / "paper/generated/e7f_hard.tex", "summary": "summary", "out": ROOT / "results/explore/e7f",
            "cap": 45.0, "stop": "e7f_STOP.json", "prefix": "e7f",
            "title": "E7f: JevGym-Hard (12 customers, itemized answer), AgentDecision-4B (tier) live",
            "entry": "E7f: AgentDecision-4B (tier) live", "arms": ("llm_only", "rl_tier"), "gpu_arms": ("rl_tier",),
            "safety": 1.05, "n_eval": 100, "n_min": 80, "diagnose": ("rl_tier",),
            "not_run": {"rl_frozen_refit": "the frozen 4B at its registered refit (E7c) always defers at tool choice, so "
                                          "the arm would cost about as much as llm_only and show nothing E7c's fit has "
                                          "not shown"},
            "note": "Flash on the direct Gemini API in both arms: dollars compare within this row only; rl_tier = "
                    "AgentDecision-4B (tier) (E7d's continued checkpoint) served on the host at E7d's fitted thresholds "
                    "(tool choice 0.99993, arguments 0.99291); the task asks for one line per customer before the total",
            # the E7f extension (Deviations 2026-10-07 "E7f extension"): the readout pools both runs (200 tasks) and
            # reports each part; $45 for the first part plus $70 for the extension
            "eval_parts": (("first", "e7f_hard", OUT_DIR / "box_e7f"), ("ext", "e7fx_hard", OUT_DIR / "box_e7fx")),
            "cap_total": 115.0},
    # the E7f extension's own run (driver, gate and watchdog only; its readout is E7f's): the remaining registered
    # seeds 310680-310799 (n = 120), the same arms, checkpoint, thresholds, setting and workers; no smoke: the gate
    # projects from the first part's logged per-episode costs (projection_basis); cap $70
    "e7fx": {"smoke": "e7fx_none", "eval": "e7fx_hard", "box": OUT_DIR / "box_e7fx",
             "tex": ROOT / "paper/generated/e7fx_hard.tex", "summary": "summary_ext_only",
             "out": ROOT / "results/explore/e7f", "cap": 70.0, "stop": "e7fx_STOP.json", "prefix": "e7fx",
             "title": "E7f extension: the remaining 120 tasks", "entry": "E7f extension",
             "arms": ("llm_only", "rl_tier"), "gpu_arms": ("rl_tier",), "safety": 1.05, "n_eval": 120, "n_min": 120,
             "projection_basis": "e7f_hard"},
    # E19 (Deviations 2026-10-08 "E19: the rule-randomised model on DecisionGym-Hard"): E12b's checkpoint
    # "AgentDecision-4B (random rules)" (trained on randomised order rules, never the tier rule) run live at thresholds
    # refitted on E7d's 20 calibration tasks, in E7f's setting on E7f's 200 tasks (seeds 310600-310799, one run
    # runs/e19_hard). Only the one arm runs: llm_only is E7f's LOGGED arm (ref_runs: both parts), paired per task, and
    # E7f's rl_tier is a second, descriptive comparison (extra_refs). Outputs in results/explore/e19_hard_random/,
    # keys e19/...; the cap as registered; the gate runs all 200, else the largest multiple of 20 that fits (n_step)
    "e19": {"smoke": "e19_hard_smoke", "eval": "e19_hard", "box": ROOT / "results/explore/e19_hard_random/box",
            "tex": ROOT / "paper/generated/e19_hard_random.tex", "summary": "summary",
            "out": ROOT / "results/explore/e19_hard_random", "cap": 62.0, "stop": "e19_STOP.json", "prefix": "e19",
            "title": "E19: JevGym-Hard (12 customers, itemized answer), AgentDecision-4B (random rules) live",
            "entry": "E19: the rule-randomised model on DecisionGym-Hard", "arms": ("llm_only", "rl_random"),
            "label": "exploratory (E19)",
            "gpu_arms": ("rl_random",), "safety": 1.05, "n_eval": 200, "n_min": 20, "n_step": 20,
            "diagnose": ("rl_random",), "ref_runs": ("e7f_hard", "e7fx_hard"),
            "extra_refs": {"vs_tier": ("rl_tier", ("e7f_hard", "e7fx_hard"))},
            "note": "Flash on the direct Gemini API, as E7f's arms (the same route and list prices); rl_random = "
                    "AgentDecision-4B (random rules) (E12b's checkpoint, never trained on the tier rule) served on the "
                    "host at thresholds refitted on E7d's 20 calibration tasks; llm_only is E7f's logged arm on the same "
                    "tasks, not re-run; the task asks for one line per customer before the total",
            # the E19 extension (Deviations 2026-10-08 "E19 extension: the remaining 20 tasks"): the registered gate ran
            # the first 180 tasks at the $56 cap; the remaining 20 ran under a total cap of $62. The readout pools both
            # runs (200 tasks) and reports each part; the $ cap above is the total over the smoke and both parts
            "eval_parts": (("first", "e19_hard", ROOT / "results/explore/e19_hard_random/box"),
                           ("ext", "e19x_hard", ROOT / "results/explore/e19_hard_random/box_e19x")),
            "cap_total": 62.0},
    # the E19 extension's own run (driver, gate and watchdog only; its readout is E19's): seeds 310780-310799 (n = 20),
    # the same arm, checkpoint, thresholds, setting, route and workers; no smoke: the gate projects from the first part's
    # logged per-episode cost plus the spend so far of the smoke and the first part (extra_runs), against the $62 total
    "e19x": {"smoke": "e19x_none", "eval": "e19x_hard", "box": ROOT / "results/explore/e19_hard_random/box_e19x",
             "tex": ROOT / "paper/generated/e19x_hard_random.tex", "summary": "summary_ext_only",
             "out": ROOT / "results/explore/e19_hard_random", "cap": 62.0, "stop": "e19_STOP.json", "prefix": "e19x",
             "title": "E19 extension: the remaining 20 tasks", "entry": "E19 extension: the remaining 20 tasks",
             "arms": ("llm_only", "rl_random"), "gpu_arms": ("rl_random",), "safety": 1.05, "n_eval": 20, "n_min": 20,
             "projection_basis": "e19_hard", "extra_runs": ("e19_hard_smoke", "e19_hard"),
             "ref_runs": ("e7f_hard", "e7fx_hard")},
}
for _x in EXPS.values():
    _x.setdefault("n_eval", N_EVAL)
    _x.setdefault("n_min", N_MIN)
    _x.setdefault("arms", ARMS)
    _x.setdefault("gpu_arms", GPU_ARMS)
    _x.setdefault("extra_runs", ())
    _x.setdefault("safety", SAFETY)
    _x.setdefault("out", OUT_DIR)
    _x.setdefault("diagnose", ())
    _x.setdefault("eval_parts", ())
LABEL = "exploratory (E7)"
TOOL_DPS = ("tool_select", "tool_params")
OP_TOOLS = ("cancel_order", "ship_order")
# on-demand $/h, us-east-1 (AWS price list): the host that served the run sets the tariff (as harness_final)
HOST_USD_PER_HOUR = {"g5.8xlarge": 2.448, "g5.4xlarge": 1.624, "g5.2xlarge": 1.212, "g6e.2xlarge": 2.24208}
S24_TARIFF = ROOT / "results/stage2/tariff.json"
NAN = float("nan")


# ============================================================================== spend, watchdog, projection
def episode_spend(t: Mapping[str, Any]) -> dict[str, float]:
    """$ of one logged episode: System-2 (Gemini via OpenRouter) and System-1 (Jev; the served 4B is priced at 0).
    Shadow (calibration) System-1 calls are logged apart from ``total`` (``usage.shadow``) and are added here: they
    were paid (E7c's calibration arms)."""
    u = t.get("usage") or {}
    s1 = float((u.get("s1") or {}).get("cost_usd") or 0.0)
    s2 = float((u.get("s2") or {}).get("cost_usd") or 0.0)
    tot = float((u.get("total") or {}).get("cost_usd") or 0.0)
    sh = float((u.get("shadow") or {}).get("cost_usd") or 0.0)
    return {"s2": s2 + max(tot - s1 - s2, 0.0), "s1": s1 + sh, "total": tot + sh}


def arm_file(run_dir: Path, arm: str) -> Path:
    """``<run_dir>/<arm>.jsonl``, or its slimmed, gzipped copy ``<arm>.jsonl.gz`` (E13 keeps its records that way) when
    only that exists; the plain path when neither does."""
    p = Path(run_dir) / f"{arm}.jsonl"
    gz = p.with_name(p.name + ".gz")
    return gz if not p.exists() and gz.exists() else p


def _records(run_dir: Path):  # noqa: ANN202
    files = sorted(Path(run_dir).glob("*.jsonl")) + sorted(p for p in Path(run_dir).glob("*.jsonl.gz")
                                                             if not p.with_suffix("").exists())
    for p in files:
        if p.name.startswith("schedule"):
            continue
        text = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
        for line in text.splitlines():
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if isinstance(t, dict) and "task_id" in t:
                yield p.stem, t


def run_spend(run_dir: Path) -> dict[str, Any]:
    """Every record of every arm (errored attempts and re-runs included: all of them were paid)."""
    out = {"s2": 0.0, "s1": 0.0, "total": 0.0, "records": 0}
    for _, t in _records(run_dir):
        s = episode_spend(t)
        for k in ("s2", "s1", "total"):
            out[k] += s[k]
        out["records"] += 1
    return out


def total_spend(runs: Path = RUNS, prior: float = 0.0, pilots: bool = False, exp: str = "e7") -> dict[str, Any]:
    X = EXPS[exp]
    names = list(dict.fromkeys([*X["extra_runs"], X["smoke"], X["eval"], *(p[1] for p in X["eval_parts"])]))
    names += [p.name for p in sorted(runs.glob(PILOT_GLOB))] if pilots else []
    per = {n: run_spend(runs / n) for n in names if (runs / n).is_dir()}
    tot = {k: sum(v[k] for v in per.values()) for k in ("s2", "s1", "total", "records")}
    return {"by_run": per, "prior": prior, **tot, "total_with_prior": tot["total"] + prior}


def cmd_spend(a: argparse.Namespace) -> int:
    s = total_spend(Path(a.runs), a.prior, a.pilots, a.exp)
    if a.short:
        print(f"spend ${s['total']:.2f} here + ${a.prior:.2f} prior = ${s['total_with_prior']:.2f}")
    else:
        print(json.dumps(s, indent=1))
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    """Poll the logged spend every ``--every`` s; when it plus ``--prior`` reaches the cap, SIGTERM (then SIGKILL) the
    runner's process group and write ``<runs>/<exp>_STOP.json``. Episodes in flight are not logged (a few cents)."""
    stop = Path(a.runs) / EXPS[a.exp]["stop"]
    cap = EXPS[a.exp]["cap"] if a.cap is None else a.cap
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(Path(a.runs), a.prior, exp=a.exp)
        if s["total_with_prior"] >= cap:
            stop.write_text(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                        "spend": s, "cap": cap}, indent=1))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(a.pid), sig)
                except OSError:
                    pass
                time.sleep(10)
            print(f"CAP REACHED: ${s['total_with_prior']:.2f} >= ${cap:.2f}; runner stopped", flush=True)
            return 1
        print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} spend ${s['total_with_prior']:.3f} (here ${s['total']:.3f}, "
              f"System-2 ${s['s2']:.3f}, Jev ${s['s1']:.4f}; {s['records']} records)", flush=True)
        time.sleep(a.every)


def projection(runs: Path = RUNS, prior: float = 0.0, n_eval: int | None = None, safety: float | None = None,
               cap: float | None = None, exp: str = "e7") -> dict[str, Any]:
    """The registered gate: the smoke's mean $ per episode per arm, summed over the arms, x the evaluation tasks x
    ``safety``, plus the spend so far (the pilot's included), against the cap; if it does not fit, the first ``N_MIN``
    tasks (same check); else stop after the smoke. The GPU tariff is not part of the $ cap (the GPU cap is in hours)."""
    X = EXPS[exp]
    cap = X["cap"] if cap is None else cap
    safety = X["safety"] if safety is None else safety
    n_eval = X["n_eval"] if n_eval is None else n_eval
    n_min = X["n_min"]
    basis = X.get("projection_basis") or X["smoke"]      # the smoke, or (E7f extension) the first part's evaluation
    run_arms = [arm for arm in X["arms"] if not (X.get("ref_runs") and arm == REF)]   # E19: llm_only is never re-run
    sm = {arm: list(C.load_latest([arm_file(runs / basis, arm)])[0].values())
          if arm_file(runs / basis, arm).exists() else [] for arm in run_arms}
    per = {arm: (float(np.mean([episode_spend(t)["total"] for t in ts])) if ts else None) for arm, ts in sm.items()}
    wall = {arm: (float(np.mean([float(t.get("wall_ms") or 0) for t in ts])) / 1000 if ts else None)
            for arm, ts in sm.items()}
    errors = {arm: sum(bool(t.get("error")) for t in ts) for arm, ts in sm.items()}
    so_far = total_spend(runs, prior, exp=exp)["total_with_prior"]
    ep = sum(v for v in per.values() if v is not None)
    out = {"basis": basis, "smoke_per_episode": per, "smoke_wall_s": wall, "smoke_n": {k: len(v) for k, v in sm.items()},
           "smoke_errors": errors, "missing_arms": [a for a, v in per.items() if v is None],
           "spent_so_far": so_far, "prior": prior, "safety": safety, "cap": cap, "per_task_all_arms": ep}
    step = X.get("n_step")       # E19: all n_eval tasks, else the largest multiple of n_step that fits
    cands = [n_eval] + (list(range(n_eval - step, 0, -step)) if step else ([n_min] if n_min < n_eval else []))
    for n in dict.fromkeys(cands):
        out[f"projected_{n}"] = ep * n * safety
        out[f"fits_{n}"] = so_far + ep * n * safety <= cap
    out["candidates"] = cands
    fit_n = next((n for n in cands if out[f"fits_{n}"]), None)
    out["decision"] = (f"run all {n_eval}" if fit_n == n_eval else f"run the first {fit_n}" if fit_n
                       else "stop after the smoke")
    if out["missing_arms"]:
        out["decision"] = "stop: smoke incomplete (" + ", ".join(out["missing_arms"]) + ")"
    elif any(errors.values()):
        out["decision"] = "stop: smoke errored (" + ", ".join(f"{k} {v}" for k, v in errors.items() if v) + ")"
    return out


def cmd_project(a: argparse.Namespace) -> int:
    p = projection(Path(a.runs), a.prior, exp=a.exp)
    out = Path(a.out) if a.out else EXPS[a.exp]["box"] / "projection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(p, indent=1))
    return 0


# ============================================================================== pilot
def suite_of(run_dir: Path) -> dict[str, Any]:
    exp = yaml.safe_load((run_dir / "experiment.yaml").read_text())
    return {"suite": exp.get("suite") or {}, "max_steps": (exp.get("harness_defaults") or {}).get("max_steps", 30),
            "llm": exp.get("llm")}


# development runs outside runs/e7_pilot_* (recorded in the Deviations entry): the aborted direct-API attempt and the
# OpenRouter route check of the cascade profile
PILOT_EXTRA = {
    "aborted_direct_api": {"episodes": 2, "errored": 2, "spend": 0.0, "seeds": "7006, 7008",
                           "note": "direct Gemini API, about 15:00 UTC 2026-10-06: HTTP 503 'high demand'; discarded"},
    "route_check": {"episodes": 4, "errored": 0, "spend": 0.1023, "seeds": "7290-7291",
                    "note": "llm_only and jevagent_best, 6 customers, Flash and Jev via OpenRouter: the route works"},
}
CHOSEN = {"families": "multi", "multi_customers": 10, "multi_rule": "tier", "max_steps": 45,
          "rule": "among the tier-rule candidates in the 60-80 % band, the one closest to 70 %"}


def pilot(runs: Path = RUNS) -> dict[str, Any]:
    """Every pilot candidate (development seeds 7000-7299, Flash LLM-only): its knobs, success per family, errors,
    retries, spend and failure types. Development numbers, never a result."""
    rows = []
    for d in sorted(runs.glob(PILOT_GLOB)):
        f = d / "llm_only.jsonl"
        if not f.exists():
            continue
        latest, attempts = C.load_latest([f])
        s = suite_of(d)
        ok = [t for t in latest.values() if not t.get("error")]
        fam: dict[str, list[bool]] = {}
        for t in ok:
            fam.setdefault(str(t["task_id"]).split("-")[1], []).append(C.success(t))
        retries = sum(max(int(c.get("attempts") or 1) - 1, 0) for t in latest.values() for c in t.get("calls") or [])
        ft: dict[str, int] = {}
        for t in ok:
            if not C.success(t):
                k = failure_type(t, s["suite"])
                ft[k] = ft.get(k, 0) + 1
        seeds = sorted(int(str(t).split("-")[2]) for t in latest)
        rows.append({"candidate": d.name, "suite": s["suite"], "max_steps": s["max_steps"], "llm": s["llm"],
                     "n": len(ok), "errored": len(latest) - len(ok), "retries": retries,
                     "success": float(np.mean([C.success(t) for t in ok])) if ok else NAN,
                     "by_family": {k: [int(sum(v)), len(v)] for k, v in fam.items()},
                     "failure_types": ft, "spend": run_spend(d)["total"],
                     "seeds": [seeds[0], seeds[-1]] if seeds else None,
                     "wall_s_mean": float(np.mean([float(t.get("wall_ms") or 0) / 1000 for t in ok])) if ok else NAN})
    spend = sum(r["spend"] for r in rows)
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "label": LABEL + ", development",
            "candidates": rows, "spend": spend, "extra": PILOT_EXTRA, "chosen": CHOSEN,
            "spend_total": spend + sum(x["spend"] for x in PILOT_EXTRA.values())}


def pilot_markdown(p: Mapping[str, Any]) -> str:
    out = ["# E7 difficulty pilot (development seeds; never a result)", "",
           f"Generated {p['generated']}; Flash LLM-only, temperature 0, minimal reasoning, via OpenRouter. "
           f"Total pilot spend ${p['spend']:.2f}.", "",
           "| candidate | families | k | rule | max steps | seeds | n | errored | success % | by family | failures | $ |",
           "|" + "---|" * 12]
    for r in p["candidates"]:
        s = r["suite"]
        fam = ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in r["by_family"].items())
        ft = ", ".join(f"{k} {v}" for k, v in r["failure_types"].items()) or "--"
        out.append(f"| `{r['candidate']}` | {s.get('families')} | {s.get('multi_customers', 3)} | "
                   f"{s.get('multi_rule', 'given')} | {r['max_steps']} | {r['seeds'][0]}-{r['seeds'][1]} | {r['n']} | "
                   f"{r['errored']} | {100 * r['success']:.1f} | {fam} | {ft} | {r['spend']:.2f} |")
    out += [""] + [f"- {k}: {v['episodes']} episodes (seeds {v['seeds']}), ${v['spend']:.2f}; {v['note']}"
                   for k, v in p.get("extra", {}).items()]
    out += [f"- pilot spend in total: ${p.get('spend_total', p['spend']):.2f}",
            f"- chosen: {p.get('chosen')}"]
    return "\n".join(out) + "\n"


def cmd_pilot(a: argparse.Namespace) -> int:
    p = pilot(Path(a.runs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pilot.json").write_text(json.dumps(MM._clean(p), indent=1, default=MM._json_default) + "\n")
    (OUT_DIR / "pilot.md").write_text(pilot_markdown(p))
    print(pilot_markdown(p))
    return 0


# ============================================================================== E7c calibration fit
CAL_RUN, CAL_DIR = "e7c_cal", OUT_DIR / "cal_e7c"
CAL_TEX = ROOT / "paper/generated/e7c_cal.tex"
CAL_SYS = {"jev_shadow_cal": "jev", "rl_shadow_cal": "rl4b"}                 # key names of the two System-1 models
CAL_ARMS = {"jev_shadow_cal": "jev_refit", "rl_shadow_cal": "rl_refit"}      # calibration arm -> evaluation arm


def _at_tau(ps: Sequence[Mapping[str, Any]], tau: float) -> dict[str, Any]:
    """On the calibration states: the share of decisions the cascade would take at ``tau`` (confidence >= tau) and the
    model's and the LLM's accuracy on them, beside both accuracies on every paired decision."""
    acc = [r for r in ps if np.isfinite(tau) and float(r["c"]) >= tau]
    return {"n": len(ps), "coverage": len(acc) / len(ps) if ps else NAN,
            "s1_acc_taken": _mean([float(r["s1_ok"]) for r in acc]), "llm_acc_taken": _mean([float(r["s2_ok"]) for r in acc]),
            "s1_acc_all": _mean([float(r["s1_ok"]) for r in ps]), "llm_acc_all": _mean([float(r["s2_ok"]) for r in ps])}


def fit_calibration(runs: Path = RUNS, out_dir: Path = CAL_DIR, run_name: str = CAL_RUN,
                    cal_arms: Mapping[str, str] = CAL_ARMS, tex: Path | None = CAL_TEX) -> dict[str, Any]:
    """The registered estimator (``threshold_theory --calibration-arm``, W = $1, lambda = 0, V fallback W; A6.5 / A6.11,
    as Stage 1 and E6a) per calibration arm and decision point; tau-hat = +inf means the fit accepts no confidence
    level: the cascade always defers there (A6.11 (c) 4). Defaults: E7c (``runs/e7c_cal``, both System-1 models, the
    ``e7c/cal/...`` keys); E13 passes its own run, arm map and output directory, and no key file (``tex=None``)."""
    import subprocess

    from jevharness.bench import threshold_theory as T

    run = runs / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fits: dict[str, Any] = {}
    for arm in cal_arms:
        f = run / f"{arm}.jsonl"
        if not f.exists():
            fits[arm] = {"state": "not run"}
            continue
        trajs = T.load_trajectories(f)
        fits[arm] = {"sha256": C.sha256_file(f), "eval_arm": cal_arms[arm],
                     "n_tasks": len({str(t.get("task_id")) for t in trajs}),
                     "errored": sum(bool(t.get("error")) for t in trajs)}
        for dp in TOOL_DPS:
            out = out_dir / f"tauhat_{arm}_{dp}.json"
            subprocess.run([sys.executable, "-m", "jevharness.bench.threshold_theory", str(run), "--arm", arm, "--dp", dp,
                            "--calibration-arm", "--task-value", "1.0", "--lam-per-hour", "0", "--v-fallback", "W",
                            "--format", "json", "--out", str(out)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            d = json.loads(out.read_text())
            tau = float(d.get("tau_hat")) if d.get("tau_hat") is not None else NAN
            ps = T.paired(T.decision_pairs(trajs, dp, arm=arm, gold="acceptable"))
            row = {"tau_hat": tau, "always_defer": bool(tau == float("inf")), "n_paired": d.get("n_paired"),
                   "tau_rule": (d.get("tau_rule") or {}).get("rule"), "V": (d.get("primary") or {}).get("V"),
                   "delta_c": (d.get("primary") or {}).get("delta_c"), "path": C.rel(out), "sha256": C.sha256_file(out),
                   "notes": d.get("notes"), "at_tau": _at_tau(ps, tau)}
            if dp == "tool_select":
                row["op_at_tau"] = _at_tau([r for r in ps if r.get("gold") in OP_TOOLS], tau)
            fits[arm][dp] = row
    res = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": C.git_head(),
           "run": C.rel(run), "estimator": "threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 "
                                            "--v-fallback W (A6.5 / A6.11)", "fits": fits,
           "spend": run_spend(run) if run.is_dir() else None}
    (out_dir / "fits.json").write_text(json.dumps(MM._clean(res), indent=1, default=MM._json_default) + "\n")
    (out_dir / "fits.md").write_text(calibration_markdown(res))
    if tex is not None:
        write_tex(cal_tex_keys(res), tex, "E7c calibration (JevGym-Hard, 12 customers; seeds 5300-5399)")
    return res


def cal_tex_keys(res: Mapping[str, Any]) -> dict[str, str]:
    """``e7c/cal/<model>/<dp>/...`` (dp: toolselect, toolparams, and ``op`` = the cancel-vs-ship subset of tool choice):
    the fitted tau-hat (``always defer`` for +inf), coverage at tau-hat on the calibration states, the model's accuracy on
    the accepted decisions and on all, the LLM's on the same, n decisions; n tasks per model."""
    v: dict[str, str] = {}
    pc = lambda x: f"{100 * x:.1f}" if MM._finite(x) else "--"  # noqa: E731
    for arm, f in (res.get("fits") or {}).items():
        if not isinstance(f, Mapping) or "tool_select" not in f:
            continue
        m = CAL_SYS.get(arm, arm.replace("_", ""))
        v[f"e7c/cal/{m}/ntasks"] = str(f.get("n_tasks"))
        v[f"e7c/cal/{m}/errored"] = str(f.get("errored"))
        for dp, key in (("tool_select", "toolselect"), ("tool_params", "toolparams"), ("op", "op")):
            r = f["tool_select"] if dp == "op" else f[dp]
            a = r["op_at_tau"] if dp == "op" else r["at_tau"]
            k = f"e7c/cal/{m}/{key}"
            v[f"{k}/tau"] = "always defer" if r["always_defer"] else f"{r['tau_hat']:.4g}"
            v[f"{k}/n"] = str(a["n"])
            v[f"{k}/coverage"] = pc(a["coverage"])
            v[f"{k}/s1acctaken"] = pc(a["s1_acc_taken"])
            v[f"{k}/llmacctaken"] = pc(a["llm_acc_taken"])
            v[f"{k}/s1accall"] = pc(a["s1_acc_all"])
            v[f"{k}/llmaccall"] = pc(a["llm_acc_all"])
    sp = res.get("spend") or {}
    if sp.get("total") is not None:
        v["e7c/cal/spend"] = f"{sp['total']:.2f}"
    v["e7c/cal/seeds"] = "5300--5399"
    return v


def calibration_markdown(res: Mapping[str, Any]) -> str:
    out = ["# E7c calibration fit (JevGym-Hard, 12 customers; calibration seeds 5300-5399)", "",
           f"Generated {res['generated']}; run `{res['run']}`; estimator: {res['estimator']}.", "",
           "| calibration arm | decision point | tau-hat | rule | paired | V | dC | coverage at tau % | model acc. taken % | "
           "LLM acc. taken % | model acc. all % | LLM acc. all % |", "|" + "---|" * 12]
    for arm, f in res["fits"].items():
        for dp in TOOL_DPS:
            r = f.get(dp)
            if not r:
                continue
            a = r["at_tau"]
            tau = "always defer" if r["always_defer"] else f"{r['tau_hat']:.10g}"
            out.append(f"| `{arm}` | {dp} | {tau} | {r['tau_rule']} | {r['n_paired']} | {r['V']} | {r['delta_c']} | "
                       f"{_pt(a['coverage'], False)} | {_pt(a['s1_acc_taken'], False)} | {_pt(a['llm_acc_taken'], False)} | "
                       f"{_pt(a['s1_acc_all'], False)} | {_pt(a['llm_acc_all'], False)} |")
    rl = (res["fits"].get("rl_shadow_cal") or {}).get("tool_select") or {}
    if rl.get("always_defer"):
        out += ["", "**Under the registered estimator, the procedure turns the frozen JevAgentRL-4B's tool choice off on this "
                "setting**: its tool_select fit accepts no confidence level (always defer), so the refitted 4B profile "
                "leaves tool choice to the LLM (its tool_params threshold is reached only inside System-1-chosen calls)."]
    out += ["", "Cancel-vs-ship decisions (tool choice where the gold next tool is a write), at the fitted tau:", ""]
    for arm, f in res["fits"].items():
        r = (f.get("tool_select") or {}).get("op_at_tau")
        if r:
            out.append(f"- `{arm}`: {r['n']} decisions, coverage {_pt(r['coverage'], False)} %, model acc. taken "
                       f"{_pt(r['s1_acc_taken'], False)} %, model acc. all {_pt(r['s1_acc_all'], False)} %, LLM acc. all "
                       f"{_pt(r['llm_acc_all'], False)} %")
    return "\n".join(out) + "\n"


def _calibration(prefix: str) -> dict[str, Any] | None:
    p = CAL_DIR / "fits.json"
    return json.loads(p.read_text()) if prefix == "e7c" and p.exists() else None


def cmd_fit(a: argparse.Namespace) -> int:
    print(calibration_markdown(fit_calibration(Path(a.runs))))
    return 0


# ============================================================================== latency breakdown (descriptive)
TAIL_S = 5.0


def _group_of(c: Mapping[str, Any]) -> str:
    """component / decision point / who served it: the provider OpenRouter reports for a System-2 call
    (``meta.served_provider``), else the backend; System-2 calls offered a single tool (the forced argument call of a
    delegated tool) are marked ``1tool``."""
    meta = c.get("meta") or {}
    comp, dp = str(c.get("component") or "?"), str(c.get("decision_point") or "?")
    who = meta.get("served_provider") or (c.get("backend") if comp == "s2" else c.get("model")) or "?"
    tools = "/1tool" if comp == "s2" and meta.get("n_tools") == 1 else ""
    return f"{comp}/{dp}{tools}/{who}"


def latency_breakdown(trajs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per call group (``_group_of``): n, p50 / p90 / mean seconds, the calls over ``TAIL_S`` s and the share of the
    arm's summed episode wall time spent in the group and in its tail; per episode: wall, System-1 and System-2 time
    (calls run one after another inside an episode, so the rest is environment, harness and queueing)."""
    groups: dict[str, list[float]] = {}
    wall = [float(t.get("wall_ms") or 0) / 1000 for t in trajs]
    tot_wall = sum(wall)
    comp_s: dict[str, float] = {}
    for t in trajs:
        for c in t.get("calls") or []:
            sec = float(c.get("latency_ms") or 0) / 1000
            groups.setdefault(_group_of(c), []).append(sec)
            comp_s[str(c.get("component"))] = comp_s.get(str(c.get("component")), 0.0) + sec
    rows = {}
    for g, xs in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        a = np.array(xs)
        tail = a[a > TAIL_S]
        rows[g] = {"n": int(len(a)), "p50": float(np.median(a)), "p90": float(np.quantile(a, 0.9)), "mean": float(a.mean()),
                   "n_over": int(len(tail)), "share_of_wall": float(a.sum() / tot_wall) if tot_wall else NAN,
                   "tail_share_of_wall": float(tail.sum() / tot_wall) if tot_wall else NAN,
                   "per_episode_s": float(a.sum() / len(trajs)) if trajs else NAN,
                   "tail_per_episode_s": float(tail.sum() / len(trajs)) if trajs else NAN}
    n = len(trajs)
    all_tail = sum(r["tail_per_episode_s"] for r in rows.values())
    s2_tail = sum(r["tail_per_episode_s"] for g, r in rows.items() if g.startswith("s2/"))
    return {"n_episodes": n, "wall_mean": _mean(wall), "s1_per_episode": comp_s.get("s1", 0.0) / n if n else NAN,
            "s2_per_episode": comp_s.get("s2", 0.0) / n if n else NAN, "tail_per_episode": all_tail,
            "s2_tail_per_episode": s2_tail, "groups": rows}


def latency_compare(lat: Mapping[str, Mapping[str, Any]], ref: str = REF) -> dict[str, Any]:
    """For each arm against ``ref``: the extra mean wall time per episode and the share of it that the System-2 tail
    (calls over ``TAIL_S`` s) accounts for (the arm's tail time per episode minus the reference's)."""
    out = {}
    r = lat.get(ref)
    for arm, a in lat.items():
        if arm == ref or not r:
            continue
        extra = a["wall_mean"] - r["wall_mean"]
        dtail = a["s2_tail_per_episode"] - r["s2_tail_per_episode"]
        out[arm] = {"extra_wall_s": extra, "extra_s2_tail_s": dtail, "extra_s1_s": a["s1_per_episode"] - r["s1_per_episode"],
                    "extra_s2_s": a["s2_per_episode"] - r["s2_per_episode"],
                    "tail_share_of_extra": dtail / extra if extra > 0 else NAN}
    return out


def latency_markdown(lat: Mapping[str, Any], cmp: Mapping[str, Any]) -> str:
    out = ["## Latency breakdown (descriptive; changes no reported number)", "",
           f"Calls per arm grouped by component / decision point / served provider (System-2: OpenRouter's "
           f"`meta.served_provider`; `1tool` = the forced single-tool argument call of a delegated tool); tail = calls "
           f"over {TAIL_S:g} s; shares are of the arm's summed episode wall time.", ""]
    for arm, a in lat.items():
        out += [f"### `{arm}`: wall {a['wall_mean']:.1f} s per episode; System-1 {a['s1_per_episode']:.1f} s, System-2 "
                f"{a['s2_per_episode']:.1f} s, System-2 tail {a['s2_tail_per_episode']:.1f} s per episode", "",
                "| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |", "|" + "---|" * 8]
        for g, r in a["groups"].items():
            out.append(f"| {g} | {r['n']} | {r['p50']:.2f} | {r['p90']:.2f} | {r['mean']:.2f} | {r['n_over']} | "
                       f"{100 * r['share_of_wall']:.1f} | {100 * r['tail_share_of_wall']:.1f} |")
        out.append("")
    for arm, c in cmp.items():
        out.append(f"- `{arm}` vs `{REF}`: +{c['extra_wall_s']:.1f} s per episode (System-1 {c['extra_s1_s']:+.1f} s, "
                   f"System-2 {c['extra_s2_s']:+.1f} s); the System-2 tail over {TAIL_S:g} s adds {c['extra_s2_tail_s']:+.1f} s, "
                   f"{_pt(c['tail_share_of_extra'], False)} % of the extra time")
    return "\n".join(out) + "\n"


# ============================================================================== readout
def _env_of(task_id: str, suite: Mapping[str, Any]) -> Any:
    from jevharness.bench.suites import jevgym_hard as H

    kw = {k: v for k, v in suite.items() if k in H.ENV_KWARGS}
    return H.JevGymHardEnv(seed=int(str(task_id).split("-")[2]), **kw)


FAILURE_CLASS = {"wrong_total": "wrong_total", "strict_stop": "wrong_write", "side_effect": "wrong_write",
                 "premature_finish": "ask_finish", "no_answer": "ask_finish", "asked": "ask_finish", "other": "other"}


def _replay(t: Mapping[str, Any], suite: Mapping[str, Any]) -> Any:
    """The episode's world after its logged calls (JevGym is deterministic): the env rebuilt from the seed and the
    suite settings, every logged non-final call stepped in order."""
    env = _env_of(str(t["task_id"]), suite)
    env.reset()
    for st in t.get("steps") or []:
        tool, args = st.get("tool"), st.get("args")
        if not tool or tool == env.FINAL_TOOL or not isinstance(args, Mapping):
            continue
        r = env.step(tool, dict(args))
        if r.done:
            break
    return env


def failure_type(t: Mapping[str, Any], suite: Mapping[str, Any]) -> str:
    """Why a failed JevGym episode failed: ``strict_stop`` (an irreversible wrong state-changing call ended it),
    ``side_effect`` (a harmful near-duplicate write), ``no_answer`` (no final answer: budget or stop),
    ``premature_finish`` (finished with gold writes still missing), ``asked`` (an ask_user in a multi task, writes
    complete), ``wrong_total`` (every write right, the multi task's total not in the answer), else ``other``.
    ``FAILURE_CLASS`` groups them: wrong final total / wrong state-changing call / ask-or-finish error."""
    res = t.get("result") or {}
    if res.get("irreversible_error"):
        return "strict_stop"
    if res.get("side_effects"):
        return "side_effect"
    if t.get("final_answer") is None:
        return "no_answer"
    try:
        env = _replay(t, suite)
    except (ValueError, IndexError, TypeError, KeyError):
        return "other"
    if env.family != "multi":
        return "other"
    gold_c = sorted((o["order_id"], "customer_request") for op, _, o in env.jobs if op == "cancel")
    gold_s = sorted((o["order_id"], "express" if u["tier"] == "gold" else "standard") for op, u, o in env.jobs
                    if op == "ship")
    if getattr(env, "failed", False) or env.side_effects:
        return "strict_stop" if getattr(env, "failed", False) else "side_effect"
    if sorted(env.effects.get("cancelled") or []) != gold_c or sorted(env.effects.get("shipped") or []) != gold_s:
        return "premature_finish"
    if f"{env.multi_total:.2f}" not in str(t.get("final_answer")).replace(",", ""):
        return "wrong_total"
    return "asked" if res.get("n_asks") else "other"


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else NAN


def tool_metrics(trajs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage and accuracy of the decision model at the tool decision points (cascades), and the reflection
    trigger's firing rate. Coverage = share of decisions executed from the model's answer (not deferred)."""
    recs = [r for r in M.decision_records(list(trajs)) if not r.get("shadow")]
    out: dict[str, Any] = {}
    pooled = [r for r in recs if r.get("dp") in TOOL_DPS and r.get("mode") == "cascade"]
    for name, rs in (*((dp, [r for r in pooled if r["dp"] == dp]) for dp in TOOL_DPS), ("tool", pooled)):
        dec = [r for r in rs if not r.get("deferred")]
        allc = [M.is_correct(r, r.get("s1_value")) for r in rs]
        acc_all = [c for c in allc if c is not None]
        accc = [c for c in (M.is_correct(r, r.get("s1_value")) for r in dec) if c is not None]
        defr = [r for r in rs if r.get("deferred")]
        llmc = [bool(r["s2_correct"]) for r in defr if r.get("s2_correct") is not None]    # graded by the harness
        out[name] = {"n": len(rs), "coverage": len(dec) / len(rs) if rs else NAN, "n_gold": len(acc_all),
                     "acc": _mean([float(c) for c in acc_all]), "acc_accepted": _mean([float(c) for c in accc]),
                     "llm_acc_deferred": _mean([float(c) for c in llmc]), "n_accepted": len(dec)}
    # the cancel-vs-ship decisions (the tier rule's operation, new to every decision model): tool choice at the
    # states whose gold next tool is a write
    ops = [r for r in pooled if r["dp"] == "tool_select" and r.get("gold") in OP_TOOLS]
    opd = [r for r in ops if not r.get("deferred")]
    opc = [c for c in (M.is_correct(r, r.get("s1_value")) for r in ops) if c is not None]
    out["op"] = {"n": len(ops), "coverage": len(opd) / len(ops) if ops else NAN,
                 "acc": _mean([float(c) for c in opc]),
                 "acc_accepted": _mean([float(c) for c in (M.is_correct(r, r.get("s1_value")) for r in opd) if c is not None]),
                 "accepted_wrong": sum(M.is_correct(r, r.get("s1_value")) is False for r in opd),
                 "llm_acc_deferred": _mean([float(bool(r["s2_correct"])) for r in ops
                                            if r.get("deferred") and r.get("s2_correct") is not None]),
                 "s1_said_other_op": sum(str(r.get("s1_value")) in OP_TOOLS and r.get("s1_value") != r.get("gold")
                                         for r in ops)}
    rf = [r for r in recs if r.get("dp") == "reflection" and r.get("mode") == "trigger"]
    gold_rf = [r for r in rf if isinstance(r.get("gold"), bool) and isinstance(r.get("s1_value"), bool)]
    out["reflection"] = {"n": len(rf), "fired": _mean([float(r.get("s1_value") is True) for r in rf]),
                         "acc": _mean([float(r["s1_value"] == r["gold"]) for r in gold_rf]),
                         "gold_positive_rate": _mean([float(r["gold"]) for r in gold_rf])}
    return out


def per_episode(trajs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wall = [float(t.get("wall_ms") or 0) / 1000 for t in trajs]
    return {"wall_s_mean": _mean(wall), "wall_s_median": float(statistics.median(wall)) if wall else NAN,
            "steps": _mean([len(t.get("steps") or []) for t in trajs]),
            "llm_calls": _mean([C.s_calls(t, "s2") for t in trajs]), "s1_calls": _mean([C.s_calls(t, "s1") for t in trajs]),
            "s1_cost": _mean([float(((t.get("usage") or {}).get("s1") or {}).get("cost_usd") or 0) for t in trajs]),
            "s2_cost": _mean([float(((t.get("usage") or {}).get("s2") or {}).get("cost_usd") or 0) for t in trajs]),
            "s1_s": _mean([float(((t.get("usage") or {}).get("s1") or {}).get("latency_ms") or 0) / 1000 for t in trajs])}


_AMOUNT = re.compile(r"\d[\d,]*\.\d{2}")
_BARE = re.compile(r"\s*\$?\s*[\d,]+\.\d{2}\s*\.?\s*")
ANSWER_FORMS = ("itemized", "sentence", "number", "none")


def answer_form(answer: Any) -> str:
    """The form of a multi task's final answer (the registered E7e readout, Deviations "E7e: itemized final answer"
    (e)): ``number`` (the amount alone), ``itemized`` (three or more amounts: per-customer lines plus the total),
    ``sentence`` (prose giving one or two amounts, e.g. only the total), ``none`` (no final answer). On E7c and E7e an
    answer holds either 1 or at least 13 amounts, so the threshold of three is not delicate."""
    if answer is None or not str(answer).strip():
        return "none"
    a = str(answer)
    if _BARE.fullmatch(a):
        return "number"
    return "itemized" if len(_AMOUNT.findall(a)) >= 3 else "sentence"


def answer_forms(trajs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per form: answers, share of the arm's episodes and success among them."""
    out: dict[str, Any] = {}
    n = len(trajs)
    for f in ANSWER_FORMS:
        ts = [t for t in trajs if answer_form(t.get("final_answer")) == f]
        out[f] = {"n": len(ts), "share": len(ts) / n if n else NAN,
                  "n_success": sum(bool(C.success(t)) for t in ts), "success": _mean([float(C.success(t)) for t in ts])}
    return out


def tariff(runs: Path = RUNS, window: Path | Sequence[Path] = BOX_DIR / "eval_window.json", exp: str = "e7",
           names: Sequence[str] | None = None) -> dict[str, Any]:
    """S2-4's provisioned tariff convention (A7.13 (iii)): q = host $/h x serving window / completed JevAgentRL
    episodes. The window is the evaluation's (pass 1 start to the last pass's end), during which the GPU served only
    the JevAgentRL arm (about a third of the pool's episodes in flight). Several windows (the parts of a pooled row,
    one host type) add up, and so do the parts' episodes (``names``: the parts' run names; default the eval run)."""
    s24 = json.loads(S24_TARIFF.read_text())["q_usd_per_episode"] if S24_TARIFF.exists() else NAN
    windows = [Path(window)] if isinstance(window, (str, Path)) else [Path(w) for w in window]
    if not windows or not all(w.exists() for w in windows):
        return {"q_usd_per_episode": NAN, "state": "no window record", "sensitivity": {"s2_4": s24, "zero": 0.0}}
    ws = [json.loads(w.read_text()) for w in windows]
    types = {str(w.get("instance_type") or "") for w in ws}
    it = types.pop() if len(types) == 1 else None
    if it not in HOST_USD_PER_HOUR:
        raise SystemExit(f"no on-demand price recorded for instance type(s) {sorted(types | {it} - {None})!r}")
    span = lambda w: (dt.datetime.fromisoformat(str(w["window_end"]).replace("Z", "+00:00"))  # noqa: E731
                      - dt.datetime.fromisoformat(str(w["window_start"]).replace("Z", "+00:00"))).total_seconds()
    hours = sum(span(w) for w in ws) / 3600
    done = 0
    for name in (names or [EXPS[exp]["eval"]]):
        for arm in EXPS[exp]["gpu_arms"]:
            p = arm_file(runs / name, arm)
            if p.exists():
                done += sum(1 for t in C.load_latest([p])[0].values() if not t.get("error"))
    q = HOST_USD_PER_HOUR[it] * hours / done if done and hours > 0 else NAN
    win = ({"start": ws[0]["window_start"], "end": ws[0]["window_end"]} if len(ws) == 1 else
           [{"start": w["window_start"], "end": w["window_end"]} for w in ws])
    return {"q_usd_per_episode": q, "host_usd_per_hour": HOST_USD_PER_HOUR[it], "instance_type": it,
            "serving_hours": hours, "episodes": done, "window": win,
            "workers": ws[0].get("workers"), "state": "ok",
            "convention": "S2-4 provisioned tariff (A7.13 (iii)): host $/h x evaluation window / completed JevAgentRL "
                          "episodes; on-demand price, us-east-1",
            "sensitivity": {"s2_4": s24, "zero": 0.0}}


def _saving(ca: np.ndarray, cb: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ca), size=(n_boot, len(ca)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = 1.0 - ca[idx].mean(1) / cb[idx].mean(1)
    boot = boot[np.isfinite(boot)]
    return {"cost": float(ca.mean()), "saving": float(1.0 - ca.mean() / cb.mean()) if cb.mean() > 0 else NAN,
            "saving_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot) else [NAN, NAN]}


def _load_arm_parts(paths: Sequence[Path]) -> dict[str, Any] | None:
    """``module_map.load_arm`` over one or more run files of the same arm (the parts of a pooled row)."""
    got = [a for a in (MM.load_arm(p) for p in paths) if a is not None]
    if not got:
        return None
    if len(got) == 1:
        return got[0]
    eps: dict[str, Any] = {}
    for a in got:
        eps.update(a["episodes"])
    return {"path": "; ".join(a["path"] for a in got), "sha256": "; ".join(a["sha256"] for a in got),
            "n_records": sum(a["n_records"] for a in got), "episodes": eps}


def summarize(runs: Path = RUNS, n_boot: int = MM.N_BOOT, seed: int = MM.SEED,
              window: Path | Sequence[Path] | None = None, exp: str = "e7", part: str | None = None,
              light: bool = False) -> dict[str, Any]:
    """The registered readout. A row with ``eval_parts`` (E7f and its extension) pools the parts' runs and adds each
    part's own comparison (``parts``); ``part`` = one part's run name; ``light`` leaves out the latency breakdown and
    the diagnosis view."""
    X = EXPS[exp]
    parts = X["eval_parts"]
    names = [part] if part else ([p[1] for p in parts if (runs / p[1]).is_dir()] or [parts[0][1]] if parts
                                 else [X["eval"]])
    run = runs / names[0]
    if window is None:
        boxes = {p[1]: p[2] for p in parts}
        window = [Path(boxes.get(n, X["box"])) / "eval_window.json" for n in names]
    exp_name = exp
    exp = yaml.safe_load((run / "experiment.yaml").read_text()) if (run / "experiment.yaml").exists() else {}
    suite = exp.get("suite") or {}
    paths = {arm: [arm_file(runs / n, arm) for n in names if arm_file(runs / n, arm).exists()]
             for arm in X["arms"]}
    if X.get("ref_runs"):       # the reference arm is another row's LOGGED run (E19: E7f's llm_only), never re-run
        paths[REF] = [arm_file(runs / n, REF) for n in X["ref_runs"] if arm_file(runs / n, REF).exists()]
    latest = {arm: C.load_latest(paths[arm])[0] if paths[arm] else {} for arm in X["arms"]}
    own = {t for arm in X["arms"] if arm != REF for t in latest.get(arm) or {}}
    ref = _load_arm_parts(paths[REF])
    if X.get("ref_runs") and own:   # the reference's own numbers (success, wall, failures) on the paired tasks only
        latest[REF] = {t: r for t, r in latest[REF].items() if t in own}
        if ref is not None:
            ref = {**ref, "episodes": {t: e for t, e in ref["episodes"].items() if t in own}}
    tf = tariff(runs, window, X["prefix"], names)
    arms: dict[str, Any] = {}
    for arm in X["arms"]:
        a = _load_arm_parts(paths[arm])
        if a is None:
            arms[arm] = {"state": "not run"}
            continue
        trajs_all = [t for t in latest[arm].values() if not t.get("error")]
        row: dict[str, Any] = {"state": "run", "n_ok": len(trajs_all), "n_records": a["n_records"],
                               "success_all": _mean([float(C.success(t)) for t in trajs_all]),
                               "cost_mean": _mean([C.cost(t) for t in trajs_all]),
                               "per_episode": per_episode(trajs_all),
                               "failure_types": _failures(trajs_all, suite),
                               "answer_forms": answer_forms(trajs_all),
                               "provenance": {k: a[k] for k in ("path", "sha256", "n_records")}}
        if arm != REF and ref is not None:
            expected = list(dict.fromkeys(list(ref["episodes"]) + list(a["episodes"])))
            cmp = MM.compare(ref, a, expected, n_boot, seed)
            keys = cmp.pop("keys")
            if keys:
                ya = [float(a["episodes"][t]["result"]["success"]) for t in keys]
                yr = [float(ref["episodes"][t]["result"]["success"]) for t in keys]
                cmp["newcombe_pt"] = C.newcombe(ya, yr)["ci_pt"]
                ca = np.array([C.cost(latest[arm][t]) for t in keys])
                cb = np.array([C.cost(latest[REF][t]) for t in keys])
                cmp["api"] = _saving(ca, cb, n_boot, seed)
                if arm in X["gpu_arms"]:
                    q = tf["q_usd_per_episode"]
                    cmp["tariff_q"] = q
                    cmp["sens"] = {k: _saving(ca + v, cb, n_boot, seed) for k, v in tf["sensitivity"].items()}
                    if MM._finite(q):
                        inc = _saving(ca + q, cb, n_boot, seed)
                        cmp.update({"cost": inc["cost"], "saving": inc["saving"], "saving_ci": inc["saving_ci"]})
                    else:
                        cmp.update({"cost": NAN, "saving": NAN, "saving_ci": [NAN, NAN]})
                pk = [latest[arm][t] for t in keys]
                row["paired_per_episode"] = per_episode(pk)
                row["ref_paired_per_episode"] = per_episode([latest[REF][t] for t in keys])
                row["points"] = tool_metrics(pk)
            row["cmp"] = cmp
            # further logged arms the row pairs with on the same tasks (E19: E7f's rl_tier); API cost only, since the
            # two GPU tariffs belong to different hosts
            for label, (xarm, xnames) in (X.get("extra_refs") or {}).items():
                xp = [arm_file(runs / n, xarm) for n in xnames if arm_file(runs / n, xarm).exists()]
                xref = _load_arm_parts(xp)
                if xref is None:
                    continue
                xref = {**xref, "episodes": {t: e for t, e in xref["episodes"].items() if t in a["episodes"]}}
                xlatest = C.load_latest(xp)[0]
                cx = MM.compare(xref, a, list(dict.fromkeys(list(xref["episodes"]) + list(a["episodes"]))), n_boot, seed)
                xkeys = cx.pop("keys")
                if xkeys:
                    ya = [float(a["episodes"][t]["result"]["success"]) for t in xkeys]
                    yr = [float(xref["episodes"][t]["result"]["success"]) for t in xkeys]
                    cx["newcombe_pt"] = C.newcombe(ya, yr)["ci_pt"]
                    cx["ref_wall_s_mean"] = per_episode([xlatest[t] for t in xkeys])["wall_s_mean"]
                cx.update({"ref_arm": xarm, "ref_runs": list(xnames)})
                row.setdefault("cmp_extra", {})[label] = cx
        arms[arm] = row
    meta = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": C.git_head(),
            "script": "scripts/explore_e7_hard.py", "run": "; ".join(C.rel(runs / n) for n in names),
            "label": X.get("label", LABEL),
            "n_boot": n_boot, "seed": seed, "suite": suite,
            "max_steps": (exp.get("harness_defaults") or {}).get("max_steps"), "llm": exp.get("llm"),
            "workers": exp.get("workers"),
            "spend": ({**run_spend(runs / part), "prior": 0.0} if part else total_spend(runs, exp=X["prefix"])),
            "cap": X.get("cap_total", X["cap"]),
            "exp": X["prefix"], "title": X["title"], "entry": X["entry"], "arms": list(X["arms"]),
            "calibration": _calibration(X["prefix"]), "not_run": X.get("not_run") or {}, "note": X.get("note"),
            "pilot_spend": (json.loads((OUT_DIR / "pilot.json").read_text()).get("spend_total")      # E7's own cap only
                            if X["prefix"] == "e7" and (OUT_DIR / "pilot.json").exists() else None),
            "projection": (json.loads((X["box"] / "projection.json").read_text())
                           if (X["box"] / "projection.json").exists() else None),
            "stopped_by_watchdog": (runs / X["stop"]).exists() or (X["box"] / X["stop"]).exists(),
            # tasks the evaluation ran (the gate may run the first n_min of the config's n: --max-tasks)
            "n_run": len(own or {t for arm in X["arms"] for t in latest.get(arm) or {}}),
            "ref_runs": list(X.get("ref_runs") or ()),
            "pilot_chosen": _pilot_chosen(suite, (exp.get("harness_defaults") or {}).get("max_steps"))}
    if light:
        return {"meta": meta, "tariff": tf, "arms": arms}
    lat = {arm: latency_breakdown([t for t in latest[arm].values() if not t.get("error")])
           for arm in X["arms"] if latest.get(arm)}
    diag = {arm: diagnose([runs / n for n in names], arm) for arm in X["diagnose"] if latest.get(arm)}
    out = {"meta": meta, "tariff": tf, "arms": arms, "latency": lat, "latency_cmp": latency_compare(lat),
           **({"diagnosis": diag} if diag else {})}
    if parts and not part:      # each part with its own window (the list above is aligned with the parts)
        wl = (list(window) if isinstance(window, (list, tuple)) else [window]) if len(names) == (
            len(window) if isinstance(window, (list, tuple)) else 1) else [None] * len(names)
        label_of = {p[1]: p[0] for p in parts}
        out["parts"] = {label_of[n]: summarize(runs, n_boot, seed, window=w, exp=exp_name, part=n, light=True)
                        for n, w in zip(names, wl)}
    return out


def _pilot_chosen(suite: Mapping[str, Any], max_steps: Any) -> dict[str, Any] | None:
    """The development pilot's LLM-only success for the evaluated setting (every pilot candidate with the same knobs,
    pooled), beside the evaluation's: the target band was 60-80 %."""
    if not (OUT_DIR / "pilot.json").exists():
        return None
    keys = ("families", "multi_customers", "multi_rule", "n_distractors", "noise_lines", "distractor_style", "strict")
    rows = [r for r in json.loads((OUT_DIR / "pilot.json").read_text())["candidates"]
            if all(r["suite"].get(k) == suite.get(k) for k in keys) and r.get("max_steps") == max_steps]
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    ok = sum(r["success"] * r["n"] for r in rows)
    return {"candidates": [r["candidate"] for r in rows], "n": n, "success": ok / n if n else NAN, "band": [0.6, 0.8]}


def _failures(trajs: Sequence[Mapping[str, Any]], suite: Mapping[str, Any]) -> dict[str, Any]:
    """Failed episodes by type and by class (wrong final total / wrong state-changing call / ask-or-finish error)."""
    types: dict[str, int] = {}
    for t in trajs:
        if not C.success(t):
            k = failure_type(t, suite)
            types[k] = types.get(k, 0) + 1
    classes = {c: 0 for c in ("wrong_total", "wrong_write", "ask_finish", "other")}
    for k, n in types.items():
        classes[FAILURE_CLASS[k]] += n
    return {"n_failed": sum(types.values()), "types": types, "classes": classes}


# ============================================================================== diagnosis (E7f (1), offline, $0)
AMOUNT_BAND = 150.0        # amount bands of the cancel-rate table: [0, 150), [150, 300), ... (the tier limits are 300 / 600)
OTHER_PREFIX = "<OTHER"


def _arm_thresholds(run: Path, arm: str) -> dict[str, dict[str, Any]]:
    """The decision points the arm ran with: the resolved profile (``profiles.json``, written by the runner) for a
    profile arm, else the arm's own decision points in ``experiment.yaml``."""
    pf = run / "profiles.json"
    if pf.exists():
        p = json.loads(pf.read_text()).get(arm)
        if p:
            return {dp: dict(v) for dp, v in (p.get("decision_points") or {}).items()}
    exp = yaml.safe_load((run / "experiment.yaml").read_text())
    a = next((a for a in exp.get("arms") or [] if a.get("name") == arm), {})
    return {dp: dict(a[dp]) for dp in ("tool_select", "tool_params", "reflection") if isinstance(a.get(dp), Mapping)}


def _write_job(env: Any, state: str) -> tuple[Any, ...] | None:
    """The (operation, customer, order) of the write decision whose state this is: the order named in the state's
    CURRENT OBSERVATION (the get_orders result the write follows), or None."""
    cur = str(state).split("CURRENT OBSERVATION:")[-1]
    by_oid = {o["order_id"]: (op, u, o) for op, u, o in env.jobs}
    for oid in re.findall(r'"order_id":\s*"([^"]+)"', cur):
        if oid in by_oid:
            return by_oid[oid]
    return None


def diagnose(run: Path | Sequence[Path], arm: str) -> dict[str, Any]:
    """Is a decision-model arm's failure the model or the pipeline? From the run's own records ($0, no model call):
    (1) the thresholds it ran with, and whether every cascade decision was accepted / deferred by them (a tool choice is
    accepted iff confidence >= tau; an argument iff its answer is not OTHER and confidence >= tau); (2) the System-1
    endpoint and model every episode and every System-1 call names; (3) what the model saw at the rule's write states
    (gold next tool cancel / ship): both writes among the options, the policy sentence in the task, the customer's tier
    and the order amount in the state, the state length against the harness cap; (4) the model's cancel rate by tier
    and amount band, and its errors by cell; (5) every failed episode's failure type with the decision that made the
    episode's last step (the model's accepted tool choice or the LLM) and, for a wrong write, whether the tool or an
    argument was wrong; for an early finish, how many writes were done."""
    from jevharness.bench.suites import jevgym_hard as H

    runs_ = [Path(run)] if isinstance(run, (str, Path)) else [Path(r) for r in run]   # one run, or a row's parts
    run = runs_[0]
    exp = yaml.safe_load((run / "experiment.yaml").read_text())
    suite = exp.get("suite") or {}
    dps = _arm_thresholds(run, arm)
    latest = C.load_latest([r / f"{arm}.jsonl" for r in runs_ if (r / f"{arm}.jsonl").exists()])[0]
    trajs = [t for t in latest.values() if not t.get("error")]
    thr = {dp: {"n": 0, "consistent": 0, "other_deferred": 0, "violations": 0, "examples": []}
           for dp in ("tool_select", "tool_params")}
    describe: dict[str, int] = {}
    s1_models: dict[str, int] = {}
    cap = None
    see = {"n": 0, "both_writes_offered": 0, "policy_in_task": 0, "tier_and_amount_seen": 0, "job_found": 0,
           "state_chars_max": 0, "state_at_cap": 0}
    cells: dict[str, dict[str, int]] = {}
    bands: dict[str, dict[str, int]] = {}
    fails: dict[str, int] = {}
    attrib: dict[str, int] = {}
    finish_after: dict[str, int] = {}
    for t in trajs:
        d0 = json.dumps(((t.get("meta") or {}).get("s1") or {}).get("describe"), sort_keys=True)
        describe[d0] = describe.get(d0, 0) + 1
        cap = cap or ((t.get("meta") or {}).get("config") or {}).get("s1_state_max_chars")
        for c in t.get("calls") or []:
            if c.get("component") == "s1":
                k = f"{c.get('backend')} / {c.get('model')}"
                s1_models[k] = s1_models.get(k, 0) + 1
        env = _env_of(str(t["task_id"]), suite)
        env.reset()
        policy = env.policy_text() if hasattr(env, "policy_text") else ""
        for st in t.get("steps") or []:
            for d in st.get("decisions") or []:
                dp = d.get("dp")
                if d.get("mode") == "cascade" and dp in thr and d.get("s1_conf") is not None and not d.get("shadow"):
                    tau = float((dps.get(dp) or {}).get("tau", NAN))
                    other = dp == "tool_params" and str(d.get("s1_value")).startswith(OTHER_PREFIX)
                    want = other or not (float(d["s1_conf"]) >= tau)
                    r = thr[dp]
                    r["n"] += 1
                    if bool(d.get("deferred")) == want:
                        r["consistent"] += 1
                        r["other_deferred"] += int(other)
                    else:
                        r["violations"] += 1
                        if len(r["examples"]) < 5:
                            r["examples"].append({"task_id": t["task_id"], "step": st.get("step"), "conf": d["s1_conf"],
                                                  "deferred": d.get("deferred"), "value": str(d.get("s1_value"))[:40]})
                if not (dp == "tool_select" and d.get("mode") == "cascade" and d.get("gold") in OP_TOOLS):
                    continue
                s = str(d.get("state") or "")
                see["n"] += 1
                see["both_writes_offered"] += all(x in (d.get("options") or []) for x in OP_TOOLS)
                see["policy_in_task"] += bool(policy) and policy in s
                see["state_chars_max"] = max(see["state_chars_max"], len(s))
                see["state_at_cap"] += bool(cap) and len(s) >= int(cap)
                job = _write_job(env, s)
                if job is None:
                    continue
                _, u, o = job
                see["job_found"] += 1
                see["tier_and_amount_seen"] += (f'"tier": "{u["tier"]}"' in s and u["user_id"] in s
                                                and f'"amount": {o["amount"]}' in s)
                lim = H.MULTI_TIER_CANCEL_ABOVE.get(u["tier"])          # gold tier: never cancelled
                side = "gold tier" if lim is None else ("above limit" if float(o["amount"]) > lim else "at or below limit")
                ck = f"{u['tier']} | {side} | gold {d['gold']}"
                cell = cells.setdefault(ck, {"n": 0, "model_wrong": 0, "accepted_wrong": 0})
                wrong = d.get("s1_value") != d["gold"]
                cell["n"] += 1
                cell["model_wrong"] += int(wrong)
                cell["accepted_wrong"] += int(wrong and not d.get("deferred"))
                lo = int(float(o["amount"]) // AMOUNT_BAND * AMOUNT_BAND)
                bk = f"{u['tier']} | {lo}-{lo + int(AMOUNT_BAND)}"
                b = bands.setdefault(bk, {"n": 0, "model_cancel": 0, "gold_cancel": 0})
                b["n"] += 1
                b["model_cancel"] += d.get("s1_value") == "cancel_order"
                b["gold_cancel"] += d["gold"] == "cancel_order"
        if C.success(t):
            continue
        ft = failure_type(t, suite)
        fails[ft] = fails.get(ft, 0) + 1
        last = (t.get("steps") or [{}])[-1]
        ts = next((d for d in last.get("decisions") or [] if d.get("dp") == "tool_select"
                   and d.get("mode") in ("cascade", "replace")), None)
        who = "model" if ts is not None and not ts.get("deferred") else "LLM"
        if FAILURE_CLASS.get(ft) == "wrong_write":
            wrong_tool = ts is not None and ts.get("gold") is not None and ts.get("final_value") != ts.get("gold")
            what = f"wrong tool {last.get('tool')} (gold {ts.get('gold') if ts else '?'})" if wrong_tool else \
                f"{last.get('tool')} with a wrong argument"
            key = f"{ft}: {what}, tool chosen by the {who}"
        elif ft == "premature_finish":
            key = f"{ft}: {last.get('tool')} chosen by the {who}"
            n_w = sum(1 for s in t.get("steps") or [] if s.get("tool") in OP_TOOLS)
            finish_after[str(n_w)] = finish_after.get(str(n_w), 0) + 1
        else:
            key = f"{ft}: last step {last.get('tool')} chosen by the {who}"
        attrib[key] = attrib.get(key, 0) + 1
    rel = lambda r: C.rel(r) if str(Path(r).resolve()).startswith(str(ROOT)) else f"{Path(r).name} (a copy outside the checkout)"  # noqa: E731
    return {"run": "; ".join(rel(r) for r in runs_), "arm": arm,
            "arm_sha256": "; ".join(C.sha256_file(r / f"{arm}.jsonl") for r in runs_ if (r / f"{arm}.jsonl").exists()),
            "n_episodes": len(trajs), "n_records": len(latest),
            "thresholds": {dp: {k: v for k, v in c.items()} for dp, c in dps.items()}, "threshold_check": thr,
            "s1_describe": describe, "s1_calls": s1_models, "s1_state_max_chars": cap, "write_states": see,
            "cells": dict(sorted(cells.items())), "cancel_rate": dict(sorted(bands.items())),
            "failures": dict(sorted(fails.items(), key=lambda kv: -kv[1])),
            "attribution": dict(sorted(attrib.items(), key=lambda kv: -kv[1])),
            "premature_finish_after_writes": dict(sorted(finish_after.items(), key=lambda kv: int(kv[0])))}



def diagnosis_markdown(d: Mapping[str, Any], title: str | None = None) -> str:
    th, see = d["threshold_check"], d["write_states"]
    out = [f"## {title or 'Diagnosis'}: `{d['arm']}` in `{d['run']}`", "",
           f"- episodes: {d['n_episodes']} complete of {d['n_records']} tasks",
           "- thresholds ran with: " + "; ".join(f"{dp} {v.get('mode')} tau {v.get('tau')}"
                                                 + (" (always defer)" if (v.get("always_defer") or (v.get('extra') or {})
                                                                          .get("always_defer")) else "")
                                                 for dp, v in d["thresholds"].items())]
    for dp, r in th.items():
        out.append(f"- {dp}: {r['n']} cascade decisions; accepted / deferred as the threshold says: {r['consistent']}"
                   f" (of which {r['other_deferred']} deferred because the answer was OTHER); violations {r['violations']}")
    out.append("- System-1 endpoint per episode: " + "; ".join(f"{k} x {v}" for k, v in d["s1_describe"].items()))
    out.append("- System-1 calls by backend / model: " + "; ".join(f"{k} x {v}" for k, v in d["s1_calls"].items()))
    out += [f"- write states (gold next tool cancel_order / ship_order): {see['n']}; both writes offered "
            f"{see['both_writes_offered']}; the policy sentence in the task {see['policy_in_task']}; the order identified "
            f"from the current observation {see['job_found']}, and of those the customer's tier and the amount both in the "
            f"state {see['tier_and_amount_seen']}; longest state {see['state_chars_max']} characters (harness cap "
            f"{d['s1_state_max_chars']}; states at the cap {see['state_at_cap']})", "",
            "| tier \\| amount vs. the tier's limit \\| gold | n | model wrong | accepted wrong |", "|---|---|---|---|"]
    for k, c in d["cells"].items():
        out.append(f"| {k} | {c['n']} | {c['model_wrong']} ({_pt(c['model_wrong'] / c['n'], False)} %) | "
                   f"{c['accepted_wrong']} |")
    out += ["", f"Cancel rate by tier and amount band (${AMOUNT_BAND:.0f} bands; the rule cancels basic above $300, "
            "silver above $600, never gold):", "", "| tier \\| amount band ($) | n | model says cancel % | rule cancels % |",
            "|---|---|---|---|"]
    for k, b in d["cancel_rate"].items():
        out.append(f"| {k} | {b['n']} | {_pt(b['model_cancel'] / b['n'], False)} | {_pt(b['gold_cancel'] / b['n'], False)} |")
    out += ["", "Failed episodes by type, with the decision that made the last step:", ""]
    out += [f"- {v} x {k}" for k, v in d["attribution"].items()]
    if d["premature_finish_after_writes"]:
        out.append("- early finishes, by the number of writes done before the finish: "
                   + ", ".join(f"{k} writes: {v}" for k, v in d["premature_finish_after_writes"].items()))
    return "\n".join(out) + "\n"


def cmd_diagnose(a: argparse.Namespace) -> int:
    d = diagnose(Path(a.run), a.arm)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(MM._clean(d), indent=1, default=MM._json_default) + "\n")
    print(diagnosis_markdown(d))
    return 0


# ---------------------------------------------------------------- writers
def _pt(x: Any, sign: bool = True, nd: int = 1) -> str:
    if not MM._finite(x):
        return "--"
    return f"{100 * x:+.{nd}f}" if sign else f"{100 * x:.{nd}f}"


def _fail_text(f: Mapping[str, Any] | None) -> str:
    if not f:
        return "--"
    c = f["classes"]
    return (f"{f['n_failed']} failed: wrong final total {c['wrong_total']}, wrong state-changing call {c['wrong_write']}, "
            f"ask / finish error {c['ask_finish']}, other {c['other']} (types {f['types']})")


def to_markdown(res: Mapping[str, Any]) -> str:
    m, tf = res["meta"], res["tariff"]
    s = m["suite"]
    sp = m["spend"]
    out = [f"# {m.get('title', 'E7: JevGym-Hard')} ({m.get('label', LABEL)}, descriptive)", "",
           f"- generated {m['generated']} at git `{(m['git'] or '?')[:12]}` by `{m['script']}`; run `{m['run']}`; "
           f"paired task bootstrap B = {m['n_boot']}, seed {m['seed']}",
           f"- setting: families {s.get('families')}, {s.get('multi_customers')} customers per task, rule "
           f"`{s.get('multi_rule')}`, {s.get('n_distractors')} {s.get('distractor_style')} distractors, "
           f"{s.get('noise_lines')} noise lines, strict {s.get('strict')}, max steps {m['max_steps']}"
           + (f", answer style `{s['answer_style']}`" if s.get("answer_style") else "") + "; seeds "
           f"{s.get('seed0')}-{int(s.get('seed0', 0)) + int(m.get('n_run') or s.get('n', 0)) - 1}"
           + (f" (the first {m['n_run']} of the config's {s.get('n')}, as the gate decided)"
              if m.get("n_run") and int(m["n_run"]) < int(s.get("n") or 0) else "")
           + (" (pooled parts: " + ", ".join(f"{k} {v['meta'].get('n_run')} tasks"
                                               for k, v in (res.get("parts") or {}).items()) + ")"
              if res.get("parts") else "")
           + f"; System-2 `{(m['llm'] or {}).get('model')}` "
           f"via {(m['llm'] or {}).get('provider')}; {m['workers']} workers, task-major",
           f"- spend: evaluation + smoke ${sp['total']:.2f} (System-2 ${sp['s2']:.2f}, Jev ${sp['s1']:.3f}); "
           + (f"pilot ${m['pilot_spend']:.2f}" if m.get("pilot_spend") is not None else "the pilot counts under E7")
           + f"; cap ${m['cap']:.0f}" + ("; **stopped by the watchdog**" if m["stopped_by_watchdog"] else ""),
           f"- GPU tariff q = ${tf.get('q_usd_per_episode', NAN):.5f} per JevAgentRL episode ({tf.get('instance_type')}, "
           f"${tf.get('host_usd_per_hour', NAN)}/h × {tf.get('serving_hours', NAN):.3f} h / {tf.get('episodes')} episodes; "
           f"sensitivity: S2-4's q ${tf['sensitivity']['s2_4']:.5f}, q = 0)" if tf.get("state") == "ok"
           else f"- GPU tariff: {tf.get('state')}",
           f"- registration: `docs/PREREG_A7_STAGE2.md`, Deviations \"{m.get('entry', 'Exploratory row E7 (JevGym-Hard)')}\". "
           "No test; outside every Holm table.", ""]
    if m.get("note"):
        out.insert(-1, f"- **route:** {m['note']}")
    for arm, why in (m.get("not_run") or {}).items():
        out.insert(-1, f"- `{arm}`: not run ({why})")
    r0 = res["arms"].get(REF) or {}
    pc = m.get("pilot_chosen")
    if pc and r0.get("state") == "run":
        inside = pc["band"][0] <= r0["success_all"] <= pc["band"][1]
        out.append(f"- tuning target: Flash alone 60-80 % on the evaluated setting; the development pilot gave "
                   f"{_pt(pc['success'], False)} % (n = {pc['n']}, {', '.join(pc['candidates'])}); the evaluation gives "
                   f"{_pt(r0['success_all'], False)} % (n = {r0['n_ok']}), "
                   + ("inside the band" if inside else "outside the band"))
    if r0.get("state") == "run":
        pe = r0["per_episode"]
        out.append(f"- `llm_only`: n = {r0['n_ok']}, success {_pt(r0['success_all'], False)} %, "
                   f"${1000 * r0['cost_mean']:.1f} per 1k episodes, wall {pe['wall_s_mean']:.1f} s mean / "
                   f"{pe['wall_s_median']:.1f} s median; failures {_fail_text(r0['failure_types'])}")
    out += ["", "| arm | n | success % | Δ pt [95 % CI] | Newcombe | wins / losses | $/1k ep. (ref) | saving % [95 % CI] | "
            "tool coverage % | acc. % (accepted) | reflection fired % | wall mean / median s (ref) |", "|" + "---|" * 12]
    for arm in res["meta"].get("arms", ARMS)[1:]:
        r = res["arms"].get(arm) or {}
        if r.get("state") != "run" or "points" not in r:
            out.append(f"| `{arm}` | -- | | | | | | | | | | not run |")
            continue
        k, p, pe, pr = r["cmp"], r["points"], r["paired_per_episode"], r["ref_paired_per_episode"]
        ci, sci, nc = k.get("delta_ci") or [NAN, NAN], k.get("saving_ci") or [NAN, NAN], k.get("newcombe_pt") or [NAN, NAN]
        out.append(f"| `{arm}` | {k['n']} | {_pt(k['success'], False)} | {_pt(k['delta'])} [{_pt(ci[0])}, {_pt(ci[1])}] | "
                   f"[{nc[0]:+.1f}, {nc[1]:+.1f}] | {k['wins']} / {k['losses']} | {1000 * k['cost']:.1f} "
                   f"({1000 * k['cost_ref']:.1f}) | {_pt(k['saving'], False)} [{_pt(sci[0], False)}, {_pt(sci[1], False)}] | "
                   f"{_pt(p['tool']['coverage'], False)} | {_pt(p['tool']['acc'], False)} ({_pt(p['tool']['acc_accepted'], False)}) | "
                   f"{_pt(p['reflection']['fired'], False)} | {pe['wall_s_mean']:.1f} / {pe['wall_s_median']:.1f} "
                   f"({pr['wall_s_mean']:.1f} / {pr['wall_s_median']:.1f}) |")
    route = ("System-2 via OpenRouter at its reported prices" if (m.get("llm") or {}).get("provider") == "openrouter"
             else f"System-2 via {(m.get('llm') or {}).get('provider')} at the list prices of configs/prices/models.yaml")
    out += ["", f"Cost: API as logged ({route}; Jev at its list price); the "
            "JevAgentRL arm adds the GPU tariff q per episode (its API-only saving and the sensitivity rows are below). "
            "Coverage: share of tool decisions (tool choice and arguments) executed from the decision model's answer. "
            "Acc.: the model's answers against the gold label on the run's own states (all of its tool answers; in "
            "brackets the accepted ones).", "", "## Per arm", ""]
    for arm in res["meta"].get("arms", ARMS)[1:]:
        r = res["arms"].get(arm) or {}
        if r.get("state") != "run" or "points" not in r:
            continue
        k, p = r["cmp"], r["points"]
        out.append(f"### `{arm}`")
        out.append("")
        out.append(f"- paired n = {k['n']} ({k['n_dropped']} dropped: {k['dropped']}); success {_pt(k['success'], False)} % vs "
                   f"{_pt(k['success_ref'], False)} %")
        out.append(f"- failures (all complete episodes): {_fail_text(r['failure_types'])}; `llm_only`: "
                   f"{_fail_text((res['arms'].get(REF) or {}).get('failure_types'))}")
        api = k["api"]
        out.append(f"- API cost ${1000 * api['cost']:.1f} per 1k episodes, saving {_pt(api['saving'], False)} % "
                   f"[{_pt(api['saving_ci'][0], False)}, {_pt(api['saving_ci'][1], False)}]")
        for name, v in (k.get("sens") or {}).items():
            out.append(f"- sensitivity q = {name}: ${1000 * v['cost']:.1f} per 1k, saving {_pt(v['saving'], False)} % "
                       f"[{_pt(v['saving_ci'][0], False)}, {_pt(v['saving_ci'][1], False)}]")
        for dp in (*TOOL_DPS, "tool"):
            d = p[dp]
            out.append(f"- {dp}: {d['n']} decisions, coverage {_pt(d['coverage'], False)} %, model acc. "
                       f"{_pt(d['acc'], False)} % (n gold {d['n_gold']}), accepted {_pt(d['acc_accepted'], False)} %, LLM on "
                       f"deferred {_pt(d['llm_acc_deferred'], False)} %")
        op = p["op"]
        out.append(f"- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): "
                   f"{op['n']}, coverage {_pt(op['coverage'], False)} %, model acc. {_pt(op['acc'], False)} %, accepted "
                   f"{_pt(op['acc_accepted'], False)} % ({op['accepted_wrong']} accepted wrong), LLM on deferred "
                   f"{_pt(op['llm_acc_deferred'], False)} %; the model named the other write {op['s1_said_other_op']} times")
        rf = p["reflection"]
        out.append(f"- reflection trigger: {rf['n']} questions, fired {_pt(rf['fired'], False)} %, acc. {_pt(rf['acc'], False)} % "
                   f"(gold positive {_pt(rf['gold_positive_rate'], False)} %)")
        pe = r["paired_per_episode"]
        out.append(f"- per episode: {pe['steps']:.1f} steps, {pe['llm_calls']:.1f} LLM calls, {pe['s1_calls']:.1f} System-1 "
                   f"calls ({pe['s1_s']:.1f} s), System-2 ${1000 * pe['s2_cost']:.1f} / System-1 ${1000 * pe['s1_cost']:.2f} per 1k")
        for label, cx in (r.get("cmp_extra") or {}).items():
            if not cx.get("n"):
                continue
            ci = cx.get("delta_ci") or [NAN, NAN]
            out.append(f"- vs `{cx['ref_arm']}` ({label}: the logged arm of `{'; '.join(cx['ref_runs'])}` on the same tasks): "
                       f"paired n = {cx['n']}, success {_pt(cx['success'], False)} % vs {_pt(cx['success_ref'], False)} %, "
                       f"Δ {_pt(cx['delta'])} [{_pt(ci[0])}, {_pt(ci[1])}], wins / losses {cx['wins']} / {cx['losses']}; "
                       f"API cost ${1000 * cx['cost']:.1f} vs ${1000 * cx['cost_ref']:.1f} per 1k (saving "
                       f"{_pt(cx['saving'], False)} %, GPU tariffs of two hosts not added); wall {pe['wall_s_mean']:.1f} vs "
                       f"{cx.get('ref_wall_s_mean', NAN):.1f} s")
        out.append("")
    forms = {arm: (res["arms"].get(arm) or {}).get("answer_forms") for arm in res["meta"].get("arms", ARMS)}
    if any(forms.values()):
        out += ["## Final-answer form", "", "Form of each complete episode's final answer (`answer_form`): the amount "
                "alone, prose with one or two amounts (e.g. only the total), or itemized (per-customer lines plus the "
                "total). Success is the arm's success among those answers.", "",
                "| arm | " + " | ".join(f"{f}: n (share %), success %" for f in ANSWER_FORMS) + " |",
                "|---|" + "---|" * len(ANSWER_FORMS)]
        for arm, fm in forms.items():
            if fm:
                out.append(f"| `{arm}` | " + " | ".join(
                    f"{fm[f]['n']} ({_pt(fm[f]['share'], False)}), {_pt(fm[f]['success'], False)}" for f in ANSWER_FORMS) + " |")
        out.append("")
    if m.get("calibration"):
        out += ["", calibration_markdown(m["calibration"]).replace("# E7c calibration fit", "## Calibration fit (E7c)", 1)]
    if res.get("latency"):
        out += ["", latency_markdown(res["latency"], res.get("latency_cmp") or {})]
    for arm, d in (res.get("diagnosis") or {}).items():
        out += ["", diagnosis_markdown(d, "Diagnosis view (as results/explore/e7f/DIAGNOSIS.md; descriptive)")]
    if res.get("parts"):
        out += ["", "## By part (each part's own paired comparison; the table above pools them)", "",
                "| part | run | seeds | arm | n | LLM-only success % | arm success % | Δ pt [95 % CI] | wins / losses | "
                "saving % [95 % CI] | wall mean s (ref) | spend $ |", "|" + "---|" * 12]
        for label, pr in res["parts"].items():
            pm, ps = pr["meta"], pr["meta"]["suite"]
            seeds = f"{ps.get('seed0')}-{int(ps.get('seed0', 0)) + int(pm.get('n_run') or 0) - 1}"
            for arm in pm.get("arms", ARMS)[1:]:
                r = pr["arms"].get(arm) or {}
                if r.get("state") != "run" or "cmp" not in r:
                    continue
                k = r["cmp"]
                ci, sci = k.get("delta_ci") or [NAN, NAN], k.get("saving_ci") or [NAN, NAN]
                pe, pr_ = r["paired_per_episode"], r["ref_paired_per_episode"]
                out.append(f"| {label} | `{pm['run']}` | {seeds} | `{arm}` | {k['n']} | {_pt(k['success_ref'], False)} | "
                           f"{_pt(k['success'], False)} | {_pt(k['delta'])} [{_pt(ci[0])}, {_pt(ci[1])}] | "
                           f"{k['wins']} / {k['losses']} | {_pt(k['saving'], False)} [{_pt(sci[0], False)}, "
                           f"{_pt(sci[1], False)}] | {pe['wall_s_mean']:.1f} ({pr_['wall_s_mean']:.1f}) | "
                           f"{pm['spend']['total']:.2f} |")
    return "\n".join(out)


def to_tex_keys(res: Mapping[str, Any]) -> dict[str, str]:
    v: dict[str, str] = {}
    m, s = res["meta"], res["meta"]["suite"]
    P = m.get("exp", "e7")
    for k in ("multi_customers", "n_distractors", "noise_lines", "seed0", "n", "answer_style"):
        if s.get(k) is not None:
            v[f"{P}/setting/{k}"] = str(s[k])
    if m.get("n_run") and int(m["n_run"]) < int(s.get("n") or 0):     # the gate ran the first n_run tasks only
        v[f"{P}/setting/n_config"] = str(s["n"])
    if m.get("n_run") and int(m["n_run"]) != int(s.get("n") or 0):    # fewer (gate) or more (pooled parts) than the config
        v[f"{P}/setting/n"] = str(m["n_run"])
    for label, pr in (res.get("parts") or {}).items():           # each part's own comparison: <P>/<part>/...
        pm = pr["meta"]
        v[f"{P}/{label}/n"] = str(pm.get("n_run"))
        v[f"{P}/{label}/seed0"] = str(pm["suite"].get("seed0"))
        v[f"{P}/{label}/spend"] = f"{pm['spend']['total']:.2f}"
        r0 = pr["arms"].get(REF) or {}
        if r0.get("state") == "run":
            v[f"{P}/{label}/llm_only/success"] = f"{100 * r0['success_all']:.1f}"
            v[f"{P}/{label}/llm_only/wall_mean"] = f"{r0['per_episode']['wall_s_mean']:.1f}"
        for arm in pm.get("arms", ARMS)[1:]:
            r = pr["arms"].get(arm) or {}
            if r.get("state") == "run" and "cmp" in r:
                v.update(MM._num_keys(f"{P}/{label}/{arm}", r["cmp"], None))
                v[f"{P}/{label}/{arm}/wall_mean"] = f"{r['paired_per_episode']['wall_s_mean']:.1f}"
                v[f"{P}/{label}/{arm}/wins"], v[f"{P}/{label}/{arm}/losses"] = (str(int(r["cmp"]["wins"])),
                                                                                str(int(r["cmp"]["losses"])))
    v[f"{P}/setting/rule"] = str(s.get("multi_rule", "given"))
    v[f"{P}/setting/families"] = str(s.get("families"))
    if m.get("max_steps") is not None:
        v[f"{P}/setting/max_steps"] = str(m["max_steps"])
    r0 = res["arms"].get(REF) or {}
    if r0.get("state") == "run":
        v[f"{P}/llm_only/success"] = f"{100 * r0['success_all']:.1f}"
        v[f"{P}/llm_only/n"] = str(r0["n_ok"])
        v[f"{P}/llm_only/wall_mean"] = f"{r0['per_episode']['wall_s_mean']:.1f}"
        v[f"{P}/llm_only/wall_median"] = f"{r0['per_episode']['wall_s_median']:.1f}"
        v[f"{P}/llm_only/cost_k"] = f"{1000 * r0['cost_mean']:.1f}"
    for arm in m.get("not_run") or {}:
        v[f"{P}/{arm}/state"] = "not run"
    if m.get("note"):
        v[f"{P}/route"] = str((m.get("llm") or {}).get("provider"))
    for arm in m.get("arms", ARMS):
        for f, d in ((res["arms"].get(arm) or {}).get("answer_forms") or {}).items():
            v[f"{P}/{arm}/form/{f}/n"] = str(d["n"])
            if MM._finite(d["share"]):
                v[f"{P}/{arm}/form/{f}/share"] = f"{100 * d['share']:.1f}"
            if MM._finite(d["success"]):
                v[f"{P}/{arm}/form/{f}/success"] = f"{100 * d['success']:.1f}"
        f = (res["arms"].get(arm) or {}).get("failure_types")
        if f:
            v[f"{P}/{arm}/fail/n"] = str(f["n_failed"])
            for c, n in f["classes"].items():
                v[f"{P}/{arm}/fail/{c.replace('_', '')}"] = str(n)
    for arm in res["meta"].get("arms", ARMS)[1:]:
        r = res["arms"].get(arm) or {}
        p = f"{P}/{arm}"
        v[f"{p}/state"] = r.get("state", "not run")
        if r.get("state") != "run" or "points" not in r:
            continue
        k, pts = r["cmp"], r["points"]
        v.update(MM._num_keys(p, k, None))
        for kk in ("wins", "losses"):
            v[f"{p}/{kk}"] = str(int(k[kk]))
        nc = k.get("newcombe_pt") or [NAN, NAN]
        if MM._finite(nc[0]):
            v[f"{p}/newcombe/lo"], v[f"{p}/newcombe/hi"] = f"{nc[0]:+.1f}", f"{nc[1]:+.1f}"
        v[f"{p}/saving_api"] = f"{100 * k['api']['saving']:.1f}"
        v[f"{p}/cost_api_k"] = f"{1000 * k['api']['cost']:.1f}"
        for dp in (*TOOL_DPS, "tool"):
            d = pts[dp]
            for kk in ("coverage", "acc", "acc_accepted"):
                if MM._finite(d.get(kk)):
                    key = f"{p}/{kk}" if dp == "tool" else f"{p}/{dp}/{kk}"
                    v[key] = f"{100 * d[kk]:.1f}"
        for kk in ("acc", "coverage", "acc_accepted"):
            if MM._finite(pts["op"].get(kk)):
                v[f"{p}/op_{kk}"] = f"{100 * pts['op'][kk]:.1f}"
        v[f"{p}/op_n"] = str(pts["op"]["n"])
        v[f"{p}/op_accepted_wrong"] = str(pts["op"]["accepted_wrong"])
        if MM._finite(pts["reflection"].get("fired")):
            v[f"{p}/reflection_fired"] = f"{100 * pts['reflection']['fired']:.1f}"
        pe, pr = r["paired_per_episode"], r["ref_paired_per_episode"]
        v[f"{p}/wall_mean"], v[f"{p}/wall_median"] = f"{pe['wall_s_mean']:.1f}", f"{pe['wall_s_median']:.1f}"
        v[f"{p}/wall_ref_mean"], v[f"{p}/wall_ref_median"] = f"{pr['wall_s_mean']:.1f}", f"{pr['wall_s_median']:.1f}"
        for name, sv in (k.get("sens") or {}).items():
            if MM._finite(sv.get("saving")):
                v[f"{p}/saving_q_{name.replace('_', '')}"] = f"{100 * sv['saving']:.1f}"
        for label, cx in (r.get("cmp_extra") or {}).items():      # E19: the paired comparison with E7f's rl_tier
            if not cx.get("n"):
                continue
            q = f"{p}/{label}"
            v.update(MM._num_keys(q, cx, None))
            for kk in ("wins", "losses"):
                v[f"{q}/{kk}"] = str(int(cx[kk]))
            nc = cx.get("newcombe_pt") or [NAN, NAN]
            if MM._finite(nc[0]):
                v[f"{q}/newcombe/lo"], v[f"{q}/newcombe/hi"] = f"{nc[0]:+.1f}", f"{nc[1]:+.1f}"
            if MM._finite(cx.get("ref_wall_s_mean")):
                v[f"{q}/wall_ref_mean"] = f"{cx['ref_wall_s_mean']:.1f}"
    for arm, f in ((m.get("calibration") or {}).get("fits") or {}).items():
        for dp in TOOL_DPS:
            r = f.get(dp) if isinstance(f, Mapping) else None
            if not r:
                continue
            k = f"{P}/cal/{f.get('eval_arm', arm).replace('_', '')}/{dp.replace('_', '')}"
            v[f"{k}/tau"] = "always defer" if r["always_defer"] else f"{r['tau_hat']:.4g}"
            if MM._finite((r.get("at_tau") or {}).get("coverage")):
                v[f"{k}/coverage"] = f"{100 * r['at_tau']['coverage']:.1f}"
    for arm, a in (res.get("latency") or {}).items():
        k = f"{P}/lat/{arm.replace('_', '')}"
        v[f"{k}/wall"] = f"{a['wall_mean']:.1f}"
        v[f"{k}/s1"] = f"{a['s1_per_episode']:.1f}"
        v[f"{k}/s2"] = f"{a['s2_per_episode']:.1f}"
        v[f"{k}/s2tail"] = f"{a['s2_tail_per_episode']:.1f}"
        for g, r in a["groups"].items():
            if g.startswith("s2/"):
                gk = g.replace("_", "").replace(" ", "").replace("/", "-")
                v[f"{k}/{gk}/n"] = str(r["n"])
                v[f"{k}/{gk}/p50"] = f"{r['p50']:.2f}"
                v[f"{k}/{gk}/p90"] = f"{r['p90']:.2f}"
                v[f"{k}/{gk}/over"] = str(r["n_over"])
                v[f"{k}/{gk}/tailshare"] = f"{100 * r['tail_share_of_wall']:.1f}"
    for arm, c in (res.get("latency_cmp") or {}).items():
        k = f"{P}/lat/{arm.replace('_', '')}"
        v[f"{k}/extra"] = f"{c['extra_wall_s']:.1f}"
        v[f"{k}/extratail"] = f"{c['extra_s2_tail_s']:.1f}"
        if MM._finite(c.get("tail_share_of_extra")):
            v[f"{k}/tailshareofextra"] = f"{100 * c['tail_share_of_extra']:.0f}"
    for arm, d in (res.get("diagnosis") or {}).items():
        k = f"{P}/{arm}/diag"
        by_model = sum(n for key, n in d["attribution"].items() if "chosen by the model" in key)
        v[f"{k}/failsbymodel"] = str(by_model)
        v[f"{k}/wrongwritebymodel"] = str(sum(n for key, n in d["attribution"].items()
                                              if key.startswith(("strict_stop", "side_effect")) and "chosen by the model" in key))
        v[f"{k}/earlyfinishbymodel"] = str(sum(n for key, n in d["attribution"].items()
                                               if key.startswith("premature_finish") and "chosen by the model" in key))
    pc = m.get("pilot_chosen")
    if pc:
        v[f"{P}/pilot/success"] = f"{100 * pc['success']:.1f}"
        v[f"{P}/pilot/n"] = str(pc["n"])
    tf = res["tariff"]
    if MM._finite(tf.get("q_usd_per_episode")):
        v[f"{P}/tariff/q"] = f"{tf['q_usd_per_episode']:.5f}"
        v[f"{P}/tariff/host_usd_h"] = f"{tf['host_usd_per_hour']:.3f}"
    sp = m["spend"]
    v[f"{P}/spend"] = f"{sp['total'] + (m['pilot_spend'] or 0):.2f}"
    if m.get("pilot_spend") is not None:
        v[f"{P}/spend_pilot"] = f"{m['pilot_spend']:.2f}"
    v[f"{P}/cap"] = f"{m['cap']:.0f}"
    return v


def write_tex(vals: Mapping[str, str], path: Path = TEX, title: str = "Exploratory row E7, JevGym-Hard") -> Path:
    lines = ["% AUTO-GENERATED by scripts/explore_e7_hard.py -- do not edit by hand.",
             f"% {title} (descriptive; docs/PREREG_A7_STAGE2.md Deviations 2026-10-06), outside both stages.",
             f"% last written {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git "
             f"{(C.git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x.replace('_', chr(92) + '_') if '/state' in k or '/setting/' in k else x}}}"
              for k, x in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


def cmd_summarize(a: argparse.Namespace) -> int:
    X = EXPS[a.exp]
    res = summarize(Path(a.runs), a.n_boot, a.seed, Path(a.window) if a.window else None, exp=a.exp)
    out = Path(X["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{X['summary']}.json").write_text(json.dumps(MM._clean(res), indent=1, default=MM._json_default) + "\n")
    (out / f"{X['summary']}.md").write_text(to_markdown(res) + "\n")
    write_tex(to_tex_keys(res), Path(a.tex) if a.tex else X["tex"], X["title"])
    print(to_markdown(res))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--exp", default="e7", choices=sorted(EXPS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    sub.add_parser("fit")
    sp = sub.add_parser("spend")
    sp.add_argument("--prior", type=float, default=0.0)
    sp.add_argument("--pilots", action="store_true")
    sp.add_argument("--short", action="store_true")
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--cap", type=float, default=None, help="default: the experiment's cap")
    w.add_argument("--prior", type=float, default=0.0)
    w.add_argument("--every", type=float, default=30.0)
    pj = sub.add_parser("project")
    pj.add_argument("--prior", type=float, default=0.0)
    pj.add_argument("--out", default=None, help="default: <the experiment's box dir>/projection.json")
    s = sub.add_parser("summarize")
    s.add_argument("--n-boot", type=int, default=MM.N_BOOT)
    s.add_argument("--seed", type=int, default=MM.SEED)
    s.add_argument("--window", default=None)
    s.add_argument("--tex", default=None)
    dg = sub.add_parser("diagnose", help="offline: thresholds, endpoint, prompts and failures of one arm of a run")
    dg.add_argument("--run", required=True, help="a run directory (experiment.yaml + <arm>.jsonl)")
    dg.add_argument("--arm", required=True)
    dg.add_argument("--out", default=None, help="JSON record")
    a = ap.parse_args(argv)
    return {"pilot": cmd_pilot, "fit": cmd_fit, "spend": cmd_spend, "watch": cmd_watch, "project": cmd_project,
            "summarize": cmd_summarize, "diagnose": cmd_diagnose}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
