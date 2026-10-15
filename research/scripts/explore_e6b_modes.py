#!/usr/bin/env python3
r"""E6b (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry
"Exploratory row E6b" of 2026-10-06): the nine module-map cells added by E6b (harness/config.py E6B_CELLS; the
classification of all 40 cells: results/explore/e6b_modes/cells.md), each in its own arm beside the LLM-only reference
on fresh JevGym seeds 300800-300999. Agent LLM Gemini 3.5 Flash (minimal reasoning, temperature 0); decision model Jev
via OpenRouter. No test; outside every Holm table; nothing here revises a registered result.

    python scripts/explore_e6b_modes.py spend                       # logged $ of every E6b run (smoke + eval)
    python scripts/explore_e6b_modes.py watch --pid PID [--cap 150]  # watchdog: stops PID's group at the cap
    python scripts/explore_e6b_modes.py project                     # smoke -> eval cost projection (the registered gate)
    python scripts/explore_e6b_modes.py summarize                   # results/explore/e6b_modes/summary.{json,md} + e6b.tex

Readout (fixed in the Deviations entry before any episode): per cell, paired complete case over the evaluation tasks
(a task whose latest record errored, or that is missing, in either arm leaves that pair); success and Delta vs the
cell's reference in points (``llm_only``; routing: ``small_only``, Flash-Lite alone, as Stage 1's C4, with the
comparison to Flash beside it) with the paired task-bootstrap 95 % interval of scripts/module_map.py (B = 10,000, seed
0) and the Newcombe interval; paired wins and losses; cost per episode and saving 1 - C_arm / C_ref (task bootstrap);
coverage and activity; the decision model's accuracy against the gold label; the descriptive verdict of
scripts/module_map.py's fixed rule. Coverage = the share of the point's decisions executed from the decision model's
answer: cascade -- answered at confidence >= tau; trigger -- every gate decision is the model's (100 %); assist and
guard -- 0 by definition (the LLM's answer is executed). Activity (``acted``, what the verdict rule needs; as
scripts/module_map.py): cascade = coverage; assist = 100 % (the LLM chooses from the shortlist at every decision; the
share of shortlists that excluded an option is reported as ``binding``); guard = proposals vetoed; trigger = questions
on which the model fired."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402
import module_map as MM  # noqa: E402

from jevharness.harness.config import DP, EQUIVALENT_NOTES, Mode  # noqa: E402
from jevharness.harness.tools import OTHER  # noqa: E402

from jevharness.bench import metrics as M  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "results/explore/e6b_modes"
TEX = ROOT / "paper/generated/e6b.tex"
RUN_SMOKE, RUN_EVAL = "e6b_modes_smoke", "e6b_modes"
CAP_USD, SAFETY, N_EVAL, N_MIN = 150.0, 1.5, 200, 100
REF, ROUTE_REF = "llm_only", "small_only"
LABEL = "exploratory (E6b)"
# arm -> (decision point, mode): one E6b cell per arm (configs/explore/e6b_modes/eval.yaml)
CELLS: dict[str, tuple[str, str]] = {
    "tp_assist": ("tool_params", "assist"), "tp_guard": ("tool_params", "guard"),
    "mw_cascade": ("memory_write", "cascade"), "mw_guard": ("memory_write", "guard"),
    "mr_cascade": ("memory_read", "cascade"), "mr_trigger": ("memory_read", "trigger"),
    "rf_cascade": ("reflection", "cascade"), "st_cascade": ("stop", "cascade"), "rt_cascade": ("routing", "cascade"),
}
ARMS = (REF, ROUTE_REF, *CELLS)
NAN = float("nan")
PREFIX = "e6b"
EXTRA_SPEND_RUNS: tuple[str, ...] = ()
CELL_RUNS: dict[str, str] = {}       # arm -> run directory, when a cell (and its reference) ran in another run
INVALID_RUNS: dict[str, str] = {}    # arm -> run directory of records excluded as infrastructure-invalid
# E6d (Deviations "Exploratory row E6d", results/explore/e6d_modes/cells.md) reuses this readout: scripts/
# explore_e6d_modes.py calls configure("e6d") before main()
ROWS: dict[str, dict[str, Any]] = {
    "e6b": {"RUN_SMOKE": "e6b_modes_smoke", "RUN_EVAL": "e6b_modes", "OUT_DIR": ROOT / "results/explore/e6b_modes",
            "TEX": ROOT / "paper/generated/e6b.tex", "CAP_USD": 150.0, "LABEL": "exploratory (E6b)", "CELLS": dict(CELLS)},
    "e6d": {"RUN_SMOKE": "e6d_modes_smoke_or", "RUN_EVAL": "e6d_modes",
            "EXTRA_SPEND_RUNS": ("e6d_modes_smoke", "e6d_modes_routing_smoke", "e6d_modes_routing"),
            # the routing repair (Deviations addendum ~16:55 UTC): the routing cells and their reference come from
            # their own run; the first OpenRouter-route routing records are reported apart as infrastructure-invalid
            "CELL_RUNS": {"rt_assist": "e6d_modes_routing", "rt_guard": "e6d_modes_routing"},
            "INVALID_RUNS": {"rt_assist": "e6d_modes", "rt_guard": "e6d_modes"}, "OUT_DIR": ROOT / "results/explore/e6d_modes",
            "TEX": ROOT / "paper/generated/e6d.tex", "CAP_USD": 120.0, "LABEL": "exploratory (E6d)",
            "CELLS": {"mw_assist": ("memory_write", "assist"), "rf_assist": ("reflection", "assist"),
                      "st_assist": ("stop", "assist"), "rt_assist": ("routing", "assist"),
                      "rf_guard": ("reflection", "guard"), "rt_guard": ("routing", "guard"),
                      "mr_guard": ("memory_read", "guard")}},
    # E6e (Deviations "Exploratory row E6e", results/explore/e6e_modes/cells.md): the five REPLACE cells, each run as
    # its own arm although it behaves as another cell at a fixed threshold (harness.config.EQUIVALENT_NOTES)
    "e6e": {"RUN_SMOKE": "e6e_modes_smoke", "RUN_EVAL": "e6e_modes", "OUT_DIR": ROOT / "results/explore/e6e_modes",
            "TEX": ROOT / "paper/generated/e6e.tex", "CAP_USD": 60.0, "LABEL": "exploratory (E6e)",
            "CELLS": {"mw_replace": ("memory_write", "replace"), "mr_replace": ("memory_read", "replace"),
                      "rf_replace": ("reflection", "replace"), "st_replace": ("stop", "replace"),
                      "rt_replace": ("routing", "replace")}},
}


def configure(row: str) -> None:
    """Point every function of this module at one exploratory row (``e6b``, the default, ``e6d`` or ``e6e``)."""
    global RUN_SMOKE, RUN_EVAL, OUT_DIR, TEX, CAP_USD, LABEL, CELLS, ARMS, PREFIX, EXTRA_SPEND_RUNS, CELL_RUNS
    global INVALID_RUNS
    r = ROWS[row]
    EXTRA_SPEND_RUNS = tuple(r.get("EXTRA_SPEND_RUNS", ()))
    CELL_RUNS, INVALID_RUNS = dict(r.get("CELL_RUNS", {})), dict(r.get("INVALID_RUNS", {}))
    RUN_SMOKE, RUN_EVAL, OUT_DIR, TEX, CAP_USD, LABEL = (r["RUN_SMOKE"], r["RUN_EVAL"], r["OUT_DIR"], r["TEX"],
                                                         r["CAP_USD"], r["LABEL"])
    CELLS = dict(r["CELLS"])
    ARMS = (REF, ROUTE_REF, *CELLS)
    PREFIX = row


# ============================================================================== spend, watchdog, projection
def episode_spend(t: Mapping[str, Any]) -> dict[str, float]:
    """$ of one logged episode (measurement calls included, if any): Gemini (System-2) and Jev (System-1)."""
    u = t.get("usage") or {}
    s1 = float((u.get("s1") or {}).get("cost_usd") or 0.0)
    s2 = float((u.get("s2") or {}).get("cost_usd") or 0.0)
    tot = float((u.get("total") or {}).get("cost_usd") or 0.0)
    sh = float((u.get("shadow") or {}).get("cost_usd") or 0.0)
    return {"gemini": s2 + max(tot - s1 - s2, 0.0), "jev": s1 + sh, "total": tot + sh}


def run_spend(run_dir: Path) -> dict[str, Any]:
    """Every record of every arm (errored attempts and re-runs included: all of them were paid)."""
    out = {"gemini": 0.0, "jev": 0.0, "total": 0.0, "records": 0}
    for p in sorted(Path(run_dir).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        for line in p.read_text().splitlines():
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if not isinstance(t, dict) or "task_id" not in t:
                continue
            s = episode_spend(t)
            for k in ("gemini", "jev", "total"):
                out[k] += s[k]
            out["records"] += 1
    return out


def total_spend(runs: Path = RUNS) -> dict[str, Any]:
    names = dict.fromkeys((*EXTRA_SPEND_RUNS, RUN_SMOKE, RUN_EVAL))   # every paid run of the row counts to the cap
    per = {name: run_spend(runs / name) for name in names if (runs / name).is_dir()}
    tot = {k: sum(v[k] for v in per.values()) for k in ("gemini", "jev", "total", "records")}
    return {"by_run": per, **tot}


def cmd_spend(a: argparse.Namespace) -> int:
    print(json.dumps(total_spend(Path(a.runs)), indent=1))
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    """Poll the logged spend every ``--every`` s; at the cap send SIGTERM to the runner's process group (then SIGKILL)
    and write ``<runs>/e6b_STOP.json``. Episodes in flight when it fires are not logged (a few cents)."""
    stop = Path(a.runs) / f"{PREFIX}_STOP.json"
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(Path(a.runs))
        if s["total"] >= a.cap:
            stop.write_text(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                        "spend": s, "cap": a.cap}, indent=1))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(a.pid), sig)
                except OSError:
                    pass
                time.sleep(10)
            print(f"CAP REACHED: ${s['total']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} spend ${s['total']:.3f} (Gemini ${s['gemini']:.3f}, "
              f"Jev ${s['jev']:.4f}; {s['records']} records)", flush=True)
        time.sleep(a.every)


def projection(runs: Path = RUNS, n_eval: int = N_EVAL, safety: float = SAFETY, cap: float | None = None) -> dict[str, Any]:
    """The registered gate: the smoke's mean $ per episode per arm, summed over the arms, x the evaluation tasks x
    ``safety``, plus the spend so far, against the cap; if it does not fit, the first ``N_MIN`` tasks (same check);
    else stop after the smoke."""
    cap = CAP_USD if cap is None else cap
    sm = {arm: list(C.load_latest([runs / RUN_SMOKE / f"{arm}.jsonl"])[0].values())
          if (runs / RUN_SMOKE / f"{arm}.jsonl").exists() else [] for arm in ARMS}
    per = {arm: (float(np.mean([episode_spend(t)["total"] for t in ts])) if ts else None) for arm, ts in sm.items()}
    wall = {arm: (float(np.mean([float(t.get("wall_ms") or 0) for t in ts])) / 1000 if ts else None)
            for arm, ts in sm.items()}
    errors = {arm: sum(bool(t.get("error")) for t in ts) for arm, ts in sm.items()}
    so_far = total_spend(runs)["total"]
    ep = sum(v for v in per.values() if v is not None)
    out = {"smoke_per_episode": per, "smoke_wall_s": wall, "smoke_n": {k: len(v) for k, v in sm.items()},
           "smoke_errors": errors, "missing_arms": [a for a, v in per.items() if v is None],
           "spent_so_far": so_far, "safety": safety, "cap": cap, "per_task_all_arms": ep}
    for n in (n_eval, N_MIN):
        out[f"projected_{n}"] = ep * n * safety
        out[f"fits_{n}"] = so_far + ep * n * safety <= cap
    out["decision"] = (f"run all {n_eval}" if out[f"fits_{n_eval}"] else
                       f"run the first {N_MIN}" if out[f"fits_{N_MIN}"] else "stop after the smoke")
    if out["missing_arms"]:
        out["decision"] = "stop: smoke incomplete (" + ", ".join(out["missing_arms"]) + ")"
    return out


def cmd_project(a: argparse.Namespace) -> int:
    p = projection(Path(a.runs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "projection.json").write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(p, indent=1))
    return 0


# ============================================================================== readout
def _correct(v: Any, gold: Any) -> bool:
    """A decision model's (or the LLM's) answer against a gold label: membership for a label set (memory read), equality
    for a yes / no gold, case- and space-insensitive equality otherwise (argument values)."""
    if isinstance(gold, list):
        return v in gold
    if isinstance(gold, bool):
        return isinstance(v, bool) and v == gold
    return v is not None and str(v).strip().lower() == str(gold).strip().lower()


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else NAN


def cell_metrics(trajs: Sequence[Mapping[str, Any]], dp: str, mode: str) -> dict[str, Any]:
    """Coverage, activity and accuracy of the arm's E6b cell (module docstring)."""
    recs = [r for r in M.decision_records(list(trajs)) if r.get("dp") == dp and r.get("mode") == mode
            and not r.get("shadow")]
    gold = [r for r in recs if r.get("gold_status") == "ok" and r.get("gold") is not None]
    out: dict[str, Any] = {"n_decisions": len(recs), "n_gold": len(gold)}
    if mode == "cascade":
        dec = [r for r in recs if not r.get("deferred")]
        out["coverage"] = len(dec) / len(recs) if recs else NAN
        out["acted"], out["acted_kind"] = out["coverage"], "decided"
        g_dec = [r for r in gold if not r.get("deferred")]
        g_def = [r for r in gold if r.get("deferred") and r.get("s2_value") is not None]
        out["acc"] = _mean([_correct(r["s1_value"], r["gold"]) for r in g_dec])      # accepted answers
        out["s1_acc_all"] = _mean([_correct(r["s1_value"], r["gold"]) for r in gold])
        out["llm_acc_deferred"] = _mean([_correct(_s2_as_s1(dp, r["s2_value"]), r["gold"]) for r in g_def])
        out["n_deferred"] = len(recs) - len(dec)
        if dp == "stop":   # what the cascade did at finish proposals / other steps
            fin = [r for r in recs if r.get("s2_value") is True]
            out["finish_rejected"] = sum(r.get("final_value") is False for r in fin)
            out["early_stops"] = sum(r.get("s2_value") is False and r.get("final_value") is True for r in recs)
            out["n_finish_proposals"] = len(fin)
            # the LLM's own stop decision is free (its tool call), so a decided answer acts only where it differs
            out["overrode"] = (out["finish_rejected"] + out["early_stops"]) / len(recs) if recs else NAN
        if dp == "memory_read":
            out["mean_injected"] = _mean([len(r.get("final_value") or []) for r in recs])
    elif mode == "replace":     # E6e: System-1's argmax decides alone at every decision (the LLM is never asked)
        out["coverage"] = 1.0 if recs else NAN
        out["acted"], out["acted_kind"] = (1.0 if recs else NAN), "decided"
        if dp == "memory_read":     # its single top memory (or none): in the gold set? does the injection cover it?
            out["acc"] = _mean([_correct(r["s1_value"], r["gold"]) for r in gold])
            out["mean_injected"] = _mean([len(r.get("final_value") or []) for r in recs])
            out["final_covers"] = _mean([all(x in set(r.get("final_value") or []) for x in r["gold"]
                                             if x != M.NONE_MEMORY) for r in gold])
        else:
            out["acc"] = _mean([_correct(_yes(r["s1_value"]), r["gold"]) for r in gold])
            out["model_yes_rate"] = _mean([_yes(r.get("s1_value")) for r in recs])
            out["gold_positive_rate"] = _mean([bool(r["gold"]) for r in gold])
        if dp == "stop":            # as the stop cascade: rejected finishes and early stops change the episode
            fin = [r for r in recs if r.get("s2_value") is True]
            out["finish_rejected"] = sum(r.get("final_value") is False for r in fin)
            out["early_stops"] = sum(r.get("s2_value") is False and r.get("final_value") is True for r in recs)
            out["n_finish_proposals"] = len(fin)
            out["overrode"] = (out["finish_rejected"] + out["early_stops"]) / len(recs) if recs else NAN
    elif mode == "trigger":
        out["coverage"] = 1.0 if recs else NAN
        fired = [r for r in recs if r.get("s1_value") is True]
        out["acted"], out["acted_kind"] = (len(fired) / len(recs) if recs else NAN), "fired"
        out["acc"] = _mean([_correct(r["s1_value"], r["gold"]) for r in gold])
        out["gold_positive_rate"] = _mean([bool(r["gold"]) for r in gold])
    elif mode == "assist" and dp != "tool_params":   # E6d "advise": the LLM decides with the model's answer as a hint
        out["coverage"] = 0.0 if recs else NAN
        out["acted"], out["acted_kind"] = (1.0 if recs else NAN), "advised"
        out["followed"] = _mean([_yes(r.get("final_value")) == _yes(r.get("s1_value")) for r in recs])
        out["model_yes_rate"] = _mean([_yes(r.get("s1_value")) for r in recs])
        out["llm_yes_rate"] = _mean([_yes(r.get("final_value")) for r in recs])
        out["acc"] = _mean([_correct(_yes(r["s1_value"]), r["gold"]) for r in gold])
        out["llm_final_acc"] = _mean([_correct(_yes(r.get("final_value")), r["gold"]) for r in gold])
    elif mode == "assist":
        out["coverage"] = 0.0 if recs else NAN
        out["acted"], out["acted_kind"] = (1.0 if recs else NAN), "shortlist"
        out["binding"] = _mean([bool((r.get("gold_aux") or {}).get("binding")) for r in recs])
        out["recall_k"] = _mean([r["gold"] in ((r.get("gold_aux") or {}).get("offered") or []) for r in gold])
        out["acc"] = out["recall_k"]
        out["s1_acc_all"] = _mean([_correct(r["s1_value"], r["gold"]) for r in gold])
        out["llm_final_acc"] = _mean([_correct(r.get("final_value"), r["gold"]) for r in gold])
        out["in_shortlist"] = _mean([bool((r.get("gold_aux") or {}).get("in_shortlist")) for r in recs])
    elif mode == "guard":
        out["coverage"] = 0.0 if recs else NAN
        if dp == "tool_params":     # one proposal = one checked call (retry 0) of one step
            props: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
            for r in recs:
                if (r.get("retry") or 0) == 0:
                    props.setdefault((r.get("task_id"), r.get("step")), []).append(r)
            vetoed = [k for k, rs in props.items() if any((x.get("gold_aux") or {}).get("veto") for x in rs)]
            out["n_proposals"], out["n_vetoed"] = len(props), len(vetoed)
            out["acted"], out["acted_kind"] = (len(vetoed) / len(props) if props else NAN), "vetoed"
            out["acc"] = _mean([_correct(r["s1_value"], r["gold"]) for r in gold])      # its argument answers
            vrec = [r for r in recs if (r.get("gold_aux") or {}).get("veto")]
            graded = [r for r in vrec if r.get("s2_correct") is not None]
            out["n_veto_graded"] = len(graded)     # vetoes on the oracle's tool, where the LLM's raw value has a gold
            out["veto_llm_was_wrong"] = _mean([r["s2_correct"] is False for r in graded])
            out["veto_on_other"] = _mean([r.get("s2_value") == OTHER for r in vrec])   # LLM value off the candidates
            out["veto_tool_changed"] = _mean([(r.get("gold_aux") or {}).get("retry_tool") != _tool_of(r) for r in vrec])
        else:                       # memory write: a proposal = a write the LLM gate proposed
            asked = [r for r in recs if isinstance(r.get("s1_value"), bool)]
            out["n_proposals"], out["n_vetoed"] = len(asked), sum(r["s1_value"] is False for r in asked)
            out["acted"], out["acted_kind"] = (out["n_vetoed"] / len(asked) if asked else NAN), "vetoed"
            out["acc"] = _mean([_correct(r["s1_value"], r["gold"]) for r in asked
                                if r.get("gold_status") == "ok" and r.get("gold") is not None])
            out["llm_store_rate"] = _mean([_yes(r.get("s2_value")) for r in recs])   # the LLM's proposal rate
            out["veto_precision"] = _mean([r["gold"] is False for r in asked if r["s1_value"] is False
                                           and r.get("gold_status") == "ok" and r.get("gold") is not None])
            if dp == "memory_read":     # E6d: does the injected selection hold every needed memory?
                out["final_covers"] = _mean([bool((r.get("gold_aux") or {}).get("final_covers")) for r in recs
                                             if r.get("gold_status") == "ok"])
    return out


def _tool_of(r: Mapping[str, Any]) -> str | None:
    """The tool named in an argument question ("... of tool `name` take?")."""
    q = str(r.get("question") or "")
    i = q.find("of tool `")
    return q[i + 9:q.index("`", i + 9)] if i >= 0 else None


def _yes(v: Any) -> bool:
    """A yes / no meta value as a boolean (routing logs "large" / "small")."""
    return v is True or v == "large"


def _s2_as_s1(dp: str, v: Any) -> Any:
    """The LLM's deferral answer in the model's answer space (routing logs "large" / "small")."""
    if dp == "routing":
        return v == "large"
    return v


def per_episode(trajs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wall = [float(t.get("wall_ms") or 0) / 1000 for t in trajs]
    return {"wall_s_mean": _mean(wall), "wall_s_median": float(statistics.median(wall)) if wall else NAN,
            "steps": _mean([len(t.get("steps") or []) for t in trajs]),
            "llm_calls": _mean([C.s_calls(t, "s2") for t in trajs]), "s1_calls": _mean([C.s_calls(t, "s1") for t in trajs]),
            "s1_cost": _mean([float(((t.get("usage") or {}).get("s1") or {}).get("cost_usd") or 0) for t in trajs]),
            "s2_cost": _mean([float(((t.get("usage") or {}).get("s2") or {}).get("cost_usd") or 0) for t in trajs]),
            # episodes in which an LLM reply had neither text nor a tool call (the loop reads it as an empty finish)
            "empty_reply_episodes": sum(int((t.get("meta") or {}).get("empty_llm_responses") or 0) > 0 for t in trajs)}


def compare_arms(run: Path, arm: str, ref: str, n_boot: int, seed: int) -> dict[str, Any] | None:
    a, r = MM.load_arm(run / f"{arm}.jsonl"), MM.load_arm(run / f"{ref}.jsonl")
    if a is None or r is None:
        return None
    expected = [t for t in dict.fromkeys(list(r["episodes"]) + list(a["episodes"]))]
    cmp = MM.compare(r, a, expected, n_boot, seed)
    keys = cmp.pop("keys")
    if keys:
        ya = [float(a["episodes"][t]["result"]["success"]) for t in keys]
        yr = [float(r["episodes"][t]["result"]["success"]) for t in keys]
        cmp["newcombe_pt"] = C.newcombe(ya, yr)["ci_pt"]
    cmp["provenance"] = {k: a[k] for k in ("path", "sha256", "n_records")}
    cmp["ref_provenance"] = {k: r[k] for k in ("path", "sha256", "n_records")}
    return {"cmp": cmp, "keys": keys}


def summarize(runs: Path = RUNS, n_boot: int = MM.N_BOOT, seed: int = MM.SEED) -> dict[str, Any]:
    run = runs / RUN_EVAL
    latest = {arm: C.load_latest([run / f"{arm}.jsonl"])[0] if (run / f"{arm}.jsonl").exists() else {}
              for arm in ARMS}
    cells: dict[str, Any] = {}
    invalid: dict[str, Any] = {}
    for arm, (dp, mode) in CELLS.items():
        ref = ROUTE_REF if dp == "routing" else REF
        if arm in INVALID_RUNS and (runs / INVALID_RUNS[arm] / f"{arm}.jsonl").exists():
            bad = runs / INVALID_RUNS[arm]
            empt = {a: sum(int((t.get("meta") or {}).get("empty_llm_responses") or 0) > 0
                           for t in C.load_latest([bad / f"{a}.jsonl"])[0].values()) for a in (arm, ref)}
            invalid[arm] = {"run": C.rel(bad), "reference": ref, "empty_reply_episodes": empt,
                            "n": {a: len(C.load_latest([bad / f"{a}.jsonl"])[0]) for a in (arm, ref)}}
        crun = runs / CELL_RUNS.get(arm, RUN_EVAL)
        if crun != run:
            for a in (arm, ref):
                latest.setdefault(f"{crun.name}/{a}", C.load_latest([crun / f"{a}.jsonl"])[0]
                                  if (crun / f"{a}.jsonl").exists() else {})
        res = compare_arms(crun, arm, ref, n_boot, seed)
        if res is None:
            cells[arm] = {"dp": dp, "mode": mode, "state": "not run", "reference": ref}
            continue
        cmp, keys = res["cmp"], res["keys"]
        la = latest[f"{crun.name}/{arm}"] if crun != run else latest[arm]
        lr = latest[f"{crun.name}/{ref}"] if crun != run else latest[ref]
        trajs = [la[t] for t in keys if t in la]
        pm = cell_metrics(trajs, dp, mode)
        v = MM.verdict(cmp.get("delta", NAN), cmp.get("saving", NAN), pm.get("acted", NAN))
        eq = EQUIVALENT_NOTES.get((DP(dp), Mode(mode)))
        cell = {"dp": dp, "mode": mode, "state": "exploratory", "arm": arm, "reference": ref, "cmp": cmp,
                **({"equals": eq} if eq else {}),
                "points": pm, "verdict": v, "per_episode": per_episode(trajs),
                "ref_per_episode": per_episode([lr[t] for t in keys if t in lr]),
                "attempts": sum(1 for _ in open(crun / f"{arm}.jsonl")), "run": C.rel(crun)}
        if dp == "routing" and (crun / f"{REF}.jsonl").exists():
            vf = compare_arms(crun, arm, REF, n_boot, seed)
            if vf is not None:
                cell["vs_flash"] = vf["cmp"]
        cells[arm] = cell
    refs = {}
    for ref in (REF, ROUTE_REF):
        ts = [t for t in latest[ref].values() if not t.get("error")]
        refs[ref] = {"n_ok": len(ts), "n_records": len(latest[ref]),
                     "success": _mean([float(C.success(t)) for t in ts]), "cost": _mean([C.cost(t) for t in ts]),
                     **per_episode(ts)}
    meta = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": C.git_head(),
            "script": f"scripts/explore_{PREFIX}_modes.py", "run": C.rel(run), "label": LABEL, "n_boot": n_boot,
            "seed": seed, "spend": total_spend(runs), "cap": CAP_USD,
            "projection": json.loads((OUT_DIR / "projection.json").read_text())
            if (OUT_DIR / "projection.json").exists() else None,
            "stopped_by_watchdog": (runs / f"{PREFIX}_STOP.json").exists(),
            "rule": "costs success if delta <= -3 pt; no saving if saving <= 0; takes over safely if delta >= -1 pt, "
                    "saving > 0 and acted >= 5 %; else mixed (scripts/module_map.py, fixed 2026-10-05)"}
    for crun_name in sorted(set(CELL_RUNS.values())):
        crun = runs / crun_name
        for ref in (REF, ROUTE_REF):
            if (crun / f"{ref}.jsonl").exists():
                ts = [t for t in C.load_latest([crun / f"{ref}.jsonl"])[0].values() if not t.get("error")]
                refs[f"{ref} ({crun_name})"] = {"n_ok": len(ts), "n_records": len(ts),
                                                 "success": _mean([float(C.success(t)) for t in ts]),
                                                 "cost": _mean([C.cost(t) for t in ts]), **per_episode(ts)}
    return {"meta": meta, "cells": cells, "references": refs, "invalid": invalid}


# ---------------------------------------------------------------- writers
def _pt(x: Any, sign: bool = True, nd: int = 1) -> str:
    if not MM._finite(x):
        return "--"
    return f"{100 * x:+.{nd}f}" if sign else f"{100 * x:.{nd}f}"


RATE_KEYS = ("coverage", "acted", "acc", "s1_acc_all", "llm_acc_deferred", "binding", "recall_k", "llm_final_acc",
             "in_shortlist", "veto_llm_was_wrong", "veto_on_other", "veto_tool_changed", "llm_store_rate",
             "veto_precision", "gold_positive_rate", "overrode", "followed", "model_yes_rate", "llm_yes_rate",
             "final_covers")


def _fmt_point(k: str, v: Any) -> str:
    if k in RATE_KEYS:
        return f"{_pt(v, False)} %" if MM._finite(v) else "--"
    if isinstance(v, float):
        return f"{v:.2f}" if MM._finite(v) else "--"
    return str(v)


def to_markdown(res: Mapping[str, Any]) -> str:
    m = res["meta"]
    out = [f"# {PREFIX.upper()}: module-map cells ({LABEL})", "",
           f"- generated {m['generated']} at git `{(m['git'] or '?')[:12]}` by `{m['script']}`; run `{m['run']}`; "
           f"paired task bootstrap B = {m['n_boot']}, seed {m['seed']}",
           f"- spend (every {PREFIX.upper()} record, smoke included): ${m['spend']['total']:.2f} (Gemini ${m['spend']['gemini']:.2f}, "
           f"Jev ${m['spend']['jev']:.2f}) of the ${m['cap']:.0f} cap"
           + ("; **stopped by the watchdog**" if m["stopped_by_watchdog"] else ""),
           f"- verdict rule (descriptive): {m['rule']}",
           f"- cells, semantics and thresholds: `results/explore/{PREFIX}_modes/cells.md`; registration: "
           f"`docs/PREREG_A7_STAGE2.md`, Deviations \"Exploratory row {PREFIX.upper()}\". No test; outside every Holm "
           "table.", ""]
    for ref, r in res["references"].items():
        out.append(f"- reference `{ref}`: n = {r['n_ok']} complete, success {_pt(r['success'], False)} %, "
                   f"${1000 * r['cost']:.1f} per 1k episodes, {r['wall_s_mean']:.1f} s per episode")
    for arm, b in (res.get("invalid") or {}).items():
        out.append(f"- **excluded (infrastructure):** `{arm}` and its reference `{b['reference']}` in `{b['run']}`: "
                   f"episodes with an empty LLM reply {b['empty_reply_episodes']} of {b['n']}; the cell is read from "
                   f"its repair run instead (Deviations addendum)")
    out += ["", "| cell | arm | ref. | n | succ. % | Δ pt [95 % CI] | wins / losses | $/1k ep. | saving % [95 % CI] | "
            "coverage % | acted % | acc. % | verdict |", "|" + "---|" * 13]
    for arm, c in res["cells"].items():
        if c["state"] != "exploratory":
            out.append(f"| {c['dp']}/{c['mode']} | `{arm}` | {c['reference']} | -- | | | | | | | | | not run |")
            continue
        k, pm = c["cmp"], c["points"]
        ci, sci = k.get("delta_ci") or [NAN, NAN], k.get("saving_ci") or [NAN, NAN]
        out.append(f"| {c['dp']}/{c['mode']} | `{arm}` | {c['reference']} | {k.get('n')} | {_pt(k.get('success'), False)} | "
                   f"{_pt(k.get('delta'))} [{_pt(ci[0])}, {_pt(ci[1])}] | {k.get('wins')} / {k.get('losses')} | "
                   f"{1000 * k.get('cost', NAN):.1f} | {_pt(k.get('saving'), False)} [{_pt(sci[0], False)}, "
                   f"{_pt(sci[1], False)}] | {_pt(pm.get('coverage'), False)} | {_pt(pm.get('acted'), False)} "
                   f"({pm.get('acted_kind')}) | {_pt(pm.get('acc'), False)} | {c['verdict'] or '--'} |")
    out += ["", "Coverage: share of the point's decisions executed from the decision model's answer (cascade: answered "
            "at confidence ≥ τ; trigger: 100 %, every gate decision is the model's; assist and guard: 0 by definition). "
            "Acted: cascade = coverage; assist = 100 % (shortlist at every decision); guard = proposals vetoed; trigger = "
            "questions on which it fired. Acc.: cascade = accepted answers; trigger and guard = the model's answers; "
            "assist = gold value in the shortlist (recall@k). Routing is compared with Flash-Lite alone (as C4). "
            "Stop cascade: the LLM's own stop decision is free (its tool call), so a decided answer changes the episode "
            "only where it differs from it (`overrode`: rejected finishes and early stops). Argument guard: "
            "`veto_on_other` = vetoes of a value outside the model's candidate list (which it cannot choose), "
            "`veto_llm_was_wrong` = vetoed values that were wrong, among the vetoes on the oracle's tool. Replace (E6e): "
            "coverage and acted = 100 % (the model's argmax decides every question; the LLM is never asked); acc. = "
            "its answers (memory read: its top memory, or none, in the gold set).", "",
            "## Per cell", ""]
    for arm, c in res["cells"].items():
        if c["state"] != "exploratory":
            continue
        k, pm, pe, pr = c["cmp"], c["points"], c["per_episode"], c["ref_per_episode"]
        out.append(f"### {c['dp']} × {c['mode']} (`{arm}`)")
        out.append("")
        if c.get("equals"):
            out.append(f"- same rule as: {c['equals']}")
        out.append(f"- n = {k['n']} paired tasks ({k['n_dropped']} dropped: {k['dropped']}); success {_pt(k.get('success'), False)} % "
                   f"vs {_pt(k.get('success_ref'), False)} %; Δ {_pt(k.get('delta'))} pt (bootstrap "
                   f"[{_pt((k.get('delta_ci') or [NAN])[0])}, {_pt((k.get('delta_ci') or [NAN, NAN])[1])}], Newcombe "
                   f"[{(k.get('newcombe_pt') or [NAN])[0]:+.1f}, {(k.get('newcombe_pt') or [NAN, NAN])[1]:+.1f}]); wins / losses "
                   f"{k['wins']} / {k['losses']}")
        out.append(f"- cost ${1000 * k['cost']:.1f} vs ${1000 * k['cost_ref']:.1f} per 1k episodes (saving "
                   f"{_pt(k.get('saving'), False)} %); System-1 ${1000 * pe['s1_cost']:.2f}, System-2 ${1000 * pe['s2_cost']:.1f} "
                   f"per 1k; LLM calls {pe['llm_calls']:.2f} vs {pr['llm_calls']:.2f}, System-1 calls {pe['s1_calls']:.2f} per "
                   f"episode; wall {pe['wall_s_mean']:.1f} s vs {pr['wall_s_mean']:.1f} s (median {pe['wall_s_median']:.1f} vs "
                   f"{pr['wall_s_median']:.1f}); episodes with an empty LLM reply {pe.get('empty_reply_episodes', 0)} vs "
                   f"{pr.get('empty_reply_episodes', 0)}")
        out.append("- decision point: " + ", ".join(f"{kk} {_fmt_point(kk, vv)}" for kk, vv in pm.items()
                                                    if kk != "acted_kind"))
        if c.get("vs_flash"):
            vf = c["vs_flash"]
            out.append(f"- vs Flash alone (`llm_only`, descriptive): Δ {_pt(vf.get('delta'))} pt, saving "
                       f"{_pt(vf.get('saving'), False)} %, wins / losses {vf.get('wins')} / {vf.get('losses')}")
        out.append("")
    return "\n".join(out)


def to_tex_keys(res: Mapping[str, Any]) -> dict[str, str]:
    vals: dict[str, str] = {}
    for arm, c in res["cells"].items():
        p = f"{PREFIX}/{arm}"
        vals[f"{p}/state"] = c["state"]
        if c["state"] != "exploratory":
            continue
        vals.update(MM._num_keys(p, c["cmp"], c["points"]))
        k = c["cmp"]
        for kk in ("wins", "losses"):
            if k.get(kk) is not None:
                vals[f"{p}/{kk}"] = str(int(k[kk]))
        if c["verdict"]:
            vals[f"{p}/verdict"] = c["verdict"]
            vals[f"{p}/vcode"] = str(MM.VCODE[c["verdict"]])
        for kk in ("binding", "llm_acc_deferred", "veto_llm_was_wrong", "veto_on_other", "veto_tool_changed",
                   "veto_precision", "overrode", "recall_k", "s1_acc_all", "followed", "llm_final_acc",
                   "llm_store_rate", "final_covers", "model_yes_rate", "gold_positive_rate"):
            if MM._finite(c["points"].get(kk)):
                vals[f"{p}/{kk}"] = f"{100 * c['points'][kk]:.1f}"
        pe = c["per_episode"]
        vals[f"{p}/wall_s"] = f"{pe['wall_s_mean']:.1f}"
        vals[f"{p}/ref"] = c["reference"].replace("_", r"\_")
        if c.get("equals"):     # E6e: the cell this replace equals (its first words, before the colon)
            vals[f"{p}/equals"] = c["equals"].split(":")[0]
        if c.get("vs_flash"):
            vals.update({kk: vv for kk, vv in MM._num_keys(f"{p}/flash", c["vs_flash"], None).items()
                         if kk.rsplit("/", 1)[-1] in ("delta", "saving", "n")})
    for ref, r in res["references"].items():
        if MM._finite(r.get("success")):
            vals[f"{PREFIX}/ref/{ref}/success"] = f"{100 * r['success']:.1f}"
            vals[f"{PREFIX}/ref/{ref}/cost_k"] = f"{1000 * r['cost']:.1f}"
            vals[f"{PREFIX}/ref/{ref}/n"] = str(r["n_ok"])
    sp = res["meta"]["spend"]
    vals[f"{PREFIX}/spend"] = f"{sp['total']:.2f}"
    vals[f"{PREFIX}/cap"] = f"{res['meta']['cap']:.0f}"
    vals[f"{PREFIX}/count/cells"] = str(len(CELLS))
    vals[f"{PREFIX}/count/run"] = str(sum(c["state"] == "exploratory" for c in res["cells"].values()))
    return vals


def write_tex(vals: Mapping[str, str], path: Path | None = None) -> Path:
    path = TEX if path is None else path
    lines = [f"% AUTO-GENERATED by scripts/explore_{PREFIX}_modes.py -- do not edit by hand.",
             f"% Exploratory row {PREFIX.upper()} (descriptive; docs/PREREG_A7_STAGE2.md Deviations 2026-10-06), outside "
             "both stages.",
             f"% last written {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git "
             f"{(C.git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


def cmd_summarize(a: argparse.Namespace) -> int:
    res = summarize(Path(a.runs), a.n_boot, a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(MM._clean(res), indent=1, default=MM._json_default) + "\n")
    (OUT_DIR / "summary.md").write_text(to_markdown(res) + "\n")
    write_tex(to_tex_keys(res), Path(a.tex) if a.tex else None)
    print(to_markdown(res))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(RUNS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spend")
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--cap", type=float, default=CAP_USD)
    w.add_argument("--every", type=float, default=30.0)
    sub.add_parser("project")
    s = sub.add_parser("summarize")
    s.add_argument("--n-boot", type=int, default=MM.N_BOOT)
    s.add_argument("--seed", type=int, default=MM.SEED)
    s.add_argument("--tex", default=None)
    a = ap.parse_args(argv)
    return {"spend": cmd_spend, "watch": cmd_watch, "project": cmd_project, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
