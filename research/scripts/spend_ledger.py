#!/usr/bin/env python3
"""Total experiment spend (docs/PREREG_A7_STAGE2.md A7.16 (g)) from the ledgers the runs wrote; no API call.

API spend per track is the sum of the per-episode (or per-decision) ledger cost over every record of the track's run
files under runs/ (every attempt, failed ones included, as billed). A record's cost is ``usage.total_with_shadow``
when present (measurement calls included: shadow System-1 questions, counterfactual System-2 calls, probes; the
harness keeps them out of ``usage.total``), else ``usage.total`` / ``cost_usd``. Exceptions:
  * simulated records (``llm`` scripted / simulated, ``backend`` simulated): never spend;
  * derived copies of a run (``pilot_jev_n50_split``: a re-split of ``pilot_jev_n50``; ``tau2/airline_test``: a split
    of ``tau2/airline``): counted once;
  * tau^2 runs (a ``raw/<domain>/<arm>/`` save directory with the side-car's ``jev_decisions.jsonl``): the side-car's
    agent ledger over every attempt plus tau2's user-simulator cost (``results.json`` and ``jev_errored/``); the
    converted per-task copies (records with ``meta.tau2``) are not counted again;
and except runs whose operational record is the ledger: Stage-1 tau^2 (``results/tau2/runs.json``: tau2's agent and
user-simulator cost plus the shadow System-1), the hard tasks (``results/hard/NOTES.md``, the round's logged total),
the training rounds (``results/gpu_round5/spend.json``; ``results/gpu_round6_miles/spend_before_*.txt``, the last
ledger written), the A7.12 (i) gpt-6.1-sol row (``results/stage2/sol/summary.json`` ``spend``: ledger + the cache-write
surcharge + the capability probe; its run files are skipped here), the LLM reviews of drafts
(``results/reviews/**/run_log*.jsonl``) and the figure drafts (``runs/dev/figures/**/ledger.jsonl``). Calls without a
verified price (``priced: false``, or a figure cost on an assumed list price) count $0 and are listed as unpriced,
never estimated. GPU spend is reported where a host bill was recorded (round 5: host $ / h x hours; the exploratory
E1/E2 box: host $ / h x its recorded window, ``results/explore/gpu/host_window.json``); the other GPU rounds' host
bills were not logged and are listed as such, never estimated.

The amortisation volume (A7.16 (g)) is JevAgentRL's training and calibration cost (round 5's API and GPU spend plus the
data v1 build) divided by S2-4's per-episode saving (the registered cost clause's estimate). The realised S2-4 host
bill (A7.13 (iii)) is bounded by the host's $ / h times the time between S2-4's start record and the next stage's start
record on the same host, over its 500 episodes.

Writes results/spend/spend_ledger.{json,md} and the paper's \\defres file paper/generated/spend.tex
(``spend/...`` keys). Usage: python scripts/spend_ledger.py [--runs runs] [--out results/spend] [--tex FILE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# track -> run directories under runs/ (paths relative to runs/; a directory counts every *.jsonl below it)
RUN_TRACKS: dict[str, list[str]] = {
    "stage1_main": ["e1_bfcl_when2call", "e2a_jevgym_tool_decisions", "e2a_jevgym_tool_decisions_intern",
                    "e2b_jevgym_meta_control", "e2c_jevgym_baselines"],
    "stage1_calibration": ["e2a_jevgym_shadow_cal", "e2a_jevgym_shadow_cal_intern", "cal", "cal_intern"],
    "stage2_evaluation": ["stage2_jevgym", "stage2_x", "stage2_tau2", "stage2_rl_states",
                          "stage2_startlux_states", "stage2_rl_attribution", "e1_bfcl_when2call_rl",
                          "e1_bfcl_when2call_startlux"],
    "stage2_calibration_checks": ["e2a_jevgym_shadow_cal_x", "e2a_jevgym_shadow_cal_startlux", "stage2_dev",
                                  "stage2_dev_tariff_rl", "stage2_dev_tariff_rl_warmup", "stage2_dev_tariff_sl4b",
                                  "stage2_dev_tariff_sl4b_warmup", "stage2_tau2_validation",
                                  "e2c_jevgym_baselines_check"],
    "round6_evaluation": ["stage2_v2", "stage2_v2_states", "e2a_jevgym_shadow_cal_v2", "stage2_dev_tariff_v2",
                          "stage2_dev_tariff_v2_warmup"],
    "exploratory": ["explore_startlux_live", "explore_telecom_rl_validation", "explore_telecom_rl_validation2",
                    "e1_bfcl_when2call_startlux_ladder"],
    "development_pilots": ["dev", "pilot", "pilot_jev", "pilot_jev_r2", "pilot_jev_v2", "pilot_jev_n50", "gpu_round3",
                           "smoke_main", "decision"],
}
TRACK_LABELS = {
    "stage1_main": "Stage 1: main runs (E1, E2a, E2a', E2b, E2c)", "stage1_tau2": "Stage 1: tau^2 (C7)",
    "stage1_calibration": "Stage 1: calibration arms", "stage2_evaluation": "Stage 2: evaluation",
    "stage2_calibration_checks": "Stage 2: calibration, tariffs and checks",
    "stage2_sol": "Stage 2: gpt-6.1-sol row (A7.12 (i), descriptive; eligibility, calibration, evaluation)",
    "round6_evaluation": "JevAgentRL-v2 evaluation (A7.19 (f), exploratory)",
    "exploratory": "Exploratory E1 go/no-go and E2 (Deviations 2026-10-05)",
    "development_pilots": "Development and pilots", "hard_tasks": "Hard tasks (H5)",
    "training_v1_data": "JevAgentRL data v1 (episodes and teacher labels)",
    "training_round5": "JevAgentRL development (round 5)", "training_round6": "JevAgentRL-v2 development (round 6)",
    "reviews": "LLM reviews of drafts (priced calls)", "figures": "Figure drafts: image generation and critiques (priced calls)"}
EXCLUDED = {"pilot_jev_n50_split": "a re-split copy of pilot_jev_n50", "tau2": "ledger: results/tau2/runs.json",
            "*/telecom, */retail, ... (records with meta.tau2)": "per-task copies of tau2 runs, counted from their raw/ "
                                                                  "side-car ledger and results.json"}
# run files whose operational record is a summary's spend block (counted there, skipped in the run tracks)
OWN_LEDGER_RUNS = {"stage2_sol": "stage2_sol", "e2a_jevgym_shadow_cal_sol": "stage2_sol",
                   "stage2_dev/sol_eligibility": "stage2_sol"}


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record_cost(r: dict[str, Any]) -> float | None:
    """The ledger cost of one record (episode or decision), or None when it carries none or is simulated."""
    if str(r.get("llm") or "").lower() in ("scripted", "simulated") or str(r.get("backend") or "") == "simulated":
        return None
    u = r.get("usage") or {}
    c = (u.get("total_with_shadow") or {}).get("cost_usd")        # what the run spent, measurement calls included
    if c is None:
        c = (u.get("total") or {}).get("cost_usd")
    if c is None:
        c = r.get("cost_usd")
    return float(c) if isinstance(c, (int, float)) else None


def tau2_user_spend(save: Path) -> tuple[float, int]:
    """tau2's user-simulator cost of one save directory (``results.json`` and pruned ``jev_errored/`` attempts)."""
    sims: list[Any] = []
    if (save / "results.json").exists():
        sims += json.loads((save / "results.json").read_text()).get("simulations") or []
    for p in sorted((save / "jev_errored").glob("*.json")) if (save / "jev_errored").is_dir() else []:
        try:
            sims.append(json.loads(p.read_text()))
        except ValueError:
            pass
    return sum(float(s.get("user_cost") or 0) for s in sims if isinstance(s, dict)), len(sims)


def run_spend(d: Path, skip: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Sum of the ledger cost over every record of every *.jsonl below ``d`` (every attempt counted), files below a
    ``skip`` path excluded. A tau2 run (a side-car ``jev_decisions.jsonl`` below ``d``) is counted from its side-car
    agent ledger plus tau2's user-simulator cost; its converted per-task copies (``meta.tau2``) are not counted again."""
    usd, n = 0.0, 0
    files = [f for f in (sorted(d.rglob("*.jsonl")) if d.is_dir() else [])
             if not any(f.is_relative_to(s) for s in skip)]
    has_sidecar = any(f.name == "jev_decisions.jsonl" for f in files)
    for f in files:
        with open(f) as fh:
            for line in fh:
                if '"cost_usd"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(r, dict):
                    continue
                if has_sidecar and f.name != "jev_decisions.jsonl" and "tau2" in (r.get("meta") or {}):
                    continue                                   # converted copy of a side-car record
                c = record_cost(r)
                if c is not None:
                    usd, n = usd + c, n + 1
        if f.name == "jev_decisions.jsonl":
            usd += tau2_user_spend(f.parent)[0]
    return {"usd": usd, "records": n}


def review_spend(root: Path) -> dict[str, Any]:
    """LLM reviews of drafts: priced cost of every call in ``results/reviews/**/run_log*.jsonl``; unpriced calls
    (``priced: false``) are counted by model, never estimated."""
    usd, n, unpriced = 0.0, 0, {}
    for f in sorted(root.rglob("run_log*.jsonl")) if root.is_dir() else []:
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or r.get("status") != "ok":
                continue
            if r.get("priced") is False:
                unpriced[str(r.get("model"))] = unpriced.get(str(r.get("model")), 0) + 1
                continue
            usd, n = usd + float(r.get("cost_usd") or 0.0), n + 1
    return {"usd": usd, "records": n, "unpriced_calls": unpriced}


PRICED_FIGURE_BASIS = "usage_tokens_x_price_table"


def figure_spend(root: Path) -> dict[str, Any]:
    """Figure drafts (image generation and critiques): ``cost_usd_estimate`` of every ``ledger.jsonl`` entry below
    ``root`` priced from the price table; entries on an assumed list price are counted by model as unpriced."""
    usd, n, unpriced = 0.0, 0, {}
    for f in sorted(root.rglob("ledger.jsonl")) if root.is_dir() else []:
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or not isinstance(r.get("cost_usd_estimate"), (int, float)):
                continue
            if r.get("cost_basis") != PRICED_FIGURE_BASIS:
                unpriced[str(r.get("model"))] = unpriced.get(str(r.get("model")), 0) + 1
                continue
            usd, n = usd + float(r["cost_usd_estimate"]), n + 1
    return {"usd": usd, "records": n, "unpriced_calls": unpriced}


def host_window_bill(window: Path, plan: Path) -> dict[str, Any] | None:
    """A GPU box's bill from its recorded window (``host_window.json``) at the planned $ / h of its instance type."""
    if not window.exists() or not plan.exists():
        return None
    w = json.loads(window.read_text())
    rate = float(json.loads(plan.read_text())["host"]["usd_per_hour_by_type"][w["instance_type"]])
    t0 = dt.datetime.fromisoformat(w["start_utc"].replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(w["stop_utc"].replace("Z", "+00:00"))
    hours = (t1 - t0).total_seconds() / 3600
    return {"usd": hours * rate, "hours": hours, "usd_per_h": rate, "instance_type": w["instance_type"],
            "start_utc": w["start_utc"], "stop_utc": w["stop_utc"],
            "source": f"{window.relative_to(ROOT) if window.is_relative_to(ROOT) else window} x "
                      f"{plan.relative_to(ROOT) if plan.is_relative_to(ROOT) else plan} (a lower bound: boot unrecorded)"}


def hard_tasks_total(notes: Path) -> float | None:
    m = re.search(r"\*\*total of this round\*\*\s*\|[^|]*\|[^|]*\|\s*\*\*\$([0-9.,]+)\*\*", notes.read_text())
    return float(m.group(1).replace(",", "")) if m else None


def round6_api(d: Path) -> tuple[float | None, str | None]:
    best: tuple[float, str] | None = None
    for f in sorted(d.glob("spend_before_*.txt")):
        m = re.search(r"^\s*([0-9.]+)\s+total\s*$", f.read_text(), re.M)
        if m and (best is None or float(m.group(1)) > best[0]):
            best = (float(m.group(1)), f.name)
    return (best[0], best[1]) if best else (None, None)


def _start(path: Path) -> dt.datetime | None:
    if not path.exists():
        return None
    m = re.search(r"^started\s+(\S+)", path.read_text(), re.M)
    return dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00")) if m else None


def ledger(runs: Path, reviews: Path | None = None, figures: Path | None = None) -> dict[str, Any]:
    tracks: dict[str, dict[str, Any]] = {}
    skip = tuple(runs / p for p in OWN_LEDGER_RUNS)
    for tr, dirs in RUN_TRACKS.items():
        per = {d: run_spend(runs / d, skip) for d in dirs}
        tracks[tr] = {"usd": sum(v["usd"] for v in per.values()), "records": sum(v["records"] for v in per.values()),
                      "runs": {k: round(v["usd"], 4) for k, v in per.items()}, "source": "run ledgers under runs/"}
    sol = ROOT / "results/stage2/sol/summary.json"
    if sol.exists():
        sp = json.loads(sol.read_text())["spend"]
        tracks["stage2_sol"] = {"usd": float(sp["total_usd"]), "at": sp.get("at"),
                                "source": "results/stage2/sol/summary.json (spend.total_usd: ledger + cache-write "
                                          "surcharge + capability probe; runs " + ", ".join(sorted(OWN_LEDGER_RUNS))
                                          + ")"}
    rv = review_spend(reviews if reviews is not None else ROOT / "results/reviews")
    tracks["reviews"] = {**rv, "source": "results/reviews/**/run_log*.jsonl (priced calls)"}
    fg = figure_spend(figures if figures is not None else runs / "dev/figures")
    tracks["figures"] = {**fg, "source": "runs/dev/figures/**/ledger.jsonl (priced from the price table)"}
    t2 = json.loads((ROOT / "results/tau2/runs.json").read_text())
    tracks["stage1_tau2"] = {"usd": float(t2["total_logged_usd"]), "source": "results/tau2/runs.json (total_logged_usd)"}
    h = hard_tasks_total(ROOT / "results/hard/NOTES.md")
    tracks["hard_tasks"] = {"usd": h, "source": "results/hard/NOTES.md (logged total of the round)"}
    man = json.loads((ROOT / "results/training/train_v1/manifest.json").read_text())
    tracks["training_v1_data"] = {"usd": float(man["cost"]["total"]), "source": "results/training/train_v1/manifest.json"}
    r5 = json.loads((ROOT / "results/gpu_round5/spend.json").read_text())
    tracks["training_round5"] = {"usd": float(r5["api_usd_total"]), "source": "results/gpu_round5/spend.json (api_usd_total)"}
    r6, r6f = round6_api(ROOT / "results/gpu_round6_miles")
    tracks["training_round6"] = {"usd": r6, "source": f"results/gpu_round6_miles/{r6f} (last ledger written; the final "
                                                      "E2 dev slice after it and OPD's lost executor / teacher records "
                                                      "are not in it: TRAINING_MANIFEST.md)" if r6f else "not logged"}
    api_total = sum(v["usd"] for v in tracks.values() if v.get("usd") is not None)
    unpriced: dict[str, int] = {}
    for v in tracks.values():
        for m, k in (v.get("unpriced_calls") or {}).items():
            unpriced[m] = unpriced.get(m, 0) + k

    gpu = {"round5": {"usd": float(r5["gpu_usd"]), "hours": float(r5["gpu_hours"]),
                      "usd_per_h": float(r5["gpu_usd_per_h"]), "source": "results/gpu_round5/spend.json"},
           "explore": host_window_bill(ROOT / "results/explore/gpu/host_window.json",
                                       ROOT / "results/stage2/tariff_v2_plan.json"),
           "not_logged": ["GPU rounds 2-4 (Intern-Decision serving and the E2a' arms)",
                          "round 6 (JevAgentRL-v2 on Miles; budget cap $400)",
                          "the Stage-2 GPU host (StartLux, JevAgentRL and JevAgentRL-v2 serving) beyond the S2-4 "
                          "bound below",
                          "the hard-task boxes (EC2 instance hours)"]}

    s2 = json.loads((ROOT / "results/stage2/stage2.json").read_text())
    saving = float(s2["family"]["S2-4"]["clauses"]["cost"]["est"])
    train_cost = float(r5["api_usd_total"]) + float(r5["gpu_usd"]) + float(man["cost"]["total"])
    amort = {"training_and_calibration_usd": train_cost, "saving_usd_per_episode": saving,
             "episodes": train_cost / saving if saving > 0 else None,
             "basis": "round 5 API + GPU spend (training, calibration arms, development slices) + data v1 build; "
                      "S2-4's registered cost clause (LLM-only minus the profile with JevAgentRL, tariff included)"}

    tariff = json.loads((ROOT / "results/stage2/tariff.json").read_text())
    t0 = _start(ROOT / "results/stage2/gpu/START_stage2_jevgym_rl.txt")
    t1 = _start(ROOT / "results/stage2/gpu/START_stage2_rl_states.txt")
    host: dict[str, Any] = {"usd_per_h": float(tariff["host_usd_per_hour"]), "tariff_q": float(tariff["q_usd_per_episode"]),
                            "episodes": int(s2["family"]["S2-4"]["n_pairs"])}
    if t0 and t1 and t1 > t0:
        hours = (t1 - t0).total_seconds() / 3600
        host.update({"hours_upper": hours, "usd_upper": hours * host["usd_per_h"],
                     "per_episode_upper": hours * host["usd_per_h"] / host["episodes"],
                     "basis": "host $ / h x (start of stage2_rl_states - start of stage2_jevgym_rl) on the same host; "
                              "an upper bound on S2-4's evaluation-host bill"})
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git_head(),
            "api": {"tracks": tracks, "total_usd": api_total, "unpriced_calls": unpriced}, "gpu": gpu,
            "amortisation": amort,
            "s24_host_bill": host, "excluded": EXCLUDED}


def to_md(L: dict[str, Any]) -> str:
    out = ["# Total experiment spend (A7.16 (g)); generated by scripts/spend_ledger.py", "",
           "| track | API spend (USD) | source |", "|---|---|---|"]
    for k, v in L["api"]["tracks"].items():
        u = "not logged" if v.get("usd") is None else f"{v['usd']:.2f}"
        out.append(f"| {TRACK_LABELS.get(k, k)} | {u} | {v['source']} |")
    out += [f"| **total** | **{L['api']['total_usd']:.2f}** | |", ""]
    up = L["api"].get("unpriced_calls") or {}
    if up:
        out += ["Unpriced calls (no verified price; $0 above, never estimated): "
                + ", ".join(f"{m} x {k}" for m, k in sorted(up.items())) + ".", ""]
    ex = L["gpu"].get("explore")
    gx = (f"; exploratory E1/E2 box ${ex['usd']:.2f} over {ex['hours']:.2f} h ({ex['instance_type']} at "
          f"${ex['usd_per_h']:.5f} / h, {ex['start_utc']} to {ex['stop_utc']}; {ex['source']})") if ex else ""
    out += [f"GPU (recorded host bills): round 5 ${L['gpu']['round5']['usd']:.2f} over {L['gpu']['round5']['hours']:.2f} h"
            f"{gx}; not logged: " + "; ".join(L["gpu"]["not_logged"]) + ".", ""]
    a = L["amortisation"]
    out += [f"Amortisation (JevAgentRL): ${a['training_and_calibration_usd']:.2f} / ${a['saving_usd_per_episode']:.4f} per "
            f"episode = {a['episodes']:.0f} episodes ({a['basis']}).", ""]
    h = L["s24_host_bill"]
    if "usd_upper" in h:
        out += [f"S2-4 realised host bill: at most ${h['usd_upper']:.2f} ({h['hours_upper']:.2f} h at ${h['usd_per_h']:.2f} / h), "
                f"i.e. at most ${h['per_episode_upper']:.5f} per episode against the frozen tariff q = ${h['tariff_q']:.5f}.", ""]
    out += ["Excluded: " + "; ".join(f"{k} ({v})" for k, v in L["excluded"].items()) + ".", ""]
    return "\n".join(out)


def _usd(v: float) -> str:
    return f"{v:,.2f}".replace(",", "{,}")


def to_tex(L: dict[str, Any]) -> str:
    vals: dict[str, str] = {}
    for k, v in L["api"]["tracks"].items():
        if v.get("usd") is not None:
            vals[f"spend/api/{k}"] = _usd(v["usd"])
    vals["spend/api/total"] = _usd(L["api"]["total_usd"])
    g = L["gpu"]["round5"]
    vals["spend/gpu/round5/usd"] = _usd(g["usd"])
    vals["spend/gpu/round5/hours"] = f"{g['hours']:.1f}"
    ex = L["gpu"].get("explore")
    if ex:
        vals["spend/gpu/explore/usd"] = _usd(ex["usd"])
        vals["spend/gpu/explore/hours"] = f"{ex['hours']:.1f}"
    vals["spend/unpriced_calls"] = str(sum((L["api"].get("unpriced_calls") or {}).values()))
    a = L["amortisation"]
    vals["spend/amortisation/training_usd"] = _usd(a["training_and_calibration_usd"])
    vals["spend/amortisation/saving_per_episode"] = f"{a['saving_usd_per_episode']:.4f}"
    if a.get("episodes"):
        vals["spend/amortisation/episodes"] = f"{a['episodes']:,.0f}".replace(",", "{,}")
    h = L["s24_host_bill"]
    vals["spend/s24_host/usd_per_h"] = f"{h['usd_per_h']:.2f}"
    vals["spend/s24_host/tariff_q"] = f"{h['tariff_q']:.5f}"
    if "usd_upper" in h:
        vals["spend/s24_host/hours_upper"] = f"{h['hours_upper']:.2f}"
        vals["spend/s24_host/usd_upper"] = f"{h['usd_upper']:.2f}"
        vals["spend/s24_host/per_episode_upper"] = f"{h['per_episode_upper']:.5f}"
    lines = ["% AUTO-GENERATED by scripts/spend_ledger.py -- do not edit by hand.",
             f"% generated {L['generated']} at git {L['git']}",
             "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(vals.items())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "spend"))
    ap.add_argument("--tex", default=str(ROOT / "paper" / "generated" / "spend.tex"))
    a = ap.parse_args(argv)
    L = ledger(Path(a.runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spend_ledger.json").write_text(json.dumps(L, indent=1) + "\n")
    (out / "spend_ledger.md").write_text(to_md(L))
    Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tex).write_text(to_tex(L))
    print(to_md(L))


if __name__ == "__main__":
    main()
