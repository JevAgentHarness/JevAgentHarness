#!/usr/bin/env python3
"""Hard-task (H5, exploratory) readouts -> \\res{} macros for the paper (paper/generated/hard.tex).

Reads only the committed H5 summaries (no run directory, no API):
  results/hard/swe/h5_summary.json                       SWE-bench Verified, 4 arms x 100, k = 1 (registered readout)
  results/hard/swe/h5_summary_reeval_sensitivity.json    the same with two load-induced evaluation timeouts re-evaluated
                                                         (sensitivity view, not the registered readout)
  results/hard/tbsci/summary.json                        Terminal-Bench-Science, 3 arms x 15 tasks x k = 3
  results/hard/tbsci_cal/select.json                     the frozen-subset selection (pool, target N)

Keys (percent for success and deltas, seconds, USD per task):
  swe_verified_100/<arm>/{success, resolved, n, vs_baseline/{delta, delta_lb90, wins, losses, cost_savings},
      wall_p50_s, wall_p90_s, cost_usd, steps, screen, faster, verdict, time_both_solved/{n, d_median_s, sign_p},
      sens/{success, vs_baseline/delta, verdict}}
  tbsci/{n_tasks, k, n_trials, pool, n_target}
  tbsci/<arm>/{success, solved, n, vs_baseline/{delta, delta_lb90, wins, losses, d_run_median_all_s, cost_savings},
      wall_p50_s, wall_p90_s, cost_usd, steps, s1_requests, timeouts, tests_passed_share, screen, faster, verdict}

Labels follow the registered H5 readout (docs/PREREG_EFFECTIVE.md, A5.2 / A6.6 / A6.16 (j) / A6.22 (d)): the
*exploratory success screen* (one-sided 90% lower bound of the success difference > -5 pt and wins >= losses; not
a size-controlled NI test) and *faster among jointly solved tasks* (median paired time difference < 0 on the tasks
both arms solve and an exact one-sided sign test p < 0.05). Usage: python scripts/hard_to_tex.py [--out FILE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARD = ROOT / "results" / "hard"


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _num(x, nd=1, signed=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    elif s.startswith("+"):
        s = "$+$" + s[1:]
    return s


def _labels(screen: bool, faster: bool, strict: bool, n_both: int) -> tuple[str, str, str]:
    """(screen mark, faster mark, displayed verdict) of one arm."""
    f_mark = "n.e." if n_both == 0 else ("yes" if faster else "no")
    if screen and faster:
        verdict = "passes the exploratory success screen; faster among jointly solved tasks"
    elif screen:
        verdict = "passes the exploratory success screen; not faster among jointly solved tasks"
    else:
        verdict = "does not pass the exploratory success screen; not faster among jointly solved tasks"
    if n_both == 0:
        verdict = verdict.split(";")[0] + "; faster not estimable (no jointly solved task)"
    if strict:
        verdict += " (faster-and-better tier: success)"
    return ("pass" if screen else "fail"), f_mark, verdict


def swe(values: dict[str, str]) -> None:
    main = json.loads((HARD / "swe" / "h5_summary.json").read_text())
    sens_path = HARD / "swe" / "h5_summary_reeval_sensitivity.json"
    sens = json.loads(sens_path.read_text()) if sens_path.exists() else None
    arms = {a["arm"]: a for a in main["arms"]}
    base = arms["baseline"]
    base_cost = base["cost_total"] / base["n"]
    values["swe_verified_100/n_tasks"] = str(base["n"])
    for name, a in arms.items():
        k = f"swe_verified_100/{name}"
        cost = a["cost_total"] / a["n"]
        values[f"{k}/success"] = _num(100 * a["success"])
        values[f"{k}/resolved"] = str(a["resolved"])
        values[f"{k}/n"] = str(a["n"])
        values[f"{k}/wall_p50_s"] = _num(a["time_p50_s"], 0)
        values[f"{k}/wall_p90_s"] = _num(a["time_p90_s"], 0)
        values[f"{k}/cost_usd"] = _num(cost, 3)
        values[f"{k}/steps"] = _num(a["steps_mean"], 1)
        if name == "baseline":
            continue
        p = main["paired"][name]["full set"]
        values[f"{k}/vs_baseline/delta"] = _num(100 * p["d_success"], 1, signed=True)
        values[f"{k}/vs_baseline/delta_lb90"] = _num(100 * p["d_success_lb"], 1, signed=True)
        values[f"{k}/vs_baseline/wins"] = str(p["wins"])
        values[f"{k}/vs_baseline/losses"] = str(p["losses"])
        values[f"{k}/vs_baseline/cost_savings"] = _num(100 * (base_cost - cost) / base_cost, 1, signed=True)
        values[f"{k}/time_both_solved/n"] = str(p["n_both_solved"])
        values[f"{k}/time_both_solved/d_median_s"] = _num(p["d_time_median_solved"], 1, signed=True)
        values[f"{k}/time_both_solved/sign_p"] = _num(p["time_sign_p"], 2)
        scr, fst, ver = _labels(bool(p["success_preserving"]), bool(p["faster"]), bool(p["strictly_better"]),
                                int(p["n_both_solved"]))
        values[f"{k}/screen"], values[f"{k}/faster"], values[f"{k}/verdict"] = scr, fst, ver
        if sens:
            sa = {x["arm"]: x for x in sens["arms"]}[name]
            sp = sens["paired"][name]["full set"]
            values[f"{k}/sens/success"] = _num(100 * sa["success"])
            values[f"{k}/sens/vs_baseline/delta"] = _num(100 * sp["d_success"], 1, signed=True)
            values[f"{k}/sens/verdict"] = _labels(bool(sp["success_preserving"]), bool(sp["faster"]),
                                                  bool(sp["strictly_better"]), int(sp["n_both_solved"]))[2]


def tbsci(values: dict[str, str]) -> None:
    d = json.loads((HARD / "tbsci" / "summary.json").read_text())
    arms = d["arms"]
    base = arms["baseline"]
    base_cost = base["llm_cost_mean"] + base["s1_cost_mean"]
    values["tbsci/n_tasks"] = str(base["tasks"])
    values["tbsci/n_trials"] = str(base["n"])
    values["tbsci/k"] = str(base["n"] // base["tasks"])
    sel_path = HARD / "tbsci_cal" / "select.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        values["tbsci/pool"] = str(len(sel["tasks"]))
        values["tbsci/n_target"] = str(sel["n_target"])
    share: dict[str, list[float]] = {}
    for r in d["rows"]:
        if r.get("tests_total"):
            share.setdefault(r["job"], []).append(r["tests_passed"] / r["tests_total"])
    for name, a in arms.items():
        k = f"tbsci/{name}"
        cost = a["llm_cost_mean"] + a["s1_cost_mean"]
        values[f"{k}/success"] = _num(100 * a["success"])
        values[f"{k}/solved"] = str(a["solved"])
        values[f"{k}/n"] = str(a["n"])
        values[f"{k}/wall_p50_s"] = _num(a["agent_run_p50_s"], 0)
        values[f"{k}/wall_p90_s"] = _num(a["agent_run_p90_s"], 0)
        values[f"{k}/cost_usd"] = _num(cost, 2)
        values[f"{k}/steps"] = _num(a["steps_mean"], 1)
        values[f"{k}/s1_requests"] = _num(a["s1_requests_mean"], 1)
        values[f"{k}/timeouts"] = str(a["timeouts"])
        if share.get(name):
            values[f"{k}/tests_passed_share"] = _num(100 * sum(share[name]) / len(share[name]), 1)
        if name == "baseline":
            continue
        p = d["paired"][name]
        values[f"{k}/vs_baseline/delta"] = _num(100 * p["d_success_mean"], 1, signed=True)
        values[f"{k}/vs_baseline/delta_lb90"] = _num(100 * p["d_success_lb90"], 1, signed=True)
        values[f"{k}/vs_baseline/wins"] = str(p["wins"])
        values[f"{k}/vs_baseline/losses"] = str(p["losses"])
        values[f"{k}/vs_baseline/d_run_median_all_s"] = _num(p["d_agent_run_s_median"], 1, signed=True)
        values[f"{k}/vs_baseline/cost_savings"] = _num(100 * (base_cost - cost) / base_cost, 1, signed=True)
        scr, fst, ver = _labels(bool(p["h5_better"]), bool(p["h5_faster"]), bool(p["h5_strictly_better"]),
                                int(p["n_h5_both_solved"]))
        values[f"{k}/screen"], values[f"{k}/faster"], values[f"{k}/verdict"] = scr, fst, ver


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated" / "hard.tex"))
    a = ap.parse_args(argv)
    values: dict[str, str] = {}
    swe(values)
    tbsci(values)
    lines = ["% AUTO-GENERATED by scripts/hard_to_tex.py -- do not edit by hand.",
             f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git {_git_head()}",
             "% H5 (exploratory, outside the Holm family): results/hard/swe/, results/hard/tbsci/",
             "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(values.items())]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(values)} keys to {out}")


if __name__ == "__main__":
    main()
