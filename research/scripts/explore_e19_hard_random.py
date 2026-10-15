#!/usr/bin/env python3
r"""E19 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E19: the rule-randomised model on
DecisionGym-Hard"; outside both registered stages and every Holm table): E12b's checkpoint "AgentDecision-4B (random
rules)", trained on randomised order rules that are never the tier rule, run live in E7f's setting on E7f's 200
JevGym-Hard tasks (seeds 310600-310799), paired per task with E7f's logged LLM-only arm (the reference, never re-run)
and with E7f's retrained arm. The readout is explore_e7_hard.py's (experiment "e19": ref_runs, extra_refs, the n_step
gate); this script adds the operations of the registration's pause rule (h) and writes the row's outputs.

    python scripts/explore_e19_hard_random.py probe --log FILE      # one probe round: six one-word Flash calls on the
                                                                    # direct API through the harness client; exit 0 if 6 of 6
    python scripts/explore_e19_hard_random.py watch --pid PID --cap 56 [--prior 0] [--every 30] [--no-pause]
        # the $ cap (-> runs/e19_STOP.json) and, unless --no-pause, a pause on a new episode that errored with 503 /
        # overload (-> <run>/PAUSED_overload.json); the runner's process group is stopped either way; records are kept
    python scripts/explore_e19_hard_random.py errored --run runs/e19_hard
    python scripts/explore_e19_hard_random.py project [--prior 0] [--out FILE]   # the registered gate (explore_e7_hard)
    python scripts/explore_e19_hard_random.py spend [--short]
    python scripts/explore_e19_hard_random.py summarize [--window FILE] [--n-boot 10000] [--seed 0]
        # -> results/explore/e19_hard_random/summary.{json,md}, paper/generated/e19_hard_random.tex (keys e19/...) and
        #    paper/generated/hard_ni.tex (scripts/hard_ni.py, row e19: the registered reading)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402
import explore_e7_hard as E7  # noqa: E402
import hard_ni  # noqa: E402
import module_map as MM  # noqa: E402

ROOT = C.ROOT
EXP = "e19"
X = E7.EXPS[EXP]
OUT = Path(X["out"])
LABEL = ("exploratory (E19), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E19: the rule-randomised model on "
         "DecisionGym-Hard'; outside every Holm table")
OVERLOAD = ("503", "overload", "high demand", "unavailable", "resource_exhausted", "429")
PROBE_N = 6
FLASH = {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0, "reasoning_effort": "minimal"}
LIVE_RUNS = (X["smoke"], X["eval"], *(p[1] for p in X.get("eval_parts") or ()))   # the smoke, the first part, the extension
N_FULL = int(X["n_eval"])


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ================================================================================================ operations (box)
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
        p.write_text("# E19: direct Gemini API probes (Deviations 2026-10-08 'E19' (h), E14's rule). One round = six "
                     "one-word Gemini 3.5 Flash calls; a live stage starts / resumes after two consecutive 6/6 rounds.\n")
    with open(p, "a") as f:
        f.write(line.replace("\n", " ") + f"  (logged ${led.summary()['total']['cost_usd']:.5f})\n")
    print(line)
    return 0 if ok == PROBE_N else 1


def latest(run: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """arm -> task -> latest record (the runner's resume rule)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        out[p.stem] = C.load_latest([p])[0]
    return out


def overload_error(t: Mapping[str, Any]) -> bool:
    e = str(t.get("error") or "").lower()
    return bool(e) and any(k in e for k in OVERLOAD)


def _kill(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        time.sleep(10)


def cmd_watch(a: argparse.Namespace) -> int:
    """The $ cap (explore_e7_hard's accounting over the smoke and the evaluation run, plus --prior) -> STOP file; and,
    unless --no-pause, a pause on a new episode that errored with 503 / overload -> <run>/PAUSED_overload.json. The
    runner's process group is stopped either way; every record is kept (the runner resumes by task id)."""
    runs = Path(a.runs)
    stop = runs / X["stop"]
    cap = X["cap"] if a.cap is None else a.cap
    seen = {name: {(arm, k) for arm, d in latest(runs / name).items() for k, t in d.items() if t.get("error")}
            for name in LIVE_RUNS if (runs / name).is_dir()}
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{_now()} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = E7.total_spend(runs, a.prior, exp=EXP)
        if s["total_with_prior"] >= cap:
            stop.write_text(json.dumps({"at": _now(), "spend": s, "cap": cap}, indent=1))
            _kill(a.pid)
            print(f"CAP REACHED: ${s['total_with_prior']:.2f} >= ${cap:.2f}; runner stopped", flush=True)
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
        print(f"{_now()} spend ${s['total_with_prior']:.3f} (here ${s['total']:.3f}; {s['records']} records)", flush=True)
        time.sleep(a.every)


def cmd_errored(a: argparse.Namespace) -> int:
    n = sum(1 for d in latest(Path(a.run)).values() for t in d.values() if t.get("error"))
    print(n)
    return 0


def cmd_project(a: argparse.Namespace) -> int:
    p = E7.projection(Path(a.runs), a.prior, exp=EXP)
    out = Path(a.out) if a.out else X["box"] / "projection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(p, indent=1))
    return 0


def cmd_spend(a: argparse.Namespace) -> int:
    s = E7.total_spend(Path(a.runs), a.prior, exp=EXP)
    if a.short:
        print(f"spend ${s['total']:.2f} here + ${a.prior:.2f} prior = ${s['total_with_prior']:.2f}")
    else:
        print(json.dumps(s, indent=1))
    return 0


# ================================================================================================ readout (local)
def reading(ni: Mapping[str, Mapping[str, Any]], n_full: int = N_FULL) -> dict[str, Any] | None:
    """The registered reading (k): non-inferior to LLM-only at 3 points on the 200 tasks iff hard_ni's row says so on
    all 200 paired tasks; with fewer tasks the row is descriptive."""
    r = ni.get(EXP)
    if not r:
        return None
    full = int(r["n"]) == n_full
    verdict = (f"non-inferior to LLM-only at {hard_ni.MARGIN * 100:.0f} points on the {n_full} tasks" if full and r["ni"] == "yes"
               else f"non-inferiority at {hard_ni.MARGIN * 100:.0f} points not shown on the {n_full} tasks" if full
               else f"descriptive only: {r['n']} of {n_full} tasks paired, no NI reading")
    return {**r, "full": full, "verdict": verdict}


def cal_keys(f: Mapping[str, Any]) -> dict[str, str]:
    """Keys of the refit (Deviations "E19: the fitted thresholds"): tau-hat per decision point and, on the calibration
    states, the coverage and accuracies at tau-hat for all tool choices, the cancel-vs-ship subset and the arguments."""
    v: dict[str, str] = {f"{EXP}/cal/ntasks": str(f.get("n_tasks"))}
    names = (("coverage", "coverage"), ("s1_acc_all", "s1accall"), ("s1_acc_taken", "s1acctaken"),
             ("llm_acc_all", "llmaccall"))
    for dp in ("tool_select", "tool_params"):
        r = f.get(dp) or {}
        if not r:
            continue
        k = f"{EXP}/cal/{dp.replace('_', '')}"
        v[f"{k}/tau"] = "always defer" if r.get("always_defer") else f"{float(r['tau_hat']):.5f}"
        v[f"{k}/npaired"] = str(r.get("n_paired"))
        for kk, name in names:
            if MM._finite((r.get("at_tau") or {}).get(kk)):
                v[f"{k}/{name}"] = f"{100 * r['at_tau'][kk]:.1f}"
        if dp == "tool_select" and r.get("op_at_tau"):
            op = r["op_at_tau"]
            v[f"{EXP}/cal/op/n"] = str(op.get("n"))
            for kk, name in names:
                if MM._finite(op.get(kk)):
                    v[f"{EXP}/cal/op/{name}"] = f"{100 * op[kk]:.1f}"
    return v


def cmd_summarize(a: argparse.Namespace) -> int:
    res = E7.summarize(Path(a.runs), a.n_boot, a.seed, Path(a.window) if a.window else None, exp=EXP)
    OUT.mkdir(parents=True, exist_ok=True)
    keys = E7.to_tex_keys(res)
    fits = Path(X["box"]) / "cal" / "fits_random.json"
    if fits.exists():
        res["calibration_fit"] = json.loads(fits.read_text())
        keys.update(cal_keys(res["calibration_fit"]))
    E7.write_tex(keys, Path(a.tex) if a.tex else X["tex"], X["title"])
    ni = hard_ni.compute(hard_ni.read_keys(ROOT / "paper/generated"))     # every post-Stage-1 row, this one included
    (ROOT / "paper/generated/hard_ni.tex").write_text(hard_ni.to_tex(ni), encoding="utf-8")
    rd = reading(ni)
    res["reading"] = rd
    md = E7.to_markdown(res)
    if rd:
        md += ("\n\n## Registered reading (E19 (k))\n\n"
               f"- paired n = {rd['n']}, wins {rd['wins']}, losses {rd['losses']}, Δ {rd['delta']:+.1f} points (Tango 95 % "
               f"[{rd['lo']:+.1f}, {rd['hi']:+.1f}]); certified ordered p = {rd['p']:.3f} at the 3-point margin "
               f"(A6.21 rule bb_ordered): **{rd['verdict']}**")
    (OUT / "summary.json").write_text(json.dumps(MM._clean(res), indent=1, default=MM._json_default) + "\n")
    (OUT / "summary.md").write_text(md + "\n")
    print(md)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(E7.RUNS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--log", default=str(OUT / "api_probe_log.txt"))
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--cap", type=float, default=None, help="default: the registered cap")
    w.add_argument("--prior", type=float, default=0.0)
    w.add_argument("--every", type=float, default=30.0)
    w.add_argument("--no-pause", action="store_true", help="the re-run of errored episodes: cap only")
    er = sub.add_parser("errored")
    er.add_argument("--run", required=True)
    pj = sub.add_parser("project")
    pj.add_argument("--prior", type=float, default=0.0)
    pj.add_argument("--out", default=None)
    sp = sub.add_parser("spend")
    sp.add_argument("--prior", type=float, default=0.0)
    sp.add_argument("--short", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--n-boot", type=int, default=MM.N_BOOT)
    s.add_argument("--seed", type=int, default=MM.SEED)
    s.add_argument("--window", default=None)
    s.add_argument("--tex", default=None)
    a = ap.parse_args(argv)
    return {"probe": cmd_probe, "watch": cmd_watch, "errored": cmd_errored, "project": cmd_project, "spend": cmd_spend,
            "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
