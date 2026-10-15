#!/usr/bin/env python3
r"""E21b (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E21b: the deployed confidence-gated
cascades on DecisionGym-Shift"; review round 23, P0-3): the final harness (tool + argument cascade at the in-domain
thresholds, no refit) on E21's 200 shifted L3 tasks, for the frozen 4B, E18 (the 0.8B head with the learned defer head)
and E12 (the released AgentDecision-4B, at in-domain thresholds fitted on the frozen 4B's calibration arm), paired with
E21's logged LLM-only row. No test; outside every Holm table; revises nothing.

    python scripts/explore_e21b_shift_cascade.py cal-build                       # local: the calibration questions
    python scripts/explore_e21b_shift_cascade.py cal-fit --answers FILE          # local: derive, fit, E12's profile
    python scripts/explore_e21b_shift_cascade.py probe | spend | errored --run R | gate --smoke R
    python scripts/explore_e21b_shift_cascade.py watch --pid PID --run RUN [--no-pause]   # cap + overload pause
    python scripts/explore_e21b_shift_cascade.py archive                         # slim per-task copies of the runs
    python scripts/explore_e21b_shift_cascade.py summarize                       # summary.{md,json} + the TeX keys

The host side runs through scripts/aws/e21b_shift_cascade.sh."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUNS = ROOT / "runs"
OUT = ROOT / "results/explore/e21b_shift_cascade"
BOX = OUT / "box"
CAL_OUT = OUT / "cal"
TEX = ROOT / "paper/generated/e21b_shift_cascade.tex"
CFG = ROOT / "configs/explore/e21b_shift_cascade"
PROFILE_E12 = ROOT / "configs/harness/explore_e21b_e12_final.yaml"
FINAL_PROFILE = ROOT / "configs/harness/jevagent_final.yaml"
PROBE_LOG = BOX / "api_probe_log.txt"
LABEL = ("exploratory (E21b), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E21b: the deployed confidence-gated "
         "cascades on DecisionGym-Shift'; outside every Holm table")
PREFIX = "e21b_"
STOP = RUNS / "e21b_STOP.json"
# slug -> (run directory, arm name)
ARMS = {"rl": ("e21b_rl", "rl_final"), "e18": ("e21b_e18", "e18_final"), "e12": ("e21b_e12", "e12_final"),
        "e12_in": ("e21b_e12_in", "e12_final_in")}
SHIFT_ARMS = ("rl", "e18", "e12")
NAMES = {"rl": "Frozen 4B (final harness)", "e18": "0.8B head + defer head (E18)", "e12": "AgentDecision-4B (E12)",
         "e12_in": "AgentDecision-4B (E12), in-domain", "llm_only": "Agent LLM alone (Flash)"}
# the calibration arm of the frozen 4B (Flash, seeds 5000-5149) and its rebuilt questions (E14's / E20's sha256)
CAL_SRC = RUNS / "e2a_jevgym_shadow_cal_rl"
CAL_ARM = "rl_shadow_cal"
CAL_WORK = RUNS / "e21b_cal"
CAL_REQ = CAL_WORK / "cal_requests.jsonl.gz"
CAL_REQ_SHA = "d46701e8e9bdc69e6a9e047227eabb1e52578ddc527342dda10b33f7dc5fff5e"
CAL_NAME = "e21bcal_e12"
DPS = ("tool_select", "tool_params")
# the reference: E21's logged LLM-only row on the same 200 tasks (not re-run), and E21's REPLACE rows of the same models
E21_LIVE = ROOT / "results/explore/e21_shift/live"
E21_REF_RUN = RUNS / "e21_live_local" / "llm_only.jsonl"
E21_REF_SLIM = (E21_LIVE / "e21_live_local.json", "llm_only")
E21_REPLACE = {"rl": (E21_LIVE / "e21_live_rl.json", "rl_ts_replace_f"),
               "e12": (E21_LIVE / "e21_live_e12.json", "e12_ts_replace_f"),
               "e18": (E21_LIVE / "e21_live_e14a.json", "e14a_ts_replace_f")}
# in-domain final-harness rows of the same models (coverage and success beside the shifted rows)
IN_DOMAIN = {"rl": (RUNS / "harness_final_dedicated", "harness_final"), "e18": (RUNS / "e18_live", "e18_final"),
             "e12": (RUNS / "e21b_e12_in", "e12_final_in")}
IN_DOMAIN_REF = (RUNS / "harness_final_dedicated", "llm_only")
L3_SEED0, L3_N = 330100, 200
CAP_USD = 35.0
GATE_SAFETY = 1.2
TARIFF_USD_H = 1.624              # the in-domain final-harness / E18 convention (g5.4xlarge on demand, one A10G)
HOST_USD_H = {"g5.2xlarge": 1.212, "g5.4xlarge": 1.624, "g6e.2xlarge": 2.24208}
NI_MARGIN, NI_ALPHA = 0.03, 0.05
N_BOOT, BOOT_SEED = 10_000, 0
OVERLOAD = ("503", "overload", "high demand", "unavailable", "resource_exhausted", "resource exhausted", "429")
ASK_TOOLS = ("request_clarification", "ask_user")
FLASH = {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0, "reasoning_effort": "minimal"}
PROBE_N = 6


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p: Path) -> list[dict[str, Any]]:
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


def git_tracked(p: Path) -> bool:
    try:
        return subprocess.run(["git", "ls-files", "--error-unmatch", str(p.relative_to(ROOT))], cwd=ROOT,
                              capture_output=True).returncode == 0
    except ValueError:
        return False


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


# ================================================================================================ calibration (E12)
def cmd_cal_build(a: argparse.Namespace) -> int:
    import e5_cal_replay as CR

    meta = CR.build(CAL_SRC, CAL_ARM, CAL_REQ)
    ok = meta["sha256"] == CAL_REQ_SHA
    print(json.dumps({**meta, "registered_sha256": CAL_REQ_SHA, "ok": ok}, indent=1))
    return 0 if ok else 1


def cal_coverage(path: Path, fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage and accepted errors at the fitted thresholds on the derived calibration arm (E17's cal_cov)."""
    from jevharness.bench.threshold_theory import decision_pairs, load_trajectories, paired

    trajs = list(load_trajectories(path))
    r: dict[str, Any] = {}
    for dp in DPS:
        t = (fits.get(dp) or {}).get("tau_hat")
        tau = float(t) if isinstance(t, (int, float)) and math.isfinite(t) and t <= 1.0 else float("inf")
        ps = paired(decision_pairs(trajs, dp))
        if not ps:
            continue
        acc = [x for x in ps if x["c"] is not None and x["c"] >= tau and x["acceptable"]]
        n = len(ps)
        r[dp] = {"tau": tau if math.isfinite(tau) else None, "always_defer": not math.isfinite(tau), "n_paired": n,
                 "accepted": len(acc), "coverage": len(acc) / n,
                 "accepted_errors": sum(1 for x in acc if not x["s1_ok"]),
                 "s1_acc_all": sum(float(x["s1_ok"]) for x in ps) / n, "s2_acc_all": sum(float(x["s2_ok"]) for x in ps) / n}
    return r


def write_e12_profile(fits: Mapping[str, Mapping[str, Any]], fit_paths: Mapping[str, Path]) -> dict[str, Any]:
    """jevagent_final.yaml + backends.e12 with the fitted thresholds (E20's rule: an 'always defer' point gets
    always_defer; if the tool choice always defers, registration (b) 3: the frozen 4B's thresholds, labelled)."""
    import yaml

    prof = yaml.safe_load(FINAL_PROFILE.read_text())
    prof["name"] = "explore_e21b_e12_final"
    prof["status"] = "development"
    sec = prof["system2"]["gemini-3.5-flash"]
    blk: dict[str, Any] = {}
    defer = {dp: not (isinstance(fits[dp].get("tau_hat"), (int, float)) and math.isfinite(fits[dp]["tau_hat"])
                      and fits[dp]["tau_hat"] <= 1.0) for dp in DPS}
    if defer["tool_select"]:
        fb = sec["backends"]["jevagent_rl"]
        blk = {dp: {"tau": float(fb[dp]["tau"])} for dp in DPS}
        rule = ("E21b (b) 3 fallback: the fit always defers at the tool choice, so the row runs at the frozen 4B's "
                "thresholds of jevagent_final.yaml (labelled so)")
    else:
        for dp in DPS:
            blk[dp] = {"tau": 1.0, "always_defer": True} if defer[dp] else {"tau": float(fits[dp]["tau_hat"])}
        rule = "E21b (b) 3: E12's in-domain fit by the registered command on rl_shadow_cal (seeds 5000-5149)"
    blk["fitted_from"] = {dp: {"path": str(fit_paths[dp].relative_to(ROOT)), "sha256": sha256_file(fit_paths[dp])}
                          for dp in DPS}
    blk["notes"] = rule
    sec["backends"]["e12"] = blk
    prof["description"] = ("Exploratory row E21b, arm e12_final (PREREG_A7_STAGE2 Deviations 2026-10-08 'E21b' (b) 3): "
                           "a copy of configs/harness/jevagent_final.yaml (tool + argument cascade, reflection off, "
                           "batch_s1) with backends.e12 = E12's in-domain thresholds. " + rule + ".")
    head = ("# Exploratory row E21b's profile for arm e12_final (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 \"E21b: the\n"
            "# deployed confidence-gated cascades on DecisionGym-Shift\" (b) 3); written by `scripts/explore_e21b_shift_cascade.py\n"
            "# cal-fit` from results/explore/e21b_shift_cascade/cal/fits.json before any E21b episode of this arm: a copy of\n"
            "# configs/harness/jevagent_final.yaml plus backends.e12 (E12's in-domain fitted thresholds; no shifted state used).\n")
    PROFILE_E12.write_text(head + yaml.safe_dump(prof, sort_keys=False, width=118))
    return {"backend": blk, "rule": rule}


def cmd_cal_fit(a: argparse.Namespace) -> int:
    import e5_cal_replay as CR

    ans = Path(a.answers)
    if not ans.exists():
        raise SystemExit(f"no {ans}")
    if sha256_file(CAL_REQ) != CAL_REQ_SHA:
        raise SystemExit("the calibration questions differ from the registered file")
    answers = jl(ans)
    keys = {q["key"] for q in jl(CAL_REQ)}
    got = {r["key"] for r in answers if r.get("error") is None}
    d = CR.derive(CAL_SRC, CAL_ARM, ans, CAL_WORK, CAL_NAME, model="agentdecision-4b-mixed")
    CAL_OUT.mkdir(parents=True, exist_ok=True)
    f = CR.fit(CAL_WORK, CAL_NAME, CAL_OUT)
    paths = {dp: Path(f[dp]["path"]) for dp in DPS}
    for dp in DPS:   # the fit files as committed records (relative paths)
        f[dp]["path"] = str(paths[dp].relative_to(ROOT))
    cov = cal_coverage(CAL_WORK / f"{CAL_NAME}.jsonl", f)
    prof = write_e12_profile(f, paths)
    # the answers as a committed record (gzip, fixed mtime)
    rec = CAL_OUT / "answers_e12_cal.jsonl.gz"
    with open(rec, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(ans.read_bytes())
    res = {"label": LABEL, "at": _now(), "questions": len(keys), "answered": len(got & keys),
           "errors": sum(1 for r in answers if r.get("error")), "derive": d, "fits": f, "coverage_at_tau": cov,
           "profile": {"path": str(PROFILE_E12.relative_to(ROOT)), "sha256": sha256_file(PROFILE_E12), **prof},
           "requests_sha256": CAL_REQ_SHA, "answers": {"path": str(rec.relative_to(ROOT)),
                                                       "sha256_raw": sha256_file(ans)}}
    (CAL_OUT / "fits.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("questions", "answered", "errors", "fits", "coverage_at_tau")}, indent=1,
                     default=str))
    return 0


# ================================================================================================ live operations
def cmd_probe(a: argparse.Namespace) -> int:
    """One round of E14's rule: six one-word Flash calls on the direct API through the harness client."""
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
        p.write_text("# E21b: direct Gemini API probes (E14's pause-and-resume rule, registration E21b (e)). One round = "
                     "six one-word Gemini 3.5 Flash calls; start / resume after two consecutive 6/6 rounds.\n")
    with open(p, "a") as f:
        f.write(line.replace("\n", " ") + f"  (logged ${led.summary()['total']['cost_usd']:.5f})\n")
    print(line)
    return 0 if ok == PROBE_N else 1


_PROBE_COST = re.compile(r"\(logged \$([0-9.]+)\)")


def probe_spend(log: Path) -> float:
    return float(sum(float(m.group(1)) for m in _PROBE_COST.finditer(Path(log).read_text()))) if Path(log).exists() else 0.0


def _comp(t: Mapping[str, Any], comp: str) -> float:
    return float(((t.get("usage") or {}).get(comp) or {}).get("cost_usd") or 0.0)


def _records(run: Path) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        for t in jl(p):
            if isinstance(t, dict) and "task_id" in t:
                out.append((p.stem, t))
    return out


def e21b_runs(runs: Path) -> list[Path]:
    return sorted(p for p in Path(runs).glob(f"{PREFIX}*") if p.is_dir() and p.name != CAL_WORK.name)


def total_spend(runs: Path, log: Path) -> dict[str, Any]:
    """Every record of every E21b run (errored attempts and re-runs included: all were paid) plus the probes; s2 = the
    agent LLM (direct Gemini), s1 = the decision model (served: $0)."""
    per = {}
    for r in e21b_runs(runs):
        recs = _records(r)
        per[r.name] = {"s2": sum(_comp(t, "s2") for _, t in recs), "s1": sum(_comp(t, "s1") for _, t in recs),
                       "total": sum(_comp(t, "total") for _, t in recs), "records": len(recs)}
    pr = probe_spend(log)
    gem = sum(v["s2"] for v in per.values()) + pr
    return {"by_run": per, "probes": pr, "gemini": gem, "s1": sum(v["s1"] for v in per.values()),
            "records": sum(v["records"] for v in per.values())}


def cmd_spend(a: argparse.Namespace) -> int:
    s = total_spend(Path(a.runs), Path(a.log))
    print(f"E21b: Gemini ${s['gemini']:.3f} (probes ${s['probes']:.4f}), System-1 ${s['s1']:.3f} in {s['records']} records"
          if a.short else json.dumps(s, indent=1))
    return 0


def overload_error(t: Mapping[str, Any]) -> bool:
    e = str(t.get("error") or "").lower()
    return bool(e) and any(k in e for k in OVERLOAD)


def latest(path: Path) -> dict[str, dict[str, Any]]:
    """task id -> the deciding record (the latest without an error, else the latest)."""
    out: dict[str, dict[str, Any]] = {}
    for t in jl(path):
        if not isinstance(t, dict) or "task_id" not in t:
            continue
        k = str(t["task_id"])
        prev = out.get(k)
        if prev is None or not t.get("error") or prev.get("error"):
            out[k] = t
    return out


def errored_tasks(run: Path) -> int:
    return sum(sum(1 for t in latest(p).values() if t.get("error"))
               for p in sorted(Path(run).glob("*.jsonl")) if not p.name.startswith("schedule"))


def cmd_errored(a: argparse.Namespace) -> int:
    print(errored_tasks(Path(a.run)))
    return 0


def _kill(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        time.sleep(10)


def cmd_watch(a: argparse.Namespace) -> int:
    """The Gemini cap over E21b's runs and probes (-> runs/e21b_STOP.json) and, unless --no-pause, a pause when a new
    record of the watched run errored with 503 / overload (-> <run>/PAUSED_overload.json); records are kept."""
    runs = Path(a.runs)
    run = runs / a.run
    base = sum(1 for _, t in _records(run) if overload_error(t)) if run.is_dir() else 0
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{_now()} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(runs, Path(a.log))
        if s["gemini"] >= a.cap:
            (runs / STOP.name).write_text(json.dumps({"at": _now(), "spend": s, "cap": a.cap}, indent=1))
            _kill(a.pid)
            print(f"{_now()} CAP REACHED: Gemini ${s['gemini']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        if not a.no_pause and run.is_dir():
            ov = [(arm, t) for arm, t in _records(run) if overload_error(t)]
            if len(ov) > base:
                (run / "PAUSED_overload.json").write_text(json.dumps(
                    {"at": _now(), "episodes": [{"arm": arm, "task": t["task_id"], "error": str(t.get("error"))[:300]}
                                                for arm, t in ov[base:]]}, indent=1))
                _kill(a.pid)
                print(f"{_now()} PAUSED: {len(ov) - base} overload error(s) in {a.run}; runner stopped", flush=True)
                return 2
        print(f"{_now()} Gemini ${s['gemini']:.3f} ({s['records']} records)", flush=True)
        time.sleep(a.every)


def gate(smoke: Path, runs: Path, log: Path, n: int = L3_N, cap: float = CAP_USD) -> dict[str, Any]:
    """Registration (e): spend so far + 1.2 x the smoke's Gemini $ per episode x n <= cap, else the first k that fit."""
    recs = _records(smoke)
    per = (sum(_comp(t, "s2") for _, t in recs) / len(recs)) if recs else float("inf")
    spent = total_spend(runs, log)["gemini"]
    if not recs or not math.isfinite(per):
        n_fit = 0
    elif per <= 0:
        n_fit = n
    else:
        n_fit = min(n, int(max(0.0, (cap - spent) / (GATE_SAFETY * per))))
    return {"smoke": smoke.name, "smoke_records": len(recs), "smoke_errors": sum(1 for _, t in recs if t.get("error")),
            "gemini_usd_per_episode": per, "n": n, "projection_usd": spent + GATE_SAFETY * per * n if recs else None,
            "spent_usd": spent, "cap_usd": cap, "n_fit": n_fit, "go": n_fit == n, "at": _now()}


def cmd_gate(a: argparse.Namespace) -> int:
    g = gate(Path(a.smoke), Path(a.runs), Path(a.log), a.n, a.cap)
    print(json.dumps(g))
    return 0 if g["n_fit"] > 0 else 1


# ================================================================================================ readout
def is_must_ask(d: Mapping[str, Any]) -> bool:
    at = (d.get("gold_aux") or {}).get("acceptable_tools")
    return isinstance(at, list) and len(at) == 1 and at[0] in ASK_TOOLS


def decision_stats(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per decision point: decisions, taken by System-1 (not deferred), deferred; among those the answer key labels:
    accepted wrong decisions (bench/metrics.is_correct), and the LLM's accuracy on the deferred ones; must-ask states
    (the acceptable set is the ask tool alone) at the tool choice separately."""
    from jevharness.bench import metrics as M

    out: dict[str, Any] = {}
    for dp in DPS:
        ds = [d for t in recs for s in (t.get("steps") or []) for d in (s.get("decisions") or [])
              if d.get("dp") == dp and not d.get("shadow") and d.get("mode") not in (None, "generate", "off")]
        taken = [d for d in ds if not d.get("deferred") and d.get("s1_value") is not None]
        deferred = [d for d in ds if d.get("deferred")]
        tl = [(d, M.is_correct(d, d["s1_value"])) for d in taken]
        tl = [(d, c) for d, c in tl if c is not None]
        dl = [(d, M.is_correct(d, d.get("final_value"))) for d in deferred]
        dl = [(d, c) for d, c in dl if c is not None]
        blk = {"decisions": len(ds), "taken": len(taken), "deferred": len(deferred),
               "coverage": (len(taken) / len(ds)) if ds else None, "taken_labelled": len(tl),
               "accepted_wrong": sum(1 for _, c in tl if not c),
               "accepted_wrong_share": (sum(1 for _, c in tl if not c) / len(tl)) if tl else None,
               "deferred_labelled": len(dl), "deferred_llm_acc": (sum(1 for _, c in dl if c) / len(dl)) if dl else None,
               "mean_conf_taken": (sum(float(d.get("s1_conf") or 0) for d in taken) / len(taken)) if taken else None}
        if dp == "tool_select":
            ma = [d for d in ds if is_must_ask(d)]
            mt = [d for d in ma if not d.get("deferred") and d.get("s1_value") is not None]
            blk["mustask"] = {"decisions": len(ma), "taken": len(mt),
                              "accepted_wrong": sum(1 for d in mt if M.is_correct(d, d["s1_value"]) is False)}
        out[dp] = blk
    return out


def arm_records(path: Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    """(task -> deciding record without error, errored tasks, all tasks)."""
    lat = latest(path)
    ok = {k: t for k, t in lat.items() if not t.get("error")}
    return ok, len(lat) - len(ok), len(lat)


def episodes(recs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    import harness_final as HF

    return {k: HF._episode_live(t) for k, t in recs.items()}


def ref_episodes() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """E21's logged llm_only row: the run file when present (full records), else E21's slim copy."""
    if E21_REF_RUN.exists():
        ok, ne, nt = arm_records(E21_REF_RUN)
        return episodes(ok), {"source": str(E21_REF_RUN.relative_to(ROOT)), "errored": ne, "n_latest": nt,
                              "sha256": sha256_file(E21_REF_RUN), "slim_tracked": git_tracked(E21_REF_SLIM[0])}
    slim = json.loads(E21_REF_SLIM[0].read_text())["arms"][E21_REF_SLIM[1]]
    eps = {k: {"success": float(bool(v["success"])), "cost": float(v["cost"]), "wall_s": float(v["wall_s"]),
               "steps": v.get("steps"), "s2_calls": v.get("s2_calls")} for k, v in slim.items() if not v.get("error")}
    return eps, {"source": str(E21_REF_SLIM[0].relative_to(ROOT)), "errored": sum(1 for v in slim.values() if v.get("error")),
                 "n_latest": len(slim), "slim_tracked": git_tracked(E21_REF_SLIM[0])}


def replace_success(slug: str) -> tuple[dict[str, float], dict[str, Any]] | None:
    p, arm = E21_REPLACE.get(slug, (None, None))
    if p is None or not p.exists():
        return None
    d = json.loads(p.read_text())["arms"].get(arm) or {}
    return ({k: float(bool(v["success"])) for k, v in d.items() if not v.get("error")},
            {"source": str(p.relative_to(ROOT)), "arm": arm, "tracked": git_tracked(p)})


def ni_block(wins: int, losses: int, n: int) -> dict[str, Any]:
    """scripts/hard_ni.py's computation: the A6.21 certified p (rule bb_ordered) and the Tango 95 % interval."""
    from jevharness.bench.metrics import ni_certified_p, tango_interval

    r = ni_certified_p(wins, losses, n, NI_MARGIN)
    ci = tango_interval(n - wins - losses, wins, losses, 0)["ci"]
    p = float(r["p"])
    return {"margin": NI_MARGIN, "alpha": NI_ALPHA, "p": p, "verdict": "non-inferior" if p <= NI_ALPHA else "not shown",
            "tango_ci_pt": [100.0 * ci[0], 100.0 * ci[1]], "rule": "bb_ordered (A6.21), as scripts/hard_ni.py"}


def kind_of(task_id: str) -> str:
    m = re.match(r"jevgym-shift-[a-z]+-(.+?)-\d{6}-", task_id)
    return m.group(1) if m else "?"


NEW_KINDS = ("clarify.ask_twice", "clarify.ask_after_lookup", "notify")


def window_host(box: Path, slug: str) -> dict[str, Any]:
    p = box / f"window_{slug}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def summarize(runs: Path = RUNS, box: Path = BOX) -> dict[str, Any]:
    import numpy as np

    import explore_common as C

    ref, ref_meta = ref_episodes()
    res: dict[str, Any] = {"label": LABEL, "generated": _now(), "git": git_head()[:12], "reference": ref_meta,
                           "reference_n": len(ref), "tariff_usd_h": TARIFF_USD_H, "arms": {}}
    rs = np.array([ref[k]["success"] for k in sorted(ref)])
    res["reference_success"] = float(rs.mean()) if len(rs) else None
    res["reference_api_usd"] = float(np.mean([ref[k]["cost"] for k in ref])) if ref else None
    res["reference_wall_s"] = float(np.mean([ref[k]["wall_s"] for k in ref])) if ref else None
    fits = json.loads((CAL_OUT / "fits.json").read_text()) if (CAL_OUT / "fits.json").exists() else {}
    res["e12_fit"] = {k: fits.get(k) for k in ("fits", "coverage_at_tau", "profile")} if fits else None
    for slug in (*SHIFT_ARMS, "e12_in"):
        rd, arm = ARMS[slug]
        p = runs / rd / f"{arm}.jsonl"
        if not p.exists():
            res["arms"][slug] = {"state": "not run"}
            continue
        recs, nerr, nlat = arm_records(p)
        eps = episodes(recs)
        win = window_host(box, slug)
        host_h = HOST_USD_H.get(str(win.get("instance_type") or ""), float("nan"))
        if slug == "e12_in":   # paired with the dedicated in-domain llm_only row
            rf = IN_DOMAIN_REF[0] / f"{IN_DOMAIN_REF[1]}.jsonl"
            r_ok, _, _ = arm_records(rf) if rf.exists() else ({}, 0, 0)
            r_eps = episodes(r_ok)
        else:
            r_eps = ref
        common = sorted(set(eps) & set(r_eps))
        a = {k: np.array([eps[t][k] for t in common], float) for k in ("success", "cost", "wall_s", "steps", "s2_calls",
                                                                        "s1_requests")}
        b = {k: np.array([r_eps[t][k] for t in common], float) for k in ("success", "cost", "wall_s")}
        q = TARIFF_USD_H / 3600 * a["wall_s"]
        qh = host_h / 3600 * a["wall_s"]
        tot = {x: sum(eps[t][x] for t in common) for x in ("tool_decisions", "tool_by_s1", "arg_decisions", "arg_by_s1")}
        nc = C.newcombe(a["success"], b["success"])
        wins, losses = nc["cells"]["only_arm"], nc["cells"]["only_ref"]
        blk: dict[str, Any] = {
            "state": "ok", "n": len(common), "n_latest": nlat, "errored": nerr, "window": win,
            "success": float(a["success"].mean()) if common else None,
            "ref_success": float(b["success"].mean()) if common else None,
            "delta": nc["diff_pt"] / 100, "newcombe_pt": nc["ci_pt"], "wins": wins, "losses": losses,
            "delta_boot": C.paired_boot(a["success"], b["success"]),
            "ni": ni_block(wins, losses, len(common)) if common else None,
            "api_usd": float(a["cost"].mean()), "q_usd": float(q.mean()), "usd_incl_gpu": float((a["cost"] + q).mean()),
            "ref_usd": float(b["cost"].mean()),
            "saving_incl_gpu": C.paired_boot(a["cost"] + q, b["cost"], ratio=True),
            "saving_api": C.paired_boot(a["cost"], b["cost"], ratio=True),
            "saving_host_tariff": (C.paired_boot(a["cost"] + qh, b["cost"], ratio=True) if math.isfinite(host_h) else None),
            "host_usd_h": host_h if math.isfinite(host_h) else None,
            "wall_s_mean": float(a["wall_s"].mean()), "wall_s_median": float(np.median(a["wall_s"])),
            "ref_wall_s_mean": float(b["wall_s"].mean()), "ref_wall_s_median": float(np.median(b["wall_s"])),
            "wall_diff": C.paired_boot(a["wall_s"], b["wall_s"]),
            "steps": float(a["steps"].mean()), "s2_calls": float(a["s2_calls"].mean()),
            "s1_requests": float(a["s1_requests"].mean()),
            "tool_coverage": tot["tool_by_s1"] / tot["tool_decisions"] if tot["tool_decisions"] else None,
            "arg_coverage": tot["arg_by_s1"] / tot["arg_decisions"] if tot["arg_decisions"] else None,
            "decisions": decision_stats([recs[t] for t in common]),
        }
        blk["handed_back"] = (1.0 - blk["tool_coverage"]) if blk["tool_coverage"] is not None else None
        if slug != "e12_in":
            by: dict[str, list[float]] = {}
            for t in common:
                by.setdefault(kind_of(t), []).append(eps[t]["success"])
            blk["by_kind"] = {k: {"n": len(v), "success": float(np.mean(v))} for k, v in sorted(by.items())}
            nk = [eps[t]["success"] for t in common if kind_of(t) in NEW_KINDS]
            gk = [eps[t]["success"] for t in common if kind_of(t) not in NEW_KINDS]
            blk["new_kinds"] = {"n": len(nk), "success": float(np.mean(nk)) if nk else None}
            blk["generator_kinds"] = {"n": len(gk), "success": float(np.mean(gk)) if gk else None}
            rp = replace_success(slug)
            if rp is not None:
                rsucc, rmeta = rp
                cm = sorted(set(common) & set(rsucc))
                if cm:
                    ncr = C.newcombe([eps[t]["success"] for t in cm], [rsucc[t] for t in cm])
                    blk["vs_replace"] = {**rmeta, "n": len(cm), "replace_success": float(np.mean([rsucc[t] for t in cm])),
                                         "delta": ncr["diff_pt"] / 100, "newcombe_pt": ncr["ci_pt"],
                                         "wins": ncr["cells"]["only_arm"], "losses": ncr["cells"]["only_ref"]}
            idp, iarm = IN_DOMAIN.get(slug, (None, None))
            ip = (idp / f"{iarm}.jsonl") if idp is not None else None
            if ip is not None and ip.exists():
                ir, _, _ = arm_records(ip)
                ie = episodes(ir)
                itot = {x: sum(e[x] for e in ie.values()) for x in ("tool_decisions", "tool_by_s1", "arg_decisions",
                                                                     "arg_by_s1")}
                blk["in_domain"] = {"source": str(ip.relative_to(ROOT)), "n": len(ie),
                                    "success": float(np.mean([e["success"] for e in ie.values()])),
                                    "tool_coverage": itot["tool_by_s1"] / itot["tool_decisions"] if itot["tool_decisions"] else None,
                                    "arg_coverage": itot["arg_by_s1"] / itot["arg_decisions"] if itot["arg_decisions"] else None,
                                    "wall_s_mean": float(np.mean([e["wall_s"] for e in ie.values()])),
                                    "decisions": decision_stats(list(ir.values()))}
            elif slug == "e12" and fits:
                cv = fits.get("coverage_at_tau") or {}
                blk["in_domain"] = {"source": "calibration states at tau-hat (no in-domain live cascade row)",
                                    "tool_coverage": (cv.get("tool_select") or {}).get("coverage"),
                                    "arg_coverage": (cv.get("tool_params") or {}).get("coverage")}
        res["arms"][slug] = blk
    sp = total_spend(runs, box / "api_probe_log.txt")
    res["spend"] = sp
    # registered reading (E21b (f)), per shifted arm
    rd: dict[str, Any] = {}
    for slug in SHIFT_ARMS:
        b = res["arms"].get(slug) or {}
        if b.get("state") != "ok":
            continue
        sv = (b.get("saving_incl_gpu") or {}).get("est")
        rd[slug] = {"protects_success": (b.get("ni") or {}).get("verdict") == "non-inferior",
                    "still_saves": sv is not None and sv > 0,
                    "defers_almost_everything": (b.get("tool_coverage") or 0.0) < 0.10}
    res["reading"] = rd
    return res


# ------------------------------------------------------------------------------------------------ output
def _pc(x: Any, d: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"


def _g(d: Any, *ks: Any) -> Any:
    for k in ks:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def to_markdown(r: Mapping[str, Any]) -> str:
    L = [f"# E21b: the deployed cascades on DecisionGym-Shift ({LABEL})", "",
         f"Generated {r['generated']} at git {r['git']} by `python scripts/explore_e21b_shift_cascade.py summarize`. "
         f"Reference: E21's logged LLM-only row ({r['reference']['source']}; n = {r['reference_n']}, success "
         f"{_pc(r['reference_success'])} %, ${r['reference_api_usd']:.4f} and {r['reference_wall_s']:.1f} s per episode; 4 "
         f"workers from the container). Arms: the final harness at the in-domain thresholds, 1 worker, the model alone "
         f"on the host's GPU. Cost includes the GPU tariff ${TARIFF_USD_H}/h × episode wall-clock.", "",
         "| arm | n | err | success | Δ vs LLM-only [Newcombe] | W/L | NI p (3 pt) | $/ep. incl. GPU | saving | wall s (mean / median) | tool cov. | arg cov. | in-domain tool / arg cov. | accepted wrong tool / arg | must-ask taken / wrong | REPLACE (E21) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for slug, b in r["arms"].items():
        if b.get("state") != "ok":
            L.append(f"| {NAMES[slug]} | – | – | {b.get('state')} |" + " |" * 12)
            continue
        dts, dap = b["decisions"]["tool_select"], b["decisions"]["tool_params"]
        idm = b.get("in_domain") or {}
        vr = b.get("vs_replace") or {}
        L.append(f"| {NAMES[slug]} | {b['n']} | {b['errored']} | {_pc(b['success'])} | {_pc(b['delta'], sign=True)} "
                 f"[{b['newcombe_pt'][0]:+.1f}, {b['newcombe_pt'][1]:+.1f}] | {b['wins']} / {b['losses']} | "
                 f"{_g(b, 'ni', 'p'):.3f} ({_g(b, 'ni', 'verdict')}) | {b['usd_incl_gpu']:.4f} | "
                 f"{_pc(_g(b, 'saving_incl_gpu', 'est'))} % | {b['wall_s_mean']:.1f} / {b['wall_s_median']:.1f} | "
                 f"{_pc(b['tool_coverage'])} | {_pc(b['arg_coverage'])} | {_pc(idm.get('tool_coverage'))} / "
                 f"{_pc(idm.get('arg_coverage'))} | {dts['accepted_wrong']} of {dts['taken_labelled']} / "
                 f"{dap['accepted_wrong']} of {dap['taken_labelled']} | {dts['mustask']['taken']} of "
                 f"{dts['mustask']['decisions']} / {dts['mustask']['accepted_wrong']} | "
                 f"{_pc(vr.get('replace_success')) if vr else '–'} |")
    L += ["", "## Per arm", ""]
    for slug, b in r["arms"].items():
        if b.get("state") != "ok":
            continue
        L.append(f"- **{NAMES[slug]}**: success {_pc(b['success'])} vs {_pc(b['ref_success'])} (paired bootstrap "
                 f"{_pc(_g(b, 'delta_boot', 'est'), sign=True)} [{_pc(_g(b, 'delta_boot', 'ci')[0], sign=True)}, "
                 f"{_pc(_g(b, 'delta_boot', 'ci')[1], sign=True)}]); NI {_g(b, 'ni', 'verdict')} (p = {_g(b, 'ni', 'p'):.3f}, "
                 f"Tango [{_g(b, 'ni', 'tango_ci_pt')[0]:+.1f}, {_g(b, 'ni', 'tango_ci_pt')[1]:+.1f}]); API ${b['api_usd']:.4f} + "
                 f"GPU ${b['q_usd']:.4f} vs ${b['ref_usd']:.4f}; saving {_pc(_g(b, 'saving_incl_gpu', 'est'))} % "
                 f"[{_pc(_g(b, 'saving_incl_gpu', 'ci')[0])}, {_pc(_g(b, 'saving_incl_gpu', 'ci')[1])}] (API only "
                 f"{_pc(_g(b, 'saving_api', 'est'))} %; host tariff {_pc(_g(b, 'saving_host_tariff', 'est'))} %); wall "
                 f"{b['wall_s_mean']:.1f} s vs {b['ref_wall_s_mean']:.1f} s; steps {b['steps']:.1f}, LLM calls "
                 f"{b['s2_calls']:.1f}, System-1 requests {b['s1_requests']:.1f}; handed back {_pc(b['handed_back'])} % of "
                 f"the tool choices; deferred tool choices right by the LLM {_pc(_g(b, 'decisions', 'tool_select', 'deferred_llm_acc'))} %.")
        if b.get("by_kind"):
            L.append("  by kind: " + ", ".join(f"{k} {_pc(v['success'], 0)} ({v['n']})" for k, v in b["by_kind"].items()))
        if b.get("vs_replace"):
            v = b["vs_replace"]
            L.append(f"  vs E21's REPLACE row ({v['arm']}, tracked {v['tracked']}): {_pc(v['replace_success'])} → cascade "
                     f"{_pc(v['delta'], sign=True)} [{v['newcombe_pt'][0]:+.1f}, {v['newcombe_pt'][1]:+.1f}], W/L {v['wins']} / {v['losses']}")
        if b.get("in_domain"):
            i = b["in_domain"]
            L.append(f"  in-domain ({i['source']}): success {_pc(i.get('success'))}, tool / arg coverage "
                     f"{_pc(i.get('tool_coverage'))} / {_pc(i.get('arg_coverage'))}")
    L += ["", "## Registered reading (E21b (f))", ""]
    for slug, v in (r.get("reading") or {}).items():
        L.append(f"- {NAMES[slug]}: protects success (NI at 3 points) {v['protects_success']}; still saves "
                 f"{v['still_saves']}; defers almost everything {v['defers_almost_everything']}")
    if r.get("e12_fit"):
        L += ["", "## E12's in-domain fit", "", "```", json.dumps(r["e12_fit"], indent=1, default=str)[:3000], "```"]
    s = r["spend"]
    L += ["", f"Spend: Gemini ${s['gemini']:.2f} (probes ${s['probes']:.4f}) in {s['records']} records; OpenRouter, Jev "
               "and OpenAI $0.", ""]
    return "\n".join(L)


def _k(v: Any, d: int = 1, pct: bool = True, sign: bool = False) -> str | None:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    x = 100 * v if pct else v
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str | None] = {"n_ref": str(r["reference_n"]), "ref/success": _k(r["reference_success"]),
                                "ref/wall": _k(r["reference_wall_s"], 1, pct=False),
                                "ref/usd_k": _k(1000 * r["reference_api_usd"], 1, pct=False) if r["reference_api_usd"] else None,
                                "spend/gemini": _k(r["spend"]["gemini"], 2, pct=False), "tariff": f"{TARIFF_USD_H:.3f}"}
    for slug, b in r["arms"].items():
        if b.get("state") != "ok":
            continue
        dts, dap = b["decisions"]["tool_select"], b["decisions"]["tool_params"]
        p = slug
        k.update({f"{p}/n": str(b["n"]), f"{p}/errored": str(b["errored"]), f"{p}/success": _k(b["success"]),
                  f"{p}/ref_success": _k(b["ref_success"]),
                  f"{p}/delta": _k(b["delta"], sign=True), f"{p}/delta_ci/lo": _k(b["newcombe_pt"][0] / 100, sign=True),
                  f"{p}/delta_ci/hi": _k(b["newcombe_pt"][1] / 100, sign=True), f"{p}/wins": str(b["wins"]),
                  f"{p}/losses": str(b["losses"]),
                  f"{p}/ni/p": _k(_g(b, "ni", "p"), 3, pct=False), f"{p}/ni/verdict": _g(b, "ni", "verdict"),
                  f"{p}/ni/lo": _k(_g(b, "ni", "tango_ci_pt")[0] / 100, sign=True) if b.get("ni") else None,
                  f"{p}/ni/hi": _k(_g(b, "ni", "tango_ci_pt")[1] / 100, sign=True) if b.get("ni") else None,
                  f"{p}/usd_k": _k(1000 * b["usd_incl_gpu"], 1, pct=False), f"{p}/api_k": _k(1000 * b["api_usd"], 1, pct=False),
                  f"{p}/saving": _k(_g(b, "saving_incl_gpu", "est")), f"{p}/saving_api": _k(_g(b, "saving_api", "est")),
                  f"{p}/saving_ci/lo": _k(_g(b, "saving_incl_gpu", "ci")[0]),
                  f"{p}/saving_ci/hi": _k(_g(b, "saving_incl_gpu", "ci")[1]),
                  f"{p}/wall": _k(b["wall_s_mean"], 1, pct=False), f"{p}/wall_median": _k(b["wall_s_median"], 1, pct=False),
                  f"{p}/ref_wall": _k(b["ref_wall_s_mean"], 1, pct=False),
                  f"{p}/toolcov": _k(b["tool_coverage"]), f"{p}/argcov": _k(b["arg_coverage"]),
                  f"{p}/handed_back": _k(b["handed_back"]),
                  f"{p}/acc_wrong_tool": str(dts["accepted_wrong"]), f"{p}/taken_tool": str(dts["taken_labelled"]),
                  f"{p}/acc_wrong_arg": str(dap["accepted_wrong"]), f"{p}/taken_arg": str(dap["taken_labelled"]),
                  f"{p}/mustask/n": str(dts["mustask"]["decisions"]), f"{p}/mustask/taken": str(dts["mustask"]["taken"]),
                  f"{p}/mustask/wrong": str(dts["mustask"]["accepted_wrong"]),
                  f"{p}/deferred_llm_acc": _k(dts.get("deferred_llm_acc")),
                  f"{p}/steps": _k(b["steps"], 1, pct=False), f"{p}/llm_calls": _k(b["s2_calls"], 1, pct=False)})
        if b.get("new_kinds"):
            k[f"{p}/newkinds"] = _k(b["new_kinds"]["success"])
            k[f"{p}/genkinds"] = _k(b["generator_kinds"]["success"])
        if b.get("vs_replace"):
            v = b["vs_replace"]
            k.update({f"{p}/replace/success": _k(v["replace_success"]), f"{p}/replace/delta": _k(v["delta"], sign=True)})
        if b.get("in_domain"):
            i = b["in_domain"]
            k.update({f"{p}/in/toolcov": _k(i.get("tool_coverage")), f"{p}/in/argcov": _k(i.get("arg_coverage")),
                      f"{p}/in/success": _k(i.get("success"))})
    fit = r.get("e12_fit") or {}
    for dp, short in (("tool_select", "tool"), ("tool_params", "arg")):
        t = _g(fit, "fits", dp, "tau_hat")
        if isinstance(t, (int, float)) and math.isfinite(t) and t <= 1.0:
            k[f"e12/tau/{short}"] = f"{t:.6f}"
        c = _g(fit, "coverage_at_tau", dp, "coverage")
        if c is not None:
            k[f"e12/cal/{short}cov"] = _k(c)
    return {kk: v for kk, v in k.items() if v is not None}


def to_tex(r: Mapping[str, Any]) -> str:
    lines = ["% AUTO-GENERATED by scripts/explore_e21b_shift_cascade.py -- do not edit by hand.",
             "% E21b: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E21b').",
             f"% generated {r['generated']} at git {r['git']}", r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{e21b/{k}}}{{{v}}}" for k, v in sorted(tex_keys(r).items())]
    return "\n".join(lines + [r"\makeatother", ""])


def cmd_archive(a: argparse.Namespace) -> int:
    """Slim per-task copies of E21b's live runs (results/explore/e21b_shift_cascade/live/<run>.json); the raw runs stay
    in s3://.../e21b_shift_cascade/runs/."""
    out_dir = OUT / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ("success", "cost", "wall_s", "steps", "s2_calls", "s1_requests", "tool_decisions", "tool_by_s1",
            "arg_decisions", "arg_by_s1")
    for slug, (rd, arm) in ARMS.items():
        for run in (rd, f"e21b_smoke_{slug}"):
            p = Path(a.runs) / run / f"{arm}.jsonl"
            if not p.exists():
                continue
            lat = latest(p)
            eps = episodes({k: t for k, t in lat.items() if not t.get("error")})
            slim = {k: {**{x: eps[k][x] for x in keep}, "error": None} if k in eps else
                    {"success": 0.0, "error": str(lat[k].get("error"))[:200]} for k in sorted(lat)}
            rec = {"run": run, "arm": arm, "label": LABEL, "source_sha256": sha256_file(p),
                   "records": sum(1 for _ in jl(p)), "decisions": decision_stats([t for t in lat.values() if not t.get("error")]),
                   "tasks": slim, "written": _now()}
            (out_dir / f"{run}.json").write_text(json.dumps(rec, indent=0, default=str) + "\n")
            print(f"{run}: {len(slim)} tasks")
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    r = summarize(Path(a.runs), Path(a.box))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (OUT / "summary.md").write_text(to_markdown(r))
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(to_tex(r))
    print(to_markdown(r))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("cal-build").set_defaults(fn=cmd_cal_build)
    p = sp.add_parser("cal-fit")
    p.add_argument("--answers", required=True)
    p.set_defaults(fn=cmd_cal_fit)
    p = sp.add_parser("probe")
    p.add_argument("--log", default=str(PROBE_LOG))
    p.set_defaults(fn=cmd_probe)
    p = sp.add_parser("spend")
    p.add_argument("--runs", default=str(RUNS))
    p.add_argument("--log", default=str(PROBE_LOG))
    p.add_argument("--short", action="store_true")
    p.set_defaults(fn=cmd_spend)
    p = sp.add_parser("errored")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_errored)
    p = sp.add_parser("watch")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--runs", default=str(RUNS))
    p.add_argument("--log", default=str(PROBE_LOG))
    p.add_argument("--cap", type=float, default=CAP_USD)
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--no-pause", action="store_true")
    p.set_defaults(fn=cmd_watch)
    p = sp.add_parser("gate")
    p.add_argument("--smoke", required=True)
    p.add_argument("--runs", default=str(RUNS))
    p.add_argument("--log", default=str(PROBE_LOG))
    p.add_argument("--n", type=int, default=L3_N)
    p.add_argument("--cap", type=float, default=CAP_USD)
    p.set_defaults(fn=cmd_gate)
    p = sp.add_parser("archive")
    p.add_argument("--runs", default=str(RUNS))
    p.set_defaults(fn=cmd_archive)
    p = sp.add_parser("summarize")
    p.add_argument("--runs", default=str(RUNS))
    p.add_argument("--box", default=str(BOX))
    p.set_defaults(fn=cmd_summarize)
    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
