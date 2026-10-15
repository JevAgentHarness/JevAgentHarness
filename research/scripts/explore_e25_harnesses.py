#!/usr/bin/env python3
"""E25 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09, "E25"): per-step routing by the decision model in two more
open-source agent harnesses, run through Harbor (exploratory, descriptive).

  harness tb   pi 0.99.2 on Terminal-Bench 2.0           (adapters/pi/run_tb2.sh, arms baseline / jev_router / always_cheap)
  harness swe  SWE-agent v1.1.0 on SWE-bench Verified    (adapters/swe_agent/run_swe.sh, the same arms)

Subcommands (no model call anywhere in this script):

  select  --registry <harbor registry.json>   the frozen task lists -> configs/explore/e25_harnesses/
  spend   <run dir> ...                       logged spend of finished trials (LLM + decision model), JSON on stdout
  gate    --harness tb|swe --smoke <run dir> [--spent X] --out gate.json
                                              the registered smoke checks and the projection (E25 (e)); exit 1 = no-go
  readout [--tb <run dir>] [--swe <run dir>]  summary.{md,json} + paper/generated/e25_harnesses.tex (E25 (f))

A run dir holds one Harbor job per arm (<run dir>/<arm>/<trial>/result.json). Per trial: success = reward 1; cost =
the harness's LLM cost (Harbor's agent_result.cost_usd: pi's per-message costs; SWE-agent's trajectory cost through
JevSweAgent) + the decision model's (logged input tokens x $0.042 per M); wall = Harbor's agent-execution seconds;
taken share = the router arm's LLM requests that went to the fast LLM (pi: the physical model of each assistant message
in pi's session log; SWE-agent: jev-calls.jsonl). An errored trial (any Harbor exception but AgentTimeoutError, after
Harbor's retries) is a failure in the full set (primary) and leaves the pair in the complete case.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench.metrics import ni_certified_p, tango_interval  # noqa: E402

CFG = ROOT / "configs/explore/e25_harnesses"
OUT_DIR = ROOT / "results/explore/e25_harnesses"
TEX = ROOT / "paper/generated/e25_harnesses.tex"
SALT = {"tb": "jev-explore-E25-tb2-2026-10-09", "swe": "jev-explore-E25-swe-agent-2026-10-09"}
HARNESS = {
    "tb": {"label": "pi on Terminal-Bench 2.0", "harness": "pi 0.99.2", "bench": "Terminal-Bench 2.0",
           "cap": 140.0, "n_full": 89},
    "swe": {"label": "SWE-agent on SWE-bench Verified", "harness": "SWE-agent v1.1.0", "bench": "SWE-bench Verified",
            "cap": 100.0, "n_full": 100},
}
ARMS = ("baseline", "jev_router", "always_cheap")
MAIN_ARMS = ("baseline", "jev_router")
# E25 (e) 5 as amended (Deviations 2026-10-09 ~01:35 UTC, "E25: scale after the smokes"): per arm, the projection's $ per
# trial is max(SMOKE_MULT x the smoke's mean, FLOOR); the registered priors were $0.60 / $0.60 / $0.15
FLOOR = {"baseline": 0.30, "jev_router": 0.30, "always_cheap": 0.08}
SMOKE_MULT = 3.0
FACTOR = 1.2
N_MIN = 40
JEV_PRICE_PER_M = 0.042
MARGIN = 0.03
ALPHA = 0.05
N_BOOT = 10_000
SEED = 25
FAST_MARK = "flash-lite"      # the fast LLM's id contains this (gemini-3.5-flash-lite)


# ------------------------------------------------------------------ task lists
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def ids_text(ids: list[str]) -> str:
    return "".join(f"{i}\n" for i in ids)


def seeded_order(ids: list[str], salt: str) -> list[str]:
    return sorted(ids, key=lambda i: sha256_text(f"{salt}\n{i}"))


def read_ids(path: Path) -> list[str]:
    return [ln.split("#")[0].strip() for ln in path.read_text().splitlines() if ln.split("#")[0].strip()]


def cmd_select(a: argparse.Namespace) -> int:
    reg = json.loads(Path(a.registry).read_text())
    def names(name: str, version: str) -> list[str]:
        for d in reg:
            if d.get("name") == name and d.get("version") == version:
                return sorted(t["name"] for t in d["tasks"])
        raise SystemExit(f"{name}@{version} not in the registry")
    tb = names("terminal-bench", "2.0")
    swe = names("swebench-verified", "1.0")
    assert len(tb) == 89 and len(swe) == 500, (len(tb), len(swe))
    CFG.mkdir(parents=True, exist_ok=True)
    tb_order = seeded_order(tb, SALT["tb"])
    swe_order = seeded_order(swe, SALT["swe"])
    files = {"tb2_tasks.txt": tb, "tb2_order.txt": tb_order, "swe_order.txt": swe_order,
             "swe_eval_ids.txt": swe_order[:100], "swe_smoke_ids.txt": swe_order[100:102],
             "tb2_smoke_tasks.txt": ["prove-plus-comm", "overfull-hbox"]}
    for fn, ids in files.items():
        (CFG / fn).write_text(ids_text(ids))
    sel = {"registered": "docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~01:35 UTC, E25 (c), (d), (e)",
           "harbor_registry": "laude-institute/harbor@4f848e367e6e2627c2c0b1826bb3b20ceab1198f registry.json",
           "salts": SALT, "rule_tb": "evaluation = all 89 tasks; if the gate cuts, the first K of tb2_order.txt "
           "(sha256(salt + newline + name)); smoke = prove-plus-comm, overfull-hbox (development only)",
           "rule_swe": "the 500 Verified ids ordered by sha256(salt + newline + id): evaluation = the first 100 (or "
           "the first n the gate allows), smoke = positions 100 and 101 (never evaluated)",
           **{fn: {"n": len(ids), "sha256": sha256_text(ids_text(ids))} for fn, ids in files.items()},
           "swe_pool_sha256": sha256_text(ids_text(swe)),
           "swe_pool_equals_e4r_pool": sha256_text(ids_text(swe)) == json.loads(
               (ROOT / "configs/explore/e4r_swe/selection.json").read_text()).get("pool_sha256")}
    (CFG / "selection.json").write_text(json.dumps(sel, indent=2) + "\n")
    print(json.dumps({k: v for k, v in sel.items() if isinstance(v, dict)}, indent=1))
    return 0


# ------------------------------------------------------------------ per-trial parsing
def _load_summarizer() -> Any:
    spec = importlib.util.spec_from_file_location("summarize_harbor", ROOT / "adapters/pi/summarize_harbor.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SH = _load_summarizer()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.exists():
        for ln in path.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def pi_requests(agent_dir: Path) -> dict[str, Any]:
    """Assistant messages of pi's session log(s) by physical model: requests and cost."""
    by: Counter[str] = Counter()
    cost: dict[str, float] = {}
    for f in sorted((agent_dir / "pi").rglob("*.jsonl")) if (agent_dir / "pi").exists() else []:
        for ev in _jsonl(f):
            m = ev.get("message") if isinstance(ev.get("message"), dict) else None
            if not m or m.get("role") != "assistant":
                continue
            name = str(m.get("model") or m.get("responseModel") or "?")
            by[name] += 1
            c = ((m.get("usage") or {}).get("cost") or {})
            cost[name] = cost.get(name, 0.0) + float(c.get("total") or 0.0) if isinstance(c, dict) else cost.get(name, 0.0)
    return {"by_model": dict(by), "cost_by_model": cost}


def swe_requests(agent_dir: Path) -> dict[str, Any]:
    by: Counter[str] = Counter()
    cost: dict[str, float] = {}
    for r in _jsonl(agent_dir / "jev-calls.jsonl"):
        name = str(r.get("model") or "?")
        by[name] += int(r.get("api_calls") or 1)
        cost[name] = cost.get(name, 0.0) + float(r.get("cost_usd") or 0.0)
    return {"by_model": dict(by), "cost_by_model": cost}


def routing_stats(agent_dir: Path) -> dict[str, Any]:
    recs = [r for r in _jsonl(agent_dir / "jev-decisions.jsonl") if r.get("dp") == "routing"]
    jev = [r for r in recs if r.get("model") is not None and not r.get("error")]
    rule = [r for r in recs if r.get("model") is None and not r.get("error")]
    err = [r for r in recs if r.get("error")]
    asked = [r for r in recs if r.get("model") is not None]
    return {"route_decisions": len(recs), "route_jev": len(jev), "route_rule": len(rule), "route_errors": len(err),
            "route_cheap": sum(1 for r in recs if r.get("final_value") == "cheap"),
            "route_cheap_jev": sum(1 for r in jev if r.get("final_value") == "cheap"),
            "route_asked": len(asked)}


SIGNATURE_ERROR = "missing a thought_signature"


def signature_error(agent_dir: Path) -> bool:
    """pi's session ended on Gemini's HTTP 400 for a function call without a thought signature (the defect of
    harness A's first router run; amendment 2026-10-09 ~03:05 UTC)."""
    for f in (agent_dir / "pi").rglob("*.jsonl") if (agent_dir / "pi").exists() else []:
        if SIGNATURE_ERROR in f.read_text(errors="replace"):
            return True
    return False


def trial(trial_dir: Path, harness: str) -> dict[str, Any]:
    row = SH.trial_row(trial_dir, JEV_PRICE_PER_M)
    agent = trial_dir / "agent"
    req = pi_requests(agent) if harness == "tb" else swe_requests(agent)
    n_req = sum(req["by_model"].values())
    n_fast = sum(v for k, v in req["by_model"].items() if FAST_MARK in k)
    llm = float(row.get("llm_cost_usd") or 0.0)
    jev = float(row.get("s1_cost_usd_est") or 0.0)
    row.update({"success": 1 if (row.get("reward") or 0) >= 1 else 0, "llm_cost": llm, "jev_cost": jev,
                "cost": llm + jev, "wall": row.get("agent_run_s"), "requests": n_req, "fast_requests": n_fast,
                "req_by_model": req["by_model"], "cost_by_model": req["cost_by_model"], **routing_stats(agent),
                "signature_error": harness == "tb" and signature_error(agent)})
    return row


def _load_job(job: Path, harness: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not job.is_dir():
        return rows
    for t in sorted(p for p in job.iterdir() if p.is_dir() and (p / "result.json").exists()):
        try:
            r = trial(t, harness)
        except (OSError, ValueError, KeyError) as e:
            print(f"skip {t}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        rows[r["task"]] = r
    return rows


def load_run(run_dir: Path, harness: str, arms: tuple[str, ...] = ARMS,
             rerun_dir: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """{arm: {task: trial row}}; with several attempts of a task (should not happen at k = 1) the last one wins.

    ``rerun_dir`` (amendment 2026-10-09 ~02:30 UTC): the one re-run of trials that ended in an infrastructure error; its
    result replaces the errored one (marked ``rerun``)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        if not (run_dir / arm).is_dir():
            continue
        rows = _load_job(run_dir / arm, harness)
        if rerun_dir is not None:
            for task, r in _load_job(rerun_dir / arm, harness).items():
                if task in rows and rows[task].get("infra_error"):
                    rows[task] = {**r, "rerun": True, "first_error": rows[task].get("error_class")}
        out[arm] = rows
    return out


def errored(run_dir: Path, harness: str) -> list[tuple[str, str]]:
    """(arm, task) of every trial whose final result carries an exception other than AgentTimeoutError."""
    return [(arm, t) for arm, rows in load_run(run_dir, harness).items() for t, r in sorted(rows.items())
            if r.get("infra_error")]


def cmd_errored(a: argparse.Namespace) -> int:
    for arm, t in errored(Path(a.run_dir), a.harness):
        print(arm, t)
    return 0


# ------------------------------------------------------------------ spend
def spend(run_dirs: list[Path]) -> dict[str, Any]:
    by: dict[str, float] = {}
    jev = 0.0
    n = 0
    for d in run_dirs:
        for job in sorted(p for p in d.iterdir() if p.is_dir()) if d.is_dir() else []:
            for t in job.iterdir():
                f = t / "result.json"
                if not f.exists():
                    continue
                try:
                    r = json.loads(f.read_text())
                except (OSError, ValueError):
                    continue
                c = float(((r.get("agent_result") or {}).get("cost_usd")) or 0.0)
                st = SH.decision_stats(t / "agent" / "jev-decisions.jsonl", JEV_PRICE_PER_M)
                by[f"{d.name}/{job.name}"] = by.get(f"{d.name}/{job.name}", 0.0) + c
                jev += float(st["s1_cost_usd_est"])
                n += 1
    llm = sum(by.values())
    return {"llm": round(llm, 4), "jev": round(jev, 6), "total": round(llm + jev, 4), "trials": n,
            "by_job": {k: round(v, 4) for k, v in sorted(by.items())}}


def cmd_spend(a: argparse.Namespace) -> int:
    print(json.dumps(spend([Path(p) for p in a.run_dirs])))
    return 0


# ------------------------------------------------------------------ gate (E25 (e))
def gate(harness: str, smoke: dict[str, dict[str, dict[str, Any]]], *, spent: float, cap: float,
         leaks: int = 0) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    rows = [(arm, t, r) for arm, ts in smoke.items() for t, r in ts.items()]
    smoke_tasks = sorted({t for _, t, _ in rows})
    missing = [f"{arm}/{t}" for arm in ARMS for t in smoke_tasks if t not in smoke.get(arm, {})]
    infra = [f"{arm}/{t}: {r.get('error_class')}" for arm, t, r in rows if r.get("infra_error")]
    checks["1_complete_no_infra_error"] = {"ok": bool(rows) and not missing and not infra and
                                          all(a in smoke for a in ARMS),
                                          "missing": missing, "infra_errors": infra, "trials": len(rows)}
    rt = list(smoke.get("jev_router", {}).values())
    dec = sum(r["route_decisions"] for r in rt)
    asked = sum(r["route_asked"] for r in rt)
    answered = sum(r["route_jev"] for r in rt)
    checks["2_router_jev_answers"] = {"ok": asked > 0 and answered >= 0.95 * asked, "decisions": dec,
                                      "asked": asked, "answered": answered,
                                      "fast_requests": sum(r["fast_requests"] for r in rt),
                                      "requests": sum(r["requests"] for r in rt)}
    zero = [f"{arm}/{t}" for arm, t, r in rows if not (r.get("llm_cost") or 0) > 0]
    checks["3_cost_reported"] = {"ok": not zero, "zero_cost": zero}
    checks["4_no_leak"] = {"ok": leaks == 0, "files": leaks}
    mean = {arm: (statistics.fmean([r["cost"] for r in smoke[arm].values()]) if smoke.get(arm) else 0.0)
            for arm in ARMS}
    per = {arm: max(SMOKE_MULT * mean[arm], FLOOR[arm]) for arm in ARMS}
    left = cap - spent
    def fits(n: int, arms: tuple[str, ...]) -> bool:
        return FACTOR * sum(per[x] for x in arms) * n <= left
    n_full = HARNESS[harness]["n_full"]
    if fits(n_full, MAIN_ARMS):
        n = n_full
    else:
        n = max((k for k in range(10, n_full, 10) if fits(k, MAIN_ARMS)), default=0)
    arms = MAIN_ARMS + (("always_cheap",) if n >= N_MIN and fits(n, ARMS) else ())
    checks["5_projection"] = {"ok": n >= N_MIN, "n": n, "arms": list(arms), "smoke_mean_per_trial": mean,
                              "per_trial_used": per, "smoke_mult": SMOKE_MULT, "floor": FLOOR, "factor": FACTOR,
                              "spent": spent, "cap": cap,
                              "projected": round(FACTOR * sum(per[x] for x in arms) * n, 2)}
    go = all(c["ok"] for c in checks.values())
    order_file = CFG / ("tb2_order.txt" if harness == "tb" else "swe_order.txt")
    tasks: list[str] = []
    if go:
        order = read_ids(order_file)
        tasks = order[:n] if (harness == "swe" or n < n_full) else sorted(order)
    return {"harness": harness, "decision": "go" if go else "no-go", "n": n if go else 0, "arms": list(arms) if go else [],
            "tasks": tasks, "full_dataset": go and harness == "tb" and n == n_full, "checks": checks,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def cmd_gate(a: argparse.Namespace) -> int:
    smoke = load_run(Path(a.smoke), a.harness)
    res = gate(a.harness, smoke, spent=a.spent if a.spent is not None else spend([Path(a.smoke)])["total"],
               cap=a.cap or HARNESS[a.harness]["cap"], leaks=a.leaks)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "tasks"}, indent=1))
    if a.include_out and res["decision"] == "go":
        Path(a.include_out).write_text("" if res["full_dataset"] else ids_text(res["tasks"]))
    return 0 if res["decision"] == "go" else 1


# ------------------------------------------------------------------ readout statistics (E25 (f))
def _boot_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(N_BOOT, n))


def paired(base: dict[str, dict[str, Any]], arm: dict[str, dict[str, Any]], *, complete_case: bool = False,
           seed: int = SEED) -> dict[str, Any]:
    tasks = sorted(set(base) & set(arm))
    if complete_case:
        tasks = [t for t in tasks if not base[t].get("infra_error") and not arm[t].get("infra_error")]
    n = len(tasks)
    if n == 0:
        return {"n": 0}
    sb = np.array([base[t]["success"] for t in tasks], float)
    sa = np.array([arm[t]["success"] for t in tasks], float)
    cb = np.array([base[t]["cost"] for t in tasks], float)
    ca = np.array([arm[t]["cost"] for t in tasks], float)
    wb = np.array([float(base[t]["wall"] or 0.0) for t in tasks], float)
    wa = np.array([float(arm[t]["wall"] or 0.0) for t in tasks], float)
    wins = int(((sa == 1) & (sb == 0)).sum())
    losses = int(((sa == 0) & (sb == 1)).sum())
    both = int(((sa == 1) & (sb == 1)).sum())
    ni = ni_certified_p(wins, losses, n, MARGIN)
    tango = tango_interval(both, wins, losses, n - both - wins - losses)["ci"]
    rng = np.random.default_rng(seed)
    idx = _boot_idx(n, rng)
    d_boot = (sa[idx].mean(1) - sb[idx].mean(1)) * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        s_boot = (1 - ca[idx].mean(1) / cb[idx].mean(1)) * 100
        t_boot = (1 - wa[idx].mean(1) / wb[idx].mean(1)) * 100
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    return {"n": n, "wins": wins, "losses": losses, "both": both,
            "delta": float((sa.mean() - sb.mean()) * 100), "delta_ci": q(d_boot),
            "tango_ci": [100 * tango[0], 100 * tango[1]], "ni_p": float(ni["p"]),
            "ni": "yes" if ni["p"] <= ALPHA else "no",
            "saving": float((1 - ca.mean() / cb.mean()) * 100) if cb.mean() > 0 else float("nan"),
            "saving_ci": q(s_boot),
            "time_saving": float((1 - wa.mean() / wb.mean()) * 100) if wb.mean() > 0 else float("nan"),
            "time_saving_ci": q(t_boot),
            "median_time_saving": float((1 - np.median(wa) / np.median(wb)) * 100) if np.median(wb) > 0 else float("nan"),
            "cheaper_share": float((ca < cb).mean() * 100), "faster_share": float((wa < wb).mean() * 100),
            "tasks": tasks}


def arm_stats(rows: dict[str, dict[str, Any]], tasks: list[str]) -> dict[str, Any]:
    rs = [rows[t] for t in tasks if t in rows]
    if not rs:
        return {"n": 0}
    walls = [float(r["wall"] or 0.0) for r in rs]
    req = sum(r["requests"] for r in rs)
    fast = sum(r["fast_requests"] for r in rs)
    dec = sum(r["route_decisions"] for r in rs)
    return {"n": len(rs), "success": 100 * statistics.fmean(r["success"] for r in rs),
            "solved": sum(r["success"] for r in rs),
            "cost_usd": statistics.fmean(r["cost"] for r in rs), "llm_cost_usd": statistics.fmean(r["llm_cost"] for r in rs),
            "jev_cost_usd": statistics.fmean(r["jev_cost"] for r in rs),
            "wall_mean_s": statistics.fmean(walls), "wall_median_s": statistics.median(walls),
            "timeouts": sum(1 for r in rs if r.get("timed_out")), "infra_errors": sum(1 for r in rs if r.get("infra_error")),
            "requests_mean": req / len(rs) if req else None, "taken_share": 100 * fast / req if req else 0.0,
            "route_decisions": dec, "route_jev_share": 100 * sum(r["route_jev"] for r in rs) / dec if dec else None,
            "route_rule_share": 100 * sum(r["route_rule"] for r in rs) / dec if dec else None,
            "route_error_share": 100 * sum(r["route_errors"] for r in rs) / dec if dec else None,
            "route_cheap_share": 100 * sum(r["route_cheap"] for r in rs) / dec if dec else None,
            "jev_s_per_task": statistics.fmean(float(r.get("s1_s_total") or 0.0) for r in rs),
            # descriptive diagnostics: agent steps (Harbor's ATIF trajectory) and LLM input tokens per task
            "steps_mean": (statistics.fmean(float(r["agent_steps"]) for r in rs if r.get("agent_steps") is not None)
                           if any(r.get("agent_steps") is not None for r in rs) else None),
            # (None where the harness reports no token counts to Harbor: SWE-agent's result.json carries 0)
            "input_tokens_mean": (statistics.fmean(float(r.get("llm_input_tokens") or 0) for r in rs)
                                  if any((r.get("llm_input_tokens") or 0) > 0 for r in rs) else None)}


STOPPED = "CancelledError"   # Harbor's record of a trial stopped from outside (the spend watchdog's SIGINT)


def restrict_to_complete(run: dict[str, dict[str, dict[str, Any]]]) -> tuple[dict[str, dict[str, dict[str, Any]]],
                                                                             dict[str, list[str]]]:
    """Amendment 2026-10-09 ~05:15 UTC (E25 (e)): after a watchdog stop the readout covers the tasks complete in every
    arm. A trial Harbor recorded as stopped (CancelledError) or never started is not complete; its task leaves both
    views. Returns the restricted run and {task: [arms without a complete trial]} (empty when every arm is complete)."""
    arms_run = [a for a in ARMS if run.get(a)]
    if not any(r.get("error") == STOPPED for a in arms_run for r in run[a].values()):
        return run, {}   # no stop: every task of every arm stays (the registered full set and complete case)
    done = {a: {t for t, r in run[a].items() if r.get("error") != STOPPED} for a in arms_run}
    every = set.union(*(set(run[a]) for a in arms_run))
    complete = set.intersection(*done.values())
    left = {t: [a for a in arms_run if t not in done[a]] for t in sorted(every - complete)}
    if not left:
        return run, {}
    return {a: {t: r for t, r in run[a].items() if t in complete} for a in arms_run}, left


def summarize_harness(harness: str, run: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    run, left_out = restrict_to_complete(run)
    base = run.get("baseline", {})
    arms_run = [a for a in ARMS if run.get(a)]
    common = sorted(set.intersection(*(set(run[a]) for a in arms_run))) if arms_run else []
    main = sorted(set(base) & set(run.get("jev_router", {})))
    out: dict[str, Any] = {"harness": harness, **HARNESS[harness], "arms_run": arms_run, "n_common_all_arms": len(common),
                           "left_out": left_out, "arms": {}, "paired": {}, "paired_complete_case": {},
                           "errored": sorted(f"{a}/{t}: {(r.get('error_class') or '')[:120]}" for a in arms_run
                                             for t, r in run[a].items() if r.get("infra_error"))}
    for arm in arms_run:
        tasks = sorted(set(base) & set(run[arm])) if arm != "baseline" else main or sorted(base)
        out["arms"][arm] = arm_stats(run[arm], tasks)
        if arm == "always_cheap" and out["arms"][arm].get("n"):
            out["arms"][arm]["taken_share"] = 100.0   # by construction (SWE-agent's unrouted runs log no per-call file)
        if arm != "baseline":
            out["paired"][arm] = paired(base, run[arm])
            out["paired_complete_case"][arm] = paired(base, run[arm], complete_case=True)
    if "jev_router" in out["paired"]:
        out["arms"]["baseline_on_router_tasks"] = arm_stats(base, out["paired"]["jev_router"]["tasks"])
    return out


# ------------------------------------------------------------------ output
def _f(x: Any, d: int = 1) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _signed(x: float, d: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def to_markdown(summ: dict[str, Any]) -> str:
    L = ["# E25: per-step routing in two more open-source agent harnesses (exploratory, descriptive)", "",
         "Registered in docs/PREREG_A7_STAGE2.md (Deviations 2026-10-09, E25) before any E25 episode. Strong LLM Gemini "
         "3.5 Flash, fast LLM Gemini 3.5 Flash-Lite (minimal thinking, direct API); decision model Jev (OpenRouter) at "
         "tau 0.5. Cost = LLM + decision model per task; time = Harbor's agent-execution seconds per task. Full set "
         "(an errored trial counts as a failure); complete case below each table.", ""]
    for h, s in summ["harnesses"].items():
        L += [f"## {s['label']} ({s['harness']})", "",
              "| arm | n | success % | cost $/task | saving % [95 % CI] | mean s | median s | time saving % [95 % CI] "
              "| fast-LLM share % | timeouts | infra errors |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm, st in s["arms"].items():
            if arm == "baseline_on_router_tasks" or not st.get("n"):
                continue
            p = s["paired"].get(arm, {})
            sv = f"{_f(p.get('saving'))} [{_f(p['saving_ci'][0])}, {_f(p['saving_ci'][1])}]" if p.get("n") else "--"
            tv = (f"{_f(p.get('time_saving'))} [{_f(p['time_saving_ci'][0])}, {_f(p['time_saving_ci'][1])}]"
                  if p.get("n") else "--")
            L.append(f"| {arm} | {st['n']} | {_f(st['success'])} | {_f(st['cost_usd'], 4)} | {sv} | {_f(st['wall_mean_s'], 0)} "
                     f"| {_f(st['wall_median_s'], 0)} | {tv} | {_f(st['taken_share'])} | {st['timeouts']} | {st['infra_errors']} |")
        L += ["", "| arm vs baseline | n | wins | losses | delta pt [bootstrap 95 %] | Tango 95 % | NI p (3 pt) | NI "
              "| median-time saving % | cheaper on % tasks | faster on % tasks |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for view, key in (("full set", "paired"), ("complete case", "paired_complete_case")):
            for arm, p in s[key].items():
                if not p.get("n"):
                    continue
                L.append(f"| {arm} ({view}) | {p['n']} | {p['wins']} | {p['losses']} | {_f(p['delta'])} "
                         f"[{_f(p['delta_ci'][0])}, {_f(p['delta_ci'][1])}] | [{_f(p['tango_ci'][0])}, {_f(p['tango_ci'][1])}] "
                         f"| {p['ni_p']:.3f} | {p['ni']} | {_f(p['median_time_saving'])} | {_f(p['cheaper_share'])} "
                         f"| {_f(p['faster_share'])} |")
        diag = [f"{arm} {_f(st.get('steps_mean'))} agent steps / "
                f"{_f(st['input_tokens_mean'] / 1e3, 0) + 'k' if st.get('input_tokens_mean') else '--'} input tokens"
                for arm, st in s["arms"].items() if arm != "baseline_on_router_tasks" and st.get("n")]
        L += ["", "Per task (descriptive): " + "; ".join(diag) + "."]
        if s.get("defect_run"):
            dr = s["defect_run"]
            L += ["", f"Defect run (not the router arm): {dr['n']} trials of the first router run, {_f(dr['success'])} % "
                  f"solved; {dr['signature_errors']} ended on Gemini's missing-thought-signature error "
                  f"({dr['baseline_tasks_signature_errors']} in baseline)."]
        if s.get("left_out"):
            L += ["", f"Watchdog stop (E25 (e); amendment 2026-10-09 ~05:15 UTC): the tables cover the {s['n_common_all_arms']} "
                  f"tasks complete in every arm; {len(s['left_out'])} task(s) left out (arms without a complete trial): "
                  + "; ".join(f"{t} ({', '.join(a)})" for t, a in s["left_out"].items())]
        if s.get("errored"):
            L += ["", "Trials whose final result carries an infrastructure error (failures in the full set; their pairs leave "
                  "the complete case): " + "; ".join(s["errored"])]
        for note in s.get("notes", []):
            L += ["", note]
        if s.get("reruns"):
            L += ["", "Re-run once after an infrastructure error (amendment 2026-10-09 ~02:30 UTC): " + "; ".join(s["reruns"])]
        r = s["arms"].get("jev_router", {})
        if r.get("n"):
            L += ["", f"Router: {r['route_decisions']} routing decisions; Jev answered {_f(r['route_jev_share'])} %, "
                  f"escalation rules {_f(r['route_rule_share'])} %, decision-model errors (fail-open to the strong LLM) "
                  f"{_f(r['route_error_share'])} %; {_f(r['taken_share'])} % of the LLM requests went to the fast LLM; "
                  f"decision-model cost ${_f(r['jev_cost_usd'], 5)} and {_f(r['jev_s_per_task'])} s per task."]
        L.append("")
    sp = summ.get("spend") or {}
    if sp:
        L += ["## Spend", "", "```", json.dumps(sp, indent=1), "```", ""]
    return "\n".join(L)


def to_tex(summ: dict[str, Any]) -> str:
    lines = ["% AUTO-GENERATED by scripts/explore_e25_harnesses.py -- do not edit by hand.",
             f"% generated {summ['generated']} at git {summ.get('git', '?')}; exploratory (E25), descriptive "
             "(PREREG_A7_STAGE2 Deviations 2026-10-09, E25); outside every Holm table",
             "\\makeatletter",
             "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    def d(k: str, v: Any) -> None:
        lines.append(f"\\defres{{e25/{k}}}{{{v}}}")
    d("margin", f"{100 * MARGIN:.0f}")
    for h, s in summ["harnesses"].items():
        d(f"{h}/n_common", s["n_common_all_arms"])
        d(f"{h}/n_left_out", len(s.get("left_out") or {}))
        for arm, st in s["arms"].items():
            if not st.get("n"):
                continue
            k = f"{h}/{arm}"
            d(f"{k}/n", st["n"])
            d(f"{k}/success", _f(st["success"]))
            d(f"{k}/cost_usd", _f(st["cost_usd"], 4))
            d(f"{k}/wall_mean_s", _f(st["wall_mean_s"], 0))
            d(f"{k}/wall_median_s", _f(st["wall_median_s"], 0))
            d(f"{k}/taken_share", _f(st["taken_share"]))
            if st.get("requests_mean") is not None:   # SWE-agent's unrouted runs log no per-call file
                d(f"{k}/requests_mean", _f(st["requests_mean"]))
            if st.get("steps_mean") is not None:
                d(f"{k}/steps_mean", _f(st["steps_mean"]))   # descriptive: agent steps from Harbor's trajectory
            if arm == "jev_router":
                d(f"{k}/jev_answer_share", _f(st["route_jev_share"]))
                d(f"{k}/route_rule_share", _f(st["route_rule_share"]))
                d(f"{k}/route_error_share", _f(st["route_error_share"]))
                d(f"{k}/route_cheap_share", _f(st["route_cheap_share"]))
                d(f"{k}/jev_cost_usd", _f(st["jev_cost_usd"], 5))
                d(f"{k}/jev_s_per_task", _f(st["jev_s_per_task"]))   # seconds of decision-model calls per task
            p = s["paired"].get(arm)
            if p and p.get("n"):
                d(f"{k}/delta", _signed(p["delta"]))
                d(f"{k}/delta_lo", _signed(p["delta_ci"][0]))
                d(f"{k}/delta_hi", _signed(p["delta_ci"][1]))
                d(f"{k}/saving", _f(p["saving"]))
                d(f"{k}/saving_lo", _f(p["saving_ci"][0]))
                d(f"{k}/saving_hi", _f(p["saving_ci"][1]))
                d(f"{k}/time_saving", _f(p["time_saving"]))
                # signed changes against the strong LLM alone (cost: minus = cheaper; time: plus = slower)
                d(f"{k}/cost_change", _signed(-p["saving"]))
                d(f"{k}/time_change", _signed(-p["time_saving"]))
                d(f"{k}/time_more", _f(-p["time_saving"]))
                d(f"{k}/time_saving_lo", _f(p["time_saving_ci"][0]))
                d(f"{k}/time_saving_hi", _f(p["time_saving_ci"][1]))
                d(f"{k}/median_time_saving", _f(p["median_time_saving"]))
                d(f"{k}/cheaper_share", _f(p["cheaper_share"], 0))
                d(f"{k}/faster_share", _f(p["faster_share"], 0))
                # hardni-style keys (scripts/hard_ni.py's layout): the certified NI p at 3 points and Tango's interval
                d(f"{k}/wins", p["wins"])
                d(f"{k}/losses", p["losses"])
                d(f"{k}/ni_p", f"{p['ni_p']:.3f}")
                d(f"{k}/ni", p["ni"])
                d(f"{k}/lo", _signed(p["tango_ci"][0]))
                d(f"{k}/hi", _signed(p["tango_ci"][1]))
            c = s["paired_complete_case"].get(arm)
            if c and c.get("n"):   # the complete case (pairs without an infrastructure error in either arm)
                d(f"{k}/cc/n", c["n"])
                d(f"{k}/cc/delta", _signed(c["delta"]))
                d(f"{k}/cc/saving", _f(c["saving"]))
                d(f"{k}/cc/time_saving", _f(c["time_saving"]))
                d(f"{k}/cc/ni_p", f"{c['ni_p']:.3f}")
                d(f"{k}/cc/ni", c["ni"])
        dr = s.get("defect_run")
        if dr:
            d(f"{h}/jev_router_defect/n", dr["n"])
            d(f"{h}/jev_router_defect/success", _f(dr["success"]))
            d(f"{h}/jev_router_defect/signature_errors", dr["signature_errors"])
    lines.append("\\makeatother")
    return "\n".join(lines) + "\n"


def _git() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def cmd_readout(a: argparse.Namespace) -> int:
    summ: dict[str, Any] = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": _git(),
                            "registered": "docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~01:35 UTC, E25",
                            "harnesses": {}}
    def rerun_of(path: str) -> Path | None:
        d = Path(path).parent / "eval_rerun"
        return d if d.is_dir() else None
    for h, path in (("tb", a.tb), ("swe", a.swe)):
        if path:
            run = load_run(Path(path), h, rerun_dir=rerun_of(path))
            if run.get("baseline"):
                summ["harnesses"][h] = summarize_harness(h, run)
                summ["harnesses"][h]["reruns"] = sorted(
                    f"{arm}/{t}: {r.get('first_error')} -> {'error ' + str(r.get('error_class')) if r.get('infra_error') else 'reward ' + str(r.get('reward'))}"
                    for arm, rows in run.items() for t, r in rows.items() if r.get("rerun"))
    if "tb" in summ["harnesses"]:
        summ["harnesses"]["tb"]["notes"] = [
            "Time (amendment 2026-10-09 ~03:05 UTC): the router arm ran again from 03:02 UTC with the fix, while baseline "
            "and always_cheap finished, so its later trials ran on a less loaded host than baseline's; the time "
            "comparison is weaker for this harness than the cost and success comparisons."]
    if a.tb and (Path(a.tb).parent / "eval_defect" / "jev_router").is_dir() and "tb" in summ["harnesses"]:
        rows = _load_job(Path(a.tb).parent / "eval_defect" / "jev_router", "tb")
        if rows:
            summ["harnesses"]["tb"]["defect_run"] = {
                "note": "first router run, stopped: pi-ai dropped thought signatures across models (amendment ~03:05 UTC)",
                "n": len(rows), "success": 100 * statistics.fmean(r["success"] for r in rows.values()),
                "signature_errors": sum(1 for r in rows.values() if r["signature_error"]),
                "baseline_tasks_signature_errors": sum(1 for r in load_run(Path(a.tb), "tb")["baseline"].values()
                                                       if r["signature_error"])}
    for sj in a.spend_json or []:   # one box record per harness (results/explore/e25_harnesses/box/<H>/status/e25_spend.json)
        if Path(sj).exists():
            h = next((x for x in HARNESS if f"/{x}/" in str(Path(sj).resolve())), Path(sj).stem)
            summ.setdefault("spend", {})[h] = json.loads(Path(sj).read_text())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(summ, default=str))
    for s in slim["harnesses"].values():
        for key in ("paired", "paired_complete_case"):
            for p in s[key].values():
                p.pop("tasks", None)
    (out / "summary.json").write_text(json.dumps(slim, indent=2) + "\n")
    (out / "summary.md").write_text(to_markdown(summ))
    Path(a.tex).write_text(to_tex(summ))
    per = out / "per_task.csv"
    with per.open("w") as f:
        f.write("harness,arm,task,success,cost,llm_cost,jev_cost,wall_s,requests,fast_requests,timed_out,infra_error,"
                "rerun\n")
        for h, path in (("tb", a.tb), ("swe", a.swe)):
            if not path:
                continue
            for arm, rows in load_run(Path(path), h, rerun_dir=rerun_of(path)).items():
                for t, r in sorted(rows.items()):
                    f.write(f"{h},{arm},{t},{r['success']},{r['cost']:.6f},{r['llm_cost']:.6f},{r['jev_cost']:.6f},"
                            f"{_f(r['wall'], 1)},{r['requests']},{r['fast_requests']},{int(bool(r.get('timed_out')))},"
                            f"{int(bool(r.get('infra_error')))},{int(bool(r.get('rerun')))}\n")
    print((out / "summary.md").read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--registry", required=True)
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("spend")
    p.add_argument("run_dirs", nargs="+")
    p.set_defaults(fn=cmd_spend)
    p = sub.add_parser("errored")
    p.add_argument("--harness", choices=sorted(HARNESS), required=True)
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_errored)
    p = sub.add_parser("gate")
    p.add_argument("--harness", choices=sorted(HARNESS), required=True)
    p.add_argument("--smoke", required=True)
    p.add_argument("--spent", type=float, default=None)
    p.add_argument("--cap", type=float, default=None)
    p.add_argument("--leaks", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--include-out", default=None, help="write the evaluation task list (empty = the full dataset)")
    p.set_defaults(fn=cmd_gate)
    p = sub.add_parser("readout")
    p.add_argument("--tb", default=None)
    p.add_argument("--swe", default=None)
    p.add_argument("--spend-json", action="append", default=None, help="a box spend record; repeat per harness")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--tex", default=str(TEX))
    p.set_defaults(fn=cmd_readout)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
