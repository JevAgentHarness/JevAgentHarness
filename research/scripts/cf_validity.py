#!/usr/bin/env python3
"""Validity check of the one-step counterfactual efficacy labels (review round 4, P1-6; META_GOLD_SPEC §4, §11.7).

DEV diagnostic, never a result. The reviewers' worry: a one-step counterfactual System-2 call on a snapshot (the
reflection / large-model / stop-rejection branch, netted against a placebo resample) may not predict what happens
over the REST of the episode. This script takes the one-step labels a measurement arm logged, rebuilds each sampled
state by replaying the logged action prefix through a fresh JevGym env (deterministic given seed + actions; the
replay is the audit's reference implementation), and runs the FULL remaining episode with the LLM-only harness
(a) with the intervention applied once at that step and (b) without, ``--reps`` replicate rollouts each (T = 0;
replicates sample the provider's nondeterminism, the same noise the placebo nets out), and records final success.
It then reports the agreement of the one-step label (helps / neutral / harms) with the sign of the full-rollout
outcome difference: confusion table, binary agreement with a Wilson CI, Cohen's kappa, per decision point where n
allows, and the spend.

Two modes. ``--mode live`` (the default) re-runs a small measurement arm (the arm's shadow gates and
counterfactual branches, no probes, a free System-1 since the one-step labels do not depend on it) on the first
``--n-tasks`` tasks of the run's suite, keeps every decision-time snapshot in process (``measure.keep_snapshots``:
the env copy and the System-2 messages WITH their provider extras), then resumes the selected snapshots
(``measure.continue_from_snapshot``) -- the continuation starts from exactly the state the one-step branch was
computed on. ``--mode replay`` instead rebuilds the states of an EXISTING arm log by replaying the logged action
prefix (``measure.continue_episode``); on Gemini that rebuilt history (placeholder thought signatures) mostly draws
``MALFORMED_FUNCTION_CALL`` (smoke, 2026-10-01: 11 of 16 samples), so replay is for providers without signatures
and for offline pipeline tests.

    # what would run (no spend): the stratified subset of an existing log and the number of rollouts
    python scripts/cf_validity.py --run runs/dev/e2b_dev_r3 --arm meta_shadow_s2lite --mode replay --n 30 --dry-run
    # live: measurement arm on 30 tasks, then 30 opportunities x 2 reps x with/without (every call through a Ledger;
    # stops once the budget is hit and reports what was done)
    python scripts/cf_validity.py --run runs/dev/e2b_dev_r3 --arm meta_shadow_s2lite --mode live --n-tasks 30 \\
        --n 30 --reps 2 --budget-usd 3.5 --out runs/dev/cf_validity
    # recompute the table from the rollouts written so far
    python scripts/cf_validity.py --analyse runs/dev/cf_validity

The System-2 is the arm's own LLM (``arms[].llm`` of the run's experiment.yaml, else ``llm``); the ``large``
intervention uses ``measure.cf.large_llm`` of the arm, as the one-step label does. ``--llm`` / ``--large-llm``
override both (``{"provider": "simulated", ...}`` runs the whole pipeline offline; such output is flagged
``simulated: true`` and is for pipeline tests only). Outputs: ``<out>/measure_arm.jsonl`` (live mode: the
measurement episodes with their one-step labels), ``<out>/subset.json`` (the opportunities and their one-step
labels), ``<out>/rollouts.jsonl`` (one record per rollout), ``<out>/summary.json`` and a Markdown table on stdout.
Notes: results/dev_round2/CF_VALIDITY_NOTES.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.accounting import Ledger  # noqa: E402
from jevharness.bench import meta_gold as MG  # noqa: E402
from jevharness.harness.config import CFConfig, DPConfig, HarnessConfig, MeasureConfig, Mode  # noqa: E402
from jevharness.harness.trajectory import load_trajectories  # noqa: E402

DP_BLOCKS = ("tool_select", "tool_params", "memory_write", "memory_read", "reflection", "stop", "routing", "action",
             "measure", "name")


def _key(o: dict[str, Any]) -> str:
    return f"{o['task_id']}|{o['step']}|{o['dp']}"


def llm_specs(exp: dict[str, Any], arm: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(the arm's System-2 spec, the spec of its ``measure.cf.large_llm``) from the run's experiment.yaml."""
    a = next((x for x in exp.get("arms", []) if x.get("name") == arm), None)
    if a is None:
        raise SystemExit(f"arm {arm!r} not in {exp.get('name')}'s experiment.yaml")
    llms = exp.get("llms") or {}
    spec = llms[a["llm"]] if a.get("llm") else exp["llm"]
    large_name = ((a.get("measure") or {}).get("cf") or {}).get("large_llm")
    large = None
    if large_name:
        large = {"main": exp.get("llm"), "small": exp.get("small_llm")}.get(large_name) or llms.get(large_name)
    return dict(spec), (dict(large) if large else None)


def llm_only_config(logged_cfg: dict[str, Any], name: str = "llm_only_cf_full") -> HarnessConfig:
    """The LLM-only harness with the arm's loop parameters (max_steps, truncation, final tool)."""
    base = {k: v for k, v in (logged_cfg or {}).items() if k not in DP_BLOCKS}
    return HarnessConfig(name=name, **base).validate_modes()


def measurement_config(logged_cfg: dict[str, Any], large_name: str = "large", sample_negatives: float = 0.2,
                       name: str = "meta_shadow_cfv") -> HarnessConfig:
    """The live measurement arm of the validity check: the arm's loop parameters, reflection / routing / stop
    shadowed (so every opportunity gets its one-step label), the three counterfactual branches with the placebo,
    no probes and no memory shadow (neither feeds a one-step label)."""
    base = {k: v for k, v in (logged_cfg or {}).items() if k not in DP_BLOCKS}
    return HarnessConfig(name=name, reflection=DPConfig(mode=Mode.SHADOW), routing=DPConfig(mode=Mode.SHADOW),
                         stop=DPConfig(mode=Mode.SHADOW),
                         measure=MeasureConfig(seed=0, cf=CFConfig(reflection=True, large_llm=large_name,
                                                                   stop_reject=True, placebo=True,
                                                                   sample_negatives=sample_negatives)),
                         **base).validate_modes()


def live_measurement(agent: Any, factories: list[Any], out_path: Path, workers: int) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    """Phase 1 of the live mode: run the measurement arm on ``factories`` (fresh file), keeping the snapshots.
    Returns (trajectory dicts, task id -> snapshots)."""
    from jevharness.harness.measure import keep_snapshots

    if out_path.exists():
        out_path.unlink()
    trajs: list[dict[str, Any]] = []
    with keep_snapshots() as snaps, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(agent.run, fac()) for fac in factories]
        for i, fut in enumerate(futs):
            t = fut.result()
            t.dump(out_path)
            d = t.to_dict()
            trajs.append(d)
            u = d.get("usage") or {}
            print(f"[measure {i + 1}/{len(futs)}] {d['task_id']}: success={d['result'].get('success')} "
                  f"steps={len(d['steps'])} ${(u.get('total_with_shadow') or u.get('total') or {}).get('cost_usd', 0):.4f}"
                  f"{' ERROR ' + str(d.get('error'))[:80] if d.get('error') else ''}", file=sys.stderr)
    return trajs, dict(snaps)


def measurement_cost(trajs: list[dict[str, Any]]) -> float:
    return sum(float(((t.get("usage") or {}).get("total_with_shadow") or (t.get("usage") or {}).get("total") or {})
                     .get("cost_usd", 0.0) or 0.0) for t in trajs)


def plan_units(subset: list[dict[str, Any]], reps: int) -> list[dict[str, Any]]:
    """The rollouts in run order: per opportunity, rep 0 with and without, then rep 1, ... (so a budget stop leaves
    complete pairs first)."""
    units = []
    for o in subset:
        for rep in range(reps):
            for arm in ("with", "without"):
                units.append({"key": _key(o), "rep": rep, "arm": arm, "opp": o})
    return units


def _done(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                out.add((r["key"], int(r["rep"]), r["arm"]))
    return out


def run_units(units: list[dict[str, Any]], agent: Any, large: Any, factories: dict[str, Any], out_path: Path,
              budget_usd: float, workers: int, simulated: bool, snapshots: dict[str, list[Any]] | None = None,
              spent0: float = 0.0) -> dict[str, Any]:
    """Run the rollouts through a shared Ledger, ``workers`` at a time, submitting nothing once the priced spend
    (``spent0`` already spent included) passes ``budget_usd``; every finished rollout is appended to ``out_path``
    (resumable). With ``snapshots`` (task id -> StepSnaps) a rollout resumes the in-process snapshot of its step
    (``continue_from_snapshot``); otherwise it rebuilds the state from the logged prefix (``continue_episode``)."""
    from jevharness.harness.measure import continue_episode, continue_from_snapshot

    ledger = Ledger()
    lock = threading.Lock()
    done = _done(out_path)
    todo = [u for u in units if (u["key"], u["rep"], u["arm"]) not in done]
    stopped = False
    n_run = n_err = 0
    t0 = time.time()

    def spent() -> float:
        return spent0 + sum(r.cost_usd for r in ledger.records)

    def one(u: dict[str, Any]) -> dict[str, Any]:
        o = u["opp"]
        sub = Ledger()
        iv = o["intervention"] if u["arm"] == "with" else None
        if snapshots is not None:
            snap = next((x for x in snapshots.get(o["task_id"], []) if x.step == o["step"]), None)
            if snap is None:
                raise RuntimeError(f"no snapshot kept for {o['task_id']} step {o['step']}")
            ro = continue_from_snapshot(agent, snap, iv, ledger=sub, large_llm=large)
        else:
            env = factories[o["task_id"]]()
            ro = continue_episode(agent, env, [tuple(x) for x in o["prefix"]], iv, ledger=sub, large_llm=large,
                                  logged_obs=o["logged_obs"], a0=tuple(o["a0"]))
        with ledger._lock:
            ledger.records.extend(sub.records)
        return {"key": u["key"], "task_id": o["task_id"], "step": o["step"], "dp": o["dp"], "source": ro.source,
                "intervention": o["intervention"], "label": o["label"], "stratum": o.get("stratum"),
                "v0": o.get("v0"), "v_branch": o.get("v_branch"), "v_placebo": o.get("v_placebo"),
                "final_step": o["final_step"], "episode_success": o["episode_success"], "rep": u["rep"],
                "arm": u["arm"], "success": ro.success, "n_steps": ro.n_steps, "start_step": ro.start_step,
                "actions": [[n, a] for n, a in ro.actions], "final_answer": ro.final_answer,
                "prefix_match": ro.prefix_match, "cost_usd": ro.cost_usd, "latency_ms": ro.latency_ms,
                "error": ro.error, "reflection": ro.reflection, "result": ro.result, "calls": ro.calls,
                "simulated": simulated}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex, open(out_path, "a") as f:
        pending: set[Any] = set()
        it = iter(todo)
        while True:
            while len(pending) < workers:
                if spent() >= budget_usd:
                    stopped = True
                    break
                u = next(it, None)
                if u is None:
                    break
                pending.add(ex.submit(one, u))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                n_run += 1
                n_err += int(bool(rec.get("error")))
                with lock:
                    f.write(json.dumps(rec, default=str) + "\n")
                    f.flush()
                print(f"[{n_run}/{len(todo)}] {rec.get('key', '?')} rep {rec.get('rep')} {rec.get('arm')}: "
                      f"success={rec.get('success')} steps={rec.get('n_steps')} ${rec.get('cost_usd', 0) or 0:.4f} "
                      f"(spent ${spent():.3f}){' ERROR ' + str(rec.get('error'))[:80] if rec.get('error') else ''}",
                      file=sys.stderr)
            if stopped and not pending:
                break
    s = ledger.summary()
    return {"n_planned": len(units), "n_already_done": len(units) - len(todo), "n_run": n_run, "n_errors": n_err,
            "budget_usd": budget_usd, "stopped_at_budget": stopped, "spent_usd": spent(),
            "rollouts_usd": spent() - spent0, "spent_before_usd": spent0,
            "wall_s": time.time() - t0, "ledger": {k: s[k] for k in ("total",)},
            "by_model": {m: sum(r.cost_usd for r in ledger.records if r.model == m)
                         for m in sorted({str(r.model) for r in ledger.records})},
            "unpriced_models": s.get("unpriced_models", [])}


def analyse(out_dir: Path | list[Path]) -> dict[str, Any]:
    """Per-opportunity delta (mean success with - without over the replicates that ran) and the agreement table.
    Several directories pool their rollouts (opportunity keys are prefixed by the directory name)."""
    dirs = [out_dir] if isinstance(out_dir, Path) else list(out_dir)
    recs = []
    for d in dirs:
        rows_path = d / "rollouts.jsonl"
        for ln in (rows_path.read_text().splitlines() if rows_path.exists() else []):
            if ln.strip():
                r = json.loads(ln)
                if len(dirs) > 1:
                    r["key"] = f"{d.name}:{r['key']}"
                recs.append(r)
    recs = [r for r in recs if not r.get("error")]
    by: dict[str, dict[str, Any]] = {}
    for r in recs:
        b = by.setdefault(r["key"], {"key": r["key"], "task_id": r["task_id"], "step": r["step"], "dp": r["dp"],
                                     "intervention": r["intervention"], "label": r["label"],
                                     "stratum": r.get("stratum"), "final_step": r.get("final_step"),
                                     "v0": r.get("v0"), "v_branch": r.get("v_branch"), "v_placebo": r.get("v_placebo"),
                                     "episode_success": r.get("episode_success"), "with": [], "without": [],
                                     "cost_usd": 0.0, "prefix_match": True, "n_steps_with": [],
                                     "n_steps_without": []})
        b[r["arm"]].append(bool(r["success"]))
        b[f"n_steps_{r['arm']}"].append(int(r.get("n_steps") or 0))
        b["cost_usd"] += float(r.get("cost_usd") or 0.0)
        if r.get("prefix_match") is False:
            b["prefix_match"] = False
    rows = []
    incomplete = []
    for b in by.values():
        if b["with"] and b["without"]:
            b["p_with"] = sum(b["with"]) / len(b["with"])
            b["p_without"] = sum(b["without"]) / len(b["without"])
            b["delta"] = b["p_with"] - b["p_without"]
            b["control_success"] = b["p_without"] >= 0.5 if len(b["without"]) else None
            b["sign"] = MG.rollout_sign(b["delta"])
            rows.append(b)
        else:
            incomplete.append(b["key"])
    table = MG.cf_validity_table(rows) if rows else {}
    out = {"n_opportunities": len(rows), "incomplete": incomplete, "n_rollouts": len(recs),
           "cost_usd": sum(float(r.get("cost_usd") or 0.0) for r in recs),
           "sources": sorted({str(r.get("source")) for r in recs}),
           "simulated": any(r.get("simulated") for r in recs), "table": table, "rows": rows,
           "control_agrees_with_log_per_rollout": (sum(1 for r in recs if r["arm"] == "without"
                                                       and bool(r["success"]) == bool(r.get("episode_success")))
                                                   / max(1, sum(1 for r in recs if r["arm"] == "without"))),
           "prefix_mismatches": sum(1 for r in recs if r.get("prefix_match") is False)}
    return out


def _f(x: Any, d: int = 3) -> str:
    try:
        return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def markdown(summary: dict[str, Any]) -> str:
    t = summary.get("table") or {}
    lines = []
    if summary.get("simulated"):
        lines.append("**SIMULATED backends (pipeline test only).**\n")
    lines.append(f"opportunities with both arms: {summary['n_opportunities']} (incomplete: {len(summary['incomplete'])}); "
                 f"rollouts: {summary['n_rollouts']}; spend ${summary['cost_usd']:.3f}; prefix mismatches: "
                 f"{summary['prefix_mismatches']}; control rollouts reproducing the logged outcome: "
                 f"{_f(summary['control_agrees_with_log_per_rollout'])}\n")

    def block(name: str, b: dict[str, Any]) -> None:
        if "confusion" not in b:
            lines.append(f"**{name}**: n = {b.get('n')} ({b.get('note')})\n")
            return
        c = b["confusion"]
        lines.append(f"**{name}** (n = {b['n']}): binary agreement (helps <-> rollout +) {_f(b['agreement'])} "
                     f"[{_f(b['agreement_ci'][0])}, {_f(b['agreement_ci'][1])}], kappa binary {_f(b['kappa_binary'])}, "
                     f"3-class {_f(b['kappa_3class'])}; helps confirmed {_f(b['helps_confirmed'])} ({b['n_helps']}), "
                     f"not-helps confirmed {_f(b['not_helps_confirmed'])} ({b['n_not_helps']}), harms confirmed "
                     f"{_f(b['harms_confirmed'])} ({b['n_harms']}); mean delta helps {_f(b['mean_delta_helps'])}, "
                     f"not-helps {_f(b['mean_delta_not_helps'])}; control reproduces log "
                     f"{_f(b['control_reproduces_log'])} ({b['n_control_checked']})\n")
        lines.append("| one-step label \\ full rollout | + | 0 | − |")
        lines.append("|---|---|---|---|")
        for lab in MG.CF_LABELS:
            lines.append(f"| {lab} | {c[lab]['+']} | {c[lab]['0']} | {c[lab]['-']} |")
        lines.append("")

    if t:
        block("overall", t["overall"])
        for dp, b in t.get("by_dp", {}).items():
            block(f"dp = {dp}", b)
        for k, b in t.get("by_final_step", {}).items():
            block(f"{k} step", b)
    rows = summary.get("rows") or []
    if rows:
        lines.append("| task | step | dp | stratum | one-step (v0 → branch / placebo) | label | logged | with | without | Δ | steps with / without | $ |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r['task_id'].replace('jevgym-', '').replace('-d60-near-strict', '')} | {r['step']} | {r['dp']} | "
                         f"{r.get('stratum')} | {r.get('v0', '')} → {r.get('v_branch', '')} / {r.get('v_placebo', '')} | "
                         f"{r['label']} | {int(bool(r.get('episode_success')))} | {_f(r['p_with'], 2)} | "
                         f"{_f(r['p_without'], 2)} | {_f(r['delta'], 2)} | "
                         f"{','.join(map(str, r['n_steps_with']))} / {','.join(map(str, r['n_steps_without']))} | "
                         f"{_f(r['cost_usd'], 3)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=None, help="run directory holding the measurement arm (experiment.yaml + <arm>.jsonl)")
    ap.add_argument("--arm", default="meta_shadow_s2lite")
    ap.add_argument("--mode", choices=["live", "replay"], default="live",
                    help="live: re-run a small measurement arm and resume its in-process snapshots (default); "
                         "replay: rebuild the states of the existing arm log from the logged action prefixes")
    ap.add_argument("--n-tasks", type=int, default=30, help="live mode: tasks of the measurement arm (N of the suite)")
    ap.add_argument("--task-offset", type=int, default=0,
                    help="live mode: skip the first K suite tasks (fresh tasks for a second run: 30 -> seeds 30..)")
    ap.add_argument("--s1", default='{"type": "heuristic"}',
                    help="live mode: the (free) System-1 of the shadow gates; the one-step labels do not depend on it")
    ap.add_argument("--sample-negatives", type=float, default=0.2, help="live mode: cf.sample_negatives")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--min-helps", type=int, default=10)
    ap.add_argument("--min-not", type=int, default=10)
    ap.add_argument("--min-harms", type=int, default=0,
                    help="of the not-helps stratum, at least this many 'harms' labels (rare; 0 = not forced)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=3.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/dev/cf_validity")
    ap.add_argument("--llm", default=None, help="System-2 spec override (JSON / YAML); simulated -> offline")
    ap.add_argument("--large-llm", default=None, help="large-model spec override (JSON / YAML)")
    ap.add_argument("--dry-run", action="store_true", help="list the subset and the rollouts; spend nothing")
    ap.add_argument("--analyse", default=None, metavar="OUT_DIR", nargs="+",
                    help="only recompute the table of OUT_DIR (several: pooled; summary.json goes to the first one "
                         "as summary_pooled.json)")
    a = ap.parse_args(argv)
    if a.analyse:
        dirs = [Path(x) for x in a.analyse]
        summary = analyse(dirs[0] if len(dirs) == 1 else dirs)
        target = dirs[0] / ("summary.json" if len(dirs) == 1 else "summary_pooled.json")
        summary["pooled_from"] = [str(d) for d in dirs]
        target.write_text(json.dumps(MG._jsonable(summary), indent=1, default=str))
        print(markdown(summary))
        return
    if not a.run:
        raise SystemExit("--run is required (or --analyse OUT_DIR)")
    run_dir, out_dir = Path(a.run), Path(a.out)
    exp = yaml.safe_load((run_dir / "experiment.yaml").read_text()) or {}
    logged = load_trajectories(run_dir / f"{a.arm}.jsonl")
    logged_cfg = (logged[0].get("meta") or {}).get("config") or {}
    spec, large_spec = llm_specs(exp, a.arm)
    if a.llm:
        spec = yaml.safe_load(a.llm)
    if a.large_llm:
        large_spec = yaml.safe_load(a.large_llm)
    simulated = str(spec.get("provider")) in ("simulated", "scripted")
    out_dir.mkdir(parents=True, exist_ok=True)
    facs = MG.suite_factories(run_dir)
    snapshots: dict[str, list[Any]] | None = None
    spent0 = 0.0
    if a.mode == "live" and not a.dry_run:
        from jevharness.decision import make_backend
        from jevharness.harness.agent import JevAgent
        from jevharness.llm import make_llm

        if not large_spec:
            raise SystemExit("live mode needs the large LLM of the routing branch (--large-llm or the arm's measure.cf.large_llm)")
        mcfg = measurement_config(logged_cfg, sample_negatives=a.sample_negatives)
        magent = JevAgent(make_llm(spec), mcfg, s1=make_backend(yaml.safe_load(a.s1)),
                          measure_llms={"large": make_llm(large_spec)}, system_prompt=exp.get("system_prompt"))
        order = list(facs.values())[a.task_offset: a.task_offset + a.n_tasks]   # suite order = seeds, as the runner
        print(f"live measurement arm {mcfg.short()} on {len(order)} tasks; System-2 {spec}; large {large_spec}; "
              f"System-1 {a.s1}", file=sys.stderr)
        trajs, snapshots = live_measurement(magent, order, out_dir / "measure_arm.jsonl", a.workers)
        spent0 = measurement_cost(trajs)
        errs = [t["task_id"] for t in trajs if t.get("error")]
        print(f"measurement arm: {len(trajs)} episodes, success {sum(bool(t['result'].get('success')) for t in trajs)}, "
              f"errors {len(errs)}, CF errors {sum(len(t.get('meta', {}).get('cf_errors') or []) for t in trajs)}, "
              f"spent ${spent0:.3f}", file=sys.stderr)
        if spent0 >= a.budget_usd:
            print("budget exhausted by the measurement arm; no rollouts", file=sys.stderr)
    else:
        trajs = logged
    opps = MG.cf_opportunities(trajs)
    subset = MG.select_cf_validity_subset(opps, n=a.n, min_helps=a.min_helps, min_not=a.min_not, seed=a.seed,
                                          min_harms=a.min_harms)
    units = plan_units(subset, a.reps)
    pool = {}
    for o in opps:
        pool.setdefault((o["dp"], o["label"]), 0)
        pool[(o["dp"], o["label"])] += 1
    print(f"pool: {len(opps)} one-step labels in {a.arm} ({len(trajs)} episodes): "
          + ", ".join(f"{dp}/{lab}={n}" for (dp, lab), n in sorted(pool.items())), file=sys.stderr)
    print(f"subset: {len(subset)} opportunities -> {len(units)} rollouts ({a.reps} reps x with/without)",
          file=sys.stderr)
    for o in subset:
        print(f"  {o['stratum']:9s} {o['task_id']:38s} step {o['step']:2d} {o['dp']:10s} v0={o['v0']:12s} "
              f"branch={o['v_branch']:12s} placebo={o['v_placebo']} label={o['label']} final={o['final_step']}",
              file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subset.json").write_text(json.dumps(MG._jsonable(
        {"run": str(run_dir), "arm": a.arm, "mode": a.mode, "n": a.n, "reps": a.reps, "seed": a.seed,
         "min_helps": a.min_helps, "min_not": a.min_not, "min_harms": a.min_harms,
         "pool": {f"{k[0]}/{k[1]}": v for k, v in pool.items()}, "subset": subset}), indent=1, default=str))
    if a.dry_run:
        print(f"dry run: {len(units)} rollouts planned, nothing spent", file=sys.stderr)
        return
    from jevharness.llm import make_llm
    from jevharness.harness.agent import JevAgent

    if any(o["intervention"] == "large" for o in subset) and not large_spec:
        raise SystemExit("the subset has routing opportunities but no large LLM is known (--large-llm)")
    llm = make_llm(spec)
    large = make_llm(large_spec) if large_spec else None
    cfg = llm_only_config(logged_cfg)
    agent = JevAgent(llm, cfg, system_prompt=exp.get("system_prompt"))
    missing = sorted({o["task_id"] for o in subset} - set(facs))
    if missing:
        raise SystemExit(f"tasks not in the suite of {run_dir}: {missing[:3]}")
    print(f"System-2: {spec}; large: {large_spec}; config: {cfg.short()} max_steps={cfg.max_steps}; "
          f"budget ${a.budget_usd} (spent so far ${spent0:.3f}); source: {a.mode}", file=sys.stderr)
    run = run_units(units, agent, large, facs, out_dir / "rollouts.jsonl", a.budget_usd, a.workers, simulated,
                    snapshots=snapshots, spent0=spent0)
    summary = analyse(out_dir)
    summary["run"] = {**run, "mode": a.mode, "spec": spec, "large_spec": large_spec,
                      "config": cfg.model_dump(mode="json"), "arm": a.arm, "source_run": str(run_dir), "n": a.n,
                      "reps": a.reps, "seed": a.seed, "n_tasks": a.n_tasks if a.mode == "live" else len(trajs),
                      "task_offset": a.task_offset if a.mode == "live" else 0,
                      "s1": a.s1 if a.mode == "live" else None, "measurement_usd": spent0}
    (out_dir / "summary.json").write_text(json.dumps(MG._jsonable(summary), indent=1, default=str))
    print(markdown(summary))
    print(f"\nrun: {run['n_run']} rollouts ({run['n_errors']} errors), rollouts ${run['rollouts_usd']:.3f} + "
          f"measurement ${spent0:.3f} = ${run['spent_usd']:.3f} of ${a.budget_usd} budget"
          f"{' -- STOPPED AT BUDGET' if run['stopped_at_budget'] else ''}; rollouts by model: "
          f"{ {m: round(c, 4) for m, c in run['by_model'].items()} }", file=sys.stderr)


if __name__ == "__main__":
    main()
