#!/usr/bin/env python3
"""Dry run of the post-freeze analysis on SIMULATED runs: every registered arm of the four evaluation configs (E2a,
the Intern-Decision E2a, E2b, E2c) and the two calibration configs runs offline with simulated System-2 LLMs and
simulated System-1 backends on DEVELOPMENT seeds (``suite.seed0 = 0``, a few tasks per arm), the E1 decision run with
simulated deciders, synthetic tau^2 results through the real converter, and synthetic Harbor jobs; then the analysis
chain the paper uses runs on them -- ``scripts/primaries.py`` (C7 selection + the 11-test family), the score cards,
the H2 theory check (``meta_gold.h2_report``), the H5 verdict (``adapters/pi/summarize_harbor.py``) and
``paper_numbers`` -- into ``--out``. It fails loudly (exit 1, every failure listed) on a missing key, a NaN where a
number is expected, or a registered arm that cannot run.

    python scripts/dryrun_main_analysis.py --out /tmp/<scratch>/dryrun [--tasks 6] [--keep]

Never touches the evaluation seeds (refuses to start when ``JEV_EVAL_RUNS`` or ``JEV_ALLOW_PENDING_TAU`` is set), never
calls an API (every LLM / System-1 spec that would is replaced by a simulated one; the lexical, rule and rate
controls run as they are), never writes the shared System-1 cache (``s1_cache: null``) and never edits a frozen YAML
(each config is loaded, overridden in memory and the effective copy is written next to its run by the runner).
The Intern-Decision C1 arms carry ``calibration: pending`` until GPU round 3 fits their tau-hat*; the dry run plays
that fit (drops the key and keeps the placeholder tau) so the arms run -- the state the main run will be in. The
matched-rate placeholders of E2c run by the registered protocol: their rates are read from this run's System-1
arms (``decision.controls.trigger_rates``) and the controls run in a second pass."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import prereg_check as PC  # noqa: E402
from jevharness.bench.runner import arm_bindings, expand_arms, run_experiment  # noqa: E402
from jevharness.harness.config import DP  # noqa: E402

EVAL_CFGS = list(PC.EVAL_CONFIGS)          # jevgym_tool_select, jevgym_meta_control, jevgym_tool_select_intern, baselines
CAL_CFGS = list(PC.CAL_CONFIGS)
OFFLINE_S1 = {"bm25", "tfidf", "heuristic", "prior", "rate", "simulated", "oracle", "random"}
# simulated stand-ins: per-step System-2 error rates and System-1 accuracies (pipeline only, never results)
S2_ERR = {"gemini-3.5-flash": 0.01, "gemini-3.5-flash-lite": 0.06}
S1_ACC = {"jev": 0.93, "intern_decision": 0.86, "llm_generative": 0.88, "embedding": 0.80, "s2_self": 0.90,
          "openjev": 0.85, "calibrated": 0.93}
S1_PRICE = {"jev": 0.042, "intern_decision": 0.01}
C7_DRYRUN_ARM = "jev_ts_cascade_tau1.0"   # the synthetic tau^2 arm standing in for the cascade at the tau-hat* refit
TAU2_ERR_TASK = "2"                        # an airline test task whose 2nd trial ends in an infrastructure error
INJECT = ("jevgym_tool_select.yaml", "jev_ts_assist_k3")   # C2F's Jev arm: one task errored after 3 attempts
PLACEHOLDER_SOURCE = {   # matched-rate placeholder -> (System-1 arm of this run, trigger_rates key) (A4 / A6.18 (f))
    "ratem_stop_lite": ("jev_stop_guard_s2lite", "stop/guard"),
    "ratem_route": ("jev_route_tau0.5", "routing/trigger"),
    "ratem_memw": ("jev_memw", "memory_write/trigger"),
}


class Failures:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, where: str, msg: str) -> None:
        self.items.append(f"{where}: {msg}")
        print(f"[dryrun] FAIL {where}: {msg}", file=sys.stderr)


def log(msg: str) -> None:
    print(f"[dryrun] {msg}", file=sys.stderr, flush=True)


# ====================================================================================== simulated configs
def sim_llm(spec: Mapping[str, Any]) -> dict[str, Any]:
    model = str(spec.get("model", "sim"))
    return {"provider": "simulated", "model": model, "error_rate": S2_ERR.get(model, 0.05),
            "simulated_latency_ms": 1200.0 if "lite" not in model else 600.0}


def sim_s1(spec: Mapping[str, Any], seed: int = 0) -> dict[str, Any]:
    t = str(spec.get("type"))
    if t in OFFLINE_S1:
        return {k: v for k, v in spec.items()}
    return {"type": "simulated", "accuracy": S1_ACC.get(t, 0.85), "latency_ms": 200.0 if t == "jev" else 330.0,
            "price_per_m_input": S1_PRICE.get(t, 0.0), "seed": seed}


def simulate(exp: dict[str, Any], workers: int = 4) -> tuple[dict[str, Any], list[str]]:
    """The registered experiment with simulated models, development seeds and no shared cache; the list of
    dry-run-only changes beyond the model swap (reported)."""
    e = copy.deepcopy(exp)
    notes = []
    e["suite"] = {**e["suite"], "seed0": 0}
    for k in ("llm", "small_llm"):
        if e.get(k):
            e[k] = sim_llm(e[k])
    if e.get("llms"):
        e["llms"] = {n: sim_llm(s) for n, s in e["llms"].items()}
    if e.get("s1"):
        e["s1"] = sim_s1(e["s1"])
    if e.get("s1_backends"):
        e["s1_backends"] = {n: sim_s1(s, seed=i + 1) for i, (n, s) in enumerate(e["s1_backends"].items())}
    e["s1_cache"] = None
    e["workers"] = workers
    for arm in e.get("arms", []):
        for dp in DP:
            d = arm.get(dp.value)
            if isinstance(d, dict) and d.get("calibration") == "pending":
                arm[dp.value] = {k: v for k, v in d.items() if k != "calibration"}
                notes.append(f"{arm['name']}.{dp.value}: `calibration: pending` dropped (plays GPU round 3's tau-hat* "
                             f"fit; tau {d.get('tau')} kept as the stand-in)")
    return e, notes


def check_arm_files(where: str, exp: Mapping[str, Any], run_dir: Path, n: int, F: Failures,
                    allow_skip: Mapping[str, str] | None = None) -> dict[str, int]:
    """Every arm of the config ran its n tasks (one record per task, no error)."""
    counts: dict[str, int] = {}
    for a in expand_arms(dict(exp)):
        p = run_dir / f"{a.name}.jsonl"
        if not p.exists():
            if allow_skip and a.name in allow_skip:
                continue
            F.add(where, f"registered arm `{a.name}` did not run (no {p.name})")
            continue
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        last = {r["task_id"]: r for r in recs}
        counts[a.name] = len(last)
        if len(last) != n:
            F.add(where, f"`{a.name}`: {len(last)} task(s), expected {n}")
        errs = [k for k, r in last.items() if r.get("error")]
        if errs:
            F.add(where, f"`{a.name}`: {len(errs)} episode(s) ended in an error, e.g. {last[errs[0]]['error'][:200]}")
    return counts


# ====================================================================================== stages
def stage_episode_runs(out: Path, n: int, F: Failures, workers: int) -> dict[str, Any]:
    runs = out / "runs"
    info: dict[str, Any] = {}
    os.environ.pop("JEV_ALLOW_PLACEHOLDER", None)
    for cfg in EVAL_CFGS + CAL_CFGS:
        reg = PC.load(cfg)
        exp, notes = simulate(reg, workers)
        t0 = time.time()
        log(f"{cfg}: {len(expand_arms(exp))} arms x {n} tasks (simulated)")
        try:
            run_experiment(exp, runs, max_tasks=n, progress=False)
        except BaseException as e:  # noqa: BLE001 - SystemExit from the seed guard included
            F.add(cfg, f"run_experiment raised {type(e).__name__}: {e}")
            continue
        rd = runs / exp["name"]
        ph = {nm for nm, s in (exp.get("s1_backends") or {}).items() if isinstance(s, dict) and s.get("placeholder")}
        skipped = {a: b["s1"] for a, b in arm_bindings(exp).items() if b.get("s1") in ph}
        if skipped:   # the registered matched-rate protocol: rates from this run's System-1 arms, then a second pass
            from jevharness.decision.controls import trigger_rates

            for nm in sorted(ph):
                src_arm, key = PLACEHOLDER_SOURCE[nm]
                r = trigger_rates(str(rd / f"{src_arm}.jsonl")).get(key, {}).get("rate")
                if r is None or not math.isfinite(float(r)):
                    F.add(cfg, f"matched-rate placeholder `{nm}`: no `{key}` rate in `{src_arm}` (got {r})")
                    r = 0.5
                exp["s1_backends"][nm] = {"type": "rate", "rate": round(float(r), 4), "seed": 0}
                notes.append(f"s1_backends.{nm}: rate {float(r):.4f} from `{src_arm}` `{key}` (A4 protocol, 2nd pass)")
            try:
                run_experiment(exp, runs, max_tasks=n, progress=False)
            except BaseException as e:  # noqa: BLE001
                F.add(cfg, f"second pass raised {type(e).__name__}: {e}")
        counts = check_arm_files(cfg, exp, rd, n, F)
        info[cfg] = {"run_dir": str(rd), "arms": len(counts), "seconds": round(time.time() - t0, 1), "notes": notes}
        log(f"  -> {rd} ({len(counts)} arm files, {time.time() - t0:.0f} s)")
    return info


def stage_e1(out: Path, n_items: int, F: Failures) -> dict[str, Any]:
    from jevharness.bench.decision_eval import run_decision_eval

    reg = yaml.safe_load((ROOT / "configs" / "decision" / "bfcl_when2call.yaml").read_text())
    exp = copy.deepcopy(reg)
    exp["backends"] = {n: sim_s1(s, seed=i + 1) for i, (n, s) in enumerate(reg["backends"].items())}
    exp["fc_llms"] = {}            # native FC needs a real model (not a primary input)
    exp.pop("fc_judge", None)
    exp["s1_cache"] = None
    exp["fc_cache"] = None
    exp["workers"] = 4
    log(f"E1: {len(exp['backends'])} simulated deciders on the first {n_items} items of the C8 population order")
    try:
        summ = run_decision_eval(exp, out / "runs", max_items=n_items)
    except Exception as e:  # noqa: BLE001
        F.add("E1", f"run_decision_eval raised {type(e).__name__}: {e}")
        return {}
    rd = out / "runs" / exp["name"]
    for b in ("jev", "gemini_flash_typed"):
        if not (rd / f"s1_{b}.jsonl").exists():
            F.add("E1", f"C8 backend `{b}` produced no record file")
    k = "jev|tool_select|bfcl"
    if k not in summ or not math.isfinite(float(summ[k].get("acc", float("nan")))):
        F.add("E1", f"summary.json lacks a finite `{k}` accuracy")
    return {"run_dir": str(rd), "n_rows": len(summ)}


def _tau2_results(domain: str, ids: list[str], success: Callable[[str, int], bool], err: Callable[[str, int], bool],
                  cost: float, rng: np.random.Generator) -> dict[str, Any]:
    sims = []
    for tid in ids:
        for trial in range(3):
            ok = success(tid, trial)
            sims.append({
                "id": f"{domain}-{tid}-{trial}", "task_id": tid, "trial": trial, "seed": 300 + trial,
                "reward_info": {"reward": 1.0 if ok else 0.0, "db_check": {"db_match": ok}},
                "termination_reason": "infrastructure_error" if err(tid, trial) else "user_stop",
                "duration": float(rng.uniform(20, 60)), "agent_cost": cost, "user_cost": cost / 10,
                "messages": [{"role": "assistant", "content": "Done.",
                              "usage": {"prompt_tokens": int(rng.integers(3000, 6000)), "completion_tokens": 80},
                              "cost": cost, "generation_time_seconds": 1.1,
                              "raw_data": {"model": "gemini/gemini-3.5-flash"}}]})
    return {"info": {"environment_info": {"domain_name": domain},
                     "agent_info": {"llm": "gemini/gemini-3.5-flash", "implementation": "jev_agent"},
                     "user_info": {"llm": "gemini/gemini-3.5-flash-lite"}}, "simulations": sims}


def stage_tau2(out: Path, F: Failures) -> dict[str, Any]:
    """Synthetic tau2 results.json per (domain, arm) -> the real converter (``suites.tau2.convert_results``) ->
    <out>/runs/tau2/<domain>/<arm>.jsonl, flagged simulated by an experiment.yaml next to them."""
    from jevharness.bench.suites.tau2 import convert_results

    split = json.loads((ROOT / "configs" / "tau2" / "airline_task_ids.json").read_text())
    ids = {"airline": [str(t) for t in split["task_ids"]], "retail": [str(i) for i in range(114)]}
    rng = np.random.default_rng(7)
    base = {d: {t: float(rng.uniform(0.35, 0.95)) for t in ts} for d, ts in ids.items()}
    arms = {"llm_only": 0.0, "jev_shadow": 0.0, "jev_ts_cascade_tau0.95": -0.01, "jev_ts_cascade_tau1.0": -0.02,
            "jev_ts_assist_k3": -0.03, "jev_guard": 0.0}
    root = out / "runs" / "tau2"
    for dom, tids in ids.items():
        dd = root / dom
        dd.mkdir(parents=True, exist_ok=True)
        (dd / "experiment.yaml").write_text(yaml.safe_dump({
            "name": f"tau2_{dom}_synthetic", "llm": {"provider": "simulated", "model": "synthetic-tau2"},
            "s1": {"type": "simulated"}, "note": "dry run: synthetic tau2 results, never results"}))
        for arm, shift in arms.items():
            arm_ids = [t for t in tids if dom != "airline" or arm != "jev_shadow" or t in split["split"]["shadow"]]
            if arm == "jev_shadow" and dom != "airline":
                continue
            seed_off = zlib.crc32(arm.encode()) % 997

            def succ(t: str, k: int, _d: str = dom, _s: float = shift, _o: int = seed_off) -> bool:
                u = np.random.default_rng([int(t) if t.isdigit() else 0, k, _o]).uniform()
                return u < base[_d][t] + _s

            def err(t: str, k: int, _a: str = arm, _d: str = dom) -> bool:   # drop-rule cases (airline test tasks)
                return _d == "airline" and ((_a == "jev_ts_cascade_tau0.95" and t in ("3", "4") and k == 1)
                                            or (_a == C7_DRYRUN_ARM and t == TAU2_ERR_TASK and k == 1))

            res = _tau2_results(dom, arm_ids, succ, err, 0.07 if arm == "llm_only" else 0.05, rng)
            raw = root / "raw" / dom / arm
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "results.json").write_text(json.dumps(res))
            try:
                convert_results(raw / "results.json", arm=arm, out=dd / f"{arm}.jsonl")
            except Exception as e:  # noqa: BLE001
                F.add("tau2", f"convert_results({dom}/{arm}) raised {type(e).__name__}: {e}")
    return {"root": str(root), "arms": list(arms)}


def stage_harbor(out: Path, F: Failures) -> dict[str, Any]:
    """Synthetic Harbor jobs (3 attempts x 20 tasks per arm) -> summarize_harbor's paired H5 verdict."""
    sys.path.insert(0, str(ROOT / "adapters" / "pi"))
    import summarize_harbor as SH  # noqa: PLC0415

    import datetime as dt  # noqa: PLC0415

    jobs = out / "harbor"
    # (success shift, run-time factor): the same per-(task, attempt) draws in every arm, so only the shifts differ
    arms = {"baseline": (0.0, 1.0), "jev_stop_reflect": (0.0, 0.6), "jev_gate_stop": (-0.25, 1.1)}
    start = dt.datetime(2026, 10, 5, 10, 0, tzinfo=dt.timezone.utc)
    for job, (dsucc, tmul) in arms.items():
        for t in range(20):
            p = 0.4 + 0.5 * ((t * 37) % 20) / 20
            for k in range(3):
                d = jobs / job / f"task-{t:02d}__{k}"
                (d / "agent").mkdir(parents=True, exist_ok=True)
                g = np.random.default_rng([t, k])
                ok = g.uniform() < min(1.0, max(0.0, p + dsucc))
                run_s = float(g.uniform(200, 900)) * tmul
                t0 = start.isoformat()
                t1 = (start + dt.timedelta(seconds=run_s)).isoformat()
                (d / "result.json").write_text(json.dumps({
                    "task_name": f"tbsci/task-{t:02d}", "agent_result": {"cost_usd": 0.05, "n_input_tokens": 9000},
                    "verifier_result": {"rewards": {"reward": 1.0 if ok else 0.0}}, "exception_info": None,
                    "agent_execution": {"started_at": t0, "finished_at": t1},
                    "started_at": t0, "finished_at": t1}))
    rows = SH.summarize([jobs / j for j in arms])
    pair = SH.paired(rows, "baseline", n_boot=2000)
    for job, v in pair.items():
        for k in SH.H5_KEYS:
            if k not in v:
                F.add("H5", f"{job}: missing `{k}`")
        for k in ("d_success_lb90", "sign_p_wins", "time_sign_p"):
            x = v.get(k)
            if x is None or (isinstance(x, float) and math.isnan(x)):
                F.add("H5", f"{job}: `{k}` is {x}")
        if v.get("h5_verdict") not in (SH.H5_LABEL_STRICT, SH.H5_LABEL_REGISTERED, "success-preserving", "faster",
                                       "neither"):
            F.add("H5", f"{job}: verdict {v.get('h5_verdict')!r} is not a registered label (A6.6 / A6.16 (j) / "
                        "A6.22 (d) 1)")
    (out / "h5_verdict.json").write_text(json.dumps(pair, indent=1, default=str))
    return {job: v.get("h5_verdict") for job, v in pair.items()}


def stage_inject(out: Path, F: Failures) -> dict[str, Any]:
    """The drop rule's path: one task of C2F's Jev arm ends in an infrastructure error on 3 attempts (the original
    record + the runner's 2 retries, appended as the runner appends them). Idempotent."""
    cfg, arm = INJECT
    p = out / "runs" / PC.load(cfg)["name"] / f"{arm}.jsonl"
    if not p.exists():
        F.add("inject", f"{p} missing")
        return {}
    recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    if any((r.get("meta") or {}).get("dryrun_injected") for r in recs):
        return {"arm": arm, "task": None, "note": "already injected"}
    last = recs[-1]
    bad = {**last, "error": "RuntimeError: simulated 503 (dry run: infrastructure error after retries)",
           "result": {"success": False}, "meta": {**(last.get("meta") or {}), "dryrun_injected": True}}
    with open(p, "a") as f:
        for _ in range(2):
            f.write(json.dumps(bad) + "\n")
    return {"arm": arm, "task": last["task_id"], "attempts": 3}


def stage_calfit(out: Path, F: Failures) -> dict[str, Any]:
    """The A6.5 / A6.11 calibration fit (what GPU round 3 runs for Intern-Decision) on the simulated calibration
    arms: ``threshold_theory --calibration-arm`` at W = $1, lambda = 0 for both decision points."""
    info = {}
    for cfg, arms in (("jevgym_shadow_cal.yaml", ("jev_shadow_cal", "jev_shadow_cal_s2lite")),
                      ("jevgym_shadow_cal_intern.yaml", ("intern_shadow_cal", "intern_shadow_cal_s2lite"))):
        rd = out / "runs" / PC.load(cfg)["name"]
        for arm in arms:
            for dp in ("tool_select", "tool_params"):
                dst = out / "calibration" / f"tauhat_{arm}_{dp}.json"
                dst.parent.mkdir(exist_ok=True)
                r = subprocess.run([sys.executable, "-m", "jevharness.bench.threshold_theory", str(rd), "--arm", arm,
                                    "--dp", dp, "--calibration-arm", "--task-value", "1.0", "--lam-per-hour", "0",
                                    "--format", "json", "--out", str(dst)], capture_output=True, text=True, cwd=ROOT)
                if r.returncode != 0 or not dst.exists():
                    F.add("calfit", f"{arm} {dp}: exit {r.returncode}: {r.stderr[-800:]}")
                    continue
                d = json.loads(dst.read_text())
                if "tau_hat" not in d:
                    F.add("calfit", f"{arm} {dp}: no tau_hat in the fit ({sorted(d)[:8]})")
                    continue
                info[f"{arm}/{dp}"] = d["tau_hat"]
    return info


# ====================================================================================== analysis stages
def stage_primaries(out: Path, n: int, F: Failures, n_boot: int) -> dict[str, Any]:
    runs = out / "runs"
    pdir = out / "primaries"
    common = ["--runs", str(runs), "--tau2", str(runs / "tau2"), "--out", str(pdir), "--n-boot", str(n_boot),
              "--max-tasks", str(n), "--allow-simulated", "--allow-dev-tasks", "--allow-partial-population"]
    py = sys.executable
    r = subprocess.run([py, str(ROOT / "scripts" / "primaries.py"), *common, "--select-c7"], capture_output=True,
                       text=True, cwd=ROOT)
    if r.returncode != 0:
        F.add("primaries --select-c7", f"exit {r.returncode}: {r.stderr[-1500:]}")
        return {}
    sel = json.loads((pdir / "c7_selection.json").read_text())
    for k in ("selected_e2_arm", "reason", "candidates", "tau2_arm_family"):
        if not sel.get(k):
            F.add("C7 selection", f"missing `{k}`")
    fam = str(sel.get("tau2_arm_family") or "")
    c7_arm = (C7_DRYRUN_ARM if fam.startswith("jev_ts_cascade") else fam.split(" ", 1)[0]) or C7_DRYRUN_ARM
    r = subprocess.run([py, str(ROOT / "scripts" / "primaries.py"), *common, "--c7-arm", c7_arm,
                        "--c7-selection", str(pdir / "c7_selection.json"), "--tau2-secondary", "jev_ts_cascade_tau0.95",
                        "--tau2-secondary", "jev_ts_assist_k3"], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        F.add("primaries", f"exit {r.returncode}: {r.stderr[-2000:]}")
        return {}
    res = json.loads((pdir / "primaries.json").read_text())
    check_primaries(res, F)
    return {"c7_arm": c7_arm, "selected": sel.get("selected_e2_arm"), "stderr_tail": r.stderr[-600:],
            "rejected": [k for k, v in res["primaries"].items() if v["holm"]["reject"]]}


REQUIRED_PRIMARY = ("p_decisive", "rule", "decision", "holm")
REQUIRED_CONSTITUENT_NI = ("n_pairs", "n_expected", "n_dropped", "wins", "losses", "diff", "ci95", "p_decisive", "rule",
                           "reading", "full_set", "status_arm", "status_ref", "success_arm", "success_ref")
REQUIRED_CONSTITUENT_SUP = ("n_pairs", "n", "wins", "losses", "diff", "ci95", "p_decisive", "full_set")


def check_primaries(res: Mapping[str, Any], F: Failures) -> None:
    from primaries import CONJ, HOLM_ORDER  # noqa: PLC0415

    if list(res["primaries"]) != HOLM_ORDER or len(res["holm"]) != 11:
        F.add("primaries", f"family is {list(res['primaries'])}, expected the 11 registered labels")
    if not res["meta"].get("simulated"):
        F.add("primaries", "the dry run's inputs are not flagged simulated")
    errs = [f for f in res["flags"] if f["level"] == "ERROR"]
    for f in errs:
        F.add("primaries flag", f"{f['where']}: {f['msg']}")
    for lab, P in res["primaries"].items():
        for k in REQUIRED_PRIMARY:
            if P.get(k) is None:
                F.add(lab, f"missing `{k}`")
        h = P.get("holm") or {}
        for k in ("rank", "alpha_adj", "p_holm"):
            if h.get(k) is None:
                F.add(lab, f"holm.{k} missing / NaN")
        consts = ["jev", "intern"] if lab in CONJ else ["jev"]
        for c in consts:
            blk = P.get(c)
            if not isinstance(blk, dict) or blk.get("missing"):
                F.add(lab, f"constituent `{c}` missing")
                continue
            req = REQUIRED_CONSTITUENT_NI if P["test"] == "NI" else REQUIRED_CONSTITUENT_SUP if P["test"] == "Sup." \
                else ("n_pairs", "diff", "ci95", "p_decisive", "diff_net", "ci95_net", "full_set", "rate_arm",
                      "rate_control")
            for k in req:
                v = blk.get(k)
                if v is None:
                    F.add(lab, f"{c}.{k} missing or NaN")
                elif isinstance(v, list) and any(x is None for x in v):
                    F.add(lab, f"{c}.{k} has a NaN entry: {v}")
            # a NaN MDE is allowed only with its reason (no discordant task; power out of reach at this n)
            if blk.get("mde") is None and not blk.get("mde_note"):
                F.add(lab, f"{c}.mde is NaN without an mde_note ({blk.get('wins')} wins / {blk.get('losses')} losses)")
        if lab in CONJ and not P.get("replication"):
            F.add(lab, "no replication label")
    c2 = (res["primaries"]["C2F"].get("jev") or {})
    if (c2.get("status_arm") or {}).get("errored") != 1 or c2.get("n_dropped") != 1:
        F.add("drop rule", f"C2F jev: injected errored task not dropped (status {c2.get('status_arm')}, "
                           f"dropped {c2.get('n_dropped')})")
    elif (c2.get("full_set") or {}).get("n") != c2.get("n_expected"):
        F.add("drop rule", "C2F jev: the full-set row does not cover every expected task")
    c7 = (res["primaries"]["C7"].get("jev") or {})
    if (c7.get("status_arm") or {}).get("errored", 0) < 1:
        F.add("drop rule", f"C7: the errored tau^2 trial did not drop its task (status {c7.get('status_arm')})")
    c4 = res["primaries"]["C4"].get("opportunities") or {}
    if c4.get("n_pos") is None:
        F.add("C4", "the routing-positive count was not read")


def stage_scorecards(out: Path, F: Failures, n_boot: int) -> dict[str, Any]:
    from jevharness.bench import scorecard as SC  # noqa: PLC0415
    from jevharness.bench.report import report  # noqa: PLC0415

    runs = out / "runs"
    e2b = runs / PC.load("jevgym_meta_control.yaml")["name"]
    specs = {
        "jevgym_tool_select.yaml": {},
        "jevgym_tool_select_intern.yaml": {},
        "jevgym_meta_control.yaml": {"measure_arms": ["meta_shadow_s2lite", "meta_shadow"]},
        "baselines_jevgym.yaml": {"pairs": [("jev_reflect_trigger_s2lite", "ratem_reflect_trigger_s2lite"),
                                            ("jev_ts_assist_k3", "bm25_ts_assist_k3"), ("jev_route_tau0.5", "ratem_route"),
                                            ("jev_memw", "ratem_memw"),
                                            ("jev_stop_guard_s2lite", "ratem_stop_guard_s2lite")],
                                  "measure_arms": ["meta_shadow_s2lite", "meta_shadow"], "measure_run": e2b,
                                  "task_values": (0.1, 1.0, 10.0)},
    }
    info = {}
    for cfg, kw in specs.items():
        rd = runs / PC.load(cfg)["name"]
        if not rd.exists():
            F.add("scorecard", f"{rd} missing")
            continue
        try:
            rs = SC.load_runs(rd)
            mr = None
            if kw.get("measure_run"):
                mr = {a: t for a, t in SC.load_runs(kw["measure_run"]).items() if a in kw["measure_arms"]}
            card = SC.compute_scorecard(rs, baseline="llm_only", meta=SC.run_meta(rd, rs), n_boot=n_boot,
                                        criterion="noninferior", margin=0.03, pairs=kw.get("pairs"),
                                        task_values=kw.get("task_values", (1.0,)), measure_arms=kw.get("measure_arms"),
                                        measure_runs=mr)
            md = SC.render(card, "md")
            tex = SC.render(card, "latex")
            (rd / "scorecard.md").write_text(md)
            (rd / "scorecard.tex").write_text(tex)
            (rd / "report.md").write_text(report(rd))
        except Exception as e:  # noqa: BLE001
            import traceback

            F.add("scorecard", f"{cfg}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            continue
        for a, row in card["arms"].items():
            for k in ("success", "cost_usd", "latency_s"):
                if not isinstance(row.get(k), (int, float)) or math.isnan(row.get(k)):
                    F.add("scorecard", f"{cfg} `{a}`.{k} = {row.get(k)}")
            vb = row.get("vs_baseline")
            if vb and not row.get("shadow_only"):
                ni = vb.get("ni") or {}
                for k in ("p_decisive", "rule", "ni_holm"):
                    if ni.get(k) is None:
                        F.add("scorecard", f"{cfg} `{a}`.vs_baseline.ni.{k} missing")
                for k, v in (("delta", vb.get("delta")), ("ni.p_decisive", ni.get("p_decisive")),
                             ("ni.p_ni_holm", ni.get("p_ni_holm"))):
                    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                        F.add("scorecard", f"{cfg} `{a}`.vs_baseline.{k} = {v}")
        for pr, v in (card["headline"].get("pairs") or {}).items():
            if not v.get("n_paired"):
                F.add("scorecard", f"{cfg} pair {pr}: no paired task")
        info[cfg] = {"arms": len(card["arms"]), "notes": card["meta"]["notes"][:5]}
    return info


def stage_h2(out: Path, F: Failures, n_boot: int) -> dict[str, Any]:
    from jevharness.bench import meta_gold as MG  # noqa: PLC0415
    from jevharness.bench.scorecard import load_runs  # noqa: PLC0415

    runs = out / "runs"
    e2b = runs / PC.load("jevgym_meta_control.yaml")["name"]
    e2c = runs / PC.load("baselines_jevgym.yaml")["name"]
    measure = ["meta_shadow_s2lite", "meta_shadow"]
    rs = {**load_runs(e2c), **{a: t for a, t in load_runs(e2b).items() if a in measure}}
    pairs = [("jev_reflect_trigger_s2lite", "ratem_reflect_trigger_s2lite"), ("jev_route_tau0.5", "ratem_route"),
             ("jev_memw", "ratem_memw"), ("jev_stop_guard_s2lite", "ratem_stop_guard_s2lite")]
    try:
        cells = MG.h2_cells(rs, measure, pairs, Ws=(0.1, 1.0, 10.0), n_boot=n_boot)
        rep = MG.h2_report(cells, Ws=(0.1, 1.0, 10.0), n_boot=n_boot)
    except Exception as e:  # noqa: BLE001
        import traceback

        F.add("H2", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        return {}
    (out / "h2_report.json").write_text(json.dumps(MG._jsonable(rep), indent=1, default=str))
    if len(cells) != len(pairs):
        F.add("H2", f"{len(cells)} cells for {len(pairs)} pairs")
    for k in ("h2_theory", "h2_verdict", "h2_regression_exploratory", "predicted_gain", "cells"):
        if k not in rep:
            F.add("H2", f"report lacks `{k}`")
    t1 = rep["h2_theory"].get("1")
    for k in ("estimable", "reason", "cells", "n_cells", "mae", "share_inside_predicted_ci"):
        if t1 is None or k not in t1:
            F.add("H2", f"h2_theory['1'] lacks `{k}`")
    return {"verdict": rep["h2_verdict"], "n_cells": rep["n_cells"], "n_estimable": rep["n_estimable"]}


def stage_paper_numbers(out: Path, F: Failures) -> dict[str, Any]:
    from jevharness.bench import paper_numbers as PN  # noqa: PLC0415

    gen = out / "generated"
    gen.mkdir(exist_ok=True)
    values, prov = PN.collect(out / "runs", include_simulated=True, n_boot=500)
    (gen / "results.tex").write_text(PN.to_tex(values, prov))
    (gen / "results.json").write_text(json.dumps({"provenance": prov, "values": values}, indent=1, sort_keys=True))
    for p in prov:
        if "error" in p:
            F.add("paper_numbers", f"{p['run']}: {p['error']}")
        elif not p.get("simulated"):
            F.add("paper_numbers", f"{p['run']}: a dry-run directory is not flagged simulated")
    bad = [k for k in values if not k.startswith("SIM/")]
    if bad:
        F.add("paper_numbers", f"{len(bad)} key(s) without the SIM/ prefix, e.g. {bad[:3]}")
    nan = [k for k, v in values.items() if v.lower() in ("nan", "inf", "-inf")]
    if nan:
        F.add("paper_numbers", f"{len(nan)} NaN / inf value(s) rendered, e.g. {nan[:3]}")
    # the generated macro file must define the keys the paper would read from the E2 runs
    e2a = PC.load("jevgym_tool_select.yaml")["name"]
    for k in (f"SIM/{e2a}/llm_only/success", f"SIM/{e2a}/jev_ts_cascade_tauhat/vs_baseline/delta"):
        if k not in values:
            F.add("paper_numbers", f"expected key `{k}` missing")
    return {"n_values": len(values), "runs": len(prov)}


# ====================================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="scratch directory (created; never under runs/ or paper/)")
    ap.add_argument("--tasks", type=int, default=6, help="tasks per arm (development seeds 0..tasks-1)")
    ap.add_argument("--e1-items", type=int, default=400, help="E1 items (stratified prefix of the C8 population order)")
    ap.add_argument("--n-boot", type=int, default=500, help="bootstrap replicates for the analysis stages")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep an existing --out (resume) instead of wiping it")
    ap.add_argument("--only", default="", help="comma-separated stages to run (runs,inject,e1,tau2,harbor,calfit,primaries,"
                                               "scorecards,h2,paper_numbers); default all")
    a = ap.parse_args(argv)
    for env in ("JEV_EVAL_RUNS", "JEV_ALLOW_PENDING_TAU"):
        if os.environ.get(env):
            raise SystemExit(f"refusing to dry-run with {env} set: the dry run never touches evaluation seeds or "
                             "pending thresholds")
    out = Path(a.out).resolve()
    for bad in (ROOT / "runs", ROOT / "paper", ROOT / "results", ROOT / "configs"):
        if out == bad or bad in out.parents:
            raise SystemExit(f"--out {out} lies inside {bad}: use a scratch directory")
    if out.exists() and not a.keep and not a.only:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in a.only.split(",") if s] or ["runs", "e1", "tau2", "harbor", "calfit", "primaries", "scorecards",
                                                      "h2", "paper_numbers"]
    F = Failures()
    summary: dict[str, Any] = {"out": str(out), "tasks": a.tasks, "stages": {}}
    t0 = time.time()
    if "runs" in stages:
        summary["stages"]["runs"] = stage_episode_runs(out, a.tasks, F, a.workers)
    if "runs" in stages or "inject" in stages:
        summary["stages"]["inject"] = stage_inject(out, F)
    if "e1" in stages:
        summary["stages"]["e1"] = stage_e1(out, a.e1_items, F)
    if "tau2" in stages:
        summary["stages"]["tau2"] = stage_tau2(out, F)
    if "harbor" in stages:
        summary["stages"]["h5"] = stage_harbor(out, F)
    if "calfit" in stages:
        summary["stages"]["calfit"] = stage_calfit(out, F)
    if "primaries" in stages:
        summary["stages"]["primaries"] = stage_primaries(out, a.tasks, F, a.n_boot)
    if "scorecards" in stages:
        summary["stages"]["scorecards"] = stage_scorecards(out, F, a.n_boot)
    if "h2" in stages:
        summary["stages"]["h2"] = stage_h2(out, F, a.n_boot)
    if "paper_numbers" in stages:
        summary["stages"]["paper_numbers"] = stage_paper_numbers(out, F)
    summary["seconds"] = round(time.time() - t0, 1)
    summary["failures"] = F.items
    (out / "dryrun_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"{len(F.items)} failure(s) in {summary['seconds']} s -> {out / 'dryrun_summary.json'}")
    for f in F.items:
        print(f"FAIL {f}")
    return 1 if F.items else 0


if __name__ == "__main__":
    sys.exit(main())
