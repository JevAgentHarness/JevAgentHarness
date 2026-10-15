"""Per-arm *score card* for a JevAgentBench run directory (one ``<arm>.jsonl`` of trajectories per arm).

    python -m jevharness.bench.scorecard RUN_DIR --baseline llm_only [--weak ts_params_replace] \
        [--sweep ts_cascade_tau ...] [--format md|latex|json] [--alpha 0.05] \
        [--criterion not_worse|noninferior --margin 0.02] [--n-boot 2000] [--linear-cost] [--single-baseline] \
        [--pair ARM:CONTROL ...] [--task-value 0.1 1 10] [--lam-per-hour 0] [--out FILE]

References: an arm is compared with the LLM-only arm of its own System-2 model (a comparison family is one suite
x System-2 model, docs/BENCHMARK_SPEC.md): with ``llm_only`` (Flash) and ``llm_only_s2lite`` (Flash-Lite) in one
run, every Flash-Lite arm is paired with ``llm_only_s2lite`` (flag ``vs llm_only_s2lite``), Holm and iso-quality
run per family, and ``llm_only_s2lite`` itself is compared with ``llm_only``. ``--single-baseline`` restores the
single-reference card.

Arm-vs-control (``--pair``, repeatable): the pre-registered comparisons that are not "vs. LLM-only" -- C3
``--pair jev_ts_assist_k3:bm25_ts_assist_k3``, C5 ``--pair jev_reflect_trigger_s2lite:ratem_reflect_trigger_s2lite``
-- with paired Delta-success, McNemar, Delta cost / latency and Delta-utility at each ``--task-value`` W.

For every arm the card reports, next to success with a task-level bootstrap CI (always together with
cost and latency -- see docs/BENCHMARK_SPEC.md for the exact definitions and references):

  vs. baseline    paired Delta-success with CI, one-sided paired-bootstrap p (worse), Holm-adjusted,
                  McNemar p, cost / latency savings, LLM-call ratio ("strong-call fraction")
  efficiency      cost-of-pass, latency-of-pass (Erol et al. 2025), effectiveness under a cost budget
                  (SWE-Effi), pass^k when tasks have several trials (tau-bench)
  frontier        Pareto membership, dominance counts, exclusive hypervolume contribution
  routing         PGR vs. the weak (System-1-only) and strong (baseline) references (RouteLLM)
  decisions       delegation ratio, LLM-free steps, System-1 accuracy / ECE / adaptive ECE / Brier /
                  AUROC / AURC / E-AURC / selective accuracy, harmful-delegation rate, guard FAR/FRR, rescue

and run-level headlines: iso-quality cost savings, quality at iso-cost, frontier cost-of-pass,
hypervolume, and per *sweep* (arms named ``<prefix><value>``, e.g. ``ts_cascade_tau0.9``) APGR, CPT(50%),
CPT(80%) and AIQ against a "zero router" that randomly mixes the two references. The references are the
tau -> 0 / tau -> 1 limits of a cascade, so a sweep's PGR curve and AIQ hull include them as anchors;
AIQ uses linear cost (RouterBench: random per-episode mixing is linear in cost), the hypervolume log10 cost
unless ``--linear-cost``.

Trials: episodes whose ``task_id`` ends in ``#<k>`` (tau2 convention) or that carry ``meta.tau2.task_id``
are trials of one task; the task is the resampling unit of every bootstrap. Runs whose experiment uses a
simulated LLM / System-1 are flagged: such numbers are for pipeline tests and labelled simulation
studies only, never headline results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..harness.config import DP
from ..harness.trajectory import load_trajectories
from .report import CACHE_KEYS, episode_cache_columns
from .metrics import (
    aiq, apgr, aurc, auroc, binary_calibration, bootstrap_ci, brier, cf_efficacy, choice_calibration_data,
    cost_of_pass, cpt, decision_records, delegation_ratio, noul_calibration_data,
    dominance_counts, eaurc, ece, ece_adaptive, effectiveness_under_budget, episode_table, frontier_cost_of_pass,
    gate_stats, gold_coverage, guard_error_rates, harmful_delegation, holm, ni_cells, ni_holm_levels, ni_pvalue,
    newcombe_interval, tango_interval, hypervolume_2d, hypervolume_contributions,
    is_delegated, iso_quality_savings, latency_of_pass, llm_free_step_ratio, mcnemar, memory_read_hits,
    paired_bootstrap_diff, param_kind_composition, pareto_front, pass_hat_k_curve, pgr, probe_error_rates,
    quality_at_iso_cost, recall_at_k, rescue_rate, selective_accuracy, tost_paired,
)

TRIAL_RE = re.compile(r"#\d+$")
SWEEP_RE = re.compile(r"^(?P<prefix>.+?_(?P<param>[A-Za-z]+))(?P<value>-?\d+(?:\.\d+)?)$")
SIM_S1_TYPES = {"simulated", "oracle", "random"}
SIM_LLM_PROVIDERS = {"simulated", "scripted"}
NAN = float("nan")
DP_NAMES = tuple(dp.value for dp in DP)


# ------------------------------------------------------------------ loading
def dedupe(trajs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the last record per ``task_id`` (a task re-run after an interrupted write must not count as an
    extra trial; genuine trials carry distinct ``#<k>`` ids). Returns (records, #dropped)."""
    last: dict[str, Mapping[str, Any]] = {}
    for t in trajs:
        last[str(t.get("task_id"))] = t
    return [dict(t) for t in last.values()], len(trajs) - len(last)


def load_runs(run_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """arm name (file stem) -> trajectories (deduplicated by ``task_id``), for every ``*.jsonl`` in the run
    directory."""
    return {p.stem: dedupe(load_trajectories(p))[0] for p in sorted(Path(run_dir).glob("*.jsonl"))}


def run_meta(run_dir: str | Path, runs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Experiment metadata (from ``experiment.yaml`` written by the runner) and whether it is simulated."""
    meta: dict[str, Any] = {"run_dir": str(run_dir)}
    exp_path = Path(run_dir) / "experiment.yaml"
    exp: dict[str, Any] = {}
    if exp_path.exists():
        try:
            exp = yaml.safe_load(exp_path.read_text()) or {}
        except yaml.YAMLError:
            exp = {}
    if exp:
        meta.update({"name": exp.get("name"), "suite": exp.get("suite"),
                     "llm": (exp.get("llm") or {}).get("model"), "s1": (exp.get("s1") or {}).get("type")})
    specs = [exp.get("llm") or {}, exp.get("small_llm") or {}, *((exp.get("llms") or {}).values())]
    s1specs = [exp.get("s1") or {}, *((exp.get("s1_backends") or {}).values())]
    sim = any(str(s.get("provider", "")).lower() in SIM_LLM_PROVIDERS for s in specs if isinstance(s, dict)) or \
        any(str(s.get("type", "")).lower() in SIM_S1_TYPES for s in s1specs if isinstance(s, dict))
    for trajs in (runs or {}).values():
        for t in trajs[:5]:
            sim = sim or str(t.get("backend") or "").lower() in SIM_S1_TYPES
    meta["simulated"] = bool(sim)
    return meta


def _shadow_only(trajs: Sequence[Mapping[str, Any]]) -> bool:
    """True when System-1 only ever ran in shadow mode in this arm (logged, never acted) -- every measurement arm
    (shadow decision points, counterfactual branches, guard probes; docs/META_GOLD_SPEC.md §5.1, §6.4) is one, so it
    is excluded from Holm, iso-quality, Pareto, hypervolume and frontier cost-of-pass, and is never a reference.
    With a logged harness config (``meta.config``) every decision point must be ``off`` or ``shadow`` -- a live
    non-System-1 mode (reflection ``always``, memory ``always`` / ``llm``) makes the arm a reference / control even if
    it carries probes; older logs fall back to the logged decision modes."""
    cfgs = [c for c in ((t.get("meta") or {}).get("config") for t in trajs[:5]) if isinstance(c, Mapping)]
    if cfgs:
        modes = {str((c.get(dp) or {}).get("mode", "off")) for c in cfgs for dp in DP_NAMES
                 if isinstance(c.get(dp, {}), Mapping)}
        return "shadow" in modes and modes <= {"off", "shadow"}
    modes = {r.get("mode") for r in decision_records(trajs)}  # type: ignore[arg-type]
    if modes - {"shadow", "off", None}:
        return False
    return "shadow" in modes


def arm_s2_model(trajs: Sequence[Mapping[str, Any]]) -> str | None:
    """The arm's System-2 model: the most common main LLM (``traj.llm``) of its episodes."""
    c = Counter(str(t.get("llm")) for t in trajs if t.get("llm"))
    return c.most_common(1)[0][0] if c else None


def is_llm_only_arm(trajs: Sequence[Mapping[str, Any]]) -> bool:
    """Every decision point ``off`` (the LLM-only harness, whatever its System-2 model): read from the logged
    harness config (``meta.config``); older logs without it count when they made no System-1 call and logged
    only ``off`` decisions."""
    cfgs = [(t.get("meta") or {}).get("config") for t in trajs[:5]]
    cfgs = [c for c in cfgs if isinstance(c, Mapping)]
    if cfgs:
        return all((c.get(dp) or {}).get("mode", "off") == "off" for c in cfgs for dp in DP_NAMES
                   if isinstance(c.get(dp, {}), Mapping))
    no_s1 = all(not ((t.get("usage") or {}).get("s1") or {}).get("calls") for t in trajs)
    plain_calls = all(c.get("decision_point") in (None, "tool_select", "tool_params", "action")
                      for t in trajs for c in (t.get("calls") or []) if c.get("component") == "s2")
    return no_s1 and plain_calls and {r.get("mode") for r in decision_records(trajs)} <= {"off"}  # type: ignore[arg-type]


def family_references(runs: Mapping[str, Sequence[Mapping[str, Any]]], baseline: str | None) -> dict[str, str]:
    """arm -> the LLM-only reference it is compared against: the LLM-only arm with the arm's own System-2 model
    (docs/BENCHMARK_SPEC.md: a comparison family is one suite x System-2 model), so a Flash-Lite System-2 arm
    (``*_s2lite``) pairs with ``llm_only_s2lite``, not with the Flash ``llm_only``. Among several LLM-only arms
    of one model, names starting with the baseline's name win (``llm_only_s2lite`` before ``small_only``).
    Arms whose model has no LLM-only arm fall back to ``baseline``; a family's own reference arm is compared
    with ``baseline`` (the System-2 strength contrast)."""
    if not baseline or baseline not in runs:
        return {}
    model = {a: arm_s2_model(t) for a, t in runs.items()}
    mb = model[baseline]
    if mb is None:
        return {a: baseline for a in runs}
    refs: dict[str, str] = {mb: baseline}
    cands = sorted((a for a, t in runs.items() if a != baseline and model[a] not in (None, mb) and t
                    and is_llm_only_arm(t)), key=lambda a: (not a.startswith(baseline), len(a), a))
    for a in cands:
        refs.setdefault(model[a], a)  # type: ignore[arg-type]
    out = {}
    for a in runs:
        r = refs.get(model[a], baseline) if model[a] is not None else baseline
        out[a] = baseline if r == a else r
    return out


def task_key(t: Mapping[str, Any]) -> str:
    """The task an episode belongs to (trials of one task share it): ``meta.tau2`` if present, else the
    ``task_id`` without a trailing ``#<trial>``."""
    tau = (t.get("meta") or {}).get("tau2") or {}
    if tau.get("task_id") is not None:
        return f"{tau.get('domain')}/{tau.get('task_id')}"
    return TRIAL_RE.sub("", str(t.get("task_id")))


def strong_calls(t: Mapping[str, Any]) -> float:
    """System-2 calls to the episode's main (strong) LLM ``t["llm"]``: routing arms also call a small LLM,
    which counts as an LLM call but not as a *strong* call (the x-axis of RouteLLM's PGR curves). Falls back
    to all System-2 calls when the per-call log is missing."""
    calls = t.get("calls")
    if calls and t.get("llm"):
        # measurement calls (shadow / counterfactual / probe generation, ``meta.shadow``) never count: they run after
        # the episode and are kept out of usage.s2 too
        return float(sum(1 for c in calls if c.get("component") == "s2" and c.get("model") == t.get("llm")
                         and not (c.get("meta") or {}).get("shadow")))
    return float(((t.get("usage") or {}).get("s2") or {}).get("calls", 0) or 0)


def s1_cached_frac(trajs: Sequence[Mapping[str, Any]]) -> float:
    """Share of an arm's acting System-1 calls served from the decision cache (``calls[].cached``: the stored answer is
    replayed with its original cost and latency; PREREGISTRATION A6.22 (e)); measurement (shadow) calls are left out.
    NaN when the arm made no System-1 call."""
    rs = [c for t in trajs for c in (t.get("calls") or []) if c.get("component") == "s1"
          and not (c.get("meta") or {}).get("shadow")]
    return sum(1 for c in rs if c.get("cached")) / len(rs) if rs else NAN


def _episode_values(t: Mapping[str, Any]) -> dict[str, float]:
    u = t.get("usage") or {}
    tot = u.get("total") or {}
    return {"success": float(bool((t.get("result") or {}).get("success"))),
            "cost": float(tot.get("cost_usd", 0.0) or 0.0),
            "s1_cost": float((u.get("s1") or {}).get("cost_usd", 0.0) or 0.0),
            "latency": float(tot.get("latency_ms", 0.0) or 0.0) / 1000.0,
            "llm_calls": float((u.get("s2") or {}).get("calls", 0) or 0),
            "strong_calls": strong_calls(t)}


def drop_errored_tasks(trajs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """The drop rule of PREREGISTRATION §1 / A6.9 (paired complete case): a task whose latest episode (of any trial)
    ended in an infrastructure error (``error`` set) leaves every paired comparison of that arm, so its episodes are
    removed before the per-task maps are built (agent-caused failures carry no ``error`` and stay failures). Returns
    (the kept episodes, the dropped task keys). ``scripts/primaries.py`` applies the same rule to the primaries and
    adds the full-set sensitivity row (an errored task counted as a failure)."""
    bad = sorted({task_key(t) for t in trajs if t.get("error")})
    if not bad:
        return [dict(t) for t in trajs], []
    drop = set(bad)
    return [dict(t) for t in trajs if task_key(t) not in drop], bad


def arm_arrays(trajs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-task means (task = resampling unit) and per-episode outcomes of one arm."""
    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    episodes: dict[str, float] = {}
    for t in trajs:
        v = _episode_values(t)
        by_task[task_key(t)].append(v)
        episodes[str(t.get("task_id"))] = v["success"]
    keys = ("success", "cost", "s1_cost", "latency", "llm_calls", "strong_calls")
    means = {k: {m: float(np.mean([e[m] for e in es])) for m in keys} for k, es in by_task.items()}
    return {**{m: {k: v[m] for k, v in means.items()} for m in keys},
            "trials": {k: [e["success"] for e in es] for k, es in by_task.items()},
            "cost_trials": {k: [(e["cost"], e["success"]) for e in es] for k, es in by_task.items()},
            "episodes": episodes}


def detect_sweeps(names: Sequence[str], only: Sequence[str] | None = None, param: str | None = None,
                  min_points: int | None = None) -> dict[str, list[tuple[float, str]]]:
    """Group arms named ``<prefix><number>`` (the runner names sweep arms ``<base>_<param><value>``,
    e.g. ``ts_cascade_tau0.9``) by prefix. ``only`` restricts to the given prefixes, ``param`` to one
    parameter name (``tau``); groups with fewer than ``min_points`` arms are dropped (default: 2 for
    explicitly requested prefixes, 3 for auto-detected groups, where two arms rarely form a sweep)."""
    if min_points is None:
        min_points = 2 if only else 3
    groups: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for n in names:
        m = SWEEP_RE.match(n)
        if not m or (param and m.group("param") != param):
            continue
        if only and m.group("prefix") not in only:
            continue
        groups[m.group("prefix")].append((float(m.group("value")), n))
    return {p: sorted(v) for p, v in sorted(groups.items()) if len(v) >= max(1, min_points)}


# ------------------------------------------------------------------ decision-level block
def decision_block(trajs: Sequence[Mapping[str, Any]], coverage: float = 0.8) -> dict[str, Any]:
    """Decision-level metrics of one arm. Calibration: System-1 ``Choice`` decisions with a gold label (argmax
    answer + top-label confidence) and, separately (``noul``), yes/no guard / trigger decisions with a binary
    gold, calibrated as forecasts P(yes) of the gold (``metrics.binary_calibration``) -- their executed value
    is thresholded at tau, so the logged top-label confidence must not be paired with it."""
    recs = decision_records(trajs)  # type: ignore[arg-type]
    episodes = {str(t.get("task_id")): bool((t.get("result") or {}).get("success")) for t in trajs}
    out: dict[str, Any] = {"n_decisions": len(recs)}
    dr = delegation_ratio(recs)
    out.update({"delegation_ratio": dr["ratio"], "delegation_by_dp": dr["by_dp"], "n_object_decisions": dr["n"],
                "llm_free_steps": llm_free_step_ratio(trajs)})
    conf, corr = choice_calibration_data(recs)
    if conf:
        out.update({"n_s1_scored": len(conf), "s1_acc": float(np.mean(corr)), "ece": ece(conf, corr),
                    "ace": ece_adaptive(conf, corr), "brier": brier(conf, corr), "auroc": auroc(conf, corr),
                    "aurc": aurc(conf, corr), "eaurc": eaurc(conf, corr),
                    f"sel_acc@{int(round(100 * coverage))}": selective_accuracy(conf, corr, coverage)})
    pys, ys = noul_calibration_data(recs)
    if pys:
        out["noul"] = binary_calibration(pys, ys)
    # memory gates log no DecisionLog (no gold on JevGym): report what the episodes record
    mem = [(t.get("meta") or {}).get("memory") or {} for t in trajs]
    offered = sum(int(m.get("offered", 0) or 0) for m in mem)
    if offered:
        out["memory_stored_frac"] = sum(int(m.get("stored", 0) or 0) for m in mem) / offered
    h = harmful_delegation(recs, episodes)
    if h["n_delegated"]:
        out["harmful"] = h
    g = guard_error_rates(recs)
    if g:
        out["guard"] = g
        out["guard_canonical"] = guard_error_rates(recs, gold="canonical")
        out["rescue"] = rescue_rate(recs)
    gold = gold_block(trajs, recs)
    if gold:
        out["gold"] = gold
    out["coverage"] = call_coverage(trajs, recs)
    return out


# steps whose ``args`` are not the arguments of a tool call: action-only envs (``act``) and tau^2 text replies
_NOT_A_TOOL_CALL = ("act", "respond_to_user")


def call_coverage(trajs: Sequence[Mapping[str, Any]],
                  recs: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The coverage quantities of PREREGISTRATION A6.22 (a) for one arm, kept apart because they answer different
    questions (review round 8, W5 and D.3 item 8: "coverage is overloaded"):

    ``tool_coverage``          share of the tool-selection opportunities (one per agent step; a guarded step once) on
                               which System-1's tool was executed (``metrics.is_delegated``; = ``delegation_by_dp``'s
                               tool_select entry; 0 for an always-defer or LLM-only arm);
    ``arg_coverage``           share of the argument slots of the executed tool calls -- every argument of every
                               executed call, whoever wrote the call, free text included -- whose value System-1 chose
                               (delegated ``tool_params`` records / slots). After a deferred tool choice the LLM writes
                               the whole call and the harness logs no ``tool_params`` record, so those slots enter the
                               denominator only; a finish the stop guard rejected was never executed and is skipped;
    ``arg_coverage_s1_calls``  the same share inside the calls System-1 chose (``delegation_by_dp``'s tool_params entry:
                               the asked bounded / copy arguments and the free-text ``generate`` records); NaN when
                               System-1 chose no call, i.e. when the argument thresholds were never reachable;
    ``llm_free_steps``         share of agent steps without any System-2 call (the steps whose native call was avoided);
    ``llm_calls_per_episode``  mean System-2 calls per episode (``usage.s2.calls``).

    The paired counterpart -- native LLM calls avoided per episode versus the arm's LLM-only reference -- is
    ``vs_baseline.llm_calls_avoided`` / ``llm_calls_avoided_frac`` (:func:`compute_scorecard`). ``delegating`` says
    whether the arm runs a delegating mode (replace / cascade) at an object-level decision point."""
    recs = list(recs) if recs is not None else decision_records(trajs)  # type: ignore[arg-type]
    dr = delegation_ratio(recs)
    slots = 0
    for t in trajs:
        for s in t.get("steps") or []:
            args = s.get("args")
            if s.get("tool") in (None, *_NOT_A_TOOL_CALL) or not isinstance(args, Mapping):
                continue
            if any(d.get("dp") == "stop" and d.get("final_value") is False and (
                    d.get("mode") == "guard" or (d.get("mode") in ("cascade", "replace") and d.get("s2_value") is True))
                   for d in s.get("decisions") or []):
                continue      # a finish the stop guard (or the E6b stop cascade / E6e stop replace) rejected
            slots += len(args)
    n_arg = sum(1 for r in recs if r.get("dp") == "tool_params" and is_delegated(r))
    n_ts = dr["by_dp"].get("tool_select")
    calls = [float(((t.get("usage") or {}).get("s2") or {}).get("calls", 0) or 0) for t in trajs]
    return {"tool_coverage": float(n_ts) if n_ts is not None else NAN,
            "arg_coverage": n_arg / slots if slots else NAN, "n_arg_slots": slots, "n_arg_delegated": n_arg,
            "arg_coverage_s1_calls": float(dr["by_dp"].get("tool_params", NAN)),
            "llm_free_steps": llm_free_step_ratio(trajs),
            "llm_calls_per_episode": float(np.mean(calls)) if calls else NAN,
            "delegating": any(r.get("dp") in ("tool_select", "tool_params", "action")
                              and r.get("mode") in ("replace", "cascade") for r in recs)}


def gold_block(trajs: Sequence[Mapping[str, Any]], recs: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Meta-decision gold readouts of one arm (docs/META_GOLD_SPEC.md §9.7): gold coverage and ``gold_status``
    counts per decision point; per gate (shadow / trigger / guard records with a binary gold) pi, AUROC, J, accuracy
    and the "no opportunity" flag (< 30 positive or negative golds); counterfactual coverage and efficacy (e, h,
    placebo delta); the guard-probe FAR / FRR table; C3 recall@k against the acceptable tool set; memory-read hits;
    and the H3 parameter-kind composition."""
    recs = list(recs if recs is not None else decision_records(trajs))  # type: ignore[arg-type]
    out: dict[str, Any] = {}
    cov = gold_coverage(trajs)
    if cov:
        out["coverage"] = cov
    gates: dict[str, Any] = {}
    live: dict[str, Any] = {}
    for key in sorted({f"{r['dp']}/{r['mode']}" for r in recs}):
        dp, mode = key.split("/", 1)
        if dp not in ("reflection", "routing", "stop", "memory_write") and not (dp == "tool_select" and mode == "guard"):
            continue
        if mode not in ("shadow", "trigger", "guard"):
            continue
        rs_all = [r for r in recs if r["dp"] == dp and r["mode"] == mode]
        rs = [r for r in rs_all if r.get("gold_status") in (None, "ok")]
        cfg = ((trajs[0].get("meta") or {}).get("config") or {}).get(dp) or {} if trajs else {}
        tau = float(cfg.get("tau", 0.5)) if mode != "shadow" else 0.5
        parts: list[tuple[str, list[Mapping[str, Any]], list[Mapping[str, Any]], bool]] = []
        if dp == "stop":
            for sub, sel in (("guard", True), ("trigger", False)):
                part = [r for r in rs if bool((r.get("gold_aux") or {}).get("kind")) == sel]
                if part and (mode == "shadow" or mode == sub):
                    parts.append((f"stop_{sub}/{mode}", part,
                                  [r for r in rs_all if bool((r.get("gold_aux") or {}).get("kind")) == sel], sel))
        else:
            parts.append((key, rs, rs_all, mode == "guard"))
        for k, part, part_all, guard in parts:
            if mode == "shadow":       # measurement arm: the H2 inputs (every opportunity labelled)
                gates[k] = gate_stats(part, tau, guard=guard)
            else:
                live[k] = _live_gate_row(part, part_all, tau, guard, dp)
    if gates:
        out["gates"] = gates
    if live:
        out["live_gates"] = live
    cf = [r for r in recs if "cf_status" in (r.get("gold_aux") or {})]
    if cf:
        st = Counter(str(r["gold_aux"]["cf_status"]) for r in cf)
        eff = {}
        for dp, br in (("reflection", "verdict_alt"), ("routing", "verdict_large"), ("stop", "verdict_reject")):
            e = cf_efficacy([r for r in cf if r["dp"] == dp], br)
            if e.get("n"):
                eff[dp] = e
        out["cf"] = {"n": len(cf), "status": dict(st), "coverage": st.get("ok", 0) / len(cf), "efficacy": eff}
    if any(t.get("probes") for t in trajs):
        out["probes"] = probe_error_rates(trajs)
        out["n_probes"] = sum(len(t.get("probes") or []) for t in trajs)
        out["n_no_valid_probe"] = sum(int(((t.get("meta") or {}).get("gold") or {}).get("n_no_valid_probe", 0) or 0)
                                      for t in trajs)
    ts_modes = {r["mode"] for r in recs if r["dp"] == "tool_select" and r.get("s1_dist")}
    if ts_modes:
        out["recall"] = recall_at_k(recs, modes=sorted(ts_modes - {"guard"}))
    mr = memory_read_hits(recs)
    if mr.get("n"):
        out["memory_read"] = mr
    pk = param_kind_composition(recs)
    if pk:
        out["param_kinds"] = pk
    return out


def _live_gate_row(labelled: Sequence[Mapping[str, Any]], every: Sequence[Mapping[str, Any]], tau: float, guard: bool,
                   dp: str) -> dict[str, Any]:
    """A live trigger / guard arm's gate vs. its gold: a CONSISTENCY CHECK, never an H2 input (META_GOLD_SPEC §9.2,
    §11.5-6). Reflection / routing triggers are selectively labelled -- a fired step's a0 is unobserved, or relabelled
    by a counterfactual call only when the arm runs ``cf.fired_steps`` -- so TPR / FPR / J / accuracy / estimability
    on the labelled rows are biased by construction (every remaining row has p_yes < tau) and arms with and without
    fired-step CF are not comparable. Reported: AUROC on the labels observed at executed actions only, the labelled
    share, the realised fire rate over EVERY opportunity (each counted once), and the retries. Guards, the stop
    trigger and the memory-write trigger are labelled at every opportunity; they still are not H2 inputs (gate AUROC
    comes from measurement arms only)."""
    from .metrics import first_opportunities

    selective = dp in ("reflection", "routing") and not guard
    first, n_retry = first_opportunities(list(every))           # each opportunity once (stop-guard retries dropped)
    keep = {id(r) for r in first}
    labelled = [r for r in labelled if id(r) in keep]
    obs = [r for r in labelled if (r.get("gold_aux") or {}).get("verdict_src", "executed") == "executed"] \
        if selective else list(labelled)
    st = gate_stats(obs, tau, guard=guard)
    fires = [(not r.get("s1_value")) if guard else bool(r.get("s1_value")) for r in first
             if isinstance(r.get("s1_value"), bool)]
    row = {"n": st.get("n", 0), "n_pos": st.get("n_pos", 0), "n_neg": st.get("n_neg", 0), "pi": st.get("pi"),
           "auroc": st.get("auroc"), "tau": tau, "guard": guard, "selective": selective,
           "n_opportunities": len(first), "n_retries": n_retry,
           "labelled_share": (len(labelled) / len(first)) if first else float("nan"),
           "n_cf_labelled": sum(1 for r in labelled if (r.get("gold_aux") or {}).get("verdict_src") == "cf"),
           "fire_rate_all": float(np.mean(fires)) if fires else float("nan"), "role": "consistency check"}
    if not selective:
        row.update({k: st.get(k) for k in ("tpr", "fpr", "J", "accuracy")})
    return row


# ------------------------------------------------------------------ arm-vs-arm comparisons
def paired_comparison(a: Mapping[str, Any], b: Mapping[str, Any], task_values: Sequence[float] = (1.0,),
                      lam_per_hour: float = 0.0, n_boot: int = 2000, alpha: float = 0.05,
                      seed: int = 0) -> dict[str, Any]:
    """Arm ``a`` vs. control ``b`` (:func:`arm_arrays` of each) on the tasks both ran -- the pre-registered
    arm-vs-control comparisons (docs/PREREGISTRATION.md: C3 Jev assist vs. BM25 assist at the same k, C5 Jev
    reflection trigger vs. the matched-rate random gate, superiority in utility). Paired task bootstrap of
    Delta-success (``p_better``: one-sided, small = ``a`` better), McNemar on episodes, mean Delta cost / latency,
    and Delta-utility for each task value W with U = W x success - cost - lambda x latency (lambda in USD/hour)."""
    keys = sorted(set(a["success"]) & set(b["success"]))
    if not keys:
        return {"n_paired": 0}
    t = paired_bootstrap_diff({k: b["success"][k] for k in keys}, {k: a["success"][k] for k in keys},
                              n=n_boot, alpha=alpha, seed=seed)
    mc = mcnemar(b["episodes"], a["episodes"])
    lam = lam_per_hour / 3600.0
    out: dict[str, Any] = {
        "n_paired": t["n"], "delta": t["diff"], "delta_ci": [t["ci_lo"], t["ci_hi"]], "lcb": t["lcb"],
        "p_better": t["p_better"], "p_worse": t["p_worse"], "mcnemar_p": mc["p"], "mcnemar_p_better": mc["p_b_better"],
        "n_a_only": mc["n10"], "n_b_only": mc["n01"],
        "delta_cost": float(np.mean([a["cost"][k] - b["cost"][k] for k in keys])),
        "delta_latency_s": float(np.mean([a["latency"][k] - b["latency"][k] for k in keys])),
        "lam_per_hour": lam_per_hour, "utility": {},
        # equivalence of the arm and its control within ±2 pt of success (the pre-registered H2 TOST for a gate vs.
        # its matched random gate; descriptive for other pairs)
        "tost_2pt": tost_paired({k: b["success"][k] for k in keys}, {k: a["success"][k] for k in keys}, 0.02,
                                n_boot=n_boot, seed=seed)}
    for w in task_values:
        ua = {k: w * a["success"][k] - a["cost"][k] - lam * a["latency"][k] for k in keys}
        ub = {k: w * b["success"][k] - b["cost"][k] - lam * b["latency"][k] for k in keys}
        u = paired_bootstrap_diff(ub, ua, n=n_boot, alpha=alpha, seed=seed)
        # gross of System-1's own price (H2: kappa is a property of the deployment, not of the gate's resolution)
        ga = {k: ua[k] + a.get("s1_cost", {}).get(k, 0.0) for k in keys}
        gb = {k: ub[k] + b.get("s1_cost", {}).get(k, 0.0) for k in keys}
        g = paired_bootstrap_diff(gb, ga, n=n_boot, alpha=alpha, seed=seed)
        out["utility"][f"{w:g}"] = {"delta_u": u["diff"], "ci": [u["ci_lo"], u["ci_hi"]], "p_better": u["p_better"],
                                    "delta_u_gross": g["diff"], "gross_ci": [g["ci_lo"], g["ci_hi"]]}
    return out


def _score_intervals(base: Mapping[str, float], arm: Mapping[str, float], alpha: float = 0.05) -> dict[str, Any]:
    """Descriptive Newcombe (1998) / Tango (1998) score intervals of Delta success on the paired single-trial cells
    (per-task means >= 1 count as a success; with several trials per task they are a summary, not the test)."""
    cells = ni_cells(base, arm)
    return {"cells": cells, "newcombe": newcombe_interval(**cells, alpha=alpha),
            "tango": tango_interval(**cells, alpha=alpha),
            "tango_lb_one_sided": tango_interval(**cells, alpha=alpha, one_sided=True)["ci"][0]}


# ------------------------------------------------------------------ the score card
def compute_scorecard(runs: Mapping[str, Sequence[Mapping[str, Any]]], baseline: str | None = "llm_only",
                      weak: str | None = None, sweeps: Sequence[str] | None = None, sweep_param: str | None = None,
                      alpha: float = 0.05, criterion: str = "not_worse", margin: float = 0.0, n_boot: int = 2000,
                      seed: int = 0, log_cost: bool = True, budget: float | None = None,
                      hv_ref: tuple[float, float] | None = None, meta: Mapping[str, Any] | None = None,
                      family_baselines: bool = True, pairs: Sequence[tuple[str, str]] | None = None,
                      task_values: Sequence[float] = (1.0,), lam_per_hour: float = 0.0,
                      measure_arms: Sequence[str] | None = None,
                      measure_runs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
                      replays: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
                      ni_margin: float | None = None, drop_errors: bool = True) -> dict[str, Any]:
    """Score card of every arm in ``runs`` (arm -> trajectories). ``baseline`` is the strong reference
    (LLM-only harness), ``weak`` the System-1-only reference used for PGR / APGR / CPT.

    ``family_baselines`` (default): an arm whose System-2 model differs from the baseline's is compared with the
    LLM-only arm of its own System-2 model when the run has one (:func:`family_references`; ``vs_baseline`` then
    names it as ``reference``), Holm runs per reference, and iso-quality / iso-cost are computed per family
    (``headline.iso_quality`` for the baseline's family, ``headline.iso_quality_by_reference`` for the others).
    ``False`` compares every arm with ``baseline`` (pre-2026-10 behaviour).

    ``pairs``: (arm, control) comparisons reported under ``headline.pairs`` (:func:`paired_comparison`), with
    Delta-utility at each of ``task_values`` (W, USD per success) and ``lam_per_hour``, net and gross of System-1's
    price. ``measure_arms`` (the measurement arms, one per System-2 family): each pair whose arm is a live gate gets
    the VoI readout of docs/META_GOLD_SPEC.md §9.1 / §9.3 (``meta_gold.voi_readout``: pi, AUROC, J at the live tau
    of the LIVE arm's System-1 -- the measurement arm's own log when it ran the same backend spec, else that spec's
    replay from ``replays`` (spec id -> rows, ``meta_gold.load_replays``); with neither, no prediction -- e / h net of
    placebo, predicted vs. observed Delta-U, the realised-rate correction). The measurement arm is chosen per gate
    (``meta_gold.measure_arm_for``: routing uses the routers' small-model arm). ``measure_runs``: measurement arms
    loaded from another run directory (the E2b measurement run next to the E2c pairs); they are used for the VoI
    readout only, never added to the card's arms.

    ``ni_margin`` (A6 / P0-1): the non-inferiority margin of the registered procedure (default: ``margin`` when it is
    positive, else 0.03 -- the JevGym / ALFWorld margin; tau^2 uses 0.05). Every arm's ``vs_baseline`` gains an ``ni``
    block (:func:`metrics.ni_pvalue`: p_NI, the 95 % and the Holm-level lower bounds, discordance, the exact check
    below 5 discordant tasks, the MDE and the ``inconclusive`` flag) and ``p_ni_holm`` / ``ni_holm`` / ``alpha_adj``
    from Holm over the family (every acting arm vs. the same reference) on ``p_decisive`` (the exact binomial p below
    5 discordant tasks, else p_NI); the Newcombe / Tango score intervals are reported descriptively
    (``score_intervals``). Existing columns are unchanged.

    ``drop_errors`` (default, PREREGISTRATION §1 / A6.9): tasks whose latest episode is an infrastructure error leave
    the arm's per-task maps, so every paired comparison (Delta, NI, Holm, iso-quality, pairs) is complete-case and
    the arm's success is over its completed tasks; ``n_tasks_errored`` per arm and a note report the counts (episode
    means such as cost and ``errors`` still cover every episode). ``False`` counts an errored episode as a failure
    (the registered full-set sensitivity view)."""
    notes: list[str] = []
    ni_m = float(ni_margin) if ni_margin is not None else (float(margin) if margin and margin > 0 else 0.03)
    deduped = {k: dedupe(v) for k, v in runs.items() if v}
    runs = {k: v for k, (v, _) in deduped.items()}
    dropped = {k: n for k, (_, n) in deduped.items() if n}
    if dropped:
        notes.append("duplicate task_id records dropped (kept the last): "
                     + ", ".join(f"{k}: {n}" for k, n in sorted(dropped.items())))
    if baseline is not None and baseline not in runs:
        notes.append(f"baseline {baseline!r} not found: baseline-relative metrics omitted")
        baseline = None
    if weak is not None and weak not in runs:
        notes.append(f"weak reference {weak!r} not found: PGR/APGR/CPT omitted")
        weak = None
    ref_of = (family_references(runs, baseline) if family_baselines else {}) if baseline else {}
    ref_of = {a: ref_of.get(a, baseline) for a in runs} if baseline else {}
    fam_refs = sorted({r for r in ref_of.values() if r != baseline})
    if fam_refs:
        notes.append("per-System-2 references: " + ", ".join(
            f"`{r}` ({arm_s2_model(runs[r])}) for {sum(1 for x in ref_of.values() if x == r)} arm(s)"
            for r in fam_refs) + f"; every other arm vs. `{baseline}`")
    errored: dict[str, list[str]] = {}
    kept: dict[str, list[dict[str, Any]]] = {}
    for a, t in runs.items():
        k, bad = drop_errored_tasks(t) if drop_errors else (list(t), [])
        if bad and not k:   # every task errored: nothing to compare; keep them (as failures) and say so
            notes.append(f"`{a}`: every task ended in an infrastructure error; kept as failures")
            k, bad = list(t), []
        kept[a] = k
        if bad:
            errored[a] = bad
    if errored:
        notes.append("drop rule (PREREGISTRATION §1 / A6.9): tasks whose latest episode is an infrastructure error "
                     "leave every paired comparison of their arm: " + ", ".join(
                         f"{a}: {len(b)}" for a, b in sorted(errored.items())))
    data = {a: arm_arrays(kept[a]) for a in runs}
    ep = {a: episode_table(t) for a, t in runs.items()}  # type: ignore[arg-type]
    S = {a: float(np.mean(list(d["success"].values()))) for a, d in data.items()}
    SC = {a: float(np.mean(list(d["strong_calls"].values()))) for a, d in data.items()}
    C = {a: float(np.mean(list(d["cost"].values()))) for a, d in data.items()}
    L = {a: float(np.mean(list(d["latency"].values()))) for a, d in data.items()}
    trials = [len(v) for d in data.values() for v in d["trials"].values()]
    min_trials = min(trials) if trials else 0
    # PREREGISTRATION A6.14 / A6.16 (e): with several trials per task (tau^2) the registered NI decision is taken on
    # MAJORITY PASS (a task passes when more than half of its trials succeed: >= 2 of 3); the Tango / exact rule needs a
    # per-task binary outcome. Delta-success and its CI stay the per-attempt (pass^1) means, descriptive.
    multi = bool(trials) and max(trials) >= 2
    ni_y = {a: ({k: float(2 * sum(v) > len(v)) for k, v in d["trials"].items()} if multi else d["success"])
            for a, d in data.items()}
    if multi:
        notes.append("several trials per task: the NI rows, Holm and the `noninferior` iso-quality qualifier use "
                     "majority pass (> half of a task's trials succeed; PREREGISTRATION A6.14 / A6.16 (e)); Delta and "
                     "its CI are per-attempt (pass^1) means")
    if budget is None and C:
        budget = 2.0 * (C[baseline] if baseline else max(C.values()))
    unpriced = sorted({m for t in runs.values() for x in t for m in ((x.get("usage") or {}).get("unpriced_models") or [])})
    if unpriced:
        notes.append(f"unpriced models (cost counted as 0): {', '.join(unpriced)}")
    arms: dict[str, dict[str, Any]] = {}
    for a, d in data.items():
        lo, hi = bootstrap_ci(list(d["success"].values()), n=n_boot, alpha=alpha, seed=seed)
        costs = [c for tr in d["cost_trials"].values() for c, _ in tr]
        succs = [s for tr in d["cost_trials"].values() for _, s in tr]
        row: dict[str, Any] = {
            "n_episodes": len(runs[a]), "n_tasks": len(d["success"]), "n_tasks_errored": len(errored.get(a, [])),
            "success": S[a], "success_ci": [lo, hi],
            "cost_usd": C[a], "s1_cost_usd": ep[a].get("s1_cost_usd", NAN), "latency_s": L[a],
            "llm_calls": ep[a].get("llm_calls", NAN), "s1_calls": ep[a].get("s1_calls", NAN),
            "llm_input_tokens": ep[a].get("llm_input_tokens", NAN), "steps": ep[a].get("steps", NAN),
            "errors": ep[a].get("errors", 0),
            "cost_of_pass": cost_of_pass(list(d["cost"].values()), list(d["success"].values())),
            "latency_of_pass": latency_of_pass(list(d["latency"].values()), list(d["success"].values())),
            "eub": effectiveness_under_budget(costs, succs, budget) if budget else NAN,
            # prompt-cache sensitivity (report.CACHE_KEYS): cost_usd stays the primary, billed figure
            **{k: v for k, v in episode_cache_columns(list(runs[a])).items() if k in CACHE_KEYS},
            "s1_cached_frac": s1_cached_frac(runs[a]),
        }
        if min_trials >= 2:
            row["pass_hat_k"] = {str(k): v for k, v in pass_hat_k_curve(d["trials"], range(1, min(min_trials, 8) + 1)).items()}
        if baseline:
            ref = ref_of.get(a, baseline)
            b = data[ref]
            row["reference"] = ref
            row["llm_call_ratio"] = (ep[a]["llm_calls"] / ep[ref]["llm_calls"]
                                     if ep[ref].get("llm_calls") else NAN)
            row["strong_call_ratio"] = SC[a] / SC[ref] if SC[ref] > 0 else NAN
            if a != baseline:
                t = paired_bootstrap_diff(b["success"], d["success"], n=n_boot, alpha=alpha, seed=seed)
                keys = sorted(set(b["cost"]) & set(d["cost"]))
                cb = float(np.mean([b["cost"][k] for k in keys])) if keys else NAN
                ca = float(np.mean([d["cost"][k] for k in keys])) if keys else NAN
                lb = float(np.mean([b["latency"][k] for k in keys])) if keys else NAN
                la = float(np.mean([d["latency"][k] for k in keys])) if keys else NAN
                # A6.22 (a): native LLM calls avoided per episode vs the reference, paired over the common tasks
                # (per-task means of usage.s2.calls; it includes trajectory-length effects -- an arm that ends
                # early also "avoids" calls -- so the within-arm `decisions.coverage.llm_free_steps` sits next to it)
                lcb_ = float(np.mean([b["llm_calls"][k] for k in keys])) if keys else NAN
                lca_ = float(np.mean([d["llm_calls"][k] for k in keys])) if keys else NAN
                row["vs_baseline"] = {
                    "reference": ref,
                    "n_paired": t["n"], "delta": t["diff"], "delta_ci": [t["ci_lo"], t["ci_hi"]], "lcb": t["lcb"],
                    "p_worse": t["p_worse"], "p_better": t["p_better"],
                    "mcnemar_p": mcnemar(b["episodes"], d["episodes"])["p"],
                    "cost_ratio": ca / cb if cb > 0 else NAN, "cost_savings": 1 - ca / cb if cb > 0 else NAN,
                    "latency_savings": 1 - la / lb if lb > 0 else NAN,
                    "llm_calls_avoided": lcb_ - lca_,
                    "llm_calls_avoided_frac": (lcb_ - lca_) / lcb_ if lcb_ > 0 else NAN,
                    # A6 / P0-1: the NI procedure (same resampling as ``delta`` / ``lcb``: _paired_boot, same seed)
                    "ni": ni_pvalue(ni_y[ref], ni_y[a], ni_m, n_boot=n_boot, seed=seed, alpha=alpha),
                    "score_intervals": _score_intervals(ni_y[ref], ni_y[a], alpha)}
                if multi:
                    row["vs_baseline"]["ni"]["endpoint"] = "majority pass"

            if weak:
                row["pgr"] = pgr(S[a], S[weak], S[baseline])
        row["decisions"] = decision_block(runs[a])
        arms[a] = row
    # shadow arms act exactly like the LLM-only harness (System-1 is only logged): they are measurement
    # arms, so they are excluded from the Holm family and from the iso-quality candidates
    shadow = {a for a in arms if _shadow_only(runs[a])}
    for a in shadow:
        arms[a]["shadow_only"] = True
    # multiple comparisons (family = every arm vs. one reference: suite x System-2 model)
    if baseline:
        for ref in sorted(set(ref_of.values())):
            adj = holm({a: r["vs_baseline"]["p_worse"] for a, r in arms.items()
                        if "vs_baseline" in r and a not in shadow and r["vs_baseline"]["reference"] == ref})
            for a, p in adj.items():
                arms[a]["vs_baseline"]["p_worse_holm"] = p
                arms[a]["vs_baseline"]["significantly_worse"] = bool(p < alpha)
            # A6: p_NI enters Holm over the same family; the test rejects iff p_NI <= alpha_adj of its Holm step,
            # equivalently iff the (1 - alpha_adj) lower bound is above -m (ni_pvalue recomputed at that level)
            lv = ni_holm_levels({a: r["vs_baseline"]["ni"]["p_decisive"] for a, r in arms.items()
                                 if "vs_baseline" in r and a not in shadow and r["vs_baseline"]["reference"] == ref},
                                alpha)
            for a, h in lv.items():
                vb = arms[a]["vs_baseline"]
                if not math.isnan(h["alpha_adj"]):
                    vb["ni"] = ni_pvalue(ni_y[ref], ni_y[a], ni_m, n_boot=n_boot, seed=seed,
                                         alpha=alpha, alpha_adj=h["alpha_adj"])
                vb["ni"].update({"p_ni_holm": h["p_holm"], "holm_rank": h.get("rank"), "holm_m": h.get("m"),
                                 "ni_holm": bool(h["reject"])})
                if multi:
                    vb["ni"]["endpoint"] = "majority pass"
                vb["ni"]["inconclusive"] = bool(not vb["ni"]["ni_holm"] and vb["ni"]["ci95"][0] <= 0 <= vb["ni"]["ci95"][1])
    # frontier over the arms that act: a shadow arm IS the LLM-only harness (System-1 only logged, its cost kept
    # out of the episode cost), so on the frontier it only adds replicate noise of the baseline -- in the E2a dev
    # run it took the Pareto front and "best cost-of-pass" by prompt-cache luck
    pts = {a: (C[a], S[a]) for a in arms if a not in shadow} or {a: (C[a], S[a]) for a in arms}
    front = set(pareto_front([(a, c, s) for a, (c, s) in pts.items()]))
    dom = dominance_counts(pts)
    cmax = max(c for c, _ in pts.values()) if pts else 1.0
    cmin = min(c for c, _ in pts.values()) if pts else 0.0
    ref = hv_ref or ((10 ** (math.log10(max(cmax, 1e-6)) + 0.1)) if log_cost else 1.1 * cmax, 0.0)
    contrib = hypervolume_contributions(pts, ref, log_cost)
    box = ((math.log10(max(ref[0], 1e-6)) - math.log10(max(cmin, 1e-6))) if log_cost else (ref[0] - cmin)) * (1 - ref[1])
    for a in arms:
        if a in pts:
            arms[a].update({"pareto": a in front, "dominates": dom[a]["dominates"],
                            "dominated_by": dom[a]["dominated_by"],
                            "hv_contribution": contrib[a] / box if box > 0 else NAN})
        else:
            arms[a].update({"pareto": False, "dominates": None, "dominated_by": None, "hv_contribution": NAN})
    acting = [a for a in arms if a in pts]
    headline: dict[str, Any] = {
        "hypervolume": {"ref": list(ref), "log_cost": log_cost,
                        "all": hypervolume_2d(list(pts.values()), ref, log_cost) / box if box > 0 else NAN,
                        "baseline_only": (hypervolume_2d([pts[baseline]], ref, log_cost) / box
                                          if baseline and baseline in pts and box > 0 else NAN),
                        "front": sorted(front)},
        "frontier_cost_of_pass": frontier_cost_of_pass({a: d["cost_trials"] for a, d in data.items() if a in pts}),
        "best_cost_of_pass": min(acting, key=lambda a: arms[a]["cost_of_pass"]) if acting else None,
    }
    headline["frontier_cost_of_pass"].pop("per_task", None)
    if baseline:
        # iso-quality / iso-cost within one System-2 family: the arms compared with that family's reference (a
        # cheaper System-2 model is a different claim, read off its own family and the S2-strength contrast)
        for fref in [baseline] + fam_refs:
            # candidates: the family's System-1 / meta configurations; another LLM-only arm of the same model
            # (e.g. small_only next to llm_only_s2lite) is a replicate of the reference, not a configuration
            fam = {a for a, r in ref_of.items() if r == fref and a not in shadow and a not in fam_refs
                   and not (fam_refs and is_llm_only_arm(runs[a]))} | {fref}
            # the `noninferior` qualifier is the registered decision statistic (A6.18 (c)): on majority pass when tasks
            # have several trials (tau^2), so its delta_success is then a majority-pass difference
            ys = ni_y if (multi and criterion == "noninferior") else {a: d["success"] for a, d in data.items()}
            succ = {a: ys[a] for a in data if a in fam}
            cost = {a: d["cost"] for a, d in data.items() if a in fam}
            iq = iso_quality_savings(succ, cost, fref, alpha=alpha, margin=margin, criterion=criterion,
                                     n_boot=n_boot, seed=seed)
            iq.pop("per_arm", None)
            ic = quality_at_iso_cost(succ, cost, fref, alpha=alpha, n_boot=n_boot, seed=seed)
            if fref == baseline:
                headline["iso_quality"], headline["iso_cost"] = iq, ic
            else:
                headline.setdefault("iso_quality_by_reference", {})[fref] = iq
                headline.setdefault("iso_cost_by_reference", {})[fref] = ic
    # arm-vs-control comparisons (pre-registered C3 / C5 style)
    if pairs:
        headline["pairs"] = {}
        for a, b in pairs:
            missing = [x for x in (a, b) if x not in data]
            if missing:
                notes.append(f"pair {a} vs {b}: arm(s) not found: {', '.join(missing)}")
                continue
            headline["pairs"][f"{a} vs {b}"] = {"arm": a, "control": b, **paired_comparison(
                data[a], data[b], task_values, lam_per_hour, n_boot, alpha, seed)}
            if measure_arms:
                from .meta_gold import arm_gate, measure_arm_for, voi_readout

                mruns = {x: dedupe(v)[0] for x, v in (measure_runs or {}).items() if v}
                pool = {**runs, **mruns}
                gate = arm_gate(runs[a])
                m = measure_arm_for(pool, measure_arms, a, gate) if gate else None
                if gate and m is None:
                    notes.append(f"pair {a} vs {b}: no measurement arm with shadow `{gate}` records for this arm among "
                                 f"{list(measure_arms)}")
                elif gate:
                    v = voi_readout(runs[a], runs[b], pool[m], task_values, n_boot=n_boot, seed=seed,
                                    replays=replays, measure_arm=m)
                    headline["pairs"][f"{a} vs {b}"]["voi"] = {"measure_arm": m, **v}
                    if v.get("note"):
                        notes.append(f"pair {a} vs {b}: {v['note']}")
    # sweeps
    sw_out: dict[str, Any] = {}
    # AIQ range: the budget span between the System-1-only and LLM-only references when both exist
    # (comparable across sweeps and independent of unrelated arms), else the span of all arms
    x_range = (min(C[weak], C[baseline]), max(C[weak], C[baseline])) if baseline and weak else (cmin, cmax)
    refs = [(C[a], S[a]) for a in (baseline, weak) if a]      # anchors: the tau -> 1 / tau -> 0 limits
    for prefix, members in detect_sweeps(list(arms), only=sweeps, param=sweep_param).items():
        names = [n for _, n in members]
        s: dict[str, Any] = {"arms": names, "values": [v for v, _ in members],
                             "success": [S[n] for n in names], "cost_usd": [C[n] for n in names]}
        s["aiq"] = aiq([(C[n], S[n]) for n in names] + refs, x_range=x_range)
        # RouterBench's "zero router": random per-episode mixing of the individual references
        # (System-1-only and LLM-only), i.e. their non-decreasing convex hull -- the bar a sweep must clear
        s["aiq_zero_router"] = aiq(refs, x_range=x_range) if len(refs) == 2 else NAN
        s["aiq_gain"] = s["aiq"] - s["aiq_zero_router"] if len(refs) == 2 else NAN
        if baseline and weak:
            # x = fraction of strong-model calls relative to the LLM-only baseline (RouteLLM's "% calls to
            # the strong model"); capped at 1 (assist / guard arms can call the LLM more than the baseline)
            xs = [min(1.0, arms[n]["strong_call_ratio"]) for n in names]
            wx = min(1.0, arms[weak]["strong_call_ratio"])
            curve = list(zip(xs, [S[n] for n in names]))
            s.update({"strong_call_frac": xs, "weak_x": wx,
                      "apgr": apgr(curve, S[weak], S[baseline], weak_x=wx),
                      "cpt50": cpt(curve, S[weak], S[baseline], 0.5, weak_x=wx),
                      "cpt80": cpt(curve, S[weak], S[baseline], 0.8, weak_x=wx)})
        sw_out[prefix] = s
    headline["sweeps"] = sw_out
    headline["aiq_range"] = list(x_range)
    return {"meta": {**(meta or {}), "baseline": baseline, "weak": weak, "alpha": alpha, "criterion": criterion,
                     "margin": margin, "ni_margin": ni_m, "drop_errors": drop_errors, "n_boot": n_boot, "seed": seed, "log_cost": log_cost,
                     "eub_budget": budget, "min_trials_per_task": min_trials, "max_trials_per_task": max(trials) if trials else 0,
                     "notes": notes},
            "arms": arms, "headline": headline}


def scorecard(run_dir: str | Path, **kw: Any) -> dict[str, Any]:
    """Load a run directory and compute its score card (see :func:`compute_scorecard`)."""
    runs = load_runs(run_dir)
    return compute_scorecard(runs, meta=run_meta(run_dir, runs), **kw)


# ------------------------------------------------------------------ rendering
def _clean(x: Any) -> Any:
    if isinstance(x, float):
        return None if math.isnan(x) else ("inf" if x == math.inf else ("-inf" if x == -math.inf else x))
    if isinstance(x, (np.floating,)):
        return _clean(float(x))
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def to_json(card: Mapping[str, Any]) -> str:
    """Strict JSON: nan -> null, +-inf -> "inf" / "-inf"."""
    return json.dumps(_clean(card), indent=1)


def _f(v: Any, spec: str = ".3g", pct: bool = False, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    v = 100 * v if pct else v
    return f"{v:+{spec}}" if sign else f"{v:{spec}}"


def _p(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _flags(a: str, r: Mapping[str, Any], card: Mapping[str, Any]) -> str:
    fl = []
    if r.get("pareto"):
        fl.append("front")
    vb = r.get("vs_baseline") or {}
    if vb.get("significantly_worse"):
        fl.append("worse")
    ref = (r.get("vs_baseline") or {}).get("reference") or r.get("reference")
    iqs = card["headline"].get("iso_quality_by_reference") or {}
    iq = iqs.get(ref) if ref in iqs else (card["headline"].get("iso_quality") or {})
    if a in (iq.get("qualified") or []):
        fl.append("iso-best" if a == iq.get("arm") else "iso-q")
    if a == card["meta"].get("baseline"):
        fl.append("baseline")
    elif ref and ref != card["meta"].get("baseline"):
        fl.append(f"vs {ref}")
    if a == card["meta"].get("weak"):
        fl.append("weak")
    return ",".join(fl)


def _iso_lines(iq: Mapping[str, Any] | None, ic: Mapping[str, Any] | None, tag: str = "") -> list[str]:
    out = []
    if iq:
        crit = ("not significantly worse, one-sided paired bootstrap" if iq["criterion"] == "not_worse"
                else f"non-inferior at margin {100 * iq['margin']:.1f} pp")
        if iq.get("arm"):
            out.append(f"- **Iso-quality cost savings**{tag}: {_f(iq['cost_savings'], '.1f', pct=True)}% "
                       f"[{_f(iq['cost_savings_ci'][0], '.1f', pct=True)}, {_f(iq['cost_savings_ci'][1], '.1f', pct=True)}] "
                       f"with `{iq['arm']}` (Δsuccess {_f(iq['delta_success'], '.1f', pct=True, sign=True)} pp "
                       f"[{_f(iq['delta_ci'][0], '.1f', pct=True)}, {_f(iq['delta_ci'][1], '.1f', pct=True)}], "
                       f"p_worse={_p(iq['p_worse'])}; {crit}, α={iq['alpha']}; {len(iq['qualified'])} arm(s) qualify)")
        else:
            out.append(f"- **Iso-quality cost savings**{tag}: no arm qualifies ({crit}, α={iq['alpha']})")
    if ic and ic.get("arm"):
        out.append(f"- **Quality at iso-cost**{tag} (≤ {_f(ic['budget'])} $/episode): `{ic['arm']}` "
                   f"{_f(ic['success'], '.1f', pct=True)}% vs. reference {_f(ic['baseline_success'], '.1f', pct=True)}% "
                   f"(Δ {_f(ic['delta_success'], '.1f', pct=True, sign=True)} pp, p_better={_p(ic.get('p_better'))}); "
                   f"hull {_f(ic['hull_success'], '.1f', pct=True)}%")
    return out


def _headline_lines(card: Mapping[str, Any]) -> list[str]:
    m, h = card["meta"], card["headline"]
    out = _iso_lines(h.get("iso_quality"), h.get("iso_cost"))
    for ref, iq in (h.get("iso_quality_by_reference") or {}).items():
        out += _iso_lines(iq, (h.get("iso_cost_by_reference") or {}).get(ref), f" (vs. `{ref}`)")
    fc = h.get("frontier_cost_of_pass") or {}
    n_shadow = sum(1 for r in card["arms"].values() if r.get("shadow_only"))
    n_act = len(card["arms"]) - n_shadow if n_shadow < len(card["arms"]) else len(card["arms"])
    out.append(f"- **Frontier cost-of-pass**: {_f(fc.get('value'))} $/success over {n_act} arms "
               + (f"({n_shadow} shadow arm(s) left out of the frontier) " if n_shadow and n_act < len(card["arms"]) else "")
               + f"(best single arm: `{h.get('best_cost_of_pass')}`)")
    hv = h.get("hypervolume") or {}
    out.append(f"- **Hypervolume** ({'log-' if hv.get('log_cost') else ''}cost × success, normalised): "
               f"all arms {_f(hv.get('all'))}, baseline alone {_f(hv.get('baseline_only'))}; "
               f"front: {', '.join(f'`{a}`' for a in hv.get('front', []))}")
    for p, s in (h.get("sweeps") or {}).items():
        extra = (f", APGR {_f(s['apgr'])}, CPT(50%) {_f(s['cpt50'])}, CPT(80%) {_f(s['cpt80'])}"
                 if "apgr" in s else "")
        zero = (f" vs. zero-router {_f(s['aiq_zero_router'])} (gain {_f(s['aiq_gain'], '+.3f')})"
                if not math.isnan(s["aiq_zero_router"]) else "")
        out.append(f"- **Sweep `{p}*`** ({len(s['arms'])} arms): AIQ {_f(s['aiq'])}{zero}{extra}")
    for n in m.get("notes") or []:
        out.append(f"- note: {n}")
    return out


EP_COLS = ["success", "Δ pp", "p_worse", "cost $", "save %", "CoP $", "lat s", "deleg %", "flags"]


def to_markdown(card: Mapping[str, Any], compact: bool = False) -> str:
    """Markdown score card; ``compact`` is the short section embedded in ``jevbench report``."""
    m, arms = card["meta"], card["arms"]
    head = "## Score card" + (f" (baseline: `{m['baseline']}`" + (f", weak: `{m['weak']}`" if m.get("weak") else "")
                              + ")" if m.get("baseline") else "")
    out = [head, ""]
    if m.get("simulated"):
        out += ["> **SIMULATED RUN** (simulated LLM and/or System-1): pipeline test / labelled simulation "
                "study only -- never headline numbers.", ""]
    out += _headline_lines(card) + [""]
    order = sorted(arms, key=lambda a: (a != m.get("baseline"), -arms[a]["success"], arms[a]["cost_usd"]))
    if compact:
        out += ["| arm | " + " | ".join(EP_COLS) + " |", "|" + "---|" * (len(EP_COLS) + 1)]
    else:
        cols = EP_COLS[:-1] + ["McNemar p", "lat save %", "LoP s", "LLM-call ratio", "strong-call ratio", "PGR",
                               "EuB", "HV contrib", "LLM-free steps %", "cache-mod $", "no-cache $", "cached %",
                               "flags"]
        out += ["| arm | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for a in order:
        r = arms[a]
        vb = r.get("vs_baseline") or {}
        d = r.get("decisions") or {}
        succ = (f"{_f(r['success'], '.1f', pct=True)} [{_f(r['success_ci'][0], '.1f', pct=True)}, "
                f"{_f(r['success_ci'][1], '.1f', pct=True)}]")
        cells = [succ, _f(vb.get("delta"), ".1f", pct=True, sign=True), _p(vb.get("p_worse")), _f(r["cost_usd"], "#.3g"),
                 _f(vb.get("cost_savings"), ".1f", pct=True), _f(r["cost_of_pass"], "#.3g"), _f(r["latency_s"], "#.3g"),
                 _f(d.get("delegation_ratio"), ".1f", pct=True)]
        if not compact:
            cells += [_p(vb.get("mcnemar_p")), _f(vb.get("latency_savings"), ".1f", pct=True),
                      _f(r["latency_of_pass"], "#.3g"), _f(r.get("llm_call_ratio"), ".2f"),
                      _f(r.get("strong_call_ratio"), ".2f"), _f(r.get("pgr"), ".2f"),
                      _f(r.get("eub"), ".3f"), _f(r.get("hv_contribution"), ".3f"),
                      _f(d.get("llm_free_steps"), ".1f", pct=True),
                      _f(r.get("cost_usd_cache_modelled"), "#.3g"), _f(r.get("cost_usd_no_cache"), "#.3g"),
                      _f(r.get("cached_frac"), ".1f", pct=True)]
        cells.append(_flags(a, r, card))
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    out += ["", "_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, "
            "small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); "
            "save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions "
            "taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse "
            "(Holm), iso-best / iso-q = qualifies for iso-quality savings._", ""]
    if compact:
        return "\n".join(out)
    out += ["_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their "
            "logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no "
            "cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._", ""]
    out += _ni_markdown(card, order)
    # pass^k
    pk = {a: r["pass_hat_k"] for a, r in arms.items() if r.get("pass_hat_k")}
    if pk:
        ks = sorted({int(k) for v in pk.values() for k in v})
        out += ["### Reliability (pass^k)", "", "| arm | " + " | ".join(f"pass^{k}" for k in ks) + " |",
                "|" + "---|" * (len(ks) + 1)]
        for a in order:
            if a in pk:
                out.append(f"| {a} | " + " | ".join(_f(pk[a].get(str(k)), ".3f") for k in ks) + " |")
        out.append("")
    # decisions
    dcols = ["n scored", "S1 acc", "ECE", "ACE", "Brier", "AUROC", "AURC", "E-AURC", "acc@80%cov", "harmful %",
             "guard FAR", "guard FRR", "rescue %", "yes/no n", "yes/no ECE", "yes/no Brier", "yes/no AUROC",
             "mem stored %"]
    rows = []
    for a in order:
        d = arms[a].get("decisions") or {}
        if not (d.get("n_s1_scored") or d.get("harmful") or d.get("guard") or d.get("noul")
                or d.get("memory_stored_frac") is not None):
            continue
        g = (d.get("guard") or {}).get("all") or {}
        rs = (d.get("rescue") or {}).get("all") or {}
        nl = d.get("noul") or {}
        rows.append(f"| {a} | " + " | ".join([
            str(d.get("n_s1_scored", 0)), _f(d.get("s1_acc"), ".3f"), _f(d.get("ece"), ".3f"), _f(d.get("ace"), ".3f"),
            _f(d.get("brier"), ".3f"), _f(d.get("auroc"), ".3f"), _f(d.get("aurc"), ".3f"), _f(d.get("eaurc"), ".3f"),
            _f(d.get("sel_acc@80"), ".3f"), _f((d.get("harmful") or {}).get("harmful_rate"), ".1f", pct=True),
            _f(g.get("far"), ".3f"), _f(g.get("frr"), ".3f"), _f(rs.get("rescue_rate"), ".1f", pct=True),
            str(nl.get("n", 0)), _f(nl.get("ece"), ".3f"), _f(nl.get("brier"), ".3f"), _f(nl.get("auroc"), ".3f"),
            _f(d.get("memory_stored_frac"), ".1f", pct=True)]) + " |")
    if rows:
        out += ["### System-1 decisions", "", "| arm | " + " | ".join(dcols) + " |", "|" + "---|" * (len(dcols) + 1),
                *rows, "", "_Calibration over System-1 Choice decisions with gold labels (top-label confidence); "
                "yes/no: guard / trigger decisions calibrated as forecasts P(yes) of the binary gold; harmful: delegated "
                "decisions that were wrong in failed episodes (per delegated decision); FAR/FRR: guard false "
                "accept / false reject rates; rescue: wrong LLM proposals caught by the guard in episodes that "
                "succeeded._", ""]
    out += _coverage_markdown(card, order)
    out += _gold_markdown(card, order)
    sw = card["headline"].get("sweeps") or {}
    if sw:
        out += ["### Sweeps", "", "| sweep | arms | values | APGR | CPT(50%) | CPT(80%) | AIQ | AIQ zero-router | ΔAIQ |",
                "|---|---|---|---|---|---|---|---|---|"]
        for p, s in sw.items():
            out.append(f"| {p} | {len(s['arms'])} | {', '.join(f'{v:g}' for v in s['values'])} | {_f(s.get('apgr'))} | "
                       f"{_f(s.get('cpt50'))} | {_f(s.get('cpt80'))} | {_f(s['aiq'])} | {_f(s['aiq_zero_router'])} | "
                       f"{_f(s.get('aiq_gain'), '+.3f')} |")
        out += ["", "_APGR / CPT: x = strong-model calls relative to the baseline, anchors weak (PGR 0) and baseline "
                "(PGR 1); AIQ: mean success of the non-decreasing convex hull over the linear cost range between the "
                "two references (sweep + references); zero-router: hull of the references alone._"]
        out.append("")
    pr = card["headline"].get("pairs") or {}
    if pr:
        ws = sorted({w for v in pr.values() for w in (v.get("utility") or {})}, key=float)
        out += ["### Paired comparisons (arm vs. control)", "",
                "| arm | control | n | Δ success pp | 95% CI | p_better | McNemar p | TOST ±2 pt (p, discordant) | Δ cost $ | "
                "Δ lat s | " + " | ".join(f"ΔU (W={w}) [CI], p" for w in ws) + " |",
                "|" + "---|" * (10 + len(ws))]
        for v in pr.values():
            if not v.get("n_paired"):
                continue
            us = [f"{_f(v['utility'][w]['delta_u'], '+.4f')} [{_f(v['utility'][w]['ci'][0], '+.4f')}, "
                  f"{_f(v['utility'][w]['ci'][1], '+.4f')}], {_p(v['utility'][w]['p_better'])}" for w in ws]
            out.append(f"| {v['arm']} | {v['control']} | {v['n_paired']} | {_f(v['delta'], '+.1f', pct=True)} | "
                       f"[{_f(v['delta_ci'][0], '.1f', pct=True)}, {_f(v['delta_ci'][1], '.1f', pct=True)}] | "
                       f"{_p(v['p_better'])} | {_p(v['mcnemar_p'])} | "
                       f"{'equiv.' if (v.get('tost_2pt') or {}).get('equivalent') else 'not shown'} "
                       f"({_p((v.get('tost_2pt') or {}).get('p'))}, {(v.get('tost_2pt') or {}).get('n_discordant', '–')}"
                       f" discordant) | {_f(v['delta_cost'], '+.4f')} | "
                       f"{_f(v['delta_latency_s'], '+.2f')} | " + " | ".join(us) + " |")
        out += ["", "_Paired task bootstrap of arm − control; p_better one-sided (small = the arm is better); "
                "TOST: equivalence within ±2 pt of success (90% paired-bootstrap CI inside the margin; the "
                "pre-registered H2 test of a System-1 gate against its matched random gate); "
                "U = W·success − cost − λ·latency (λ = "
                f"{next(iter(pr.values())).get('lam_per_hour', 0)} USD/h)._", ""]
        vo = [(k, v["voi"]) for k, v in pr.items() if v.get("voi") and v["voi"].get("stats")]
        if vo:
            w0 = ws[min(range(len(ws)), key=lambda i: abs(float(ws[i]) - 1.0))] if ws else "1"
            out += [f"### Gate VoI: predicted vs. observed (W = {w0}; docs/META_GOLD_SPEC.md §9.1)", "",
                    "| pair | gate | meas. arm | System-1 answers | n+/n− | π | AUROC | J(τ) | e | h | n̄ | "
                    "rate live/ctrl | pred. ΔU gross | obs. ΔU gross [CI] | estimable |", "|" + "---|" * 15]
            for k, v in vo:
                st, ef = v["stats"], v.get("efficacy") or {}
                pdu = (v.get("predicted") or {}).get(w0) or {}
                odu = (v.get("observed") or {}).get(w0) or {}
                out.append(f"| {k} | {v.get('gate')} | {v.get('measure_arm')} | {v.get('stats_source')} | "
                           f"{st.get('n_pos', 0)}/{st.get('n_neg', 0)} | "
                           f"{_f(st.get('pi'), '.3f')} | {_f(st.get('auroc'), '.3f')} | {_f(st.get('J'), '.3f')} | "
                           f"{_f(ef.get('e'), '.3f')} | {_f(ef.get('h'), '.3f')} | {_f(v.get('n_bar'), '.2f')} | "
                           f"{_f(v.get('rate_live'), '.3f')}/{_f(v.get('rate_control'), '.3f')} | "
                           f"{_f(pdu.get('delta_u_gross'), '+.4f')} | {_f(odu.get('gross'), '+.4f')} "
                           f"[{_f((odu.get('gross_ci') or [None, None])[0], '+.4f')}, "
                           f"{_f((odu.get('gross_ci') or [None, None])[1], '+.4f')}] | "
                           f"{'yes' if v.get('estimable') else 'no opportunity'} |")
            out += ["", "_π, AUROC, J: the LIVE arm's System-1 on the measurement arm's shadow golds at the live τ -- "
                    "the measurement arm's own log when it ran the same backend spec, else that spec's replay "
                    "(`--replay`); pairs with neither get no prediction (see the notes). e, h: counterfactual "
                    "efficacy net of the placebo; predicted ΔU = n̄·W·(e+h)·π(1−π)·J (gross of System-1's price); "
                    "observed: paired task bootstrap of the gross utility difference._", ""]
        vn = [(k, v["voi"]) for k, v in pr.items() if v.get("voi") and not v["voi"].get("stats")]
        if vn:
            out += ["Gate VoI not predicted for: " + "; ".join(f"{k} ({v.get('note') or v.get('stats_source')})"
                                                               for k, v in vn), ""]
    return "\n".join(out)


def _coverage_markdown(card: Mapping[str, Any], order: Sequence[str]) -> list[str]:
    """PREREGISTRATION A6.22 (a): tool-decision coverage, argument coverage (over every executed argument slot and
    inside System-1-chosen calls) and native LLM calls avoided per episode vs the reference, for every arm that runs a
    delegating mode (replace / cascade) at an object-level decision point."""
    arms = card["arms"]
    rows = []
    for a in order:
        cv = (arms[a].get("decisions") or {}).get("coverage") or {}
        if not cv.get("delegating"):
            continue
        vb = arms[a].get("vs_baseline") or {}
        rows.append(f"| {a} | {_f(cv.get('tool_coverage'), '.1f', pct=True)} | {_f(cv.get('arg_coverage'), '.1f', pct=True)} "
                    f"({cv.get('n_arg_delegated', 0)}/{cv.get('n_arg_slots', 0)}) | "
                    f"{_f(cv.get('arg_coverage_s1_calls'), '.1f', pct=True)} | {_f(cv.get('llm_free_steps'), '.1f', pct=True)} | "
                    f"{_f(cv.get('llm_calls_per_episode'), '.2f')} | {_f(vb.get('llm_calls_avoided'), '+.2f')} | "
                    f"{_f(vb.get('llm_calls_avoided_frac'), '.1f', pct=True)} | {vb.get('reference', '–')} |")
    if not rows:
        return []
    return ["### Delegation coverage and native LLM calls (A6.22 (a))", "",
            "| arm | tool cov. % | arg cov. % (S1 / slots) | arg cov. in S1 calls % | LLM-free steps % | LLM calls / ep. | "
            "calls avoided / ep. | avoided % | reference |", "|" + "---|" * 9, *rows, "",
            "_tool cov.: tool choices executed from System-1 per tool-selection opportunity; arg cov.: argument values "
            "System-1 chose per argument slot of the executed calls (a deferred tool choice leaves the whole call, and "
            "its slots, to the LLM); in S1 calls: the same share inside System-1-chosen calls (– when it chose none: the "
            "argument thresholds were never reached); calls avoided: paired per-task mean of the reference's minus the "
            "arm's System-2 calls (trajectory length included)._", ""]


def _ni_markdown(card: Mapping[str, Any], order: Sequence[str]) -> list[str]:
    """The A6 non-inferiority readout (one row per acting arm with a reference): p_NI, Holm level and bound, the
    discordance, the exact check when it applies, the MDE, and the verdict (NI / inconclusive / inferior)."""
    arms = card["arms"]
    rows = [(a, arms[a]["vs_baseline"]) for a in order
            if (arms[a].get("vs_baseline") or {}).get("ni") and not arms[a].get("shadow_only")]
    if not rows:
        return []
    m = card["meta"].get("ni_margin")
    out = [f"### Non-inferiority (A6 procedure; margin {_f(m, '.1f', pct=True)} pp)", "",
           "| arm | reference | n | Δ pp | 95% LCB pp | p (rule) | Holm p | α_adj | Tango LB(1−α_adj) pp | "
           "discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | p_boot | verdict |", "|" + "---|" * 14]
    for a, vb in rows:
        ni = vb["ni"]
        ex = ni.get("exact") or {}
        mde = ni.get("mde") or {}
        si = vb.get("score_intervals") or {}
        if ni.get("ni_holm"):
            verdict = "NI"
        elif ni.get("inconclusive"):
            verdict = "inconclusive at this n"
        else:
            verdict = "not NI"
        exact = (f"{_p(ex.get('p'))} ({_f(ex.get('loss_rate_ub'), '.2f', pct=True)})" if ex else "–")
        out.append(f"| {a} | {vb.get('reference')} | {ni['n']} | {_f(ni['diff'], '+.1f', pct=True)} | "
                   f"{_f(ni['lcb95'], '+.1f', pct=True)} | {_p(ni.get('p_decisive'))} ({ni.get('rule', '–')}) | "
                   f"{_p(ni.get('p_ni_holm'))} | "
                   f"{_f(ni.get('alpha_adj'), '.4f')} | {_f(ni.get('tango_lcb_adj'), '+.1f', pct=True)} | "
                   f"{ni['n_discordant']} ({ni['wins']}/{ni['losses']}) | {exact} | "
                   f"{_f(mde.get('delta_mde'), '+.1f', pct=True)} | {_p(ni['p_ni'])} | "
                   f"{verdict} |")
    out += ["", "_A6.14: p_T = 1 − Φ(Z(−m)), Tango's one-sided score test of H0: Δ ≤ −m on the paired (wins, losses, n) "
            "counts, enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_T ≤ α_adj of its "
            "Holm step, equivalently iff the Tango (1 − α_adj) lower bound is above −m. With zero wins or < 5 "
            "discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides instead (`rule` = exact; it is the p "
            "that enters Holm, `Holm p`) and the one-sided 95% upper "
            "bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails "
            "while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ "
            "shown NI with 80% power at the achieved discordance and α_adj. p: the certified Berger-Boos p-value of "
            "one ordering of all (wins, losses) outcomes (A6.21; rule bb_ordered). "
            "Tango LB: descriptive score interval._", ""]
    return out


def _gold_markdown(card: Mapping[str, Any], order: Sequence[str]) -> list[str]:
    """Score-card sections of the meta-decision golds (docs/META_GOLD_SPEC.md §9.7)."""
    arms = card["arms"]
    out: list[str] = []
    gold = {a: (arms[a].get("decisions") or {}).get("gold") or {} for a in order}
    cov = [(a, k, r) for a in order for k, r in (gold[a].get("coverage") or {}).items()]
    if cov:
        out += ["### Meta-decision golds: coverage", "", "| arm | dp/mode | n | labelled | coverage | doomed | "
                "unobserved | no_oracle | cf_error |", "|" + "---|" * 9]
        for a, k, r in cov:
            st = r.get("status") or {}
            out.append(f"| {a} | {k} | {r['n']} | {r['n_labelled']} | {_f(r.get('coverage'), '.1%')} | "
                       f"{st.get('doomed', 0)} | {st.get('unobserved', 0)} | {st.get('no_oracle', 0)} | "
                       f"{st.get('cf_error', 0)} |")
        out.append("")
    gates = [(a, k, r) for a in order for k, r in (gold[a].get("gates") or {}).items() if r.get("n")]
    if gates:
        out += ["### Gates vs. their primary gold (H2 inputs: measurement arms)", "", "| arm | gate/mode | n | "
                "n+ / n− | π | AUROC | τ | J(τ) | acc(τ) | fire rate | ECE | H2 cell |", "|" + "---|" * 12]
        for a, k, r in gates:
            out.append(f"| {a} | {k} | {r['n']} | {r['n_pos']} / {r['n_neg']} | {_f(r.get('pi'), '.3f')} | "
                       f"{_f(r.get('auroc'), '.3f')} | {_f(r.get('tau'), '.2f')} | {_f(r.get('J'), '.3f')} | "
                       f"{_f(r.get('accuracy'), '.3f')} | {_f(r.get('fire_rate'), '.3f')} | {_f(r.get('ece'), '.3f')} | "
                       f"{'estimable' if r.get('estimable') else 'no opportunity'} |")
        out += ["", "_Shadow records of measurement arms (every opportunity labelled), the arm's own System-1; other "
                "backends: `meta_gold replay` + `meta_gold h2`. Primary golds: reflection / routing = the System-2 "
                "action at the state is failing (need_fatal); stop guard = finishing now with the proposed answer "
                "succeeds; stop trigger = only the answer remains; memory_write = the offered text holds a value the "
                "remaining plan consumes; tool guard = the proposal is progress. A guard fires (rejects) on p_yes < τ, "
                "so J = 1 − FAR − FRR. H2 cell: ≥ 30 positive and ≥ 30 negative golds._", ""]
    lg = [(a, k, r) for a in order for k, r in (gold[a].get("live_gates") or {}).items() if r.get("n_opportunities")]
    if lg:
        out += ["### Live gates vs. their gold (consistency check; not H2 inputs)", "",
                "| arm | gate/mode | opportunities (retries) | labelled share | n+ / n− labelled | AUROC (observed "
                "labels) | J(τ) | realised fire rate |", "|" + "---|" * 8]
        for a, k, r in lg:
            out.append(f"| {a} | {k} | {r['n_opportunities']} ({r.get('n_retries', 0)}) | "
                       f"{_f(r.get('labelled_share'), '.1%')} | {r.get('n_pos', 0)} / {r.get('n_neg', 0)} | "
                       f"{_f(r.get('auroc'), '.3f')} | "
                       f"{'selective' if r.get('selective') else _f(r.get('J'), '.3f')} | "
                       f"{_f(r.get('fire_rate_all'), '.3f')} |")
        out += ["", "_Live trigger arms label a reflection / routing opportunity only where the gate did not fire "
                "(selective labels): AUROC is computed on the labels observed at executed actions, J / accuracy / "
                "estimability are not reported, and these rows never enter H2 (META_GOLD_SPEC §9.2, §11.5–6). The "
                "realised fire rate counts every opportunity once (a stop-guard resubmission of a rejected answer is "
                "a retry)._", ""]
    cf = [(a, gold[a]["cf"]) for a in order if gold[a].get("cf")]
    if cf:
        out += ["### Counterfactual efficacy (§4)", "", "| arm | dp | n | n+ / n− | e | h | placebo δ | final share | "
                "CF coverage |", "|" + "---|" * 9]
        for a, c in cf:
            for dp, e in (c.get("efficacy") or {}).items():
                out.append(f"| {a} | {dp} | {e['n']} | {e.get('n_pos', 0)} / {e.get('n_neg', 0)} | {_f(e.get('e'), '+.3f')} | "
                           f"{_f(e.get('h'), '+.3f')} | {_f(e.get('delta'), '.3f')} | {_f(e.get('final_share'), '.2f')} | "
                           f"{_f(c.get('coverage'), '.1%')} |")
        out.append("")
    pr = [(a, gold[a]["probes"]) for a in order if gold[a].get("probes")]
    if pr:
        out += ["### Guard probes (measurement arms; §6.3)", "", "| arm | guard | FAR_model [CI] (n wrong / tasks) | "
                "FAR_nat (n) | FRR natural right [CI] (n) | FAR_fatal (n) | FAR reweighted | nat − model TOST ±0.10 | "
                "style check | AUROC p_yes |", "|" + "---|" * 10]
        for a, p in pr:
            for dp, b in p.items():
                fm = b.get("far_model") or b.get("headline") or {}
                ff = b.get("far_fatal") or {}
                fr = b.get("frr") or fm
                fn = b.get("far_nat") or {}
                sc = b.get("style_check") or {}
                rp = b.get("representativeness") or {}
                tost = (f"{_f(rp.get('diff'), '+.3f')} [{_f((rp.get('ci90') or [None])[0], '+.2f')}, "
                        f"{_f((rp.get('ci90') or [None, None])[1], '+.2f')}] "
                        f"{'equiv.' if rp.get('equivalent') else 'not shown'}") \
                    if rp and not (isinstance(rp.get("diff"), float) and math.isnan(rp["diff"])) else "n/a"
                out.append(f"| {a} | {dp} | {_f(fm.get('far'), '.3f')} [{_f((fm.get('far_ci') or [None])[0], '.2f')}, "
                           f"{_f((fm.get('far_ci') or [None, None])[1], '.2f')}] ({fm.get('n_wrong', 0)} / "
                           f"{fm.get('n_tasks_wrong', 0)}) | {_f(fn.get('far'), '.3f')} ({fn.get('n_wrong', 0)}) | "
                           f"{_f(fr.get('frr'), '.3f')} [{_f((fr.get('frr_ci') or [None])[0], '.2f')}, "
                           f"{_f((fr.get('frr_ci') or [None, None])[1], '.2f')}] ({fr.get('n_right', 0)}) | "
                           f"{_f(ff.get('far'), '.3f')} ({ff.get('n_wrong', 0)}) | "
                           f"{_f(b.get('far_reweighted'), '.3f')} | {tost} | {sc.get('ok')} ({_f(sc.get('diff'), '+.3f')}) | "
                           f"{_f(b.get('auroc_p_yes'), '.3f')} |")
        out += ["", "_Stop: FAR_model over natural, forced-premature and transplant wrong finishes (the C6 readout, "
                "≥ 50 wrong per System-2 model); tool guard: FAR / FRR against progress over natural + transplant "
                "proposals, FAR_fatal over every fatal probe (T1–T4 edits, fatal transplants). CIs: task-clustered "
                "percentile bootstrap (several probes per task); TOST: FAR_nat − FAR_model, 90% task-bootstrap CI "
                "inside ±0.10 (§6.3). Probes are scored after the episode, never executed, and never enter success, "
                "cost, Holm or the matched-rate protocol._", ""]
    rc = [(a, gold[a]["recall"]) for a in order if (gold[a].get("recall") or {}).get("n_acceptable")]
    if rc:
        out += ["### Tool-selection recall@k vs. the acceptable tool set (C3)", "",
                "| arm | n | recall@1 | recall@3 | recall@5 (undefined) | recall@3 (canonical) | offered recall "
                "(offered k) |", "|" + "---|" * 7]
        for a, r in rc:
            out.append(f"| {a} | {r['n_acceptable']} | {_f(r.get('recall@1'), '.3f')} | {_f(r.get('recall@3'), '.3f')} | "
                       f"{_f(r.get('recall@5'), '.3f')} ({r.get('n_undefined@5', 0)}) | "
                       f"{_f(r.get('recall@3_canonical'), '.3f')} | "
                       f"{_f(r.get('offered_recall'), '.3f')} ({_f(r.get('offered_k'), '.1f')}) |")
        out += ["", "_recall@k: the top-k of System-1's full ranking (tournaments log it as `s1_rank`: finalists, then "
                "each round's eliminated options by score); records whose ranking is incomplete at k (older tournament "
                "logs with only their finalists) are undefined and left out (count in brackets). Offered k = mean "
                "number of tools an assist arm offered System-2 (top-k + finish)._", ""]
    pk = [(a, gold[a]["param_kinds"]) for a in order if gold[a].get("param_kinds")]
    if pk:
        out += ["### Tool arguments by fill kind (H3; §7.4)", "", "| arm | kind | n | delegated | delegation | harmful |",
                "|" + "---|" * 6]
        for a, p in pk:
            for k, r in p.items():
                out.append(f"| {a} | {k} | {r['n']} | {r['delegated']} | {_f(r.get('delegation_ratio'), '.1%')} | "
                           f"{_f(r.get('harmful_rate'), '.1%')} |")
        out.append("")
    return out


def _tex(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _t(v: Any, spec: str = ".3g", pct: bool = False, sign: bool = False) -> str:
    s = _f(v, spec, pct, sign)
    return {"–": "--", "∞": r"$\infty$"}.get(s, s)


def to_latex(card: Mapping[str, Any]) -> str:
    """Booktabs table (``\\usepackage{booktabs}``): one row per arm plus a sweep table. Markers:
    $^\\star$ Pareto front, $^\\dagger$ significantly worse than the baseline (Holm), bold = the
    iso-quality arm; the headline numbers follow as LaTeX comments."""
    m, arms = card["meta"], card["arms"]
    order = sorted(arms, key=lambda a: (a != m.get("baseline"), -arms[a]["success"], arms[a]["cost_usd"]))
    iq = card["headline"].get("iso_quality") or {}
    lines = []
    if m.get("simulated"):
        lines.append("% SIMULATED RUN -- pipeline test / labelled simulation study only, never headline numbers")
    lines += [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
              r"Configuration & Succ.\ (\%) & $\Delta$ (pp) & Cost (\$) & Saved (\%) & CoP (\$) & Lat.\ (s) & Deleg.\ (\%) \\",
              r"\midrule"]
    for a in order:
        r = arms[a]
        vb = r.get("vs_baseline") or {}
        name = _tex(a)
        if a == iq.get("arm"):
            name = rf"\textbf{{{name}}}"
        if r.get("pareto"):
            name += r"$^\star$"
        if vb.get("significantly_worse"):
            name += r"$^\dagger$"
        lines.append(" & ".join([name, _t(r["success"], ".1f", pct=True), _t(vb.get("delta"), ".1f", pct=True, sign=True),
                                 _t(r["cost_usd"], "#.3g"), _t(vb.get("cost_savings"), ".1f", pct=True),
                                 _t(r["cost_of_pass"], "#.3g"), _t(r["latency_s"], "#.3g"),
                                 _t((r.get("decisions") or {}).get("delegation_ratio"), ".1f", pct=True)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    sw = card["headline"].get("sweeps") or {}
    if sw:
        lines += ["", r"\begin{tabular}{lrrrrrr}", r"\toprule",
                  r"Sweep & APGR & CPT(50\%) & CPT(80\%) & AIQ & AIQ$_0$ & $\Delta$AIQ \\", r"\midrule"]
        for p, s in sw.items():
            lines.append(" & ".join([_tex(p), _t(s.get("apgr")), _t(s.get("cpt50")), _t(s.get("cpt80")), _t(s["aiq"]),
                                     _t(s["aiq_zero_router"]), _t(s.get("aiq_gain"), "+.3f")]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [f"% {ln.lstrip('- ').replace('**', '')}" for ln in _headline_lines(card)]
    return "\n".join(lines)


def render(card: Mapping[str, Any], fmt: str = "md") -> str:
    if fmt == "json":
        return to_json(card)
    if fmt == "latex":
        return to_latex(card)
    return to_markdown(card)


# ------------------------------------------------------------------ CLI
def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser("python -m jevharness.bench.scorecard",
                                 description="Per-arm score card of a JevAgentBench run directory.")
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default="llm_only", help="strong reference arm (LLM-only harness)")
    ap.add_argument("--weak", default=None, help="weak reference arm (System-1-only) for PGR / APGR / CPT")
    ap.add_argument("--sweep", action="append", default=None, help="sweep prefix, e.g. ts_cascade_tau (repeatable)")
    ap.add_argument("--sweep-param", default=None, help="only auto-detect sweeps over this parameter, e.g. tau")
    ap.add_argument("--format", choices=["md", "latex", "json"], default="md")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--criterion", choices=["not_worse", "noninferior"], default="not_worse")
    ap.add_argument("--margin", type=float, default=0.0, help="non-inferiority margin (success, e.g. 0.02)")
    ap.add_argument("--ni-margin", type=float, default=None,
                    help="A6 NI margin for the p_NI rows (default: --margin if > 0, else 0.03; tau^2: 0.05)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear-cost", action="store_true",
                    help="hypervolume on linear instead of log10 cost (AIQ is always on linear cost)")
    ap.add_argument("--budget", type=float, default=None, help="cost budget ($/episode) of EuB (default 2x baseline)")
    ap.add_argument("--hv-ref", default=None, help="hypervolume reference point COST,SUCCESS")
    ap.add_argument("--pair", action="append", default=None, metavar="ARM:CONTROL",
                    help="arm-vs-control comparison (repeatable), e.g. jev_ts_assist_k3:bm25_ts_assist_k3")
    ap.add_argument("--task-value", type=float, nargs="+", default=[1.0],
                    help="W values (USD per success) for the utility difference of --pair comparisons")
    ap.add_argument("--lam-per-hour", type=float, default=0.0, help="price of latency for --pair utilities, USD/h")
    ap.add_argument("--measure-arm", action="append", default=None,
                    help="measurement arm (repeatable, one per System-2 family, e.g. meta_shadow_s2lite): --pair "
                         "comparisons of live gates get the VoI readout (predicted vs. observed Delta-U)")
    ap.add_argument("--measure-run", default=None,
                    help="run directory holding the --measure-arm arms when they ran in another experiment (e.g. "
                         "runs/e2b_jevgym_meta_control for the E2c pairs); used for the VoI readout only")
    ap.add_argument("--replay", action="append", default=None,
                    help="replay JSONL(s) of other System-1 backends on the measurement arm (meta_gold replay): the "
                         "VoI readout of a live arm whose backend differs from the measurement arm's uses its replay")
    ap.add_argument("--single-baseline", action="store_true",
                    help="compare every arm with --baseline (default: arms of another System-2 model are compared "
                         "with that model's LLM-only arm, e.g. *_s2lite vs. llm_only_s2lite)")
    ap.add_argument("--keep-errors", action="store_true",
                    help="count episodes that ended in an infrastructure error as failures (the full-set sensitivity "
                         "view) instead of dropping their tasks from every paired comparison (PREREGISTRATION §1)")
    ap.add_argument("--out", default=None, help="write to this file instead of stdout")
    a = ap.parse_args(argv)
    if not Path(a.run_dir).is_dir():
        ap.error(f"not a directory: {a.run_dir}")
    ref = tuple(float(x) for x in a.hv_ref.split(",")) if a.hv_ref else None
    mruns = None
    if a.measure_run and a.measure_arm:
        mr = load_runs(a.measure_run)
        miss = [x for x in a.measure_arm if x not in mr]
        if miss:
            ap.error(f"measurement arm(s) {miss} not in {a.measure_run}")
        mruns = {x: mr[x] for x in a.measure_arm}
    reps = None
    if a.replay:
        from .meta_gold import load_replays

        reps = load_replays(a.replay)
    card = scorecard(a.run_dir, baseline=a.baseline, weak=a.weak, sweeps=a.sweep, sweep_param=a.sweep_param,
                     alpha=a.alpha, criterion=a.criterion, margin=a.margin, n_boot=a.n_boot, seed=a.seed,
                     ni_margin=a.ni_margin,
                     log_cost=not a.linear_cost, budget=a.budget, hv_ref=ref,  # type: ignore[arg-type]
                     family_baselines=not a.single_baseline,
                     pairs=[tuple(x.split(":", 1)) for x in a.pair] if a.pair else None,  # type: ignore[misc]
                     task_values=a.task_value, lam_per_hour=a.lam_per_hour, measure_arms=a.measure_arm,
                     measure_runs=mruns, replays=reps, drop_errors=not a.keep_errors)
    text = render(card, a.format)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
