#!/usr/bin/env python3
"""H3 (exploratory) exactly as registered: the mechanical rules of PREREGISTRATION A6.22 (c) -- delegation classes
credited at 5 % (A6.20 (e)), NI and the front (descriptive), the class ranking and its Kendall tau-b.

    python scripts/h3_delegation_classes.py [--runs runs] [--out results/secondary]

Analysis of existing logs only (no API call, no episode).

Universe (A6.22 (c), H3): the ACTING arms whose System-1 is Jev or Intern-Decision-4B, per suite x System-2 model.
L2 = the four JevGym evaluation configs (E2a, E2a', E2b, E2c: ``runs/<experiment name>``); L3 = the tau^2 airline TEST
tasks (the A5.4 (e) configurations: the CASCADE sweep, ASSIST k = 3 and the policy GUARD; 35 tasks x 3 trials) and
ALFWorld (E6, not run in Stage 1: absent). Shadow / measurement arms, references, controls (BM25, TF-IDF, embedding,
rate, random-score, heuristic, prior, s2self) and generative stand-in rows are outside it: an arm enters iff its logged
System-1 (``meta.s1.describe``) is Jev or ``internlm/Intern-Decision-4B`` and one of its decision points runs a System-1
mode (replace, cascade, assist, guard, trigger) -- reflection ``always`` / memory ``llm`` arms are LLM controls.

Class of an executed System-1 decision:
  meta                 reflection, routing, stop, memory_write, memory_read in trigger / guard / assist mode;
  bounded-recoverable  tool_select (or action) in assist or guard mode (tau^2's policy guard included), System-1 tool
                       choices executed in replace / cascade mode that are not irreversible, executed bounded
                       arguments (options without *Other*);
  copy-dependent       executed copy arguments (options ending in *Other*, A4's copy class);
  irreversible         System-1 tool choices executed in replace / cascade mode of an irreversible or final tool:
                       JevGym ``threshold_theory.JEVGYM_FATAL_TOOLS`` (cancel_order, refund, ship_order, finish); tau^2
                       WRITE tools and ``transfer_to_human_agents``.
Credit: class K iff its executed System-1 decisions are >= 5 % of the opportunities of the decision point at which
they are taken (tool-selection opportunities for tool choices, argument slots of the executed tool calls for argument
values, as in A6.22 (a) 5); assist, guard and trigger modes act at every opportunity and are credited whenever the arm
has one. No class -> *negligible delegation* (leaves the front analysis and the ranking). A multi-class arm ranks
under its riskiest class (irreversible > copy-dependent > bounded-recoverable > meta) and is a counterexample
candidate whenever it is credited with copy-dependent or irreversible.

NI and the front (descriptive): NI = the ``noninferior`` qualifier at the unadjusted level against the arm's suite x
System-2 LLM-only reference (A6.18 (c): ``metrics.ni_pvalue(...)["ni_95"]``; 3 pt on JevGym, 5 pt on tau^2 majority
pass); front = no other qualifying (credited) arm and not the reference dominates it in (cost, success)
(``metrics.pareto_front``). Ranking: per suite x System-2 and class, the median net cost savings over the class's
qualifying arms (0, the reference's, without one); Kendall tau-b over the four classes for Flash vs. Flash-Lite (L2)
and L2 vs. L3 (Flash), with the exact permutation p over the 24 orderings. Verdict: falsified iff a credited
copy-dependent or irreversible arm is NI and on the front, or a tau-b is < 0; otherwise "no registered
counterexample", which does not establish the ordering. Cross-family clause untested (one System-2 family, §5, A3).

Readings where the registration leaves a choice (printed in the output):
* reference: each arm's own run directory's LLM-only arm of its System-2 model (the score card's
  ``family_references``: the main LLM of the episodes decides, so routing arms -- Flash-Lite by default, Flash when
  the trigger fires -- belong to Flash, as in the E2b score card);
* the front pools the four L2 configs per System-2 model in paired relative coordinates (cost ratio 1 - savings and
  Delta-success against the arm's own reference, the reference at (1, 0)), because each config has its own LLM-only
  replicate; the within-config front in absolute (cost, success) means (the score card's view) is printed as a
  sensitivity column;
* bounded-recoverable is credited when either of its decision points reaches 5 % (tool choices of tool-selection
  opportunities, bounded arguments of argument slots); the tau^2 success of NI and the front is majority pass.

Outputs: ``<out>/h3.{json,md}``; ``scripts/secondary_tex.py`` emits ``\\res{secondary/h3/...}``."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import primaries as P  # noqa: E402
import tau2_proxy_check as PX  # noqa: E402
from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.scorecard import call_coverage  # noqa: E402
from jevharness.bench.threshold_theory import JEVGYM_FATAL_TOOLS  # noqa: E402
from jevharness.harness.tools import OTHER  # noqa: E402
from jevharness.harness.trajectory import iter_records  # noqa: E402

REGISTERED = "PREREGISTRATION §3 H3 made mechanical by A6.22 (c); A6.20 (e) (5 % rule); A6.18 (c) (noninferior qualifier)"
CLASSES = ("meta", "bounded-recoverable", "copy-dependent", "irreversible")      # riskiest last
RISK = {c: i for i, c in enumerate(CLASSES)}
META_DPS = frozenset({"reflection", "routing", "stop", "memory_write", "memory_read"})
EVERY_OPP_MODES = frozenset({"trigger", "guard", "assist"})
S1_MODES = frozenset({"replace", "cascade", "assist", "guard", "trigger"})
TOOL_DPS = ("tool_select", "action")
CREDIT = 0.05                                                                  # A6.20 (e)
L2_RUNS = ("E2a", "E2a_intern", "E2b", "E2c")
L2_MARGIN, L3_MARGIN = 0.03, 0.05
INTERN_4B = "internlm/Intern-Decision-4B"
# tau^2 airline ToolType.WRITE tools (tau2-bench 5bfa7e3, ``tau2.domains.airline.tools``; ``bench.suites.tau2._tool_kind``)
TAU2_AIRLINE_WRITE = frozenset({"book_reservation", "cancel_reservation", "send_certificate",
                                "update_reservation_baggages", "update_reservation_flights",
                                "update_reservation_passengers"})
TAU2_IRREVERSIBLE = TAU2_AIRLINE_WRITE | {"transfer_to_human_agents"}
S2_SHORT = {"gemini-3.5-flash": "flash", "gemini-3.5-flash-lite": "lite", "gemini/gemini-3.5-flash": "flash",
            "gemini/gemini-3.5-flash-lite": "lite"}


# ====================================================================== classes of executed System-1 decisions
def _modes(meta: Mapping[str, Any]) -> dict[str, str]:
    """dp -> mode from a logged harness config (JevGym: ``meta.config``; tau^2: ``meta.config.harness`` plus its
    ``policy_guard``)."""
    cfg = (meta or {}).get("config") or {}
    h = cfg.get("harness") if isinstance(cfg.get("harness"), Mapping) else cfg
    out = {k: str(v.get("mode", "off")) for k, v in h.items() if isinstance(v, Mapping) and "mode" in v}
    pg = cfg.get("policy_guard")
    if isinstance(pg, Mapping) and "mode" in pg:
        out["policy"] = str(pg.get("mode", "off"))
    return out


def is_copy(r: Mapping[str, Any]) -> bool:
    """A4's copy class: ``gold_aux.param_kind == "copy"`` (JevGym), else an option list ending in *Other* (tau^2)."""
    kind = (r.get("gold_aux") or {}).get("param_kind")
    if kind is not None:
        return kind == "copy"
    opts = r.get("options") or []
    return bool(opts) and opts[-1] == OTHER


def tool_opportunities(recs: Iterable[Mapping[str, Any]], dp: str = "tool_select") -> int:
    """Tool-selection opportunities: one per agent step, a guarded step once (``metrics.delegation_ratio``'s
    denominator, A6.22 (a) 5)."""
    seen: set[tuple[Any, ...]] = set()
    n = 0
    for r in recs:
        if r.get("dp") != dp or r.get("mode") not in ("off", "shadow", "replace", "cascade", "assist", "guard"):
            continue
        if r.get("mode") == "guard":
            key = (r.get("config"), r.get("task_id"), r.get("step"))
            if key in seen:
                continue
            seen.add(key)
        n += 1
    return n


def classify(trajs: Sequence[Mapping[str, Any]], irreversible_tools: Iterable[str]) -> dict[str, Any]:
    """Counts, shares and credited classes of one arm's executed System-1 decisions (A6.22 (c), H3)."""
    fatal = {str(t) for t in irreversible_tools}
    recs = M.decision_records(trajs)  # type: ignore[arg-type]
    n_ts = tool_opportunities(recs, "tool_select") + tool_opportunities(recs, "action")
    slots = int(call_coverage(trajs, recs)["n_arg_slots"])
    tool_irrev = tool_bounded = arg_copy = arg_bounded = 0
    every: Counter[str] = Counter()          # (class) -> decisions in a mode that acts at every opportunity
    unknown: Counter[str] = Counter()
    irrev_tools: Counter[str] = Counter()
    for r in recs:
        dp, mode = r.get("dp"), r.get("mode")
        if mode in (None, "off", "shadow", "generate"):
            continue
        if dp in META_DPS and mode in EVERY_OPP_MODES:
            every["meta"] += 1
        elif dp in TOOL_DPS and mode in ("assist", "guard"):
            every["bounded-recoverable"] += 1
        elif dp == "policy" and mode == "guard":          # tau^2's policy guard
            every["bounded-recoverable"] += 1
        elif dp in TOOL_DPS and mode in ("replace", "cascade"):
            if M.is_delegated(r):
                if str(r.get("s1_value")) in fatal:
                    tool_irrev += 1
                    irrev_tools[str(r.get("s1_value"))] += 1
                else:
                    tool_bounded += 1
        elif dp == "tool_params" and mode in ("replace", "cascade"):
            if M.is_delegated(r):
                if is_copy(r):
                    arg_copy += 1
                else:
                    arg_bounded += 1
        else:
            unknown[f"{dp}/{mode}"] += 1

    def share(k: int, n: int) -> float:
        return k / n if n else math.nan

    sh = {"irreversible_tool": share(tool_irrev, n_ts), "bounded_tool": share(tool_bounded, n_ts),
          "bounded_arg": share(arg_bounded, slots), "copy_arg": share(arg_copy, slots)}

    def ok(x: float) -> bool:
        return not math.isnan(x) and x >= CREDIT

    credited = []
    if every["meta"]:
        credited.append("meta")
    if every["bounded-recoverable"] or ok(sh["bounded_tool"]) or ok(sh["bounded_arg"]):
        credited.append("bounded-recoverable")
    if ok(sh["copy_arg"]):
        credited.append("copy-dependent")
    if ok(sh["irreversible_tool"]):
        credited.append("irreversible")
    return {"n_episodes": len(trajs), "tool_opportunities": n_ts, "arg_slots": slots,
            "counts": {"irreversible_tool": tool_irrev, "bounded_tool": tool_bounded, "bounded_arg": arg_bounded,
                       "copy_arg": arg_copy, "every_opportunity": dict(every)},
            "irreversible_tools": dict(irrev_tools), "shares": sh, "credited": credited,
            "riskiest": max(credited, key=RISK.get) if credited else None,
            "negligible": not credited, "unclassified_modes": dict(unknown)}


# ====================================================================== universe
def first_meta(path: Path) -> tuple[dict[str, Any], str | None]:
    """(meta, main LLM) of the first record of an arm file; the record's ``backend`` is kept as ``meta["_backend"]``."""
    for r in iter_records(path):
        if "task_id" in r:
            return {**dict(r.get("meta") or {}), "_backend": r.get("backend")}, r.get("llm")
    return {}, None


def s1_identity(meta: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """(backend, model) of the arm's System-1: the logged ``meta.s1.describe`` (JevGym), else the logged spec
    ``meta.config.s1`` (tau^2), else the record's ``backend``."""
    d = ((meta.get("s1") or {}).get("describe")) or {}
    if d.get("backend"):
        return d.get("backend"), d.get("model")
    spec = ((meta.get("config") or {}).get("s1")) or {}
    if isinstance(spec, Mapping) and spec.get("type"):
        return str(spec["type"]), spec.get("model")
    return meta.get("_backend"), None


def in_universe(meta: Mapping[str, Any]) -> tuple[bool, str]:
    """A6.22 (c): acting Jev / Intern-Decision-4B arm? (and why not)."""
    backend, model = s1_identity(meta)
    modes = _modes(meta)
    acting = sorted(dp for dp, m in modes.items() if m in S1_MODES)
    if not acting:
        nonoff = sorted({m for m in modes.values() if m != "off"})
        return False, ("reference (every decision point off)" if not nonoff else
                       "shadow / measurement arm" if set(nonoff) <= {"shadow"} else
                       f"no System-1 mode (LLM control: {', '.join(nonoff)})")
    if backend == "jev":
        return True, "Jev"
    if backend == "intern_decision" and model == INTERN_4B:
        return True, "Intern-Decision-4B"
    return False, f"System-1 {backend or '?'}{' ' + model if model else ''} (control / stand-in / other size)"


def acting_modes(meta: Mapping[str, Any]) -> dict[str, str]:
    return {dp: m for dp, m in _modes(meta).items() if m in S1_MODES}


# ====================================================================== L2 (JevGym)
def l2_arms(runs: Path, flags: P.Flags, n_boot: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                                         dict[str, Any]]:
    """Universe arms of the four JevGym configs with their class block and the paired comparison against their run's
    LLM-only reference of the same System-2 model; (rows, excluded, references)."""
    rows, excluded, refs_out = [], [], {}
    for key in L2_RUNS:
        cfg = P.CONFIGS[key]
        exp_name = P.PC.load(cfg).get("name")
        R = P.EpisodeRun(key, cfg, runs / exp_name, flags, None, False)
        if not R.exists:
            flags.error("H3", f"{exp_name}: run directory missing")
            continue
        files = sorted((runs / exp_name).glob("*.jsonl"))
        refs: dict[str, str] = {}
        for nm in ("llm_only", "llm_only_s2lite"):
            p = runs / exp_name / f"{nm}.jsonl"
            if p.exists():
                _, llm = first_meta(p)
                refs[S2_SHORT.get(str(llm), str(llm))] = nm
        refs_out[key] = refs
        for p in files:
            meta, llm = first_meta(p)
            ok, why = in_universe(meta)
            s2 = S2_SHORT.get(str(llm), str(llm))
            if not ok:
                excluded.append({"suite": "L2", "run": key, "arm": p.stem, "reason": why})
                continue
            ref = refs.get(s2)
            if ref is None:
                flags.error("H3", f"{key}/{p.stem}: no LLM-only reference for System-2 {llm}")
                continue
            A = R.arm(p.stem, keep_trajs=True)
            B = R.arm(ref)
            exp = R.expected(p.stem, ref)
            ta, tb = P.task_table(A, exp), P.task_table(B, exp)
            pt = P.pair_tables(tb, ta, exp, names=(ref, p.stem))
            blk = P.ni_block(pt["ref_y"], pt["arm_y"], L2_MARGIN, n_boot, seed)
            cb = P.cost_block(tb, ta, pt["keys"], n_boot, seed)
            keep = set(pt["keys"])
            trajs = [t for t in A["trajs"] if P.task_key(t) in keep and not t.get("error")]
            cls = classify(trajs, JEVGYM_FATAL_TOOLS)
            # within-config absolute view (the score card's): the arm's own task means, the reference over its tasks
            ref_all = [e for e in B["episodes"].values() if not e["error"]]
            rows.append({"suite": "L2", "run": key, "run_dir": P._rel(R.dir), "arm": p.stem, "id": f"{key}/{p.stem}",
                         "s1": why, "s2": s2, "s2_model": llm, "reference": ref, "modes": acting_modes(meta),
                         "n": pt["n_pairs"], "n_dropped": pt["n_dropped"], "delta": blk["diff"], "delta_ci": blk["ci95"],
                         "ni": bool(blk["ni_95"]), "p_decisive": blk["p_decisive"], "rule": blk["rule"],
                         "margin": L2_MARGIN, "cost_savings": cb.get("cost_savings", math.nan),
                         "cost_arm": cb.get("cost_arm"), "cost_ref": cb.get("cost_ref"),
                         "success_arm": blk["success_arm"], "success_ref": blk["success_ref"],
                         "abs_ref": {"cost": float(np.mean([e["cost"] for e in ref_all])),
                                     "success": float(np.mean([e["success"] for e in ref_all])), "n": len(ref_all)},
                         "file": A["path"], "sha256": A["sha256"], **cls})
            R.arms[p.stem] = None                                   # free the trajectories
    return rows, excluded, refs_out


# ====================================================================== L3 (tau^2 airline test tasks)
def l3_arms(tau2_dir: Path, flags: P.Flags, n_boot: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = json.loads(P.AIRLINE_SPLIT.read_text())["split"]
    exp = [f"airline/{t}" for t in split["test"]]
    keep = set(exp)
    ref = P.load_arm(tau2_dir / "airline" / "llm_only.jsonl")
    if ref is None:
        flags.error("H3", f"{tau2_dir}/airline/llm_only.jsonl missing: L3 absent")
        return [], {}
    tb = P.task_table(ref, exp, trials=PX.TAU2_TRIALS, retry_rule=False)
    rows = []
    for name, _, arm, _, tp in PX.CONFIGS:
        if arm == "llm_only":
            continue
        path = tau2_dir / "airline" / f"{arm}.jsonl"
        A = P.load_arm(path, keep_trajs=True)
        if A is None:
            flags.warn("H3", f"tau^2 arm {arm} missing")
            continue
        meta, llm = first_meta(path)
        ok, why = in_universe(meta)
        if not ok:
            flags.error("H3", f"tau^2 {arm}: not a Jev acting arm ({why})")
            continue
        ta = P.task_table(A, exp, trials=PX.TAU2_TRIALS, retry_rule=False)
        pt = P.pair_tables(tb, ta, exp, names=("llm_only", arm))
        blk = P.ni_block(pt["ref_y"], pt["arm_y"], L3_MARGIN, n_boot, seed)
        cb = P.cost_block(tb, ta, pt["keys"], n_boot, seed)
        trajs = [t for t in A["trajs"] if P.task_key(t) in keep and not t.get("error")]
        cls = classify(trajs, TAU2_IRREVERSIBLE)
        ref_eps = [e for e in ref["episodes"].values() if e["task"] in keep and not e["error"]]
        rows.append({"suite": "L3", "run": "tau2_airline_test", "run_dir": "runs/tau2/airline (35 test tasks x 3)",
                     "arm": arm, "id": f"tau2/{arm}", "configuration": name, "s1": why,
                     "s2": S2_SHORT.get(str(llm), str(llm)), "s2_model": llm, "reference": "llm_only",
                     "modes": acting_modes(meta), "n": pt["n_pairs"], "n_dropped": pt["n_dropped"],
                     "delta": blk["diff"], "delta_ci": blk["ci95"], "ni": bool(blk["ni_95"]),
                     "p_decisive": blk["p_decisive"], "rule": blk["rule"], "margin": L3_MARGIN,
                     "endpoint": "majority pass (>= 2 of 3)", "cost_savings": cb.get("cost_savings", math.nan),
                     "cost_arm": cb.get("cost_arm"), "cost_ref": cb.get("cost_ref"),
                     "success_arm": blk["success_arm"], "success_ref": blk["success_ref"],
                     "abs_ref": {"cost": float(np.mean([tb[k]["cost"] for k in exp])),
                                 "success": float(np.mean([tb[k]["y"] for k in exp])), "n": len(exp)},
                     "file": A["path"], "sha256": A["sha256"], **cls})
    return rows, {"tau2": {"llm_only": ref["path"], "sha256": ref["sha256"], "n_tasks": len(exp)}}


# ====================================================================== front, ranking, verdict
def fronts(rows: list[dict[str, Any]]) -> None:
    """Front membership (in place) of every credited NI arm: pooled per suite x System-2 in relative coordinates
    (primary reading) and within its own run in absolute means (sensitivity)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        r["front"] = r["front_within_run"] = None
        if not r["negligible"] and r["ni"]:
            groups.setdefault((r["suite"], r["s2"]), []).append(r)
    for (suite, s2), rs in groups.items():
        pts = [(r["id"], 1.0 - r["cost_savings"], r["delta"]) for r in rs] + [("__reference__", 1.0, 0.0)]
        front = set(M.pareto_front(pts))
        for r in rs:
            r["front"] = r["id"] in front
            r["front_group"] = f"{suite}/{s2}: {len(rs)} qualifying credited arm(s) + the reference"
    by_run: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rs in groups.values():
        for r in rs:
            by_run.setdefault((r["run"], r["s2"]), []).append(r)
    for (_, _), rs in by_run.items():
        ref = rs[0]["abs_ref"]
        pts = [(r["id"], float(r["cost_arm"]), float(r["success_arm"])) for r in rs] + \
              [("__reference__", ref["cost"], ref["success"])]
        front = set(M.pareto_front(pts))
        for r in rs:
            r["front_within_run"] = r["id"] in front


def ranking(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per suite x System-2 and class: the median net cost savings over the class's qualifying arms (ranked under
    their riskiest class), 0 without one."""
    out: dict[str, Any] = {}
    for suite, s2 in (("L2", "flash"), ("L2", "lite"), ("L3", "flash")):
        g: dict[str, Any] = {}
        for c in CLASSES:
            q = [r for r in rows if r["suite"] == suite and r["s2"] == s2 and r["riskiest"] == c and r["ni"]]
            vals = [float(r["cost_savings"]) for r in q]
            g[c] = {"median": float(np.median(vals)) if vals else 0.0, "n_qualifying": len(q),
                    "arms": [r["id"] for r in q],
                    "n_credited": sum(1 for r in rows if r["suite"] == suite and r["s2"] == s2 and r["riskiest"] == c)}
        out[f"{suite}/{s2}"] = g
    return out


def taus(rank: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for name, (a, b) in (("flash_vs_lite_L2", ("L2/flash", "L2/lite")), ("L2_vs_L3_flash", ("L2/flash", "L3/flash"))):
        x = [rank[a][c]["median"] for c in CLASSES]
        y = [rank[b][c]["median"] for c in CLASSES]
        out[name] = {"x": a, "y": b, "x_medians": x, "y_medians": y, **PX.kendall_permutation(x, y)}
    return out


def verdict(rows: Sequence[Mapping[str, Any]], kt: Mapping[str, Any]) -> dict[str, Any]:
    cands = [r for r in rows if {"copy-dependent", "irreversible"} & set(r["credited"])]
    counter = [r for r in cands if r["ni"] and r["front"]]
    counter_run = [r for r in cands if r["ni"] and r["front_within_run"]]
    neg = {k: v["tau_b"] for k, v in kt.items() if not math.isnan(v["tau_b"]) and v["tau_b"] < 0}
    falsified = bool(counter) or bool(neg)
    reasons = [f"{r['id']} (credited {', '.join(r['credited'])}; n = {r['n']}) is NI and on the front" for r in counter]
    reasons += [f"tau-b {k} = {v:.3f} < 0" for k, v in neg.items()]
    return {"verdict": "falsified" if falsified else "no registered counterexample",
            "counterexamples": [r["id"] for r in counter], "negative_tau_b": neg, "reasons": reasons,
            "candidates": [{"id": r["id"], "credited": r["credited"], "ni": r["ni"], "front": r["front"],
                            "front_within_run": r["front_within_run"], "n": r["n"]} for r in cands],
            "sensitivity_within_run_front": {
                "counterexamples": [r["id"] for r in counter_run],
                "verdict": "falsified" if (counter_run or neg) else "no registered counterexample"},
            "note": ("'no registered counterexample' does not establish the ordering (A6.22 (c)); the cross-family "
                     "clause is untested while only one System-2 family (Gemini) runs (§5, A3)")}


READINGS = [
    "reference: the arm's own run's LLM-only arm of its System-2 model (score card `family_references`: the episodes' "
    "main LLM decides, so routing arms count as Flash, as in the E2b score card)",
    "front: pooled per suite x System-2 in paired relative coordinates (1 - savings, Delta-success vs the arm's own "
    "reference; the reference at (1, 0)); the within-run absolute (cost, success) front is a sensitivity column",
    "bounded-recoverable is credited when either of its decision points reaches 5 % (tool choices / tool-selection "
    "opportunities, bounded arguments / argument slots); assist / guard / trigger whenever the arm has an opportunity",
    "tau^2: the A5.4 (e) configurations on the 35 airline test tasks x 3 (majority pass for NI and the front); "
    "irreversible = WRITE tools of tau2-bench's airline toolkit + transfer_to_human_agents; ALFWorld (E6) not run",
    "Kendall tau-b over the four class medians; exact permutation p over the 24 orderings (two-sided; p_less = "
    "P(tau <= observed) as the reversal side); tau-b undefined when a vector is constant",
]


def _f(x: Any, nd: int = 1, pct: bool = False, sign: bool = False) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    if math.isnan(v):
        return "–"
    v = 100 * v if pct else v
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _yn(x: Any) -> str:
    return "–" if x is None else ("yes" if x else "no")


def to_markdown(res: Mapping[str, Any]) -> str:
    v = res["verdict"]
    L = ["# H3 — delegation classes (exploratory; A6.22 (c) mechanical rules)", "",
         f"- generated {res['generated']} at git `{res['git']}` by `scripts/h3_delegation_classes.py`",
         f"- registered: {res['registered']}", f"- **verdict: {v['verdict']}**"]
    L += [f"  - {x}" for x in v["reasons"]]
    L += [f"- sensitivity (within-run absolute front): {v['sensitivity_within_run_front']['verdict']}"
          + (f" ({', '.join(v['sensitivity_within_run_front']['counterexamples'])})"
             if v['sensitivity_within_run_front']['counterexamples'] else ""),
          f"- {v['note']}", f"- universe: {res['n_universe']} acting Jev / Intern-Decision-4B arms "
          f"({res['n_negligible']} negligible delegation, leaving the analysis); excluded arms: {len(res['excluded'])}",
          f"- absent: {', '.join(res['absent'])}"]
    L += [f"- reading: {r}" for r in res["readings"]]
    L += ["", "## Kendall τ-b of the class ranking (medians of net savings over qualifying arms, 0 without one)", "",
          "| comparison | meta | bounded-rec. | copy-dep. | irreversible | τ-b | p two-sided | p (τ ≤ obs.) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, t in res["kendall"].items():
        L.append(f"| {t['x']} | " + " | ".join(_f(m, 1, True) for m in t["x_medians"]) + " | | | |")
        L.append(f"| {t['y']} ({k}) | " + " | ".join(_f(m, 1, True) for m in t["y_medians"])
                 + f" | {_f(t['tau_b'], 3)} | {_f(t['p_two_sided'], 3)} | {_f(t['p_less'], 3)} |")
    L += ["", "## Class medians", "", "| suite / System-2 | class | qualifying arms | credited arms | median savings % |",
          "|---|---|---|---|---|"]
    for g, cl in res["ranking"].items():
        for c in CLASSES:
            x = cl[c]
            L.append(f"| {g} | {c} | {x['n_qualifying']} ({', '.join(x['arms']) or '–'}) | {x['n_credited']} | "
                     f"{_f(x['median'], 1, True)} |")
    L += ["", "## Arms", "",
          "| suite | System-2 | arm | System-1 | n | credited (riskiest first) | irrev. tool % | bounded tool % | bounded arg % | copy arg % | Δ pt | NI | savings % | front | front (run) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res["arms"], key=lambda r: (r["suite"], r["s2"], -RISK.get(r["riskiest"] or "", -1), r["id"])):
        cr = ", ".join(sorted(r["credited"], key=lambda c: -RISK[c])) or "negligible"
        sh = r["shares"]
        L.append(f"| {r['suite']} | {r['s2']} | `{r['id']}` | {r['s1']} | {r['n']} | {cr} | "
                 f"{_f(sh['irreversible_tool'], 1, True)} | {_f(sh['bounded_tool'], 1, True)} | "
                 f"{_f(sh['bounded_arg'], 1, True)} | {_f(sh['copy_arg'], 1, True)} | {_f(r['delta'], 1, True, True)} | "
                 f"{_yn(r['ni'])} | {_f(r['cost_savings'], 1, True)} | {_yn(r['front'])} | {_yn(r['front_within_run'])} |")
    L += ["", "## Excluded arms", ""]
    L += [f"- {x['suite']} {x['run']} `{x['arm']}`: {x['reason']}" for x in res["excluded"]]
    if res["flags"]:
        L += ["", "## Flags", ""] + [f"- {f['level']} {f['where']}: {f['msg']}" for f in res["flags"]]
    return "\n".join(L) + "\n"


def build(runs: Path, tau2_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    flags = P.Flags()
    l2, excluded, refs = l2_arms(runs, flags, n_boot, seed)
    l3, refs3 = l3_arms(tau2_dir, flags, n_boot, seed)
    rows = l2 + l3
    fronts(rows)
    rank = ranking([r for r in rows if not r["negligible"]])
    kt = taus(rank)
    v = verdict([r for r in rows if not r["negligible"]], kt)
    return {"registered": REGISTERED, "readings": READINGS, "verdict": v, "kendall": kt, "ranking": rank,
            "arms": rows, "excluded": excluded, "references": {**refs, **refs3},
            "n_universe": len(rows), "n_negligible": sum(1 for r in rows if r["negligible"]),
            "absent": ["L3 ALFWorld (E6 not run in Stage 1)"], "flags": list(flags.items)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--tau2", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "secondary"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    runs = Path(a.runs).resolve()
    res = build(runs, Path(a.tau2).resolve() if a.tau2 else runs / "tau2", a.n_boot, a.seed)
    res = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": P._git_head(), **res}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "h3.json").write_text(json.dumps(P._jsonable(res), indent=1, default=str))
    (out / "h3.md").write_text(to_markdown(res))
    print(f"H3: {res['verdict']['verdict']}; " + "; ".join(res["verdict"]["reasons"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
