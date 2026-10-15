"""Analysis of the meta-decision golds (docs/META_GOLD_SPEC.md §6.2, §7.1, §9, §12).

    python -m jevharness.bench.meta_gold replay RUN_DIR --arm meta_shadow_s2lite --s1 '<backend spec>' [--workers 4]
        score another System-1 backend on the logged (state, question) pairs of an arm -- the yes/no records of
        ``steps[]`` and the probes -- with exactly the logged question text: paired, state-identical comparisons
        (free for the rule / rate / lexical backends) -> RUN_DIR/replay/<arm>__<backend>.jsonl
    python -m jevharness.bench.meta_gold audit RUN_DIR --arm ARM [--implications]
        recompute the JevGym golds of the executed actions by deterministic replay of the logged actions (the
        reference implementation; §1.6) and list every label that differs from the online log; ``--implications``
        adds the A6 / P0-8 implication audit over every logged trajectory (executed action with a failing verdict =>
        the episode failed; an executed finish with finish_ok => success; violations must be 0), the share of
        need_fatal positives whose fatal_kind is ``budget`` (oracle-relative) and the H2 gate stats with those
        positives excluded (the sensitivity row)
    python -m jevharness.bench.meta_gold h2 RUN_DIR --measure meta_shadow_s2lite --pair LIVE:CONTROL ... [--W 0.1 1 10]
            [--measure-run MEASURE_RUN_DIR] [--replay RUN/replay/<arm>__<backend>.jsonl ...]
        the H2 input table: one cell per (gate x System-1 backend x System-2 family) with pi, AUROC, J and accuracy
        at the live tau of the LIVE backend on the measurement arm (its own log when the spec ids match, else its
        replay; a cell with neither is not estimable), estimability (>= 30 / >= 30), the paired gross / net Delta-U of
        the live arm against its matched-rate control and the ±2 pt TOST; then the A6.16 (i) primary H2 readout
        (``h2_theory``: Eq. 2's predicted gross Delta-U against the observed gross / net Delta-U per cell with CIs from
        one JOINT task bootstrap -- the shared task ids resampled once per replicate for every cell, every predictor
        recomputed -- and, with >= 4 estimable cells of full column rank, the through-origin slope of observed on
        predicted; ``h2_verdict`` words it: consistent / inconsistent with Eq. 2, or *not estimable*, never
        falsified), the pooled regression of Delta-U on J and accuracy now labelled EXPLORATORY
        (``h2_regression_exploratory``; the pre-A6.16 key ``regression_j`` still points at it), the Eq. 2 predicted
        gain per cell (``predicted_gain``) and the AUROC regression as the tau-free bound (``regression_auroc``)
    python -m jevharness.bench.meta_gold ceiling RUN_DIR --measure meta_shadow_s2lite [--dp reflection] [--W 1]
        the C5 one-step oracle ceiling W n pi (1 - pi) (e + h) with a task-bootstrap CI and the "no headroom" rule
    python -m jevharness.bench.meta_gold coverage [--seeds 5000-5999] [--suite '<yaml>']
        the copy-parameter coverage valve of the free-text policy (§7.1): per copy parameter, the share of dev-seed
        oracle states whose raw value is among the System-1 candidates; < 0.8 -> reclassified free

The one-step counterfactual validity check (review round 4, P1-6) lives in ``scripts/cf_validity.py``; its
opportunity listing, stratified subset and agreement table are :func:`cf_opportunities`,
:func:`select_cf_validity_subset` and :func:`cf_validity_table` here (offline, testable).

Golds never reach a replayed backend: the replay sends only the logged state string and question text
(``Question.meta`` stays empty). Simulated runs are flagged like everywhere else (pipeline tests only).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

from ..harness.trajectory import load_trajectories
from .metrics import (FAILING_VERDICTS, MIN_POS_NEG, cf_efficacy, first_opportunities, gate_stats, p_yes_of,
                      paired_bootstrap_diff, tost_paired, voi_identity, wilson_ci)

NAN = float("nan")
GATES = ("reflection", "routing", "stop_guard", "stop_trigger", "memory_write")
BRANCH_OF = {"reflection": "verdict_alt", "routing": "verdict_large", "stop_guard": "verdict_reject"}
NO_HEADROOM_MDE = 0.033          # the pre-registered MDE (3.3 pt of success at W = 1)


# ====================================================================== records
def gate_records(trajs: Iterable[Mapping[str, Any]], gate: str, modes: Sequence[str] | None = None,
                 probes: bool = False) -> list[dict[str, Any]]:
    """The yes/no records of one gate (``stop_guard`` = stop records at finish proposals, ``stop_trigger`` = at the
    other steps), flattened with task_id / step / episode success. ``probes`` adds the probe-list records of that
    gate (never used for H2: augmented / wording records are descriptive only)."""
    dp = "stop" if gate.startswith("stop") else gate
    out = []
    for t in trajs:
        steps = list(t.get("steps") or [])
        srcs: list[tuple[Any, Mapping[str, Any]]] = [(st.get("step"), d) for st in steps for d in st.get("decisions") or []]
        if probes:
            srcs += [((p.get("gold_aux") or {}).get("step"), p) for p in t.get("probes") or []]
        for step, d in srcs:
            if d.get("dp") != dp or d.get("options"):
                continue
            if modes is not None and d.get("mode") not in modes:
                continue
            if gate == "stop_guard" and not (d.get("gold_aux") or {}).get("kind") and \
                    not str(d.get("question", "")).startswith("The agent wants to stop"):
                continue
            if gate == "stop_trigger" and ((d.get("gold_aux") or {}).get("kind")
                                           or str(d.get("question", "")).startswith("The agent wants to stop")):
                continue
            out.append({**d, "task_id": t.get("task_id"), "step": step, "config": t.get("config"),
                        "episode_success": bool((t.get("result") or {}).get("success"))})
    return out


def arm_tau(trajs: Sequence[Mapping[str, Any]], gate: str) -> float | None:
    dp = "stop" if gate.startswith("stop") else gate
    for t in trajs[:5]:
        c = ((t.get("meta") or {}).get("config") or {}).get(dp) or {}
        if c.get("mode") in ("trigger", "guard"):
            return float(c.get("tau", 0.8))
    return None


def arm_gate(trajs: Sequence[Mapping[str, Any]]) -> str | None:
    """The gate a live arm puts System-1 on (its one trigger / guard meta decision point)."""
    for t in trajs[:5]:
        c = (t.get("meta") or {}).get("config") or {}
        for dp in ("reflection", "routing", "memory_write"):
            if (c.get(dp) or {}).get("mode") == "trigger":
                return dp
        m = (c.get("stop") or {}).get("mode")
        if m in ("guard", "trigger"):
            return f"stop_{m}"
    return None


def arm_backend_id(trajs: Sequence[Mapping[str, Any]]) -> str | None:
    """The System-1 spec id an arm logged (``meta.s1.id``: a hash of the backend's full ``describe()``), or None for
    older logs. Backends are matched on it, never on the backend name alone."""
    for t in trajs[:5]:
        i = ((t.get("meta") or {}).get("s1") or {}).get("id")
        if i:
            return str(i)
    return None


def arm_backend_label(trajs: Sequence[Mapping[str, Any]]) -> str:
    """A readable backend label (name and model) of an arm, for notes."""
    for t in trajs[:5]:
        d = ((t.get("meta") or {}).get("s1") or {}).get("describe") or {}
        if d:
            return f"{d.get('backend')}:{d.get('model')}"
    return str(trajs[0].get("backend")) if trajs else "?"


def _s2_models_besides_main(trajs: Sequence[Mapping[str, Any]]) -> set[str]:
    """Models of an arm's acting System-2 calls other than its main LLM (a router's small model)."""
    out: set[str] = set()
    for t in trajs:
        for c in t.get("calls") or []:
            if c.get("component") == "s2" and not (c.get("meta") or {}).get("shadow") and c.get("model") \
                    and c.get("model") != t.get("llm"):
                out.add(str(c["model"]))
    return out


def measure_arm_for(runs: Mapping[str, Sequence[Mapping[str, Any]]], measure_arms: Sequence[str], live: str,
                    gate: str) -> str | None:
    """The measurement arm whose shadow golds belong to a live gate arm's H2 cell / VoI readout. Only measurement arms
    with shadow records of that gate qualify. Routing: the arm whose System-2 is the router's *small* model -- the
    routers' default policy, shadowed in ``meta_shadow_s2lite`` (§3.2) -- because a router's ``traj.llm`` is the large
    model, whose measurement arm does not shadow routing (round-3 review: the routing cell always came out "no
    opportunity"). Every other gate: the arm of the live arm's own System-2 family."""
    from .scorecard import arm_s2_model

    cands = [a for a in measure_arms if a in runs and gate_records(runs[a], gate, modes=("shadow",))]
    if not cands or live not in runs:
        return None
    if gate == "routing":
        small = _s2_models_besides_main(runs[live])
        hit = [a for a in cands if arm_s2_model(runs[a]) in small]
        if hit:
            return hit[0]
        return cands[0] if len(cands) == 1 else None
    fam = arm_s2_model(runs[live])
    return next((a for a in cands if arm_s2_model(runs[a]) == fam), None)


# ====================================================================== replay of other backends (§6.2)
def replay_rows(trajs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The (state, question) pairs a backend replay scores: every yes/no record with a logged state, in steps[]
    (source ``steps``) and in probes (their ``probe_source``). Only ``source == "steps"`` shadow rows enter H2
    (:func:`h2_replay_records`); the probe-list rows (guard probes, the wording ablation, augmented stop-trigger
    queries) feed the probe tables only."""
    rows = []
    for t in trajs:
        for st in t.get("steps") or []:
            for d in st.get("decisions") or []:
                if d.get("options") or not d.get("state") or not isinstance(d.get("s1_dist"), Mapping) \
                        or "yes" not in d["s1_dist"]:
                    continue
                rows.append({"task_id": t.get("task_id"), "step": st.get("step"), "dp": d["dp"], "mode": d["mode"],
                             "source": "steps", "question": d["question"], "state": d["state"], "gold": d.get("gold"),
                             "gold_status": d.get("gold_status"), "gold_aux": d.get("gold_aux"),
                             "logged_p_yes": p_yes_of(d)})
        for p in t.get("probes") or []:
            if not p.get("state"):
                continue
            a = p.get("gold_aux") or {}
            rows.append({"task_id": t.get("task_id"), "step": a.get("step"), "dp": p["dp"], "mode": p["mode"],
                         "source": a.get("probe_source"), "question": p["question"], "state": p["state"],
                         "gold": p.get("gold"), "gold_status": p.get("gold_status"), "gold_aux": a,
                         "logged_p_yes": p_yes_of(p)})
    return rows


def replay_backend(trajs: Sequence[Mapping[str, Any]], backend: Any, workers: int = 1,
                   arm: str | None = None) -> list[dict[str, Any]]:
    """Score ``backend`` on the logged pairs; one single-question call per pair (never batched). The question goes
    out with its logged text and decision point only -- no gold in ``Question.meta``. Rows carry the backend's spec
    id (``backend_id``, as logged by live arms in ``meta.s1.id``) and the replayed ``arm``."""
    from ..accounting import Ledger
    from ..decision.base import backend_id
    from ..decision.types import Noul

    rows = replay_rows(trajs)
    desc = backend.describe() or {}
    name = desc.get("backend", getattr(backend, "name", "backend"))
    bid = backend_id(desc)

    def one(r: dict[str, Any]) -> dict[str, Any]:
        led = Ledger()
        try:
            a = backend.ask(r["state"], Noul(r["question"], decision_point=r["dp"]), ledger=led)
            p = a.p_yes
            err = None
        except Exception as e:  # noqa: BLE001 - a failed replay is a missing row, never a crash
            p, err = None, f"{type(e).__name__}: {str(e)[:200]}"
        out = {k: v for k, v in r.items() if k != "state"}
        out.update({"backend": name, "backend_id": bid, "arm": arm, "p_yes": p,
                    "s1_dist": {"yes": p, "no": None if p is None else 1 - p},
                    "cost_usd": sum(x.cost_usd for x in led.records), "error": err})
        return out

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, rows))
    return [one(r) for r in rows]


# ====================================================================== deterministic replay of the golds (§1.6, §12)
def suite_factories(run_dir: str | Path) -> dict[str, Callable[[], Any]]:
    """task_id -> env factory of the suite in RUN_DIR/experiment.yaml (JevGym only needs no data)."""
    from .suites import load_suite

    exp = yaml.safe_load((Path(run_dir) / "experiment.yaml").read_text()) or {}
    kw = dict(exp.get("suite") or {})
    facs = load_suite(kw.pop("name"), **kw)
    return {f().task_id: f for f in facs}


def replay_golds(traj: Mapping[str, Any], env: Any) -> list[dict[str, Any]]:
    """Recompute, by replaying the logged actions through a fresh env, the golds of every executed step: the
    verdict of the step's System-2 action at s_t (need_fatal of reflection / routing), the stop gold (finish_ok /
    world_complete) and the memory_write gold of the offered text. The env is deterministic given (seed, actions),
    including the heartbeat RNG, so this is the reference implementation of the online labels."""
    cfg = (traj.get("meta") or {}).get("config") or {}
    max_steps = int(cfg.get("max_steps", 30))
    final = cfg.get("final_tool", "finish")
    env.reset()
    tools = {t["function"]["name"] for t in env.tools()}
    out = []
    for st in traj.get("steps") or []:
        t = int(st["step"])
        B = max_steps - t
        tool, args = st.get("tool"), st.get("args") or {}
        if tool is None:
            break
        etool = getattr(env, "FINAL_TOOL", "finish") if tool == final else tool
        v = env.oracle_verdict(etool, args, B)
        rec: dict[str, Any] = {"step": t, "tool": tool, "verdict": v.get("verdict"), "d_before": v.get("d_before"),
                               "need_fatal": None if v.get("verdict") is None else v["verdict"] in FAILING_VERDICTS,
                               "fatal_kind": v.get("fatal_kind"), "world_complete": env.oracle_world_complete()}
        if tool == final:
            rec["finish_ok"] = env.oracle_finish_ok(str(args.get("answer", "")))
        rejected = str(st.get("observation", "")).startswith("(finish rejected")
        if tool == final and not rejected:
            out.append(rec)
            break
        if not rejected and tool in tools:
            res = env.step(tool, args)
            rec["obs_match"] = _obs_prefix_match(res.observation, st.get("observation", ""))
            mw = [d for d in st.get("decisions") or [] if d.get("dp") == "memory_write"]
            if mw and callable(getattr(env, "oracle_mem_gold", None)):
                text = str(mw[0].get("state", "")).split("CANDIDATE MEMORY:\n", 1)[-1]
                rec["mem_need"] = env.oracle_mem_gold(text)["mem_need"]
            out.append(rec)
            if res.done:
                break
        else:
            out.append(rec)
    return out


def _obs_prefix_match(a: str, logged: str) -> bool:
    k = min(len(a), len(logged), 300)
    return a[:k] == logged[:k]


def audit(trajs: Sequence[Mapping[str, Any]], factories: Mapping[str, Callable[[], Any]]) -> dict[str, Any]:
    """Compare the online golds with :func:`replay_golds`; returns the differing labels and a determinism check."""
    diffs: list[dict[str, Any]] = []
    n = n_obs_bad = 0
    for t in trajs:
        fac = factories.get(str(t.get("task_id")))
        if fac is None:
            continue
        rep = {r["step"]: r for r in replay_golds(t, fac())}
        for st in t.get("steps") or []:
            r = rep.get(st["step"])
            if r is None:
                continue
            if r.get("obs_match") is False:
                n_obs_bad += 1
            for d in st.get("decisions") or []:
                if d.get("gold_status") != "ok":
                    continue
                want = None
                if d["dp"] in ("reflection", "routing") and (d.get("gold_aux") or {}).get("verdict_src") == "executed":
                    want = r["need_fatal"]
                elif d["dp"] == "stop" and d.get("mode") in ("shadow", "guard", "trigger"):
                    want = r.get("finish_ok") if (d.get("gold_aux") or {}).get("kind") else r["world_complete"]
                elif d["dp"] == "memory_write" and "mem_need" in r:
                    want = r["mem_need"]
                else:
                    continue
                n += 1
                if want != d.get("gold"):
                    diffs.append({"task_id": t.get("task_id"), "step": st["step"], "dp": d["dp"], "mode": d["mode"],
                                  "logged": d.get("gold"), "replayed": want})
    return {"n_checked": n, "n_diff": len(diffs), "diffs": diffs, "n_observation_mismatch": n_obs_bad}


def _finish_executed(st: Mapping[str, Any], final: str) -> bool:
    return st.get("tool") == final and not str(st.get("observation", "")).startswith("(finish rejected")


def implication_audit(trajs: Sequence[Mapping[str, Any]], final: str = "finish") -> dict[str, Any]:
    """The A6 / P0-8 implication audit, on the logs alone (no replay, no env): the cheapest external check of the
    labels. Over every trajectory that did not error:

    * an executed System-2 action whose verdict is failing (``fatal`` / ``premature`` / ``wrong_answer``;
      reflection / routing records with ``verdict_src == "executed"``, tool-guard records whose accepted proposal was
      executed) implies the episode failed (``exec_fail``);
    * an executed finish (the step's tool is ``finish`` and it was not rejected by a guard) with the stop gold
      ``finish_ok`` (the last stop record of that step with a ``kind``) implies success, and ``finish_ok = False``
      implies failure (``finish_ok`` / ``finish_not_ok``).
    Returns the counts checked and violated per implication (``n_violations`` must be 0), the violations themselves,
    and the fatal-kind share of the need_fatal positives (:func:`fatal_kind_share`)."""
    counts = {k: {"checked": 0, "violations": 0} for k in ("exec_fail", "finish_ok", "finish_not_ok")}
    viol: list[dict[str, Any]] = []
    n_traj = 0
    for t in trajs:
        if t.get("error"):
            continue
        n_traj += 1
        success = bool((t.get("result") or {}).get("success"))
        for st in t.get("steps") or []:
            decs = st.get("decisions") or []
            for d in decs:
                if d.get("gold_status") != "ok":
                    continue
                aux = d.get("gold_aux") or {}
                executed = False
                if d.get("dp") in ("reflection", "routing") and aux.get("verdict_src") == "executed":
                    executed = True
                elif d.get("dp") == "tool_select" and d.get("mode") == "guard" and d.get("final_value") is not None \
                        and st.get("tool") == d.get("final_value"):
                    executed = True
                if executed and aux.get("verdict") in FAILING_VERDICTS:
                    counts["exec_fail"]["checked"] += 1
                    if success:
                        counts["exec_fail"]["violations"] += 1
                        viol.append({"task_id": t.get("task_id"), "step": st.get("step"), "dp": d.get("dp"),
                                     "mode": d.get("mode"), "verdict": aux.get("verdict"), "implication": "exec_fail"})
            if _finish_executed(st, final):
                stops = [d for d in decs if d.get("dp") == "stop" and d.get("gold_status") == "ok"
                         and (d.get("gold_aux") or {}).get("kind")]
                if stops:
                    g = bool(stops[-1].get("gold"))
                    key = "finish_ok" if g else "finish_not_ok"
                    counts[key]["checked"] += 1
                    if g != success:
                        counts[key]["violations"] += 1
                        viol.append({"task_id": t.get("task_id"), "step": st.get("step"), "dp": "stop",
                                     "mode": stops[-1].get("mode"), "finish_ok": g, "success": success,
                                     "implication": key})
    return {"n_trajectories": n_traj, "implications": counts,
            "n_checked": sum(c["checked"] for c in counts.values()),
            "n_violations": sum(c["violations"] for c in counts.values()), "violations": viol,
            "fatal_kind": fatal_kind_share(trajs)}


def fatal_kind_share(trajs: Sequence[Mapping[str, Any]], dps: Sequence[str] = ("reflection", "routing")) -> dict[str, Any]:
    """Among the labelled need_fatal positives of the gate records at ``dps`` (shadow and executed labels alike):
    how many are ``terminal`` (the step itself ended the episode unsuccessfully: proven), ``budget`` (D(s') = inf
    only for the canonical plan within the budget: oracle-relative; A6 reports this share and a sensitivity row
    without them), ``finish`` (premature / wrong_answer finishes: evaluate-based, proven) and ``unknown`` (logs that
    predate the label)."""
    c: Counter[str] = Counter()
    for t in trajs:
        for st in t.get("steps") or []:
            for d in st.get("decisions") or []:
                if d.get("dp") not in dps or d.get("gold_status") != "ok" or d.get("gold") is not True:
                    continue
                aux = d.get("gold_aux") or {}
                v = aux.get("verdict")
                if v in ("premature", "wrong_answer"):
                    c["finish"] += 1
                elif v == "fatal":
                    c[aux.get("fatal_kind") or "unknown"] += 1
                else:
                    c["other"] += 1
    n = sum(c.values())
    return {"n_positives": n, "terminal": c.get("terminal", 0), "budget": c.get("budget", 0),
            "finish": c.get("finish", 0), "unknown": c.get("unknown", 0),
            "budget_share": c.get("budget", 0) / n if n else NAN}


def exclude_budget_fatal(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The A6 sensitivity population: the gate records without the need_fatal positives whose ``fatal_kind`` is
    ``budget`` (negatives and proven positives stay)."""
    return [r for r in records
            if not (r.get("gold") is True and (r.get("gold_aux") or {}).get("fatal_kind") == "budget")]


def retest_agreement(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]],
                     factories: Mapping[str, Callable[[], Any]]) -> dict[str, Any]:
    """Test-retest agreement of need_fatal between two identical replicate arms (e.g. ``llm_only_s2lite`` and
    ``small_only``), by replay, at the steps where both arms reached the same state (identical action prefix)."""
    bt = {str(t.get("task_id")): t for t in b}
    agree = n = 0
    for ta in a:
        tb = bt.get(str(ta.get("task_id")))
        fac = factories.get(str(ta.get("task_id")))
        if tb is None or fac is None:
            continue
        ra, rb = replay_golds(ta, fac()), replay_golds(tb, fac())
        for sa, sb, xa, xb in zip(ta.get("steps") or [], tb.get("steps") or [], ra, rb):
            if xa["need_fatal"] is not None and xb["need_fatal"] is not None:
                n += 1
                agree += int(xa["need_fatal"] == xb["need_fatal"])
            if (sa.get("tool"), json.dumps(sa.get("args"), sort_keys=True)) != \
                    (sb.get("tool"), json.dumps(sb.get("args"), sort_keys=True)):
                break
    return {"n": n, "agreement": agree / n if n else NAN}


# ====================================================================== H2 cells, VoI readout, ceiling (§9)
def _per_task(trajs: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], float]) -> dict[str, float]:
    from .scorecard import task_key

    by: dict[str, list[float]] = defaultdict(list)
    for t in trajs:
        by[task_key(t)].append(float(key(t)))
    return {k: float(np.mean(v)) for k, v in by.items()}


def _succ(t: Mapping[str, Any]) -> float:
    return float(bool((t.get("result") or {}).get("success")))


def _cost(t: Mapping[str, Any]) -> float:
    return float(((t.get("usage") or {}).get("total") or {}).get("cost_usd", 0.0) or 0.0)


def _s1_cost(t: Mapping[str, Any]) -> float:
    return float(((t.get("usage") or {}).get("s1") or {}).get("cost_usd", 0.0) or 0.0)


def delta_u(live: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]], Ws: Sequence[float],
            n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Paired (by task) Delta-U = U(live) - U(control), U = W success - cost, net and gross of System-1's own price
    (gross = Delta-U + Delta S1 cost: kappa is a property of the deployment, not of the gate's resolution)."""
    sa, sb = _per_task(live, _succ), _per_task(control, _succ)
    ca, cb = _per_task(live, _cost), _per_task(control, _cost)
    ka, kb = _per_task(live, _s1_cost), _per_task(control, _s1_cost)
    keys = sorted(set(sa) & set(sb))
    out: dict[str, Any] = {"n_paired": len(keys)}
    for w in Ws:
        net_a = {k: w * sa[k] - ca[k] for k in keys}
        net_b = {k: w * sb[k] - cb[k] for k in keys}
        g_a = {k: net_a[k] + ka[k] for k in keys}
        g_b = {k: net_b[k] + kb[k] for k in keys}
        n = paired_bootstrap_diff(net_b, net_a, n=n_boot, seed=seed)
        g = paired_bootstrap_diff(g_b, g_a, n=n_boot, seed=seed)
        out[f"{w:g}"] = {"net": n["diff"], "net_ci": [n["ci_lo"], n["ci_hi"]], "gross": g["diff"],
                         "gross_ci": [g["ci_lo"], g["ci_hi"]]}
    return out


def _fire_rate(trajs: Sequence[Mapping[str, Any]], gate: str) -> tuple[float, float]:
    """(realised firing rate per opportunity, opportunities per episode) of a live gate arm. Each opportunity counts
    once (``metrics.first_opportunities``: a stop-guard resubmission of a rejected answer is a retry, not a new
    opportunity)."""
    rs, _ = first_opportunities(gate_records(trajs, gate, modes=("trigger", "guard")))
    if not rs or not trajs:
        return NAN, NAN
    fires = [(not r["s1_value"]) if gate == "stop_guard" else bool(r["s1_value"]) for r in rs]
    return float(np.mean(fires)), len(rs) / len(trajs)


def _fired_cost(trajs: Sequence[Mapping[str, Any]], gate: str) -> float:
    """c_f, from the ledger (§9.6): the mean reflection call (reflection); the mean large-model minus small-model
    step call (routing); one retry step = the mean System-2 tool call (stop guard); 0 for memory (input tokens).
    Acting calls are used when the arms made them; a measurement arm (whose own reflection / large calls exist only
    as counterfactual branches) falls back to its CF calls of the same prompt: the reflection branch's reflection call,
    the large-model branch as the large step -- so a Flash-Lite measurement arm's calls are never all "large"."""
    def calls(pred: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]) -> list[Mapping[str, Any]]:
        return [c for t in trajs for c in t.get("calls") or [] if c.get("component") == "s2" and pred(t, c)]

    def acting(c: Mapping[str, Any]) -> bool:
        return not (c.get("meta") or {}).get("shadow")

    def cf(c: Mapping[str, Any], tag: str) -> bool:
        m = c.get("meta") or {}
        return m.get("measure") == "cf" and str(m.get("cf_of", "")).endswith(f":{tag}")

    def mean(cs: Sequence[Mapping[str, Any]]) -> float:
        return float(np.mean([float(c.get("cost_usd") or 0.0) for c in cs])) if cs else 0.0

    if gate == "reflection":
        live = calls(lambda t, c: acting(c) and c.get("decision_point") == "reflection")
        return mean(live or calls(lambda t, c: cf(c, "reflection") and c.get("decision_point") == "reflection"))
    if gate == "routing":
        tool = calls(lambda t, c: acting(c) and c.get("decision_point") == "tool_select")
        cf_large = calls(lambda t, c: cf(c, "large"))
        if cf_large:                      # measurement arm: the large step exists only as the CF branch
            big = {c.get("model") for c in cf_large}
            return mean(cf_large) - mean([c for c in tool if c.get("model") not in big])
        big = {t.get("llm") for t in trajs}
        return mean([c for c in tool if c.get("model") in big]) - mean([c for c in tool if c.get("model") not in big])
    if gate == "stop_guard":
        return mean(calls(lambda t, c: acting(c) and c.get("decision_point") == "tool_select"))
    return 0.0


def h2_replay_records(rows: Iterable[Mapping[str, Any]], gate: str, arm: str | None = None) -> list[dict[str, Any]]:
    """The H2 population of a backend replay: rows of ``steps[]`` shadow records only (``source == "steps"``, mode
    ``shadow``), labelled (``gold_status == "ok"``), of ``gate`` (and of the replayed ``arm`` when rows carry one) --
    exactly the records :func:`gate_records` ``(measure, gate, modes=("shadow",))`` gives the measurement arm's own
    backend. Probe-list rows (wording ablation, augmented stop-trigger queries, guard probes) never enter H2 (§3.4,
    §6.4; round-3 review: they doubled the reflection cell and filled the stop-guard cell with phantoms)."""
    return [dict(r) for r in rows if r.get("source") == "steps" and r.get("mode") == "shadow"
            and r.get("gold_status") == "ok" and _gate_of_row(r) == gate
            and (arm is None or r.get("arm") in (None, arm))]


def gate_inputs(live: Sequence[Mapping[str, Any]], measure: Sequence[Mapping[str, Any]], gate: str,
                replays: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
                measure_arm: str | None = None) -> tuple[list[dict[str, Any]] | None, str, bool]:
    """(records, source note, matched): the measurement arm's labelled shadow records of ``gate`` scored by the LIVE
    arm's System-1 -- the measurement arm's own log when it ran the same backend spec (``meta.s1.id``), else the
    replay of that spec on the measurement arm (``replays``: spec id -> replay rows). Never a fallback to another
    backend's answers: without either, (None, note, False) and the cell is not estimable."""
    lid, mid = arm_backend_id(live), arm_backend_id(measure)
    label = arm_backend_label(live) if live else "?"
    if lid is None:
        return None, f"live backend spec unknown ({label}: no meta.s1 logged)", False
    if lid == mid:
        return ([r for r in gate_records(measure, gate, modes=("shadow",)) if r.get("gold_status") == "ok"],
                "measurement arm log", True)
    rows = (replays or {}).get(lid)
    if rows:
        return h2_replay_records(rows, gate, measure_arm), f"replay ({label})", True
    return None, f"replay missing for {label} (spec {lid})", False


def _efficacy(measure: Sequence[Mapping[str, Any]], gate: str) -> dict[str, Any]:
    """e, h of the gate's intervention from the measurement arm's CF branches (a property of the intervention and the
    System-2 family, not of the System-1 backend)."""
    if gate not in BRANCH_OF:
        return {}
    recs = [r for r in gate_records(measure, gate, modes=("shadow",)) if r.get("gold_status") == "ok"]
    return cf_efficacy(recs, BRANCH_OF[gate])


def voi_readout(live: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]],
                measure: Sequence[Mapping[str, Any]], Ws: Sequence[float] = (1.0,), gate: str | None = None,
                n_boot: int = 2000, seed: int = 0, replays: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
                measure_arm: str | None = None) -> dict[str, Any]:
    """Predicted (§9.1) against observed Delta-U of a live gate arm vs. its matched-rate control: pi, AUROC, J at the
    live tau of the LIVE arm's System-1 on the measurement arm's shadow golds (:func:`gate_inputs`: its own log when
    the specs match, else its replay -- otherwise no stats and no prediction, a note says why); e, h from the
    measurement arm's counterfactual branches (net of placebo); n-bar and the realised rates from the live / control
    arms; kappa from the live arm's System-1 cost; the realised-rate correction."""
    gate = gate or arm_gate(live)
    if gate is None:
        return {"note": "the live arm has no trigger / guard gate"}
    tau = arm_tau(live, gate) or 0.5
    recs, src, matched = gate_inputs(live, measure, gate, replays, measure_arm)
    eff = _efficacy(measure, gate)
    e, h = eff.get("e", NAN), eff.get("h", NAN)
    r_live, n_bar = _fire_rate(live, gate)
    r_ctrl, _ = _fire_rate(control, gate)
    opp = sum(len(first_opportunities(gate_records([t], gate, modes=("trigger", "guard")))[0]) for t in live)
    kappa = sum(_s1_cost(t) for t in live) / opp if opp else 0.0
    c_f = _fired_cost(live, gate)
    obs = delta_u(live, control, Ws, n_boot, seed)
    gs = gate_stats(recs, tau, guard=gate == "stop_guard") if matched and recs is not None else None
    out: dict[str, Any] = {"gate": gate, "tau": tau, "stats": gs, "stats_source": src, "stats_matched": matched,
                           "efficacy": eff, "n_bar": n_bar, "kappa": kappa, "c_f": c_f, "rate_live": r_live,
                           "rate_control": r_ctrl, "estimable": bool(gs and gs.get("estimable")), "observed": obs,
                           "predicted": {}}
    if gs is None:
        out["note"] = f"no prediction: {src}"
    elif eff and not eff.get("net", False):
        out["note"] = "no prediction: e / h are not net of the placebo (placebo verdicts missing)"
    if gs and gs.get("n") and not any(math.isnan(x) for x in (e, h, n_bar)):
        dr = (r_ctrl - r_live) if not (math.isnan(r_ctrl) or math.isnan(r_live)) else 0.0
        for w in Ws:
            out["predicted"][f"{w:g}"] = voi_identity(n_bar, w, e, h, gs["pi"], gs["J"], kappa, dr, c_f)
    # A6 / P0-3: for guards the live alternative to a rejection is EXECUTING the original proposal, so e / h are
    # also measured against it (the gross branch rates e_raw / h_raw); for pre-action gates the placebo-netted
    # values stay primary (the default resample is the live alternative) and the raw row is secondary
    e_raw, h_raw = eff.get("e_raw", NAN), eff.get("h_raw", NAN)
    out["efficacy_primary"] = "raw" if gate == "stop_guard" else "net"
    out["predicted_raw"] = {}
    if gs and gs.get("n") and not any(math.isnan(x) for x in (e_raw, h_raw, n_bar)):
        dr = (r_ctrl - r_live) if not (math.isnan(r_ctrl) or math.isnan(r_live)) else 0.0
        for w in Ws:
            out["predicted_raw"][f"{w:g}"] = voi_identity(n_bar, w, e_raw, h_raw, gs["pi"], gs["J"], kappa, dr, c_f)
    return out


def h2_cells(runs: Mapping[str, Sequence[Mapping[str, Any]]], measure_arms: Sequence[str],
             pairs: Sequence[tuple[str, str]], Ws: Sequence[float] = (0.1, 1.0, 10.0),
             replays: Mapping[str, Sequence[Mapping[str, Any]]] | None = None, n_boot: int = 2000,
             seed: int = 0, tost_margin: float = 0.02, task_data: bool = True) -> list[dict[str, Any]]:
    """The H2 input table (§9.2). One cell per live arm of ``pairs`` (live, matched-rate control): its gate, System-1
    backend (spec id) and System-2 family; the measurement arm of the gate (:func:`measure_arm_for`: routing takes the
    routers' small-model arm); AUROC / J / accuracy of the LIVE backend on that arm's shadow golds -- its own log when
    the specs match, else a replay (``replays``: spec id -> replay rows), never another backend's answers
    (:func:`gate_inputs`); estimability (>= 30 positive and >= 30 negative primary golds; otherwise "no opportunity",
    predicted gross Delta-U ~ 0); the paired gross / net Delta-U at each W; and the pre-registered TOST (±2 pt of
    success) of the System-1 gate against its matched random gate.

    A6 / P0-3 additions per cell: ``kappa`` (the live arm's System-1 price per opportunity), ``efficacy`` with
    ``e_raw`` / ``h_raw`` (for guards the primary e / h: the live alternative to a rejection is executing the
    proposal), ``stats_no_budget`` (the gate stats with the ``budget``-kind need_fatal positives excluded: the P0-8
    sensitivity row) and, with ``task_data``, the per-task inputs of the task-level bootstraps of
    :func:`h2_regression_j` and :func:`h2_theory_check` (``task_data``: ``delta_u`` -- per-task gross / net utility
    differences at each W; ``records`` -- the gate's (task, p_yes, gold) triples; ``cf`` -- the measurement arm's
    counterfactual rows (task, v0, branch verdict, placebo verdict, weight) behind e / h; ``live`` -- per task of
    the live arm [episodes, opportunities, System-1 cost] behind n-bar and kappa); the CLI drops them from the
    printed cells."""
    from .scorecard import arm_s2_model

    cells = []
    for live, ctrl in pairs:
        if live not in runs or ctrl not in runs:
            continue
        gate = arm_gate(runs[live])
        model = arm_s2_model(runs[live])
        m_arm = measure_arm_for(runs, measure_arms, live, gate) if gate else None
        backend = (runs[live][0].get("backend") if runs[live] else None)
        cell: dict[str, Any] = {"gate": gate, "live": live, "control": ctrl, "s2_family": model, "backend": backend,
                                "backend_id": arm_backend_id(runs[live]), "measure_arm": m_arm}
        tau = arm_tau(runs[live], gate) if gate else None
        cell["stats_matched"] = False
        if gate and m_arm:
            recs, src, matched = gate_inputs(runs[live], runs[m_arm], gate, replays, m_arm)
            cell["stats_source"], cell["stats_matched"] = src, matched
            cell["stats"] = gate_stats(recs, tau or 0.5, guard=gate == "stop_guard") if matched and recs is not None \
                else None
            cell["estimable"] = bool(cell["stats"] and cell["stats"].get("estimable"))
            if matched and recs is not None:
                cell["stats_no_budget"] = gate_stats(exclude_budget_fatal(recs), tau or 0.5, guard=gate == "stop_guard")
                if task_data:
                    cell.setdefault("task_data", {})["records"] = [
                        (str(r.get("task_id")), p_yes_of(r), bool(r["gold"])) for r in recs
                        if isinstance(r.get("gold"), bool) and p_yes_of(r) is not None]
            if gate in BRANCH_OF:
                cell["efficacy"] = _efficacy(runs[m_arm], gate)
        else:
            cell["stats_source"] = "no measurement arm with shadow records of this gate" if gate else "no live gate"
            cell["estimable"] = False
        cell["no_opportunity"] = not cell["estimable"]
        cell["delta_u"] = delta_u(runs[live], runs[ctrl], Ws, n_boot, seed)
        cell["tost"] = tost_paired(_per_task(runs[ctrl], _succ), _per_task(runs[live], _succ), tost_margin,
                                   n_boot=n_boot, seed=seed)
        if gate:
            cell["rate_live"], cell["n_bar"] = _fire_rate(runs[live], gate)
            opp = sum(len(first_opportunities(gate_records([t], gate, modes=("trigger", "guard")))[0])
                      for t in runs[live])
            cell["kappa"] = sum(_s1_cost(t) for t in runs[live]) / opp if opp else 0.0
            cell["efficacy_primary"] = "raw" if gate == "stop_guard" else "net"
            if task_data:
                cell.setdefault("task_data", {})["live"] = _live_rows(runs[live], gate)
                if gate in BRANCH_OF and m_arm:
                    cell["task_data"]["cf"] = _cf_rows(runs[m_arm], gate)
        if task_data:
            cell.setdefault("task_data", {})["delta_u"] = _task_delta_u(runs[live], runs[ctrl], Ws)
        cells.append(cell)
    return cells


def _task_delta_u(live: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]],
                  Ws: Sequence[float]) -> dict[str, dict[str, list[float]]]:
    """Per-task [gross, net] utility differences live - control at each W (the bootstrap unit of
    :func:`h2_regression_j`)."""
    sa, sb = _per_task(live, _succ), _per_task(control, _succ)
    ca, cb = _per_task(live, _cost), _per_task(control, _cost)
    ka, kb = _per_task(live, _s1_cost), _per_task(control, _s1_cost)
    keys = sorted(set(sa) & set(sb))
    out: dict[str, dict[str, list[float]]] = {}
    for w in Ws:
        out[f"{w:g}"] = {k: [w * (sa[k] - sb[k]) - (ca[k] - cb[k]) + (ka[k] - kb[k]),
                             w * (sa[k] - sb[k]) - (ca[k] - cb[k])] for k in keys}
    return out


def _tid(x: Any) -> str:
    """A record's task id in the :func:`scorecard.task_key` namespace (trials of one task share it): the resampling
    unit of the task bootstraps."""
    from .scorecard import TRIAL_RE

    return TRIAL_RE.sub("", str(x))


def _cf_rows(measure: Sequence[Mapping[str, Any]], gate: str) -> list[tuple[str, str, str, str | None, float]]:
    """(task id, v0, branch verdict, placebo verdict or None, cf weight) of the measurement arm's labelled shadow
    records of ``gate`` that carry its counterfactual branch -- exactly the rows :func:`metrics.cf_efficacy` reduces
    in :func:`_efficacy`, keyed by task so that :func:`h2_theory_check` can recompute e / h on a task resample."""
    if gate not in BRANCH_OF:
        return []
    branch, rows = BRANCH_OF[gate], []
    for r in gate_records(measure, gate, modes=("shadow",)):
        a = r.get("gold_aux") or {}
        if r.get("gold_status") == "ok" and a.get("cf_status") == "ok" and a.get(branch) is not None \
                and a.get("verdict") is not None:
            pl = a.get("verdict_placebo")
            rows.append((_tid(r.get("task_id")), str(a["verdict"]), str(a[branch]), None if pl is None else str(pl),
                         float(a.get("cf_weight", 1.0) or 1.0)))
    return rows


def _live_rows(trajs: Sequence[Mapping[str, Any]], gate: str) -> dict[str, list[float]]:
    """Per task of a live gate arm: [episodes, gate opportunities (each counted once), System-1 cost] -- the
    task-level inputs of n-bar (opportunities per episode) and kappa (System-1 price per opportunity) for the joint
    task bootstrap of :func:`h2_theory_check`."""
    from .scorecard import task_key

    out: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for t in trajs:
        o = out[task_key(t)]
        o[0] += 1.0
        o[1] += float(len(first_opportunities(gate_records([t], gate, modes=("trigger", "guard")))[0]))
        o[2] += _s1_cost(t)
    return dict(out)


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, float(np.sum((y - X @ beta) ** 2))


def h2_regression_auroc(cells: Sequence[Mapping[str, Any]], W: float = 1.0) -> dict[str, Any]:
    """The pre-A6 H2 regression, now SECONDARY (A6 / P0-3: AUROC is the tau-free upper bound on attainable J,
    max_tau J <= 2 AUROC - 1, so its slope is exploratory): Delta-U (gross) on AUROC, and on AUROC + accuracy, with a
    likelihood-ratio test for adding accuracy (Gaussian errors: LR = n ln(RSS_A / RSS_B) ~ chi2(1)); and the theory
    check -- Delta-U_gross / (W n-bar pi (1 - pi) (e + h)) against J(tau_op) should have slope 1 and intercept 0
    (§9.1). Cells without the inputs are skipped (and counted). The primary test is :func:`h2_regression_j`."""
    w = f"{W:g}"
    rows = []
    for c in cells:
        st, du = c.get("stats") or {}, ((c.get("delta_u") or {}).get(w) or {})
        # only cells whose regressors are the live backend's own answers (its log or its replay; ``stats_matched``)
        if c.get("estimable") and c.get("stats_matched", True) and du.get("gross") is not None \
                and not math.isnan(st.get("auroc", NAN)):
            rows.append((du["gross"], st["auroc"], st.get("accuracy", NAN), st.get("J", NAN), st.get("pi", NAN),
                         (c.get("efficacy") or {}).get("e", NAN) + (c.get("efficacy") or {}).get("h", NAN),
                         c.get("n_bar", NAN)))
    out: dict[str, Any] = {"W": W, "n_cells": len(cells), "n_estimable": len(rows)}
    if len(rows) < 3:
        out["note"] = "fewer than 3 estimable cells: no regression"
        return out
    a = np.array(rows, float)
    y = a[:, 0]
    XA = np.column_stack([np.ones(len(a)), a[:, 1]])
    XB = np.column_stack([XA, a[:, 2]])
    bA, rA = _ols(XA, y)
    bB, rB = _ols(XB, y)
    lr = len(a) * math.log(max(rA, 1e-300) / max(rB, 1e-300))
    out.update({"auroc_only": {"intercept": float(bA[0]), "slope_auroc": float(bA[1]), "rss": rA},
                "auroc_accuracy": {"intercept": float(bB[0]), "slope_auroc": float(bB[1]),
                                   "slope_accuracy": float(bB[2]), "rss": rB},
                "lr_accuracy": lr, "lr_p": math.erfc(math.sqrt(max(lr, 0.0) / 2))})
    den = W * a[:, 6] * a[:, 4] * (1 - a[:, 4]) * a[:, 5]
    ok = np.isfinite(den) & (np.abs(den) > 1e-12) & np.isfinite(a[:, 3])
    if ok.sum() >= 3:
        bT, _ = _ols(np.column_stack([np.ones(int(ok.sum())), a[ok, 3]]), y[ok] / den[ok])
        out["theory_check"] = {"n": int(ok.sum()), "intercept": float(bT[0]), "slope_J": float(bT[1]),
                               "expected": {"intercept": 0.0, "slope_J": 1.0}}
    return out


h2_regression = h2_regression_auroc      # backward-compatible name (secondary since A6)


def _cell_rows_j(cells: Sequence[Mapping[str, Any]], w: str) -> list[tuple[int, float, float, float]]:
    """(cell index, Delta-U gross, J at the operating tau, accuracy) of the cells that enter the J regression."""
    rows = []
    for i, c in enumerate(cells):
        st, du = c.get("stats") or {}, ((c.get("delta_u") or {}).get(w) or {})
        if c.get("estimable") and c.get("stats_matched", True) and du.get("gross") is not None \
                and not math.isnan(st.get("J", NAN)):
            rows.append((i, float(du["gross"]), float(st["J"]), float(st.get("accuracy", NAN))))
    return rows


def _fit_models(y: np.ndarray, J: np.ndarray, acc: np.ndarray) -> dict[str, Any]:
    n = len(y)
    one = np.ones(n)
    bJ, rJ = _ols(np.column_stack([one, J]), y)
    out: dict[str, Any] = {"j_only": {"intercept": float(bJ[0]), "slope_J": float(bJ[1]), "rss": rJ}}
    if np.all(np.isfinite(acc)):
        bA, rA = _ols(np.column_stack([one, acc]), y)
        bB, rB = _ols(np.column_stack([one, J, acc]), y)
        lr_acc = n * math.log(max(rJ, 1e-300) / max(rB, 1e-300))      # adding accuracy to J
        lr_j = n * math.log(max(rA, 1e-300) / max(rB, 1e-300))        # adding J to accuracy
        out.update({"accuracy_only": {"intercept": float(bA[0]), "slope_accuracy": float(bA[1]), "rss": rA},
                    "j_accuracy": {"intercept": float(bB[0]), "slope_J": float(bB[1]),
                                   "slope_accuracy": float(bB[2]), "rss": rB},
                    "lr_add_accuracy": lr_acc, "lr_add_accuracy_p": math.erfc(math.sqrt(max(lr_acc, 0.0) / 2)),
                    "lr_add_J": lr_j, "lr_add_J_p": math.erfc(math.sqrt(max(lr_j, 0.0) / 2)),
                    # the two one-regressor models have the same size: the smaller RSS is the better likelihood
                    "accuracy_explains_better": bool(rA < rJ)})
    return out


def h2_regression_j(cells: Sequence[Mapping[str, Any]], W: float = 1.0, n_boot: int = 2000, seed: int = 0,
                    alpha: float = 0.05) -> dict[str, Any]:
    """The A6 / P0-3 primary H2 test: each estimable cell's Delta-U (live gate - matched-rate random gate, gross of
    System-1's price) regressed on **J at the operating tau** (Prop. 2 is an operating-point statement) and on
    accuracy. Readouts: the J slope with a task-level bootstrap CI (``slope_J_ci``: the cells' shared tasks are
    resampled with replacement, each cell's Delta-U and J(tau) recomputed on the resample from ``task_data`` --
    cells losing both label classes in a replicate drop out of it; without task data: no CI), the model comparison
    (J only, accuracy only, J + accuracy; LR tests for adding accuracy to J and J to accuracy; Gaussian errors), and
    ``falsified`` = J slope <= 0, or accuracy explaining Delta-U better (smaller RSS of the one-regressor model AND a
    significant gain from adding accuracy to J without a significant gain from adding J to accuracy). The theory
    check (slope 1 / intercept 0 of the normalised gain on J) is in :func:`h2_regression_auroc`; the Eq. 2 point
    prediction per cell in :func:`predicted_gain`."""
    w = f"{W:g}"
    rows = _cell_rows_j(cells, w)
    out: dict[str, Any] = {"W": W, "n_cells": len(cells), "n_estimable": len(rows), "alpha": alpha}
    if len(rows) < 3:
        out["note"] = "fewer than 3 estimable cells: no regression"
        return out
    y = np.array([r[1] for r in rows]); J = np.array([r[2] for r in rows]); acc = np.array([r[3] for r in rows])
    out.update(_fit_models(y, J, acc))
    fit = out
    out["slope_J"] = fit["j_only"]["slope_J"]
    better = fit.get("accuracy_explains_better", False) and fit.get("lr_add_accuracy_p", 1.0) < alpha \
        and fit.get("lr_add_J_p", 0.0) >= alpha
    out["falsified"] = bool(out["slope_J"] <= 0 or better)
    # task-level bootstrap of the J slope
    td = [(cells[i].get("task_data") or {}) for i, *_ in rows]
    ok = all(("delta_u" in t and w in t["delta_u"] and "records" in t) for t in td)
    if not ok:
        out["slope_J_ci"] = [NAN, NAN]
        out["bootstrap_note"] = "no task data in the cells: no task-level CI"
        return out
    guards = [bool((cells[i].get("gate") or "") == "stop_guard") for i, *_ in rows]
    taus = [float((cells[i].get("stats") or {}).get("tau", 0.5)) for i, *_ in rows]
    tasks = sorted({k for t in td for k in t["delta_u"][w]} | {r[0] for t in td for r in t["records"]})
    idx = {k: n for n, k in enumerate(tasks)}
    rng = np.random.default_rng(seed)
    slopes = []
    # per cell: arrays over tasks of (sum dU, count) and the (task index, score, label) triples
    cell_du = [(np.array([idx[k] for k in t["delta_u"][w]]), np.array([v[0] for v in t["delta_u"][w].values()]))
               for t in td]
    cell_rec = [(np.array([idx[r[0]] for r in t["records"]]), np.array([float(r[1]) for r in t["records"]]),
                 np.array([bool(r[2]) for r in t["records"]])) for t in td]
    for _ in range(n_boot):
        wts = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)).astype(float)
        ys, js = [], []
        for (ti, dv), (ri, sc, gl), guard, tau in zip(cell_du, cell_rec, guards, taus):
            wd = wts[ti]
            if wd.sum() <= 0:
                continue
            du = float((wd * dv).sum() / wd.sum())
            wr = wts[ri]
            y_ = ~gl if guard else gl
            fire = (sc < tau) if guard else (sc >= tau)
            wp, wn = wr[y_].sum(), wr[~y_].sum()
            if wp <= 0 or wn <= 0:
                continue
            tpr = float((wr[y_] * fire[y_]).sum() / wp)
            fpr = float((wr[~y_] * fire[~y_]).sum() / wn)
            ys.append(du); js.append(tpr - fpr)
        if len(ys) >= 3 and np.std(js) > 1e-12:
            b, _ = _ols(np.column_stack([np.ones(len(ys)), np.array(js)]), np.array(ys))
            slopes.append(float(b[1]))
    if slopes:
        out["slope_J_ci"] = [float(np.quantile(slopes, alpha / 2)), float(np.quantile(slopes, 1 - alpha / 2))]
        out["n_boot_effective"] = len(slopes)
        out["slope_J_p_le_0"] = float(np.mean(np.array(slopes) <= 0))
    else:
        out["slope_J_ci"] = [NAN, NAN]
    return out


def predicted_gain(cell: Mapping[str, Any], W: float = 1.0) -> dict[str, Any]:
    """The Eq. 2 / Prop. 2 point prediction of one H2 cell next to its observed Delta-U (A6 / P0-3, astra's theory
    check): predicted Delta-U = W n-bar pi (1 - pi) (e + h) J(tau) - kappa n-bar, gross = without the - kappa n-bar
    term; from the cell's gate stats (pi, J at the live tau), efficacy (e, h net of placebo, and e_raw / h_raw
    measured against executing the original proposal: primary for guards, where that is the live alternative to a
    rejection), n-bar and kappa; ``observed`` = the paired task-bootstrap Delta-U (gross / net with CIs)."""
    w = f"{W:g}"
    st, eff, du = cell.get("stats") or {}, cell.get("efficacy") or {}, (cell.get("delta_u") or {}).get(w) or {}
    n_bar, kappa = float(cell.get("n_bar", NAN)), float(cell.get("kappa", 0.0) or 0.0)
    pi, J = float(st.get("pi", NAN)), float(st.get("J", NAN))
    primary = cell.get("efficacy_primary") or ("raw" if cell.get("gate") == "stop_guard" else "net")
    out: dict[str, Any] = {"gate": cell.get("gate"), "live": cell.get("live"), "W": W, "n_bar": n_bar, "pi": pi,
                           "J": J, "kappa": kappa, "estimable": bool(cell.get("estimable")),
                           "efficacy_primary": primary, "observed": du}
    for tag, (e, h) in (("net", (eff.get("e", NAN), eff.get("h", NAN))),
                        ("raw", (eff.get("e_raw", NAN), eff.get("h_raw", NAN)))):
        gross = W * n_bar * pi * (1 - pi) * (float(e) + float(h)) * J
        out[tag] = {"e": e, "h": h, "pred_gross": gross, "pred_net": gross - kappa * n_bar}
    out["pred_gross"], out["pred_net"] = out[primary]["pred_gross"], out[primary]["pred_net"]
    out["gap_gross"] = (float(du["gross"]) - out["pred_gross"]) if du.get("gross") is not None else NAN
    return out


# ====================================================================== H2 theory check (A6.16 (i))
H2_MIN_CELLS = 4           # A6.16 (i): >= 4 estimable cells (and full column rank) for any H2 regression
_AGG_KEYS = ("du_n", "du_gross", "du_net", "pos", "neg", "fire_pos", "fire_neg", "cf_pos_w", "cf_pos_alt",
             "cf_pos_pl_w", "cf_pos_pl", "cf_neg_w", "cf_neg_alt", "cf_neg_pl_w", "cf_neg_pl", "eps", "opps", "s1")


def _joint_resample(tasks: Sequence[str], n_boot: int, seed: int = 0) -> np.ndarray:
    """The joint task bootstrap of A6.16 (i): an ``n_boot`` x ``len(tasks)`` matrix of task multiplicities -- one
    multiset of task ids per replicate (``tasks`` resampled with replacement, ``numpy.random.default_rng(seed)``)
    that every cell of the check shares, so cells with the same task ids see the same multiset in every replicate
    and the between-cell comparisons stay paired."""
    T = len(tasks)
    out = np.zeros((max(n_boot, 0), T), float)
    if T == 0 or n_boot <= 0:
        return out
    draws = np.random.default_rng(seed).integers(0, T, size=(n_boot, T))
    for b in range(n_boot):
        out[b] = np.bincount(draws[b], minlength=T)
    return out


def _pct_ci(x: np.ndarray, alpha: float = 0.05) -> list[float]:
    """Percentile (1 - alpha) interval over the finite entries of ``x`` (NaN replicates are the non-estimable ones)."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [NAN, NAN]
    return [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]


def _inside(x: float, lo: float, hi: float, rel: float = 1e-9) -> bool:
    """lo <= x <= hi up to floating-point rounding (a degenerate interval [p, p] still contains p)."""
    tol = rel * max(1.0, abs(x), abs(lo), abs(hi))
    return bool(lo - tol <= x <= hi + tol)


def _has_task_data(cell: Mapping[str, Any], w: str) -> bool:
    td = cell.get("task_data") or {}
    return bool(((td.get("delta_u") or {}).get(w)) and td.get("records") and td.get("live")
                and (cell.get("gate") not in BRANCH_OF or "cf" in td))


def _cell_task_ids(cell: Mapping[str, Any], w: str) -> set[str]:
    td = cell.get("task_data") or {}
    ids = {_tid(k) for k in ((td.get("delta_u") or {}).get(w) or {})}
    ids |= {_tid(r[0]) for r in td.get("records") or []}
    ids |= {_tid(r[0]) for r in td.get("cf") or []}
    ids |= {_tid(k) for k in td.get("live") or {}}
    return ids


def _theory_aggregates(cell: Mapping[str, Any], w: str, idx: Mapping[str, int]) -> dict[str, np.ndarray]:
    """Per-task sums, over the shared task index ``idx``, of everything Eq. 2's inputs and the observed Delta-U of one
    cell are ratios of -- so a task resample with multiplicities m evaluates as the products m @ sums: the paired
    gross / net utility differences; the gate's positive / negative golds and their firings at the live tau (guard
    polarity as in :func:`metrics.gate_stats`); the counterfactual strata of :func:`metrics.cf_efficacy` (branch and
    placebo progress-or-failing weights, v0 failing = the e stratum, v0 progress = the h stratum); and the live
    arm's episodes, opportunities and System-1 cost."""
    td = cell.get("task_data") or {}
    A = {k: np.zeros(len(idx)) for k in _AGG_KEYS}
    for k, (g, n) in ((td.get("delta_u") or {}).get(w) or {}).items():
        i = idx[_tid(k)]
        A["du_n"][i] += 1.0
        A["du_gross"][i] += float(g)
        A["du_net"][i] += float(n)
    guard = (cell.get("gate") or "") == "stop_guard"
    tau = float((cell.get("stats") or {}).get("tau", 0.5))
    for t, p, g in td.get("records") or []:
        i = idx[_tid(t)]
        y = (not g) if guard else bool(g)
        fire = (float(p) < tau) if guard else (float(p) >= tau)
        A["pos" if y else "neg"][i] += 1.0
        if fire:
            A["fire_pos" if y else "fire_neg"][i] += 1.0
    for t, v0, alt, pl, wt in td.get("cf") or []:
        i = idx[_tid(t)]
        wt = float(wt)
        if v0 in FAILING_VERDICTS:
            A["cf_pos_w"][i] += wt
            A["cf_pos_alt"][i] += wt * (alt == "progress")
            if pl is not None:
                A["cf_pos_pl_w"][i] += wt
                A["cf_pos_pl"][i] += wt * (pl == "progress")
        elif v0 == "progress":
            A["cf_neg_w"][i] += wt
            A["cf_neg_alt"][i] += wt * (alt in FAILING_VERDICTS)
            if pl is not None:
                A["cf_neg_pl_w"][i] += wt
                A["cf_neg_pl"][i] += wt * (pl in FAILING_VERDICTS)
    for t, (eps, opps, s1) in (td.get("live") or {}).items():
        i = idx[_tid(t)]
        A["eps"][i] += float(eps)
        A["opps"][i] += float(opps)
        A["s1"][i] += float(s1)
    return A


def _theory_replicates(A: Mapping[str, np.ndarray], M: np.ndarray, W: float, primary: str = "net",
                       min_pos_neg: int = MIN_POS_NEG) -> dict[str, np.ndarray]:
    """Eq. 2's inputs (pi, J at the live tau, e, h under the cell's primary efficacy -- ``net`` of the placebo or the
    ``raw`` branch rates for guards -- n-bar, kappa), its predicted gross / net Delta-U and the observed gross / net
    Delta-U of one cell on every row of the multiplicity matrix ``M`` (replicates x tasks; a single row of ones gives
    the point estimates). A replicate is NaN wherever the cell loses estimability: fewer than ``min_pos_neg``
    positive or negative golds (A4), no opportunity or episode, no paired task, or an undefined e / h (an empty
    stratum, or a missing placebo under the net primary)."""
    S = {k: M @ np.asarray(v, float) for k, v in A.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_gross, obs_net = S["du_gross"] / S["du_n"], S["du_net"] / S["du_n"]
        pi = S["pos"] / (S["pos"] + S["neg"])
        J = S["fire_pos"] / S["pos"] - S["fire_neg"] / S["neg"]
        alt_p, alt_h = S["cf_pos_alt"] / S["cf_pos_w"], S["cf_neg_alt"] / S["cf_neg_w"]
        if primary == "raw":
            e, h = alt_p, alt_h
        else:
            e, h = alt_p - S["cf_pos_pl"] / S["cf_pos_pl_w"], alt_h - S["cf_neg_pl"] / S["cf_neg_pl_w"]
        n_bar, kappa = S["opps"] / S["eps"], S["s1"] / S["opps"]
        pred_gross = W * n_bar * pi * (1 - pi) * (e + h) * J
        pred_net = pred_gross - kappa * n_bar
    ok = (S["pos"] >= min_pos_neg) & (S["neg"] >= min_pos_neg) & (S["opps"] > 0) & (S["eps"] > 0) \
        & (S["du_n"] > 0) & np.isfinite(pred_gross) & np.isfinite(obs_gross)
    out = {"pred_gross": pred_gross, "pred_net": pred_net, "obs_gross": obs_gross, "obs_net": obs_net, "pi": pi,
           "J": J, "e": e, "h": h, "n_bar": n_bar, "kappa": kappa}
    for k, v in out.items():
        v = np.array(v, float, copy=True)
        v[~ok] = NAN
        out[k] = v
    out["ok"] = ok
    return out


def h2_theory_check(cells: Sequence[Mapping[str, Any]], W: float = 1.0, n_boot: int = 2000, seed: int = 0,
                    min_cells: int = H2_MIN_CELLS, alpha: float = 0.05,
                    min_pos_neg: int = MIN_POS_NEG) -> dict[str, Any]:
    """The pre-registered H2 inference of A6.16 (i): Eq. 2's predicted Delta-U-hat = W n-bar pi (1 - pi) (e + h)
    J_tau - kappa n-bar (:func:`predicted_gain`; e / h under the cell's ``efficacy_primary``) against the observed
    paired Delta-U of the live gate over its matched-rate control, cell by cell. **Gross** Delta-U (without the
    System-1 query cost kappa n-bar) is what Eq. 2 predicts; **net** Delta-U (with it) is reported next to it.

    Intervals come from one JOINT task bootstrap (:func:`_joint_resample`): in every replicate one multiset of task
    ids -- the union of the estimable cells' task ids, resampled with replacement -- is drawn and shared by every
    cell, and for every cell both the predictors (pi, J_tau, e, h, n-bar, kappa, recomputed from that cell's
    ``task_data`` on the resampled tasks: :func:`_theory_aggregates` / :func:`_theory_replicates`) and the observed
    gross / net Delta-U are recomputed. A cell that loses estimability in a replicate (fewer than ``min_pos_neg``
    positive or negative golds, no opportunity, no paired task, undefined e / h) contributes NaN to it; every CI is
    the percentile interval of the non-NaN replicates and ``n_boot_effective`` reports their count.

    Per estimable cell: ``pred_gross`` / ``pred_net``, ``obs_gross`` / ``obs_net``, the paired differences
    ``diff_gross`` / ``diff_net`` (observed - predicted), each with ``*_ci``, ``inside_predicted_ci`` (A6.2's
    readout), the point inputs and the per-cell replicate count. Across cells: ``mae`` (mean |observed - predicted|
    gross) and ``mean_diff_gross`` with CIs, ``share_inside_predicted_ci``, and -- only with at least ``min_cells``
    estimable cells and a full-column-rank design -- ``slope``: the through-origin slope of observed gross on
    predicted gross with its joint-bootstrap CI (Eq. 2 predicts 1; ``ci_covers_1`` is the consistency flag, the
    wording lives in :func:`h2_theory_wording`). Otherwise ``estimable`` is False with ``reason`` "fewer than
    ``min_cells`` estimable cells" or "rank-deficient design": H2 is then *not estimable* -- never falsified. The
    pooled regression of Delta-U on J and accuracy (:func:`h2_regression_j`, unchanged) is returned under
    ``exploratory``."""
    w = f"{W:g}"
    rows: list[dict[str, Any]] = []
    for i, c in enumerate(cells):
        pg = predicted_gain(c, W)
        prim = pg["efficacy_primary"]
        du = pg.get("observed") or {}
        obs_g = float(du["gross"]) if du.get("gross") is not None else NAN
        obs_n = float(du["net"]) if du.get("net") is not None else NAN
        r: dict[str, Any] = {
            "cell": i, "gate": c.get("gate"), "live": c.get("live"), "control": c.get("control"),
            "s2_family": c.get("s2_family"), "backend": c.get("backend"), "efficacy_primary": prim,
            "inputs": {"pi": pg["pi"], "J": pg["J"], "e": pg[prim]["e"], "h": pg[prim]["h"], "n_bar": pg["n_bar"],
                       "kappa": pg["kappa"]},
            "pred_gross": float(pg["pred_gross"]), "pred_net": float(pg["pred_net"]),
            "obs_gross": obs_g, "obs_net": obs_n,
            "obs_gross_ci_paired": du.get("gross_ci"), "obs_net_ci_paired": du.get("net_ci")}
        r["diff_gross"], r["diff_net"] = obs_g - r["pred_gross"], obs_n - r["pred_net"]
        if not c.get("estimable"):
            reason = f"cell not estimable (fewer than {min_pos_neg} positive or negative golds, or no gate stats)"
        elif not c.get("stats_matched", True):
            reason = "gate stats are not the live backend's own answers"
        elif not math.isfinite(r["pred_gross"]):
            reason = "prediction undefined (pi, J, e, h or n-bar missing)"
        elif not math.isfinite(obs_g):
            reason = "no paired observed Delta-U"
        else:
            reason = None
        r["estimable"], r["reason"] = reason is None, reason
        for k in ("pred_gross", "pred_net", "obs_gross", "obs_net", "diff_gross", "diff_net"):
            r[f"{k}_ci"] = [NAN, NAN]
        r["inside_predicted_ci"] = None
        r["n_boot_effective"] = 0
        rows.append(r)
    est = [r for r in rows if r["estimable"]]
    out: dict[str, Any] = {"W": W, "alpha": alpha, "min_cells": min_cells, "min_pos_neg": min_pos_neg,
                           "n_cells_total": len(cells), "n_cells": len(est), "n_boot": n_boot, "seed": seed,
                           "estimable": False, "reason": None, "cells": rows, "slope": None, "n_boot_effective": 0,
                           "mae": NAN, "mae_ci": [NAN, NAN], "mean_diff_gross": NAN,
                           "mean_diff_gross_ci": [NAN, NAN], "share_inside_predicted_ci": NAN,
                           "bootstrap": "joint task bootstrap (one task multiset per replicate, shared by every cell)"}
    p = np.array([r["pred_gross"] for r in est], float)
    o = np.array([r["obs_gross"] for r in est], float)
    if est:
        out["mae"], out["mean_diff_gross"] = float(np.mean(np.abs(o - p))), float(np.mean(o - p))
    if len(est) < min_cells:
        out["reason"] = f"fewer than {min_cells} estimable cells"
    elif np.linalg.matrix_rank(p[:, None]) < 1:
        out["reason"] = "rank-deficient design"
    else:
        out["estimable"] = True
        out["slope"] = {"value": float(p @ o / (p @ p)), "ci": [NAN, NAN], "expected": 1.0, "ci_covers_1": None,
                        "n_cells": len(est), "n_boot_effective": 0,
                        "model": "observed gross Delta-U = slope x predicted gross Delta-U (through the origin)"}
    # the joint task bootstrap over the estimable cells
    boot_cells = [r for r in est if _has_task_data(cells[r["cell"]], w)]
    if len(boot_cells) < len(est):
        out["bootstrap_note"] = (f"{len(est) - len(boot_cells)} estimable cell(s) without task data: no task-level "
                                 f"CI for them" + (" and none for the slope" if out["slope"] else ""))
    if boot_cells and n_boot > 0:
        tasks = sorted(set().union(*(_cell_task_ids(cells[r["cell"]], w) for r in boot_cells)))
        idx = {t: n for n, t in enumerate(tasks)}
        M = _joint_resample(tasks, n_boot, seed)
        P, O = [], []
        for r in boot_cells:
            rep = _theory_replicates(_theory_aggregates(cells[r["cell"]], w, idx), M, W, r["efficacy_primary"],
                                     min_pos_neg)
            r["n_boot_effective"] = int(rep["ok"].sum())
            for k in ("pred_gross", "pred_net", "obs_gross", "obs_net"):
                r[f"{k}_ci"] = _pct_ci(rep[k], alpha)
            r["diff_gross_ci"] = _pct_ci(rep["obs_gross"] - rep["pred_gross"], alpha)
            r["diff_net_ci"] = _pct_ci(rep["obs_net"] - rep["pred_net"], alpha)
            lo, hi = r["pred_gross_ci"]
            r["inside_predicted_ci"] = _inside(r["obs_gross"], lo, hi) if math.isfinite(lo) else None
            P.append(rep["pred_gross"])
            O.append(rep["obs_gross"])
        Pm, Om = np.array(P), np.array(O)                   # cells x replicates
        ok = np.isfinite(Pm) & np.isfinite(Om)
        n_ok = ok.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mae_b = np.where(n_ok > 0, np.where(ok, np.abs(Om - Pm), 0.0).sum(0) / n_ok, NAN)
            md_b = np.where(n_ok > 0, np.where(ok, Om - Pm, 0.0).sum(0) / n_ok, NAN)
        out["mae_ci"], out["mean_diff_gross_ci"] = _pct_ci(mae_b, alpha), _pct_ci(md_b, alpha)
        inside = [r["inside_predicted_ci"] for r in boot_cells if r["inside_predicted_ci"] is not None]
        out["share_inside_predicted_ci"] = float(np.mean(inside)) if inside else NAN
        out["n_boot_effective"] = int((n_ok > 0).sum())
        if out["slope"] is not None and len(boot_cells) == len(est):
            # a replicate's design is the column of the estimable cells' predictions: full rank iff not all zero
            valid = (n_ok >= min_cells) & (ok & (Pm != 0)).any(0)
            num, den = np.where(ok, Pm * Om, 0.0).sum(0), np.where(ok, Pm * Pm, 0.0).sum(0)
            s = np.full(Pm.shape[1], NAN)
            s[valid] = num[valid] / den[valid]
            lo, hi = _pct_ci(s, alpha)
            out["slope"].update({"ci": [lo, hi], "n_boot_effective": int(valid.sum()),
                                 "ci_covers_1": bool(lo <= 1.0 <= hi) if math.isfinite(lo) else None})
            out["n_boot_effective"] = int(valid.sum())
    out["exploratory"] = {"regression_j": h2_regression_j(cells, W, n_boot=n_boot, seed=seed, alpha=alpha),
                          "note": "exploratory (A6.16 (i)): the pooled regression of Delta-U on J and accuracy is "
                                  "not a falsification condition of H2"}
    return out


def h2_theory_wording(t: Mapping[str, Any]) -> str:
    """The report wording of :func:`h2_theory_check` (A6.16 (i)): *not estimable* with its reason, or consistent /
    inconsistent with Eq. 2 by whether the through-origin slope's CI covers 1 -- never "falsified"."""
    if not t.get("estimable"):
        return f"H2 not estimable: {t.get('reason')}"
    s = t.get("slope") or {}
    lo, hi = (s.get("ci") or [NAN, NAN])
    level = f"{100 * (1 - float(t.get('alpha', 0.05))):g}%"
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (f"H2 estimable on {t['n_cells']} cells: through-origin slope {s.get('value', NAN):.3f}, no "
                f"task-level CI ({t.get('bootstrap_note') or 'no bootstrap replicates'})")
    verdict = "consistent" if lo <= 1.0 <= hi else "inconsistent"
    return (f"H2 {verdict} with Eq. 2: through-origin slope of observed on predicted gross Delta-U "
            f"{s['value']:.3f} ({level} joint-bootstrap CI {lo:.3f} to {hi:.3f}; expected 1) on {t['n_cells']} cells")


def h2_report(cells: Sequence[Mapping[str, Any]], Ws: Sequence[float] = (0.1, 1.0, 10.0), n_boot: int = 2000,
              seed: int = 0) -> dict[str, Any]:
    """The ``h2`` CLI's output from the cells of :func:`h2_cells`: the A6.16 (i) theory check per W (``h2_theory``,
    worded by ``h2_verdict``) as the primary H2 readout; the pooled J / accuracy regression labelled exploratory
    (``h2_regression_exploratory`` -- the pre-A6.16 key ``regression_j`` is kept and points at the same object);
    the Eq. 2 per-cell predictions (``predicted_gain``); the AUROC bound (``regression_auroc``, old key
    ``regression``); and the cells without their task data."""
    theory = {f"{w:g}": h2_theory_check(cells, w, n_boot=n_boot, seed=seed) for w in Ws}
    expl = {k: t["exploratory"]["regression_j"] for k, t in theory.items()}
    out: dict[str, Any] = {"h2_theory": theory, "h2_verdict": {k: h2_theory_wording(t) for k, t in theory.items()},
                           "h2_regression_exploratory": expl, "regression_j": expl,
                           "predicted_gain": {f"{w:g}": [predicted_gain(c, w) for c in cells] for w in Ws},
                           "regression_auroc": {f"{w:g}": h2_regression_auroc(cells, w) for w in Ws},
                           "n_cells": len(cells), "n_estimable": sum(1 for c in cells if c.get("estimable"))}
    out["cells"] = [{k: v for k, v in c.items() if k != "task_data"} for c in cells]
    out["regression"] = out["regression_auroc"]          # pre-A6 key (secondary)
    return out


def _gate_of_row(r: Mapping[str, Any]) -> str | None:
    """The gate of a replay row; None for probe-list rows (``source != "steps"``), which never enter H2."""
    if r.get("source", "steps") != "steps":
        return None
    if r["dp"] == "stop":
        return "stop_guard" if (r.get("gold_aux") or {}).get("kind") or str(r.get("question", "")).startswith(
            "The agent wants to stop") else "stop_trigger"
    return str(r["dp"])


def ceiling(measure: Sequence[Mapping[str, Any]], gate: str = "reflection", W: float = 1.0, n_boot: int = 2000,
            seed: int = 0, mde: float = NO_HEADROOM_MDE, exclude_budget: bool = False) -> dict[str, Any]:
    """C5 one-step oracle ceiling (§9.3): Delta-U_ceiling = W n-bar pi (1 - pi) (e + h) -- a J = 1 gate against the
    matched random gate -- from the measurement arm's golds and counterfactual branches, with a task-bootstrap CI.
    ``no_headroom`` when the CI lies within +-mde (C5 is then reported as "no headroom on this suite").
    ``exclude_budget`` (A6 / P0-8 sensitivity row): drop the need_fatal positives whose ``fatal_kind`` is ``budget``."""
    from .scorecard import task_key

    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eps: dict[str, int] = Counter()
    for t in measure:
        k = task_key(t)
        eps[k] += 1
        recs_t = [r for r in gate_records([t], gate, modes=("shadow",)) if r.get("gold_status") == "ok"]
        by[k] += exclude_budget_fatal(recs_t) if exclude_budget else recs_t
    keys = sorted(eps)
    if not keys:
        return {"n_tasks": 0}

    def stat(ks: Sequence[str]) -> float:
        recs = [r for k in ks for r in by[k]]
        n_ep = sum(eps[k] for k in ks)
        if not recs or not n_ep:
            return NAN
        pi = float(np.mean([bool(r["gold"]) for r in recs]))
        if pi * (1 - pi) == 0:            # no positive (or no negative) state: a perfect gate has nothing to add
            return 0.0
        eff = cf_efficacy(recs, BRANCH_OF.get(gate, "verdict_alt"))
        eb = eff.get("e", NAN) + eff.get("h", NAN)
        return W * (len(recs) / n_ep) * pi * (1 - pi) * eb

    val = stat(keys)
    eff_all = cf_efficacy([r for k in keys for r in by[k]], BRANCH_OF.get(gate, "verdict_alt"))
    if eff_all.get("n") and not eff_all.get("net"):
        return {"gate": gate, "W": W, "n_tasks": len(keys), "value": NAN, "ci": [NAN, NAN], "mde": mde,
                "no_headroom": False, "note": "e / h not net of the placebo (placebo verdicts missing): no ceiling"}
    rng = np.random.default_rng(seed)
    boot = [stat([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(n_boot)]
    boot = [b for b in boot if not math.isnan(b)]
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else [NAN, NAN]
    return {"gate": gate, "W": W, "n_tasks": len(keys), "value": val, "ci": ci, "mde": mde,
            "no_headroom": bool(boot) and -mde <= ci[0] and ci[1] <= mde, "exclude_budget": exclude_budget}


# ====================================================================== one-step CF validity (round 4, P1-6)
CF_BRANCHES = {"reflection": ("verdict_alt", "reflection"), "routing": ("verdict_large", "large"),
               "stop": ("verdict_reject", "reject")}
CF_LABELS = ("helps", "neutral", "harms")
ROLLOUT_SIGNS = ("+", "0", "-")


def cf_label(aux: Mapping[str, Any]) -> str:
    """The one-step label of a counterfactual record: ``helps`` (v0 failing, branch progress), ``harms`` (v0 progress,
    branch failing), else ``neutral`` -- the §4 ``benefit`` / ``harm`` flags."""
    return "helps" if aux.get("benefit") else ("harms" if aux.get("harm") else "neutral")


def cf_opportunities(trajs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every one-step counterfactual label (``cf_status == "ok"`` with a branch verdict) of a measurement arm's
    ``steps[]``, with what a full-episode continuation needs: the logged action prefix (the steps before it, with
    their observations) and the step's own action ``a0``. Fired-step records of live trigger arms (no branch
    verdict, no placebo) are not opportunities."""
    out = []
    for t in trajs:
        steps = list(t.get("steps") or [])
        for i, st in enumerate(steps):
            if st.get("tool") is None:
                break
            for d in st.get("decisions") or []:
                a = d.get("gold_aux") or {}
                br = CF_BRANCHES.get(str(d.get("dp")))
                if a.get("cf_status") != "ok" or br is None or a.get(br[0]) is None or a.get("verdict") is None:
                    continue
                prefix = [(str(x.get("tool")), dict(x.get("args") or {})) for x in steps[:i]]
                out.append({"task_id": str(t.get("task_id")), "step": int(st.get("step", i)), "dp": d["dp"],
                            "branch": br[0], "intervention": br[1], "gold": d.get("gold"), "v0": a["verdict"],
                            "v_branch": a[br[0]], "v_placebo": a.get("verdict_placebo"),
                            "benefit": bool(a.get("benefit")), "harm": bool(a.get("harm")), "label": cf_label(a),
                            "cf_weight": float(a.get("cf_weight", 1.0) or 1.0), "final_step": i == len(steps) - 1,
                            "a0": (str(st.get("tool")), dict(st.get("args") or {})), "prefix": prefix,
                            "logged_obs": [str(x.get("observation", "")) for x in steps[:i]],
                            "episode_success": bool((t.get("result") or {}).get("success"))})
    return out


def select_cf_validity_subset(opps: Sequence[Mapping[str, Any]], n: int = 30, min_helps: int = 10,
                              min_not: int = 10, seed: int = 0, min_harms: int = 0) -> list[dict[str, Any]]:
    """A stratified subset of ``n`` opportunities: at least ``min_helps`` whose one-step label says the intervention
    helps, at least ``min_not`` whose label says it does not (neutral or harms; ``min_harms`` of them drawn from the
    ``harms`` labels first, which are rare), the rest seeded-random from what is left. Within a stratum the decision
    points are drawn round-robin (seeded shuffle inside each), so every intervention is represented. The strata are
    interleaved in the returned order, so a run that stops at its budget keeps the balance. Each row gets
    ``stratum`` in {helps, not_helps, random}. ``min_harms = 0`` (the default) reproduces the seed-0 dev selection."""
    rng = random.Random(seed)

    def pick(pool: Sequence[Mapping[str, Any]], k: int) -> list[dict[str, Any]]:
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for o in pool:
            by[str(o["dp"])].append(dict(o))
        for v in by.values():
            rng.shuffle(v)
        out: list[dict[str, Any]] = []
        while len(out) < k and any(by.values()):
            for dp in sorted(by):
                if by[dp] and len(out) < k:
                    out.append(by[dp].pop())
        return out

    key = lambda o: (o["task_id"], o["step"], o["dp"])    # noqa: E731
    helps = pick([o for o in opps if o["label"] == "helps"], min_helps)
    harms = pick([o for o in opps if o["label"] == "harms"], min_harms) if min_harms > 0 else []
    seen = {key(o) for o in harms}
    nots = harms + pick([o for o in opps if o["label"] != "helps" and key(o) not in seen], max(0, min_not - len(harms)))
    taken = {key(o) for o in helps + nots}
    rest = [dict(o) for o in opps if key(o) not in taken]
    rng.shuffle(rest)
    extra = rest[: max(0, n - len(helps) - len(nots))]
    for o, st in ((x, "helps") for x in helps):
        o["stratum"] = st
    for o in nots:
        o["stratum"] = "not_helps"
    for o in extra:
        o["stratum"] = "random"
    out: list[dict[str, Any]] = []
    queues = [helps, nots, extra]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def cohen_kappa(pairs: Sequence[tuple[Any, Any]]) -> float:
    """Cohen's kappa of two labellings (NaN without pairs or when chance agreement is 1)."""
    if not pairs:
        return NAN
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return NAN if pe >= 1.0 else (po - pe) / (1 - pe)


def rollout_sign(delta: float, eps: float = 1e-9) -> str:
    return "+" if delta > eps else ("-" if delta < -eps else "0")


def cf_validity_table(rows: Sequence[Mapping[str, Any]], min_n: int = 5) -> dict[str, Any]:
    """Agreement of the one-step label with the full-rollout outcome difference. ``rows``: one per opportunity with
    ``label`` (helps / neutral / harms), ``delta`` = mean success with the intervention minus mean success without
    (its sign is the rollout verdict: + / 0 / -), ``dp``, and optionally ``final_step``, ``control_success`` and
    ``episode_success``. Overall and per decision point (``min_n`` rows): the 3 x 3 confusion table, the binary
    agreement (helps <-> +) with a Wilson 95% CI, Cohen's kappa (binary and 3-class), the directional checks and
    the share of control rollouts that reproduced the logged episode outcome (System-2 nondeterminism)."""
    def block(rs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        conf = {lab: {sg: 0 for sg in ROLLOUT_SIGNS} for lab in CF_LABELS}
        pairs3, pairs2 = [], []
        for r in rs:
            sg = rollout_sign(float(r["delta"]))
            conf[str(r["label"])][sg] += 1
            pairs3.append((str(r["label"]), {"+": "helps", "0": "neutral", "-": "harms"}[sg]))
            pairs2.append((str(r["label"]) == "helps", sg == "+"))
        n = len(rs)
        agree = sum(1 for a, b in pairs2 if a == b)
        lo, hi = wilson_ci(agree, n) if n else (NAN, NAN)
        helps = [r for r in rs if r["label"] == "helps"]
        nots = [r for r in rs if r["label"] != "helps"]
        harms = [r for r in rs if r["label"] == "harms"]
        ctrl = [(bool(r["control_success"]), bool(r["episode_success"])) for r in rs
                if r.get("control_success") is not None and r.get("episode_success") is not None]
        return {"n": n, "confusion": conf, "agreement": agree / n if n else NAN, "agreement_ci": [lo, hi],
                "kappa_binary": cohen_kappa(pairs2), "kappa_3class": cohen_kappa(pairs3),
                "helps_confirmed": (sum(1 for r in helps if rollout_sign(r["delta"]) == "+") / len(helps)
                                    if helps else NAN), "n_helps": len(helps),
                "not_helps_confirmed": (sum(1 for r in nots if rollout_sign(r["delta"]) != "+") / len(nots)
                                        if nots else NAN), "n_not_helps": len(nots),
                "harms_confirmed": (sum(1 for r in harms if rollout_sign(r["delta"]) == "-") / len(harms)
                                    if harms else NAN), "n_harms": len(harms),
                "mean_delta_helps": float(np.mean([r["delta"] for r in helps])) if helps else NAN,
                "mean_delta_not_helps": float(np.mean([r["delta"] for r in nots])) if nots else NAN,
                "control_reproduces_log": (sum(1 for a, b in ctrl if a == b) / len(ctrl)) if ctrl else NAN,
                "n_control_checked": len(ctrl)}

    out: dict[str, Any] = {"overall": block(rows), "by_dp": {}, "by_final_step": {}}
    for dp in sorted({str(r["dp"]) for r in rows}):
        rs = [r for r in rows if str(r["dp"]) == dp]
        out["by_dp"][dp] = block(rs) if len(rs) >= min_n else {"n": len(rs), "note": f"fewer than {min_n} rows"}
    for fin in (True, False):
        rs = [r for r in rows if r.get("final_step") is fin]
        if rs:
            out["by_final_step"]["final" if fin else "non_final"] = block(rs) if len(rs) >= min_n else \
                {"n": len(rs), "note": f"fewer than {min_n} rows"}
    return out


# ====================================================================== the copy-coverage valve (§7.1)
DEV_SEEDS = (5000, 5999)


def copy_coverage(envs: Iterable[Any], kinds: Mapping[str, str] | None = None, task_first: int = 25,
                  max_steps: int = 30, threshold: float = 0.8) -> dict[str, Any]:
    """Coverage of every copy parameter: the share of oracle states (the oracle agent, ``SimulatedLLM(error_rate=0)``
    behaviour, $0) at which the raw oracle value is among ``extract_candidates(state, task_first)`` -- the options
    System-1 would get. Parameters below ``threshold`` are listed in ``reclassify_free`` (freeze them as
    ``tool_params.extra.param_kinds: {tool.param: free}``)."""
    from ..decision.types import MAX_OPTIONS
    from ..harness.agent import JevAgent
    from ..harness.config import HarnessConfig
    from ..harness.tools import extract_candidates, finish_tool, params_of, tool_name
    from ..llm.scripted import ScriptedLLM

    agent = JevAgent(ScriptedLLM("render"), HarnessConfig())
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for env in envs:
        obs = env.reset()
        tools = {tool_name(t): t for t in env.tools() + [finish_tool()]}
        k = {**dict(getattr(env, "param_kinds", None) or {}), **dict(kinds or {})}
        history: list[tuple[str, str]] = []
        for _ in range(max_steps):
            a = env.oracle_action()
            if a is None:
                break
            tool, args = a
            state = agent._s1_state(env, history, obs)
            cands = extract_candidates(state, task_first=task_first)[: MAX_OPTIONS - 1]
            for p in params_of(tools[tool], k) if tool in tools else []:
                if p.kind != "copy" or p.name not in args:
                    continue
                h = hits[f"{tool}.{p.name}"]
                h[0] += 1
                h[1] += int(str(args[p.name]) in cands)
            if tool == "finish":
                break
            res = env.step(tool, args)
            history.append((f"{tool}({json.dumps(args)[:200]})", res.observation))
            obs = res.observation
            if res.done:
                break
    table = {p: {"n": n, "covered": c, "coverage": c / n if n else NAN} for p, (n, c) in sorted(hits.items())}
    return {"threshold": threshold, "task_first": task_first, "table": table,
            "reclassify_free": sorted(p for p, r in table.items() if r["n"] and r["coverage"] < threshold)}


# ====================================================================== CLI
def _load(run_dir: str | Path, arms: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    files = {p.stem: p for p in sorted(Path(run_dir).glob("*.jsonl"))}
    want = list(arms) if arms else list(files)
    missing = [a for a in want if a not in files]
    if missing:
        raise SystemExit(f"arm(s) not found in {run_dir}: {missing} (have: {sorted(files)})")
    return {a: load_trajectories(files[a]) for a in want}


def load_runs_split(run_dir: str | Path, arms: Sequence[str], measure_arms: Sequence[str],
                    measure_run: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Live / control arms from ``run_dir``, measurement arms from ``measure_run`` (default: ``run_dir``) -- the C5
    pair lives in the E2c run, the measurement arms in the E2b run. A name present in both runs is refused unless it
    is loaded from one place only."""
    runs = _load(run_dir, [a for a in arms if a not in measure_arms])
    m = _load(measure_run or run_dir, list(measure_arms))
    clash = sorted(set(runs) & set(m))
    if clash:
        raise SystemExit(f"arm name(s) {clash} are both live / control and measurement arms")
    return {**runs, **m}


def load_replays(paths: Sequence[str | Path]) -> dict[str, list[dict[str, Any]]]:
    """Replay JSONL(s) -> {backend spec id: rows} (rows of older replays without an id are keyed by backend name and
    never match a live arm)."""
    reps: dict[str, list[dict[str, Any]]] = {}
    for fp in paths:
        for line in open(fp):
            if line.strip():
                r = json.loads(line)
                reps.setdefault(str(r.get("backend_id") or f"name:{r.get('backend')}"), []).append(r)
    return reps


def _jsonable(x: Any) -> Any:
    if isinstance(x, float):
        return None if math.isnan(x) else x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    return x


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser("python -m jevharness.bench.meta_gold", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("run_dir")
    r.add_argument("--arm", required=True)
    r.add_argument("--s1", required=True, help="backend spec (JSON / YAML), as in an experiment's s1:")
    r.add_argument("--cache", default=None)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--out", default=None)
    a_ = sub.add_parser("audit")
    a_.add_argument("run_dir")
    a_.add_argument("--arm", action="append", required=True)
    a_.add_argument("--implications", action="store_true",
                    help="A6 / P0-8: the implication audit, the budget-kind share of need_fatal and the sensitivity "
                         "gate stats without budget-kind positives (no replay needed)")
    a_.add_argument("--no-replay", action="store_true", help="skip the replay audit (implications only)")
    h = sub.add_parser("h2")
    h.add_argument("run_dir")
    h.add_argument("--measure", action="append", required=True, help="measurement arm(s) (one per System-2 family)")
    h.add_argument("--pair", action="append", required=True, metavar="LIVE:CONTROL")
    h.add_argument("--W", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    h.add_argument("--replay", action="append", default=None,
                   help="replay JSONL(s) of other backends on the measurement arm (matched to live arms by spec id)")
    h.add_argument("--measure-run", default=None,
                   help="run directory of the measurement arms when they live in another experiment (e.g. "
                        "runs/e2b_jevgym_meta_control next to the C5 pair in runs/e2c_jevgym_baselines)")
    h.add_argument("--n-boot", type=int, default=2000)
    c = sub.add_parser("ceiling")
    c.add_argument("run_dir")
    c.add_argument("--measure", required=True)
    c.add_argument("--gate", default="reflection")
    c.add_argument("--W", type=float, default=1.0)
    c.add_argument("--n-boot", type=int, default=2000)
    c.add_argument("--exclude-budget", action="store_true", help="A6 / P0-8 sensitivity: drop budget-kind positives")
    v = sub.add_parser("coverage")
    v.add_argument("--seeds", default=f"{DEV_SEEDS[0]}-{DEV_SEEDS[1]}")
    v.add_argument("--suite", default="{name: jevgym, families: v2, distractor_style: near, n_distractors: 60, "
                                      "noise_lines: 20, strict: true}")
    v.add_argument("--task-first", type=int, default=25)
    for p in (r, a_, h, c, v):
        p.add_argument("--format", choices=["json", "md"], default="json")
    a = ap.parse_args(argv)
    out: Any
    if a.cmd == "replay":
        from ..decision import make_backend

        trajs = _load(a.run_dir, [a.arm])[a.arm]
        backend = make_backend(yaml.safe_load(a.s1), cache=a.cache)
        rows = replay_backend(trajs, backend, workers=a.workers, arm=a.arm)
        name = re.sub(r"[^\w.\-]+", "_", str(backend.describe().get("backend", "backend")))
        path = Path(a.out) if a.out else Path(a.run_dir) / "replay" / f"{a.arm}__{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(_jsonable(row), default=str) + "\n")
        out = {"rows": len(rows), "errors": sum(1 for x in rows if x["error"]), "out": str(path)}
    elif a.cmd == "audit":
        loaded = _load(a.run_dir, a.arm)
        out = {}
        facs = None if a.no_replay else suite_factories(a.run_dir)
        for arm, t in loaded.items():
            out[arm] = audit(t, facs) if facs is not None else {}
            if a.implications:
                out[arm]["implications"] = implication_audit(t)
                out[arm]["sensitivity_no_budget"] = {
                    g: {"all": gate_stats(rs, arm_tau(t, g) or 0.5, guard=g == "stop_guard"),
                        "no_budget": gate_stats(exclude_budget_fatal(rs), arm_tau(t, g) or 0.5, guard=g == "stop_guard")}
                    for g in ("reflection", "routing")
                    for rs in [[r for r in gate_records(t, g, modes=("shadow",)) if r.get("gold_status") == "ok"]]
                    if rs}
    elif a.cmd == "h2":
        pairs = [tuple(x.split(":", 1)) for x in a.pair]
        runs = load_runs_split(a.run_dir, sorted({x for p in pairs for x in p}), a.measure, a.measure_run)
        reps = load_replays(a.replay or [])
        cells = h2_cells(runs, a.measure, pairs, a.W, reps, a.n_boot)   # type: ignore[arg-type]
        out = h2_report(cells, a.W, n_boot=a.n_boot)
    elif a.cmd == "ceiling":
        out = ceiling(_load(a.run_dir, [a.measure])[a.measure], a.gate, a.W, a.n_boot,
                      exclude_budget=a.exclude_budget)
    else:
        from .suites import load_suite

        lo, hi = (int(x) for x in a.seeds.split("-"))
        kw = dict(yaml.safe_load(a.suite))
        facs = load_suite(kw.pop("name"), n=hi - lo + 1, seed0=lo, **kw)
        out = copy_coverage((f() for f in facs), task_first=a.task_first)
    out = _jsonable(out)
    if a.format == "md" and a.cmd == "coverage":
        lines = ["| parameter | n | coverage |", "|---|---|---|"]
        lines += [f"| {p} | {r['n']} | {r['coverage']:.3f} |" for p, r in out["table"].items()]
        lines.append(f"\nreclassified free (< {out['threshold']}): {', '.join(out['reclassify_free']) or 'none'}")
        print("\n".join(lines))
    else:
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
