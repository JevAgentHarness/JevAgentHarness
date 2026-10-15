#!/usr/bin/env python3
r"""E6f (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry
"Exploratory row E6f" of 2026-10-07): the trigger cells at tool choice, tool arguments and action, the last three cells
of the module map (harness/config.py E6F_CELLS; semantics: results/explore/e6f_modes/cells.md, written by this
script from harness/prompts.py). TRIGGER = the decision model decides whether the LLM runs one verification call on
its own proposal; ``always`` (no decision model) verifies every proposal. Tool choice and tool arguments on fresh
JevGym seeds 301600-301799 (``runs/e6f_modes``), the action on ALFWorld valid_seen (``runs/e6f_alfworld_eval``), each
cell against the ``llm_only`` arm of its own run. No test; outside every Holm table.

    python scripts/explore_e6f_modes.py spend                          # logged $ of every E6f run (smoke + eval)
    python scripts/explore_e6f_modes.py watch --pid PID [--pid PID2]   # watchdog: Gemini $60 / Jev $2 over all E6f runs
    python scripts/explore_e6f_modes.py project                        # smoke -> eval cost projection (registered gate)
    python scripts/explore_e6f_modes.py cells                          # results/explore/e6f_modes/cells.md
    python scripts/explore_e6f_modes.py summarize                      # summary.{json,md} + paper/generated/e6f.tex

Readout (fixed in the Deviations entry before any episode). Per cell, the module map's paired comparison with the
same run's ``llm_only`` (paired complete case; Delta success with the paired task bootstrap, B = 10,000, seed 0, and
the Newcombe interval; wins / losses; cost per episode and saving with a task-bootstrap interval; the fixed verdict
rule of scripts/module_map.py with acted = the share of proposals on which the trigger fired; accuracy = the
trigger's answer against the gold "the proposal is wrong"). Then the verification itself, for the trigger arm and
its always-verify reference: proposals, the share verified, the share of the point's wrong proposals verified, and
what each verification did by the gold -- fixed (wrong -> right), broke (right -> wrong), confirmed, revised without a
change of correctness, replies without a call. Tool choice is judged at the tool level (the tool in the acceptable set
AT(s)), with the call level beside it; tool arguments at the call level (the call is progress); the action against
the single gold action (a lower bound, as E6c). The always-verify comparison: its own Delta and saving against
``llm_only``; the trigger's share of its extra cost, (C_trigger - C_ref) / (C_always - C_ref); its share of its
effect, Delta_trigger / Delta_always (only when Delta_always != 0) and fixes_trigger / fixes_always; and the paired
trigger-vs-always Delta. Everything descriptive."""
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402
import explore_e6b_modes as E  # noqa: E402
import module_map as MM  # noqa: E402

from jevharness.bench import metrics as M  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "results/explore/e6f_modes"
TEX = ROOT / "paper/generated/e6f.tex"
PREFIX, LABEL = "e6f", "exploratory (E6f)"
GEMINI_CAP, JEV_CAP, SAFETY = 60.0, 2.0, 1.5
REF = "llm_only"
# setting -> (smoke run, eval run, evaluation sizes tried in order, arms)
SETTINGS: dict[str, dict[str, Any]] = {
    "jevgym": {"smoke": "e6f_modes_smoke", "eval": "e6f_modes", "n": (200, 100),
               "arms": (REF, "ts_trigger", "ts_verify_all", "tp_trigger", "tp_verify_all")},
    "alfworld": {"smoke": "e6f_alfworld_smoke", "eval": "e6f_alfworld_eval", "n": (140,),
                 "arms": (REF, "jev_action_trigger", "action_verify_all")},
}
# cell arm -> (decision point, mode, setting, always-verify reference arm)
CELLS: dict[str, tuple[str, str, str, str]] = {
    "ts_trigger": ("tool_select", "trigger", "jevgym", "ts_verify_all"),
    "tp_trigger": ("tool_params", "trigger", "jevgym", "tp_verify_all"),
    "jev_action_trigger": ("action", "trigger", "alfworld", "action_verify_all"),
}
TAU = {"tool_select": 0.5, "tool_params": 0.9, "action": 0.5}
OUTCOMES = ("fixed", "broke", "confirmed_right", "confirmed_wrong", "revised_right", "revised_wrong")
NAN = float("nan")


def arm_path(run_dir: Path, arm: str) -> Path | None:
    """``<arm>.jsonl``, or its gzipped copy (records are gzipped after the run to save disk)."""
    for p in (run_dir / f"{arm}.jsonl", run_dir / f"{arm}.jsonl.gz"):
        if p.exists():
            return p
    return None


def _records(p: Path) -> list[dict[str, Any]]:
    import gzip
    text = gzip.decompress(p.read_bytes()).decode() if p.name.endswith(".gz") else p.read_text()
    out = []
    for line in text.splitlines():
        try:
            t = json.loads(line)
        except ValueError:
            continue
        if isinstance(t, dict) and "task_id" in t:
            out.append(t)
    return out


# ============================================================================== spend, watchdog, projection
def run_spend(run_dir: Path) -> dict[str, Any]:
    """Every record of every arm of one run (errored attempts and re-runs included: all of them were paid)."""
    out = {"gemini": 0.0, "jev": 0.0, "total": 0.0, "records": 0}
    for p in sorted(run_dir.glob("*.jsonl*")) if run_dir.is_dir() else []:
        if p.name.startswith("schedule"):
            continue
        for t in _records(p):
            s = E.episode_spend(t)
            for k in ("gemini", "jev", "total"):
                out[k] += s[k]
            out["records"] += 1
    return out


def total_spend(runs: Path = RUNS) -> dict[str, Any]:
    names = [s[k] for s in SETTINGS.values() for k in ("smoke", "eval")]
    per = {n: run_spend(runs / n) for n in names if (runs / n).is_dir()}
    tot = {k: sum(v[k] for v in per.values()) for k in ("gemini", "jev", "total", "records")}
    return {"by_run": per, **tot}


def cmd_spend(a: argparse.Namespace) -> int:
    print(json.dumps(total_spend(Path(a.runs)), indent=1))
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    """Poll the logged spend of every E6f run; when Gemini reaches ``--gemini-cap`` or Jev ``--jev-cap``, send SIGTERM
    (then SIGKILL) to every runner's process group and write ``<runs>/e6f_STOP.json``."""
    stop = Path(a.runs) / f"{PREFIX}_STOP.json"
    while True:
        alive = []
        for pid in a.pid:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except OSError:
                pass
        if not alive:
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} runners {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(Path(a.runs))
        if s["gemini"] >= a.gemini_cap or s["jev"] >= a.jev_cap:
            stop.write_text(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                        "spend": s, "caps": {"gemini": a.gemini_cap, "jev": a.jev_cap}}, indent=1))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for pid in alive:
                    try:
                        os.killpg(os.getpgid(pid), sig)
                    except OSError:
                        pass
                time.sleep(10)
            print(f"CAP REACHED: Gemini ${s['gemini']:.2f} / Jev ${s['jev']:.3f}; runners stopped", flush=True)
            return 1
        print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} spend Gemini ${s['gemini']:.3f}, Jev ${s['jev']:.4f} "
              f"({s['records']} records)", flush=True)
        time.sleep(a.every)


def projection(runs: Path = RUNS, safety: float = SAFETY) -> dict[str, Any]:
    """The registered gate: per setting, the smoke's mean $ per episode per arm (Gemini and Jev apart), summed over the
    arms, x the evaluation size x ``safety``; plus the spend so far, against the Gemini cap ($60) and the Jev cap ($2).
    Options in order: JevGym 200 + ALFWorld 140; JevGym 100 + ALFWorld 140; else stop after the smoke."""
    out: dict[str, Any] = {"safety": safety, "caps": {"gemini": GEMINI_CAP, "jev": JEV_CAP}, "settings": {}}
    missing = []
    for name, s in SETTINGS.items():
        per: dict[str, Any] = {}
        for arm in s["arms"]:
            p = arm_path(runs / s["smoke"], arm)
            ts = list(C.load_latest([p])[0].values()) if p else []
            if not ts:
                missing.append(f"{name}/{arm}")
                continue
            sp = [E.episode_spend(t) for t in ts]
            per[arm] = {"n": len(ts), "errors": sum(bool(t.get("error")) for t in ts),
                        "gemini": float(np.mean([x["gemini"] for x in sp])), "jev": float(np.mean([x["jev"] for x in sp])),
                        "wall_s": float(np.mean([float(t.get("wall_ms") or 0) for t in ts])) / 1000}
        out["settings"][name] = {"per_episode": per, "per_task_gemini": sum(v["gemini"] for v in per.values()),
                                 "per_task_jev": sum(v["jev"] for v in per.values())}
    so_far = total_spend(runs)
    out["spent_so_far"] = {k: so_far[k] for k in ("gemini", "jev", "total")}
    out["missing"] = missing
    options = [(200, 140), (100, 140)]
    for nj, na in options:
        sj, sa = out["settings"]["jevgym"], out["settings"]["alfworld"]
        g = so_far["gemini"] + safety * (sj["per_task_gemini"] * nj + sa["per_task_gemini"] * na)
        j = so_far["jev"] + safety * (sj["per_task_jev"] * nj + sa["per_task_jev"] * na)
        out[f"projected_{nj}_{na}"] = {"gemini": g, "jev": j, "fits": g <= GEMINI_CAP and j <= JEV_CAP}
    fit = [o for o in options if out[f"projected_{o[0]}_{o[1]}"]["fits"]]
    out["decision"] = (f"run JevGym {fit[0][0]} tasks and ALFWorld {fit[0][1]} games" if fit
                       else "stop after the smoke")
    if missing:
        out["decision"] = "stop: smoke incomplete (" + ", ".join(missing) + ")"
    return out


def cmd_project(a: argparse.Namespace) -> int:
    p = projection(Path(a.runs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "projection.json").write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(p, indent=1))
    return 0


# ============================================================================== readout
def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else NAN


def verify_metrics(trajs: Sequence[Mapping[str, Any]], dp: str, mode: str) -> dict[str, Any]:
    """What the verification pass did at ``dp`` in one arm (trigger or always): proposals, verified share, outcomes."""
    recs = [r for r in M.decision_records(list(trajs)) if r.get("dp") == dp and r.get("mode") == mode
            and not r.get("shadow")]
    aux = [(r, r.get("gold_aux") or {}) for r in recs]
    fired = [(r, a) for r, a in aux if a.get("fired")]
    gold = [(r, a) for r, a in aux if r.get("gold_status") == "ok" and isinstance(r.get("gold"), bool)]
    wrong = [(r, a) for r, a in gold if r["gold"] is True]          # the proposal needed a second look
    out: dict[str, Any] = {"n_proposals": len(recs), "n_verified": len(fired),
                           "verified": len(fired) / len(recs) if recs else NAN,
                           "n_gold": len(gold), "proposal_wrong": _mean([r["gold"] for r, _ in gold]),
                           "n_wrong": len(wrong), "wrong_verified": _mean([bool(a.get("fired")) for _, a in wrong]),
                           "right_verified": _mean([bool(a.get("fired")) for r, a in gold if r["gold"] is False]),
                           "changed": _mean([bool(a.get("changed")) for _, a in fired]),
                           "reply_no_call": sum(a.get("verify_reply") in ("text", "empty") for _, a in fired)}
    for o in OUTCOMES:
        out[o] = sum(a.get("outcome") == o for _, a in fired)
    out["net_fixed"] = out["fixed"] - out["broke"]
    if dp == "tool_select":        # the whole call (tool and arguments) beside the tool level
        for o in OUTCOMES:
            out[f"call_{o}"] = sum(a.get("outcome_call") == o for _, a in fired)
    out["final_wrong"] = _mean([a.get("final_ok") is False for _, a in gold if a.get("final_ok") is not None])
    return out


def _per_ep_calls(trajs: Sequence[Mapping[str, Any]], dp: str) -> float:
    """System-2 calls per episode at ``dp`` (the proposal and, where it ran, the verification)."""
    return _mean([sum(1 for c in t.get("calls") or [] if c.get("component") == "s2" and c.get("decision_point") == dp)
                   for t in trajs])


def compare(run: Path, arm: str, ref: str, n_boot: int, seed: int) -> dict[str, Any] | None:
    pa, pr = arm_path(run, arm), arm_path(run, ref)
    if pa is None or pr is None:
        return None
    a, r = MM.load_arm(pa), MM.load_arm(pr)
    expected = list(dict.fromkeys(list(r["episodes"]) + list(a["episodes"])))
    cmp = MM.compare(r, a, expected, n_boot, seed)
    keys = cmp.pop("keys")
    if keys:
        ya = [float(a["episodes"][t]["result"]["success"]) for t in keys]
        yr = [float(r["episodes"][t]["result"]["success"]) for t in keys]
        cmp["newcombe_pt"] = C.newcombe(ya, yr)["ci_pt"]
    cmp["provenance"] = {k: a[k] for k in ("path", "sha256", "n_records")}
    cmp["ref_provenance"] = {k: r[k] for k in ("path", "sha256", "n_records")}
    return {"cmp": cmp, "keys": keys}


def _ratio(num: float, den: float) -> float:
    return num / den if MM._finite(num) and MM._finite(den) and den != 0 else NAN


def summarize(runs: Path = RUNS, n_boot: int = MM.N_BOOT, seed: int = MM.SEED) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    refs: dict[str, Any] = {}
    for arm, (dp, mode, setting, always) in CELLS.items():
        run = runs / SETTINGS[setting]["eval"]
        res = compare(run, arm, REF, n_boot, seed)
        if res is None:
            cells[arm] = {"dp": dp, "mode": mode, "state": "not run", "reference": REF, "setting": setting}
            continue
        cmp, keys = res["cmp"], res["keys"]
        latest = {a: C.load_latest([p])[0] for a in (arm, always, REF) if (p := arm_path(run, a))}
        trajs = [latest[arm][t] for t in keys if t in latest[arm]]
        ref_trajs = [latest[REF][t] for t in keys if t in latest[REF]]
        pm = E.cell_metrics(trajs, dp, mode)
        vm = verify_metrics(trajs, dp, mode)
        v = MM.verdict(cmp.get("delta", NAN), cmp.get("saving", NAN), pm.get("acted", NAN))
        cell = {"dp": dp, "mode": mode, "state": "exploratory", "arm": arm, "reference": REF, "setting": setting,
                "tau": TAU[dp], "cmp": cmp, "points": pm, "verify": vm, "verdict": v,
                "per_episode": {**E.per_episode(trajs), "dp_llm_calls": _per_ep_calls(trajs, dp)},
                "ref_per_episode": {**E.per_episode(ref_trajs), "dp_llm_calls": _per_ep_calls(ref_trajs, dp)},
                "attempts": len(_records(arm_path(run, arm))), "run": C.rel(run)}
        alw = compare(run, always, REF, n_boot, seed)
        if alw is not None:
            atrajs = [latest[always][t] for t in alw["keys"] if t in latest[always]]
            ac = alw["cmp"]
            tva = compare(run, arm, always, n_boot, seed)       # trigger vs always, paired
            cell["always"] = {"arm": always, "cmp": ac, "verify": verify_metrics(atrajs, dp, "always"),
                              "per_episode": {**E.per_episode(atrajs), "dp_llm_calls": _per_ep_calls(atrajs, dp)},
                              "trigger_vs_always": tva["cmp"] if tva else None}
            extra_t = cmp.get("cost", NAN) - cmp.get("cost_ref", NAN)
            extra_a = ac.get("cost", NAN) - ac.get("cost_ref", NAN)
            cell["always"]["cost_share"] = _ratio(extra_t, extra_a)     # trigger's share of always' extra cost
            cell["always"]["effect_share"] = _ratio(cmp.get("delta", NAN), ac.get("delta", NAN))
            cell["always"]["fix_share"] = _ratio(float(vm["fixed"]), float(cell["always"]["verify"]["fixed"]))
            cell["always"]["net_fix_share"] = _ratio(float(vm["net_fixed"]),
                                                     float(cell["always"]["verify"]["net_fixed"]))
        cells[arm] = cell
        if setting not in refs:
            ts = [t for t in latest.get(REF, {}).values() if not t.get("error")]
            refs[setting] = {"run": C.rel(run), "n_ok": len(ts), "n_records": len(latest.get(REF, {})),
                             "success": _mean([float(C.success(t)) for t in ts]), "cost": _mean([C.cost(t) for t in ts]),
                             **E.per_episode(ts)}
    meta = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": C.git_head(),
            "script": "scripts/explore_e6f_modes.py", "label": LABEL, "n_boot": n_boot, "seed": seed,
            "run": ", ".join(C.rel(runs / s["eval"]) for s in SETTINGS.values()),
            "spend": total_spend(runs), "caps": {"gemini": GEMINI_CAP, "jev": JEV_CAP},
            "projection": json.loads((OUT_DIR / "projection.json").read_text())
            if (OUT_DIR / "projection.json").exists() else None,
            "stopped_by_watchdog": (runs / f"{PREFIX}_STOP.json").exists(),
            "rule": "costs success if delta <= -3 pt; no saving if saving <= 0; takes over safely if delta >= -1 pt, "
                    "saving > 0 and acted >= 5 %; else mixed (scripts/module_map.py, fixed 2026-10-05)"}
    return {"meta": meta, "cells": cells, "references": refs}


# ---------------------------------------------------------------- writers
_pt = E._pt


def _ci(c: Mapping[str, Any], k: str, sign: bool = True) -> str:
    lo, hi = (c.get(f"{k}_ci") or [NAN, NAN])[:2]
    return f"[{_pt(lo, sign)}, {_pt(hi, sign)}]"


def _outc(v: Mapping[str, Any], prefix: str = "") -> str:
    return ", ".join(f"{o.replace('_', ' ')} {v.get(prefix + o, 0)}" for o in OUTCOMES)


def to_markdown(res: Mapping[str, Any]) -> str:
    m = res["meta"]
    sp = m["spend"]
    out = [f"# E6F: the trigger cells at tool choice, tool arguments and action ({LABEL})", "",
           f"- generated {m['generated']} at git `{(m['git'] or '?')[:12]}` by `{m['script']}`; runs `{m['run']}`; "
           f"paired task bootstrap B = {m['n_boot']}, seed {m['seed']}",
           f"- spend (every E6f record, smokes included): Gemini ${sp['gemini']:.2f} of the ${m['caps']['gemini']:.0f} cap, "
           f"Jev ${sp['jev']:.3f} of the ${m['caps']['jev']:.0f} cap" + ("; **stopped by the watchdog**"
                                                                        if m["stopped_by_watchdog"] else ""),
           f"- verdict rule (descriptive): {m['rule']}",
           "- definition, questions, verification notes and thresholds: `results/explore/e6f_modes/cells.md`; "
           "registration: `docs/PREREG_A7_STAGE2.md`, Deviations \"Exploratory row E6f\". No test; outside every Holm "
           "table.", ""]
    for s, r in res["references"].items():
        out.append(f"- reference `llm_only` ({s}, `{r['run']}`): n = {r['n_ok']} complete, success "
                   f"{_pt(r['success'], False)} %, ${1000 * r['cost']:.1f} per 1k episodes, {r['wall_s_mean']:.1f} s "
                   "per episode")
    out += ["", "| cell | arm | τ | n | succ. % | Δ pt [95 % CI] | wins / losses | $/1k ep. | saving % [95 % CI] | "
            "verified % | fixed / broke | acc. % | verdict |", "|" + "---|" * 13]
    for arm, c in res["cells"].items():
        if c["state"] != "exploratory":
            out.append(f"| {c['dp']}/{c['mode']} | `{arm}` | | -- | | | | | | | | | not run |")
            continue
        k, pm, vm = c["cmp"], c["points"], c["verify"]
        out.append(f"| {c['dp']}/trigger | `{arm}` | {c['tau']} | {k.get('n')} | {_pt(k.get('success'), False)} | "
                   f"{_pt(k.get('delta'))} {_ci(k, 'delta')} | {k.get('wins')} / {k.get('losses')} | "
                   f"{1000 * k.get('cost', NAN):.1f} | {_pt(k.get('saving'), False)} {_ci(k, 'saving', False)} | "
                   f"{_pt(vm.get('verified'), False)} | {vm['fixed']} / {vm['broke']} | {_pt(pm.get('acc'), False)} | "
                   f"{c['verdict'] or '--'} |")
    out += ["", "| always-verify reference | arm | n | Δ pt vs LLM-only | saving % | verified % | fixed / broke | "
            "trigger's share of its extra cost | of its Δ | of its fixes | trigger − always, Δ pt |", "|" + "---|" * 11]
    for arm, c in res["cells"].items():
        a = c.get("always")
        if not a:
            continue
        ac, av, tv = a["cmp"], a["verify"], a.get("trigger_vs_always") or {}
        out.append(f"| {c['dp']} | `{a['arm']}` | {ac.get('n')} | {_pt(ac.get('delta'))} {_ci(ac, 'delta')} | "
                   f"{_pt(ac.get('saving'), False)} | {_pt(av.get('verified'), False)} | {av['fixed']} / {av['broke']} | "
                   f"{_pt(a.get('cost_share'), False)} % | {_pt(a.get('effect_share'), False)} % | "
                   f"{_pt(a.get('fix_share'), False)} % | {_pt(tv.get('delta'))} {_ci(tv, 'delta')} |")
    out += ["", "Verified: the share of the point's proposals on which the LLM ran its verification call (trigger: the "
            "decision model fired, the module map's \"acted\"; always: 100 %). Fixed / broke: verifications that turned "
            "a wrong proposal into a right one / a right one into a wrong one, by the gold (tool choice: the tool is in "
            "the acceptable set AT(s); tool arguments: the call is progress; action: the single gold action, a lower "
            "bound). Acc.: the trigger's answer (fire = the proposal is wrong) against that gold. Share of the extra "
            "cost: (C_trigger − C_LLM-only) / (C_always − C_LLM-only); share of Δ only where always-verify's Δ ≠ 0.",
            "", "## Per cell", ""]
    for arm, c in res["cells"].items():
        if c["state"] != "exploratory":
            continue
        k, pm, vm, pe, pr = c["cmp"], c["points"], c["verify"], c["per_episode"], c["ref_per_episode"]
        out.append(f"### {c['dp']} × trigger (`{arm}`, τ = {c['tau']}; {c['setting']}, `{c['run']}`)")
        out.append("")
        out.append(f"- n = {k['n']} paired tasks ({k['n_dropped']} dropped: {k['dropped']}); success "
                   f"{_pt(k.get('success'), False)} % vs {_pt(k.get('success_ref'), False)} %; Δ {_pt(k.get('delta'))} pt "
                   f"(bootstrap {_ci(k, 'delta')}, Newcombe [{(k.get('newcombe_pt') or [NAN])[0]:+.1f}, "
                   f"{(k.get('newcombe_pt') or [NAN, NAN])[1]:+.1f}]); wins / losses {k['wins']} / {k['losses']}")
        out.append(f"- cost ${1000 * k['cost']:.1f} vs ${1000 * k['cost_ref']:.1f} per 1k episodes (saving "
                   f"{_pt(k.get('saving'), False)} %); System-1 ${1000 * pe['s1_cost']:.2f}, System-2 "
                   f"${1000 * pe['s2_cost']:.1f} per 1k; LLM calls {pe['llm_calls']:.2f} vs {pr['llm_calls']:.2f} per "
                   f"episode ({pe['dp_llm_calls']:.2f} vs {pr['dp_llm_calls']:.2f} at this point), System-1 calls "
                   f"{pe['s1_calls']:.2f}; wall {pe['wall_s_mean']:.1f} s vs {pr['wall_s_mean']:.1f} s (median "
                   f"{pe['wall_s_median']:.1f} vs {pr['wall_s_median']:.1f}); episodes with an empty LLM reply "
                   f"{pe.get('empty_reply_episodes', 0)} vs {pr.get('empty_reply_episodes', 0)}")
        out.append(f"- trigger: {vm['n_proposals']} proposals, verified {_pt(vm['verified'], False)} % "
                   f"({vm['n_verified']}); wrong proposals {_pt(vm['proposal_wrong'], False)} % of {vm['n_gold']} with a "
                   f"gold, of which verified {_pt(vm['wrong_verified'], False)} % (right ones verified "
                   f"{_pt(vm['right_verified'], False)} %); its answers' accuracy {_pt(pm.get('acc'), False)} %")
        out.append(f"- what the verifications did: {_outc(vm)}; changed {_pt(vm['changed'], False)} % of the verified "
                   f"proposals; replies without a call {vm['reply_no_call']}"
                   + (f"; whole call: {_outc(vm, 'call_')}" if c["dp"] == "tool_select" else ""))
        a = c.get("always")
        if a:
            ac, av, ape = a["cmp"], a["verify"], a["per_episode"]
            out.append(f"- always-verify (`{a['arm']}`): n = {ac.get('n')}; success {_pt(ac.get('success'), False)} %, "
                       f"Δ {_pt(ac.get('delta'))} pt {_ci(ac, 'delta')} (wins / losses {ac.get('wins')} / "
                       f"{ac.get('losses')}); ${1000 * ac.get('cost', NAN):.1f} per 1k (saving "
                       f"{_pt(ac.get('saving'), False)} %); {ape['dp_llm_calls']:.2f} LLM calls per episode at this point; "
                       f"wall {ape['wall_s_mean']:.1f} s; {av['n_proposals']} proposals verified, {_outc(av)}"
                       + (f"; whole call: {_outc(av, 'call_')}" if c["dp"] == "tool_select" else ""))
            tv = a.get("trigger_vs_always") or {}
            out.append(f"- trigger vs always-verify: share of its extra cost {_pt(a.get('cost_share'), False)} %, of "
                       f"its Δ {_pt(a.get('effect_share'), False)} %, of its fixes {_pt(a.get('fix_share'), False)} % "
                       f"(net of breaks {_pt(a.get('net_fix_share'), False)} %); paired trigger − always Δ "
                       f"{_pt(tv.get('delta'))} pt {_ci(tv, 'delta')}, wins / losses {tv.get('wins')} / {tv.get('losses')}")
        out.append("")
    return "\n".join(out)


def to_tex_keys(res: Mapping[str, Any]) -> dict[str, str]:
    vals: dict[str, str] = {}
    for arm, c in res["cells"].items():
        p = f"{PREFIX}/{arm}"
        vals[f"{p}/state"] = c["state"]
        if c["state"] != "exploratory":
            continue
        vals.update(MM._num_keys(p, c["cmp"], c["points"]))
        for kk in ("wins", "losses"):
            if c["cmp"].get(kk) is not None:
                vals[f"{p}/{kk}"] = str(int(c["cmp"][kk]))
        if c["verdict"]:
            vals[f"{p}/verdict"] = c["verdict"]
            vals[f"{p}/vcode"] = str(MM.VCODE[c["verdict"]])
        vm = c["verify"]
        for kk in ("verified", "proposal_wrong", "wrong_verified", "right_verified", "changed"):
            if MM._finite(vm.get(kk)):
                vals[f"{p}/{kk}"] = f"{100 * vm[kk]:.1f}"
        for kk in ("n_proposals", "n_verified", "net_fixed", "reply_no_call", *OUTCOMES):
            vals[f"{p}/{kk}"] = str(int(vm.get(kk, 0)))
        vals[f"{p}/tau"] = f"{c['tau']}"
        vals[f"{p}/wall_s"] = f"{c['per_episode']['wall_s_mean']:.1f}"
        a = c.get("always")
        if a:
            q = f"{p}/always"
            vals.update({kk: vv for kk, vv in MM._num_keys(q, a["cmp"], None).items()})
            for kk in ("fixed", "broke", "net_fixed", "n_proposals"):
                vals[f"{q}/{kk}"] = str(int(a["verify"].get(kk, 0)))
            for kk in ("cost_share", "effect_share", "fix_share", "net_fix_share"):
                if MM._finite(a.get(kk)):
                    vals[f"{q}/{kk}"] = f"{100 * a[kk]:.1f}"
            tv = a.get("trigger_vs_always")
            if tv and MM._finite(tv.get("delta")):
                vals[f"{q}/tva_delta"] = f"{100 * tv['delta']:+.1f}"
    for s, r in res["references"].items():
        if MM._finite(r.get("success")):
            vals[f"{PREFIX}/ref/{s}/success"] = f"{100 * r['success']:.1f}"
            vals[f"{PREFIX}/ref/{s}/cost_k"] = f"{1000 * r['cost']:.1f}"
            vals[f"{PREFIX}/ref/{s}/n"] = str(r["n_ok"])
    sp = res["meta"]["spend"]
    vals[f"{PREFIX}/spend/gemini"] = f"{sp['gemini']:.2f}"
    vals[f"{PREFIX}/spend/jev"] = f"{sp['jev']:.2f}"
    vals[f"{PREFIX}/count/cells"] = str(len(CELLS))
    vals[f"{PREFIX}/count/run"] = str(sum(c["state"] == "exploratory" for c in res["cells"].values()))
    return vals


def write_tex(vals: Mapping[str, str], path: Path | None = None) -> Path:
    path = TEX if path is None else path
    lines = ["% AUTO-GENERATED by scripts/explore_e6f_modes.py -- do not edit by hand.",
             "% Exploratory row E6f (descriptive; docs/PREREG_A7_STAGE2.md Deviations 2026-10-07), outside both stages.",
             f"% last written {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git "
             f"{(C.git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


# ---------------------------------------------------------------- cells.md (from the code, so it never drifts)
def cells_markdown() -> str:
    from jevharness.harness import prompts as P
    from jevharness.harness.config import E6F_CELLS, UNDEFINED_CELLS
    name, args, act = "<tool>", {"<argument>": "<value>"}, "<action>"
    cells = ", ".join(sorted(f"{dp.value} × {m.value}" for dp, m in E6F_CELLS))
    return "\n".join([
        "# E6f: the trigger cells at tool choice, tool arguments and action", "",
        "Exploratory row E6f (2026-10-07; `docs/PREREG_A7_STAGE2.md`, Deviations entry \"Exploratory row E6f\"). Until "
        "E6f these three cells of the module map were undefined: a trigger decides whether an optional module runs, and "
        "the tool, its arguments and the action are needed at every step, so there was no optional module to trigger. "
        "At the author's request E6f gives each of these points an optional module, an **LLM verification pass** on "
        "the LLM's own proposal, and lets the decision model decide when it runs. Machine-readable: `harness/config.py` "
        f"`E6F_CELLS` ({cells}); `UNDEFINED_CELLS` is now empty ({len(UNDEFINED_CELLS)} cells), so every one of the "
        "40 cells is defined. This file is written by `scripts/explore_e6f_modes.py cells` from `harness/prompts.py`.",
        "", "## Definition", "",
        "1. The LLM proposes the tool call (tool choice, tool arguments) or the admissible action (action) exactly as "
        "in the LLM-only harness.",
        "2. The decision model is asked whether the proposal needs a second look (the fixed question below, with the "
        "proposal filled in), on the same System-1 state the other cells use.",
        "3. If p_yes ≥ τ, the LLM runs **one** verification call: the same context with a check note that names its "
        "proposal. It re-reads the state and its proposal and either confirms it (the same call or action again) or "
        "revises it once. At tool choice every tool is offered; at tool arguments the call is forced to the proposed "
        "tool, so only the arguments can change; at the action point the admissible list is shown again. A reply "
        "without a call (or an empty action reply) leaves the proposal standing.",
        "4. Otherwise the proposal runs unchecked. The final proposal is executed.", "",
        "**Not a guard.** In a guard the decision model judges the proposal and vetoes it, which forces a "
        "regeneration. Here the decision model never judges or vetoes anything. It only decides when the LLM checks "
        "itself, and the LLM's verification decides the outcome.", "",
        "**Reference arm \"always verify\"** (`always` mode, no decision model): the verification call runs on every "
        "proposal. It shows how much of verification's effect the trigger keeps and at what share of its cost. The "
        "module-map cell itself compares the trigger with `llm_only`, like every other cell.", "",
        "## Questions (System-1) and verification notes (System-2)", "",
        "| point | trigger question | verification note | τ |", "|---|---|---|---|",
        f"| tool choice | {P.q_verify_tool(name, args)} | {P.verify_tool_note(name, args)} | {TAU['tool_select']} |",
        f"| tool arguments | {P.q_verify_args(name, args)} | {P.verify_args_note(name, args)} (forced to `<tool>`) | "
        f"{TAU['tool_params']} |",
        f"| action | {P.q_verify_action(act)} | Admissible actions: … then: {P.verify_action_note(act)} | {TAU['action']} |",
        "", "The argument trigger is asked about every call of a known tool that has parameters (the final `finish` "
        "call included, whose argument is the answer); the tool-choice trigger about every proposed call.", "",
        "## Thresholds", "",
        "Fixed before any episode by E6b's rule (E6b entry (c), continued by E6d and E6e): the registered τ of the same "
        "decision point's trigger or guard where one exists, else 0.9. Tool choice: **0.5**, the Stage-1 tool guard on "
        "Flash (`jev_ts_guard`). Tool arguments: **0.9**, E6b's argument guard (no Stage-1 τ at that point; E6b's "
        "else-clause gives the same 0.9). Action: **0.5**, E6c's action guard. Nothing is fitted or tuned on "
        "evaluation data.", "",
        "## Golds (log only; never on the wire)", "",
        "- **Tool choice:** the proposed tool is in the information-feasible acceptable set AT(s) of JevGym's oracle "
        "(the tool-level gold of the other tool-choice cells). Whether the whole call (tool and arguments) is "
        "progress, the tool guard's gold, is logged beside it.",
        "- **Tool arguments:** the proposed call is progress (`oracle_verdict`, the tool guard's gold).",
        "- **Action:** the proposal equals ALFWorld's single gold action, a lower bound of \"right\" (as E6c).",
        "", "Each decision logs `gold` = the proposal needed a second look (it was wrong), and in `gold_aux` whether "
        "the verification ran (`fired`), whether it changed the proposal, the proposal's and the executed value's "
        "correctness, and the outcome: `fixed` (wrong → right), `broke` (right → wrong), `confirmed_right` / "
        "`confirmed_wrong` (unchanged), `revised_right` / `revised_wrong` (changed, same correctness). Every System-1 "
        "and System-2 call goes through the episode ledger at its decision point.", ""])


def cmd_cells(a: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cells.md").write_text(cells_markdown())
    print(OUT_DIR / "cells.md")
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    res = summarize(Path(a.runs), a.n_boot, a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(MM._clean(res), indent=1, default=MM._json_default) + "\n")
    (OUT_DIR / "summary.md").write_text(to_markdown(res) + "\n")
    (OUT_DIR / "cells.md").write_text(cells_markdown())
    write_tex(to_tex_keys(res), Path(a.tex) if a.tex else None)
    print(to_markdown(res))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(RUNS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spend")
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, action="append", required=True)
    w.add_argument("--gemini-cap", type=float, default=GEMINI_CAP)
    w.add_argument("--jev-cap", type=float, default=JEV_CAP)
    w.add_argument("--every", type=float, default=30.0)
    sub.add_parser("project")
    sub.add_parser("cells")
    s = sub.add_parser("summarize")
    s.add_argument("--n-boot", type=int, default=MM.N_BOOT)
    s.add_argument("--seed", type=int, default=MM.SEED)
    s.add_argument("--tex", default=None)
    a = ap.parse_args(argv)
    return {"spend": cmd_spend, "watch": cmd_watch, "project": cmd_project, "cells": cmd_cells,
            "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
