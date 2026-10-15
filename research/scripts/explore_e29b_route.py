#!/usr/bin/env python3
"""E29b (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~19:55 UTC, "E29b: step-type routing with a write guard";
exploratory, descriptive): E29's host-side routing proxy (``jevharness.adapters.route_proxy``) with a typed step-kind
question (inspect / edit / run / finish), the fast LLM on predicted inspection steps only, and a mechanical write guard
(``route_formats.read_only_calls``) on every fast-LLM action; each harness against the unchanged harness on the same
tasks. Harnesses, LLM pairs, calibration splits and pilot tasks are E29's (``scripts/explore_e29_port.py``).

Subcommands (no model call anywhere in this script):

  lists                                            the registered full-run subset -> configs/explore/e29b_route/
  fit      --harness H --logs DIR --out gate.json  the registered fit (lowest tau with read-only precision >= 0.95)
  smoke    --harness H --logs DIR --run DIR        the registered smoke checks (exit 1 = stop)
  summary  --harness H --stage pilot|full --logs DIR --run DIR --out X.json   per-task rows joined with the proxy log
  go       --tb X.json --mswe Y.json --out go.json                             the registered go rule (exit 0 go, 3 no-go)
  spend    DIR ...                                                             logged spend of proxy log dirs (JSON)
  readout  [--box results/explore/e29b_route/box]  summary.{md,json} + paper/generated/e29b_route.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e29_port as E  # noqa: E402

CFG = ROOT / "configs/explore/e29b_route"
OUT_DIR = ROOT / "results/explore/e29b_route"
BOX = OUT_DIR / "box"
TEX = ROOT / "paper/generated/e29b_route.tex"
REG = "docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~19:55 UTC, E29b"
HARNESSES = E.HARNESSES
PRECISION = 0.95                 # registration (d)
K = 6                            # registration (b) 2
GO_SAVING = 0.20                 # registration (f)
GO_MAX_DROP = 3
N_FULL_MSWE = 100                # registration (g): the first 100 of E4r's extension list
CAPS = {"pilot": {"tb": 35.0, "mswe": 5.0}, "full": {"tb": 95.0, "mswe": 14.0, "swea": 60.0}}
ROW_CAP = 250.0
TOTAL_CAP = 300.0                # E29 + E29b
E29_SPENT = 13.3822              # results/explore/e29_port/summary.json (spend.total)
STEP_OPTIONS = ("inspect", "edit", "run", "finish")


# ------------------------------------------------------------------------------------------------ lists
def full_mswe_ids() -> list[str]:
    ids = E.read_ids(ROOT / "configs/explore/e4r_swe/extension_ids.txt")
    assert len(ids) == 200
    return ids[:N_FULL_MSWE]


def cmd_lists(a: argparse.Namespace) -> int:
    CFG.mkdir(parents=True, exist_ok=True)
    text = E.ids_text(full_mswe_ids())
    (CFG / "mswe_full.txt").write_text(text)
    meta = {"registered": REG, "rule": "mswe_full = the first 100 of configs/explore/e4r_swe/extension_ids.txt (E29's "
            "pilot = its first 30, run afresh); every other list is E29's (configs/explore/e29_port/)",
            "files": {"mswe_full.txt": {"n": N_FULL_MSWE, "sha256": E.sha256_text(text)}}}
    (CFG / "selection.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta["files"]))
    return 0


# ------------------------------------------------------------------------------------------------ fit
def typed_steps(calls: list[dict[str, Any]], decisions: list[dict[str, Any]], arm: str = "shadow"
                ) -> list[dict[str, Any]]:
    """One row per calibration step t >= 1 with a parsed strong (primary) response and a typed answer: the label
    y = the strong response is read-only, Jev's answer and c = p(inspect), the fast shadow response's verdict."""
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in calls:
        if r.get("arm") != arm or r.get("status") != 200 or r.get("key") is None:
            continue
        by[(str(r["episode"]), int(r["step"]))][str(r["role"])] = r
    dec = {(str(d["episode"]), int(d["step"])): d for d in decisions if d.get("arm") == arm}
    rows = []
    for (ep, t), roles in sorted(by.items()):
        p = roles.get("primary")
        if t < 1 or p is None or p.get("ro") is None:
            continue
        d = dec.get((ep, t))
        if d is None or d.get("c") is None or d.get("value") is None:
            continue
        f = roles.get("shadow_fast") or {}
        rows.append({"episode": ep, "step": t, "c": float(d["c"]), "value": str(d["value"]), "y": int(bool(p["ro"])),
                     "f_ro": (None if f.get("ro") is None else int(bool(f["ro"]))),
                     "cost": float(p.get("cost_usd") or 0.0),
                     "dc": (float(p.get("cost_usd") or 0.0) - float(f.get("cost_usd") or 0.0)) if f else None})
    return rows


def fit_typed(rows: list[dict[str, Any]], precision: float = PRECISION) -> dict[str, Any]:
    """Registration (d): eligible = Jev answers ``inspect``; candidates = the eligible steps' c; tau-hat = the lowest
    candidate whose precision (share of read-only strong actions among eligible steps with c >= tau) is >= 0.95;
    none -> the gate routes nothing."""
    n = len(rows)
    out: dict[str, Any] = {"n_steps": n, "n_episodes": len({r["episode"] for r in rows}), "precision_rule": precision,
                           "k": K}
    if not n:
        return {**out, "tau": None, "reason": "no calibration step"}
    elig = [r for r in rows if r["value"] == "inspect"]
    pos = sum(r["y"] for r in rows)
    tau, sel = None, []
    curve = []
    for c in sorted({r["c"] for r in elig}):
        s = [r for r in elig if r["c"] >= c]
        p = sum(r["y"] for r in s) / len(s)
        curve.append({"tau": round(c, 6), "n": len(s), "precision": round(p, 4)})
        if tau is None and p >= precision:
            tau, sel = c, s
    total_cost = sum(r["cost"] for r in rows)
    out.update({
        "tau": tau, "n_eligible": len(elig), "answers": dict(Counter(r["value"] for r in rows)),
        "base_rate": pos / n, "eligible_precision": (sum(r["y"] for r in elig) / len(elig)) if elig else None,
        "precision": (sum(r["y"] for r in sel) / len(sel)) if sel else None,
        "n_routed": len(sel), "implied_share": len(sel) / n,
        "recall": (sum(r["y"] for r in sel) / pos) if pos else None,
        "routed_cost_share": (sum(r["cost"] for r in sel) / total_cost) if total_cost > 0 else None,
        "sim_llm_saving": (sum(r["dc"] for r in sel if r["dc"] is not None) / total_cost) if total_cost > 0 else None,
        "guard_accept_sim": (_mean([float(r["f_ro"]) for r in sel if r["f_ro"] is not None])),
        "fast_ro_share_all": _mean([float(r["f_ro"]) for r in rows if r["f_ro"] is not None]),
        "c_inspect_mean": sum(r["c"] for r in rows) / n,
        "curve": curve[:: max(1, len(curve) // 60)],
        "reason": ("lowest tau with read-only precision >= %.2f" % precision) if tau is not None else
                  ("no threshold reaches read-only precision %.2f: the gate routes nothing" % precision),
    })
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def calib_descriptives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Post hoc, descriptive (the same calibration records as the fit; not part of the registered rule): how well the
    typed answer and c = p(inspect) separate read-only next steps (AUROC), the share of read-only strong actions per
    answer, and how often the fast LLM's response was read-only when the strong LLM's was (and was not)."""
    if not rows:
        return {"n": 0}
    by_ans = {}
    for v in STEP_OPTIONS:
        rs = [r for r in rows if r["value"] == v]
        by_ans[v] = {"n": len(rs), "read_only": _mean([float(r["y"]) for r in rs])}
    fr = [r for r in rows if r["f_ro"] is not None]
    tot = sum(r["cost"] for r in rows)
    return {"n": len(rows), "auroc_c_inspect": E._auroc([r["c"] for r in rows], [r["y"] for r in rows]),
            "ro_cost_share": (sum(r["cost"] for r in rows if r["y"]) / tot) if tot > 0 else None,
            "by_answer": by_ans,
            "fast_ro_when_strong_ro": _mean([float(r["f_ro"]) for r in fr if r["y"]]),
            "fast_ro_when_strong_not": _mean([float(r["f_ro"]) for r in fr if not r["y"]]),
            "note": "post hoc, descriptive; not part of the registered fit"}


def cmd_fit(a: argparse.Namespace) -> int:
    calls, decisions = E.read_logs([Path(a.logs)])
    rows = typed_steps(calls, decisions, arm=a.arm)
    res = fit_typed(rows)
    res.update({"harness": a.harness, "registered": REG, "written": E._now(), "logs": str(a.logs),
                "decisions_total": sum(1 for d in decisions if d.get("arm") == a.arm),
                "decision_errors": sum(1 for d in decisions if d.get("arm") == a.arm and d.get("c") is None),
                "shadow_calls_failed": sum(1 for c in calls if c.get("arm") == a.arm and str(c.get("role", "")).startswith(
                    "shadow") and c.get("status") != 200)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res.get(k) for k in ("tau", "precision", "n_steps", "n_episodes", "n_eligible", "n_routed",
                                              "implied_share", "base_rate", "recall", "guard_accept_sim",
                                              "sim_llm_saving", "reason")}))
    return 0


# ------------------------------------------------------------------------------------------------ smoke
def smoke_checks(harness: str, calls: list[dict[str, Any]], decisions: list[dict[str, Any]],
                 trials: Mapping[str, Mapping[str, Mapping[str, Any]]], leaks: int) -> dict[str, Any]:
    """Registration (e): no infrastructure error; no LLM error passed to the agent; Jev answered >= 95 % of the typed
    questions; >= 1 fast call judged by the guard in smoke_gated; the shadow trial logged a verdict for the primary and
    the fast shadow call; cost > 0; 0 leaks."""
    chk: dict[str, Any] = {}
    chk["1_no_infra_error"] = all(trials.get(a) for a in ("smoke_gated", "shadow")) and \
        all(not r.get("infra_error") for arm in trials.values() for r in arm.values())
    prim = [c for c in calls if c.get("role") == "primary"]
    recovered = {(c["arm"], c["episode"], c["step"]) for c in prim if c.get("status") == 200}
    chk["2_no_llm_error_to_agent"] = not [c for c in prim if c.get("status") != 200
                                          and (c["arm"], c["episode"], c["step"]) not in recovered]
    ds = [d for d in decisions if d.get("arm") in ("shadow", "smoke_gated")]
    chk["3_typed_answers"] = bool(ds) and sum(d.get("c") is not None and d.get("value") in STEP_OPTIONS
                                              for d in ds) / len(ds) >= 0.95
    g = [c for c in prim if c.get("arm") == "smoke_gated"]
    chk["4_guard_judged"] = any(c.get("guard") in ("accept", "reject") for c in g)
    sh = [c for c in calls if c.get("arm") == "shadow" and c.get("status") == 200 and c.get("ro") is not None]
    chk["5_shadow_verdicts"] = {"primary", "shadow_fast"} <= {c["role"] for c in sh}
    chk["6_cost_positive"] = sum(float(c.get("cost_usd") or 0) for c in prim) > 0
    chk["7_no_leak"] = leaks == 0
    tiers = [(c["step"], c.get("tier"), c.get("guard")) for c in sorted(g, key=lambda c: c["step"])]
    return {"harness": harness, "checks": chk, "ok": all(chk.values()), "gated_calls": tiers,
            "answers": dict(Counter(str(d.get("value")) for d in ds)), "n_calls": len(calls), "n_decisions": len(ds),
            "spend": E.spend_of(calls, decisions), "written": E._now()}


def cmd_smoke(a: argparse.Namespace) -> int:
    calls, decisions = E.read_logs([Path(a.logs)])
    run = Path(a.run)
    kind = HARNESSES[a.harness]["kind"]
    trials = {arm: (E.mswe_trials(run / arm) if kind == "mswe" else E.harbor_trials(run / arm))
              for arm in ("smoke_gated", "shadow")}
    res = smoke_checks(a.harness, calls, decisions, trials, int(a.leaks))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"ok": res["ok"], **res["checks"], "answers": res["answers"]}))
    return 0 if res["ok"] else 1


# ------------------------------------------------------------------------------------------------ summary, go
def cmd_summary(a: argparse.Namespace) -> int:
    tasks = E.read_ids(Path(a.tasks)) if a.tasks else None
    res = E.summarize(a.harness, Path(a.run), [Path(d) for d in a.logs], tasks, k=K,
                      rerun_dir=Path(a.rerun_run) if a.rerun_run else None,
                      rerun_logs=[Path(d) for d in (a.rerun_logs or [])])
    res["stage"] = a.stage
    res["registered"] = REG
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    p, g = res.get("paired") or {}, res["arms"]["gated"]
    print(json.dumps({"harness": a.harness, "stage": a.stage, "n": res["n"],
                      "missing": {k: len(v) for k, v in res["missing"].items()},
                      "solved": [p.get("solved_base"), p.get("solved_arm")], "saving": p.get("saving"),
                      "fast_step_share": g.get("fast_step_share"), "guard_reject": g.get("guard_reject"),
                      "time_saving": p.get("time_saving")}))
    return 0


def go_rule(pilots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Registration (f): pass(h) = saving >= 20 % and drop <= 3; ok(h) = drop <= 3; GO iff (pass(tb) and ok(mswe)) or
    (pass(mswe) and ok(tb)). A harness whose gate routes nothing: saving 0, drop 0; no pilot summary: pass, ok false."""
    per: dict[str, Any] = {}
    for h in ("tb", "mswe"):
        s = pilots.get(h) or {}
        p = s.get("paired") or {}
        if s.get("no_gate"):
            have, saving, drop = True, 0.0, 0
        elif p:
            have, saving = True, float(p.get("saving") or 0.0)
            drop = int(p.get("solved_base", 0)) - int(p.get("solved_arm", 0))
        else:
            have, saving, drop = False, 0.0, None
        per[h] = {"n": s.get("n", 0), "saving": saving, "drop": drop, "solved_base": p.get("solved_base"),
                  "solved_gated": p.get("solved_arm"), "no_gate": bool(s.get("no_gate")), "have_pilot": have,
                  "pass": have and saving >= GO_SAVING and drop is not None and drop <= GO_MAX_DROP,
                  "ok": have and drop is not None and drop <= GO_MAX_DROP, "status": s.get("status")}
    t, m = per["tb"], per["mswe"]
    go = (t["pass"] and m["ok"]) or (m["pass"] and t["ok"])
    return {"decision": "go" if go else "no-go", "per_harness": per,
            "rule": "GO iff (pass(tb) and ok(mswe)) or (pass(mswe) and ok(tb)); pass = saving >= 20 % and at most 3 "
                    "of 30 tasks lost; ok = at most 3 lost; a gate that routes nothing: saving 0, drop 0",
            "registered": REG, "written": E._now()}


def cmd_go(a: argparse.Namespace) -> int:
    pilots = {}
    for h, path in (("tb", a.tb), ("mswe", a.mswe)):
        if path and Path(path).exists():
            pilots[h] = json.loads(Path(path).read_text())
    res = go_rule(pilots)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if res["decision"] == "go" else 3


# ------------------------------------------------------------------------------------------------ readout
_pct, _num, _sgn = E._pct, E._num, E._sgn


def tex_values(summ: Mapping[str, Any]) -> dict[str, str]:
    """``e29b/...`` -> string, numbers only from the run records (E29's keys under the e29b prefix + guard keys)."""
    base = {k: v for k, v in summ.items() if k in ("pilot", "full", "go")}
    vals = {"e29b/" + k[len("e29/"):]: v for k, v in E.tex_values(base).items()}
    for stage in ("full", "pilot"):
        for h, s in (summ.get(stage) or {}).items():
            if not s or not s.get("paired"):
                continue
            pre = f"e29b/{h}" if stage == "full" else f"e29b/pilot/{h}"
            g = s["arms"]["gated"]
            fast_calls = int(g.get("guard_accept") or 0) + int(g.get("guard_reject") or 0)
            vals.update({f"{pre}/fast_share": _pct(g.get("fast_step_share")),
                         f"{pre}/fast_call_share": _pct(g.get("fast_share")),
                         f"{pre}/guard_accept": str(int(g.get("guard_accept") or 0)),
                         f"{pre}/guard_reject": str(int(g.get("guard_reject") or 0)),
                         f"{pre}/guard_reject_share": _pct((int(g.get("guard_reject") or 0) / fast_calls)
                                                           if fast_calls else None),
                         f"{pre}/guard_reject_cost": f"{float(g.get('guard_reject_cost_mean') or 0):.4f}",
                         f"{pre}/steps": _num((g.get("steps") or 0) / max(1, g.get("n") or 1))})
    for h, fit in (summ.get("gates") or {}).items():
        pre = f"e29b/gate/{h}"
        vals.update({f"{pre}/tau": "none" if fit.get("tau") is None else f"{fit['tau']:.3f}",
                     f"{pre}/precision": _pct(fit.get("precision")), f"{pre}/steps": str(fit.get("n_steps")),
                     f"{pre}/tasks": str(fit.get("n_episodes")), f"{pre}/eligible": str(fit.get("n_eligible")),
                     f"{pre}/routed": str(fit.get("n_routed")), f"{pre}/implied_share": _pct(fit.get("implied_share")),
                     f"{pre}/base_rate": _pct(fit.get("base_rate")), f"{pre}/recall": _pct(fit.get("recall")),
                     f"{pre}/eligible_precision": _pct(fit.get("eligible_precision")),
                     f"{pre}/guard_sim": _pct(fit.get("guard_accept_sim")),
                     f"{pre}/sim_saving": _pct(fit.get("sim_llm_saving")),
                     f"{pre}/routed_cost_share": _pct(fit.get("routed_cost_share"))})
        d = fit.get("descriptive") or {}
        if d.get("n"):
            vals.update({f"{pre}/auroc": _num(d.get("auroc_c_inspect"), 2),
                         f"{pre}/ro_cost_share": _pct(d.get("ro_cost_share")),
                         f"{pre}/fast_ro_when_ro": _pct(d.get("fast_ro_when_strong_ro")),
                         f"{pre}/fast_ro_when_not": _pct(d.get("fast_ro_when_strong_not"))})
            for v, x in (d.get("by_answer") or {}).items():
                vals[f"{pre}/ans_{v}_n"] = str(x.get("n"))
                vals[f"{pre}/ans_{v}_ro"] = _pct(x.get("read_only"))
    for k in [k for k in vals if k.startswith("e29b/go/")]:
        vals.pop(k)
    g = summ.get("go") or {}
    if g:
        vals["e29b/go/decision"] = g.get("decision", "--")
        for h, v in (g.get("per_harness") or {}).items():
            vals[f"e29b/go/{h}/saving"] = _pct(v.get("saving"))
            if v.get("drop") is not None:
                vals[f"e29b/go/{h}/drop"] = str(v.get("drop"))
    sp = summ.get("spend") or {}
    if sp:
        vals["e29b/spend/total"] = _num(sp.get("total"), 2)
        vals["e29b/spend/cap"] = _num(ROW_CAP, 0)
        vals["e29b/spend/with_e29"] = _num(float(sp.get("total") or 0) + E29_SPENT, 2)
    lk = summ.get("leaks") or {}
    if lk:
        vals["e29b/leaks/max_files"] = str(max(int(v.get("max_files") or 0) for v in lk.values()))
        vals["e29b/leaks/checks"] = str(sum(int(v.get("checks") or 0) for v in lk.values()))
    return vals


def write_tex(vals: Mapping[str, str], path: Path = TEX) -> None:
    lines = ["% AUTO-GENERATED by scripts/explore_e29b_route.py readout -- do not edit by hand.",
             f"% generated {E._now()} at git {E._git()}; exploratory (E29b), descriptive ({REG}); outside every Holm table",
             "\\makeatletter",
             "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{vals[k]}}}" for k in sorted(vals)]
    lines.append("\\makeatother")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def to_markdown(summ: Mapping[str, Any]) -> str:
    out = ["# E29b: step-type routing with a write guard (exploratory, descriptive)", "",
           f"Registered in {REG}, after E29's pilot was read and before any E29b call. E29's host-side proxy asks the "
           "decision model which kind of step comes next (inspect / edit / run / finish) while the previous action "
           "runs; the fast LLM writes a step only when the answer is inspect with c = p(inspect) >= tau (sticky, at "
           f"most k = {K} fast steps in a row); the proxy executes a fast-LLM action only if a fixed classifier finds it "
           "read-only, else discards it and the strong LLM writes the step. Cost = proxy-logged LLM (discarded calls "
           "included) + decision-model $ per task; time = agent execution seconds per task.", ""]
    if summ.get("gates"):
        out += ["## Fitted gates (typed shadow calibration)", "",
                "| harness | tau | precision % | steps / tasks | read-only base rate % | Jev says inspect (n) | routed "
                "(n) | implied routed share % | recall % | guard would accept % | routed cost share % | simulated LLM "
                "saving % |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for h, f in summ["gates"].items():
            out.append(f"| {h} | {'none' if f.get('tau') is None else round(f['tau'], 4)} | {_pct(f.get('precision'))} "
                       f"| {f.get('n_steps')} / {f.get('n_episodes')} | {_pct(f.get('base_rate'))} | "
                       f"{f.get('n_eligible')} | {f.get('n_routed')} | {_pct(f.get('implied_share'))} | "
                       f"{_pct(f.get('recall'))} | {_pct(f.get('guard_accept_sim'))} | "
                       f"{_pct(f.get('routed_cost_share'))} | {_pct(f.get('sim_llm_saving'))} |")
        out += ["", "Answers: " + "; ".join(f"{h}: {json.dumps(f.get('answers'))}" for h, f in summ["gates"].items()),
                ""]
        out += ["Post hoc, descriptive (the same calibration records; not part of the registered fit):", ""]
        for h, f in summ["gates"].items():
            d = f.get("descriptive") or {}
            if not d.get("n"):
                continue
            out.append(f"- {h}: AUROC of p(inspect) for a read-only next step {_num(d.get('auroc_c_inspect'), 2)}; "
                       f"read-only steps carried {_pct(d.get('ro_cost_share'))} % of the strong LLM's cost; "
                       "read-only share of the strong LLM's next step by Jev's answer: " + "; ".join(
                           f"{v} n {x['n']}: {_pct(x['read_only'])} %" for v, x in d["by_answer"].items()) +
                       f"; the fast LLM's response was read-only in {_pct(d.get('fast_ro_when_strong_ro'))} % of the "
                       f"steps where the strong LLM's was and in {_pct(d.get('fast_ro_when_strong_not'))} % where it "
                       "was not.")
        out.append("")
    for stage in ("pilot", "full"):
        st = summ.get(stage) or {}
        if not st:
            continue
        out += [f"## {'Pilot (30 tasks per harness; decided the go)' if stage == 'pilot' else 'Full run'}", "",
                "| harness | n | success base / gated % | delta pt [95 %] | W / L | NI p (3 pt) | cost $/task base / "
                "gated | saving % [95 %] | mean s base / gated | time saving % | median-time saving % | fast steps % | "
                "guard rejections |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for h, s in st.items():
            if not s or not s.get("paired"):
                out.append(f"| {HARNESSES[h]['label']} | -- | {s.get('status', 'not run') if s else 'not run'} "
                           "| | | | | | | | | | |")
                continue
            p, g = s["paired"], s["arms"]["gated"]
            fc = int(g.get("guard_accept") or 0) + int(g.get("guard_reject") or 0)
            out.append(f"| {HARNESSES[h]['label']} | {p['n']} | {_pct(p['success_base'])} / {_pct(p['success_arm'])} | "
                       f"{_sgn(p['delta_pt'])} [{_sgn(p['delta_ci'][0])}, {_sgn(p['delta_ci'][1])}] | "
                       f"{p['wins']} / {p['losses']} | {p['ni_p']:.3f} ({'shown' if p['ni'] else 'not shown'}) | "
                       f"{p['cost_base']:.4f} / {p['cost_arm']:.4f} | {_pct(p['saving'])} [{_pct(p['saving_ci'][0])}, "
                       f"{_pct(p['saving_ci'][1])}] | {_num(p.get('time_base'), 0)} / {_num(p.get('time_arm'), 0)} | "
                       f"{_pct(p.get('time_saving'))} | {_pct(p.get('median_time_saving'))} | "
                       f"{_pct(g.get('fast_step_share'))} | {g.get('guard_reject')} of {fc} fast calls |")
        out.append("")
        for h, s in st.items():
            if not s or not s.get("paired"):
                continue
            g, b = s["arms"]["gated"], s["arms"]["baseline"]
            cc = s.get("paired_cc") or {}
            out.append(f"- {HARNESSES[h]['label']}: gated arm {g['requests_mean']:.1f} LLM requests per task (baseline "
                       f"{b['requests_mean']:.1f}), {g.get('steps')} executed steps of which {g.get('fast_steps')} by "
                       f"the fast LLM; {_num(g.get('segments_per_task'))} fast segments per task of mean length "
                       f"{_num(g.get('segment_len_mean'), 2)} ({_pct(g.get('segments_capped_share'))} % ended by k = "
                       f"{K}); guard: {g.get('guard_accept')} accepted, {g.get('guard_reject')} rejected (discarded "
                       f"${float(g.get('guard_reject_cost_mean') or 0):.4f} per task); decision model "
                       f"{_num(g.get('decisions_per_task'))} calls, ${float(g.get('dm_cost_mean') or 0):.4f} and "
                       f"{_num(g.get('dm_latency_per_task_s'))} s per task, of which requests waited "
                       f"{_num(g.get('waited_per_task_s'), 2)} s; fail-open steps {g.get('fail_open')}, decision errors "
                       f"{g.get('decision_errors')}; route reasons {json.dumps(g.get('reasons'))}; cached input share "
                       f"{_pct(b.get('cached_share'))} % (baseline) vs {_pct(g.get('cached_share'))} % (gated); "
                       f"infrastructure errors {b['infra_errors']} / {g['infra_errors']}; complete case n {cc.get('n')}, "
                       f"delta {_sgn(cc.get('delta_pt'))}, NI p {_num(cc.get('ni_p'), 3)}. Missing: "
                       f"{json.dumps({k: len(v) for k, v in s['missing'].items()})}.")
        out.append("")
    if summ.get("go"):
        out += ["## Go decision (registered rule, pilot only)", "", "```", json.dumps(summ["go"], indent=1), "```", ""]
    for k in ("spend", "leaks", "hosts", "notes"):
        if summ.get(k):
            out += [f"## {k.capitalize()}", "", "```", json.dumps(summ[k], indent=1, default=str), "```", ""]
    return "\n".join(out)


def cmd_readout(a: argparse.Namespace) -> int:
    box = Path(a.box)
    summ: dict[str, Any] = {"registered": REG, "code": E._git(), "written": E._now(), "pilot": {}, "full": {},
                            "gates": {}}
    for stage in ("pilot", "full"):
        for h in HARNESSES:
            f = box / "summaries" / f"{stage}_{h}.json"
            if f.exists():
                summ[stage][h] = json.loads(f.read_text())
    for h in HARNESSES:
        f = box / "gates" / f"gate_{h}.json"
        if f.exists():
            summ["gates"][h] = {k: v for k, v in json.loads(f.read_text()).items() if k != "curve"}
            cal = box / h / "proxy" / "cal"
            if cal.is_dir():      # post hoc, descriptive: the calibration records the fit read (written before it)
                calls, decs = E.read_logs([cal])
                cut = E._ts(summ["gates"][h].get("written"))
                if cut is not None:
                    calls = [r for r in calls if (E._ts(r.get("ts")) or cut) <= cut]
                    decs = [r for r in decs if (E._ts(r.get("ts")) or cut) <= cut]
                summ["gates"][h]["descriptive"] = calib_descriptives(typed_steps(calls, decs))
    for name in ("go", "hosts", "notes"):
        f = box / f"{name}.json"
        if f.exists():
            summ[name] = json.loads(f.read_text())
    spend: dict[str, Any] = {}
    leaks: dict[str, Any] = {}
    for h in HARNESSES:
        st = box / h / "status"
        f = st / "e29_spend.json"
        if f.exists():
            spend[h] = json.loads(f.read_text())
        lk = E.jsonl(st / "e29_leaks.jsonl")
        if lk:
            leaks[h] = {"checks": len(lk), "max_files": max(int(x.get("files") or 0) for x in lk),
                        "stages": [x.get("stage") for x in lk]}
    if spend:
        tot = sum(float(((v.get("logged") or v) or {}).get("total") or 0.0) for v in spend.values())
        spend["total"] = round(tot, 4)
        spend["cap"] = ROW_CAP
        spend["e29_spent"] = E29_SPENT
        spend["e29_plus_e29b"] = round(tot + E29_SPENT, 4)
        spend["total_cap"] = TOTAL_CAP
        summ["spend"] = spend
    if leaks:
        summ["leaks"] = leaks
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(summ, default=str))
    for stage in ("pilot", "full"):
        for s in slim[stage].values():
            s.pop("per_task", None)
    (OUT_DIR / "summary.json").write_text(json.dumps(slim, indent=1) + "\n")
    (OUT_DIR / "summary.md").write_text(to_markdown(summ) + "\n")
    write_tex(tex_values(summ), Path(a.tex) if a.tex else TEX)
    print(f"wrote {OUT_DIR / 'summary.md'}, {OUT_DIR / 'summary.json'}, {a.tex or TEX}")
    return 0


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lists")
    p = sub.add_parser("spend")
    p.add_argument("dirs", nargs="+")
    p = sub.add_parser("fit")
    p.add_argument("--harness", required=True, choices=sorted(HARNESSES))
    p.add_argument("--logs", required=True)
    p.add_argument("--arm", default="shadow")
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--harness", required=True, choices=sorted(HARNESSES))
    p.add_argument("--logs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--leaks", default="0")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--harness", required=True, choices=sorted(HARNESSES))
    p.add_argument("--stage", required=True, choices=("pilot", "full"))
    p.add_argument("--run", required=True)
    p.add_argument("--logs", required=True, nargs="+")
    p.add_argument("--tasks", default=None)
    p.add_argument("--rerun-run", default=None)
    p.add_argument("--rerun-logs", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("go")
    p.add_argument("--tb", default=None)
    p.add_argument("--mswe", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("readout")
    p.add_argument("--box", default=str(BOX))
    p.add_argument("--tex", default=None)
    a = ap.parse_args(argv)
    return {"lists": cmd_lists, "spend": E.cmd_spend, "fit": cmd_fit, "smoke": cmd_smoke, "summary": cmd_summary,
            "go": cmd_go, "readout": cmd_readout}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
