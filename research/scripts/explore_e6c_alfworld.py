#!/usr/bin/env python3
r"""E6c (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry
"Exploratory row E6c" of 2026-10-06): the action GUARD on ALFWorld valid_seen, the last meaningful cell of the module
map. Arms: LLM only and the decision model (Jev via OpenRouter) as the action guard (tau = 0.5, the fixed tau of the
Stage-1 tool guard on Flash): the LLM proposes an admissible action, Jev vetoes it when p_yes < tau, the LLM
regenerates once and the rewrite is executed unchecked. Games, order, models and settings as E6a
(scripts/explore_e6a_alfworld.py, whose spend / readout helpers this script reuses). Not the registered S2-L3b.

    python scripts/explore_e6c_alfworld.py spend                           # logged $ of every E6c run
    python scripts/explore_e6c_alfworld.py watch --cap 25 --pid PID        # watchdog: kills PID's group at the cap
    python scripts/explore_e6c_alfworld.py project                         # smoke -> eval cost projection
    python scripts/explore_e6c_alfworld.py summarize                       # results/explore/e6c_alfworld + e6c.tex

Readout: paired complete case over the eval list (a game whose latest record errored, or that is missing, in either
arm leaves that pair). Success and Delta vs LLM only (points; paired task bootstrap B = 10,000, seed 0, and the
Newcombe interval), paired wins / losses, cost per episode ($, Gemini + Jev at list prices) and saving
1 - C_arm / C_ref (task bootstrap), mean / median wall time per episode, steps and calls per episode, and the guard's
coverage, which for a guard is the module map's activity: the share of proposals vetoed (the share of decisions executed
from the model's answer is 0 by definition). Against the single gold action (lower bounds: many states admit several
good actions): the guard's verdict accuracy, the LLM's proposal accuracy, the vetoed proposals' rewrite accuracy. The
module map's fixed descriptive verdict (scripts/module_map.py ``verdict``). No test; outside every Holm table."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402
import explore_e6a_alfworld as E6A  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "results/explore/e6c_alfworld"
TEX = ROOT / "paper/generated/e6c.tex"
EVAL_LIST = E6A.EVAL_LIST                                # the same 140 valid_seen games, same order
E6A_SUMMARY = E6A.OUT_DIR / "summary.json"
RUN_SMOKE, RUN_EVAL = "e6c_alfworld_smoke", "e6c_alfworld_eval"
REF, ARM = "llm_only", "jev_action_guard"
TAU = 0.5
LABEL = "exploratory (E6c)"
CAP_USD = 25.0
SAFETY = 1.5
MIN_EVAL_GAMES = 100


# ============================================================================== spend, watchdog, projection
def total_spend(runs: Path = RUNS) -> dict[str, Any]:
    """Every record of every E6c run (errored attempts and re-runs included: all of them were paid)."""
    per = {name: E6A.run_spend(runs / name) for name in (RUN_SMOKE, RUN_EVAL) if (runs / name).is_dir()}
    tot = {k: sum(v[k] for v in per.values()) for k in ("gemini", "jev", "total", "records")}
    return {"by_run": per, **tot}


def cmd_spend(a: argparse.Namespace) -> int:
    print(json.dumps(total_spend(Path(a.runs)), indent=1))
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    """Poll the logged spend every ``--every`` s; at the cap send SIGTERM to the runner's process group (then SIGKILL)
    and write ``<runs>/e6c_STOP.json``. Episodes in flight when it fires are not logged (a few cents)."""
    stop = Path(a.runs) / "e6c_STOP.json"
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(Path(a.runs))
        if s["total"] >= a.cap:
            stop.write_text(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                        "spend": s, "cap": a.cap}, indent=1))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(a.pid), sig)
                except OSError:
                    pass
                time.sleep(10)
            print(f"CAP REACHED: ${s['total']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} spend ${s['total']:.3f} (Gemini ${s['gemini']:.3f}, "
              f"Jev ${s['jev']:.4f}; {s['records']} records)", flush=True)
        time.sleep(a.every)


def projection(runs: Path = RUNS, n_games: int = 140, safety: float = SAFETY, cap: float = CAP_USD) -> dict[str, Any]:
    """The registered gate: smoke mean $ per episode per arm x the eval games x ``safety``, plus the spend so far,
    against the cap; if it does not fit, the first ``MIN_EVAL_GAMES`` games of the eval list are run (still checked)."""
    sm = {arm: list(E6A._latest(runs / RUN_SMOKE / f"{arm}.jsonl").values()) for arm in (REF, ARM)}
    per = {arm: (float(np.mean([E6A.episode_spend(t)["total"] for t in ts])) if ts else None) for arm, ts in sm.items()}
    wall = {arm: (float(np.mean([float(t.get("wall_ms") or 0) for t in ts])) / 1000 if ts else None)
            for arm, ts in sm.items()}
    so_far = total_spend(runs)["total"]
    ep = sum(v for v in per.values() if v is not None)
    out = {"smoke_per_episode": per, "smoke_wall_s": wall, "smoke_n": {k: len(v) for k, v in sm.items()},
           "smoke_errored": {k: sum(bool(t.get("error")) for t in v) for k, v in sm.items()},
           "spent_so_far": so_far, "safety": safety, "cap": cap}
    for n in (n_games, MIN_EVAL_GAMES):
        out[f"projected_{n}"] = ep * n * safety
        out[f"fits_{n}"] = so_far + ep * n * safety <= cap
    out["decision"] = (f"run all {n_games}" if out[f"fits_{n_games}"] else
                       f"run the first {MIN_EVAL_GAMES}" if out[f"fits_{MIN_EVAL_GAMES}"] else "stop after the smoke")
    return out


def cmd_project(a: argparse.Namespace) -> int:
    p = projection(Path(a.runs), a.n_games)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "projection.json").write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(p, indent=1))
    return 0


# ============================================================================== readout
def _guard_recs(t: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [d for st in t.get("steps") or [] for d in st.get("decisions") or []
            if d.get("dp") == "action" and d.get("mode") == "guard" and not d.get("shadow")]


def _mean(xs: Sequence[Any]) -> float | None:
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs else None


def guard_stats(trajs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The guard's decisions: proposals checked, vetoed (the module map's activity = this readout's coverage), the
    share executed from the model's answer (0 by definition), and accuracies against the single gold action (lower
    bounds): the verdict (yes = the proposal is the gold action), the LLM's proposals, the executed actions (accepted
    proposals and rewrites), the rewrites of vetoed proposals; vetoes per episode and how often the rewrite repeated the
    vetoed action."""
    from jevharness.bench import metrics as M
    recs = [d for t in trajs for d in _guard_recs(t)]
    out: dict[str, Any] = {"n_decisions": len(recs)}
    if not recs:
        return out
    veto = [d for d in recs if d.get("s1_value") is False]
    g = [d for d in recs if isinstance(d.get("gold"), bool)]
    gv = [d for d in veto if isinstance(d.get("gold"), bool)]
    rw = [d for d in veto if isinstance((d.get("gold_aux") or {}).get("rewrite_gold"), bool)]
    wrong = [d for d in g if d["gold"] is False]
    executed = [d["gold"] for d in g if d.get("s1_value") is True] + [d["gold_aux"]["rewrite_gold"] for d in rw]
    out.update({
        "coverage": len(veto) / len(recs), "vetoed": len(veto) / len(recs), "n_vetoed": len(veto),
        "executed_from_model": M.delegation_ratio(recs, dps=("action",))["ratio"],
        "vetoes_per_episode": len(veto) / len(trajs), "n_gold": len(g),
        "acc": _mean([d["s1_value"] is d["gold"] for d in g]),
        "proposal_acc": _mean([d["gold"] for d in g]),
        "veto_precision": _mean([d["gold"] is False for d in gv]),    # vetoed proposals that were not the gold action
        "veto_recall": _mean([d.get("s1_value") is False for d in wrong]),
        "rewrite_acc": _mean([d["gold_aux"]["rewrite_gold"] for d in rw]),
        "rewrite_same": _mean([bool((d.get("gold_aux") or {}).get("rewrite_same")) for d in veto]),
        "executed_acc": _mean(executed),
        "mean_p_yes": _mean([(d.get("s1_dist") or {}).get("yes") for d in recs
                             if (d.get("s1_dist") or {}).get("yes") is not None])})
    return out


def e6a_reference(ref_t: Mapping[str, Mapping[str, Any]], ids: Sequence[str], runs: Path) -> dict[str, Any]:
    """Descriptive: E6a's ``llm_only`` on the same games (another run of the same configuration at temperature 0), its
    success and the per-game agreement with this run's reference where E6a's run file is available. Not paired."""
    out: dict[str, Any] = {}
    if E6A_SUMMARY.exists():
        s = json.loads(E6A_SUMMARY.read_text()).get("reference") or {}
        out.update({"success": s.get("success"), "n": s.get("n"), "cost_usd": s.get("cost_usd")})
    p = runs / E6A.RUN_EVAL / f"{REF}.jsonl"
    if p.exists():
        a = E6A._latest(p)
        keys = [t for t in ids if t in a and t in ref_t and not a[t].get("error") and not ref_t[t].get("error")]
        if keys:
            ya, yb = [C.success(a[t]) for t in keys], [C.success(ref_t[t]) for t in keys]
            out.update({"n_paired": len(keys), "agree": int(sum(x == y for x, y in zip(ya, yb))),
                        "e6a_only": int(sum(x and not y for x, y in zip(ya, yb))),
                        "e6c_only": int(sum(y and not x for x, y in zip(ya, yb)))})
    return out


def summarize(runs: Path = RUNS) -> dict[str, Any]:
    import module_map as MM
    ev = json.loads(EVAL_LIST.read_text())
    run = runs / RUN_EVAL
    exp_n = None
    if (run / "experiment.yaml").exists():
        import yaml
        exp = yaml.safe_load((run / "experiment.yaml").read_text())
        exp_n = int((exp.get("suite") or {}).get("n") or ev["n"])
    ids = ev["task_ids"][: exp_n or ev["n"]]
    tt = {name: E6A._latest(run / f"{name}.jsonl") for name in (REF, ARM)}
    res: dict[str, Any] = {
        "row": LABEL, "label": "descriptive, revises no registered result, not the registered S2-L3b",
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": C.git_head(),
        "eval": {"split": "valid_seen", "n_games": len(ids), "list_sha256": E6A._sha("\n".join(ids)),
                 "run": C.rel(run), "files": {n: {"sha256": C.sha256_file(run / f"{n}.jsonl"),
                                                  "records": sum(1 for _ in open(run / f"{n}.jsonl"))}
                                              for n in tt if (run / f"{n}.jsonl").exists()}},
        "settings": {"agent_llm": "gemini-3.5-flash (temperature 0, reasoning_effort minimal)",
                     "decision_model": "Jev via OpenRouter (typesafe/jev-1.13)", "max_steps": 50,
                     "history_window": 12, "tau": TAU, "guard_max_retries": 1,
                     "tau_rule": "fixed: the Stage-1 tool guard's tau on Flash (jev_ts_guard, E2a)",
                     "schedule": "task_major, seed 732"},
        "reference": E6A.ref_block(tt[REF], ids)}
    if tt[ARM]:
        a = {"arm": ARM, "status": "run", **E6A.arm_block(tt[ARM], tt[REF], ids, "guard")}
        if a.get("n"):
            keys = [t for t in ids if t in tt[ARM] and t in tt[REF] and not tt[ARM][t].get("error")
                    and not tt[REF][t].get("error")]
            a["decisions"] = guard_stats([tt[ARM][t] for t in keys])
            a["verdict"] = MM.verdict(a["delta_pt"] / 100, a["saving"], a["decisions"].get("vetoed", float("nan")))
        res["arm"] = a
    else:
        res["arm"] = {"arm": ARM, "status": "not run"}
    res["e6a_llm_only"] = e6a_reference(tt[REF], ids, runs)
    res["by_task_type"] = E6A.by_type({"llm_only": tt[REF], "guard": tt[ARM]}, ids)
    res["spend"] = total_spend(runs)
    return res


_f = E6A._f


def _pc(x: Any, d: int = 1) -> str:
    return _f(100 * x if x is not None else None, d)


def to_markdown(r: Mapping[str, Any]) -> str:
    ref, a, s = r["reference"], r["arm"], r["settings"]
    L = [f"# E6c: the action guard on ALFWorld valid_seen ({r['row']})", "",
         f"- **{r['label']}.** Outside every Holm table; no test. Generated {r['generated']} at git "
         f"`{(r['git'] or '?')[:12]}` by `scripts/explore_e6c_alfworld.py`.",
         f"- Games: {r['eval']['n_games']} valid_seen games (list sha256 `{r['eval']['list_sha256'][:16]}`, E6a's list "
         f"and order); run `{r['eval']['run']}`. Agent LLM {s['agent_llm']}; decision model {s['decision_model']}; max "
         f"{s['max_steps']} steps; {s['schedule']}.",
         f"- GUARD: the LLM proposes an admissible action; Jev vetoes it when p_yes < tau = {s['tau']} ({s['tau_rule']}); "
         "the LLM regenerates once with the rejection note and the rewrite is executed unchecked.",
         f"- LLM only: success {_pc(ref.get('success'))} % (n {ref['n']}, errored {ref['n_errored']}), "
         f"${_f(ref.get('cost_usd'), 4)} per episode, wall time mean / median {_f(ref.get('wall_mean_s'))} / "
         f"{_f(ref.get('wall_median_s'))} s, {_f(ref.get('steps'))} steps; its executed action matched the single gold "
         f"action at {_pc((ref.get('decisions') or {}).get('llm_acc'))} % of its steps (a lower bound).", ""]
    if a.get("status") != "run" or not a.get("n"):
        L += [f"`{ARM}`: not run.", ""]
    else:
        d = a["decisions"]
        L += ["| arm | n | success % | Δ pt vs LLM only [95 % bootstrap] | Newcombe | wins / losses | $ / episode "
              "(Gemini + Jev) | saving % [95 %] | wall s mean / median | steps | LLM calls | DM calls | coverage "
              "(vetoed) % | verdict acc. % | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
              f"| guard (`{ARM}`, tau {s['tau']}) | {a['n']} | {_pc(a['success'])} | {_f(a['delta_pt'], sign=True)} "
              f"[{_f(a['delta_ci_pt'][0], sign=True)}, {_f(a['delta_ci_pt'][1], sign=True)}] | "
              f"[{_f(a['delta_newcombe_pt'][0], sign=True)}, {_f(a['delta_newcombe_pt'][1], sign=True)}] | "
              f"{a['wins']} / {a['losses']} | {_f(a['cost_usd'], 4)} ({_f(a['gemini_usd'], 4)} + {_f(a['jev_usd'], 4)}) | "
              f"{_pc(a['saving'])} [{_pc(a['saving_ci'][0])}, {_pc(a['saving_ci'][1])}] | "
              f"{_f(a['wall_mean_s'])} / {_f(a['wall_median_s'])} | {_f(a['steps'])} | {_f(a['llm_calls'])} | "
              f"{_f(a['s1_calls'])} | {_pc(d.get('vetoed'))} | {_pc(d.get('acc'))} | {a.get('verdict') or '--'} |", "",
              f"Reference on the same pairs: ${_f(a.get('cost_ref_usd'), 4)} per episode, wall time mean / median "
              f"{_f(a.get('wall_ref_mean_s'))} / {_f(a.get('wall_ref_median_s'))} s; paired median wall-time difference "
              f"{_f(a.get('wall_delta_median_s'), sign=True)} s.",
              f"Guard decisions: {d['n_decisions']} proposals checked, {d.get('n_vetoed')} vetoed "
              f"({_f(d.get('vetoes_per_episode'), 2)} per episode); executed from the model's answer: "
              f"{_pc(d.get('executed_from_model'))} % (0 by definition). Against the single gold action (lower bounds; "
              f"{d.get('n_gold')} proposals with a gold action): verdict accuracy {_pc(d.get('acc'))} %, the LLM's "
              f"proposals {_pc(d.get('proposal_acc'))} %, vetoed proposals that were not the gold action "
              f"{_pc(d.get('veto_precision'))} %, non-gold proposals vetoed {_pc(d.get('veto_recall'))} %, rewrites of "
              f"vetoed proposals {_pc(d.get('rewrite_acc'))} % (the rewrite repeated the vetoed action "
              f"{_pc(d.get('rewrite_same'))} %), executed actions {_pc(d.get('executed_acc'))} %; mean p_yes "
              f"{_f(d.get('mean_p_yes'), 3)}.",
              "Verdict: the module map's fixed descriptive rule (scripts/module_map.py `verdict`; activity = proposals "
              "vetoed).", ""]
    x = r.get("e6a_llm_only") or {}
    if x:
        L += [f"E6a's `llm_only` on the same games (another run of the same configuration; descriptive, not paired): "
              f"success {_pc(x.get('success'))} % (n {x.get('n')})"
              + (f"; per game, {x['agree']} of {x['n_paired']} outcomes agree with this run's reference (E6a only "
                 f"{x['e6a_only']}, E6c only {x['e6c_only']})" if x.get("n_paired") else "") + ".", ""]
    L += ["## Success by task type (descriptive, small n)", "", "| task type | n | LLM only | guard |", "|---|---|---|---|"]
    for typ, v in r["by_task_type"].items():
        L.append(f"| {typ} | {v['n']} | " + " | ".join(_f(100 * v[k] if v.get(k) is not None else None, 0)
                                                     for k in ("llm_only", "guard")) + " |")
    sp = r["spend"]
    L += ["", "## Spend (logged, list prices)", "",
          f"Total ${sp['total']:.2f} (Gemini ${sp['gemini']:.2f}, Jev ${sp['jev']:.3f}; {sp['records']} records incl. "
          "errored attempts) over " + ", ".join(f"`{k}` ${v['total']:.2f}" for k, v in sp["by_run"].items())
          + f"; cap ${CAP_USD:.0f}.", ""]
    return "\n".join(L)


def to_tex(r: Mapping[str, Any]) -> str:
    v: dict[str, str] = {}

    def put(k: str, x: Any, d: int = 1, sign: bool = False, scale: float = 1.0) -> None:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return
        y = x * scale
        v[k] = (f"{y:+.{d}f}" if sign else f"{y:.{d}f}")
    v["e6c/n_games"] = str(r["eval"]["n_games"])
    v["e6c/label"] = LABEL
    put("e6c/tau", r["settings"]["tau"], 1)
    ref = r["reference"]
    v["e6c/llm_only/n"] = str(ref["n"])
    put("e6c/llm_only/success", ref.get("success"), 1, scale=100)
    put("e6c/llm_only/cost_usd", ref.get("cost_usd"), 4)
    put("e6c/llm_only/cost_k", ref.get("cost_usd"), 1, scale=1000)
    put("e6c/llm_only/wall_mean", ref.get("wall_mean_s"), 1)
    put("e6c/llm_only/wall_median", ref.get("wall_median_s"), 1)
    put("e6c/llm_only/steps", ref.get("steps"), 1)
    put("e6c/llm_only/llm_acc", (ref.get("decisions") or {}).get("llm_acc"), 1, scale=100)
    a, p = r["arm"], "e6c/guard"
    if a.get("status") != "run" or not a.get("n"):
        v[f"{p}/status"] = "not run"
    else:
        v[f"{p}/status"] = "run"
        v[f"{p}/n"], v[f"{p}/n_dropped"] = str(a["n"]), str(a["n_dropped"])
        put(f"{p}/success", a["success"], 1, scale=100)
        put(f"{p}/delta", a["delta_pt"], 1, sign=True)
        put(f"{p}/delta_ci/lo", a["delta_ci_pt"][0], 1, sign=True)
        put(f"{p}/delta_ci/hi", a["delta_ci_pt"][1], 1, sign=True)
        v[f"{p}/wins"], v[f"{p}/losses"] = str(a["wins"]), str(a["losses"])
        put(f"{p}/cost_usd", a["cost_usd"], 4)
        put(f"{p}/cost_k", a["cost_usd"], 1, scale=1000)
        put(f"{p}/saving", a["saving"], 1, scale=100)
        put(f"{p}/saving_ci/lo", a["saving_ci"][0], 1, scale=100)
        put(f"{p}/saving_ci/hi", a["saving_ci"][1], 1, scale=100)
        put(f"{p}/wall_mean", a["wall_mean_s"], 1)
        put(f"{p}/wall_median", a["wall_median_s"], 1)
        put(f"{p}/steps", a["steps"], 1)
        put(f"{p}/llm_calls", a["llm_calls"], 1)
        put(f"{p}/s1_calls", a["s1_calls"], 1)
        d = a["decisions"]
        for k in ("vetoed", "acc", "proposal_acc", "veto_precision", "veto_recall", "rewrite_acc", "rewrite_same",
                  "executed_acc"):
            put(f"{p}/{k}", d.get(k), 1, scale=100)
        put(f"{p}/coverage", d.get("vetoed"), 1, scale=100)
        put(f"{p}/vetoes_per_episode", d.get("vetoes_per_episode"), 2)
        if a.get("verdict"):
            v[f"{p}/verdict"] = a["verdict"]
    sp = r["spend"]
    put("e6c/spend/total", sp["total"], 2)
    put("e6c/spend/gemini", sp["gemini"], 2)
    put("e6c/spend/jev", sp["jev"], 3)
    lines = ["% AUTO-GENERATED by scripts/explore_e6c_alfworld.py -- do not edit by hand.",
             "% E6c: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06); not S2-L3b.",
             f"% generated {r['generated']} at git {(r['git'] or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(v.items())]
    return "\n".join(lines + [r"\makeatother", ""])


def cmd_summarize(a: argparse.Namespace) -> int:
    r = summarize(Path(a.runs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (OUT_DIR / "summary.md").write_text(to_markdown(r) + "\n")
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(to_tex(r))
    print(to_markdown(r))
    print(f"wrote {C.rel(OUT_DIR / 'summary.json')}, {C.rel(OUT_DIR / 'summary.md')}, {C.rel(TEX)}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("spend", "project", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("--runs", default=str(RUNS))
        if name == "project":
            s.add_argument("--n-games", type=int, default=140)
    s = sub.add_parser("watch")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("--cap", type=float, default=CAP_USD)
    s.add_argument("--every", type=float, default=30.0)
    a = ap.parse_args(argv)
    return {"spend": cmd_spend, "watch": cmd_watch, "project": cmd_project, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
