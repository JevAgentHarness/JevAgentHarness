"""Summarise Harbor job directories of pi (+ Jev extensions) runs: one row per trial, per-arm statistics, and
paired per-task comparisons against a baseline arm.

    python adapters/pi/summarize_harbor.py runs/tb2/baseline runs/tb2/jev_gate_stop [--json out.json]
    python adapters/pi/summarize_harbor.py runs/tbsci/* --baseline baseline --budgets 900,1800,3600

Per trial: reward, tests passed / total (the verifier's ``ctrf.json``), LLM tokens and cost (Harbor's
agent_result, from pi's usage), agent steps and tool calls (ATIF ``agent/trajectory.json``), wall-clock of the
phases (environment setup, agent setup = pi install, agent run, verifier) and of the whole trial, whether the agent
hit its timeout, and from ``agent/jev-decisions.jsonl``: S1 requests, gate blocks, stop nudges, reflections, S1
latency and input tokens. The S1 cost is NOT in pi's cost; with ``--s1-price-per-m`` (USD per 1M input tokens of
the decision model) it is estimated from the logged tokens (deduplicated by ``request_id``).

Per arm (``arm_stats``): success rate, p50 / p90 of the agent-run and trial wall-clock (all trials; a timed-out
trial counts at its timeout), mean cost and steps, cost per solved trial, and success within each ``--budgets``
agent-run time ("solved by t", the faster-and-better curve).

Paired (``--baseline <job name>``, ``paired``): per task, attempts are averaged first; then for each other arm on
the tasks both arms ran: mean success difference, task wins / losses (higher / lower per-task success) with an
exact two-sided sign test, the median per-task difference in agent-run time and in cost (arm - baseline), and the
median time difference on tasks both arms solved in every attempt.

H5 verdict (pre-registration A5.2 / A6.6 / A6.16 (j), ``jevharness.bench.metrics.h5_verdict``; printed with
``--baseline``; exploratory, outside the Holm family): *success-preserving* = the exploratory success-preservation
screen (one-sided 90% task-bootstrap lower bound of the paired success difference > -5 pt and wins >= losses; not a
size-controlled NI test); *strictly better* = that bound > 0 and exact one-sided sign test p < 0.05; *faster* = on tasks
both arms solve at least once (each task timed by the mean agent-run time of its solved attempts), median time
difference < 0 with the exact one-sided sign test on the per-task differences p < 0.05 (the Wilcoxon p is printed as
descriptive). Displayed verdicts (A6.22 (d) 1: only the labels changed): *faster and better (faster among jointly
solved tasks)* (strictly better + faster), *passes the exploratory success screen; faster among jointly solved tasks*
(the registered verdict, ``success_preserving_and_faster``), *success-preserving*, *faster*, *neither*. The all-tasks
median (timeouts at the cap) and solved-by-t at 600 / 1800 / 3600 s are printed on the same row, descriptive only.

Errored trials (A6.22 (e) 1, 5): a Harbor trial exception other than ``AgentTimeoutError`` is an infrastructure error
(a timeout is an agent failure, timed at the cap). The default paired view is the *full set* (an errored trial counts
as a failed attempt); ``paired(..., complete_case=True)`` is the *complete case*: a task with an errored trial in
either arm leaves that pair, and the dropped tasks are listed (task, arm(s) that errored, error class = exception type
and first line, attempts, the other arm's outcome). Both views are printed with ``--baseline``. An errored trial is
re-run up to twice by hand (A6.22 (e) 4); move the errored trial directory aside when you re-run it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from jevharness.bench import metrics as M
except ImportError:  # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from jevharness.bench import metrics as M


def _secs(phase: dict[str, Any] | None) -> float | None:
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return None
    t0 = datetime.fromisoformat(phase["started_at"].replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(phase["finished_at"].replace("Z", "+00:00"))
    return (t1 - t0).total_seconds()


def decision_stats(path: Path, s1_price_per_m: float = 0.0) -> dict[str, Any]:
    recs = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []
    reqs: dict[str, dict[str, Any]] = {}
    for r in recs:
        if r.get("model") is None:  # retry/nudge limits: no model call
            continue
        reqs.setdefault(r.get("request_id") or f"{r.get('ts')}-{len(reqs)}", r)
    blocks = sum(1 for r in recs if r.get("final_value") == "block")
    # stop/guard: "continue" = the stop was rejected and the agent nudged; stop/trigger: "stop" = S1 ended the run
    nudges = sum(1 for r in recs if r.get("dp") == "stop" and r.get("mode", "guard") == "guard"
                 and r.get("final_value") == "continue")
    reflections = sum(1 for r in recs if r.get("dp") == "reflection" and r.get("final_value") is True)
    tokens = sum(int(r.get("input_tokens") or 0) for r in reqs.values())
    lat = [float(r["latency_ms"]) for r in reqs.values() if r.get("latency_ms") is not None]
    return {"s1_requests": len(reqs), "s1_decisions": len(recs), "blocks": blocks, "stop_nudges": nudges,
            "reflections": reflections, "s1_errors": sum(1 for r in recs if r.get("error")),
            "s1_input_tokens": tokens, "s1_ms_mean": round(sum(lat) / len(lat), 1) if lat else None,
            "s1_s_total": round(sum(lat) / 1000, 1) if lat else 0.0,
            "s1_cost_usd_est": round(tokens * s1_price_per_m / 1e6, 6)}


def trajectory_stats(path: Path) -> dict[str, Any]:
    """Agent steps (LLM calls) and tool calls from Harbor's ATIF trajectory, if present."""
    if not path.exists():
        return {"agent_steps": None, "tool_calls": None}
    try:
        steps = json.loads(path.read_text()).get("steps") or []
    except (json.JSONDecodeError, AttributeError):
        return {"agent_steps": None, "tool_calls": None}
    agent = [s for s in steps if s.get("source") == "agent"]
    return {"agent_steps": sum(int(s.get("llm_call_count") or 1) for s in agent),
            "tool_calls": sum(len(s.get("tool_calls") or []) for s in agent)}


def tests_stats(path: Path) -> dict[str, Any]:
    try:
        summ = json.loads(path.read_text())["results"]["summary"]
        return {"tests_passed": int(summ["passed"]), "tests_total": int(summ["tests"])}
    except (OSError, KeyError, TypeError, ValueError):
        return {"tests_passed": None, "tests_total": None}


def task_key(name: str) -> str:
    """Hub tasks are named ``org/slug``, local ones ``slug``: compare runs from either source by slug."""
    return name.rsplit("/", 1)[-1]


def trial_row(trial_dir: Path, s1_price_per_m: float = 0.0) -> dict[str, Any]:
    r = json.loads((trial_dir / "result.json").read_text())
    ar = r.get("agent_result") or {}
    vr = (r.get("verifier_result") or {}).get("rewards") or {}
    exc = r.get("exception_info") or {}
    err = exc.get("exception_type")
    msg = str(exc.get("exception_message") or "").strip().splitlines()
    return {"job": trial_dir.parent.name, "trial": trial_dir.name, "started_at": r.get("started_at"),
            "task": task_key(r.get("task_name") or trial_dir.name.split("__")[0]),
            "reward": vr.get("reward"), "error": err, "timed_out": err == "AgentTimeoutError",
            "infra_error": bool(err) and err != "AgentTimeoutError",       # A6.22 (e) 1
            "error_class": (f"{err}: {msg[0][:200]}" if msg else err) if err else None,
            "llm_cost_usd": ar.get("cost_usd"), "llm_input_tokens": ar.get("n_input_tokens"),
            "llm_cache_tokens": ar.get("n_cache_tokens"), "llm_output_tokens": ar.get("n_output_tokens"),
            "env_setup_s": _secs(r.get("environment_setup")), "agent_setup_s": _secs(r.get("agent_setup")),
            "agent_run_s": _secs(r.get("agent_execution")), "verifier_s": _secs(r.get("verifier")),
            "trial_s": _secs(r),
            **tests_stats(trial_dir / "verifier" / "ctrf.json"),
            **trajectory_stats(trial_dir / "agent" / "trajectory.json"),
            **decision_stats(trial_dir / "agent" / "jev-decisions.jsonl", s1_price_per_m)}


def summarize(job_dirs: list[Path], s1_price_per_m: float = 0.0) -> list[dict[str, Any]]:
    rows = []
    for job in job_dirs:
        for t in sorted(p for p in job.iterdir() if p.is_dir() and (p / "result.json").exists()):
            rows.append(trial_row(t, s1_price_per_m))
    return rows


# ------------------------------------------------------------------ statistics
def quantile(xs: list[float], q: float) -> float | None:
    """Linear-interpolation quantile (numpy's default); None for no data."""
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _solved(r: dict[str, Any]) -> bool:
    return r.get("reward") is not None and float(r["reward"]) >= 1.0


def _vals(rs: list[dict[str, Any]], k: str) -> list[float]:
    return [float(r[k]) for r in rs if r.get(k) is not None]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def arm_stats(rows: list[dict[str, Any]], budgets: list[float] | None = None) -> dict[str, dict[str, Any]]:
    by_job: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_job.setdefault(r["job"], []).append(r)
    out = {}
    for job, rs in by_job.items():
        run, trial = _vals(rs, "agent_run_s"), _vals(rs, "trial_s")
        cost = [float(r.get("llm_cost_usd") or 0) + float(r.get("s1_cost_usd_est") or 0) for r in rs]
        n_solved = sum(_solved(r) for r in rs)
        st: dict[str, Any] = {
            "n": len(rs), "tasks": len({r["task"] for r in rs}), "solved": n_solved,
            "success": n_solved / len(rs) if rs else None,
            "timeouts": sum(bool(r.get("timed_out")) for r in rs),
            "agent_run_p50_s": quantile(run, 0.5), "agent_run_p90_s": quantile(run, 0.9),
            "trial_p50_s": quantile(trial, 0.5), "trial_p90_s": quantile(trial, 0.9),
            "llm_cost_mean": _mean(_vals(rs, "llm_cost_usd")), "s1_cost_mean": _mean(_vals(rs, "s1_cost_usd_est")),
            "cost_total": sum(cost), "cost_per_solved": sum(cost) / n_solved if n_solved else None,
            "steps_mean": _mean(_vals(rs, "agent_steps")), "s1_requests_mean": _mean(_vals(rs, "s1_requests")),
        }
        for b in budgets or []:
            st[f"solved_by_{int(b)}s"] = sum(_solved(r) and (r.get("agent_run_s") or 0) <= b for r in rs) / len(rs)
        out[job] = st
    return out


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def per_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """job -> task -> attempt means (success, agent_run_s, cost, steps) plus ``n_solved`` and ``agent_run_s_solved``,
    the mean agent-run time of the solved attempts (None when no attempt solved the task)."""
    acc: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        acc.setdefault(r["job"], {}).setdefault(r["task"], []).append(r)
    return {job: {task: {"n": len(rs), "success": _mean([float(_solved(r)) for r in rs]),
                         "agent_run_s": _mean(_vals(rs, "agent_run_s")),
                         "n_solved": sum(_solved(r) for r in rs),
                         "agent_run_s_solved": _mean(_vals([r for r in rs if _solved(r)], "agent_run_s")),
                         "cost": _mean([float(r.get("llm_cost_usd") or 0) + float(r.get("s1_cost_usd_est") or 0)
                                        for r in rs]),
                         "steps": _mean(_vals(rs, "agent_steps"))}
                  for task, rs in tasks.items()}
            for job, tasks in acc.items()}


H5_KEYS = ("d_success_lb90", "sign_p_wins", "n_h5_both_solved", "d_solved_run_s_median", "time_sign_p", "wilcoxon_p",
           "h5_better", "h5_strictly_better", "h5_faster", "h5_verdict")
# A6.22 (d) 1: the displayed verdict labels (rule, thresholds and field names unchanged)
H5_LABEL_REGISTERED = "passes the exploratory success screen; faster among jointly solved tasks"
H5_LABEL_STRICT = "faster and better (faster among jointly solved tasks)"
H5_BUDGETS = (600, 1800, 3600)    # solved-by-t shown on the verdict row (A6.22 (d) 1)


def h5_fields(tasks: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]], common: list[str],
              n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    """The A5.2 verdict of one arm (``tasks``) against the baseline on the ``common`` tasks, as table columns."""
    def times(pt: dict[str, dict[str, Any]], key: str) -> dict[str, float]:
        return {t: float(pt[t][key]) for t in common if pt[t].get(key) is not None}
    v = M.h5_verdict({t: base[t]["success"] for t in common}, {t: tasks[t]["success"] for t in common},
                     times(base, "agent_run_s"), times(tasks, "agent_run_s"),
                     times(base, "agent_run_s_solved"), times(tasks, "agent_run_s_solved"),
                     n_boot=n_boot, seed=seed)
    return {"d_success_lb90": v["d_success_lb"], "sign_p_wins": v["sign_p_wins"],
            "n_h5_both_solved": v["n_both_solved"], "d_solved_run_s_median": v["d_time_median_solved"],
            "time_sign_p": v["time_sign_p"], "wilcoxon_p": v["wilcoxon_p"], "h5_better": v["success_preserving"],
            "h5_strictly_better": v["strictly_better"], "h5_faster": v["faster"],
            # A6.6 tiers, A6.16 (j) wording: the screen is exploratory, "faster" is on co-solved tasks
            "h5_verdict": (H5_LABEL_STRICT if v["faster_and_better"]
                           else H5_LABEL_REGISTERED if v["success_preserving_and_faster"]
                           else "success-preserving" if v["success_preserving"]
                           else "faster" if v["faster"] else "neither")}


def _errored(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """job -> task -> its infrastructure-errored trials (A6.22 (e) 1)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        if r.get("infra_error"):
            out.setdefault(r["job"], {}).setdefault(r["task"], []).append(r)
    return out


def _solved_by(tasks: dict[str, dict[str, Any]], trials: dict[str, list[dict[str, Any]]], common: list[str],
               t: float) -> float | None:
    """Mean over ``common`` of each task's share of attempts solved within ``t`` s of agent-run time."""
    vals = [sum(_solved(r) and (r.get("agent_run_s") or 0) <= t for r in trials[k]) / len(trials[k])
            for k in common if trials.get(k)]
    return _mean(vals)


def paired(rows: list[dict[str, Any]], baseline: str, n_boot: int = 10000, seed: int = 0,
           complete_case: bool = False) -> dict[str, dict[str, Any]]:
    """Per-arm paired comparison against ``baseline`` (see the module docstring), including the H5 verdict columns
    (``H5_KEYS``) and solved-by-t at ``H5_BUDGETS`` for both arms. ``complete_case`` (A6.22 (e) 5): a task with an
    infrastructure-errored trial in the arm or the baseline leaves that pair and is listed under ``dropped``; the
    default is the full-set view (every task both arms ran, an errored trial counted as a failed attempt)."""
    pt = per_task(rows)
    if baseline not in pt:
        raise ValueError(f"baseline job {baseline!r} not among {sorted(pt)}")
    base = pt[baseline]
    errs = _errored(rows)
    trials: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        trials.setdefault(r["job"], {}).setdefault(r["task"], []).append(r)
    out = {}
    for job, tasks in pt.items():
        if job == baseline:
            continue
        common = sorted(set(tasks) & set(base))
        bad = [t for t in common if (errs.get(job) or {}).get(t) or (errs.get(baseline) or {}).get(t)]
        dropped = []
        if complete_case:
            for t in bad:
                sides = {baseline: base[t], job: tasks[t]}
                where = [j for j in sides if (errs.get(j) or {}).get(t)]
                dropped.append({"task": t, "errored_in": where,
                                "error_class": {j: sorted({r["error_class"] for r in errs[j][t]}) for j in where},
                                "attempts": {j: sides[j]["n"] for j in where},
                                "errored_attempts": {j: len(errs[j][t]) for j in where},
                                "other_outcome": {j: sides[j]["success"] for j in sides if j not in where}})
            common = [t for t in common if t not in set(bad)]
        d_succ = [tasks[t]["success"] - base[t]["success"] for t in common]
        d_time = [tasks[t]["agent_run_s"] - base[t]["agent_run_s"] for t in common
                  if tasks[t]["agent_run_s"] is not None and base[t]["agent_run_s"] is not None]
        d_time_solved = [tasks[t]["agent_run_s"] - base[t]["agent_run_s"] for t in common
                         if tasks[t]["success"] == 1.0 and base[t]["success"] == 1.0]
        d_cost = [tasks[t]["cost"] - base[t]["cost"] for t in common]
        wins, losses = sum(d > 0 for d in d_succ), sum(d < 0 for d in d_succ)
        out[job] = {"tasks": len(common), "success_arm": _mean([tasks[t]["success"] for t in common]),
                    "success_base": _mean([base[t]["success"] for t in common]),
                    "d_success_mean": _mean(d_succ), "wins": wins, "losses": losses,
                    "sign_test_p": sign_test_p(wins, losses),
                    "d_agent_run_s_median": quantile(d_time, 0.5),
                    "faster_tasks": sum(d < 0 for d in d_time),
                    "d_agent_run_s_median_both_solved": quantile(d_time_solved, 0.5),
                    "n_both_solved": len(d_time_solved), "d_cost_median": quantile(d_cost, 0.5),
                    "view": "complete case" if complete_case else "full set", "n_errored_tasks": len(bad),
                    **({"dropped": dropped} if complete_case else {}),
                    **{f"solved_by_{b}s": _solved_by(tasks, trials[job], common, b) for b in H5_BUDGETS},
                    **{f"solved_by_{b}s_base": _solved_by(base, trials[baseline], common, b) for b in H5_BUDGETS},
                    **h5_fields(tasks, base, common, n_boot=n_boot, seed=seed)}
    return out


# ------------------------------------------------------------------ printing
def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.1f}"
    return str(v)


def _table(cols: list[str], rows: list[dict[str, Any]]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", nargs="+", type=Path, help="Harbor job directories (<jobs-dir>/<job-name>)")
    ap.add_argument("--s1-price-per-m", type=float, default=0.0, help="decision-model USD per 1M input tokens")
    ap.add_argument("--baseline", default=None, help="job name of the baseline arm: adds paired per-task deltas")
    ap.add_argument("--budgets", default="", help="comma-separated agent-run seconds for 'solved by t' columns")
    ap.add_argument("--json", type=Path, default=None, help="also write rows (+ arms, paired) as JSON")
    a = ap.parse_args(argv)
    rows = summarize(a.jobs, a.s1_price_per_m)
    budgets = [float(b) for b in a.budgets.split(",") if b.strip()]
    cols = ["job", "task", "reward", "tests_passed", "tests_total", "error", "llm_cost_usd", "llm_input_tokens",
            "agent_steps", "agent_setup_s", "agent_run_s", "verifier_s", "trial_s", "s1_requests", "blocks",
            "stop_nudges", "reflections", "s1_ms_mean", "s1_cost_usd_est", "s1_errors"]
    print(_table(cols, rows))
    arms = arm_stats(rows, budgets)
    acols = ["job", "n", "tasks", "success", "timeouts", "agent_run_p50_s", "agent_run_p90_s", "trial_p50_s",
             "trial_p90_s", "llm_cost_mean", "s1_cost_mean", "cost_per_solved", "steps_mean", "s1_requests_mean"]
    acols += [k for k in next(iter(arms.values()), {}) if k.startswith("solved_by_")]
    print("\n" + _table(acols, [{"job": j, **s} for j, s in arms.items()]))
    pair = paired(rows, a.baseline) if a.baseline else {}
    cc = paired(rows, a.baseline, complete_case=True) if a.baseline else {}
    if pair:
        pcols = ["job", "tasks", "success_base", "success_arm", "d_success_mean", "wins", "losses", "sign_test_p",
                 "d_agent_run_s_median", "faster_tasks", "n_both_solved", "d_agent_run_s_median_both_solved",
                 "d_cost_median"]
        print(f"\npaired vs {a.baseline} (per-task attempt means; d = arm - baseline)\n"
              + _table(pcols, [{"job": j, **s} for j, s in pair.items()]))
        hcols = ["job", "view", "tasks", "d_success_mean", "d_success_lb90", "wins", "losses", "sign_p_wins",
                 "n_h5_both_solved", "d_solved_run_s_median", "time_sign_p", "wilcoxon_p", "d_agent_run_s_median",
                 *(f"solved_by_{b}s{x}" for b in H5_BUDGETS for x in ("", "_base")),
                 "h5_better", "h5_strictly_better", "h5_faster", "h5_verdict"]
        print("\nH5 verdict (A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory): h5_better = the success-"
              "preservation screen (LB90(d_success) > -0.05 and wins >= losses; not an NI test); h5_strictly_better = "
              "LB90 > 0 and sign p < 0.05; faster = median d agent-run time on tasks both solve < 0 and exact one-sided "
              "sign test p < 0.05 (wilcoxon_p descriptive); d_agent_run_s_median (all tasks, timeouts at the cap) and "
              "solved_by_t (arm, _base) descriptive. Full set: an errored trial (any Harbor exception except "
              "AgentTimeoutError) counts as a failure; complete case: a task with an errored trial in either arm "
              "leaves the pair (A6.22 (e) 5)\n"
              + _table(hcols, [{"job": j, **s} for j, s in pair.items()] + [{"job": j, **s} for j, s in cc.items()]))
        drop = [(j, d) for j, s in cc.items() for d in s.get("dropped", [])]
        if drop:
            print("\ncomplete case: dropped tasks (A6.22 (e) 5)\n" + _table(
                ["job", "task", "errored_in", "error_class", "attempts", "errored_attempts", "other_outcome"],
                [{"job": j, **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in d.items()}}
                 for j, d in drop]))
    if a.json:
        a.json.write_text(json.dumps({"rows": rows, "arms": arms, "paired": pair, "paired_complete_case": cc}
                                     if (budgets or pair) else rows, indent=1))


if __name__ == "__main__":
    main()
