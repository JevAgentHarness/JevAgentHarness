"""H5 readout (pre-registration A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory) for mini-SWE-agent runs.

    python adapters/mini_swe/h5_summary.py runs/swe_verified_100 --baseline baseline [--json out.json]

Reads ``<run_dir>/<arm>.jsonl`` (jevharness Trajectories written by ``python -m jevharness.adapters.mini_swe collect``
with the SWE-bench evaluation report, so ``result.success`` = resolved) and applies the same functions as
``adapters/pi/summarize_harbor.py`` (``jevharness.bench.metrics.h5_verdict``) with k = 1: per instance, success is
resolved (0/1) and time is the agent-run wall clock (``wall_ms``; mini-SWE-agent has no time cap, its caps are 250
steps and $3). McNemar's test (A5.2, k = 1) is printed next to the exact sign test. Views (A6.22 (e) 5): *full set*
(an instance whose record has ``error`` set, i.e. an exception escaped the agent, counts as unresolved) and *complete
case* (an instance errored in either arm leaves that pair; the dropped instances are listed). Descriptive only: the
all-instance median time difference, solved-by-t at 600 / 1800 / 3600 s, cost (System-2 from mini-SWE's litellm
accounting, System-1 from the Jev usage) and the exit statuses.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from jevharness.bench import metrics as M
    from jevharness.harness.trajectory import load_trajectories
except ImportError:  # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from jevharness.bench import metrics as M
    from jevharness.harness.trajectory import load_trajectories

BUDGETS = (600, 1800, 3600)
LABEL_STRICT = "faster and better (faster among jointly solved tasks)"
LABEL_REGISTERED = "passes the exploratory success screen; faster among jointly solved tasks"


def _cost(usage: dict[str, Any], comp: str) -> float:
    return float(((usage or {}).get(comp) or {}).get("cost_usd") or 0.0)


def load_arm(path: Path) -> dict[str, dict[str, Any]]:
    """task id -> {success, time_s, error, exit_status, s1/s2/total cost, steps}; the latest record per task wins."""
    out: dict[str, dict[str, Any]] = {}
    for t in load_trajectories(str(path)):
        u = t.get("usage") or {}
        res = t.get("result") or {}
        out[t["task_id"]] = {
            "success": float(bool(res.get("success"))), "resolved": res.get("resolved"),
            "time_s": float(t.get("wall_ms") or 0.0) / 1000.0, "error": t.get("error"),
            "exit_status": (t.get("meta") or {}).get("exit_status") or res.get("exit_status"),
            "s1_cost": _cost(u, "s1"), "s2_cost": _cost(u, "s2") or _cost(u, "llm"),
            "total_cost": _cost(u, "total"), "steps": len(t.get("steps") or []),
        }
    return out


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def arm_row(name: str, d: dict[str, dict[str, Any]]) -> dict[str, Any]:
    v = list(d.values())
    n = len(v)
    times = [r["time_s"] for r in v]
    row = {"arm": name, "n": n, "resolved": int(sum(r["success"] for r in v)),
           "success": sum(r["success"] for r in v) / n if n else None,
           "unevaluated": sum(r["resolved"] is None for r in v), "errors": sum(bool(r["error"]) for r in v),
           "time_p50_s": _q(times, 0.5), "time_p90_s": _q(times, 0.9),
           "s2_cost_mean": sum(r["s2_cost"] for r in v) / n if n else None,
           "s1_cost_mean": sum(r["s1_cost"] for r in v) / n if n else None,
           "cost_total": sum(r["total_cost"] for r in v),
           "cost_per_resolved": (sum(r["total_cost"] for r in v) / sum(r["success"] for r in v)
                                 if any(r["success"] for r in v) else None),
           "steps_mean": sum(r["steps"] for r in v) / n if n else None,
           "exit_statuses": dict(Counter(str(r["exit_status"]) for r in v).most_common())}
    for b in BUDGETS:
        row[f"solved_by_{b}s"] = sum(r["success"] and r["time_s"] <= b for r in v) / n if n else None
    return row


def paired(base: dict[str, dict[str, Any]], arm: dict[str, dict[str, Any]], complete_case: bool,
           n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    common = sorted(set(base) & set(arm))
    bad = [t for t in common if base[t]["error"] or arm[t]["error"]]
    dropped = []
    if complete_case:
        for t in bad:
            where = [n for n, d in (("baseline", base), ("arm", arm)) if d[t]["error"]]
            dropped.append({"task": t, "errored_in": where,
                            "error_class": {n: str((base if n == "baseline" else arm)[t]["error"]).splitlines()[0][:200]
                                            for n in where},
                            "other_outcome": {n: (base if n == "baseline" else arm)[t]["success"]
                                              for n in ("baseline", "arm") if n not in where}})
        common = [t for t in common if t not in set(bad)]
    bs = {t: base[t]["success"] for t in common}
    as_ = {t: arm[t]["success"] for t in common}
    bt = {t: base[t]["time_s"] for t in common}
    at = {t: arm[t]["time_s"] for t in common}
    v = M.h5_verdict(bs, as_, bt, at, n_boot=n_boot, seed=seed)
    mc = M.mcnemar(bs, as_)
    out = {"view": "complete case" if complete_case else "full set", "tasks": len(common), "n_errored_tasks": len(bad),
           "success_base": sum(bs.values()) / len(bs) if bs else None,
           "success_arm": sum(as_.values()) / len(as_) if as_ else None,
           "mcnemar_p": mc["p"], "mcnemar_p_arm_worse": mc["p_b_worse"], "mcnemar_p_arm_better": mc["p_b_better"],
           "d_cost_median": _q([arm[t]["total_cost"] - base[t]["total_cost"] for t in common], 0.5),
           **{k: v[k] for k in ("d_success", "d_success_lb", "wins", "losses", "sign_p", "sign_p_wins",
                                "n_both_solved", "d_time_median_solved", "time_sign_p", "wilcoxon_p",
                                "d_time_median_all", "success_preserving", "strictly_better", "faster",
                                "success_preserving_and_faster", "faster_and_better", "screen_label")},
           "h5_verdict": (LABEL_STRICT if v["faster_and_better"] else LABEL_REGISTERED
                          if v["success_preserving_and_faster"] else
                          "success-preserving" if v["success_preserving"] else "faster" if v["faster"] else "neither")}
    for b in BUDGETS:
        out[f"solved_by_{b}s"] = sum(as_[t] and at[t] <= b for t in common) / len(common) if common else None
        out[f"solved_by_{b}s_base"] = sum(bs[t] and bt[t] <= b for t in common) / len(common) if common else None
    if complete_case:
        out["dropped"] = dropped
    return out


def _fmt(x: Any) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4f}" if abs(x) < 1 else f"{x:.1f}"
    if isinstance(x, (dict, list)):
        return json.dumps(x)
    return str(x)


def table(cols: list[str], rows: list[dict[str, Any]]) -> str:
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--arms", nargs="*", default=None, help="default: every <arm>.jsonl in run_dir")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    names = a.arms or sorted(p.stem for p in a.run_dir.glob("*.jsonl"))
    data = {n: load_arm(a.run_dir / f"{n}.jsonl") for n in names if (a.run_dir / f"{n}.jsonl").exists()}
    rows = [arm_row(n, d) for n, d in data.items()]
    print("## Arms (k = 1; time = agent-run wall clock)\n")
    print(table(["arm", "n", "resolved", "success", "unevaluated", "errors", "time_p50_s", "time_p90_s",
                 "s2_cost_mean", "s1_cost_mean", "cost_total", "cost_per_resolved", "steps_mean",
                 *(f"solved_by_{b}s" for b in BUDGETS), "exit_statuses"], rows))
    out: dict[str, Any] = {"arms": rows, "paired": {}}
    if a.baseline in data:
        prs = []
        for n, d in data.items():
            if n == a.baseline:
                continue
            for cc in (False, True):
                p = paired(data[a.baseline], d, cc, n_boot=a.n_boot)
                out["paired"].setdefault(n, {})[p["view"]] = p
                prs.append({"arm": n, **p})
        print(f"\n## H5 vs {a.baseline} (A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory, outside the Holm "
              "family). success_preserving = the exploratory success screen (LB90 of d_success > -0.05 and wins >= "
              "losses; not an NI test); strictly_better = LB90 > 0 and sign p_wins < 0.05; faster = median d time on "
              "instances both resolve < 0 and exact one-sided sign test p < 0.05 (wilcoxon_p descriptive). McNemar "
              "(k = 1) and d_time_median_all / solved_by_t / d_cost_median are descriptive.\n")
        print(table(["arm", "view", "tasks", "n_errored_tasks", "success_base", "success_arm", "d_success",
                     "d_success_lb", "wins", "losses", "sign_p", "sign_p_wins", "mcnemar_p", "n_both_solved",
                     "d_time_median_solved", "time_sign_p", "wilcoxon_p", "d_time_median_all", "d_cost_median",
                     *(f"solved_by_{b}s{x}" for b in BUDGETS for x in ("", "_base")),
                     "success_preserving", "strictly_better", "faster", "h5_verdict"], prs))
        drop = [{"arm": r["arm"], **d} for r in prs for d in r.get("dropped", [])]
        if drop:
            print("\ncomplete case: dropped instances (A6.22 (e) 5)\n"
                  + table(["arm", "task", "errored_in", "error_class", "other_outcome"], drop))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
