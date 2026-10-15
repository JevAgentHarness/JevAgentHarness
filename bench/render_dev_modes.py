"""Merge the dev-modes rows into bench/results/dev_modes.json and print the Markdown tables for dev_modes.md.

    JAH_RUNS=<dir holding dev-modes, dev-modes2, dev-modes3> python bench/render_dev_modes.py [row ...]

It also writes the table sections as sec_*.md in the current directory. The "summary" block of dev_modes.json
(spend per row) was added by hand from the proxies' stats.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_modes import F, S, analyze, counterfactuals  # noqa: E402

RUNS = Path(os.environ.get("JAH_RUNS", "runs/jah"))
OUT_JSON = Path(__file__).resolve().parent / "results" / "dev_modes.json"
PHASE = {"dev-modes": "P1", "dev-modes2": "P2", "dev-modes3": "P3"}
ORDER = ["mswe:jah-shadow", "mswe:jah-safe", "mswe:jah-auto-t75", "mswe:jah-submit", "mswe:jah-reflect",
         "mswe:jah-routeguard", "pi:jah-shadow", "pi:jah-stopguard", "pi:jah-reflect"]
DESC = {"jah-shadow": "default + every other point in shadow", "jah-safe": "default + action guard (safe preset)",
        "jah-auto-t75": "default (README setting)", "jah-submit": "default + finish check on mswe's submit "
        "(stop shadow, `submit`)", "jah-reflect": "default + reflection trigger (tau 0.8)",
        "jah-routeguard": "routing guard (tau 0.5) instead of the cascade", "jah-stopguard": "default with the "
        "stop guard (tau 0.9) instead of the finish check"}


OUT: dict[str, list[str]] = {"arms": [], "points": [], "fits": [], "grid": []}
_SEC = ["arms"]


def emit(line: str = "") -> None:
    OUT[_SEC[0]].append(line)
    print(line)


def main() -> int:
    rows = sys.argv[1:] or list(PHASE)
    arms, fits, cfs = {}, {}, {}
    for r in rows:
        p = RUNS / r
        if not (p / "jobs").exists():
            continue
        res = analyze(p)
        for k, v in res["arms"].items():
            v["row"], v["phase"] = r, PHASE[r]
            arms[f"{k} {PHASE[r]}"] = v
        for k, f in res["fits"].items():
            fits[f"{k} {PHASE[r]}"] = f
        for h, e in counterfactuals(p, ("jah-shadow", "jah-stopguard", "jah-submit")).items():
            cfs[f"{h} {PHASE[r]}"] = e
    keys = sorted(arms, key=lambda k: (ORDER.index(k.split()[0]) if k.split()[0] in ORDER else 99, k.split()[1]))
    out = {"note": "Terminal-Bench 2.0 DEV split (15 tasks), one trial per task, 2026-10-15: for choosing settings, "
                   "not a claim. P1/P2/P3 = rows dev-modes / dev-modes2 / dev-modes3 (run side by side within a row).",
           "arms": {k: arms[k] for k in keys}, "fits": fits, "counterfactuals": cfs}
    OUT_JSON.write_text(
        json.dumps(out, indent=1, default=str) + "\n")
    _SEC[0] = "arms"
    emit("| harness:arm | phase | what it is | success | $/task | agent s/task | strong share | steps/task | "
          "LLM calls/step | decision calls/task | decision $/task (share) | point acted/task |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = arms[k]
        name = k.split()[0]
        acted = ", ".join(f"{p} {e['acted_per_task']:g}" for p, e in r["points"].items()
                          if set(e["modes"]) - {"shadow"} and not (p == "routing" and set(e["modes"]) == {"cascade"}))
        emit(f"| {name} | {r['phase']} | {DESC.get(name.split(':')[1], '')} | {round(r['success'] * r['n'])}/{r['n']} "
              f"({100 * r['success']:.0f}%) | {r['cost_per_task']:.3f} | {r['agent_s_per_task']:.0f} | "
              f"{S._pct(r['strong_share'])} | {r['steps_per_task']} | {r['llm_calls_per_step']:.2f} | "
              f"{r['decision_calls_per_task']} | {r['decision_cost_usd'] / r['n']:.4f} "
              f"({100 * r['decision_cost_share']:.1f}%) | {acted or '-'} |")
    _SEC[0] = "points"
    for k in keys:
        r = arms[k]
        for p, e in r["points"].items():
            if p == "routing" and set(e["modes"]) == {"cascade"}:
                continue
            extra = ""
            if e.get("rechecks"):
                extra += f" rechecks {e['rechecks']}"
            if "labels_by_effect" in e:
                extra += f" labels {e['labels_by_effect']}"
            emit(f"- {k} {p}: modes {e['modes']}, calls {e['calls']} ({e['calls_per_task']}/task), errors "
                  f"{e['errors']}, acted {e['acted']}, effects {e['effects']}, "
                  f"fires@tau {e.get('fires_at_tau')}{extra}")
    _SEC[0] = "fits"
    for k, f in fits.items():
        emit(f"\n**{k}** (`jah fit`, target 95%, at least 30 decisions, z 1.645)\n\n{F.table(f)}")
    # per-task grid: success and $ per arm (T = agent timeout)
    trials = {}
    for r in rows:
        p = RUNS / r
        if (p / "jobs").exists():
            for t in S.join(S.load_trials(p / "jobs"), S.load_steps(p / "proxy")):
                trials[(f"{t['harness']}:{t['arm']} {PHASE[r]}", t["task"])] = t
    _SEC[0] = "grid"
    for h in ("mswe", "pi"):
        cols = [k for k in keys if k.startswith(h + ":")]
        emit(f"\n{h}:\n\n| task | " + " | ".join(c.split(":")[1] for c in cols) + " |")
        emit("|---|" + "---|" * len(cols))
        for task in sorted({tk for (_, tk) in trials}):
            cells = []
            for c in cols:
                t = trials.get((c, task))
                cells.append("-" if t is None else f"{'✓' if t['success'] else '✗'} {t['cost']:.2f}"
                             + (" T" if t["timeout"] else ""))
            emit(f"| {task} | " + " | ".join(cells) + " |")
    for k, v in OUT.items():
        (Path(__file__).parent / f"sec_{k}.md").write_text("\n".join(v).strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
