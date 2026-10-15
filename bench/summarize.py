"""Join Harbor trial results with the proxy's per-step log and compare arms.

    python bench/summarize.py runs/jobs runs/proxy [--baseline jah-strong] [--json out.json] [--md out.md]

Per trial: success (reward >= 1), agent seconds (Harbor's agent-execution time), LLM + decision-model cost and the
strong-tier share from the proxy (sessions are named by the trial, see harbor_agents.py). Per harness, every arm is
compared with the baseline arm on the tasks both completed: success delta with wins/losses, cost and time savings
(ratio of means), and a paired bootstrap 95% interval for the success delta.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

INFRA = ("Docker", "docker compose", "EnvironmentStart", "apt-get", "ImagePull", "NonZeroAgentExitCode")


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _secs(t: dict[str, Any] | None) -> float | None:
    if not t:
        return None
    a, b = _ts(t.get("started_at")), _ts(t.get("finished_at"))
    return (b - a).total_seconds() if a and b else None


def safe(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)


def load_trials(jobs: Path) -> list[dict[str, Any]]:
    out = []
    for res in sorted(jobs.glob("*/*/result.json")):
        job = res.parent.parent.name
        harness, _, arm = job.partition("-")
        try:
            d = json.loads(res.read_text())
        except json.JSONDecodeError:
            continue
        rewards = ((d.get("verifier_result") or {}).get("rewards")) or {}
        reward = float(rewards.get("reward", max(rewards.values()) if rewards else 0.0))
        exc = d.get("exception_info") or {}
        etype = exc.get("exception_type") or ""
        out.append({"harness": harness, "arm": arm, "task": d.get("task_name"), "trial": res.parent.name,
                    "success": int(reward >= 1.0), "agent_s": _secs(d.get("agent_execution")),
                    "exception": etype or None,
                    "infra": bool(etype) and any(k in (etype + (exc.get("exception_message") or "")) for k in INFRA),
                    "timeout": "Timeout" in etype})
    return out


def load_steps(proxy: Path) -> dict[str, dict[str, Any]]:
    """session -> {cost, llm_s, steps, strong, fast, decision_cost}."""
    p = proxy / "steps.jsonl" if proxy.is_dir() else proxy
    agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"cost": 0.0, "llm_s": 0.0, "steps": 0, "strong": 0,
                                                          "fast": 0, "decision_cost": 0.0, "errors": 0})
    for ln in p.read_text(errors="replace").splitlines():
        try:
            s = json.loads(ln)
        except json.JSONDecodeError:
            continue
        a = agg[str(s.get("session"))]
        a["cost"] += float(s.get("cost_usd") or 0)
        if s.get("kind") == "cache":           # a context-cache write: cost only, not a step
            continue
        a["decision_cost"] += float(s.get("decision_cost_usd") or 0)
        a["llm_s"] += float(s.get("latency_s") or 0)
        a["steps"] += 1
        a["strong" if s.get("tier") == "strong" else "fast"] += 1
        a["errors"] += 1 if s.get("error") else 0
    return agg


def join(trials: list[dict[str, Any]], steps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    for t in trials:
        key = f"{t['arm']}/{safe(t['trial'])}"
        s = steps.get(key) or {}
        t.update({"cost": s.get("cost", 0.0), "llm_s": s.get("llm_s", 0.0), "steps": s.get("steps", 0),
                  "strong_share": (s["strong"] / s["steps"]) if s.get("steps") else None,
                  "decision_cost": s.get("decision_cost", 0.0), "proxy_errors": s.get("errors", 0)})
    return trials


def _boot(deltas: list[int], n: int = 4000, seed: int = 0) -> tuple[float, float]:
    if not deltas:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(deltas)
    xs = sorted(sum(deltas[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


def compare(trials: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    by: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for t in trials:
        by[(t["harness"], t["arm"])][t["task"]] = t
    out: dict[str, Any] = {}
    for (h, arm), tasks in sorted(by.items()):
        n = len(tasks)
        row: dict[str, Any] = {
            "n": n, "success": sum(t["success"] for t in tasks.values()) / max(1, n),
            "cost_per_task": sum(t["cost"] for t in tasks.values()) / max(1, n),
            "agent_s_per_task": sum(t["agent_s"] or 0 for t in tasks.values()) / max(1, n),
            "strong_share": _mean([t["strong_share"] for t in tasks.values() if t["strong_share"] is not None]),
            "infra_errors": sum(t["infra"] for t in tasks.values()),
            "timeouts": sum(t["timeout"] for t in tasks.values()),
        }
        base = by.get((h, baseline))
        if base and arm != baseline:
            common = sorted(set(tasks) & set(base))
            d = [tasks[k]["success"] - base[k]["success"] for k in common]
            ca, cb = sum(tasks[k]["cost"] for k in common), sum(base[k]["cost"] for k in common)
            ta, tb = sum(tasks[k]["agent_s"] or 0 for k in common), sum(base[k]["agent_s"] or 0 for k in common)
            lo, hi = _boot(d)
            row.update({"vs": baseline, "paired_n": len(common), "delta_pt": 100 * sum(d) / max(1, len(d)),
                        "delta_ci_pt": [100 * lo, 100 * hi], "wins": sum(x > 0 for x in d),
                        "losses": sum(x < 0 for x in d),
                        "cost_saving": (1 - ca / cb) if cb else None, "time_saving": (1 - ta / tb) if tb else None})
        out[f"{h}:{arm}"] = row
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def markdown(cmp: dict[str, Any]) -> str:
    lines = ["| harness:arm | n | success | Δ vs baseline (pt) [95% CI] | W/L | $/task | cost saving | agent s/task "
             "| time saving | strong share | infra/timeouts |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in cmp.items():
        delta = (f"{r['delta_pt']:+.1f} [{r['delta_ci_pt'][0]:+.1f}, {r['delta_ci_pt'][1]:+.1f}]"
                 if "delta_pt" in r else "baseline")
        wl = f"{r['wins']}/{r['losses']}" if "wins" in r else "–"
        lines.append(f"| {k} | {r['n']} | {100 * r['success']:.1f}% | {delta} | {wl} | {r['cost_per_task']:.3f} | "
                     f"{_pct(r.get('cost_saving'))} | {r['agent_s_per_task']:.0f} | {_pct(r.get('time_saving'))} | "
                     f"{_pct(r['strong_share'])} | {r['infra_errors']}/{r['timeouts']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Join Harbor trial results with the proxy's step log and compare arms.")
    ap.add_argument("jobs", type=Path, help="Harbor jobs directory (one folder per job, named <harness>-<arm>)")
    ap.add_argument("proxy", type=Path, help="the proxy's log directory or its steps.jsonl")
    ap.add_argument("--baseline", default="jah-strong", help="arm every other arm is compared with")
    ap.add_argument("--json", type=Path, help="write the comparison as JSON")
    ap.add_argument("--md", type=Path, help="write the comparison table as Markdown")
    ap.add_argument("--per-task", type=Path, help="write one CSV row per trial")
    a = ap.parse_args(argv)
    trials = join(load_trials(a.jobs), load_steps(a.proxy))
    cmp = compare(trials, a.baseline)
    md = markdown(cmp)
    print(md)
    if a.json:
        a.json.write_text(json.dumps(cmp, indent=1))
    if a.md:
        a.md.write_text(md + "\n")
    if a.per_task:
        cols = ["harness", "arm", "task", "success", "cost", "agent_s", "llm_s", "steps", "strong_share", "exception"]
        rows = [",".join(re.sub(r"[,\n]", " ", str(t.get(c) if t.get(c) is not None else "")) for c in cols)
                for t in trials]
        a.per_task.write_text("\n".join([",".join(cols), *rows]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
