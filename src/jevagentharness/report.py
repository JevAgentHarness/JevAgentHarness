"""Summaries of a proxy log (``steps.jsonl``): per model name, sessions, steps, tier shares, cost and latency; and of
the decision calls per decision point (``decisions.jsonl``)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_steps(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "steps.jsonl"
    out = []
    for ln in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def summarize(steps: Iterable[dict[str, Any]], by: str = "model_name") -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cache_cost: dict[str, float] = defaultdict(float)
    steps = list(steps)
    model_of = {s.get("session"): s.get(by) for s in steps if s.get("kind") != "cache"}
    for s in steps:
        if s.get("kind") == "cache":
            cache_cost[str(model_of.get(s.get("session")))] += float(s.get("cost_usd") or 0)
        else:
            groups[str(s.get(by))].append(s)
    out: dict[str, dict[str, Any]] = {}
    for g, ss in sorted(groups.items()):
        sessions = {s.get("session") for s in ss}
        n = len(ss)
        tiers: dict[str, int] = defaultdict(int)
        reasons: dict[str, int] = defaultdict(int)
        for s in ss:
            tiers[str(s.get("tier"))] += 1
            reasons[str(s.get("reason"))] += 1
        cost = sum(float(s.get("cost_usd") or 0) for s in ss) + cache_cost.get(g, 0.0)
        dec = sum(float(s.get("decision_cost_usd") or 0) for s in ss)
        lat = sum(float(s.get("latency_s") or 0) for s in ss)
        dlat = sum(float((s.get("decision") or {}).get("latency_s") or 0) for s in ss)
        out[g] = {"sessions": len(sessions), "steps": n, "steps_per_session": round(n / max(1, len(sessions)), 1),
                  "tier_share": {k: round(v / n, 3) for k, v in sorted(tiers.items())},
                  "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
                  "cost_usd": round(cost, 4), "cost_per_session_usd": round(cost / max(1, len(sessions)), 4),
                  "decision_cost_usd": round(dec, 4), "cache_cost_usd": round(cache_cost.get(g, 0.0), 4),
                  "llm_seconds": round(lat, 1),
                  "llm_seconds_per_session": round(lat / max(1, len(sessions)), 1),
                  "decision_seconds": round(dlat, 1),
                  "errors": sum(1 for s in ss if s.get("error")),
                  "unpriced_calls": sum(1 for s in ss for c in s.get("calls") or [] if not c.get("priced", True))}
    return out


def table(summary: dict[str, dict[str, Any]]) -> str:
    rows = ["| model | sessions | steps/session | strong share | cost/session $ | LLM s/session | errors |",
            "|---|---|---|---|---|---|---|"]
    for g, s in summary.items():
        strong = s["tier_share"].get("strong", 0.0)
        rows.append(f"| {g} | {s['sessions']} | {s['steps_per_session']} | {strong:.0%} | "
                    f"{s['cost_per_session_usd']:.4f} | {s['llm_seconds_per_session']} | {s['errors']} |")
    return "\n".join(rows)


def decisions_table(described: dict[str, dict[str, Any]]) -> str:
    """Decision calls per point and mode (from decisions.jsonl, via fit.describe)."""
    rows = ["| decision point:mode | calls | errors | acted | p50 latency s | cost $ |", "|---|---|---|---|---|---|"]
    for g, d in described.items():
        lat = "-" if d["latency_p50_s"] is None else round(d["latency_p50_s"], 3)
        rows.append(f"| {g} | {d['calls']} | {d['errors']} | {d['acted']} | {lat} | {d['cost_usd']} |")
    return "\n".join(rows)
