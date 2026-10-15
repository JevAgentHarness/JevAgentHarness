#!/usr/bin/env python3
"""A7.12 (b), (c): the eligibility of the second System-2 pair and the outcome-blind cost gate that fixes S2-1X's n.
Reads the development smoke (`configs/dev/x_eligibility.yaml`, seeds 0-49) and never a success number.

    jevbench run configs/dev/x_eligibility.yaml --out runs/stage2_dev -o workers=8      # (three passes for errors)
    python scripts/x_eligibility.py --run runs/stage2_dev/x_eligibility [--write]   # -> results/stage2/x_cost_gate.json

Eligibility per member (A7.12 (b)): malformed responses <= 5 % of steps (a tool call naming no offered tool, arguments
that are not JSON, an empty response; a legitimate final-text answer is not malformed), no infrastructure error left
after the retries, no context-length error, and every System-2 call served by the pinned provider (the ledger's
`served_provider`, A7.12 (h)). The pair is eligible iff both members are.

Cost gate (A7.12 (c)): per-episode cost of each member at the verified list prices of A7.12 (a) (input, output and
cache-read tokens; the provider-reported cost is shown beside it), then predicted total = the smoke's actual cost +
calibration (150 X episodes with Jev) + `llm_only_x` (n) + `jevagent_best_x` (n, bounded by `llm_only_x`'s per-episode
cost) + `llm_only_xsmall` (n) + Jev's calls, x 1.2 contingency; n = 500 if it fits $25, else 300 if it fits, else
S2-1X is not tested (resource gate). Decided before the first calibration episode."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.harness.trajectory import iter_records, latest_by_task  # noqa: E402

# A7.12 (a), verified 2026-10-02 21:43 UTC on OpenRouter: $ per 1M input / output / cache-read tokens
LIST_PRICES = {"deepseek/deepseek-v4-pro": (0.2088, 0.4176, 0.0174), "deepseek/deepseek-v4-flash": (0.028, 0.056, 0.0056)}
MEMBERS = {"llm_only_x": "deepseek/deepseek-v4-pro", "llm_only_xsmall": "deepseek/deepseek-v4-flash"}
MALFORMED_MAX, ENVELOPE, CONTINGENCY, N_CAL = 0.05, 25.0, 1.2, 150
JEV_PER_EPISODE = 0.003      # planning: Jev $ per X episode with System-1 calls (dev check: $0.0014 cascade, shadow higher)
OUT = ROOT / "results/stage2/x_cost_gate.json"


def _malformed(t: Mapping[str, Any]) -> int:
    n = int((t.get("meta") or {}).get("empty_llm_responses") or 0)
    for st in t.get("steps") or []:
        if "__raw__" in (st.get("args") or {}) or str(st.get("observation") or "").startswith("Error: unknown tool"):
            n += 1
    return n


def member(path: Path, model: str, pinned: Sequence[str]) -> dict[str, Any]:
    recs = list(iter_records(path)) if path.exists() else []
    latest = latest_by_task(r for r in recs if "task_id" in r)
    eps = list(latest.values())
    errors = [t for t in eps if t.get("error")]
    ctx = [t for t in errors if "context" in str(t.get("error")).lower()]
    steps = sum(len(t.get("steps") or []) for t in eps if not t.get("error"))
    bad = sum(_malformed(t) for t in eps if not t.get("error"))
    served: dict[str, int] = {}
    tin = tout = tcache = rep = 0.0
    calls = 0
    for t in eps:
        for c in t.get("calls") or []:
            if c.get("component") != "s2":
                continue
            calls += 1
            m = c.get("meta") or {}
            sp = str(m.get("served_provider") or "(none)")
            served[sp] = served.get(sp, 0) + 1
            tin += float(c.get("input_tokens") or 0)
            tout += float(c.get("output_tokens") or 0)
            tcache += float(m.get("cached_input_tokens") or 0)
            rep += float(m.get("provider_cost_usd") or 0.0)
    pi, po, pc = LIST_PRICES[model]
    list_cost = ((tin - tcache) * pi + tcache * pc + tout * po) / 1e6
    n = max(1, len(eps))
    rate = bad / steps if steps else float("nan")
    pin_ok = bool(served) and set(served) <= set(pinned)
    return {"model": model, "path": str(path), "episodes": len(eps), "attempts": len(recs), "steps": steps,
            "malformed": bad, "malformed_rate": rate, "infra_errors_left": len(errors), "context_errors": len(ctx),
            "s2_calls": calls, "served_providers": served, "pinned": list(pinned),
            "eligible": bool(eps) and steps > 0 and rate <= MALFORMED_MAX and not errors and not ctx and pin_ok,
            "cost_per_episode_list": list_cost / n, "cost_per_episode_reported": rep / n, "smoke_cost_list": list_cost,
            "tokens_per_episode": {"input": tin / n, "output": tout / n, "cached": tcache / n}}


def gate(rows: Mapping[str, Mapping[str, Any]], jev_per_episode: float = JEV_PER_EPISODE) -> dict[str, Any]:
    cx = rows["llm_only_x"]["cost_per_episode_list"]
    cs = rows["llm_only_xsmall"]["cost_per_episode_list"]
    smoke = sum(r["smoke_cost_list"] for r in rows.values())
    pred = {}
    for n in (500, 300):
        pred[n] = CONTINGENCY * (smoke + N_CAL * cx + n * cx + n * cx + n * cs + (N_CAL + n) * jev_per_episode)
    n = 500 if pred[500] <= ENVELOPE else 300 if pred[300] <= ENVELOPE else None
    eligible = all(r["eligible"] for r in rows.values())
    return {"eligible": eligible, "predicted_usd": {str(k): v for k, v in pred.items()}, "envelope_usd": ENVELOPE,
            "n": n if eligible else None,
            "decision": (f"n = {n}" if n else "not tested (resource gate, A7.12 (c))") if eligible else
            "pair ineligible (A7.12 (b)): the next candidate pair, else S2-1X not tested",
            "assumptions": {"jev_per_episode_usd": jev_per_episode, "calibration_episodes": N_CAL,
                            "jevagent_best_x_bound": "llm_only_x per-episode cost", "contingency": CONTINGENCY,
                            "list_prices_per_M": LIST_PRICES}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="runs/stage2_dev/x_eligibility")
    ap.add_argument("--config", default=str(ROOT / "configs/experiments/stage2_x.yaml"))
    ap.add_argument("--jev-per-episode", type=float, default=JEV_PER_EPISODE)
    ap.add_argument("--write", action="store_true", help=f"write {OUT.relative_to(ROOT)}")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    pinned = (((cfg.get("llm") or {}).get("extra_body") or {}).get("provider") or {}).get("order") or []
    rows = {arm: member(Path(a.run) / f"{arm}.jsonl", model, pinned) for arm, model in MEMBERS.items()}
    res = {"registered": "PREREG_A7_STAGE2 A7.12 (b), (c)",
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "members": rows,
           **gate(rows, a.jev_per_episode)}
    print(json.dumps({k: res[k] for k in ("eligible", "predicted_usd", "n", "decision")}, indent=1))
    if a.write:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
