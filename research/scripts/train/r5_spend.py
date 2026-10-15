"""API spend of GPU round 5 (JevAgentRL phase 1) from the ledgers of the pulled artefacts (results/gpu_round5/).

    python scripts/train/r5_spend.py results/gpu_round5 [--gpu-start 2026-10-02T20:11:03Z --gpu-end ... --gpu-usd-h 3.004]

Sources: E2 development-slice summaries (n x mean episode $; System-2 + priced System-1 calls; shadow calls are kept out
of the episode cost by the harness and are negligible), the superseded seed 0-49 reference run, OPD metrics (rollout
episode $ incl. greedy dev evaluations, teacher ledger $), GRPO metrics (rollout + evaluation episodes), the calibration
shadow arms. Executor prices come from configs/prices/models.yaml through the harness ledger.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def summary_cost(p: Path) -> float:
    rows = json.loads(p.read_text()) if p.exists() else {}
    return sum((r.get("n") or 0) * (r.get("cost_usd") or 0.0) for r in rows.values())


def rollout_cost(rows: list[dict], eval_n: int) -> tuple[float, float]:
    """(training rollouts $, greedy evaluation $) from opd / grpo metrics rows."""
    train = sum((r.get("episodes") or r.get("tasks") or 0) * (r.get("cost_usd") or 0.0) for r in rows if r.get("iter", 0) > 0)
    ev = 0.0
    for r in rows:
        for k, v in r.items():
            if k.startswith("eval/") and k.endswith("/cost_usd") and v:
                ev += eval_n * v
    return train, ev


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", nargs="?", default="results/gpu_round5")
    ap.add_argument("--gpu-start", default="2026-10-02T20:11:03Z")
    ap.add_argument("--gpu-end", default=None, help="instance stop time (default: now)")
    ap.add_argument("--gpu-usd-h", type=float, default=3.004)
    a = ap.parse_args(argv)
    root = Path(a.root)
    out: dict[str, float] = {}
    for p in sorted((root / "e2dev").glob("*_summary.json")):
        if p.name.startswith("summary_all"):
            continue
        out[f"e2dev:{p.stem.replace('_summary', '')}"] = summary_cost(p)
    out["e2dev:superseded_seeds0-49_refs_api"] = summary_cost(root / "e2dev_seeds0_49_superseded" / "e2dev_rl_refs_api_summary.json")
    opd = _jsonl(root / "opd_metrics.jsonl")
    tr, ev = rollout_cost(opd, 48)
    out["opd:rollouts"], out["opd:greedy_evals"] = tr, ev
    out["opd:teacher_jev"] = max([r.get("teacher_cost_total") or 0.0 for r in opd] or [0.0])
    grpo = _jsonl(root / "grpo_metrics.jsonl")
    tr, ev = rollout_cost(grpo, 48)
    out["grpo:rollouts"], out["grpo:greedy_evals"] = tr, ev
    out["cal:shadow_arms"] = summary_cost(root / "cal" / "shadow_cal_rl_summary.json")
    total = sum(out.values())
    end = datetime.fromisoformat(a.gpu_end.replace("Z", "+00:00")) if a.gpu_end else datetime.now().astimezone()
    hours = (end - datetime.fromisoformat(a.gpu_start.replace("Z", "+00:00"))).total_seconds() / 3600
    res = {"api_usd": {k: round(v, 3) for k, v in out.items()}, "api_usd_total": round(total, 2),
           "gpu_hours": round(hours, 2), "gpu_usd": round(hours * a.gpu_usd_h, 2), "gpu_usd_per_h": a.gpu_usd_h}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
