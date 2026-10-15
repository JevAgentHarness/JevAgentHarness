"""API spend of JevAgentRL-v2 (GPU round 6) from its ledgers, and the API cap check (manifest §3: $125).

    python scripts/train/r6_spend.py --dir runs/gpu_round6_miles --ck checkpoints/r6            # table
    python scripts/train/r6_spend.py --dir ... --ck ... --check --cap 125 --reserve 11           # exit 1 if no room

Sources: live-run summaries (stress / E2 dev: n x mean cost_usd per episode, `scripts/train/r5_summary.py`), the OPD
teacher ledger (`teacher_ledger.jsonl`: running teacher_cost_total), the OPD / GRPO executors (`rollout_metrics.jsonl`:
episodes x mean cost_usd per episode), for the stages and their smoke runs alike.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _jsonl(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def spend(out: Path, ck: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    for sub in ("stress", "e2dev"):
        for f in sorted((out / sub).glob("*_summary.json")):
            for k, v in json.loads(f.read_text()).items():
                if isinstance(v, dict) and v.get("n") and v.get("cost_usd") is not None:
                    rows[f"{sub}: {k}"] = float(v["n"]) * float(v["cost_usd"])
    stage_dirs = {p.name: p for p in sorted(ck.glob("*")) if p.is_dir()}
    for name, d in stage_dirs.items():
        ex = 0.0
        for r in _jsonl(d / "rollout_metrics.jsonl"):
            c = r.get("cost_usd")
            if "episodes" in r and isinstance(c, (int, float)) and not math.isnan(c):
                ex += r["episodes"] * c
        if ex:
            rows[f"executors: {name}"] = ex
        led = _jsonl(d / "teacher_ledger.jsonl")
        if led:
            rows[f"teacher: {name}"] = max(float(r.get("teacher_cost_total", 0.0)) for r in led)
    rows["total"] = sum(rows.values())
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="runs/gpu_round6_miles")
    ap.add_argument("--ck", default="checkpoints/r6")
    ap.add_argument("--cap", type=float, default=125.0)
    ap.add_argument("--reserve", type=float, default=0.0, help="the expected cost of the run about to start")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    rows = spend(Path(a.dir), Path(a.ck))
    for k, v in rows.items():
        print(f"{v:9.3f}  {k}")
    if a.check and rows["total"] + a.reserve > a.cap:
        print(f"API cap: {rows['total']:.2f} spent + {a.reserve:.2f} reserved > {a.cap:.2f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
