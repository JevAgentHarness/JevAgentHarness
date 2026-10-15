#!/usr/bin/env python3
"""A7.13 (iii): S2-4's provisioned per-episode tariff q, frozen before any priced JevAgentRL evaluation episode.

    # after the checkpoint and profile freeze: commit the serving window's boundaries and the development task list,
    # run jevagent_best_rl's configuration on development tasks inside that window, then
    python scripts/stage2_tariff.py --run runs/stage2_dev/tariff_workload --arm jevagent_best_rl \
        --host-usd-per-hour 1.32 --window-start 2026-10-05T10:00:00Z --window-end 2026-10-05T11:10:00Z \
        --served-model jevagent-rl --hardware "1x L4 24 GB" --precision bf16 --concurrency 4 \
        --sensitivity busy_throughput=0.0004 --workload-matches     # -> results/stage2/tariff.json (frozen once)

q = host $/h x serving hours (window end - start, including executor waiting and provisioned idle time) / completed
development episodes (latest record without an infrastructure error). The analysis charges q to every evaluated
JevAgentRL episode on top of its API cost; the served model's per-token price must be an exact 0 entry in
configs/prices/models.yaml so nothing is charged twice (the script refuses otherwise). Sensitivity rows (the
busy-throughput tariff, Jev's API price, $0) are reported beside the test, never instead of it."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.accounting import PriceTable  # noqa: E402
from jevharness.harness.trajectory import iter_records, latest_by_task  # noqa: E402

OUT = ROOT / "results/stage2/tariff.json"


def _t(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def tariff(run: Path, arm: str, host_usd_h: float, start: str, end: str) -> dict:
    path = run / f"{arm}.jsonl"
    latest = latest_by_task(r for r in iter_records(path) if "task_id" in r) if path.exists() else {}
    done = [t for t in latest.values() if not t.get("error")]
    hours = (_t(end) - _t(start)).total_seconds() / 3600.0
    if hours <= 0 or not done:
        raise SystemExit(f"no tariff: {len(done)} completed episode(s) in {path}, window {hours:.3f} h")
    return {"q_usd_per_episode": host_usd_h * hours / len(done), "host_usd_per_hour": host_usd_h,
            "serving_hours": hours, "episodes": len(done), "episodes_errored": len(latest) - len(done),
            "window": {"start": start, "end": end}, "workload": {"run": str(run), "arm": arm}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", default="jevagent_best_rl")
    ap.add_argument("--host-usd-per-hour", type=float, required=True)
    ap.add_argument("--window-start", required=True)
    ap.add_argument("--window-end", required=True)
    ap.add_argument("--served-model", default="jevagent-rl")
    ap.add_argument("--hardware", default="")
    ap.add_argument("--precision", default="")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--sensitivity", action="append", default=[], metavar="NAME=USD_PER_EPISODE")
    ap.add_argument("--workload-matches", action="store_true",
                    help="the measured workload matches the evaluated deployment (else S2-4 is not analysed under q)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: the tariff is frozen once (A7.13 (iii))")
    pt = PriceTable.load()
    p = pt.prices.get(a.served_model)
    if p is None or p.input_per_m != 0 or p.output_per_m != 0 or p.per_call != 0:
        raise SystemExit(f"configs/prices/models.yaml needs an exact `{a.served_model}: {{input_per_m: 0.0, "
                         "output_per_m: 0.0}}` entry before the tariff is frozen (A7.13 (iii) adaptation)")
    rec = {"registered": "PREREG_A7_STAGE2 A7.13 (iii)",
           "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "served_model": a.served_model, "hardware": a.hardware, "precision": a.precision,
           "concurrency": a.concurrency, "workload_matches": bool(a.workload_matches),
           **tariff(Path(a.run), a.arm, a.host_usd_per_hour, a.window_start, a.window_end),
           "sensitivity": {k: float(v) for k, v in (s.split("=", 1) for s in a.sensitivity)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"q = ${rec['q_usd_per_episode']:.5f} per episode ({rec['episodes']} episodes, {rec['serving_hours']:.2f} h) "
          f"-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
