"""Compact development-slice table for GPU round 5 (JevAgentRL): one row per arm across several run directories.

    python scripts/train/r5_summary.py runs/gpu_round5_e2dev/e2dev_rl_refs runs/gpu_round5_e2dev/e2dev_rl_sft_v1 ... \
        --out results/gpu_round5/e2dev_summary

Per arm: n, success, $ / episode, System-2 calls, System-1 calls, System-1 decision accuracy on its executed value
(tool_select judged against the acceptable set AT(s), as `jevbench report`), per decision point and mode, ECE of the
tool_select top-label confidence, and System-1 latency per call (ledger: s1 latency_ms / s1 calls, mean over episodes
and the pooled p50 / p95 of per-call latencies from the call log when present). Writes <out>.json and <out>.md.
Development numbers only (dev seeds); never headline numbers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DPS = ("tool_select", "tool_params", "stop")


def _s1_call_latencies(trajs: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for t in trajs:
        for c in t.get("calls") or []:  # ledger CallRecords; shadow calls (meta.shadow) count: same decider, same call
            if c.get("component") == "s1" and c.get("latency_ms") is not None and not c.get("cached"):
                out.append(float(c["latency_ms"]))
    return out


def summarise(run_dir: Path) -> dict[str, dict[str, Any]]:
    from jevharness.bench.metrics import decision_records, decision_table, episode_table
    from jevharness.bench.report import collect

    runs = collect(run_dir)
    rows: dict[str, dict[str, Any]] = {}
    for arm, trajs in runs.items():
        if not trajs:
            continue
        ep = episode_table(trajs)
        dec = decision_table(decision_records(trajs))
        lat = _s1_call_latencies(trajs)
        per_call = lat
        # the GRPO training reward per episode (call-count-penalised success, configs/train/grpo_online_round5.yaml):
        # R = 1[success] * (1 - 0.5 * min(1, S2 calls / 10)) - 0.2 * 1[irreversible error]; errored episodes excluded
        rew = [float(bool(t["result"].get("success"))) * (1 - 0.5 * min(1.0, t["usage"].get("s2", {}).get("calls", 0) / 10))
               - 0.2 * float(bool((t.get("result") or {}).get("irreversible_error"))) for t in trajs if not t.get("error")]
        row: dict[str, Any] = {"run": run_dir.name, "n": len(trajs), "success": ep.get("success"),
                               "reward_ccps": float(np.mean(rew)) if rew else None,
                               "irreversible_rate": float(np.mean([bool((t.get("result") or {}).get("irreversible_error"))
                                                                   for t in trajs])),
                               "cost_usd": ep.get("cost_usd"), "llm_calls": ep.get("llm_calls"),
                               "s1_calls": ep.get("s1_calls"), "errors": ep.get("errors"),
                               "s1_latency_ms_per_call": float(np.mean(per_call)) if per_call else None,
                               "s1_latency_ms_p50": float(np.percentile(lat, 50)) if lat else None,
                               "s1_latency_ms_p95": float(np.percentile(lat, 95)) if lat else None}
        for key, r in dec.items():
            dp, mode = key.split("/", 1)
            if dp in DPS and r.get("s1_acc") is not None:
                row[f"{dp}/{mode}/s1_acc"] = r["s1_acc"]
                row[f"{dp}/{mode}/n"] = r.get("n_gold")
                if r.get("ece") is not None:
                    row[f"{dp}/{mode}/ece"] = r["ece"]
        rows[arm] = row
    return rows


def _f(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def to_md(all_rows: dict[str, dict[str, Any]]) -> str:
    cols = sorted({k for r in all_rows.values() for k in r if k.endswith("/s1_acc")})
    head = ["arm", "run", "n", "success", "$/ep", "S2 calls", "S1 ms/call"] + [c.replace("/s1_acc", "") for c in cols]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm, r in all_rows.items():
        cells = [arm.split("::")[-1], r["run"], str(r["n"]), _f(r["success"]), _f(r["cost_usd"], 4), _f(r["llm_calls"], 1),
                 _f(r["s1_latency_ms_per_call"], 0)]
        for c in cols:
            v = r.get(c)
            n = r.get(c.replace("/s1_acc", "/n"))
            cells.append(f"{_f(v)} ({n})" if v is not None else "-")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True, help="output prefix (.json / .md)")
    a = ap.parse_args(argv)
    all_rows: dict[str, dict[str, Any]] = {}
    for d in a.runs:
        p = Path(d)
        if not p.exists():
            print(f"[r5_summary] missing {d}")
            continue
        for arm, row in summarise(p).items():
            all_rows[f"{p.name}::{arm}"] = row
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(all_rows, indent=1, default=float) + "\n")
    Path(a.out + ".md").write_text(to_md(all_rows))
    print(to_md(all_rows))


if __name__ == "__main__":
    main()
