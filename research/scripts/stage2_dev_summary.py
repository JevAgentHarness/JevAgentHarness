#!/usr/bin/env python3
"""Summary of the Stage-2 DEVELOPMENT check (configs/dev/stage2_dev.yaml; docs/JEVAGENT_HARNESS.md §6): per arm
success, cost, latency, System-1 / System-2 calls, decision counts and a ledger audit (every call record of an episode
sums to its usage totals), and per pair (vs the same System-2's LLM-only arm) Δsuccess, the registered NI p (3 pt),
cost and latency savings with their paired t-test p. DEVELOPMENT data: descriptive only, never a paper number.

    python scripts/stage2_dev_summary.py runs/stage2_dev/stage2_dev --out results/stage2_dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import primaries as P  # noqa: E402
from jevharness.bench import metrics as M  # noqa: E402

PAIRS = [("jevagent_default", "llm_only"), ("jev_ts_cascade_tauhat_dev", "llm_only"),
         ("jevagent_default", "jev_ts_cascade_tauhat_dev"), ("jevagent_default_s2lite", "llm_only_s2lite")]


def ledger_audit(trajs: list[dict]) -> dict:
    """Every non-measurement call record of an episode sums to ``usage.total`` (calls, cost)."""
    bad, n_calls = 0, 0
    for t in trajs:
        calls = [c for c in t.get("calls") or [] if not (c.get("meta") or {}).get("shadow")]
        n_calls += len(calls)
        tot = (t.get("usage") or {}).get("total") or {}
        cost = sum(float(c.get("cost_usd", 0) or 0) for c in calls)
        if len(calls) != int(tot.get("calls", -1)) or abs(cost - float(tot.get("cost_usd", 0) or 0)) > 1e-9:
            bad += 1
    return {"episodes": len(trajs), "call_records": n_calls, "episodes_not_matching": bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="results/stage2_dev")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    run = Path(a.run)
    arms: dict[str, dict] = {}
    rows = {}
    for p in sorted(run.glob("*.jsonl")):
        A = P.load_arm(p, keep_trajs=True)
        arms[p.stem] = A
        tr = A["trajs"]
        ok = [t for t in tr if not t.get("error")]
        recs = M.decision_records(ok)
        u = [t.get("usage") or {} for t in ok]
        rows[p.stem] = {
            "n": len(tr), "errors": len(tr) - len(ok), "success": float(np.mean([bool(t["result"].get("success"))
                                                                                for t in ok])),
            "cost_per_episode": float(np.mean([x["total"]["cost_usd"] for x in u])),
            "s1_cost_per_episode": float(np.mean([(x.get("s1") or {}).get("cost_usd", 0) for x in u])),
            "latency_s": float(np.mean([x["total"]["latency_ms"] for x in u])) / 1000,
            "s2_calls": float(np.mean([(x.get("s2") or {}).get("calls", 0) for x in u])),
            "s1_calls": float(np.mean([(x.get("s1") or {}).get("calls", 0) for x in u])),
            "unpriced": sorted({m for x in u for m in x.get("unpriced_models") or []}),
            "decisions": dict(Counter(f"{r['dp']}/{r['mode']}" + ("/deferred" if r.get("deferred") else "")
                                      for r in recs)),
            "coverage": {k: A["coverage"].get(k) for k in ("tool_coverage", "arg_coverage", "llm_free_steps")},
            "ledger": ledger_audit(tr)}
    pairs = {}
    for arm, ref in PAIRS:
        if arm not in arms or ref not in arms:
            continue
        ids = sorted(set(e["task"] for e in arms[ref]["episodes"].values()))
        ta, tb = P.task_table(arms[arm], ids), P.task_table(arms[ref], ids)
        pt = P.pair_tables(tb, ta, ids, names=(ref, arm))
        keys = pt["keys"]
        nb = P.ni_block(pt["ref_y"], pt["arm_y"], 0.03, a.n_boot, 0)
        cb = P.cost_block(tb, ta, keys, a.n_boot, 0)
        dc = [tb[k]["cost"] - ta[k]["cost"] for k in keys]
        dl = [tb[k]["latency"] - ta[k]["latency"] for k in keys]
        pairs[f"{arm} vs {ref}"] = {
            "n": nb["n"], "wins": nb["wins"], "losses": nb["losses"], "diff_success": nb["diff"], "ci95": nb["ci95"],
            "p_ni_3pt": nb["p_decisive"], "cost_savings": cb.get("cost_savings"),
            "cost_savings_ci": cb.get("cost_savings_ci"), "p_cheaper": M.paired_t_test(dc)["p_greater"],
            "latency_saving_s": float(np.mean(dl)), "p_faster": M.paired_t_test(dl)["p_greater"]}
    res = {"status": "DEVELOPMENT (dev seeds 900-999; descriptive, never a paper number)",
           "run": P._rel(run), "arms": rows, "pairs": pairs,
           "provenance": {k: {x: v.get(x) for x in ("path", "sha256", "n_records")} for k, v in arms.items()}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(P._jsonable(res), indent=1, default=P._json_default))
    print(json.dumps(P._jsonable({"arms": rows, "pairs": pairs}), indent=1, default=P._json_default))


if __name__ == "__main__":
    main()
