"""Competence gate for a System-1 backend on the training-side dev set (PREREGISTRATION amendment A5, P0-1).

Runs any `make_backend` spec on `data/train_v0/dev.jsonl` (210 gold decisions from training-seed JevGym episodes,
never used for evaluation) and reports decision accuracy overall and per decision point, Brier, and the Ledger
totals. Jev and Intern-Decision are scored through the same harness backends, so the gate compares like with like.

    python scripts/gate_s1_dev.py --s1 '{"type":"jev","provider":"openrouter"}' --out results/pilot_jev/gate_jev_dev.json
    python scripts/gate_s1_dev.py --s1 '{"type":"intern_decision","model":"internlm/Intern-Decision-4B"}' --out ...

Gate rule (A5): the open decider is co-primary iff acc_open >= 0.8 * acc_jev on these 210 items.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevharness.accounting import Ledger  # noqa: E402
from jevharness.decision import make_backend  # noqa: E402
from jevharness.decision.types import Choice, Noul  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True, help="backend spec JSON")
    ap.add_argument("--dev", default="data/train_v0/dev.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.dev)][: a.limit]
    backend = make_backend(json.loads(a.s1))
    ledger = Ledger()
    hit: dict[str, list[int]] = defaultdict(list)
    brier: dict[str, list[float]] = defaultdict(list)
    t0 = time.time()
    for r in rows:
        tgt = r["target"]
        gold = max(tgt, key=tgt.get)
        if r["kind"] == "choice":
            q = Choice(r["question"], r["options"], decision_point=r["dp"])
        else:
            q = Noul(r["question"], decision_point=r["dp"])
        ans = backend.decide(r["state"], [q], ledger).answers[0]
        if r["kind"] == "choice":
            pred = ans.value
            p_gold = float(ans.distribution.get(gold, 0.0))
            correct = pred == gold
        else:
            gold_b = str(gold).lower() in ("true", "yes", "1")
            p_yes = float(ans.distribution.get("yes", ans.distribution.get("True", ans.distribution.get(True, 0.0))))
            pred = bool(ans.value)
            p_gold = p_yes if gold_b else 1.0 - p_yes
            correct = pred == gold_b
        hit[r["dp"]].append(int(correct))
        brier[r["dp"]].append((1.0 - p_gold) ** 2)
    out = {"s1": json.loads(a.s1), "dev": a.dev, "n": len(rows), "wall_s": round(time.time() - t0, 1),
           "acc": sum(sum(v) for v in hit.values()) / len(rows),
           "by_dp": {k: {"n": len(v), "acc": sum(v) / len(v), "brier": sum(brier[k]) / len(v)} for k, v in hit.items()},
           "ledger": ledger.summary()}
    print(json.dumps({k: v for k, v in out.items() if k != "ledger"}, indent=1))
    s = out["ledger"].get("total", {})
    print("ledger:", {k: s.get(k) for k in ("calls", "input_tokens", "cost_usd", "latency_ms")})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
