"""Decision-level evaluation of a checkpoint on held-out decision files: the numbers behind the go / no-go gates of
docs/TRAINING_RECIPES.md §10 and docs/TRAINING_AGENT_BRIEF.md §6.

  python -m jevharness.training.eval_decisions --model Qwen/Qwen3-4B --adapter checkpoints/r2 \\
      --dev data/train_v0/dev.jsonl data/train_v0/dev_when2call.jsonl.gz --out checkpoints/r2/eval_dev.json

  # R0 (zero-shot base model): leave --adapter out.  Full-FT checkpoints: --adapter <dir> works too.

Scoring goes through ``serve_jev.Engine`` -- the exact prompt, readout, knockout and calibration of the served model
(``--calibration``) -- so these numbers describe what the harness will see. Per file, per decision point and per
kind: n, acc (argmax vs the gold = argmax of the target), nll (-sum_a q_a log p_a), brier (top-label confidence vs
correctness), ece (10 equal-width bins), ece_adaptive, auroc (confidence -> correct), aurc, and the option-order
**flip rate** (``--n-perm`` random permutations; share of decisions whose chosen option changes), plus wall time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import read_jsonl


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from ..bench.metrics import aurc, auroc, brier, ece, ece_adaptive

    conf = [r["conf"] for r in rows]
    corr = [float(r["correct"]) for r in rows]
    out: dict[str, Any] = {"n": len(rows)}
    if not rows:
        return out
    out.update(acc=sum(corr) / len(rows), nll=sum(r["nll"] for r in rows) / len(rows), brier=brier(conf, corr),
               ece=ece(conf, corr), ece_adaptive=ece_adaptive(conf, corr), auroc=auroc(conf, corr),
               aurc=aurc(conf, corr), mean_conf=sum(conf) / len(rows))
    flips = [r["flip"] for r in rows if r.get("flip") is not None]
    if flips:
        out["flip_rate"] = sum(flips) / len(flips)
    return {k: ((round(v, 5) if math.isfinite(v) else None) if isinstance(v, float) else v) for k, v in out.items()}


def evaluate(engine: Any, examples: list[dict[str, Any]], n_perm: int = 1, seed: int = 0) -> dict[str, Any]:
    """Score ``examples`` (training format) with ``engine`` (a ``serve_jev.Engine``). Returns overall / by_dp /
    by_kind metrics (see the module docstring)."""
    rng = random.Random(seed)
    ex = [e for e in examples if 2 <= len(e.get("options") or []) <= 52 and e.get("target")]  # harvest caps at 52
    items = [(e["state"], e["question"], e["options"], e.get("kind", "choice")) for e in ex]
    t0 = time.perf_counter()
    probs = engine._probs(items)
    secs = time.perf_counter() - t0
    perm_pred: list[list[int]] = [[] for _ in ex]
    for _ in range(n_perm):
        orders = [rng.sample(range(len(e["options"])), len(e["options"])) for e in ex]
        pitems = [(s, q, [o[j] for j in order], k) for (s, q, o, k), order in zip(items, orders)]
        pp = engine._probs(pitems)
        for i, (p, order) in enumerate(zip(pp, orders)):
            perm_pred[i].append(order[max(range(len(p)), key=lambda j: p[j])])
    rows = []
    for i, (e, p) in enumerate(zip(ex, probs)):
        q = [float(e["target"].get(o, 0.0)) for o in e["options"]]
        z = sum(q) or 1.0
        q = [v / z for v in q]
        gold = max(range(len(q)), key=lambda j: q[j])
        pred = max(range(len(p)), key=lambda j: p[j])
        nll = -sum(v * math.log(max(p[j], 1e-12)) for j, v in enumerate(q) if v)
        rows.append({"dp": str(e.get("dp")), "kind": str(e.get("kind", "choice")), "conf": float(p[pred]),
                     "correct": pred == gold, "nll": nll,
                     "flip": (sum(x != pred for x in perm_pred[i]) / n_perm) if n_perm else None})
    by_dp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dp[r["dp"]].append(r)
        by_kind[r["kind"]].append(r)
    return {"overall": _metrics(rows), "by_dp": {k: _metrics(v) for k, v in sorted(by_dp.items())},
            "by_kind": {k: _metrics(v) for k, v in sorted(by_kind.items())},
            "seconds": round(secs, 2), "decisions_per_s": round(len(rows) / secs, 2) if secs > 0 else None}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Decision-level evaluation of a checkpoint (gates of the training plan)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter or full fine-tuned checkpoint dir ('' = base)")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--dev", nargs="+", required=True, help="held-out decision files (.jsonl / .jsonl.gz)")
    ap.add_argument("--limit", type=int, default=None, help="first N decisions per file")
    ap.add_argument("--n-perm", type=int, default=1, help="random option orders for the flip rate (0 = skip)")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON output (default: print only)")
    a = ap.parse_args(argv)
    from .serve_jev import Engine

    eng = Engine(a.model, a.adapter or None, a.calibration, name="eval", max_len=a.max_len, batch_size=a.batch_size)
    res: dict[str, Any] = {"model": a.model, "adapter": a.adapter or None, "calibration": a.calibration,
                           "n_perm": a.n_perm, "files": {}}
    for f in a.dev:
        ex = read_jsonl(f)[: a.limit] if a.limit else read_jsonl(f)
        res["files"][f] = evaluate(eng, ex, a.n_perm, a.seed)
        print(json.dumps({f: res["files"][f]["overall"]}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    main()
