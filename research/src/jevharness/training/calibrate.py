"""Post-hoc temperature scaling for a trained decision model (global + per option-count bucket).

    python -m jevharness.training.calibrate --model Qwen/Qwen3-4B --adapter checkpoints/jev4b-sft \
        --dev data/decisions/dev.jsonl --out checkpoints/jev4b-sft/calibration.json

Fit on held-out *tasks/domains* and refit after every policy change (the state distribution of an
agent shifts when its decision policy changes).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Sequence

import numpy as np


def nll(logits: Sequence[np.ndarray], gold: Sequence[int], t: float) -> float:
    tot = 0.0
    for z, g in zip(logits, gold):
        zt = z / t
        m = zt.max()
        tot += -(zt[g] - m - math.log(np.exp(zt - m).sum()))
    return tot / max(1, len(gold))


def fit_temperature(logits: Sequence[np.ndarray], gold: Sequence[int], lo: float = 0.05, hi: float = 20.0) -> float:
    """Golden-section search on log T (NLL is unimodal in T for a fixed set of logits)."""
    a, b = math.log(lo), math.log(hi)
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        if nll(logits, gold, math.exp(c)) < nll(logits, gold, math.exp(d)):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return math.exp((a + b) / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-bucket", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16, help="scoring batch (the Engine default; smaller on a small GPU)")
    a = ap.parse_args()
    from .data import read_jsonl
    from .serve_jev import Engine, _bucket

    eng = Engine(a.model, a.adapter, batch_size=a.batch_size)
    ex = [e for e in read_jsonl(a.dev) if 2 <= len(e["options"]) <= 52]
    items = [(e["state"], e["question"], e["options"], e.get("kind", "choice")) for e in ex]
    logits, gold, keys = [], [], []
    import torch  # noqa: F401  (Engine already imported it)

    probs = eng._probs(items)  # at T=1 -> recover log-probs as logits (softmax-invariant)
    for e, p in zip(ex, probs):
        logits.append(np.log(np.clip(np.asarray(p), 1e-12, 1)))
        gold.append(int(np.argmax([e["target"].get(o, 0.0) for o in e["options"]])))
        keys.append(f"{e.get('kind', 'choice')}:{_bucket(len(e['options']))}")
    cal = {"global": fit_temperature(logits, gold), "buckets": {}}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    for k, idx in groups.items():
        if len(idx) >= a.min_bucket:
            cal["buckets"][k] = fit_temperature([logits[i] for i in idx], [gold[i] for i in idx])
    json.dump(cal, open(a.out, "w"), indent=1)
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
