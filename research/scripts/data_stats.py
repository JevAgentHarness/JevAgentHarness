"""Dataset statistics quoted in the paper (no model calls), written where paper_numbers picks them up.

    python scripts/data_stats.py --out runs/data_stats

Writes runs/data_stats/summary.json with, per BFCL sample (the E1 sample of configs/decision/bfcl_when2call.yaml
and the full categories):
  n_tool_items, n_param_slots, in_candidates (share of argument slots whose gold value is among the offered
  candidates, i.e. reachable without generation), n_skipped_parallel/none
and the When2Call label distribution. Keys in the paper: \\res{data_stats/<sample>/<field>}.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import yaml

from jevharness.bench.suites.static import BFCL_TOOL_CATS, load_bfcl, load_when2call


def bfcl_stats(**kw) -> dict:
    items = load_bfcl(**kw)
    params = [i for i in items if i.dp == "tool_params"]
    inc = [bool((i.meta or {}).get("gold_in_candidates", True)) for i in params]
    by_cat = collections.Counter(i.source for i in items if i.dp == "tool_select")
    return {"n_tool_items": sum(by_cat.values()), "n_param_slots": len(params),
            "in_candidates": sum(inc) / len(inc) if inc else float("nan"),
            "by_category": dict(sorted(by_cat.items()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/data_stats")
    ap.add_argument("--config", default="configs/decision/bfcl_when2call.yaml")
    a = ap.parse_args()
    exp = yaml.safe_load(open(a.config))
    bfcl_kw = next({k: v for k, v in s.items() if k != "name"} for s in exp["suites"] if s["name"] == "bfcl")
    w2c_kw = next({k: v for k, v in s.items() if k != "name"} for s in exp["suites"] if s["name"] == "when2call")
    out = {"bfcl_e1": bfcl_stats(**bfcl_kw),
           "bfcl_full": bfcl_stats(categories=list(BFCL_TOOL_CATS))}
    w2c = load_when2call(**w2c_kw)
    labels = collections.Counter(i.gold[0] for i in w2c)
    out["when2call_e1"] = {"n": len(w2c), "label_share": {k: v / len(w2c) for k, v in sorted(labels.items())}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
