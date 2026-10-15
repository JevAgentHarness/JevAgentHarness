#!/usr/bin/env python3
"""Freeze the C8 population (PREREGISTRATION A6.13, A6.16 (l)): the L1 items of configs/decision/bfcl_when2call.yaml
after the parent-item exclusion of the exposed dev-prefix rows, as per-source / per-decision-point counts and the
sha256 of the remaining ids, written to configs/eval/e1_c8_population.json BEFORE the C8 run. scripts/prereg_check.py
re-derives the counts from the config and fails on any difference.

    python scripts/c8_population.py [--config configs/decision/bfcl_when2call.yaml] [--out configs/eval/e1_c8_population.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision/bfcl_when2call.yaml")
    ap.add_argument("--out", default="configs/eval/e1_c8_population.json")
    a = ap.parse_args()
    from jevharness.bench.decision_eval import load_population, load_static, population_counts

    exp = yaml.safe_load((ROOT / a.config).read_text())
    raw: list = []
    for s in exp["suites"]:
        s = dict(s)
        raw.extend(load_static(s.pop("name"), **s))
    items = load_population(exp)
    counts = population_counts(items)
    excl = json.loads((ROOT / exp["exclude_ids_file"]).read_text()) if exp.get("exclude_ids_file") else {"ids": []}
    out = {"registered": "PREREGISTRATION A6.13, A6.16 (l)", "config": a.config, "suites": exp["suites"],
           "exclude_ids_file": exp.get("exclude_ids_file"), "exclusion_rule": "by parent (source) item",
           "n_excluded_records_listed": len(excl["ids"]), "n_items_before_exclusion": len(raw),
           "generated": time.strftime("%Y-%m-%d"), **counts}
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}: {counts['n_items']} items ({counts['n_source_items']} source items) from {len(raw)}; "
          f"by source {counts['by_source']}; sha256 {counts['sha256'][:12]}")


if __name__ == "__main__":
    main()
