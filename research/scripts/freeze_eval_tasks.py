#!/usr/bin/env python3
"""Freeze the evaluation task list (PREREGISTRATION A6.13, A6.16 (l)): the 500 JevGym task ids the E2 configs' suite
produces, with a sha256 over the id list, the sha256 of the task-generating suite block and the generator's file hash
and last commit (bench/suites/jevgym.py), written to configs/eval/jevgym_eval_tasks.json. scripts/prereg_check.py
verifies that the configs still reproduce the ids and that neither the suite block nor the generator changed.   python scripts/freeze_eval_tasks.py [--config configs/experiments/jevgym_tool_select.yaml]

Stage 2 (docs/PREREG_A7_STAGE2.md, A7.2 (a)): the fresh seeds 300000-300499 of configs/experiments/stage2_jevgym.yaml
    python scripts/freeze_eval_tasks.py --stage2
writes configs/eval/jevgym_stage2_tasks.json (task generation only: no episode runs, nothing is shown to a model)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/jevgym_tool_select.yaml")
    ap.add_argument("--out", default="configs/eval/jevgym_eval_tasks.json")
    ap.add_argument("--stage2", action="store_true", help="freeze the Stage-2 list (A7.2 (a)): config "
                    "configs/experiments/stage2_jevgym.yaml, range 300000-300499, out configs/eval/jevgym_stage2_tasks.json")
    a = ap.parse_args()
    from jevharness.bench.registry import EVAL_SEED_RANGES, STAGE2_EVAL_RANGES
    from jevharness.bench.suites import load_suite

    registered = "PREREGISTRATION A6.13, A6.16 (l)"
    if a.stage2:
        a.config, a.out = "configs/experiments/stage2_jevgym.yaml", "configs/eval/jevgym_stage2_tasks.json"
        EVAL_SEED_RANGES, registered = STAGE2_EVAL_RANGES, "PREREG_A7_STAGE2 A7.2 (a)"  # noqa: N806

    exp = yaml.safe_load((ROOT / a.config).read_text())
    suite = dict(exp["suite"])
    seed0, n = int(suite["seed0"]), int(suite["n"])
    lo, hi = EVAL_SEED_RANGES[0]
    if seed0 != lo or seed0 + n - 1 != hi:
        raise SystemExit(f"{a.config}: suite seeds {seed0}-{seed0 + n - 1} are not the registered range {lo}-{hi}")
    facs = load_suite(suite.pop("name"), **suite)
    ids = [f().task_id for f in facs]
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    import subprocess
    gen = ROOT / "src" / "jevharness" / "bench" / "suites" / "jevgym.py"
    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(gen)], capture_output=True, text=True,
                                cwd=ROOT, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    out = {"registered": registered, "config": a.config, "suite": exp["suite"],
           "suite_sha256": hashlib.sha256(json.dumps(exp["suite"], sort_keys=True).encode()).hexdigest(),
           "generator": {"file": str(gen.relative_to(ROOT)), "sha256": hashlib.sha256(gen.read_bytes()).hexdigest(),
                         "commit": commit},
           "seed_range": [lo, hi], "n": len(ids), "sha256": digest, "task_ids": ids}
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")
    fam = {}
    for t in ids:
        fam[t.split("-")[1]] = fam.get(t.split("-")[1], 0) + 1
    print(f"wrote {a.out}: {len(ids)} ids, sha256 {digest}, families {fam}")


if __name__ == "__main__":
    main()
