#!/usr/bin/env python3
"""S2-L3a's telecom population (docs/PREREG_A7_STAGE2.md A7.15): the inventory of the pinned tau2-bench telecom `full`
pool, the 32 validation tasks, and (after validation) the frozen random sample. Reads task definitions only: no
simulation runs, no model is called, no outcome is read.

    # (a), (b): before any telecom simulation
    python scripts/telecom_stage2.py inventory          # -> configs/tau2/stage2/telecom_inventory.json
    # validation (adapter check + cost measurement) on the 32 reserved tasks, both arms, then (c), (d):
    python scripts/telecom_stage2.py sample --validation-spend V --cost-per-sim C   # -> telecom_task_ids.json

Rules: the eligible population is the pool minus exact duplicates (same task content apart from the id; the first id
in pool order is kept); one seeded order (seed 731, `random.Random(731).shuffle` of the eligible ids in pool order)
gives the 32 validation tasks (its first 32) and the sample (the next n): a simple random sample without replacement of
eligible, non-validation tasks. n = min(800, N_eligible - 32, floor((250 - V) / (2 x 1.25 x C))), V the actual
validation spend and C the mean cost per simulation across both validation arms (failed attempts included); no
success number of the validation tasks enters."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs/tau2/stage2"
INVENTORY = OUT_DIR / "telecom_inventory.json"
SAMPLE = OUT_DIR / "telecom_task_ids.json"
SEED, N_VALIDATION, N_MAX, BUDGET, RESERVE = 731, 32, 800, 250.0, 1.25
PINNED_COMMIT = "5bfa7e3"                         # docs/TAU2.md
ID_RE = re.compile(r"^\[(?P<issue>[^\]]+)\](?P<faults>.*)\[PERSONA:(?P<persona>[^\]]*)\]$")


def _data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    from tau2.utils import DATA_DIR

    return Path(DATA_DIR) / "tau2" / "domains" / "telecom"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def inventory(data: Path) -> dict[str, Any]:
    pool_p, split_p = data / "tasks_full.json", data / "split_tasks.json"
    pool = json.loads(pool_p.read_text())
    splits = json.loads(split_p.read_text())
    member = {s: set(map(str, ids)) for s, ids in splits.items()}
    rows, first_of, dup = [], {}, []
    for t in pool:
        tid = str(t["id"])
        m = ID_RE.match(tid)
        content = _sha(json.dumps({k: v for k, v in t.items() if k != "id"}, sort_keys=True).encode())
        persona = ((t.get("user_scenario") or {}).get("persona"))
        row = {"id": tid, "splits": sorted(s for s, ids in member.items() if tid in ids),
               "issue_type": m.group("issue") if m else None, "faults": m.group("faults") if m else None,
               "persona": m.group("persona") if m else persona, "content_sha256": content}
        if content in first_of:
            row["duplicate_of"] = first_of[content]
            dup.append(tid)
        else:
            first_of[content] = tid
        rows.append(row)
    ids = [r["id"] for r in rows]
    eligible = [r["id"] for r in rows if "duplicate_of" not in r]
    order = list(eligible)
    random.Random(SEED).shuffle(order)
    return {"registered": "PREREG_A7_STAGE2 A7.15 (a), (b)",
            "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "source": {"repo": "https://github.com/sierra-research/tau2-bench", "commit": PINNED_COMMIT,
                       "files": {p.name: _sha(p.read_bytes()) for p in (pool_p, split_p)}},
            "n_pool": len(ids), "n_distinct_ids": len(set(ids)), "n_exact_duplicates": len(dup),
            "n_eligible": len(eligible), "issue_types": dict(Counter(r["issue_type"] for r in rows)),
            "personas": dict(Counter(r["persona"] for r in rows)),
            "split_sizes": {s: len(v) for s, v in member.items()},
            "distinct_fault_persona": len({(r["faults"], r["persona"]) for r in rows}),
            "validation": {"seed": SEED, "n": N_VALIDATION, "rule": "the first 32 of random.Random(731).shuffle(eligible "
                           "ids in pool order); reserved for adapter validation and cost measurement, never evaluated",
                           "task_ids": order[:N_VALIDATION]},
            "eligible_sha256": _sha("\n".join(eligible).encode()), "tasks": rows}


def sample_size(n_eligible: int, validation_spend: float, cost_per_sim: float) -> int:
    """A7.15 (c): min(800, N_eligible - 32, floor((250 - V) / (2 x 1.25 x C)))."""
    if cost_per_sim <= 0:
        raise ValueError("cost per simulation must be positive")
    by_budget = math.floor((BUDGET - validation_spend) / (2 * RESERVE * cost_per_sim))
    return max(0, min(N_MAX, n_eligible - N_VALIDATION, by_budget))


def sample(inv: dict[str, Any], validation_spend: float, cost_per_sim: float) -> dict[str, Any]:
    eligible = [r["id"] for r in inv["tasks"] if "duplicate_of" not in r]
    if _sha("\n".join(eligible).encode()) != inv["eligible_sha256"]:
        raise SystemExit("the inventory's eligible list does not match its recorded sha256")
    order = list(eligible)
    random.Random(SEED).shuffle(order)
    if order[:N_VALIDATION] != inv["validation"]["task_ids"]:
        raise SystemExit("the seeded order no longer reproduces the reserved validation tasks")
    n = sample_size(len(eligible), validation_spend, cost_per_sim)
    ids = order[N_VALIDATION:N_VALIDATION + n]
    by = {r["id"]: r for r in inv["tasks"]}
    return {"registered": "PREREG_A7_STAGE2 A7.15 (c), (d)",
            "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "inventory_sha256": inv.get("_sha256"), "seed": SEED,
            "rule": {"validation_spend_usd": validation_spend, "cost_per_sim_usd": cost_per_sim, "n_max": N_MAX,
                     "budget_usd": BUDGET, "reserve": RESERVE, "n_eligible": len(eligible), "n": n},
            "issue_types": dict(Counter(by[i]["issue_type"] for i in ids)),
            "in_base_split": sum(1 for i in ids if "base" in by[i]["splits"]),
            "sha256": _sha("\n".join(ids).encode()), "task_ids": ids}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("--data", default=None, help="tau2 telecom data dir (default: the installed tau2's)")
    i.add_argument("--out", default=str(INVENTORY))
    s = sub.add_parser("sample")
    s.add_argument("--inventory", default=str(INVENTORY))
    s.add_argument("--validation-spend", type=float, required=True, help="V: actual validation spend in $")
    s.add_argument("--cost-per-sim", type=float, required=True, help="C: mean $ per simulation over both arms")
    s.add_argument("--out", default=str(SAMPLE))
    a = ap.parse_args(argv)
    if a.cmd == "inventory":
        inv = inventory(_data_dir(a.data))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(inv, indent=1) + "\n")
        print(f"[telecom] pool {inv['n_pool']}, eligible {inv['n_eligible']}, duplicates {inv['n_exact_duplicates']}, "
              f"issue types {inv['issue_types']} -> {a.out}", file=sys.stderr)
        return 0
    p = Path(a.inventory)
    inv = json.loads(p.read_text())
    inv["_sha256"] = _sha(p.read_bytes())
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: the sample is frozen once (A7.15 (d))")
    rec = sample(inv, a.validation_spend, a.cost_per_sim)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[telecom] n = {rec['rule']['n']} tasks, sha256 {rec['sha256'][:12]} -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
