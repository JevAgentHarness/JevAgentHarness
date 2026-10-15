"""Write ``MANIFEST.json`` for a decision-data release directory (``scripts/train/make_data_v0.sh``).

  python scripts/train/_manifest_v0.py data/train_v0 [--max-mb 15]

Records, per file: bytes, sha256, rows, and for training-format files the counts by source / decision point / kind /
option bucket, the JevGym seed range and the label sources; for the trajectory dirs: the suite, the System-2 and
System-1 models, episodes, success and the ledger cost; for teacher files: the teacher spec and its Ledger summary.
Exits non-zero when the directory is larger than ``--max-mb`` or a JevGym evaluation seed (< 1000) slipped in.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.training.data import open_text, read_jsonl, stats  # noqa: E402

MIN_TRAIN_SEED = 1000  # JevGym seeds below this are evaluation / pilot seeds (E2 uses 0..199)
SEED_RE = re.compile(r"jevgym-[a-z0-9_]+-(\d+)-")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jevgym_seeds(ids: list[str]) -> list[int]:
    return sorted({int(m.group(1)) for i in ids if (m := SEED_RE.match(i))})


def describe_examples(path: Path) -> dict[str, Any]:
    ex = read_jsonl(path)
    seeds = jevgym_seeds([e["id"] for e in ex])
    st = stats(ex)
    return {"rows": len(ex), "tasks": st["tasks"], "by_source": dict(st["by_source"]), "by_dp": dict(st["by_dp"]),
            "by_kind": dict(st["by_kind"]), "by_options": dict(st["by_options"]),
            "label_sources": dict(Counter(str((e.get("meta") or {}).get("label_source")) for e in ex)),
            "soft_target_frac": round(st["soft_frac"], 4), "with_q_values": st["with_q_values"],
            "jevgym_seed_range": [seeds[0], seeds[-1]] if seeds else None, "jevgym_tasks": len(seeds)}


def describe_run(d: Path) -> dict[str, Any]:
    exp = yaml.safe_load((d / "experiment.yaml").read_text())
    out: dict[str, Any] = {"suite": exp.get("suite"), "system2": exp.get("llm"), "system1_shadow": exp.get("s1"),
                           "arms": [a.get("name") for a in exp.get("arms") or []], "files": {}}
    for f in sorted(d.glob("*.jsonl*")):
        n = succ = 0
        cost = cost_s2 = 0.0
        with open_text(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                t = json.loads(line)
                n += 1
                succ += bool((t.get("result") or {}).get("success"))
                u = t.get("usage") or {}
                cost += float((u.get("total_with_shadow") or u.get("total") or {}).get("cost_usd") or 0.0)
                cost_s2 += float((u.get("total") or {}).get("cost_usd") or 0.0)
        out["files"][f.name] = {"episodes": n, "success": succ, "cost_usd_total_with_shadow": round(cost, 4),
                                "cost_usd_system2": round(cost_s2, 4)}
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def build(out_dir: Path, notes: dict[str, Any] | None = None) -> dict[str, Any]:
    files: dict[str, Any] = {}
    seeds_all: list[int] = []
    runs: dict[str, Any] = {}
    cost: dict[str, float] = {}
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.json":
            continue
        rel = str(p.relative_to(out_dir))
        info: dict[str, Any] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        if p.name.endswith((".jsonl", ".jsonl.gz")) and not rel.startswith("trajectories/"):
            info.update(describe_examples(p))
            seeds_all += jevgym_seeds([e["id"] for e in read_jsonl(p)])
        if p.name.endswith(".summary.json") and "executor" in (s := json.loads(p.read_text())):  # mc_rollouts
            info["mc"] = s
            cost[f"mc:{rel}"] = round(float(s.get("cost_usd") or 0.0), 4)
        elif p.name.endswith(".summary.json"):  # teacher_label
            info["teachers"] = {n: {"spec": t.get("spec"), "acc_vs_gold": t.get("acc"), "nll": t.get("nll"),
                                    "ece": t.get("ece"), "answered": t.get("answered"),
                                    "cost_usd": t.get("ledger", {}).get("cost_usd"),
                                    "calls": t.get("ledger", {}).get("calls")}
                                for n, t in (s.get("teachers") or {}).items()}
            info["outcomes"] = s.get("outcomes")
            cost[f"teacher:{rel}"] = round(sum(t.get("ledger", {}).get("cost_usd") or 0.0
                                               for t in (s.get("teachers") or {}).values()), 4)
        files[rel] = info
    for d in sorted({p.parent for p in out_dir.glob("trajectories/*/experiment.yaml")}):
        runs[d.name] = describe_run(d)
        for fn, r in runs[d.name]["files"].items():
            cost[f"episodes:{d.name}/{fn}"] = r["cost_usd_total_with_shadow"]
    total = sum(v["bytes"] for v in files.values())
    bad = sorted({s for s in seeds_all if s < MIN_TRAIN_SEED})
    return {"name": out_dir.name, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(), "total_bytes": total, "total_mb": round(total / 2**20, 3),
            "holdout": {"min_jevgym_seed": MIN_TRAIN_SEED, "eval_seeds_found": bad,
                        "rule": "JevGym seeds < 1000 are evaluation/pilot seeds (E2: 0..199) and never appear here; "
                                "no BFCL or When2Call TEST item is included"},
            "files": files, "trajectory_runs": runs,
            "cost_usd": {**cost, "total": round(sum(cost.values()), 4)}, **({"notes": notes} if notes else {})}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--max-mb", type=float, default=15.0)
    ap.add_argument("--notes", default=None, help="JSON file merged into the manifest as `notes`")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    notes = json.loads(Path(a.notes).read_text()) if a.notes else None
    m = build(out, notes)
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1, default=str) + "\n")
    print(json.dumps({"total_mb": m["total_mb"], "files": {k: v.get("rows", v["bytes"]) for k, v in m["files"].items()},
                      "cost_usd": m["cost_usd"]["total"], "eval_seeds_found": m["holdout"]["eval_seeds_found"]},
                     indent=1))
    if m["holdout"]["eval_seeds_found"]:
        sys.exit(f"JevGym evaluation seeds found: {m['holdout']['eval_seeds_found'][:10]}")
    if m["total_mb"] > a.max_mb:
        sys.exit(f"{out} is {m['total_mb']} MB > {a.max_mb} MB")


if __name__ == "__main__":
    main()
