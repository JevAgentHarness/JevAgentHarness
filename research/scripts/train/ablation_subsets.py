#!/usr/bin/env python3
"""Exploratory SFT training ablation (plan: results/training/ablation_plan.md): the arm training files from data v1.

No API call, no GPU. Every arm is a subset and / or a relabelling of the SFT v1 training rows
(data/train_v1/train.jsonl.gz: 15,694 rows, mixed targets q = 0.3 gold + 0.7 p_Jev, gold one-hot where Jev's argmax
is wrong); no new episode and no JevGym seed is drawn, so every training row keeps its data-v1 seed (>= 10000, outside
100000-100499 / 200000-200499 / 300000-300999 / 5000-5999; re-checked here with training/seed_guard.TRAIN_FORBIDDEN).

    python scripts/train/ablation_subsets.py plan                       # row counts and SFT minutes per arm (JSON)
    python scripts/train/ablation_subsets.py build --out runs/train_ablation/data \\
        [--teacher-only data/cache/train_v1/teacher_only.jsonl] [--arms D10 D25 ...]

Arms (ARMS): data size D10 / D25 / D50 (nested task subsets, stratified by data block v0 / v1, seed 0; mixed labels);
label source L_gold_* (one-hot gold) and L_teach_* (p_Jev alone, alpha 0, no gold fallback; needs --teacher-only,
the Jev relabel built from the teacher cache at $0, see the plan) on all rows (100) and on D25's tasks (25);
state distribution S_canon (rows with no earlier off-plan step) vs S_match (a random row subsample of the same size,
seed 0), and the same pair inside D25's rows (S_canon_25 = D25's rows with no earlier off-plan step, S_match_25 = a
random subsample of D25's rows of the same size, seed 0; addendum 2026-10-05 of the plan); D100 = SFT v1's data and
labels retrained on the ablation host (the within-stack reference of every contrast; the existing SFT v1 checkpoint is
evaluated beside it as the reproduction anchor); tier 2: S_nonm (without the scripted near-miss arm) vs S_nonm_match.
The When2Call replay is mixed in by scripts/train/10_sft.sh with REPLAY_FACTOR = 0.45 x rows / 15,694 (SFT v1's
replay share); the plan prints it per arm. CORE / EXTENDED: the budget split and execution order of the plan's
addendum (a "+" binds arms that are only useful together, so the driver's cap check admits them as one unit; "_s1" =
the same data file with trainer seed 1).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "train_v1" / "train.jsonl.gz"
REPLAY_ROWS = 2685            # data/train_v1/replay_when2call.jsonl.gz
REPLAY_FACTOR_V1 = 0.45       # SFT v1's replay factor
SFT_V1_ROWS = 16902           # 15,694 train + round(0.45 x 2,685) replay
SFT_V1_MIN = 134              # results/gpu_round5/status.txt: stage sft exit=0 (134 min), 2 epochs on one L40S
LOAD_MIN = 3                  # model load + adapter save per run (planning allowance)
SEED = 0


# ------------------------------------------------------------------ row helpers
def load(path: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def task_of(r: dict[str, Any]) -> str:
    return r["id"].split("/")[0]


def block_of(r: dict[str, Any]) -> str:
    """v0 = the data-v0 rows carried into v1 (Flash shadow, seeds 10000-12059); v1 = the new harness episodes."""
    return "v0" if (r.get("meta") or {}).get("arm") == "v0_flash_shadow" else "v1"


def is_recovery(r: dict[str, Any]) -> bool:
    return bool((r.get("meta") or {}).get("after_error") or 0)


def gold_of(r: dict[str, Any]) -> str:
    m, t = r.get("meta") or {}, r["target"]
    g = m.get("gold") if m.get("label_source") == "teacher" else max(t, key=t.get)
    if g not in r["options"]:
        raise ValueError(f"{r['id']}: gold {g!r} not among the options")
    return g


def to_gold(r: dict[str, Any]) -> dict[str, Any]:
    g = gold_of(r)
    meta = {k: v for k, v in (r.get("meta") or {}).items() if k not in ("teachers", "teacher_agree", "alpha",
                                                                         "teacher_dists", "teacher_outcome")}
    return {**r, "target": {o: float(o == g) for o in r["options"]}, "meta": {**meta, "label_source": "gold",
                                                                              "gold": g, "ablation_labels": "gold"}}


def task_subset(rows: list[dict[str, Any]], frac: float, seed: int = SEED) -> set[str]:
    """The first round(frac x n_block) tasks of each block's seeded shuffle: nested in frac, stratified by block."""
    keep: set[str] = set()
    for blk in ("v0", "v1"):
        tasks = sorted({task_of(r) for r in rows if block_of(r) == blk})
        random.Random(f"{seed}:{blk}").shuffle(tasks)
        keep |= set(tasks[: int(round(frac * len(tasks)))])
    return keep


def row_sample(rows: list[dict[str, Any]], n: int, seed: int = SEED) -> list[dict[str, Any]]:
    idx = sorted(random.Random(f"{seed}:rows").sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def check_seeds(rows: Iterable[dict[str, Any]]) -> None:
    from jevharness.training.seed_guard import TRAIN_FORBIDDEN, TRAIN_MIN_SEED

    for r in rows:
        s = (r.get("meta") or {}).get("jevgym_seed")
        if s is None:
            continue
        if s < TRAIN_MIN_SEED or any(lo <= s <= hi for lo, hi in TRAIN_FORBIDDEN):
            raise ValueError(f"{r['id']}: JevGym seed {s} is not a training seed")


# ------------------------------------------------------------------ arms
Selector = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


def _tasks(frac: float) -> Selector:
    def sel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keep = task_subset(rows, frac)
        return [r for r in rows if task_of(r) in keep]
    return sel


def _all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(rows)


def _canon(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not is_recovery(r)]


def _nonm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if (r.get("meta") or {}).get("arm") != "nearmiss_replace"]


def _match(other: Selector) -> Selector:
    def sel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return row_sample(rows, len(other(rows)))
    return sel


def _within(base: Selector, inner: Selector) -> Selector:
    """``inner`` applied to the rows that ``base`` selects (e.g. the state pair inside D25's rows)."""
    def sel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return inner(base(rows))
    return sel


# arm -> (factor, row selector, labels, tier)
ARMS: dict[str, tuple[str, Selector, str, int]] = {
    "D10": ("size", _tasks(0.10), "mixed", 1),
    "D25": ("size", _tasks(0.25), "mixed", 1),
    "D50": ("size", _tasks(0.50), "mixed", 1),
    "D100": ("size", _all, "mixed", 1),            # SFT v1's data, retrained on the ablation host (within-stack reference)
    "L_gold_100": ("labels", _all, "gold", 1),
    "L_teach_100": ("labels", _all, "teacher", 1),
    "L_gold_25": ("labels", _tasks(0.25), "gold", 1),
    "L_teach_25": ("labels", _tasks(0.25), "teacher", 1),
    "S_canon": ("states", _canon, "mixed", 1),
    "S_match": ("states", _match(_canon), "mixed", 1),
    "S_canon_25": ("states", _within(_tasks(0.25), _canon), "mixed", 1),
    "S_match_25": ("states", _within(_tasks(0.25), _match(_canon)), "mixed", 1),
    "S_nonm": ("states", _nonm, "mixed", 2),
    "S_nonm_match": ("states", _match(_nonm), "mixed", 2),
}
ANCHOR = "sft_v1"   # the existing SFT v1 checkpoint (no training); evaluated first, the pipeline's go / no-go
# budget split of the plan's addendum (2026-10-05), in execution order after the anchor; "+" = one cap unit
CORE = ["D25", "L_gold_25", "L_teach_25", "S_canon_25+S_match_25", "D10", "D50"]
EXTENDED = ["D25_s1", "D10_s1", "D100", "L_gold_100+L_teach_100", "S_canon+S_match"]


def data_arm(arm: str) -> str:
    """The data file of a training arm: the seed replicates ``<arm>_s<k>`` reuse ``<arm>``'s rows."""
    base, _, sfx = arm.rpartition("_s")
    return base if base and sfx.isdigit() else arm


def trainer_seed(arm: str) -> int:
    base, _, sfx = arm.rpartition("_s")
    return int(sfx) if base and sfx.isdigit() else 0


def units(spec: list[str] | str) -> list[list[str]]:
    """'a,b+c' or ['a', 'b+c'] -> [['a'], ['b', 'c']] (every arm checked against ARMS via data_arm)."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out = [[x for x in u.split("+") if x] for u in items if u.strip()]
    bad = [a for u in out for a in u if data_arm(a) not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; known {list(ARMS)} (+ '_s<k>' seed replicates)")
    return out


def arm_rows(arm: str, rows: list[dict[str, Any]], teacher: dict[str, dict[str, Any]] | None = None
             ) -> list[dict[str, Any]]:
    _, sel, labels, _ = ARMS[arm]
    out = sel(rows)
    if labels == "gold":
        out = [to_gold(r) for r in out]
    elif labels == "teacher":
        if teacher is None:
            raise SystemExit(f"{arm}: needs --teacher-only (the Jev relabel with alpha 0; see the plan)")
        bad = [r["id"] for r in out if r["id"] not in teacher
               or (teacher[r["id"]].get("meta") or {}).get("alpha", None) not in (0, 0.0)]
        if bad:
            raise SystemExit(f"{arm}: {len(bad)} rows without an alpha-0 Jev target, e.g. {bad[:3]}")
        out = [{**teacher[r["id"]], "meta": {**(teacher[r["id"]].get("meta") or {}), "ablation_labels": "teacher"}}
               for r in out]
    return out


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rec = sum(is_recovery(r) for r in rows)
    by_dp: dict[str, int] = {}
    for r in rows:
        by_dp[r["dp"]] = by_dp.get(r["dp"], 0) + 1
    replay_factor = REPLAY_FACTOR_V1 * n / 15694
    total = n + int(round(replay_factor * REPLAY_ROWS))
    return {"rows": n, "tasks": len({task_of(r) for r in rows}), "recovery_rows": rec,
            "recovery_share": round(rec / n, 4) if n else None, "by_dp": dict(sorted(by_dp.items())),
            "replay_factor": round(replay_factor, 4), "rows_with_replay": total,
            "sft_min_est": int(math.ceil(SFT_V1_MIN * total / SFT_V1_ROWS + LOAD_MIN))}


def plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"source": str(TRAIN.relative_to(ROOT)), "rows": len(rows), "seed": SEED,
           "D100 (= SFT v1, existing checkpoint)": describe(rows), "arms": {}}
    for arm, (factor, _, labels, tier) in ARMS.items():
        out["arms"][arm] = {"factor": factor, "labels": labels, "tier": tier, **describe(ARMS[arm][1](rows))}
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def content_sha(p: Path) -> str:
    """sha256 of the file's bytes after gzip decompression (the same for x.jsonl and x.jsonl.gz)."""
    op = gzip.open if str(p).endswith(".gz") else open
    h = hashlib.sha256()
    with op(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(rows: list[dict[str, Any]], out: Path, arms: list[str], teacher_path: Path | None,
          train_path: Path = TRAIN) -> dict[str, Any]:
    check_seeds(rows)
    teacher = {r["id"]: r for r in load(teacher_path)} if teacher_path else None
    out.mkdir(parents=True, exist_ok=True)
    man: dict[str, Any] = {"source": str(train_path), "source_sha256": _sha(train_path) if train_path.exists() else None,
                           "seed": SEED,
                           "teacher_only": str(teacher_path) if teacher_path else None,
                           "teacher_only_sha256": _sha(teacher_path) if teacher_path else None,
                           "teacher_only_content_sha256": content_sha(teacher_path) if teacher_path else None,
                           "arms": {}}
    for arm in arms:
        if ARMS[arm][2] == "teacher" and teacher is None:
            man["arms"][arm] = {"skipped": "no --teacher-only file"}
            continue
        rs = arm_rows(arm, rows, teacher)
        p = out / f"{arm}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
        man["arms"][arm] = {**describe(rs), "file": p.name, "sha256": _sha(p)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--train", default=str(TRAIN))
    b = sub.add_parser("build")
    b.add_argument("--train", default=str(TRAIN))
    b.add_argument("--out", required=True)
    b.add_argument("--arms", nargs="+", default=[a for a, s in ARMS.items() if s[3] == 1])
    b.add_argument("--teacher-only", default=None, help="Jev relabel of train_gold.jsonl with alpha 0 (see the plan)")
    u = sub.add_parser("units", help="print the cap units of an arm list, one per line (arms space-separated)")
    u.add_argument("spec", help="core | extended | a comma list, '+' binding arms into one unit (e.g. D25,S_canon+S_match)")
    a = ap.parse_args(argv)
    if a.cmd == "units":
        spec = {"core": CORE, "extended": EXTENDED}.get(a.spec, a.spec)
        print("\n".join(" ".join(x) for x in units(spec)))
        return 0
    rows = load(Path(a.train))
    if a.cmd == "plan":
        print(json.dumps(plan(rows), indent=1))
        return 0
    unknown = [x for x in a.arms if x not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known {list(ARMS)}")
    man = build(rows, Path(a.out), a.arms, Path(a.teacher_only) if a.teacher_only else None, Path(a.train))
    print(json.dumps({k: {kk: v.get(kk) for kk in ("rows", "tasks", "recovery_rows", "replay_factor", "skipped")}
                      for k, v in man["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
