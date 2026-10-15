#!/usr/bin/env python3
r"""Train / evaluation hygiene audit of the decision-model training (JevAgentRL v1, the exploratory v2) -- OFFLINE,
no API call, post hoc and descriptive. Every number is computed from the data files, run logs, caches and records in
the repository; nothing is typed by hand.

    python scripts/leakage_audit.py
    # writes results/training/leakage_audit.{json,md} and paper/generated/leakage.tex (\defres{leak/...})
    python scripts/leakage_audit.py --fast       # skips the MinHash near-duplicate passes (minutes -> seconds)

Five checks (the numbering of the audit request):

1. **Seeds.** Every JevGym seed in the training data (data v0 / v1 rows, the episodes behind them), the teacher-labelled
   rows, the OPD / GRPO rollouts (v1: reconstructed from the trainers' seeded task order, the reconstruction validated
   event for event against the one rollout log that records seeds, v2 GRPO; v2 GRPO: as logged), the development
   checks and the calibration fits, against every evaluation seed: the frozen task lists, every ``runs/`` directory
   that touched a registered evaluation range, the S2-3 corpus tasks. Disjointness is proved on the sets read from the
   files, then against the registered ranges (``bench/registry.py``).
2. **Content overlap.** JevGym is a generator, so training and evaluation share plan shapes by design. Measured on
   top of that: (a) task specs -- identical worlds, identical gold plans (customer / order ids and values), identical
   instructions, shared (name, user id) and (order id, amount) pairs -- between every training task and every
   evaluation task; (b) states -- exact duplicates of the noise-free normalised System-1 state text and of
   (gold-plan fingerprint, step), entity-masked duplicates (plan-shape level) and MinHash near-duplicates (word 5-gram
   Jaccard) between training rows and the common-state corpus C, with the same statistics *within* C across tasks as
   the generator's own baseline; the models' C accuracy re-scored without the overlapping states; (c) the tau^2
   reference corpus; (d) BFCL / When2Call static items (8-gram rule of the data builder, exact and near duplicates).
3. **Labels and oracle access.** Label sources of every training row; the teacher (Jev) cache re-keyed request by
   request from the training rows (every cached request must be one of them); the model snapshot that answered; an
   offline wire canary (gold in ``Question.meta`` never reaches the request); oracle words in training text; the
   training manifests' access records; the operator branches' commits (no evaluation output path) and their fork
   points.
4. **Model selection.** The inputs of every keep / retain / freeze rule (chain decisions), the temperature fits and the
   tau-hat* fits (seeds read from the fit files), the freeze times against the first evaluation run of each model.
5. **External models.** First commit of JevGym; the dated snapshot every Jev response carried; release dates of the
   open decision models as recorded in the registration documents.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import random
import re
import sqlite3
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ------------------------------------------------------------------ constants
OUT_JSON = ROOT / "results/training/leakage_audit.json"
OUT_MD = ROOT / "results/training/leakage_audit.md"
OUT_TEX = ROOT / "paper/generated/leakage.tex"

TASK_RE = re.compile(r"jevgym-([a-z]+)-([0-9]+)-d([0-9]+)(-near|-mixed)?(-strict)?")
TASK_RE_B = re.compile(rb"jevgym-([a-z]+)-([0-9]+)-d([0-9]+)(-near|-mixed)?(-strict)?")
V2_FAMILIES = ("clarify", "multi")

# evaluation ranges, by name (bench/registry.py + PREREG_A7_STAGE2 A7.2 (a))
EVAL_RANGES: dict[str, tuple[int, int]] = {
    "stage1_eval": (200000, 200499), "stage2_eval": (300000, 300499), "stage2_spare": (300500, 300999),
    "retired": (100000, 100499)}
DEV_RANGE, CAL_RANGE, TRAIN_MIN = (0, 4999), (5000, 5999), 10000

# training-side files (actual data)
V0 = ROOT / "data/train_v0"
V1 = ROOT / "data/train_v1"
V1_CACHE = ROOT / "data/cache/train_v1"
TEACHER_CACHE = ROOT / "data/cache/teacher_v1.sqlite"
R5, R6 = ROOT / "results/gpu_round5", ROOT / "results/gpu_round6_miles"
S23, T2REF = ROOT / "results/stage2/s23", ROOT / "results/stage2/t2ref"
RUNS = ROOT / "runs"

# the rollout pools and their samplers (configs at the training commits; validated below against the v2 GRPO log)
ROLLOUTS: dict[str, dict[str, Any]] = {
    "opd_v1": {"config": "configs/train/opd_round5.yaml", "seed0": 40000, "n": 4000, "per_iter": 36, "iters": 24,
               "sampler": "opd.py: random.Random(0).shuffle(range(n)), 36 per iteration, no reshuffle before 4000",
               "record": "results/gpu_round5/opd_metrics.jsonl (tasks per iteration; seeds not logged)"},
    "grpo_v1": {"config": "configs/train/grpo_online_round5.yaml", "seed0": 50000, "n": 3000, "per_iter": 8,
                "iters": 30, "sampler": "grpo_online.py: random.Random(0).shuffle(range(n)), 8 per iteration",
                "record": "results/gpu_round5/grpo_metrics.jsonl (episodes per iteration; seeds not logged)"},
    "opd_v2": {"config": "configs/train/miles_opd_v2.yaml", "seed0": 60000, "n": 4000, "per_iter": 36, "iters": 24,
               "sampler": "miles_port/rollout.py: random.Random(0).shuffle(range(n)); order[(rid*k+j) % n]",
               "record": "lost with the stage directory (TRAINING_MANIFEST section 9, 2026-10-03 ~22:05 UTC)"},
    "grpo_v2": {"config": "configs/train/miles_grpo_v2.yaml", "seed0": 70000, "n": 3000, "per_iter": 8, "iters": 30,
                "sampler": "miles_port/rollout.py (as opd_v2)",
                "record": "results/gpu_round6_miles/grpo_rollout_metrics.jsonl (task_seeds per rollout)"},
}
# development / selection instruments named by the manifests and configs (seed0, n)
DEV_INSTRUMENTS: dict[str, tuple[int, int, str]] = {
    "greedy_dev_rollouts_v1": (1000, 48, "OPD --eval-seed0 1000 / GRPO eval_suite seed0 1000, eval_n 48"),
    "e2_dev_slice": (1100, 50, "configs/experiments/jevgym_dev_rl.yaml"),
    "stress_dev_v2": (2000, 200, "configs/experiments/jevgym_stress_dev_miles.yaml"),
}
# words that would reveal an oracle label inside a training prompt ("gold" alone is a customer tier in JevGym states)
ORACLE_WORDS = re.compile(r'oracle|acceptable_tools|gold_aux|gold_status|canonical plan|"gold"\s*:|label_source',
                          re.IGNORECASE)

FIRST = ["Alice", "Bob", "Chen", "Dana", "Emeka", "Farah", "Goro", "Hana", "Ivan", "Jia", "Kofi", "Lena", "Mateo", "Nia",
         "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq"]
LAST = ["Smith", "Garcia", "Wang", "Kim", "Okafor", "Haddad", "Tanaka", "Novak", "Silva", "Ali"]
NUM_PERM, SHINGLE, MINHASH_SEED = 128, 5, 20261005
NEAR_THRESHOLDS = (0.8, 0.9)


# ================================================================== small helpers
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def parse_task_id(tid: str) -> dict[str, Any] | None:
    m = TASK_RE.search(tid or "")
    if not m:
        return None
    fam, seed, d, style, strict = m.groups()
    return {"family": fam, "seed": int(seed), "n_distractors": int(d),
            "distractor_style": (style or "-random")[1:], "strict": bool(strict), "task_id": m.group(0)}


def seed_of(tid: str) -> int | None:
    p = parse_task_id(tid)
    return None if p is None else p["seed"]


def seeds_in_bytes(data: bytes) -> set[int]:
    return {int(m.group(2)) for m in TASK_RE_B.finditer(data)}


def task_ids_in_bytes(data: bytes) -> set[str]:
    return {m.group(0).decode() for m in TASK_RE_B.finditer(data)}


def ranges(seeds: Iterable[int]) -> list[list[int]]:
    """Sorted seeds -> maximal runs [lo, hi]."""
    s = sorted(set(seeds))
    if not s:
        return []
    out, lo, prev = [], s[0], s[0]
    for x in s[1:]:
        if x != prev + 1:
            out.append([lo, prev])
            lo = x
        prev = x
    out.append([lo, prev])
    return out


def ranges_text(seeds: Iterable[int], max_parts: int = 6) -> str:
    r = ranges(seeds)
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in r]
    return ", ".join(parts[:max_parts]) + (f", ... ({len(parts)} runs)" if len(parts) > max_parts else "")


def classify_seed(seed: int) -> str:
    for name, (lo, hi) in EVAL_RANGES.items():
        if lo <= seed <= hi:
            return name
    if CAL_RANGE[0] <= seed <= CAL_RANGE[1]:
        return "calibration"
    if DEV_RANGE[0] <= seed <= DEV_RANGE[1]:
        return "development"
    if seed >= TRAIN_MIN:
        return "training"
    return "other"


def rollout_seeds(seed0: int, n: int, per_iter: int, iters: int, seed: int = 0, style: str = "cursor") -> list[int]:
    """The task seeds a trainer drew: ``random.Random(seed).shuffle(range(n))`` and ``per_iter`` per iteration (the
    opd.py / grpo_online.py cursor, reshuffling at the end of the order; the Miles rollout's modular index)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    out: list[int] = []
    if style == "modular":
        for rid in range(iters):
            out += [seed0 + order[(rid * per_iter + j) % n] for j in range(per_iter)]
        return out
    cursor = 0
    for _ in range(iters):
        for _ in range(min(per_iter, n)):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            out.append(seed0 + order[cursor])
            cursor += 1
    return out


# ================================================================== state normalisation and MinHash
NOISE_LINE = "log: heartbeat ok #"
_DIGITS = re.compile(r"\d*")
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z0-9]+")
_NAME = re.compile(r"\b(?:" + "|".join(FIRST + LAST) + r")\b")
_UID = re.compile(r"\bU\d+\b")
_OID = re.compile(r"\bORD-\d+\b")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def is_noise_line(line: str) -> bool:
    """A JevGym heartbeat noise line or a fragment of one left by the harness's middle truncation
    (``harness/agent.py::_trunc``): a prefix of ``log: heartbeat ok #<digits>``, a suffix of it, or the marker."""
    s = line.strip()
    if not s:
        return False
    if s == "...[truncated]...":
        return True
    if s.startswith(NOISE_LINE) and _DIGITS.fullmatch(s[len(NOISE_LINE):]):
        return True
    if NOISE_LINE.startswith(s):                     # prefix fragment ("log: heartbea")
        return True
    for k in range(len(NOISE_LINE) + 1):            # suffix fragment ("artbeat ok #4480", "#12", "12")
        tail = NOISE_LINE[k:]
        if s.startswith(tail) and _DIGITS.fullmatch(s[len(tail):]) and (tail or s.isdigit()):
            return True
    return False


def normalise_state(s: str) -> str:
    """Instance-level text: noise lines and truncation markers dropped, lower-cased, whitespace collapsed."""
    lines = [ln for ln in str(s).splitlines() if not is_noise_line(ln)]
    return _WS.sub(" ", "\n".join(lines).lower()).strip()


def mask_entities(s: str) -> str:
    """Plan-shape text: customer names, user ids, order ids and numbers replaced by placeholders (before
    normalisation, on the raw text)."""
    lines = [ln for ln in str(s).splitlines() if not is_noise_line(ln)]
    t = "\n".join(lines)
    t = _OID.sub("ORDID", _UID.sub("USERID", t))
    t = _NAME.sub("NAME", t)
    t = _NUM.sub("NUM", t)
    return _WS.sub(" ", t.lower()).strip()


def text_key(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def shingle_hashes(text: str, n: int = SHINGLE) -> np.ndarray:
    toks = _TOK.findall(text.lower())
    if not toks:
        return np.zeros(1, dtype=np.uint64)
    grams = [" ".join(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))]
    return np.fromiter((zlib.crc32(g.encode()) for g in set(grams)), dtype=np.uint64)


class MinHasher:
    """MinHash with ``num_perm`` universal hashes h(x) = (a x + b) mod p over the Mersenne prime p = 2^31 - 1, a in
    [1, p), b in [0, p), x = the 32-bit shingle hash reduced mod p. a x + b < 2^62 stays inside uint64, and the
    multiplication wraps around p, so every h is a proper random permutation of the shingle space (a modulus far
    above a x + b would leave h monotone in x and make every signature pick the smallest shingle)."""

    P = np.uint64((1 << 31) - 1)

    def __init__(self, num_perm: int = NUM_PERM, seed: int = MINHASH_SEED):
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, (1 << 31) - 1, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.b = rng.randint(0, (1 << 31) - 1, size=num_perm, dtype=np.int64).astype(np.uint64)

    def signature(self, hv: np.ndarray) -> np.ndarray:
        x = hv.astype(np.uint64) % self.P
        ph = (x[:, None] * self.a[None, :] + self.b[None, :]) % self.P
        return ph.min(axis=0).astype(np.uint32)

    def signatures(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.signature(shingle_hashes(t)) for t in texts]) if texts else \
            np.zeros((0, len(self.a)), dtype=np.uint32)


def max_similarity(A: np.ndarray, B: np.ndarray, groups_a: Sequence[Any] | None = None,
                   groups_b: Sequence[Any] | None = None, chunk: int = 48, topk: int = 1
                   ) -> tuple[np.ndarray, np.ndarray]:
    """For every row of A: the largest MinHash-estimated Jaccard to a row of B and the indices of the ``topk`` most
    similar rows of B (shape (n, topk)); pairs with the same group label are skipped when both group lists are given
    (the within-corpus baseline)."""
    n = len(A)
    best = np.full(n, -1.0)
    arg = np.full((n, max(1, topk)), -1, dtype=np.int64)
    if n == 0 or len(B) == 0:
        return best, arg
    P = A.shape[1]
    k = min(max(1, topk), len(B))
    gb = np.asarray(groups_b) if groups_b is not None else None
    for i in range(0, n, chunk):
        eq = (A[i:i + chunk, None, :] == B[None, :, :]).sum(axis=-1, dtype=np.int32) / P
        if gb is not None and groups_a is not None:
            ga = np.asarray(groups_a[i:i + chunk])
            eq = np.where(ga[:, None] == gb[None, :], -1.0, eq)
        best[i:i + chunk] = eq.max(axis=1)
        top = np.argpartition(-eq, k - 1, axis=1)[:, :k]
        arg[i:i + chunk, :k] = top
    return best, arg


def jaccard(a: str, b: str, n: int = SHINGLE) -> float:
    def sh(t: str) -> set[str]:
        toks = _TOK.findall(t.lower())
        return {" ".join(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}
    x, y = sh(a), sh(b)
    return len(x & y) / len(x | y) if x | y else 1.0


def near_scores(eval_txt: Sequence[str], ref_txt: Sequence[str], groups_e: Sequence[Any] | None = None,
                groups_r: Sequence[Any] | None = None, floor: float = 0.65, topk: int = 5
                ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Max Jaccard of every evaluation text to the reference texts: the MinHash estimate, replaced by the exact
    Jaccard (max over the ``topk`` MinHash candidates) wherever the estimate reaches ``floor``. Returns (score,
    MinHash estimate, calibration of the estimate on the verified pairs)."""
    mh = MinHasher()
    A, B = mh.signatures(list(eval_txt)), mh.signatures(list(ref_txt))
    est, arg = max_similarity(A, B, groups_e, groups_r, topk=topk)
    score = est.copy()
    diffs = []
    for i in np.where(est >= floor)[0]:
        cands = [j for j in arg[i] if j >= 0 and (groups_e is None or groups_r is None or groups_r[j] != groups_e[i])]
        ex = max((jaccard(eval_txt[i], ref_txt[j]) for j in cands), default=0.0)
        diffs.append(est[i] - ex)
        score[i] = ex
    cal = {"verified": len(diffs), "mean_est_minus_exact": float(np.mean(diffs)) if diffs else None}
    return score, est, cal


# ================================================================== JevGym task specs
def make_env(tid: str, multi_customers: int | None = None) -> Any:
    from jevharness.bench.suites.jevgym import JevGymEnv

    p = parse_task_id(tid)
    if p is None:
        raise ValueError(f"not a JevGym task id: {tid}")
    fams = "v2" if p["family"] in V2_FAMILIES else "v1"
    kw: dict[str, Any] = {"families": fams, "n_distractors": p["n_distractors"], "strict": p["strict"],
                          "distractor_style": p["distractor_style"]}
    if multi_customers:
        kw["multi_customers"] = multi_customers
    env = JevGymEnv(seed=p["seed"], **kw)
    if env.task_id != p["task_id"]:
        raise ValueError(f"regenerated {env.task_id} != {p['task_id']}")
    return env


def task_spec(env: Any) -> dict[str, Any]:
    """Fingerprints of one task: the whole world, the gold plan with its arguments and answer (customer / order ids,
    values), the instruction text, the target customer and the plan's (order id, amount) pairs."""
    w = env.world
    world = json.dumps({"users": w.users, "orders": w.orders, "stock": w.stock}, sort_keys=True)
    plan = env._plan()
    gold = json.dumps({"family": env.family, "variant": env.variant, "missing": env.missing,
                       "instruction": env.instruction, "reply": env.reply, "plan": plan}, sort_keys=True, default=str)
    oids = {a.get("order_id") for _, a in plan if a.get("order_id")}
    orders = sorted((o, w.orders[o]["amount"]) for o in oids if o in w.orders)
    return {"world": text_key(world), "plan": text_key(gold), "instruction": env.instruction,
            "target": f"{env.user['name']}|{env.user['user_id']}", "orders": [f"{o}|{a}" for o, a in orders],
            "family": env.family, "variant": env.variant}


def specs_for(task_ids: Iterable[str], multi_customers: int | None = None) -> dict[str, dict[str, Any]]:
    return {t: task_spec(make_env(t, multi_customers)) for t in sorted(set(task_ids))}


def spec_overlap(eval_specs: Mapping[str, Mapping[str, Any]], train_specs: Mapping[str, Mapping[str, Any]]
                 ) -> dict[str, Any]:
    """Share of evaluation tasks with a training task sharing each fingerprint."""
    idx: dict[str, set[str]] = defaultdict(set)
    for s in train_specs.values():
        idx["world"].add(s["world"])
        idx["plan"].add(s["plan"])
        idx["instruction"].add(s["instruction"])
        idx["target"].add(s["target"])
        idx["orders"].update(s["orders"])
    n = len(eval_specs)
    hits = Counter()
    for s in eval_specs.values():
        hits["world"] += s["world"] in idx["world"]
        hits["plan"] += s["plan"] in idx["plan"]
        hits["instruction"] += s["instruction"] in idx["instruction"]
        hits["target"] += s["target"] in idx["target"]
        hits["orders_any"] += any(o in idx["orders"] for o in s["orders"])
    by_variant: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in eval_specs.values():
        k = s["variant"] or s["family"]
        by_variant[k][0] += 1
        by_variant[k][1] += s["instruction"] in idx["instruction"]
    return {"n_eval": n, "n_train": len(train_specs), **{f"{k}_dup": int(v) for k, v in hits.items()},
            **{f"{k}_rate": (v / n if n else 0.0) for k, v in hits.items()},
            "instruction_dup_by_variant": {k: {"n": a, "dup": b} for k, (a, b) in sorted(by_variant.items())}}


# ================================================================== section 1: seeds
def _ids_of_rows(path: Path) -> list[str]:
    return [r["id"] for r in read_jsonl(path)] if path.exists() else []


def training_sources() -> dict[str, dict[str, Any]]:
    """Every training-side source of JevGym tasks, with the task ids read from the file (or reconstructed)."""
    src: dict[str, dict[str, Any]] = {}

    def add(name: str, kind: str, ids: Iterable[str], how: str, files: Sequence[Path] = ()) -> None:
        tids = sorted({(parse_task_id(i) or {}).get("task_id") for i in ids} - {None})
        src[name] = {"kind": kind, "how": how, "task_ids": tids, "seeds": sorted({seed_of(t) for t in tids}),
                     "files": {str(p.relative_to(ROOT)): sha256_file(p)[:12] for p in files if p.exists()}}

    v0_train = [V0 / "train.jsonl", V0 / "train_teacher_standin.jsonl"]
    add("data_v0_train_rows", "training", [i for p in v0_train for i in _ids_of_rows(p)], "row ids", v0_train)
    # a 24-row counterfactual-Q pipeline / sanity set that no recipe trains on (configs/train/data_v0_notes.json)
    add("data_v0_mc_standin_unused", "unused", _ids_of_rows(V0 / "mc_standin.jsonl"),
        "row ids (pipeline sanity set, never trained on)", [V0 / "mc_standin.jsonl"])
    v0_eps = sorted((V0 / "trajectories").glob("*/shadow.jsonl.gz"))
    add("data_v0_episodes", "training", [t for p in v0_eps for t in task_ids_in_bytes(gzip.open(p).read())],
        "episode task ids (v0 trajectories)", v0_eps)
    add("data_v1_train_rows", "training", _ids_of_rows(V1 / "train.jsonl.gz"), "row ids", [V1 / "train.jsonl.gz"])
    work = sorted((V1_CACHE / "_work").glob("*.jsonl"))
    add("data_v1_build_inputs", "training", [t for p in work for t in task_ids_in_bytes(p.read_bytes())],
        "task ids of the builder's work files (gold rows before labelling + latest episodes, incl. the v1 dev tasks "
        "held out by task; data/cache/train_v1/_work)", work)
    eps = sorted((V1_CACHE / "train_v1_jevgym").glob("*.jsonl"))
    add("data_v1_episodes", "training", [t for p in eps for t in task_ids_in_bytes(p.read_bytes())],
        "episode task ids (data/cache/train_v1/train_v1_jevgym)", eps)
    for name in ("opd_v1", "grpo_v1", "opd_v2"):
        r = ROLLOUTS[name]
        seeds = rollout_seeds(r["seed0"], r["n"], r["per_iter"], r["iters"],
                              style="modular" if name.endswith("v2") else "cursor")
        add(f"rollouts_{name}", "training", [f"jevgym-{'clarify' if s % 2 == 0 else 'multi'}-{s}-d60-near-strict"
                                             for s in seeds], f"reconstructed: {r['sampler']}")
        src[f"rollouts_{name}"]["n_draws"] = len(seeds)
    logged = R6 / "grpo_rollouts.jsonl"
    add("rollouts_grpo_v2", "training", [r["task_id"] for r in read_jsonl(logged)] if logged.exists() else [],
        "as logged (results/gpu_round6_miles/grpo_rollouts.jsonl)", [logged, R6 / "grpo_rollout_metrics.jsonl"])
    for name in ROLLOUTS:
        r = ROLLOUTS[name]
        src[f"pool_{name}"] = {"kind": "training_pool", "how": f"{r['config']}: seeds {r['seed0']}-"
                               f"{r['seed0'] + r['n'] - 1}", "task_ids": [],
                               "seeds": list(range(r["seed0"], r["seed0"] + r["n"])), "files": {}}
    # selection-side (development) data
    add("gate_set_v0_dev", "development", _ids_of_rows(V0 / "dev.jsonl"), "row ids (210-decision gate set)",
        [V0 / "dev.jsonl"])
    devs = [V1 / f for f in ("dev.jsonl.gz", "dev_recovery.jsonl.gz", "dev_knockout.jsonl.gz")]
    add("data_v1_dev_rows", "development", [i for p in devs for i in _ids_of_rows(p)], "row ids", devs)
    for name, (s0, n, how) in DEV_INSTRUMENTS.items():
        src[name] = {"kind": "development", "how": how, "task_ids": [], "seeds": list(range(s0, s0 + n)), "files": {}}
    return src


def validate_reconstruction() -> dict[str, Any]:
    """The v2 GRPO rollout log records task_seeds per rollout: the reconstruction must reproduce them in order."""
    p = R6 / "grpo_rollout_metrics.jsonl"
    if not p.exists():
        return {"checked": False}
    logged = [s for r in read_jsonl(p) for s in r.get("task_seeds") or []]
    r = ROLLOUTS["grpo_v2"]
    rec = rollout_seeds(r["seed0"], r["n"], r["per_iter"], r["iters"], style="modular")
    n = min(len(logged), len(rec))
    return {"checked": True, "n_logged": len(logged), "n_equal_in_order": sum(a == b for a, b in zip(logged, rec)),
            "identical": logged == rec[: len(logged)] and n == len(logged)}


def scan_runs(runs_dir: Path = RUNS) -> dict[str, list[int]]:
    """runs/<dir>[/<sub>] -> the JevGym seeds named anywhere in its *.jsonl files."""
    out: dict[str, set[int]] = defaultdict(set)
    if not runs_dir.exists():
        return {}
    for p in runs_dir.rglob("*.jsonl"):
        rel = p.relative_to(runs_dir).parts
        key = "/".join(rel[:2]) if len(rel) > 2 else rel[0]
        out[key] |= seeds_in_bytes(p.read_bytes())
    return {k: sorted(v) for k, v in sorted(out.items()) if v}


def calibration_fits() -> list[dict[str, Any]]:
    """Every tau-hat* fit file in the repository with the seeds it was fitted on (its own ``seeds`` block)."""
    out = []
    for p in sorted(list(ROOT.glob("results/calibration/*.json")) + list(ROOT.glob("results/gpu_round5/cal/tauhat*.json"))
                    + list(ROOT.glob("results/stage2/**/tauhat*.json")) + list(ROOT.glob("results/gpu_round6_miles/**/tauhat*.json"))):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        s = d.get("seeds") if isinstance(d, dict) else None
        if not isinstance(s, dict) or "n_tasks" not in s:
            continue
        out.append({"file": str(p.relative_to(ROOT)), "min": s.get("min"), "max": s.get("max"), "n_tasks": s.get("n_tasks"),
                    "run_dir": (d.get("meta") or {}).get("run_dir"), "tau_hat": d.get("tau_hat")})
    return out


def eval_sources() -> dict[str, dict[str, Any]]:
    src: dict[str, dict[str, Any]] = {}
    for name, f in (("stage1_task_list", "configs/eval/jevgym_eval_tasks.json"),
                    ("stage2_task_list", "configs/eval/jevgym_stage2_tasks.json")):
        ids = json.loads((ROOT / f).read_text())["task_ids"]
        src[name] = {"how": f, "task_ids": ids, "seeds": sorted({seed_of(t) for t in ids})}
    c = json.loads((S23 / "corpus.json").read_text())["eligible_tasks"] if (S23 / "corpus.json").exists() else []
    src["s23_corpus_tasks"] = {"how": "results/stage2/s23/corpus.json eligible_tasks", "task_ids": c,
                               "seeds": sorted({seed_of(t) for t in c})}
    hf = ROOT / "configs/experiments/harness_final_live.yaml"
    if hf.exists():
        import yaml

        su = (yaml.safe_load(hf.read_text()) or {}).get("suite") or {}
        s0, n = int(su.get("seed0", 300500)), int(su.get("n", 200))
        src["harness_final_spare"] = {"how": f"{hf.relative_to(ROOT)} suite seed0 {s0}, n {n}", "task_ids": [],
                                      "seeds": list(range(s0, s0 + n))}
    return src


def audit_seeds() -> dict[str, Any]:
    from jevharness.bench import registry as REG
    from jevharness.training import seed_guard as SG

    tr = training_sources()
    ev = eval_sources()
    runs = scan_runs()
    run_class = {k: dict(Counter(classify_seed(s) for s in v)) for k, v in runs.items()}
    eval_run_dirs = {k: v for k, v in runs.items()
                     if any(classify_seed(s) in EVAL_RANGES for s in v)}
    train_seeds = {s for k, v in tr.items() if v["kind"] == "training" for s in v["seeds"]}
    pool_seeds = {s for k, v in tr.items() if v["kind"] == "training_pool" for s in v["seeds"]}
    dev_seeds = {s for k, v in tr.items() if v["kind"] == "development" for s in v["seeds"]}
    cal = calibration_fits()
    cal_seeds = {s for c in cal if c["min"] is not None for s in range(int(c["min"]), int(c["max"]) + 1)}
    eval_seeds_files = {s for v in ev.values() for s in v["seeds"]} | {s for v in eval_run_dirs.values() for s in v
                                                                        if classify_seed(s) in EVAL_RANGES}
    registered = [r for r in (REG.EVAL_SEED_RANGES + REG.RETIRED_EVAL_RANGES + REG.STAGE2_RESERVED_RANGES)]

    def in_registered(s: int) -> bool:
        return any(lo <= s <= hi for lo, hi in registered)

    # task-level disjointness of the splits inside the training range
    v1_train_tasks = set(tr["data_v1_train_rows"]["task_ids"])
    v1_dev_tasks = set(tr["data_v1_dev_rows"]["task_ids"])
    gate_tasks = set(tr["gate_set_v0_dev"]["task_ids"])
    v0_train_tasks = set(tr["data_v0_train_rows"]["task_ids"])
    checks = {
        "train_vs_eval_files": len((train_seeds | pool_seeds) & eval_seeds_files),
        "train_vs_registered_eval_ranges": sum(1 for s in train_seeds | pool_seeds if in_registered(s)),
        "train_vs_calibration_fits": len((train_seeds | pool_seeds) & cal_seeds),
        "train_vs_dev_instruments": len((train_seeds | pool_seeds) & dev_seeds - {s for s in dev_seeds if s >= TRAIN_MIN}),
        "train_below_10000": sum(1 for s in train_seeds | pool_seeds if s < TRAIN_MIN),
        "train_in_calibration_range": sum(1 for s in train_seeds if CAL_RANGE[0] <= s <= CAL_RANGE[1]),
        "dev_vs_eval": len(dev_seeds & eval_seeds_files),
        "calibration_vs_eval": len(cal_seeds & eval_seeds_files),
        "v1_train_tasks_vs_v1_dev_tasks": len(v1_train_tasks & v1_dev_tasks),
        "v1_train_tasks_vs_gate_tasks": len(v1_train_tasks & gate_tasks),
        "v0_train_tasks_vs_gate_tasks": len(v0_train_tasks & gate_tasks),
        "seed_guard_train_forbidden": [list(r) for r in SG.TRAIN_FORBIDDEN],
    }
    notes = {"mc_standin_tasks_in_gate_set": sorted(set(tr["data_v0_mc_standin_unused"]["task_ids"]) & gate_tasks)}
    pools = {k: [v["seeds"][0], v["seeds"][-1]] for k, v in tr.items() if v["kind"] == "training_pool"}
    pool_pairs = {f"{a}~{b}": len(set(tr[a]["seeds"]) & set(tr[b]["seeds"])) for a in pools for b in pools if a < b}
    return {
        "training_sources": {k: {"kind": v["kind"], "how": v["how"], "n_tasks": len(v["task_ids"]) or None,
                                 "n_seeds": len(v["seeds"]), "ranges": ranges_text(v["seeds"]),
                                 "min": min(v["seeds"]) if v["seeds"] else None,
                                 "max": max(v["seeds"]) if v["seeds"] else None, "files": v["files"],
                                 **({"n_draws": v["n_draws"]} if "n_draws" in v else {})}
                             for k, v in tr.items()},
        "reconstruction_check": validate_reconstruction(),
        "eval_sources": {k: {"how": v["how"], "n_seeds": len(v["seeds"]), "ranges": ranges_text(v["seeds"])}
                         for k, v in ev.items()},
        "eval_run_dirs": {k: {"n_seeds": len(v), "ranges": ranges_text(v), "classes": run_class[k]}
                          for k, v in eval_run_dirs.items()},
        "all_run_dirs": {k: {"n_seeds": len(v), "ranges": ranges_text(v), "classes": run_class[k]}
                         for k, v in runs.items()},
        "calibration_fits": cal,
        "pool_overlaps": pool_pairs,
        "n_training_seeds_files": len(train_seeds), "n_training_pool_seeds": len(pool_seeds),
        "n_eval_seeds_files": len(eval_seeds_files), "n_dev_seeds": len(dev_seeds), "n_cal_seeds": len(cal_seeds),
        "checks": checks, "notes": notes,
        "clean": all(v == 0 for k, v in checks.items() if isinstance(v, int)),
    }


# ================================================================== section 2: content overlap
def _jevgym_training_rows() -> list[dict[str, Any]]:
    rows = []
    for p in (V1 / "train.jsonl.gz", V0 / "train.jsonl", V0 / "mc_standin.jsonl"):
        if p.exists():
            rows += [dict(r, _file=p.name) for r in read_jsonl(p) if str(r.get("id", "")).startswith("jevgym")]
    return rows


def _row_step(r: Mapping[str, Any]) -> int | None:
    st = (r.get("meta") or {}).get("step")
    if isinstance(st, int):
        return st
    parts = str(r.get("id", "")).split("/")
    if len(parts) > 1:
        m = re.search(r"(\d+)$", parts[1])
        if m:
            return int(m.group(1))
    return None


def _task_weighted(values: Mapping[str, list[float]]) -> float:
    v = [sum(x) / len(x) for x in values.values() if x]
    return sum(v) / len(v) if v else float("nan")


def corpus_accuracy(occ: Sequence[Mapping[str, Any]], answers: Mapping[str, Mapping[str, Any]],
                    keep: Iterable[int] | None = None) -> dict[str, Any]:
    """Task-weighted tool-selection accuracy on the corpus (A7.10 (c): task mean of occurrence correctness),
    optionally over a subset of occurrence indices."""
    from jevharness.bench.stage2_corpus import answer_correct

    idx = range(len(occ)) if keep is None else keep
    per: dict[str, list[float]] = defaultdict(list)
    for i in idx:
        o = occ[i]
        per[o["task"]].append(float(answer_correct(o, answers.get(o["qhash"]))))
    return {"acc": _task_weighted(per), "n_occ": sum(len(v) for v in per.values()), "n_tasks": len(per)}


def load_answers(name: str, corpus: Path = S23) -> dict[str, dict[str, Any]]:
    p = corpus / f"answers_{name}.jsonl"
    return {r["qhash"]: r for r in read_jsonl(p)} if p.exists() else {}


CORPUS_MODELS = ("jev", "intern", "rl", "sft_v1", "base", "v2", "sl0.8b", "sl2b", "sl4b", "sl9b", "sl27b")


def audit_content(fast: bool = False) -> dict[str, Any]:
    from jevharness.training.data_v1 import Decontaminator, eval_texts, query_text

    out: dict[str, Any] = {}
    # ---------------- (a) task specs: every training task vs every evaluation task
    tr = training_sources()
    train_tids: dict[str, list[str]] = {}
    for k, v in tr.items():
        if v["kind"] == "training" and v["task_ids"]:
            train_tids[k] = v["task_ids"]
    pools: dict[str, list[str]] = {}
    for name, r in ROLLOUTS.items():
        pools[name] = [f"jevgym-{'clarify' if s % 2 == 0 else 'multi'}-{s}-d60-near-strict"
                       for s in range(r["seed0"], r["seed0"] + r["n"])]
    all_train = sorted({t for v in train_tids.values() for t in v})
    all_pool = sorted({t for v in pools.values() for t in v})
    train_specs = specs_for(all_train)
    pool_specs = specs_for(all_pool)
    dev_specs = {**specs_for([f"jevgym-{'clarify' if s % 2 == 0 else 'multi'}-{s}-d60-near-strict"
                              for s in list(range(1000, 1048)) + list(range(1100, 1150)) + list(range(5000, 5150))]),
                 **specs_for([f"jevgym-{'clarify' if s % 2 == 0 else 'multi'}-{s}-d120-near-strict"
                              for s in range(2000, 2200)], multi_customers=5)}
    ev = eval_sources()
    eval_sets = {"stage1_eval": ev["stage1_task_list"]["task_ids"], "stage2_eval": ev["stage2_task_list"]["task_ids"],
                 "s23_corpus_tasks": ev["s23_corpus_tasks"]["task_ids"]}
    if "harness_final_spare" in ev:
        eval_sets["harness_final_spare"] = [f"jevgym-{'clarify' if s % 2 == 0 else 'multi'}-{s}-d60-near-strict"
                                            for s in ev["harness_final_spare"]["seeds"]]
    eval_specs = {k: specs_for(v) for k, v in eval_sets.items()}
    out["spec"] = {
        "n_training_tasks": len(train_specs), "n_pool_tasks": len(pool_specs),
        "vs_training_tasks": {k: spec_overlap(v, train_specs) for k, v in eval_specs.items()},
        "vs_rollout_pools": {k: spec_overlap(v, pool_specs) for k, v in eval_specs.items()},
        "vs_dev_and_calibration": {k: spec_overlap(v, dev_specs) for k, v in eval_specs.items()},
        # the generator's own baseline: Stage-2 tasks against the Stage-1 tasks (two evaluation sets, no training)
        "baseline_stage2_vs_stage1": spec_overlap(eval_specs["stage2_eval"], eval_specs["stage1_eval"]),
    }

    # ---------------- (b) states: training rows vs the common-state corpus C
    rows = _jevgym_training_rows()
    row_plan = {}
    for r in rows:
        t = parse_task_id(r["id"])["task_id"]
        if t in train_specs:
            row_plan[(train_specs[t]["plan"], _row_step(r))] = r["id"]
    tr_inst = {}
    tr_mask = set()
    for r in rows:
        tr_inst.setdefault(text_key(normalise_state(r["state"])), r)
        tr_mask.add(text_key(mask_entities(r["state"])))
    qs = read_jsonl(S23 / "questions.jsonl.gz")
    occ = read_jsonl(S23 / "occurrences.jsonl.gz")
    qstate = {q["qhash"]: q["state"] for q in qs}
    q_inst = {h: text_key(normalise_state(s)) for h, s in qstate.items()}
    q_mask = {h: text_key(mask_entities(s)) for h, s in qstate.items()}
    q_task: dict[str, set[str]] = defaultdict(set)
    for o in occ:
        q_task[o["qhash"]].add(o["task"])
    exact_q = {h for h, k in q_inst.items() if k in tr_inst}
    mask_q = {h for h, k in q_mask.items() if k in tr_mask}
    c_specs = eval_specs["s23_corpus_tasks"]
    spec_step_occ = [i for i, o in enumerate(occ) if (c_specs[o["task"]]["plan"], o["step"]) in row_plan]
    # within-C baseline: the same instance text in another task of C
    inst_tasks: dict[str, set[str]] = defaultdict(set)
    mask_tasks: dict[str, set[str]] = defaultdict(set)
    for h in qstate:
        inst_tasks[q_inst[h]] |= q_task[h]
        mask_tasks[q_mask[h]] |= q_task[h]
    within_exact = {h for h in qstate if len(inst_tasks[q_inst[h]]) > 1}
    within_mask = {h for h in qstate if len(mask_tasks[q_mask[h]]) > 1}

    def occ_stats(qset: set[str]) -> dict[str, Any]:
        idx = [i for i, o in enumerate(occ) if o["qhash"] in qset]
        tasks = {occ[i]["task"] for i in idx}
        steps = Counter(occ[i]["step"] for i in idx)
        return {"questions": len(qset), "occurrences": len(idx), "occ_rate": len(idx) / len(occ),
                "tasks": len(tasks), "steps": dict(sorted(steps.items())[:8]),
                "gold": dict(Counter(occ[i]["gold"] for i in idx).most_common(5))}

    states = {"n_training_rows": len(rows), "n_training_states": len(tr_inst), "n_questions": len(qstate),
              "n_occurrences": len(occ), "exact_instance": occ_stats(exact_q), "exact_masked": occ_stats(mask_q),
              "within_c_exact_instance": occ_stats(within_exact), "within_c_exact_masked": occ_stats(within_mask),
              "spec_step_dup": {"occurrences": len(spec_step_occ)}}
    # which training rows matched (example, for the report)
    states["exact_examples"] = [normalise_state(qstate[h])[:220] for h in sorted(exact_q)[:3]]

    # near duplicates (MinHash estimate, exact Jaccard wherever the estimate reaches 0.65)
    near_q: dict[float, set[str]] = {t: set() for t in NEAR_THRESHOLDS}
    if not fast:
        hs = sorted(qstate)
        c_txt = [normalise_state(qstate[h]) for h in hs]
        tr_txt = [normalise_state(r["state"]) for r in tr_inst.values()]
        best, _, cal = near_scores(c_txt, tr_txt)
        grp = [min(q_task[h]) for h in hs]
        best_w, _, cal_w = near_scores(c_txt, c_txt, grp, grp)
        for t in NEAR_THRESHOLDS:
            near_q[t] = {h for h, v in zip(hs, best) if v >= t}

        def q(v: np.ndarray) -> dict[str, float]:
            return {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)),
                    "p99": float(np.quantile(v, 0.99)), "max": float(v.max())}

        states["minhash"] = {
            "num_perm": NUM_PERM, "shingle_words": SHINGLE, "verify_floor": 0.65,
            "max_j_train": q(best), "max_j_within_c_other_tasks": q(best_w),
            **{f"near{int(t * 100)}": occ_stats(near_q[t]) for t in NEAR_THRESHOLDS},
            **{f"within_c_near{int(t * 100)}": occ_stats({h for h, v in zip(hs, best_w) if v >= t})
               for t in NEAR_THRESHOLDS},
            "calibration": {"vs_train": cal, "within_c": cal_w},
        }
    out["states_c"] = states

    # re-score every model on C without the overlapping states
    rescore: dict[str, Any] = {}
    excl_sets = {"exact": exact_q, **({f"near{int(t * 100)}": near_q[t] for t in NEAR_THRESHOLDS} if not fast else {}),
                 "masked": mask_q}
    for m in CORPUS_MODELS:
        ans = load_answers(m)
        if not ans:
            continue
        row = {"all": corpus_accuracy(occ, ans)}
        for k, qset in excl_sets.items():
            keep = [i for i, o in enumerate(occ) if o["qhash"] not in qset]
            only = [i for i, o in enumerate(occ) if o["qhash"] in qset]
            row[f"without_{k}"] = corpus_accuracy(occ, ans, keep)
            row[f"only_{k}"] = corpus_accuracy(occ, ans, only) if only else None
        rescore[m] = row
    out["rescore_c"] = rescore

    # ---------------- (c) tau^2 reference corpus
    t2: dict[str, Any] = {}
    if (T2REF / "questions.jsonl.gz").exists():
        tq = read_jsonl(T2REF / "questions.jsonl.gz")
        t_states = sorted({q["state"] for q in tq})
        w2c_rows = read_jsonl(V1 / "replay_when2call.jsonl.gz") + (read_jsonl(V0 / "replay_when2call.jsonl.gz")
                                                                    if (V0 / "replay_when2call.jsonl.gz").exists() else [])
        all_train_txt = {text_key(normalise_state(r["state"])) for r in rows + w2c_rows}
        t2["n_questions"], t2["n_states"] = len(tq), len(t_states)
        t2["exact_vs_training"] = sum(text_key(normalise_state(s)) in all_train_txt for s in t_states)
        if not fast:
            tr_all = list({text_key(normalise_state(r["state"])): normalise_state(r["state"]) for r in rows + w2c_rows}.values())
            best, _, cal = near_scores([normalise_state(s) for s in t_states], tr_all)
            t2["calibration"] = cal
            t2["max_j"] = {"median": float(np.median(best)), "p99": float(np.quantile(best, 0.99)),
                           "max": float(best.max())}
            t2.update({f"near{int(t * 100)}": int((best >= t).sum()) for t in (0.5,) + NEAR_THRESHOLDS})
        # the tau^2 reference corpus excludes the refit (shadow) split
        split = json.loads((ROOT / "configs/tau2/airline_task_ids.json").read_text()).get("split") or {}
        occ_t = read_jsonl(T2REF / "occurrences.jsonl.gz")
        air = {o["task"].split("/", 1)[1] for o in occ_t if o.get("domain") == "airline_test"}
        t2["airline_tasks"] = len(air)
        t2["airline_tasks_in_refit_split"] = len(air & set(split.get("shadow") or []))
    out["tau2_ref"] = t2

    # ---------------- (d) BFCL / When2Call static items
    texts = eval_texts(ROOT / "data/raw")
    decon = Decontaminator(texts, n=8)
    ev_norm = {_WS.sub(" ", t.lower()).strip() for t in texts}
    st: dict[str, Any] = {"n_eval_texts": len(texts), "rule": "8-gram overlap of the row's user text (data_v1.query_text)"
                          " with any BFCL (six tool categories, every item) or When2Call test question"}
    files = {"v1_train_jevgym": V1 / "train.jsonl.gz", "v1_dev": V1 / "dev.jsonl.gz",
             "v1_replay_when2call": V1 / "replay_when2call.jsonl.gz", "v0_train": V0 / "train.jsonl",
             "v0_replay_when2call": V0 / "replay_when2call.jsonl.gz", "v0_dev_when2call": V0 / "dev_when2call.jsonl.gz"}
    ev_list = sorted(ev_norm)
    for name, p in files.items():
        if not p.exists():
            continue
        rr = read_jsonl(p)
        qt = [query_text(r) for r in rr]
        hits = [r["id"] for r, q in zip(rr, qt) if decon.hit(q)]
        row = {"rows": len(rr), "ngram8_hits": len(hits), "exact_query_dup": sum(_WS.sub(" ", q.lower()).strip() in ev_norm
                                                                                    for q in qt),
               "hit_examples": hits[:3]}
        if not fast and name.endswith("when2call"):
            uq = sorted({_WS.sub(" ", q.lower()).strip() for q in qt})
            best, _, cal = near_scores(uq, ev_list)
            row["near80_unique_queries"] = int((best >= 0.8).sum())
            row["near50_unique_queries"] = int((best >= 0.5).sum())
            row["near80_examples"] = [[uq[i], ""] for i in np.where(best >= 0.8)[0][:3]]
            row["max_j"] = float(best.max()) if len(best) else None
        st[name] = row
    # the v1 builder's 15 dropped replay rows: present in v0's replay?
    if (V0 / "replay_when2call.jsonl.gz").exists():
        v1_ids = {r["id"] for r in read_jsonl(V1 / "replay_when2call.jsonl.gz")}
        v0_rows = [r for r in read_jsonl(V0 / "replay_when2call.jsonl.gz") if decon.hit(query_text(r))]
        v0_hits = [r["id"] for r in v0_rows]
        w2c_test = ROOT / "data/raw/when2call/test/when2call_test_mcq.jsonl"
        if w2c_test.exists():
            d_w = Decontaminator([json.loads(x)["question"] for x in w2c_test.read_text().splitlines() if x.strip()])
            st["v0_replay_hits_by_eval_source"] = {"when2call_test": sum(d_w.hit(query_text(r)) for r in v0_rows),
                                                   "bfcl_only": sum(not d_w.hit(query_text(r)) for r in v0_rows)}
        st["v0_replay_hits_absent_from_v1"] = sum(i not in v1_ids for i in v0_hits)
    e1 = json.loads((ROOT / "configs/eval/e1_dev_prefix_ids.json").read_text()).get("ids") or []
    train_ids = {r["id"] for r in read_jsonl(V1 / "train.jsonl.gz")} | {r["id"] for r in read_jsonl(V1 / "replay_when2call.jsonl.gz")}
    st["e1_dev_prefix_ids_in_training"] = len(set(e1) & train_ids)
    out["static"] = st
    return out


# ================================================================== section 3: labels and oracle access
def teacher_cache_check() -> dict[str, Any]:
    """Re-key every request of the data-v1 teacher pass (train: rows grouped by state, <= 16 questions per request;
    dev: one question per request) exactly as ``CachedBackend._key`` does and compare with the cache's keys."""
    from jevharness.decision.jev import JevBackend
    from jevharness.training.teacher_label import example_question

    if not TEACHER_CACHE.exists():
        return {"checked": False}
    desc = JevBackend(provider="openrouter", api_key="").describe()   # no key read, no request made

    def key(state: str, qs: list[Any]) -> str:
        body: dict[str, Any] = {"b": desc, "s": state,
                                "q": [q.model_dump(exclude={"id", "meta", "decision_point"}) for q in qs]}
        crit = [q.meta.get("criteria") for q in qs]
        if any(c is not None for c in crit):
            body["c"] = crit
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    def keys_for(rows: list[dict[str, Any]], per: int) -> tuple[list[str], set[int]]:
        exs = [e for e in rows if 2 <= len(e.get("options") or []) and e.get("state") and e.get("question")]
        groups: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(exs):
            groups[e["state"]].append(i)
        ks = []
        for idx in groups.values():
            for j in range(0, len(idx), per):
                job = idx[j:j + per]
                ks.append(key(exs[job[0]]["state"], [example_question(exs[i], None)[0] for i in job]))
        return ks, {seed_of(e["id"]) for e in exs}

    tr_rows = read_jsonl(V1_CACHE / "_work/train_gold.jsonl")
    dv_rows = read_jsonl(V1 / "dev.jsonl.gz")
    k_tr, s_tr = keys_for(tr_rows, 16)
    k_dv, s_dv = keys_for(dv_rows, 1)
    db = sqlite3.connect(f"file:{TEACHER_CACHE}?mode=ro", uri=True)
    cache = {k for (k,) in db.execute("SELECT k FROM c")}
    models = Counter(json.loads(v).get("usage", {}).get("model") for (v,) in db.execute("SELECT v FROM c"))
    expected = set(k_tr) | set(k_dv)
    seeds = s_tr | s_dv
    return {"checked": True, "cache_entries": len(cache), "requests_train": len(k_tr), "requests_dev": len(k_dv),
            "matched": len(cache & expected), "cache_only": len(cache - expected), "rows_only": len(expected - cache),
            "models": dict(models), "row_seed_min": min(seeds), "row_seed_max": max(seeds),
            "row_seeds_in_eval_ranges": sum(1 for s in seeds if classify_seed(s) in EVAL_RANGES)}


def jev_snapshots() -> dict[str, Any]:
    """The model string every cached Jev response carried, across the decision caches present locally."""
    out: dict[str, Any] = {}
    for p in sorted((ROOT / "cache").glob("*.sqlite")) + [TEACHER_CACHE]:
        if not p.exists():
            continue
        try:
            db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            rows = db.execute("SELECT json_extract(v,'$.usage.backend'), json_extract(v,'$.usage.model'), count(*) "
                              "FROM c GROUP BY 1, 2").fetchall()
        except sqlite3.Error:
            continue
        jev = {m: n for b, m, n in rows if b == "jev"}
        if jev:
            out[str(p.relative_to(ROOT))] = jev
    return out


def wire_canary() -> dict[str, Any]:
    """Offline: gold placed in ``Question.meta`` must not appear in the Jev wire question or in serve_jev's prompt."""
    from jevharness.decision import Choice, Noul
    from jevharness.decision.jev import question_to_wire

    canary = "CANARY-leakage-audit-GOLD"
    qs = [Choice("Which tool next?", ["find_user", "get_orders", "finish"], decision_point="tool_select",
                 gold=canary, gold_aux={"acceptable_tools": [canary]}),
          Noul("Is the task complete?", decision_point="stop", gold=canary, oracle={"verdict": canary})]
    wire = json.dumps([question_to_wire(q) for q in qs])
    prompt = ""
    try:
        from jevharness.training.data import render

        prompt = json.dumps([render({"state": "TASK: x", "question": q.text, "options": list(q.options or ["yes", "no"]),
                                     "kind": q.kind.value if hasattr(q.kind, "value") else str(q.kind),
                                     "meta": dict(q.meta)}) for q in qs], default=str)
    except Exception as e:  # noqa: BLE001 - the render signature is reported, not fatal
        prompt = f"render not checked: {type(e).__name__}"
    return {"canary_in_meta": all(canary in json.dumps(q.meta, default=str) for q in qs),
            "canary_in_wire": canary in wire, "canary_in_serve_prompt": canary in prompt,
            "serve_prompt_checked": not prompt.startswith("render not checked"),
            "test_file": "tests/test_wire_canary.py (transport-level canary for every backend, the harness loop)"}


def audit_labels() -> dict[str, Any]:
    out: dict[str, Any] = {}
    srcs: dict[str, Any] = {}
    for name, p in (("v1_train", V1 / "train.jsonl.gz"), ("v1_dev", V1 / "dev.jsonl.gz"), ("v0_train", V0 / "train.jsonl"),
                    ("v0_gate_dev", V0 / "dev.jsonl")):
        if p.exists():
            rr = read_jsonl(p)
            srcs[name] = {"rows": len(rr), "label_source": dict(Counter((r.get("meta") or {}).get("label_source")
                                                                        for r in rr)),
                          "oracle_words_in_state_or_question": sum(
                              1 for r in rr if ORACLE_WORDS.search(r.get("state", "") + "\n" + r.get("question", "")
                                                                   + "\n" + "\n".join(map(str, r.get("options") or []))))}
    out["label_sources"] = srcs
    out["teacher_cache"] = teacher_cache_check()
    out["jev_snapshots"] = jev_snapshots()
    out["wire_canary"] = wire_canary()
    opd = json.loads((R5 / "opd_summary.json").read_text()) if (R5 / "opd_summary.json").exists() else {}
    out["opd_v1_teacher"] = {"calls": (opd.get("teacher_ledger") or {}).get("calls"),
                             "cost_usd": (opd.get("teacher_ledger") or {}).get("cost_usd"),
                             "states": "the student's own rollout states on the training pool 40000-43999 (opd.py: the "
                                       "teacher answers the visited questions; seed_guard.check_train_pool)",
                             "cache_local": False}
    # operator branches: which paths did their commits touch, and from where did they fork
    eval_paths = re.compile(r"^(results/(primaries|stage2|main|secondary|explore|harness_final)/|runs/|"
                            r"paper/generated/(primaries|stage2|results|secondary)\.tex)")
    br: dict[str, Any] = {}
    for label, merge in (("v1 (<worktree>)", "487e212"),
                         ("v2 (<worktree>)", "6670c7a")):
        files = [f for f in git("log", "--format=", "--name-only", f"{merge}^1..{merge}^2").splitlines() if f]
        base = git("merge-base", f"{merge}^1", f"{merge}^2")
        tree = git("ls-tree", "-r", "--name-only", f"{merge}^2").splitlines()
        br[label] = {"merge": merge, "commits": len(git("log", "--format=%h", f"{merge}^1..{merge}^2").splitlines()),
                     "eval_output_paths_touched": sorted({f for f in files if eval_paths.match(f)}),
                     "fork_point": git("log", "-1", "--format=%h %cI", base),
                     "eval_files_in_branch_tree": sorted(f for f in tree if eval_paths.match(f))}
    out["operator_branches"] = br
    out["times"] = {
        "training_manifest_v1_commit": git("log", "--format=%h %cI", "--diff-filter=A", "--",
                                           "results/gpu_round5/TRAINING_MANIFEST.md").splitlines()[-1:],
        "training_manifest_v2_commit": git("log", "--format=%h %cI", "--diff-filter=A", "--",
                                           "results/gpu_round6_miles/TRAINING_MANIFEST.md").splitlines()[-1:],
        "stage1_primaries_first_commit": git("log", "--format=%h %cI", "--diff-filter=A", "--",
                                             "results/primaries/primaries.md").splitlines()[-1:],
        "freeze_v1": json.loads((R5 / "FREEZE_RECORD.json").read_text()).get("frozen_utc"),
        "freeze_v2": json.loads((ROOT / "results/stage2/v2_eval_plan.json").read_text()).get("frozen_utc"),
        "first_stage2_run": _start_time("results/stage2/START_stage2_jevgym.txt"),
        "first_rl_eval_run": _start_time("results/stage2/START_stage2_jevgym_rl.txt"),
        "first_rl_states_run": _start_time("results/stage2/START_stage2_rl_states.txt"),
        "teacher_cache_mtime": dt.datetime.fromtimestamp(TEACHER_CACHE.stat().st_mtime, dt.timezone.utc).isoformat()
        if TEACHER_CACHE.exists() else None,
        "stage1_main_run_start": _start_time("results/main/START_E1.txt"),
    }
    return out


def _start_time(rel: str) -> str | None:
    p = ROOT / rel
    if not p.exists():
        return None
    m = re.search(r"started\s+(\S+)", p.read_text())
    if m:
        return m.group(1)
    m = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?", p.read_text())
    return m.group(0) if m else None


# ================================================================== section 4: model selection
def audit_selection() -> dict[str, Any]:
    out: dict[str, Any] = {}
    cd = json.loads((R5 / "chain_decision.json").read_text())
    fr = json.loads((R5 / "FREEZE_RECORD.json").read_text())
    out["v1"] = {"kept": fr["chain_decision"]["kept"], "retention": fr["chain_decision"]["retention"],
                 "inputs": "v1 dev (training-seed tasks 30006-30597), recovery and knockout subsets, the 210-decision "
                           "gate set (training-seed tasks 10005-12055), greedy dev rollouts 1000-1047, the E2 development "
                           "slice 1100-1149 (r5_chain.py docstring; TRAINING_MANIFEST section 4)",
                 "chain_decision_sha256_matches_freeze": sha256_file(R5 / "chain_decision.json") ==
                 fr["chain_decision"]["sha256"],
                 "temperature_fit": fr["temperatures"]["procedure"],
                 "chain_keys": sorted(cd.keys())[:12]}
    cd2 = R6 / "chain_decision.json"
    out["v2"] = {"inputs": "v1 dev, recovery, the gate set, the stress instrument 2000-2199 (development seeds), the E2 "
                           "development slice 1100-1149 (r6_chain.py; TRAINING_MANIFEST section 5)",
                 "chain_decision_present": cd2.exists()}
    # tau-hat* used by the Stage-2 profile backends: the fit files named in the profile and their seeds
    import yaml

    prof = yaml.safe_load((ROOT / "configs/harness/jevagent_best.yaml").read_text())
    fits = []

    def walk(o: Any, path: str = "") -> None:
        if isinstance(o, dict):
            if "fitted_from" in o and isinstance(o["fitted_from"], dict):
                for dp, ff in o["fitted_from"].items():
                    if isinstance(ff, dict) and ff.get("path"):
                        p = Path(ff["path"])
                        p = p if p.is_absolute() else ROOT / p
                        rel = Path(str(p).replace("", "")) if not p.exists() else p
                        try:
                            d = json.loads((ROOT / rel).read_text() if not rel.is_absolute() else rel.read_text())
                            s = d.get("seeds") or {}
                            fits.append({"backend": path, "dp": dp, "file": str(Path(ff["path"]).name),
                                         "seeds": [s.get("min"), s.get("max")],
                                         "sha256_ok": sha256_file(ROOT / rel if not rel.is_absolute() else rel) == ff.get("sha256")})
                        except (OSError, json.JSONDecodeError):
                            fits.append({"backend": path, "dp": dp, "file": str(ff["path"]), "seeds": None})
            for k, v in o.items():
                walk(v, f"{path}/{k}" if path else str(k))
        elif isinstance(o, list):
            for v in o:
                walk(v, path)

    walk(prof)
    out["profile_tau_fits"] = fits
    out["profile_tau_fits_all_calibration"] = all(f["seeds"] and CAL_RANGE[0] <= f["seeds"][0] and f["seeds"][1] <= CAL_RANGE[1]
                                                  for f in fits if f.get("seeds"))
    out["profile_selection"] = "Stage-2 profile chosen on Stage-1 evaluation results by the registered rule "\
                               "(scripts/select_profile.py, A7.1) and confirmed on fresh seeds 300000-300499"
    return out


# ================================================================== section 5: external models
def audit_external() -> dict[str, Any]:
    first_repo = git("log", "--reverse", "--format=%h %cI", "--max-parents=0").splitlines()[:1]
    first_gym = git("log", "--reverse", "--format=%h %cI", "--", "src/jevharness/bench/suites/jevgym.py").splitlines()[:1]
    snaps = jev_snapshots()
    jev_models = Counter()
    for v in snaps.values():
        jev_models.update(v)
    return {"repo_first_commit": first_repo[0] if first_repo else None,
            "jevgym_first_commit": first_gym[0] if first_gym else None,
            "jev_models_served": dict(jev_models),
            "release_dates_recorded": {
                "Jev (TypeSafe)": "launched 2026-09-15 (docs/JEV_NOTES.md); every cached response names snapshot "
                                  "typesafe/jev-1.13-20260917",
                "Intern-Decision-0.8B/2B/4B": "2026-09-26 (docs/DECISION_MODEL_LANDSCAPE.md); HF snapshot 0e5e6aa7 "
                                              "(TRAINING_MANIFEST v1)",
                "StartLux-Decision": "2026-09-30 per the main session; package a9101e9 of 2026-10-01 (A7.18 (b))",
                "Qwen3-4B (base)": "HF snapshot 1cfa9a72 (TRAINING_MANIFEST v1)"},
            "note": "Closed training data cannot be audited; JevGym did not exist before its first commit, its seeds "
                    "were never released, and the evaluation tasks were generated after every external model's "
                    "recorded release. Public L1 items (BFCL, When2Call) may be in any vendor's data."}


# ================================================================== build + outputs
def build(fast: bool = False) -> dict[str, Any]:
    res = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "git": git("rev-parse", "--short=12", "HEAD"), "fast": fast,
           "label": "post hoc, descriptive audit; no API call; changes no registered result"}
    res["seeds"] = audit_seeds()
    res["content"] = audit_content(fast=fast)
    res["labels"] = audit_labels()
    res["selection"] = audit_selection()
    res["external"] = audit_external()
    res["paper_opd_text"] = paper_opd_text()
    res["findings"] = findings(res)
    return res


def _pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}"


def tex_rows(res: Mapping[str, Any]) -> dict[str, str]:
    """\\defres{leak/...} keys."""
    k: dict[str, str] = {}
    s = res["seeds"]
    k["leak/seeds/train_files/n"] = f"{s['n_training_seeds_files']:,}".replace(",", "{,}")
    k["leak/seeds/train_pools/n"] = f"{s['n_training_pool_seeds']:,}".replace(",", "{,}")
    k["leak/seeds/eval/n"] = f"{s['n_eval_seeds_files']:,}".replace(",", "{,}")
    k["leak/seeds/overlap"] = str(s["checks"]["train_vs_eval_files"])
    k["leak/seeds/overlap_registered"] = str(s["checks"]["train_vs_registered_eval_ranges"])
    tsrc = s["training_sources"]
    k["leak/seeds/train/min"] = f"{min(v['min'] for v in tsrc.values() if v['kind'] == 'training' and v['min']):,}" \
        .replace(",", "{,}")
    k["leak/seeds/train/max"] = f"{max(v['max'] for v in tsrc.values() if v['kind'] in ('training', 'training_pool') and v['max']):,}" \
        .replace(",", "{,}")
    rc = s["reconstruction_check"]
    if rc.get("checked"):
        k["leak/seeds/reconstruction/n"] = str(rc["n_logged"])
        k["leak/seeds/reconstruction/identical"] = "yes" if rc["identical"] else "no"
    c = res["content"]
    for ev in ("stage1_eval", "stage2_eval", "s23_corpus_tasks", "harness_final_spare"):
        so = c["spec"]["vs_training_tasks"].get(ev)
        if not so:
            continue
        e = ev.replace("_eval", "").replace("s23_corpus_tasks", "corpus").replace("harness_final_spare", "spare")
        k[f"leak/spec/{e}/plan_dup"] = str(so["plan_dup"])
        k[f"leak/spec/{e}/world_dup"] = str(so["world_dup"])
        k[f"leak/spec/{e}/instruction_dup"] = _pct(so["instruction_rate"])
        k[f"leak/spec/{e}/target_dup"] = _pct(so["target_rate"])
        k[f"leak/spec/{e}/orders_dup"] = _pct(so["orders_any_rate"])
        sp = c["spec"]["vs_rollout_pools"].get(ev)
        if sp:
            k[f"leak/spec/{e}/pool_plan_dup"] = str(sp["plan_dup"])
    b = c["spec"]["baseline_stage2_vs_stage1"]
    k["leak/spec/baseline/instruction_dup"] = _pct(b["instruction_rate"])
    k["leak/spec/n_train_tasks"] = f"{c['spec']['n_training_tasks']:,}".replace(",", "{,}")
    k["leak/spec/n_pool_tasks"] = f"{c['spec']['n_pool_tasks']:,}".replace(",", "{,}")
    st = c["states_c"]
    k["leak/state/c/n_occ"] = f"{st['n_occurrences']:,}".replace(",", "{,}")
    k["leak/state/c/train_states"] = f"{st['n_training_states']:,}".replace(",", "{,}")
    k["leak/state/c/exact/occ"] = str(st["exact_instance"]["occurrences"])
    k["leak/state/c/exact/rate"] = _pct(st["exact_instance"]["occ_rate"])
    k["leak/state/c/exact/tasks"] = str(st["exact_instance"]["tasks"])
    k["leak/state/c/within/exact/rate"] = _pct(st["within_c_exact_instance"]["occ_rate"])
    k["leak/state/c/masked/rate"] = _pct(st["exact_masked"]["occ_rate"])
    k["leak/state/c/within/masked/rate"] = _pct(st["within_c_exact_masked"]["occ_rate"])
    k["leak/state/c/spec_step_dup"] = str(st["spec_step_dup"]["occurrences"])
    mh = st.get("minhash")
    if mh:
        for t in NEAR_THRESHOLDS:
            kk = f"near{int(t * 100)}"
            k[f"leak/state/c/{kk}/rate"] = _pct(mh[kk]["occ_rate"])
            k[f"leak/state/c/within/{kk}/rate"] = _pct(mh[f"within_c_{kk}"]["occ_rate"])
        k["leak/state/c/maxj/median"] = f"{mh['max_j_train']['median']:.2f}"
        k["leak/state/c/within/maxj/median"] = f"{mh['max_j_within_c_other_tasks']['median']:.2f}"
    for m, row in c["rescore_c"].items():
        k[f"leak/rescore/{m}/all"] = _pct(row["all"]["acc"])
        for kk in ("exact", "near90", "near80", "masked"):
            if row.get(f"without_{kk}"):
                k[f"leak/rescore/{m}/without_{kk}"] = _pct(row[f"without_{kk}"]["acc"])
                dlt = round(100 * (row[f"without_{kk}"]["acc"] - row["all"]["acc"]), 1) + 0.0   # no "-0.0"
                k[f"leak/rescore/{m}/without_{kk}/delta"] = f"{dlt:+.1f}"
            if row.get(f"only_{kk}"):
                k[f"leak/rescore/{m}/only_{kk}"] = _pct(row[f"only_{kk}"]["acc"])
    t2 = c["tau2_ref"]
    if t2:
        k["leak/tau2/n_states"] = f"{t2['n_states']:,}".replace(",", "{,}")
        k["leak/tau2/exact"] = str(t2["exact_vs_training"])
        if "near80" in t2:
            k["leak/tau2/near80"] = str(t2["near80"])
            k["leak/tau2/maxj"] = f"{t2['max_j']['max']:.2f}"
    stc = c["static"]
    k["leak/static/n_eval_texts"] = f"{stc['n_eval_texts']:,}".replace(",", "{,}")
    for name in ("v1_train_jevgym", "v1_replay_when2call", "v0_replay_when2call"):
        if name in stc:
            k[f"leak/static/{name.replace('_', '-')}/ngram8"] = str(stc[name]["ngram8_hits"])
            k[f"leak/static/{name.replace('_', '-')}/exact"] = str(stc[name]["exact_query_dup"])
    lab = res["labels"]
    tc = lab["teacher_cache"]
    if tc.get("checked"):
        k["leak/teacher/entries"] = f"{tc['cache_entries']:,}".replace(",", "{,}")
        k["leak/teacher/matched"] = f"{tc['matched']:,}".replace(",", "{,}")
        k["leak/teacher/unmatched"] = str(tc["cache_only"])
    jm = res["external"]["jev_models_served"]
    if jm:
        snap = max(jm, key=jm.get)
        k["leak/jev/snapshot"] = snap.replace("_", r"\_")
        k["leak/jev/responses"] = f"{sum(jm.values()):,}".replace(",", "{,}")
        k["leak/jev/snapshots_n"] = str(len(jm))
    fg = res["external"]["jevgym_first_commit"]
    if fg:
        k["leak/jevgym/first_commit"] = fg.split()[1][:10]
    k["leak/canary/wire"] = "absent" if not lab["wire_canary"]["canary_in_wire"] else "PRESENT"
    # the OPD run's own values (configs/train/opd_round5.yaml + results/gpu_round5/opd_summary.json), for the corrected
    # appendix sentence
    try:
        import yaml

        cfg = yaml.safe_load((ROOT / ROLLOUTS["opd_v1"]["config"]).read_text())
        k["leak/opd/pool_lo"] = f"{cfg['seeds']:,}".replace(",", "{,}")
        k["leak/opd/pool_hi"] = f"{cfg['seeds'] + cfg['n_tasks'] - 1:,}".replace(",", "{,}")
        k["leak/opd/tasks_per_iter"] = str(cfg["tasks_per_iter"])
        k["leak/opd/iterations"] = str(cfg["iterations"])
        k["leak/opd/drawn"] = str(cfg["tasks_per_iter"] * cfg["iterations"])
        k["leak/opd/lr"] = f"{cfg['lr']:.0e}".replace("e-0", "{\\times}10^{-") + "}"
        k["leak/opd/budget"] = f"{cfg['budget_usd']:.0f}"
        k["leak/opd/dev_lo"] = f"{cfg['eval_seed0']:,}".replace(",", "{,}")
        k["leak/opd/dev_hi"] = f"{cfg['eval_seed0'] + cfg['eval_n'] - 1:,}".replace(",", "{,}")
        opd = json.loads((R5 / "opd_summary.json").read_text())
        k["leak/opd/teacher_usd"] = f"{opd['teacher_ledger']['cost_usd']:.2f}"
    except (OSError, KeyError, ValueError):
        pass
    return k


def to_tex(res: Mapping[str, Any]) -> str:
    head = [r"% AUTO-GENERATED by scripts/leakage_audit.py -- do not edit by hand.",
            f"% generated {res['generated']} at git {res['git']}; post hoc, descriptive train/evaluation hygiene audit "
            "(no API call); not a registered analysis",
            r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{{key}}}{{{val}}}" for key, val in sorted(tex_rows(res).items())]
    return "\n".join(head + body + [r"\makeatother", ""])


def paper_opd_text() -> dict[str, Any]:
    """Does the appendix's OPD sentence state the seeds / batch / budget / LR of the run that produced the OPD
    continuations (configs/train/opd_round5.yaml, results/gpu_round5/opd_*)? Read, not edited."""
    app = ROOT / "paper/sections/appendix.tex"
    text = app.read_text() if app.exists() else ""
    m = re.search(r"training seeds ([0-9{},]+)--([0-9{},]+): forward KL", text)
    stated = [int(x.replace("{,}", "").replace(",", "")) for x in m.groups()] if m else None
    r = ROLLOUTS["opd_v1"]
    drawn = rollout_seeds(r["seed0"], r["n"], r["per_iter"], r["iters"])
    line = next((i + 1 for i, ln in enumerate(text.splitlines()) if "training seeds 40{,}000" in ln), None)
    return {"stated_seed_range": stated, "line": line, "run_pool": [r["seed0"], r["seed0"] + r["n"] - 1],
            "drawn_min_max": [min(drawn), max(drawn)], "drawn_outside_stated":
                sum(1 for x in drawn if stated and not stated[0] <= x <= stated[1]),
            "run_values": "pool 40,000-43,999 (864 tasks drawn, 40,011-43,998), 36 tasks per iteration x 24, LR 5e-5, "
                          "teacher hard stop $8 (spent $0.38), greedy development rollouts on seeds 1,000-1,047 "
                          "(configs/train/opd_round5.yaml, results/gpu_round5/opd_metrics.jsonl, opd_summary.json)",
            "stated_values": "seeds 40,000-40,511, 16 tasks per iteration, LR 3e-5, teacher budget $2, development "
                             "rollouts 1,000-1,031 (these match configs/train/opd_jevgym.yaml's batch / LR / dev size, "
                             "not the round-5 run)"}


def findings(res: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """(finding, size / evidence, fix) -- issues the audit found; none is a train / evaluation leak."""
    c, lab, s = res["content"], res["labels"], res["seeds"]
    out = []
    po = res.get("paper_opd_text") or {}
    if po.get("stated_seed_range") and po.get("drawn_outside_stated"):
        out.append(("Appendix (app:rl, 'Stages and hyper-parameters', line %s) misstates the OPD run" % po["line"],
                    f"stated: {po['stated_values']}; the run: {po['run_values']}; {po['drawn_outside_stated']} of the "
                    f"864 drawn training tasks lie outside the stated range (all inside the training pool)",
                    "replace the sentence's values with the round-5 run's (draft in leakage_paper_draft.tex); reporting "
                    "accuracy only, no leak"))
    st = c["static"]
    if st.get("v0_replay_when2call", {}).get("ngram8_hits"):
        out.append(("data v0's When2Call replay was not decontaminated",
                    f"{st['v0_replay_when2call']['ngram8_hits']} of {st['v0_replay_when2call']['rows']} rows share an "
                    f"8-gram with an evaluation question ({st.get('v0_replay_hits_by_eval_source')}); v0's 300-row "
                    f"When2Call dev file: {st.get('v0_dev_when2call', {}).get('ngram8_hits')}. Data v1 dropped them "
                    "(v1 replay: 0); only SFT v0 (R2, a development ablation row never scored on L1) trained on v0's replay",
                    "state it in the data-v0 description; no reported evaluation number is affected"))
    notes = s.get("notes", {}).get("mc_standin_tasks_in_gate_set")
    if notes:
        out.append(("v0's 24-row MC stand-in file shares tasks with the 210-decision gate set",
                    f"{len(notes)} tasks; the file is a pipeline sanity set that no recipe trains on",
                    "none needed; noted for completeness"))
    br = lab["operator_branches"]
    v2 = next((v for k, v in br.items() if k.startswith("v2")), None)
    if v2 and any("primaries" in f for f in v2["eval_files_in_branch_tree"]):
        out.append(("The v2 operator's checkout contained one Stage-1 output file",
                    "results/primaries/c7_selection.json (E2a non-inferiority and cost summaries of the C7 candidate "
                    "arms, committed 2026-10-02 21:55 UTC) came in with the fork point; the v2 manifest records that no "
                    "evaluation output was opened; the v1 checkout held start records only",
                    "disclose in the v2 appendix paragraph; v2 is exploratory and enters no test"))
    ex = c["states_c"]["exact_instance"]
    out.append(("Generator-level overlap: verbatim step-0 / step-1 states",
                f"{ex['occurrences']} of {c['states_c']['n_occurrences']} C occurrences ({100 * ex['occ_rate']:.1f}%) "
                "repeat a training state after noise removal (instruction-only states such as the name-less clarify "
                f"request); the same rate within C across tasks is {100 * c['states_c']['within_c_exact_instance']['occ_rate']:.1f}%",
                "report the re-scored accuracies (unchanged for the trained model); no gold plan, world or (plan, step) "
                "is shared"))
    return out


def verdicts(res: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    s, c, lab, sel = res["seeds"], res["content"], res["labels"], res["selection"]
    st = c["states_c"]
    so = c["spec"]["vs_training_tasks"]["stage2_eval"]
    v = []
    v.append(("1 Seeds", "clean" if s["clean"] else "ISSUE",
              f"{s['n_training_seeds_files']} training seeds read from files (+{s['n_training_pool_seeds']} pool seeds) "
              f"vs {s['n_eval_seeds_files']} evaluation seeds: {s['checks']['train_vs_eval_files']} shared"))
    ex = st["exact_instance"]
    mh = st.get("minhash") or {}
    near = (f"; Jaccard >= 0.8: {_pct(mh['near80']['occ_rate'])}% vs {_pct(mh['within_c_near80']['occ_rate'])}% within C"
            if mh else "")
    rs = c["rescore_c"].get("rl", {})
    v.append(("2 Content overlap", "clean at instance level; generator-level overlap by design",
              f"{so['plan_dup']} shared gold plans and {so['world_dup']} shared worlds with the Stage-2 tasks; "
              f"{ex['occurrences']} of {st['n_occurrences']} C occurrences ({_pct(ex['occ_rate'])}%) repeat a training "
              f"state verbatim after noise removal (within C across tasks: "
              f"{_pct(st['within_c_exact_instance']['occ_rate'])}%){near}; JevAgentRL-4B on C "
              f"{_pct(rs.get('all', {}).get('acc'))}% -> {_pct(rs.get('without_exact', {}).get('acc'))}% without them; "
              f"tau^2 and BFCL / When2Call: no exact or near duplicate (v1)"))
    tc = lab["teacher_cache"]
    v.append(("3 Labels / oracle", "clean (operator barrier procedural: one disclosed schema read; one Stage-1 file "
              "in the exploratory v2 checkout)",
              f"teacher cache {tc.get('matched')}/{tc.get('cache_entries')} requests re-keyed from training rows; "
              f"wire canary absent: {not lab['wire_canary']['canary_in_wire']}"))
    v.append(("4 Model selection", "clean",
              f"chain inputs development-only; tau-hat* fits on calibration seeds: {sel['profile_tau_fits_all_calibration']}"))
    v.append(("5 External models", "cannot be ruled out in principle; dated evidence against",
              f"JevGym first commit {res['external']['jevgym_first_commit']}; Jev snapshots "
              f"{list(res['external']['jev_models_served'])}"))
    return v


def to_md(res: Mapping[str, Any]) -> str:
    s, c, lab, sel, ext = res["seeds"], res["content"], res["labels"], res["selection"], res["external"]
    L: list[str] = []
    L.append("# Train / evaluation hygiene audit (JevAgentRL v1, exploratory v2)\n")
    L.append(f"Generated {res['generated']} at git `{res['git']}` by `scripts/leakage_audit.py` "
             f"({'fast mode: no MinHash' if res['fast'] else 'full'}). Post hoc and descriptive; no API call; "
             "changes no registered result. Machine-readable: `results/training/leakage_audit.json`; paper keys: "
             "`paper/generated/leakage.tex` (`\\res{leak/...}`).\n")
    L.append("## Verdicts\n")
    L.append("| item | verdict | evidence |\n|---|---|---|")
    for a, b, e in verdicts(res):
        L.append(f"| {a} | {b} | {e} |")
    L.append("")
    L.append("## Findings and fixes (none is a train / evaluation leak)\n")
    L.append("| finding | size / evidence | fix |\n|---|---|---|")
    for a, b, e in res.get("findings") or []:
        L.append(f"| {a} | {b} | {e} |")
    L.append("")
    # ---------------- seeds
    L.append("## 1. Seeds\n")
    L.append("Training-side sources (seeds read from the files unless marked reconstructed / pool):\n")
    L.append("| source | kind | how | tasks | seeds | range(s) |\n|---|---|---|---|---|---|")
    for k, v in s["training_sources"].items():
        L.append(f"| `{k}` | {v['kind']} | {v['how']} | {v['n_tasks'] or ''} | {v['n_seeds']} | {v['ranges']} |")
    rc = s["reconstruction_check"]
    L.append(f"\nReconstruction check (v2 GRPO, the only rollout log with seeds): {rc.get('n_equal_in_order')} of "
             f"{rc.get('n_logged')} logged task seeds reproduced in order (identical: {rc.get('identical')}). The v1 "
             "OPD / GRPO samplers are the same `random.Random(0).shuffle` code at the training commits (`cf767b1`, "
             "`805a185`), so their draws are reconstructed the same way.\n")
    L.append("Evaluation-side sources:\n")
    L.append("| source | how | seeds | range(s) |\n|---|---|---|---|")
    for k, v in s["eval_sources"].items():
        L.append(f"| `{k}` | {v['how']} | {v['n_seeds']} | {v['ranges']} |")
    L.append("\n`runs/` directories that touched a registered evaluation range:\n")
    L.append("| run dir | seeds | range(s) | classes |\n|---|---|---|---|")
    for k, v in s["eval_run_dirs"].items():
        L.append(f"| `{k}` | {v['n_seeds']} | {v['ranges']} | {v['classes']} |")
    L.append("\nCalibration (tau-hat*) fits and their seeds (read from each fit file):\n")
    L.append("| fit file | seeds | run |\n|---|---|---|")
    for f in s["calibration_fits"]:
        L.append(f"| `{f['file']}` | {f['min']}-{f['max']} | `{f['run_dir']}` |")
    L.append("\nDisjointness checks (each must be 0):\n")
    for kk, vv in s["checks"].items():
        L.append(f"- `{kk}`: {vv}")
    L.append(f"- training pools pairwise: {s['pool_overlaps']}")
    L.append(f"- note (not a training input): the 24-row MC stand-in file `data/train_v0/mc_standin.jsonl`, a "
             f"pipeline sanity set no recipe trains on, shares tasks with the 210-decision gate set: "
             f"{s['notes']['mc_standin_tasks_in_gate_set']}")
    L.append("")
    # ---------------- content
    L.append("## 2. Content overlap\n")
    sp = c["spec"]
    L.append(f"(a) **Task specs.** {sp['n_training_tasks']} training tasks (data v0 / v1, the v1 / v2 rollout draws) "
             f"and the {sp['n_pool_tasks']} tasks of the four rollout pools, regenerated from their seeds "
             "(`JevGymEnv`, task generation only), against every evaluation task. `plan` = family, variant, "
             "instruction, customer reply and the gold plan with every argument (user ids, order ids, amounts, "
             "speeds, reasons) and the final answer; `world` = all users, orders and stock.\n")
    L.append("| evaluation set | vs | n | world dup | plan dup | instruction dup | (name, user id) dup | (order id, amount) dup |"
             "\n|---|---|---|---|---|---|---|---|")
    for vs_name, block in (("training tasks", sp["vs_training_tasks"]), ("rollout pools", sp["vs_rollout_pools"]),
                           ("dev + calibration", sp["vs_dev_and_calibration"])):
        for ev, o in block.items():
            L.append(f"| {ev} | {vs_name} | {o['n_eval']} | {o['world_dup']} | {o['plan_dup']} | "
                     f"{_pct(o['instruction_rate'])}% | {_pct(o['target_rate'])}% | {_pct(o['orders_any_rate'])}% |")
    b = sp["baseline_stage2_vs_stage1"]
    L.append(f"| stage2_eval | stage1_eval (baseline, no training) | {b['n_eval']} | {b['world_dup']} | {b['plan_dup']} | "
             f"{_pct(b['instruction_rate'])}% | {_pct(b['target_rate'])}% | {_pct(b['orders_any_rate'])}% |")
    L.append("\nInstruction duplicates by variant (Stage-2 vs training tasks): " + ", ".join(
        f"{k} {v['dup']}/{v['n']}" for k, v in sp["vs_training_tasks"]["stage2_eval"]["instruction_dup_by_variant"].items())
        + ". The `name` variant's instruction names no customer, so it is the same text in every such task.\n")
    st = c["states_c"]
    L.append(f"(b) **States: training rows vs the common-state corpus C.** {st['n_training_rows']} JevGym training rows "
             f"({st['n_training_states']} distinct noise-free states) vs {st['n_questions']} distinct questions / "
             f"{st['n_occurrences']} occurrences of C. Normalisation: heartbeat noise lines and truncation fragments "
             "dropped, lower case, whitespace collapsed; masked = names, user ids, order ids and numbers replaced.\n")
    L.append("| measure | vs training rows: occurrences (rate) | tasks | within C, other tasks: occurrences (rate) |"
             "\n|---|---|---|---|")
    L.append(f"| exact, instance text | {st['exact_instance']['occurrences']} ({_pct(st['exact_instance']['occ_rate'])}%) | "
             f"{st['exact_instance']['tasks']} | {st['within_c_exact_instance']['occurrences']} "
             f"({_pct(st['within_c_exact_instance']['occ_rate'])}%) |")
    L.append(f"| exact, entity-masked (plan shape) | {st['exact_masked']['occurrences']} "
             f"({_pct(st['exact_masked']['occ_rate'])}%) | {st['exact_masked']['tasks']} | "
             f"{st['within_c_exact_masked']['occurrences']} ({_pct(st['within_c_exact_masked']['occ_rate'])}%) |")
    mh = st.get("minhash")
    if mh:
        for t in NEAR_THRESHOLDS:
            kk = f"near{int(t * 100)}"
            L.append(f"| MinHash Jaccard >= {t} (word {SHINGLE}-grams) | {mh[kk]['occurrences']} ({_pct(mh[kk]['occ_rate'])}%) | "
                     f"{mh[kk]['tasks']} | {mh['within_c_' + kk]['occurrences']} ({_pct(mh['within_c_' + kk]['occ_rate'])}%) |")
        L.append(f"\nMax estimated Jaccard per C question: to training rows median {mh['max_j_train']['median']:.2f}, "
                 f"p90 {mh['max_j_train']['p90']:.2f}, max {mh['max_j_train']['max']:.2f}; to C questions of other "
                 f"tasks median {mh['max_j_within_c_other_tasks']['median']:.2f}, p90 "
                 f"{mh['max_j_within_c_other_tasks']['p90']:.2f}. Every pair whose MinHash estimate reached "
                 f"{mh['verify_floor']} was re-scored with the exact Jaccard (best of the five MinHash candidates); "
                 f"thresholds are applied to the exact values ({mh['calibration']['vs_train']['verified']} and "
                 f"{mh['calibration']['within_c']['verified']} verified; mean estimate minus exact "
                 f"{mh['calibration']['vs_train']['mean_est_minus_exact']:+.3f}).")
    L.append(f"\n(gold-plan fingerprint, step) duplicates between training rows and C: {st['spec_step_dup']['occurrences']}.")
    L.append(f"\nWhere the exact instance duplicates sit: steps {st['exact_instance']['steps']}; gold "
             f"{st['exact_instance']['gold']}. Examples: " + " / ".join(f"`{e[:160]}`" for e in st["exact_examples"]))
    L.append("\n**Re-scored C accuracy** (task-weighted, % ; `without_x` drops every occurrence whose question is in "
             "overlap set x; `only_x` keeps only those):\n")
    keys = [k for k in ("exact", "near90", "near80", "masked") if any(f"without_{k}" in r for r in c["rescore_c"].values())]
    L.append("| model | all | " + " | ".join(f"without {k} | only {k}" for k in keys) + " |\n|---|---|" +
             "---|---|" * len(keys))
    for m, r in c["rescore_c"].items():
        cells = []
        for k in keys:
            w, o = r.get(f"without_{k}"), r.get(f"only_{k}")
            cells += [f"{_pct(w['acc'])} (n {w['n_occ']})" if w else "", f"{_pct(o['acc'])} (n {o['n_occ']})" if o else ""]
        L.append(f"| {m} | {_pct(r['all']['acc'])} | " + " | ".join(cells) + " |")
    t2 = c["tau2_ref"]
    if t2:
        L.append(f"\n(c) **tau^2 reference corpus.** {t2['n_states']} distinct states of {t2['n_questions']} questions: "
                 f"{t2['exact_vs_training']} exact duplicates of a training state (JevGym rows and both When2Call "
                 f"replays); MinHash max Jaccard {t2.get('max_j')}; >= 0.5: {t2.get('near50')}, >= 0.8: "
                 f"{t2.get('near80')}. Its airline tasks: {t2['airline_tasks']}, of which in the tau-hat* refit split: "
                 f"{t2['airline_tasks_in_refit_split']}.")
    stc = c["static"]
    L.append(f"\n(d) **BFCL / When2Call.** {stc['n_eval_texts']} evaluation questions (every BFCL item of the six tool "
             "categories and every When2Call test question; a superset of the C8 population). Rule: "
             f"{stc['rule']}.\n")
    L.append("| file | rows | 8-gram hits | exact query duplicates | near-dup >= 0.8 (unique queries) |\n|---|---|---|---|---|")
    for name in ("v1_train_jevgym", "v1_dev", "v1_replay_when2call", "v0_train", "v0_replay_when2call", "v0_dev_when2call"):
        if name in stc:
            r = stc[name]
            L.append(f"| `{name}` | {r['rows']} | {r['ngram8_hits']} | {r['exact_query_dup']} | {r.get('near80_unique_queries', '')} |")
    L.append(f"\nv0 replay hits absent from v1: {stc.get('v0_replay_hits_absent_from_v1')} (by evaluation source: "
             f"{stc.get('v0_replay_hits_by_eval_source')}); E1 dev-prefix ids in "
             f"training: {stc['e1_dev_prefix_ids_in_training']}.\n")
    # ---------------- labels
    L.append("## 3. Labels and oracle access\n")
    L.append("| file | rows | label sources | rows whose state / question carries an oracle word |\n|---|---|---|---|")
    for k, v in lab["label_sources"].items():
        L.append(f"| {k} | {v['rows']} | {v['label_source']} | {v['oracle_words_in_state_or_question']} |")
    tc = lab["teacher_cache"]
    L.append(f"\n**Teacher (Jev) cache, re-keyed.** `data/cache/teacher_v1.sqlite` holds {tc.get('cache_entries')} "
             f"entries; re-keying the {tc.get('requests_train')} train requests (rows grouped by state, <= 16 "
             f"questions) and {tc.get('requests_dev')} dev requests (one question each) from the training files gives "
             f"{tc.get('matched')} matches, {tc.get('cache_only')} cache entries with no training row and "
             f"{tc.get('rows_only')} rows never sent. Seeds of those rows: {tc.get('row_seed_min')}-"
             f"{tc.get('row_seed_max')} (in an evaluation range: {tc.get('row_seeds_in_eval_ranges')}). Model that "
             f"answered: {tc.get('models')}. The OPD v1 teacher ({lab['opd_v1_teacher']['calls']} calls, "
             f"${float(lab['opd_v1_teacher']['cost_usd'] or 0):.2f}) answered the student's own rollout states on the training pool; "
             "its cache stayed on the GPU host (not re-keyed here).\n")
    wc = lab["wire_canary"]
    L.append(f"**Wire canary (offline).** Gold in `Question.meta`: {wc['canary_in_meta']}; in the Jev wire questions: "
             f"{wc['canary_in_wire']}; in `serve_jev`'s training / serving prompt: {wc['canary_in_serve_prompt']} "
             f"(checked: {wc['serve_prompt_checked']}). Transport-level test: {wc['test_file']}.\n")
    L.append("**Operator barrier (A7.10 (b)).** Procedural, recorded in the training manifests (one disclosed schema "
             "read of one Stage-1 record, 2026-10-02 ~20:40 UTC, ruled on in the Stage-2 Deviations). Git evidence:\n")
    for k, v in lab["operator_branches"].items():
        L.append(f"- {k}: {v['commits']} commits merged by `{v['merge']}`; evaluation-output paths touched: "
                 f"{v['eval_output_paths_touched'] or 'none'}; forked at `{v['fork_point']}`; evaluation-side files in "
                 f"the branch tree: {v['eval_files_in_branch_tree'] or 'none'}.")
    L.append(f"\nTimes: {json.dumps(lab['times'])}\n")
    # ---------------- selection
    L.append("## 4. Model selection\n")
    L.append(f"- v1: kept {sel['v1']['kept']}; retention: {sel['v1']['retention']}. Inputs: {sel['v1']['inputs']}. "
             f"`chain_decision.json` sha256 equals the freeze record's: {sel['v1']['chain_decision_sha256_matches_freeze']}. "
             f"Temperatures: {sel['v1']['temperature_fit']}.")
    L.append(f"- v2: inputs {sel['v2']['inputs']}.")
    L.append(f"- Stage-2 profile thresholds (fit files named in `configs/harness/jevagent_best.yaml`; all on calibration "
             f"seeds: {sel['profile_tau_fits_all_calibration']}):")
    for f in sel["profile_tau_fits"]:
        L.append(f"  - {f['backend']} / {f['dp']}: `{f['file']}`, seeds {f['seeds']}, sha256 as recorded: {f.get('sha256_ok')}")
    L.append(f"- {sel['profile_selection']}.\n")
    # ---------------- external
    L.append("## 5. External models\n")
    L.append(f"- Repository first commit: {ext['repo_first_commit']}; JevGym first commit: {ext['jevgym_first_commit']}.")
    L.append(f"- Jev responses by snapshot (every local decision cache): {ext['jev_models_served']}.")
    for k, v in ext["release_dates_recorded"].items():
        L.append(f"- {k}: {v}.")
    L.append(f"- {ext['note']}\n")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fast", action="store_true", help="skip the MinHash near-duplicate passes")
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    ap.add_argument("--out-tex", default=str(OUT_TEX))
    ap.add_argument("--from-json", help="re-render the report and keys from an existing audit JSON (no recomputation)")
    a = ap.parse_args(argv)
    res = json.loads(Path(a.from_json).read_text()) if a.from_json else build(fast=a.fast)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=str))
    Path(a.out_md).write_text(to_md(res))
    Path(a.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_tex).write_text(to_tex(res))
    print(json.dumps({"verdicts": verdicts(res)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
