#!/usr/bin/env python3
r"""E20: can several decision models voting together, or a hosted decision API behind the confidence gate, replace
training on the agent's states?

PART A -- OFFLINE, post hoc and descriptive, $0 (no model call, no GPU; the convention of scripts/explore_drift.py:
no registration entry, labelled so in every output). Every decision model's stored answers on L1 (E1's 3,991 fixed
decisions, ``runs/e1_bfcl_when2call*`` and E15's slim copies in ``results/explore/e15_baselines/l1``) and on L2 (the
common states C, ``results/stage2/s23`` + E12 / E14 / E15 / E16 answer files) are combined into ensembles of decision
models (no LLM is ever a voter; the agent LLM only receives the deferred cases, as in every cascade):

* majority vote (plurality of the members' top answers; ties broken by the summed confidence, then member order);
* confidence-weighted vote (each member adds its confidence to its own top answer -- the stored records hold the top
  answer and its probability, not every option's -- and the answer with the largest sum wins);
* agreement gates "m of k" (delegate only when at least m of the k members give the same answer, else defer to the
  LLM): coverage, accepted errors and the accuracy of the accepted answers.

Three families: off the shelf only (``ots``: Jev, the OpenAI Decisions API, Intern-Decision, StartLux 0.8B-27B and
E15's open models), mixed (``mixed``: the same plus our trained models AgentDecision-4B, E12, E12c, the 0.8B scoring
head of E14 arm A and E16's two arms) and hosted only (``hosted``: Jev and the Decisions API, both callable from the
coordinating container; the candidate of PART B). Members and m are chosen on a DEVELOPMENT half and reported on the
other half: C split by task, L1 by item (a fixed hash, ``split_of``, written down before any accuracy was computed:
``results/explore/e20_ensembles/split.json``). Also: the drift script's consequence-aware predicted live success
(walk the reference trajectories of C; a task is lost iff the first wrong answer is irreversible), the must-ask
accuracy, whether agreement is a better confidence signal than a model's own confidence (accepted errors at equal
coverage), how correlated the models' errors are, and a DecisionGym-Hard slice (the tier rule) for the vote of
AgentDecision-4B with Jev on the states both answered in the two shadow calibration arms of E7c.

    python scripts/explore_e20_ensembles.py split      # the fixed dev / test split (no accuracy read)
    python scripts/explore_e20_ensembles.py choose     # members and m on the dev halves -> results/explore/e20_ensembles/choice.json
    python scripts/explore_e20_ensembles.py offline    # the held-out readout -> offline.{json,md}

PART B -- LIVE (registered: docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E20: decision-model ensembles and a
hosted decision API behind the gate"): see the ``live`` section below and ``summarize``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import itertools
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/explore/e20_ensembles"
TEX = ROOT / "paper/generated/e20_ensembles.tex"
S23 = ROOT / "results/stage2/s23"
RUNS = ROOT / "runs"
E15 = ROOT / "results/explore/e15_baselines"
HARD_RUN = RUNS / "e7c_cal"                     # E7c's two shadow calibration arms on DecisionGym-Hard (12 customers)
LABEL_A = ("post hoc, descriptive; stored answers only (no model call, no GPU); revises no registered result; no "
           "registration entry (the convention of scripts/explore_drift.py)")
LABEL_B = ("exploratory (E20), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E20: decision-model ensembles and "
           "a hosted decision API behind the gate'; outside every Holm table")
SPLIT_SALT = "e20-split-v1"
K_DP = {"tool_select": 0.0058322, "tool_params": 0.0034651}   # the frozen 4B's cost margins (E18 (c) 2)
OP_TOOLS = ("cancel_order", "ship_order")       # the tier rule's two write tools (explore_e7_hard.OP_TOOLS)
MAX_K = 9

# slug -> (label, group, L1 record file, C answer file); group: ots = off the shelf, ours = trained by us
_L1 = RUNS
_E15L1 = E15 / "l1"
_E15C = E15 / "c"
MODELS: dict[str, tuple[str, str, Path | None, Path | None]] = {
    "jev": ("Jev", "ots", _L1 / "e1_bfcl_when2call/s1_jev.jsonl", S23 / "answers_jev.jsonl"),
    "oaidec": ("OpenAI Decisions API", "ots", _E15L1 / "s1_oaidec.jsonl.gz", _E15C / "answers_oaidec.jsonl.gz"),
    "intern": ("Intern-Decision-4B", "ots", _L1 / "e1_bfcl_when2call_startlux/s1_intern_4b.jsonl", S23 / "answers_intern.jsonl"),
    "sl0.8b": ("StartLux-0.8B", "ots", _L1 / "e1_bfcl_when2call_startlux_ladder/s1_startlux_0_8b.jsonl", S23 / "answers_sl0.8b.jsonl"),
    "sl2b": ("StartLux-2B", "ots", _L1 / "e1_bfcl_when2call_startlux_ladder/s1_startlux_2b.jsonl", S23 / "answers_sl2b.jsonl"),
    "sl4b": ("StartLux-4B", "ots", _L1 / "e1_bfcl_when2call_startlux/s1_startlux_4b.jsonl", S23 / "answers_sl4b.jsonl"),
    "sl9b": ("StartLux-9B", "ots", _L1 / "e1_bfcl_when2call_startlux_ladder/s1_startlux_9b.jsonl", S23 / "answers_sl9b.jsonl"),
    "sl27b": ("StartLux-27B", "ots", _L1 / "e1_bfcl_when2call_startlux/s1_startlux_27b.jsonl", S23 / "answers_sl27b.jsonl"),
    "openjev": ("OpenJev", "ots", _E15L1 / "s1_openjev.jsonl.gz", _E15C / "answers_openjev.jsonl.gz"),
    "openjevhf": ("OpenJev-27B", "ots", _E15L1 / "s1_openjevhf.jsonl.gz", _E15C / "answers_openjevhf.jsonl.gz"),
    "clefflash": ("Clef-Flash", "ots", _E15L1 / "s1_clefflash.jsonl.gz", _E15C / "answers_clefflash.jsonl.gz"),
    "clef": ("Clef 27B (8-bit)", "ots", _E15L1 / "s1_clef.jsonl.gz", None),
    "jevk5": ("JevK5", "ots", _E15L1 / "s1_jevk5.jsonl.gz", _E15C / "answers_jevk5.jsonl.gz"),
    "clm": ("CLM", "ots", _E15L1 / "s1_clm.jsonl.gz", _E15C / "answers_clm.jsonl.gz"),
    "laya": ("Laya", "ots", _E15L1 / "s1_laya.jsonl.gz", _E15C / "answers_laya.jsonl.gz"),
    "verdict": ("Verdict", "ots", _E15L1 / "s1_verdict.jsonl.gz", _E15C / "answers_verdict.jsonl.gz"),
    "decider": ("decider-4b", "ots", _E15L1 / "s1_decider.jsonl.gz", _E15C / "answers_decider.jsonl.gz"),
    "imajev": ("Imajev-4B", "ots", _E15L1 / "s1_imajev.jsonl.gz", _E15C / "answers_imajev.jsonl.gz"),
    "bosun": ("Bosun", "ots", _E15L1 / "s1_bosun.jsonl.gz", _E15C / "answers_bosun.jsonl.gz"),
    "rl": ("AgentDecision-4B", "ours", _L1 / "e1_bfcl_when2call_rl/s1_jevagent_rl.jsonl", S23 / "answers_rl.jsonl"),
    "e12": ("AgentDecision-4B + public data (E12)", "ours", _L1 / "e1_bfcl_when2call_e12/s1_e12_mixed.jsonl",
            ROOT / "results/explore/e12/box/s23/answers_e12_mixed.jsonl"),
    "e12c": ("StartLux-4B + our recipe (E12c)", "ours", _L1 / "e1_bfcl_when2call_e12c/s1_e12c_sl4b.jsonl",
             ROOT / "results/explore/e12/box_c/s23/answers_e12c_sl4b.jsonl"),
    "e14a": ("0.8B scoring head (E14 A)", "ours", _L1 / "e1_bfcl_when2call_e14/s1_e14a.jsonl",
             ROOT / "results/explore/e14_small/records/s23/answers_e14a.jsonl.gz"),
    "e16a": ("AgentDecision-4B (mixed+, E16 A)", "ours", _L1 / "e1_bfcl_when2call_e16/s1_e16a.jsonl",
             ROOT / "results/explore/e16_best_open/box/s23/answers_e16a.jsonl"),
    "e16b": ("StartLux-9B (mixed, E16 B)", "ours", _L1 / "e1_bfcl_when2call_e16/s1_e16b.jsonl",
             ROOT / "results/explore/e16_best_open/box/s23/answers_e16b.jsonl"),
}
FAMILIES: dict[str, tuple[str, ...]] = {
    "ots": tuple(m for m, v in MODELS.items() if v[1] == "ots"),
    "mixed": tuple(MODELS),
    "hosted": ("jev", "oaidec"),
}
FAMILY_LABEL = {"ots": "off the shelf", "mixed": "off the shelf + ours", "hosted": "hosted only (Jev + Decisions API)"}
TRAINED_REF = ("rl", "e14a")                  # the comparison rows (our trained 4B and 0.8B head)
# the trained models' fitted tool-choice thresholds (gates of the comparison rows)
TAU_REF = {"rl": 0.4999999999978404, "e14a": 0.9999999997650646}
RULES = ("majority", "weighted")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _jl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    out = []
    with op(p, "rt") as f:
        for x in f:
            if x.strip():
                try:
                    out.append(json.loads(x))
                except json.JSONDecodeError:
                    continue
    return out


def _mean(v: Sequence[float]) -> float | None:
    return sum(v) / len(v) if v else None


def _share(a: float, b: float) -> float | None:
    return a / b if b else None


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================== the split (fixed before any accuracy is read)
def split_of(key: str, salt: str = SPLIT_SALT) -> str:
    """'dev' or 'test' by the parity of a fixed hash: C by task id, L1 by item id (the BFCL / When2Call item, the part
    of the decision id before the first '/', so an item's tool choice and arguments land in the same half)."""
    return "dev" if int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:16], 16) % 2 == 0 else "test"


def l1_item(decision_id: str) -> str:
    """The item of an E1 decision: BFCL ids are ``<item>/tool`` and ``<item>/param/<name>`` (the part before the first
    '/'); When2Call ids are ``w2c/<uuid>`` (one decision per item: the whole id). The first version keyed both by the
    part before the first '/', which put all 928 When2Call decisions into the development half; fixed on 2026-10-08
    ~15:45 UTC (``split.json`` rewritten, ``choose`` re-run; the BFCL part of the split is unchanged)."""
    d = str(decision_id)
    return d if d.startswith("w2c/") else d.split("/")[0]


def cmd_split(a: argparse.Namespace) -> int:
    occ = _jl(S23 / "occurrences.jsonl.gz")
    tasks = sorted({o["task"] for o in occ})
    ids = [r["id"] for r in _jl(MODELS["jev"][2])]  # type: ignore[arg-type]
    items = sorted({l1_item(i) for i in ids})
    res = {"label": LABEL_A, "written": _now(), "rule": f"dev iff int(sha256('{SPLIT_SALT}|' + key)[:16], 16) % 2 == 0",
           "c": {"key": "task id", "dev": [t for t in tasks if split_of(t) == "dev"],
                 "test": [t for t in tasks if split_of(t) == "test"]},
           "l1": {"key": "item id (BFCL: the decision id before the first '/'; When2Call: the whole id)",
                  "dev_items": sum(split_of(i) == "dev" for i in items), "test_items": sum(split_of(i) == "test" for i in items),
                  "dev_decisions": sum(split_of(l1_item(i)) == "dev" for i in ids),
                  "test_decisions": sum(split_of(l1_item(i)) == "test" for i in ids)},
           "note": "v1 (15:22 UTC) was written before any accuracy of any model or ensemble was computed on either half",
           "revision": ("v2 (~15:45 UTC): When2Call decisions keyed by their whole id; v1 keyed them by the part before "
                        "the first '/' (all are 'w2c/<uuid>'), which put all 928 into the development half. Found while "
                        "reading v1's held-out L1 numbers (BFCL only); C and the BFCL part are unchanged; choose and "
                        "the readout were re-run on v2")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"c_dev": len(res["c"]["dev"]), "c_test": len(res["c"]["test"]), **res["l1"]}))
    return 0


# ================================================================== units: what every model answered
class Level:
    """One level's units (C occurrences or L1 decisions): per unit the members' answers (None = no answer: an error,
    an invalid answer or a refusal, which never votes and is wrong) and confidences, and a correctness function."""

    def __init__(self, name: str, units: list[dict[str, Any]], answers: dict[str, list[tuple[str | None, float]]],
                 correct: Callable[[dict[str, Any], str | None], bool]):
        self.name = name
        self.units = units
        self.answers = answers           # model -> per unit (answer, confidence)
        self.correct = correct
        self._ok: dict[tuple[int, str | None], bool] = {}

    def ok(self, i: int, val: str | None) -> bool:
        k = (i, val)
        if k not in self._ok:
            self._ok[k] = bool(val is not None and self.correct(self.units[i], val))
        return self._ok[k]

    def models(self) -> list[str]:
        return list(self.answers)


def _c_answer(a: Mapping[str, Any] | None) -> tuple[str | None, float]:
    if not a or a.get("error") or not a.get("valid", True) or a.get("answer") is None or a.get("refusal"):
        return None, 0.0
    val = str(a["answer"])
    if val == "<refusal>":
        return None, 0.0
    conf = a.get("confidence")
    return val, float(conf) if conf is not None else 0.0


def load_c(models: Iterable[str] = MODELS) -> Level:
    from jevharness.bench import stage2_corpus as SC

    occ = _jl(S23 / "occurrences.jsonl.gz")
    units = [dict(o, i=i) for i, o in enumerate(occ)]
    ans: dict[str, list[tuple[str | None, float]]] = {}
    for m in models:
        p = MODELS[m][3]
        if p is None or not p.exists():
            continue
        by: dict[str, dict[str, Any]] = {}
        for r in _jl(p):
            by[r["qhash"]] = r                  # the latest record of a question decides (E15's rule)
        ans[m] = [_c_answer(by.get(u["qhash"])) for u in units]

    def correct(u: Mapping[str, Any], val: str | None) -> bool:
        return SC.answer_correct(u, {"answer": val}) if val is not None else False
    return Level("c", units, ans, correct)


def _l1_answer(r: Mapping[str, Any] | None) -> tuple[str | None, float]:
    if not r or r.get("error") or r.get("refusal") or r.get("pred") is None or r.get("pred") == "<refusal>":
        return None, 0.0
    return str(r["pred"]), float(r.get("conf") or 0.0)


def load_l1(models: Iterable[str] = MODELS) -> Level:
    from jevharness.bench.decision_eval import norm

    base = _jl(MODELS["jev"][2])  # type: ignore[arg-type]
    units = [{"id": r["id"], "dp": r["dp"], "source": r["source"], "gold": r.get("gold") or [],
              "item": l1_item(r["id"]), "i": i} for i, r in enumerate(base)]
    idx = {u["id"]: u["i"] for u in units}
    ans: dict[str, list[tuple[str | None, float]]] = {}
    for m in models:
        p = MODELS[m][2]
        if p is None or not p.exists():
            continue
        by = {r["id"]: r for r in _jl(p) if r.get("kind", "s1") == "s1" and r.get("id") in idx}
        ans[m] = [_l1_answer(by.get(u["id"])) for u in units]
    gold_n = [{norm(g) for g in u["gold"]} for u in units]

    def correct(u: Mapping[str, Any], val: str | None) -> bool:
        return val is not None and norm(val) in gold_n[u["i"]]
    return Level("l1", units, ans, correct)


def l1_family(u: Mapping[str, Any]) -> str:
    src = str(u["source"]).split(":")[0]
    return "w2c" if src == "when2call" else ("toolchoice" if u["dp"] == "tool_select" else "args")


# ================================================================== ensembles
def vote(answers: Sequence[tuple[str | None, float]], rule: str) -> tuple[str | None, int, float]:
    """The ensemble's answer on one unit: (value, number of members that gave it, summed confidence on it).
    ``majority``: most votes, ties by summed confidence, then member order; ``weighted``: largest summed confidence,
    ties by votes, then member order. Members without an answer abstain."""
    cnt: dict[str, int] = {}
    wsum: dict[str, float] = {}
    first: dict[str, int] = {}
    for j, (v, c) in enumerate(answers):
        if v is None:
            continue
        cnt[v] = cnt.get(v, 0) + 1
        wsum[v] = wsum.get(v, 0.0) + float(c)
        first.setdefault(v, j)
    if not cnt:
        return None, 0, 0.0
    if rule == "majority":
        best = min(cnt, key=lambda v: (-cnt[v], -wsum[v], first[v]))
    elif rule == "weighted":
        best = min(cnt, key=lambda v: (-wsum[v], -cnt[v], first[v]))
    else:
        raise ValueError(rule)
    return best, cnt[best], wsum[best]


def ensemble_answers(lv: Level, members: Sequence[str], rule: str) -> list[tuple[str | None, int, float]]:
    cols = [lv.answers[m] for m in members]
    return [vote([c[i] for c in cols], rule) for i in range(len(lv.units))]


def accuracy(lv: Level, vals: Sequence[str | None], idx: Sequence[int]) -> float | None:
    """C: the published task mean (every occurrence of a task weighs 1 / its count; stage2_corpus.task_accuracy);
    L1: pooled over decisions (E9's rule)."""
    if not idx:
        return None
    if lv.name == "c":
        per: dict[str, list[float]] = defaultdict(list)
        for i in idx:
            per[lv.units[i]["task"]].append(float(lv.ok(i, vals[i])))
        return _mean([sum(v) / len(v) for v in per.values()])
    return _mean([float(lv.ok(i, vals[i])) for i in idx])


def gate_stats(lv: Level, ens: Sequence[tuple[str | None, int, float]], m: int, idx: Sequence[int]) -> dict[str, Any]:
    """Agreement gate: accept a unit iff at least m members gave its top answer (pooled over units, as E18 counts)."""
    acc = [i for i in idx if ens[i][0] is not None and ens[i][1] >= m]
    wrong = sum(1 for i in acc if not lv.ok(i, ens[i][0]))
    return {"n": len(idx), "accepted": len(acc), "coverage": _share(len(acc), len(idx)), "accepted_errors": wrong,
            "accepted_acc": _share(len(acc) - wrong, len(acc)), "error_rate": _share(wrong, len(acc)) if acc else 0.0}


def threshold_stats(lv: Level, model: str, tau: float, idx: Sequence[int]) -> dict[str, Any]:
    """A single model behind the cascade at its own threshold: accept iff answered and confidence >= tau."""
    col = lv.answers[model]
    acc = [i for i in idx if col[i][0] is not None and col[i][1] >= tau]
    wrong = sum(1 for i in acc if not lv.ok(i, col[i][0]))
    return {"n": len(idx), "accepted": len(acc), "coverage": _share(len(acc), len(idx)), "accepted_errors": wrong,
            "accepted_acc": _share(len(acc) - wrong, len(acc)), "tau": tau}


def errors_at_coverage(lv: Level, model: str, n_accept: int, idx: Sequence[int]) -> dict[str, Any]:
    """Accepted errors when a model accepts its n_accept most confident units (its own confidence as the gate; ties at
    the boundary split at random: the expected count). Units it did not answer rank last."""
    col = lv.answers[model]
    scored = sorted(((col[i][1] if col[i][0] is not None else -1.0), i) for i in idx)
    scored.reverse()
    if n_accept <= 0:
        return {"accepted": 0, "accepted_errors": 0.0, "tau": None}
    n_accept = min(n_accept, len(scored))
    cut = scored[n_accept - 1][0]
    above = [i for c, i in scored if c > cut]
    tied = [i for c, i in scored if c == cut]
    need = n_accept - len(above)
    err_above = sum(1 for i in above if not lv.ok(i, col[i][0]))
    err_tied = sum(1 for i in tied if not lv.ok(i, col[i][0]))
    return {"accepted": n_accept, "accepted_errors": err_above + (need / len(tied)) * err_tied if tied else float(err_above),
            "tau": cut, "tied": len(tied)}


# ================================================================== selection on the dev half
def rank_singles(lv: Level, pool: Sequence[str], idx: Sequence[int]) -> list[tuple[str, float]]:
    out = []
    for m in pool:
        if m not in lv.answers:
            continue
        vals = [v for v, _ in lv.answers[m]]
        out.append((m, accuracy(lv, vals, idx) or 0.0))
    order = list(pool)
    return sorted(out, key=lambda x: (-x[1], order.index(x[0])))


def greedy(lv: Level, pool: Sequence[str], idx: Sequence[int], rule: str, start: str, k_max: int = MAX_K) -> list[list[str]]:
    """Forward selection from the best single model: add the member that most raises the rule's dev accuracy (ties:
    pool order); every step's set is a candidate."""
    sets = [[start]]
    cur = [start]
    rest = [m for m in pool if m != start and m in lv.answers]
    while rest and len(cur) < k_max:
        best = max(rest, key=lambda m: (accuracy(lv, [v for v, _, _ in ensemble_answers(lv, cur + [m], rule)], idx) or 0.0,
                                         -rest.index(m)))
        cur = cur + [best]
        rest.remove(best)
        sets.append(list(cur))
    return sets


def choose_level(lv: Level, family: str, dev: Sequence[int]) -> dict[str, Any]:
    pool = [m for m in FAMILIES[family] if m in lv.answers]
    singles = rank_singles(lv, pool, dev)
    order = [m for m, _ in singles]
    res: dict[str, Any] = {"pool": pool, "dev_singles": singles, "best_single": order[0] if order else None}
    if len(order) < 2:
        return res
    cands: dict[str, list[list[str]]] = {}
    for rule in RULES:
        sets = [order[:k] for k in range(2, min(MAX_K, len(order)) + 1)]
        sets += [s for s in greedy(lv, pool, dev, rule, order[0]) if len(s) >= 2]
        uniq: list[list[str]] = []
        for s in sets:
            if s not in uniq:
                uniq.append(s)
        cands[rule] = uniq
        scored = [(accuracy(lv, [v for v, _, _ in ensemble_answers(lv, s, rule)], dev) or 0.0, s) for s in uniq]
        best = max(scored, key=lambda x: (x[0], -len(x[1])))
        res[rule] = {"members": best[1], "dev_acc": best[0], "candidates": len(uniq)}
    # agreement gates: every candidate set of either rule, every m from a strict majority to unanimity; the chosen gate
    # has the highest dev coverage among those whose dev accepted-error rate is at most the tool-choice cost margin
    # k_dp (Proposition 1: accept when the chance of an error is below the cost margin); none -> the lowest error rate
    gates = []
    seen: list[list[str]] = []
    for s in cands["majority"] + cands["weighted"]:
        if s in seen:
            continue
        seen.append(s)
        ens = ensemble_answers(lv, s, "majority")
        for m in range(len(s) // 2 + 1, len(s) + 1):
            g = gate_stats(lv, ens, m, dev)
            gates.append({"members": s, "m": m, **g})
    # (the second tier was added on the dev half, before any test-half number was computed, after the first version's
    # fallback -- the lowest error rate over every gate -- picked a gate that accepted nothing)
    ok = [g for g in gates if (g["error_rate"] or 0.0) <= K_DP["tool_select"] and g["accepted"]]
    half = [g for g in gates if g["accepted"] and (g["coverage"] or 0.0) >= 0.5]
    if ok:
        pick = max(ok, key=lambda g: (g["coverage"], -g["accepted_errors"], -len(g["members"])))
        rule_used = "max coverage with accepted-error rate <= k_dp"
    elif half:
        pick = min(half, key=lambda g: (g["error_rate"], -g["coverage"], len(g["members"])))
        rule_used = "no gate within k_dp: the lowest accepted-error rate among gates delegating at least half"
    else:
        pick = min((g for g in gates if g["accepted"]), key=lambda g: (g["error_rate"], -g["coverage"]))
        rule_used = "no gate within k_dp or delegating half: the lowest accepted-error rate"
    res["gate"] = {"members": pick["members"], "m": pick["m"], "dev": {k: pick[k] for k in ("coverage", "accepted",
                   "accepted_errors", "accepted_acc", "error_rate")}, "rule": rule_used, "candidates": len(gates),
                   "within_kdp": len(ok)}
    return res


def halves(lv: Level) -> tuple[list[int], list[int]]:
    key = (lambda u: u["task"]) if lv.name == "c" else (lambda u: u["item"])
    dev = [u["i"] for u in lv.units if split_of(key(u)) == "dev"]
    test = [u["i"] for u in lv.units if split_of(key(u)) == "test"]
    return dev, test


def hosted_dev_check(lv: Level, dev: Sequence[int]) -> dict[str, Any]:
    """Part B's decision (registration): does the hosted agreement gate (Jev and the Decisions API agree) accept fewer
    errors than the Decisions API alone at the same coverage on the dev half? Also against Jev alone."""
    if not {"jev", "oaidec"} <= set(lv.answers):
        return {}
    ens = ensemble_answers(lv, ["jev", "oaidec"], "majority")
    g = gate_stats(lv, ens, 2, dev)
    o = errors_at_coverage(lv, "oaidec", g["accepted"], dev)
    j = errors_at_coverage(lv, "jev", g["accepted"], dev)
    return {"gate": g, "oaidec_at_equal_coverage": o, "jev_at_equal_coverage": j,
            "beats_oaidec": g["accepted_errors"] < o["accepted_errors"],
            "beats_jev": g["accepted_errors"] < j["accepted_errors"]}


def choose() -> dict[str, Any]:
    res: dict[str, Any] = {"label": LABEL_A, "written": _now(), "git": _git(),
                           "split_sha256": sha256_file(OUT / "split.json") if (OUT / "split.json").exists() else None,
                           "rule": {"vote": "members = the best dev accuracy among the top-k sets by dev single accuracy "
                                            "(k = 2..9) and greedy forward selection from the best single model",
                                    "gate": "m from a strict majority to unanimity over the same candidate sets; the "
                                            "highest dev coverage with accepted-error rate <= k_dp(tool choice) = "
                                            f"{K_DP['tool_select']}; else the lowest error rate among gates that "
                                            "delegate at least half of the dev units; else the lowest error rate"},
                           "levels": {}}
    for name, loader in (("c", load_c), ("l1", load_l1)):
        lv = loader()
        dev, _ = halves(lv)
        res["levels"][name] = {"dev_units": len(dev), "families": {f: choose_level(lv, f, dev) for f in FAMILIES}}
        res["levels"][name]["hosted_check"] = hosted_dev_check(lv, dev)
    hc = res["levels"]["c"]["hosted_check"]
    res["part_b"] = {"run_row_ii": bool(hc.get("beats_oaidec")),
                     "reading": "row (ii) runs iff the hosted agreement gate accepts fewer errors than the Decisions "
                                "API alone at equal coverage on the dev half of C"}
    return res


def cmd_choose(a: argparse.Namespace) -> int:
    res = choose()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "choice.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    brief = {lvn: {f: {k: (v.get("members"), v.get("m")) for k, v in fam.items() if isinstance(v, dict) and "members" in v}
                   for f, fam in lv["families"].items()} for lvn, lv in res["levels"].items()}
    print(json.dumps(brief, indent=1))
    print(json.dumps(res["levels"]["c"]["hosted_check"], indent=1, default=str))
    print(json.dumps(res["part_b"]))
    return 0



# ================================================================== the held-out readout (PART A)
def _vals(ens: Sequence[tuple[str | None, int, float]]) -> list[str | None]:
    return [v for v, _, _ in ens]


def _corpus_parts(lv: Level, idx: Sequence[int]) -> tuple[list[dict[str, Any]], list[str]]:
    occ = [lv.units[i] for i in idx]
    return occ, sorted({o["task"] for o in occ})


def _writes() -> set[str]:
    import explore_drift as D
    from benchmark_difficulty import near_map

    return D.write_tools(near_map())


def walk(occ: Sequence[Mapping[str, Any]], vals: Mapping[int, str | None], ok: Callable[[int, str | None], bool],
         writes: set[str], deferred: set[int] | None = None) -> dict[str, dict[str, float]]:
    """The drift script's consequence-aware walk (``explore_drift.walk_predictions``) on unit indices: each reference
    trajectory of C in step order; the first wrong answer decides (lost iff irreversible: a wrong write action or a
    premature finish). A deferred unit (an agreement gate that hands the decision to the LLM) counts as right: the
    agent LLM is right on 98.9-100 % of the labelled calibration tool choices (E9's DATA.md), an optimistic plug-in."""
    import explore_drift as D

    by: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for o in occ:
        by[(o["task"], o["source"])].append(o)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, src), seq in by.items():
        pred = 1.0
        for o in sorted(seq, key=lambda x: int(x.get("step", 0))):
            i = o["i"]
            if deferred is not None and i in deferred:
                continue
            if ok(i, vals[i]):
                continue
            pred = 0.0 if D.consequence(vals[i], writes) != "recoverable" else 1.0
            break
        out[task][src] = pred
    return out


def c_scores(lv: Level, vals: Sequence[str | None], idx: Sequence[int], writes: set[str],
             deferred: set[int] | None = None) -> dict[str, Any]:
    """Task-mean accuracy, the must-ask stratum's task mean and the predicted live success on the given units."""
    import explore_drift as D

    occ, tasks = _corpus_parts(lv, idx)
    must = [o["i"] for o in occ if D._must_ask(o)]
    pred = walk(occ, {i: vals[i] for i in idx}, lv.ok, writes, deferred)
    out = {"acc": accuracy(lv, vals, idx), "mustask_acc": accuracy(lv, vals, must) if must else None,
           "mustask_n": len(must), "pred_live": D.predicted_success(pred, tasks)}
    if deferred is not None:
        acc_idx = [i for i in idx if i not in deferred]
        out["mustask_accepted"] = sum(1 for i in must if i not in deferred)
        out["mustask_accepted_errors"] = sum(1 for i in must if i not in deferred and not lv.ok(i, vals[i]))
        out["accepted_tasks_acc"] = accuracy(lv, vals, acc_idx) if acc_idx else None
    return out


def l1_scores(lv: Level, vals: Sequence[str | None], idx: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"acc": accuracy(lv, vals, idx)}
    for fam in ("toolchoice", "args", "w2c"):
        sub = [i for i in idx if l1_family(lv.units[i]) == fam]
        out[fam] = accuracy(lv, vals, sub) if sub else None
    return out


def gate_block(lv: Level, members: Sequence[str], m: int, idx: Sequence[int], writes: set[str] | None,
               refs: Sequence[str]) -> dict[str, Any]:
    """An agreement gate on held-out units: coverage, accepted errors, accepted accuracy, and the accepted errors of
    single models (their own confidence as the gate) at the SAME coverage."""
    ens = ensemble_answers(lv, members, "majority")
    g = gate_stats(lv, ens, m, idx)
    deferred = {i for i in idx if not (ens[i][0] is not None and ens[i][1] >= m)}
    out: dict[str, Any] = {"members": list(members), "m": m, **g,
                           "own_conf_at_equal_coverage": {r: errors_at_coverage(lv, r, g["accepted"], idx)
                                                          for r in dict.fromkeys([*members, *refs]) if r in lv.answers}}
    if lv.name == "c" and writes is not None:
        out.update({k: v for k, v in c_scores(lv, _vals(ens), idx, writes, deferred).items()
                    if k in ("pred_live", "mustask_accepted", "mustask_accepted_errors")})
    return out


def single_gate(lv: Level, model: str, tau: float, idx: Sequence[int], writes: set[str] | None) -> dict[str, Any]:
    col = lv.answers[model]
    st = threshold_stats(lv, model, tau, idx)
    if lv.name == "c" and writes is not None:
        deferred = {i for i in idx if not (col[i][0] is not None and col[i][1] >= tau)}
        st["pred_live"] = c_scores(lv, [v for v, _ in col], idx, writes, deferred)["pred_live"]
    return st


def sweep(lv: Level, members: Sequence[str], idx: Sequence[int], singles: Sequence[str],
          taus: Sequence[float] = (0.0, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)) -> list[dict[str, Any]]:
    """The hosted pair as the live backend gates it: accept iff both members give the same answer AND the smaller
    of their confidences is >= t; against each single model accepting its most confident units at the same coverage."""
    cols = [lv.answers[m] for m in members]
    rows = []
    for t in taus:
        acc = [i for i in idx if all(c[i][0] is not None for c in cols) and len({c[i][0] for c in cols}) == 1
               and min(c[i][1] for c in cols) >= t]
        wrong = sum(1 for i in acc if not lv.ok(i, cols[0][i][0]))
        rows.append({"t": t, "accepted": len(acc), "coverage": _share(len(acc), len(idx)), "accepted_errors": wrong,
                     "singles": {s: errors_at_coverage(lv, s, len(acc), idx)["accepted_errors"] for s in singles}})
    return rows


def error_correlation(lv: Level, models: Sequence[str], idx: Sequence[int]) -> dict[str, Any]:
    """Pairwise error overlap on C occurrences (no selection involved): Jaccard of the two error sets against its
    value if the two models erred independently at their own rates, and the share of joint errors with the SAME wrong
    answer; on the must-ask stratum, how many models fail each state."""
    import explore_drift as D

    err = {m: {i for i in idx if not lv.ok(i, lv.answers[m][i][0])} for m in models}
    n = len(idx)
    pairs = []
    for a, b in itertools.combinations(models, 2):
        ea, eb = err[a], err[b]
        both = ea & eb
        either = ea | eb
        pa, pb = len(ea) / n, len(eb) / n
        indep = (pa * pb) / (pa + pb - pa * pb) if (pa + pb - pa * pb) > 0 else None
        jac = _share(len(both), len(either))
        same = sum(1 for i in both if lv.answers[a][i][0] == lv.answers[b][i][0])
        pairs.append({"a": a, "b": b, "err_a": len(ea), "err_b": len(eb), "both": len(both), "jaccard": jac,
                      "jaccard_indep": indep, "ratio": (jac / indep) if jac is not None and indep else None,
                      "same_wrong_share": _share(same, len(both))})
    must = [i for i in idx if D._must_ask(lv.units[i])]
    wrong_count = {i: sum(1 for m in models if i in err[m]) for i in must}
    best = max(models, key=lambda m: -len(err[m]))
    others = [m for m in models if m != best]
    best_must_err = [i for i in must if i in err[best]]
    shared = sum(1 for i in best_must_err if sum(1 for m in others if i in err[m]) >= len(others) / 2)
    ratios = [p["ratio"] for p in pairs if p["ratio"] is not None]
    return {"models": list(models), "n": n, "pairs": pairs,
            "median_jaccard": statistics.median([p["jaccard"] for p in pairs if p["jaccard"] is not None]) if pairs else None,
            "median_jaccard_indep": statistics.median([p["jaccard_indep"] for p in pairs if p["jaccard_indep"] is not None]) if pairs else None,
            "median_ratio": statistics.median(ratios) if ratios else None,
            "median_same_wrong": statistics.median([p["same_wrong_share"] for p in pairs if p["same_wrong_share"] is not None]) if pairs else None,
            "mustask": {"n": len(must), "best": best, "best_errors": len(best_must_err), "best_errors_shared_by_half": shared,
                        "share_failed_by_half": _share(sum(1 for i in must if wrong_count[i] >= len(models) / 2), len(must)),
                        "share_failed_by_all": _share(sum(1 for i in must if wrong_count[i] == len(models)), len(must)),
                        "share_failed_by_none": _share(sum(1 for i in must if wrong_count[i] == 0), len(must))}}


def mustask_pair(lv: Level, pair: Sequence[str], idx: Sequence[int]) -> dict[str, Any]:
    """On the must-ask states: each model's errors, the errors both make, and how many of those are the SAME wrong
    tool (the drift analysis found the 27B models fail the same must-ask state)."""
    import explore_drift as D

    must = [i for i in idx if D._must_ask(lv.units[i])]
    a, b = pair
    ea = {i for i in must if not lv.ok(i, lv.answers[a][i][0])}
    eb = {i for i in must if not lv.ok(i, lv.answers[b][i][0])}
    both = ea & eb
    return {"pair": list(pair), "n": len(must), "err_a": len(ea), "err_b": len(eb), "both": len(both),
            "same_wrong": sum(1 for i in both if lv.answers[a][i][0] == lv.answers[b][i][0]),
            "tasks_both": len({lv.units[i]["task"] for i in both})}


def hard_slice(run: Path = HARD_RUN) -> dict[str, Any]:
    """DecisionGym-Hard (E7c's 12-customer setting, calibration seeds 5300-5399): the frozen AgentDecision-4B and Jev
    each answered every tool choice in its own shadow arm (Flash executes); the states both arms reached (same task,
    state text and options) carry both models' answers. Accuracy, the two-member vote (larger confidence wins a
    disagreement) and the agreement gate, on the tier rule's cancel-or-ship states (the gold next tool is a write) and
    on the other states."""
    from jevharness.bench import metrics as M

    arms = {"jev": run / "jev_shadow_cal.jsonl", "rl": run / "rl_shadow_cal.jsonl"}
    if not all(p.exists() for p in arms.values()):
        return {"skipped": "E7c's shadow calibration arms are not in runs/e7c_cal"}
    states: dict[str, dict[str, dict[str, Any]]] = {k: {} for k in arms}
    for k, p in arms.items():
        lat: dict[str, dict[str, Any]] = {}
        for r in _jl(p):
            if isinstance(r, dict) and "task_id" in r:
                lat[str(r["task_id"])] = r
        for tid, tr in lat.items():
            if tr.get("error"):
                continue
            for st in tr.get("steps") or []:
                rec = next((d for d in st.get("decisions") or [] if d.get("dp") == "tool_select" and d.get("state")
                            and d.get("options") and d.get("s1_value") is not None), None)
                if rec is None or not M.has_gold(rec):
                    continue
                key = hashlib.sha256(json.dumps([tid, rec["state"], list(rec["options"])]).encode()).hexdigest()
                states[k].setdefault(key, {"task": tid, "gold": rec.get("gold"), "rec": rec,
                                           "v": rec.get("s1_value"), "c": float(rec.get("s1_conf") or 0.0)})
    common = sorted(set(states["jev"]) & set(states["rl"]))
    rows = []
    for key in common:
        a, b = states["jev"][key], states["rl"][key]
        rec = a["rec"]
        ja, ra = a["v"], b["v"]
        agree = ja == ra
        vote_v = ja if agree else (ja if a["c"] >= b["c"] else ra)
        rows.append({"tier": rec.get("gold") in OP_TOOLS, "jev": bool(M.is_correct(rec, ja)), "rl": bool(M.is_correct(rec, ra)),
                     "vote": bool(M.is_correct(rec, vote_v)), "agree": agree, "agree_ok": bool(agree and M.is_correct(rec, ja)),
                     "task": a["task"]})

    def blk(sub: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        acc = [r for r in sub if r["agree"]]
        return {"n": len(sub), "tasks": len({r["task"] for r in sub}),
                "jev": _mean([float(r["jev"]) for r in sub]), "rl": _mean([float(r["rl"]) for r in sub]),
                "vote": _mean([float(r["vote"]) for r in sub]), "gate_coverage": _share(len(acc), len(sub)),
                "gate_accepted_errors": sum(1 for r in acc if not r["agree_ok"]),
                "gate_accepted_acc": _share(sum(1 for r in acc if r["agree_ok"]), len(acc))}
    return {"run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
            "states": {k: len(v) for k, v in states.items()}, "common": len(common),
            "tier": blk([r for r in rows if r["tier"]]), "other": blk([r for r in rows if not r["tier"]]),
            "all": blk(rows)}


def offline() -> dict[str, Any]:
    choice = json.loads((OUT / "choice.json").read_text())
    writes = _writes()
    res: dict[str, Any] = {"label": LABEL_A, "generated": _now(), "git": _git(), "split": json.loads((OUT / "split.json").read_text())["rule"],
                           "choice_written": choice["written"], "levels": {}}
    for name, loader in (("c", load_c), ("l1", load_l1)):
        lv = loader()
        dev, test = halves(lv)
        ch = choice["levels"][name]
        L: dict[str, Any] = {"test_units": len(test), "dev_units": len(dev), "families": {}, "refs": {}}
        if name == "c":
            L["test_tasks"] = len({lv.units[i]["task"] for i in test})
        score = (lambda vals: c_scores(lv, vals, test, writes)) if name == "c" else (lambda vals: l1_scores(lv, vals, test))
        for fam, cf in ch["families"].items():
            F: dict[str, Any] = {}
            bs = cf.get("best_single")
            if bs:
                F["best_single"] = {"model": bs, **score([v for v, _ in lv.answers[bs]])}
            for rule in RULES:
                if rule in cf:
                    ens = ensemble_answers(lv, cf[rule]["members"], rule)
                    F[rule] = {"members": cf[rule]["members"], "dev_acc": cf[rule]["dev_acc"], **score(_vals(ens))}
            if "gate" in cf:
                F["gate"] = {**gate_block(lv, cf["gate"]["members"], cf["gate"]["m"], test, writes if name == "c" else None,
                                          [bs, *TRAINED_REF] if bs else list(TRAINED_REF)), "dev": cf["gate"]["dev"],
                             "rule": cf["gate"]["rule"]}
            L["families"][fam] = F
        for r in TRAINED_REF:
            if r in lv.answers:
                L["refs"][r] = {**score([v for v, _ in lv.answers[r]]),
                                "gate": single_gate(lv, r, TAU_REF[r], test, writes if name == "c" else None)}
        # our trained 4B with Jev: the pair the DecisionGym-Hard slice reads (no selection)
        pair = ["rl", "jev"]
        if all(m in lv.answers for m in pair):
            L["pair_rl_jev"] = {"weighted": score(_vals(ensemble_answers(lv, pair, "weighted"))),
                                "gate": gate_block(lv, pair, 2, test, writes if name == "c" else None, pair)}
        if all(m in lv.answers for m in FAMILIES["hosted"]):
            L["hosted_sweep"] = sweep(lv, FAMILIES["hosted"], test, ["oaidec", "jev"])
        if name == "c":
            ots = [m for m in FAMILIES["ots"] if m in lv.answers]
            top = sorted(ots, key=lambda m: -(accuracy(lv, [v for v, _ in lv.answers[m]], list(range(len(lv.units)))) or 0))
            L["errcorr"] = {"all_ots": error_correlation(lv, ots, list(range(len(lv.units)))),
                            "top6_ots": error_correlation(lv, top[:6], list(range(len(lv.units))))}
            L["errcorr"]["mustask_27b"] = mustask_pair(lv, ("openjevhf", "sl27b"), list(range(len(lv.units))))
            L["errcorr"]["note"] = "all of C (5,314 occurrences; no selection involved)"
        res["levels"][name] = L
    # cross-level transfer of the chosen members (descriptive)
    lv1, lvc = load_l1(), load_c()
    _, t1 = halves(lv1)
    _, tc = halves(lvc)
    res["transfer"] = {}
    for fam in FAMILIES:
        for rule in RULES:
            mc = choice["levels"]["c"]["families"][fam].get(rule, {}).get("members")
            m1 = choice["levels"]["l1"]["families"][fam].get(rule, {}).get("members")
            if mc and all(m in lv1.answers for m in mc):
                res["transfer"][f"{fam}/{rule}/c_members_on_l1"] = accuracy(lv1, _vals(ensemble_answers(lv1, mc, rule)), t1)
            if m1 and all(m in lvc.answers for m in m1):
                res["transfer"][f"{fam}/{rule}/l1_members_on_c"] = accuracy(lvc, _vals(ensemble_answers(lvc, m1, rule)), tc)
    res["hard"] = hard_slice()
    return res


def cmd_offline(a: argparse.Namespace) -> int:
    res = offline()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offline.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({lv: {f: {k: (round(100 * v["acc"], 2) if isinstance(v, dict) and v.get("acc") is not None else None)
                               for k, v in F.items()} for f, F in d["families"].items()} for lv, d in res["levels"].items()},
                     indent=1))
    return 0



# ================================================================== PART B: live (registered)
CAL_SRC = RUNS / "e2a_jevgym_shadow_cal_rl"
CAL_ARM = "rl_shadow_cal"
CAL_DIR = OUT / "cal"                                  # fits + gzipped member answers (committed)
CAL_WORK = RUNS / "e20_cal"                            # requests, raw answers, derived arm copies (untracked)
CAL_REQ = CAL_WORK / "cal_requests.jsonl.gz"
E18_LABELLED = ROOT / "results/explore/e17_e18/e18_labelled.jsonl.gz"   # the labelled states E18 used (keys)
DPS = ("tool_select", "tool_params")
MEMBER_SPECS: dict[str, dict[str, Any]] = {
    "oaidec": {"type": "openai_decisions", "model": "gpt-6-luna", "timeout": 120.0},
    "jev": {"type": "jev", "provider": "openrouter", "timeout": 120.0},
}
ROWS: dict[str, tuple[str, ...]] = {"oaidec": ("oaidec",), "agree": ("jev", "oaidec")}
ROW_ARM = {"oaidec": "oaidec_final", "agree": "agree_final"}
ROW_LABEL = {"oaidec": "OpenAI Decisions API behind the gate", "agree": "Jev and the Decisions API, agreement gate"}
LIVE_RUNS = ("e20_live_smoke_oaidec", "e20_live_oaidec", "e20_live_smoke_agree", "e20_live_agree")
CAPS = {"gemini": 25.0, "openai": 10.0, "jev": 3.0}
PROFILE = ROOT / "configs/harness/explore_e20_final.yaml"
FINAL_PROFILE = ROOT / "configs/harness/jevagent_final.yaml"
LOGGED_B = RUNS / "harness_final_dedicated"           # the logged dedicated rows (seeds 300500-300699, 2026-10-05)
PROBE_LOG = OUT / "api_probe_log.txt"
OVERLOAD = ("503", "overload", "high demand", "unavailable", "resource_exhausted", "429")
PROVIDER_OF = {"gemini": "gemini", "openai_decisions": "openai", "jev": "jev"}


def provider_of(call: Mapping[str, Any]) -> str:
    b = str(call.get("backend") or "")
    if b in PROVIDER_OF:
        return PROVIDER_OF[b]
    if call.get("component") == "s2":
        return "gemini"
    return "other"


# ------------------------------------------------------------------ spend (every record of every attempt is paid)
def run_spend(run: Path) -> dict[str, Any]:
    out: dict[str, Any] = {p: 0.0 for p in (*CAPS, "other")}
    out["records"] = 0
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        for t in _jl(p):
            if not (isinstance(t, dict) and "task_id" in t):
                continue
            out["records"] += 1
            for c in t.get("calls") or []:
                out[provider_of(c)] += float(c.get("cost_usd") or 0.0)
    return out


def cal_spend(work: Path = CAL_WORK, committed: Path = CAL_DIR) -> dict[str, float]:
    """Every calibration answer's logged cost: the working log, else its committed gzip copy (``cal-fit``)."""
    out = {p: 0.0 for p in CAPS}
    for m in MEMBER_SPECS:
        f = work / f"answers_{m}.jsonl"
        if not f.exists():
            f = committed / f"answers_{m}.jsonl.gz"
        if f.exists():
            prov = "openai" if m == "oaidec" else "jev"
            out[prov] += sum(float(r.get("cost_usd") or 0.0) for r in _jl(f))
    return out


def total_spend(runs: Path = RUNS, work: Path = CAL_WORK) -> dict[str, Any]:
    per = {name: run_spend(runs / name) for name in LIVE_RUNS if (runs / name).is_dir()}
    cal = cal_spend(work)
    tot = {p: cal.get(p, 0.0) + sum(v[p] for v in per.values()) for p in CAPS}
    return {"by_run": per, "calibration": cal, "total": tot, "records": sum(v["records"] for v in per.values())}


def cmd_spend(a: argparse.Namespace) -> int:
    s = total_spend()
    if a.short:
        print(" ".join(f"{p} ${v:.3f}/{CAPS[p]:.0f}" for p, v in s["total"].items()) + f" ({s['records']} records)")
    else:
        print(json.dumps(s, indent=1))
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    """Registration (d): per provider, the smoke's mean $ per episode x 200 x 1.5 + the spend so far fits the cap."""
    sm = run_spend(Path(a.smoke))
    tot = total_spend()["total"]
    n = max(1, sm["records"])
    res = {"smoke": str(a.smoke), "records": sm["records"], "at": _now(), "n": a.n, "providers": {}}
    go = sm["records"] > 0
    for p, cap in CAPS.items():
        proj = sm[p] / n * a.n * 1.5 + tot[p]
        res["providers"][p] = {"usd_per_episode": sm[p] / n, "spent": tot[p], "projection": proj, "cap": cap,
                               "ok": proj <= cap}
        go = go and proj <= cap
    res["go"] = go
    print(json.dumps(res))
    return 0 if go else 1


def cmd_probe(a: argparse.Namespace) -> int:
    import explore_e14_small as E14

    return E14.cmd_probe(argparse.Namespace(log=str(PROBE_LOG)))


def _latest_by_arm(run: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        d: dict[str, dict[str, Any]] = {}
        for t in _jl(p):
            if isinstance(t, dict) and "task_id" in t:
                d[str(t["task_id"])] = t
        out[p.stem] = d
    return out


def _overload(t: Mapping[str, Any]) -> bool:
    e = str(t.get("error") or "").lower()
    return bool(e) and any(k in e for k in OVERLOAD)


def cmd_errored(a: argparse.Namespace) -> int:
    print(sum(1 for d in _latest_by_arm(Path(a.run)).values() for t in d.values() if t.get("error")))
    return 0


def _kill(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        time.sleep(5)


def cmd_watch(a: argparse.Namespace) -> int:
    """Per-provider caps over everything E20 logged (-> STOP file, runner stopped) and a pause on a new episode that
    errored with an overload (-> <run>/PAUSED_overload.json); the runner's process group is stopped; records kept."""
    run = RUNS / a.run
    seen = {(arm, k) for arm, d in _latest_by_arm(run).items() for k, t in d.items() if t.get("error")} if run.is_dir() else set()
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{_now()} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend()
        over = {p: v for p, v in s["total"].items() if v >= CAPS[p]}
        if over:
            (RUNS / f"e20_STOP_{a.run}.json").write_text(json.dumps({"at": _now(), "spend": s, "over": over}, indent=1))
            _kill(a.pid)
            print(f"CAP REACHED: {over}; runner stopped", flush=True)
            return 1
        if run.is_dir():
            new = [(arm, k, t) for arm, d in _latest_by_arm(run).items() for k, t in d.items()
                   if t.get("error") and (arm, k) not in seen and _overload(t)]
            if new:
                (run / "PAUSED_overload.json").write_text(json.dumps(
                    {"at": _now(), "episodes": [{"arm": arm, "task": k, "error": str(t.get("error"))[:300]}
                                                for arm, k, t in new]}, indent=1))
                _kill(a.pid)
                print(f"{_now()} PAUSED: {len(new)} overload error(s) in {run.name}; runner stopped", flush=True)
                return 2
        print(f"{_now()} " + " ".join(f"{p} ${v:.3f}" for p, v in s["total"].items()) + f" ({s['records']} records)",
              flush=True)
        time.sleep(a.every)


# ------------------------------------------------------------------ calibration (registration (c))
def cmd_cal_build(a: argparse.Namespace) -> int:
    import e5_cal_replay as R

    print(json.dumps(R.build(CAL_SRC, CAL_ARM, CAL_REQ)))
    return 0


def answer_one(shell: Any, q: Mapping[str, Any]) -> dict[str, Any]:
    """One calibration question through the harness's asking path (E5's ``e5_cal_replay.answer``), priced."""
    from jevharness.accounting import Ledger
    from jevharness.decision.types import Choice

    led = Ledger()
    rec: dict[str, Any] = {"key": q["key"], "dp": q["dp"], "value": None, "conf": None, "dist": None, "refusal": False,
                           "error": None}
    t0 = time.perf_counter()
    try:
        if q["dp"] == "tool_select":
            ans = shell._choice_tournament(q["state"], q["text"], list(q["options"]), "tool_select", led,
                                           q.get("descriptions") or None)
        else:
            ans = shell._ask(q["state"], Choice(q["text"], list(q["options"]), decision_point="tool_params"), led)
        rec.update({"value": ans.value, "conf": float(ans.confidence), "refusal": bool((ans.raw or {}).get("refusal")),
                    "dist": {str(k): float(v) for k, v in ans.distribution.items()}})
    except Exception as e:  # noqa: BLE001 - a failed question is recorded (no answer), never a crash
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec.update({"cost_usd": sum(r.cost_usd for r in led.records), "latency_ms": sum(r.latency_ms for r in led.records),
                "input_tokens": sum(r.input_tokens for r in led.records), "requests": len(led.records),
                "wall_ms": (time.perf_counter() - t0) * 1000})
    return rec


def cmd_cal_answer(a: argparse.Namespace) -> int:
    """Every calibration question answered once by one member (resumes: keys already answered without an error are
    skipped; an errored key is asked again once per pass); stops before a provider's cap."""
    import threading as th
    from concurrent.futures import ThreadPoolExecutor

    from dotenv import load_dotenv

    from jevharness.decision import make_backend
    from jevharness.harness.agent import JevAgent
    from jevharness.harness.config import DPConfig, HarnessConfig, Mode

    load_dotenv(ROOT / ".env")
    m = a.member
    out = CAL_WORK / f"answers_{m}.jsonl"
    done = {r["key"] for r in _jl(out) if r.get("error") is None} if out.exists() else set()
    labelled = {r["key"] for r in _jl(E18_LABELLED)}            # registration (c): the 2,453 labelled questions
    qs = [q for q in _jl(CAL_REQ) if q["key"] in labelled and q["key"] not in done]
    if a.limit:
        qs = qs[: a.limit]
    backend = make_backend(dict(MEMBER_SPECS[m]))
    shell = JevAgent(llm=None, cfg=HarnessConfig(name="e20_cal", tool_select=DPConfig(mode=Mode.REPLACE)), s1=backend)
    prov = "openai" if m == "oaidec" else "jev"
    lock = th.Lock()
    spent = [total_spend()["total"][prov]]
    stop = [False]
    n = [0]

    def work(q: Mapping[str, Any]) -> None:
        if stop[0]:
            return
        rec = answer_one(shell, q)
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            spent[0] += rec["cost_usd"]
            n[0] += 1
            if spent[0] >= CAPS[prov] * 0.95:
                stop[0] = True
            if n[0] % 200 == 0:
                print(f"{_now()} {m}: {n[0]}/{len(qs)} answered, {prov} ${spent[0]:.3f}", flush=True)
    CAL_WORK.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, qs))
    recs = {r["key"]: r for r in _jl(out)}
    print(json.dumps({"member": m, "answered": len(recs), "errors": sum(1 for r in recs.values() if r.get("error")),
                      "refusals": sum(1 for r in recs.values() if r.get("refusal")), "spent": spent[0],
                      "stopped_at_cap": stop[0]}))
    return 1 if stop[0] else 0


def _answer_obj(q: Mapping[str, Any], r: Mapping[str, Any] | None) -> Any:
    from jevharness.decision.types import Answer, Choice

    if r is None or r.get("error") or r.get("value") is None:
        return None
    qq = Choice(q["text"], list(q["options"]), decision_point=q["dp"])
    return Answer(question_id=qq.id, kind=qq.kind, distribution=dict(r.get("dist") or {}), value=r["value"],
                  confidence=float(r.get("conf") or 0.0), raw={"refusal": True} if r.get("refusal") else None)


def row_answers(row: str, work: Path = CAL_WORK, reqs: Path = CAL_REQ) -> list[dict[str, Any]]:
    """The calibration answers of a row in E5's answer format: a single member's own, or the ensemble's computed from
    its members' answers by ``decision.ensemble.combine`` (the live backend's rule). A refusal is no answer."""
    from jevharness.decision.ensemble import combine
    from jevharness.decision.types import Choice

    mem = {m: {r["key"]: r for r in _jl(work / f"answers_{m}.jsonl")} for m in ROWS[row]}
    out = []
    for q in _jl(reqs):
        per = [mem[m].get(q["key"]) for m in ROWS[row]]
        if len(ROWS[row]) == 1:
            r = per[0]
            if r is None or r.get("error") or r.get("refusal"):
                out.append({"key": q["key"], "dp": q["dp"], "value": None, "conf": None, "dist": None,
                            "error": (r or {}).get("error") or ("refusal" if r else "missing")})
            else:
                out.append({"key": q["key"], "dp": q["dp"], "value": r["value"], "conf": r["conf"], "dist": r["dist"],
                            "error": None, "latency_ms": r.get("latency_ms")})
            continue
        qq = Choice(q["text"], list(q["options"]), decision_point=q["dp"])
        a = combine(qq, [_answer_obj(q, r) for r in per], "agree", list(ROWS[row]))
        out.append({"key": q["key"], "dp": q["dp"], "value": a.value, "conf": a.confidence, "dist": a.distribution,
                    "error": None, "agree": bool((a.raw or {}).get("ensemble", {}).get("agree")),
                    "latency_ms": max(float((r or {}).get("latency_ms") or 0.0) for r in per)})
    return out


def cal_coverage(arm_path: Path, taus: Mapping[str, float | None]) -> dict[str, Any]:
    """Coverage of a row at its fitted thresholds on the calibration states it was fitted on: accepted = answered,
    not OTHER (arguments) and confidence >= tau; accepted errors by the registered rule."""
    from jevharness.bench import metrics as M
    from jevharness.harness.tools import OTHER

    out: dict[str, Any] = {}
    recs = [r for r in _jl(arm_path)] if arm_path.exists() else []
    for dp in DPS:
        tau = taus.get(dp)
        ds = [d for r in recs for s in r.get("steps") or [] for d in s.get("decisions") or []
              if d.get("dp") == dp and d.get("options") and M.has_gold(d)]
        ans = [d for d in ds if d.get("s1_value") is not None]
        acc = [d for d in ans if tau is not None and math.isfinite(tau) and float(d.get("s1_conf") or 0) >= tau
               and (dp == "tool_select" or d.get("s1_value") != OTHER)]
        wrong = sum(1 for d in acc if M.is_correct(d, d["s1_value"]) is False)
        right_all = sum(1 for d in ans if M.is_correct(d, d["s1_value"]))
        out[dp] = {"n": len(ds), "answered": len(ans), "acc": _share(right_all, len(ds)), "tau": tau,
                   "accepted": len(acc), "coverage": _share(len(acc), len(ds)), "accepted_errors": wrong,
                   "accepted_acc": _share(len(acc) - wrong, len(acc))}
    return out


def cmd_cal_fit(a: argparse.Namespace) -> int:
    import e5_cal_replay as R

    CAL_DIR.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"label": LABEL_B, "at": _now(), "rows": {}}
    for row in ROWS:
        ans = row_answers(row)
        path = CAL_WORK / f"row_{row}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in ans))
        name = f"e20cal_{row}"
        R.derive(CAL_SRC, CAL_ARM, path, CAL_WORK / "arms", name, model=row)
        fits = R.fit(CAL_WORK / "arms", name, CAL_DIR)
        taus = {dp: (None if fits[dp]["tau_hat"] is None else float(fits[dp]["tau_hat"])) for dp in DPS}
        res["rows"][row] = {"fits": fits, "coverage": cal_coverage(CAL_WORK / "arms" / f"{name}.jsonl", taus),
                            "answers": len(ans), "no_answer": sum(1 for r in ans if r["value"] is None)}
        if row == "agree":
            res["rows"][row]["agree_share"] = {dp: _share(sum(1 for r in ans if r["dp"] == dp and r.get("agree")),
                                                          sum(1 for r in ans if r["dp"] == dp)) for dp in DPS}
    for m in MEMBER_SPECS:
        f = CAL_WORK / f"answers_{m}.jsonl"
        if f.exists():
            recs = list({r["key"]: r for r in _jl(f)}.values())
            res.setdefault("members", {})[m] = {"answers": len(recs), "errors": sum(1 for r in recs if r.get("error")),
                                                "refusals": sum(1 for r in recs if r.get("refusal")),
                                                "cost_usd": sum(float(r.get("cost_usd") or 0) for r in _jl(f)),
                                                "sha256": sha256_file(f)}
            import shutil
            with open(f, "rb") as fi, gzip.GzipFile(str(CAL_DIR / f"answers_{m}.jsonl.gz"), "wb", mtime=0) as fo:
                shutil.copyfileobj(fi, fo)
    (CAL_DIR / "fits.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({r: {dp: (v["fits"][dp]["tau_hat"], v["coverage"][dp]["coverage"], v["coverage"][dp]["accepted_errors"])
                          for dp in DPS} for r, v in res["rows"].items()}, indent=1, default=str))
    return 0


def cal_facts(answers_dir: Path = CAL_DIR) -> dict[str, Any]:
    """Descriptive, no fit: each member's accuracy on the labelled calibration states, its answers at confidence 1.0,
    the members' agreement and the agreed answers at a smaller confidence of at least 0.99, against the live LLM's
    logged answers (registered correctness, ``bench/metrics.is_correct``)."""
    import e5_cal_replay as R
    from jevharness.bench import metrics as M

    lab = {r["key"]: r for r in _jl(E18_LABELLED)}
    mem = {m: {r["key"]: r for r in _jl(answers_dir / f"answers_{m}.jsonl.gz")} for m in MEMBER_SPECS
           if (answers_dir / f"answers_{m}.jsonl.gz").exists()}
    if len(mem) < 2:
        return {}
    dec: dict[str, Mapping[str, Any]] = {}
    for tid, r in R.load_latest(CAL_SRC / f"{CAL_ARM}.jsonl").items():
        for st in r.get("steps") or []:
            for i, d in enumerate(st.get("decisions") or []):
                k = R.key_of(tid, st.get("step"), i)
                if k in lab:
                    dec[k] = d

    def right(k: str, v: Any) -> bool:
        return bool(v is not None and M.is_correct(dec[k], v))
    out: dict[str, Any] = {}
    for dp in DPS:
        ks = [k for k in lab if lab[k]["dp"] == dp and k in dec]
        b: dict[str, Any] = {"n": len(ks), "llm_right": sum(1 for k in ks if lab[k]["llm_ok"])}
        for m, ans in mem.items():
            one = [k for k in ks if float(ans[k].get("conf") or 0) >= 1.0]
            b[m] = {"right": sum(right(k, ans[k].get("value")) for k in ks), "conf1": len(one),
                    "conf1_wrong": sum(1 for k in one if not right(k, ans[k].get("value")))}
        if {"jev", "oaidec"} <= set(mem):
            ag = [k for k in ks if mem["jev"][k].get("value") == mem["oaidec"][k].get("value")]
            ag99 = [k for k in ag if min(float(mem["jev"][k].get("conf") or 0), float(mem["oaidec"][k].get("conf") or 0)) >= 0.99]
            b["agree"] = {"n": len(ag), "right": sum(right(k, mem["jev"][k].get("value")) for k in ag),
                          "n99": len(ag99), "wrong99": sum(1 for k in ag99 if not right(k, mem["jev"][k].get("value")))}
        out[dp] = b
    return out


def cmd_prepare_live(a: argparse.Namespace) -> int:
    """The profile of registration (b): jevagent_final.yaml plus backends.<row> with the fitted thresholds (an
    'always defer' fit -> always_defer at that point; both points -> skip)."""
    import yaml

    fits = json.loads((CAL_DIR / "fits.json").read_text())
    prof = yaml.safe_load(FINAL_PROFILE.read_text())
    prof["name"] = "explore_e20_final"
    prof["description"] = ("Exploratory row E20 (PREREG_A7_STAGE2 Deviations 2026-10-08 'E20'): the final harness with "
                           "a hosted decision model or an agreement gate of two hosted decision models as System-1.")
    sec = prof["system2"]["gemini-3.5-flash"]
    sec["backends"] = {}
    for row in ROWS:
        blk: dict[str, Any] = {}
        n_def = 0
        for dp in DPS:
            t = fits["rows"][row]["fits"][dp]["tau_hat"]
            t = None if t is None else float(t)
            if t is None or not math.isfinite(t) or t > 1.0:
                blk[dp] = {"tau": 1.0, "always_defer": True}
                n_def += 1
            else:
                blk[dp] = {"tau": t}
        if n_def == len(DPS):
            blk = {"skip": "E20 (c): both fitted thresholds always defer (the row equals LLM-only; not run live)"}
        elif blk["tool_select"].get("always_defer"):
            # amendment "E20: the fitted thresholds": System-1 fills arguments only for a tool System-1 chose
            # (JevAgent._delegated_call), so with the tool choice always deferring no System-1 decision is reachable
            blk["skip"] = ("E20 amendment (the fitted thresholds): the tool choice always defers and the argument "
                           "cascade is reached only through a System-1 tool choice, so the row makes no System-1 "
                           "call and equals LLM-only; not run live")
        blk["fitted_from"] = {}
        for dp in DPS:
            fp = Path(fits["rows"][row]["fits"][dp]["path"])
            fp = fp if fp.is_absolute() else ROOT / fp
            blk["fitted_from"][dp] = {"path": str(fp.relative_to(ROOT)), "sha256": sha256_file(fp)}
        blk["notes"] = f"E20 row {row}: {ROW_LABEL[row]}; fitted by the registered command on rl_shadow_cal (E20 (c))"
        sec["backends"][row] = blk
    head = ("# Exploratory row E20 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 \"E20: decision-model ensembles and a\n"
            "# hosted decision API behind the gate\" (b), (c)); written by `scripts/explore_e20_ensembles.py prepare-live`\n"
            "# from results/explore/e20_ensembles/cal/fits.json before any E20 smoke or live episode. A copy of\n"
            "# configs/harness/jevagent_final.yaml (tool + argument cascade, reflection off, argument questions batched per\n"
            "# step) with one backends.<row> block per E20 row.\n")
    PROFILE.write_text(head + yaml.safe_dump(prof, sort_keys=False))
    print(PROFILE.read_text())
    return 0



# ------------------------------------------------------------------ live readout (registration (e), (f))
def episode_extra(r: Mapping[str, Any]) -> dict[str, Any]:
    """Per episode: API $ by provider, the parallel System-1 time (the two calls of one ensemble decision overlap: the
    slower one counts), accepted System-1 decisions and accepted errors per decision point, refusals, disagreements."""
    from jevharness.bench import metrics as M
    from jevharness.harness.tools import OTHER

    calls = r.get("calls") or []
    prov = {p: 0.0 for p in CAPS}
    for c in calls:
        pv = provider_of(c)
        if pv in prov:
            prov[pv] += float(c.get("cost_usd") or 0.0)
    groups: dict[Any, float] = {}
    s1_par = 0.0
    for c in calls:
        if c.get("component") != "s1":
            continue
        g = ((c.get("meta") or {}).get("ensemble") or {}).get("group")
        if g is None:
            s1_par += float(c.get("latency_ms") or 0.0)
        else:
            groups[g] = max(groups.get(g, 0.0), float(c.get("latency_ms") or 0.0))
    s1_par += sum(groups.values())
    out: dict[str, Any] = {"prov": prov, "s1_par_s": s1_par / 1000}
    for dp in DPS:
        ds = [d for st in r.get("steps") or [] for d in st.get("decisions") or []
              if d.get("dp") == dp and not d.get("shadow") and d.get("mode") in ("cascade", "replace")]
        asked = [d for d in ds if d.get("s1_value") is not None]
        acc = [d for d in ds if not d.get("deferred")]
        out[dp] = {"n": len(ds), "asked": len(asked), "accepted": len(acc),
                   "accepted_labelled": sum(1 for d in acc if M.is_correct(d, d.get("s1_value")) is not None),
                   "accepted_errors": sum(1 for d in acc if M.is_correct(d, d.get("s1_value")) is False),
                   "zero_conf": sum(1 for d in asked if float(d.get("s1_conf") or 0.0) == 0.0),
                   "other": sum(1 for d in asked if d.get("s1_value") == OTHER),
                   "refusals": sum(1 for d in ds if (d.get("gold_aux") or {}).get("s1_refusal"))}
    return out


def live(runs: Path = RUNS) -> dict[str, Any]:
    import numpy as np

    import explore_common as C
    import explore_e14_small as E14
    import harness_final as HF

    tar = json.loads(E14.TARIFF_DED.read_text())
    host_h = float(tar["host_usd_per_hour"])
    rows = {ROW_ARM[r]: runs / f"e20_live_{r}" / f"{ROW_ARM[r]}.jsonl" for r in ROWS}
    refs = {"llm_only": LOGGED_B / "llm_only.jsonl", "harness_final": LOGGED_B / "harness_final.jsonl",
            "e18_final": runs / "e18_live" / "e18_final.jsonl"}
    allp = {k: f for k, f in {**rows, **refs}.items() if f.exists()}
    if not any(k in allp for k in rows):
        return {}
    lat: dict[str, dict[str, dict[str, Any]]] = {}
    res: dict[str, Any] = {"host_usd_per_hour": host_h, "arms": {}, "contrasts": {}, "errored": {}, "n_latest": {},
                           "reference_note": "llm_only, harness_final: logged 2026-10-05 (runs/harness_final_dedicated); "
                                             "e18_final: logged 2026-10-08 (runs/e18_live)"}
    for k, f in allp.items():
        latest_, _ = C.load_latest([f])
        res["errored"][k] = sum(1 for r in latest_.values() if r.get("error"))
        res["n_latest"][k] = len(latest_)
        lat[k] = {t: r for t, r in latest_.items() if not r.get("error")}
    gpu = {"harness_final", "e18_final"}
    for k, d in lat.items():
        ts = sorted(d)
        eps = [HF._episode_live(d[t]) for t in ts]
        ex = [episode_extra(d[t]) for t in ts]
        wall = np.array([e["wall_s"] for e in eps])
        api = np.array([e["cost"] for e in eps])
        q = host_h / 3600 * wall if k in gpu else np.zeros_like(wall)
        tot = {x: sum(e[x] for e in eps) for x in ("tool_decisions", "tool_by_s1", "arg_decisions", "arg_by_s1")}
        blk = {"n": len(ts), "success": float(np.mean([e["success"] for e in eps])), "api_usd": float(api.mean()),
               "usd_incl_gpu": float((api + q).mean()), "wall_s_mean": float(wall.mean()), "wall_s_median": float(np.median(wall)),
               "s2_s": float(np.mean([e["s2_s"] for e in eps])), "s1_s": float(np.mean([e["s1_s"] for e in eps])),
               "s1_par_s": float(np.mean([x["s1_par_s"] for x in ex])), "s2_calls": float(np.mean([e["s2_calls"] for e in eps])),
               "steps": float(np.mean([e["steps"] for e in eps])),
               "prov_usd": {p: float(np.mean([x["prov"][p] for x in ex])) for p in CAPS},
               "tool_coverage": _share(tot["tool_by_s1"], tot["tool_decisions"]),
               "arg_coverage": _share(tot["arg_by_s1"], tot["arg_decisions"])}
        for dp in DPS:
            blk[dp] = {f: sum(x[dp][f] for x in ex) for f in ("n", "asked", "accepted", "accepted_labelled", "accepted_errors",
                                                               "zero_conf", "other", "refusals")}
        res["arms"][k] = blk
    for k in rows:
        if k not in lat:
            continue
        for ref in refs:
            if ref not in lat:
                continue
            common = sorted(set(lat[k]) & set(lat[ref]))
            if not common:
                continue
            a = [HF._episode_live(lat[k][t]) for t in common]
            b = [HF._episode_live(lat[ref][t]) for t in common]
            sa, sb = np.array([e["success"] for e in a]), np.array([e["success"] for e in b])
            ca = np.array([e["cost"] for e in a])
            wb = np.array([e["wall_s"] for e in b])
            cb = np.array([e["cost"] for e in b]) + (host_h / 3600 * wb if ref in gpu else 0.0)
            nc = C.newcombe(sa, sb)
            pb = C.paired_boot(sa, sb)
            sv = C.paired_boot(ca, cb, ratio=True)
            wl = C.paired_boot(np.array([e["wall_s"] for e in a]), wb)
            res["contrasts"][f"{k}_vs_{ref}"] = {
                "n": len(common), "success": {"delta": nc["diff_pt"] / 100, "lo": nc["ci_pt"][0] / 100, "hi": nc["ci_pt"][1] / 100,
                                              "cells": nc.get("cells")},
                "success_boot": {"delta": pb["est"], "lo": pb["ci"][0], "hi": pb["ci"][1]},
                "saving": {"delta": sv["est"], "lo": sv["ci"][0], "hi": sv["ci"][1]},
                "wall_s": {"delta": wl["est"], "lo": wl["ci"][0], "hi": wl["ci"][1]},
                "time_saving": 1 - float(np.mean([e["wall_s"] for e in a])) / float(wb.mean()) if wb.mean() else None}
    res["readings"] = {}
    for r, k in ROW_ARM.items():
        c = res["contrasts"].get(f"{k}_vs_llm_only")
        if c:
            d = 100 * c["success"]["delta"]
            res["readings"][r] = {"success_minus_llm_pt": d, "saving": c["saving"]["delta"],
                                  "holds": bool(d >= -3.0 and (c["saving"]["delta"] or 0) > 0),
                                  "rule": "success - llm_only >= -3.0 points and saving > 0 (registration (f))"}
    return res


# ------------------------------------------------------------------ summary, markdown, TeX
def _pc(x: Any, nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.{nd}f}"


def _k(v: Any, nd: int = 1, pct: bool = True, sign: bool = False) -> str | None:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    x = 100 * float(v) if pct else float(v)
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") or "-" in s else s


def _tc(n: Any) -> str:
    return f"{int(round(float(n))):,}".replace(",", "{,}")


def _tau(t: Any) -> str | None:
    if t is None:
        return None
    t = float(t)
    if not math.isfinite(t):
        return r"$\infty$"
    return f"{t:.2f}" if abs(t - round(t, 2)) < 5e-4 else f"{t:.4f}"


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {}

    def put(key: str, val: str | None) -> None:
        if val is not None:
            k[f"e20/{key}"] = val
    off = r.get("offline") or {}
    for lvn, L in (off.get("levels") or {}).items():
        put(f"a/{lvn}/test_units", _tc(L["test_units"]))
        if L.get("test_tasks"):
            put(f"a/{lvn}/test_tasks", str(L["test_tasks"]))
        for fam, F in L["families"].items():
            for name, b in F.items():
                pre = f"a/{lvn}/{fam}/{'single' if name == 'best_single' else name}"
                if name == "gate":
                    put(f"{pre}/cov", _k(b["coverage"]))
                    put(f"{pre}/err", _tc(b["accepted_errors"]))
                    put(f"{pre}/accacc", _k(b["accepted_acc"]))
                    put(f"{pre}/m", str(b["m"]))
                    put(f"{pre}/k", str(len(b["members"])))
                    put(f"{pre}/pred", _k(b.get("pred_live")))
                    put(f"{pre}/mustacc_n", None if b.get("mustask_accepted") is None else _tc(b["mustask_accepted"]))
                    put(f"{pre}/musterr", None if b.get("mustask_accepted_errors") is None else _tc(b["mustask_accepted_errors"]))
                    put(f"{pre}/devcov", _k(b["dev"]["coverage"]))
                    put(f"{pre}/deverr", _tc(b["dev"]["accepted_errors"]))
                    for m, e in (b.get("own_conf_at_equal_coverage") or {}).items():
                        put(f"{pre}/own/{m}", f"{e['accepted_errors']:.0f}")
                    continue
                put(f"{pre}/acc", _k(b.get("acc")))
                put(f"{pre}/must", _k(b.get("mustask_acc")))
                put(f"{pre}/pred", _k(b.get("pred_live")))
                for fm in ("toolchoice", "args", "w2c"):
                    put(f"{pre}/{fm}", _k(b.get(fm)))
                if name == "best_single":
                    put(f"{pre}/name", MODELS[b["model"]][0])
                else:
                    put(f"{pre}/k", str(len(b["members"])))
                    put(f"{pre}/devacc", _k(b.get("dev_acc")))
        for m, b in (L.get("refs") or {}).items():
            pre = f"a/{lvn}/ref/{m}"
            put(f"{pre}/acc", _k(b.get("acc")))
            put(f"{pre}/must", _k(b.get("mustask_acc")))
            put(f"{pre}/pred", _k(b.get("pred_live")))
            g = b.get("gate") or {}
            put(f"{pre}/cov", _k(g.get("coverage")))
            put(f"{pre}/err", None if g.get("accepted_errors") is None else _tc(g["accepted_errors"]))
            put(f"{pre}/gpred", _k(g.get("pred_live")))
        pr = L.get("pair_rl_jev")
        if pr:
            put(f"a/{lvn}/pair/acc", _k(pr["weighted"].get("acc")))
            put(f"a/{lvn}/pair/must", _k(pr["weighted"].get("mustask_acc")))
            put(f"a/{lvn}/pair/pred", _k(pr["weighted"].get("pred_live")))
            put(f"a/{lvn}/pair/cov", _k(pr["gate"]["coverage"]))
            put(f"a/{lvn}/pair/err", _tc(pr["gate"]["accepted_errors"]))
            put(f"a/{lvn}/pair/gpred", _k(pr["gate"].get("pred_live")))
        for row in L.get("hosted_sweep") or []:
            t = f"{row['t']:g}".replace(".", "")
            put(f"a/{lvn}/sweep/{t}/cov", _k(row["coverage"]))
            put(f"a/{lvn}/sweep/{t}/err", _tc(row["accepted_errors"]))
            for m, e in row["singles"].items():
                put(f"a/{lvn}/sweep/{t}/{m}", f"{e:.0f}")
        ec = L.get("errcorr")
        if ec:
            for name in ("all_ots", "top6_ots"):
                x = ec[name]
                tag = "all" if name == "all_ots" else "top6"
                put(f"a/corr/{tag}/n_models", str(len(x["models"])))
                put(f"a/corr/{tag}/jac", _k(x["median_jaccard"]))
                put(f"a/corr/{tag}/jacind", _k(x["median_jaccard_indep"]))
                put(f"a/corr/{tag}/ratio", _k(x["median_ratio"], 1, pct=False))
                put(f"a/corr/{tag}/samewrong", _k(x["median_same_wrong"], 0))
                mu = x["mustask"]
                put(f"a/corr/{tag}/must/n", _tc(mu["n"]))
                put(f"a/corr/{tag}/must/failall", _k(mu["share_failed_by_all"], 0))
                put(f"a/corr/{tag}/must/failhalf", _k(mu["share_failed_by_half"], 0))
                put(f"a/corr/{tag}/must/failnone", _k(mu["share_failed_by_none"], 0))
                put(f"a/corr/{tag}/must/best", MODELS[mu["best"]][0])
                put(f"a/corr/{tag}/must/besterr", _tc(mu["best_errors"]))
                put(f"a/corr/{tag}/must/bestshared", _tc(mu["best_errors_shared_by_half"]))
            m27 = ec.get("mustask_27b")
            if m27:
                put("a/corr/must27b/erra", _tc(m27["err_a"]))
                put("a/corr/must27b/errb", _tc(m27["err_b"]))
                put("a/corr/must27b/both", _tc(m27["both"]))
                put("a/corr/must27b/same", _tc(m27["same_wrong"]))
    h = off.get("hard") or {}
    if h.get("common"):
        put("a/hard/common", _tc(h["common"]))
        for part in ("tier", "other", "all"):
            b = h[part]
            put(f"a/hard/{part}/n", _tc(b["n"]))
            put(f"a/hard/{part}/tasks", str(b["tasks"]))
            for f, tag in (("jev", "jev"), ("rl", "rl"), ("vote", "vote"), ("gate_coverage", "gatecov"),
                           ("gate_accepted_acc", "gateacc")):
                put(f"a/hard/{part}/{tag}", _k(b[f]))
            put(f"a/hard/{part}/gateerr", _tc(b["gate_accepted_errors"]))
    for kk, v in (off.get("transfer") or {}).items():
        put(f"a/transfer/{kk}", _k(v))
    # calibration
    cal = r.get("cal") or {}
    for row, b in (cal.get("rows") or {}).items():
        for dp in DPS:
            tag = "tool" if dp == "tool_select" else "arg"
            f = (b.get("fits") or {}).get(dp) or {}
            c = (b.get("coverage") or {}).get(dp) or {}
            put(f"cal/{row}/{tag}/tau", _tau(f.get("tau_hat")))
            put(f"cal/{row}/{tag}/cov", _k(c.get("coverage")))
            put(f"cal/{row}/{tag}/err", None if c.get("accepted_errors") is None else _tc(c["accepted_errors"]))
            put(f"cal/{row}/{tag}/acc", _k(c.get("acc")))
            put(f"cal/{row}/{tag}/n", None if c.get("n") is None else _tc(c["n"]))
        for dp, v in (b.get("agree_share") or {}).items():
            put(f"cal/{row}/{'tool' if dp == 'tool_select' else 'arg'}/agree", _k(v))
    for dp, b in ((cal.get("facts") or {}).items()):
        tag = "tool" if dp == "tool_select" else "arg"
        put(f"cal/facts/{tag}/n", _tc(b["n"]))
        put(f"cal/facts/{tag}/llm", _k(_share(b["llm_right"], b["n"])))
        for m in MEMBER_SPECS:
            if m in b:
                put(f"cal/facts/{tag}/{m}/acc", _k(_share(b[m]["right"], b["n"])))
                put(f"cal/facts/{tag}/{m}/conf1", _tc(b[m]["conf1"]))
                put(f"cal/facts/{tag}/{m}/conf1wrong", _tc(b[m]["conf1_wrong"]))
        if "agree" in b:
            put(f"cal/facts/{tag}/agree", _k(_share(b["agree"]["n"], b["n"])))
            put(f"cal/facts/{tag}/agreeacc", _k(_share(b["agree"]["right"], b["agree"]["n"])))
            put(f"cal/facts/{tag}/agree99", _tc(b["agree"]["n99"]))
            put(f"cal/facts/{tag}/agree99wrong", _tc(b["agree"]["wrong99"]))
    for row, st in (r.get("live_status") or {}).items():
        put(f"live/{row}/status", "run" if st == "run" else "not run")
    # live
    lv = r.get("live") or {}
    for arm, b in (lv.get("arms") or {}).items():
        tag = {"oaidec_final": "oaidec", "agree_final": "agree", "llm_only": "llm", "harness_final": "hf4b",
               "e18_final": "e18"}.get(arm, arm)
        pre = f"live/{tag}"
        put(f"{pre}/n", str(b["n"]))
        put(f"{pre}/errored", str(lv["errored"].get(arm, 0)))
        put(f"{pre}/success", _k(b["success"]))
        put(f"{pre}/api", _k(1000 * b["api_usd"], 2, pct=False))
        put(f"{pre}/tot", _k(1000 * b["usd_incl_gpu"], 2, pct=False))
        for p, v in b["prov_usd"].items():
            put(f"{pre}/{p}", _k(1000 * v, 2, pct=False))
        put(f"{pre}/wall", _k(b["wall_s_mean"], 1, pct=False))
        put(f"{pre}/wallmed", _k(b["wall_s_median"], 1, pct=False))
        put(f"{pre}/s2s", _k(b["s2_s"], 2, pct=False))
        put(f"{pre}/s1s", _k(b["s1_par_s"], 2, pct=False))
        put(f"{pre}/toolcov", _k(b.get("tool_coverage")))
        put(f"{pre}/argcov", _k(b.get("arg_coverage")))
        for dp in DPS:
            t = "tool" if dp == "tool_select" else "arg"
            put(f"{pre}/{t}err", _tc(b[dp]["accepted_errors"]))
            put(f"{pre}/{t}acc_n", _tc(b[dp]["accepted"]))
            put(f"{pre}/{t}n", _tc(b[dp]["n"]))
            put(f"{pre}/{t}zero", _k(_share(b[dp]["zero_conf"], b[dp]["asked"])))
            put(f"{pre}/{t}ref", _tc(b[dp]["refusals"]))
    for name, c in (lv.get("contrasts") or {}).items():
        arm, _, ref = name.partition("_vs_")
        tag = {"oaidec_final": "oaidec", "agree_final": "agree"}[arm]
        rt = {"llm_only": "llm", "harness_final": "hf4b", "e18_final": "e18"}[ref]
        pre = f"live/{tag}/vs{rt}"
        put(f"{pre}/n", str(c["n"]))
        put(f"{pre}/success", _k(c["success"]["delta"], sign=True))
        put(f"{pre}/success/lo", _k(c["success"]["lo"], sign=True))
        put(f"{pre}/success/hi", _k(c["success"]["hi"], sign=True))
        put(f"{pre}/saving", _k(c["saving"]["delta"]))
        put(f"{pre}/saving/lo", _k(c["saving"]["lo"]))
        put(f"{pre}/saving/hi", _k(c["saving"]["hi"]))
        put(f"{pre}/wall", _k(c["wall_s"]["delta"], 1, pct=False, sign=True))
        put(f"{pre}/timesaving", _k(c.get("time_saving")))
    for row, rd in (lv.get("readings") or {}).items():
        put(f"live/{row}/reading", "holds" if rd["holds"] else "does not hold")
    sp = r.get("spend") or {}
    for p, v in (sp.get("total") or {}).items():
        put(f"spend/{p}", _k(v, 2, pct=False))
    return k


def write_tex(keys: Mapping[str, str], path: Path = TEX) -> Path:
    head = [r"% AUTO-GENERATED by scripts/explore_e20_ensembles.py summarize -- do not edit by hand.",
            f"% generated {_now()} at git {_git()}; PART A: {LABEL_A}; PART B: {LABEL_B}",
            r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(keys.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + body + [r"\makeatother", ""]))
    return path


def markdown(r: Mapping[str, Any]) -> str:
    off = r.get("offline") or {}
    L = ["# E20: decision-model ensembles and a hosted decision API behind the gate", "",
         f"Generated {r['generated']} at git `{r['git']}` by `python scripts/explore_e20_ensembles.py summarize`.", "",
         f"**Part A** ({LABEL_A}): {off.get('split', '')}; C split by task, L1 by item (`split.json`); members and m "
         "chosen on the development half (`choice.json`), every number below on the held-out half. No LLM is a voter.", ""]
    for lvn, Lv in (off.get("levels") or {}).items():
        L += [f"## Part A, {'common states C' if lvn == 'c' else 'L1 fixed decisions (E1)'}: held-out half "
              f"({Lv['test_units']:,} {'occurrences, ' + str(Lv.get('test_tasks')) + ' tasks' if lvn == 'c' else 'decisions'})", "",
              "| family | rule | members | acc % | must-ask % | predicted live % | gate coverage % | accepted errors | accepted acc % | own-confidence errors at equal coverage |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for fam, F in Lv["families"].items():
            for name, b in F.items():
                if name == "gate":
                    own = "; ".join(f"{m} {e['accepted_errors']:.0f}" for m, e in b["own_conf_at_equal_coverage"].items())
                    L.append(f"| {fam} | gate {b['m']} of {len(b['members'])} | {', '.join(b['members'])} | | | {_pc(b.get('pred_live'))} | "
                             f"{_pc(b['coverage'])} | {b['accepted_errors']} | {_pc(b['accepted_acc'])} | {own} |")
                else:
                    mem = b.get("model") or ", ".join(b["members"])
                    L.append(f"| {fam} | {name} | {mem} | {_pc(b.get('acc'))} | {_pc(b.get('mustask_acc'))} | {_pc(b.get('pred_live'))} | | | | |")
        for m, b in Lv["refs"].items():
            g = b["gate"]
            L.append(f"| ours | {m} at its threshold {g.get('tau')} | {MODELS[m][0]} | {_pc(b.get('acc'))} | {_pc(b.get('mustask_acc'))} | "
                     f"{_pc(b.get('pred_live'))} | {_pc(g['coverage'])} | {g['accepted_errors']} | {_pc(g['accepted_acc'])} | |")
        pr = Lv.get("pair_rl_jev")
        if pr:
            L.append(f"| pair | AgentDecision-4B + Jev (weighted; agreement gate) | rl, jev | {_pc(pr['weighted'].get('acc'))} | "
                     f"{_pc(pr['weighted'].get('mustask_acc'))} | {_pc(pr['weighted'].get('pred_live'))} | {_pc(pr['gate']['coverage'])} | "
                     f"{pr['gate']['accepted_errors']} | {_pc(pr['gate']['accepted_acc'])} | |")
        if Lv.get("hosted_sweep"):
            L += ["", "Hosted pair as the live backend gates it (both agree and the smaller confidence >= t) against each "
                  "model's own confidence at the same coverage (accepted errors):", "",
                  "| t | coverage % | accepted errors (agree) | Decisions API alone | Jev alone |", "|---|---|---|---|---|"]
            for row in Lv["hosted_sweep"]:
                L.append(f"| {row['t']} | {_pc(row['coverage'])} | {row['accepted_errors']} | {row['singles'].get('oaidec', 0):.1f} | "
                         f"{row['singles'].get('jev', 0):.1f} |")
        L.append("")
        ec = Lv.get("errcorr")
        if ec:
            L += ["### Error correlation (all of C, no selection)", "",
                  "| models | median Jaccard of error sets | if independent | ratio | same wrong answer among joint errors % | "
                  "must-ask states failed by all % | by half % | by none % |", "|---|---|---|---|---|---|---|---|"]
            for name in ("all_ots", "top6_ots"):
                x = ec[name]
                mu = x["mustask"]
                L.append(f"| {name} ({len(x['models'])}) | {_pc(x['median_jaccard'])} | {_pc(x['median_jaccard_indep'])} | "
                         f"{x['median_ratio']:.1f} | {_pc(x['median_same_wrong'])} | {_pc(mu['share_failed_by_all'])} | "
                         f"{_pc(mu['share_failed_by_half'])} | {_pc(mu['share_failed_by_none'])} |")
            m27 = ec.get("mustask_27b")
            if m27:
                L += ["", f"Must-ask states ({m27['n']} occurrences): OpenJev-27B fails {m27['err_a']}, StartLux-27B "
                      f"{m27['err_b']}, both {m27['both']} (the same wrong tool on {m27['same_wrong']}, in {m27['tasks_both']} tasks).", ""]
    h = off.get("hard") or {}
    if h.get("common"):
        L += ["## Part A, DecisionGym-Hard (E7c shadow calibration arms, the states both reached)", "",
              f"{h['common']} common labelled tool-choice states (Jev's arm {h['states']['jev']}, the 4B's {h['states']['rl']}).", "",
              "| states | n (tasks) | Jev % | AgentDecision-4B % | vote (larger confidence) % | agreement gate coverage % | accepted errors | accepted acc % |",
              "|---|---|---|---|---|---|---|---|"]
        for part in ("tier", "other", "all"):
            b = h[part]
            L.append(f"| {part} | {b['n']} ({b['tasks']}) | {_pc(b['jev'])} | {_pc(b['rl'])} | {_pc(b['vote'])} | "
                     f"{_pc(b['gate_coverage'])} | {b['gate_accepted_errors']} | {_pc(b['gate_accepted_acc'])} |")
        L.append("")
    if off.get("transfer"):
        L += ["Transfer of the chosen vote members across levels (held-out half): " +
              "; ".join(f"{k} {_pc(v)}" for k, v in off["transfer"].items()), ""]
    cal = r.get("cal") or {}
    if cal.get("rows"):
        L += [f"## Part B ({LABEL_B})", "", "### Thresholds on the calibration states (rl_shadow_cal)", "",
              "| row | point | tau-hat | n | accuracy % | coverage at tau-hat % | accepted errors | agreement % |", "|---|---|---|---|---|---|---|---|"]
        for row, b in cal["rows"].items():
            for dp in DPS:
                c = b["coverage"][dp]
                L.append(f"| {row} | {dp} | {b['fits'][dp]['tau_hat']} | {c['n']} | {_pc(c['acc'])} | {_pc(c['coverage'])} | "
                         f"{c['accepted_errors']} | {_pc((b.get('agree_share') or {}).get(dp))} |")
        L.append("")
    facts = cal.get("facts") or {}
    if facts:
        L += ["Descriptive (no fit): on the same labelled states", ""]
        for dp, b in facts.items():
            parts = [f"{dp}: n {b['n']}, the live LLM right on {b['llm_right']}"]
            for m in MEMBER_SPECS:
                if m in b:
                    parts.append(f"{m} right on {b[m]['right']} (confidence 1.0: {b[m]['conf1']} answers, {b[m]['conf1_wrong']} wrong)")
            if "agree" in b:
                parts.append(f"members agree on {b['agree']['n']} ({b['agree']['right']} right; both confidences >= 0.99 on "
                             f"{b['agree']['n99']}, {b['agree']['wrong99']} wrong)")
            L.append("- " + "; ".join(parts))
        L.append("")
    if r.get("live_status"):
        L += ["Live rows: " + "; ".join(f"{k}: {v}" for k, v in r["live_status"].items()), ""]
    lv = r.get("live") or {}
    if lv.get("arms"):
        L += ["### Live (seeds 300500-300699)", "", f"{lv.get('reference_note')}; errored (left out): {json.dumps(lv['errored'])}", "",
              "| arm | n | success % | API $/1k ep (Gemini / OpenAI / Jev) | incl. GPU | wall s (median) | S1 s (parallel) | S2 s | tool cov % | arg cov % | accepted errors tool / arg |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm, b in lv["arms"].items():
            pv = b["prov_usd"]
            L.append(f"| {arm} | {b['n']} | {_pc(b['success'])} | {1000 * b['api_usd']:.2f} ({1000 * pv['gemini']:.2f} / "
                     f"{1000 * pv['openai']:.2f} / {1000 * pv['jev']:.2f}) | {1000 * b['usd_incl_gpu']:.2f} | {b['wall_s_mean']:.1f} "
                     f"({b['wall_s_median']:.1f}) | {b['s1_par_s']:.2f} | {b['s2_s']:.2f} | {_pc(b.get('tool_coverage'))} | "
                     f"{_pc(b.get('arg_coverage'))} | {b['tool_select']['accepted_errors']} / {b['tool_params']['accepted_errors']} |")
        L += ["", "| contrast | n | success delta pt [Newcombe] | saving % [boot] | wall delta s |", "|---|---|---|---|---|"]
        for name, c in lv.get("contrasts", {}).items():
            L.append(f"| {name} | {c['n']} | {100 * c['success']['delta']:+.1f} [{100 * c['success']['lo']:+.1f}, {100 * c['success']['hi']:+.1f}] | "
                     f"{_pc(c['saving']['delta'])} [{_pc(c['saving']['lo'])}, {_pc(c['saving']['hi'])}] | {c['wall_s']['delta']:+.1f} |")
        L += ["", "Registered reading: " + "; ".join(f"{k}: {'holds' if v['holds'] else 'does not hold'} "
                                                    f"({v['success_minus_llm_pt']:+.1f} pt, saving {_pc(v['saving'])} %)"
                                                    for k, v in lv.get("readings", {}).items()), ""]
    sp = r.get("spend") or {}
    if sp.get("total"):
        L += ["Spend (calibration + every live record, smokes included): " +
              ", ".join(f"{p} ${v:.2f} of ${CAPS[p]:.0f}" for p, v in sp["total"].items()), ""]
    return "\n".join(L) + "\n"


def live_status() -> dict[str, str]:
    """Per row: 'run' when its live records exist, else the profile's skip reason (the fitted thresholds)."""
    import yaml

    out: dict[str, str] = {}
    prof = yaml.safe_load(PROFILE.read_text()) if PROFILE.exists() else {}
    blocks = (((prof.get("system2") or {}).get("gemini-3.5-flash") or {}).get("backends") or {})
    for row in ROWS:
        if (RUNS / f"e20_live_{row}" / f"{ROW_ARM[row]}.jsonl").exists():
            out[row] = "run"
        else:
            out[row] = str((blocks.get(row) or {}).get("skip") or "not run")
    return out


def cmd_summarize(a: argparse.Namespace) -> int:
    r: dict[str, Any] = {"generated": _now(), "git": _git()}
    r["offline"] = json.loads((OUT / "offline.json").read_text()) if (OUT / "offline.json").exists() else {}
    r["cal"] = json.loads((CAL_DIR / "fits.json").read_text()) if (CAL_DIR / "fits.json").exists() else {}
    if r["cal"]:
        r["cal"]["facts"] = cal_facts()
    r["live_status"] = live_status()
    r["live"] = live()
    r["spend"] = total_spend()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (OUT / "summary.md").write_text(markdown(r))
    keys = tex_keys(r)
    write_tex(keys)
    print(f"{len(keys)} keys -> {TEX.relative_to(ROOT)}")
    return 0


# ================================================================== CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split", help="the fixed dev / test split (no accuracy read)").set_defaults(fn=cmd_split)
    sub.add_parser("choose", help="members and m on the dev halves").set_defaults(fn=cmd_choose)
    sub.add_parser("offline", help="PART A's held-out readout -> offline.json").set_defaults(fn=cmd_offline)
    sub.add_parser("cal-build", help="rebuild the calibration questions (local, no call)").set_defaults(fn=cmd_cal_build)
    ca = sub.add_parser("cal-answer", help="one member answers every calibration question (priced; resumes)")
    ca.add_argument("--member", required=True, choices=sorted(MEMBER_SPECS))
    ca.add_argument("--workers", type=int, default=4)
    ca.add_argument("--limit", type=int, default=0)
    ca.set_defaults(fn=cmd_cal_answer)
    sub.add_parser("cal-fit", help="derive the rows' arm copies and fit tau-hat (registered command)").set_defaults(fn=cmd_cal_fit)
    sub.add_parser("prepare-live", help="write configs/harness/explore_e20_final.yaml from the fits").set_defaults(fn=cmd_prepare_live)
    sub.add_parser("probe", help="one probe round of the direct Gemini API").set_defaults(fn=cmd_probe)
    sp = sub.add_parser("spend", help="spend per provider (calibration + every live record)")
    sp.add_argument("--short", action="store_true")
    sp.set_defaults(fn=cmd_spend)
    g = sub.add_parser("gate", help="the smoke's projection per provider (registration (d))")
    g.add_argument("--smoke", required=True)
    g.add_argument("--n", type=int, default=200)
    g.set_defaults(fn=cmd_gate)
    w = sub.add_parser("watch", help="caps and the overload pause for one run")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--run", required=True)
    w.add_argument("--every", type=float, default=20.0)
    w.set_defaults(fn=cmd_watch)
    e = sub.add_parser("errored", help="episodes whose latest record errored")
    e.add_argument("--run", required=True)
    e.set_defaults(fn=cmd_errored)
    sub.add_parser("summarize", help="summary.{md,json} + paper/generated/e20_ensembles.tex").set_defaults(fn=cmd_summarize)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
