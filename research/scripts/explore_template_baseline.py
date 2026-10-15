#!/usr/bin/env python3
r"""Is ~99 % on the common states (L2) leakage, or is the in-domain task learnable from the generator's templates?
-- OFFLINE baselines fitted on the training rows only and scored on the common-state corpus C exactly as the decision
models are; post hoc and descriptive, stored data only (no API call, no GPU, no model call); revises no registered
result; no registration entry (the convention of scripts/benchmark_difficulty.py and scripts/explore_drift.py).

    python scripts/explore_template_baseline.py
    # -> results/explore/template_baseline/{summary.md,summary.json} and paper/generated/template_baseline.tex
    #    (\defres{tpl/...})

The leakage audit (``scripts/leakage_audit.py``) found no instance-level overlap between AgentDecision-4B's training
rows and C. This script tests the other explanation directly: how far does a lookup that only knows the generator's
templates get?

Training rows (the only data the baselines see): the JevGym tool-choice rows of SFT data v1
(``data/train_v1/train.jsonl.gz``: question "Which tool should the agent call next?"; it includes the v0 rows), the
label being the argmax of the row's training target (the distribution SFT v1 was trained on; teacher-or-gold, A4).
Knockout-final variants (``#ko`` rows: the same state with fewer options and the same label) are left out so that
each training state counts once (``--with-augmented`` keeps them). The GRPO / OPD stages of the frozen model trained
on their own rollouts, which were never stored as rows; the baselines therefore see a subset of what the model saw.

Baselines (every key is read from the state text alone, the way a decision model sees it):

(a) ``prior``: the most frequent training label (tool choice is the only question type of C).
(b) ``lookup``: the most frequent training label given (task family, the sequence of tools already executed in the
    episode as visible in the state's HISTORY); the family comes from the TASK line by a fixed keyword rule
    (``family``: multi-request tasks keep the ordered request kinds, e.g. ``multi:ship,cancel,cancel``; the
    single-request templates are told apart by their wording, e.g. a ship request without a speed policy). Unseen
    key -> (a).
(c) ``lookup_last``: as (b) with only the last executed tool (``<start>`` before the first call).
(b+) ``lookup_backoff``: (b), backing off to (c) for an unseen key and then to (a) (an n-gram style back-off; the
    strongest of these trivially learned baselines).
(d) coverage of (b)'s and (c)'s keys over C's occurrences, and every row's accuracy on covered vs uncovered ones.
(e) ``key_ceiling`` (a bound, not a baseline): the best tool per (b)-key chosen on C itself (in-sample); how much of
    C a function of the template key alone could get right.

Every prediction is restricted to the tools the C question offers (the most frequent label among the offered tools).
Scoring is the benchmark's own: an answer is right iff it is in the occurrence's logged acceptable set
(``bench/stage2_corpus.answer_correct``), averaged per task and then over tasks (the published C accuracy;
``stage2_corpus.task_accuracy``); pooled occurrence means are given too. Subsets are those of
``scripts/benchmark_difficulty.py`` (``c_items``: after an off-plan step, the next tool is ask_user, must-ask = the
acceptable set is exactly {ask_user}, state-change trap, finish next, long history, the pre-declared hard union, the
Flash states). Reference rows: the stored C answers of AgentDecision-4B (frozen JevAgentRL-4B), SFT v1, the 0.8B
scoring head (E14 arm A), Jev, StartLux-27B and the untrained Qwen3-4B.

The renamed-tool subset is not defined on C (C keeps the generator's own tool names); it belongs to E21's shifted
corpus (``results/explore/e21_shift/c``: core tools renamed, requests reworded). When that corpus is present, the
same baselines (still fitted on the training rows) are also scored there on E21's subsets (all, must-ask, after an
off-plan step, renamed = the acceptable set holds a tool other than ``finish``; the rules of
``explore_e21_shift.occ_subsets``), for the baselines only: the decision models' rows there are E21's readout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "results/explore/template_baseline"
OUT_TEX = ROOT / "paper/generated/template_baseline.tex"
TRAIN = ROOT / "data/train_v1/train.jsonl.gz"
S23 = ROOT / "results/stage2/s23"
SHIFT = ROOT / "results/explore/e21_shift/c"
TOOL_Q = "Which tool should the agent call next?"
START = "<start>"
LABEL = ("post hoc, descriptive; stored data only (no API call, no GPU, no model call); baselines fitted on the "
         "training rows only; revises no registered result; no registration entry (the convention of "
         "scripts/benchmark_difficulty.py / scripts/explore_drift.py)")
N_BOOT, SEED = 2000, 0
TEX_SUBSETS = ("all", "must_ask", "after_off_plan", "long_history", "renamed")   # keyed in TeX (the rest: md / json)

BASELINES: dict[str, str] = {"prior": "(a) most frequent training label",
                             "lookup": "(b) template lookup: family + executed-tool sequence",
                             "lookup_last": "(c) template lookup: family + last executed tool",
                             "lookup_backoff": "(b+) lookup (b), backing off to (c), then (a)",
                             "key_ceiling": "(e) best tool per (b)-key, chosen on the scored set itself (bound, in-sample)"}
# reference rows: slug -> (label, stored C answers)
REFS: dict[str, tuple[str, Path]] = {
    "rl": ("AgentDecision-4B", S23 / "answers_rl.jsonl"),
    "sft_v1": ("SFT v1 (the SFT stage of AgentDecision-4B)", S23 / "answers_sft_v1.jsonl"),
    "e14a": ("0.8B scoring head (E14 A)", ROOT / "results/explore/e14_small/records/s23/answers_e14a.jsonl.gz"),
    "jev": ("Jev", S23 / "answers_jev.jsonl"),
    "sl27b": ("StartLux-27B", S23 / "answers_sl27b.jsonl"),
    "base": ("Qwen3-4B (untrained)", S23 / "answers_base.jsonl"),
}
C_SUBSETS: dict[str, str] = {
    "all": "every occurrence", "must_ask": "the acceptable set is exactly {ask_user}",
    "after_off_plan": "after an earlier executed tool outside its acceptable set",
    "ask_gold": "the canonical next tool is ask_user", "hard": "after an off-plan step, or ask_user next",
    "state_change_trap": "the next tool changes the world and a state-changing look-alike is offered",
    "finish_gold": "the next tool is finish", "long_history": "step >= 8 (history window full)",
    "llm_states": "occurrences of the LLM-only (Flash) trajectories"}
SHIFT_ASK = "request_clarification"
SHIFT_SUBSETS: dict[str, str] = {"all": "every occurrence", "must_ask": "the acceptable set is exactly "
                                 "{request_clarification}", "after_off_plan": "after an off-plan step",
                                 "renamed": "the acceptable set holds a tool other than finish (a renamed tool)"}


# ================================================================== helpers
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(path).endswith(".gz") else open
    out = []
    with op(path, "rt") as f:
        for x in f:
            if x.strip():
                try:
                    out.append(json.loads(x))
                except json.JSONDecodeError:
                    continue
    return out


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _mean(v: Sequence[float]) -> float | None:
    return sum(v) / len(v) if v else None


def option_name(o: str) -> str:
    """Training options are rendered "name -- description"; C options are bare names."""
    return str(o).split(" -- ", 1)[0].strip()


# ================================================================== the template key (read from the state text)
_TASK = re.compile(r"TASK:\s*(.*?)(?:\n\s*\n|\nHISTORY:|\nCURRENT OBSERVATION:|$)", re.S)
_HIST = re.compile(r"^\[(\d+)\] ([A-Za-z_][A-Za-z0-9_]*)\(", re.M)
_PART = re.compile(r"\(\d+\)\s+([a-z]+)\b")


def request_text(state: str) -> str:
    m = _TASK.search(str(state))
    return (m.group(1) if m else "").strip()


def family(state: str, coarse: bool = False) -> str:
    """The task family from the request text by a fixed keyword rule (the generator's request templates, the
    customer names and ids ignored). Multi-request tasks keep the ordered request kinds unless ``coarse``."""
    t = request_text(state).lower()
    if "process these requests in order" in t:
        return "multi" if coarse else "multi:" + ",".join(_PART.findall(t))
    if "did not say who they are" in t:
        return "ship:no_name"
    if "how many units" in t:
        return "stock"
    if "total amount of all" in t:
        return "total"
    if "refund" in t:
        return "refund" if "store credit" in t else "refund:no_method"
    if "cancel" in t:
        return "cancel"
    if "ship" in t:
        return "ship" if "express" in t else "ship:no_speed"
    return "other"


def tool_history(state: str) -> tuple[str, ...]:
    """The executed tools in order, as visible in the state's HISTORY lines ("[k] tool(args) -> observation"); a
    history window or a truncated state shows only what it shows."""
    s = str(state)
    i = s.find("HISTORY:")
    if i < 0:
        return ()
    j = s.find("CURRENT OBSERVATION:", i)
    return tuple(m.group(2) for m in _HIST.finditer(s[i: j if j > 0 else len(s)]))


def key_full(state: str, coarse: bool = False) -> tuple[str, tuple[str, ...]]:
    return family(state, coarse), tool_history(state)


def key_last(state: str) -> tuple[str, str]:
    h = tool_history(state)
    return family(state), (h[-1] if h else START)


# ================================================================== training rows and the baselines
def training_rows(path: Path = TRAIN, with_augmented: bool = False) -> list[dict[str, Any]]:
    """JevGym tool-choice rows: state, offered tool names, the label (argmax of the training target, by name)."""
    out = []
    for r in read_jsonl(path):
        if r.get("dp") != "tool_select" or r.get("question") != TOOL_Q or not str(r.get("id", "")).startswith("jevgym"):
            continue
        if not with_augmented and ("#ko" in str(r.get("id")) or (r.get("meta") or {}).get("augment")):
            continue
        tgt = r.get("target")
        if isinstance(tgt, Mapping) and tgt:
            lab = option_name(max(tgt, key=lambda k: (tgt[k], k)))
        elif isinstance(tgt, str):
            lab = option_name(tgt)
        else:
            continue
        out.append({"id": r["id"], "task": str(r["id"]).split("/")[0], "state": r.get("state") or "",
                    "options": [option_name(o) for o in r.get("options") or []], "label": lab})
    return out


def pick(counts: Mapping[str, int] | None, offered: Iterable[str]) -> str | None:
    """The most frequent label among the offered tools (ties: alphabetical); None when none is offered."""
    off = set(offered)
    best = sorted(((-c, lab) for lab, c in (counts or {}).items() if lab in off))
    return best[0][1] if best else None


class Lookup:
    """Most frequent training label per key; an unseen key (or a key none of whose labels is offered) falls back
    to the prior."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], keyfn: Callable[[str], Any] | None):
        self.keyfn = keyfn
        self.prior: Counter = Counter(r["label"] for r in rows)
        self.table: dict[Any, Counter] = defaultdict(Counter)
        if keyfn is not None:
            for r in rows:
                self.table[keyfn(r["state"])][r["label"]] += 1

    def covered(self, state: str) -> bool:
        return self.keyfn is not None and self.keyfn(state) in self.table

    def predict(self, state: str, offered: Sequence[str]) -> tuple[str | None, str]:
        if self.keyfn is not None:
            k = self.keyfn(state)
            if k in self.table:
                a = pick(self.table[k], offered)
                if a is not None:
                    return a, "key"
                return pick(self.prior, offered), "key_not_offered"
            return pick(self.prior, offered), "unseen"
        return pick(self.prior, offered), "prior"


def key_ceiling(items: Sequence[Mapping[str, Any]], keyfn: Callable[[str], Any]) -> dict[str, str | None]:
    """In-sample bound: per key the offered tool acceptable at the most occurrences of that key on the scored set
    itself; returns qhash -> that tool (the same answer for every question with the key)."""
    from jevharness.bench import metrics as M

    votes: dict[Any, Counter] = defaultdict(Counter)
    for it in items:
        for t in set(M.acceptable_of(it["rec"]) or [it["rec"].get("gold")]):
            if t is not None:
                votes[keyfn(it["state"])][t] += 1
    best = {k: sorted((-c, t) for t, c in v.items())[0][1] for k, v in votes.items()}
    return {it["qhash"]: best.get(keyfn(it["state"])) for it in items}


# ================================================================== the scored items
def c_items(s23: Path = S23) -> list[dict[str, Any]]:
    """C's occurrences with the difficulty analysis's strata (benchmark_difficulty.c_items) and their state and
    offered tools."""
    from benchmark_difficulty import c_items as diff_items

    items, qs = diff_items()
    for it in items:
        q = qs[it["qhash"]]
        it["state"] = q["state"]
        it["options"] = [option_name(o) for o in (json.loads(q["options"]) if isinstance(q["options"], str)
                                                  else q["options"])]
    return items


def shift_items(cdir: Path = SHIFT) -> list[dict[str, Any]]:
    """E21's shifted corpus with E21's subset rules (explore_e21_shift.occ_subsets: must-ask, after an off-plan step,
    renamed); empty when the corpus is absent."""
    if not (cdir / "occurrences.jsonl.gz").exists() or not (cdir / "questions.jsonl.gz").exists():
        return []
    qs = {q["qhash"]: q for q in read_jsonl(cdir / "questions.jsonl.gz")}
    out = []
    for o in read_jsonl(cdir / "occurrences.jsonl.gz"):
        q = qs.get(o["qhash"])
        if q is None:
            continue
        acc = list(((o.get("gold_aux") or {}).get("acceptable_tools")) or [])
        out.append({"task": o["task"], "qhash": o["qhash"], "state": q["state"],
                    "options": [option_name(x) for x in q["options"]],
                    "rec": {"dp": "tool_select", "gold": o.get("gold"), "gold_aux": o.get("gold_aux"),
                            "gold_status": o.get("gold_status")},
                    "strata": {"all": True, "must_ask": acc == [SHIFT_ASK], "after_off_plan": o.get("stratum") == "after_off_plan",
                               "renamed": bool(acc) and "finish" not in acc}})
    return out


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    """qhash -> the stored answer record (the latest record of a question decides)."""
    return {r["qhash"]: r for r in read_jsonl(path)} if path.exists() else {}


def correct(it: Mapping[str, Any], ans: Mapping[str, Any] | None) -> float:
    from jevharness.bench.stage2_corpus import answer_correct

    return float(answer_correct(it["rec"], ans))


def accuracy(items: Sequence[Mapping[str, Any]], ok: Mapping[int, float]) -> dict[str, Any]:
    """Task mean (the published C accuracy) and pooled occurrence mean over ``items`` (``ok``: item index -> 0/1)."""
    per: dict[str, list[float]] = defaultdict(list)
    for it in items:
        per[it["task"]].append(ok[it["i"]])
    vals = [v for vs in per.values() for v in vs]
    return {"acc": _mean([sum(v) / len(v) for v in per.values()]), "pooled": _mean(vals), "n": len(vals),
            "tasks": len(per)}


def paired_task_ci(items: Sequence[Mapping[str, Any]], ok_a: Mapping[int, float], ok_b: Mapping[int, float],
                   n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    """Task mean of a minus task mean of b, with a 95 % percentile task bootstrap (descriptive)."""
    import numpy as np

    per: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for it in items:
        per[it["task"]].append((ok_a[it["i"]], ok_b[it["i"]]))
    tasks = sorted(per)
    if not tasks:
        return {"diff": None, "lo": None, "hi": None}
    d = np.array([np.mean([a for a, _ in per[t]]) - np.mean([b for _, b in per[t]]) for t in tasks])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "tasks": len(tasks)}


# ================================================================== build
def score_set(items: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]], subsets: Mapping[str, str],
              refs: Mapping[str, Mapping[str, Mapping[str, Any]]], with_ceiling: bool = True) -> dict[str, Any]:
    """Baselines + reference rows on every subset, coverage of the lookup keys, and where the lookup and the trained
    model disagree."""
    for i, it in enumerate(items):
        it["i"] = i
    prior, full, last = Lookup(rows, None), Lookup(rows, key_full), Lookup(rows, key_last)
    preds: dict[str, dict[int, str | None]] = {"prior": {}, "lookup": {}, "lookup_last": {}, "lookup_backoff": {}}
    route: dict[str, Counter] = {"lookup": Counter(), "lookup_last": Counter(), "lookup_backoff": Counter()}
    for it in items:
        preds["prior"][it["i"]] = prior.predict(it["state"], it["options"])[0]
        got = {}
        for name, lk in (("lookup", full), ("lookup_last", last)):
            got[name] = a, how = lk.predict(it["state"], it["options"])
            preds[name][it["i"]] = a
            route[name][how] += 1
        a, how = got["lookup"] if got["lookup"][1] == "key" else (
            (got["lookup_last"][0], "backoff_last") if got["lookup_last"][1] == "key" else (preds["prior"][it["i"]], "prior"))
        preds["lookup_backoff"][it["i"]] = a
        route["lookup_backoff"][how] += 1
    if with_ceiling:
        ceil = key_ceiling(items, key_full)
        preds["key_ceiling"] = {it["i"]: ceil[it["qhash"]] for it in items}
    ok: dict[str, dict[int, float]] = {
        m: {i: correct(items[i], {"answer": a, "valid": True} if a is not None else None) for i, a in p.items()}
        for m, p in preds.items()}
    for m, ans in refs.items():
        ok[m] = {it["i"]: correct(it, ans.get(it["qhash"])) for it in items}
    res: dict[str, Any] = {"n": len(items), "tasks": len({it["task"] for it in items}),
                           "questions": len({it["qhash"] for it in items}), "subsets": {}, "routes": {k: dict(v) for k, v in route.items()}}
    for s in subsets:
        sub = [it for it in items if it["strata"].get(s)]
        if not sub:
            continue
        res["subsets"][s] = {"n": len(sub), "tasks": len({it["task"] for it in sub}),
                             "rows": {m: accuracy(sub, o) for m, o in ok.items()}}
    # (d) coverage of the keys and accuracy on covered / uncovered occurrences
    res["coverage"] = {}
    for name, lk in (("lookup", full), ("lookup_last", last)):
        cov = [it for it in items if lk.covered(it["state"])]
        unc = [it for it in items if not lk.covered(it["state"])]
        blk: dict[str, Any] = {"keys_train": len(lk.table), "covered_n": len(cov), "uncovered_n": len(unc),
                               "share": len(cov) / len(items) if items else None,
                               "question_share": (len({it["qhash"] for it in cov}) / len({it["qhash"] for it in items})
                                                  if items else None),
                               "keys_scored": len({lk.keyfn(it["state"]) for it in items}),
                               "keys_scored_seen": len({lk.keyfn(it["state"]) for it in cov}),
                               "covered": {m: accuracy(cov, o) for m, o in ok.items()} if cov else {},
                               "uncovered": {m: accuracy(unc, o) for m, o in ok.items()} if unc else {}}
        for s in ("must_ask", "after_off_plan", "renamed", "long_history"):
            if not any(s in it["strata"] for it in items):    # a stratum this corpus does not define
                continue
            sub = [it for it in items if it["strata"].get(s)]
            if sub:
                blk[f"share_{s}"] = sum(lk.covered(it["state"]) for it in sub) / len(sub)
            if unc:                                       # what the uncovered occurrences are
                blk[f"uncovered_share_{s}"] = sum(bool(it["strata"].get(s)) for it in unc) / len(unc)
        res["coverage"][name] = blk
    # where the lookup is wrong and the trained model right (and the converse), by the answer key and stratum
    if "rl" in ok:
        dis: dict[str, Any] = {}
        for name in ("lookup", "lookup_backoff", "lookup_last"):
            lw_rr = [it for it in items if not ok[name][it["i"]] and ok["rl"][it["i"]]]
            lr_rw = [it for it in items if ok[name][it["i"]] and not ok["rl"][it["i"]]]
            dis[name] = {"lookup_wrong_rl_right": len(lw_rr), "lookup_right_rl_wrong": len(lr_rw),
                         "lookup_wrong_rl_right_by_gold": dict(Counter(it["rec"]["gold"] for it in lw_rr).most_common()),
                         "lookup_wrong_rl_right_by_family": dict(Counter(family(it["state"], coarse=True) for it in lw_rr).most_common()),
                         "lookup_wrong_rl_right_covered": sum((last if name == "lookup_last" else full).covered(it["state"])
                                                              for it in lw_rr),
                         "lookup_wrong_rl_right_strata": {s: sum(bool(it["strata"].get(s)) for it in lw_rr) for s in subsets},
                         "lookup_wrong_answer_was": dict(Counter(str(preds[name][it["i"]]) + " (gold " + str(it["rec"]["gold"]) + ")"
                                                                 for it in lw_rr).most_common(12))}
        res["disagreement"] = dis
        res["gap_rl_minus_lookup"] = paired_task_ci(items, ok["rl"], ok["lookup"])
        res["gap_rl_minus_lookup_last"] = paired_task_ci(items, ok["rl"], ok["lookup_last"])
        res["gap_rl_minus_lookup_backoff"] = paired_task_ci(items, ok["rl"], ok["lookup_backoff"])
    # the coarse-family variant of (b) (sensitivity; JSON / md only)
    coarse = Lookup(rows, lambda s: key_full(s, coarse=True))
    okc = {it["i"]: correct(it, {"answer": coarse.predict(it["state"], it["options"])[0], "valid": True}) for it in items}
    res["lookup_coarse"] = {"acc": accuracy(items, okc), "share": _mean([float(coarse.covered(it["state"])) for it in items]),
                            "keys_train": len(coarse.table)}
    return res


def build(train: Path = TRAIN, with_augmented: bool = False, items: list[dict[str, Any]] | None = None,
          refs: Mapping[str, tuple[str, Path]] = REFS, shift: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = training_rows(train, with_augmented)
    items = c_items() if items is None else items
    ref_ans = {m: load_answers(p) for m, (_, p) in refs.items() if p.exists()}
    fam = Counter(family(r["state"]) for r in rows)
    res: dict[str, Any] = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git("rev-parse", "--short=12", "HEAD"), "label": LABEL,
        "train": {"path": str(train.relative_to(ROOT)) if train.is_relative_to(ROOT) else str(train),
                  "rows": len(rows), "tasks": len({r["task"] for r in rows}), "with_augmented": with_augmented,
                  "labels": dict(Counter(r["label"] for r in rows).most_common()), "families": dict(fam.most_common()),
                  "keys_lookup": len({key_full(r["state"]) for r in rows}),
                  "keys_lookup_last": len({key_last(r["state"]) for r in rows})},
        "c_families": dict(Counter(family(it["state"]) for it in items).most_common()),
        "refs": {m: refs[m][0] for m in ref_ans}, "baselines": BASELINES, "subsets": C_SUBSETS,
        "c": score_set(items, rows, C_SUBSETS, ref_ans),
    }
    shift = shift_items() if shift is None else shift
    if shift:
        res["shift"] = score_set(shift, rows, SHIFT_SUBSETS, {}, with_ceiling=True)
        res["shift_subsets"] = SHIFT_SUBSETS
    return res


# ================================================================== outputs
def _p(x: Any, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.{d}f}"


def _tc(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def tex_rows(res: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {}

    def put(key: str, val: Any, fmt: Callable[[Any], str] = _p) -> None:
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            k[key] = fmt(val)

    tr = res["train"]
    put("tpl/train/rows", tr["rows"], _tc)
    put("tpl/train/tasks", tr["tasks"], str)
    put("tpl/train/keys", tr["keys_lookup"], _tc)
    put("tpl/train/keys_last", tr["keys_lookup_last"], str)
    for blk_name, prefix in (("c", "tpl/c"), ("shift", "tpl/shift")):
        blk = res.get(blk_name)
        if not blk:
            continue
        put(f"{prefix}/questions", blk["questions"], _tc)
        for s, sb in blk["subsets"].items():
            if s not in TEX_SUBSETS:                      # the other subsets: summary.md / summary.json only
                continue
            put(f"{prefix}/{s}/n", sb["n"], _tc)
            put(f"{prefix}/{s}/tasks", sb["tasks"], str)
            for m, r in sb["rows"].items():
                if blk_name == "shift" and m not in BASELINES:
                    continue
                put(f"{prefix}/{s}/{m}/acc", r["acc"])
                if s == "all":
                    put(f"{prefix}/{s}/{m}/pooled", r["pooled"])
        for name, cv in blk["coverage"].items():
            slug = "full" if name == "lookup" else "last"
            put(f"{prefix}/cov/{slug}/share", cv["share"])
            put(f"{prefix}/cov/{slug}/qshare", cv["question_share"])
            put(f"{prefix}/cov/{slug}/covered/n", cv["covered_n"], _tc)
            put(f"{prefix}/cov/{slug}/uncovered/n", cv["uncovered_n"], _tc)
            put(f"{prefix}/cov/{slug}/keys", cv["keys_scored"], _tc)
            put(f"{prefix}/cov/{slug}/keys_seen", cv["keys_scored_seen"], _tc)
            for s in ("must_ask", "after_off_plan", "renamed", "long_history"):
                put(f"{prefix}/cov/{slug}/share/{s}", cv.get(f"share_{s}"))
                put(f"{prefix}/cov/{slug}/uncovered/share/{s}", cv.get(f"uncovered_share_{s}"))
            if slug != "full":
                continue
            for part in ("covered", "uncovered"):
                for m, r in cv[part].items():
                    if blk_name == "shift" and m not in BASELINES:
                        continue
                    put(f"{prefix}/cov/{slug}/{part}/{m}/acc", r["acc"])
        for g, slug in (("gap_rl_minus_lookup", "full"), ("gap_rl_minus_lookup_last", "last"),
                        ("gap_rl_minus_lookup_backoff", "backoff")):
            v = blk.get(g)
            if v and v.get("diff") is not None:
                put(f"{prefix}/gap/{slug}", v["diff"])
                put(f"{prefix}/gap/{slug}/lo", v["lo"])
                put(f"{prefix}/gap/{slug}/hi", v["hi"])
        for name, slug in (("lookup", "full"), ("lookup_backoff", "backoff"), ("lookup_last", "last")):
            dis = (blk.get("disagreement") or {}).get(name)
            if dis:
                put(f"{prefix}/dis/{slug}/lw_rr", dis["lookup_wrong_rl_right"], _tc)
                put(f"{prefix}/dis/{slug}/lr_rw", dis["lookup_right_rl_wrong"], _tc)
                put(f"{prefix}/dis/{slug}/lw_rr_covered", dis["lookup_wrong_rl_right_covered"], _tc)
                put(f"{prefix}/dis/{slug}/lw_rr_offplan", dis["lookup_wrong_rl_right_strata"].get("after_off_plan"), _tc)
        lc = blk.get("lookup_coarse")
        if lc:
            put(f"{prefix}/all/lookup_coarse/acc", lc["acc"]["acc"])
    return k


def to_tex(res: Mapping[str, Any]) -> str:
    head = [r"% AUTO-GENERATED by scripts/explore_template_baseline.py -- do not edit by hand.",
            f"% generated {res['generated']} at git {res['git']}; {res['label']}",
            r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{{key}}}{{{val}}}" for key, val in sorted(tex_rows(res).items())]
    return "\n".join(head + body + [r"\makeatother", ""])


def _name(m: str, res: Mapping[str, Any]) -> str:
    return {**{k: v for k, v in BASELINES.items()}, **res.get("refs", {}), "lookup_coarse": "(b') lookup, coarse family"}.get(m, m)


def _table(blk: Mapping[str, Any], cols: Sequence[str], rows: Sequence[str], res: Mapping[str, Any]) -> list[str]:
    cols = [c for c in cols if c in blk["subsets"]]
    L = ["| row | " + " | ".join(f"{c} (n {blk['subsets'][c]['n']:,}; {blk['subsets'][c]['tasks']} tasks)" for c in cols) + " |",
         "|---|" + "---|" * len(cols)]
    for m in rows:
        cells = [_p(blk["subsets"][c]["rows"][m]["acc"]) if m in blk["subsets"][c]["rows"] else "–" for c in cols]
        if any(x != "–" for x in cells):
            L.append(f"| {_name(m, res)} | " + " | ".join(cells) + " |")
    return L


def to_md(res: Mapping[str, Any]) -> str:
    c, tr = res["c"], res["train"]
    order = list(BASELINES) + list(res["refs"])
    L = ["# Template baselines on the common states: leakage, or a learnable generator?", "",
         f"Generated {res['generated']} at git `{res['git']}` by `scripts/explore_template_baseline.py`. {res['label']}. "
         "Paper keys: `paper/generated/template_baseline.tex` (`\\res{tpl/...}`); App. `app:tplbase`.", "",
         "## Training rows (all the baselines see)", "",
         f"`{tr['path']}`: {tr['rows']:,} JevGym tool-choice rows of {tr['tasks']} training tasks "
         f"({'with' if tr['with_augmented'] else 'without'} the knockout-final duplicates); label = argmax of the "
         f"training target. Distinct keys: (b) {tr['keys_lookup']:,}, (c) {tr['keys_lookup_last']}.", "",
         "Labels: " + ", ".join(f"{a} {b}" for a, b in tr["labels"].items()) + ".", "",
         "Families (keyword rule) in training: " + ", ".join(f"`{a}` {b}" for a, b in tr["families"].items()) + ".", "",
         "Families on C (occurrences): " + ", ".join(f"`{a}` {b}" for a, b in res["c_families"].items()) + ".", "",
         "## C: accuracy (task mean %, the published C metric)", ""]
    L += _table(c, list(C_SUBSETS), order, res)
    L += ["", "Subset definitions: " + "; ".join(f"`{a}`: {b}" for a, b in C_SUBSETS.items()) + ".", "",
          "Pooled occurrence accuracy on all of C: " + ", ".join(
              f"{_name(m, res)} {_p(c['subsets']['all']['rows'][m]['pooled'])}" for m in order if m in c["subsets"]["all"]["rows"]) + ".",
          "", f"(b') the lookup with the coarse family (`multi` without the request kinds): "
          f"{_p(c['lookup_coarse']['acc']['acc'])} % (key coverage {_p(c['lookup_coarse']['share'])} %).", "",
          "How the lookups answered (occurrences): " + "; ".join(f"{k}: " + ", ".join(f"{a} {b}" for a, b in v.items())
                                                              for k, v in c["routes"].items()) + ".", "",
          "## (d) Key coverage on C and accuracy on covered / uncovered occurrences (task mean %)", "",
          "| lookup | keys in training | keys on C (seen) | covered occurrences | covered questions | covered must-ask | covered after off-plan | "
          + " | ".join(f"{_name(m, res)}: covered / uncovered" for m in order if m != "key_ceiling") + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len([m for m in order if m != "key_ceiling"])]
    for name, cv in c["coverage"].items():
        cells = []
        for m in order:
            if m == "key_ceiling":
                continue
            a = (cv["covered"].get(m) or {}).get("acc")
            b = (cv["uncovered"].get(m) or {}).get("acc")
            cells.append(f"{_p(a)} / {_p(b)}")
        L.append(f"| {name} | {cv['keys_train']:,} | {cv['keys_scored']:,} ({cv['keys_scored_seen']:,}) | "
                 f"{cv['covered_n']:,} of {cv['covered_n'] + cv['uncovered_n']:,} ({_p(cv['share'])} %) | {_p(cv['question_share'])} % | "
                 f"{_p(cv.get('share_must_ask'))} % | {_p(cv.get('share_after_off_plan'))} % | " + " | ".join(cells) + " |")
    cv = c["coverage"]["lookup"]
    L += ["", f"The {cv['uncovered_n']:,} occurrences with an unseen (b)-key: {_p(cv.get('uncovered_share_after_off_plan'))} % "
          f"after an off-plan step, {_p(cv.get('uncovered_share_long_history'))} % with a long history (step >= 8), "
          f"{_p(cv.get('uncovered_share_must_ask'))} % must-ask."]
    L += ["", "## Where the lookup fails and AgentDecision-4B does not", ""]
    for g, v in (("lookup", c.get("gap_rl_minus_lookup")), ("lookup_backoff", c.get("gap_rl_minus_lookup_backoff")),
                 ("lookup_last", c.get("gap_rl_minus_lookup_last"))):
        if v and v.get("diff") is not None:
            L.append(f"- AgentDecision-4B minus {g}: {100 * v['diff']:+.1f} points (95 % task bootstrap "
                     f"[{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}], {v['tasks']} tasks; descriptive).")
    for name, d in (c.get("disagreement") or {}).items():
        L += [f"- {name}: wrong where AgentDecision-4B is right at {d['lookup_wrong_rl_right']:,} occurrences "
              f"({d['lookup_wrong_rl_right_covered']:,} of them with a covered key); right where it is wrong at "
              f"{d['lookup_right_rl_wrong']:,}. By answer key: " + ", ".join(f"{a} {b}" for a, b in d["lookup_wrong_rl_right_by_gold"].items())
              + ". By family: " + ", ".join(f"{a} {b}" for a, b in d["lookup_wrong_rl_right_by_family"].items())
              + ". In strata: " + ", ".join(f"{a} {b}" for a, b in d["lookup_wrong_rl_right_strata"].items() if b)
              + ". Most frequent wrong answers: " + "; ".join(f"{a} x{b}" for a, b in d["lookup_wrong_answer_was"].items()) + "."]
    sh = res.get("shift")
    if sh:
        L += ["", "## E21's shifted corpus (renamed core tools, reworded requests): the baselines only", "",
              "The same baselines (fitted on the same training rows) on `results/explore/e21_shift/c` with E21's subset "
              "rules; the decision models' rows on this corpus are E21's readout (App. app:e21).", ""]
        L += _table(sh, list(SHIFT_SUBSETS), list(BASELINES), res)
        cv = sh["coverage"]["lookup"]
        L += ["", f"(b) key coverage on the shifted corpus: {_p(cv['share'])} % of {sh['n']:,} occurrences "
              f"(renamed subset {_p(cv.get('share_renamed'))} %)."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--out-tex", default=str(OUT_TEX))
    ap.add_argument("--with-augmented", action="store_true", help="keep the knockout-final duplicate rows")
    ap.add_argument("--no-shift", action="store_true", help="skip E21's shifted corpus")
    a = ap.parse_args(argv)
    res = build(with_augmented=a.with_augmented, shift=[] if a.no_shift else None)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(to_md(res))
    Path(a.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_tex).write_text(to_tex(res))
    allc = res["c"]["subsets"]["all"]["rows"]
    print(json.dumps({m: round(r["acc"], 4) for m, r in allc.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
