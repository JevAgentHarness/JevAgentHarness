#!/usr/bin/env python3
r"""Is JevAgentBench's tool selection too easy? -- OFFLINE difficulty analysis from stored answers only (no API call,
no GPU), post hoc and descriptive. Every number is computed from run logs and stored replay answers.

    python scripts/benchmark_difficulty.py
    # writes results/benchmark_difficulty/{difficulty.json,difficulty.md} and paper/generated/difficulty.tex
    # (\defres{diff/...}); results/benchmark_difficulty/{proposal.md,replay_plan.md} are hand-written drafts that cite
    # these keys.

Sources (each already holds every model answer it is scored on):

* **Ceiling.** LLM-only success of the agent LLMs on JevGym (Stage 1 Flash and Flash-Lite, Stage 2 Flash; latest record
  per task) and the agent LLM's own per-decision tool accuracy; the first wrong decision of every failed Flash-Lite
  episode, by type.
* **C**, the S2-3 common-state corpus (``results/stage2/s23``): 5,314 tool-selection occurrences of 200 fresh tasks
  answered by every decision model (replay answers). Hard subsets are fixed by rule from the state and its gold label
  (never from a model's answer): after an off-plan step; the canonical next tool is ``ask_user`` (clarify); the next
  tool changes the world and has a state-changing look-alike among the offered tools; the next tool is ``finish``;
  long history (step >= 8, the harness's history window is full); and their pre-declared union "hard" = after an
  off-plan step or an ``ask_user`` decision. A subset is reported only with >= ``MIN_N`` occurrences (smaller strata are
  merged into a named parent; listed in ``merged``). The agent LLM's own accuracy is given on the occurrences of its own
  (LLM-only) trajectories.
* **Stage-1 shadow logs** (evaluation seeds 200000-200499; ``runs/e2a_jevgym_tool_decisions*``): Jev and
  Intern-Decision-4B asked in shadow at every state of Flash and Flash-Lite episodes, the LLM's own choice logged at the
  same state. States are matched across the two runs by the exact question (state, question text, options); tool
  selection and arguments (argument subsets by candidate count and by gold-outside-the-candidates).
* **tau^2 reference corpus** (``results/stage2/t2ref``): JevAgentRL-4B, Intern-Decision-4B, the untrained base
  (replay) and Jev (as logged); reference-agreement labels (not oracle accuracy).
* **L1 static** (E1, C8 population; ``runs/e1_bfcl_when2call*``): every decision model and Flash's native function
  calling, by BFCL / When2Call category.
* **Stage-1 meta-decision shadow** (``runs/e2b_jevgym_meta_control/meta_shadow*``): Jev only (item counts for the
  replay plan, Jev against the majority label).
* **Development stress slice** (v2 track, seeds 2000-2199, 120 distractors / 40 noise lines / five customers;
  ``results/gpu_round6_miles/stress``): development data, labelled as such.

Intervals (appendix only): 95 % percentile bootstrap, B = ``N_BOOT``, seed ``SEED``, resampling tasks for JevGym and
tau^2 (the unit of analysis) and items for L1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "results/benchmark_difficulty"
OUT_TEX = ROOT / "paper/generated/difficulty.tex"
S23, T2REF = ROOT / "results/stage2/s23", ROOT / "results/stage2/t2ref"
RUNS = ROOT / "runs"
MIN_N, N_BOOT, SEED = 300, 2000, 0
STATE_CHANGING = ("cancel_order", "refund", "ship_order")

# the decision models of the paper (Figure fig:levels) + attribution / exploratory rows
MODELS: dict[str, str] = {"jev": "Jev", "intern": "Intern-Decision-4B", "sl0.8b": "StartLux-0.8B",
                          "sl2b": "StartLux-2B", "sl4b": "StartLux-4B", "sl9b": "StartLux-9B",
                          "sl27b": "StartLux-27B", "rl": "JevAgentRL-4B"}
EXTRA_MODELS: dict[str, str] = {"base": "Qwen3-4B (untrained)", "sft_v1": "SFT only (v1)",
                                "v2": "JevAgentRL-v2 (exploratory)"}
E1_FILES: dict[str, str] = {
    "jev": "e1_bfcl_when2call/s1_jev.jsonl", "intern": "e1_bfcl_when2call_startlux/s1_intern_4b.jsonl",
    "sl0.8b": "e1_bfcl_when2call_startlux_ladder/s1_startlux_0_8b.jsonl",
    "sl2b": "e1_bfcl_when2call_startlux_ladder/s1_startlux_2b.jsonl",
    "sl4b": "e1_bfcl_when2call_startlux/s1_startlux_4b.jsonl", "sl9b": "e1_bfcl_when2call_startlux_ladder/s1_startlux_9b.jsonl",
    "sl27b": "e1_bfcl_when2call_startlux/s1_startlux_27b.jsonl", "rl": "e1_bfcl_when2call_rl/s1_jevagent_rl.jsonl",
    "base": "e1_bfcl_when2call_rl/s1_rl_base.jsonl", "sft_v1": "e1_bfcl_when2call_rl/s1_rl_sft_v1.jsonl",
    "llm": "e1_bfcl_when2call/fc_gemini_flash_fc.jsonl"}


# ================================================================== helpers
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def latest(path: Path) -> dict[str, dict[str, Any]]:
    from jevharness.harness.trajectory import iter_records, latest_by_task

    if not path.exists():
        return {}
    return latest_by_task(r for r in iter_records(path) if "task_id" in r)


def near_map() -> dict[str, list[tuple[str, bool]]]:
    """core tool -> its near-duplicate distractors (name, state-changing)."""
    from jevharness.bench.suites.jevgym import NEAR_BY_NAME

    out: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for n, t in NEAR_BY_NAME.items():
        out[t.shadows].append((n, bool(t.write)))
    return dict(out)


def opts_of(x: Any) -> list[str]:
    return json.loads(x) if isinstance(x, str) else list(x or [])


def qkey(state: str, question: str, options: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps([state, question, list(options)]).encode()).hexdigest()


def cluster_mean(values: Mapping[str, Sequence[float]]) -> float:
    """Mean over clusters (tasks) of the cluster mean; NaN when empty."""
    v = [sum(x) / len(x) for x in values.values() if x]
    return sum(v) / len(v) if v else float("nan")


def bootstrap_ci(per_cluster: Sequence[Mapping[str, Sequence[float]]],
                 fn: Callable[[np.ndarray], np.ndarray] | None = None, n_boot: int = N_BOOT, seed: int = SEED
                 ) -> tuple[float, float]:
    """95 % percentile interval over clusters resampled with replacement (every row with the same clusters, paired).
    ``fn`` maps the (rows, n_boot) array of resampled cluster-weighted means to n_boot statistics (default: row 0)."""
    keys = sorted(set().union(*[set(k for k, v in m.items() if v) for m in per_cluster]))
    if not keys:
        return float("nan"), float("nan")
    M = np.array([[np.mean(m[k]) if m.get(k) else np.nan for k in keys] for m in per_cluster])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = np.nanmean(M[:, idx], axis=2)
    stats = fn(means) if fn else means[0]
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def subsets_by_rule(items: Sequence[Mapping[str, Any]], rules: Mapping[str, Callable[[Mapping[str, Any]], bool]],
                    min_n: int = MIN_N) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, str]]:
    """Apply every rule; keep the subsets with at least ``min_n`` items and list the others as dropped."""
    kept: dict[str, list[Mapping[str, Any]]] = {}
    dropped: dict[str, str] = {}
    for k, rule in rules.items():
        sub = [it for it in items if rule(it)]
        if len(sub) < min_n:
            dropped[k] = f"n = {len(sub)} < {min_n}: dropped"
        else:
            kept[k] = sub
    return kept, dropped


def score_block(items: Sequence[Mapping[str, Any]], answer_ok: Mapping[str, Callable[[Mapping[str, Any]], float | None]],
                cluster: str = "task", ci: bool = True) -> dict[str, Any]:
    """Accuracy per model over ``items`` (cluster-weighted), with n, clusters and a bootstrap interval."""
    out: dict[str, Any] = {"n": len(items), "clusters": len({it[cluster] for it in items}), "models": {}}
    for m, fn in answer_ok.items():
        per: dict[str, list[float]] = defaultdict(list)
        for it in items:
            v = fn(it)
            if v is not None:
                per[it[cluster]].append(float(v))
        if not per:
            continue
        est = cluster_mean(per)
        row = {"acc": est, "n": sum(len(v) for v in per.values())}
        if ci:
            row["ci"] = list(bootstrap_ci([per]))
        out["models"][m] = row
    off = {k: r["acc"] for k, r in out["models"].items() if k in MODELS and k != "rl"}   # off-the-shelf rows
    if len(off) >= 2:
        lo, hi = min(off, key=off.get), max(off, key=off.get)
        out["offshelf"] = {"min": off[lo], "min_model": lo, "max": off[hi], "max_model": hi, "n_models": len(off)}
    accs = [r["acc"] for k, r in out["models"].items() if k in MODELS]
    if len(accs) >= 2:
        out["spread_8"] = max(accs) - min(accs)
        out["min_8"], out["max_8"], out["n_models"] = min(accs), max(accs), len(accs)
    return out


# ================================================================== ceiling
def first_error(traj: Mapping[str, Any], near: Mapping[str, list[tuple[str, bool]]]) -> str:
    """Type of the first executed tool outside the step's acceptable set; 'no wrong tool' when every executed tool was
    acceptable (the failure then lies in an argument or the final answer)."""
    from jevharness.bench import metrics as M

    near_names = {n for v in near.values() for n, _ in v}
    for st in traj.get("steps") or []:
        rec = next((d for d in st.get("decisions") or [] if d.get("dp") == "tool_select" and M.has_gold(d)), None)
        ex = st.get("tool")
        if rec is None or ex is None:
            continue
        if M.is_correct(rec, ex):
            continue
        gold = rec.get("gold")
        at = set(M.acceptable_of(rec) or [gold])
        if ex in near_names:
            return "near-duplicate distractor"
        if ex in STATE_CHANGING and (gold == "ask_user" or "ask_user" in at):
            return "changed state before asking"
        if ex == "finish":
            return "finished too early"
        if ex == "ask_user":
            return "unneeded question"
        if ex in ("find_user", "get_orders", "cancel_order", "refund", "check_stock", "ship_order", "send_message"):
            return "wrong core tool (skip / order)"
        return "unrelated distractor"
    r = traj.get("result") or {}
    if r.get("n_calls") == r.get("optimal_calls") and not r.get("side_effects"):
        return "every call on plan, final answer wrong"
    return "no wrong tool (argument or step limit)"


def ceiling() -> dict[str, Any]:
    from jevharness.bench import metrics as M

    near = near_map()
    arms = {"stage1_flash": RUNS / "e2a_jevgym_tool_decisions/llm_only.jsonl",
            "stage1_lite": RUNS / "e2a_jevgym_tool_decisions/llm_only_s2lite.jsonl",
            "stage2_flash": RUNS / "stage2_jevgym/llm_only.jsonl"}
    out: dict[str, Any] = {}
    for k, p in arms.items():
        lat = latest(p)
        if not lat:
            continue
        ok = [t for t in lat.values() if not t.get("error")]
        succ = sum(bool((t.get("result") or {}).get("success")) for t in ok)
        dec, hit = 0, 0
        for t in ok:
            for st in t.get("steps") or []:
                for d in st.get("decisions") or []:
                    if d.get("dp") == "tool_select" and M.has_gold(d) and d.get("s2_value") is not None:
                        dec += 1
                        hit += bool(M.is_correct(d, d["s2_value"]))
        fails = [t for t in ok if not (t.get("result") or {}).get("success")]
        out[k] = {"tasks": len(ok), "success": succ / len(ok), "tool_decisions": dec, "tool_acc": hit / dec if dec else None,
                  "failures": len(fails), "first_error": dict(Counter(first_error(t, near) for t in fails).most_common()),
                  "failed_families": dict(Counter(str(t["task_id"]).split("-")[1] for t in fails).most_common())}
    return out


# ================================================================== C (common-state corpus)
def c_items() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Occurrences of C with their rule-based strata and the agent LLM's own choice where the occurrence comes from the
    LLM-only trajectory."""
    from jevharness.bench import metrics as M

    occ = read_jsonl(S23 / "occurrences.jsonl.gz")
    qs = {q["qhash"]: q for q in read_jsonl(S23 / "questions.jsonl.gz")}
    near = near_map()
    llm = latest(RUNS / "stage2_jevgym/llm_only.jsonl")
    llm_choice: dict[tuple[str, int], str] = {}
    for t, tr in llm.items():
        for st in tr.get("steps") or []:
            if st.get("tool") is not None:
                llm_choice[(t, int(st.get("step", 0)))] = st["tool"]
    items = []
    for i, o in enumerate(occ):
        opts = opts_of(qs[o["qhash"]]["options"])
        gold = o.get("gold")
        sc_trap = gold in STATE_CHANGING and any(n in opts and w for n, w in near.get(gold, []))
        rec = {"dp": "tool_select", "gold": gold, "gold_aux": o.get("gold_aux"), "gold_status": o.get("gold_status")}
        it = {"i": i, "task": o["task"], "qhash": o["qhash"], "rec": rec, "source": o["source"], "step": o["step"],
              "family": o["task"].split("-")[1], "gold": gold,
              "strata": {"all": True, "after_off_plan": o["stratum"] == "after_off_plan", "ask_gold": gold == "ask_user",
                         "must_ask": set(M.acceptable_of(rec) or []) == {"ask_user"},
                         "state_change_trap": sc_trap, "finish_gold": gold == "finish", "long_history": o["step"] >= 8,
                         "llm_states": o["source"] == "llm_only"}}
        it["strata"]["hard"] = it["strata"]["after_off_plan"] or it["strata"]["ask_gold"]
        if o["source"] == "llm_only":
            it["llm"] = llm_choice.get((o["task"], int(o["step"])))
        items.append(it)
    return items, qs


C_SUBSETS = {"all": "every occurrence", "hard": "after an off-plan step, or the next tool is ask_user (pre-declared union)",
             "after_off_plan": "after an earlier executed tool outside its acceptable set",
             "ask_gold": "the canonical next tool is ask_user (clarify; the must-ask stratum with ask_user as the only "
                         "acceptable tool, n < 300, is merged here)",
             "state_change_trap": "the next tool changes the world and a state-changing look-alike is offered",
             "finish_gold": "the next tool is finish (stop)", "long_history": "step >= 8 (history window full)",
             "llm_states": "occurrences of the LLM-only (Flash) trajectories"}


def error_type(ans: str | None, it: Mapping[str, Any], near: Mapping[str, list[tuple[str, bool]]]) -> str:
    from jevharness.bench import metrics as M

    if ans is None:
        return "no valid answer"
    at = set(M.acceptable_of(it["rec"]) or [it["gold"]])
    if ans in {n for g in at | {it["gold"]} for n, _ in near.get(g, [])}:
        return "look-alike of the gold tool"
    if any(ans in {n for n, _ in v} for v in near.values()):
        return "other look-alike"
    if ans in STATE_CHANGING and "ask_user" in at:
        return "changed state before asking"
    if ans == "finish":
        return "finished too early"
    if ans == "ask_user":
        return "unneeded question"
    if ans in ("find_user", "get_orders", "cancel_order", "refund", "check_stock", "ship_order", "send_message"):
        return "wrong core tool"
    return "unrelated distractor"


def analyse_c() -> dict[str, Any]:
    from jevharness.bench.stage2_corpus import answer_correct

    items, _ = c_items()
    answers = {m: {r["qhash"]: r for r in read_jsonl(S23 / f"answers_{m}.jsonl")}
               for m in list(MODELS) + list(EXTRA_MODELS) if (S23 / f"answers_{m}.jsonl").exists()}
    fns: dict[str, Callable[[Mapping[str, Any]], float | None]] = {
        m: (lambda it, a=a: float(answer_correct(it["rec"] | {"gold": it["gold"]}, a.get(it["qhash"]))))
        for m, a in answers.items()}
    from jevharness.bench import metrics as M

    fns_llm = {"llm": lambda it: (None if it.get("llm") is None else float(bool(M.is_correct(it["rec"], it["llm"]))))}
    res: dict[str, Any] = {"subsets": {}, "definitions": C_SUBSETS, "min_n": MIN_N}
    counts = {k: sum(it["strata"][k] for it in items) for k in list(C_SUBSETS) + ["must_ask"]}
    res["counts"] = counts
    res["merged"] = {"must_ask": f"n = {counts['must_ask']} < {MIN_N}: merged into ask_gold"}
    kept, dropped = subsets_by_rule(items, {k: (lambda it, k=k: bool(it["strata"][k])) for k in C_SUBSETS})
    res["merged"].update(dropped)
    for k, sub in kept.items():
        res["subsets"][k] = score_block(sub, {**fns, **fns_llm})
    llm_items = [it for it in items if it["strata"]["llm_states"] and it.get("llm") is not None]
    res["llm_own_states"] = {"n": len(llm_items),
                             "wrong": sum(not M.is_correct(it["rec"], it["llm"]) for it in llm_items)}
    # error types
    near = near_map()
    err: dict[str, dict[str, int]] = {}
    for m, a in answers.items():
        c = Counter()
        for it in items:
            r = a.get(it["qhash"]) or {}
            val = r.get("answer") if not r.get("error") and r.get("valid", True) else None
            if not answer_correct(it["rec"] | {"gold": it["gold"]}, r):
                c[error_type(val, it, near)] += 1
        err[m] = dict(c.most_common())
    res["error_types"] = err
    # how much wider the 8-model spread is on the hard subsets than on all states
    res["spread"] = {k: v.get("spread_8") for k, v in res["subsets"].items()}
    # home-state check: each model's accuracy by source arm (the corpus pools four policies' trajectories)
    by_src: dict[str, dict[str, float]] = {}
    for src in ("llm_only", "jev_ts_replace_f", "intern_ts_replace_f", "rl_ts_replace_f"):
        sub = [it for it in items if it["source"] == src]
        blk = score_block(sub, fns, ci=False)
        by_src[src] = {m: r["acc"] for m, r in blk["models"].items()}
        by_src[src]["n"] = len(sub)
    res["by_source"] = by_src
    return res


# ================================================================== Stage-1 shadow logs (Flash / Flash-Lite states)
def shadow_items(jev_path: Path, intern_path: Path) -> list[dict[str, Any]]:
    """Every gold-labelled tool_select / tool_params shadow decision of the Jev run, matched to the Intern run's
    decision at the identical question; the LLM's own value as logged in the Jev run."""
    from jevharness.bench import metrics as M

    def collect(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        lat = latest(path)
        idx: dict[str, dict[str, Any]] = {}
        order: list[dict[str, Any]] = []
        for t, tr in lat.items():
            if tr.get("error"):
                continue
            off = False
            for st in tr.get("steps") or []:
                ts = None
                for d in st.get("decisions") or []:
                    if d.get("dp") not in ("tool_select", "tool_params") or d.get("mode") != "shadow" or not M.has_gold(d):
                        continue
                    opts = opts_of(d.get("options"))
                    k = qkey(d.get("state") or "", d.get("question") or "", opts)
                    rec = {"task": t, "dp": d["dp"], "step": int(st.get("step", 0)), "key": k, "n_opts": len(opts),
                           "gold": d.get("gold"), "rec": {kk: d.get(kk) for kk in ("dp", "gold", "gold_aux", "gold_status")},
                           "s1": d.get("s1_value"), "llm": d.get("s2_value"), "after_off_plan": off,
                           "param_kind": (d.get("gold_aux") or {}).get("param_kind"),
                           "gold_other": str(d.get("gold") or "").startswith("<OTHER")}
                    idx.setdefault(k + d["dp"], rec)
                    order.append(rec)
                    if d["dp"] == "tool_select":
                        ts = d
                if ts is not None and st.get("tool") is not None:
                    ok = M.is_correct(ts, st["tool"])
                    off = off or (ok is False)
        return idx, order

    _, jev = collect(jev_path)
    intern_idx, _ = collect(intern_path)
    out = []
    for r in jev:
        o = intern_idx.get(r["key"] + r["dp"])
        out.append({**r, "intern": None if o is None else o["s1"], "matched": o is not None})
    return out


S1_SUBSETS_TS = {"all": "every tool decision", "after_off_plan": "after an off-plan step of the LLM",
                 "ask_gold": "the next tool is ask_user", "state_change": "the next tool changes the world",
                 "long_history": "step >= 8"}
S1_SUBSETS_TP = {"all": "every argument decision", "copy": "copy argument (value copied from the history)",
                 "many_candidates": "copy argument with >= 20 candidate values",
                 "gold_outside": "the gold value is not among the candidates (OTHER)"}


def analyse_shadow() -> dict[str, Any]:
    from jevharness.bench import metrics as M

    res: dict[str, Any] = {}
    pairs = {"flash_lite": (RUNS / "e2a_jevgym_tool_decisions/jev_shadow_s2lite.jsonl",
                            RUNS / "e2a_jevgym_tool_decisions_intern/intern_shadow_s2lite.jsonl"),
             "flash": (RUNS / "e2a_jevgym_tool_decisions/jev_shadow.jsonl",
                       RUNS / "e2a_jevgym_tool_decisions_intern/intern_shadow.jsonl")}
    for name, (pj, pi) in pairs.items():
        if not pj.exists():
            continue
        items = shadow_items(pj, pi)

        def ok(field: str) -> Callable[[Mapping[str, Any]], float | None]:
            return lambda it: None if it.get(field) is None else float(bool(M.is_correct(it["rec"], it[field])))

        fns = {"llm": ok("llm"), "jev": ok("s1"), "intern": ok("intern")}
        blk: dict[str, Any] = {"items": len(items), "matched": sum(it["matched"] for it in items), "subsets": {},
                               "merged": {}}
        ts = [it for it in items if it["dp"] == "tool_select" and it["matched"]]
        tp = [it for it in items if it["dp"] == "tool_params" and it["matched"]]
        rules_ts = {"all": lambda it: True, "after_off_plan": lambda it: it["after_off_plan"],
                    "ask_gold": lambda it: it["gold"] == "ask_user",
                    "state_change": lambda it: it["gold"] in STATE_CHANGING, "long_history": lambda it: it["step"] >= 8}
        # argument items whose gold value is among the candidates (when it is not, the label is OTHER and the LLM's
        # own typed value cannot be scored against it); the OTHER items are counted separately
        rules_tp = {"all": lambda it: not it["gold_other"],
                    "copy": lambda it: it["param_kind"] == "copy" and not it["gold_other"],
                    "many_candidates": lambda it: it["param_kind"] == "copy" and it["n_opts"] >= 20 and not it["gold_other"],
                    "gold_outside": lambda it: it["gold_other"]}
        for dp, pool, rules in (("tool_select", ts, rules_ts), ("tool_params", tp, rules_tp)):
            kept, dropped = subsets_by_rule(pool, {f"{dp}/{k}": r for k, r in rules.items()})
            blk["merged"].update(dropped)
            for key, sub in kept.items():
                blk["subsets"][key] = score_block(sub, fns)
        blk["distinct_questions"] = {"tool_select": len({it["key"] for it in items if it["dp"] == "tool_select"}),
                                     "tool_params": len({it["key"] for it in items if it["dp"] == "tool_params"})}
        res[name] = blk
    return res


# ================================================================== tau^2 reference corpus
def analyse_tau2() -> dict[str, Any]:
    occ = [o for o in read_jsonl(T2REF / "occurrences.jsonl.gz") if o.get("gold") is not None]
    qs = {q["qhash"]: q for q in read_jsonl(T2REF / "questions.jsonl.gz")}
    names = {"rl": "frozen", "intern": "intern", "base": "base"}
    ans = {}
    for m, n in names.items():
        p = T2REF / f"answers_{n}.jsonl"
        if p.exists():
            ans[m] = {r["qhash"]: r for r in read_jsonl(p)}

    def agree(a: Mapping[str, Any] | None, gold: str) -> float:
        a = a or {}
        if a.get("error") or not a.get("valid", True) or a.get("answer") is None:
            return 0.0
        return float(a["answer"] == gold)

    fns: dict[str, Callable[[Mapping[str, Any]], float | None]] = {
        m: (lambda o, a=a: agree(a.get(o["qhash"]), o["gold"])) for m, a in ans.items()}
    fns["jev"] = lambda o: None if o.get("jev_logged") is None else float(o["jev_logged"] == o["gold"])
    for o in occ:
        o["n_opts"] = len(opts_of(qs[o["qhash"]]["options"])) if o["qhash"] in qs else 0
    rules = {"tool_select/all": lambda o: o["dp"] == "tool_select",
             "tool_select/non_majority": lambda o: o["dp"] == "tool_select" and o["gold"] != "respond_to_user",
             "tool_params/all": lambda o: o["dp"] == "tool_params",
             "tool_params/many_candidates": lambda o: o["dp"] == "tool_params" and o["n_opts"] >= 10}
    res: dict[str, Any] = {"subsets": {}, "merged": {}, "label": "agreement with one reference trajectory "
                           "(A6.22 (b)), not oracle accuracy; Jev = as logged at decision time (its occurrences only)"}
    kept, res["merged"] = subsets_by_rule(occ, rules)
    for k, sub in kept.items():
        res["subsets"][k] = score_block(sub, fns)
    res["distinct_questions"] = len(qs)
    return res


# ================================================================== L1 static (E1, C8 population)
def analyse_e1() -> dict[str, Any]:
    recs: dict[str, dict[str, dict[str, Any]]] = {}
    for m, f in E1_FILES.items():
        p = RUNS / f
        if p.exists():
            recs[m] = {r["id"]: r for r in read_jsonl(p)}
    if not recs:
        return {}
    ids = sorted(recs["jev"]) if "jev" in recs else sorted(next(iter(recs.values())))
    base = recs.get("jev") or next(iter(recs.values()))
    items = [{"id": i, "task": i, "source": base[i]["source"], "dp": base[i]["dp"], "gold": base[i]["gold"]} for i in ids]
    # an item without a scored answer (``correct`` None: skipped -- e.g. a When2Call item that native function calling
    # cannot express -- or errored) is left out of that row and the row's n says so
    fns = {m: (lambda it, r=r: None if it["id"] not in r or r[it["id"]].get("correct") is None
               else float(bool(r[it["id"]]["correct"])))
           for m, r in recs.items()}
    rules = {"bfcl_tool_select": lambda it: it["source"].startswith("bfcl") and it["dp"] == "tool_select",
             "bfcl_irrelevance": lambda it: it["source"] in ("bfcl:irrelevance", "bfcl:live_irrelevance"),
             "bfcl_multi_tool": lambda it: it["source"] in ("bfcl:multiple", "bfcl:live_multiple") and it["dp"] == "tool_select",
             "bfcl_arguments": lambda it: it["dp"] == "tool_params",
             "bfcl_live_arguments": lambda it: it["dp"] == "tool_params" and it["source"].startswith("bfcl:live"),
             "when2call": lambda it: it["source"].startswith("when2call"),
             "when2call_noncall": lambda it: it["source"].startswith("when2call") and "calloneoftheavailabletools" not in it["gold"]}
    res: dict[str, Any] = {"subsets": {}, "merged": {}, "unit": "item (bootstrap over items)"}
    kept, res["merged"] = subsets_by_rule(items, rules)
    for k, sub in kept.items():
        res["subsets"][k] = score_block(sub, fns)
    return res


# ================================================================== Stage-1 meta shadow (Jev only)
def analyse_meta() -> dict[str, Any]:
    from jevharness.bench import metrics as M

    res: dict[str, Any] = {}
    for name, f in (("flash", "e2b_jevgym_meta_control/meta_shadow.jsonl"),
                    ("flash_lite", "e2b_jevgym_meta_control/meta_shadow_s2lite.jsonl")):
        lat = latest(RUNS / f)
        if not lat:
            continue
        per: dict[str, list[tuple[Any, Any, str]]] = defaultdict(list)
        keys: dict[str, set[str]] = defaultdict(set)
        for t, tr in lat.items():
            for st in tr.get("steps") or []:
                for d in st.get("decisions") or []:
                    if d.get("mode") == "shadow" and d.get("dp") in ("stop", "reflection", "memory_write", "routing") \
                            and M.has_gold(d) and d.get("s1_value") is not None:
                        per[d["dp"]].append((d["s1_value"], d["gold"], t))
                        keys[d["dp"]].add(qkey(d.get("state") or "", d.get("question") or "", opts_of(d.get("options"))))
        blk = {}
        for dp, rows in per.items():
            golds = Counter(str(g) for _, g, _ in rows)
            maj = golds.most_common(1)[0]
            blk[dp] = {"n": len(rows), "distinct_questions": len(keys[dp]),
                       "jev_acc": sum(str(v) == str(g) for v, g, _ in rows) / len(rows),
                       "majority_label": maj[0], "majority_acc": maj[1] / len(rows),
                       "minority_n": len(rows) - maj[1],
                       "jev_acc_minority": (sum(str(v) == str(g) for v, g, _ in rows if str(g) != maj[0])
                                            / max(1, len(rows) - maj[1]))}
        res[name] = blk
    return res


# ================================================================== development stress slice
def analyse_stress() -> dict[str, Any]:
    d = ROOT / "results/gpu_round6_miles/stress"
    out: dict[str, Any] = {"label": "development data (seeds 2000-2199; v2 training track); not an evaluation"}
    for p in sorted(d.glob("stress_*_summary.json")):
        for k, v in json.loads(p.read_text()).items():
            arm = k.split("::")[0].replace("stress_", "")
            out[arm] = {"n": v.get("n"), "success": v.get("success"), "tool_acc": v.get("tool_select/replace/s1_acc"),
                        "arg_acc": v.get("tool_params/replace/s1_acc"), "reward": v.get("reward_ccps")}
    return out


# ================================================================== replay inventory (for replay_plan.md)
def replay_inventory() -> dict[str, Any]:
    """Distinct questions, characters per question and the models already answering them, for every set that a
    replay would add; the per-question time and Jev cost measured on C (L40S hosts, OpenRouter) for scaling."""
    from jevharness.bench import metrics as M

    inv: dict[str, Any] = {}
    # calibration on C: measured wall time per question per model, Jev $ per question, characters per question
    qs = read_jsonl(S23 / "questions.jsonl.gz")
    c_chars = [len(q["state"]) + sum(map(len, opts_of(q["options"]))) + sum(map(len, opts_of(q["descriptions"])))
               for q in qs]
    desc_chars = float(np.mean([sum(map(len, opts_of(q["descriptions"]))) for q in qs]))
    cal: dict[str, Any] = {"questions": len(qs), "chars_mean": float(np.mean(c_chars)), "tool_desc_chars": desc_chars,
                           "models": {}}
    for m in list(MODELS) + list(EXTRA_MODELS):
        p = S23 / f"answers_{m}.jsonl"
        if p.exists():
            a = read_jsonl(p)
            w = [r.get("wall_ms") or r.get("latency_ms") or 0.0 for r in a]
            cal["models"][m] = {"wall_ms_mean": float(np.mean(w)), "cost_usd": float(sum(r.get("cost_usd") or 0 for r in a)),
                                "requests_per_q": float(np.mean([r.get("requests") or 1 for r in a]))}
    inv["calibration_c"] = cal
    # recovery extension: tool-selection states after an off-plan step in the OTHER live replace arms on C's 200 fresh
    # tasks (the arms not among C's four sources); only the arm's own model answered them, live
    from jevharness.bench import stage2_corpus as SC
    from jevharness.bench.meta_gold import suite_factories

    c_q = {q["qhash"] for q in qs}
    facs = suite_factories(RUNS / "stage2_jevgym") if (RUNS / "stage2_jevgym/experiment.yaml").exists() else {}
    arms = {"sl0.8b": "explore_startlux_live/sl0.8b_ts_replace_f.jsonl", "sl2b": "explore_startlux_live/sl2b_ts_replace_f.jsonl",
            "sl9b": "explore_startlux_live/sl9b_ts_replace_f.jsonl", "sl4b": "stage2_startlux_states/sl4b_ts_replace_f.jsonl",
            "sl27b": "stage2_startlux_states/sl27b_ts_replace_f.jsonl", "base": "stage2_rl_attribution/rl_base_ts_replace_f.jsonl",
            "sft_v1": "stage2_rl_attribution/rl_sft_v1_ts_replace_f.jsonl", "v2": "stage2_v2_states/v2_ts_replace_f.jsonl"}
    rec: dict[str, Any] = {"arms": {}}
    all_new, off_new, off_new_occ = set(), set(), 0
    for m, f in arms.items():
        lat = latest(RUNS / f)
        occ = []
        for t, tr in lat.items():
            if not tr.get("error"):
                occ += SC.occurrences(tr, m, SC.tool_descriptions(facs.get(t)))
        off = [o for o in occ if o["stratum"] == "after_off_plan"]
        rec["arms"][m] = {"tasks": len(lat), "occurrences": len(occ), "after_off_plan": len(off),
                          "after_off_plan_new_questions": len({o["qhash"] for o in off} - c_q)}
        all_new |= {o["qhash"] for o in occ} - c_q
        off_new |= {o["qhash"] for o in off} - c_q
        off_new_occ += sum(o["qhash"] not in c_q for o in off)
    rec.update({"new_questions_all": len(all_new), "new_questions_after_off_plan": len(off_new),
                "new_occurrences_after_off_plan": off_new_occ,
                "c_after_off_plan_occurrences": sum(1 for o in read_jsonl(S23 / "occurrences.jsonl.gz")
                                                    if o["stratum"] == "after_off_plan")})
    inv["recovery_extension"] = rec
    # Stage-1 shadow sets (the Jev run's states; Jev's own answers logged at decision time)
    for name, f in (("shadow_flash_lite", "e2a_jevgym_tool_decisions/jev_shadow_s2lite.jsonl"),
                    ("shadow_flash", "e2a_jevgym_tool_decisions/jev_shadow.jsonl")):
        lat = latest(RUNS / f)
        seen: dict[str, dict[str, Any]] = {}
        sub = Counter()
        for t, tr in lat.items():
            if tr.get("error"):
                continue
            off = False
            for st in tr.get("steps") or []:
                ts = None
                for d in st.get("decisions") or []:
                    if d.get("mode") != "shadow" or d.get("dp") not in ("tool_select", "tool_params") or not M.has_gold(d):
                        continue
                    opts = opts_of(d.get("options"))
                    k = qkey(d.get("state") or "", d.get("question") or "", opts) + d["dp"]
                    if k not in seen:
                        seen[k] = {"dp": d["dp"], "chars": len(d.get("state") or "") + sum(map(len, opts))
                                   + (desc_chars if d["dp"] == "tool_select" else 0)}
                    if d["dp"] == "tool_select":
                        ts = d
                        sub["tool_select/all"] += 1
                        sub["tool_select/after_off_plan"] += off
                        sub["tool_select/ask_gold"] += d.get("gold") == "ask_user"
                        sub["tool_select/state_change"] += d.get("gold") in STATE_CHANGING
                        sub["tool_select/long_history"] += int(st.get("step", 0)) >= 8
                    else:
                        other = str(d.get("gold") or "").startswith("<OTHER")
                        kind = (d.get("gold_aux") or {}).get("param_kind")
                        sub["tool_params/all"] += not other
                        sub["tool_params/many_candidates"] += (kind == "copy" and len(opts) >= 20 and not other)
                        sub["tool_params/gold_outside"] += other
                if ts is not None and st.get("tool") is not None:
                    off = off or (M.is_correct(ts, st["tool"]) is False)
        by_dp = Counter(v["dp"] for v in seen.values())
        inv[name] = {"distinct_questions": dict(by_dp), "chars_mean": float(np.mean([v["chars"] for v in seen.values()])),
                     "occurrences_by_subset": dict(sub), "answered": ["jev (logged, shadow)", "intern (matched states)",
                                                                     "agent LLM (logged)"]}
    # tau^2 reference corpus
    tq = read_jsonl(T2REF / "questions.jsonl.gz")
    t_occ = read_jsonl(T2REF / "occurrences.jsonl.gz")
    inv["tau2_ref"] = {"distinct_questions": len(tq),
                       "chars_mean": float(np.mean([len(q["state"]) + sum(map(len, opts_of(q["options"])))
                                                    + sum(map(len, opts_of(q.get("descriptions")))) for q in tq])),
                       "occurrences_with_jev_logged": sum(o.get("jev_logged") is not None for o in t_occ),
                       "occurrences": len(t_occ),
                       "answered": sorted(p.name.split(".")[0][len("answers_"):] for p in T2REF.glob("answers_*.jsonl"))}
    # Stage-1 meta shadow (Noul questions)
    meta_q: dict[str, int] = Counter()
    meta_chars: list[int] = []
    for f in ("e2b_jevgym_meta_control/meta_shadow.jsonl", "e2b_jevgym_meta_control/meta_shadow_s2lite.jsonl"):
        seen_m: set[str] = set()
        for t, tr in latest(RUNS / f).items():
            for st in tr.get("steps") or []:
                for d in st.get("decisions") or []:
                    if d.get("mode") == "shadow" and d.get("dp") in ("stop", "reflection", "memory_write", "routing") \
                            and M.has_gold(d):
                        k = qkey(d.get("state") or "", d.get("question") or "", ["yes", "no"]) + d["dp"]
                        if k not in seen_m:
                            seen_m.add(k)
                            meta_q[d["dp"]] += 1
                            meta_chars.append(len(d.get("state") or "") + len(d.get("question") or ""))
    inv["meta_shadow"] = {"distinct_questions": dict(meta_q), "chars_mean": float(np.mean(meta_chars)) if meta_chars else None,
                          "answered": ["jev (logged, shadow)"]}
    return inv


# ================================================================== build + outputs
def build() -> dict[str, Any]:
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git": git("rev-parse", "--short=12", "HEAD"),
            "label": "post hoc, descriptive; stored answers only (no API call, no GPU); revises no registered result",
            "min_n": MIN_N, "n_boot": N_BOOT, "seed": SEED,
            "ceiling": ceiling(), "corpus_c": analyse_c(), "shadow_stage1": analyse_shadow(), "tau2": analyse_tau2(),
            "e1": analyse_e1(), "meta_stage1": analyse_meta(), "stress_dev": analyse_stress(),
            "replay_inventory": replay_inventory()}


def _p(x: Any, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.{d}f}"


def _tc(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def tex_rows(res: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {}
    for arm, v in res["ceiling"].items():
        k[f"diff/ceiling/{arm}/success"] = _p(v["success"])
        k[f"diff/ceiling/{arm}/tool_acc"] = _p(v["tool_acc"])
        k[f"diff/ceiling/{arm}/failures"] = str(v["failures"])
        fe = v["first_error"]
        if fe and v["failures"]:
            k[f"diff/ceiling/{arm}/fail_final_answer"] = _p(fe.get("every call on plan, final answer wrong", 0) / v["failures"], 0)
            k[f"diff/ceiling/{arm}/fail_final_answer_n"] = str(fe.get("every call on plan, final answer wrong", 0))
            k[f"diff/ceiling/{arm}/fail_lookalike"] = _p(fe.get("near-duplicate distractor", 0) / v["failures"], 0)
            k[f"diff/ceiling/{arm}/fail_premature"] = _p(fe.get("changed state before asking", 0) / v["failures"], 0)
    c = res["corpus_c"]
    for sub, blk in c["subsets"].items():
        k[f"diff/c/{sub}/n"] = _tc(blk["n"])
        k[f"diff/c/{sub}/tasks"] = str(blk["clusters"])
        if "spread_8" in blk:
            k[f"diff/c/{sub}/spread"] = _p(blk["spread_8"])
        if "offshelf" in blk:
            o = blk["offshelf"]
            k[f"diff/c/{sub}/offshelf/min"], k[f"diff/c/{sub}/offshelf/max"] = _p(o["min"]), _p(o["max"])
            k[f"diff/c/{sub}/offshelf/min_model"] = _name(o["min_model"])
            k[f"diff/c/{sub}/offshelf/max_model"] = _name(o["max_model"])
        for m, r in blk["models"].items():
            k[f"diff/c/{sub}/{m}/acc"] = _p(r["acc"])
            if "ci" in r and sub in ("all", "hard", "after_off_plan", "ask_gold"):
                k[f"diff/c/{sub}/{m}/ci/lo"], k[f"diff/c/{sub}/{m}/ci/hi"] = _p(r["ci"][0]), _p(r["ci"][1])
    for name, blk in res["shadow_stage1"].items():
        for sub, b in blk["subsets"].items():
            s = sub.replace("tool_select", "ts").replace("tool_params", "tp")
            k[f"diff/s1/{name}/{s}/n"] = _tc(b["n"])
            for m, r in b["models"].items():
                k[f"diff/s1/{name}/{s}/{m}/acc"] = _p(r["acc"])
    for sub, b in res["tau2"]["subsets"].items():
        s = sub.replace("tool_select", "ts").replace("tool_params", "tp")
        k[f"diff/t2/{s}/n"] = _tc(b["n"])
        for m, r in b["models"].items():
            k[f"diff/t2/{s}/{m}/acc"] = _p(r["acc"])
    for sub, b in res["e1"].get("subsets", {}).items():
        k[f"diff/e1/{sub}/n"] = _tc(b["n"])
        if "spread_8" in b:
            k[f"diff/e1/{sub}/spread"] = _p(b["spread_8"])
        for m, r in b["models"].items():
            k[f"diff/e1/{sub}/{m}/acc"] = _p(r["acc"])
    for arm, v in res["stress_dev"].items():
        if isinstance(v, dict):
            k[f"diff/stress/{arm}/success"] = _p(v["success"])
            k[f"diff/stress/{arm}/tool_acc"] = _p(v["tool_acc"])
    return k


def to_tex(res: Mapping[str, Any]) -> str:
    head = [r"% AUTO-GENERATED by scripts/benchmark_difficulty.py -- do not edit by hand.",
            f"% generated {res['generated']} at git {res['git']}; post hoc, descriptive difficulty analysis from stored "
            "answers (no API call); not a registered analysis",
            r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{{key}}}{{{val}}}" for key, val in sorted(tex_rows(res).items())]
    return "\n".join(head + body + [r"\makeatother", ""])


def _name(m: str) -> str:
    return {**MODELS, **EXTRA_MODELS, "llm": "agent LLM (own choice)"}.get(m, m)


def _table(blocks: Mapping[str, Mapping[str, Any]], order: Sequence[str], ci: bool, defs: Mapping[str, str] | None = None
           ) -> list[str]:
    cols = list(blocks)
    L = ["| model | " + " | ".join(f"{c} (n {blocks[c]['n']:,}; {blocks[c]['clusters']} clusters)" for c in cols) + " |",
         "|---|" + "---|" * len(cols)]
    for m in order:
        cells = []
        for c in cols:
            r = blocks[c]["models"].get(m)
            if r is None:
                cells.append("–")
            elif ci and "ci" in r:
                cells.append(f"{_p(r['acc'])} [{_p(r['ci'][0])}, {_p(r['ci'][1])}]"
                             + (f" (n {r['n']})" if r["n"] != blocks[c]["n"] else ""))
            else:
                cells.append(_p(r["acc"]) + (f" (n {r['n']})" if r["n"] != blocks[c]["n"] else ""))
        if any(x != "–" for x in cells):
            L.append(f"| {_name(m)} | " + " | ".join(cells) + " |")
    sp = [(f"{_p(blocks[c].get('spread_8'))} ({blocks[c].get('n_models')})" if blocks[c].get("spread_8") is not None
           else "–") for c in cols]
    if any(x != "–" for x in sp):
        L.append("| *spread max - min over the decision models scored (pt; number of models)* | " + " | ".join(sp) + " |")
    if defs:
        L.append("")
        L += [f"- `{c}`: {defs.get(c, '')}" for c in cols if defs.get(c)]
    return L


def to_md(res: Mapping[str, Any]) -> str:
    order = ["llm"] + list(MODELS) + list(EXTRA_MODELS)
    L = ["# Is JevAgentBench's tool selection too easy? (difficulty analysis)", "",
         f"Generated {res['generated']} at git `{res['git']}` by `scripts/benchmark_difficulty.py`. {res['label']}. "
         f"Subsets with fewer than {res['min_n']} items are merged or dropped (listed per block). Point estimates in the "
         "main tables; the 95 % bootstrap intervals (tasks resampled for JevGym / tau^2, items for L1; B = "
         f"{res['n_boot']}, seed {res['seed']}) are in the appendix tables at the end. Paper keys: "
         "`paper/generated/difficulty.tex` (`\\res{diff/...}`).", ""]
    L += ["## 1. The ceiling is the agent LLM's, not the decision models'", "",
          "| agent LLM, LLM-only | tasks | success % | own tool-decision accuracy % | failures | first wrong decision of the failed episodes |",
          "|---|---|---|---|---|---|"]
    for arm, v in res["ceiling"].items():
        fe = "; ".join(f"{a} {b}" for a, b in v["first_error"].items())
        fam = ", ".join(f"{a} {b}" for a, b in v["failed_families"].items())
        L.append(f"| {arm} | {v['tasks']} | {_p(v['success'])} | {_p(v['tool_acc'])} ({v['tool_decisions']:,}) | "
                 f"{v['failures']} ({fam}) | {fe} |")
    c = res["corpus_c"]
    L += ["", "## 2. Common-state corpus C: hard subsets (tool selection, task-weighted accuracy %)", "",
          "Hard subsets are fixed by rule from the state and its label, never from a model's answer. The agent LLM "
          "row is Flash's own choice on the occurrences of its own trajectories (`llm_states`). Counts per rule: "
          + ", ".join(f"{a} {b}" for a, b in c["counts"].items()) + ". Merged / dropped: "
          + "; ".join(f"{a}: {b}" for a, b in c["merged"].items()) + f". Flash chose an unacceptable tool at "
          f"{c['llm_own_states']['wrong']} of its {c['llm_own_states']['n']} own-state occurrences.", ""]
    L += _table(c["subsets"], order, ci=False, defs=c["definitions"])
    few = [f"{k} ({v['n']} occurrences from {v['clusters']} tasks)" for k, v in c["subsets"].items() if v["clusters"] < 100]
    if few:
        L += ["", "Subsets with at least 300 occurrences but fewer than 100 tasks (the task is the resampling unit, so "
              "their intervals are wide): " + "; ".join(few) + "."]
    L += ["", "Error types on C (wrong answers by kind; counts over occurrences):", "",
          "| model | " + " | ".join(sorted({t for v in c["error_types"].values() for t in v})) + " |",
          "|---|" + "---|" * len({t for v in c["error_types"].values() for t in v})]
    types = sorted({t for v in c["error_types"].values() for t in v})
    for m, v in c["error_types"].items():
        L.append(f"| {_name(m)} | " + " | ".join(str(v.get(t, 0)) for t in types) + " |")
    L += ["", "Accuracy by source arm of the occurrence (each model on the states of each policy; a home-state check):", "",
          "| source arm (n) | " + " | ".join(_name(m) for m in list(MODELS) + list(EXTRA_MODELS)) + " |",
          "|---|" + "---|" * (len(MODELS) + len(EXTRA_MODELS))]
    for src, v in c["by_source"].items():
        L.append(f"| {src} ({v['n']}) | " + " | ".join(_p(v.get(m)) for m in list(MODELS) + list(EXTRA_MODELS)) + " |")
    L += ["", "## 3. Stage-1 shadow logs: the same states for the agent LLM, Jev and Intern-Decision-4B", "",
          "Evaluation seeds 200000-200499 (Stage 1). Only states asked identically in both shadow runs are scored "
          "(matched by state, question and options). StartLux, JevAgentRL-4B and the other rows need a replay "
          "(`replay_plan.md`).", ""]
    for name, blk in res["shadow_stage1"].items():
        L += [f"### {name} states ({blk['matched']} of {blk['items']} decisions matched; distinct questions "
              f"{blk['distinct_questions']})", ""]
        L += _table(blk["subsets"], ["llm", "jev", "intern"], ci=False)
        if blk["merged"]:
            L.append("\nDropped (n < 300): " + "; ".join(f"{a}: {b}" for a, b in blk["merged"].items()))
        L.append("")
    t = res["tau2"]
    L += ["## 4. tau^2 reference corpus (agreement %, " + t["label"] + ")", ""]
    L += _table(t["subsets"], ["jev", "rl", "intern", "base"], ci=False)
    if t["merged"]:
        L.append("\nDropped: " + "; ".join(f"{a}: {b}" for a, b in t["merged"].items()))
    e = res["e1"]
    L += ["", "## 5. L1 static (E1, C8 population; accuracy %)", ""]
    L += _table(e["subsets"], order, ci=False)
    if e.get("merged"):
        L.append("\nDropped: " + "; ".join(f"{a}: {b}" for a, b in e["merged"].items()))
    L += ["", "## 6. Stage-1 meta decisions in shadow (Jev only; the other models need a replay)", "",
          "| states | decision | n | distinct questions | Jev % | majority label | majority % | minority n | Jev on minority % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, blk in res["meta_stage1"].items():
        for dp, v in blk.items():
            L.append(f"| {name} | {dp} | {v['n']} | {v['distinct_questions']} | {_p(v['jev_acc'])} | {v['majority_label']} | "
                     f"{_p(v['majority_acc'])} | {v['minority_n']} | {_p(v['jev_acc_minority'])} |")
    s = res["stress_dev"]
    L += ["", "## 7. Development stress slice (" + s["label"] + ")", "",
          "| checkpoint / reference | n | success % | own-state tool acc % | argument acc % | call-count-penalised reward |",
          "|---|---|---|---|---|---|"]
    for arm, v in s.items():
        if isinstance(v, dict):
            L.append(f"| {arm} | {v['n']} | {_p(v['success'])} | {_p(v['tool_acc'])} | {_p(v['arg_acc'])} | "
                     f"{v['reward'] if v['reward'] is None else round(v['reward'], 3)} |")
    # appendix: intervals
    L += ["", "## Appendix: 95 % bootstrap intervals", "", "### C", ""]
    L += _table(c["subsets"], order, ci=True)
    for name, blk in res["shadow_stage1"].items():
        L += ["", f"### Stage-1 shadow, {name} states", ""]
        L += _table(blk["subsets"], ["llm", "jev", "intern"], ci=True)
    L += ["", "### tau^2", ""]
    L += _table(t["subsets"], ["jev", "rl", "intern", "base"], ci=True)
    L += ["", "### L1 static", ""]
    L += _table(e["subsets"], order, ci=True)
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--out-tex", default=str(OUT_TEX))
    a = ap.parse_args(argv)
    res = build()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "difficulty.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "difficulty.md").write_text(to_md(res))
    Path(a.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_tex).write_text(to_tex(res))
    print(json.dumps({k: {s: round(v["models"].get("rl", {}).get("acc", float("nan")), 3) for s, v in
                          res["corpus_c"]["subsets"].items()} for k in ["rl"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
