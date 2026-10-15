#!/usr/bin/env python3
r"""Why can a decision model match Jev on the common states (L2) yet lose about 20 points live (L3)? -- OFFLINE
analysis of stored live logs only (no API call, no GPU, no Jev call), post hoc and descriptive; revises no registered
result. Precedent: scripts/benchmark_difficulty.py (no registration entry; labelled so in every output).

    python scripts/explore_drift.py fetch --dest DIR      # the two E15 GPU live logs from S3 (explicit; never implicit)
    python scripts/explore_drift.py summarize --e15-dir DIR
    # -> results/explore/drift/{drift.md,drift.json,first_errors.jsonl.gz} and paper/generated/drift.tex (\defres{drift/...})

Every model with a live REPLACE run at the tool choice (the model picks every tool), Gemini 3.5 Flash as the agent
LLM, on the first 200 fresh Stage-2 DecisionGym tasks (seeds 300000-300199): the eight rows of the size ladder
(``results/explore/startlux_live``), the attribution rows (untrained base, SFT v1, v2), E12 / E12c / E14 (arms A and
G) and E15 (OpenAI Decisions API, OpenJev, OpenJev-27B). Three questions, each answered from the logs:

(a) *Faithfulness.* On the live states that are in C (the common-state corpus ``results/stage2/s23``), is the model's
    live tool choice as accurate as its offline C answer on the same states (live and offline answer agreement; the
    discordant pairs; an exact sign test, descriptive)?
(b) *Drift.* Which live states are outside C ("novel"), split into novel *on plan* (every earlier executed tool was
    acceptable: a benign divergence such as another acceptable tool order or other LLM-typed arguments) and novel
    *after an off-plan step* (an earlier executed tool outside its acceptable set); accuracy on each; and the first
    wrong tool choice of each task (its step, whether its state is in C or novel, its error type, and for an in-C
    state what the model answered offline).
(c) *Gate recovery.* How many wrong live decisions had System-1 confidence below the cascade's fitted threshold
    (the model's own tau-hat where one was fitted, else the Jev arm's), and below 0.5 and 0.9; the same for the
    right decisions (the saving a gate gives up); mean confidence of right and wrong decisions.

The state hash is the corpus's own (``bench/stage2_corpus.question_hash`` over the exact System-1 state, the option
list, the option descriptions of the task's environment and the question text); correctness is the benchmark's own
rule (``bench/metrics.is_correct``: membership in the logged acceptable tool set). Accuracies over states are pooled
over labelled decisions; the C column reproduces the published task-mean (``bench/stage2_corpus.task_accuracy``).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "results/explore/drift"
OUT_TEX = ROOT / "paper/generated/drift.tex"
S23 = ROOT / "results/stage2/s23"
RUNS = ROOT / "runs"
CAL = ROOT / "results/calibration"
LABEL = ("post hoc, descriptive; stored live logs only (no API call, no GPU, no Jev call); revises no registered "
         "result; no registration entry (the convention of scripts/benchmark_difficulty.py)")
S3_BUCKET = "s3://<bucket>/e15_baselines/runs"
S3_FILES = {"openjev": "e15_live_openjev/openjev_ts_replace_f.jsonl",
            "openjevhf": "e15_live_openjevhf/openjevhf_ts_replace_f.jsonl"}
TAUS_FIXED = (0.5, 0.9)
JEV_TAU_KEY = "jev"

# model key -> (label, live log under runs/ (or "e15:" + path under --e15-dir), offline C answers, own tau-hat record,
# published keys (C accuracy, live success) in paper/generated/*.tex for the reproduction check, group)
MODELS: dict[str, dict[str, Any]] = {
    "jev": dict(label="Jev", live="stage2_rl_states/jev_ts_replace_f.jsonl", answers=S23 / "answers_jev.jsonl",
                tau=CAL / "tauhat_jev_shadow_cal_tool_select.json",
                pub=("explore/startlux_live/jev/corpus_acc", "explore/startlux_live/jev/live_success"), group="off the shelf"),
    "intern": dict(label="Intern-Decision-4B", live="stage2_rl_states/intern_ts_replace_f.jsonl",
                   answers=S23 / "answers_intern.jsonl", tau=CAL / "tauhat_intern_shadow_cal_tool_select.json",
                   pub=("explore/startlux_live/intern/corpus_acc", "explore/startlux_live/intern/live_success"), group="off the shelf"),
    "sl0.8b": dict(label="StartLux-0.8B", live="explore_startlux_live/sl0.8b_ts_replace_f.jsonl",
                   answers=S23 / "answers_sl0.8b.jsonl", tau=None,
                   pub=("explore/startlux_live/sl0.8b/corpus_acc", "explore/startlux_live/sl0.8b/live_success"), group="off the shelf"),
    "sl2b": dict(label="StartLux-2B", live="explore_startlux_live/sl2b_ts_replace_f.jsonl",
                 answers=S23 / "answers_sl2b.jsonl", tau=None,
                 pub=("explore/startlux_live/sl2b/corpus_acc", "explore/startlux_live/sl2b/live_success"), group="off the shelf"),
    "sl4b": dict(label="StartLux-4B", live="stage2_startlux_states/sl4b_ts_replace_f.jsonl",
                 answers=S23 / "answers_sl4b.jsonl", tau=CAL / "tauhat_sl4b_shadow_cal_tool_select.json",
                 pub=("explore/startlux_live/sl4b/corpus_acc", "explore/startlux_live/sl4b/live_success"), group="off the shelf"),
    "sl9b": dict(label="StartLux-9B", live="explore_startlux_live/sl9b_ts_replace_f.jsonl",
                 answers=S23 / "answers_sl9b.jsonl", tau=None,
                 pub=("explore/startlux_live/sl9b/corpus_acc", "explore/startlux_live/sl9b/live_success"), group="off the shelf"),
    "sl27b": dict(label="StartLux-27B", live="stage2_startlux_states/sl27b_ts_replace_f.jsonl",
                  answers=S23 / "answers_sl27b.jsonl", tau=None,
                  pub=("explore/startlux_live/sl27b/corpus_acc", "explore/startlux_live/sl27b/live_success"), group="off the shelf"),
    "oaidec": dict(label="OpenAI Decisions API (gpt-6-luna)", live="e15_live_oaidec/oaidec_ts_replace_f.jsonl",
                   answers=ROOT / "results/explore/e15_baselines/c/answers_oaidec.jsonl.gz", tau=None,
                   pub=("e15/oaidec/corpus_acc", "e15/oaidec/live_success"), group="off the shelf (E15)"),
    "openjev": dict(label="OpenJev (DiffusionGemma-26B-A4B)", live="e15:e15_live_openjev/openjev_ts_replace_f.jsonl",
                    answers=ROOT / "results/explore/e15_baselines/c/answers_openjev.jsonl.gz", tau=None,
                    pub=("e15/openjev/corpus_acc", "e15/openjev/live_success"), group="off the shelf (E15)"),
    "openjevhf": dict(label="OpenJev-27B (HF, FP8)", live="e15:e15_live_openjevhf/openjevhf_ts_replace_f.jsonl",
                      answers=ROOT / "results/explore/e15_baselines/c/answers_openjevhf.jsonl.gz", tau=None,
                      pub=("e15/openjevhf/corpus_acc", "e15/openjevhf/live_success"), group="off the shelf (E15)"),
    "rl": dict(label="AgentDecision-4B (JevAgentRL-4B, frozen)", live="stage2_rl_states/rl_ts_replace_f.jsonl",
               answers=S23 / "answers_rl.jsonl", tau=ROOT / "results/gpu_round5/cal/tauhat_rl_shadow_cal_tool_select.json",
               pub=("explore/startlux_live/frozen/corpus_acc", "explore/startlux_live/frozen/live_success"), group="ours"),
    "e12": dict(label="AgentDecision-4B + public data (E12)", live="e12_live/e12_ts_replace_f.jsonl",
                answers=ROOT / "results/explore/e12/box/s23/answers_e12_mixed.jsonl", tau=None,
                pub=("e12/e12/c/acc", "e12/e12/live/success"), group="ours"),
    "e12c": dict(label="StartLux-4B + our recipe (E12c)", live="e12c_live/e12c_ts_replace_f.jsonl",
                 answers=ROOT / "results/explore/e12/box_c/s23/answers_e12c_sl4b.jsonl", tau=None,
                 pub=("e12c/e12c/c/acc", "e12c/e12c/live/success"), group="ours"),
    "e14a": dict(label="0.8B scoring head, agent states (E14 A)", live="e14_live_a/e14a_ts_replace_f.jsonl",
                 answers=ROOT / "results/explore/e14_small/records/s23/answers_e14a.jsonl.gz",
                 tau=ROOT / "results/explore/e14_small/cal/tauhat_e14cal_e14a_tool_select.json",
                 pub=("e14/a/c/acc", "e14/a/live/success"), group="ours"),
    "e14g": dict(label="0.8B scoring head, generic mix (E14 G)", live="e14_live_a_g/e14g_ts_replace_f.jsonl",
                 answers=ROOT / "results/explore/e14_small/records/s23/answers_e14g.jsonl.gz",
                 tau=ROOT / "results/explore/e14_small/cal/tauhat_e14cal_e14g_tool_select.json",
                 pub=("e14/g/c/acc", "e14/g/live/success"), group="ours"),
    "base": dict(label="Qwen3-4B (untrained)", live="stage2_rl_attribution/rl_base_ts_replace_f.jsonl",
                 answers=S23 / "answers_base.jsonl", tau=None,
                 pub=("stage2_rl/base/corpus_acc", "stage2_rl/base/live_success"), group="attribution"),
    "sft_v1": dict(label="SFT only (v1)", live="stage2_rl_attribution/rl_sft_v1_ts_replace_f.jsonl",
                   answers=S23 / "answers_sft_v1.jsonl", tau=None,
                   pub=("stage2_rl/sft_v1/corpus_acc", "stage2_rl/sft_v1/live_success"), group="attribution"),
    "v2": dict(label="JevAgentRL-v2 (exploratory)", live="stage2_v2_states/v2_ts_replace_f.jsonl",
               answers=S23 / "answers_v2.jsonl", tau=CAL / "tauhat_v2_shadow_cal_tool_select.json",
               pub=("stage2_v2/corpus_acc", "stage2_v2/live_success"), group="attribution"),
}
# live runs on these tasks that are NOT rows, and why (checked 2026-10-08)
EXCLUDED = {
    "e14ag (E14 arm AG)": "no live run on seeds 300000-300199 (runs/e14_live_b is arm e14_final on seeds 300500+)",
    "E5 students": "live on 100 tasks of seeds 300700+ through OpenRouter (runs/e5_small_live), another protocol",
    "E7/E7c/E7e/E7f rows": "DecisionGym-Hard (seeds 310000+), another suite",
    "stage2_x / stage2_sol": "DeepSeek-V4-Pro / GPT-6.1-Sol as the agent LLM, not Flash, and profile arms (cascade)",
    "E16 arms": "in progress in another session; no record read",
    "llm_only (Flash alone)": "not a decision model; its states are a C source (hash check only)",
}
# the C source arms (A7.10 (c)): their live states must land in C by construction (the hash check)
C_SOURCES = {"llm_only": "stage2_jevgym/llm_only.jsonl", "jev": "stage2_rl_states/jev_ts_replace_f.jsonl",
             "intern": "stage2_rl_states/intern_ts_replace_f.jsonl", "rl": "stage2_rl_states/rl_ts_replace_f.jsonl"}


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


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def latest(path: Path) -> dict[str, dict[str, Any]]:
    from jevharness.harness.trajectory import iter_records, latest_by_task

    if not path.exists():
        return {}
    return latest_by_task(r for r in iter_records(path) if "task_id" in r)


def _mean(v: Sequence[float]) -> float | None:
    return sum(v) / len(v) if v else None


def _share(num: int, den: int) -> float | None:
    return num / den if den else None


def sign_test_p(a: int, b: int) -> float | None:
    """Two-sided exact sign test on the discordant pairs (a vs b): P(X <= min) * 2 under Binomial(a + b, 1/2)."""
    n = a + b
    if n == 0:
        return None
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = (x[i] > x[j]) - (x[i] < x[j]), (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else None


def published_keys(gen_dir: Path = ROOT / "paper/generated") -> dict[str, str]:
    """\\defres{key}{value} of every generated TeX file (the published numbers, for the reproduction check)."""
    import re

    out: dict[str, str] = {}
    pat = re.compile(r"\\defres\{([^}]*)\}\{([^}]*)\}")
    for p in sorted(gen_dir.glob("*.tex")) if gen_dir.exists() else []:
        for key, val in pat.findall(p.read_text(errors="ignore")):
            out.setdefault(key, val)
    return out


# ================================================================== the corpus C and the live states
def load_corpus(s23: Path = S23) -> dict[str, Any]:
    """C: question hash -> {state, options, descriptions}; occurrences (task, source, step, qhash, label); the label
    (acceptable set) of each hashed state as the corpus recorded it."""
    qs = {q["qhash"]: q for q in read_jsonl(s23 / "questions.jsonl.gz")}
    occ = read_jsonl(s23 / "occurrences.jsonl.gz")
    label: dict[str, list[str] | None] = {}
    for o in occ:
        at = (o.get("gold_aux") or {}).get("acceptable_tools")
        label.setdefault(o["qhash"], sorted(at) if isinstance(at, list) else None)
    meta = json.loads((s23 / "corpus.json").read_text()) if (s23 / "corpus.json").exists() else {}
    return {"questions": qs, "occurrences": occ, "label": label, "hashes": set(qs),
            "tasks": sorted({o["task"] for o in occ}), "meta": meta}


def description_lookup(run_dir: Path = RUNS / "stage2_jevgym") -> Callable[[str], Mapping[str, str]]:
    """task_id -> {tool name: the one-line description System-1 saw}, from the suite's own environments (exactly as
    stage2_corpus.build_corpus hashes them)."""
    from jevharness.bench import stage2_corpus as SC
    from jevharness.bench.meta_gold import suite_factories

    facs = suite_factories(run_dir)
    cache: dict[str, Mapping[str, str]] = {}

    def look(task: str) -> Mapping[str, str]:
        if task not in cache:
            cache[task] = SC.tool_descriptions(facs.get(task))
        return cache[task]
    return look


def live_states(traj: Mapping[str, Any], descs: Mapping[str, str]) -> list[dict[str, Any]]:
    """One record per step with a tool-selection question (the step's first tool_select decision, as the corpus
    takes it): the corpus hash of its state, the label, the model's answer and confidence, the executed tool, whether
    an earlier executed tool was outside its acceptable set (after an off-plan step)."""
    from jevharness.bench import metrics as M
    from jevharness.bench import stage2_corpus as SC

    out: list[dict[str, Any]] = []
    off_plan_seen = False
    for st in traj.get("steps") or []:
        rec = next((d for d in st.get("decisions") or [] if d.get("dp") == "tool_select"
                    and d.get("mode") in SC.TOOL_MODES and d.get("options") and d.get("state")), None)
        if rec is None:
            continue
        opts = list(rec["options"])
        executed = st.get("tool")
        s1 = rec.get("s1_value")
        labelled = M.has_gold(rec)
        at = M.acceptable_of(rec)
        r = {"step": int(st.get("step", 0)), "qhash": SC.question_hash(rec["state"], opts, [descs.get(o, "") for o in opts]),
             "labelled": labelled, "gold": rec.get("gold"), "at": sorted(at) if at is not None else None,
             "rec": {k: rec.get(k) for k in ("dp", "gold", "gold_aux", "gold_status")},
             "s1_value": s1, "s1_conf": rec.get("s1_conf"), "executed": executed, "mode": rec.get("mode"),
             "fallthrough": s1 is not None and executed is not None and s1 != executed,
             "after_off_plan": off_plan_seen,
             "s1_correct": M.is_correct(rec, s1) if labelled and s1 is not None else None,
             "exec_correct": M.is_correct(rec, executed) if labelled and executed is not None else None}
        out.append(r)
        if r["exec_correct"] is False:
            off_plan_seen = True
    return out


def offline_answers(path: Path | None) -> dict[str, dict[str, Any]]:
    """qhash -> the replay answer record (the latest record of a question decides, as E15 reads them)."""
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(path):
        out[r["qhash"]] = r
    return out


def read_tau(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    t = json.loads(path.read_text()).get("tau_hat")
    return None if t is None else float(t)


# ================================================================== per-model analysis
def first_error_type(state: Mapping[str, Any], near: Mapping[str, list[tuple[str, bool]]]) -> str:
    from benchmark_difficulty import error_type  # the difficulty analysis's classifier (same scripts/ directory)

    return error_type(state["executed"], {"rec": state["rec"], "gold": state["gold"]}, near)


def write_tools(near: Mapping[str, list[tuple[str, bool]]] | None) -> set[str]:
    """The tools that change the world: the three core state-changing tools and their state-changing look-alikes."""
    from benchmark_difficulty import STATE_CHANGING

    out = set(STATE_CHANGING)
    for v in (near or {}).values():
        out |= {n for n, w in v if w}
    return out


def consequence(executed: str | None, writes: set[str], final_tool: str = "finish") -> str:
    """What a wrong tool choice does to the task: an *irreversible* world change or a premature finish (the task is
    lost at once) against a *recoverable* wrong choice (a lookup, an unneeded question, a read-only look-alike)."""
    if executed in writes:
        return "wrong write action"
    if executed == final_tool:
        return "premature finish"
    return "recoverable"


def analyse_model(trajs: Mapping[str, Mapping[str, Any]], corpus: Mapping[str, Any],
                  descs_for: Callable[[str], Mapping[str, str]], answers: Mapping[str, Mapping[str, Any]],
                  taus: Mapping[str, float | None], near: Mapping[str, list[tuple[str, bool]]] | None = None
                  ) -> dict[str, Any]:
    """Everything the readout needs for one live run: tasks, states (in C / novel, on plan / after an off-plan
    step), live accuracies, the faithfulness pairs, the first wrong tool choice per task and the gate counts."""
    from jevharness.bench import stage2_corpus as SC

    hashes: set[str] = corpus["hashes"]
    c_label: Mapping[str, list[str] | None] = corpus["label"]
    writes = write_tools(near)
    ok_tasks = {t: tr for t, tr in trajs.items() if not tr.get("error")}
    res: dict[str, Any] = {"tasks": len(ok_tasks), "errored": len(trajs) - len(ok_tasks),
                           "success": _mean([float(bool((tr.get("result") or {}).get("success"))) for tr in ok_tasks.values()])}
    states: list[dict[str, Any]] = []
    firsts: list[dict[str, Any]] = []
    unlabelled = fall = label_mismatch = 0
    for t, tr in ok_tasks.items():
        sts = live_states(tr, descs_for(t))
        first: dict[str, Any] | None = None
        for s in sts:
            s["task"] = t
            s["in_c"] = s["qhash"] in hashes
            fall += s["fallthrough"]
            if not s["labelled"]:
                unlabelled += 1
                continue
            if s["in_c"] and c_label.get(s["qhash"]) is not None and s["at"] is not None \
                    and c_label[s["qhash"]] != s["at"]:
                label_mismatch += 1
            a = answers.get(s["qhash"]) if s["in_c"] else None
            s["offline_answer"] = None if a is None else (a.get("answer") if not a.get("error") and a.get("valid", True) else None)
            s["offline_correct"] = None if a is None else SC.answer_correct(
                {"gold": s["gold"], "gold_aux": s["rec"]["gold_aux"], "gold_status": s["rec"]["gold_status"]}, a)
            states.append(s)
            if first is None and s["exec_correct"] is False:
                first = {"task": t, "step": s["step"], "in_c": s["in_c"], "conf": s["s1_conf"],
                         "success": bool((tr.get("result") or {}).get("success")),
                         "type": first_error_type(s, near) if near is not None else None,
                         "consequence": consequence(s["executed"], writes),
                         "offline_correct": s["offline_correct"], "offline_answer": s["offline_answer"],
                         "executed": s["executed"], "gold": s["gold"]}
        firsts.append(first or {"task": t, "step": None, "in_c": None, "conf": None,
                                "success": bool((tr.get("result") or {}).get("success")), "type": None,
                                "consequence": None, "offline_correct": None})
    # ---- states and accuracies (pooled over labelled decisions)
    def acc(sub: Sequence[Mapping[str, Any]], key: str = "s1_correct") -> dict[str, Any]:
        v = [float(bool(s[key])) for s in sub if s.get(key) is not None]
        return {"n": len(v), "acc": _mean(v)}

    in_c = [s for s in states if s["in_c"]]
    novel = [s for s in states if not s["in_c"]]
    novel_on = [s for s in novel if not s["after_off_plan"]]
    novel_off = [s for s in novel if s["after_off_plan"]]
    in_c_on = [s for s in in_c if not s["after_off_plan"]]
    in_c_off = [s for s in in_c if s["after_off_plan"]]
    res["states"] = {"labelled": len(states), "unlabelled": unlabelled, "fallthrough": fall,
                     "label_mismatch_in_c": label_mismatch, "in_c": len(in_c), "novel": len(novel),
                     "novel_on_plan": len(novel_on), "novel_after_off_plan": len(novel_off),
                     "in_c_on_plan": len(in_c_on), "in_c_after_off_plan": len(in_c_off),
                     "in_c_share": _share(len(in_c), len(states)), "novel_share": _share(len(novel), len(states)),
                     "novel_on_plan_share": _share(len(novel_on), len(states)),
                     "novel_after_off_plan_share": _share(len(novel_off), len(states)),
                     "after_off_plan_share": _share(len(novel_off) + len(in_c_off), len(states)),
                     "distinct_in_c": len({s["qhash"] for s in in_c}), "distinct_novel": len({s["qhash"] for s in novel})}
    res["acc"] = {"all": acc(states), "in_c": acc(in_c), "novel": acc(novel), "novel_on_plan": acc(novel_on),
                  "novel_after_off_plan": acc(novel_off), "in_c_on_plan": acc(in_c_on), "in_c_after_off_plan": acc(in_c_off),
                  "on_plan": acc([s for s in states if not s["after_off_plan"]]),
                  "after_off_plan": acc([s for s in states if s["after_off_plan"]]),
                  "executed_all": acc(states, "exec_correct")}
    ask = [s for s in states if s["gold"] == "ask_user"]
    ask_wrong = [s for s in ask if s["s1_correct"] is False]
    must = [s for s in states if s["at"] == ["ask_user"]]
    early = [s for s in states if s["executed"] == "ask_user" and s["gold"] != "ask_user" and s["s1_correct"]]
    res["mustask"] = {"n": len(must), "tasks": len({s["task"] for s in must}),
                      "acc": _mean([float(bool(s["s1_correct"])) for s in must]),
                      "in_c": sum(1 for s in must if s["in_c"]),
                      "acted": sum(1 for s in must if s["s1_correct"] is False and s["executed"] in writes),
                      "acted_share": _share(sum(1 for s in must if s["s1_correct"] is False and s["executed"] in writes), len(must)),
                      "step_median": statistics.median([s["step"] for s in must]) if must else None,
                      "c_acc": corpus_task_accuracy(corpus, answers, must_ask=True)}
    res["askearly"] = {"n": len(early), "tasks": len({s["task"] for s in early}),
                       "tasks_share": _share(len({s["task"] for s in early}), len(ok_tasks))}
    res["ask"] = {"n": len(ask), "acc": _mean([float(bool(s["s1_correct"])) for s in ask]),
                  "in_c": sum(1 for s in ask if s["in_c"]), "wrong": len(ask_wrong),
                  "acted": sum(1 for s in ask_wrong if s["executed"] in writes),
                  "acted_share": _share(sum(1 for s in ask_wrong if s["executed"] in writes), len(ask)),
                  "acted_share_of_wrong": _share(sum(1 for s in ask_wrong if s["executed"] in writes), len(ask_wrong)),
                  "tasks": len({s["task"] for s in ask}),
                  "c_acc": corpus_task_accuracy(corpus, answers, gold="ask_user")}
    # ---- (a) faithfulness on the in-C states the model visited (live answer vs its offline C answer, same state)
    pairs = [s for s in in_c if s["offline_correct"] is not None]
    lw_or = sum(1 for s in pairs if not s["s1_correct"] and s["offline_correct"])
    lr_ow = sum(1 for s in pairs if s["s1_correct"] and not s["offline_correct"])
    res["faith"] = {"n": len(pairs), "live_acc": _mean([float(bool(s["s1_correct"])) for s in pairs]),
                    "offline_acc": _mean([float(bool(s["offline_correct"])) for s in pairs]),
                    "agree": _mean([float(s["s1_value"] == s["offline_answer"]) for s in pairs]),
                    "live_wrong_offline_right": lw_or, "live_right_offline_wrong": lr_ow,
                    "sign_p": sign_test_p(lw_or, lr_ow), "missing_offline": len(in_c) - len(pairs)}
    # ---- (b) the first wrong tool choice per task
    with_first = [f for f in firsts if f["step"] is not None]
    failed = [f for f in firsts if not f["success"]]
    failed_with = [f for f in failed if f["step"] is not None]
    inc_first = [f for f in with_first if f["in_c"]]
    res["first"] = {"tasks": len(firsts), "with_wrong_choice": len(with_first),
                    "in_c": len(inc_first), "novel": len(with_first) - len(inc_first),
                    "none": len(firsts) - len(with_first),
                    "in_c_share": _share(len(inc_first), len(firsts)),
                    "novel_share": _share(len(with_first) - len(inc_first), len(firsts)),
                    "none_share": _share(len(firsts) - len(with_first), len(firsts)),
                    "failed": len(failed), "failed_with_wrong_choice": len(failed_with),
                    "failed_with_wrong_choice_share": _share(len(failed_with), len(failed)),
                    "failed_without_wrong_choice": len(failed) - len(failed_with),
                    "recovered": sum(1 for f in with_first if f["success"]),
                    "recovered_share": _share(sum(1 for f in with_first if f["success"]), len(with_first)),
                    "step_median": statistics.median([f["step"] for f in with_first]) if with_first else None,
                    "step_mean": _mean([float(f["step"]) for f in with_first]),
                    "step_le1_share": _share(sum(1 for f in with_first if f["step"] <= 1), len(with_first)),
                    "steps": dict(Counter(f["step"] for f in with_first)),
                    "types": dict(Counter(f["type"] for f in with_first).most_common()),
                    "in_c_offline_right": sum(1 for f in inc_first if f["offline_correct"]),
                    "in_c_offline_wrong": sum(1 for f in inc_first if f["offline_correct"] is False),
                    "in_c_offline_right_share": _share(sum(1 for f in inc_first if f["offline_correct"]), len(inc_first)),
                    "in_c_same_answer": sum(1 for f in inc_first if f["offline_answer"] == f["executed"]),
                    "consequence": dict(Counter(f["consequence"] for f in with_first).most_common()),
                    "irreversible": sum(1 for f in with_first if f["consequence"] != "recoverable"),
                    "irreversible_share": _share(sum(1 for f in with_first if f["consequence"] != "recoverable"), len(firsts)),
                    "irreversible_of_failed_share": _share(sum(1 for f in failed_with if f["consequence"] != "recoverable"), len(failed)),
                    "loss": {"total": _share(len(failed), len(firsts)),
                             "irreversible": _share(sum(1 for f in failed_with if f["consequence"] != "recoverable"), len(firsts)),
                             "compounding": _share(sum(1 for f in failed_with if f["consequence"] == "recoverable"), len(firsts)),
                             "no_wrong_choice": _share(len(failed) - len(failed_with), len(firsts))},
                    "recovered_by_consequence": {c: _share(sum(1 for f in with_first if f["consequence"] == c and f["success"]),
                                                           sum(1 for f in with_first if f["consequence"] == c))
                                                 for c in ("wrong write action", "premature finish", "recoverable")},
                    "per_task": [{k: v for k, v in f.items()} for f in firsts]}
    # ---- (c) the gate: wrong / right decisions below each threshold, and the failed tasks' first wrong choice
    right = [s["s1_conf"] for s in states if s["s1_correct"] and s["s1_conf"] is not None]
    wrong = [s["s1_conf"] for s in states if s["s1_correct"] is False and s["s1_conf"] is not None]
    gate: dict[str, Any] = {"conf_right_mean": _mean(right), "conf_wrong_mean": _mean(wrong),
                            "conf_right_median": statistics.median(right) if right else None,
                            "conf_wrong_median": statistics.median(wrong) if wrong else None,
                            "n_right": len(right), "n_wrong": len(wrong), "at": {}}
    for name, tau in taus.items():
        if tau is None:
            continue
        fw = [f for f in failed_with if f["conf"] is not None]
        gate["at"][name] = {"tau": tau, "wrong_below": _share(sum(c < tau for c in wrong), len(wrong)),
                            "right_below": _share(sum(c < tau for c in right), len(right)),
                            "all_below": _share(sum(c < tau for c in right + wrong), len(right) + len(wrong)),
                            "failed_first_below": _share(sum(f["conf"] < tau for f in fw), len(fw)),
                            "failed_first_below_n": sum(f["conf"] < tau for f in fw), "failed_first_n": len(fw)}
    res["gate"] = gate
    return res


def _must_ask(o: Mapping[str, Any]) -> bool:
    at = (o.get("gold_aux") or {}).get("acceptable_tools")
    return isinstance(at, list) and sorted(at) == ["ask_user"]


def walk_predictions(corpus: Mapping[str, Any], answers: Mapping[str, Mapping[str, Any]], writes: set[str],
                     sources: Sequence[str] | None = None, on_plan_only: bool = False,
                     early_ask_resolves: bool = False) -> dict[str, dict[str, float]]:
    """A consequence-aware offline score from the C answers alone, no live log: walk each reference trajectory of C
    (the source arm's labelled tool-choice states of a task in step order), find the first state where the model's
    offline answer is wrong and predict the task as lost (0) iff that first error is irreversible (a wrong write
    action or a premature finish); otherwise 1. Variants: ``on_plan_only`` stops the walk at the reference's own
    first off-plan step (states after it are the reference policy's drift); ``early_ask_resolves`` ends the walk as
    a success when the model asks the customer where asking is acceptable but not the canonical tool (the live logs
    show that such an early ask settles the task's later must-ask state). Returns task -> source -> prediction."""
    from jevharness.bench import stage2_corpus as SC

    by: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for o in corpus["occurrences"]:
        if sources is None or o["source"] in sources:
            by[(o["task"], o["source"])].append(o)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, src), occ in by.items():
        pred = 1.0
        for o in sorted(occ, key=lambda x: int(x.get("step", 0))):
            if on_plan_only and o.get("stratum") != "on_plan":
                break
            a = answers.get(o["qhash"]) or {}
            val = a.get("answer") if not a.get("error") and a.get("valid", True) else None
            if SC.answer_correct(o, a):
                if early_ask_resolves and val == "ask_user" and o.get("gold") != "ask_user":
                    break
                continue
            pred = 0.0 if consequence(val, writes) != "recoverable" else 1.0
            break
        out[task][src] = pred
    return out


def predicted_success(preds: Mapping[str, Mapping[str, float]], tasks: Iterable[str]) -> float | None:
    """Mean over tasks of the mean over the task's reference trajectories (every available source)."""
    per = [_mean(list(preds[t].values())) for t in tasks if preds.get(t)]
    return _mean([x for x in per if x is not None])


def corpus_task_accuracy(corpus: Mapping[str, Any], answers: Mapping[str, Mapping[str, Any]],
                         gold: str | None = None, must_ask: bool = False) -> float | None:
    """The published C accuracy (task mean over the corpus's occurrences; stage2_corpus.task_accuracy); with
    ``gold`` the stratum of occurrences with that answer key (``ask_user``: E15's ask-vs-act column); with
    ``must_ask`` the occurrences whose acceptable set is exactly {ask_user} (the difficulty analysis's must-ask
    stratum)."""
    from jevharness.bench import stage2_corpus as SC

    if not answers:
        return None
    occ = [o for o in corpus["occurrences"] if (gold is None or o.get("gold") == gold) and (not must_ask or _must_ask(o))]
    per = SC.task_accuracy(occ, answers)
    return _mean(list(per.values()))


def hash_check(corpus: Mapping[str, Any], descs_for: Callable[[str], Mapping[str, str]],
               sources: Mapping[str, str] = C_SOURCES, runs: Path = RUNS) -> dict[str, Any]:
    """The corpus hash recomputed on the C source arms' own logs (restricted to C's tasks) must land in C."""
    out: dict[str, Any] = {}
    tasks = set(corpus["tasks"])
    for name, rel in sources.items():
        lat = latest(runs / rel)
        n = hit = 0
        for t, tr in lat.items():
            if t not in tasks or tr.get("error"):
                continue
            for s in live_states(tr, descs_for(t)):
                if s["labelled"]:
                    n += 1
                    hit += s["qhash"] in corpus["hashes"]
        out[name] = {"labelled_states": n, "in_c": hit, "match": _share(hit, n), "found": bool(lat)}
    return out


# ================================================================== build
def resolve_live(spec: Mapping[str, Any], e15_dir: Path | None) -> Path | None:
    rel = spec["live"]
    if rel.startswith("e15:"):
        return None if e15_dir is None else e15_dir / rel[4:]
    return RUNS / rel


def build(e15_dir: Path | None, models: Mapping[str, Mapping[str, Any]] = MODELS, corpus: Mapping[str, Any] | None = None,
          descs_for: Callable[[str], Mapping[str, str]] | None = None, with_check: bool = True) -> dict[str, Any]:
    from benchmark_difficulty import near_map

    corpus = corpus or load_corpus()
    descs_for = descs_for or description_lookup()
    near = near_map()
    pub = published_keys()
    jev_tau = read_tau(models.get(JEV_TAU_KEY, {}).get("tau")) if JEV_TAU_KEY in models else None
    res: dict[str, Any] = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "git": git("rev-parse", "--short=12", "HEAD"), "label": LABEL,
                           "corpus": {"questions": len(corpus["questions"]), "occurrences": len(corpus["occurrences"]),
                                      "tasks": len(corpus["tasks"]), "files": (corpus.get("meta") or {}).get("files"),
                                      "ask_occurrences": sum(1 for o in corpus["occurrences"] if o.get("gold") == "ask_user"),
                                      "ask_tasks": len({o["task"] for o in corpus["occurrences"] if o.get("gold") == "ask_user"}),
                                      "ask_share": _share(sum(1 for o in corpus["occurrences"] if o.get("gold") == "ask_user"),
                                                          len(corpus["occurrences"])),
                                      "mustask_occurrences": sum(1 for o in corpus["occurrences"] if _must_ask(o)),
                                      "mustask_tasks": len({o["task"] for o in corpus["occurrences"] if _must_ask(o)}),
                                      "mustask_share": _share(sum(1 for o in corpus["occurrences"] if _must_ask(o)),
                                                              len(corpus["occurrences"]))},
                           "s3": {k: f"{S3_BUCKET}/{v}" for k, v in S3_FILES.items()}, "jev_tau": jev_tau,
                           "models": {}, "not_found": {}, "excluded": dict(EXCLUDED)}
    if with_check:
        res["hash_check"] = hash_check(corpus, descs_for)
    for key, spec in models.items():
        p = resolve_live(spec, e15_dir)
        if p is None or not p.exists():
            res["not_found"][key] = (f"{spec['live'][4:]} not under --e15-dir (run `fetch` first)" if spec["live"].startswith("e15:")
                                     else f"runs/{spec['live']} missing")
            continue
        trajs = latest(p)
        ans = offline_answers(spec.get("answers"))
        own = read_tau(spec.get("tau"))
        taus: dict[str, float | None] = {"own": own, "jev": jev_tau}
        for t in TAUS_FIXED:
            taus[f"fixed_{t}"] = t
        row = analyse_model(trajs, corpus, descs_for, ans, taus, near)
        live_tasks = [t for t, tr in trajs.items() if not tr.get("error")]
        writes = write_tools(near)
        row["pred"] = {"tasks": len(live_tasks)}
        for name, kw in (("all_sources", {}), ("llm_only", {"sources": ("llm_only",)}),
                         ("on_plan", {"on_plan_only": True}), ("early_ask", {"early_ask_resolves": True})):
            row["pred"][name] = predicted_success(walk_predictions(corpus, ans, writes, **kw), live_tasks) if ans else None
        row.update({"label": spec["label"], "group": spec.get("group"), "live_path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                    "live_sha256": sha16(p), "answers_path": str(spec["answers"].relative_to(ROOT)) if spec.get("answers") and spec["answers"].exists() else None,
                    "own_tau": own, "tau_used": own if own is not None else jev_tau,
                    "tau_source": "own fit" if own is not None else "Jev arm's fit (none fitted for this model)",
                    "c_acc": corpus_task_accuracy(corpus, ans),
                    "published": {"c_acc": pub.get(spec["pub"][0]) if spec.get("pub") else None,
                                  "live_success": pub.get(spec["pub"][1]) if spec.get("pub") else None}})
        res["models"][key] = row
    # the offline scores against live success, over every row and over the off-the-shelf rows (E15's set)
    res["pred"] = prediction_summary(res["models"])
    # across models: do weaker-live models visit more novel states?
    rows = [v for v in res["models"].values() if v["success"] is not None]
    if len(rows) >= 3:
        res["kendall"] = {"novel_share_vs_live": kendall_tau_b([r["states"]["novel_share"] for r in rows], [r["success"] for r in rows]),
                          "novel_after_off_plan_share_vs_live": kendall_tau_b([r["states"]["novel_after_off_plan_share"] for r in rows],
                                                                              [r["success"] for r in rows]),
                          "novel_on_plan_share_vs_live": kendall_tau_b([r["states"]["novel_on_plan_share"] for r in rows],
                                                                       [r["success"] for r in rows]),
                          "n_models": len(rows)}
    return res


def prediction_summary(models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Predicted live success (the walk), the plain C mean and the must-ask accuracy against actual live success:
    per-row errors, the mean absolute error and Kendall's tau-b over every row and over the off-the-shelf rows."""
    sets = {"all": [m for m, r in models.items() if r.get("success") is not None and r.get("c_acc") is not None],
            "offtheshelf": [m for m, r in models.items() if r.get("success") is not None and r.get("c_acc") is not None
                            and str(r.get("group", "")).startswith("off the shelf")]}
    scores = {"cmean": lambda r: r["c_acc"], "mustask": lambda r: r["mustask"]["c_acc"],
              "pred": lambda r: r["pred"]["all_sources"], "pred_llm": lambda r: r["pred"]["llm_only"],
              "pred_onplan": lambda r: r["pred"]["on_plan"], "pred_earlyask": lambda r: r["pred"]["early_ask"]}
    out: dict[str, Any] = {"rows": {}, "kendall": {}, "mae": {}, "n": {k: len(v) for k, v in sets.items()}, "sets": sets}
    for m in sets["all"]:
        r = models[m]
        out["rows"][m] = {k: f(r) for k, f in scores.items()}
        out["rows"][m]["actual"] = r["success"]
        for k in ("pred", "pred_llm", "pred_onplan", "pred_earlyask"):
            v = out["rows"][m][k]
            out["rows"][m][f"{k}_err"] = None if v is None else v - r["success"]
    for name, rows in sets.items():
        out["kendall"][name] = {}
        out["mae"][name] = {}
        for k, f in scores.items():
            xs = [(f(models[m]), models[m]["success"]) for m in rows if f(models[m]) is not None]
            out["kendall"][name][k] = kendall_tau_b([x for x, _ in xs], [y for _, y in xs]) if len(xs) >= 3 else None
            if k.startswith("pred"):
                out["mae"][name][k] = _mean([abs(x - y) for x, y in xs])
    return out


# ================================================================== outputs
def _p(x: Any, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.{d}f}"


def _f(x: Any, d: int = 2) -> str:
    return "n/a" if x is None else ("inf" if isinstance(x, float) and math.isinf(x) else f"{float(x):.{d}f}")


def _tau(x: float | None, tex: bool = False) -> str:
    if x is None:
        return "n/a"
    if math.isinf(x):
        return r"$\infty$" if tex else "always defer"
    return f"{x:.2f}" if abs(x - round(x, 2)) < 5e-4 else f"{x:.4f}"


TYPE_SLUG = {"changed state before asking": "act_before_ask", "wrong core tool": "wrong_core_tool",
             "unneeded question": "unneeded_question", "look-alike of the gold tool": "lookalike_gold",
             "other look-alike": "lookalike_other", "finished too early": "finished_too_early",
             "unrelated distractor": "unrelated", "no valid answer": "no_answer"}
CONSEQ_SLUG = {"wrong write action": "write", "premature finish": "finish", "recoverable": "recoverable"}


def _tc(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def tex_rows(res: Mapping[str, Any]) -> dict[str, str]:
    """\\defres keys; a quantity the run does not define (no novel state, no wrong decision) is left out so that
    \\resna prints --."""
    k: dict[str, str] = {}

    def put(key: str, val: Any, fmt: Callable[[Any], str] = _p) -> None:
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            k[key] = fmt(val)

    put("drift/n_models", len(res["models"]), str)
    put("drift/c/questions", res["corpus"]["questions"], _tc)
    put("drift/c/occurrences", res["corpus"]["occurrences"], _tc)
    put("drift/c/ask/occurrences", res["corpus"].get("ask_occurrences"), _tc)
    put("drift/c/ask/tasks", res["corpus"].get("ask_tasks"), str)
    put("drift/c/ask/share", res["corpus"].get("ask_share"))
    put("drift/c/mustask/occurrences", res["corpus"].get("mustask_occurrences"), _tc)
    put("drift/c/mustask/tasks", res["corpus"].get("mustask_tasks"), str)
    put("drift/c/mustask/share", res["corpus"].get("mustask_share"))
    put("drift/jev_tau", res.get("jev_tau"), lambda x: _tau(x, tex=True))
    for name, v in (res.get("hash_check") or {}).items():
        put(f"drift/match/{name}", v["match"])
        put(f"drift/match/{name}/n", v["labelled_states"], _tc)
    for name, v in (res.get("kendall") or {}).items():
        if isinstance(v, float):
            put(f"drift/kendall/{name}", v, _f)
    pr = res.get("pred") or {}
    for m, v in (pr.get("rows") or {}).items():
        for sc in ("pred", "pred_llm", "pred_onplan", "pred_earlyask"):
            put(f"drift/pred/{m}/{sc}", v.get(sc))
            put(f"drift/pred/{m}/{sc}/err", v.get(f"{sc}_err"), lambda x: f"{100 * x:+.1f}")
    for name in ("all", "offtheshelf"):
        put(f"drift/pred/kendall/{name}/n", (pr.get("n") or {}).get(name), str)
        for sc, v in ((pr.get("kendall") or {}).get(name) or {}).items():
            put(f"drift/pred/kendall/{name}/{sc}", v, _f)
        for sc, v in ((pr.get("mae") or {}).get(name) or {}).items():
            put(f"drift/pred/mae/{name}/{sc}", v)
    for m, r in res["models"].items():
        s, a, f, g, fa, ask = r["states"], r["acc"], r["first"], r["gate"], r["faith"], r["ask"]
        put(f"drift/{m}/live/success", r["success"])
        put(f"drift/{m}/live/n", r["tasks"], str)
        put(f"drift/{m}/c/acc", r["c_acc"])
        put(f"drift/{m}/c/ask_acc", ask["c_acc"])
        put(f"drift/{m}/states/n", s["labelled"], _tc)
        put(f"drift/{m}/states/inc_share", s["in_c_share"])
        put(f"drift/{m}/states/novel_share", s["novel_share"])
        put(f"drift/{m}/states/novel_onplan_share", s["novel_on_plan_share"])
        put(f"drift/{m}/states/novel_offplan_share", s["novel_after_off_plan_share"])
        put(f"drift/{m}/states/offplan_share", s["after_off_plan_share"])
        for key, blk in a.items():
            put(f"drift/{m}/acc/{key}", blk["acc"])
            put(f"drift/{m}/acc/{key}/n", blk["n"], _tc)
        put(f"drift/{m}/ask/n", ask["n"], str)
        put(f"drift/{m}/ask/acc", ask["acc"])
        put(f"drift/{m}/ask/acted", ask["acted_share"])
        put(f"drift/{m}/ask/acted/n", ask["acted"], str)
        put(f"drift/{m}/ask/acted_of_wrong", ask["acted_share_of_wrong"])
        mu, ea = r["mustask"], r["askearly"]
        put(f"drift/{m}/mustask/n", mu["n"], str)
        put(f"drift/{m}/mustask/tasks", mu["tasks"], str)
        put(f"drift/{m}/mustask/acc", mu["acc"])
        put(f"drift/{m}/mustask/acted", mu["acted_share"])
        put(f"drift/{m}/mustask/acted/n", mu["acted"], str)
        put(f"drift/{m}/mustask/step_median", mu["step_median"], lambda x: f"{x:g}")
        put(f"drift/{m}/c/mustask_acc", mu["c_acc"])
        put(f"drift/{m}/askearly/tasks", ea["tasks"], str)
        put(f"drift/{m}/askearly/share", ea["tasks_share"])
        put(f"drift/{m}/faith/live", fa["live_acc"])
        put(f"drift/{m}/faith/offline", fa["offline_acc"])
        put(f"drift/{m}/faith/agree", fa["agree"])
        put(f"drift/{m}/faith/n", fa["n"], _tc)
        put(f"drift/{m}/faith/lw_or", fa["live_wrong_offline_right"], str)
        put(f"drift/{m}/faith/lr_ow", fa["live_right_offline_wrong"], str)
        put(f"drift/{m}/faith/sign_p", fa["sign_p"], lambda x: _f(x, 3))
        for key in ("in_c", "novel", "none"):
            put(f"drift/{m}/first/{key}", f[f"{key}_share"])
            put(f"drift/{m}/first/{key}/n", f[key], str)
        put(f"drift/{m}/first/with_wrong/n", f["with_wrong_choice"], str)
        put(f"drift/{m}/first/failed", f["failed"], str)
        put(f"drift/{m}/first/failed_with_wrong", f["failed_with_wrong_choice_share"])
        put(f"drift/{m}/first/recovered", f["recovered_share"])
        put(f"drift/{m}/first/recovered/n", f["recovered"], str)
        put(f"drift/{m}/first/step_median", f["step_median"], lambda x: f"{x:g}")
        put(f"drift/{m}/first/step_le1", f["step_le1_share"])
        put(f"drift/{m}/first/inc_offline_right", f["in_c_offline_right_share"])
        put(f"drift/{m}/first/inc_offline_right/n", f["in_c_offline_right"], str)
        put(f"drift/{m}/first/inc_same_answer/n", f["in_c_same_answer"], str)
        put(f"drift/{m}/first/irreversible", f["irreversible_share"])
        put(f"drift/{m}/first/irreversible/n", f["irreversible"], str)
        put(f"drift/{m}/first/irreversible_of_failed", f["irreversible_of_failed_share"])
        for name, v in f["loss"].items():
            put(f"drift/{m}/loss/{name}", v)
        for name, n in f["types"].items():
            slug = TYPE_SLUG.get(name, name.replace(" ", "_"))
            put(f"drift/{m}/first/type/{slug}/n", n, str)
            put(f"drift/{m}/first/type/{slug}/share", _share(n, f["with_wrong_choice"]))
            put(f"drift/{m}/first/type/{slug}/tasks", _share(n, f["tasks"]))
        for name, n in f["consequence"].items():
            slug = CONSEQ_SLUG.get(name, name.replace(" ", "_"))
            put(f"drift/{m}/first/conseq/{slug}/n", n, str)
            put(f"drift/{m}/first/conseq/{slug}/share", _share(n, f["with_wrong_choice"]))
            put(f"drift/{m}/first/conseq/{slug}/tasks", _share(n, f["tasks"]))
            put(f"drift/{m}/first/conseq/{slug}/recovered", f["recovered_by_consequence"].get(name))
        if f["types"]:
            top = next(iter(f["types"].items()))
            put(f"drift/{m}/first/top_type", top[0], str)
            put(f"drift/{m}/first/top_type/share", _share(top[1], f["with_wrong_choice"]))
        put(f"drift/{m}/gate/tau", r["tau_used"], lambda x: _tau(x, tex=True))
        put(f"drift/{m}/gate/source", "own" if r["own_tau"] is not None else "jev", str)
        put(f"drift/{m}/conf/right", g["conf_right_mean"])
        put(f"drift/{m}/conf/wrong", g["conf_wrong_mean"])
        put(f"drift/{m}/gate/n_wrong", g["n_wrong"], str)
        used = g["at"].get("own") or g["at"].get("jev")
        if used:
            put(f"drift/{m}/gate/wrong_below", used["wrong_below"])
            put(f"drift/{m}/gate/right_below", used["right_below"])
            put(f"drift/{m}/gate/failed_first_below", used["failed_first_below"])
        for name, blk in g["at"].items():
            tag = name.replace("fixed_0.", "0")
            put(f"drift/{m}/gate/{tag}/wrong_below", blk["wrong_below"])
            put(f"drift/{m}/gate/{tag}/right_below", blk["right_below"])
            put(f"drift/{m}/gate/{tag}/failed_first_below", blk["failed_first_below"])
    return k


def to_tex(res: Mapping[str, Any]) -> str:
    head = [r"% AUTO-GENERATED by scripts/explore_drift.py -- do not edit by hand.",
            f"% generated {res['generated']} at git {res['git']}; {res['label']}",
            r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{{key}}}{{{val}}}" for key, val in sorted(tex_rows(res).items())]
    return "\n".join(head + body + [r"\makeatother", ""])


def to_md(res: Mapping[str, Any]) -> str:
    M = res["models"]
    order = [m for m in MODELS if m in M] + [m for m in M if m not in MODELS]
    L = ["# Why a decision model can match Jev on the common states yet lose live (drift analysis)", "",
         f"Generated {res['generated']} at git `{res['git']}` by `python scripts/explore_drift.py summarize`. {res['label']}. "
         "Live = REPLACE at the tool choice, Gemini 3.5 Flash, the first 200 fresh Stage-2 DecisionGym tasks (seeds "
         "300000-300199), latest record per task, errored tasks left out. C = the common-state corpus "
         f"`results/stage2/s23` ({res['corpus']['questions']:,} distinct questions, {res['corpus']['occurrences']:,} "
         "occurrences, 200 tasks; recorded from the LLM-only, Jev, Intern-Decision-4B and AgentDecision-4B live arms). "
         "A live state is *in C* when its corpus hash (`bench/stage2_corpus.question_hash`: the exact System-1 state, "
         "option list, option descriptions and question text) is a C question, else *novel*; novel states are split "
         "into *on plan* (every earlier executed tool was acceptable) and *after an off-plan step*. Correctness is "
         "the benchmark's rule (membership in the logged acceptable tool set). State accuracies are pooled over "
         "labelled decisions; the C column is the published task mean. Paper keys: `paper/generated/drift.tex` "
         "(`\\res{drift/...}`).", ""]
    if res.get("s3"):
        L += ["The two E15 GPU live logs are read from S3 copies (`fetch`): " + "; ".join(f"{k}: `{v}`" for k, v in res["s3"].items()) + ".", ""]
    fewer = {m: r["tasks"] for m, r in M.items() if r["tasks"] != max(r["tasks"] for r in M.values())}
    if fewer:
        L += ["Rows with fewer live tasks than the others: " + "; ".join(f"{M[m]['label']} {n}" for m, n in fewer.items()) + ".", ""]
    # ---- 1. verification
    L += ["## 1. Verification", "", "The corpus hash recomputed on the C source arms' own logs (their labelled "
          "tool-choice states must be in C by construction):", "",
          "| source arm | labelled states | in C | match % |", "|---|---|---|---|"]
    for name, v in (res.get("hash_check") or {}).items():
        L.append(f"| {name} | {v['labelled_states']:,} | {v['in_c']:,} | {_p(v['match'])} |")
    L += ["", "Published numbers against this script's reproduction (the C accuracy as the task mean over the corpus; "
          "live success over the latest records), the in-C labels against the corpus's, fall-throughs (System-1 "
          "answer not executed) and unlabelled states (excluded, as in C):", "",
          "| model | C acc published | reproduced | live success published | reproduced | in-C label mismatches | fall-throughs | unlabelled |",
          "|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = M[m]
        L.append(f"| {r['label']} | {r['published']['c_acc'] or 'n/a'} | {_p(r['c_acc'])} | {r['published']['live_success'] or 'n/a'} | "
                 f"{_p(r['success'])} | {r['states']['label_mismatch_in_c']} | {r['states']['fallthrough']} | {r['states']['unlabelled']} |")
    # ---- 2. main table
    L += ["", "## 2. Main table (the paper's Table tab:drift)", "",
          "| model | live success % | C offline acc % | live acc, in-C states % | live acc, novel states % | novel-state share % "
          "| first wrong choice: in C / novel / none (% of tasks) | irreversible first error (% of tasks) "
          "| wrong decisions below tau-hat % (tau-hat) | below 0.9: wrong / right % |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = M[m]
        g = r["gate"]["at"].get("own") or r["gate"]["at"].get("jev") or {}
        a9 = r["gate"]["at"].get("fixed_0.9", {})
        f = r["first"]
        L.append(f"| {r['label']} | {_p(r['success'])} | {_p(r['c_acc'])} | {_p(r['acc']['in_c']['acc'])} | {_p(r['acc']['novel']['acc'])} | "
                 f"{_p(r['states']['novel_share'])} | {_p(f['in_c_share'])} / {_p(f['novel_share'])} / {_p(f['none_share'])} | "
                 f"{_p(f['irreversible_share'])} | {_p(g.get('wrong_below'))} ({_tau(r['tau_used'])}, {'own' if r['own_tau'] is not None else 'Jev arm'}) | "
                 f"{_p(a9.get('wrong_below'))} / {_p(a9.get('right_below'))} |")
    # ---- 3. faithfulness
    L += ["", "## 3. (a) Faithfulness: the in-C states each model visited live, live answer against its offline C answer", "",
          "| model | in-C states | live acc % | offline acc on the same states % | same answer % | live wrong, offline right | live right, offline wrong | sign test p | offline answer missing |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        fa = M[m]["faith"]
        L.append(f"| {M[m]['label']} | {fa['n']:,} | {_p(fa['live_acc'])} | {_p(fa['offline_acc'])} | {_p(fa['agree'])} | "
                 f"{fa['live_wrong_offline_right']} | {fa['live_right_offline_wrong']} | {_f(fa['sign_p'], 3)} | {fa['missing_offline']} |")
    # ---- 4. drift
    L += ["", "## 4. (b) Drift: novel states and the first wrong tool choice", "",
          "| model | states | in C % | novel on plan % | novel after off-plan % | acc in C % | acc novel on plan % | acc novel after off-plan % "
          "| acc in C after off-plan % (n) | tasks with a wrong choice | first wrong choice in C / novel | median step (share at step 0-1) "
          "| failed tasks | failed with a wrong choice % | recovered after a wrong choice % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = M[m]
        s, a, f = r["states"], r["acc"], r["first"]
        L.append(f"| {r['label']} | {s['labelled']:,} | {_p(s['in_c_share'])} | {_p(s['novel_on_plan_share'])} | {_p(s['novel_after_off_plan_share'])} | "
                 f"{_p(a['in_c']['acc'])} | {_p(a['novel_on_plan']['acc'])} | {_p(a['novel_after_off_plan']['acc'])} | "
                 f"{_p(a['in_c_after_off_plan']['acc'])} ({a['in_c_after_off_plan']['n']}) | {f['with_wrong_choice']} | {f['in_c']} / {f['novel']} | "
                 f"{'n/a' if f['step_median'] is None else f['step_median']} ({_p(f['step_le1_share'])}) | {f['failed']} | "
                 f"{_p(f['failed_with_wrong_choice_share'])} | {_p(f['recovered_share'])} |")
    L += ["", "First wrong tool choices on an in-C state: what the model answered offline on that very state, and the error types "
          "(the difficulty analysis's classifier):", "",
          "| model | first errors in C | offline right on that state | offline wrong | offline gave the same wrong tool | error types of all first errors |",
          "|---|---|---|---|---|---|"]
    for m in order:
        f = M[m]["first"]
        types = "; ".join(f"{a} {b}" for a, b in f["types"].items())
        L.append(f"| {M[m]['label']} | {f['in_c']} | {f['in_c_offline_right']} | {f['in_c_offline_wrong']} | {f['in_c_same_answer']} | {types} |")
    L += ["", "The ask-or-act states (the answer key is `ask_user`: "
          f"{res['corpus'].get('ask_occurrences', 0):,} of C's occurrences, {_p(res['corpus'].get('ask_share'))} %, in "
          f"{res['corpus'].get('ask_tasks', 0)} tasks) and what each model did there live; the *must-ask* states among "
          f"them (the acceptable set is exactly `ask_user`: {res['corpus'].get('mustask_occurrences', 0):,} occurrences, "
          f"{_p(res['corpus'].get('mustask_share'))} % of C, {res['corpus'].get('mustask_tasks', 0)} tasks), how many a model "
          "met live (exposure: a model that asks earlier, where asking is one acceptable tool among others, never reaches "
          "the later must-ask state) and in how many tasks it asked early that way; the consequence of the first "
          "wrong tool choice (a wrong write action or a premature finish ends the task; a wrong lookup, an unneeded "
          "question or a read-only look-alike can be recovered from), with the share of such tasks that still succeeded:", "",
          "| model | C ask-or-act acc % (task mean) | live ask-or-act states (in C) | live acc % | acted instead of asking (% of ask states) "
          "| must-ask states: C acc % (task mean) | live n (tasks) | live acc % | asked early (tasks) "
          "| first errors: wrong write / premature finish / recoverable | recovered % by kind (write / finish / recoverable) "
          "| irreversible first error: % of tasks, % of failed tasks | loss (% of tasks): at an irreversible first error / after a recoverable one / no wrong choice |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = M[m]
        ask, f = r["ask"], r["first"]
        c = f["consequence"]
        rb = f["recovered_by_consequence"]
        mu, ea = r["mustask"], r["askearly"]
        L.append(f"| {r['label']} | {_p(ask['c_acc'])} | {ask['n']} ({ask['in_c']}) | {_p(ask['acc'])} | {_p(ask['acted_share'])} | "
                 f"{_p(mu['c_acc'])} | {mu['n']} ({mu['tasks']}) | {_p(mu['acc'])} | {ea['tasks']} | "
                 f"{c.get('wrong write action', 0)} / {c.get('premature finish', 0)} / {c.get('recoverable', 0)} | "
                 f"{_p(rb.get('wrong write action'))} / {_p(rb.get('premature finish'))} / {_p(rb.get('recoverable'))} | "
                 f"{_p(f['irreversible_share'])}, {_p(f['irreversible_of_failed_share'])} | "
                 f"{_p(f['loss']['irreversible'])} / {_p(f['loss']['compounding'])} / {_p(f['loss']['no_wrong_choice'])} |")
    if res.get("kendall"):
        kd = res["kendall"]
        L += ["", f"Across the {kd['n_models']} models, Kendall's tau-b of live success with the novel-state share: "
              f"{_f(kd['novel_share_vs_live'])}; with the share of states after an off-plan step and outside C: "
              f"{_f(kd['novel_after_off_plan_share_vs_live'])}; with the novel on-plan share: {_f(kd['novel_on_plan_share_vs_live'])}."]
    # ---- 5. gate
    L += ["", "## 5. (c) Gate recovery: wrong and right live decisions below a confidence threshold", "",
          "tau-hat = the model's own fitted tool-choice threshold (results/calibration, results/gpu_round5/cal, "
          f"results/explore/e14_small/cal) where one exists, else the Jev arm's ({_tau(res.get('jev_tau'))}); a cascade "
          "accepts System-1's answer at confidence >= tau and otherwise asks the LLM, so a decision *below* tau is handed "
          "back. 'Failed tasks' = share of failed tasks whose first wrong tool choice was below tau (an upper bound on "
          "what a gate hands back; whether the LLM then chooses right is not measured here).", "",
          "| model | wrong decisions | mean conf right / wrong % | tau-hat (source) | wrong below tau-hat % | right below tau-hat % | failed tasks' first error below tau-hat % "
          "| at 0.5: wrong / right / failed % | at 0.9: wrong / right / failed % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = M[m]
        g = r["gate"]
        used = g["at"].get("own") or g["at"].get("jev") or {}
        a5, a9 = g["at"].get("fixed_0.5", {}), g["at"].get("fixed_0.9", {})
        L.append(f"| {r['label']} | {g['n_wrong']} | {_p(g['conf_right_mean'])} / {_p(g['conf_wrong_mean'])} | {_tau(r['tau_used'])} "
                 f"({'own' if r['own_tau'] is not None else 'Jev arm'}) | {_p(used.get('wrong_below'))} | {_p(used.get('right_below'))} | "
                 f"{_p(used.get('failed_first_below'))} | {_p(a5.get('wrong_below'))} / {_p(a5.get('right_below'))} / {_p(a5.get('failed_first_below'))} | "
                 f"{_p(a9.get('wrong_below'))} / {_p(a9.get('right_below'))} / {_p(a9.get('failed_first_below'))} |")
    # ---- 7. an offline score that counts consequences
    pr = res.get("pred") or {}
    if pr.get("rows"):
        L += ["", "## 7. An offline score that counts consequences: predicting live success from the C answers alone", "",
              "No live log enters this score. For each task, each reference trajectory recorded in C (the LLM-only, Jev, "
              "Intern-Decision-4B and AgentDecision-4B source arms) is walked in step order with the model's offline C "
              "answers; the first state where the answer is wrong decides: the task is predicted lost iff that error is "
              "irreversible (a wrong write action or a premature finish), else predicted solved; the task's prediction is "
              "the mean over its available trajectories, the model's over its live tasks. Variants: the LLM-only "
              "trajectory alone; the walk stopped at the reference's own first off-plan step; an acceptable early ask "
              "(asking where asking is acceptable but not canonical) ends the walk as a success.", "",
              "| model | C mean % | must-ask acc % | predicted live % (4 sources) | LLM-only traj. | on-plan prefix | early ask resolves | actual live % | error (4 sources) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in order:
            v = pr["rows"].get(m)
            if v is None:
                continue
            err = "n/a" if v["pred_err"] is None else f"{100 * v['pred_err']:+.1f}"
            L.append(f"| {M[m]['label']} | {_p(v['cmean'])} | {_p(v['mustask'])} | {_p(v['pred'])} | {_p(v['pred_llm'])} | "
                     f"{_p(v['pred_onplan'])} | {_p(v['pred_earlyask'])} | {_p(v['actual'])} | {err} |")
        L += ["", "| row set (n) | Kendall tau-b with live: C mean | must-ask acc | predicted (4 sources) | LLM-only traj. | on-plan prefix | early ask resolves "
              "| MAE points: predicted / LLM-only / on-plan / early ask |", "|---|---|---|---|---|---|---|---|"]
        for name in ("all", "offtheshelf"):
            kd, mae = pr["kendall"][name], pr["mae"][name]
            L.append(f"| {name} ({pr['n'][name]}) | {_f(kd['cmean'])} | {_f(kd['mustask'])} | {_f(kd['pred'])} | {_f(kd['pred_llm'])} | "
                     f"{_f(kd['pred_onplan'])} | {_f(kd['pred_earlyask'])} | {_p(mae['pred'])} / {_p(mae['pred_llm'])} / "
                     f"{_p(mae['pred_onplan'])} / {_p(mae['pred_earlyask'])} |")
        L += ["", "Off-the-shelf rows = E15's set (every row except the paper's trained models and the attribution rows): "
              + ", ".join(pr["sets"]["offtheshelf"]) + "."]
    # ---- 6. inputs, not found, excluded
    L += ["", "## 6. Inputs, rows not found and runs not used", "", "| model | live log (sha256 prefix) | offline C answers |", "|---|---|---|"]
    for m in order:
        r = M[m]
        L.append(f"| {r['label']} | `{r['live_path']}` ({r['live_sha256']}) | `{r['answers_path'] or 'none'}` |")
    if res.get("not_found"):
        L += ["", "Not found: " + "; ".join(f"{k}: {v}" for k, v in res["not_found"].items()) + "."]
    L += ["", "Not rows: " + "; ".join(f"{k}: {v}" for k, v in res["excluded"].items()) + "."]
    return "\n".join(L) + "\n"


# ================================================================== CLI
def cmd_fetch(a: argparse.Namespace) -> int:
    """Download the two E15 GPU live logs from S3 into DEST/<run>/<file> (explicit; the analysis never downloads)."""
    import os

    dest = Path(a.dest)
    env = {k: v for k, v in os.environ.items() if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
    env["AWS_PAGER"] = ""
    for key, rel in S3_FILES.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not a.force:
            print(f"{key}: {target} exists (use --force)")
            continue
        cmd = ["aws", "--region", a.region, "s3", "cp", f"{S3_BUCKET}/{rel}", str(target)]
        print(" ".join(cmd))
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            return r.returncode
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    e15 = Path(a.e15_dir) if a.e15_dir else None
    res = build(e15)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # the per-task first wrong choices (model, task, step, in C, confidence, type, consequence, offline verdict) go to
    # a compact gzipped JSONL beside the summary
    with gzip.open(out / "first_errors.jsonl.gz", "wt") as f:
        for m, r in res["models"].items():
            for row in r["first"].pop("per_task"):
                f.write(json.dumps({"model": m, **row}) + "\n")
    (out / "drift.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "drift.md").write_text(to_md(res))
    Path(a.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_tex).write_text(to_tex(res))
    print(json.dumps({m: {"live": round(r["success"], 3), "c": round(r["c_acc"] or 0, 3), "in_c": round(r["acc"]["in_c"]["acc"] or 0, 3),
                          "novel": round(r["acc"]["novel"]["acc"] or 0, 3), "novel_share": round(r["states"]["novel_share"] or 0, 3)}
                      for m, r in res["models"].items()}, indent=1))
    if res["not_found"]:
        print("not found:", json.dumps(res["not_found"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the two E15 GPU live logs from S3 (explicit)")
    f.add_argument("--dest", required=True, help="directory that receives <run>/<file> (keep it outside the repo)")
    f.add_argument("--region", default="us-east-1")
    f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("summarize", help="read the stored logs and write the readout")
    s.add_argument("--e15-dir", default=None, help="the directory `fetch` wrote (the two E15 rows are skipped without it)")
    s.add_argument("--out-dir", default=str(OUT_DIR))
    s.add_argument("--out-tex", default=str(OUT_TEX))
    s.set_defaults(fn=cmd_summarize)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
