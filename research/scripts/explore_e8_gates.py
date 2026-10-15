#!/usr/bin/env python3
"""E8 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "Exploratory row E8"): what decides
whether the decision model's answer is accepted. An offline comparison of acceptance gates on 1,000 decisions of the
registered calibration shadow arm (JevGym seeds 5000-5149, Gemini 3.5 Flash as the agent, Jev answering in shadow at
every state; tool choice and tool arguments).

    python scripts/explore_e8_gates.py sample                         # results/explore/e8_gates/sample.json
    python scripts/explore_e8_gates.py run --gate g3|g4 [--limit 10] [--workers 8] [--dry-run]
    python scripts/explore_e8_gates.py project                        # the registered smoke projection vs the cap
    python scripts/explore_e8_gates.py score                          # summary.{md,json} + paper/generated/e8_gates.tex

Gates (each a score for "accept the decision model's answer"):
  G1 the model's own confidence field (the registered gate; free);  G2 the margin of its top two options (free);
  G3 a second decision-model call: a yes / no question on the same state whether the proposal is the right next step,
     score P(yes) (one Jev call per decision);
  G4 an LLM verifier: Gemini 3.5 Flash (direct API, temperature 0, minimal reasoning) sees the state the model saw and
     the proposal and answers a verdict and a 0-100 confidence that the proposal is right (one call per decision);
  G5 agreement with the LLM's logged choice (no new call here; deployed, it needs the agent's own LLM call);
  G1>G3, G1>G4 two-stage: G1 accepts at the registered tau-hat and the second gate is asked only below it.

Economics (the paper's rule, Sec. 2.1 / App. A): accepting the model's answer at a decision saves the LLM's cost of
that decision dC and changes the expected loss by V (q - g) (q, g = the LLM's and the model's correctness). Per
decision, the expected saving against the LLM alone is sum over accepted decisions of [dC - V (q - g)] / n minus the
model's own call kappa and the gate's own call; it is maximised over the gate's thresholds on the same decisions (an
in-sample upper bound). dC and kappa are the logged per-decision costs as in the registered fit
(``threshold_theory.cost_terms``: argument decisions share their step's LLM call); the gate's cost and latency come
from E8's own ledger. No prompt holds the answer key: the builders take the state, the options and the proposal only
(tests/test_explore_e8.py)."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import random
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import explore_common as C  # noqa: E402
from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.threshold_theory import decision_pairs  # noqa: E402

SRC = ROOT / "runs/e2a_jevgym_shadow_cal/jev_shadow_cal.jsonl"
SRC_SHA256 = "fff21516e0aad24334066232225144f4406da1448e8ceb0326be390a9a303fd8"
SEED_RANGE = (5000, 5149)                     # calibration seeds; outside every evaluation block
EVAL_BLOCKS = ((100000, 100499), (200000, 200499), (300000, 300999), (310000, 310599))
OUT = ROOT / "results/explore/e8_gates"
TEX = ROOT / "paper/generated/e8_gates.tex"
DPS = ("tool_select", "tool_params")
DP_KEY = {"tool_select": "toolselect", "tool_params": "toolparams"}
DP_LABEL = {"tool_select": "tool choice", "tool_params": "tool arguments"}
N_PER_DP, SEED = 500, 0
CAP_USD, SMOKE_N, PROJ_FACTOR = 40.0, 10, 1.3
TAU_HAT = {"tool_select": 1.0, "tool_params": 0.99}   # the registered fit (runs/cal/tauhat_jev_shadow_cal_*.json)
V_PRIMARY, V_GRID = 1.0, (0.1, 1.0, 10.0)
COVERAGES = (0.25, 0.5, 0.75, 0.9)
N_BOOT, BOOT_SEED = 2000, 0
LLM_SPEC = {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0, "reasoning_effort": "minimal"}
LLM_MAX_TOKENS = 64
GATES = ("g1", "g1tau", "g2", "g3", "g4", "g4v", "g5", "g1g3", "g1g4", "g1g4v")
# g4v / g1g4v: post-hoc rows (Deviations 2026-10-07 "E8: the verifier's verdict as its score", analysis only, after
# the replies were read): every G4 reply gave CONFIDENCE 100 whatever its verdict, so the registered score is constant
GATE_LABEL = {"g1": "G1 model's confidence (registered)",
              "g1tau": "G1 at the registered tau-hat (fixed, the deployed rule)", "g2": "G2 top-two margin",
              "g3": "G3 second decision-model call (yes/no)", "g4": "G4 LLM verifier (Flash, yes/no + 0-100)",
              "g4v": "G4v LLM verifier, its verdict as the score (post hoc)",
              "g5": "G5 agreement with the LLM's own choice", "g1g3": "G1, then G3 below tau-hat",
              "g1g4": "G1, then G4 below tau-hat",
              "g1g4v": "G1, then G4v below tau-hat (post hoc)", "perfect": "answer key (reference, not a gate)"}
GATE_CALL = {"g1": "none", "g1tau": "none", "g2": "none", "g3": "1 decision-model call", "g4": "1 LLM call", "g4v": "1 LLM call",
             "g5": "the agent's own LLM call", "g1g3": "1 decision-model call below tau-hat",
             "g1g4": "1 LLM call below tau-hat", "g1g4v": "1 LLM call below tau-hat",
             "perfect": "none"}


# ================================================================================================ data
def _eligible(r: Mapping[str, Any]) -> bool:
    """Paired (model and LLM both judged) and executable (not OTHER): the decisions a gate can accept."""
    return r.get("c") is not None and r.get("s1_ok") is not None and r.get("s2_ok") is not None \
        and bool(r.get("acceptable"))


def task_seed(task_id: str) -> int:
    m = re.search(r"-(\d+)-d\d+", str(task_id))
    if not m:
        raise ValueError(f"no seed in task id {task_id!r}")
    return int(m.group(1))


def load_decisions(path: Path = SRC) -> list[dict[str, Any]]:
    """Every tool-choice and argument decision of the shadow log, with the registered fit's correctness and cost
    fields (``threshold_theory.decision_pairs``) next to the raw log entry's state, options, question and answers.
    ``dc`` / ``dl``: the LLM cost and latency the decision would avoid, split over the step's argument decisions for
    tool_params exactly as ``cost_terms(couple_by_step=True)`` splits them."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        for dp in DPS:
            raws = [(st.get("step"), r) for st in t.get("steps") or [] for r in st.get("decisions") or []
                    if r.get("dp") == dp and not (str(r.get("mode")) == "guard" and dp != "stop")]
            pairs = decision_pairs([t], dp)
            if len(pairs) != len(raws):
                raise RuntimeError(f"{t.get('task_id')}: {len(pairs)} pairs vs {len(raws)} log entries at {dp}")
            share: dict[Any, int] = defaultdict(int)
            if dp == "tool_params":
                for p in pairs:
                    if p.get("s2_ok") is not None:
                        share[p["step"]] += 1
            seen: dict[Any, int] = defaultdict(int)
            for (step, raw), p in zip(raws, pairs):
                j = seen[step]
                seen[step] += 1
                k = max(1, share.get(step, 1)) if dp == "tool_params" else 1
                out.append({"id": f"{t['task_id']}|{step}|{dp}|{j}", "dp": dp, "task_id": str(t["task_id"]),
                            "step": step, "state": raw.get("state") or "", "options": raw.get("options") or [],
                            "question": raw.get("question") or "", "s1_value": raw.get("s1_value"),
                            "s1_conf": p["c"], "s1_dist": raw.get("s1_dist") or {}, "s2_value": raw.get("s2_value"),
                            "c": p["c"], "s1_ok": p["s1_ok"], "s2_ok": p["s2_ok"], "acceptable": p["acceptable"],
                            "dc": p["s2_cost"] / k, "dl": p["s2_lat_ms"] / k, "k_usd": p["s1_cost"],
                            "k_ms": p["s1_lat_ms"]})
    return out


def draw_sample(recs: Sequence[Mapping[str, Any]], n_per_dp: int = N_PER_DP, seed: int = SEED) -> dict[str, Any]:
    """The registered sample: per decision point ``n_per_dp`` eligible decisions, stratified in proportion by whether
    the model's answer is right (round(n x wrong share) wrong ones), drawn with ``random.Random("e8:<dp>:<seed>")``
    from the id-sorted pool; the call order is the union shuffled with ``random.Random("e8:order:<seed>")``."""
    ids: list[str] = []
    strata: dict[str, Any] = {}
    for dp in DPS:
        pool = sorted((r for r in recs if r["dp"] == dp and _eligible(r)), key=lambda r: r["id"])
        wrong = [r["id"] for r in pool if not r["s1_ok"]]
        right = [r["id"] for r in pool if r["s1_ok"]]
        n = min(n_per_dp, len(pool))
        k = int(round(n * len(wrong) / len(pool))) if pool else 0
        rng = random.Random(f"e8:{dp}:{seed}")
        pick = rng.sample(wrong, min(k, len(wrong))) + rng.sample(right, min(n - k, len(right)))
        ids += pick
        strata[dp] = {"pool": len(pool), "pool_wrong": len(wrong), "n": len(pick), "wrong": min(k, len(wrong)),
                      "right": len(pick) - min(k, len(wrong))}
    order = sorted(ids)
    random.Random(f"e8:order:{seed}").shuffle(order)
    return {"ids": order, "strata": strata, "seed": seed, "n_per_dp": n_per_dp}


# ================================================================================================ prompts
_PQ = re.compile(r"^Which value should the argument `(?P<param>[^`]+)` of tool `(?P<tool>[^`]+)` take\?"
                 r"(?: \((?P<desc>.*)\))?\s*$", re.S)
VERDICT_FORMAT = ("Reply with exactly two lines:\nVERDICT: yes or no\nCONFIDENCE: an integer from 0 to 100, your "
                  "probability that the proposal is right")


def tool_descriptions() -> dict[str, str]:
    """JevGym's tools (core, ask_user, the 60 near distractors, finish) -> their one-line descriptions, as the
    decision model's tool-choice question shows them (``tools.tool_desc``)."""
    from jevharness.bench.suites import jevgym as J
    from jevharness.harness.tools import finish_tool, tool_desc, tool_name

    ts = J.CORE_TOOLS + [J.ASK_TOOL] + J.near_distractor_tools(60) + [finish_tool()]
    return {tool_name(t): tool_desc(t) for t in ts}


def arg_parts(question: str) -> tuple[str, str, str]:
    """(tool, argument, its description) of a logged argument question."""
    m = _PQ.match(question or "")
    if not m:
        raise ValueError(f"not an argument question: {question[:120]!r}")
    return m.group("tool"), m.group("param"), (m.group("desc") or "").strip()


def g3_question(dp: str, question: str, value: Any, descs: Mapping[str, str]) -> str:
    """The yes / no question the decision model answers about its own proposal (the harness's tool-guard wording)."""
    if dp == "tool_select":
        d = descs.get(str(value), "")
        return (f"The agent proposes to call `{value}` next{f' ({d})' if d else ''}. "
                "Is this the right next step toward completing the task?")
    tool, param, desc = arg_parts(question)
    return (f"The agent is about to call `{tool}`. It proposes the value `{value}` for the argument "
            f"`{param}`{f' ({desc})' if desc else ''}. Is this the right value for this argument?")


def g4_prompt(dp: str, state: str, options: Sequence[str], question: str, value: Any,
              descs: Mapping[str, str]) -> str:
    """The LLM verifier's single user message: the state the decision model saw, (tool choice) the tools it chose
    from with their descriptions, the proposal and the reply format. Nothing else."""
    if dp == "tool_select":
        tools = "\n".join(f"- {o}: {descs[o]}" if descs.get(o) else f"- {o}" for o in options)
        return (f"{state}\n\nAVAILABLE TOOLS:\n{tools}\n\nA fast decision model proposes that the agent call "
                f"`{value}` next. Is this the right next step toward completing the task?\n\n{VERDICT_FORMAT}")
    tool, param, desc = arg_parts(question)
    return (f"{state}\n\nThe agent is about to call `{tool}`. A fast decision model proposes the value `{value}` for "
            f"the argument `{param}`{f' ({desc})' if desc else ''}. Is this the right value for this argument?\n\n"
            f"{VERDICT_FORMAT}")


_VERDICT = re.compile(r"VERDICT\s*[:=]?\s*\**\s*(yes|no)\b", re.I)
_CONF = re.compile(r"CONFIDENCE\s*[:=]?\s*\**\s*(\d+(?:\.\d+)?)", re.I)


def parse_g4(text: str | None) -> tuple[float | None, str | None, float | None]:
    """(score, verdict, confidence): score = confidence / 100 clipped to [0, 1]; the verdict (1 / 0) when no number
    came back; None when neither did."""
    t = text or ""
    v = _VERDICT.search(t)
    c = _CONF.search(t)
    verdict = v.group(1).lower() if v else None
    if not verdict:
        head = t.strip().lower()
        verdict = "yes" if head.startswith("yes") else "no" if head.startswith("no") else None
    conf = float(c.group(1)) if c else None
    if conf is not None:
        return min(max(conf / 100.0, 0.0), 1.0), verdict, conf
    if verdict is not None:
        return (1.0 if verdict == "yes" else 0.0), verdict, None
    return None, None, None


def top2_margin(dist: Mapping[str, float]) -> float:
    ps = sorted((float(v) for v in (dist or {}).values()), reverse=True)
    return (ps[0] - (ps[1] if len(ps) > 1 else 0.0)) if ps else float("nan")


# ================================================================================================ metrics
def net_saving(score: Sequence[float], g: Sequence[float], q: Sequence[float], dc: Sequence[float],
               kappa: Sequence[float], gate_cost: Sequence[float], V: float, *, consult: Sequence[bool] | None = None,
               pre: Sequence[bool] | None = None, saves: bool = True) -> dict[str, Any]:
    """The best expected saving per decision against the LLM alone over every threshold of a gate (in-sample).

    Accepting decision i saves ``dc[i]`` (unless ``saves`` is False: the gate needed the LLM's own call) and changes
    the expected loss by V (q_i - g_i). Every decision pays the model's own call ``kappa[i]``; the gate's call
    ``gate_cost[i]`` is paid where the gate is consulted (``consult``, default all). ``pre`` marks decisions accepted
    before the gate (two-stage rows). Thresholds: every distinct score of the consulted decisions (accept iff score >=
    t) and +inf (accept none of them). Returns the best saving (USD per decision), its threshold, the accepted mask and
    the coverage; ties in the saving go to the higher threshold."""
    s = np.asarray(score, float)
    n = len(s)
    g_, q_, dc_ = (np.asarray(x, float) for x in (g, q, dc))
    k_, gc_ = np.asarray(kappa, float), np.asarray(gate_cost, float)
    cons = np.ones(n, bool) if consult is None else np.asarray(consult, bool)
    pre_ = np.zeros(n, bool) if pre is None else np.asarray(pre, bool)
    gain = (dc_ if saves else 0.0) - V * (q_ - g_)
    base = float(gain[pre_].sum() - k_.sum() - gc_[cons].sum())
    cand = np.nonzero(cons & ~pre_)[0]
    order = cand[np.argsort(-s[cand], kind="stable")]
    ss, cum = s[order], np.cumsum(gain[order])
    last = [i for i in range(len(ss)) if i == len(ss) - 1 or ss[i] != ss[i + 1]]
    totals = [base] + [base + float(cum[i]) for i in last]
    thr = [math.inf] + [float(ss[i]) for i in last]
    j = int(np.argmax(totals))
    acc = pre_.copy()
    if j > 0:
        acc[order[: last[j - 1] + 1]] = True
    return {"usd": totals[j] / n if n else float("nan"), "threshold": thr[j], "accepted": acc,
            "coverage": float(acc.mean()) if n else float("nan")}


def latency_change(accepted: Sequence[bool], dl: Sequence[float], k_ms: Sequence[float], gate_ms: Sequence[float],
                   consult: Sequence[bool] | None = None, saves: bool = True, model_on_path: bool = True) -> float:
    """Critical-path time saved per decision (ms; negative = slower): the LLM's time on accepted decisions, minus the
    model's call (sequential before the LLM; ``model_on_path`` False when it runs beside the LLM, as for G5) and the
    gate's call where it is consulted."""
    a = np.asarray(accepted, bool)
    n = len(a)
    cons = np.ones(n, bool) if consult is None else np.asarray(consult, bool)
    saved = float(np.asarray(dl, float)[a].sum()) if saves else 0.0
    model = float(np.asarray(k_ms, float).sum()) if model_on_path else 0.0
    return (saved - model - float(np.asarray(gate_ms, float)[cons].sum())) / n if n else float("nan")


def saving_at(t: float, score: Sequence[float], g: Sequence[float], q: Sequence[float], dc: Sequence[float],
              kappa: Sequence[float], gate_cost: Sequence[float], V: float, *, consult: Sequence[bool] | None = None,
              pre: Sequence[bool] | None = None, saves: bool = True) -> float:
    """Total (not per-decision) saving of the fixed threshold ``t`` (accept the consulted decisions with score >= t),
    accounted as in :func:`net_saving`."""
    s = np.asarray(score, float)
    n = len(s)
    cons = np.ones(n, bool) if consult is None else np.asarray(consult, bool)
    pre_ = np.zeros(n, bool) if pre is None else np.asarray(pre, bool)
    gain = (np.asarray(dc, float) if saves else 0.0) - V * (np.asarray(q, float) - np.asarray(g, float))
    acc = pre_ | (cons & ~pre_ & (s >= t))
    return float(gain[acc].sum() - np.asarray(kappa, float).sum() - np.asarray(gate_cost, float)[cons].sum())


def crossfit_saving(tasks: Sequence[str], score: Sequence[float], g: Sequence[float], q: Sequence[float],
                    dc: Sequence[float], kappa: Sequence[float], gate_cost: Sequence[float], V: float, *,
                    consult: Sequence[bool] | None = None, pre: Sequence[bool] | None = None, saves: bool = True,
                    n_splits: int = 10, seed: int = 0) -> dict[str, Any]:
    """Held-out saving per decision: the tasks are split in two halves (``random.Random(f"e8:cv:{seed + r}")``), the
    threshold is chosen on one half (:func:`net_saving`) and scored on the other, both ways; the mean over
    ``n_splits`` splits (post hoc, descriptive: the in-sample optimum favours gates with many distinct scores)."""
    s, n = np.asarray(score, float), len(score)
    arrs = [np.asarray(x, float) for x in (g, q, dc, kappa, gate_cost)]
    cons = np.ones(n, bool) if consult is None else np.asarray(consult, bool)
    pre_ = np.zeros(n, bool) if pre is None else np.asarray(pre, bool)
    keys = sorted(set(tasks))
    vals = []
    for r in range(n_splits):
        ks = list(keys)
        random.Random(f"e8:cv:{seed + r}").shuffle(ks)
        half = set(ks[: len(ks) // 2])
        a = np.array([t in half for t in tasks])
        tot = 0.0
        for tr, te in ((a, ~a), (~a, a)):
            e = net_saving(s[tr], *(x[tr] for x in arrs), V, consult=cons[tr], pre=pre_[tr], saves=saves)
            tot += saving_at(e["threshold"], s[te], *(x[te] for x in arrs), V, consult=cons[te], pre=pre_[te],
                             saves=saves)
        vals.append(tot / n)
    return {"usd": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "n_splits": n_splits}


def boot_auroc(scores: Mapping[str, np.ndarray], g: np.ndarray, tasks: Sequence[str], ref: str = "g1",
               n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, dict[str, Any]]:
    """Task-level bootstrap (tasks resampled with replacement) of every gate's AUROC and of its paired difference to
    ``ref``: percentile 95 % intervals."""
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tasks):
        by_task[t].append(i)
    keys = sorted(by_task)
    groups = [np.asarray(by_task[k]) for k in keys]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in scores}
    diffs: dict[str, list[float]] = {k: [] for k in scores}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        r = M.auroc(scores[ref][idx], g[idx]) if ref in scores else float("nan")
        for k, s in scores.items():
            a = M.auroc(s[idx], g[idx])
            draws[k].append(a)
            diffs[k].append(a - r)
    out: dict[str, dict[str, Any]] = {}
    for k in scores:
        d, f = np.asarray(draws[k]), np.asarray(diffs[k])
        d, f = d[np.isfinite(d)], f[np.isfinite(f)]
        out[k] = {"ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))] if len(d) else [None, None],
                  "diff": float(M.auroc(scores[k], g) - M.auroc(scores[ref], g)) if ref in scores else None,
                  "diff_ci": [float(np.quantile(f, .025)), float(np.quantile(f, .975))] if len(f) else [None, None],
                  "n_boot": n_boot, "seed": seed}
    return out


# ================================================================================================ calls
_LOCK = threading.Lock()


def calls_path(gate: str) -> Path:
    return OUT / f"calls_{gate}.jsonl"


def read_calls(gate: str) -> list[dict[str, Any]]:
    """Every call record of a gate: the committed ``calls_<gate>.jsonl.gz`` first, then any later appends."""
    import gzip

    p = calls_path(gate)
    z = p.with_name(p.name + ".gz")
    text = (gzip.decompress(z.read_bytes()).decode() if z.exists() else "") + (p.read_text() if p.exists() else "")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def latest_scores(gate: str) -> dict[str, dict[str, Any]]:
    """id -> the latest call record with a score (re-runs append)."""
    out: dict[str, dict[str, Any]] = {}
    for r in read_calls(gate):
        if r.get("score") is not None and not r.get("error"):
            out[r["id"]] = r
    return out


def spend(gates: Sequence[str] = ("g3", "g4")) -> dict[str, float]:
    """Logged spend of E8 (every ledger record of every call attempt that reached the ledger)."""
    s = {g: sum(float(c.get("cost_usd") or 0.0) for r in read_calls(g) for c in r.get("calls") or []) for g in gates}
    s["total"] = sum(s[g] for g in gates)
    return s


def load_sample() -> dict[str, Any]:
    p = OUT / "sample.json"
    if not p.exists():
        raise SystemExit("no sample: run `python scripts/explore_e8_gates.py sample` first")
    return json.loads(p.read_text())


def sample_records(sample: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    sample = sample or load_sample()
    by_id = {r["id"]: r for r in load_decisions()}
    return [by_id[i] for i in sample["ids"]]


def _check_source() -> None:
    sha = C.sha256_file(SRC)
    if sha != SRC_SHA256:
        raise SystemExit(f"source log changed: sha256 {sha} != registered {SRC_SHA256}")


def call_g3(backend: Any, rec: Mapping[str, Any], descs: Mapping[str, str]) -> dict[str, Any]:
    from jevharness.accounting import Ledger
    from jevharness.decision.types import Noul

    led = Ledger()
    q = Noul(g3_question(rec["dp"], rec["question"], rec["s1_value"], descs), decision_point=rec["dp"])
    a = backend.ask(rec["state"], q, ledger=led)
    raw = a.raw or {}
    return {"score": a.p_yes, "wire_noul": raw.get("noul"), "calls": [dataclasses.asdict(r) for r in led.records]}


def call_g4(llm: Any, rec: Mapping[str, Any], descs: Mapping[str, str]) -> dict[str, Any]:
    from jevharness.accounting import Ledger

    led = Ledger()
    msg = g4_prompt(rec["dp"], rec["state"], rec["options"], rec["question"], rec["s1_value"], descs)
    r = llm.chat([{"role": "user", "content": msg}], ledger=led, decision_point=rec["dp"], component="s2",
                 max_tokens=LLM_MAX_TOKENS)
    score, verdict, conf = parse_g4(r.text)
    return {"score": score, "verdict": verdict, "conf": conf, "text": (r.text or "")[:200],
            "finish_reason": r.finish_reason, "calls": [dataclasses.asdict(x) for x in led.records]}


def cmd_sample(a: argparse.Namespace) -> int:
    _check_source()
    recs = load_decisions()
    seeds = sorted({task_seed(r["task_id"]) for r in recs})
    if seeds[0] < SEED_RANGE[0] or seeds[-1] > SEED_RANGE[1] or any(
            lo <= s <= hi for s in seeds for lo, hi in EVAL_BLOCKS):
        raise SystemExit(f"seeds {seeds[0]}-{seeds[-1]} outside the calibration range or inside an evaluation block")
    smp = draw_sample(recs)
    smp.update({"source": C.rel(SRC), "sha256": SRC_SHA256, "seed_range": [seeds[0], seeds[-1]],
                "n_tasks_source": len(seeds), "n_tasks_sample": len({i.split("|")[0] for i in smp["ids"]}),
                "rng": "random.Random('e8:<dp>:0') per point, random.Random('e8:order:0') for the call order",
                "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps(smp, indent=1) + "\n")
    print(json.dumps({k: v for k, v in smp.items() if k != "ids"}, indent=1))
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    _check_source()
    recs = sample_records()
    if a.limit:
        recs = recs[: a.limit]
    done = latest_scores(a.gate)
    todo = [r for r in recs if r["id"] not in done]
    descs = tool_descriptions()
    print(f"{a.gate}: {len(recs)} decisions in scope, {len(done)} scored already, {len(todo)} to call")
    if a.dry_run:
        for dp in DPS:
            r = next((x for x in todo if x["dp"] == dp), None)
            if r is None:
                continue
            if a.gate == "g3":
                text = g3_question(dp, r["question"], r["s1_value"], descs)
                print(f"--- {dp} G3 question (state of {len(r['state'])} chars not shown):\n{text}")
            else:
                text = g4_prompt(dp, r["state"], r["options"], r["question"], r["s1_value"], descs)
                print(f"--- {dp} G4 prompt ({len(text)} chars, ~{len(text) // 4} tokens), tail:\n{text[-900:]}")
        chars = [len(g4_prompt(r["dp"], r["state"], r["options"], r["question"], r["s1_value"], descs))
                 if a.gate == "g4" else len(r["state"]) for r in todo]
        if chars:
            print(f"input ~{sum(chars) / 4 / 1e6:.2f} M tokens over {len(todo)} calls (chars / 4); no call made")
        return 0
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if a.gate == "g3":
        from jevharness.decision.jev import JevBackend

        client: Any = JevBackend(provider="openrouter", timeout=30.0, max_retries=3)   # Jev-only key (decision/jev.py)
        fn = call_g3
    else:
        from jevharness.llm import make_llm

        client = make_llm(dict(LLM_SPEC))
        fn = call_g4
    spent = [spend()["total"]]
    est = [0.0]
    stats = {"ok": 0, "err": 0, "capped": 0}

    def work(rec: Mapping[str, Any]) -> None:
        with _LOCK:
            if spent[0] + est[0] > CAP_USD:
                stats["capped"] += 1
                return
        row: dict[str, Any] = {"id": rec["id"], "gate": a.gate, "dp": rec["dp"],
                               "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        try:
            row.update(fn(client, rec, descs))
            row["error"] = None if row.get("score") is not None else "no score parsed"
        except Exception as e:  # noqa: BLE001 -- recorded and retried in a later pass
            row.update({"score": None, "error": f"{type(e).__name__}: {str(e)[:300]}", "calls": []})
        cost = sum(float(c.get("cost_usd") or 0.0) for c in row.get("calls") or [])
        with _LOCK:
            spent[0] += cost
            est[0] = max(est[0], cost)
            stats["ok" if row["error"] is None else "err"] += 1
            with calls_path(a.gate).open("a") as f:
                f.write(json.dumps(row) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    print(f"{a.gate}: {stats}; E8 spend so far ${spent[0]:.4f} (cap ${CAP_USD:.0f})")
    return 0 if stats["capped"] == 0 else 2


def cmd_project(a: argparse.Namespace) -> int:
    """The registered gate: the smoke's mean cost per call x 1,000 x 1.3 per gate, plus the spend so far."""
    ids = load_sample()["ids"][:SMOKE_N]
    sp = spend()
    proj = sp["total"]
    rows = {}
    for gate in ("g3", "g4"):
        sc = latest_scores(gate)
        costs = [sum(float(c.get("cost_usd") or 0.0) for c in sc[i].get("calls") or []) for i in ids if i in sc]
        mean = float(np.mean(costs)) if costs else float("nan")
        rows[gate] = {"n_smoke": len(costs), "mean_cost_per_call": mean,
                      "projected": mean * N_PER_DP * len(DPS) * PROJ_FACTOR}
        proj += rows[gate]["projected"] if costs else math.inf
    out = {"spend": sp, "gates": rows, "projection_total": proj, "cap": CAP_USD, "go": bool(proj <= CAP_USD),
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    (OUT / "projection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out["go"] else 1


# ================================================================================================ score
def gate_arrays(recs: Sequence[Mapping[str, Any]], g3: Mapping[str, Mapping[str, Any]],
                g4: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per-decision arrays of one decision point: labels, costs and every gate's score and own call."""
    def call_sum(row: Mapping[str, Any] | None, key: str) -> float:
        return float(sum(float(c.get(key) or 0.0) for c in (row or {}).get("calls") or []))

    c = np.array([float(r["c"]) for r in recs])
    A: dict[str, Any] = {
        "ids": [r["id"] for r in recs], "tasks": [r["task_id"] for r in recs],
        "g": np.array([float(r["s1_ok"]) for r in recs]), "q": np.array([float(r["s2_ok"]) for r in recs]),
        "dc": np.array([float(r["dc"]) for r in recs]), "dl": np.array([float(r["dl"]) for r in recs]),
        "k_usd": np.array([float(r["k_usd"]) for r in recs]), "k_ms": np.array([float(r["k_ms"]) for r in recs]),
        "c": c}
    A["scores"] = {
        "g1": c,
        "g2": np.array([top2_margin(r["s1_dist"]) for r in recs]),
        "g3": np.array([float(g3[r["id"]]["score"]) for r in recs]),
        "g4": np.array([float(g4[r["id"]]["score"]) for r in recs]),
        "g4v": np.array([1.0 if g4[r["id"]].get("verdict") == "yes" else 0.0 for r in recs]),
        "g5": np.array([1.0 if M._norm(r["s1_value"]) == M._norm(r["s2_value"]) else 0.0 for r in recs])}
    A["g3_usd"] = np.array([call_sum(g3.get(r["id"]), "cost_usd") for r in recs])
    A["g3_ms"] = np.array([call_sum(g3.get(r["id"]), "latency_ms") for r in recs])
    A["g4_usd"] = np.array([call_sum(g4.get(r["id"]), "cost_usd") for r in recs])
    A["g4_ms"] = np.array([call_sum(g4.get(r["id"]), "latency_ms") for r in recs])
    A["g4_verdict"] = [g4[r["id"]].get("verdict") for r in recs]
    return A


def gate_specs(A: Mapping[str, Any], tau: float) -> dict[str, dict[str, Any]]:
    """Score, consultation mask, pre-accepted mask, gate cost / latency and accounting flags of every gate."""
    n = len(A["g"])
    zero, none_, all_ = np.zeros(n), np.zeros(n, bool), np.ones(n, bool)
    hi = A["c"] >= tau - 1e-12
    S = A["scores"]
    spec = {
        "g1": {"score": S["g1"], "cost": zero, "ms": zero},
        # the deployed rule: G1 at the registered tau-hat, no threshold chosen on these decisions (tau-hat was fitted on
        # the whole calibration log, which contains them); the baseline of the two-stage rows
        "g1tau": {"score": S["g1"], "cost": zero, "ms": zero, "fixed": tau - 1e-12},
        "g2": {"score": S["g2"], "cost": zero, "ms": zero},
        "g3": {"score": S["g3"], "cost": A["g3_usd"], "ms": A["g3_ms"]},
        "g4": {"score": S["g4"], "cost": A["g4_usd"], "ms": A["g4_ms"]},
        "g4v": {"score": S["g4v"], "cost": A["g4_usd"], "ms": A["g4_ms"]},
        # G5: the LLM's own call is made on every decision (so accepting saves nothing), and the model's call can run
        # beside it (no added critical-path time)
        "g5": {"score": S["g5"], "cost": zero, "ms": zero, "saves": False, "model_on_path": False},
        "g1g3": {"score": np.where(hi, 1.0 + A["c"], S["g3"]), "gate_score": S["g3"], "cost": A["g3_usd"],
                 "ms": A["g3_ms"], "consult": ~hi, "pre": hi},
        "g1g4": {"score": np.where(hi, 1.0 + A["c"], S["g4"]), "gate_score": S["g4"], "cost": A["g4_usd"],
                 "ms": A["g4_ms"], "consult": ~hi, "pre": hi},
        "g1g4v": {"score": np.where(hi, 1.0 + A["c"], S["g4v"]), "gate_score": S["g4v"], "cost": A["g4_usd"],
                  "ms": A["g4_ms"], "consult": ~hi, "pre": hi},
        "perfect": {"score": A["g"].copy(), "cost": zero, "ms": zero}}
    for s in spec.values():
        s.setdefault("consult", all_)
        s.setdefault("pre", none_)
        s.setdefault("saves", True)
        s.setdefault("model_on_path", True)
        s.setdefault("gate_score", s["score"])
    return spec


def score_point(A: Mapping[str, Any], tau: float, n_boot: int = N_BOOT) -> dict[str, Any]:
    g = A["g"]
    spec = gate_specs(A, tau)
    boots = boot_auroc({k: v["score"] for k, v in spec.items() if k != "perfect"}, g, A["tasks"], "g1", n_boot)
    dc_mean = float(A["dc"].mean())
    out: dict[str, Any] = {
        "n": int(len(g)), "n_wrong": int((g == 0).sum()), "model_acc": float(g.mean()), "llm_acc": float(A["q"].mean()),
        "n_tasks": len(set(A["tasks"])), "dc_usd": dc_mean, "dl_ms": float(A["dl"].mean()),
        "kappa_usd": float(A["k_usd"].mean()), "kappa_ms": float(A["k_ms"].mean()), "tau_hat": tau,
        "share_above_tau": float((A["c"] >= tau - 1e-12).mean()),
        "g3_call_usd": float(A["g3_usd"].mean()), "g3_call_ms": float(A["g3_ms"].mean()),
        "g4_call_usd": float(A["g4_usd"].mean()), "g4_call_ms": float(A["g4_ms"].mean()),
        "median_ms": {"llm": float(np.median(A["dl"])), "model": float(np.median(A["k_ms"])),
                      "g3": float(np.median(A["g3_ms"])), "g4": float(np.median(A["g4_ms"]))}, "gates": {}}
    v = [x for x in A["g4_verdict"]]
    s4 = A["scores"]["g4"]
    out["g4_verdicts"] = {"yes": sum(1 for x in v if x == "yes"), "no": sum(1 for x in v if x == "no"),
                          "none": sum(1 for x in v if x is None),
                          "no_with_conf_ge_50": int(sum(1 for x, s in zip(v, s4) if x == "no" and s >= 0.5)),
                          "yes_with_conf_lt_50": int(sum(1 for x, s in zip(v, s4) if x == "yes" and s < 0.5)),
                          "auroc_verdict_only": M.auroc([1.0 if x == "yes" else 0.0 for x in v], g),
                          "conf_100": int(sum(1 for x in s4 if x >= 1.0 - 1e-12))}
    for k, s in spec.items():
        row: dict[str, Any] = {"label": GATE_LABEL[k], "extra_call": GATE_CALL[k],
                               "auroc": M.auroc(s["score"], g),
                               "sel_acc": {f"{int(cv * 100)}": M.selective_accuracy(s["score"], g, cv)
                                           for cv in COVERAGES},
                               "gate_call_usd": float(s["cost"][s["consult"]].sum() / len(g)),
                               "gate_call_ms": float(s["ms"][s["consult"]].sum() / len(g)),
                               "consulted": float(np.mean(s["consult"])), "econ": {}}
        if k in boots:
            row.update({"auroc_ci": boots[k]["ci"], "auroc_diff_g1": boots[k]["diff"],
                        "auroc_diff_g1_ci": boots[k]["diff_ci"]})
        for V in V_GRID:
            for free in (False, True):
                cost = np.zeros(len(g)) if free else s["cost"]
                ms = np.zeros(len(g)) if free else s["ms"]
                if "fixed" in s:
                    acc = s["consult"] & (s["gate_score"] >= s["fixed"])
                    e = {"usd": saving_at(s["fixed"], s["gate_score"], g, A["q"], A["dc"], A["k_usd"], cost, V,
                                          consult=s["consult"], pre=s["pre"], saves=s["saves"]) / len(g),
                         "threshold": s["fixed"] + 1e-12, "coverage": float(acc.mean())}
                else:
                    e = net_saving(s["gate_score"], g, A["q"], A["dc"], A["k_usd"], cost, V, consult=s["consult"],
                                   pre=s["pre"], saves=s["saves"])
                    acc = e.pop("accepted")
                e["latency_ms"] = latency_change(acc, A["dl"], A["k_ms"], ms, s["consult"], s["saves"],
                                                 s["model_on_path"])
                e["pct_of_llm_cost"] = 100.0 * e["usd"] / dc_mean if dc_mean else float("nan")
                e["final_acc"] = float((np.where(acc, g, A["q"])).mean())
                e["accepted_acc"] = float(g[acc].mean()) if acc.any() else float("nan")
                if V == V_PRIMARY and "fixed" in s:
                    e["heldout_usd"], e["heldout_pct"] = e["usd"], 100.0 * e["usd"] / dc_mean
                    e["heldout_range_pct"] = [e["heldout_pct"], e["heldout_pct"]]
                elif V == V_PRIMARY:
                    cv = crossfit_saving(A["tasks"], s["gate_score"], g, A["q"], A["dc"], A["k_usd"], cost, V,
                                         consult=s["consult"], pre=s["pre"], saves=s["saves"])
                    e["heldout_usd"] = cv["usd"]
                    e["heldout_pct"] = 100.0 * cv["usd"] / dc_mean if dc_mean else float("nan")
                    e["heldout_range_pct"] = [100.0 * cv["min"] / dc_mean, 100.0 * cv["max"] / dc_mean]
                row["econ"][f"V{V:g}{'_free' if free else ''}"] = e
        out["gates"][k] = row
    return out


def _f(x: Any, d: int = 1, pct: bool = False, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–" if x is None or not (isinstance(x, float) and math.isinf(x)) else ("∞" if x > 0 else "-∞")
    v = 100.0 * x if pct else float(x)
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def render_md(res: Mapping[str, Any]) -> str:
    L = ["# E8: what decides whether the decision model's answer is accepted (exploratory, descriptive)", "",
         f"Registered: docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 \"Exploratory row E8\". Outside both stages; no "
         "test; revises no registered result, selects nothing and changes no threshold.", "",
         f"- Data: `{res['source']}` (sha256 `{res['sha256'][:16]}…`), the registered calibration shadow arm: JevGym "
         f"calibration seeds {res['seed_range'][0]}–{res['seed_range'][1]}, Gemini 3.5 Flash as the agent, Jev "
         f"answering in shadow at every state. Sample: {res['n']} decisions from {res['n_tasks']} tasks "
         "(500 per decision point, seed 0, stratified in proportion by whether the decision model is right).",
         f"- Spend: ${res['spend']['total']:.2f} logged (G3 ${res['spend']['g3']:.2f}, G4 ${res['spend']['g4']:.2f}); "
         f"cap ${CAP_USD:.0f}.",
         f"- Saving = the best expected saving per decision against the LLM alone over the gate's thresholds, on these "
         f"decisions (an in-sample upper bound), V = ${V_PRIMARY:g} per wrong decision, with the decision model's own "
         "call and the gate's call charged; in % of the LLM's cost of the decision. Latency = critical-path time saved "
         "per decision at that threshold (negative = slower). 'If free' = the same optimum with the gate's call not "
         "charged (its ranking alone). 'Held-out' (post hoc) = the threshold chosen on half of the tasks and scored on "
         "the other half, both ways, mean of 10 random halvings: the in-sample optimum favours gates with many "
         "distinct scores.", ""]
    for dp in DPS:
        P = res["points"][dp]
        L += [f"## {DP_LABEL[dp]} (n = {P['n']}, {P['n_wrong']} wrong answers of the decision model; its accuracy "
              f"{100 * P['model_acc']:.1f} %, the LLM's {100 * P['llm_acc']:.1f} %)", "",
              f"LLM cost of the decision ΔC = ${P['dc_usd']:.5f} ({P['dl_ms']:.0f} ms); decision-model call "
              f"κ = ${P['kappa_usd']:.6f} ({P['kappa_ms']:.0f} ms); G3 call ${P['g3_call_usd']:.6f} "
              f"({P['g3_call_ms']:.0f} ms); G4 call ${P['g4_call_usd']:.5f} ({P['g4_call_ms']:.0f} ms) = "
              f"{P['g4_call_usd'] / P['dc_usd']:.2f} × ΔC (medians: LLM {P['median_ms']['llm']:.0f} ms, model "
              f"{P['median_ms']['model']:.0f} ms, G3 {P['median_ms']['g3']:.0f} ms, G4 {P['median_ms']['g4']:.0f} ms). "
              f"Registered τ̂ = {P['tau_hat']:g} "
              f"({100 * P['share_above_tau']:.1f} % of decisions at or above it).", "",
              "| gate | extra call | AUROC [95 % CI] | Δ vs G1 [95 % CI] | sel. acc. @25/50/75/90 % | saving % of ΔC "
              "(USD) | coverage | latency ms | saving if free | held-out saving |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k in (*GATES, "perfect"):
            r = P["gates"][k]
            e, ef = r["econ"][f"V{V_PRIMARY:g}"], r["econ"][f"V{V_PRIMARY:g}_free"]
            ci = r.get("auroc_ci") or [None, None]
            dci = r.get("auroc_diff_g1_ci") or [None, None]
            sel = "/".join(_f(r["sel_acc"][str(int(cv * 100))], 1, pct=True) for cv in COVERAGES)
            L.append(f"| {r['label']} | {r['extra_call']} | {_f(r['auroc'], 3)} [{_f(ci[0], 3)}, {_f(ci[1], 3)}] | "
                     + (f"{_f(r.get('auroc_diff_g1'), 3, sign=True)} [{_f(dci[0], 3, sign=True)}, "
                        f"{_f(dci[1], 3, sign=True)}]" if k not in ("g1", "g1tau", "perfect") else "–")
                     + f" | {sel} | {_f(e['pct_of_llm_cost'], 1, sign=True)} % (${e['usd']:+.5f}) | "
                     f"{_f(e['coverage'], 1, pct=True)} % | {_f(e['latency_ms'], 0, sign=True)} | "
                     f"{_f(ef['pct_of_llm_cost'], 1, sign=True)} % | {_f(e['heldout_pct'], 1, sign=True)} % |")
        L += ["", f"Saving (% of ΔC) by the value of a wrong decision V:", "",
              "| gate | " + " | ".join(f"V = ${V:g}" for V in V_GRID) + " |", "|---|" + "---|" * len(V_GRID)]
        for k in (*GATES, "perfect"):
            r = P["gates"][k]
            L.append(f"| {r['label']} | " + " | ".join(
                f"{_f(r['econ'][f'V{V:g}']['pct_of_llm_cost'], 1, sign=True)} % (cov. "
                f"{_f(r['econ'][f'V{V:g}']['coverage'], 0, pct=True)} %)" for V in V_GRID) + " |")
        gv = P["g4_verdicts"]
        L += ["", f"G4 replies: {gv['yes']} yes, {gv['no']} no, {gv['none']} without a verdict; confidence 100 in "
              f"{gv['conf_100']} of {P['n']}; {gv['no_with_conf_ge_50']}"
              f" 'no' with confidence ≥ 50 and {gv['yes_with_conf_lt_50']} 'yes' below 50; AUROC of the verdict alone "
              f"{_f(gv['auroc_verdict_only'], 3)}.", ""]
    L += ["## Reading", "", *res["reading"], "",
          "Notes: AUROC with ties counted ½; selective accuracy resolves ties in expectation "
          "(`metrics.selective_accuracy`); intervals are a task-level bootstrap (B = 2,000, seed 0). ΔC and κ are the "
          "logged per-decision costs of the registered fit (argument decisions share their step's LLM call); G3 and G4 "
          "costs and latencies come from E8's own ledger (`calls_g3.jsonl.gz`, `calls_g4.jsonl.gz`). G5 needs the agent's own "
          "LLM call on every decision, so accepting saves nothing; its decision-model call is taken to run beside the "
          "LLM (no added time). Two-stage rows: the decision model's confidence accepts at the registered τ̂; the "
          "second gate is asked (and paid) only below it, so their baseline is G1 at the registered τ̂ (fitted on the "
          "whole calibration log, which contains these decisions: no threshold is chosen here, so its held-out value "
          "equals its in-sample one). G4v and G1>G4v are post hoc (docs/PREREG_A7_STAGE2.md, "
          "Deviations 2026-10-07 \"E8: the verifier's verdict as its score\"): every G4 reply gave confidence 100 "
          "whatever its verdict, so the registered G4 score is constant; G4v scores the verdict (yes 1, no 0) from the "
          "same replies, no new call. Latencies of G3 and G4 were measured on 2026-10-07, those of ΔC and κ in the "
          "calibration run (2026-10-01): compare them with care.", ""]
    return "\n".join(L)


def reading(res: Mapping[str, Any]) -> list[str]:
    """Plain sentences built from the numbers (no claim beyond them)."""
    out = []
    short = {"g1": "G1", "g1tau": "G1@tau", "g2": "G2", "g3": "G3", "g4": "G4", "g4v": "G4v", "g5": "G5", "g1g3": "G1>G3",
             "g1g4": "G1>G4", "g1g4v": "G1>G4v"}
    for dp in DPS:
        P = res["points"][dp]
        G = P["gates"]
        e = {k: G[k]["econ"][f"V{V_PRIMARY:g}"] for k in G}
        ef = {k: G[k]["econ"][f"V{V_PRIMARY:g}_free"] for k in G}
        best = max(GATES, key=lambda k: e[k]["usd"])
        au = ", ".join(f"{short[k]} {G[k]['auroc']:.3f}" for k in GATES)
        sv = ", ".join(f"{short[k]} {e[k]['pct_of_llm_cost']:+.1f} %"
                       + (f" (free: {ef[k]['pct_of_llm_cost']:+.1f} %)" if k in ("g3", "g4v", "g1g4v") else "")
                       for k in GATES)
        ho = ", ".join(f"{short[k]} {e[k]['heldout_pct']:+.1f} %" for k in GATES)
        out.append(f"- {DP_LABEL[dp].capitalize()}: AUROC {au}. Best saving per decision (V = ${V_PRIMARY:g}): {sv}; "
                   f"held out: {ho}; a "
                   f"perfect free gate {e['perfect']['pct_of_llm_cost']:+.1f} %. Highest: {GATE_LABEL[best]}. One G4 "
                   f"call costs {P['g4_call_usd'] / P['dc_usd']:.2f} × the LLM's cost of the decision; G4's 0-100 "
                   f"confidence was 100 in {P['g4_verdicts']['conf_100']} of {P['n']} replies.")
    return out


def tex_keys(res: Mapping[str, Any]) -> dict[str, str]:
    v: dict[str, str] = {"e8/n": C.fmt(int(res["n"])), "e8/ntasks": C.fmt(int(res["n_tasks"])),
                         "e8/spend": f"{res['spend']['total']:.2f}", "e8/cap": f"{CAP_USD:.0f}",
                         "e8/V": f"{V_PRIMARY:g}"}
    for dp in DPS:
        P, d = res["points"][dp], DP_KEY[dp]
        v[f"e8/{d}/n"] = C.fmt(int(P["n"]))
        v[f"e8/{d}/nwrong"] = C.fmt(int(P["n_wrong"]))
        v[f"e8/{d}/s1acc"] = C.fmt(100 * P["model_acc"], 1)
        v[f"e8/{d}/llmacc"] = C.fmt(100 * P["llm_acc"], 1)
        v[f"e8/{d}/dc"] = f"{P['dc_usd']:.4f}"
        v[f"e8/{d}/dlms"] = C.fmt(P["dl_ms"], 0)
        v[f"e8/{d}/g3cost"] = f"{P['g3_call_usd']:.5f}"
        v[f"e8/{d}/g3ms"] = C.fmt(P["g3_call_ms"], 0)
        v[f"e8/{d}/g4cost"] = f"{P['g4_call_usd']:.4f}"
        v[f"e8/{d}/g4ms"] = C.fmt(P["g4_call_ms"], 0)
        v[f"e8/{d}/g4overdc"] = C.fmt(P["g4_call_usd"] / P["dc_usd"], 2)
        for k in (*GATES, "perfect"):
            r = P["gates"][k]
            e, ef = r["econ"][f"V{V_PRIMARY:g}"], r["econ"][f"V{V_PRIMARY:g}_free"]
            b = f"e8/{d}/{k}"
            if k != "perfect":
                v[f"{b}/auroc"] = C.fmt(r["auroc"], 2)
            for cv in (50, 90):
                v[f"{b}/sel{cv}"] = C.fmt(100 * r["sel_acc"][str(cv)], 1)
            v[f"{b}/save"] = C.fmt(e["pct_of_llm_cost"], 0)
            v[f"{b}/savefree"] = C.fmt(ef["pct_of_llm_cost"], 0)
            v[f"{b}/saveheld"] = C.fmt(e["heldout_pct"], 0)
            v[f"{b}/cov"] = C.fmt(100 * e["coverage"], 0)
            v[f"{b}/latms"] = C.fmt(e["latency_ms"], 0)
    return v


def write_tex(vals: Mapping[str, str], path: Path = TEX) -> Path:
    lines = ["% AUTO-GENERATED by scripts/explore_e8_gates.py -- do not edit by hand.",
             "% Exploratory row E8, acceptance gates (descriptive; docs/PREREG_A7_STAGE2.md Deviations 2026-10-07), "
             "outside both stages.",
             f"% last written {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git "
             f"{(C.git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else ("inf" if f > 0 else "-inf" if f < 0 else None)
    if isinstance(o, np.integer):
        return int(o)
    return o


def cmd_score(a: argparse.Namespace) -> int:
    _check_source()
    smp = load_sample()
    recs = sample_records(smp)
    g3, g4 = latest_scores("g3"), latest_scores("g4")
    missing = {"g3": [r["id"] for r in recs if r["id"] not in g3], "g4": [r["id"] for r in recs if r["id"] not in g4]}
    keep = [r for r in recs if r["id"] in g3 and r["id"] in g4]
    res: dict[str, Any] = {"source": C.rel(SRC), "sha256": SRC_SHA256, "seed_range": smp["seed_range"],
                           "n_sample": len(recs), "n": len(keep), "n_tasks": len({r["task_id"] for r in keep}),
                           "missing": {k: len(x) for k, x in missing.items()}, "spend": spend(),
                           "cap": CAP_USD, "V": V_PRIMARY, "V_grid": list(V_GRID), "coverages": list(COVERAGES),
                           "llm": LLM_SPEC, "points": {},
                           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "git": C.git_head()}
    for dp in DPS:
        A = gate_arrays([r for r in keep if r["dp"] == dp], g3, g4)
        res["points"][dp] = score_point(A, TAU_HAT[dp], a.n_boot)
    res["reading"] = reading(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(_jsonable(res), indent=1) + "\n")
    (OUT / "summary.md").write_text(render_md(res))
    write_tex(tex_keys(res))
    print(render_md(res))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    r = sub.add_parser("run")
    r.add_argument("--gate", choices=("g3", "g4"), required=True)
    r.add_argument("--limit", type=int, default=0, help="only the first N decisions of the sample order (smoke)")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--dry-run", action="store_true", help="build the prompts and print them; no call")
    sub.add_parser("project")
    s = sub.add_parser("score")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    return {"sample": cmd_sample, "run": cmd_run, "project": cmd_project, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
