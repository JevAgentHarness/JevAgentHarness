"""S2-3's common-state corpus and its tool_select replay (docs/PREREG_A7_STAGE2.md A7.10 (c), (d)).

**Corpus.** For each eligible task t, C_t is the multiset of labelled pre-action tool-selection question occurrences in
the final valid trajectory of each of the four source arms (Stage 2's `llm_only` and the three `*_ts_replace_f` arms of
configs/experiments/stage2_rl_states.yaml, on the first 200 Stage-2 tasks). Every occurrence has equal weight within its
task and every task equal weight in the mean. An occurrence keeps the task id, source arm, step, the canonical
question hash (sha256 of the exact System-1 state string, option list, option descriptions and question text; no
oracle label and no source answer enter the hash) and its label (the logged A4 acceptable set / oracle tool, with its
provenance). A task without four valid source trajectories leaves the primary corpus, listed with the failing source.

**Replay.** Every backend answers every distinct canonical question through ONE path: the live harness's own option
handling (`JevAgent._choice_tournament` over the backend's `max_options`; a served checkpoint does its own knockout),
one question per request, in sorted question-hash order, through a replay cache separate from rollout caches, never
reusing the answers that generated the source trajectories. A scoring failure is retried twice and then counted
incorrect for that backend (A7.10 (d)); invalid option outputs are model failures.

**Score.** acc_X(t) = mean over the occurrences of C_t of `metrics.is_correct(record, answer_X(question))`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..accounting import Ledger
from ..harness.prompts import Q_TOOL_SELECT
from ..harness.tools import finish_tool, tool_desc, tool_name
from ..harness.trajectory import iter_records, latest_by_task
from . import metrics as M

SOURCES = (("stage2", "llm_only"), ("states", "jev_ts_replace_f"), ("states", "intern_ts_replace_f"),
           ("states", "rl_ts_replace_f"))
TOOL_MODES = ("off", "replace", "cascade", "assist", "guard", "shadow")
LABEL_PROVENANCE = "oracle: the A4 acceptable tool set logged at decision time (gold_aux.acceptable_tools, else gold)"


def question_hash(state: str, options: Sequence[str], descriptions: Sequence[str] | None, text: str = Q_TOOL_SELECT) -> str:
    blob = json.dumps({"state": state, "options": list(options), "descriptions": list(descriptions or []),
                       "text": text}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return latest_by_task(r for r in iter_records(path) if "task_id" in r)


def _executed_off_plan(rec: Mapping[str, Any], executed: str | None) -> bool | None:
    """Was the executed tool outside the step's acceptable set? None without a label."""
    if executed is None or not M.has_gold(rec):
        return None
    ok = M.is_correct(rec, executed)
    return None if ok is None else not ok


def occurrences(traj: Mapping[str, Any], source: str, descs: Mapping[str, str]) -> list[dict[str, Any]]:
    """The labelled pre-action tool-selection question occurrences of one trajectory (one per step: the step's first
    tool_select record, i.e. the question asked before the step's action), with the on-plan / after-an-off-plan-step
    stratum (off-plan = an earlier executed tool outside its acceptable set)."""
    out: list[dict[str, Any]] = []
    off_plan_seen = False
    for st in traj.get("steps") or []:
        rec = next((d for d in st.get("decisions") or [] if d.get("dp") == "tool_select"
                    and d.get("mode") in TOOL_MODES and d.get("options") and d.get("state")), None)
        if rec is not None and M.has_gold(rec):
            opts = list(rec["options"])
            d = [descs.get(o, "") for o in opts]
            out.append({"task": str(traj["task_id"]), "source": source, "step": int(st.get("step", 0)),
                        "qhash": question_hash(rec["state"], opts, d), "state": rec["state"], "options": opts,
                        "descriptions": d, "gold": rec.get("gold"), "gold_aux": rec.get("gold_aux"),
                        "gold_status": rec.get("gold_status"), "dp": "tool_select",
                        "stratum": "after_off_plan" if off_plan_seen else "on_plan",
                        "label_provenance": LABEL_PROVENANCE})
        executed = st.get("tool")
        if rec is not None:
            off = _executed_off_plan(rec, executed)
            off_plan_seen = off_plan_seen or bool(off)
    return out


def tool_descriptions(env_factory: Callable[[], Any] | None, final_tool: str = "finish") -> dict[str, str]:
    """name -> the one-line description System-1 sees (harness/agent.py passes `tool_desc` of every offered tool)."""
    if env_factory is None:
        return {}
    env = env_factory()
    env.reset()
    tools = list(env.tools()) + [finish_tool(final_tool)]
    return {tool_name(t): tool_desc(t) for t in tools}


def build_corpus(sources: Sequence[tuple[str, Path]], tasks: Sequence[str],
                 factories: Mapping[str, Callable[[], Any]] | None = None) -> dict[str, Any]:
    """(source label, arm JSONL) x tasks -> {occurrences, questions, excluded, summary}."""
    latest = {label: load_latest(Path(p)) for label, p in sources}
    occ: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for t in tasks:
        bad = {label: ("missing" if t not in latest[label] else "infrastructure error")
               for label, _ in sources if t not in latest[label] or latest[label][t].get("error")}
        if bad:
            excluded.append({"task": t, "failing_sources": bad})
            continue
        descs = tool_descriptions((factories or {}).get(t))
        for label, _ in sources:
            occ.extend(occurrences(latest[label][t], label, descs))
    questions: dict[str, dict[str, Any]] = {}
    for o in occ:
        questions.setdefault(o["qhash"], {k: o[k] for k in ("state", "options", "descriptions")})
    per_task = Counter(o["task"] for o in occ)
    by_src = Counter(o["source"] for o in occ)
    eligible = [t for t in tasks if t not in {e["task"] for e in excluded}]
    summary = {"tasks": len(tasks), "eligible_tasks": len(eligible), "tasks_with_occurrences": len(per_task),
               "occurrences": len(occ), "unique_questions": len(questions),
               "source_weights": {k: v / max(1, len(occ)) for k, v in by_src.items()},
               "strata": dict(Counter(o["stratum"] for o in occ)), "excluded": len(excluded)}
    return {"registered": "PREREG_A7_STAGE2 A7.10 (c)", "occurrences": occ, "questions": questions,
            "excluded": excluded, "eligible_tasks": eligible, "summary": summary}


def write_corpus(corpus: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    occ_p, q_p, meta_p = out_dir / "occurrences.jsonl.gz", out_dir / "questions.jsonl.gz", out_dir / "corpus.json"
    with gzip.open(occ_p, "wt") as f:
        for o in corpus["occurrences"]:
            f.write(json.dumps({k: v for k, v in o.items() if k not in ("state", "options", "descriptions")}) + "\n")
    with gzip.open(q_p, "wt") as f:
        for h in sorted(corpus["questions"]):
            f.write(json.dumps({"qhash": h, **corpus["questions"][h]}, ensure_ascii=False) + "\n")
    meta = {k: corpus[k] for k in ("registered", "excluded", "eligible_tasks", "summary")}
    meta["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (occ_p, q_p)}
    meta_p.write_text(json.dumps(meta, indent=1))
    return {"occurrences": str(occ_p), "questions": str(q_p), "meta": str(meta_p)}


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


# ------------------------------------------------------------------ replay
def replay(questions: Sequence[Mapping[str, Any]], backend: Any, retries: int = 2,
           on_answer: Callable[[dict[str, Any]], None] | None = None,
           done: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Answer every distinct question (sorted by hash) with ``backend`` through the live harness's option handling;
    a failure is retried ``retries`` times, then recorded with ``error`` (scored incorrect)."""
    from ..harness.agent import JevAgent
    from ..harness.config import DPConfig, HarnessConfig, Mode

    shell = JevAgent(llm=None, cfg=HarnessConfig(name="replay", tool_select=DPConfig(mode=Mode.REPLACE)), s1=backend)
    skip = set(done)
    out: list[dict[str, Any]] = []
    for q in sorted(questions, key=lambda x: x["qhash"]):
        if q["qhash"] in skip:
            continue
        rec: dict[str, Any] = {"qhash": q["qhash"], "answer": None, "confidence": None, "error": None, "attempts": 0}
        for attempt in range(retries + 1):
            led = Ledger()
            rec["attempts"] = attempt + 1
            t0 = time.perf_counter()
            try:
                a = shell._choice_tournament(q["state"], Q_TOOL_SELECT, list(q["options"]), "tool_select", led,
                                             list(q.get("descriptions") or []) or None, gold=None)
                rec.update({"answer": a.value, "confidence": a.confidence, "error": None,
                            "valid": a.value in q["options"]})
                if (a.raw or {}).get("refusal"):     # E15: a refusal is no answer (scored incorrect, counted)
                    rec.update({"answer": None, "confidence": 0.0, "valid": False, "refusal": True})
            except Exception as e:  # noqa: BLE001 - a failed replay is counted, never a crash
                rec.update({"answer": None, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            rec.update({"cost_usd": sum(r.cost_usd for r in led.records),
                        "latency_ms": sum(r.latency_ms for r in led.records),
                        "wall_ms": (time.perf_counter() - t0) * 1000, "requests": len(led.records)})
            if rec["error"] is None:
                break
        out.append(rec)
        if on_answer is not None:
            on_answer(rec)
    return out


# ------------------------------------------------------------------ score
def answer_correct(o: Mapping[str, Any], a: Mapping[str, Any] | None) -> bool:
    """Is replay answer ``a`` correct for occurrence ``o``? A missing, failed or invalid answer is incorrect (A7.10 (d))."""
    a = a or {}
    val = a.get("answer") if not a.get("error") and a.get("valid", True) else None
    if val is None:
        return False
    return bool(M.is_correct({"dp": "tool_select", "gold": o.get("gold"), "gold_aux": o.get("gold_aux"),
                              "gold_status": o.get("gold_status")}, val))


def task_accuracy(occ: Sequence[Mapping[str, Any]], answers: Mapping[str, Mapping[str, Any]],
                  tasks: Sequence[str] | None = None) -> dict[str, float]:
    """task -> mean correctness of the backend's answers over the task's occurrences (a missing, failed or invalid
    answer is incorrect, A7.10 (d))."""
    acc: dict[str, list[float]] = defaultdict(list)
    keep = set(tasks) if tasks is not None else None
    for o in occ:
        if keep is not None and o["task"] not in keep:
            continue
        acc[o["task"]].append(float(answer_correct(o, answers.get(o["qhash"]))))
    return {t: sum(v) / len(v) for t, v in acc.items() if v}
