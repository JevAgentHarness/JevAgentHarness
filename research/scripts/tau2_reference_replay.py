#!/usr/bin/env python3
r"""A7.13 (iv) external reference agreement (docs/PREREG_A7_STAGE2.md): the frozen JevAgentRL, Intern-Decision-4B and
the untrained base answer the System-1 questions logged by the Stage-1 τ² arms (airline test, retail), scored as
**agreement with the A6.22 (b) reference labels** (not oracle accuracy). An offline distribution-shift diagnostic: not
evidence of τ² success or savings.

**Corpus** (`build`). Every tool_select / tool_params decision record that System-1 was asked (mode not `off`, not
an always-defer record; the `off` records of llm_only / jev_guard log the state for measurement only and are counted
as `not_asked`) with a logged state in the final trajectory of each (task, trial) of every arm file under
runs/tau2/{airline_test,retail}/ is an *occurrence*; Jev's live answer is logged on every one of them. Its
question is rebuilt exactly as the adapter asked it (`bench/suites/tau2.py`): the logged state, question text and
option list; tool_select's option descriptions are the τ² tool schemas' first description lines + RESPOND_DESC in the
adapter's order (`_select`), checked against the logged option list; tool_params Choices carry no descriptions
(`_param_question`). Questions are keyed by a canonical hash (state, text, options, descriptions, decision point; no
label, no logged answer) and written WITHOUT labels (questions.jsonl.gz, the only file the GPU host reads); labels stay
in occurrences.jsonl.gz. Label construction: A6.22 (b) 1-2 (tool_select: the reference next step, a singleton;
tool_params: the reference action's value in the option-label space, OTHER for a value outside the candidates, and a
label only when the executed tool is the reference next step). An occurrence without a label is counted in the
coverage and left out of the agreement.

**Replay** (`replay`): one path for every backend, as A7.10 (c)'s corpus replay (`JevAgent._choice_tournament`, one
question per request, sorted question hash, a separate replay cache, a failure retried twice then counted as
disagreement); `--shard i/n` splits the sorted list for parallel GPUs (answers_<name>.<i>of<n>.jsonl, merged by
`score`).

**Score** (`score`): per backend, domain and decision point, agreement = mean over tasks (τ² task id, trials pooled) of
the mean over the task's labelled occurrences (all arms pooled), with a task-bootstrap 95% interval; paired
differences between the frozen model and each other row on the same occurrences; Jev's answer as logged at decision
time (the live Stage-1 System-1) on the occurrences where one was logged, beside every row on that same subset; the
most-frequent-label rate (tool_select) as a floor. Descriptive throughout.

    python scripts/tau2_reference_replay.py build --runs runs --out results/stage2/t2ref
    python scripts/tau2_reference_replay.py replay --corpus results/stage2/t2ref --name intern --shard 0/2 \
        --s1 '{"type": "intern_decision", "model": "internlm/Intern-Decision-4B", "device": "auto", "max_length": 8192}'
    python scripts/tau2_reference_replay.py score --corpus results/stage2/t2ref --tex paper/generated/t2ref.tex
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOMAINS = ("airline_test", "retail")
DPS = ("tool_select", "tool_params")
ROWS = ("frozen", "intern", "base")          # A7.13 (iv)'s three rows; any other answers_<name> file is scored too
REGISTERED = "PREREG_A7_STAGE2 A7.13 (iv)"
LABELS = ("reference-agreement labels, A6.22 (b) 1-2: tool_select = the reference next step (singleton; "
          "respond_to_user when no reference action is pending and grounded); tool_params = the reference action's "
          "value in the option-label space (OTHER outside the candidates), only when the executed tool is the "
          "reference next step")
N_BOOT = 10000
BOOT_SEED = 20261004


# ------------------------------------------------------------------ corpus
def question_hash(dp: str, state: str, text: str, options: Sequence[str], descriptions: Sequence[str] | None) -> str:
    blob = json.dumps({"dp": dp, "state": state, "text": text, "options": list(options),
                       "descriptions": list(descriptions or [])}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def tool_select_descriptions(domain: str) -> tuple[list[str], list[str]]:
    """(option names, descriptions) of the adapter's tool_select Choice for a τ² domain (`JevTau2Agent._select`)."""
    from jevharness.bench.suites.tau2 import RESPOND, RESPOND_DESC, _s1_schema
    from jevharness.harness.tools import tool_desc
    from tau2.registry import registry

    tools = registry.get_env_constructor(domain)().get_tools()
    names = [t.name for t in tools] + [RESPOND]
    return names, [tool_desc(_s1_schema(t.openai_schema)) for t in tools] + [RESPOND_DESC]


def _final_records(path: Path) -> list[dict[str, Any]]:
    """The last record of every task id (task id = "<domain>/<task>#<trial>") of a τ² arm file."""
    latest: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a cut final line
            if "task_id" in r:
                latest[r["task_id"]] = r
    return [latest[k] for k in sorted(latest)]


def build_corpus(runs: Path, ts_desc: Mapping[str, tuple[list[str], list[str]]] | None = None) -> dict[str, Any]:
    """Occurrences + distinct label-free questions from runs/tau2/<domain>/*.jsonl. ``ts_desc`` maps the τ² domain
    ("airline", "retail") to (option names, descriptions); by default from the installed τ² package."""
    occ: list[dict[str, Any]] = []
    qs: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    all_counts: Counter = Counter()
    ts_desc = dict(ts_desc or {})
    mismatch: Counter = Counter()
    for dom in DOMAINS:
        for f in sorted((runs / "tau2" / dom).glob("*.jsonl")):
            if f.stat().st_size == 0:
                continue
            sources[f"{dom}/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
            arm = f.stem
            for r in _final_records(f):
                tdom = ((r.get("meta") or {}).get("tau2") or {}).get("domain") or r["task_id"].split("/")[0]
                task = r["task_id"].split("#")[0]
                for si, st in enumerate(r.get("steps") or []):
                    for d in st.get("decisions") or []:
                        dp = d.get("dp")
                        if dp not in DPS:
                            continue
                        all_counts[(dom, dp)] += 1
                        if d.get("mode") == "off" or (d.get("gold_aux") or {}).get("always_defer"):
                            all_counts[(dom, dp, "not_asked")] += 1      # logged for measurement; S1 not asked
                            continue
                        if not d.get("state") or not d.get("options"):
                            all_counts[(dom, dp, "no_state")] += 1
                            continue
                        opts = list(d["options"])
                        desc = None
                        if dp == "tool_select":
                            if tdom not in ts_desc:
                                ts_desc[tdom] = tool_select_descriptions(tdom)
                            names, descs = ts_desc[tdom]
                            if names != opts:
                                mismatch[(dom, arm)] += 1
                                continue
                            desc = descs
                        h = question_hash(dp, d["state"], d["question"], opts, desc)
                        if h not in qs:
                            qs[h] = {"qhash": h, "dp": dp, "state": d["state"], "text": d["question"],
                                     "options": opts, "descriptions": desc}
                        gold = d.get("gold")
                        occ.append({"domain": dom, "arm": arm, "task": task, "task_id": r["task_id"], "step": si,
                                    "dp": dp, "mode": d.get("mode"), "qhash": h,
                                    "gold": gold if gold is None or isinstance(gold, str) else json.dumps(gold),
                                    "jev_logged": d.get("s1_value") if d.get("s1_value") is not None else None})
    cov = {}
    for dom in DOMAINS:
        for dp in DPS:
            o = [x for x in occ if x["domain"] == dom and x["dp"] == dp]
            lab = [x for x in o if x["gold"] is not None]
            cov[f"{dom}/{dp}"] = {"logged": all_counts[(dom, dp)], "not_asked": all_counts[(dom, dp, "not_asked")],
                                  "no_state": all_counts[(dom, dp, "no_state")],
                                  "occurrences": len(o), "labelled": len(lab),
                                  "coverage": round(len(lab) / len(o), 4) if o else None,
                                  "tasks": len({x["task"] for x in lab}),
                                  "distinct_labelled_questions": len({x["qhash"] for x in lab}),
                                  "jev_logged": sum(x["jev_logged"] is not None for x in lab)}
    used = {x["qhash"] for x in occ if x["gold"] is not None}
    questions = [qs[h] for h in sorted(used)]        # only labelled questions are replayed
    return {"occurrences": occ, "questions": questions,
            "summary": {"registered": REGISTERED, "labels": LABELS, "sources_sha256": sources,
                        "coverage": cov, "n_questions": len(questions),
                        "option_list_mismatches": {f"{a}/{b}": n for (a, b), n in sorted(mismatch.items())}}}


def _write_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    with gzip.GzipFile(path, "wb", mtime=0) as g:        # deterministic bytes (sha256 recorded)
        for r in rows:
            g.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_corpus(c: Mapping[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    summ = dict(c["summary"])
    summ["questions_sha256"] = _write_gz(out / "questions.jsonl.gz", c["questions"])
    summ["occurrences_sha256"] = _write_gz(out / "occurrences.jsonl.gz", c["occurrences"])
    (out / "corpus.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    return summ


# ------------------------------------------------------------------ replay
def replay(questions: Sequence[Mapping[str, Any]], backend: Any, retries: int = 2, on_answer: Any = None,
           done: Iterable[str] = ()) -> list[dict[str, Any]]:
    """A7.10 (c)'s replay path for both decision points: the live harness's option handling, one question per
    request, in the given order; never a label on the wire (gold=None)."""
    from jevharness.accounting import Ledger
    from jevharness.harness.agent import JevAgent
    from jevharness.harness.config import DPConfig, HarnessConfig, Mode

    shell = JevAgent(llm=None, cfg=HarnessConfig(name="replay", tool_select=DPConfig(mode=Mode.REPLACE)), s1=backend)
    skip = set(done)
    out = []
    for q in questions:
        if q["qhash"] in skip:
            continue
        rec: dict[str, Any] = {"qhash": q["qhash"], "answer": None, "confidence": None, "error": None, "attempts": 0}
        for attempt in range(retries + 1):
            led = Ledger()
            rec["attempts"] = attempt + 1
            t0 = time.perf_counter()
            try:
                a = shell._choice_tournament(q["state"], q["text"], list(q["options"]), q["dp"], led,
                                             list(q.get("descriptions") or []) or None, gold=None)
                rec.update({"answer": a.value, "confidence": a.confidence, "error": None,
                            "valid": a.value in q["options"]})
            except Exception as e:  # noqa: BLE001 - a failed replay is counted, never a crash
                rec.update({"answer": None, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            rec.update({"latency_ms": sum(r.latency_ms for r in led.records),
                        "wall_ms": (time.perf_counter() - t0) * 1000, "requests": len(led.records)})
            if rec["error"] is None:
                break
        out.append(rec)
        if on_answer is not None:
            on_answer(rec)
    return out


def shard_of(questions: Sequence[Mapping[str, Any]], shard: str | None) -> tuple[list[Mapping[str, Any]], str]:
    qs = sorted(questions, key=lambda x: x["qhash"])
    if not shard:
        return qs, ""
    i, n = (int(x) for x in shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad shard {shard}")
    return qs[i::n], f".{i}of{n}"


def load_answers(cdir: Path, name: str) -> dict[str, dict[str, Any]]:
    """answers_<name>.jsonl and its shards (answers_<name>.<i>of<n>.jsonl), or their committed .gz copies; a later
    line for a hash wins."""
    out: dict[str, dict[str, Any]] = {}
    files = [f for pat in (f"answers_{name}.jsonl", f"answers_{name}.*of*.jsonl") for ext in ("", ".gz")
             for f in sorted(cdir.glob(pat + ext))]
    for f in files:
        text = gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text()
        for line in text.splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[r["qhash"]] = r
    return out


# ------------------------------------------------------------------ score
def _agree(ans: Mapping[str, Any] | None, gold: str) -> float:
    a = ans or {}
    if a.get("error") or not a.get("valid", True) or a.get("answer") is None:
        return 0.0
    return float(a["answer"] == gold)


def _task_means(occ: Sequence[Mapping[str, Any]], val: Any) -> dict[str, float]:
    per: dict[str, list[float]] = defaultdict(list)
    for o in occ:
        per[o["task"]].append(val(o))
    return {t: sum(v) / len(v) for t, v in per.items()}


def _boot(tasks: Sequence[str], means: Sequence[Mapping[str, float]], fn: Any, seed: int = BOOT_SEED,
          n: int = N_BOOT) -> tuple[float, float]:
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        s = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        stats.append(fn([sum(m[t] for t in s) / len(s) for m in means]))
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def score(cdir: Path, rows: Sequence[str] | None = None) -> dict[str, Any]:
    occ = [o for o in read_gz(cdir / "occurrences.jsonl.gz") if o["gold"] is not None]
    summ = json.loads((cdir / "corpus.json").read_text())
    names = list(rows or [])
    if not names:
        found = {p.name[len("answers_"):].split(".")[0] for p in cdir.glob("answers_*.jsonl*")}
        names = [r for r in ROWS if r in found] + sorted(found - set(ROWS))
    answers = {n: load_answers(cdir, n) for n in names}
    res: dict[str, Any] = {"registered": REGISTERED, "labels": LABELS, "coverage": summ["coverage"],
                           "n_questions": summ["n_questions"], "rows": names,
                           "answered": {n: {"questions": len(a), "errors": sum(bool(x.get("error")) for x in a.values()),
                                            "invalid": sum(x.get("valid") is False for x in a.values())}
                                        for n, a in answers.items()},
                           "strata": {}}
    strata = [(d, p) for d in DOMAINS for p in DPS] + [("all", p) for p in DPS]
    for dom, dp in strata:
        o = [x for x in occ if x["dp"] == dp and (dom == "all" or x["domain"] == dom)]
        if not o:
            continue
        key = f"{dom}/{dp}"
        s: dict[str, Any] = {"occurrences": len(o), "tasks": len({x["task"] for x in o}),
                             "questions": len({x["qhash"] for x in o}), "rows": {}, "paired": {}, "jev_subset": {}}
        tm = {n: _task_means(o, lambda x, n=n: _agree(answers[n].get(x["qhash"]), x["gold"])) for n in names}
        tasks = sorted({x["task"] for x in o})
        for n in names:
            est = sum(tm[n][t] for t in tasks) / len(tasks)
            lo, hi = _boot(tasks, [tm[n]], lambda v: v[0])
            s["rows"][n] = {"agree": est, "ci": [lo, hi],
                            "answered": sum(x["qhash"] in answers[n] for x in o) / len(o)}
        if "frozen" in names:
            for n in names:
                if n == "frozen":
                    continue
                d = sum(tm["frozen"][t] - tm[n][t] for t in tasks) / len(tasks)
                lo, hi = _boot(tasks, [tm["frozen"], tm[n]], lambda v: v[0] - v[1])
                s["paired"][f"frozen_vs_{n}"] = {"delta": d, "ci": [lo, hi]}
        if dp == "tool_select":
            top, cnt = Counter(x["gold"] for x in o).most_common(1)[0]
            pm = _task_means(o, lambda x: float(x["gold"] == top))
            s["majority"] = {"label": top, "agree": sum(pm[t] for t in tasks) / len(tasks),
                             "occurrence_share": cnt / len(o)}
        oj = [x for x in o if x["jev_logged"] is not None]
        if oj:
            tj = sorted({x["task"] for x in oj})
            jm = _task_means(oj, lambda x: float(x["jev_logged"] == x["gold"]))
            s["jev_subset"] = {"occurrences": len(oj), "tasks": len(tj),
                               "jev_logged": {"agree": sum(jm[t] for t in tj) / len(tj),
                                              "ci": list(_boot(tj, [jm], lambda v: v[0]))}}
            for n in names:
                nm = _task_means(oj, lambda x, n=n: _agree(answers[n].get(x["qhash"]), x["gold"]))
                d = sum(nm[t] - jm[t] for t in tj) / len(tj)
                s["jev_subset"][n] = {"agree": sum(nm[t] for t in tj) / len(tj), "delta_vs_jev": d,
                                      "delta_ci": list(_boot(tj, [nm, jm], lambda v: v[0] - v[1]))}
        res["strata"][key] = s
    return res


def to_markdown(r: Mapping[str, Any]) -> str:
    pct = lambda v: f"{100 * v:.1f}"                                   # noqa: E731
    ci = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"            # noqa: E731
    L = ["# τ² external reference agreement (A7.13 (iv); descriptive)", "",
         f"Labels: {r['labels']}. Agreement with these labels is not oracle accuracy and is not evidence of τ² "
         "success or savings. Task-weighted means (τ² task, trials and arms pooled) with task-bootstrap 95% "
         "intervals.", "", "## Coverage", "",
         "| stratum | logged | occurrences (state logged) | labelled | coverage | tasks | distinct labelled questions |",
         "|---|---|---|---|---|---|---|"]
    for k, c in r["coverage"].items():
        L.append(f"| {k} | {c['logged']} | {c['occurrences']} | {c['labelled']} | {c['coverage']} | {c['tasks']} | "
                 f"{c['distinct_labelled_questions']} |")
    L += ["", "## Agreement", "", "| stratum | " + " | ".join(r["rows"]) + " | majority label | frozen − others |",
          "|---|" + "---|" * (len(r["rows"]) + 2)]
    for k, s in r["strata"].items():
        cells = [f"{pct(s['rows'][n]['agree'])} [{pct(s['rows'][n]['ci'][0])}, {pct(s['rows'][n]['ci'][1])}]"
                 for n in r["rows"]]
        maj = f"{pct(s['majority']['agree'])} ({s['majority']['label']})" if "majority" in s else "–"
        pr = "; ".join(f"{k2.split('_vs_')[1]} {100 * v['delta']:+.1f} {ci(v['ci'])}" for k2, v in s["paired"].items())
        L.append(f"| {k} | " + " | ".join(cells) + f" | {maj} | {pr or '–'} |")
    L += ["", "## On the occurrences where Jev's live answer was logged", "",
          "| stratum | n | Jev (logged) | " + " | ".join(f"{n} (Δ vs Jev)" for n in r["rows"]) + " |",
          "|---|---|---|" + "---|" * len(r["rows"])]
    for k, s in r["strata"].items():
        j = s.get("jev_subset") or {}
        if not j:
            continue
        cells = [f"{pct(j[n]['agree'])} ({100 * j[n]['delta_vs_jev']:+.1f} {ci(j[n]['delta_ci'])})" for n in r["rows"]]
        L.append(f"| {k} | {j['occurrences']} | {pct(j['jev_logged']['agree'])} | " + " | ".join(cells) + " |")
    L += ["", "Answered: " + "; ".join(f"{n} {a['questions']} questions ({a['errors']} errors, {a['invalid']} invalid)"
                                       for n, a in r["answered"].items()), ""]
    return "\n".join(L)


def to_tex(r: Mapping[str, Any]) -> str:
    L = ["% AUTO-GENERATED by scripts/tau2_reference_replay.py score -- do not edit by hand.",
         "\\makeatletter",
         "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
         "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}", "\\makeatother"]
    d = lambda k, v: L.append(f"\\defres{{t2ref/{k}}}{{{v}}}")          # noqa: E731
    p = lambda v: f"{100 * v:.1f}"                                     # noqa: E731
    s_ = lambda v: f"{100 * v:+.1f}"                                    # noqa: E731
    d("n_questions", f"{r['n_questions']:,}".replace(",", "{,}"))
    for k, c in r["coverage"].items():
        d(f"{k}/labelled", c["labelled"]); d(f"{k}/occurrences", c["occurrences"])
        d(f"{k}/coverage", p(c["coverage"] or 0)); d(f"{k}/tasks", c["tasks"])
    for k, s in r["strata"].items():
        d(f"{k}/n", s["occurrences"]); d(f"{k}/tasks", s["tasks"])
        for n, v in s["rows"].items():
            d(f"{k}/{n}/agree", p(v["agree"])); d(f"{k}/{n}/ci/lo", p(v["ci"][0])); d(f"{k}/{n}/ci/hi", p(v["ci"][1]))
        for n, v in s["paired"].items():
            d(f"{k}/{n}/delta", s_(v["delta"])); d(f"{k}/{n}/ci/lo", s_(v["ci"][0])); d(f"{k}/{n}/ci/hi", s_(v["ci"][1]))
        if "majority" in s:
            d(f"{k}/majority/agree", p(s["majority"]["agree"]))
            d(f"{k}/majority/label", s["majority"]["label"].replace("_", "\\_"))
        j = s.get("jev_subset") or {}
        if j:
            d(f"{k}/jevsub/n", j["occurrences"]); d(f"{k}/jevsub/jev/agree", p(j["jev_logged"]["agree"]))
            for n in r["rows"]:
                if n in j:
                    d(f"{k}/jevsub/{n}/agree", p(j[n]["agree"])); d(f"{k}/jevsub/{n}/delta", s_(j[n]["delta_vs_jev"]))
                    d(f"{k}/jevsub/{n}/ci/lo", s_(j[n]["delta_ci"][0])); d(f"{k}/jevsub/{n}/ci/hi", s_(j[n]["delta_ci"][1]))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ CLI
def cmd_build(a: argparse.Namespace) -> int:
    c = build_corpus(Path(a.runs))
    summ = write_corpus(c, Path(a.out))
    print(json.dumps({k: summ[k] for k in ("n_questions", "coverage", "option_list_mismatches")}, indent=1))
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    from jevharness.decision import make_backend

    cdir = Path(a.corpus)
    summ = json.loads((cdir / "corpus.json").read_text())
    qpath = cdir / "questions.jsonl.gz"
    if hashlib.sha256(qpath.read_bytes()).hexdigest() != summ["questions_sha256"]:
        raise SystemExit("questions.jsonl.gz differs from corpus.json's sha256")
    qs, suffix = shard_of(read_gz(qpath), a.shard)
    out = cdir / f"answers_{a.name}{suffix}.jsonl"
    done = set(load_answers(cdir, a.name)) if out.exists() else set()
    backend = make_backend(json.loads(a.s1), cache=a.cache)
    (cdir / f"replay_{a.name}{suffix}.json").write_text(json.dumps({
        "registered": REGISTERED, "backend": backend.describe(), "cache": a.cache, "shard": a.shard,
        "order": "sorted question hash (shard i of n = every n-th from i)",
        "batching": "one question per request (the harness tournament for Choices wider than the backend's max_options)",
        "retries": 2, "n_questions": len(qs), "resumed_with": len(done & {q['qhash'] for q in qs})},
        indent=1, default=str))
    with open(out, "a") as f:
        def write(rec: dict) -> None:
            f.write(json.dumps(rec) + "\n")
            f.flush()

        replay(qs, backend, retries=2, on_answer=write, done=done)
    print(f"answers -> {out}", file=sys.stderr)
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    cdir = Path(a.corpus)
    r = score(cdir, a.rows.split(",") if a.rows else None)
    (cdir / "t2ref.json").write_text(json.dumps(r, indent=1) + "\n")
    (cdir / "t2ref.md").write_text(to_markdown(r))
    if a.tex:
        Path(a.tex).write_text(to_tex(r))
    print(to_markdown(r))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--runs", default="runs")
    b.add_argument("--out", default="results/stage2/t2ref")
    r = sub.add_parser("replay")
    r.add_argument("--corpus", default="results/stage2/t2ref")
    r.add_argument("--name", required=True)
    r.add_argument("--s1", required=True)
    r.add_argument("--cache", default="cache/s1_t2ref_replay.sqlite")
    r.add_argument("--shard", default=None, help="i/n: every n-th question of the sorted list, from i")
    s = sub.add_parser("score")
    s.add_argument("--corpus", default="results/stage2/t2ref")
    s.add_argument("--rows", default=None)
    s.add_argument("--tex", default=None)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "replay": cmd_replay, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
