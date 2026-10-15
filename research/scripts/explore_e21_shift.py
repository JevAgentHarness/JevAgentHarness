#!/usr/bin/env python3
r"""E21 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E21: a shifted DecisionGym split
(DecisionGym-Shift)"): the shifted suite's L2 corpus (built as C from LLM-only and Jev REPLACE on 100 tasks), every
decision model's offline answers on it (and on two axis re-renderings), L3 REPLACE live on 200 tasks, and the readout.

    # operations (the GPU host runs its part through scripts/aws/e21_shift.sh; this container through scripts/e21_local.sh)
    python scripts/explore_e21_shift.py probe --log FILE                       # pause rule (registration (g))
    python scripts/explore_e21_shift.py watch --pid PID --runs runs --cap USD [--s1-cap USD]
    python scripts/explore_e21_shift.py spend | errored --run DIR | gate --smoke DIR --n N --cap USD
    python scripts/explore_e21_shift.py build [--runs runs]                    # the L2 corpus -> results/explore/e21_shift/c
    python scripts/explore_e21_shift.py variants --corpus DIR                  # surface-undone / distraction-undone questions
    python scripts/explore_e21_shift.py replay --corpus DIR --slug SLUG [--variant main|nosurf|nodist] [--s1 JSON]
    python scripts/explore_e21_shift.py quickc --corpus DIR --slug SLUG [--variant V]
    python scripts/explore_e21_shift.py ledger                                 # spend per provider (local records)
    python scripts/explore_e21_shift.py archive --live RUN...                  # slim copies -> results/explore/e21_shift/live
    python scripts/explore_e21_shift.py summarize                              # summary.{md,json} + paper/generated/e21_shift.tex
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/explore/e21_shift"
CDIR = OUT / "c"
TEX = ROOT / "paper/generated/e21_shift.tex"
CFG = ROOT / "configs/explore/e21_shift"
LABEL = ("exploratory (E21), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E21: a shifted DecisionGym split "
         "(DecisionGym-Shift)'; outside every Holm table")
PREFIX = "e21_"
L2_SEED0, L2_N, L3_SEED0, L3_N = 330000, 100, 330100, 200
OVERLOAD = ("503", "overload", "high demand", "unavailable", "resource_exhausted", "429")
VARIANTS = ("main", "nosurf", "nodist")
N_BOOT, BOOT_SEED = 2000, 0
SOURCES = (("llm_only", "llm_only"), ("jev_ts_replace_f", "jev_ts_replace_f"))
TRAINED = ("rl", "e12", "e12c", "e14a")
OFFSHELF = ("jev", "oaidec", "sl4b", "sl9b", "sl27b", "openjevhf", "intern")
# "E21 addendum 2: the untrained and SFT rows on the shift" (Deviations 2026-10-09): the training ladder's first two rows
# (the untrained Qwen3-4B and Qwen3-4B + SFT v1, served as their Stage-2 rows). They enter none of E21's registered
# readings (the lead and the three Kendall rows stay over TRAINED + OFFSHELF); one descriptive Kendall row adds them.
LADDER = ("base", "sft_v1")
LIVE_ROWS = ("llm_only", "jev", "rl", "e14a", "e12", "sl9b", "oaidec", "openjevhf", "sl27b", "sl4b", "intern", "e12c",
             "base", "sft_v1")
NAMES = {"jev": "Jev", "oaidec": "OpenAI Decisions API", "rl": "AgentDecision-4B", "e12": "AgentDecision-4B (mixed)",
         "e12c": "StartLux-4B (mixed)", "e14a": "0.8B scoring head", "sl4b": "StartLux-4B", "sl9b": "StartLux-9B",
         "sl27b": "StartLux-27B (Q8_0)", "openjevhf": "OpenJev-27B", "intern": "Intern-Decision-4B",
         "llm_only": "Agent LLM alone (Flash)", "base": "Qwen3-4B (untrained)", "sft_v1": "Qwen3-4B + SFT v1"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def jl(p: Path) -> list[dict[str, Any]]:
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


def backends() -> dict[str, dict[str, Any]]:
    import yaml

    return dict((yaml.safe_load((CFG / "backends.yaml").read_text()) or {}).get("backends") or {})


# ================================================================================================ live operations
def cmd_probe(a: argparse.Namespace) -> int:
    """One round of the pause rule: six one-word Flash calls on the direct API (E15's probe, E21's log)."""
    import explore_e15_baselines as E15

    p = Path(a.log)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# E21: direct Gemini API probes (Deviations 2026-10-08 'E21' (g)). One round = six one-word Gemini "
                     "3.5 Flash calls; start / resume after two consecutive 6/6 rounds.\n")
    return E15.cmd_probe(argparse.Namespace(log=str(p), dotenv=a.dotenv))


def episode_comp(t: Mapping[str, Any], comp: str) -> float:
    return float(((t.get("usage") or {}).get(comp) or {}).get("cost_usd") or 0.0)


def e21_runs(runs: Path) -> list[Path]:
    return sorted(p for p in Path(runs).glob(f"{PREFIX}*") if p.is_dir())


def run_spend(run: Path) -> dict[str, Any]:
    """Every record of every arm (errored attempts and re-runs included: all were paid); s2 = the agent LLM (direct
    Gemini), s1 = the decision model (Jev / OpenAI / served = 0)."""
    out = {"total": 0.0, "s2": 0.0, "s1": 0.0, "records": 0, "s1_by_arm": {}}
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        for t in jl(p):
            if isinstance(t, dict) and "task_id" in t:
                out["total"] += episode_comp(t, "total")
                out["s2"] += episode_comp(t, "s2")
                out["s1"] += episode_comp(t, "s1")
                out["s1_by_arm"][p.stem] = out["s1_by_arm"].get(p.stem, 0.0) + episode_comp(t, "s1")
                out["records"] += 1
    return out


def total_spend(runs: Path) -> dict[str, Any]:
    per = {p.name: run_spend(p) for p in e21_runs(runs)}
    return {"by_run": per, "s2": sum(v["s2"] for v in per.values()), "s1": sum(v["s1"] for v in per.values()),
            "total": sum(v["total"] for v in per.values()), "records": sum(v["records"] for v in per.values())}


def latest(run: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        d: dict[str, dict[str, Any]] = {}
        for t in jl(p):
            if isinstance(t, dict) and "task_id" in t:
                d[str(t["task_id"])] = t
        out[p.stem] = d
    return out


def overload_error(t: Mapping[str, Any]) -> bool:
    e = str(t.get("error") or "").lower()
    return bool(e) and any(k in e for k in OVERLOAD)


def cmd_errored(a: argparse.Namespace) -> int:
    print(sum(1 for d in latest(Path(a.run)).values() for t in d.values() if t.get("error")))
    return 0


def cmd_spend(a: argparse.Namespace) -> int:
    s = total_spend(Path(a.runs))
    if a.short:
        print(f"E21 runs: Gemini ${s['s2']:.3f}, System-1 ${s['s1']:.3f} in {s['records']} records")
    else:
        print(json.dumps(s, indent=1))
    return 0


def _kill(pid: int) -> None:
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        time.sleep(10)


def cmd_watch(a: argparse.Namespace) -> int:
    """The Gemini cap (s2 of every E21 run here) and an optional System-1 cap (-> STOP file), and unless --no-pause a
    pause on a new episode that errored with 503 / overload (-> <run>/PAUSED_overload.json); records are kept."""
    runs = Path(a.runs)
    seen = {p.name: {(arm, k) for arm, d in latest(p).items() for k, t in d.items() if t.get("error")}
            for p in e21_runs(runs)}
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{_now()} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(runs)
        if a.s1_arm:     # the System-1 cap of one provider: only the arms of that backend count
            s["s1"] = sum(v for r in s["by_run"].values() for arm, v in r["s1_by_arm"].items() if a.s1_arm in arm)
        extra = float(a.extra or 0.0)
        over = s["s2"] + extra >= a.cap or (a.s1_cap is not None and s["s1"] + float(a.s1_extra or 0) >= a.s1_cap)
        if over:
            Path(a.stop).write_text(json.dumps({"at": _now(), "spend": {k: s[k] for k in ("s2", "s1", "records")},
                                                "extra": extra, "cap": a.cap, "s1_cap": a.s1_cap}, indent=1))
            _kill(a.pid)
            print(f"{_now()} CAP REACHED: Gemini ${s['s2']:.2f} (+{extra:.2f}) / S1 ${s['s1']:.2f}; runner stopped",
                  flush=True)
            return 1
        if not a.no_pause:
            for p in e21_runs(runs):
                new = [(arm, k, t) for arm, d in latest(p).items() for k, t in d.items()
                       if t.get("error") and (arm, k) not in seen.get(p.name, set()) and overload_error(t)]
                if new:
                    (p / "PAUSED_overload.json").write_text(json.dumps(
                        {"at": _now(), "episodes": [{"arm": arm, "task": k, "error": str(t.get("error"))[:300]}
                                                    for arm, k, t in new]}, indent=1))
                    _kill(a.pid)
                    print(f"{_now()} PAUSED: {len(new)} overload error(s) in {p.name}; runner stopped", flush=True)
                    return 2
        print(f"{_now()} Gemini ${s['s2']:.3f} S1 ${s['s1']:.3f} ({s['records']} records)", flush=True)
        time.sleep(a.every)


def cmd_gate(a: argparse.Namespace) -> int:
    """Registration (e): spend so far + 1.2 x the smoke's $ per episode x the tasks still to run <= the side's cap;
    otherwise the first k tasks that fit (printed as n_fit)."""
    sm = run_spend(Path(a.smoke))
    tot = total_spend(Path(a.runs))
    per = sm["s2"] / sm["records"] if sm["records"] else float("inf")
    spent = tot["s2"] + float(a.extra or 0.0)
    per_task = per * a.arms
    n_fit = int(max(0.0, (a.cap - spent) / (1.2 * per_task))) if per > 0 and math.isfinite(per) else 0
    n_fit = min(a.n, n_fit)
    res = {"smoke": str(a.smoke), "smoke_records": sm["records"], "gemini_usd_per_episode": per, "n": a.n,
           "arms": a.arms, "projection_usd": spent + 1.2 * per_task * a.n, "spent_usd": spent, "cap_usd": a.cap,
           "n_fit": n_fit, "go": n_fit == a.n, "at": _now()}
    print(json.dumps(res))
    return 0 if n_fit > 0 else 1


# ================================================================================================ L2 corpus
def task_seed(task_id: str) -> int:
    m = re.search(r"-(\d{6})-d\d+", str(task_id))
    if not m:
        raise ValueError(f"no seed in task id {task_id!r}")
    return int(m.group(1))


def suite_kw() -> dict[str, Any]:
    import yaml

    exp = yaml.safe_load((CFG / "l2_sources.yaml").read_text())
    kw = dict(exp["suite"])
    kw.pop("name")
    return kw


def cmd_build(a: argparse.Namespace) -> int:
    """The L2 corpus exactly as C (A7.10 (c)): both sources' final valid trajectories, the 100 L2 tasks."""
    from jevharness.bench import stage2_corpus as SC
    from jevharness.bench.suites import load_suite

    run = Path(a.runs) / "e21_l2"
    facs = {f().task_id: f for f in load_suite("jevgym", **{**suite_kw(), "n": L2_N, "seed0": L2_SEED0})}
    tasks = list(facs)
    sources = [(label, run / f"{arm}.jsonl") for label, arm in SOURCES]
    corpus = SC.build_corpus(sources, tasks, facs)
    corpus["registered"] = "PREREG_A7_STAGE2 Deviations 2026-10-08 'E21' (d) 2 (built as A7.10 (c))"
    paths = SC.write_corpus(corpus, Path(a.out))
    print(json.dumps({"summary": corpus["summary"], "files": paths}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------- axis re-renderings
def _canon_texts(seed: int) -> tuple[Any, Any]:
    """The task under all axes and the same task with the surface axis off (distraction and structure on)."""
    from jevharness.bench.suites.jevgym_shift import JevGymShiftEnv

    kw = suite_kw()
    kw.pop("shift", None)
    kw.pop("n", None)
    kw.pop("seed0", None)
    return JevGymShiftEnv(seed, **kw), JevGymShiftEnv(seed, surface=False, **kw)


def _reply_pairs(sh: Any, ca: Any) -> list[tuple[str, str]]:
    from jevharness.bench.suites import jevgym_shift as S

    pairs = [(S.REPEAT_REPLY[True], S.REPEAT_REPLY[False]), (S.NOTHING_REPLY[True], S.NOTHING_REPLY[False])]
    a, b = sh.core, ca.core
    if getattr(a, "replies", None):
        pairs += [(f"Customer: {a.replies[k]}", f"Customer: {b.replies[k]}") for k in a.replies]
    if a.family == "clarify" and a.reply:
        pairs.append((f"Customer: {a.reply}", f"Customer: {b.reply}"))
    return [(x, y) for x, y in pairs if x and x != y]


def undo_surface(state: str, sh: Any, ca: Any) -> str:
    """The state as the surface-off task of the same seed renders it: the generator's instruction, policy line and
    replies, canonical tool names (aliases are whole identifiers: their near-duplicates' longer names are untouched)."""
    from jevharness.bench.suites import jevgym_shift as S

    s = state.replace(f"TASK: {sh.instruction}", f"TASK: {ca.instruction}", 1)
    s = s.replace(S.OPENING[True] + S.POLICY_SHIFT.format(ask=S.ALIASES[S.ASK][0]), S.OPENING[False] + S.ASK_POLICY)
    for x, y in _reply_pairs(sh, ca):
        s = s.replace(x, y)
    for c, (alias, _, _) in S.ALIASES.items():
        s = re.sub(rf"(?<![\w]){re.escape(alias)}(?![\w])", c, s)
    return s


def cmd_variants(a: argparse.Namespace) -> int:
    """Registration (d) 5: questions_nosurf / questions_nodist (same qhash keys; ``map`` = answer -> alias)."""
    from jevharness.bench import stage2_corpus as SC
    from jevharness.bench.suites import jevgym as J
    from jevharness.bench.suites import jevgym_shift as S
    from jevharness.harness.tools import tool_desc

    cdir = Path(a.corpus)
    qs = SC.read_jsonl_gz(cdir / "questions.jsonl.gz")
    occ = SC.read_jsonl_gz(cdir / "occurrences.jsonl.gz")
    task_of = {o["qhash"]: o["task"] for o in occ}
    canon_desc = {t["function"]["name"]: tool_desc(t) for t in J.CORE_TOOLS + [J.ASK_TOOL]}
    added = ({t["function"]["name"] for t in S.shift_distractors(60, "near")}
             - {t["function"]["name"] for t in J.make_distractors(60, "near")})
    to_canon = {alias: c for c, (alias, _, _) in S.ALIASES.items()}
    envs: dict[int, tuple[Any, Any]] = {}
    nos, nod, bad = [], [], []
    for q in qs:
        seed = task_seed(task_of[q["qhash"]])
        if seed not in envs:
            envs[seed] = _canon_texts(seed)
        sh, ca = envs[seed]
        st = undo_surface(q["state"], sh, ca)
        left = [al for al in to_canon if re.search(rf"(?<![\w]){re.escape(al)}(?![\w])", st)]
        if left or f"TASK: {ca.instruction}" not in st:
            bad.append({"qhash": q["qhash"], "aliases_left": left, "instruction": f"TASK: {ca.instruction}" in st})
        opts = [to_canon.get(o, o) for o in q["options"]]
        descs = [canon_desc[c] if o in to_canon else d for o, c, d in zip(q["options"], opts, q["descriptions"] or
                                                                            [""] * len(opts))]
        nos.append({"qhash": q["qhash"], "state": st, "options": opts, "descriptions": descs,
                    "map": {c: o for o, c in zip(q["options"], opts) if c != o}})
        keep = [i for i, o in enumerate(q["options"]) if o not in added]
        nod.append({"qhash": q["qhash"], "state": q["state"], "options": [q["options"][i] for i in keep],
                    "descriptions": [(q["descriptions"] or [""] * len(q["options"]))[i] for i in keep], "map": {}})
    for name, rows in (("nosurf", nos), ("nodist", nod)):
        with gzip.open(cdir / f"questions_{name}.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"registered": "PREREG_A7_STAGE2 Deviations 2026-10-08 'E21' (d) 5", "questions": len(qs),
            "nosurf_render_failures": bad[:20], "n_nosurf_render_failures": len(bad), "added_distractors": len(added),
            "nodist_options_mean": float(np.mean([len(r["options"]) for r in nod])) if nod else None,
            "main_options_mean": float(np.mean([len(q["options"]) for q in qs])) if qs else None, "written": _now()}
    (cdir / "variants.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "nosurf_render_failures"}))
    return 0 if not bad else 2


def questions_of(cdir: Path, variant: str) -> list[dict[str, Any]]:
    from jevharness.bench import stage2_corpus as SC

    return SC.read_jsonl_gz(cdir / ("questions.jsonl.gz" if variant == "main" else f"questions_{variant}.jsonl.gz"))


def answers_path(cdir: Path, slug: str, variant: str) -> Path:
    return cdir / (f"answers_{slug}.jsonl" if variant == "main" else f"answers_{slug}__{variant}.jsonl")


def cmd_replay(a: argparse.Namespace) -> int:
    """Every distinct question answered through ``stage2_corpus.replay`` (one question per request, the harness
    knockout above the backend's read width, two retries), ``workers`` in flight; resumes; ``--cap-usd`` stops a hosted
    backend once its logged cost over all its E21 answer files reaches the cap (checked between chunks of 25)."""
    from jevharness.bench import stage2_corpus as SC
    from jevharness.decision import make_backend

    cdir = Path(a.corpus)
    qs = questions_of(cdir, a.variant)
    out = answers_path(cdir, a.slug, a.variant)
    done = {r["qhash"] for r in jl(out) if not r.get("error")} if out.exists() else set()
    spec = dict(json.loads(a.s1) if a.s1 else backends()[a.slug])
    spec.pop("cache", None)      # a live-config key (the runner's System-1 cache switch); the replay cache is --cache
    backend = make_backend(spec, cache=a.cache)
    meta = cdir / (f"replay_{a.slug}.json" if a.variant == "main" else f"replay_{a.slug}__{a.variant}.json")
    meta.write_text(json.dumps({
        "registered": "PREREG_A7_STAGE2 Deviations 2026-10-08 'E21' (d) 3" + (" / (d) 5" if a.variant != "main" else ""),
        "variant": a.variant, "backend": backend.describe(), "cache": a.cache, "workers": a.workers,
        "order": "sorted question hash, sharded over workers", "batching": "one question per request (the harness "
        "tournament for Choices wider than the backend's max_options)", "retries": 2, "n_questions": len(qs),
        "resumed_with": len(done), "written": _now()}, indent=1, default=str))
    todo = sorted((q for q in qs if q["qhash"] not in done), key=lambda x: x["qhash"])
    spent = sum(float(r.get("cost_usd") or 0) for v in VARIANTS for p in [answers_path(cdir, a.slug, v)] if p.exists()
                for r in jl(p))
    lock = threading.Lock()
    state = {"spent": spent, "stop": False}
    f = open(out, "a")

    def write(rec: dict[str, Any]) -> None:
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            state["spent"] += float(rec.get("cost_usd") or 0.0)
            if a.cap_usd is not None and state["spent"] >= a.cap_usd:
                state["stop"] = True

    def work(shard: list[dict[str, Any]]) -> None:
        for i in range(0, len(shard), 25):
            if state["stop"]:
                return
            SC.replay(shard[i:i + 25], backend, retries=2, on_answer=write)

    shards = [todo[i::max(1, a.workers)] for i in range(max(1, a.workers))]
    ths = [threading.Thread(target=work, args=(s,)) for s in shards if s]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    f.close()
    recs = {r["qhash"]: r for r in jl(out)}
    n_err = sum(1 for q in qs if (recs.get(q["qhash"]) or {}).get("error") or q["qhash"] not in recs)
    print(f"answers -> {out}: {len(recs)} of {len(qs)}; errors / missing {n_err}; logged ${state['spent']:.4f}"
          + ("; STOPPED at the cap" if state["stop"] else ""), file=sys.stderr)
    return 0 if n_err == 0 else (3 if state["stop"] else 2)


def score_answers(occ: Sequence[Mapping[str, Any]], qs: Mapping[str, Mapping[str, Any]],
                  ans: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """occurrence index -> correctness (answers of a re-rendering are mapped back to the alias names first)."""
    from jevharness.bench import stage2_corpus as SC

    out = {}
    for i, o in enumerate(occ):
        r = dict(ans.get(o["qhash"]) or {})
        m = (qs.get(o["qhash"]) or {}).get("map") or {}
        if r.get("answer") in m:
            r["answer"] = m[r["answer"]]
        out[i] = float(SC.answer_correct(o, r if r else None))
    return out


def cmd_quickc(a: argparse.Namespace) -> int:
    from jevharness.bench import stage2_corpus as SC

    cdir = Path(a.corpus)
    ap = answers_path(cdir, a.slug, a.variant)
    if not ap.exists():
        print("no answers")
        return 1
    occ = SC.read_jsonl_gz(cdir / "occurrences.jsonl.gz")
    elig = set(json.loads((cdir / "corpus.json").read_text())["eligible_tasks"])
    qs = {q["qhash"]: q for q in questions_of(cdir, a.variant)}
    ans = {r["qhash"]: r for r in jl(ap)}
    ok = score_answers(occ, qs, ans)
    per: dict[str, list[float]] = defaultdict(list)
    for i, o in enumerate(occ):
        if o["task"] in elig:
            per[o["task"]].append(ok[i])
    acc = float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")
    err = sum(1 for r in ans.values() if r.get("error"))
    print(f"{a.variant} acc {100 * acc:.1f} % over {len(per)} tasks; {len(ans)} answers, {err} errors, "
          f"{sum(1 for r in ans.values() if r.get('refusal'))} refusals")
    return 0


def cmd_ledger(a: argparse.Namespace) -> int:
    """Spend per provider from this container's records (live runs + replay answer files)."""
    s = total_spend(Path(a.runs))
    rep: dict[str, float] = {}
    cdir = Path(a.corpus)
    for p in sorted(cdir.glob("answers_*.jsonl*")):
        slug = p.name[len("answers_"):].split(".jsonl")[0].split("__")[0]
        rep[slug] = rep.get(slug, 0.0) + sum(float(r.get("cost_usd") or 0) for r in jl(p))
    s1_jev = sum(v["s1_by_arm"].get("jev_ts_replace_f", 0.0) for v in s["by_run"].values())
    s1_oai = sum(v["s1_by_arm"].get("oaidec_ts_replace_f", 0.0) for v in s["by_run"].values())
    res = {"gemini_live": s["s2"], "jev": s1_jev + rep.get("jev", 0.0), "openai": s1_oai + rep.get("oaidec", 0.0),
           "jev_live": s1_jev, "jev_replay": rep.get("jev", 0.0), "openai_live": s1_oai,
           "openai_replay": rep.get("oaidec", 0.0), "records": s["records"], "at": _now()}
    print(json.dumps(res) if not a.pretty else json.dumps(res, indent=1))
    return 0


# ================================================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("probe")
    p.add_argument("--log", default=str(OUT / "api_probe_log.txt"))
    p.add_argument("--dotenv", action="store_true")
    p = sp.add_parser("watch")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--extra", type=float, default=0.0, help="Gemini spent outside --runs on this side (probes, smokes)")
    p.add_argument("--s1-cap", type=float, default=None)
    p.add_argument("--s1-extra", type=float, default=0.0)
    p.add_argument("--s1-arm", default=None, help="count System-1 spend only of arms whose name contains this")
    p.add_argument("--stop", default="runs/e21_STOP.json")
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--no-pause", action="store_true")
    p = sp.add_parser("spend")
    p.add_argument("--runs", default="runs")
    p.add_argument("--short", action="store_true")
    p = sp.add_parser("errored")
    p.add_argument("--run", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--smoke", required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--n", type=int, required=True, help="tasks still to run")
    p.add_argument("--arms", type=int, default=1, help="arms per task")
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--extra", type=float, default=0.0)
    p = sp.add_parser("build")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default=str(CDIR))
    p = sp.add_parser("variants")
    p.add_argument("--corpus", default=str(CDIR))
    p = sp.add_parser("replay")
    p.add_argument("--corpus", default=str(CDIR))
    p.add_argument("--slug", required=True)
    p.add_argument("--variant", default="main", choices=VARIANTS)
    p.add_argument("--s1", default=None, help="backend JSON (default: configs/explore/e21_shift/backends.yaml)")
    p.add_argument("--cache", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cap-usd", type=float, default=None)
    p = sp.add_parser("quickc")
    p.add_argument("--corpus", default=str(CDIR))
    p.add_argument("--slug", required=True)
    p.add_argument("--variant", default="main", choices=VARIANTS)
    p = sp.add_parser("ledger")
    p.add_argument("--runs", default="runs")
    p.add_argument("--corpus", default=str(CDIR))
    p.add_argument("--pretty", action="store_true")
    for name, fn in (("archive", None), ("summarize", None)):
        p = sp.add_parser(name)
        p.add_argument("--runs", default="runs")
        p.add_argument("--live", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "archive":
        return cmd_archive(a)
    if a.cmd == "summarize":
        return cmd_summarize(a)
    return {"probe": cmd_probe, "watch": cmd_watch, "spend": cmd_spend, "errored": cmd_errored, "gate": cmd_gate,
            "build": cmd_build, "variants": cmd_variants, "replay": cmd_replay, "quickc": cmd_quickc,
            "ledger": cmd_ledger}[a.cmd](a)


def slim_live(run: Path) -> dict[str, Any]:
    """Per arm, per task: the record that decides the task (explore_common.load_latest) in a slim form."""
    import explore_common as EC
    from explore_startlux_live import own_accuracy  # type: ignore[import-not-found]

    out: dict[str, Any] = {}
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        lat, att = EC.load_latest([p])
        own = own_accuracy([t for t in lat.values() if not t.get("error")])
        rows = {}
        for k, t in lat.items():
            refusals = sum(1 for st in (t.get("steps") or []) for d in (st.get("decisions") or [])
                           if (d.get("gold_aux") or {}).get("s1_refusal"))
            rows[k] = {"success": EC.success(t), "error": bool(t.get("error")),
                       "error_text": str(t.get("error") or "")[:200] or None, "cost": EC.cost(t),
                       "s1_cost": episode_comp(t, "s1"), "s2_cost": episode_comp(t, "s2"),
                       "wall_s": float(t.get("wall_ms") or 0) / 1000, "steps": len(t.get("steps") or []),
                       "s1_calls": EC.s_calls(t, "s1"), "s2_calls": EC.s_calls(t, "s2"), "attempts": att.get(k, 1),
                       "refusals": refusals, "own_acc": own.get(k)}
        out[p.stem] = rows
    return out


def cmd_archive(a: argparse.Namespace) -> int:
    """Slim copies of the live runs (one record per task) -> results/explore/e21_shift/live/<run>.json."""
    import shutil

    done = []
    for run in a.live or []:
        rp = Path(run)
        if not rp.is_dir():
            continue
        dst = OUT / "live" / f"{rp.name}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"run": rp.name, "arms": slim_live(rp), "spend": run_spend(rp), "written": _now()},
                                  indent=1) + "\n")
        if (rp / "experiment.yaml").exists():
            shutil.copyfile(rp / "experiment.yaml", OUT / "live" / f"{rp.name}.experiment.yaml")
        done.append(dst)
    print("\n".join(str(p.relative_to(ROOT)) for p in done))
    return 0


# ================================================================================================ readout
ASK_ALIAS = "request_clarification"
LIVE_SOURCES = {"llm_only": ("e21_live_local", "llm_only"), "jev": ("e21_live_local", "jev_ts_replace_f"),
                "oaidec": ("e21_live_oaidec", "oaidec_ts_replace_f"), "rl": ("e21_live_rl", "rl_ts_replace_f"),
                "e14a": ("e21_live_e14a", "e14a_ts_replace_f"), "e12": ("e21_live_e12", "e12_ts_replace_f"),
                "sl9b": ("e21_live_sl9b", "sl9b_ts_replace_f"),
                # the addendum's rows ("E21 addendum: shift live rows for every model")
                **{s: (f"e21_live_{s}", f"{s}_ts_replace_f") for s in ("openjevhf", "sl27b", "sl4b", "intern", "e12c")},
                # "E21 addendum 2" (the training ladder's untrained and SFT rows)
                **{s: (f"e21_live_{s}", f"{s}_ts_replace_f") for s in LADDER}}
SUBSETS = ("all", "mustask", "offplan", "renamed", "newkind", "genkind", "askany")


def kind_of_task(task: str) -> str:
    m = re.match(r"jevgym-shift-[a-z]+-([a-z_.]+)-\d+", str(task))
    return m.group(1) if m else "?"


def occ_subsets(o: Mapping[str, Any]) -> set[str]:
    acc = list(((o.get("gold_aux") or {}).get("acceptable_tools")) or [])
    k = kind_of_task(o["task"])
    out = {"all"}
    if acc == [ASK_ALIAS]:
        out.add("mustask")
    if ASK_ALIAS in acc:
        out.add("askany")
    if o.get("stratum") == "after_off_plan":
        out.add("offplan")
    if acc and "finish" not in acc:
        out.add("renamed")
    out.add("newkind" if k in ("notify", "clarify.ask_twice", "clarify.ask_after_lookup") else "genkind")
    return out


def in_domain() -> dict[str, dict[str, Any]]:
    """The logged in-domain numbers: C accuracy (task means, 200 tasks) and live success on the first 200 Stage-2 tasks."""
    out: dict[str, dict[str, Any]] = {}
    sl = json.loads((ROOT / "results/explore/startlux_live/summary.json").read_text())["rows"]
    for slug, key in (("jev", "jev"), ("rl", "frozen"), ("sl4b", "sl4b"), ("sl9b", "sl9b"), ("sl27b", "sl27b"),
                      ("intern", "intern")):
        r = sl.get(key) or {}
        out[slug] = {"c": r.get("corpus_acc"), "live": r.get("live_success"), "own": r.get("own_acc"),
                     "source": "explore/startlux_live"}
    e12 = json.loads((ROOT / "results/explore/e12/summary.json").read_text())
    out["e12"] = {"c": e12["c"]["e12"]["c_acc"], "live": e12["live"]["e12"]["live_success"],
                  "own": e12["live"]["e12"].get("own_acc"), "source": "E12"}
    out["e12c"] = {"c": e12["e12c"]["c"]["e12c"]["c_acc"], "live": e12["e12c"]["live"]["e12c"]["live_success"],
                   "own": e12["e12c"]["live"]["e12c"].get("own_acc"), "source": "E12c"}
    e14 = json.loads((ROOT / "results/explore/e14_small/summary.json").read_text())
    out["e14a"] = {"c": e14["corpus"]["rows"]["e14a"]["c_acc"], "live": e14["live_a"]["e14a"]["live_success"],
                   "own": e14["live_a"]["e14a"].get("own_acc"), "source": "E14"}
    e15 = json.loads((ROOT / "results/explore/e15_baselines/summary.json").read_text())["models"]
    for slug in ("oaidec", "openjevhf"):
        m = e15.get(slug) or {}
        l3 = m.get("l3") or {}
        out[slug] = {"c": (m.get("l2") or {}).get("corpus_acc"), "live": l3.get("success", l3.get("live_success")),
                     "own": l3.get("own_acc"), "source": "E15"}
    p = ROOT / "runs/stage2_jevgym/llm_only.jsonl"
    if p.exists():
        import explore_common as EC

        tasks = json.loads((ROOT / "configs/eval/jevgym_stage2_tasks.json").read_text())["task_ids"][:200]
        lat, _ = EC.load_latest([p])
        y = [float(EC.success(lat[t])) for t in tasks if t in lat and not lat[t].get("error")]
        out["llm_only"] = {"c": None, "live": float(np.mean(y)) if y else None, "live_n": len(y),
                           "source": "Stage 2 llm_only, first 200 tasks"}
    # "E21 addendum 2" (g): the ladder rows' logged Stage-2 numbers (C on 200 tasks, live on the first 100 Stage-2 tasks)
    s2 = ROOT / "results/stage2/stage2.json"
    if s2.exists():
        rl = (json.loads(s2.read_text()).get("descriptive") or {}).get("stage2_rl") or {}
        for slug in LADDER:
            r = rl.get(slug) or {}
            if r:
                out[slug] = {"c": r.get("corpus_acc"), "live": r.get("live_success"), "live_n": r.get("live_n"),
                             "own": r.get("own_acc"), "source": f"Stage 2 stage2_rl/{slug} (live: first 100 tasks)"}
    return out


def l2_rows(cdir: Path) -> dict[str, Any]:
    import explore_common as EC
    from jevharness.bench import stage2_corpus as SC

    occ = SC.read_jsonl_gz(cdir / "occurrences.jsonl.gz")
    meta = json.loads((cdir / "corpus.json").read_text())
    elig = [str(t) for t in meta["eligible_tasks"]]
    eset = set(elig)
    occ = [o for o in occ if o["task"] in eset]
    subs = [occ_subsets(o) for o in occ]
    qsets = {v: {q["qhash"]: q for q in questions_of(cdir, v)} for v in VARIANTS
             if (cdir / ("questions.jsonl.gz" if v == "main" else f"questions_{v}.jsonl.gz")).exists()}
    corpus = {"tasks": len(elig), "occurrences": len(occ), "questions": len({o["qhash"] for o in occ}),
              "excluded": meta.get("excluded"), "by_source": dict(Counter(o["source"] for o in occ)),
              "subset_n": {s: sum(1 for x in subs if s in x) for s in SUBSETS},
              "subset_tasks": {s: len({o["task"] for o, x in zip(occ, subs) if s in x}) for s in SUBSETS},
              "options_mean": float(np.mean([len(q["options"]) for q in qsets["main"].values()])) if "main" in qsets else None,
              "kinds": dict(Counter(kind_of_task(t) for t in elig))}
    per: dict[str, dict[str, Any]] = {}
    taskacc: dict[tuple[str, str, str], dict[str, float]] = {}
    for p in sorted(cdir.glob("answers_*.jsonl*")):
        stem = p.name[len("answers_"):].split(".jsonl")[0]
        slug, _, var = stem.partition("__")
        var = var or "main"
        if var not in qsets:
            continue
        ans = {r["qhash"]: r for r in jl(p)}
        ok = score_answers(occ, qsets[var], ans)
        row: dict[str, Any] = {"answers": len(ans), "errors": sum(1 for r in ans.values() if r.get("error")),
                               "refusals": sum(1 for r in ans.values() if r.get("refusal")),
                               "missing": sum(1 for q in qsets[var] if q not in ans),
                               "cost_usd": sum(float(r.get("cost_usd") or 0) for r in ans.values()),
                               "latency_ms_median": float(np.median([float(r.get("latency_ms") or 0) for r in ans.values()]))
                               if ans else None}
        for s in SUBSETS:
            t: dict[str, list[float]] = defaultdict(list)
            for i, o in enumerate(occ):
                if s in subs[i]:
                    t[o["task"]].append(ok[i])
            tm = {k: float(np.mean(v)) for k, v in t.items()}
            taskacc[(slug, var, s)] = tm
            row[s] = {"acc": float(np.mean(list(tm.values()))) if tm else None, "tasks": len(tm),
                      "n": sum(len(v) for v in t.values()), "occ_acc": float(np.mean([x for v in t.values() for x in v]))
                      if t else None}
        per.setdefault(slug, {})[var] = row

    # post hoc, descriptive: what a wrong main-rendering answer was (registration (f) names no error breakdown)
    from jevharness.bench.suites import jevgym as J
    from jevharness.bench.suites import jevgym_shift as S

    added = ({t["function"]["name"] for t in S.shift_distractors(60, "near")}
             - {t["function"]["name"] for t in J.make_distractors(60, "near")})
    aliases = {a for a, _, _ in S.ALIASES.values()}
    for p in sorted(cdir.glob("answers_*.jsonl*")):
        slug = p.name[len("answers_"):].split(".jsonl")[0]
        if "__" in slug or slug not in per:
            continue
        ans = {r["qhash"]: r for r in jl(p)}
        cats: Counter = Counter()
        ma: Counter = Counter()
        for o, x in zip(occ, subs):
            r = ans.get(o["qhash"]) or {}
            a = r.get("answer") if not r.get("error") and r.get("valid", True) else None
            accset = set(((o.get("gold_aux") or {}).get("acceptable_tools")) or [])
            if a in accset:
                continue
            cat = ("no answer" if a is None else "finish" if a == "finish" else "ask" if a == ASK_ALIAS
                   else "other core tool" if a in aliases else "added distractor" if a in added
                   else "generator distractor")
            cats[cat] += 1
            if "mustask" in x:
                ma[cat] += 1
        per[slug]["main"]["errors_by_kind"] = dict(cats)
        per[slug]["main"]["mustask_errors_by_kind"] = dict(ma)

    def paired(a: str, b: str, var: str = "main", s: str = "all", var_b: str | None = None) -> dict[str, Any] | None:
        ta, tb = taskacc.get((a, var, s)), taskacc.get((b, var_b or var, s))
        if not ta or not tb:
            return None
        common = sorted(set(ta) & set(tb))
        r = EC.paired_boot([ta[k] for k in common], [tb[k] for k in common], n_boot=N_BOOT, seed=BOOT_SEED)
        return {"delta": r["est"], "lo": r["ci"][0], "hi": r["ci"][1], "n_tasks": len(common)}

    for slug, v in per.items():
        m = v.get("main")
        if not m:
            continue
        m["vs_rl"] = paired(slug, "rl") if slug != "rl" else None
        m["vs_jev"] = paired(slug, "jev") if slug != "jev" else None
        for var, key in (("nosurf", "cost_surface"), ("nodist", "cost_distraction")):
            if var in v:
                m[key] = paired(slug, slug, var="main", var_b=var)       # main - undone (negative = the axis costs)
                if m[key]:
                    m[key] = {"delta": -m[key]["delta"], "lo": -m[key]["hi"], "hi": -m[key]["lo"],
                              "n_tasks": m[key]["n_tasks"]}         # undone - main: what the axis costs (points)
        m["newkind_vs_genkind"] = None
        if m.get("newkind", {}).get("acc") is not None and m.get("genkind", {}).get("acc") is not None:
            m["newkind_vs_genkind"] = m["newkind"]["acc"] - m["genkind"]["acc"]
    return {"corpus": corpus, "rows": per, "taskacc": {f"{a}|{b}|{c}": d for (a, b, c), d in taskacc.items()}}


def l3_rows(live_dir: Path) -> dict[str, Any]:
    import explore_common as EC

    rows: dict[str, Any] = {}
    ys: dict[str, dict[str, float]] = {}
    for slug, (run, arm) in LIVE_SOURCES.items():
        p = live_dir / f"{run}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())["arms"].get(arm)
        if not d:
            continue
        tasks = [k for k in d if L3_SEED0 <= task_seed(k) < L3_SEED0 + L3_N]
        ok = [k for k in tasks if not d[k]["error"]]
        y = {k: float(d[k]["success"]) for k in ok}
        ys[slug] = y
        kinds: dict[str, list[float]] = defaultdict(list)
        for k in ok:
            kinds[kind_of_task(k)].append(y[k])
        own = [d[k]["own_acc"] for k in ok if d[k].get("own_acc") is not None]
        rows[slug] = {"n": len(ok), "errored": len(tasks) - len(ok), "success": float(np.mean(list(y.values()))) if y else None,
                      "own_acc": float(np.mean(own)) if own else None,
                      "steps": float(np.mean([d[k]["steps"] for k in ok])) if ok else None,
                      "s1_calls": float(np.mean([d[k]["s1_calls"] for k in ok])) if ok else None,
                      "s2_calls": float(np.mean([d[k]["s2_calls"] for k in ok])) if ok else None,
                      "api_usd": float(np.mean([d[k]["cost"] for k in ok])) if ok else None,
                      "s1_usd": float(np.mean([d[k]["s1_cost"] for k in ok])) if ok else None,
                      "wall_s": float(np.mean([d[k]["wall_s"] for k in ok])) if ok else None,
                      "refusals": int(sum(d[k].get("refusals") or 0 for k in ok)),
                      "by_kind": {k: {"success": float(np.mean(v)), "n": len(v)} for k, v in sorted(kinds.items())},
                      "newkind": float(np.mean([y[k] for k in ok if kind_of_task(k) in ("notify", "clarify.ask_twice",
                                                                                         "clarify.ask_after_lookup")]))
                      if ok else None}
    for slug in rows:
        for ref in ("llm_only", "jev"):
            if slug == ref or ref not in ys:
                continue
            common = sorted(set(ys[slug]) & set(ys[ref]))
            if common:
                rows[slug][f"vs_{ref}"] = {**EC.newcombe([ys[slug][k] for k in common], [ys[ref][k] for k in common]),
                                          "n": len(common)}
    return rows


HOST_RUNS = ("e21_live_rl", "e21_live_e14a", "e21_live_e12", "e21_live_sl9b", "e21_live_openjevhf", "e21_live_sl27b",
             "e21_live_sl4b", "e21_live_intern", "e21_live_e12c", "e21_live_base", "e21_live_sft_v1")


def _probe_usd(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(float(m.group(1)) for m in re.finditer(r"\(logged \$([0-9.]+)\)", p.read_text()))


def spend_summary() -> dict[str, Any]:
    """Per provider, from the slim live copies (every record of every run, errored attempts included), the replay
    answer files and the probe logs; this container's runs and the host's runs separately."""
    g_local = g_host = jev = oai = 0.0
    runs = {}
    for p in sorted((OUT / "live").glob("e21_*.json")):
        d = json.loads(p.read_text())
        sp = d.get("spend") or {}
        name = d["run"]
        host = any(name in (h, h.replace("e21_live_", "e21_live_smoke_")) for h in HOST_RUNS)
        runs[name] = {"gemini": sp.get("s2", 0.0), "s1": sp.get("s1", 0.0), "records": sp.get("records"), "host": host}
        if host:
            g_host += sp.get("s2", 0.0)
        else:
            g_local += sp.get("s2", 0.0)
        for arm, v in (sp.get("s1_by_arm") or {}).items():
            if arm.startswith("jev"):
                jev += v
            elif arm.startswith("oaidec"):
                oai += v
    rep: dict[str, float] = {}
    for p in sorted(CDIR.glob("answers_*.jsonl*")):
        slug = p.name[len("answers_"):].split(".jsonl")[0].split("__")[0]
        rep[slug] = rep.get(slug, 0.0) + sum(float(r.get("cost_usd") or 0) for r in jl(p))
    probes = _probe_usd(OUT / "api_probe_log.txt") + _probe_usd(OUT / "box" / "api_probe_log.txt")
    return {"gemini": g_local + g_host + probes, "gemini_local": g_local, "gemini_host": g_host, "gemini_probes": probes,
            "jev": jev + rep.get("jev", 0.0), "jev_replay": rep.get("jev", 0.0), "openai": oai + rep.get("oaidec", 0.0),
            "openai_replay": rep.get("oaidec", 0.0), "runs": runs}


def best(vals: Mapping[str, Any], group: Sequence[str]) -> tuple[str | None, float | None]:
    xs = [(s, vals[s]) for s in group if vals.get(s) is not None]
    if not xs:
        return None, None
    s, v = max(xs, key=lambda x: x[1])
    return s, v


def cmd_summarize(a: argparse.Namespace) -> int:
    from explore_startlux_live import kendall_tau_b  # type: ignore[import-not-found]

    l2 = l2_rows(CDIR) if (CDIR / "corpus.json").exists() else {"corpus": {}, "rows": {}, "taskacc": {}}
    l3 = l3_rows(OUT / "live")
    ind = in_domain()
    main = {s: (v.get("main") or {}) for s, v in l2["rows"].items()}
    acc = {s: (m.get("all") or {}).get("acc") for s, m in main.items()}
    lead: dict[str, Any] = {}
    for meas, vals in (("l2_all", acc),
                       ("l2_mustask", {s: (m.get("mustask") or {}).get("acc") for s, m in main.items()}),
                       ("l2_offplan", {s: (m.get("offplan") or {}).get("acc") for s, m in main.items()}),
                       ("l2_renamed", {s: (m.get("renamed") or {}).get("acc") for s, m in main.items()}),
                       ("l2_newkind", {s: (m.get("newkind") or {}).get("acc") for s, m in main.items()}),
                       ("l2_genkind", {s: (m.get("genkind") or {}).get("acc") for s, m in main.items()}),
                       ("l3", {s: r.get("success") for s, r in l3.items()}),
                       ("l3_newkind", {s: r.get("newkind") for s, r in l3.items()}),
                       ("in_c", {s: d.get("c") for s, d in ind.items()}),
                       ("in_live", {s: d.get("live") for s, d in ind.items() if s != "llm_only"})):
        bt, vt = best(vals, TRAINED)
        bo, vo = best(vals, OFFSHELF)
        lead[meas] = {"best_trained": bt, "trained": vt, "best_offshelf": bo, "offshelf": vo,
                      "gap": (vt - vo) if vt is not None and vo is not None else None,
                      "keeps": (vt >= vo) if vt is not None and vo is not None else None}
    for axis in ("surface", "distraction"):
        ct = [main[s][f"cost_{axis}"]["delta"] for s in TRAINED if (main.get(s) or {}).get(f"cost_{axis}")]
        co = [main[s][f"cost_{axis}"]["delta"] for s in OFFSHELF if (main.get(s) or {}).get(f"cost_{axis}")]
        lead[f"cost_{axis}"] = {"trained_mean": float(np.mean(ct)) if ct else None, "offshelf_mean": float(np.mean(co))
                                if co else None, "n_trained": len(ct), "n_offshelf": len(co),
                                "diff": (float(np.mean(ct)) - float(np.mean(co))) if ct and co else None}
    kend: dict[str, Any] = {}
    dm = set(TRAINED + OFFSHELF)          # E21's decision models: the registered Kendall rows (registration (f))
    for name, xs, ys_, rows_ in (("c_vs_l2", {s: d.get("c") for s, d in ind.items()}, acc, dm),
                                 ("live_vs_l3", {s: d.get("live") for s, d in ind.items() if s != "llm_only"},
                                  {s: r.get("success") for s, r in l3.items() if s != "llm_only"}, dm),
                                 ("l2_vs_l3", acc, {s: r.get("success") for s, r in l3.items() if s != "llm_only"}, dm),
                                 ("c_vs_l3", {s: d.get("c") for s, d in ind.items()},
                                  {s: r.get("success") for s, r in l3.items() if s != "llm_only"}, dm),
                                 # "E21 addendum 2" (g), descriptive: shift L2 vs shift L3 with the two ladder rows added
                                 ("l2_vs_l3_ladder", acc, {s: r.get("success") for s, r in l3.items() if s != "llm_only"},
                                  dm | set(LADDER))):
        common = sorted(s for s in xs if s in rows_ and xs.get(s) is not None and ys_.get(s) is not None)
        kend[name] = {"tau_b": kendall_tau_b([xs[s] for s in common], [ys_[s] for s in common]) if len(common) >= 3
                      else None, "n": len(common), "rows": common}
    spend = spend_summary()
    box = OUT / "box"
    host = {}
    for f in ("host.json", "model_runs.jsonl", "gates.jsonl"):
        if (box / f).exists():
            host[f] = (box / f).read_text().strip().splitlines()
    res = {"label": LABEL, "generated": _now(), "corpus": l2["corpus"], "l2": main,
           "l2_variants": {s: {v: {k: (x.get(k) or {}).get("acc") if isinstance(x.get(k), dict) else x.get(k)
                                   for k in SUBSETS} for v, x in vv.items()} for s, vv in l2["rows"].items()},
           "l3": l3, "in_domain": ind, "lead": lead, "kendall": kend, "host": host, "spend": spend,
           "variants": json.loads((CDIR / "variants.json").read_text()) if (CDIR / "variants.json").exists() else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (OUT / "summary.md").write_text(summary_md(res))
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(tex(res))
    print(f"-> {OUT / 'summary.md'}, {TEX}")
    return 0


def _p(v: Any, d: int = 1) -> str:
    return "–" if v is None else f"{100 * float(v):.{d}f}"


def _pt(v: Any, d: int = 1) -> str:
    return "–" if v is None else f"{100 * float(v):+.{d}f}"


def summary_md(r: Mapping[str, Any]) -> str:
    c = r["corpus"]
    L = ["# E21: DecisionGym-Shift (exploratory, descriptive)", "", r["label"], "",
         f"Generated {r['generated']} by `python scripts/explore_e21_shift.py summarize`. L2 = the shift corpus (LLM-only "
         "and Jev REPLACE trajectories on the 100 L2 tasks, built as C; task means); L3 = REPLACE at the tool choice, "
         "Flash, the 200 L3 tasks; in-domain = the logged C accuracy and live success on the first 200 Stage-2 tasks (the "
         "ladder rows of 'E21 addendum 2', Qwen3-4B untrained and + SFT v1: their Stage-2 rows, live on the first 100).", "",
         f"Corpus: {c.get('tasks')} tasks, {c.get('occurrences')} occurrences, {c.get('questions')} questions, "
         f"{c.get('options_mean') and round(c['options_mean'], 1)} options per question; subsets (occurrences / tasks): "
         + ", ".join(f"{s} {c.get('subset_n', {}).get(s)} / {c.get('subset_tasks', {}).get(s)}" for s in SUBSETS), "",
         "## The shift table", "",
         "| model | in-domain C | shift L2 | must-ask | after off-plan | renamed-tool | new kinds | generator kinds | "
         "surface cost | distraction cost | in-domain live | shift L3 (n) | L3 new kinds |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["llm_only", "base", "sft_v1", "rl", "e12", "e12c", "e14a", "jev", "oaidec", "sl4b", "sl9b", "sl27b", "openjevhf",
             "intern"]
    for s in order:
        m = r["l2"].get(s) or {}
        ind = r["in_domain"].get(s) or {}
        l3 = r["l3"].get(s) or {}
        if not m and not l3 and not ind:
            continue
        g = lambda k: _p((m.get(k) or {}).get("acc"))   # noqa: E731
        cs, cd = m.get("cost_surface") or {}, m.get("cost_distraction") or {}
        L.append(f"| {NAMES.get(s, s)} | {_p(ind.get('c'))} | {g('all')} | {g('mustask')} | {g('offplan')} | "
                 f"{g('renamed')} | {g('newkind')} | {g('genkind')} | {_pt(cs.get('delta'))} | {_pt(cd.get('delta'))} | "
                 f"{_p(ind.get('live'))} | {_p(l3.get('success'))} ({l3.get('n', '–')}) | {_p(l3.get('newkind'))} |")
    L += ["", "Axis cost = accuracy with the axis undone (re-rendered states) minus accuracy on the recorded states, in "
          "points (positive = the axis costs the model).", "", "## Paired L2 deltas (task bootstrap, B = 2,000, seed 0)", "",
          "| model | vs AgentDecision-4B | vs Jev | surface cost [95 %] | distraction cost [95 %] |", "|---|---|---|---|---|"]
    f = lambda d: "–" if not d or d.get("delta") is None else f"{100 * d['delta']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"  # noqa: E731
    for s in order:
        m = r["l2"].get(s)
        if m:
            L.append(f"| {NAMES.get(s, s)} | {f(m.get('vs_rl'))} | {f(m.get('vs_jev'))} | {f(m.get('cost_surface'))} | "
                     f"{f(m.get('cost_distraction'))} |")
    L += ["", "## L3 rows", "", "| row | n | errored | success | own-state acc | steps | S1 / S2 calls | API $ / ep. | "
          "wall s | vs LLM-only [Newcombe] | vs Jev [Newcombe] | refusals |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    nc = lambda d: "–" if not d else f"{d['diff_pt']:+.1f} [{d['ci_pt'][0]:+.1f}, {d['ci_pt'][1]:+.1f}]"  # noqa: E731
    for s in order:
        x = r["l3"].get(s)
        if x:
            L.append(f"| {NAMES.get(s, s)} | {x['n']} | {x['errored']} | {_p(x['success'])} | {_p(x['own_acc'])} | "
                     f"{x['steps'] and round(x['steps'], 1)} | {x['s1_calls'] and round(x['s1_calls'], 1)} / "
                     f"{x['s2_calls'] and round(x['s2_calls'], 1)} | {x['api_usd'] and round(x['api_usd'], 4)} | "
                     f"{x['wall_s'] and round(x['wall_s'], 1)} | {nc(x.get('vs_llm_only'))} | {nc(x.get('vs_jev'))} | "
                     f"{x['refusals']} |")
    L += ["", "L3 success per kind: " + "; ".join(f"{NAMES.get(s, s)}: " + ", ".join(
        f"{k} {_p(v['success'], 0)} ({v['n']})" for k, v in x["by_kind"].items()) for s, x in r["l3"].items()), "",
          "## Lead of the trained rows (best trained minus best off-the-shelf, points)", ""]
    for k, v in r["lead"].items():
        if k.startswith("cost_"):
            L.append(f"- {k}: trained mean {_pt(v['trained_mean'])}, off-the-shelf mean {_pt(v['offshelf_mean'])}, "
                     f"difference {_pt(v['diff'])} (n {v['n_trained']} / {v['n_offshelf']})")
        else:
            L.append(f"- {k}: {NAMES.get(v['best_trained'], v['best_trained'])} {_p(v['trained'])} vs "
                     f"{NAMES.get(v['best_offshelf'], v['best_offshelf'])} {_p(v['offshelf'])}: gap {_pt(v['gap'])}, "
                     f"keeps the lead: {v['keeps']}")
    L += ["", "## Wrong answers on the recorded states, by kind (post hoc, descriptive; occurrences)", ""]
    for s in order:
        m = r["l2"].get(s) or {}
        if m.get("errors_by_kind") is not None:
            L.append(f"- {NAMES.get(s, s)}: " + ", ".join(f"{k} {v}" for k, v in sorted(m["errors_by_kind"].items()))
                     + "; at must-ask states: " + (", ".join(f"{k} {v}" for k, v in sorted(m["mustask_errors_by_kind"].items()))
                                                    or "none"))
    L += ["", "## Kendall tau-b (in-domain vs shift)", ""]
    for k, v in r["kendall"].items():
        L.append(f"- {k}: {v['tau_b'] if v['tau_b'] is None else round(v['tau_b'], 2)} (n {v['n']}: {', '.join(v['rows'])})")
    sp = r.get("spend") or {}
    L += ["", "## Spend", "", f"- direct Gemini ${sp.get('gemini', 0):.2f} (this container ${sp.get('gemini_local', 0):.2f}, "
          f"the host ${sp.get('gemini_host', 0):.2f}, probes ${sp.get('gemini_probes', 0):.3f}); OpenAI ${sp.get('openai', 0):.2f} "
          f"(replays ${sp.get('openai_replay', 0):.2f}); Jev ${sp.get('jev', 0):.2f} (replays ${sp.get('jev_replay', 0):.2f}); "
          "OpenRouter other than the Jev key $0."]
    L += ["", "## Records", "", "- answers per model and rendering: `results/explore/e21_shift/c/answers_<slug>[__<variant>].jsonl.gz`;"
          " live slim copies: `results/explore/e21_shift/live/`; host records: `results/explore/e21_shift/box/`.", ""]
    for s, v in sorted(r["l2"].items()):
        L.append(f"- {s}: {v.get('answers')} answers, {v.get('errors')} errors, {v.get('refusals')} refusals, "
                 f"{v.get('missing')} missing, logged ${v.get('cost_usd', 0):.4f}, median {v.get('latency_ms_median')} ms")
    return "\n".join(L) + "\n"


def tex(r: Mapping[str, Any]) -> str:
    import explore_common as EC

    K: dict[str, str] = {}
    p1 = lambda v: EC.fmt(None if v is None else 100 * float(v), 1)   # noqa: E731
    c = r["corpus"]
    for k in ("tasks", "occurrences", "questions"):
        if c.get(k) is not None:
            K[f"e21/corpus/{k}"] = EC.fmt(int(c[k]))
    if c.get("options_mean"):
        K["e21/corpus/options"] = EC.fmt(int(round(c["options_mean"])))
    for s in SUBSETS:
        if (c.get("subset_n") or {}).get(s) is not None:
            K[f"e21/corpus/{s}/n"] = EC.fmt(int(c["subset_n"][s]))
            K[f"e21/corpus/{s}/tasks"] = EC.fmt(int(c["subset_tasks"][s]))
    for slug, m in r["l2"].items():
        for s in SUBSETS:
            if (m.get(s) or {}).get("acc") is not None:
                K[f"e21/{slug}/l2/{s}"] = p1(m[s]["acc"])
        for key in ("vs_rl", "vs_jev", "cost_surface", "cost_distraction"):
            d = m.get(key)
            if d and d.get("delta") is not None:
                K[f"e21/{slug}/l2/{key}"] = p1(d["delta"])
                K[f"e21/{slug}/l2/{key}/lo"] = p1(d["lo"])
                K[f"e21/{slug}/l2/{key}/hi"] = p1(d["hi"])
    for slug, x in r["l3"].items():
        if x.get("success") is not None:
            K[f"e21/{slug}/l3/success"] = p1(x["success"])
            K[f"e21/{slug}/l3/n"] = EC.fmt(int(x["n"]))
        if x.get("own_acc") is not None:
            K[f"e21/{slug}/l3/own"] = p1(x["own_acc"])
        if x.get("newkind") is not None:
            K[f"e21/{slug}/l3/newkind"] = p1(x["newkind"])
        if x.get("api_usd") is not None:
            K[f"e21/{slug}/l3/usd"] = EC.fmt(float(x["api_usd"]), 4)
        for kind, v in (x.get("by_kind") or {}).items():
            K[f"e21/{slug}/l3/kind/{kind}"] = p1(v["success"])
        for ref in ("llm_only", "jev"):
            d = x.get(f"vs_{ref}")
            if d:
                K[f"e21/{slug}/l3/vs_{ref}"] = EC.fmt(float(d["diff_pt"]), 1)
                K[f"e21/{slug}/l3/vs_{ref}/lo"] = EC.fmt(float(d["ci_pt"][0]), 1)
                K[f"e21/{slug}/l3/vs_{ref}/hi"] = EC.fmt(float(d["ci_pt"][1]), 1)
    for slug, vv in (r.get("l2_variants") or {}).items():
        for var in ("nosurf", "nodist"):
            if (vv.get(var) or {}).get("all") is not None:
                K[f"e21/{slug}/l2/{var}"] = p1(vv[var]["all"])
    for slug, d in r["in_domain"].items():
        if d.get("c") is not None:
            K[f"e21/{slug}/in/c"] = p1(d["c"])
        if d.get("live") is not None:
            K[f"e21/{slug}/in/live"] = p1(d["live"])
    for k, v in r["lead"].items():
        if k.startswith("cost_"):
            for kk in ("trained_mean", "offshelf_mean", "diff"):
                if v.get(kk) is not None:
                    K[f"e21/lead/{k}/{kk}"] = p1(v[kk])
        elif v.get("gap") is not None:
            K[f"e21/lead/{k}/gap"] = p1(v["gap"])
            K[f"e21/lead/{k}/keeps"] = EC.fmt(bool(v["keeps"]))
            K[f"e21/lead/{k}/trained"] = EC.fmt(NAMES.get(v["best_trained"], v["best_trained"]))
            K[f"e21/lead/{k}/offshelf"] = EC.fmt(NAMES.get(v["best_offshelf"], v["best_offshelf"]))
    for k, v in r["kendall"].items():
        if v.get("tau_b") is not None:
            K[f"e21/kendall/{k}"] = EC.fmt(float(v["tau_b"]), 2)
            K[f"e21/kendall/{k}/n"] = EC.fmt(int(v["n"]))
    for k in ("gemini", "gemini_local", "gemini_host", "openai", "jev"):
        if (r.get("spend") or {}).get(k) is not None:
            K[f"e21/spend/{k}"] = EC.fmt(float(r["spend"][k]), 2)
    gh = OUT / "gpu_hours.json"
    if gh.exists():
        K["e21/gpu/hours"] = EC.fmt(float(json.loads(gh.read_text())["host_hours"]), 1)
    for k, v in (r.get("extra_keys") or {}).items():
        K[k] = v
    head = ["% AUTO-GENERATED by scripts/explore_e21_shift.py -- do not edit by hand.",
            "% Exploratory row E21, DecisionGym-Shift (descriptive; docs/PREREG_A7_STAGE2.md Deviations 2026-10-08), "
            "outside both stages.",
            f"% last written {r['generated']} at git {(EC.git_head() or '?')[:12]}", r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    return "\n".join(head + [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(K.items())] + [r"\makeatother", ""])


if __name__ == "__main__":
    sys.exit(main())
