#!/usr/bin/env python3
r"""E15 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E15: more decision-model
baselines"): every public decision model we can run, and OpenAI's hosted Decisions API, at the three levels -- E1's
3,991 static decisions (L1, E9's cost rule), the common states C (L2) and live REPLACE on the first 200 fresh Stage-2
tasks (L3, the hosted API plus the top two GPU models by L2) -- and Kendall's tau-b over the old and new rows (RQ1).

    # operations (the E15 host runs most of them through scripts/aws/e15_baselines.sh)
    python scripts/explore_e15_baselines.py spec --slug jevk5                  # the harness backend spec of a model
    python scripts/explore_e15_baselines.py replay --corpus DIR --name SLUG --s1 JSON [--workers 4]   # L2 (resumes)
    python scripts/explore_e15_baselines.py quick --run runs/e1_bfcl_when2call_e15 --slug SLUG
    python scripts/explore_e15_baselines.py probe | watch | errored | spend | gate     # live pause rule (registration (d))
    python scripts/explore_e15_baselines.py archive --l1 DIR --dev DIR --c DIR [--live RUN ...]   # slim copies -> results/
    python scripts/explore_e15_baselines.py summarize       # summary.{md,json} + paper/generated/e15_baselines.tex

Cost per decision by E9's rule (``explore_e9_frontier``): the API at its logged list price, open weights at $2.24208
per GPU-hour times the logged serving time per decision; E9's frontier and keys stay as they are (the recomputed
frontier: ``e9/front15/...``). Static column of Figure 4b: BFCL tool-choice accuracy (``e15/<slug>/e1_acc``, as
``explore/startlux_live/<row>/e1_acc``)."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

OUT = ROOT / "results/explore/e15_baselines"
TEX = ROOT / "paper/generated/e15_baselines.tex"
E1CFG = ROOT / "configs/explore/e15_baselines/e1_e15.yaml"
S23 = ROOT / "results/stage2/s23"
STARTLUX = ROOT / "results/explore/startlux_live/summary.json"
TASKS = ROOT / "configs/eval/jevgym_stage2_tasks.json"
LIVE_N = 200
N_BOOT, BOOT_SEED = 2000, 0
LABEL = ("exploratory (E15), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-07 'E15: more decision-model "
         "baselines'; outside every Holm table")
OVERLOAD = ("503", "overload", "high demand", "unavailable", "resource_exhausted", "429")
PROBE_N = 6
FLASH = {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0, "reasoning_effort": "minimal"}
FAMILIES = ("toolchoice", "args", "w2c")

# slug -> (label, TeX name, cost rule, licence, served by, read-out)
MODELS: dict[str, tuple[str, str, str, str, str, str]] = {
    "oaidec": ("OpenAI Decisions API (gpt-6-luna)", "OpenAI Decisions", "api", "proprietary API",
               "OpenAI, POST /v1/decisions (public beta)", "hosted option scores"),
    "clefflash": ("Clef-Flash 9B", "Clef-Flash", "gpu", "Apache-2.0", "the card's systemone() (bf16)",
                  "joint schema head"),
    "clef": ("Clef 27B (8-bit)", "Clef (8-bit)", "gpu", "Apache-2.0", "the card's systemone() (LLM.int8 backbone)",
             "joint schema head"),
    "openjev": ("OpenJev (DiffusionGemma-26B-A4B)", "OpenJev", "gpu", "Apache-2.0", "razorback16/openjev:0.5.0",
                "masked-slot read of a diffusion LM"),
    "openjevhf": ("OpenJev 27B (HF, FP8)", "OpenJev-27B", "gpu", "CC BY-NC 4.0 (weights)",
                  "vLLM 0.29.0 + the card's helper", "first-token letter scores"),
    "jevk5": ("JevK5 v0.2 (4B)", "JevK5", "gpu", "Apache-2.0", "razorback16/openjev-jevk5:0.5.1",
              "answer-letter logits"),
    "clm": ("CLM v0.1 (8B)", "CLM", "gpu", "Apache-2.0", "razorback16/openjev-clm:0.5.1",
            "contrastive heads on embeddings"),
    "laya": ("Laya (421M)", "Laya", "gpu", "Apache-2.0", "razorback16/openjev-laya:0.5.1",
             "encoder head, 1,024 tokens"),
    "verdict": ("Verdict 1.4 (151M)", "Verdict", "gpu", "Apache-2.0", "razorback16/openjev-verdict:0.5.1",
                "encoder head, 512 tokens"),
    "decider": ("decider-4b v2.1", "decider-4b", "gpu", "Apache-2.0", "its own server (decider.serve)",
                "answer-letter logits"),
    "imajev": ("Imajev-4B", "Imajev-4B", "gpu", "Apache-2.0", "its own server (4 option orders, calibration)",
               "option-code logits"),
    "bosun": ("Bosun 3.1 1.7B", "Bosun", "gpu", "Apache-2.0", "the card's model.predict()", "decision-token slots"),
}
GPU_ORDER = ("clefflash", "openjev", "jevk5", "decider", "imajev", "bosun", "clm", "laya", "verdict", "openjevhf",
             "clef")
LIVE_PREFIX = "e15_live"


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


def gz_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fo:
        shutil.copyfileobj(fi, fo)
    return dst


# ================================================================================================ operations
def backend_spec(slug: str) -> dict[str, Any]:
    import yaml

    spec = (yaml.safe_load(E1CFG.read_text()).get("backends") or {}).get(slug)
    if spec is None:
        raise SystemExit(f"no backend {slug!r} in {E1CFG}")
    return dict(spec)


def cmd_spec(a: argparse.Namespace) -> int:
    spec = backend_spec(a.slug)
    if a.model:
        spec["model"] = a.model
    if a.base_url and spec.get("type") == "jev":
        spec["base_url"] = a.base_url
    print(json.dumps(spec))
    return 0


def cmd_quick(a: argparse.Namespace) -> int:
    p = Path(a.run) / f"s1_{a.slug}.jsonl"
    if not p.exists():
        print("no records")
        return 1
    rs = jl(p)
    n = len(rs)
    err = sum(1 for r in rs if r.get("error"))
    ref = sum(1 for r in rs if r.get("refusal"))
    acc = sum(1 for r in rs if r.get("correct") and not r.get("error")) / n if n else float("nan")
    lat = sorted(float(r.get("latency_ms") or 0) for r in rs if not r.get("error"))
    print(f"{n} answers, acc {100 * acc:.1f} %, errors {err}, refusals {ref}, "
          f"median latency {lat[len(lat) // 2] if lat else float('nan'):.0f} ms")
    return 0


def cmd_wire_ok(a: argparse.Namespace) -> int:
    """The wire check passes iff at most half the development-prefix answers errored."""
    p = Path(a.run) / f"s1_{a.slug}.jsonl"
    rs = jl(p) if p.exists() else []
    err = sum(1 for r in rs if r.get("error"))
    ok = bool(rs) and err <= len(rs) / 2
    print(json.dumps({"slug": a.slug, "n": len(rs), "errors": err, "ok": ok,
                      "first_error": next((r["error"] for r in rs if r.get("error")), None)}))
    return 0 if ok else 1


def cmd_quickc(a: argparse.Namespace) -> int:
    from jevharness.bench import stage2_corpus as SC

    cdir = Path(a.corpus)
    ap = cdir / f"answers_{a.slug}.jsonl"
    if not ap.exists():
        print("no answers")
        return 1
    occ = SC.read_jsonl_gz(cdir / "occurrences.jsonl.gz")
    elig = [str(t) for t in json.loads((cdir / "corpus.json").read_text())["eligible_tasks"]]
    ans = {r["qhash"]: r for r in jl(ap)}
    acc = SC.task_accuracy(occ, ans, elig)
    err = sum(1 for r in ans.values() if r.get("error"))
    print(f"C acc {100 * float(np.mean(list(acc.values()))):.1f} % over {len(acc)} tasks; {len(ans)} answers, "
          f"{err} errors, {sum(1 for r in ans.values() if r.get('refusal'))} refusals")
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    """L2: every distinct question of C answered through the harness's option handling (``stage2_corpus.replay``: one
    question per request, the harness knockout above the backend's read width), ``workers`` questions in flight;
    resumes from ``answers_<name>.jsonl``; writes ``replay_<name>.json`` first."""
    from jevharness.bench import stage2_corpus as SC
    from jevharness.decision import make_backend

    cdir = Path(a.corpus)
    qs = SC.read_jsonl_gz(cdir / "questions.jsonl.gz")
    out = cdir / f"answers_{a.name}.jsonl"
    done: set[str] = set()
    if out.exists():
        done = {r["qhash"] for r in jl(out) if not r.get("error")}
    spec = json.loads(a.s1)
    backend = make_backend(spec, cache=a.cache)
    (cdir / f"replay_{a.name}.json").write_text(json.dumps({
        "registered": "PREREG_A7_STAGE2 Deviations 2026-10-07 'E15: more decision-model baselines' (c) 3",
        "backend": backend.describe(), "cache": a.cache, "order": "sorted question hash, sharded over workers",
        "workers": a.workers, "batching": "one question per request (the harness tournament for Choices wider than "
        "the backend's max_options)", "retries": 2, "n_questions": len(qs), "resumed_with": len(done),
        "written": _now()}, indent=1, default=str))
    todo = sorted((q for q in qs if q["qhash"] not in done), key=lambda x: x["qhash"])
    lock = threading.Lock()
    f = open(out, "a")

    def write(rec: dict[str, Any]) -> None:
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    shards = [todo[i::max(1, a.workers)] for i in range(max(1, a.workers))]
    ths = [threading.Thread(target=SC.replay, args=(s, backend), kwargs={"retries": 2, "on_answer": write})
           for s in shards if s]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    f.close()
    recs = {r["qhash"]: r for r in jl(out)}
    n_err = sum(1 for q in qs if (recs.get(q["qhash"]) or {}).get("error") or q["qhash"] not in recs)
    print(f"answers -> {out}: {len(recs)} of {len(qs)}; errors / missing {n_err}", file=sys.stderr)
    return 0 if n_err == 0 else 2


LIVE_HEAD = """# Exploratory row E15 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E15: more decision-model baselines" (c) 4;
# descriptive, outside every Holm table): {label} LIVE, as `jev_ts_replace_f` of configs/experiments/stage2_rl_states.yaml,
# the E2 ladder and E12: tool_select REPLACE, the same suite block, {tasks}, Flash on the direct Gemini API (temperature 0,
# minimal reasoning), a new System-1 cache. Written by `scripts/explore_e15_baselines.py livecfg --slug {slug}` from the
# model's L1 backend (configs/explore/e15_baselines/e1_e15.yaml); {where}; probe-gated, paused on 503 / overload.
"""


def cmd_livecfg(a: argparse.Namespace) -> int:
    """The live and smoke configs of a GPU model selected by (c) 4 (the same backend as its L1 / L2 rows)."""
    import yaml

    spec = backend_spec(a.slug)
    where = ("served on :8090 of the E15 host (scripts/aws/e15_baselines.sh live_{s})".format(s=a.slug)
             if spec.get("type") == "jev" else "run from the coordinating container")
    for smoke in (False, True):
        exp = {"name": f"{LIVE_PREFIX}{'_smoke' if smoke else ''}_{a.slug}",
               "suite": ({"name": "jevgym", "n": 2, "seed0": 906, "families": "v2", "distractor_style": "near",
                          "n_distractors": 60, "noise_lines": 20, "strict": True} if smoke else
                         {"name": "jevgym", "n": 500, "seed0": 300000, "families": "v2", "distractor_style": "near",
                          "n_distractors": 60, "noise_lines": 20, "strict": True}),
               "n_tasks": 2 if smoke else LIVE_N,
               "llm": dict(FLASH),
               "s1_cache": f"cache/s1_e15_live{'_smoke' if smoke else ''}_{a.slug}.sqlite",
               "s1_backends": {a.slug: spec},
               "workers": 2 if smoke else 4,
               "arms": [{"name": f"{a.slug}_ts_replace_f", "s1": a.slug, "tool_select": {"mode": "replace"}}]}
        head = LIVE_HEAD.format(label=MODELS[a.slug][0], slug=a.slug, where=where,
                                tasks=("2 JevGym development tasks (seeds 906-907, never reported; the cost gate's smoke)"
                                       if smoke else "the first 200 fresh Stage-2 tasks (seeds 300000..300199; "
                                       "evaluation seeds: JEV_EVAL_RUNS=1), workers 4"))
        out = ROOT / "configs/explore/e15_baselines" / f"live{'_smoke' if smoke else ''}_{a.slug}.yaml"
        out.write_text(head + yaml.safe_dump(exp, sort_keys=False, width=200))
        print(out.relative_to(ROOT))
    return 0


def cmd_rank(a: argparse.Namespace) -> int:
    """Registration (c) 4 on the host: the GPU models with a complete L2 (answers for at least 99 % of C's questions),
    ranked by L2 accuracy (task means), ties by L1 accuracy, then by lower E9-rule cost; prints the top ``--top``."""
    from jevharness.bench import stage2_corpus as SC

    cdir = Path(a.corpus)
    occ = SC.read_jsonl_gz(cdir / "occurrences.jsonl.gz")
    elig = [str(t) for t in json.loads((cdir / "corpus.json").read_text())["eligible_tasks"]]
    qs = {o["qhash"] for o in occ if o["task"] in set(elig)}
    rows = {}
    for slug, m in MODELS.items():
        ap = cdir / f"answers_{slug}.jsonl"
        if m[2] != "gpu" or not ap.exists():
            continue
        ans = {r["qhash"]: r for r in jl(ap)}
        cov = len(qs & {k for k, r in ans.items() if not r.get("error")}) / len(qs)
        acc = SC.task_accuracy(occ, ans, elig)
        l1p = Path(a.l1) / f"s1_{slug}.jsonl"
        l1 = jl(l1p) if l1p.exists() else []
        rows[slug] = {"corpus_acc": float(np.mean(list(acc.values()))) if acc else 0.0, "coverage": cov,
                      "l1_acc": (sum(1 for r in l1 if r.get("correct") and not r.get("error")) / len(l1)) if l1 else 0.0,
                      "l1_usd": float(np.mean([2.24208 / 3.6e6 * float(r.get("latency_ms") or 0.0)
                                               / max(1, int(r.get("request_questions") or 1)) for r in l1])) if l1 else 0.0}
    ok = [s for s, r in rows.items() if r["coverage"] >= 0.99]
    ranked = sorted(ok, key=lambda s: (-rows[s]["corpus_acc"], -rows[s]["l1_acc"], rows[s]["l1_usd"]))
    top = ranked[: a.top]
    if a.out:
        Path(a.out).write_text(json.dumps({"rule": "registration (c) 4", "rows": rows, "ranked": ranked, "top": top,
                                           "excluded_incomplete": sorted(set(rows) - set(ok)), "at": _now()}, indent=1))
    print(" ".join(top))
    return 0 if top else 1


# ------------------------------------------------------------------------------------------------ live pause rule
def cmd_probe(a: argparse.Namespace) -> int:
    """One round: six one-word Flash calls on the direct API through the harness client (its retries included)."""
    from jevharness.accounting import Ledger
    from jevharness.llm import make_llm

    if a.dotenv:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    llm = make_llm(dict(FLASH))
    led = Ledger()
    ok, lat, errs = 0, [], []
    for i in range(PROBE_N):
        t0 = time.perf_counter()
        try:
            llm.chat([{"role": "user", "content": f"Reply with one word: ready ({i})"}], ledger=led,
                     decision_point="probe", max_tokens=8)
            ok += 1
        except Exception as e:  # noqa: BLE001 -- every failure counts against the round
            errs.append(f"{type(e).__name__}: {str(e)[:120]}")
        lat.append(time.perf_counter() - t0)
    lat.sort()
    line = (f"{_now()} direct Gemini 3.5 Flash: {ok}/{PROBE_N} ok, median {lat[len(lat) // 2]:.1f} s, max {lat[-1]:.1f} s"
            + (f"; errors: {errs[0]}" if errs else "") + ("  CLEAN" if ok == PROBE_N else "  UNHEALTHY"))
    p = Path(a.log)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# E15: direct Gemini API probes (Deviations 2026-10-07 'E15: more decision-model baselines' (d)). "
                     "One round = six one-word Gemini 3.5 Flash calls; start / resume after two consecutive 6/6 rounds.\n")
    with open(p, "a") as f:
        f.write(line.replace("\n", " ") + f"  (logged ${led.summary()['total']['cost_usd']:.5f})\n")
    print(line)
    return 0 if ok == PROBE_N else 1


def episode_cost(t: Mapping[str, Any]) -> float:
    return float(((t.get("usage") or {}).get("total") or {}).get("cost_usd") or 0.0)


def episode_comp_cost(t: Mapping[str, Any], comp: str) -> float:
    return float(((t.get("usage") or {}).get(comp) or {}).get("cost_usd") or 0.0)


def run_spend(run: Path) -> dict[str, Any]:
    """Every record of every arm (errored attempts and re-runs included: all were paid)."""
    out = {"total": 0.0, "s2": 0.0, "s1": 0.0, "records": 0}
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        for t in jl(p):
            if isinstance(t, dict) and "task_id" in t:
                out["total"] += episode_cost(t)
                out["s2"] += episode_comp_cost(t, "s2")
                out["s1"] += episode_comp_cost(t, "s1")
                out["records"] += 1
    return out


def live_runs(runs: Path) -> list[Path]:
    return sorted(p for p in Path(runs).glob(f"{LIVE_PREFIX}*") if p.is_dir())


def total_spend(runs: Path) -> dict[str, Any]:
    per = {p.name: run_spend(p) for p in live_runs(runs)}
    return {"by_run": per, "total": sum(v["total"] for v in per.values()), "s2": sum(v["s2"] for v in per.values()),
            "records": sum(v["records"] for v in per.values())}


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
    print(f"live ${s['total']:.3f} (Gemini ${s['s2']:.3f}) in {s['records']} records" if a.short else json.dumps(s, indent=1))
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    """The smoke's mean $ per episode x n x 1.5 + the spend so far must fit the cap (registration (c) 4)."""
    sm = run_spend(Path(a.smoke))
    tot = total_spend(Path(a.runs))
    per = sm["total"] / sm["records"] if sm["records"] else float("inf")
    proj = per * a.n * 1.5 + tot["total"]
    res = {"smoke": str(a.smoke), "smoke_records": sm["records"], "usd_per_episode": per, "n": a.n,
           "projection_usd": proj, "spent_usd": tot["total"], "cap_usd": a.cap, "go": proj <= a.cap, "at": _now()}
    print(json.dumps(res))
    return 0 if res["go"] else 1


def _kill(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        time.sleep(10)


def cmd_watch(a: argparse.Namespace) -> int:
    """Spend cap (-> STOP file) and, unless --no-pause, a pause on a new episode that errored with 503 / overload
    (-> <run>/PAUSED_overload.json); the runner's process group is stopped either way; records are kept."""
    runs = Path(a.runs)
    seen = {p.name: {(arm, k) for arm, d in latest(p).items() for k, t in d.items() if t.get("error")}
            for p in live_runs(runs)}
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{_now()} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(runs)
        if s["total"] >= a.cap:
            Path(a.stop).write_text(json.dumps({"at": _now(), "spend": s, "cap": a.cap}, indent=1))
            _kill(a.pid)
            print(f"CAP REACHED: ${s['total']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        if not a.no_pause:
            for p in live_runs(runs):
                new = [(arm, k, t) for arm, d in latest(p).items() for k, t in d.items()
                       if t.get("error") and (arm, k) not in seen.get(p.name, set()) and overload_error(t)]
                if new:
                    (p / "PAUSED_overload.json").write_text(json.dumps(
                        {"at": _now(), "episodes": [{"arm": arm, "task": k, "error": str(t.get("error"))[:300]}
                                                    for arm, k, t in new]}, indent=1))
                    _kill(a.pid)
                    print(f"{_now()} PAUSED: {len(new)} overload error(s) in {p.name}; runner stopped", flush=True)
                    return 2
        print(f"{_now()} spend ${s['total']:.3f} ({s['records']} records)", flush=True)
        time.sleep(a.every)


# ------------------------------------------------------------------------------------------------ archive
def slim_live(run: Path, live_tasks: set[str]) -> dict[str, Any]:
    """Per arm, per task: the record that decides the task (explore_common.load_latest) in a slim form."""
    import explore_common as C

    out: dict[str, Any] = {}
    for p in sorted(run.glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        lat, att = C.load_latest([p])
        rows = {}
        for k, t in lat.items():
            refusals = sum(1 for s in (t.get("steps") or []) for d in (s.get("decisions") or [])
                           if (d.get("gold_aux") or {}).get("s1_refusal"))
            rows[k] = {"in_live": k in live_tasks, "success": C.success(t), "error": bool(t.get("error")),
                       "error_text": str(t.get("error") or "")[:200] or None, "cost": C.cost(t),
                       "s1_cost": episode_comp_cost(t, "s1"), "s2_cost": episode_comp_cost(t, "s2"),
                       "wall_s": float(t.get("wall_ms") or 0) / 1000, "steps": len(t.get("steps") or []),
                       "s1_calls": C.s_calls(t, "s1"), "s2_calls": C.s_calls(t, "s2"), "attempts": att.get(k, 1),
                       "refusals": refusals}
        out[p.stem] = rows
    return out


def own_acc(run: Path, arm: str, live_tasks: set[str]) -> dict[str, float]:
    import explore_common as C
    from explore_startlux_live import own_accuracy  # type: ignore[import-not-found]
    from jevharness.bench.scorecard import task_key

    p = run / f"{arm}.jsonl"
    if not p.exists():
        return {}
    lat, _ = C.load_latest([p])
    return own_accuracy([t for t in lat.values() if not t.get("error") and task_key(t) in live_tasks])


def cmd_archive(a: argparse.Namespace) -> int:
    done = []
    for src, sub, pat in ((a.l1, "l1", "s1_*.jsonl"), (a.dev, "dev", "s1_*.jsonl"), (a.c, "c", "answers_*.jsonl")):
        if not src:
            continue
        for p in sorted(Path(src).glob(pat)):
            done.append(gz_copy(p, OUT / sub / (p.name + ".gz")))
        if sub == "c":
            for p in sorted(Path(src).glob("replay_*.json")):
                shutil.copyfile(p, OUT / "c" / p.name)
        for extra in ("ledger.json",):
            if (Path(src) / extra).exists() and sub != "c":
                shutil.copyfile(Path(src) / extra, OUT / sub / f"{Path(src).name}_{extra}")
    live = set(json.loads(TASKS.read_text())["task_ids"][:LIVE_N])
    for run in a.live or []:
        rp = Path(run)
        if not rp.is_dir():
            continue
        slim = slim_live(rp, live)
        own = {arm: own_acc(rp, arm, live) for arm in slim}
        dst = OUT / "live" / f"{rp.name}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"run": rp.name, "arms": slim, "own_acc": own, "written": _now()}, indent=1) + "\n")
        if (rp / "experiment.yaml").exists():
            shutil.copyfile(rp / "experiment.yaml", OUT / "live" / f"{rp.name}.experiment.yaml")
        done.append(dst)
    print("\n".join(str(p.relative_to(ROOT)) for p in done))
    return 0


# ================================================================================================ analysis
def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def l1_stats(recs: Sequence[Mapping[str, Any]], rule: str, e9: Any) -> dict[str, Any]:
    s = e9.model_stats(recs, rule)
    s["refusals"] = sum(1 for r in recs if r.get("refusal"))
    return s


def e1_toolchoice(recs: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """item -> correctness on BFCL tool_select records (errors count wrong), as explore_startlux_live.e1_items."""
    from jevharness.bench.decision_eval import _rescore

    return {str(r.get("id")): (float(_rescore(r)) if "error" not in r else 0.0) for r in recs
            if r.get("dp") == "tool_select" and str(r.get("source", "")).startswith("bfcl:")}


def task_boot(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    tasks = sorted(set(a) & set(b))
    d = np.array([a[t] - b[t] for t in tasks], float)
    if not len(d):
        return {"delta": None, "lo": None, "hi": None, "n_tasks": 0}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bs = d[idx].mean(axis=1)
    return {"delta": float(d.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)),
            "n_tasks": len(tasks)}


def c_level(occ: Sequence[Mapping[str, Any]], eligible: Sequence[str], ans: Mapping[str, Mapping[str, Any]],
            ref: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """C accuracy (task means) over all states, after an off-plan step and on the ask-vs-act stratum (answer key
    ``ask_user``), C ECE (as E5 / stage2_ladder); paired task bootstrap against each reference's per-task accuracy."""
    from jevharness.bench import metrics as Mx
    from jevharness.bench import stage2_corpus as SC

    keep = set(eligible)
    acc = SC.task_accuracy(occ, ans, eligible)
    off = SC.task_accuracy([o for o in occ if o.get("stratum") == "after_off_plan"], ans, eligible)
    ask = SC.task_accuracy([o for o in occ if o.get("gold") == "ask_user"], ans, eligible)
    conf, corr = [], []
    for o in occ:
        if o["task"] in keep:
            x = ans.get(o["qhash"]) or {}
            conf.append(float(x.get("confidence") or 0.0) if not x.get("error") else 0.0)
            corr.append(float(SC.answer_correct(o, x)))
    asked = {o["qhash"] for o in occ if o["task"] in keep}
    out = {"corpus_acc": float(np.mean(list(acc.values()))) if acc else None, "corpus_tasks": len(acc),
           "corpus_acc_offplan": float(np.mean(list(off.values()))) if off else None,
           "corpus_acc_ask": float(np.mean(list(ask.values()))) if ask else None, "corpus_tasks_ask": len(ask),
           "ece": Mx.ece(conf, corr) if conf else None, "missing": len(asked - set(ans)),
           # the direction of the calibration error: mean confidence against occurrence-level accuracy
           "mean_conf": float(np.mean(conf)) if conf else None, "occ_acc": float(np.mean(corr)) if corr else None,
           "errors": sum(1 for q in asked if (ans.get(q) or {}).get("error")),
           "refusals": sum(1 for q in asked if (ans.get(q) or {}).get("refusal")),
           "occurrences": len(conf), "questions": len(asked)}
    for name, r in ref.items():
        out[f"vs_{name}"] = task_boot(acc, r)
    out["_per_task"] = acc
    return out


def kendall_block(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from explore_startlux_live import kendall_tau_b  # type: ignore[import-not-found]

    full = sorted(r for r, v in rows.items() if all(v.get(k) is not None for k in ("e1_acc", "corpus_acc", "live_success")))
    two = sorted(r for r, v in rows.items() if all(v.get(k) is not None for k in ("e1_acc", "corpus_acc")))
    out: dict[str, Any] = {"rows_all_three": full, "rows_static_common": two}
    if len(full) >= 3:
        out["e1_vs_live_all_rows"] = kendall_tau_b([rows[r]["e1_acc"] for r in full], [rows[r]["live_success"] for r in full])
        out["corpus_vs_live_all_rows"] = kendall_tau_b([rows[r]["corpus_acc"] for r in full],
                                                       [rows[r]["live_success"] for r in full])
    if len(two) >= 3:
        out["e1_vs_corpus_all_rows"] = kendall_tau_b([rows[r]["e1_acc"] for r in two], [rows[r]["corpus_acc"] for r in two])
    sl, cl = out.get("e1_vs_live_all_rows"), out.get("corpus_vs_live_all_rows")
    out["holds"] = (None if sl is None or cl is None else bool(sl < cl and sl < 0.5))
    return out


def analyse(n_boot: int = N_BOOT) -> dict[str, Any]:
    import explore_common as C
    import explore_e9_frontier as E9
    from jevharness.bench import stage2_corpus as SC

    data = E9.load_e1()
    rates = E9.token_rates()
    e1 = E9.analyse_e1(data, rates, n_boot=n_boot)
    ids = sorted(next(iter(data.values())))
    M = e1["models"]
    drawn = {m: (v["usd"], v["acc"]) for m, v in M.items() if v["drawn"]}
    sl4b_rate = rates["sl4b"]["usd_per_m"]
    res: dict[str, Any] = {"label": LABEL, "n_e1": len(ids), "gpu_usd_per_h": E9.GPU_USD_PER_H, "models": {},
                           "dropped": {}, "e9_frontier": list(e1["frontier"])}
    occ = SC.read_jsonl_gz(S23 / "occurrences.jsonl.gz")
    eligible = [str(t) for t in json.loads((S23 / "corpus.json").read_text())["eligible_tasks"]]
    ref_ans = {"jev": {r["qhash"]: r for r in jl(S23 / "answers_jev.jsonl")},
               "frozen": {r["qhash"]: r for r in jl(S23 / "answers_rl.jsonl")}}
    ref_c = {k: SC.task_accuracy(occ, v, eligible) for k, v in ref_ans.items()}
    live_tasks = set(json.loads(TASKS.read_text())["task_ids"][:LIVE_N])
    # Jev's logged live row (the reference of every live row): stage2_rl_states/jev_ts_replace_f
    jev_live: dict[str, float] = {}
    jp = ROOT / "runs/stage2_rl_states/jev_ts_replace_f.jsonl"
    if jp.exists():
        from jevharness.bench.scorecard import task_key

        lat, _ = C.load_latest([jp])
        jev_live = {task_key(t): float(C.success(t)) for t in lat.values() if not t.get("error") and task_key(t) in live_tasks}
    jev_e1tc = e1_toolchoice(data["jev"].values())
    points: dict[str, tuple[float, float]] = {}
    for slug, (label, tex, rule, lic, served, readout) in MODELS.items():
        row: dict[str, Any] = {"label": label, "tex": tex, "cost_rule": rule, "licence": lic, "served_by": served,
                               "readout": readout}
        # ---- wire check (dev prefix)
        dp = OUT / "dev" / f"s1_{slug}.jsonl.gz"
        if dp.exists():
            dv = jl(dp)
            row["dev"] = {"n": len(dv), "acc": (sum(1 for r in dv if r.get("correct") and not r.get("error")) / len(dv))
                          if dv else None, "errors": sum(1 for r in dv if r.get("error")),
                          "refusals": sum(1 for r in dv if r.get("refusal"))}
        # ---- L1
        lp = OUT / "l1" / f"s1_{slug}.jsonl.gz"
        if lp.exists():
            recs = {r["id"]: r for r in jl(lp)}
            if set(recs) >= set(ids):
                rr = [recs[i] for i in ids]
                s = l1_stats(rr, rule, E9)
                pts = {**drawn, slug: (s["usd"], s["acc"])}
                fr = E9.frontier(pts)
                tc = e1_toolchoice(rr)
                gj = E9.paired_gap([E9.is_correct(recs[i]) for i in ids], [E9.is_correct(data["jev"][i]) for i in ids],
                                   n_boot)
                gf = E9.paired_gap([E9.is_correct(recs[i]) for i in ids], [E9.is_correct(data["rl"][i]) for i in ids],
                                   n_boot)
                sens = {}
                if rule == "gpu":
                    tok = float(np.mean([float(r.get("input_tokens") or 0.0) for r in rr])) * sl4b_rate / 1e6
                    for name, usd in (("zero", 0.0), ("token_tariff", tok)):
                        base = {m: (e1["sensitivity"][name]["usd"].get(m, M[m]["usd"]), M[m]["acc"]) for m in drawn}
                        sens[name] = {"usd": usd, "on_frontier": slug in E9.frontier({**base, slug: (usd, s["acc"])})}
                row["l1"] = {**{k: v for k, v in s.items()}, "on_frontier": slug in fr, "gap_jev": gj, "gap_frozen": gf,
                             "cost_vs_jev": s["usd"] / M["jev"]["usd"] if M["jev"]["usd"] else None,
                             "dominated_by": sorted(m for m, (c, a_) in drawn.items()
                                                    if c <= s["usd"] and a_ >= s["acc"] and (c < s["usd"] or a_ > s["acc"])),
                             "sensitivity": sens, "e1_acc": float(np.mean(list(tc.values()))) if tc else None,
                             "e1_n": len(tc)}
                common = sorted(set(tc) & set(jev_e1tc))
                if common:
                    nc = C.newcombe([tc[k] for k in common], [jev_e1tc[k] for k in common])
                    row["l1"]["e1_vs_jev"] = nc["diff_pt"] / 100
                    row["l1"]["e1_vs_jev_ci"] = [nc["ci_pt"][0] / 100, nc["ci_pt"][1] / 100]
                points[slug] = (s["usd"], s["acc"])
            else:
                row["l1_partial"] = f"{len(set(recs) & set(ids))} of {len(ids)} decisions answered"
        # ---- L2
        cp = OUT / "c" / f"answers_{slug}.jsonl.gz"
        if cp.exists():
            ans = {}
            for r in jl(cp):
                ans[r["qhash"]] = r            # the latest record of a question decides
            row["l2"] = c_level(occ, eligible, ans, ref_c)
        # ---- L3
        lv = OUT / "live" / f"{LIVE_PREFIX}_{slug}.json"
        if lv.exists():
            d = json.loads(lv.read_text())
            arm = f"{slug}_ts_replace_f"
            rows = (d.get("arms") or {}).get(arm) or {}
            ok = {k: v for k, v in rows.items() if v["in_live"] and not v["error"]}
            errd = [k for k, v in rows.items() if v["in_live"] and v["error"]]
            y = {k: float(v["success"]) for k, v in ok.items()}
            own = (d.get("own_acc") or {}).get(arm) or {}
            l3: dict[str, Any] = {"live_n": len(ok), "live_errored": len(errd),
                                  "live_success": float(np.mean(list(y.values()))) if y else None,
                                  "own_acc": float(np.mean(list(own.values()))) if own else None,
                                  "steps": float(np.mean([v["steps"] for v in ok.values()])) if ok else None,
                                  "api_cost": float(np.mean([v["cost"] for v in ok.values()])) if ok else None,
                                  "s1_cost": float(np.mean([v["s1_cost"] for v in ok.values()])) if ok else None,
                                  "wall_s": float(np.mean([v["wall_s"] for v in ok.values()])) if ok else None,
                                  "refusals": int(sum(v.get("refusals", 0) for v in ok.values())),
                                  "spend_all_records": float(sum(v["cost"] for v in rows.values()))}
            common = sorted(set(y) & set(jev_live))
            if common:
                nc = C.newcombe([y[k] for k in common], [jev_live[k] for k in common])
                l3["live_vs_jev"] = nc["diff_pt"] / 100
                l3["live_vs_jev_ci"] = [nc["ci_pt"][0] / 100, nc["ci_pt"][1] / 100]
                l3["live_vs_jev_n"] = len(common)
            row["l3"] = l3
        res["models"][slug] = row
    # ---- the recomputed frontier with every new point (E9's own frontier unchanged)
    if points:
        fw = E9.frontier({**drawn, **points})
        res["frontier_with"] = fw
        res["frontier_changed"] = fw != list(e1["frontier"])
        for slug in points:
            res["models"][slug]["l1"]["on_frontier_all"] = slug in fw
    # ---- L3 selection (registration (c) 4): the top two GPU models by L2 accuracy
    ranked = sorted((s for s, r in res["models"].items() if MODELS[s][2] == "gpu" and (r.get("l2") or {}).get("corpus_acc")
                     is not None),
                    key=lambda s: (-res["models"][s]["l2"]["corpus_acc"],
                                   -((res["models"][s].get("l1") or {}).get("acc") or 0.0),
                                   (res["models"][s].get("l1") or {}).get("usd") or float("inf")))
    res["l2_ranking"] = ranked
    res["live_selection"] = ["oaidec"] + ranked[:2]
    # ---- RQ1: Kendall over old and new rows
    old = json.loads(STARTLUX.read_text())["rows"] if STARTLUX.exists() else {}
    krows: dict[str, dict[str, Any]] = {r: {"e1_acc": v.get("e1_acc"), "corpus_acc": v.get("corpus_acc"),
                                            "live_success": v.get("live_success"), "old": True} for r, v in old.items()}
    for slug, r in res["models"].items():
        krows[f"e15:{slug}"] = {"e1_acc": (r.get("l1") or {}).get("e1_acc"), "corpus_acc": (r.get("l2") or {}).get("corpus_acc"),
                                "live_success": (r.get("l3") or {}).get("live_success"), "old": False}
    res["kendall_rows"] = krows
    res["kendall"] = kendall_block(krows)
    res["kendall_old_only"] = kendall_block({k: v for k, v in krows.items() if v["old"]})
    # off the shelf (a reviewer's request, 2026-10-07; no new call): the rows without our in-domain trained model
    # (`frozen` = AgentDecision-4B). Trained rows of other exploratory runs (E12 / E12c / E14) are not in this row set;
    # `notrained` drops every row whose key names one, so it equals `offtheshelf` here.
    trained = {k for k in krows if k == "frozen" or any(t in k for t in ("e12", "e14"))}
    for name, drop in (("offtheshelf", {"frozen"}), ("notrained", trained)):
        sub = {k: v for k, v in krows.items() if k not in drop}
        kb = kendall_block(sub)
        live = {k: v["live_success"] for k, v in sub.items() if v.get("live_success") is not None}
        kb["dropped"] = sorted(drop)
        kb["live_spread"] = (max(live.values()) - min(live.values())) if len(live) >= 2 else None
        kb["live_best"] = max(live, key=live.get) if live else None
        kb["live_worst"] = min(live, key=live.get) if live else None
        res[f"kendall_{name}"] = kb
    live_all = {k: v["live_success"] for k, v in krows.items() if v.get("live_success") is not None}
    res["live_spread_all"] = (max(live_all.values()) - min(live_all.values())) if len(live_all) >= 2 else None
    # ---- spend
    res["spend"] = spend_summary(res)
    box = OUT / "box"
    if (box / "model_runs.jsonl").exists():
        res["model_runs"] = jl(box / "model_runs.jsonl")
    if (OUT / "gpu_hours.json").exists():
        res["gpu"] = json.loads((OUT / "gpu_hours.json").read_text())
    if (OUT / "dropped.json").exists():
        res["dropped"] = json.loads((OUT / "dropped.json").read_text())
    return res


def spend_summary(res: Mapping[str, Any]) -> dict[str, Any]:
    """OpenAI: every Decisions-API record (wire check, L1, L2, the live System-1 calls); Gemini: the live rows."""
    oa = 0.0
    for sub, pat in (("dev", "s1_oaidec.jsonl.gz"), ("l1", "s1_oaidec.jsonl.gz")):
        p = OUT / sub / pat
        if p.exists():
            oa += sum(float(r.get("cost_usd") or 0.0) for r in jl(p) if not r.get("cached"))
    cp = OUT / "c" / "answers_oaidec.jsonl.gz"
    if cp.exists():
        oa += sum(float(r.get("cost_usd") or 0.0) for r in jl(cp))
    gem, oa_live = 0.0, 0.0
    for p in sorted((OUT / "live").glob("*.json")) if (OUT / "live").exists() else []:
        d = json.loads(p.read_text())
        for arm, rows in (d.get("arms") or {}).items():
            for v in rows.values():
                gem += float(v.get("s2_cost") or 0.0)
                if "oaidec" in p.name:
                    oa_live += float(v.get("s1_cost") or 0.0)
    return {"openai_usd": oa + oa_live, "openai_offline_usd": oa, "openai_live_usd": oa_live, "gemini_live_usd": gem,
            "note": "live spend counts the deciding record per task (re-run attempts are in the run's own records)"}


# ================================================================================================ outputs
def _pct(x: Any, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{d}f}"


def _num(x: Any, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def tex_keys(res: Mapping[str, Any]) -> dict[str, str]:
    import explore_common as C
    import explore_e9_frontier as E9

    k: dict[str, str] = {"e15/n/models": str(sum(1 for r in res["models"].values() if r.get("l1") or r.get("l2")))}
    for slug, r in res["models"].items():
        l1, l2, l3 = r.get("l1"), r.get("l2"), r.get("l3")
        if l1:
            k[f"e9/{slug}/acc"] = f"{100 * l1['acc']:.1f}"
            for f, v in (l1.get("families") or {}).items():
                k[f"e9/{slug}/acc/{f}"] = f"{100 * v['acc']:.1f}"
            k[f"e9/{slug}/usdk"] = E9._sig(1000 * l1["usd"])
            if l1["usd"] > 0:
                k[f"e9/{slug}/lg"] = f"{math.log10(l1['usd']):.4f}"
            k[f"e9/{slug}/lat"] = C.fmt(int(round(l1["lat_ms_median"])))
            k[f"e9/{slug}/front"] = "yes" if l1["on_frontier"] else "no"
            k[f"e9/{slug}/frontall"] = "yes" if l1.get("on_frontier_all") else "no"
            k[f"e9/{slug}/gapjev"] = E9._pm(l1["gap_jev"]["est"])
            k[f"e9/{slug}/gapfrozen"] = E9._pm(l1["gap_frozen"]["est"])
            if l1.get("cost_vs_jev"):
                k[f"e9/{slug}/costvsjev"] = E9._ratio(l1["cost_vs_jev"])
            k[f"e15/{slug}/l1/errors"] = str(l1["n_errors"])
            k[f"e15/{slug}/l1/refusals"] = str(l1.get("refusals", 0))
            if l1.get("e1_acc") is not None:
                k[f"e15/{slug}/e1_acc"] = C.fmt(100 * l1["e1_acc"])
            if l1.get("e1_vs_jev") is not None:
                k[f"e15/{slug}/e1_vs_jev"] = C.fmt(100 * l1["e1_vs_jev"])
        if l2:
            for m in ("corpus_acc", "corpus_acc_offplan", "corpus_acc_ask"):
                if l2.get(m) is not None:
                    k[f"e15/{slug}/{m}"] = C.fmt(100 * l2[m])
            if l2.get("ece") is not None:
                k[f"e15/{slug}/ece"] = C.fmt(l2["ece"], 3)
            if l2.get("mean_conf") is not None:
                k[f"e15/{slug}/meanconf"] = C.fmt(100 * l2["mean_conf"])
                k[f"e15/{slug}/occacc"] = C.fmt(100 * l2["occ_acc"])
            for ref in ("jev", "frozen"):
                v = l2.get(f"vs_{ref}") or {}
                if v.get("delta") is not None:
                    k[f"e15/{slug}/corpus_vs_{ref}"] = C.fmt(100 * v["delta"])
                    k[f"e15/{slug}/corpus_vs_{ref}/lo"] = C.fmt(100 * v["lo"])
                    k[f"e15/{slug}/corpus_vs_{ref}/hi"] = C.fmt(100 * v["hi"])
            k[f"e15/{slug}/l2/errors"] = str(l2["errors"] + l2["missing"])
            k[f"e15/{slug}/l2/refusals"] = str(l2["refusals"])
        if l3:
            for m in ("live_success", "own_acc", "live_vs_jev"):
                if l3.get(m) is not None:
                    k[f"e15/{slug}/{m}"] = C.fmt(100 * l3[m])
            if l3.get("live_vs_jev_ci"):
                k[f"e15/{slug}/live_vs_jev/lo"] = C.fmt(100 * l3["live_vs_jev_ci"][0])
                k[f"e15/{slug}/live_vs_jev/hi"] = C.fmt(100 * l3["live_vs_jev_ci"][1])
            k[f"e15/{slug}/live_n"] = C.fmt(int(l3["live_n"]))
            if l3.get("api_cost") is not None:
                k[f"e15/{slug}/api_cost"] = C.fmt(l3["api_cost"], 4)
            if l3.get("steps") is not None:
                k[f"e15/{slug}/steps"] = C.fmt(l3["steps"])
            k[f"e15/{slug}/l3/refusals"] = str(l3.get("refusals", 0))
        k[f"e15/{slug}/live"] = "yes" if l3 else "no"
        k[f"e15/{slug}/livecell"] = (C.fmt(100 * l3["live_success"]) if l3 and l3.get("live_success") is not None
                                     else "--")    # a table cell: the live success, or a dash when not run live
    if res.get("frontier_with"):
        fw = res["frontier_with"]
        k.update({"e9/front15/first": E9.tikz_name(fw[0]), "e9/front15/rest": ",".join(E9.tikz_name(m) for m in fw[1:]),
                  "e9/front15/n": str(len(fw)), "e9/front15/changed": "yes" if res["frontier_changed"] else "no"})
    for name, kb in (("kendall", res["kendall"]), ("kendallold", res["kendall_old_only"])):
        for m in ("e1_vs_live_all_rows", "corpus_vs_live_all_rows", "e1_vs_corpus_all_rows"):
            if kb.get(m) is not None:
                k[f"e15/{name}/{m}"] = C.fmt(kb[m], 2)
        k[f"e15/{name}/n_rows_all_three"] = str(len(kb["rows_all_three"]))
        k[f"e15/{name}/n_rows_static_common"] = str(len(kb["rows_static_common"]))
        if kb.get("holds") is not None:
            k[f"e15/{name}/holds"] = "yes" if kb["holds"] else "no"
    for name in ("offtheshelf", "notrained"):
        kb = res.get(f"kendall_{name}") or {}
        for m, short in (("e1_vs_live_all_rows", "static_live"), ("corpus_vs_live_all_rows", "common_live"),
                         ("e1_vs_corpus_all_rows", "static_common")):
            if kb.get(m) is not None:
                k[f"e15/kendall/{name}/{short}"] = C.fmt(kb[m], 2)
        if kb:
            k[f"e15/kendall/{name}/n"] = str(len(kb["rows_all_three"]))
            k[f"e15/kendall/{name}/n_static_common"] = str(len(kb["rows_static_common"]))
            if kb.get("holds") is not None:
                k[f"e15/kendall/{name}/holds"] = "yes" if kb["holds"] else "no"
        if kb.get("live_spread") is not None:
            k[f"e15/spread/{name}/live"] = C.fmt(100 * kb["live_spread"])
    if res.get("live_spread_all") is not None:
        k["e15/spread/all/live"] = C.fmt(100 * res["live_spread_all"])
    sp = res.get("spend") or {}
    k["e15/spend/openai"] = f"{sp.get('openai_usd', 0.0):.2f}"
    k["e15/spend/gemini"] = f"{sp.get('gemini_live_usd', 0.0):.2f}"
    if res.get("gpu", {}).get("gpu_hours") is not None:
        k["e15/gpuh"] = f"{res['gpu']['gpu_hours']:.1f}"
    return k


def write_tex(vals: Mapping[str, str], path: Path = TEX) -> Path:
    import explore_common as C

    lines = ["% AUTO-GENERATED by scripts/explore_e15_baselines.py -- do not edit by hand.",
             "% Exploratory row E15, more decision-model baselines (descriptive; docs/PREREG_A7_STAGE2.md Deviations "
             "2026-10-07), outside both stages. e9/<slug>/... are E15 points beside E9's drawn set (E9's keys unchanged).",
             f"% last written {_now()} at git {(C.git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


def render_md(res: Mapping[str, Any]) -> str:
    L = ["# E15: more decision-model baselines (exploratory, descriptive)", "", res["label"] + ".", "",
         f"Generated {res['written']} at git {(res.get('git') or '?')[:12]} by `python scripts/explore_e15_baselines.py "
         "summarize`. L1 = E1's 3,991 decisions (E9's cost rule: API at logged list price; open weights at "
         f"${res['gpu_usd_per_h']:.5f} per GPU-hour x logged serving time per decision); L2 = the common states C "
         "(5,314 occurrences, 200 tasks; task means); L3 = REPLACE at the tool choice, Flash, the first 200 fresh "
         "Stage-2 tasks.", "",
         "## Candidates", "", "| model | licence | served by | read-out | wire check (299 dev items): acc / errors / refusals |",
         "|---|---|---|---|---|"]
    for slug, r in res["models"].items():
        dv = r.get("dev") or {}
        wc = f"{_pct(dv.get('acc'))} / {dv.get('errors', '–')} / {dv.get('refusals', '–')}" if dv else "not run"
        L.append(f"| {r['label']} | {r['licence']} | {r['served_by']} | {r['readout']} | {wc} |")
    if res.get("dropped"):
        L += ["", "Dropped: " + "; ".join(f"{k}: {v}" for k, v in res["dropped"].items()) + "."]
    L += ["", "## The three levels", "",
          "| model | L1 acc % [Wilson] | tool choice / args / When2Call | $ / 1,000 | median ms | L1 − Jev [95 %] | "
          "BFCL tool choice (static) | C acc | after off-plan | ask-vs-act | ECE | C − Jev [95 %] | live success | "
          "live − Jev [Newcombe] | errors L1 / L2 | refusals L1 / L2 / L3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for slug, r in res["models"].items():
        l1, l2, l3 = r.get("l1") or {}, r.get("l2") or {}, r.get("l3") or {}
        if not (l1 or l2 or l3):
            L.append(f"| {r['label']} | {r.get('l1_partial', 'not run')} |" + " – |" * 14)
            continue
        fam = " / ".join(_pct((l1.get("families") or {}).get(f, {}).get("acc")) for f in FAMILIES) if l1 else "–"
        g = l1.get("gap_jev") or {}
        cv = l2.get("vs_jev") or {}
        L.append(
            f"| {r['label']} | {_pct(l1.get('acc'))}"
            + (f" [{_pct(l1['acc_ci'][0])}, {_pct(l1['acc_ci'][1])}]" if l1 else "")
            + f" | {fam} | {_num(1000 * l1['usd'], 4) if l1 else '–'} | {_num(l1['lat_ms_median'], 0) if l1 else '–'} | "
            + (f"{g['est']:+.1f} [{g['ci'][0]:+.1f}, {g['ci'][1]:+.1f}]" if g.get("est") is not None else "–")
            + f" | {_pct(l1.get('e1_acc'))} | {_pct(l2.get('corpus_acc'))} | {_pct(l2.get('corpus_acc_offplan'))} | "
            f"{_pct(l2.get('corpus_acc_ask'))} | {_num(l2.get('ece'), 3)} | "
            + (f"{100 * cv['delta']:+.1f} [{100 * cv['lo']:+.1f}, {100 * cv['hi']:+.1f}]" if cv.get("delta") is not None else "–")
            + f" | {_pct(l3.get('live_success'))}" + (f" (n {l3['live_n']})" if l3 else "") + " | "
            + (f"{100 * l3['live_vs_jev']:+.1f} [{100 * l3['live_vs_jev_ci'][0]:+.1f}, {100 * l3['live_vs_jev_ci'][1]:+.1f}]"
               if l3.get("live_vs_jev") is not None else "–")
            + f" | {l1.get('n_errors', '–')} / {(l2.get('errors', 0) + l2.get('missing', 0)) if l2 else '–'} | "
            f"{l1.get('refusals', '–')} / {l2.get('refusals', '–')} / {l3.get('refusals', '–')} |")
    L += ["", "Reference rows (logged): Jev L1 86.4 %, C 91.5 %, live 83.0 %; AgentDecision-4B L1 78.8 %, C 99.2 %, "
          "live 100.0 % (E9 summary, explore/startlux_live).", ""]
    if res.get("frontier_with"):
        L.append(f"Frontier with the E15 points (by cost): {' → '.join(res['frontier_with'])}; changed from E9's "
                 f"({' → '.join(res['e9_frontier'])}): {'yes' if res['frontier_changed'] else 'no'}.")
        for slug, r in res["models"].items():
            l1 = r.get("l1")
            if l1:
                sens = "; ".join(f"{n}: {'on' if v['on_frontier'] else 'off'}" for n, v in (l1.get("sensitivity") or {}).items())
                L.append(f"- {r['label']}: on the frontier alone with E9's drawn set: {'yes' if l1['on_frontier'] else 'no'}"
                         + (f" (dominated by {', '.join(l1['dominated_by'])})" if l1["dominated_by"] else "")
                         + (f"; sensitivity {sens}" if sens else "") + ".")
    L += ["", f"L3 selection (registration (c) 4): {', '.join(res['live_selection'])}; L2 ranking of the GPU models: "
          f"{', '.join(res['l2_ranking'])}.", "", "## RQ1: Kendall's tau-b over old and new rows", ""]
    for name, kb in (("old + new", res["kendall"]), ("old rows only", res["kendall_old_only"])):
        L.append(f"- {name}: static vs live {_num(kb.get('e1_vs_live_all_rows'), 2)}"
                 f", common vs live {_num(kb.get('corpus_vs_live_all_rows'), 2)}"
                 f" ({len(kb['rows_all_three'])} rows with all three: {', '.join(kb['rows_all_three'])}); static vs "
                 f"common {_num(kb.get('e1_vs_corpus_all_rows'), 2)} "
                 f"({len(kb['rows_static_common'])} rows); RQ1's reading holds (static-live below common-live and "
                 f"below 0.5): {kb.get('holds')}.")
    for name, what in (("offtheshelf", "off the shelf (without AgentDecision-4B)"),
                       ("notrained", "without any trained model (no E12 / E12c / E14 row is in this set)")):
        kb = res.get(f"kendall_{name}") or {}
        if kb:
            L.append(f"- {what}: static vs live {_num(kb.get('e1_vs_live_all_rows'), 2)}, common vs live "
                     f"{_num(kb.get('corpus_vs_live_all_rows'), 2)} ({len(kb['rows_all_three'])} rows); static vs common "
                     f"{_num(kb.get('e1_vs_corpus_all_rows'), 2)} ({len(kb['rows_static_common'])} rows); reading holds: "
                     f"{kb.get('holds')}; live-success spread (best − worst) "
                     f"{_pct(kb.get('live_spread'))} points ({kb.get('live_best')} − {kb.get('live_worst')}).")
    if res.get("live_spread_all") is not None:
        L.append(f"- live-success spread over every row with a live run: {_pct(res['live_spread_all'])} points.")
    sp = res.get("spend") or {}
    L += ["", "## Spend", "", f"OpenAI ${sp.get('openai_usd', 0):.2f} (offline ${sp.get('openai_offline_usd', 0):.2f}, "
          f"live System-1 ${sp.get('openai_live_usd', 0):.2f}); direct Gemini (live) ${sp.get('gemini_live_usd', 0):.2f}; "
          "OpenRouter $0; Jev $0." + (f" GPU: {res['gpu']}." if res.get("gpu") else ""), ""]
    return "\n".join(L)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(x) for k, x in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def cmd_summarize(a: argparse.Namespace) -> int:
    import explore_common as C

    res = analyse(n_boot=a.n_boot)
    res["written"] = _now()
    res["git"] = C.git_head()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(_jsonable(res), indent=1) + "\n")
    (OUT / "summary.md").write_text(render_md(res))
    write_tex(tex_keys(res))
    print(render_md(res))
    return 0


# ================================================================================================ CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec")
    s.add_argument("--slug", required=True)
    s.add_argument("--model")
    s.add_argument("--base-url")
    s.set_defaults(fn=cmd_spec)
    q = sub.add_parser("quick")
    q.add_argument("--run", required=True)
    q.add_argument("--slug", required=True)
    q.set_defaults(fn=cmd_quick)
    w = sub.add_parser("wire-ok")
    w.add_argument("--run", required=True)
    w.add_argument("--slug", required=True)
    w.set_defaults(fn=cmd_wire_ok)
    qc = sub.add_parser("quickc")
    qc.add_argument("--corpus", required=True)
    qc.add_argument("--slug", required=True)
    qc.set_defaults(fn=cmd_quickc)
    r = sub.add_parser("replay")
    r.add_argument("--corpus", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--s1", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--workers", type=int, default=4)
    r.set_defaults(fn=cmd_replay)
    p = sub.add_parser("probe")
    p.add_argument("--log", default=str(OUT / "api_probe_log.txt"))
    p.add_argument("--dotenv", action="store_true", help="read the key from .env (coordinating container only)")
    p.set_defaults(fn=cmd_probe)
    wa = sub.add_parser("watch")
    wa.add_argument("--pid", type=int, required=True)
    wa.add_argument("--cap", type=float, default=35.0)
    wa.add_argument("--runs", default="runs")
    wa.add_argument("--stop", default="runs/e15_STOP.json")
    wa.add_argument("--every", type=int, default=30)
    wa.add_argument("--no-pause", action="store_true")
    wa.set_defaults(fn=cmd_watch)
    e = sub.add_parser("errored")
    e.add_argument("--run", required=True)
    e.set_defaults(fn=cmd_errored)
    sp = sub.add_parser("spend")
    sp.add_argument("--runs", default="runs")
    sp.add_argument("--short", action="store_true")
    sp.set_defaults(fn=cmd_spend)
    g = sub.add_parser("gate")
    g.add_argument("--runs", default="runs")
    g.add_argument("--smoke", required=True)
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--cap", type=float, default=35.0)
    g.set_defaults(fn=cmd_gate)
    lc = sub.add_parser("livecfg")
    lc.add_argument("--slug", required=True)
    lc.set_defaults(fn=cmd_livecfg)
    rk = sub.add_parser("rank")
    rk.add_argument("--corpus", required=True)
    rk.add_argument("--l1", required=True)
    rk.add_argument("--top", type=int, default=2)
    rk.add_argument("--out")
    rk.set_defaults(fn=cmd_rank)
    ar = sub.add_parser("archive")
    ar.add_argument("--l1")
    ar.add_argument("--dev")
    ar.add_argument("--c")
    ar.add_argument("--live", nargs="*")
    ar.set_defaults(fn=cmd_archive)
    su = sub.add_parser("summarize")
    su.add_argument("--n-boot", type=int, default=N_BOOT)
    su.set_defaults(fn=cmd_summarize)
    a = ap.parse_args(list(argv) if argv is not None else None)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
