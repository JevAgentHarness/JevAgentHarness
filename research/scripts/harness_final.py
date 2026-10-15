#!/usr/bin/env python3
r"""The final JevAgentHarness: latency diagnosis, replay workload, offline serving benchmark and the live analysis.
EXPLORATORY / DESCRIPTIVE (docs/PREREG_A7_STAGE2.md, Deviations entry of 2026-10-05 "harness_final"): post hoc,
outside both registered stages, no test. Every number is computed from run files; nothing is typed by hand.

    python scripts/harness_final.py diagnose            # Stage-2 logs -> results/harness_final/latency_diagnosis.{json,md}
    python scripts/harness_final.py workload            # dev tariff run -> results/harness_final/replay_workload.jsonl.gz
    python scripts/harness_final.py bench --url http://127.0.0.1:8081 --label v0 --path old --concurrency 1 4
                                                        # GPU host: replay the workload against a served model
    python scripts/harness_final.py bench-report        # results/harness_final/bench/*.json -> replay_bench.{json,md}
    python scripts/harness_final.py analyze --run runs/harness_final_live
                                                        # -> results/harness_final/summary.{json,md},
                                                        #    paper/generated/harness_final.tex (\defres{hf/...})

``diagnose`` breaks every episode of the Stage-2 arms ``llm_only``, ``best_single``, ``jevagent_best`` and
``jevagent_best_rl`` (and the development tariff run of ``jevagent_best_rl``) into System-1 time, System-2 time and the
rest (tool execution, harness bookkeeping, the decision cache), counts System-1 requests and questions per step and per
episode, and -- for the self-hosted JevAgentRL server, which serves one request at a time -- reconstructs each request's
service and queueing time from the logged absolute call intervals (start = ``ts`` - ``latency_ms``; departures in order,
service_i = end_i - max(start_i, end_{i-1})). Cache-hit counters were not logged by the client in Stage 2.

``workload`` turns a logged ``jevagent_best_rl`` run into replayable System-1 requests in the Jev wire format: per
episode and step, the OLD call graph (reflection question; tool choice; one request per argument question) and the NEW
one (tool choice; every argument question of the step in ONE request; no reflection question), with the logged answers
of every question (``s1_dist``), the logged System-2 time and the rest of each step (to emulate the episode clock).
Tool descriptions are regenerated from the (development) task seed exactly as the agent sends them.

``bench`` replays the workload against a running server, ``c`` episodes in parallel, steps in order, with the logged
System-2 / rest time of each step slept in between (``--think``), and records per request the client latency, the server
usage (prefilled / cache-hit tokens, passes), the answers, and per step the System-1 time. Decision equivalence: every
replayed answer is compared with the logged one (max |dp|, argmax, the frozen cascade's accept / defer).
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
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results/harness_final"
TEX_OUT = ROOT / "paper/generated/harness_final.tex"
LABEL = "exploratory, descriptive (post hoc; PREREG_A7_STAGE2 Deviations 2026-10-05 harness_final); no test"
STAGE2 = ROOT / "runs/stage2_jevgym"
DEV_TARIFF_RUN = ROOT / "runs/stage2_dev_tariff_rl"
DIAG_ARMS = ("llm_only", "best_single", "jevagent_best", "jevagent_best_rl")
SCHEDULE_FILES = {"jevagent_best_rl": "schedule_rl.json"}
S1_DPS = ("reflection", "tool_select", "tool_params", "stop", "routing", "memory_write", "memory_read", "action")


# ------------------------------------------------------------------ small helpers
def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _mean(xs: Sequence[float]) -> float | None:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _median(xs: Sequence[float]) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _pct(xs: Sequence[float], q: float) -> float | None:
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    """task_id -> the latest record (the runner's resume rule)."""
    last: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and "task_id" in r:
                last[str(r["task_id"])] = r
    return last


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


# ------------------------------------------------------------------ diagnosis
def _calls(rec: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [c for c in rec.get("calls") or [] if not (c.get("meta") or {}).get("shadow")]


def episode_breakdown(rec: Mapping[str, Any]) -> dict[str, Any]:
    """One episode: wall, System-1 / System-2 latency sums, the rest, request / question counts, per decision point,
    per step, and the idle gaps between consecutive calls (start = ts - latency)."""
    calls = _calls(rec)
    s1 = [c for c in calls if c.get("component") == "s1"]
    s2 = [c for c in calls if c.get("component") == "s2"]
    wall = float(rec.get("wall_ms") or 0.0)
    s1_ms = sum(float(c.get("latency_ms") or 0) for c in s1)
    s2_ms = sum(float(c.get("latency_ms") or 0) for c in s2)
    by_dp: dict[str, dict[str, float]] = defaultdict(lambda: {"requests": 0, "questions": 0, "ms": 0.0})
    for c in s1:
        d = by_dp[str(c.get("decision_point"))]
        d["requests"] += 1
        d["questions"] += int((c.get("meta") or {}).get("n_questions") or 1)
        d["ms"] += float(c.get("latency_ms") or 0)
    steps = rec.get("steps") or []
    per_step = []
    for s in steps:
        sc = [c for c in calls if c.get("step") == s.get("step")]
        per_step.append({"s1_requests": sum(1 for c in sc if c.get("component") == "s1"),
                         "s1_ms": sum(float(c.get("latency_ms") or 0) for c in sc if c.get("component") == "s1"),
                         "s2_calls": sum(1 for c in sc if c.get("component") == "s2"),
                         "s2_ms": sum(float(c.get("latency_ms") or 0) for c in sc if c.get("component") == "s2"),
                         "state_tokens": max([int(c.get("input_tokens") or 0) for c in sc
                                              if c.get("component") == "s1"] or [0])})
    # gaps between consecutive calls of the episode (the loop is sequential: one call at a time)
    iv = sorted((_ts(c["ts"]) - float(c.get("latency_ms") or 0) / 1000.0, _ts(c["ts"]), c) for c in calls if c.get("ts"))
    gaps: dict[str, float] = defaultdict(float)
    for (a0, e0, c0), (a1, e1, c1) in zip(iv, iv[1:]):
        g = max(0.0, a1 - e0) * 1000.0
        same = c0.get("step") == c1.get("step")
        kind = (f"{'in-step' if same else 'step boundary'} {c0.get('component')}->{c1.get('component')}")
        gaps[kind] += g
    refl = [d for s in steps for d in s.get("decisions") or [] if d.get("dp") == "reflection"
            and d.get("mode") == "trigger"]
    return {"wall_ms": wall, "s1_ms": s1_ms, "s2_ms": s2_ms, "rest_ms": wall - s1_ms - s2_ms,
            "s1_requests": len(s1), "s1_questions": sum(int((c.get("meta") or {}).get("n_questions") or 1) for c in s1),
            "s2_calls": len(s2), "steps": len(steps), "by_dp": {k: dict(v) for k, v in by_dp.items()},
            "per_step": per_step, "gaps_ms": dict(gaps), "gap_total_ms": sum(gaps.values()),
            "reflection_questions": len(refl), "reflection_fired": sum(1 for d in refl if d.get("final_value") is True),
            "reflection_p_yes_max": max([float((d.get("s1_dist") or {}).get("yes", 0.0)) for d in refl] or [0.0]),
            "success": bool((rec.get("result") or {}).get("success")),
            "accepted_steps_multi_param": sum(1 for s in steps if sum(
                1 for d in s.get("decisions") or [] if d.get("dp") == "tool_params" and d.get("s1_calls")) >= 2)}


def queue_reconstruction(recs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Single-server reconstruction over every System-1 request of one invocation (the self-hosted server takes one
    request at a time under one lock): departures in time order, service_i = end_i - max(start_i, end_{i-1}), queueing
    = latency - service. Approximate (HTTP time counts as service; the lock is not strictly FIFO); the server's busy
    time (the union of in-flight intervals) is exact up to the timestamps' millisecond resolution."""
    iv = []
    for r in recs:
        for c in _calls(r):
            if c.get("component") == "s1" and c.get("ts") and not c.get("cached"):
                end = _ts(c["ts"])
                iv.append((end, end - float(c.get("latency_ms") or 0) / 1000.0, str(c.get("decision_point"))))
    if not iv:
        return {}
    iv.sort()
    prev = -1e18
    svc: dict[str, list[float]] = defaultdict(list)
    que: dict[str, list[float]] = defaultdict(list)
    busy = 0.0
    for end, start, dp in iv:
        s = end - max(start, prev)
        s = max(0.0, s)
        svc[dp].append(s * 1000)
        que[dp].append(max(0.0, (end - start) - s) * 1000)
        busy += s
        prev = max(prev, end)
    span = max(e for e, _, _ in iv) - min(s for _, s, _ in iv)
    all_s = [x for v in svc.values() for x in v]
    all_q = [x for v in que.values() for x in v]
    return {"requests": len(iv), "span_s": span, "busy_s": busy, "busy_fraction": busy / span if span > 0 else None,
            "service_ms_mean": _mean(all_s), "queue_ms_mean": _mean(all_q),
            "queue_share": (sum(all_q) / (sum(all_q) + sum(all_s))) if all_s else None,
            "by_dp": {dp: {"requests": len(svc[dp]), "service_ms_mean": _mean(svc[dp]),
                           "service_ms_median": _median(svc[dp]), "queue_ms_mean": _mean(que[dp])} for dp in svc}}


def closed_loop(a: Mapping[str, Any]) -> dict[str, Any] | None:
    """Interactive-system check (N episodes sharing one server that serves one request at a time): with S the server's
    busy time per step and Z the rest of a step (System-2 + tools), the System-1 time per step is R = max(S, N S - Z)
    once the server saturates (N S > S + Z). Compared with the logged System-1 time per step."""
    q, n = a.get("queue") or {}, a.get("concurrency")
    if not q or not n or not a.get("steps"):
        return None
    eps = a["n"]
    S = q["busy_s"] / (eps * a["steps"])
    Z = (a["s2_s"] + a["rest_s"]) / a["steps"]
    return {"N": n, "service_s_per_step": S, "think_s_per_step": Z, "saturated": n * S > S + Z,
            "predicted_s1_s_per_step": max(S, n * S - Z), "observed_s1_s_per_step": a["s1_s"] / a["steps"],
            "capacity_steps_per_s": 1 / S}


def summarise_arm(recs: Mapping[str, Mapping[str, Any]], concurrency: int | None) -> dict[str, Any]:
    eps = [episode_breakdown(r) for r in recs.values() if not r.get("error")]
    n = len(eps)
    steps = [s for e in eps for s in e["per_step"]]
    dps = sorted({k for e in eps for k in e["by_dp"]})
    gap_kinds = sorted({k for e in eps for k in e["gaps_ms"]})
    s2_lat = [float(c.get("latency_ms") or 0) for r in recs.values() if not r.get("error") for c in _calls(r)
              if c.get("component") == "s2"]
    out = {"n": n, "concurrency": concurrency,
           "success": _mean([1.0 if e["success"] else 0.0 for e in eps]),
           "wall_s": _mean([e["wall_ms"] / 1000 for e in eps]), "wall_s_median": _median([e["wall_ms"] / 1000 for e in eps]),
           "s1_s": _mean([e["s1_ms"] / 1000 for e in eps]), "s2_s": _mean([e["s2_ms"] / 1000 for e in eps]),
           "rest_s": _mean([e["rest_ms"] / 1000 for e in eps]),
           "gap_s": _mean([e["gap_total_ms"] / 1000 for e in eps]),
           "steps": _mean([e["steps"] for e in eps]),
           "s1_requests": _mean([e["s1_requests"] for e in eps]), "s1_questions": _mean([e["s1_questions"] for e in eps]),
           "s2_calls": _mean([e["s2_calls"] for e in eps]),
           "s2_ms_per_call": _mean(s2_lat), "s2_ms_per_call_median": _median(s2_lat),
           "s1_requests_per_step": _mean([s["s1_requests"] for s in steps]),
           "s1_ms_per_step": _mean([s["s1_ms"] for s in steps]),
           "s2_ms_per_step": _mean([s["s2_ms"] for s in steps]),
           "state_tokens_mean": _mean([s["state_tokens"] for s in steps if s["state_tokens"]]),
           "state_tokens_p90": _pct([s["state_tokens"] for s in steps if s["state_tokens"]], 0.9),
           "by_dp": {dp: {"requests": _mean([e["by_dp"].get(dp, {}).get("requests", 0) for e in eps]),
                          "questions": _mean([e["by_dp"].get(dp, {}).get("questions", 0) for e in eps]),
                          "s": _mean([e["by_dp"].get(dp, {}).get("ms", 0) / 1000 for e in eps]),
                          "ms_per_request": (sum(e["by_dp"].get(dp, {}).get("ms", 0) for e in eps)
                                             / max(1, sum(e["by_dp"].get(dp, {}).get("requests", 0) for e in eps)))}
                     for dp in dps},
           "gaps_s": {k: _mean([e["gaps_ms"].get(k, 0) / 1000 for e in eps]) for k in gap_kinds},
           "reflection": {"questions_per_episode": _mean([e["reflection_questions"] for e in eps]),
                          "questions": sum(e["reflection_questions"] for e in eps),
                          "fired": sum(e["reflection_fired"] for e in eps),
                          "p_yes_max": max([e["reflection_p_yes_max"] for e in eps] or [0.0])},
           "steps_with_two_or_more_argument_requests": _mean([e["accepted_steps_multi_param"] for e in eps])}
    return out


def _concurrency(run: Path, arm: str) -> int | None:
    f = run / SCHEDULE_FILES.get(arm, "schedule.json")
    try:
        return int(json.loads(f.read_text()).get("concurrency"))
    except Exception:  # noqa: BLE001
        return None


def diagnose(stage2: Path = STAGE2, dev_run: Path = DEV_TARIFF_RUN, out: Path = OUT) -> dict[str, Any]:
    res: dict[str, Any] = {"label": LABEL, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "git": _git_head(), "arms": {}, "inputs": {}}
    sets: list[tuple[str, Path, str]] = [(a, stage2, a) for a in DIAG_ARMS]
    if (dev_run / "jevagent_best_rl.jsonl").exists():
        sets.append(("dev_tariff_rl", dev_run, "jevagent_best_rl"))
    for key, run, arm in sets:
        p = run / f"{arm}.jsonl"
        if not p.exists():
            continue
        recs = load_latest(p)
        res["inputs"][key] = {"path": _rel(p), "sha256": _sha256(p)[:16]}
        a = summarise_arm(recs, _concurrency(run, arm))
        if arm == "jevagent_best_rl":
            a["queue"] = queue_reconstruction(recs.values())
            a["closed_loop"] = closed_loop(a)
        res["arms"][key] = a
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency_diagnosis.json").write_text(json.dumps(_jsonable(res), indent=1))
    (out / "latency_diagnosis.md").write_text(diagnosis_markdown(res))
    return res


def _speed(saving_frac: float) -> str:
    """'12.3 % faster' or '157.4 % slower' (a negative saving is a slowdown)."""
    return f"{100 * saving_frac:.1f} % faster" if saving_frac >= 0 else f"{-100 * saving_frac:.1f} % slower"


def _f(v: Any, nd: int = 2, unit: str = "") -> str:
    if v is None:
        return "–"
    return f"{v:.{nd}f}{unit}"


def diagnosis_markdown(res: Mapping[str, Any]) -> str:
    A = res["arms"]
    L = ["# Why the cheaper profiles were slower: latency diagnosis of the Stage-2 logs", "",
         f"Label: {res['label']}. Generated {res['generated']} at git {res['git'][:12]} by "
         "`python scripts/harness_final.py diagnose`; inputs: " +
         ", ".join(f"`{v['path']}` ({v['sha256']})" for v in res["inputs"].values()) + ".", "",
         "## 1. Where an episode's time goes (s per episode, means)", "",
         "| arm | conc. | n | success | wall | System-1 | System-2 | rest (tools, harness, cache) | steps | S1 requests | "
         "S1 questions | S2 calls | S2 ms / call (median) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, a in A.items():
        L.append(f"| `{k}` | {a['concurrency']} | {a['n']} | {_f(100 * a['success'], 1)} | {_f(a['wall_s'])} | "
                 f"{_f(a['s1_s'])} | {_f(a['s2_s'])} | {_f(a['rest_s'])} | {_f(a['steps'])} | {_f(a['s1_requests'])} | "
                 f"{_f(a['s1_questions'])} | {_f(a['s2_calls'])} | {_f(a['s2_ms_per_call'], 0)} "
                 f"({_f(a['s2_ms_per_call_median'], 0)}) |")
    L += ["", "System-1 and System-2 calls of one episode are sequential in the reference loop, so wall ≈ System-1 + "
          "System-2 + rest. `jevagent_best_rl` and `dev_tariff_rl` ran on the GPU host (AWS, concurrency 4, one "
          "JevAgentRL server); the other arms ran in another invocation and location (concurrency 8, hosted Jev): "
          "their System-2 calls were slower per call, so cross-invocation differences mix the arm with the serving "
          "conditions.", "", "## 2. System-1 requests by decision point (per episode)", "",
          "| arm | decision point | requests | questions | s | ms / request |", "|---|---|---|---|---|---|"]
    for k, a in A.items():
        for dp, d in a["by_dp"].items():
            L.append(f"| `{k}` | {dp} | {_f(d['requests'])} | {_f(d['questions'])} | {_f(d['s'])} | "
                     f"{_f(d['ms_per_request'], 0)} |")
    L += ["", "Every System-1 request carried exactly one question (the registered call graph, A6.22 (a)).", "",
          "## 3. Per step", "", "| arm | S1 requests / step | S1 ms / step | S2 ms / step | state tokens mean (p90) |",
          "|---|---|---|---|---|"]
    for k, a in A.items():
        L.append(f"| `{k}` | {_f(a['s1_requests_per_step'])} | {_f(a['s1_ms_per_step'], 0)} | "
                 f"{_f(a['s2_ms_per_step'], 0)} | {_f(a['state_tokens_mean'], 0)} ({_f(a['state_tokens_p90'], 0)}) |")
    for k in ("jevagent_best_rl", "dev_tariff_rl"):
        q = (A.get(k) or {}).get("queue")
        if not q:
            continue
        L += ["", f"## 4. Queueing at the single JevAgentRL server (`{k}`)", "",
              f"{q['requests']} requests over {_f(q['span_s'], 0)} s; the server was busy (≥ 1 request in flight) "
              f"{_f(100 * q['busy_fraction'], 1)} % of the time. Reconstructed mean service {_f(q['service_ms_mean'], 0)} ms "
              f"and queueing {_f(q['queue_ms_mean'], 0)} ms per request: {_f(100 * q['queue_share'], 1)} % of the "
              "System-1 latency was waiting for the other episodes' requests.", "",
              "| decision point | requests | service ms mean (median) | queueing ms mean |", "|---|---|---|---|"]
        for dp, d in q["by_dp"].items():
            L.append(f"| {dp} | {d['requests']} | {_f(d['service_ms_mean'], 0)} ({_f(d['service_ms_median'], 0)}) | "
                     f"{_f(d['queue_ms_mean'], 0)} |")
    L += ["", "## 5. Gaps between calls (s per episode)", "", "| arm | " + " | ".join(
        sorted({g for a in A.values() for g in a["gaps_s"]})) + " |",
          "|---|" + "---|" * len({g for a in A.values() for g in a["gaps_s"]})]
    kinds = sorted({g for a in A.values() for g in a["gaps_s"]})
    for k, a in A.items():
        L.append(f"| `{k}` | " + " | ".join(_f(a["gaps_s"].get(g), 3) for g in kinds) + " |")
    L += ["", "## 6. The reflection trigger", "", "| arm | questions / episode | questions | fired | max p(yes) |",
          "|---|---|---|---|---|"]
    for k, a in A.items():
        r = a["reflection"]
        if r["questions"]:
            L.append(f"| `{k}` | {_f(r['questions_per_episode'])} | {r['questions']} | {r['fired']} | "
                     f"{_f(r['p_yes_max'], 4)} |")
    L += ["", "## 7. Reading", ""]
    L += reading(res)
    return "\n".join(L) + "\n"


def reading(res: Mapping[str, Any]) -> list[str]:
    A = res["arms"]
    lo, rl = A.get("llm_only"), A.get("jevagent_best_rl")
    if not lo or not rl:
        return ["(the reference or the JevAgentRL arm is missing)"]
    q = rl.get("queue") or {}
    by = rl["by_dp"]
    out = [f"- **The LLM calls JevAgentRL removed were cheap in time.** `jevagent_best_rl` made "
           f"{_f(rl['s2_calls'])} System-2 calls per episode instead of {_f(lo['s2_calls'])} "
           f"({_f(rl['s2_s'])} s instead of {_f(lo['s2_s'])} s), but it spent {_f(rl['s1_s'])} s in "
           f"{_f(rl['s1_requests'])} System-1 requests ({_f(rl['s1_requests_per_step'])} per step, one question each).",
           f"- **Request count.** Per episode: reflection {_f(by.get('reflection', {}).get('requests'))}, tool choice "
           f"{_f(by.get('tool_select', {}).get('requests'))}, arguments {_f(by.get('tool_params', {}).get('requests'))} "
           f"requests. The reflection trigger was asked {rl['reflection']['questions']} times and never fired (max "
           f"p(yes) {_f(rl['reflection']['p_yes_max'], 4)} vs τ 0.5); "
           f"{_f(rl['steps_with_two_or_more_argument_requests'])} steps per episode sent two or more argument "
           "questions one request at a time although they share the state."]
    if q:
        out.append(f"- **Queueing on one GPU.** Four episodes shared one server that answers one request at a time: it "
                   f"was busy {_f(100 * q['busy_fraction'], 1)} % of the run and {_f(100 * q['queue_share'], 1)} % of "
                   f"the System-1 latency was queueing (mean service {_f(q['service_ms_mean'], 0)} ms, queueing "
                   f"{_f(q['queue_ms_mean'], 0)} ms per request). The tool choice (69 options: a two-round knockout "
                   f"over chunks of 52 with every option's description re-read after the state at every step) is the "
                   f"most expensive request (service {_f(q['by_dp'].get('tool_select', {}).get('service_ms_mean'), 0)} ms).")
    cl = rl.get("closed_loop")
    if cl:
        out.append(f"- **A closed loop at saturation.** With {cl['N']} episodes, a server busy time of "
                   f"{_f(1000 * cl['service_s_per_step'], 0)} ms per step and {_f(1000 * cl['think_s_per_step'], 0)} ms "
                   f"of other work per step, the interactive-system law R = N·S − Z predicts "
                   f"{_f(1000 * cl['predicted_s1_s_per_step'], 0)} ms of System-1 time per step; the logs show "
                   f"{_f(1000 * cl['observed_s1_s_per_step'], 0)} ms. One GPU serving this path completes at most "
                   f"{_f(cl['capacity_steps_per_s'], 1)} steps per second, so the episode time is set by the server's "
                   "work per step times the number of episodes sharing it, not by the per-request latency.")
    out.append(f"- **Rest.** {_f(rl['rest_s'])} s per episode outside model calls vs {_f(lo['rest_s'])} s for "
               "LLM-only: the System-1 decision cache (one SQLite read, insert and commit per request), the gold "
               "bookkeeping of the extra decision logs and the per-request HTTP handling.")
    return out


# ------------------------------------------------------------------ replay workload
def _env_tools(task_id: str, suite: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Tool names and descriptions (as the agent sends them) of a JevGym task, regenerated from its seed."""
    from jevharness.bench.suites import load_suite
    from jevharness.harness.tools import finish_tool, tool_desc, tool_name

    seed = int(task_id.split("-")[2])
    kw = {k: v for k, v in suite.items() if k not in ("name", "n", "seed0")}
    env = load_suite("jevgym", n=1, seed0=seed, **kw)[0]()
    assert env.task_id == task_id, (env.task_id, task_id)
    env.reset()
    tools = env.tools() + [finish_tool("finish")]
    return [tool_name(t) for t in tools], [tool_desc(t) for t in tools]


def _wire(q: Any) -> dict[str, Any]:
    from jevharness.decision.jev import question_to_wire

    return question_to_wire(q)


def build_workload(run: Path = DEV_TARIFF_RUN, arm: str = "jevagent_best_rl",
                   out: Path = OUT / "replay_workload.jsonl.gz", limit: int | None = None) -> dict[str, Any]:
    """Replayable System-1 requests of a logged run (see the module docstring)."""
    import yaml

    from jevharness.decision.types import Choice, Noul
    from jevharness.harness.prompts import Q_REFLECTION, Q_TOOL_SELECT

    exp = yaml.safe_load((run / "experiment.yaml").read_text())
    suite = exp["suite"]
    recs = load_latest(run / f"{arm}.jsonl")
    n_ep = n_req_old = n_req_new = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        for tid, r in sorted(recs.items()):
            if r.get("error"):
                continue
            if limit is not None and n_ep >= limit:
                break
            names, descs = _env_tools(tid, suite)
            calls = _calls(r)
            ep = {"task_id": tid, "steps": []}
            for s in r.get("steps") or []:
                step = s["step"]
                sc = [c for c in calls if c.get("step") == step]
                s2_ms = sum(float(c.get("latency_ms") or 0) for c in sc if c.get("component") == "s2")
                dls = [d for d in s.get("decisions") or [] if d.get("s1_calls") and not d.get("shadow")]
                old: list[dict[str, Any]] = []
                params: list[dict[str, Any]] = []
                tool = None
                for d in dls:
                    st = d["state"]
                    if d["dp"] == "reflection":
                        q = _wire(Noul(Q_REFLECTION))
                        old.append({"state": st, "questions": {"q0": q}, "dp": "reflection",
                                    "logged": [d.get("s1_dist")]})
                    elif d["dp"] == "tool_select":
                        assert d.get("options") == names, (tid, step)
                        q = _wire(Choice(Q_TOOL_SELECT, names, descs))
                        tool = {"state": st, "questions": {"q0": q}, "dp": "tool_select", "logged": [d.get("s1_dist")],
                                "tau": None}
                        old.append(tool)
                    elif d["dp"] == "tool_params":
                        q = _wire(Choice(d["question"], list(d["options"])))
                        item = {"state": st, "questions": {"q0": q}, "dp": "tool_params", "logged": [d.get("s1_dist")]}
                        old.append(item)
                        params.append(item)
                new = [x for x in old if x["dp"] == "tool_select"]
                if params:
                    assert len({p["state"] for p in params}) == 1
                    new.append({"state": params[0]["state"], "dp": "tool_params",
                                "questions": {f"q{i}": p["questions"]["q0"] for i, p in enumerate(params)},
                                "logged": [p["logged"][0] for p in params]})
                s1_ms = sum(float(c.get("latency_ms") or 0) for c in sc if c.get("component") == "s1")
                ep["steps"].append({"step": step, "old": old, "new": new, "logged_s1_ms": s1_ms, "s2_ms": s2_ms})
                n_req_old += len(old)
                n_req_new += len(new)
            ep["rest_ms"] = float(r.get("wall_ms") or 0) - sum(float(c.get("latency_ms") or 0) for c in calls)
            f.write(json.dumps(ep) + "\n")
            n_ep += 1
    meta = {"run": _rel(run / f"{arm}.jsonl"), "sha256": _sha256(run / f"{arm}.jsonl")[:16], "episodes": n_ep,
            "requests_old": n_req_old, "requests_new": n_req_new, "out": _rel(out), "out_sha256": _sha256(out)[:16]}
    (out.parent / "replay_workload.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_workload(path: Path = OUT / "replay_workload.jsonl.gz") -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ decision equivalence
TAU_TS = 0.4999999999978404          # the frozen profile's backends.jevagent_rl thresholds (configs/harness/jevagent_best.yaml)
TAU_TP = 0.9999998807907104
OTHER = "(other)"


def _dist(wire_answer: Mapping[str, Any], q: Mapping[str, Any]) -> dict[str, float]:
    if q.get("type") == "noul":
        p = float(wire_answer.get("noul", 0.5))
        return {"yes": p, "no": 1.0 - p}
    return {str(k): float(v) for k, v in (wire_answer.get("probabilities") or {}).items()}


def _decision(dp: str, dist: Mapping[str, float]) -> tuple[Any, bool]:
    """(argmax value, accepted by the frozen cascade) of one answer."""
    if not dist:
        return None, False
    v = max(dist, key=lambda k: dist[k])
    c = dist[v]
    if dp == "tool_select":
        return v, c >= TAU_TS
    if dp == "tool_params":
        return v, v != OTHER and c >= TAU_TP
    if dp == "reflection":
        return dist.get("yes", 0.0) >= 0.5, dist.get("yes", 0.0) >= 0.5
    return v, False


def compare(dp: str, a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    keys = set(a) | set(b)
    return {"max_abs": max([abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys] or [0.0]),
            "same_argmax": _decision(dp, a)[0] == _decision(dp, b)[0],
            "same_decision": _decision(dp, a) == _decision(dp, b)}


# ------------------------------------------------------------------ bench (GPU host)
THINK_CAP_STEP_MS = 5000.0     # a step's emulated System-2 time is capped (the dev run had one 50 s call)
THINK_CAP_REST_MS = 2000.0     # an episode's emulated rest is capped (failed-attempt gaps of up to 68 s)


def bench(url: str, label: str, path: str, concurrencies: Sequence[int], workload: Path, out_dir: Path,
          think: bool = True, limit: int | None = None, timeout: float = 120.0) -> list[dict[str, Any]]:
    """Replay ``path`` ("old" | "new") requests of the workload, ``c`` episodes at a time (each episode's steps in
    order), recording client latency, server usage and answers (decision equivalence vs the logged answers)."""
    import httpx

    from jevharness.decision.jev import replica_index

    eps = load_workload(workload)[:limit] if limit else load_workload(workload)
    urls = [u.strip().rstrip("/") for u in url.split(",") if u.strip()]
    results = []
    for c in concurrencies:
        clients = [httpx.Client(base_url=u, timeout=timeout,
                                limits=httpx.Limits(max_connections=64, max_keepalive_connections=64)) for u in urls]

        def health() -> dict[str, float]:
            hs = [cl.get("/health").json() for cl in clients]
            return {k: sum(float(h.get(k, 0) or 0) for h in hs) for k in hs[0] if k != "status"}

        def models() -> dict[str, Any]:
            ms = [cl.get("/v1/models").json() for cl in clients]
            out = ms[0]
            if len(ms) > 1:   # phase timers summed over replicas
                ph: dict[str, float] = {}
                for m in ms:
                    for k, v in (((m.get("data") or [{}])[0].get("backend") or {}).get("phase_ms") or {}).items():
                        ph[k] = ph.get(k, 0.0) + v
                if ph:
                    out["data"][0]["backend"]["phase_ms"] = ph
                out["data"][0]["backend"]["replicas"] = len(ms)
            return out

        health0 = health()
        models0 = models()
        lock = threading.Lock()
        rows: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        ep_rows: list[dict[str, Any]] = []
        todo = list(range(len(eps)))

        def worker() -> None:
            while True:
                with lock:
                    if not todo:
                        return
                    i = todo.pop(0)
                e = eps[i]
                t_ep = time.perf_counter()
                s1_total = 0.0
                for s in e["steps"]:
                    t_step = time.perf_counter()
                    for req in s[path]:
                        body = {"state": req["state"], "model": "replay", "questions": req["questions"]}
                        t = time.perf_counter()
                        r = clients[replica_index(req["state"], len(clients))].post("/v1/systemone", json=body)
                        ms = (time.perf_counter() - t) * 1000
                        r.raise_for_status()
                        d = r.json()
                        u = d.get("usage") or {}
                        cmp = []
                        for (qid, q), logged in zip(req["questions"].items(), req["logged"]):
                            got = _dist(d["answers"][qid], q)
                            cmp.append({**compare(req["dp"], got, {k: float(v) for k, v in (logged or {}).items()}),
                                        "dist": got, "q": q.get("instructions")})
                        with lock:
                            rows.append({"task_id": e["task_id"], "step": s["step"], "dp": req["dp"], "ms": ms,
                                         "n_questions": len(req["questions"]),
                                         "server_ms": _server_ms(r.headers.get("server-timing")),
                                         "prefilled_tokens": u.get("prefilled_tokens"),
                                         "cache_hit_tokens": u.get("cache_hit_tokens"),
                                         "stateless_tokens": u.get("stateless_tokens"), "n_passes": u.get("n_passes"),
                                         "cmp": cmp})
                    s1_ms = (time.perf_counter() - t_step) * 1000
                    s1_total += s1_ms
                    with lock:
                        step_rows.append({"task_id": e["task_id"], "step": s["step"], "s1_ms": s1_ms,
                                          "n_requests": len(s[path])})
                    if think and s["s2_ms"] > 0:   # the step's logged System-2 time (outliers capped)
                        time.sleep(min(s["s2_ms"], THINK_CAP_STEP_MS) / 1000)
                if think and e.get("rest_ms", 0) > 0:   # tools and harness (retry outliers capped)
                    time.sleep(min(max(0.0, e["rest_ms"]), THINK_CAP_REST_MS) / 1000)
                with lock:
                    ep_rows.append({"task_id": e["task_id"], "s1_ms": s1_total,
                                    "wall_ms": (time.perf_counter() - t_ep) * 1000})

        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker) for _ in range(c)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        span = time.perf_counter() - t0
        health1 = health()
        mods = models()
        for cl in clients:
            cl.close()
        res = {"label": label, "path": path, "concurrency": c, "think": think, "episodes": len(ep_rows),
               "span_s": span, "health_before": health0, "health_after": health1,
               "server": (mods.get("data") or [{}])[0].get("backend"), "replicas": len(urls),
               "server_before": (models0.get("data") or [{}])[0].get("backend"), "requests": rows, "steps": step_rows,
               "episodes_rows": ep_rows, "workload": _rel(workload), "url": url,
               "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"bench_{label}_{path}_c{c}{'' if think else '_nothink'}.json").write_text(json.dumps(res))
        print(json.dumps(bench_summary(res)), flush=True)
        results.append(res)
    return results


def _server_ms(h: str | None) -> float | None:
    if not h or "dur=" not in h:
        return None
    try:
        return float(h.split("dur=")[1].split(",")[0].split(";")[0])
    except ValueError:
        return None


def bench_summary(res: Mapping[str, Any]) -> dict[str, Any]:
    rows, steps, eps = res["requests"], res["steps"], res["episodes_rows"]
    cmps = [c for r in rows for c in r["cmp"]]
    by_dp: dict[str, Any] = {}
    for dp in sorted({r["dp"] for r in rows}):
        rr = [r for r in rows if r["dp"] == dp]
        cc = [c for r in rr for c in r["cmp"]]
        by_dp[dp] = {"requests": len(rr), "ms_mean": _mean([r["ms"] for r in rr]), "ms_p50": _median([r["ms"] for r in rr]),
                     "server_ms_mean": _mean([r["server_ms"] for r in rr if r["server_ms"] is not None]),
                     "questions": sum(r["n_questions"] for r in rr),
                     "decision_flips": sum(1 for c in cc if not c["same_decision"]),
                     "argmax_flips": sum(1 for c in cc if not c["same_argmax"]),
                     "max_abs_dp": max([c["max_abs"] for c in cc] or [0.0])}
    h0, h1 = res.get("health_before") or {}, res.get("health_after") or {}
    hit = (h1.get("cache_hit_tokens", 0) - h0.get("cache_hit_tokens", 0))
    pre = (h1.get("prefilled_tokens", 0) - h0.get("prefilled_tokens", 0))
    sl = (h1.get("stateless_tokens", 0) - h0.get("stateless_tokens", 0))
    ph1 = (res.get("server") or {}).get("phase_ms") or {}
    ph0 = (res.get("server_before") or {}).get("phase_ms") or {}
    n_steps = max(1, len(steps))
    phases = {k: (ph1[k] - ph0.get(k, 0.0)) / n_steps for k in ph1} if ph1 else None
    return {"label": res["label"], "path": res["path"], "concurrency": res["concurrency"], "think": res["think"],
            "phase_ms_per_step": phases,
            "episodes": res["episodes"], "span_s": res["span_s"],
            "throughput_eps_per_h": 3600 * res["episodes"] / res["span_s"] if res["span_s"] else None,
            "requests_per_episode": len(rows) / max(1, res["episodes"]),
            "s1_ms_per_step": _mean([s["s1_ms"] for s in steps]), "s1_ms_per_step_p50": _median([s["s1_ms"] for s in steps]),
            "s1_s_per_episode": _mean([e["s1_ms"] / 1000 for e in eps]),
            "wall_s_per_episode": _mean([e["wall_ms"] / 1000 for e in eps]),
            "ms_per_request": _mean([r["ms"] for r in rows]),
            "decision_flips": sum(1 for c in cmps if not c["same_decision"]),
            "argmax_flips": sum(1 for c in cmps if not c["same_argmax"]), "questions": len(cmps),
            "max_abs_dp": max([c["max_abs"] for c in cmps] or [0.0]),
            "server_requests": h1.get("requests", 0) - h0.get("requests", 0),
            "server_passes": h1.get("passes", 0) - h0.get("passes", 0),
            "prefilled_tokens": pre, "cache_hit_tokens": hit, "stateless_tokens": sl,
            "cache_hit_share": hit / sl if sl else None, "by_dp": by_dp}


# ------------------------------------------------------------------ CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--stage2", default=str(STAGE2))
    d.add_argument("--dev-run", default=str(DEV_TARIFF_RUN))
    d.add_argument("--out", default=str(OUT))
    w = sub.add_parser("workload")
    w.add_argument("--run", default=str(DEV_TARIFF_RUN))
    w.add_argument("--out", default=str(OUT / "replay_workload.jsonl.gz"))
    w.add_argument("--limit", type=int, default=None)
    b = sub.add_parser("bench")
    b.add_argument("--url", default="http://127.0.0.1:8081")
    b.add_argument("--label", required=True)
    b.add_argument("--path", choices=["old", "new"], required=True)
    b.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    b.add_argument("--workload", default=str(OUT / "replay_workload.jsonl.gz"))
    b.add_argument("--out", default=str(OUT / "bench"))
    b.add_argument("--no-think", action="store_true")
    b.add_argument("--limit", type=int, default=None)
    br = sub.add_parser("bench-report")
    br.add_argument("--dir", default=str(OUT / "bench"))
    br.add_argument("--out", default=str(OUT))
    an = sub.add_parser("analyze")
    an.add_argument("--run", default=str(ROOT / "runs/harness_final_live"))
    an.add_argument("--out", default=str(OUT))
    an.add_argument("--tex", default=str(TEX_OUT))
    an.add_argument("--tariff", default=str(OUT / "tariff.json"))
    an.add_argument("--n-expected", type=int, default=200)
    an.add_argument("--name", default="summary", choices=[n for n, _, _ in ROWS])
    po = sub.add_parser("positions")
    po.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    po.add_argument("--limit", type=int, default=None)
    tf = sub.add_parser("tariff")
    tf.add_argument("--run", default=str(ROOT / "runs/harness_final_tariff"))
    tf.add_argument("--window", default=str(OUT / "gpu/tariff_window.json"))
    tf.add_argument("--out", default=str(OUT / "tariff.json"))
    a = ap.parse_args(argv)
    if a.cmd == "diagnose":
        res = diagnose(Path(a.stage2), Path(a.dev_run), Path(a.out))
        write_tex(Path(a.out))
        print(f"wrote {Path(a.out) / 'latency_diagnosis.md'} ({len(res['arms'])} arms)")
    elif a.cmd == "workload":
        print(json.dumps(build_workload(Path(a.run), out=Path(a.out), limit=a.limit)))
    elif a.cmd == "bench":
        bench(a.url, a.label, a.path, a.concurrency, Path(a.workload), Path(a.out), think=not a.no_think,
              limit=a.limit)
    elif a.cmd == "bench-report":
        bench_report(Path(a.dir), Path(a.out))
        write_tex(Path(a.out))
        print((Path(a.out) / "replay_bench.md").read_text())
    elif a.cmd == "analyze":
        analyze(Path(a.run), Path(a.out), Path(a.tex), tariff=Path(a.tariff), n_expected=a.n_expected, name=a.name)
        print((Path(a.out) / f"{a.name}.md").read_text())
    elif a.cmd == "positions":
        print(json.dumps(positions(a.tokenizer, limit=a.limit)["rows"], indent=1))
        write_tex()
    elif a.cmd == "tariff":
        print(json.dumps(tariff(Path(a.run), Path(a.window), Path(a.out)), indent=1))
        write_tex(Path(a.out).parent)
    return 0


# ------------------------------------------------------------------ compute per step (hardware-free)
QWEN3_4B = {"layers": 36, "hidden": 2560, "q_dim": 4096, "kv_dim": 1024, "intermediate": 9728, "vocab": 151936}


def flops_per_position(c: Mapping[str, int] = QWEN3_4B) -> float:
    """Dense matmul FLOPs of one token position through the Qwen3-4B body (attention scores excluded)."""
    per_layer = c["hidden"] * (c["q_dim"] + 2 * c["kv_dim"]) + c["q_dim"] * c["hidden"] + 3 * c["hidden"] * c["intermediate"]
    return 2.0 * c["layers"] * per_layer


def positions(tokenizer: str = "Qwen/Qwen3-0.6B", workload: Path = OUT / "replay_workload.jsonl.gz",
              out: Path = OUT / "positions.json", limit: int | None = None) -> dict[str, Any]:
    """Token positions each server version runs through the model per step when it replays the workload (the Stage-2
    server with the old call graph vs ``serve_jev --fast`` with the new one), counted on a tiny random model that uses
    the served model's tokenizer (Qwen3), so prefixes, suffixes, cache hits and paddings are exact; the GPU time is
    not (that is ``bench``)."""
    import tempfile

    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from jevharness.training.serve_jev import Engine

    eps = load_workload(workload)[:limit] if limit else load_workload(workload)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    d = Path(tempfile.mkdtemp(prefix="hf_tiny_"))
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, tie_word_embeddings=True,
                      max_position_embeddings=32768)
    Qwen3ForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    res: dict[str, Any] = {"tokenizer": tokenizer, "episodes": len(eps), "flops_per_position": flops_per_position(),
                           "lm_head_flops_per_row": 2.0 * QWEN3_4B["hidden"] * QWEN3_4B["vocab"], "rows": {}}
    for tag, path, fast in (("v0_old", "old", False), ("v0_new", "new", False), ("v2_new", "new", True)):
        eng = Engine(str(d), name="tiny", max_len=4096, session_cache=True, device="cpu",
                     **({"fuse_multi": True, "suffix_pad_slack": 0.25, "readout_gather": True, "preencode": True}
                        if fast else {}))
        cnt = {"forwards": 0, "positions": 0, "head_rows": 0}

        def pre(mod, args, kwargs, cnt=cnt):  # noqa: ANN001, ANN202
            x = kwargs.get("input_ids", args[0] if args else None)
            cnt["forwards"] += 1
            cnt["positions"] += int(x.shape[0] * x.shape[1])

        def head(mod, args, out, cnt=cnt):  # noqa: ANN001, ANN202
            cnt["head_rows"] += int(out.shape[0] * out.shape[1])

        eng.model.register_forward_pre_hook(pre, with_kwargs=True)
        eng.model.lm_head.register_forward_hook(head)
        n_steps = n_req = 0
        for e in eps:
            for s in e["steps"]:
                n_steps += 1
                for req in s[path]:
                    eng.answer_all(req["state"], req["questions"])
                    n_req += 1
        st_ = eng.stats
        res["rows"][tag] = {"steps": n_steps, "requests": n_req, **cnt,
                            "positions_per_step": cnt["positions"] / n_steps, "forwards_per_step": cnt["forwards"] / n_steps,
                            "head_rows_per_step": cnt["head_rows"] / n_steps,
                            "gflop_per_step": (cnt["positions"] * res["flops_per_position"]
                                               + cnt["head_rows"] * res["lm_head_flops_per_row"]) / n_steps / 1e9,
                            "cache_hit_share": st_["cache_hit_tokens"] / st_["stateless_tokens"] if st_["stateless_tokens"] else None,
                            "prefilled_tokens": st_["prefilled_tokens"], "stateless_tokens": st_["stateless_tokens"]}
    out.write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ bench report
LOGGED: dict[tuple, tuple[Any, bool]] = {}


def logged_decisions(run: Path = DEV_TARIFF_RUN, arm: str = "jevagent_best_rl") -> dict[tuple, tuple[Any, bool]]:
    """(task, step, dp, question) -> (value, acted on) of every logged System-1 decision of the replayed run: the
    harness's own outcome (``s1_value``, ``not deferred`` / the trigger's ``final_value``), which used the answer's
    full-precision confidence; the logged ``s1_dist`` is stored smoothed, so it cannot decide a threshold at 1 - 1e-7."""
    out: dict[tuple, tuple[Any, bool]] = {}
    for tid, r in load_latest(run / f"{arm}.jsonl").items():
        for st in r.get("steps") or []:
            for dl in st.get("decisions") or []:
                if not dl.get("s1_calls") or dl.get("shadow"):
                    continue
                q = None if dl["dp"] == "tool_select" else dl["question"]
                acted = bool(dl.get("final_value")) if dl["dp"] == "reflection" else not dl.get("deferred")
                out[(tid, st["step"], dl["dp"], q)] = (dl.get("s1_value"), acted)
    return out


def _replay_decision(dp: str, dist: Mapping[str, float]) -> tuple[Any, bool]:
    v, acted = _decision(dp, dist)
    return (bool(acted), acted) if dp == "reflection" else (v, acted)


def bench_report(d: Path = OUT / "bench", out: Path = OUT) -> dict[str, Any]:
    """Every ``bench_*.json`` of the GPU host -> ``replay_bench.{json,md}``: one row per (server, call graph,
    concurrency) and the cross-replay decision equivalence (each new-graph answer vs the same question's old-graph
    answer on the Stage-2 server)."""
    global LOGGED
    LOGGED = logged_decisions()
    files = sorted(d.glob("bench_*.json"))
    rows, dists = [], {}
    for f in files:
        res = json.loads(f.read_text())
        sm = bench_summary(res)
        srv = res.get("server") or {}
        sm["server_flags"] = {k: srv.get(k) for k in ("fuse_multi", "suffix_pad_slack", "readout_gather",
                                                      "batch_suffixes", "session_cache")}
        sm["file"] = _rel(f)
        # decisions vs the LOGGED decisions (value, acted on) of the replayed run, on the replay's own answers
        n_q = fl = fl_arg = 0
        by: dict[str, list[int]] = {}
        for r in res["requests"]:
            for c in r["cmp"]:
                key = (r["task_id"], r["step"], r["dp"], None if r["dp"] == "tool_select" else c.get("q"))
                lg = LOGGED.get(key)
                if lg is None:
                    continue
                got = _replay_decision(r["dp"], c["dist"])
                n_q += 1
                bad = (got[1] != lg[1]) or (got[1] and r["dp"] != "reflection" and str(got[0]) != str(lg[0]))
                fl += bad
                by.setdefault(r["dp"], [0, 0])
                by[r["dp"]][0] += 1
                by[r["dp"]][1] += bad
        sm["vs_logged_decisions"] = {"questions": n_q, "flips": fl, "by_dp": by}
        rows.append(sm)
        dd = {}
        # per question: (task, step, dp, question text + options) -> distribution
        for r in res["requests"]:
            for c in r["cmp"]:
                dd[json.dumps([r["task_id"], r["step"], r["dp"], c.get("q"), sorted(c["dist"])])] = c["dist"]
        dists[(res["label"], res["path"], res["concurrency"])] = dd
    cross = []
    ref_keys = [k for k in dists if k[0] == "v0" and k[1] == "old"]
    for k, dd in dists.items():
        if k[0] == "v0" and k[1] == "old":
            continue
        for rk in ref_keys:
            ref = dists[rk]
            shared = [x for x in dd if x in ref]
            if not shared:
                continue
            flips = sum(1 for x in shared if not compare(json.loads(x)[2], dd[x], ref[x])["same_decision"])
            mx = max(compare(json.loads(x)[2], dd[x], ref[x])["max_abs"] for x in shared)
            cross.append({"run": list(k), "ref": list(rk), "questions": len(shared), "decision_flips": flips,
                          "max_abs_dp": mx})
    rep = {"label": LABEL, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "workload": json.loads((OUT / "replay_workload.json").read_text()) if (OUT / "replay_workload.json").exists()
           else None, "rows": rows, "cross": cross}
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay_bench.json").write_text(json.dumps(_jsonable(rep), indent=1))
    (out / "replay_bench.md").write_text(bench_markdown(rep))
    return rep


def bench_markdown(rep: Mapping[str, Any]) -> str:
    L = ["# Offline serving benchmark: replayed System-1 requests (no LLM call)", "",
         f"Label: {rep['label']}. Generated {rep['generated']} by `python scripts/harness_final.py bench-report` from "
         "the GPU host's `bench_*.json` (`scripts/aws/harness_final_gpu.sh bench`). Workload: "
         + (f"`{rep['workload']['run']}` ({rep['workload']['episodes']} development episodes; "
            f"{rep['workload']['requests_old']} requests in the old call graph, {rep['workload']['requests_new']} in the "
            "new one)" if rep.get("workload") else "–") + ". Episodes replayed `c` at a time, steps in order, with each "
         "step's logged System-2 time (capped at 5 s) and the episode's rest (capped at 2 s) slept in between; c = 1 "
         "replays the first 25 episodes.", "",
         "Servers: `v0` = the Stage-2 server (serve_jev defaults); `v2` = `serve_jev --fast` (fused multi-question pass, "
         "length-grouped suffix batches, LM head at the read-out positions only, prompts tokenized outside the engine "
         "lock); `p` = the same with device-synchronised phase timers. Host: one NVIDIA A10G 24 GB (g5.4xlarge); S2-4 "
         "ran on one L40S. Two fast replicas on the one GPU ran out of memory (not measured). Call graphs: `old` = reflection "
         "question + tool choice + one request per argument question (Stage 2, S2-4); `new` = tool choice + all argument "
         "questions of the step in one request (the final harness).", "",
         "| server | graph | c | episodes | S1 requests / ep. | S1 ms / step (p50) | S1 s / ep. | emulated wall s / ep. | "
         "episodes / GPU-hour | cache-hit share | decisions differing from the logged ones / questions |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rep["rows"], key=lambda r: (r["label"], r["path"], r["concurrency"])):
        vl = r.get("vs_logged_decisions") or {}
        L.append(f"| {r['label']} | {r['path']} | {r['concurrency']} | {r['episodes']} | {_f(r['requests_per_episode'])} | "
                 f"{_f(r['s1_ms_per_step'], 0)} ({_f(r['s1_ms_per_step_p50'], 0)}) | {_f(r['s1_s_per_episode'])} | "
                 f"{_f(r['wall_s_per_episode'])} | {_f(r['throughput_eps_per_h'], 0)} | "
                 f"{_f(100 * r['cache_hit_share'], 1) if r.get('cache_hit_share') is not None else '–'} % | "
                 f"{vl.get('flips', '–')} / {vl.get('questions', '–')} |")
    L += ["", "The logged decisions are the harness's own outcomes in the replayed Stage-2 development run (one L40S): the "
          "chosen value and whether the frozen cascade acted on it (tool choice τ ≈ 0.5, arguments τ ≈ 1 − 1.2e-7, the "
          "reflection trigger τ 0.5). A differing decision on this host can come from the GPU (A10G vs L40S bf16 "
          "numerics), the batch composition, or a knockout near-tie; the cross-replay table below isolates the server "
          "and call graph on one host."]
    L += ["", "Per decision point (ms per request, mean; server time in parentheses):", "",
          "| server | graph | c | " + " | ".join(("tool_select", "tool_params", "reflection")) + " |", "|---|---|---|---|---|---|"]
    for r in sorted(rep["rows"], key=lambda r: (r["label"], r["path"], r["concurrency"])):
        cells = []
        for dp in ("tool_select", "tool_params", "reflection"):
            b = r["by_dp"].get(dp)
            cells.append(f"{_f(b['ms_mean'], 0)} ({_f(b['server_ms_mean'], 0)})" if b else "–")
        L.append(f"| {r['label']} | {r['path']} | {r['concurrency']} | " + " | ".join(cells) + " |")
    pos = OUT / "positions.json"
    if pos.exists():
        P = json.loads(pos.read_text())
        L += ["", f"Work per step, hardware-free (`python scripts/harness_final.py positions`: the same {P['episodes']} "
              "episodes replayed through a tiny random model with the served model's tokenizer, so token positions, "
              "forward passes and LM-head rows are exact; GFLOP = Qwen3-4B dense matmuls):", "",
              "| server / graph | requests / step | forward passes / step | positions / step | LM-head rows / step | "
              "GFLOP / step | prefix-cache hit share |", "|---|---|---|---|---|---|---|"]
        for tag, r in P["rows"].items():
            L.append(f"| {tag} | {_f(r['requests'] / r['steps'])} | {_f(r['forwards_per_step'])} | "
                     f"{_f(r['positions_per_step'], 0)} | {_f(r['head_rows_per_step'], 0)} | {_f(r['gflop_per_step'], 0)} | "
                     f"{_f(100 * r['cache_hit_share'], 1)} % |")
    prof = [r for r in rep["rows"] if r.get("phase_ms_per_step")]
    if prof:
        names = sorted({k for r in prof for k in r["phase_ms_per_step"]})
        L += ["", "Server phase profile (ms per step, device-synchronised timers; profiled servers only):", "",
              "| server | graph | c | " + " | ".join(names) + " | sum | S1 ms / step (client) |",
              "|---|---|---|" + "---|" * (len(names) + 2)]
        for r in prof:
            ph = r["phase_ms_per_step"]
            L.append(f"| {r['label']} | {r['path']} | {r['concurrency']} | " + " | ".join(_f(ph.get(k), 1) for k in names)
                     + f" | {_f(sum(ph.values()), 1)} | {_f(r['s1_ms_per_step'], 1)} |")
    if rep.get("cross"):
        L += ["", "Decision equivalence between replays (every shared question; the frozen cascade's accept / defer and "
              "argmax):", "", "| replay | vs | questions | decision flips | max abs Δp |", "|---|---|---|---|---|"]
        for c in rep["cross"]:
            L.append(f"| {'/'.join(map(str, c['run']))} | {'/'.join(map(str, c['ref']))} | {c['questions']} | "
                     f"{c['decision_flips']} | {c['max_abs_dp']:.2g} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ tariff (as S2-4, A7.13 (iii))
# on-demand $/h, us-east-1 (AWS price list): the host that served the run sets the tariff (Deviations 2026-10-05)
HOST_USD_PER_HOUR = {"g6e.2xlarge": 2.24208,    # S2-4's tariff host (results/stage2/tariff.json), 1x L40S 48 GB
                     "g5.4xlarge": 1.624}       # the harness_final host (no g6e capacity on 2026-10-05), 1x A10G 24 GB
HARDWARE = {"g6e.2xlarge": "1x NVIDIA L40S 48 GB", "g5.4xlarge": "1x NVIDIA A10G 24 GB"}


def tariff(run: Path = ROOT / "runs/harness_final_tariff", window: Path = OUT / "gpu/tariff_window.json",
           out: Path = OUT / "tariff.json", host_usd_h: float | None = None) -> dict[str, Any]:
    """q = host $/h x serving hours / completed development episodes (scripts/stage2_tariff.py's rule), on the S2-4
    tariff's development tasks at the same concurrency; the sensitivity rows of S2-4 beside it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import stage2_tariff as T

    w = json.loads(window.read_text())
    it = str(w.get("instance_type") or "")
    if host_usd_h is None:
        if it not in HOST_USD_PER_HOUR:
            raise SystemExit(f"no on-demand price recorded for instance type {it!r}")
        host_usd_h = HOST_USD_PER_HOUR[it]
    rec = T.tariff(run, "harness_final", host_usd_h, w["window_start"], w["window_end"])
    s24 = json.loads((ROOT / "results/stage2/tariff.json").read_text())
    rec.update({"label": LABEL, "served_model": "jevagentrl-4b", "server": "serve_jev --fast",
                "instance_type": w.get("instance_type"), "concurrency": w.get("concurrency"),
                "hardware": HARDWARE.get(it, it), "precision": "bf16",
                "host_price_note": f"{it} on-demand price, us-east-1 (AWS price list, 2026-10-05); S2-4's tariff host was "
                                   "a g6e.2xlarge (1x L40S) at $2.24208/h",
                "s2_4": {"q_usd_per_episode": s24["q_usd_per_episode"], "serving_hours": s24["serving_hours"],
                         "episodes": s24["episodes"], "window": s24["window"]},
                "sensitivity": {"zero": 0.0, "s2_4_tariff": s24["q_usd_per_episode"]}})
    out.write_text(json.dumps(rec, indent=1))
    return rec


# ------------------------------------------------------------------ live analysis
ARMS = ("llm_only", "harness_final")


def _boot(n: int, b: int, seed: int = 0):  # noqa: ANN202
    import numpy as np

    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def _ci(vals) -> list[float]:  # noqa: ANN001
    import numpy as np

    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _episode_live(r: Mapping[str, Any]) -> dict[str, Any]:
    calls = _calls(r)
    s1 = [c for c in calls if c.get("component") == "s1"]
    s2 = [c for c in calls if c.get("component") == "s2"]
    srv = [(c.get("meta") or {}).get("server") or {} for c in s1]
    steps = r.get("steps") or []
    ts = [d for s in steps for d in s.get("decisions") or [] if d.get("dp") == "tool_select" and not d.get("shadow")]
    tp = [d for s in steps for d in s.get("decisions") or [] if d.get("dp") == "tool_params" and not d.get("shadow")
          and d.get("mode") != "generate"]
    return {"success": 1.0 if (r.get("result") or {}).get("success") else 0.0,
            "cost": float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0.0),
            "wall_s": float(r.get("wall_ms") or 0.0) / 1000,
            "model_s": float(((r.get("usage") or {}).get("total") or {}).get("latency_ms") or 0.0) / 1000,
            "s1_s": sum(float(c.get("latency_ms") or 0) for c in s1) / 1000,
            "s2_s": sum(float(c.get("latency_ms") or 0) for c in s2) / 1000,
            "s1_requests": len(s1), "s1_questions": sum(int((c.get("meta") or {}).get("n_questions") or 1) for c in s1),
            "s2_calls": len(s2), "steps": len(steps),
            "lock_wait_s": sum(float(x.get("lock_wait_ms") or 0) for x in srv) / 1000,
            "compute_s": sum(float(x.get("compute_ms") or 0) for x in srv) / 1000,
            "cache_hit_tokens": sum(float(x.get("cache_hit_tokens") or 0) for x in srv),
            "stateless_tokens": sum(float(x.get("stateless_tokens") or 0) for x in srv),
            "tool_decisions": len(ts), "tool_by_s1": sum(1 for d in ts if d.get("mode") in ("cascade", "replace")
                                                         and not d.get("deferred")),
            "arg_decisions": len(tp), "arg_by_s1": sum(1 for d in tp if not d.get("deferred")),
            "error": r.get("error")}


def analyze(run: Path = ROOT / "runs/harness_final_live", out: Path = OUT, tex: Path = TEX_OUT,
            tariff_path: Path | None = None, n_expected: int = 200, n_boot: int = 10000,
            tariff: Path | None = None, name: str = "summary") -> dict[str, Any]:
    """The live confirmation (descriptive): paired success, cost (GPU tariff included as S2-4), wall-clock and
    model-call time per episode; the final harness's time decomposition; intervals only in ``summary.json`` /
    the appendix keys."""
    import numpy as np

    tp = tariff or tariff_path or (OUT / "tariff.json")
    trec = json.loads(Path(tp).read_text()) if Path(tp).exists() else {}
    q = float(trec["q_usd_per_episode"]) if "q_usd_per_episode" in trec else None
    recs = {a: load_latest(run / f"{a}.jsonl") for a in ARMS}
    ok = {a: {t: r for t, r in recs[a].items() if not r.get("error")} for a in ARMS}
    keys = sorted(set(ok["llm_only"]) & set(ok["harness_final"]))
    ep = {a: [_episode_live(ok[a][t]) for t in keys] for a in ARMS}
    n = len(keys)
    if n == 0:
        raise SystemExit(f"no complete pairs in {run}")
    v = {a: {k: np.array([e[k] for e in ep[a]], dtype=float) for k in ("success", "cost", "wall_s", "model_s", "s1_s",
                                                                       "s2_s", "s1_requests", "s1_questions",
                                                                       "s2_calls", "steps")} for a in ARMS}
    qq = q or 0.0
    cost_hf = v["harness_final"]["cost"] + qq
    idx = _boot(n, n_boot)
    lo, hf = v["llm_only"], v["harness_final"]
    d_succ = (hf["success"] - lo["success"]) * 100
    b_succ = d_succ[idx].mean(1)
    sav = 1 - cost_hf[idx].mean(1) / lo["cost"][idx].mean(1)

    def timing(k: str) -> dict[str, Any]:
        d = hf[k] - lo[k]
        bd = d[idx].mean(1)
        bm = np.median(hf[k][idx], 1) - np.median(lo[k][idx], 1)
        return {"llm_only": float(lo[k].mean()), "harness_final": float(hf[k].mean()),
                "llm_only_median": float(np.median(lo[k])), "harness_final_median": float(np.median(hf[k])),
                "delta": float(d.mean()), "delta_ci": _ci(bd),
                "delta_median": float(np.median(hf[k]) - np.median(lo[k])), "delta_median_ci": _ci(bm),
                "saving_frac": float(1 - hf[k].mean() / lo[k].mean()) if lo[k].mean() else None,
                "faster_share": float((d < 0).mean())}

    tot = {k: sum(e[k] for e in ep["harness_final"]) for k in ("lock_wait_s", "compute_s", "cache_hit_tokens",
                                                                "stateless_tokens", "tool_decisions", "tool_by_s1",
                                                                "arg_decisions", "arg_by_s1")}
    sched = {}
    try:
        sched = json.loads((run / "schedule.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    res = {"label": LABEL, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "git": _git_head(), "run": _rel(run), "concurrency": sched.get("concurrency"),
           "inputs": {a: {"path": _rel(run / f"{a}.jsonl"), "sha256": _sha256(run / f"{a}.jsonl")[:16]} for a in ARMS},
           "n_expected": n_expected, "errored_latest": {a: len(recs[a]) - len(ok[a]) for a in ARMS},
           "tariff": {"path": _rel(Path(tp)), "q_usd_per_episode": q, "instance_type": trec.get("instance_type"),
                      "hardware": trec.get("hardware"), "host_usd_per_hour": trec.get("host_usd_per_hour"),
                      "sensitivity": trec.get("sensitivity")},
           "paired": {
               "n_pairs": n,
               "success": {a: float(v[a]["success"].mean()) for a in ARMS},
               "success_delta_pt": float(d_succ.mean()), "success_delta_ci": _ci(b_succ),
               "discordant": {"llm_only_only": int(((lo["success"] == 1) & (hf["success"] == 0)).sum()),
                              "harness_final_only": int(((lo["success"] == 0) & (hf["success"] == 1)).sum())},
               "cost": {"llm_only": float(lo["cost"].mean()), "harness_final_api": float(hf["cost"].mean()),
                        "tariff_q": q, "harness_final": float(cost_hf.mean()),
                        "saving_frac": float(1 - cost_hf.mean() / lo["cost"].mean()), "saving_ci": _ci(sav),
                        "saving_abs": float(lo["cost"].mean() - cost_hf.mean()),
                        "saving_frac_no_tariff": float(1 - hf["cost"].mean() / lo["cost"].mean())},
               "wall_s": timing("wall_s"), "model_s": timing("model_s"),
               "calls": {a: {k: float(v[a][k].mean()) for k in ("s1_requests", "s1_questions", "s2_calls", "steps",
                                                                "s1_s", "s2_s")} for a in ARMS}},
           "harness_final": {"queue_share": (tot["lock_wait_s"] / (tot["lock_wait_s"] + tot["compute_s"])
                                             if tot["lock_wait_s"] + tot["compute_s"] else None),
                             "lock_wait_s_per_episode": tot["lock_wait_s"] / n, "compute_s_per_episode": tot["compute_s"] / n,
                             "cache_hit_share": (tot["cache_hit_tokens"] / tot["stateless_tokens"]
                                                 if tot["stateless_tokens"] else None),
                             "tool_coverage": tot["tool_by_s1"] / tot["tool_decisions"] if tot["tool_decisions"] else None,
                             "arg_coverage": tot["arg_by_s1"] / tot["arg_decisions"] if tot["arg_decisions"] else None,
                             "rest_s": float((hf["wall_s"] - hf["s1_s"] - hf["s2_s"]).mean())}}
    if sched.get("invocations"):
        inv = sched["invocations"][0]
        if inv.get("started") and inv.get("finished"):
            res["span_s_pass1"] = _ts(inv["finished"]) - _ts(inv["started"])
    out.mkdir(parents=True, exist_ok=True)
    res["row"] = name
    (out / f"{name}.json").write_text(json.dumps(_jsonable(res), indent=1))
    (out / f"{name}.md").write_text(summary_markdown(res))
    rows = rows_markdown(out)
    if rows:
        (out / "summary_rows.md").write_text(rows)
    write_tex(out, tex)
    return res


ROWS = (("summary", "hf", "shared GPU (about 4 harness episodes per GPU)"),
        ("summary_dedicated", "hf_ded", "dedicated GPU (at most 1 harness episode per GPU)"))


def rows_markdown(out: Path = OUT) -> str | None:
    """The two provisioning rows side by side (registered live row first)."""
    have = [(n, p, lab, json.loads((out / f"{n}.json").read_text())) for n, p, lab in ROWS if (out / f"{n}.json").exists()]
    if not have:
        return None
    L = ["# The final JevAgentHarness: both provisioning rows (exploratory, descriptive; no test)", "",
         "| row | workers | pairs | success llm_only / harness | Δ success pt | $ / ep. llm_only / harness (q) | saving | "
         "wall s / ep. llm_only / harness | Δ wall s | model-call s / ep. llm_only / harness | S2 calls / ep. |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, _, lab, r in have:
        p = r["paired"]
        c, w, m = p["cost"], p["wall_s"], p["model_s"]
        L.append(f"| {lab} | {r.get('concurrency')} | {p['n_pairs']} | {100 * p['success']['llm_only']:.1f} / "
                 f"{100 * p['success']['harness_final']:.1f} | {p['success_delta_pt']:+.1f} | {c['llm_only']:.4f} / "
                 f"{c['harness_final']:.4f} ({_f(c['tariff_q'], 5)}) | {100 * c['saving_frac']:.1f} % | "
                 f"{w['llm_only']:.2f} / {w['harness_final']:.2f} | {w['delta']:+.2f} | {m['llm_only']:.2f} / "
                 f"{m['harness_final']:.2f} | {p['calls']['llm_only']['s2_calls']:.2f} / "
                 f"{p['calls']['harness_final']['s2_calls']:.2f} |")
    L += ["", "Intervals (paired task bootstrap, 95 %) are in each row's summary file. Host: one NVIDIA A10G (g5.4xlarge); "
          "q = $1.624 / h × the row's own tariff window / episodes."]
    return "\n".join(L) + "\n"


def summary_markdown(res: Mapping[str, Any]) -> str:
    p, h = res["paired"], res["harness_final"]
    c, w, m = p["cost"], p["wall_s"], p["model_s"]
    lab = {n: l for n, _, l in ROWS}.get(res.get("row", "summary"), "")
    L = [f"# The final JevAgentHarness, live: {lab} (exploratory, descriptive)", "",
         f"Label: {res['label']}. Generated {res['generated']} at git {res['git'][:12]} by `python scripts/harness_final.py "
         f"analyze`; inputs: " + ", ".join(f"`{x['path']}` ({x['sha256']})" for x in res["inputs"].values()) +
         f"; tariff `{res['tariff']['path']}`.", "",
         f"Arms `llm_only` and `harness_final` (the frozen JevAgentRL-4B, tool + argument cascade at its frozen "
         f"thresholds, reflection off, argument questions batched per step, `serve_jev --fast`), one invocation, "
         f"task-major interleaved, concurrency {res['concurrency']}; {p['n_pairs']} complete pairs of {res['n_expected']} "
         f"expected (latest record errored: {res['errored_latest']}). Host: {res['tariff'].get('instance_type') or '–'} "
         f"({res['tariff'].get('hardware') or '–'}); System-1 latencies are figures of that GPU (S2-4's JevAgentRL rows "
         "ran on one L40S).", "",
         "| quantity | llm_only | harness_final | difference (harness_final − llm_only) |", "|---|---|---|---|",
         f"| success | {100 * p['success']['llm_only']:.1f} % | {100 * p['success']['harness_final']:.1f} % | "
         f"{p['success_delta_pt']:+.1f} pt [{p['success_delta_ci'][0]:+.1f}, {p['success_delta_ci'][1]:+.1f}] |",
         f"| cost $ / episode (GPU tariff incl.) | {c['llm_only']:.4f} | {c['harness_final']:.4f} "
         f"(API {c['harness_final_api']:.4f} + q {_f(c['tariff_q'], 5)}) | saving {100 * c['saving_frac']:.1f} % "
         f"[{100 * c['saving_ci'][0]:.1f}, {100 * c['saving_ci'][1]:.1f}] |",
         f"| wall-clock s / episode, mean (median) | {w['llm_only']:.2f} ({w['llm_only_median']:.2f}) | "
         f"{w['harness_final']:.2f} ({w['harness_final_median']:.2f}) | {w['delta']:+.2f} s [{w['delta_ci'][0]:+.2f}, "
         f"{w['delta_ci'][1]:+.2f}]; median {w['delta_median']:+.2f} [{w['delta_median_ci'][0]:+.2f}, "
         f"{w['delta_median_ci'][1]:+.2f}]; {_speed(w['saving_frac'])} |",
         f"| model-call time s / episode, mean (median) | {m['llm_only']:.2f} ({m['llm_only_median']:.2f}) | "
         f"{m['harness_final']:.2f} ({m['harness_final_median']:.2f}) | {m['delta']:+.2f} s [{m['delta_ci'][0]:+.2f}, "
         f"{m['delta_ci'][1]:+.2f}] |",
         f"| System-2 calls / episode | {p['calls']['llm_only']['s2_calls']:.2f} | "
         f"{p['calls']['harness_final']['s2_calls']:.2f} | |",
         f"| System-1 requests (questions) / episode | – | {p['calls']['harness_final']['s1_requests']:.2f} "
         f"({p['calls']['harness_final']['s1_questions']:.2f}) | |", "",
         f"Discordant pairs: llm_only only {p['discordant']['llm_only_only']}, harness_final only "
         f"{p['discordant']['harness_final_only']}. Intervals: paired task bootstrap (95 %), descriptive (no test).", "",
         "Final harness decomposition (s / episode): System-1 "
         f"{p['calls']['harness_final']['s1_s']:.2f} (server compute {h['compute_s_per_episode']:.2f}, waiting for the "
         f"engine {h['lock_wait_s_per_episode']:.2f}: {100 * (h['queue_share'] or 0):.1f} % of server time), System-2 "
         f"{p['calls']['harness_final']['s2_s']:.2f}, rest {h['rest_s']:.2f}; prefix-cache hit share "
         f"{100 * (h['cache_hit_share'] or 0):.1f} % of the prompt tokens; tool decisions by System-1 "
         f"{100 * (h['tool_coverage'] or 0):.1f} %, arguments {100 * (h['arg_coverage'] or 0):.1f} %."]
    return "\n".join(L) + "\n"


def _tex_num(x: Any, nd: int = 1) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def tex_keys(out: Path = OUT) -> dict[str, str]:
    """``\\defres{hf/...}`` keys from whatever exists of latency_diagnosis.json (``hf/diag/...``), replay_bench.json
    (``hf/bench/...``), tariff.json and summary.json (the live run, ``hf/...``)."""
    kv: dict[str, str] = {}
    f = out / "latency_diagnosis.json"
    if f.exists():
        d = json.loads(f.read_text())["arms"]
        for arm, key in (("llm_only", "llm_only"), ("jevagent_best_rl", "rl"), ("jevagent_best", "jev")):
            a = d.get(arm)
            if not a:
                continue
            kv[f"hf/diag/{key}/wall_s"] = f"{a['wall_s']:.2f}"
            kv[f"hf/diag/{key}/s1_s"] = f"{a['s1_s']:.2f}"
            kv[f"hf/diag/{key}/s2_s"] = f"{a['s2_s']:.2f}"
            kv[f"hf/diag/{key}/s2_calls"] = f"{a['s2_calls']:.2f}"
            kv[f"hf/diag/{key}/s1_requests"] = f"{a['s1_requests']:.1f}"
            kv[f"hf/diag/{key}/s1_requests_per_step"] = f"{a['s1_requests_per_step']:.2f}"
        q = (d.get("jevagent_best_rl") or {}).get("queue")
        if q:
            kv["hf/diag/rl/busy"] = _tex_num(100 * q["busy_fraction"])
            kv["hf/diag/rl/queue_share"] = _tex_num(100 * q["queue_share"])
            kv["hf/diag/rl/service_ms"] = _tex_num(q["service_ms_mean"], 0)
            kv["hf/diag/rl/queue_ms"] = _tex_num(q["queue_ms_mean"], 0)
            kv["hf/diag/rl/ts_service_ms"] = _tex_num(q["by_dp"]["tool_select"]["service_ms_mean"], 0)
        r = (d.get("jevagent_best_rl") or {}).get("reflection")
        if r:
            kv["hf/diag/rl/reflection_questions"] = str(r["questions"])
            kv["hf/diag/rl/reflection_fired"] = str(r["fired"])
            kv["hf/diag/rl/reflection_pmax"] = f"{r['p_yes_max']:.2f}"
    f = out / "replay_bench.json"
    if f.exists():
        rep = json.loads(f.read_text())
        for r in rep["rows"]:
            k = f"hf/bench/{r['label']}/{r['path']}/c{r['concurrency']}"
            kv[f"{k}/s1_ms_step"] = _tex_num(r["s1_ms_per_step"], 0)
            kv[f"{k}/s1_s_ep"] = f"{r['s1_s_per_episode']:.2f}"
            kv[f"{k}/wall_s_ep"] = f"{r['wall_s_per_episode']:.2f}"
            kv[f"{k}/eps_per_gpu_h"] = _tex_num(r["throughput_eps_per_h"], 0)
            kv[f"{k}/requests_ep"] = f"{r['requests_per_episode']:.1f}"
            # decisions differing from the harness's own logged decisions (the report's column); `decision_flips`
            # compares against the stored, smoothed s1_dist and is kept only as `flips_vs_dist`
            vl = r.get("vs_logged_decisions") or {}
            kv[f"{k}/flips"] = str(vl.get("flips", r["decision_flips"]))
            kv[f"{k}/questions"] = str(vl.get("questions", r["questions"]))
            kv[f"{k}/flips_vs_dist"] = str(r["decision_flips"])
        for c in rep.get("cross") or []:
            k = "hf/bench/cross/" + "/".join(map(str, c["run"]))
            kv[f"{k}/flips"] = str(c["decision_flips"])
            kv[f"{k}/questions"] = str(c["questions"])
    f = out / "positions.json"
    if f.exists():
        for tag, r in json.loads(f.read_text())["rows"].items():
            kv[f"hf/positions/{tag}/per_step"] = _tex_num(r["positions_per_step"], 0)
            kv[f"hf/positions/{tag}/gflop_step"] = _tex_num(r["gflop_per_step"], 0)
            kv[f"hf/positions/{tag}/forwards_step"] = _tex_num(r["forwards_per_step"], 1)
            kv[f"hf/positions/{tag}/requests_step"] = _tex_num(r["requests"] / r["steps"], 2)
    for tn, pre in (("tariff", "hf"), ("tariff_dedicated", "hf_ded")):
        f = out / f"{tn}.json"
        if f.exists():
            t = json.loads(f.read_text())
            kv[f"{pre}/tariff/q"] = f"{t['q_usd_per_episode']:.5f}"
            kv[f"{pre}/tariff/q_s24"] = f"{t['s2_4']['q_usd_per_episode']:.5f}"
            kv[f"{pre}/tariff/host_usd_h"] = f"{t['host_usd_per_hour']:.3f}"
    for sn, pre, _ in ROWS:
        f = out / f"{sn}.json"
        if f.exists():
            kv.update(_summary_keys(json.loads(f.read_text()), pre))
    return kv


def _summary_keys(res: Mapping[str, Any], pre: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    if True:  # noqa: SIM108
        p, h = res["paired"], res["harness_final"]
        c, w, m = p["cost"], p["wall_s"], p["model_s"]
        kv.update({
            f"{pre}/n": str(p["n_pairs"]), f"{pre}/conc": str(res["concurrency"]),
            f"{pre}/success/llm_only": _tex_num(100 * p["success"]["llm_only"]),
            f"{pre}/success/harness_final": _tex_num(100 * p["success"]["harness_final"]),
            f"{pre}/success/delta": f"{p['success_delta_pt']:+.1f}",
            f"{pre}/success/delta_ci/lo": f"{p['success_delta_ci'][0]:+.1f}",
            f"{pre}/success/delta_ci/hi": f"{p['success_delta_ci'][1]:+.1f}",
            f"{pre}/cost/llm_only": f"{c['llm_only']:.4f}", f"{pre}/cost/harness_final": f"{c['harness_final']:.4f}",
            f"{pre}/cost/q": "--" if c["tariff_q"] is None else f"{c['tariff_q']:.5f}",
            f"{pre}/cost/saving": _tex_num(100 * c["saving_frac"]),
            f"{pre}/cost/saving_ci/lo": _tex_num(100 * c["saving_ci"][0]),
            f"{pre}/cost/saving_ci/hi": _tex_num(100 * c["saving_ci"][1]),
            f"{pre}/cost/saving_no_tariff": _tex_num(100 * c["saving_frac_no_tariff"]),
            f"{pre}/wall_s/llm_only": f"{w['llm_only']:.2f}", f"{pre}/wall_s/harness_final": f"{w['harness_final']:.2f}",
            f"{pre}/wall_s/delta": f"{w['delta']:+.2f}", f"{pre}/wall_s/delta_ci/lo": f"{w['delta_ci'][0]:+.2f}",
            f"{pre}/wall_s/delta_ci/hi": f"{w['delta_ci'][1]:+.2f}", f"{pre}/wall_s/saving": _tex_num(100 * w["saving_frac"]),
            f"{pre}/wall_s/median/llm_only": f"{w['llm_only_median']:.2f}",
            f"{pre}/wall_s/median/harness_final": f"{w['harness_final_median']:.2f}",
            f"{pre}/model_s/llm_only": f"{m['llm_only']:.2f}", f"{pre}/model_s/harness_final": f"{m['harness_final']:.2f}",
            f"{pre}/model_s/delta": f"{m['delta']:+.2f}", f"{pre}/model_s/delta_ci/lo": f"{m['delta_ci'][0]:+.2f}",
            f"{pre}/model_s/delta_ci/hi": f"{m['delta_ci'][1]:+.2f}",
            f"{pre}/s2_calls/llm_only": f"{p['calls']['llm_only']['s2_calls']:.2f}",
            f"{pre}/s2_calls/harness_final": f"{p['calls']['harness_final']['s2_calls']:.2f}",
            f"{pre}/s1_requests": f"{p['calls']['harness_final']['s1_requests']:.2f}",
            f"{pre}/s1_s": f"{p['calls']['harness_final']['s1_s']:.2f}", f"{pre}/s2_s": f"{p['calls']['harness_final']['s2_s']:.2f}",
            f"{pre}/queue_share": _tex_num(100 * (h["queue_share"] or 0)),
            f"{pre}/cache_hit_share": _tex_num(100 * (h["cache_hit_share"] or 0)),
            f"{pre}/tool_coverage": _tex_num(100 * (h["tool_coverage"] or 0)),
            f"{pre}/arg_coverage": _tex_num(100 * (h["arg_coverage"] or 0))})
    return kv


def write_tex(out: Path = OUT, tex: Path = TEX_OUT) -> Path:
    kv = tex_keys(out)
    lines = ["% AUTO-GENERATED by scripts/harness_final.py -- do not edit by hand.",
             f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git {_git_head()[:12]}; "
             f"{LABEL}", r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(kv.items())]
    lines.append(r"\makeatother")
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text("\n".join(lines) + "\n")
    return tex


if __name__ == "__main__":
    raise SystemExit(main())
