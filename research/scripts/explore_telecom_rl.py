#!/usr/bin/env python3
"""E1 (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry of
2026-10-05): live tau^2 telecom backend substitution with the frozen JevAgentRL-4B.

Arms (paired per task, ONE attempt each, the A7.15 (e) telecom settings): `llm_only` (configs/tau2/llm_only.yaml) and
`jevagent_best_rl`, the registered tau^2 rendering of the selected profile (A7.1 (i): JevGym-carried thresholds, no
refit on L3 data; the reflection trigger is off on tau^2) with the JevAgentRL thresholds of the profile's
`backends.jevagent_rl` block (A7.1 (f)) and System-1 = the frozen JevAgentRL-4B served by training/serve_jev.py on the
Stage-2 GPU host (results/gpu_round5/FREEZE_RECORD.json). No test, no NI decision; nothing here revises S2-L3a.

    python scripts/explore_telecom_rl.py select        # -> configs/tau2/explore/telecom_rl_task_ids.json (never-run ids)
    python scripts/explore_telecom_rl.py render        # -> configs/tau2/explore/jevagent_best_rl.yaml
    python scripts/explore_telecom_rl.py spend RUN...  # API spend (agent ledger + user simulator) of tau2 save dirs
    python scripts/explore_telecom_rl.py gonogo        # day-1 go/no-go on the 5 validation tasks (RL arm)
    python scripts/explore_telecom_rl.py gonogo --round 2   # the re-run after the adapter fix: validation tasks 6-10
    python scripts/explore_telecom_rl.py analyze       # -> results/explore/telecom_rl/summary.{json,md} + explore.tex

Task ids: every id of the pinned telecom `full` pool (configs/tau2/stage2/telecom_inventory.json) that appears in no run
of this project (every *.json / *.jsonl under the scanned run roots), outside the frozen S2-L3a sample and the 32
reserved validation tasks, ordered by sha256(SALT + "\\n" + id); the first N = 150. The go/no-go uses the first 5 of
the 32 reserved validation tasks (A7.15 (b): reserved for adapter validation, never evaluated).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.scorecard import call_coverage  # noqa: E402
from jevharness.harness import profile as HP  # noqa: E402

ROOT = C.ROOT
INVENTORY = ROOT / "configs/tau2/stage2/telecom_inventory.json"
FROZEN = ROOT / "configs/tau2/stage2/telecom_task_ids.json"
IDS_OUT = ROOT / "configs/tau2/explore/telecom_rl_task_ids.json"
RENDER_OUT = ROOT / "configs/tau2/explore/jevagent_best_rl.yaml"
PROFILE = ROOT / "configs/harness/jevagent_best.yaml"
OUT_DIR = ROOT / "results/explore/telecom_rl"
RUN = ROOT / "runs/explore_telecom_rl"
VAL_RUN = ROOT / "runs/explore_telecom_rl_validation"
# go/no-go rounds: (task-id key in IDS_OUT, run directory). Round 2 (Deviations 2026-10-05, the E1 go/no-go fix):
# validation tasks 6-10 after the adapter fix (respond_fallback: defer); round 1 found the defect
GONOGO_ROUNDS = {1: ("validation_gonogo", VAL_RUN), 2: ("validation_gonogo_2", ROOT / "runs/explore_telecom_rl_validation2")}
WINDOWS = ROOT / "results/explore/gpu/e1_rl_windows.json"
SALT = "jev-explore-E1-telecom-rl-2026-10-05"
N_TASKS, N_VAL = 150, 5
ARM, REF = "jevagent_best_rl", "llm_only"
FLASH = "gemini-3.5-flash"
BACKEND = "jevagent_rl"
SERVED = "jevagentrl-4b"
RL_URL = "http://127.0.0.1:8081"
CAP_USD = 75.0
USD_PER_HOUR = {"g6e.2xlarge": 2.24208, "g6e.16xlarge": 7.57719}   # results/stage2/v2_eval_plan.json host prices
REFERENCE_TYPE = "g6e.2xlarge"
JEV_API_PER_M = 0.042                                               # the S2-4 tariff's jev_api_price sensitivity row
ID_RE = re.compile(rb"\[(?:mms_issue|mobile_data_issue|service_issue)\][^\"\\\[\]]*\[PERSONA:[A-Za-z]+\]")


def _sha_list(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _order_key(tid: str) -> str:
    return hashlib.sha256(f"{SALT}\n{tid}".encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ select
def touched_ids(roots: Iterable[Path], pool: set[str]) -> tuple[set[str], list[str]]:
    """Every pool id that appears in any *.json / *.jsonl file under ``roots`` (byte scan; a run log names its task id)."""
    seen: set[str] = set()
    files: list[str] = []
    for r in roots:
        if not Path(r).exists():
            continue
        for p in sorted(Path(r).rglob("*")):
            if p.suffix not in (".json", ".jsonl") or not p.is_file():
                continue
            hits = {m.decode() for m in ID_RE.findall(p.read_bytes())} & pool
            if hits:
                seen |= hits
                files.append(C.rel(p))
    return seen, files


def select(inv: Mapping[str, Any], frozen_ids: Sequence[str], touched: set[str], n: int = N_TASKS,
           n_val: int = N_VAL) -> dict[str, Any]:
    pool = [t["id"] for t in inv["tasks"]]
    val = list(inv["validation"]["task_ids"])
    excluded = set(frozen_ids) | set(val) | set(touched)
    eligible = [t for t in pool if t not in excluded]
    order = sorted(eligible, key=_order_key)
    ids = order[:n]
    issue = {t["id"]: t["issue_type"] for t in inv["tasks"]}
    persona = {t["id"]: t["persona"] for t in inv["tasks"]}
    return {"task_ids": ids, "validation_gonogo": val[:n_val], "validation_gonogo_2": val[n_val:2 * n_val],
            "n_pool": len(pool), "n_excluded": len(excluded),
            "n_eligible": len(eligible), "issue_types": dict(Counter(issue[t] for t in ids)),
            "personas": dict(Counter(persona[t] for t in ids)), "sha256": _sha_list(ids),
            "eligible_sha256": _sha_list(eligible)}


def cmd_select(a: argparse.Namespace) -> int:
    inv = json.loads(INVENTORY.read_text())
    fz = json.loads(FROZEN.read_text())
    frozen_ids = fz["task_ids"] if isinstance(fz, dict) else fz
    pool = {t["id"] for t in inv["tasks"]}
    roots = [Path(r) for r in (a.scan or [str(ROOT / "runs")])]
    touched, files = touched_ids(roots, pool)
    val = set(inv["validation"]["task_ids"])
    missing = (set(frozen_ids) | val) - touched
    if missing and not a.allow_unscanned:
        print(f"refusing: {len(missing)} frozen / validation ids not found in the scanned runs (scan incomplete?)",
              file=sys.stderr)
        return 2
    rec = select(inv, frozen_ids, touched, a.n, N_VAL)
    out = {"registered": "exploratory E1, docs/PREREG_A7_STAGE2.md Deviations (2026-10-05); outside both stages",
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "rule": ("pool ids of configs/tau2/stage2/telecom_inventory.json minus every id appearing in any scanned run, "
                    "the frozen S2-L3a sample and the 32 reserved validation ids; ordered by sha256(salt + '\\n' + id) "
                    "(hex, ascending); the first n. validation_gonogo = the first 5 of the inventory's validation ids "
                    "(reserved for adapter validation, never evaluated)"),
           "salt": SALT, "n": a.n,
           "inventory_sha256": C.sha256_file(INVENTORY), "frozen_sample_sha256": fz.get("sha256"),
           "scanned_roots": [C.rel(r) for r in roots], "files_naming_a_pool_id": len(files),
           "n_touched_ids": len(touched), **{k: v for k, v in rec.items() if k != "task_ids"},
           "task_ids": rec["task_ids"]}
    IDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    IDS_OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[E1] {len(rec['task_ids'])} never-run ids of {rec['n_eligible']} eligible ({len(touched)} touched ids in "
          f"{len(files)} files); sha256 {rec['sha256']} -> {C.rel(IDS_OUT)}; file sha256 {C.sha256_file(IDS_OUT)}")
    return 0


# ------------------------------------------------------------------------------------------------ render
def render(profile: Mapping[str, Any], url: str = RL_URL, cache: str = "cache/s1_explore_telecom_rl.sqlite") \
        -> tuple[dict[str, Any] | None, list[str]]:
    """A7.1 (i)'s fixed tau^2 translation of the profile's Flash section with the `backends.jevagent_rl` thresholds
    (A7.1 (f)), System-1 = the served frozen checkpoint."""
    dps, _, skip = HP.decision_points(profile, FLASH, backend=BACKEND)
    if skip:
        return None, [skip]
    s1 = {"type": "jev", "name": SERVED, "base_url": url, "model": SERVED, "api_key": "", "timeout": 60.0}
    out, notes = HP.tau2_rendering(dps, name=ARM, s1=s1, cache=cache)
    if out is not None:  # Deviations 2026-10-05 (E1 go/no-go fix): a text-less reply to respond_to_user is deferred
        out["respond_fallback"] = "defer"
    return out, notes


def cmd_render(a: argparse.Namespace) -> int:
    prof = HP.load_profile(PROFILE)
    t2, notes = render(prof)
    if t2 is None:
        print(f"no tau^2 rendering: {notes}", file=sys.stderr)
        return 2
    RENDER_OUT.parent.mkdir(parents=True, exist_ok=True)
    RENDER_OUT.write_text(
        "# E1 (exploratory, Deviations 2026-10-05): the tau^2 rendering (A7.1 (i)) of configs/harness/jevagent_best.yaml's\n"
        "# Flash section with its backends.jevagent_rl thresholds (A7.1 (f)); System-1 = the frozen JevAgentRL-4B served\n"
        f"# by training/serve_jev.py on the Stage-2 GPU host. WRITTEN BY scripts/explore_telecom_rl.py render from the profile\n"
        f"# (sha256 {C.sha256_file(PROFILE)[:16]}); do not edit by hand.\n"
        + "".join(f"# dropped: {n}\n" for n in notes) + yaml.safe_dump(t2, sort_keys=False))
    print(f"-> {C.rel(RENDER_OUT)}: {yaml.safe_dump(t2, sort_keys=False, default_flow_style=True).strip()}")
    return 0


# ------------------------------------------------------------------------------------------------ spend
def raw_spend(save_dirs: Iterable[Path]) -> dict[str, Any]:
    """API spend of tau2 save dirs: every simulation in results.json and in jev_errored/ (pruned errored attempts):
    the agent ledger's cost (meta agent_cost) + the user simulator's cost."""
    out: dict[str, Any] = {"dirs": {}, "total": 0.0, "sims": 0}
    for d in save_dirs:
        d = Path(d)
        sims: list[Mapping[str, Any]] = []
        if (d / "results.json").exists():
            sims += json.loads((d / "results.json").read_text()).get("simulations") or []
        for p in sorted((d / "jev_errored").glob("*.json")) if (d / "jev_errored").exists() else []:
            try:
                sims.append(json.loads(p.read_text()))
            except ValueError:
                pass
        s = sum(float(x.get("agent_cost") or 0) + float(x.get("user_cost") or 0) for x in sims)
        out["dirs"][C.rel(d)] = {"sims": len(sims), "usd": s}
        out["total"] += s
        out["sims"] += len(sims)
    return out


def cmd_spend(a: argparse.Namespace) -> int:
    r = raw_spend([Path(x) for x in a.dirs])
    print(json.dumps(r, indent=1) if a.json else f"{r['total']:.4f} {r['sims']}")
    return 0


# ------------------------------------------------------------------------------------------------ go / no-go
def _tool_calls(raw_sim: Mapping[str, Any]) -> list[tuple[str, str]]:
    out = []
    for m in raw_sim.get("messages") or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                out.append((tc.get("name"), json.dumps(tc.get("arguments"), sort_keys=True)))
    return out


def gonogo(trajs: Mapping[str, Mapping[str, Any]], raw_sims: Sequence[Mapping[str, Any]], expected: Sequence[str],
           ref_cost_per_sim: float, spent_before: float, n_main: int = N_TASKS, cap: float = CAP_USD) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    exp = [f"telecom/{t}" for t in expected]
    got = [k for k in exp if k in trajs]
    errs = {k: trajs[k].get("error") for k in got if trajs[k].get("error")}
    checks["complete_no_error"] = {"ok": len(got) == len(exp) and not errs, "n_expected": len(exp), "n_got": len(got),
                                   "errors": errs}
    s1_models: Counter = Counter()
    bad_calls, ts, ts_s1, ts_bad, tp, tp_s1, deleg = [], 0, 0, [], 0, 0, 0
    for k in got:
        t = trajs[k]
        for c in t.get("calls") or []:
            if c.get("component") == "s1":
                s1_models[c.get("model")] += 1
                if not c.get("priced") or float(c.get("cost_usd") or 0) != 0.0:
                    bad_calls.append({"task": k, "model": c.get("model"), "priced": c.get("priced"),
                                      "cost": c.get("cost_usd")})
        for s in t.get("steps") or []:
            for d in s.get("decisions") or []:
                if d.get("shadow"):
                    continue
                if d.get("dp") == "tool_select":
                    ts += 1
                    if d.get("s1_value") is not None:
                        ts_s1 += 1
                        dist = d.get("s1_dist") or {}
                        conf = d.get("s1_conf")
                        okd = (d["s1_value"] in (d.get("options") or []) and conf is not None and 0.0 <= conf <= 1.0
                               and (not dist or abs(sum(dist.values()) - 1.0) < 1e-3)
                               and set(dist) <= set(d.get("options") or []))
                        if not okd:
                            ts_bad.append({"task": k, "step": s.get("step"), "value": d.get("s1_value"), "conf": conf})
                    deleg += bool(M.is_delegated(d))
                elif d.get("dp") == "tool_params":
                    tp += 1
                    tp_s1 += d.get("s1_value") is not None
    checks["s1_answers"] = {"ok": ts_s1 > 0 and not ts_bad, "tool_select_decisions": ts, "with_s1_answer": ts_s1,
                            "malformed": ts_bad[:10], "delegated": deleg, "tool_params_decisions": tp,
                            "tool_params_with_s1_answer": tp_s1}
    unpriced = sorted({m for k in got for m in ((trajs[k].get("usage") or {}).get("unpriced_models") or [])})
    s1_tot = {k: ((trajs[k].get("usage") or {}).get("s1") or {}) for k in got}
    s2_cost = sum(float(((trajs[k].get("usage") or {}).get("s2") or {}).get("cost_usd") or 0) for k in got)
    ledger_ok = (not unpriced and not bad_calls and set(s1_models) == {SERVED}
                 and all(float(v.get("cost_usd") or 0) == 0 for v in s1_tot.values())
                 and all(abs(C.cost(trajs[k]) - float(((trajs[k].get("usage") or {}).get("s2") or {}).get("cost_usd") or 0))
                         < 1e-9 for k in got) and s2_cost > 0)
    checks["ledger"] = {"ok": ledger_ok, "s1_models": dict(s1_models), "unpriced_models": unpriced,
                        "nonzero_or_unpriced_s1_calls": bad_calls[:10], "s1_calls": sum(s1_models.values()),
                        "s2_cost_usd": s2_cost}
    loops, terms, tool_err, n_msgs = [], Counter(), 0, []
    known = {o for k in got for s in trajs[k].get("steps") or [] for d in s.get("decisions") or []
             if d.get("dp") == "tool_select" for o in d.get("options") or []}
    unknown = Counter(n for sim in raw_sims for n, _ in _tool_calls(sim) if known and n not in known)
    for sim in raw_sims:
        terms[str(sim.get("termination_reason"))] += 1
        calls = _tool_calls(sim)
        run, best = 1, 1
        for i in range(1, len(calls)):
            run = run + 1 if calls[i] == calls[i - 1] else 1
            best = max(best, run)
        if best > 3:
            loops.append({"task": sim.get("task_id"), "repeats": best})
        tool_err += sum(1 for m in sim.get("messages") or [] if m.get("role") == "tool" and m.get("error"))
        n_msgs.append(len(sim.get("messages") or []))
    bad_terms = {k: v for k, v in terms.items() if k not in ("user_stop", "agent_stop", "transfer")}
    checks["no_crash_or_loop"] = {"ok": not loops and not bad_terms, "termination": dict(terms),
                                  "repeated_identical_calls_gt3": loops, "agent_tool_errors": tool_err,
                                  "executed_calls_to_unknown_tools": sum(unknown.values()),
                                  "messages_per_sim": n_msgs}
    per_sim = raw_spend_sims(raw_sims) / max(1, len(raw_sims))
    proj = spent_before + n_main * (per_sim + ref_cost_per_sim)
    checks["projected_spend"] = {"ok": proj <= cap, "rl_arm_usd_per_sim": per_sim,
                                 "llm_only_usd_per_sim_assumed": ref_cost_per_sim, "spent_before": spent_before,
                                 "projected_total": proj, "cap": cap}
    go = all(v["ok"] for v in checks.values())
    return {"go": go, "checks": checks}


def raw_spend_sims(sims: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(x.get("agent_cost") or 0) + float(x.get("user_cost") or 0) for x in sims)


def _ref_cost_per_sim() -> float:
    """S2-L3a's llm_only API cost per simulation (agent + user), from its raw results (operational field)."""
    p = ROOT / "runs/stage2_tau2/raw/telecom/llm_only/results.json"
    if not p.exists():
        return 0.1645
    sims = json.loads(p.read_text())["simulations"]
    return raw_spend_sims(sims) / len(sims)


def _gonogo_md(res: Mapping[str, Any]) -> str:
    lines = [f"# E1 day-1 go/no-go, round {res['round']} — {'GO' if res['go'] else 'NO-GO'}", "",
             f"Run `{res['run']}` (RL arm only), validation tasks `{res['task_key']}` of "
             f"`configs/tau2/explore/telecom_rl_task_ids.json` ({len(res['task_ids'])} tasks); decided {res['decided']} "
             "by `scripts/explore_telecom_rl.py gonogo` under the criteria of the E1 Deviations entry (d). "
             f"{res['note']}.", "", "| check | ok | details |", "|---|---|---|"]
    for k, v in res["checks"].items():
        det = json.dumps({x: y for x, y in v.items() if x != "ok"}, default=str)
        lines.append(f"| {k} | {'ok' if v['ok'] else '**FAIL**'} | `{det[:400]}` |")
    if res.get("diagnosis"):
        lines += ["", f"Diagnosis (operational fields only): {res['diagnosis']}"]
    return "\n".join(lines + [""])


def gonogo_tex_keys(rounds: Mapping[int, Mapping[str, Any]]) -> dict[str, str]:
    """Keys of the go/no-go rounds (operational fields only; no success number exists in these records)."""
    k: dict[str, str] = {}
    for n, r in sorted(rounds.items()):
        ch = r["checks"]
        loop = ch["no_crash_or_loop"]
        k.update({f"gonogo/r{n}/go": C.fmt(bool(r["go"])), f"gonogo/r{n}/n_tasks": C.fmt(len(r["task_ids"])),
                  f"gonogo/r{n}/ts_decisions": C.fmt(int(ch["s1_answers"]["tool_select_decisions"])),
                  f"gonogo/r{n}/s1_calls": C.fmt(int(ch["ledger"]["s1_calls"])),
                  f"gonogo/r{n}/too_many_errors": C.fmt(int(loop["termination"].get("too_many_errors", 0))),
                  f"gonogo/r{n}/unknown_tool_calls": C.fmt(int(loop.get("executed_calls_to_unknown_tools", 0))),
                  f"gonogo/r{n}/max_repeats": C.fmt(max([int(x["repeats"]) for x in loop["repeated_identical_calls_gt3"]]
                                                        or [0])),
                  f"gonogo/r{n}/tool_errors": C.fmt(int(loop["agent_tool_errors"]))})
    last = rounds[max(rounds)]
    k.update({"gonogo/decision": "go" if last["go"] else "no-go", "gonogo/rounds": C.fmt(len(rounds)),
              "gonogo/n_val_tasks": C.fmt(sum(len(r["task_ids"]) for r in rounds.values())),
              "gonogo/spend": C.fmt(float(last["checks"]["projected_spend"]["spent_before"]), 2)})
    return k


def cmd_gonogo(a: argparse.Namespace) -> int:
    ids = json.loads(IDS_OUT.read_text())
    key, default_run = GONOGO_ROUNDS[a.round]
    if key not in ids:
        print(f"no {key} in {C.rel(IDS_OUT)}", file=sys.stderr)
        return 2
    run = Path(a.run) if a.run else default_run
    trajs, _ = C.load_latest([run / "telecom" / f"{ARM}.jsonl"])
    raw = run / "raw/telecom" / ARM / "results.json"
    sims = json.loads(raw.read_text())["simulations"] if raw.exists() else []
    # every validation round so far counts against the cap (all E1 API spend, validation included)
    spent = raw_spend([r / "raw/telecom" / ARM for k, (_, r) in sorted(GONOGO_ROUNDS.items()) if k <= a.round])["total"]
    res = gonogo(trajs, sims, ids[key], _ref_cost_per_sim(), spent)
    res.update({"round": a.round, "run": C.rel(run), "task_key": key, "task_ids": ids[key],
                "decided": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "note": "validation tasks: adapter checks only; no success number read"})
    if a.diagnosis:
        res["diagnosis"] = a.diagnosis
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in (f"gonogo_round{a.round}", "gonogo"):  # gonogo.{json,md} = the latest round's decision
        (OUT_DIR / f"{stem}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        (OUT_DIR / f"{stem}.md").write_text(_gonogo_md(res))
    rounds = {n: json.loads((OUT_DIR / f"gonogo_round{n}.json").read_text()) for n in GONOGO_ROUNDS
              if n <= a.round and (OUT_DIR / f"gonogo_round{n}.json").exists()}
    if not a.no_tex:
        C.write_tex("telecom_rl", gonogo_tex_keys(rounds), "scripts/explore_telecom_rl.py gonogo", Path(a.tex))
    for k, v in res["checks"].items():
        print(f"  {k:20s} {'ok' if v['ok'] else 'FAIL'}  " + json.dumps({x: y for x, y in v.items() if x != 'ok'},
                                                                         default=str)[:300])
    print(f"round {a.round}: " + ("GO" if res["go"] else "NO-GO"))
    return 0 if res["go"] else 1


# ------------------------------------------------------------------------------------------------ analysis
def _busy_hours(trajs: Iterable[Mapping[str, Any]]) -> float:
    """Union of the System-1 request intervals (not cached), hours."""
    iv = []
    for t in trajs:
        for c in t.get("calls") or []:
            if c.get("component") == "s1" and not c.get("cached") and c.get("ts"):
                end = dt.datetime.fromisoformat(c["ts"]).timestamp()
                iv.append((end - float(c.get("latency_ms") or 0) / 1000, end))
    if not iv:
        return 0.0
    iv.sort()
    busy, (cs, ce) = 0.0, iv[0]
    for s, e in iv[1:]:
        if s > ce:
            busy, cs, ce = busy + ce - cs, s, e
        else:
            ce = max(ce, e)
    return (busy + ce - cs) / 3600


def tariff(trajs: Mapping[str, Mapping[str, Any]], windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The provisioned-GPU tariff convention of S2-4 (A7.13 (iii): q = host $/h x serving window / episodes), measured
    on E1's own RL-arm windows (no separate development workload; descriptive), with the S2-4 sensitivity rows."""
    done = [t for t in trajs.values() if not t.get("error")]
    n = max(1, len(done))
    hours = 0.0
    itypes = set()
    for w in windows:
        a = dt.datetime.fromisoformat(str(w["start"]).replace("Z", "+00:00"))
        b = dt.datetime.fromisoformat(str(w["end"]).replace("Z", "+00:00"))
        hours += (b - a).total_seconds() / 3600
        itypes.add(w.get("instance_type"))
    itype = sorted(x for x in itypes if x)[0] if itypes - {None} else None
    usd = USD_PER_HOUR.get(itype or "", None)
    tok = sum(int(c.get("input_tokens") or 0) for t in trajs.values() for c in t.get("calls") or []
              if c.get("component") == "s1" and not c.get("cached"))
    busy = _busy_hours(trajs.values())
    q = usd * hours / n if usd is not None else None
    sens = {"busy_throughput": (usd * busy / len(trajs)) if usd is not None and trajs else None,
            "jev_api_price": tok * JEV_API_PER_M / 1e6 / max(1, len(trajs)), "zero": 0.0,
            "reference_host": USD_PER_HOUR[REFERENCE_TYPE] * hours / n}
    return {"q_usd_per_task": q, "instance_type": itype, "usd_per_hour": usd, "serving_hours": hours,
            "tasks_completed": len(done), "busy_hours": busy, "s1_input_tokens": tok, "sensitivity": sens,
            "convention": "S2-4 provisioned tariff (results/stage2/tariff.json): host $/h x window / tasks; window = "
                          "the RL arm's own tau2 runs on the GPU host (E1 workload, concurrency as run)"}


def arm_stats(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ts = [t for t in tasks.values() if not t.get("error")]
    n = len(ts)
    if not n:
        return {"n": 0}
    cov = call_coverage(ts)
    recs = [r for r in M.decision_records(ts) if not r.get("shadow")]
    tsr = [r for r in recs if r.get("dp") == "tool_select"]
    dl = [r for r in tsr if M.is_delegated(r)]
    gold = [r for r in dl if r.get("gold") is not None]
    asked = [r for r in tsr if r.get("s1_value") is not None]
    resp = [r for r in dl if r.get("final_value") == "respond_to_user"]
    return {"n": n, "success": float(np.mean([C.success(t) for t in ts])),
            "agent_cost_per_task": float(np.mean([C.cost(t) for t in ts])),
            "user_cost_per_task": float(np.mean([float(((t.get("meta") or {}).get("tau2") or {}).get("user_cost") or 0)
                                                 for t in ts])),
            "s2_calls_per_task": float(np.mean([C.s_calls(t, "s2") for t in ts])),
            "s1_calls_per_task": float(np.mean([C.s_calls(t, "s1") for t in ts])),
            "latency_s_per_task": float(np.mean([C.latency_ms(t) for t in ts])) / 1000,
            "wall_s_per_task": float(np.mean([float(t.get("wall_ms") or 0) for t in ts])) / 1000,
            "tool_decisions": len(tsr), "tool_asked": len(asked), "tool_delegated": len(dl),
            "tool_coverage": cov.get("tool_coverage"), "arg_coverage": cov.get("arg_coverage"),
            "delegated_respond_share": len(resp) / len(dl) if dl else None,
            "delegated_agree_reference": (sum(r["final_value"] == r["gold"] for r in gold) / len(gold)) if gold else None,
            "delegated_with_reference": len(gold),
            "unpriced_models": sorted({m for t in ts for m in ((t.get("usage") or {}).get("unpriced_models") or [])})}


def analyze(arm: Mapping[str, Mapping[str, Any]], ref: Mapping[str, Mapping[str, Any]], expected: Sequence[str],
            windows: Sequence[Mapping[str, Any]], attempts: Mapping[str, Mapping[str, int]] | None = None) -> dict[str, Any]:
    exp = [f"telecom/{t}" for t in expected]
    miss = {nm: [k for k in exp if k not in d] for nm, d in ((ARM, arm), (REF, ref))}
    errd = {nm: [k for k in exp if k in d and d[k].get("error")] for nm, d in ((ARM, arm), (REF, ref))}
    pairs = [k for k in exp if k in arm and k in ref and not arm[k].get("error") and not ref[k].get("error")]
    ya = [float(C.success(arm[k])) for k in pairs]
    yb = [float(C.success(ref[k])) for k in pairs]
    tf = tariff({k: v for k, v in arm.items() if k in set(exp)}, windows)
    q = tf["q_usd_per_task"] or 0.0
    ca_api = [C.cost(arm[k]) for k in pairs]
    cb = [C.cost(ref[k]) for k in pairs]
    ca = [c + q for c in ca_api]
    out: dict[str, Any] = {
        "label": "exploratory E1 (Deviations 2026-10-05): descriptive, no test; revises no registered result",
        "arm": ARM, "reference": REF, "n_expected": len(exp), "n_pairs": len(pairs),
        "missing": {k: len(v) for k, v in miss.items()}, "errored": errd,
        "attempts_max": {k: max(v.values()) if v else 0 for k, v in (attempts or {}).items()},
        "success": {ARM: float(np.mean(ya)) if ya else None, REF: float(np.mean(yb)) if yb else None},
        "delta_success_pt": {**C.paired_boot([100 * y for y in ya], [100 * y for y in yb]), **{"newcombe": C.newcombe(ya, yb)}},
        "cost_per_task": {ARM + "_api": float(np.mean(ca_api)) if ca_api else None,
                          ARM + "_incl_tariff": float(np.mean(ca)) if ca else None, REF: float(np.mean(cb)) if cb else None},
        "savings_incl_tariff": C.paired_boot(ca, cb, ratio=True),
        "savings_api_only": C.paired_boot(ca_api, cb, ratio=True),
        "delta_cost_usd_incl_tariff": C.paired_boot(ca, cb),
        "tariff": tf,
        "savings_sensitivity": {k: C.paired_boot([c + (v or 0.0) for c in ca_api], cb, ratio=True)["est"]
                                for k, v in tf["sensitivity"].items()},
        "calls_avoided_per_task": C.paired_boot([C.s_calls(ref[k], "s2") for k in pairs],
                                                [C.s_calls(arm[k], "s2") for k in pairs]),
        "delta_latency_s": C.paired_boot([C.latency_ms(arm[k]) / 1000 for k in pairs],
                                         [C.latency_ms(ref[k]) / 1000 for k in pairs]),
        "delta_wall_s": C.paired_boot([float(arm[k].get("wall_ms") or 0) / 1000 for k in pairs],
                                      [float(ref[k].get("wall_ms") or 0) / 1000 for k in pairs]),
        "arms": {ARM: arm_stats({k: arm[k] for k in pairs}), REF: arm_stats({k: ref[k] for k in pairs})},
    }
    sp = [float(((t.get("meta") or {}).get("tau2") or {}).get("agent_cost") or 0)
          + float(((t.get("meta") or {}).get("tau2") or {}).get("user_cost") or 0) for d in (arm, ref) for t in d.values()]
    out["api_spend_converted_usd"] = float(sum(sp))
    return out


def _md(r: Mapping[str, Any], prov: Mapping[str, Any]) -> str:
    d, s = r["delta_success_pt"], r["savings_incl_tariff"]
    A, B = r["arms"][ARM], r["arms"][REF]
    f = lambda x, n=1: "–" if x is None else f"{x:.{n}f}"  # noqa: E731
    lines = [
        "# E1 — live τ² telecom backend substitution with the frozen JevAgentRL-4B (exploratory)",
        "",
        f"{r['label']}. Arms `{ARM}` (τ² rendering of the profile, A7.1 (i), with `backends.jevagent_rl` thresholds; "
        f"System-1 = frozen JevAgentRL-4B served on the GPU host) vs `{REF}`; one attempt per task, paired per task "
        "(same tau2 seed), never-run telecom tasks (`configs/tau2/explore/telecom_rl_task_ids.json`).",
        "",
        f"n′ = {r['n_pairs']} of {r['n_expected']} (missing {r['missing']}, errored after retries "
        f"{ {k: len(v) for k, v in r['errored'].items()} }).",
        "",
        "| row | success % | Δ success pt [bootstrap 95 %] | Newcombe 95 % | $ / task (agent) | savings incl. tariff [95 %] | "
        "S2 calls / task | calls avoided | tool cov. | arg cov. | latency s | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| `{REF}` | {f(100 * B['success'])} | – | – | {f(B['agent_cost_per_task'], 4)} | – | {f(B['s2_calls_per_task'], 2)} | – "
        f"| {f(100 * (B['tool_coverage'] or 0))} | {f(100 * (B['arg_coverage'] or 0))} | {f(B['latency_s_per_task'], 2)} "
        f"| {f(B['wall_s_per_task'], 1)} |",
        f"| `{ARM}` | {f(100 * A['success'])} | {f(d['est'])} [{f(d['ci'][0])}, {f(d['ci'][1])}] | "
        f"{f(d['newcombe']['diff_pt'])} [{f(d['newcombe']['ci_pt'][0])}, {f(d['newcombe']['ci_pt'][1])}] | "
        f"{f(r['cost_per_task'][ARM + '_incl_tariff'], 4)} (API {f(r['cost_per_task'][ARM + '_api'], 4)}) | "
        f"{f(100 * s['est'])} % [{f(100 * s['ci'][0])}, {f(100 * s['ci'][1])}] | {f(A['s2_calls_per_task'], 2)} | "
        f"{f(r['calls_avoided_per_task']['est'], 2)} | {f(100 * (A['tool_coverage'] or 0))} | "
        f"{f(100 * (A['arg_coverage'] or 0))} | {f(A['latency_s_per_task'], 2)} | {f(A['wall_s_per_task'], 1)} |",
        "",
        f"Tariff q = {f(r['tariff']['q_usd_per_task'], 5)} $ / task ({r['tariff']['instance_type']}, "
        f"{f(r['tariff']['usd_per_hour'], 5)} $ / h, {f(r['tariff']['serving_hours'], 3)} h serving window, "
        f"{r['tariff']['tasks_completed']} tasks; {r['tariff']['convention']}). Savings under the sensitivity rows: "
        + "; ".join(f"{k} {f(100 * v)} %" for k, v in r["savings_sensitivity"].items() if v is not None) + ". "
        f"API-only savings {f(100 * r['savings_api_only']['est'])} % [{f(100 * r['savings_api_only']['ci'][0])}, "
        f"{f(100 * r['savings_api_only']['ci'][1])}].",
        "",
        f"System-1: {f(A['s1_calls_per_task'], 2)} calls / task; {A['tool_delegated']} of {A['tool_decisions']} tool "
        f"decisions executed System-1's choice ({f(100 * (A['delegated_respond_share'] or 0))} % of those "
        f"`respond_to_user`); on the {A['delegated_with_reference']} delegated choices with a reference next step, "
        f"agreement {f(100 * (A['delegated_agree_reference'] or 0))} % (approximate τ² reference, not ground truth). "
        f"Δ latency {f(r['delta_latency_s']['est'], 2)} s [{f(r['delta_latency_s']['ci'][0], 2)}, "
        f"{f(r['delta_latency_s']['ci'][1], 2)}] (model calls; descriptive), Δ wall {f(r['delta_wall_s']['est'], 1)} s.",
        "",
        f"API spend in the converted records: ${r['api_spend_converted_usd']:.2f} (agent ledger + user simulator).",
        "",
        "Provenance: " + "; ".join(f"{k} {v}" for k, v in prov.items()),
        "",
    ]
    return "\n".join(lines)


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    A, B = r["arms"][ARM], r["arms"][REF]
    d, s = r["delta_success_pt"], r["savings_incl_tariff"]
    k = {"n": C.fmt(r["n_pairs"]), "n_expected": C.fmt(r["n_expected"]),
         "success_arm": C.fmt(100 * A["success"]), "success_ref": C.fmt(100 * B["success"]),
         "dsucc": C.fmt(d["est"]), "dsucc_lo": C.fmt(d["ci"][0]), "dsucc_hi": C.fmt(d["ci"][1]),
         "dsucc_newcombe_lo": C.fmt(d["newcombe"]["ci_pt"][0]), "dsucc_newcombe_hi": C.fmt(d["newcombe"]["ci_pt"][1]),
         "cost_arm": C.fmt(r["cost_per_task"][ARM + "_incl_tariff"], 4), "cost_arm_api": C.fmt(r["cost_per_task"][ARM + "_api"], 4),
         "cost_ref": C.fmt(r["cost_per_task"][REF], 4),
         "savings": C.fmt(100 * s["est"]), "savings_lo": C.fmt(100 * s["ci"][0]), "savings_hi": C.fmt(100 * s["ci"][1]),
         "savings_api": C.fmt(100 * r["savings_api_only"]["est"]),
         "q": C.fmt(r["tariff"]["q_usd_per_task"], 5), "tool_cov": C.fmt(100 * (A["tool_coverage"] or 0)),
         "arg_cov": C.fmt(100 * (A["arg_coverage"] or 0)), "calls_avoided": C.fmt(r["calls_avoided_per_task"]["est"], 2),
         "s2_calls_arm": C.fmt(A["s2_calls_per_task"], 2), "s2_calls_ref": C.fmt(B["s2_calls_per_task"], 2),
         "s1_calls_arm": C.fmt(A["s1_calls_per_task"], 2),
         "dlat": C.fmt(r["delta_latency_s"]["est"], 2), "dwall": C.fmt(r["delta_wall_s"]["est"], 1),
         "agree_ref": C.fmt(100 * (A["delegated_agree_reference"] or 0)),
         "respond_share": C.fmt(100 * (A["delegated_respond_share"] or 0)),
         "spend": C.fmt(r["api_spend_converted_usd"], 2)}
    for nm, v in r["savings_sensitivity"].items():
        k[f"savings_{nm}"] = C.fmt(100 * v) if v is not None else r"\tbd"
    return k


def cmd_analyze(a: argparse.Namespace) -> int:
    ids = json.loads(IDS_OUT.read_text())
    run = Path(a.run)
    arm, na = C.load_latest(sorted((run / "telecom").glob(f"{ARM}*.jsonl")))
    ref, nr = C.load_latest(sorted((run / "telecom").glob(f"{REF}*.jsonl")))
    if not arm or not ref:
        print(f"no converted records in {run}/telecom", file=sys.stderr)
        return 2
    windows = json.loads(Path(a.windows).read_text()) if Path(a.windows).exists() else []
    if isinstance(windows, dict):
        windows = windows.get("windows") or []
    r = analyze(arm, ref, ids["task_ids"], windows, {ARM: na, REF: nr})
    prov = {C.rel(p): C.sha256_file(p)[:16] for p in sorted((run / "telecom").glob("*.jsonl"))}
    prov[C.rel(IDS_OUT)] = C.sha256_file(IDS_OUT)[:16]
    if Path(a.windows).exists():
        prov[C.rel(Path(a.windows))] = C.sha256_file(Path(a.windows))[:16]
    r["provenance"] = prov
    r["spend_raw"] = raw_spend(sorted(p for p in (run / "raw/telecom").glob("*") if p.is_dir())) \
        if (run / "raw/telecom").exists() else None
    r["generated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    r["git"] = C.git_head()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(_md(r, prov))
    if not a.no_tex:
        C.write_tex("telecom_rl", tex_keys(r), "scripts/explore_telecom_rl.py", Path(a.tex))
    print(_md(r, prov))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--n", type=int, default=N_TASKS)
    p.add_argument("--scan", action="append", help="run roots to scan (default: runs/)")
    p.add_argument("--allow-unscanned", action="store_true", help="fixtures only")
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("render")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("spend")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_spend)
    p = sub.add_parser("gonogo")
    p.add_argument("--round", type=int, default=1, choices=sorted(GONOGO_ROUNDS))
    p.add_argument("--run", default=None, help="default: the round's run directory")
    p.add_argument("--diagnosis", default=None, help="the session's diagnosis of a failed check (recorded verbatim)")
    p.add_argument("--tex", default=str(C.TEX))
    p.add_argument("--no-tex", action="store_true")
    p.set_defaults(fn=cmd_gonogo)
    p = sub.add_parser("analyze")
    p.add_argument("--run", default=str(RUN))
    p.add_argument("--windows", default=str(WINDOWS))
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--tex", default=str(C.TEX))
    p.add_argument("--no-tex", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
