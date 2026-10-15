#!/usr/bin/env python3
"""E29 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~16:05 UTC, "E29: the full harness in open-source harnesses";
exploratory, descriptive): the decision layer as a host-side model proxy (``jevharness.adapters.route_proxy``) in three
open-source harnesses, each against the unchanged harness on the same tasks.

  harness tb    pi 0.99.2 on Terminal-Bench 2.0 (Harbor)            Gemini 3.5 Flash / Flash-Lite   calibration: TB-Pro
  harness mswe  mini-SWE-agent 2.4.6 on SWE-bench Verified          DeepSeek-V4-Pro / V4-Flash      calibration: SWE-bench
  harness swea  SWE-agent v1.1.0 on SWE-bench Verified (Harbor)     Gemini 3.5 Flash / Flash-Lite   calibration: SWE-bench

Subcommands (no model call anywhere in this script):

  select   --registry <harbor registry.json>       the frozen task lists -> configs/explore/e29_port/
  fit      --harness H --logs DIR --out gate.json  the registered gate fit from a shadow (calibration) proxy log
  smoke    --harness H --logs DIR --run DIR        the registered smoke checks (exit 1 = stop)
  summary  --harness H --stage pilot|full --logs DIR --run DIR --out X.json   per-task rows joined with the proxy log
  go       --tb X.json --mswe Y.json --out go.json                             the registered go rule (exit 0 go, 3 no-go)
  spend    DIR ...                                                             logged spend of proxy log dirs (JSON)
  readout  [--box results/explore/e29_port/box]    summary.{md,json} + paper/generated/e29_port.tex

Cost of a task = every proxy-logged LLM call of its episode (OpenRouter's usage.cost; Gemini at the configured prices)
plus its decision-model calls; time = Harbor's agent-execution seconds (tb, swea) or mini-SWE-agent's agent wall-clock
(mswe); success = Harbor reward 1 / SWE-bench resolved. An errored trial (an exception other than an agent timeout) is
a failure in the full set (primary) and leaves the pair in the complete case.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CFG = ROOT / "configs/explore/e29_port"
OUT_DIR = ROOT / "results/explore/e29_port"
BOX = OUT_DIR / "box"
TEX = ROOT / "paper/generated/e29_port.tex"
REG = "docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~16:05 UTC, E29"

SALT_TBPRO = "jev-explore-E29-tbpro-2026-10-09"
SALT_SWECAL = "jev-explore-E29-swecal-2026-10-09"
N_CAL = {"tb": 10, "mswe": 20, "swea": 15}
N_PILOT = 30
HARNESSES: dict[str, dict[str, Any]] = {
    "tb": {"label": "pi on Terminal-Bench 2.0", "harness": "pi 0.99.2", "bench": "Terminal-Bench 2.0",
           "llms": "Gemini 3.5 Flash / Flash-Lite", "kind": "harbor", "ref": "E25 harness A",
           "ref_saving": 0.096, "ref_delta_pt": 7.9},
    "mswe": {"label": "mini-SWE-agent on SWE-bench Verified", "harness": "mini-SWE-agent 2.4.6",
             "bench": "SWE-bench Verified", "llms": "DeepSeek-V4-Pro / V4-Flash", "kind": "mswe",
             "ref": "E4r (pooled 300)", "ref_saving": 0.105, "ref_delta_pt": -1.0},
    "swea": {"label": "SWE-agent on SWE-bench Verified", "harness": "SWE-agent v1.1.0", "bench": "SWE-bench Verified",
             "llms": "Gemini 3.5 Flash / Flash-Lite", "kind": "harbor", "ref": "E25 harness B",
             "ref_saving": 0.077, "ref_delta_pt": -6.0},
}
ARMS = ("baseline", "gated")
CAPS = {"pilot": {"tb": 38.0, "mswe": 7.0}, "full": {"tb": 100.0, "mswe": 25.0, "swea": 80.0}}
TOTAL_CAP = 300.0
V_USD = 1.0                      # registration (c) 3
GO_SAVING = 0.20                 # registration (e)
GO_MAX_DROP = 3
MARGIN = 0.03
N_BOOT = 10_000
SEED = 29


# ------------------------------------------------------------------------------------------------ small helpers
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def read_ids(path: Path) -> list[str]:
    return [ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines() if ln.split("#")[0].strip()]


def ids_text(ids: list[str]) -> str:
    return "\n".join(ids) + "\n"


def seeded_order(ids: Iterable[str], salt: str) -> list[str]:
    return sorted(set(ids), key=lambda x: sha256_text(f"{salt}\n{x}"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        if not gz.exists():
            return []
        text = gzip.open(gz, "rt").read()
    else:
        text = path.read_text(errors="replace")
    out = []
    for ln in text.splitlines():
        try:
            obj = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


# ------------------------------------------------------------------------------------------------ select
SWE_EXCLUDE = {   # registration (c) 1: every E29 evaluation list and every smoke / development instance of E4/E4r/E25/H5
    "e4_eval": "configs/explore/e4_swe/eval_ids.txt",
    "e4_dev": "configs/explore/e4_swe/dev_ids.txt",
    "e4_smoke": "configs/explore/e4_swe/smoke_ids.txt",
    "e4r_extension": "configs/explore/e4r_swe/extension_ids.txt",
    "e4r_smoke": "configs/explore/e4r_swe/smoke_ids.txt",
    "e25_swe_eval": "configs/explore/e25_harnesses/swe_eval_ids.txt",
    "e25_swe_smoke": "configs/explore/e25_harnesses/swe_smoke_ids.txt",
    "h5_smoke": "results/hard/smoke/swe/instance_ids.txt",
}
SWE_DEV_IDS = ("django__django-11099", "pytest-dev__pytest-7432", "sympy__sympy-20590")   # explore_e4_swe.DEV_SWE_IDS


def select(registry: list[dict[str, Any]]) -> dict[str, list[str]]:
    pro = [d for d in registry if d.get("name") == "terminal-bench-pro" and str(d.get("version")) == "1.0"]
    tb2 = [d for d in registry if d.get("name") == "terminal-bench" and str(d.get("version")) == "2.0"]
    if len(pro) != 1 or len(tb2) != 1:
        raise SystemExit("registry: expected terminal-bench-pro@1.0 and terminal-bench@2.0")
    pro_names = [t["name"] for t in pro[0]["tasks"]]
    tb2_names = {t["name"] for t in tb2[0]["tasks"]}
    if len(pro_names) != 200 or tb2_names & set(pro_names):
        raise SystemExit("Terminal-Bench Pro must have 200 tasks and none named like a Terminal-Bench 2.0 task")
    pro_order = seeded_order(pro_names, SALT_TBPRO)
    pool = read_ids(ROOT / "configs/explore/e25_harnesses/swe_order.txt")
    if len(set(pool)) != 500:
        raise SystemExit("expected the 500 SWE-bench Verified ids")
    excluded: set[str] = set(SWE_DEV_IDS)
    for p in SWE_EXCLUDE.values():
        excluded |= set(read_ids(ROOT / p))
    swe_order = seeded_order([i for i in pool if i not in excluded], SALT_SWECAL)
    tb_eval = read_ids(ROOT / "configs/explore/e25_harnesses/tb2_order.txt")
    mswe_eval = read_ids(ROOT / "configs/explore/e4r_swe/extension_ids.txt")
    swea_eval = read_ids(ROOT / "configs/explore/e25_harnesses/swe_eval_ids.txt")
    assert len(tb_eval) == 89 and len(mswe_eval) == 200 and len(swea_eval) == 100
    lists = {
        "tb_cal": pro_order[:N_CAL["tb"]], "tb_smoke": pro_order[N_CAL["tb"]:N_CAL["tb"] + 1],
        "tb_pilot": tb_eval[:N_PILOT], "tb_eval": tb_eval,
        "mswe_cal": swe_order[:N_CAL["mswe"]], "mswe_smoke": read_ids(ROOT / SWE_EXCLUDE["e4r_smoke"]),
        "mswe_pilot": mswe_eval[:N_PILOT], "mswe_eval": mswe_eval,
        "swea_cal": swe_order[:N_CAL["swea"]], "swea_smoke": read_ids(ROOT / SWE_EXCLUDE["e25_swe_smoke"]),
        "swea_eval": swea_eval,
    }
    evals = set(tb_eval) | set(mswe_eval) | set(swea_eval)
    for h in ("tb", "mswe", "swea"):
        assert not set(lists[f"{h}_cal"]) & evals and not set(lists[f"{h}_smoke"]) & set(lists[f"{h}_eval"])
    return lists


def cmd_select(a: argparse.Namespace) -> int:
    lists = select(json.loads(Path(a.registry).read_text()))
    CFG.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"registered": REG, "harbor_registry": "laude-institute/harbor@4f848e3 registry.json",
                            "salts": {"tbpro": SALT_TBPRO, "swecal": SALT_SWECAL},
                            "rule": "tb_cal = the first 10 Terminal-Bench Pro tasks in sha256(salt + newline + name) "
                                    "order, tb_smoke the 11th; SWE calibration = the 500 Verified ids minus "
                                    + ", ".join(SWE_EXCLUDE) + " and the adapter's development ids, in "
                                    "sha256(salt + newline + id) order: mswe the first 20, swea the first 15; pilots = "
                                    "the first 30 of tb2_order.txt / E4r's extension list; evaluation = tb2_order.txt "
                                    "(89), E4r's extension (200), E25's swe_eval_ids (100)",
                            "files": {}}
    for name, ids in lists.items():
        text = ids_text(ids)
        (CFG / f"{name}.txt").write_text(text)
        meta["files"][f"{name}.txt"] = {"n": len(ids), "sha256": sha256_text(text)}
    (CFG / "selection.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v["n"] for k, v in meta["files"].items()}))
    return 0


# ------------------------------------------------------------------------------------------------ proxy logs
def read_logs(dirs: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls, decisions = [], []
    for d in dirs:
        d = Path(d)
        calls += jsonl(d / "calls.jsonl")
        decisions += jsonl(d / "decisions.jsonl")
    return calls, decisions


def spend_of(calls: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: Counter[str] = Counter()
    for r in calls + decisions:
        by_arm[str(r.get("arm"))] += float(r.get("cost_usd") or 0.0)
    llm = sum(float(r.get("cost_usd") or 0.0) for r in calls)
    dm = sum(float(r.get("cost_usd") or 0.0) for r in decisions)
    return {"llm": round(llm, 6), "decision_model": round(dm, 6), "total": round(llm + dm, 6),
            "by_arm": {k: round(v, 6) for k, v in sorted(by_arm.items())}, "calls": len(calls),
            "decisions": len(decisions), "unpriced_calls": sum(1 for r in calls if r.get("priced") is False)}


def cmd_spend(a: argparse.Namespace) -> int:
    print(json.dumps(spend_of(*read_logs([Path(d) for d in a.dirs]))))
    return 0


# ------------------------------------------------------------------------------------------------ fit
def shadow_steps(calls: list[dict[str, Any]], decisions: list[dict[str, Any]], arm: str = "shadow"
                 ) -> list[dict[str, Any]]:
    """One row per calibration step t >= 1 with all three parsed actions and a decision-model answer."""
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in calls:
        if r.get("arm") != arm or r.get("status") != 200 or r.get("key") is None:
            continue
        by[(str(r["episode"]), int(r["step"]))][str(r["role"])] = r
    dec = {(str(d["episode"]), int(d["step"])): d for d in decisions if d.get("arm") == arm}
    rows = []
    for (ep, t), roles in sorted(by.items()):
        if t < 1 or not {"primary", "shadow_fast", "shadow_strong"} <= set(roles):
            continue
        d = dec.get((ep, t))
        if d is None or d.get("c") is None:
            continue
        p, f, s = roles["primary"], roles["shadow_fast"], roles["shadow_strong"]
        rows.append({"episode": ep, "step": t, "c": float(d["c"]), "y_f": int(f["key"] == p["key"]),
                     "y_s": int(s["key"] == p["key"]),
                     "dc": float(p.get("cost_usd") or 0.0) - float(f.get("cost_usd") or 0.0)})
    return rows


def fit_gate(rows: list[dict[str, Any]], V: float = V_USD) -> dict[str, Any]:
    """Registration (c) 3: tau-hat = argmax_tau G(tau), G(tau) = sum_{c_t >= tau} [y_f - y_s + dC / V] over the
    candidate thresholds {every observed c} (and "never", G = 0); ties go to the higher tau; max G <= 0 -> no gate."""
    n = len(rows)
    out: dict[str, Any] = {"n_steps": n, "n_episodes": len({r["episode"] for r in rows}), "V_usd": V}
    if not n:
        return {**out, "tau": None, "reason": "no calibration step", "G_max": 0.0}
    dC = sum(r["dc"] for r in rows) / n
    gain = [(r["c"], r["y_f"] - r["y_s"] + dC / V, r) for r in rows]
    gain.sort(key=lambda x: -x[0])
    best_tau, best_G, G = None, 0.0, 0.0
    curve = []
    i = 0
    while i < n:
        c = gain[i][0]
        while i < n and gain[i][0] == c:          # all steps tied at this c enter together
            G += gain[i][1]
            i += 1
        curve.append((c, G, i))
        if G > best_G + 1e-12:                    # strictly better: a lower tau must gain more (ties -> higher tau)
            best_tau, best_G = c, G
    routed = [r for r in rows if best_tau is not None and r["c"] >= best_tau]
    step = max(1, len(curve) // 60)
    out.update({
        "tau": best_tau, "G_max": round(best_G, 6), "dC_usd": dC, "dC_over_V": dC / V,
        "agree_fast": sum(r["y_f"] for r in rows) / n, "agree_self": sum(r["y_s"] for r in rows) / n,
        "n_routed_sim": len(routed), "sim_fast_share": len(routed) / n,
        "routed_agree_fast": (sum(r["y_f"] for r in routed) / len(routed)) if routed else None,
        "routed_agree_self": (sum(r["y_s"] for r in routed) / len(routed)) if routed else None,
        "c_mean": sum(r["c"] for r in rows) / n,
        "curve": [{"tau": round(c, 6), "G": round(g, 4), "n": k} for c, g, k in curve[::step]],
        "reason": "argmax G" if best_tau is not None else "max G <= 0: no step pays, the gate routes nothing",
    })
    return out


def _tool(key: str | None) -> str:
    if not key or key == "<no tool call>":
        return "none"
    return key.split(":", 1)[0]


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def calib_descriptives(calls: list[dict[str, Any]], decisions: list[dict[str, Any]], arm: str = "shadow"
                       ) -> dict[str, Any]:
    """Post hoc, descriptive (not part of the registered fit): why a gate does or does not route. Agreement of the fast
    LLM and of the second strong sample with the strong LLM's action, by the tool of that action and by the decision
    model's answer; Jev's AUROC for "the fast action equals the strong one"; a looser label (the same tool only); the
    steps a fixed tau = 0.5 (E25's) would have routed."""
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in calls:
        if r.get("arm") == arm and r.get("status") == 200 and r.get("key") is not None:
            by[(str(r["episode"]), int(r["step"]))][str(r["role"])] = r
    dec = {(str(d["episode"]), int(d["step"])): d for d in decisions if d.get("arm") == arm}
    rows = []
    for (ep, t), roles in by.items():
        if t < 1 or not {"primary", "shadow_fast", "shadow_strong"} <= set(roles) or (ep, t) not in dec:
            continue
        c = dec[(ep, t)].get("c")
        if c is None:
            continue
        kp, kf, ks = roles["primary"]["key"], roles["shadow_fast"]["key"], roles["shadow_strong"]["key"]
        rows.append({"c": float(c), "tool": _tool(kp), "y_f": int(kf == kp), "y_s": int(ks == kp),
                     "t_f": int(_tool(kf) == _tool(kp)), "t_s": int(_tool(ks) == _tool(kp))})
    if not rows:
        return {"n": 0}
    n = len(rows)

    def agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        k = len(rs)
        return {"n": k, "agree_fast": sum(r["y_f"] for r in rs) / k if k else None,
                "agree_self": sum(r["y_s"] for r in rs) / k if k else None,
                "same_tool_fast": sum(r["t_f"] for r in rs) / k if k else None,
                "same_tool_self": sum(r["t_s"] for r in rs) / k if k else None}
    tools = Counter(r["tool"] for r in rows)
    return {"n": n, "all": agg(rows),
            "by_tool": {tl: agg([r for r in rows if r["tool"] == tl]) for tl, _ in tools.most_common(6)},
            "jev_says_fast": agg([r for r in rows if r["c"] >= 0.5]),
            "jev_says_strong": agg([r for r in rows if r["c"] < 0.5]),
            "auroc_c_for_fast_agreement": _auroc([r["c"] for r in rows], [r["y_f"] for r in rows]),
            "auroc_c_for_self_agreement": _auroc([r["c"] for r in rows], [r["y_s"] for r in rows]),
            "fixed_tau_0p5_share": sum(r["c"] >= 0.5 for r in rows) / n,
            "note": "post hoc, descriptive; not part of the registered fit"}


def cmd_fit(a: argparse.Namespace) -> int:
    calls, decisions = read_logs([Path(a.logs)])
    rows = shadow_steps(calls, decisions, arm=a.arm)
    res = fit_gate(rows)
    res["descriptive"] = calib_descriptives(calls, decisions, arm=a.arm)
    res.update({"harness": a.harness, "registered": REG, "written": _now(), "logs": str(a.logs),
                "decisions_total": sum(1 for d in decisions if d.get("arm") == a.arm),
                "decision_errors": sum(1 for d in decisions if d.get("arm") == a.arm and d.get("c") is None),
                "shadow_calls_failed": sum(1 for c in calls if c.get("arm") == a.arm and c.get("role", "").startswith(
                    "shadow") and c.get("status") != 200)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res.get(k) for k in ("tau", "n_steps", "n_episodes", "sim_fast_share", "agree_fast",
                                              "agree_self", "routed_agree_fast", "routed_agree_self", "G_max",
                                              "dC_usd", "reason")}))
    return 0


def _ts(x: Any) -> datetime | None:
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------------ episodes
def episode_stats(calls: list[dict[str, Any]], decisions: list[dict[str, Any]], arm: str, k: int = 4
                  ) -> dict[str, dict[str, Any]]:
    """Per episode of one arm: cost, requests, fast requests, the route reasons, sticky segments, decision calls."""
    out: dict[str, dict[str, Any]] = {}
    prim: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in calls:
        if r.get("arm") != arm:
            continue
        ep = str(r["episode"])
        e = out.setdefault(ep, {"llm_cost": 0.0, "dm_cost": 0.0, "requests": 0, "fast_requests": 0, "errors": 0,
                                "reasons": Counter(), "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                                "decisions": 0, "decision_errors": 0, "dm_latency_s": 0.0, "waited_s": 0.0,
                                "llm_latency_s": 0.0, "guard_accept": 0, "guard_reject": 0,
                                "guard_reject_cost": 0.0, "answers": Counter()})
        e["llm_cost"] += float(r.get("cost_usd") or 0.0)
        if r.get("role") != "primary":
            continue
        e["requests"] += 1
        e["llm_latency_s"] += float(r.get("latency_s") or 0.0)
        for t in ("input_tokens", "cached_tokens", "output_tokens"):
            e[t] += int(r.get(t) or 0)
        if r.get("status") != 200:
            e["errors"] += 1
            continue
        if r.get("tier") == "fast":
            e["fast_requests"] += 1
        if r.get("guard") == "accept":          # E29b's write guard (absent in E29's records)
            e["guard_accept"] += 1
        elif r.get("guard") == "reject":
            e["guard_reject"] += 1
            e["guard_reject_cost"] += float(r.get("cost_usd") or 0.0)
        e["reasons"][str(r.get("route_reason"))] += 1
        e["waited_s"] += float(r.get("waited_s") or 0.0)
        prim[ep][int(r["step"])] = r
    for d in decisions:
        if d.get("arm") != arm:
            continue
        e = out.get(str(d["episode"]))
        if e is None:
            continue
        e["dm_cost"] += float(d.get("cost_usd") or 0.0)
        e["decisions"] += 1
        if d.get("value") is not None:
            e["answers"][str(d.get("value"))] += 1
        e["decision_errors"] += int(d.get("c") is None)
        e["dm_latency_s"] += float(d.get("latency_s") or 0.0)
    for ep, steps in prim.items():
        tiers = [steps[t].get("tier") for t in sorted(steps)]
        segs, cur = [], 0
        for x in tiers:
            if x == "fast":
                cur += 1
            elif cur:
                segs.append(cur)
                cur = 0
        if cur:
            segs.append(cur)
        e = out[ep]
        e["steps"] = len(tiers)
        e["fast_steps"] = sum(1 for x in tiers if x == "fast")
        e["segments"] = len(segs)
        e["segment_len_mean"] = _mean([float(s) for s in segs])
        e["segments_capped"] = sum(1 for s in segs if s >= k)
        e["switches"] = sum(1 for a, b in zip(tiers, tiers[1:]) if a != b)
    for e in out.values():
        e["cost"] = e["llm_cost"] + e["dm_cost"]
        e["reasons"] = dict(e["reasons"])
        e["answers"] = dict(e["answers"])
        e.setdefault("steps", 0)
        e.setdefault("fast_steps", 0)
        e.setdefault("segments", 0)
        e.setdefault("segment_len_mean", None)
        e.setdefault("segments_capped", 0)
        e.setdefault("switches", 0)
    return out


def _sanitize(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)


def harbor_trials(job_dir: Path) -> dict[str, dict[str, Any]]:
    """task -> trial row of one Harbor job (latest started trial wins; k = 1)."""
    rows: dict[str, dict[str, Any]] = {}
    if not job_dir.is_dir():
        return rows
    for t in sorted(p for p in job_dir.iterdir() if p.is_dir() and (p / "result.json").exists()):
        try:
            r = json.loads((t / "result.json").read_text())
        except (OSError, ValueError):
            continue
        exc = r.get("exception_info") or {}
        err = exc.get("exception_type")
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        ae = r.get("agent_execution") or {}
        wall = None
        try:
            s, e = ae.get("started_at"), ae.get("finished_at")
            if s and e:
                wall = (datetime.fromisoformat(e.replace("Z", "+00:00"))
                        - datetime.fromisoformat(s.replace("Z", "+00:00"))).total_seconds()
        except (ValueError, TypeError):
            wall = None
        task = str(r.get("task_name") or t.name.split("__")[0]).split("/")[-1]
        started = str(r.get("started_at") or "")
        if task in rows and rows[task]["started_at"] > started:      # Harbor's retries: the latest attempt counts
            continue
        rows[task] = {"task": task, "trial": t.name, "episode": _sanitize(t.name), "started_at": started,
                      "success": int(reward is not None and float(reward) >= 1.0), "reward": reward,
                      "wall": wall, "error": err, "timed_out": err == "AgentTimeoutError",
                      "infra_error": bool(err) and err not in ("AgentTimeoutError",),
                      "cancelled": err == "CancelledError",
                      "error_msg": str(exc.get("exception_message") or "")[:200] if err else None}
    return rows


def mswe_trials(arm_dir: Path) -> dict[str, dict[str, Any]]:
    """instance -> row from mini-SWE-agent's collected ``<arm>.jsonl`` (resolved from the SWE-bench report)."""
    import explore_e4_swe as E4
    path = arm_dir.parent / f"{arm_dir.name}.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for iid, r in E4.load_arm(path).items():
        rows[iid] = {"task": iid, "trial": iid, "episode": iid, "success": int(bool(r.get("success"))),
                     "wall": r.get("time_s"), "error": r.get("error"), "timed_out": False,
                     "infra_error": bool(r.get("error")), "cancelled": False,
                     "harness_cost": r.get("total_cost"), "exit_status": r.get("exit_status")}
    return rows


def load_arm_rows(harness: str, run_dir: Path, arm: str, stats: Mapping[str, Mapping[str, Any]]
                  ) -> dict[str, dict[str, Any]]:
    trials = mswe_trials(run_dir / arm) if HARNESSES[harness]["kind"] == "mswe" else harbor_trials(run_dir / arm)
    out = {}
    for task, r in trials.items():
        st = stats.get(r["episode"]) or {}
        out[task] = {**r, **{k: st.get(k) for k in ("cost", "llm_cost", "dm_cost", "requests", "fast_requests",
                                                    "errors", "reasons", "decisions", "decision_errors",
                                                    "dm_latency_s", "waited_s", "segments", "segment_len_mean",
                                                    "segments_capped", "switches", "input_tokens", "cached_tokens",
                                                    "output_tokens", "llm_latency_s", "steps", "fast_steps",
                                                    "guard_accept", "guard_reject", "guard_reject_cost",
                                                    "answers")},
                     "in_proxy_log": bool(st)}
        if out[task]["cost"] is None:
            out[task]["cost"] = 0.0
    return out


# ------------------------------------------------------------------------------------------------ paired statistics
def paired(base: Mapping[str, Mapping[str, Any]], arm: Mapping[str, Mapping[str, Any]], tasks: list[str],
           complete_case: bool = False) -> dict[str, Any]:
    from jevharness.bench.metrics import ni_pvalue, tango_interval
    if complete_case:
        tasks = [t for t in tasks if not base[t].get("infra_error") and not arm[t].get("infra_error")]
    n = len(tasks)
    if not n:
        return {"n": 0}
    b = np.array([float(base[t]["success"]) for t in tasks])
    a = np.array([float(arm[t]["success"]) for t in tasks])
    cb = np.array([float(base[t]["cost"] or 0.0) for t in tasks])
    ca = np.array([float(arm[t]["cost"] or 0.0) for t in tasks])
    wb = [float(base[t]["wall"]) for t in tasks if base[t].get("wall") is not None]
    wa = [float(arm[t]["wall"]) for t in tasks if arm[t].get("wall") is not None]
    both_w = [t for t in tasks if base[t].get("wall") is not None and arm[t].get("wall") is not None]
    tb = np.array([float(base[t]["wall"]) for t in both_w])
    ta = np.array([float(arm[t]["wall"]) for t in both_w])
    wins = int(((a == 1) & (b == 0)).sum())
    losses = int(((a == 0) & (b == 1)).sum())
    both = int(((a == 1) & (b == 1)).sum())
    neither = n - wins - losses - both
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d_boot = (a[idx] - b[idx]).mean(axis=1) * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        s_boot = 1 - ca[idx].mean(axis=1) / cb[idx].mean(axis=1)
    out: dict[str, Any] = {
        "n": n, "wins": wins, "losses": losses, "both": both, "neither": neither,
        "success_base": float(b.mean()), "success_arm": float(a.mean()),
        "solved_base": int(b.sum()), "solved_arm": int(a.sum()),
        "delta_pt": float((a - b).mean() * 100),
        "delta_ci": [float(np.percentile(d_boot, 2.5)), float(np.percentile(d_boot, 97.5))],
        "cost_base": float(cb.mean()), "cost_arm": float(ca.mean()),
        "saving": float(1 - ca.mean() / cb.mean()) if cb.mean() > 0 else None,
        "saving_ci": [float(np.nanpercentile(s_boot, 2.5)), float(np.nanpercentile(s_boot, 97.5))],
        "cheaper_share": float((ca < cb).mean()),
    }
    try:
        tg = tango_interval(both, wins, losses, neither)
        out["tango"] = [float(tg[0]) * 100, float(tg[1]) * 100] if isinstance(tg, (list, tuple)) else tg
    except Exception:  # noqa: BLE001 - descriptive only
        out["tango"] = None
    ni = ni_pvalue({t: float(base[t]["success"]) for t in tasks}, {t: float(arm[t]["success"]) for t in tasks},
                   margin=MARGIN, seed=SEED)
    out["ni_p"] = float(ni.get("p_decisive", ni.get("p_bb")))
    out["ni"] = bool(out["ni_p"] <= 0.05)
    if len(both_w):
        jb = rng.integers(0, len(both_w), size=(N_BOOT, len(both_w)))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_boot = 1 - ta[jb].mean(axis=1) / tb[jb].mean(axis=1)
        out.update({"time_base": float(tb.mean()), "time_arm": float(ta.mean()),
                    "time_saving": float(1 - ta.mean() / tb.mean()),
                    "time_saving_ci": [float(np.nanpercentile(t_boot, 2.5)), float(np.nanpercentile(t_boot, 97.5))],
                    "median_time_base": float(np.median(tb)), "median_time_arm": float(np.median(ta)),
                    "median_time_saving": float(1 - np.median(ta) / np.median(tb)),
                    "faster_share": float((ta < tb).mean()), "n_time": len(both_w)})
    return out


def arm_summary(rows: Mapping[str, Mapping[str, Any]], tasks: list[str]) -> dict[str, Any]:
    rs = [rows[t] for t in tasks]
    req = sum(int(r.get("requests") or 0) for r in rs)
    fast = sum(int(r.get("fast_requests") or 0) for r in rs)
    reasons: Counter[str] = Counter()
    for r in rs:
        reasons.update(r.get("reasons") or {})
    segs = sum(int(r.get("segments") or 0) for r in rs)
    seg_lens = [float(r["segment_len_mean"]) * int(r["segments"]) for r in rs if r.get("segments")]
    return {"n": len(rs), "solved": sum(int(r["success"]) for r in rs),
            "success": _mean([float(r["success"]) for r in rs]),
            "cost_mean": _mean([float(r.get("cost") or 0.0) for r in rs]),
            "llm_cost_mean": _mean([float(r.get("llm_cost") or 0.0) for r in rs]),
            "dm_cost_mean": _mean([float(r.get("dm_cost") or 0.0) for r in rs]),
            "wall_mean": _mean([float(r["wall"]) for r in rs if r.get("wall") is not None]),
            "wall_median": _median([float(r["wall"]) for r in rs if r.get("wall") is not None]),
            "requests_mean": req / len(rs) if rs else None, "fast_share": fast / req if req else 0.0,
            "fast_requests": fast, "requests": req,
            "input_tokens_mean": _mean([float(r.get("input_tokens") or 0) for r in rs]),
            "cached_share": (sum(int(r.get("cached_tokens") or 0) for r in rs)
                             / max(1, sum(int(r.get("input_tokens") or 0) for r in rs))),
            "reasons": dict(reasons), "segments_per_task": segs / len(rs) if rs else None,
            "segment_len_mean": (sum(seg_lens) / segs) if segs else None,
            "segments_capped_share": (sum(int(r.get("segments_capped") or 0) for r in rs) / segs) if segs else None,
            "switches_per_task": _mean([float(r.get("switches") or 0) for r in rs]),
            "decisions_per_task": _mean([float(r.get("decisions") or 0) for r in rs]),
            "decision_errors": sum(int(r.get("decision_errors") or 0) for r in rs),
            "fail_open": sum(v for k, v in reasons.items() if k.startswith("fail_open")),
            "dm_latency_per_task_s": _mean([float(r.get("dm_latency_s") or 0.0) for r in rs]),
            "waited_per_task_s": _mean([float(r.get("waited_s") or 0.0) for r in rs]),
            "llm_errors": sum(int(r.get("errors") or 0) for r in rs),
            "infra_errors": sum(1 for r in rs if r.get("infra_error")),
            "timeouts": sum(1 for r in rs if r.get("timed_out")),
            "not_in_proxy_log": sum(1 for r in rs if not r.get("in_proxy_log")),
            # E29b: executed steps written by the fast LLM and the write guard's verdicts (0 in E29's records)
            "steps": sum(int(r.get("steps") or 0) for r in rs),
            "fast_steps": sum(int(r.get("fast_steps") or 0) for r in rs),
            "fast_step_share": (sum(int(r.get("fast_steps") or 0) for r in rs)
                                / max(1, sum(int(r.get("steps") or 0) for r in rs))),
            "guard_accept": sum(int(r.get("guard_accept") or 0) for r in rs),
            "guard_reject": sum(int(r.get("guard_reject") or 0) for r in rs),
            "guard_reject_cost_mean": _mean([float(r.get("guard_reject_cost") or 0.0) for r in rs])}


def summarize(harness: str, run_dir: Path, log_dirs: list[Path], tasks: list[str] | None = None, k: int = 4,
              rerun_dir: Path | None = None, rerun_logs: list[Path] | None = None) -> dict[str, Any]:
    """Per-task rows of both arms joined with the proxy logs, and the paired statistics. ``rerun_dir`` /
    ``rerun_logs``: the one re-run of trials with an infrastructure error; a re-run row replaces the errored row of the
    same task (marked ``rerun``), joined with the re-run's own proxy log."""
    calls, decisions = read_logs(log_dirs)
    rows = {arm: load_arm_rows(harness, run_dir, arm, episode_stats(calls, decisions, arm, k)) for arm in ARMS}
    reruns: dict[str, list[str]] = {arm: [] for arm in ARMS}
    if rerun_dir is not None and Path(rerun_dir).is_dir():
        rc, rd = read_logs(rerun_logs or [])
        for arm in ARMS:
            again = load_arm_rows(harness, Path(rerun_dir), arm, episode_stats(rc, rd, arm, k))
            for t, r in again.items():
                if t in rows[arm] and rows[arm][t].get("infra_error"):
                    rows[arm][t] = {**r, "rerun": True, "first_error": rows[arm][t].get("error")}
                    reruns[arm].append(t)
    calls, decisions = calls + (rc if rerun_dir is not None and Path(rerun_dir).is_dir() else []), \
        decisions + (rd if rerun_dir is not None and Path(rerun_dir).is_dir() else [])
    listed = tasks or sorted(set(rows["baseline"]) | set(rows["gated"]))
    complete = [t for t in listed if t in rows["baseline"] and t in rows["gated"]
                and not rows["baseline"][t].get("cancelled") and not rows["gated"][t].get("cancelled")]
    missing = {arm: [t for t in listed if t not in rows[arm] or rows[arm][t].get("cancelled")] for arm in ARMS}
    out: dict[str, Any] = {"harness": harness, **{k2: HARNESSES[harness][k2] for k2 in ("label", "llms", "ref")},
                           "n_listed": len(listed), "n": len(complete), "missing": missing,
                           "arms": {arm: arm_summary(rows[arm], complete) for arm in ARMS},
                           "spend": spend_of(calls, decisions), "reruns": reruns, "written": _now()}
    if complete:
        out["paired"] = paired(rows["baseline"], rows["gated"], complete)
        out["paired_cc"] = paired(rows["baseline"], rows["gated"], complete, complete_case=True)
    out["per_task"] = {arm: {t: {k2: v for k2, v in rows[arm][t].items() if k2 != "reasons"} for t in complete}
                       for arm in ARMS}
    return out


def cmd_summary(a: argparse.Namespace) -> int:
    tasks = read_ids(Path(a.tasks)) if a.tasks else None
    res = summarize(a.harness, Path(a.run), [Path(d) for d in a.logs], tasks,
                    rerun_dir=Path(a.rerun_run) if a.rerun_run else None,
                    rerun_logs=[Path(d) for d in (a.rerun_logs or [])])
    res["stage"] = a.stage
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    p = res.get("paired") or {}
    print(json.dumps({"harness": a.harness, "stage": a.stage, "n": res["n"], "missing": {k: len(v) for k, v in
                                                                                         res["missing"].items()},
                      "solved": [p.get("solved_base"), p.get("solved_arm")], "saving": p.get("saving"),
                      "fast_share": res["arms"]["gated"]["fast_share"], "time_saving": p.get("time_saving")}))
    return 0


# ------------------------------------------------------------------------------------------------ go rule
def go_rule(pilots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Registration (e): pass(h) = saving >= 20 % and drop <= 3; not_worse(h) = saving >= the reference row's and
    drop <= 3; GO iff pass(tb) and pass(mswe), or pass(one) and not_worse(other). A harness whose gate routes nothing
    (or that has no pilot) has saving 0."""
    per: dict[str, Any] = {}
    for h in ("tb", "mswe"):
        s = pilots.get(h) or {}
        p = s.get("paired") or {}
        saving = float(p.get("saving") or 0.0) if not s.get("no_gate") else 0.0
        drop = int(p.get("solved_base", 0)) - int(p.get("solved_arm", 0)) if p else 99
        per[h] = {"n": s.get("n", 0), "saving": saving, "drop": drop,
                  "solved_base": p.get("solved_base"), "solved_gated": p.get("solved_arm"),
                  "pass": bool(p) and saving >= GO_SAVING and drop <= GO_MAX_DROP,
                  "not_worse": bool(p) and saving >= HARNESSES[h]["ref_saving"] and drop <= GO_MAX_DROP,
                  "no_gate": bool(s.get("no_gate"))}
    t, m = per["tb"], per["mswe"]
    go = (t["pass"] and m["pass"]) or (t["pass"] and m["not_worse"]) or (m["pass"] and t["not_worse"])
    return {"decision": "go" if go else "no-go", "per_harness": per, "rule": (
        "GO iff pass(tb) and pass(mswe), or pass of one and not_worse of the other; pass = saving >= 20 % and at "
        "most 3 tasks of 30 lost; not_worse = saving >= the reference row's (pi 9.6 %, mini-SWE-agent 10.5 %) and at "
        "most 3 tasks lost"), "registered": REG, "written": _now()}


def cmd_go(a: argparse.Namespace) -> int:
    pilots = {}
    for h, path in (("tb", a.tb), ("mswe", a.mswe)):
        if path and Path(path).exists():
            pilots[h] = json.loads(Path(path).read_text())
    res = go_rule(pilots)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if res["decision"] == "go" else 3


# ------------------------------------------------------------------------------------------------ smoke checks
def smoke_checks(harness: str, calls: list[dict[str, Any]], decisions: list[dict[str, Any]],
                 trials: Mapping[str, Mapping[str, Mapping[str, Any]]], leaks: int) -> dict[str, Any]:
    """Registration (d): no infrastructure error, no LLM error passed to the agent, >= 1 fast and >= 1 strong call and
    >= 1 switch in the gated smoke, parsed actions for all three roles in the shadow smoke, Jev answered >= 95 % of
    the decisions, cost > 0, 0 leaks."""
    chk: dict[str, Any] = {}
    chk["1_no_infra_error"] = all(not r.get("infra_error") for arm in trials.values() for r in arm.values()) and \
        all(trials.get(a) for a in ("baseline", "smoke_gated", "shadow"))
    prim = [c for c in calls if c.get("role") == "primary"]
    errs = [c for c in prim if c.get("status") != 200]
    # an error is passed to the agent unless the proxy recovered the step (escalation after a failed fast call)
    recovered = {(c["arm"], c["episode"], c["step"]) for c in prim if c.get("status") == 200}
    chk["2_no_llm_error_to_agent"] = not [c for c in errs if (c["arm"], c["episode"], c["step"]) not in recovered]
    g = sorted((c for c in prim if c.get("arm") == "smoke_gated" and c.get("status") == 200), key=lambda c: c["step"])
    tiers = [c["tier"] for c in g]
    chk["3_gated_switches"] = "fast" in tiers and "strong" in tiers and any(x != y for x, y in zip(tiers, tiers[1:]))
    sh = [c for c in calls if c.get("arm") == "shadow" and c.get("status") == 200 and c.get("key") is not None]
    roles = {c["role"] for c in sh}
    chk["4_shadow_roles"] = {"primary", "shadow_fast", "shadow_strong"} <= roles
    ds = [d for d in decisions if d.get("arm") in ("shadow", "smoke_gated")]
    chk["5_jev_answers"] = bool(ds) and sum(d.get("c") is not None for d in ds) / len(ds) >= 0.95
    chk["6_cost_positive"] = sum(float(c.get("cost_usd") or 0) for c in prim) > 0
    chk["7_no_leak"] = leaks == 0
    return {"harness": harness, "checks": chk, "ok": all(chk.values()), "tiers_gated": tiers,
            "n_calls": len(calls), "n_decisions": len(ds), "spend": spend_of(calls, decisions), "written": _now()}


def cmd_smoke(a: argparse.Namespace) -> int:
    calls, decisions = read_logs([Path(a.logs)])
    run = Path(a.run)
    trials = {}
    for arm in ("baseline", "smoke_gated", "shadow"):
        trials[arm] = mswe_trials(run / arm) if HARNESSES[a.harness]["kind"] == "mswe" else harbor_trials(run / arm)
    res = smoke_checks(a.harness, calls, decisions, trials, int(a.leaks))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"ok": res["ok"], **res["checks"], "tiers": res["tiers_gated"]}))
    return 0 if res["ok"] else 1


# ------------------------------------------------------------------------------------------------ readout
def _pct(x: Any, nd: int = 1) -> str:
    return "--" if x is None else f"{100 * float(x):.{nd}f}"


def _num(x: Any, nd: int = 1) -> str:
    return "--" if x is None else f"{float(x):.{nd}f}"


def _sgn(x: Any, nd: int = 1) -> str:
    if x is None:
        return "--"
    v = float(x)
    return f"{'+' if v > 0 else ('-' if v < 0 else '')}{abs(v):.{nd}f}"


def tex_values(summ: Mapping[str, Any]) -> dict[str, str]:
    """``e29/<harness>/<key>`` -> string (numbers only from the run records)."""
    vals: dict[str, str] = {}
    go = (summ.get("go") or {}).get("decision")
    for stage in ("full", "pilot"):
        names = list(HARNESSES) if stage == "full" else ["tb", "mswe"]
        for h in names:
            s = (summ.get(stage) or {}).get(h)
            pre = f"e29/{h}" if stage == "full" else f"e29/pilot/{h}"
            if not s or not s.get("paired"):
                default = "not run (no-go)" if (stage == "full" and go == "no-go") else "not run"
                vals[f"{pre}/status"] = (s or {}).get("status", default)
                continue
            p, g, b = s["paired"], s["arms"]["gated"], s["arms"]["baseline"]
            vals.update({
                f"{pre}/n": str(p["n"]), f"{pre}/ref_success": _pct(p["success_base"]),
                f"{pre}/success": _pct(p["success_arm"]), f"{pre}/delta": _sgn(p["delta_pt"]),
                f"{pre}/delta_lo": _sgn(p["delta_ci"][0]), f"{pre}/delta_hi": _sgn(p["delta_ci"][1]),
                f"{pre}/wins": str(p["wins"]), f"{pre}/losses": str(p["losses"]),
                f"{pre}/ni": "yes" if p["ni"] else "no", f"{pre}/ni_p": f"{p['ni_p']:.3f}",
                f"{pre}/saving": _pct(p["saving"]), f"{pre}/saving_lo": _pct(p["saving_ci"][0]),
                f"{pre}/saving_hi": _pct(p["saving_ci"][1]),
                f"{pre}/cost_ref": f"{p['cost_base']:.4f}", f"{pre}/cost": f"{p['cost_arm']:.4f}",
                f"{pre}/time_saving": _pct(p.get("time_saving")),
                f"{pre}/median_time_saving": _pct(p.get("median_time_saving")),
                f"{pre}/time_ref": _num(p.get("time_base"), 0), f"{pre}/time": _num(p.get("time_arm"), 0),
                f"{pre}/median_time_ref": _num(p.get("median_time_base"), 0),
                f"{pre}/median_time": _num(p.get("median_time_arm"), 0),
                f"{pre}/fast_share": _pct(g["fast_share"]),
                f"{pre}/segments_per_task": _num(g.get("segments_per_task")),
                f"{pre}/segment_len": _num(g.get("segment_len_mean")),
                f"{pre}/segments_capped": _pct(g.get("segments_capped_share")),
                f"{pre}/dm_latency": _num(g.get("dm_latency_per_task_s")),
                f"{pre}/waited": _num(g.get("waited_per_task_s"), 2),
                f"{pre}/decisions": _num(g.get("decisions_per_task")),
                f"{pre}/dm_cost": f"{float(g.get('dm_cost_mean') or 0):.4f}",
                f"{pre}/fail_open": str(g.get("fail_open", 0)),
                f"{pre}/cached_ref": _pct(b.get("cached_share")), f"{pre}/cached": _pct(g.get("cached_share")),
            })
            cc = s.get("paired_cc") or {}
            if cc.get("n"):
                vals.update({f"{pre}/cc_n": str(cc["n"]), f"{pre}/cc_delta": _sgn(cc["delta_pt"]),
                             f"{pre}/cc_ni_p": f"{cc['ni_p']:.3f}"})
    for h, fit in (summ.get("gates") or {}).items():
        pre = f"e29/gate/{h}"
        vals.update({f"{pre}/tau": "none" if fit.get("tau") is None else f"{fit['tau']:.3f}",
                     f"{pre}/steps": str(fit.get("n_steps")), f"{pre}/tasks": str(fit.get("n_episodes")),
                     f"{pre}/sim_share": _pct(fit.get("sim_fast_share")),
                     f"{pre}/agree_fast": _pct(fit.get("agree_fast")), f"{pre}/agree_self": _pct(fit.get("agree_self")),
                     f"{pre}/routed_agree_fast": _pct(fit.get("routed_agree_fast")),
                     f"{pre}/routed_agree_self": _pct(fit.get("routed_agree_self"))})
        d = fit.get("descriptive") or {}
        if d.get("n"):
            vals.update({f"{pre}/same_tool_fast": _pct(d["all"]["same_tool_fast"]),
                         f"{pre}/same_tool_self": _pct(d["all"]["same_tool_self"]),
                         f"{pre}/auroc": _num(d.get("auroc_c_for_fast_agreement"), 2),
                         f"{pre}/jevfast_share": _pct(d.get("fixed_tau_0p5_share")),
                         f"{pre}/jevfast_agree_fast": _pct(d["jev_says_fast"]["agree_fast"]),
                         f"{pre}/jevfast_agree_self": _pct(d["jev_says_fast"]["agree_self"])})
    g = summ.get("go") or {}
    if g:
        vals["e29/go/decision"] = g.get("decision", "--")
        for h, v in (g.get("per_harness") or {}).items():
            vals[f"e29/go/{h}/saving"] = _pct(v.get("saving"))
            if v.get("drop") is not None and v.get("drop") != 99:
                vals[f"e29/go/{h}/drop"] = str(v.get("drop"))
    sp = summ.get("spend") or {}
    if sp:
        vals["e29/spend/total"] = _num(sp.get("total"), 2)
        vals["e29/spend/cap"] = _num(TOTAL_CAP, 0)
    lk = summ.get("leaks") or {}
    if lk:
        vals["e29/leaks/max_files"] = str(max(int(v.get("max_files") or 0) for v in lk.values()))
        vals["e29/leaks/checks"] = str(sum(int(v.get("checks") or 0) for v in lk.values()))
    return vals


def write_tex(vals: Mapping[str, str], path: Path = TEX) -> None:
    lines = ["% AUTO-GENERATED by scripts/explore_e29_port.py readout -- do not edit by hand.",
             f"% generated {_now()} at git {_git()}; exploratory (E29), descriptive ({REG}); outside every Holm table",
             "\\makeatletter",
             "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    for k in sorted(vals):
        lines.append(f"\\defres{{{k}}}{{{vals[k]}}}")
    lines.append("\\makeatother")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def to_markdown(summ: Mapping[str, Any]) -> str:
    out = ["# E29: the full harness in open-source harnesses (exploratory, descriptive)", "",
           f"Registered in {REG} before any E29 code, call or episode. The decision layer is a host-side model proxy "
           "(`jevharness.adapters.route_proxy`) that holds the keys: a typed fast/strong question to Jev asked while "
           "the previous action runs, a gate fitted on shadow logs, sticky fast segments of at most 4 steps, fail-open. "
           "Cost = proxy-logged LLM + decision-model $ per task; time = agent execution seconds per task.", ""]
    for stage in ("pilot", "full"):
        st = summ.get(stage) or {}
        if not st:
            continue
        out += [f"## {'Pilot (30 tasks per harness; decided the go)' if stage == 'pilot' else 'Full run'}", "",
                "| harness | n | success base / gated % | delta pt [95 %] | W / L | NI p (3 pt) | cost $/task base / "
                "gated | saving % [95 %] | mean s base / gated | time saving % | median-time saving % | fast share % |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for h, s in st.items():
            if not s or not s.get("paired"):
                out.append(f"| {HARNESSES[h]['label']} | -- | {s.get('status', 'not run') if s else 'not run'} "
                           "| | | | | | | | | |")
                continue
            p, g = s["paired"], s["arms"]["gated"]
            out.append(f"| {HARNESSES[h]['label']} | {p['n']} | {_pct(p['success_base'])} / {_pct(p['success_arm'])} | "
                       f"{_sgn(p['delta_pt'])} [{_sgn(p['delta_ci'][0])}, {_sgn(p['delta_ci'][1])}] | "
                       f"{p['wins']} / {p['losses']} | {p['ni_p']:.3f} ({'shown' if p['ni'] else 'not shown'}) | "
                       f"{p['cost_base']:.4f} / {p['cost_arm']:.4f} | {_pct(p['saving'])} [{_pct(p['saving_ci'][0])}, "
                       f"{_pct(p['saving_ci'][1])}] | {_num(p.get('time_base'), 0)} / {_num(p.get('time_arm'), 0)} | "
                       f"{_pct(p.get('time_saving'))} | {_pct(p.get('median_time_saving'))} | {_pct(g['fast_share'])} |")
        out.append("")
        for h, s in st.items():
            if not s or not s.get("paired"):
                continue
            g, b = s["arms"]["gated"], s["arms"]["baseline"]
            cc = s.get("paired_cc") or {}
            out.append(f"- {HARNESSES[h]['label']}: gated arm {g['requests_mean']:.1f} requests per task "
                       f"(baseline {b['requests_mean']:.1f}), {_num(g.get('segments_per_task'))} fast segments per task "
                       f"of mean length {_num(g.get('segment_len_mean'), 2)} ({_pct(g.get('segments_capped_share'))} % "
                       f"ended by the cap k = 4), {_num(g.get('switches_per_task'))} switches per task; decision model "
                       f"{_num(g.get('decisions_per_task'))} calls, ${float(g.get('dm_cost_mean') or 0):.4f} and "
                       f"{_num(g.get('dm_latency_per_task_s'))} s per task, of which requests waited "
                       f"{_num(g.get('waited_per_task_s'), 2)} s; fail-open steps {g.get('fail_open')}, decision errors "
                       f"{g.get('decision_errors')}; cached input share {_pct(b.get('cached_share'))} % (baseline) vs "
                       f"{_pct(g.get('cached_share'))} % (gated); infrastructure errors {b['infra_errors']} / "
                       f"{g['infra_errors']}; complete case n {cc.get('n')}, delta {_sgn(cc.get('delta_pt'))}, NI p "
                       f"{_num(cc.get('ni_p'), 3)}. Missing: {json.dumps({k: len(v) for k, v in s['missing'].items()})}.")
        out.append("")
    if summ.get("gates"):
        out += ["## Fitted gates (calibration, shadow mode)", "",
                "| harness | tau | steps / tasks | agree fast / self % | routed: agree fast / self % | simulated fast share % "
                "| dC $ |", "|---|---|---|---|---|---|---|"]
        for h, f in summ["gates"].items():
            out.append(f"| {h} | {'none' if f.get('tau') is None else round(f['tau'], 4)} | {f.get('n_steps')} / "
                       f"{f.get('n_episodes')} | {_pct(f.get('agree_fast'))} / {_pct(f.get('agree_self'))} | "
                       f"{_pct(f.get('routed_agree_fast'))} / {_pct(f.get('routed_agree_self'))} | "
                       f"{_pct(f.get('sim_fast_share'))} | {_num(f.get('dC_usd'), 5)} |")
        out.append("")
        out += ["Post hoc, descriptive (from the same calibration records; not part of the registered fit): agreement "
                "with the strong LLM's action of the fast LLM / of a second strong sample.", "",
                "| harness | steps | exact action % (fast / self) | same tool % (fast / self) | Jev says fast: n, exact % "
                "(fast / self) | Jev says strong: n, exact % (fast / self) | AUROC of c for fast = strong | share c >= 0.5 % |",
                "|---|---|---|---|---|---|---|---|"]
        for h, f in summ["gates"].items():
            d = f.get("descriptive") or {}
            if not d.get("n"):
                continue
            a, jf, js = d["all"], d["jev_says_fast"], d["jev_says_strong"]
            out.append(f"| {h} | {d['n']} | {_pct(a['agree_fast'])} / {_pct(a['agree_self'])} | "
                       f"{_pct(a['same_tool_fast'])} / {_pct(a['same_tool_self'])} | {jf['n']}, {_pct(jf['agree_fast'])} / "
                       f"{_pct(jf['agree_self'])} | {js['n']}, {_pct(js['agree_fast'])} / {_pct(js['agree_self'])} | "
                       f"{_num(d.get('auroc_c_for_fast_agreement'), 2)} | {_pct(d.get('fixed_tau_0p5_share'))} |")
            out.append("")
            out.append(f"- {h} by the strong action's tool: " + "; ".join(
                f"{tl} n {v['n']}: {_pct(v['agree_fast'])} / {_pct(v['agree_self'])}" for tl, v in d["by_tool"].items()))
        out.append("")
    if summ.get("go"):
        out += ["## Go decision (registered rule, pilot only)", "", "```", json.dumps(summ["go"], indent=1), "```", ""]
    for k in ("spend", "leaks", "hosts", "notes"):
        if summ.get(k):
            out += [f"## {k.capitalize()}", "", "```", json.dumps(summ[k], indent=1, default=str), "```", ""]
    return "\n".join(out)


def _git() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def cmd_readout(a: argparse.Namespace) -> int:
    box = Path(a.box)
    summ: dict[str, Any] = {"registered": REG, "code": _git(), "written": _now(), "pilot": {}, "full": {}, "gates": {}}
    for stage in ("pilot", "full"):
        for h in HARNESSES:
            f = box / "summaries" / f"{stage}_{h}.json"
            if f.exists():
                summ[stage][h] = json.loads(f.read_text())
    for h in HARNESSES:
        f = box / "gates" / f"gate_{h}.json"
        if f.exists():
            summ["gates"][h] = {k: v for k, v in json.loads(f.read_text()).items() if k != "curve"}
            cal = box / h / "proxy" / "cal"
            if cal.is_dir():        # post hoc, descriptive: from the calibration proxy log (the same records as the fit)
                calls, decs = read_logs([cal])
                cut = _ts(summ["gates"][h].get("written"))
                if cut is not None:   # records the proxy's shutdown wrote after the fit had read the log are left out
                    late = [r for r in calls + decs if (_ts(r.get("ts")) or cut) > cut]
                    calls = [r for r in calls if (_ts(r.get("ts")) or cut) <= cut]
                    decs = [r for r in decs if (_ts(r.get("ts")) or cut) <= cut]
                    summ["gates"][h]["records_after_fit"] = len(late)
                summ["gates"][h]["descriptive"] = calib_descriptives(calls, decs)
    for name in ("go", "hosts", "notes"):
        f = box / f"{name}.json"
        if f.exists():
            summ[name] = json.loads(f.read_text())
    # per host: the logged spend (proxy logs of every stage) and the OpenRouter key's usage, and every leak check
    spend: dict[str, Any] = {}
    leaks: dict[str, Any] = {}
    for h in HARNESSES:
        st = box / h / "status"
        f = st / "e29_spend.json"
        if f.exists():
            spend[h] = json.loads(f.read_text())
        lk = jsonl(st / "e29_leaks.jsonl")
        if lk:
            leaks[h] = {"checks": len(lk), "max_files": max(int(x.get("files") or 0) for x in lk),
                        "stages": [x.get("stage") for x in lk]}
    local = box / "local_smoke_spend.json"
    if local.exists():
        spend["local_smoke"] = json.loads(local.read_text())
    if spend:
        tot = sum(float(((v.get("logged") or v) or {}).get("total") or 0.0) for v in spend.values())
        spend["total"] = round(tot, 4)
        spend["cap"] = TOTAL_CAP
        summ["spend"] = spend
    if leaks:
        summ["leaks"] = leaks
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(summ, default=str))
    for stage in ("pilot", "full"):
        for s in slim[stage].values():
            s.pop("per_task", None)
    (OUT_DIR / "summary.json").write_text(json.dumps(slim, indent=1) + "\n")
    (OUT_DIR / "summary.md").write_text(to_markdown(summ) + "\n")
    write_tex(tex_values(summ), Path(a.tex) if a.tex else TEX)
    print(f"wrote {OUT_DIR / 'summary.md'}, {OUT_DIR / 'summary.json'}, {a.tex or TEX}")
    return 0


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--registry", required=True)
    p = sub.add_parser("spend")
    p.add_argument("dirs", nargs="+")
    p = sub.add_parser("fit")
    p.add_argument("--harness", required=True, choices=sorted(HARNESSES))
    p.add_argument("--logs", required=True)
    p.add_argument("--arm", default="shadow")
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--harness", required=True, choices=sorted(HARNESSES))
    p.add_argument("--logs", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--leaks", default="0")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--harness", required=True, choices=sorted(HARNESSES))
    p.add_argument("--stage", required=True, choices=("pilot", "full"))
    p.add_argument("--run", required=True)
    p.add_argument("--logs", required=True, nargs="+")
    p.add_argument("--tasks", default=None)
    p.add_argument("--rerun-run", default=None)
    p.add_argument("--rerun-logs", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("go")
    p.add_argument("--tb", default=None)
    p.add_argument("--mswe", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("readout")
    p.add_argument("--box", default=str(BOX))
    p.add_argument("--tex", default=None)
    a = ap.parse_args(argv)
    return {"select": cmd_select, "spend": cmd_spend, "fit": cmd_fit, "smoke": cmd_smoke, "summary": cmd_summary,
            "go": cmd_go, "readout": cmd_readout}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
