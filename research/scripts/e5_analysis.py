#!/usr/bin/env python3
r"""Exploratory row E5, small decision models: the analysis (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06
"Exploratory row E5" (h)). Descriptive, outside every Holm table, reported whatever the outcome. Offline: no GPU, no
API; every number is computed from run files.

    python scripts/e5_analysis.py [--run runs/e5_small] [--live runs/e5_small_live] [--fit]
        -> results/explore/e5_small/{summary.md, summary.json}, paper/generated/e5_small.tex (\defres{e5/...})

Reads the GPU run (``aws s3 sync s3://.../e5_small runs/e5_small``): ``eval/<slot>_v1.json`` (eval_decisions, raw),
``s23/answers_e5_<slot>.jsonl`` (corpus C), ``settings/<slot>.json`` / ``smoke/<slot>.json`` (training setting, minutes,
peak memory), ``latency/bench/bench_<label>_new_c{1,4}.json`` (the final harness's request stream),
``latency/<label>_synthetic.json`` (measure_latency.py), ``latency/smi_<label>_c*.csv`` (GPU memory while serving),
``latency/models_<label>.json`` (cache facts), ``cal/answers_<slot>.jsonl`` and ``cal/check_4b.json`` (entry (g)).
The 4B references are committed rows of the same pipeline: SFT v1 (the recipe at 4B) and the frozen JevAgentRL-4B.

``--fit`` runs entry (g)'s offline threshold fit for every model with calibration-state answers
(``scripts/e5_cal_replay.py derive`` + ``fit`` -> ``results/explore/e5_small/cal/``). The live-row model is chosen by
the registered rule (:func:`live_choice`) from the offline records only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench import stage2_corpus as C  # noqa: E402

import ablation_score as AS  # noqa: E402
import e5_box as E  # noqa: E402

OUT = ROOT / "results" / "explore" / "e5_small"
TEX = ROOT / "paper" / "generated" / "e5_small.tex"
S23 = ROOT / "results" / "stage2" / "s23"
LABEL = ("exploratory (E5), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-06 'Exploratory row E5'; outside every "
         "Holm table")
SHORT = {"e5_q17": "q17", "e5_q06": "q06", "e5_mc1": "mc1", "e5_gr1": "gr1"}
# the 4B references: committed rows of the same pipeline (v1 dev: round-5 eval_decisions; C: Stage-2 replay answers)
REFS = {"sftv1": {"label": "SFT v1 (4B, the same recipe)", "v1": "results/gpu_round5/eval/eval_sft_v1_v1.json",
                  "c": "results/stage2/s23/answers_sft_v1.jsonl"},
        "rl": {"label": "JevAgentRL-4B (frozen)", "v1": "results/gpu_round5/eval/eval_grpo_i15_v1.json",
               "c": "results/stage2/s23/answers_rl.jsonl"}}
KEEP_LO = -0.010          # plan section 8: keeps the 4B's accuracy if the interval's lower end is >= -1.0 point
LIVE_MIN_C = 0.95         # entry (g): the live row needs a model with >= 95.0 % C accuracy
LIVE_TIE = 0.010          # entry (g): within 1.0 point -> the lower c = 1 p50
N_BOOT = 2000
N_BOOT_LIVE = 10000
LIVE_ARMS = ("llm_only", "harness_final", "e5_final")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _pct(xs: Sequence[float], q: float) -> float | None:
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _jl(p: Path) -> list[dict[str, Any]]:
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


# ------------------------------------------------------------------ corpus C
def c_rows(occ: Sequence[Mapping[str, Any]], eligible: Sequence[str], ans: Mapping[str, Mapping[str, Any]]
           ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """C accuracy (task means) over all states, after an off-plan step and on the ask-vs-act stratum (oracle answer
    ask_user), C ECE (with the model's temperatures) and C-replay time per question; per-task accuracies per stratum."""
    m, acc_all = AS.c_metrics(occ, eligible, ans)
    ask = C.task_accuracy([o for o in occ if o.get("gold") == "ask_user"], ans, eligible)
    off = C.task_accuracy([o for o in occ if o.get("stratum") == "after_off_plan"], ans, eligible)
    m.update({"c_acc_ask": float(np.mean(list(ask.values()))) if ask else float("nan"), "c_tasks_ask": len(ask)})
    lat = [float(a["latency_ms"]) for a in ans.values() if a.get("latency_ms") is not None and not a.get("error")]
    m["c_replay_ms_p50"] = _pct(lat, 0.5)
    return m, {"all": acc_all, "offplan": off, "ask": ask}


def reading(ci: Mapping[str, float]) -> str:
    """Plan section 8: keeps the 4B's accuracy if the lower end >= -1.0 point; loses accuracy if the upper end < 0.
    When both clauses hold (a loss smaller than the 1-point margin whose interval excludes 0), both are reported."""
    keeps, loses = ci["lo"] >= KEEP_LO, ci["hi"] < 0
    if keeps and loses:
        return "keeps the 4B's accuracy within 1 point (small loss: interval below 0)"
    if keeps:
        return "keeps the 4B's accuracy"
    if loses:
        return "loses accuracy"
    return "inconclusive"


# ------------------------------------------------------------------ latency
def bench_stats(p: Path) -> dict[str, Any] | None:
    if not p.exists():   # the box's compact copy (scripts/train/e5_box.py compact-bench) when the raw file is not synced
        p = p.parent.parent / "bench_compact" / p.name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rows, steps = d.get("requests") or [], d.get("steps") or []
    out: dict[str, Any] = {"concurrency": d.get("concurrency"), "episodes": d.get("episodes"), "requests": len(rows),
                           "p50_ms": _pct([r["ms"] for r in rows], 0.5), "p90_ms": _pct([r["ms"] for r in rows], 0.9),
                           "mean_ms": float(np.mean([r["ms"] for r in rows])) if rows else None,
                           "s1_ms_per_step_p50": _pct([s["s1_ms"] for s in steps], 0.5),
                           "s1_ms_per_step_p90": _pct([s["s1_ms"] for s in steps], 0.9),
                           "s1_ms_per_step_mean": float(np.mean([s["s1_ms"] for s in steps])) if steps else None,
                           "span_s": d.get("span_s"), "by_dp": {}}
    for dp in sorted({r["dp"] for r in rows}):
        ms = [r["ms"] for r in rows if r["dp"] == dp]
        out["by_dp"][dp] = {"requests": len(ms), "p50_ms": _pct(ms, 0.5), "p90_ms": _pct(ms, 0.9)}
    return out


def synthetic_stats(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return {k: {kk: v.get(kk) for kk in ("p50_ms", "p90_ms", "p95_ms", "req_per_s")} for k, v in d["results"].items()}


def latency(run: Path, labels: Sequence[str]) -> dict[str, Any]:
    lat = run / "latency"
    res: dict[str, Any] = {}
    for lab in labels:
        r: dict[str, Any] = {"bench": {f"c{c}": bench_stats(lat / "bench" / f"bench_{lab}_new_c{c}.json") for c in (1, 4)},
                             "synthetic": synthetic_stats(lat / f"{lab}_synthetic.json")}
        peaks = [E.peak_mib(lat / f"smi_{lab}_c{c}.csv") for c in (1, 4)]
        peaks = [x for x in peaks if x is not None]
        r["serve_peak_mib"] = max(peaks) if peaks else None
        mp = lat / f"models_{lab}.json"
        if mp.exists():
            b = (json.loads(mp.read_text()).get("data") or [{}])[0].get("backend") or {}
            r["cache"] = {k: b.get(k) for k in ("croppable", "bytes_per_token", "cache_class", "dtype")}
        res[lab] = r
    end = bench_stats(lat / "bench" / "bench_4b_end_new_c1.json")
    if end:
        res["4b_end"] = {"bench": {"c1": end}}
    base = ((res.get("4b") or {}).get("bench") or {})
    for lab, r in res.items():
        if lab in ("4b", "4b_end"):
            continue
        sp: dict[str, Any] = {}
        for c in ("c1", "c4"):
            a, b = (base.get(c) or {}), (r["bench"].get(c) or {})
            for q in ("p50_ms", "p90_ms"):
                if a.get(q) and b.get(q):
                    sp[f"{c}_{q[:3]}"] = a[q] / b[q]
            if a.get("s1_ms_per_step_p50") and b.get("s1_ms_per_step_p50"):
                sp[f"{c}_s1_step_p50"] = a["s1_ms_per_step_p50"] / b["s1_ms_per_step_p50"]
        r["speedup_vs_4b"] = sp
    return res


# ------------------------------------------------------------------ the live-row model (entry (g))
def live_choice(models: Mapping[str, Mapping[str, Any]], lat: Mapping[str, Any]) -> dict[str, Any]:
    """Entry (g): the highest C accuracy among the models with >= 95.0 %; among those within 1.0 point of it, the
    one with the lowest c = 1 p50 per request on the final harness's request stream."""
    elig = {s: m["c_acc"] for s, m in models.items() if m.get("c_acc") is not None and m["c_acc"] >= LIVE_MIN_C}
    if not elig:
        return {"run": False, "slot": None, "eligible": {}, "rule": "no model reaches 95.0 % C accuracy"}
    best = max(elig.values())
    near = [s for s, v in elig.items() if best - v <= LIVE_TIE + 1e-12]

    def p50(s: str) -> float:
        v = (((lat.get(s) or {}).get("bench") or {}).get("c1") or {}).get("p50_ms")
        return float(v) if v is not None else float("inf")

    pick = min(near, key=lambda s: (p50(s), -elig[s], s)) if len(near) > 1 else near[0]
    return {"run": True, "slot": pick, "eligible": elig, "within_1pt": near,
            "c1_p50": {s: (None if p50(s) == float("inf") else p50(s)) for s in near},
            "rule": "highest C accuracy >= 95.0 %; within 1.0 point the lower c = 1 p50 (Deviations 2026-10-06 (g))"}


# ------------------------------------------------------------------ calibration fits (entry (g))
def fit_all(run: Path, slots: Sequence[str], out: Path = OUT / "cal", work: Path = ROOT / "runs" / "e5_small_cal"
            ) -> dict[str, Any]:
    import e5_cal_replay as CR

    res: dict[str, Any] = {}
    src = ROOT / "runs" / "e2a_jevgym_shadow_cal_rl"
    for s in list(slots) + ["4b"]:
        ans = run / "cal" / f"answers_{s}.jsonl"
        if not ans.exists():
            continue
        name = f"e5cal_{s}"
        d = CR.derive(src, "rl_shadow_cal", ans, work, name, model=s)
        res[s] = {"derive": d, "fit": CR.fit(work, name, out)}
    return res


def fits(slots: Sequence[str], d: Path = OUT / "cal") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in list(slots) + ["4b"]:
        r = {}
        for dp in ("tool_select", "tool_params"):
            p = d / f"tauhat_e5cal_{s}_{dp}.json"
            if p.exists():
                f = json.loads(p.read_text())
                t = f.get("tau_hat")
                r[dp] = {"tau_hat": t, "n_paired": f.get("n_paired"), "tau_rule": (f.get("tau_rule") or {}).get("rule"),
                         "accepts": t is not None and isinstance(t, (int, float)) and math.isfinite(t) and t <= 1.0}
        if r:
            out[s] = r
    return out


FROZEN_TAU = {"tool_select": 0.4999999999978404, "tool_params": 0.9999998807907104}   # configs/harness/jevagent_final.yaml
CAL_SRC = ROOT / "runs" / "e2a_jevgym_shadow_cal_rl"
CAL_WORK = ROOT / "runs" / "e5_small_cal"


def cal_coverage(slots: Sequence[str], fits_dir: Path = OUT / "cal", work: Path = CAL_WORK,
                 src: Path = CAL_SRC) -> dict[str, Any]:
    """On the calibration states of entry (g) (the frozen 4B's own calibration arm, Flash, seeds 5000-5149): the share
    of System-1 decisions the cascade would take at the fitted thresholds (confidence >= tau and not OTHER), System-1's
    accuracy on them, the LLM's accuracy on the same decisions, and the cascade's accuracy over all paired decisions
    (System-1 where taken, the LLM's logged decision elsewhere). The frozen 4B at its frozen thresholds on its logged
    answers (and on its replayed answers when the check replay exists). Descriptive; nothing is selected on it."""
    from jevharness.bench.threshold_theory import decision_pairs, load_trajectories, paired

    rows: dict[str, Any] = {}
    cands = [(s, work / f"e5cal_{s}.jsonl", None) for s in slots]
    cands += [("4b_logged", src / "rl_shadow_cal.jsonl", FROZEN_TAU), ("4b", work / "e5cal_4b.jsonl", FROZEN_TAU)]
    for name, path, taus in cands:
        if not path.exists():
            continue
        trajs = list(load_trajectories(path))
        r: dict[str, Any] = {}
        for dp in ("tool_select", "tool_params"):
            if taus is None:
                fp = fits_dir / f"tauhat_e5cal_{name}_{dp}.json"
                if not fp.exists():
                    continue
                t = json.loads(fp.read_text()).get("tau_hat")
                tau = float(t) if t is not None and math.isfinite(float(t)) and float(t) <= 1.0 else float("inf")
            else:
                tau = taus[dp]
            ps = paired(decision_pairs(trajs, dp))
            if not ps:
                continue
            acc = [x for x in ps if x["c"] is not None and x["c"] >= tau and x["acceptable"]]
            n = len(ps)
            casc = (sum(float(x["s1_ok"]) for x in acc) + sum(float(x["s2_ok"]) for x in ps if x not in acc)) / n
            r[dp] = {"tau": tau, "n_paired": n, "accepted": len(acc), "coverage": len(acc) / n,
                     "s1_acc_accepted": (sum(float(x["s1_ok"]) for x in acc) / len(acc)) if acc else None,
                     "s2_acc_accepted": (sum(float(x["s2_ok"]) for x in acc) / len(acc)) if acc else None,
                     "s1_acc_all": sum(float(x["s1_ok"]) for x in ps) / n,
                     "s2_acc_all": sum(float(x["s2_ok"]) for x in ps) / n, "cascade_acc": casc}
        if r:
            rows[name] = r
    return rows


# ------------------------------------------------------------------ live row
SLOW_S = 5.0


def _lat_stats(xs: Sequence[float]) -> dict[str, Any]:
    tot = sum(xs)
    slow = [x for x in xs if x > SLOW_S]
    return {"n": len(xs), "p50": _pct(xs, 0.5), "p90": _pct(xs, 0.9), "mean": (tot / len(xs)) if xs else None,
            "total_s": tot, "n_slow5": len(slow), "slow_share": (sum(slow) / tot) if tot else None}


def llm_latency(recs: Mapping[str, Mapping[str, Mapping[str, Any]]], keys: Sequence[str]) -> dict[str, Any]:
    """System-2 (LLM) call latency in seconds on the tasks complete in every arm, per arm: by decision point and by the
    provider that served the call (OpenRouter reports it; the direct API is one provider), median / p90 / mean, the
    calls above 5 s and their share of the arm's LLM time (Deviations 2026-10-06 ~19:40 UTC, route diagnostic)."""
    out: dict[str, Any] = {}
    for a, by_task in recs.items():
        calls = [c for t in keys for c in (by_task[t].get("calls") or []) if c.get("component") == "s2"]
        by_dp: dict[str, list[float]] = {}
        by_prov: dict[str, list[float]] = {}
        for c in calls:
            x = float(c.get("latency_ms") or 0.0) / 1000
            by_dp.setdefault(str(c.get("decision_point")), []).append(x)
            prov = str((c.get("meta") or {}).get("served_provider") or c.get("backend") or "?")
            by_prov.setdefault(prov, []).append(x)
        out[a] = {"all": _lat_stats([float(c.get("latency_ms") or 0.0) / 1000 for c in calls]),
                  "by_dp": {k: _lat_stats(v) for k, v in sorted(by_dp.items())},
                  "by_provider": {k: _lat_stats(v) for k, v in sorted(by_prov.items())}}
    return out


def live(run: Path, host_usd_per_h: float | None) -> dict[str, Any] | None:
    from harness_final import _episode_live, load_latest

    if not all((run / f"{a}.jsonl").exists() for a in LIVE_ARMS):
        return None
    recs = {a: load_latest(run / f"{a}.jsonl") for a in LIVE_ARMS}
    ok = {a: {t: r for t, r in recs[a].items() if not r.get("error")} for a in LIVE_ARMS}
    keys = sorted(set.intersection(*(set(ok[a]) for a in LIVE_ARMS)))
    n = len(keys)
    if n == 0:
        return {"n": 0}
    ep = {a: [_episode_live(ok[a][t]) for t in keys] for a in LIVE_ARMS}
    v = {a: {k: np.array([e[k] for e in ep[a]], dtype=float) for k in ("success", "cost", "wall_s", "model_s", "s1_s",
                                                                       "s2_s", "steps", "s1_requests")} for a in LIVE_ARMS}
    idx = np.random.default_rng(0).integers(0, n, size=(N_BOOT_LIVE, n))
    q = {a: (0.0 if a == "llm_only" or not host_usd_per_h else host_usd_per_h * float(v[a]["wall_s"].mean()) / 3600)
         for a in LIVE_ARMS}

    def ci(x: np.ndarray) -> list[float]:
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    llm = {}
    if (run / "experiment.yaml").exists():
        import yaml

        llm = yaml.safe_load((run / "experiment.yaml").read_text()).get("llm") or {}
    res: dict[str, Any] = {"n_pairs": n, "errored_latest": {a: len(recs[a]) - len(ok[a]) for a in LIVE_ARMS},
                           "system2": {k: llm.get(k) for k in ("provider", "model", "temperature", "reasoning_effort")},
                           "dollar_scope": ("within-E5 only: System-2 through OpenRouter, which reports about half the "
                                            "direct API's input tokens (Deviations 2026-10-06 ~18:00 UTC); not "
                                            "comparable with harness_final's direct-API rows")
                           if llm.get("provider") == "openrouter" else "direct API",
                           "host_usd_per_h": host_usd_per_h, "q_usd_per_episode": q, "arms": {}, "contrasts": {}}
    for a in LIVE_ARMS:
        res["arms"][a] = {"success": float(v[a]["success"].mean()), "api_cost": float(v[a]["cost"].mean()),
                          "cost_with_q": float(v[a]["cost"].mean() + q[a]),
                          "wall_s": float(v[a]["wall_s"].mean()), "wall_s_median": float(np.median(v[a]["wall_s"])),
                          "s1_s": float(v[a]["s1_s"].mean()), "s2_s": float(v[a]["s2_s"].mean()),
                          "model_s": float(v[a]["model_s"].mean()), "steps": float(v[a]["steps"].mean())}
        tot = [_episode_live(ok[a][t]) for t in keys]
        td, tb = sum(e["tool_decisions"] for e in tot), sum(e["tool_by_s1"] for e in tot)
        ad, ab = sum(e["arg_decisions"] for e in tot), sum(e["arg_by_s1"] for e in tot)
        res["arms"][a].update({"tool_coverage": tb / td if td else None, "arg_coverage": ab / ad if ad else None})
    for a, b in (("e5_final", "llm_only"), ("harness_final", "llm_only"), ("e5_final", "harness_final")):
        ds = (v[a]["success"] - v[b]["success"]) * 100
        ca, cb = v[a]["cost"] + q[a], v[b]["cost"] + q[b]
        dw = v[a]["wall_s"] - v[b]["wall_s"]
        res["contrasts"][f"{a}_vs_{b}"] = {
            "success_delta_pt": float(ds.mean()), "success_delta_ci": ci(ds[idx].mean(1)),
            "discordant": {"a_only": int(((v[a]["success"] == 1) & (v[b]["success"] == 0)).sum()),
                           "b_only": int(((v[a]["success"] == 0) & (v[b]["success"] == 1)).sum())},
            "cost_saving_frac": float(1 - ca.mean() / cb.mean()) if cb.mean() else None,
            "cost_saving_ci": ci(1 - ca[idx].mean(1) / cb[idx].mean(1)) if cb.mean() else None,
            "cost_saving_frac_no_q": float(1 - v[a]["cost"].mean() / v[b]["cost"].mean()) if v[b]["cost"].mean() else None,
            "wall_delta_s": float(dw.mean()), "wall_delta_ci": ci(dw[idx].mean(1)),
            "wall_saving_frac": float(1 - v[a]["wall_s"].mean() / v[b]["wall_s"].mean())}
    res["tasks"] = keys
    res["llm_latency"] = llm_latency({a: ok[a] for a in LIVE_ARMS}, keys)
    return res


# ------------------------------------------------------------------ assemble
PART_FILES = ("eval/{s}_v1.json", "s23/answers_e5_{s}.jsonl", "s23/replay_e5_{s}.json", "settings/{s}.json",
              "settings/{s}_eval_batch.txt", "smoke/{s}.json", "smoke/{s}_readout.json", "done/{s}", "mem/{s}.txt",
              "cal/{s}_calibration.json", "cal/answers_{s}.jsonl", "models/{s}.json")


def _stage_hours(run: Path) -> float:
    """GPU hours of one host = the summed durations of its finished stages (stage_times.jsonl). A synced
    cur_stage.json is not read: S3 sync never deletes it, so it can outlive its stage."""
    p = run / "stage_times.jsonl"
    if not p.exists():
        return 0.0
    return sum(float(json.loads(x)["min"]) for x in p.read_text().splitlines() if x.strip()) / 60


def _merge_parts(run: Path, parts: Sequence[Path]) -> None:
    """Copy each part's per-model files into the main run. A file the main run already holds must be identical (a
    model's records come from one host), except the main host's own download record (models/<slot>.json)."""
    import shutil

    for part in parts:
        slots = [s for s in E.MODELS if (part / "done" / s).exists() or (part / "eval" / f"{s}_v1.json").exists()]
        for s in slots:
            for f in PART_FILES:
                src, dst = part / f.format(s=s), run / f.format(s=s)
                if not src.exists():
                    continue
                if dst.exists():
                    if dst.read_bytes() != src.read_bytes() and not f.startswith("models/"):
                        raise SystemExit(f"{dst} differs from {src}: a model's records must come from one host")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def analyze(run: Path, live_run: Path | None = None, corpus: Path = S23, do_fit: bool = False,
            parts: Sequence[Path] = (), live_direct: Path | None = None) -> dict[str, Any]:
    """``run``: the main host's run (latency, the 4B check); ``parts``: further hosts' runs (Deviations 2026-10-06,
    hosts), whose per-model files are copied into ``run`` first (a model's files exist on exactly one host)."""
    _merge_parts(run, parts)
    occ, eligible = AS.load_corpus(corpus)
    slots = [s for s in E.MODELS if (run / "eval" / f"{s}_v1.json").exists()
             or (run / "s23" / f"answers_e5_{s}.jsonl").exists()]
    host = json.loads((run / "host.json").read_text()) if (run / "host.json").exists() else {}
    models: dict[str, Any] = {}
    per_task: dict[str, dict[str, dict[str, float]]] = {}
    for s in slots:
        m: dict[str, Any] = {"slot": s, **{k: E.MODELS[s][k] for k in ("repo", "revision", "params_b", "nonemb_b",
                                                                       "label")}}
        p = run / "settings" / f"{s}.json"
        if p.exists():
            st = json.loads(p.read_text())
            m.update({k: st.get(k) for k in ("setting", "sft_min", "peak_mem_mib", "train_mix_ok", "instance_type",
                                             "grad_accum")})
            m["train_peak_mib"] = m.pop("peak_mem_mib")
        p = run / "smoke" / f"{s}.json"
        if p.exists():
            sm = json.loads(p.read_text())
            m["smoke"] = {k: sm.get(k) for k in ("setting", "tried", "peak_mem_mib", "failed")}
        p = run / "eval" / f"{s}_v1.json"
        if p.exists():
            m.update(AS.v1_metrics(p))
        p = run / "s23" / f"answers_e5_{s}.jsonl"
        if p.exists():
            cm, pt = c_rows(occ, eligible, AS.load_answers(p))
            m.update(cm)
            per_task[s] = pt
        m["done"] = (run / "done" / s).exists()
        models[s] = m
    refs: dict[str, Any] = {}
    for r, spec in REFS.items():
        row: dict[str, Any] = {"label": spec["label"]}
        if (ROOT / spec["v1"]).exists():
            row.update(AS.v1_metrics(ROOT / spec["v1"]))
        if (ROOT / spec["c"]).exists():
            cm, pt = c_rows(occ, eligible, AS.load_answers(ROOT / spec["c"]))
            row.update(cm)
            per_task[r] = pt
        refs[r] = row
    contrasts: dict[str, Any] = {}
    for s in models:
        for r in refs:
            if s not in per_task or r not in per_task:
                continue
            c: dict[str, Any] = {"model": s, "reference": r}
            for stratum in ("all", "offplan", "ask"):
                b = AS.paired_boot(per_task[s][stratum], per_task[r][stratum], N_BOOT, 0)
                b["reading"] = reading(b)
                c[stratum] = b
            for k in ("v1_acc", "v1_recovery", "v1_knockout", "v1_ece"):
                if models[s].get(k) is not None and refs[r].get(k) is not None:
                    c[f"{k}_diff"] = models[s][k] - refs[r][k]
            contrasts[f"{SHORT[s]}_vs_{r}"] = c
    lat = latency(run, ["4b"] + slots)
    if do_fit:
        fit_all(run, slots)
    chk = run / "cal" / "check_4b.json"
    res = {"label": LABEL, "generated": _now(), "run": str(run), "host": {k: host.get(k) for k in (
        "instance_id", "instance_type", "gpu_name", "gpu_class", "usd_per_h")},
           "gpu_hours": sum(_stage_hours(r) for r in [run, *parts]) or None,
           "gpu_hours_by_host": {str(r): round(_stage_hours(r), 3) for r in [run, *parts]
                                 if (r / "stage_times.jsonl").exists()},
           "swap": json.loads((run / "swap.json").read_text()) if (run / "swap.json").exists() else None,
           "models": models, "references": refs, "contrasts": contrasts, "latency": lat,
           "live_choice": live_choice(models, lat), "cal_fits": fits(slots), "cal_coverage": cal_coverage(slots),
           "cal_check_4b": json.loads(chk.read_text()) if chk.exists() else None,
           "not_run": [s for s in E.slots(run) if s not in models or not models[s].get("done")]}
    if live_run is not None and live_run.exists():
        res["live"] = live(live_run, host.get("usd_per_h"))
    if live_direct is not None and live_direct.exists():   # the route diagnostic (Deviations 2026-10-06 ~19:40 UTC)
        res["live_direct"] = live(live_direct, host.get("usd_per_h"))
    return res


# ------------------------------------------------------------------ live configuration (before the first live episode)
LIVE_CFGS = (ROOT / "configs" / "experiments" / "e5_small_live.yaml", ROOT / "configs" / "dev" / "e5_small_smoke.yaml")
E5_PROFILE = ROOT / "configs" / "harness" / "explore_e5_final.yaml"


def prepare_live(d: Mapping[str, Any], fits_dir: Path = OUT / "cal", profile: Path = E5_PROFILE,
                 cfgs: Sequence[Path] = LIVE_CFGS) -> dict[str, Any]:
    """Entry (g), applied from the offline records only: the chosen model's served name into the live and smoke
    configs, and its offline-fitted thresholds into ``backends.e5_small`` of the E5 profile (``fill_backend_tau.fill``;
    a fit that accepts no level -> always defer; nothing delegable -> ``skip``, and the live row is not run). Refuses
    when no model qualifies or when the frozen 4B's replay check did not pass (then the registered calibration command
    on seeds 5000-5149 applies instead)."""
    import hashlib

    import yaml

    from fill_backend_tau import fill
    from jevharness.harness import profile as HP

    lc = d["live_choice"]
    if not lc.get("run"):
        raise SystemExit(f"no live row: {lc.get('rule')}")
    chk = d.get("cal_check_4b") or {}
    if not chk.get("ok"):
        raise SystemExit("the frozen 4B's calibration-state replay check did not pass (or is missing): the registered "
                         "calibration command on seeds 5000-5149 applies (Deviations 2026-10-06 (g))")
    slot = lc["slot"]
    fits, parents = {}, {}
    for dp in ("tool_select", "tool_params"):
        fp = fits_dir / f"tauhat_e5cal_{slot}_{dp}.json"
        if not fp.exists():
            raise SystemExit(f"missing fit {fp} (run with --fit first)")
        fits[dp] = json.loads(fp.read_text())
        parents[dp] = {"path": str(fp.relative_to(ROOT)) if fp.is_relative_to(ROOT) else str(fp),
                       "sha256": hashlib.sha256(fp.read_bytes()).hexdigest()}
    prof = HP.load_profile(profile)
    sec = prof["system2"]["gemini-3.5-flash"]
    sec["backends"]["e5_small"] = {"tool_select": {"calibration": "pending"}, "tool_params": {"calibration": "pending"}}
    out = fill(prof, "e5_small", fits)
    blk = out["system2"]["gemini-3.5-flash"]["backends"]["e5_small"]
    notes = blk.pop("notes")
    blk["fitted_from"] = {**parents, "fit_arm": f"e5cal_{slot}", "model": f"e5-{slot}",
                          "rule": "PREREG_A7_STAGE2 Deviations 2026-10-06 (g): offline fit on rl_shadow_cal's states"}
    head = [x for x in profile.read_text().splitlines() if x.startswith("#") and "backends.e5_small filled" not in x]
    head.append(f"# backends.e5_small filled by scripts/e5_analysis.py --prepare-live for e5-{slot}: {'; '.join(notes)}")
    profile.write_text("\n".join(head) + "\n" + yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    HP.load_profile(profile)
    for c in cfgs:
        t = c.read_text()
        c.write_text(re.sub(r"e5-e5_[a-z0-9]+", f"e5-{slot}", t))
    return {"slot": slot, "taus": {dp: blk.get(dp) for dp in ("tool_select", "tool_params")},
            "skip": blk.get("skip"), "notes": notes}


# ------------------------------------------------------------------ outputs
def _p(x: Any, nd: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}"


def _ms(x: Any) -> str:
    return "–" if x is None else f"{x:,.0f}"


def _x(x: Any) -> str:
    return "–" if x is None else f"{x:.2f}×"


def markdown(d: Mapping[str, Any]) -> str:
    L = ["# E5: small decision models (exploratory; descriptive)", "", d["label"] + ".", "",
         f"Generated {d['generated']} from `{d['run']}`; host {d['host'].get('instance_type') or '?'} "
         f"({d['host'].get('gpu_name') or '?'}); GPU hours used {d['gpu_hours'] and round(d['gpu_hours'], 2)}.", ""]
    if d.get("swap"):
        L += [f"Smoke swap: {d['swap']}.", ""]
    if d.get("not_run"):
        L += [f"Not run (cap or failure): {', '.join(d['not_run'])}.", ""]
    L += ["## Accuracy (v1 dev = development; C = common states, descriptive)", "",
          "| model | params (non-emb.) | setting | SFT min | v1 acc | ts | tp | stop | recov. | knockout | v1 ECE | "
          "C acc | C after off-plan | C ask-vs-act | C ECE |", "|" + "---|" * 15]
    rows = [(SHORT[s], m) for s, m in d["models"].items()] + [(r, m) for r, m in d["references"].items()]
    for name, m in rows:
        L.append(f"| {m.get('label', name)} | {m.get('params_b', '–')} ({m.get('nonemb_b', '–')}) | "
                 f"{m.get('setting') or '–'} | {m.get('sft_min', '–')} | {_p(m.get('v1_acc'), 2)} | "
                 f"{_p(m.get('v1_tool_select'), 2)} | {_p(m.get('v1_tool_params'), 2)} | {_p(m.get('v1_stop'), 2)} | "
                 f"{_p(m.get('v1_recovery'), 2)} | {_p(m.get('v1_knockout'), 2)} | "
                 + ("–" if m.get("v1_ece") is None else f"{m['v1_ece']:.3f}")
                 + f" | {_p(m.get('c_acc'))} | {_p(m.get('c_acc_offplan'))} | {_p(m.get('c_acc_ask'))} | "
                 + ("–" if m.get("c_ece") is None else f"{m['c_ece']:.3f}") + " |")
    L += ["", "Paired task bootstrap on C (B = 2,000, seed 0), model − reference, points; reading rule of plan §8 "
          "(keeps if the lower end ≥ −1.0; loses if the upper end < 0; else inconclusive). One seed per model.", "",
          "| contrast | all states | after off-plan | ask-vs-act | v1 acc diff |", "|---|---|---|---|---|"]
    for n, c in d["contrasts"].items():
        cell = [f"{_sg(100 * c[k]['delta'])} [{_sg(100 * c[k]['lo'])}, {_sg(100 * c[k]['hi'])}] ({c[k]['reading']})"
                for k in ("all", "offplan", "ask")]
        L.append(f"| {n} | " + " | ".join(cell) + f" | {_p(c.get('v1_acc_diff'), 2)} |")
    L += ["", "## Latency and memory (one session, one GPU; per request on the final harness's request stream)", "",
          "Serve peak = nvidia-smi memory in use while serving (PyTorch's allocator cache included); train peak = the "
          "same during SFT. The 4B's thresholds stay frozen; its row \"4b\" in the fits below is the refit on its "
          "replayed answers (information only).", "",
          "| served | c=1 p50 | c=1 p90 | c=4 p50 | c=4 p90 | S1/step c=1 p50 | S1/step c=4 p50 | synth. c=1 p50/p90 | "
          "synth. c=4 p50/p90 | speed-up c=1 / c=4 (p50) | serve peak MiB | train peak MiB | KV B/token |",
          "|" + "---|" * 13]
    for lab, r in d["latency"].items():
        b1, b4 = (r.get("bench") or {}).get("c1") or {}, (r.get("bench") or {}).get("c4") or {}
        sy = r.get("synthetic") or {}
        s1, s4 = sy.get("c=1") or {}, sy.get("c=4") or {}
        sp = r.get("speedup_vs_4b") or {}
        tp = (d["models"].get(lab) or {}).get("train_peak_mib")
        L.append(f"| {lab} | {_ms(b1.get('p50_ms'))} | {_ms(b1.get('p90_ms'))} | {_ms(b4.get('p50_ms'))} | "
                 f"{_ms(b4.get('p90_ms'))} | {_ms(b1.get('s1_ms_per_step_p50'))} | {_ms(b4.get('s1_ms_per_step_p50'))} | "
                 f"{_ms(s1.get('p50_ms'))} / {_ms(s1.get('p90_ms'))} | {_ms(s4.get('p50_ms'))} / {_ms(s4.get('p90_ms'))} | "
                 f"{_x(sp.get('c1_p50'))} / {_x(sp.get('c4_p50'))} | {_ms(r.get('serve_peak_mib'))} | {_ms(tp)} | "
                 f"{_ms((r.get('cache') or {}).get('bytes_per_token'))} |")
    lc = d["live_choice"]
    L += ["", "## Live row", "", f"Model choice (registered rule, offline records only): {lc}.", ""]
    if d.get("cal_fits"):
        L += ["Offline threshold fits (entry (g)): " + "; ".join(
            f"{s}: " + ", ".join(f"{dp} τ̂ {v['tau_hat']:.10g} ({v['tau_rule']}, n {v['n_paired']})"
                                 if v.get("tau_hat") is not None else f"{dp} –" for dp, v in f.items())
            for s, f in d["cal_fits"].items()), ""]
    if d.get("cal_coverage"):
        L += ["Coverage on the calibration states at the fitted thresholds (frozen 4B: frozen thresholds): share of "
              "decisions the cascade takes, System-1 accuracy on them, the LLM's on the same, the cascade's overall "
              "accuracy vs the LLM alone.", "",
              "| System-1 | decision point | τ | paired | coverage | S1 acc (taken) | LLM acc (taken) | cascade acc | "
              "LLM acc |", "|---|---|---|---|---|---|---|---|---|"]
        for n, r in d["cal_coverage"].items():
            for dp, v in r.items():
                L.append(f"| {n} | {dp} | {v['tau']:.10g} | {v['n_paired']} | {_p(v['coverage'])} | "
                         f"{_p(v['s1_acc_accepted'], 2)} | {_p(v['s2_acc_accepted'], 2)} | {_p(v['cascade_acc'], 2)} | "
                         f"{_p(v['s2_acc_all'], 2)} |")
        L.append("")
    if d.get("cal_check_4b"):
        L += [f"4B replay check: " + ", ".join(f"{dp} agreement {v['agreement']:.4f}"
                                                for dp, v in d["cal_check_4b"]["by_dp"].items())
              + f" (ok {d['cal_check_4b']['ok']}).", ""]
    for row, title in (("live", "Live row"), ("live_direct", "Route diagnostic: the same tasks on the direct API")):
        lv = d.get(row)
        if not (lv and lv.get("n_pairs")):
            continue
        L += [f"### {title}", "", f"System-2: {lv.get('system2')}. Dollar figures: {lv.get('dollar_scope')}.", "",
              f"n = {lv['n_pairs']} tasks complete in every arm; host ${lv['host_usd_per_h']}/h; q per episode "
              + ", ".join(f"{a} ${q:.4f}" for a, q in lv["q_usd_per_episode"].items()) + ".", "",
              "| arm | success | API $/ep | $/ep incl. q | wall s/ep | S1 s/ep | S2 s/ep | tool cov. | arg cov. |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a, r in lv["arms"].items():
            L.append(f"| {a} | {_p(r['success'])} | {r['api_cost']:.4f} | {r['cost_with_q']:.4f} | {r['wall_s']:.2f} | "
                     f"{r['s1_s']:.2f} | {r['s2_s']:.2f} | {_p(r['tool_coverage'])} | {_p(r['arg_coverage'])} |")
        L += ["", "| contrast | success Δ pt [95 %] | cost saving (incl. q) | wall Δ s [95 %] |", "|---|---|---|---|"]
        for n, c in lv["contrasts"].items():
            L.append(f"| {n} | {c['success_delta_pt']:+.1f} [{c['success_delta_ci'][0]:+.1f}, "
                     f"{c['success_delta_ci'][1]:+.1f}] | {_p(c['cost_saving_frac'])} % | {c['wall_delta_s']:+.2f} "
                     f"[{c['wall_delta_ci'][0]:+.2f}, {c['wall_delta_ci'][1]:+.2f}] |")
        lat = lv.get("llm_latency") or {}
        if lat:
            L += ["", "LLM (System-2) call latency, seconds: per decision point and per served provider.", "",
                  "| arm | group | calls | p50 | p90 | mean | calls > 5 s | their share of LLM time |",
                  "|---|---|---|---|---|---|---|---|"]
            for a, r in lat.items():
                for grp, stats in [("all", r["all"])] + [(f"dp {k}", v) for k, v in r["by_dp"].items()] \
                        + [(f"served by {k}", v) for k, v in r["by_provider"].items()]:
                    L.append(f"| {a} | {grp} | {stats['n']} | " + " | ".join(
                        "–" if stats[k] is None else f"{stats[k]:.2f}" for k in ("p50", "p90", "mean"))
                        + f" | {stats['n_slow5']} | {_p(stats['slow_share'])} |")
        L.append("")
    if not (d.get("live") or {}).get("n_pairs") and lc.get("run"):
        L += ["Live row not run yet.", ""]
    return "\n".join(L) + "\n"


def _sg(x: float, nd: int = 1) -> str:
    """Signed, rounded to ``nd`` decimals, never "-0.0"."""
    v = round(float(x), nd)
    return f"{0.0 if v == 0 else v:+.{nd}f}"


def _t(x: Any, nd: int = 1, pct: bool = True) -> str | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    v = 100 * x if pct else x
    return f"{v:,.{nd}f}".replace(",", "{,}")


def tex_keys(d: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {}

    def put(key: str, val: str | None) -> None:
        if val is not None:
            k[key] = val

    for s, m in list(d["models"].items()) + list(d["references"].items()):
        n = SHORT.get(s, s)
        for f in ("v1_acc", "v1_tool_select", "v1_tool_params", "v1_stop", "v1_recovery", "v1_knockout"):
            put(f"e5/{n}/{f}", _t(m.get(f), 2))
        for f in ("c_acc", "c_acc_offplan", "c_acc_ask"):
            put(f"e5/{n}/{f}", _t(m.get(f), 1))
        for f in ("v1_ece", "c_ece"):
            put(f"e5/{n}/{f}", _t(m.get(f), 3, pct=False))
        put(f"e5/{n}/sft_min", None if m.get("sft_min") is None else str(int(m["sft_min"])))
        put(f"e5/{n}/train_peak_gib", _t(None if m.get("train_peak_mib") is None else m["train_peak_mib"] / 1024, 1,
                                          pct=False))
    for n, c in d["contrasts"].items():
        for st in ("all", "offplan", "ask"):
            for f in ("delta", "lo", "hi"):
                put(f"e5/contrast/{n}/c_{st}_{f}", _sg(100 * c[st][f]))
    for lab, r in d["latency"].items():
        n = SHORT.get(lab, lab)
        for c in ("c1", "c4"):
            b = (r.get("bench") or {}).get(c) or {}
            for q in ("p50_ms", "p90_ms", "s1_ms_per_step_p50"):
                put(f"e5/{n}/lat_{c}_{q.replace('_ms', '')}", _t(b.get(q), 0, pct=False))
            sp = r.get("speedup_vs_4b") or {}
            for q in ("p50", "p90"):
                put(f"e5/{n}/speedup_{c}_{q}", _t(sp.get(f"{c}_{q}"), 2, pct=False))
        put(f"e5/{n}/serve_peak_gib", _t(None if r.get("serve_peak_mib") is None else r["serve_peak_mib"] / 1024, 1,
                                          pct=False))
    for row in ("live", "live_direct"):
        lv = d.get(row)
        if not (lv and lv.get("n_pairs")):
            continue
        pre = f"e5/{row}"
        put(f"{pre}/n", str(lv["n_pairs"]))
        put(f"{pre}/q_usd_per_h", _t(lv.get("host_usd_per_h"), 3, pct=False))
        for a, r in lv["arms"].items():
            put(f"{pre}/{a}/success", _t(r["success"], 1))
            put(f"{pre}/{a}/cost", _t(r["cost_with_q"], 4, pct=False))
            put(f"{pre}/{a}/api_cost", _t(r["api_cost"], 4, pct=False))
            put(f"{pre}/{a}/wall_s", _t(r["wall_s"], 2, pct=False))
            put(f"{pre}/{a}/s1_s", _t(r["s1_s"], 2, pct=False))
            put(f"{pre}/{a}/s2_s", _t(r["s2_s"], 2, pct=False))
            put(f"{pre}/{a}/tool_cov", _t(r.get("tool_coverage"), 1))
            put(f"{pre}/{a}/arg_cov", _t(r.get("arg_coverage"), 1))
        for n, c in lv["contrasts"].items():
            put(f"{pre}/{n}/success_delta", _sg(c["success_delta_pt"]))
            put(f"{pre}/{n}/cost_saving", _t(c["cost_saving_frac"], 1))
            put(f"{pre}/{n}/wall_saving", _t(c["wall_saving_frac"], 1))
            put(f"{pre}/{n}/cost_saving_no_q", _t(c.get("cost_saving_frac_no_q"), 1))
            put(f"{pre}/{n}/wall_delta_s", _sg(c["wall_delta_s"], 2))
        for a, r in (lv.get("llm_latency") or {}).items():
            for dp, st in [("all", r["all"])] + list(r["by_dp"].items()):
                for f in ("p50", "p90", "mean"):
                    put(f"{pre}/lat/{a}/{dp}/{f}", _t(st[f], 2, pct=False))
                put(f"{pre}/lat/{a}/{dp}/n", str(st["n"]))
                put(f"{pre}/lat/{a}/{dp}/n_slow5", str(st["n_slow5"]))
                put(f"{pre}/lat/{a}/{dp}/slow_share", _t(st["slow_share"], 1))
    return k


def write(d: Mapping[str, Any], out: Path, tex: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(d, indent=1, default=float) + "\n")
    (out / "summary.md").write_text(markdown(d))
    if tex is not None:
        keys = tex_keys(d)
        lines = ["% AUTO-GENERATED by scripts/e5_analysis.py -- do not edit by hand (exploratory row E5; descriptive).",
                 "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
        tex.parent.mkdir(parents=True, exist_ok=True)
        tex.write_text("\n".join(lines + [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(keys.items())]) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=str(ROOT / "runs" / "e5_small"))
    ap.add_argument("--part", action="append", default=[], help="a further host's run dir (repeatable)")
    ap.add_argument("--live", default=str(ROOT / "runs" / "e5_small_live"))
    ap.add_argument("--live-direct", default=str(ROOT / "runs" / "e5_small_live_direct"),
                    help="the route diagnostic's run (Deviations 2026-10-06 ~19:40 UTC)")
    ap.add_argument("--corpus", default=str(S23))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tex", default=str(TEX))
    ap.add_argument("--no-tex", action="store_true")
    ap.add_argument("--fit", action="store_true", help="entry (g)'s offline threshold fits (derive + threshold_theory)")
    ap.add_argument("--prepare-live", action="store_true",
                    help="write the chosen model and its fitted thresholds into the live configs and the E5 profile")
    a = ap.parse_args(argv)
    d = analyze(Path(a.run), Path(a.live), Path(a.corpus), a.fit, [Path(x) for x in a.part], Path(a.live_direct))
    write(d, Path(a.out), None if a.no_tex else Path(a.tex))
    if a.prepare_live:
        print(json.dumps({"prepare_live": prepare_live(d)}, default=str))
    print(json.dumps({"models": {s: {k: m.get(k) for k in ("c_acc", "c_acc_offplan", "c_acc_ask", "v1_acc")}
                                 for s, m in d["models"].items()}, "live_choice": d["live_choice"]}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
