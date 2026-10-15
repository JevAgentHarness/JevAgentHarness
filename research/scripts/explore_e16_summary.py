#!/usr/bin/env python3
r"""E16's readout (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E16: the best open-weights decision model on all
three levels"; descriptive): per arm the three levels (L1 = E1's 3,991 decisions, L2 = the common states C, L3 = live
REPLACE on the first 200 fresh Stage-2 tasks), each against AgentDecision-4B, Jev and the best open-weights decision
model we did not train at that level (registration (d) 4), the same-engine timing, the host and the spend.

    python scripts/explore_e16_best_open.py summarize [--box DIR] [--runs DIR]
      -> results/explore/e16_best_open/summary.{md,json}, paper/generated/e16_best_open.tex (\res keys only)

Inputs: the E16 box records (results/explore/e16_best_open/box: selection, gates, corpus answers, timing, host), the
E16 runs synced from the host (runs/e1_bfcl_when2call_e16, runs/e16{a,b}_live{,_smoke}) and the logged rows of E9, the
size ladder, Stage 2 and E15 (their records under runs/ and results/)."""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import explore_e16_best_open as E16  # noqa: E402

S23 = ROOT / "results" / "stage2" / "s23"
E15 = ROOT / "results" / "explore" / "e15_baselines"
LADDER_LIVE = ROOT / "runs" / "explore_startlux_live"
S2_LIVE = ROOT / "runs" / "stage2_startlux_states"
RL_LIVE = ROOT / "runs" / "stage2_rl_states"
# the open-weights decision models we did not train (registration (d) 4): key -> (label, E1 records, C answers, live records)
# E1 records of E9's models are E9.MODELS[key][1]; E15's are results/explore/e15_baselines/l1/s1_<slug>.jsonl.gz
THIRD_PARTY: dict[str, tuple[str, str | None, Path | None, Path | None]] = {
    "sl0.8b": ("StartLux-0.8B", "e9", S23 / "answers_sl0.8b.jsonl", LADDER_LIVE / "sl0.8b_ts_replace_f.jsonl"),
    "sl2b": ("StartLux-2B", "e9", S23 / "answers_sl2b.jsonl", LADDER_LIVE / "sl2b_ts_replace_f.jsonl"),
    "sl4b": ("StartLux-4B", "e9", S23 / "answers_sl4b.jsonl", S2_LIVE / "sl4b_ts_replace_f.jsonl"),
    "sl9b": ("StartLux-9B", "e9", S23 / "answers_sl9b.jsonl", LADDER_LIVE / "sl9b_ts_replace_f.jsonl"),
    "sl27b": ("StartLux-27B (Q8_0)", "e9", S23 / "answers_sl27b.jsonl", S2_LIVE / "sl27b_ts_replace_f.jsonl"),
    "intern": ("Intern-Decision-4B", "e9", S23 / "answers_intern.jsonl", RL_LIVE / "intern_ts_replace_f.jsonl"),
    "openjev": ("OpenJev", "e15", E15 / "c" / "answers_openjev.jsonl.gz", E15 / "live" / "e15_live_openjev.json"),
    "openjevhf": ("OpenJev-27B", "e15", E15 / "c" / "answers_openjevhf.jsonl.gz", E15 / "live" / "e15_live_openjevhf.json"),
    "clefflash": ("Clef-Flash", "e15", E15 / "c" / "answers_clefflash.jsonl.gz", None),
    "clef": ("Clef-27B (8-bit)", "e15", None, None),
    "jevk5": ("JevK5", "e15", E15 / "c" / "answers_jevk5.jsonl.gz", None),
    "clm": ("CLM", "e15", E15 / "c" / "answers_clm.jsonl.gz", None),
    "laya": ("Laya", "e15", E15 / "c" / "answers_laya.jsonl.gz", None),
    "verdict": ("Verdict", "e15", E15 / "c" / "answers_verdict.jsonl.gz", None),
    "decider": ("decider-4b", "e15", E15 / "c" / "answers_decider.jsonl.gz", None),
    "imajev": ("Imajev-4B", "e15", E15 / "c" / "answers_imajev.jsonl.gz", None),
    "bosun": ("Bosun", "e15", E15 / "c" / "answers_bosun.jsonl.gz", None),
}
# our own earlier open models, reported beside the arms (never "the best open model" in the registered rule)
OURS: dict[str, tuple[str, str | None, Path | None, Path | None]] = {
    "frozen": ("AgentDecision-4B", "e1_bfcl_when2call_rl/s1_jevagent_rl.jsonl", S23 / "answers_rl.jsonl",
               RL_LIVE / "rl_ts_replace_f.jsonl"),
    "e12": ("AgentDecision-4B (mixed, E12)", "e1_bfcl_when2call_e12/s1_e12_mixed.jsonl",
            ROOT / "results/explore/e12/box/s23/answers_e12_mixed.jsonl", ROOT / "runs/e12_live/e12_ts_replace_f.jsonl"),
    "e12c": ("StartLux-4B (mixed, E12c)", "e1_bfcl_when2call_e12c/s1_e12c_sl4b.jsonl",
             ROOT / "results/explore/e12/box_c/s23/answers_e12c_sl4b.jsonl", ROOT / "runs/e12c_live/e12c_ts_replace_f.jsonl"),
}
REFS_E1 = {"frozen": "rl", "jev": "jev", "sl4b": "sl4b", "sl9b": "sl9b", "sl27b": "sl27b"}
NAN = float("nan")


# ============================================================================== helpers
def _jsonl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def latest(p: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in _jsonl(p):
        out[str(r["id"])] = r
    return out


def tex_values(path: Path) -> dict[str, str]:
    import re

    out: dict[str, str] = {}
    if path.exists():
        for m in re.finditer(r"\\defres\{([^}]*)\}\{([^}]*)\}", path.read_text()):
            out[m.group(1)] = m.group(2)
    return out


def _pt(x: Any, d: int = 1) -> str:
    return f"{100 * x:.{d}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "--"


def _pm(x: Any, d: int = 1) -> str:
    return f"{x:+.{d}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "--"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_answers_any(p: Path) -> dict[str, dict[str, Any]]:
    recs: dict[str, dict[str, Any]] = {}
    for r in _jsonl(p):
        recs[r["qhash"]] = r
    return recs


def live_y(path: Path | None, tasks: set[str]) -> tuple[dict[str, Any], dict[str, float]]:
    """Per-task live success of a logged row: a runner record file (the ladder, Stage 2, E12 / E12c, E16) or an
    E15 slim live record (results/explore/e15_baselines/live/*.json)."""
    import explore_startlux_live as SL

    if path is None or not path.exists():
        return {}, {}
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        arm = next(iter(d["arms"]))
        y = {t: float(r["success"]) for t, r in d["arms"][arm].items() if r.get("in_live") and not r.get("error") and t in tasks}
        return {"live_n": len(y), "live_success": float(np.mean(list(y.values()))) if y else None}, y
    r, y = SL.live_row(path, tasks)
    return r, y


# ============================================================================== the logged rows (E9, ladder, Stage 2, E15)
def e1_records(key: str, src: str | None, runs: Path) -> dict[str, dict[str, Any]] | None:
    import explore_e9_frontier as E9

    if src == "e9":
        p = runs / E9.MODELS[key][1]
    elif src == "e15":
        p = E15 / "l1" / f"s1_{key}.jsonl.gz"
    elif src:
        p = runs / src
    else:
        return None
    return latest(p) if p.exists() else None


def logged_levels(runs: Path, tasks: set[str]) -> dict[str, dict[str, Any]]:
    """L1 / L2 / L3 point estimates of every third-party open row and of ours (from their records; E15's summary when a
    record is slimmed away)."""
    import ablation_score as AS
    import e5_analysis as EA
    import explore_e9_frontier as E9

    e15 = json.loads((E15 / "summary.json").read_text())["models"] if (E15 / "summary.json").exists() else {}
    occ, eligible = AS.load_corpus(S23)
    out: dict[str, dict[str, Any]] = {}
    for key, (label, src, cpath, lpath) in {**THIRD_PARTY, **OURS}.items():
        row: dict[str, Any] = {"label": label, "ours": key in OURS}
        recs = e1_records(key, src, runs)
        if recs:
            s = E9.model_stats(list(recs.values()), "gpu")
            row["l1"] = {"acc": s["acc"], "n": s["n"], "families": {f: v["acc"] for f, v in s["families"].items()},
                         "usd_per_1000": 1000 * s["usd"]}
        elif key in e15 and (e15[key].get("l1") or {}).get("acc") is not None:
            row["l1"] = {"acc": e15[key]["l1"]["acc"], "n": e15[key]["l1"]["n"],
                         "families": {f: v["acc"] for f, v in e15[key]["l1"]["families"].items()},
                         "usd_per_1000": 1000 * e15[key]["l1"]["usd"]}
        if cpath is not None and cpath.exists():
            m, per = EA.c_rows(occ, eligible, load_answers_any(cpath))
            row["l2"] = {"acc": m["c_acc"], "offplan": m["c_acc_offplan"], "ask": m["c_acc_ask"], "missing": m["c_missing"],
                         "per_task": per["all"]}
        elif key in e15 and (e15[key].get("l2") or {}).get("corpus_acc") is not None:
            row["l2"] = {"acc": e15[key]["l2"]["corpus_acc"], "offplan": e15[key]["l2"].get("corpus_acc_offplan"),
                         "ask": e15[key]["l2"].get("corpus_acc_ask"), "missing": e15[key]["l2"].get("missing")}
        r, y = live_y(lpath, tasks)
        if y:
            row["l3"] = {"success": r["live_success"], "n": r["live_n"], "y": y}
        elif key in e15 and (e15[key].get("l3") or {}).get("live_success") is not None:
            row["l3"] = {"success": e15[key]["l3"]["live_success"], "n": e15[key]["l3"]["live_n"]}
        out[key] = row
    return out


def best_open(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per level the third-party open row with the highest value (registration (d) 4)."""
    best: dict[str, dict[str, Any]] = {}
    for lvl, field in (("l1", "acc"), ("l2", "acc"), ("l3", "success")):
        cands = {k: r[lvl][field] for k, r in rows.items() if not r.get("ours") and r.get(lvl) and r[lvl].get(field) is not None}
        if cands:
            k = max(cands, key=lambda m: (cands[m], m))
            best[lvl] = {"key": k, "label": rows[k]["label"], "value": cands[k],
                         "ranking": sorted(cands, key=lambda m: -cands[m])[:5]}
    return best


# ============================================================================== the E16 arms
def arm_e1(arm: str, runs: Path, rows: Mapping[str, Mapping[str, Any]], best: Mapping[str, Any]) -> dict[str, Any]:
    import explore_e9_frontier as E9

    f = runs / E16.E1_RUN / f"s1_{E16.ARMS[arm]['e1_backend']}.jsonl"
    if not f.exists():
        return {"available": False, "reason": f"no {f.relative_to(ROOT)}"}
    new = latest(f)
    ref = latest(runs / E9.MODELS["rl"][1])
    ids = sorted(ref)
    if set(new) != set(ids):
        return {"available": False, "reason": f"{len(set(new) & set(ids))} of {len(ids)} E1 decisions answered"}
    s = E9.model_stats([new[i] for i in ids], "gpu")
    out: dict[str, Any] = {"available": True, "n": len(ids), "acc": s["acc"], "acc_ci": s["acc_ci"],
                           "families": {k: v["acc"] for k, v in s["families"].items()}, "usd_per_1000": 1000 * s["usd"],
                           "lat_ms_median": s["lat_ms_median"], "n_errors": s["n_errors"], "vs": {}}
    refs = dict(REFS_E1)
    refs["e12" if arm == "a" else "e12c"] = None
    if best.get("l1"):
        refs["bestopen"] = best["l1"]["key"]
    for name, key in refs.items():
        if key in E9.MODELS:
            d = latest(runs / E9.MODELS[key][1])
        elif name in OURS:
            p = runs / OURS[name][1]
            d = latest(p) if p.exists() else {}
        elif key in THIRD_PARTY:
            d = e1_records(key, THIRD_PARTY[key][1], runs) or {}
        else:
            d = {}
        if set(d) < set(ids):
            continue
        g: dict[str, Any] = {"all": E9.paired_gap([E9.is_correct(new[i]) for i in ids], [E9.is_correct(d[i]) for i in ids])}
        for fam in E9.FAMILIES:
            fi = [i for i in ids if E9.family_of(d[i]) == fam]
            g[fam] = E9.paired_gap([E9.is_correct(new[i]) for i in fi], [E9.is_correct(d[i]) for i in fi])
        g["ref_acc"] = sum(E9.is_correct(d[i]) for i in ids) / len(ids)
        g["ref_key"] = key
        out["vs"][name] = g
    w2c: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i in ids:
        if E9.family_of(ref[i]) == "w2c":
            lab = (ref[i].get("gold") or ["?"])[0]
            w2c[lab][0] += int(E9.is_correct(new[i]))
            w2c[lab][1] += 1
    out["w2c_by_label"] = {k: {"n": v[1], "acc": v[0] / v[1]} for k, v in w2c.items()}
    conf: dict[tuple[str, str], int] = defaultdict(int)
    for i in ids:
        if E9.family_of(ref[i]) == "w2c":
            conf[((ref[i].get("gold") or ["?"])[0][:4], str(new[i].get("pred"))[:4])] += 1
    out["w2c_confusion"] = {f"{g}->{p}": n for (g, p), n in sorted(conf.items())}
    return out


def arm_c(arm: str, box: Path, rows: Mapping[str, Mapping[str, Any]], best: Mapping[str, Any]) -> dict[str, Any]:
    import ablation_score as AS
    import e5_analysis as EA

    f = box / "s23" / f"answers_e16{arm}.jsonl"
    if not f.exists():
        return {"available": False}
    occ, eligible = AS.load_corpus(S23)
    m, per = EA.c_rows(occ, eligible, AS.load_answers(f))
    out: dict[str, Any] = {"available": True, "occurrences": len(occ), "acc": m["c_acc"], "offplan": m["c_acc_offplan"],
                           "ask": m["c_acc_ask"], "missing": m["c_missing"], "errors": m["c_errors"], "tasks": m["c_tasks"]}
    fr = rows.get("frozen", {}).get("l2") or {}
    if fr.get("per_task"):
        out["delta_vs_frozen"] = AS.paired_boot(per["all"], fr["per_task"], EA.N_BOOT, 0)
        out["delta_vs_frozen_offplan"] = AS.paired_boot(per["offplan"], EA.c_rows(occ, eligible, AS.load_answers(S23 / "answers_rl.jsonl"))[1]["offplan"], EA.N_BOOT, 0)
        out["non_inferior"] = out["delta_vs_frozen"]["lo"] >= -0.01
    if best.get("l2"):
        b = rows[best["l2"]["key"]]["l2"]
        if b.get("per_task"):
            out["delta_vs_bestopen"] = AS.paired_boot(per["all"], b["per_task"], EA.N_BOOT, 0)
    gate = box / f"{arm}_cgate.json"
    if gate.exists():
        out["gate"] = json.loads(gate.read_text())
    return out


def arm_live(arm: str, runs: Path, tasks: set[str], rows: Mapping[str, Mapping[str, Any]], best: Mapping[str, Any]) -> dict[str, Any]:
    import explore_common as C

    p = runs / E16.ARMS[arm]["live_run"] / f"{E16.ARMS[arm]['arm']}.jsonl"
    r, y = live_y(p, tasks)
    if not y:
        return {"available": False}
    out: dict[str, Any] = {"available": True, **r, "vs": {}}
    refs = {"frozen": rows.get("frozen", {}).get("l3", {}).get("y")}
    jev_r, jev_y = live_y(RL_LIVE / "jev_ts_replace_f.jsonl", tasks)
    refs["jev"] = jev_y
    if best.get("l3"):
        refs["bestopen"] = rows[best["l3"]["key"]]["l3"].get("y")
    for name, yr in refs.items():
        if not yr:
            continue
        common = sorted(set(y) & set(yr))
        if common:
            nc = C.newcombe([y[t] for t in common], [yr[t] for t in common])
            out["vs"][name] = {"n": len(common), "diff_pt": nc["diff_pt"], "ci_pt": nc["ci_pt"],
                               "ref_success": float(np.mean([yr[t] for t in common]))}
    return out


def arm_box(arm: str, box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sel = box / f"sel_{arm}" / "selection.json"
    if sel.exists():
        out["selection"] = json.loads(sel.read_text())
    tr = box / "train_runs.jsonl"
    if tr.exists():
        out["train_runs"] = [r for r in _jsonl(tr) if r.get("arm") == arm]
    if arm == "b":
        for k, f in (("plan", box / "b_plan.json"),):
            if f.exists():
                out[k] = json.loads(f.read_text())
        sm = box / "b_smoke.jsonl"
        if sm.exists():
            out["smoke"] = _jsonl(sm)
    for part in ("v1", "pubdev"):
        f = box / "eval" / f"{arm}_{part}.json"
        if f.exists():
            files = json.loads(f.read_text())["files"]
            out.setdefault("offline", {})[part] = {Path(k).name.split(".")[0]: v["overall"]["acc"] for k, v in files.items()}
    return out


def fast_block(box: Path, runs: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    vf = box / "fast" / "vllm_version.txt"
    if vf.exists():
        out["engine"] = vf.read_text().strip()
    served = {"e16a": runs / E16.E1_RUN / "s1_e16a.jsonl", "e16b": runs / E16.E1_RUN / "s1_e16b.jsonl",
              "frozen": runs / E16.E1_RUN / "s1_frozen_rehost.jsonl"}
    for m in ("e16a", "e16b", "frozen"):
        f = box / "fast" / f"time_{m}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        row = {k: v for k, v in d.items() if k not in ("answers", "label")}
        if d.get("answers") and served[m].exists():
            s = latest(served[m])
            ids = sorted(set(d["answers"]) & set(s))
            row["agreement_with_served"] = sum(1 for i in ids if d["answers"][i] == s[i].get("pred")) / len(ids) if ids else None
        out[m] = row
    return out


def rehost_block(runs: Path) -> dict[str, Any]:
    import explore_e12_train as E12

    run = runs / E16.E1_RUN
    if not (run / "s1_frozen_rehost.jsonl").exists():
        return {"available": False}
    r = E12.rehost(run)
    r["available"] = True
    r["host_factor"] = (r["usd_per_decision_logged"] / r["usd_per_decision_rehost"]
                        if r.get("usd_per_decision_rehost") else None)
    return r


def host_block(box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, f in (("setup", box / "host_setup.json"), ("final", box / "host_final.json"), ("host", box / "host.json")):
        if f.exists():
            out[k] = json.loads(f.read_text())
    launch = (out.get("host") or {}).get("launch_utc")
    end = (out.get("host") or {}).get("stop_utc") or (out.get("final") or {}).get("final_utc")
    if launch and end:
        t0 = dt.datetime.fromisoformat(launch.replace("Z", "+00:00"))
        t1 = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
        out["host_hours"] = (t1 - t0).total_seconds() / 3600
    return out


# ============================================================================== summarize
def summarize(box: Path, runs: Path) -> int:
    import explore_startlux_live as SL

    tasks = set(json.loads(SL.TASKS.read_text())["task_ids"][:E16.LIVE_N])
    rows = logged_levels(runs, tasks)
    best = best_open(rows)
    d: dict[str, Any] = {"label": E16.LABEL, "generated": _now(), "best_open": best, "arms": {}}
    bi = E16.OUT / "build_info.json"
    if bi.exists():
        b = json.loads(bi.read_text())
        d["data"] = {k: b[k] for k in ("rows", "train_by_family", "train_by_source", "train_units", "units", "audit",
                                       "replay", "pool_train_units", "quota_train") if k in b}
    for arm in ("a", "b"):
        x: dict[str, Any] = {"name": E16.ARMS[arm]["name"], "served": E16.ARMS[arm]["served"], **arm_box(arm, box)}
        x["e1"] = arm_e1(arm, runs, rows, best)
        x["c"] = arm_c(arm, box, rows, best)
        x["live"] = arm_live(arm, runs, tasks, rows, best)
        bo: dict[str, Any] = {}
        if x["e1"].get("available") and best.get("l1"):
            bo["l1"] = {"gap_pt": 100 * (x["e1"]["acc"] - best["l1"]["value"]), "beats": x["e1"]["acc"] >= best["l1"]["value"],
                        "model": best["l1"]["label"]}
        if x["c"].get("available") and best.get("l2"):
            bo["l2"] = {"gap_pt": 100 * (x["c"]["acc"] - best["l2"]["value"]), "beats": x["c"]["acc"] >= best["l2"]["value"],
                        "model": best["l2"]["label"]}
        if x["live"].get("available") and best.get("l3"):
            bo["l3"] = {"gap_pt": 100 * (x["live"]["live_success"] - best["l3"]["value"]),
                        "beats": x["live"]["live_success"] >= best["l3"]["value"], "model": best["l3"]["label"]}
        x["best_open"] = bo
        d["arms"][arm] = x
    d["rows"] = {k: {kk: ({f: v for f, v in vv.items() if f not in ("per_task", "y")} if isinstance(vv, dict) else vv)
                     for kk, vv in r.items()} for k, r in rows.items()}
    d["rehost"] = rehost_block(runs)
    d["fast"] = fast_block(box, runs)
    d["host"] = host_block(box)
    sp = E16.total_spend(runs)
    d["spend"] = {"gemini": sp["gemini"], "total": sp["total"], "records": sp["records"], "cap": E16.CAP_USD}
    for arm in ("a", "b"):
        f = box / f"{arm}_projection.json"
        if f.exists():
            d["arms"][arm]["projection"] = json.loads(f.read_text())
    E16.OUT.mkdir(parents=True, exist_ok=True)
    (E16.OUT / "summary.json").write_text(json.dumps(_jsonable(d), indent=1) + "\n")
    keys = tex_keys(d)
    (E16.OUT / "summary.md").write_text(markdown(d, keys))
    write_tex(keys)
    print(f"wrote {E16.OUT / 'summary.md'}, {E16.TEX} ({len(keys)} keys)")
    return 0


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def tex_keys(d: Mapping[str, Any]) -> dict[str, str]:
    import explore_common as C

    v: dict[str, str] = {}
    data = d.get("data") or {}
    if data:
        v["e16/data/new"] = C.fmt(int(data["rows"]["public_train"]))
        v["e16/data/replay"] = C.fmt(int(data["rows"]["replay"]))
        v["e16/data/total"] = C.fmt(int(data["rows"]["train_mix"]))
        v["e16/data/units"] = C.fmt(int(data["train_units"]))
        for f, n in data["train_by_family"].items():
            v[f"e16/data/{f.replace('_', '')}"] = C.fmt(int(n))
        v["e16/audit/units"] = C.fmt(int(data["audit"]["units_with_hit"]))
        v["e16/audit/items"] = C.fmt(int(data["audit"]["items_dropped"]))
    for lvl, b in (d.get("best_open") or {}).items():
        v[f"e16/best_open/{lvl}/model"] = b["label"]
        v[f"e16/best_open/{lvl}/value"] = _pt(b["value"])
    for k, r in (d.get("rows") or {}).items():
        kk = k.replace(".", "")
        if r.get("l1"):
            v[f"e16/row/{kk}/l1"] = _pt(r["l1"]["acc"])
        if r.get("l2"):
            v[f"e16/row/{kk}/l2"] = _pt(r["l2"]["acc"])
            if r["l2"].get("ask") is not None:
                v[f"e16/row/{kk}/l2ask"] = _pt(r["l2"]["ask"])
        if r.get("l3"):
            v[f"e16/row/{kk}/l3"] = _pt(r["l3"]["success"])
    for arm, x in (d.get("arms") or {}).items():
        a = f"e16/{arm}"
        v[f"{a}/name"] = x["name"]
        e1 = x.get("e1") or {}
        v[f"{a}/e1/avail"] = "yes" if e1.get("available") else "no"
        if e1.get("available"):
            v[f"{a}/e1/acc"] = _pt(e1["acc"])
            v[f"{a}/e1/acc/lo"], v[f"{a}/e1/acc/hi"] = _pt(e1["acc_ci"][0]), _pt(e1["acc_ci"][1])
            for fam, acc in e1["families"].items():
                v[f"{a}/e1/acc/{fam}"] = _pt(acc)
            v[f"{a}/e1/usdk"] = f"{e1['usd_per_1000']:.4f}"
            v[f"{a}/e1/lat"] = C.fmt(int(round(e1["lat_ms_median"])))
            for lab, y in (e1.get("w2c_by_label") or {}).items():
                v[f"{a}/e1/w2c/{lab[:4]}"] = _pt(y["acc"])
            for ref, g in (e1.get("vs") or {}).items():
                for fam, gg in g.items():
                    if isinstance(gg, dict) and gg.get("est") is not None:
                        v[f"{a}/e1/vs/{ref}/{fam}"] = _pm(gg["est"])
                        v[f"{a}/e1/vs/{ref}/{fam}/lo"] = _pm(gg["ci"][0])
                        v[f"{a}/e1/vs/{ref}/{fam}/hi"] = _pm(gg["ci"][1])
                        v[f"{a}/e1/vs/{ref}/{fam}/abs"] = f"{abs(gg['est']):.1f}"
        c = x.get("c") or {}
        v[f"{a}/c/avail"] = "yes" if c.get("available") else "no"
        if c.get("available"):
            v[f"{a}/c/acc"], v[f"{a}/c/ask"], v[f"{a}/c/offplan"] = _pt(c["acc"]), _pt(c["ask"]), _pt(c["offplan"])
            v[f"{a}/c/n"] = C.fmt(int(c["occurrences"]))
            if c.get("delta_vs_frozen"):
                g = c["delta_vs_frozen"]
                v[f"{a}/c/delta"], v[f"{a}/c/delta/lo"], v[f"{a}/c/delta/hi"] = _pm(100 * g["delta"]), _pm(100 * g["lo"]), _pm(100 * g["hi"])
                v[f"{a}/c/ni"] = "yes" if c["non_inferior"] else "no"
            if c.get("delta_vs_frozen_offplan"):
                g = c["delta_vs_frozen_offplan"]
                v[f"{a}/c/offplan/delta"], v[f"{a}/c/offplan/lo"], v[f"{a}/c/offplan/hi"] = _pm(100 * g["delta"]), _pm(100 * g["lo"]), _pm(100 * g["hi"])
            if c.get("delta_vs_bestopen"):
                g = c["delta_vs_bestopen"]
                v[f"{a}/c/vsbestopen"], v[f"{a}/c/vsbestopen/lo"], v[f"{a}/c/vsbestopen/hi"] = _pm(100 * g["delta"]), _pm(100 * g["lo"]), _pm(100 * g["hi"])
            if c.get("gate"):
                v[f"{a}/live/gate"] = "yes" if c["gate"]["live"] else "no"
        lv = x.get("live") or {}
        v[f"{a}/live/avail"] = "yes" if lv.get("available") else "no"
        if lv.get("available"):
            v[f"{a}/live/success"] = _pt(lv["live_success"])
            v[f"{a}/live/n"] = C.fmt(int(lv["live_n"]))
            if lv.get("own_acc") is not None:
                v[f"{a}/live/own"] = _pt(lv["own_acc"])
            if lv.get("api_cost") is not None:
                v[f"{a}/live/usdep"] = f"{lv['api_cost']:.4f}"
            if lv.get("steps") is not None:
                v[f"{a}/live/steps"] = f"{lv['steps']:.1f}"
            if lv.get("wall_s") is not None:
                v[f"{a}/live/walls"] = f"{lv['wall_s']:.0f}"
            for ref, g in (lv.get("vs") or {}).items():
                v[f"{a}/live/vs/{ref}"] = _pm(g["diff_pt"])
                v[f"{a}/live/vs/{ref}/lo"], v[f"{a}/live/vs/{ref}/hi"] = _pm(g["ci_pt"][0]), _pm(g["ci_pt"][1])
        for lvl, bo in (x.get("best_open") or {}).items():
            v[f"{a}/best_open/{lvl}"] = "yes" if bo["beats"] else "no"
            v[f"{a}/best_open/{lvl}/gap"] = _pm(bo["gap_pt"])
            v[f"{a}/best_open/{lvl}/gapabs"] = f"{abs(bo['gap_pt']):.1f}"
            v[f"{a}/best_open/{lvl}/model"] = bo["model"]
        for lvl in ("l1", "l2", "l3"):
            v.setdefault(f"{a}/best_open/{lvl}", "--")
        sel = (x.get("selection") or {}).get("selected")
        if sel:
            v[f"{a}/sel/epoch"] = str(sel["epoch"])
        for cand in (x.get("selection") or {}).get("candidates") or []:
            v[f"{a}/sel/ep{cand['epoch']}/devprefix"] = _pt(cand["acc"].get("e1_devprefix"))
            v[f"{a}/sel/ep{cand['epoch']}/v1dev"] = _pt(cand["acc"].get("dev"))
        tr = [r for r in (x.get("train_runs") or []) if r.get("exit") == 0]
        if tr:
            v[f"{a}/train/min"] = str(tr[-1]["minutes"])
            v[f"{a}/train/setting"] = tr[-1]["setting"]
            if tr[-1].get("epochs") is not None:
                v[f"{a}/train/epochs"] = str(tr[-1]["epochs"])
        elif x.get("train_runs"):
            v[f"{a}/train/min"] = str(x["train_runs"][-1]["minutes"])
        if arm == "b" and x.get("plan"):
            v["e16/b/plan"] = x["plan"].get("b", "--")
            if x["plan"].get("reason"):
                v["e16/b/reason"] = str(x["plan"]["reason"])
            if x["plan"].get("epochs") is not None:
                v["e16/b/plan/epochs"] = str(x["plan"]["epochs"])
            if x["plan"].get("projected_min_per_epoch") is not None:
                v["e16/b/plan/minperepoch"] = str(x["plan"]["projected_min_per_epoch"])
        for part, files in (x.get("offline") or {}).items():
            for name, acc in files.items():
                v[f"{a}/{part}/{name.replace('_', '')}/acc"] = _pt(acc)
    rh = d.get("rehost") or {}
    if rh.get("available"):
        v["e16/rehost/agree"] = _pt(rh["agreement"])
        v["e16/rehost/usdk"] = f"{1000 * rh['usd_per_decision_rehost']:.4f}"
        if rh.get("host_factor"):
            v["e16/rehost/factor"] = f"{rh['host_factor']:.2f}"
    for m, row in (d.get("fast") or {}).items():
        if not isinstance(row, dict) or row.get("usd_per_1000_wall") is None:
            continue
        v[f"e16/fast/{m}/usdkwall"] = f"{row['usd_per_1000_wall']:.4f}"
        v[f"e16/fast/{m}/usdke9"] = f"{row['usd_per_1000_e9_rule']:.4f}"
        v[f"e16/fast/{m}/msdec"] = f"{1000 * row['gpu_s_per_decision']:.1f}"
        for w, s in (row.get("by_workers") or {}).items():
            v[f"e16/fast/{m}/c{w}/p50"] = f"{s['latency_ms_median']:.0f}"
            v[f"e16/fast/{m}/c{w}/usdkwall"] = f"{s['usd_per_1000_wall']:.4f}"
        if row.get("weights_gb") is not None:
            v[f"e16/fast/{m}/gb"] = f"{row['weights_gb']:.1f}"
        if row.get("agreement_with_served") is not None:
            v[f"e16/fast/{m}/agree"] = _pt(row["agreement_with_served"])
    sp = d.get("spend") or {}
    if sp:
        v["e16/spend"] = f"{sp['total']:.2f}"
    h = d.get("host") or {}
    if h.get("host_hours") is not None:
        v["e16/gpuh"] = f"{h['host_hours']:.1f}"
    it = (h.get("setup") or {}).get("instance_type") or (h.get("host") or {}).get("instance_type")
    if it:
        v["e16/host/type"] = it
    return v


def markdown(d: Mapping[str, Any], v: Mapping[str, str]) -> str:
    g = lambda k: v.get(k, "--")  # noqa: E731
    best = d.get("best_open") or {}
    L = ["# E16: the best open-weights decision model on all three levels", "", f"_{d['label']}_", "",
         "L1 = E1's 3,991 fixed decisions (accuracy; $ per 1,000 by E9's rule); L2 = the common states C (5,314 "
         "occurrences, task means); L3 = REPLACE at the tool choice, Flash, the first 200 fresh Stage-2 tasks (success).",
         "", "## The best open-weights decision model we did not train, per level (registration (d) 4)", ""]
    for lvl in ("l1", "l2", "l3"):
        b = best.get(lvl)
        L.append(f"- {lvl.upper()}: {b['label']} {_pt(b['value'])} % (then {', '.join(b['ranking'][1:4])})" if b else f"- {lvl.upper()}: --")
    L += ["", "## The arms", "", "| arm | L1 all | tool choice / arguments / When2Call | L2 (C) | L3 (live) | best open L1 / L2 / L3 (gap) |",
          "|---|---|---|---|---|---|"]
    for arm, x in d["arms"].items():
        a = f"e16/{arm}"
        bo = " / ".join(f"{g(f'{a}/best_open/{l}')} ({g(f'{a}/best_open/{l}/gap')})" for l in ("l1", "l2", "l3"))
        L.append(f"| {x['name']} | {g(a + '/e1/acc')} | {g(a + '/e1/acc/toolchoice')} / {g(a + '/e1/acc/args')} / "
                 f"{g(a + '/e1/acc/w2c')} | {g(a + '/c/acc')} | {g(a + '/live/success')} (n {g(a + '/live/n')}) | {bo} |")
    L += ["", "Reference rows: AgentDecision-4B L1 " + g("e16/row/frozen/l1") + " / L2 " + g("e16/row/frozen/l2") + " / L3 "
          + g("e16/row/frozen/l3") + "; E12 " + g("e16/row/e12/l1") + " / " + g("e16/row/e12/l2") + " / " + g("e16/row/e12/l3")
          + "; E12c " + g("e16/row/e12c/l1") + " / " + g("e16/row/e12c/l2") + " / " + g("e16/row/e12c/l3") + ".", ""]
    for arm, x in d["arms"].items():
        a = f"e16/{arm}"
        L += [f"## Arm {arm.upper()}: {x['name']}", ""]
        sel = (x.get("selection") or {}).get("selected")
        if sel:
            L.append(f"Selected epoch {sel['epoch']} (dev prefix {_pt(sel['acc'].get('e1_devprefix'))} %, v1 dev "
                     f"{_pt(sel['acc'].get('dev'))} %); candidates: " + "; ".join(
                         f"epoch {c['epoch']}: {_pt(c['acc'].get('e1_devprefix'))} / {_pt(c['acc'].get('dev'))}"
                         for c in x["selection"]["candidates"]) + f". Training {g(a + '/train/min')} min ({g(a + '/train/setting')}).")
        if arm == "b" and x.get("plan"):
            L.append(f"Plan: {json.dumps(x['plan'])}")
        e1 = x.get("e1") or {}
        if e1.get("available"):
            L.append(f"L1: {g(a + '/e1/acc')} % [{g(a + '/e1/acc/lo')}, {g(a + '/e1/acc/hi')}]; ${g(a + '/e1/usdk')} per 1,000 "
                     f"(E9's rule, this host), median {g(a + '/e1/lat')} ms. When2Call by gold label: "
                     + ", ".join(f"{k[:4]} {_pt(y['acc'])} (n {y['n']})" for k, y in e1["w2c_by_label"].items())
                     + f". Confusion (gold->pred): {e1.get('w2c_confusion')}.")
            for ref, gg in e1["vs"].items():
                L.append(f"- vs {ref} ({gg.get('ref_key')}, {_pt(gg['ref_acc'])} %): {g(f'{a}/e1/vs/{ref}/all')} "
                         f"[{g(f'{a}/e1/vs/{ref}/all/lo')}, {g(f'{a}/e1/vs/{ref}/all/hi')}]; tool choice "
                         f"{g(f'{a}/e1/vs/{ref}/toolchoice')}, arguments {g(f'{a}/e1/vs/{ref}/args')}, When2Call {g(f'{a}/e1/vs/{ref}/w2c')}")
        else:
            L.append(f"L1: not available ({e1.get('reason')}).")
        c = x.get("c") or {}
        if c.get("available"):
            L.append(f"L2: {g(a + '/c/acc')} % (after off-plan {g(a + '/c/offplan')}, ask-vs-act {g(a + '/c/ask')}); vs "
                     f"AgentDecision-4B {g(a + '/c/delta')} [{g(a + '/c/delta/lo')}, {g(a + '/c/delta/hi')}], non-inferior at 1 point: "
                     f"{g(a + '/c/ni')}; vs the best open model {g(a + '/c/vsbestopen')} [{g(a + '/c/vsbestopen/lo')}, "
                     f"{g(a + '/c/vsbestopen/hi')}]; live gate: {g(a + '/live/gate')}; missing {c['missing']}, errors {c['errors']}.")
        else:
            L.append("L2: not available.")
        lv = x.get("live") or {}
        if lv.get("available"):
            L.append(f"L3: {g(a + '/live/success')} % (n {g(a + '/live/n')}; own-state accuracy {g(a + '/live/own')} %, "
                     f"{g(a + '/live/steps')} steps, ${g(a + '/live/usdep')} per episode, {g(a + '/live/walls')} s); "
                     + "; ".join(f"vs {r} {g(f'{a}/live/vs/{r}')} [{g(f'{a}/live/vs/{r}/lo')}, {g(f'{a}/live/vs/{r}/hi')}]" for r in lv["vs"]))
        else:
            L.append(f"L3: not run (gate {g(a + '/live/gate')}).")
        L.append("")
    L += ["## Same-engine timing (vLLM, merged LoRA; $ per 1,000 decisions, wall clock, four states in flight)", ""]
    for m, row in (d.get("fast") or {}).items():
        if isinstance(row, dict) and row.get("usd_per_1000_wall") is not None:
            L.append(f"- {m}: {g(f'e16/fast/{m}/usdkwall')} (E9's rule on this engine {g(f'e16/fast/{m}/usdke9')}; "
                     f"{g(f'e16/fast/{m}/msdec')} ms per decision; p50 {g(f'e16/fast/{m}/c4/p50')} ms at 4, "
                     f"{g(f'e16/fast/{m}/c1/p50')} ms at 1; agreement with the served answers {g(f'e16/fast/{m}/agree')} %; "
                     f"{g(f'e16/fast/{m}/gb')} GB)")
        elif isinstance(row, dict):
            L.append(f"- {m}: {row.get('state')}")
    rh = d.get("rehost") or {}
    if rh.get("available"):
        L.append(f"\nServing-path check: the frozen checkpoint re-served on the E16 host agrees with its logged E1 answers on "
                 f"{g('e16/rehost/agree')} % (rule >= 99 %: {'ok' if rh['ok'] else 'FAILED'}); same-host cost ${g('e16/rehost/usdk')} "
                 f"per 1,000 (host factor {g('e16/rehost/factor')}).")
    L += ["", f"Spend: ${g('e16/spend')} of Gemini (cap ${E16.CAP_USD:.0f}). Host: {g('e16/host/type')}, {g('e16/gpuh')} host-hours.",
          "", f"Generated {d['generated']}."]
    return "\n".join(L) + "\n"


def write_tex(vals: Mapping[str, str], path: Path = E16.TEX) -> Path:
    import explore_common as C

    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["% AUTO-GENERATED by scripts/explore_e16_best_open.py summarize -- do not edit by hand.",
         f"% {E16.LABEL}", f"% last written {_now()} at git {C.git_head()}",
         "\\makeatletter",
         "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
         "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    for k in sorted(vals):
        val = str(vals[k]).replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")
        if val.startswith("-") and val[1:2].isdigit():
            val = "$-$" + val[1:]
        L.append(f"\\defres{{{k}}}{{{val}}}")
    L.append("\\makeatother")
    path.write_text("\n".join(L) + "\n")
    return path
