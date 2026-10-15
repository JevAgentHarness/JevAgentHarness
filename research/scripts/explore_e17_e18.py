#!/usr/bin/env python3
"""Exploratory rows E17 and E18 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E17 and E18: read-out ablation and
a deferral-aware objective"; descriptive, outside every Holm table).

Operations (the E17/E18 host, scripts/aws/e17_e18.sh; E14's probe / watchdog / gate / spend commands on these runs):

    python scripts/explore_e17_e18.py probe --log A/api_probe_log.txt      # exit 0 iff 6/6
    python scripts/explore_e17_e18.py watch --pid PID --cap 8 --runs runs --stop runs/e18_STOP.json
    python scripts/explore_e17_e18.py errored --run runs/e18_live | spend --runs runs [--short]
    python scripts/explore_e17_e18.py gate --runs runs --smoke runs/e18_live_smoke --n 200 --cap 8
    python scripts/explore_e17_e18.py smoke-file --mix D/gen/train_mix.jsonl --out D/gen/smoke.jsonl
    python scripts/explore_e17_e18.py projection --log LOG --rows 16902 --cap-min 170 --setting S --out A/gen_projection.json
    python scripts/explore_e17_e18.py check-profile                        # tau = 1 - k in the E18 profile

Analysis (local, $0):

    python scripts/explore_e17_e18.py fit-head      # the defer head from logged records -> results/explore/e17_e18/e18_head.json
    python scripts/explore_e17_e18.py prepare-live  # configs/harness/explore_e18_final.yaml (tau = 1 - k per point)
    python scripts/explore_e17_e18.py data-sha --data D   # results/explore/e17_e18/data_sha256.json of the built records
    python scripts/explore_e17_e18.py fit-gen       # the generative arm's calibration fit (E5's (g)) from its answers
    python scripts/explore_e17_e18.py collect       # slim records from the synced box -> results/explore/e17_e18/records/
    python scripts/explore_e17_e18.py summarize     # -> results/explore/e17_e18/summary.{md,json} + paper/generated/e17_e18.tex
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import explore_e14_small as E14  # noqa: E402

OUT = ROOT / "results/explore/e17_e18"
REC = OUT / "records"
TEX = ROOT / "paper/generated/e17_e18.tex"
BOX = ROOT / "runs/e17_e18/box"                    # s3://.../e17_e18/box synced here (untracked)
RUNS = ROOT / "runs"
S23 = ROOT / "results/stage2/s23"
E14REC = ROOT / "results/explore/e14_small/records"
E14CAL = ROOT / "results/explore/e14_small/cal"
HEAD = OUT / "e18_head.json"
PROFILE = ROOT / "configs/harness/explore_e18_final.yaml"
FINAL_PROFILE = ROOT / "configs/harness/jevagent_final.yaml"
DATA_SHA = OUT / "data_sha256.json"
CAL_SRC = ROOT / "runs/e2a_jevgym_shadow_cal_rl"
CAL_WORK = ROOT / "runs/e17_e18_cal"
CAL_OUT = OUT / "cal"
REF_ANSWERS = E14REC / "cal" / "answers_e14a.jsonl.gz"
LIVE_RUNS = ("e18_live_smoke", "e18_live")
E14.LIVE_RUNS = LIVE_RUNS                           # E14's watchdog / gate / spend read these run names
LABEL = ("exploratory (E17, E18), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E17 and E18: read-out "
         "ablation and a deferral-aware objective'; outside every Holm table")
DPS = ("tool_select", "tool_params")
TAU_A = {"tool_select": 0.9999999997650646, "tool_params": 0.9999999999989255}      # E14 arm A's fitted thresholds
TAU_RL = {"tool_select": 0.4999999999978404, "tool_params": 0.9999998807907104}     # the frozen 4B's
E14_LIVE_COV = {"tool": 0.855, "arg": 0.864}                                         # E14's e14_final row (reading (c) 7)
ROWS_PER_STEP = 64


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _jl(p: Path) -> list[dict[str, Any]]:
    return E14._jl(p)


# ================================================================================================ operations
def cmd_smoke_file(a: argparse.Namespace) -> int:
    """E5's smoke rows: the 48 longest prompts (by state length) plus 48 random other rows of the mix."""
    rows = [json.loads(x) for x in open(a.mix) if x.strip()]
    longest = sorted(rows, key=lambda r: -len(r.get("state", "")))[:48]
    rest = [r for r in rows if r not in longest]
    out = longest + random.Random(0).sample(rest, min(48, len(rest)))
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(len(out))
    return 0


def projection(log_text: str, rows: int, cap_min: float, setting: str, rows_per_step: int = ROWS_PER_STEP,
               smoke_steps: int = 4, max_epochs: int = 2) -> dict[str, Any]:
    """Registration (b) 2: the smoke's step time (train_runtime over its optimizer steps) projects the epochs: 2 when
    both fit within cap_min, else 1 when one does, else none. The amendment of 2026-10-08 06:10 UTC caps the epochs at
    one (``max_epochs``) within 280 minutes."""
    m = re.findall(r"'train_runtime':\s*'?([0-9.eE+-]+)", log_text)
    if not m:
        return {"not_run": "no train_runtime in the smoke log", "setting": setting}
    s_per_step = float(m[-1]) / smoke_steps
    steps = math.ceil(rows / rows_per_step)
    one = s_per_step * steps / 60
    epochs = 2 if 2 * one <= cap_min else 1 if one <= cap_min else 0
    epochs = min(epochs, int(max_epochs))
    return {"setting": setting, "s_per_step_smoke": s_per_step, "steps_per_epoch": steps, "minutes_per_epoch": one,
            "cap_min": cap_min, "max_epochs": int(max_epochs), "epochs": epochs, "projected_minutes": epochs * one,
            "rule": "2 epochs when both fit within cap_min, else 1, else not run (registration (b) 2); at most "
                    "max_epochs (the amendment of 2026-10-08 06:10 UTC: one epoch within 280 minutes)"}


def cmd_projection(a: argparse.Namespace) -> int:
    res = projection(Path(a.log).read_text(), a.rows, a.cap_min, a.setting, max_epochs=a.max_epochs)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0


def check_profile(profile: Path = PROFILE, head: Path = HEAD) -> dict[str, Any]:
    import yaml

    from jevharness.harness import profile as HP

    prof = HP.load_profile(profile)
    h = json.loads(head.read_text())
    blk = prof["system2"]["gemini-3.5-flash"]["backends"]["e18"]
    out = {"ok": True, "taus": {}}
    for dp in DPS:
        tau = float(blk[dp]["tau"])
        want = 1.0 - float(h["k"][dp])
        out["taus"][dp] = {"tau": tau, "k": h["k"][dp], "ok": abs(tau - want) < 1e-12}
        out["ok"] &= out["taus"][dp]["ok"]
    out["head_sha256"] = hashlib.sha256(head.read_bytes()).hexdigest()
    out["profile_head_sha256"] = blk.get("fitted_from", {}).get("head_sha256")
    out["ok"] &= out["head_sha256"] == out["profile_head_sha256"]
    yaml.safe_load(profile.read_text())
    return out


def cmd_check_profile(a: argparse.Namespace) -> int:
    r = check_profile()
    print(json.dumps(r))
    return 0 if r["ok"] else 1


# ================================================================================================ E18: the head
def cmd_fit_head(a: argparse.Namespace) -> int:
    from jevharness.training import defer_head as DH

    rows = DH.labelled_questions(CAL_SRC / "rl_shadow_cal.jsonl", Path(a.ref_answers))
    h = DH.fit_head(rows)
    h["sources"] = {"shadow_arm": str((CAL_SRC / "rl_shadow_cal.jsonl").relative_to(ROOT)),
                    "shadow_arm_sha256": hashlib.sha256((CAL_SRC / "rl_shadow_cal.jsonl").read_bytes()).hexdigest(),
                    "ref_answers": str(Path(a.ref_answers).relative_to(ROOT)) if Path(a.ref_answers).is_relative_to(ROOT) else a.ref_answers,
                    "ref_answers_sha256": hashlib.sha256(Path(a.ref_answers).read_bytes()).hexdigest(),
                    "fitted_utc": _now(), "label": LABEL}
    OUT.mkdir(parents=True, exist_ok=True)
    HEAD.write_text(json.dumps(h, indent=1) + "\n")
    (OUT / "e18_labelled.jsonl.gz").write_bytes(gzip.compress(
        "".join(json.dumps(r) + "\n" for r in rows).encode(), 9))
    print(json.dumps({"weights": h["weights"], "by_dp": h["by_dp"], "fit": h["fit"]}, indent=1))
    return 0


def cmd_prepare_live(a: argparse.Namespace) -> int:
    """backends.e18 of a copy of the final-harness profile: tau = 1 - k per decision point (registration (c) 3)."""
    import yaml

    from jevharness.harness import profile as HP

    h = json.loads(HEAD.read_text())
    prof = HP.load_profile(FINAL_PROFILE)
    prof["name"] = "explore_e18_final"
    prof["description"] = ("Exploratory row E18's live profile (PREREG_A7_STAGE2 Deviations 2026-10-08 'E17 and E18' (c) 3 "
                           "and 6): a copy of configs/harness/jevagent_final.yaml (tool + argument cascade, reflection off, "
                           "batch_s1; the frozen JevAgentRL-4B block unchanged) with backends.e18: tau = 1 - k per decision "
                           "point, so the cascade accepts iff the defer head's P(defer) <= k. No threshold was fitted or searched.")
    sec = prof["system2"]["gemini-3.5-flash"]
    sec["backends"]["e18"] = {dp: {"tau": 1.0 - float(h["k"][dp])} for dp in DPS}
    sec["backends"]["e18"]["fitted_from"] = {
        "head": str(HEAD.relative_to(ROOT)), "head_sha256": hashlib.sha256(HEAD.read_bytes()).hexdigest(),
        "rule": "served confidence = 1 - P(defer); tau = 1 - k_dp with k from results/gpu_round5/cal (the frozen "
                "4B's calibration fits: Delta c / V, V = W = $1); PREREG_A7_STAGE2 Deviations 2026-10-08 'E17 and E18' (c) 3",
        "k": {dp: float(h["k"][dp]) for dp in DPS}}
    for k in [k for k in prof if k.startswith("_")]:
        prof.pop(k)
    head = ["# Exploratory row E18's live profile (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E17 and E18: read-out",
            "# ablation and a deferral-aware objective' (c) 3 and (c) 6): configs/harness/jevagent_final.yaml copied unchanged",
            "# (tool + argument cascade, reflection off, batch_s1; the frozen JevAgentRL-4B block unchanged) plus backends.e18:",
            "# tau = 1 - k_dp (the cost margins of the frozen 4B's calibration fits), the served confidence being 1 - P(defer)",
            "# of the logistic defer head results/explore/e17_e18/e18_head.json. Written by scripts/explore_e17_e18.py",
            "# prepare-live before any live episode; nothing fitted or searched on any evaluation data."]
    PROFILE.write_text("\n".join(head) + "\n" + yaml.safe_dump(prof, sort_keys=False, allow_unicode=True))
    HP.load_profile(PROFILE)
    print(json.dumps(check_profile(), indent=1))
    return 0


def cmd_data_sha(a: argparse.Namespace) -> int:
    d = Path(a.data)
    sha = {}
    for f in sorted(d.rglob("*.jsonl.gz")) + sorted(d.rglob("build_*.json")):
        sha[str(f.relative_to(d))] = hashlib.sha256(f.read_bytes()).hexdigest()
    rec = {"label": LABEL, "built_utc": _now(), "sha256": sha}
    for name in ("a/build_a.json", "f/build_f.json"):
        p = d / name
        if p.exists():
            b = json.loads(p.read_text())
            rec[name] = {k: v for k, v in b.items() if k in ("train_mix_sha256", "rows", "labelled", "files")}
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_SHA.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(sha, indent=1))
    return 0


# ================================================================================================ E17: the gen fit
def cmd_fit_gen(a: argparse.Namespace) -> int:
    import e5_cal_replay as CR

    ans = Path(a.answers)
    if not ans.exists():
        raise SystemExit(f"no {ans}")
    d = CR.derive(CAL_SRC, "rl_shadow_cal", ans, CAL_WORK, "e17cal_gen", model="e17-gen")
    f = CR.fit(CAL_WORK, "e17cal_gen", CAL_OUT)
    print(json.dumps({"derive": d, "fit": f}, indent=1, default=str))
    return 0


def cal_cov(fits: Mapping[str, Mapping[str, Any]], path: Path) -> dict[str, Any]:
    """Coverage at the fitted thresholds on a derived calibration arm (E14's cal_coverage with explicit fits)."""
    from jevharness.bench.threshold_theory import decision_pairs, load_trajectories, paired

    if not path.exists():
        return {}
    trajs = list(load_trajectories(path))
    r: dict[str, Any] = {}
    for dp in DPS:
        if dp not in fits:
            continue
        t = fits[dp].get("tau_hat")
        tau = float(t) if t is not None and isinstance(t, (int, float)) and math.isfinite(t) and t <= 1.0 else float("inf")
        ps = paired(decision_pairs(trajs, dp))
        if not ps:
            continue
        acc = [x for x in ps if x["c"] is not None and x["c"] >= tau and x["acceptable"]]
        n = len(ps)
        casc = (sum(float(x["s1_ok"]) for x in acc) + sum(float(x["s2_ok"]) for x in ps if x not in acc)) / n
        r[dp] = {"tau": tau, "n_paired": n, "accepted": len(acc), "coverage": len(acc) / n,
                 "s1_acc_accepted": (sum(float(x["s1_ok"]) for x in acc) / len(acc)) if acc else None,
                 "s2_acc_accepted": (sum(float(x["s2_ok"]) for x in acc) / len(acc)) if acc else None,
                 "s1_acc_all": sum(float(x["s1_ok"]) for x in ps) / n, "s2_acc_all": sum(float(x["s2_ok"]) for x in ps) / n,
                 "cascade_acc": casc, "accepted_all_right": bool(acc) and all(x["s1_ok"] for x in acc),
                 "conf_quantiles": E14._quantiles([x["c"] for x in ps if x["c"] is not None])}
    return r


def _fits(d: Path, prefix: str) -> dict[str, Any]:
    out = {}
    for dp in DPS:
        fp = d / f"tauhat_{prefix}_{dp}.json"
        if fp.exists():
            f = json.loads(fp.read_text())
            out[dp] = {"tau_hat": f.get("tau_hat"), "n_paired": f.get("n_paired"), "k": (f.get("primary") or {}).get("delta_c"),
                       "tau_rule": (f.get("tau_rule") or {}).get("rule") if isinstance(f.get("tau_rule"), dict) else f.get("tau_rule")}
    return out


# ================================================================================================ summarize
def _answers(p: Path) -> dict[str, dict[str, Any]]:
    return {r["qhash"]: r for r in _jl(p)} if p.exists() else {}


def corpus_rows(box: Path) -> dict[str, Any]:
    import ablation_score as AS
    import e5_analysis as E5A

    occ, eligible = AS.load_corpus(S23)
    srcs = {"head": E14REC / "s23" / "answers_e14a.jsonl.gz", "gen": box / "s23" / "answers_e17gen.jsonl",
            "e18f": box / "s23" / "answers_e18f.jsonl", "rl": S23 / "answers_rl.jsonl", "sft_v1": S23 / "answers_sft_v1.jsonl"}
    rows, per = {}, {}
    for name, p in srcs.items():
        ans = _answers(p)
        if not ans:
            continue
        m, pt = E5A.c_rows(occ, eligible, ans)
        rows[name] = m
        per[name] = pt
    con = {}
    for k in ("gen", "e18f"):
        for ref in ("head", "rl", "sft_v1"):
            if k in per and ref in per:
                con[f"{k}_vs_{ref}"] = {st: E14._boot(per[k][st], per[ref][st]) for st in ("all", "offplan", "ask")}
    return {"rows": rows, "contrasts": con}


def _llm_tool_at(task: str, step: int, llm_recs: Mapping[str, Mapping[str, Any]]) -> str | None:
    r = llm_recs.get(task)
    if not r:
        return None
    for st in r.get("steps") or []:
        if int(st.get("step", -1)) == step:
            return st.get("tool")
    return None


def defer_on_c(answers: Mapping[str, Mapping[str, Any]], accept_of: Any, label: str) -> dict[str, Any]:
    """Coverage of a rule on C's occurrences: ``accept_of(answer, n_options) -> bool``; the accuracy of accepted and
    deferred decisions; on the llm_only occurrences the share of deferred ones the LLM got right."""
    import ablation_score as AS

    from jevharness.bench import stage2_corpus as C

    occ, eligible = AS.load_corpus(S23)
    keep = set(eligible)
    qs = {q["qhash"]: q for q in C.read_jsonl_gz(S23 / "questions.jsonl.gz")}
    llm_recs = C.load_latest(RUNS / "stage2_jevgym" / "llm_only.jsonl")
    n = acc = acc_ok = def_ok = 0
    llm_n = llm_def = llm_def_right = llm_acc_wrong_llm_right = 0
    per_task: dict[str, list[float]] = {}
    for o in occ:
        if o["task"] not in keep:
            continue
        a = answers.get(o["qhash"]) or {}
        n += 1
        ok = C.answer_correct(o, a)
        accept = bool(a) and not a.get("error") and a.get("confidence") is not None and accept_of(a, len(qs[o["qhash"]]["options"]))
        if accept:
            acc += 1
            acc_ok += ok
        else:
            def_ok += ok
        if o["source"] == "llm_only":
            llm_n += 1
            tool = _llm_tool_at(o["task"], int(o["step"]), llm_recs)
            llm_ok = bool(C.answer_correct(o, {"answer": tool})) if tool is not None else None
            if not accept:
                llm_def += 1
                llm_def_right += bool(llm_ok)
            elif not ok and llm_ok:
                llm_acc_wrong_llm_right += 1
            per_task.setdefault(o["task"], []).append(float(ok if accept else bool(llm_ok)))
    return {"label": label, "n": n, "accepted": acc, "coverage": acc / n if n else None,
            "accepted_accuracy": acc_ok / acc if acc else None, "deferred_accuracy": def_ok / (n - acc) if n - acc else None,
            "accepted_errors": acc - acc_ok, "llm_only": {"n": llm_n, "deferred": llm_def, "deferred_llm_right": llm_def_right,
                                                          "deferred_llm_right_share": llm_def_right / llm_def if llm_def else None,
                                                          "accepted_wrong_llm_right": llm_acc_wrong_llm_right,
                                                          "cascade_task_acc": (sum(sum(v) / len(v) for v in per_task.values()) / len(per_task)) if per_task else None}}


def e18_corpus(box: Path) -> dict[str, Any]:
    from jevharness.training.clef_engine import head_features
    from jevharness.training.defer_head import p_defer

    out: dict[str, Any] = {}
    a_ans = _answers(E14REC / "s23" / "answers_e14a.jsonl.gz")
    if HEAD.exists() and a_ans:
        h = json.loads(HEAD.read_text())
        k = float(h["k"]["tool_select"])
        w = h["weights"]
        out["e18"] = defer_on_c(a_ans, lambda a, n: p_defer(w, head_features(float(a["confidence"]), True, n)) <= k,
                                "arm A + the defer head: accept iff P(defer) <= k")
        out["e18"]["p_defer_quantiles"] = E14._quantiles([p_defer(w, head_features(float(a["confidence"]), True, 69))
                                                          for a in a_ans.values() if a.get("confidence") is not None])
        out["head_at_tauhat"] = defer_on_c(a_ans, lambda a, n: float(a["confidence"]) >= TAU_A["tool_select"],
                                           "arm A at its fitted tau-hat")
        out["head_replace"] = defer_on_c(a_ans, lambda a, n: True, "arm A, every decision accepted")
    rl = _answers(S23 / "answers_rl.jsonl")
    if rl:
        out["rl_frozen"] = defer_on_c(rl, lambda a, n: float(a["confidence"]) >= TAU_RL["tool_select"],
                                      "AgentDecision-4B at its frozen threshold")
    gen = _answers(box / "s23" / "answers_e17gen.jsonl")
    fits = _fits(CAL_OUT, "e17cal_gen")
    if gen and "tool_select" in fits and fits["tool_select"].get("tau_hat") is not None:
        t = float(fits["tool_select"]["tau_hat"])
        out["gen_at_tauhat"] = defer_on_c(gen, lambda a, n: float(a["confidence"]) >= t, "the generative arm at its fitted tau-hat")
    f = _answers(box / "s23" / "answers_e18f.jsonl")
    if f and HEAD.exists():
        k = float(json.loads(HEAD.read_text())["k"]["tool_select"])
        out["e18f"] = defer_on_c(f, lambda a, n: float(a["confidence"]) >= 1.0 - k, "E18-F: accept iff p_defer <= k")
    return out


def live(runs: Path = RUNS) -> dict[str, Any]:
    """The final harness with E18 against the logged dedicated rows and E14's e14_final on the same 200 tasks."""
    import numpy as np

    import explore_common as C
    import harness_final as HF

    p = runs / "e18_live" / "e18_final.jsonl"
    if not p.exists():
        return {}
    tar = json.loads(E14.TARIFF_DED.read_text())
    host_h = float(tar["host_usd_per_hour"])
    arms = {"e18_final": p, "llm_only": E14.LOGGED_B / "llm_only.jsonl", "harness_final": E14.LOGGED_B / "harness_final.jsonl",
            "e14_final": runs / "e14_live_b" / "e14_final.jsonl"}
    arms = {k: f for k, f in arms.items() if f.exists()}
    lat: dict[str, dict[str, dict[str, Any]]] = {}
    for k, f in arms.items():
        latest_, _ = C.load_latest([f])
        lat[k] = {t: r for t, r in latest_.items() if not r.get("error")}
    common = sorted(set.intersection(*(set(v) for v in lat.values())))
    res: dict[str, Any] = {"n_common": len(common), "host_usd_per_hour": host_h, "e9_usd_per_hour": E14.E9_USD_H,
                           "errored": {k: sum(1 for t, r in C.load_latest([arms[k]])[0].items() if r.get("error")) for k in arms},
                           "n_latest": {k: len(C.load_latest([arms[k]])[0]) for k in arms}, "arms": {}}
    per: dict[str, dict[str, Any]] = {}
    for k, d in lat.items():
        eps = [HF._episode_live(d[t]) for t in common]
        wall = np.array([e["wall_s"] for e in eps])
        api = np.array([e["cost"] for e in eps])
        succ = np.array([e["success"] for e in eps])
        gpu = k != "llm_only"
        q = host_h / 3600 * wall if gpu else np.zeros_like(wall)
        q9 = E14.E9_USD_H / 3600 * wall if gpu else np.zeros_like(wall)
        per[k] = {"succ": succ, "api": api, "tot": api + q, "tot9": api + q9, "wall": wall}
        tot = {x: sum(e[x] for e in eps) for x in ("tool_decisions", "tool_by_s1", "arg_decisions", "arg_by_s1")}
        res["arms"][k] = {"success": float(succ.mean()), "api_usd": float(api.mean()), "q_usd": float(q.mean()),
                          "q_usd_e9": float(q9.mean()), "usd_incl_gpu": float((api + q).mean()),
                          "usd_incl_gpu_e9": float((api + q9).mean()), "wall_s_mean": float(wall.mean()),
                          "wall_s_median": float(np.median(wall)),
                          "s1_s": float(np.mean([e["s1_s"] for e in eps])), "s2_s": float(np.mean([e["s2_s"] for e in eps])),
                          "s2_calls": float(np.mean([e["s2_calls"] for e in eps])),
                          "s1_requests": float(np.mean([e["s1_requests"] for e in eps])),
                          "steps": float(np.mean([e["steps"] for e in eps])),
                          "tool_coverage": tot["tool_by_s1"] / tot["tool_decisions"] if tot["tool_decisions"] else None,
                          "arg_coverage": tot["arg_by_s1"] / tot["arg_decisions"] if tot["arg_decisions"] else None}
    def _nc(d: Mapping[str, Any]) -> dict[str, Any]:   # Newcombe (points) -> fractions {delta, lo, hi}
        return {"delta": d["diff_pt"] / 100, "lo": d["ci_pt"][0] / 100, "hi": d["ci_pt"][1] / 100, "cells": d.get("cells")}

    def _pb(d: Mapping[str, Any]) -> dict[str, Any]:   # paired bootstrap {est, ci} -> {delta, lo, hi}
        return {"delta": d["est"], "lo": d["ci"][0], "hi": d["ci"][1], "n": d.get("n")}

    con: dict[str, Any] = {}
    for ref in ("llm_only", "harness_final", "e14_final"):
        if ref not in per:
            continue
        a, b = per["e18_final"], per[ref]
        con[ref] = {"success": _nc(C.newcombe(a["succ"], b["succ"])), "success_boot": _pb(C.paired_boot(a["succ"], b["succ"])),
                    "saving_incl_gpu": _pb(C.paired_boot(a["tot"], b["tot"], ratio=True)),
                    "saving_incl_gpu_e9": _pb(C.paired_boot(a["tot9"], b["tot9"], ratio=True)),
                    "saving_api": _pb(C.paired_boot(a["api"], b["api"], ratio=True)), "wall_s": _pb(C.paired_boot(a["wall"], b["wall"]))}
    res["contrasts"] = con
    return res


def training(box: Path) -> dict[str, Any]:
    import e5_box as EB

    out: dict[str, Any] = {}
    runs = [json.loads(x) for x in (box / "train_runs.jsonl").read_text().splitlines() if x.strip()] if (box / "train_runs.jsonl").exists() else []
    if (box / "gen_projection.json").exists():
        out["gen"] = {"projection": json.loads((box / "gen_projection.json").read_text()),
                      "runs": [r for r in runs if r.get("arm") == "gen"],
                      "peak_mib": EB.peak_mib(box / "logs" / "train_smi_gen.csv") if (box / "logs" / "train_smi_gen.csv").exists() else None,
                      "readout_check": json.loads((box / "gen_readout.json").read_text()) if (box / "gen_readout.json").exists() else None}
        log = box / "logs" / "train_gen.log"
        if log.exists():
            m = re.findall(r"'train_runtime':\s*'?([0-9.eE+-]+)", log.read_text())
            out["gen"]["train_runtime_s"] = float(m[-1]) if m else None
            m2 = re.findall(r"'train_loss':\s*'?([0-9.eE+-]+)", log.read_text())
            out["gen"]["train_loss"] = float(m2[-1]) if m2 else None
    p = box / "train_record_F.json"
    if p.exists():
        r = json.loads(p.read_text())
        out["e18f"] = {k: r.get(k) for k in ("build", "train_records", "dev", "base_dev", "calibration_dev", "temperature",
                                            "head_temperature", "minutes_total", "peak_reserved_gb", "gpu", "oom")}
        out["e18f"]["epochs"] = [{k: e.get(k) for k in ("tag", "minutes", "s_per_step", "steps", "loss_first_last", "dev",
                                                        "peak_alloc_gb")} for e in r.get("epochs", [])]
        out["e18f"]["peak_mib"] = EB.peak_mib(box / "logs" / "train_smi_F.csv") if (box / "logs" / "train_smi_F.csv").exists() else None
    if (box / "e18f_not_run.json").exists():
        out["e18f_not_run"] = json.loads((box / "e18f_not_run.json").read_text())
    return out


def v1(box: Path) -> dict[str, Any]:
    import ablation_score as AS

    out = {}
    for name, p in (("head", E14REC / "eval" / "e14a_v1.json"), ("gen", box / "eval" / "gen_v1.json"),
                    ("gen_raw", box / "eval" / "gen_v1_raw.json"), ("e18f", box / "eval" / "e18f_v1.json")):
        if p.exists():
            out[name] = AS.v1_metrics(p)
    e14 = json.loads((ROOT / "results/explore/e14_small/summary.json").read_text()) if (ROOT / "results/explore/e14_small/summary.json").exists() else {}
    out["refs"] = (e14.get("v1") or {}).get("refs") or {}
    return out


def calibration(box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fa = _fits(E14CAL, "e14cal_e14a")
    if fa:
        out["head"] = {"fits": fa, "coverage": cal_cov(fa, E14.CAL_WORK / "e14cal_e14a.jsonl")}
    fg = _fits(CAL_OUT, "e17cal_gen")
    if fg:
        out["gen"] = {"fits": fg, "coverage": cal_cov(fg, CAL_WORK / "e17cal_gen.jsonl")}
    out["4b_logged"] = {"coverage": cal_cov({dp: {"tau_hat": TAU_RL[dp]} for dp in DPS}, CAL_SRC / "rl_shadow_cal.jsonl")}
    if (box / "eval" / "e18f_devread.json").exists():
        out["e18f_devread"] = json.loads((box / "eval" / "e18f_devread.json").read_text())
    return out


def latency(box: Path) -> dict[str, Any]:
    import e5_analysis as E5A

    if not (box / "latency").exists():
        return {}
    labels = ["4b"] + [k for k in ("head", "gen") if (box / "latency" / "bench" / f"bench_{k}_new_c1.json").exists()
                       or (box / "latency" / "bench_compact" / f"bench_{k}_new_c1.json").exists()]
    return E5A.latency(box, labels)


def readings(r: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cal = r.get("calibration") or {}
    for dp in DPS:
        cands = {}
        for name in ("head", "gen"):
            c = ((cal.get(name) or {}).get("coverage") or {}).get(dp)
            if c:
                cands[name] = c
        if cands:
            clean = {k: v for k, v in cands.items() if v.get("accepted_all_right")}
            win = max(clean, key=lambda k: clean[k]["coverage"]) if clean else None
            if win and len(clean) > 1 and sum(1 for k in clean if clean[k]["coverage"] == clean[win]["coverage"]) > 1:
                win = "tie"
            out[f"e17/{dp}"] = {"winner": win, "coverage": {k: v["coverage"] for k, v in cands.items()},
                                "accepted_all_right": {k: bool(v.get("accepted_all_right")) for k, v in cands.items()},
                                "tau": {k: v["tau"] for k, v in cands.items()}}
    lv = (r.get("live") or {}).get("arms") or {}
    if "e18_final" in lv and "llm_only" in lv:
        e, l = lv["e18_final"], lv["llm_only"]
        d = 100 * (e["success"] - l["success"])
        cov_ok = (e.get("tool_coverage") or 0) > E14_LIVE_COV["tool"] and (e.get("arg_coverage") or 0) > E14_LIVE_COV["arg"]
        out["e18/live"] = {"holds": bool(cov_ok and d >= -3.0), "success_minus_llm_pt": d, "tool_coverage": e.get("tool_coverage"),
                           "arg_coverage": e.get("arg_coverage"), "e14_coverage": E14_LIVE_COV, "coverage_higher": cov_ok}
    c = r.get("e18_corpus") or {}
    if "e18" in c and "head_at_tauhat" in c:
        a, b = c["e18"], c["head_at_tauhat"]
        out["e18/c"] = {"coverage": a["coverage"], "a_coverage": b["coverage"], "accepted_errors": a["accepted_errors"],
                        "a_accepted_errors": b["accepted_errors"],
                        "holds": bool(a["coverage"] > b["coverage"] and a["accepted_errors"] <= b["accepted_errors"])}
    return out


def summarize(box: Path = BOX) -> dict[str, Any]:
    res: dict[str, Any] = {"label": LABEL, "generated": _now(), "box": str(box)}
    for name, fn in (("training", lambda: training(box)), ("v1", lambda: v1(box)), ("corpus", lambda: corpus_rows(box)),
                     ("calibration", lambda: calibration(box)), ("latency", lambda: latency(box)),
                     ("e18_head", lambda: json.loads(HEAD.read_text()) if HEAD.exists() else {}),
                     ("e18_corpus", lambda: e18_corpus(box)), ("live", live)):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001 -- a missing block is reported, the rest still summarized
            res[name] = {"error": f"{type(e).__name__}: {e}"}
    for k in ("host.json", "host_final.json", "stage_times.jsonl", "arm_a_sha256.txt", "env_e14venv.txt", "env_v2venv.txt",
              "live_window.json", "gen_projection.json"):
        p = box / k
        if p.exists():
            if k.endswith((".txt", ".jsonl")):
                v: Any = p.read_text()
            else:
                try:
                    v = json.loads(p.read_text())
                except json.JSONDecodeError:   # live_window.json holds two JSON lines (start, then end): merged
                    v = {}
                    for line in p.read_text().splitlines():
                        if line.strip():
                            v.update(json.loads(line))
            res.setdefault("host", {})[k] = v
    res["readings"] = readings(res)
    res["gpu_hours"] = E14._gpu_hours(box)
    if DATA_SHA.exists():   # the shared training rows of both read-outs (SFT v1's train_mix, checked on the host)
        b = json.loads(DATA_SHA.read_text()).get("a/build_a.json") or {}
        res["data"] = {"rows": b.get("rows"), "train_mix_sha256": b.get("train_mix_sha256")}
    res["spend"] = E14.total_spend(RUNS)
    return res


# ------------------------------------------------------------------ outputs
def _pc(x: Any, nd: int = 1) -> str:
    return E14._pc(x, nd)


def _nm(x: Any, nd: int = 2) -> str:
    return E14._nm(x, nd)


def _g(d: Any, *ks: Any) -> Any:
    return E14._g(d, *ks)


def markdown(r: Mapping[str, Any]) -> str:
    L = [f"# E17 and E18: read-out ablation and a deferral-aware objective (exploratory; descriptive)", "", r["label"], "",
         f"Generated {r['generated']} from `{r['box']}`; GPU host hours used {r.get('gpu_hours')}; Gemini spend logged "
         f"${_g(r, 'spend', 'total') or 0:.2f}.", ""]
    tr = r.get("training") or {}
    L += ["## E17 training (the generative arm; the head is E14 arm A, not retrained)", ""]
    g = tr.get("gen") or {}
    if g:
        pj = g.get("projection") or {}
        L += [f"- projection: {json.dumps(pj)}", f"- runs: {json.dumps(g.get('runs'))}; train_runtime_s {g.get('train_runtime_s')}; "
              f"train_loss {g.get('train_loss')}; training peak MiB {g.get('peak_mib')}", f"- read-out check: {json.dumps(g.get('readout_check'))}", ""]
    v = r.get("v1") or {}
    L += ["## SFT v1 development files", "", "| model | v1 acc | tool choice | arguments | stop | after off-plan | knockout | ECE |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ("head", "gen", "gen_raw", "e18f"):
        m = v.get(k)
        if m:
            L.append(f"| {k} | {_pc(m.get('v1_acc'))} | {_pc(m.get('v1_tool_select'))} | {_pc(m.get('v1_tool_params'))} | "
                     f"{_pc(m.get('v1_stop'))} | {_pc(m.get('v1_recovery'))} | {_pc(m.get('v1_knockout'))} | {_nm(m.get('v1_ece'), 3)} |")
    co = r.get("corpus") or {}
    L += ["", "## The common states C", "", "| model | C acc | after off-plan | ask-vs-act | ECE |", "|---|---|---|---|---|"]
    for k, m in (co.get("rows") or {}).items():
        L.append(f"| {k} | {_pc(m.get('c_acc'))} | {_pc(m.get('c_acc_offplan'))} | {_pc(m.get('c_acc_ask'))} | {_nm(m.get('c_ece'), 3)} |")
    for c, x in (co.get("contrasts") or {}).items():
        L.append(f"- {c}: " + "; ".join(f"{st} {x[st]['delta'] * 100:+.1f} [{x[st]['lo'] * 100:+.1f}, {x[st]['hi'] * 100:+.1f}]" for st in ("all", "offplan", "ask")))
    cal = r.get("calibration") or {}
    L += ["", "## Calibration states (E5's procedure): tau-hat and coverage", "",
          "| read-out | point | tau-hat | paired | coverage | S1 acc (taken) | all taken right | LLM acc (taken) | cascade acc |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in ("head", "gen", "4b_logged"):
        for dp, c in ((cal.get(k) or {}).get("coverage") or {}).items():
            L.append(f"| {k} | {dp} | {c['tau'] if c['tau'] != float('inf') else 'always defers'} | {c['n_paired']} | {_pc(c['coverage'])} | "
                     f"{_pc(c.get('s1_acc_accepted'))} | {c.get('accepted_all_right')} | {_pc(c.get('s2_acc_accepted'))} | {_pc(c['cascade_acc'])} |")
    lat = r.get("latency") or {}
    L += ["", "## Serving (one session, one A10G)", "", "| served | c=1 p50 | c=1 p90 | c=4 p50 | c=4 p90 | S1/step c=1 p50 | peak MiB |",
          "|---|---|---|---|---|---|---|"]
    for k in ("4b", "head", "gen"):
        x = lat.get(k)
        if x:
            L.append(f"| {k} | {_nm(_g(x, 'bench', 'c1', 'p50_ms'), 0)} | {_nm(_g(x, 'bench', 'c1', 'p90_ms'), 0)} | "
                     f"{_nm(_g(x, 'bench', 'c4', 'p50_ms'), 0)} | {_nm(_g(x, 'bench', 'c4', 'p90_ms'), 0)} | "
                     f"{_nm(_g(x, 'bench', 'c1', 's1_ms_per_step_p50'), 0)} | {x.get('serve_peak_mib')} |")
    h = r.get("e18_head") or {}
    L += ["", "## E18: the defer head", ""]
    if h:
        L += [f"- weights {h.get('weights')} on {h.get('features')}; fit {json.dumps(h.get('fit'))}", f"- by decision point: {json.dumps(h.get('by_dp'))}", ""]
    ec = r.get("e18_corpus") or {}
    L += ["| rule on C | n | coverage | accepted acc | accepted errors | deferred acc | llm_only: deferred | LLM right among them |",
          "|---|---|---|---|---|---|---|---|"]
    for k, x in ec.items():
        lo = x.get("llm_only") or {}
        L.append(f"| {x.get('label', k)} | {x['n']} | {_pc(x['coverage'])} | {_pc(x.get('accepted_accuracy'))} | {x.get('accepted_errors')} | "
                 f"{_pc(x.get('deferred_accuracy'))} | {lo.get('deferred')} | {_pc(lo.get('deferred_llm_right_share'))} |")
    lv = r.get("live") or {}
    if lv.get("arms"):
        L += ["", f"## Live: the final harness with E18 (seeds 300500-300699; {lv['n_common']} common tasks; errored {json.dumps(lv.get('errored'))})", "",
              "| arm | success | API $ / ep | q $ / ep | $ / ep incl. GPU | wall s | S1 s | S2 s | S2 calls | tool cov. | arg cov. |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, x in lv["arms"].items():
            L.append(f"| {k} | {_pc(x['success'])} | {x['api_usd']:.4f} | {x['q_usd']:.5f} | {x['usd_incl_gpu']:.4f} | {x['wall_s_mean']:.2f} | "
                     f"{x['s1_s']:.2f} | {x['s2_s']:.2f} | {x['s2_calls']:.2f} | {_pc(x.get('tool_coverage'))} | {_pc(x.get('arg_coverage'))} |")
        for ref, c in (lv.get("contrasts") or {}).items():
            L.append(f"- vs {ref}: success {c['success']['delta'] * 100:+.1f} [{c['success']['lo'] * 100:+.1f}, {c['success']['hi'] * 100:+.1f}]; "
                     f"saving incl. GPU {c['saving_incl_gpu']['delta'] * 100:.1f} % [{c['saving_incl_gpu']['lo'] * 100:.1f}, {c['saving_incl_gpu']['hi'] * 100:.1f}]; "
                     f"wall {c['wall_s']['delta']:+.2f} s")
    L += ["", "## Registered readings", ""]
    for k, x in (r.get("readings") or {}).items():
        L.append(f"- {k}: {json.dumps(x)}")
    return "\n".join(L)


def _k(v: Any, nd: int = 1, pct: bool = True) -> str | None:
    return E14._k(v, nd, pct)


def _n(v: Any) -> str | None:
    """A count with TeX thousands separators (1{,}125), as the other generated files write them."""
    return None if v is None else f"{int(v):,}".replace(",", "{,}")


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str | None] = {}
    v = r.get("v1") or {}
    for name in ("head", "gen", "gen_raw", "e18f"):
        m = v.get(name)
        if not m:
            continue
        key = {"gen_raw": "genraw"}.get(name, name)
        for f, n in (("v1_acc", "acc"), ("v1_tool_select", "ts"), ("v1_tool_params", "tp"), ("v1_stop", "stop"),
                     ("v1_recovery", "rec"), ("v1_knockout", "ko")):
            k[f"e17/{key}/v1/{n}"] = _k(m.get(f))
        k[f"e17/{key}/v1/ece"] = _k(m.get("v1_ece"), 3, False)
    co = r.get("corpus") or {}
    for name, m in (co.get("rows") or {}).items():
        k[f"e17/{name}/c/acc"] = _k(m.get("c_acc"))
        k[f"e17/{name}/c/offplan"] = _k(m.get("c_acc_offplan"))
        k[f"e17/{name}/c/ask"] = _k(m.get("c_acc_ask"))
        k[f"e17/{name}/c/ece"] = _k(m.get("c_ece"), 3, False)
    for c, x in (co.get("contrasts") or {}).items():
        a, ref = c.split("_vs_")
        for st in ("all", "offplan", "ask"):
            k[f"e17/{a}/c/vs{ref.replace('_', '')}/{st}"] = _k(x[st]["delta"])
            k[f"e17/{a}/c/vs{ref.replace('_', '')}/{st}/lo"] = _k(x[st]["lo"])
            k[f"e17/{a}/c/vs{ref.replace('_', '')}/{st}/hi"] = _k(x[st]["hi"])
    cal = r.get("calibration") or {}
    for name in ("head", "gen", "4b_logged"):
        key = {"4b_logged": "fourb"}.get(name, name)
        for dp, c in ((cal.get(name) or {}).get("coverage") or {}).items():
            d = {"tool_select": "ts", "tool_params": "tp"}[dp]
            k[f"e17/{key}/cal/{d}/tau"] = ("always defers" if c["tau"] == float("inf") else f"{c['tau']:.12f}")
            k[f"e17/{key}/cal/{d}/taushort"] = ("always defers" if c["tau"] == float("inf") else f"{c['tau']:.4f}")
            k[f"e17/{key}/cal/{d}/cov"] = _k(c["coverage"])
            k[f"e17/{key}/cal/{d}/acc"] = _k(c.get("s1_acc_accepted"))
            k[f"e17/{key}/cal/{d}/llmacc"] = _k(c.get("s2_acc_accepted"))
            k[f"e17/{key}/cal/{d}/casc"] = _k(c["cascade_acc"])
            k[f"e17/{key}/cal/{d}/n"] = _n(c["n_paired"])
            k[f"e17/{key}/cal/{d}/clean"] = "yes" if c.get("accepted_all_right") else "no"
            k[f"e17/{key}/cal/{d}/errors"] = str(int(round((1 - (c.get("s1_acc_accepted") or 1)) * c["accepted"])))
    lat = r.get("latency") or {}
    for name in ("4b", "head", "gen"):
        x = lat.get(name)
        if not x:
            continue
        key = {"4b": "fourb"}.get(name, name)
        for c in ("c1", "c4"):
            k[f"e17/lat/{key}/{c}/p50"] = _k(_g(x, "bench", c, "p50_ms"), 0, False)
            k[f"e17/lat/{key}/{c}/p90"] = _k(_g(x, "bench", c, "p90_ms"), 0, False)
            k[f"e17/lat/{key}/{c}/s1step"] = _k(_g(x, "bench", c, "s1_ms_per_step_p50"), 0, False)
        k[f"e17/lat/{key}/mem"] = _k((x.get("serve_peak_mib") or 0) / 1024, 1, False) if x.get("serve_peak_mib") else None
        sp = x.get("speedup_vs_4b") or {}
        for c in ("c1_p50", "c4_p50"):
            if sp.get(c):
                k[f"e17/lat/{key}/speedup/{c.replace('_p50', '')}"] = _k(sp[c], 1, False)
    for c in ("c1", "c4"):   # the head's median per request against the one-token read-out's (same session)
        hp, gp = _g(lat, "head", "bench", c, "p50_ms"), _g(lat, "gen", "bench", c, "p50_ms")
        if hp and gp:
            k[f"e17/lat/head/vsgen/{c}"] = _k(gp / hp, 1, False)
    if _g(r, "data", "rows"):
        k["e17/rows"] = _n(_g(r, "data", "rows"))
    tr = r.get("training") or {}
    g = tr.get("gen") or {}
    if g:
        pj = g.get("projection") or {}
        k["e17/gen/train/epochs"] = str(pj.get("epochs")) if pj.get("epochs") is not None else None
        k["e17/gen/train/minutes"] = _k(((g.get("runs") or [{}])[-1]).get("minutes"), 0, False)
        k["e17/gen/train/setting"] = str(pj.get("setting")) if pj.get("setting") else None
        k["e17/gen/train/mem"] = _k((g.get("peak_mib") or 0) / 1024, 1, False) if g.get("peak_mib") else None
        k["e17/gen/train/sperstep"] = _k(pj.get("s_per_step_smoke"), 1, False)
        k["e17/gen/train/loss"] = _k(g.get("train_loss"), 3, False)
    f = tr.get("e18f") or {}
    if f:
        k["e18/f/train/records"] = _n(f.get("train_records"))
        k["e18/f/train/minutes"] = _k(f.get("minutes_total"), 0, False)
        k["e18/f/train/devcal"] = _k(_g(f, "calibration_dev", "accuracy"))
        for e in f.get("epochs") or []:
            k[f"e18/f/train/{e['tag']}/dev"] = _k(_g(e, "dev", "accuracy"))
    # E18-F with the option (registration (c) 4): coverage, accepted and deferred accuracy on the development records
    # (dev_f) and on its own labelled calibration-state records (cal_f, in its training data)
    dr = _g(r, "calibration", "e18f_devread", "files") or {}
    for fname, fk in (("dev_f", "dev"), ("cal_f", "cal")):
        for dp, x in (dr.get(fname) or {}).items():
            d = {"tool_select": "ts", "tool_params": "tp"}.get(dp)
            if not d:
                continue
            k[f"e18/f/{fk}/{d}/n"] = _n(x.get("n"))
            k[f"e18/f/{fk}/{d}/cov"] = _k(x.get("coverage"))
            k[f"e18/f/{fk}/{d}/accacc"] = _k(x.get("accepted_accuracy"))
            k[f"e18/f/{fk}/{d}/defacc"] = _k(x.get("deferred_accuracy"))
            k[f"e18/f/{fk}/{d}/acc"] = _k(x.get("accuracy_all"))
            if x.get("n") is not None and x.get("coverage") is not None:
                k[f"e18/f/{fk}/{d}/deferred"] = str(int(round(x["n"] * (1 - x["coverage"]))))
    h = r.get("e18_head") or {}
    if h.get("weights"):
        for i, w in enumerate(h["weights"]):
            k[f"e18/head/w{i}"] = f"{w:.2f}"
        k["e18/head/n"] = _n(_g(h, "fit", "n"))
        k["e18/head/positives"] = str(_g(h, "fit", "positives"))
        k["e18/head/logloss"] = _k(_g(h, "fit", "log_loss_mean"), 3, False)
        for dp, x in (h.get("by_dp") or {}).items():
            d = {"tool_select": "ts", "tool_params": "tp"}[dp]
            k[f"e18/head/{d}/n"] = _n(x["n"])
            k[f"e18/head/{d}/positives"] = str(x["positives"])
            k[f"e18/head/{d}/llmright"] = _n(x["llm_right"])
            k[f"e18/head/{d}/refright"] = _n(x["ref_right"])
            k[f"e18/head/{d}/k"] = f"{x['k']:.4f}"
            k[f"e18/head/{d}/kpct"] = _k(x["k"], 2)
            k[f"e18/head/{d}/covinsample"] = _k(x["coverage_in_sample"])
            k[f"e18/head/{d}/posdeferred"] = str(x["positives_deferred"])
            k[f"e18/head/{d}/tau"] = f"{1 - x['k']:.4f}"
    ec = r.get("e18_corpus") or {}
    for name, x in ec.items():
        key = {"head_at_tauhat": "a", "head_replace": "areplace", "rl_frozen": "rl", "gen_at_tauhat": "gen"}.get(name, name)
        k[f"e18/c/{key}/n"] = _n(x["n"])
        k[f"e18/c/{key}/cov"] = _k(x["coverage"])
        k[f"e18/c/{key}/accacc"] = _k(x.get("accepted_accuracy"))
        k[f"e18/c/{key}/accerr"] = str(x.get("accepted_errors"))
        k[f"e18/c/{key}/defacc"] = _k(x.get("deferred_accuracy"))
        lo = x.get("llm_only") or {}
        k[f"e18/c/{key}/llm/n"] = _n(lo.get("n"))
        k[f"e18/c/{key}/llm/deferred"] = str(lo.get("deferred"))
        k[f"e18/c/{key}/llm/deferredright"] = _k(lo.get("deferred_llm_right_share"))
        k[f"e18/c/{key}/llm/cascacc"] = _k(lo.get("cascade_task_acc"))
        k[f"e18/c/{key}/llm/accwrongllmright"] = str(lo.get("accepted_wrong_llm_right"))
    lv = r.get("live") or {}
    if lv.get("arms"):
        k["e18/final/n"] = str(lv.get("n_common"))
        k["e18/final/errored"] = str(_g(lv, "errored", "e18_final"))
        for arm, x in lv["arms"].items():
            key = {"e18_final": "", "llm_only": "llm/", "harness_final": "hf4b/", "e14_final": "e14/"}[arm]
            k[f"e18/final/{key}success"] = _k(x["success"])
            k[f"e18/final/{key}api"] = _k(1000 * x["api_usd"], 2, False)
            k[f"e18/final/{key}q"] = _k(1000 * x["q_usd"], 2, False)
            k[f"e18/final/{key}tot"] = _k(1000 * x["usd_incl_gpu"], 2, False)
            k[f"e18/final/{key}tote9"] = _k(1000 * x["usd_incl_gpu_e9"], 2, False)
            k[f"e18/final/{key}wall"] = _k(x["wall_s_mean"], 1, False)
            k[f"e18/final/{key}s1s"] = _k(x["s1_s"], 2, False)
            k[f"e18/final/{key}s2s"] = _k(x["s2_s"], 2, False)
            k[f"e18/final/{key}s2calls"] = _k(x["s2_calls"], 2, False)
            k[f"e18/final/{key}toolcov"] = _k(x.get("tool_coverage"))
            k[f"e18/final/{key}argcov"] = _k(x.get("arg_coverage"))
        for ref, c in (lv.get("contrasts") or {}).items():
            key = {"llm_only": "llm", "harness_final": "hf4b", "e14_final": "e14"}[ref]
            k[f"e18/final/saving_vs_{key}"] = _k(c["saving_incl_gpu"]["delta"])
            k[f"e18/final/saving_vs_{key}/lo"] = _k(c["saving_incl_gpu"]["lo"])
            k[f"e18/final/saving_vs_{key}/hi"] = _k(c["saving_incl_gpu"]["hi"])
            k[f"e18/final/saving_vs_{key}/e9"] = _k(c["saving_incl_gpu_e9"]["delta"])
            k[f"e18/final/vs{key}/success"] = _k(c["success"]["delta"])
            k[f"e18/final/vs{key}/success/lo"] = _k(c["success"]["lo"])
            k[f"e18/final/vs{key}/success/hi"] = _k(c["success"]["hi"])
            k[f"e18/final/vs{key}/wall"] = _k(c["wall_s"]["delta"], 2, False)
            w_e, w_r = lv["arms"]["e18_final"]["wall_s_mean"], lv["arms"][ref]["wall_s_mean"]
            k[f"e18/final/time_saving_vs_{key}"] = _k(1 - w_e / w_r) if w_r else None
    rd = r.get("readings") or {}
    for dp in DPS:
        x = rd.get(f"e17/{dp}")
        if x:
            d = {"tool_select": "ts", "tool_params": "tp"}[dp]
            k[f"e17/reading/{d}"] = {"head": "the scoring head", "gen": "the one-token read-out", "tie": "neither (a tie)",
                                     None: "neither (both accepted an error)"}.get(x["winner"], str(x["winner"]))
    if "e18/live" in rd:
        k["e18/reading/live"] = "holds" if rd["e18/live"]["holds"] else "does not hold"
        k["e18/reading/live/delta"] = _k(rd["e18/live"]["success_minus_llm_pt"] / 100)
    if "e18/c" in rd:
        k["e18/reading/c"] = "holds" if rd["e18/c"]["holds"] else "does not hold"
    if r.get("gpu_hours") is not None:
        k["e17/gpuhours"] = _k(r["gpu_hours"], 1, False)
    k["e17/spend"] = _k(_g(r, "spend", "total"), 2, False)
    return {kk: vv for kk, vv in k.items() if vv is not None}


def cmd_summarize(a: argparse.Namespace) -> int:
    res = E14._jsonable(summarize(Path(a.box)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "summary.md").write_text(markdown(res) + "\n")
    keys = tex_keys(res)
    E14.write_tex(keys, TEX)
    lines = TEX.read_text().split("\n")   # E14's writer names E14 in its header: this file's own three lines
    lines[:3] = ["% AUTO-GENERATED by scripts/explore_e17_e18.py summarize -- do not edit by hand.",
                 "% Exploratory rows E17 and E18 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E17 and E18: read-out",
                 "% ablation and a deferral-aware objective'); descriptive, outside every Holm table. Keys e17/... and e18/..."]
    TEX.write_text("\n".join(lines))
    print(f"{len(keys)} keys -> {TEX.relative_to(ROOT)}; summary -> {OUT.relative_to(ROOT)}")
    return 0


def cmd_collect(a: argparse.Namespace) -> int:
    """The box records the readout needs, slimmed into results/explore/e17_e18/records/ (raw files stay on S3)."""
    import shutil

    import e5_analysis as E5A
    import explore_e13_llms as E13

    box = Path(a.box)
    REC.mkdir(parents=True, exist_ok=True)
    n = 0
    for pat in ("train_record_*.json", "eval/*.json", "host*.json", "stage_times.jsonl", "env_*.txt", "e14venv_freeze.txt",
                "serve_windows.jsonl", "live_window.json", "frozen_verify.json", "arm_a_sha256.txt", "gen_*.json",
                "gen_train_mix.sha256", "logs/gate_*.json", "train_runs.jsonl", "status.txt", "api_probe_log.txt",
                "e18f_not_run.json", "logs/serve_*.cmd", "logs/serve_*_models.json"):
        for f in box.glob(pat):
            dst = REC / f.relative_to(box)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    for sub in ("s23", "cal"):
        for f in sorted((box / sub).glob("answers_*.jsonl")):
            dst = REC / sub / (f.name + ".gz")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(gzip.compress(f.read_bytes(), 9))
            n += 1
    lat = box / "latency" / "bench"
    if lat.exists():
        comp = {f.name: E5A.bench_stats(f) for f in sorted(lat.glob("bench_*.json"))}
        (REC / "latency").mkdir(parents=True, exist_ok=True)
        (REC / "latency" / "bench_stats.json").write_text(json.dumps(comp, indent=1) + "\n")
        for f in (box / "latency").glob("models_*.json"):
            shutil.copy2(f, REC / "latency" / f.name)
        n += 1
    for log in ("train_gen.log", "train_F.log"):
        p = box / "logs" / log
        if p.exists():
            (REC / "logs").mkdir(parents=True, exist_ok=True)
            text = p.read_text()
            lines = [x for x in text.splitlines() if "train_runtime" in x or "train_loss" in x or "loss" in x[:60]][-30:]
            (REC / "logs" / (log + ".tail")).write_text("\n".join(lines) + "\n")
            n += 1
    for run in LIVE_RUNS:
        if (RUNS / run).is_dir():
            E13.slim_run(RUNS / run, REC / "live" / run)
            n += 1
    print(f"{n} record files -> {REC.relative_to(ROOT)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("--log", required=True); p.set_defaults(fn=E14.cmd_probe)
    w = sub.add_parser("watch"); w.add_argument("--pid", type=int, required=True); w.add_argument("--cap", type=float, required=True)
    w.add_argument("--runs", default="runs"); w.add_argument("--stop", required=True); w.add_argument("--every", type=int, default=30)
    w.add_argument("--no-pause", action="store_true"); w.set_defaults(fn=E14.cmd_watch)
    e = sub.add_parser("errored"); e.add_argument("--run", required=True); e.set_defaults(fn=E14.cmd_errored)
    s = sub.add_parser("spend"); s.add_argument("--runs", default="runs"); s.add_argument("--short", action="store_true"); s.set_defaults(fn=E14.cmd_spend)
    g = sub.add_parser("gate"); g.add_argument("--runs", default="runs"); g.add_argument("--smoke", required=True)
    g.add_argument("--n", type=int, required=True); g.add_argument("--cap", type=float, required=True); g.set_defaults(fn=E14.cmd_gate)
    sf = sub.add_parser("smoke-file"); sf.add_argument("--mix", required=True); sf.add_argument("--out", required=True); sf.set_defaults(fn=cmd_smoke_file)
    pj = sub.add_parser("projection"); pj.add_argument("--log", required=True); pj.add_argument("--rows", type=int, default=16902)
    pj.add_argument("--cap-min", type=float, default=280); pj.add_argument("--max-epochs", type=int, default=1)
    pj.add_argument("--setting", default="nogc_6144"); pj.add_argument("--out", required=True)
    pj.set_defaults(fn=cmd_projection)
    cp = sub.add_parser("check-profile"); cp.set_defaults(fn=cmd_check_profile)
    fh = sub.add_parser("fit-head"); fh.add_argument("--ref-answers", default=str(REF_ANSWERS)); fh.set_defaults(fn=cmd_fit_head)
    pl = sub.add_parser("prepare-live"); pl.set_defaults(fn=cmd_prepare_live)
    ds = sub.add_parser("data-sha"); ds.add_argument("--data", required=True); ds.set_defaults(fn=cmd_data_sha)
    fg = sub.add_parser("fit-gen"); fg.add_argument("--answers", default=str(BOX / "cal" / "answers_e17gen.jsonl")); fg.set_defaults(fn=cmd_fit_gen)
    co = sub.add_parser("collect"); co.add_argument("--box", default=str(BOX)); co.set_defaults(fn=cmd_collect)
    sm = sub.add_parser("summarize"); sm.add_argument("--box", default=str(BOX)); sm.set_defaults(fn=cmd_summarize)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
