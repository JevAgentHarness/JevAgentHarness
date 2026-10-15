#!/usr/bin/env python3
r"""Recipe takeaways of the decision-model training work -> \res{recipe/...} keys and a takeaway table.

Reads only committed records (no checkpoint, no GPU, no API):
  data/train_v0/{MANIFEST.json, train.jsonl}             data v0 (logged states of LLM-only episodes, gold labels)
  results/training/train_v1/manifest.json                 data v1 build manifest (states, labels, cost)
  data/train_v1/dev_teacher.summary.json                  Jev on v1 dev (on-plan / after-error accuracy)
  results/gpu_round5/{ablation.json, chain_decision.json, FREEZE_RECORD.json, spend.json, status.txt,
      eval/eval_<tag>_v1.json, final_calibration.json, base_calibration.json, cal/tauhat_rl_shadow_cal_*.json,
      latency_*.txt}                                      JevAgentRL development and freeze records (round 5)
  results/stage2/gpu/rl_attr/sft_v1_calibration.json      SFT v1's temperatures (fitted after the freeze, A7.13 (i))
  results/calibration/tauhat_jev_shadow_cal_*.json        Jev's registered tau-hat* fit (Flash)
  results/gpu_round6_miles/{ablation.json, base.json, final_calibration.json}   JevAgentRL-v2 (A7.19, exploratory)
  results/stage2/stage2.json                              the descriptive attribution rows on the common states C
and the generated key files of the paper (paper/generated/{rl,stage2,spend,t2ref,explore,data_stats}.tex) for the
values that already have keys. It writes only the keys no other script writes (namespace ``recipe/``):

  recipe/data/{v0,v1}/...        composition of the two training sets (rows, tasks, after-error rows, ...)
  recipe/dev/<tag>/...           v1-dev accuracy on on-plan states, the recovery gap, recovery tool selection
  recipe/states/...              SFT v1 minus SFT v0 (development)
  recipe/labels/...              student minus teacher (v1 dev: development; C: descriptive)
  recipe/stages/...              stage minus its init (development; v2 exploratory)
  recipe/stage/<stage>/...       GPU minutes / hours and API $ of each round-5 stage
  recipe/cost/...                derived costs (SFT-only recipe, development-slice share)
  recipe/calib/...               mean confidence, fitted temperatures, calibrated v1-dev ECE (in-sample)
  recipe/base/...                base-model gaps before and after SFT (v1 vs v2)
  recipe/deploy/...              tau-hat* per backend, latency ratios

and results/training/recipe_takeaways.md: every takeaway with its evidence (numbers as printed, key, source) and
its label (registered / descriptive / development / exploratory). Accuracies in percent with one decimal; signed
differences in points ("+7.6", printed with \resm); ECE with three decimals; USD with two.
Usage: python scripts/recipe_takeaways.py [--out-tex FILE] [--out-md FILE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results" / "gpu_round5"
R6 = ROOT / "results" / "gpu_round6_miles"
TRAIN_V1 = ROOT / "results" / "training" / "train_v1"
DATA_V0 = ROOT / "data" / "train_v0"
DATA_V1 = ROOT / "data" / "train_v1"
S2 = ROOT / "results" / "stage2"
CAL = ROOT / "results" / "calibration"
GENERATED = ROOT / "paper" / "generated"
GEN_FILES = ("rl", "stage2", "spend", "t2ref", "explore", "data_stats")
OUT_TEX = GENERATED / "recipe.tex"
OUT_MD = ROOT / "results" / "training" / "recipe_takeaways.md"
V1_DEV = "data/train_v1/dev.jsonl.gz"
V1_REC = "data/train_v1/dev_recovery.jsonl.gz"
DEFRES = re.compile(r"^\\defres\{([^}]*)\}\{(.*)\}\s*$")


# ------------------------------------------------------------------ formatting
def pct(x: float | None, nd: int = 1) -> str | None:
    return None if x is None else f"{100 * x:.{nd}f}"


def signed(x: float | None, nd: int = 1) -> str | None:
    """A difference in points (x already in points) with an explicit sign; -0.0 prints as +0.0."""
    if x is None:
        return None
    s = f"{x:+.{nd}f}"
    return "+" + s[1:] if s.startswith("-") and float(s) == 0 else s


def num(n: float | int) -> str:
    """Thousands separator as the other generated files print it (1{,}225)."""
    return f"{int(round(n)):,}".replace(",", "{,}")


def usd(x: float) -> str:
    return f"{x:,.2f}".replace(",", "{,}")


def plain(v: str) -> str:
    """A key value as it reads in markdown ({,} -> ,; $-$ -> -; $<$ -> <)."""
    return v.replace("{,}", ",").replace("$-$", "-").replace("$<$", "<").replace("\\_", "_")


# ------------------------------------------------------------------ readers
def _json(p: Path) -> Any:
    return json.loads(p.read_text())


def parse_defres(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = DEFRES.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def generated_keys(names: Iterable[str] = GEN_FILES, root: Path = GENERATED) -> dict[str, str]:
    out: dict[str, str] = {}
    for n in names:
        p = root / f"{n}.tex"
        if p.exists():
            out.update(parse_defres(p.read_text()))
    return out


def stage_minutes(status: str) -> dict[str, int]:
    """status.txt lines ``... stage <name> exit=0 (<m> min)`` -> name -> minutes of the last successful run."""
    out: dict[str, int] = {}
    for m in re.finditer(r"stage (\S+) exit=0 \((\d+) min\)", status):
        out[m.group(1)] = int(m.group(2))
    return out


def onplan_acc(acc_all: float, n_all: int, acc_rec: float, n_rec: int) -> float:
    """Accuracy on the dev rows that are not after an off-plan step (dev_recovery is the after-error subset of dev)."""
    hits = round(acc_all * n_all) - round(acc_rec * n_rec)
    return hits / (n_all - n_rec)


def eval_file(tag: str) -> dict[str, Any] | None:
    p = R5 / "eval" / f"eval_{tag}_v1.json"
    return _json(p)["files"] if p.exists() else None


# ------------------------------------------------------------------ keys
def data_keys(v: dict[str, str]) -> None:
    m0 = _json(DATA_V0 / "MANIFEST.json")["files"]["train.jsonl"]
    v["recipe/data/v0/rows"] = num(m0["rows"])
    v["recipe/data/v0/tasks"] = num(m0["tasks"])
    v["recipe/data/v0/gold_rows"] = num(m0["label_sources"].get("gold", 0))
    p0 = DATA_V0 / "train.jsonl"
    if p0.exists():
        rows = [json.loads(x) for x in p0.read_text().splitlines() if x.strip()]
        v["recipe/data/v0/after_error_rows"] = num(sum(bool((r.get("meta") or {}).get("after_error")) for r in rows))
        v["recipe/data/v0/state_chars_median"] = num(statistics.median(len(r["state"]) for r in rows))
    man = _json(TRAIN_V1 / "manifest.json")
    tr = man["files"]["train.jsonl.gz"]
    v["recipe/data/v1/rows"] = num(tr["rows"])
    v["recipe/data/v1/tasks"] = num(tr["tasks"])
    v["recipe/data/v1/after_error_rows"] = num(tr["after_error_rows"])
    v["recipe/data/v1/after_error_share"] = pct(tr["after_error_rows"] / tr["rows"])
    v["recipe/data/v1/state_chars_median"] = num(tr["state_chars_median"])
    v["recipe/data/v1/rows_ratio"] = f"{tr['rows'] / m0['rows']:.1f}"
    arm = tr["by_arm"]
    v["recipe/data/v1/nearmiss_rows"] = num(arm.get("nearmiss_replace", 0))
    v["recipe/data/v1/nearmiss_share"] = pct(arm.get("nearmiss_replace", 0) / tr["rows"])
    b = tr["by_bucket"]
    v["recipe/data/v1/guard_no_rows"] = num(b.get("tool_guard:no", 0) + b.get("stop_guard:no", 0))
    ls = tr["label_sources"]
    v["recipe/data/v1/gold_fallback_rows"] = num(ls.get("gold", 0))
    v["recipe/data/v1/gold_fallback_share"] = pct(ls.get("gold", 0) / tr["rows"])
    v["recipe/data/v1/teacher_rows"] = num(ls.get("teacher", 0))
    v["recipe/data/v1/teacher_train_acc"] = pct(man["teacher_train"]["teachers"]["jev"]["acc"])
    v["recipe/data/v1/knockout_rows"] = num(man["build"]["knockout_final_variants"]["rows"])
    eps = [a for r in man["build"]["episodes"].values() for a in r["arms"].values()]
    v["recipe/data/v1/episodes"] = num(sum(a["episodes"] for a in eps))
    v["recipe/data/v1/offplan_episodes"] = num(sum(a["off_plan_episodes"] for a in eps))
    nm = [a for r in man["build"]["episodes"].values() for k, a in r["arms"].items() if k == "nearmiss_replace"]
    if nm:
        v["recipe/data/v1/nearmiss_offplan_share"] = pct(sum(a["off_plan_episodes"] for a in nm)
                                                         / sum(a["episodes"] for a in nm))
    cost = man["cost"]
    ep = sum(x for k, x in cost.items() if k.startswith("episodes"))
    te = sum(x for k, x in cost.items() if k.startswith("teacher"))
    v["recipe/data/v1/episodes_usd"] = usd(ep)
    v["recipe/data/v1/teacher_usd"] = usd(te)
    v["recipe/data/v1/build_usd"] = usd(cost["total"])
    dv = man["files"]["dev.jsonl.gz"]
    v["recipe/data/v1/dev_rows"] = num(dv["rows"])
    v["recipe/data/v1/dev_recovery_rows"] = num(dv["after_error_rows"])
    p1 = DATA_V1 / "train.jsonl.gz"
    if p1.exists():   # the mixing rule's effect on the targets: whose option is the argmax, how much mass the gold keeps
        with gzip.open(p1, "rt") as f:
            rows1 = [json.loads(x) for x in f if x.strip()]
        hit, mass = 0, 0.0
        for r in rows1:
            t, m = r["target"], r.get("meta") or {}
            gold = m.get("gold") if m.get("label_source") == "teacher" else max(t, key=t.get)
            hit += max(t, key=t.get) == gold
            mass += float(t.get(gold, 0.0))
        v["recipe/data/v1/target_argmax_gold_share"] = pct(hit / len(rows1))
        v["recipe/data/v1/target_gold_mass_mean"] = pct(mass / len(rows1))


DEV_TAGS = ("base", "sft_v0", "sft_v1", "opd_i12", "opd", "grpo_i15", "grpo")


def dev_keys(v: dict[str, str], raw: dict[str, Any]) -> None:
    """On-plan accuracy, recovery gap (on-plan minus after-error accuracy) and recovery tool selection on v1 dev."""
    for tag in DEV_TAGS:
        f = eval_file(tag)
        if f is None:
            continue
        a, r = f[V1_DEV]["overall"], f[V1_REC]["overall"]
        on = onplan_acc(a["acc"], a["n"], r["acc"], r["n"])
        raw[tag] = {"acc": a["acc"], "rec": r["acc"], "on": on, "ece": a["ece"], "conf": a["mean_conf"],
                    "ts": f[V1_DEV]["by_dp"]["tool_select"]["acc"],
                    "rec_ts": f[V1_REC]["by_dp"]["tool_select"]["acc"]}
        v[f"recipe/dev/{tag}/onplan_acc"] = pct(on)
        v[f"recipe/dev/{tag}/recovery_gap"] = signed(100 * (on - r["acc"]))
        v[f"recipe/dev/{tag}/recovery_tool_select"] = pct(raw[tag]["rec_ts"])
    jt = _json(DATA_V1 / "dev_teacher.summary.json")["metrics"] if (DATA_V1 / "dev_teacher.summary.json").exists() \
        else _json(TRAIN_V1 / "dev_teacher.summary.json")["metrics"]
    on, rec = jt["after_error:0"]["acc"], jt["after_error:1"]["acc"]
    raw["jev"] = {"acc": jt["overall"]["acc"], "rec": rec, "on": on, "ece": jt["overall"]["ece"],
                  "ts": jt["dp:tool_select"]["acc"]}
    v["recipe/dev/jev/onplan_acc"] = pct(on)
    v["recipe/dev/jev/recovery_gap"] = signed(100 * (on - rec))


def states_keys(v: dict[str, str], raw: dict[str, Any]) -> None:
    """SFT v1 (data v1) minus SFT v0 (data v0) on v1 dev and the development slice (development numbers)."""
    a, b = raw.get("sft_v1"), raw.get("sft_v0")
    if not (a and b):
        return
    v["recipe/states/delta_v1_acc"] = signed(100 * (a["acc"] - b["acc"]))
    v["recipe/states/delta_recovery"] = signed(100 * (a["rec"] - b["rec"]))
    v["recipe/states/delta_tool_select"] = signed(100 * (a["ts"] - b["ts"]))
    v["recipe/states/delta_recovery_tool_select"] = signed(100 * (a["rec_ts"] - b["rec_ts"]))
    rows = {r["tag"]: r for r in _json(R5 / "ablation.json")["rows"]}
    s0, s1 = rows["sft_v0"].get("ts_replace_success"), rows["sft_v1"].get("ts_replace_success")
    if s0 is not None and s1 is not None:
        v["recipe/states/delta_ts_replace_success"] = signed(100 * (s1 - s0))
        v["recipe/states/delta_ts_replace_own_acc"] = signed(100 * (rows["sft_v1"]["ts_replace_ts"]
                                                                    - rows["sft_v0"]["ts_replace_ts"]))


def stage2_rows() -> dict[str, Any]:
    d = _json(S2 / "stage2.json")["descriptive"]
    rows = dict(d.get("stage2_rl") or {})
    s1 = (d.get("stage2_s1") or {}).get("rows") or {}
    for k in ("sl4b", "sl27b"):
        if k in s1:
            rows[k] = s1[k]
    v2 = (d.get("stage2_v2") or {}).get("row")
    if v2:
        rows["v2"] = v2
    return rows


def labels_keys(v: dict[str, str], raw: dict[str, Any], c: Mapping[str, Any]) -> None:
    s, j = raw.get("sft_v1"), raw.get("jev")
    if s and j:
        v["recipe/labels/v1/student_minus_teacher"] = signed(100 * (s["acc"] - j["acc"]))
        v["recipe/labels/recovery/student_minus_teacher"] = signed(100 * (s["rec"] - j["rec"]))
        v["recipe/labels/tool_select/student_minus_teacher"] = signed(100 * (s["ts"] - j["ts"]))
    jc = c.get("jev") or {}
    for row in ("sft_v1", "frozen", "v2"):
        r = c.get(row) or {}
        if r.get("corpus_acc") is not None and jc.get("corpus_acc") is not None:
            v[f"recipe/labels/c/{row}_minus_jev"] = signed(100 * (r["corpus_acc"] - jc["corpus_acc"]))
            v[f"recipe/labels/c/{row}_minus_jev_offplan"] = signed(100 * (r["corpus_acc_offplan"]
                                                                        - jc["corpus_acc_offplan"]))


def stages_keys(v: dict[str, str], raw: dict[str, Any]) -> None:
    """Each stage minus its init on development data; the RL stages' cost (round 5) and the v2 stress slice."""
    base = raw.get("sft_v1")
    for tag in ("opd_i12", "opd", "grpo_i15", "grpo"):
        r = raw.get(tag)
        if base and r:
            v[f"recipe/stages/{tag}/delta_v1_acc"] = signed(100 * (r["acc"] - base["acc"]))
            v[f"recipe/stages/{tag}/delta_recovery"] = signed(100 * (r["rec"] - base["rec"]))
    ch = _json(R5 / "chain_decision.json").get("dev_slice") or {}
    if "sft_v1" in ch:
        for tag in ("opd_i12", "opd", "grpo_i15", "grpo"):
            if tag in ch:
                v[f"recipe/stages/{tag}/delta_reward_live"] = f"{ch[tag]['reward_live'] - ch['sft_v1']['reward_live']:+.4f}"
    a2 = _json(R6 / "ablation.json")["v2"] if (R6 / "ablation.json").exists() else {}
    s2 = (a2.get("sft_v2") or {}).get("stress") or {}
    for tag in ("opd_i12", "opd", "grpo_i15", "grpo"):
        st = (a2.get(tag) or {}).get("stress") or {}
        if s2.get("S") is not None and st.get("S") is not None:
            v[f"recipe/stages/v2/{tag}/delta_stress_success"] = signed(100 * (st["S"] - s2["S"]))
            v[f"recipe/stages/v2/{tag}/delta_stress_reward"] = f"{st['R'] - s2['R']:+.3f}"
        d, d0 = (a2.get(tag) or {}).get("decision") or {}, (a2.get("sft_v2") or {}).get("decision") or {}
        if d.get("acc") is not None and d0.get("acc") is not None:
            v[f"recipe/stages/v2/{tag}/delta_v1_acc"] = signed(100 * (d["acc"] - d0["acc"]))


def cost_keys(v: dict[str, str]) -> None:
    sp = _json(R5 / "spend.json")
    rate = float(sp["gpu_usd_per_h"])
    mins = stage_minutes((R5 / "status.txt").read_text())
    api = sp["api_usd"]
    stage_api = {"sft_v1": None,
                 "opd": sum(x for k, x in api.items() if k.startswith("opd:")),
                 "grpo": sum(x for k, x in api.items() if k.startswith("grpo:"))}
    for st, key in (("sft_v1", "sft"), ("opd", "opd"), ("grpo", "grpo")):
        if key in mins:
            v[f"recipe/stage/{st}/gpu_min"] = num(mins[key])
            v[f"recipe/stage/{st}/gpu_h"] = f"{mins[key] / 60:.1f}"
            v[f"recipe/stage/{st}/gpu_usd"] = usd(mins[key] / 60 * rate)
        if stage_api[st] is not None:
            v[f"recipe/stage/{st}/api_usd"] = usd(stage_api[st])
    if "opd:teacher_jev" in api:
        v["recipe/stage/opd/teacher_usd"] = usd(api["opd:teacher_jev"])
    if "opd" in mins and "grpo" in mins:
        m = mins["opd"] + mins["grpo"]
        v["recipe/stage/rl_extra/gpu_h"] = f"{m / 60:.1f}"
        v["recipe/stage/rl_extra/gpu_usd"] = usd(m / 60 * rate)
        v["recipe/stage/rl_extra/api_usd"] = usd(stage_api["opd"] + stage_api["grpo"])
    v["recipe/cost/gpu_usd_per_h"] = f"{rate:.3f}"
    build = _json(TRAIN_V1 / "manifest.json")["cost"]["total"]
    if "sft" in mins:
        sft_usd = mins["sft"] / 60 * rate
        v["recipe/cost/sft_gpu_usd"] = usd(sft_usd)
        v["recipe/cost/sft_only_usd"] = usd(build + sft_usd)
    dev = sum(x for k, x in api.items() if k.startswith("e2dev:"))
    v["recipe/cost/dev_slices_usd"] = usd(dev)
    v["recipe/cost/dev_slices_share"] = pct(dev / sp["api_usd_total"])
    v["recipe/cost/cal_arms_usd"] = usd(sum(x for k, x in api.items() if k.startswith("cal:")))


def calib_keys(v: dict[str, str], raw: dict[str, Any]) -> None:
    for tag in ("base", "sft_v1", "grpo_i15"):
        r = raw.get(tag)
        if r:
            v[f"recipe/calib/{tag}/mean_conf"] = pct(r["conf"])
            v[f"recipe/calib/{tag}/acc_minus_conf"] = signed(100 * (r["acc"] - r["conf"]))
    temps = {"base": R5 / "base_calibration.json", "sft_v1": S2 / "gpu" / "rl_attr" / "sft_v1_calibration.json",
             "frozen": R5 / "final_calibration.json", "v2": R6 / "final_calibration.json"}
    for k, p in temps.items():
        if p.exists():
            v[f"recipe/calib/{k}/temp_global"] = f"{_json(p)['global']:.2f}"
    p = R5 / "eval" / "eval_final_calibrated_v1.json"
    if p.exists():
        f = _json(p)["files"]
        v["recipe/calib/frozen/v1_ece_temp"] = f"{f[V1_DEV]['overall']['ece']:.3f}"
        v["recipe/calib/frozen/v1_acc_temp"] = pct(f[V1_DEV]["overall"]["acc"])


def base_keys(v: dict[str, str]) -> None:
    b = _json(R6 / "base.json")["checks"]
    q, s = b["qwen"]["zeroshot_v1_dev_acc"], b["startlux"]["zeroshot_v1_dev_acc"]
    v["recipe/base/zeroshot_gap"] = signed(100 * (s - q))
    a2 = _json(R6 / "ablation.json")["v2"]["sft_v2"]["decision"]["acc"]
    a1 = next(r for r in _json(R5 / "ablation.json")["rows"] if r["tag"] == "sft_v1")["v1_acc"]
    v["recipe/base/sft_gap"] = signed(100 * (a2 - a1))


def deploy_keys(v: dict[str, str]) -> None:
    fits = {("jev", "ts"): CAL / "tauhat_jev_shadow_cal_tool_select.json",
            ("jev", "tp"): CAL / "tauhat_jev_shadow_cal_tool_params.json",
            ("rl", "ts"): R5 / "cal" / "tauhat_rl_shadow_cal_tool_select.json",
            ("rl", "tp"): R5 / "cal" / "tauhat_rl_shadow_cal_tool_params.json"}
    for (who, dp), p in fits.items():
        if not p.exists():
            continue
        d = _json(p)
        t = d["tau_hat"]
        v[f"recipe/deploy/tauhat/{who}_flash_{dp}"] = "never" if t in (None, float("inf")) else (
            f"{t:.2f}" if t <= 0.999 or t >= 1.0 else f"{t:.7f}")
        v[f"recipe/deploy/tauhat/{who}_flash_{dp}_n"] = num(d["n_paired"])
        fs = d.get("fatal_share") or {}
        if fs.get("n_wrong") is not None:
            v[f"recipe/deploy/tauhat/{who}_flash_{dp}_wrong"] = num(fs["n_wrong"])
    fr = _json(R5 / "FREEZE_RECORD.json")["tau_hat_star"]["flash"]
    for dp, k in (("tool_select", "ts"), ("tool_params", "tp")):
        v[f"recipe/deploy/tauhat/rl_flash_{k}_accept"] = pct(fr[dp]["scalar_accept_rate"])
    lat = {}
    for name, f in (("rl", "latency_jevagentrl4b.txt"), ("jev", "latency_jev.txt")):
        t = (R5 / f).read_text()
        lat[name] = json.loads(t[t.index("{"):])["results"]
    for c in ("c=1", "c=4"):
        v[f"recipe/deploy/latency/ratio_{c.replace('=', '')}"] = f"{lat['rl'][c]['p50_ms'] / lat['jev'][c]['p50_ms']:.1f}"
    for name in ("rl", "jev"):
        v[f"recipe/deploy/latency/{name}/ms_per_question_c1"] = f"{lat[name]['c=1']['p50_ms'] / 3:.0f}"


def build(gen: Mapping[str, str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    v: dict[str, str] = {}
    raw: dict[str, Any] = {}
    data_keys(v)
    dev_keys(v, raw)
    states_keys(v, raw)
    c = stage2_rows()
    labels_keys(v, raw, c)
    stages_keys(v, raw)
    cost_keys(v)
    calib_keys(v, raw)
    base_keys(v)
    deploy_keys(v)
    v = {k: x for k, x in v.items() if x is not None}
    clash = sorted(set(v) & set(gen or {}))
    if clash:
        raise SystemExit(f"recipe keys already written by another script: {clash[:5]}")
    return v, {"dev": raw, "c": c}


# ------------------------------------------------------------------ takeaways
# (description, key) pairs; every key must resolve in recipe.tex or another generated file (test-enforced)
TAKEAWAYS: list[dict[str, Any]] = [
    {"id": "T1", "topic": "States",
     "claim": "Build the training set from the states the harness itself visits, including the states after its own "
              "off-plan steps.",
     "label": "development (v1 dev, 50-task development slice); the Stage-2 attribution rows are descriptive",
     "support": "Supported for the v0 -> v1 switch as a whole. The switch changed four things at once (states, size, "
                "task setting, labels), so no record isolates the recovery states.",
     "evidence": [
         ("data v0: rows (logged states of successful LLM-only episodes)", "recipe/data/v0/rows"),
         ("data v0: tasks", "recipe/data/v0/tasks"),
         ("data v0: rows after an off-plan step", "recipe/data/v0/after_error_rows"),
         ("data v0: median state length (chars)", "recipe/data/v0/state_chars_median"),
         ("data v1: rows", "recipe/data/v1/rows"),
         ("data v1: tasks", "recipe/data/v1/tasks"),
         ("data v1: rows after an off-plan step", "recipe/data/v1/after_error_rows"),
         ("data v1: share after an off-plan step (%)", "recipe/data/v1/after_error_share"),
         ("data v1: median state length (chars)", "recipe/data/v1/state_chars_median"),
         ("data v1: rows from the scripted near-miss arm", "recipe/data/v1/nearmiss_rows"),
         ("data v1: near-miss episodes with an off-plan step (%)", "recipe/data/v1/nearmiss_offplan_share"),
         ("data v1: guard 'no' rows (wrong proposals, premature finish)", "recipe/data/v1/guard_no_rows"),
         ("data v1 / data v0 rows (x)", "recipe/data/v1/rows_ratio"),
         ("SFT v0: v1-dev accuracy (%)", "rl/dev/sft_v0/v1_acc"),
         ("SFT v1: v1-dev accuracy (%)", "rl/dev/sft_v1/v1_acc"),
         ("SFT v0: recovery accuracy (%)", "rl/dev/sft_v0/v1_recovery"),
         ("SFT v1: recovery accuracy (%)", "rl/dev/sft_v1/v1_recovery"),
         ("SFT v1 - SFT v0: recovery (points)", "recipe/states/delta_recovery"),
         ("SFT v1 - SFT v0: tool selection (points)", "recipe/states/delta_tool_select"),
         ("SFT v1 - SFT v0: tool selection after an off-plan step (points)", "recipe/states/delta_recovery_tool_select"),
         ("recovery gap (on-plan minus recovery accuracy), untrained base", "recipe/dev/base/recovery_gap"),
         ("recovery gap, SFT v0", "recipe/dev/sft_v0/recovery_gap"),
         ("recovery gap, SFT v1", "recipe/dev/sft_v1/recovery_gap"),
         ("recovery gap, Jev", "recipe/dev/jev/recovery_gap"),
         ("dev slice, tool replace success: SFT v0 (%)", "rl/dev/sft_v0/ts_replace_success"),
         ("dev slice, tool replace success: SFT v1 (%)", "rl/dev/sft_v1/ts_replace_success"),
         ("common states C after an off-plan step: untrained base (%)", "stage2_rl/base/corpus_acc_offplan"),
         ("common states C after an off-plan step: SFT v1 (%)", "stage2_rl/sft_v1/corpus_acc_offplan")],
     "sources": ["data/train_v0/MANIFEST.json", "data/train_v0/train.jsonl", "results/training/train_v1/manifest.json",
                 "results/gpu_round5/eval/eval_{base,sft_v0,sft_v1}_v1.json", "results/gpu_round5/ablation.json",
                 "results/stage2/stage2.json (descriptive.stage2_rl)"],
     "caveats": ["v0 -> v1 also grew the data 12.8x, moved the tool-selection rows to the evaluation setting, replaced "
                 "gold labels by 0.3 gold + 0.7 Jev and added guard 'no' rows and knockout-final variants.",
                 "v1 dev and the dev slice are training-side data of the same generator (development numbers).",
                 "The near-miss policy is scripted; it chooses states and never labels them."]},
    {"id": "T2", "topic": "Labels",
     "claim": "Label with the environment's oracle where it exists and add a teacher's soft distribution on top "
              "(0.3 gold + 0.7 Jev, gold one-hot where the teacher's argmax is wrong); the student then exceeded "
              "its teacher on the same states.",
     "label": "registered for 'no more than 5 points below Jev on C' (S2-3); the positive difference is descriptive; "
              "v1-dev differences are development numbers",
     "support": "Supported: the student beat the teacher on v1 dev and on C. By construction every training target's "
                "argmax is the gold option (the gold fallback covers the rows where Jev's argmax is wrong), so the "
                "teacher shaped the soft mass and never the target's top option; a student above its teacher is "
                "what this rule allows. Not supported: which label source carries the gain (no gold-only or "
                "teacher-only run exists).",
     "evidence": [
         ("Jev agreement with gold on the v1 training rows (%)", "recipe/data/v1/teacher_train_acc"),
         ("rows with the gold fallback (Jev argmax wrong)", "recipe/data/v1/gold_fallback_rows"),
         ("share with the gold fallback (%)", "recipe/data/v1/gold_fallback_share"),
         ("training targets whose argmax is the gold option (%)", "recipe/data/v1/target_argmax_gold_share"),
         ("mean target mass on the gold option (%)", "recipe/data/v1/target_gold_mass_mean"),
         ("v1 dev: SFT v1 - Jev (points)", "recipe/labels/v1/student_minus_teacher"),
         ("v1 dev recovery: SFT v1 - Jev (points)", "recipe/labels/recovery/student_minus_teacher"),
         ("v1 dev tool selection: SFT v1 - Jev (points)", "recipe/labels/tool_select/student_minus_teacher"),
         ("C: SFT v1 (%)", "stage2_rl/sft_v1/corpus_acc"),
         ("C: Jev (%)", "stage2_rl/jev/corpus_acc"),
         ("C: SFT v1 - Jev (points, difference of task means)", "recipe/labels/c/sft_v1_minus_jev"),
         ("C after an off-plan step: SFT v1 - Jev (points)", "recipe/labels/c/sft_v1_minus_jev_offplan"),
         ("S2-3 frozen vs Jev, paired estimate (points; NI clause registered)", "stage2/S2-3/jev/est"),
         ("data v1 labelling cost, Jev (USD)", "recipe/data/v1/teacher_usd")],
     "sources": ["results/training/train_v1/manifest.json (teacher_train, label_sources)",
                 "data/train_v1/dev_teacher.summary.json", "results/stage2/stage2.json", "results/stage2/stage2.md"],
     "caveats": ["The gold labels and C's labels come from the same JevGym oracle (in-domain).",
                 "SFT v0 is gold-only but on other states and less data, so it is no label ablation."]},
    {"id": "T3", "topic": "Stages",
     "claim": "Supervised fine-tuning on visited states carries the gain. On-policy distillation from the teacher "
              "lowered accuracy; GRPO kept accuracy and lowered raw ECE, and a temperature fit on SFT alone gave the "
              "same calibration on the common states.",
     "label": "development (chain records, both tracks); attribution on C descriptive; v2 exploratory",
     "support": "Supported on development data and C. Not measurable here: a GRPO effect on success (the development "
                "slice was saturated); in v2, GRPO lowered stress-slice success.",
     "evidence": [
         ("OPD it. 12 - SFT v1: v1-dev accuracy (points)", "recipe/stages/opd_i12/delta_v1_acc"),
         ("OPD it. 12 - SFT v1: recovery (points)", "recipe/stages/opd_i12/delta_recovery"),
         ("v2: OPD it. 12 - SFT v2: stress success (points)", "recipe/stages/v2/opd_i12/delta_stress_success"),
         ("GRPO it. 15 - SFT v1: v1-dev accuracy (points)", "recipe/stages/grpo_i15/delta_v1_acc"),
         ("raw v1-dev ECE: SFT v1", "rl/dev/sft_v1/v1_ece"),
         ("raw v1-dev ECE: GRPO it. 15 (frozen)", "rl/dev/grpo_i15/v1_ece"),
         ("GRPO it. 15 - SFT v1: live call-count-penalised success", "recipe/stages/grpo_i15/delta_reward_live"),
         ("SFT v1 mean confidence on v1 dev (%)", "recipe/calib/sft_v1/mean_conf"),
         ("SFT v1 accuracy minus mean confidence (points; under-confident)", "recipe/calib/sft_v1/acc_minus_conf"),
         ("SFT v1 fitted global temperature (< 1 sharpens)", "recipe/calib/sft_v1/temp_global"),
         ("frozen fitted global temperature", "recipe/calib/frozen/temp_global"),
         ("C ECE with temperatures: SFT v1", "stage2_rl/sft_v1/ece"),
         ("C ECE with temperatures: frozen (SFT v1 + GRPO)", "stage2_rl/frozen/ece"),
         ("C ECE with temperatures: v2 (SFT only)", "stage2_v2/ece"),
         ("C accuracy: SFT v1 (%)", "stage2_rl/sft_v1/corpus_acc"),
         ("C accuracy: frozen (%)", "stage2_rl/frozen/corpus_acc"),
         ("live success, first 100 fresh tasks: SFT v1 (%)", "stage2_rl/sft_v1/live_success"),
         ("live success, first 100 fresh tasks: frozen (%)", "stage2_rl/frozen/live_success"),
         ("v2: GRPO it. 15 - SFT v2: stress success (points)", "recipe/stages/v2/grpo_i15/delta_stress_success"),
         ("v2: GRPO it. 30 - SFT v2: stress success (points)", "recipe/stages/v2/grpo/delta_stress_success"),
         ("v2: GRPO it. 15 - SFT v2: stress reward", "recipe/stages/v2/grpo_i15/delta_stress_reward")],
     "sources": ["results/gpu_round5/{ablation.json, chain_decision.json, eval/, final_calibration.json}",
                 "results/stage2/gpu/rl_attr/sft_v1_calibration.json", "results/gpu_round6_miles/ablation.json",
                 "results/stage2/stage2.json"],
     "caveats": ["SFT v1's temperatures were fitted after the freeze on v1 dev (PREREG_A7_STAGE2 Deviations "
                 "2026-10-03); its calibrated v1-dev ECE is not recorded, only C's.",
                 "The frozen model's calibrated v1-dev ECE is in-sample (temperatures fitted on v1 dev).",
                 "No reverse-KL OPD and no linear GRPO reward were run."]},
    {"id": "T4", "topic": "Base model",
     "claim": "After the same visited-state SFT, the choice of 4B base made little difference on this generator, "
              "although the bases differed widely zero-shot.",
     "label": "development (zero-shot and SFT on v1 dev); C rows descriptive; v2 exploratory",
     "support": "Supported for two bases on one generator. Confounded with the training stack (LoRA with PEFT vs full "
                "fine-tuning with Miles).",
     "evidence": [
         ("zero-shot v1 dev, Qwen3-4B (%)", "rl2/zeroshot/qwen/v1_acc"),
         ("zero-shot v1 dev, StartLux-Decision-4B (%)", "rl2/zeroshot/startlux/v1_acc"),
         ("zero-shot gap (points)", "recipe/base/zeroshot_gap"),
         ("after SFT: v1 dev, SFT v1 on Qwen3-4B (%)", "rl/dev/sft_v1/v1_acc"),
         ("after SFT: v1 dev, SFT v2 on StartLux-4B (%)", "rl2/dev/sft_v2/v1_acc"),
         ("gap after SFT (points)", "recipe/base/sft_gap"),
         ("C: SFT v1 (%)", "stage2_rl/sft_v1/corpus_acc"),
         ("C: v2 (%)", "stage2_v2/corpus_acc")],
     "sources": ["results/gpu_round6_miles/{base.json, ablation.json}", "results/gpu_round5/ablation.json",
                 "results/stage2/stage2.json"],
     "caveats": ["StartLux-Decision-4B is CC BY-NC 4.0; a model trained from it inherits the licence.",
                 "Round 6's GPU bill is not logged (cap $400), so the v2 recipe has no cost row."]},
    {"id": "T5", "topic": "Calibration and deployment",
     "claim": "Fit temperatures and cascade thresholds again for every backend, and budget for serving latency "
              "and a GPU tariff.",
     "label": "registered procedures (tau-hat* A6.5 / A6.11; tariff A7.13 (iii)); latency descriptive",
     "support": "Supported: on the same calibration states the registered threshold fit deferred Jev's tool choices "
                "(tau-hat* 1.00 with Flash) and accepted every JevAgentRL tool choice (tau-hat* 0.50).",
     "evidence": [
         ("tau-hat*, Flash, tool selection: Jev", "recipe/deploy/tauhat/jev_flash_ts"),
         ("tau-hat*, Flash, tool selection: JevAgentRL", "recipe/deploy/tauhat/rl_flash_ts"),
         ("wrong shadow tool choices on the calibration states: Jev", "recipe/deploy/tauhat/jev_flash_ts_wrong"),
         ("wrong shadow tool choices on the calibration states: JevAgentRL", "recipe/deploy/tauhat/rl_flash_ts_wrong"),
         ("paired calibration decisions (tool selection)", "recipe/deploy/tauhat/rl_flash_ts_n"),
         ("JevAgentRL tool-choice acceptance rate at its tau-hat* (%)", "recipe/deploy/tauhat/rl_flash_ts_accept"),
         ("JevAgentRL argument acceptance rate at its tau-hat* (%)", "recipe/deploy/tauhat/rl_flash_tp_accept"),
         ("p50 latency per 3-question request, c = 1: JevAgentRL (ms)", "rl/latency/jevagentrl/p50_ms_c1"),
         ("p50 latency per 3-question request, c = 1: Jev API (ms)", "rl/latency/jev/p50_ms_c1"),
         ("latency ratio at c = 1 (x)", "recipe/deploy/latency/ratio_c1"),
         ("latency ratio at c = 4 (x)", "recipe/deploy/latency/ratio_c4"),
         ("GPU tariff q, S2-4 (USD per episode)", "stage2/S2-4/tariff/q"),
         ("host price of the tariff (USD per hour)", "spend/s24_host/usd_per_h"),
         ("S2-4 cost saving incl. tariff (%)", "stage2/S2-4/cost_savings"),
         ("v2 tariff on a larger host (USD per episode)", "stage2_v2/profile/tariff/q"),
         ("v2 cost saving incl. that tariff (%)", "stage2_v2/profile/cost_savings")],
     "sources": ["results/calibration/tauhat_jev_shadow_cal_tool_*.json", "results/gpu_round5/cal/",
                 "results/gpu_round5/FREEZE_RECORD.json", "results/gpu_round5/latency_*.txt",
                 "results/stage2/tariff.json", "results/stage2/tariff_v2.json"],
     "caveats": ["serve_jev serialises requests on one GPU; latency grows with concurrency.",
                 "The v2 tariff was measured on the only one-GPU host available that day."]},
    {"id": "T6", "topic": "Development instruments",
     "claim": "Build a development check that the SFT model does not saturate before training RL stages; a "
              "saturated slice cannot tell stages apart.",
     "label": "development (v1 slice) and exploratory (v2 stress slice)",
     "support": "Supported: every Flash arm of the v1 slice reached 100% from SFT v1 on; the harder v2 stress slice "
                "separated SFT, OPD, GRPO and the references.",
     "evidence": [
         ("v1 dev slice, tool replace success: SFT v1 (%)", "rl/dev/sft_v1/ts_replace_success"),
         ("v1 dev slice, tool replace success: GRPO it. 15 (%)", "rl/dev/grpo_i15/ts_replace_success"),
         ("v1 dev slice, tool replace success: OPD it. 12 (%)", "rl/dev/opd_i12/ts_replace_success"),
         ("v2 stress success: SFT v2 (%)", "rl2/stress/sft_v2/S"),
         ("v2 stress success: OPD it. 12 (%)", "rl2/stress/opd_i12/S"),
         ("v2 stress success: GRPO it. 15 (%)", "rl2/stress/grpo_i15/S"),
         ("v2 stress success: Jev (%)", "rl2/stress/jev/S"),
         ("v2 stress success: Intern-Decision-4B (%)", "rl2/stress/intern/S")],
     "sources": ["results/gpu_round5/ablation.md", "results/gpu_round6_miles/ablation.md"],
     "caveats": ["The stress slice has 200 development tasks of one generator with harder settings."]},
    {"id": "T7", "topic": "Cost",
     "claim": "The recipe that carried the gain is cheap: about $5 of data and about two GPU-hours of SFT on one "
              "L40S. The RL stages and development evaluation were the expensive parts.",
     "label": "spend records (descriptive)",
     "support": "Supported by the ledgers; the 'SFT-only' cost is a sum of recorded items (data build + the SFT "
                "stage's host time), not a separate run.",
     "evidence": [
         ("data v1 build (USD)", "spend/api/training_v1_data"),
         ("data v1: episodes (USD)", "recipe/data/v1/episodes_usd"),
         ("data v1: Jev labels (USD)", "recipe/data/v1/teacher_usd"),
         ("SFT v1 stage (GPU hours)", "recipe/stage/sft_v1/gpu_h"),
         ("SFT v1 stage host cost (USD)", "recipe/cost/sft_gpu_usd"),
         ("data + SFT (USD)", "recipe/cost/sft_only_usd"),
         ("OPD + GRPO stages (GPU hours)", "recipe/stage/rl_extra/gpu_h"),
         ("OPD + GRPO host cost (USD)", "recipe/stage/rl_extra/gpu_usd"),
         ("OPD + GRPO API (rollouts, teacher, greedy evaluations; USD)", "recipe/stage/rl_extra/api_usd"),
         ("OPD teacher (USD)", "recipe/stage/opd/teacher_usd"),
         ("round-5 development slices (USD)", "recipe/cost/dev_slices_usd"),
         ("share of round-5 API spend (%)", "recipe/cost/dev_slices_share"),
         ("round-5 API total (USD)", "spend/api/training_round5"),
         ("round-5 GPU hours", "spend/gpu/round5/hours"),
         ("round-5 GPU (USD)", "spend/gpu/round5/usd"),
         ("amortisation of round 5 + data at S2-4's saving (episodes)", "spend/amortisation/episodes")],
     "sources": ["results/gpu_round5/{spend.json, status.txt}", "results/training/train_v1/manifest.json",
                 "results/spend/spend_ledger.md"],
     "caveats": ["GPU hours are host hours of one L40S (g6e.4xlarge, $3.004 / h); GPU-hours per recipe and peak "
                 "memory beyond the stage times were not logged.",
                 "v2 (round 6) spend: API recorded, GPU bill not logged."]},
    {"id": "T8", "topic": "Limits",
     "claim": "Everything above holds inside the training generator. Outside it the trained model agreed with the "
              "reference less often than its teacher, and the one live attempt stopped at its go/no-go check.",
     "label": "descriptive (tau2 reference agreement, A7.13 (iv)); exploratory (telecom go/no-go)",
     "support": "Supported as a limit; there is no out-of-generator success number.",
     "evidence": [
         ("tau2 tool selection agreement: frozen - Jev (points)", "t2ref/all/tool_select/jevsub/frozen/delta"),
         ("tau2 argument agreement: frozen - Jev (points)", "t2ref/all/tool_params/jevsub/frozen/delta"),
         ("tau2 tool selection agreement: frozen - its base (points)", "t2ref/all/tool_select/frozen_vs_base/delta"),
         ("tau2 argument agreement: frozen - its base (points)", "t2ref/all/tool_params/frozen_vs_base/delta"),
         ("tau2: always respond_to_user agreement (%)", "t2ref/all/tool_select/majority/agree"),
         ("tau2: frozen tool selection agreement (%)", "t2ref/all/tool_select/frozen/agree"),
         ("telecom go/no-go rounds", "explore/telecom_rl/gonogo/rounds"),
         ("telecom decision", "explore/telecom_rl/gonogo/decision"),
         ("telecom round 2: identical consecutive calls", "explore/telecom_rl/gonogo/r2/max_repeats"),
         ("fresh JevGym tasks: distinct plan shapes", "data_stats/jevgym_eval/n_distinct_plans")],
     "sources": ["results/stage2/t2ref/t2ref.md", "results/explore/telecom_rl/gonogo*.md",
                 "paper/generated/data_stats.tex"],
     "caveats": ["Agreement with one reference trajectory is not oracle accuracy.",
                 "Telecom round 1 failed on an adapter defect (fixed); round 2 failed on one of five validation "
                 "tasks (identical repeated calls). It is a stop, not a success measurement."]},
]

NOT_SUPPORTED = [
    "That recovery states, rather than data size, task setting or labels, explain the v0 -> v1 gain (confounded).",
    "That teacher soft labels beat gold-only labels, or the reverse (no label-source run).",
    "That GRPO improves success or calibration beyond a temperature fit (saturated slice; C ECE matched by "
    "temperature-scaled SFT; v2 GRPO lowered stress success).",
    "That on-policy distillation helps once the student is more accurate than the teacher (it lowered accuracy in "
    "both tracks).",
    "Any data-size curve: only two training sets (v0, v1) exist.",
    "Any claim outside the JevGym generator beyond the tau2 reference agreement.",
]


def referenced_keys() -> list[str]:
    return [k for t in TAKEAWAYS for _, k in t["evidence"]]


def render_md(values: Mapping[str, str], gen: Mapping[str, str]) -> str:
    allv = {**gen, **values}
    L = ["# Recipe takeaways: training the decision model (JevAgentRL)",
         "",
         "Generated by `scripts/recipe_takeaways.py` from committed records only (no checkpoint, GPU or API). Numbers "
         "are printed as the paper prints them (`\\res{key}`); keys under `recipe/` are written by this script to "
         "`paper/generated/recipe.tex`, every other key comes from the generated file named in its prefix "
         "(`rl/` rl.tex, `rl2/` rl.tex, `stage2*/` stage2.tex, `spend/` spend.tex, `t2ref/` t2ref.tex, `explore/` "
         "explore.tex, `data_stats/` data_stats.tex).",
         "",
         "Labels: **registered** = a pre-registered test decides it; **descriptive** = evaluation data, no test; "
         "**development** = training-side data (v1 dev, the 50-task development slice, the gate set), never an "
         "evaluation number; **exploratory** = the v2 Miles track or the 2026-10-05 exploratory rows.",
         "",
         "| # | topic | takeaway | label |",
         "|---|---|---|---|"]
    for t in TAKEAWAYS:
        L.append(f"| {t['id']} | {t['topic']} | {t['claim']} | {t['label']} |")
    for t in TAKEAWAYS:
        L += ["", f"## {t['id']}. {t['topic']}", "", f"**Takeaway.** {t['claim']}", "",
              f"**Label.** {t['label']}.", "", f"**What the records support.** {t['support']}", "",
              "| evidence | value | key |", "|---|---|---|"]
        for desc, k in t["evidence"]:
            val = allv.get(k)
            L.append(f"| {desc} | {plain(val) if val is not None else 'MISSING'} | `{k}` |")
        L += ["", "Sources: " + "; ".join(f"`{s}`" for s in t["sources"]) + ".", ""]
        L += ["Caveats:"] + [f"- {c}" for c in t["caveats"]]
    L += ["", "## Not supported by the records", ""] + [f"- {x}" for x in NOT_SUPPORTED]
    L += ["", "## Keys written to `paper/generated/recipe.tex`", "", "| key | value |", "|---|---|"]
    L += [f"| `{k}` | {plain(x)} |" for k, x in sorted(values.items())]
    return "\n".join(L) + "\n"


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render_tex(values: Mapping[str, str]) -> str:
    lines = ["% AUTO-GENERATED by scripts/recipe_takeaways.py -- do not edit by hand.",
             f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git {_git_head()}",
             "% Recipe takeaways (results/training/recipe_takeaways.md): derived from committed training, development",
             "% and Stage-2 records; development numbers are never evaluation numbers.",
             "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(values.items())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, str]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-tex", default=str(OUT_TEX))
    ap.add_argument("--out-md", default=str(OUT_MD))
    a = ap.parse_args(argv)
    gen = generated_keys()
    values, _ = build(gen)
    missing = [k for k in referenced_keys() if k not in values and k not in gen]
    if missing:
        raise SystemExit(f"takeaway keys without a value: {missing}")
    for path, text in ((a.out_tex, render_tex(values)), (a.out_md, render_md(values, gen))):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    print(f"wrote {len(values)} keys to {a.out_tex} and {len(TAKEAWAYS)} takeaways to {a.out_md}")
    return values


if __name__ == "__main__":
    main()
