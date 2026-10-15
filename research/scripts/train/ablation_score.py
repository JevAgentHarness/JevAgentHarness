#!/usr/bin/env python3
"""Score the exploratory SFT training ablation (plan: results/training/ablation_plan.md). Offline: no GPU, no API.

Reads one run directory (default runs/train_ablation, as synced from s3://.../train_ablation/):
  data/manifest.json             scripts/train/ablation_subsets.py build (rows, tasks, recovery share per arm)
  eval/<arm>_v1.json             eval_decisions on v1 dev / dev_recovery / dev_knockout (raw temperature)
  eval/<arm>_v1_cal.json         the same with the arm's temperatures (in-sample: fitted on v1 dev; optional)
  s23/answers_abl_<arm>.jsonl    the arm's answers to every distinct question of the common-state corpus C
                                 (scripts/stage2_corpus.py replay on a copy of results/stage2/s23)
  status.txt                     driver status lines "stage sft_<arm> exit=0 (<m> min)" (GPU minutes)
and the frozen corpus results/stage2/s23 (occurrences, eligible tasks) plus its published answer files, which are
scored with the same code as reference rows (pub_sft_v1 = SFT v1, pub_frozen = JevAgentRL-4B, pub_jev, pub_base).

Per arm: v1-dev accuracy (all, tool_select, tool_params, stop, after an off-plan step, its tool selection, on-plan,
knockout), raw and calibrated ECE; on C: task-mean accuracy (all, after an off-plan step), occurrence-level ECE of the
answers' confidence (as scripts/stage2_analysis.py s1_rows). Contrasts (CONTRASTS): differences of the v1-dev point
estimates and, on C, a paired task bootstrap (B = 2,000, seed 0; descriptive, no test). Writes <out>/ablation.json,
<out>/ablation.md and, with --tex, \\defres keys recipe_abl/<arm>/<metric> and recipe_abl/contrast/<name>/<metric>.

    python scripts/train/ablation_score.py --run runs/train_ablation --out results/training/ablation \\
        [--tex paper/generated/recipe_ablation.tex]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench import stage2_corpus as C  # noqa: E402

S23 = ROOT / "results" / "stage2" / "s23"
V1 = {"dev": "data/train_v1/dev.jsonl.gz", "rec": "data/train_v1/dev_recovery.jsonl.gz",
      "ko": "data/train_v1/dev_knockout.jsonl.gz"}
PUBLISHED = {"pub_sft_v1": "sft_v1", "pub_frozen": "rl", "pub_jev": "jev", "pub_base": "base"}
ANCHOR = "sft_v1"              # the existing SFT v1 checkpoint, evaluated through this pipeline (reproduction anchor)
REF = "D100"                   # SFT v1's data and labels retrained on the ablation host: the within-stack reference
# name -> (arm, reference)
CONTRASTS: dict[str, tuple[str, str]] = {
    "size_10": ("D10", REF), "size_25": ("D25", REF), "size_50": ("D50", REF),
    "gold_vs_mixed_100": ("L_gold_100", REF), "teacher_vs_mixed_100": ("L_teach_100", REF),
    "gold_vs_teacher_100": ("L_gold_100", "L_teach_100"),
    "gold_vs_mixed_25": ("L_gold_25", "D25"), "teacher_vs_mixed_25": ("L_teach_25", "D25"),
    "canon_vs_match": ("S_canon", "S_match"), "nonm_vs_match": ("S_nonm", "S_nonm_match"),
    "canon_vs_match_25": ("S_canon_25", "S_match_25"),
    "seed_10": ("D10_s1", "D10"), "seed_25": ("D25_s1", "D25"), "seed_100": ("D100_s1", REF),
    "retrain_vs_anchor": (REF, ANCHOR), "anchor_vs_published": (ANCHOR, "pub_sft_v1"),
    # the core block of the plan's addendum (no D100): the size curve within the ablation host (vs D50) and against
    # the existing SFT v1 checkpoint (the 100 % point, trained on the round-5 host: a cross-stack comparison)
    "size_10_vs_50": ("D10", "D50"), "size_25_vs_50": ("D25", "D50"),
    "size_10_vs_anchor": ("D10", ANCHOR), "size_25_vs_anchor": ("D25", ANCHOR), "size_50_vs_anchor": ("D50", ANCHOR),
}
N_BOOT = 2000


# ------------------------------------------------------------------ v1 dev (eval_decisions output)
def v1_metrics(path: Path) -> dict[str, float]:
    f = json.loads(path.read_text())["files"]
    out: dict[str, float] = {}
    d = f.get(V1["dev"])
    if d:
        o = d["overall"]
        out.update({"v1_acc": o["acc"], "v1_ece": o["ece"], "v1_nll": o["nll"], "v1_n": o["n"]})
        for dp in ("tool_select", "tool_params", "stop"):
            if dp in d.get("by_dp", {}):
                out[f"v1_{dp}"] = d["by_dp"][dp]["acc"]
    r = f.get(V1["rec"])
    if r:
        out["v1_recovery"] = r["overall"]["acc"]
        if "tool_select" in r.get("by_dp", {}):
            out["v1_recovery_tool_select"] = r["by_dp"]["tool_select"]["acc"]
        if d:
            hits = round(d["overall"]["acc"] * d["overall"]["n"]) - round(r["overall"]["acc"] * r["overall"]["n"])
            out["v1_onplan"] = hits / (d["overall"]["n"] - r["overall"]["n"])
    k = f.get(V1["ko"])
    if k:
        out["v1_knockout"] = k["overall"]["acc"]
    return out


# ------------------------------------------------------------------ common states C
def load_corpus(cdir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with gzip.open(cdir / "occurrences.jsonl.gz", "rt") as f:
        occ = [json.loads(x) for x in f if x.strip()]
    return occ, [str(t) for t in json.loads((cdir / "corpus.json").read_text())["eligible_tasks"]]


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    recs: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["qhash"]] = r          # the latest record of a question decides (as stage2_analysis)
    return recs


def c_metrics(occ: Sequence[Mapping[str, Any]], eligible: Sequence[str], ans: Mapping[str, Mapping[str, Any]]
              ) -> tuple[dict[str, float], dict[str, float]]:
    acc = C.task_accuracy(occ, ans, eligible)
    off = C.task_accuracy([o for o in occ if o.get("stratum") == "after_off_plan"], ans, eligible)
    keep = set(eligible)
    conf, corr = [], []
    for o in occ:
        if o["task"] in keep:
            a = ans.get(o["qhash"]) or {}
            conf.append(float(a.get("confidence") or 0.0) if not a.get("error") else 0.0)
            corr.append(float(C.answer_correct(o, a)))
    asked = {o["qhash"] for o in occ if o["task"] in keep}
    out = {"c_acc": float(np.mean(list(acc.values()))) if acc else float("nan"), "c_tasks": len(acc),
           "c_acc_offplan": float(np.mean(list(off.values()))) if off else float("nan"), "c_tasks_offplan": len(off),
           "c_ece": M.ece(conf, corr) if conf else float("nan"),
           "c_missing": len(asked - set(ans)), "c_errors": sum(1 for q in asked if (ans.get(q) or {}).get("error"))}
    return out, acc


def paired_boot(a: Mapping[str, float], b: Mapping[str, float], n_boot: int = N_BOOT, seed: int = 0
                ) -> dict[str, float]:
    tasks = sorted(set(a) & set(b))
    d = np.array([a[t] - b[t] for t in tasks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    return {"delta": float(d.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "n_tasks": len(tasks)}


def stage_minutes(status: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"stage sft_(\S+) exit=0 \((\d+) min\)", status)}


# ------------------------------------------------------------------ score
def score(run: Path, corpus: Path = S23) -> dict[str, Any]:
    occ, eligible = load_corpus(corpus)
    man = json.loads((run / "data" / "manifest.json").read_text()) if (run / "data" / "manifest.json").exists() else {}
    mins = stage_minutes((run / "status.txt").read_text()) if (run / "status.txt").exists() else {}
    arms: set[str] = set()
    for p in (run / "eval").glob("*_v1.json"):
        arms.add(p.name[: -len("_v1.json")])
    for p in (run / "s23").glob("answers_abl_*.jsonl"):
        arms.add(p.name[len("answers_abl_"): -len(".jsonl")])
    rows: dict[str, dict[str, Any]] = {}
    per_task: dict[str, dict[str, float]] = {}
    for arm in sorted(arms):
        r: dict[str, Any] = {"label": "exploratory (training ablation)"}
        r.update({k: v for k, v in ((man.get("arms") or {}).get(arm) or {}).items()
                  if k in ("rows", "tasks", "recovery_share", "replay_factor")})
        if arm in mins:
            r["sft_min"] = mins[arm]
        p = run / "settings" / f"{arm}.json"       # the driver's record of the memory setting the arm trained with
        if p.exists():
            s = json.loads(p.read_text())
            r["train_setting"] = s.get("setting")
            r.update({k: s[k] for k in ("instance_type", "trainer_seed") if k in s})
        p = run / "eval" / f"{arm}_v1.json"
        if p.exists():
            r.update(v1_metrics(p))
        p = run / "eval" / f"{arm}_v1_cal.json"
        if p.exists():
            r["v1_ece_cal"] = v1_metrics(p).get("v1_ece")
        p = run / "s23" / f"answers_abl_{arm}.jsonl"
        if p.exists():
            m, acc = c_metrics(occ, eligible, load_answers(p))
            r.update(m)
            per_task[arm] = acc
        rows[arm] = r
    for name, src in PUBLISHED.items():       # reference rows: the published answers, scored with the same code
        p = corpus / f"answers_{src}.jsonl"
        if p.exists():
            m, acc = c_metrics(occ, eligible, load_answers(p))
            rows[name] = {"label": "published Stage-2 replay (descriptive)", **m}
            per_task[name] = acc
    contrasts: dict[str, Any] = {}
    for name, (a, b) in CONTRASTS.items():
        if a not in rows or b not in rows:
            continue
        c: dict[str, Any] = {"arm": a, "reference": b}
        for k in ("v1_acc", "v1_tool_select", "v1_recovery", "v1_recovery_tool_select", "v1_ece"):
            if k in rows[a] and k in rows[b]:
                c[f"{k}_diff"] = rows[a][k] - rows[b][k]
        if a in per_task and b in per_task:
            c["c_acc"] = paired_boot(per_task[a], per_task[b])
        if (ans_a := run / "s23" / f"answers_abl_{a}.jsonl").exists() and name == "anchor_vs_published":
            pa, pb = load_answers(ans_a), load_answers(corpus / f"answers_{PUBLISHED[b]}.jsonl")
            common = set(pa) & set(pb)
            c["answer_agreement"] = (sum(pa[q].get("answer") == pb[q].get("answer") for q in common) / len(common)
                                     if common else float("nan"))
        contrasts[name] = c
    return {"label": "exploratory SFT training ablation (Deviations entry in results/training/ablation_plan.md); "
                     "descriptive, outside both registered stages",
            "corpus": str(corpus), "n_boot": N_BOOT, "arms": rows, "contrasts": contrasts}


# ------------------------------------------------------------------ outputs
def _p(x: Any) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def _sg(x: float) -> str:
    return f"{100 * x:+.1f}"


def render_md(d: Mapping[str, Any]) -> str:
    L = ["# SFT training ablation (exploratory; descriptive)", "", d["label"] + ".", "",
         "| arm | rows | tasks | recov. share | SFT min | v1 acc | ts | tp | stop | recov. | recov. ts | ECE raw | "
         "ECE cal | C acc | C off-plan | C ECE | train setting |", "|" + "---|" * 17]
    for a, r in d["arms"].items():
        L.append(f"| {a} | {r.get('rows', '-')} | {r.get('tasks', '-')} | {_p(r.get('recovery_share'))} | "
                 f"{r.get('sft_min', '-')} | {_p(r.get('v1_acc'))} | {_p(r.get('v1_tool_select'))} | "
                 f"{_p(r.get('v1_tool_params'))} | {_p(r.get('v1_stop'))} | {_p(r.get('v1_recovery'))} | "
                 f"{_p(r.get('v1_recovery_tool_select'))} | "
                 + " | ".join("-" if r.get(k) is None else f"{r[k]:.3f}" for k in ("v1_ece", "v1_ece_cal"))
                 + f" | {_p(r.get('c_acc'))} | {_p(r.get('c_acc_offplan'))} | "
                 + ("-" if r.get("c_ece") is None else f"{r['c_ece']:.3f}") + f" | {r.get('train_setting') or '-'} |")
    L += ["", "| contrast | arm - reference | v1 acc | v1 recov. | C acc [task bootstrap 95 %] |", "|---|---|---|---|---|"]
    for n, c in d["contrasts"].items():
        ca = c.get("c_acc")
        L.append(f"| {n} | {c['arm']} - {c['reference']} | "
                 f"{_sg(c['v1_acc_diff']) if 'v1_acc_diff' in c else '-'} | "
                 f"{_sg(c['v1_recovery_diff']) if 'v1_recovery_diff' in c else '-'} | "
                 + (f"{_sg(ca['delta'])} [{_sg(ca['lo'])}, {_sg(ca['hi'])}]" if ca else "-") + " |")
    return "\n".join(L) + "\n"


def tex_keys(d: Mapping[str, Any]) -> dict[str, str]:
    v: dict[str, str] = {}
    for a, r in d["arms"].items():
        for k in ("v1_acc", "v1_tool_select", "v1_tool_params", "v1_stop", "v1_recovery", "v1_recovery_tool_select",
                  "v1_onplan", "c_acc", "c_acc_offplan"):
            if r.get(k) is not None and not np.isnan(r[k]):
                v[f"recipe_abl/{a}/{k}"] = f"{100 * r[k]:.1f}"
        for k in ("v1_ece", "v1_ece_cal", "c_ece"):
            if r.get(k) is not None and not np.isnan(r[k]):
                v[f"recipe_abl/{a}/{k}"] = f"{r[k]:.3f}"
        for k in ("rows", "tasks", "sft_min"):
            if r.get(k) is not None:
                v[f"recipe_abl/{a}/{k}"] = f"{int(r[k]):,}".replace(",", "{,}")
    for n, c in d["contrasts"].items():
        for k in ("v1_acc_diff", "v1_recovery_diff", "v1_recovery_tool_select_diff"):
            if k in c:
                v[f"recipe_abl/contrast/{n}/{k}"] = _sg(c[k])
        if c.get("c_acc"):
            for k in ("delta", "lo", "hi"):
                v[f"recipe_abl/contrast/{n}/c_acc_{k}"] = _sg(c["c_acc"][k])
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/train_ablation")
    ap.add_argument("--corpus", default=str(S23))
    ap.add_argument("--out", default="results/training/ablation")
    ap.add_argument("--tex", default=None)
    a = ap.parse_args(argv)
    d = score(Path(a.run), Path(a.corpus))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(d, indent=1, default=float))
    (out / "ablation.md").write_text(render_md(d))
    if a.tex:
        keys = tex_keys(d)
        lines = ["% AUTO-GENERATED by scripts/train/ablation_score.py -- do not edit by hand (exploratory ablation).",
                 "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
        Path(a.tex).write_text("\n".join(lines + [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(keys.items())]) + "\n")
    print(f"scored {len(d['arms'])} rows, {len(d['contrasts'])} contrasts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
