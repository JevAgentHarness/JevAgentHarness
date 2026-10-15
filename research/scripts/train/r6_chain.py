"""The mechanical JevAgentRL-v2 decisions (GPU round 6, the Miles track; results/gpu_round6_miles/TRAINING_MANIFEST.md).

This script, not a person, picks the base model, keeps or drops each stage of SFT v2 -> OPD v2 -> GRPO v2, retains one
continuation per kept stage and names the frozen checkpoint. It reads development artefacts only: the smoke reports of
the Miles image, the decision-level evaluations (``eval_<tag>_gate.json`` / ``eval_<tag>_v1.json``,
``training/eval_decisions.py`` on the 210-decision gate set and data/train_v1's dev files), the §18.4 go / no-go files
(``gonogo_<tag>.json``) and the STRESS instrument summaries (``stress/stress_<tag>_summary.json``: development seeds
2000-2199, ``configs/experiments/jevgym_stress_dev_miles.yaml``, ``scripts/train/r5_summary.py``).

    python scripts/train/r6_chain.py base   --out runs/gpu_round6_miles/base.json
    python scripts/train/r6_chain.py decide --out runs/gpu_round6_miles/chain_decision.json
    python scripts/train/r6_chain.py ablation --out runs/gpu_round6_miles/ablation

Rules (fixed in the manifest before any training step; accuracies on [0, 1]). Notation per checkpoint X: V = v1-dev
accuracy, Rc = dev_recovery accuracy, E = v1-dev ECE (raw); on the stress slice: S = success, A = own-state tool_select
accuracy, R = mean call-count-penalised success (the GRPO reward formula), C = (S + A) / 2.
  BASE   StartLux-Decision-4B iff S1 (4 Miles FSDP SFT steps with finite loss), S2 (the SGLang engine serves the
         model faithfully on 32 v1-dev prompts: argmax agreement with the fp32 HF forward >= 0.95, mean total-variation
         distance of the option distributions <= 0.02, max <= 0.10; amended 2026-10-03 before any base decision,
         manifest §9: the first wording, max |delta log p| <= 0.1 over every option vs the bf16 HF forward, measured
         bf16 noise on options without probability mass), S3 (fla gated-delta-rule and causal-conv1d kernels run on the
         GPU) and S4 (zero-shot v1-dev accuracy through serve_jev's prompt >= 0.766 and >= Qwen3-4B's on the same rows)
         all hold; else Qwen3-4B (which must pass S1 and S2; otherwise the track stops: no base).
  SFT    the base of the chain (tag sft_v2); kept when it was evaluated.
  OPD    continuations opd_i12, opd (step order), each against SFT v2. PASS iff
           (keep)    V >= SFT - 0.005, Rc >= SFT - 0.010, E <= SFT + 0.010, S >= SFT - 0.015, A >= SFT - 0.010, and
           (gain)    [V >= SFT + 0.010 and Rc >= SFT]  or  [S >= SFT + 0.030]  or  [A >= SFT + 0.020].
  GRPO   trained from the retained continuation of the last kept stage; continuations grpo_i15, grpo, each against that
         init. PASS iff the GRPO validity tests passed and the on-host replay check passed (A7.10 (f)), V >= init -
         0.010, Rc >= init - 0.010, E <= init + 0.020, R > init's R + 0.010, S >= init - 0.010.
  RETAIN a stage is KEPT iff at least one continuation passes; among passing continuations the retained one has the
         highest C, then the highest V, then the later step.
  FROZEN the retained continuation of the last kept stage that passes G1-G5 (``data_v1 gonogo``, G3 on the actual
         Intern-Decision-4B v1-dev reference) AND G6 (stress vs Intern-Decision-4B on the same slice: A >= Intern's A +
         0.05 and S >= Intern's S) is frozen as GO; if no kept stage passes, the retained continuation of the last kept
         stage is frozen as NO-GO. No RL claim unless the frozen stage is GRPO.
A missing input makes the rule it feeds fail (the continuation does not pass), and the reason is written down.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

OUT = Path("runs/gpu_round6_miles")
OPD_CONT = ("opd_i12", "opd")
GRPO_CONT = ("grpo_i15", "grpo")
STRESS_ARM = "ours_stress_ts_params_replace"
INTERN_STRESS_ARM = "intern_stress_ts_params_replace"
BASES = {"startlux": "StartLux-Decision-4B", "qwen": "Qwen3-4B"}


def _load(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except json.JSONDecodeError:
        return None


def _file(ev: dict[str, Any] | None, suffix: str) -> dict[str, Any]:
    for f, m in ((ev or {}).get("files") or {}).items():
        if f.endswith(suffix) or f.endswith(suffix + ".gz"):
            return m
    return {}


def _ge(a: float | None, b: float | None, margin: float) -> bool:
    return a is not None and b is not None and a >= b + margin - 1e-12


def _le(a: float | None, b: float | None, margin: float) -> bool:
    return a is not None and b is not None and a <= b + margin + 1e-12


# =============================================================================== read-outs
def decision_level(eval_dir: Path, tag: str) -> dict[str, Any]:
    v1 = _load(eval_dir / f"eval_{tag}_v1.json")
    gate = _load(eval_dir / f"eval_{tag}_gate.json")
    dev, rec = _file(v1, "train_v1/dev.jsonl"), _file(v1, "train_v1/dev_recovery.jsonl")
    ko = _file(v1, "train_v1/dev_knockout.jsonl")
    go = _load(eval_dir / f"gonogo_{tag}.json") or {}
    src = ((go.get("references") or {}).get("intern_v1_dev") or {}).get("source")
    gdev = _file(gate, "train_v0/dev.jsonl")
    by_dp = {dp: (dev.get(dp) or (dev.get("by_dp") or {}).get(dp) or {}).get("acc")
             for dp in ("tool_select", "tool_params", "stop")}
    return {"acc": (dev.get("overall") or {}).get("acc"), "recovery": (rec.get("overall") or {}).get("acc"),
            "knockout": (ko.get("overall") or {}).get("acc"), "ece": (dev.get("overall") or {}).get("ece"),
            "gate_acc": (gdev.get("overall") or {}).get("acc"), "by_dp": by_dp, "intern_reference": src,
            "go_g1_g5": bool(go.get("go")) and src == "v1 dev",
            "gates": {k: v.get("pass") for k, v in (go.get("checks") or {}).items()}}


def stress(out: Path, run: str, arm: str = STRESS_ARM) -> dict[str, Any]:
    rows = _load(out / "stress" / f"{run}_summary.json") or {}
    r = next((v for k, v in rows.items() if k.split("::")[-1] == arm), None) or {}
    S = r.get("success")
    A = r.get("tool_select/replace/s1_acc")
    return {"S": S, "A": A, "P": r.get("tool_params/replace/s1_acc"), "R": r.get("reward_ccps"),
            "C": (S + A) / 2 if S is not None and A is not None else None, "n": r.get("n"),
            "A_n": r.get("tool_select/replace/n"), "cost_usd": r.get("cost_usd"), "errors": r.get("errors")}


# =============================================================================== base rule
def _finite_losses(log: Path) -> tuple[bool, list[float]]:
    if not log.exists():
        return False, []
    vals = [float(x) for x in re.findall(r"'train/loss':\s*([-+0-9.eEnaif]+)", log.read_text(errors="ignore"))]
    return bool(vals) and all(math.isfinite(v) for v in vals), vals[:8]


def _kernels(out: Path) -> dict[str, Any]:
    txt = (out / "smoke_env.txt").read_text(errors="ignore") if (out / "smoke_env.txt").exists() else ""
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        rep = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        rep = {}
    k = rep.get("kernels") or {}
    return {"report": k, "ok": str(k.get("causal_conv1d", "")).startswith("ok")
            and str(k.get("fla_gated_delta_rule", "")).startswith("ok")}


def base_rule(out: Path, models: Path) -> dict[str, Any]:
    eval_dir = out / "eval"
    kern = _kernels(out)
    res: dict[str, Any] = {"checks": {}}
    for key, name in BASES.items():
        eng = _load(out / f"smoke_engine_{name}.json") or {}
        s1, losses = _finite_losses(out / f"smoke_train_{name}.log")
        acc = decision_level(eval_dir, f"zeroshot_{key}")["acc"]
        res["checks"][key] = {"S1_miles_fsdp_sft_finite": s1, "losses": losses,
                              "S2_sglang_vs_hf_max_abs_dlogp": eng.get("max_abs_dlogp"),
                              "S2": bool(eng.get("pass_s2")), "zeroshot_v1_dev_acc": acc}
    s, q = res["checks"]["startlux"], res["checks"]["qwen"]
    s["S3_kernels"] = kern["ok"]
    s["S4"] = _ge(s["zeroshot_v1_dev_acc"], 0.766, 0.0) and _ge(s["zeroshot_v1_dev_acc"], q["zeroshot_v1_dev_acc"], 0.0)
    res["kernels"] = kern["report"]
    if s["S1_miles_fsdp_sft_finite"] and s["S2"] and s["S3_kernels"] and s["S4"]:
        res["base"] = "startlux"
    elif q["S1_miles_fsdp_sft_finite"] and q["S2"]:
        res["base"] = "qwen"
    else:
        res["base"] = None
    res["name"] = BASES.get(res["base"]) if res["base"] else None
    res["path"] = str(models / res["name"]) if res["name"] else None
    return res


# =============================================================================== keep / retain / freeze
def opd_passes(c: dict, b: dict, sc: dict, sb: dict) -> tuple[bool, dict]:
    keep = {"V>=sft-0.005": _ge(c["acc"], b["acc"], -0.005), "Rc>=sft-0.01": _ge(c["recovery"], b["recovery"], -0.010),
            "E<=sft+0.01": _le(c["ece"], b["ece"], 0.010), "S>=sft-0.015": _ge(sc["S"], sb["S"], -0.015),
            "A>=sft-0.01": _ge(sc["A"], sb["A"], -0.010)}
    gain = {"V>=sft+0.01&Rc>=sft": _ge(c["acc"], b["acc"], 0.010) and _ge(c["recovery"], b["recovery"], 0.0),
            "S>=sft+0.03": _ge(sc["S"], sb["S"], 0.030), "A>=sft+0.02": _ge(sc["A"], sb["A"], 0.020)}
    ok = all(keep.values()) and any(gain.values())
    return ok, {"keep": keep, "gain": gain}


def grpo_passes(c: dict, i: dict, sc: dict, si: dict, validity_ok: bool, replay_ok: bool) -> tuple[bool, dict]:
    r = {"validity_tests_passed": validity_ok, "replay_check_passed": replay_ok,
         "V>=init-0.01": _ge(c["acc"], i["acc"], -0.010), "Rc>=init-0.01": _ge(c["recovery"], i["recovery"], -0.010),
         "E<=init+0.02": _le(c["ece"], i["ece"], 0.020),
         "R>init+0.01": sc["R"] is not None and si["R"] is not None and sc["R"] > si["R"] + 0.010,
         "S>=init-0.01": _ge(sc["S"], si["S"], -0.010)}
    return all(r.values()), r


def retain(passing: list[str], dl: dict, st: dict, order: tuple[str, ...]) -> str | None:
    if not passing:
        return None
    return max(passing, key=lambda t: (st[t]["C"], dl[t]["acc"], order.index(t)))


def g6(sx: dict, si: dict) -> dict[str, Any]:
    a = _ge(sx.get("A"), si.get("A"), 0.05)
    s = _ge(sx.get("S"), si.get("S"), 0.0)
    return {"A>=intern+0.05": a, "S>=intern": s, "pass": a and s, "intern": si}


def decide(out: Path, validity_ok: bool, replay_ok: bool) -> dict[str, Any]:
    eval_dir = out / "eval"
    tags = ("sft_v2",) + OPD_CONT + GRPO_CONT
    dl = {t: decision_level(eval_dir, t) for t in tags}
    st = {t: stress(out, f"stress_{t}") for t in tags}
    si = stress(out, "stress_refs_intern", INTERN_STRESS_ARM)
    res: dict[str, Any] = {"stages": {}, "decision_level": dl, "stress": st, "stress_intern": si}
    kept: list[str] = []
    if dl["sft_v2"]["acc"] is not None:
        kept.append("sft_v2")
    res["stages"]["sft_v2"] = {"kept": bool(kept), "retained": "sft_v2" if kept else None}
    out_c, passing = {}, []
    for t in OPD_CONT:
        if dl[t]["acc"] is None or not kept:
            out_c[t] = {"pass": False, "why": "not trained / not evaluated / no SFT v2"}
            continue
        ok, why = opd_passes(dl[t], dl["sft_v2"], st[t], st["sft_v2"])
        out_c[t] = {"pass": ok, **why}
        if ok:
            passing.append(t)
    r = retain(passing, dl, st, OPD_CONT)
    res["stages"]["opd"] = {"kept": r is not None, "retained": r, "against": "sft_v2", "continuations": out_c}
    if r:
        kept.append(r)
    init = kept[-1] if kept else None
    res["grpo_init"] = init
    out_c, passing = {}, []
    for t in GRPO_CONT:
        if dl[t]["acc"] is None or init is None:
            out_c[t] = {"pass": False, "why": "not trained / not evaluated"}
            continue
        ok, why = grpo_passes(dl[t], dl[init], st[t], st[init], validity_ok, replay_ok)
        out_c[t] = {"pass": ok, "rule": why}
        if ok:
            passing.append(t)
    r = retain(passing, dl, st, GRPO_CONT)
    res["stages"]["grpo"] = {"kept": r is not None, "retained": r, "against": init, "continuations": out_c}
    if r:
        kept.append(r)
    gates = {t: {"G1_G5": dl[t]["go_g1_g5"], "G6": g6(st[t], si)} for t in kept}
    go = [t for t in kept if gates[t]["G1_G5"] and gates[t]["G6"]["pass"]]
    frozen = go[-1] if go else (kept[-1] if kept else None)
    res.update({"kept": kept, "gates": gates, "frozen": frozen, "go": bool(go),
                "rl_claim": bool(frozen and frozen.startswith("grpo")),
                "label": ("GO: the frozen JevAgentRL-v2 passes G1-G6 (development; generator-adapted vs off-the-shelf)"
                          if go else "NO-GO: frozen as a training-study row (exploratory)")})
    return res


# =============================================================================== ablation table
def validity_passed(text: str) -> bool:
    """Did the GRPO validity tests pass? The ``pytest_exit=N`` line that the host stage records decides; without it,
    the pytest output: a "N passed" summary without failures, or -- under the repository's ``addopts = "-q"`` plus
    ``-q`` (no summary line) -- progress lines of only passes / skips that reach [100%]."""
    m = re.findall(r"pytest_exit=(\d+)", text)
    if m:
        return m[-1] == "0"
    low = text.lower()
    if re.search(r"\b\d+ (failed|errors?)\b", low) or "traceback" in low:
        return False
    if re.search(r"\b\d+ passed\b", low):
        return True
    prog = [ln.split("[")[0].strip() for ln in text.splitlines() if re.match(r"^[.sxFE]+\s+\[\s*\d+%\]\s*$", ln.strip())]
    return (bool(prog) and text.rstrip().endswith("[100%]") and all(set(p) <= {".", "s"} for p in prog)
            and any("." in p for p in prog))


def ablation(out: Path, v1_eval: Path, v1_e2dev: Path) -> dict[str, Any]:
    eval_dir = out / "eval"
    chain = _load(out / "chain_decision.json") or {}
    base = (_load(out / "base.json") or {}).get("base") or "qwen"
    rows: dict[str, Any] = {}
    order = [f"zeroshot_{base}", "sft_v2", *OPD_CONT, *GRPO_CONT]
    for t in order:
        d = decision_level(eval_dir, t)
        if d["acc"] is None:
            continue
        rows[t] = {"decision": d, "stress": stress(out, f"stress_{t}"),
                   "e2dev": (_load(out / "e2dev" / f"e2dev_v2_{t}_summary.json") or {})}
    v1 = {t: decision_level(v1_eval, t) for t in ("sft_v0", "sft_v1", "opd_i12", "opd", "grpo_i15", "grpo")}
    v1 = {t: r for t, r in v1.items() if r["acc"] is not None}
    return {"v2": rows, "v1_reference": v1, "v1_e2dev_dir": str(v1_e2dev), "chain": {k: chain.get(k) for k in
            ("kept", "frozen", "go", "label")}}


def ablation_md(tab: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    head = ["row", "gate acc", "v1-dev acc", "ts", "params", "stop", "recovery", "knockout", "ECE",
            "stress S", "stress A", "stress P", "stress R"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t, r in tab["v2"].items():
        d, s = r["decision"], r["stress"]
        lines.append("| " + " | ".join([f"v2 {t}", f(d["gate_acc"]), f(d["acc"]), f(d["by_dp"]["tool_select"]),
                                         f(d["by_dp"]["tool_params"]), f(d["by_dp"]["stop"]), f(d["recovery"]),
                                         f(d["knockout"]), f(d["ece"]), f(s["S"]), f(s["A"]), f(s["P"]),
                                         f(s["R"])]) + " |")
    for t, d in tab["v1_reference"].items():
        lines.append("| " + " | ".join([f"v1 {t}", f(d["gate_acc"]), f(d["acc"]), f(d["by_dp"]["tool_select"]),
                                         f(d["by_dp"]["tool_params"]), f(d["by_dp"]["stop"]), f(d["recovery"]),
                                         f(d["knockout"]), f(d["ece"]), "-", "-", "-", "-"]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["base", "decide", "ablation"])
    ap.add_argument("--dir", default=str(OUT))
    ap.add_argument("--models", default="/opt/jev/models")
    ap.add_argument("--grpo-validity", default=str(OUT / "grpo_validity.txt"))
    ap.add_argument("--replay-check", default=str(OUT / "grpo_replay_check.json"))
    ap.add_argument("--v1-eval", default="results/gpu_round5/eval")
    ap.add_argument("--v1-e2dev", default="results/gpu_round5/e2dev")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.dir)
    if a.cmd == "base":
        res = base_rule(out, Path(a.models))
    elif a.cmd == "decide":
        vt = Path(a.grpo_validity).read_text() if Path(a.grpo_validity).exists() else ""
        validity_ok = validity_passed(vt)
        replay_ok = bool((_load(Path(a.replay_check)) or {}).get("pass"))
        res = decide(out, validity_ok, replay_ok)
        res["inputs"] = {"grpo_validity_ok": validity_ok, "replay_check_ok": replay_ok}
    else:
        res = ablation(out, Path(a.v1_eval), Path(a.v1_e2dev))
        Path(a.out + ".md").write_text(ablation_md(res))
    p = Path(a.out if a.cmd != "ablation" else a.out + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res.get(k) for k in ("base", "name", "kept", "frozen", "go", "label") if k in res}, indent=1))


if __name__ == "__main__":
    main()
