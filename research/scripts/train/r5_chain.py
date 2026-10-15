"""The mechanical JevAgentRL chain decision (GPU round 5; PREREGISTRATION A7.10 (a), results/gpu_round5/TRAINING_MANIFEST.md).

This script, not a person, keeps or drops each stage of SFT v1 -> OPD -> GRPO, retains one continuation per kept stage
and names the frozen checkpoint. It reads only development artefacts: the decision-level evaluations
(``eval_<tag>_gate.json`` / ``eval_<tag>_v1.json``, ``training/eval_decisions.py`` on the 210-decision gate set and
data/train_v1's dev files), the §18.4 go / no-go files (``gonogo_<tag>.json``) and the E2 development-slice summaries
(``e2dev_rl_<tag>_summary.json``, development seeds 1100-1149, ``scripts/train/r5_summary.py``).

    python scripts/train/r5_chain.py --eval-dir runs/gpu_round5/eval --e2dev runs/gpu_round5/e2dev \
        --grpo-validity runs/gpu_round5/grpo_validity.txt --out runs/gpu_round5/chain_decision.json

Rules (committed before any Stage-1 result was read; accuracies on [0, 1]):
  SFT v1   the base of the chain (tag sft_v1); always kept.
  OPD      continuations (tags in step order) opd_i12 (iteration 12), opd (iteration 24). A continuation PASSES against
           SFT v1 iff rule A or rule B holds:
           A (TRAINING_RECIPES §18.4)  v1-dev acc >= SFT + 0.010, dev_recovery acc >= SFT, v1-dev ECE <= SFT + 0.010
           B (in-harness)              v1-dev acc >= SFT - 0.005, dev_recovery acc >= SFT - 0.010, ECE <= SFT + 0.010,
                                       E2-dev own-state tool_select accuracy (ours_ts_replace + ours_ts_params_replace,
                                       pooled by decisions) >= SFT + 0.020, mean success over the live Flash arms
                                       >= SFT - 0.010
  GRPO     trained from the retained continuation of the last kept stage; continuations grpo_i15 (iteration 15), grpo
           (iteration 30). A continuation PASSES against its init iff the GRPO validity tests passed (A7.10 (f)), v1-dev acc
           >= init - 0.010, dev_recovery acc >= init - 0.010, ECE <= init + 0.020, the E2-dev mean call-count-penalised
           success over the live Flash arms is strictly above the init's and the mean success >= init - 0.010.
  RETAIN   a stage is KEPT iff at least one continuation passes; among passing continuations the retained one has the
           highest v1-dev accuracy, then the later step.
  FROZEN   the retained continuation of the last kept stage that passes the §18.4 gates G1-G5 (GO), where G3 must use the
           actual Intern-Decision-4B v1-dev reference (a gonogo built on the round-2 proxy counts as not passing); if no
           kept stage passes, the retained continuation of the last kept stage, labelled NO-GO (Intern-Decision-4B stays
           the open System-1; no RL claim unless the frozen stage is GRPO).
  Confirmation (reported with the GO, §18.4): on the same development seeds, ours_shadow tool_select accuracy >=
  intern_shadow's, ours_ts_replace success and own-state accuracy >= intern_ts_replace's.
Live Flash arms: ours_ts_replace, ours_ts_params_replace, ours_ts_assist_k3, ours_stop_guard, ours_ts_guard.
A missing input makes the rule it feeds fail (the continuation does not pass), and the reason is written down.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LIVE = ("ts_replace", "ts_params_replace", "ts_assist_k3", "stop_guard", "ts_guard")
OPD_CONT = ("opd_i12", "opd")          # step order
GRPO_CONT = ("grpo_i15", "grpo")


def _load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text()) if p.exists() else None


def _file(ev: dict[str, Any] | None, suffix: str) -> dict[str, Any]:
    for f, m in ((ev or {}).get("files") or {}).items():
        if f.endswith(suffix) or f.endswith(suffix + ".gz"):
            return m
    return {}


def decision_level(eval_dir: Path, tag: str) -> dict[str, Any]:
    v1 = _load(eval_dir / f"eval_{tag}_v1.json")
    dev, rec = _file(v1, "train_v1/dev.jsonl"), _file(v1, "train_v1/dev_recovery.jsonl")
    go = _load(eval_dir / f"gonogo_{tag}.json") or {}
    src = ((go.get("references") or {}).get("intern_v1_dev") or {}).get("source")
    return {"acc": (dev.get("overall") or {}).get("acc"), "recovery": (rec.get("overall") or {}).get("acc"),
            "ece": (dev.get("overall") or {}).get("ece"), "intern_reference": src,
            "go": bool(go.get("go")) and src == "v1 dev",
            "gates": {k: v.get("pass") for k, v in (go.get("checks") or {}).items()}}


def dev_slice(e2dev: Path, run: str, prefix: str = "ours") -> dict[str, Any]:
    rows = _load(e2dev / f"{run}_summary.json") or {}
    arms = {k.split("::")[-1]: v for k, v in rows.items()}
    live = [arms.get(f"{prefix}_{a}") for a in LIVE]
    live = [r for r in live if r]
    own_n = own_c = 0.0
    for a in ("ts_replace", "ts_params_replace"):
        r = arms.get(f"{prefix}_{a}") or {}
        acc, n = r.get("tool_select/replace/s1_acc"), r.get("tool_select/replace/n")
        if acc is not None and n:
            own_c += acc * n
            own_n += n
    sh = arms.get(f"{prefix}_shadow") or {}
    rp = arms.get(f"{prefix}_ts_replace") or {}
    full = len(live) == len(LIVE)
    return {"n_live_arms": len(live),
            "success_live": sum(r["success"] for r in live) / len(live) if full else None,
            "reward_live": (sum(r["reward_ccps"] for r in live) / len(live)
                            if full and all(r.get("reward_ccps") is not None for r in live) else None),
            "own_state_ts_acc": own_c / own_n if own_n else None,
            "shadow_ts_acc": sh.get("tool_select/shadow/s1_acc"),
            "ts_replace_success": rp.get("success"), "ts_replace_acc": rp.get("tool_select/replace/s1_acc")}


def _ge(a: float | None, b: float | None, margin: float) -> bool:
    return a is not None and b is not None and a >= b + margin - 1e-12


def _le(a: float | None, b: float | None, margin: float) -> bool:
    return a is not None and b is not None and a <= b + margin + 1e-12


def opd_passes(c: dict[str, Any], base: dict[str, Any], ec: dict[str, Any], eb: dict[str, Any]) -> tuple[bool, dict]:
    a = {"acc>=base+0.01": _ge(c["acc"], base["acc"], 0.010), "recovery>=base": _ge(c["recovery"], base["recovery"], 0.0),
         "ece<=base+0.01": _le(c["ece"], base["ece"], 0.010)}
    b = {"acc>=base-0.005": _ge(c["acc"], base["acc"], -0.005),
         "recovery>=base-0.01": _ge(c["recovery"], base["recovery"], -0.010),
         "ece<=base+0.01": _le(c["ece"], base["ece"], 0.010),
         "own_state_ts_acc>=base+0.02": _ge(ec["own_state_ts_acc"], eb["own_state_ts_acc"], 0.020),
         "success_live>=base-0.01": _ge(ec["success_live"], eb["success_live"], -0.010)}
    return all(a.values()) or all(b.values()), {"rule_A": a, "rule_A_pass": all(a.values()), "rule_B": b,
                                                "rule_B_pass": all(b.values())}


def grpo_passes(c: dict[str, Any], init: dict[str, Any], ec: dict[str, Any], ei: dict[str, Any],
                validity_ok: bool) -> tuple[bool, dict]:
    r = {"validity_tests_passed": validity_ok,
         "acc>=init-0.01": _ge(c["acc"], init["acc"], -0.010),
         "recovery>=init-0.01": _ge(c["recovery"], init["recovery"], -0.010),
         "ece<=init+0.02": _le(c["ece"], init["ece"], 0.020),
         "reward_live>init": (ec["reward_live"] is not None and ei["reward_live"] is not None
                              and ec["reward_live"] > ei["reward_live"]),
         "success_live>=init-0.01": _ge(ec["success_live"], ei["success_live"], -0.010)}
    return all(r.values()), r


def retain(passing: list[str], dl: dict[str, dict[str, Any]], order: tuple[str, ...]) -> str | None:
    """Highest v1-dev accuracy, then the later step."""
    if not passing:
        return None
    return max(passing, key=lambda t: (dl[t]["acc"], order.index(t)))


def decide(eval_dir: Path, e2dev: Path, validity_ok: bool) -> dict[str, Any]:
    tags = ("sft_v1",) + OPD_CONT + GRPO_CONT
    dl = {t: decision_level(eval_dir, t) for t in tags}
    ed = {t: dev_slice(e2dev, f"e2dev_rl_{t}") for t in tags}
    out: dict[str, Any] = {"stages": {}, "decision_level": dl, "dev_slice": ed}
    kept: list[str] = []
    if dl["sft_v1"]["acc"] is not None:
        kept.append("sft_v1")
    out["stages"]["sft_v1"] = {"kept": bool(kept), "retained": "sft_v1" if kept else None}
    # OPD against SFT v1
    res, passing = {}, []
    for t in OPD_CONT:
        if dl[t]["acc"] is None:
            res[t] = {"pass": False, "why": "not trained / not evaluated"}
            continue
        ok, why = opd_passes(dl[t], dl["sft_v1"], ed[t], ed["sft_v1"]) if kept else (False, {"why": "no SFT v1"})
        res[t] = {"pass": ok, **why}
        if ok:
            passing.append(t)
    r = retain(passing, dl, OPD_CONT)
    out["stages"]["opd"] = {"kept": r is not None, "retained": r, "against": "sft_v1", "continuations": res}
    if r:
        kept.append(r)
    # GRPO against its init (the retained continuation of the last kept stage)
    init = kept[-1] if kept else None
    out["grpo_init"] = init
    res, passing = {}, []
    for t in GRPO_CONT:
        if dl[t]["acc"] is None or init is None:
            res[t] = {"pass": False, "why": "not trained / not evaluated"}
            continue
        ok, why = grpo_passes(dl[t], dl[init], ed[t], ed[init], validity_ok)
        res[t] = {"pass": ok, "rule": why}
        if ok:
            passing.append(t)
    r = retain(passing, dl, GRPO_CONT)
    out["stages"]["grpo"] = {"kept": r is not None, "retained": r, "against": init, "continuations": res}
    if r:
        kept.append(r)
    go = [t for t in kept if dl[t]["go"]]
    frozen = go[-1] if go else (kept[-1] if kept else None)
    out.update({"kept": kept, "frozen": frozen, "go": bool(go), "rl_claim": bool(frozen and frozen.startswith("grpo")),
                "label": ("GO: the frozen JevAgentRL-4B passes G1-G5 (in-domain, generator-adapted vs off-the-shelf)"
                          if go else "NO-GO: Intern-Decision-4B stays the open System-1; JevAgentRL is reported as is")})
    refs = dev_slice(e2dev, "e2dev_rl_refs_intern", "intern")
    fr = ed.get(frozen) if frozen in ed else {}
    out["confirmation_dev_slice"] = {
        "ours": fr, "intern": refs,
        "shadow_ts_acc>=intern": _ge((fr or {}).get("shadow_ts_acc"), refs.get("shadow_ts_acc"), 0.0),
        "ts_replace_success>=intern": _ge((fr or {}).get("ts_replace_success"), refs.get("ts_replace_success"), 0.0),
        "ts_replace_acc>=intern": _ge((fr or {}).get("ts_replace_acc"), refs.get("ts_replace_acc"), 0.0)}
    return out


def validity_passed(text: str) -> bool:
    """Did tests/test_grpo_validity.py pass? An explicit ``pytest_exit=N`` line (gpu_round5_rl.sh records it) decides;
    otherwise the pytest output: a "N passed" summary without failures, or -- with the repository's ``addopts = "-q"``
    plus ``-q`` (= -qq, no summary line) -- progress lines of only passes / skips reaching [100%]."""
    import re

    m = re.findall(r"pytest_exit=(\d+)", text)
    if m:
        return m[-1] == "0"
    low = text.lower()
    if re.search(r"\b\d+ (failed|errors?)\b", low) or "traceback" in low:
        return False
    if re.search(r"\b\d+ passed\b", low):
        return True
    prog = [ln.split("[")[0].strip() for ln in text.splitlines() if re.match(r"^[.sxFE]+\s+\[\s*\d+%\]\s*$", ln.strip())]
    return bool(prog) and text.rstrip().endswith("[100%]") and all(set(p) <= {".", "s"} for p in prog) and \
        any("." in p for p in prog)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-dir", default="runs/gpu_round5/eval")
    ap.add_argument("--e2dev", default="runs/gpu_round5/e2dev")
    ap.add_argument("--grpo-validity", default="runs/gpu_round5/grpo_validity.txt",
                    help="pytest output of tests/test_grpo_validity.py on the training commit ('passed', no failure)")
    ap.add_argument("--out", default="runs/gpu_round5/chain_decision.json")
    a = ap.parse_args(argv)
    vt = Path(a.grpo_validity).read_text() if Path(a.grpo_validity).exists() else ""
    validity_ok = validity_passed(vt)
    res = decide(Path(a.eval_dir), Path(a.e2dev), validity_ok)
    res["inputs"] = {"eval_dir": a.eval_dir, "e2dev": a.e2dev, "grpo_validity": a.grpo_validity,
                     "grpo_validity_ok": validity_ok}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("kept", "frozen", "go", "rl_claim", "label")}, indent=1))


if __name__ == "__main__":
    main()
