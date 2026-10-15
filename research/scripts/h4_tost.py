#!/usr/bin/env python3
"""H4 (secondary, training) exactly as registered (PREREGISTRATION §3 H4 as replaced by A6.22 (c)): R4
(KL-to-Monte-Carlo targets) vs. R3 (teacher distillation) at matched decision accuracy, downstream on JevGym.

    python scripts/h4_tost.py [--r3-dev-acc ACC] [--r4-dev-acc STEP=ACC ...] [--r3-run FILE] [--r4-run FILE] \
        [--out results/secondary]

Rule (A6.22 (c), H4):
* matching on the v1 development set (``training.eval_decisions`` on ``data/train_v1/dev.jsonl.gz``): matched iff
  |acc(R4) - acc(R3)| <= 1 pt; the R4 checkpoint used is the matched one closest to R3 (ties -> the later step); with
  none within 1 pt, H4 is *not tested at matched accuracy* and the comparison is descriptive, with the gap;
* downstream: R3 and the matched R4 (served by ``training/serve_jev.py``) as System-1 of ``jev_ts_params_replace``
  (Flash) on the first 200 frozen evaluation tasks; Delta = success(R4) - success(R3), paired task bootstrap
  (B = 10,000): *supported* iff the one-sided 95 % lower bound > 0; *falsified* iff TOST shows equivalence within
  +-3 pt (the 90 % CI inside (-3, +3) pt, ``metrics.tost_paired(..., 0.03)``); otherwise *inconclusive at this n*.
  "Falsified inside +-3 points" never means a point estimate inside the band.

Inputs: the development accuracies (``--r3-dev-acc`` / ``--r4-dev-acc``, or ``results/training/h4_dev_acc.json``
``{"r3": acc, "r4": {step: acc}}``) and the two downstream arm logs (``--r3-run`` / ``--r4-run``: JSONL episodes of
the replace configuration). Stage 1 closed without an R4 checkpoint or the downstream runs: with no R4 checkpoint none
is within 1 pt, so the registered label is *not tested at matched accuracy* (no gap exists to report). Analysis of
existing files only. Outputs: ``<out>/h4.{json,md}``."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import metrics as M  # noqa: E402

REGISTERED = "PREREGISTRATION §3 H4 as replaced by A6.22 (c)"
MATCH_PT = 0.01
TOST_MARGIN = 0.03
N_BOOT = 10000
N_TASKS = 200
DEV_FILE = ROOT / "results" / "training" / "h4_dev_acc.json"


def match(r3_acc: float | None, r4_accs: Mapping[Any, float]) -> dict[str, Any]:
    """The matched R4 checkpoint: |acc(R4) - acc(R3)| <= 1 pt, the closest to R3, ties -> the later step."""
    if r3_acc is None or not r4_accs:
        return {"matched": False, "step": None, "gap": None,
                "reason": "no R4 checkpoint" if r3_acc is not None else "no R3 development accuracy"}
    gaps = {s: float(a) - float(r3_acc) for s, a in r4_accs.items()}
    best = min(gaps, key=lambda s: (abs(gaps[s]), -float(s)))
    ok = abs(gaps[best]) <= MATCH_PT + 1e-12
    return {"matched": ok, "step": best, "gap": gaps[best], "acc_r3": float(r3_acc), "acc_r4": float(r4_accs[best]),
            "reason": None if ok else f"closest R4 checkpoint {abs(gaps[best]) * 100:.2f} pt from R3 (> 1 pt)"}


def decide(r3: Mapping[str, float], r4: Mapping[str, float], seed: int = 0) -> dict[str, Any]:
    """Delta = success(R4) - success(R3) on the paired tasks: supported / falsified / inconclusive at this n."""
    b95 = M.paired_bootstrap_diff(r3, r4, n=N_BOOT, alpha=0.05, seed=seed)  # lcb = the 5 % quantile (one-sided 95 %)
    tost = M.tost_paired(r3, r4, TOST_MARGIN, alpha=0.05, n_boot=N_BOOT, seed=seed)
    if b95["lcb"] > 0:
        label = "supported"
    elif tost["equivalent"]:
        label = "falsified"
    else:
        label = "inconclusive at this n"
    return {"n": b95["n"], "delta": b95["diff"], "lb95_one_sided": b95["lcb"], "ci95": [b95["ci_lo"], b95["ci_hi"]],
            "ci90": tost["ci90"], "tost": tost, "label": label}


def _success(path: Path, n: int = N_TASKS) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["task_id"])] = float(bool((r.get("result") or {}).get("success")))
    keys = list(out)[:n]
    return {k: out[k] for k in keys}


def run(r3_acc: float | None, r4_accs: Mapping[Any, float], r3_run: Path | None, r4_run: Path | None) -> dict[str, Any]:
    m = match(r3_acc, r4_accs)
    res: dict[str, Any] = {"registered": REGISTERED, "matching": m}
    if not m["matched"]:
        res["verdict"] = "not tested at matched accuracy"
        res["note"] = ("A6.22 (c): with no R4 checkpoint within 1 pt of R3, H4 is not tested at matched accuracy and "
                       "the comparison is descriptive, with the gap" +
                       ("; Stage 1 closed without an R4 checkpoint or the downstream runs, so no gap and no comparison "
                        "exist" if not r4_accs else ""))
        if r3_run and r4_run and r3_run.exists() and r4_run.exists():
            res["descriptive"] = decide(_success(r3_run), _success(r4_run))
        return res
    if not (r3_run and r4_run and r3_run.exists() and r4_run.exists()):
        res["verdict"] = "not tested at matched accuracy"
        res["note"] = "matched checkpoints exist but the downstream runs do not"
        return res
    res["downstream"] = decide(_success(r3_run), _success(r4_run))
    res["verdict"] = res["downstream"]["label"]
    return res


def to_markdown(res: Mapping[str, Any]) -> str:
    m = res["matching"]
    L = ["# H4 — R4 vs. R3 at matched decision accuracy (secondary; A6.22 (c))", "",
         f"- generated {res['generated']} by `scripts/h4_tost.py`; registered: {res['registered']}",
         f"- **verdict: {res['verdict']}**",
         f"- matching: {'matched' if m['matched'] else 'not matched'}" + (f" ({m['reason']})" if m.get("reason") else "")]
    if res.get("note"):
        L.append(f"- {res['note']}")
    d = res.get("downstream") or res.get("descriptive")
    if d:
        L.append(f"- Δ = {100 * d['delta']:+.1f} pt (n = {d['n']}), one-sided 95 % LB {100 * d['lb95_one_sided']:+.1f}, "
                 f"90 % CI [{100 * d['tost']['ci90'][0]:+.1f}, {100 * d['tost']['ci90'][1]:+.1f}] (TOST ±3 pt: "
                 f"{'equivalent' if d['tost']['equivalent'] else 'not shown'})")
    L += [f"- inputs: {res['inputs']}"]
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--r3-dev-acc", type=float, default=None)
    ap.add_argument("--r4-dev-acc", nargs="*", default=[], help="STEP=ACC per saved R4 checkpoint")
    ap.add_argument("--r3-run", default=None)
    ap.add_argument("--r4-run", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "secondary"))
    a = ap.parse_args(argv)
    r3_acc, r4 = a.r3_dev_acc, {s.split("=")[0]: float(s.split("=")[1]) for s in a.r4_dev_acc}
    if r3_acc is None and not r4 and DEV_FILE.exists():
        d = json.loads(DEV_FILE.read_text())
        r3_acc, r4 = d.get("r3"), dict(d.get("r4") or {})
    r3_run, r4_run = (Path(a.r3_run) if a.r3_run else None), (Path(a.r4_run) if a.r4_run else None)
    res = run(r3_acc, r4, r3_run, r4_run)
    res = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **res,
           "inputs": {"r3_dev_acc": r3_acc, "r4_dev_acc": r4, "dev_file": str(DEV_FILE.relative_to(ROOT)),
                      "dev_file_exists": DEV_FILE.exists(), "r3_run": a.r3_run, "r4_run": a.r4_run}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "h4.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "h4.md").write_text(to_markdown(res))
    print(f"H4: {res['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
