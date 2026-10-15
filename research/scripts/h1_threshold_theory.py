#!/usr/bin/env python3
"""H1 (secondary) exactly as registered: the 72-cell verdict rule of PREREGISTRATION A6.22 (c) over the n = 500
evaluation shadow arms (docs/PREREG_EFFECTIVE.md, "Other registered hypotheses").

    python scripts/h1_threshold_theory.py [--runs runs] [--out results/secondary] [--workers 3]

Cells: {tool_select, tool_params} x {Jev, Intern-Decision-4B} x {Flash, Flash-Lite} x W in {0.1, 1, 10} x
lambda in {0, 0.5, 5} $/h = 72. Each cell runs the REGISTERED command (a subprocess, nothing re-implemented)

    python -m jevharness.bench.threshold_theory RUN --arm <arm> --dp <dp> --train-frac 0.3 --seed 0 --taus h1 \
        --rule general --task-value W --lam-per-hour LAMBDA --format json --out <cell.json>

on `jev_shadow`, `jev_shadow_s2lite` (runs/e2a_jevgym_tool_decisions) and `intern_shadow`, `intern_shadow_s2lite`
(runs/e2a_jevgym_tool_decisions_intern). When the counterfactual V of the training split or of the cell is undefined
or not positive (the module then reports "V is not positive ... tau-hat undefined"), the cell is run again with
`--V W` (A6.11 (c).1) and the cell says so (``v_source``).

Verdict rule (A6.22 (c), H1):
* statistic d = held-out loss of tau = 0.8 minus that of tau-hat* (``regret.default_minus_hat.diff``; tau-hat* =
  ``candidates.tau_hat``, the ``general`` plug-in); win d > 0, tie |d| <= 1e-12 (a non-win), loss d < 0; one vote
  per cell; the deployed projection (``tau_hat_projection``) gets the same count as a secondary line; the
  decision-weighted share is descriptive;
* absent cells (arm missing, no paired decision on a split, tau-hat* undefined) are listed and leave the denominator;
  below 36 estimable cells H1 is *not estimable*;
* clause 1 holds iff wins > half of the estimable cells (point estimates); every cell prints the 95 % paired
  task-bootstrap CI of d (``default_minus_hat.ci``) and the counts of cells whose CI lies above / below 0;
* clause 2 at W = $1, lambda = 0, per (backend, decision point, System-2) cell (``by_V[0].h1_clause2``): over-confident
  iff the training-split gap's one-sided 95 % lower bound > 0; returns to the front iff the raw default regret CI's
  lower end > 0 and the recalibrated one's <= 0; clause 2 holds iff >= half of the over-confident cells return,
  *not applicable* with none;
* supported iff clause 1 holds and clause 2 holds or is not applicable; otherwise falsified (or *not estimable*).
  Binary-score caveat: a monotone recalibration cannot move tau-hat*, the deployed projection or the attainable
  front (rank-based); clause 2 measures the cost of a mis-set fixed threshold, never a frontier change.

Outputs: ``<out>/h1.json`` (every cell, the commands, provenance), ``<out>/h1.md`` and the per-cell module outputs
``<out>/h1_cells/<backend>_<s2>_<dp>_W<W>_lam<lambda>.json.gz`` (gzip of the module's ``--out`` file). ``scripts/secondary_tex.py`` turns h1.json into
``\\res{secondary/h1/...}`` keys. Analysis of existing logs only (no API call, no episode)."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

REGISTERED = "PREREGISTRATION §3 H1 as replaced by A6.22 (c); A6.5, A6.11 (c).1, A6.16 (h), A6.19 (d)"
BACKENDS = {  # backend -> (run directory name, {System-2: shadow arm})
    "jev": ("e2a_jevgym_tool_decisions", {"flash": "jev_shadow", "lite": "jev_shadow_s2lite"}),
    "intern": ("e2a_jevgym_tool_decisions_intern", {"flash": "intern_shadow", "lite": "intern_shadow_s2lite"}),
}
BACKEND_NAMES = {"jev": "Jev", "intern": "Intern-Decision-4B"}
S2_NAMES = {"flash": "gemini-3.5-flash", "lite": "gemini-3.5-flash-lite"}
DPS = ("tool_select", "tool_params")
WS = (0.1, 1.0, 10.0)
LAMS = (0.0, 0.5, 5.0)
TIE = 1e-12
MIN_ESTIMABLE = 36
CLAUSE2_W, CLAUSE2_LAM = 1.0, 0.0
REGISTERED_ARGS = ["--train-frac", "0.3", "--seed", "0", "--taus", "h1", "--rule", "general"]


def _num(x: float) -> str:
    """CLI form of a grid value: 0.1, 1, 10, 0, 0.5, 5."""
    return f"{x:g}"


def cell_id(backend: str, s2: str, dp: str, W: float, lam: float) -> str:
    return f"{backend}_{s2}_{dp}_W{_num(W)}_lam{_num(lam)}"


def cells() -> list[dict[str, Any]]:
    """The 72 registered cells, in a fixed order."""
    out = []
    for backend, (run, arms) in BACKENDS.items():
        for s2, arm in arms.items():
            for dp in DPS:
                for W in WS:
                    for lam in LAMS:
                        out.append({"id": cell_id(backend, s2, dp, W, lam), "backend": backend, "s2": s2, "dp": dp,
                                    "W": W, "lam": lam, "run": run, "arm": arm})
    return out


def command(c: Mapping[str, Any], runs: Path, out_file: Path, v_w: bool = False) -> list[str]:
    """The registered threshold_theory command of one cell (``v_w``: with ``--V W``, A6.11 (c).1)."""
    cmd = [sys.executable, "-m", "jevharness.bench.threshold_theory", str(runs / c["run"]), "--arm", c["arm"],
           "--dp", c["dp"], *REGISTERED_ARGS, "--task-value", _num(c["W"]), "--lam-per-hour", _num(c["lam"])]
    if v_w:
        cmd += ["--V", _num(c["W"])]
    return cmd + ["--format", "json", "--out", str(out_file)]


def _display(cmd: Sequence[str]) -> str:
    """The command as a person would type it (``python -m ...``, repository-relative paths)."""
    parts = ["python" if i == 0 else str(x) for i, x in enumerate(cmd)]
    return " ".join(p.replace(str(ROOT) + "/", "") for p in parts)


def v_undefined(res: Mapping[str, Any]) -> bool:
    """True when the registered run skipped the cell because the counterfactual V of the training split or of the
    cell is undefined or not positive (``threshold_theory.analyze`` then writes a ``note`` and no ``regret``)."""
    bv = res.get("by_V") or []
    if not bv:
        return True
    c = bv[0]
    V, Vp = c.get("V"), c.get("V_plugin")
    bad = any(v is None or (isinstance(v, float) and math.isnan(v)) or not v > 0 for v in (V, Vp))
    return bool(bad or "note" in c and "regret" not in c)


def run_cell(c: Mapping[str, Any], runs: Path, cell_dir: Path) -> dict[str, Any]:
    """Run the registered command of one cell; re-run with ``--V W`` when V is undefined / not positive."""
    out_file = cell_dir / f"{c['id']}.json"
    rec: dict[str, Any] = dict(c)
    arm_file = runs / c["run"] / f"{c['arm']}.jsonl"
    if not arm_file.exists():
        rec.update({"absent": "arm missing", "command": None})
        return rec
    cmd = command(c, runs, out_file)
    rec["command"] = _display(cmd)
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    res = json.loads(out_file.read_text())
    rec["v_source"] = "counterfactual"
    if v_undefined(res):
        first = (res.get("by_V") or [{}])[0]
        rec["v_registered_run"] = {"V": first.get("V"), "V_plugin": first.get("V_plugin"), "note": first.get("note")}
        cmd = command(c, runs, out_file, v_w=True)
        rec["command"] = _display(cmd)
        rec["v_source"] = "--V W (A6.11 (c).1): the counterfactual V of the training split or of the cell is undefined or not positive"
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        res = json.loads(out_file.read_text())
    gz = out_file.with_suffix(".json.gz")                 # the module output, kept compressed (provenance)
    gz.write_bytes(gzip.compress(out_file.read_bytes(), mtime=0))
    out_file.unlink()
    rec["out_file"] = str(gz.relative_to(ROOT)) if gz.is_relative_to(ROOT) else str(gz)
    rec.update(summarise(res))
    return rec


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def summarise(res: Mapping[str, Any]) -> dict[str, Any]:
    """The registered quantities of one cell from the module's JSON output."""
    out: dict[str, Any] = {"n_paired": res.get("n_paired"), "n_records": res.get("n_records")}
    bv = (res.get("by_V") or [{}])[0]
    out["V"], out["V_plugin"] = _f(bv.get("V")), _f(bv.get("V_plugin"))
    reg = bv.get("regret") or {}
    out["n_train"], out["n_test"] = reg.get("n_train"), reg.get("n_test")
    cand = reg.get("candidates") or {}
    th = _f((cand.get("tau_hat") or {}).get("tau"))
    out["tau_hat"] = th
    out["tau_best_test"] = _f(reg.get("tau_best_test"))
    if not reg or not reg.get("n_train") or not reg.get("n_test"):
        out["absent"] = "no paired decision on a split"
        return out
    if "default_minus_hat" not in reg or math.isnan(th):
        out["absent"] = "tau-hat* undefined"
        return out
    dmh = reg["default_minus_hat"]
    d = _f(dmh.get("diff"))
    out.update({"d": d, "d_ci": [_f(x) for x in dmh.get("ci", [math.nan, math.nan])],
                "outcome": outcome(d),
                "loss_default": _f(cand["tau_default"]["loss"]), "loss_hat": _f(cand["tau_hat"]["loss"]),
                "regret_hat": _f(cand["tau_hat"]["regret"]), "regret_default": _f(cand["tau_default"]["regret"])})
    proj = cand.get("tau_hat_projection")
    if proj:
        dp_ = _f(cand["tau_default"]["loss"]) - _f(proj["loss"])
        out.update({"tau_hat_projection": _f(proj["tau"]), "d_projection": dp_, "outcome_projection": outcome(dp_)})
    c2 = bv.get("h1_clause2")
    if c2:
        oc = c2.get("overconfidence") or {}
        out["clause2"] = {"over_confident": c2.get("over_confident"), "gap": _f(oc.get("gap")), "gap_lb": _f(oc.get("lb")),
                          "n": oc.get("n"), "n_tasks": oc.get("n_tasks"),
                          "raw_regret": _f((c2.get("raw") or {}).get("regret")),
                          "raw_ci": (c2.get("raw") or {}).get("regret_ci"),
                          "recal_regret": _f((c2.get("recalibrated") or {}).get("regret")),
                          "recal_ci": (c2.get("recalibrated") or {}).get("regret_ci"),
                          "t1_ci": (c2.get("dist_t1") or {}).get("regret_ci"),
                          "temperatures": c2.get("temperatures"), "returns_to_front": c2.get("returns_to_front")}
    return out


def outcome(d: float) -> str:
    if math.isnan(d):
        return "absent"
    if abs(d) <= TIE:
        return "tie"
    return "win" if d > 0 else "loss"


def verdict(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Clause 1, the projection line, clause 2 and the verdict of A6.22 (c) from the cell records."""
    est = [r for r in recs if not r.get("absent")]
    absent = [{"id": r["id"], "reason": r["absent"]} for r in recs if r.get("absent")]
    wins = sum(r["outcome"] == "win" for r in est)
    ties = sum(r["outcome"] == "tie" for r in est)
    losses = sum(r["outcome"] == "loss" for r in est)
    above = sum(r["d_ci"][0] > 0 for r in est)
    below = sum(r["d_ci"][1] < 0 for r in est)
    n = len(est)
    nt = sum(int(r.get("n_test") or 0) for r in est)
    share_w = (sum(int(r.get("n_test") or 0) for r in est if r["outcome"] == "win") / nt) if nt else math.nan
    c1: dict[str, Any] = {"n_cells": len(recs), "n_estimable": n, "n_absent": len(absent), "absent": absent,
                          "wins": wins, "ties": ties, "losses": losses, "ci_above_0": above, "ci_below_0": below,
                          "decision_weighted_win_share": share_w, "min_estimable": MIN_ESTIMABLE}
    if n < MIN_ESTIMABLE:
        c1["holds"] = None
        c1["label"] = "not estimable"
    else:
        c1["holds"] = bool(wins > n / 2)
        c1["label"] = "holds" if c1["holds"] else "does not hold"
    pj = [r for r in est if r.get("outcome_projection")]
    proj = {"n": len(pj), "wins": sum(r["outcome_projection"] == "win" for r in pj),
            "ties": sum(r["outcome_projection"] == "tie" for r in pj),
            "losses": sum(r["outcome_projection"] == "loss" for r in pj),
            "note": "secondary line (A6.22 (c)): the deployed projection counted like tau-hat*"}
    c2_cells = [r for r in recs if r["W"] == CLAUSE2_W and r["lam"] == CLAUSE2_LAM]
    c2rows = []
    for r in c2_cells:
        c2 = r.get("clause2") or {}
        c2rows.append({"id": r["id"], "backend": r["backend"], "s2": r["s2"], "dp": r["dp"],
                       "over_confident": c2.get("over_confident"), "gap": c2.get("gap"), "gap_lb": c2.get("gap_lb"),
                       "raw_ci": c2.get("raw_ci"), "recal_ci": c2.get("recal_ci"), "t1_ci": c2.get("t1_ci"),
                       "returns_to_front": c2.get("returns_to_front"), "evaluated": bool(c2)})
    oc = [x for x in c2rows if x["over_confident"]]
    ret = [x for x in oc if x["returns_to_front"]]
    if not all(x["evaluated"] for x in c2rows):
        c2v = {"label": "not evaluated", "holds": None}
    elif not oc:
        c2v = {"label": "not applicable", "holds": None}
    else:
        h = len(ret) >= len(oc) / 2
        c2v = {"label": "holds" if h else "does not hold", "holds": bool(h)}
    clause2 = {"W": CLAUSE2_W, "lam": CLAUSE2_LAM, "cells": c2rows, "n_cells": len(c2rows),
               "n_over_confident": len(oc), "n_return": len(ret), **c2v}
    if c1["holds"] is None:
        v = "not estimable"
    elif c1["holds"] and (clause2["label"] in ("holds", "not applicable")):
        v = "supported"
    else:
        v = "falsified"
    return {"clause1": c1, "projection": proj, "clause2": clause2, "verdict": v}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    x = _f(x)
    if math.isnan(x):
        return "–"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:.{nd}g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"


def to_markdown(res: Mapping[str, Any]) -> str:
    v, c1, c2, pj = res["verdict"], res["clause1"], res["clause2"], res["projection"]
    L = ["# H1 — plug-in thresholds vs. the fixed default (secondary; A6.22 (c) verdict rule)", "",
         f"- generated {res['generated']} at git `{res['git']}` by `scripts/h1_threshold_theory.py`",
         f"- registered: {res['registered']}",
         f"- **verdict: {v}** (clause 1: {c1['label']}; clause 2: {c2['label']})",
         f"- clause 1: {c1['wins']} wins, {c1['ties']} ties, {c1['losses']} losses over {c1['n_estimable']} estimable "
         f"cells of {c1['n_cells']} ({c1['n_absent']} absent; below {MIN_ESTIMABLE} estimable: not estimable); "
         f"wins > half of the estimable cells: {_fmt(c1['holds'])}",
         f"- cells whose 95 % paired task-bootstrap CI of d lies above 0: {c1['ci_above_0']}; below 0: {c1['ci_below_0']}; "
         f"decision-weighted win share {_fmt(c1['decision_weighted_win_share'], 3)} (descriptive)",
         f"- secondary line, the deployed projection (`tau_hat_projection`) vs. τ = 0.8: {pj['wins']} wins, {pj['ties']} "
         f"ties, {pj['losses']} losses over {pj['n']} cells",
         f"- clause 2 (W = $1, λ = 0): {c2['n_over_confident']} of {c2['n_cells']} (backend, decision point, System-2) "
         f"cells over-confident, {c2['n_return']} return to the front → {c2['label']}",
         "- binary-score caveat (A6.22 (c)): a monotone recalibration cannot move τ̂\\*, the deployed projection or the "
         "attainable front (rank-based); clause 2 measures the cost of a mis-set fixed threshold, never a frontier change",
         f"- V: {sum(1 for r in res['cells'] if r.get('v_source', '').startswith('--V'))} of {len(res['cells'])} cells "
         "ran with `--V W` because the counterfactual V of the training split or of the cell is undefined or not positive "
         "(A6.11 (c).1); the others use the counterfactual V", ""]
    if c1["absent"]:
        L += ["Absent cells (out of the denominator): " + ", ".join(f"`{a['id']}` ({a['reason']})" for a in c1["absent"]), ""]
    L += ["## Cells (d = held-out loss of τ = 0.8 minus that of τ̂\\*, USD per decision)", "",
          "| backend | System-2 | dp | W | λ $/h | V | V source | τ̂\\* | d | 95 % CI | vote | τ̂\\* proj. | d proj. | n train / test |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["cells"]:
        if r.get("absent"):
            L.append(f"| {BACKEND_NAMES[r['backend']]} | {r['s2']} | {r['dp']} | {_num(r['W'])} | {_num(r['lam'])} | "
                     f"– | – | – | – | – | absent: {r['absent']} | – | – | – |")
            continue
        vs = "W" if r["v_source"].startswith("--V") else "cf"
        L.append(f"| {BACKEND_NAMES[r['backend']]} | {r['s2']} | {r['dp']} | {_num(r['W'])} | {_num(r['lam'])} | "
                 f"{_fmt(r['V'], 3)} | {vs} | {_fmt(r['tau_hat'], 3)} | {_fmt(r['d'])} | "
                 f"[{_fmt(r['d_ci'][0])}, {_fmt(r['d_ci'][1])}] | {r['outcome']} | {_fmt(r.get('tau_hat_projection'), 3)} | "
                 f"{_fmt(r.get('d_projection'))} | {r.get('n_train')} / {r.get('n_test')} |")
    L += ["", "## Clause 2 (W = $1, λ = 0)", "",
          "| backend | System-2 | dp | gap (conf − acc) | one-sided 95 % LB | over-confident | default regret CI, raw | recalibrated | T = 1 control | returns |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in c2["cells"]:
        def ci(c: Any) -> str:
            return f"[{_fmt(c[0])}, {_fmt(c[1])}]" if c else "–"
        L.append(f"| {BACKEND_NAMES[x['backend']]} | {x['s2']} | {x['dp']} | {_fmt(x['gap'], 3)} | {_fmt(x['gap_lb'], 3)} | "
                 f"{_fmt(x['over_confident'])} | {ci(x['raw_ci'])} | {ci(x['recal_ci'])} | {ci(x['t1_ci'])} | "
                 f"{_fmt(x['returns_to_front'])} |")
    L += ["", "## Registered commands", "", "Each cell ran the command below (the `--V W` form only where noted).", ""]
    L += [f"- `{r['command']}`" for r in res["cells"] if r.get("command")]
    L += ["", "## Inputs", ""]
    L += [f"- `{k}`: sha256 `{v['sha256']}`, {v['bytes']} bytes" for k, v in res["provenance"].items()]
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "secondary"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    runs, out = Path(a.runs).resolve(), Path(a.out).resolve()
    cell_dir = out / "h1_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cs = cells()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        recs = list(ex.map(lambda c: run_cell(c, runs, cell_dir), cs))
    prov = {}
    for backend, (run, arms) in BACKENDS.items():
        for arm in arms.values():
            p = runs / run / f"{arm}.jsonl"
            if p.exists():
                prov[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = {"sha256": _sha256(p),
                                                                                       "bytes": p.stat().st_size}
    res = {"registered": REGISTERED, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "git": _git_head(), "grid": {"dp": list(DPS), "backends": BACKEND_NAMES, "s2": S2_NAMES, "W": list(WS),
                                        "lam_per_hour": list(LAMS), "taus": "h1 (0.5 0.7 0.8 0.9 0.95 0.99)",
                                        "default_tau": 0.8, "train_frac": 0.3, "seed": 0, "rule": "general"},
           "cells": recs, **verdict(recs), "provenance": prov}
    (out / "h1.json").write_text(json.dumps(res, indent=1, default=str, allow_nan=True))
    (out / "h1.md").write_text(to_markdown(res))
    print(f"H1: {res['verdict']} (clause 1 {res['clause1']['label']}: {res['clause1']['wins']} wins / "
          f"{res['clause1']['n_estimable']} estimable; clause 2 {res['clause2']['label']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
