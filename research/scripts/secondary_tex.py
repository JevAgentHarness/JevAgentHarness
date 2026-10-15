#!/usr/bin/env python3
"""Stage-1 secondary analyses -> ``\\res{}`` keys for the paper (``paper/generated/secondary.tex``, with a copy in
``results/secondary/secondary.tex``). Reads only committed result files (no run directory, no API):

  results/secondary/h1.json                 scripts/h1_threshold_theory.py   (H1, A6.22 (c) 72-cell rule)
  results/secondary/h3.json                 scripts/h3_delegation_classes.py (H3, A6.22 (c) class credit at 5 %)
  results/secondary/h4.json                 scripts/h4_tost.py               (H4, A6.22 (c))
  results/secondary/tau2_proxy_check.json   scripts/tau2_proxy_check.py      (C7 proxy check, A5.4 (e))
  results/calibration/tauhat_*.json         threshold_theory --calibration-arm (the C1 fits, A6.5 / A6.11; the tau^2
                                            refit of C7, A5.4 (c) / A6.20 (b) / A6.22 (b))

Keys:
  secondary/h1/{verdict, clause1, clause2, n_cells, n_estimable, n_absent, wins, ties, losses, ci_above_0,
      ci_below_0, n_v_w, projection/{wins, ties, losses}, clause2/{n_cells, n_over_confident, n_return,
      over_confident_cells}}
  secondary/h3/{verdict, n_universe, n_negligible, n_counterexamples, counterexample/{credited, n, delta,
      cost_savings, irreversible_share, copy_share}, n_qualifying/{l2_flash, l2_lite, l3_flash}, n_credited/{...},
      tau_b/{flash_vs_lite, l2_vs_l3}/{value, p}, absent}
  secondary/h4/verdict
  secondary/proxy/{n, n_sens, n_qualifying_e2, n_qualifying_tau2}; secondary/proxy/{delta_success, iso_quality_savings,
      delta_success_pass1, net_cost_savings}/{tau_b, p (one-sided, agreement), p_two}
      (``*_sens`` for the sensitivity row with the E2-carried tau-hat* cascade)
  calibration/c1/{jev,intern}/{flash,lite}/{tool_select,tool_params}/tauhat   ("always defer" where the fit accepts no
      level, A6.11 (c).4)
  tau2/refit/{tool_select,tool_params}/{tauhat, n_paired}
Thresholds print with two decimals, or four where the deployed value needs them (0.6309, 0.8518)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "results" / "secondary"
CAL = ROOT / "results" / "calibration"
C1_FITS = {("jev", "flash"): "jev_shadow_cal", ("jev", "lite"): "jev_shadow_cal_s2lite",
           ("intern", "flash"): "intern_shadow_cal", ("intern", "lite"): "intern_shadow_cal_s2lite"}
TAU2_REFIT = "tau2_jev_shadow"
DPS = ("tool_select", "tool_params")


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _bad(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def tau(x: Any) -> str:
    """A deployed threshold: two decimals, four where needed; never -> "always defer" (A6.11 (c).4)."""
    if _bad(x):
        return "--"
    v = float(x)
    if math.isinf(v) or v > 1.0:
        return "always defer"
    return f"{v:.2f}" if abs(round(v, 2) - v) < 1e-9 else f"{v:.4f}"


def num(x: Any, nd: int = 3, signed: bool = False) -> str:
    if _bad(x):
        return "--"
    v = float(x)
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def pct(x: Any, nd: int = 1, signed: bool = False) -> str:
    return "--" if _bad(x) else num(100 * float(x), nd, signed)


def pval(x: Any) -> str:
    if _bad(x):
        return "--"
    v = float(x)
    return "$<$0.001" if v < 0.001 else f"{v:.3f}"


def _load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text()) if p.exists() else None


def keys_h1(h1: Mapping[str, Any]) -> dict[str, str]:
    c1, c2, pj = h1["clause1"], h1["clause2"], h1["projection"]
    k = {"verdict": h1["verdict"], "clause1": c1["label"], "clause2": c2["label"],
         "n_cells": str(c1["n_cells"]), "n_estimable": str(c1["n_estimable"]), "n_absent": str(c1["n_absent"]),
         "wins": str(c1["wins"]), "ties": str(c1["ties"]), "losses": str(c1["losses"]),
         "ci_above_0": str(c1["ci_above_0"]), "ci_below_0": str(c1["ci_below_0"]),
         "n_v_w": str(sum(1 for r in h1["cells"] if str(r.get("v_source", "")).startswith("--V"))),
         "projection/wins": str(pj["wins"]), "projection/ties": str(pj["ties"]), "projection/losses": str(pj["losses"]),
         "clause2/n_cells": str(c2["n_cells"]), "clause2/n_over_confident": str(c2["n_over_confident"]),
         "clause2/n_return": str(c2["n_return"]),
         "clause2/over_confident_cells": "; ".join(_cell_name(x) for x in c2["cells"] if x.get("over_confident")) or "none"}
    return {f"secondary/h1/{a}": b for a, b in k.items()}


def _cell_name(x: Mapping[str, Any]) -> str:
    """A clause-2 cell in words: \\jev{} / Intern-Decision, decision point, System-2 model."""
    b = {"jev": "\\jev{}", "intern": "Intern-Decision"}.get(x["backend"], x["backend"])
    d = {"tool_select": "tool selection", "tool_params": "arguments"}.get(x["dp"], x["dp"])
    m = {"flash": "Flash", "lite": "Flash-Lite"}.get(x["s2"], x["s2"])
    return f"{b} {d} ({m})"


def keys_h3(h3: Mapping[str, Any]) -> dict[str, str]:
    v = h3["verdict"]
    k = {"verdict": v["verdict"], "n_universe": str(h3["n_universe"]), "n_negligible": str(h3["n_negligible"]),
         "n_counterexamples": str(len(v["counterexamples"])), "absent": "ALFWorld (E6 not run)"}
    arms = {r["id"]: r for r in h3["arms"]}
    if v["counterexamples"]:
        r = arms[v["counterexamples"][0]]
        k.update({"counterexample/n": str(r["n"]), "counterexample/delta": pct(r["delta"], 1, True),
                  "counterexample/cost_savings": pct(r["cost_savings"]),
                  "counterexample/irreversible_share": pct(r["shares"]["irreversible_tool"]),
                  "counterexample/copy_share": pct(r["shares"]["copy_arg"]),
                  "counterexample/credited": ", ".join(sorted(r["credited"], key=lambda c: -["meta", "bounded-recoverable",
                                                                                               "copy-dependent",
                                                                                               "irreversible"].index(c)))})
    for g, key in (("L2/flash", "l2_flash"), ("L2/lite", "l2_lite"), ("L3/flash", "l3_flash")):
        cl = h3["ranking"][g]
        k[f"n_qualifying/{key}"] = str(sum(cl[c]["n_qualifying"] for c in cl))
        k[f"n_credited/{key}"] = str(sum(cl[c]["n_credited"] for c in cl))
    for name, key in (("flash_vs_lite_L2", "flash_vs_lite"), ("L2_vs_L3_flash", "l2_vs_l3")):
        t = h3["kendall"][name]
        k[f"tau_b/{key}/value"] = "undefined" if _bad(t["tau_b"]) else num(t["tau_b"], 2, True)
        k[f"tau_b/{key}/p"] = pval(t["p_two_sided"])
    return {f"secondary/h3/{a}": b for a, b in k.items()}


def keys_proxy(px: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {"n": str(len(px["primary"]["configurations"])),
                         "n_sens": str(len(px["sensitivity_with_tauhat_e2"]["configurations"]))}
    prim = [r for r in px["rows"] if r["configuration"] in px["primary"]["configurations"] and r["e2_arm"] != "llm_only"]
    k["n_qualifying_e2"] = str(sum(1 for r in prim if r["e2"].get("qualifies")))
    k["n_qualifying_tau2"] = str(sum(1 for r in prim if r["tau2"].get("qualifies")))
    for blk, suf in (("primary", ""), ("sensitivity_with_tauhat_e2", "_sens")):
        for stat in ("delta_success", "iso_quality_savings", "delta_success_pass1", "net_cost_savings"):
            r = px[blk][stat]
            k[f"{stat}{suf}/tau_b"] = "undefined" if _bad(r["tau_b"]) else num(r["tau_b"], 2, True)
            k[f"{stat}{suf}/p"] = pval(r["p_greater"])
            k[f"{stat}{suf}/p_two"] = pval(r["p_two_sided"])
    return {f"secondary/proxy/{a}": b for a, b in k.items()}


def keys_calibration() -> dict[str, str]:
    out: dict[str, str] = {}
    for (backend, s2), arm in C1_FITS.items():
        for dp in DPS:
            d = _load(CAL / f"tauhat_{arm}_{dp}.json")
            if d is not None:
                out[f"calibration/c1/{backend}/{s2}/{dp}/tauhat"] = tau((d.get("primary") or {}).get("tau_hat"))
    for dp in DPS:
        d = _load(CAL / f"tauhat_{TAU2_REFIT}_{dp}.json")
        if d is not None:
            out[f"tau2/refit/{dp}/tauhat"] = tau((d.get("primary") or {}).get("tau_hat"))
            out[f"tau2/refit/{dp}/n_paired"] = str(d.get("n_paired"))
    return out


def collect(sec: Path = SEC) -> tuple[dict[str, str], list[str]]:
    keys: dict[str, str] = {}
    sources = []
    for name, fn in (("h1.json", keys_h1), ("h3.json", keys_h3), ("tau2_proxy_check.json", keys_proxy)):
        d = _load(sec / name)
        if d is not None:
            keys.update(fn(d))
            sources.append(f"results/secondary/{name} (generated {d.get('generated')}, git {str(d.get('git'))[:12]})")
    h4 = _load(sec / "h4.json")
    if h4 is not None:
        keys["secondary/h4/verdict"] = h4["verdict"]
        sources.append(f"results/secondary/h4.json (generated {h4.get('generated')})")
    keys.update(keys_calibration())
    sources.append("results/calibration/tauhat_{jev,intern}_shadow_cal{,_s2lite}_{tool_select,tool_params}.json, "
                   "tauhat_tau2_jev_shadow_{tool_select,tool_params}.json")
    return keys, sources


def to_tex(keys: Mapping[str, str], sources: list[str]) -> str:
    L = ["% AUTO-GENERATED by scripts/secondary_tex.py -- do not edit by hand.",
         f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git {_git_head()}"]
    L += [f"% source: {s}" for s in sources]
    L += ["\\makeatletter",
          "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
          "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    L += [f"\\defres{{{k}}}{{{keys[k]}}}" for k in sorted(keys)]
    L.append("\\makeatother")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated" / "secondary.tex"))
    ap.add_argument("--copy", default=str(SEC / "secondary.tex"))
    a = ap.parse_args(argv)
    keys, sources = collect()
    tex = to_tex(keys, sources)
    for p in (Path(a.out), Path(a.copy)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tex)
    print(f"{len(keys)} keys -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
