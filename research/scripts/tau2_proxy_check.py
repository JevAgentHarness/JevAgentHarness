#!/usr/bin/env python3
"""The C7 proxy check of PREREGISTRATION A5.4 (e) (an A6.20 (f) code item): Kendall's tau between the E2 (JevGym,
Flash, Jev) and the tau^2-airline rankings of every configuration that ran on both, ranked by paired Delta-success
vs. LLM-only and, separately, by iso-quality cost savings; reported with n and a permutation p.

    python scripts/tau2_proxy_check.py [--runs runs] [--out results/secondary]

Analysis of existing logs only (no API call, no episode). Inputs: ``runs/e2a_jevgym_tool_decisions`` (E2a) and
``runs/tau2/airline`` restricted to the 35 airline TEST tasks of the committed A5.4 (c) split
(``configs/tau2/airline_task_ids.json``; the 15 shadow tasks never enter, as in every C7 comparison), 3 trials each.

Configurations (A5.4 (e)'s list; E2a arm -> tau^2 airline arm):
  LLM-only                       llm_only               -> llm_only               (the reference: Delta 0, savings 0)
  CASCADE at each swept tau      jev_ts_cascade_tau<t>  -> jev_ts_cascade_tau<t>  t in {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}
  ASSIST k = 3                   jev_ts_assist_k3       -> jev_ts_assist_k3
  policy GUARD                   jev_ts_guard           -> jev_guard              (interpretation recorded in
                                                                                  results/primaries/c7_selection.json)
Stated difference (PREREGISTRATION Deviations 2026-10-03, tau^2 runs, item (7)): the airline sweep rows were run with
``scripts/run_tau2.sh`` as registered in MAIN_RUN_PLAN step 9, which sets tool_select tau = tool_params tau, whereas
the E2a sweep arms fix the argument threshold at 0.90 (A6.16 (h)); the E2 and tau^2 sweep configurations therefore
share their argument threshold only at tau = 0.9. Nothing is re-run; the check is reported with that difference.

Readings where A5.4 (e) leaves a choice (stated in the output):
* the configuration set is A5.4 (e)'s parenthetical list (the airline proxy-check arms of MAIN_RUN_PLAN step 9);
  the cascade at the E2-carried tau-hat* (E2a ``jev_ts_cascade_tauhat`` at 1.00 / 0.99 = tau^2
  ``jev_ts_cascade_tauhat_e2``) also ran on both and is added in a sensitivity row (n = 10);
* paired Delta-success: on E2 the per-task binary success (the arm's registered n; the reference on the same tasks);
  on tau^2 the registered tau^2 endpoint, majority pass (>= 2 of 3 trials; A6.14, A6.16 (e)), with the pass^1
  ranking as a descriptive row;
* iso-quality cost savings: a configuration has an iso-quality saving iff it passes the ``noninferior`` qualifier at
  the unadjusted level (A6.18 (c): ``metrics.ni_pvalue(...)["ni_95"]``, 3 pt on JevGym, 5 pt on tau^2 majority pass);
  qualifying configurations rank by their paired net cost savings 1 - C_arm / C_ref, every non-qualifying one ties
  below them (LLM-only qualifies with savings 0); the ranking by raw net savings is a descriptive row;
* Kendall's tau is tau-b (ties); the permutation p is exact over all n! assignments of the tau^2 values to the
  configurations, one-sided for agreement (tau > 0: the E2 order predicts the tau^2 order), with the two-sided p
  next to it; tau-b is undefined when either ranking is constant.

Outputs: ``<out>/tau2_proxy_check.{json,md}``; ``scripts/secondary_tex.py`` emits ``\\res{secondary/proxy/...}``."""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

REGISTERED = "PREREGISTRATION A5.4 (e); A6.20 (f) (code item); Deviations 2026-10-03 (tau^2 runs) item (7)"
SWEEP = ("0.5", "0.7", "0.8", "0.9", "0.95", "0.99")
# (configuration, E2a arm, tau^2 airline arm, tool_params tau on E2a, tool_params tau on tau^2)
CONFIGS: tuple[tuple[str, str, str, str, str], ...] = (
    ("LLM-only", "llm_only", "llm_only", "-", "-"),
    *((f"CASCADE tau={t}", f"jev_ts_cascade_tau{t}", f"jev_ts_cascade_tau{t}", "0.9", t) for t in SWEEP),
    ("ASSIST k=3", "jev_ts_assist_k3", "jev_ts_assist_k3", "-", "-"),
    ("policy GUARD", "jev_ts_guard", "jev_guard", "-", "-"),
)
EXTRA: tuple[tuple[str, str, str, str, str], ...] = (
    ("CASCADE tau-hat* (E2-carried, 1.00 / 0.99)", "jev_ts_cascade_tauhat", "jev_ts_cascade_tauhat_e2", "0.99", "0.99"),
)
E2_MARGIN, TAU2_MARGIN = 0.03, 0.05          # JevGym NI margin; C7's tau^2 margin (majority pass)
TAU2_TRIALS = 3
ITER_LIMIT = 10                               # exact enumeration up to 10! permutations


# ====================================================================== Kendall tau-b and its exact permutation p
def _sgn_pairs(v: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    return np.sign(v[..., I] - v[..., J])


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b (Kendall 1945): (C - D) / sqrt((n0 - n1)(n0 - n2)) with n1, n2 the tied pairs of x, y; NaN when
    either variable is constant (or n < 2)."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    n = len(xa)
    if n < 2 or len(ya) != n:
        return math.nan
    I, J = np.triu_indices(n, 1)
    sx, sy = _sgn_pairs(xa, I, J), _sgn_pairs(ya, I, J)
    n0 = len(I)
    n1, n2 = int(np.sum(sx == 0)), int(np.sum(sy == 0))
    den = math.sqrt((n0 - n1) * (n0 - n2))
    return float(np.sum(sx * sy) / den) if den > 0 else math.nan


def kendall_permutation(x: Sequence[float], y: Sequence[float], limit: int = ITER_LIMIT, n_mc: int = 200000,
                        seed: int = 0) -> dict[str, Any]:
    """tau-b and its permutation distribution under H0 (no association): y's values permuted over the configurations
    (x fixed), exactly over all n! assignments when n <= ``limit``, else ``n_mc`` seeded draws. ``p_greater`` =
    P(tau_perm >= tau_obs) (one-sided, agreement), ``p_less`` = P(tau_perm <= tau_obs), ``p_two_sided`` =
    P(|tau_perm| >= |tau_obs|); 1e-12 tolerance for ties of the statistic."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    n = len(xa)
    tau = kendall_tau_b(xa, ya)
    out: dict[str, Any] = {"n": n, "tau_b": tau}
    if math.isnan(tau):
        out.update({"p_greater": math.nan, "p_less": math.nan, "p_two_sided": math.nan, "method": None,
                    "note": "tau-b undefined: a ranking is constant"})
        return out
    I, J = np.triu_indices(n, 1)
    sx = _sgn_pairs(xa, I, J)
    n0 = len(I)
    den = math.sqrt((n0 - int(np.sum(sx == 0))) * (n0 - int(np.sum(_sgn_pairs(ya, I, J) == 0))))
    if n <= limit:
        method, total = "exact", math.factorial(n)
        ge = le = two = 0
        chunk = 200000
        it = itertools.permutations(range(n))
        while True:
            block = list(itertools.islice(it, chunk))
            if not block:
                break
            P = np.asarray(block, dtype=np.int8)
            t = (_sgn_pairs(ya[P], I, J) * sx).sum(axis=1) / den
            ge += int(np.sum(t >= tau - 1e-12))
            le += int(np.sum(t <= tau + 1e-12))
            two += int(np.sum(np.abs(t) >= abs(tau) - 1e-12))
    else:
        method, total = f"monte carlo ({n_mc} draws, seed {seed})", n_mc
        rng = np.random.default_rng(seed)
        P = np.argsort(rng.random((n_mc, n)), axis=1)
        t = (_sgn_pairs(ya[P], I, J) * sx).sum(axis=1) / den
        ge, le = int(np.sum(t >= tau - 1e-12)), int(np.sum(t <= tau + 1e-12))
        two = int(np.sum(np.abs(t) >= abs(tau) - 1e-12))
    out.update({"p_greater": ge / total, "p_less": le / total, "p_two_sided": two / total, "method": method,
                "n_permutations": total})
    return out


# ====================================================================== ranking values
def iso_quality_value(qualifies: bool, savings: float, floor: float) -> float:
    """The iso-quality ranking value of one configuration: its paired net cost savings when it passes the
    ``noninferior`` qualifier, else ``floor`` (below every qualifying value: non-qualifying configurations tie at the
    bottom)."""
    return float(savings) if qualifies and not math.isnan(savings) else floor


def ranking_vectors(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    """(E2 values, tau^2 values) per ranking statistic over the configuration rows (each row: ``e2`` and ``tau2``
    blocks with ``delta``, ``qualifies``, ``cost_savings`` and, on tau^2, ``delta_pass1``)."""
    sav = [r[s]["cost_savings"] for r in rows for s in ("e2", "tau2") if not math.isnan(r[s]["cost_savings"])]
    floor = (min(sav) if sav else 0.0) - 1.0
    return {
        "delta_success": ([r["e2"]["delta"] for r in rows], [r["tau2"]["delta"] for r in rows]),
        "iso_quality_savings": ([iso_quality_value(r["e2"]["qualifies"], r["e2"]["cost_savings"], floor) for r in rows],
                                [iso_quality_value(r["tau2"]["qualifies"], r["tau2"]["cost_savings"], floor) for r in rows]),
        "delta_success_pass1": ([r["e2"]["delta"] for r in rows], [r["tau2"]["delta_pass1"] for r in rows]),
        "net_cost_savings": ([r["e2"]["cost_savings"] for r in rows], [r["tau2"]["cost_savings"] for r in rows]),
    }


RANKINGS = {  # statistic -> (role, description)
    "delta_success": ("registered", "paired Delta-success vs. LLM-only (E2: per-task success; tau^2: majority pass)"),
    "iso_quality_savings": ("registered", "iso-quality cost savings (qualifying configurations by net savings, the rest tied below)"),
    "delta_success_pass1": ("descriptive", "paired Delta-success with tau^2 pass^1 (mean over the 3 trials)"),
    "net_cost_savings": ("descriptive", "paired net cost savings without the qualifier"),
}


def proxy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Kendall tau-b with its exact permutation p for every ranking statistic over ``rows``."""
    out = {}
    for k, (x, y) in ranking_vectors(rows).items():
        role, desc = RANKINGS[k]
        out[k] = {"role": role, "statistic": desc, **kendall_permutation(x, y), "e2": x, "tau2": y}
    return out


# ====================================================================== data
def _blank(margin: float) -> dict[str, Any]:
    return {"n": None, "delta": 0.0, "qualifies": True, "cost_savings": 0.0, "p_decisive": math.nan, "margin": margin,
            "note": "the reference"}


def e2_rows(runs: Path, configs: Sequence[tuple[str, str, str, str, str]], n_boot: int, seed: int,
            flags: Any) -> dict[str, dict[str, Any]]:
    """E2a (Flash, Jev): each configuration vs. ``llm_only`` on the arm's registered tasks (``primaries`` helpers:
    the frozen task list, the paired complete case, the NI rule and the savings readout)."""
    import primaries as P  # noqa: PLC0415

    R = P.EpisodeRun("E2a", P.CONFIGS["E2a"], runs / "e2a_jevgym_tool_decisions", flags, None, False)
    out: dict[str, dict[str, Any]] = {}
    for name, arm, _, _, _ in configs:
        if arm == "llm_only":
            out[name] = _blank(E2_MARGIN)
            continue
        A, B = R.arm(arm), R.arm("llm_only")
        if A is None or B is None:
            out[name] = {"missing": True}
            continue
        exp = R.expected(arm, "llm_only")
        ta, tb = P.task_table(A, exp), P.task_table(B, exp)
        p = P.pair_tables(tb, ta, exp, names=("llm_only", arm))
        blk = P.ni_block(p["ref_y"], p["arm_y"], E2_MARGIN, n_boot, seed)
        cb = P.cost_block(tb, ta, p["keys"], n_boot, seed)
        out[name] = {"arm": arm, "n": p["n_pairs"], "n_dropped": p["n_dropped"], "delta": blk["diff"],
                     "delta_ci": blk["ci95"], "qualifies": bool(blk["ni_95"]), "p_decisive": blk["p_decisive"],
                     "rule": blk["rule"], "margin": E2_MARGIN, "cost_savings": cb.get("cost_savings", math.nan),
                     "success_arm": blk["success_arm"], "success_ref": blk["success_ref"],
                     "file": A["path"], "sha256": A["sha256"]}
    out["_reference"] = {"arm": "llm_only", "file": R.arm("llm_only")["path"], "sha256": R.arm("llm_only")["sha256"]}
    return out


def tau2_rows(tau2_dir: Path, configs: Sequence[tuple[str, str, str, str, str]], n_boot: int,
              seed: int) -> dict[str, dict[str, Any]]:
    """tau^2 airline, 35 test tasks x 3 trials: each configuration vs. ``llm_only`` (majority pass for Delta and the
    qualifier, pass^1 descriptive, cost = per-task mean over the trials)."""
    import primaries as P  # noqa: PLC0415

    split = json.loads(P.AIRLINE_SPLIT.read_text())["split"]
    exp = [f"airline/{t}" for t in split["test"]]
    ref = P.load_arm(tau2_dir / "airline" / "llm_only.jsonl")
    if ref is None:
        raise SystemExit(f"{tau2_dir}/airline/llm_only.jsonl missing")
    tb = P.task_table(ref, exp, trials=TAU2_TRIALS, retry_rule=False)
    out: dict[str, dict[str, Any]] = {}
    for name, _, arm, _, _ in configs:
        if arm == "llm_only":
            out[name] = {**_blank(TAU2_MARGIN), "delta_pass1": 0.0}
            continue
        A = P.load_arm(tau2_dir / "airline" / f"{arm}.jsonl")
        if A is None:
            out[name] = {"missing": True}
            continue
        ta = P.task_table(A, exp, trials=TAU2_TRIALS, retry_rule=False)
        p = P.pair_tables(tb, ta, exp, names=("llm_only", arm))
        blk = P.ni_block(p["ref_y"], p["arm_y"], TAU2_MARGIN, n_boot, seed)
        keys = p["keys"]
        d1 = float(np.mean([ta[k]["pass1"] - tb[k]["pass1"] for k in keys])) if keys else math.nan
        cb = P.cost_block(tb, ta, keys, n_boot, seed)
        out[name] = {"arm": arm, "n": p["n_pairs"], "n_dropped": p["n_dropped"], "delta": blk["diff"],
                     "delta_ci": blk["ci95"], "delta_pass1": d1, "qualifies": bool(blk["ni_95"]),
                     "p_decisive": blk["p_decisive"], "rule": blk["rule"], "margin": TAU2_MARGIN,
                     "cost_savings": cb.get("cost_savings", math.nan), "success_arm": blk["success_arm"],
                     "success_ref": blk["success_ref"], "file": A["path"], "sha256": A["sha256"]}
    out["_reference"] = {"arm": "llm_only", "file": ref["path"], "sha256": ref["sha256"], "n_tasks": len(exp),
                         "trials": TAU2_TRIALS}
    return out


def build(runs: Path, tau2_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    import primaries as P  # noqa: PLC0415

    flags = P.Flags()
    allc = (*CONFIGS, *EXTRA)
    e2, t2 = e2_rows(runs, allc, n_boot, seed, flags), tau2_rows(tau2_dir, allc, n_boot, seed)
    rows = []
    for name, ea, ta, tp_e2, tp_t2 in allc:
        rows.append({"configuration": name, "e2_arm": ea, "tau2_arm": ta, "tool_params_tau_e2": tp_e2,
                     "tool_params_tau_tau2": tp_t2, "e2": e2[name], "tau2": t2[name],
                     "set": "A5.4 (e) list" if (name, ea, ta, tp_e2, tp_t2) in CONFIGS else "sensitivity"})
    missing = [r["configuration"] for r in rows if r["e2"].get("missing") or r["tau2"].get("missing")]
    prim = [r for r in rows if r["set"] == "A5.4 (e) list" and r["configuration"] not in missing]
    sens = [r for r in rows if r["configuration"] not in missing]
    res = {"registered": REGISTERED, "rows": rows, "missing": missing,
           "primary": {"configurations": [r["configuration"] for r in prim], **proxy(prim)},
           "sensitivity_with_tauhat_e2": {"configurations": [r["configuration"] for r in sens], **proxy(sens)},
           "readings": READINGS, "stated_difference": STATED_DIFFERENCE, "guard_mapping": GUARD_MAPPING,
           "slip_rule": SLIP, "references": {"e2": e2["_reference"], "tau2": t2["_reference"]},
           "flags": list(flags.items)}
    return res


READINGS = [
    "configurations: A5.4 (e)'s parenthetical list (LLM-only, CASCADE at each swept tau, ASSIST k = 3, policy GUARD; "
    "the airline proxy-check arms of MAIN_RUN_PLAN step 9); the E2-carried tau-hat* cascade, which also ran on both, "
    "is a sensitivity row (n = 10)",
    "paired Delta-success: E2 per-task success on the arm's registered tasks; tau^2 majority pass (>= 2 of 3; A6.14, "
    "A6.16 (e)) on the 35 airline test tasks, pass^1 descriptive",
    "iso-quality cost savings: paired net savings of the configurations that pass the `noninferior` qualifier at the "
    "unadjusted level (A6.18 (c); 3 pt on JevGym, 5 pt on tau^2 majority pass); the others tie below them",
    "Kendall tau-b; exact permutation p over all n! assignments, one-sided for agreement (tau > 0), two-sided alongside; "
    "tau-b undefined when a ranking is constant",
]
STATED_DIFFERENCE = (
    "PREREGISTRATION Deviations 2026-10-03 (tau^2 runs) item (7): the airline sweep rows were run with "
    "scripts/run_tau2.sh as registered in MAIN_RUN_PLAN step 9, which sets tool_select tau = tool_params tau, whereas "
    "the E2a sweep arms fix the argument threshold at 0.90 (A6.16 (h)); the E2 and tau^2 sweep configurations "
    "therefore share their argument threshold only at tau = 0.9, and the proxy check is reported with that difference.")
GUARD_MAPPING = ("tau^2 `jev_guard` (policy guard before WRITE calls + transfer guard) has no E2 twin; E2a's tool-call "
                 "guard `jev_ts_guard` is read as its E2 counterpart (results/primaries/c7_selection.json, A6.20 (d))")
SLIP = ("A6.20 (f) lists the proxy check's Kendall-tau code as a slip-rule item; it landed after the tau^2 runs and "
        "before the check is reported (Deviations item (7)); nothing was re-run")


def _f(x: Any, nd: int = 3, pct: bool = False, sign: bool = False) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    if math.isnan(v):
        return "–"
    v = 100 * v if pct else v
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def to_markdown(res: Mapping[str, Any]) -> str:
    L = ["# C7 proxy check (A5.4 (e)): E2 (JevGym, Flash) vs. τ² airline rankings", "",
         f"- generated {res['generated']} at git `{res['git']}` by `scripts/tau2_proxy_check.py`; registered: {res['registered']}",
         f"- stated difference: {res['stated_difference']}",
         f"- guard mapping (interpretation): {res['guard_mapping']}",
         f"- slip rule: {res['slip_rule']}"]
    L += [f"- reading: {r}" for r in res["readings"]]
    for key, title in (("primary", "A5.4 (e) configurations"), ("sensitivity_with_tauhat_e2", "sensitivity: + the E2-carried τ̂\\* cascade")):
        blk = res[key]
        L += ["", f"## Kendall τ-b — {title} (n = {len(blk['configurations'])})", "",
              "| ranking | role | τ-b | p (one-sided, τ > 0) | p (two-sided) | n | method |", "|---|---|---|---|---|---|---|"]
        for k in RANKINGS:
            r = blk[k]
            L.append(f"| {r['statistic']} | {r['role']} | {_f(r['tau_b'])} | {_f(r['p_greater'], 4)} | "
                     f"{_f(r['p_two_sided'], 4)} | {r['n']} | {r.get('method') or r.get('note')} |")
    L += ["", "## Configurations", "",
          "| configuration | E2a arm | τ² arm | args τ E2 / τ² | E2 n | E2 Δ pt | E2 NI (3 pt) | E2 savings % | τ² n | τ² Δ maj. pt | τ² Δ pass¹ pt | τ² NI (5 pt) | τ² savings % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        e, t = r["e2"], r["tau2"]
        L.append(f"| {r['configuration']}{' (sens.)' if r['set'] == 'sensitivity' else ''} | `{r['e2_arm']}` | `{r['tau2_arm']}` | "
                 f"{r['tool_params_tau_e2']} / {r['tool_params_tau_tau2']} | {e.get('n')} | {_f(e.get('delta'), 1, True, True)} | "
                 f"{'yes' if e.get('qualifies') else 'no'} | {_f(e.get('cost_savings'), 1, True)} | {t.get('n')} | "
                 f"{_f(t.get('delta'), 1, True, True)} | {_f(t.get('delta_pass1'), 1, True, True)} | "
                 f"{'yes' if t.get('qualifies') else 'no'} | {_f(t.get('cost_savings'), 1, True)} |")
    if res["missing"]:
        L += ["", "Missing configurations (left out): " + ", ".join(res["missing"])]
    L += ["", "## Inputs", "", f"- E2 reference `{res['references']['e2']['file']}` sha256 `{res['references']['e2']['sha256']}`",
          f"- τ² reference `{res['references']['tau2']['file']}` sha256 `{res['references']['tau2']['sha256']}` "
          f"({res['references']['tau2']['n_tasks']} airline test tasks × {res['references']['tau2']['trials']})"]
    for r in res["rows"]:
        for s in ("e2", "tau2"):
            if r[s].get("file"):
                L.append(f"- `{r[s]['file']}` sha256 `{r[s]['sha256']}`")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--tau2", default=None, help="tau^2 run directory (default <runs>/tau2)")
    ap.add_argument("--out", default=str(ROOT / "results" / "secondary"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import primaries as P  # noqa: PLC0415

    runs = Path(a.runs).resolve()
    res = build(runs, Path(a.tau2).resolve() if a.tau2 else runs / "tau2", a.n_boot, a.seed)
    res = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": P._git_head(), **res}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tau2_proxy_check.json").write_text(json.dumps(P._jsonable(res), indent=1, default=str))
    (out / "tau2_proxy_check.md").write_text(to_markdown(res))
    pr = res["primary"]
    print("proxy check (n = %d): tau-b Delta-success %s (p %s), iso-quality savings %s (p %s)" % (
        len(pr["configurations"]), _f(pr["delta_success"]["tau_b"]), _f(pr["delta_success"]["p_greater"], 4),
        _f(pr["iso_quality_savings"]["tau_b"]), _f(pr["iso_quality_savings"]["p_greater"], 4)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
