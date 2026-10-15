#!/usr/bin/env python3
"""The optional closed-family row of docs/PREREG_A7_STAGE2.md A7.12 (i): gpt-6.1-sol (X) with gpt-6-luna (X-small),
the three S2-1X arms under the same procedure. DESCRIPTIVE ONLY: no test, no p-value, outside every Holm table
(A7.12 (i)); it reports the quantities of the X rows of scripts/stage2_analysis.py (success, the success difference to
`llm_only_sol` with the paired task bootstrap and the Newcombe score interval, cost per episode and savings with the
paired task bootstrap, latency, System-1 / System-2 calls, query coverage, A7.16 (d) intervention counts and trigger
activity) plus the row's procedure record (eligibility, n, the calibration fits, the derived profile, spend). Every
number traces to the files it names (path, sha256, record count in the provenance block); nothing is typed by hand.

    python scripts/stage2_sol.py --strict
    # reads runs/stage2_sol (llm_only_sol, jevagent_best_sol, llm_only_solsmall), configs/harness/jevagent_best_sol.yaml,
    # results/stage2/sol_cost_gate.json, results/stage2/sol_capability_manifest.json, results/calibration/
    # tauhat_jev_shadow_cal_sol_{tool_select,tool_params}.json; writes results/stage2/sol/{summary.json,summary.md} and
    # paper/generated/stage2_sol.tex (\\defres{stage2_sol/...}{...}; the paper adds \\InputIfFileExists{generated/stage2_sol}{}{})

Cost: the ledger's cost at the registered frozen price (Sol $2.00 / $10.00 per M, cached input $0.10, A7.12 (i); Luna
$0.10 / $0.50 / $0.01), Jev's System-1 calls included, as every Stage-2 row; ``cost_incl_cache_writes`` adds the
cache-write surcharge of the pricing page (input x 1.25 on ``cache_write_tokens``; not part of the frozen price) as a
sensitivity. Pairs keep one outcome per task (the latest record; a task whose latest record is an infrastructure error
after three attempts leaves the pair, the drop rule). Simulated inputs and episodes outside the frozen Stage-2 list are
refused unless ``--allow-simulated`` / ``--allow-dev-tasks`` (then keys go under ``SIM/``)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import primaries as P  # noqa: E402
import sol_eligibility as SE  # noqa: E402
import stage2_analysis as S2  # noqa: E402
from jevharness.bench import metrics as M  # noqa: E402
from jevharness.harness import profile as HP  # noqa: E402

LABEL = "descriptive (A7.12 (i)): no test, outside every Holm table"
SOL, LUNA = SE.SOL, SE.LUNA
REF, PROFILE_ARM, SMALL_ARM = "llm_only_sol", "jevagent_best_sol", "llm_only_solsmall"
ARMS = (REF, PROFILE_ARM, SMALL_ARM)
PAIRS = ((PROFILE_ARM, REF), (SMALL_ARM, REF))
RUN_DIR = "stage2_sol"
PROFILE_SOL = ROOT / "configs/harness/jevagent_best_sol.yaml"
FIT_ARM = "jev_shadow_cal_sol"
FITS_DIR = ROOT / "results/calibration"
TEX_OUT = ROOT / "paper/generated/stage2_sol.tex"
OUT_DIR = ROOT / "results/stage2/sol"
NAN = float("nan")
_fin = S2._finite


def _surcharge(t: Mapping[str, Any]) -> float:
    return SE.record_cost(t)["cache_write_surcharge"]


def _wall(t: Mapping[str, Any]) -> float:
    return float(t.get("wall_ms") or 0.0) / 1000.0


class SolRun:
    def __init__(self, d: Path, flags: P.Flags, allow_dev: bool) -> None:
        self.r = S2.Run("sol", d, flags, S2.STAGE2_TASKS, allow_dev)
        self.dir, self.flags = d, flags
        self.tasks: dict[str, dict[str, dict[str, Any]]] = {}

    def arm(self, name: str) -> dict[str, Any] | None:
        return self.r.arm(name, keep=True)

    def per_task(self, name: str) -> dict[str, dict[str, Any]]:
        """task key -> the latest record (one outcome per task, as the paired analysis keeps)."""
        if name not in self.tasks:
            A = self.arm(name)
            self.tasks[name] = {P.task_key(t): t for t in (A or {}).get("trajs") or []}
        return self.tasks[name]


def arm_row(R: SolRun, name: str, n_boot: int, seed: int) -> dict[str, Any]:
    """One arm on its own expected tasks: success (Wilson 95 %), cost (frozen price; + cache writes), latency (model
    calls; episode wall clock), calls, tokens, coverage, served models; never a test."""
    A = R.arm(name)
    if A is None:
        return {"arm": name, "status": "not run (no trajectory file)"}
    exp = R.r.expected(name)
    tab = P.task_table(A, exp)
    ok = [k for k in exp if tab[k]["status"] == "ok"]
    pt = R.per_task(name)
    recs = [pt[k] for k in ok if k in pt]
    s2 = [c for t in recs for c in t.get("calls") or [] if c.get("component") == "s2"]
    served: dict[str, int] = {}
    tiers: dict[str, int] = {}
    for c in s2:
        m = c.get("meta") or {}
        served[str(m.get("served_model") or "(none)")] = served.get(str(m.get("served_model") or "(none)"), 0) + 1
        tiers[str(m.get("service_tier") or "(none)")] = tiers.get(str(m.get("service_tier") or "(none)"), 0) + 1
    n = len(ok)

    def per_ep(f: Any) -> float:
        return float(np.mean([f(t) for t in recs])) if recs else NAN

    def tok(key: str, meta: bool = False) -> float:
        return per_ep(lambda t: sum(float(((c.get("meta") or {}) if meta else c).get(key) or 0)
                                    for c in t.get("calls") or [] if c.get("component") == "s2"))

    k_succ = int(sum(tab[k]["y"] for k in ok))
    cost = float(np.mean([tab[k]["cost"] for k in ok])) if ok else NAN
    row = {"arm": name, "n_expected": len(exp), "n": n,
           "status_counts": {s: sum(1 for k in exp if tab[k]["status"] == s)
                             for s in ("ok", "errored", "retry_pending", "missing")},
           "attempts": int(A.get("n_records") or 0),
           "success": k_succ / n if n else NAN, "success_ci": list(M.wilson_ci(k_succ, n)) if n else [NAN, NAN],
           "cost_usd": cost, "cost_usd_incl_cache_writes": cost + per_ep(_surcharge) if ok else NAN,
           "s1_cost_usd": float(np.mean([tab[k]["s1_cost"] for k in ok])) if ok else NAN,
           "latency_s": float(np.mean([tab[k]["latency"] for k in ok])) if ok else NAN,
           "wall_s": per_ep(_wall),
           **S2._calls(A, ok),
           "input_tokens": tok("input_tokens"), "output_tokens": tok("output_tokens"),
           "cached_input_tokens": tok("cached_input_tokens", True), "cache_write_tokens": tok("cache_write_tokens", True),
           "reasoning_tokens": tok("reasoning_tokens", True),
           "coverage": S2._per_dp_coverage(A), "coverage_pooled": A.get("coverage"),
           "served_models": served, "service_tiers": tiers, "unpriced_models": S2._unpriced(A),
           "s2_model": A.get("s2_model")}
    row["llm_calls"] = row.pop("s2_calls")
    row["cached_share"] = (row["cached_input_tokens"] / row["input_tokens"]
                           if _fin(row["input_tokens"]) and row["input_tokens"] > 0 else NAN)
    return row


def pair_row(R: SolRun, arm: str, ref: str, n_boot: int, seed: int) -> dict[str, Any]:
    """arm vs ref on the arm's expected tasks (descriptive): Δ success (paired task bootstrap 95 %; Newcombe and
    Tango score intervals beside it), discordant counts, cost and savings (paired task bootstrap; frozen price, and with
    the cache-write surcharge), latency and wall-clock differences (bootstrap), calls."""
    A, B = R.arm(arm), R.arm(ref)
    out: dict[str, Any] = {"arm": arm, "reference": ref}
    if A is None or B is None:
        return {**out, "status": "not run (missing trajectory file)"}
    exp = R.r.expected(arm)
    ta, tb = P.task_table(A, exp), P.task_table(B, exp)
    p = P.pair_tables(tb, ta, exp, names=(ref, arm))
    P._status_flags(R.flags, f"sol {arm}", "arm", arm, p["status_arm"], len(exp))
    P._status_flags(R.flags, f"sol {arm}", "reference", ref, p["status_ref"], len(exp))
    keys = p["keys"]
    if not keys:
        return {**out, "status": "not available (no paired task)", "n_expected": len(exp)}
    bt = M.paired_bootstrap_diff(p["ref_y"], p["arm_y"], n=n_boot, seed=seed)
    cells = M.ni_cells(p["ref_y"], p["arm_y"])
    nc, tg = M.newcombe_interval(**cells), M.tango_interval(**cells)
    pa, pb = R.per_task(arm), R.per_task(ref)
    ta_w = {k: {**v, "cost": v["cost"] + _surcharge(pa[k])} for k, v in ta.items() if k in keys}
    tb_w = {k: {**v, "cost": v["cost"] + _surcharge(pb[k])} for k, v in tb.items() if k in keys}
    lat = [ta[k]["latency"] - tb[k]["latency"] for k in keys]
    wall = [_wall(pa[k]) - _wall(pb[k]) for k in keys]
    ca, cr = S2._calls(A, keys), S2._calls(B, keys)
    return {**out, "n_expected": p["n_expected"], "n_pairs": p["n_pairs"], "n_dropped": p["n_dropped"],
            "dropped": p["dropped"],
            "success_arm": float(np.mean([p["arm_y"][k] for k in keys])),
            "success_ref": float(np.mean([p["ref_y"][k] for k in keys])),
            "delta": bt["diff"], "delta_ci": [bt["ci_lo"], bt["ci_hi"]],
            "delta_ci_newcombe": nc["ci"], "delta_ci_tango": tg["ci"],
            "wins": cells["only_a"], "losses": cells["only_b"],
            "cost": P.cost_block(tb, ta, keys, n_boot, seed),
            "cost_incl_cache_writes": P.cost_block(tb_w, ta_w, keys, n_boot, seed),
            "latency_diff_s": float(np.mean(lat)), "latency_diff_s_ci": S2._boot_ci(lat, n_boot, seed),
            "wall_diff_s": float(np.mean(wall)), "wall_diff_s_ci": S2._boot_ci(wall, n_boot, seed),
            "calls_arm": ca, "calls_ref": cr}


def intervention(R: SolRun) -> dict[str, Any]:
    """A7.16 (d) counts per arm (``stage2_analysis.intervention_stats`` on one record per task) and calls avoided
    against `llm_only_sol`."""
    out: dict[str, Any] = {}
    for name in ARMS:
        path = R.dir / f"{name}.jsonl"
        if path.exists():
            out[name] = {"reference": None if name == REF else REF, **S2.intervention_stats(S2._one_per_task(path))}
    ref = out.get(REF)
    for name, row in out.items():
        if name != REF and ref and _fin(ref.get("llm_calls")) and _fin(row.get("llm_calls")):
            row["llm_calls_avoided"] = ref["llm_calls"] - row["llm_calls"]
    return out


def procedure(profile_path: Path, gate_path: Path, manifest_path: Path, fits_dir: Path) -> dict[str, Any]:
    """The row's procedure record: eligibility and n (the cost gate), the capability manifest, the calibration fits
    and the derived profile (A7.12 (b)-(e) as applied by A7.12 (i))."""
    out: dict[str, Any] = {"files": {nm: {"path": P._rel(p), "sha256": P._sha256(p)}
                                     for nm, p in (("gate", gate_path), ("manifest", manifest_path)) if p.exists()}}
    if gate_path.exists():
        g = json.loads(gate_path.read_text())
        out["gate"] = {"eligible": g.get("eligible"), "n": g.get("n"), "decision": g.get("decision"),
                       "envelope_usd": g.get("envelope_usd"), "predicted_usd": g.get("predicted_usd"),
                       "members": {k: {kk: v.get(kk) for kk in ("malformed_rate", "infra_errors_left", "served_models",
                                                               "cost_per_episode", "eligible")}
                                   for k, v in (g.get("members") or {}).items()}}
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        out["manifest"] = {"written": m.get("written"), "eligible_capabilities": m.get("eligible_capabilities"),
                           "served_models_seen": m.get("served_models_seen"),
                           "request_settings": m.get("request_settings")}
    fits = {}
    for dp in ("tool_select", "tool_params"):
        f = fits_dir / f"tauhat_{FIT_ARM}_{dp}.json"
        if f.exists():
            fit = json.loads(f.read_text())
            pr = fit.get("primary") or {}
            fits[dp] = {"path": P._rel(f), "sha256": P._sha256(f), "tau_hat": fit_tau(fit),
                        "n_records": fit.get("n_records"), "V": pr.get("V"), "delta_c": pr.get("delta_c")}
    out["fits"] = fits
    if profile_path.exists():
        prof = HP.load_profile(profile_path)
        sec = (prof.get("system2") or {}).get(SOL) or {}
        out["profile"] = {"path": P._rel(profile_path), "sha256": prof.get("_sha256"), "status": prof.get("status"),
                          "decision_points": sec.get("decision_points"), "notes": sec.get("notes"),
                          "skip": sec.get("skip") or (None if prof.get("status") == "selected"
                                                      else f"profile status {prof.get('status')!r}"),
                          "derived_from": prof.get("derived_from")}
    return out


def fit_tau(fit: Mapping[str, Any]) -> float | None:
    """The deployed τ̂* of a fit, None when no level is accepted (the profile then always defers)."""
    t = (fit.get("primary") or {}).get("tau_hat", fit.get("tau_hat"))
    if t is None:
        return None
    t = float(t)
    return None if not math.isfinite(t) or t > 1.0 else t


def analyse(a: argparse.Namespace) -> dict[str, Any]:
    flags = P.Flags()
    run_dir = Path(a.run) if a.run else Path(a.runs) / RUN_DIR
    proc = procedure(Path(a.profile), Path(a.gate), Path(a.manifest), Path(a.fits_dir))
    res: dict[str, Any] = {"registered": "docs/PREREG_A7_STAGE2.md A7.12 (i) (Deviations 2026-10-05)", "label": LABEL,
                           "procedure": proc, "arms": {}, "pairs": {}, "intervention": {}, "flags": flags.items,
                           "provenance": {"run": P._rel(run_dir), "arms": {}}}
    gate = proc.get("gate") or {}
    status = None
    if gate and not gate.get("eligible"):
        status = f"not run: {gate.get('decision')}"
    elif gate and gate.get("n") is None:
        status = f"not run: {gate.get('decision')}"
    elif not run_dir.is_dir():
        status = "not run (no run directory)"
        flags.warn("sol", f"{P._rel(run_dir)} missing")
    if status:
        res["status"] = status
        return _finish(res, a, simulated=False)
    R = SolRun(run_dir, flags, a.allow_dev_tasks)
    simulated = False
    for name in ARMS:
        A = R.arm(name)
        if A is None:
            flags.error("sol", f"arm `{name}` not found in {P._rel(run_dir)}")
            continue
        res["provenance"]["arms"][name] = {k: A.get(k) for k in ("path", "sha256", "n_records", "n_episodes",
                                                                 "partial_lines", "s2_model", "s1")}
        simulated = simulated or P.run_is_simulated(run_dir, [A])
        res["arms"][name] = arm_row(R, name, a.n_boot, a.seed)
        if res["arms"][name].get("unpriced_models"):
            flags.error("sol", f"`{name}`: unpriced model(s) {res['arms'][name]['unpriced_models']} (A7.12 (h))")
        want = {REF: SOL, PROFILE_ARM: SOL, SMALL_ARM: LUNA}[name]
        seen = set(res["arms"][name].get("served_models") or {})
        if seen and seen != {want}:
            flags.error("sol", f"`{name}`: served model(s) {sorted(seen)} differ from the pinned {want} (A7.12 (b))")
    skip = (proc.get("profile") or {}).get("skip")
    for arm, ref in PAIRS:
        if arm == PROFILE_ARM and skip:
            res["pairs"][arm] = {"arm": arm, "reference": ref, "status": f"not run ({skip})"}
            continue
        res["pairs"][arm] = pair_row(R, arm, ref, a.n_boot, a.seed)
    res["intervention"] = intervention(R)
    exp_n = {name: R.r.n(name) for name in ARMS}
    gn = gate.get("n")
    if gn is not None and any(v != gn for v in exp_n.values()):
        flags.warn("sol", f"arm task counts {exp_n} differ from the cost gate's n = {gn}")
    incomplete = [nm for nm, r in res["arms"].items() if (r.get("status_counts") or {}).get("missing")]
    res["status"] = ("incomplete (an arm is missing expected tasks: a forced stop or a run in progress; the achieved n "
                     "is each arm's n)" if incomplete else "complete")
    res["spend"] = SE.spend() if not a.no_spend else None
    if simulated and not a.allow_simulated:
        flags.error("inputs", "simulated run: never a Stage-2 number; --allow-simulated for a dry run")
    return _finish(res, a, simulated)


def _finish(res: dict[str, Any], a: argparse.Namespace, simulated: bool) -> dict[str, Any]:
    res["meta"] = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": P._git_head(),
                   "simulated": simulated, "n_boot": a.n_boot, "seed": a.seed}
    return res


# ====================================================================================== rendering
def _pct(v: Any, sign: bool = False) -> str:
    return "–" if not _fin(v) else (f"{100 * float(v):+.1f}" if sign else f"{100 * float(v):.1f}")


def _ci(c: Any, sign: bool = True) -> str:
    if not isinstance(c, (list, tuple)) or len(c) != 2 or not all(_fin(x) for x in c):
        return "–"
    return f"[{_pct(c[0], sign)}, {_pct(c[1], sign)}]"


def _num(v: Any, fmt: str = ".4g") -> str:
    return "–" if not _fin(v) else format(float(v), fmt)


def to_markdown(res: Mapping[str, Any]) -> str:
    sim = " (SIMULATED)" if res["meta"]["simulated"] else ""
    L = [f"# Stage 2, the optional `gpt-6.1-sol` row{sim} — {res['label']}", "",
         f"Status: **{res.get('status')}**. System-2 `gpt-6.1-sol` (reasoning effort low, its lowest), small model "
         "`gpt-6-luna` (effort none), OpenAI Responses API; Jev via OpenRouter as System-1; fresh Stage-2 tasks.", ""]
    pr = res.get("procedure") or {}
    g = pr.get("gate") or {}
    if g:
        L += [f"Procedure: eligibility {g.get('eligible')} ({g.get('decision')}; envelope ${g.get('envelope_usd')}, "
              f"predicted {json.dumps({k: round(v, 2) for k, v in (g.get('predicted_usd') or {}).items()})})."]
    fits = pr.get("fits") or {}
    if fits:
        L += ["Calibration (seeds 5150–5299): " + "; ".join(
            f"{dp} τ̂* {_num(f.get('tau_hat'))}" if f.get("tau_hat") is not None else f"{dp}: no accepted level"
            for dp, f in fits.items()) + "."]
    prof = pr.get("profile") or {}
    if prof:
        L += [f"Derived profile `{prof.get('path')}` ({prof.get('status')}): "
              f"{json.dumps(prof.get('decision_points'))}" + (f"; skip: {prof['skip']}" if prof.get("skip") else "")
              + "."]
    if res.get("spend"):
        L += [f"Spend so far (ledgers + cache-write surcharge, probe and smoke included): "
              f"${res['spend']['total_usd']:.2f}."]
    if not res.get("arms"):
        return "\n".join(L + [""])
    L += ["", "## Arms", "",
          "| arm | n | success [Wilson 95 %] | $ / episode | $ incl. cache writes | latency s | wall s | LLM calls | "
          "S1 calls | input tok | cached | reasoning tok | served model |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, r in res["arms"].items():
        if r.get("status"):
            L.append(f"| {nm} | – | {r['status']} |" + " |" * 10)
            continue
        L.append(f"| {nm} | {r['n']} / {r['n_expected']} | {_pct(r['success'])} {_ci(r['success_ci'], False)} | "
                 f"{_num(r['cost_usd'])} | {_num(r['cost_usd_incl_cache_writes'])} | {_num(r['latency_s'], '.3g')} | "
                 f"{_num(r['wall_s'], '.3g')} | {_num(r['llm_calls'], '.3g')} | {_num(r['s1_calls'], '.3g')} | "
                 f"{_num(r['input_tokens'], '.0f')} | {_pct(r['cached_share'])} % | {_num(r['reasoning_tokens'], '.0f')} | "
                 f"{', '.join(r.get('served_models') or {}) or '–'} |")
    L += ["", "## Against `llm_only_sol` (descriptive; no test)", "",
          "| row | n′ | success arm / ref | Δ success [paired bootstrap 95 %] | Newcombe 95 % | wins / losses | "
          "savings [95 %] | savings incl. cache writes | Δ latency s | Δ wall s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, r in res["pairs"].items():
        if r.get("status"):
            L.append(f"| {nm} | – | {r['status']} |" + " |" * 7)
            continue
        c, cw = r.get("cost") or {}, r.get("cost_incl_cache_writes") or {}
        L.append(f"| {nm} | {r['n_pairs']} | {_pct(r['success_arm'])} / {_pct(r['success_ref'])} | "
                 f"{_pct(r['delta'], True)} {_ci(r['delta_ci'])} | {_ci(r['delta_ci_newcombe'])} | "
                 f"{r['wins']} / {r['losses']} | {_pct(c.get('cost_savings'))} % {_ci(c.get('cost_savings_ci'))} | "
                 f"{_pct(cw.get('cost_savings'))} % {_ci(cw.get('cost_savings_ci'))} | "
                 f"{_num(r['latency_diff_s'], '+.3g')} | {_num(r['wall_diff_s'], '+.3g')} |")
    L += ["", "## Intervention reporting (A7.16 (d); descriptive)", "",
          "| arm | n | tool cov. | arg cov. | S2 calls / ep. | calls avoided | trigger questions | fires | fire rate | "
          "tasks with a fire |", "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, r in (res.get("intervention") or {}).items():
        L.append(f"| {nm} | {r['n']} | {_pct(r['tool_coverage'])} | {_pct(r['arg_coverage'])} | "
                 f"{_num(r.get('llm_calls'), '.4g')} | {_num(r.get('llm_calls_avoided'), '.4g')} | "
                 f"{r['trigger_questions']} | {r['trigger_fires']} | {_pct(r.get('trigger_fire_rate'))} | "
                 f"{r['trigger_tasks']} |")
    if res.get("flags"):
        L += ["", "## Flags", ""] + [f"- {f['level']} [{f['where']}] {f['msg']}" for f in res["flags"]]
    L += ["", "_Δ success: arm − `llm_only_sol` on the paired tasks, paired task bootstrap 95 % percentile interval "
          "(Newcombe score interval beside it). Savings: 1 − cost(arm) / cost(`llm_only_sol`) per episode at the frozen "
          "price (Jev's calls included), paired task bootstrap; 'incl. cache writes' adds the pricing page's cache-write "
          "surcharge (a sensitivity, not the registered price). Latency: summed model-call latency per episode "
          "(backoff excluded); wall: episode wall clock. Coverage is never called intervention (A7.16 (d))._", ""]
    return "\n".join(L)


def tex_rows(res: Mapping[str, Any]) -> dict[tuple[str, ...], tuple[str, Any]]:
    """(key path) -> (format kind, value) under ``stage2_sol/``: ``status``, ``label``, ``n``, ``envelope``,
    ``predicted_500``, ``fit/<dp>/tau``, ``spend``; ``arm/<arm>/{n, success, success_ci/lo|hi, cost_usd,
    cost_usd_cw, latency_s, wall_s, llm_calls, s1_calls, cached_share, reasoning_tokens}``; ``<arm>/{n_pairs, delta,
    delta_ci/lo|hi, delta_ci_newcombe/lo|hi, wins, losses, success_arm, success_ref, cost_arm, cost_ref,
    cost_savings, cost_savings_ci/lo|hi, cost_savings_cw, cost_savings_cw_ci/lo|hi, latency_diff_s, wall_diff_s}``
    for `jevagent_best_sol` and `llm_only_solsmall`; ``intervention/<arm>/{tool_coverage, arg_coverage, llm_calls,
    llm_calls_avoided, trigger_questions, trigger_fires, trigger_fire_rate, trigger_tasks}``."""
    rows: dict[tuple[str, ...], tuple[str, Any]] = {}

    def put(path: tuple[str, ...], kind: str, v: Any) -> None:
        rows[("stage2_sol",) + path] = (kind, v)

    def ci(path: tuple[str, ...], kind: str, c: Any) -> None:
        if isinstance(c, (list, tuple)) and len(c) == 2:
            put(path + ("lo",), kind, c[0])
            put(path + ("hi",), kind, c[1])

    put(("status",), "text", res.get("status"))
    put(("label",), "text", res.get("label"))
    pr = res.get("procedure") or {}
    g = pr.get("gate") or {}
    put(("n",), "int", g.get("n"))
    put(("envelope",), "usd", g.get("envelope_usd"))
    put(("predicted_500",), "f2", (g.get("predicted_usd") or {}).get("500"))
    for dp, f in (pr.get("fits") or {}).items():
        put(("fit", dp, "tau"), "usd", f.get("tau_hat"))
        put(("fit", dp, "always_defer"), "text", "yes" if f.get("tau_hat") is None else "no")
    if res.get("spend"):
        put(("spend",), "f2", res["spend"].get("total_usd"))
    for nm, r in (res.get("arms") or {}).items():
        b = ("arm", nm)
        if r.get("status"):
            put(b + ("status",), "text", r["status"])
            continue
        put(b + ("n",), "int", r.get("n"))
        put(b + ("success",), "pct", r.get("success"))
        ci(b + ("success_ci",), "pct", r.get("success_ci"))
        for k, kind in (("cost_usd", "usd"), ("latency_s", "f2"), ("wall_s", "f2"), ("llm_calls", "f2"),
                        ("s1_calls", "f2"), ("cached_share", "pct"), ("reasoning_tokens", "int")):
            put(b + (k,), kind, r.get(k))
        put(b + ("cost_usd_cw",), "usd", r.get("cost_usd_incl_cache_writes"))
    for nm, r in (res.get("pairs") or {}).items():
        b = (nm,)
        if r.get("status"):
            put(b + ("status",), "text", r["status"])
            continue
        for k in ("n_pairs", "n_dropped", "wins", "losses"):
            put(b + (k,), "int", r.get(k))
        put(b + ("delta",), "spct", r.get("delta"))
        ci(b + ("delta_ci",), "spct", r.get("delta_ci"))
        ci(b + ("delta_ci_newcombe",), "spct", r.get("delta_ci_newcombe"))
        put(b + ("success_arm",), "pct", r.get("success_arm"))
        put(b + ("success_ref",), "pct", r.get("success_ref"))
        c, cw = r.get("cost") or {}, r.get("cost_incl_cache_writes") or {}
        put(b + ("cost_arm",), "usd", c.get("cost_arm"))
        put(b + ("cost_ref",), "usd", c.get("cost_ref"))
        put(b + ("cost_savings",), "pct", c.get("cost_savings"))
        ci(b + ("cost_savings_ci",), "pct", c.get("cost_savings_ci"))
        put(b + ("cost_savings_cw",), "pct", cw.get("cost_savings"))
        ci(b + ("cost_savings_cw_ci",), "pct", cw.get("cost_savings_ci"))
        put(b + ("latency_diff_s",), "s", r.get("latency_diff_s"))
        ci(b + ("latency_diff_s_ci",), "s", r.get("latency_diff_s_ci"))
        put(b + ("wall_diff_s",), "s", r.get("wall_diff_s"))
    for nm, r in (res.get("intervention") or {}).items():
        b = ("intervention", nm)
        for k in ("tool_coverage", "arg_coverage", "trigger_fire_rate"):
            put(b + (k,), "pct", r.get(k))
        put(b + ("llm_calls",), "f2", r.get("llm_calls"))
        put(b + ("llm_calls_avoided",), "f2s", r.get("llm_calls_avoided"))
        for k in ("trigger_questions", "trigger_fires", "trigger_tasks"):
            put(b + (k,), "intc", r.get(k))
    return rows


def to_tex(res: Mapping[str, Any]) -> str:
    from jevharness.bench.paper_numbers import _seg

    meta = res["meta"]
    prefix = ("SIM",) if meta["simulated"] else ()
    vals: dict[str, str] = {}
    for path, (kind, v) in tex_rows(res).items():
        s = S2._tex_fmt(kind, v)
        if s is not None:
            vals["/".join(prefix + tuple(_seg(x) for x in path))] = s
    lines = ["% AUTO-GENERATED by scripts/stage2_sol.py -- do not edit by hand. A7.12 (i): descriptive, no test.",
             f"% generated {meta['generated']} at git {str(meta['git'])[:12]}"
             + ("; SIMULATED DRY RUN: keys under SIM/, never results" if meta["simulated"] else ""),
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(vals.items())]
    lines += [r"\makeatother", ""]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--run", default=None, help=f"the run directory (default <runs>/{RUN_DIR})")
    ap.add_argument("--profile", default=str(PROFILE_SOL))
    ap.add_argument("--gate", default=str(SE.OUT))
    ap.add_argument("--manifest", default=str(SE.MANIFEST))
    ap.add_argument("--fits-dir", default=str(FITS_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--tex", default=str(TEX_OUT))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-spend", action="store_true", help="do not read the spend ledgers (tests)")
    ap.add_argument("--allow-simulated", action="store_true")
    ap.add_argument("--allow-dev-tasks", action="store_true")
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    res = analyse(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(P._jsonable(P._strip(res)), indent=1, default=P._json_default) + "\n")
    (out / "summary.md").write_text(to_markdown(res))
    if a.tex:
        Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
        Path(a.tex).write_text(to_tex(res))
    errors = sum(1 for f in res["flags"] if f["level"] == "ERROR")
    print(f"[stage2_sol] {res.get('status')}; {errors} ERROR flag(s) -> {out}", file=sys.stderr)
    return 1 if (a.strict and errors) else 0


if __name__ == "__main__":
    sys.exit(main())
