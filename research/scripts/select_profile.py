#!/usr/bin/env python3
"""Select the JevAgentHarness best-practice profile from the Stage-1 main-run results by the mechanical rule of
docs/PREREG_A7_STAGE2.md A7.1, and write it (A7.1 (h)).

    # once, after the three passes of E2a, E2b, E2c (and E6 if complete), before any Stage-2 run
    python scripts/select_profile.py --runs runs --out results/stage2
    # dry run on simulated / development runs (outputs marked SIMULATED / DEV, profile status `development`)
    python scripts/select_profile.py --runs /tmp/dry --out /tmp/dry/stage2 --profile-out /tmp/dry/best.yaml \
        --tau2-out /tmp/dry/tau2_best.yaml --allow-simulated --allow-dev-tasks --max-tasks 30

Inputs (exhaustive, A7.1 (b)): the candidate arms of :data:`CANDIDATES` and their LLM-only references, read from
``<runs>/<experiment name of the registered YAML>`` (``--run CFG=DIR`` overrides one). Per candidate (A7.1 (d)): the
pair exactly as ``scripts/primaries.py`` builds it (``load_arm`` / ``task_table`` / ``pair_tables``: latest record per
task, paired complete case); *complete* = no missing task and no errored task with retries outstanding on either side;
*delegating* = non-negligible coverage (A6.20 (e) / A6.22 (a) 5) for replace / cascade, >= 1 executed System-1
decision for assist / guard / trigger; *non-inferior* = ``metrics.ni_pvalue(...)["p_decisive"] <= 0.05`` (``bb_ordered``,
m = 3 pt); *cheaper* = one-sided paired t-test of the per-task cost difference (reference - arm) p <= 0.05. Choice
(A7.1 (e)): the largest cost saving per group, ties within 2 pt -> the lowest authority rank; no qualifying candidate ->
the group stays with the LLM. Settings are copied from the candidate's registered decision-point blocks (A7.1 (f)).

Outputs: the profile (``--profile-out``, default configs/harness/jevagent_best.yaml; ``status: selected``), its tau^2
rendering (``--tau2-out``, default configs/tau2/stage2/jevagent_best.yaml; absent when nothing tau^2 exposes is
delegated) and ``<out>/profile_selection.json`` (every candidate's statistics and fate, provenance, flags)."""
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

import prereg_check as PC  # noqa: E402
import primaries as P  # noqa: E402
from jevharness.bench import metrics as M  # noqa: E402
from jevharness.harness import profile as HP  # noqa: E402

FLASH, LITE = "gemini-3.5-flash", "gemini-3.5-flash-lite"
MARGIN = 0.03                       # A7.1 (d) 3: JevGym and ALFWorld (§2)
ALPHA = 0.05                        # A7.1 (d) 3-4: unadjusted
TIE = 0.02                          # A7.1 (e) 2: the C7 tie width (A5.4 (b))
NEGLIGIBLE = P.NEGLIGIBLE_COVERAGE  # A6.20 (e)
CONFIGS = {"E2a": "jevgym_tool_select.yaml", "E2b": "jevgym_meta_control.yaml", "E2c": "baselines_jevgym.yaml",
           "E6": "alfworld_action_only.yaml"}
REQUIRED = ("E2a", "E2b", "E2c")    # E6 is optional (A7.1 (h)): without it the action group stays with the LLM
JEVGYM_GROUPS = ("tool", "stop", "reflection", "routing", "memory")
CALIBRATED_BACKENDS = ("jevagent_rl", "startlux_4b")   # System-1s with their own cascade fits (A7.1 (f), A7.18 (e) 4)
# A7.1 (b): (System-2, group, candidate arm, run key, reference arm, authority rank)
CANDIDATES: list[tuple[str, str, str, str, str, float]] = [
    (FLASH, "tool", "jev_ts_cascade_tauhat", "E2a", "llm_only", 4.0),
    (FLASH, "tool", "jev_ts_assist_k5", "E2a", "llm_only", 2.0),
    (FLASH, "tool", "jev_ts_assist_k3", "E2a", "llm_only", 2.5),
    (FLASH, "tool", "jev_ts_guard", "E2a", "llm_only", 1.0),
    (FLASH, "tool", "jev_ts_replace", "E2a", "llm_only", 5.0),
    (FLASH, "tool", "jev_ts_params_replace", "E2a", "llm_only", 6.0),
    (FLASH, "stop", "jev_stop_guard", "E2b", "llm_only", 1.0),
    (FLASH, "stop", "jev_stop_trigger", "E2b", "llm_only", 3.0),
    (FLASH, "reflection", "jev_reflect_trigger", "E2b", "llm_only", 3.0),
    (FLASH, "routing", "jev_route_tau0.4", "E2b", "llm_only", 3.0),
    (FLASH, "memory", "jev_mem", "E2b", "llm_only", 3.0),
    (FLASH, "action", "jev_action_assist_k3", "E6", "llm_react", 2.5),
    (FLASH, "action", "jev_action_replace", "E6", "llm_react", 5.0),
    (LITE, "tool", "jev_ts_cascade_tau0.9_s2lite", "E2a", "llm_only_s2lite", 4.0),
    (LITE, "tool", "jev_ts_assist_k3_s2lite", "E2a", "llm_only_s2lite", 2.5),
    (LITE, "tool", "jev_ts_guard_s2lite", "E2a", "llm_only_s2lite", 1.0),
    (LITE, "stop", "jev_stop_guard_s2lite", "E2b", "llm_only_s2lite", 1.0),
    (LITE, "reflection", "jev_reflect_trigger_s2lite", "E2c", "llm_only_s2lite", 3.0),
]
DELEGATING_MODES = ("replace", "cascade")
NAN = float("nan")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def registered_setting(cfg: Mapping[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    """The arm's decision-point blocks exactly as the registered YAML writes them (A7.1 (f)); arm-level keys dropped."""
    for a in cfg.get("arms") or []:
        if a.get("name") == arm:
            return {k: dict(v) for k, v in a.items() if k in HP.DP_NAMES}
    raise KeyError(f"arm {arm!r} not in the registered config {cfg.get('name')!r}")


def _modes(setting: Mapping[str, Mapping[str, Any]], group: str) -> list[str]:
    return [str(setting[dp].get("mode")) for dp in HP.DP_GROUPS[group] if dp in setting]


def activation(row: Mapping[str, Any], trajs: Sequence[Mapping[str, Any]], cov: Mapping[str, Any]) -> dict[str, Any]:
    """A7.14 (a): a control's activation statistic, read from its own intervention mechanism (never the Jev-specific
    delegation function for controls without those records). active = the control changed at least one decision."""
    kind = row.get("intervention")
    if kind == "llm_swap":
        return {"statistic": "void", "active": True}
    if kind in ("replace", "cascade"):
        cls = P.delegation_class(cov.get("tool_coverage"), cov.get("arg_coverage"))
        return {"statistic": "delegation", "tool_coverage": cov.get("tool_coverage"),
                "arg_coverage": cov.get("arg_coverage"), "class": cls, "active": cls == "non-negligible"}
    dps = {dp for dp in (row.get("setting") or {}) if str((row["setting"][dp] or {}).get("mode")) == kind}
    n = changed = 0
    for r in M.decision_records(list(trajs)):
        if r.get("dp") not in dps or r.get("mode") != kind or r.get("shadow"):
            continue
        n += 1
        if kind == "assist":
            offered = (r.get("gold_aux") or {}).get("offered")
            changed += int(offered is not None and len(offered) < len(r.get("options") or offered))
        elif kind == "guard":
            changed += int(r.get("s1_value") is False)
        elif kind == "trigger":
            fv = r.get("final_value")
            changed += int(fv == "small" if r.get("dp") == "routing" else bool(fv))
    names = {"assist": "strict candidate reductions", "guard": "vetoes", "trigger": "state changes"}
    return {"statistic": names.get(kind, kind), "decisions": n, "changes": changed, "active": changed > 0}


def control_config(sel: Mapping[str, Any]) -> dict[str, Any] | None:
    """A7.14 (b): the selected control as a fresh-seed experiment (n = 500, three passes), or None when an existing
    fresh arm has the identical configuration (`small_only` = Flash-Lite LLM-only = stage2_jevgym's llm_only_s2lite)."""
    if sel.get("arm") == "small_only":
        return None
    stage2 = yaml.safe_load((ROOT / "configs/experiments/stage2_jevgym.yaml").read_text())
    arm: dict[str, Any] = {"name": "best_control", **{k: v for k, v in (sel.get("setting") or {}).items()}}
    exp: dict[str, Any] = {
        "name": "stage2_best_control", "suite": stage2["suite"], "n_tasks": 500, "llm": stage2["llm"],
        "small_llm": stage2["small_llm"], "s1": {"type": "jev", "provider": "openrouter"},
        "s1_cache": "cache/s1_stage2.sqlite", "workers": 8, "arms": [arm],
        "selected_from": {k: sel.get(k) for k in ("run", "config", "arm", "savings", "p_ni", "p_cheaper", "n_pairs")}}
    spec = sel.get("s1_run") or sel.get("s1")
    if spec:
        exp["s1_backends"] = {"control": dict(spec)}
        arm["s1"] = "control"
    return exp


class ActionRun:
    """E6 (ALFWorld) has no frozen task list: the expected games are the reference arm's task ids, and the run must
    hold all n of its registered suite."""

    def __init__(self, run_dir: Path, flags: P.Flags) -> None:
        self.key, self.dir, self.flags = "E6", run_dir, flags
        self.exists = run_dir.is_dir()
        self.registered = PC.load(CONFIGS["E6"]) if (ROOT / "configs/experiments" / CONFIGS["E6"]).exists() else {}
        self.arms: dict[str, Any] = {}

    def arm(self, name: str, keep_trajs: bool = False) -> dict[str, Any] | None:
        if name not in self.arms or (keep_trajs and self.arms[name] is not None and "trajs" not in self.arms[name]):
            self.arms[name] = P.load_arm(self.dir / f"{name}.jsonl", keep_trajs) if self.exists else None
        return self.arms[name]

    def expected(self, ref: str) -> list[str]:
        r = self.arm(ref)
        ids = sorted({e["task"] for e in (r or {}).get("episodes", {}).values()})
        n = int((self.registered.get("suite") or {}).get("n") or len(ids))
        if len(ids) < n:
            self.flags.warn("E6", f"`{ref}` holds {len(ids)} of the suite's {n} games: the E6 pair is incomplete")
        return ids


class Selection:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.flags = P.Flags()
        root = Path(a.runs)
        dirs = {k: root / PC.load(cfg)["name"] for k, cfg in CONFIGS.items()}
        for kv in a.run or []:
            k, v = kv.split("=", 1)
            key = next((x for x, c in CONFIGS.items() if c == k or x == k), None)
            if key is None:
                raise SystemExit(f"--run {kv}: unknown config (one of {sorted(CONFIGS.values())})")
            dirs[key] = Path(v)
        self.runs: dict[str, Any] = {}
        for k, d in dirs.items():
            if k == "E6":
                self.runs[k] = ActionRun(d, self.flags)
            else:
                self.runs[k] = P.EpisodeRun(k, CONFIGS[k], d, self.flags, a.max_tasks, a.allow_dev_tasks)
        self.provenance: dict[str, Any] = {}
        self.simulated: dict[str, bool] = {}

    # ------------------------------------------------------------------ one candidate
    def _note(self, key: str, name: str, arm: Mapping[str, Any], run_dir: Path) -> None:
        self.provenance.setdefault(key, {"dir": P._rel(run_dir), "arms": {}})["arms"][name] = {
            k: arm.get(k) for k in ("path", "sha256", "n_records", "n_episodes", "partial_lines", "s2_model")}
        if key not in self.simulated or not self.simulated[key]:
            self.simulated[key] = P.run_is_simulated(run_dir, [arm])

    def evaluate(self, s2: str, group: str, arm: str, key: str, ref: str, rank: float) -> dict[str, Any]:
        R = self.runs[key]
        out: dict[str, Any] = {"system2": s2, "group": group, "arm": arm, "run": key, "config": CONFIGS[key],
                               "reference": ref, "authority_rank": rank, "qualifies": False}
        try:
            setting = registered_setting(PC.load(CONFIGS[key]), arm)
        except Exception as e:  # noqa: BLE001
            out["fate"] = f"not in the registered config: {e}"
            self.flags.error("selection", out["fate"])
            return out
        out["setting"] = setting
        if not R.exists:
            out["fate"] = f"run directory {P._rel(R.dir)} missing"
            if key in REQUIRED:
                self.flags.error("selection", f"{arm}: {out['fate']}")
            return out
        A, B = R.arm(arm, keep_trajs=True), R.arm(ref, keep_trajs=True)
        if A is None or B is None:
            out["fate"] = f"arm `{arm if A is None else ref}` not found in {P._rel(R.dir)}"
            (self.flags.error if key in REQUIRED else self.flags.warn)("selection", out["fate"])
            return out
        self._note(key, arm, A, R.dir)
        self._note(key, ref, B, R.dir)
        run_arm = next((x for x in (R.exp if key != "E6" else {}).get("arms") or [] if x.get("name") == arm), None)
        if run_arm is not None and {k: v for k, v in run_arm.items() if k in HP.DP_NAMES} != setting:
            self.flags.error("selection", f"{arm}: the run's experiment.yaml block differs from the registered one")
        out.update(self.pair_stats(R, key, A, B, arm, ref))
        # (d) 2 delegating
        modes = _modes(setting, group)
        trajs = [t for t in A.get("trajs") or [] if not t.get("error")]
        recs = M.decision_records(trajs)
        cov = A.get("coverage") or {}
        if any(m in DELEGATING_MODES for m in modes):
            if group == "action":
                share = (M.delegation_ratio(recs).get("by_dp") or {}).get("action", NAN)
                out["coverage"] = {"action_coverage": share}
                out["delegating"] = bool(_finite(share) and share >= NEGLIGIBLE)
            else:
                out["coverage"] = {k: cov.get(k) for k in ("tool_coverage", "arg_coverage", "arg_coverage_s1_calls",
                                                           "llm_free_steps")}
                out["delegation_class"] = P.delegation_class(cov.get("tool_coverage"), cov.get("arg_coverage"))
                out["delegating"] = out["delegation_class"] == "non-negligible"
        else:
            dps = HP.DP_GROUPS[group]
            n_exec = sum(1 for r in recs if r.get("dp") in dps and r.get("mode") in modes and not r.get("shadow"))
            out["coverage"] = {"s1_decisions_executed": n_exec}
            out["delegating"] = n_exec > 0
        out["qualifies"] = bool(out["complete"] and out["delegating"] and out.get("noninferior")
                                and out.get("cheaper") and _finite(out.get("savings")))
        out["fate"] = "qualifies" if out["qualifies"] else "not qualified: " + ", ".join(
            w for w, ok in (("incomplete", out["complete"]), ("not delegating", out["delegating"]),
                            ("not non-inferior", out.get("noninferior")), ("not cheaper", out.get("cheaper")))
            if not ok)
        return out

    def pair_stats(self, R: Any, key: str, A: Mapping[str, Any], B: Mapping[str, Any], arm: str,
                   ref: str) -> dict[str, Any]:
        """A7.1 (d) 1, 3, 4 and the printed readouts on one pair, exactly as scripts/primaries.py builds a pair."""
        if key == "E6":
            exp_ids = R.expected(ref)
        else:
            n = min(R.n_tasks(arm), R.n_tasks(ref))
            exp_ids = R.task_ids[:n]
        ta, tb = P.task_table(A, exp_ids), P.task_table(B, exp_ids)
        p = P.pair_tables(tb, ta, exp_ids, names=(ref, arm))
        bad = {side: {k: v for k, v in st.items() if k in ("missing", "retry_pending", "incomplete") and v}
               for side, st in (("arm", p["status_arm"]), ("reference", p["status_ref"]))}
        out: dict[str, Any] = {"n_expected": p["n_expected"], "n_pairs": p["n_pairs"], "n_dropped": p["n_dropped"],
                               "status_arm": p["status_arm"], "status_ref": p["status_ref"],
                               "complete": not any(bad.values())}           # (d) 1
        if not p["n_pairs"]:
            return {**out, "noninferior": False, "cheaper": False, "savings": NAN}
        nb = P.ni_block(p["ref_y"], p["arm_y"], MARGIN, self.a.n_boot, self.a.seed)
        out["ni"] = {k: nb.get(k) for k in ("n", "wins", "losses", "diff", "ci95", "p_decisive", "rule", "ni_95",
                                             "success_ref", "success_arm", "mde")}
        keys = p["keys"]
        d_cost = [tb[k]["cost"] - ta[k]["cost"] for k in keys]
        d_lat = [tb[k]["latency"] - ta[k]["latency"] for k in keys]
        tt = M.paired_t_test(d_cost) if len(keys) > 1 else {"p_greater": NAN}
        cb = P.cost_block(tb, ta, keys, self.a.n_boot, self.a.seed)

        def calls(X: Mapping[str, Any]) -> dict[str, float]:
            return {str(t.get("task_id")): float(((t.get("usage") or {}).get("s2") or {}).get("calls", 0) or 0)
                    for t in X.get("trajs") or []}

        s2a, s2b = calls(A), calls(B)
        common = [k for k in keys if k in s2a and k in s2b]
        out.update({
            "p_ni": nb["p_decisive"], "noninferior": bool(nb["p_decisive"] <= ALPHA),       # (d) 3
            "p_cheaper": tt.get("p_greater", NAN),
            "cheaper": bool(_finite(tt.get("p_greater")) and tt["p_greater"] <= ALPHA),     # (d) 4
            "cost_arm": cb.get("cost_arm"), "cost_ref": cb.get("cost_ref"), "savings": cb.get("cost_savings"),
            "savings_ci": cb.get("cost_savings_ci"),
            "latency_saving_s": float(np.mean(d_lat)) if d_lat else NAN,
            "llm_calls_avoided": float(np.mean([s2b[k] - s2a[k] for k in common])) if common else NAN})
        return out

    # ------------------------------------------------------------------ A7.14: best_control
    def evaluate_control(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """A7.14 (b): one row of configs/harness/control_candidates.yaml through A7.1 (d)'s complete-case, NI and cost
        screens and the row's own activation screen."""
        key, arm, ref = row["run"], row["arm"], row["reference"]
        out: dict[str, Any] = {k: row.get(k) for k in ("run", "config", "arm", "reference", "n", "intervention",
                                                       "groups", "s1", "s1_name", "setting", "duplicate_of")}
        out["qualifies"] = False
        if row.get("duplicate_of"):
            out["fate"] = f"not read: duplicate of {row['duplicate_of']} (A7.14 (a))"
            return out
        R = self.runs.get(key)
        A = R.arm(arm, keep_trajs=True) if R is not None and R.exists else None
        B = R.arm(ref, keep_trajs=True) if R is not None and R.exists else None
        if A is None or B is None:
            out["fate"] = f"arm `{arm if A is None else ref}` not found in {P._rel(R.dir) if R else key}"
            self.flags.warn("best_control", out["fate"])
            return out
        self._note(key, arm, A, R.dir)
        out.update(self.pair_stats(R, key, A, B, arm, ref))
        trajs = [t for t in A.get("trajs") or [] if not t.get("error")]
        out["activation"] = act = activation(row, trajs, A.get("coverage") or {})
        out["active"] = bool(act.get("active"))
        out["qualifies"] = bool(out["complete"] and out["active"] and out.get("noninferior") and out.get("cheaper")
                                and _finite(out.get("savings")))
        out["fate"] = "qualifies" if out["qualifies"] else "not qualified: " + ", ".join(
            w for w, ok in (("incomplete", out["complete"]), ("not active", out["active"]),
                            ("not non-inferior", out.get("noninferior")), ("not cheaper", out.get("cheaper")))
            if not ok)
        # the System-1 spec the Stage-1 run actually used (placeholder rates are set at run time, A4 / A6.18 (f))
        if row.get("s1_name"):
            out["s1_run"] = {k: v for k, v in ((R.exp.get("s1_backends") or {}).get(row["s1_name"]) or {}).items()
                             if k != "placeholder"} or None
        elif row.get("s1") is not None:
            out["s1_run"] = {k: v for k, v in (R.exp.get("s1") or {}).items() if k != "placeholder"} or None
        return out

    def run_controls(self) -> dict[str, Any]:
        path = ROOT / "configs/harness/control_candidates.yaml"
        table = yaml.safe_load(path.read_text())
        rows = [self.evaluate_control(r) for r in table["rows"]]
        q = [r for r in rows if r["qualifies"]]
        best = min(q, key=lambda r: (-r["savings"], -int(r.get("n_pairs") or 0), r["arm"]), default=None)
        if best:
            best["selected"] = True
        return {"registered": "docs/PREREG_A7_STAGE2.md A7.14", "table": P._rel(path),
                "table_sha256": P._sha256(path), "rows": rows,
                "best_control": best["arm"] if best else None,
                "reason": (f"the qualifying control with the largest Stage-1 saving ({100 * best['savings']:.1f} %); "
                           "Stage-1 savings are selection-biased, the fresh row is the estimate" if best else
                           "no listed control satisfied the prespecified Stage-1 screen")}

    # ------------------------------------------------------------------ the rule
    def run(self) -> dict[str, Any]:
        cands = [self.evaluate(*c) for c in CANDIDATES]
        res: dict[str, Any] = {"registered": "docs/PREREG_A7_STAGE2.md A7.1", "margin": MARGIN, "alpha": ALPHA,
                               "tie": TIE, "candidates": cands, "system2": {}}
        for s2 in (FLASH, LITE):
            sec: dict[str, Any] = {"groups": {}, "decision_points": {}}
            for g in HP.DP_GROUPS:
                q = [c for c in cands if c["system2"] == s2 and c["group"] == g and c["qualifies"]]
                sel, reason = choose(q)
                sec["groups"][g] = {"arm": sel["arm"] if sel else None, "reason": reason,
                                    "candidates": [c["arm"] for c in cands if c["system2"] == s2 and c["group"] == g]}
                if sel:
                    sec["decision_points"].update(sel["setting"])
                    sel["selected"] = True
            # A7.1 (g) 3 as first written (recorded, not run since A7.11 (a)): the strongest qualifying candidate overall
            key = lambda c: (-c["savings"], c["authority_rank"], -int(c.get("n_pairs") or 0), c["arm"])  # noqa: E731
            jq = [c for c in cands if c["system2"] == s2 and c["qualifies"] and c["group"] in JEVGYM_GROUPS]
            orig = min(jq, key=key, default=None)
            sec["best_single_original"] = orig["arm"] if orig else None
            # A7.11 (a): g* = the selected JevGym-applicable group whose SELECTED candidate saves most in Stage 1
            sel = [c for c in cands if c["system2"] == s2 and c.get("selected") and c["group"] in JEVGYM_GROUPS]
            top = min(sel, key=key, default=None)
            sec["best_single"] = top["arm"] if top else None
            sec["best_single_group"] = top["group"] if top else None
            sec["active_jevgym_groups"] = HP.active_groups(sec["decision_points"], JEVGYM_GROUPS)
            res["system2"][s2] = sec
        return res


LOGO_N, LOGO_COST_PER_EP, LOGO_ENVELOPE = 200, 0.043, 45.0   # A7.11 (c): n, planning $ / Flash episode, envelope


def logo_plan(active: Sequence[str], g_star: str | None) -> dict[str, Any]:
    """A7.11 (c): which leave-one-group-out arms run, decided outcome-blind when the profile is written. Every active
    group if |G| x n x the planning cost fits the envelope, else only g*; with |G| = 2 the row without g' is the
    `best_single` arm itself (configuration-identical rows reuse trajectories) and is not run again."""
    G = list(active)
    if len(G) < 2 or g_star is None:
        return {}
    est = len(G) * LOGO_N * LOGO_COST_PER_EP
    run = set(G) if est <= LOGO_ENVELOPE else {g_star}
    plan: dict[str, Any] = {}
    for g in G:
        if g not in run:
            plan[g] = (f"not run: |G| x {LOGO_N} x ${LOGO_COST_PER_EP} = ${est:.0f} exceeds ${LOGO_ENVELOPE:.0f}; only "
                       f"the row without g* ({g_star}) runs (A7.11 (c))")
        elif len(G) == 2 and g != g_star:
            plan[g] = (f"configuration-identical to best_single (the profile without {g} is {g_star} alone): its "
                       "first 200 tasks are reused (A7.11 (c))")
        else:
            plan[g] = None
    return plan


def choose(q: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, str]:
    """A7.1 (e): the largest saving; candidates within ``TIE`` of it tie and the lowest authority rank among them wins
    (then the larger n, then the arm name); no qualifying candidate -> None (the group stays with the LLM)."""
    if not q:
        return None, "no qualifying candidate: the group stays with the LLM (A7.1 (e) 1)"
    b = max(q, key=lambda c: c["savings"])
    tied = [c for c in q if c["savings"] >= b["savings"] - TIE]
    sel = min(tied, key=lambda c: (c["authority_rank"], -int(c.get("n_pairs") or 0), c["arm"]))
    if len(tied) > 1:
        reason = (f"{len(tied)} qualifying candidates within {100 * TIE:.0f} pt of the best saving "
                  f"({', '.join(c['arm'] for c in tied)}) -> the most conservative, {sel['arm']} (A7.1 (e) 2)")
    else:
        reason = f"the only qualifying candidate within {100 * TIE:.0f} pt of the best saving (A7.1 (e) 2)"
    return sel, reason


def build_profile(res: Mapping[str, Any], cands: Sequence[Mapping[str, Any]], marks: Sequence[str],
                  meta: Mapping[str, Any]) -> dict[str, Any]:
    """The profile YAML (A7.1 (g), (h)): both System-2 sections, best_single, the JevAgentRL threshold block, the
    ALFWorld rendering; ``status: selected`` (``development`` for dry runs, with the marks)."""
    by_arm = {(c["system2"], c["arm"]): c for c in cands}
    prof: dict[str, Any] = {
        "kind": "jevagent_profile", "version": 1, "name": "jevagent_best",
        "status": "development" if marks else "selected",
        "description": ("Selected per decision-point group from the Stage-1 main-run results by the rule of "
                        "docs/PREREG_A7_STAGE2.md A7.1 (scripts/select_profile.py); see results/stage2/"
                        "profile_selection.json for every candidate's statistics."),
        "selection": dict(meta), "system2": {}}
    if marks:
        prof["marks"] = list(marks)
    for s2, sec in res["system2"].items():
        dps = sec["decision_points"]
        out: dict[str, Any] = {"decision_points": dps,
                               "groups": {g: v["arm"] for g, v in sec["groups"].items()}}
        active = sec["active_jevgym_groups"]
        if not active:
            out["skip"] = ("not tested (no decision point qualified in Stage 1; the best-practice profile is LLM-only) "
                           "-- A7.1 (g) 5")
        bs, g_star = sec["best_single"], sec.get("best_single_group")
        orig = sec.get("best_single_original")
        if bs:
            # A7.11 (a): the profile restricted to g* (the same frozen profile and harness scalars, every other group off)
            restricted = {k: v for k, v in dps.items() if k in HP.DP_GROUPS[g_star]}
            entry: dict[str, Any] = {"group": g_star, "arm": bs, "decision_points": restricted,
                                     "stage1": {k: by_arm[(s2, bs)].get(k) for k in ("savings", "p_ni", "p_cheaper",
                                                                                      "n_pairs")}}
            if orig:
                entry["original_candidate"] = {  # A7.1 (g) 3 as first written: recorded, not run (A7.11 (a))
                    "arm": orig, "decision_points": by_arm[(s2, orig)]["setting"],
                    "stage1": {k: by_arm[(s2, orig)].get(k) for k in ("savings", "p_ni", "p_cheaper", "n_pairs")}}
            if len(active) < 2:
                entry["skip"] = ("composition not tested (the selected profile delegates at one decision-point group) "
                                 "-- A7.1 (g) 4")
            out["best_single"] = entry
        else:
            out["best_single"] = {"skip": "no qualifying candidate (A7.1 (g) 5)"}
        if s2 == FLASH:
            plan = logo_plan(active, g_star)
            if plan:
                out["logo"] = plan
        tool = {k: v for k, v in dps.items() if k in HP.DP_GROUPS["tool"]}
        if any(str(v.get("mode")) == "cascade" for v in tool.values()):
            # A7.1 (f) / A7.18 (e) 4: the cascade thresholds of JevAgentRL and StartLux-4B come from their own
            # calibration fits (scripts/fill_backend_tau.py); pending until written, so the runner skips those arms
            out["backends"] = {be: {dp: {"calibration": "pending"} for dp, v in tool.items()
                                    if str(v.get("mode")) == "cascade"} for be in CALIBRATED_BACKENDS}
        rend = HP.alfworld_rendering(dps)
        out["alfworld"] = rend if rend["decision_points"] else {
            "skip": "the profile's ALFWorld rendering delegates no decision point (A7.1 (i)); S2-L3b is not run"}
        prof["system2"][s2] = out
    return prof


def _jsonable(x: Any) -> Any:
    return P._jsonable(x)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--run", action="append", metavar="CFG=DIR", help="run directory of one config; repeatable")
    ap.add_argument("--out", default="results/stage2", help="directory of profile_selection.json")
    ap.add_argument("--profile-out", default="configs/harness/jevagent_best.yaml")
    ap.add_argument("--tau2-out", default="configs/tau2/stage2/jevagent_best.yaml")
    ap.add_argument("--control-out", default="configs/experiments/stage2_best_control.yaml",
                    help="the best_control fresh-seed config (A7.14 (b))")
    ap.add_argument("--n-boot", type=int, default=2000, help="task-bootstrap replicates (descriptive intervals only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=None, help="DRY RUNS: cap the expected task list per arm")
    ap.add_argument("--allow-simulated", action="store_true", help="DRY RUNS: accept simulated inputs (marked)")
    ap.add_argument("--allow-dev-tasks", action="store_true", help="DRY RUNS: accept non-frozen task lists (marked)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any ERROR flag (nothing is written then)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    sel = Selection(a)
    res = sel.run()
    sim = any(sel.simulated.values())
    if sim and not a.allow_simulated:
        sel.flags.error("inputs", "simulated run(s): never a profile; pass --allow-simulated for a dry run")
    dev = any(not getattr(r, "frozen", True) for k, r in sel.runs.items() if k != "E6" and r.exists)
    marks = (["SIMULATED"] if sim else []) + (["DEV"] if dev or a.max_tasks else [])
    meta = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": P._git_head(),
            "rule": "docs/PREREG_A7_STAGE2.md A7.1", "runs": {k: P._rel(r.dir) for k, r in sel.runs.items()}}
    res.update({"meta": {**meta, "marks": marks}, "flags": sel.flags.items, "provenance": sel.provenance})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    errors = sel.flags.n_errors
    if errors and a.strict:
        for f in sel.flags.items:
            print(f"[select_profile] {f['level']} {f['where']}: {f['msg']}", file=sys.stderr)
        (out / "profile_selection.json").write_text(json.dumps(_jsonable(res), indent=1, default=P._json_default))
        print(f"[select_profile] {errors} ERROR flag(s): no profile written", file=sys.stderr)
        return 1
    prof = build_profile(res, res["candidates"], marks, {**meta, "errors": errors})
    res["profile"] = prof
    (out / "profile_selection.json").write_text(json.dumps(_jsonable(res), indent=1, default=P._json_default))
    pp = Path(a.profile_out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    head = ("# JevAgentHarness best-practice profile, WRITTEN BY scripts/select_profile.py (docs/PREREG_A7_STAGE2.md "
            "A7.1); do not edit by hand.\n" + (f"# DRY RUN: {', '.join(marks)} inputs -- never a Stage-2 profile.\n"
                                               if marks else ""))
    pp.write_text(head + yaml.safe_dump(_jsonable(prof), sort_keys=False, allow_unicode=True))
    HP.load_profile(pp)          # the written file must load
    t2, notes = HP.tau2_rendering(prof["system2"][FLASH]["decision_points"])
    tp = Path(a.tau2_out)
    if t2 is not None:
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text("# tau^2 rendering of configs/harness/jevagent_best.yaml (A7.1 (i)), written by "
                      "scripts/select_profile.py" + (f"; DRY RUN ({', '.join(marks)})" if marks else "") + "\n"
                      + "".join(f"# dropped: {n}\n" for n in notes) + yaml.safe_dump(t2, sort_keys=False))
    elif tp.exists():
        tp.unlink()
    # A7.14: best_control in the same single run; its statistics and fate in control_selection.json
    ctl = sel.run_controls()
    ctl["meta"] = {**meta, "marks": marks}
    (out / "control_selection.json").write_text(json.dumps(_jsonable(ctl), indent=1, default=P._json_default))
    cp = Path(a.control_out)
    best = next((r for r in ctl["rows"] if r.get("selected")), None)
    cfg = control_config(best) if best else None
    if cfg is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text("# best_control (docs/PREREG_A7_STAGE2.md A7.14), WRITTEN BY scripts/select_profile.py: the qualifying "
                      "control with the\n# largest Stage-1 saving, run unchanged on the fresh seeds (descriptive; no test)."
                      + (f" DRY RUN ({', '.join(marks)})" if marks else "") + "\n#   for pass in 1 2 3; do "
                      "JEV_EVAL_RUNS=1 jevbench run " + str(cp) + " --out runs -o workers=8; done\n"
                      + yaml.safe_dump(_jsonable(cfg), sort_keys=False))
    elif cp.exists():
        cp.unlink()
    print(f"[select_profile] best_control: {ctl['best_control'] or 'none'}"
          + (" (reuses llm_only_s2lite: identical configuration)" if best and cfg is None else ""), file=sys.stderr)
    for s2, sec in res["system2"].items():
        groups = ", ".join(f"{g}={v['arm'] or 'LLM'}" for g, v in sec["groups"].items())
        print(f"[select_profile] {s2}: {groups}; best single: {sec['best_single']}", file=sys.stderr)
    print(f"[select_profile] -> {pp}{' and ' + str(tp) if t2 is not None else ''}; "
          f"{out / 'profile_selection.json'}{' (' + ', '.join(marks) + ')' if marks else ''}; {errors} ERROR flag(s)",
          file=sys.stderr)
    return 1 if (a.strict and errors) else 0


if __name__ == "__main__":
    sys.exit(main())
