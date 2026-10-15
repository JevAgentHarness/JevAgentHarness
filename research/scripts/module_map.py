#!/usr/bin/env python3
r"""The *module map*: what happens when a decision model takes over one module (decision point) of the agent harness
in one integration mode (paper Figure ``fig:modulemap``, App. ``app:modulemap``). Descriptive throughout: it re-reads
the Stage-1 run JSONLs (never the markdown reports) and labels the registered cells with their registered outcome,
which it copies from ``results/primaries/primaries.json`` and never re-decides.

    python scripts/module_map.py                       # runs/ -> results/module_map/, paper/generated/module_map.tex
    python scripts/module_map.py --runs DIR --out DIR --tex FILE [--n-boot 10000] [--seed 0] [--quiet]

**Inputs.** Stage 1: ``runs/e2a_jevgym_tool_decisions`` (E2a), ``runs/e2b_jevgym_meta_control`` (E2b),
``runs/e2c_jevgym_baselines`` (E2c), one ``<arm>.jsonl`` per arm; the effective harness configuration of an arm is read
from its records (``meta.config``), its decision model from ``backend`` / ``meta.s1.describe`` and the arm name, its
agent LLM from ``llm``. Stage 2 (a separate column, never mixed into a Stage-1 number or verdict):
``runs/stage2_jevgym`` arms that run a single decision-point group of the profile (``best_single``: the tool cascade;
``jevagent_minus_tool``: the reflection trigger), matched to the Stage-1 configuration with the same settings.
Measurement (shadow) arms and the LLM-only references are not rows. ALFWorld action-only control (``action``) was
not run in Stage 1: its cells read "not run". The exploratory row E6a (``runs/e6a_alfworld_eval``: ALFWorld valid_seen,
Flash, Jev at the action point in replace / cascade / assist vs ``llm_only``; ``docs/PREREG_A7_STAGE2.md`` Deviations
2026-10-06) and the exploratory row E6c (``runs/e6c_alfworld_eval``: the same games and settings, Jev as the action
guard vs that run's own ``llm_only``) are read by their own reader (:func:`explore_action`): same comparison, coverage
and verdict rule, but kept apart from every Stage-1 row, cell, count and module verdict, under their own keys
``mm/xcell/flash/action/<mode>/...`` (the figure draws them as exploratory tiles where the Stage-1 cell is "not run"). The exploratory row E6b
(``results/explore/e6b_modes/summary.json``, written by ``scripts/explore_e6b_modes.py`` from ``runs/e6b_modes``: fresh
JevGym seeds 300800-300999, Flash, the nine cells E6b added to the harness, each against its reference with the same
comparison and verdict rule) is read by :func:`explore_e6b` into the same ``mm/xcell/...`` keys (routing: both maps, as
the Stage-1 routing rows), equally apart from every Stage-1 row, cell, count and module verdict. Cells outside
``VALID_MODES`` carry their reason from ``harness.config.UNDEFINED_CELLS`` (state ``undefined`` or ``equivalent``, with
the equivalent mode in ``mm/cell/.../equiv``) or ``ACTION_PENDING`` (state ``not implemented``, drawn as not run). The
five REPLACE cells that E6b declared equivalent are implemented since E6e and run as their own arms (exploratory row
E6e, ``results/explore/e6e_modes/summary.json``, read like E6b's); the cell each one equals
(``harness.config.EQUIVALENT_NOTES``) stays in ``mm/cell/.../equiv`` and ``mm/xcell/.../equiv`` (the tile notes it).
The last three cells, the triggers at tool choice, tool arguments and action, were undefined until the exploratory row
E6f defined them as verification triggers (``harness.config.E6F_CELLS``; ``results/explore/e6f_modes/summary.json``,
written by ``scripts/explore_e6f_modes.py``: JevGym seeds 301600-301799 and ALFWorld valid_seen) and are read like
E6b's, so no cell is undefined any more.

**Rows.** One row per (decision point, mode, decision model, agent LLM) configuration = one arm. An arm that sets
several decision points appears in the cell of each (``joint``): every argument arm also delegates the tool choice,
and the memory-read assist arm also gates memory writes. Cells (rows x columns of the figure) are the decision points
x {replace, cascade, assist, guard, trigger}; ``harness.config.VALID_MODES`` fixes which pairs exist (else
"undefined" / "equivalent" / "not implemented", from ``harness.config``).
Agent LLMs: Flash, Flash-Lite, and for routing the Flash-Lite agent that hands a step to Flash ("route").

**Replicates.** E2c re-ran some E2a / E2b configurations next to its controls. Rows with identical settings, decision
model and agent LLM form one configuration; its *canonical* row is the registered-primary arm if there is one, else
the row with the most tasks, ties to the earlier run (E2a, E2b, E2c) -- the rule of the registered Stage-2 selection
(``results/stage2/profile_selection.json`` read ``jev_reflect_trigger`` from E2b). Other rows are ``replicate``s: they
are reported, never used for a verdict or a "best" choice.

**Comparison.** Each row is paired by task with the LLM-only arm of the same agent LLM in the same run (``llm_only``
for Flash, ``llm_only_s2lite`` for Flash-Lite; routing: Flash-Lite alone, ``small_only`` in E2b as registered for C4,
``llm_only_s2lite`` in E2c), on the first n frozen evaluation tasks both ran (``configs/eval/jevgym_eval_tasks.json``),
paired complete case (a task whose latest record errored, or that is missing, in either arm leaves the pair, A6.9).
Delta success in points with a paired task-bootstrap 95 % interval (``metrics.paired_bootstrap_diff``, B = 10000,
seed 0, the resampling of the registered NI test); cost per episode and saving 1 - C_arm / C_ref with a task-bootstrap
interval (as ``primaries.cost_block``). Routing rows also carry the comparison with Flash alone (descriptive).

**Coverage and activity.** ``coverage`` is the paper's coverage (Sec. 3, A6.22 (a) 5): the share of the point's
decisions executed from the decision model's answer -- tool choice: ``scorecard.call_coverage``'s ``tool_coverage``;
tool arguments: its ``arg_coverage`` (executed argument slots); defined for replace / cascade only (assist and guard
execute the LLM's answer). ``acted`` is the share of the point's opportunities at which the decision model changed
what the harness did, the activity the verdict rule needs: replace / cascade = coverage; assist = 100 % (the LLM
chooses from the model's shortlist at every decision); guard = proposals vetoed; trigger = questions on which the
model fired the module (yes). Rows without a decision model (always / LLM-gate references) have neither.

**Accuracy** (where a gold label exists; ``metrics.decision_table`` on the point's DecisionLog records): ``s1_acc``
(the model's answer), ``accepted_acc`` (answers not deferred), ``recall_k`` (tool-choice assist: an acceptable tool
in the top k); ``acc`` picks the mode's quantity: replace s1_acc, cascade accepted_acc, assist recall_k (memory read:
s1_acc), guard / trigger s1_acc (verdict against the gold label).

**Verdict rule (descriptive; fixed in this docstring before any number was computed, 2026-10-05).** With Delta in
points, saving s in % and activity a in %:

  1. *costs success*      Delta <= -3;
  2. *no saving*          s <= 0 (and Delta > -3);
  3. *takes over safely*  Delta >= -1 and s > 0 and a >= 5;
  4. *mixed*              otherwise (-3 < Delta < -1 with s > 0, or a < 5 with s > 0).

The verdict is not a test: a registered cell keeps its registered outcome (``registered``: label, decision string,
"holds" / "not established" by the rule of ``\resword``), printed next to the verdict, never replaced by it.

**Best configuration per module.** Among the canonical decision-model rows (Jev by default) of a module and agent LLM:
verdict order takes over safely > mixed > no saving > costs success; within takes-over-safely and mixed the larger
saving, then the larger Delta; within no-saving and costs-success the larger Delta, then the larger saving. A cell
of the figure shows its registered arm when the cell holds a registered primary on that agent LLM (the one whose
reference is the cell's reference first), else its best row, preferring rows that set this decision point alone
(``joint`` rows only when no single-point row exists). The module verdict is the best of the module's shown cells;
``global_best`` (over every canonical row) is reported, with a flag where its verdict differs.

**Outputs.** ``results/module_map/module_map.json`` (everything, provenance with sha256), ``module_map.md`` (the
verdict grid and per-module tables) and ``paper/generated/module_map.tex``: ``\defres{mm/<dp>/<mode>/<row>/<metric>}``
(rows), ``\defres{mm/cell/<llm>/<dp>/<mode>/<metric>}`` (the figure's cells: llm in flash, lite; routing shows the
route rows in both), ``\defres{mm/best/<llm>/<dp>/<metric>}`` (module verdicts) and ``mm/count/...``. Runs whose
configuration names a simulated LLM / decision model are refused."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.runner import arm_task_counts  # noqa: E402
from jevharness.bench.scorecard import SIM_LLM_PROVIDERS, SIM_S1_TYPES, call_coverage  # noqa: E402
from jevharness.harness.config import (ACTION_PENDING, DP, EQUIVALENT_NOTES, UNDEFINED_CELLS, VALID_MODES,  # noqa: E402
                                       Mode)
from jevharness.harness.trajectory import iter_records  # noqa: E402

NAN = float("nan")
N_BOOT, SEED = 10000, 0
STAGE1 = (("E2a", "e2a_jevgym_tool_decisions"), ("E2b", "e2b_jevgym_meta_control"), ("E2c", "e2c_jevgym_baselines"))
STAGE2 = ("S2", "stage2_jevgym")
STAGE2_ARMS = ("best_single", "jevagent_minus_tool")   # the profile's single-group arms that ran (A7.11)
# exploratory action rows (Deviations 2026-10-06; never Stage 1): E6a (replace / cascade / assist), E6c (guard)
EXPLORE_ACTION = (("E6a", "e6a_alfworld_eval"), ("E6c", "e6c_alfworld_eval"))
EXPLORE_E6B = ROOT / "results" / "explore" / "e6b_modes" / "summary.json"   # E6b cells (Deviations 2026-10-06)
EXPLORE_E6D = ROOT / "results" / "explore" / "e6d_modes" / "summary.json"   # E6d cells (Deviations 2026-10-06)
EXPLORE_E6E = ROOT / "results" / "explore" / "e6e_modes" / "summary.json"   # E6e replace cells (Deviations 2026-10-06)
EXPLORE_E6F = ROOT / "results" / "explore" / "e6f_modes" / "summary.json"   # E6f trigger cells (Deviations 2026-10-07)
FROZEN_TASKS = ROOT / "configs" / "eval" / "jevgym_eval_tasks.json"
PRIMARIES = ROOT / "results" / "primaries" / "primaries.json"
FLASH, LITE = "gemini-3.5-flash", "gemini-3.5-flash-lite"

# ---- the verdict rule (docstring) ---------------------------------------------------------------------------------
SAFE_DELTA, COST_DELTA, MIN_ACTED = -1.0, -3.0, 5.0       # points, points, percent
SAFE, MIXED, NOSAVE, COSTS = "takes over safely", "mixed", "no saving", "costs success"
VERDICTS = (SAFE, MIXED, NOSAVE, COSTS)
VCODE = {SAFE: 1, MIXED: 2, NOSAVE: 3, COSTS: 4, "not run": 0, "n/a": 9}
COLOUR = {SAFE: "green", MIXED: "yellow", NOSAVE: "yellow", COSTS: "pink"}   # the figure's colour classes
EPS = 1e-9

DPS = tuple(dp.value for dp in DP)
MODES = ("replace", "cascade", "assist", "guard", "trigger")   # the figure's columns
S1_MODES = frozenset(MODES)
REF_MODES = ("always", "llm")                                 # references without a decision model
LLM_GROUPS = ("flash", "lite", "route")
FIG_LLMS = ("flash", "lite")
REF_ARM = {"flash": "llm_only", "lite": "llm_only_s2lite"}
ROUTE_REF = {"E2b": "small_only", "E2c": "llm_only_s2lite"}   # Flash-Lite alone (C4's registered reference in E2b)
MODULE_NAME = {"tool_select": "tool choice", "tool_params": "tool arguments", "memory_write": "memory write",
               "memory_read": "memory read", "reflection": "reflection", "stop": "stop", "routing": "routing",
               "action": "action"}
# decision-model families (arm-name prefix; the registered arm names are systematic) and their display order
FAMILY_ORDER = ("jev", "lite", "embed", "bm25", "bm25task", "tfidf", "heuristic", "prior", "random", "rate10",
                "rate30", "rate50", "ratem", "randscore", "s2self", "none")
FAMILY_NAME = {"jev": "Jev", "lite": "Flash-Lite", "embed": "embedding", "bm25": "BM25", "bm25task": "BM25 (task+obs.)",
               "tfidf": "TF-IDF", "heuristic": "heuristic", "prior": "prior (most recent)", "random": "random",
               "rate10": "random, 10 %", "rate30": "random, 30 %", "rate50": "random, 50 %",
               "ratem": "random, matched rate", "randscore": "random score", "s2self": "the agent LLM (typed question)",
               "none": "no decision model"}
DEC_KEYS = ("dp", "mode", "options", "s1_value", "s1_conf", "s1_dist", "deferred", "s2_value", "final_value", "gold",
            "gold_status", "shadow", "s1_rank", "retry")


# ====================================================================================== helpers
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _rel(p: Path | str) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def word_of(decision: str | None) -> str:
    """A registered decision string as one status word, the rule of ``\\resword`` (paper/macros.tex)."""
    s = str(decision or "")
    if re.match(r"^not tested", s):
        return "not tested"
    if re.match(r"^not run", s):
        return "not run"
    if re.match(r"^(NI|superior|rejects)([^a-z]|$)", s):
        return "holds"
    if re.search(r"(^|[^a-z])not([^a-z]|$)", s):
        return "not established"
    return "?"


# ====================================================================================== loading
def slim(r: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of one trajectory record this analysis reads (states, observations and prompts dropped)."""
    res, u, meta = r.get("result") or {}, r.get("usage") or {}, r.get("meta") or {}
    steps = []
    for st in r.get("steps") or []:
        args = st.get("args")
        decs = []
        for d in st.get("decisions") or []:
            ga = d.get("gold_aux") or {}
            dd = {k: d.get(k) for k in DEC_KEYS}
            dd["gold_aux"] = {k: ga[k] for k in ("acceptable_tools", "offered") if k in ga}
            decs.append(dd)
        steps.append({"step": st.get("step"), "tool": st.get("tool"),
                      "args": {k: None for k in args} if isinstance(args, Mapping) else args,
                      "llm_calls": st.get("llm_calls", 0), "decisions": decs})
    tot, s1, s2 = u.get("total") or {}, u.get("s1") or {}, u.get("s2") or {}
    return {"task_id": str(r.get("task_id")), "config": r.get("config"), "backend": r.get("backend"),
            "llm": r.get("llm"), "error": r.get("error") or None,
            "result": {"success": bool(res.get("success"))},
            "usage": {"total": {"cost_usd": float(tot.get("cost_usd", 0.0) or 0.0)},
                      "s1": {"cost_usd": float(s1.get("cost_usd", 0.0) or 0.0), "calls": s1.get("calls", 0)},
                      "s2": {"calls": s2.get("calls", 0), "input_tokens": s2.get("input_tokens", 0)}},
            "steps": steps,
            "meta": {"config": meta.get("config") or {}, "s1": {"describe": (meta.get("s1") or {}).get("describe")}}}


def load_arm(path: Path) -> dict[str, Any] | None:
    """The latest record per episode id (the runner appends a fresh record when it retries an errored task)."""
    if not path.exists():
        return None
    latest: dict[str, dict[str, Any]] = {}
    n = 0
    for r in iter_records(path):
        if "task_id" not in r:
            continue
        n += 1
        latest[str(r["task_id"])] = slim(r)
    return {"path": _rel(path), "sha256": _sha256(path), "n_records": n, "episodes": latest}


def setting_of(ep: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The decision points an episode's harness configuration sets (off and shadow left out), with the parameters
    that define the mode: tau (cascade / guard / trigger), k (assist) and mode-specific extras."""
    cfg = (ep.get("meta") or {}).get("config") or {}
    out: dict[str, dict[str, Any]] = {}
    for dp in DPS:
        c = cfg.get(dp) or {}
        mode = c.get("mode", "off")
        if mode in ("off", "shadow"):
            continue
        s: dict[str, Any] = {"mode": mode}
        if mode in ("cascade", "guard", "trigger"):
            s["tau"] = c.get("tau")
        if mode == "assist":
            s["k"] = c.get("k")
        extra = {k: v for k, v in (c.get("extra") or {}).items() if k in ("always_defer", "answer_as_copy")}
        s.update(extra)
        out[dp] = s
    return out


def is_measurement(ep: Mapping[str, Any]) -> bool:
    cfg = (ep.get("meta") or {}).get("config") or {}
    return any((cfg.get(dp) or {}).get("mode") == "shadow" for dp in DPS)


def family_of(arm: str, backend: Any) -> str:
    if backend in (None, "", "none"):
        return "none"
    fam = arm.split("_", 1)[0]
    return fam if fam in FAMILY_NAME else str(backend)


def llm_group(setting: Mapping[str, Any], llm: str | None) -> str | None:
    if (setting.get("routing") or {}).get("mode") == "trigger":
        return "route"
    return {FLASH: "flash", LITE: "lite"}.get(str(llm))


def arm_cells(setting: Mapping[str, Mapping[str, Any]]) -> tuple[list[tuple[str, str]], bool]:
    """(the (decision point, mode) cells an arm belongs to, its listing cell first; joint = several decision points
    run by the decision model)."""
    s1 = {dp: c["mode"] for dp, c in setting.items() if c["mode"] in S1_MODES}
    ref = {dp: c["mode"] for dp, c in setting.items() if c["mode"] in REF_MODES}
    if "tool_select" in s1 and "tool_params" in s1:
        if s1["tool_select"] == "cascade":      # the sweep varies the tool threshold: listed under tool choice
            cells = [("tool_select", s1["tool_select"]), ("tool_params", s1["tool_params"])]
        else:                                   # adds argument delegation to the tool-only arm of its mode
            cells = [("tool_params", s1["tool_params"]), ("tool_select", s1["tool_select"])]
    elif s1.get("memory_read") == "assist" and "memory_write" in s1:
        cells = [("memory_read", "assist"), ("memory_write", s1["memory_write"])]
    elif s1:
        cells = [(dp, s1[dp]) for dp in DPS if dp in s1]
    elif ref:
        dp = next(d for d in DPS if d in ref)
        cells = [(dp, ref[dp])]
    else:
        cells = []
    return cells, len(s1) > 1


def run_is_simulated(exp: Mapping[str, Any]) -> bool:
    llm_specs = [exp.get("llm") or {}, exp.get("small_llm") or {}, *((exp.get("llms") or {}).values())]
    s1_specs = [exp.get("s1") or {}, *((exp.get("s1_backends") or {}).values())]
    sim = any(str(s.get("provider", "")).lower() in SIM_LLM_PROVIDERS for s in llm_specs if isinstance(s, dict))
    return sim or any(str(s.get("type", "")).lower() in SIM_S1_TYPES for s in s1_specs if isinstance(s, dict))


# ====================================================================================== paired comparison
def task_table(arm: Mapping[str, Any] | None, expected: Sequence[str]) -> dict[str, dict[str, Any]]:
    eps = (arm or {}).get("episodes") or {}
    out = {}
    for t in expected:
        e = eps.get(t)
        if e is None:
            out[t] = {"status": "missing"}
        elif e.get("error"):
            out[t] = {"status": "errored"}
        else:
            out[t] = {"status": "ok", "y": float(e["result"]["success"]),
                      "cost": float(e["usage"]["total"]["cost_usd"])}
    return out


def compare(ref: Mapping[str, Any], arm: Mapping[str, Any], expected: Sequence[str], n_boot: int = N_BOOT,
            seed: int = SEED) -> dict[str, Any]:
    """Paired complete case over ``expected`` (task order kept): Delta success (points) with the paired task-bootstrap
    95 % interval of ``metrics.paired_bootstrap_diff``, and the cost saving 1 - C_arm / C_ref with a task-bootstrap
    interval drawn as ``primaries.cost_block`` does."""
    ra, aa = task_table(ref, expected), task_table(arm, expected)
    keys = [t for t in expected if ra[t]["status"] == "ok" and aa[t]["status"] == "ok"]
    out: dict[str, Any] = {"n_expected": len(expected), "n": len(keys), "n_dropped": len(expected) - len(keys),
                           "dropped": {"missing_arm": sum(aa[t]["status"] == "missing" for t in expected),
                                       "missing_ref": sum(ra[t]["status"] == "missing" for t in expected),
                                       "errored_arm": sum(aa[t]["status"] == "errored" for t in expected),
                                       "errored_ref": sum(ra[t]["status"] == "errored" for t in expected)},
                           "keys": keys}
    if not keys:
        return out
    ref_y = {t: ra[t]["y"] for t in keys}
    arm_y = {t: aa[t]["y"] for t in keys}
    pb = M.paired_bootstrap_diff(ref_y, arm_y, n=n_boot, seed=seed)
    ca = np.array([aa[t]["cost"] for t in keys])
    cb = np.array([ra[t]["cost"] for t in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = 1.0 - ca[idx].mean(1) / cb[idx].mean(1)
    boot = boot[np.isfinite(boot)]
    sav = 1.0 - ca.mean() / cb.mean() if cb.mean() > 0 else NAN
    out.update({"success": float(np.mean(list(arm_y.values()))), "success_ref": float(np.mean(list(ref_y.values()))),
                "delta": pb["diff"], "delta_ci": [pb["ci_lo"], pb["ci_hi"]],
                "wins": int(sum(arm_y[t] > ref_y[t] for t in keys)), "losses": int(sum(arm_y[t] < ref_y[t] for t in keys)),
                "cost": float(ca.mean()), "cost_ref": float(cb.mean()), "saving": float(sav),
                "saving_ci": ([float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot)
                              else [NAN, NAN])})
    return out


# ====================================================================================== decision-level readouts
def point_metrics(trajs: Sequence[Mapping[str, Any]], dp: str, mode: str, setting: Mapping[str, Any],
                  has_model: bool) -> dict[str, Any]:
    """Coverage, activity and accuracy of one decision point of one arm (docstring)."""
    recs = [r for r in M.decision_records(list(trajs)) if not r.get("shadow")]  # type: ignore[arg-type]
    rs = [r for r in recs if r.get("dp") == dp and r.get("mode") == mode]
    out: dict[str, Any] = {"n_decisions": len(rs)}
    if not has_model:
        if mode == "llm" and rs:   # the LLM's own gate: how often it chose to act (descriptive)
            out["llm_gate_rate"] = float(np.mean([r.get("final_value") is True for r in rs]))
        return out
    cov = call_coverage(list(trajs), recs)
    coverage = NAN
    if mode in ("replace", "cascade"):
        coverage = cov["tool_coverage"] if dp == "tool_select" else (cov["arg_coverage"] if dp == "tool_params" else
                                                                      M.delegation_ratio(rs, dps=(dp,))["ratio"])
        acted, kind = coverage, "decided"
    elif mode == "assist":
        acted, kind = (1.0 if rs else NAN), "shortlist"
    elif mode == "guard":
        g = [r for r in rs if isinstance(r.get("s1_value"), bool)]
        acted, kind = (float(np.mean([r["s1_value"] is False for r in g])) if g else NAN), "vetoed"
    elif mode == "trigger":
        g = [r for r in rs if isinstance(r.get("s1_value"), bool)]
        acted, kind = (float(np.mean([r["s1_value"] is True for r in g])) if g else NAN), "fired"
    else:
        acted, kind = NAN, None
    out.update({"coverage": coverage, "acted": acted, "acted_kind": kind,
                "tool_coverage": cov["tool_coverage"], "arg_coverage": cov["arg_coverage"],
                "llm_calls": cov["llm_calls_per_episode"]})
    row = M.decision_table(rs).get(f"{dp}/{mode}", {}) if rs else {}
    out.update({"n_gold": row.get("n_gold", 0), "s1_acc": row.get("s1_acc", NAN),
                "accepted_acc": row.get("accepted_acc", NAN)})
    if mode == "assist" and dp == "tool_select":
        k = int(setting.get("k") or 3)
        rk = M.recall_at_k(rs, ks=(k,), modes=("assist",))
        out["recall_k"] = rk.get(f"recall@{k}", NAN)
        out["k"] = k
    acc = {"replace": out["s1_acc"], "cascade": out["accepted_acc"], "guard": out["s1_acc"],
           "trigger": out["s1_acc"]}.get(mode, NAN)
    if mode == "assist":
        acc = out.get("recall_k", out["s1_acc"]) if dp == "tool_select" else out["s1_acc"]
    out["acc"] = acc
    out["acc_kind"] = {"replace": "answer", "cascade": "accepted answers", "assist": "gold in shortlist"
                       if dp == "tool_select" else "answer", "guard": "verdict", "trigger": "verdict"}.get(mode)
    return out


# ====================================================================================== verdicts and ranking
def verdict(delta: float, saving: float, acted: float) -> str | None:
    """The descriptive verdict of one row (docstring rule). delta and saving are fractions (0.01 = 1 point / 1 %),
    acted a fraction or NaN (a row without a decision model: no verdict)."""
    if not (_finite(delta) and _finite(saving)):
        return None
    d, s = 100.0 * delta, 100.0 * saving
    if d <= COST_DELTA + EPS:
        return COSTS
    if s <= EPS:
        return NOSAVE
    if not _finite(acted):
        return None
    if d >= SAFE_DELTA - EPS and 100.0 * acted >= MIN_ACTED - EPS:
        return SAFE
    return MIXED


def rank_key(v: str | None, delta: float, saving: float) -> tuple[float, float, float]:
    """Smaller is better (docstring: best configuration per module)."""
    o = VERDICTS.index(v) if v in VERDICTS else len(VERDICTS)
    d = delta if _finite(delta) else -1e9
    s = saving if _finite(saving) else -1e9
    if v in (SAFE, MIXED):
        return (o, -s, -d)
    return (o, -d, -s)


# ====================================================================================== the analysis
class ModuleMap:
    def __init__(self, runs: Path, n_boot: int = N_BOOT, seed: int = SEED, primaries: Path = PRIMARIES,
                 frozen: Path = FROZEN_TASKS, stage1: Sequence[tuple[str, str]] = STAGE1,
                 stage2: tuple[str, str] | None = STAGE2, quiet: bool = False,
                 explore: tuple[str, str] | Sequence[tuple[str, str]] | None = EXPLORE_ACTION,
                 e6b: Path | None = EXPLORE_E6B, e6d: Path | None = EXPLORE_E6D,
                 e6e: Path | None = EXPLORE_E6E, e6f: Path | None = EXPLORE_E6F) -> None:
        self.runs, self.n_boot, self.seed, self.quiet = Path(runs), n_boot, seed, quiet
        self.stage1, self.stage2 = list(stage1), stage2
        # one (key, run directory) or several: the exploratory action rows (E6a, E6c)
        self.explore = ([tuple(explore)] if explore and isinstance(explore[0], str) else
                        [tuple(x) for x in explore or ()])
        self.e6b = Path(e6b) if e6b is not None else None
        self.e6d = Path(e6d) if e6d is not None else None
        self.e6e = Path(e6e) if e6e is not None else None
        self.e6f = Path(e6f) if e6f is not None else None
        self.frozen = json.loads(Path(frozen).read_text())["task_ids"] if Path(frozen).exists() else None
        self.prim = json.loads(Path(primaries).read_text()) if Path(primaries).exists() else {"primaries": {}}
        self.flags: list[str] = []
        self.provenance: dict[str, Any] = {}

    def log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, file=sys.stderr, flush=True)

    # ---------------------------------------------------------------- registered labels
    def registered(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """(run key, arm) -> registered primaries whose Jev constituent is that arm (C7, C8: other suites)."""
        out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        names = {f"runs/{d}": k for k, d in self.stage1}
        for lab, P in (self.prim.get("primaries") or {}).items():
            j = P.get("jev") or {}
            run = names.get(str(j.get("run") or "").rstrip("/"))
            if not run or not j.get("arm"):
                continue
            out[(run, j["arm"])].append({"label": lab, "decision": P.get("decision"), "word": word_of(P.get("decision")),
                                         "reference": j.get("ref"),
                                         "reference_run": names.get(str(j.get("ref_run") or "").rstrip("/"))})
            ref_run = names.get(str(j.get("ref_run") or "").rstrip("/"))
            if j.get("ref") and ref_run and j.get("ref") not in ("llm_only", "llm_only_s2lite", "small_only"):
                out[(ref_run, j["ref"])].append({"label": lab, "decision": P.get("decision"),
                                                 "word": word_of(P.get("decision")), "role": "reference",
                                                 "of_arm": j["arm"]})
        return out

    # ---------------------------------------------------------------- one run
    def expected(self, n: int, ref_n: int, arm_ids: Iterable[str], ref_ids: Iterable[str]) -> list[str]:
        """The tasks a row is paired on: the first min(n_arm, n_ref) frozen evaluation tasks (Stage 1); for another
        suite (Stage 2's fresh tasks) the tasks the arm ran, in the reference's record order."""
        arm_ids, ref_ids = list(arm_ids), list(ref_ids)
        if self.frozen is not None and set(arm_ids) | set(ref_ids) <= set(self.frozen):
            return list(self.frozen[: min(n, ref_n)])
        arm_set = set(arm_ids)
        return [t for t in dict.fromkeys(ref_ids) if t in arm_set][: min(n, ref_n)]

    def run_rows(self, key: str, dirname: str, reg: Mapping[tuple[str, str], list[dict[str, Any]]],
                 only: Sequence[str] | None = None) -> list[dict[str, Any]]:
        d = self.runs / dirname
        if not d.is_dir():
            self.flags.append(f"ERROR {key}: run directory {_rel(d)} missing")
            return []
        exp = yaml.safe_load((d / "experiment.yaml").read_text()) if (d / "experiment.yaml").exists() else {}
        if run_is_simulated(exp):
            raise SystemExit(f"{_rel(d)} names a simulated LLM / decision model: refused")
        counts = arm_task_counts(exp) if exp else {}
        default_n = int(exp.get("n_tasks") or 500)
        names = sorted(p.stem for p in d.glob("*.jsonl"))
        if only is not None:
            names = [n for n in names if n in only or n in REF_ARM.values() or n in ROUTE_REF.values()]
        prov: dict[str, Any] = {}
        refs: dict[str, dict[str, Any] | None] = {}

        def ref_arm(name: str) -> dict[str, Any] | None:
            if name not in refs:
                refs[name] = load_arm(d / f"{name}.jsonl")
                if refs[name] is not None:
                    prov[name] = {k: refs[name][k] for k in ("path", "sha256", "n_records")}
            return refs[name]

        rows: list[dict[str, Any]] = []
        for name in names:
            if name in REF_ARM.values() or name in ROUTE_REF.values():
                continue
            if only is not None and name not in only:
                continue
            arm = load_arm(d / f"{name}.jsonl")
            if arm is None or not arm["episodes"]:
                continue
            eps = arm["episodes"]
            first = next((e for e in eps.values() if not e.get("error") and e["meta"]["config"]),
                         next(iter(eps.values())))
            if is_measurement(first):
                del arm
                continue
            setting = setting_of(first)
            cells, joint = arm_cells(setting)
            if not cells:
                continue
            grp = llm_group(setting, first.get("llm"))
            if grp is None:
                self.flags.append(f"WARN {key}/{name}: agent LLM {first.get('llm')!r} is neither Flash nor Flash-Lite")
                continue
            ref_name = ROUTE_REF.get(key, "llm_only_s2lite") if grp == "route" else REF_ARM[grp]
            ref = ref_arm(ref_name)
            if ref is None:
                self.flags.append(f"ERROR {key}/{name}: reference {ref_name} missing")
                continue
            prov[name] = {k: arm[k] for k in ("path", "sha256", "n_records")}
            n_arm = int(counts.get(name, default_n))
            n_ref = int(counts.get(ref_name, default_n))
            expected = self.expected(n_arm, n_ref, arm["episodes"], ref["episodes"])
            cmp = compare(ref, arm, expected, self.n_boot, self.seed)
            if cmp["dropped"]["missing_arm"] or cmp["dropped"]["missing_ref"]:
                self.flags.append(f"WARN {key}/{name}: {cmp['dropped']} (incomplete run; tasks left the pair)")
            backend = first.get("backend")
            fam = family_of(name, backend)
            describe = (first.get("meta") or {}).get("s1", {}).get("describe") or {}
            ident = json.dumps({"setting": setting, "describe": {k: v for k, v in describe.items() if k != "cached"},
                                "family": fam, "llm": first.get("llm"), "group": grp}, sort_keys=True, default=str)
            trajs = [arm["episodes"][t] for t in cmp["keys"]]
            points = {}
            for dp, mode in cells:
                points[dp] = point_metrics(trajs, dp, mode, setting.get(dp) or {}, fam != "none")
            row = {"run": key, "arm": name, "family": fam, "family_name": FAMILY_NAME.get(fam, fam),
                   "group": grp, "llm": first.get("llm"), "reference": ref_name, "setting": setting,
                   "cells": [list(c) for c in cells], "joint": joint, "identity": ident,
                   "n_registered": n_arm, "registered": reg.get((key, name), []),
                   "cmp": {k: v for k, v in cmp.items() if k != "keys"}, "points": points}
            if grp == "route":   # descriptive second reference: Flash alone in the same run
                fl = ref_arm("llm_only")
                if fl is not None:
                    exp_f = self.expected(n_arm, int(counts.get("llm_only", default_n)), arm["episodes"],
                                          fl["episodes"])
                    row["vs_flash"] = {k: v for k, v in compare(fl, arm, exp_f, self.n_boot, self.seed).items()
                                       if k != "keys"}
            rows.append(row)
            self.log(f"  {key}/{name}: n {cmp['n']}, delta {100 * cmp.get('delta', NAN):+.1f} pt, "
                     f"saving {100 * cmp.get('saving', NAN):.1f} %")
            del arm, trajs
        self.provenance[key] = {"dir": _rel(d), "arms": prov}
        return rows

    # ---------------------------------------------------------------- everything
    def build(self) -> dict[str, Any]:
        reg = self.registered()
        rows: list[dict[str, Any]] = []
        for key, dirname in self.stage1:
            self.log(f"{key}: {dirname}")
            rows += self.run_rows(key, dirname, reg)
        order = {k: i for i, (k, _) in enumerate(self.stage1)}
        # canonical rows and replicates
        by_ident: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_ident[r["identity"]].append(r)
        for grp in by_ident.values():
            grp.sort(key=lambda r: (0 if any("role" not in x for x in r["registered"]) else 1,
                                    -r["n_registered"], order[r["run"]]))
            canon = grp[0]
            canon["canonical"] = True
            canon["replicates"] = [f"{r['arm']}@{r['run']}" for r in grp[1:]]
            for r in grp[1:]:
                r["canonical"] = False
                r["replicate_of"] = f"{canon['arm']}@{canon['run']}"
        # row ids (key-safe): the arm name; replicates get the run as a suffix
        for r in rows:
            r["id"] = r["arm"] if r["canonical"] else f"{r['arm']}-{r['run'].lower()}"
        # per-cell verdicts
        for r in rows:
            c = r["cmp"]
            r["cell_verdicts"] = {}
            for dp, mode in r["cells"]:
                pm = r["points"][dp]
                v = verdict(c.get("delta", NAN), c.get("saving", NAN), pm.get("acted", NAN)) \
                    if r["family"] != "none" else None
                r["cell_verdicts"][f"{dp}/{mode}"] = v
            for x in r["registered"]:
                x["same_reference"] = (x.get("role") != "reference" and x.get("reference") == r["reference"])
        # Stage 2 (fresh tasks): a separate column of the matching canonical row
        fresh: dict[str, dict[str, Any]] = {}
        if self.stage2 is not None:
            key, dirname = self.stage2
            self.log(f"{key}: {dirname} (fresh tasks, separate column)")
            for f in self.run_rows(key, dirname, {}, only=STAGE2_ARMS):
                match = [r for r in rows if r["canonical"] and r["identity"] == f["identity"]]
                if not match:
                    self.flags.append(f"WARN Stage 2 `{f['arm']}`: no Stage-1 configuration with its settings")
                    continue
                m = match[0]
                f["cell_verdicts"] = {f"{dp}/{mode}": verdict(f["cmp"].get("delta", NAN), f["cmp"].get("saving", NAN),
                                                              f["points"][dp].get("acted", NAN))
                                      for dp, mode in f["cells"]}
                m["fresh"] = {k: f[k] for k in ("run", "arm", "reference", "cmp", "points", "cell_verdicts")}
                fresh[f["arm"]] = {"matched": f"{m['arm']}@{m['run']}"}
        # stability: replicate / fresh verdicts that differ from the canonical one
        ids = {f"{r['arm']}@{r['run']}": r for r in rows}
        for r in rows:
            if not r["canonical"]:
                continue
            diff = {}
            for cell, v in r["cell_verdicts"].items():
                others = {f"replicate {x}": ids[x]["cell_verdicts"].get(cell) for x in r["replicates"]}
                if r.get("fresh"):
                    others["fresh tasks"] = r["fresh"]["cell_verdicts"].get(cell)
                d = {k: w for k, w in others.items() if w is not None and v is not None and w != v}
                if d:
                    diff[cell] = d
            r["unstable"] = diff
        res = {"rows": rows, "fresh": fresh}
        res["cells"], res["modules"] = self.cells(rows)
        res["counts"] = self.counts(res["cells"])
        res["explore_action"] = explore_action(self)          # E6a, E6c: apart from every Stage-1 row / cell / count
        res["explore_e6b"] = explore_e6b(self)                # E6b: likewise
        res["meta"] = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       "git": _git_head(), "script": "scripts/module_map.py", "n_boot": self.n_boot, "seed": self.seed,
                       "runs": {k: d for k, d in self.stage1}, "stage2": dict([self.stage2]) if self.stage2 else {},
                       "rule": {"costs_success": f"delta <= {COST_DELTA} pt", "no_saving": "saving <= 0",
                                "takes_over_safely": f"delta >= {SAFE_DELTA} pt, saving > 0, acted >= {MIN_ACTED} %",
                                "mixed": "otherwise"},
                       "provenance": self.provenance, "flags": self.flags,
                       "primaries": _rel(PRIMARIES) if PRIMARIES.exists() else None}
        return res

    @staticmethod
    def _cell_candidates(rows: Sequence[Mapping[str, Any]], group: str, dp: str, mode: str,
                         family: str = "jev") -> list[Mapping[str, Any]]:
        return [r for r in rows if r["canonical"] and r["family"] == family and r["group"] == group
                and [dp, mode] in r["cells"]]

    def cells(self, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        cells: dict[str, Any] = {}
        modules: dict[str, Any] = {}
        for llm in FIG_LLMS:
            for dp in DPS:
                group = "route" if dp == "routing" else llm
                shown = []
                for mode in MODES:
                    ck = f"{llm}/{dp}/{mode}"
                    if mode not in {m.value for m in VALID_MODES[DP(dp)]}:
                        cells[ck] = outside_cell(dp, mode)
                        continue
                    cand = self._cell_candidates(rows, group, dp, mode)
                    if not cand:
                        cells[ck] = {"state": "not run", "verdict": "not run", "vcode": VCODE["not run"],
                                     **equiv_of(dp, mode)}
                        continue
                    regd = [r for r in cand if any(x.get("role") != "reference" for x in r["registered"])]
                    regd.sort(key=lambda r: 0 if any(x.get("same_reference") for x in r["registered"]) else 1)
                    if regd:
                        pick, why = regd[0], "registered"
                    else:
                        single = [r for r in cand if not r["joint"]] or list(cand)
                        single.sort(key=lambda r: rank_key(r["cell_verdicts"][f"{dp}/{mode}"], r["cmp"].get("delta", NAN),
                                                           r["cmp"].get("saving", NAN)))
                        pick, why = single[0], "best"
                    v = pick["cell_verdicts"][f"{dp}/{mode}"]
                    labs = [x for x in pick["registered"] if x.get("role") != "reference"]
                    labs.sort(key=lambda x: 0 if x.get("same_reference") else 1)
                    pm = pick["points"][dp]
                    cells[ck] = {"state": "run", "row": pick["id"], "arm": pick["arm"], "run": pick["run"], "why": why,
                                 "joint": pick["joint"], "verdict": v, "vcode": VCODE.get(v or "", 0),
                                 "delta": pick["cmp"].get("delta"), "saving": pick["cmp"].get("saving"),
                                 "acted": pm.get("acted"), "coverage": pm.get("coverage"), "n": pick["cmp"].get("n"),
                                 "registered": labs[0]["label"] if labs else None,
                                 "registered_word": labs[0]["word"] if labs else None,
                                 "registered_all": [x["label"] for x in labs],
                                 "unstable": pick.get("unstable", {}).get(f"{dp}/{mode}", {}),
                                 "n_candidates": len(cand)}
                    alt = self._fresh_or_rep(pick, f"{dp}/{mode}")
                    if alt:
                        cells[ck]["alt_verdict"], cells[ck]["alt_vcode"] = alt, VCODE[alt]
                    shown.append((mode, cells[ck], pick))
                mk = f"{llm}/{dp}"
                if not shown:
                    modules[mk] = {"verdict": "not run", "vcode": 0}
                    continue
                shown.sort(key=lambda t: rank_key(t[1]["verdict"], t[1]["delta"], t[1]["saving"]))
                mode, c, pick = shown[0]
                allc = [r for r in rows if r["canonical"] and r["family"] == "jev" and r["group"] == group
                        and any(cc[0] == dp for cc in r["cells"])]
                allv = [(r, cc[1], r["cell_verdicts"][f"{dp}/{cc[1]}"]) for r in allc for cc in r["cells"] if cc[0] == dp]
                allv.sort(key=lambda t: rank_key(t[2], t[0]["cmp"].get("delta", NAN), t[0]["cmp"].get("saving", NAN)))
                gb = allv[0]
                modules[mk] = {"verdict": c["verdict"], "vcode": c["vcode"], "mode": mode, "row": c["row"],
                               "arm": c["arm"], "run": c["run"], "delta": c["delta"], "saving": c["saving"],
                               "acted": c["acted"], "registered": c["registered"],
                               "registered_word": c["registered_word"],
                               "global_best": {"row": gb[0]["id"], "mode": gb[1], "verdict": gb[2],
                                               "delta": gb[0]["cmp"].get("delta"), "saving": gb[0]["cmp"].get("saving")}}
                if gb[2] != c["verdict"]:
                    self.flags.append(f"NOTE {mk}: the shown cells give {c['verdict']!r}; the best of every canonical "
                                      f"row is {gb[0]['id']} ({gb[1]}): {gb[2]!r}")
        return cells, modules

    @staticmethod
    def _fresh_or_rep(row: Mapping[str, Any], cell: str) -> str | None:
        """The figure's corner: the verdict of fresh tasks (preferred) or of a replicate where its colour class differs
        from the row's (mixed and no saving share one colour; ``unstable`` keeps every difference)."""
        d = (row.get("unstable") or {}).get(cell) or {}
        own = COLOUR.get(row["cell_verdicts"].get(cell))
        for k in sorted(d, key=lambda k: 0 if k == "fresh tasks" else 1):
            if COLOUR.get(d[k]) != own:
                return d[k]
        return None

    @staticmethod
    def counts(cells: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for llm in FIG_LLMS:
            c = [v for k, v in cells.items() if k.startswith(llm + "/")]
            out[llm] = {"pairs": len(c), "valid": sum(v["state"] in ("run", "not run") for v in c),
                        "run": sum(v["state"] == "run" for v in c), "not_run": sum(v["state"] == "not run" for v in c),
                        "undefined": sum(v["state"] == "undefined" for v in c),
                        "equivalent": sum(v["state"] == "equivalent" for v in c),
                        "same_rule": sum(bool(v.get("equiv")) and v["state"] in ("run", "not run") for v in c),
                        "not_implemented": sum(v["state"] == "not implemented" for v in c),
                        **{VERDICTS_KEY[v]: sum(x.get("verdict") == v for x in c) for v in VERDICTS}}
        return out


VERDICTS_KEY = {SAFE: "safe", MIXED: "mixed", NOSAVE: "no_saving", COSTS: "costs"}
EQUIV_MODE = {"trigger": "trigger", "assist": "assist", "cascade": "cascade"}   # first word of an "equivalent" reason


def _equiv_parts(why: str) -> dict[str, str]:
    """``equiv`` (the mode) and ``equiv_note`` (e.g. "with k = 1", "at tau 0.5") of an equivalence reason."""
    head = why.split(":")[0]
    mode = EQUIV_MODE[head.split(" ")[0]]
    return {"equiv": mode, "equiv_note": head[len(mode):].strip()}


def equiv_of(dp: str, mode: str) -> dict[str, str]:
    """An implemented cell that behaves exactly as another cell at a fixed setting (``EQUIVALENT_NOTES``: the E6e
    replace cells): its ``equiv`` / ``equiv_note``; empty otherwise."""
    why = EQUIVALENT_NOTES.get((DP(dp), Mode(mode)))
    return _equiv_parts(why) if why else {}


def outside_cell(dp: str, mode: str) -> dict[str, Any]:
    """A (decision point, mode) pair outside ``VALID_MODES``: undefined, or equivalent to an implemented cell (drawn
    as the dashed n/a tile, with "--" or "= <mode>"), or meaningful but not implemented (``ACTION_PENDING``: drawn as
    not run). The reason comes from ``harness.config``."""
    key = (DP(dp), Mode(mode))
    if key in UNDEFINED_CELLS:
        kind, why = UNDEFINED_CELLS[key]
        out = {"state": kind, "verdict": "n/a", "vcode": VCODE["n/a"], "reason": why}
        if kind == "equivalent":
            out.update(_equiv_parts(why))
        return out
    if key in ACTION_PENDING:
        return {"state": "not implemented", "verdict": "not run", "vcode": VCODE["not run"],
                "reason": ACTION_PENDING[key]}
    return {"state": "n/a", "verdict": "n/a", "vcode": VCODE["n/a"]}


# ====================================================================================== exploratory action row (E6a)
def explore_action(mm: ModuleMap) -> dict[str, Any]:
    """The exploratory action rows (``mm.explore``: E6a, E6c): each run's Jev action arms on Flash, each paired with
    the same run's ``llm_only`` exactly as a Stage-1 row is (:meth:`ModuleMap.run_rows`: paired complete case, the
    task-bootstrap Delta and saving, coverage = the share of action decisions executed from the model's answer, assist
    acts at every step, guard acts on the proposals it vetoes) and labelled by the same verdict rule. The rows never
    join ``res["rows"]``, so no Stage-1 cell, count, best configuration or module verdict reads them; a cell here is
    ``state: "exploratory"``. A missing run leaves its cells empty."""
    runs = [(k, d) for k, d in mm.explore if (mm.runs / d).is_dir()]
    if not runs:
        return {"run": None, "rows": [], "cells": {}}
    rows: list[dict[str, Any]] = []
    for key, dirname in runs:
        mm.log(f"{key}: {dirname} (exploratory action row, apart from Stage 1)")
        rows += [r for r in mm.run_rows(key, dirname, {}) if r["family"] == "jev" and r["group"] in FIG_LLMS
                 and [c[0] for c in r["cells"]] == ["action"]]
    cells: dict[str, Any] = {}
    for r in rows:
        key = r["run"]
        dp, mode = r["cells"][0]
        r.update({"id": r["arm"], "canonical": True, "explore": key, "replicates": [], "unstable": {}})
        pm, c = r["points"][dp], r["cmp"]
        v = verdict(c.get("delta", NAN), c.get("saving", NAN), pm.get("acted", NAN))
        r["cell_verdicts"] = {f"{dp}/{mode}": v}
        ck = f"{r['group']}/{dp}/{mode}"
        if ck in cells:   # one arm per mode is registered over E6a and E6c; a second would be a configuration error
            mm.flags.append(f"WARN {key}: several arms for {ck}; showing {cells[ck]['arm']}")
            continue
        cells[ck] = {"state": "exploratory", "explore": key, "row": r["id"], "arm": r["arm"], "run": key,
                     "verdict": v, "vcode": VCODE.get(v or "", 0), "delta": c.get("delta"), "saving": c.get("saving"),
                     "acted": pm.get("acted"), "coverage": pm.get("coverage"), "acc": pm.get("acc"),
                     "n": c.get("n"), "wins": c.get("wins"), "losses": c.get("losses"),
                     "setting": r["setting"].get(dp)}
    return {"run": ", ".join(k for k, _ in runs), "dir": ", ".join(_rel(mm.runs / d) for _, d in runs),
            "runs": {k: _rel(mm.runs / d) for k, d in runs}, "rows": rows, "cells": cells}


def explore_action_markdown(res: Mapping[str, Any]) -> list[str]:
    x = res.get("explore_action") or {}
    if not x.get("cells"):
        return []
    out = [f"## Exploratory action row ({x['run']}: ALFWorld valid_seen, `{x['dir']}`; not Stage 1)", "",
           "Descriptive, outside every Holm table, not the registered S2-L3b (docs/PREREG_A7_STAGE2.md, Deviations "
           "2026-10-06). Same comparison and verdict rule as the Stage-1 rows; never counted in the grid above.", "",
           "| cell | arm | setting | n | succ. % | Δ pt [95 % CI] | wins / losses | $/1k ep. | saving % [95 % CI] | "
           "cov. % | acted % | acc. % | verdict |", "|" + "---|" * 13]
    for r in x["rows"]:
        dp, mode = r["cells"][0]
        c, pm = r["cmp"], r["points"][dp]
        ci, sci = c.get("delta_ci") or [NAN, NAN], c.get("saving_ci") or [NAN, NAN]
        out.append(f"| {dp}/{mode} ({r['run']}) | `{r['arm']}` | {_setting_text(r)} | {c.get('n')} | {_pt(c.get('success'), False)} | "
                   f"{_pt(c.get('delta'))} [{_pt(ci[0])}, {_pt(ci[1])}] | {c.get('wins')} / {c.get('losses')} | "
                   f"{1000 * c.get('cost', NAN):.1f} | {_pt(c.get('saving'), False)} [{_pt(sci[0], False)}, "
                   f"{_pt(sci[1], False)}] | {_pt(pm.get('coverage'), False)} | {_pt(pm.get('acted'), False)} | "
                   f"{_pt(pm.get('acc'), False)} | {r['cell_verdicts'].get(f'{dp}/{mode}') or '--'} |")
    return out + [""]


def explore_action_tex(res: Mapping[str, Any]) -> dict[str, str]:
    """Keys ``mm/xcell/<llm>/action/<mode>/...`` of the exploratory cells (the Stage-1 ``mm/cell`` keys are untouched)."""
    vals: dict[str, str] = {}
    for ck, c in ((res.get("explore_action") or {}).get("cells") or {}).items():
        p = f"mm/xcell/{ck}"
        vals[f"{p}/state"] = c["state"]
        vals[f"{p}/explore"] = c["explore"]
        vals[f"{p}/vcode"] = str(c["vcode"])
        vals[f"{p}/verdict"] = c["verdict"] or "?"
        vals[f"{p}/row"] = _tex_text(c["row"])
        vals.update(_num_keys(p, {"delta": c["delta"], "saving": c["saving"], "n": c["n"]},
                              {"acted": c["acted"], "coverage": c["coverage"], "acc": c["acc"]}))
        for k in ("wins", "losses"):
            if _finite(c.get(k)):
                vals[f"{p}/{k}"] = f"{c[k]:.0f}"
        if c.get("equiv"):       # E6e: the cell this replace equals (the tile notes "= <mode>")
            vals[f"{p}/equiv"] = c["equiv"]
            if c.get("equiv_note"):
                vals[f"{p}/equiv_note"] = _tex_text(c["equiv_note"])
    return vals


# ====================================================================================== exploratory cells (E6b)
def explore_e6b(mm: ModuleMap) -> dict[str, Any]:
    """The E6b cells (``mm.e6b``: ``results/explore/e6b_modes/summary.json`` of ``scripts/explore_e6b_modes.py``, which
    pairs each arm with its reference by this script's ``compare`` and labels it with this script's ``verdict``). Each
    becomes an exploratory cell of the Flash map (routing: of both maps, as the Stage-1 routing rows); the cells never
    join ``res["rows"]``, so no Stage-1 cell, count, best configuration or module verdict reads them. A cell that was
    run in Stage 1 is never overdrawn. A missing summary leaves it empty. The E6d cells (``mm.e6d``: the same readout,
    ``scripts/explore_e6d_modes.py``, fresh seeds 301000-301199) and the E6e replace cells (``mm.e6e``,
    ``scripts/explore_e6e_modes.py``, fresh seeds 301400-301599; each with the cell it equals, ``equiv``) are read the
    same way into the same cell map, and so are the E6f trigger cells (``mm.e6f``, ``scripts/explore_e6f_modes.py``:
    tool choice and arguments on JevGym seeds 301600-301799, the action on ALFWorld valid_seen; the four rows' cells are
    disjoint)."""
    cells: dict[str, Any] = {}
    runs: dict[str, Any] = {}
    for label, path in (("E6b", mm.e6b), ("E6d", getattr(mm, "e6d", None)), ("E6e", getattr(mm, "e6e", None)),
                        ("E6f", getattr(mm, "e6f", None))):
        if path is None or not path.exists():
            continue
        js = json.loads(path.read_text())
        mm.provenance[label] = {"summary": _rel(path), "sha256": _sha256(path), "run": js["meta"].get("run")}
        runs[label] = js["meta"].get("run")
        for arm, c in (js.get("cells") or {}).items():
            if c.get("state") != "exploratory":
                continue
            dp, mode = c["dp"], c["mode"]
            k, pm = c["cmp"], c["points"]
            for llm in (FIG_LLMS if dp == "routing" else ("flash",)):
                cells[f"{llm}/{dp}/{mode}"] = {
                    "state": "exploratory", "explore": label, "row": arm, "arm": arm, "run": js["meta"].get("run"),
                    "reference": c.get("reference"), "verdict": c.get("verdict"),
                    "vcode": VCODE.get(c.get("verdict") or "", 0), "delta": k.get("delta"), "saving": k.get("saving"),
                    "acted": pm.get("acted"), "coverage": pm.get("coverage"), "acc": pm.get("acc"), "n": k.get("n"),
                    "wins": k.get("wins"), "losses": k.get("losses"), **equiv_of(dp, mode)}
    if not runs:
        return {"run": None, "cells": {}}
    return {"run": ", ".join(str(r) for r in runs.values()), "runs": runs, "cells": cells}


def explore_e6b_markdown(res: Mapping[str, Any]) -> list[str]:
    x = res.get("explore_e6b") or {}
    if not x.get("cells"):
        return []
    out = [f"## Exploratory cells E6b / E6d / E6e / E6f (`{x['run']}`: fresh JevGym seeds 300800-300999 / "
           "301000-301399 / 301400-301599 / 301600-301799, and ALFWorld valid_seen for E6f's action trigger, Flash; not "
           "Stage 1)", "",
           "Descriptive, outside every Holm table (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06 and 2026-10-07; "
           "semantics: results/explore/e6{b,d,e,f}_modes/cells.md; full readouts: "
           "results/explore/e6{b,d,e,f}_modes/summary.md). Same "
           "comparison and verdict rule as the Stage-1 rows; never counted in the grid above. An E6e replace cell "
           "behaves as the cell after its \"=\" (harness.config.EQUIVALENT_NOTES) and is still run as its own arm.", "",
           "| cell | row | arm | reference | n | Δ pt | wins / losses | saving % | cov. % | acted % | acc. % | verdict |",
           "|" + "---|" * 12]
    for ck, c in sorted(x["cells"].items()):
        if not ck.startswith("flash/"):
            continue
        eq = f" (= {c['equiv']} {c.get('equiv_note') or ''})".replace(" )", ")") if c.get("equiv") else ""
        out.append(f"| {ck[6:]}{eq} | {c['explore']} | `{c['arm']}` | {c['reference']} | {c['n']} | {_pt(c['delta'])} | {c['wins']} / "
                   f"{c['losses']} | {_pt(c['saving'], False)} | {_pt(c['coverage'], False)} | {_pt(c['acted'], False)} | "
                   f"{_pt(c['acc'], False)} | {c['verdict'] or '--'} |")
    return out + [""]


def explore_e6b_tex(res: Mapping[str, Any]) -> dict[str, str]:
    """Keys ``mm/xcell/<llm>/<dp>/<mode>/...`` of the E6b cells, the format of :func:`explore_action_tex`."""
    return explore_action_tex({"explore_action": res.get("explore_e6b") or {}})


# ====================================================================================== reports
def _pt(x: Any, sign: bool = True) -> str:
    return "--" if not _finite(x) else (f"{100 * x:+.1f}" if sign else f"{100 * x:.1f}")


def _setting_text(r: Mapping[str, Any]) -> str:
    parts = []
    for dp in DPS:
        s = r["setting"].get(dp)
        if not s:
            continue
        p = f"{dp} {s['mode']}"
        if s.get("tau") is not None:
            p += f" tau={s['tau']}"
        if s.get("k") is not None:
            p += f" k={s['k']}"
        if s.get("always_defer"):
            p += " (always defer)"
        if s.get("answer_as_copy"):
            p += " (answer as copy)"
        parts.append(p)
    return "; ".join(parts)


SHORT = {SAFE: "SAFE", MIXED: "mixed", NOSAVE: "no-sav", COSTS: "COSTS", "not run": "not run", "n/a": "n/a", None: "?"}


def grid_text(res: Mapping[str, Any]) -> str:
    lines = []
    for llm in FIG_LLMS:
        lines.append(f"{'Flash' if llm == 'flash' else 'Flash-Lite'} (routing: Flash-Lite agent handing steps to Flash, "
                     "vs. Flash-Lite alone)")
        head = f"{'module':<15}" + "".join(f"{m.upper():<21}" for m in MODES) + "| module verdict (best shown cell)"
        lines.append(head)
        for dp in DPS:
            s = f"{MODULE_NAME[dp]:<15}"
            for mode in MODES:
                c = res["cells"][f"{llm}/{dp}/{mode}"]
                t = SHORT.get(c.get("verdict"), str(c.get("verdict")))
                if c["state"] == "undefined":
                    t = "--"
                elif c["state"] == "equivalent":
                    t = f"= {c['equiv']}"
                elif c["state"] == "not implemented":
                    t = "not impl."
                if c.get("equiv") and c["state"] in ("run", "not run"):     # E6e replace: same rule as that mode
                    t += f" (={c['equiv']})"
                if c["state"] == "run":
                    t += f" {100 * c['delta']:+.1f}/{100 * c['saving']:.0f}%"
                    if c.get("registered"):
                        t += f" {c['registered']}"
                    if c.get("alt_verdict"):
                        t += "*"
                s += f"{t:<21}"
            m = res["modules"][f"{llm}/{dp}"]
            s += f"| {m['verdict']}"
            if m.get("mode"):
                s += f" ({m['mode']}, {m['row']}"
                if m.get("registered"):
                    s += f"; registered {m['registered']}: {m['registered_word']}"
                s += ")"
            lines.append(s)
        lines.append("")
    lines.append("cell: verdict delta(pt)/saving(%) [registered label]; * = a replicate or the fresh-task run of the same "
                 "configuration gave a verdict of another colour class (mixed and no saving share one)")
    x6 = {k: c for k, c in ((res.get("explore_e6b") or {}).get("cells") or {}).items() if k.startswith("flash/")}
    if x6:   # E6b, E6d, E6e, E6f: exploratory, not part of the grid's Stage-1 cells or counts
        lines.append("exploratory E6b / E6d / E6e / E6f cells (" + str(res["explore_e6b"]["run"]) + ", fresh JevGym seeds; not "
                     "Stage 1): " + "; ".join(f"{k[6:]} {SHORT.get(c['verdict'], c['verdict'])} {_pt(c['delta'])}/"
                                              f"{_pt(c['saving'], False)}%" for k, c in sorted(x6.items())))
    lines.append("undefined / equivalent cells (harness.config.UNDEFINED_CELLS): " + ("; ".join(
        f"{dp.value}/{m.value} {kind}" for (dp, m), (kind, _) in UNDEFINED_CELLS.items())
        or "none (E6f defined the last three)"))
    lines.append("implemented with the rule of another cell (harness.config.EQUIVALENT_NOTES; run as their own arms by "
                 "E6e): " + "; ".join(f"{dp.value}/{m.value} = {why.split(':')[0]}"
                                      for (dp, m), why in EQUIVALENT_NOTES.items()))
    xc = (res.get("explore_action") or {}).get("cells") or {}
    if xc:   # E6a: exploratory, not part of the grid's Stage-1 cells or counts
        lines.append("exploratory action row (" + str(res["explore_action"]["run"]) + ", ALFWorld valid_seen; not Stage 1): "
                     + "; ".join(f"{k} {SHORT.get(c['verdict'], c['verdict'])} {_pt(c['delta'])}/{_pt(c['saving'], False)}%"
                                 for k, c in sorted(xc.items())))
    return "\n".join(lines)


def to_markdown(res: Mapping[str, Any]) -> str:
    meta = res["meta"]
    out = ["# Module map (descriptive)", "",
           f"- generated {meta['generated']} at git `{meta['git'][:12]}` by `{meta['script']}`; B = {meta['n_boot']}, "
           f"seed {meta['seed']}; runs: " + ", ".join(f"{k} `runs/{d}`" for k, d in meta["runs"].items())
           + "; Stage 2 (separate column): " + ", ".join(f"`runs/{d}`" for d in meta["stage2"].values()),
           "- verdict rule (fixed before the numbers): costs success if Δ ≤ −3 pt; no saving if saving ≤ 0; takes over "
           "safely if Δ ≥ −1 pt, saving > 0 and the model acted at ≥ 5 % of the point's opportunities; else mixed. "
           "Registered cells keep their registered outcome (label: word).",
           "- flags: " + ("; ".join(meta["flags"]) if meta["flags"] else "none"), "",
           "## Verdict grid", "", "```", grid_text(res), "```", ""]
    out += explore_action_markdown(res)
    out += explore_e6b_markdown(res)
    for llm_grp in LLM_GROUPS:
        rs = [r for r in res["rows"] if r["group"] == llm_grp]
        if not rs:
            continue
        name = {"flash": "Flash", "lite": "Flash-Lite", "route": "Routing (Flash-Lite → Flash) vs. Flash-Lite alone"}[llm_grp]
        out += [f"## {name}", ""]
        out.append("| cell | row | run | model | setting | status | n | succ. % | Δ pt [95 % CI] | $/1k ep. | saving % "
                   "[95 % CI] | cov. % | acted % | acc. % | verdict | fresh: n, Δ, saving, verdict |")
        out.append("|" + "---|" * 16)
        for dp in DPS:
            for mode in (*MODES, *REF_MODES):
                for r in sorted([r for r in rs if r["cells"][0] == [dp, mode]], key=_row_order):
                    out.append(_md_row(r, dp, mode))
        out.append("")
    return "\n".join(out)


def _row_order(r: Mapping[str, Any]) -> tuple[Any, ...]:
    fam = FAMILY_ORDER.index(r["family"]) if r["family"] in FAMILY_ORDER else 99
    s = r["setting"].get(r["cells"][0][0]) or {}
    return (fam, float(s.get("tau") or 0), int(s.get("k") or 0), r["arm"], 0 if r["canonical"] else 1)


def _md_row(r: Mapping[str, Any], dp: str, mode: str) -> str:
    c, pm = r["cmp"], r["points"][dp]
    regs = ", ".join(f"{x['label']}{' (reference)' if x.get('role') == 'reference' else ''}: {x['word']}"
                     for x in r["registered"])
    status = f"registered {regs}" if regs else ("replicate of " + r["replicate_of"] if not r["canonical"] else "descriptive")
    ci = c.get("delta_ci") or [NAN, NAN]
    sci = c.get("saving_ci") or [NAN, NAN]
    fr = r.get("fresh")
    ftxt = "--"
    if fr:
        fc = fr["cmp"]
        ftxt = f"{fc['n']}, {_pt(fc.get('delta'))}, {_pt(fc.get('saving'), False)}, {fr['cell_verdicts'].get(f'{dp}/{mode}')}"
    cells = ", ".join(f"{a}/{b}" for a, b in r["cells"]) + (" (joint)" if r["joint"] else "")
    v = r["cell_verdicts"].get(f"{dp}/{mode}") or "--"
    extra = ""
    if r.get("vs_flash"):
        vf = r["vs_flash"]
        extra = f"; vs Flash alone Δ {_pt(vf.get('delta'))} pt, saving {_pt(vf.get('saving'), False)} %"
    return (f"| {cells} | `{r['id']}` | {r['run']} | {r['family_name']} | {_setting_text(r)} | {status} | {c.get('n')} | "
            f"{_pt(c.get('success'), False)} | {_pt(c.get('delta'))} [{_pt(ci[0])}, {_pt(ci[1])}] | "
            f"{1000 * c.get('cost', NAN):.1f} | {_pt(c.get('saving'), False)} [{_pt(sci[0], False)}, {_pt(sci[1], False)}] | "
            f"{_pt(pm.get('coverage'), False)} | {_pt(pm.get('acted'), False)} ({pm.get('acted_kind') or '-'}) | "
            f"{_pt(pm.get('acc'), False)} | {v}{extra} | {ftxt} |")


# ---------------------------------------------------------------- TeX keys
def _tex_text(s: str) -> str:
    return str(s).replace("\\", "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _seg(k: Any) -> str:
    return re.sub(r"[{}\\#%$&~^ ]", "", str(k).replace("/", "-").replace("@", "at"))


def _num_keys(prefix: str, c: Mapping[str, Any], pm: Mapping[str, Any] | None) -> dict[str, str]:
    v: dict[str, str] = {}

    def put(k: str, x: Any, fmt: str) -> None:
        if _finite(x):
            v[f"{prefix}/{k}"] = fmt.format(x)
    put("n", c.get("n"), "{:.0f}")
    put("n_dropped", c.get("n_dropped"), "{:.0f}")
    put("success", 100 * c["success"] if _finite(c.get("success")) else NAN, "{:.1f}")
    put("success_ref", 100 * c["success_ref"] if _finite(c.get("success_ref")) else NAN, "{:.1f}")
    for k, sgn in (("delta", True), ("saving", False)):
        x = c.get(k)
        if _finite(x):
            v[f"{prefix}/{k}"] = f"{100 * x:+.1f}" if sgn else f"{100 * x:.1f}"
            v[f"{prefix}/fig_{k}"] = f"{100 * x:+.1f}" if sgn else f"{round(100 * x):.0f}"
        ci = c.get(f"{k}_ci") or [NAN, NAN]
        for side, val in zip(("lo", "hi"), ci):
            if _finite(val):
                v[f"{prefix}/{k}_ci/{side}"] = f"{100 * val:+.1f}" if sgn else f"{100 * val:.1f}"
    put("cost_k", 1000 * c["cost"] if _finite(c.get("cost")) else NAN, "{:.1f}")
    put("cost_ref_k", 1000 * c["cost_ref"] if _finite(c.get("cost_ref")) else NAN, "{:.1f}")
    if pm is not None:
        for k in ("coverage", "acted", "acc", "s1_acc", "accepted_acc", "recall_k"):
            x = pm.get(k)
            if _finite(x):
                v[f"{prefix}/{k}"] = f"{100 * x:.1f}"
        if pm.get("acted_kind"):
            v[f"{prefix}/acted_kind"] = pm["acted_kind"]
    return v


def to_tex(res: Mapping[str, Any]) -> str:
    vals: dict[str, str] = {}
    for r in res["rows"]:
        for dp, mode in r["cells"]:
            p = f"mm/{dp}/{mode}/{_seg(r['id'])}"
            vals.update(_num_keys(p, r["cmp"], r["points"][dp]))
            v = r["cell_verdicts"].get(f"{dp}/{mode}")
            if v:
                vals[f"{p}/verdict"] = v
                vals[f"{p}/vcode"] = str(VCODE[v])
            regs = [x for x in r["registered"] if x.get("role") != "reference"]
            vals[f"{p}/status"] = ("registered" if regs else "replicate" if not r["canonical"] else "descriptive")
            if regs:
                vals[f"{p}/reg"] = ", ".join(x["label"] for x in regs)
            if r.get("fresh"):   # point estimates only: Stage 2's own intervals are scripts/stage2_analysis.py's
                f = r["fresh"]
                vals.update({k: x for k, x in _num_keys(f"{p}/fresh", f["cmp"], f["points"].get(dp)).items()
                             if "_ci/" not in k})
                fv = f["cell_verdicts"].get(f"{dp}/{mode}")
                if fv:
                    vals[f"{p}/fresh/verdict"] = fv
            if r.get("vs_flash"):
                vals.update({k: x for k, x in _num_keys(f"{p}/flash", r["vs_flash"], None).items()
                             if k.rsplit("/", 1)[-1] in ("delta", "saving", "fig_delta", "fig_saving", "n")
                             or "/delta_ci/" in k or "/saving_ci/" in k})
    for ck, c in res["cells"].items():
        p = f"mm/cell/{ck}"
        vals[f"{p}/state"] = c["state"]
        vals[f"{p}/vcode"] = str(c["vcode"])
        vals[f"{p}/verdict"] = c["verdict"] or "?"
        if c["state"] in ("undefined", "equivalent"):
            vals[f"{p}/undef"] = c["state"]
        if c.get("equiv"):           # an equivalent cell, or an implemented one with the same rule (E6e replace)
            vals[f"{p}/equiv"] = c["equiv"]
        if c["state"] != "run":
            continue
        vals[f"{p}/row"] = _tex_text(c["row"])
        vals.update(_num_keys(p, {"delta": c["delta"], "saving": c["saving"], "n": c["n"]},
                              {"acted": c["acted"], "coverage": c["coverage"]}))
        if c.get("registered"):
            vals[f"{p}/reg"] = c["registered"]
        if c.get("alt_vcode") is not None:
            vals[f"{p}/alt_vcode"] = str(c["alt_vcode"])
            vals[f"{p}/alt_verdict"] = c["alt_verdict"]
        vals[f"{p}/joint"] = "1" if c["joint"] else "0"
    for mk, m in res["modules"].items():
        p = f"mm/best/{mk}"
        vals[f"{p}/verdict"] = m["verdict"]
        vals[f"{p}/vcode"] = str(m["vcode"])
        if m.get("mode"):
            vals[f"{p}/mode"] = m["mode"]
            vals[f"{p}/row"] = _tex_text(m["row"])
            vals.update(_num_keys(p, {"delta": m["delta"], "saving": m["saving"]}, {"acted": m["acted"]}))
            if m.get("registered"):
                vals[f"{p}/reg"] = m["registered"]
    for llm, cnt in res["counts"].items():
        for k, x in cnt.items():
            vals[f"mm/count/{llm}/{k}"] = str(x)
    vals.update(explore_action_tex(res))
    vals.update(explore_e6b_tex(res))
    meta = res["meta"]
    lines = ["% AUTO-GENERATED by scripts/module_map.py -- do not edit by hand (descriptive module map; registered",
             "% outcomes are read from generated/primaries.tex, never from here).",
             f"% generated {meta['generated']} at git {meta['git'][:12]}; B = {meta['n_boot']}, seed {meta['seed']}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(vals.items())]
    lines += [r"\makeatother", ""]
    return "\n".join(lines)


# ====================================================================================== main
def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Mapping):
        return {k: _clean(v) for k, v in x.items() if k != "identity"}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating,)):
        return _clean(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "module_map"))
    ap.add_argument("--tex", default=str(ROOT / "paper" / "generated" / "module_map.tex"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    mm = ModuleMap(Path(a.runs), a.n_boot, a.seed, stage2=None if a.no_stage2 else STAGE2, quiet=a.quiet)
    res = mm.build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "module_map.json").write_text(json.dumps(_clean(res), indent=1, default=_json_default) + "\n")
    (out / "module_map.md").write_text(to_markdown(res) + "\n")
    Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tex).write_text(to_tex(res))
    print(grid_text(res))
    for f in res["meta"]["flags"]:
        print(f, file=sys.stderr)
    print(f"wrote {_rel(out / 'module_map.json')}, {_rel(out / 'module_map.md')}, {_rel(a.tex)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
