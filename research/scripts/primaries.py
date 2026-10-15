#!/usr/bin/env python3
"""The ONE family-level analysis of the 11 registered primaries (docs/PREREG_EFFECTIVE.md §1; PREREGISTRATION A5.3:
C1F C1L C2F C2L C3 C4 C5 C6F C6L C7 C8 in one Holm family at FWER 0.05). Every number it prints traces to the run
files it names (path, sha256 and record counts in the provenance block); nothing is typed by hand.

    # before any tau^2 run (A5.4 (a)): the C7 configuration, selected on E2 only
    python scripts/primaries.py --runs runs --select-c7 --out results/primaries
    # the analysis (after every main run; --strict exits 1 on any missing input / incomplete primary)
    python scripts/primaries.py --runs runs --c7-arm <tau^2 arm of C7> --out results/primaries --strict

Run directories (default ``<runs>/<experiment name of the registered YAML>``; ``--run CFG=DIR`` overrides one):

    E2a   configs/experiments/jevgym_tool_select.yaml        C1F C1L C2F C2L, C3's arm, the C7 selection
    E2a'  configs/experiments/jevgym_tool_select_intern.yaml Intern-Decision constituents of C1 / C2 / C6
    E2b   configs/experiments/jevgym_meta_control.yaml       C4, C6F, C6L; C4's opportunity count and C5's ceiling
                                                             (measurement arm `meta_shadow_s2lite`)
    E2c   configs/experiments/baselines_jevgym.yaml          C3's control (`bm25_ts_assist_k3`), C5
    tau2  ``--tau2`` (default <runs>/tau2): <domain>/<arm>.jsonl from scripts/run_tau2.sh   C7
    E1    ``--e1`` (default <runs>/e1_bfcl_when2call): s1_<backend>.jsonl                  C8

The tests, exactly as registered (PREREG_EFFECTIVE.md §1 and the shared rules under it):

* **NI** (C1F C1L C2F C2L C6F C6L C7 C8): ``metrics.ni_pvalue(...)["p_decisive"]`` on the paired per-task binary
  outcomes -- the certified Berger-Boos p-value of one ordering of all (wins, losses) outcomes, the monotone hull of
  the A6.14 switching rule's nominal p (``rule = bb_ordered``, A6.21 (a); ``metrics.ni_certified_p``). Margin 3 pt (JevGym, E1), 5 pt for C7 on **majority pass** (>= 2 of the 3 tau^2 trials,
  A6.14 / A6.16 (e)) over the 149 tested tasks (35 airline test + 114 retail; the 15 airline shadow tasks never
  enter, A5.4 (c)). C8: item-weighted accuracy over the 1,114 BFCL tool_select items of
  ``configs/eval/e1_c8_population.json`` (A6.9, A6.16 (l), A6.18 (d)).
* **Superiority**: C3 / C4 by the exact conditional sign test on the discordant tasks (``metrics.sign_test_exact``,
  ``p_wins``); C5 by the one-sided paired t-test of the mean paired Delta-U at W = $1, lambda = 0, **gross** of
  System-1's price (``metrics.paired_t_test``, ``p_greater``; A6.21 (b): the sign-flip p of A6.16 (g) is descriptive,
  net alongside). C4 enters with p = 1
  (*insufficient observed opportunities*) when ``meta_shadow_s2lite`` holds fewer than 30 labelled routing positives
  (A5.5, A6.3); the count is printed before C4 is scored.
* **Two System-1 families** (C1, C2, C6): p = max(p_Jev, p_Intern) (A5.1, A6.1) while the competence gate
  (``results/gpu_round2/gate_verdict.json``) says Intern-Decision is co-primary; labels *cross-backend replication*
  (both pass; "... of policy NI" when a constituent's delegation is negligible, A6.16 (m), A6.20 (e)), *split* (one
  passes), *Jev-specific confirmation* (the gate demoted Intern-Decision and Jev passes) (A6.9 #6). A C1 constituent's
  delegation is read from the two coverages of A6.22 (a) 5 (``scorecard.call_coverage``: tool-decision coverage and
  argument coverage) with A6.20 (e)'s 5 % rule: *zero* iff both are 0, *negligible* iff both are below 5 % (A6.22 (a) 7;
  the pooled delegation ratio is printed as a descriptive figure only).
* **Drop rule** (§1, A6.9): paired complete case -- a task whose latest record in either arm is an infrastructure
  error (after the runner's two extra attempts), or that is missing, leaves that comparison; per-arm counts and n' are
  printed; a *full-set* sensitivity row counts every such task as a failure in the arm where it errored. Errored
  episodes with fewer than 3 attempts are flagged as *retries outstanding* (re-run the config to retry them). Next to
  every full-set row the dropped tasks are listed (A6.22 (e) 5): task id, suite family (JevGym ``clarify`` / ``multi``;
  tau^2 domain and split), the arm(s) in which they errored, the error class (exception type and first line; tau^2
  termination reason), the attempts made and the other arm's outcome. tau^2 attempt counts come from the re-run ledger
  of ``suites.tau2 rerun-errored`` (A6.22 (e) 4), stamped into ``meta.tau2.attempts`` by the converter; with them an
  errored simulation with attempts left is *retries outstanding* as on JevGym.
* **Holm** over the 11 decisive p-values (``metrics.ni_holm_levels``); every primary stays in the family (a missing
  input enters with p = 1 and an ERROR flag). Reading rule (A6.1): a failed NI test whose two-sided 95 % task-bootstrap
  CI covers 0 is *inconclusive at this n*; a CI entirely below 0 is a loss of the stated size. C7 failures read
  "transfer NI was not established" (A6.16 (e)). The MDE at the achieved discordance and the primary's Holm level is
  printed next to every result (``metrics.ni_mde``; C5: normal approximation on the per-task Delta-U sd).
* **C7 selection** (A5.4, A6.9 #11, A6.18 (c)): among LLM-only, the Flash Jev cascade at each swept tau and at tau-hat*,
  assist k = 3 and the guard, the most cost-saving configuration that is NI on E2 at the unadjusted level
  (``ni_pvalue(...)["ni_95"]``); configurations within 2 pt of the best savings tie, and ties and the no-NI case go to
  the C1 arm.

Outputs in ``--out``: ``primaries.json`` (everything, with provenance), ``primaries.md`` and ``primaries.tex``
(``\\defres{primaries/<label>/<metric>}{...}``, the paper's ``\\res{}`` mechanism). Simulated inputs are refused
unless ``--allow-simulated``; then every LaTeX key carries the ``SIM/`` prefix and every heading says SIMULATED."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import prereg_check as PC  # noqa: E402  (registered sizes, seed constants, configs)
from prereg_effective import PRIMARIES as REGISTERED  # noqa: E402  (labels, arms, references, margins: one registry)
from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.runner import arm_task_counts  # noqa: E402
from jevharness.bench.scorecard import SIM_LLM_PROVIDERS, SIM_S1_TYPES, call_coverage, task_key  # noqa: E402
from jevharness.harness.trajectory import iter_records  # noqa: E402

NAN = float("nan")
ALPHA = 0.05                                   # family-wise (A5.3)
MAX_ATTEMPTS = 3                               # first attempt + 2 extra attempts (§1)
C4_MIN_POSITIVES = 30                          # A5.5 / A6.3
C5_W, C5_LAM = 1.0, 0.0                        # A6.5, A6.9: W = $1 per success, lambda = 0
C7_TRIALS, C7_MAJORITY = 3, 2                  # A6.14 / A6.16 (e): majority pass = >= 2 of 3
C7_TIE = 0.02                                  # A5.4 (b): within 2 pt of cost savings of the best
NEGLIGIBLE_COVERAGE = 0.05                     # A6.20 (e) / A6.22 (a) 5: negligible delegation below 5 %
MDE_POWER = 0.8
HOLM_ORDER = [p["label"] for p in REGISTERED]
REG = {p["label"]: p for p in REGISTERED}
CONJ = ("C1F", "C1L", "C2F", "C2L", "C6F", "C6L")       # two System-1 families (A5.1)
NI_LABELS = ("C1F", "C1L", "C2F", "C2L", "C6F", "C6L", "C7", "C8")
GATE_FILE = ROOT / "results" / "gpu_round2" / "gate_verdict.json"
FROZEN_TASKS = ROOT / "configs" / "eval" / "jevgym_eval_tasks.json"
C8_POPULATION = ROOT / "configs" / "eval" / "e1_c8_population.json"
E1_CONFIG = ROOT / "configs" / "decision" / "bfcl_when2call.yaml"
AIRLINE_SPLIT = ROOT / "configs" / "tau2" / "airline_task_ids.json"
TAU2_RETAIL_N = 114                            # §1 / A6.9 #1
E1_ARM, E1_REF = "jev", "gemini_flash_typed"   # C8 (REGISTERED["C8"])
C4_MEASURE_ARM = "meta_shadow_s2lite"          # A5.5
C5_MEASURE_ARM = "meta_shadow_s2lite"          # C5's ceiling / no-headroom readout (A4)
# C7's eligible E2 configurations (A6.9 #11): E2a arm -> the tau^2 arm family that runs it. tau^2's `jev_guard` (policy
# guard before WRITE calls + stop guard) has no E2 twin; the E2a tool-call guard `jev_ts_guard` is read as its E2
# counterpart (an interpretation: no other guard of E2a / E2b gates the agent's tool calls).
C7_ELIGIBLE_FIXED = {"llm_only": "llm_only", "jev_ts_cascade_tauhat": "jev_ts_cascade (tau-hat* refit, A5.4 (c))",
                     "jev_ts_assist_k3": "jev_ts_assist_k3", "jev_ts_guard": "jev_guard"}
C7_SWEEP_PREFIX = "jev_ts_cascade_tau"
C7_C1_ARM = "jev_ts_cascade_tauhat"
CONFIGS = {"E2a": "jevgym_tool_select.yaml", "E2a_intern": "jevgym_tool_select_intern.yaml",
           "E2b": "jevgym_meta_control.yaml", "E2c": "baselines_jevgym.yaml"}


# ====================================================================================== small helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path | str) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class Flags:
    """ERROR (the primary or the family cannot be read as registered; --strict exits 1) and WARN entries."""

    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def _add(self, level: str, where: str, msg: str) -> None:
        item = {"level": level, "where": where, "msg": msg}
        if item not in self.items:          # the same arm is read by several primaries: report a finding once
            self.items.append(item)

    def error(self, where: str, msg: str) -> None:
        self._add("ERROR", where, msg)

    def warn(self, where: str, msg: str) -> None:
        self._add("WARN", where, msg)

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.items if f["level"] == "ERROR")


# ====================================================================================== loading episode runs
def _episode(r: Mapping[str, Any], attempts: int) -> dict[str, Any]:
    u = r.get("usage") or {}
    tot, s1, s2 = u.get("total") or {}, u.get("s1") or {}, u.get("s2") or {}
    tau = (r.get("meta") or {}).get("tau2") or {}
    trial = tau.get("trial")
    if trial is None and "#" in str(r.get("task_id")):
        try:
            trial = int(str(r.get("task_id")).rsplit("#", 1)[1])
        except ValueError:
            trial = None
    # tau^2 writes one record per simulation; its attempts come from the re-run ledger (A6.22 (e) 4), stamped by
    # ``suites.tau2.convert_results`` -- without the stamp the count is not logged (every error is then final)
    logged = not tau or tau.get("attempts") is not None
    if tau.get("attempts") is not None:
        attempts = int(tau["attempts"])
    return {"episode": str(r.get("task_id")), "task": task_key(r), "trial": trial,
            "success": float(bool((r.get("result") or {}).get("success"))),
            "cost": float(tot.get("cost_usd", 0.0) or 0.0), "s1_cost": float(s1.get("cost_usd", 0.0) or 0.0),
            "latency": float(tot.get("latency_ms", 0.0) or 0.0) / 1000.0,
            "llm_input_tokens": float(s2.get("input_tokens", 0) or 0), "error": r.get("error") or None,
            "attempts": attempts, "attempts_logged": logged, "llm": r.get("llm"), "backend": r.get("backend"),
            "termination_reason": (r.get("result") or {}).get("termination_reason")}


def load_arm(path: Path, keep_trajs: bool = False) -> dict[str, Any] | None:
    """Every record of ``<arm>.jsonl`` in file order: the latest record per episode id (``task_id``) decides, the number
    of records per id is the number of attempts (the runner appends a fresh record when it retries an errored task)."""
    if not path.exists():
        return None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recs = [r for r in iter_records(path) if "task_id" in r]
    attempts: Counter[str] = Counter()
    latest: dict[str, dict[str, Any]] = {}
    for r in recs:
        tid = str(r["task_id"])
        attempts[tid] += 1
        latest[tid] = r
    eps = {tid: _episode(r, attempts[tid]) for tid, r in latest.items()}
    out: dict[str, Any] = {"path": _rel(path), "sha256": _sha256(path), "n_records": len(recs),
                           "n_episodes": len(eps), "partial_lines": len(caught), "episodes": eps}
    meta = [(r.get("meta") or {}) for r in list(latest.values())[:5]]
    out["s1"] = next(((m.get("s1") or {}).get("describe") for m in meta if (m.get("s1") or {}).get("describe")), None)
    llms = Counter(e["llm"] for e in eps.values() if e["llm"])
    out["s2_model"] = llms.most_common(1)[0][0] if llms else None
    ok = [r for r in latest.values() if not r.get("error")]
    recs_ok = M.decision_records(ok)  # type: ignore[arg-type]
    # the pooled delegation ratio (descriptive) and the A6.22 (a) 5 coverages (tool-decision, argument) of the arm
    out["coverage"] = {**M.delegation_ratio(recs_ok), **call_coverage(ok, recs_ok)}
    if keep_trajs:
        out["trajs"] = list(latest.values())
    return out


def run_is_simulated(run_dir: Path, arms: Sequence[Mapping[str, Any]] = ()) -> bool:
    """A run is simulated when its experiment.yaml names a simulated LLM / System-1 (as ``scorecard.run_meta``) or a
    loaded episode's backend is a simulated type."""
    exp_path = run_dir / "experiment.yaml"
    exp: dict[str, Any] = {}
    if exp_path.exists():
        try:
            exp = yaml.safe_load(exp_path.read_text()) or {}
        except yaml.YAMLError:
            exp = {}
    llm_specs = [exp.get("llm") or {}, exp.get("small_llm") or {}, *((exp.get("llms") or {}).values()),
                 *((exp.get("fc_llms") or {}).values())]
    s1_specs = [exp.get("s1") or {}, *((exp.get("s1_backends") or {}).values()), *((exp.get("backends") or {}).values())]
    sim = any(str(s.get("provider", "")).lower() in SIM_LLM_PROVIDERS for s in llm_specs if isinstance(s, dict))
    sim = sim or any(str(s.get("type", "")).lower() in SIM_S1_TYPES for s in s1_specs if isinstance(s, dict))
    for a in arms:
        sim = sim or any(str(e.get("backend") or "").lower() in SIM_S1_TYPES for e in list(a["episodes"].values())[:5])
    return bool(sim)


class EpisodeRun:
    """One JevGym experiment directory: its experiment.yaml, the expected task list per arm and lazily loaded arms."""

    def __init__(self, key: str, cfg: str, run_dir: Path, flags: Flags, max_tasks: int | None,
                 allow_dev_tasks: bool) -> None:
        self.key, self.cfg, self.dir, self.flags = key, cfg, run_dir, flags
        self.max_tasks = max_tasks
        self.arms: dict[str, dict[str, Any] | None] = {}
        self.exp: dict[str, Any] = {}
        self.registered = PC.load(cfg)
        self.counts: dict[str, int] = {}
        self.task_ids: list[str] = []
        self.exists = run_dir.is_dir()
        if not self.exists:
            return
        p = run_dir / "experiment.yaml"
        if p.exists():
            self.exp = yaml.safe_load(p.read_text()) or {}
        else:
            flags.error(key, f"{_rel(run_dir)}: no experiment.yaml (the runner writes it): the task list cannot be checked")
        self.counts = arm_task_counts(self.exp) if self.exp else {}
        suite = dict(self.exp.get("suite") or {})
        frozen = json.loads(FROZEN_TASKS.read_text())
        self.frozen = suite == frozen["suite"]
        if self.frozen:
            self.task_ids = list(frozen["task_ids"])
        else:
            msg = (f"{_rel(run_dir)}: suite {suite} is not the frozen evaluation suite of {_rel(FROZEN_TASKS)} "
                   "(A6.13): the task list is generated from the run's own suite block")
            (flags.warn if allow_dev_tasks else flags.error)(key, msg)
            self.task_ids = self._suite_ids(suite)
        reg_suite = self.registered.get("suite")
        if self.exp and {k: v for k, v in suite.items() if k != "seed0"} != {k: v for k, v in (reg_suite or {}).items()
                                                                              if k != "seed0"}:
            flags.error(key, f"{_rel(run_dir)}: suite settings differ from {cfg} beyond seed0: {suite} vs {reg_suite}")

    @staticmethod
    def _suite_ids(suite: Mapping[str, Any]) -> list[str]:
        from jevharness.bench.suites import load_suite

        kw = dict(suite)
        name = kw.pop("name", "jevgym")
        return [f().task_id for f in load_suite(name, **kw)]

    def arm(self, name: str, keep_trajs: bool = False) -> dict[str, Any] | None:
        if name not in self.arms or (keep_trajs and self.arms[name] is not None and "trajs" not in self.arms[name]):
            self.arms[name] = load_arm(self.dir / f"{name}.jsonl", keep_trajs) if self.exists else None
        return self.arms[name]

    def n_tasks(self, arm: str) -> int:
        """Tasks the arm is registered to run: its n_tasks in the REGISTERED YAML (the A2 / A5 sizes; an arm-subset
        ``-o arms=[...]`` invocation rewrites the run's experiment.yaml, so that file is only cross-checked), capped by
        ``--max-tasks`` (dry runs)."""
        reg_counts = getattr(self, "_reg_counts", None)
        if reg_counts is None:
            reg_counts = self._reg_counts = arm_task_counts(self.registered)
        n = reg_counts.get(arm)
        if n is None:
            n = int(self.registered.get("n_tasks") or len(self.task_ids))
        run_n = self.counts.get(arm)
        if run_n is not None and run_n != n:
            self.flags.error(self.key, f"`{arm}` ran n_tasks = {run_n} in {_rel(self.dir)}/experiment.yaml, registered {n}")
        return min(n, self.max_tasks) if self.max_tasks else n

    def expected(self, *arms: str) -> list[str]:
        return self.task_ids[: min(self.n_tasks(a) for a in arms)]


# ====================================================================================== per-task outcomes and pairs
def task_table(arm: Mapping[str, Any] | None, expected: Sequence[str], trials: int = 1,
               retry_rule: bool = True) -> dict[str, dict[str, Any]]:
    """task -> {status, y (the tested binary outcome), pass1, cost, s1_cost, tokens}. status: ok | errored (latest record
    an infrastructure error after the registered attempts; with ``retry_rule`` False -- tau^2, E1 -- every error is
    final) | retry_pending (errored with attempts left) | missing | incomplete (fewer than ``trials`` trials).
    ``trials`` > 1 (tau^2): the first ``trials`` trials of the task; y = majority pass (>= ``C7_MAJORITY``)."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in (arm or {}).get("episodes", {}).values():
        by_task[e["task"]].append(e)
    out: dict[str, dict[str, Any]] = {}
    for t in expected:
        eps = by_task.get(t, [])
        if trials > 1:
            eps = sorted((e for e in eps if e["trial"] is None or int(e["trial"]) < trials),
                         key=lambda e: (e["trial"] is None, e["trial"]))[:trials]
        row: dict[str, Any] = {"n_trials": len(eps)}
        if not eps:
            row["status"] = "missing"
        elif any(e["error"] for e in eps):
            pending = retry_rule and any(e["error"] and e["attempts"] < MAX_ATTEMPTS for e in eps)
            row["status"] = "retry_pending" if pending else "errored"
        elif len(eps) < trials:
            row["status"] = "incomplete"
        else:
            row["status"] = "ok"
        errs = [e for e in eps if e["error"]]
        if errs:   # the dropped-task listing (A6.22 (e) 5): error class and attempts of the errored episode(s)
            row["error_class"] = error_class(errs[0])
            row["attempts"] = max(int(e.get("attempts") or 1) for e in errs)
            if trials > 1:
                row["errored_trials"] = sorted(int(e["trial"]) for e in errs if e["trial"] is not None)
        elif eps:
            row["attempts"] = max(int(e.get("attempts") or 1) for e in eps)
        ok_eps = [e for e in eps if not e["error"]]
        s = [e["success"] for e in eps]
        if trials > 1:
            row["y"] = float(sum(e["success"] for e in ok_eps) >= C7_MAJORITY) if row["status"] == "ok" else 0.0
        else:
            row["y"] = float(s[0]) if row["status"] == "ok" else 0.0
        row["pass1"] = float(np.mean([e["success"] for e in ok_eps])) if row["status"] == "ok" else 0.0
        for k, src in (("cost", "cost"), ("s1_cost", "s1_cost"), ("tokens", "llm_input_tokens"), ("latency", "latency")):
            row[k] = float(np.mean([e[src] for e in eps])) if eps else 0.0
        out[t] = row
    return out


def error_class(e: Mapping[str, Any]) -> str | None:
    """PREREGISTRATION A6.22 (e) 5: the error class of an errored episode -- the exception type and first line of
    its ``error`` (JevGym, E1: ``"<Type>: <message>"`` + traceback), on tau^2 the termination reason (plus the
    side-car exception of ``JevTau2Agent`` when there is one)."""
    lines = str(e.get("error") or "").strip().splitlines()
    first = lines[0].strip()[:200] if lines else ""
    term = e.get("termination_reason")
    if term not in (None, "", "None"):
        generic = first.startswith("tau2 termination_reason=")
        return f"termination_reason={term}" + (f"; {first}" if first and not generic else "")
    return first or None


def suite_family(task: str) -> str | None:
    """A6.22 (e) 5: the suite family of a task id -- JevGym ``clarify`` / ``multi`` (``jevgym-<family>-<seed>-...``),
    tau^2 ``<domain> (<split>)`` (the A5.4 airline shadow / test split; every task of another domain is a test task);
    None for other ids."""
    s = str(task)
    if s.startswith("jevgym-"):
        parts = s.split("-")
        return parts[1] if len(parts) > 1 else None
    if "/" in s:
        dom, tid = s.split("/", 1)
        try:
            from jevharness.bench.suites.tau2 import split_of  # noqa: PLC0415

            sp = split_of(dom, tid)
        except Exception:  # noqa: BLE001 - the listing is descriptive
            sp = None
        return f"{dom} ({sp or 'test'})"
    return None


def delegation_class(tool_cov: Any, arg_cov: Any, threshold: float = NEGLIGIBLE_COVERAGE) -> str:
    """A6.20 (e)'s 5 % rule on the two coverages of A6.22 (a) 5 (tool-decision coverage, argument coverage): ``zero``
    iff both are 0, ``negligible`` iff both are below ``threshold`` (5 %), else ``non-negligible``. A coverage without
    any opportunity (NaN: no tool-selection record, no executed argument slot) cannot exceed the threshold and does
    not count; ``unknown`` when neither is defined."""
    vals = [float(v) for v in (tool_cov, arg_cov) if _finite(v)]
    if not vals:
        return "unknown"
    if all(v == 0 for v in vals):
        return "zero"
    return "negligible" if all(v < threshold for v in vals) else "non-negligible"


def _outcome(row: Mapping[str, Any]) -> str:
    if row["status"] != "ok":
        return row["status"]
    return "success" if row["y"] >= 1.0 else "failure"


def pair_tables(ref: Mapping[str, Mapping[str, Any]], arm: Mapping[str, Mapping[str, Any]],
                expected: Sequence[str], names: tuple[str, str] = ("reference", "arm"),
                family: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    """Paired complete case + full-set view of two task tables over ``expected``. ``dropped`` lists every task that
    leaves the pair (A6.22 (e) 5): task id, suite family (``family``, default :func:`suite_family`), the side(s) in
    which it errored or is missing (``names`` = (reference, arm) names), their status, error class and attempts, and
    the other side's outcome."""
    keep = [t for t in expected if ref[t]["status"] == "ok" and arm[t]["status"] == "ok"]

    def counts(tab: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
        c = Counter(tab[t]["status"] for t in expected)
        return {k: int(c.get(k, 0)) for k in ("ok", "errored", "retry_pending", "missing", "incomplete")}

    fam = family or suite_family
    kept = set(keep)
    dropped = []
    for t in expected:
        if t in kept:
            continue
        sides = {names[0]: ref[t], names[1]: arm[t]}
        bad = [s for s, r in sides.items() if r["status"] != "ok"]
        d: dict[str, Any] = {"task": t, "family": fam(t), "errored_in": bad,
                             "status": {s: sides[s]["status"] for s in bad},
                             "error_class": {s: sides[s].get("error_class") for s in bad},
                             "attempts": {s: sides[s].get("attempts") for s in bad},
                             "other_outcome": {s: _outcome(r) for s, r in sides.items() if s not in bad}}
        trials = {s: sides[s]["errored_trials"] for s in bad if sides[s].get("errored_trials")}
        if trials:
            d["errored_trials"] = trials
        dropped.append(d)
    return {"n_expected": len(expected), "n_pairs": len(keep), "n_dropped": len(expected) - len(keep),
            "status_ref": counts(ref), "status_arm": counts(arm), "keys": keep, "dropped": dropped,
            "ref_y": {t: ref[t]["y"] for t in keep}, "arm_y": {t: arm[t]["y"] for t in keep},
            "ref_y_full": {t: ref[t]["y"] for t in expected}, "arm_y_full": {t: arm[t]["y"] for t in expected}}


def _status_flags(flags: Flags, where: str, side: str, name: str, st: Mapping[str, int], n_exp: int,
                  attempts_logged: bool = True) -> None:
    if st.get("missing"):
        flags.error(where, f"{side} `{name}`: {st['missing']} of {n_exp} expected task(s) have no record "
                           "(incomplete run: they leave the pair, full-set row counts them as failures)")
    if st.get("retry_pending"):
        flags.error(where, f"{side} `{name}`: {st['retry_pending']} task(s) ended in an infrastructure error with fewer "
                           f"than {MAX_ATTEMPTS} attempts (retries outstanding: re-run the config)")
    if st.get("incomplete"):
        flags.error(where, f"{side} `{name}`: {st['incomplete']} task(s) have fewer than {C7_TRIALS} trials")
    if st.get("errored"):
        how = ("errored after the registered attempts" if attempts_logged else
               "errored (attempt counts are not logged for this suite: re-run the errored episodes before the "
               "analysis, then they are final)")
        flags.warn(where, f"{side} `{name}`: {st['errored']} task(s) {how}: dropped from the pair (drop rule, §1 / A6.9)")


# ====================================================================================== the tests
def ni_block(ref_y: Mapping[str, float], arm_y: Mapping[str, float], margin: float, n_boot: int, seed: int,
             alpha_adj: float | None = None) -> dict[str, Any]:
    """``metrics.ni_pvalue`` (the A6.21 rule, ``bb_ordered``) on binary per-task outcomes, trimmed for the report."""
    r = M.ni_pvalue(ref_y, arm_y, margin, n_boot=n_boot, seed=seed, alpha=ALPHA, alpha_adj=alpha_adj)
    n = r["n"]
    ex = r.get("exact") or {}
    out = {"test": "NI", "margin": margin, "n": n, "wins": r["wins"], "losses": r["losses"],
           "discordance": (r["n_discordant"] / n) if n else NAN, "diff": r["diff"], "ci95": r["ci95"],
           "p_decisive": r["p_decisive"] if n else NAN, "rule": r.get("rule") if n else None,
           "p_tango_asymptotic": r.get("p_tango", NAN), "p_uncond": r.get("p_uncond", NAN),
           "tango_lcb_adj": r.get("tango_lcb_adj", NAN), "p_boot_descriptive": r.get("p_ni", NAN),
           "lcb_boot_adj_descriptive": r.get("lcb_adj", NAN), "ni_95": bool(r.get("ni_95")),
           "loss_rate_ub": ex.get("loss_rate_ub", NAN) if ex else NAN,
           "mde": (r.get("mde") or {}).get("delta_mde", NAN) if n else NAN,
           "margin_mde": (r.get("mde") or {}).get("margin_mde", NAN) if n else NAN,
           "mde_note": _mde_note(n, r["n_discordant"]) if n else "no paired task",
           "success_ref": float(np.mean(list(ref_y.values()))) if n else NAN,
           "success_arm": float(np.mean([arm_y[k] for k in ref_y if k in arm_y])) if n else NAN}
    if n:
        cells = M.ni_cells(ref_y, arm_y)
        out["newcombe95"] = M.newcombe_interval(**cells)["ci"]
    return out


def _mde_note(n: int, n_disc: int) -> str | None:
    """Why an MDE is undefined (``metrics.ni_mde`` returns NaN): no discordant task (the exact rule's loss-rate upper
    bound is the reported bound, A6.1), or 80 % power is out of reach even if every discordant task were a win."""
    if n_disc == 0:
        return "no discordant task: the exact loss-rate upper bound is reported instead (A6.1)"
    return None


def sign_block(ref_y: Mapping[str, float], arm_y: Mapping[str, float], n_boot: int, seed: int) -> dict[str, Any]:
    """Superiority on a binary endpoint (C3, C4): exact conditional sign test on the discordant tasks (A6.16 (g));
    the paired task bootstrap is descriptive."""
    keys = sorted(set(ref_y) & set(arm_y))
    wins = sum(1 for k in keys if arm_y[k] > ref_y[k])
    losses = sum(1 for k in keys if arm_y[k] < ref_y[k])
    st = M.sign_test_exact(wins, losses)
    bt = M.paired_bootstrap_diff({k: ref_y[k] for k in keys}, {k: arm_y[k] for k in keys}, n=n_boot, seed=seed)
    n = len(keys)
    return {"test": "Sup.", "n": n, "wins": wins, "losses": losses, "discordance": (wins + losses) / n if n else NAN,
            "diff": bt["diff"], "ci95": [bt["ci_lo"], bt["ci_hi"]], "p_decisive": st["p_wins"] if n else NAN,
            "rule": "sign_exact", "p_sup_boot_descriptive": bt["p_better"],
            "success_ref": float(np.mean([ref_y[k] for k in keys])) if n else NAN,
            "success_arm": float(np.mean([arm_y[k] for k in keys])) if n else NAN}


def du_block(d_gross: Mapping[str, float], d_net: Mapping[str, float], n_boot: int, seed: int) -> dict[str, Any]:
    """C5: one-sided paired t-test of the mean paired Delta-U, gross of System-1's price (decisive, A6.21 (b)); the
    sign-flip p (A6.16 (g)) and the net figures are descriptive."""
    keys = sorted(d_gross)
    g = [d_gross[k] for k in keys]
    tt = M.paired_t_test(g)
    sf = M.sign_flip_test(g, n_perm=20000, seed=0)
    sfn = M.sign_flip_test([d_net[k] for k in keys], n_perm=20000, seed=0)
    zero = {k: 0.0 for k in keys}
    bg = M.paired_bootstrap_diff(zero, d_gross, n=n_boot, seed=seed)
    bn = M.paired_bootstrap_diff(zero, d_net, n=n_boot, seed=seed)
    n = len(keys)
    return {"test": "Sup. (ΔU)", "n": n, "W": C5_W, "lam_per_hour": C5_LAM, "diff": float(np.mean(g)) if n else NAN,
            "ci95": [bg["ci_lo"], bg["ci_hi"]], "p_decisive": tt["p_greater"] if n else NAN,
            "rule": "paired_t", "t": tt.get("t"), "p_sign_flip_descriptive": sf["p_greater"] if n else NAN,
            "sign_flip_method": sf.get("method"), "n_perm": sf.get("n_perm"),
            "diff_net": float(np.mean([d_net[k] for k in keys])) if n else NAN, "ci95_net": [bn["ci_lo"], bn["ci_hi"]],
            "p_net_descriptive": sfn["p_greater"] if n else NAN,
            "sd": float(np.std(g, ddof=1)) if n > 1 else NAN,
            "n_nonzero": int(sum(1 for x in g if abs(x) > 1e-12))}


def mde_du(sd: float, n: int, alpha: float, power: float = MDE_POWER) -> float:
    """C5 MDE: the mean Delta-U the one-sided test detects with ``power`` at level ``alpha`` (normal approximation)."""
    if not (n and _finite(sd)) or sd <= 0:
        return NAN
    return (M._z(1 - alpha) + M._z(power)) * sd / math.sqrt(n)


def cost_block(ref: Mapping[str, Mapping[str, Any]], arm: Mapping[str, Mapping[str, Any]], keys: Sequence[str],
               n_boot: int, seed: int) -> dict[str, Any]:
    """Descriptive readouts on the paired tasks: cost per episode and savings 1 - C_arm / C_ref with a task-bootstrap CI
    (C1's readout), System-2 input tokens and their savings (C2's readout)."""
    if not keys:
        return {}
    ca = np.array([arm[k]["cost"] for k in keys])
    cb = np.array([ref[k]["cost"] for k in keys])
    ta = np.array([arm[k]["tokens"] for k in keys])
    tb = np.array([ref[k]["tokens"] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = 1.0 - ca[idx].mean(1) / cb[idx].mean(1)
    boot = boot[np.isfinite(boot)]
    sav = 1.0 - ca.mean() / cb.mean() if cb.mean() > 0 else NAN
    tsav = 1.0 - ta.mean() / tb.mean() if tb.mean() > 0 else NAN
    return {"cost_arm": float(ca.mean()), "cost_ref": float(cb.mean()), "cost_savings": float(sav),
            "cost_savings_ci": ([float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot)
                                else [NAN, NAN]),
            "tokens_arm": float(ta.mean()), "tokens_ref": float(tb.mean()), "token_savings": float(tsav)}


def choose_c7(cands: Mapping[str, Mapping[str, Any]], c1_arm: str = C7_C1_ARM,
              tie: float = C7_TIE) -> tuple[str, str, bool]:
    """A5.4 (b) / A6.9 #11 / A6.18 (c): (selected E2 arm, reason, degenerate). Candidates: E2 arm -> {ni_95 (the
    registered NI statistic at the unadjusted level), cost_savings}; ``llm_only`` is eligible with savings 0. The most
    cost-saving NI configuration wins; configurations within ``tie`` (2 pt) of the best savings tie, and ties and the
    case of no NI System-1 configuration resolve to the C1 arm. If LLM-only itself is the unique best (every NI System-1
    configuration costs more than 2 pt more), the comparison would be LLM-only against itself: resolved to the C1 arm
    and returned as ``degenerate`` (an interpretation, flagged)."""
    ni = {k: v for k, v in cands.items() if v.get("ni_95") and _finite(v.get("cost_savings"))}
    if not [k for k in ni if k != "llm_only"]:
        sel, reason, deg = c1_arm, "no System-1 configuration is NI on E2 -> the C1 arm (A5.4 (b))", False
    else:
        best = max(ni, key=lambda k: (ni[k]["cost_savings"], k == c1_arm))
        ties = sorted(k for k, v in ni.items() if v["cost_savings"] >= ni[best]["cost_savings"] - tie)
        if len(ties) > 1:
            sel, deg = c1_arm, False
            reason = (f"{len(ties)} configurations within {100 * tie:.0f} pt of the best savings ({', '.join(ties)}) "
                      "-> the C1 arm (A5.4 (b))")
        elif best == "llm_only":
            sel, deg = c1_arm, True
            reason = ("LLM-only is the most cost-saving NI configuration (no NI System-1 configuration within 2 pt): "
                      "degenerate, resolved to the C1 arm (interpretation of A5.4 (b))")
        else:
            sel, deg = best, False
            reason = f"most cost-saving NI configuration ({100 * ni[best]['cost_savings']:.1f} % savings)"
    if sel == c1_arm and c1_arm in cands and not cands[c1_arm].get("ni_95"):
        reason += "; the C1 arm itself is not NI on E2: C7 runs it anyway and A5.4 (d)'s fallback rows apply"
    return sel, reason, deg


def exact_size_settings(prim: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A6.17 (b): the (n', margin) of every NI constituent, and those missing from ``SETTINGS`` of
    ``scripts/ni_exact_size.py`` (add them and re-run the exact-size table before the analysis is reported)."""
    need: dict[tuple[int, float], list[str]] = {}
    for lab in NI_LABELS:
        for c in ("jev", "intern"):
            b = (prim.get(lab) or {}).get(c)
            if isinstance(b, dict) and b.get("n_pairs"):
                need.setdefault((int(b["n_pairs"]), float(b["margin"])), []).append(f"{lab}/{c}")
    try:
        import ni_exact_size  # noqa: PLC0415  (scripts/; the registry of the exact-size table)
        have = {(int(n), float(m)) for n, m, *_ in ni_exact_size.SETTINGS}
    except Exception:  # noqa: BLE001
        have = set()
    missing = sorted(k for k in need if k not in have)
    return {"needed": [{"n": n, "margin": m, "for": need[(n, m)]} for n, m in sorted(need)],
            "missing_from_SETTINGS": [f"({n}, {m}, \"final n' of {', '.join(need[(n, m)])}\")" for n, m in missing]}


def holm(pvals: Mapping[str, float], alpha: float = ALPHA) -> dict[str, dict[str, Any]]:
    """``metrics.ni_holm_levels`` plus whether Holm's step-down reached the test (every smaller p rejected)."""
    lv = M.ni_holm_levels(pvals, alpha)
    order = sorted(pvals, key=lambda k: (lv[k]["rank"] if _finite(lv[k]["rank"]) else 1e9))   # Holm's own order
    reached, out = True, {}
    for k in order:
        out[k] = {**lv[k], "reached": reached}
        reached = reached and bool(lv[k]["reject"])
    return out


def reading(blk: Mapping[str, Any], rejected: bool) -> str:
    """The A6.1 reading rule for an NI test."""
    if rejected:
        return "NI"
    lo, hi = (blk.get("ci95") or [NAN, NAN])
    if not (_finite(lo) and _finite(hi)):
        return "not tested (no paired tasks)"
    if lo <= 0 <= hi:
        return "inconclusive at this n"
    if hi < 0:
        return f"not NI: loss of {-100 * blk['diff']:.1f} pt (95 % CI entirely below 0)"
    return "not NI at its Holm step (CI above 0)"


# ====================================================================================== the analysis
class Analysis:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.flags = Flags()
        self.runs_root = Path(a.runs)
        dirs = {k: self.runs_root / PC.load(cfg)["name"] for k, cfg in CONFIGS.items()}
        for kv in a.run or []:
            k, v = kv.split("=", 1)
            key = next((x for x, c in CONFIGS.items() if c == k or x == k), None)
            if key is None:
                raise SystemExit(f"--run {kv}: unknown config (one of {sorted(CONFIGS.values())})")
            dirs[key] = Path(v)
        self.runs = {k: EpisodeRun(k, CONFIGS[k], d, self.flags, a.max_tasks, a.allow_dev_tasks)
                     for k, d in dirs.items()}
        self.tau2_dir = Path(a.tau2) if a.tau2 else self.runs_root / "tau2"
        e1_name = yaml.safe_load(E1_CONFIG.read_text())["name"]
        self.e1_dir = Path(a.e1) if a.e1 else self.runs_root / e1_name
        self.simulated: dict[str, bool] = {}
        self.provenance: dict[str, Any] = {}

    # -------------------------------------------------------------------------- provenance / simulated guard
    def _note_arm(self, run: str, name: str, arm: Mapping[str, Any] | None, run_dir: Path) -> None:
        if arm is None:
            return
        self.provenance.setdefault(run, {"dir": _rel(run_dir), "arms": {}})["arms"][name] = {
            k: arm.get(k) for k in ("path", "sha256", "n_records", "n_episodes", "partial_lines", "s2_model", "s1")}
        if arm.get("partial_lines"):
            self.flags.warn(run, f"`{name}`: {arm['partial_lines']} undecodable line(s) skipped (partial write)")
        if run not in self.simulated:
            self.simulated[run] = run_is_simulated(run_dir, [arm])

    def _guard_simulated(self) -> None:
        sim = sorted(k for k, v in self.simulated.items() if v)
        if sim and not self.a.allow_simulated:
            self.flags.error("inputs", f"simulated run(s) {sim}: never headline numbers; "
                                       "pass --allow-simulated for a dry run")

    # -------------------------------------------------------------------------- one JevGym pair
    def jevgym_pair(self, where: str, run_key: str, arm: str, ref: str, ref_run_key: str | None = None,
                    keep_trajs: bool = False) -> dict[str, Any] | None:
        R, RR = self.runs[run_key], self.runs[ref_run_key or run_key]
        if not R.exists or not RR.exists:
            self.flags.error(where, f"run directory missing: {_rel(R.dir) if not R.exists else _rel(RR.dir)}")
            return None
        A, B = R.arm(arm, keep_trajs), RR.arm(ref)
        for nm, x, rr in ((arm, A, R), (ref, B, RR)):
            if x is None:
                self.flags.error(where, f"arm `{nm}` not found in {_rel(rr.dir)}")
            else:
                self._note_arm(rr.key, nm, x, rr.dir)
        if A is None or B is None:
            return None
        for nm, x, rr in ((arm, A, R), (ref, B, RR)):
            foreign = sorted({e["task"] for e in x["episodes"].values()} - set(rr.task_ids))
            if foreign:   # e.g. a dev-seed smoke run written into the main run directory
                self.flags.error(where, f"`{nm}` in {_rel(rr.dir)} holds {len(foreign)} episode task(s) outside the "
                                        f"run's task list (e.g. {foreign[0]}): another suite's episodes share the file; "
                                        "score cards and paper numbers would mix them (move them out)")
        n = min(R.n_tasks(arm), RR.n_tasks(ref))
        exp_ids = R.task_ids[:n]
        if RR.task_ids[:n] != exp_ids:
            self.flags.error(where, f"{_rel(R.dir)} and {_rel(RR.dir)} run different task lists: no pairing (A6.15)")
            return None
        ta, tb = task_table(A, exp_ids), task_table(B, exp_ids)
        p = pair_tables(tb, ta, exp_ids, names=(ref, arm))
        _status_flags(self.flags, where, "arm", arm, p["status_arm"], n)
        _status_flags(self.flags, where, "reference", ref, p["status_ref"], n)
        p.update({"arm": arm, "ref": ref, "run": _rel(R.dir), "ref_run": _rel(RR.dir), "tab_arm": ta, "tab_ref": tb,
                  "s2_arm": A.get("s2_model"), "s2_ref": B.get("s2_model"), "coverage": A.get("coverage")})
        return p

    def ni_constituent(self, where: str, p: Mapping[str, Any] | None, margin: float) -> dict[str, Any]:
        if p is None or not p["n_pairs"]:
            if p is not None:
                self.flags.error(where, "no paired task without an error: the test cannot run")
            return {"missing": True, "p_decisive": NAN}
        blk = ni_block(p["ref_y"], p["arm_y"], margin, self.a.n_boot, self.a.seed)
        blk["full_set"] = ni_block(p["ref_y_full"], p["arm_y_full"], margin, self.a.n_boot, self.a.seed)
        blk.update({k: p[k] for k in ("arm", "ref", "run", "ref_run", "n_expected", "n_pairs", "n_dropped",
                                      "status_ref", "status_arm", "s2_arm", "s2_ref")})
        blk["dropped"] = p.get("dropped", [])
        # the C1 cascades' "policy NI with zero / negligible coverage" reading (A6.11 (c).4, A6.12, A6.16 (m), A6.20 (e))
        # is decided on the two coverages of A6.22 (a) 5 -- tool-decision coverage and argument coverage, the 5 % rule
        # (A6.22 (a) 7); the pooled delegation ratio (``coverage``) stays as a descriptive figure. Assist / guard
        # policies never delegate, so none of this is reported for them (System-1 acts there on every opportunity)
        if where.split(" ", 1)[0] in ("C1F", "C1L"):
            cov = p.get("coverage") or {}
            blk["coverage"] = cov.get("ratio", NAN)
            blk["coverage_n"] = cov.get("n")
            for k in ("tool_coverage", "arg_coverage", "arg_coverage_s1_calls"):
                blk[k] = cov.get(k, NAN)
            blk["n_arg_slots"], blk["n_arg_delegated"] = cov.get("n_arg_slots"), cov.get("n_arg_delegated")
            blk["delegation"] = delegation_class(blk["tool_coverage"], blk["arg_coverage"])
        blk["readouts"] = cost_block(p["tab_ref"], p["tab_arm"], p["keys"], self.a.n_boot, self.a.seed)
        blk["_pair"] = p
        return blk

    # -------------------------------------------------------------------------- the competence gate
    def gate(self) -> dict[str, Any]:
        g = json.loads(Path(self.a.gate).read_text()) if Path(self.a.gate).exists() else None
        if g is None:
            self.flags.error("gate", f"{self.a.gate} not found: the A5.1 gate verdict decides the conjunction")
            return {"co_primary": True, "source": self.a.gate, "candidate": None, "verdict": None}
        model = ((self.runs["E2a_intern"].exp or {}).get("s1") or {}).get("model") or \
            (PC.load(CONFIGS["E2a_intern"]).get("s1") or {}).get("model", "")
        size = str(model).rsplit("-", 1)[-1]          # internlm/Intern-Decision-4B -> 4B
        cand = f"intern_{size}"
        c = (g.get("candidates") or {}).get(cand)
        if c is None:
            self.flags.error("gate", f"gate verdict has no candidate {cand!r} (the Intern run's System-1 is {model!r})")
            return {"co_primary": True, "source": _rel(self.a.gate), "candidate": cand, "verdict": None}
        return {"co_primary": c.get("verdict") == "co-primary", "source": _rel(self.a.gate), "candidate": cand,
                "model": model, "verdict": c.get("verdict"), "ratio_to_jev": c.get("ratio_to_jev"),
                "acc": c.get("acc"), "jev_acc": g.get("jev_acc")}

    # -------------------------------------------------------------------------- C1 / C2 / C6
    def conj_primary(self, label: str, gate: Mapping[str, Any]) -> dict[str, Any]:
        reg = REG[label]
        run_key = {"jevgym_tool_select.yaml": "E2a", "jevgym_meta_control.yaml": "E2b"}[reg["cfg"]]
        m = reg["margin"] / 100.0
        jev = self.ni_constituent(label, self.jevgym_pair(label, run_key, reg["arm"], reg["ref"]), m)
        out: dict[str, Any] = {"label": label, "test": "NI", "margin": m, "family": "two System-1 families (A5.1)",
                               "jev": jev, "co_primary": gate["co_primary"]}
        icfg, iarm, iref = reg["intern"]
        intern = self.ni_constituent(f"{label} (Intern)", self.jevgym_pair(f"{label} (Intern)", "E2a_intern", iarm, iref), m)
        out["intern"] = intern
        pj, pi = jev.get("p_decisive", NAN), intern.get("p_decisive", NAN)
        if gate["co_primary"]:
            if _finite(pj) and _finite(pi):
                out["p_decisive"], out["rule"] = M.holm_conjunction(pj, pi), "max(p_Jev, p_Intern)"
            else:
                self.flags.error(label, "a constituent of the two-family conjunction is missing: the primary enters "
                                        "Holm with p = 1 until both have run (A5.1)")
                out["p_decisive"], out["rule"], out["incomplete"] = 1.0, "incomplete conjunction -> 1", True
        else:
            out["p_decisive"], out["rule"] = (pj, "p_Jev (Intern-Decision demoted by the gate)") if _finite(pj) else \
                (1.0, "missing -> 1")
        return out

    # -------------------------------------------------------------------------- C3
    def c3(self) -> dict[str, Any]:
        reg = REG["C3"]
        p = self.jevgym_pair("C3", "E2a", reg["arm"], reg["ref"], ref_run_key="E2c")
        out: dict[str, Any] = {"label": "C3", "test": "Sup.", "margin": 0.0}
        if p is None or not p["n_pairs"]:
            out.update({"p_decisive": 1.0, "rule": "missing -> 1", "incomplete": True})
            self.flags.error("C3", "no paired data: C3 enters Holm with p = 1")
            return out
        out["jev"] = {**sign_block(p["ref_y"], p["arm_y"], self.a.n_boot, self.a.seed),
                      **{k: p[k] for k in ("arm", "ref", "run", "ref_run", "n_expected", "n_pairs", "n_dropped",
                                           "status_ref", "status_arm", "s2_arm", "s2_ref")}}
        out["jev"]["full_set"] = sign_block(p["ref_y_full"], p["arm_y_full"], self.a.n_boot, self.a.seed)
        out["jev"]["dropped"] = p.get("dropped", [])
        out["jev"]["_pair"] = p
        out["p_decisive"], out["rule"] = out["jev"]["p_decisive"], "sign_exact"
        return out

    # -------------------------------------------------------------------------- C4
    def routing_positives(self) -> dict[str, Any]:
        from jevharness.bench.meta_gold import gate_records

        R = self.runs["E2b"]
        mx = R.arm(C4_MEASURE_ARM, keep_trajs=True) if R.exists else None
        if mx is None:
            self.flags.error("C4", f"measurement arm `{C4_MEASURE_ARM}` not found in {_rel(R.dir)}: the A5.5 count "
                                   "cannot be read (C4 enters with p = 1)")
            return {"n_pos": None}
        self._note_arm("E2b", C4_MEASURE_ARM, mx, R.dir)
        expected = set(R.expected(C4_MEASURE_ARM))
        trajs = [t for t in mx["trajs"] if not t.get("error") and task_key(t) in expected]
        recs = [r for r in gate_records(trajs, "routing", modes=("shadow",)) if r.get("gold_status") == "ok"]
        st = M.gate_stats(recs)
        n_err = sum(1 for t in mx["trajs"] if t.get("error"))
        return {"arm": C4_MEASURE_ARM, "n_episodes": len(trajs), "n_errored_episodes": n_err,
                "n_labelled": st.get("n", 0), "n_pos": st.get("n_pos", 0), "n_neg": st.get("n_neg", 0),
                "rule": f">= {C4_MIN_POSITIVES} labelled routing positives (A5.5, A6.3)"}

    def c4(self) -> dict[str, Any]:
        reg = REG["C4"]
        out: dict[str, Any] = {"label": "C4", "test": "Sup.", "margin": 0.0}
        cnt = self.routing_positives()
        out["opportunities"] = cnt
        p = self.jevgym_pair("C4", "E2b", reg["arm"], reg["ref"])
        if p is not None and p["n_pairs"]:
            out["jev"] = {**sign_block(p["ref_y"], p["arm_y"], self.a.n_boot, self.a.seed),
                          **{k: p[k] for k in ("arm", "ref", "run", "ref_run", "n_expected", "n_pairs", "n_dropped",
                                               "status_ref", "status_arm", "s2_arm", "s2_ref")}}
            out["jev"]["full_set"] = sign_block(p["ref_y_full"], p["arm_y_full"], self.a.n_boot, self.a.seed)
            out["jev"]["readouts"] = cost_block(p["tab_ref"], p["tab_arm"], p["keys"], self.a.n_boot, self.a.seed)
            out["jev"]["dropped"] = p.get("dropped", [])
            out["jev"]["_pair"] = p
        if cnt.get("n_pos") is None or cnt["n_pos"] < C4_MIN_POSITIVES:
            out["p_decisive"], out["rule"] = 1.0, "insufficient observed opportunities -> p = 1 (A6.3)"
            out["insufficient"] = True
            if cnt.get("n_pos") is not None:
                self.flags.warn("C4", f"{cnt['n_pos']} routing positives < {C4_MIN_POSITIVES}: C4 enters Holm with p = 1 "
                                      "(insufficient observed opportunities)")
        elif "jev" in out:
            out["p_decisive"], out["rule"] = out["jev"]["p_decisive"], "sign_exact"
        else:
            out["p_decisive"], out["rule"], out["incomplete"] = 1.0, "missing -> 1", True
            self.flags.error("C4", "no paired data: C4 enters Holm with p = 1")
        return out

    # -------------------------------------------------------------------------- C5
    def c5(self) -> dict[str, Any]:
        reg = REG["C5"]
        out: dict[str, Any] = {"label": "C5", "test": "Sup. (ΔU)", "margin": 0.0}
        p = self.jevgym_pair("C5", "E2c", reg["arm"], reg["ref"])
        if p is None or not p["n_pairs"]:
            out.update({"p_decisive": 1.0, "rule": "missing -> 1", "incomplete": True})
            self.flags.error("C5", "no paired data: C5 enters Holm with p = 1")
            return out
        ta, tb = p["tab_arm"], p["tab_ref"]

        def du(keys: Sequence[str], gross: bool) -> dict[str, float]:
            def u(r: Mapping[str, Any]) -> float:
                return C5_W * r["y"] - r["cost"] + (r["s1_cost"] if gross else 0.0) - C5_LAM / 3600 * r["latency"]
            return {k: u(ta[k]) - u(tb[k]) for k in keys}

        keys = p["keys"]
        exp_ids = list(ta)
        blk = du_block(du(keys, True), du(keys, False), self.a.n_boot, self.a.seed)
        blk["full_set"] = du_block(du(exp_ids, True), du(exp_ids, False), self.a.n_boot, self.a.seed)
        blk.update({k: p[k] for k in ("arm", "ref", "run", "ref_run", "n_expected", "n_pairs", "n_dropped",
                                      "status_ref", "status_arm", "s2_arm", "s2_ref")})
        blk["success_arm"] = float(np.mean([ta[k]["y"] for k in keys]))
        blk["success_ref"] = float(np.mean([tb[k]["y"] for k in keys]))
        A, B = self.runs["E2c"].arm(reg["arm"], keep_trajs=True), self.runs["E2c"].arm(reg["ref"], keep_trajs=True)
        from jevharness.bench.meta_gold import _fire_rate

        blk["rate_arm"] = _fire_rate([t for t in A["trajs"] if not t.get("error")], "reflection")[0]
        blk["rate_control"] = _fire_rate([t for t in B["trajs"] if not t.get("error")], "reflection")[0]
        blk["dropped"] = p.get("dropped", [])
        blk["_pair"] = p
        out["jev"] = blk
        out["p_decisive"], out["rule"] = blk["p_decisive"], blk["rule"]
        out["ceiling"] = self.c5_ceiling()
        return out

    def c5_ceiling(self) -> dict[str, Any]:
        """The one-step oracle ceiling and the *no headroom* rule (A4; descriptive, outside the decision)."""
        from jevharness.bench.meta_gold import ceiling

        R = self.runs["E2b"]
        mx = R.arm(C5_MEASURE_ARM, keep_trajs=True) if R.exists else None
        if mx is None:
            self.flags.warn("C5", f"`{C5_MEASURE_ARM}` missing: no ceiling / no-headroom readout")
            return {}
        trajs = [t for t in mx["trajs"] if not t.get("error")]
        out = {}
        for tag, ex in (("all", False), ("no_budget_fatals", True)):
            c = ceiling(trajs, "reflection", W=C5_W, n_boot=min(self.a.n_boot, 2000), seed=self.a.seed,
                        exclude_budget=ex)
            out[tag] = {k: c.get(k) for k in ("value", "ci", "no_headroom", "n_tasks", "note")}
        return out

    # -------------------------------------------------------------------------- C7
    def select_c7(self) -> dict[str, Any]:
        """A5.4 / A6.9 #11 / A6.18 (c): the C7 configuration from E2a (Flash, Jev) only."""
        R = self.runs["E2a"]
        out: dict[str, Any] = {"registered": "A5.4 (a), (b); A6.9 #11; A6.18 (c)", "tie_pt": C7_TIE, "candidates": {}}
        if not R.exists:
            self.flags.error("C7 selection", f"{_rel(R.dir)} missing")
            return out
        names = list(C7_ELIGIBLE_FIXED)
        sweep = sorted({n for n in arm_task_counts(R.registered) if n.startswith(C7_SWEEP_PREFIX) and n != C7_C1_ARM
                        and not n.endswith("_s2lite")}, key=lambda s: float(s[len(C7_SWEEP_PREFIX):]))
        names += sweep
        cands: dict[str, dict[str, Any]] = {}
        for nm in names:
            if nm == "llm_only":
                cands[nm] = {"ni_95": True, "cost_savings": 0.0, "tau2_arm": "llm_only", "note": "the reference"}
                continue
            p = self.jevgym_pair("C7 selection", "E2a", nm, "llm_only")
            if p is None or not p["n_pairs"]:
                cands[nm] = {"ni_95": False, "cost_savings": NAN, "missing": True}
                continue
            blk = ni_block(p["ref_y"], p["arm_y"], 0.03, min(self.a.n_boot, 2000), self.a.seed)
            cb = cost_block(p["tab_ref"], p["tab_arm"], p["keys"], min(self.a.n_boot, 2000), self.a.seed)
            fam = C7_ELIGIBLE_FIXED.get(nm, "jev_ts_cascade (tau-hat* refit, A5.4 (c))")
            cands[nm] = {"ni_95": blk["ni_95"], "p_decisive": blk["p_decisive"], "rule": blk["rule"], "n": blk["n"],
                         "diff": blk["diff"], "cost_savings": cb.get("cost_savings", NAN), "tau2_arm": fam}
        out["candidates"] = cands
        sel, reason, degenerate = choose_c7(cands)
        if degenerate:
            self.flags.warn("C7 selection", reason)
        out.update({"selected_e2_arm": sel, "reason": reason,
                    "tau2_arm_family": (cands.get(sel) or {}).get("tau2_arm") or C7_ELIGIBLE_FIXED.get(sel)})
        if "jev_ts_guard" in cands:
            out["interpretation"] = ("tau^2 `jev_guard` (policy guard + stop guard) has no E2 twin; E2a's tool-call guard "
                                     "`jev_ts_guard` is read as its E2 counterpart")
        return out

    def tau2_arm(self, name: str) -> dict[str, Any] | None:
        """One tau^2 arm merged over the domain directories (<tau2>/<domain>/<arm>.jsonl)."""
        merged: dict[str, Any] = {"episodes": {}, "files": []}
        found = False
        for dom in sorted(p for p in self.tau2_dir.glob("*") if p.is_dir() and not p.name.endswith(("_test", "_shadow"))):
            a = load_arm(dom / f"{name}.jsonl")
            if a is None:
                continue
            found = True
            merged["episodes"].update(a["episodes"])
            merged["files"].append({k: a.get(k) for k in ("path", "sha256", "n_records", "n_episodes", "partial_lines",
                                                           "s2_model")})
            self.simulated[f"tau2/{dom.name}"] = self.simulated.get(f"tau2/{dom.name}", False) or \
                run_is_simulated(dom, [a])
        if not found:
            return None
        self.provenance.setdefault("tau2", {"dir": _rel(self.tau2_dir), "arms": {}})["arms"][name] = merged["files"]
        return merged

    def tau2_expected(self, ref: Mapping[str, Any], arm: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
        """The 149 tested tasks: the 35 airline test ids of the committed split + every retail task (A5.4 (c))."""
        sp = json.loads(AIRLINE_SPLIT.read_text())["split"]
        air = [f"airline/{t}" for t in sp["test"]]
        retail = sorted({e["task"] for x in (ref, arm) for e in x["episodes"].values()
                         if str(e["task"]).startswith("retail/")}, key=lambda s: (len(s), s))
        info = {"airline_test": len(air), "airline_shadow_excluded": len(sp["shadow"]), "retail_found": len(retail)}
        if len(retail) != TAU2_RETAIL_N:
            (self.flags.warn if self.a.allow_dev_tasks else self.flags.error)(
                "C7", f"{len(retail)} retail tasks found in the two arms, registered {TAU2_RETAIL_N} (§1 / A6.9 #1)")
        return air + retail, info

    def c7(self, sel: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"label": "C7", "test": "NI", "margin": REG["C7"]["margin"] / 100.0,
                               "endpoint": "majority pass (>= 2 of 3 trials; A6.14, A6.16 (e))", "selection": sel}
        arm_name = self.a.c7_arm
        if not arm_name:
            self.flags.error("C7", "no --c7-arm given (the tau^2 arm of the selected configuration): C7 enters with p = 1")
            out.update({"p_decisive": 1.0, "rule": "missing -> 1", "incomplete": True})
            return out
        fam = str(sel.get("tau2_arm_family") or "")
        if fam and not arm_name.startswith(fam.split(" ", 1)[0]):
            self.flags.error("C7", f"--c7-arm {arm_name!r} does not belong to the selected configuration's tau^2 family "
                                   f"{fam!r} (A5.4 (a): the configuration is fixed on E2 before any tau^2 run)")
        ref, arm = self.tau2_arm("llm_only"), self.tau2_arm(arm_name)
        if ref is None or arm is None:
            self.flags.error("C7", f"tau^2 arm(s) missing under {_rel(self.tau2_dir)}: "
                                   + ", ".join(n for n, x in (("llm_only", ref), (arm_name, arm)) if x is None))
            out.update({"p_decisive": 1.0, "rule": "missing -> 1", "incomplete": True})
            return out
        exp_ids, info = self.tau2_expected(ref, arm)
        # A6.22 (e) 4: with the re-run ledger's attempt counts (every simulation stamped by the converter) an errored
        # simulation with attempts left is "retries outstanding"; without them every error is final (as before)
        logged = all(e.get("attempts_logged") for x in (ref, arm) for e in x["episodes"].values())
        ta = task_table(arm, exp_ids, trials=C7_TRIALS, retry_rule=logged)
        tb = task_table(ref, exp_ids, trials=C7_TRIALS, retry_rule=logged)
        p = pair_tables(tb, ta, exp_ids, names=("llm_only", arm_name))
        _status_flags(self.flags, "C7", "arm", arm_name, p["status_arm"], len(exp_ids), attempts_logged=logged)
        _status_flags(self.flags, "C7", "reference", "llm_only", p["status_ref"], len(exp_ids), attempts_logged=logged)
        m = out["margin"]
        if not p["n_pairs"]:
            out.update({"p_decisive": 1.0, "rule": "missing -> 1", "incomplete": True})
            self.flags.error("C7", "no complete paired task")
            return out
        blk = ni_block(p["ref_y"], p["arm_y"], m, self.a.n_boot, self.a.seed)
        blk["full_set"] = ni_block(p["ref_y_full"], p["arm_y_full"], m, self.a.n_boot, self.a.seed)
        keys = p["keys"]
        pa, pb = {k: ta[k]["pass1"] for k in keys}, {k: tb[k]["pass1"] for k in keys}
        bt = M.paired_bootstrap_diff(pb, pa, n=self.a.n_boot, seed=self.a.seed)
        blk["pass1"] = {"arm": float(np.mean(list(pa.values()))), "ref": float(np.mean(list(pb.values()))),
                        "diff": bt["diff"], "ci95": [bt["ci_lo"], bt["ci_hi"]],
                        "note": "descriptive; majority-pass NI does not establish pass^1 NI (A6.16 (e))"}
        s2 = {nm: sorted({f.get("s2_model") for f in x["files"] if f.get("s2_model")}) for nm, x in
              (("arm", arm), ("ref", ref))}
        blk.update({"s2_arm": ", ".join(s2["arm"]) or None, "s2_ref": ", ".join(s2["ref"]) or None})
        blk.update({"arm": arm_name, "ref": "llm_only", "run": _rel(self.tau2_dir), "ref_run": _rel(self.tau2_dir),
                    "n_expected": p["n_expected"], "n_pairs": p["n_pairs"], "n_dropped": p["n_dropped"],
                    "status_ref": p["status_ref"], "status_arm": p["status_arm"], "tasks": info,
                    "dropped": p["dropped"], "attempts_logged": logged})
        blk["readouts"] = cost_block(tb, ta, keys, self.a.n_boot, self.a.seed)
        blk["_pair"] = p
        out["jev"] = blk
        out["p_decisive"], out["rule"] = blk["p_decisive"], blk["rule"]
        # secondary rows (Holm within the tau^2 table, outside the family): the E2-carried tau-hat* and fallbacks
        sec = {}
        for nm in self.a.tau2_secondary or []:
            x = self.tau2_arm(nm)
            if x is None:
                self.flags.warn("C7", f"secondary tau^2 arm `{nm}` not found")
                continue
            tx = task_table(x, exp_ids, trials=C7_TRIALS, retry_rule=logged)
            q = pair_tables(tb, tx, exp_ids, names=("llm_only", nm))
            if q["n_pairs"]:
                sec[nm] = {k: v for k, v in ni_block(q["ref_y"], q["arm_y"], m, self.a.n_boot, self.a.seed).items()}
                sec[nm]["n_dropped"] = q["n_dropped"]
        if sec:
            lv = M.ni_holm_levels({k: v["p_decisive"] for k, v in sec.items()}, ALPHA)
            for k, v in sec.items():
                v.update({"p_holm_within_tau2": lv[k]["p_holm"], "ni_holm_within_tau2": bool(lv[k]["reject"])})
            out["secondary"] = sec
        return out

    # -------------------------------------------------------------------------- C8
    def e1_records(self, name: str) -> dict[str, dict[str, Any]] | None:
        from jevharness.bench.decision_eval import _rescore

        for prefix in ("s1_", "fc_"):
            p = self.e1_dir / f"{prefix}{name}.jsonl"
            if p.exists():
                break
        else:
            return None
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        out: dict[str, dict[str, Any]] = {}
        for r in recs:
            if r.get("dp") != "tool_select" or not str(r.get("source", "")).startswith("bfcl:"):
                continue
            out[str(r["id"])] = {"error": r.get("error"), "y": float(_rescore(r)) if "error" not in r else 0.0,
                                 "cost": float(r.get("cost_usd", 0.0) or 0.0),
                                 "latency_ms": float(r.get("latency_ms", 0.0) or 0.0), "source": r.get("source")}
        self.provenance.setdefault("E1", {"dir": _rel(self.e1_dir), "files": {}})["files"][name] = {
            "path": _rel(p), "sha256": _sha256(p), "n_records": len(recs), "n_bfcl_tool_select": len(out)}
        self.simulated["E1"] = self.simulated.get("E1", False) or run_is_simulated(self.e1_dir)
        return out

    def c8_population(self) -> list[str]:
        """The BFCL tool_select item ids of the registered C8 population, rebuilt from the E1 config and verified
        against the sha256 of ``configs/eval/e1_c8_population.json`` (A6.16 (l))."""
        from jevharness.bench.decision_eval import load_population, population_counts

        reg = json.loads(C8_POPULATION.read_text())
        exp = yaml.safe_load(E1_CONFIG.read_text())
        try:
            items = load_population(exp)
        except Exception as e:  # noqa: BLE001 - raw BFCL / When2Call data missing on this machine
            self.flags.error("C8", f"cannot rebuild the registered population from {_rel(E1_CONFIG)} ({type(e).__name__}: "
                                   f"{e}); using the run's BFCL tool_select records, checked by count only")
            return []
        pc = population_counts(items)
        if pc["sha256"] != reg["sha256"]:
            self.flags.error("C8", f"the rebuilt E1 population (sha256 {pc['sha256'][:12]}) differs from "
                                   f"{_rel(C8_POPULATION)} ({reg['sha256'][:12]}): data or exclusion drifted")
        ids = sorted(it.id for it in items if it.dp == "tool_select" and it.source.startswith("bfcl:"))
        want = sum(v for k, v in reg["by_source_dp"].items() if k.startswith("bfcl:") and k.endswith("|tool_select"))
        if len(ids) != want:
            self.flags.error("C8", f"{len(ids)} BFCL tool_select items rebuilt, registered {want}")
        return ids

    def c8(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": "C8", "test": "NI", "margin": REG["C8"]["margin"] / 100.0,
                               "endpoint": "item-weighted accuracy, BFCL tool_select items (A6.9)"}
        A, B = self.e1_records(E1_ARM), self.e1_records(E1_REF)
        if A is None or B is None:
            self.flags.error("C8", f"E1 record file(s) missing in {_rel(self.e1_dir)}: "
                                   + ", ".join(n for n, x in ((E1_ARM, A), (E1_REF, B)) if x is None))
            out.update({"p_decisive": 1.0, "rule": "missing -> 1", "incomplete": True})
            return out
        pop = self.c8_population()
        if not pop:   # population not rebuildable: the union of the two record files, counted against the registry
            reg = json.loads(C8_POPULATION.read_text())
            want = sum(v for k, v in reg["by_source_dp"].items() if k.startswith("bfcl:") and k.endswith("|tool_select"))
            pop = sorted(set(A) | set(B))
            if len(pop) != want and not self.a.allow_partial_population:
                self.flags.error("C8", f"{len(pop)} BFCL tool_select items in the records, registered {want}")
        if self.a.allow_partial_population:
            ran = set(A) | set(B)
            if len(ran & set(pop)) < len(pop):
                self.flags.warn("C8", f"partial population (dry run): {len(ran & set(pop))} of {len(pop)} items ran")
            pop = [i for i in pop if i in ran]
        extra = sorted((set(A) | set(B)) - set(pop))
        if extra and not self.a.allow_partial_population:
            self.flags.warn("C8", f"{len(extra)} BFCL tool_select record(s) outside the registered population ignored")

        def tab(R: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            t = {}
            for i in pop:
                r = R.get(i)
                st = "missing" if r is None else ("errored" if r["error"] else "ok")
                t[i] = {"status": st, "y": r["y"] if st == "ok" else 0.0}
                if st == "errored":   # A6.22 (e) 5; E1 logs no attempt count
                    t[i].update({"error_class": error_class(r), "attempts": None})
            return t

        ta, tb = tab(A), tab(B)
        src = {i: (A.get(i) or B.get(i) or {}).get("source") for i in pop}
        p = pair_tables(tb, ta, pop, names=(E1_REF, E1_ARM), family=src.get)
        _status_flags(self.flags, "C8", "arm", E1_ARM, p["status_arm"], len(pop), attempts_logged=False)
        _status_flags(self.flags, "C8", "reference", E1_REF, p["status_ref"], len(pop), attempts_logged=False)
        if not p["n_pairs"]:
            out.update({"p_decisive": 1.0, "rule": "missing -> 1", "incomplete": True})
            self.flags.error("C8", "no paired item")
            return out
        blk = ni_block(p["ref_y"], p["arm_y"], out["margin"], self.a.n_boot, self.a.seed)
        blk["full_set"] = ni_block(p["ref_y_full"], p["arm_y_full"], out["margin"], self.a.n_boot, self.a.seed)
        keys = p["keys"]

        def p50(R: Mapping[str, dict[str, Any]]) -> float:
            return float(np.percentile([R[k]["latency_ms"] for k in keys], 50))

        c_a = 1000 * float(np.mean([A[k]["cost"] for k in keys]))
        c_b = 1000 * float(np.mean([B[k]["cost"] for k in keys]))
        blk["readouts"] = {"cost_per_1k_arm": c_a, "cost_per_1k_ref": c_b, "cost_ratio": c_a / c_b if c_b else NAN,
                           "p50_latency_ms_arm": p50(A), "p50_latency_ms_ref": p50(B),
                           "p50_latency_ratio": p50(A) / p50(B) if p50(B) else NAN}
        blk.update({"arm": E1_ARM, "ref": E1_REF, "run": _rel(self.e1_dir), "ref_run": _rel(self.e1_dir),
                    "n_expected": p["n_expected"], "n_pairs": p["n_pairs"], "n_dropped": p["n_dropped"],
                    "status_ref": p["status_ref"], "status_arm": p["status_arm"], "dropped": p["dropped"]})
        blk["_pair"] = p
        out["jev"] = blk
        out["p_decisive"], out["rule"] = blk["p_decisive"], blk["rule"]
        return out

    # -------------------------------------------------------------------------- the family
    def run(self) -> dict[str, Any]:
        gate = self.gate()
        prim: dict[str, dict[str, Any]] = {}
        for lab in CONJ:
            prim[lab] = self.conj_primary(lab, gate)
        prim["C3"] = self.c3()
        prim["C4"] = self.c4()
        prim["C5"] = self.c5()
        sel = self._c7_selection()
        prim["C7"] = self.c7(sel)
        prim["C8"] = self.c8()
        self._guard_simulated()
        pvals = {}
        for lab in HOLM_ORDER:
            pv = prim[lab].get("p_decisive", NAN)
            if not _finite(pv):
                self.flags.error(lab, "decisive p is not a number: enters Holm with p = 1")
                pv = prim[lab]["p_decisive"] = 1.0
            pvals[lab] = float(pv)
        H = holm(pvals)
        for lab in HOLM_ORDER:
            self._decide(prim[lab], H[lab], gate)
        out = {"meta": self._meta(gate), "holm": {k: H[k] for k in HOLM_ORDER},
               "sensitivity_full_set": self._full_set_family(prim, gate, H),
               "ni_exact_size_settings": exact_size_settings(prim),
               "primaries": {k: _strip(prim[k]) for k in HOLM_ORDER}, "flags": self.flags.items,
               "provenance": self.provenance}
        return out

    @staticmethod
    def _full_set_family(prim: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any],
                         H: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """The registered full-set sensitivity row (§1 / A6.9) at the family level: every primary's decisive p with
        errored / missing tasks counted as failures where they errored (same conjunction and C4 rules), Holm again."""
        def fsp(P: Mapping[str, Any], c: str) -> float:
            v = ((P.get(c) or {}).get("full_set") or {}).get("p_decisive")
            return float(v) if _finite(v) else 1.0

        ps: dict[str, float] = {}
        for lab in HOLM_ORDER:
            P = prim[lab]
            if P.get("incomplete") or P.get("insufficient"):
                ps[lab] = 1.0
            elif lab in CONJ and gate["co_primary"]:
                ps[lab] = max(fsp(P, "jev"), fsp(P, "intern"))
            else:
                ps[lab] = fsp(P, "jev")
        Hf = holm(ps)
        rows = {lab: {"p": ps[lab], "p_holm": Hf[lab]["p_holm"], "alpha_adj": Hf[lab]["alpha_adj"],
                      "reject": bool(Hf[lab]["reject"]), "differs_from_primary": bool(Hf[lab]["reject"]) !=
                      bool(H[lab]["reject"])} for lab in HOLM_ORDER}
        return {"rule": "errored / missing tasks count as failures in the arm where they errored (§1, A6.9)",
                "rows": rows, "n_reject": sum(r["reject"] for r in rows.values()),
                "differs": [k for k, r in rows.items() if r["differs_from_primary"]]}

    def _c7_selection(self) -> dict[str, Any]:
        if self.a.c7_selection and Path(self.a.c7_selection).exists():
            sel = json.loads(Path(self.a.c7_selection).read_text())
            sel["source"] = _rel(self.a.c7_selection)
            again = self.select_c7()
            if again.get("selected_e2_arm") != sel.get("selected_e2_arm"):
                self.flags.error("C7 selection", f"the recorded selection ({sel.get('selected_e2_arm')}) differs from the "
                                                 f"one recomputed from E2a now ({again.get('selected_e2_arm')})")
            return sel
        sel = self.select_c7()
        sel["source"] = "recomputed from E2a (no --c7-selection file)"
        return sel

    def _decide(self, P: dict[str, Any], h: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
        """Holm decision, Holm-level descriptive bounds and MDE, and the registered labels."""
        a_adj, rej, reached = h["alpha_adj"], bool(h["reject"]), bool(h["reached"])
        P["holm"] = {"rank": h["rank"], "m": h.get("m"), "alpha_adj": a_adj, "p_holm": h["p_holm"], "reject": rej,
                     "reached": reached}
        lab = P["label"]
        consts = [c for c in ("jev", "intern") if isinstance(P.get(c), dict) and P[c].get("_pair")]
        for c in consts:
            blk = P[c]
            pr = blk["_pair"]
            if P["test"] == "NI":
                adj = ni_block(pr["ref_y"], pr["arm_y"], P["margin"], self.a.n_boot, self.a.seed, alpha_adj=a_adj)
                blk["tango_lcb_adj"], blk["mde"], blk["margin_mde"] = adj["tango_lcb_adj"], adj["mde"], adj["margin_mde"]
                blk["lcb_boot_adj_descriptive"] = adj["lcb_boot_adj_descriptive"]
                blk["pass_at_level"] = bool(reached and _finite(blk["p_decisive"]) and blk["p_decisive"] <= a_adj)
                blk["reading"] = reading(blk, blk["pass_at_level"])
                blk["mde_note"] = adj["mde_note"]
            elif P["test"] == "Sup.":
                blk["mde"] = M.ni_mde(blk["n"], blk["discordance"], 0.0, a_adj)["delta_mde"] if blk["n"] else NAN
                blk["mde_note"] = _mde_note(blk["n"], blk["wins"] + blk["losses"]) if blk["n"] else "no paired task"
            else:
                blk["mde"] = mde_du(blk.get("sd", NAN), blk["n"], a_adj)
                blk["mde_note"] = None if _finite(blk["mde"]) else "per-task Delta-U has no spread (sd 0) or n < 2"
            if not _finite(blk.get("mde")) and not blk.get("mde_note"):
                blk["mde_note"] = ("80 % power is out of reach at this n and Holm level even if every discordant task "
                                   "were a win")
        # labels
        if P.get("incomplete"):
            P["decision"] = "INCOMPLETE (p = 1 entered Holm)"
        if lab in CONJ:
            j, i = P.get("jev") or {}, P.get("intern") or {}
            # A6.22 (a) 7: the C1 constituents' delegation class from the two coverages and the 5 % rule (A6.20 (e))
            cls = {nm: x.get("delegation") for nm, x in (("Jev", j), ("Intern-Decision", i)) if x}
            policy = any(c in ("zero", "negligible") for c in cls.values())
            if gate["co_primary"]:
                pj, pi = bool(j.get("pass_at_level")), bool(i.get("pass_at_level"))
                if not reached:
                    rep = "not reached (Holm step-down stopped at an earlier primary)"
                elif pj and pi:
                    rep = "cross-backend replication" + (" of policy NI" if policy else "")
                elif pj or pi:
                    rep = f"split ({'Jev' if pj else 'Intern-Decision'} passes)"
                else:
                    rep = "neither family establishes NI"
            else:
                rep = "Jev-specific confirmation" if rej else "not confirmed (Jev only; Intern-Decision demoted)"
            P["replication"] = rep
            if "decision" not in P:
                if rej:
                    P["decision"] = "NI"
                    zero = [nm for nm, c in cls.items() if c == "zero"]
                    negl = [nm for nm, c in cls.items() if c == "negligible"]
                    what = ([f"zero coverage for {' and '.join(zero)}"] if zero else []) + \
                        ([f"negligible coverage (< 5 %) for {' and '.join(negl)}"] if negl else [])
                    if what:
                        P["decision"] = (f"NI: policy NI with {'; '.join(what)} "
                                         "(A6.11 (c).4, A6.12, A6.16 (m), A6.20 (e), A6.22 (a))")
                elif gate["co_primary"]:   # the conjunction fails: each constituent's reading (A6.12: loss vs inconclusive)
                    P["decision"] = (f"not NI as a conjunction: Jev {j.get('reading', 'not tested')}; Intern-Decision "
                                     f"{i.get('reading', 'not tested')}")
                else:
                    P["decision"] = reading(j, False) if j.get("_pair") else "not tested"
        elif lab in ("C7", "C8"):
            j = P.get("jev") or {}
            if "decision" not in P:
                P["decision"] = "NI" if rej else reading(j, False)
                if lab == "C7" and not rej:
                    P["decision"] = "transfer NI was not established; " + P["decision"]
        else:   # C3, C4, C5
            if P.get("insufficient"):
                P["decision"] = "insufficient observed opportunities (p = 1, A6.3)"
            elif "decision" not in P:
                P["decision"] = "superior" if rej else ("superiority not established" if reached else
                                                       "superiority not established (Holm step-down stopped earlier)")

    def _meta(self, gate: Mapping[str, Any]) -> dict[str, Any]:
        sim = any(self.simulated.values())
        return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git_head(),
                "script": "scripts/primaries.py", "registry": "scripts/prereg_effective.py PRIMARIES",
                "alpha": ALPHA, "family": HOLM_ORDER, "n_boot": self.a.n_boot, "seed": self.a.seed,
                "simulated": sim, "simulated_runs": sorted(k for k, v in self.simulated.items() if v),
                "max_tasks": self.a.max_tasks, "gate": dict(gate),
                "runs": {k: _rel(r.dir) for k, r in self.runs.items()} | {"tau2": _rel(self.tau2_dir),
                                                                          "E1": _rel(self.e1_dir)},
                "n_errors": self.flags.n_errors}


def _strip(x: Any) -> Any:
    """Drop the private ``_pair`` working data (per-task maps) from the JSON; keep the per-task outcomes of the decisive
    pair under ``per_task`` so every count can be re-derived from the output alone."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "_pair":
                out["per_task"] = {"keys": v["keys"], "arm": [v["arm_y"][t] for t in v["keys"]],
                                   "ref": [v["ref_y"][t] for t in v["keys"]]}
                continue
            out[k] = _strip(v)
        return out
    if isinstance(x, list):
        return [_strip(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ====================================================================================== rendering
def _pct(v: Any, sign: bool = False) -> str:
    if not _finite(v):
        return "–"
    return f"{100 * float(v):+.1f}" if sign else f"{100 * float(v):.1f}"


def _pv(v: Any) -> str:
    if not _finite(v):
        return "–"
    v = float(v)
    return "<0.0001" if 0 < v < 1e-4 else f"{v:.4f}"


def _ci(c: Any) -> str:
    if not isinstance(c, (list, tuple)) or len(c) != 2 or not all(_finite(x) for x in c):
        return "–"
    return f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def to_markdown(res: Mapping[str, Any]) -> str:
    meta = res["meta"]
    sim = meta["simulated"]
    head = "SIMULATED DRY RUN — NOT RESULTS. " if sim else ""
    L = [f"# {head}Primary family (11 tests, Holm, FWER {meta['alpha']})", "",
         f"- generated {meta['generated']} at git `{meta['git'][:12]}` by `scripts/primaries.py`; registry "
         f"`{meta['registry']}`; B = {meta['n_boot']}, seed {meta['seed']}",
         f"- runs: " + ", ".join(f"{k} `{v}`" for k, v in meta["runs"].items()),
         f"- competence gate (`{meta['gate'].get('source')}`): {meta['gate'].get('candidate')} "
         f"{meta['gate'].get('verdict')} → {'two-family conjunction' if meta['gate'].get('co_primary') else 'Jev only'}",
         f"- flags: {meta['n_errors']} ERROR(s), {sum(1 for f in res['flags'] if f['level'] == 'WARN')} WARN(s)"
         + (f"; simulated inputs: {', '.join(meta['simulated_runs'])}" if sim else ""), "",
         "| # | test | arm → reference | n′ / n (dropped) | Δ (95 % CI), pt | W / L | p (rule) | Holm: rank, α_adj, "
         "p_Holm | decision | label | MDE, pt |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in meta["family"]:
        P = res["primaries"][lab]
        h = P.get("holm") or {}
        rows = [("jev", P.get("jev"))] + ([("intern", P.get("intern"))] if lab in CONJ else [])
        first = True
        for c, blk in rows:
            if not blk or blk.get("missing"):
                arm = "missing"
                L.append(f"| {lab if first else ''} | {P['test'] if first else ''} | {c}: {arm} | – | – | – | – | "
                         f"{'' if not first else _holm_cell(h)} | {'' if not first else P.get('decision', '')} | "
                         f"{'' if not first else P.get('replication', '')} | – |")
                first = False
                continue
            fam = f"{c}: " if lab in CONJ else ""
            npr = f"{blk.get('n_pairs', blk.get('n'))} / {blk.get('n_expected', '–')} ({blk.get('n_dropped', 0)})"
            wl = f"{blk.get('wins', '–')} / {blk.get('losses', '–')}" if "wins" in blk else \
                f"{blk.get('n_nonzero', '–')} ≠ 0"
            dlt = (f"{_pct(blk['diff'], True)} {_ci(blk.get('ci95'))}" if P["test"] != "Sup. (ΔU)" else
                   f"ΔU ${blk['diff']:+.4f} [{blk['ci95'][0]:+.4f}, {blk['ci95'][1]:+.4f}]" if _finite(blk.get("diff"))
                   else "–")
            mde = (f"{_pct(blk.get('mde'), True)}" if P["test"] != "Sup. (ΔU)" else
                   (f"${blk['mde']:.4f}" if _finite(blk.get("mde")) else "–"))
            pc = f"{_pv(blk.get('p_decisive'))} ({blk.get('rule')})"
            if first and lab not in CONJ and P.get("p_decisive") != blk.get("p_decisive"):
                pc = f"{_pv(P.get('p_decisive'))} ({P.get('rule')}; test p {_pv(blk.get('p_decisive'))})"
            L.append(f"| {lab if first else ''} | {P['test'] if first else ''} | {fam}`{blk.get('arm')}` → "
                     f"`{blk.get('ref')}` | {npr} | {dlt} | {wl} | {pc} | {_holm_cell(h) if first else ''} | "
                     f"{P.get('decision', '') if first else blk.get('reading', '')} | "
                     f"{P.get('replication', '') if first else ''} | {mde} |")
            first = False
        if first:
            L.append(f"| {lab} | {P['test']} | – | – | – | – | – | {_holm_cell(h)} | {P.get('decision', '')} | – | – |")
    fs = res.get("sensitivity_full_set") or {}
    if fs:
        rej = [k for k, r in fs["rows"].items() if r["reject"]]
        L += ["", f"Full-set sensitivity row ({fs['rule']}): Holm rejects {', '.join(rej) or 'none'}"
              + (f"; differs from the primary decision for {', '.join(fs['differs'])}" if fs.get("differs")
                 else "; identical decisions")]
    es = res.get("ni_exact_size_settings") or {}
    if es.get("missing_from_SETTINGS"):
        L += ["", "A6.17 (b): add to `SETTINGS` in `scripts/ni_exact_size.py` and re-run it before reporting: "
              + "; ".join(f"`{x}`" for x in es["missing_from_SETTINGS"])]
    L += ["", "p enters Holm as listed (C1 / C2 / C6: max(p_Jev, p_Intern) when Intern-Decision is co-primary; the row "
          "of each constituent shows its own p). NI rows: certified Berger–Boos p of one ordering of all "
          "(wins, losses) outcomes (`bb_ordered`, A6.21); superiority rows: exact sign test (C3, C4) / one-sided "
          "paired t-test (C5; the sign-flip p is descriptive). Δ in percentage points (arm − reference; C5 in $ of utility at W = $1). MDE: the true Δ at "
          "which the registered test has 80 % power at the achieved discordance and the primary's Holm level.", ""]
    L += ["## Per primary", ""]
    for lab in meta["family"]:
        P = res["primaries"][lab]
        L.append(f"### {lab} — {REG[lab]['claim'].split(';')[0]}")
        L.append("")
        L.append(f"- p entering Holm: {_pv(P.get('p_decisive'))} ({P.get('rule')}); decision: **{P.get('decision')}**"
                 + (f"; label: **{P['replication']}**" if P.get("replication") else ""))
        if lab == "C4":
            o = P.get("opportunities") or {}
            L.append(f"- opportunity rule (A5.5 / A6.3): `{o.get('arm')}` holds {o.get('n_pos')} labelled routing "
                     f"positives ({o.get('n_neg')} negatives, {o.get('n_episodes')} episodes) — threshold "
                     f"{C4_MIN_POSITIVES}")
        if lab == "C7":
            s = P.get("selection") or {}
            L.append(f"- configuration (selected on E2, {s.get('source')}): `{s.get('selected_e2_arm')}` — "
                     f"{s.get('reason')}; tau² arm family {s.get('tau2_arm_family')}")
        for c in ("jev", "intern"):
            blk = P.get(c)
            if not isinstance(blk, dict) or blk.get("missing"):
                continue
            fs = blk.get("full_set") or {}
            L.append(f"- {c}: `{blk.get('arm')}` ({blk.get('run')}) vs `{blk.get('ref')}` ({blk.get('ref_run')}); "
                     f"System-2 {blk.get('s2_arm', '–')} / {blk.get('s2_ref', '–')}")
            L.append(f"  - n′ = {blk.get('n_pairs')} of {blk.get('n_expected')}; arm status {blk.get('status_arm')}, "
                     f"reference status {blk.get('status_ref')}")
            if P["test"] == "Sup. (ΔU)":
                L.append(f"  - ΔU gross {blk['diff']:+.4f} $ (95 % CI {blk['ci95'][0]:+.4f}, {blk['ci95'][1]:+.4f}), "
                         f"net {blk['diff_net']:+.4f} (CI {blk['ci95_net'][0]:+.4f}, {blk['ci95_net'][1]:+.4f}); "
                         f"success {_pct(blk.get('success_arm'))} vs {_pct(blk.get('success_ref'))} %; realised firing "
                         f"rates {blk.get('rate_arm', NAN):.4f} (arm) / {blk.get('rate_control', NAN):.4f} (control)")
                if fs:
                    L.append(f"  - full-set row (errored = failure): ΔU {fs.get('diff', NAN):+.4f}, p "
                             f"{_pv(fs.get('p_decisive'))}, n {fs.get('n')}")
                L += _dropped_lines(blk)
                ce = P.get("ceiling") or {}
                if ce.get("all"):
                    L.append(f"  - one-step oracle ceiling (W = 1): {ce['all'].get('value')} CI {ce['all'].get('ci')}, "
                             f"no headroom: {ce['all'].get('no_headroom')}; without budget-kind fatals: "
                             f"{(ce.get('no_budget_fatals') or {}).get('value')}")
                continue
            L.append(f"  - success {_pct(blk.get('success_arm'))} vs {_pct(blk.get('success_ref'))} %; Δ "
                     f"{_pct(blk.get('diff'), True)} pt {_ci(blk.get('ci95'))}; wins {blk.get('wins')}, losses "
                     f"{blk.get('losses')} (discordance {_pct(blk.get('discordance'))} %)")
            if P["test"] == "NI":
                L.append(f"  - decisive p {_pv(blk.get('p_decisive'))} ({blk.get('rule')}); asymptotic Tango p "
                         f"{_pv(blk.get('p_tango_asymptotic'))}; Tango lower bound at α_adj "
                         f"{_pct(blk.get('tango_lcb_adj'), True)} pt; bootstrap p_NI {_pv(blk.get('p_boot_descriptive'))} "
                         f"(descriptive); loss-rate UB {_pct(blk.get('loss_rate_ub'))} %; reading: "
                         f"{blk.get('reading')}; MDE {_mde_txt(blk)}")
            else:
                L.append(f"  - decisive p {_pv(blk.get('p_decisive'))} (exact sign test, p_wins); bootstrap p_sup "
                         f"{_pv(blk.get('p_sup_boot_descriptive'))} (descriptive); MDE {_mde_txt(blk)}")
            if fs:
                L.append(f"  - full-set row (errored / missing = failure): n {fs.get('n')}, Δ {_pct(fs.get('diff'), True)} "
                         f"pt, p {_pv(fs.get('p_decisive'))} ({fs.get('rule')})")
            L += _dropped_lines(blk)
            if blk.get("delegation"):
                L.append(f"  - delegation (A6.22 (a) 5, 5 % rule of A6.20 (e)): **{blk['delegation']}** -- tool-decision "
                         f"coverage {_pct(blk.get('tool_coverage'))} %, argument coverage {_pct(blk.get('arg_coverage'))} % "
                         f"({blk.get('n_arg_delegated', '–')} of {blk.get('n_arg_slots', '–')} executed argument slots; "
                         f"inside System-1-chosen calls {_pct(blk.get('arg_coverage_s1_calls'))} %); pooled delegation "
                         f"ratio {_pct(blk.get('coverage'))} % (descriptive)")
            elif _finite(blk.get("coverage")):
                L.append(f"  - coverage (System-1 delegation ratio): {_pct(blk.get('coverage'))} %")
            ro = blk.get("readouts") or {}
            if "cost_savings" in ro:
                L.append(f"  - cost / episode ${ro['cost_arm']:.4f} vs ${ro['cost_ref']:.4f}: savings "
                         f"{_pct(ro['cost_savings'])} % {_ci(ro.get('cost_savings_ci'))}; System-2 input tokens "
                         f"{ro['tokens_arm']:.0f} vs {ro['tokens_ref']:.0f} (savings {_pct(ro['token_savings'])} %)")
            if "cost_per_1k_arm" in ro:
                L.append(f"  - $ / 1k decisions {ro['cost_per_1k_arm']:.4f} vs {ro['cost_per_1k_ref']:.4f}; p50 latency "
                         f"{ro['p50_latency_ms_arm']:.0f} vs {ro['p50_latency_ms_ref']:.0f} ms (ratio "
                         f"{ro['p50_latency_ratio']:.3f})")
            if "pass1" in blk:
                q = blk["pass1"]
                L.append(f"  - pass^1 (descriptive): {_pct(q['arm'])} vs {_pct(q['ref'])} %, Δ {_pct(q['diff'], True)} "
                         f"pt {_ci(q['ci95'])}")
        for nm, v in (P.get("secondary") or {}).items():
            L.append(f"- secondary row `{nm}` (Holm within the τ² table): Δ {_pct(v.get('diff'), True)} pt, p "
                     f"{_pv(v.get('p_decisive'))}, NI within the table: {v.get('ni_holm_within_tau2')}")
        L.append("")
    L += ["## Flags", ""]
    L += [f"- **{f['level']}** {f['where']}: {f['msg']}" for f in res["flags"]] or ["- none"]
    L += ["", "## Provenance", ""]
    for run, pv in res["provenance"].items():
        L.append(f"- {run} `{pv.get('dir')}`")
        for nm, x in (pv.get("arms") or pv.get("files") or {}).items():
            for f in (x if isinstance(x, list) else [x]):
                L.append(f"  - `{nm}`: `{f.get('path')}` sha256 `{str(f.get('sha256'))[:16]}` records "
                         f"{f.get('n_records')}")
    return "\n".join(L) + "\n"


def _dropped_lines(blk: Mapping[str, Any]) -> list[str]:
    """A6.22 (e) 5: the tasks that left the pair, listed next to the full-set row."""
    rows = blk.get("dropped") or []
    if not rows:
        return []
    out = [f"  - dropped from the pair ({len(rows)}; A6.22 (e) 5):"]
    for d in rows:
        bad = []
        for s in d["errored_in"]:
            info = [x for x in (d["error_class"].get(s),
                                f"{d['attempts'][s]} attempt(s)" if d["attempts"].get(s) is not None else None,
                                f"trials {d['errored_trials'][s]}" if (d.get("errored_trials") or {}).get(s) else None)
                    if x]
            bad.append(f"`{s}` {d['status'][s]}" + (f" ({'; '.join(info)})" if info else ""))
        other = ", ".join(f"`{s}` {o}" for s, o in d["other_outcome"].items()) or "–"
        out.append(f"    - `{d['task']}` [{d.get('family') or '–'}]: {'; '.join(bad)}; other arm: {other}")
    return out


def _mde_txt(blk: Mapping[str, Any]) -> str:
    if _finite(blk.get("mde")):
        return f"{_pct(blk['mde'], True)} pt"
    return f"n/a ({blk.get('mde_note') or 'undefined'})"


def _holm_cell(h: Mapping[str, Any]) -> str:
    if not h:
        return "–"
    return f"{h.get('rank')}, {_pv(h.get('alpha_adj'))}, {_pv(h.get('p_holm'))}{' ✓' if h.get('reject') else ''}"


TEX_SIGNED_PCT = {"delta", "mde", "tango_lcb_adj", "diff_pass1"}
TEX_PCT = {"success_arm", "success_ref", "coverage", "tool_coverage", "arg_coverage", "cost_savings", "token_savings",
           "loss_rate_ub", "discordance"}
TEX_P = {"p_decisive", "p_holm", "alpha_adj"}
TEX_USD = {"du_gross", "du_net", "du_mde", "cost_arm", "cost_ref", "cost_per_1k_arm", "cost_per_1k_ref"}
TEX_INT = {"n_pairs", "n_expected", "n_dropped", "wins", "losses", "holm_rank", "n_pos", "n_neg"}


def tex_value(path: Sequence[str], v: float) -> str:
    """Format one primaries number by its metric name (the last path segment; ``lo`` / ``hi`` take their parent's):
    percentage points with a sign for differences and MDEs, percent for rates, p-values to 3 decimals (``$<$0.001``),
    dollars to 3 significant digits, counts as integers."""
    last = path[-1]
    if last in ("lo", "hi") and len(path) > 1:
        last = path[-2]
    if last in ("delta_ci",):
        return f"{100 * v:+.1f}"
    if last in ("cost_savings_ci",):
        return f"{100 * v:.1f}"
    if last in TEX_SIGNED_PCT:
        return f"{100 * v:+.1f}"
    if last in TEX_PCT:
        return f"{100 * v:.1f}"
    if last in TEX_P:
        return "$<$0.001" if v < 0.001 else f"{v:.3f}"
    if last in TEX_INT:
        return f"{v:.0f}"
    if last in TEX_USD:
        return np.format_float_positional(v, precision=3, unique=True, fractional=False, trim="-")
    return np.format_float_positional(v, precision=3, unique=True, fractional=False, trim="-")


def to_tex(res: Mapping[str, Any]) -> str:
    """``\\defres{primaries/<label>/<...>}{value}`` (the paper_numbers preamble, so either file can be \\input
    first; :func:`tex_value` formats). Simulated results go under ``SIM/``."""
    from jevharness.bench.paper_numbers import _flatten, _seg, _tex_text

    meta = res["meta"]
    prefix = ("SIM",) if meta["simulated"] else ()
    nested: dict[str, Any] = {}
    for lab in meta["family"]:
        P = res["primaries"][lab]
        h = P.get("holm") or {}
        row: dict[str, Any] = {"p_decisive": P.get("p_decisive"), "p_holm": h.get("p_holm"),
                               "alpha_adj": h.get("alpha_adj"), "holm_rank": h.get("rank")}
        for c in ("jev", "intern"):
            blk = P.get(c)
            if not isinstance(blk, dict) or blk.get("missing"):
                continue
            keep = {k: blk.get(k) for k in ("n_pairs", "n_expected", "n_dropped", "wins", "losses", "p_decisive",
                                             "success_arm", "success_ref", "coverage", "tool_coverage", "arg_coverage",
                                             "tango_lcb_adj", "loss_rate_ub", "discordance")}
            fs = blk.get("full_set") or {}
            if P["test"] == "Sup. (ΔU)":     # C5: dollars of utility at W = $1, not percentage points
                keep.update({"du_gross": blk.get("diff"), "du_gross_ci": blk.get("ci95"), "du_net": blk.get("diff_net"),
                             "du_net_ci": blk.get("ci95_net"), "du_mde": blk.get("mde")})
                keep["full_set"] = {"du_gross": fs.get("diff"), "p_decisive": fs.get("p_decisive")}
            else:
                keep.update({"delta": blk.get("diff"), "delta_ci": blk.get("ci95"), "mde": blk.get("mde")})
                keep["full_set"] = {"delta": fs.get("diff"), "p_decisive": fs.get("p_decisive")}
            keep["readouts"] = {k: v for k, v in (blk.get("readouts") or {}).items()
                                if k in ("cost_arm", "cost_ref", "cost_savings", "cost_savings_ci", "token_savings",
                                         "cost_per_1k_arm", "cost_per_1k_ref", "p50_latency_ratio")}
            row[c] = keep
        fsr = (res.get("sensitivity_full_set") or {}).get("rows", {}).get(lab)
        if fsr:
            row["full_set_family"] = {"p_decisive": fsr["p"], "p_holm": fsr["p_holm"]}
        nested[lab] = row
    lines = ["% AUTO-GENERATED by scripts/primaries.py -- do not edit by hand.",
             f"% generated {meta['generated']} at git {meta['git'][:12]}"
             + ("; SIMULATED DRY RUN: keys under SIM/, never results" if meta["simulated"] else ""),
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    vals: dict[str, str] = {}
    for path, v in _flatten(nested, prefix + ("primaries",)):
        vals["/".join(path)] = tex_value(path, float(v))
    for lab in meta["family"]:
        P = res["primaries"][lab]
        base = "/".join(prefix + ("primaries", _seg(lab)))
        vals[f"{base}/decision"] = _tex_text(str(P.get("decision", "")))
        vals[f"{base}/rule"] = _tex_text(str(P.get("rule", "")))
        if P.get("replication"):
            vals[f"{base}/replication"] = _tex_text(str(P["replication"]))
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(vals.items())]
    lines += [r"\makeatother", ""]
    return "\n".join(lines)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _jsonable(x: Any) -> Any:
    """NaN / inf -> None (strict JSON)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ====================================================================================== CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="runs root (default runs/)")
    ap.add_argument("--run", action="append", metavar="CFG=DIR",
                    help="run directory of one config (e.g. jevgym_tool_select.yaml=runs/e2a_x); repeatable")
    ap.add_argument("--tau2", default=None, help="tau^2 output root (<domain>/<arm>.jsonl; default <runs>/tau2)")
    ap.add_argument("--e1", default=None, help="E1 run directory (default <runs>/e1_bfcl_when2call)")
    ap.add_argument("--gate", default=str(GATE_FILE), help="A5.1 competence-gate verdict")
    ap.add_argument("--c7-arm", default=None, help="the tau^2 arm of C7 (the selected configuration; the cascade at "
                                                  "the tau-hat* refit when a cascade is selected)")
    ap.add_argument("--c7-selection", default=None, help="c7_selection.json written by --select-c7 before the tau^2 runs")
    ap.add_argument("--tau2-secondary", action="append", metavar="ARM",
                    help="secondary tau^2 rows (the E2-carried tau-hat* arm, A5.4 (d) fallbacks); repeatable")
    ap.add_argument("--select-c7", action="store_true", help="only select C7's configuration from E2a and write "
                                                             "<out>/c7_selection.json (before any tau^2 run)")
    ap.add_argument("--out", required=True, help="output directory (primaries.json / .md / .tex)")
    ap.add_argument("--n-boot", type=int, default=10000, help="task-bootstrap replicates (descriptive intervals)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=None, help="DRY RUNS: cap the expected task list per arm")
    ap.add_argument("--allow-simulated", action="store_true", help="DRY RUNS: accept simulated inputs (SIM/ keys)")
    ap.add_argument("--allow-dev-tasks", action="store_true",
                    help="DRY RUNS: accept a task list that is not the frozen evaluation list (A6.13)")
    ap.add_argument("--allow-partial-population", action="store_true",
                    help="DRY RUNS: restrict C8 to the population items the run evaluated")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any ERROR flag is raised")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    an = Analysis(a)
    if a.select_c7:
        sel = an.select_c7()
        an._guard_simulated()
        sel.update({"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git_head(),
                    "e2a_run": _rel(an.runs["E2a"].dir), "flags": an.flags.items,
                    "provenance": an.provenance.get("E2a"), "simulated": any(an.simulated.values())})
        (out / "c7_selection.json").write_text(json.dumps(_jsonable(sel), indent=1, default=_json_default))
        print(f"[primaries] C7 configuration: {sel.get('selected_e2_arm')} ({sel.get('reason')}) -> "
              f"{_rel(out / 'c7_selection.json')}", file=sys.stderr)
        return 1 if (a.strict and an.flags.n_errors) else 0
    res = an.run()
    (out / "primaries.json").write_text(json.dumps(_jsonable(res), indent=1, default=_json_default))
    (out / "primaries.md").write_text(to_markdown(res))
    (out / "primaries.tex").write_text(to_tex(res))
    for f in res["flags"]:
        if f["level"] == "ERROR":
            print(f"[primaries] ERROR {f['where']}: {f['msg']}", file=sys.stderr)
    rej = [k for k in HOLM_ORDER if res["primaries"][k]["holm"]["reject"]]
    print(f"[primaries] {len(rej)} of {len(HOLM_ORDER)} primaries reject at FWER {ALPHA}: {', '.join(rej) or 'none'}"
          f"{' (SIMULATED)' if res['meta']['simulated'] else ''}; {res['meta']['n_errors']} ERROR flag(s); "
          f"-> {_rel(out)}", file=sys.stderr)
    return 1 if (a.strict and res["meta"]["n_errors"]) else 0


if __name__ == "__main__":
    sys.exit(main())
