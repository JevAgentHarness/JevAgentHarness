#!/usr/bin/env python3
"""The Stage-2 analysis (docs/PREREG_A7_STAGE2.md A7.4 as amended by A7.9-A7.18): the four-test family S2-1..S2-4
(Holm at FWER 0.05; every test an intersection-union test with decisive p = the maximum of its clause p values), the
six-row secondary table S2-1T / S2-1L / S2-L3a / S2-L3b / S2-1X / S2-4S (Holm within it, A7.9 (c), A7.12 (g),
A7.18 (f)) and the descriptive rows (`jevagent_default`, `best_control`, the leave-one-group-out rows, the score card,
S2-3's corpus readouts, the A7.13 attribution rows, the A7.18 (h) static-vs-in-the-loop table) and, when their inputs
exist, the exploratory JevAgentRL-v2 rows of A7.19 (f) (descriptive, outside every Holm table, labelled "exploratory
(A7.19)"). Every number traces to the files it names (path, sha256, record count in the provenance block); nothing is
typed by hand.

    python scripts/stage2_analysis.py --runs runs --out results/stage2 --strict
    # writes results/stage2/stage2.{json,md,tex} and the paper's \\defres{stage2/<test>/...} file paper/generated/stage2.tex

Inputs (``--run KEY=DIR`` overrides one run directory):

    stage2       <runs>/stage2_jevgym          S2-1, S2-2, S2-4, S2-1T, S2-1L, S2-4S, LOGO rows, jevagent_default
    states       <runs>/stage2_rl_states       S2-3's live replace arms (descriptive; the corpus sources)
    x            <runs>/stage2_x               S2-1X (A7.12)
    control      <runs>/stage2_best_control    best_control (A7.14; descriptive)
    attribution  <runs>/stage2_rl_attribution  A7.13 (ii) live rows (descriptive)
    sl_states    <runs>/stage2_startlux_states A7.18 (e) 3 StartLux live replace arms (descriptive)
    v2           <runs>/stage2_v2              A7.19 (f) JevAgentRL-v2 profile row `jevagent_best_v2` (exploratory)
    v2_states    <runs>/stage2_v2_states       A7.19 (f) JevAgentRL-v2 live replace arm `v2_ts_replace_f` (exploratory)
    e1 / e1_rl / e1_sl  <runs>/e1_bfcl_when2call[_rl|_startlux]   E1 rows of the System-1 tables (descriptive)
    tau2         <runs>/stage2_tau2/telecom    S2-L3a (single attempt per task, A7.15)
    alfworld     <runs>/stage2_alfworld        S2-L3b
    --s23            results/stage2/s23                 S2-3's common-state corpus + answers_{rl,intern,jev}.jsonl (A7.10 (c))
    --freeze-record  results/gpu_round5/FREEZE_RECORD.json   JevAgentRL's freeze record (A7.10 (b), (h)); none -> S2-3, S2-4 p = 1
    --tariff         results/stage2/tariff.json         S2-4's frozen per-episode tariff q (A7.13 (iii)); none -> S2-4 not analysed
    --tariff-sl      results/stage2/tariff_sl4b.json    S2-4S's tariff q_S (A7.18 (e) 5); none -> S2-4S not analysed
    --tariff-v2      results/stage2/tariff_v2.json      the v2 profile row's tariff (A7.19 (f), A7.13 (iii)); none -> cost not analysed
    --v2-freeze-record  results/gpu_round6_miles/final_FROZEN.txt   JevAgentRL-v2's freeze record (A7.19 (e)); none -> v2 rows not run
    --startlux-manifest  results/stage2/startlux_manifest.json   which StartLux sizes were served (A7.18 (d), (g))
    --profile / --profile-x                             the selected profile and the derived X profile (A7.12 (e))

The blocks are scripts/primaries.py's (``load_arm``, ``task_table``, ``pair_tables``, ``ni_block``, ``cost_block``,
``holm``, ``reading``); clause tests: NI = ``metrics.ni_pvalue(...)["p_decisive"]`` (``bb_ordered``), cost / latency /
accuracy = ``metrics.paired_t_test(...)["p_greater"]``. Simulated inputs are refused unless ``--allow-simulated``
(then every heading says SIMULATED and the TeX keys go under ``SIM/``)."""
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
from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench import stage2_corpus as C  # noqa: E402
from jevharness.bench.runner import arm_task_counts  # noqa: E402
from jevharness.bench.scorecard import SIM_S1_TYPES  # noqa: E402
from jevharness.harness import profile as HP  # noqa: E402

ALPHA = 0.05
M_JEVGYM, M_TAU2, M_ACC = 0.03, 0.05, 0.05        # A7.4 margins
FLASH, LITE, X_S2 = "gemini-3.5-flash", "gemini-3.5-flash-lite", "deepseek/deepseek-v4-pro"
LOGO_N = 200                                      # A7.11 (c): seeds 300000-300199
STAGE2_TASKS = ROOT / "configs/eval/jevgym_stage2_tasks.json"
PROFILE = ROOT / "configs/harness/jevagent_best.yaml"
PROFILE_X = ROOT / "configs/harness/jevagent_best_x.yaml"
S23_DIR = ROOT / "results/stage2/s23"
FREEZE_RECORD = ROOT / "results/gpu_round5/FREEZE_RECORD.json"
FREEZE_DEADLINE = dt.datetime(2026, 10, 7, 12, 0, tzinfo=dt.timezone.utc)     # A7.10 (h)
TARIFF = ROOT / "results/stage2/tariff.json"
TARIFF_SL = ROOT / "results/stage2/tariff_sl4b.json"                          # A7.18 (e) 5
TARIFF_V2 = ROOT / "results/stage2/tariff_v2.json"                            # A7.19 (f), measured as A7.13 (iii)
V2_FREEZE_RECORD = ROOT / "results/gpu_round6_miles/final_FROZEN.txt"         # A7.19 (e)
V2_FREEZE_DEADLINE = dt.datetime(2026, 10, 7, 12, 0, tzinfo=dt.timezone.utc)  # A7.19 (e)
STARTLUX_MANIFEST = ROOT / "results/stage2/startlux_manifest.json"            # A7.18 (d), (g)
CONTROL_SELECTION = ROOT / "results/stage2/control_selection.json"
RL_EVAL_PLAN = ROOT / "results/stage2/rl_eval_plan.json"                       # A7.13 (i): rows the freeze record has
PROFILE_SELECTION = ROOT / "results/stage2/profile_selection.json"            # A7.1 (h): read for S2-L3b only
TELECOM_IDS = ROOT / "configs/tau2/stage2/telecom_task_ids.json"
SCHEDULE_FILE = "schedule.json"                   # A7.16 (a): the task-major scheduler's record in the run directory
TEX_OUT = ROOT / "paper/generated/stage2.tex"
FAMILY = ("S2-1", "S2-2", "S2-3", "S2-4")
SECONDARY = ("S2-1T", "S2-1L", "S2-L3a", "S2-L3b", "S2-1X", "S2-4S")   # A7.9 (c), A7.12 (g), A7.18 (f): Holm within
S23_BACKENDS = ("rl", "intern", "jev")            # answers_<name>.jsonl of the corpus replay (A7.10 (c))
DIRS = {"stage2": "stage2_jevgym", "states": "stage2_rl_states", "x": "stage2_x", "control": "stage2_best_control",
        "attribution": "stage2_rl_attribution", "tau2": "stage2_tau2/telecom", "alfworld": "stage2_alfworld",
        "e1": "e1_bfcl_when2call", "e1_rl": "e1_bfcl_when2call_rl", "sl_states": "stage2_startlux_states",
        "e1_sl": "e1_bfcl_when2call_startlux", "v2": "stage2_v2", "v2_states": "stage2_v2_states"}
# A7.13's attribution table (`stage2_rl/<row>/...`): row -> (corpus replay name, live arm (run key, arm), E1 record
# file (run key, backend)). Every live row is read on the first LIVE_N Stage-2 tasks (A7.13 (ii): n = 100; the frozen
# model's row reuses its first 100 A7.10 tasks) so the rows pair. Intern-Decision's E1 row is the one of the StartLux
# E1 config (Stage 1's E1 config has no Intern backend).
RL_ROWS = {"jev": ("jev", ("states", "jev_ts_replace_f"), ("e1", "jev")),
           "intern": ("intern", ("states", "intern_ts_replace_f"), ("e1_sl", "intern_4b")),
           "base": ("base", ("attribution", "rl_base_ts_replace_f"), ("e1_rl", "rl_base")),
           "sft_v1": ("sft_v1", ("attribution", "rl_sft_v1_ts_replace_f"), ("e1_rl", "rl_sft_v1")),
           "sft_v1_opd": ("sft_v1_opd", ("attribution", "rl_sft_v1_opd_ts_replace_f"), ("e1_rl", "rl_sft_v1_opd")),
           "frozen": ("rl", ("states", "rl_ts_replace_f"), ("e1_rl", "jevagent_rl"))}
LIVE_N = 100
# A7.18 (h)'s static-vs-in-the-loop table (`stage2_s1/<row>/...`): the same row spec; live rows on the first 200 tasks
S1_ROWS = {"jev": RL_ROWS["jev"], "intern": RL_ROWS["intern"], "frozen": RL_ROWS["frozen"],
           "sl4b": ("sl4b", ("sl_states", "sl4b_ts_replace_f"), ("e1_sl", "startlux_4b")),
           "sl27b": ("sl27b", ("sl_states", "sl27b_ts_replace_f"), ("e1_sl", "startlux_27b"))}
S1_LIVE_N = 200
# A7.18 (e) 2: the StartLux size ladder, on the common-state corpus only (descriptive; `stage2_ladder/<row>/...`)
LADDER_ROWS = {f"sl{z}": (f"sl{z}", ("sl_states", f"sl{z}_ts_replace_f"), ("e1_sl", f"startlux_{z}"))
               for z in ("0.8b", "2b", "9b")}
S1_REFERENCE = "jev"
# A7.19 (f): JevAgentRL-v2 -- exploratory, descriptive, outside every Holm table, never under the JevAgentRL name
# (`stage2_v2/...`): the corpus replay of C (`answers_v2`), the live replace arm on the first 200 Stage-2 tasks (no E1
# row: A7.19 (f) names A7.18 (e) 2, 3 and 5 only) and the profile row vs `llm_only` with its own tariff
V2_LABEL = "exploratory (A7.19)"
V2_REPLAY, V2_BACKEND, V2_PROFILE_ARM, V2_LIVE_ARM = "v2", "jevagentrl_v2", "jevagent_best_v2", "v2_ts_replace_f"
V2_ROWS = {"v2": (V2_REPLAY, ("v2_states", V2_LIVE_ARM), None)}
# the Stage-2 score card (`stage2/scorecard/<arm>/...`): arm -> (run key, reference arm of the same System-2, or None);
# `best_control` is added from the A7.14 selection (its own run, or `llm_only_s2lite` when `small_only` is selected)
SCORECARD_ARMS = {"llm_only": ("stage2", None), "jevagent_best": ("stage2", "llm_only"),
                  "best_single": ("stage2", "llm_only"), "jevagent_best_rl": ("stage2", "llm_only"),
                  "jevagent_default": ("stage2", "llm_only"), "jevagent_best_sl4b": ("stage2", "llm_only"),
                  "llm_only_s2lite": ("stage2", None),
                  "jevagent_best_s2lite": ("stage2", "llm_only_s2lite"), "llm_only_x": ("x", None),
                  "jevagent_best_x": ("x", "llm_only_x"), "llm_only_xsmall": ("x", "llm_only_x")}
FAVOURABLE = "favourable estimate; the registered joint claim was not established"     # A7.16 (c)
NAN = float("nan")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _boot_ci(d: Sequence[float], n_boot: int, seed: int) -> list[float]:
    """Task-bootstrap 95 % percentile CI of a mean (descriptive)."""
    x = np.asarray(list(d), float)
    if len(x) < 2:
        return [NAN, NAN]
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def t_clause(d: Sequence[float], what: str, null: float = 0.0) -> dict[str, Any]:
    """One-sided paired t-test of mean(d) > 0 with the two-sided 95 % CI of the mean and its reading (A7.4; ``null``
    is the boundary on the scale of the effect, reported per A7.16 (c))."""
    x = np.asarray(list(d), float)
    n = len(x)
    if n < 2:
        m = float(x.mean()) if n else NAN
        return {"test": "paired_t", "n": n, "mean": m, "sd": NAN, "ci95": [NAN, NAN], "p": 1.0, "null": null,
                "est": m, "est_ci": [NAN, NAN], "boundary": null,
                "reading": "no paired task" if not n else "one paired task: not tested", "mde_floor": NAN}
    tt = M.paired_t_test(x)
    sd = float(x.std(ddof=1))
    half = 1.959964 * sd / math.sqrt(n)
    lo, hi = float(x.mean()) - half, float(x.mean()) + half
    if tt["p_greater"] <= ALPHA:
        rd = f"{what} established"
    elif _finite(lo) and lo <= 0 <= hi:
        rd = f"no {what} established at this n"
    elif not _finite(lo):
        rd = f"no {what} established at this n (no variation)"
    else:
        rd = f"the wrong direction: {what} reversed (95 % CI entirely below 0)"
    return {"test": "paired_t", "n": n, "mean": float(x.mean()), "sd": sd, "ci95": [lo, hi],
            "p": float(tt["p_greater"]), "null": null, "reading": rd,
            "est": float(x.mean()), "est_ci": [lo, hi], "boundary": null,     # A7.16 (c): effect, interval, boundary
            "mde_floor": (M._z(1 - ALPHA / 4) + M._z(0.8)) * sd / math.sqrt(n) if _finite(sd) else NAN}


class Run:
    """One run directory: experiment.yaml, the expected task list (a frozen list when given) and lazily loaded arms."""

    def __init__(self, key: str, d: Path, flags: P.Flags, frozen: Path | None = None, allow_dev: bool = False):
        self.key, self.dir, self.flags = key, d, flags
        self.exists = d.is_dir()
        self.exp = yaml.safe_load((d / "experiment.yaml").read_text()) if (d / "experiment.yaml").exists() else {}
        self.arms: dict[str, Any] = {}
        self.task_ids: list[str] = []
        if frozen is not None and self.exp:
            fz = json.loads(frozen.read_text())
            if dict(self.exp.get("suite") or {}) == fz["suite"]:
                self.task_ids = list(fz["task_ids"])
            else:
                (flags.warn if allow_dev else flags.error)(key, f"{P._rel(d)}: suite is not the frozen list of "
                                                                f"{P._rel(frozen)}")
                self.task_ids = P.EpisodeRun._suite_ids(self.exp["suite"])

    def arm(self, name: str, keep: bool = False) -> dict[str, Any] | None:
        if name not in self.arms or (keep and self.arms[name] is not None and "trajs" not in self.arms[name]):
            self.arms[name] = P.load_arm(self.dir / f"{name}.jsonl", keep) if self.exists else None
        return self.arms[name]

    def n(self, arm: str) -> int:
        return int(arm_task_counts(self.exp).get(arm) or len(self.task_ids))

    def expected(self, arm: str, cap: int | None = None) -> list[str]:
        n = self.n(arm) if cap is None else min(cap, self.n(arm))
        return self.task_ids[:n]


def _skip(profile: Mapping[str, Any] | None, s2: str, part: str | None = None) -> str | None:
    if profile is None:
        return "not tested (no profile file)"
    try:
        return HP.decision_points(profile, s2, part=part)[2]
    except HP.ProfileError as e:
        return f"not tested ({e})"


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's τ-b of two rankings (ties handled; NaN when either is constant). Descriptive (A7.18 (h))."""
    n, c, d, tx, ty = len(x), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                tx += 1
            elif sy == 0:
                ty += 1
            elif sx == sy:
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + tx) * (c + d + ty))
    return (c - d) / den if den > 0 else NAN


def _skip_backend(profile: Mapping[str, Any], s2: str, backend: str) -> str | None:
    """The skip reason of a profile arm run with ``profile_backend: backend`` (a `backends.<name>.skip`, or a
    decision point whose threshold is still `calibration: pending`)."""
    try:
        dps, _, why = HP.decision_points(profile, s2, backend=backend)
    except HP.ProfileError as e:
        return f"not tested ({e})"
    if why:
        return why
    if any((c or {}).get("calibration") == "pending" for c in dps.values()):
        return f"not tested (the profile's backends.{backend} thresholds are still pending calibration)"
    return None


def read_v2_freeze(path: Path) -> dict[str, Any]:
    """JevAgentRL-v2's freeze record (results/gpu_round6_miles/final_FROZEN.txt, written by gpu_round6_stages.sh stage
    freeze): ``key value`` lines (name, frozen_from, label, base, ..., frozen_at) and ``sha256  ./file`` lines, the
    latter under ``files`` (file -> sha256)."""
    rec: dict[str, Any] = {"files": {}}
    for line in path.read_text().splitlines():
        k, _, v = line.strip().partition(" ")
        v = v.strip()
        if not k:
            continue
        if len(k) == 64 and all(c in "0123456789abcdef" for c in k):
            rec["files"][v[2:] if v.startswith("./") else v] = k
        else:
            rec.setdefault(k, v)
    return rec


def v2_freeze_status(path: Path) -> tuple[str | None, dict[str, Any]]:
    """A7.19 (e): (None, record) when the v2 freeze record exists and was written by the deadline, else (the reason the
    v2 rows are not run, record). One rule for the launch gate (scripts/stage2_spec.py) and the analysis."""
    if not path.exists():
        return "not run (no JevAgentRL-v2 freeze record, A7.19 (e))", {}
    rec = read_v2_freeze(path)
    try:
        at = dt.datetime.fromisoformat(str(rec["frozen_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as e:
        return f"not run (unreadable JevAgentRL-v2 freeze record: {e})", rec
    if at.tzinfo is None:
        at = at.replace(tzinfo=dt.timezone.utc)
    rec["frozen_at_utc"] = at.isoformat()
    if at > V2_FREEZE_DEADLINE:
        return (f"not run (JevAgentRL-v2 freeze record {at.isoformat()} after the A7.19 (e) deadline "
                f"{V2_FREEZE_DEADLINE.isoformat()})"), rec
    return None, rec


def _unpriced(arm: Mapping[str, Any] | None) -> list[str]:
    return sorted({m for t in (arm or {}).get("trajs") or [] for m in ((t.get("usage") or {}).get("unpriced_models") or [])})


def _calls(arm: Mapping[str, Any] | None, keys: Sequence[str]) -> dict[str, float]:
    """Mean System-1 and System-2 calls per episode over ``keys`` (the latest valid record of each task)."""
    by = {P.task_key(t): t for t in (arm or {}).get("trajs") or [] if not t.get("error")}
    rows = [by[k] for k in keys if k in by]
    if not rows:
        return {"s1_calls": NAN, "s2_calls": NAN}
    u = [r.get("usage") or {} for r in rows]
    return {"s1_calls": float(np.mean([float((x.get("s1") or {}).get("calls", 0) or 0) for x in u])),
            "s2_calls": float(np.mean([float((x.get("s2") or {}).get("calls", 0) or 0) for x in u]))}


def _per_dp_coverage(arm: Mapping[str, Any] | None) -> dict[str, float]:
    """Query coverage per decision point: the share of logged decision records that consulted System-1 (descriptive;
    A7.16 (d): coverage is never called intervention)."""
    recs = M.decision_records([t for t in (arm or {}).get("trajs") or [] if not t.get("error")])
    out: dict[str, list[float]] = {}
    for r in recs:
        if r.get("dp") and r.get("mode") not in (None, "off"):
            out.setdefault(str(r["dp"]), []).append(float(r.get("s1_value") is not None))
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


# A7.16 (d) intervention reporting (descriptive): per profile arm, the share of tool decisions whose System-1 answer was
# executed (tool coverage) and of argument slots (arg coverage), System-2 calls per episode and calls avoided against
# the arm's LLM-only reference, and the reflection trigger's activity (questions asked, fires, tasks with a fire). One
# outcome per task (the last record without an error, as the paired analysis keeps one outcome per task).
INTERVENTION_ARMS = (  # (key, run key, arm, reference arm)
    ("llm_only", "stage2", "llm_only", None), ("best_single", "stage2", "best_single", "llm_only"),
    ("jevagent_best", "stage2", "jevagent_best", "llm_only"),
    ("jevagent_minus_tool", "stage2", "jevagent_minus_tool", "llm_only"),
    ("jevagent_best_rl", "stage2", "jevagent_best_rl", "llm_only"),
    ("jevagent_best_sl4b", "stage2", "jevagent_best_sl4b", "llm_only"),
    ("jevagent_default", "stage2", "jevagent_default", "llm_only"),
    ("llm_only_x", "x", "llm_only_x", None), ("jevagent_best_x", "x", "jevagent_best_x", "llm_only_x"),
    ("llm_only_xsmall", "x", "llm_only_xsmall", "llm_only_x"),
    ("telecom_llm_only", "tau2", "llm_only", None), ("telecom_jevagent_best", "tau2", "jevagent_best", "telecom_llm_only"))


def _one_per_task(path: Path) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if "task_id" not in t:
            continue
        prev = by_task.get(t["task_id"])
        if prev is None or not t.get("error") or prev.get("error"):
            by_task[t["task_id"]] = t
    return by_task


def _success(t: Mapping[str, Any]) -> bool:
    return bool((t.get("result") or {}).get("success"))


def _fired(t: Mapping[str, Any]) -> bool:
    return any(d.get("dp") == "reflection" and d.get("final_value") is True and not d.get("shadow")
               for s in t.get("steps") or [] for d in s.get("decisions") or [])


def intervention_stats(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive intervention counts of one arm (A7.16 (d)); ``tasks``: task id -> one trajectory."""
    from jevharness.bench.scorecard import call_coverage
    ts = dl = rq = rf = ft = 0
    for t in tasks.values():
        fired = False
        for s in t.get("steps") or []:
            for d in s.get("decisions") or []:
                if d.get("shadow"):
                    continue
                if d.get("dp") == "tool_select":
                    ts += 1
                    dl += bool(M.is_delegated(d))
                elif d.get("dp") == "reflection":
                    rq += 1
                    if d.get("final_value") is True:
                        rf, fired = rf + 1, True
        ft += fired
    n = len(tasks)
    cov = call_coverage(list(tasks.values())) if n else {}
    arg = cov.get("arg_coverage")
    return {"n": n, "success": sum(_success(t) for t in tasks.values()) / n if n else None,
            "tool_decisions": ts, "tool_delegated": dl, "tool_coverage": dl / ts if ts else 0.0,
            "arg_coverage": arg if _finite(arg) else 0.0, "llm_calls": cov.get("llm_calls_per_episode"),
            "trigger_questions": rq, "trigger_fires": rf, "trigger_fire_rate": rf / rq if rq else None,
            "trigger_tasks": ft}


class Analysis:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.flags = P.Flags()
        dirs = {k: Path(a.runs) / v for k, v in DIRS.items()}
        for kv in a.run or []:
            k, v = kv.split("=", 1)
            if k not in DIRS:
                raise SystemExit(f"--run {kv}: unknown key (one of {sorted(DIRS)})")
            dirs[k] = Path(v)
        self.dirs = dirs
        dev = a.allow_dev_tasks
        self.stage2 = Run("stage2", dirs["stage2"], self.flags, STAGE2_TASKS, dev)
        self.states = Run("states", dirs["states"], self.flags, STAGE2_TASKS, dev)
        self.x = Run("x", dirs["x"], self.flags, STAGE2_TASKS, dev)
        self.provenance: dict[str, Any] = {}
        self.simulated = False
        self.profile = HP.load_profile(a.profile) if Path(a.profile).exists() else None
        self.profile_x = HP.load_profile(a.profile_x) if Path(a.profile_x).exists() else None

    def _note(self, run: str, name: str, arm: Mapping[str, Any] | None, d: Path) -> None:
        if arm is None:
            return
        self.provenance.setdefault(run, {"dir": P._rel(d), "arms": {}})["arms"][name] = {
            k: arm.get(k) for k in ("path", "sha256", "n_records", "n_episodes", "partial_lines", "s2_model", "s1")}
        self.simulated = self.simulated or P.run_is_simulated(d, [arm])

    def _note_file(self, where: str, path: Path, **extra: Any) -> None:
        self.provenance.setdefault("files", {})[where] = {"path": P._rel(path), "sha256": P._sha256(path), **extra}

    # -------------------------------------------------------------------------- pairs
    def pair(self, where: str, R: Run, arm: str, ref: str, expected: Sequence[str], trials: int = 1,
             keep: bool = False, R_ref: Run | None = None) -> dict[str, Any] | None:
        Rr = R_ref or R
        A, B = R.arm(arm, keep), Rr.arm(ref, keep)
        for nm, x, RR in ((arm, A, R), (ref, B, Rr)):
            if x is None:
                self.flags.error(where, f"arm `{nm}` not found in {P._rel(RR.dir)}")
            self._note(RR.key, nm, x, RR.dir)
        if A is None or B is None:
            return None
        ta, tb = P.task_table(A, expected, trials=trials), P.task_table(B, expected, trials=trials)
        p = P.pair_tables(tb, ta, expected, names=(ref, arm))
        P._status_flags(self.flags, where, "arm", arm, p["status_arm"], len(expected))
        P._status_flags(self.flags, where, "reference", ref, p["status_ref"], len(expected))
        p.update({"tab_arm": ta, "tab_ref": tb, "A": A, "B": B})
        return p

    def ni_cost(self, label: str, R: Run, arm: str, ref: str, expected: Sequence[str], margin: float,
                latency: bool = False, trials: int = 1, arm_extra_cost: float = 0.0, keep: bool = False,
                R_ref: Run | None = None) -> dict[str, Any]:
        """NI on success (m = ``margin``, ``bb_ordered``) ∧ cost (paired t-test), optionally ∧ latency; decisive p =
        max. ``arm_extra_cost`` is a per-episode charge added to the arm's cost (S2-4's tariff q, A7.13 (iii))."""
        out: dict[str, Any] = {"label": label, "arm": arm, "reference": ref, "run": P._rel(R.dir), "margin": margin}
        p = self.pair(label, R, arm, ref, expected, trials=trials, keep=keep, R_ref=R_ref) if R.exists else None
        if p is None or not p["n_pairs"]:
            if not R.exists:
                self.flags.error(label, f"run directory {P._rel(R.dir)} missing")
            elif p is not None:
                self.flags.error(label, "no paired task")
            return {**out, "p_decisive": 1.0, "status": "not tested (missing input)"}
        keys, ta, tb = p["keys"], p["tab_arm"], p["tab_ref"]
        if arm_extra_cost:
            ta = {k: {**v, "cost": v["cost"] + arm_extra_cost} for k, v in ta.items()}
        nb = P.ni_block(p["ref_y"], p["arm_y"], margin, self.a.n_boot, self.a.seed)
        clauses = {"ni": {**nb, "p": nb["p_decisive"], "null": -margin, "est": nb["diff"], "est_ci": nb["ci95"],
                          "boundary": -margin, "reading": P.reading(nb, nb["p_decisive"] <= ALPHA)},
                   "cost": t_clause([tb[k]["cost"] - ta[k]["cost"] for k in keys], "cost saving")}
        if latency:
            clauses["latency"] = t_clause([tb[k]["latency"] - ta[k]["latency"] for k in keys], "latency saving")
        full = P.ni_block(p["ref_y_full"], p["arm_y_full"], margin, min(self.a.n_boot, 2000), self.a.seed)
        out.update({"n_expected": p["n_expected"], "n_pairs": p["n_pairs"], "n_dropped": p["n_dropped"],
                    "dropped": p["dropped"], "clauses": clauses,
                    "p_decisive": max(c["p"] for c in clauses.values()),
                    "cost": P.cost_block(tb, ta, keys, self.a.n_boot, self.a.seed),
                    "latency": {"arm_s": float(np.mean([ta[k]["latency"] for k in keys])),
                                "ref_s": float(np.mean([tb[k]["latency"] for k in keys]))},
                    "coverage": (p["A"] or {}).get("coverage"),
                    "full_set": {"p_ni": full["p_decisive"], "diff": full["diff"], "n": full["n"]},
                    "per_task": {"keys": keys, "arm": [p["arm_y"][k] for k in keys],
                                 "ref": [p["ref_y"][k] for k in keys]}})
        return out

    def describe_pair(self, label: str, R: Run, arm: str, ref: str, expected: Sequence[str],
                      R_ref: Run | None = None, trials: int = 1) -> dict[str, Any]:
        """A descriptive arm-vs-reference row (no test): success difference with the Newcombe and Tango score
        intervals (non-degenerate at zero discordance; a [0, 0] bootstrap is never shown, A7.11 (c)), cost with the
        paired task bootstrap, latency, System-1 / System-2 calls and per-decision-point query coverage."""
        out: dict[str, Any] = {"label": label, "arm": arm, "reference": ref, "run": P._rel(R.dir)}
        if not R.exists or not (R_ref or R).exists:
            return {**out, "status": "not run"}
        p = self.pair(label, R, arm, ref, expected, trials=trials, keep=True, R_ref=R_ref)
        if p is None or not p["n_pairs"]:
            return {**out, "status": "not available (missing input)"}
        keys, ta, tb = p["keys"], p["tab_arm"], p["tab_ref"]
        cells = M.ni_cells(p["ref_y"], p["arm_y"])
        nc, tg = M.newcombe_interval(**cells), M.tango_interval(**cells)
        lat = [ta[k]["latency"] - tb[k]["latency"] for k in keys]
        ca, cr = _calls(p["A"], keys), _calls(p["B"], keys)
        return {**out, "n_expected": p["n_expected"], "n_pairs": p["n_pairs"], "n_dropped": p["n_dropped"],
                "success_arm": float(np.mean([p["arm_y"][k] for k in keys])),
                "success_ref": float(np.mean([p["ref_y"][k] for k in keys])), "cells": cells,
                "delta": nc["diff"], "delta_ci": nc["ci"], "delta_ci_tango": tg["ci"],
                "cost": P.cost_block(tb, ta, keys, self.a.n_boot, self.a.seed),
                "latency_diff_s": float(np.mean(lat)), "latency_diff_s_ci": _boot_ci(lat, self.a.n_boot, self.a.seed),
                "calls_arm": ca, "calls_ref": cr, "coverage_arm": _per_dp_coverage(p["A"]),
                "coverage_pooled_arm": (p["A"] or {}).get("coverage")}

    # -------------------------------------------------------------------------- JevAgentRL gates
    def rl_gate(self) -> str | None:
        """A7.10 (b), (h): S2-3 and S2-4 need a freeze record written by the deadline and no pre-freeze exposure."""
        path = Path(self.a.freeze_record)
        if not path.exists():
            return "not tested (no frozen JevAgentRL checkpoint: no freeze record, A7.10 (h))"
        try:
            rec = json.loads(path.read_text())
            # `frozen_at` (docs/JEVAGENT_HARNESS.md §7) or `frozen_utc` (the key the round-5 record uses)
            at = dt.datetime.fromisoformat(str(rec["frozen_at"] if "frozen_at" in rec else rec["frozen_utc"]).replace("Z", "+00:00"))
        except (ValueError, KeyError, TypeError) as e:
            self.flags.error("JevAgentRL", f"{P._rel(path)}: unreadable freeze record ({e})")
            return "not tested (unreadable freeze record)"
        self._note_file("freeze_record", path, frozen_at=at.isoformat())
        if at.tzinfo is None:
            at = at.replace(tzinfo=dt.timezone.utc)
        if at > FREEZE_DEADLINE:
            return f"not tested (freeze record {at.isoformat()} after the deadline {FREEZE_DEADLINE.isoformat()}, A7.10 (h))"
        if rec.get("evaluation_exposure_before_freeze"):
            return "not tested (exploratory: evaluation information reached development before the freeze, A7.10 (b))"
        return None

    def tariff(self, tariff_path: str, label: str) -> tuple[dict[str, Any] | None, str | None]:
        path = Path(tariff_path)
        if not path.exists():
            return None, "not analysed (no frozen tariff q, A7.13 (iii))"
        t = json.loads(path.read_text())
        self._note_file(f"tariff {label}", path)
        if not _finite(t.get("q_usd_per_episode")):
            self.flags.error(label, f"{P._rel(path)}: no finite q_usd_per_episode")
            return None, "not analysed (tariff file without q)"
        if t.get("workload_matches") is False:
            return None, "not analysed (the measured workload does not match the evaluated deployment, A7.13 (iii))"
        return t, None

    # -------------------------------------------------------------------------- the family
    def s2_1(self) -> dict[str, Any]:
        why = _skip(self.profile, FLASH)
        if why:
            return {"label": "S2-1", "arm": "jevagent_best", "reference": "llm_only", "p_decisive": 1.0, "status": why}
        # A7.9 (a): NI on success and cost; the latency clause is the secondary row S2-1T
        return self.ni_cost("S2-1", self.stage2, "jevagent_best", "llm_only", self.stage2.expected("jevagent_best"),
                            M_JEVGYM)

    def s2_2(self) -> dict[str, Any]:
        """A7.11 (b): the composed profile vs the selected component with the largest standalone Stage-1 saving."""
        why = _skip(self.profile, FLASH) or _skip(self.profile, FLASH, "best_single")
        ref_info: dict[str, Any] = {}
        if self.profile is not None:
            try:
                bs = HP.section(self.profile, FLASH).get("best_single") or {}
                ref_info = {k: bs.get(k) for k in ("group", "arm", "original_candidate") if k in bs}
            except HP.ProfileError:
                pass
        if why:
            return {"label": "S2-2", "arm": "jevagent_best", "reference": "best_single", "p_decisive": 1.0,
                    "status": why, "reference_component": ref_info}
        out = self.ni_cost("S2-2", self.stage2, "jevagent_best", "best_single", self.stage2.expected("jevagent_best"),
                           M_JEVGYM)
        out["reference_component"] = ref_info
        return out

    def s2_3(self) -> dict[str, Any]:
        """A7.10 (c), (d): pooled common states; acc_X(t) = mean correctness over the task's occurrences; clauses (i)
        rl - intern > 0, (ii) rl - jev + 0.05 > 0, one common task set; worst-case sensitivity for excluded tasks."""
        cdir = Path(self.a.s23)
        out: dict[str, Any] = {"label": "S2-3", "arm": "rl", "references": ["intern", "jev"], "margin": M_ACC,
                               "corpus": P._rel(cdir)}
        why = self.rl_gate()
        if why:
            return {**out, "p_decisive": 1.0, "status": why}
        meta_p, occ_p = cdir / "corpus.json", cdir / "occurrences.jsonl.gz"
        if not meta_p.exists() or not occ_p.exists():
            self.flags.error("S2-3", f"common-state corpus missing in {P._rel(cdir)} (scripts/stage2_corpus.py build)")
            return {**out, "p_decisive": 1.0, "status": "not tested (missing input: no corpus)"}
        meta = json.loads(meta_p.read_text())
        for name, h in (meta.get("files") or {}).items():
            if (cdir / name).exists() and P._sha256(cdir / name) != h:
                self.flags.error("S2-3", f"{name}: sha256 differs from corpus.json (the corpus changed after it was frozen)")
        occ = C.read_jsonl_gz(occ_p)
        self._note_file("s23_occurrences", occ_p, n=len(occ))
        answers: dict[str, dict[str, dict[str, Any]]] = {}
        qs = {o["qhash"] for o in occ}
        for nm in S23_BACKENDS:
            ap = cdir / f"answers_{nm}.jsonl"
            if not ap.exists():
                self.flags.error("S2-3", f"no replay answers for `{nm}` ({P._rel(ap)})")
                return {**out, "p_decisive": 1.0, "status": f"not tested (missing input: answers_{nm})"}
            answers[nm] = self._answers(ap, nm, cdir)
            miss = len(qs - set(answers[nm]))
            if miss:
                self.flags.error("S2-3", f"`{nm}`: {miss} of {len(qs)} corpus questions have no replay record "
                                         "(replay incomplete; counted incorrect, A7.10 (d))")
            fails = sum(1 for h in qs if (answers[nm].get(h) or {}).get("error"))
            if fails:
                self.flags.warn("S2-3", f"`{nm}`: {fails} question(s) failed after the retries: incorrect (A7.10 (d))")
        eligible = [str(t) for t in meta.get("eligible_tasks") or []]
        acc = {nm: C.task_accuracy(occ, answers[nm], eligible) for nm in S23_BACKENDS}
        keys = [t for t in eligible if all(t in acc[nm] for nm in S23_BACKENDS)]
        excluded = [str(e["task"]) for e in meta.get("excluded") or []]

        def clauses(a: Mapping[str, Mapping[str, float]], ks: Sequence[str]) -> dict[str, Any]:
            cl = {}
            for nm, shift, what in (("intern", 0.0, "accuracy gain over Intern-Decision-4B"),
                                    ("jev", M_ACC, "non-inferiority to Jev (no more than 5 pt less accurate)")):
                d = [a["rl"][k] - a[nm][k] for k in ks]
                c = t_clause([x + shift for x in d], what, null=0.0 - shift)
                lo, hi = c["ci95"]
                c.update({"diff": float(np.mean(d)) if d else NAN, "est": float(np.mean(d)) if d else NAN,
                          "est_ci": [lo - shift, hi - shift], "boundary": 0.0 - shift,   # accuracy-difference scale
                          "diff_ci": _boot_ci(d, self.a.n_boot, self.a.seed),
                          "ci_method": "est_ci: t interval of the clause; diff_ci: task bootstrap (descriptive)"})
                cl[nm] = c
            return cl

        cl = clauses(acc, keys)
        if not cl["jev"]["p"] <= ALPHA:
            cl["jev"]["reading"] = "JevAgentRL was not shown to be no more than 5 pt less accurate than Jev at this n"
        res = {**out, "n_pairs": len(keys), "n_eligible": len(eligible), "n_excluded": len(excluded),
               "clauses": cl, "p_decisive": max(c["p"] for c in cl.values()),
               "accuracy": {nm: float(np.mean([acc[nm][k] for k in keys])) if keys else NAN for nm in S23_BACKENDS},
               "corpus_summary": meta.get("summary"), "excluded": meta.get("excluded") or []}
        if excluded:   # A7.10 (d): excluded tasks at accuracy 0 for JevAgentRL and 1 for each comparator
            wc = {nm: dict(acc[nm]) for nm in S23_BACKENDS}
            for t in excluded:
                wc["rl"][t], wc["intern"][t], wc["jev"][t] = 0.0, 1.0, 1.0
            wcl = clauses(wc, keys + [t for t in excluded if t not in keys])
            res["sensitivity_worst_case"] = {"n": len(keys) + len(excluded), "clauses": {k: {"p": v["p"], "diff": v["diff"]}
                                                                                       for k, v in wcl.items()},
                                             "p_decisive": max(v["p"] for v in wcl.values())}
        res["strata"] = self._strata(occ, answers, eligible)
        res["other_rows"] = self._other_answer_rows(cdir, occ, eligible)
        res["live"] = self._live_rows(self.states, ("rl_ts_replace_f", "intern_ts_replace_f", "jev_ts_replace_f"))
        return res

    def _answers(self, path: Path, name: str, cdir: Path) -> dict[str, dict[str, Any]]:
        recs: dict[str, dict[str, Any]] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["qhash"]] = r                     # the latest record of a question decides
        rp = cdir / f"replay_{name}.json"
        desc = json.loads(rp.read_text()).get("backend") if rp.exists() else None
        self._note_file(f"s23_answers_{name}", path, n=len(recs), backend=desc)
        if isinstance(desc, Mapping):
            kind = str(desc.get("backend") or desc.get("type") or "").lower()
            self.simulated = self.simulated or kind in SIM_S1_TYPES
        return recs

    @staticmethod
    def _strata(occ: Sequence[Mapping[str, Any]], answers: Mapping[str, Mapping[str, Any]],
                tasks: Sequence[str]) -> dict[str, Any]:
        """Accuracy by stratum (on-plan / after an off-plan step, and the LLM-only source), task-weighted (A7.10 (c))."""
        out: dict[str, Any] = {}
        subsets = {"on_plan": lambda o: o.get("stratum") == "on_plan",
                   "after_off_plan": lambda o: o.get("stratum") == "after_off_plan",
                   "llm_only_source": lambda o: o.get("source") == "llm_only"}
        for name, keep in subsets.items():
            sub = [o for o in occ if keep(o)]
            row: dict[str, Any] = {"occurrences": len(sub), "tasks": len({o["task"] for o in sub})}
            for nm, ans in answers.items():
                a = C.task_accuracy(sub, ans, tasks)
                row[nm] = float(np.mean(list(a.values()))) if a else NAN
            out[name] = row
        return out

    def _other_answer_rows(self, cdir: Path, occ: Sequence[Mapping[str, Any]], tasks: Sequence[str]) -> dict[str, Any]:
        """A7.13 (ii): further replays of the corpus (the base, SFT v1, SFT v1 + OPD rows), descriptive."""
        rows: dict[str, Any] = {}
        for ap in sorted(cdir.glob("answers_*.jsonl")):
            nm = ap.stem[len("answers_"):]
            if nm in S23_BACKENDS or nm == V2_REPLAY:     # v2: its own exploratory section (A7.19 (f))
                continue
            ans = self._answers(ap, nm, cdir)
            a = C.task_accuracy(occ, ans, tasks)
            rows[nm] = {"accuracy": float(np.mean(list(a.values()))) if a else NAN, "tasks": len(a),
                        "strata": self._strata(occ, {nm: ans}, tasks)}
        return rows

    def _live_rows(self, R: Run, arms: Sequence[str]) -> dict[str, Any]:
        """Live success, own-state accuracy and trajectory length per replace arm (descriptive, A7.10 (c))."""
        rows: dict[str, Any] = {}
        if not R.exists:
            return rows
        for nm in arms:
            A = R.arm(nm, keep=True)
            if A is None:
                continue
            self._note(R.key, nm, A, R.dir)
            ok = [t for t in A["trajs"] if not t.get("error")]
            own = self.accuracy(ok)
            rows[nm] = {"episodes": len(ok), "success": float(np.mean([bool((t.get("result") or {}).get("success"))
                                                                        for t in ok])) if ok else NAN,
                        "own_state_accuracy": float(np.mean(list(own.values()))) if own else NAN,
                        "steps": float(np.mean([len(t.get("steps") or []) for t in ok])) if ok else NAN}
        return rows

    @staticmethod
    def accuracy(trajs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """task -> share of its executed replace-mode tool_select decisions that are correct (own states)."""
        out: dict[str, list[float]] = {}
        for r in M.decision_records([t for t in trajs if not t.get("error")]):
            if r.get("dp") != "tool_select" or r.get("mode") != "replace" or r.get("s1_value") is None:
                continue
            ok = M.is_correct(r, r["s1_value"])
            if ok is not None:
                out.setdefault(str(r["task_id"]), []).append(float(ok))
        return {k: float(np.mean(v)) for k, v in out.items()}

    def tariffed(self, label: str, arm: str, tariff_path: str, who: str) -> dict[str, Any]:
        """NI ∧ cost with a frozen per-episode tariff q charged on top of the episode ledger (A7.13 (iii); S2-4 for
        JevAgentRL, S2-4S for StartLux-4B, A7.18 (e) 5), with its sensitivity rows."""
        base = {"label": label, "arm": arm, "reference": "llm_only"}
        tariff, why = self.tariff(tariff_path, label)
        if tariff is None:
            return {**base, "p_decisive": 1.0, "status": why}
        q = float(tariff["q_usd_per_episode"])
        exp = self.stage2.expected(arm)
        out = self.ni_cost(label, self.stage2, arm, "llm_only", exp, M_JEVGYM, arm_extra_cost=q, keep=True)
        A = self.stage2.arm(arm, keep=True)
        unpriced = _unpriced(A)
        if unpriced:
            self.flags.error(label, f"unpriced model(s) {unpriced}: set the served model's per-token entry to 0 "
                                    "(A7.13 (iii) adaptation)")
            return {**out, "p_decisive": 1.0, "status": f"not analysed (unpriced {who})"}
        if A is not None and any(e["s1_cost"] > 0 for e in A["episodes"].values()):
            self.flags.error(label, f"{who} episodes carry a per-token System-1 cost: with the tariff q it would be "
                                    "charged twice (A7.13 (iii) adaptation)")
            return {**out, "p_decisive": 1.0, "status": "not analysed (double-charged System-1 cost)"}
        out["tariff"] = {"q_usd_per_episode": q, **{k: tariff.get(k) for k in ("frozen_at", "served_model", "host",
                                                                               "serving_hours", "episodes")}}
        sens: dict[str, Any] = {}
        for nm, qq in {"zero": 0.0, **(tariff.get("sensitivity") or {})}.items():
            if not _finite(qq):
                continue
            r = self.ni_cost(f"{label} sensitivity", self.stage2, arm, "llm_only", exp, M_JEVGYM,
                             arm_extra_cost=float(qq))
            sens[nm] = {"q": float(qq), "p_cost": (r.get("clauses") or {}).get("cost", {}).get("p", NAN),
                        "cost_savings": (r.get("cost") or {}).get("cost_savings", NAN),
                        "cost_savings_ci": (r.get("cost") or {}).get("cost_savings_ci", [NAN, NAN])}
        out["tariff_sensitivity"] = sens
        return out

    def s2_4(self) -> dict[str, Any]:
        """A7.13 (iii): NI ∧ cost with the frozen per-episode tariff q charged on top of the episode ledger."""
        why = (_skip(self.profile, FLASH) or self.rl_gate()
               or (_skip_backend(self.profile, FLASH, "jevagent_rl") if self.profile is not None else None))
        if why:
            return {"label": "S2-4", "arm": "jevagent_best_rl", "reference": "llm_only", "p_decisive": 1.0,
                    "status": why}
        return self.tariffed("S2-4", "jevagent_best_rl", self.a.tariff, "JevAgentRL")

    def startlux_status(self, size: str) -> str | None:
        """'not run (...)' when the serving manifest (A7.18 (d)) records that the size could not be served."""
        path = Path(self.a.startlux_manifest)
        if not path.exists():
            return None
        man = json.loads(path.read_text())
        self._note_file("startlux_manifest", path)
        rec = next((s for s in man.get("sizes") or [] if s.get("size") == size), None)
        if rec is not None and rec.get("status") == "not_run":
            return f"not run (StartLux-Decision-{size}: {rec.get('why') or 'not served'}, A7.18 (g))"
        return None

    def s2_4s(self) -> dict[str, Any]:
        """A7.18 (e) 5: JevAgentHarness[profile, StartLux-4B] vs LLM-only, NI ∧ cost with its own tariff q_S."""
        arm = "jevagent_best_sl4b"
        why = (_skip(self.profile, FLASH) or self.startlux_status("4B")
               or (_skip_backend(self.profile, FLASH, "startlux_4b") if self.profile is not None else None))
        if why:
            return {"label": "S2-4S", "arm": arm, "reference": "llm_only", "p_decisive": 1.0, "status": why}
        return self.tariffed("S2-4S", arm, self.a.tariff_sl, "StartLux-4B")

    # -------------------------------------------------------------------------- secondary
    def s2_1t(self) -> dict[str, Any]:
        """A7.9 (b): faster per episode than LLM-only (paired t-test on ``usage.total.latency_ms``, reference - arm);
        not tested exactly when S2-1 is not; p = 1 unless the pair ran under the task-major scheduler (A7.16 (a))."""
        out: dict[str, Any] = {"label": "S2-1T", "arm": "jevagent_best", "reference": "llm_only"}
        why = _skip(self.profile, FLASH)
        if why:
            return {**out, "p_decisive": 1.0, "status": why}
        exp = self.stage2.expected("jevagent_best")
        p = self.pair("S2-1T", self.stage2, "jevagent_best", "llm_only", exp) if self.stage2.exists else None
        if p is None or not p["n_pairs"]:
            return {**out, "p_decisive": 1.0, "status": "not tested (missing input)"}
        keys, ta, tb = p["keys"], p["tab_arm"], p["tab_ref"]
        lat = t_clause([tb[k]["latency"] - ta[k]["latency"] for k in keys], "latency saving")
        res = {**out, "n_expected": p["n_expected"], "n_pairs": p["n_pairs"], "n_dropped": p["n_dropped"],
               "dropped": p["dropped"], "clauses": {"latency": lat}, "p_decisive": lat["p"],
               "latency_ref_s": float(np.mean([tb[k]["latency"] for k in keys])),
               "latency_arm_s": float(np.mean([ta[k]["latency"] for k in keys]))}
        sched = self.schedule(("jevagent_best", "llm_only"))
        res["scheduler"] = sched
        if not sched.get("task_major"):
            res.update({"p_decisive": 1.0, "status": "p = 1: the pair did not run under the task-major scheduler "
                                                     "(A7.16 (a)); latency is descriptive"})
        return res

    def schedule(self, arms: Sequence[str]) -> dict[str, Any]:
        path = self.stage2.dir / SCHEDULE_FILE
        if not path.exists():
            return {"task_major": False, "record": None}
        s = json.loads(path.read_text())
        self._note_file("schedule", path)
        # the pair ran under the scheduler when one invocation scheduled both arms (bench/runner.py, task_major)
        invs = s.get("invocations") or [{"arms": s.get("arms") or []}]
        ok = s.get("mode") == "task_major" and any(set(arms) <= set(i.get("arms") or []) for i in invs)
        return {"task_major": bool(ok), "record": P._rel(path), "seed": s.get("seed"),
                "concurrency": s.get("concurrency")}

    def s2_1l(self) -> dict[str, Any]:
        why = _skip(self.profile, LITE)
        if why:
            return {"label": "S2-1L", "arm": "jevagent_best_s2lite", "reference": "llm_only_s2lite", "p_decisive": 1.0,
                    "status": why}
        return self.ni_cost("S2-1L", self.stage2, "jevagent_best_s2lite", "llm_only_s2lite",
                            self.stage2.expected("jevagent_best_s2lite"), M_JEVGYM)

    def s2_l3a(self) -> dict[str, Any]:
        """A7.15 (e): NI on single-attempt task success (m = 5 pt) ∧ cost per task, on the frozen telecom sample."""
        base = {"label": "S2-L3a", "arm": "jevagent_best", "reference": "llm_only"}
        R = Run("tau2", self.dirs["tau2"], self.flags)
        ids = Path(self.a.telecom_ids)
        if not R.exists or not ids.exists():
            return {**base, "p_decisive": 1.0, "status": "not run (no telecom run or frozen id list)"}
        d = json.loads(ids.read_text())
        self._note_file("telecom_task_ids", ids)
        exp = [f"telecom/{t}" for t in (d["task_ids"] if isinstance(d, dict) else d)]
        return self.ni_cost("S2-L3a", R, "jevagent_best", "llm_only", exp, M_TAU2, trials=1)

    def _e6_missing_at_selection(self) -> bool:
        """A7.1 (h): when E6 was not complete at the profile selection, the action group stays off and S2-L3b is not
        run (Deviations entry of 2026-10-03). Read from the selection record that wrote the profile (the committed
        results/stage2/profile_selection.json beside the default profile, or --profile-selection)."""
        sel = self.a.profile_selection or (str(PROFILE_SELECTION) if Path(self.a.profile).resolve() == PROFILE.resolve()
                                           else None)
        if not sel or not Path(sel).exists():
            return False
        d = json.loads(Path(sel).read_text())
        act = [c for c in d.get("candidates") or [] if c.get("group") == "action"]
        if act and all(str(c.get("fate", "")).startswith("run directory") for c in act):
            self._note_file("profile_selection", Path(sel))
            return True
        return False

    def s2_l3b(self) -> dict[str, Any]:
        if self._e6_missing_at_selection():
            return {"label": "S2-L3b", "arm": "jevagent_best", "reference": "llm_react", "p_decisive": 1.0,
                    "status": "not run (A7.1 (h): E6 not run in Stage 1)"}
        R = Run("alfworld", self.dirs["alfworld"], self.flags)
        if not R.exists or R.arm("llm_react") is None:
            return {"label": "S2-L3b", "arm": "jevagent_best", "reference": "llm_react", "p_decisive": 1.0,
                    "status": "not run (no ALFWorld valid_seen run)"}
        exp = sorted({e["task"] for e in R.arm("llm_react")["episodes"].values()})
        return self.ni_cost("S2-L3b", R, "jevagent_best", "llm_react", exp, M_JEVGYM)

    def s2_1x(self) -> dict[str, Any]:
        """A7.12 (g): the derived X profile vs LLM-only on X: NI (m = 3 pt) ∧ cost; unpriced -> not analysed."""
        base = {"label": "S2-1X", "arm": "jevagent_best_x", "reference": "llm_only_x"}
        why = _skip(self.profile_x, X_S2)
        if why:
            return {**base, "p_decisive": 1.0, "status": why}
        if not self.x.exists:
            self.flags.error("S2-1X", f"run directory {P._rel(self.x.dir)} missing")
            return {**base, "p_decisive": 1.0, "status": "not tested (missing input)"}
        out = self.ni_cost("S2-1X", self.x, "jevagent_best_x", "llm_only_x", self.x.expected("jevagent_best_x"),
                           M_JEVGYM, keep=True)
        unpriced = sorted(set(_unpriced(self.x.arm("jevagent_best_x", keep=True)))
                          | set(_unpriced(self.x.arm("llm_only_x", keep=True))))
        if unpriced:
            self.flags.error("S2-1X", f"unpriced model(s) {unpriced} (A7.12 (h))")
            out.update({"p_decisive": 1.0, "status": "not analysed (unpriced X model)"})
        return out

    # -------------------------------------------------------------------------- descriptive
    def logo(self) -> dict[str, Any]:
        """A7.11 (c): every listed leave-one-group-out row vs `jevagent_best` on the first 200 tasks."""
        if self.profile is None or _skip(self.profile, FLASH):
            return {}
        sec = HP.section(self.profile, FLASH)
        rows: dict[str, Any] = {}
        exp = self.stage2.task_ids[:LOGO_N]
        for g, why in (sec.get("logo") or {}).items():
            if why and "configuration-identical to best_single" in str(why):
                arm = "best_single"
            elif why:
                rows[g] = {"label": f"LOGO-{g}", "removed_group": g, "status": str(why)}
                continue
            else:
                arm = f"jevagent_minus_{g}"
            rows[g] = {**self.describe_pair(f"LOGO-{g}", self.stage2, arm, "jevagent_best", exp), "removed_group": g}
        return rows

    def best_control(self) -> dict[str, Any]:
        """A7.14 (c): the selected control beside `llm_only` and `jevagent_best` (differences to both; no test)."""
        path = Path(self.a.control_selection)
        if not path.exists():
            return {"status": "no control selection file (scripts/select_profile.py --control-out)"}
        sel = json.loads(path.read_text())
        self._note_file("control_selection", path)
        arm = sel.get("best_control")
        if not arm:
            return {"status": sel.get("reason") or "no listed control satisfied the prespecified Stage-1 screen"}
        if arm == "small_only":
            R, nm = self.stage2, "llm_only_s2lite"
        else:
            R, nm = Run("control", self.dirs["control"], self.flags, STAGE2_TASKS, self.a.allow_dev_tasks), "best_control"
        exp = self.stage2.task_ids[: self.stage2.n("llm_only")]
        return {"selected": arm, "arm": nm, "reason": sel.get("reason"),
                "vs_llm_only": self.describe_pair("best_control", R, nm, "llm_only", exp, R_ref=self.stage2),
                "vs_jevagent_best": self.describe_pair("best_control", R, nm, "jevagent_best", exp, R_ref=self.stage2)}

    def scorecard(self, control: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Per-arm readout on its own expected tasks (success with its Wilson interval, cost, latency, LLM and
        System-1 calls, coverage) and the paired difference to the LLM-only arm of the same System-2 (descriptive;
        A7.14 (c)). Arm names as in the paper: `stage2/scorecard/<arm>/...`."""
        runs = {"stage2": self.stage2, "x": self.x}
        arms: dict[str, tuple[Run, str, Run | None, str | None]] = {}
        for nm, (rk, ref) in SCORECARD_ARMS.items():
            arms[nm] = (runs[rk], nm, runs[rk] if ref else None, ref)
        if control and control.get("arm"):
            Rc = self.stage2 if control["arm"] == "llm_only_s2lite" else \
                Run("control", self.dirs["control"], self.flags, STAGE2_TASKS, self.a.allow_dev_tasks)
            arms["best_control"] = (Rc, control["arm"], self.stage2, "llm_only")
        rows: dict[str, Any] = {}
        for key, (R, nm, Rr, ref) in arms.items():
            if not R.exists:
                continue
            A = R.arm(nm, keep=True)
            if A is None:
                continue
            self._note(R.key, nm, A, R.dir)
            exp = R.expected(nm) if R.key != "control" else self.stage2.expected("llm_only")
            tab = P.task_table(A, exp)
            ok = [k for k in exp if tab[k]["status"] == "ok"]
            row: dict[str, Any] = {
                "run": P._rel(R.dir), "arm": nm, "n_expected": len(exp), "n": len(ok),
                "success": float(np.mean([tab[k]["y"] for k in ok])) if ok else NAN,
                "success_ci": list(M.wilson_ci(int(sum(tab[k]["y"] for k in ok)), len(ok))) if ok else [NAN, NAN],
                "cost_usd": float(np.mean([tab[k]["cost"] for k in ok])) if ok else NAN,
                "latency_s": float(np.mean([tab[k]["latency"] for k in ok])) if ok else NAN,
                **_calls(A, ok), "coverage": _per_dp_coverage(A)}
            row["llm_calls"] = row.pop("s2_calls")
            if ref and Rr is not None and Rr.arm(ref) is not None:
                d = self.describe_pair(f"scorecard {key}", R, nm, ref, exp, R_ref=Rr)
                if not d.get("status"):
                    row["vs_baseline"] = {"reference": ref, "n_pairs": d["n_pairs"], "delta": d["delta"],
                                          "delta_ci": d["delta_ci"],
                                          "cost_savings": (d.get("cost") or {}).get("cost_savings", NAN),
                                          "cost_savings_ci": (d.get("cost") or {}).get("cost_savings_ci", [NAN, NAN])}
            rows[key] = row
        return rows

    # -------------------------------------------------------------------------- A7.13 / A7.18 System-1 tables
    def e1_items(self, run_key: str, backend: str) -> dict[str, float] | None:
        """item id -> correctness on the BFCL tool_select records of one E1 backend (errors count wrong; as C8)."""
        from jevharness.bench.decision_eval import _rescore

        p = self.dirs[run_key] / f"s1_{backend}.jsonl"
        if not p.exists():
            return None
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        ys = {str(r.get("id")): (float(_rescore(r)) if "error" not in r else 0.0) for r in recs
              if r.get("dp") == "tool_select" and str(r.get("source", "")).startswith("bfcl:")}
        self._note_file(f"e1_{backend}", p, n=len(ys))
        self.simulated = self.simulated or P.run_is_simulated(self.dirs[run_key])
        return ys

    def e1_accuracy(self, run_key: str, backend: str) -> tuple[float, int]:
        ys = self.e1_items(run_key, backend)
        return (float(np.mean(list(ys.values()))) if ys else NAN), len(ys or {})

    def _corpus(self) -> tuple[list[dict[str, Any]], list[str], Path]:
        cdir = Path(self.a.s23)
        if (cdir / "occurrences.jsonl.gz").exists() and (cdir / "corpus.json").exists():
            occ = C.read_jsonl_gz(cdir / "occurrences.jsonl.gz")
            return occ, [str(t) for t in json.loads((cdir / "corpus.json").read_text()).get("eligible_tasks") or []], cdir
        return [], [], cdir

    def s1_rows(self, rows: Mapping[str, tuple], live_n: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Per row: accuracy on the common-state corpus C (all, after an off-plan step), ECE of the answers' confidence
        on C, live success and own-state accuracy of its tool_select replace arm on the first ``live_n`` Stage-2 tasks,
        and E1 accuracy (BFCL tool_select). Returns (rows, per-row raw maps for paired differences). A row without its
        input is left out of the cell (`\\res` prints the placeholder)."""
        occ, eligible, cdir = self._corpus()
        runs = {"states": self.states,
                "attribution": Run("attribution", self.dirs["attribution"], self.flags, STAGE2_TASKS,
                                   self.a.allow_dev_tasks),
                "sl_states": Run("sl_states", self.dirs["sl_states"], self.flags, STAGE2_TASKS, self.a.allow_dev_tasks)}
        if any(live[0] == "v2_states" for _, live, _ in rows.values()):         # A7.19 (f) only
            runs["v2_states"] = Run("v2_states", self.dirs["v2_states"], self.flags, STAGE2_TASKS,
                                    self.a.allow_dev_tasks)
        live_tasks = set(self.stage2.task_ids[:live_n] or self.states.task_ids[:live_n])
        out: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        for row, (rname, (rk, arm), e1_spec) in rows.items():
            ek, backend = e1_spec or (None, None)
            r: dict[str, Any] = {"replay": rname, "live_arm": f"{DIRS[rk]}/{arm}",
                                 "e1": f"{DIRS[ek]}/s1_{backend}" if ek else None}
            rr: dict[str, Any] = {}
            ap = cdir / f"answers_{rname}.jsonl"
            if occ and ap.exists():
                ans = self._answers(ap, rname, cdir)
                acc = C.task_accuracy(occ, ans, eligible)
                off = C.task_accuracy([o for o in occ if o.get("stratum") == "after_off_plan"], ans, eligible)
                keep = set(eligible)
                conf, corr = [], []      # occurrence level: the chosen option's confidence vs its correctness
                for o in occ:
                    if o["task"] in keep:
                        a = ans.get(o["qhash"]) or {}
                        conf.append(float(a.get("confidence") or 0.0) if not a.get("error") else 0.0)
                        corr.append(float(C.answer_correct(o, a)))
                r.update({"corpus_acc": float(np.mean(list(acc.values()))) if acc else NAN, "corpus_tasks": len(acc),
                          "corpus_acc_offplan": float(np.mean(list(off.values()))) if off else NAN,
                          "corpus_tasks_offplan": len(off), "ece": M.ece(conf, corr) if conf else NAN})
                rr["corpus"] = acc
            R = runs[rk]
            A = R.arm(arm, keep=True) if R.exists else None
            if A is not None:
                self._note(R.key, arm, A, R.dir)
                ok_t = [t for t in A["trajs"] if not t.get("error") and P.task_key(t) in live_tasks]
                own = self.accuracy(ok_t)
                r.update({"live_n": len(ok_t),
                          "live_success": float(np.mean([bool((t.get("result") or {}).get("success")) for t in ok_t]))
                          if ok_t else NAN,
                          "own_acc": float(np.mean(list(own.values()))) if own else NAN})
                rr["live"] = {P.task_key(t): float(bool((t.get("result") or {}).get("success"))) for t in ok_t}
            if ek and self.dirs[ek].is_dir():
                items = self.e1_items(ek, backend)
                if items is not None:
                    r["l1_acc"], r["l1_n"] = (float(np.mean(list(items.values()))) if items else NAN), len(items)
                    rr["e1"] = items
            out[row] = r
            raw[row] = rr
        return out, raw

    def rl_table(self) -> dict[str, Any]:
        """A7.13 (i)-(ii) rows (`stage2_rl/<row>/...`, descriptive; live rows on the first 100 Stage-2 tasks). A row the
        freeze record does not have (results/stage2/rl_eval_plan.json `not_run`) carries that status."""
        rows = self.s1_rows(RL_ROWS, LIVE_N)[0]
        if RL_EVAL_PLAN.exists():
            for row, why in (json.loads(RL_EVAL_PLAN.read_text()).get("not_run") or {}).items():
                if row in rows:
                    rows[row]["status"] = why
        return rows

    def ladder_table(self) -> dict[str, Any]:
        """A7.18 (e) 2: the size ladder (0.8B, 2B, 9B) answers C only: accuracy on C (all, after an off-plan step), ECE
        and the difference to Jev on C with the task bootstrap (descriptive; 4B and 27B are in the A7.18 (h) table)."""
        rows, raw = self.s1_rows(LADDER_ROWS, S1_LIVE_N)
        ref = (self.s1_rows({S1_REFERENCE: S1_ROWS[S1_REFERENCE]}, S1_LIVE_N)[1].get(S1_REFERENCE) or {}).get("corpus")
        out: dict[str, Any] = {}
        for row, r in rows.items():
            keep = {k: r[k] for k in ("replay", "corpus_acc", "corpus_tasks", "corpus_acc_offplan", "corpus_tasks_offplan",
                                      "ece") if k in r}
            a = (raw.get(row) or {}).get("corpus")
            if a and ref:
                common = sorted(set(a) & set(ref))
                d = [a[k] - ref[k] for k in common]
                keep["vs_jev"] = {"corpus_delta": float(np.mean(d)) if d else NAN,
                                  "corpus_delta_ci": _boot_ci(d, self.a.n_boot, self.a.seed), "corpus_n": len(common)}
            out[row] = keep
        return out

    def s1_table(self) -> dict[str, Any]:
        """A7.18 (h): static vs common-state vs in-the-loop, side by side (descriptive, no test). Per row: E1, C and
        live measures; paired differences to Jev (E1 and live success: Newcombe score intervals on the common items /
        tasks; C: task bootstrap); Kendall's τ between the rank orders of the rows that have all three measures."""
        rows, raw = self.s1_rows(S1_ROWS, S1_LIVE_N)
        ref = raw.get(S1_REFERENCE) or {}
        for row, rr in raw.items():
            if row == S1_REFERENCE:
                continue
            vs: dict[str, Any] = {}
            for key, what in (("e1", "e1"), ("live", "live")):
                a, b = rr.get(key), ref.get(key)
                if a and b:
                    common = sorted(set(a) & set(b))
                    cells = M.ni_cells({k: b[k] for k in common}, {k: a[k] for k in common})
                    nc = M.newcombe_interval(**cells)
                    vs[f"{what}_delta"], vs[f"{what}_delta_ci"], vs[f"{what}_n"] = nc["diff"], nc["ci"], len(common)
            a, b = rr.get("corpus"), ref.get("corpus")
            if a and b:
                common = sorted(set(a) & set(b))
                d = [a[k] - b[k] for k in common]
                vs["corpus_delta"] = float(np.mean(d)) if d else NAN
                vs["corpus_delta_ci"] = _boot_ci(d, self.a.n_boot, self.a.seed)
                vs["corpus_n"] = len(common)
            if vs:
                rows[row]["vs_jev"] = vs
            if row in ("sl4b", "sl27b"):
                st = self.startlux_status("4B" if row == "sl4b" else "27B")
                if st:
                    rows[row]["status"] = st
        full = [r for r, v in rows.items() if all(_finite(v.get(k)) for k in ("l1_acc", "corpus_acc", "live_success"))]
        kend: dict[str, Any] = {"rows": full}
        if len(full) >= 3:
            e1 = [rows[r]["l1_acc"] for r in full]
            for k, nm in (("corpus_acc", "e1_vs_corpus"), ("live_success", "e1_vs_live")):
                kend[nm] = kendall_tau_b(e1, [rows[r][k] for r in full])
        return {"rows": rows, "kendall": kend, "reference": S1_REFERENCE,
                "note": "StartLux E1 rows: public BFCL / When2Call items; overlap with the vendor's training data cannot "
                        "be excluded (A7.18 (c))"}

    def attribution(self) -> dict[str, Any]:
        R = Run("attribution", self.dirs["attribution"], self.flags, STAGE2_TASKS, self.a.allow_dev_tasks)
        return self._live_rows(R, ("rl_base_ts_replace_f", "rl_sft_v1_ts_replace_f", "rl_sft_v1_opd_ts_replace_f"))

    # -------------------------------------------------------------------------- A7.19 (f): JevAgentRL-v2 (exploratory)
    def v2_backend_status(self) -> str | None:
        """Why the v2 profile row cannot be read: no profile or a skipped Flash section; no `backends.jevagentrl_v2`
        block (an arm without it would have run with the section's thresholds, never with v2's own fit); a block still
        pending or one whose fit delegates nothing."""
        why = _skip(self.profile, FLASH)
        if why:
            return why
        try:
            blk = (HP.section(self.profile, FLASH).get("backends") or {}).get(V2_BACKEND)  # type: ignore[arg-type]
        except HP.ProfileError as e:
            return f"not run ({e})"
        if blk is None:
            return (f"not run (the profile has no backends.{V2_BACKEND} block: v2's own thresholds were never fitted, "
                    "A7.19 (f))")
        return _skip_backend(self.profile, FLASH, V2_BACKEND)  # type: ignore[arg-type]

    def v2_profile(self) -> dict[str, Any]:
        """A7.19 (f), "as A7.18 (e) 5" without a test: `jevagent_best_v2` (runs/stage2_v2) vs `llm_only` of
        runs/stage2_jevgym on the Stage-2 list (n = 500): success with the Newcombe and Tango intervals, cost with v2's
        own frozen tariff q per episode on top of the episode ledger (A7.13 (iii)) and its sensitivity rows, latency,
        calls, per-decision-point coverage and the A7.16 (d) intervention counts. Descriptive only."""
        arm = V2_PROFILE_ARM
        R = Run("v2", self.dirs["v2"], self.flags, STAGE2_TASKS, self.a.allow_dev_tasks)
        where = f"JevAgentRL-v2 profile row ({V2_LABEL})"
        out: dict[str, Any] = {"label": V2_LABEL, "arm": arm, "reference": "llm_only", "run": P._rel(R.dir),
                               "reference_run": P._rel(self.stage2.dir)}
        if not (R.dir / f"{arm}.jsonl").exists():
            return {**out, "status": "not run"}
        why = self.v2_backend_status()
        if why:
            return {**out, "status": why}
        tariff, why = self.tariff(self.a.tariff_v2, where)
        if tariff is None:
            return {**out, "status": str(why)}
        p = self.pair(where, R, arm, "llm_only", R.expected(arm), keep=True, R_ref=self.stage2)
        if p is None or not p["n_pairs"]:
            return {**out, "status": "not available (missing input)"}
        A = p["A"]
        unpriced = _unpriced(A)
        if unpriced:
            self.flags.error(where, f"unpriced model(s) {unpriced}: the served model's per-token entry must be an exact "
                                    "0 (A7.13 (iii) adaptation)")
            return {**out, "status": "not analysed (unpriced JevAgentRL-v2)"}
        if any(e["s1_cost"] > 0 for e in A["episodes"].values()):
            self.flags.error(where, "JevAgentRL-v2 episodes carry a per-token System-1 cost: with the tariff q it would "
                                    "be charged twice (A7.13 (iii) adaptation)")
            return {**out, "status": "not analysed (double-charged System-1 cost)"}
        q = float(tariff["q_usd_per_episode"])
        keys, ta, tb = p["keys"], p["tab_arm"], p["tab_ref"]

        def with_q(qq: float) -> dict[str, dict[str, Any]]:
            return {k: {**v, "cost": v["cost"] + qq} for k, v in ta.items()}

        cells = M.ni_cells(p["ref_y"], p["arm_y"])
        nc, tg = M.newcombe_interval(**cells), M.tango_interval(**cells)
        lat = [ta[k]["latency"] - tb[k]["latency"] for k in keys]
        sens: dict[str, Any] = {}
        for nm, qq in {"zero": 0.0, **(tariff.get("sensitivity") or {})}.items():
            if _finite(qq):
                c = P.cost_block(tb, with_q(float(qq)), keys, self.a.n_boot, self.a.seed)
                sens[nm] = {"q": float(qq), "cost_savings": c.get("cost_savings", NAN),
                            "cost_savings_ci": c.get("cost_savings_ci", [NAN, NAN])}
        return {**out, "n_expected": p["n_expected"], "n_pairs": p["n_pairs"], "n_dropped": p["n_dropped"],
                "dropped": p["dropped"], "success_arm": float(np.mean([p["arm_y"][k] for k in keys])),
                "success_ref": float(np.mean([p["ref_y"][k] for k in keys])), "cells": cells,
                "delta": nc["diff"], "delta_ci": nc["ci"], "delta_ci_tango": tg["ci"],
                "cost": P.cost_block(tb, with_q(q), keys, self.a.n_boot, self.a.seed),
                "tariff": {"q_usd_per_episode": q, **{k: tariff.get(k) for k in (
                    "frozen_at", "served_model", "hardware", "host_usd_per_hour", "serving_hours", "episodes")}},
                "tariff_sensitivity": sens, "latency_diff_s": float(np.mean(lat)),
                "latency_diff_s_ci": _boot_ci(lat, self.a.n_boot, self.a.seed),
                "calls_arm": _calls(A, keys), "calls_ref": _calls(p["B"], keys), "coverage_arm": _per_dp_coverage(A),
                "intervention": intervention_stats(_one_per_task(R.dir / f"{arm}.jsonl"))}

    def v2(self) -> dict[str, Any] | None:
        """A7.19 (f): the JevAgentRL-v2 rows, exploratory and descriptive (no test, outside every Holm table), reported
        beside JevAgentRL under the label "exploratory (A7.19)": accuracy on the common-state corpus C (all, after an
        off-plan step, ECE; difference to Jev with the task bootstrap, as the A7.18 rows), the live replace arm on the
        first 200 Stage-2 tasks (success, own-state accuracy, trajectory length; success difference to Jev with the
        Newcombe interval) and the profile row (:meth:`v2_profile`). None -- and nothing added to any output -- when no
        v2 input exists; every row needs the v2 freeze record by the A7.19 (e) deadline."""
        cdir = Path(self.a.s23)
        have = {"corpus": (cdir / f"answers_{V2_REPLAY}.jsonl").exists(),
                "live": (self.dirs["v2_states"] / f"{V2_LIVE_ARM}.jsonl").exists(),
                "profile": (self.dirs["v2"] / f"{V2_PROFILE_ARM}.jsonl").exists()}
        if not any(have.values()):
            return None
        path = Path(self.a.v2_freeze_record)
        why, rec = v2_freeze_status(path)
        if path.exists():
            self._note_file("v2_freeze_record", path, frozen_at=rec.get("frozen_at_utc"))
        out: dict[str, Any] = {"label": V2_LABEL, "registered": "docs/PREREG_A7_STAGE2.md A7.19 (f): descriptive, "
                               "outside every Holm table; JevAgentRL (S2-3, S2-4) is A7.10's checkpoint",
                               "inputs": have, "freeze": {k: rec.get(k) for k in ("name", "frozen_from", "label",
                                                                                   "frozen_at") if k in rec}}
        if why:
            return {**out, "status": why}
        rows, raw = self.s1_rows(V2_ROWS, S1_LIVE_N)
        row, rr = rows["v2"], raw.get("v2") or {}
        ref = self.s1_rows({S1_REFERENCE: S1_ROWS[S1_REFERENCE]}, S1_LIVE_N)[1].get(S1_REFERENCE) or {}
        vs: dict[str, Any] = {}
        a, b = rr.get("corpus"), ref.get("corpus")
        if a and b:
            common = sorted(set(a) & set(b))
            d = [a[k] - b[k] for k in common]
            vs.update({"corpus_delta": float(np.mean(d)) if d else NAN,
                       "corpus_delta_ci": _boot_ci(d, self.a.n_boot, self.a.seed), "corpus_n": len(common)})
        a, b = rr.get("live"), ref.get("live")
        if a and b:
            common = sorted(set(a) & set(b))
            nc = M.newcombe_interval(**M.ni_cells({k: b[k] for k in common}, {k: a[k] for k in common}))
            vs.update({"live_delta": nc["diff"], "live_delta_ci": nc["ci"], "live_n": len(common)})
        if vs:
            row["vs_jev"] = vs
        if have["live"]:              # trajectory length on the same tasks as the live success (A7.18 (e) 3)
            live_tasks = set(self.stage2.task_ids[:S1_LIVE_N] or self.states.task_ids[:S1_LIVE_N])
            A = P.load_arm(self.dirs["v2_states"] / f"{V2_LIVE_ARM}.jsonl", keep_trajs=True) or {}
            ok = [t for t in A.get("trajs") or [] if not t.get("error") and P.task_key(t) in live_tasks]
            row["steps"] = float(np.mean([len(t.get("steps") or []) for t in ok])) if ok else NAN
        return {**out, "row": row, "profile": self.v2_profile()}

    def intervention(self) -> dict[str, Any]:
        """A7.16 (d), descriptive: intervention counts per profile arm (``INTERVENTION_ARMS``), the Jev profile's
        failures with and without a trigger fire, and the four-arm comparison on the first 200 fresh seeds that the
        leave-one-group-out arm shares (LLM-only, trigger alone, cascade alone, both): the interaction contrast
        (both - cascade) - (trigger - LLM-only) in points with a paired task-bootstrap interval (never a test)."""
        runs = {"stage2": self.dirs["stage2"], "x": self.dirs["x"], "tau2": self.dirs["tau2"]}
        loaded: dict[str, dict[str, dict[str, Any]]] = {}
        out: dict[str, Any] = {"arms": {}}
        for key, run, arm, ref in INTERVENTION_ARMS:
            path = runs[run] / f"{arm}.jsonl"
            if not path.exists():
                continue
            loaded[key] = _one_per_task(path)
            out["arms"][key] = {"reference": ref, **intervention_stats(loaded[key])}
        for key, row in out["arms"].items():
            ref = out["arms"].get(row["reference"]) if row["reference"] else None
            if ref and _finite(ref.get("llm_calls")) and _finite(row.get("llm_calls")):
                row["llm_calls_avoided"] = ref["llm_calls"] - row["llm_calls"]
        prof = loaded.get("jevagent_best")
        if prof:
            fails = [k for k, t in prof.items() if not _success(t)]
            out["jev_profile_failures"] = {"n": len(fails), "with_trigger_fired": sum(_fired(prof[k]) for k in fails)}
        need = ("llm_only", "jevagent_minus_tool", "best_single", "jevagent_best")
        if all(k in loaded for k in need):
            ids = [k for k in loaded["jevagent_minus_tool"] if all(k in loaded[a] for a in need)]
            y = {a: np.array([float(_success(loaded[a][k])) for k in ids]) for a in need}
            per_task = (y["jevagent_best"] - y["best_single"]) - (y["jevagent_minus_tool"] - y["llm_only"])
            out["first200"] = {"n": len(ids), "successes": {a: int(y[a].sum()) for a in need},
                               "interaction": float(per_task.mean()) if len(ids) else None,
                               "interaction_ci": _boot_ci(per_task, self.a.n_boot, self.a.seed) if len(ids) else None}
        return out

    def selection(self) -> dict[str, Any]:
        """The selected candidates of the A7.1 selection record (descriptive context for the selected profile: the
        Stage-1 sample size, savings and discordance on which each selected group qualified)."""
        sel = self.a.profile_selection or (str(PROFILE_SELECTION) if Path(self.a.profile).resolve() == PROFILE.resolve()
                                           else None)
        if not sel or not Path(sel).exists():
            return {}
        out: dict[str, Any] = {}
        for c in json.loads(Path(sel).read_text()).get("candidates") or []:
            if c.get("selected"):
                ni = c.get("ni") or {}
                out[f"{c['group']}"] = {"system2": c.get("system2"), "arm": c.get("arm"), "n": c.get("n_pairs"),
                                        "savings": c.get("savings"), "savings_ci": c.get("savings_ci"),
                                        "wins": ni.get("wins"), "losses": ni.get("losses")}
        return out

    # -------------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        fam = {"S2-1": self.s2_1(), "S2-2": self.s2_2(), "S2-3": self.s2_3(), "S2-4": self.s2_4()}
        sec = {"S2-1T": self.s2_1t(), "S2-1L": self.s2_1l(), "S2-L3a": self.s2_l3a(), "S2-L3b": self.s2_l3b(),
               "S2-1X": self.s2_1x(), "S2-4S": self.s2_4s()}
        for tab in (fam, sec):
            h = P.holm({k: float(v["p_decisive"]) for k, v in tab.items()})
            for k, v in tab.items():
                v["holm"] = h[k]
                v["verdict"] = verdict(v)
        desc: dict[str, Any] = {}
        if self.stage2.exists and self.stage2.arm("jevagent_default") is not None:
            desc["jevagent_default"] = self.ni_cost("descriptive", self.stage2, "jevagent_default", "llm_only",
                                                    self.stage2.expected("jevagent_default"), M_JEVGYM, latency=True)
        desc["logo"] = self.logo()
        desc["best_control"] = self.best_control()
        if self.x.exists and self.x.arm("llm_only_xsmall") is not None:
            desc["llm_only_xsmall"] = self.describe_pair("llm_only_xsmall", self.x, "llm_only_xsmall", "llm_only_x",
                                                         self.x.expected("llm_only_xsmall"))
        desc["attribution_live"] = self.attribution()
        desc["intervention"] = self.intervention()
        desc["selection"] = self.selection()
        desc["stage2_rl"] = self.rl_table()
        desc["stage2_s1"] = self.s1_table()
        desc["stage2_ladder"] = self.ladder_table()
        v2 = self.v2()
        if v2 is not None:            # A7.19 (f): only when a v2 input exists; nothing is added otherwise
            desc["stage2_v2"] = v2
        desc["scorecard"] = self.scorecard(desc["best_control"])
        if self.simulated and not self.a.allow_simulated:
            self.flags.error("inputs", "simulated run(s): never Stage-2 numbers; --allow-simulated for a dry run")
        return {"registered": "docs/PREREG_A7_STAGE2.md A7.4 as amended by A7.9-A7.16", "family": fam,
                "secondary": sec, "descriptive": desc, "flags": self.flags.items, "provenance": self.provenance,
                "meta": {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                         "git": P._git_head(), "simulated": self.simulated,
                         "profile": {k: (self.profile or {}).get(k) for k in ("_path", "_sha256", "status")},
                         "profile_x": {k: (self.profile_x or {}).get(k) for k in ("_path", "_sha256", "status")}}}


def verdict(v: Mapping[str, Any]) -> str:
    """A7.16 (c): the joint adjusted decision; a non-rejected test whose clauses are all nominally favourable reads
    "favourable estimate; the registered joint claim was not established"."""
    if v["holm"]["reject"]:
        return "rejects: the claim holds"
    if v.get("status"):
        return str(v["status"])
    cl = v.get("clauses") or {}
    if cl and all(_finite(c.get("p")) and c["p"] <= ALPHA for c in cl.values()):
        return FAVOURABLE
    return "not rejected: the registered joint claim was not established"


# ====================================================================================== rendering
def _cl_cell(c: Mapping[str, Any]) -> str:
    """raw p, estimate with its two-sided 95 % interval, null boundary (A7.16 (c)); points for success / accuracy."""
    pts = c.get("test") == "NI" or "diff" in c
    est = P._pct(c.get("est"), True) if pts else _num(c.get("est"))
    ci = P._ci(c.get("est_ci")) if pts else _numci(c.get("est_ci"))
    bnd = P._pct(c.get("boundary"), True) if pts else _num(c.get("boundary"))
    return f"p {P._pv(c.get('p'))}, est {est} {ci}, boundary {bnd}"


def _num(v: Any) -> str:
    return "–" if not _finite(v) else f"{float(v):.4g}"


def _numci(c: Any) -> str:
    if not isinstance(c, (list, tuple)) or len(c) != 2 or not all(_finite(x) for x in c):
        return "–"
    return f"[{float(c[0]):.4g}, {float(c[1]):.4g}]"


def to_markdown(res: Mapping[str, Any]) -> str:
    sim = " (SIMULATED)" if res["meta"]["simulated"] else ""
    L = [f"# Stage 2{sim}: the confirmatory family and the secondary table (A7.4 as amended)", ""]
    for title, tab in (("Family (Holm, FWER 0.05)", res["family"]),
                       ("Secondary (Holm within, six rows; A7.9 (c), A7.12 (g), A7.18 (f))", res["secondary"])):
        L += [f"## {title}", "", "| test | arm → reference | n′ | clauses | decisive p | Holm α | verdict |",
              "|---|---|---|---|---|---|---|"]
        for k, v in tab.items():
            cl = "; ".join(f"{c}: {_cl_cell(x)} ({x.get('reading', '')})" for c, x in (v.get("clauses") or {}).items())
            refs = v.get("reference") or ", ".join(v.get("references") or [])
            L.append(f"| {k} | `{v.get('arm', '–')}` → `{refs or '–'}` | {v.get('n_pairs', '–')} | {cl or '–'} | "
                     f"{P._pv(v['p_decisive'])} | {P._pv(v['holm'].get('alpha_adj'))} | {v['verdict']} |")
        L.append("")
    s23 = res["family"]["S2-3"]
    if s23.get("corpus_summary"):
        sm = s23["corpus_summary"]
        L += ["## S2-3: the common-state corpus (A7.10 (c))", "",
              f"{sm.get('occurrences')} occurrences, {sm.get('unique_questions')} unique questions, "
              f"{sm.get('eligible_tasks')} eligible tasks ({sm.get('excluded')} excluded); source weights "
              f"{json.dumps({k: round(v, 3) for k, v in (sm.get('source_weights') or {}).items()})}.", "",
              "| stratum | occurrences | tasks | " + " | ".join(S23_BACKENDS) + " |", "|---|---|---|" + "---|" * 3]
        for nm, r in (s23.get("strata") or {}).items():
            L.append(f"| {nm} | {r['occurrences']} | {r['tasks']} | "
                     + " | ".join(P._pct(r.get(b)) for b in S23_BACKENDS) + " |")
        if s23.get("sensitivity_worst_case"):
            L += ["", f"Worst-case sensitivity (excluded tasks: JevAgentRL 0, comparators 1): decisive p "
                      f"{P._pv(s23['sensitivity_worst_case']['p_decisive'])} at n = {s23['sensitivity_worst_case']['n']}."]
        L.append("")
    d = res["descriptive"]
    rows = [(f"LOGO without {g}", r) for g, r in (d.get("logo") or {}).items()]
    bc = d.get("best_control") or {}
    rows += [(f"best_control ({bc.get('selected')}) vs {k[3:]}", bc[k]) for k in ("vs_llm_only", "vs_jevagent_best")
             if isinstance(bc.get(k), dict)]
    if d.get("llm_only_xsmall"):
        rows.append(("llm_only_xsmall vs llm_only_x", d["llm_only_xsmall"]))
    if rows:
        L += ["## Descriptive rows (no test)", "", "| row | n′ | Δ success [Newcombe 95 %] | savings [95 %] | "
              "Δ latency s | S1 / S2 calls (arm) |", "|---|---|---|---|---|---|"]
        for nm, r in rows:
            if r.get("status"):
                L.append(f"| {nm} | – | {r['status']} | | | |")
                continue
            c = r.get("cost") or {}
            L.append(f"| {nm} | {r['n_pairs']} | {P._pct(r['delta'], True)} {P._ci(r['delta_ci'])} | "
                     f"{P._pct(c.get('cost_savings'))} % {P._ci(c.get('cost_savings_ci'))} | {_num(r['latency_diff_s'])} | "
                     f"{_num(r['calls_arm']['s1_calls'])} / {_num(r['calls_arm']['s2_calls'])} |")
        if bc.get("status"):
            L.append(f"\nbest_control: {bc['status']}")
        L.append("")
    rl = {k: r for k, r in (d.get("stage2_rl") or {}).items() if len(r) > 3}
    if rl:
        L += ["## JevAgentRL attribution (A7.13; descriptive)", "",
              "| row | acc on C | acc on C, after off-plan | ECE | live success (first 100) | own-state acc | E1 acc | status |",
              "|---|---|---|---|---|---|---|---|"]
        for k, r in rl.items():
            L.append(f"| {k} | {P._pct(r.get('corpus_acc'))} | {P._pct(r.get('corpus_acc_offplan'))} | "
                     f"{_num(r.get('ece'))} | {P._pct(r.get('live_success'))} | {P._pct(r.get('own_acc'))} | "
                     f"{P._pct(r.get('l1_acc'))} | {r.get('status') or ''} |")
        L.append("")
    lad = {k: r for k, r in (d.get("stage2_ladder") or {}).items() if "corpus_acc" in r}
    if lad:
        L += ["## StartLux-Decision size ladder on C (A7.18 (e) 2; descriptive)", "",
              "| row | acc on C (Δ vs Jev [bootstrap]) | acc on C, after off-plan | ECE |", "|---|---|---|---|"]
        for k, r in lad.items():
            vs = r.get("vs_jev") or {}
            L.append(f"| {k} | {P._pct(r.get('corpus_acc'))}" + (f" ({P._pct(vs.get('corpus_delta'), True)} "
                                                                    f"{P._ci(vs.get('corpus_delta_ci'))})" if vs else "")
                     + f" | {P._pct(r.get('corpus_acc_offplan'))} | {_num(r.get('ece'))} |")
        L.append("")
    s1 = d.get("stage2_s1") or {}
    s1rows = {k: r for k, r in (s1.get("rows") or {}).items() if len(r) > 3 or r.get("status")}
    if s1rows:
        L += ["## Static vs in the loop (A7.18 (h); descriptive)", "",
              "| row | E1 acc (Δ vs Jev [Newcombe]) | acc on C (Δ vs Jev [bootstrap]) | live success, first 200 "
              "(Δ vs Jev [Newcombe]) | own-state acc | status |", "|---|---|---|---|---|---|"]
        for k, r in s1rows.items():
            vs = r.get("vs_jev") or {}

            def cell(v: str, dk: str, r=r, vs=vs) -> str:  # noqa: ANN001
                return f"{P._pct(r.get(v))}" + (f" ({P._pct(vs.get(dk), True)} {P._ci(vs.get(dk + '_ci'))})"
                                                 if dk in vs else "")
            L.append(f"| {k} | {cell('l1_acc', 'e1_delta')} | {cell('corpus_acc', 'corpus_delta')} | "
                     f"{cell('live_success', 'live_delta')} | {P._pct(r.get('own_acc'))} | {r.get('status', '')} |")
        kd = s1.get("kendall") or {}
        L += ["", f"Kendall's τ-b of the rank orders over {', '.join(kd.get('rows') or []) or 'no complete row'}: "
                  f"E1 vs C {_num(kd.get('e1_vs_corpus'))}, E1 vs live {_num(kd.get('e1_vs_live'))}. {s1.get('note', '')}",
              ""]
    L += _v2_markdown(res)
    iv = d.get("intervention") or {}
    if iv.get("arms"):
        L += ["## Intervention reporting (A7.16 (d); descriptive)", "",
              "| arm | n | success | tool cov. | arg cov. | S2 calls / ep. | calls avoided | trigger questions | fires | "
              "fire rate | tasks with a fire |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in iv["arms"].items():
            L.append(f"| {k} | {r['n']} | {P._pct(r.get('success'))} | {P._pct(r.get('tool_coverage'))} | "
                     f"{P._pct(r.get('arg_coverage'))} | {_num(r.get('llm_calls'))} | {_num(r.get('llm_calls_avoided'))} | "
                     f"{r['trigger_questions']} | {r['trigger_fires']} | {P._pct(r.get('trigger_fire_rate'))} | "
                     f"{r['trigger_tasks']} |")
        jf, f2 = iv.get("jev_profile_failures") or {}, iv.get("first200") or {}
        if jf:
            L += ["", f"Jev profile (S2-1 arm): {jf['n']} failures, the trigger fired in {jf['with_trigger_fired']} of them."]
        if f2.get("n"):
            L += [f"First {f2['n']} fresh seeds, successes: " + ", ".join(f"{a} {v}" for a, v in f2["successes"].items())
                  + f"; interaction contrast (both - cascade) - (trigger - LLM-only) {P._pct(f2.get('interaction'), True)} "
                  f"{P._ci(f2.get('interaction_ci'))} (paired task bootstrap; descriptive, not a test)."]
        L.append("")
    if res["flags"]:
        L += ["## Flags", ""] + [f"- {f['level']} {f['where']}: {f['msg']}" for f in res["flags"]] + [""]
    return "\n".join(L)


def _v2_markdown(res: Mapping[str, Any]) -> list[str]:
    """A7.19 (f): the JevAgentRL-v2 section (only when a v2 input exists), beside JevAgentRL's rows."""
    d = res["descriptive"]
    v2 = d.get("stage2_v2")
    if not v2:
        return []
    L = [f"## JevAgentRL-v2 — {v2['label']}: descriptive, outside every Holm table (A7.19 (f))", "",
         "JevAgentRL (S2-3, S2-4) is A7.10's checkpoint; v2 is the exploratory Miles track (freeze record "
         f"{json.dumps(v2.get('freeze') or {})}). No test.", ""]
    if v2.get("status"):
        return L + [str(v2["status"]), ""]
    fr = ((d.get("stage2_s1") or {}).get("rows") or {}).get("frozen") or {}
    L += ["| row | acc on C (Δ vs Jev [bootstrap]) | acc on C, after off-plan | ECE | live success, first 200 "
          "(Δ vs Jev [Newcombe]) | own-state acc | steps |", "|---|---|---|---|---|---|---|"]
    for nm, r in (("JevAgentRL (A7.10)", fr), (f"JevAgentRL-v2, {v2['label']}", v2.get("row") or {})):
        vs = r.get("vs_jev") or {}

        def cell(v: str, dk: str, r=r, vs=vs) -> str:  # noqa: ANN001
            return f"{P._pct(r.get(v))}" + (f" ({P._pct(vs.get(dk), True)} {P._ci(vs.get(dk + '_ci'))})"
                                             if dk in vs else "")
        L.append(f"| {nm} | {cell('corpus_acc', 'corpus_delta')} | {P._pct(r.get('corpus_acc_offplan'))} | "
                 f"{_num(r.get('ece'))} | {cell('live_success', 'live_delta')} | {P._pct(r.get('own_acc'))} | "
                 f"{_num(r.get('steps'))} |")
    pr = v2.get("profile") or {}
    L += ["", "| profile row → `llm_only` | n′ | success arm / ref | Δ success [Newcombe 95 %] | cost savings incl. "
              "tariff [95 %] | q $ / episode |", "|---|---|---|---|---|---|"]
    s24 = (res.get("family") or {}).get("S2-4") or {}
    ni = (s24.get("clauses") or {}).get("ni") or {}
    if s24.get("n_pairs"):
        c = s24.get("cost") or {}
        L.append(f"| `jevagent_best_rl` (S2-4, A7.10) | {s24['n_pairs']} | {P._pct(ni.get('success_arm'))} / "
                 f"{P._pct(ni.get('success_ref'))} | {P._pct(ni.get('est'), True)} {P._ci(ni.get('newcombe95'))} | "
                 f"{P._pct(c.get('cost_savings'))} % {P._ci(c.get('cost_savings_ci'))} | "
                 f"{_num((s24.get('tariff') or {}).get('q_usd_per_episode'))} |")
    if pr.get("status"):
        L.append(f"| `{pr.get('arm')}` ({v2['label']}) | – | {pr['status']} | | | |")
    else:
        c = pr.get("cost") or {}
        L.append(f"| `{pr['arm']}` ({v2['label']}) | {pr['n_pairs']} | {P._pct(pr.get('success_arm'))} / "
                 f"{P._pct(pr.get('success_ref'))} | {P._pct(pr.get('delta'), True)} {P._ci(pr.get('delta_ci'))} | "
                 f"{P._pct(c.get('cost_savings'))} % {P._ci(c.get('cost_savings_ci'))} | "
                 f"{_num((pr.get('tariff') or {}).get('q_usd_per_episode'))} |")
        sens = "; ".join(f"{k} (q {_num(x.get('q'))}): {P._pct(x.get('cost_savings'))} % {P._ci(x.get('cost_savings_ci'))}"
                         for k, x in (pr.get("tariff_sensitivity") or {}).items())
        L += ["", f"v2 tariff sensitivity (cost savings): {sens or '–'}. Δ latency {_num(pr.get('latency_diff_s'))} s "
                  f"{_numci(pr.get('latency_diff_s_ci'))} (descriptive; serving host and queueing)."]
    return L + [""]


CLAUSE_UNIT = {"ni": "spct", "intern": "spct", "jev": "spct", "cost": "usd", "latency": "s"}   # est / est_ci / boundary


def _tex_fmt(kind: str, v: Any) -> str | None:
    if kind == "text":
        from jevharness.bench.paper_numbers import _tex_text
        return _tex_text(str(v)) if v not in (None, "") else None
    if not _finite(v):
        return None
    v = float(v)
    if kind == "p":
        return "$<$0.001" if v < 0.001 else f"{v:.3f}"
    if kind == "spct":
        return f"{100 * v:+.1f}"
    if kind == "pct":
        return f"{100 * v:.1f}"
    if kind == "int":
        return f"{v:.0f}"
    if kind == "s":
        return "0" if v == 0 else f"{v:+.2f}"
    if kind == "f3":
        return f"{v:.3f}"
    if kind == "f2":
        return f"{v:.2f}"
    if kind == "f2s":
        return f"{v:+.2f}"
    if kind == "intc":                                   # an integer count with a thousands separator for the paper
        return f"{v:,.0f}".replace(",", "{,}")
    return np.format_float_positional(v, precision=3, unique=True, fractional=False, trim="-")   # usd / other


def _tex_rows(res: Mapping[str, Any]) -> dict[tuple[str, ...], tuple[str, Any]]:
    """(key path) -> (format kind, value) for every Stage-2 number the paper may cite (docs/JEVAGENT_HARNESS.md §6
    lists the schema):

    * ``stage2/<test>/{delta, delta_ci/lo|hi, cost_savings, cost_savings_ci/lo|hi, p_decisive, p_holm, alpha_adj,
      holm_rank, decision, n_pairs, n_expected, n_dropped}`` for the nine tests (``delta``: success difference arm -
      reference in points; S2-3: JevAgentRL - Intern-Decision accuracy in points, ``delta_jev`` the Jev clause;
      S2-1T: latency saving reference - arm in seconds);
    * ``stage2/<test>/<clause>/{est, est_ci/lo|hi, boundary, p, decision}``, clauses ``ni`` (points), ``cost`` ($ per
      episode saved), ``latency`` (s per episode saved), ``intern`` / ``jev`` (accuracy points; ``jev``'s boundary
      is -5);
    * ``stage2_rl/<row>/{corpus_acc, corpus_acc_offplan, ece, live_success, own_acc, l1_acc}`` (A7.13);
    * ``stage2/scorecard/<arm>/{success, success_ci, cost_usd, latency_s, llm_calls, s1_calls, n,
      vs_baseline/{delta, delta_ci, cost_savings, cost_savings_ci}}``;
    * ``stage2/descriptive/<row>/...`` (LOGO, best_control, llm_only_xsmall, jevagent_default);
    * ``stage2_v2/...`` (A7.19 (f), exploratory; only when a v2 input exists): ``label``, ``status``, the corpus and
      live measures with ``vs_jev/...``, and ``profile/{n_pairs, success_arm, success_ref, delta, delta_ci,
      cost_savings, cost_savings_ci, tariff/q, tariff/<sensitivity>/..., latency_diff_s, tool_coverage, ...}``."""
    rows: dict[tuple[str, ...], tuple[str, Any]] = {}

    def put(path: tuple[str, ...], kind: str, v: Any) -> None:
        rows[path] = (kind, v)

    def ci(path: tuple[str, ...], kind: str, c: Any) -> None:
        if isinstance(c, (list, tuple)) and len(c) == 2:
            put(path + ("lo",), kind, c[0])
            put(path + ("hi",), kind, c[1])

    for table in ("family", "secondary"):
        for lab, v in res[table].items():
            b = ("stage2", lab)
            h = v.get("holm") or {}
            put(b + ("p_decisive",), "p", v.get("p_decisive"))
            put(b + ("p_holm",), "p", h.get("p_holm"))
            put(b + ("alpha_adj",), "p", h.get("alpha_adj"))
            put(b + ("holm_rank",), "int", h.get("rank"))
            put(b + ("decision",), "text", v.get("verdict"))
            for k in ("n_pairs", "n_expected", "n_dropped", "n_eligible", "n_excluded"):
                put(b + (k,), "int", v.get(k))
            cl = v.get("clauses") or {}
            for cn, c in cl.items():
                cb, unit = b + (cn,), CLAUSE_UNIT.get(cn, "usd")
                put(cb + ("est",), unit, c.get("est"))
                ci(cb + ("est_ci",), unit, c.get("est_ci"))
                put(cb + ("boundary",), unit, c.get("boundary"))
                put(cb + ("p",), "p", c.get("p"))
                put(cb + ("decision",), "text", c.get("reading"))
                if cn == "ni":
                    ci(cb + ("est_ci_newcombe",), "spct", c.get("newcombe95"))
                    for k in ("success_arm", "success_ref", "discordance", "loss_rate_ub"):
                        put(cb + (k,), "pct", c.get(k))
                    for k in ("wins", "losses"):
                        put(cb + (k,), "int", c.get(k))
                elif cn in ("intern", "jev"):
                    ci(cb + ("est_ci_boot",), "spct", c.get("diff_ci"))
            if "ni" in cl:
                put(b + ("delta",), "spct", cl["ni"].get("est"))
                ci(b + ("delta_ci",), "spct", cl["ni"].get("est_ci"))
            elif "intern" in cl:
                put(b + ("delta",), "spct", cl["intern"].get("est"))
                ci(b + ("delta_ci",), "spct", cl["intern"].get("est_ci"))
                put(b + ("delta_jev",), "spct", cl["jev"].get("est"))
                ci(b + ("delta_jev_ci",), "spct", cl["jev"].get("est_ci"))
            elif "latency" in cl:
                put(b + ("delta",), "s", cl["latency"].get("est"))
                ci(b + ("delta_ci",), "s", cl["latency"].get("est_ci"))
                if _finite(v.get("latency_ref_s")) and v["latency_ref_s"] > 0:
                    put(b + ("latency_savings",), "pct", 1.0 - v["latency_arm_s"] / v["latency_ref_s"])
            cost = v.get("cost") or {}
            put(b + ("cost_savings",), "pct", cost.get("cost_savings"))
            ci(b + ("cost_savings_ci",), "pct", cost.get("cost_savings_ci"))
            for k in ("cost_arm", "cost_ref"):
                put(b + (k,), "usd", cost.get(k))
            for k in ("latency_arm_s", "latency_ref_s"):
                put(b + (k,), "usd", v.get(k))
            for nm, a in (v.get("accuracy") or {}).items():
                put(b + ("accuracy", nm), "pct", a)
            for k, x in (v.get("corpus_summary") or {}).items():
                if k in ("occurrences", "unique_questions", "eligible_tasks", "excluded", "tasks"):
                    put(b + ("corpus", k), "int", x)
            wc = v.get("sensitivity_worst_case")
            if wc:
                put(b + ("worst_case", "p_decisive"), "p", wc.get("p_decisive"))
                put(b + ("worst_case", "n"), "int", wc.get("n"))
            for nm, s in (v.get("tariff_sensitivity") or {}).items():
                put(b + ("tariff", nm, "q"), "usd", s.get("q"))
                put(b + ("tariff", nm, "cost_savings"), "pct", s.get("cost_savings"))
                put(b + ("tariff", nm, "p_cost"), "p", s.get("p_cost"))
            if v.get("tariff"):
                put(b + ("tariff", "q"), "usd", v["tariff"].get("q_usd_per_episode"))
    for table in ("family", "secondary"):                              # A6.22 (a) coverage of each test's arm
        for lab, v in res[table].items():
            cv = v.get("coverage") or {}
            for k in ("tool_coverage", "arg_coverage"):
                put(("stage2", lab, "coverage", k), "pct", cv.get(k))
    for st, r in ((res["family"].get("S2-3") or {}).get("strata") or {}).items():   # A7.10 (c) robustness strata
        for k in ("rl", "intern", "jev"):
            put(("stage2", "S2-3", "strata", st, k), "pct", r.get(k))
        for k in ("occurrences", "tasks"):
            put(("stage2", "S2-3", "strata", st, k), "intc", r.get(k))
    d = res["descriptive"]
    for g, r in (d.get("selection") or {}).items():                  # A7.1: what each selected group qualified on
        b = ("stage2", "selection", g)
        put(b + ("arm",), "text", r.get("arm"))
        put(b + ("n",), "int", r.get("n"))
        put(b + ("savings",), "pct", r.get("savings"))
        ci(b + ("savings_ci",), "pct", r.get("savings_ci"))
        put(b + ("wins",), "int", r.get("wins"))
        put(b + ("losses",), "int", r.get("losses"))
    iv = d.get("intervention") or {}                                   # A7.16 (d), descriptive
    for arm, r in (iv.get("arms") or {}).items():
        b = ("stage2", "intervention", arm)
        put(b + ("n",), "int", r.get("n"))
        put(b + ("success",), "pct", r.get("success"))
        put(b + ("tool_coverage",), "pct", r.get("tool_coverage"))
        put(b + ("arg_coverage",), "pct", r.get("arg_coverage"))
        put(b + ("llm_calls",), "f2", r.get("llm_calls"))
        put(b + ("llm_calls_avoided",), "f2s", r.get("llm_calls_avoided"))
        for k in ("trigger_questions", "trigger_fires", "trigger_tasks", "tool_decisions"):
            put(b + (k,), "intc", r.get(k))
        put(b + ("trigger_fire_rate",), "pct", r.get("trigger_fire_rate"))
    jf = iv.get("jev_profile_failures") or {}
    put(("stage2", "intervention", "jevagent_best", "failures"), "int", jf.get("n"))
    put(("stage2", "intervention", "jevagent_best", "failures_fired"), "int", jf.get("with_trigger_fired"))
    f2 = iv.get("first200") or {}
    if f2.get("n"):
        b = ("stage2", "intervention", "first200")
        put(b + ("n",), "int", f2["n"])
        for a_, v_ in (f2.get("successes") or {}).items():
            put(b + ("successes", a_), "int", v_)
        put(b + ("interaction",), "spct", f2.get("interaction"))
        ci(b + ("interaction_ci",), "spct", f2.get("interaction_ci"))
    drows = {f"logo-{g}": r for g, r in (d.get("logo") or {}).items()}
    bc = d.get("best_control") or {}
    drows.update({f"best_control-{k[3:]}": bc[k] for k in ("vs_llm_only", "vs_jevagent_best")
                  if isinstance(bc.get(k), dict)})
    if d.get("llm_only_xsmall"):
        drows["llm_only_xsmall"] = d["llm_only_xsmall"]
    for nm, r in drows.items():
        b = ("stage2", "descriptive", nm)
        if r.get("status"):
            put(b + ("status",), "text", r["status"])
            continue
        put(b + ("n_pairs",), "int", r.get("n_pairs"))
        put(b + ("delta",), "spct", r.get("delta"))
        ci(b + ("delta_ci",), "spct", r.get("delta_ci"))
        ci(b + ("delta_ci_tango",), "spct", r.get("delta_ci_tango"))
        put(b + ("success_arm",), "pct", r.get("success_arm"))
        put(b + ("success_ref",), "pct", r.get("success_ref"))
        c = r.get("cost") or {}
        put(b + ("cost_savings",), "pct", c.get("cost_savings"))
        ci(b + ("cost_savings_ci",), "pct", c.get("cost_savings_ci"))
        put(b + ("latency_diff_s",), "s", r.get("latency_diff_s"))
        ci(b + ("latency_diff_s_ci",), "s", r.get("latency_diff_s_ci"))
    jd = d.get("jevagent_default") or {}
    if jd.get("cost"):
        b = ("stage2", "descriptive", "jevagent_default")
        put(b + ("cost_savings",), "pct", jd["cost"].get("cost_savings"))
        ci(b + ("cost_savings_ci",), "pct", jd["cost"].get("cost_savings_ci"))
        ni = (jd.get("clauses") or {}).get("ni") or {}
        put(b + ("delta",), "spct", ni.get("est"))
        ci(b + ("delta_ci",), "spct", ni.get("est_ci"))
    if bc.get("selected"):
        put(("stage2", "descriptive", "best_control", "selected"), "text", bc["selected"])
        put(("stage2", "descriptive", "best_control", "arm"), "text", bc.get("arm"))
    for nm, r in (d.get("scorecard") or {}).items():
        b = ("stage2", "scorecard", nm)
        put(b + ("success",), "pct", r.get("success"))
        ci(b + ("success_ci",), "pct", r.get("success_ci"))
        put(b + ("cost_usd",), "usd", r.get("cost_usd"))
        put(b + ("latency_s",), "usd", r.get("latency_s"))
        put(b + ("llm_calls",), "usd", r.get("llm_calls"))
        put(b + ("s1_calls",), "usd", r.get("s1_calls"))
        put(b + ("n",), "int", r.get("n"))
        vb = r.get("vs_baseline") or {}
        put(b + ("vs_baseline", "delta"), "spct", vb.get("delta"))
        ci(b + ("vs_baseline", "delta_ci"), "spct", vb.get("delta_ci"))
        put(b + ("vs_baseline", "cost_savings"), "pct", vb.get("cost_savings"))
        ci(b + ("vs_baseline", "cost_savings_ci"), "pct", vb.get("cost_savings_ci"))
    for row, r in (d.get("stage2_rl") or {}).items():
        b = ("stage2_rl", row)
        for k in ("corpus_acc", "corpus_acc_offplan", "live_success", "own_acc", "l1_acc"):
            put(b + (k,), "pct", r.get(k))
        put(b + ("ece",), "f3", r.get("ece"))
        for k in ("corpus_tasks", "live_n", "l1_n"):
            put(b + (k,), "int", r.get(k))
    s1 = d.get("stage2_s1") or {}
    for row, r in (s1.get("rows") or {}).items():                     # A7.18 (h): static vs in the loop
        b = ("stage2_s1", row)
        for k in ("corpus_acc", "corpus_acc_offplan", "live_success", "own_acc", "l1_acc"):
            put(b + (k,), "pct", r.get(k))
        put(b + ("ece",), "f3", r.get("ece"))
        for k in ("corpus_tasks", "live_n", "l1_n"):
            put(b + (k,), "int", r.get(k))
        if r.get("status"):
            put(b + ("status",), "text", r["status"])
        vs = r.get("vs_jev") or {}
        for k in ("e1", "live", "corpus"):
            put(b + ("vs_jev", f"{k}_delta"), "spct", vs.get(f"{k}_delta"))
            ci(b + ("vs_jev", f"{k}_delta_ci"), "spct", vs.get(f"{k}_delta_ci"))
            put(b + ("vs_jev", f"{k}_n"), "int", vs.get(f"{k}_n"))
    for k in ("e1_vs_corpus", "e1_vs_live"):
        put(("stage2_s1", "kendall", k), "f3", (s1.get("kendall") or {}).get(k))
    for row, r in (d.get("stage2_ladder") or {}).items():             # A7.18 (e) 2: the size ladder on C
        b = ("stage2_ladder", row)
        for k in ("corpus_acc", "corpus_acc_offplan"):
            put(b + (k,), "pct", r.get(k))
        put(b + ("ece",), "f3", r.get("ece"))
        put(b + ("corpus_tasks",), "int", r.get("corpus_tasks"))
        vs = r.get("vs_jev") or {}
        put(b + ("vs_jev", "corpus_delta"), "spct", vs.get("corpus_delta"))
        ci(b + ("vs_jev", "corpus_delta_ci"), "spct", vs.get("corpus_delta_ci"))
    v2 = d.get("stage2_v2")                                            # A7.19 (f): only when a v2 input exists
    if v2:
        b = ("stage2_v2",)
        put(b + ("label",), "text", v2.get("label"))
        if v2.get("status"):
            put(b + ("status",), "text", v2["status"])
        r = v2.get("row") or {}
        for k in ("corpus_acc", "corpus_acc_offplan", "live_success", "own_acc"):
            put(b + (k,), "pct", r.get(k))
        put(b + ("ece",), "f3", r.get("ece"))
        put(b + ("steps",), "f2", r.get("steps"))
        for k in ("corpus_tasks", "live_n"):
            put(b + (k,), "int", r.get(k))
        vs = r.get("vs_jev") or {}
        for k in ("live", "corpus"):
            put(b + ("vs_jev", f"{k}_delta"), "spct", vs.get(f"{k}_delta"))
            ci(b + ("vs_jev", f"{k}_delta_ci"), "spct", vs.get(f"{k}_delta_ci"))
            put(b + ("vs_jev", f"{k}_n"), "int", vs.get(f"{k}_n"))
        pr = v2.get("profile") or {}
        pb = b + ("profile",)
        if pr.get("status"):
            put(pb + ("status",), "text", pr["status"])
        for k in ("n_pairs", "n_expected", "n_dropped"):
            put(pb + (k,), "int", pr.get(k))
        for k in ("success_arm", "success_ref"):
            put(pb + (k,), "pct", pr.get(k))
        put(pb + ("delta",), "spct", pr.get("delta"))
        ci(pb + ("delta_ci",), "spct", pr.get("delta_ci"))
        ci(pb + ("delta_ci_tango",), "spct", pr.get("delta_ci_tango"))
        c = pr.get("cost") or {}
        put(pb + ("cost_savings",), "pct", c.get("cost_savings"))
        ci(pb + ("cost_savings_ci",), "pct", c.get("cost_savings_ci"))
        for k in ("cost_arm", "cost_ref"):
            put(pb + (k,), "usd", c.get(k))
        put(pb + ("tariff", "q"), "usd", (pr.get("tariff") or {}).get("q_usd_per_episode"))
        for nm, x in (pr.get("tariff_sensitivity") or {}).items():
            put(pb + ("tariff", nm, "q"), "usd", x.get("q"))
            put(pb + ("tariff", nm, "cost_savings"), "pct", x.get("cost_savings"))
            ci(pb + ("tariff", nm, "cost_savings_ci"), "pct", x.get("cost_savings_ci"))
        put(pb + ("latency_diff_s",), "s", pr.get("latency_diff_s"))
        ci(pb + ("latency_diff_s_ci",), "s", pr.get("latency_diff_s_ci"))
        iv = pr.get("intervention") or {}
        for k in ("tool_coverage", "arg_coverage"):
            put(pb + (k,), "pct", iv.get(k))
        put(pb + ("llm_calls",), "f2", iv.get("llm_calls"))
    return rows


def to_tex(res: Mapping[str, Any]) -> str:
    """``\\defres{stage2/<test>/<...>}{value}`` and ``\\defres{stage2_rl/<row>/<...>}{value}`` with the paper_numbers /
    primaries preamble (either file can be \\input first). Simulated results go under ``SIM/``; a value that does not
    exist is not written (``\\res`` then prints ``\\tbd``)."""
    from jevharness.bench.paper_numbers import _seg

    meta = res["meta"]
    prefix = ("SIM",) if meta["simulated"] else ()
    vals: dict[str, str] = {}
    for path, (kind, v) in _tex_rows(res).items():
        s = _tex_fmt(kind, v)
        if s is not None:
            vals["/".join(prefix + tuple(_seg(x) for x in path))] = s
    lines = ["% AUTO-GENERATED by scripts/stage2_analysis.py -- do not edit by hand.",
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
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--run", action="append", metavar="KEY=DIR")
    ap.add_argument("--profile", default=str(PROFILE))
    ap.add_argument("--profile-x", default=str(PROFILE_X))
    ap.add_argument("--s23", default=str(S23_DIR), help="S2-3's corpus directory (scripts/stage2_corpus.py)")
    ap.add_argument("--freeze-record", default=str(FREEZE_RECORD))
    ap.add_argument("--tariff", default=str(TARIFF))
    ap.add_argument("--tariff-sl", default=str(TARIFF_SL), help="S2-4S's frozen tariff (A7.18 (e) 5)")
    ap.add_argument("--tariff-v2", default=str(TARIFF_V2), help="the v2 profile row's frozen tariff (A7.19 (f))")
    ap.add_argument("--v2-freeze-record", default=str(V2_FREEZE_RECORD), help="JevAgentRL-v2's freeze record (A7.19 (e))")
    ap.add_argument("--startlux-manifest", default=str(STARTLUX_MANIFEST), help="A7.18 (d) serving manifest")
    ap.add_argument("--control-selection", default=str(CONTROL_SELECTION))
    ap.add_argument("--profile-selection", default=None,
                    help="A7.1 (h) selection record (default: results/stage2/profile_selection.json with the default profile)")
    ap.add_argument("--telecom-ids", default=str(TELECOM_IDS))
    ap.add_argument("--out", default="results/stage2")
    ap.add_argument("--tex", default=str(TEX_OUT), help="the paper's \\defres file (also written to <out>/stage2.tex)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-simulated", action="store_true")
    ap.add_argument("--allow-dev-tasks", action="store_true")
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    an = Analysis(a)
    res = an.run()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slim = P._strip(res)
    (out / "stage2.json").write_text(json.dumps(P._jsonable(slim), indent=1, default=P._json_default))
    (out / "stage2.md").write_text(to_markdown(res))
    tex = to_tex(res)
    (out / "stage2.tex").write_text(tex)
    if a.tex:
        Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
        Path(a.tex).write_text(tex)
    errors = sum(1 for f in res["flags"] if f["level"] == "ERROR")
    rej = [k for k, v in res["family"].items() if v["holm"]["reject"]]
    print(f"[stage2] {len(rej)} of 4 family tests reject: {', '.join(rej) or 'none'}; {errors} ERROR flag(s) -> {out}",
          file=sys.stderr)
    return 1 if (a.strict and errors) else 0


if __name__ == "__main__":
    sys.exit(main())
