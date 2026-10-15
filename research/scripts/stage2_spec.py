#!/usr/bin/env python3
"""The effective Stage-2 specification and its launch gate (docs/PREREG_A7_STAGE2.md A7.17 (c)).

    python scripts/stage2_spec.py                    # rewrite docs/STAGE2_EFFECTIVE.md
    python scripts/stage2_spec.py --check            # exit 1 when docs/STAGE2_EFFECTIVE.md has drifted (header aside)
    python scripts/stage2_spec.py --gate stage2_jevgym [--rl] [--startlux] [--skip-tests]   # launch gate (exit 1 = no)
    python scripts/stage2_spec.py --gate stage2_v2 | stage2_v2_states | jevgym_shadow_cal_v2  # A7.19 (f), exploratory

One executable view of Stage 2 as amended (A7, A7.9-A7.17): every test with its arms, run, population, endpoint,
estimator, margin, table and not-tested conditions; every run with its arms, task counts, System-1 bindings, profile
parts, schedule and seeds (read from the experiment YAMLs through jevharness.bench.runner); the frozen populations with
their hashes; the failure rules; the resource gates; the prerequisites of each run. docs/PREREG_A7_STAGE2.md stays the
authoritative, append-only record; where the two disagree, PREREG_A7_STAGE2.md wins and this script is wrong.
tests/test_stage2_spec.py regenerates the file and fails when the committed copy has drifted.

The gate (A7.17 (c): "No Stage-2 run launches before its fixtures, the effective specification, A7.10's access
boundary and its cost gate pass") checks, for the named run: the specification is in sync; the Stage-2 fixture tests
pass (unless --skip-tests); and the run's prerequisites (profile selected, JevAgentRL freeze record and training
manifest for every run with a JevAgentRL arm, the task-major schedule, the frozen telecom sample, the X cost-gate
record and prices, the control selection). It prints every check and exits 1 on the first run that fails one."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import stage2_analysis as S2  # noqa: E402
from jevharness.bench import registry as R  # noqa: E402
from jevharness.bench.runner import arm_bindings, arm_profiles, arm_task_counts, expand_arms  # noqa: E402

OUT = ROOT / "docs/STAGE2_EFFECTIVE.md"
GEN_PREFIX = "- Generated:"
CONFIGS = {"stage2_jevgym": "configs/experiments/stage2_jevgym.yaml",
           "stage2_rl_states": "configs/experiments/stage2_rl_states.yaml",
           "stage2_x": "configs/experiments/stage2_x.yaml",
           "jevgym_shadow_cal_x": "configs/experiments/jevgym_shadow_cal_x.yaml",
           "jevgym_shadow_cal_rl": "configs/experiments/jevgym_shadow_cal_rl.yaml",
           "stage2_rl_attribution": "configs/experiments/stage2_rl_attribution.yaml",
           "stage2_alfworld": "configs/experiments/stage2_alfworld.yaml",
           "stage2_startlux_states": "configs/experiments/stage2_startlux_states.yaml",
           "jevgym_shadow_cal_startlux": "configs/experiments/jevgym_shadow_cal_startlux.yaml",
           # A7.19 (f): the JevAgentRL-v2 rows (exploratory, descriptive; results/stage2/v2_eval_plan.json)
           "jevgym_shadow_cal_v2": "configs/experiments/jevgym_shadow_cal_v2.yaml",
           "stage2_v2_states": "configs/experiments/stage2_v2_states.yaml",
           "stage2_v2": "configs/experiments/stage2_v2.yaml",
           # A7.12 (i): the optional closed-family row (descriptive; Deviations 2026-10-05, scripts/sol_eligibility.py)
           "jevgym_shadow_cal_sol": "configs/experiments/jevgym_shadow_cal_sol.yaml",
           "stage2_sol": "configs/experiments/stage2_sol.yaml"}
FIXTURE_TESTS = ["tests/test_stage2_analysis.py", "tests/test_stage2_corpus.py", "tests/test_stage2_amendments.py",
                 "tests/test_telecom_stage2.py", "tests/test_select_profile.py", "tests/test_profile.py",
                 "tests/test_startlux.py", "tests/test_stage2_v2.py", "tests/test_stage2_sol.py"]
TRAINING_MANIFEST = ROOT / "results/gpu_round5/TRAINING_MANIFEST.md"
V2_TRAINING_MANIFEST = ROOT / "results/gpu_round6_miles/TRAINING_MANIFEST.md"   # A7.19 (c)
V2_PLAN = ROOT / "results/stage2/v2_eval_plan.json"                            # A7.19 (f): serving and row plan
V2_FIT_ARM = "v2_shadow_cal"                                                   # jevgym_shadow_cal_v2.yaml
X_COST_GATE = ROOT / "results/stage2/x_cost_gate.json"
X_MODELS = ("deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash")
SOL_COST_GATE = ROOT / "results/stage2/sol_cost_gate.json"                     # A7.12 (i) with (c)
SOL_MANIFEST = ROOT / "results/stage2/sol_capability_manifest.json"
SOL_MODELS = ("gpt-6.1-sol", "gpt-6-luna")

# ---------------------------------------------------------------------------------------------- the tests (hand-written)
# (label, table, arm -> reference, run, population, endpoint and clauses, defined by, not tested when)
TESTS = [
    ("S2-1", "family", "`jevagent_best` → `llm_only`", "stage2_jevgym", "Stage-2 list, n = 500",
     "NI on success (m = 3 pt, `bb_ordered`) ∧ cost (paired t)", "A7.4, A7.9 (a)",
     "the profile's Flash section is empty / skipped (A7.1 (g) 5)"),
    ("S2-2", "family", "`jevagent_best` → `best_single`", "stage2_jevgym", "Stage-2 list, n = 500",
     "NI on success (m = 3 pt) ∧ cost; reference = the profile restricted to g*", "A7.4, A7.11 (a), (b)",
     "single-group profile (A7.1 (g) 4) or S2-1 not tested"),
    ("S2-3", "family", "JevAgentRL → Intern-Decision-4B; → Jev", "stage2_rl_states + corpus",
     "first 200 Stage-2 tasks, pooled common states", "(i) paired t on acc_RL − acc_Intern; (ii) paired t on "
     "acc_RL − acc_Jev + 0.05", "A7.10 (c), (d)",
     "no freeze record by 2026-10-07 12:00 UTC, or pre-freeze exposure (A7.10 (b), (h); see Deviations)"),
    ("S2-4", "family", "`jevagent_best_rl` → `llm_only`", "stage2_jevgym", "Stage-2 list, n = 500",
     "NI on success (m = 3 pt) ∧ cost with the frozen tariff q per episode", "A7.4, A7.13 (iii)",
     "no freeze record (A7.10 (h)); no frozen tariff or a mismatched workload (A7.13 (iii)); unpriced"),
    ("S2-1T", "secondary", "`jevagent_best` → `llm_only`", "stage2_jevgym", "Stage-2 list, n = 500",
     "paired t on `usage.total.latency_ms` (reference − arm)", "A7.9 (b), A7.16 (a)",
     "S2-1 not tested; p = 1 without a task-major schedule record"),
    ("S2-1L", "secondary", "`jevagent_best_s2lite` → `llm_only_s2lite`", "stage2_jevgym", "Stage-2 list, n = 500",
     "NI on success (m = 3 pt) ∧ cost", "A7.4", "the Flash-Lite section is empty / skipped"),
    ("S2-L3a", "secondary", "`jevagent_best` (τ² rendering) → `llm_only`", "stage2_tau2/telecom",
     "A7.15 telecom sample (n ≤ 800), one attempt", "NI on single-attempt success (m = 5 pt) ∧ cost per task",
     "A7.15 (e)", "all-`off` rendering; adapter validation fails (A7.15 (f))"),
    ("S2-L3b", "secondary", "`jevagent_best` (ALFWorld rendering) → `llm_react`", "stage2_alfworld",
     "ALFWorld valid_seen, 140 games", "NI on success (m = 3 pt) ∧ cost", "A7.4",
     "no action-loop decision point in the profile (A7.1 (i))"),
    ("S2-1X", "secondary", "`jevagent_best_x` → `llm_only_x`", "stage2_x", "Stage-2 list, n = 500 or 300 (A7.12 (c))",
     "NI on success (m = 3 pt) ∧ cost", "A7.12 (g)",
     "resource gate (A7.12 (c)); no eligible pair; the fitted profile delegates nothing on X; unpriced"),
    ("S2-4S", "secondary", "`jevagent_best_sl4b` → `llm_only`", "stage2_jevgym", "Stage-2 list, n = 500",
     "NI on success (m = 3 pt) ∧ cost with StartLux-4B's own frozen tariff q_S per episode", "A7.18 (e) 5, (f)",
     "the 4B server cannot start (A7.18 (g)); thresholds pending or the fit delegates nothing; no tariff; unpriced"),
]
FAILURE_RULES = [
    "Three passes per evaluation command; a task whose latest record is an infrastructure error after three attempts "
    "leaves the pair (drop rule; full-set sensitivity row) (A7.6, A6.22 (e) 4-5).",
    "Agent failures are failures; a missing record is an ERROR flag (incomplete run) (A7.4).",
    "S2-3: a task without four valid source trajectories leaves the corpus (worst-case sensitivity: JevAgentRL 0, "
    "comparators 1); a replay scoring failure is retried twice, then incorrect; invalid options are model failures; "
    "no question leaves C (A7.10 (d)).",
    "A missing or not-tested member enters its Holm table with p = 1 (A7.4).",
    "Simulated inputs and episodes outside the frozen lists are refused by the analysis (A7.6).",
]
RESOURCE_GATES = [
    "S2-1X: n = 500 if the predicted total fits $25, else 300 if it fits, else not tested; decided before the first "
    "calibration episode (A7.12 (c)); record `results/stage2/x_cost_gate.json`.",
    "Telecom: n = min(800, N_eligible − 32, ⌊(250 − V) / (2 × 1.25 × C)⌋), frozen with the sample (A7.15 (c)).",
    "LOGO rows: every active group if |G| × 200 × $0.043 fits $45, else only g* removed; decided when the profile is "
    "written (A7.11 (c)).",
    "`best_control`: n = 500, three passes, ≤ $22 (A7.14); optional `gpt-6.1-sol` rows only if the budget allows "
    "after every registered row (A7.12 (i)): n = 500 if the predicted total fits the row's $80 envelope, else 300, "
    "else not run; decided before its first calibration episode; record `results/stage2/sol_cost_gate.json` "
    "(Deviations 2026-10-05).",
    "JevAgentRL: ≤ 48 L4 hours of training, ≈ 24 L4 hours of serving (A7.10 (h)).",
    "StartLux (A7.18 (g)): 4B on the L4 host; 27B from original weights on one GPU ≥ 64 GiB, else the Q8_0 GGUF on "
    "≥ 40 GiB (\"27B (Q8_0)\"), else not run; chosen by availability before the first 27B request.",
    "JevAgentRL-v2 (A7.19): ≤ $400 GPU and ≤ $100 API; freeze by 2026-10-07 12:00 UTC or no evaluation row.",
]


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "(missing)"


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_table(key: str, cfg: str) -> list[str]:
    exp = yaml.safe_load((ROOT / cfg).read_text())
    counts, binds, profs = arm_task_counts(exp), arm_bindings(exp), arm_profiles(exp)
    s = exp["suite"]
    seeds = (f"seeds {s['seed0']}–{int(s['seed0']) + int(s.get('n', 0)) - 1}" if s.get("seed0") is not None else
             f"{s.get('n')} tasks, split {s.get('split', '–')}")
    sched = (exp.get("schedule") or {}).get("mode", "arm_major")
    ph = sorted(n for n, sp in (exp.get("s1_backends") or {}).items() if isinstance(sp, dict) and sp.get("placeholder"))
    L = [f"### `{key}` (`{cfg}`)", "",
         f"Suite {s.get('name')}: {seeds}; schedule `{sched}`; workers {exp.get('workers')}; System-1 cache "
         f"`{exp.get('s1_cache')}`" + (f"; placeholder backends {', '.join(f'`{p}`' for p in ph)}" if ph else "") + ".",
         "", "| arm | n | System-1 | System-2 | profile part |", "|---|---|---|---|---|"]
    for a in expand_arms(exp):
        b = binds.get(a.name, {})
        pr = profs.get(a.name) or {}
        part = "–" if not pr else f"{Path(str(pr.get('path'))).name}: {pr.get('s2')}" + (
            f" / {pr['part']}" if pr.get("part") else "") + (f" / backend {pr['backend']}" if pr.get("backend") else "") \
            + (f" / env {pr['env']}" if pr.get("env") else "")
        s1 = b.get("s1", "default") if (a.uses_s1() or pr or "s1" in b) else "—"     # profile arms: when delegating
        L.append(f"| `{a.name}` | {counts.get(a.name, s.get('n'))} | {s1} | "
                 f"{b.get('llm', 'main')} | {part} |")
    return L + [""]


def render() -> str:
    head = ["# Stage 2 — effective specification (generated)", "",
            f"{GEN_PREFIX} by `scripts/stage2_spec.py` at commit `{_git_head()}` on "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
            "- Source of truth: `docs/PREREG_A7_STAGE2.md` (A7, A7.9–A7.17, Deviations). Where the two disagree, "
            "PREREG_A7_STAGE2.md wins and this script is wrong.",
            f"- Registration file sha256: `{_sha_file(ROOT / 'docs/PREREG_A7_STAGE2.md')}`.", ""]
    L = head + ["## 1. Tests", "",
                f"Family {', '.join(S2.FAMILY)}: Holm at FWER {S2.ALPHA}. Secondary table {', '.join(S2.SECONDARY)}: "
                f"Holm within at α = {S2.ALPHA} (floor {S2.ALPHA / len(S2.SECONDARY):g}). Every test is an "
                "intersection–union test (decisive p = max of its clause p values). NI clauses: "
                "`metrics.ni_pvalue(...)[\"p_decisive\"]` (A6.21 `bb_ordered`); cost / latency / accuracy clauses: "
                "`metrics.paired_t_test(...)[\"p_greater\"]`. Margins: JevGym "
                f"{S2.M_JEVGYM * 100:g} pt, telecom {S2.M_TAU2 * 100:g} pt, S2-3 accuracy {S2.M_ACC * 100:g} pt. "
                "Reading rules: A7.4 and A7.16 (c).", "",
                "| test | table | arm → reference | run | population | endpoint / clauses | defined by | not tested when |",
                "|---|---|---|---|---|---|---|---|"]
    for t in TESTS:
        L.append(f"| **{t[0]}** | {t[1]} | {t[2]} | `{t[3]}` | {t[4]} | {t[5]} | {t[6]} | {t[7]} |")
    L += ["", "Descriptive (no test): `jevagent_default`, `best_control` (A7.14), the leave-one-group-out rows (A7.11 "
          "(c)), `llm_only_xsmall`, the A7.13 attribution rows, the optional `gpt-6.1-sol` rows (A7.12 (i)), E1 for "
          "JevAgentRL; StartLux-Decision E1, corpus-replay (4B, 27B, size ladder 0.8B / 2B / 9B), live replace and "
          "calibration rows and the static-vs-in-the-loop table (A7.18 (e) 1-4, (h)); the JevAgentRL-v2 rows if "
          "frozen (A7.19 (f)).", "", "## 2. Runs", ""]
    for key, cfg in CONFIGS.items():
        L += run_table(key, cfg)
    L += ["Not JevGym-runner configs: τ² telecom (`scripts/run_stage2_tau2.sh`, one attempt per task, `--task-split-name "
          "full`), E1 (`configs/decision/bfcl_when2call_rl.yaml`, `configs/decision/bfcl_when2call_startlux.yaml`), the "
          "corpus replay (`scripts/stage2_corpus.py`), the StartLux servers (`scripts/aws/serve_startlux.sh`).", "",
          "JevAgentRL-v2 (A7.19 (f); exploratory, descriptive, label \"exploratory (A7.19)\"): `jevgym_shadow_cal_v2`, "
          "`stage2_v2_states` and `stage2_v2` above, the corpus replay `--name v2` and the tariff "
          "(`results/stage2/tariff_v2_plan.json`); served and run by `scripts/aws/stage2_gpu.sh` stages `v2_*` "
          "(`results/stage2/v2_eval_plan.json`).", ""]
    fz = json.loads((ROOT / "configs/eval/jevgym_stage2_tasks.json").read_text())
    tel = ROOT / "configs/tau2/stage2/telecom_inventory.json"
    inv = json.loads(tel.read_text()) if tel.exists() else {}
    L += ["## 3. Populations and seeds", "",
          f"- JevGym Stage-2 list: `configs/eval/jevgym_stage2_tasks.json`, n = {fz['n']}, seeds "
          f"{fz['seed_range'][0]}–{fz['seed_range'][1]}, id-list sha256 `{fz['sha256']}`; S2-3 and the LOGO rows use "
          "its first 200 tasks, the A7.13 live rows its first 100.",
          f"- Evaluation seed blocks refused without `JEV_EVAL_RUNS=1`: {list(R.EVAL_SEED_RANGES)} (Stage 1), "
          f"{list(R.STAGE2_EVAL_RANGES)} (Stage 2); excluded from every training harvest: "
          f"{list(R.training_exclusions())}.",
          f"- Calibration seeds of the X fit: 5150–5299 (`jevgym_shadow_cal_x.yaml`, A7.12 (d)); the optional Sol row's "
          "fit uses the same seeds (`jevgym_shadow_cal_sol.yaml`, A7.12 (i)).",
          (f"- Telecom pool: `configs/tau2/stage2/telecom_inventory.json` (tau2-bench {inv['source']['commit']}), "
           f"{inv['n_pool']} ids, {inv['n_eligible']} eligible (eligible-list sha256 `{inv['eligible_sha256']}`), 32 "
           "validation tasks (seed 731); the sample is frozen in `configs/tau2/stage2/telecom_task_ids.json` after "
           "validation (A7.15).") if inv else "- Telecom pool: inventory not written yet (A7.15 (a)).",
          "- ALFWorld valid_seen: 140 games (A7.2 (c)).", "",
          "## 4. Failure rules", ""] + [f"- {x}" for x in FAILURE_RULES] + ["", "## 5. Resource gates", ""] + \
         [f"- {x}" for x in RESOURCE_GATES] + ["", "## 6. Prerequisites (checked by `--gate <run>`)", ""]
    for run, checks in GATES.items():
        L.append(f"- `{run}`: " + "; ".join(c[0] for c in checks) + ".")
    L += ["", "Every gate also requires this file to be in sync and the Stage-2 fixture tests to pass: "
          + ", ".join(f"`{t}`" for t in FIXTURE_TESTS) + ".", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- the launch gate
def _profile_selected(path: str, s2: str | None = None) -> Callable[[], tuple[bool, str]]:
    def f() -> tuple[bool, str]:
        p = ROOT / path
        if not p.exists():
            return False, f"{path} missing"
        prof = yaml.safe_load(p.read_text()) or {}
        if prof.get("status") != "selected":
            return False, f"{path} status {prof.get('status')!r} (not selected)"
        if s2 and ((prof.get("system2") or {}).get(s2) or {}).get("skip"):
            return True, f"{path} selected; section {s2} skipped (its arms are skipped, the tests enter with p = 1)"
        return True, f"{path} selected"
    return f


def _freeze_record() -> tuple[bool, str]:
    class A:   # the analysis' own gate, so the launch and the analysis cannot disagree
        freeze_record = str(S2.FREEZE_RECORD)
    an = S2.Analysis.__new__(S2.Analysis)
    an.a, an.flags, an.provenance = A, S2.P.Flags(), {}
    why = an.rl_gate()
    return (why is None), (why or f"freeze record {S2.P._rel(S2.FREEZE_RECORD)} before the deadline")


def _exists(path: Path, what: str) -> Callable[[], tuple[bool, str]]:
    return lambda: (path.exists(), f"{what}: {S2.P._rel(path)}" + ("" if path.exists() else " missing"))


def _task_major(cfg: str) -> Callable[[], tuple[bool, str]]:
    def f() -> tuple[bool, str]:
        exp = yaml.safe_load((ROOT / cfg).read_text())
        mode = (exp.get("schedule") or {}).get("mode")
        return mode == "task_major", f"{cfg}: schedule {mode or 'arm_major'}"
    return f


def _x_cost_gate() -> tuple[bool, str]:
    if not X_COST_GATE.exists():
        return False, f"{S2.P._rel(X_COST_GATE)} missing (A7.12 (c): n decided before calibration)"
    g = json.loads(X_COST_GATE.read_text())
    exp = yaml.safe_load((ROOT / CONFIGS["stage2_x"]).read_text())
    ok = g.get("n") in (300, 500) and int(exp.get("n_tasks", 0)) == int(g.get("n", -1))
    return ok, f"X cost gate n = {g.get('n')} (config n_tasks {exp.get('n_tasks')})"


def _x_prices() -> tuple[bool, str]:
    from jevharness.accounting import PriceTable

    pt = PriceTable.load()
    missing = [m for m in X_MODELS if m not in pt.prices]     # exact ids: a suffix match is not a price (A7.12 (h))
    return not missing, "X prices in configs/prices/models.yaml" + (f": missing {missing}" if missing else "")


def _sol_cost_gate() -> tuple[bool, str]:
    """A7.12 (i) with (b), (c): the pair is eligible and n is 500 or 300 and equals the config's n_tasks."""
    if not SOL_COST_GATE.exists():
        return False, f"{S2.P._rel(SOL_COST_GATE)} missing (A7.12 (c), (i): n decided before calibration)"
    g = json.loads(SOL_COST_GATE.read_text())
    exp = yaml.safe_load((ROOT / CONFIGS["stage2_sol"]).read_text())
    ok = bool(g.get("eligible")) and g.get("n") in (300, 500) and int(exp.get("n_tasks", 0)) == int(g.get("n") or -1)
    return ok, f"Sol cost gate: {g.get('decision')} (config n_tasks {exp.get('n_tasks')})"


def _sol_prices() -> tuple[bool, str]:
    from jevharness.accounting import PriceTable

    pt = PriceTable.load()
    missing = [m for m in SOL_MODELS if m not in pt.prices]
    bad = [m for m in SOL_MODELS if m == "gpt-6.1-sol" and m in pt.prices and
           (pt.prices[m].input_per_m, pt.prices[m].output_per_m, pt.prices[m].cached_input_per_m) != (2.0, 10.0, 0.1)]
    return not (missing or bad), "Sol prices in configs/prices/models.yaml" + (
        f": missing {missing}" if missing else "") + (": gpt-6.1-sol differs from the frozen $2 / $10 / $0.10 "
                                                       "(A7.12 (i))" if bad else "")


def _startlux_served(size: str) -> Callable[[], tuple[bool, str]]:
    def f() -> tuple[bool, str]:
        p = ROOT / "results/stage2/startlux_manifest.json"
        if not p.exists():
            return False, "results/stage2/startlux_manifest.json missing (scripts/aws/serve_startlux.sh, A7.18 (d))"
        rec = next((r for r in json.loads(p.read_text()).get("sizes") or [] if r.get("size") == size), None)
        ok = rec is not None and rec.get("status") in ("original", "gguf_q8_0")
        return ok, f"StartLux-Decision-{size}: " + (str(rec.get("status")) + (f" ({rec.get('why')})" if rec.get("why")
                                                                            else "") if rec else "not in the manifest")
    return f


def _v2_freeze() -> tuple[bool, str]:
    """A7.19 (e): the v2 freeze record exists and was written by the deadline (the analysis' own rule)."""
    why, rec = S2.v2_freeze_status(S2.V2_FREEZE_RECORD)
    return (why is None), (why or f"v2 freeze record {S2.P._rel(S2.V2_FREEZE_RECORD)} ({rec.get('frozen_at_utc')}) "
                                  "before the A7.19 (e) deadline")


def _v2_fitted() -> tuple[bool, str]:
    """A7.19 (f) / A7.18 (e) 4: the profile's `backends.jevagentrl_v2` block exists, nothing in it is pending, it was
    filled from v2's own calibration arm and the fits it names are the committed files (sha256). A block without it
    would let the v2 arm run with the section's (Jev's) thresholds."""
    p = ROOT / "configs/harness/jevagent_best.yaml"
    prof = yaml.safe_load(p.read_text()) or {}
    blk = (((prof.get("system2") or {}).get(S2.FLASH) or {}).get("backends") or {}).get(S2.V2_BACKEND)
    if blk is None:
        return False, f"no backends.{S2.V2_BACKEND} block (scripts/fill_backend_tau.py --add-pending, then the fit)"
    pending = [k for k, v in blk.items() if isinstance(v, dict) and v.get("calibration") == "pending"]
    if pending:
        return False, f"backends.{S2.V2_BACKEND}: pending {pending}"
    ff = blk.get("fitted_from") or {}
    if ff.get("fit_arm") != V2_FIT_ARM:
        return False, f"backends.{S2.V2_BACKEND} fitted from {ff.get('fit_arm')!r}, not {V2_FIT_ARM!r}"
    bad = []
    for dp in ("tool_select", "tool_params"):
        rec = ff.get(dp)
        if rec is None:
            continue
        f = ROOT / "results/calibration" / Path(str(rec.get("path"))).name
        if not f.exists() or _sha_file(f) != rec.get("sha256"):
            bad.append(f"{dp} ({S2.P._rel(f)})")
    if bad:
        return False, f"backends.{S2.V2_BACKEND}: fit file missing or changed: {bad}"
    return True, f"backends.{S2.V2_BACKEND} fitted from {V2_FIT_ARM}" + (f"; {blk['skip']}" if blk.get("skip") else "")


def _backend_fitted(backend: str) -> Callable[[], tuple[bool, str]]:
    def f() -> tuple[bool, str]:
        p = ROOT / "configs/harness/jevagent_best.yaml"
        prof = yaml.safe_load(p.read_text()) or {}
        blk = (((prof.get("system2") or {}).get("gemini-3.5-flash") or {}).get("backends") or {}).get(backend)
        if blk is None:
            return True, f"no backends.{backend} block (no cascade to calibrate)"
        pending = [k for k, v in blk.items() if isinstance(v, dict) and v.get("calibration") == "pending"]
        return not pending, f"backends.{backend}: " + (f"pending {pending}" if pending else "fitted")
    return f


GATES: dict[str, list[tuple[str, Callable[[], tuple[bool, str]]]]] = {
    "stage2_jevgym": [("profile `jevagent_best.yaml` selected", _profile_selected("configs/harness/jevagent_best.yaml")),
                      ("task-major schedule", _task_major(CONFIGS["stage2_jevgym"]))],
    "stage2_jevgym --rl": [("JevAgentRL freeze record before the deadline", _freeze_record),
                           ("training manifest", _exists(TRAINING_MANIFEST, "A7.10 (a) training manifest"))],
    "stage2_rl_states": [("JevAgentRL freeze record before the deadline", _freeze_record),
                         ("training manifest", _exists(TRAINING_MANIFEST, "A7.10 (a) training manifest"))],
    "stage2_rl_attribution": [("JevAgentRL freeze record before the deadline", _freeze_record)],
    "stage2_x": [("derived profile `jevagent_best_x.yaml` selected",
                  _profile_selected("configs/harness/jevagent_best_x.yaml", "deepseek/deepseek-v4-pro")),
                 ("X cost-gate record", _x_cost_gate), ("X prices", _x_prices),
                 ("task-major schedule", _task_major(CONFIGS["stage2_x"]))],
    "stage2_tau2": [("profile selected", _profile_selected("configs/harness/jevagent_best.yaml")),
                    ("telecom inventory", _exists(ROOT / "configs/tau2/stage2/telecom_inventory.json", "inventory")),
                    ("frozen telecom sample", _exists(ROOT / "configs/tau2/stage2/telecom_task_ids.json", "sample"))],
    "stage2_alfworld": [("profile selected", _profile_selected("configs/harness/jevagent_best.yaml"))],
    "stage2_jevgym --startlux": [("StartLux serving manifest with the 4B served", _startlux_served("4B")),
                                 ("StartLux-4B thresholds fitted", _backend_fitted("startlux_4b")),
                                 ("S2-4S tariff frozen", _exists(ROOT / "results/stage2/tariff_sl4b.json",
                                                                 "A7.18 (e) 5 tariff"))],
    "stage2_startlux_states": [("StartLux serving manifest with the 4B served", _startlux_served("4B"))],
    "jevgym_shadow_cal_startlux": [("StartLux serving manifest with the 4B served", _startlux_served("4B"))],
    # A7.19 (f): the exploratory JevAgentRL-v2 rows (no test); the order is calibration, then the fits and the tariff
    # committed, then the evaluation rows (scripts/aws/stage2_gpu.sh v2_*)
    "jevgym_shadow_cal_v2": [("JevAgentRL-v2 freeze record before the A7.19 (e) deadline", _v2_freeze),
                             ("v2 training manifest", _exists(V2_TRAINING_MANIFEST, "A7.19 (c) training manifest")),
                             ("v2 serving plan", _exists(V2_PLAN, "A7.19 (f) serving plan"))],
    "stage2_v2_states": [("JevAgentRL-v2 freeze record before the A7.19 (e) deadline", _v2_freeze),
                         ("v2 training manifest", _exists(V2_TRAINING_MANIFEST, "A7.19 (c) training manifest"))],
    "stage2_v2": [("profile `jevagent_best.yaml` selected", _profile_selected("configs/harness/jevagent_best.yaml")),
                  ("task-major schedule", _task_major(CONFIGS["stage2_v2"])),
                  ("JevAgentRL-v2 freeze record before the A7.19 (e) deadline", _v2_freeze),
                  ("JevAgentRL-v2 thresholds fitted from its own calibration", _v2_fitted),
                  ("v2 tariff frozen", _exists(ROOT / "results/stage2/tariff_v2.json", "A7.19 (f) / A7.13 (iii) tariff"))],
    # A7.12 (i): the optional closed-family row (descriptive, no test); calibration only after the cost gate
    "jevgym_shadow_cal_sol": [("Sol capability manifest", _exists(SOL_MANIFEST, "A7.12 (i) / (a) manifest")),
                              ("Sol eligibility and cost-gate record", _sol_cost_gate), ("Sol prices", _sol_prices)],
    "stage2_sol": [("derived profile `jevagent_best_sol.yaml` selected",
                    _profile_selected("configs/harness/jevagent_best_sol.yaml", "gpt-6.1-sol")),
                   ("Sol eligibility and cost-gate record", _sol_cost_gate), ("Sol prices", _sol_prices),
                   ("task-major schedule", _task_major(CONFIGS["stage2_sol"]))],
    "stage2_best_control": [("control selection", _exists(ROOT / "results/stage2/control_selection.json", "A7.14")),
                            ("control config", _exists(ROOT / "configs/experiments/stage2_best_control.yaml", "A7.14"))],
}


def strip_generated(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(GEN_PREFIX))


def gate(run: str, rl: bool, skip_tests: bool, startlux: bool = False) -> int:
    checks: list[tuple[str, bool, str]] = []
    ok_spec = OUT.exists() and strip_generated(OUT.read_text()) == strip_generated(render())
    checks.append(("effective specification in sync", ok_spec, S2.P._rel(OUT)))
    if not skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:warnings", *FIXTURE_TESTS],
                           cwd=ROOT, capture_output=True, text=True)
        checks.append(("Stage-2 fixture tests", r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1]))
    keys = [run] + ([f"{run} --rl"] if rl and f"{run} --rl" in GATES else []) \
        + ([f"{run} --startlux"] if startlux and f"{run} --startlux" in GATES else [])
    if run not in GATES:
        raise SystemExit(f"unknown run {run!r} (one of {sorted(k for k in GATES if ' ' not in k)})")
    for k in keys:
        for name, fn in GATES[k]:
            ok, msg = fn()
            checks.append((name, ok, msg))
    for name, ok, msg in checks:
        print(f"[{'ok' if ok else 'NO'}] {name}: {msg}")
    bad = [c for c in checks if not c[1]]
    print(f"[gate] {run}: {'PASS' if not bad else f'FAIL ({len(bad)} check(s))'}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate", default=None, metavar="RUN")
    ap.add_argument("--rl", action="store_true", help="the gate also covers the run's JevAgentRL arms")
    ap.add_argument("--startlux", action="store_true", help="the gate also covers the run's StartLux arm (S2-4S)")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    if a.gate:
        return gate(a.gate, a.rl, a.skip_tests, a.startlux)
    text = render()
    out = Path(a.out)
    if a.check:
        same = out.exists() and strip_generated(out.read_text()) == strip_generated(text)
        print("in sync" if same else f"{out} has drifted: run python scripts/stage2_spec.py")
        return 0 if same else 1
    out.write_text(text)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
