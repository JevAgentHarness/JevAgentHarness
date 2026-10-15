#!/usr/bin/env python3
"""Render docs/PREREG_EFFECTIVE.md: the EFFECTIVE pre-registration as of now, resolved across docs/PREREGISTRATION.md
sections 1-5 and amendments A1 .. A6.20 (later amendments supersede earlier text where they say so), generated from the
machine-readable sources wherever possible (review round 6, gpt-6-astra E7.6: "a reader should not have to resolve six
superseding amendments to identify the deployed test").

    python scripts/prereg_effective.py                 # rewrite docs/PREREG_EFFECTIVE.md
    python scripts/prereg_effective.py --out PATH      # render somewhere else
    python scripts/prereg_effective.py --check         # exit 1 when docs/PREREG_EFFECTIVE.md has drifted (header aside)

docs/PREREGISTRATION.md remains the authoritative, append-only record; the rendered file is a derived view. Where the two
disagree, PREREGISTRATION.md wins and this script is wrong. Machine-readable sources: the registries of
scripts/prereg_check.py (REGISTERED_SIZES, C1_TAUS, ALWAYS_DEFER, seed constants), the experiment YAMLs read through
jevharness.bench.runner (expand_arms / arm_task_counts / arm_bindings), jevharness.bench.registry, the frozen lists in
configs/eval/, the fitted thresholds in results/calibration/*.json, configs/tau2/airline_task_ids.json,
configs/decision/bfcl_when2call.yaml and scripts/run_tau2.sh. The Deviations section is copied verbatim from
PREREGISTRATION.md; the one-line claims and the "defined by" citations are the hand-written part (keep them in step with
the amendments). tests/test_prereg_effective.py regenerates the manifest and fails when the committed one has drifted."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import prereg_check as PC  # noqa: E402  (the registries: one source for prereg_check and this manifest)
from jevharness.bench import registry as R  # noqa: E402
from jevharness.bench.runner import arm_bindings, arm_task_counts, expand_arms  # noqa: E402
from jevharness.harness.config import DP, Mode  # noqa: E402

OUT_DEFAULT = ROOT / "docs" / "PREREG_EFFECTIVE.md"
PREREG = ROOT / "docs" / "PREREGISTRATION.md"
GEN_PREFIX = "- Generated:"          # the one line that may differ between two renders (commit / date)
CAL_DIR = ROOT / "results" / "calibration"
EXP_DIR = "configs/experiments"
# calibration arm -> (calibration config, C1 arm it sets, evaluation config of that arm)   (A6.5)
CAL_TO_C1 = {"jev_shadow_cal": ("jevgym_shadow_cal.yaml", "jev_ts_cascade_tauhat", "jevgym_tool_select.yaml"),
             "jev_shadow_cal_s2lite": ("jevgym_shadow_cal.yaml", "jev_ts_cascade_tau0.9_s2lite", "jevgym_tool_select.yaml"),
             "intern_shadow_cal": ("jevgym_shadow_cal_intern.yaml", "intern_ts_cascade_tauhat", "jevgym_tool_select_intern.yaml"),
             "intern_shadow_cal_s2lite": ("jevgym_shadow_cal_intern.yaml", "intern_ts_cascade_tau0.9_s2lite",
                                          "jevgym_tool_select_intern.yaml")}
CAL_DPS = ("tool_select", "tool_params")
NI_RULE = ("A6.21: certified Berger–Boos p-value of one ordering of all (wins, losses) outcomes (the monotone hull of "
           "the A6.14 switching rule's nominal p; `metrics.ni_certified_p`, rule `bb_ordered`), rejecting at the Holm "
           "level; Tango's asymptotic p, A6.17's p_U and the task bootstrap are descriptive")
SUP_RULE = ("A6.16 (g): exact conditional sign test on the discordant pairs (`metrics.sign_test_exact`, p_wins) for the "
            "binary endpoints, one-sided paired t-test of the mean paired ΔU (`metrics.paired_t_test`; the sign-flip p is "
            "descriptive; A6.21 (b)) for C5; the task-bootstrap p_sup and CI are descriptive (supersedes A6.1's p_sup)")
CONJ = "conjunction p = max(p_Jev, p_Intern) (A5.1, A6.1)"

# ----------------------------------------------------------------------------------------------- the 11 primaries
# Hand-written: claim, which config / arms, test and the amendments that define the row. Sizes, models, modes and
# thresholds are read from the YAMLs / fits below, never typed here.
PRIMARIES: list[dict] = [
    dict(label="C1F", suite="E2a", cfg="jevgym_tool_select.yaml", arm="jev_ts_cascade_tauhat", ref="llm_only",
         intern=("jevgym_tool_select_intern.yaml", "intern_ts_cascade_tauhat", "llm_only"), test="NI", margin=3,
         claim="tool_select + tool_params CASCADE at the deployed plug-in τ̂\\* (the calibration-arm fit: scalar "
               "projection, A6.5 / A6.11 (c).7 / A6.19) loses ≤ 3 pt success vs LLM-only on Flash; a deferred tool "
               "choice leaves the whole call (tool and arguments) to one native LLM call, after an accepted one each "
               "bounded / copy argument cascades at its own τ and the LLM fills free-text and still-open required "
               "arguments in one forced call (`harness/agent.py::_delegated_call`; argument questions are asked only "
               "inside System-1-chosen calls, A6.11 (c).4); readout: cost savings with CI, cost-of-pass, coverage",
         defined="§2 C1; A1; A2 sign-off; A5.1; A5.3; A6.1; A6.5; A6.11 (c); A6.12; A6.16 (b), (c), (h); A6.17; A6.21 (a)",
         defined_if={"A6.22": "A6.22 (a) (call graph, coverage quantities)"}),
    dict(label="C1L", suite="E2a", cfg="jevgym_tool_select.yaml", arm="jev_ts_cascade_tau0.9_s2lite", ref="llm_only_s2lite",
         intern=("jevgym_tool_select_intern.yaml", "intern_ts_cascade_tau0.9_s2lite", "llm_only_s2lite"), test="NI", margin=3,
         claim="the same plug-in CASCADE policy on Flash-Lite; the Flash-Lite fit accepts no tool_select level, so tool "
               "selection always defers and the LLM writes every call; the fitted tool_params threshold is consulted "
               "only inside System-1-chosen calls and is therefore never reached, so the policy delegates nothing and "
               "an NI pass is *policy non-inferiority with zero coverage* (A6.11 (c).4)",
         defined="§2 C1; A1; A5.3; A6.1; A6.5; A6.11 (c); A6.12; A6.16 (b), (c), (h); A6.17; A6.21 (a)",
         defined_if={"A6.22": "A6.22 (a) (call graph, coverage quantities)"}),
    dict(label="C2F", suite="E2a", cfg="jevgym_tool_select.yaml", arm="jev_ts_assist_k3", ref="llm_only",
         intern=("jevgym_tool_select_intern.yaml", "intern_ts_assist_k3", "llm_only"), test="NI", margin=3,
         claim="tool_select ASSIST k = 3 (System-1 prunes to top-3 + finish, System-2 decides) loses ≤ 3 pt vs LLM-only on "
               "Flash; readout: input-token savings",
         defined="§2 C2; A1; A2 sign-off; A4 (acceptable tool set); A5.1; A5.3; A6.1; A6.12; A6.17; A6.21 (a)"),
    dict(label="C2L", suite="E2a", cfg="jevgym_tool_select.yaml", arm="jev_ts_assist_k3_s2lite", ref="llm_only_s2lite",
         intern=("jevgym_tool_select_intern.yaml", "intern_ts_assist_k3_s2lite", "llm_only_s2lite"), test="NI", margin=3,
         claim="tool_select ASSIST k = 3 on Flash-Lite (the \"where System-1 adds success\" reading)",
         defined="§2 C2; A1; A4; A5.3; A6.1; A6.12; A6.17; A6.21 (a)"),
    dict(label="C3", suite="E2a / E2c", cfg="jevgym_tool_select.yaml", arm="jev_ts_assist_k3", ref="bm25_ts_assist_k3",
         ref_cfg="baselines_jevgym.yaml", intern=None, test="Sup.", margin=0,
         claim="Jev ASSIST k = 3 beats BM25 ASSIST at the same k on Flash (superiority of success); also vs embedding "
               "assist; recall@k against the information-feasible acceptable tool set (A4)",
         defined="§2 C3; A1; A4; A5.1 (Jev-only); A6.1; A6.15 (pair at n = 500 across E2a / E2c); A6.16 (g) (exact sign test)"),
    dict(label="C4", suite="E2b", cfg="jevgym_meta_control.yaml", arm="jev_route_tau0.4", ref="small_only",
         intern=None, test="Sup.", margin=0,
         claim="Jev routing TRIGGER (Flash-Lite by default, Flash when it fires) beats small-only; secondary NI vs "
               "LLM-only and cost savings; tested only with ≥ 30 routing positives in `meta_shadow_s2lite`, else it "
               "stays in the family with p = 1 (*insufficient observed opportunities*)",
         defined="§2 C4; A1; A4 (routing gold); A5.1 (Jev-only); A5.5; A6.1; A6.3; A6.15 (τ = 0.4, n = 500); A6.16 (g) "
                 "(exact sign test), (k) (pilot-selected threshold, stated)"),
    dict(label="C5", suite="E2b / E2c", cfg="baselines_jevgym.yaml", arm="jev_reflect_trigger_s2lite",
         ref="ratem_reflect_trigger_s2lite", intern=None, test="Sup. (ΔU)", margin=0,
         claim="Jev reflection TRIGGER beats the matched-rate random gate on Flash-Lite in paired ΔU at W = $1, λ = 0 "
               "(gross of System-1's price, net alongside); the control's firing rate is fixed before the main runs from "
               "the retired-seed n = 50 pilot's live arm (A6.16 (i)) and inference treats it as fixed; readouts: VoI "
               "with CI (H2), AUROC / J, ē net of placebo, predicted vs observed ΔU; *no headroom on this suite* if the "
               "one-step oracle-ceiling ΔU CI lies within ±3.3 pt; sensitivity row without budget-kind fatals",
         defined="§2 C5; A1 (Flash-Lite); A4; A5.1 (Jev-only); A5.3; A6.1; A6.5 (W = 1); A6.7; A6.9 (C5's W = 1); "
                 "A6.16 (g) (sign-flip test, descriptive since A6.21 (b)), (i) (control rate); A6.21 (b) (paired t-test)"),
    dict(label="C6F", suite="E2b", cfg="jevgym_meta_control.yaml", arm="jev_stop_guard", ref="llm_only",
         intern=("jevgym_tool_select_intern.yaml", "intern_stop_guard", "llm_only"), test="NI", margin=3,
         claim="stop GUARD (System-1 vets the proposed finish) loses ≤ 3 pt vs LLM-only on Flash; readout: FAR on ≥ 50 "
               "wrong finish proposals per System-2 model (natural, forced-premature, transplant; answer-aware gold), "
               "FRR on natural right proposals, rescue from natural wrong proposals",
         defined="§2 C6; A1; A2 sign-off; A4 (C6 readout, probes); A5.1; A5.3; A6.1; A6.12; A6.17; A6.21 (a)"),
    dict(label="C6L", suite="E2b", cfg="jevgym_meta_control.yaml", arm="jev_stop_guard_s2lite", ref="llm_only_s2lite",
         intern=("jevgym_tool_select_intern.yaml", "intern_stop_guard_s2lite", "llm_only_s2lite"), test="NI", margin=3,
         claim="stop GUARD on Flash-Lite; natural FAR with a ±0.10 equivalence check against the probe FAR",
         defined="§2 C6; A1; A4; A5.3; A6.1; A6.12; A6.17; A6.21 (a)"),
    dict(label="C7", suite="E5 τ²", cfg=None, arm="selected on E2", ref="llm_only", intern=None, test="NI", margin=5,
         claim="the most cost-saving E2 (Flash, Jev) System-1 configuration that is NI on E2 at the unadjusted level "
               "(A6.18 (c)) — eligible: tool+args CASCADE at each swept τ and at τ̂\\*, tool ASSIST k = 3, policy GUARD "
               "(E2a's `jev_ts_guard`; on τ² `configs/tau2/jev_guard.yaml`); LLM-only is the reference and never the "
               "selection (A6.20 (d)); ties within 2 pt, the no-NI case and LLM-only being the most cost-saving → the C1 "
               "arm; selected by `scripts/primaries.py --select-c7` into `c7_selection.json` before the first τ² run — "
               "is non-inferior to LLM-only on τ² in **majority pass** (at least 2 of the 3 registered attempts succeed) "
               "with a 5-pt margin (A6.14, A6.16 (e)); the pass^1 mean and its paired task-bootstrap interval are "
               "descriptive endpoints and majority-pass NI does not establish pass^1 NI; τ̂\\* refit per decision point on "
               "the 15 airline shadow tasks (A6.20 (b)); the E2-carried τ̂\\* (= the Flash calibration-arm τ̂\\*) is a "
               "secondary row; after a failed test the wording is \"transfer NI was not established\" (never \"transfer "
               "failed\"), with the fallback rows",
         defined="§1; §2 C7; A5.4; A6.1; A6.5; A6.9 #1, #11; A6.14 (majority pass); A6.15 (System-2 model); A6.16 (e); A6.17; A6.21 (a); "
                 "A6.18 (c); A6.20 (a), (b), (d)",
         defined_if={"A6.22": "A6.22 (b) (refit labels, non-cascade selections), (e) (execution semantics)"}),
    dict(label="C8", suite="E1", cfg=None, arm="jev", ref="gemini_flash_typed", intern=None, test="NI", margin=3,
         claim="Jev answering the typed tool-selection question loses ≤ 3 pt item-weighted accuracy vs the System-2 "
               "model answering the same typed question through structured output, over the BFCL items of the six "
               "categories; argument-slot accuracy and When2Call are secondary rows (Holm within the L1 table); "
               "readout: cost per 1k and p50 latency ratio",
         defined="§1; §2 C8; A5.1 (Jev-only); A6.1; A6.9 (C8's endpoint); A6.13 (C8 population); A6.16 (l) (population frozen "
                 "by source item, `e1_c8_population.json` before the run); A6.17; A6.21 (a)"),
]
HOLM_LABELS = [p["label"] for p in PRIMARIES]

# A6.10 / A6.11 / A5.4 / A5.6 code items ("may land later") -> (symbol, file) presence check in src/
SLIP_CODE = [
    ("`metrics.ni_pvalue` (A6.1)", "def ni_pvalue", "bench/metrics.py"),
    ("`metrics.ni_exact_check` (A6.1)", "def ni_exact_check", "bench/metrics.py"),
    ("`metrics.ni_power` (A6.1)", "def ni_power", "bench/metrics.py"),
    ("`h5_verdict`'s `strictly_better` tier (A6.6)", "strictly_better", "bench/metrics.py"),
    ("`meta_gold.h2_regression_j` (A6.2)", "def h2_regression_j", "bench/meta_gold.py"),
    ("`voi_readout`'s e_raw / h_raw for guards (A6.2)", "e_raw", "bench/meta_gold.py"),
    ("`meta_gold.implication_audit` (A6.7's `audit_implications`, renamed per A6.15 / A6.18 (h))", "def implication_audit",
     "bench/meta_gold.py"),
    ("`oracle_verdict`'s `fatal_kind` (A6.7)", "fatal_kind", "harness/agent.py"),
    ("`threshold_theory.fit_calibration_arm(v_fallback)` (A6.11)", "v_fallback", "bench/threshold_theory.py"),
    ("`threshold_theory.scalar_threshold_projection` (A6.11 (c).7)", "def scalar_threshold_projection",
     "bench/threshold_theory.py"),
    ("`cost_terms(couple_by_step=True)` (A6.11 (c).5)", "couple_by_step", "bench/threshold_theory.py"),
    ("cascade `always_defer` sentinel (A6.11 (c).4)", "always_defer", "harness/agent.py"),
    ("harvest guard excludes the reserved ranges (A6.4 / A6.13)", "reserved_ranges", "training/harvest.py"),
    ("`grpo_online` default eval suite in the 5000 range (A6.4)", '"seed0": 5000', "training/grpo_online.py"),
    ("τ² shadow mode and the airline split (A5.4)", "def airline_split", "bench/suites/tau2.py"),
    ("`exclude_ids_file` for the C8 population (A6.13)", "exclude_ids_file", "bench/decision_eval.py"),
    ("H5 readout `paired_success_lb` / `sign_test_exact` / `wilcoxon_one_sided` / `h5_verdict` (A5.2)",
     "def h5_verdict", "bench/metrics.py"),
    ("`metrics.sign_flip_test` for C5 (A6.16 (g))", "def sign_flip_test", "bench/metrics.py"),
    ("`metrics.ni_unconditional_p` — Berger–Boos exact unconditional p of the score branch (A6.17; comparison rule)",
     "def ni_unconditional_p", "bench/metrics.py"),
    ("`metrics.ni_certified_p` — certified Berger–Boos p over one ordering (A6.21 (a))", "def ni_certified_p",
     "bench/metrics.py"),
    ("`metrics.paired_t_test` — C5's decisive test (A6.21 (b))", "def paired_t_test", "bench/metrics.py"),
    ("runner refuses `calibration: pending` arms (A6.16 (b))", "JEV_ALLOW_PENDING_TAU", "bench/runner.py"),
    ("tie-preserving `equal_mass_bins` (A6.16 (c))", "def equal_mass_bins", "bench/threshold_theory.py"),
    ("H5 `time_sign_p` / `success_preserving` screen label (A6.16 (j))", "time_sign_p", "bench/metrics.py"),
    ("`meta_gold.h2_theory_check` — predicted-vs-observed ΔU with the joint task bootstrap and the ≥ 4-cell / rank "
     "requirements (A6.16 (i))", "def h2_theory_check", "bench/meta_gold.py"),
]
# A6.22 (g): listed once the amendment is appended; a path under adapters/ or scripts/ is read from the repository root
SLIP_CODE_A622 = [
    ("score-card coverage block `decisions.coverage` and `vs_baseline.llm_calls_avoided` (A6.22 (a))",
     "def call_coverage", "bench/scorecard.py"),
    ("share of cache-served System-1 calls `s1_cached_frac` (A6.22 (e) 8)", "def s1_cached_frac", "bench/scorecard.py"),
    ("H5 displayed verdict \"passes the exploratory success screen; faster among jointly solved tasks\" in "
     "`metrics.h5_verdict` (A6.22 (d) 1)", "faster among jointly solved tasks", "bench/metrics.py"),
    ("the same label in `adapters/pi/summarize_harbor.py` (A6.22 (d) 1)", "faster among jointly solved tasks",
     "adapters/pi/summarize_harbor.py"),
]


# ----------------------------------------------------------------------------------------------- helpers
def short_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001  (no git: the header says so)
        return "unknown"


def fmt_tau(x: float | None) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "never (no accepting level)"
    return f"{x:.2f}" if abs(x - round(x, 2)) < 1e-12 else f"{x:.6g}"


def code(s: str) -> str:
    return f"`{s}`"


def cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


class Sources:
    """Everything the manifest reads, loaded once."""

    def __init__(self) -> None:
        self.names = list(dict.fromkeys(PC.EVAL_CONFIGS + PC.CAL_CONFIGS))
        self.exps = {n: PC.load(n) for n in self.names}
        self.counts = {n: arm_task_counts(e) for n, e in self.exps.items()}
        self.cfgs = {n: {c.name: c for c in expand_arms(e)} for n, e in self.exps.items()}
        self.binds = {n: arm_bindings(e) for n, e in self.exps.items()}
        self.prereg = PREREG.read_text()
        self.frozen = json.loads((ROOT / R.EVAL_TASKS_FILE).read_text())
        self.e1_excl = json.loads((ROOT / "configs" / "eval" / "e1_dev_prefix_ids.json").read_text())
        self.e1_cfg = yaml.safe_load((ROOT / "configs" / "decision" / "bfcl_when2call.yaml").read_text())
        self.airline = json.loads((ROOT / "configs" / "tau2" / "airline_task_ids.json").read_text())
        self.tau2_sh = (ROOT / "scripts" / "run_tau2.sh").read_text()
        self.fits = {}
        for p in sorted(CAL_DIR.glob("tauhat_*.json")):
            d = json.loads(p.read_text())
            self.fits[(d["meta"]["arms"][0], d["meta"]["dp"])] = (p, d)
        self.flags: list[str] = []

    # -- models and backends
    def s2_model(self, cfg: str, arm: str) -> str:
        e = self.exps[cfg]
        b = self.binds[cfg].get(arm, {})
        if "llm" in b:
            return e["llms"][b["llm"]]["model"]
        return e["llm"]["model"]

    def s1_spec(self, cfg: str, arm: str) -> dict:
        e = self.exps[cfg]
        b = self.binds[cfg].get(arm, {})
        if "s1" in b:
            return e["s1_backends"][b["s1"]]
        return e.get("s1") or {}

    @staticmethod
    def s1_desc(spec: dict) -> str:
        t = spec.get("type")
        if t == "jev":
            return "Jev (`type: jev`, OpenRouter; pinned `typesafe/jev-1.13`, §1)"
        if t == "intern_decision":
            return f"Intern-Decision (`{spec.get('model')}`, in-process)"
        if t == "rate":
            ph = "; **placeholder** until set from the Jev arm's realised rate" if spec.get("placeholder") else ""
            return f"matched-rate random (`type: rate`, rate {spec.get('rate')}{ph})"
        if t == "bm25":
            return "BM25 (`type: bm25`, non-learned)"
        if t == "llm_generative":
            return f"generative stand-in (`{spec['llm']['model']}`; reference row only, §1)"
        if t == "s2_self":
            return f"System-2 deciding for itself (`{spec['llm']['model']}`)"
        return code(str(t))

    def n(self, cfg: str, arm: str) -> int | None:
        return self.counts[cfg].get(arm)

    def size(self, cfg: str, arm: str) -> str:
        """n from the YAML, cross-checked with prereg_check.REGISTERED_SIZES."""
        n = self.n(cfg, arm)
        if n is None:
            self.flags.append(f"`{arm}` is no arm of `{cfg}`")
            return "missing"
        reg = PC.REGISTERED_SIZES.get(cfg, {}).get(arm)
        if reg is None:
            self.flags.append(f"`{arm}` (`{cfg}`) runs the YAML size n = {n} but is not in `prereg_check.REGISTERED_SIZES`"
                              " (its size is not locked by the check)")
            return f"{n}†"
        if reg != n:
            self.flags.append(f"`{arm}` (`{cfg}`): YAML n = {n} vs REGISTERED_SIZES {reg}")
            return f"{n} (registered {reg})"
        return str(n)

    def modes(self, cfg: str, arm: str, taus_from_fit: bool = False) -> str:
        c = self.cfgs[cfg][arm]
        parts = []
        for dp in DP:
            d = c.dp(dp)
            if d.mode == Mode.OFF:
                continue
            s = f"{dp.value} {d.mode.value}"
            if d.mode in (Mode.CASCADE, Mode.GUARD, Mode.TRIGGER):
                if d.extra.get("always_defer"):
                    s += f" **always-defer** (τ {fmt_tau(d.tau)} in the YAML, unused)"
                else:
                    s += f" τ {fmt_tau(d.tau)}"
            elif d.mode == Mode.ASSIST:
                s += f" k = {d.k}"
            parts.append(s)
        ts, tp = c.dp(DP.TOOL_SELECT), c.dp(DP.TOOL_PARAMS)
        if ts.mode == Mode.CASCADE and ts.extra.get("always_defer") and tp.mode in (Mode.CASCADE, Mode.REPLACE):
            # A6.11 (c).4: the harness asks argument questions only inside System-1-chosen calls
            # (harness/agent.py::_delegated_call), so after an always-deferred tool choice the LLM writes every call
            parts.append("**the tool_params threshold is unreachable**: tool selection always defers, so the LLM "
                         "writes every call (tool and arguments) and no argument question is asked "
                         "(`harness/agent.py::_delegated_call`): the policy delegates nothing (A6.11 (c).4)")
        return "; ".join(parts) if parts else "LLM-only"

    # -- the registration text
    def amendments(self) -> list[str]:
        """The A6.x amendment numbers with a heading in PREREGISTRATION.md, in numeric order."""
        nums = {int(m.group(1)) for m in re.finditer(r"^### A6\.(\d+)\b", self.prereg, re.M)}
        return [f"A6.{k}" for k in sorted(nums)]

    def has_amendment(self, name: str) -> bool:
        """True once ``### <name>`` is a heading of PREREGISTRATION.md: text that depends on an amendment is rendered
        only after the amendment has been appended (appended 2026-10-02)."""
        return re.search(rf"^### {re.escape(name)}\b", self.prereg, re.M) is not None

    def fit_check(self, cfg: str, arm: str) -> str:
        """For a C1 arm: the fitted τ̂* per decision point against the deployed YAML value."""
        cal = next((k for k, v in CAL_TO_C1.items() if v[1] == arm and v[2] == cfg), None)
        if cal is None:
            return ""
        c = self.cfgs[cfg][arm]
        out = []
        for dp in CAL_DPS:
            hit = self.fits.get((cal, dp))
            d = c.dp(DP(dp))
            if hit is None:
                out.append(f"{dp}: fit pending (`results/calibration/tauhat_{cal}_{dp}.json` absent; YAML τ "
                           f"{fmt_tau(d.tau)} is a placeholder)")
                continue
            tau = hit[1]["tau_hat"]
            if isinstance(tau, float) and math.isinf(tau):
                ok = bool(d.extra.get("always_defer"))
                out.append(f"{dp}: fit never → always defer {'✓' if ok else '**≠ YAML**'}")
                if not ok:
                    self.flags.append(f"`{arm}`.{dp}: the fit says never (always defer) but the YAML has no always_defer")
            else:
                ok = abs(float(d.tau) - float(tau)) < 1e-9 and not d.extra.get("always_defer")
                out.append(f"{dp}: fit {fmt_tau(tau)} {'✓' if ok else '**≠ YAML**'}")
                if not ok:
                    self.flags.append(f"`{arm}`.{dp}: fitted τ̂* {fmt_tau(tau)} vs YAML {fmt_tau(d.tau)}")
        return "; ".join(out)


# ----------------------------------------------------------------------------------------------- sections
def header(S: Sources) -> list[str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    last = (S.amendments() or ["A6.20"])[-1]
    a622 = ("; A6.22 settles the C1 call graph and the coverage quantities, the τ² refit labels, the H1 / H3 / H4 verdict "
            "rules, H5's displayed verdict and Terminal-Bench-Science subset and the execution semantics, and withdraws "
            "A6.6's order-statistic sentence" if S.has_amendment("A6.22") else "")
    return [
        "# Effective pre-registration (derived view)",
        "",
        f"{GEN_PREFIX} {now} at commit `{short_commit()}` by `scripts/prereg_effective.py` "
        "(re-run `python scripts/prereg_effective.py` after any amendment, size, config or fit change).",
        f"- Resolves `docs/PREREGISTRATION.md` §1–5 and amendments A1–{last} into the tests as they stand: later "
        "amendments supersede earlier text where they say so (A5 over §2's family size, A2's budget line and C7's "
        "airline size; A6 over §2's NI sentence, §3 H2's test, A5.5's family clause, A6.4's seed range, A5.2's verdict "
        "names and the source of τ̂\\*; A6.11 (c) over A6.11 (a)/(b); A6.13 over A6.4's evaluation seeds; A6.14 over "
        "A6.1's NI decision; A6.16 (g) over A6.1's superiority p_sup, (i) over A4's live-arm control rate, (j) over A5.2's "
        "Wilcoxon; A6.17 over A6.14's asymptotic score p and A6.16 (f)'s contingency; A6.18 over the stale sentences it "
        "names; A6.21 over A6.14's and A6.17's decisive p, A6.16 (g)'s C5 clause and the A6.1 / A6.14 power readings; A6.19 over A6.11 (c).7's non-tail clause and A6.5's threshold-form sentence; A6.20 settles C7's eligible set, "
        f"τ² always-defer / per-decision-point τ / System-1 endpoint and negligible coverage{a622}).",
        "- **`docs/PREREGISTRATION.md` remains the authoritative, append-only record; this file is a derived view. "
        "Where this file and PREREGISTRATION.md disagree, PREREGISTRATION.md wins and the script is wrong** (fix "
        "`scripts/prereg_effective.py`; never edit the registration to match this file).",
        "- Generated from: `scripts/prereg_check.py` (registries), the experiment YAMLs through "
        "`jevharness.bench.runner.expand_arms` / `arm_task_counts` / `arm_bindings`, `jevharness.bench.registry`, "
        f"`{R.EVAL_TASKS_FILE}`, `configs/eval/e1_dev_prefix_ids.json`, `configs/decision/bfcl_when2call.yaml`, "
        "`configs/tau2/airline_task_ids.json`, `scripts/run_tau2.sh`, `results/calibration/*.json` and "
        "PREREGISTRATION.md itself (Deviations verbatim). The one-line claims and the \"defined by\" citations are "
        "hand-written in the script. `tests/test_prereg_effective.py` fails when this file drifts from those sources.",
        "",
    ]


def primaries_table(S: Sources) -> list[str]:
    L = ["## 1. The Holm family: 11 one-sided tests, family-wise α = 0.05 (A5.3; §2's \"across the 8\" is superseded)",
         "",
         "| Test | Claim | Suite · config | Arm → reference | System-2 | System-1 | n (arm / ref) | Test | Decision rule "
         "| Deployed modes / thresholds | Defined by |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in PRIMARIES:
        lab = p["label"]
        if lab == "C4":
            row = row_c4(S, p)
        elif lab == "C7":
            row = row_c7(S, p)
        elif lab == "C8":
            row = row_c8(S, p)
        else:
            row = row_jevgym(S, p)
        row[-1] = defined_by(S, p)
        L.append("| " + " | ".join(cell(x) for x in row) + " |")
    L += ["",
          "† size read from the YAML default; the arm is not locked by `prereg_check.REGISTERED_SIZES` (see the "
          "consistency notes below).",
          "",
          "> **NI decision rule (A6.21 (a)).** Every NI row (H0: P(loss) − P(win) ≥ m on n paired binary outcomes) is "
          "decided by one ordering of all (wins W, losses L) outcomes: s̃, the monotone hull of the log nominal p of the "
          "A6.14 switching rule (Tango's one-sided score p when W ≥ 1 and W + L ≥ 5, else P[Bin(n, m) ≤ L]); the "
          "decisive p = min{1, γ + U}, with U a certified upper bound (branch and bound with a Hölder bound, tolerance "
          "max(1e-7, 1e-3 · sup)) on sup over the boundary nuisance w in the (1 − γ) Clopper–Pearson set from the "
          "discordant count of P_w[s̃ ≤ s̃_obs], γ = 1e-4 (`metrics.ni_certified_p`, `rule = bb_ordered`); it rejects "
          "iff p ≤ its Holm level. Monotone tails and a coupling make the boundary point with the same discordance least "
          "favourable, so P(p ≤ α) ≤ α at every n, level and null point. Tango's asymptotic p, A6.17's p_U, the exact "
          "loss-count p, the loss-rate bound 1 − α^{1/n} at zero discordance and the paired task bootstrap are "
          "descriptive. Evidence (a check, not the argument): `results/reviews/2026-10-02/ni_exact_size_r8.md` (exact "
          "summation: ≤ 0.997 α at every Holm level for n ∈ {149, 300, 500, 1000, 1114}, ≤ 0.66 α inside the null) and "
          "`ni_power_exact_r8.md` (exact power with feasibility labels and two-family conjunctions, A6.21 (c)). C7 "
          "decides on majority pass over the 3 trials (A6.14, A6.16 (e)).",
          "",
          "Rules shared by every row (effective text, with the amendment that sets it):",
          "",
          "- **Family and level.** The 11 tests form one Holm family at FWER α = 0.05; NI nulls (H0: Δ ≤ −m) and "
          "superiority nulls (H0: Δ ≤ 0) share it (A5.3, A6.1). C4 stays in the family with p = 1 when "
          "`meta_shadow_s2lite` yields < 30 routing positives (A6.3 replaces A5.5's removal clause).",
          "- **Statistic.** Δ\\* = arm − reference over the paired task bootstrap (the task is the resampling unit; "
          "τ² trials of one task stay together; `metrics.paired_bootstrap_diff`); margins m = 3 pt on JevGym, ALFWorld "
          "and L1, 5 pt on τ² majority pass (§2, A6.1, A6.16 (e)). §2's one-sided 95 % lower bound and Newcombe / Tango "
          "score intervals are reported as descriptive, never as the decision (A6.1). Per-decision cost allocations "
          "(ΔC) are approximations to the joint tool-and-argument policy, never summed as independently avoidable "
          "calls; deployment savings come from the complete configuration's ledger (A6.16 (d)).",
          "- **Reading rule.** A failed NI test whose two-sided 95 % CI covers 0 is *inconclusive at this n*, never "
          "inferiority and never \"harm\"; a CI entirely below 0 is a loss of the stated size (A6.1). MDE at the "
          "achieved discordance and the primary's Holm level is printed next to every result (`metrics.ni_power`).",
          "- **Two System-1 families (C1, C2, C6).** Jev and Intern-Decision-4B run on identical tasks and seeds; the "
          "primary is confirmed only if it holds for both: p = max(p_Jev, p_Intern) enters Holm (A5.1, A6.1). "
          "Labels: *cross-backend replication* (both pass), *Jev-specific confirmation* (the competence gate demoted "
          "Intern-Decision and Jev passes), *split* (both co-primary, one fails) (A6.9 #6); *cross-backend replication "
          "of policy NI* when both pass and either constituent's coverage is negligible, below 5 % of a decision "
          "point's opportunities (A6.16 (m), A6.20 (e))" + (
              ", read per decision point on tool-decision and argument coverage, with native LLM calls avoided per "
              "episode beside them (A6.22 (a))" if S.has_amendment("A6.22") else "") + ". The gate (A5.1: ≥ 0.8 × "
          "Jev's accuracy on the 210 static dev decisions; verdict committed in `results/gpu_round2/gate_verdict.json` (A6.16 (b), A6.18 (e)) before any "
          "C1 / C2 / C6 arm runs) establishes eligibility only; once promoted the family stays co-primary whatever "
          "later pilots show, and a gate-passing-but-weaker family is reported as a static-to-sequential competence "
          "gap with same-state shadow accuracy, live accuracy, coverage and net savings (A6.12). C3, C4, C5, C7, C8 "
          "are Jev-only primaries (A5.1).",
          "- **Operating point.** W = $1 per task success, λ = 0 for C1, C5 and H2 alike; W ∈ {0.1, 10}, λ ∈ "
          "{0.5, 5} $/h are sensitivity rows (A6.5, A6.9); W = $1 is a declared preference, not an economic truth "
          "(A6.11 (c).6).",
          "- **Drop rule.** Paired complete-case: a task still errored in either arm after two extra attempts leaves "
          "that comparison; per-arm dropped counts and n′ ≤ n are printed, and a full-set sensitivity row counts an "
          "errored task as a failure in the arm where it errored (§1, A6.9). Agent-caused failures are failures, "
          "never exclusions (§1)." + (
              " The error taxonomy (an infrastructure error is an episode record with `error` set: an exception escaping "
              "the episode, including a live System-1 failure, which is not fail-open in the reference loop; τ² "
              "`agent_error` / `user_error` / `infrastructure_error`; a Harbor exception other than the agent timeout), "
              "retry semantics (at most three attempts, each a fresh execution; the latest record decides), the dropped "
              "tasks listed with their characteristics next to the full-set row, guard exhaustion, fail-open events "
              "counted as deployed-policy behaviour, cached System-1 versus fresh System-2 responses and the hard-task "
              "caps are A6.22 (e)." if S.has_amendment("A6.22") else ""),
          "- **Sizes.** n = 500 JevGym tasks for every NI pair and 149 tested τ² tasks; margins are not widened to fit "
          "the sample. Power at the registered sizes is the exact table of A6.21 (c) "
          "(`results/reviews/2026-10-02/ni_power_exact_r8.md`; \"adequate\" and \"well powered\" are withdrawn): at the "
          "Holm floor three feasible scenarios reach 80 % power for one family (Flash at 2 % discordance and Δ = 0; C8 at "
          "6 % and Δ = 0, and at 10 % and Δ = +3 pt) and one two-family conjunction does (Flash at 2 %, Δ = 0); a "
          "primary that fails elsewhere is expected to read *inconclusive at this n* (A6.21 (c)). No stopping rule, no interim looks (§1). Prices, model ids, prompts "
          "and harness defaults are frozen before the main runs (§1).",
          "- **Iso-quality wording.** Paper tables qualify arms with `noninferior` = the registered decision statistic at "
          "the unadjusted level and margin (the A6.21 rule's p ≤ 0.05; A6.18 (c)); `not_worse` is not used for a paper number "
          "(A6.9 #10). C7's selection rule (A6.9 #11) uses the same criterion.",
          ""]
    if S.flags:
        L += ["Generated consistency notes (YAMLs / fits / registries against the registration text; the registration "
              "is not edited to match):", ""]
        L += [f"- {f}" for f in dict.fromkeys(S.flags)]
        L.append("")
    return L


def s1_cell(S: Sources, cfg: str, arm: str, ref: str, intern: tuple | None, rcfg: str | None = None) -> str:
    rcfg = rcfg or cfg
    a, r = S.s1_spec(cfg, arm), S.s1_spec(rcfg, ref)
    s = S.s1_desc(a)
    ref_cfg = S.cfgs[rcfg][ref]
    if any(ref_cfg.dp(dp).mode != Mode.OFF for dp in DP):
        s += f"; reference: {S.s1_desc(r)}"
    if intern:
        icfg, iarm, _ = intern
        s += f"; co-primary {S.s1_desc(S.s1_spec(icfg, iarm))} — {CONJ}"
    else:
        s += "; Jev-only primary (A5.1)"
    return s


def row_jevgym(S: Sources, p: dict) -> list[str]:
    cfg, arm, ref, intern = p["cfg"], p["arm"], p["ref"], p["intern"]
    rcfg = p.get("ref_cfg", cfg)   # C3 (A6.15): the reference lives in another experiment file, same task list
    arms = f"`{arm}` → `{ref}`"
    n = f"{S.size(cfg, arm)} / {S.size(rcfg, ref)}"
    modes = S.modes(cfg, arm)
    fit = S.fit_check(cfg, arm)
    if fit:
        modes += f" — {fit}"
    s2 = f"`{S.s2_model(cfg, arm)}`"
    if S.s2_model(cfg, arm) != S.s2_model(rcfg, ref):
        S.flags.append(f"`{arm}` and `{ref}` ({cfg}) run different System-2 models")
    if intern:
        icfg, iarm, iref = intern
        arms += f"; Intern: `{iarm}` → `{iref}` (`{icfg}`)"
        n += f"; Intern {S.size(icfg, iarm)} / {S.size(icfg, iref)}"
        imodes = S.modes(icfg, iarm)
        ifit = S.fit_check(icfg, iarm)
        modes += f"; Intern: {imodes}" + (f" — {ifit}" if ifit else "")
        if S.s2_model(icfg, iarm) != S.s2_model(cfg, arm):
            S.flags.append(f"Intern arm `{iarm}` runs `{S.s2_model(icfg, iarm)}`, the Jev arm `{S.s2_model(cfg, arm)}`")
    if p["label"] == "C3":
        bl = p["ref_cfg"]
        n_ref = S.n(bl, ref)
        arms = f"`{arm}` (E2a, `{EXP_DIR}/{cfg}`) → `{ref}` (E2c, `{EXP_DIR}/{bl}`): the same frozen task list (A6.15)"
        n = f"{S.n(cfg, arm)} / {n_ref}"
        if n_ref != 500 or S.n(cfg, arm) != 500:
            S.flags.append(f"C3: A6.15 registers both arms at n = 500; found {S.n(cfg, arm)} / {n_ref}")
    if p["label"] == "C5":
        rate = S.s1_spec(rcfg, ref)
        if rate.get("placeholder"):
            arms += "; the runner skips the placeholder control until its rate is set from this run's Jev arm"
    test = f"NI, m = {p['margin']} pt" if p["test"] == "NI" else p["test"]
    rule = NI_RULE if p["test"] == "NI" else SUP_RULE
    if intern:
        rule += f"; {CONJ}"
    return [f"**{p['label']}**", p["claim"], f"{p['suite']} · `{EXP_DIR}/{cfg}`", arms, s2,
            s1_cell(S, cfg, arm, ref, intern, rcfg), n, test, rule, modes, p["defined"]]


def row_c4(S: Sources, p: dict) -> list[str]:
    cfg, arm, ref = p["cfg"], p["arm"], p["ref"]
    sweep = sorted(a for a in S.counts[cfg] if a.startswith("jev_route_tau") and a != arm)
    bl = "baselines_jevgym.yaml"
    bl_route = sorted(a for a in S.counts[bl] if a.startswith("jev_route_tau"))
    arms = (f"`{arm}` → `{ref}` (Flash-Lite only; A6.15: τ fixed at 0.4 before the main runs); secondary: the sweep "
            f"{', '.join(code(a) for a in sweep)} at n = {sorted({S.n(cfg, a) for a in sweep})} and E2c's "
            f"{', '.join(code(a) for a in bl_route)} paired with `ratem_route` (H2)")
    if S.n(cfg, arm) != 500 or S.n(cfg, ref) != 500:
        S.flags.append(f"C4: A6.15 registers `{arm}` and `{ref}` at n = 500; found {S.n(cfg, arm)} / {S.n(cfg, ref)}")
    n = f"{S.n(cfg, arm)} / {S.n(cfg, ref)}"
    small = S.exps[cfg]["small_llm"]["model"]
    s2 = f"routing between `{S.s2_model(cfg, 'llm_only')}` (fires) and `{small}` (default); reference `{ref}` = `{S.s2_model(cfg, ref)}`"
    modes = f"`{arm}`: {S.modes(cfg, arm)}"
    meas = S.cfgs["jevgym_meta_control.yaml"]["meta_shadow_s2lite"]
    rule = (SUP_RULE + f"; opportunity rule: ≥ 30 routing positives in `meta_shadow_s2lite` (n = "
            f"{S.n('jevgym_meta_control.yaml', 'meta_shadow_s2lite')}, routing {meas.routing.mode.value}) read before any "
            "C4 live arm is scored, else p = 1 (A5.5, A6.3)")
    return [f"**{p['label']}**", p["claim"], f"{p['suite']} · `{EXP_DIR}/{cfg}` (+ `{EXP_DIR}/{bl}`)", arms, s2,
            s1_cell(S, cfg, arm, ref, None), n, "Sup.", rule, modes, p["defined"]]


def row_c7(S: Sources, p: dict) -> list[str]:
    sp = S.airline["split"]
    n_sh, n_te, n_all = len(sp["shadow"]), len(sp["test"]), len(S.airline["task_ids"])
    m = re.search(r'AGENT_LLM:=([^}"]+)', S.tau2_sh)
    agent = m.group(1) if m else "?"
    m2 = re.search(r'USER_LLM:=([^}"]+)', S.tau2_sh)
    m3 = re.search(r'NUM_TRIALS:=([^}"]+)', S.tau2_sh)
    trials = m3.group(1) if m3 else "3"
    cfgs = sorted(q.stem for q in (ROOT / "configs" / "tau2").glob("*.yaml"))
    reg_agent, reg_user = "gemini/gemini-3.5-flash", "gemini/gemini-3.5-flash-lite"   # A6.15
    if agent != reg_agent or (m2 and m2.group(1) != reg_user):
        S.flags.append(f"C7: A6.15 registers the τ² System-2 as `{reg_agent}` (user simulator `{reg_user}`); "
                       f"`scripts/run_tau2.sh` defaults to `{agent}` / `{m2.group(1) if m2 else '?'}`")
    arms = (f"`jev_ts_cascade_tau<τ̂\\* refit>` (or the selected configuration) → `llm_only`; arm configs "
            f"`configs/tau2/{{{', '.join(cfgs)}}}.yaml` via `scripts/run_tau2.sh`; `jev_shadow` on the {n_sh} airline "
            f"shadow tasks × {trials} refits τ̂\\* (A5.4 (c)); secondary rows `jev_ts_cascade_tau<τ̂\\*_E2>` and, on "
            "failure, the C1 arm and `jev_ts_assist_k3` on airline only (A5.4 (d))")
    n = (f"run: {n_all} airline + 114 retail tasks × {trials} trials; tested: {n_te} airline test + 114 retail = "
         f"{n_te + 114} tasks (A6.9 #1); {n_sh} airline shadow tasks descriptive (split seed {sp['seed']}, "
         f"`configs/tau2/airline_task_ids.json`)")
    s2 = f"`{agent}` (registered by A6.15; `scripts/run_tau2.sh` default, drift flagged above)"
    s1 = "Jev (`type: jev`); Jev-only primary (A5.1)"
    modes = ("when the selection is a cascade: tool_select + tool_params CASCADE at the τ̂\\* refit per decision point "
             "from the airline shadow log with the deployed C1 estimator (`threshold_theory --calibration-arm`: "
             "`scalar_threshold_projection`, A6.11 (c).7 / A6.19; not H1's isotonic plug-in), the V = W rule of A6.11 "
             "(c).1, always-defer where the refit accepts no level (A6.20 (a), (b)); an ASSIST or GUARD selection runs "
             "its E2 settings (`jev_ts_assist_k3.yaml`, `jev_guard.yaml`); every τ² arm runs all 50 airline tasks so arms "
             "share one list")
    if S.has_amendment("A6.22"):
        modes += ("; the refit's labels are reference-agreement labels (single reference next step, reference argument "
                  "values, the LLM's executed action as the paired label, one episode per simulation for V, the "
                  "sample-size rule), and a non-cascade selection refits nothing (A6.22 (b))")
    return [f"**{p['label']}**", p["claim"], f"{p['suite']} · `scripts/run_tau2.sh`, `configs/tau2/`", arms, s2, s1, n,
            f"NI, m = {p['margin']} pt (majority pass over 3 attempts, A6.16 (e))", NI_RULE, modes, p["defined"]]


def defined_by(S: Sources, p: dict) -> str:
    """The row's "defined by" citations; ``defined_if`` entries are added once their amendment has been appended."""
    extra = [txt for name, txt in (p.get("defined_if") or {}).items() if S.has_amendment(name)]
    return "; ".join([p["defined"], *extra])


def row_c8(S: Sources, p: dict) -> list[str]:
    e = S.e1_cfg
    bfcl = next(s for s in e["suites"] if s["name"] == "bfcl")
    w2c = next(s for s in e["suites"] if s["name"] == "when2call")
    cats = bfcl["categories"]
    excl = S.e1_excl["ids"]
    n_w2c = sum(1 for i in excl if i.startswith("w2c"))
    per_cat = {c: 0 for c in cats}
    for i in excl:
        for c in sorted(cats, key=len, reverse=True):
            if i.startswith(c + "_"):
                per_cat[c] += 1
                break
    n_bfcl = sum(per_cat.values())
    if n_bfcl + n_w2c != len(excl):
        S.flags.append(f"C8: {len(excl) - n_bfcl - n_w2c} excluded E1 ids match neither a BFCL category nor When2Call")
    ref = e["backends"][p["ref"]]
    arms = (f"backend `{p['arm']}` → `{p['ref']}` (`type: {ref['type']}`, `{ref['llm']['model']}` answering the same "
            f"typed question); `configs/decision/bfcl_when2call.yaml`; `exclude_ids_file: {e.get('exclude_ids_file')}`")
    pop_path = ROOT / "configs" / "eval" / "e1_c8_population.json"
    if pop_path.exists():   # A6.16 (l) / A6.18 (d): the frozen population counts (exclusion by parent item)
        pop = json.loads(pop_path.read_text())
        ts = {k.split("|")[0].split(":", 1)[1]: v for k, v in pop["by_source_dp"].items()
              if k.startswith("bfcl:") and k.endswith("|tool_select")}
        n_ts = sum(ts.values())
        n_tp = sum(v for k, v in pop["by_source_dp"].items() if k.startswith("bfcl:") and k.endswith("|tool_params"))
        n_w2 = sum(v for k, v in pop["by_source"].items() if k.startswith("when2call:"))
        n = (f"**C8 primary n = {n_ts:,} BFCL tool_select items** ({', '.join(f'{c} {k}' for c, k in sorted(ts.items()))}) "
             f"after excluding the {len(excl)} dev-prefix records by parent item (`{pop_path.relative_to(ROOT)}`, sha256 "
             f"`{pop['sha256'][:12]}`, {pop['n_items']:,} records / {pop['n_source_items']:,} source items); secondary rows: "
             f"{n_tp:,} BFCL argument slots, {n_w2:,} When2Call items")
    else:
        n = (f"BFCL: {len(cats)} categories × ≤ {bfcl['max_per_cat']} items = ≤ {len(cats) * bfcl['max_per_cat']}, minus "
             f"{n_bfcl} dev-prefix records ({', '.join(f'{c} {k}' for c, k in per_cat.items())}); "
             f"`configs/eval/e1_c8_population.json` not yet written (scripts/c8_population.py, A6.16 (l))")
    s2 = f"`{ref['llm']['model']}` (structured output, `reasoning_effort: {ref.get('reasoning_effort')}`)"
    s1 = f"{S.s1_desc(e['backends'][p['arm']])}; Jev-only primary (A5.1)"
    modes = ("item-weighted accuracy over the BFCL tool-selection items of the six categories; argument-slot accuracy "
             "and When2Call secondary (A6.9); batch_by_state " + str(e.get("batch_by_state")))
    return [f"**{p['label']}**", p["claim"], f"{p['suite']} · `configs/decision/bfcl_when2call.yaml`", arms, s2, s1, n,
            f"NI on accuracy, m = {p['margin']} pt", NI_RULE, modes, p["defined"]]


def seeds_section(S: Sources) -> list[str]:
    fz = S.frozen
    fam: dict[str, int] = {}
    for t in fz["task_ids"]:
        fam[t.split("-")[1]] = fam.get(t.split("-")[1], 0) + 1
    ev = R.EVAL_SEED_RANGES
    L = ["## 2. Seed namespaces and frozen lists (A6.4 as amended by A6.13; `src/jevharness/bench/registry.py`)", "",
         "| Namespace | Seeds | Source of the number | Rule |", "|---|---|---|---|"]
    L.append(f"| Evaluation | {R.ranges_spec(ev)} (n = {fz['n']}) | `registry.EVAL_SEED_RANGES`; `{R.EVAL_TASKS_FILE}` "
             f"(sha256 `{fz['sha256']}`; families {', '.join(f'{k} {v}' for k, v in sorted(fam.items()))}; registered "
             f"{fz['registered']}) | the frozen task list every E2 arm runs the first `n_tasks` of; `scripts/prereg_check.py` "
             f"fails when the configs stop reproducing it; the runner refuses the range without `{R.EVAL_RUNS_ENV}=1`, "
             "which only the main runs after the freeze set (A6.13); no seed in it has ever been run |")
    L.append(f"| Retired | {R.ranges_spec(R.RETIRED_EVAL_RANGES)} | `registry.RETIRED_EVAL_RANGES` | the former evaluation "
             "range of A6.4: 100000–100049 were run once by the n = 50 E2b pilot resume on 2026-10-01 (Deviations), so the "
             "whole range is reserved — never evaluated, never trained on, never run again (A6.13) |")
    L.append(f"| Calibration | {PC.CAL_SEED0}–{PC.CAL_SEED0 + PC.CAL_N - 1} (n = {PC.CAL_N}) inside "
             f"{R.CALIBRATION_RANGE[0]}–{R.CALIBRATION_RANGE[1]} | `prereg_check.CAL_SEED0 / CAL_N`; "
             "`registry.CALIBRATION_RANGE` (BENCHMARK_SPEC §8) | the A6.5 calibration shadow arms "
             f"({', '.join(code(c) for c in PC.CAL_CONFIGS)}) that set the C1 thresholds; `grpo_online`'s default eval "
             "suite also lives here (A6.4) |")
    L.append(f"| Development | 0–{R.DEV_MAX_SEED} | `registry.DEV_MAX_SEED` | pilots, dev rounds, the CF-validity pilot "
             "and simulated runs used 0–149 (A6.4 manifest; prompts were changed on this evidence); 1000–4999 are the "
             "training-side dev defaults (`harvest.py --min-jevgym-seed 1000`) |")
    L.append(f"| Training | ≥ {R.TRAIN_MIN_SEED} minus the reserved ranges {R.ranges_spec(R.reserved_ranges())} | "
             "`registry.TRAIN_MIN_SEED`, `registry.reserved_ranges()`; `training/harvest.py --exclude-jevgym-seeds` "
             f"defaults to `{R.ranges_spec(R.reserved_ranges())}` | `data/train_v0` holds 10000–10029 and 12000–12059; the "
             "A5.1 competence-gate dev set (210 decisions) draws from 10005–12055, never an evaluation seed (A6.4, A6.9 #13) |")
    L += ["", f"- Runner guard: `{R.EVAL_RUNS_ENV}=1` is required for any jevgym suite touching an evaluation or retired "
          "range (`registry.check_suite_seeds`; `tests/test_registry.py`).",
          "- Frozen suite settings (A1, P4 calibration): " +
          ", ".join(f"`{k}: {v}`" for k, v in PC.SUITE_KEYS.items()) + "; checked on every evaluation and calibration config.",
          "- Evaluation configs and their suites (read from the YAMLs):"]
    for cfg in PC.EVAL_CONFIGS:
        s = S.exps[cfg].get("suite") or {}
        L.append(f"  - `{cfg}`: `{s.get('name')}` n = {s.get('n')}, seed0 = {s.get('seed0')} → "
                 f"{s.get('seed0')}–{int(s.get('seed0', 0)) + int(s.get('n', 0)) - 1}; default `n_tasks` "
                 f"{S.exps[cfg].get('n_tasks')}")
    for cfg in PC.CAL_CONFIGS:
        s = S.exps[cfg].get("suite") or {}
        L.append(f"  - `{cfg}` (calibration): `{s.get('name')}` n = {s.get('n')}, seed0 = {s.get('seed0')} → "
                 f"{s.get('seed0')}–{int(s.get('seed0', 0)) + int(s.get('n', 0)) - 1}; `n_tasks` {S.exps[cfg].get('n_tasks')}")
    ex = S.e1_excl
    L += [f"- C8 population (A6.13): the {ex['n']} E1 dev-prefix ids in `configs/eval/e1_dev_prefix_ids.json` (sha256 "
          f"`{ex['sha256']}`) are excluded from the confirmatory population through `exclude_ids_file` in "
          "`configs/decision/bfcl_when2call.yaml`; the registered pools are the BFCL categories "
          f"{', '.join(next(s for s in S.e1_cfg['suites'] if s['name'] == 'bfcl')['categories'])} (≤ "
          f"{next(s for s in S.e1_cfg['suites'] if s['name'] == 'bfcl')['max_per_cat']} each) and When2Call (≤ "
          f"{next(s for s in S.e1_cfg['suites'] if s['name'] == 'when2call')['max_items']}; test splits never used for training).",
          f"- τ² (A5.4 (c)): the 50 airline ids of tau2-bench `{S.airline['source']['describe']}` are committed in "
          f"`configs/tau2/airline_task_ids.json`; split seed {S.airline['split']['seed']} over the sorted ids → "
          f"{S.airline['split']['n_shadow']} shadow (τ̂\\* refit input only) + {len(S.airline['split']['test'])} test; "
          "retail (114 tasks) has no shadow split.", ""]
    return L


def fits_section(S: Sources) -> list[str]:
    L = ["## 3. Calibration fits: the deployed C1 thresholds (A6.5, A6.11 (c); `results/calibration/*.json`)", "",
         "One τ̂\\* per decision point and System-2 model from the calibration shadow arms (seeds "
         f"{PC.CAL_SEED0}–{PC.CAL_SEED0 + PC.CAL_N - 1}); deployed estimator `threshold_theory.scalar_threshold_projection` "
         "(approximate equal-mass confidence bins of ≥ 20 paired decisions formed from whole tie groups, every bin edge "
         "an observed value, so identical deployed confidences are never split and the bin count may be below ten "
         "(A6.16 (c)); advantage a(c) = g(c) − q_L(c) + ΔC/V per bin, accept region = bins with a ≥ 0, deployed "
         "τ̂\\* = the lower edge of the accepting upper tail, isolated accepting bins below it reported as islands and "
         "`tau_gain` reported, no accepting tail = always defer (A6.19)), operating point "
         "W = $1, λ = 0, tie rule `>=` on the wire's two-decimal confidence, full-precision thresholds (A6.11 (c).3, .7). "
         "V = W is the prespecified conservative assumption when paired continuations identify no positive stake "
         "(A6.11 (c).1), an upper bound, not an estimate; `fatal_share` is a reported diagnostic, never deployed (A6.11 (c).2).",
         "",
         "| Calibration arm | System-2 | Decision point | n paired | V | V_source | wrong-S2 episodes | ΔC | slots / call "
         "| τ̂\\* (fit) | accepted share | deployed in the C1 arm | fatal_share (n_fatal / n_wrong) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cal, (ccfg, c1, ecfg) in CAL_TO_C1.items():
        s2 = S.s2_model(ccfg, cal)
        for dp in CAL_DPS:
            hit = S.fits.get((cal, dp))
            d = S.cfgs[ecfg][c1].dp(DP(dp))
            dep = (f"`{c1}`.{dp}: always-defer" if d.extra.get("always_defer") else f"`{c1}`.{dp}: τ {fmt_tau(d.tau)}")
            if hit is None:
                L.append(f"| `{cal}` | `{s2}` | {dp} | — | — | **pending** (`tauhat_{cal}_{dp}.json` absent) | — | — | — "
                         f"| — | — | {dep} (placeholder) | — |")
                continue
            path, f = hit
            pr = f["primary"]
            proj = pr.get("projection", {})
            fs = f.get("fatal_share")
            if fs is None:
                fsc = "— (not in the artefact)"
                S.flags.append(f"`{path.name}` carries no `fatal_share`, though A6.11 (c).2 quotes a tool_select share "
                               "(Flash 56 / 67 = 0.84, Flash-Lite 68 / 81 = 0.84)")
            elif not fs.get("n_wrong"):
                fsc = "undefined (no wrong System-1 answer)"
            else:
                fsc = f"{fs['n_fatal']} / {fs['n_wrong']} = {fs['p_fatal']:.2f}"
            tau = f["tau_hat"]
            acc = proj.get("scalar_accept_rate")
            slots = f"{pr['slots_per_call']:.2f}" if "slots_per_call" in pr else "—"
            acc_s = (f"{acc:.3f} ({round(acc * proj.get('n', 0))} / {proj.get('n')})"
                     if isinstance(acc, (int, float)) else "—")
            L.append(f"| `{cal}` (`{path.name}`) | `{s2}` | {dp} | {f.get('n_paired')} | {pr['V']:g} (W = {pr['W']:g}) | "
                     f"{cell(pr['V_source'])} | {pr.get('n_wrong_episodes', '—')} | ${pr['delta_c']:.4g} "
                     f"({cell(pr.get('delta_c_source', 'ledger'))}) | {slots} | {fmt_tau(tau)} | {acc_s} | {dep} | {fsc} |")
    L += ["", "Diagnostics and audit facts recorded with the fits (A6.11):", "",
          "- Flash calibration log: 150 / 150 episodes with a right System-2 decision on tool_select and tool_params, so "
          "the counterfactual V is undefined there; Flash-Lite tool_select has 13 wrong-decision episodes with a "
          "non-positive counterfactual V; all four fits therefore use V = W (A6.11 (a), restated by (c).1).",
          "- τ̂\\* under V = W × fatal_share is unchanged for all four fits (A6.11 (c).2); the oracle-verdict "
          "classification of System-1's actual proposals on the n = 500 shadow logs and paired counterfactual "
          "continuations (under the A6.8 budget rule) are the other registered diagnostics / sensitivity analyses.",
          "- Jev wire audit (`cache/s1.sqlite`, 22,740 responses): confidence at two decimals, 4,950 / 9,809 choice "
          "answers at exactly 1.0, none in (0.995, 1); τ = 1.00 accepts the *reported-one set* (A6.11 (c).3).",
          "- ΔC for argument decisions is one avoided System-2 call split over that call's argument slots "
          "(`cost_terms(couple_by_step=True)`, A6.11 (c).5); the deployed configuration's ledger is authoritative.",
          "- Intern-Decision thresholds come from `jevgym_shadow_cal_intern.yaml` on the GPU host under the same rules "
          "(A6.5, A6.11, A6.12); each backend uses its own fit, and an always-defer outcome does not demote it.", ""]
    return L


TBSCI_POOL = ROOT / "adapters" / "pi" / "tbsci_hard_pool.txt"
TBSCI_POOL_SHA256 = "f131ee5289003d95facdb055bb308a18a9d68280ddabb2aeb1c9fd122c9261fe"   # A6.22 (d)
TBSCI_HARD = ROOT / "adapters" / "pi" / "tbsci_hard.txt"


def _sh_default(text: str, var: str) -> str | None:
    """The default of ``VAR="${VAR-x}"`` / ``VAR="${VAR:-x}"`` in a shell script ('' for an empty default)."""
    m = re.search(rf'{var}="\$\{{{var}:?-([^}}]*)\}}"', text)
    return m.group(1) if m else None


def hard_task_lines(S: Sources) -> list[str]:
    """H5's frozen subset and the hard-task caps (A6.22 (d), (e)), read from the adapters so that drift shows."""
    out = []
    pool = TBSCI_POOL.read_bytes() if TBSCI_POOL.exists() else b""
    sha = hashlib.sha256(pool).hexdigest()
    slugs = [ln.strip() for ln in pool.decode().splitlines() if ln.strip() and not ln.startswith("#")]
    if sha != TBSCI_POOL_SHA256:
        S.flags.append(f"H5: `{TBSCI_POOL.relative_to(ROOT)}` hashes to `{sha[:12]}`, A6.22 (d) registers "
                       f"`{TBSCI_POOL_SHA256[:12]}`")
    if TBSCI_HARD.exists():
        hb = TBSCI_HARD.read_bytes()
        n_hard = sum(1 for ln in hb.decode().splitlines() if ln.strip() and not ln.startswith("#"))
        hard = (f"frozen: `{TBSCI_HARD.relative_to(ROOT)}` ({n_hard} tasks, {len(hb):,} bytes, sha256 "
                f"`{hashlib.sha256(hb).hexdigest()}`)")
    else:
        hard = (f"`{TBSCI_HARD.relative_to(ROOT)}` is not written yet: it is produced by that algorithm from one "
                "calibration pass and frozen (sha256 and size in the Deviations section) before any evaluation trial")
    out.append(f"- Terminal-Bench-Science subset (A6.22 (d)): from the {len(slugs)}-task pool "
               f"`{TBSCI_POOL.relative_to(ROOT)}` (sha256 `{sha[:12]}`), drop tasks whose `oracle` misses reward 1 or "
               "runs over 1,800 s or whose build / verifier fails twice; keep a task iff one capped `baseline` attempt "
               "solves it or passes more verifier tests than the `nop` floor; fill N = 20 with 4 per domain (earth, "
               "engineering, life, mathematical, physical), lowest expert estimate first, ties by slug, unfilled slots "
               f"round-robin over the domains in that order; {hard}.")
    tbsci = (ROOT / "adapters" / "pi" / "run_tbsci.sh").read_text()
    tb2 = (ROOT / "adapters" / "pi" / "run_tb2.sh").read_text()
    swe = yaml.safe_load((ROOT / "adapters" / "mini_swe" / "swebench_jev.yaml").read_text())
    mult, turns = _sh_default(tbsci, "AGENT_TIMEOUT_MULT"), _sh_default(tbsci, "MAX_TURNS")
    hours = 8 * float(mult) if mult else None
    out.append("- Hard-task caps (A6.22 (e); read from the adapters, every arm of a suite under the same caps; a cap is "
               "a budget deviation stated in the paper): SWE-bench Verified `step_limit` "
               f"{swe['agent'].get('step_limit')}, `cost_limit` ${swe['agent'].get('cost_limit'):g} per instance, "
               f"{swe['environment'].get('timeout')} s per command (`adapters/mini_swe/swebench_jev.yaml`); "
               "Terminal-Bench 2.0 "
               + ("each task's own agent timeout" if not _sh_default(tb2, "AGENT_TIMEOUT_MULT") else
                  f"agent-timeout multiplier {_sh_default(tb2, 'AGENT_TIMEOUT_MULT')}")
               + (", no turn cap" if not _sh_default(tb2, "MAX_TURNS") else f", {_sh_default(tb2, 'MAX_TURNS')} turns")
               + " (`adapters/pi/run_tb2.sh` defaults; the main run sets `N_ATTEMPTS=3`); Terminal-Bench-Science "
               + (f"{hours:g} h of agent time (`AGENT_TIMEOUT_MULT={mult}` of the stated 8 h)" if hours else "no time cap")
               + (f" and {turns} turns (`MAX_TURNS`)" if turns else ", no turn cap") + " (`adapters/pi/run_tbsci.sh` defaults).")
    return out


def secondary_section(S: Sources) -> list[str]:
    named = {(p["cfg"], p["arm"]) for p in PRIMARIES if p["cfg"]} | {(p["cfg"], p["ref"]) for p in PRIMARIES if p["cfg"]}
    for p in PRIMARIES:
        if p.get("intern"):
            named |= {(p["intern"][0], p["intern"][1]), (p["intern"][0], p["intern"][2])}
    L = ["## 4. Secondary / exploratory (outside the Holm family; Holm or BH within their own tables, worded as exploratory)", ""]
    a622 = S.has_amendment("A6.22")
    # H5
    if a622:
        L += ["### H5 — \"passes the exploratory success screen; faster among jointly solved tasks\" on hard tasks "
              "(A5.2; renamed by A6.6 and A6.22 (d); bounded by A6.16 (j))", ""]
    else:
        L += ["### H5 — \"success-preserving and faster\" on hard tasks (A5.2, renamed by A6.6, bounded by A6.16 (j))", ""]
    L += ["- Exploratory, outside C1–C8, never a stopping or selection rule. Arms share the unmodified harness's tasks, "
          "System-2 model and caps, paired per task; k = 3 attempts on Terminal-Bench-Science and Terminal-Bench 2.0, "
          "k = 1 on SWE-bench Verified (the 37-task hard pool and its deterministic selection rule, the model ids and the "
          "adapters' time caps are what is frozen; backend hardware, concurrency, queueing, retries and tool-execution "
          "time are logged per run, arm order interleaved per task; A6.16 (j))."]
    if a622:
        L += hard_task_lines(S)
        L.append("- **Passes the exploratory success screen; faster among jointly solved tasks** (the registered verdict "
                 "`success_preserving_and_faster`, displayed so since A6.22 (d); only the label changed): the "
                 "**exploratory success screen** (one-sided 90 % task-bootstrap lower bound of Δsuccess > −5 pt **and** "
                 "wins ≥ losses; not a size-controlled NI test — with a ceiling reference and loss probability 0.05, zero "
                 "observed losses has probability ≈ 0.36 at n = 20) **and** *faster among jointly solved tasks* (median "
                 "paired Δ agent-run time < 0 on the tasks both arms solve, exact one-sided sign test on the per-task time "
                 "differences p < 0.05; A5.2's Wilcoxon is printed as descriptive). *Faster and better (faster among "
                 "jointly solved tasks)* (stricter tier, `strictly_better`): 90 % lower bound > 0 **and** exact one-sided "
                 "sign test p_wins < 0.05, with *faster*. Both tiers are reported for every arm; the all-task capped-time "
                 "median Δ and solved-by-t at t ∈ {600, 1800, 3600} s sit in the same row, never as the criterion (A6.6, "
                 "A6.22 (d)). A6.6's sentence that one capped attempt moves \"each median by at most one order "
                 "statistic\" is withdrawn: changing one binary trial outcome moves the all-task mean success by 1/(k·n), "
                 "and co-solved timing summaries can also change through their inclusion set (A6.22 (d)).")
    else:
        L.append("- *Success-preserving and faster* (the registered verdict): the **exploratory success-preservation screen** "
                 "(one-sided 90 % task-bootstrap lower bound of Δsuccess > −5 pt **and** wins ≥ losses; not a size-controlled "
                 "NI test — with a ceiling reference and loss probability 0.05, zero observed losses has probability ≈ 0.36 at "
                 "n = 20; worded \"passes the exploratory success-preservation screen\") **and** *faster on co-solved tasks* "
                 "(median paired Δ agent-run time < 0 with the exact one-sided sign test on the per-task time differences "
                 "p < 0.05; A5.2's Wilcoxon is printed as descriptive). *Faster and better* (stricter tier, `strictly_better`): "
                 "90 % lower bound > 0 **and** exact one-sided sign test p_wins < 0.05, with *faster*. Both tiers are reported "
                 "for every arm; all-task capped time and solved-by-time curves sit next to the co-solved median, never as the "
                 "criterion (A6.6).")
    L += ["- Readout: `metrics.paired_success_lb`, `sign_test_exact`, `wilcoxon_one_sided`, `h5_verdict` "
          "(`success_preserving`, `strictly_better`, `faster`, `time_sign_p`, `screen_label`); "
          "`adapters/pi/summarize_harbor.py --baseline <arm>`. Intern-Decision's local latency (≈ 330 ms per decision "
          "on one L4 vs ≈ 200 ms for Jev) can fail the *faster* tier on its own and is reported as such (A6.12).", ""]
    # tau sweeps
    L += ["### τ sweeps (A2: n = 200; the forced-τ diagnostic of A6.11 (c).4)", ""]
    for cfg in PC.EVAL_CONFIGS:
        sw = sorted(a for a in S.counts[cfg] if "_cascade_tau" in a and (cfg, a) not in named)
        if sw:
            by_s2: dict[str, list[str]] = {}
            for a in sw:
                by_s2.setdefault(S.s2_model(cfg, a), []).append(a)
            for s2, arms in by_s2.items():
                L.append(f"- `{cfg}` on `{s2}`: " + ", ".join(f"`{a}` (n = {S.n(cfg, a)})" for a in arms) +
                         f"; tool_params τ {fmt_tau(S.cfgs[cfg][arms[0]].tool_params.tau)} fixed")
    lite_sweep = [a for a in S.counts["jevgym_tool_select.yaml"] if a.startswith("jev_ts_cascade_tau") and
                  S.s2_model("jevgym_tool_select.yaml", a) == S.s2_model("jevgym_tool_select.yaml", "llm_only_s2lite")
                  and a != "jev_ts_cascade_tau0.9_s2lite"]
    if not lite_sweep:
        pass  # A6.15: no Flash-Lite forced-threshold diagnostic arm exists; the C1L diagnostic is "none" (settled)
        L.append("- No Flash-Lite Jev cascade sweep arm exists (see the consistency notes).")
    L.append("")
    # routing sweep
    L += ["### Routing sweep and matched controls", ""]
    for cfg in ("jevgym_meta_control.yaml", "baselines_jevgym.yaml"):
        rt = sorted(a for a in S.counts[cfg] if "route" in a)
        L.append(f"- `{cfg}`: " + ", ".join(f"`{a}` (n = {S.n(cfg, a)}, τ {fmt_tau(S.cfgs[cfg][a].routing.tau)})" for a in rt))
    L.append("")
    # measurement arms
    L += ["### Measurement arms (A4; shadow-only: never in Holm, iso-quality, Pareto or as a reference)", ""]
    for a in ("meta_shadow", "meta_shadow_s2lite"):
        cfg = "jevgym_meta_control.yaml"
        c = S.cfgs[cfg][a]
        shadow = [dp.value for dp in DP if c.dp(dp).mode == Mode.SHADOW]
        pr = c.measure.probes
        cf = c.measure.cf
        L.append(f"- `{a}` (`{S.s2_model(cfg, a)}`, n = {S.size(cfg, a)}): shadows {', '.join(shadow)}; probes stop "
                 f"{pr.stop if pr else '—'}, tool_select {pr.tool_select if pr else '—'}, tau_ref {pr.tau_ref if pr else '—'}, "
                 f"transplant `{pr.transplant_llm if pr else '—'}` at T = {pr.transplant_temperature if pr else '—'}; "
                 + (f"counterfactual branches: reflection {cf.reflection}, large_llm `{cf.large_llm}`, stop_reject "
                    f"{cf.stop_reject}, placebo {cf.placebo}, sample_negatives {cf.sample_negatives}" if cf else
                    "no counterfactual branches")
                 + (". H2 scores, C5's secondary readouts, C6's probe FAR and the ≥ 30-positive counts for C4 / H2 cells "
                    "come from here; Intern-Decision is scored by replaying its logged states (A5.1, A6.2)" if a.endswith("s2lite") else ""))
    L += ["", "### Counterfactual-validity run (A6.8)", "",
          "- `meta_shadow_s2lite` through `scripts/cf_validity.py --mode live`: N = 150 opportunities, stratified "
          "(every `helps` and `harms` up to 50 each, ≥ 60 non-final `neutral`, the rest seeded-random), R = 3 "
          "replicates per branch (≈ 900 rollouts); task-cluster Wilson / bootstrap intervals.",
          "- e and h are *validated* iff binary agreement (one-step `helps` ↔ full-episode Δ > 0) ≥ 0.8 with a Wilson "
          "95 % lower bound ≥ 0.7 **and** the non-final-`neutral` stratum's mean Δ has a 95 % task-bootstrap CI within "
          "±0.05; otherwise they are *unvalidated one-step estimates* and the H2 theory readout is labelled so. The "
          "3 × 3 table and κ are descriptive; ALFWorld repeats the protocol if the budget allows (secondary).", "",
          "### Evaluation shadow arms (H1 regret analysis and the Prop. 1 tests; never the deployed τ)", ""]
    for cfg, arms in (("jevgym_tool_select.yaml", ("jev_shadow", "jev_shadow_s2lite")),
                      ("jevgym_tool_select_intern.yaml", ("intern_shadow", "intern_shadow_s2lite"))):
        L.append(f"- `{cfg}`: " + ", ".join(f"`{a}` (`{S.s2_model(cfg, a)}`, n = {S.size(cfg, a)})" for a in arms) +
                 "; τ̂\\* regret vs the swept τ and vs τ = 0.8 on the registered 30/70 split (`--train-frac 0.3`), "
                 "W ∈ {0.1, 1, 10}, λ ∈ {0, 0.5, 5} $/h (§3 H1, A6.5).")
    # H2 / H3 / H4 short
    L += ["", "### Other registered hypotheses", "",
          "- **H2** (A6.2 replaces §3's test; A6.16 (i) fixes the inference): per estimable cell (≥ 30 positive and ≥ 30 "
          "negative golds, A4) the theory check is the equation-based comparison of the predicted ΔÛ = "
          "W·n̄·π(1−π)·(e+h)·J_τ − κ·n̄ with the observed ΔU of the live gate over its matched-rate control at W = $1 "
          "(gross of System-1's query cost, net alongside), with shared task ids resampled jointly across cells and the "
          "predictors recomputed in every resample; the pooled regression of ΔU on J_τ (TPR − FPR at the live τ) and on "
          "accuracy (LR test) is exploratory. Minimum requirements: ≥ 4 estimable cells and full column rank for any "
          "regression, otherwise H2 is *not estimable*, never falsified. The C5 control's firing rate is fixed from the "
          "retired-seed n = 50 pilot before the main runs and treated as fixed. AUROC regression and the ±2-pt TOST are "
          "exploratory / descriptive. Guards use e_raw / h_raw. Sensitivity row without budget-kind fatals (A6.7); "
          "implication-audit violation count must be 0 (A6.7).",
          ]
    if a622:
        L += ["- **H1** (§3, secondary; verdict rules A6.22 (c)): 72 cells = {tool_select, tool_params} × {Jev, "
              "Intern-Decision-4B} × {Flash, Flash-Lite} × W ∈ {0.1, 1, 10} × λ ∈ {0, 0.5, 5} $/h from the n = 500 "
              "evaluation shadow arms (`--train-frac 0.3 --seed 0 --taus h1 --rule general`; `--V W` where the "
              "counterfactual V is undefined or not positive); per cell d = held-out loss of τ = 0.8 minus that of τ̂\\* "
              "(`regret.default_minus_hat`), win d > 0, tie |d| ≤ 1e-12 (a non-win), one vote per cell, absent cells "
              "listed and out of the denominator (< 36 estimable: *not estimable*); clause 1 holds iff wins > half the "
              "estimable cells (point estimates; per-cell 95 % CIs and the counts above / below 0 printed); clause 2 at "
              "W = $1, λ = 0: a (backend, decision point, System-2) cell is over-confident iff mean confidence minus "
              "accuracy on its training split has a 95 % task-bootstrap lower bound > 0, and returns to the front iff the "
              "held-out regret of τ = 0.8 is credibly positive (95 % CI lower bound > 0) on raw and not on recalibrated "
              "confidences; clause 2 holds iff ≥ half the over-confident cells return (none: not applicable). Monotone "
              "recalibration cannot move τ̂\\*, the deployed projection or the attainable front (rank-based); clause 2 "
              "measures a mis-set fixed threshold, never a frontier change. Supported iff clause 1 holds and clause 2 "
              "holds or is not applicable.",
              "- **H3** (§3, exploratory; mechanical rules A6.22 (c)): universe = acting Jev / Intern-Decision-4B arms per "
              "suite × System-2 model (L2: the JevGym evaluation configs; L3: τ² airline test tasks and ALFWorld; "
              "controls, stand-ins, shadow arms and references excluded). Class of an executed System-1 decision: meta "
              "(reflection, routing, stop, memory gates); bounded-recoverable (tool / action assist or guard, executed "
              "non-irreversible tool or action choices, executed bounded arguments); copy-dependent (executed copy "
              "arguments, A4); irreversible (executed choices of `threshold_theory.JEVGYM_FATAL_TOOLS`, τ² WRITE tools and "
              "`transfer_to_human_agents`). An arm is credited with a class iff those decisions are ≥ 5 % of their "
              "decision point's opportunities (A6.20 (e)); assist / guard / trigger act at every opportunity; uncredited "
              "arms are *negligible delegation* and leave the analysis; multi-class arms rank under the riskiest class. "
              "NI = the `noninferior` qualifier; front = not (cost, success)-dominated by a qualifying arm or the "
              "reference — descriptive, n printed. Ranking statistic: the class median of `vs_baseline.cost_savings` over "
              "qualifying arms (0 without one); Kendall τ-b over the four classes, Flash vs Flash-Lite and L2 vs L3, exact "
              "permutation p. Falsified iff a credited copy-dependent or irreversible arm is NI and on the front, or a "
              "τ-b < 0; otherwise \"no registered counterexample\". Cross-family clause untested while only one System-2 "
              "family runs (§5, A3).",
              "- **H4** (§3, training, secondary; A6.22 (c)): checkpoints matched on the v1 development set "
              "(`training.eval_decisions`, `data/train_v1/dev.jsonl.gz`; |acc(R4) − acc(R3)| ≤ 1 pt, the matched R4 "
              "checkpoint closest to R3, ties → later step; none: *not tested at matched accuracy*); downstream on the "
              "first 200 frozen evaluation tasks with each checkpoint as System-1 of `jev_ts_params_replace` (Flash); "
              "supported iff the one-sided 95 % lower bound of Δsuccess (R4 − R3) > 0, falsified iff TOST shows "
              "equivalence within ±3 pt (90 % CI inside (−3, +3) pt, `metrics.tost_paired`), otherwise *inconclusive at "
              "this n*."]
    else:
        L += ["- **H3** (§3): the front of NI configurations per suite and System-2 family by delegation class (meta; "
              "bounded-recoverable; copy-dependent = the copy class only, A4; irreversible); Kendall τ of the class ranking "
              "across families and L2 → L3; cross-family clause reported as untested while only one System-2 family runs (§5, A3).",
              "- **H4** (§3, training, secondary): counterfactual Monte-Carlo targets vs teacher distillation at matched "
              "decision accuracy; falsified if the downstream difference is inside ±3 pt."]
    L += ["- **C7 proxy check** (A5.4 (e)): Kendall τ between the E2 (Flash) and τ²-airline rankings of every "
          "configuration that ran on both, with n and a permutation p.", ""]
    # every other arm
    L += ["### Every other arm, by config and size (read from the YAMLs; A2: default n = 200)", ""]
    listed = set(named)
    for p in PRIMARIES:
        if p["label"] == "C4":
            listed |= {("jevgym_meta_control.yaml", a) for a in S.counts["jevgym_meta_control.yaml"] if a.startswith("jev_route_tau")}
    for cfg in PC.EVAL_CONFIGS:
        rest = [a for a in S.counts[cfg] if (cfg, a) not in listed]
        if not rest:
            continue
        by_n: dict[int, list[str]] = {}
        for a in rest:
            by_n.setdefault(S.n(cfg, a), []).append(a)
        L.append(f"- `{cfg}`:")
        for n_, arms in sorted(by_n.items(), key=lambda kv: -kv[0]):
            L.append(f"  - n = {n_}: " + ", ".join(f"`{a}`" for a in arms))
    ph = sorted(k for k, v in (S.exps["baselines_jevgym.yaml"].get("s1_backends") or {}).items() if v.get("placeholder"))
    if ph:
        L.append(f"- Matched-rate placeholders in `baselines_jevgym.yaml` (`placeholder: true`, their arms are skipped "
                 f"until the rate is set from this run's Jev arm): {', '.join(code(k) for k in ph)}.")
    L.append("")
    return L


def deviations_section(S: Sources) -> list[str]:
    lines = S.prereg.splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == "## Deviations")
    except StopIteration:
        return ["## 5. Deviations so far", "", "(PREREGISTRATION.md has no `## Deviations` section)", ""]
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
    body = lines[i + 1:j]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return ["## 5. Deviations so far (verbatim from PREREGISTRATION.md `## Deviations`)", "",
            "Effective reading: the first entry's \"Flash-Lite tool_select never → 0.99 (A6.11 (b))\" is superseded by "
            "A6.11 (c).4 (always defer, as deployed in the YAML); the second entry's exposure is resolved by A6.13 "
            "(evaluation seeds 200000–200499, never run); the third entry (A6.16 (f) validation) is acted on by A6.17 "
            "(the Berger–Boos exact p replaces the asymptotic score p before any main run; the (f) contingency is "
            "withdrawn); A6.21 replaces A6.17's decisive p by the certified one.", ""] + body + [""]


def open_section(S: Sources) -> list[str]:
    L = ["## 6. Open before the freeze (2026-10-03)", ""]
    ex_path = ROOT / "results" / "reviews" / "2026-10-02" / "ni_exact_size_r8.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text())
        sts = ex.get("settings", [])
        dense = [(st["n"], lv, v) for st in sts for lv, v in st.get("bb", {}).get("dense_max", {}).items()
                 if lv in ex["holm_levels"]]
        worst = max((v["ratio"] for _, _, v in dense), default=float("nan"))
        inner = max((v["ratio"] for st in sts for lv, v in st.get("bb", {}).get("interior_max", {}).items()
                     if lv in ex["holm_levels"]), default=float("nan"))
        above = sum(1 for _, _, v in dense if v["ratio"] > 1.0)
        L.append(f"- **NI rule, exact size (A6.21 (a) 5).** `{ex_path.relative_to(ROOT)}` ({ex.get('date')}): the deployed "
                 f"A6.21 rule (`rule = bb_ordered`) is ≤ {worst:.3f} α on the boundary and ≤ {inner:.2f} α inside the null "
                 f"at every Holm level, {above} cell(s) above α, over n ∈ {{{', '.join(str(st['n']) for st in sts)}}}. "
                 "Still to do: add the final retained n′ of each NI pair to `SETTINGS` and re-run before analysis; a size "
                 "above α would indicate an implementation error and stops the analysis until it is found (A6.21 (a) 8).")
    else:
        L.append("- **NI rule, exact size (A6.21 (a) 5).** `results/reviews/2026-10-02/ni_exact_size_r8.json` is missing: "
                 "run `python scripts/ni_exact_size.py` before the tag.")
    if not (ROOT / "configs" / "eval" / "e1_c8_population.json").exists():
        L.append("- **C8 population counts** (A6.16 (l)): `configs/eval/e1_c8_population.json` (per-category item counts after "
                 "the parent-item exclusion) is written before the C8 run, not yet.")
    if "FREEZE_COMMIT" in S.prereg:
        L.append("- **`FREEZE_COMMIT` placeholder** (A6.9 #12): the header's \"Frozen 2026-10-03 at commit `FREEZE_COMMIT`\" "
                 "is replaced by the hash at the tag (`git tag prereg-freeze`; `prereg_check` then verifies a hash is present) "
                 "and the same hash goes into the paper's §6 `\\tbd` (HANDOFF §7.1 step 3).")
    pend = [(cal, dp) for cal in CAL_TO_C1 for dp in CAL_DPS if (cal, dp) not in S.fits]
    if pend:
        L.append("- **Intern-Decision calibration fits (GPU round 3).** Missing: " +
                 ", ".join(f"`tauhat_{c}_{d}.json`" for c, d in pend) +
                 "; the Intern C1 arms carry placeholder τ (" +
                 "; ".join(f"`{CAL_TO_C1[c][1]}` {S.modes(CAL_TO_C1[c][2], CAL_TO_C1[c][1])}" for c in dict.fromkeys(c for c, _ in pend)) +
                 ") until `jevgym_shadow_cal_intern.yaml` has run on the GPU host and the fits are written under the "
                 "A6.5 / A6.11 rules (before any Intern C1 arm runs; A6.12). GPU round 3 also re-runs the OOM-affected "
                 "Intern assist / cascade pilots with the shared-weights fix (A6.12).")
    gvp = ROOT / "results" / "gpu_round2" / "gate_verdict.json"
    if gvp.exists():
        gv = json.loads(gvp.read_text())
        c4 = gv["candidates"]["intern_4B"]
        L.append(f"- **Competence gate verdict** (A5.1, path per A6.16 (b) / A6.18 (e)): done — `{gvp.relative_to(ROOT)}`: "
                 f"Intern-Decision-4B accuracy {c4['acc']:.3f} vs Jev {gv['jev_acc']:.3f} on {gv['n']} dev decisions, ratio "
                 f"{c4['ratio_to_jev']:.3f} (95 % {c4.get('ratio_ci95', ['—', '—'])[0]}–{c4.get('ratio_ci95', ['—', '—'])[1]}, "
                 "stratified binomial bootstrap, descriptive) ≥ 0.8 → co-primary; 2B / 0.8B fail. Nothing open.")
    else:
        L.append("- **Competence gate verdict** (A5.1 / A6.16 (b)): `results/gpu_round2/gate_verdict.json` is missing; it must be "
                 "committed before any C1 / C2 / C6 arm runs.")
    L.append("- **A5.1 prices**: `internlm/Intern-Decision-*` in `configs/prices/models.yaml` are an assumed amortised GPU "
             "cost until `scripts/measure_latency.py` has run on the real GPU.")
    rl = (S.exps["baselines_jevgym.yaml"].get("s1_backends") or {}).get("ratem_lite") or {}
    L.append(f"- **Matched-rate controls** (A4 as amended by A6.16 (i), scoped by A6.18 (f)): done for C5 — "
             f"`s1_backends.ratem_lite` rate {rl.get('rate')} fixed from the retired-seed n = 50 pilot's live arm "
             "(47 / 330; Deviations; locked by `prereg_check.C5_CONTROL_RATE`), never from the evaluation run. Still open: "
             "the secondary-row controls `ratem_stop_lite`, `ratem_route`, `ratem_memw` keep A4's rule (set from this run's "
             "System-1 arm before the control runs; realised rates reported).")
    L.append("- **τ² System-2 model**: registered by A6.15 (agent `gemini/gemini-3.5-flash`, temperature 0, minimal "
             "thinking; user simulator and routing small model `gemini/gemini-3.5-flash-lite`); `scripts/run_tau2.sh` "
             "defaults to it (a consistency note appears above if the defaults drift). Still to run: the 15-task airline "
             "shadow refit before the C7 arms.")
    L.append("- **Second System-2 family** (A3, A6.9 #15): Gemma 4 is quota-limited; H3's cross-family clause is "
             "reported as untested unless another family runs; the paper keeps `\\todo{A non-Google family (P1-3)}`.")
    if S.has_amendment("A6.22"):
        L.append("- **A6.22 code items without a symbol check** (slip rule A6.22 (g)): `scripts/primaries.py` reads the "
                 "tool-decision and argument coverage and applies the 5 % rule to the replication label ((a) 7) and lists "
                 "the dropped tasks with their characteristics next to the full-set row ((e) 5); the τ² sample-size "
                 "switch in `threshold_theory.fit_calibration_arm` ((b) 5); H1 clause 2's over-confidence gap and "
                 "recalibrated-default regret in `threshold_theory` ((c)); the Terminal-Bench-Science selection "
                 "(`tbsci.py`, (d) 2) before the calibration pass; the complete-case H5 view in "
                 "`adapters/pi/summarize_harbor.py` ((e) 5); the automatic τ² re-run of errored simulations (A6.20 (f)).")
    # slip-rule code
    landed, pending = [], []
    for label, sym, rel in SLIP_CODE + (SLIP_CODE_A622 if S.has_amendment("A6.22") else []):
        p = ROOT / rel if rel.startswith(("adapters/", "scripts/")) else ROOT / "src" / "jevharness" / rel
        (landed if p.exists() and sym in p.read_text() else pending).append(label)
    if pending:
        L.append("- **Slip-rule code still missing** (A5.6 / A6.10 \"may land later, before the run or analysis it "
                 "serves\"; anything missing at analysis time is a dated deviation): " + "; ".join(pending) + ".")
    L.append("- Slip-rule code present in `src/jevharness` (checked by symbol): " + "; ".join(landed) + ".")
    L += ["- **Before submission, not freeze items**: exhaustive-planner certification of budget-kind fatals (A6.7, "
          "P1-4); the score card's iso-quality default (A6.9 #10, P1-9); ledger fields for the prices wording (A6.9, "
          "P1-3); the P1 list of `docs/REVIEW_ROUND6.md` §2.", ""]
    return L


def render() -> str:
    S = Sources()
    body: list[str] = []
    # the consistency notes are collected while rendering the later sections, so render them first and splice
    sec_seeds = seeds_section(S)
    sec_fits = fits_section(S)
    sec_sec = secondary_section(S)
    sec_dev = deviations_section(S)
    sec_open = open_section(S)
    sec_prim = primaries_table(S)       # last: it prints every flag the other sections raised
    body += header(S) + sec_prim + sec_seeds + sec_fits + sec_sec + sec_dev + sec_open
    return "\n".join(body).rstrip() + "\n"


def strip_generated(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(GEN_PREFIX))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing; exit 1 on drift")
    a = ap.parse_args(argv)
    text = render()
    out = Path(a.out)
    if a.check:
        if not out.exists():
            print(f"{out} is missing; run python scripts/prereg_effective.py")
            return 1
        if strip_generated(out.read_text()) != strip_generated(text):
            print(f"{out} has drifted from the configs / fits / registration; run python scripts/prereg_effective.py")
            return 1
        print(f"{out}: up to date")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
