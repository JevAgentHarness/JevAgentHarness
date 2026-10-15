#!/usr/bin/env python3
"""Mechanical check of the experiment configs against the pre-registration (docs/PREREGISTRATION.md) before the
freeze: registered arm names exist, the registered sizes (A2 sign-off, A5.1, A5.3, A4) and seed ranges (A6.13: evaluation
200000..200499, 100000..100499 retired; A6.5: calibration 5000..5149) hold, the calibration shadow arms mirror the evaluation shadow arms, and
every arm name the registration mentions in backticks resolves to an arm of some config. Exit 1 on any mismatch.

    python scripts/prereg_check.py            # report
    python scripts/prereg_check.py --quiet    # only failures

The registry below is the human-readable summary of what the registration says; when the registration changes (append-
only amendments), update the registry here in the same commit. tests/test_prereg_check.py runs this check offline."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVAL_SEED0, EVAL_N = 200000, 500          # A6.13 (100000-100499 retired: exposed before the freeze)
RETIRED = (100000, 100499)
CAL_SEED0, CAL_N = 5000, 150              # A6.5
SUITE_KEYS = {"families": "v2", "distractor_style": "near", "n_distractors": 60, "noise_lines": 20, "strict": True}  # A1

# config -> {arm: registered n_tasks}; arms not listed keep the config default (200, A2). Sources in comments.
REGISTERED_SIZES: dict[str, dict[str, int]] = {
    "jevgym_tool_select.yaml": {                      # A2 sign-off (Flash) + A5.3 (Flash-Lite) + A6.5 (C1 arms)
        "llm_only": 500, "jev_shadow": 500, "jev_ts_assist_k3": 500, "jev_ts_cascade_tauhat": 500,
        "llm_only_s2lite": 500, "jev_shadow_s2lite": 500, "jev_ts_assist_k3_s2lite": 500,
        "jev_ts_cascade_tau0.9_s2lite": 500},
    "jevgym_meta_control.yaml": {                     # A2 sign-off + A5.3 (C6) + A4 (measurement arm) + A6.15 (C4)
        "llm_only": 500, "jev_stop_guard": 500, "llm_only_s2lite": 500, "jev_stop_guard_s2lite": 500,
        "meta_shadow_s2lite": 500, "meta_shadow": 200, "jev_route_tau0.4": 500, "small_only": 500},
    "baselines_jevgym.yaml": {                        # A4 (C5 pair) + A6.15 (C3 control)
        "jev_reflect_trigger_s2lite": 500, "ratem_reflect_trigger_s2lite": 500, "bm25_ts_assist_k3": 500},
    "jevgym_tool_select_intern.yaml": {               # A5.1: Intern-Decision C1/C2/C6 arms and references at 500
        "llm_only": 500, "intern_shadow": 500, "intern_ts_assist_k3": 500, "intern_ts_cascade_tauhat": 500,
        "intern_stop_guard": 500, "llm_only_s2lite": 500, "intern_shadow_s2lite": 500,
        "intern_ts_assist_k3_s2lite": 500, "intern_ts_cascade_tau0.9_s2lite": 500, "intern_stop_guard_s2lite": 500},
    "jevgym_shadow_cal.yaml": {"jev_shadow_cal": 150, "jev_shadow_cal_s2lite": 150},                  # A6.5
    "jevgym_shadow_cal_intern.yaml": {"intern_shadow_cal": 150, "intern_shadow_cal_s2lite": 150},     # A6.5
}
# C1 arms: the deployed tau per decision point = the fitted tau-hat* (A6.5 / A6.11, PREREGISTRATION Deviations 2026-10-01)
C1_TAUS: dict[str, dict[str, dict[str, float]]] = {
    "jevgym_tool_select.yaml": {"jev_ts_cascade_tauhat": {"tool_select": 1.0, "tool_params": 0.99},
                                "jev_ts_cascade_tau0.9_s2lite": {"tool_select": 0.99, "tool_params": 0.13}},
    # Intern-Decision fits, 2026-10-02 (GPU round 3 calibration logs, fitted with the registered command; A6.19)
    "jevgym_tool_select_intern.yaml": {"intern_ts_cascade_tauhat": {"tool_select": 0.99, "tool_params": 0.6309452181036016},
                                       "intern_ts_cascade_tau0.9_s2lite": {"tool_select": 0.99, "tool_params": 0.8518233437140553}},
}
ALWAYS_DEFER = {"jevgym_tool_select.yaml": {"jev_ts_cascade_tau0.9_s2lite": ["tool_select"]},     # A6.11 (c): "never"
                "jevgym_tool_select_intern.yaml": {"intern_ts_cascade_tauhat": ["tool_select"],
                                                   "intern_ts_cascade_tau0.9_s2lite": ["tool_select"]}}
# A6.16: C1 arms whose tau-hat* is not fitted yet carry `calibration: pending` (the runner refuses them); drop the
# entry here when the fit lands and add the arm to C1_TAUS
PENDING_TAUS: dict[str, dict[str, list[str]]] = {}   # the Intern-Decision fits landed 2026-10-02 (GPU round 3)
EVAL_CONFIGS = ["jevgym_tool_select.yaml", "jevgym_meta_control.yaml", "jevgym_tool_select_intern.yaml",
                "baselines_jevgym.yaml"]
CAL_CONFIGS = ["jevgym_shadow_cal.yaml", "jevgym_shadow_cal_intern.yaml"]
# calibration arm -> (evaluation config, evaluation shadow arm it must mirror in modes and System-2 model)
CAL_MIRRORS = {"jev_shadow_cal": ("jevgym_tool_select.yaml", "jev_shadow"),
               "jev_shadow_cal_s2lite": ("jevgym_tool_select.yaml", "jev_shadow_s2lite"),
               "intern_shadow_cal": ("jevgym_tool_select_intern.yaml", "intern_shadow"),
               "intern_shadow_cal_s2lite": ("jevgym_tool_select_intern.yaml", "intern_shadow_s2lite")}
# backticked tokens in PREREGISTRATION.md that look like arm names but are run directories / files / spec ids
NOT_ARMS = re.compile(r"^(pilot_|runs?_|e[0-9]|results|data|cache|configs|scripts|docs|train_|dev_|gate_|replay_|"
                      r"tauhat_|summary|report|metrics|ledger|experiment|seed|n_tasks|task_|ts_|s1_|s2_|gold_|fatal_|"
                      r"cf_|meta_gold|cost_|latency_|input_|cached_|retry_|wall_|final_|s1|accepted_|defer_|"
                      r"paired_|sign_|wilcoxon_|h[0-9]_|ni_|holm_|newcombe_|tango_|predicted_|implication_|"
                      r"scalar_|general_|prop1|threshold_|option_|clip_|q_|audit_|strictly_|airline_|jev_shadow\.yaml|jev_attempts\.json|"
                      r"jevgym_|sim_|alfworld_|tau2|measure_|check_|bench_|serve_|train_decision|harvest|opd|"
                      r"max_|min_|noise_|n_|distractor_|enable_|reasoning_|thought_|previous_|logprobs|tool_choice|"
                      r"openai_|intern_decision$|llm_generative|llm_logprob|typesafe|internlm|openrouter|gemini)")
# A6.16 (i): the C5 matched-rate control's firing rate, fixed from the retired-seed n = 50 pilot's live arm (47 / 330)
C5_CONTROL_RATE = ("baselines_jevgym.yaml", "ratem_lite", 0.1424)
ARM_PREFIX = re.compile(r"^(jev_|intern|lite_|llm_only|meta_shadow|mem_|reflect|small_only|bm25|embed|ratem|qwen|"
                        r"openjev|gemma)")


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "experiments" / name).read_text())


def arm_names(exp: dict) -> dict[str, int]:
    from jevharness.bench.runner import arm_task_counts

    return arm_task_counts(exp)


def arm_spec(exp: dict, name: str) -> dict | None:
    for a in exp.get("arms", []) or []:
        if a.get("name") == name:
            return a
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    fails: list[str] = []
    notes: list[str] = []
    exps = {n: load(n) for n in set(EVAL_CONFIGS + CAL_CONFIGS)}
    counts = {n: arm_names(e) for n, e in exps.items()}

    # 1. registered sizes
    for cfg, reg in REGISTERED_SIZES.items():
        for arm, n in reg.items():
            got = counts[cfg].get(arm)
            if got is None:
                fails.append(f"{cfg}: registered arm `{arm}` is missing")
            elif got != n:
                fails.append(f"{cfg}: `{arm}` runs n_tasks={got}, registered {n}")
            else:
                notes.append(f"{cfg}: `{arm}` n={n} ok")
    # 2. seed ranges and the frozen suite settings (A1 / A6.4 / A6.5)
    for cfg in EVAL_CONFIGS:
        s = exps[cfg].get("suite", {})
        if s.get("name") == "jevgym":
            if s.get("seed0") != EVAL_SEED0 or int(s.get("n", 0)) != EVAL_N:
                fails.append(f"{cfg}: suite seed0/n = {s.get('seed0')}/{s.get('n')}, registered {EVAL_SEED0}/{EVAL_N}")
            for k, v in SUITE_KEYS.items():
                if s.get(k) != v:
                    fails.append(f"{cfg}: suite.{k} = {s.get(k)!r}, frozen A1 value {v!r}")
            top = int(exps[cfg].get("n_tasks", 0) or 0)
            if top > EVAL_N:
                fails.append(f"{cfg}: n_tasks {top} exceeds the suite")
    for cfg in CAL_CONFIGS:
        s = exps[cfg].get("suite", {})
        if s.get("seed0") != CAL_SEED0 or int(s.get("n", 0)) != CAL_N or int(exps[cfg].get("n_tasks", 0)) != CAL_N:
            fails.append(f"{cfg}: calibration suite seed0/n/n_tasks = {s.get('seed0')}/{s.get('n')}/"
                         f"{exps[cfg].get('n_tasks')}, registered {CAL_SEED0}/{CAL_N}/{CAL_N}")
        for k, v in SUITE_KEYS.items():
            if s.get(k) != v:
                fails.append(f"{cfg}: suite.{k} = {s.get(k)!r}, frozen A1 value {v!r}")
        for i in range(CAL_N):  # the calibration seeds never overlap the evaluation range
            if EVAL_SEED0 <= CAL_SEED0 + i < EVAL_SEED0 + EVAL_N:
                fails.append(f"{cfg}: calibration seed {CAL_SEED0 + i} lies in the evaluation range")
                break
    # 2b. no config touches the retired range; the frozen task list matches the configs (A6.13)
    for cfg in EVAL_CONFIGS:
        s0 = int((exps[cfg].get("suite") or {}).get("seed0", 0) or 0)
        if RETIRED[0] <= s0 <= RETIRED[1]:
            fails.append(f"{cfg}: suite.seed0 {s0} lies in the retired range {RETIRED}")
    frozen = ROOT / "configs" / "eval" / "jevgym_eval_tasks.json"
    if frozen.exists():
        import hashlib
        import json as _json

        from jevharness.bench.suites import load_suite

        fz = _json.loads(frozen.read_text())
        suite = dict((exps["jevgym_tool_select.yaml"].get("suite") or {}))
        facs = load_suite(suite.pop("name"), **suite)
        ids = [f().task_id for f in facs]
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if ids != fz.get("task_ids") or digest != fz.get("sha256"):
            fails.append(f"frozen task list {frozen.name}: the configs' suite does not reproduce it (sha256 {digest[:12]} vs {str(fz.get('sha256'))[:12]})")
        else:
            notes.append(f"frozen task list: {len(ids)} ids, sha256 {digest[:12]} reproduced from the configs")
        for cfg in EVAL_CONFIGS:
            sc = dict(exps[cfg].get("suite") or {})
            if sc.get("name") == "jevgym" and (sc.get("seed0"), sc.get("n")) != (suite.get("seed0"), suite.get("n")):
                fails.append(f"{cfg}: suite seed0/n differ from the frozen list")
    else:
        fails.append("configs/eval/jevgym_eval_tasks.json is missing (scripts/freeze_eval_tasks.py)")
    # 2c. the C5 matched-rate control is fixed from the retired pilot, not a placeholder (A6.16 (i))
    cfg, backend, rate = C5_CONTROL_RATE
    spec = ((exps[cfg].get("s1_backends") or {}).get(backend) or {})
    if spec.get("placeholder") or abs(float(spec.get("rate") or 0.0) - rate) > 1e-9:
        fails.append(f"{cfg}: s1_backends.{backend} must be {{type: rate, rate: {rate}}} without `placeholder` (A6.16 (i)); "
                     f"found {spec}")
    else:
        notes.append(f"{cfg}: C5 control `{backend}` rate {rate} fixed from the retired pilot (A6.16 (i))")
    # 2d. the frozen list carries the task-generating configuration and the generator's revision (A6.16 (l))
    if frozen.exists():
        import hashlib as _hl
        fz = _json.loads(frozen.read_text())
        gen = ROOT / "src" / "jevharness" / "bench" / "suites" / "jevgym.py"
        gen_sha = _hl.sha256(gen.read_bytes()).hexdigest()
        suite_sha = _hl.sha256(_json.dumps(exps["jevgym_tool_select.yaml"].get("suite") or {}, sort_keys=True).encode()).hexdigest()
        if fz.get("generator", {}).get("sha256") != gen_sha:
            fails.append(f"frozen task list: {gen.relative_to(ROOT)} changed since the freeze (sha256 {gen_sha[:12]} vs "
                         f"{str(fz.get('generator', {}).get('sha256'))[:12]}); re-run scripts/freeze_eval_tasks.py only before the tag (A6.16 (l))")
        elif fz.get("suite_sha256") != suite_sha:
            fails.append(f"frozen task list: the suite block of jevgym_tool_select.yaml changed since the freeze (A6.16 (l))")
        else:
            notes.append(f"frozen task list: generator {gen_sha[:12]} at commit {str(fz.get('generator', {}).get('commit'))[:12]} and "
                         f"suite block {suite_sha[:12]} match (A6.16 (l))")
    # 2e. the C8 population file matches what the decision config produces today (A6.16 (l))
    pop = ROOT / "configs" / "eval" / "e1_c8_population.json"
    dcfg = ROOT / "configs" / "decision" / "bfcl_when2call.yaml"
    if pop.exists() and dcfg.exists():
        try:
            import yaml as _yaml
            from jevharness.bench.decision_eval import load_population, population_counts
            dexp = _yaml.safe_load(dcfg.read_text())
            want = _json.loads(pop.read_text())
            have = population_counts(load_population(dexp))
            if want.get("sha256") != have["sha256"] or want.get("by_source") != have["by_source"]:
                fails.append(f"C8 population {pop.name}: the decision config no longer reproduces it (sha256 "
                             f"{have['sha256'][:12]} vs {str(want.get('sha256'))[:12]}); re-run scripts/c8_population.py only before the tag (A6.16 (l))")
            else:
                notes.append(f"C8 population: {have['n_items']} items / {have['n_source_items']} source items reproduced "
                             f"({', '.join(f'{k} {v}' for k, v in have['by_source'].items())}; A6.16 (l))")
        except FileNotFoundError as e:  # the raw data is not on this machine: the file cannot be re-derived here
            notes.append(f"C8 population: not re-derived here ({e})")
    elif dcfg.exists():
        fails.append("configs/eval/e1_c8_population.json is missing (scripts/c8_population.py; A6.16 (l))")
    # 3. calibration arms mirror the evaluation shadow arms (modes, System-2 binding)
    for cal, (ecfg, earm) in CAL_MIRRORS.items():
        ccfg = "jevgym_shadow_cal.yaml" if cal.startswith("jev") else "jevgym_shadow_cal_intern.yaml"
        c, e = arm_spec(exps[ccfg], cal), arm_spec(exps[ecfg], earm)
        if c is None or e is None:
            fails.append(f"calibration mirror: `{cal}` or `{earm}` missing")
            continue
        for key in ("tool_select", "tool_params", "stop", "llm", "s1"):
            if c.get(key) != e.get(key):
                fails.append(f"`{cal}`.{key} = {c.get(key)} differs from `{earm}`.{key} = {e.get(key)}")
        for key in ("llm", "s1"):
            cv, ev = exps[ccfg].get(key), exps[ecfg].get(key)
            if cv != ev:
                fails.append(f"{ccfg}: top-level {key} {cv} differs from {ecfg}: {ev}")
        for name in ("llms", "s1_backends"):
            for k in (c.get("llm"), c.get("s1")):
                if k and (exps[ccfg].get(name) or {}).get(k) != (exps[ecfg].get(name) or {}).get(k):
                    fails.append(f"{ccfg}: {name}.{k} differs from {ecfg}")
    # 3b. the C1 arms carry the fitted thresholds
    for cfg, arms in C1_TAUS.items():
        for arm, taus in arms.items():
            spec = arm_spec(exps[cfg], arm) or {}
            for dp, tau in taus.items():
                got = (spec.get(dp) or {}).get("tau")
                if got is None or abs(float(got) - tau) > 1e-9 or (spec.get(dp) or {}).get("mode") != "cascade":
                    fails.append(f"{cfg}: `{arm}`.{dp} = {spec.get(dp)}, registered cascade at tau {tau}")
                else:
                    notes.append(f"{cfg}: `{arm}`.{dp} tau {tau} ok")
                want = dp in ALWAYS_DEFER.get(cfg, {}).get(arm, [])
                if bool((spec.get(dp) or {}).get("always_defer")) != want:
                    fails.append(f"{cfg}: `{arm}`.{dp} always_defer should be {want} (A6.11 (c))")
    # 3c. unfitted C1 thresholds are marked pending (A6.16), never a bare number
    for cfg, arms in PENDING_TAUS.items():
        for arm, dps in arms.items():
            spec = arm_spec(exps[cfg], arm) or {}
            for dp in dps:
                if (spec.get(dp) or {}).get("calibration") != "pending":
                    fails.append(f"{cfg}: `{arm}`.{dp} has no fitted tau-hat* and is not marked `calibration: pending` (A6.16)")
                else:
                    notes.append(f"{cfg}: `{arm}`.{dp} pending calibration (marked)")
    # 4. every arm name the registration mentions resolves to an arm of some config
    text = (ROOT / "docs" / "PREREGISTRATION.md").read_text()
    known = {arm for c in counts.values() for arm in c}
    mentioned = sorted({m for m in re.findall(r"`([a-z][a-z0-9_.]*)`", text)
                        if "_" in m and ARM_PREFIX.match(m) and not NOT_ARMS.match(m)})
    for m in mentioned:
        if m not in known:
            fails.append(f"PREREGISTRATION mentions `{m}`, which is no arm of any experiment config")
    notes.append(f"{len(mentioned)} arm names mentioned in PREREGISTRATION.md all resolve" if not any(
        "no arm of any" in f for f in fails) else "arm-name resolution: see failures")
    # 5. freeze placeholders
    m = re.search(r"Frozen 2026-10-03 at commit `([^`]+)`", text)
    if not m:
        fails.append("PREREGISTRATION.md header lacks the line \"Frozen 2026-10-03 at commit `...`\" (A6.9 #12)")
    elif m.group(1) == "FREEZE_COMMIT":
        notes.append("FREEZE_COMMIT placeholder still present in the header (expected before the freeze tag)")
    elif not re.fullmatch(r"[0-9a-f]{7,40}", m.group(1)):
        fails.append(f"PREREGISTRATION.md header: `{m.group(1)}` is not a commit hash")
    else:
        notes.append(f"frozen at commit {m.group(1)}")
    if not a.quiet:
        for n in notes:
            print("ok   ", n)
    for f in fails:
        print("FAIL ", f)
    print(f"prereg_check: {len(fails)} failure(s), {len(notes)} check(s) ok")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
