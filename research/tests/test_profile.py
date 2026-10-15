"""JevAgentHarness profiles (harness/profile.py; docs/JEVAGENT_HARNESS.md; PREREG_A7_STAGE2 A7.1): loading and
validation, resolution of profile arms in the runner (decision points from the profile's System-2 section, pending
profiles and `skip` sections never run), provenance in every episode, the L3 renderings, and no change to arms that
name no profile."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from jevharness.bench.runner import arm_profiles, expand_arms, run_experiment
from jevharness.harness import profile as HP

ROOT = Path(__file__).resolve().parents[1]
SUITE = {"name": "jevgym", "n": 2, "seed0": 0, "families": "v2", "distractor_style": "near", "n_distractors": 60,
         "noise_lines": 20, "strict": True}
SIM_LLM = {"provider": "simulated", "model": "gemini-3.5-flash", "error_rate": 0.0, "simulated_latency_ms": 1}
SIM_S1 = {"type": "simulated", "accuracy": 0.95, "latency_ms": 1}


def _profile(tmp: Path, status: str = "development", **sections) -> Path:
    p = tmp / f"prof_{status}.yaml"
    p.write_text(yaml.safe_dump({"kind": "jevagent_profile", "version": 1, "name": "t", "status": status,
                                 "system2": sections}))
    return p


def test_shipped_profiles_load_and_validate():
    d = HP.load_profile(ROOT / "configs/harness/jevagent_default.yaml")
    assert d["status"] == "development"
    dps, harness, skip = HP.decision_points(d, "gemini-3.5-flash")
    assert skip is None and dps == {"tool_select": {"mode": "cascade", "tau": 1.0},
                                    "tool_params": {"mode": "cascade", "tau": 0.99}}
    assert HP.decision_points(d, "gemini-3.5-flash-lite")[:1] == ({},)   # Flash-Lite: every decision with the LLM
    assert HP.decision_points(d, "gemini-3.5-flash", env="alfworld")[2]          # nothing for the action loop
    b = HP.load_profile(ROOT / "configs/harness/jevagent_best.yaml")
    if b["status"] == "pending":       # before the A7.1 selection: the runner skips every arm that names it
        assert "pending" in HP.decision_points(b, "gemini-3.5-flash")[2]
    else:                              # written once by scripts/select_profile.py (A7.1 (h)); never a dry run
        assert b["status"] == "selected" and not b.get("marks")
        assert b["selection"]["rule"] == "docs/PREREG_A7_STAGE2.md A7.1" and not b["selection"].get("errors")


def test_profile_validation_errors(tmp_path: Path):
    with pytest.raises(HP.ProfileError):
        HP.load_profile(_profile(tmp_path, **{"m": {"decision_points": {"bogus": {"mode": "cascade"}}}}))
    with pytest.raises(HP.ProfileError):   # a mode the decision point does not support (harness/config.py; the
        # routing cascade / assist exist since E6b / E6d, the tool-choice trigger since E6f; an LLM gate at tool choice
        # does not exist)
        HP.load_profile(_profile(tmp_path, **{"m": {"decision_points": {"tool_select": {"mode": "llm"}}}}))
    p = _profile(tmp_path, **{"gemini-3.5-flash": {"decision_points": {"stop": {"mode": "guard", "tau": 0.5}}}})
    exp = {"llm": SIM_LLM, "arms": []}
    with pytest.raises(HP.ProfileError, match="also sets"):
        HP.resolve_arm({"name": "x", "profile": str(p), "stop": {"mode": "off"}}, exp)
    with pytest.raises(HP.ProfileError, match="without `profile`"):
        HP.resolve_arm({"name": "x", "profile_s2": "gemini-3.5-flash"}, exp)
    with pytest.raises(HP.ProfileError, match="no section"):
        HP.resolve_arm({"name": "x", "profile": str(p), "profile_s2": "other"}, exp)


def test_runner_resolves_profile_arms_and_skips_pending_ones(tmp_path: Path):
    p = _profile(tmp_path, **{"gemini-3.5-flash": {
        "decision_points": {"tool_select": {"mode": "cascade", "tau": 0.5}, "tool_params": {"mode": "cascade", "tau": 0.5},
                            "stop": {"mode": "trigger", "tau": 0.9}},
        "harness": {"guard_max_retries": 2},
        "best_single": {"arm": "x", "decision_points": {"tool_select": {"mode": "cascade", "tau": 0.5}},
                        "skip": "composition not tested"}}})
    pend = _profile(tmp_path, status="pending")
    exp = {"name": "prof", "suite": SUITE, "llm": SIM_LLM, "s1": SIM_S1, "workers": 1,
           "arms": [{"name": "llm_only"},
                    {"name": "composed", "profile": str(p)},                    # profile_s2 = the arm's LLM model
                    {"name": "single", "profile": str(p), "profile_part": "best_single"},
                    {"name": "pending", "profile": str(pend), "profile_s2": "gemini-3.5-flash"}]}
    cfgs = {c.name: c for c in expand_arms(exp)}
    assert cfgs["composed"].tool_select.mode.value == "cascade" and cfgs["composed"].stop.tau == 0.9
    assert cfgs["composed"].guard_max_retries == 2 and cfgs["llm_only"].short() == "llm_only"
    prof = arm_profiles(exp)
    assert prof["single"]["skip"] == "composition not tested" and "pending" in prof["pending"]["skip"]
    paths = run_experiment(exp, tmp_path / "out", progress=False)
    assert set(paths) == {"llm_only", "composed"}                               # skipped arms never run
    recs = [json.loads(x) for x in (tmp_path / "out/prof/composed.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and all(r["meta"]["profile"]["sha256"] == prof["composed"]["sha256"] for r in recs)
    dps = {d["dp"] + "/" + d["mode"] for r in recs for s in r["steps"] for d in s["decisions"]}
    assert "tool_select/cascade" in dps and "stop/trigger" in dps
    assert json.loads((tmp_path / "out/prof/profiles.json").read_text())["composed"]["s2"] == "gemini-3.5-flash"


def test_registered_configs_resolve_unchanged():
    """No registered (Stage-1) config names a profile: expand_arms gives the same configs as before the mechanism."""
    from jevharness.harness.config import HarnessConfig

    for cfg in ("jevgym_tool_select.yaml", "jevgym_meta_control.yaml", "baselines_jevgym.yaml",
                "jevgym_tool_select_intern.yaml"):
        exp = yaml.safe_load((ROOT / "configs/experiments" / cfg).read_text())
        assert not arm_profiles(exp)
        for a, c in zip([a for a in exp["arms"]], expand_arms(exp)):
            raw = {k: v for k, v in a.items() if k not in ("s1", "llm", "n_tasks")}
            assert c == HarnessConfig(**{**exp.get("harness_defaults", {}), **raw}).validate_modes()


def test_stage2_configs_resolve():
    exp = yaml.safe_load((ROOT / "configs/experiments/stage2_jevgym.yaml").read_text())
    prof = arm_profiles(exp)
    logo = {f"jevagent_minus_{g}" for g in ("tool", "stop", "reflection", "routing", "memory")}   # A7.11 (c)
    assert set(prof) == {"jevagent_best", "best_single", "jevagent_best_rl", "jevagent_default",
                         "jevagent_best_s2lite", "jevagent_best_sl4b"} | logo                  # sl4b: S2-4S (A7.18)
    alf = arm_profiles(yaml.safe_load((ROOT / "configs/experiments/stage2_alfworld.yaml").read_text()))
    if HP.load_profile(ROOT / "configs/harness/jevagent_best.yaml")["status"] == "pending":
        assert all("pending" in prof[a]["skip"] for a in prof if a != "jevagent_default")   # until select_profile.py ran
        assert "pending" in alf["jevagent_best"]["skip"]
    else:   # selected (A7.1 (h)): every arm resolves against the written profile; skips are the profile's own
        assert all(prof[a]["status"] == "selected" for a in prof if a != "jevagent_default")
        assert all("pending" not in (prof[a]["skip"] or "") for a in prof)
        assert alf["jevagent_best"]["status"] == "selected"
    assert prof["jevagent_default"]["skip"] is None
    assert prof["jevagent_best_s2lite"]["s2"] == "gemini-3.5-flash-lite"


def test_tau2_rendering_translation():
    dps = {"tool_select": {"mode": "assist", "k": 3}, "stop": {"mode": "guard", "tau": 0.5},
           "routing": {"mode": "trigger", "tau": 0.4}, "reflection": {"mode": "trigger", "tau": 0.5}}
    t2, notes = HP.tau2_rendering(dps)
    assert t2["tool_select"] == {"mode": "assist", "k": 3} and t2["stop"]["mode"] == "guard"
    assert t2["routing"]["tau"] == 0.4 and t2["small_llm"] == HP.TAU2_SMALL_LLM
    assert any("reflection" in n for n in notes)
    from jevharness.bench.suites.tau2 import Tau2JevConfig
    Tau2JevConfig.from_dict(t2)                                   # a valid JevTau2Agent config
    assert HP.tau2_rendering({"reflection": {"mode": "trigger", "tau": 0.5}})[0] is None
    assert HP.alfworld_rendering(dps)["decision_points"].keys() == {"routing", "reflection"}


def test_real_harness_renderings():
    d = HP.load_profile(ROOT / "configs/harness/jevagent_default.yaml")
    dps = HP.decision_points(d, "gemini-3.5-flash")[0]
    ext, env, notes = HP.pi_rendering(dps)
    assert ext == [] and env == {}                                    # the default is a tool cascade only:
    assert any("not deployable on pi" in n for n in notes)            # it needs the reference loop
    gated = {**dps, "stop": {"mode": "trigger", "tau": 0.9}, "routing": {"mode": "trigger", "tau": 0.4}}
    ext, env, _ = HP.pi_rendering(gated)
    assert ext == ["jev-stop-verify.ts", "jev-router.ts"] and env["JEV_STOP_MODE"] == "trigger"
    assert env["JEV_STOP_TAU"] == "0.9" and env["JEV_ROUTER_TAU"] == "0.4"
    for e in ext:
        assert (ROOT / "adapters/pi" / e).exists()
    arm, notes = HP.mini_swe_rendering({**gated, "reflection": {"mode": "trigger", "tau": 0.5},
                                        "tool_select": {"mode": "guard", "tau": 0.4}})
    arm.pop("routing")
    assert arm == {"action_guard": {"mode": "guard", "tau": 0.4}, "stop": {"mode": "trigger", "tau": 0.9},
                   "reflection": {"mode": "trigger", "tau": 0.5}}
    try:
        from jevharness.adapters.mini_swe import JevMiniSweConfig
    except Exception:  # noqa: BLE001 - mini-swe-agent not installed
        return
    JevMiniSweConfig(**arm).validate_modes()                          # a valid adapter arm
