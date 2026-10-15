"""The effective Stage-2 specification (scripts/stage2_spec.py -> docs/STAGE2_EFFECTIVE.md, PREREG_A7_STAGE2 A7.17 (c))
renders, names every test of both Holm tables and every arm of the Stage-2 configs, and the committed copy is what the
configs and the registration produce today (header line aside); the launch gate passes only when every check does."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage2_spec as SS  # noqa: E402
from jevharness.bench.runner import expand_arms  # noqa: E402


def test_specification_names_every_test_and_arm_and_matches_the_committed_copy():
    text = SS.render()
    for label in SS.S2.FAMILY + SS.S2.SECONDARY:
        assert f"**{label}**" in text, label
    for key, cfg in SS.CONFIGS.items():
        for arm in expand_arms(yaml.safe_load((ROOT / cfg).read_text())):
            assert f"`{arm.name}`" in text, f"{key}: {arm.name}"
    assert "PREREG_A7_STAGE2.md wins" in text and "## 6. Prerequisites" in text
    assert SS.OUT.exists(), "run python scripts/stage2_spec.py and commit docs/STAGE2_EFFECTIVE.md"
    assert SS.strip_generated(SS.OUT.read_text()) == SS.strip_generated(text), (
        "docs/STAGE2_EFFECTIVE.md differs from what the configs / registration render now: "
        "run python scripts/stage2_spec.py and commit the result")


def test_launch_gate_passes_only_when_every_check_passes(monkeypatch, capsys, tmp_path: Path):
    ok = [("always", lambda: (True, "fine"))]
    monkeypatch.setitem(SS.GATES, "stage2_jevgym", ok)
    monkeypatch.setitem(SS.GATES, "stage2_jevgym --rl", [("rl", lambda: (False, "no freeze record"))])
    spec = tmp_path / "STAGE2_EFFECTIVE.md"
    monkeypatch.setattr(SS, "OUT", spec)
    assert SS.gate("stage2_jevgym", rl=False, skip_tests=True) == 1                 # no specification yet
    spec.write_text(SS.render())
    assert SS.gate("stage2_jevgym", rl=False, skip_tests=True) == 0
    assert SS.gate("stage2_jevgym", rl=True, skip_tests=True) == 1
    assert "[NO] rl: no freeze record" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        SS.gate("nope", rl=False, skip_tests=True)
    # the freeze-record check is the analysis' own gate (one rule for launch and analysis)
    rec = tmp_path / "freeze.json"
    monkeypatch.setattr(SS.S2, "FREEZE_RECORD", rec)
    assert SS._freeze_record()[0] is False
    rec.write_text(json.dumps({"frozen_at": "2026-10-04T09:00:00Z"}))
    assert SS._freeze_record()[0] is True
    rec.write_text(json.dumps({"frozen_at": "2026-10-08T09:00:00Z"}))
    assert SS._freeze_record()[0] is False
    # the X cost gate must match the config's n_tasks (A7.12 (c))
    g = tmp_path / "x_cost_gate.json"
    monkeypatch.setattr(SS, "X_COST_GATE", g)
    assert SS._x_cost_gate()[0] is False
    n = yaml.safe_load((ROOT / SS.CONFIGS["stage2_x"]).read_text())["n_tasks"]
    g.write_text(json.dumps({"n": n, "predicted_usd": 20.0}))
    assert SS._x_cost_gate()[0] is True
    g.write_text(json.dumps({"n": 300 if n == 500 else 500}))
    assert SS._x_cost_gate()[0] is False


def test_v2_gate_checks(monkeypatch, tmp_path: Path):
    """A7.19 (e), (f): the v2 freeze record by the deadline (the analysis' own rule) and the v2 thresholds: a
    backends.jevagentrl_v2 block that exists, is not pending, was filled from v2_shadow_cal and names committed fits."""
    import hashlib

    rec = tmp_path / "FROZEN.txt"
    monkeypatch.setattr(SS.S2, "V2_FREEZE_RECORD", rec)
    assert SS._v2_freeze()[0] is False
    rec.write_text("name JevAgentRL-v2\nfrozen_at 2026-10-04T00:53:20Z\n" + "a" * 64 + "  ./model.safetensors\n")
    assert SS._v2_freeze()[0] is True
    rec.write_text("frozen_at 2026-10-07T12:00:01Z\n")
    assert SS._v2_freeze()[0] is False and "deadline" in SS._v2_freeze()[1]
    monkeypatch.setattr(SS, "ROOT", tmp_path)
    (tmp_path / "configs/harness").mkdir(parents=True)
    (tmp_path / "results/calibration").mkdir(parents=True)
    fit = tmp_path / "results/calibration/tauhat_v2_shadow_cal_tool_select.json"
    fit.write_text('{"tau_hat": 0.9}')
    sha = hashlib.sha256(fit.read_bytes()).hexdigest()

    def prof(block):
        sec = {"decision_points": {"tool_select": {"mode": "cascade", "tau": 1.0}}}
        if block is not None:
            sec["backends"] = {"jevagentrl_v2": block}
        (tmp_path / "configs/harness/jevagent_best.yaml").write_text(yaml.safe_dump(
            {"kind": "jevagent_profile", "version": 1, "status": "selected", "system2": {"gemini-3.5-flash": sec}}))
        return SS._v2_fitted()

    ff = {"tool_select": {"path": "/elsewhere/results/calibration/" + fit.name, "sha256": sha}, "fit_arm": "v2_shadow_cal"}
    assert prof(None)[0] is False and "no backends.jevagentrl_v2" in prof(None)[1]
    assert prof({"tool_select": {"calibration": "pending"}})[0] is False
    assert prof({"tool_select": {"tau": 0.9}, "fitted_from": {**ff, "fit_arm": "sl4b_shadow_cal"}})[0] is False
    assert prof({"tool_select": {"tau": 0.9}, "fitted_from": ff}) == (True, "backends.jevagentrl_v2 fitted from v2_shadow_cal")
    ok, msg = prof({"tool_select": {"tau": 0.99, "always_defer": True}, "skip": "not tested (delegates nothing)",
                    "fitted_from": ff})
    assert ok and "delegates nothing" in msg
    fit.write_text('{"tau_hat": 0.8}')                                     # the committed fit changed after the fill
    assert prof({"tool_select": {"tau": 0.9}, "fitted_from": ff})[0] is False


def test_v2_runs_are_in_the_specification_and_gated():
    for run in ("jevgym_shadow_cal_v2", "stage2_v2_states", "stage2_v2"):
        assert run in SS.CONFIGS and run in SS.GATES
    names = [c[0] for c in SS.GATES["stage2_v2"]]
    assert any("thresholds fitted" in n for n in names) and any("tariff" in n for n in names)
    assert any("freeze record" in n for n in names) and any("task-major" in n for n in names)
    ok, msg = SS._v2_fitted()                          # the committed profile: pending until v2's own fit is filled in
    assert ok or "pending" in msg, msg
