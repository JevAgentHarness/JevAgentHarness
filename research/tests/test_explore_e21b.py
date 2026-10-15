"""Exploratory row E21b (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E21b: the deployed confidence-gated cascades
on DecisionGym-Shift"): the configurations, profiles and thresholds as registered, the driver's syntax, and the readout's
counting on synthetic records. Offline; no model or API call."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e21b_shift_cascade as X  # noqa: E402

from jevharness.harness.profile import load_profile, resolve_arm  # noqa: E402

CFG = ROOT / "configs/explore/e21b_shift_cascade"
REGISTERED = {"rl": ("configs/harness/jevagent_final.yaml", "jevagent_rl", 0.4999999999978404, 0.9999998807907104),
              "e18": ("configs/harness/explore_e18_final.yaml", "e18", 0.9941678, 0.9965348931475904)}
SMOKE_SEEDS = {"rl": 339020, "e18": 339022, "e12": 339024, "e12_in": 918}


def _cfg(name: str) -> dict:
    return yaml.safe_load((CFG / name).read_text())


@pytest.mark.parametrize("slug", ["rl", "e18", "e12", "e12_in"])
def test_configs_as_registered(slug):
    live, smoke = _cfg(f"live_{slug}.yaml"), _cfg(f"smoke_{slug}.yaml")
    for c in (live, smoke):
        assert c["workers"] == 1 and c["schedule"] == {"mode": "task_major", "seed": 732}
        assert c["llm"] == {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0,
                            "reasoning_effort": "minimal"}
        assert "s1_cache" not in c
        (be, spec), = c["s1_backends"].items()
        assert spec["cache"] is None and spec["type"] == "jev"
        arm, = c["arms"]
        assert arm["name"] == X.ARMS[slug][1] and arm["s1"] == be and arm["profile_backend"] == be
        assert c["suite"]["families"] == "v2" and c["suite"]["n_distractors"] == 60 and c["suite"]["strict"] is True
    assert live["name"] == X.ARMS[slug][0]
    assert smoke["suite"]["seed0"] == SMOKE_SEEDS[slug] and smoke["suite"]["n"] == 2
    if slug == "e12_in":
        assert "shift" not in live["suite"] and live["suite"]["seed0"] == 300500 and live["suite"]["n"] == 200
    else:
        assert live["suite"]["shift"] is True and live["suite"]["seed0"] == 330100 and live["suite"]["n"] == 200
        assert smoke["suite"]["shift"] is True and 339000 <= smoke["suite"]["seed0"] <= 339098
        # the same tasks as E21's live rows (suite block of configs/explore/e21_shift/live_rl.yaml)
        e21 = yaml.safe_load((ROOT / "configs/explore/e21_shift/live_rl.yaml").read_text())["suite"]
        assert live["suite"] == e21


@pytest.mark.parametrize("slug", ["rl", "e18"])
def test_in_domain_thresholds_unchanged(slug):
    path, be, t_tool, t_arg = REGISTERED[slug]
    c = _cfg(f"live_{slug}.yaml")
    assert c["arms"][0]["profile"] == path
    out, meta = resolve_arm(c["arms"][0], c)
    assert meta["skip"] is None
    assert out["tool_select"]["mode"] == "cascade" and out["tool_select"]["tau"] == t_tool
    assert out["tool_params"]["mode"] == "cascade" and out["tool_params"]["tau"] == t_arg
    assert out["tool_params"]["batch_s1"] is True
    assert "reflection" not in out


def test_e12_profile_pending_or_fitted():
    prof = load_profile(X.PROFILE_E12)
    base = load_profile(X.FINAL_PROFILE)
    sec, bsec = prof["system2"]["gemini-3.5-flash"], base["system2"]["gemini-3.5-flash"]
    assert sec["decision_points"] == bsec["decision_points"]
    assert sec["backends"]["jevagent_rl"] == bsec["backends"]["jevagent_rl"]
    c = _cfg("live_e12.yaml")
    out, meta = resolve_arm(c["arms"][0], c)
    if prof["status"] == "pending":
        assert meta["skip"]          # the runner skips the arm until the fit is committed
        return
    blk = sec["backends"]["e12"]
    fits = json.loads((X.CAL_OUT / "fits.json").read_text())
    for dp in X.DPS:
        f = blk["fitted_from"][dp]
        assert X.sha256_file(ROOT / f["path"]) == f["sha256"]
        t = fits["fits"][dp]["tau_hat"]
        if blk[dp].get("always_defer"):
            assert t is None or t > 1.0
        elif "fallback" not in blk["notes"]:
            assert blk[dp]["tau"] == pytest.approx(float(t))
    assert meta["skip"] is None and out["tool_select"]["mode"] == "cascade"


def test_driver_syntax_and_guards():
    p = ROOT / "scripts/aws/e21b_shift_cascade.sh"
    subprocess.run(["bash", "-n", str(p)], check=True)
    txt = p.read_text()
    for iid in ("<instance-id>", "<instance-id>", "<instance-id>", "<instance-id>"):
        assert iid in txt.split("FORBIDDEN_IIDS=")[1].split("\n")[0]
    assert "GUARD_MIN=${GUARD_MIN:-230}" in txt and "CAP_USD=${CAP_USD:-35}" in txt
    assert X.CAL_REQ_SHA in txt


def _dec(dp, s1, deferred, at=None, gold=None, final=None, mode="cascade"):
    d = {"dp": dp, "mode": mode, "s1_value": s1, "s1_conf": 0.9, "deferred": deferred, "shadow": False,
         "final_value": final if final is not None else s1, "gold": gold, "gold_status": "ok"}
    if at is not None:
        d["gold_aux"] = {"v": 1, "acceptable_tools": at}
    return d


def test_decision_stats_counts_accepted_wrong_and_must_ask():
    rec = {"steps": [{"decisions": [
        _dec("tool_select", "lookup_customer", False, ["lookup_customer"], "lookup_customer"),
        _dec("tool_select", "cancel_purchase", False, ["request_clarification"], "request_clarification"),  # must-ask, wrong
        _dec("tool_select", "x", True, ["request_clarification"], "request_clarification", final="request_clarification"),
        _dec("tool_params", "Alice", False, gold="Alice Smith"),
        _dec("tool_params", "7", False, gold="7"),
        _dec("tool_params", None, True, gold="free text", final="free text", mode="generate"),
    ]}]}
    s = X.decision_stats([rec])
    ts, tp = s["tool_select"], s["tool_params"]
    assert (ts["decisions"], ts["taken"], ts["deferred"], ts["accepted_wrong"]) == (3, 2, 1, 1)
    assert ts["mustask"] == {"decisions": 2, "taken": 1, "accepted_wrong": 1}
    assert ts["deferred_llm_acc"] == 1.0
    assert (tp["decisions"], tp["taken"], tp["accepted_wrong"]) == (2, 2, 1)


def _write_run(d: Path, arm: str, costs: list[float], errors: list[str | None] | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    errors = errors or [None] * len(costs)
    with open(d / f"{arm}.jsonl", "w") as f:
        for i, (c, e) in enumerate(zip(costs, errors)):
            f.write(json.dumps({"task_id": f"t{i}", "error": e, "usage": {"s2": {"cost_usd": c}, "s1": {"cost_usd": 0.0},
                                                                          "total": {"cost_usd": c}}}) + "\n")


def test_spend_gate_and_errored(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs / "e21b_smoke_rl", "rl_final", [0.02, 0.04])
    _write_run(runs / "e21b_rl", "rl_final", [1.0, 2.0], [None, "HTTP 503 overloaded"])
    _write_run(runs / "e21_live_local", "llm_only", [100.0])        # another row's run: not E21b spend
    log = tmp_path / "probe.txt"
    log.write_text("x 6/6 ok  CLEAN  (logged $0.00100)\n")
    s = X.total_spend(runs, log)
    assert s["gemini"] == pytest.approx(3.06 + 0.001) and set(s["by_run"]) == {"e21b_smoke_rl", "e21b_rl"}
    g = X.gate(runs / "e21b_smoke_rl", runs, log, n=200, cap=35.0)
    assert g["gemini_usd_per_episode"] == pytest.approx(0.03)
    assert g["n_fit"] == 200 and g["go"]
    g2 = X.gate(runs / "e21b_smoke_rl", runs, log, n=200, cap=8.0)
    assert g2["n_fit"] == int((8.0 - 3.061) / (1.2 * 0.03)) and not g2["go"]
    assert X.errored_tasks(runs / "e21b_rl") == 1


def test_kind_parse_and_ni_block():
    assert X.kind_of("jevgym-shift-sdt-clarify.ask_after_lookup-330109-d120-near-strict") == "clarify.ask_after_lookup"
    assert X.kind_of("jevgym-shift-sdt-multi-330103-d120-near-strict") == "multi"
    b = X.ni_block(0, 0, 200)
    assert b["verdict"] == "non-inferior" and b["p"] <= 0.05
    assert X.ni_block(0, 10, 200)["verdict"] == "not shown"


def test_calibration_questions_are_the_registered_file():
    p = ROOT / "runs/e5_small_inputs/cal_requests.jsonl.gz"
    if not p.exists():
        pytest.skip("the logged calibration questions are not in this checkout")
    assert X.sha256_file(p) == X.CAL_REQ_SHA
    with gzip.open(p, "rt") as f:
        assert sum(1 for _ in f) == 2719
