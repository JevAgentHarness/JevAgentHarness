"""E7d (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06 "E7d: re-run the recipe on the new rule"): the $0 harvest and
the row builder of scripts/explore_e7d_tier.py, the training guards it relies on, and its key writer (offline)."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

from jevharness.bench import registry as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explore_e7d_tier as X  # noqa: E402


def test_guards_exclude_the_hard_blocks_and_allow_the_e7d_training_seeds():
    from jevharness.training import data_v1, seed_guard

    for lo, hi in ((310000, 310999), (321000, 321999)):
        assert (lo, hi) in R.training_exclusions() and (lo, hi) in seed_guard.TRAIN_FORBIDDEN
    data_v1.check_seed_plan(*X.TRAIN_SEEDS)
    with pytest.raises(ValueError):
        data_v1.check_seed_plan(X.HELDOUT_SEEDS[0], 10)
    with pytest.raises(ValueError):
        data_v1.check_seed_plan(310400, 10)
    assert not R.overlaps(X.TRAIN_SEEDS[0], X.TRAIN_SEEDS[1], R.training_exclusions() + (R.CALIBRATION_RANGE,))


def test_stratum_reads_the_tool_name_of_the_gold_option():
    row = lambda gold, dp="tool_select": {"dp": dp, "target": {gold: 1.0, "x -- y": 0.0}}  # noqa: E731
    assert X.stratum(row("cancel_order -- Cancel an order.")) == "op"
    assert X.stratum(row("ship_order -- Ship a pending order.")) == "op"
    assert X.stratum(row("find_user -- Look up a customer.")) == "ts_other"
    assert X.stratum(row("U123", "tool_params")) == "tp" and X.stratum(row("yes", "stop")) is None


def test_harvest_and_build_keep_heldout_apart(tmp_path, monkeypatch):
    out = tmp_path / "h"
    X.harvest("train", out, n=3, workers=2)
    X.harvest("heldout", out, n=2, workers=2)
    rows = [json.loads(x) for x in gzip.open(ROOT / "data/train_v1/dev.jsonl.gz", "rt")][:30]
    d100 = tmp_path / "D100.jsonl"
    d100.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(X, "N_TIER", {"op": 5, "ts_other": 5, "tp": 5})
    monkeypatch.setattr(X, "N_REPLAY", 4)
    monkeypatch.setattr(X, "N_HELDOUT", {"op": 3, "ts_other": 3, "tp": 3})
    info = X.build(out, d100, ROOT / "data/train_v1/dev.jsonl.gz", tmp_path / "data")
    assert info["rows"]["train_mix"] == sum(info["tier_strata"].values()) + 4
    assert info["tier_strata"].get("op", 0) > 0
    mix = [json.loads(x) for x in open(tmp_path / "data/train_mix.jsonl")]
    held = [json.loads(x) for f in ("op", "ts_other", "tp") for x in open(tmp_path / f"data/heldout_{f}.jsonl")]
    seed = lambda r: int(str(r["id"]).split("/", 1)[0].split("-")[2])  # noqa: E731
    tier_seeds = {seed(r) for r in mix if (r.get("meta") or {}).get("e7d") == "tier"}
    assert tier_seeds and all(X.TRAIN_SEEDS[0] <= s < X.TRAIN_SEEDS[0] + X.TRAIN_SEEDS[1] for s in tier_seeds)
    assert all(X.HELDOUT_SEEDS[0] <= seed(r) < X.HELDOUT_SEEDS[0] + X.HELDOUT_SEEDS[1] for r in held)
    for r in mix + held:
        assert {"state", "question", "options", "target", "dp"} <= set(r)
    assert info["harvest"]["train"]["cost_usd"] == 0 and info["harvest"]["heldout"]["cost_usd"] == 0


def test_tex_keys_for_both_checkpoints():
    at = {"coverage": 0.5, "s1_acc_all": 0.9, "s1_acc_taken": 1.0}
    cal = {"tool_select": {"tau_hat": 0.95, "always_defer": False, "at_tau": at, "op_at_tau": at},
           "tool_params": {"tau_hat": float("inf"), "always_defer": True, "at_tau": at}}
    d = {"models": {"frozen": {"heldout": {"op": {"n": 10, "acc": 0.8}}, "cal": cal, "c": {"c_acc": 0.992}},
                    "tier": {"heldout": {"op": {"n": 10, "acc": 0.99}}, "cal": cal, "v1": {"v1_acc": 0.99}}},
         "c_delta": {"all": {"delta": -0.002, "lo": -0.006, "hi": 0.001}}}
    k = X.tex_keys(d)
    assert k["e7d/tier/heldout/op/acc"] == "99.0" and k["e7d/frozen/c/acc"] == "99.2"
    assert k["e7d/tier/cal/toolparams/tau"] == "always defer" and k["e7d/frozen/cal/toolselect/tau"] == "0.95"
    assert k["e7d/c/delta"] == "-0.2" and k["e7d/c/delta_ci/lo"] == "-0.6"
    d["label"] = "x"
    d["models"]["frozen"]["name"], d["models"]["tier"]["name"] = X.OLD_NAME, X.NEW_NAME
    md = X.markdown(d)
    assert X.NEW_NAME in md and "| 80.0 | 99.0 |" in md and "-0.2 points" in md


def test_time_bounded_scoring_helpers(tmp_path):
    qs = [{"key": f"t{t}|{s}|0", "task_id": f"t{t}", "dp": "tool_params", "options": ["a", "b"],
           "logged": {"value": "a", "conf": 0.9, "dist": {"a": 0.9, "b": 0.1}}} for t in (1, 2, 3) for s in (0, 1)]
    req = tmp_path / "req.jsonl"
    X._write(qs, req)
    ans = [{"key": q["key"], "dp": q["dp"], "value": "b", "conf": 0.6, "dist": {"a": 0.4, "b": 0.6}, "error": None}
           for q in qs[:5]]                                   # task t3 is cut by the deadline after one question
    (tmp_path / "ans.jsonl").write_text("".join(__import__("json").dumps(a) + "\n" for a in ans) + '{"key": "t3|1')
    r = X.complete(req, tmp_path / "ans.jsonl", tmp_path, min_tasks=2)
    assert r["tasks"] == ["t1", "t2"] and r["enough"] and r["rows_tier"] == 4 and r["rows_frozen"] == 4
    fz = X._read(tmp_path / "answers_frozen_cal.jsonl")
    assert {a["value"] for a in fz} == {"a"} and {a["value"] for a in X._read(tmp_path / "answers_tier_cal.jsonl")} == {"b"}
    h = X.head_tasks(req, 2, tmp_path / "head.jsonl")
    assert h["tasks"] == ["t1", "t2"] and h["questions"] == 4
    data = tmp_path / "data"
    X._write([{"i": i} for i in range(3000)], data / "tier_dev.jsonl")
    X._write([{"i": i} for i in range(500)], data / "heldout_ts_other.jsonl")
    X._write([{"i": i} for i in range(100)], data / "heldout_tp.jsonl")
    X._write([{"v": i} for i in range(2377)], tmp_path / "v1.jsonl")
    info = X.trim_all(data, tmp_path / "v1.jsonl")
    assert info["cal_dev_small"]["rows"] == X.TRIM["cal_v1"] + X.TRIM["cal_tier"]
    assert info["heldout_ts_other"]["rows"] == X.TRIM["heldout_ts_other"] and info["heldout_tp"]["rows"] == 100
    again = X.trim(data / "tier_dev.jsonl", X.TRIM["cal_tier"], tmp_path / "again.jsonl")
    assert X._read(tmp_path / "again.jsonl") == X._read(data / "cal_tier_small.jsonl") and again["rows"] == 1000
