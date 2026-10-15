"""E16 (exploratory; docs/PREREG_A7_STAGE2.md Deviations 2026-10-08 "E16: the best open-weights decision model on all
three levels"): E12's builders, audit and development rule are reused unchanged; the quotas and the 1 : 1 replay; the
configs match E1 and the frozen live row; the smoke seeds are development seeds; the registered gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e12_train as E12  # noqa: E402
import explore_e16_best_open as E16  # noqa: E402
import explore_e9_frontier as E9  # noqa: E402

CFG = ROOT / "configs/explore/e16_best_open"


def test_quotas_ratio_and_e12_reuse():
    assert sum(E16.QUOTA_TRAIN.values()) == E16.N_REPLAY == 14000       # new : replay = 1 : 1, twice E12's 7,000
    assert set(E16.QUOTA_TRAIN) == set(E12.QUOTA_TRAIN)
    for fam in ("w2c_call", "w2c_ask", "w2c_cannot", "bfcl_tool", "bfcl_none", "bfcl_args"):
        assert E16.QUOTA_TRAIN[fam] >= E12.QUOTA_TRAIN[fam], fam           # never fewer items of a family than E12
    assert E16.select16.__doc__ and E16.E12_FILES[0] == "public_dev.jsonl.gz"


def test_build_record_if_present():
    rec = E16.OUT / "build_info.json"
    if not rec.exists():
        return
    d = json.loads(rec.read_text())
    assert d["rows"]["replay"] == d["rows"]["public_train"]              # 1 : 1 as built
    assert d["rows"]["train_mix"] == 2 * d["rows"]["public_train"]
    assert d["audit"]["units_with_hit"] == 207 and d["audit"]["same_hits_as_e12"]   # E12's audit, unchanged
    e12 = json.loads((ROOT / "results/explore/e12/build_info.json").read_text())
    for f, sha in d["e12_files"].items():
        assert e12["sha256"][f] == sha == d["sha256"][f]                   # E12's dev split and temperature set
    assert all(d["train_by_family"][f] <= q for f, q in d["quota_train"].items())
    assert "bfcl_args" in d["train_by_family"] and d["replay"]["rows"] == d["rows"]["replay"]


def test_configs_match_e1_and_the_frozen_live_row():
    e1 = yaml.safe_load((ROOT / "configs/decision/bfcl_when2call_rl.yaml").read_text())
    e16 = yaml.safe_load((CFG / "e1_e16.yaml").read_text())
    for k in ("suites", "exclude_ids_file", "batch_by_state", "workers"):
        assert e16[k] == e1[k]
    assert e16["s1_cache"] not in (e1["s1_cache"], "cache/s1_stage2.sqlite", "cache/s1_e12_e1.sqlite")
    assert e16["backends"]["e16a"]["name"] == E16.ARMS["a"]["served"]
    assert e16["backends"]["e16b"]["name"] == E16.ARMS["b"]["served"]
    assert e16["backends"]["frozen_rehost"]["name"] == "jevagentrl-4b"
    ref = yaml.safe_load((ROOT / "configs/explore/e12_train/live_e12c.yaml").read_text())
    for arm in ("a", "b"):
        live = yaml.safe_load((CFG / f"live_e16{arm}.yaml").read_text())
        assert live["suite"] == ref["suite"] and live["n_tasks"] == 200 and live["llm"] == ref["llm"]
        assert live["arms"][0]["tool_select"] == {"mode": "replace"} and live["arms"][0]["name"] == E16.ARMS[arm]["arm"]
        assert live["name"] == E16.ARMS[arm]["live_run"] and live["s1_cache"] != ref["s1_cache"]
        smoke = yaml.safe_load((CFG / f"live_e16{arm}_smoke.yaml").read_text())
        assert smoke["name"] == E16.ARMS[arm]["smoke_run"] and smoke["n_tasks"] == 2
        assert 900 <= smoke["suite"]["seed0"] <= 998                        # development seeds, never reported
        assert smoke["suite"]["seed0"] not in (900, 902, 906)               # not E12's, E12c's or E14 / E15's smoke seeds
        assert smoke["llm"] == live["llm"] and smoke["arms"][0]["name"] == live["arms"][0]["name"]
    assert E16.RUNS["e16"] == ("e16a_live_smoke", "e16a_live", "e16b_live_smoke", "e16b_live")


def test_e9_hook_keeps_e12_keys_and_adds_e16():
    assert E9.FRONT12 == ("e12", "e12c") and set(E9.FRONT16) == {"e12", "e12c", "e16a", "e16b"}
    for k in ("e16a", "e16b"):
        label, rec, rehost = E9.EXTRA[k]
        assert rec.startswith("e1_bfcl_when2call_e16/") and rehost == "e1_bfcl_when2call_e16/s1_frozen_rehost.jsonl"


def test_gates():
    assert E16.CAP_USD == 8.0 and E16.PROJ_FACTOR == 1.2 and E16.C_MARGIN_PT == 1.0 and E16.LIVE_N == 200
