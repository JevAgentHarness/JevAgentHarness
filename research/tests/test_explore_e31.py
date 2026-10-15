"""E31 (exploratory; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-10 "E31: GRPO in the agent loop without SFT"):
offline checks of the registered configuration, the host driver's guards and the readout helpers."""
from __future__ import annotations

import gzip
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
DRIVER = ROOT / "scripts/aws/e31_rl_only.sh"
CFG = ROOT / "configs/explore/e31_rl_only"
_spec = importlib.util.spec_from_file_location("explore_e31", ROOT / "scripts" / "explore_e31.py")
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def test_training_config_differs_from_v1_only_as_registered():
    from jevharness.training.grpo_online import load_config

    e31 = load_config(X.TRAIN_CFG, [], model="Qwen/Qwen3-4B")
    v1 = load_config(X.V1_CFG, [], model="Qwen/Qwen3-4B", init="checkpoints/r5/sft_v1")   # v1's --init (chain decision)
    a, b = _flat(e31.model_dump(mode="json")), _flat(v1.model_dump(mode="json"))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"name", "init", "ref", "out", "grpo.iterations"}
    assert e31.init is None and e31.ref == "base" and e31.grpo.iterations == 15
    assert (e31.lora.r, e31.lora.alpha, e31.lora.target_modules) == (32, 64, "all-linear")
    assert e31.grpo.group_size == 8 and e31.reward.kind == "gated" and e31.reward.lambda_gate == 0.5
    assert e31.reward.cost_scale == 10.0 and e31.grpo.kl_beta == 0.02 and e31.seed == 0
    e31.check_seeds()


def test_live_configs_equal_e21_sft_row_but_backend():
    ref = yaml.safe_load((ROOT / "configs/explore/e21_shift/live_sft_v1.yaml").read_text())
    for f, seed0, n in (("live_shift.yaml", 330100, 200), ("smoke_shift.yaml", 339070, 2)):
        d = yaml.safe_load((CFG / f).read_text())
        assert d["suite"] == {**ref["suite"], "seed0": seed0, "n": n}
        assert d["llm"] == ref["llm"] and d["n_tasks"] == n
        assert d["arms"] == [{"name": "e31_ts_replace_f", "s1": "e31", "tool_select": {"mode": "replace"}}]
        b = d["s1_backends"]["e31"]
        assert b["name"] == b["model"] == X.NAME and b["base_url"] == "http://127.0.0.1:8081"
        assert d["s1_cache"] != ref["s1_cache"]
    e1 = yaml.safe_load((CFG / "e1_e31.yaml").read_text())
    e12 = yaml.safe_load((ROOT / "configs/explore/e12_train/e1_e12.yaml").read_text())
    assert e1["suites"] == e12["suites"] and e1["exclude_ids_file"] == e12["exclude_ids_file"]
    assert e1["name"] == X.E1_RUN and list(e1["backends"]) == ["e31"]


def test_driver_syntax_and_guards():
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    txt = DRIVER.read_text()
    forb = set(re.findall(r"(?m)^(i-0[0-9a-f]{16})$", (CFG / "forbidden_iids.txt").read_text()))
    assert {"<instance-id>", "<instance-id>"} <= forb
    others = set()
    for d in (ROOT / "scripts/aws").glob("*.sh"):
        if d.name != DRIVER.name:
            others |= set(re.findall(r"(?<![A-Za-z0-9])i-0[0-9a-f]{16}", d.read_text()))
    assert others <= forb
    assert not re.findall(r"(?<![A-Za-z0-9])i-0[0-9a-f]{16}", txt)        # the list lives in the config file
    assert "TAG=jev-e31-rlonly" in txt and "GUARD_MIN=${GUARD_MIN:-590}" in txt and "CAP_USD=${CAP_USD:-45}" in txt
    assert "TRAIN_CAP_USD=${TRAIN_CAP_USD:-30}" in txt and "TRAIN_MAX_MIN=${TRAIN_MAX_MIN:-390}" in txt
    assert "types=(g6e.2xlarge g6e.4xlarge g6e.xlarge)" in txt and "<aws-resource-id>" in txt
    assert "OPENROUTER" not in txt and "OPENAI_API" not in txt
    assert "configs/train/grpo_online_e31_rlonly.yaml" in txt and "1cfa9a7208912126459214e8b04321603b3df60c" in txt


def _rows(updated):
    return [{"iter": 0, "eval/rl_ts_params_replace/cost_usd": 0.01, "eval/rl_ts_params_replace/success": 0.5}] + [
        {"iter": i + 1, "episodes": 64, "decisions": 100, "updated_decisions": u, "cost_usd": 0.01, "success": 0.5,
         "frac_zero_std_groups": 0.25, "defer_rate": 0.5, "reward_mean": 0.3, "dec_acc": 0.6}
        for i, u in enumerate(updated)]


def test_degeneracy_rule_and_training_spend():
    assert not X.degenerate(_rows([5, 0, 0]))
    assert X.degenerate(_rows([5, 0, 0, 0]))
    assert not X.degenerate(_rows([0, 0, 5]))
    sp = X.train_spend(_rows([1, 2]))
    assert sp["rollouts"] == pytest.approx(2 * 64 * 0.01) and sp["greedy_evals"] == pytest.approx(48 * 0.01)


def test_training_block_counts_decisions_made(tmp_path, monkeypatch):
    box = tmp_path / "box"
    (box / "train").mkdir(parents=True)
    rows = _rows([10, 20])
    (box / "train" / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    roll = [{"iter": it, "arm": arm, "decisions": 25} for it in (1, 2) for arm in ("rl_ts_params_replace", "rl_cascade",
                                                                                      "rl_ts_assist_k3", "rl_guard_flash")]
    with gzip.open(box / "train" / "rollouts.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in roll))
    (box / "train_run.json").write_text(json.dumps({"minutes": 120.0}))
    t = X.training_block(box)
    assert t["decisions"] == 200 and t["decisions_made"] == pytest.approx(200 - 2 * 0.5 * 25)
    assert t["zero_std_share"] == pytest.approx(0.25) and t["updated_decisions"] == 30 and t["gpu_h"] == 2.0


def test_tex_keys_carry_the_coordinators_names():
    cells = {k: {"l1": 0.7, "c": 0.9, "l2": 0.8, "mustask": 0.5, "l3": 0.6} for k in ("e31", "base", "sft_v1", "rl")}
    tr = {"gpu_h": 2.5, "gpu_min": 150, "spend": {"total": 12.345}, "iterations": 15, "zero_std_share": 0.4,
          "decisions": 12000, "decisions_made": 11000.4, "decisions_made_share": 0.9167, "updated_decisions": 5000,
          "iterations_without_update": 0, "greedy_eval_success": {}, "v1_first15": {}, "train_stop": None}
    r = {"cells": cells, "training": tr, "share_of_sft_gain": {"l2": 0.5}, "l2": {}, "c": {}, "l3": {}, "reading": None}
    k = X.tex_keys(r)
    for key in ("gpu_h", "api_usd", "l1/acc", "l2/all", "l2/mustask", "l3/success", "c/acc", "zero_std_share",
                "decisions_made"):
        assert f"e31/rlonly/{key}" in k
    assert k["e31/rlonly/l2/all"] == "80.0" and k["e31/rlonly/api_usd"] == "12.35" and k["e31/rlonly/gpu_h"] == "2.5"
    cells["e31"]["l3"] = None
    assert "e31/rlonly/l3/success" not in X.tex_keys(r)                 # an unmeasured key is not defined
