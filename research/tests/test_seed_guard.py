"""Round-5 (JevAgentRL) seed rules: training >= 10000 minus every reserved range (incl. Stage-2 300000-300999),
development 0-999 / 1000-4999, calibration 5000-5149 only for the tau-hat* fit (training/seed_guard.py)."""
from __future__ import annotations

import pytest

from jevharness.training import seed_guard as sg


def test_train_pool_rules():
    sg.check_train_pool(10000, 512)
    sg.check_train_pool(40000, 2000)
    for seed0, n in ((5000, 10), (9990, 20), (99990, 20), (199900, 200), (299990, 20), (300000, 1), (300500, 10)):
        with pytest.raises(ValueError):
            sg.check_train_pool(seed0, n)
    sg.check_train_pool(301000, 10)  # past the Stage-2 reservation


def test_dev_and_calibration_pools():
    sg.check_dev_pool(0, 50)
    sg.check_dev_pool(1000, 64)
    for seed0, n in ((990, 20), (4990, 20), (5000, 10), (200000, 5), (300000, 5)):
        with pytest.raises(ValueError):
            sg.check_dev_pool(seed0, n)
    sg.check_cal_pool(5000, 150)
    with pytest.raises(ValueError):
        sg.check_cal_pool(5000, 151)
    with pytest.raises(ValueError):
        sg.check_disjoint((1000, 64), (1050, 10))


def test_exclude_spec_and_data_v1_forbidden_cover_stage2():
    from jevharness.training import data_v1
    from jevharness.training.harvest import seed_allowed, seed_ranges

    excl = seed_ranges(sg.train_exclude_spec())
    assert (300000, 300999) in excl and (200000, 200499) in excl and (5000, 5999) in excl
    assert not seed_allowed(300123, 10000, excl) and seed_allowed(40000, 10000, excl)
    assert sg.STAGE2_RANGE in data_v1.FORBIDDEN_RANGES
    with pytest.raises(ValueError):
        data_v1.check_seed_plan(300000, 10)


def test_opd_strict_seeds_refuses_calibration_eval_and_stage2_training():
    pytest.importorskip("torch")
    from jevharness.training.opd import parse_args

    base = ["--student", "tiny", "--teacher", "sim", "--executor", "sim"]
    with pytest.raises(SystemExit):
        parse_args(base + ["--seeds", "300100", "--n-tasks", "16"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--strict-seeds", "--eval-every", "5", "--eval-seed0", "5000"])
    with pytest.raises(SystemExit):  # never-run ranges are refused with or without --strict-seeds
        parse_args(base + ["--eval-every", "5", "--eval-seed0", "300000"])
    a = parse_args(base + ["--strict-seeds", "--seeds", "40000", "--n-tasks", "512", "--eval-every", "5",
                           "--eval-seed0", "1000", "--eval-n", "64", "--eval-arms", "a1",
                           "--arms", "[{name: a1, tool_select: {mode: replace}}, {name: a2, llm: flash, "
                           "tool_select: {mode: shadow}}]", "--llms", "{flash: {provider: simulated, model: s}}"])
    assert a.eval_arms == ["a1"] and a.llms["flash"]["provider"] == "simulated"
    with pytest.raises(SystemExit):  # an arm naming an unknown executor
        parse_args(base + ["--arms", "[{name: a1, llm: nope, tool_select: {mode: replace}}]"])


def test_grpo_online_seed_policy():
    pytest.importorskip("torch")
    from jevharness.training.grpo_online import load_config

    load_config(None, [], model="m").check_seeds()          # registered defaults (A6.4) pass
    with pytest.raises(ValueError):
        load_config(None, ["seed_policy=strict"], model="m").check_seeds()   # default suite 1000 is not training
    ok = load_config(None, ["seed_policy=strict", "suite.seed0=40000", "eval_suite.seed0=1000"], model="m")
    ok.check_seeds()
    with pytest.raises(ValueError):
        load_config(None, ["suite.seed0=300000"], model="m").check_seeds()     # Stage-2: never run
    cfg = load_config(None, ["executors={lite: {provider: simulated, model: s}}",
                             "arms=[{name: a, llm: lite, tool_select: {mode: replace}}]", "eval_arms=[a]"], model="m")
    assert [a.name for a in cfg.harness_arms()] == ["a"]
    with pytest.raises(ValueError):
        load_config(None, ["arms=[{name: a, llm: nope, tool_select: {mode: replace}}]"], model="m").harness_arms()
