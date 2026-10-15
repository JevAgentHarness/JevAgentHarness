"""E13, other agent LLMs on DecisionGym-Hard (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E13: other agent LLMs
on DecisionGym-Hard"): the evaluation-seed and training guards of the new block, the configs against E7e's, the
refitted arm, the projection gate, the slimmed records (the readout and the refit read nothing they drop), the
``price_model`` row of the ledger, the NI helper's rows, the handoff package, and a $0 dry run of every stage with the
simulated LLM and System-1 (no network, no key)."""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

from jevharness.bench import registry as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explore_e13_llms as E  # noqa: E402
import explore_e7_hard as E7  # noqa: E402
import hard_ni  # noqa: E402

BLOCKS = {"luna": (311000, 311099), "deepseek": (311100, 311199)}


# ------------------------------------------------------------------------------------------- seeds and guards
def test_e13_block_is_guarded_and_never_trained_on(monkeypatch):
    from jevharness.training import data_v1, seed_guard

    monkeypatch.delenv(R.EVAL_RUNS_ENV, raising=False)
    for llm, (lo, hi) in BLOCKS.items():
        assert (E.LLMS[llm]["seeds"]["eval"], E.LLMS[llm]["seeds"]["eval"] + E.N_FULL - 1) == (lo, hi)
        for seed0, n in ((lo, 100), (hi, 1), (lo - 5, 10)):
            with pytest.raises(SystemExit, match="exploratory evaluation seeds"):
                R.check_suite_seeds("jevgym", seed0, n)
        for s in (lo, hi):
            assert R.in_ranges(s, R.EXPLORE_EVAL_RANGES)
            assert R.in_ranges(s, R.training_exclusions())
            assert R.in_ranges(s, seed_guard.TRAIN_FORBIDDEN) and R.in_ranges(s, data_v1.FORBIDDEN_RANGES)
        with pytest.raises(ValueError):
            seed_guard.check_train_pool(lo, 10)
    assert (311000, 311999) in R.EXPLORE_TRAIN_EXCLUDED           # the whole E13 block, reserve included
    R.check_suite_seeds("jevgym", 310800, 200)                    # between the E7 and E13 blocks: not guarded
    R.check_suite_seeds("jevgym", 311300, 10)                     # above the E13 evaluation seeds
    monkeypatch.setenv(R.EVAL_RUNS_ENV, "1")
    for lo, hi in BLOCKS.values():
        R.check_suite_seeds("jevgym", lo, 100)


def test_calibration_and_smoke_seeds_are_fresh_and_disjoint():
    cal = [range(E.LLMS[x]["seeds"]["cal"], E.LLMS[x]["seeds"]["cal"] + E.N_FULL) for x in E.LLMS]
    smoke = [range(E.LLMS[x]["seeds"]["smoke"], E.LLMS[x]["seeds"]["smoke"] + E.N_SMOKE) for x in E.LLMS]
    used_cal = set(range(5000, 5400))                             # Stage 1-2 calibrations, E5, E7c / E7d (5300-5399)
    used_dev = set(range(7000, 7276)) | set(range(7290, 7300))    # the E7 pilot and every E7-E7f smoke / route check
    for group in (cal, smoke):
        flat = [s for r in group for s in r]
        assert len(flat) == len(set(flat))                        # one block per LLM
    for r in cal:
        assert set(r) <= set(range(R.CALIBRATION_RANGE[0], R.CALIBRATION_RANGE[1] + 1)) and not set(r) & used_cal
    for r in smoke:
        assert not set(r) & used_dev and not R.overlaps(r[0], len(r), R.EXPLORE_EVAL_RANGES + R.reserved_ranges()
                                                         + R.STAGE2_RESERVED_RANGES)
    for stage, s0 in E.DRY["seeds"].items():                      # the dry run stays on development seeds
        assert s0 + E.DRY["n"][stage] - 1 <= R.DEV_MAX_SEED


# ------------------------------------------------------------------------------------------- configs
def test_configs_keep_e7e_setting_and_change_only_llm_seeds_and_arms():
    e7e = yaml.safe_load((ROOT / "configs/explore/e7_hard/eval_e7e.yaml").read_text())
    strip = lambda s: {k: v for k, v in s.items() if k not in ("seed0", "n")}  # noqa: E731
    jev_e7e = next(a for a in e7e["arms"] if a["name"] == "jev_refit")
    for llm in E.LLMS:
        for stage in ("smoke", "cal", "eval"):
            cfg = E.stage_config(llm, stage, taus=(0.5, 0.6) if stage == "eval" else None)
            assert strip(cfg["suite"]) == strip(e7e["suite"])
            assert cfg["harness_defaults"] == e7e["harness_defaults"] and cfg["schedule"] == e7e["schedule"]
            assert cfg["s1"] == e7e["s1"] and "s1_cache" not in cfg
            assert cfg["suite"]["seed0"] == E.LLMS[llm]["seeds"][stage] and cfg["n_tasks"] == cfg["suite"]["n"]
            assert cfg["llm"] == E.LLMS[llm]["llm"]
        ev = E.stage_config(llm, "eval", taus=(0.5, 0.6))
        assert [a["name"] for a in ev["arms"]] == ["llm_only", "jev_refit"]
        arm = ev["arms"][1]
        # E7e's jev_refit structure, only the two cascade thresholds differ
        assert {k: (v if k != "tool_select" and k != "tool_params" else {**v, "tau": None}) for k, v in arm.items()} == {
            k: (v if k != "tool_select" and k != "tool_params" else {**v, "tau": None}) for k, v in jev_e7e.items()}
        assert (arm["tool_select"]["tau"], arm["tool_params"]["tau"]) == (0.5, 0.6)
        cal = E.stage_config(llm, "cal")
        assert cal["arms"] == [{"name": "jev_shadow_cal", "tool_select": {"mode": "shadow"},
                                "tool_params": {"mode": "shadow"}}]
        sm = E.stage_config(llm, "smoke")
        assert [a["name"] for a in sm["arms"]] == ["llm_only", "jev_shadow_cal", "jev_cascade_dev"]
        assert (sm["arms"][2]["tool_select"]["tau"], sm["arms"][2]["tool_params"]["tau"]) == E.DEV_TAU
    luna = E.LLMS["luna"]["llm"]
    assert luna == {"provider": "openai_responses", "model": "gpt-6-luna", "reasoning_effort": "none", "temperature": 0}
    ds = E.LLMS["deepseek"]["llm"]
    assert ds["extra_body"]["provider"] == {"order": ["StreamLake"], "allow_fallbacks": False}
    assert ds["extra_body"]["reasoning"] == {"enabled": False}


def test_every_e13_model_has_an_exact_price_row():
    from jevharness.accounting import PriceTable

    pt = PriceTable.load()
    for llm, L in E.LLMS.items():
        key = L["llm"].get("price_model") or L["llm"]["model"]
        assert key in pt.prices, key
        if L.get("pin"):
            _, p_in, p_out = L["pin"]
            assert abs(pt.prices[key].input_per_m - p_in * 1e6) < 1e-9 and abs(pt.prices[key].output_per_m - p_out * 1e6) < 1e-9
    assert pt.prices["deepseek/deepseek-v4-flash"].input_per_m == 0.028    # S2-1X's row is untouched


def test_refit_arm_deploys_infinite_thresholds_as_registered():
    fits = {"jev_shadow_cal": {"tool_select": {"tau_hat": None, "always_defer": True},
                               "tool_params": {"tau_hat": 0.93, "always_defer": False}}}
    arm = E.eval_arm(fits)
    assert arm["tool_select"] == {"mode": "cascade", "tau": 1.0, "always_defer": True}
    assert arm["tool_params"] == {"mode": "cascade", "tau": 0.93} and arm["reflection"] == {"mode": "trigger", "tau": 0.5}
    arm = E.eval_arm(taus=(0.9, math.inf))
    assert arm["tool_params"]["tau"] == math.inf
    cfg = yaml.safe_load(yaml.safe_dump(E.stage_config("luna", "eval", taus=(0.9, math.inf))))
    from jevharness.bench.runner import expand_arms

    jev = {a.name: a for a in expand_arms(cfg)}["jev_refit"]
    assert jev.tool_params.tau == math.inf and jev.tool_select.tau == 0.9
    with pytest.raises(SystemExit, match="fit failed"):
        E.eval_arm({"jev_shadow_cal": {"tool_select": {"tau_hat": None}, "tool_params": {"tau_hat": 0.9}}})


# ------------------------------------------------------------------------------------------- gate
def _costs(llm=0.03, jev=0.01, errors=0, no_tool=0):
    return {a: {"n": 2, "errors": errors if a == "llm_only" else 0, "no_tool_call": no_tool if a == "jev_cascade_dev" else 0,
                "no_answer": 0, "llm": llm, "jev": jev, "wall_s": 60.0, "retried_calls": 0}
            for a in ("llm_only", "jev_shadow_cal", "jev_cascade_dev")}


def test_gate_tiers_caps_and_stops():
    caps = {"llm": 60.0, "jev": 5.0}
    g = E.gate("luna", _costs(0.03, 0.01), {"llm": 0.2, "jev": 0.02}, caps=caps)
    assert g["decision"] == "run: calibration 100, evaluation 100" and (g["n_cal"], g["n_eval"]) == (100, 100)
    # 0.2 + 1.2 * (100 * 0.03 + 100 * 0.06) = 11.0 LLM dollars
    assert abs(g["tiers"][0]["projected"]["llm"] - 11.0) < 1e-9
    g = E.gate("luna", _costs(0.03, 0.02), {"llm": 0.2, "jev": 0.04}, caps=caps)     # Jev: 0.04 + 1.2 * 6 = 7.24 > 5
    assert (g["n_cal"], g["n_eval"]) == (50, 50)                                     # 0.04 + 3.6 = 3.64 fits
    g = E.gate("luna", _costs(0.3, 0.01), {"llm": 0.2, "jev": 0.0}, caps=caps)       # 0.2 + 1.2 * 45 = 54.2 fits?
    assert (g["n_cal"], g["n_eval"]) == (50, 50)
    assert E.gate("luna", _costs(0.5, 0.01), {"llm": 0, "jev": 0}, caps=caps)["decision"] == "stop after the smoke"
    assert E.gate("luna", _costs(errors=1), {"llm": 0, "jev": 0}, caps=caps)["decision"].startswith("stop: smoke")
    assert E.gate("luna", _costs(no_tool=1), {"llm": 0, "jev": 0}, caps=caps)["decision"].startswith("stop: smoke")
    ge = E.gate("luna", _costs(0.2, 0.01), {"llm": 30.0, "jev": 1.0}, "pre_eval", n_cal_done=100, caps=caps)
    assert (ge["n_cal"], ge["n_eval"]) == (100, 50)          # 30 + 1.2 * 100 * 0.4 = 78 > 60; 30 + 24 = 54 fits
    ge = E.gate("luna", _costs(0.03, 0.01), {"llm": 5.0, "jev": 1.0}, "pre_eval", n_cal_done=50, max_eval=50, caps=caps)
    assert [t["n_eval"] for t in ge["tiers"]] == [50]        # the evaluation never grows past the first gate's tier


# ------------------------------------------------------------------------------------------- slim records, readout
def _dec(dp, s1, gold, deferred, s2=None, mode="cascade", dist=None):
    d = {"dp": dp, "mode": mode, "question": "q" * 50, "options": ["a", "b"] if dp != "reflection" else None,
         "s1_value": s1, "s1_conf": 0.95, "s1_dist": dist, "deferred": deferred, "s2_value": s2,
         "final_value": s2 if deferred else s1, "gold": gold, "state": "STATE " * 100, "shadow": False,
         "gold_status": "ok", "gold_aux": {}, "s2_correct": (s2 == gold) if deferred else None}
    if dp == "tool_select":
        d["gold_aux"] = {"acceptable_tools": [gold]}
    return d


def _rec(tid, success, cost, wall, decisions=(), s1_cost=0.0):
    return {"task_id": tid, "config": "x", "backend": None, "llm": "m", "final_answer": "a: 1.00\nb: 2.00\ntotal 3.00",
            "result": {"success": success}, "usage": {"total": {"cost_usd": cost}, "s1": {"cost_usd": s1_cost, "calls": 3},
                                                      "s2": {"cost_usd": cost - s1_cost, "calls": 4}},
            "steps": [{"step": 0, "tool": "find_user", "args": {"name": "x"}, "observation": "OBS " * 200, "llm_calls": 1,
                       "decisions": list(decisions)}],
            "calls": [{"component": "s2", "decision_point": "tool_select", "latency_ms": 900.0, "cost_usd": cost,
                       "meta": {"n_tools": 69}, "backend": "openai_responses"}],
            "wall_ms": wall * 1000, "error": None, "meta": {}}


def _write_run(run: Path, arm: str, recs) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_slim_records_drop_only_what_no_readout_reads(tmp_path):
    big = {f"tool_{i}": 1.0 / 70 for i in range(69)} | {"find_user": 0.3}
    d = _dec("tool_select", "find_user", "find_user", False, dist=big)
    r = E.slim_record(_rec("t", True, 0.1, 1.0, [d, _dec("reflection", False, False, False, mode="trigger",
                                                        dist={"yes": 0.1, "no": 0.9})]))
    st = r["steps"][0]
    assert "observation" not in st and all("state" not in x and "question" not in x for x in st["decisions"])
    assert len(st["decisions"][0]["s1_dist"]) == 5 and "find_user" in st["decisions"][0]["s1_dist"]
    assert st["decisions"][1]["s1_dist"] == {"yes": 0.1, "no": 0.9}           # yes/no distributions kept whole
    assert st["args"] == {"name": "x"} and r["calls"] and r["usage"] and r["slim"] == "e13"
    # E7's readout gives the same numbers on the raw records and on their slimmed, gzipped copies
    runs = tmp_path / "raw"
    tids = [f"jevgym-multi-{7300 + i}-d60-near-strict" for i in range(8)]
    dec = [_dec("tool_select", "find_user", "find_user", False, dist=big), _dec("tool_params", "U1", "U1", False),
           _dec("tool_params", "U2", "U1", True, s2="U1"), _dec("reflection", False, False, False, mode="trigger")]
    run = runs / "e13_luna_eval"
    _write_run(run, "llm_only", [_rec(t, bool(i % 3), 0.04, 50.0) for i, t in enumerate(tids)])
    _write_run(run, "jev_refit", [_rec(t, bool(i % 2), 0.03, 70.0, dec, 0.005) for i, t in enumerate(tids)])
    (run / "experiment.yaml").write_text(yaml.safe_dump(E.stage_config("luna", "eval", n=8, taus=(0.9, 0.8))))
    E.slim_run(run, tmp_path / "slim" / "e13_luna_eval")
    for name in ("smoke", "cal"):
        (runs / f"e13_luna_{name}").mkdir()
        (tmp_path / "slim" / f"e13_luna_{name}").mkdir()
    key = E._exp_key("luna", tmp_path)
    a = E7.summarize(runs, 300, 0, window=[], exp=key)
    b = E7.summarize(tmp_path / "slim", 300, 0, window=[], exp=key)
    for arm in ("llm_only", "jev_refit"):
        for k in ("success_all", "cost_mean", "answer_forms", "failure_types"):
            assert a["arms"][arm][k] == b["arms"][arm][k]
    ka, kb = a["arms"]["jev_refit"], b["arms"]["jev_refit"]
    assert {k: ka["cmp"][k] for k in ("delta", "wins", "losses", "saving", "n")} == {
        k: kb["cmp"][k] for k in ("delta", "wins", "losses", "saving", "n")}
    assert ka["points"] == kb["points"] and ka["paired_per_episode"] == kb["paired_per_episode"]
    assert abs(kb["points"]["tool"]["coverage"] - 2 / 3) < 1e-12
    with gzip.open(tmp_path / "slim" / "e13_luna_eval" / "jev_refit.jsonl.gz", "rt") as f:
        assert len(f.read().splitlines()) == 8


def test_price_model_row_is_charged_and_never_sent():
    from jevharness.accounting import Ledger
    from jevharness.llm.base import ChatResponse, LLMClient

    seen = {}

    class Fake(LLMClient):
        provider = "fake"

        def _chat(self, messages, tools=None, **kw):
            seen.update(kw)
            return ChatResponse(text="ok", input_tokens=1_000_000, output_tokens=1_000_000, model="deepseek/deepseek-v4-flash")

    led = Ledger()
    Fake("deepseek/deepseek-v4-flash", price_model="deepseek/deepseek-v4-flash@streamlake-2026-10-07",
         temperature=0).chat([{"role": "user", "content": "x"}], ledger=led)
    rec = led.records[-1]
    assert "price_model" not in seen and seen["temperature"] == 0
    assert abs(rec.cost_usd - (0.042 + 0.084)) < 1e-12 and rec.meta["price_model"].endswith("2026-10-07")
    led2 = Ledger()
    Fake("deepseek/deepseek-v4-flash").chat([{"role": "user", "content": "x"}], ledger=led2)
    assert abs(led2.records[-1].cost_usd - (0.028 + 0.056)) < 1e-12 and "price_model" not in led2.records[-1].meta


def test_ni_helper_has_the_e13_rows():
    for llm in E.LLMS:
        assert hard_ni.ROWS[f"e13_{llm}"] == (f"e13/{llm}/jev_refit/wins", f"e13/{llm}/jev_refit/losses",
                                              f"e13/{llm}/jev_refit/n")
    keys = {"e13/luna/jev_refit/wins": "9", "e13/luna/jev_refit/losses": "7", "e13/luna/jev_refit/n": "100"}
    res = hard_ni.compute(keys)
    assert set(res) == {"e13_luna"} and res["e13_luna"]["n"] == 100 and abs(res["e13_luna"]["delta"] - 2.0) < 1e-12


# ------------------------------------------------------------------------------------------- handoff package
# ------------------------------------------------------------------------------------------- $0 dry run
def test_dry_run_every_stage_with_simulated_models(tmp_path, monkeypatch):
    monkeypatch.setitem(E.DRY, "n", {"smoke": 2, "cal": 4, "eval": 4})
    runs, out = tmp_path / "runs", tmp_path / "out"
    d = E.run("deepseek", runs, out, dry=True, every=0.5)
    assert d.get("finished") and not d.get("stopped")
    assert d["gate"]["decision"] == "run: calibration 100, evaluation 100"
    st = json.loads((out / "deepseek" / "status.json").read_text())
    assert st["fit"]["at"] <= st["eval"]["invocations"][0]["start"]
    for stage in ("smoke", "cal", "eval"):
        assert (out / "records/deepseek" / f"e13dry_deepseek_{stage}" / "slim.json").exists()
    assert not any(str(tmp_path) in f.read_text() for f in (out / "deepseek").rglob("*.json"))   # paths scrubbed
    # the registered fit read from the slimmed, gzipped calibration records gives the same thresholds
    refit_runs = tmp_path / "refit"
    (refit_runs / "e13dry_deepseek_cal").mkdir(parents=True)
    (refit_runs / "e13dry_deepseek_cal" / "jev_shadow_cal.jsonl").write_bytes(
        gzip.decompress((out / "records/deepseek/e13dry_deepseek_cal/jev_shadow_cal.jsonl.gz").read_bytes()))
    re_fit = E7.fit_calibration(refit_runs, tmp_path / "refit_out", run_name="e13dry_deepseek_cal",
                                cal_arms={"jev_shadow_cal": "jev_refit"}, tex=None)
    assert {dp: E._tau_of(re_fit["fits"]["jev_shadow_cal"][dp]) for dp in ("tool_select", "tool_params")} == st["fit"]["taus"]
    res = E.summarize(out, None, ["deepseek"], n_boot=200)
    s = res["llms"]["deepseek"]
    assert s["state"] == "run" and s["summary"]["dry_run"] and s["summary"]["arms"]["jev_refit"]["cmp"]["n"] == 4
    keys = E.tex_keys(res)
    assert keys["e13/deepseek/state"] == "run" and "e13/deepseek/jev_refit/delta" in keys
    assert keys["e13/luna/state"] == "not run" and keys["e13/deepseek/spend"] == "0.00"
    md = E.merged_markdown(res)
    assert "DRY RUN" in md and "exploratory (E13)" in md
    # a rerun resumes: nothing runs again
    n_inv = len(st["eval"]["invocations"])
    E.run("deepseek", runs, out, dry=True, every=0.5)
    assert len(json.loads((out / "deepseek" / "status.json").read_text())["eval"]["invocations"]) == n_inv


def test_a_crashed_runner_leaves_the_stage_unfinished(tmp_path, monkeypatch):
    """2026-10-07: the disk filled while the Luna calibration ran, a record could not be written and the runner died;
    the stage must stay unfinished (a rerun resumes it), never be marked done with the tasks it lacks."""
    calls = []

    def fake_launch(cfg, cfg_path, runs, env_extra, llm, out, caps, margin, dry, every):
        calls.append(cfg["name"])
        return {"start": "t0", "end": "t1", "returncode": 1, "capped": None, "disk": None}

    monkeypatch.setattr(E, "_launch", fake_launch)
    st = E.Status(tmp_path / "out" / "luna" / "status.json")
    cfg = E.stage_config("luna", "cal", dry=True)
    with pytest.raises(SystemExit, match="rerun to resume"):
        E.run_stage("luna", "cal", cfg, tmp_path / "runs", tmp_path / "out", st, True, 3, {"llm": 0, "jev": 0}, 0.1)
    assert len(calls) == 1 and "done" not in (st.get("cal") or {})
    monkeypatch.setattr(E, "_launch", lambda *a, **k: {"start": "t0", "end": "t1", "returncode": None, "capped": None,
                                                         "disk": {"free_mb": 10.0}})
    with pytest.raises(SystemExit, match="disk"):
        E.run_stage("luna", "cal", cfg, tmp_path / "runs", tmp_path / "out", st, True, 3, {"llm": 0, "jev": 0}, 0.1)
    assert "done" not in (st.get("cal") or {})


def test_order_check_reads_the_runner_schedule_and_cache_share(tmp_path):
    st = {"fit": {"at": "2026-10-07T14:56:05+00:00"},
          "eval": {"invocations": [{"start": "2026-10-07T15:43:27+00:00"}]}}          # a resumed invocation only
    sched = {"invocations": [{"k": 0, "started": "2026-10-07T15:11:26+00:00"},
                             {"k": 1, "started": "2026-10-07T15:43:28+00:00"}]}
    oc = E.order_check(st, sched)
    assert oc["ok"] and oc["eval_start"] == "2026-10-07T15:11:26+00:00"
    assert not E.order_check({**st, "fit": {"at": "2026-10-07T15:20:00+00:00"}}, sched)["ok"]
    run = tmp_path / "e13_luna_eval"
    rec = _rec("t1", True, 0.01, 1.0)
    rec["calls"] = [{"component": "s2", "input_tokens": 1000, "meta": {"cached_input_tokens": 900}},
                    {"component": "s1", "input_tokens": 500, "meta": {}}]
    _write_run(run, "llm_only", [rec])
    c = E.cache_share(run)
    assert abs(c["llm_only"]["cached_share"] - 0.9) < 1e-12 and c["llm_only"]["steps"] == 1.0 and "jev_refit" not in c


def test_calibration_keys_carry_both_accuracies(monkeypatch):
    monkeypatch.setattr(E7, "to_tex_keys", lambda res: {})       # E7's own keys are tested in test_explore_e7.py
    fits = {"fits": {"jev_shadow_cal": {
        "tool_select": {"tau_hat": 0.49, "always_defer": False,
                        "at_tau": {"n": 10, "coverage": 1.0, "s1_acc_taken": 0.9, "s1_acc_all": 0.9, "llm_acc_all": 0.8},
                        "op_at_tau": {"n": 4, "coverage": 1.0, "s1_acc_all": 1.0, "llm_acc_all": 0.75}},
        "tool_params": {"tau_hat": None, "always_defer": True, "at_tau": {"n": 5, "coverage": 0.0}}}}}
    res = {"llms": {"luna": {"state": "run", "summary": {
        "meta": {"exp": "e13/luna", "suite": {}, "spend": {"total": 0.0}, "cap": 65, "arms": ["llm_only", "jev_refit"]},
        "arms": {}, "tariff": {}, "calibration": fits, "spend_split": {"llm": 0.0, "jev": 0.0, "total": 0.0},
        "status": {}}}}}
    v = E.tex_keys(res)
    assert v["e13/luna/cal/toolselect/llmaccall"] == "80.0" and v["e13/luna/cal/toolselect/n"] == "10"
    assert v["e13/luna/cal/op/llmaccall"] == "75.0" and v["e13/luna/cal/op/n"] == "4"
    assert v["e13/luna/cal/toolparams/tau"] == "always defer" and v["e13/deepseek/state"] == "not run"


def test_cost_diagnosis_splits_calls_gap_and_perfect_fix(tmp_path):
    run = tmp_path / "e13_luna_eval"

    def rec(tid, calls):
        r = _rec(tid, True, 0.01, 1.0)
        r["calls"] = calls
        return r
    ref_calls = [{"component": "s2", "input_tokens": 1000, "meta": {"n_tools": 69, "cached_input_tokens": 900,
                                                                     "cache_write_tokens": 100}}]
    arm_calls = [{"component": "s2", "input_tokens": 400, "meta": {"n_tools": 1, "cached_input_tokens": 0,
                                                                    "cache_write_tokens": 400}},
                 {"component": "s2", "input_tokens": 100, "meta": {"n_tools": 0}},
                 {"component": "s1", "input_tokens": 999, "meta": {}}]
    _write_run(run, "llm_only", [rec("t1", ref_calls)])
    _write_run(run, "jev_refit", [rec("t1", arm_calls)])
    res = {"cache": E.cache_share(run),
           "arms": {"jev_refit": {"cmp": {"cost": 0.008, "cost_ref": 0.002},
                                  "paired_per_episode": {"s1_cost": 0.005, "s2_cost": 0.003},
                                  "ref_paired_per_episode": {"s2_cost": 0.002}}}}
    d = E.cache_diag(run, res, "luna")
    g = d["arms"]["jev_refit"]["groups"]
    assert g["onetool"]["calls"] == 1 and g["onetool"]["cached_share"] == 0.0 and abs(g["onetool"]["token_share"] - 0.8) < 1e-12
    assert g["notools"]["calls"] == 1 and d["arms"]["llm_only"]["groups"]["alltools"]["cached_share"] == 0.9
    assert abs(d["ratio"] - 4.0) < 1e-12 and abs(d["jev_share_of_gap"] - 0.005 / 0.006) < 1e-12
    # perfect fix: 500 LLM input tokens at LLM-only's 0.002 $ / 1000 tokens = 0.001, plus Jev 0.005 -> 3x LLM-only
    assert abs(d["perfect_fix_llm_cost"] - 0.001) < 1e-12 and abs(d["perfect_fix_ratio"] - 3.0) < 1e-12
    assert abs(d["write_surcharge"]["jev_refit"] - 400 * 0.25 * 0.10 / 1e6) < 1e-15        # gpt-6-luna: writes 1.25x
