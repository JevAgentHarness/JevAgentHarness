"""Training data v1 builder (src/jevharness/training/data_v1.py, scripts/train/make_data_v1.sh): offline, simulated
executor and System-1 only. The hold-out tests are the point: no reserved JevGym seed, no task of the 210-decision
gate set and no E1 dev-prefix / BFCL / When2Call-test item may enter v1."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from jevharness.decision.types import Choice, Noul
from jevharness.harness.tools import OTHER
from jevharness.training import data_v1 as v1
from jevharness.training.data import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SIM_LLM = {"provider": "simulated", "model": "scripted", "error_rate": 0.25, "simulated_latency_ms": 0}
TOOLS = ["find_user", "get_orders", "get_order", "get_order_status", "cancel_order", "ship_order", "refund",
         "ask_user", "request_callback", "finish", "get_weather", "translate_text"]


def _row(rid: str, *, dp: str = "tool_select", kind: str = "choice", state: str = "TASK: cancel an order",
         options: list[str] | None = None, gold: str | None = None, **meta) -> dict:
    options = options or ["a", "b"]
    gold = gold or options[0]
    return {"id": rid, "source": "test", "dp": dp, "kind": kind, "state": state, "question": "q?",
            "options": options, "target": {o: float(o == gold) for o in options}, "q_values": None, "meta": meta}


# ------------------------------------------------------------------ seed plan
@pytest.mark.parametrize("seed0,n", [(0, 10), (999, 1), (5000, 150), (5100, 10), (9990, 20), (99990, 20),
                                     (100000, 1), (100499, 1), (199999, 2), (200000, 500), (200400, 1)])
def test_seed_plan_refuses_dev_calibration_retired_and_evaluation_seeds(seed0: int, n: int):
    with pytest.raises(ValueError):
        v1.check_seed_plan(seed0, n)
    with pytest.raises(ValueError):
        v1.build_experiment(seed0=seed0, n=n)


def test_seed_plan_accepts_training_seeds_and_the_v1_default():
    v1.check_seed_plan(v1.V1_SEED0, 600)
    v1.check_seed_plan(10000, 1)
    v1.check_seed_plan(100500, 10)
    exp = v1.build_experiment(n=600, n_by_arm={"lite_guard": 150})
    s = exp["suite"]
    assert s["seed0"] == 30000 and s["n"] == 600
    frozen = json.loads((ROOT / "configs/eval/jevgym_eval_tasks.json").read_text())["suite"]
    assert {k: v for k, v in s.items() if k not in ("seed0", "n")} == \
        {k: v for k, v in frozen.items() if k not in ("seed0", "n")}       # the evaluation SETTING, training SEEDS
    assert [a["name"] for a in exp["arms"]] == ["lite_shadow", "nearmiss_replace", "lite_guard"]
    assert next(a for a in exp["arms"] if a["name"] == "lite_guard")["n_tasks"] == 150
    assert exp["llm"]["model"] == "gemini-3.5-flash-lite"


# ------------------------------------------------------------------ behaviour policy
def test_nearmiss_policy_answers_gold_without_noise_and_near_misses_with_it():
    exact = v1.NearMissBackend(eps_tool=0.0, eps_param=0.0, eps_gate=0.0)
    q = Choice("next tool", TOOLS, decision_point="tool_select", gold="get_orders")
    assert exact.ask("s", q).value == "get_orders"
    assert exact.ask("s", Noul("ok?", decision_point="stop", gold=False)).value is False

    noisy = v1.NearMissBackend(eps_tool=1.0, eps_param=1.0, eps_gate=1.0, seed=3)
    picks = {noisy.ask(f"state {i}", Choice("next tool", TOOLS, decision_point="tool_select",
                                             gold="get_orders")).value for i in range(60)}
    assert "get_orders" not in picks
    # the near-duplicates of get_orders and the skipped-to actions dominate; the modes are all exercised
    assert picks & {"get_order", "get_order_status"} and picks & {"cancel_order", "ship_order", "refund", "finish"}
    modes = {k.split(":")[1] for k in noisy.counts}
    assert {"skip", "near"} <= modes
    assert noisy.ask("s", Noul("ok?", decision_point="stop", gold=True)).value is False   # flipped gate
    # deterministic per (state, question, gold)
    again = v1.NearMissBackend(eps_tool=1.0, eps_param=1.0, eps_gate=1.0, seed=3)
    q2 = Choice("next tool", TOOLS, decision_point="tool_select", gold="ask_user")
    assert again.ask("x", q2).value == v1.NearMissBackend(eps_tool=1.0, seed=3).ask("x", q2).value
    # an ask_user gold is answered by acting / a near-duplicate of the ask (round 2: "cancel_order where gold is ask")
    acts = {v1.NearMissBackend(eps_tool=1.0, seed=s).ask("x", q2).value for s in range(40)}
    assert acts & {"cancel_order", "ship_order", "refund"} and "ask_user" not in acts


def test_nearmiss_params_pick_a_confusable_value():
    opts = ["Dana Kim", "U1178", "U2210", "ORD-2522", "ORD-2525", "pending", OTHER]
    b = v1.NearMissBackend(eps_param=1.0, param_modes={"confusable": 1.0})
    for i in range(20):
        a = b.ask(f"s{i}", Choice("order_id?", opts, decision_point="tool_params", gold="ORD-2522"))
        assert a.value == "ORD-2525"                      # the other order id: same lexical shape
    b2 = v1.NearMissBackend(eps_param=1.0, param_modes={"other": 1.0})
    assert b2.ask("s", Choice("order_id?", opts, decision_point="tool_params", gold="ORD-2522")).value == OTHER
    assert v1._shape("ORD-4759") == v1._shape("ORD-12") and v1._shape("U8166") != v1._shape("ORD-1")


def test_nearmiss_policy_costs_nothing_and_describes_itself():
    b = v1.NearMissBackend()
    from jevharness.accounting import Ledger

    led = Ledger()
    b.decide("s", [Choice("next tool", TOOLS, decision_point="tool_select", gold="find_user")], ledger=led)
    assert led.records[0].cost_usd == 0.0 and led.records[0].priced
    d = b.describe()
    assert d["backend"] == "nearmiss" and d["eps_tool"] == 0.2


# ------------------------------------------------------------------ hold-out rules
@pytest.fixture
def eval_texts(tmp_path: Path) -> v1.Decontaminator:
    return v1.Decontaminator(["USER: Can you help me translate this English text to Danish: 'The quick brown fox "
                              "jumps over the lazy dog.'", "what is the weather in Paris today"], n=8)


def test_no_reserved_seed_gate_task_or_e1_item_can_enter_v1(eval_texts: v1.Decontaminator):
    gate = {"jevgym-clarify-30001-d60-near-strict"}
    e1 = {"irrelevance_14/tool", "w2c/0c0ecee8-729a-4d62-88aa-18e81c381f21"}
    bad = [
        _row("jevgym-clarify-200000-d60-near-strict/lite_shadow:0/tool_select"),    # evaluation seed
        _row("jevgym-multi-200499-d60-near-strict/lite_shadow:3/stop:shadow"),      # evaluation seed (last)
        _row("jevgym-multi-100010-d60-near-strict/lite_shadow:0/tool_select"),      # retired evaluation range
        _row("jevgym-clarify-5000-d60-near-strict/lite_shadow:0/tool_select"),      # calibration
        _row("jevgym-clarify-5149-d60-near-strict/lite_shadow:0/tool_select"),      # calibration
        _row("jevgym-cancel-42-d30-mixed-strict/0/tool_select"),                    # dev seed
        _row("jevgym-cancel-9999-d30-mixed-strict/0/tool_select"),                  # below the training floor
        _row("jevgym-clarify-30001-d60-near-strict/nearmiss_replace:2/tool_params"),   # gate-dev task
        _row("irrelevance_14/tool"),                                                # E1 dev-prefix id
        _row("w2c/0c0ecee8-729a-4d62-88aa-18e81c381f21/x"),                         # E1 dev-prefix task
        _row("w2ctrain-abc/when2call", state="AVAILABLE TOOLS:\n(none)\n\nUSER: Can you help me translate this "
             "English text to Danish: 'The quick brown fox jumps over the lazy dog.'"),   # When2Call-test text
        _row("w2ctrain-def/when2call", state="AVAILABLE TOOLS:\n(none)\n\nUSER: what is the weather in Paris today"),
    ]
    good = [_row("jevgym-clarify-30000-d60-near-strict/lite_shadow:0/tool_select"),
            _row("jevgym-multi-10003-d60-near-strict/0/stop:shadow", dp="stop", kind="noul", options=["yes", "no"]),
            _row("w2ctrain-0001/when2call", state="AVAILABLE TOOLS:\n(none)\n\nUSER: book me a table for two")]
    kept, info = v1.verify_exclusions(bad + good, gate_tasks=gate, e1_ids=e1, decon=eval_texts)
    assert [r["id"] for r in kept] == [r["id"] for r in good]
    assert info["dropped"] == {"reserved_or_low_seed": 7, "gate_dev_task": 1, "e1_dev_prefix_id": 2,
                               "ngram_overlap_eval": 2}
    v1.assert_clean(kept, gate_tasks=gate, e1_ids=e1)
    for r in bad[:10]:
        with pytest.raises(AssertionError):
            v1.assert_clean([r], gate_tasks=gate, e1_ids=e1)


def test_gate_dev_tasks_and_e1_ids_load_from_the_repo_files():
    gate = v1.gate_dev_tasks()
    assert len(gate) == 13 and all(t.startswith("jevgym-") for t in gate)
    e1 = v1.e1_dev_prefix_ids()
    assert len(e1) == 299 and "irrelevance_14/tool" in e1


def test_decontaminator_matches_8grams_and_whole_short_texts():
    d = v1.Decontaminator(["Find the area of a triangle with base 10 and height 5 units please", "hi there"], n=8)
    assert d.hit("could you find the area of a triangle with base 10 and height 5")
    assert not d.hit("find the area of a circle with radius 10")
    assert d.hit("Hi there!") and not d.hit("hi")


def test_query_text_reads_the_user_turn_or_the_task_line():
    assert v1.query_text({"state": "AVAILABLE TOOLS:\nx\n\nUSER: book a flight"}) == "book a flight"
    assert v1.query_text({"state": "TASK: Cancel Dana's order.\n\nCURRENT OBSERVATION:\nok"}) == "Cancel Dana's order."


# ------------------------------------------------------------------ dedupe / balance / augmentation
def test_dedupe_merges_identical_decisions_across_arms():
    a = _row("t-1/lite_shadow:0/tool_select", arm="lite_shadow")
    b = {**_row("t-1/lite_guard:0/tool_select", arm="lite_guard")}
    c = _row("t-1/lite_guard:1/tool_select", state="other", arm="lite_guard")
    out, n = v1.dedupe([a, b, c])
    assert n == 1 and [r["id"] for r in out] == [a["id"], c["id"]]
    assert out[0]["meta"]["arms"] == ["lite_shadow", "lite_guard"]


def test_balance_caps_stop_trigger_no_and_other_args_but_keeps_recovery_rows():
    yes = [_row(f"t{i}/a:0/stop", dp="stop", kind="noul", options=["yes", "no"], gold="yes") for i in range(5)]
    no = [_row(f"t{i}/a:1/stop", dp="stop", kind="noul", options=["yes", "no"], gold="no") for i in range(100)]
    no_rec = [_row(f"r{i}/a:4/stop", dp="stop", kind="noul", options=["yes", "no"], gold="no", after_error=1)
              for i in range(3)]
    par = [_row(f"p{i}/a:0/tool_params", dp="tool_params", options=["x", OTHER], gold="x") for i in range(30)]
    oth = [_row(f"o{i}/a:0/tool_params", dp="tool_params", options=["x", OTHER], gold=OTHER) for i in range(30)]
    guard = [_row(f"g{i}/a:0/tool_select:guard", kind="noul", options=["yes", "no"], gold="no") for i in range(9)]
    for g in guard:
        g["question"] = "The agent proposes to call `x` with arguments {}. Is this the right next step?"
    out, info = v1.balance(yes + no + no_rec + par + oth + guard, max_no_per_yes=4, max_other_frac=0.25)
    b = info["after"]
    assert b["stop_trigger:yes"] == 5 and b["stop_trigger:no"] == 20 and b["tool_params"] == 30
    assert b["tool_params:OTHER"] == 10 and b["tool_guard:no"] == 9
    ids = {r["id"] for r in out}
    assert all(r["id"] in ids for r in no_rec)            # recovery states are never subsampled


def test_knockout_final_variant_keeps_gold_and_hard_negatives():
    opts = [f"{t} -- desc of {t}" for t in TOOLS] + [f"dist_{i} -- unrelated" for i in range(20)]
    r = _row("t-1/a:2/tool_select", options=opts, gold="get_orders -- desc of get_orders")
    v = v1.knockout_final_variant(r)
    names = [o.split(" -- ")[0] for o in v["options"]]
    assert v["id"] == "t-1/a:2/tool_select#ko" and len(names) == 4 and "get_orders" in names
    assert set(names) <= {"get_orders", "get_order", "get_order_status", "cancel_order", "ship_order", "refund",
                          "finish", "ask_user"}
    assert v["target"]["get_orders -- desc of get_orders"] == 1.0 and abs(sum(v["target"].values()) - 1) < 1e-9
    assert v1.knockout_final_variant(_row("t/a:0/tool_select", options=opts[:5], gold=opts[0])) is None
    rows, n = v1.add_knockout_variants([r] * 10, frac=1.0)
    assert n == 10 and len(rows) == 20


# ------------------------------------------------------------------ end to end (simulated executor, training seeds)
@pytest.fixture(scope="module")
def sim_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("runs")
    exp = v1.build_experiment("v1_sim", seed0=30000, n=4, llm=SIM_LLM, workers=2,
                              n_by_arm={"lite_guard": 2})
    v1.collect(exp, out, s1=v1.NearMissBackend(eps_tool=0.3), progress=False)
    return out / "v1_sim"


def test_collect_and_harvest_tag_arms_seeds_and_recovery_states(sim_run: Path, tmp_path: Path):
    assert (sim_run / "nearmiss_counts.json").exists()
    rows, summ = v1.harvest_runs(sim_run, work_dir=tmp_path)
    assert set(summ["arms"]) == {"lite_shadow", "nearmiss_replace", "lite_guard"}
    assert summ["arms"]["lite_guard"]["episodes"] == 2 and summ["arms"]["lite_shadow"]["episodes"] == 4
    assert {r["meta"]["arm"] for r in rows} == set(summ["arms"])
    for r in rows:
        task, rest = r["id"].split("/", 1)
        assert rest.startswith(r["meta"]["arm"] + ":") and r["meta"]["jevgym_seed"] >= 30000
        assert 2 <= len(r["options"]) <= 52
    assert {r["dp"] for r in rows} >= {"tool_select", "tool_params", "stop"}
    sel = [r for r in rows if r["dp"] == "tool_select" and r["kind"] == "choice"]
    assert all(" -- " in o for r in sel for o in r["options"])          # rendered as the served model sees them
    assert any(r["meta"]["after_error"] for r in rows)                   # the near-miss arm reaches recovery states


def test_build_cli_writes_clean_splits_and_a_manifest(sim_run: Path, tmp_path: Path):
    gate_task = "jevgym-multi-30001-d60-near-strict"
    gate = tmp_path / "gate_dev.jsonl"
    write_jsonl([_row(f"{gate_task}/0/tool_select")], gate)
    out, work = tmp_path / "v1", tmp_path / "work"
    v1.main(["build", "--runs", str(sim_run), "--out", str(out), "--work", str(work), "--v0-train", "",
             "--replay", "", "--gate-dev", str(gate), "--dev-frac", "0.25", "--ko-frac", "1.0"])
    train, dev = read_jsonl(work / "train_gold.jsonl"), read_jsonl(out / "dev.jsonl.gz")
    assert not (out / "_work").exists() and {p.name for p in out.glob("*.jsonl")} == set()   # gz only
    assert train and dev
    tasks_tr, tasks_dev = {v1.task_of(r["id"]) for r in train}, {v1.task_of(r["id"]) for r in dev}
    assert not tasks_tr & tasks_dev and gate_task not in tasks_tr | tasks_dev
    v1.assert_clean(train + dev, gate_tasks={gate_task}, e1_ids=v1.e1_dev_prefix_ids())
    assert any(r["id"].endswith("#ko") for r in train) and not any(r["id"].endswith("#ko") for r in dev)
    info = json.loads((out / "build_info.json").read_text())["build"]
    assert info["exclusions"]["dropped"].get("gate_dev_task", 0) > 0
    # labels: a simulated teacher (the only kind that may see gold) -> soft targets q = alpha * gold + (1-alpha) p_T
    v1.main(["label", "--out", str(out), "--work", str(work), "--teacher", "sim", "--cache", "", "--workers", "1"])
    lab = read_jsonl(out / "train.jsonl.gz")
    assert len(lab) == len(train) and (out / "dev_teacher.summary.json").exists()
    assert any((r["meta"] or {}).get("label_source") == "teacher" for r in lab)
    md = tmp_path / "MANIFEST.md"
    v1.main(["manifest", "--out", str(out), "--info", str(out / "build_info.json"), "--md", str(md)])
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["files"]["train.jsonl.gz"]["rows"] == len(lab) and "total" in man["cost"]
    assert "Hold-out rules verified" in md.read_text()


# ------------------------------------------------------------------ scripts and the built data
@pytest.mark.parametrize("script", ["make_data_v1.sh", "12_sft_v1.sh"])
def test_v1_scripts_parse(script: str):
    subprocess.run(["bash", "-n", str(ROOT / "scripts/train" / script)], check=True)


def test_make_data_v1_dry_run_wires_every_step(tmp_path: Path):
    env = {**os.environ, "DRY_RUN": "1", "OUT": str(tmp_path / "v1"), "RUNS_DIR": str(tmp_path / "runs"),
           "RESULTS": str(tmp_path / "res")}
    r = subprocess.run(["bash", str(ROOT / "scripts/train/make_data_v1.sh")], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    cmds = [c for c in r.stdout.splitlines() if c.startswith("+ ")]
    for step in ("collect", "build", "label", "manifest"):
        assert any(f"jevharness.training.data_v1 {step}" in c for c in cmds), step
    assert any("--seed0 30000" in c for c in cmds)


def test_12_sft_v1_dry_run_trains_evaluates_and_continues_with_opd(tmp_path: Path):
    env = {**os.environ, "DRY_RUN": "1", "HW": "l4", "OUT": str(tmp_path / "r3v1"), "EVAL_DIR": str(tmp_path / "ev")}
    r = subprocess.run(["bash", str(ROOT / "scripts/train/12_sft_v1.sh")], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = r.stdout
    assert "jevharness.training.train_decision" in out and "--batch-size 2" in out
    assert "data/train_v0/dev.jsonl" in out and "data/train_v1/dev.jsonl" in out      # gate set + v1 dev
    assert "jevharness.training.opd" in out and "--seeds 40000" in out and "--eval-seed0 1000" in out
    assert "--eval-seed0 5000" not in out and "eval_seed0: 5000" not in out      # never the calibration seeds


BUILT = ROOT / "data/train_v1"


@pytest.mark.skipif(not (BUILT / "dev.jsonl.gz").exists(), reason="data/train_v1 not built")
def test_built_v1_respects_every_hold_out_rule():
    gate, e1 = v1.gate_dev_tasks(), v1.e1_dev_prefix_ids()
    names = ("train.jsonl.gz", "dev.jsonl.gz", "dev_recovery.jsonl.gz", "dev_knockout.jsonl.gz",
             "replay_when2call.jsonl.gz")
    rows = {p: read_jsonl(BUILT / p) for p in names if (BUILT / p).exists()}
    for p, rs in rows.items():
        v1.assert_clean(rs, gate_tasks=gate, e1_ids=e1)
        for r in rs:
            assert 2 <= len(r["options"]) <= 52 and abs(sum(r["target"].values()) - 1) < 1e-6, (p, r["id"])
    dev_tasks = {v1.task_of(r["id"]) for r in rows["dev.jsonl.gz"]}
    if "train.jsonl.gz" in rows:
        assert not {v1.task_of(r["id"]) for r in rows["train.jsonl.gz"]} & dev_tasks
    man = json.loads((BUILT / "MANIFEST.json").read_text())
    for name, f in man["files"].items():   # the manifest describes the files on disk
        if "rows" in f:
            assert f["rows"] == len(read_jsonl(BUILT / name)), name


def test_committed_v1_manifest_seed_ranges_are_training_seeds():
    p = ROOT / "results/training/train_v1/manifest.json"
    if not p.exists():
        pytest.skip("v1 manifest not written yet")
    man = json.loads(p.read_text())
    for name, f in man["files"].items():
        rng = f.get("jevgym_seed_range")
        if rng:
            assert v1.seed_ok(rng[0]) and v1.seed_ok(rng[1]), name
            assert not any(lo <= rng[0] <= hi or lo <= rng[1] <= hi or (rng[0] < lo and rng[1] > hi)
                           for lo, hi in v1.FORBIDDEN_RANGES), name


# ------------------------------------------------------------------ go / no-go
def _ev(path: str, acc: float, by_dp: dict[str, float], ece: float = 0.05) -> dict:
    return {"files": {path: {"overall": {"n": 100, "acc": acc, "ece": ece},
                             "by_dp": {k: {"n": 30, "acc": v} for k, v in by_dp.items()}}}}


def test_gonogo_needs_the_static_bar_and_the_in_harness_margin(tmp_path: Path):
    gv = json.loads((ROOT / "results/gpu_round2/gate_verdict.json").read_text())
    jev = {"metrics": {"overall": {"acc": 0.958}, "dp:tool_select": {"acc": 0.916}, "dp:tool_params": {"acc": 0.979},
                       "dp:stop": {"acc": 0.987}}}
    intern = {"overall": {"acc": 0.80, "ece": 0.08}, "by_dp": {"tool_select": {"acc": 0.72}, "tool_params": {"acc": 0.85},
                                                               "stop": {"acc": 0.88}}}
    good = {"files": {**_ev("data/train_v0/dev.jsonl", 0.90, {})["files"],
                      **_ev("data/train_v1/dev.jsonl.gz", 0.86, {"tool_select": 0.82, "tool_params": 0.90,
                                                                 "stop": 0.90})["files"]}}
    res = v1.gonogo(good, jev_v1=jev, intern_v1=intern, gate_verdict=gv)
    assert res["go"], res["checks"]
    assert res["checks"]["G1_min_gate"]["threshold"] == round(0.8 * gv["jev_acc"], 4)
    # passes the static gate but is no better than Intern on the harness's states -> NO-GO (the round-2 pattern)
    weak = json.loads(json.dumps(good))
    weak["files"]["data/train_v1/dev.jsonl.gz"]["by_dp"]["tool_select"]["acc"] = 0.74
    res = v1.gonogo(weak, jev_v1=jev, intern_v1=intern, gate_verdict=gv)
    assert not res["go"] and not res["checks"]["G3_in_harness_vs_intern"]["pass"] and res["checks"]["G1_min_gate"]["pass"]
    # without Intern's v1-dev run the round-2 shadow accuracies are the (flagged) proxy
    res = v1.gonogo(good, jev_v1=jev, intern_v1=None, gate_verdict=gv)
    assert res["references"]["intern_v1_dev"]["source"].startswith("proxy") and res["go"]
    # CLI: several eval files merge
    a, b = tmp_path / "g.json", tmp_path / "v.json"
    a.write_text(json.dumps(_ev("data/train_v0/dev.jsonl", 0.90, {})))
    b.write_text(json.dumps(_ev("data/train_v1/dev.jsonl.gz", 0.86, {"tool_select": 0.82, "tool_params": 0.9,
                                                                       "stop": 0.9})))
    j = tmp_path / "jev.json"
    j.write_text(json.dumps(jev))
    v1.main(["gonogo", "--eval", str(a), str(b), "--jev-v1", str(j), "--out", str(tmp_path / "out.json")])
    assert json.loads((tmp_path / "out.json").read_text())["go"]


def test_opd_refuses_reserved_training_pools_and_evaluation_dev_seeds(capsys: pytest.CaptureFixture):
    pytest.importorskip("torch")
    from jevharness.training.opd import parse_args

    base = ["--student", "tiny", "--teacher", "sim", "--executor", "sim"]
    for seeds, n in ((99990, 20), (199800, 512), (200000, 16)):
        with pytest.raises(SystemExit):
            parse_args(base + ["--seeds", str(seeds), "--n-tasks", str(n)])
    with pytest.raises(SystemExit):
        parse_args(base + ["--eval-every", "1", "--eval-seed0", "200000", "--eval-n", "8"])
    a = parse_args(base + ["--seeds", "40000", "--n-tasks", "512", "--eval-every", "5"])
    assert a.eval_seed0 == 1000                                   # default: the training-side dev range
    parse_args(base + ["--eval-every", "5", "--eval-seed0", "5000"])   # calibration range: warned, not refused
    assert "calibration range" in capsys.readouterr().err
