"""The exploratory SFT training ablation tools (results/training/ablation_plan.md and its addendum of 2026-10-05): the
arm builder (scripts/train/ablation_subsets.py: nested stratified task subsets, gold / teacher relabelling, the
size-matched state arms at 100 % and inside D25, the training-seed guard, the core / extended cap units), the offline
scorer (scripts/train/ablation_score.py: v1-dev read-outs, the common-state scoring of scripts/stage2_analysis.py
reproduced on the published answers, paired task-bootstrap contrasts), the box-side checks of the GPU driver
(scripts/train/ablation_box.py: host priors, memory fallbacks, the cap projection, the anchor go / no-go, the $0 check)
and the driver itself (scripts/aws/train_ablation.sh: syntax and a DRY_RUN chain for one arm)."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import ablation_box as BX  # noqa: E402
import ablation_score as SC  # noqa: E402
import ablation_subsets as AB  # noqa: E402

DRIVER = ROOT / "scripts" / "aws" / "train_ablation.sh"


def _row(task: str, i: int, *, arm: str = "lite_shadow", after: int = 0, seed: int = 30001, src: str = "teacher"):
    opts = ["a", "b", "c"]
    meta = {"arm": arm, "after_error": after, "jevgym_seed": seed, "label_source": src}
    if src == "teacher":
        meta.update({"gold": "b", "alpha": 0.3, "teacher_dists": {"jev": [0.1, 0.8, 0.1]}})
        target = {"a": 0.07, "b": 0.86, "c": 0.07}
    else:
        target = {"a": 0.0, "b": 1.0, "c": 0.0}
    return {"id": f"{task}/{i}/tool_select", "dp": "tool_select", "options": opts, "target": target, "meta": meta}


def _rows():
    rows = []
    for t in range(40):
        for i in range(5):
            rows.append(_row(f"jevgym-x-{30000 + t}-d60", i, after=int(i >= 3), seed=30000 + t))
    for t in range(10):
        rows.append(_row(f"jevgym-v0-{10000 + t}-d30", 0, arm="v0_flash_shadow", seed=10000 + t, src="gold"))
    return rows


def test_task_subsets_nested_and_stratified():
    rows = _rows()
    s10, s25, s50 = (AB.task_subset(rows, f) for f in (0.10, 0.25, 0.50))
    assert s10 <= s25 <= s50                                         # nested
    v0 = {AB.task_of(r) for r in rows if AB.block_of(r) == "v0"}
    assert len(s25 & v0) == round(0.25 * 10) and len(s25 - v0) == 40 // 4   # stratified by block
    assert AB.task_subset(rows, 0.25) == s25                          # deterministic


def test_arm_rows_labels_states_and_match():
    rows = _rows()
    gold = AB.arm_rows("L_gold_100", rows)
    assert all(r["target"] == {"a": 0.0, "b": 1.0, "c": 0.0} and r["meta"]["label_source"] == "gold" for r in gold)
    assert "teacher_dists" not in gold[0]["meta"]
    canon, match = AB.arm_rows("S_canon", rows), AB.arm_rows("S_match", rows)
    assert len(canon) == len(match) == sum(not AB.is_recovery(r) for r in rows)
    assert not any(AB.is_recovery(r) for r in canon) and any(AB.is_recovery(r) for r in match)
    with pytest.raises(SystemExit):                                   # teacher arms need the alpha-0 relabel
        AB.arm_rows("L_teach_100", rows)
    teach = {r["id"]: {**r, "target": {"a": 0.1, "b": 0.8, "c": 0.1}, "meta": {**r["meta"], "alpha": 0.0}}
             for r in rows}
    out = AB.arm_rows("L_teach_25", rows, teach)
    assert out and all(r["meta"]["ablation_labels"] == "teacher" and r["target"]["b"] == 0.8 for r in out)
    with pytest.raises(SystemExit):
        AB.arm_rows("L_teach_100", rows, {k: {**v, "meta": {**v["meta"], "alpha": 0.3}} for k, v in teach.items()})


def test_state_pair_inside_d25_nested_and_size_matched():
    rows = _rows()
    ids = lambda rs: [r["id"] for r in rs]  # noqa: E731
    d25, canon = set(ids(AB.arm_rows("D25", rows))), set(ids(AB.arm_rows("S_canon", rows)))
    c25, m25 = AB.arm_rows("S_canon_25", rows), AB.arm_rows("S_match_25", rows)
    assert len(c25) == len(m25) == sum(not AB.is_recovery(r) for r in AB.arm_rows("D25", rows))
    assert set(ids(c25)) <= d25 and set(ids(c25)) <= canon and set(ids(m25)) <= d25   # nested
    assert not any(AB.is_recovery(r) for r in c25) and any(AB.is_recovery(r) for r in m25)
    assert ids(AB.arm_rows("S_match_25", rows)) == ids(m25)                            # deterministic
    assert ids(m25) == sorted(ids(m25), key=ids(rows).index)                          # the source order is kept


def test_units_and_seed_replicates():
    assert AB.data_arm("D25_s1") == "D25" and AB.trainer_seed("D25_s1") == 1
    assert AB.data_arm("S_canon_25") == "S_canon_25" and AB.trainer_seed("L_gold_25") == 0
    assert AB.units("D25,S_canon+S_match") == [["D25"], ["S_canon", "S_match"]]
    core = [a for u in AB.units(AB.CORE) for a in u]
    ext = [a for u in AB.units(AB.EXTENDED) for a in u]
    assert core == ["D25", "L_gold_25", "L_teach_25", "S_canon_25", "S_match_25", "D10", "D50"]
    assert sorted(ext) == sorted(["D100", "L_gold_100", "L_teach_100", "S_canon", "S_match", "D10_s1", "D25_s1"])
    assert ["S_canon_25", "S_match_25"] in AB.units(AB.CORE) and ["S_canon", "S_match"] in AB.units(AB.EXTENDED)
    tier1 = {a for a, v in AB.ARMS.items() if v[3] == 1}
    assert {AB.data_arm(a) for a in core + ext} == tier1                              # core + extended = tier 1
    with pytest.raises(SystemExit):
        AB.units("D25,nope")


def test_seed_guard_and_build(tmp_path):
    rows = _rows()
    with pytest.raises(ValueError):
        AB.check_seeds(rows + [_row("jevgym-x-300001-d60", 0, seed=300001)])
    man = AB.build(rows, tmp_path / "d", ["D25", "S_canon", "L_teach_100"], None)
    assert man["arms"]["L_teach_100"] == {"skipped": "no --teacher-only file"}
    assert man["arms"]["D25"]["rows"] == len((tmp_path / "d" / "D25.jsonl").read_text().splitlines())
    assert man["arms"]["S_canon"]["recovery_rows"] == 0


@pytest.mark.skipif(not AB.TRAIN.exists(), reason="data/train_v1 not present")
def test_plan_on_data_v1():
    p = AB.plan(AB.load(AB.TRAIN))
    assert p["rows"] == 15694 and p["D100 (= SFT v1, existing checkpoint)"]["rows_with_replay"] == 16902
    a = p["arms"]
    assert a["S_canon"]["rows"] == a["S_match"]["rows"] == 10851 and a["S_canon"]["recovery_rows"] == 0
    assert a["D10"]["tasks"] < a["D25"]["tasks"] < a["D50"]["tasks"] < 587
    assert a["L_gold_25"]["rows"] == a["L_teach_25"]["rows"] == a["D25"]["rows"]
    assert a["S_canon_25"]["rows"] == a["S_match_25"]["rows"] == 2438 and a["S_canon_25"]["recovery_rows"] == 0
    assert a["S_canon_25"]["rows"] == a["D25"]["rows"] - a["D25"]["recovery_rows"]


@pytest.mark.skipif(not AB.TRAIN.exists(), reason="data/train_v1 not present")
def test_committed_inputs_record_matches_a_fresh_build(tmp_path):
    """results/training/ablation_inputs.json (the box's abl_inputs check) holds the sha256 of every arm file; the arms
    that need no teacher file are rebuilt here from the committed data and must match byte for byte."""
    rec = json.loads(BX.INPUTS.read_text())
    assert rec["teacher_only"]["calls_fresh"] == 0 and rec["teacher_only"]["outcomes"] == {"kept": 15694}
    assert rec["train_mix"]["arms"]["D100"]["sha256"] == rec["train_mix"]["sft_v1_round5_train_mix_sha256"]
    man = AB.build(AB.load(AB.TRAIN), tmp_path / "d", ["D10", "L_gold_25", "S_canon_25", "S_match_25"], None)
    for arm in ("D10", "L_gold_25", "S_canon_25", "S_match_25"):
        assert man["arms"][arm]["sha256"] == rec["arms"][arm]["sha256"], arm
        assert man["arms"][arm]["rows_with_replay"] == rec["train_mix"]["arms"][arm]["rows"]
    assert man["source_sha256"] == rec["train_v1_sha256"]


@pytest.mark.skipif(not (SC.S23 / "answers_sft_v1.jsonl").exists(), reason="common-state corpus not present")
def test_scorer_reproduces_published_common_state_rows(tmp_path):
    (tmp_path / "eval").mkdir()
    d = SC.score(tmp_path)
    ref = json.loads((ROOT / "results" / "stage2" / "stage2.json").read_text())["descriptive"]["stage2_rl"]
    for name, row in (("pub_sft_v1", "sft_v1"), ("pub_frozen", "frozen"), ("pub_jev", "jev"), ("pub_base", "base")):
        got, want = d["arms"][name], ref[row]
        for k, w in (("c_acc", "corpus_acc"), ("c_acc_offplan", "corpus_acc_offplan"), ("c_ece", "ece")):
            assert got[k] == pytest.approx(want[w], abs=1e-12), (name, k)


def test_scorer_synthetic_run(tmp_path):
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    occ = []
    for t in range(4):
        for q in range(3):
            occ.append({"task": f"t{t}", "qhash": f"q{t}{q}", "gold": "g", "gold_status": "ok", "dp": "tool_select",
                        "gold_aux": {"acceptable_tools": ["g"]}, "stratum": "after_off_plan" if q == 2 else "on_plan"})
    with gzip.open(cdir / "occurrences.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(o) + "\n" for o in occ))
    (cdir / "corpus.json").write_text(json.dumps({"eligible_tasks": [f"t{t}" for t in range(4)]}))
    run = tmp_path / "run"
    (run / "s23").mkdir(parents=True)
    (run / "eval").mkdir()
    for arm, wrong in (("sft_v1", set()), ("S_canon", {"q02", "q12"}), ("S_match", set()), ("S_canon_25", {"q02"}),
                       ("S_match_25", set())):
        (run / "s23" / f"answers_abl_{arm}.jsonl").write_text("".join(
            json.dumps({"qhash": o["qhash"], "answer": "x" if o["qhash"] in wrong else "g", "confidence": 0.9,
                        "error": None, "valid": True}) + "\n" for o in occ))
    ev = {"files": {SC.V1["dev"]: {"overall": {"n": 10, "acc": 0.9, "ece": 0.02, "nll": 0.1},
                                   "by_dp": {"tool_select": {"acc": 0.8}}},
                    SC.V1["rec"]: {"overall": {"n": 4, "acc": 0.75}, "by_dp": {"tool_select": {"acc": 0.5}}}}}
    (run / "eval" / "S_canon_v1.json").write_text(json.dumps(ev))
    (run / "eval" / "S_match_v1.json").write_text(json.dumps({**ev, "files": {**ev["files"], SC.V1["dev"]: {
        **ev["files"][SC.V1["dev"]], "overall": {"n": 10, "acc": 1.0, "ece": 0.01, "nll": 0.0}}}}))
    (run / "status.txt").write_text("x stage sft_S_canon exit=0 (96 min)\nx stage sft_S_canon exit=1 (3 min)\n")
    (run / "settings").mkdir()
    (run / "settings" / "S_canon.json").write_text(json.dumps({"setting": "gc_3072", "instance_type": "g5.4xlarge"}))
    d = SC.score(run, cdir)
    a = d["arms"]["S_canon"]
    assert a["c_acc"] == pytest.approx(10 / 12) and a["c_acc_offplan"] == pytest.approx(0.5)
    assert a["v1_onplan"] == pytest.approx((9 - 3) / 6) and a["sft_min"] == 96
    c = d["contrasts"]["canon_vs_match"]
    assert c["v1_acc_diff"] == pytest.approx(-0.1) and c["c_acc"]["delta"] == pytest.approx(10 / 12 - 1)
    assert c["c_acc"]["lo"] <= c["c_acc"]["delta"] <= c["c_acc"]["hi"]
    keys = SC.tex_keys(d)
    assert keys["recipe_abl/S_canon/c_acc"] == "83.3" and keys["recipe_abl/contrast/canon_vs_match/v1_acc_diff"] == "-10.0"
    assert "| canon_vs_match |" in SC.render_md(d) and "| gc_3072 |" in SC.render_md(d)
    assert a["train_setting"] == "gc_3072" and a["instance_type"] == "g5.4xlarge"
    c25 = d["contrasts"]["canon_vs_match_25"]
    assert (c25["arm"], c25["reference"]) == ("S_canon_25", "S_match_25")
    assert c25["c_acc"]["delta"] == pytest.approx(11 / 12 - 1)
    assert "recipe_abl/contrast/canon_vs_match_25/c_acc_delta" in keys
    assert {"size_10_vs_50", "size_25_vs_anchor"} <= set(SC.CONTRASTS) and SC.CONTRASTS["size_25_vs_anchor"] == ("D25", "sft_v1")


# ------------------------------------------------------------------ box-side checks (scripts/train/ablation_box.py)
def test_host_priors_settings_and_oom():
    a10 = BX.host_profile("g5.4xlarge", "NVIDIA A10G", 24)
    l40 = BX.host_profile("g6e.2xlarge", "NVIDIA L40S", 24)
    assert a10["gpu_class"] == "A10G" and a10["usd_per_h"] == 1.624 and a10["priors"]["temp_rel_tol"] == 2e-2
    assert l40["gpu_class"] == "L40S" and l40["usd_per_h"] == 2.24208 and l40["priors"]["temp_rel_tol"] == 1e-3
    assert BX.host_profile("x9.large", "", 24)["priors"] == BX.PRIORS["A10G"]           # unknown: the slower priors
    assert BX.setting_flags("nogc_3072") == "--max-batch-tokens 3072 --eff-batch 64 --no-gradient-checkpointing"
    assert BX.setting_flags("gc_1536").endswith("--gradient-checkpointing") and "--eff-batch 64" in BX.setting_flags("gc_1536")
    order, s = [], "nogc_3072"
    while s != "none":
        order.append(s)
        s = BX.next_setting(s)
    assert order == ["nogc_3072", "gc_3072", "gc_2048", "gc_1536", "gc_1024"]          # checkpointing first, then budget
    assert BX.is_oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert not BX.is_oom("ValueError: something else")


def test_smoke_rows_take_the_longest_prompts():
    d100 = [{"id": f"t/{i}", "state": "x" * i, "question": "q", "options": ["a", "b"]} for i in range(200)]
    d10 = d100[:50]
    rows = BX.smoke_rows(d100, d10, n_long=5, n_rand=10)
    assert [r["id"] for r in rows[:5]] == [f"t/{i}" for i in range(195, 200)] and len(rows) == 15
    assert rows == BX.smoke_rows(d100, d10, n_long=5, n_rand=10)


def _run_dir(tmp_path, itype="g5.4xlarge", gpu="NVIDIA A10G"):
    run = tmp_path / "run"
    for d in ("eval", "cal", "s23", "done"):
        (run / d).mkdir(parents=True, exist_ok=True)
    (run / "host.json").write_text(json.dumps(BX.host_profile(itype, gpu, 24)))
    return run


def test_cap_projection_priors_then_measured(tmp_path):
    run = _run_dir(tmp_path)
    rows = BX.rows_with_replay("D25")
    assert rows == 4126 and BX.rows_with_replay("D25_s1") == 4126 and BX.rows_with_replay("sft_v1") == 0
    est = BX.estimate_min(run, "D25")
    assert est == pytest.approx(BX.SFT_MIN_PER_ROW_L40S * 2.9 * rows + 5 + 52)       # A10G priors, fast server
    (run / "mem_setting.txt").write_text("gc_3072\n")
    assert BX.estimate_min(run, "D25") == pytest.approx(BX.SFT_MIN_PER_ROW_L40S * 2.9 * 1.3 * rows + 5 + 52)
    (run / "arm_times.jsonl").write_text(json.dumps({"arm": "D10", "rows": 1752, "sft_min": 105.0}) + "\n"
                                         + json.dumps({"arm": "D10", "eval_min": 40.0}) + "\n")
    assert BX.estimate_min(run, "D25") == pytest.approx(100 / 1752 * rows + 5 + 40)    # measured rate and eval
    (run / "stage_times.jsonl").write_text(json.dumps({"stage": "rearm", "min": 1300.0}) + "\n")
    ok, msg = BX.cap_check(run, ["D25"], 24)
    assert not ok and "vs cap 24 h" in msg
    (run / "done" / "D25").touch()
    assert BX.cap_check(run, ["D25"], 24)[0]                                           # nothing left to run
    assert BX.used_hours(run) == pytest.approx(1300 / 60)


def test_anchor_check_tolerances_and_fallback_code(tmp_path):
    run = _run_dir(tmp_path)
    shutil.copy(BX.ANCHOR_EVAL, run / "eval" / "sft_v1_v1.json")
    cal = json.loads(BX.ANCHOR_CAL.read_text())
    cal["global"] *= 1.015                                                             # 1.5 % off: A10G ok, L40S not
    (run / "cal" / "sft_v1_calibration.json").write_text(json.dumps(cal))
    pub = [json.loads(x) for x in BX.ANCHOR_ANSWERS.read_text().splitlines() if x.strip()]
    (run / "s23" / "answers_abl_sft_v1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pub))
    res = BX.anchor_check(run, "fast")
    assert res["ok"] and res["c_answers"]["agreement"] == 1.0 and res["v1_dev"]["ok"]
    assert not BX.anchor_check(run, "fast", BX.host_profile("g6e.2xlarge", "NVIDIA L40S", 24))["temperatures"]["ok"]
    bad = [dict(r, answer="__other__") if i % 50 == 0 else r for i, r in enumerate(pub)]   # 2 % of the answers differ
    (run / "s23" / "answers_abl_sft_v1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad))
    assert BX.main(["anchor", "--run", str(run), "--serve-mode", "fast"]) == 2              # only C: the plain retry
    (run / "eval" / "sft_v1_v1.json").write_text("{\"files\": {}}")
    assert BX.main(["anchor", "--run", str(run), "--serve-mode", "fast"]) == 1


def test_cost_check_and_record(tmp_path):
    run = _run_dir(tmp_path)
    ok = {"qhash": "a", "answer": "x", "cost_usd": 0.0}
    (run / "s23" / "answers_abl_D10.jsonl").write_text(json.dumps(ok) + "\n")
    (run / "s23" / "replay_abl_D10.json").write_text(json.dumps({"backend": {"base_url": "http://127.0.0.1:8082"}}))
    assert BX.cost_check(run)["ok"]
    (run / "s23" / "answers_abl_D50.jsonl").write_text(json.dumps({**ok, "cost_usd": 0.0001}) + "\n")
    assert not BX.cost_check(run)["ok"]
    (run / "s23" / "answers_abl_D50.jsonl").unlink()
    (run / "s23" / "replay_abl_D50.json").write_text(json.dumps({"backend": {"base_url": "https://openrouter.ai/api"}}))
    assert BX.cost_check(run)["nonlocal_backends"] == ["replay_abl_D50.json"]
    with gzip.open(run / "s23" / "questions.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps({"qhash": q}) + "\n" for q in ("a", "b")))
    assert not BX.answers_complete(run, "D10")["ok"]                                   # b missing: replay resumes
    with open(run / "s23" / "answers_abl_D10.jsonl", "a") as f:
        f.write(json.dumps({**ok, "qhash": "b", "answer": None, "error": "Timeout"}) + "\n")
    assert BX.answers_complete(run, "D10") == {"questions": 2, "answered": 2, "errors": 1, "ok": True}
    log = tmp_path / "sft.log"
    log.write_text("[train_decision] token budget 3072: 1650 micro-batches/epoch, 2.50 rows each on average, "
                   "grad-accum 26 (~65 rows/step), mean length 900 tokens, padding 0.050\n")
    want = json.loads(BX.INPUTS.read_text())["train_mix"]["arms"]["D10_s1"]["sha256"]
    rec = BX.record_setting(run, "D10_s1", "gc_3072", "nogc_3072:oom,gc_3072:ok", 61, "g5.4xlarge", log, want)
    assert rec["trainer_seed"] == 1 and rec["data"] == "D10" and rec["gradient_checkpointing"] and rec["train_mix_ok"]
    assert rec["grad_accum"] == 26 and rec["rows_per_step"] == 65.0 and rec["eff_batch_rows"] == 64
    assert not BX.record_setting(run, "D10", "nogc_3072", "nogc_3072:ok", 50, "", None, "0" * 64)["train_mix_ok"]


# ------------------------------------------------------------------ the driver (scripts/aws/train_ablation.sh)
def test_driver_syntax_and_dry_run_one_arm(tmp_path):
    assert subprocess.run(["bash", "-n", str(DRIVER)]).returncode == 0
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_API_KEYS"))}
    env.update({"W": str(tmp_path / "w"), "ABL_RUN": str(tmp_path / "run"), "DRY_RUN": "1", "PY": sys.executable,
                "DRY_ITYPE": "g6e.2xlarge", "DRY_GPU": "NVIDIA L40S", "OPENROUTER_API_KEY": "dummy-not-a-key"})
    r = subprocess.run(["bash", str(DRIVER), "box", "rearm abl_inputs abl_anchor abl_smoke abl_arms:D10_s1 abl_final"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    run = tmp_path / "run"
    status = (run / "status.txt").read_text()
    assert "stopping the chain" not in status and "box done" in status
    assert "stage sft_D10_s1 exit=0" in status and "cap: go unit D10_s1" in status
    assert (run / "keys_in_env.txt").read_text() == ""                     # the preamble unset the key
    dry = (run / "dry_run.log").read_text()
    trainer = [x for x in dry.splitlines() if "train_decision" in x and "ck_abl/D10_s1" in x]
    assert trainer and all(f in trainer[0] for f in ("--lora-r 32", "--lora-alpha 64", "--lr 1e-4", "--epochs 2",
                                                     "--seed 1", "--max-batch-tokens 3072", "--eff-batch 64",
                                                     "--no-gradient-checkpointing", "--loss soft_ce_brier"))
    assert "replay_when2call.jsonl.gz:0.0467" in dry and "--max-steps 4" in dry      # the arm's replay share; smoke
    assert "--name abl-D10_s1 --port 8082 --fast" in dry and "checkpoints/train_ablation/D10_s1" in dry
    assert "sudo shutdown -h +2" in dry and json.loads((run / "host.json").read_text())["gpu_class"] == "L40S"
    s = json.loads((run / "settings" / "D10_s1.json").read_text())
    assert s["setting"] == "nogc_3072" and s["trainer_seed"] == 1 and s["instance_type"] == "g6e.2xlarge"
