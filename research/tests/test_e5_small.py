"""Exploratory row E5, small decision models (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06): the box helpers
(scripts/train/e5_box.py), the offline threshold-fit replay (scripts/e5_cal_replay.py), the analysis
(scripts/e5_analysis.py) and the driver's dry run (scripts/aws/e5_small.sh). Offline: no GPU, no API, no network."""
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import e5_analysis as EA  # noqa: E402
import e5_box as E  # noqa: E402
import e5_cal_replay as CR  # noqa: E402

DRIVER = ROOT / "scripts" / "aws" / "e5_small.sh"


# ------------------------------------------------------------------ e5_box
def test_models_are_the_registered_revisions_and_the_swap_applies(tmp_path):
    reg = (ROOT / "docs" / "PREREG_A7_STAGE2.md").read_text()
    assert "Exploratory row E5" in reg
    for slot, m in E.MODELS.items():
        assert m["repo"] in reg and m["revision"] in reg, slot          # every pinned revision is the registered one
    assert E.ORDER == ["e5_q17", "e5_q06", "e5_mc1"] and E.BACKUP == {"e5_mc1": "e5_gr1"}
    assert E.slots(tmp_path) == E.ORDER
    (tmp_path / "swap.json").write_text(json.dumps({"swap": {"e5_mc1": "e5_gr1"}}))
    assert E.slots(tmp_path) == ["e5_q17", "e5_q06", "e5_gr1"]
    assert E.expected_mix_sha().startswith("9fe1088d") and E.rows_with_replay() == 16902


def test_cap_projection_priors_then_measured(tmp_path):
    run = tmp_path
    (run / "host.json").write_text(json.dumps(E.host_profile("g5.8xlarge", "NVIDIA A10G", 14, "<instance-id>")))
    est = {s: E.estimate_min(run, s) for s in E.ORDER}
    assert est["e5_q17"] > est["e5_mc1"] > est["e5_q06"] > 0
    assert 150 < est["e5_q17"] < 260                      # the plan's 1.7B estimate without checkpointing (~3.6 h)
    (run / "mem").mkdir()
    (run / "mem" / "e5_q17.txt").write_text("gc_3072")
    assert E.estimate_min(run, "e5_q17") > est["e5_q17"]   # checkpointing costs the prior's 30 %
    ok, msg = E.cap_check(run, "e5_q17", 14.0)
    assert ok and "reserved 1.5 h" in msg
    assert not E.cap_check(run, "e5_q17", 4.0)[0]
    # after a finished model: measured minutes per row and per parameter ratio
    rec = E.record_setting(run, "e5_q17", "nogc_3072", "nogc_3072:ok", 120, "g5.8xlarge", None, "0" * 64, "9000")
    assert rec["train_mix_ok"] is False and rec["peak_mem_mib"] == 9000.0 and rec["repo"] == "Qwen/Qwen3-1.7B"
    E.record_eval_time(run, "e5_q17", 30)
    q06 = E.estimate_min(run, "e5_q06")
    assert q06 == pytest.approx((120 - 5) / E.ratio("e5_q17") * E.ratio("e5_q06") + 5
                                + max(15, 30 / E.ratio("e5_q17") * E.ratio("e5_q06")), rel=1e-6)
    assert E.record_setting(run, "e5_q06", "gc_3072", "nogc_3072:oom,gc_3072:ok", 50, "", None,
                            E.expected_mix_sha())["train_mix_ok"] is True


def test_cost_answers_peak_spend_and_bad_loss(tmp_path):
    run = tmp_path
    (run / "s23").mkdir()
    (run / "cal").mkdir()
    (run / "s23" / "answers_e5_e5_q17.jsonl").write_text(json.dumps({"qhash": "a", "cost_usd": 0.0}) + "\n")
    (run / "cal" / "answers_e5_q17.jsonl").write_text(json.dumps({"key": "k", "cost_usd": 0.0}) + "\n")
    (run / "s23" / "replay_e5_e5_q17.json").write_text(json.dumps({"backend": {"base_url": "http://127.0.0.1:8082"}}))
    assert E.cost_check(run)["ok"] and E.cost_check(run)["records"] == 2
    (run / "cal" / "answers_e5_q06.jsonl").write_text(json.dumps({"key": "k", "cost_usd": 0.001}) + "\n")
    assert not E.cost_check(run)["ok"]
    with gzip.open(run / "s23" / "questions.jsonl.gz", "wt") as f:
        f.write(json.dumps({"qhash": "a"}) + "\n" + json.dumps({"qhash": "b"}) + "\n")
    assert not E.answers_complete(run, "e5_q17")["ok"]
    (run / "smi.csv").write_text("2026/10/06 10:00:00.000, 50, 9000\n2026/10/06 10:00:00.500, 90, 12345\n")
    assert E.peak_mib(run / "smi.csv") == 12345
    (run / "smi2.csv").write_text("8000, 23028\n11000, 23028\n")
    assert E.peak_mib(run / "smi2.csv") == 11000
    live = run / "live.jsonl"
    live.write_text("\n".join(json.dumps({"task_id": t, "usage": {"total": {"cost_usd": c}}})
                              for t, c in (("a", 0.04), ("a", 0.05), ("b", 0.01))) + "\nnot json\n")
    assert E.spend([live, run / "missing.jsonl"]) == {"records": 3, "usd": 0.1}   # every attempt counts
    assert E.bad_loss("{'loss': nan, 'grad_norm': 3.0}") and not E.bad_loss("{'loss': 0.41}")


# ------------------------------------------------------------------ e5_cal_replay
def _shadow_run(d: Path) -> Path:
    """A tiny calibration shadow arm: two tasks, tool_select + tool_params + stop decisions."""
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for k, tid in enumerate(("jevgym-x-5000-d60-near-strict", "jevgym-x-5001-d60-near-strict")):
        steps = []
        for s in range(3):
            steps.append({"step": s, "tool": "find_user", "decisions": [
                {"dp": "tool_select", "mode": "shadow", "question": "next tool", "options": ["find_user", "ask_user",
                                                                                            "finish"],
                 "s1_value": "find_user", "s1_conf": 0.9, "s1_dist": {"find_user": 0.9, "ask_user": 0.05, "finish": 0.05},
                 "deferred": True, "s2_value": "find_user", "final_value": "find_user", "gold": "find_user",
                 "gold_status": "ok", "gold_aux": {"v": 1, "acceptable_tools": ["find_user"]}, "state": f"S{k}{s}",
                 "shadow": True, "s1_cost_usd": 0.0, "s2_cost_usd": 0.005, "s2_correct": True},
                {"dp": "tool_params", "mode": "shadow", "question": "Which value should `name` take? ()",
                 "options": ["Kofi", "Ama", "<OTHER: none of these / needs generation>"], "s1_value": "Kofi",
                 "s1_conf": 0.99, "s1_dist": {"Kofi": 0.99, "Ama": 0.005, "<OTHER: none of these / needs generation>": 0.005},
                 "deferred": True, "s2_value": "Kofi", "final_value": "Kofi", "gold": "Kofi", "gold_status": "ok",
                 "state": f"S{k}{s}", "shadow": True, "s1_cost_usd": 0.0, "s2_cost_usd": 0.005, "s2_correct": True},
                {"dp": "stop", "mode": "shadow", "question": "done?", "options": None, "s1_value": False, "s1_conf": 0.9,
                 "s1_dist": {"yes": 0.1, "no": 0.9}, "state": f"S{k}{s}", "shadow": True}]})
        recs.append({"task_id": tid, "config": "rl_shadow_cal", "steps": steps, "result": {"success": True},
                     "calls": [], "error": None})
    (d / "rl_shadow_cal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "experiment.yaml").write_text("name: tiny\nsuite: {name: jevgym, n: 2, seed0: 5000}\narms:\n"
                                       "  - {name: rl_shadow_cal, tool_select: {mode: shadow}}\n")
    return d


def test_cal_replay_questions_answer_check_derive_and_fit(tmp_path):
    run = _shadow_run(tmp_path / "run")
    recs = CR.load_latest(run / "rl_shadow_cal.jsonl")
    qs = list(CR.iter_questions(recs, lambda tid: (["find_user", "ask_user", "finish"], ["d1", "d2", "d3"])))
    assert len(qs) == 12 and {q["dp"] for q in qs} == {"tool_select", "tool_params"}
    ts = [q for q in qs if q["dp"] == "tool_select"][0]
    assert ts["text"] == "Which tool should the agent call next?" and ts["descriptions"] == ["d1", "d2", "d3"]
    with pytest.raises(ValueError):          # the regenerated tools must be the logged options
        list(CR.iter_questions(recs, lambda tid: (["other"], ["x"])))
    req = tmp_path / "req.jsonl.gz"
    with gzip.open(req, "wt") as f:
        f.write("".join(json.dumps(q) + "\n" for q in qs))
    # the harness's own asking path, with an offline backend; resumable; $0
    from jevharness.decision import make_backend

    ans = tmp_path / "answers.jsonl"
    r1 = CR.answer(req, "", "sim", ans, backend=make_backend({"type": "simulated", "accuracy": 1.0, "seed": 0}))
    assert r1["answered_now"] == 12 and r1["errors_now"] == 0 and r1["cost_usd"] == 0
    assert CR.answer(req, "", "sim", ans, backend=make_backend({"type": "simulated"}))["answered_now"] == 0
    got = [json.loads(x) for x in ans.read_text().splitlines()]
    assert all(a["value"] in q["options"] for a, q in zip(got, qs))
    # the logged answers reproduce themselves: check passes; derive replaces the S1 fields and drops other dps
    logged = tmp_path / "logged.jsonl"
    logged.write_text("".join(json.dumps({"key": q["key"], "value": q["logged"]["value"], "conf": 0.5,
                                          "dist": {q["logged"]["value"]: 0.5}, "error": None, "latency_ms": 1.0})
                              + "\n" for q in qs))
    chk = CR.check(req, logged)
    assert chk["ok"] and chk["by_dp"]["tool_select"]["agreement"] == 1.0
    n = CR.derive(run, "rl_shadow_cal", logged, tmp_path / "cal", "e5cal_t", model="t")
    assert n == {"replaced": 12, "missing": 0, "dropped_other_dps": 6}
    d = [json.loads(x) for x in (tmp_path / "cal" / "e5cal_t.jsonl").read_text().splitlines()]
    dec = [x for r in d for s in r["steps"] for x in s["decisions"]]
    assert {x["dp"] for x in dec} == {"tool_select", "tool_params"} and all(x["s1_conf"] == 0.5 for x in dec)
    assert d[0]["config"] == "e5cal_t" and d[0]["meta"]["e5_cal_replay"]["model"] == "t"
    fits = CR.fit(tmp_path / "cal", "e5cal_t", tmp_path / "fits")
    assert set(fits) == {"tool_select", "tool_params"} and fits["tool_select"]["n_paired"] == 6
    assert fits["tool_select"]["tau_rule"] == "fixed"          # A6.5: fewer than 10 paired decisions -> tau 0.9


def test_committed_inputs_record():
    rec = json.loads((ROOT / "results" / "explore" / "e5_small" / "inputs.json").read_text())
    assert rec["training_file"]["sha256"].startswith("65fa28a1")
    assert rec["training_file"]["train_mix_sha256"].startswith("9fe1088d")
    assert rec["cal_requests"]["questions"] == {"tool_select": 1125, "tool_params": 1594}
    src = ROOT / "runs" / "e2a_jevgym_shadow_cal_rl" / "rl_shadow_cal.jsonl"
    if src.exists():                                           # local only (runs/ is not committed)
        assert CR._sha256(src) == rec["cal_requests"]["source_sha256"]


# ------------------------------------------------------------------ e5_analysis
def _fake_run(run: Path) -> None:
    """An E5 run whose model e5_q17 answers C exactly as the published SFT v1 row (and v1 dev likewise)."""
    for d in ("eval", "s23", "settings", "latency/bench", "done", "cal"):
        (run / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "results" / "gpu_round5" / "eval" / "eval_sft_v1_v1.json", run / "eval" / "e5_q17_v1.json")
    shutil.copy(ROOT / "results" / "stage2" / "s23" / "answers_sft_v1.jsonl", run / "s23" / "answers_e5_e5_q17.jsonl")
    shutil.copy(ROOT / "results" / "stage2" / "s23" / "answers_base.jsonl", run / "s23" / "answers_e5_e5_q06.jsonl")
    (run / "done" / "e5_q17").touch()
    (run / "host.json").write_text(json.dumps(E.host_profile("g5.8xlarge", "NVIDIA A10G", 14, "<instance-id>")))
    (run / "settings" / "e5_q17.json").write_text(json.dumps({"setting": "nogc_3072", "sft_min": 180,
                                                               "peak_mem_mib": 15000.0, "train_mix_ok": True}))

    def bench(label: str, c: int, ms: float) -> None:
        rows = [{"dp": "tool_select" if i % 2 else "tool_params", "ms": ms * (1 + i / 100)} for i in range(100)]
        steps = [{"s1_ms": 2 * ms} for _ in range(50)]
        (run / "latency" / "bench" / f"bench_{label}_new_c{c}.json").write_text(json.dumps(
            {"concurrency": c, "episodes": 25, "requests": rows, "steps": steps, "span_s": 10.0}))

    for c in (1, 4):
        bench("4b", c, 400.0 * c)
        bench("e5_q17", c, 160.0 * c)
    (run / "latency" / "e5_q17_synthetic.json").write_text(json.dumps(
        {"results": {"c=1": {"p50_ms": 100, "p90_ms": 120, "p95_ms": 130, "req_per_s": 9}}}))
    (run / "latency" / "smi_e5_q17_c4.csv").write_text("t, 80, 7000\nt, 90, 7400\n")


def test_analysis_reproduces_the_published_rows_and_applies_the_rules(tmp_path):
    run = tmp_path / "run"
    _fake_run(run)
    d = EA.analyze(run, None)
    q17 = d["models"]["e5_q17"]
    assert round(100 * q17["c_acc"], 1) == 99.2 and round(100 * q17["c_acc_ask"], 1) == 100.0     # = SFT v1 (coverage.md)
    assert q17["v1_acc"] == pytest.approx(0.99706) and q17["train_peak_mib"] == 15000.0
    assert round(100 * d["references"]["rl"]["c_acc"], 1) == 99.2
    assert round(100 * d["models"]["e5_q06"]["c_acc_ask"], 1) == 47.3                             # = base Qwen3-4B
    c = d["contrasts"]["q17_vs_sftv1"]
    assert c["all"]["delta"] == 0 and c["all"]["reading"] == "keeps the 4B's accuracy"
    assert d["contrasts"]["q06_vs_sftv1"]["all"]["reading"] == "loses accuracy"
    lat = d["latency"]
    assert lat["e5_q17"]["speedup_vs_4b"]["c1_p50"] == pytest.approx(2.5)
    assert lat["e5_q17"]["serve_peak_mib"] == 7400 and lat["e5_q17"]["synthetic"]["c=1"]["p90_ms"] == 120
    lc = d["live_choice"]
    assert lc["run"] and lc["slot"] == "e5_q17" and "e5_q06" not in lc["eligible"]
    keys = EA.tex_keys(d)
    assert keys["e5/q17/c_acc"] == "99.2" and keys["e5/q17/speedup_c1_p50"] == "2.50"
    assert keys["e5/contrast/q17_vs_rl/c_all_delta"] == "+0.0"
    EA.write(d, tmp_path / "out", tmp_path / "e5.tex")
    assert "\\defres{e5/q17/c_acc}{99.2}" in (tmp_path / "e5.tex").read_text()
    assert "keeps the 4B's accuracy" in (tmp_path / "out" / "summary.md").read_text()


def test_live_choice_rule():
    lat = {"a": {"bench": {"c1": {"p50_ms": 200.0}}}, "b": {"bench": {"c1": {"p50_ms": 90.0}}}}
    m = {"a": {"c_acc": 0.990}, "b": {"c_acc": 0.982}, "c": {"c_acc": 0.94}}
    assert EA.live_choice(m, lat)["slot"] == "b"                     # within 1.0 point -> the faster one
    m["b"]["c_acc"] = 0.975
    assert EA.live_choice(m, lat)["slot"] == "a"                     # 1.5 points behind -> the more accurate one
    assert not EA.live_choice({"c": {"c_acc": 0.94}}, lat)["run"]


def test_live_analysis_three_arms(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    for arm, succ, cost, wall in (("llm_only", 1, 0.04, 8.0), ("harness_final", 1, 0.006, 9.0), ("e5_final", 1, 0.008, 6.0)):
        recs = [{"task_id": f"t{i}", "result": {"success": bool(succ) if i else False},
                 "usage": {"total": {"cost_usd": cost, "latency_ms": 1000 * wall / 2}}, "wall_ms": 1000 * wall,
                 "calls": [{"component": "s1", "latency_ms": 100.0}], "steps": [], "error": None} for i in range(10)]
        (live / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    res = EA.live(live, 3.6)
    assert res["n_pairs"] == 10 and res["arms"]["e5_final"]["success"] == 0.9
    assert res["q_usd_per_episode"]["e5_final"] == pytest.approx(3.6 * 6.0 / 3600)
    c = res["contrasts"]["e5_final_vs_llm_only"]
    assert c["success_delta_pt"] == 0 and c["wall_delta_s"] == pytest.approx(-2.0)
    assert c["cost_saving_frac"] == pytest.approx(1 - (0.008 + 0.006) / 0.04)


# ------------------------------------------------------------------ the driver
def _env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_API_KEYS"))}
    env.update({"W": str(tmp_path / "w"), "E5_RUN": str(tmp_path / "run"), "DRY_RUN": "1", "PY": sys.executable,
                "DRY_ITYPE": "g5.8xlarge", "DRY_GPU": "NVIDIA A10G", "OPENROUTER_API_KEY": "dummy-not-a-key"})
    return env


def test_driver_syntax_refusals_and_offline_dry_run(tmp_path):
    assert subprocess.run(["bash", "-n", str(DRIVER)]).returncode == 0
    env = _env(tmp_path)
    for iid in ("", "<instance-id>", "<instance-id>", "<instance-id>", "not-an-id"):
        r = subprocess.run(["bash", str(DRIVER), "send", "rearm"], env={**env, "E5_IID": iid}, capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 1 and ("refused" in r.stdout or "E5_IID" in r.stdout), (iid, r.stdout)
    r = subprocess.run(["bash", str(DRIVER), "box", "rearm e5_inputs e5_smoke e5_arms e5_latency e5_cal4b e5_final"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    run = tmp_path / "run"
    status = (run / "status.txt").read_text()
    assert "stopping the chain" not in status and "box done" in status
    for s in E.ORDER:
        assert f"stage sft_{s} exit=0" in status and f"model {s} done" in status
    assert "cap: go model e5_q17" in status and "latency: session done" in status
    assert (run / "keys_in_env.txt").read_text() == ""                     # the preamble unset the key
    dry = (run / "dry_run.log").read_text()
    assert "train_ablation" not in dry                                      # never E3's S3 prefix
    assert "s3://<bucket>/e5_small" in dry and "checkpoints/e5_small/e5_q17" in dry
    trainer = [x for x in dry.splitlines() if "train_decision" in x and "ck_e5/e5_q06 " in x]
    assert trainer and all(f in trainer[0] for f in ("--lora-r 32", "--lora-alpha 64", "--lr 1e-4", "--epochs 2",
                                                     "--seed 0", "--max-batch-tokens 3072", "--eff-batch 64",
                                                     "--no-gradient-checkpointing", "--loss soft_ce_brier",
                                                     "--max-steps -1", "hf/e5_q06"))
    assert "D100.jsonl:1 data/train_v1/replay_when2call.jsonl.gz:0.45" in dry
    for s in E.ORDER:
        m = E.MODELS[s]
        assert f"snapshot_download\\({m['repo']}\\,\\ revision={m['revision']}\\)" in dry
    assert "--name e5-e5_mc1 --port 8082 --fast" in dry and "--name jevagentrl-4b --port 8081 --fast" in dry
    assert "--label 4b --path new --concurrency 4" in dry and "--label 4b_end --path new --concurrency 1 --limit 25" in dry
    assert "e5_cal_replay.py answer" in dry and "sudo shutdown -h +2" in dry
    assert json.loads((run / "host.json").read_text())["gpu_class"] == "A10G"
    assert json.loads((run / "final_check.json").read_text())["cost_check_ok"] is True


def test_driver_live_dry_run(tmp_path):
    env = _env(tmp_path)
    r = subprocess.run(["bash", str(DRIVER), "box", "rearm live_serve live_smoke live live_final"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    status = (tmp_path / "run" / "status.txt").read_text()
    assert "live pass 3 exit=0" in status and "stopping the chain" not in status
    dry = (tmp_path / "run" / "dry_run.log").read_text()
    assert "env JEV_EVAL_RUNS=1" in dry and "configs/experiments/e5_small_live.yaml --out runs -o workers=1" in dry
    assert "configs/dev/e5_small_smoke.yaml --out runs -o workers=1" in dry
    assert "--name jevagentrl-4b --port 8081 --fast --cache-tokens 16384" in dry
    assert "--name e5-e5_mc1 --port 8082 --fast --cache-tokens 16384" in dry
    assert (tmp_path / "run" / "keys_in_env.txt").read_text() == ""


def test_live_configs_open_only_the_registered_seeds():
    import yaml

    live = yaml.safe_load((ROOT / "configs" / "experiments" / "e5_small_live.yaml").read_text())
    s = live["suite"]
    assert (s["seed0"], s["n"], live["n_tasks"]) == (300700, 100, 100) and live["workers"] == 1
    assert {k: s[k] for k in ("families", "distractor_style", "n_distractors", "noise_lines", "strict")} == \
        {"families": "v2", "distractor_style": "near", "n_distractors": 60, "noise_lines": 20, "strict": True}
    assert [a["name"] for a in live["arms"]] == ["llm_only", "harness_final", "e5_final"]
    # Deviations 2026-10-06 ~18:00 UTC: Gemini 3.5 Flash through OpenRouter for every arm (direct API overloaded)
    assert live["llm"] == {"provider": "openrouter", "model": "google/gemini-3.5-flash", "temperature": 0,
                           "reasoning_effort": "minimal"}
    smoke = yaml.safe_load((ROOT / "configs" / "dev" / "e5_small_smoke.yaml").read_text())
    assert smoke["suite"]["seed0"] == 900 and smoke["arms"] == live["arms"]
    prof = yaml.safe_load((ROOT / "configs" / "harness" / "explore_e5_final.yaml").read_text())
    final = yaml.safe_load((ROOT / "configs" / "harness" / "jevagent_final.yaml").read_text())
    sec, fsec = prof["system2"]["gemini-3.5-flash"], final["system2"]["gemini-3.5-flash"]
    assert sec["decision_points"] == fsec["decision_points"]
    assert sec["backends"]["jevagent_rl"] == fsec["backends"]["jevagent_rl"]
    assert "e5_small" in sec["backends"]


def test_prepare_live_fills_the_profile_and_the_configs(tmp_path):
    import yaml

    prof = tmp_path / "explore_e5_final.yaml"
    shutil.copy(ROOT / "configs" / "harness" / "explore_e5_final.yaml", prof)
    cfgs = []
    for c in ("configs/experiments/e5_small_live.yaml", "configs/dev/e5_small_smoke.yaml"):
        shutil.copy(ROOT / c, tmp_path / Path(c).name)
        cfgs.append(tmp_path / Path(c).name)
    fits = tmp_path / "fits"
    fits.mkdir()
    for dp, tau in (("tool_select", 0.97), ("tool_params", float("nan"))):
        (fits / f"tauhat_e5cal_e5_q06_{dp}.json").write_text(json.dumps({"tau_hat": tau, "primary": {"tau_hat": tau}}))
    d = {"live_choice": {"run": True, "slot": "e5_q06"}, "cal_check_4b": {"ok": False}}
    with pytest.raises(SystemExit):                       # the 4B replay check must pass first
        EA.prepare_live(d, fits, prof, cfgs)
    d["cal_check_4b"]["ok"] = True
    res = EA.prepare_live(d, fits, prof, cfgs)
    assert res["slot"] == "e5_q06" and res["taus"]["tool_select"] == {"tau": 0.97}
    assert res["taus"]["tool_params"] == {"tau": 0.99, "always_defer": True} and not res["skip"]
    blk = yaml.safe_load(prof.read_text())["system2"]["gemini-3.5-flash"]["backends"]
    assert blk["e5_small"]["tool_select"] == {"tau": 0.97} and blk["e5_small"]["fitted_from"]["model"] == "e5-e5_q06"
    assert blk["jevagent_rl"]["tool_select"]["tau"] == 0.4999999999978404          # the frozen block unchanged
    live = yaml.safe_load(cfgs[0].read_text())
    assert live["s1_backends"]["e5_small"]["name"] == live["s1_backends"]["e5_small"]["model"] == "e5-e5_q06"
    assert "calibration: pending" not in prof.read_text()
    with pytest.raises(SystemExit):
        EA.prepare_live({"live_choice": {"run": False, "rule": "none"}}, fits, prof, cfgs)


def test_compact_bench_keeps_what_the_analysis_reads(tmp_path):
    run = tmp_path / "run"
    _fake_run(run)
    full = EA.bench_stats(run / "latency" / "bench" / "bench_e5_q17_new_c4.json")
    assert E.compact_bench(run / "latency" / "bench", run / "latency" / "bench_compact") == 4
    for p in (run / "latency" / "bench").glob("*.json"):
        p.unlink()
    assert EA.bench_stats(run / "latency" / "bench" / "bench_e5_q17_new_c4.json") == full


def test_driver_part_host_dry_run_and_analysis_merge(tmp_path):
    env = {**_env(tmp_path), "E5_PART": "b"}
    r = subprocess.run(["bash", str(DRIVER), "box", "rearm e5_inputs e5_smoke:e5_q06 e5_arm:e5_q06 e5_final"], cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    run = tmp_path / "run"
    status = (run / "status.txt").read_text()
    assert "smoke e5_q06" in status and "smoke e5_q17" not in status and "smoke e5_mc1" not in status
    assert "model e5_q06 done" in status and "sft_e5_q17" not in status and "sft_e5_mc1" not in status
    dry = (run / "dry_run.log").read_text()
    assert "s3://<bucket>/e5_small/part_b --exclude" in dry and "train_ablation" not in dry
    assert "s3://<bucket>/e5_small/inputs/cal_requests.jsonl.gz" in dry     # one copy of the inputs
    assert "checkpoints/e5_small/e5_q06" in dry
    r = subprocess.run(["bash", str(DRIVER), "send", "rearm"], env={**env, "E5_PART": "B!"}, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 1
    # the analysis merges a part's per-model files into the main run and sums GPU hours over hosts
    main, part = tmp_path / "main", tmp_path / "part"
    _fake_run(main)
    (part / "eval").mkdir(parents=True)
    (part / "done").mkdir()
    shutil.copy(main / "eval" / "e5_q17_v1.json", part / "eval" / "e5_mc1_v1.json")
    (part / "done" / "e5_mc1").touch()
    (part / "stage_times.jsonl").write_text(json.dumps({"stage": "x", "min": 60}) + "\n")
    (main / "stage_times.jsonl").write_text(json.dumps({"stage": "y", "min": 30}) + "\n")
    d = EA.analyze(main, None, parts=[part])
    assert d["models"]["e5_mc1"]["v1_acc"] == pytest.approx(0.99706) and d["gpu_hours"] == pytest.approx(1.5)
    (part / "eval" / "e5_q17_v1.json").write_text("{}")
    (part / "done" / "e5_q17").touch()
    with pytest.raises(SystemExit):                  # one model's records come from one host
        EA.analyze(main, None, parts=[part])


def test_reading_rule_reports_both_clauses():
    assert EA.reading({"lo": -0.009, "hi": -0.004}).startswith("keeps the 4B's accuracy within 1 point (small loss")
    assert EA.reading({"lo": -0.005, "hi": 0.002}) == "keeps the 4B's accuracy"
    assert EA.reading({"lo": -0.03, "hi": -0.01}) == "loses accuracy"
    assert EA.reading({"lo": -0.03, "hi": 0.01}) == "inconclusive"


def test_live_route_health_guard(tmp_path):
    f = tmp_path / "e5_final.jsonl"
    ok = {"task_id": "a", "error": None, "calls": [{"component": "s2", "attempts": 1}, {"component": "s1", "attempts": 3}]}
    f.write_text(json.dumps(ok) + "\n")
    assert E.health([f, tmp_path / "missing.jsonl"]) == {"tasks": 1, "errored": 0, "errored_route": 0, "errored_s1": 0,
                                                         "errored_other": 0, "s2_calls": 1, "s2_retried": 0}
    bad = {"task_id": "b", "error": "InternalServerError", "calls": [{"component": "s2", "attempts": 2},
                                                                     {"component": "s2", "retry_wait_ms": 500.0}]}
    f.write_text(json.dumps(ok) + "\n" + json.dumps(bad) + "\n")
    assert E.health([f]) == {"tasks": 2, "errored": 1, "errored_route": 1, "errored_s1": 0, "errored_other": 0,
                             "s2_calls": 3, "s2_retried": 2}
    assert E.main(["health", str(f)]) == 4 and E.main(["health", str(f), "--max-errors", "1"]) == 0
    s1 = {"task_id": "c", "error": "JevAPIError: Jev API error 500: Internal Server Error\nTraceback ...", "calls": []}
    f.write_text(json.dumps(ok) + "\n" + json.dumps(s1) + "\n")
    assert E.health([f])["errored_s1"] == 1 and E.main(["health", str(f)]) == 4
    assert E.main(["health", str(f), "--max-s1-errors", "1"]) == 0
    assert E.error_kind("InternalServerError: Error code: 503") == "route" and E.error_kind("KeyError: x") == "other"
    live = (ROOT / "configs" / "experiments" / "e5_small_live.yaml").read_text()
    assert "provider: openrouter, model: google/gemini-3.5-flash" in live
