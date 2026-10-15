"""The mechanical JevAgentRL-v2 chain (scripts/train/r6_chain.py): base rule, keep / retain / freeze with the stress
instrument, on synthetic development artefacts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r6_chain", ROOT / "scripts/train/r6_chain.py")
r6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r6)  # type: ignore[union-attr]


def _eval(d: Path, tag: str, acc: float, rec: float, ece: float, go: bool = True) -> None:
    ev = d / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    files = {"data/train_v1/dev.jsonl.gz": {"overall": {"acc": acc, "ece": ece},
                                            "by_dp": {"tool_select": {"acc": acc}}},
             "data/train_v1/dev_recovery.jsonl.gz": {"overall": {"acc": rec}}}
    (ev / f"eval_{tag}_v1.json").write_text(json.dumps({"files": files}))
    (ev / f"eval_{tag}_gate.json").write_text(json.dumps({"files": {"data/train_v0/dev.jsonl": {"overall": {"acc": 1.0}}}}))
    (ev / f"gonogo_{tag}.json").write_text(json.dumps({"go": go, "checks": {"G1": {"pass": go}},
                                                       "references": {"intern_v1_dev": {"source": "v1 dev"}}}))


def _stress(d: Path, run: str, S: float, A: float, R: float, arm: str = r6.STRESS_ARM) -> None:
    (d / "stress").mkdir(parents=True, exist_ok=True)
    (d / "stress" / f"{run}_summary.json").write_text(json.dumps(
        {f"{run}::{arm}": {"success": S, "tool_select/replace/s1_acc": A, "reward_ccps": R, "n": 200}}))


def test_opd_kept_on_stress_gain_and_grpo_dropped_without_reward_gain(tmp_path: Path):
    _eval(tmp_path, "sft_v2", 0.997, 0.998, 0.03)
    _stress(tmp_path, "stress_sft_v2", 0.60, 0.80, 0.40)
    _eval(tmp_path, "opd_i12", 0.995, 0.995, 0.03)            # standard gates hold (within the keep margins)
    _stress(tmp_path, "stress_opd_i12", 0.64, 0.81, 0.42)      # S +0.04 >= +0.03 -> gain
    _eval(tmp_path, "opd", 0.996, 0.996, 0.03)
    _stress(tmp_path, "stress_opd", 0.66, 0.82, 0.43)          # higher C -> retained
    _eval(tmp_path, "grpo_i15", 0.996, 0.996, 0.03)
    _stress(tmp_path, "stress_grpo_i15", 0.66, 0.82, 0.435)    # R +0.005: not > +0.01
    _eval(tmp_path, "grpo", 0.990, 0.990, 0.03)
    _stress(tmp_path, "stress_grpo", 0.70, 0.84, 0.50)         # passes everything ...
    _stress(tmp_path, "stress_refs_intern", 0.30, 0.60, 0.20, r6.INTERN_STRESS_ARM)
    res = r6.decide(tmp_path, validity_ok=True, replay_ok=False)   # ... but the replay check failed
    assert res["stages"]["opd"]["retained"] == "opd" and res["stages"]["opd"]["kept"]
    assert not res["stages"]["grpo"]["kept"] and res["grpo_init"] == "opd"
    assert res["frozen"] == "opd" and res["go"] and not res["rl_claim"]
    res = r6.decide(tmp_path, validity_ok=True, replay_ok=True)
    assert res["stages"]["grpo"]["retained"] == "grpo" and res["frozen"] == "grpo" and res["rl_claim"]


def test_opd_dropped_when_stress_degrades_and_no_go_without_intern_reference(tmp_path: Path):
    _eval(tmp_path, "sft_v2", 0.997, 0.998, 0.03)
    _stress(tmp_path, "stress_sft_v2", 0.60, 0.80, 0.40)
    _eval(tmp_path, "opd", 0.999, 0.999, 0.03)                 # standard gain ...
    _stress(tmp_path, "stress_opd", 0.55, 0.80, 0.38)          # ... but S -0.05 < -0.015
    res = r6.decide(tmp_path, validity_ok=True, replay_ok=True)
    assert not res["stages"]["opd"]["kept"] and res["kept"] == ["sft_v2"]
    assert res["frozen"] == "sft_v2" and not res["go"]           # G6 needs the Intern stress reference: missing -> fail
    _stress(tmp_path, "stress_refs_intern", 0.30, 0.60, 0.20, r6.INTERN_STRESS_ARM)
    assert r6.decide(tmp_path, True, True)["go"]


def test_base_rule(tmp_path: Path):
    out = tmp_path
    (out / "smoke_env.txt").write_text("gpu table\n" + json.dumps(
        {"kernels": {"causal_conv1d": "ok (1, 64, 32)", "fla_gated_delta_rule": "ok (1, 64, 2, 64)"}}))
    for name in ("Qwen3-4B", "StartLux-Decision-4B"):
        (out / f"smoke_engine_{name}.json").write_text(json.dumps({"max_abs_dlogp": 0.03, "pass_s2": True}))
        (out / f"smoke_train_{name}.log").write_text("step 0: {'train/loss': 1.23, 'train/acc': 0.5}\n"
                                                     "step 1: {'train/loss': 1.10}\n")
    _eval(out, "zeroshot_qwen", 0.40, 0.3, 0.2)
    _eval(out, "zeroshot_startlux", 0.80, 0.7, 0.1)
    res = r6.base_rule(out, Path("/m"))
    assert res["base"] == "startlux" and res["path"] == "/m/StartLux-Decision-4B"
    _eval(out, "zeroshot_startlux", 0.70, 0.7, 0.1)            # below 0.766 -> Qwen3-4B
    assert r6.base_rule(out, Path("/m"))["base"] == "qwen"
    _eval(out, "zeroshot_startlux", 0.80, 0.7, 0.1)
    (out / "smoke_train_StartLux-Decision-4B.log").write_text("step 0: {'train/loss': nan}\n")
    assert r6.base_rule(out, Path("/m"))["base"] == "qwen"


def test_validity_detection_reads_the_exit_line_and_quiet_pytest_output():
    quiet = "....... [100%]\n"            # addopts -q plus -q: progress only, no summary line
    assert r6.validity_passed(quiet)
    assert r6.validity_passed("..s.... [100%]\n7 passed, 1 skipped in 3.2s\n")
    assert not r6.validity_passed("..F.... [100%]\n1 failed, 6 passed in 3.2s\n")
    assert not r6.validity_passed("..F.... [100%]\n")
    assert not r6.validity_passed("")
    assert r6.validity_passed("anything\npytest_exit=0\n")
    assert not r6.validity_passed("7 passed in 3.2s\npytest_exit=1\n")
