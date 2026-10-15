"""Exploratory rows E17 and E18 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E17 and E18: read-out ablation and a
deferral-aware objective"): the deferral option and head in the Clef engine (training/clef_engine.py), the target
and the logistic head (training/defer_head.py), the E18-F records (training/clef_data.py, scripts/train/e18_defer.py),
the E18 profile and live configs, the analysis script and the driver. Offline: a stand-in model, no GPU, no API."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from jevharness.decision.jev import answer_from_wire
from jevharness.decision.types import QKind, Question
from jevharness.harness.prompts import Q_TOOL_SELECT
from jevharness.training import clef_data as CD
from jevharness.training import defer_head as DH
from jevharness.training.clef_engine import DEFER_LABEL, ClefEngine, DeferOption, Q_TOOL_SELECT_TEXT, head_features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import e18_defer as E18D  # noqa: E402
import explore_e17_e18 as X  # noqa: E402

HEAD = ROOT / "results/explore/e17_e18/e18_head.json"
PROFILE = ROOT / "configs/harness/explore_e18_final.yaml"


# ------------------------------------------------------------------ the engine with a stand-in model
class _Q:
    def __init__(self, qid, option_ids):
        self.question_id, self.option_ids = qid, tuple(option_ids)


class _Rec:
    def __init__(self, state, questions):
        self.input_ids = list(range(10 + len(str(state))))
        self.questions = [_Q(qid, ["true", "false"] if q["type"] == "noul" else sorted(str(k) for k in q["criteria"]))
                          for qid, q in questions.items()]


def _engine(logit_of, defer=None):
    def encode(state, questions):
        return _Rec(state, questions)

    def forward(records):
        return [[logit_of(o) for o in q.option_ids] for r in records for q in r.questions]

    def scales(records, questions, logits):
        return [1.0] * len(logits)

    return ClefEngine(encode, forward, scales, tokenizer=None, name="e18-test", defer=defer)


def test_option_mode_scores_the_defer_criterion_and_reports_one_minus_its_mass():
    eng = _engine(lambda o: {"a": 2.0, "b": 1.0, DEFER_LABEL: 1.5}.get(o, 0.0), DeferOption(mode="option"))
    qs = {"q0": {"type": "choice", "instructions": Q_TOOL_SELECT, "criteria": {"a": "A", "b": "B"}},
          "q1": {"type": "noul", "instructions": "stop?"}}
    ans, usage = eng.answer_all("s", qs)
    z = {k: math.exp(v) for k, v in {"a": 2.0, "b": 1.0, DEFER_LABEL: 1.5}.items()}
    tot = sum(z.values())
    pd = z[DEFER_LABEL] / tot
    a0 = ans["q0"]
    assert a0["choice"] == "a" and a0["confidence"] == pytest.approx(1 - pd) and a0["defer"] == pytest.approx(pd)
    assert set(a0["probabilities"]) == {"a", "b"} and sum(a0["probabilities"].values()) == pytest.approx(1.0)
    assert a0["probabilities"]["a"] == pytest.approx(z["a"] / (z["a"] + z["b"])) and a0["p_best"] == pytest.approx(z["a"] / tot)
    assert ans["q1"]["type"] == "noul"                      # noul questions carry no option
    assert eng.describe()["defer"] == {"mode": "option", "label": DEFER_LABEL, "weights": None}
    # the harness keeps the served confidence (the cascade compares it with tau = 1 - k) and the real options
    q = Question(kind=QKind.CHOICE, text=Q_TOOL_SELECT, options=["a", "b"])
    got = answer_from_wire(q, a0)
    assert got.value == "a" and got.confidence == pytest.approx(1 - pd) and set(got.distribution) == {"a", "b"}
    # without the rule the same engine answers as E14
    plain = _engine(lambda o: {"a": 2.0, "b": 1.0}.get(o, 0.0))
    p0 = plain.answer_all("s", qs)[0]["q0"]
    assert "defer" not in p0 and p0["confidence"] == pytest.approx(math.exp(2) / (math.exp(2) + math.exp(1)))


def test_head_mode_applies_the_logistic_head_to_the_served_top_probability():
    w = [3.4, 1.29, 0.33, -1.08]
    eng = _engine(lambda o: {"a": 6.0, "b": 0.0, "c": 0.0}.get(o, 0.0), DeferOption(mode="head", weights=w))
    crit = {"a": None, "b": None, "c": None}
    a0 = eng.answer_all("s", {"q0": {"type": "choice", "instructions": Q_TOOL_SELECT, "criteria": crit}})[0]["q0"]
    p = {k: math.exp(v) for k, v in {"a": 6.0, "b": 0.0, "c": 0.0}.items()}
    pa = p["a"] / sum(p.values())
    f = head_features(pa, True, 3)
    want = 1 / (1 + math.exp(-sum(x * y for x, y in zip(w, f))))
    assert a0["defer"] == pytest.approx(want) and a0["confidence"] == pytest.approx(1 - want)
    assert a0["probabilities"]["a"] == pytest.approx(pa) and a0["p_best"] == pytest.approx(pa)
    arg = eng.answer_all("s", {"q0": {"type": "choice", "instructions": "Which value should the argument x take?",
                                      "criteria": crit}})[0]["q0"]
    f2 = head_features(pa, False, 3)
    assert arg["defer"] == pytest.approx(1 / (1 + math.exp(-sum(x * y for x, y in zip(w, f2)))))
    assert Q_TOOL_SELECT_TEXT == Q_TOOL_SELECT
    with pytest.raises(ValueError):
        DeferOption(mode="head", weights=[1.0])
    with pytest.raises(ValueError):
        DeferOption(mode="other")


def test_head_features_clip_the_log_gap():
    assert head_features(1.0, True, 69) == [1.0, -15.0, 1.0, pytest.approx(math.log(69))]
    assert head_features(0.9, False, 5)[1] == pytest.approx(-1.0)
    assert head_features(1 - 1e-20, True, 1)[1] == -15.0


# ------------------------------------------------------------------ the target and the logistic head
def test_defer_target_rule_and_the_fit_on_synthetic_states(tmp_path):
    assert DH.defer_target(True, False, 1.0, 0.005) == 1
    assert DH.defer_target(True, True, 1.0, 0.005) == 0
    assert DH.defer_target(True, True, 0.001, 0.005) == 1          # margin below the cost margin
    assert DH.defer_target(False, False, 1.0, 0.005) == 0          # the LLM is wrong: no gain from deferring
    rng = np.random.default_rng(0)
    gaps = rng.uniform(-8, 0, 400)
    F = np.array([head_features(1 - 10 ** g, True, 20) for g in gaps])
    d = (gaps > -2).astype(float) * (rng.uniform(size=400) < 0.9)
    fit = DH.fit_logistic(F, d, lam=1.0)
    assert fit["positives"] == int(d.sum()) and fit["weights"][1] > 0            # a larger gap -> more deferral
    p_hi = DH.p_defer(fit["weights"], head_features(1 - 10 ** -0.5, True, 20))
    p_lo = DH.p_defer(fit["weights"], head_features(1 - 10 ** -7, True, 20))
    assert p_hi > 0.5 > p_lo
    # labelled questions from a tiny shadow arm and reference answers
    step = {"step": 0, "decisions": [
        {"dp": "tool_select", "mode": "shadow", "question": Q_TOOL_SELECT, "options": ["ask_user", "find_user", "finish"],
         "s1_value": "find_user", "s1_conf": 0.9, "s1_dist": {"ask_user": 0.05, "find_user": 0.9, "finish": 0.05},
         "s2_value": "ask_user", "gold": "ask_user", "gold_status": "ok", "gold_aux": {"acceptable_tools": ["ask_user", "find_user"]},
         "state": "S"},
        {"dp": "tool_params", "mode": "shadow", "question": "Which value should the argument `name` take?", "options": ["A", "B", "<OTHER>"],
         "s1_value": "A", "s1_conf": 0.99, "s1_dist": {"A": 0.99, "B": 0.005, "<OTHER>": 0.005}, "s2_value": "B", "gold": "B",
         "gold_status": "ok", "state": "S"},
        {"dp": "tool_params", "mode": "shadow", "question": "q", "options": ["A", "B"], "s1_value": "A", "s2_value": "A",
         "gold": None, "state": "S"}]}
    arm = tmp_path / "arm.jsonl"
    arm.write_text(json.dumps({"task_id": "t1", "steps": [step]}) + "\n")
    ans = tmp_path / "ans.jsonl"
    ans.write_text("\n".join(json.dumps(x) for x in [
        {"key": "t1|0|0", "dp": "tool_select", "value": "finish", "conf": 0.6, "dist": {"ask_user": 0.3, "find_user": 0.1, "finish": 0.6}, "error": None},
        {"key": "t1|0|1", "dp": "tool_params", "value": "A", "conf": 0.999999, "dist": {"A": 0.999999, "B": 1e-6, "<OTHER>": 0.0}, "error": None}]) + "\n")
    rows = DH.labelled_questions(arm, ans)
    assert [r["key"] for r in rows] == ["t1|0|0", "t1|0|1"]      # the unlabelled argument is left out
    assert rows[0]["llm_ok"] and not rows[0]["ref_ok"] and rows[0]["d"] == 1
    assert rows[1]["llm_ok"] and not rows[1]["ref_ok"] and rows[1]["d"] == 1
    assert rows[0]["ref_margin"] == pytest.approx(0.3) and rows[0]["n_options"] == 3
    h = DH.fit_head(rows)
    assert set(h["by_dp"]) == {"tool_select", "tool_params"} and h["k"] == DH.K_MARGIN


# ------------------------------------------------------------------ E18-F records
def test_add_defer_option_and_the_calibration_state_records():
    rec = {"id": "t/1/tool_params", "state": "S", "questions": {
        "q0": {"type": "choice", "instructions": "arg a", "criteria": {"x": None, "y": None}},
        "q1": {"type": "choice", "instructions": "arg b", "criteria": {"u": None, "v": None}},
        "q2": {"type": "noul", "instructions": "stop?"}},
        "gold": {"q0": {"label": "x", "probabilities": {"x": 1.0, "y": 0.0}}, "q1": {"label": "v", "probabilities": {"u": 0.0, "v": 1.0}},
                 "q2": {"label": "false", "probabilities": {"true": 0.0, "false": 1.0}}}, "meta": {"dp": "tool_params", "row_ids": ["a", "b"]}}
    out = CD.add_defer_option(rec, ["q1"])
    assert out["questions"]["q0"]["criteria"][DEFER_LABEL] == CD.DEFER_DESC and out["gold"]["q0"]["probabilities"][DEFER_LABEL] == 0.0
    assert out["gold"]["q1"] == {"label": DEFER_LABEL, "probabilities": {"u": 0.0, "v": 0.0, DEFER_LABEL: 1.0}}
    assert DEFER_LABEL not in json.dumps(out["questions"]["q2"]) and rec["questions"]["q0"]["criteria"] == {"x": None, "y": None}
    with pytest.raises(ValueError):
        CD.add_defer_option(rec, ["q2"])
    with pytest.raises(ValueError):
        CD.add_defer_option(out)
    wire = CD.wire_request(out)
    assert DEFER_LABEL in wire["questions"]["q0"]["criteria"] and "gold" not in json.dumps(wire)
    rows = [{"key": "t|3|0", "task_id": "t", "dp": "tool_select", "n_options": 3, "gold": "ask_user", "llm_ok": True, "ref_ok": False, "d": 1},
            {"key": "t|3|1", "task_id": "t", "dp": "tool_params", "n_options": 2, "gold": "B", "llm_ok": True, "ref_ok": True, "d": 0},
            {"key": "t|3|2", "task_id": "t", "dp": "tool_params", "n_options": 2, "gold": "Q", "llm_ok": True, "ref_ok": False, "d": 1}]
    reqs = {"t|3|0": {"state": "S", "options": ["find_user", "ask_user", "finish"], "descriptions": ["Find.", "Ask.", "Done."], "text": Q_TOOL_SELECT},
            "t|3|1": {"state": "S", "options": ["A", "B"], "text": "arg one"}, "t|3|2": {"state": "S", "options": ["P", "Q"], "text": "arg two"}}
    crows = E18D.cal_rows(rows, reqs)
    assert crows[0]["options"] == ["find_user -- Find.", "ask_user -- Ask.", "finish -- Done."] and crows[0]["target"]["ask_user -- Ask."] == 1.0
    assert crows[0]["meta"]["defer"] == 1 and crows[1]["id"] == "t/3/tool_params" and crows[2]["id"] == "t/3/tool_params#1"
    recs = E18D._records_with_option(crows)
    assert len(recs) == 2                                        # the two arguments of the step share a record
    ts = next(r for r in recs if r["meta"]["dp"] == "tool_select")
    assert ts["gold"]["q0"]["label"] == DEFER_LABEL and ts["questions"]["q0"]["criteria"]["ask_user"] == "Ask."
    tp = next(r for r in recs if r["meta"]["dp"] == "tool_params")
    assert tp["gold"]["q0"]["label"] == "B" and tp["gold"]["q1"]["label"] == DEFER_LABEL and tp["meta"]["defer_qids"] == ["q1"]
    # the cap keeps the gold
    big = {"t|4|0": {"state": "S", "options": [f"tool{i}" for i in range(69)], "descriptions": [f"d{i}" for i in range(69)], "text": Q_TOOL_SELECT}}
    r = E18D.cal_rows([{"key": "t|4|0", "task_id": "t", "dp": "tool_select", "n_options": 69, "gold": "tool68", "llm_ok": True, "ref_ok": True, "d": 0}], big)
    assert len(r[0]["options"]) == 52 and "tool68 -- d68" in r[0]["options"]


# ------------------------------------------------------------------ the analysis script, profile and configs
def test_projection_rule():
    log = "... {'train_runtime': '40.0', 'train_loss': '0.5', 'epoch': '0.1'}"
    p = X.projection(log, rows=16902, cap_min=170, setting="nogc_3072")
    assert p["s_per_step_smoke"] == 10.0 and p["steps_per_epoch"] == 265 and p["epochs"] == 2
    assert X.projection(log.replace("40.0", "120.0"), 16902, 170, "s")["epochs"] == 1
    assert X.projection(log.replace("40.0", "400.0"), 16902, 170, "s")["epochs"] == 0
    assert X.projection(log, 16902, 280, "s", max_epochs=1)["epochs"] == 1          # the amended rule: one epoch
    assert X.projection(log.replace("40.0", "241.1"), 16902, 280, "s", max_epochs=1)["epochs"] == 1   # 60.3 s/step: 266 min
    assert X.projection(log.replace("40.0", "260.0"), 16902, 280, "s", max_epochs=1)["epochs"] == 0
    assert "not_run" in X.projection("nothing", 16902, 170, "s")


def test_committed_head_profile_and_live_configs():
    h = json.loads(HEAD.read_text())
    assert h["fit"]["positives"] == 15 and h["fit"]["n"] == 2453 and len(h["weights"]) == 4
    assert h["k"] == {"tool_select": pytest.approx(0.0058322), "tool_params": pytest.approx(0.0034651068524096384)}
    chk = X.check_profile()
    assert chk["ok"], chk
    prof = yaml.safe_load(PROFILE.read_text())
    blk = prof["system2"]["gemini-3.5-flash"]["backends"]
    assert blk["e18"]["tool_select"]["tau"] == pytest.approx(1 - 0.0058322) and blk["e18"]["tool_params"]["tau"] == pytest.approx(1 - 0.0034651068524096384)
    assert blk["jevagent_rl"]["tool_select"]["tau"] == pytest.approx(0.4999999999978404)   # the frozen block unchanged
    assert prof["system2"]["gemini-3.5-flash"]["decision_points"]["tool_params"]["batch_s1"] is True
    live = yaml.safe_load((ROOT / "configs/explore/e17_e18/live_e18.yaml").read_text())
    assert (live["suite"]["seed0"], live["suite"]["n"], live["n_tasks"], live["workers"]) == (300500, 200, 200, 1)
    assert live["arms"] == [{"name": "e18_final", "s1": "e18", "profile": "configs/harness/explore_e18_final.yaml",
                             "profile_s2": "gemini-3.5-flash", "profile_backend": "e18"}]
    assert live["s1_backends"]["e18"]["base_url"] == "http://127.0.0.1:8083" and live["llm"]["reasoning_effort"] == "minimal"
    ded = yaml.safe_load((ROOT / "runs/harness_final_dedicated/experiment.yaml").read_text())
    assert ded["suite"] == live["suite"] and ded["schedule"] == live["schedule"]
    smoke = yaml.safe_load((ROOT / "configs/explore/e17_e18/live_smoke_e18.yaml").read_text())
    assert (smoke["suite"]["seed0"], smoke["suite"]["n"], smoke["n_tasks"]) == (906, 2, 2) and smoke["arms"] == live["arms"]
    reg = (ROOT / "docs/PREREG_A7_STAGE2.md").read_text()
    assert "E17 and E18: read-out ablation and a deferral-aware objective" in reg
    assert "0.0058322" in reg and "0.0034651" in reg and "d = 1 on 3 tool choices and 12" in reg


def test_tex_keys_and_markdown_on_a_minimal_summary():
    r = {"label": "x", "generated": "now", "box": "b", "training": {}, "v1": {"gen": {"v1_acc": 0.99, "v1_ece": 0.01}},
         "corpus": {"rows": {"gen": {"c_acc": 0.5, "c_acc_offplan": 0.4, "c_acc_ask": 0.3, "c_ece": 0.1}}, "contrasts": {}},
         "calibration": {"head": {"coverage": {"tool_select": {"tau": 0.9, "n_paired": 10, "accepted": 8, "coverage": 0.8,
                                                              "s1_acc_accepted": 1.0, "s2_acc_accepted": 1.0, "cascade_acc": 1.0,
                                                              "accepted_all_right": True}}},
                         "gen": {"coverage": {"tool_select": {"tau": 0.8, "n_paired": 10, "accepted": 9, "coverage": 0.9,
                                                             "s1_acc_accepted": 0.9, "s2_acc_accepted": 1.0, "cascade_acc": 1.0,
                                                             "accepted_all_right": False}}}},
         "latency": {}, "e18_head": {"weights": [1, 2, 3, 4], "fit": {"n": 2, "positives": 1, "log_loss_mean": 0.1},
                                     "by_dp": {"tool_select": {"n": 2, "positives": 1, "k": 0.005, "llm_right": 2, "ref_right": 1,
                                                               "coverage_in_sample": 0.5, "positives_deferred": 1}}},
         "e18_corpus": {"e18": {"label": "l", "n": 4, "coverage": 0.75, "accepted_accuracy": 1.0, "accepted_errors": 0,
                                "deferred_accuracy": 0.0, "llm_only": {"n": 2, "deferred": 1, "deferred_llm_right_share": 1.0,
                                                                       "cascade_task_acc": 1.0, "accepted_wrong_llm_right": 0}},
                        "head_at_tauhat": {"label": "a", "n": 4, "coverage": 0.5, "accepted_accuracy": 1.0, "accepted_errors": 0,
                                           "deferred_accuracy": 0.5, "llm_only": {}}},
         "live": {}, "readings": {}, "gpu_hours": 1.5, "spend": {"total": 0.5}}
    r["readings"] = X.readings(r)
    assert r["readings"]["e17/tool_select"]["winner"] == "head"           # the only error-free read-out wins
    assert r["readings"]["e18/c"]["holds"] is True
    k = X.tex_keys(r)
    assert k["e17/reading/ts"] == "the scoring head" and k["e18/c/e18/cov"] == "75.0" and k["e18/c/a/cov"] == "50.0"
    assert k["e17/gen/cal/ts/clean"] == "no" and k["e17/head/cal/ts/cov"] == "80.0" and k["e18/head/ts/tau"] == "0.9950"
    md = X.markdown(r)
    assert "## E18: the defer head" in md and "arm A at its fitted tau-hat" not in md  # labels come from the records


def test_driver_syntax_and_never_touch_list():
    drv = ROOT / "scripts/aws/e17_e18.sh"
    subprocess.run(["bash", "-n", str(drv)], check=True)
    text = drv.read_text()
    for iid in ("<instance-id>", "<instance-id>", "<instance-id>", "<instance-id>", "<instance-id>"):
        assert iid in text
    assert "GUARD_MIN=${GUARD_MIN:-470}" in text and "CAP_USD=${CAP_USD:-8}" in text and "OPENROUTER" not in text
    assert "GEN_CAP_MIN=${GEN_CAP_MIN:-280}" in text and "nogc_6144" in text and "stage_kill_prev" in text
    assert "--clef-defer-head" in text and "stage_live_e18" in text and "stage_e18f_train" in text
    out = subprocess.run(["bash", str(drv)], capture_output=True, text=True)
    assert out.returncode == 0 and "LOCAL:" in out.stdout
