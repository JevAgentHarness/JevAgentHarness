"""scripts/explore_drift.py (offline): on a three-task synthetic fixture, the corpus-hash match of live states (in C /
novel, on plan / after an off-plan step), the live and offline accuracies, the faithfulness pairs, the first-wrong-choice
attribution with its consequence, the gate counts at every threshold, the loss decomposition and the TeX keys (an
undefined quantity is left out so that \\resna prints --)."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import explore_drift as X  # noqa: E402
from jevharness.bench import stage2_corpus as SC  # noqa: E402

OPTS = ["find_user", "get_orders", "cancel_order", "ask_user", "finish"]


def _rec(state, gold, at, s1, conf):
    return {"dp": "tool_select", "mode": "replace", "question": "next tool", "options": list(OPTS), "state": state,
            "s1_value": s1, "s1_conf": conf, "final_value": s1, "gold": gold, "gold_status": "ok",
            "gold_aux": {"v": 1, "acceptable_tools": list(at)}}


def _traj(task, steps, success):
    return {"task_id": task, "config": "x_ts_replace_f", "error": None, "result": {"success": success},
            "steps": [{"step": i, "tool": r["s1_value"], "args": {}, "decisions": [r]} for i, r in enumerate(steps)]}


@pytest.fixture
def fixture(tmp_path: Path):
    descs: dict[str, str] = {}                      # the fixture's environment has no tool descriptions
    q1, q2 = "S1: customer asks to cancel", "S2: user found"
    h1 = SC.question_hash(q1, OPTS, [descs.get(o, "") for o in OPTS])
    h2 = SC.question_hash(q2, OPTS, [descs.get(o, "") for o in OPTS])
    s23 = tmp_path / "s23"
    s23.mkdir()
    with gzip.open(s23 / "questions.jsonl.gz", "wt") as f:
        for h, st in ((h1, q1), (h2, q2)):
            f.write(json.dumps({"qhash": h, "state": st, "options": OPTS, "descriptions": [""] * len(OPTS)}) + "\n")
    occ = [{"task": "A", "source": "llm_only", "step": 0, "qhash": h1, "gold": "find_user", "gold_status": "ok",
            "gold_aux": {"v": 1, "acceptable_tools": ["find_user", "ask_user"]}, "stratum": "on_plan"},
           {"task": "A", "source": "llm_only", "step": 1, "qhash": h2, "gold": "get_orders", "gold_status": "ok",
            "gold_aux": {"v": 1, "acceptable_tools": ["get_orders"]}, "stratum": "on_plan"},
           {"task": "B", "source": "llm_only", "step": 0, "qhash": h1, "gold": "find_user", "gold_status": "ok",
            "gold_aux": {"v": 1, "acceptable_tools": ["find_user", "ask_user"]}, "stratum": "on_plan"},
           {"task": "C", "source": "llm_only", "step": 0, "qhash": h1, "gold": "find_user", "gold_status": "ok",
            "gold_aux": {"v": 1, "acceptable_tools": ["find_user", "ask_user"]}, "stratum": "on_plan"}]
    with gzip.open(s23 / "occurrences.jsonl.gz", "wt") as f:
        for o in occ:
            f.write(json.dumps(o) + "\n")
    (s23 / "corpus.json").write_text(json.dumps({"eligible_tasks": ["A", "B", "C"], "files": {}}))
    answers = {h1: {"qhash": h1, "answer": "find_user", "confidence": 0.9, "error": None, "valid": True},
               h2: {"qhash": h2, "answer": "get_orders", "confidence": 0.8, "error": None, "valid": True}}
    trajs = {
        # A succeeds: two in-C states, both right
        "A": _traj("A", [_rec(q1, "find_user", ["find_user", "ask_user"], "find_user", 0.95),
                         _rec(q2, "get_orders", ["get_orders"], "get_orders", 0.8)], True),
        # B fails after a recoverable first error on a novel on-plan state (an unneeded question, conf 0.4), then a
        # wrong write on a novel state after the off-plan step (conf 0.95): compounding
        "B": _traj("B", [_rec(q1, "find_user", ["find_user", "ask_user"], "find_user", 0.95),
                         _rec("S-novel-1", "get_orders", ["get_orders"], "ask_user", 0.4),
                         _rec("S-novel-2", "get_orders", ["get_orders"], "cancel_order", 0.95)], False),
        # C fails at once: an irreversible action before asking, on an in-C state it answers right offline (conf 0.6)
        "C": _traj("C", [_rec(q1, "find_user", ["find_user", "ask_user"], "cancel_order", 0.6),
                         _rec("S-novel-3", "ask_user", ["ask_user"], "ask_user", 0.7)], False),
    }
    return s23, trajs, answers, (lambda task: descs)


def test_live_states_hash_and_off_plan(fixture):
    s23, trajs, _, descs_for = fixture
    corpus = X.load_corpus(s23)
    assert len(corpus["hashes"]) == 2 and corpus["tasks"] == ["A", "B", "C"]
    sts = X.live_states(trajs["B"], descs_for("B"))
    assert [s["qhash"] in corpus["hashes"] for s in sts] == [True, False, False]
    assert [s["after_off_plan"] for s in sts] == [False, False, True]
    assert [s["s1_correct"] for s in sts] == [True, False, False] and not any(s["fallthrough"] for s in sts)


def test_analysis_on_the_fixture(fixture):
    from benchmark_difficulty import near_map

    s23, trajs, answers, descs_for = fixture
    corpus = X.load_corpus(s23)
    taus = {"own": None, "jev": 1.0, "fixed_0.5": 0.5, "fixed_0.9": 0.9}
    r = X.analyse_model(trajs, corpus, descs_for, answers, taus, near_map())
    assert r["tasks"] == 3 and r["errored"] == 0 and r["success"] == pytest.approx(1 / 3)
    s = r["states"]
    assert (s["labelled"], s["in_c"], s["novel"], s["novel_on_plan"], s["novel_after_off_plan"]) == (7, 4, 3, 1, 2)
    assert s["label_mismatch_in_c"] == 0 and s["fallthrough"] == 0 and s["unlabelled"] == 0
    a = r["acc"]
    assert a["in_c"]["acc"] == pytest.approx(3 / 4) and a["novel"]["acc"] == pytest.approx(1 / 3)
    assert a["novel_on_plan"]["acc"] == 0.0 and a["novel_after_off_plan"]["acc"] == pytest.approx(1 / 2)
    # (a) faithfulness: the in-C live answers against the offline C answers on the same states
    fa = r["faith"]
    assert (fa["n"], fa["live_wrong_offline_right"], fa["live_right_offline_wrong"]) == (4, 1, 0)
    assert fa["live_acc"] == pytest.approx(3 / 4) and fa["offline_acc"] == 1.0 and fa["agree"] == pytest.approx(3 / 4)
    assert fa["sign_p"] == 1.0
    # (b) the first wrong choice: B on a novel state at step 1 (recoverable), C on an in-C state at step 0 (irreversible,
    # answered right offline), A none
    f = r["first"]
    assert (f["in_c"], f["novel"], f["none"]) == (1, 1, 1) and f["step_median"] == 0.5
    assert f["in_c_offline_right"] == 1 and f["in_c_offline_wrong"] == 0 and f["in_c_same_answer"] == 0
    assert f["types"] == {"unneeded question": 1, "changed state before asking": 1}
    assert f["consequence"] == {"recoverable": 1, "wrong write action": 1}
    assert f["failed"] == 2 and f["failed_with_wrong_choice"] == 2 and f["recovered"] == 0
    assert f["irreversible"] == 1 and f["irreversible_of_failed_share"] == pytest.approx(1 / 2)
    assert f["loss"] == {"total": pytest.approx(2 / 3), "irreversible": pytest.approx(1 / 3),
                         "compounding": pytest.approx(1 / 3), "no_wrong_choice": 0.0}
    per = {x["task"]: x for x in f["per_task"]}
    assert per["B"]["step"] == 1 and per["B"]["in_c"] is False and per["C"]["step"] == 0 and per["C"]["in_c"] is True
    # (c) the gate: wrong confidences 0.4, 0.95, 0.6; right 0.95, 0.8, 0.95, 0.7
    g = r["gate"]
    assert g["n_wrong"] == 3 and g["n_right"] == 4
    assert g["conf_wrong_mean"] == pytest.approx((0.4 + 0.95 + 0.6) / 3)
    assert g["at"]["fixed_0.9"]["wrong_below"] == pytest.approx(2 / 3)
    assert g["at"]["fixed_0.9"]["right_below"] == pytest.approx(2 / 4)
    assert g["at"]["fixed_0.9"]["failed_first_below"] == 1.0           # B at 0.4 and C at 0.6
    assert g["at"]["fixed_0.5"]["wrong_below"] == pytest.approx(1 / 3)
    assert g["at"]["fixed_0.5"]["failed_first_below"] == pytest.approx(1 / 2)
    assert g["at"]["jev"]["wrong_below"] == 1.0 and g["at"]["jev"]["right_below"] == 1.0 and "own" not in g["at"]
    # the ask strata: one must-ask state (C, step 1), answered right; nobody asked early
    assert r["mustask"]["n"] == 1 and r["mustask"]["acc"] == 1.0 and r["mustask"]["c_acc"] is None
    assert r["askearly"]["tasks"] == 0 and r["ask"]["n"] == 1
    # the published C accuracy rule reproduces on the fixture (every occurrence answered right offline)
    assert X.corpus_task_accuracy(corpus, answers) == 1.0


def test_tex_keys_skip_undefined_quantities(fixture):
    from benchmark_difficulty import near_map

    s23, trajs, answers, descs_for = fixture
    corpus = X.load_corpus(s23)
    row = X.analyse_model(trajs, corpus, descs_for, answers, {"own": None, "jev": 1.0, "fixed_0.5": 0.5, "fixed_0.9": 0.9},
                          near_map())
    row.update({"c_acc": X.corpus_task_accuracy(corpus, answers), "tau_used": 1.0, "own_tau": None})
    res = {"models": {"m": row}, "corpus": {"questions": 2, "occurrences": 4, "ask_share": None},
           "hash_check": {"src": {"match": 1.0, "labelled_states": 4}}}
    k = X.tex_rows(res)
    assert k["drift/m/live/success"] == "33.3" and k["drift/m/c/acc"] == "100.0"
    assert k["drift/m/acc/in_c"] == "75.0" and k["drift/m/acc/novel"] == "33.3" and k["drift/m/acc/novel_on_plan"] == "0.0"
    assert k["drift/m/first/in_c"] == "33.3" and k["drift/m/first/novel"] == "33.3" and k["drift/m/first/none"] == "33.3"
    assert k["drift/m/first/irreversible"] == "33.3" and k["drift/m/loss/compounding"] == "33.3"
    assert k["drift/m/first/type/unneeded_question/n"] == "1" and k["drift/m/first/conseq/write/n"] == "1"
    assert k["drift/m/gate/09/wrong_below"] == "66.7" and k["drift/m/gate/09/right_below"] == "50.0"
    assert k["drift/m/gate/wrong_below"] == "100.0" and k["drift/m/gate/tau"] == "1.00" and k["drift/m/gate/source"] == "jev"
    assert k["drift/match/src"] == "100.0" and k["drift/c/questions"] == "2"
    assert "drift/m/c/mustask_acc" not in k and "drift/c/ask/share" not in k      # undefined -> left out (\resna: --)
    assert k["drift/m/states/novel_share"] == "42.9"
    tex = X.to_tex({**res, "generated": "t", "git": "g", "label": "l"})
    assert "\\defres{drift/m/acc/novel}{33.3}" in tex and tex.strip().endswith("\\makeatother")


def test_helpers():
    assert X.sign_test_p(0, 0) is None and X.sign_test_p(1, 0) == 1.0 and X.sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert X.kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0 and X.kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert X.kendall_tau_b([1, 2], [1, 2]) is None
    assert X._tau(float("inf")) == "always defer" and X._tau(float("inf"), tex=True) == r"$\infty$"
    assert X._tau(1.0) == "1.00" and X._tau(0.9756255) == "0.9756" and X._tau(None) == "n/a"
    assert X.consequence("cancel_order", {"cancel_order"}) == "wrong write action"
    assert X.consequence("finish", set()) == "premature finish" and X.consequence("get_orders", set()) == "recoverable"
    assert "cancel_order" in X.write_tools(None)


def test_registry_and_fetch_are_explicit():
    assert set(X.S3_FILES) == {"openjev", "openjevhf"}
    for key, spec in X.MODELS.items():
        assert spec["live"].endswith("_ts_replace_f.jsonl")
        assert spec["live"].startswith("e15:") == (key in X.S3_FILES)        # only the S3 rows need `fetch`
    assert X.resolve_live(X.MODELS["openjev"], None) is None                  # never implicit
    assert X.resolve_live(X.MODELS["jev"], None) == X.RUNS / "stage2_rl_states/jev_ts_replace_f.jsonl"


@pytest.mark.skipif(not (X.S23 / "occurrences.jsonl.gz").exists() or not (X.RUNS / "stage2_jevgym/experiment.yaml").exists()
                    or not (X.RUNS / "stage2_rl_states/jev_ts_replace_f.jsonl").exists(),
                    reason="the S2-3 corpus or the Stage-2 live logs are not present")
def test_corpus_hash_matches_the_source_arms_logs():
    corpus = X.load_corpus()
    chk = X.hash_check(corpus, X.description_lookup(), {"jev": X.C_SOURCES["jev"]})
    assert chk["jev"]["found"] and chk["jev"]["match"] == 1.0 and chk["jev"]["labelled_states"] > 1000


def test_walk_predictions_count_consequences(fixture):
    """The offline score from C alone: walk each reference trajectory with the offline answers; a task is lost iff the
    first wrong answer is irreversible; the variants stop at the reference's off-plan step or at an early ask."""
    from benchmark_difficulty import near_map

    s23, trajs, answers, _ = fixture
    corpus = X.load_corpus(s23)
    writes = X.write_tools(near_map())
    h1 = next(o["qhash"] for o in corpus["occurrences"] if o["task"] == "B")
    h2 = next(o["qhash"] for o in corpus["occurrences"] if o["task"] == "A" and o["step"] == 1)
    tasks = ["A", "B", "C"]
    # every answer right -> every task predicted solved, on the one reference trajectory each task has
    p = X.walk_predictions(corpus, answers, writes)
    assert p == {"A": {"llm_only": 1.0}, "B": {"llm_only": 1.0}, "C": {"llm_only": 1.0}}
    assert X.predicted_success(p, tasks) == 1.0
    # a wrong write at the first state -> every task predicted lost; a wrong lookup there -> predicted solved
    bad = {**answers, h1: {**answers[h1], "answer": "cancel_order"}}
    assert X.predicted_success(X.walk_predictions(corpus, bad, writes), tasks) == 0.0
    look = {**answers, h1: {**answers[h1], "answer": "get_orders"}}
    assert X.predicted_success(X.walk_predictions(corpus, look, writes), tasks) == 1.0
    # an acceptable early ask at the first state, then a wrong write at A's second state: lost on the plain walk,
    # solved when the early ask ends the walk
    early = {**answers, h1: {**answers[h1], "answer": "ask_user"}, h2: {**answers[h2], "answer": "cancel_order"}}
    assert X.walk_predictions(corpus, early, writes)["A"] == {"llm_only": 0.0}
    assert X.walk_predictions(corpus, early, writes, early_ask_resolves=True)["A"] == {"llm_only": 1.0}
    # a wrong write only on a state after the reference's own off-plan step: lost on the plain walk, solved when the
    # walk stops at the on-plan prefix; a second source trajectory averages in per task; the LLM-only variant
    corpus["occurrences"].append({"task": "A", "source": "jev_ts_replace_f", "step": 0, "qhash": h1, "gold": "find_user",
                                  "gold_status": "ok", "gold_aux": {"v": 1, "acceptable_tools": ["find_user", "ask_user"]},
                                  "stratum": "on_plan"})
    corpus["occurrences"].append({"task": "A", "source": "jev_ts_replace_f", "step": 1, "qhash": h2, "gold": "get_orders",
                                  "gold_status": "ok", "gold_aux": {"v": 1, "acceptable_tools": ["get_orders"]},
                                  "stratum": "after_off_plan"})
    late = {**answers, h2: {**answers[h2], "answer": "cancel_order"}}
    assert X.walk_predictions(corpus, late, writes)["A"] == {"llm_only": 0.0, "jev_ts_replace_f": 0.0}
    assert X.walk_predictions(corpus, late, writes, on_plan_only=True)["A"] == {"llm_only": 0.0, "jev_ts_replace_f": 1.0}
    assert X.walk_predictions(corpus, late, writes, sources=("llm_only",))["A"] == {"llm_only": 0.0}
    assert X.predicted_success(X.walk_predictions(corpus, late, writes, on_plan_only=True), tasks) == pytest.approx((0.5 + 1 + 1) / 3)


def test_prediction_summary_kendall_and_keys():
    def row(success, c, must, pred, group):
        return {"success": success, "c_acc": c, "mustask": {"c_acc": must}, "group": group,
                "pred": {"all_sources": pred, "llm_only": pred, "on_plan": pred, "early_ask": pred}}
    models = {"a": row(0.9, 0.9, 0.5, 0.8, "off the shelf"), "b": row(0.6, 0.95, 0.3, 0.7, "off the shelf"),
              "c": row(0.3, 0.7, 0.2, 0.2, "off the shelf (E15)"), "d": row(1.0, 0.99, 1.0, 1.0, "ours")}
    pr = X.prediction_summary(models)
    assert pr["n"] == {"all": 4, "offtheshelf": 3} and pr["sets"]["offtheshelf"] == ["a", "b", "c"]
    assert pr["kendall"]["offtheshelf"]["pred"] == 1.0 and pr["kendall"]["offtheshelf"]["cmean"] == pytest.approx(1 / 3)
    assert pr["mae"]["offtheshelf"]["pred"] == pytest.approx((0.1 + 0.1 + 0.1) / 3)
    assert pr["rows"]["b"]["pred_err"] == pytest.approx(0.1)
    k = X.tex_rows({"models": {}, "corpus": {"questions": 0, "occurrences": 0}, "pred": pr})
    assert k["drift/pred/kendall/offtheshelf/pred"] == "1.00" and k["drift/pred/kendall/all/n"] == "4"
    assert k["drift/pred/b/pred"] == "70.0" and k["drift/pred/b/pred/err"] == "+10.0" and k["drift/pred/mae/offtheshelf/pred"] == "10.0"
