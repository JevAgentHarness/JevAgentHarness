"""scripts/stage2_failure_audit.py on tiny synthetic episodes (offline): the matching rule (same family and plan shape,
nearest seed, ties to the lower seed, without replacement, flagged fallback), the per-episode fields (first divergence
from the oracle's acceptable set, who chose the diverging action, accepted System-1 errors against gold, premature
finish, strict-mode termination), the paired split against the cascade alone (free-text wording ignored), the labels,
the reading rule, the registered-count check and the TeX keys."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import stage2_failure_audit as A  # noqa: E402

FREE = {"ask_user.question", "finish.answer"}


def tid(seed: int, fam: str = "clarify") -> str:
    return f"jevgym-{fam}-{seed}-d60-near-strict"


def ts(s1: str, conf: float, acc: list[str], tau: float = 1.0, mode: str = "cascade") -> dict:
    deferred = conf < tau
    return {"dp": "tool_select", "mode": mode, "s1_value": s1 if mode != "off" else None,
            "s1_conf": conf if mode != "off" else None, "deferred": deferred, "final_value": None,
            "gold": acc[0], "gold_status": "ok", "gold_aux": {"acceptable_tools": acc}, "shadow": False}


def refl(fire: bool) -> dict:
    return {"dp": "reflection", "mode": "trigger", "s1_value": fire, "final_value": fire, "gold": None,
            "gold_status": "unobserved" if fire else "ok", "shadow": False}


def step(i: int, tool: str, args: dict, dec: dict, oracle: str, extra: list | None = None, obs: str = "ok") -> dict:
    d = dict(dec)
    d["final_value"] = tool
    return {"step": i, "tool": tool, "args": args, "oracle_tool": oracle, "observation": obs,
            "decisions": list(extra or []) + [d]}


def ep(task: str, steps: list[dict], success: bool, irreversible: bool = False) -> dict:
    return {"task_id": task, "error": None, "steps": steps, "meta": {"config": {"max_steps": 30,
                                                                              "tool_select": {"tau": 1.0}}},
            "result": {"success": success, "irreversible_error": irreversible, "side_effects": 0}}


def failing_profile(task: str) -> dict:
    """find_user (LLM) -> reflection fires -> get_orders (S1 accepted) -> refund accepted at 1.0 (not acceptable;
    irreversible)."""
    return ep(task, [
        step(0, "find_user", {"name": "A B"}, ts("ask_user", 0.6, ["ask_user", "find_user"]), "ask_user"),
        step(1, "get_orders", {"user_id": "U1", "status": "delivered"}, ts("get_orders", 1.0, ["ask_user", "get_orders"]),
             "ask_user", extra=[refl(True)]),
        step(2, "refund", {"order_id": "O1"}, ts("refund", 1.0, ["ask_user"]), "ask_user", extra=[refl(False)],
             obs="Irreversible error: refund(...)"),
    ], success=False, irreversible=True)


def cascade_twin(task: str) -> dict:
    """The same seed without the trigger: same first two actions (the question wording of a later ask differs only in
    free text), then refund proposed at 0.99 -> deferred -> the LLM asks."""
    return ep(task, [
        step(0, "find_user", {"name": "A B"}, ts("ask_user", 0.6, ["ask_user", "find_user"]), "ask_user"),
        step(1, "get_orders", {"status": "delivered", "user_id": "U1"}, ts("get_orders", 0.99, ["ask_user", "get_orders"]),
             "ask_user"),
        step(2, "ask_user", {"missing": "refund_method", "question": "Which method?"}, ts("refund", 0.99, ["ask_user"]),
             "ask_user"),
        step(3, "refund", {"order_id": "O1", "method": "card"}, ts("refund", 1.0, ["refund"]), "refund"),
        step(4, "finish", {"answer": "done"}, ts("finish", 0.9, ["finish"]), "finish"),
    ], success=True)


def success_profile(task: str, fire: bool = True) -> dict:
    return ep(task, [
        step(0, "find_user", {"name": "C D"}, ts("ask_user", 0.6, ["ask_user", "find_user"]), "ask_user"),
        step(1, "get_orders", {"user_id": "U2", "status": "delivered"}, ts("get_orders", 0.99, ["ask_user", "get_orders"]),
             "ask_user", extra=[refl(fire)]),
        step(2, "ask_user", {"missing": "refund_method", "question": "How should we refund?"},
             ts("refund", 0.99, ["ask_user"]), "ask_user", extra=[refl(False)]),
        step(3, "refund", {"order_id": "O2", "method": "card"}, ts("refund", 1.0, ["refund"]), "refund"),
        step(4, "finish", {"answer": "ok"}, ts("finish", 0.9, ["finish"]), "finish"),
    ], success=True)


def test_matching_rule() -> None:
    meta = {tid(s): {"family": "clarify", "shape": "A", "variant": "method"} for s in (10, 14, 16, 18, 30)}
    meta[tid(13)] = {"family": "clarify", "shape": "B", "variant": "order"}
    meta[tid(20)] = {"family": "clarify", "shape": "C", "variant": "speed"}
    meta[tid(21)] = {"family": "multi", "shape": "C", "variant": None}
    meta[tid(40)] = {"family": "clarify", "shape": "B", "variant": "order"}
    fails = [tid(20), tid(14), tid(13)]
    succ = [tid(10), tid(16), tid(18), tid(30), tid(21), tid(40)]
    m = A.match_successes(fails, succ, meta)
    # failures are processed in seed order: 13, 14, 20
    assert [x["failure"] for x in m] == [tid(13), tid(14), tid(20)]
    by = {x["failure"]: x for x in m}
    # 13 (shape B): the only shape-B success is 40, far away -- the shape still wins over a nearer seed
    assert by[tid(13)]["match"] == tid(40) and by[tid(13)]["rule"] == "family+shape"
    # 14 (shape A): 10 is 4 away, 18 is 4 away, 16 is 2 away -> 16; a tie would go to the lower seed
    assert by[tid(14)]["match"] == tid(16) and by[tid(14)]["seed_gap"] == 2
    # 20 (shape C, clarify): the shape-C success is another family -> same family, nearest unused seed (18)
    assert by[tid(20)]["rule"] == "family" and by[tid(20)]["match"] == tid(18)
    assert len({x["match"] for x in m}) == len(m)                  # without replacement


def test_matching_ties_go_to_the_lower_seed() -> None:
    meta = {tid(s): {"family": "clarify", "shape": "A", "variant": None} for s in (10, 12, 14, 16)}
    m = A.match_successes([tid(14), tid(12)], [tid(10), tid(16)], meta)
    # 12 first: 10 (gap 2) beats 16 (gap 4); 14: 10 is used, 16 (gap 2) remains
    assert [(x["failure"], x["match"]) for x in m] == [(tid(12), tid(10)), (tid(14), tid(16))]
    m2 = A.match_successes([tid(14)], [tid(12), tid(16)], meta)
    assert m2[0]["match"] == tid(12)                               # equal gaps -> the lower seed


def test_episode_fields_and_label() -> None:
    f = A.episode_fields(failing_profile(tid(1)), FREE, trigger=True)
    assert f["first_div"] == 2 and f["div_kind"] == "tool" and f["div_by"] == "s1_tool"
    assert f["s1_err_tool_steps"] == [2] and f["s1_err_arg_steps"] == []
    assert f["fires"] == [1] and f["fired_before_div"]
    assert f["termination"] == "strict_irreversible" and f["fatal"] == {"step": 2, "tool": "refund", "by": "s1_tool"}
    assert A.label_of(f, trigger=True) == "accepted System-1 tool choice after a reflection"
    g = A.episode_fields(cascade_twin(tid(1)), FREE, trigger=False)
    assert g["first_div"] is None and A.label_of(g, trigger=False) == "no divergence"
    assert g["termination"] == "finished"


def test_llm_divergence_and_premature_finish() -> None:
    e = ep(tid(2), [
        step(0, "find_user", {"name": "X"}, ts("find_user", 1.0, ["find_user"]), "find_user"),
        step(1, "finish", {"answer": "done"}, ts("get_orders", 0.5, ["get_orders"]), "get_orders"),
    ], success=False)
    f = A.episode_fields(e, FREE, trigger=False)
    assert f["first_div"] == 1 and f["div_by"] == "llm" and f["premature_finish"]
    assert A.label_of(f, trigger=False) == "premature finish"
    # an acceptable tool with irreversible arguments chosen by the LLM: an argument divergence by the LLM
    e2 = ep(tid(3), [step(0, "cancel_order", {"order_id": "WRONG"}, ts("get_orders", 0.4, ["cancel_order"]),
                          "cancel_order", obs="Irreversible error: cancel_order(...)")], success=False, irreversible=True)
    f2 = A.episode_fields(e2, FREE, trigger=False)
    assert f2["div_kind"] == "arguments" and f2["div_by"] == "llm" and f2["fatal"]["by"] == "llm"


def test_accepted_argument_error_counts() -> None:
    arg = {"dp": "tool_params", "mode": "cascade", "s1_value": "O9", "deferred": False, "final_value": "O9",
           "gold": "O1", "gold_status": "ok", "question": "Which value should the argument `order_id` take?",
           "shadow": False}
    e = ep(tid(4), [step(0, "cancel_order", {"order_id": "O9"}, ts("cancel_order", 1.0, ["cancel_order"]),
                         "cancel_order", extra=[arg], obs="Irreversible error: cancel_order(...)")],
           success=False, irreversible=True)
    f = A.episode_fields(e, FREE, trigger=False)
    assert f["div_kind"] == "arguments" and f["div_by"] == "s1_arg" and f["s1_err_arg_steps"] == [0]
    assert A.label_of(f, trigger=False) == "accepted System-1 argument"


def test_split_ignores_free_text_and_reads_the_reflection() -> None:
    sp = A.split_fields(failing_profile(tid(1)), cascade_twin(tid(1)), FREE)
    # step 1's args differ only in key order (same dict); step 2 differs in tool -> split at 2, after the fire at 1
    assert sp["split"] == 2 and sp["split_kind"] == "tool" and sp["vs_cascade"] == "split_after_reflection"
    at = sp["at_split"]
    assert at["profile"]["s1"] == {"tool": "refund", "conf": 1.0, "accepted": True, "final": "refund"}
    assert at["cascade"]["s1"]["accepted"] is False and at["fires_up_to_split"] == 1
    # the same episode twice except the question wording: no split, the free-text difference is recorded
    a, b = success_profile(tid(5)), success_profile(tid(5))
    b["steps"][2]["args"]["question"] = "Refund to which method?"
    sp2 = A.split_fields(a, b, FREE)
    assert sp2["split"] is None and sp2["free_text_diffs"] == [2] and sp2["vs_cascade"] == "no_split"
    sp3 = A.split_fields(success_profile(tid(5), fire=False), b, FREE)
    assert sp3["vs_cascade"] == "no_reflection_no_split"


def _write_run(tmp: Path, n_reg: int) -> tuple[Path, Path, Path, dict]:
    fails, succ = [tid(300000), tid(300002)], [tid(300001), tid(300003), tid(300004)]
    tasks = sorted(fails + succ, key=A.seed_of)
    run = tmp / "run"
    run.mkdir()
    prof = {t: failing_profile(t) for t in fails} | {t: success_profile(t) for t in succ}
    casc = {t: cascade_twin(t) for t in tasks}
    llm = {t: cascade_twin(t) for t in tasks}
    for name, d in ((A.PROFILE, prof), (A.CASCADE, casc), (A.LLM, llm)):
        (run / f"{name}.jsonl").write_text("".join(json.dumps(d[t]) + "\n" for t in tasks))
    tp = tmp / "tasks.json"
    tp.write_text(json.dumps({"suite": {"name": "jevgym"}, "task_ids": tasks}))
    reg = tmp / "stage2.json"
    reg.write_text(json.dumps({"descriptive": {"intervention": {
        "jev_profile_failures": {"n": n_reg, "with_trigger_fired": 2},
        "arms": {A.PROFILE: {"trigger_tasks": 5}}}}}))
    meta = {t: {"family": "clarify", "variant": "method", "shape": "ask_user > find_user > get_orders > refund > finish",
                "free_args": sorted(FREE)} for t in tasks}
    return run, tp, reg, meta


def test_audit_end_to_end(tmp_path: Path) -> None:
    run, tp, reg, meta = _write_run(tmp_path, 2)
    res = A.audit(run, tp, reg, meta_fn=lambda fz: meta)
    S = res["summary"]
    assert res["n_failures"] == 2 and S["fail"]["split_after_reflection"] == 2 and S["fail"]["div_by_s1_tool"] == 2
    assert S["fail_cascade"]["success"] == 2 and S["match"]["n"] == 2 and S["success_all"]["n"] == 3
    # every failure has the same label and splits after a reflection -> "consistent with", never a cause
    assert res["consistent"] and "does not establish" in res["reading"]
    assert [m["match"] for m in res["matches"]] == [tid(300001), tid(300003)]
    assert res["cases"]["failure"]["seed"] == 300000
    assert res["cases"]["deferral"]["s1_tool"] == "refund" and res["cases"]["deferral"]["llm_tool"] == "ask_user"
    tex = A.to_tex(res)
    assert r"\defres{audit/fail/n}{2}" in tex and r"\defres{audit/case/defer/s1_tool}{\texttt{refund}}" in tex
    assert r"\defres{audit/fail/fatal_tool/refund}{2}" in tex
    md = A.cases_markdown(res)
    assert "caused" not in md and "does not identify a cause" in md


def test_mixed_labels_read_as_not_identified(tmp_path: Path) -> None:
    run, tp, reg, meta = _write_run(tmp_path, 2)
    # turn the second failure's diverging action into an LLM choice: two labels -> not consistent
    lines = (run / f"{A.PROFILE}.jsonl").read_text().splitlines()
    recs = [json.loads(x) for x in lines]
    r = next(x for x in recs if x["task_id"] == tid(300002))
    r["steps"][2]["decisions"][-1].update({"s1_conf": 0.5, "deferred": True})
    (run / f"{A.PROFILE}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    res = A.audit(run, tp, reg, meta_fn=lambda fz: meta)
    assert not res["consistent"] and res["reading"] == "these counts do not identify how the failures arose"


def test_registered_failure_count_must_match(tmp_path: Path) -> None:
    run, tp, reg, meta = _write_run(tmp_path, 3)
    with pytest.raises(A.AuditError, match="registered"):
        A.audit(run, tp, reg, meta_fn=lambda fz: meta)
