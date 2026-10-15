"""A7.13 (iv): the τ² reference-agreement corpus, replay and score (scripts/tau2_reference_replay.py), offline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t2ref", ROOT / "scripts/tau2_reference_replay.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)  # type: ignore[union-attr]

NAMES = ["get_order_details", "cancel_pending_order", "respond_to_user"]
DESCS = ["Get an order.", "Cancel an order.", "Send a text message to the user."]


def _dec(dp, state, gold, mode="cascade", s1="respond_to_user", options=None, question="Q?"):
    return {"dp": dp, "mode": mode, "question": question, "options": options or NAMES, "state": state,
            "gold": gold, "s1_value": s1 if mode != "off" else None}


def _write_runs(tmp: Path) -> Path:
    runs = tmp / "runs"
    for dom, tdom in (("retail", "retail"), ("airline_test", "airline")):
        d = runs / "tau2" / dom
        d.mkdir(parents=True)
        recs = []
        for task in ("1", "2"):
            steps = [{"decisions": [_dec("tool_select", f"{dom} s{task}a", "get_order_details", s1="get_order_details"),
                                    _dec("tool_params", f"{dom} s{task}a", None if task == "2" else "#W1",
                                         options=["#W1", "#W2", "OTHER"], s1="#W1", question="Which value?")]},
                     {"decisions": [_dec("tool_select", f"{dom} s{task}b", "respond_to_user")]}]
            recs.append({"task_id": f"{tdom}/{task}#0", "meta": {"tau2": {"domain": tdom}}, "steps": steps})
        recs.append({"task_id": f"{tdom}/1#0", "meta": {"tau2": {"domain": tdom}}, "steps": recs[0]["steps"]})  # resumed
        (d / "jev_ts_cascade_tauhat_refit.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        off = {"task_id": f"{tdom}/1#0", "meta": {"tau2": {"domain": tdom}},
               "steps": [{"decisions": [_dec("tool_select", f"{dom} off", "respond_to_user", mode="off")]}]}
        (d / "llm_only.jsonl").write_text(json.dumps(off) + "\n")
    return runs


def test_build_replay_score(tmp_path):
    runs = _write_runs(tmp_path)
    desc = {"retail": (NAMES, DESCS), "airline": (NAMES, DESCS)}
    c = T.build_corpus(runs, ts_desc=desc)
    cov = c["summary"]["coverage"]
    # the `off` record is not a System-1 question; the unlabelled tool_params occurrence counts in the coverage only
    assert cov["retail/tool_select"]["not_asked"] == 1 and cov["retail/tool_select"]["occurrences"] == 4
    assert cov["retail/tool_params"]["occurrences"] == 2 and cov["retail/tool_params"]["labelled"] == 1
    out = tmp_path / "t2ref"
    summ = T.write_corpus(c, out)
    qs = T.read_gz(out / "questions.jsonl.gz")
    # no label and no logged answer on the questions the GPU host reads
    assert all(set(q) == {"qhash", "dp", "state", "text", "options", "descriptions"} for q in qs)
    assert all(q["descriptions"] == DESCS for q in qs if q["dp"] == "tool_select")
    assert all(q["descriptions"] is None for q in qs if q["dp"] == "tool_params")
    assert summ["n_questions"] == len(qs) == 10      # 4 labelled ts + 1 labelled tp per domain
    first, _ = T.shard_of(qs, "0/2")
    second, _ = T.shard_of(qs, "1/2")
    assert {q["qhash"] for q in first} | {q["qhash"] for q in second} == {q["qhash"] for q in qs}

    from jevharness.decision import make_backend

    be = make_backend({"type": "heuristic"})
    for name in ("frozen", "base"):
        with open(out / f"answers_{name}.jsonl", "w") as f:
            T.replay(qs, be, on_answer=lambda r, f=f: f.write(json.dumps(r) + "\n"))
    # a perfect row from the labels, to check the arithmetic
    occ = [o for o in T.read_gz(out / "occurrences.jsonl.gz") if o["gold"] is not None]
    gold = {o["qhash"]: o["gold"] for o in occ}
    with open(out / "answers_intern.jsonl", "w") as f:
        for h, g in gold.items():
            f.write(json.dumps({"qhash": h, "answer": g, "valid": True, "error": None}) + "\n")
    r = T.score(out)
    assert r["rows"] == ["frozen", "intern", "base"]
    s = r["strata"]["all/tool_select"]
    assert s["rows"]["intern"]["agree"] == 1.0
    assert s["paired"]["frozen_vs_intern"]["delta"] <= 0
    assert s["majority"]["agree"] == 0.5
    assert s["jev_subset"]["jev_logged"]["agree"] == 1.0          # the logged answers above match the labels
    tex = T.to_tex(r)
    assert "\\defres{t2ref/all/tool_select/intern/agree}{100.0}" in tex
    assert "t2ref" in T.to_markdown(r) or "τ²" in T.to_markdown(r)
