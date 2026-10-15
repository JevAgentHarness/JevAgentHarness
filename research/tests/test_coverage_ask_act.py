"""scripts/coverage_ask_act.py (offline): S2-3's scoring on synthetic corpora (strata, task weighting, answer kinds,
over-ask rate, failed answers), the reproduction check against published keys (a mismatch writes nothing and exits 1),
the TeX keys, and -- when the committed corpus C is present -- the exact reproduction of every published C row."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import coverage_ask_act as A  # noqa: E402
import coverage_tex as CT  # noqa: E402


def occ(task: str, q: str, gold: str, acc: list[str], stratum: str = "on_plan") -> dict:
    return {"task": task, "source": "llm_only", "step": 0, "qhash": q, "gold": gold, "dp": "tool_select",
            "gold_aux": {"v": 1, "acceptable_tools": acc}, "gold_status": "ok", "stratum": stratum}


OCC = [
    # task t1 (clarify): ask now (only ask), ask now (a read also acceptable), then act
    occ("t1", "q1", "ask_user", ["ask_user"]),
    occ("t1", "q2", "ask_user", ["ask_user", "find_user"]),
    occ("t1", "q3", "ship_order", ["ship_order"], stratum="after_off_plan"),
    # task t2 (multi): ask acceptable but the oracle reads first; then act
    occ("t2", "q4", "find_user", ["find_user", "ask_user"]),
    occ("t2", "q5", "cancel_order", ["cancel_order"]),
    # task t3: not eligible (must be ignored everywhere)
    occ("t3", "q6", "ask_user", ["ask_user"]),
]
ANSWERS = {
    "rl": {"q1": "ask_user", "q2": "find_user", "q3": "ship_order", "q4": "find_user", "q5": "cancel_order",
           "q6": "refund"},
    # jev: acts on q1, asks on q2, over-asks on q5 (an act state), q3 failed
    "jev": {"q1": "refund", "q2": "ask_user", "q3": None, "q4": "ask_user", "q5": "ask_user", "q6": "ask_user"},
}


def corpus(tmp_path: Path, answers: dict = ANSWERS) -> Path:
    d = tmp_path / "s23"
    d.mkdir()
    (d / "corpus.json").write_text(json.dumps({"registered": "test", "eligible_tasks": ["t1", "t2"], "excluded": []}))
    with gzip.open(d / "occurrences.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(o) for o in OCC) + "\n")
    for name, ans in answers.items():
        lines = [json.dumps({"qhash": q, "answer": "finish", "confidence": 0.5, "error": None, "valid": True})
                 for q in ans]           # an older record per question: the latest one must decide
        for q, a in ans.items():
            rec = {"qhash": q, "answer": a, "confidence": 0.9, "error": None if a else "timeout", "valid": True}
            lines.append(json.dumps(rec))
        (d / f"answers_{name}.jsonl").write_text("\n".join(lines) + "\n")
        (d / f"replay_{name}.json").write_text(json.dumps({"backend": {"backend": name}}))
    return d


def published(tmp_path: Path, rl_all: str = "100.0", jev_all: str = "25.0", jev_ask: str = "50.0") -> list[Path]:
    p, q = tmp_path / "stage2.tex", tmp_path / "difficulty.tex"
    p.write_text("\\defres{stage2_rl/frozen/corpus_acc}{%s}\n\\defres{stage2_s1/frozen/corpus_acc}{%s}\n"
                 "\\defres{stage2_rl/frozen/corpus_acc_offplan}{100.0}\n"
                 "\\defres{stage2_rl/jev/corpus_acc}{%s}\n\\defres{stage2_rl/jev/corpus_acc_offplan}{0.0}\n"
                 % (rl_all, rl_all, jev_all))
    q.write_text("\\defres{diff/c/ask_gold/jev/acc}{%s}\n\\defres{diff/c/ask_gold/llm/acc}{100.0}\n" % jev_ask)
    return [p, q]


def test_strata_task_weighting_and_answer_kinds(tmp_path):
    res = A.analyse(corpus(tmp_path), published(tmp_path))
    assert res["strata"]["all"] == {"occurrences": 5, "tasks": 2}             # t3 is not eligible
    assert res["strata"]["ask_now"] == {"occurrences": 2, "tasks": 1}
    assert res["strata"]["ask_only"] == {"occurrences": 1, "tasks": 1}
    assert res["strata"]["ask_optional"] == {"occurrences": 1, "tasks": 1}
    assert res["strata"]["act"] == {"occurrences": 2, "tasks": 2}
    rl, jev = res["rows"]["rl"], res["rows"]["jev"]
    assert rl["acc"]["all"] == 1.0 and rl["acc"]["ask_now"] == 1.0
    # jev: t1 = mean(q1 wrong, q2 right, q3 failed) = 1/3; t2 = mean(q4 right, q5 wrong) = 1/2 -> task mean 5/12
    assert jev["acc"]["all"] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert jev["acc"]["ask_now"] == 0.5 and jev["acc"]["ask_only"] == 0.0 and jev["acc"]["ask_optional"] == 1.0
    assert jev["acc"]["after_off_plan"] == 0.0                                 # the failed answer is incorrect
    assert rl["ask_now_answers"] == {"asked": 1, "other_acceptable": 1, "acted": 0, "finished": 0, "other_wrong": 0,
                                     "failed": 0}
    assert jev["ask_now_answers"]["acted"] == 1 and jev["ask_now_answers"]["asked"] == 1
    # over-ask on act states, task-weighted: t1 (q3 failed: not an ask) 0, t2 (q5 asked) 1 -> 0.5
    assert jev["over_ask"] == 0.5 and rl["over_ask"] == 0.0


def test_reproduction_check_and_tex(tmp_path):
    out, tex = tmp_path / "out", tmp_path / "coverage.tex"
    s23 = corpus(tmp_path)
    jev_all = CT.pct((1 / 3 + 1 / 2) / 2)
    pub = published(tmp_path, jev_all=jev_all)
    assert A.main(["--s23", str(s23), "--published", *map(str, pub), "--out-dir", str(out), "--tex", str(tex)]) == 0
    res = json.loads((out / "ask_act.json").read_text())
    assert res["reproduced"] and len(res["checks"]) == 6          # 5 stage2 keys + the difficulty ask_gold key
    t = tex.read_text()
    assert r"\defres{cov/ask/jev/ask_now/acc}{50.0}" in t and r"\defres{cov/ask/rl/ask_now/acc}{100.0}" in t
    assert r"\defres{cov/ask/ask_now/occurrences}{2}" in t and r"\defres{cov/ask/jev/ask_now/acted}{1}" in t
    assert r"\defres{cov/ask/jev/act/over_ask}{50.0}" in t
    assert t.count(r"\providecommand{\res}") == 1 and t.rstrip().endswith(r"\makeatother")
    assert "Reproduction of the published C rows: 6 / 6" in (out / "ask_act.md").read_text()


@pytest.mark.parametrize("bad", [{"jev_all": "99.9"}, {"jev_ask": "64.4"}])
def test_mismatch_with_published_rows_writes_nothing(tmp_path, bad):
    out = tmp_path / "out"
    rc = A.main(["--s23", str(corpus(tmp_path)), "--published", *map(str, published(tmp_path, **bad)),
                 "--out-dir", str(out), "--tex", str(tmp_path / "c.tex")])
    assert rc == 1 and not out.exists() and not (tmp_path / "c.tex").exists()


@pytest.mark.skipif(not (A.S23 / "occurrences.jsonl.gz").exists() or not all(p.exists() for p in A.PUBLISHED),
                    reason="committed corpus C or the published stage2.tex / difficulty.tex not present")
def test_committed_corpus_reproduces_published_rows():
    res = A.analyse(A.S23, A.PUBLISHED)
    assert res["reproduced"], [c for c in res["checks"] if not c["ok"]]
    assert {c["backend"] for c in res["checks"]} >= {"rl", "jev", "intern", "sl0.8b", "sl27b"}
    assert {c["stratum"] for c in res["checks"]} == {"all", "after_off_plan", "ask_now"}
    assert res["strata"]["ask_now"] == {"occurrences": 582, "tasks": 100}
    acc = {k: CT.pct(r["acc"]["ask_now"]) for k, r in res["rows"].items()}
    assert (acc["rl"], acc["jev"], acc["intern"]) == ("100.0", "64.4", "59.6")
