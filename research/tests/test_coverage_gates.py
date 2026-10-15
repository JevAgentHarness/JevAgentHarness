"""scripts/coverage_gates.py (offline) on tiny synthetic Stage-1 logs: latest record per episode, errored episodes
skipped, first proposals only for the tool guard, verdict and state-changing tallies, the AUROC / recall minimum,
first vs all finishes and the episode outcomes for the stop guard, the limits read off the counts, the TeX keys, and
the merge of both analyses into one coverage.tex (scripts/coverage_tex.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import coverage_gates as G  # noqa: E402
import coverage_tex as CT  # noqa: E402


def tool_dec(tool: str, right: bool, p: float, verdict: str | None = None, retry: int = 0,
             status: str = "ok") -> dict:
    return {"dp": "tool_select", "mode": "guard", "s1_value": p >= 0.5, "s1_conf": max(p, 1 - p),
            "s1_dist": {"yes": p, "no": 1 - p}, "s2_value": tool, "gold": right if status == "ok" else None,
            "gold_status": status, "gold_aux": {"verdict": verdict or ("progress" if right else "detour")},
            "retry": retry}


def stop_dec(right: bool, p: float, retry: int, kind: str | None = None) -> dict:
    return {"dp": "stop", "mode": "guard", "s1_value": p >= 0.5, "s1_conf": max(p, 1 - p),
            "s1_dist": {"yes": p, "no": 1 - p}, "gold": right, "gold_status": "ok",
            "gold_aux": {"kind": kind or ("ok" if right else "wrong_answer")}, "retry": retry}


def ep(task: str, decs: list[list[dict]], success: bool, error: str | None = None) -> dict:
    return {"task_id": task, "error": error, "result": {"success": success},
            "steps": [{"step": i, "decisions": d} for i, d in enumerate(decs)]}


def write(runs: Path, run: str, arm: str, eps: list[dict]) -> None:
    (runs / run).mkdir(parents=True, exist_ok=True)
    (runs / run / f"{arm}.jsonl").write_text("\n".join(json.dumps(e) for e in eps) + "\n")


def tool_eps(n_wrong: int, n_right: int, sc_wrong: int = 0) -> list[dict]:
    """Episodes with n_right right proposals (p 0.9; the first one vetoed at p 0.3) and n_wrong wrong ones (p 0.2,
    the last one accepted at p 0.6), sc_wrong of them state-changing and fatal; plus noise the reader must skip."""
    eps = []
    for i in range(n_right):
        eps.append(ep(f"r{i}", [[tool_dec("find_user", True, 0.3 if i == 0 else 0.9)]], True))
    for i in range(n_wrong):
        last = i == n_wrong - 1
        sc = i < sc_wrong
        eps.append(ep(f"w{i}", [[tool_dec("refund" if sc else "search_docs", False, 0.6 if last else 0.2,
                                          verdict="fatal" if sc else "detour"),
                                 tool_dec("find_user", True, 0.9, retry=1)]], False))   # retry 1: never read
    eps.append(ep("doomed", [[tool_dec("find_user", True, 0.9, status="doomed")]], True))
    eps.append(ep("err", [[tool_dec("finish", False, 0.9)]], False, error="boom"))       # errored: skipped
    return eps


def stop_eps() -> list[dict]:
    return [
        ep("a", [[stop_dec(True, 0.9, 0)]], True),                                     # right, accepted
        ep("b", [[stop_dec(True, 0.2, 0)], [stop_dec(True, 0.9, 1)]], True),           # right blocked, then accepted
        ep("c", [[stop_dec(False, 0.1, 0)], [stop_dec(True, 0.8, 1)]], True),          # caught, then fixed
        ep("d", [[stop_dec(False, 0.1, 0)], [stop_dec(False, 0.2, 1)], [stop_dec(False, 0.7, 2)]], False),
        ep("e", [[stop_dec(False, 0.7, 0, kind="premature")]], False),                 # wrong, accepted
        ep("e", [[stop_dec(True, 0.9, 0)]], True),                                     # ... re-run: latest wins
    ]


@pytest.fixture()
def runs(tmp_path: Path) -> Path:
    r = tmp_path / "runs"
    for rid, (run, arm, *_rest) in G.TOOL_ROWS.items():
        write(r, run, arm, tool_eps(12, 15) if rid == "lite_jev" else tool_eps(1, 5, sc_wrong=1 if rid == "flash_lite"
                                                                                 else 0))
    for rid, (run, arm, *_rest) in G.STOP_ROWS.items():
        write(r, run, arm, stop_eps())
    return r


def test_tool_guard_counts(runs):
    res = G.analyse(runs)
    r = res["tool"]["lite_jev"]
    assert r["episodes"] == 12 + 15 + 1 and r["errored"] == 1
    assert (r["n"], r["right"], r["wrong"]) == (27, 15, 12)
    assert r["vetoed_wrong"] == 11 and r["vetoed_right"] == 1
    assert r["recall"] == pytest.approx(11 / 12) and r["false_veto"] == pytest.approx(1 / 15)
    assert r["auroc"] is not None and 0.5 < r["auroc"] < 1.0
    assert r["excluded"] == {"doomed": 1} and r["retries"] == {"0": 28, "1": 12}
    assert r["wrong_by_verdict"] == {"detour": {"n": 12, "vetoed": 11}}
    small = res["tool"]["flash_jev"]
    assert small["wrong"] == 1 and small["auroc"] is None and small["auroc_raw"] is not None   # below MIN_CLASS
    assert res["tool"]["flash_lite"]["state_changing"] == {"n": 1, "right": 0, "wrong": 1, "wrong_vetoed": 0,
                                                          "right_vetoed": 0}
    assert res["tool"]["flash_lite"]["wrong_by_verdict"]["fatal"] == {"n": 1, "vetoed": 0}


def test_stop_guard_first_vs_all_and_episode_outcomes(runs):
    r = G.analyse(runs)["stop"]["lite_jev"]
    f, a = r["first"], r["all"]
    assert (f["n"], f["right"], f["wrong"], f["vetoed_wrong"], f["vetoed_right"]) == (5, 3, 2, 2, 1)
    assert (a["n"], a["right"], a["wrong"]) == (9, 5, 4)
    assert r["episodes_by_first_finish"] == {"accepted_right": 2, "blocked": 1, "blocked_then_success": 1,
                                             "wrong_accepted": 0, "caught": 2, "caught_then_success": 1}
    assert r["all_by_kind"] == {"ok": 5, "wrong_answer": 4} and f["auroc"] is None      # the re-run replaced "e"


def test_limits_are_read_off_the_counts(runs):
    res = G.analyse(runs)
    text = " ".join(res["limits"])
    assert "cannot test the safety role" in text                 # no wrong state-changing call under a decision model
    assert "Flash → Flash-Lite: 0 of 1 vetoed" in text            # the one fatal proposal, under the LLM gate
    assert "none of the" in text and "was premature" in text      # no premature finish once "e" was re-run
    assert "Too few wrong proposals" in text and "Flash → Jev (1 wrong" in text


def test_main_writes_outputs_and_merged_tex(runs, tmp_path):
    out, tex = tmp_path / "out", tmp_path / "coverage.tex"
    (tmp_path / "out").mkdir()
    (out / "ask_act.json").write_text(json.dumps({"meta": {"git": "abc", "generated": "now"},
                                                  "tex": {"cov/ask/jev/ask_now/acc": "64.4"}}))
    assert G.main(["--runs", str(runs), "--out-dir", str(out), "--tex", str(tex)]) == 0
    t = tex.read_text()
    assert r"\defres{cov/ask/jev/ask_now/acc}{64.4}" in t                 # the other analysis's keys are kept
    assert r"\defres{cov/gate/tool/lite_jev/vetoed_wrong}{11}" in t
    assert r"\defres{cov/gate/tool/lite_jev/recall}{91.7}" in t
    assert "cov/gate/tool/flash_jev/recall" not in t                     # 1 wrong proposal: no recall key
    assert r"\defres{cov/gate/stop/lite_jev/first/vetoed_right}{1}" in t
    assert r"\defres{cov/gate/tool/dm/state_changing/wrong}{0}" in t
    md = (out / "gates.md").read_text()
    assert "What these logs can and cannot show" in md and "sha256" in md


def test_missing_arm_fails(tmp_path):
    assert G.main(["--runs", str(tmp_path / "none"), "--out-dir", str(tmp_path / "o"),
                   "--tex", str(tmp_path / "c.tex")]) == 1
    assert not (tmp_path / "o").exists()


def test_tex_merge_refuses_conflicting_keys():
    a = ("ask_act.json", {"meta": {}, "tex": {"cov/x": "1"}})
    b = ("gates.json", {"meta": {}, "tex": {"cov/x": "2"}})
    with pytest.raises(ValueError):
        CT.to_tex([a, b])
    assert r"\defres{cov/x}{1}" in CT.to_tex([a, ("gates.json", {"meta": {}, "tex": {"cov/x": "1"}})])
