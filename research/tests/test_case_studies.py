"""scripts/case_studies.py (offline): the three selection rules on tiny synthetic trajectories (seed / game-id order,
the first qualifying task), who decided each choice of a step (a cascade decision below its threshold and free text go
to the LLM's lane, a REPLACE run's arguments are the LLM's), the runner-up option, abridging, the row layout, and that
every key the section and its figures read exists in paper/generated/cases.tex. With the stored logs present, the
generated file is rebuilt from runs/ and must equal the committed one, and its numbers are re-read from the raw logs."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import case_studies as CS  # noqa: E402

GEN = ROOT / "paper/generated/cases.tex"
SECTION = ROOT / "paper/sections/app_cases.tex"
FIGS = ROOT / "paper/figures/fig_cases.tex"
OPTS = ["find_user", "get_orders", "refund", "ask_user", "finish"]
WRITES = {"refund", "cancel_order", "ship_order"}


def _ts(s1: str, conf: float, at: list[str], mode: str = "cascade", deferred: bool = False) -> dict:
    dist = {o: 0.0 for o in OPTS}
    dist[s1] = conf
    rest = [o for o in OPTS if o != s1]
    dist[rest[0]] = 1.0 - conf
    return {"dp": "tool_select", "mode": mode, "options": list(OPTS), "s1_value": s1, "s1_conf": conf, "s1_dist": dist,
            "deferred": deferred, "gold": at[0], "gold_status": "ok", "gold_aux": {"v": 1, "acceptable_tools": at},
            "state": "TASK: Refund Ann Lee's order.\n\nCURRENT OBSERVATION: ..."}


def _arg(name: str, value: str, conf: float | None, mode: str = "cascade", deferred: bool = False) -> dict:
    q = (f"Which value should the argument `{name}` of tool `x` take?" if mode != "generate"
         else f"free-text argument `{name}` of tool `x`")
    return {"dp": "tool_params", "mode": mode, "question": q, "s1_value": None if mode == "generate" else value,
            "s1_conf": conf, "deferred": deferred, "final_value": value}


def _ep(task: str, steps: list[dict], success: bool) -> dict:
    return {"task_id": task, "error": None, "result": {"success": success}, "steps": steps, "calls": [],
            "usage": {"total": {"cost_usd": 0.0}}, "wall_ms": 1000.0}


def _step(i: int, tool: str, decisions: list[dict], args: dict | None = None, obs: str = "", llm: int = 0) -> dict:
    return {"step": i, "tool": tool, "args": args or {}, "decisions": decisions, "observation": obs, "llm_calls": llm}


# ---------------------------------------------------------------- helpers
def test_runner_up_and_formatting():
    d = _ts("refund", 0.7, ["ask_user"])
    assert CS.runner_up(d) == ("find_user", pytest.approx(0.3))
    assert CS.fp(0.004) == "$<$0.01" and CS.fp(0.0) == "0.00" and CS.fp(0.9871) == "0.99"
    assert CS.fp(0.99931, 3) == "0.999"
    text, cut = CS.abridge("Could you please specify the shipping speed for the order?", 30)
    assert cut and text.endswith("…") and len(text) <= 31 and " " not in text[-2:]
    assert CS.abridge("short", 30) == ("short", False)
    assert CS.tex("ask_user 5% $x") == r"ask\_user 5\% \$x"
    assert CS.seed_of("jevgym-clarify-300026-d60-near-strict") == 300026


# ---------------------------------------------------------------- who decided what
def test_step_row_cascade_lanes():
    st = _step(0, "ask_user", [_ts("ask_user", 0.96, ["ask_user", "find_user"]),
                               _arg("missing", "shipping_speed", 0.993, deferred=True),
                               _arg("question", "Which speed would you like for your order today, please?", None,
                                    mode="generate", deferred=True)],
               obs="Customer: Express shipping, please.", llm=1)
    row = CS.step_row(st, WRITES, "take", "tool_select")
    assert row["by"] == "dm" and row["conf"] == pytest.approx(0.96) and row["ok"] is True and not row["must"]
    assert row["m"] == [] and [x["k"] for x in row["l"]] == ["missing", "question"]
    assert row["l"][0]["c"] == pytest.approx(0.993) and not row["l"][0]["free"] and row["l"][1]["free"]
    assert row["obs"] == "Express shipping, please." and row["obsk"] == "user" and row["llm_calls"] == 1
    st2 = _step(1, "refund", [_ts("refund", 0.97, ["refund"]), _arg("order_id", "ORD-1", 1.0)],
                obs="Irreversible error: refund({'order_id': 'ORD-1'}) was applied incorrectly.")
    row2 = CS.step_row(st2, WRITES, "take", "tool_select")
    assert row2["irr"] and [x["v"] for x in row2["m"]] == ["ORD-1"] and row2["l"] == []
    assert row2["obsk"] == "err" and "(…)" in row2["obs"]


def test_step_row_replace_and_deferred_tool():
    st = _step(2, "refund", [_ts("refund", 0.98, ["ask_user"], mode="replace")],
               args={"order_id": "ORD-3726", "method": "original_payment"}, llm=1)
    row = CS.step_row(st, WRITES, "ask", "tool_select")
    assert row["by"] == "dm" and row["ok"] is False and row["must"] and row["irr"]
    assert [(x["k"], x["free"]) for x in row["l"]] == [("order_id", False), ("method", False)]
    st2 = _step(0, "find_user", [_ts("get_orders", 0.3, ["find_user"], deferred=True)])
    assert CS.step_row(st2, WRITES, "take", "tool_select")["by"] == "llm"


# ---------------------------------------------------------------- the rules
def test_select_take_first_seed_with_a_handover():
    no_defer = [_step(0, "find_user", [_ts("find_user", 0.9, ["find_user"]), _arg("name", "Ann", 1.0)])]
    defer = [_step(0, "find_user", [_ts("find_user", 0.9, ["find_user"]), _arg("name", "Ann", 0.5, deferred=True)])]
    run = {"jevgym-a-300003-x": _ep("jevgym-a-300003-x", defer, True),
           "jevgym-a-300001-x": _ep("jevgym-a-300001-x", no_defer, True),
           "jevgym-a-300002-x": _ep("jevgym-a-300002-x", defer, False),
           "jevgym-a-300004-x": _ep("jevgym-a-300004-x", defer, True)}
    ref = {k: _ep(k, [], True) for k in run}
    assert CS.select_take(run, ref) == ("jevgym-a-300003-x", 3)
    ref["jevgym-a-300003-x"]["result"]["success"] = False
    assert CS.select_take(run, ref)[0] == "jevgym-a-300004-x"


def test_select_ask_first_irreversible_must_ask_error():
    def jev(at_third: list[str], tool: str = "refund") -> list[dict]:
        return [_step(0, "find_user", [_ts("find_user", 0.55, ["ask_user", "find_user"], mode="replace")]),
                _step(1, tool, [_ts(tool, 0.98, at_third, mode="replace")])]
    a = {"t-300010-x": _ep("t-300010-x", jev(["ask_user", "get_orders"]), False),   # not a must-ask state
         "t-300011-x": _ep("t-300011-x", jev(["ask_user"], tool="get_orders"), False),   # not a write
         "t-300012-x": _ep("t-300012-x", jev(["ask_user"]), False),
         "t-300013-x": _ep("t-300013-x", jev(["ask_user"]), False)}
    b = {k: _ep(k, [], True) for k in a}
    assert CS.select_ask(a, b, WRITES) == ("t-300012-x", 3)
    b["t-300012-x"]["result"]["success"] = False
    assert CS.select_ask(a, b, WRITES)[0] == "t-300013-x"


def test_select_alf_first_game_both_solved():
    run = {"g/b": _ep("g/b", [], True), "g/a": _ep("g/a", [], False), "g/c": _ep("g/c", [], True)}
    ref = {"g/a": _ep("g/a", [], True), "g/b": _ep("g/b", [], True), "g/c": _ep("g/c", [], True)}
    assert CS.select_alf(run, ref) == ("g/b", 2)


def test_layout_aligns_rows_across_runs():
    r1 = {"by": "dm", "alt": "x", "must": True, "m": [], "l": [{"free": False}] * 3, "obs": "o", "obs_lines": 1}
    r2 = {"by": "dm", "alt": "x", "must": False, "m": [], "l": [{"free": False}], "obs": "", "obs_lines": 0}
    case = {"runs": {"a": {"rows": [r1]}, "b": {"rows": [r2, r2]}, "ref": {"rows": None}}}
    lay = CS.layout(case, "ask")
    assert len(lay) == 2 and lay[0]["y"] == 0 and lay[1]["y"] == pytest.approx(-lay[0]["h"])
    assert lay[0]["h"] == pytest.approx(CS.TOOL_LINE + 2 * CS.PITCH + CS.PITCH + CS.ROW_PAD)
    assert lay[1]["h"] == pytest.approx(CS.TOOL_LINE + CS.PITCH + CS.ROW_PAD)


# ---------------------------------------------------------------- the paper files
def _defs() -> dict[str, str]:
    return dict(re.findall(r"\\defres\{([^}]*)\}\{(.*)\}%?$", GEN.read_text(), flags=re.M))


@pytest.mark.skipif(not GEN.exists(), reason="paper/generated/cases.tex not generated")
def test_every_key_the_paper_reads_exists():
    defs = _defs()
    sec = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in SECTION.read_text().splitlines())   # comments out
    used = set(re.findall(r"\\res\{(cases/[^}]*)\}", sec))
    assert used and not used - set(defs), used - set(defs)
    # each figure reads cases/<case>/... through \csC{...} (literal keys in the section) and, in figures/fig_cases.tex,
    # the per-step keys of every drawn run and the row layout of its case
    pics = re.findall(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", sec, flags=re.S)
    assert len(pics) == 3
    for pic in pics:
        case = re.search(r"\\def\\csCase\{(\w+)\}", pic).group(1)
        for key in re.findall(r"\\csC\{([^}\\]*)\}", pic):
            assert f"cases/{case}/{key}" in defs, (case, key)
        for run in re.findall(r"\\def\\csRun\{(\w+)\}", pic):
            n = int(defs[f"cases/{case}/{run}/n"])
            assert int(defs[f"cases/{case}/n_rows"]) >= n
            for leaf in ("success",):
                assert f"cases/{case}/{run}/{leaf}" in defs
            for i in range(1, n + 1):
                for leaf in ("tool", "by", "llm", "conf", "cnum", "alt", "altp", "ok", "irr", "must", "nm", "nl",
                             "obs", "obsk", "obsn"):
                    assert f"cases/{case}/{run}/s/{i}/{leaf}" in defs, (case, run, i, leaf)
                for j in range(1, int(defs[f"cases/{case}/{run}/s/{i}/nm"]) + 1):
                    assert all(f"cases/{case}/{run}/s/{i}/m/{j}/{x}" in defs for x in "kvc")
                for j in range(1, int(defs[f"cases/{case}/{run}/s/{i}/nl"]) + 1):
                    assert all(f"cases/{case}/{run}/s/{i}/l/{j}/{x}" in defs for x in ("k", "v", "c", "free"))
        for i in range(1, int(defs[f"cases/{case}/n_rows"]) + 1):
            assert all(f"cases/{case}/row/{i}/{x}" in defs for x in ("y", "h", "obs"))
    assert "\\tbd" not in GEN.read_text()


LOGS = [CS.run_path(s) for s in (CS.TAKE_RUN, CS.TAKE_REF, CS.ASK_A, CS.ASK_B, CS.ASK_REF, CS.ALF_RUN, CS.ALF_REF)]


@pytest.mark.skipif(not all(p.exists() for p in LOGS) or not CS.E11_SUMMARY.exists(),
                    reason="stored logs under runs/ not present")
def test_generated_file_matches_the_logs():
    res = CS.build()
    assert CS.to_tex(res) == GEN.read_text(), "run `python scripts/case_studies.py` to regenerate"
    defs = _defs()
    # re-read a few numbers straight from the raw records
    for case, run, spec in (("take", "run", CS.TAKE_RUN), ("take", "ref", CS.TAKE_REF), ("alf", "ref", CS.ALF_REF)):
        task = res["cases"][case]["task_id"]
        rec = CS.latest(CS.run_path(spec))[task]
        assert int(defs[f"cases/{case}/{run}/llm_calls"]) == sum(1 for c in rec["calls"] if c["component"] == "s2")
        assert defs[f"cases/{case}/{run}/wall"] == f"{rec['wall_ms'] / 1000:.1f}"
        assert int(defs[f"cases/{case}/{run}/success"]) == int(bool(rec["result"]["success"]))
    jev = CS.latest(CS.run_path(CS.ASK_A))[res["cases"]["ask"]["task_id"]]
    k = int(defs["cases/ask/first_wrong"])
    d = CS.first_decision(jev["steps"][k - 1], "tool_select")
    assert defs["cases/ask/first_wrong_conf"] == f"{d['s1_conf']:.2f}"
    assert d["gold_aux"]["acceptable_tools"] == ["ask_user"] and jev["steps"][k - 1]["tool"] in CS.write_tools()
    assert not jev["result"]["success"]
    prof = json.loads((CS.RUNS / CS.TAKE_RUN[0] / "profiles.json").read_text())[CS.TAKE_RUN[1]]["decision_points"]
    assert defs["cases/take/tau_arg"] == f"{prof['tool_params']['tau']:.3f}"
