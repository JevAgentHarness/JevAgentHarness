"""scripts/module_map.py -- the descriptive module map -- on tiny synthetic run directories (offline, fast): the verdict
rule's boundaries, the cell assignment of multi-point arms, the paired comparison (drop rule, saving), replicates,
registered labels copied from primaries.json, the figure's cell choice and the generated TeX keys."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import module_map as MM  # noqa: E402

TOOLS = ["find_user", "get_orders", "ship_order", "finish"]


# ------------------------------------------------------------------ the verdict rule (docstring)
@pytest.mark.parametrize("delta,saving,acted,expected", [
    (-0.03, 0.50, 1.0, MM.COSTS),          # -3 pt is "costs success" (<=)
    (-0.031, -0.2, 1.0, MM.COSTS),         # a loss of 3 pt or more wins over "no saving"
    (-0.029, 0.10, 1.0, MM.MIXED),         # between -3 and -1 with a saving
    (-0.01, 0.001, 0.05, MM.SAFE),         # -1 pt, any saving, acted 5 %: takes over safely
    (0.0, 0.0, 1.0, MM.NOSAVE),            # zero saving is no saving
    (0.02, -0.3, 1.0, MM.NOSAVE),
    (0.0, 0.14, 0.049, MM.MIXED),          # saving without activity (policy NI with ~zero coverage)
    (0.0, 0.14, float("nan"), None),       # no decision model: no verdict
])
def test_verdict_rule(delta: float, saving: float, acted: float, expected: str | None) -> None:
    assert MM.verdict(delta, saving, acted) == expected


def test_rank_prefers_verdict_then_saving_or_delta() -> None:
    safe_small = MM.rank_key(MM.SAFE, 0.0, 0.05)
    safe_big = MM.rank_key(MM.SAFE, -0.005, 0.2)
    nosave_good = MM.rank_key(MM.NOSAVE, 0.01, -0.5)
    nosave_bad = MM.rank_key(MM.NOSAVE, -0.01, -0.01)
    costs = MM.rank_key(MM.COSTS, -0.05, 0.9)
    assert sorted([costs, nosave_bad, safe_small, nosave_good, safe_big]) == \
        [safe_big, safe_small, nosave_good, nosave_bad, costs]


def test_word_of_matches_resword() -> None:
    assert MM.word_of("NI") == "holds"
    assert MM.word_of("superior") == "holds"
    assert MM.word_of("NI: policy NI with zero coverage for Intern-Decision (A6.11 (c).4)") == "holds"
    assert MM.word_of("superiority not established (Holm step-down stopped earlier)") == "not established"
    assert MM.word_of("not NI as a conjunction: Jev NI; Intern-Decision inconclusive at this n") == "not established"
    assert MM.word_of("not run (A7.1 (h))") == "not run"


@pytest.mark.parametrize("setting,cells,joint", [
    ({"tool_select": {"mode": "cascade", "tau": 1.0}, "tool_params": {"mode": "cascade", "tau": 0.99}},
     [("tool_select", "cascade"), ("tool_params", "cascade")], True),
    ({"tool_select": {"mode": "replace"}, "tool_params": {"mode": "replace"}},
     [("tool_params", "replace"), ("tool_select", "replace")], True),
    ({"memory_write": {"mode": "trigger", "tau": 0.5}, "memory_read": {"mode": "assist", "k": 3}},
     [("memory_read", "assist"), ("memory_write", "trigger")], True),
    ({"memory_write": {"mode": "trigger", "tau": 0.5}, "memory_read": {"mode": "always"}},
     [("memory_write", "trigger")], False),
    ({"memory_write": {"mode": "llm"}, "memory_read": {"mode": "always"}}, [("memory_write", "llm")], False),
    ({"reflection": {"mode": "always"}}, [("reflection", "always")], False),
    ({"routing": {"mode": "trigger", "tau": 0.4}}, [("routing", "trigger")], False),
    ({}, [], False),
])
def test_arm_cells(setting: dict[str, Any], cells: list[tuple[str, str]], joint: bool) -> None:
    assert MM.arm_cells(setting) == (cells, joint)


def test_llm_group_routing_and_models() -> None:
    assert MM.llm_group({"routing": {"mode": "trigger"}}, MM.FLASH) == "route"
    assert MM.llm_group({}, MM.FLASH) == "flash"
    assert MM.llm_group({}, MM.LITE) == "lite"
    assert MM.llm_group({}, "other-model") is None


# ------------------------------------------------------------------ synthetic runs
def _cfg(**dps: dict[str, Any]) -> dict[str, Any]:
    base = {dp: {"mode": "off", "tau": 0.8, "k": 3, "extra": {}} for dp in MM.DPS}
    for dp, c in dps.items():
        base[dp] = {**base[dp], **c}
    return base


def _episode(tid: str, arm: str, success: bool, cost: float, cfg: dict[str, Any], backend: str | None,
             llm: str = MM.FLASH, decisions: list[dict[str, Any]] | None = None, error: str | None = None) -> dict:
    return {"task_id": tid, "config": arm, "backend": backend, "llm": llm, "error": error,
            "steps": [{"step": 0, "tool": "find_user", "args": {"name": "A"}, "llm_calls": 1,
                       "decisions": decisions or []}],
            "result": {"success": success}, "final_answer": "",
            "usage": {"total": {"cost_usd": cost}, "s1": {"cost_usd": 0.0, "calls": 1}, "s2": {"calls": 1}},
            "meta": {"config": {"name": arm, **cfg},
                     "s1": {"describe": {"backend": backend, "model": "jev-x"} if backend else None}}}


def _tool_dec(mode: str, s1: str, deferred: bool, gold: str = "find_user") -> dict[str, Any]:
    return {"dp": "tool_select", "mode": mode, "options": TOOLS, "s1_value": s1, "s1_conf": 0.9,
            "s1_dist": {t: (0.9 if t == s1 else 0.1 / 3) for t in TOOLS}, "deferred": deferred,
            "s2_value": gold if deferred else None, "final_value": gold if deferred else s1, "gold": gold,
            "gold_status": "ok", "gold_aux": {"acceptable_tools": [gold]}, "shadow": False}


def _write(d: Path, arm: str, eps: list[dict[str, Any]]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{arm}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))


@pytest.fixture()
def runs(tmp_path: Path) -> dict[str, Path]:
    tasks = [f"jevgym-clarify-{200000 + i}-d60-near-strict" for i in range(10)]
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"task_ids": tasks}))
    exp = {"name": "x", "n_tasks": 10, "llm": {"provider": "gemini", "model": MM.FLASH},
           "s1": {"type": "jev", "provider": "openrouter"}, "arms": [{"name": "llm_only"}]}
    off = _cfg()
    cas = _cfg(tool_select={"mode": "cascade", "tau": 1.0}, tool_params={"mode": "cascade", "tau": 0.99})
    rep = _cfg(tool_select={"mode": "replace"})
    e2a, e2c = tmp_path / "runs" / "e2a", tmp_path / "runs" / "e2c"
    for d in (e2a, e2c):
        d.mkdir(parents=True)
        (d / "experiment.yaml").write_text(yaml.safe_dump(exp))
        _write(d, "llm_only", [_episode(t, "llm_only", True, 0.04, off, None) for t in tasks])
    # registered cascade: no loss, 25 % cheaper, half of the tool decisions accepted; task 9 errored (dropped)
    _write(e2a, "jev_ts_cascade_tauhat", [
        _episode(t, "jev_ts_cascade_tauhat", True, 0.03, cas, "jev",
                 decisions=[_tool_dec("cascade", "find_user", deferred=bool(i % 2))],
                 error="Boom: infra" if i == 9 else None) for i, t in enumerate(tasks)])
    # replace: loses 4 of 10 tasks at a 75 % saving (costs success); E2c re-runs it (a replicate)
    for d, losses in ((e2a, 4), (e2c, 5)):
        _write(d, "jev_ts_replace", [
            _episode(t, "jev_ts_replace", i >= losses, 0.01, rep, "jev",
                     decisions=[_tool_dec("replace", "find_user", deferred=False)]) for i, t in enumerate(tasks)])
    prim = tmp_path / "primaries.json"
    prim.write_text(json.dumps({"primaries": {"C1F": {"decision": "NI", "jev": {
        "arm": "jev_ts_cascade_tauhat", "ref": "llm_only", "run": "runs/e2a", "ref_run": "runs/e2a"}}}}))
    return {"runs": tmp_path / "runs", "frozen": frozen, "primaries": prim}


def _build(runs: dict[str, Path], e6b: Path | None = None) -> dict[str, Any]:
    mm = MM.ModuleMap(runs["runs"], n_boot=500, seed=0, primaries=runs["primaries"], frozen=runs["frozen"],
                      stage1=(("E2a", "e2a"), ("E2c", "e2c")), stage2=None, quiet=True, e6b=e6b, e6d=None,
                      e6e=None, e6f=None)
    return mm.build()


def test_compare_drop_rule_and_saving(runs: dict[str, Path]) -> None:
    ref = MM.load_arm(runs["runs"] / "e2a" / "llm_only.jsonl")
    arm = MM.load_arm(runs["runs"] / "e2a" / "jev_ts_cascade_tauhat.jsonl")
    tasks = json.loads(runs["frozen"].read_text())["task_ids"]
    c = MM.compare(ref, arm, tasks, n_boot=500, seed=0)
    assert c["n"] == 9 and c["n_dropped"] == 1 and c["dropped"]["errored_arm"] == 1
    assert c["delta"] == 0.0 and c["delta_ci"] == [0.0, 0.0]
    assert c["saving"] == pytest.approx(0.25)
    assert c["saving_ci"][0] == pytest.approx(0.25) and c["saving_ci"][1] == pytest.approx(0.25)


def test_build_rows_cells_and_registered_labels(runs: dict[str, Path]) -> None:
    res = _build(runs)
    rows = {r["id"]: r for r in res["rows"]}
    assert set(rows) == {"jev_ts_cascade_tauhat", "jev_ts_replace", "jev_ts_replace-e2c"}
    cas = rows["jev_ts_cascade_tauhat"]
    assert cas["registered"][0]["label"] == "C1F" and cas["registered"][0]["word"] == "holds"
    assert cas["registered"][0]["same_reference"] is True
    assert cas["points"]["tool_select"]["coverage"] == pytest.approx(5 / 9)    # tasks 1, 3, 5, 7 deferred; 9 dropped
    assert cas["cell_verdicts"] == {"tool_select/cascade": MM.SAFE, "tool_params/cascade": MM.MIXED}  # no arg slot delegated
    rep, rep2 = rows["jev_ts_replace"], rows["jev_ts_replace-e2c"]
    assert rep["canonical"] and not rep2["canonical"] and rep2["replicate_of"] == "jev_ts_replace@E2a"
    assert rep["cmp"]["delta"] == pytest.approx(-0.4) and rep["cell_verdicts"]["tool_select/replace"] == MM.COSTS
    assert rep["points"]["tool_select"]["acc"] == pytest.approx(1.0)
    # the figure's cells: registered arm shown; undefined / equivalent / not-run states from harness.config
    c = res["cells"]["flash/tool_select/cascade"]
    assert c["state"] == "run" and c["why"] == "registered" and c["registered"] == "C1F" and c["vcode"] == 1
    # the triggers at tool choice, tool arguments and action: undefined until E6f, implemented since (not Stage 1)
    for dp in ("tool_select", "tool_params", "action"):
        assert res["cells"][f"flash/{dp}/trigger"]["state"] == "not run"
        assert res["cells"][f"flash/{dp}/trigger"]["vcode"] == 0
    # the five replace cells E6b declared equivalent: implemented since E6e (not run in Stage 1), the cell they equal
    # kept (harness.config.EQUIVALENT_NOTES)
    eq = res["cells"]["flash/memory_write/replace"]
    assert eq["state"] == "not run" and eq["equiv"] == "trigger" and eq["vcode"] == 0
    assert eq["equiv_note"] == "at tau = 0.5"
    assert res["cells"]["flash/memory_read/replace"]["equiv"] == "assist"
    assert res["cells"]["flash/memory_read/replace"]["equiv_note"] == "with k = 1"
    assert res["cells"]["flash/stop/replace"]["equiv"] == "cascade"
    assert "equiv" not in res["cells"]["flash/tool_select/replace"]
    assert res["cells"]["flash/action/guard"]["state"] == "not run"             # implemented by E6c, not Stage 1
    assert res["cells"]["flash/action/guard"]["vcode"] == 0
    assert res["cells"]["flash/tool_params/assist"]["state"] == "not run"          # an E6b cell: not run in Stage 1
    assert res["cells"]["flash/tool_select/assist"]["state"] == "not run"
    assert res["cells"]["flash/action/replace"]["state"] == "not run"
    assert res["cells"]["lite/tool_select/cascade"]["state"] == "not run"
    m = res["modules"]["flash/tool_select"]
    assert m["verdict"] == MM.SAFE and m["mode"] == "cascade" and m["registered"] == "C1F"
    cnt = res["counts"]["flash"]
    assert cnt["valid"] == 40 and cnt["pairs"] == 40     # 15 before E6b + 9 (E6b) + 1 (E6c) + 7 (E6d) + 5 (E6e) + 3 (E6f)
    assert (cnt["undefined"], cnt["equivalent"], cnt["not_implemented"], cnt["same_rule"]) == (0, 0, 0, 5)


def test_corner_marks_only_colour_changes() -> None:
    row = {"cell_verdicts": {"stop/trigger": MM.NOSAVE},
           "unstable": {"stop/trigger": {"replicate a@E2c": MM.MIXED}}}
    assert MM.ModuleMap._fresh_or_rep(row, "stop/trigger") is None          # mixed and no saving share a colour
    row["unstable"]["stop/trigger"]["fresh tasks"] = MM.SAFE
    assert MM.ModuleMap._fresh_or_rep(row, "stop/trigger") == MM.SAFE


def test_tex_keys_and_grid(runs: dict[str, Path]) -> None:
    res = _build(runs)
    tex = MM.to_tex(res)
    assert r"\defres{mm/tool_select/cascade/jev_ts_cascade_tauhat/saving}{25.0}" in tex
    assert r"\defres{mm/tool_select/cascade/jev_ts_cascade_tauhat/delta}{+0.0}" in tex
    assert r"\defres{mm/tool_select/cascade/jev_ts_cascade_tauhat/reg}{C1F}" in tex
    assert r"\defres{mm/tool_select/replace/jev_ts_replace-e2c/status}{replicate}" in tex
    assert r"\defres{mm/cell/flash/tool_select/cascade/vcode}{1}" in tex
    assert r"\defres{mm/cell/flash/tool_select/cascade/fig_saving}{25}" in tex
    assert r"\defres{mm/cell/flash/tool_select/trigger/state}{not run}" in tex   # defined by E6f
    assert "/undef}" not in tex
    assert r"\defres{mm/cell/flash/reflection/replace/equiv}{trigger}" in tex
    assert r"\defres{mm/best/flash/tool_select/verdict}{takes over safely}" in tex
    assert "{nan}" not in tex.lower() and "{inf}" not in tex.lower()
    grid = MM.grid_text(res)
    assert "SAFE +0.0/25% C1F" in grid and "COSTS -40.0/75%" in grid
    js = json.dumps(MM._clean(res), default=MM._json_default)
    assert "NaN" not in js and "Infinity" not in js


def test_simulated_runs_are_refused(runs: dict[str, Path]) -> None:
    exp = yaml.safe_load((runs["runs"] / "e2a" / "experiment.yaml").read_text())
    exp["llm"]["provider"] = "simulated"
    (runs["runs"] / "e2a" / "experiment.yaml").write_text(yaml.safe_dump(exp))
    with pytest.raises(SystemExit):
        _build(runs)


def test_e6b_exploratory_cells_from_the_summary(runs: dict[str, Path], tmp_path: Path) -> None:
    """E6b's cells (results/explore/e6b_modes/summary.json) become mm/xcell keys of the Flash map (routing: both maps);
    no Stage-1 cell, count or module verdict changes."""
    base = _build(runs)
    summ = {"meta": {"run": "runs/e6b_modes"}, "cells": {
        "tp_assist": {"dp": "tool_params", "mode": "assist", "state": "exploratory", "reference": "llm_only",
                      "verdict": MM.NOSAVE, "cmp": {"delta": 0.0, "saving": -0.02, "n": 200, "wins": 1, "losses": 1},
                      "points": {"acted": 1.0, "coverage": 0.0, "acc": 0.9}},
        "rt_cascade": {"dp": "routing", "mode": "cascade", "state": "exploratory", "reference": "small_only",
                       "verdict": MM.SAFE, "cmp": {"delta": 0.01, "saving": 0.1, "n": 199, "wins": 3, "losses": 1},
                       "points": {"acted": 0.4, "coverage": 0.4, "acc": 0.8}},
        "mw_guard": {"dp": "memory_write", "mode": "guard", "state": "not run", "reference": "llm_only"}}}
    f = tmp_path / "summary.json"
    f.write_text(json.dumps(summ))
    res = _build(runs, e6b=f)
    x = res["explore_e6b"]["cells"]
    assert set(x) == {"flash/tool_params/assist", "flash/routing/cascade", "lite/routing/cascade"}
    assert x["flash/tool_params/assist"]["vcode"] == MM.VCODE[MM.NOSAVE] and x["flash/routing/cascade"]["vcode"] == 1
    assert res["cells"] == base["cells"] and res["counts"] == base["counts"] and res["modules"] == base["modules"]
    tex = MM.to_tex(res)
    assert r"\defres{mm/xcell/flash/tool_params/assist/vcode}{3}" in tex
    assert r"\defres{mm/xcell/flash/tool_params/assist/explore}{E6b}" in tex
    assert r"\defres{mm/xcell/lite/routing/cascade/fig_saving}{10}" in tex
    assert r"\defres{mm/cell/flash/tool_params/assist/vcode}{0}" in tex     # drawn as the exploratory tile
    assert "Exploratory cells E6b" in MM.to_markdown(res) and "exploratory E6b / E6d / E6e / E6f cells" in MM.grid_text(res)
    # E6d's summary (the same format) joins the same exploratory cell map
    e6d = tmp_path / "e6d.json"
    e6d.write_text(json.dumps({"meta": {"run": "runs/e6d_modes"}, "cells": {
        "rf_guard": {"dp": "reflection", "mode": "guard", "state": "exploratory", "reference": "llm_only",
                     "verdict": MM.NOSAVE, "cmp": {"delta": 0.0, "saving": -0.05, "n": 200, "wins": 0, "losses": 0},
                     "points": {"acted": 0.1, "coverage": 0.0, "acc": 0.7}}}}))
    mm = MM.ModuleMap(runs["runs"], n_boot=500, seed=0, primaries=runs["primaries"], frozen=runs["frozen"],
                      stage1=(("E2a", "e2a"), ("E2c", "e2c")), stage2=None, quiet=True, e6b=f, e6d=e6d, e6e=None, e6f=None)
    both = mm.build()
    xx = both["explore_e6b"]["cells"]
    assert xx["flash/reflection/guard"]["explore"] == "E6d" and xx["flash/tool_params/assist"]["explore"] == "E6b"
    assert r"\defres{mm/xcell/flash/reflection/guard/vcode}{3}" in MM.to_tex(both)
    # E6e's replace cells (the same format) join it too, each with the cell it equals
    e6e = tmp_path / "e6e.json"
    e6e.write_text(json.dumps({"meta": {"run": "runs/e6e_modes"}, "cells": {
        "st_replace": {"dp": "stop", "mode": "replace", "state": "exploratory", "reference": "llm_only",
                       "verdict": MM.COSTS, "cmp": {"delta": -0.05, "saving": 0.02, "n": 200, "wins": 1, "losses": 11},
                       "points": {"acted": 1.0, "coverage": 1.0, "acc": 0.8}},
        "rt_replace": {"dp": "routing", "mode": "replace", "state": "exploratory", "reference": "small_only",
                       "verdict": MM.NOSAVE, "cmp": {"delta": 0.02, "saving": -0.3, "n": 200, "wins": 9, "losses": 5},
                       "points": {"acted": 1.0, "coverage": 1.0, "acc": 0.6}}}}))
    mm = MM.ModuleMap(runs["runs"], n_boot=500, seed=0, primaries=runs["primaries"], frozen=runs["frozen"],
                      stage1=(("E2a", "e2a"), ("E2c", "e2c")), stage2=None, quiet=True, e6b=f, e6d=e6d, e6e=e6e, e6f=None)
    three = mm.build()
    x3 = three["explore_e6b"]["cells"]
    assert x3["flash/stop/replace"]["explore"] == "E6e" and x3["flash/stop/replace"]["equiv"] == "cascade"
    assert {"flash/routing/replace", "lite/routing/replace"} <= set(x3) and x3["flash/reflection/guard"]["explore"] == "E6d"
    assert three["cells"] == base["cells"] and three["counts"] == base["counts"]
    tex = MM.to_tex(three)
    assert r"\defres{mm/xcell/flash/stop/replace/vcode}{4}" in tex
    assert r"\defres{mm/xcell/flash/stop/replace/equiv}{cascade}" in tex
    assert r"\defres{mm/xcell/lite/routing/replace/equiv}{trigger}" in tex
    assert r"\defres{mm/cell/flash/stop/replace/vcode}{0}" in tex and r"\defres{mm/cell/flash/stop/replace/equiv}{cascade}" in tex
    assert "stop/replace (= cascade at tau = 0.5)" in MM.to_markdown(three)
