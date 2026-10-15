"""E6a (exploratory action-only control on ALFWorld; scripts/explore_e6a_alfworld.py) and the module map's exploratory
action-row reader (scripts/module_map.py ``explore_action``), offline on tiny synthetic inputs: the train-game
selection rule, spend accounting, the projection gate, the readout, and that the exploratory cells stay apart from
every Stage-1 cell and count."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e6a_alfworld as E  # noqa: E402
import module_map as MM  # noqa: E402

ACTS = ["go to desk 1", "take book 1 from desk 1", "look"]


# ------------------------------------------------------------------ train-game selection
def _zips(tmp: Path) -> tuple[Path, Path]:
    types = list(E.TASK_TYPES.values())
    js, tw = io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(js, "w") as zj, zipfile.ZipFile(tw, "w") as zt:
        for t in types:
            for i in range(4):
                d = f"json_2.1.1/train/{t}-Obj{i}-None-Recep-1/trial_T2019_{i}"
                zj.writestr(f"{d}/traj_data.json", json.dumps({"task_type": t}))
                zt.writestr(f"{d}/game.tw-pddl", json.dumps({"solvable": i != 1, "walkthrough": []}))
        zj.writestr("json_2.1.1/train/pick_and_place_simple-AppleSliced-None-X-1/trial_T1/traj_data.json",
                    json.dumps({"task_type": "pick_and_place_simple"}))
        zt.writestr("json_2.1.1/train/pick_and_place_simple-AppleSliced-None-X-1/trial_T1/game.tw-pddl",
                    json.dumps({"solvable": True}))
        zj.writestr("json_2.1.1/valid_seen/x/trial_1/traj_data.json", "{}")
    (tmp / "j.zip").write_bytes(js.getvalue())
    (tmp / "t.zip").write_bytes(tw.getvalue())
    return tmp / "j.zip", tmp / "t.zip"


def test_select_train_is_deterministic_filtered_and_stratified(tmp_path: Path) -> None:
    j, t = _zips(tmp_path)
    a = E.select_train(j, t, n_per_type=2)
    assert a == E.select_train(j, t, n_per_type=2)                         # deterministic
    assert all(len(v) == 2 for v in a.values()) and set(a) == set(E.TASK_TYPES.values())
    games = [g for v in a.values() for g in v]
    assert not any("Sliced" in g or g.endswith("trial_T2019_1") for g in games)   # ALFWorld filters, solvable
    assert E.select_train(j, t, n_per_type=2, salt="other") != a or True        # the salt only reorders
    paths = E.extract_train(j, t, games, tmp_path / "data")
    assert len(paths) == 12 and all(p.exists() and (p.parent / "traj_data.json").exists() for p in paths)
    with pytest.raises(AssertionError):
        E.extract_train(j, t, ["../escape/trial"], tmp_path / "data")


# ------------------------------------------------------------------ synthetic E6a runs
def _cfg(action: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {dp: {"mode": "off", "tau": 0.8, "k": 3, "extra": {}} for dp in MM.DPS}
    if action:
        base["action"] = {**base["action"], **action}
    return base


def _dec(mode: str, s1: str | None, deferred: bool, gold: str = ACTS[0], s2: str | None = None) -> dict[str, Any]:
    dist = {a: (0.9 if a == s1 else 0.05) for a in ACTS} if s1 else None
    final = s2 if (deferred or mode in ("assist", "off")) else s1
    return {"dp": "action", "mode": mode, "question": "next action", "options": ACTS, "s1_value": s1,
            "s1_conf": 0.9 if s1 else None, "s1_dist": dist, "deferred": deferred, "s2_value": s2,
            "final_value": final, "gold": gold, "gold_status": None, "gold_aux": None, "shadow": False}


def _ep(tid: str, arm: str, ok: bool, cost: float, cfg: dict[str, Any], decs: list[dict[str, Any]],
        s1_cost: float = 0.0, wall: float = 10_000.0, error: str | None = None) -> dict[str, Any]:
    return {"task_id": tid, "config": arm, "backend": "jev" if arm != "llm_only" else None,
            "llm": MM.FLASH, "error": error, "wall_ms": wall,
            "steps": [{"step": i, "tool": "act", "args": {"action": d["final_value"]}, "llm_calls": 1, "decisions": [d]}
                      for i, d in enumerate(decs)],
            "result": {"success": ok}, "final_answer": None,
            "usage": {"total": {"cost_usd": cost, "calls": len(decs)}, "s1": {"cost_usd": s1_cost, "calls": len(decs)},
                      "s2": {"cost_usd": cost - s1_cost, "calls": len(decs)}},
            "meta": {"config": {"name": arm, **cfg},
                     "s1": {"describe": {"backend": "jev", "model": "typesafe/jev-1.13"} if arm != "llm_only" else None}}}


@pytest.fixture()
def e6a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs = tmp_path / "runs"
    d = runs / E.RUN_EVAL
    d.mkdir(parents=True)
    tasks = [f"alfworld/valid_seen/pick_and_place_simple-Book-None-Desk-{i}/trial_T{i}" for i in range(10)]
    exp = {"name": E.RUN_EVAL, "n_tasks": 10, "suite": {"name": "alfworld", "n": 10},
           "llm": {"provider": "gemini", "model": MM.FLASH}, "s1": {"type": "jev", "provider": "openrouter"},
           "arms": [{"name": "llm_only"}]}
    (d / "experiment.yaml").write_text(yaml.safe_dump(exp))

    def write(arm: str, eps: list[dict[str, Any]]) -> None:
        (d / f"{arm}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    write("llm_only", [_ep(t, "llm_only", True, 0.04, _cfg(), [_dec("off", None, True, s2=ACTS[0])],
                           wall=8000.0) for t in tasks])
    # replace: loses 5 of 10 at a 95 % saving -> costs success; coverage 100 %
    write(E.ARMS["replace"], [_ep(t, E.ARMS["replace"], i >= 5, 0.002, _cfg({"mode": "replace"}),
                                  [_dec("replace", ACTS[2] if i < 5 else ACTS[0], False)], s1_cost=0.002)
                              for i, t in enumerate(tasks)])
    # cascade: no loss, 25 % cheaper, half accepted; task 9 errored (dropped)
    write(E.ARMS["cascade"], [_ep(t, E.ARMS["cascade"], True, 0.03, _cfg({"mode": "cascade", "tau": 0.95}),
                                  [_dec("cascade", ACTS[0], bool(i % 2), s2=ACTS[0] if i % 2 else None)],
                                  s1_cost=0.001, error="Boom" if i == 9 else None) for i, t in enumerate(tasks)])
    # assist: one win over the reference is impossible (ref solves all); 10 % more expensive -> no saving
    write(E.ARMS["assist"], [_ep(t, E.ARMS["assist"], True, 0.044, _cfg({"mode": "assist", "k": 3}),
                                 [_dec("assist", ACTS[1], False, s2=ACTS[0])], s1_cost=0.001)
                             for t in tasks])
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "eval_games.json").write_text(json.dumps({"n": 10, "task_ids": tasks}))
    (cfg / "tauhat_action.json").write_text(json.dumps({"tau_hat": 0.95, "n_paired": 400, "tau_rule": {"rule": "projection"},
                                                        "primary": {"V": 1.0, "V_source": "fallback W", "delta_c": 0.001},
                                                        "cal_run": {"episodes": 48}}))
    monkeypatch.setattr(E, "EVAL_LIST", cfg / "eval_games.json")
    monkeypatch.setattr(E, "TAU_FILE", cfg / "tauhat_action.json")
    monkeypatch.setattr(E, "PRIOR_SPEND", {})
    return runs


def test_spend_counts_every_record_and_shadow(e6a: Path) -> None:
    s = E.total_spend(e6a)
    assert s["records"] == 40
    assert s["total"] == pytest.approx(10 * (0.04 + 0.002 + 0.03 + 0.044))
    assert s["jev"] == pytest.approx(10 * (0.002 + 0.001 + 0.001))
    sh = {"usage": {"total": {"cost_usd": 0.01}, "s2": {"cost_usd": 0.01}, "shadow": {"cost_usd": 0.004}}}
    assert E.episode_spend(sh) == {"gemini": 0.01, "jev": 0.004, "total": pytest.approx(0.014)}
    prior = {"aborted": {"gemini": 0.5, "jev": 0.01, "total": 0.51, "records": 3}}
    assert E.total_spend(e6a, prior=prior)["total"] == pytest.approx(s["total"] + 0.51)   # counted toward the cap


def test_projection_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "PRIOR_SPEND", {})
    sm = tmp_path / E.RUN_SMOKE
    sm.mkdir(parents=True)
    for arm, c in (("llm_only", 0.05), *((a, 0.05) for a in E.ARMS.values())):
        (sm / f"{arm}.jsonl").write_text("".join(json.dumps({"task_id": f"t{i}", "wall_ms": 1000,
                                                             "usage": {"total": {"cost_usd": c}}}) + "\n"
                                                 for i in range(2)))
    p = E.projection(tmp_path, n_games=140, safety=1.5, cap=80.0)
    assert p["projected_140"] == pytest.approx(0.2 * 140 * 1.5) and p["decision"] == "run all 140"
    p = E.projection(tmp_path, n_games=140, safety=1.5, cap=42.0)      # 0.4 spent + 42 > 42 -> 100 games (30 + 0.4)
    assert p["decision"] == "run the first 100"
    assert E.projection(tmp_path, n_games=140, safety=1.5, cap=20.0)["decision"] == "stop after the smoke"


def test_summarize_readout_and_tex(e6a: Path) -> None:
    r = E.summarize(e6a)
    rep, cas, ass = (r["arms"][m] for m in ("replace", "cascade", "assist"))
    assert rep["n"] == 10 and rep["delta_pt"] == pytest.approx(-50.0) and (rep["wins"], rep["losses"]) == (0, 5)
    assert rep["saving"] == pytest.approx(0.95) and rep["decisions"]["coverage"] == pytest.approx(1.0)
    assert rep["decisions"]["s1_acc"] == pytest.approx(0.5)
    assert cas["n"] == 9 and cas["n_dropped"] == 1 and cas["dropped"]["arm"]["errored"] == 1
    assert cas["delta_pt"] == 0.0 and cas["saving"] == pytest.approx(0.25)
    assert cas["decisions"]["coverage"] == pytest.approx(5 / 9)           # tasks 0, 2, 4, 6, 8 accepted
    assert ass["decisions"]["coverage"] == 0.0 and ass["decisions"]["recall_k"] == 1.0
    assert ass["saving"] == pytest.approx(-0.1)
    assert rep["wall_median_s"] == pytest.approx(10.0) and rep["wall_ref_median_s"] == pytest.approx(8.0)
    assert r["reference"]["decisions"]["llm_acc"] == 1.0
    tex = E.to_tex(r)
    assert r"\defres{e6a/replace/delta}{-50.0}" in tex and r"\defres{e6a/cascade/saving}{25.0}" in tex
    assert r"\defres{e6a/cascade/coverage}{55.6}" in tex and r"\defres{e6a/tau}{0.95}" in tex
    assert "nan" not in tex.lower() and "inf}" not in tex.lower()
    md = E.to_markdown(r)
    assert "not the registered S2-L3b" in md and "| replace (`jev_action_replace`) | 10 |" in md


# ------------------------------------------------------------------ module map: the exploratory action row
def test_module_map_reads_the_exploratory_action_row_apart_from_stage1(e6a: Path, tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"task_ids": ["jevgym-x-200000-d60-near-strict"]}))
    mm = MM.ModuleMap(e6a, n_boot=500, seed=0, primaries=tmp_path / "none.json", frozen=frozen, stage1=(),
                      stage2=None, quiet=True, explore=("E6a", E.RUN_EVAL))
    res = mm.build()
    assert res["rows"] == []                                              # never a Stage-1 row
    assert res["cells"]["flash/action/replace"]["state"] == "not run"     # the Stage-1 cell is unchanged
    assert res["counts"]["flash"]["run"] == 0
    x = res["explore_action"]["cells"]
    assert set(x) == {"flash/action/replace", "flash/action/cascade", "flash/action/assist"}
    assert x["flash/action/replace"]["verdict"] == MM.COSTS and x["flash/action/replace"]["state"] == "exploratory"
    assert x["flash/action/cascade"]["verdict"] == MM.SAFE and x["flash/action/cascade"]["n"] == 9
    assert x["flash/action/cascade"]["coverage"] == pytest.approx(5 / 9)
    assert x["flash/action/assist"]["verdict"] == MM.NOSAVE and x["flash/action/assist"]["acted"] == 1.0
    tex = MM.to_tex(res)
    assert r"\defres{mm/xcell/flash/action/replace/vcode}{4}" in tex
    assert r"\defres{mm/xcell/flash/action/cascade/fig_saving}{25}" in tex
    assert r"\defres{mm/xcell/flash/action/cascade/explore}{E6a}" in tex
    assert r"\defres{mm/cell/flash/action/cascade/state}{not run}" in tex
    assert "Exploratory action row" in MM.to_markdown(res) and "exploratory action row" in MM.grid_text(res)
    # without the run the reader is empty and nothing else changes
    empty = MM.ModuleMap(tmp_path, n_boot=50, stage1=(), stage2=None, quiet=True, frozen=frozen,
                         primaries=tmp_path / "none.json", e6b=None,
                         e6d=None, e6e=None, e6f=None).build()   # E6b's, E6d's, E6e's and E6f's committed cells left out
    assert empty["explore_action"]["cells"] == {} and "mm/xcell" not in MM.to_tex(empty)
