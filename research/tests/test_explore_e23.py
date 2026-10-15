"""E23 (exploratory: E11's checkpoint in ALFWorld's unseen rooms; scripts/explore_e23_alfworld_unseen.py,
scripts/aws/e23_alfworld_unseen.sh), offline: the configs against E11's (only the split changes; E11's tau-hat, no
refit), the committed game list against the registration (134 games, rooms disjoint from E11's training and calibration
games), the checkpoint check against E11's sha256 record, the overload pause and spend watchdog inputs, the gate, the
tariff, the non-inferiority rule against scripts/hard_ni.py, and the readout on fake records."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e11_alfworld as E11  # noqa: E402
import explore_e23_alfworld_unseen as E  # noqa: E402
import hard_ni  # noqa: E402


def test_configs_are_e11s_with_only_the_split_changed() -> None:
    e11 = {n: yaml.safe_load((E11.CFG_DIR / f"{n}.yaml").read_text()) for n in ("eval", "smoke")}
    e23 = {n: yaml.safe_load((E.CFG_DIR / f"{n}.yaml").read_text()) for n in ("eval", "smoke")}
    for n in ("eval", "smoke"):
        a, b = dict(e11[n]), dict(e23[n])
        assert b.pop("name") == ("e23_alfworld_unseen_eval" if n == "eval" else "e23_alfworld_smoke")
        a.pop("name")
        if n == "eval":
            assert b["suite"].pop("split") == "eval_out_of_distribution" and a["suite"].pop("split") == "eval_in_distribution"
            assert b["suite"].pop("n") == b.pop("n_tasks") == 134 and a["suite"].pop("n") == a.pop("n_tasks") == 140
        assert a == b, n
    fit = json.loads(E11.TAU_FILE.read_text())
    for n in ("eval", "smoke"):
        arms = {x["name"]: x for x in e23[n]["arms"]}
        assert list(arms) == [E.REF, E.ARMS["cascade"], E.ARMS["replace"]]
        assert arms[E.ARMS["cascade"]]["action"] == {"mode": "cascade", "tau": fit["deploy"]["tau"]} == \
            {"mode": "cascade", "tau": E.tau_hat()}
        assert e23[n]["s1_backends"]["e11"]["name"] == E.SERVED and e23[n]["s1_backends"]["e11"]["cache"] is None
    assert e23["smoke"]["suite"]["split"] == "train" and e23["smoke"]["suite"]["data_dir"] == "data/raw/alfworld_e6a"


def test_committed_list_is_the_registered_unseen_split() -> None:
    rec = json.loads(E.EVAL_LIST.read_text())
    ids = rec["task_ids"]
    assert rec["n"] == len(ids) == len(set(ids)) == 134
    assert rec["list_sha256"] == E._sha("\n".join(ids)) == E.REGISTERED_LIST_SHA
    assert all(t.startswith("alfworld/valid_unseen/") for t in ids)
    assert set(rec["scenes"]) == set(E.UNSEEN_SCENES)
    assert sorted(rec["pilot_contact"]) == sorted(E.PILOT_IDS)
    reg = (ROOT / "docs/PREREG_A7_STAGE2.md").read_text()
    assert "E23: the E11 checkpoint in ALFWorld's unseen rooms" in reg and E.REGISTERED_LIST_SHA[:16] in reg
    # the unseen rooms occur in none of E11's harvest games nor E6a's calibration games; valid_seen's mostly do
    h = json.loads(E11.HARVEST_LIST.read_text())
    train = [g for v in h["games"].values() for g in v] + E11.cal_games()
    train_scenes = {g.split("/")[0].rsplit("-", 1)[1] for g in train}
    assert not train_scenes & set(E.UNSEEN_SCENES)
    seen = json.loads(E11.EVAL_LIST.read_text())["task_ids"]
    assert sum(E.scene_of(t) in train_scenes for t in seen) > 0.9 * len(seen)


def test_verify_ckpt_against_e11_record(tmp_path: Path) -> None:
    ck = tmp_path / "ck"
    ck.mkdir()
    for i, f in enumerate(E.CKPT_FILES):
        (ck / f).write_bytes(f"content {i}".encode())
    assert not E.verify_ckpt(ck)["ok"]                                                  # not E11's bytes
    rec = tmp_path / "rec.txt"
    rec.write_text("".join(f"{E.C.sha256_file(ck / f)}  /opt/x/{f}\n" for f in E.CKPT_FILES))
    assert E.verify_ckpt(ck, rec)["ok"]
    (ck / "calibration.json").write_bytes(b"changed")
    assert not E.verify_ckpt(ck, rec)["ok"]
    want = E.ckpt_record()
    assert set(E.CKPT_FILES) <= set(want) and want["adapter_model.safetensors"].startswith("53c5893b")


def _dec(value: str, gold: str | None, conf: float, deferred: bool = False, mode: str = "cascade") -> dict[str, Any]:
    return {"dp": "action", "mode": mode, "s1_value": value, "s1_conf": conf, "deferred": deferred,
            "s2_value": "x" if deferred else None, "final_value": "x" if deferred else value, "gold": gold,
            "shadow": False}


def _ep(task: str, cost: float, ok: bool = True, err: str | None = None, wall: float = 10_000.0,
        decs: list[dict[str, Any]] | None = None, s2_calls: int = 0) -> dict[str, Any]:
    return {"task_id": task, "config": "x", "steps": [{"decisions": decs or []}], "result": {"success": ok},
            "error": err, "wall_ms": wall,
            "usage": {"total": {"cost_usd": cost}, "s2": {"cost_usd": cost, "calls": s2_calls},
                      "s1": {"cost_usd": 0.0, "calls": len(decs or [])}}}


def test_overload_detection_errored_and_spend(tmp_path: Path) -> None:
    assert E.overload_error({"error": "HTTP 503: The model is overloaded"})
    assert E.overload_error({"error": "429 RESOURCE_EXHAUSTED"})
    assert not E.overload_error({"error": "ValueError: bad"}) and not E.overload_error({"error": None})
    run = tmp_path / E.RUN_EVAL
    run.mkdir()
    (run / f"{E.REF}.jsonl").write_text("\n".join(json.dumps(x) for x in (
        _ep("t0", 0.02), _ep("t1", 0.01, err="503 overloaded"), _ep("t1", 0.02), _ep("t2", 0.0, err="boom"))) + "\n")
    assert E.overload_count(run) == 1 and E.errored_tasks(run) == 1
    log = tmp_path / "probe.txt"
    log.write_text("# head\n2026 x 6/6 ok  CLEAN  (logged $0.00010)\n2026 y 5/6 ok  (logged $0.00020)\n")
    s = E.total_spend(tmp_path, log)
    assert s["total"] == pytest.approx(0.05 + 0.0003) and s["records"] == 4


def test_projection_gate_and_tariff(tmp_path: Path) -> None:
    sm = tmp_path / E.RUN_SMOKE
    sm.mkdir(parents=True)
    for arm, c in ((E.REF, 0.02), (E.ARMS["cascade"], 0.001), (E.ARMS["replace"], 0.0)):
        (sm / f"{arm}.jsonl").write_text("\n".join(json.dumps(_ep(f"t{i}", c)) for i in range(2)) + "\n")
    log = tmp_path / "none.txt"
    p = E.projection(tmp_path, log=log)
    assert p["decision"] == "run all 134" and p["projected_134"] == pytest.approx(0.021 * 134 * 1.5)
    assert E.projection(tmp_path, cap=0.042 + 0.021 * 100 * 1.5 + 0.001, log=log)["decision"] == "run the first 100"
    assert E.projection(tmp_path, cap=0.05, log=log)["decision"] == "stop after the smoke"
    ev = tmp_path / E.RUN_EVAL
    ev.mkdir()
    for arm in E.GPU_ARMS:
        (ev / f"{arm}.jsonl").write_text("\n".join(json.dumps(_ep(f"t{i}", 0.0)) for i in range(10)) + "\n")
    w = tmp_path / "eval_window.json"
    w.write_text(json.dumps({"window_start": "2026-10-08T00:00:00Z", "window_end": "2026-10-08T00:30:00Z",
                             "instance_type": "g5.2xlarge"}))
    tf = E.tariff(tmp_path, w)
    assert tf["episodes"] == 20 and tf["q_usd_per_episode"] == pytest.approx(1.212 * 0.5 / 20)


def test_ni_rule_is_hard_ni_for_e11s_row() -> None:
    keys = hard_ni.read_keys(ROOT / "paper/generated")
    if not all(k in keys for k in hard_ni.ROWS["e11"]):
        pytest.skip("E11's keys are not generated")
    w, l, n = (hard_ni._int(keys[k]) for k in hard_ni.ROWS["e11"])
    ours = E.ni_block(w, l, n)
    theirs = hard_ni.compute({k: keys[k] for k in hard_ni.ROWS["e11"]})["e11"]
    assert ours["p"] == pytest.approx(theirs["p"]) and (ours["verdict"] == "non-inferior") == (theirs["ni"] == "yes")
    assert ours["tango_ci_pt"] == pytest.approx([theirs["lo"], theirs["hi"]])
    assert hard_ni.ROWS["e23"] == ("e23/cascade/wins", "e23/cascade/losses", "e23/cascade/n")


def test_readout_on_fake_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = json.loads(E.EVAL_LIST.read_text())["task_ids"][:6]
    lst = tmp_path / "list.json"
    lst.write_text(json.dumps({"n": 6, "task_ids": ids, "scenes": {"10": 6}}))
    monkeypatch.setattr(E, "EVAL_LIST", lst)
    run = tmp_path / "runs" / E.RUN_EVAL
    run.mkdir(parents=True)
    ref = [_ep(t, 0.02, ok=(i != 0), s2_calls=10) for i, t in enumerate(ids)]
    cas = [_ep(t, 0.0, ok=(i != 1), decs=[_dec("a", "a", 0.9), _dec("b", "a", 0.5), _dec("c", "c", 0.05, True)])
           for i, t in enumerate(ids)]
    rep = [_ep(t, 0.0, ok=True, decs=[_dec("a", "b", 0.9, mode="replace")]) for t in ids]
    for name, eps in ((E.REF, ref), (E.ARMS["cascade"], cas), (E.ARMS["replace"], rep)):
        (run / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in eps) + "\n")
    box = tmp_path / "box"
    box.mkdir()
    (box / "eval_window.json").write_text(json.dumps({"window_start": "2026-10-08T00:00:00Z",
                                                      "window_end": "2026-10-08T00:06:00Z",
                                                      "instance_type": "g5.4xlarge", "max_tasks": None}))
    r = E.summarize(tmp_path / "runs", box)
    c, p = r["arms"]["cascade"], r["arms"]["replace"]
    assert c["n"] == 6 and c["wins"] == 1 and c["losses"] == 1 and c["success"] == pytest.approx(5 / 6)
    assert c["taken"]["n_taken"] == 12 and c["taken"]["n_wrong"] == 6 and c["taken"]["n_deferred"] == 6
    assert p["taken"]["n_wrong"] == 6 and p["wins"] == 1 and p["losses"] == 0
    q = 1.624 * 0.1 / 12
    assert r["tariff"]["q_usd_per_episode"] == pytest.approx(q)
    assert c["saving"] == pytest.approx(1 - q / 0.02)
    assert c["ni"]["verdict"] in ("non-inferior", "not shown") and 0 <= c["ni"]["p"] <= 1
    keys = E.tex_keys(r)
    assert keys["e23/cascade/n"] == "6" and keys["e23/cascade/delta"] == "+0.0" and keys["e23/replace/delta"] == "+16.7"
    assert keys["e23/cascade/ni/verdict"] in ("non-inferior", "not shown") and "e23/type/simple/n" in keys
    assert "e23/seen/cascade/ni/verdict" in keys and "e23/seen_change/cascade" in keys
    md = E.to_markdown(r)
    assert "seen (E11)" in md and "unseen" in md


def test_driver_syntax_and_forbidden_hosts() -> None:
    p = ROOT / "scripts/aws/e23_alfworld_unseen.sh"
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0
    text = p.read_text()
    forb = set(text.split('FORBIDDEN_IIDS="', 1)[1].split('"', 1)[0].split())
    assert {"<instance-id>", "<instance-id>", "<instance-id>", "<instance-id>",
            "<instance-id>"} <= forb
    assert "GUARD_MIN=${GUARD_MIN:-230}" in text and "CAP_USD=${CAP_USD:-15}" in text
    assert "checkpoints/e11_alfworld" in text and "verify-ckpt" in text and "--calibration \"$CK/calibration.json\"" in text
    assert "train_decision" not in text and "calibrate" not in text.replace("--calibration", "")
