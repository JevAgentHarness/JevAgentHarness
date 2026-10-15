"""scripts/primaries.py -- the family-level analysis of the 11 registered primaries -- on synthetic paired data whose
answer is known (offline, fast), plus the analysis-side fixes the dry run (scripts/dryrun_main_analysis.py) found in
the score card (drop rule; majority pass for several trials) and paper_numbers (NI-block formatting)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import primaries as P  # noqa: E402
from jevharness.bench import metrics as M  # noqa: E402


def _maps(n: int, wins: int, losses: int, both: int | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """(reference, arm) per-task binary outcomes over n tasks with the given wins (arm 1, ref 0) and losses."""
    both = n - wins - losses if both is None else both
    ref, arm = {}, {}
    for i in range(n):
        t = f"t{i:04d}"
        if i < wins:
            ref[t], arm[t] = 0.0, 1.0
        elif i < wins + losses:
            ref[t], arm[t] = 1.0, 0.0
        elif i < wins + losses + both:
            ref[t], arm[t] = 1.0, 1.0
        else:
            ref[t], arm[t] = 0.0, 0.0
    return ref, arm


# ------------------------------------------------------------------ the NI rule (A6.21; A6.14 / A6.17 superseded)
def test_identical_arm_zero_discordance_gives_the_loss_count_tail_plus_gamma():
    ref, arm = _maps(500, 0, 0)
    b = P.ni_block(ref, arm, 0.03, n_boot=200, seed=0)
    assert b["rule"] == "bb_ordered" and b["wins"] == 0 and b["losses"] == 0
    assert b["p_decisive"] == pytest.approx(0.97 ** 500 + 1e-4, rel=1e-6)   # A6.21: (1 - m)^n + gamma
    assert b["loss_rate_ub"] == pytest.approx(1 - 0.05 ** (1 / 500), rel=1e-6)
    assert math.isnan(b["mde"]) and "no discordant task" in b["mde_note"]
    assert P.reading(b, b["p_decisive"] <= 0.05 / 11) == "NI"


def test_clear_ni_pass_is_decided_by_the_certified_p():
    ref, arm = _maps(500, 25, 15)
    b = P.ni_block(ref, arm, 0.03, n_boot=500, seed=0)
    assert b["rule"] == "bb_ordered"
    assert b["p_decisive"] == pytest.approx(M.ni_certified_p(25, 15, 500, 0.03)["p"])
    assert b["p_decisive"] < 0.05 / 11                                         # passes even at the Holm floor
    # on ordinary counts the certified p agrees with the superseded A6.17 p_U to within the 1e-3 tolerance
    assert b["p_decisive"] == pytest.approx(M.ni_unconditional_p(25, 15, 500, 0.03)["p"], rel=2e-3)


def test_clear_loss_reads_as_a_loss_of_the_stated_size():
    ref, arm = _maps(500, 0, 60)
    b = P.ni_block(ref, arm, 0.03, n_boot=500, seed=0)
    assert b["rule"] == "bb_ordered"                                           # one rule for every outcome (A6.21)
    assert b["p_decisive"] > 0.5 and b["ci95"][1] < 0
    assert P.reading(b, False).startswith("not NI: loss of 12.0 pt")


def test_failed_ni_with_ci_covering_zero_is_inconclusive():
    ref, arm = _maps(200, 20, 22)
    b = P.ni_block(ref, arm, 0.03, n_boot=1000, seed=0)
    assert b["p_decisive"] > 0.05 and b["ci95"][0] < 0 < b["ci95"][1]
    assert P.reading(b, False) == "inconclusive at this n"


def test_superiority_sign_test_and_c5_paired_t():
    ref, arm = _maps(100, 12, 2)
    s = P.sign_block(ref, arm, n_boot=200, seed=0)
    assert (s["wins"], s["losses"]) == (12, 2)
    assert s["p_decisive"] == pytest.approx(M.sign_test_exact(12, 2)["p_wins"])
    d = {f"t{i}": (0.01 if i % 3 else -0.002) for i in range(60)}
    u = P.du_block(d, d, n_boot=200, seed=0)
    assert u["p_decisive"] == pytest.approx(M.paired_t_test(list(d.values()))["p_greater"]) and u["rule"] == "paired_t"
    assert u["p_sign_flip_descriptive"] == pytest.approx(
        M.sign_flip_test(list(d.values()), n_perm=20000, seed=0)["p_greater"])
    assert u["p_decisive"] < 0.01 and u["diff"] > 0


# ------------------------------------------------------------------ Holm (A5.3)
def test_holm_order_levels_and_step_down():
    ps = {"C8": 0.001, "C3": 0.004, "C1F": 0.006, "C7": 0.2, "C5": 0.007}
    h = P.holm(ps)
    assert [h[k]["rank"] for k in ("C8", "C3", "C1F", "C5", "C7")] == [1, 2, 3, 4, 5]
    assert h["C8"]["alpha_adj"] == pytest.approx(0.05 / 5) and h["C7"]["alpha_adj"] == pytest.approx(0.05)
    assert h["C8"]["reject"] and h["C3"]["reject"] and h["C1F"]["reject"] and h["C5"]["reject"]
    assert not h["C7"]["reject"] and h["C7"]["reached"]
    # a failure stops the step-down: C1F (0.006 > 0.05/3) fails, so C5 is retained although 0.007 <= 0.05/2
    h2 = P.holm({"C8": 0.001, "C1F": 0.02, "C5": 0.021, "C7": 0.9})
    assert h2["C8"]["reject"] and not h2["C1F"]["reject"] and h2["C1F"]["reached"]
    assert not h2["C5"]["reject"] and not h2["C5"]["reached"]


# ------------------------------------------------------------------ drop rule (§1, A6.9) and tau^2 majority pass
def _ep(task: str, success: bool, error: str | None = None, attempts: int = 1, trial: int | None = None) -> dict:
    return {"episode": task if trial is None else f"{task}#{trial}", "task": task, "trial": trial,
            "success": float(success), "cost": 0.01, "s1_cost": 0.0, "latency": 1.0, "llm_input_tokens": 100.0,
            "error": error, "attempts": attempts, "llm": "m", "backend": None}


def test_drop_rule_statuses_and_full_set_row():
    exp = ["a", "b", "c", "d", "e"]
    arm = {"episodes": {e["episode"]: e for e in [_ep("a", True), _ep("b", False, "503", attempts=3),
                                                  _ep("c", False, "503", attempts=1), _ep("e", True)]}}
    ref = {"episodes": {e["episode"]: e for e in [_ep(t, True) for t in exp]}}
    ta, tb = P.task_table(arm, exp), P.task_table(ref, exp)
    assert [ta[t]["status"] for t in exp] == ["ok", "errored", "retry_pending", "missing", "ok"]
    p = P.pair_tables(tb, ta, exp)
    assert p["n_pairs"] == 2 and p["n_dropped"] == 3 and p["keys"] == ["a", "e"]
    assert p["status_arm"] == {"ok": 2, "errored": 1, "retry_pending": 1, "missing": 1, "incomplete": 0}
    assert p["arm_y_full"] == {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 1.0}   # errored / missing = failure
    fl = P.Flags()
    P._status_flags(fl, "X", "arm", "x", p["status_arm"], 5)
    assert [f["level"] for f in fl.items] == ["ERROR", "ERROR", "WARN"]          # missing, retry pending; errored


def test_tau2_majority_pass_over_three_trials():
    eps = [_ep("airline/1", s, trial=k) for k, s in enumerate((1, 1, 0))] + \
          [_ep("airline/2", s, trial=k) for k, s in enumerate((1, 0, 0))] + \
          [_ep("airline/3", s, trial=k, error="infrastructure_error" if k == 2 else None) for k, s in enumerate((1, 1, 1))] + \
          [_ep("retail/4", 1, trial=0), _ep("retail/4", 1, trial=1)]
    tab = P.task_table({"episodes": {e["episode"]: e for e in eps}}, ["airline/1", "airline/2", "airline/3", "retail/4"],
                       trials=3, retry_rule=False)
    assert tab["airline/1"]["y"] == 1.0 and tab["airline/1"]["pass1"] == pytest.approx(2 / 3)
    assert tab["airline/2"]["y"] == 0.0
    assert tab["airline/3"]["status"] == "errored"            # tau^2 errors are final (no attempt count logged)
    assert tab["retail/4"]["status"] == "incomplete"


# ------------------------------------------------------------------ C7 selection (A5.4 (b), A6.9 #11)
def test_c7_choice_rules():
    base = {"llm_only": {"ni_95": True, "cost_savings": 0.0}}
    c1 = P.C7_C1_ARM
    # no NI System-1 configuration -> the C1 arm (and the fallback note when the C1 arm is not NI itself)
    sel, why, deg = P.choose_c7({**base, c1: {"ni_95": False, "cost_savings": 0.4},
                                 "jev_ts_assist_k3": {"ni_95": False, "cost_savings": 0.7}})
    assert sel == c1 and "no System-1 configuration is NI" in why and "fallback" in why and not deg
    # a unique best NI configuration
    sel, _, _ = P.choose_c7({**base, c1: {"ni_95": True, "cost_savings": 0.30},
                             "jev_ts_assist_k3": {"ni_95": True, "cost_savings": 0.55},
                             "jev_ts_cascade_tau0.9": {"ni_95": False, "cost_savings": 0.8}})
    assert sel == "jev_ts_assist_k3"
    # two within 2 pt of the best -> the C1 arm
    sel, why, _ = P.choose_c7({**base, c1: {"ni_95": True, "cost_savings": 0.30},
                               "jev_ts_assist_k3": {"ni_95": True, "cost_savings": 0.55},
                               "jev_ts_cascade_tau0.9": {"ni_95": True, "cost_savings": 0.54}})
    assert sel == c1 and "within 2 pt" in why
    # LLM-only the unique best: degenerate, resolved to the C1 arm and flagged
    sel, _, deg = P.choose_c7({**base, "jev_ts_guard": {"ni_95": True, "cost_savings": -0.10}})
    assert sel == c1 and deg


# ------------------------------------------------------------------ C4 opportunity rule, conjunction, labels
class _Args:
    def __init__(self, tmp: Path, **kw: Any) -> None:
        self.runs, self.run, self.tau2, self.e1 = str(tmp), None, None, None
        self.gate, self.c7_arm, self.c7_selection, self.tau2_secondary = str(P.GATE_FILE), None, None, None
        self.select_c7, self.out, self.n_boot, self.seed = False, str(tmp / "out"), 200, 0
        self.max_tasks, self.allow_simulated, self.allow_dev_tasks = None, True, True
        self.allow_partial_population, self.strict = False, False
        self.__dict__.update(kw)


def _fake_pair(ref: dict[str, float], arm: dict[str, float], name: str = "a", coverage: float = 0.3) -> dict[str, Any]:
    tab = lambda y: {t: {"status": "ok", "y": v, "cost": 0.01, "s1_cost": 0.0, "tokens": 1.0, "latency": 1.0}  # noqa: E731
                     for t, v in y.items()}
    keys = sorted(ref)
    st = {"ok": len(keys), "errored": 0, "retry_pending": 0, "missing": 0, "incomplete": 0}
    return {"n_expected": len(keys), "n_pairs": len(keys), "n_dropped": 0, "status_ref": st, "status_arm": st,
            "keys": keys, "ref_y": ref, "arm_y": arm, "ref_y_full": ref, "arm_y_full": arm, "arm": name, "ref": "r",
            "run": "x", "ref_run": "x", "tab_arm": tab(arm), "tab_ref": tab(ref), "s2_arm": "m", "s2_ref": "m",
            "coverage": {"ratio": coverage, "n": 100, "tool_coverage": coverage, "arg_coverage": coverage}}


@pytest.mark.parametrize("n_pos,insufficient", [(29, True), (30, False)])
def test_c4_enters_with_p_one_below_30_routing_positives(tmp_path, monkeypatch, n_pos, insufficient):
    an = P.Analysis(_Args(tmp_path))
    ref, arm = _maps(500, 30, 2)
    monkeypatch.setattr(an, "routing_positives", lambda: {"n_pos": n_pos, "n_neg": 400})
    monkeypatch.setattr(an, "jevgym_pair", lambda *a, **k: _fake_pair(ref, arm))
    c4 = an.c4()
    if insufficient:
        assert c4["p_decisive"] == 1.0 and c4["insufficient"]
        assert c4["jev"]["p_decisive"] < 1e-6                     # the sign test is still reported, descriptively
    else:
        assert c4["p_decisive"] == pytest.approx(M.sign_test_exact(30, 2)["p_wins"]) and not c4.get("insufficient")


def _conj(tmp_path, monkeypatch, co_primary: bool, jev: tuple[int, int], intern: tuple[int, int]) -> tuple:
    an = P.Analysis(_Args(tmp_path))
    pairs = {"E2a": _maps(500, *jev), "E2a_intern": _maps(500, *intern)}
    monkeypatch.setattr(an, "jevgym_pair", lambda where, run_key, *a, **k: _fake_pair(*pairs[run_key]))
    gate = {"co_primary": co_primary}
    C = an.conj_primary("C1F", gate)
    return an, C, gate


def test_conjunction_is_the_max_and_the_labels(tmp_path, monkeypatch):
    an, C, gate = _conj(tmp_path, monkeypatch, True, (30, 5), (20, 10))
    pj, pi = C["jev"]["p_decisive"], C["intern"]["p_decisive"]
    assert C["p_decisive"] == pytest.approx(max(pj, pi)) and pi > pj
    an._decide(C, {"rank": 1, "m": 11, "alpha_adj": 0.05 / 11, "p_holm": C["p_decisive"] * 11,
                   "reject": C["p_decisive"] <= 0.05 / 11, "reached": True}, gate)
    assert C["replication"] == "cross-backend replication" and C["decision"] == "NI"
    # one family fails: split
    an, C, gate = _conj(tmp_path, monkeypatch, True, (30, 5), (0, 40))
    an._decide(C, {"rank": 1, "m": 11, "alpha_adj": 0.05 / 11, "p_holm": 1.0, "reject": False, "reached": True}, gate)
    assert C["replication"].startswith("split (Jev passes)")
    assert C["intern"]["reading"].startswith("not NI: loss")
    # the gate demoted Intern-Decision: p = p_Jev and "Jev-specific confirmation"
    an, C, gate = _conj(tmp_path, monkeypatch, False, (30, 5), (0, 40))
    assert C["p_decisive"] == pytest.approx(C["jev"]["p_decisive"])
    an._decide(C, {"rank": 1, "m": 11, "alpha_adj": 0.05 / 11, "p_holm": 0.001, "reject": True, "reached": True}, gate)
    assert C["replication"] == "Jev-specific confirmation"


def test_c1_zero_coverage_reads_as_policy_ni(tmp_path, monkeypatch):
    an = P.Analysis(_Args(tmp_path))
    pairs = {"E2a": (*_maps(500, 0, 0), "j", 0.0), "E2a_intern": (*_maps(500, 30, 5), "i", 0.4)}
    monkeypatch.setattr(an, "jevgym_pair", lambda where, run_key, *a, **k: _fake_pair(*pairs[run_key]))
    gate = {"co_primary": True}
    C = an.conj_primary("C1L", gate)
    assert C["jev"]["coverage"] == 0.0 and C["intern"]["coverage"] == 0.4
    an._decide(C, {"rank": 1, "m": 11, "alpha_adj": 0.05 / 11, "p_holm": 0.001, "reject": True, "reached": True}, gate)
    assert C["replication"] == "cross-backend replication of policy NI"
    assert C["decision"].startswith("NI: policy NI with zero coverage for Jev")
    C2 = an.conj_primary("C2L", gate)                       # assist never delegates: coverage is not a C2 readout
    assert "coverage" not in C2["jev"]


def test_gate_verdict_file_says_co_primary(tmp_path):
    g = P.Analysis(_Args(tmp_path)).gate()
    assert g["candidate"] == "intern_4B" and g["co_primary"] is True


# ------------------------------------------------------------------ end to end on a tiny synthetic run tree
def _traj(task: str, arm: str, success: bool, llm: str, error: str | None = None) -> dict[str, Any]:
    u = {"calls": 1, "input_tokens": 100, "output_tokens": 5, "cost_usd": 0.01, "latency_ms": 1000.0}
    return {"task_id": task, "config": arm, "backend": "simulated", "llm": llm, "steps": [],
            "result": {"success": success}, "usage": {"total": u, "s2": u, "s1": {"cost_usd": 0.0, "calls": 0}},
            "error": error, "meta": {"config": {}}}


def _routing_steps(gold_pos: bool) -> list[dict[str, Any]]:
    """Two labelled routing shadow records (the A5.5 measurement-arm format): one positive when ``gold_pos``."""
    def rec(g: bool) -> dict[str, Any]:
        return {"dp": "routing", "mode": "shadow", "question": "Does the next step need the strong model?",
                "options": None, "s1_value": g, "s1_conf": 0.7, "s1_dist": {"yes": 0.7 if g else 0.3, "no": 0.3 if g else 0.7},
                "deferred": True, "gold": g, "gold_status": "ok", "state": "s", "shadow": True}
    return [{"step": 0, "decisions": [rec(gold_pos)]}, {"step": 1, "decisions": [rec(False)]}]


def _write_run(root: Path, cfg: str, n: int, outcome) -> Path:  # noqa: ANN001
    exp = yaml.safe_load((ROOT / "configs" / "experiments" / cfg).read_text())
    exp["suite"] = {**exp["suite"], "seed0": 0}
    exp["llm"] = {"provider": "simulated", "model": "gemini-3.5-flash"}
    d = root / exp["name"]
    d.mkdir(parents=True)
    (d / "experiment.yaml").write_text(yaml.safe_dump(exp))
    from jevharness.bench.runner import arm_task_counts
    from jevharness.bench.suites import load_suite

    kw = dict(exp["suite"])
    ids = [f().task_id for f in load_suite(kw.pop("name"), **kw)[:n]]
    for arm in arm_task_counts(exp):
        with open(d / f"{arm}.jsonl", "w") as f:
            for i, t in enumerate(ids):
                tr = _traj(t, arm, outcome(arm, i), "gemini-3.5-flash")
                if arm == "meta_shadow_s2lite":
                    tr["steps"] = _routing_steps(gold_pos=True)       # one routing positive per task
                f.write(json.dumps(tr) + "\n")
    return d


def test_end_to_end_family_on_synthetic_runs(tmp_path):
    n = 40
    for cfg in P.CONFIGS.values():   # every arm fails every 10th task, except the C3 control and C4's live arm
        _write_run(tmp_path, cfg, n, lambda arm, i: (i % 10 != 0) or arm.startswith("bm25") or arm == "jev_route_tau0.4")
    out = tmp_path / "out"
    rc = P.main(["--runs", str(tmp_path), "--out", str(out), "--n-boot", "200", "--max-tasks", str(n),
                 "--allow-simulated", "--allow-dev-tasks", "--strict"])
    res = json.loads((out / "primaries.json").read_text())
    assert list(res["primaries"]) == P.HOLM_ORDER and len(res["holm"]) == 11
    # no tau^2 run and no E1 run here: C7 and C8 stay in the family with p = 1 and ERROR flags; --strict exits 1
    assert res["primaries"]["C7"]["p_decisive"] == 1.0 and res["primaries"]["C8"]["p_decisive"] == 1.0
    assert rc == 1 and any(f["where"] == "C8" and f["level"] == "ERROR" for f in res["flags"])
    # identical arms: zero discordance everywhere -> decided by the A6.21 rule, n' = n
    c1 = res["primaries"]["C1F"]
    assert c1["jev"]["rule"] == "bb_ordered" and c1["jev"]["n_pairs"] == n and c1["intern"]["n_pairs"] == n
    # C4: 40 labelled routing positives in meta_shadow_s2lite (>= 30) -> tested: 4 wins, 0 losses vs small_only
    c4 = res["primaries"]["C4"]
    assert c4["opportunities"]["n_pos"] == n and c4["opportunities"]["n_neg"] == n
    assert c4["p_decisive"] == pytest.approx(M.sign_test_exact(4, 0)["p_wins"]) and c4["rule"] == "sign_exact"
    # C3: the BM25 control never fails -> 0 wins / 4 losses for Jev assist: not superior
    assert res["primaries"]["C3"]["jev"]["losses"] == 4 and res["primaries"]["C3"]["decision"].startswith("superiority not")
    assert res["meta"]["simulated"] and "SIM/primaries/C1F/p_decisive" in (out / "primaries.tex").read_text()
    assert "SIMULATED DRY RUN" in (out / "primaries.md").read_text()
    assert {"n": n, "margin": 0.03} == {k: res["ni_exact_size_settings"]["needed"][0][k] for k in ("n", "margin")}
    # every number traces to a file: provenance carries path + sha256 of every arm read
    arms = res["provenance"]["E2a"]["arms"]
    assert arms["jev_ts_cascade_tauhat"]["sha256"] and arms["llm_only"]["n_records"] == n
    # simulated inputs are refused without --allow-simulated
    rc2 = P.main(["--runs", str(tmp_path), "--out", str(tmp_path / "out2"), "--n-boot", "100", "--max-tasks", str(n),
                  "--allow-dev-tasks"])
    res2 = json.loads((tmp_path / "out2" / "primaries.json").read_text())
    assert any("simulated run(s)" in f["msg"] for f in res2["flags"]) and rc2 == 0


def test_evaluation_run_uses_the_frozen_task_list_and_registered_sizes(tmp_path):
    d = tmp_path / "e2a_jevgym_tool_decisions"
    d.mkdir()
    exp = yaml.safe_load((ROOT / "configs" / "experiments" / "jevgym_tool_select.yaml").read_text())
    (d / "experiment.yaml").write_text(yaml.safe_dump({**exp, "workers": 8}))   # what the runner writes (-o workers=8)
    fl = P.Flags()
    r = P.EpisodeRun("E2a", P.CONFIGS["E2a"], d, fl, None, allow_dev_tasks=False)
    frozen = json.loads(P.FROZEN_TASKS.read_text())
    assert r.frozen and r.task_ids == frozen["task_ids"] and not fl.items
    assert r.n_tasks("jev_ts_assist_k3") == 500 and r.n_tasks("embed_ts_assist_k3") == 200
    assert r.n_tasks("jev_ts_cascade_tau0.9") == 200 and not fl.items
    # an override of a registered size is an ERROR (the size is registered, A2 / A5)
    (d / "experiment.yaml").write_text(yaml.safe_dump({**exp, "n_tasks": 500}))
    r2 = P.EpisodeRun("E2a", P.CONFIGS["E2a"], d, fl, None, allow_dev_tasks=False)
    assert r2.n_tasks("embed_ts_assist_k3") == 200 and any("registered 200" in f["msg"] for f in fl.items)


def test_foreign_episodes_in_an_arm_file_are_an_error(tmp_path):
    n = 6
    for cfg in P.CONFIGS.values():
        _write_run(tmp_path, cfg, n, lambda arm, i: True)
    d = tmp_path / "e2a_jevgym_tool_decisions"
    with open(d / "llm_only.jsonl", "a") as f:     # a smoke run on other seeds written into the main run directory
        f.write(json.dumps(_traj("jevgym-clarify-999998-d60-near-strict", "llm_only", True, "m")) + "\n")
    an = P.Analysis(_Args(tmp_path, max_tasks=n))
    p = an.jevgym_pair("C1F", "E2a", "jev_ts_cascade_tauhat", "llm_only")
    assert p["n_pairs"] == n
    assert any(f["level"] == "ERROR" and "outside the run's task list" in f["msg"] for f in an.flags.items)


def test_dev_task_list_is_an_error_without_the_flag(tmp_path):
    _write_run(tmp_path, P.CONFIGS["E2a"], 5, lambda arm, i: True)
    fl = P.Flags()
    r = P.EpisodeRun("E2a", P.CONFIGS["E2a"], tmp_path / "e2a_jevgym_tool_decisions", fl, None, allow_dev_tasks=False)
    assert r.task_ids[0] == "jevgym-clarify-0-d60-near-strict"
    assert any(f["level"] == "ERROR" and "frozen evaluation suite" in f["msg"] for f in fl.items)


# ------------------------------------------------------------------ fixes in the score card / paper_numbers
def test_scorecard_drops_errored_tasks_from_paired_comparisons():
    from jevharness.bench.scorecard import compute_scorecard

    ids = [f"t{i}" for i in range(20)]
    ref = [_traj(t, "llm_only", True, "m") for t in ids]
    arm = [_traj(t, "x", True, "m", error=("503" if i < 3 else None)) for i, t in enumerate(ids)]
    for t in arm[:3]:
        t["result"] = {"success": False}
    card = compute_scorecard({"llm_only": ref, "x": arm}, baseline="llm_only", n_boot=200)
    vb = card["arms"]["x"]["vs_baseline"]
    assert vb["n_paired"] == 17 and vb["delta"] == 0.0 and card["arms"]["x"]["n_tasks_errored"] == 3
    assert any("drop rule" in n for n in card["meta"]["notes"])
    full = compute_scorecard({"llm_only": ref, "x": arm}, baseline="llm_only", n_boot=200, drop_errors=False)
    assert full["arms"]["x"]["vs_baseline"]["n_paired"] == 20
    assert full["arms"]["x"]["vs_baseline"]["delta"] == pytest.approx(-0.15)


def test_scorecard_ni_uses_majority_pass_with_several_trials():
    from jevharness.bench.scorecard import compute_scorecard

    def trials(arm: str, pattern: dict[int, tuple[int, ...]]) -> list[dict[str, Any]]:
        out = []
        for i in range(30):
            for k, s in enumerate(pattern.get(i, (1, 1, 1))):
                t = _traj(f"airline/{i}#{k}", arm, bool(s), "m")
                t["meta"]["tau2"] = {"domain": "airline", "task_id": str(i), "trial": k}
                out.append(t)
        return out

    ref = trials("llm_only", {})
    arm = trials("x", {i: (1, 1, 0) for i in range(10)})     # pass^1 drops on 10 tasks, majority pass never does
    card = compute_scorecard({"llm_only": ref, "x": arm}, baseline="llm_only", n_boot=200, ni_margin=0.05)
    ni = card["arms"]["x"]["vs_baseline"]["ni"]
    assert ni["endpoint"] == "majority pass" and ni["wins"] == 0 and ni["losses"] == 0 and ni["diff"] == 0.0
    assert card["arms"]["x"]["vs_baseline"]["delta"] == pytest.approx(-10 / 90)   # pass^1 Delta stays descriptive


def test_paper_numbers_formats_the_ni_block_as_percentage_points():
    from jevharness.bench.paper_numbers import _flatten, fmt

    x = {"vs_baseline": {"ni": {"diff": -0.012, "ci95": [-0.03, 0.004], "tango_lcb_adj": -0.041,
                                "mde": {"delta_mde": -0.006}, "exact": {"loss_rate_ub": 0.006}, "p_decisive": 0.0004}}}
    got = {"/".join(p[3:]): fmt(p, float(v)) for p, v in _flatten(x, ("r", "a"))}
    assert got == {"ni/diff": "-1.2", "ni/ci95/lo": "-3.0", "ni/ci95/hi": "+0.4", "ni/tango_lcb_adj": "-4.1",
                   "ni/mde/delta_mde": "-0.6", "ni/exact/loss_rate_ub": "0.6", "ni/p_decisive": "$<$0.001"}
