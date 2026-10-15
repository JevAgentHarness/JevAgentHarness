"""scripts/spend_ledger.py (A7.16 (g) total spend): ledger cost per record (measurement calls included), simulated
records never spend, every attempt of a run counted, tau2 runs from their side-car ledger plus the user simulator,
reviews / figure drafts priced or listed as unpriced, the hard-task and round-6 ledgers parsed, a recorded GPU window
billed, and the paper's spend/ keys."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import spend_ledger as SL  # noqa: E402


def test_record_cost_and_run_spend(tmp_path: Path):
    assert SL.record_cost({"llm": "gemini-3.5-flash", "usage": {"total": {"cost_usd": 0.25}}}) == 0.25
    assert SL.record_cost({"backend": "jev", "cost_usd": 0.001}) == 0.001                       # decision-level record
    assert SL.record_cost({"llm": "scripted", "backend": "simulated", "usage": {"total": {"cost_usd": 9.0}}}) is None
    assert SL.record_cost({"llm": "gemini-3.5-flash", "usage": {}}) is None
    assert SL.record_cost({"llm": "x", "usage": {"total": {"cost_usd": 0.25},                  # shadow calls are spent
                                                  "total_with_shadow": {"cost_usd": 0.4}}}) == 0.4
    d = tmp_path / "run" / "sub"
    d.mkdir(parents=True)
    recs = [{"task_id": "a", "llm": "x", "usage": {"total": {"cost_usd": 0.5}}, "error": "timeout"},   # failed attempt
            {"task_id": "a", "llm": "x", "usage": {"total": {"cost_usd": 0.25}}},
            {"task_id": "b", "llm": "scripted", "backend": "simulated", "usage": {"total": {"cost_usd": 3.0}}}]
    (d / "arm.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    s = SL.run_spend(tmp_path / "run")
    assert s == {"usd": 0.75, "records": 2}                                    # every billed attempt, no simulation
    assert SL.run_spend(tmp_path / "missing") == {"usd": 0.0, "records": 0}
    assert SL.run_spend(tmp_path / "run", skip=(tmp_path / "run" / "sub",)) == {"usd": 0.0, "records": 0}


def test_tau2_run_counted_from_sidecar(tmp_path: Path):
    """Every side-car attempt + tau2's user simulator (results.json, jev_errored); converted copies not again."""
    save = tmp_path / "t2" / "raw" / "telecom" / "arm"
    (save / "jev_errored").mkdir(parents=True)
    side = [{"task_id": "a", "llm": "x", "usage": {"total": {"cost_usd": 0.5}}},          # retried attempt
            {"task_id": "a", "llm": "x", "usage": {"total": {"cost_usd": 0.25}}}]
    (save / "jev_decisions.jsonl").write_text("\n".join(json.dumps(r) for r in side) + "\n")
    (save / "results.json").write_text(json.dumps({"simulations": [{"agent_cost": 0.25, "user_cost": 0.125}]}))
    (save / "jev_errored" / "e1.json").write_text(json.dumps({"agent_cost": 0.5, "user_cost": 0.0625}))
    conv = tmp_path / "t2" / "telecom"
    conv.mkdir()
    (conv / "arm.jsonl").write_text(json.dumps({"task_id": "a", "llm": "x", "usage": {"total": {"cost_usd": 0.25}},
                                                "meta": {"tau2": {"user_cost": 0.125}}}) + "\n")
    assert SL.run_spend(tmp_path / "t2") == {"usd": 0.9375, "records": 2}
    assert SL.tau2_user_spend(save) == (0.1875, 2)


def test_reviews_figures_and_gpu_window(tmp_path: Path):
    rv = tmp_path / "reviews" / "d" / "round"
    rv.mkdir(parents=True)
    rows = [{"model": "a", "status": "ok", "priced": True, "cost_usd": 1.5},
            {"model": "b", "status": "ok", "priced": False, "cost_usd": 0.0},
            {"model": "a", "status": "error"}]
    (rv / "run_log.round2.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert SL.review_spend(tmp_path / "reviews") == {"usd": 1.5, "records": 1, "unpriced_calls": {"b": 1}}
    fg = tmp_path / "fig" / "r12"
    fg.mkdir(parents=True)
    rows = [{"model": "img", "cost_usd_estimate": 0.25, "cost_basis": SL.PRICED_FIGURE_BASIS},
            {"model": "pro", "cost_usd_estimate": 0.04, "cost_basis": "usage_tokens_x_assumed_list_price"}]
    (fg / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert SL.figure_spend(tmp_path / "fig") == {"usd": 0.25, "records": 1, "unpriced_calls": {"pro": 1}}
    w, plan = tmp_path / "host_window.json", tmp_path / "plan.json"
    w.write_text(json.dumps({"instance_type": "g", "start_utc": "2026-10-05T07:00:00Z", "stop_utc": "2026-10-05T08:30:00Z"}))
    plan.write_text(json.dumps({"host": {"usd_per_hour_by_type": {"g": 2.0}}}))
    b = SL.host_window_bill(w, plan)
    assert b["hours"] == 1.5 and b["usd"] == 3.0 and b["usd_per_h"] == 2.0
    assert SL.host_window_bill(tmp_path / "missing.json", plan) is None


def test_ledger_parsers(tmp_path: Path):
    notes = tmp_path / "NOTES.md"
    notes.write_text("| **total of this round** | **$412.35** | **$1.29** | **$413.63** | cap $440 |\n")
    assert SL.hard_tasks_total(notes) == 413.63
    r6 = tmp_path / "r6"
    r6.mkdir()
    (r6 / "spend_before_stress_opd.txt").write_text("   1.000  stress: x\n   20.500  total\n")
    (r6 / "spend_before_stress_grpo.txt").write_text("   33.319  total\n")
    assert SL.round6_api(r6) == (33.319, "spend_before_stress_grpo.txt")
    (r6 / "spend_before_e2dev_v2_final.txt").write_text("   42.428  total\n")               # a later ledger
    assert SL.round6_api(r6) == (42.428, "spend_before_e2dev_v2_final.txt")


def test_spend_tex_keys():
    L = {"generated": "t", "git": "g",
         "api": {"tracks": {"stage1_main": {"usd": 911.37, "source": "s"}, "training_round6": {"usd": None, "source": "x"}},
                 "total_usd": 911.37, "unpriced_calls": {"gemini-3.1-pro-preview": 3}},
         "gpu": {"round5": {"usd": 45.29, "hours": 15.07, "usd_per_h": 3.004, "source": "s"}, "not_logged": [],
                 "explore": {"usd": 12.09, "hours": 1.596, "usd_per_h": 7.5, "instance_type": "g", "start_utc": "a",
                             "stop_utc": "b", "source": "s"}},
         "amortisation": {"training_and_calibration_usd": 192.93, "saving_usd_per_episode": 0.0335, "episodes": 5759.1,
                          "basis": "b"},
         "s24_host_bill": {"usd_per_h": 2.24208, "tariff_q": 0.00196, "episodes": 500, "hours_upper": 0.366,
                           "usd_upper": 0.82, "per_episode_upper": 0.00164}, "excluded": {}}
    tex = SL.to_tex(L)
    for k in ("spend/api/total", "spend/api/stage1_main", "spend/gpu/round5/usd", "spend/amortisation/episodes",
              "spend/s24_host/per_episode_upper", "spend/s24_host/tariff_q", "spend/gpu/explore/usd",
              "spend/gpu/explore/hours"):
        assert "\\defres{" + k + "}" in tex, k
    assert "spend/api/training_round6" not in tex                              # not logged -> no key, never a 0
    assert "\\defres{spend/amortisation/episodes}{5{,}759}" in tex
    assert "\\defres{spend/unpriced_calls}{3}" in tex
    md = SL.to_md(L)
    assert "not logged" in md and "gemini-3.1-pro-preview x 3" in md and "exploratory E1/E2 box $12.09" in md
