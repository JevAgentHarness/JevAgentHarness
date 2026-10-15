"""scripts/stage2_analysis.py (docs/PREREG_A7_STAGE2.md A7.4 as amended by A7.9-A7.16) on synthetic fixture runs: the
intersection-union clauses (NI by the registered rule, cost / latency by the paired t-test), S2-3 on a common-state
corpus (A7.10 (c), (d)), S2-4 with the frozen tariff (A7.13 (iii)), S2-1X (A7.12), the S2-1T scheduler gate (A7.16 (a)),
the degenerate-profile labels (p = 1), Holm over the four tests and the five secondary rows, the descriptive rows
(LOGO, best_control, the A7.13 attribution table) and the paper's \\defres keys. Hand-made records only (no model is
called)."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage2_analysis as S2  # noqa: E402
from jevharness.bench import stage2_corpus as C  # noqa: E402

FLASH, X = "gemini-3.5-flash", "deepseek/deepseek-v4-pro"
TASKS = json.loads((ROOT / "configs/eval/jevgym_stage2_tasks.json").read_text())["task_ids"]


def _rec(task: str, arm: str, ok: bool, cost: float, lat: float, decisions: list[dict] | None = None,
         s1_cost: float = 1e-4) -> dict:
    return {"task_id": task, "config": arm, "llm": FLASH, "backend": "jev", "result": {"success": ok},
            "usage": {"total": {"cost_usd": cost, "latency_ms": 1000 * lat}, "s1": {"cost_usd": s1_cost, "calls": 4},
                      "s2": {"calls": 3, "input_tokens": 100}, "unpriced_models": []},
            "steps": [{"step": 0, "tool": "lookup", "args": {"id": "1"}, "decisions": decisions or []}],
            "meta": {}, "error": None}


def _arm(d: Path, arm: str, tasks: list[str], cost: float, lat: float, seed: int, losses: int = 0,
         acc: float | None = None, s1_cost: float = 1e-4) -> None:
    rng = np.random.default_rng(seed)
    with open(d / f"{arm}.jsonl", "w") as f:
        for i, t in enumerate(tasks):
            dec = None
            if acc is not None:   # five replace-mode tool decisions per task, each right with probability acc
                dec = [{"dp": "tool_select", "mode": "replace", "s1_value": "a" if rng.random() < acc else "b",
                        "gold": "a", "options": ["a", "b"]} for _ in range(5)]
            f.write(json.dumps(_rec(t, arm, i >= losses, max(1e-5, cost + rng.normal(0, 0.003)),
                                    lat + rng.normal(0, 2.0), dec, s1_cost)) + "\n")


def _profile(path: Path, single_skip: bool) -> None:
    best = {"group": "tool", "arm": "jev_ts_cascade_tauhat",
            "decision_points": {"tool_select": {"mode": "cascade", "tau": 1.0}}}
    if single_skip:
        best["skip"] = "composition not tested (the selected profile delegates at one decision-point group)"
    path.write_text(yaml.safe_dump({"kind": "jevagent_profile", "version": 1, "name": "jevagent_best",
                                    "status": "selected", "system2": {
                                        FLASH: {"decision_points": {"tool_select": {"mode": "cascade", "tau": 1.0},
                                                                    "stop": {"mode": "trigger", "tau": 0.9}},
                                                "best_single": best,
                                                "logo": {"tool": None, "stop": "configuration-identical to best_single "
                                                         "(the profile without stop is tool alone): reused"}},
                                        "gemini-3.5-flash-lite": {"decision_points": {}, "skip": "LLM-only"}}}))


def _profile_x(path: Path) -> None:
    path.write_text(yaml.safe_dump({"kind": "jevagent_profile", "version": 1, "name": "jevagent_best_x",
                                    "status": "selected", "system2": {X: {"decision_points": {
                                        "tool_select": {"mode": "cascade", "tau": 0.97}}}}}))


def _corpus(cdir: Path, tasks: list[str], accs: dict[str, float], seed: int = 11) -> None:
    """Two labelled occurrences per source and task (step 0 on plan, step 1 after an off-plan step) and one replay
    answer per question and backend, right with probability accs[name]."""
    occ, qs = [], {}
    for t in tasks:
        for _, src in C.SOURCES:
            for step in range(2):
                st = f"{t}|{src}|{step}"
                h = C.question_hash(st, ["a", "b"], ["", ""])
                occ.append({"task": t, "source": src, "step": step, "qhash": h, "state": st, "options": ["a", "b"],
                            "descriptions": ["", ""], "gold": "a", "gold_aux": None, "gold_status": None,
                            "dp": "tool_select", "stratum": "on_plan" if step == 0 else "after_off_plan",
                            "label_provenance": C.LABEL_PROVENANCE})
                qs[h] = {"state": st, "options": ["a", "b"], "descriptions": ["", ""]}
    excl = [{"task": "excluded-task", "failing_sources": {"rl_ts_replace_f": "missing"}}]
    C.write_corpus({"registered": "test", "occurrences": occ, "questions": qs, "excluded": excl,
                    "eligible_tasks": tasks, "summary": {"occurrences": len(occ), "unique_questions": len(qs),
                                                          "eligible_tasks": len(tasks), "excluded": 1}}, cdir)
    rng = np.random.default_rng(seed)
    for nm, acc in accs.items():
        with open(cdir / f"answers_{nm}.jsonl", "w") as f:
            for h in sorted(qs):
                ok = rng.random() < acc
                f.write(json.dumps({"qhash": h, "answer": "a" if ok else "b", "confidence": 0.9 if ok else 0.6,
                                    "error": None, "valid": True}) + "\n")


def _e1(d: Path, backend: str, acc: float, seed: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(seed).permutation(50)          # exactly round(50 acc) items right, in random places
    right = set(order[: int(round(50 * acc))].tolist())
    with open(d / f"s1_{backend}.jsonl", "w") as f:
        for i in range(50):
            f.write(json.dumps({"id": f"b{i}", "dp": "tool_select", "source": "bfcl:simple_python", "gold": ["x"],
                                "pred": "x" if i in right else "y"}) + "\n")


def _fixture(root: Path, tmp: Path, schedule: bool = True) -> list[str]:
    runs = root
    s2 = runs / "stage2_jevgym"
    s2.mkdir(parents=True)
    shutil.copy(ROOT / "configs/experiments/stage2_jevgym.yaml", s2 / "experiment.yaml")
    _arm(s2, "llm_only", TASKS, 0.040, 15.0, 1)
    _arm(s2, "jevagent_best", TASKS, 0.024, 11.0, 2)
    _arm(s2, "best_single", TASKS, 0.026, 11.5, 3)
    _arm(s2, "jevagent_best_rl", TASKS, 0.026, 12.0, 4, losses=1, s1_cost=0.0)   # tariff charged by the analysis
    _arm(s2, "jevagent_default", TASKS, 0.025, 11.0, 5)
    _arm(s2, "jevagent_best_sl4b", TASKS, 0.027, 12.5, 8, losses=2, s1_cost=0.0)   # S2-4S (A7.18 (e) 5)
    _arm(s2, "jevagent_minus_tool", TASKS[:200], 0.033, 13.0, 6)
    _arm(s2, "llm_only_s2lite", TASKS, 0.008, 9.0, 7, losses=40)
    if schedule:
        (s2 / "schedule.json").write_text(json.dumps({"mode": "task_major", "seed": 732, "concurrency": 8,
                                                      "arms": ["llm_only", "jevagent_best", "best_single"]}))
    st = runs / "stage2_rl_states"
    st.mkdir()
    shutil.copy(ROOT / "configs/experiments/stage2_rl_states.yaml", st / "experiment.yaml")
    for i, (nm, acc) in enumerate((("jev_ts_replace_f", 0.96), ("intern_ts_replace_f", 0.45),
                                   ("rl_ts_replace_f", 0.95))):
        _arm(st, nm, TASKS[:200], 0.02, 10.0, 20 + i, acc=acc)
    x = runs / "stage2_x"
    x.mkdir()
    shutil.copy(ROOT / "configs/experiments/stage2_x.yaml", x / "experiment.yaml")
    _arm(x, "llm_only_x", TASKS, 0.008, 14.0, 30)
    _arm(x, "jevagent_best_x", TASKS, 0.005, 12.0, 31)
    _arm(x, "llm_only_xsmall", TASKS, 0.002, 9.0, 32, losses=60)
    sl = runs / "stage2_startlux_states"                                              # A7.18 (e) 3
    sl.mkdir()
    shutil.copy(ROOT / "configs/experiments/stage2_startlux_states.yaml", sl / "experiment.yaml")
    _arm(sl, "sl4b_ts_replace_f", TASKS[:200], 0.02, 10.0, 25, acc=0.80, losses=20)
    _e1(runs / "e1_bfcl_when2call", "jev", 0.9, 40)
    _e1(runs / "e1_bfcl_when2call_rl", "jevagent_rl", 0.85, 41)
    for nm, acc, seed in (("startlux_4b", 0.97, 42), ("startlux_27b", 0.98, 43), ("intern_4b", 0.8, 44)):
        _e1(runs / "e1_bfcl_when2call_startlux", nm, acc, seed)
    cdir = tmp / "s23"
    _corpus(cdir, TASKS[:200], {"rl": 0.93, "intern": 0.45, "jev": 0.95, "base": 0.3, "sl4b": 0.8, "sl27b": 0.9})
    (tmp / "freeze.json").write_text(json.dumps({"frozen_at": "2026-10-04T10:00:00Z", "checkpoint": "test"}))
    (tmp / "tariff.json").write_text(json.dumps({"q_usd_per_episode": 0.001, "frozen_at": "2026-10-05T00:00:00Z",
                                                 "sensitivity": {"busy_throughput": 0.0004}}))
    (tmp / "tariff_sl.json").write_text(json.dumps({"q_usd_per_episode": 0.0008, "frozen_at": "2026-10-05T00:00:00Z"}))
    (tmp / "slman.json").write_text(json.dumps({"sizes": [{"size": "4B", "status": "original"},
                                                          {"size": "27B", "status": "not_run",
                                                           "why": "no free GPU with >= 64 GiB"}]}))
    (tmp / "control.json").write_text(json.dumps({"best_control": "small_only", "reason": "test"}))
    _profile(tmp / "best.yaml", single_skip=False)
    _profile_x(tmp / "best_x.yaml")
    return ["--runs", str(runs), "--profile", str(tmp / "best.yaml"), "--profile-x", str(tmp / "best_x.yaml"),
            "--s23", str(cdir), "--freeze-record", str(tmp / "freeze.json"), "--tariff", str(tmp / "tariff.json"),
            "--control-selection", str(tmp / "control.json"), "--telecom-ids", str(tmp / "none.json"),
            "--tariff-sl", str(tmp / "tariff_sl.json"), "--startlux-manifest", str(tmp / "slman.json"),
            "--n-boot", "300"]


def _tex(path: Path) -> dict[str, str]:
    return dict(re.findall(r"\\defres\{([^}]*)\}\{(.*)\}", path.read_text()))


def test_stage2_analysis_family_secondary_and_tex_on_fixture_runs(tmp_path: Path):
    args = _fixture(tmp_path / "runs", tmp_path)
    out, tex = tmp_path / "out", tmp_path / "paper" / "stage2.tex"
    assert S2.main(args + ["--out", str(out), "--tex", str(tex)]) == 0
    res = json.loads((out / "stage2.json").read_text())
    fam, sec = res["family"], res["secondary"]
    s21 = fam["S2-1"]
    assert s21["n_pairs"] == 500 and set(s21["clauses"]) == {"ni", "cost"}          # A7.9 (a): no latency clause
    assert s21["p_decisive"] == max(c["p"] for c in s21["clauses"].values()) and s21["holm"]["reject"]
    assert s21["clauses"]["ni"]["boundary"] == -0.03 and s21["clauses"]["cost"]["boundary"] == 0.0
    lo, hi = s21["clauses"]["ni"]["newcombe95"]                                      # zero discordance (A7.17 (c)):
    assert s21["clauses"]["ni"]["discordance"] == 0 and lo < 0 < hi                  # NI decided, interval not [0, 0]
    assert list(sec) == ["S2-1T", "S2-1L", "S2-L3a", "S2-L3b", "S2-1X", "S2-4S"]      # A7.18 (f): six rows
    assert min(v["holm"]["alpha_adj"] for v in sec.values()) == 0.05 / 6
    s24s = sec["S2-4S"]                                                              # A7.18 (e) 5: q_S on top
    assert s24s["arm"] == "jevagent_best_sl4b" and s24s["tariff"]["q_usd_per_episode"] == 0.0008
    assert s24s["n_pairs"] == 500 and set(s24s["clauses"]) == {"ni", "cost"} and s24s["holm"]["reject"]
    s21t = sec["S2-1T"]                                                              # A7.9 (b), A7.16 (a)
    assert set(s21t["clauses"]) == {"latency"} and s21t["scheduler"]["task_major"]
    assert s21t["latency_arm_s"] < s21t["latency_ref_s"] and s21t["holm"]["reject"]
    assert fam["S2-2"]["holm"]["reject"] and fam["S2-2"]["reference_component"]["group"] == "tool"
    s23 = fam["S2-3"]                                                                # A7.10 (c): common states
    assert set(s23["clauses"]) == {"intern", "jev"} and s23["n_pairs"] == 200 and s23["n_excluded"] == 1
    assert s23["clauses"]["intern"]["p"] < 1e-6 and s23["clauses"]["jev"]["boundary"] == -0.05
    assert s23["accuracy"]["intern"] < 0.6 < s23["accuracy"]["rl"] and s23["holm"]["reject"]
    assert s23["sensitivity_worst_case"]["n"] == 201                                # A7.10 (d)
    assert set(s23["strata"]) == {"on_plan", "after_off_plan", "llm_only_source"}
    s24 = fam["S2-4"]                                                                # A7.13 (iii): q on top
    assert s24["tariff"]["q_usd_per_episode"] == 0.001 and s24["holm"]["reject"]
    assert s24["cost"]["cost_arm"] > 0.026 and set(s24["tariff_sensitivity"]) == {"zero", "busy_throughput"}
    assert sec["S2-1X"]["n_pairs"] == 500 and sec["S2-1X"]["holm"]["reject"]
    assert all(sec[k]["p_decisive"] == 1.0 for k in ("S2-1L", "S2-L3a", "S2-L3b"))  # Lite skipped, no L3
    assert "LLM-only" in sec["S2-1L"]["status"]
    d = res["descriptive"]
    assert d["logo"]["tool"]["arm"] == "jevagent_minus_tool" and d["logo"]["tool"]["n_pairs"] == 200
    assert d["logo"]["stop"]["arm"] == "best_single"                                # configuration-identical reuse
    lo, hi = d["logo"]["tool"]["delta_ci"]
    assert lo < hi                                                                  # never a [0, 0] interval
    assert d["best_control"]["arm"] == "llm_only_s2lite"                            # small_only (A7.14 (b))
    rl = d["stage2_rl"]
    assert rl["frozen"]["corpus_acc"] > 0.85 > rl["base"]["corpus_acc"] and rl["intern"]["live_n"] == 100
    assert 0.8 < rl["jev"]["l1_acc"] <= 1.0 and "corpus_acc" not in rl["sft_v1"]    # rows without input stay empty
    s1 = d["stage2_s1"]                                                              # A7.18 (h)
    r4, r27 = s1["rows"]["sl4b"], s1["rows"]["sl27b"]
    assert r4["live_n"] == 200 and r4["l1_acc"] > s1["rows"]["jev"]["l1_acc"]       # leads statically ...
    assert r4["vs_jev"]["live_delta"] < 0 and r4["vs_jev"]["live_delta_ci"][1] < 0  # ... not in the loop
    assert r4["vs_jev"]["corpus_delta"] < 0 and r4["vs_jev"]["e1_n"] == 50
    assert "not run" in r27["status"] and "live_success" not in r27                  # manifest: 27B not served
    assert s1["kendall"]["rows"] == ["jev", "intern", "frozen", "sl4b"] and s1["kendall"]["e1_vs_live"] < 1
    k = _tex(tex)
    assert k == _tex(out / "stage2.tex")
    for t in S2.FAMILY + S2.SECONDARY:                                              # the paper's tab:tests keys
        assert f"stage2/{t}/p_decisive" in k and f"stage2/{t}/decision" in k and f"stage2/{t}/p_holm" in k
    for c in ("est", "est_ci/lo", "est_ci/hi", "boundary", "p", "decision"):          # tab:stage2-clauses
        assert f"stage2/S2-1/ni/{c}" in k and f"stage2/S2-1/cost/{c}" in k and f"stage2/S2-3/jev/{c}" in k
    assert k["stage2/S2-1/ni/boundary"] == "-3.0" and k["stage2/S2-3/jev/boundary"] == "-5.0"
    assert k["stage2/S2-1/cost/boundary"] == "0" and "stage2/S2-1T/latency/est" in k
    assert "stage2/S2-1/delta" in k and "stage2/S2-1/cost_savings_ci/lo" in k and "stage2/S2-3/delta_jev" in k
    for row in ("jev", "intern", "frozen"):                                          # tab:rl
        for m in ("corpus_acc", "corpus_acc_offplan", "ece", "live_success", "own_acc"):
            assert f"stage2_rl/{row}/{m}" in k
    assert "stage2_rl/frozen/l1_acc" in k and "stage2_rl/base/corpus_acc" in k and "stage2_rl/base/live_success" not in k
    for arm in ("llm_only", "jevagent_best", "best_control", "llm_only_x", "jevagent_best_x", "llm_only_xsmall"):
        assert f"stage2/scorecard/{arm}/success" in k and f"stage2/scorecard/{arm}/cost_usd" in k
    assert "stage2/scorecard/jevagent_best_x/vs_baseline/cost_savings" in k
    assert "stage2/scorecard/jevagent_best_sl4b/success" in k and "stage2/S2-4S/tariff/q" in k
    for m in ("l1_acc", "corpus_acc", "live_success", "vs_jev/live_delta", "vs_jev/live_delta_ci/lo", "vs_jev/e1_delta"):
        assert f"stage2_s1/sl4b/{m}" in k, m
    assert "stage2_s1/sl27b/status" in k and "stage2_s1/kendall/e1_vs_live" in k
    iv = d["intervention"]                                                           # A7.16 (d), descriptive
    assert {"llm_only", "jevagent_best", "best_single"} <= set(iv["arms"]) and iv["arms"]["llm_only"]["n"] == 500
    assert iv["arms"]["jevagent_best"]["reference"] == "llm_only" and "llm_calls_avoided" in iv["arms"]["jevagent_best"]
    for m in ("tool_coverage", "arg_coverage", "llm_calls", "trigger_questions", "trigger_fires", "trigger_tasks"):
        assert f"stage2/intervention/jevagent_best/{m}" in k, m
    assert "stage2/intervention/jevagent_best/llm_calls_avoided" in k
    for st in ("on_plan", "after_off_plan", "llm_only_source"):                     # A7.10 (c) strata keys
        assert f"stage2/S2-3/strata/{st}/rl" in k and f"stage2/S2-3/strata/{st}/occurrences" in k
    assert "stage2/S2-1/coverage/tool_coverage" in k or not (fam["S2-1"].get("coverage") or {}).get("tool_coverage")
    assert not any(x.startswith("SIM/") for x in k)
    assert "S2-1" in (out / "stage2.md").read_text()


def test_stage2_analysis_gates_enter_with_p_one(tmp_path: Path):
    """No scheduler record -> S2-1T p = 1 (A7.16 (a)); no freeze record -> S2-3, S2-4 not tested (A7.10 (h)); no tariff
    -> S2-4 not analysed (A7.13 (iii)); a single-group profile -> S2-2 not tested; a profile_x that delegates nothing
    -> S2-1X not tested (A7.12 (g))."""
    args = _fixture(tmp_path / "runs", tmp_path, schedule=False)
    _profile(tmp_path / "best.yaml", single_skip=True)
    (tmp_path / "freeze.json").unlink()
    (tmp_path / "best_x.yaml").write_text(yaml.safe_dump({"kind": "jevagent_profile", "version": 1, "status": "selected",
                                                          "system2": {X: {"decision_points": {}, "skip":
                                                                          "not tested (the fitted profile delegates "
                                                                          "nothing on X)"}}}))
    assert S2.main(args + ["--out", str(tmp_path / "o"), "--tex", str(tmp_path / "t.tex")]) == 0
    res = json.loads((tmp_path / "o" / "stage2.json").read_text())
    fam, sec = res["family"], res["secondary"]
    assert fam["S2-2"]["p_decisive"] == 1.0 and "composition not tested" in fam["S2-2"]["verdict"]
    assert all(fam[t]["p_decisive"] == 1.0 and "no frozen JevAgentRL" in fam[t]["verdict"] for t in ("S2-3", "S2-4"))
    assert not fam["S2-2"]["holm"]["reject"] and fam["S2-1"]["holm"]["reject"]
    assert sec["S2-1T"]["p_decisive"] == 1.0 and "A7.16 (a)" in sec["S2-1T"]["verdict"]
    assert sec["S2-1T"]["clauses"]["latency"]["p"] < 0.05                            # still reported, descriptive
    assert sec["S2-1X"]["p_decisive"] == 1.0 and "delegates nothing" in sec["S2-1X"]["verdict"]
    (tmp_path / "freeze.json").write_text(json.dumps({"frozen_at": "2026-10-04T10:00:00Z"}))
    (tmp_path / "tariff.json").unlink()
    assert S2.main(args + ["--out", str(tmp_path / "o2"), "--tex", str(tmp_path / "t2.tex")]) == 0
    fam = json.loads((tmp_path / "o2" / "stage2.json").read_text())["family"]
    assert fam["S2-3"]["holm"]["reject"] and "no frozen tariff" in fam["S2-4"]["verdict"]
    late = json.dumps({"frozen_at": "2026-10-07T12:30:00Z"})                          # after the deadline
    (tmp_path / "freeze.json").write_text(late)
    assert S2.main(args + ["--out", str(tmp_path / "o3"), "--tex", str(tmp_path / "t3.tex")]) == 0
    assert "after the deadline" in json.loads((tmp_path / "o3" / "stage2.json").read_text())["family"]["S2-3"]["verdict"]
    # the round-5 record's key `frozen_utc` is read as `frozen_at`
    (tmp_path / "freeze.json").write_text(json.dumps({"frozen_utc": "2026-10-03T10:17:50Z", "name": "JevAgentRL-4B"}))
    assert S2.main(args + ["--out", str(tmp_path / "o4"), "--tex", str(tmp_path / "t4.tex")]) == 0
    assert json.loads((tmp_path / "o4" / "stage2.json").read_text())["family"]["S2-3"]["holm"]["reject"]


def test_s2_4s_not_run_and_not_tested_cases(tmp_path: Path):
    """A7.18 (f), (g): the 4B server could not start -> S2-4S not run (p = 1); a fitted profile that delegates nothing
    with StartLux-4B -> not tested; thresholds still pending -> not tested; none of these touches the family."""
    args = _fixture(tmp_path / "runs", tmp_path)
    (tmp_path / "slman.json").write_text(json.dumps({"sizes": [{"size": "4B", "status": "not_run",
                                                                "why": "kernel check: fast kernels NOT active"}]}))
    assert S2.main(args + ["--out", str(tmp_path / "o"), "--tex", str(tmp_path / "t.tex")]) == 0
    res = json.loads((tmp_path / "o" / "stage2.json").read_text())
    s = res["secondary"]["S2-4S"]
    assert s["p_decisive"] == 1.0 and "not run" in s["verdict"] and "kernel" in s["verdict"]
    assert res["family"]["S2-1"]["holm"]["reject"] and "not run" in res["descriptive"]["stage2_s1"]["rows"]["sl4b"]["status"]
    (tmp_path / "slman.json").unlink()
    prof = yaml.safe_load((tmp_path / "best.yaml").read_text())
    blk = prof["system2"][FLASH].setdefault("backends", {})
    for case, block, word in (("pending", {"tool_select": {"calibration": "pending"}}, "pending calibration"),
                              ("skip", {"tool_select": {"tau": 0.99, "always_defer": True},
                                        "skip": "not tested (the fitted profile delegates nothing with startlux_4b)"},
                               "delegates nothing")):
        blk["startlux_4b"] = block
        (tmp_path / "best.yaml").write_text(yaml.safe_dump(prof))
        assert S2.main(args + ["--out", str(tmp_path / case), "--tex", str(tmp_path / f"{case}.tex")]) == 0
        s = json.loads((tmp_path / case / "stage2.json").read_text())["secondary"]["S2-4S"]
        assert s["p_decisive"] == 1.0 and word in s["verdict"], case


def test_stage2_analysis_empty_profile(tmp_path: Path):
    """A7.1 (g) 5 / A7.17 (c): an empty profile (no decision point qualified) leaves every profile test at p = 1 with
    its label and runs no LOGO row; S2-3 does not depend on the profile."""
    args = _fixture(tmp_path / "runs", tmp_path)
    why = "No decision point qualified in Stage 1: the procedure keeps every decision with the LLM (A7.1 (g) 5)"
    (tmp_path / "best.yaml").write_text(yaml.safe_dump({"kind": "jevagent_profile", "version": 1, "status": "selected",
                                                        "system2": {FLASH: {"decision_points": {}, "skip": why},
                                                                    "gemini-3.5-flash-lite": {"skip": why}}}))
    assert S2.main(args + ["--out", str(tmp_path / "o"), "--tex", str(tmp_path / "t.tex")]) == 0
    res = json.loads((tmp_path / "o" / "stage2.json").read_text())
    for tab, t in (("family", "S2-1"), ("family", "S2-2"), ("family", "S2-4"), ("secondary", "S2-1T"),
                   ("secondary", "S2-1L")):
        assert res[tab][t]["p_decisive"] == 1.0 and "No decision point qualified" in res[tab][t]["verdict"], t
    assert res["family"]["S2-3"]["holm"]["reject"] and res["descriptive"]["logo"] == {}


def test_stage2_analysis_slower_profile_fails_only_the_secondary_latency_row(tmp_path: Path):
    """A7.9 (d): a profile that saves cost at non-inferior success but adds latency passes S2-1 and fails S2-1T."""
    args = _fixture(tmp_path / "runs", tmp_path)
    _arm(tmp_path / "runs" / "stage2_jevgym", "jevagent_best", TASKS, 0.024, 18.0, 2)   # cheaper, 3 s slower
    assert S2.main(args + ["--out", str(tmp_path / "o"), "--tex", str(tmp_path / "t.tex")]) == 0
    res = json.loads((tmp_path / "o" / "stage2.json").read_text())
    assert res["family"]["S2-1"]["holm"]["reject"]
    t = res["secondary"]["S2-1T"]
    assert not t["holm"]["reject"] and "wrong direction" in t["clauses"]["latency"]["reading"]


def test_failed_replay_answers_count_incorrect_and_a_favourable_non_rejection_reads_as_registered(tmp_path: Path):
    """A7.10 (d): a scoring failure is incorrect for that backend; A7.16 (c): the reading of a favourable estimate."""
    occ = [{"task": "t", "qhash": "h", "gold": "a", "gold_aux": None, "gold_status": None}]
    assert C.task_accuracy(occ, {"h": {"answer": "a", "error": None}}) == {"t": 1.0}
    assert C.task_accuracy(occ, {"h": {"answer": "a", "error": "RuntimeError"}}) == {"t": 0.0}
    assert C.task_accuracy(occ, {}) == {"t": 0.0}
    v = {"holm": {"reject": False}, "clauses": {"ni": {"p": 0.03}, "cost": {"p": 0.01}}}
    assert S2.verdict(v) == S2.FAVOURABLE
    assert S2.verdict({**v, "clauses": {"ni": {"p": 0.2}}}).startswith("not rejected")


def test_s2_l3b_not_run_when_e6_was_missing_at_the_selection(tmp_path: Path):
    """A7.1 (h) (Deviations entry of 2026-10-03): E6 was not complete when the profile was selected, so the action
    group stays off and S2-L3b is not run: p = 1 with the registered reason, read from the selection record."""
    args = _fixture(tmp_path / "runs", tmp_path)
    sel = {"candidates": [{"group": "action", "arm": a, "fate": "run directory runs/e6_alfworld_action_only missing"}
                          for a in ("jev_action_assist_k3", "jev_action_replace")]}
    (tmp_path / "sel.json").write_text(json.dumps(sel))
    assert S2.main(args + ["--profile-selection", str(tmp_path / "sel.json"), "--out", str(tmp_path / "o"),
                           "--tex", str(tmp_path / "t.tex")]) == 0
    res = json.loads((tmp_path / "o" / "stage2.json").read_text())
    s = res["secondary"]["S2-L3b"]
    assert s["p_decisive"] == 1.0 and "A7.1 (h): E6 not run in Stage 1" in s["verdict"]
    assert res["family"]["S2-1"]["holm"]["reject"]                                     # nothing else changes
    sel["candidates"][0]["fate"] = "not qualified: not non-inferior"                    # E6 present at the selection
    (tmp_path / "sel.json").write_text(json.dumps(sel))
    assert S2.main(args + ["--profile-selection", str(tmp_path / "sel.json"), "--out", str(tmp_path / "o2"),
                           "--tex", str(tmp_path / "t2.tex")]) == 0
    assert "A7.1 (h)" not in json.loads((tmp_path / "o2" / "stage2.json").read_text())["secondary"]["S2-L3b"]["verdict"]


def test_intervention_stats_counts_delegation_and_trigger_fires():
    """A7.16 (d): tool coverage over executed (non-shadow) tool decisions, trigger questions / fires / tasks with a
    fire, one outcome per task."""
    def traj(task: str, ok: bool, decisions: list[dict]) -> dict:
        r = _rec(task, "arm", ok, 0.01, 1.0, decisions)
        return r
    cas = {"dp": "tool_select", "mode": "cascade", "s1_value": "a", "deferred": False}
    dfr = {"dp": "tool_select", "mode": "cascade", "s1_value": "a", "deferred": True}
    shadow = {"dp": "tool_select", "mode": "replace", "s1_value": "a", "shadow": True}
    fire = {"dp": "reflection", "mode": "trigger", "s1_value": True, "final_value": True}
    quiet = {"dp": "reflection", "mode": "trigger", "s1_value": False, "final_value": False}
    tasks = {"t1": traj("t1", True, [cas, dfr, shadow, quiet]), "t2": traj("t2", False, [cas, fire, fire])}
    st = S2.intervention_stats(tasks)
    assert st["n"] == 2 and st["success"] == 0.5
    assert st["tool_decisions"] == 3 and st["tool_delegated"] == 2 and abs(st["tool_coverage"] - 2 / 3) < 1e-12
    assert st["trigger_questions"] == 3 and st["trigger_fires"] == 2 and st["trigger_tasks"] == 1
    assert abs(st["trigger_fire_rate"] - 2 / 3) < 1e-12
    empty = S2.intervention_stats({"t": traj("t", True, [])})
    assert empty["tool_coverage"] == 0.0 and empty["trigger_fire_rate"] is None


def _v2_inputs(runs: Path, tmp: Path, block: dict | None) -> list[str]:
    """A7.19 (f): JevAgentRL-v2's corpus answers, live replace arm, profile run (runs/stage2_v2, read beside
    stage2_jevgym's llm_only), tariff and freeze record; ``block`` is the profile's backends.jevagentrl_v2 (None: none)."""
    cdir = tmp / "s23"
    rng = np.random.default_rng(12)
    with open(cdir / "answers_v2.jsonl", "w") as f:
        for line in (cdir / "answers_jev.jsonl").read_text().splitlines():
            ok = rng.random() < 0.9
            f.write(json.dumps({"qhash": json.loads(line)["qhash"], "answer": "a" if ok else "b",
                                "confidence": 0.9 if ok else 0.6, "error": None, "valid": True}) + "\n")
    st = runs / "stage2_v2_states"
    st.mkdir()
    shutil.copy(ROOT / "configs/experiments/stage2_v2_states.yaml", st / "experiment.yaml")
    _arm(st, "v2_ts_replace_f", TASKS[:200], 0.02, 10.0, 26, acc=0.9, losses=10)
    pr = runs / "stage2_v2"
    pr.mkdir()
    shutil.copy(ROOT / "configs/experiments/stage2_v2.yaml", pr / "experiment.yaml")
    _arm(pr, "jevagent_best_v2", TASKS, 0.026, 12.0, 27, losses=3, s1_cost=0.0)     # GPU cost: the tariff
    (tmp / "tariff_v2.json").write_text(json.dumps({"q_usd_per_episode": 0.002, "frozen_at": "2026-10-05T00:00:00Z",
                                                    "served_model": "jevagentrl-v2", "sensitivity": {
                                                        "busy_throughput": 0.0005, "reference_host": 0.0006}}))
    (tmp / "v2_FROZEN.txt").write_text("name JevAgentRL-v2 (test)\nlabel GO: test\nfrozen_at 2026-10-04T00:53:20Z\n"
                                       + "0" * 64 + "  ./model.safetensors\n")
    prof = yaml.safe_load((tmp / "best.yaml").read_text())
    if block is not None:
        prof["system2"][FLASH].setdefault("backends", {})["jevagentrl_v2"] = block
    (tmp / "best.yaml").write_text(yaml.safe_dump(prof))
    return ["--tariff-v2", str(tmp / "tariff_v2.json"), "--v2-freeze-record", str(tmp / "v2_FROZEN.txt")]


def test_v2_rows_are_exploratory_and_change_nothing_else(tmp_path: Path):
    """A7.19 (f): without a v2 input nothing is added to any output; with them the v2 section (corpus vs Jev with the task
    bootstrap, live replace vs Jev with Newcombe, the profile row vs llm_only with v2's own tariff) is added under the
    label "exploratory (A7.19)" and every family / secondary result, every other TeX key and S2-3 stay as they were."""
    args = _fixture(tmp_path / "runs", tmp_path)
    assert S2.main(args + ["--out", str(tmp_path / "o0"), "--tex", str(tmp_path / "t0.tex"), "--strict"]) == 0
    r0 = json.loads((tmp_path / "o0" / "stage2.json").read_text())
    assert "stage2_v2" not in r0["descriptive"] and not any(k.startswith("stage2_v2/") for k in _tex(tmp_path / "t0.tex"))
    blk = {"tool_select": {"tau": 0.9}, "fitted_from": {"fit_arm": "v2_shadow_cal"}}
    v2 = _v2_inputs(tmp_path / "runs", tmp_path, blk)
    assert S2.main(args + v2 + ["--out", str(tmp_path / "o1"), "--tex", str(tmp_path / "t1.tex"), "--strict"]) == 0
    r1 = json.loads((tmp_path / "o1" / "stage2.json").read_text())
    for tab in ("family", "secondary"):                                   # outside every Holm table
        assert list(r1[tab]) == list(r0[tab])
        for t in r0[tab]:
            assert r1[tab][t]["p_decisive"] == r0[tab][t]["p_decisive"] and r1[tab][t]["verdict"] == r0[tab][t]["verdict"]
    assert "v2" not in r1["family"]["S2-3"]["other_rows"] and "v2" not in r1["family"]["S2-3"]["accuracy"]
    k0, k1 = _tex(tmp_path / "t0.tex"), _tex(tmp_path / "t1.tex")
    assert {k: v for k, v in k1.items() if not k.startswith("stage2_v2/")} == k0
    d = r1["descriptive"]["stage2_v2"]
    assert d["label"] == S2.V2_LABEL == "exploratory (A7.19)" and not d.get("status") and d["freeze"]["label"] == "GO: test"
    row = d["row"]
    assert row["e1"] is None and "l1_acc" not in row                       # A7.19 (f) names no E1 row
    assert 0.8 < row["corpus_acc"] < 1.0 and row["corpus_tasks"] == 200
    lo, hi = row["vs_jev"]["corpus_delta_ci"]
    assert lo < row["vs_jev"]["corpus_delta"] < hi and row["vs_jev"]["corpus_n"] == 200
    assert row["live_n"] == 200 and abs(row["live_success"] - 0.95) < 1e-12 and row["steps"] == 1.0
    assert row["vs_jev"]["live_n"] == 200 and row["vs_jev"]["live_delta"] < 0 and 0.8 < row["own_acc"] <= 1.0
    pr = d["profile"]
    assert pr["arm"] == "jevagent_best_v2" and pr["reference"] == "llm_only" and pr["n_pairs"] == 500
    assert pr["tariff"]["q_usd_per_episode"] == 0.002 and pr["cost"]["cost_arm"] > 0.026
    assert set(pr["tariff_sensitivity"]) == {"zero", "busy_throughput", "reference_host"}
    assert pr["tariff_sensitivity"]["zero"]["cost_savings"] > pr["cost"]["cost_savings"] > 0
    assert pr["delta"] < 0 and pr["delta_ci"][0] < pr["delta"] < pr["delta_ci"][1] and pr["intervention"]["n"] == 500
    for key in ("stage2_v2/label", "stage2_v2/corpus_acc", "stage2_v2/vs_jev/corpus_delta_ci/lo", "stage2_v2/live_success",
                "stage2_v2/vs_jev/live_delta", "stage2_v2/profile/delta", "stage2_v2/profile/cost_savings_ci/hi",
                "stage2_v2/profile/tariff/q", "stage2_v2/profile/tariff/reference_host/cost_savings"):
        assert key in k1, key
    assert r1["family"] == r0["family"] and r1["secondary"] == r0["secondary"]   # nothing of v2 in either table
    assert "jevagent_best_v2" not in json.dumps(r1["family"]) and "answers_v2" not in json.dumps(r1["family"])
    md = (tmp_path / "o1" / "stage2.md").read_text()
    assert "JevAgentRL-v2 — exploratory (A7.19)" in md and "`jevagent_best_v2` (exploratory (A7.19))" in md


def test_v2_rows_not_run_cases(tmp_path: Path):
    """A7.19 (e), (f): a late or missing freeze record -> the v2 section reads not run (no rows); no
    backends.jevagentrl_v2 block, a pending block, a block that delegates nothing or no tariff -> the profile row reads
    not run / not analysed while the corpus and live rows stay; the analysis never fails strict for an absent v2 row."""
    args = _fixture(tmp_path / "runs", tmp_path)
    v2 = _v2_inputs(tmp_path / "runs", tmp_path, None)

    def run(tag: str) -> dict:
        assert S2.main(args + v2 + ["--out", str(tmp_path / tag), "--tex", str(tmp_path / f"{tag}.tex")]) == 0
        return json.loads((tmp_path / tag / "stage2.json").read_text())["descriptive"]["stage2_v2"]

    d = run("noblock")
    assert "no backends.jevagentrl_v2 block" in d["profile"]["status"] and d["row"]["corpus_acc"] > 0.8
    prof = yaml.safe_load((tmp_path / "best.yaml").read_text())
    for tag, block, word in (("pending", {"tool_select": {"calibration": "pending"}}, "pending calibration"),
                             ("skip", {"tool_select": {"tau": 0.99, "always_defer": True},
                                       "skip": "not tested (the fitted profile delegates nothing with jevagentrl_v2)"},
                              "delegates nothing")):
        prof["system2"][FLASH].setdefault("backends", {})["jevagentrl_v2"] = block
        (tmp_path / "best.yaml").write_text(yaml.safe_dump(prof))
        assert word in run(tag)["profile"]["status"], tag
    prof["system2"][FLASH]["backends"]["jevagentrl_v2"] = {"tool_select": {"tau": 0.9}}
    (tmp_path / "best.yaml").write_text(yaml.safe_dump(prof))
    (tmp_path / "tariff_v2.json").unlink()
    assert "no frozen tariff" in run("notariff")["profile"]["status"]
    (tmp_path / "v2_FROZEN.txt").write_text("frozen_at 2026-10-07T12:00:01Z\n")
    d = run("late")
    assert "after the A7.19 (e) deadline" in d["status"] and "row" not in d and "profile" not in d
    (tmp_path / "v2_FROZEN.txt").unlink()
    assert "no JevAgentRL-v2 freeze record" in run("nofreeze")["status"]
    assert S2.read_v2_freeze(ROOT / "results/gpu_round6_miles/final_FROZEN.txt")["files"]["model.safetensors"] == (
        "d7bfe3ed50bc8d3ff4821500b0975a9c8ba7b626a7129f47a979d99d4c686282")
