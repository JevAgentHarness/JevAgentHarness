"""Amendment A6 (review round 5), code half -- hand-computed offline checks (docs/METRICS_A6.md):

1. the NI procedure (P0-1 / ruling a): p_NI from the registered task bootstrap, the exact binomial check, the
   power / sample-size / MDE helpers reproducing results/reviews/2026-10-01b/ni_power.out, Newcombe / Tango
   intervals, the two-family conjunction, and the score-card rows;
2. H2 on J at the operating point (P0-3): the J slope with its task-level bootstrap CI, the accuracy-vs-J model
   comparison and the Eq. 2 predicted gain with e_raw / h_raw for guards;
3. the tau-hat* calibration-arm fit and the scalar-threshold projection of Eq. 3 (P0-6);
4. the oracle label kind ``gold_aux.fatal_kind`` and the implication audit (P0-8 / ruling d);
5. the additive ledger fields and the report's retry column (ruling e).
Simulated backends only; the real dev runs (runs/dev, gitignored) are used when present and skipped otherwise."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest import approx

from jevharness.accounting import CallRecord, Ledger
from jevharness.bench import meta_gold as MG
from jevharness.bench import metrics as M
from jevharness.bench import threshold_theory as T
from jevharness.bench.report import RETRY_KEYS, episode_retry_columns, report
from jevharness.bench.scorecard import compute_scorecard, to_markdown
from jevharness.bench.suites.jevgym import NEAR_TOOLS, JevGymEnv
from jevharness.decision.jev import JevBackend
from jevharness.decision.simulated import SimulatedBackend
from jevharness.decision.types import Choice
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.llm.scripted import ScriptedLLM
from jevharness.llm.simulated import SimulatedLLM

SUITE = dict(families="v2", strict=True, distractor_style="near", n_distractors=20, noise_lines=5)
B = 30


# ====================================================================== 1. the NI procedure (P0-1)
def test_ni_sample_size_and_power_reproduce_ni_power_out():
    """The numbers of results/reviews/2026-10-01b/ni_power.out (REVIEW_ROUND5 §5a table)."""
    # 80 % power at 3 pt, d = 5 %, Delta = 0: 339 tasks at alpha = .05, 611 at .05/8, 652 at the 11-test floor
    assert round(M.ni_sample_size(0.05, 0.03, 0.05)) == 339
    assert round(M.ni_sample_size(0.05, 0.03, 0.05 / 8)) == 611
    assert round(M.ni_sample_size(0.05, 0.03, 0.05 / 11)) == 652
    assert round(M.ni_sample_size(0.02, 0.03, 0.05, delta=-0.01)) == 299       # Flash ceiling, 1-pt true loss
    assert round(M.ni_sample_size(0.30, 0.05, 0.05 / 11, delta=0.03)) == 554   # tau^2 margin, +3 pt
    # power at n = 500 / 149
    for alpha, p500, p149 in ((0.05, 0.91, 0.50), (0.05 / 8, 0.70, 0.20), (0.05 / 11, 0.66, 0.17)):
        assert round(M.ni_power(500, 0.05, 0.03, alpha)["power"], 2) == p500
        assert round(M.ni_power(149, 0.05, 0.03, alpha)["power"], 2) == p149
    assert round(M.ni_power(500, 0.20, 0.03, 0.05)["power"], 2) == 0.44        # Flash-Lite discordance, Delta = 0
    assert round(M.ni_power(149, 0.20, 0.05, 0.05)["power"], 2) == 0.39        # C7 proxy
    assert M.ni_power(500, 0.05, 0.03, 0.05)["n_for_80"] == approx(M.ni_sample_size(0.05, 0.03, 0.05))


def test_ni_power_simulation_of_the_registered_rule_holds_its_size():
    r = M.ni_power(500, 0.05, 0.03, 0.05, sims=120, n_boot=800, seed=1)
    assert 0.0 <= r["type1_boot"] <= 0.15 and r["power_boot"] >= 0.75        # review: 0.059 / 0.89
    assert 0.0 <= r["type1_score"] <= 0.18 and r["power_score"] >= 0.75
    bad = M.ni_power(500, 0.02, 0.03, 0.05, sims=5, n_boot=10)                # p_loss < 0 at the boundary: nan
    assert math.isnan(bad["type1_boot"])


def test_ni_exact_check_by_hand():
    e = M.ni_exact_check(0, 0, 500, 0.03)
    assert e["p"] == approx(0.97 ** 500) and e["p"] == approx(2.43e-7, rel=0.01) and e["ni"]
    assert e["loss_rate_ub"] == approx(1 - 0.05 ** (1 / 500)) and e["loss_rate_ub"] == approx(0.006, abs=2e-4)
    e1 = M.ni_exact_check(2, 1, 500, 0.03)
    assert e1["p"] == approx(0.97 ** 500 + 500 * 0.03 * 0.97 ** 499) and e1["n_discordant"] == 3
    # the Clopper-Pearson upper bound solves P[L <= 1 | q] = alpha
    q = e1["loss_rate_ub"]
    assert (1 - q) ** 500 + 500 * q * (1 - q) ** 499 == approx(0.05, abs=1e-6) and q > e["loss_rate_ub"]
    assert not M.ni_exact_check(0, 20, 500, 0.03)["ni"]                     # 20 losses at m = 3 pt: p ~ 0.9


def _paired_outcomes(n: int, wins: int, losses: int) -> tuple[dict[str, float], dict[str, float]]:
    base = {f"t{i}": 1.0 for i in range(n)}
    arm = dict(base)
    for i in range(wins):
        base[f"t{i}"] = 0.0                      # arm solves, base fails
    for i in range(wins, wins + losses):
        arm[f"t{i}"] = 0.0                       # base solves, arm fails
    return base, arm


def test_ni_pvalue_is_the_registered_bootstrap_with_a_margin():
    base, arm = _paired_outcomes(200, 10, 6)
    r = M.ni_pvalue(base, arm, 0.03, n_boot=4000, seed=3)
    keys, d, boot = M._paired_boot(base, arm, 4000, 3)
    pb = M.paired_bootstrap_diff(base, arm, n=4000, alpha=0.05, seed=3)
    assert r["n"] == 200 and r["diff"] == approx(0.02) and r["wins"] == 10 and r["losses"] == 6
    assert r["n_discordant"] == 16 and r["exact"] is None
    assert r["p_ni"] == approx(float(np.mean(boot <= -0.03))) and r["p_sup"] == pb["p_better"]
    assert r["lcb95"] == pb["lcb"] and r["ci95"] == [pb["ci_lo"], pb["ci_hi"]]
    # A6.14 (review round 6): Tango's one-sided score p on (wins, losses, n), the bootstrap p_ni / lcb descriptive
    # (``ni_boot_adj`` is the pre-A6.14 decision). A6.21 (review round 8): the decisive p is the certified
    # Berger-Boos p of one ordering of the whole outcome space (rule bb_ordered); p_T stays reported.
    assert r["rule"] == "bb_ordered" and r["p_decisive"] == r["p_bb"] and 0 < r["p_tango"] < 1
    assert r["p_tango"] == approx(M._norm_sf(M._tango_z(-0.03, 10, 6, 200)))
    assert r["p_bb"] == approx(M.ni_certified_p(10, 6, 200, 0.03)["p"]) and 0 < r["p_bb"] < 1
    assert r["bb"]["converged"] and r["bb"]["lb"] <= r["bb"]["ub"] and r["p_bb"] == approx(1e-4 + r["bb"]["ub"])
    assert r["ni_adj"] == (r["p_bb"] <= 0.05) and (r["p_tango"] <= 0.05) == (r["tango_lcb_adj"] > -0.03) and r["ni_adj"]
    assert r["ni_95"] == (r["p_bb"] <= 0.05) and r["ni_boot_adj"] == (r["p_ni"] <= 0.05)
    assert r["mde"]["discordance"] == approx(0.08) and r["mde"]["n"] == 200
    # the superseded A6.17 rule stays available for comparison
    u = M.ni_pvalue(base, arm, 0.03, n_boot=4000, seed=3, rule="tango_uncond")
    assert u["rule"] == "tango_uncond" and u["p_decisive"] == approx(M.ni_unconditional_p(10, 6, 200, 0.03)["p"])
    # at the Holm floor the level drops and the bounds are recomputed at that level
    f = M.ni_pvalue(base, arm, 0.03, n_boot=4000, seed=3, alpha_adj=0.05 / 11)
    assert f["alpha_adj"] == approx(0.05 / 11) and f["lcb_adj"] <= r["lcb95"] and f["p_ni"] == r["p_ni"]
    assert f["tango_lcb_adj"] < r["tango_lcb_adj"] and f["ni_adj"] == (f["p_bb"] <= 0.05 / 11)
    assert f["p_bb"] == r["p_bb"]                                      # the p-value does not depend on the level


def test_ni_pvalue_inconclusive_and_the_exact_branch():
    # a 3-pt loss on 100 tasks with 13 discordant: NI fails, the 95 % CI covers 0 -> inconclusive at this n
    base, arm = _paired_outcomes(100, 5, 8)
    r = M.ni_pvalue(base, arm, 0.03, n_boot=4000, seed=0)
    assert r["diff"] == approx(-0.03) and not r["ni_adj"] and r["ci95"][0] < 0 < r["ci95"][1] and r["inconclusive"]
    # 500 concordant tasks: the bootstrap is degenerate (zero width); A6.21: the nuisance set is {0} (the
    # discordance bound lies below m), so p = (1 - m)^n + gamma; the exact check's loss-rate bound is still reported
    base, arm = _paired_outcomes(500, 0, 0)
    z = M.ni_pvalue(base, arm, 0.03, n_boot=2000, seed=0, alpha_adj=0.05 / 11)
    assert z["n_discordant"] == 0 and z["lcb95"] == 0.0 and z["p_ni"] == 0.0
    assert z["exact"]["p"] == approx(0.97 ** 500) and z["exact"]["alpha"] == approx(0.05 / 11)
    assert z["rule"] == "bb_ordered" and z["p_decisive"] == approx(0.97 ** 500 + 1e-4, rel=1e-6)
    assert z["ni_adj"] and z["ni_95"] and not z["inconclusive"]
    assert z["exact"]["loss_rate_ub"] == approx(1 - (0.05 / 11) ** (1 / 500))
    # 2 wins / 2 losses on 100 tasks: under A6.14 / A6.17 the exact loss-count test decided; A6.21 orders this
    # outcome with all others (the two wins count), and neither rule establishes NI here
    base, arm = _paired_outcomes(100, 2, 2)
    e = M.ni_pvalue(base, arm, 0.03, n_boot=2000, seed=0)
    assert e["exact"]["p"] == approx(M._binom_cdf(2, 100, 0.03)) and e["rule"] == "bb_ordered"
    assert e["p_decisive"] == e["p_bb"] and not e["ni_adj"] and e["p_bb"] < e["exact"]["p"]
    e17 = M.ni_pvalue(base, arm, 0.03, n_boot=2000, seed=0, rule="tango_uncond")
    assert e17["rule"] == "exact" and e17["p_decisive"] == e17["exact"]["p"]
    # losses only (0 wins / 6 losses on 500): the bootstrap rule was anti-conservative here (ni_size_sim.md, S1); the
    # A6.21 p is the loss-count tail at w = 0 plus gamma, up to the 1e-3 branch-and-bound tolerance
    base, arm = _paired_outcomes(500, 0, 6)
    lo = M.ni_pvalue(base, arm, 0.03, n_boot=2000, seed=0, alpha_adj=0.05 / 11)
    assert lo["rule"] == "bb_ordered" and lo["wins"] == 0 and lo["n_discordant"] == 6
    assert lo["p_decisive"] == approx(M._binom_cdf(6, 500, 0.03) + 1e-4, rel=2e-3)
    assert lo["ni_adj"] == (lo["p_decisive"] <= 0.05 / 11)
    assert M.ni_pvalue({}, {}, 0.03)["n"] == 0


def test_ni_mde_by_hand():
    # 80 % power exactly at n = 339 and Delta = 0 -> the MDE is a zero true difference
    assert abs(M.ni_mde(339, 0.05, 0.03)["delta_mde"]) < 1e-3
    assert M.ni_mde(339, 0.05, 0.03)["margin_mde"] == approx(0.03, abs=1e-3)
    # n = 500: a true loss of ~0.53 pt is still shown NI with 80 % power
    m = M.ni_mde(500, 0.05, 0.03)
    assert m["delta_mde"] == approx(-0.0053, abs=5e-4) and m["margin_mde"] < 0.03
    # under-powered: 30 % discordance at the Holm floor, n = 149: nan (the sample cannot support the margin)
    assert math.isnan(M.ni_mde(149, 0.30, 0.03, 0.05 / 11)["margin_mde"]) or \
        M.ni_mde(149, 0.30, 0.03, 0.05 / 11)["margin_mde"] > 0.1


def test_holm_conjunction_and_step_down_levels():
    assert M.holm_conjunction(0.01, 0.04) == 0.04 and math.isnan(M.holm_conjunction(float("nan"), 0.01))
    lv = M.ni_holm_levels({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05)
    assert lv["a"]["alpha_adj"] == approx(0.05 / 3) and lv["b"]["alpha_adj"] == approx(0.05 / 2)
    assert lv["c"]["alpha_adj"] == approx(0.05) and all(lv[k]["reject"] for k in "abc")
    assert lv["a"]["p_holm"] == approx(0.003) and lv["b"]["p_holm"] == approx(0.04) and lv["c"]["p_holm"] == approx(0.04)
    lv = M.ni_holm_levels({"a": 0.001, "b": 0.03, "c": 0.04, "d": float("nan")}, 0.05)
    assert lv["a"]["reject"] and not lv["b"]["reject"] and not lv["c"]["reject"]     # Holm stops at b
    assert lv["b"]["p_holm"] == approx(0.06) and lv["c"]["p_holm"] == approx(0.06) and math.isnan(lv["d"]["alpha_adj"])


def test_newcombe_and_tango_score_intervals_by_hand():
    z1, z2 = M._z(0.95), M._z(0.975)
    # zero discordance (both = n): Tango's one-sided lower bound is -z^2 / (n + z^2) (0.54 pt at n = 500)
    t = M.tango_interval(500, 0, 0, 0, one_sided=True)
    assert t["ci"][0] == approx(-z1 ** 2 / (500 + z1 ** 2), abs=1e-6) and t["ci"][1] == 1.0
    assert t["ci"][0] == approx(-0.0054, abs=1e-4)
    t2 = M.tango_interval(500, 0, 0, 0)
    assert t2["ci"] == approx([-z2 ** 2 / (500 + z2 ** 2), z2 ** 2 / (500 + z2 ** 2)], abs=1e-6)
    nc = M.newcombe_interval(500, 0, 0, 0)
    assert nc["ci"] == approx([-z2 ** 2 / (500 + z2 ** 2), z2 ** 2 / (500 + z2 ** 2)], abs=1e-9) and nc["phi"] == 0
    # the score statistic vanishes at the estimate, the interval is symmetric when wins = losses
    assert M._tango_z(0.04, 30, 10, 500) == approx(0.0, abs=1e-12)
    s = M.tango_interval(400, 20, 20, 60)
    assert s["diff"] == 0 and s["ci"][0] == approx(-s["ci"][1])
    # 30 wins / 10 losses on 500: both score intervals sit around the Wald interval 0.04 +- 1.96 * 0.0125
    w = 1.96 * math.sqrt((40 / 500 - 0.04 ** 2) / 500)
    for iv in (M.tango_interval(400, 30, 10, 60), M.newcombe_interval(400, 30, 10, 60)):
        assert iv["diff"] == approx(0.04) and iv["ci"][0] == approx(0.04 - w, abs=0.004)
        assert iv["ci"][1] == approx(0.04 + w, abs=0.004)
    assert M.ni_cells({"a": 1, "b": 1, "c": 0}, {"a": 1, "b": 0, "c": 1}) == {"both": 1, "only_a": 1, "only_b": 1,
                                                                             "neither": 0}


def _traj(task: str, config: str, success: bool, calls: list[dict] | None = None) -> dict[str, Any]:
    usage = {"total": {"calls": 1, "input_tokens": 10, "output_tokens": 1, "cost_usd": 0.01, "latency_ms": 100.0},
             "s2": {"calls": 1, "input_tokens": 10, "output_tokens": 1, "cost_usd": 0.01, "latency_ms": 100.0},
             "unpriced_models": []}
    return {"task_id": task, "config": config, "backend": None, "llm": "flash",
            "steps": [{"step": 0, "tool": "finish", "args": {}, "decisions": []}], "final_answer": "x",
            "result": {"success": success}, "usage": usage, "calls": calls or [], "wall_ms": 1.0, "error": None,
            "meta": {}, "probes": []}


def test_scorecard_ni_rows_are_additive_and_holm_adjusted():
    n = 120
    runs = {"llm_only": [_traj(f"t{i}", "llm_only", True) for i in range(n)],
            "jev_a": [_traj(f"t{i}", "jev_a", i >= 2) for i in range(n)],        # 2 losses, 0 wins
            "jev_b": [_traj(f"t{i}", "jev_b", i >= 14) for i in range(n)]}      # 14 losses: 11.7 pt worse
    card = compute_scorecard(runs, baseline="llm_only", n_boot=2000, seed=0)
    assert card["meta"]["ni_margin"] == 0.03
    a, b = card["arms"]["jev_a"]["vs_baseline"], card["arms"]["jev_b"]["vs_baseline"]
    # the existing columns are what they were
    pb = M.paired_bootstrap_diff({f"t{i}": 1.0 for i in range(n)}, {f"t{i}": float(i >= 2) for i in range(n)},
                                 n=2000, alpha=0.05, seed=0)
    assert a["delta"] == pb["diff"] and a["lcb"] == pb["lcb"] and a["p_worse"] == pb["p_worse"]
    assert {"p_ni", "p_ni_holm", "ni_holm", "alpha_adj", "lcb_adj", "inconclusive", "exact", "mde"} <= set(a["ni"])
    assert a["ni"]["n_discordant"] == 2 and a["ni"]["exact"] is not None       # < 5 discordant: exact check
    assert a["ni"]["exact"]["p"] == approx(M._binom_cdf(2, n, 0.03)) and b["ni"]["n_discordant"] == 14
    # A6.21: the decisive p is the certified one (rule bb_ordered) also on the former exact-switch region
    assert a["ni"]["p_decisive"] == a["ni"]["p_bb"] and a["ni"]["ni_holm"] == (a["ni"]["p_bb"] <= a["ni"]["alpha_adj"])
    # Holm over the two-arm family: alpha_adj in {0.025, 0.05}; the 11.7-pt loss is not NI and not inconclusive
    assert sorted([a["ni"]["alpha_adj"], b["ni"]["alpha_adj"]]) == approx([0.025, 0.05])
    assert a["ni"]["holm_m"] == 2 and {a["ni"]["holm_rank"], b["ni"]["holm_rank"]} == {1, 2}
    assert not b["ni"]["ni_holm"] and not b["ni"]["inconclusive"] and b["ni"]["ci95"][1] < 0
    assert b["ni"]["p_ni_holm"] >= b["ni"]["p_decisive"]   # Holm adjusts the decisive p (A6.21, 0 wins here)
    assert b["ni"]["rule"] == "bb_ordered" and a["ni"]["rule"] == "bb_ordered" and "p_tango" in b["ni"]
    assert "cells" in a["score_intervals"] and a["score_intervals"]["tango"]["diff"] == approx(-2 / n)
    md = to_markdown(card)
    assert "Non-inferiority (A6 procedure; margin 3.0 pp)" in md and "| jev_b |" in md and "not NI" in md
    # a custom margin and the tau^2 one
    card5 = compute_scorecard(runs, baseline="llm_only", n_boot=500, seed=0, ni_margin=0.05)
    assert card5["meta"]["ni_margin"] == 0.05 and card5["arms"]["jev_a"]["vs_baseline"]["ni"]["margin"] == 0.05


# ====================================================================== 2. H2 on J (P0-3)
def _j_cells(n_cells: int = 12, slope: float = 2.0, seed: int = 0, task_data: bool = True,
             acc_driven: bool = False) -> list[dict[str, Any]]:
    """Synthetic estimable cells: 30 positive / 70 negative shared tasks, tau = 0.5, TPR_i and FPR_i vary so that J
    and accuracy are not collinear; Delta-U_gross = slope * J (+ a hair of noise), or slope * accuracy."""
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(n_cells):
        tpr = 0.5 + 0.04 * i + 0.03 * (i % 3)
        fpr = 0.2 + 0.03 * (i % 3)
        pos = [(f"t{k}", 0.9 if k < round(30 * tpr) else 0.1, True) for k in range(30)]
        neg = [(f"t{30 + k}", 0.9 if k < round(70 * fpr) else 0.1, False) for k in range(70)]
        recs = pos + neg
        J = sum(1 for _, p, g in pos if p >= 0.5) / 30 - sum(1 for _, p, g in neg if p >= 0.5) / 70
        acc = (sum(1 for _, p, g in pos if p >= 0.5) + sum(1 for _, p, g in neg if p < 0.5)) / 100
        y = slope * (acc if acc_driven else J) + rng.normal(0, 1e-4)
        c: dict[str, Any] = {"gate": "reflection", "live": f"arm{i}", "estimable": True, "stats_matched": True,
                             "stats": {"J": J, "accuracy": acc, "pi": 0.3, "tau": 0.5, "auroc": 0.5 + J / 2},
                             "efficacy": {"e": 0.3, "h": 0.1, "e_raw": 0.5, "h_raw": 0.2}, "n_bar": 5.0,
                             "kappa": 0.001, "delta_u": {"1": {"gross": y, "net": y - 0.005}}}
        if task_data:
            c["task_data"] = {"records": recs, "delta_u": {"1": {t: [y, y - 0.005] for t, _, _ in recs}}}
        cells.append(c)
    return cells


def test_h2_regression_j_recovers_the_j_slope_with_a_task_bootstrap_ci():
    r = MG.h2_regression_j(_j_cells() + [{"estimable": False}], 1.0, n_boot=300, seed=1)
    assert r["n_estimable"] == 12 and r["slope_J"] == approx(2.0, rel=0.02) and not r["falsified"]
    lo, hi = r["slope_J_ci"]
    assert lo < 2.0 < hi and hi - lo < 2.0 and r["n_boot_effective"] > 250 and r["slope_J_p_le_0"] == 0.0
    assert r["j_accuracy"]["slope_J"] == approx(2.0, abs=0.1) and r["lr_add_accuracy_p"] > 0.01
    assert not r["accuracy_explains_better"] and r["lr_add_J_p"] < 1e-6
    # without task data: the point estimates, no CI
    r2 = MG.h2_regression_j(_j_cells(task_data=False), 1.0)
    assert r2["slope_J"] == approx(2.0, rel=0.02) and math.isnan(r2["slope_J_ci"][0]) and "bootstrap_note" in r2
    assert "note" in MG.h2_regression_j(_j_cells(2), 1.0)


def test_h2_regression_j_falsifies_an_accuracy_driven_gain():
    r = MG.h2_regression_j(_j_cells(acc_driven=True, task_data=False), 1.0)
    assert r["accuracy_explains_better"] and r["lr_add_accuracy_p"] < 1e-6 and r["lr_add_J_p"] > 0.01
    assert r["falsified"]
    neg = MG.h2_regression_j(_j_cells(slope=-1.0, task_data=False), 1.0)
    assert neg["slope_J"] < 0 and neg["falsified"]


def test_predicted_gain_eq2_by_hand_and_the_auroc_secondary():
    cell = _j_cells(1)[0]
    cell["stats"].update({"pi": 0.1, "J": 0.4})
    p = MG.predicted_gain(cell, 1.0)
    assert p["net"]["pred_gross"] == approx(5 * 0.1 * 0.9 * 0.4 * 0.4) and p["net"]["pred_net"] == approx(0.072 - 0.005)
    assert p["raw"]["pred_gross"] == approx(5 * 0.09 * 0.7 * 0.4) and p["efficacy_primary"] == "net"
    assert p["pred_gross"] == p["net"]["pred_gross"] and p["observed"]["gross"] == cell["delta_u"]["1"]["gross"]
    g = MG.predicted_gain({**cell, "gate": "stop_guard"}, 10.0)
    assert g["efficacy_primary"] == "raw" and g["pred_gross"] == approx(10 * 5 * 0.09 * 0.7 * 0.4)
    assert g["gap_gross"] == approx(cell["delta_u"]["1"]["gross"] - g["pred_gross"]) or math.isnan(g["gap_gross"])
    # the AUROC regression is still there under both names
    assert MG.h2_regression is MG.h2_regression_auroc
    assert "auroc_only" in MG.h2_regression_auroc(_j_cells(task_data=False), 1.0)


# ====================================================================== 3. tau-hat* calibration arm (P0-6)
def _synth(n_tasks: int = 300, per: int = 10, q_l=lambda c: 0.9, dc: float = 0.002, lo: float = 0.3, hi: float = 1.0,
           seed: int = 1, seed0: int = 5000) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_tasks):
        for k in range(per):
            c = float(rng.uniform(lo, hi))
            s1, s2 = bool(rng.random() < c), bool(rng.random() < q_l(c))
            out.append({"arm": "jev_shadow_cal", "task_id": f"jevgym-cancel-{seed0 + t}-d60-near-strict", "step": k,
                        "mode": "shadow", "shadow": True, "c": c, "s1_ok": s1, "s2_ok": s2, "acceptable": True,
                        "final_ok": s2, "s1_cost": 1e-5, "s1_lat_ms": 100.0, "s2_cost": dc, "s2_lat_ms": 1000.0,
                        "episode_success": True})
    return out


def test_equal_mass_bins_merge_small_bins():
    c = np.arange(0.05, 1.0, 0.1)
    b = T.equal_mass_bins(c, 5, 2)
    assert len(b) == 5 and b[0][0] == -math.inf and b[-1][1] == math.inf
    b3 = T.equal_mass_bins(c, 5, 3)
    counts = [int(np.sum((c >= lo) & (c < hi))) for lo, hi in b3]
    assert all(k >= 3 for k in counts) and sum(counts) == 10 and len(b3) == 2
    assert T.equal_mass_bins(np.zeros(0), 5) == [(-math.inf, math.inf)]


def test_scalar_projection_is_the_crossing_when_eq3_is_monotone():
    # calibrated S1 (g(c) = c), q_L = 0.9, dC / V = 0.1: a(c) = c - 0.8 -> tau-hat* = 0.8
    pr = T.scalar_threshold_projection(_synth(), V=0.02, delta_c=0.002)
    assert pr["k"] == approx(0.1) and pr["monotone"] and pr["islands"] == []
    assert pr["tau_hat"] == approx(0.8, abs=0.08) and pr["crossing"] == pr["tau_hat"]
    assert pr["tau_gain"] == approx(0.8, abs=0.08) and pr["disagreement"] == 0.0
    assert len(pr["bins"]) == 10 and all(b["n"] >= 20 for b in pr["bins"])
    assert math.isinf(T.scalar_threshold_projection(_synth(q_l=lambda c: 1.0), 0.02, 0.0)["tau_hat"])   # never accept


def test_scalar_projection_takes_the_accepting_tail_when_eq3_is_not_monotone():
    # a(c) = c - q_L(c) + 0.1: accepts on [0.3, 0.4) (an island), rejects on [0.4, 0.7), accepts from 0.7 on
    q = lambda c: 0.4 if c < 0.4 else (0.95 if c < 0.7 else 0.6)  # noqa: E731
    pr = T.scalar_threshold_projection(_synth(q_l=q, n_tasks=600), V=0.02, delta_c=0.002)
    assert not pr["monotone"] and pr["islands"] and pr["crossing"] == approx(0.3, abs=0.05)
    assert pr["tau_hat"] == approx(0.7, abs=0.08) and pr["tau_hat"] > pr["crossing"]
    assert 0 < pr["disagreement"] < 0.3 and pr["pointwise"]["accept_rate"] > pr["scalar_accept_rate"]


def test_calibration_arm_fit_primary_and_sensitivity_rows():
    recs = _synth()
    res = T.fit_calibration_arm(recs, task_value=1.0, lam_per_hour=0.0, n_boot=50)
    assert res["seeds"] == {"n_tasks": 300, "min": 5000, "max": 5299} and T.task_seeds(recs)[:2] == [5000, 5001]
    assert any("outside the calibration range" in n for n in res["notes"])     # 5150-5299 are not calibration seeds
    ok = T.fit_calibration_arm(_synth(n_tasks=150), n_boot=50)
    assert ok["notes"] == [] and ok["seeds"]["max"] == 5149
    assert [r["W"] for r in res["sensitivity"]] == [0.1, 10.0, 1.0, 1.0]
    assert [r["lam_per_hour"] for r in res["sensitivity"]] == [0.0, 0.0, 0.5, 5.0]
    prim = res["primary"]
    # every episode succeeded in the synthetic arm, so the counterfactual V-hat is undefined: A6.11 falls back to
    # V = W (the conservative bound) and says so; v_fallback=None keeps tau-hat undefined (the pre-A6.11 behaviour)
    assert prim["W"] == 1.0 and prim["delta_c"] == approx(0.002) and prim["V"] == 1.0
    assert prim["V_source"].startswith("fallback W") and "A6.11" in prim["note"] and not math.isnan(prim["tau_hat"])
    assert prim["n_wrong_episodes"] > 0 and [r["V"] for r in res["sensitivity"]] == [0.1, 10.0, 1.0, 1.0]
    assert "not positive" in prim["note"]   # wrong decisions exist but every episode succeeded: V-hat = 0
    strict = T.fit_calibration_arm(recs, n_boot=50, v_fallback=None)
    assert math.isnan(strict["tau_hat"]) and strict["primary"]["V_source"] == "counterfactual"
    num = T.fit_calibration_arm(recs, n_boot=50, v_fallback=0.5, w_grid=[10.0], lam_grid=[])
    assert num["primary"]["V"] == 0.5 and num["sensitivity"][0]["V"] == approx(5.0)   # scaled with W
    # with final_ok carrying the outcome the counterfactual estimate exists and the fallback is not used
    recs2 = [{**r, "episode_success": bool(r["final_ok"])} for r in recs]
    res2 = T.fit_calibration_arm(recs2, n_boot=50, w_grid=[10.0], lam_grid=[])
    assert res2["primary"]["V"] > 0 and not math.isnan(res2["tau_hat"]) and len(res2["sensitivity"]) == 1
    assert res2["primary"]["V_source"] == "counterfactual" and res2["primary"]["n_wrong_episodes"] > 0
    assert res2["primary"]["projection"]["n"] == len(recs) and "plug_in" in res2["primary"]


def test_calibration_arm_cli_on_a_tiny_shadow_run(tmp_path, capsys):
    from jevharness.bench.runner import run_experiment

    exp = {"name": "cal", "suite": {"name": "jevgym", "n": 4, "seed0": 5000, **SUITE},
           "llm": {"provider": "simulated", "model": "sim-lite", "error_rate": 0.3, "simulated_latency_ms": 10},
           "s1": {"type": "simulated", "accuracy": 0.8, "latency_ms": 5, "price_per_m_input": 0.05},
           "workers": 1, "arms": [{"name": "jev_shadow_cal", "tool_select": {"mode": "shadow"}}]}
    run_experiment(exp, tmp_path, progress=False)
    run = tmp_path / "cal"
    T.main([str(run), "--calibration-arm", "--arm", "jev_shadow_cal", "--dp", "tool_select", "--n-boot", "20",
            "--min-bin", "2", "--out", str(run / "tauhat.json")])
    out = capsys.readouterr().out
    assert "tau-hat* from the calibration arm" in out and "primary" in out
    res = json.loads((run / "tauhat.json").read_text())
    assert res["seeds"]["min"] == 5000 and res["seeds"]["max"] == 5003 and res["notes"] == []
    assert res["meta"]["arms"] == ["jev_shadow_cal"] and "primary" in res and len(res["sensitivity"]) == 4


# ====================================================================== 4. oracle labels: fatal_kind, implications (P0-8)
def _env(seed: int, **kw: Any) -> JevGymEnv:
    e = JevGymEnv(seed=seed, **{**SUITE, **kw})
    e.reset()
    return e


def _seed_of(family: str) -> int:
    for s in range(200):
        if JevGymEnv(seed=s, **SUITE).family == family:
            return s
    raise AssertionError(family)


def test_fatal_kind_terminal_budget_boundary_and_plan_failure():
    e = _env(_seed_of("multi"))
    # a wrong write in the strict world ends the episode: terminal (proven)
    near_write = next(t.name for t in NEAR_TOOLS if t.write and t.name in e._tool_names)
    v = e.oracle_verdict(near_write, {}, B)
    assert v["verdict"] == "fatal" and v["fatal_kind"] == "terminal" and v["fatal_detail"] == "terminal"
    # budget boundary: a read-only distractor is a detour with one spare step and a budget-kind fatal with none
    d0 = e.oracle_cost_to_go(B)
    reader = next(t.name for t in NEAR_TOOLS if not t.write and t.name in e._tool_names)
    assert e.oracle_verdict(reader, {}, d0 + 1)["verdict"] == "detour"
    v = e.oracle_verdict(reader, {}, d0)
    assert v["verdict"] == "fatal" and v["fatal_kind"] == "budget" and v["fatal_detail"] == "budget_steps"
    assert e.oracle_verdict(reader, {}, d0 - 1)["status"] == "doomed"             # the plan is longer than B
    assert "fatal_kind" not in e.oracle_verdict(*e.oracle_action(), B)            # progress: no kind
    # the canonical plan itself can no longer succeed: acting before the required ask (clarify)
    c = _env(8)
    c.step("find_user", {"name": c.user["name"]})
    v = c.oracle_verdict("cancel_order", {"order_id": c.pending_order["order_id"], "reason": "customer_request"}, B)
    snap = c.oracle_snapshot()
    r = snap.step("cancel_order", {"order_id": c.pending_order["order_id"], "reason": "customer_request"})
    expect = "terminal" if (r.done and not snap.evaluate(None)["success"]) else "budget"
    assert v["verdict"] == "fatal" and v["fatal_kind"] == expect
    assert v["fatal_detail"] in ("terminal", "plan_fails", "plan_stuck", "budget_steps")
    # the rollout reason API itself
    D, why = JevGymEnv._rollout_reason(e, B)
    assert D == d0 and why is None and JevGymEnv._rollout_reason(e, 1) == (math.inf, "budget_steps")


def _run(agent: JevAgent, n: int = 10, seed0: int = 0) -> list[dict[str, Any]]:
    return [agent.run(JevGymEnv(seed=seed0 + i, **SUITE)).to_dict() for i in range(n)]


def _shadow_agent(err: float = 0.3, seed: int = 1) -> JevAgent:
    cfg = HarnessConfig(name="meta_shadow", reflection=DPConfig(mode=Mode.SHADOW), routing=DPConfig(mode=Mode.SHADOW),
                        stop=DPConfig(mode=Mode.SHADOW))
    return JevAgent(SimulatedLLM("sim-lite", error_rate=err, simulated_latency_ms=1, seed=seed), cfg,
                    s1=SimulatedBackend(accuracy=0.8, seed=1))


def test_logged_fatal_kind_matches_the_replay_and_the_implication_audit_holds():
    trajs = _run(_shadow_agent(), n=14)
    positives = [(t, d) for t in trajs for st in t["steps"] for d in st["decisions"]
                 if d["dp"] in ("reflection", "routing") and d.get("gold_status") == "ok" and d["gold"] is True]
    assert positives, "the simulated LLM at 30 % error should produce need_fatal positives"
    fatals = [(t, d) for t, d in positives if d["gold_aux"]["verdict"] == "fatal"]
    assert all(d["gold_aux"].get("fatal_kind") in ("terminal", "budget") for _, d in fatals)
    share = MG.fatal_kind_share(trajs)
    assert share["n_positives"] == len(positives) and share["unknown"] == 0
    assert share["terminal"] + share["budget"] == len(fatals) and share["finish"] + len(fatals) + 0 == len(positives)
    # the replay carries the same kind
    for t in trajs:
        env = JevGymEnv(seed=int(t["task_id"].split("-")[2]), **SUITE)
        rep = {r["step"]: r for r in MG.replay_golds(t, env)}
        for st in t["steps"]:
            for d in st["decisions"]:
                if d["dp"] == "reflection" and d.get("gold_status") == "ok" and d["gold_aux"]["verdict"] == "fatal":
                    assert rep[st["step"]]["fatal_kind"] == d["gold_aux"]["fatal_kind"]
    # implications: executed failing verdict => failed; executed finish_ok => success (0 violations)
    imp = MG.implication_audit(trajs)
    assert imp["n_violations"] == 0 and imp["n_checked"] > 0 and imp["implications"]["exec_fail"]["checked"] > 0
    assert imp["implications"]["finish_ok"]["checked"] > 0 and imp["fatal_kind"] == share
    # a corrupted outcome is caught
    bad = json.loads(json.dumps(trajs))
    flipped = 0
    for t in bad:
        if any(d["dp"] in ("reflection", "routing") and d.get("gold") is True for st in t["steps"] for d in st["decisions"]) \
                and not t["result"]["success"]:
            t["result"]["success"] = True
            flipped += 1
            break
    assert flipped == 1 and MG.implication_audit(bad)["n_violations"] >= 1
    assert MG.implication_audit([{**trajs[0], "error": "boom"}])["n_trajectories"] == 0


def test_budget_kind_sensitivity_excludes_only_budget_positives():
    rec = lambda p, g, kind=None: {"gold": g, "gold_status": "ok", "s1_dist": {"yes": p, "no": 1 - p},  # noqa: E731
                                   "gold_aux": {"verdict": "fatal" if g else "progress",
                                                **({"fatal_kind": kind} if kind else {})}}
    recs = [rec(0.9, True, "terminal"), rec(0.8, True, "budget"), rec(0.2, True, "budget"), rec(0.7, False),
            rec(0.1, False)]
    kept = MG.exclude_budget_fatal(recs)
    assert len(kept) == 3 and all((r["gold_aux"].get("fatal_kind") != "budget") for r in kept)
    full, sens = M.gate_stats(recs, 0.5), M.gate_stats(kept, 0.5)
    assert full["tpr"] == approx(2 / 3) and sens["tpr"] == 1.0 and full["fpr"] == sens["fpr"] == 0.5
    assert MG.fatal_kind_share([{"steps": [{"decisions": [{"dp": "reflection", **r} for r in recs]}]}]) == {
        "n_positives": 3, "terminal": 1, "budget": 2, "finish": 0, "unknown": 0, "budget_share": approx(2 / 3)}


def test_ceiling_sensitivity_without_budget_positives():
    trajs = _run(_shadow_agent(), n=8)
    c_all = MG.ceiling(trajs, "reflection", 1.0, n_boot=30)
    c_nb = MG.ceiling(trajs, "reflection", 1.0, n_boot=30, exclude_budget=True)
    assert c_all["n_tasks"] == c_nb["n_tasks"] == 8 and c_nb["exclude_budget"] and not c_all["exclude_budget"]


def test_meta_gold_audit_cli_with_implications(tmp_path, capsys):
    from jevharness.bench.runner import run_experiment

    exp = {"name": "imp", "suite": {"name": "jevgym", "n": 6, **SUITE},
           "llm": {"provider": "simulated", "model": "sim-lite", "error_rate": 0.3, "simulated_latency_ms": 10},
           "s1": {"type": "simulated", "accuracy": 0.8, "latency_ms": 5, "price_per_m_input": 0.05},
           "workers": 1, "arms": [{"name": "meta_shadow", "reflection": {"mode": "shadow"}, "routing": {"mode": "shadow"},
                                   "stop": {"mode": "shadow"}}]}
    run_experiment(exp, tmp_path, progress=False)
    MG.main(["audit", str(tmp_path / "imp"), "--arm", "meta_shadow", "--implications"])
    out = json.loads(capsys.readouterr().out)["meta_shadow"]
    assert out["n_diff"] == 0 and out["implications"]["n_violations"] == 0
    assert "fatal_kind" in out["implications"] and "sensitivity_no_budget" in out
    for g, v in out["sensitivity_no_budget"].items():
        assert v["no_budget"]["n"] <= v["all"]["n"], g


def test_dev_round3_trajectories_pass_the_implication_audit():
    run = Path("runs/dev/e2b_dev_r3")
    if not (run / "meta_shadow_s2lite.jsonl").exists():
        pytest.skip("runs/dev not present")
    from jevharness.harness.trajectory import load_trajectories

    for arm in ("meta_shadow_s2lite", "meta_shadow", "jev_reflect_trigger_s2lite"):
        if not (run / f"{arm}.jsonl").exists():
            continue
        imp = MG.implication_audit(load_trajectories(run / f"{arm}.jsonl"))
        assert imp["n_violations"] == 0, (arm, imp["violations"][:3])
        assert imp["fatal_kind"]["unknown"] == 0 or imp["fatal_kind"]["n_positives"] == imp["fatal_kind"]["finish"]


# ====================================================================== 5. ledger fields (ruling e)
def test_call_record_fields_are_additive_and_timestamped():
    led = Ledger()
    led.add(CallRecord(component="s1", backend="x", model=None, decision_point=None))
    row = led.to_list()[0]
    assert {"attempts", "retry_wait_ms", "request_id", "ts", "tournament_round"} <= set(row)
    assert row["attempts"] is None and row["tournament_round"] is None and row["ts"].endswith("+00:00")
    led.add(CallRecord(component="s2", backend="x", model="m", decision_point=None, attempts=3, retry_wait_ms=1500.0,
                       request_id="req_1", ts="2026-10-01T00:00:00+00:00"))
    assert led.to_list()[1]["ts"] == "2026-10-01T00:00:00+00:00" and led.to_list()[1]["attempts"] == 3
    assert led.summary()["total"]["calls"] == 2                                   # the summary is unchanged


def test_tournament_rounds_are_tagged_on_the_ledger_records():
    class Small(SimulatedBackend):
        def __init__(self):
            super().__init__(accuracy=0.9, seed=0)
            self.max_options = 10

    agent = JevAgent(ScriptedLLM("s"), HarnessConfig(), s1=Small())
    led = Ledger()
    agent._choice_tournament("S", "Which tool should the agent call next?", [f"t{i}" for i in range(35)],
                             "tool_select", led, gold="t3")
    rounds = [r.tournament_round for r in led.records]
    assert rounds == [0, 0, 0, 0, 1] and all(r.ts for r in led.records)         # 4 chunks, then the final round
    led2 = Ledger()
    agent._choice_tournament("S", "q", ["a", "b", "c"], "tool_select", led2)
    assert led2.records[0].tournament_round is None                              # no tournament: untagged


def test_jev_client_counts_attempts_and_backoff(monkeypatch):
    class R:
        def __init__(self, status: int, body: dict | None = None, headers: dict | None = None):
            self.status_code, self._body, self.headers, self.text = status, body or {}, headers or {}, "err"

        def json(self):
            return self._body

    class Http:
        def __init__(self):
            self.calls = 0

        def post(self, path, json=None, headers=None):
            self.calls += 1
            if self.calls == 1:
                return R(429, headers={"retry-after-ms": "5"})
            return R(200, {"answers": {"q0": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}},
                           "usage": {"input_tokens": 10, "output_tokens": 0}, "id": "resp_9"},
                     headers={"x-request-id": "req_42"})

    naps: list[float] = []
    monkeypatch.setattr("jevharness.decision.jev.time.sleep", lambda s: naps.append(s))
    b = JevBackend(provider="openrouter", api_key="k", max_retries=2)
    b._http = Http()
    led = Ledger()
    res = b.decide("state", [Choice("q?", ["a", "b"], decision_point="tool_select")], ledger=led)
    u = res.usage
    assert u.n_calls == 1 and u.attempts == 2 and u.retry_wait_ms == approx(sum(naps) * 1000, abs=1e-2) and naps
    assert u.request_id == "req_42" and res.answers[0].value == "a"
    rec = led.records[0]
    assert rec.attempts == 2 and rec.retry_wait_ms == u.retry_wait_ms and rec.request_id == "req_42"
    assert rec.latency_ms >= 0 and rec.meta == {"n_questions": 1, "n_calls": 1}


def test_report_retry_column_by_hand(tmp_path):
    calls = [{"component": "s2", "attempts": 1, "retry_wait_ms": 0.0, "meta": {}},
             {"component": "s2", "attempts": 2, "retry_wait_ms": 1200.0, "meta": {}},
             {"component": "s1", "meta": {}},
             {"component": "s1", "attempts": 3, "retry_wait_ms": 300.0, "meta": {}}]
    trajs = [_traj("t0", "a", True, calls), _traj("t1", "a", True, [])]
    r = episode_retry_columns(trajs)
    assert r["retry_rate"] == approx(2 / 3) and r["retry_known_frac"] == approx(3 / 4)
    assert r["retry_wait_s"] == approx(1.5 / 2) and set(r) == set(RETRY_KEYS)
    assert math.isnan(episode_retry_columns([_traj("t", "a", True, [])])["retry_rate"])
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "llm_only.jsonl", "w") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")
    md = report(run, scorecard=False)
    assert "retry_rate" in md.splitlines()[2] and "| llm_only" in md
    js = json.loads(report(run, fmt="json", scorecard=False))
    assert js["episode"]["llm_only"]["retry_rate"] == approx(2 / 3)


def test_sign_flip_test_is_exact_under_the_sharp_null_and_detects_a_shift():
    """A6.16 (review round 7): C5's superiority test on paired Delta U is the one-sided sign-flip permutation test,
    exact under the sharp null whatever the skew or the share of zero differences."""
    rng = np.random.default_rng(1)
    # exact branch: 10 differences, mostly zeros (rare discordance), one large win
    d = [0.0] * 7 + [0.4, -0.05, 0.02]
    r = M.sign_flip_test(d)
    # the 7 zeros do not move the statistic, so the 2^10 patterns collapse to the 8 sign patterns of the 3 discordant
    # tasks: 3 of them (0.47, 0.43 and the observed 0.37 sums) reach the observed mean -> p = 3/8 exactly
    assert r["method"] == "exact" and r["n_perm"] == 2 ** 10 and r["p_greater"] == approx(3 / 8) and r["mean"] == approx(0.037)
    assert r["p_less"] == approx(6 / 8)                                # 0.37 is the 3rd-highest of 8 sums (ties counted)
    # size under the null: symmetric heavy-tailed differences, 300 tasks, Monte-Carlo branch
    rejections = 0
    for k in range(200):
        x = rng.standard_t(3, size=300) * (rng.random(300) < 0.15)   # 85 % zeros, skew-free
        rejections += M.sign_flip_test(x, n_perm=1000, seed=k)["p_greater"] <= 0.05
    assert rejections <= 25                                            # ~ 5 % of 200 (binomial 95 % band ≤ 17; slack)
    # power: a genuine shift is found
    x = np.where(rng.random(300) < 0.2, 0.05, 0.0) + rng.normal(0, 0.01, 300)
    assert M.sign_flip_test(x, n_perm=4000, seed=0)["p_greater"] < 0.001
    assert M.sign_flip_test([])["p_greater"] == 1.0


def test_a617_unconditional_p_is_valid_and_close_to_tango():
    """A6.17: the decisive p of the score branch is the Berger-Boos exact unconditional p-value on the branch's region.
    (1) exact size <= alpha at every w of a fine grid for a small n (exhaustive enumeration, no Monte Carlo);
    (2) p_U tracks p_T (within 10 % relative on ordinary counts); (3) ni_pvalue wires it as p_decisive."""
    n, m = 40, 0.05
    lo, hi = M.clopper_pearson(0, 10, 0.05)
    assert lo == 0.0 and hi == approx(0.3085, abs=2e-4)                # Clopper-Pearson 0/10
    lo, hi = M.clopper_pearson(5, 10, 0.05)
    assert lo == approx(0.1871, abs=2e-4) and hi == approx(0.8129, abs=2e-4)
    # p_U on every branch pair (W >= 1, W + L >= 5) of the n = 40 triangle, then the two-branch rule's size at
    # off-grid w (the maximisation grid has step 0.0025; 0.0133 / 0.0377 / 0.1111 / 0.2733 are off it)
    pairs = [(w, l) for w in range(1, n + 1) for l in range(0, n + 1 - w) if w + l >= 5]
    pu = {wl: M.ni_unconditional_p(wl[0], wl[1], n, m)["p"] for wl in pairs}
    lgt = M._lgamma_table(n)
    for alpha in (0.05, 0.05 / 11):
        for w in (0.0133, 0.0377, 0.1111, 0.2733, 0.4):
            size = 0.0
            for W in range(n + 1):
                for L in range(n + 1 - W):
                    pm = math.exp(float(M._trinomial_logpmf(w, m, np.array([W]), np.array([L]), n, lgt)[0]))
                    if W == 0 or W + L < 5:
                        rej = M.ni_exact_check(W, L, n, m, alpha)["ni"]
                    else:
                        rej = pu[(W, L)] <= alpha
                    size += pm * rej
            assert size <= alpha * (1 + 1e-9), (alpha, w, size)
    for W, L, nn, mm in ((40, 30, 149, 0.05), (25, 20, 500, 0.03), (30, 36, 500, 0.03)):
        u = M.ni_unconditional_p(W, L, nn, mm)
        assert 0.85 * u["p_tango"] <= u["p"] <= 1.10 * u["p_tango"] and u["w_set"][0] <= u["w_max"] <= u["w_set"][1] + 1e-9
    # wiring (superseded by A6.21, kept as comparison rules): rule="tango_uncond" decides on p_uncond;
    # unconditional=False reproduces the A6.14 p_T
    base = {f"t{i}": 1.0 if i < 470 else 0.0 for i in range(500)}
    arm = {f"t{i}": (0.0 if 440 <= i < 460 else 1.0) if i < 470 else (1.0 if i < 495 else 0.0) for i in range(500)}
    r = M.ni_pvalue(base, arm, 0.03, n_boot=200, rule="tango_uncond")
    assert r["rule"] == "tango_uncond" and r["p_decisive"] == r["p_uncond"] and (r["wins"], r["losses"]) == (25, 20)
    r0 = M.ni_pvalue(base, arm, 0.03, n_boot=200, unconditional=False)
    assert r0["rule"] == "tango" and r0["p_decisive"] == r0["p_tango"] == r["p_tango"]
    assert abs(r["p_uncond"] - r["p_tango"]) < 0.2 * r["p_tango"]
    # the exact branch of the A6.17 rule is untouched
    base = {f"t{i}": 1.0 for i in range(100)}
    arm = {f"t{i}": 0.0 if i < 2 else 1.0 for i in range(100)}
    r = M.ni_pvalue(base, arm, 0.03, n_boot=200, rule="tango_uncond")
    assert r["rule"] == "exact" and r["p_decisive"] == r["exact"]["p"] and math.isnan(r["w_max"])
