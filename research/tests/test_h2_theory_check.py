"""A6.16 (i): the H2 theory check -- Eq. 2's predicted Delta-U against the observed gross / net Delta-U per cell with
one JOINT task bootstrap, the >= 4-cell / full-rank requirements ("not estimable", never falsified) and the CLI
wire-up (``h2_theory`` primary, ``h2_regression_exploratory`` next to it, old keys preserved)."""

from __future__ import annotations

import json
import math
from typing import Any, Callable

import numpy as np
import pytest
from pytest import approx

from jevharness.bench import meta_gold as MG
from jevharness.bench import metrics as M

W1 = "1"


# ====================================================================== synthetic cells with full task data
def _rec(p: float, g: bool) -> dict[str, Any]:
    return {"gold": g, "s1_dist": {"yes": p, "no": 1 - p}}


def _aux(v0: str, alt: str, pl: str | None, w: float, branch: str) -> dict[str, Any]:
    return {"gold_aux": {"cf_status": "ok", "verdict": v0, branch: alt, "verdict_placebo": pl, "cf_weight": w}}


def _cell(name: str, n_opp: float, n_tasks: int = 100, kappa: float = 0.001, gate: str = "reflection",
          obs: Callable[[str, float], float] | None = None, uniform: bool = True, placebo_pos: float = 0.0,
          W: float = 1.0) -> dict[str, Any]:
    """One H2 cell the way :func:`meta_gold.h2_cells` lays it out, with ``task_data`` for the joint bootstrap.

    ``uniform``: every task carries 3 positive (p = 0.9) and 7 negative (p = 0.1) gate records and the same CF
    rows, so pi = 0.3, J = 1, e = h = 1 are invariant under any task resample (the exact case); otherwise the first
    35 tasks carry one positive record each and the other 65 one negative each (about one resample in eight drops
    below the A4 minimum of 30 positives). ``obs(task, pred)`` gives the per-task gross Delta-U (default: exactly
    the Eq. 2 prediction); the net value is gross minus the live arm's System-1 cost kappa * n_opp (the control
    pays no System-1 cost).
    ``placebo_pos``: share of the e-stratum CF rows whose placebo also progresses (raw e = 1, net e = 1 - share)."""
    branch = MG.BRANCH_OF[gate]
    guard = gate == "stop_guard"
    tasks = [f"jevgym-x-{200000 + k}-d60" for k in range(n_tasks)]
    records: list[tuple[str, float, bool]] = []
    cf: list[tuple[str, str, str, str | None, float]] = []
    live: dict[str, list[float]] = {}
    for k, t in enumerate(tasks):
        # a trigger's positive class is gold True (need); a guard's is gold False (a wrong proposal), rejected
        # at p < tau
        pos, neg = ((t, 0.1, False), (t, 0.9, True)) if guard else ((t, 0.9, True), (t, 0.1, False))
        if uniform:
            records += [pos] * 3 + [neg] * 7
            pl = "progress" if k % 100 < 100 * placebo_pos else "fatal"
            cf += [(t, "fatal", "progress", pl, 1.0)] * 2
            cf += [(t, "progress", "fatal", "progress", 1.0)] * 2
        else:
            records.append(pos if k < 35 else neg)
            cf.append((t, "fatal", "progress", "fatal", 1.0) if k % 2 == 0
                      else (t, "progress", "fatal", "progress", 1.0))
        live[t] = [1.0, float(n_opp), kappa * n_opp]
    stats = M.gate_stats([_rec(p, g) for _, p, g in records], tau=0.5, guard=guard)
    eff = M.cf_efficacy([_aux(v0, alt, pl, w, branch) for _, v0, alt, pl, w in cf], branch)
    primary = "raw" if guard else "net"
    e, h = (eff["e_raw"], eff["h_raw"]) if primary == "raw" else (eff["e"], eff["h"])
    pred = W * n_opp * stats["pi"] * (1 - stats["pi"]) * (e + h) * stats["J"]
    per = {t: (pred if obs is None else obs(t, pred)) for t in tasks}
    du = {t: [per[t], per[t] - kappa * n_opp] for t in tasks}
    g, n = float(np.mean([v[0] for v in du.values()])), float(np.mean([v[1] for v in du.values()]))
    return {"gate": gate, "live": name, "control": f"ctrl_{name}", "s2_family": "sim", "backend": "sim",
            "estimable": stats["estimable"], "stats_matched": True, "stats": stats, "efficacy": eff,
            "n_bar": float(n_opp), "kappa": kappa, "efficacy_primary": primary,
            "delta_u": {W1: {"gross": g, "net": n, "gross_ci": [g, g], "net_ci": [n, n]}},
            "task_data": {"records": records, "delta_u": {W1: du}, "cf": cf, "live": live}}


def _exact_cells(n: int = 5, **kw: Any) -> list[dict[str, Any]]:
    return [_cell(f"arm{i}", n_opp=1.0 + i, **kw) for i in range(n)]


# ====================================================================== 1. exact agreement -> slope 1, tight CI
def test_exact_agreement_gives_slope_one_with_a_degenerate_ci_and_is_estimable():
    cells = _exact_cells(5)
    r = MG.h2_theory_check(cells, 1.0, n_boot=200, seed=3)
    assert r["estimable"] and r["reason"] is None and r["n_cells"] == 5 and r["n_boot_effective"] == 200
    assert r["slope"]["value"] == approx(1.0) and r["slope"]["ci"] == approx([1.0, 1.0], abs=1e-9)
    assert r["slope"]["ci_covers_1"] and r["slope"]["n_boot_effective"] == 200 and r["slope"]["expected"] == 1.0
    assert r["mae"] == approx(0.0) and r["mae_ci"] == approx([0.0, 0.0], abs=1e-12)
    assert r["mean_diff_gross"] == approx(0.0) and r["share_inside_predicted_ci"] == 1.0
    for i, c in enumerate(r["cells"]):
        assert c["estimable"] and c["reason"] is None and c["n_boot_effective"] == 200
        # the Eq. 2 prediction is predicted_gain's, the observed value the cell's paired Delta-U
        pg = MG.predicted_gain(cells[i], 1.0)
        assert c["pred_gross"] == pg["pred_gross"] and c["pred_net"] == pg["pred_net"]
        assert c["pred_gross"] == approx(0.42 * (1 + i)) and c["obs_gross"] == cells[i]["delta_u"][W1]["gross"]
        assert c["diff_gross"] == approx(0.0) and c["diff_gross_ci"] == approx([0.0, 0.0], abs=1e-12)
        assert c["pred_gross_ci"] == approx([c["pred_gross"]] * 2)
        assert c["obs_gross_ci"] == approx([c["obs_gross"]] * 2)
        assert c["inside_predicted_ci"] is True
        assert c["inputs"]["pi"] == approx(0.3) and c["inputs"]["J"] == 1.0 and c["inputs"]["e"] == 1.0
    assert "falsified" not in r
    assert r["exploratory"]["regression_j"]["n_estimable"] == 5 and "exploratory" in r["exploratory"]["note"]
    assert MG.h2_theory_wording(r).startswith("H2 consistent with Eq. 2")


def test_noisy_cells_cover_one_and_lost_estimability_contributes_nan_replicates():
    rng = np.random.default_rng(7)
    cells = [_cell(f"arm{i}", n_opp=2.0 + i, uniform=False, obs=lambda t, p: p + rng.normal(0, 0.02))
             for i in range(5)]
    assert all(c["stats"]["n_pos"] == 35 and c["stats"]["n_neg"] == 65 and c["estimable"] for c in cells)
    r = MG.h2_theory_check(cells, 1.0, n_boot=300, seed=1)
    assert r["estimable"] and r["slope"]["ci_covers_1"] and abs(r["slope"]["value"] - 1) < 0.1
    lo, hi = r["slope"]["ci"]
    assert lo < 1 < hi and hi - lo < 0.5
    # a resample with fewer than 30 positives (about one in eight) is NaN for every cell, and the count says so
    for c in r["cells"]:
        assert 0 < c["n_boot_effective"] < 300
        assert all(math.isfinite(x) for x in c["pred_gross_ci"] + c["obs_net_ci"] + c["diff_gross_ci"])
    assert r["slope"]["n_boot_effective"] == r["cells"][0]["n_boot_effective"] == r["n_boot_effective"]
    assert 0 < r["slope"]["n_boot_effective"] < 300


# ====================================================================== 2. fewer than 4 cells / rank deficiency
def test_three_cells_are_not_estimable_and_never_falsified():
    r = MG.h2_theory_check(_exact_cells(3), 1.0, n_boot=50)
    assert r["estimable"] is False and r["reason"] == "fewer than 4 estimable cells" and r["slope"] is None
    assert r["n_cells"] == 3 and "falsified" not in r and "falsified" not in json.dumps({k: v for k, v in r.items()
                                                                                       if k != "exploratory"})
    assert MG.h2_theory_wording(r) == "H2 not estimable: fewer than 4 estimable cells"
    # the per-cell comparison is still reported (with CIs), so is the exploratory regression
    assert all(c["estimable"] and c["n_boot_effective"] == 50 for c in r["cells"]) and r["mae"] == approx(0.0)
    assert "regression_j" in r["exploratory"]
    # four cells of which one is not estimable: the same reason; a non-estimable cell carries its own reason
    cells = _exact_cells(4)
    cells[0]["estimable"] = False
    r4 = MG.h2_theory_check(cells, 1.0, n_boot=20)
    assert not r4["estimable"] and r4["reason"] == "fewer than 4 estimable cells" and r4["n_cells"] == 3
    assert not r4["cells"][0]["estimable"] and "not estimable" in r4["cells"][0]["reason"]
    # min_cells is a parameter
    assert MG.h2_theory_check(_exact_cells(3), 1.0, n_boot=20, min_cells=3)["estimable"]


def test_rank_deficient_design_is_not_estimable():
    cells = _exact_cells(5)
    for c in cells:                       # J = 0 at the operating point: every prediction is exactly 0
        c["stats"]["J"] = 0.0
    r = MG.h2_theory_check(cells, 1.0, n_boot=20)
    assert not r["estimable"] and r["reason"] == "rank-deficient design" and r["slope"] is None
    assert all(c["pred_gross"] == 0.0 and c["estimable"] for c in r["cells"])


# ====================================================================== 3. the resample is joint
def test_joint_resample_is_one_multiset_per_replicate_shared_by_every_cell():
    tasks = [f"t{k}" for k in range(40)]
    m = MG._joint_resample(tasks, 6, seed=11)
    assert m.shape == (6, 40) and np.array_equal(m, MG._joint_resample(tasks, 6, seed=11))
    assert np.all(m.sum(1) == 40) and np.all(m == np.round(m)) and np.any(m == 0) and np.any(m > 1)
    assert not np.array_equal(m, MG._joint_resample(tasks, 6, seed=12))
    assert MG._joint_resample(tasks, 0, seed=1).shape == (0, 40) and MG._joint_resample([], 3).shape == (3, 0)
    # two cells sharing all task ids: cell B's per-task observed Delta-U is cell A's plus a constant, so under a joint
    # resample every replicate of B is A's replicate shifted -- and so is its percentile CI, exactly (independent
    # resamples would not shift the quantiles by the constant)
    rng = np.random.default_rng(5)
    noise = {f"jevgym-x-{200000 + k}-d60": float(rng.normal(0, 0.3)) for k in range(100)}
    shift = 0.25
    a = _cell("a", 2.0, obs=lambda t, p: p + noise[t])
    b = _cell("b", 2.0, obs=lambda t, p: p + noise[t] + shift)
    r = MG.h2_theory_check(_exact_cells(3) + [a, b], 1.0, n_boot=150, seed=2)
    ca, cb = r["cells"][3], r["cells"][4]
    assert ca["obs_gross_ci"][1] - ca["obs_gross_ci"][0] > 0.05          # a real interval
    assert cb["obs_gross_ci"] == approx([x + shift for x in ca["obs_gross_ci"]], abs=1e-12)
    assert cb["diff_gross_ci"] == approx([x + shift for x in ca["diff_gross_ci"]], abs=1e-12)
    # the point recomputation on the unit multiset reproduces predicted_gain and the cell's paired Delta-U
    tasks = sorted(MG._cell_task_ids(a, W1))
    idx = {t: i for i, t in enumerate(tasks)}
    rep = MG._theory_replicates(MG._theory_aggregates(a, W1, idx), np.ones((1, len(tasks))), 1.0, "net")
    assert rep["ok"][0] and rep["pred_gross"][0] == approx(MG.predicted_gain(a, 1.0)["pred_gross"])
    assert rep["pred_net"][0] == approx(MG.predicted_gain(a, 1.0)["pred_net"])
    assert rep["obs_gross"][0] == approx(a["delta_u"][W1]["gross"])
    assert rep["obs_net"][0] == approx(a["delta_u"][W1]["net"])
    assert rep["pi"][0] == approx(0.3) and rep["J"][0] == 1.0 and rep["e"][0] == 1.0 and rep["h"][0] == 1.0
    assert rep["n_bar"][0] == 2.0 and rep["kappa"][0] == approx(0.001)


def test_guard_cells_use_the_raw_efficacy_and_the_guard_polarity():
    g = _cell("guard", 3.0, gate="stop_guard", placebo_pos=0.5)
    assert g["efficacy_primary"] == "raw" and g["efficacy"]["e_raw"] == 1.0 and g["efficacy"]["e"] == approx(0.5)
    pg = MG.predicted_gain(g, 1.0)
    assert pg["pred_gross"] == pg["raw"]["pred_gross"] != pg["net"]["pred_gross"]
    tasks = sorted(MG._cell_task_ids(g, W1))
    idx = {t: i for i, t in enumerate(tasks)}
    A = MG._theory_aggregates(g, W1, idx)
    raw = MG._theory_replicates(A, np.ones((1, len(tasks))), 1.0, "raw")
    net = MG._theory_replicates(A, np.ones((1, len(tasks))), 1.0, "net")
    assert raw["pred_gross"][0] == approx(pg["raw"]["pred_gross"])
    assert net["pred_gross"][0] == approx(pg["net"]["pred_gross"])
    assert raw["pi"][0] == approx(g["stats"]["pi"]) and raw["J"][0] == approx(g["stats"]["J"]) == 1.0
    r = MG.h2_theory_check(_exact_cells(4) + [g], 1.0, n_boot=30)
    assert r["estimable"] and r["cells"][4]["efficacy_primary"] == "raw" and r["cells"][4]["diff_gross"] == approx(0.0)


# ====================================================================== 4. gross vs. net
def test_gross_and_net_differ_by_exactly_the_system1_query_cost():
    kappa = 0.0025
    cells = _exact_cells(5, kappa=kappa)
    r = MG.h2_theory_check(cells, 1.0, n_boot=100, seed=0)
    for i, c in enumerate(r["cells"]):
        n_bar = cells[i]["n_bar"]
        assert c["pred_gross"] - c["pred_net"] == approx(kappa * n_bar)                     # Eq. 2's kappa n-bar term
        td = cells[i]["task_data"]["delta_u"][W1]
        s1_term = float(np.mean([g - n for g, n in td.values()]))                          # the live arm's S1 cost
        assert s1_term == approx(kappa * n_bar) and c["obs_gross"] - c["obs_net"] == approx(s1_term)
        assert np.array(c["pred_gross_ci"]) - np.array(c["pred_net_ci"]) == approx([kappa * n_bar] * 2)
        assert np.array(c["obs_gross_ci"]) - np.array(c["obs_net_ci"]) == approx([s1_term] * 2)
        assert c["diff_net"] == approx(c["diff_gross"])     # exact cells: both differences vanish
    # the slope (the Eq. 2 readout) is on GROSS observed vs. GROSS predicted
    assert r["slope"]["value"] == approx(1.0)


# ====================================================================== 5. wire-up
def test_h2_report_stores_the_theory_check_as_primary_and_the_regression_as_exploratory():
    cells = _exact_cells(5)
    out = MG.h2_report(cells, [1.0], n_boot=40, seed=0)
    assert {"h2_theory", "h2_verdict", "h2_regression_exploratory", "regression_j", "predicted_gain",
            "regression_auroc", "regression", "cells", "n_cells", "n_estimable"} <= set(out)
    assert out["h2_theory"][W1]["estimable"] and out["h2_theory"][W1]["slope"]["value"] == approx(1.0)
    assert out["h2_verdict"][W1].startswith("H2 consistent with Eq. 2")
    assert out["regression_j"] is out["h2_regression_exploratory"]
    assert out["h2_regression_exploratory"][W1] is out["h2_theory"][W1]["exploratory"]["regression_j"]
    assert out["h2_regression_exploratory"][W1]["slope_J"] is not None and out["n_estimable"] == 5
    assert all("task_data" not in c for c in out["cells"]) and out["regression"] is out["regression_auroc"]
    # serialisable (NaN -> null) like the CLI prints it
    json.dumps(MG._jsonable(out))
    # not estimable: the verdict says so and no "falsified" wording appears outside the exploratory block
    out3 = MG.h2_report(_exact_cells(3), [1.0], n_boot=10)
    assert out3["h2_verdict"][W1] == "H2 not estimable: fewer than 4 estimable cells"
    t = {k: v for k, v in out3["h2_theory"][W1].items() if k != "exploratory"}
    assert "falsified" not in json.dumps(MG._jsonable(t))


def test_h2_cells_carry_the_cf_and_live_rows_and_the_cli_prints_both_keys(tmp_path, capsys):
    from test_meta_gold import _sim_exp
    from jevharness.bench.scorecard import load_runs

    run = _sim_exp(tmp_path)
    runs = load_runs(run)
    pair = ("jev_reflect_trigger_s2lite", "ratem_reflect_trigger_s2lite")
    cell = MG.h2_cells(runs, ["meta_shadow_s2lite"], [pair], [1.0], n_boot=20)[0]
    td = cell["task_data"]
    assert {"records", "delta_u", "cf", "live"} <= set(td)
    live = td["live"]
    assert len(live) == 10
    assert sum(v[1] for v in live.values()) / sum(v[0] for v in live.values()) == approx(cell["n_bar"])
    assert sum(v[2] for v in live.values()) / sum(v[1] for v in live.values()) == approx(cell["kappa"])
    assert len(td["cf"]) == cell["efficacy"]["n"] and all(r[1] and r[2] for r in td["cf"])
    assert all(isinstance(r[4], float) for r in td["cf"])
    # the task namespaces agree (records, cf, live and delta_u index the same tasks)
    tasks = {MG._tid(r[0]) for r in td["records"]}
    assert tasks <= set(live) and {MG._tid(r[0]) for r in td["cf"]} <= set(live)
    assert set(td["delta_u"][W1]) == set(live)
    # existing consumers still get their keys
    assert "slope_J_ci" in MG.h2_regression_j([cell] * 3, 1.0, n_boot=5) or "note" in MG.h2_regression_j([cell], 1.0)
    # the CLI: h2_theory (primary) and h2_regression_exploratory next to the old keys
    MG.main(["h2", str(run), "--measure", "meta_shadow_s2lite", "--pair", ":".join(pair), "--W", "1",
             "--n-boot", "10"])
    out = json.loads(capsys.readouterr().out)
    assert {"h2_theory", "h2_verdict", "h2_regression_exploratory", "regression_j", "predicted_gain",
            "regression_auroc", "regression", "cells"} <= set(out)
    th = out["h2_theory"][W1]
    assert th["n_cells_total"] == 1 and th["estimable"] is False and th["reason"] == "fewer than 4 estimable cells"
    assert th["cells"][0]["live"] == pair[0] and "falsified" not in {k for k in th}
    assert out["h2_verdict"][W1].startswith("H2 not estimable")
    assert out["h2_regression_exploratory"] == out["regression_j"]
    assert all("task_data" not in c for c in out["cells"])


@pytest.mark.parametrize("n_boot", [0, 1])
def test_degenerate_bootstrap_sizes(n_boot):
    r = MG.h2_theory_check(_exact_cells(4), 1.0, n_boot=n_boot)
    assert r["estimable"] and r["slope"]["value"] == approx(1.0)
    if n_boot == 0:
        assert math.isnan(r["slope"]["ci"][0]) and r["n_boot_effective"] == 0
    else:
        assert r["slope"]["ci"] == approx([1.0, 1.0]) and r["n_boot_effective"] == 1


def test_cells_without_task_data_get_point_estimates_only():
    cells = _exact_cells(4)
    for c in cells:
        c.pop("task_data")
    r = MG.h2_theory_check(cells, 1.0, n_boot=20)
    assert r["estimable"] and r["slope"]["value"] == approx(1.0) and math.isnan(r["slope"]["ci"][0])
    assert "without task data" in r["bootstrap_note"] and r["n_boot_effective"] == 0
    assert all(math.isnan(c["pred_gross_ci"][0]) and c["n_boot_effective"] == 0 for c in r["cells"])
    assert "no task-level CI" in MG.h2_theory_wording(r)
