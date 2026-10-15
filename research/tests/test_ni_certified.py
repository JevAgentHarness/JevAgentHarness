"""A6.21 (review round 8, W1): the certified NI p-value ``bb_ordered`` (metrics.ni_certified_p).

The validity argument is the proof in metrics.py / docs/drafts/A6_21_draft.md; these tests check its ingredients and
its consequences numerically: the ordering is a monotone hull, the Hoelder bound and the branch-and-bound supremum are
upper bounds (against brute force), the exact size is <= alpha (exhaustive at a small n, exact rejection regions at a
registered n, and the committed round-8 table), p-values lie in [gamma, 1], and the rule agrees with the superseded
A6.17 rule on clear cases."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from jevharness.bench import metrics as M

ROOT = Path(__file__).resolve().parents[1]


def _tail(region: np.ndarray, n: int, m: float, ws) -> np.ndarray:
    """P_w(region) by full enumeration of the triangle (no window), for each w."""
    W, L = np.nonzero(region)
    keep = W + L <= n
    W, L = W[keep], L[keep]
    lgt = M._lgamma_table(n)
    return np.array([float(np.exp(M._trinomial_logpmf(float(w), m, W, L, n, lgt)).sum()) for w in ws])


def _trinomial(n: int, p_win: float, p_loss: float) -> np.ndarray:
    """(n + 1) x (n + 1) pmf of (wins, losses), zero off the triangle."""
    lf = M._lgamma_table(n)
    W, L = np.indices((n + 1, n + 1))
    T = n - W - L
    out = np.zeros((n + 1, n + 1))
    tri = T >= 0
    pt = 1.0 - p_win - p_loss

    def term(k, p):
        return np.where(k == 0, 0.0, k * math.log(p)) if p > 0 else np.where(k == 0, 0.0, -np.inf)
    lp = lf[n] - lf[W[tri]] - lf[L[tri]] - lf[T[tri]] + term(W[tri], p_win) + term(L[tri], p_loss) + term(T[tri], pt)
    out[tri] = np.exp(lp)
    return out


def test_order_table_is_the_monotone_hull_of_the_switching_rule():
    n, m = 25, 0.05
    s, st = M.ni_order_table(n, m, hull=False), M.ni_order_table(n, m)
    for W in range(n + 1):
        for L in range(n + 1 - W):
            dom = [s[wi, li] for wi in range(W + 1) for li in range(L, n + 1 - wi)]
            assert st[W, L] == min(dom)                                     # brute-force definition of the hull
            assert st[W, L] <= s[W, L]
            if W + 1 + L <= n:
                assert st[W + 1, L] <= st[W, L]                             # more wins: at least as significant
            if W + L + 1 <= n:
                assert st[W, L + 1] >= st[W, L]                             # more losses: at most as significant
    # the nominal statistic is the A6.14 switching rule: exact loss-count p off the score region, Tango's p on it
    assert math.exp(s[0, 3]) == approx(M._binom_cdf(3, n, m))
    assert math.exp(s[2, 2]) == approx(M._binom_cdf(2, n, m))                # W + L < 5: still the loss-count p
    assert math.exp(s[6, 3]) == approx(M._norm_sf(M._tango_z(-m, 6, 3, n)))
    assert np.all(np.isinf(st[np.add.outer(np.arange(n + 1), np.arange(n + 1)) > n]))


def test_holder_bound_is_an_upper_bound():
    rng = np.random.default_rng(7)
    worst = -np.inf
    for _ in range(150):
        n, m = int(rng.choice([30, 80, 149])), float(rng.choice([0.03, 0.05]))
        wmax = (1 - m) / 2
        st = M.ni_order_table(n, m)
        W0, L0 = int(rng.integers(0, n // 3)), int(rng.integers(0, n // 3))
        region = st <= st[W0, L0] + M.NI_BB_TIE
        a = float(rng.uniform(1e-4, wmax - 1e-4))
        x = float(np.clip(a + rng.normal(0, 0.02), 0.0, wmax))
        fa, fx = _tail(region, n, m, [a, x])
        if fa <= 0:
            continue
        b = float(M.ni_holder_logbound(math.log(fa), a, x, n, m)[0])
        worst = max(worst, math.log(max(fx, 1e-300)) - b)
        # the bound at x covers the whole segment between a and x (M_q(a, .) is convex with minimum 1 at a)
        for y in np.linspace(a, x, 5):
            assert _tail(region, n, m, [y])[0] <= math.exp(b) * (1 + 1e-9)
    assert worst <= 1e-12
    # no absolute continuity from the edges of the path: the bound is infinite there
    assert np.isinf(M.ni_holder_logbound(-3.0, 0.0, 0.01, 100, 0.03)[0])
    assert np.isinf(M.ni_holder_logbound(-3.0, 0.25, 0.20, 100, 0.5)[0])       # (1 - m)/2 with P(tie) = 0 exactly


@pytest.mark.parametrize("n,m,W,L", [(149, 0.05, 20, 18), (149, 0.05, 0, 9), (500, 0.03, 25, 20), (500, 0.03, 2, 10),
                                     (500, 0.03, 5, 5)])
def test_certified_sup_bounds_a_dense_grid_within_tolerance(n, m, W, L):
    r = M.ni_certified_p(W, L, n, m)
    assert r["converged"] and r["ub"] - r["lb"] <= max(M.NI_BB_ABS_TOL, M.NI_BB_REL_TOL * r["lb"]) + 2 * n * 1e-15
    st = M.ni_order_table(n, m)
    lo, hi = r["w_set"]
    dense = _tail(st <= st[W, L] + M.NI_BB_TIE, n, m, np.linspace(lo, hi, 1201)).max()
    assert r["ub"] >= dense                                                  # certified: above every grid value
    assert r["ub"] <= dense * (1 + 2 * M.NI_BB_REL_TOL) + 2 * M.NI_BB_ABS_TOL     # and within the stated tolerance
    assert r["p"] == approx(min(1.0, M.NI_BB_GAMMA + r["ub"]))
    assert M.NI_BB_GAMMA <= r["p"] <= 1.0


def test_exact_size_is_at_most_alpha_exhaustively_at_small_n():
    """Every outcome's p-value at n = 30, then the exact size on boundary AND interior null points (a check of the
    proof's conclusion, including off any grid and away from the boundary), at three levels."""
    n, m = 30, 0.05
    P = np.full((n + 1, n + 1), np.inf)
    for W in range(n + 1):
        for L in range(n + 1 - W):
            P[W, L] = M.ni_certified_p(W, L, n, m)["p"]
    tri = np.isfinite(P)
    assert np.all(P[tri] >= M.NI_BB_GAMMA) and np.all(P[tri] <= 1.0) and P[0, n] == 1.0     # capped at 1
    points = [(w, w + m) for w in (0.0, 0.0007, 0.013, 0.0377, 0.1111, 0.2733, 0.4, 0.475)]
    points += [(w, w + m + e) for w in (0.0, 0.02, 0.1, 0.3) for e in (0.005, 0.05, 0.2) if 2 * w + m + e <= 1]
    for alpha in (0.05, 0.05 / 6, 0.05 / 11):
        for p_win, p_loss in points:
            size = float(_trinomial(n, p_win, p_loss)[P <= alpha].sum())
            assert size <= alpha * (1 + 1e-12), (alpha, p_win, p_loss, size)


def test_exact_size_at_a_registered_n_and_in_the_round8_table():
    """n = 149 (C7), m = 5 pt: the deployed rule's exact rejection regions (scripts/ni_exact_size.CertifiedRule, gray
    cells decided by ni_certified_p) have size <= alpha at every Holm level on a grid of boundary and interior points;
    and the committed round-8 table (n = 500 / 149 / 300 / 1000 / 1114) shows no cell above its level."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from ni_exact_size import LEVELS, CertifiedRule  # noqa: PLC0415

    cr = CertifiedRule(149, 0.05, LEVELS, rho_max=0.8, workers=1, log=lambda *_: None)
    assert cr.info["gray_all_converged"]
    pts = [(w, w + 0.05) for w in (0.0, 0.001, 0.004, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35)]
    pts += [(w, w + 0.05 + e) for w in (0.0, 0.05, 0.2) for e in (0.01, 0.1)]
    for p_win, p_loss in pts:
        r = cr.probs(p_win, p_loss)
        assert r["mass_beyond_Dmax"] < 1e-12
        for lv, a in LEVELS.items():
            assert r["reject"][lv] <= a * (1 + 1e-12), (lv, p_win, p_loss, r["reject"][lv])
    path = ROOT / "results/reviews/2026-10-02/ni_exact_size_r8.json"
    if path.exists():
        res = json.loads(path.read_text())
        assert {st["n"] for st in res["settings"]} >= {500, 149, 1114}
        for st in res["settings"]:
            assert not st["bb"]["bracket_check"]["violations"] and st["bb"]["info"]["gray_all_converged"]
            for d in st["bb"]["dense"] + st["bb"]["interior"]:
                for lv, s in d["size"].items():
                    assert s <= res["levels"][lv] * (1 + 1e-9), (st["n"], lv, s)


def test_agreement_with_the_superseded_rule_on_clear_cases():
    # ordinary counts: within the branch-and-bound tolerance of the A6.17 p_U (same score ordering there)
    for W, L, n, m in ((25, 20, 500, 0.03), (50, 60, 500, 0.03), (40, 30, 149, 0.05)):
        new, old = M.ni_certified_p(W, L, n, m)["p"], M.ni_unconditional_p(W, L, n, m)["p"]
        assert new == approx(old, rel=3e-3)
    # clear decisions coincide at the Holm floor and at 0.05
    for W, L, n in ((25, 15, 500), (0, 0, 500), (0, 60, 500), (10, 40, 500), (60, 20, 500)):
        new = M.ni_certified_p(W, L, n, 0.03)["p"]
        old = (M.ni_exact_check(W, L, n, 0.03)["p"] if W == 0 or W + L < 5
               else M.ni_unconditional_p(W, L, n, 0.03)["p"])
        for a in (0.05, 0.05 / 11):
            assert (new <= a) == (old <= a), (W, L, a, new, old)
    # losses only: the w = 0 tail is the loss-count tail, so p >= the exact loss-count p + gamma
    for L in (3, 6, 10):
        r = M.ni_certified_p(0, L, 500, 0.03)
        assert r["w_set"][0] == 0.0 and r["p"] >= M._binom_cdf(L, 500, 0.03) + M.NI_BB_GAMMA - 1e-12


def test_ni_pvalue_decides_on_the_certified_p_and_keeps_the_comparison_rules():
    n = 300
    base = {f"t{i}": 1.0 if i >= 12 else 0.0 for i in range(n)}            # 12 wins (reference fails, arm succeeds)
    arm = {f"t{i}": 1.0 if (i < 12 or i >= 22) else 0.0 for i in range(n)}  # 10 losses
    r = M.ni_pvalue(base, arm, 0.03, n_boot=300, alpha_adj=0.05 / 11)
    c = M.ni_certified_p(12, 10, n, 0.03)
    assert (r["wins"], r["losses"]) == (12, 10) and r["rule"] == "bb_ordered"
    assert r["p_decisive"] == r["p_bb"] == c["p"] and r["bb"]["ub"] == c["ub"]
    assert r["ni_adj"] == (c["p"] <= 0.05 / 11) and r["ni_95"] == (c["p"] <= 0.05)
    assert M.ni_pvalue(base, arm, 0.03, n_boot=300, rule="tango_uncond")["rule"] == "tango_uncond"
    assert M.ni_pvalue(base, arm, 0.03, n_boot=300, unconditional=False)["rule"] == "tango"
    with pytest.raises(ValueError):
        M.ni_pvalue(base, arm, 0.03, n_boot=300, rule="nope")
    assert math.isnan(M.ni_certified_p(1, 1, 10, 0.0)["p"])                # no positive margin: undefined


def test_c5_sign_flip_equals_the_studentized_test_and_the_paired_t_helper():
    """A6.21 (b): under sign flips sum d^2 is fixed, so the sign-flip p of the mean equals that of the paired t
    statistic; the proposed paired t-test uses an exact Student-t tail (checked against tabulated quantiles)."""
    rng = np.random.default_rng(3)
    x = np.where(rng.random(300) < 0.1, rng.choice([-1.0, 1.0], 300), 0.0) + rng.normal(0, 1e-4, 300) + 0.004
    r = M.sign_flip_test(list(x), n_perm=4000, seed=0)
    S = np.random.default_rng(0)
    signs = np.concatenate([S.choice([-1.0, 1.0], size=(2000, 300)) for _ in range(2)])
    means = signs @ x / 300
    ss = float(np.sum(x * x))

    def tstat(mu):
        return mu / np.sqrt(np.maximum(ss / 300 - mu * mu, 1e-300) / 299)
    p_t = (np.sum(tstat(means) >= tstat(x.mean()) - 1e-9) + 1) / 4001
    assert r["p_greater"] == approx(p_t)
    # Student-t tail: tabulated one-sided quantiles
    assert M.student_t_sf(2.228138852, 10) == approx(0.025, abs=1e-9)
    assert M.student_t_sf(3.169272673, 10) == approx(0.005, abs=1e-9)
    assert M.student_t_sf(12.706204736, 1) == approx(0.025, abs=1e-9)
    assert M.student_t_sf(-2.228138852, 10) == approx(0.975, abs=1e-9) and M.student_t_sf(0.0, 7) == 0.5
    t = M.paired_t_test(list(x))
    assert t["df"] == 299 and t["t"] == approx(x.mean() / (x.std(ddof=1) / math.sqrt(300)))
    assert t["p_greater"] == approx(M.student_t_sf(t["t"], 299)) and t["p_greater"] + t["p_less"] == approx(1.0)
    assert M.paired_t_test([0.0] * 5)["p_greater"] == 0.5 and M.paired_t_test([0.01] * 5)["p_greater"] == 0.0
