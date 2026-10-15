"""E30 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E30: every open decision model at its cheapest common
deployment"): the offline parts of the host bench (scripts/e30_bench.py: the fixed order, the pass logic) and of the
readout (scripts/explore_e30_cost.py: the harness's parse of each read-out family, the cost formula, the plotted
precision rule, the TeX keys), plus the registration and the driver's guards."""
from __future__ import annotations

import asyncio
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e30_bench as B  # noqa: E402
import explore_e30_cost as X  # noqa: E402

from jevharness.decision.types import Choice  # noqa: E402


# ------------------------------------------------------------------------------------------------ the bench
def test_order_is_a_fixed_permutation():
    ids = [f"i{k}" for k in range(50)]
    a, b = B.shuffled_order(ids), B.shuffled_order(list(reversed(ids)))
    assert a == b and sorted(a) == sorted(ids) and a != sorted(ids)


def test_limited_and_quantiles():
    order = [str(k) for k in range(3991)]
    assert B.limited(order, None) == (order, B.WARMUP, B.LAT_N) == (order, 200, 1000)
    o, w, n = B.limited(order, 40)
    assert o == order[:40] and w == 8 and n == 20
    assert B.pct([1, 2, 3, 4], 0.5) == 2.5 and B.pct([5], 0.9) == 5
    s = B.lat_summary([10.0, 20.0, 30.0])
    assert s["p50_ms"] == 20.0 and s["n"] == 3


def test_run_passes_saturation_agreement_and_latency():
    rows = [{"id": f"d{k}", "kind": "single", "reads": [{"ids": [1, 2], "allowed": [7, 8]}]} for k in range(30)]
    order = B.shuffled_order([r["id"] for r in rows])
    resets = []
    seen = []

    async def decide(r):
        seen.append(r["id"])
        await asyncio.sleep(0.001)
        k = int(r["id"][1:])
        return {"logits": [[float(k % 2), 1.0 - k % 2]]}

    res = asyncio.run(B.run_passes(decide, rows, order, [4, 8], lambda: resets.append(1), warmup=5, lat_n=6))
    assert [p["c"] for p in res["passes"]] == [4, 8]
    assert res["passes"][1]["agree_with_first"] == 1.0
    assert res["answers_c"] == 4 and len(res["answers"]) == 30
    assert res["c1"]["decisions"] == 6 and res["saturated"]["dps"] == max(p["dps"] for p in res["passes"])
    assert len(resets) == 3                                     # before each saturated pass and before C = 1
    assert len(seen) == 5 + 30 + 30 + 6                         # warm-up, two passes, latency


def test_wide_answer_takes_the_vendor_best():
    class Dec:
        def _wide(self, state, spec):
            return {"a": 0.2, "b": 0.5, "c": 0.3}, 10

    a = B.wide_answer(Dec(), "s", {"type": "choice"})
    assert a == {"type": "choice", "choice": "b", "confidence": 0.5, "probabilities": {"a": 0.2, "b": 0.5, "c": 0.3}}


# ------------------------------------------------------------------------------------------------ the readout
def test_cost_formula_and_coordinate():
    usdk = X.usd_per_1000(0.8048, 100.0)
    assert usdk == pytest.approx(0.8048 / 360.0)
    assert X.lg(usdk) == pytest.approx(math.log10(0.8048 / 360.0 / 1000.0))
    assert X.PRICE_H == 0.8048 and X.INSTANCE == "g6.xlarge"


def test_plotted_precision_rule():
    rows = {"bf16": {"acc_pct": 85.0, "usdk": 0.010}, "fp8": {"acc_pct": 84.2, "usdk": 0.007}}
    assert X.plotted(rows, 85.1) == ("fp8", "")                 # 84.2 >= 85.1 - 1
    assert X.plotted(rows, 85.3) == ("bf16", "")                # FP8 more than 1 point below
    assert X.plotted(rows, 90.0) == ("bf16", "below")           # none qualifies: the most accurate, flagged
    assert X.plotted(rows, None) == ("fp8", "no reference")     # BF16 is the reference
    assert X.plotted({}, 80.0) == (None, "not measured")


def _q():
    return Choice("Which tool?", ["get_weather", "none (no fit)", "Send-Mail"], ["d1", None, "d3"],
                  decision_point="tool_select")


def test_predict_every_family():
    q = _q()
    # serve_jev: the label of the most likely code
    assert X.predict("code", q, {"logits": [[0.1, 2.0, 0.3]], "meta": {"labels_norm": ["a", "b", "c"]}}) == "b"
    # the vendor's letters: temperature never changes the argmax; the harness normalises the option label
    meta = {"opt_ids": q.options, "T": 1.75, "type": "choice"}
    assert X.predict("startlux", q, {"logits": [[0.0, 0.5, 3.0]], "meta": meta}) == "sendmail"
    # Intern: softmax, temperature, the reference tie-break (highest probability, then lexical)
    meta = {"values": q.options, "T": 1.99}
    assert X.predict("intern", q, {"logits": [[4.0, 1.0, 1.0]], "meta": meta}) == "getweather"
    assert X.predict("intern", q, {"logits": [[1.0, 2.0, 2.0]], "meta": meta}) == "sendmail"   # "S" < "n"
    # finished wire answers (a wide choice) and the wire row's own prediction
    wire = {"type": "choice", "choice": "get_weather", "confidence": 0.6,
            "probabilities": {"get_weather": 0.6, "none (no fit)": 0.3, "Send-Mail": 0.1}}
    assert X.predict("startlux", q, {"answer": wire}) == "getweather"
    assert X.predict("wire", q, {"pred": "x"}) == "x"
    # the OpenJev helper: softmax at 0.85, probabilities rounded to 4 decimals, then the harness's parse
    meta = {"opt_ids": q.options, "T": 0.85, "round": 4}
    assert X.predict("openjev", q, {"logits": [[-3.0, -1.0, -2.0]], "meta": meta}) == "none(nofit)"


def test_tex_keys_names_and_escaping():
    res = {"models": {"sl27b": {"label": "StartLux-27B", "family": "startlux", "ref_acc_pct": 87.7,
                                "flag": "agreement 87.3 % < 95 %",
                                "rows": {"q4_k_m": {"state": "ok", "usdk": 0.05, "lg": X.lg(0.05), "acc_pct": 87.0,
                                                    "dps": 4.5, "conc": 16, "lat_p50": 250.0, "lat_p90": 400.0,
                                                    "agree_ref": 0.97}},
                                "point": {"usdk": 0.05, "lg": X.lg(0.05), "acc_pct": 87.0, "dps": 4.5, "conc": 16,
                                          "lat_p50": 250.0, "lat_p90": 400.0, "prec": "q4_k_m",
                                          "cheaper_than_e9": 22.0}}}}
    meta = {"instance": "g6.xlarge", "gpu": "NVIDIA L4 24 GB", "price_h": 0.8048, "engine": "vLLM", "version": "0.31.0"}
    k = X.tex_keys(res, meta)
    for f in ("usdk", "lg", "acc", "prec", "dps", "lat_p50", "lat_p90", "conc"):
        assert f"e30/sl27b/{f}" in k
    for f in ("instance", "price_h", "engine", "version"):
        assert f"e30/meta/{f}" in k
    assert k["e30/sl27b/prec"] == r"Q4\_K\_M" and k["e30/sl27b/usdk"] == "0.050"
    assert k["e30/sl27b/lg"] == f"{math.log10(0.05 / 1000):.4f}" and k["e30/meta/price_h"] == "0.8048"
    assert k["e30/sl27b/q4_k_m/agree"] == "97.0" and k["e30/sl27b/vse9"] == "22"
    assert k["e30/sl27b/agree"] == "97.0" and k["e30/sl27b/dacc"] == "-0.7"
    assert k["e30/sl27b/flag"] == r"agreement 87.3 \% $<$ 95 \%"   # no bare % in a TeX definition


def test_models_match_the_registration():
    assert set(X.MODELS) == {"e12", "rl", "e12c", "e28", "sl0.8b", "sl2b", "sl4b", "sl9b", "sl27b", "intern", "e27sp",
                             "openjevhf", "decider", "jevk5", "jevk5v02", "clefflash", "imajev", "openjev"}
    for slug, (_, fam, refkey, _) in X.MODELS.items():
        assert fam in ("code", "startlux", "intern", "wire", "openjev", "decider", "jevk5", "imajev")
        assert refkey == ("e28/l1/acc" if slug == "e28" else f"e9/{X.E9_SLUG.get(slug, slug)}/acc")
    # amendment "E30: JevK5 at the weights E15 loaded": the key slug Figure 4a reads takes E15's weights (stage jevk5m)
    assert X.RUN_SLUG == {"jevk5": "jevk5m", "jevk5v02": "jevk5"} and "jevk5v02" not in X.FIG_OPEN.values()
    reg = (ROOT / "docs/PREREG_A7_STAGE2.md").read_text()
    assert "E30: JevK5 at the weights E15 loaded" in reg and "c4f7fdb3aeab5582336406e78d3bef11bf98833d" in reg
    drv = (ROOT / "scripts/aws/e30_cost.sh").read_text()
    assert "REV_JEVK5_MAIN=c4f7fdb3aeab5582336406e78d3bef11bf98833d" in drv and "stage_jevk5m()" in drv


# ------------------------------------------------------------------------------------------------ registration, driver
def test_registration_and_driver_guards():
    reg = (ROOT / "docs/PREREG_A7_STAGE2.md").read_text()
    assert "E30: every open decision model at its cheapest common deployment" in reg
    drv = (ROOT / "scripts/aws/e30_cost.sh").read_text()
    assert "TAG=jev-e30-cost" in drv and "vllm==0.31.0" in drv and "GUARD_MIN=${GUARD_MIN:-355}" in drv
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/aws/e30_cost.sh")]).returncode == 0
    forb = set(re.findall(r"^i-0[0-9a-f]{16}", (ROOT / "configs/explore/e30_cost/forbidden_iids.txt").read_text(), re.M))
    for iid in ("<instance-id>", "<instance-id>", "<instance-id>", "<instance-id>",
                "<instance-id>", "<instance-id>", "<instance-id>"):
        assert iid in forb
    for v in ("REV_08B=bd4f76a600e23227547fee7bfc1825e12a32764c", "REV_2B=3e2e456409a7fe44be69eee231566f7830872ea2",
              "REV_9B=342aad64840be8eee314455428a5a775f18ef547"):
        assert v in drv                                         # E9's rows' revisions (the E2 ladder's manifests)


def test_openjev_prompt_is_the_helper_text_lane():
    p = B.openjev_prompt("USER: hi", "Which tool?", [("get_weather", "d1"), ("none", "")])
    assert p == ("State:\nUSER: hi\n\nQuestion: Which tool?\nOptions:\n[A] get_weather: d1\n[B] none: \n\n"
                 "Answer with the letter of the best option only.")
    assert B.OPENJEV_LETTERS[:2] == ["A", "B"] and B.OPENJEV_LETTERS[26] == "a" and B.OPENJEV_T == 0.85


def test_decider_and_jevk5_parse_and_the_frontier():
    q = _q()
    meta = {"opt_ids": q.options, "T": 1.11, "round": 4}
    assert X.predict("decider", q, {"logits": [[0.0, 2.0, 1.0]], "meta": meta}) == "none(nofit)"
    meta = {"opt_ids": q.options, "T": 1.532}
    assert X.predict("jevk5", q, {"logits": [[3.0, 2.0, 1.0]], "meta": meta}) == "getweather"
    pts = {"a": (1e-6, 70.0), "b": (2e-6, 80.0), "c": (3e-6, 75.0), "d": (1e-5, 90.0), "e": (1e-6, 69.0)}
    assert X.frontier(pts) == ["a", "b", "d"]
    res = {"models": {"e12": {"point": {"usdk": 0.0045, "acc_pct": 84.8}}, "sl0.8b": {"point": {"usdk": 0.0017,
                                                                                               "acc_pct": 66.0}}}}
    fp = X.figure_points(res, {"e9/jev/lg": "-4.7877", "e9/jev/acc": "86.4"})
    assert set(fp) == {"ours", "jev"} and fp["ours"] == pytest.approx((4.5e-6, 84.8))   # 66 % < 68: not drawn
