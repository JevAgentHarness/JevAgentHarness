"""Exploratory row E4 (PREREG_A7_STAGE2 Deviations 2026-10-05): offline checks of the instance rule, the arm configs
(reasoning toggles, caps, routing), the timeout-configurable OpenRouter model, the task-major schedule, the registered
dev threshold rule and the readout. No API, no Docker."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("minisweagent")
import explore_e4_swe as E  # noqa: E402
from jevharness.adapters import mini_swe as MS  # noqa: E402


def test_instance_lists_are_disjoint_fresh_and_match_the_registered_hashes():
    sel = json.loads((E.IDS_DIR / "selection.json").read_text())
    lists = {k: E.read_ids(ROOT / sel[k]["file"]) for k in ("smoke", "dev", "eval")}
    for k, ids in lists.items():
        assert E.sha256_text(E._ids_text(ids)) == sel[k]["sha256"], k
        assert len(ids) == len(set(ids)) == sel[k]["n"]
    assert len(lists["dev"]) == E.N_DEV and len(lists["eval"]) == E.N_EVAL
    h5 = set(E.read_ids(E.H5_EVAL_IDS))
    assert len(h5) == 100
    assert not (set(lists["dev"]) | set(lists["eval"])) & (h5 | set(lists["smoke"]))
    assert not set(lists["dev"]) & set(lists["eval"])
    assert lists["smoke"] == E.read_ids(E.H5_SMOKE_IDS)
    # the rule: dev + eval are the first 145 of the salted sha256 order of the remaining pool
    assert lists["dev"] + lists["eval"] == sorted(lists["dev"] + lists["eval"], key=E.order_key)


def test_select_rule_on_a_toy_pool():
    pool = [f"r__r-{i}" for i in range(20)]
    sel = E.select(pool, pool[:5], pool[5:7], n_dev=3, n_eval=4)
    rest = sorted(pool[7:], key=E.order_key)
    assert sel == {"smoke": pool[5:7], "dev": rest[:3], "eval": rest[3:7]}
    with pytest.raises(ValueError):
        E.select(pool, pool[:5], pool[5:7], n_dev=10, n_eval=10)


def _resolved(arm: str):
    specs, jarm, info = E.arm_specs(arm)
    cfg = MS.load_config([str(ROOT / s) if "=" not in s else s for s in specs])
    return cfg, MS.JevMiniSweConfig.for_arm(cfg["jev"], jarm), info


def test_arm_configs_toggle_reasoning_and_keep_the_caps():
    deep, jd, _ = _resolved("always_deep")
    fast, jf, _ = _resolved("always_fast")
    route, jr, info = _resolved("jevagent_t060")
    for cfg in (deep, fast, route):
        assert cfg["model"]["model_class"] == "jevharness.adapters.mini_swe.OpenRouterTimeoutModel"
        assert cfg["model"]["model_kwargs"]["provider"] == {"order": ["StreamLake"], "allow_fallbacks": False}
        assert cfg["agent"]["step_limit"] == 250 and cfg["agent"]["cost_limit"] == pytest.approx(0.42)
        assert cfg["jev"]["s1"] == {"type": "jev", "provider": "openrouter", "model": None}
    assert deep["model"]["model_name"] == E.DEEP_MODEL and deep["model"]["model_kwargs"]["reasoning"] == {"enabled": True}
    assert fast["model"]["model_name"] == E.FAST_MODEL and fast["model"]["model_kwargs"]["reasoning"] == {"enabled": False}
    assert route["model"] == deep["model"]
    assert not jd.uses_s1() and not jf.uses_s1()
    assert jr.routing.mode == "trigger" and jr.routing.tau == pytest.approx(0.6) and info["tau"] == pytest.approx(0.6)
    assert [h for h in MS.HOOKS if jr.hook(h).mode != "off"] == ["routing"]
    cheap = MS.build_cheap_model(route["model"], jr.cheap_model)
    assert type(cheap).__name__ == "OpenRouterTimeoutModel" and cheap.request_timeout == 600
    assert cheap.config.model_name == E.FAST_MODEL
    assert cheap.config.model_kwargs["reasoning"] == {"enabled": False}
    assert cheap.config.model_kwargs["provider"] == route["model"]["model_kwargs"]["provider"]
    assert cheap.config.observation_template == route["model"]["observation_template"]


class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, json.dumps(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._d


def test_openrouter_timeout_model_passes_timeout_and_reasoning(monkeypatch):
    seen = {}
    ok = {"choices": [{"message": {"role": "assistant", "content": "x", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 1e-6}}

    def post(url, headers=None, data=None, timeout=None):
        seen.update(payload=json.loads(data), timeout=timeout)
        return _Resp(seen.get("next", ok))

    monkeypatch.setattr(MS.requests, "post", post)
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")
    m = MS.OpenRouterTimeoutModel(model_name=E.DEEP_MODEL, request_timeout=321,
                                  model_kwargs={"reasoning": {"enabled": True}})
    out = m.query([{"role": "user", "content": "hi"}])
    assert seen["timeout"] == 321 and seen["payload"]["reasoning"] == {"enabled": True}
    assert seen["payload"]["usage"] == {"include": True} and seen["payload"]["model"] == E.DEEP_MODEL
    assert out["extra"]["cost"] == pytest.approx(1e-6) and out["extra"]["actions"][0]["command"] == "ls"
    seen["next"] = {"error": {"message": "upstream"}}
    from minisweagent.models.openrouter_model import OpenRouterAPIError
    with pytest.raises(OpenRouterAPIError):
        m._query([{"role": "user", "content": "hi"}])
    # an unbilled, cut-off generation (seen on E4 dev) is retried, not booked at $0 or failed in the cost accounting
    seen["next"] = {**ok, "usage": {"prompt_tokens": 18049, "completion_tokens": 14, "cost": 0,
                                    "cost_details": {"upstream_inference_cost": 0.0038}}}
    with pytest.raises(OpenRouterAPIError, match="unbilled"):
        m._query([{"role": "user", "content": "hi"}])
    calls = iter([seen["next"], ok])
    monkeypatch.setattr(MS.requests, "post", lambda *a, **k: _Resp(next(calls)))
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")
    import minisweagent.models.utils.retry as R
    monkeypatch.setattr(R, "wait_exponential", lambda **k: __import__("tenacity").wait_none())
    out = m.query([{"role": "user", "content": "hi"}])
    assert out["extra"]["cost"] == pytest.approx(1e-6)


def test_schedule_is_task_major_and_complete():
    ids, arms = ["a", "b", "c"], ["always_deep", "always_fast", "jevagent"]
    jobs = E.schedule(ids, arms)
    assert sorted(jobs) == sorted((i, a) for i in ids for a in arms)
    assert [j[0] for j in jobs] == [i for i in ids for _ in arms]
    assert jobs == E.schedule(ids, arms)


def test_tau_of_and_eval_gate(tmp_path):
    assert E.tau_of("jevagent_t050") == 0.5 and E.tau_of("jevagent_t070") == pytest.approx(0.7)
    th = tmp_path / "threshold.json"
    with pytest.raises(SystemExit):
        E.check_eval_gate(th, E.IDS_DIR / "eval_ids.txt")
    sel = json.loads((E.IDS_DIR / "selection.json").read_text())
    th.write_text(json.dumps({"tau": 0.6, "eval_ids_sha256": sel["eval"]["sha256"]}))
    assert E.check_eval_gate(th, E.IDS_DIR / "eval_ids.txt") == 0.6
    th.write_text(json.dumps({"tau": 0.6, "eval_ids_sha256": "0" * 64}))
    with pytest.raises(SystemExit):
        E.check_eval_gate(th, E.IDS_DIR / "eval_ids.txt")
    with pytest.raises(SystemExit):
        E.check_eval_gate(tmp_path / "missing.json", E.IDS_DIR / "eval_ids.txt")


def test_threshold_rule_cases():
    shares = {"jevagent_t050": 0.2, "jevagent_t060": 0.4, "jevagent_t070": 0.7}
    # always_fast within the margin: dev cannot separate -> registered route tau
    r = E.choose_threshold({"always_deep": 30, "always_fast": 28, "jevagent_t050": 30, "jevagent_t060": 30,
                            "jevagent_t070": 30}, shares)
    assert r["case"] == "not_separable" and r["tau"] == 0.5
    # the largest fast-share within 2 of always_deep
    r = E.choose_threshold({"always_deep": 30, "always_fast": 20, "jevagent_t050": 29, "jevagent_t060": 28,
                            "jevagent_t070": 27}, shares)
    assert r["case"] == "rule" and r["tau"] == pytest.approx(0.6)
    # better than always_deep is eligible too
    r = E.choose_threshold({"always_deep": 30, "always_fast": 20, "jevagent_t050": 29, "jevagent_t060": 26,
                            "jevagent_t070": 31}, shares)
    assert r["tau"] == pytest.approx(0.7)
    # none within the margin -> registered route tau
    r = E.choose_threshold({"always_deep": 30, "always_fast": 20, "jevagent_t050": 27, "jevagent_t060": 26,
                            "jevagent_t070": 25}, shares)
    assert r["case"] == "none_eligible" and r["tau"] == 0.5
    # ties in fast-share (3 decimals) -> the lower tau
    r = E.choose_threshold({"always_deep": 30, "always_fast": 20, "jevagent_t050": 30, "jevagent_t060": 30,
                            "jevagent_t070": 20}, {"jevagent_t050": 0.4001, "jevagent_t060": 0.4004,
                                                   "jevagent_t070": 0.9})
    assert r["tau"] == 0.5


def _traj(iid, arm, success, wall_s, cost, s1_cost, deep_calls, fast_calls, error=None):
    per_model = {}
    if deep_calls:
        per_model["strong"] = {"model": E.DEEP_MODEL, "calls": deep_calls}
    if fast_calls:
        per_model["cheap" if deep_calls else "strong"] = {"model": E.FAST_MODEL, "calls": fast_calls}
    return {"task_id": iid, "config": arm, "steps": [{}] * (deep_calls + fast_calls), "wall_ms": wall_s * 1000,
            "error": error, "result": {"success": success, "resolved": success, "exit_status": "Submitted"},
            "usage": {"total": {"cost_usd": cost + s1_cost}, "s2": {"cost_usd": cost},
                      "s1": {"cost_usd": s1_cost, "calls": deep_calls + fast_calls if s1_cost else 0}},
            "meta": {"exit_status": "Submitted", "model_stats": {"per_model": per_model}}}


def test_summary_fast_share_saving_and_cosolved_time(tmp_path):
    rows = {"always_deep": [_traj("i1", "d", True, 100, 0.10, 0, 10, 0), _traj("i2", "d", True, 200, 0.20, 0, 20, 0),
                            _traj("i3", "d", False, 300, 0.30, 0, 30, 0)],
            "always_fast": [_traj("i1", "f", True, 50, 0.01, 0, 0, 10), _traj("i2", "f", False, 60, 0.02, 0, 0, 20),
                            _traj("i3", "f", False, 70, 0.03, 0, 0, 30)],
            "jevagent": [_traj("i1", "j", True, 80, 0.05, 0.001, 5, 5), _traj("i2", "j", True, 150, 0.10, 0.001, 10, 10),
                         _traj("i3", "j", True, 200, 0.15, 0.001, 15, 15)]}
    for arm, ts in rows.items():
        (tmp_path / f"{arm}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ts))
    s = E.summarize(tmp_path, list(rows))
    by = {r["arm"]: r for r in s["arms"]}
    assert by["always_deep"]["fast_share"] == 0 and by["always_fast"]["fast_share"] == 1
    assert by["jevagent"]["fast_share"] == pytest.approx(0.5) and by["jevagent"]["resolved"] == 3
    pj = s["paired"]["jevagent"]
    assert (pj["d_resolved"], pj["wins"], pj["losses"]) == (1, 1, 0)
    assert pj["saving_pct"] == pytest.approx(100 * (1 - 0.303 / 0.6))
    assert pj["n_cosolved"] == 2 and pj["d_time_median_cosolved_s"] == pytest.approx(-35.0)
    pf = s["paired"]["always_fast"]
    assert (pf["d_resolved"], pf["wins"], pf["losses"]) == (-1, 0, 1)
    vals = E.tex_values(s)
    assert vals["e4/jevagent/fast_share"] == "50.0" and vals["e4/jevagent/vs_deep/delta"] == "$+$1"
    assert vals["e4/always_fast/vs_deep/time_cosolved/n"] == "1"
    md = E.markdown(s, "t")
    assert "exploratory (E4), descriptive" in md
    E.per_instance_csv(s, tmp_path / "p.csv")
    assert len((tmp_path / "p.csv").read_text().splitlines()) == 1 + 9


def test_e4b_arms_share_one_model_and_differ_only_in_thinking():
    try:
        E.set_exp("e4b")
        deep, jd, _ = _resolved("always_deep")
        fast, jf, _ = _resolved("always_fast")
        route, jr, _ = _resolved("jevagent_t050")
        assert deep["model"]["model_name"] == fast["model"]["model_name"] == "gemini/gemini-3.8-flash"
        assert deep["model"]["model_kwargs"]["thinkingConfig"] == {"thinkingLevel": "high"}
        assert fast["model"]["model_kwargs"]["thinkingConfig"] == {"thinkingLevel": "low"}
        assert deep["agent"]["cost_limit"] == pytest.approx(1.5) and deep["agent"]["step_limit"] == 250
        cheap = MS.build_cheap_model(route["model"], jr.cheap_model)
        assert cheap.config.model_name == "gemini/gemini-3.8-flash"
        assert cheap.config.model_kwargs["thinkingConfig"] == {"thinkingLevel": "low"}
        assert jr.routing.mode == "trigger" and jr.s1["provider"] == "openrouter"
        sel = json.loads((E.IDS_DIR / "selection.json").read_text())
        e4sel = json.loads((ROOT / "configs/explore/e4_swe/selection.json").read_text())
        assert sel["eval"]["sha256"] == e4sel["eval"]["sha256"]                   # the eval list is shared with E4
        assert E.read_ids(E.IDS_DIR / "dev_ids.txt") == E.read_ids(ROOT / "configs/explore/e4_swe/dev_ids.txt")[:30]
        assert E.PHASES["dev"]["arms"] == ["always_deep", "always_fast", "jevagent_t050", "jevagent_t060"]
        assert E.PHASES["eval"]["arms"] == ["always_deep", "always_fast", "jevagent"]
    finally:
        E.set_exp("e4")
    assert E.PHASES["dev"]["arms"][-1] == "jevagent_t070" and E.EXP == "e4"


def test_fast_share_reads_the_role_when_both_paths_are_one_model(tmp_path):
    t = _traj("i1", "j", True, 10, 0.1, 0.0, 0, 0)
    t["meta"]["model_stats"]["per_model"] = {"strong": {"model": "m", "calls": 6}, "cheap": {"model": "m", "calls": 4}}
    (tmp_path / "jevagent.jsonl").write_text(json.dumps(t) + "\n")
    (tmp_path / "always_fast.jsonl").write_text(json.dumps({**t, "meta": {**t["meta"], "model_stats": {
        "per_model": {"strong": {"model": "m", "calls": 7}}}}}) + "\n")
    assert E.fast_share(E.load_arm(tmp_path / "jevagent.jsonl")) == pytest.approx(0.4)
    assert E.fast_share(E.load_arm(tmp_path / "always_fast.jsonl")) == 1.0


# ------------------------------------------------------------------------------- E4r (Deviations 2026-10-07)
def _e4r(fn):
    try:
        E.set_exp("e4r")
        return fn()
    finally:
        E.set_exp("e4")


def test_e4r_lists_replicate_e4_and_extend_with_fresh_instances_only():
    def check():
        sel = json.loads((E.IDS_DIR / "selection.json").read_text())
        lists = {k: E.read_ids(ROOT / sel[k]["file"]) for k in ("smoke", "replicate", "extension", "eval", "reserve")}
        for k, ids in lists.items():
            assert E.sha256_text(E._ids_text(ids)) == sel[k]["sha256"], k
            assert len(ids) == len(set(ids)) == sel[k]["n"]
        e4 = ROOT / "configs/explore/e4_swe"
        assert lists["replicate"] == E.read_ids(e4 / "eval_ids.txt")                 # E4's eval list, unchanged
        assert lists["smoke"] == [E.E4R_SMOKE_ID] and E.E4R_SMOKE_ID in E.read_ids(E.H5_SMOKE_IDS)   # 1 instance
        used = set().union(*map(set, E.e4r_exclusions().values()))
        used |= set(E.read_ids(ROOT / "configs/explore/e4b_swe/dev_ids.txt")) | set(E.read_ids(ROOT / "configs/explore/e4b_swe/eval_ids.txt"))
        assert len(used) == 250 and not (set(lists["extension"]) | set(lists["reserve"])) & used
        assert len(lists["extension"]) == E.N_EXT == 200 and len(lists["reserve"]) == 50
        rest = lists["extension"] + lists["reserve"]                                 # the pool minus every used id
        assert rest == sorted(rest, key=E.e4r_order_key)
        assert lists["eval"] == lists["extension"]          # minimum-spend amendment: the replicate is not run again
        assert sel["superseded"]["eval"]["sha256"].startswith("99831a6c") and sel["superseded"]["eval"]["n"] == 300
        th = json.loads((E.IDS_DIR / "threshold.json").read_text())
        assert th["tau"] == json.loads((e4 / "threshold.json").read_text())["tau"] == 0.5
        assert E.check_eval_gate(E.IDS_DIR / "threshold.json", E.IDS_DIR / "eval_ids.txt") == 0.5
        assert E.PHASES["eval"]["arms"] == ["always_deep", "always_fast", "jevagent"]
        assert E.tau_of("jevagent") == 0.5 and E.EXP == "e4r" and E.LABEL.startswith("exploratory (E4r)")
    _e4r(check)


def test_e4r_select_rule_and_interleave_on_a_toy_pool():
    assert E.interleave(["r1", "r2"], ["e1", "e2", "e3", "e4"]) == ["r1", "e1", "e2", "r2", "e3", "e4"]
    pool = [f"r__r-{i}" for i in range(30)]
    ex = {"a": pool[:5], "b": pool[5:10]}
    sel = E.select_e4r(pool, pool[5:9], ex, pool[:2], n_ext=8)
    rest = sorted(pool[10:], key=E.e4r_order_key)
    assert sel["extension"] == rest[:8] and sel["reserve"] == rest[8:] and sel["replicate"] == pool[5:9]
    assert sel["eval"] == rest[:8] and sel["smoke"] == [pool[0]]                 # toy pool: no E4R_SMOKE_ID
    with pytest.raises(ValueError):
        E.select_e4r(pool, pool[20:22], ex, pool[:2], n_ext=8)                 # a replicate id not among the used ones
    with pytest.raises(ValueError):
        E.select_e4r(pool, pool[5:9], ex, pool[:2], n_ext=25)


def test_e4r_arms_are_e4s_arms():
    def specs():
        return {a: E.arm_specs(a) for a in ("always_deep", "always_fast", "jevagent_t050")}
    assert _e4r(specs) == specs()
    assert E.EXPS["e4r"]["E4_CONFIG"] == E.E4_CONFIG and E.EXPS["e4r"]["DEEP_MODEL"] == E.DEEP_MODEL


def _row(success, t, cost, s1=0.0, deep=10, fast=0, err=None):
    return {"success": float(success), "resolved": bool(success), "time_s": float(t), "error": err,
            "exit_status": "Submitted", "steps": deep + fast, "llm_calls": deep + fast, "fast_calls": fast,
            "deep_calls": deep, "s2_cost": cost, "s1_cost": s1, "total_cost": cost + s1, "s1_calls": deep + fast if s1 else 0}


def test_e4r_readout_views_ni_savings_agreement_and_tex(tmp_path):
    rep = [f"r__r-{i}" for i in range(10)]
    ext = [f"e__e-{i}" for i in range(20)]
    d = tmp_path / "ids"
    d.mkdir()
    (d / "replicate_ids.txt").write_text(E._ids_text(rep))
    (d / "extension_ids.txt").write_text(E._ids_text(ext))
    (d / "threshold.json").write_text(json.dumps({"tau": 0.5}))
    ids = rep + ext
    rows = {"always_deep": {k: _row(i % 5 != 0, 100, 0.04) for i, k in enumerate(ids)},
            "jevagent": {k: _row(i % 5 != 0 or i == 5, 90, 0.03, 0.003, 7, 3) for i, k in enumerate(ids)},
            "always_fast": {k: _row(i % 3 != 0, 60, 0.01, 0, 0, 12) for i, k in enumerate(ids)}}
    e4 = {a: {k: dict(v) for k, v in rows[a].items() if k in rep} for a in rows}
    e4["always_deep"][rep[1]] = _row(False, 100, 0.04)
    r = E.e4r_readout(rows, e4, {}, ids_dir=d)
    pp = r["views"]["pooled"]["paired"]["jevagent"]                              # E4's run on rep + rows on ext
    assert (pp["wins"], pp["losses"]) == (2, 0) and r["views"]["pooled"]["n"] == 30
    pj = r["views"]["e4r_run"]["paired"]["jevagent"]                             # the original design's own pool
    assert r["views"]["e4r_run"]["n"] == 30 and r["views"]["replicate"]["n"] == 10 and r["views"]["extension"]["n"] == 20
    assert (pj["wins"], pj["losses"], pj["delta_pt"]) == (1, 0, pytest.approx(100 / 30))
    assert pj["cost_saving_pct"] == pytest.approx(100 * (1 - 0.033 / 0.04))
    assert pj["time_saving_pct"] == pytest.approx(10.0)
    lo, hi = pj["cost_saving_ci"]
    assert lo == pytest.approx(pj["cost_saving_pct"]) and hi == pytest.approx(pj["cost_saving_pct"])  # constant ratio
    assert pj["delta_boot_ci_pt"][0] <= pj["delta_pt"] <= pj["delta_boot_ci_pt"][1]
    from jevharness.bench import metrics as M
    assert pj["ni_p"] == pytest.approx(M.ni_certified_p(1, 0, 30, 0.03)["p"]) and pj["ni"] in ("yes", "no")
    g = r["agreement"]["always_deep"]
    assert (g["n"], g["e4_resolved"], g["e4r_resolved"], g["only_e4r"], g["only_e4"]) == (10, 7, 8, 1, 0)
    assert g["same_outcome_pct"] == pytest.approx(90.0)
    assert r["agreement"]["jevagent"]["kappa"] == pytest.approx(1.0)
    assert r["views"]["e4"]["n"] == 10
    assert set(r["agreement_routing"]) == {"e4", "e4r"}
    v = E.e4r_tex_values(r)
    assert v["e4r/n_eval"] == "30" and v["e4r/jevagent/vs_deep/wins"] == "2" and v["e4r/rep/n_eval"] == "10"
    assert v["e4r/jevagent/vs_deep/delta"] == "$+$6.7" and v["e4r/ext/always_fast/n"] == "20"
    assert v["e4r/e4rrun/jevagent/vs_deep/wins"] == "1" and v["e4r/e4/n_eval"] == "10"
    # the amended design: E4r ran the extension only -> pooled = E4 + extension, no replicate view, no agreement
    ext_rows = {a: {k: x for k, x in rows[a].items() if k in ext} for a in rows}
    r3 = E.e4r_readout(ext_rows, e4, {}, ids_dir=d)
    assert set(r3["views"]) == {"pooled", "extension", "e4"} and r3["agreement"] == {}
    assert r3["views"]["pooled"]["n"] == 30 and r3["views"]["pooled"]["paired"]["jevagent"]["wins"] == 2
    assert "Replicate agreement" not in E.e4r_markdown(r3)
    assert v["e4r/agree/always_deep/e4r_resolved"] == "8" and v["e4r/tau"] == "0.50"
    md = E.e4r_markdown(r)
    assert "Replicate agreement" in md and "exploratory" in md
    # a missing record drops the instance from every arm of the view (complete set)
    del rows["always_fast"][ext[0]]
    r2 = E.e4r_readout(rows, e4, {}, ids_dir=d)
    assert r2["views"]["pooled"]["n"] == 29 and r2["views"]["pooled"]["missing"] == 1


def test_rows_from_csv_round_trips_per_instance_csv(tmp_path):
    rows = {"always_deep": {"a__a-1": _row(True, 12.5, 0.02), "a__a-2": {**_row(False, 3, 0.01), "resolved": None}},
            "jevagent": {"a__a-1": _row(True, 10.0, 0.01, 0.002, 3, 2)}}
    E.per_instance_csv({"data": rows}, tmp_path / "p.csv")
    back = E.rows_from_csv(tmp_path / "p.csv")
    assert back["always_deep"]["a__a-1"]["success"] == 1.0 and back["always_deep"]["a__a-2"]["resolved"] is None
    assert back["jevagent"]["a__a-1"]["total_cost"] == pytest.approx(0.012)
    assert back["jevagent"]["a__a-1"]["fast_calls"] == 2 and back["always_deep"]["a__a-1"]["time_s"] == pytest.approx(12.5)
    e4 = E.rows_from_csv(ROOT / "results/e4_swe/per_instance.csv")                # E4's committed rows load
    assert {a: len(v) for a, v in e4.items()} == {"always_deep": 100, "always_fast": 100, "jevagent": 100}
    assert sum(r["success"] for r in e4["jevagent"].values()) == 81


def test_e4r_gate_decision_full_cut_and_stop():
    e4 = {"always_deep": 0.0411, "jevagent": 0.0324, "always_fast": 0.0108}
    assert E.E4R_CAP_USD == 40 and E.E4R_FALLBACK_K == 100
    d = E.gate_decision({"always_deep": 0.03, "jevagent": 0.03, "always_fast": 0.01}, e4, 0.05, 200)
    assert d["decision"] == "full" and d["k"] == 200                 # the larger of smoke and E4 per arm, x 1.5
    assert d["projection"] == pytest.approx(1.5 * 200 * (0.0411 + 0.0324 + 0.0108))     # about $25 of the $40
    d = E.gate_decision({"always_deep": 0.08, "jevagent": 0.07, "always_fast": 0.02}, e4, 0.05, 200)
    assert d["decision"] == "cut" and d["k"] == 100                  # 200 do not fit $40, the first 100 do
    d = E.gate_decision({"always_deep": 0.2, "jevagent": 0.2, "always_fast": 0.05}, e4, 0.05, 200)
    assert d["decision"] == "stop" and d["k"] == 0


def _traj_file(path, msgs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"messages": [{"role": "assistant", "extra": {"response": {
        "model": m, "provider": p, "usage": {"completion_tokens_details": {"reasoning_tokens": rt}}}}}
        for m, p, rt in msgs]}))


def test_e4r_smoke_check_reads_provider_and_reasoning_toggle(tmp_path):
    ids = ["a__a-1"]
    deep, fast = E.DEEP_MODEL + "-20260423", E.FAST_MODEL + "-20260423"
    plan = {"always_deep": [(deep, "StreamLake", 120)], "always_fast": [(fast, "StreamLake", 0)],
            "jevagent": [(deep, "StreamLake", 40), (fast, "StreamLake", 0)]}
    for arm, msgs in plan.items():
        _traj_file(tmp_path / arm / ids[0] / f"{ids[0]}.traj.json", msgs)
        (tmp_path / arm / ids[0] / f"{ids[0]}.jev.json").write_text(json.dumps({"error": None, "s1_errors": [], "usage": {
            "total": {"cost_usd": 0.01}, "s1": {"calls": 2 if arm == "jevagent" else 0}}}))
    r = E.smoke_check(tmp_path, list(plan), ids)
    assert r["ok"] and (r["deep_calls"], r["deep_reasoning"], r["fast_calls"]) == (2, 2, 2)
    assert r["s1_calls"] == 2 and r["s1_errors"] == 0
    jp = tmp_path / "jevagent" / ids[0] / f"{ids[0]}.jev.json"                 # Jev silent: the route did not run
    ok_rec = jp.read_text()
    jp.write_text(json.dumps({"error": None, "s1_errors": ["timeout"] * 2, "usage": {"total": {"cost_usd": 0.01},
                                                                                     "s1": {"calls": 2}}}))
    assert any(f.startswith("Jev") for f in E.smoke_check(tmp_path, list(plan), ids)["fails"])
    jp.write_text(ok_rec)
    assert E.logged_spend(tmp_path) == pytest.approx(0.03)
    _traj_file(tmp_path / "always_fast" / ids[0] / f"{ids[0]}.traj.json", [(fast, "Other", 5)])
    r = E.smoke_check(tmp_path, list(plan), ids)
    assert not r["ok"] and len(r["fails"]) == 2                     # wrong provider and a reasoning fast call
    (tmp_path / "jevagent" / ids[0] / f"{ids[0]}.jev.json").write_text(json.dumps({"error": "boom"}))
    assert any("infrastructure" in f for f in E.smoke_check(tmp_path, list(plan), ids)["fails"])


def test_e4r_driver_is_valid_bash_refuses_other_hosts_and_writes_no_secret():
    import subprocess
    drv = ROOT / "scripts/aws/e4r_swe.sh"
    assert subprocess.run(["bash", "-n", str(drv)]).returncode == 0
    src = drv.read_text()
    forb = src.split('FORBIDDEN_IIDS="', 1)[1].split('"', 1)[0].split()
    for iid in ("<instance-id>", "<instance-id>", "<instance-id>", "<instance-id>",
                "<instance-id>"):
        assert iid in forb
    assert "put-parameter" not in src and "OPENROUTER_API_KEY_JEV" not in src       # reads SSM only; never the Jev-only key
    for stage in ("rearm", "setup", "update", "prepull", "smoke", "gate", "eval", "final"):
        assert f"stage_{stage}()" in src
    assert 'EXP=e4r PHASE=eval' in src and 'EXP=e4r PHASE=smoke' in src
    e4 = (ROOT / "scripts/aws/e4_swe.sh").read_text()
    assert "e4r) STATUS_S3=${STATUS_S3:-$B/e4r/status}" in e4                      # E4's status log is never overwritten
