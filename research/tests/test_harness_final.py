"""The final JevAgentHarness (exploratory; docs/JEVAGENT_HARNESS.md §8, PREREG_A7_STAGE2 Deviations 2026-10-05):
behaviour-preserving engineering, checked offline.

* ``tool_params.extra.batch_s1``: the argument questions of one call go to System-1 in ONE request; every answer,
  threshold test, executed call and LLM request equals the per-question path's (stub backend);
* the profile ``configs/harness/jevagent_final.yaml`` resolves to the tool + argument cascade at JevAgentRL's frozen
  thresholds with the reflection trigger off;
* ``serve_jev`` fast paths (``fuse_multi``, ``suffix_pad_slack``, ``readout_gather``): the option logits equal the
  stateless path and the Stage-2 server's within fp32 tolerance on a tiny random model, with fewer / smaller forward
  passes; the replayed Stage-2 development requests (old call graph on the Stage-2 server vs the new call graph on the
  fast server) give the same answers; the server reports its lock wait and compute time and the client logs them;
* ``scripts/harness_final.py``: the decision-equivalence rule, the diagnosis and the live analysis on synthetic runs.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.harness.profile import decision_points, load_profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cascade_call_graph import FixedS1, RecordingLLM, TinyEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-4


def _hf():
    spec = importlib.util.spec_from_file_location("harness_final", ROOT / "scripts/harness_final.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


# ------------------------------------------------------------------ batched argument questions
def _cfg(batch: bool, tp_tau: float = 0.9) -> HarnessConfig:
    return HarnessConfig(name="hf", tool_select=DPConfig(mode=Mode.CASCADE, tau=0.5),
                         tool_params=DPConfig(mode=Mode.CASCADE, tau=tp_tau, extra={"batch_s1": True} if batch else {}))


def _episode(cfg: HarnessConfig, plan: str, conf: dict[str, float], batched_backend: bool = True):
    s1 = FixedS1(conf, batched=batched_backend)
    llm = RecordingLLM()
    t = JevAgent(llm, cfg, s1=s1).run(TinyEnv(plan)).to_dict()
    assert t["error"] is None, t["error"]
    return t, s1, llm


def _decisions(t: dict[str, Any]) -> list[tuple]:
    return [(s["step"], d["dp"], d["question"], d["s1_value"], round(d["s1_conf"] or 0, 12), d["deferred"],
             d["final_value"], d["s2_value"]) for s in t["steps"] for d in s["decisions"]]


@pytest.mark.parametrize("plan,conf", [
    ("cancel", {}),                                                     # both arguments accepted: no LLM call
    ("cancel", {"tool_params:reason": 0.5}),                            # one deferred: one forced call
    ("priority", {"tool_params:level": 0.95, "tool_params:urgent": 0.2}),   # optional argument deferred, rule fill
    ("lookup_cancel", {"tool_params:order_id": 0.3}),
])
def test_batched_argument_questions_are_one_request_with_identical_decisions(plan, conf):
    a, s1a, llma = _episode(_cfg(False), plan, conf)
    b, s1b, llmb = _episode(_cfg(True), plan, conf)
    assert _decisions(a) == _decisions(b)
    assert [s["tool"] for s in a["steps"]] == [s["tool"] for s in b["steps"]]
    assert [s["args"] for s in a["steps"]] == [s["args"] for s in b["steps"]]
    assert llma.requests == llmb.requests and a["result"] == b["result"]
    # the per-question path asks one request per argument question; the batched path one per step
    qa = [q for r in s1a.requests for q in r]
    qb = [q for r in s1b.requests for q in r]
    assert qa == qb
    multi = [r for r in s1b.requests if len(r) > 1]
    for r in s1b.requests:
        assert len(r) == 1 or all("argument" in q for q in r)
    n_arg_steps = sum(1 for s in b["steps"] if sum(d["dp"] == "tool_params" and d["s1_value"] is not None
                                                     for d in s["decisions"]) > 1)
    assert len(multi) == n_arg_steps and len(s1b.requests) == len(s1a.requests) - sum(len(r) - 1 for r in multi)
    # accounting: one ledger record per request; the members' cost shares sum to the ledger's System-1 cost
    s1_calls = [c for c in b["calls"] if c["component"] == "s1"]
    assert len(s1_calls) == len(s1b.requests)
    assert sum(c["meta"]["n_questions"] for c in s1_calls) == len(qb)
    dsum = sum(d["s1_cost_usd"] or 0 for s in b["steps"] for d in s["decisions"])
    assert dsum == pytest.approx(b["usage"]["s1"]["cost_usd"] if "s1" in b["usage"] else 0.0)
    batched = [d for s in b["steps"] for d in s["decisions"] if (d.get("gold_aux") or {}).get("s1_batch")]
    assert all(d["gold_aux"]["s1_batch"]["size"] > 1 for d in batched)


def test_batch_s1_needs_a_batching_backend_and_is_off_by_default():
    a, s1a, _ = _episode(_cfg(True), "cancel", {}, batched_backend=False)
    assert all(len(r) == 1 for r in s1a.requests)
    b, s1b, _ = _episode(_cfg(False), "cancel", {})
    assert all(len(r) == 1 for r in s1b.requests)
    assert _decisions(a) == _decisions(b)


# ------------------------------------------------------------------ the profile
def test_final_profile_is_the_cascade_at_the_frozen_thresholds_without_reflection():
    prof = load_profile("configs/harness/jevagent_final.yaml")
    best = load_profile("configs/harness/jevagent_best.yaml")
    assert prof["status"] == "development" and prof["name"] == "jevagent_final"
    dps, harness, skip = decision_points(prof, "gemini-3.5-flash", backend="jevagent_rl")
    assert skip is None and harness == {}
    assert set(dps) == {"tool_select", "tool_params"}
    rl = best["system2"]["gemini-3.5-flash"]["backends"]["jevagent_rl"]
    assert dps["tool_select"] == {"mode": "cascade", "tau": rl["tool_select"]["tau"]}
    assert dps["tool_params"] == {"mode": "cascade", "tau": rl["tool_params"]["tau"], "batch_s1": True}
    cfg = HarnessConfig(name="hf", **dps).validate_modes()
    assert cfg.tool_params.extra == {"batch_s1": True} and cfg.reflection.mode == Mode.OFF
    # the selected profile's tool group, unchanged
    sel = best["system2"]["gemini-3.5-flash"]["decision_points"]
    assert {k: v["mode"] for k, v in dps.items()} == {k: sel[k]["mode"] for k in ("tool_select", "tool_params")}
    assert decision_points(prof, "gemini-3.5-flash-lite")[2]


# ------------------------------------------------------------------ serving (tiny random model, CPU, fp32)
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_dense(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(ROOT / "scripts/train"))
    from make_tiny_model import make_tiny_model

    return make_tiny_model(tmp_path_factory.mktemp("tiny_dense_hf"))


def _engine(path: Path, fast: bool = False, **kw):  # noqa: ANN202
    from jevharness.training.serve_jev import Engine

    if fast:
        kw.update(fuse_multi=True, suffix_pad_slack=0.25, readout_gather=True, preencode=True)
    return Engine(str(path), name=path.name, max_len=4096, session_cache=True, **kw)


def _states(n: int) -> list[str]:
    from test_serving import jevgym_states

    return jevgym_states(n)


def _mixed_questions() -> dict[str, dict]:
    tools = {f"tool_{i:02d}": f"Tool number {i}: does thing {i} to an order" for i in range(68)}
    tools["finish"] = "The task is complete"
    return {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?", "criteria": tools},
            "arg": {"type": "choice", "instructions": "Which value should the argument `order_id` take? ()",
                    "criteria": {"ORD-1": None, "ORD-2": None, "(other)": None}},
            "lvl": {"type": "choice", "instructions": "Which value should the argument `level` take? ()",
                    "criteria": {str(i): None for i in range(1, 6)}},
            "done": {"type": "noul", "instructions": "Is the task complete?"}}


def _close(a: dict[str, Any], b: dict[str, Any]) -> float:
    worst = 0.0
    for k in a:
        if a[k]["type"] == "noul":
            worst = max(worst, abs(a[k]["noul"] - b[k]["noul"]))
        else:
            assert a[k]["choice"] == b[k]["choice"], k
            for o, p in a[k]["probabilities"].items():
                worst = max(worst, abs(p - b[k]["probabilities"][o]))
    assert worst <= TOL, worst
    return worst


def test_fast_server_paths_match_the_stage2_server_and_stateless(tiny_dense: Path, monkeypatch: pytest.MonkeyPatch):
    old, new = _engine(tiny_dense), _engine(tiny_dense, fast=True)
    positions: dict[str, list[int]] = {"old": [], "new": []}   # token positions run through the model per forward

    def hook(tag: str):  # noqa: ANN202
        def f(mod, args, kwargs):  # noqa: ANN001, ANN202
            x = kwargs.get("input_ids", args[0] if args else None)
            positions[tag].append(int(x.shape[0] * x.shape[1]))
        return f

    for tag, eng in (("old", old), ("new", new)):
        eng.model.register_forward_pre_hook(hook(tag), with_kwargs=True)
    qs = _mixed_questions()
    for s in _states(4):
        for sub in (qs, {k: qs[k] for k in ("arg", "lvl")}, {"tool": qs["tool"]}):
            a, ua = old.answer_all(s, sub)
            b, ub = new.answer_all(s, sub)
            _close(a, b)
            assert ub["lock_wait_ms"] >= 0 and ub["compute_ms"] > 0
    # fewer positions (no padding of a 17-option chunk to a 52-option one); the fused multi-question pass saves the
    # separate prefix pass where the state is new (the length grouping may add a pass where it removes padding)
    assert sum(positions["new"]) < 0.9 * sum(positions["old"]), (sum(positions["new"]), sum(positions["old"]))
    d = new.describe()
    assert d["fuse_multi"] and d["readout_gather"] and d["suffix_pad_slack"] == 0.25 and d["preencode"]
    assert not new._enc_memo   # every pre-encoded prompt was consumed
    monkeypatch.setenv("JEV_SERVE_NO_CACHE", "1")
    s = _states(4)[-1]
    _close(new.answer_all(s, qs)[0], old.answer_all(s, qs)[0])


def test_suffix_groups_split_unequal_lengths_only_with_slack(tiny_dense: Path):
    old, new = _engine(tiny_dense), _engine(tiny_dense, fast=True)
    lens = [850, 300, 40, 38, 36]
    assert old._suffix_groups([0, 1, 2, 3, 4], lens, 16) == [[0, 1, 2, 3, 4]]
    g = new._suffix_groups([0, 1, 2, 3, 4], lens, 16)
    assert sorted(j for x in g for j in x) == [0, 1, 2, 3, 4]
    for grp in g:   # every batch keeps its padding within the slack
        assert len(grp) * max(lens[j] for j in grp) <= 1.25 * sum(lens[j] for j in grp)
    assert [0] in g and [2, 3, 4] in g


def test_readout_gather_equals_the_window_path(tiny_dense: Path):
    from jevharness.training.decision_model import option_logits_cached, prefill

    eng = _engine(tiny_dense)
    ids = eng.collator.encode("STATE: a shared prefix of some length " * 6)
    cache, _ = prefill(eng.model, ids, None)
    sufs = [list(range(5, 45)), list(range(7, 12)), list(range(9, 30))]
    codes = torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1], [6, 7, 8, 9]])
    pad = eng.tok.pad_token_id
    a = option_logits_cached(eng.model, cache, sufs, codes, pad)
    b = option_logits_cached(eng.model, cache, sufs, codes, pad, gather=True)
    fin = torch.isfinite(a)
    assert torch.equal(fin, torch.isfinite(b)) and torch.allclose(a[fin], b[fin], atol=TOL)


def test_replayed_development_requests_old_and_new_call_graph_agree(tiny_dense: Path, tmp_path: Path):
    """Stored Stage-2 development requests (the dev tariff run's first episodes): the OLD call graph (reflection, tool,
    one request per argument) on the Stage-2 server and the NEW one (tool; the step's argument questions in one
    request) on the fast server answer every shared question identically (fp32, tiny model)."""
    hf = _hf()
    wl = ROOT / "tests/fixtures/harness_final_replay.jsonl.gz"
    with gzip.open(wl, "rt") as f:
        eps = [json.loads(line) for line in f][:2]
    old, new = _engine(tiny_dense), _engine(tiny_dense, fast=True)
    n = 0
    for e in eps:
        for s in e["steps"][:4]:
            got_old: dict[tuple[str, str], dict[str, float]] = {}
            for req in s["old"]:
                ans, _ = old.answer_all(req["state"], req["questions"])
                q = req["questions"]["q0"]
                got_old[(req["dp"], json.dumps(q, sort_keys=True))] = hf._dist(ans["q0"], q)
            for req in s["new"]:
                ans, _ = new.answer_all(req["state"], req["questions"])
                for qid, q in req["questions"].items():
                    ref = got_old[(req["dp"], json.dumps(q, sort_keys=True))]
                    c = hf.compare(req["dp"], hf._dist(ans[qid], q), ref)
                    assert c["same_decision"] and c["max_abs"] <= TOL, c
                    n += 1
            # the new graph drops exactly the reflection requests and merges the argument requests
            assert sum(len(r["questions"]) for r in s["new"]) == sum(1 for r in s["old"] if r["dp"] != "reflection")
    assert n > 5


def test_jev_client_logs_the_servers_cache_and_timing(tiny_dense: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from jevharness.accounting import Ledger
    from jevharness.decision import Choice, Noul
    from jevharness.decision.jev import JevBackend
    from jevharness.training.serve_jev import build_app

    client = TestClient(build_app(_engine(tiny_dense, fast=True)))
    b = JevBackend(model=tiny_dense.name, base_url="http://testserver", api_key="")
    b._http = client
    led = Ledger()
    for s in _states(2):
        b.decide(s, [Choice("Q?", ["a", "b", "c"]), Noul("ok?")], ledger=led)
    srv = [r.meta["server"] for r in led.records]
    assert all({"prefilled_tokens", "cache_hit_tokens", "lock_wait_ms", "compute_ms", "ms"} <= set(x) for x in srv)
    assert srv[1]["cache_hit_tokens"] > 0 and srv[0]["cache_hit_tokens"] == 0


# ------------------------------------------------------------------ the analysis script
def test_decision_rule_and_compare():
    hf = _hf()
    assert hf._decision("tool_params", {"A": 1.0, "B": 0.0}) == ("A", True)
    assert hf._decision("tool_params", {"A": 0.9999998, "B": 2e-7})[1] is False
    assert hf._decision("tool_params", {"(other)": 1.0}) == ("(other)", False)
    assert hf._decision("tool_select", {"x": 0.6, "y": 0.4}) == ("x", True)
    c = hf.compare("tool_select", {"x": 0.6, "y": 0.4}, {"x": 0.4, "y": 0.6})
    assert not c["same_argmax"] and c["max_abs"] == pytest.approx(0.2)


def _rec(task: str, arm: str, success: bool, s2: list[float], s1: list[float], wall: float, cost: float) -> dict:
    calls = [{"component": "s2", "latency_ms": x, "cost_usd": cost / max(1, len(s2)), "step": i, "decision_point":
              "tool_select", "ts": f"2026-10-05T18:00:{i:02d}.000+00:00", "meta": {}, "input_tokens": 10,
              "output_tokens": 2} for i, x in enumerate(s2)]
    calls += [{"component": "s1", "latency_ms": x, "cost_usd": 0.0, "step": i, "decision_point": "tool_select",
               "ts": f"2026-10-05T18:01:{i:02d}.000+00:00", "meta": {"n_questions": 1, "server": {
                   "lock_wait_ms": 1.0, "compute_ms": x - 2, "cache_hit_tokens": 50.0, "prefilled_tokens": 20.0,
                   "stateless_tokens": 70.0}}, "input_tokens": 10, "output_tokens": 0} for i, x in enumerate(s1)]
    return {"task_id": task, "config": arm, "steps": [{"step": i, "decisions": []} for i in range(max(len(s1), len(s2)))],
            "result": {"success": success}, "wall_ms": wall, "error": None, "calls": calls,
            "usage": {"total": {"cost_usd": cost, "latency_ms": sum(s1) + sum(s2)}}, "meta": {}}


def test_live_analysis_on_a_synthetic_run(tmp_path: Path):
    hf = _hf()
    run = tmp_path / "harness_final_live"
    run.mkdir()
    lo, hfa = [], []
    for i in range(12):
        t = f"jevgym-x-{300500 + i}-d60-near-strict"
        lo.append(_rec(t, "llm_only", True, [900.0] * 7, [], 6400.0, 0.04))
        hfa.append(_rec(t, "harness_final", i != 3, [800.0] * 3, [120.0] * 8, 4200.0, 0.006))
    (run / "llm_only.jsonl").write_text("\n".join(json.dumps(r) for r in lo) + "\n")
    (run / "harness_final.jsonl").write_text("\n".join(json.dumps(r) for r in hfa) + "\n")
    (run / "schedule.json").write_text(json.dumps({"concurrency": 4, "invocations": [
        {"k": 0, "started": "2026-10-05T18:00:00+00:00", "finished": "2026-10-05T18:10:00+00:00", "concurrency": 4}]}))
    tariff = tmp_path / "tariff.json"
    tariff.write_text(json.dumps({"q_usd_per_episode": 0.001, "host_usd_per_hour": 2.24208}))
    res = hf.analyze(run, tmp_path / "out", tmp_path / "hf.tex", tariff=tariff, n_expected=12, n_boot=200)
    s = res["paired"]
    assert s["n_pairs"] == 12 and s["success_delta_pt"] == pytest.approx(-100 / 12)
    assert s["wall_s"]["delta"] == pytest.approx(-2.2) and s["cost"]["saving_frac"] == pytest.approx(1 - 0.007 / 0.04)
    tex = (tmp_path / "hf.tex").read_text()
    assert r"\defres{hf/wall_s/llm_only}" in tex and r"\defres{hf/cost/saving}" in tex
    assert (tmp_path / "out" / "summary.md").exists() and (tmp_path / "out" / "summary.json").exists()
    # the dedicated-GPU row: its own keys (hf_ded/...) and both rows side by side
    hf.analyze(run, tmp_path / "out", tmp_path / "hf.tex", tariff=tariff, n_expected=12, n_boot=200,
               name="summary_dedicated")
    tex = (tmp_path / "hf.tex").read_text()
    assert r"\defres{hf_ded/wall_s/delta}{-2.20}" in tex and r"\defres{hf/wall_s/delta}{-2.20}" in tex
    rows = (tmp_path / "out" / "summary_rows.md").read_text()
    assert "shared GPU" in rows and "dedicated GPU" in rows


def test_replicas_route_every_request_of_an_episode_to_one_server():
    import httpx

    from jevharness.decision import Choice
    from jevharness.decision.jev import JevBackend, replica_index

    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append((req.url.host, body["state"].split("\n\n", 1)[0]))
        return httpx.Response(200, json={"model": "m", "answers": {"q0": {"type": "choice", "choice": "a",
                                                                         "probabilities": {"a": 0.9, "b": 0.1}}},
                                         "usage": {"input_tokens": 3}})

    b = JevBackend(model="m", base_url="http://r0", api_key="", transport=httpx.MockTransport(handler),
                   replicas=["http://r0", "http://r1"])
    tasks = [f"TASK: job {i}" for i in range(12)]
    for t in tasks:
        for step in range(3):
            b.decide(f"{t}\n\nHISTORY:\n[{step}] x\n\nCURRENT OBSERVATION:\nobs {step}", [Choice("Q?", ["a", "b"])])
    hosts = {}
    for host, task in seen:
        hosts.setdefault(task, set()).add(host)
    assert all(len(h) == 1 for h in hosts.values())                        # sticky per episode
    assert {h for v in hosts.values() for h in v} == {"r0", "r1"}           # both replicas used
    assert all(next(iter(hosts[t])) == f"r{replica_index(t + chr(10) * 2, 2)}" for t in tasks)
    assert b.describe()["replicas"] == ["http://r0", "http://r1"] and "replicas" not in JevBackend(
        model="m", base_url="http://x", api_key="").describe()


def test_replay_decisions_use_the_harness_outcome_not_the_smoothed_distribution():
    hf = _hf()
    # the server's confidence is exactly 1.0 -> accepted at tau = 1 - 1.2e-7, as the harness decided in the log
    assert hf._replay_decision("tool_params", {"U1": 1.0, "U2": 0.0}) == ("U1", True)
    assert hf._replay_decision("reflection", {"yes": 0.3, "no": 0.7}) == (False, False)
    assert hf._replay_decision("tool_select", {"get_orders": 0.97, "finish": 0.03}) == ("get_orders", True)
