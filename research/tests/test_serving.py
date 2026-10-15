"""Incremental decision serving (``training/serve_jev.py``): prefix reuse inside a request, the episode session
cache across requests, and the bench's smoke mode. Offline, CPU, fp32, tiny random models: a 1-layer dense Qwen3
(``scripts/train/make_tiny_model.py``) and a 4-layer hybrid Qwen3-Next (Gated DeltaNet linear layers + full
attention: its cache cannot be cropped, like Qwen3.5). Outputs mean nothing; only equality with the stateless
path and the bookkeeping are checked."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-4


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def tiny_dense(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(ROOT / "scripts/train"))
    from make_tiny_model import make_tiny_model

    return make_tiny_model(tmp_path_factory.mktemp("tiny_dense"))


def _hybrid_config(cls: Any, tok: Any) -> Any:
    return cls(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
               num_key_value_heads=1, head_dim=16, linear_num_value_heads=2, linear_num_key_heads=1,
               linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, tie_word_embeddings=True,
               layer_types=["linear_attention", "full_attention"] * 2, eos_token_id=tok.eos_token_id,
               pad_token_id=tok.pad_token_id)


def _make_hybrid(tmp_path_factory: pytest.TempPathFactory, tiny_dense: Path, family: str) -> Path:
    """Same tokenizer, hybrid Gated-DeltaNet / full-attention body (non-croppable cache): the Qwen3.5 text model
    (``Qwen3_5ForCausalLM``, what ``load_causal_lm`` serves for Qwen/Qwen3.5-* and Intern-Decision) or Qwen3-Next."""
    import transformers
    from transformers import AutoTokenizer

    try:
        if family == "qwen3_5":
            cfg_cls, model_cls = transformers.Qwen3_5TextConfig, transformers.Qwen3_5ForCausalLM
        else:
            cfg_cls, model_cls = transformers.Qwen3NextConfig, transformers.Qwen3NextForCausalLM
    except AttributeError:  # pragma: no cover
        pytest.skip(f"transformers {transformers.__version__} without {family}")
    tok = AutoTokenizer.from_pretrained(str(tiny_dense))
    cfg = _hybrid_config(cfg_cls, tok)
    if family != "qwen3_5":
        cfg.num_experts, cfg.decoder_sparse_step = 0, 0
    torch.manual_seed(1)
    d = tmp_path_factory.mktemp(f"tiny_{family}")
    try:
        model_cls(cfg).save_pretrained(d)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot build a tiny {family}: {e}")
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny_hybrid(tmp_path_factory: pytest.TempPathFactory, tiny_dense: Path) -> Path:
    return _make_hybrid(tmp_path_factory, tiny_dense, "qwen3_next")


@pytest.fixture(scope="module")
def tiny_qwen35(tmp_path_factory: pytest.TempPathFactory, tiny_dense: Path) -> Path:
    return _make_hybrid(tmp_path_factory, tiny_dense, "qwen3_5")


@pytest.fixture(params=["dense", "qwen3_5", "qwen3_next"])
def tiny(request: pytest.FixtureRequest, tiny_dense: Path, tiny_qwen35: Path, tiny_hybrid: Path) -> Path:
    return {"dense": tiny_dense, "qwen3_5": tiny_qwen35, "qwen3_next": tiny_hybrid}[request.param]


def _engine(path: Path, **kw):  # noqa: ANN202
    from jevharness.training.serve_jev import Engine

    kw.setdefault("session_cache", True)
    return Engine(str(path), name=path.name, max_len=4096, **kw)


def jevgym_states(n_steps: int, lines_per_step: int = 12, seed: int = 0) -> list[str]:
    """Growing JevGym-like states: TASK, HISTORY (tool calls + heartbeat noise), CURRENT OBSERVATION. As in the real
    logs, the observation of step t becomes a history line at t+1, so consecutive states share everything before
    ``CURRENT OBSERVATION`` and differ after it."""
    tools = [("find_user", '{"name": "Jia Wang"}', '{"user_id": "U5563", "tier": "basic"}'),
             ("get_orders", '{"user_id": "U5563"}', '[{"order_id": "ORD-3443", "amount": 458.24}]'),
             ("ask_user", '{"missing": "refund_method"}', "Customer: original payment method please."),
             ("refund", '{"order_id": "ORD-3443", "amount": 458.24}', "Refunded 458.24 via original_payment."),
             ("get_order_status", '{"order_id": "ORD-3443"}', '{"status": "refunded"}')]
    task = "TASK: Refund the full amount of Jia Wang's most recent delivered order (highest order id)."
    out, hist = [], []
    for t in range(n_steps):
        name, args, obs = tools[t % len(tools)]
        cur = f"\n\nCURRENT OBSERVATION:\n{obs}" if t else "\n\nCURRENT OBSERVATION:\nYou have access to the tools."
        body = task + (("\n\nHISTORY:\n" + "\n".join(hist)) if hist else "") + cur
        out.append(body)
        noise = "\n".join(f"log: heartbeat ok #{(seed * 7919 + t * 101 + i * 37) % 100000}"
                          for i in range(lines_per_step))
        hist.append(f"[{t}] {name}({args}) -> {obs}\n{noise}")
    return out


def wire_questions(n: int = 3) -> dict[str, dict]:
    qs = {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
                   "criteria": {"find_user": "Look a user up by name", "get_orders": "List a user's orders",
                                "refund": "Refund an order", "finish": None}},
          "done": {"type": "noul", "instructions": "Is the task complete?"},
          "risk": {"type": "score", "instructions": "How risky is the next action?",
                   "criteria": ["none", "low", "medium", "high", "irreversible"]}}
    return dict(list(qs.items())[:n])


def _items(state: str, qs: dict[str, dict]) -> list[tuple[str, str, list[str], str]]:
    from jevharness.training.serve_jev import Engine

    return [(state, instr, opts, kind) for kind, instr, opts, _ in (Engine._plan(q) for q in qs.values())]


def _close(a: list[list[float]], b: list[list[float]], tol: float = TOL) -> float:
    worst = 0.0
    for x, y in zip(a, b):
        assert len(x) == len(y)
        for u, v in zip(x, y):
            if u == float("-inf") and v == float("-inf"):
                continue
            worst = max(worst, abs(u - v))
    assert worst <= tol, worst
    return worst


# ------------------------------------------------------------------ identical outputs, cache vs no cache
def test_three_questions_one_state_match_stateless(tiny: Path, monkeypatch: pytest.MonkeyPatch):
    eng = _engine(tiny, session_cache=False)  # within-request prefix sharing only
    state = jevgym_states(3)[-1]
    items = _items(state, wire_questions(3))
    z_shared = eng._option_logits(items)
    st = [dict(s) for s in eng._item_stats]
    monkeypatch.setenv("JEV_SERVE_NO_CACHE", "1")
    assert eng.describe()["stateless"]
    z_plain = eng._option_logits(items)
    _close(z_shared, z_plain)
    # the shared path prefilled the state once: strictly fewer tokens than three full prompts
    assert sum(s["prefilled"] for s in st) < sum(s["stateless"] for s in st)
    assert sum(s["stateless"] for s in st) == sum(s["stateless"] for s in eng._item_stats)
    monkeypatch.delenv("JEV_SERVE_NO_CACHE")
    a, u = eng.answer_all(state, wire_questions(3))
    monkeypatch.setenv("JEV_SERVE_NO_CACHE", "1")
    b, u0 = eng.answer_all(state, wire_questions(3))
    assert a["tool"]["choice"] == b["tool"]["choice"] and abs(a["done"]["noul"] - b["done"]["noul"]) < 1e-5
    assert abs(a["risk"]["score"] - b["risk"]["score"]) < 1e-5
    assert u["prefilled_tokens"] < u["stateless_tokens"] == u0["stateless_tokens"] == u0["prefilled_tokens"]
    assert u["cache_hit_tokens"] == 0 and not u["cache_hit"]


def test_growing_states_match_stateless_and_reuse_the_episode_prefix(tiny: Path, monkeypatch: pytest.MonkeyPatch):
    eng = _engine(tiny)
    assert eng.cache_info["croppable"] == (tiny.name.startswith("tiny_dense"))
    states = jevgym_states(4)
    qs = wire_questions(3)
    cached, usage = [], []
    for s in states:
        cached.append(eng._option_logits(_items(s, qs)))
        _, u = eng.answer_all(s, qs)  # same state again: exact hit on the stored common prefix
        usage.append(u)
    monkeypatch.setenv("JEV_SERVE_NO_CACHE", "1")
    for s, z in zip(states, cached):
        _close(z, eng._option_logits(_items(s, qs)))
    monkeypatch.delenv("JEV_SERVE_NO_CACHE")
    # bookkeeping: every repeat of a state is a full hit; the episode prefix is reused from the second step on
    for u in usage:
        assert u["cache_hit"] and u["prefilled_tokens"] < u["stateless_tokens"]
    c = eng.stats
    assert c["prefilled_tokens"] < c["stateless_tokens"] and c["cache_hits"] >= len(states) + 2
    # a fresh engine replays the episode once. The most a step can reuse is what the previous state contained:
    # everything before the previous state's CURRENT OBSERVATION block (that block becomes a history line).
    # dense: the stored prefix is cropped to that point from step 2 on; hybrid (no crop): step 2 hits the
    # snapshot taken where step 1 diverged from step 0, step 3 hits the snapshot taken at the predicted
    # divergence point ("\n\nCURRENT OBSERVATION" marker) of step 2 -> full reuse with a one-step lag
    eng2 = _engine(tiny)
    hits = [eng2.answer_all(s, qs)[1]["cache_hit_tokens"] for s in states]
    shared = [len(eng2.collator.encode(s[: s.rindex("\n\nCURRENT OBSERVATION")])) for s in states]
    assert hits[0] == 0 and all(h > 0 for h in hits[2:]), hits
    assert hits[3] >= shared[2] - 8, (hits, shared)
    if eng2.cache_info["croppable"]:
        assert hits[1] > 0 and hits[2] >= shared[1] - 8, (hits, shared)
    else:
        assert hits[1] == 0 and hits[2] < shared[1], (hits, shared)
    assert eng2.stats["prefilled_tokens"] < eng2.stats["stateless_tokens"]


def test_single_question_requests_still_share_the_prefix(tiny_dense: Path):
    """k = 1: the common prefix is the whole prompt; the read-out token is kept out of the cached prefix."""
    eng = _engine(tiny_dense)
    states = jevgym_states(3)
    for s in states:
        a1, u1 = eng.answer_all(s, wire_questions(1))
        a2, u2 = eng.answer_all(s, wire_questions(1))
        assert u2["cache_hit_tokens"] >= u1["stateless_tokens"] - 2 and u2["prefilled_tokens"] >= 1
        assert a1["tool"]["probabilities"] == pytest.approx(a2["tool"]["probabilities"], abs=1e-5)
    assert eng.answer(states[-1], wire_questions(1)["tool"])["probabilities"] == pytest.approx(
        a2["tool"]["probabilities"], abs=1e-5)


def test_knockout_over_52_options_matches_single_path_and_stateless(tiny_dense: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    eng = _engine(tiny_dense)
    state = jevgym_states(2)[-1]
    q = {"type": "choice", "instructions": "Which tool should the agent call next?",
         "criteria": {f"tool_{i:02d}": f"Tool number {i}" for i in range(60)}}
    answers, usage = eng.answer_all(state, {"t": q, "done": wire_questions(2)["done"]})
    a = answers["t"]
    b = eng.answer(state, q)
    assert a["probabilities"] == pytest.approx(b["probabilities"], abs=1e-5)
    # review P1-4: 60 options = chunks 52 + 8 (2 reads) -> 4 finalists -> final read; 2 rounds, several passes
    ko = a["knockout"]
    assert ko["rounds"] == 2 and ko["reads"] == 3 and len(ko["finalists"]) == 4 and ko["full_set"] == "linked"
    assert usage["n_rounds"] == 2 and usage["n_passes"] >= 2 and "knockout" not in answers["done"]
    # linked full-set distribution: every option keeps mass, the argmax is a finalist, sums to one
    assert all(v > 0 for v in a["probabilities"].values()) and a["choice"] in ko["finalists"]
    assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    monkeypatch.setenv("JEV_SERVE_NO_CACHE", "1")
    c = eng.answer(state, q)
    assert a["probabilities"] == pytest.approx(c["probabilities"], abs=1e-5)
    # the legacy composition (before 2026-10-02): the final read, eliminated options keep 0, same argmax
    eng.knockout_probs = "finalists"
    d = eng.answer(state, q)
    assert sum(v > 0 for v in d["probabilities"].values()) <= 4 and d["choice"] == a["choice"]
    assert d["confidence"] >= a["confidence"]  # conditional on the finalists: never below the full-set probability


# ------------------------------------------------------------------ session cache bookkeeping
def test_lru_eviction_bounds_tokens_and_entries(tiny_dense: Path):
    eng = _engine(tiny_dense, cache_tokens=1500, cache_entries=3)
    states = [f"TASK: job {i}.\n\nHISTORY:\n" + "\n".join(f"log: heartbeat ok #{i * 1000 + j}" for j in range(40))
              + f"\n\nCURRENT OBSERVATION:\nstep {i}" for i in range(5)]
    for s in states:
        eng.answer_all(s, wire_questions(2))
        ses = eng.session
        assert len(ses) <= 3 and ses.tokens <= 1500
    assert eng.session.evictions > 0 and eng.stats["evictions"] == eng.session.evictions
    hit_new = eng.answer_all(states[-1], wire_questions(2))[1]["cache_hit_tokens"]  # newest still cached
    hit_old = eng.answer_all(states[0], wire_questions(2))[1]["cache_hit_tokens"]  # oldest evicted ...
    # ... except for the prefix every prompt shares (chat template + system prompt + "TASK: job "), which a
    # croppable cache serves from any entry
    shared = len(eng.collator.encode("")) + 8
    assert hit_new > 4 * shared and hit_old <= shared and hit_old < hit_new / 4, (hit_new, hit_old, shared)
    d = eng.describe()
    assert d["session"]["max_tokens"] == 1500 and d["session"]["entries"] <= 3 and d["counters"]["requests"] == 7
    # an entry larger than the budget is never stored
    big = _engine(tiny_dense, cache_tokens=50)
    big.answer_all(states[0], wire_questions(2))
    assert len(big.session) == 0


def test_session_cache_can_be_switched_off(tiny_dense: Path):
    from jevharness.training.serve_jev import Engine

    eng = _engine(tiny_dense, session_cache=False)
    s = jevgym_states(2)[-1]
    for _ in range(2):
        _, u = eng.answer_all(s, wire_questions(3))
        assert not u["cache_hit"] and u["prefilled_tokens"] < u["stateless_tokens"]
    assert eng.session is None and not eng.describe()["session_cache"]
    # the constructor default (in-process batch scorers) keeps the session cache off
    assert Engine(str(tiny_dense), name="d").session is None


# ------------------------------------------------------------------ wire path
def test_wire_server_answers_all_questions_in_one_call_and_reports_usage(tiny_dense: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from jevharness.decision import Choice, Noul
    from jevharness.decision.jev import JevBackend
    from jevharness.training.serve_jev import build_app

    eng = _engine(tiny_dense)
    client = TestClient(build_app(eng))
    states = jevgym_states(3)
    r1 = client.post("/v1/systemone", json={"state": states[0], "questions": wire_questions(3)}).json()
    r2 = client.post("/v1/systemone", json={"state": states[1], "questions": wire_questions(3)}).json()
    assert set(r1["answers"]) == {"tool", "done", "risk"}
    assert not r1["usage"]["cache_hit"] and r2["usage"]["cache_hit"] and r2["usage"]["cache_hit_tokens"] > 0
    assert r2["usage"]["prefilled_tokens"] < r2["usage"]["stateless_tokens"]
    assert r1["usage"]["input_tokens"] == len(eng.tok(states[0], add_special_tokens=False).input_ids)
    assert eng.stats["requests"] == 2 and eng.stats["questions"] == 6
    m = client.get("/v1/models").json()["data"][0]["backend"]
    assert m["session_cache"] and m["counters"]["requests"] == 2 and "croppable" in m
    assert client.get("/health").json()["requests"] == 2
    # the harness client is unaffected by the extra usage keys
    b = JevBackend(model=tiny_dense.name, base_url="http://testserver", api_key="")
    b._http = client
    r = b.decide(states[2], [Choice("Q?", ["a", "b", "c"]), Noul("ok?")])
    assert r.answers[0].value in {"a", "b", "c"} and 0 <= r.answers[1].p_yes <= 1 and r.usage.input_tokens > 0


def test_answer_batch_scores_several_states_at_once(tiny_dense: Path, monkeypatch: pytest.MonkeyPatch):
    eng = _engine(tiny_dense)
    reqs = [(s, wire_questions(3)) for s in jevgym_states(3)]
    out = eng.answer_batch(reqs)
    assert len(out) == 3 and eng.stats["requests"] == 3
    singles = [eng.answer_all(s, q)[0] for s, q in reqs]
    for (a, _), b in zip(out, singles):
        assert a["tool"]["probabilities"] == pytest.approx(b["tool"]["probabilities"], abs=1e-5)
    monkeypatch.setenv("JEV_SERVE_NO_CACHE", "1")
    for (a, _), (s, q) in zip(out, reqs):  # stateless batch over distinct states agrees too
        plain = eng.answer_all(s, q)[0]["risk"]["probabilities"]
        assert a["risk"]["probabilities"] == pytest.approx(plain, abs=1e-5)


# ------------------------------------------------------------------ bench
def test_bench_serving_smoke_runs_on_cpu(tmp_path: Path):
    t = time.perf_counter()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/bench_serving.py"), "--smoke", "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr[-2000:]
    assert time.perf_counter() - t < 150  # ~25 s alone on 4 CPU cores; headroom for a loaded machine
    res = json.loads((tmp_path / "bench_serving.json").read_text())
    cells = [c for c in res["cells"] if "error" not in c]
    assert cells and (tmp_path / "bench_serving.md").exists()
    modes = {c["mode"] for c in cells}
    assert modes == {"stateless", "prefix", "session"}
    for c in cells:
        assert c["ms_per_decision"]["p50"] > 0 and c["ms_per_decision"]["p95"] >= c["ms_per_decision"]["p50"]
        assert c["prefilled_tokens_per_decision"] <= c["stateless_tokens_per_decision"] + 1e-9
    sess = [c for c in cells if c["mode"] == "session" and c["questions"] == 3]
    assert all(c["reuse"] > 1.0 for c in sess)


def test_single_question_session_requests_use_one_forward_pass_and_match_stateless(tiny_dense: Path,
                                                                                     monkeypatch: pytest.MonkeyPatch):
    """Dense (croppable) cache + one question: the prefix delta and the suffix read-out are ONE forward pass (GPU
    round 2: the two-pass path made session mode slower than stateless at <= 16k tokens), the stored snapshot is
    cropped back to the prefix so the next step still hits, and the logits equal the stateless and two-pass paths."""
    eng = _engine(tiny_dense)
    eng.profile = True
    calls: list[int] = []
    hook = eng.model.register_forward_hook(lambda *a: calls.append(1))
    states = jevgym_states(3)
    q = wire_questions(1)
    got = []
    for s in states:
        calls.clear()
        got.append(eng._option_logits(_items(s, q)))
        assert len(calls) == 1, calls
    hook.remove()
    assert {"encode", "lookup", "prefill", "store", "suffix"} <= set(eng.phase_ms)
    assert eng.describe()["phase_ms"]["prefill"] > 0
    assert eng.stats["cache_hits"] >= 1 and eng.stats["prefilled_tokens"] < eng.stats["stateless_tokens"]
    monkeypatch.setenv("JEV_SERVE_NO_CACHE", "1")
    for s, z in zip(states, got):
        _close(z, eng._option_logits(_items(s, q)))
    monkeypatch.delenv("JEV_SERVE_NO_CACHE")
    eng.batch_suffixes = False  # the two-pass reference (prefix, then the suffix on a copy)
    eng.session.clear()
    calls.clear()
    hook = eng.model.register_forward_hook(lambda *a: calls.append(1))
    for s, z in zip(states, got):
        _close(z, eng._option_logits(_items(s, q)))
    hook.remove()
    assert len(calls) == 2 * len(states)


def test_bench_serving_markdown_tolerates_cells_without_a_profile(tmp_path):
    """GPU round 3: with --profile the stateless cells carry phase_ms_per_request = None; the table must render."""
    import importlib.util, json
    spec = importlib.util.spec_from_file_location("bench_serving", ROOT / "scripts" / "bench_serving.py")
    bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
    cell = {"model": "m", "source": "synthetic:1000", "state_tokens": 1000, "questions": 1, "batch": 1,
            "ms_per_decision": {"p50": 10.0, "p95": 12.0}, "ms_per_request": {"p50": 10.0, "p95": 12.0},
            "prefilled_tokens_per_decision": 1000, "stateless_tokens_per_decision": 1000, "reuse": 1.0}
    meta = {"device": "cuda", "engines": {}, "torch": "t", "transformers": "x", "args": {"repeats": 3, "steps": 6, "profile": True}}
    res = {"meta": meta, "cells": [{**cell, "mode": "stateless", "phase_ms_per_request": None},
                                                          {**cell, "mode": "prefix", "phase_ms_per_request": {"encode": 1.0}}]}
    (tmp_path / "bench_serving.json").write_text(json.dumps(res))
    bs.main(["--out", str(tmp_path), "--render-only"])
    md = (tmp_path / "bench_serving.md").read_text()
    assert "| stateless |" in md and "encode=1.0" in md and " - |" in md
