"""E25 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09, "E25"): offline tests of the SWE-agent router, the Harbor
agents, the launchers and the gate / readout script. No model, no Docker, no network."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("jev_sweagent_e25", ROOT / "adapters/swe_agent/jev_sweagent.py")
E = _load("explore_e25_harnesses", ROOT / "scripts/explore_e25_harnesses.py")


# ------------------------------------------------------------------ the routed SWE-agent model
def test_question_is_e4s():
    src = (ROOT / "src/jevharness/adapters/mini_swe.py").read_text()
    start = src.index("Q_ROUTING = (")
    ns: dict = {}
    end = src.index("\n", src.index('?")', start))
    exec(src[start:end] + "\n", ns)  # the assignment only
    assert ns["Q_ROUTING"] == R.Q_ROUTING


def _history(n_steps: int, task: str = "Fix the bug in foo()") -> list[dict]:
    h = [{"role": "system", "content": "sys", "message_type": "system_prompt"},
         {"role": "user", "content": f"<pr_description>{task}</pr_description>", "message_type": "observation"}]
    for i in range(n_steps):
        h.append({"role": "assistant", "content": f"thinking {i}", "action": f"cat file{i}.py",
                  "message_type": "action", "tool_calls": [{"id": f"c{i}", "function": {"name": "bash"}}]})
        h.append({"role": "tool", "content": f"OBSERVATION {i}", "message_type": "observation", "tool_call_ids": [f"c{i}"]})
    return h


def test_render_state_e4_layout():
    state, step = R.render_state(_history(0))
    assert step == 0 and state.startswith("TASK (issue to resolve):\n<pr_description>Fix the bug")
    assert "CURRENT OBSERVATION:\n(no command executed yet)" in state and state.endswith("STEP: 0")
    state, step = R.render_state(_history(10))
    assert step == 10
    hist = state.split("HISTORY (most recent last):\n")[1].split("\n\nCURRENT")[0].splitlines()
    assert len(hist) == R.HISTORY_WINDOW and hist[0] == "[3] $ cat file2.py -> OBSERVATION 2"
    assert hist[-1] == "[10] $ cat file9.py -> OBSERVATION 9"
    assert "CURRENT OBSERVATION:\nOBSERVATION 9" in state
    long = R.render_state(_history(1, task="x" * 10_000))[0]
    assert "...[truncated]..." in long and len(long) < 5_000


class _Resp:
    def __init__(self, data: dict):
        self._b = json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def test_wire_client_parses_and_retries(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    sent = []
    calls = {"n": 0}

    def opener(req, timeout):
        calls["n"] += 1
        sent.append(json.loads(req.data))
        if calls["n"] == 1:
            raise R.urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return _Resp({"model": "jev-1.13", "answers": {"hard": {"type": "noul", "noul": 0.25}},
                      "usage": {"input_tokens": 120, "cost": 5e-6}})

    c = R.JevWireClient("http://h:8090/", "jev-1.13", opener=opener)
    p, usage, ms, served = c.ask("STATE")
    assert p == 0.25 and usage["input_tokens"] == 120 and served == "jev-1.13" and calls["n"] == 2
    assert c.url == "http://h:8090/v1/systemone"
    assert sent[0]["questions"] == {"hard": {"type": "noul", "instructions": R.Q_ROUTING}}

    def bad(req, timeout):
        raise R.urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)
    with pytest.raises(RuntimeError):
        R.JevWireClient("http://h", "m", opener=bad).ask("s")


@dataclass
class _Stats:
    instance_cost: float = 0.0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, o):
        return _Stats(**{f.name: getattr(self, f.name) + getattr(o, f.name) for f in fields(self)})

    def __sub__(self, o):
        return _Stats(**{f.name: getattr(self, f.name) - getattr(o, f.name) for f in fields(self)})

    def model_copy(self):
        return _Stats(**vars(self))


class _Model:
    def __init__(self, name: str, cost: float):
        self.config = type("C", (), {"name": name})()
        self.stats = _Stats()
        self.cost = cost
        self.seen = 0

    def query(self, history, **kw):
        self.seen += 1
        self.stats = self.stats + _Stats(self.cost, 1000, 10, 1)
        return {"message": "ok", "tool_calls": []}

    def reset_stats(self):
        self.stats = _Stats()


class _Client:
    wire_model = "jev-1.13"

    def __init__(self, ps):
        self.ps = list(ps)

    def ask(self, state):
        p = self.ps.pop(0)
        if p is None:
            raise RuntimeError("down")
        return p, {"input_tokens": 100, "cost": 4.2e-6}, 12.0, "jev-1.13"


def _routed(tmp_path, ps, *, cost_limit=0.0, call_limit=0, mode="trigger"):
    cfg = type("Cfg", (), {"per_instance_cost_limit": cost_limit, "per_instance_call_limit": call_limit,
                           "total_cost_limit": 0.0, "name": "gemini/gemini-3.5-flash"})()
    strong, cheap = _Model("gemini/gemini-3.5-flash", 0.01), _Model("gemini/gemini-3.5-flash-lite", 0.002)
    m = R.RoutedModel(cfg, None, strong, cheap, _Client(ps), tau=0.5, mode=mode, stats_factory=_Stats,
                      decisions=R.JsonlLog(str(tmp_path / "jev-decisions.jsonl")),
                      calls=R.JsonlLog(str(tmp_path / "jev-calls.jsonl")))
    return m, strong, cheap


def test_routed_model_routes_fails_open_and_aggregates(tmp_path):
    m, strong, cheap = _routed(tmp_path, [0.9, 0.1, None, 0.5])
    for _ in range(4):
        m.query(_history(2))
    assert (strong.seen, cheap.seen) == (3, 1)        # 0.9 strong, 0.1 cheap, error -> strong, 0.5 >= tau -> strong
    assert m.stats.api_calls == 4 and abs(m.stats.instance_cost - 0.032) < 1e-9
    dec = [json.loads(x) for x in (tmp_path / "jev-decisions.jsonl").read_text().splitlines()]
    assert [d["final_value"] for d in dec] == ["strong", "cheap", "strong", "strong"]
    assert dec[2]["error"] and dec[2]["model"] == "jev-1.13" and dec[1]["s1_dist"] == {"yes": 0.1, "no": 0.9}
    assert all(d["dp"] == "routing" and d["question"] == R.Q_ROUTING for d in dec)
    calls = [json.loads(x) for x in (tmp_path / "jev-calls.jsonl").read_text().splitlines()]
    assert [c["role"] for c in calls] == ["strong", "cheap", "strong", "strong"]
    assert calls[1]["model"] == "gemini/gemini-3.5-flash-lite" and calls[1]["cost_usd"] == pytest.approx(0.002)
    m.reset_stats()
    assert m.stats.api_calls == 0 and strong.stats.api_calls == 0


def test_routed_model_caps_count_both_models(tmp_path):
    m, _, _ = _routed(tmp_path, [0.9, 0.1, 0.1], call_limit=2)
    m.query(_history(0))
    m.query(_history(1))
    with pytest.raises(Exception, match="call limit"):
        m.query(_history(2))
    m, _, _ = _routed(tmp_path, [0.9, 0.9], cost_limit=0.015)
    m.query(_history(0))
    with pytest.raises(Exception, match="cost limit"):
        m.query(_history(1))


def test_shadow_mode_keeps_the_strong_model(tmp_path):
    m, strong, cheap = _routed(tmp_path, [0.1, 0.1], mode="shadow")
    m.query(_history(0))
    m.query(_history(1))
    assert (strong.seen, cheap.seen) == (2, 0)


def test_install_is_a_noop_when_off(monkeypatch):
    monkeypatch.delenv("JEV_ROUTER_MODE", raising=False)
    monkeypatch.delenv("JEV_MODE", raising=False)
    assert R.install() is False


def test_trajectory_cost():
    pytest.importorskip("harbor")
    sys.path.insert(0, str(ROOT / "adapters/swe_agent"))
    from harbor_jev_swe_agent import trajectory_cost
    assert trajectory_cost({"info": {"model_stats": {"instance_cost": 0.42}}}) == 0.42
    assert trajectory_cost({"info": {}}) is None


def test_config_is_default_without_cache_control():
    import yaml
    cfg = yaml.safe_load((ROOT / "adapters/swe_agent/e25_default.yaml").read_text())
    assert "history_processors" not in cfg["agent"]
    assert cfg["agent"]["model"] == {"per_instance_cost_limit": 3.0, "per_instance_call_limit": 50,
                                     "completion_kwargs": {"thinkingConfig": {"thinkingLevel": "minimal"}}}
    assert cfg["agent"]["tools"]["parse_function"]["type"] == "function_calling"
    assert cfg["agent"]["tools"]["env_variables"] == {"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
                                                     "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}


def test_pi_router_gets_the_google_key(monkeypatch, tmp_path):
    pytest.importorskip("harbor")
    sys.path.insert(0, str(ROOT / "adapters/pi"))
    from harbor_jev_pi import JevPi
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-key")
    a = JevPi(logs_dir=tmp_path, model_name="s1/router", jev_extensions="router", version="0.99.2")
    conn = a.model_connection
    assert conn.provider == "s1" and dict(conn.env) == {"GEMINI_API_KEY": "test-not-a-key"}
    b = JevPi(logs_dir=tmp_path, model_name="google/gemini-3.5-flash", version="0.99.2")
    assert dict(b.model_connection.env) == {"GEMINI_API_KEY": "test-not-a-key"}


# ------------------------------------------------------------------ launchers
def _dry(script: str, *arms: str, **env: str) -> str:
    e = {**os.environ, "DRY_RUN": "1", "ENV_FILE": "/nonexistent", **env}
    return subprocess.run(["bash", str(ROOT / script), *arms], capture_output=True, text=True, env=e,
                          check=True).stderr


def test_tb2_launcher_arms():
    out = _dry("adapters/pi/run_tb2.sh", "baseline", "jev_router", "always_cheap", MAX_TURNS="100",
               JEV_WIRE_MODEL="jev-1.13")
    base, router, cheap = out.strip().splitlines()
    assert "-m google/gemini-3.5-flash " in base and "jev_extensions" not in base and "max_turns=100" in base
    assert "-m s1/router" in router and "jev_extensions=router" in router
    assert "JEV_ROUTER_STRONG=google/gemini-3.5-flash " in router + " " and "JEV_ROUTER_CHEAP=google/gemini-3.5-flash-lite" in router
    assert "JEV_ROUTER_TAU=0.5" in router and "JEV_WIRE_MODEL=jev-1.13" in router
    assert "-m google/gemini-3.5-flash-lite" in cheap and "jev_extensions" not in cheap
    assert "API_KEY" not in out


def test_swe_launcher_arms():
    ids = ROOT / "configs/explore/e25_harnesses/swe_smoke_ids.txt"
    out = _dry("adapters/swe_agent/run_swe.sh", "baseline", "jev_router", "always_cheap", IDS_FILE=str(ids))
    base, router, cheap = out.strip().splitlines()
    for line in (base, router, cheap):
        assert "harbor_jev_swe_agent:JevSweAgent" in line and "version=v1.1.0" in line and "litellm_version=1.82.6" in line
        assert all(f"-i *{i}" in line for i in E.read_ids(ids))
    assert "-m gemini/gemini-3.5-flash " in base and "jev_router=off" in base
    assert "jev_router=trigger" in router and "JEV_ROUTER_CHEAP=gemini/gemini-3.5-flash-lite" in router
    assert "-m gemini/gemini-3.5-flash-lite" in cheap and "jev_router=off" in cheap
    assert "API_KEY" not in out


# ------------------------------------------------------------------ task lists
def test_frozen_lists_match_the_rule():
    sel = json.loads((E.CFG / "selection.json").read_text())
    for fn in ("tb2_tasks.txt", "tb2_order.txt", "swe_order.txt", "swe_eval_ids.txt", "swe_smoke_ids.txt"):
        ids = E.read_ids(E.CFG / fn)
        assert sel[fn]["n"] == len(ids) and sel[fn]["sha256"] == E.sha256_text(E.ids_text(ids))
    tb = E.read_ids(E.CFG / "tb2_tasks.txt")
    assert E.read_ids(E.CFG / "tb2_order.txt") == E.seeded_order(tb, E.SALT["tb"])
    order = E.read_ids(E.CFG / "swe_order.txt")
    assert E.read_ids(E.CFG / "swe_eval_ids.txt") == order[:100]
    assert E.read_ids(E.CFG / "swe_smoke_ids.txt") == order[100:102]
    assert order == E.seeded_order(sorted(order), E.SALT["swe"]) and sel["swe_pool_equals_e4r_pool"]
    assert set(E.read_ids(E.CFG / "tb2_smoke_tasks.txt")) <= set(tb)


# ------------------------------------------------------------------ gate and readout on synthetic Harbor runs
def _trial(job: Path, task: str, *, reward: float, cost: float, secs: float, harness: str, arm: str,
           error: str | None = None, fast: int = 0, strong: int = 3, jev_err: int = 0) -> None:
    t = job / f"{task}__abc"
    (t / "agent").mkdir(parents=True)
    t0 = "2026-10-09T02:00:00+00:00"
    t1 = f"2026-10-09T02:{int(secs // 60):02d}:{int(secs % 60):02d}+00:00"
    (t / "result.json").write_text(json.dumps({
        "task_name": task, "agent_result": {"cost_usd": cost},
        "verifier_result": {"rewards": {"reward": reward}} if error is None else None,
        "exception_info": {"exception_type": error, "exception_message": "boom"} if error else None,
        "agent_execution": {"started_at": t0, "finished_at": t1}}))
    if arm == "jev_router":
        recs = [{"dp": "routing", "model": "jev-1.13", "final_value": "cheap", "input_tokens": 1000,
                 "request_id": f"f{i}"} for i in range(fast)]
        recs += [{"dp": "routing", "model": "jev-1.13", "final_value": "strong", "input_tokens": 1000,
                  "request_id": f"s{i}"} for i in range(strong - jev_err)]
        recs += [{"dp": "routing", "model": "jev-1.13", "final_value": "strong", "error": "x", "request_id": f"e{i}"}
                 for i in range(jev_err)]
        (t / "agent" / "jev-decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    names = ["gemini-3.5-flash-lite"] * (fast if arm == "jev_router" else (strong if arm == "always_cheap" else 0))
    names += ["gemini-3.5-flash"] * (0 if arm == "always_cheap" else strong)
    if harness == "tb":
        (t / "agent" / "pi" / "sessions").mkdir(parents=True)
        (t / "agent" / "pi" / "sessions" / "s.jsonl").write_text("".join(
            json.dumps({"type": "message", "message": {"role": "assistant", "model": n,
                                                       "usage": {"cost": {"total": 0.001}}}}) + "\n" for n in names))
    else:
        (t / "agent" / "jev-calls.jsonl").write_text("".join(
            json.dumps({"model": f"gemini/{n}", "cost_usd": 0.001, "api_calls": 1}) + "\n" for n in names))


def _run(root: Path, harness: str, tasks: list[str], *, router_cost=0.3, base_cost=0.5, cheap_cost=0.1, router_lose=(),
         **kw) -> Path:
    for arm in E.ARMS:
        for i, task in enumerate(tasks):
            cost = {"baseline": base_cost, "jev_router": router_cost, "always_cheap": cheap_cost}[arm]
            reward = 0.0 if (arm == "jev_router" and task in router_lose) else float(i % 2 == 0)
            _trial(root / arm, task, reward=reward, cost=cost, secs=100 if arm != "baseline" else 120,
                   harness=harness, arm=arm, fast=2, strong=3, **kw)
    return root


def test_gate_go_and_cut(tmp_path):
    # amended projection: per arm max(3 x smoke mean, floor 0.30 / 0.30 / 0.08), factor 1.2
    smoke = E.load_run(_run(tmp_path / "s", "swe", ["a__a-1", "b__b-2"], base_cost=0.12, router_cost=0.10,
                            cheap_cost=0.02), "swe")
    g = E.gate("swe", smoke, spent=1.0, cap=100.0)
    assert g["decision"] == "go" and g["checks"]["2_router_jev_answers"]["ok"]
    # 0.36 + 0.30 (+ 0.08) per task x 1.2 x 100 = 79.2 (88.8) <= 99: n = 100 with all three arms
    assert g["n"] == 100 and g["arms"] == list(E.ARMS)
    assert g["tasks"] == E.read_ids(E.CFG / "swe_order.txt")[:100]
    smoke = E.load_run(_run(tmp_path / "s2", "swe", ["a__a-1", "b__b-2"], base_cost=0.3, router_cost=0.2), "swe")
    g = E.gate("swe", smoke, spent=2.0, cap=100.0)
    # 0.9 + 0.6 = 1.5 x 1.2 = 1.8 per task: (100 - 2) / 1.8 = 54 -> 50; + 0.3 x 1.2 x 50 = 18 -> 108 > 98: no cheap arm
    assert g["n"] == 50 and g["arms"] == ["baseline", "jev_router"] and g["tasks"] == E.read_ids(E.CFG / "swe_order.txt")[:50]
    g = E.gate("tb", E.load_run(_run(tmp_path / "t", "tb", ["prove-plus-comm", "overfull-hbox"], base_cost=0.12,
                                     router_cost=0.10, cheap_cost=0.06), "tb"), spent=1.0, cap=140.0)
    # (0.36 + 0.30 + 0.18) x 1.2 x 89 = 89.7 <= 139: all 89 tasks, all three arms
    assert g["decision"] == "go" and g["n"] == 89 and g["full_dataset"] and g["arms"] == list(E.ARMS)
    g = E.gate("tb", E.load_run(tmp_path / "t", "tb"), spent=1.0, cap=60.0)
    # 0.66 x 1.2 x 89 = 70.5 > 59 -> the first 70 of the seeded order; cheap: 0.84 x 1.2 x 70 = 70.6 > 59
    assert g["n"] == 70 and not g["full_dataset"] and g["tasks"] == E.read_ids(E.CFG / "tb2_order.txt")[:70]
    assert g["arms"] == ["baseline", "jev_router"]


def test_gate_no_go(tmp_path):
    run = _run(tmp_path / "s", "swe", ["a__a-1", "b__b-2"], jev_err=3)
    g = E.gate("swe", E.load_run(run, "swe"), spent=1.0, cap=100.0)
    assert g["decision"] == "no-go" and not g["checks"]["2_router_jev_answers"]["ok"]
    run = _run(tmp_path / "s2", "swe", ["a__a-1"])
    _trial(run / "baseline", "b__b-2", reward=0, cost=0.5, secs=10, harness="swe", arm="baseline", error="RuntimeError")
    g = E.gate("swe", E.load_run(run, "swe"), spent=1.0, cap=100.0, leaks=1)
    assert g["decision"] == "no-go"
    assert g["checks"]["1_complete_no_infra_error"]["infra_errors"] and not g["checks"]["4_no_leak"]["ok"]
    g = E.gate("swe", E.load_run(_run(tmp_path / "s3", "swe", ["a__a-1"], base_cost=1.0), "swe"), spent=1.0, cap=100.0)
    assert g["decision"] == "no-go" and g["checks"]["5_projection"]["n"] < E.N_MIN


def test_readout_keys_and_stats(tmp_path):
    tasks = [f"t{i:02d}" for i in range(20)]
    tb = _run(tmp_path / "tb", "tb", tasks, router_lose={"t00"})
    swe = _run(tmp_path / "swe", "swe", tasks)
    out = tmp_path / "out"
    tex = tmp_path / "e25.tex"
    assert E.main(["readout", "--tb", str(tb), "--swe", str(swe), "--out-dir", str(out), "--tex", str(tex)]) == 0
    s = json.loads((out / "summary.json").read_text())
    p = s["harnesses"]["tb"]["paired"]["jev_router"]
    assert p["n"] == 20 and p["wins"] == 0 and p["losses"] == 1 and p["delta"] == pytest.approx(-5.0)
    assert p["ni"] == "no" and 0 <= p["ni_p"] <= 1
    jev = 2 * 1000 * 0.042 / 1e6 + 3 * 1000 * 0.042 / 1e6
    assert p["saving"] == pytest.approx((1 - (0.3 + jev) / 0.5) * 100)
    assert p["time_saving"] == pytest.approx(100 * (1 - 100 / 120)) and p["cheaper_share"] == 100
    r = s["harnesses"]["swe"]["arms"]["jev_router"]
    assert r["taken_share"] == pytest.approx(40.0) and r["route_jev_share"] == 100
    assert s["harnesses"]["swe"]["arms"]["always_cheap"]["taken_share"] == 100
    keys = dict(E_line.split("}{", 1) for E_line in
                (ln[len("\\defres{"):-1] for ln in tex.read_text().splitlines() if ln.startswith("\\defres{")))
    for h in ("tb", "swe"):
        for arm in ("baseline", "jev_router", "always_cheap"):
            for k in ("n", "success", "cost_usd", "wall_mean_s", "wall_median_s", "taken_share"):
                assert f"e25/{h}/{arm}/{k}" in keys
        for k in ("delta", "saving", "time_saving", "ni", "ni_p", "wins", "losses", "lo", "hi"):
            assert f"e25/{h}/jev_router/{k}" in keys
    assert keys["e25/tb/jev_router/delta"] == "$-$5.0" and keys["e25/swe/jev_router/delta"] == "$+$0.0"
    assert (out / "summary.md").read_text().startswith("# E25")


def test_spend_sums_finished_trials(tmp_path):
    run = _run(tmp_path / "s", "swe", ["a__a-1", "b__b-2"])
    sp = E.spend([run])
    assert sp["trials"] == 6 and sp["llm"] == pytest.approx(2 * (0.5 + 0.3 + 0.1))
    assert sp["jev"] == pytest.approx(2 * 5 * 1000 * 0.042 / 1e6)


def test_swe_agent_subclass_settings(monkeypatch, tmp_path):
    pytest.importorskip("harbor")
    sys.path.insert(0, str(ROOT / "adapters/swe_agent"))
    from harbor_jev_swe_agent import JevSweAgent
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-key")
    a = JevSweAgent(logs_dir=tmp_path, model_name="gemini/gemini-3.5-flash", version="v1.1.0", jev_router="trigger")
    assert a.extra_env["SWEAGENT_CONFIG"] == "/opt/jev-swe/e25_default.yaml" and a.extra_env["JEV_ROUTER_MODE"] == "trigger"
    assert "jev_router" not in a.build_cli_flags() and "litellm" not in a.build_cli_flags()
    assert dict(a.model_connection.env) == {"GEMINI_API_KEY": "test-not-a-key"}
    with pytest.raises(ValueError):
        JevSweAgent(logs_dir=tmp_path, model_name="gemini/gemini-3.5-flash", jev_router="sometimes")


def test_swe_wrapper_keeps_the_task_python():
    pytest.importorskip("harbor")
    sys.path.insert(0, str(ROOT / "adapters/swe_agent"))
    from harbor_jev_swe_agent import WRAPPER
    assert "activate" not in WRAPPER and WRAPPER.startswith("#!/bin/bash\n")
    assert 'exec /opt/sweagent-venv/bin/python /opt/jev-swe/jev_sweagent.py "$@"' in WRAPPER


def test_rerun_replaces_infra_errors(tmp_path):
    run = _run(tmp_path / "e" / "eval", "swe", ["a__a-1", "b__b-2"])
    _trial(run / "jev_router", "c__c-3", reward=0, cost=0.0, secs=10, harness="swe", arm="jev_router",
           error="RuntimeError")
    _trial(run / "baseline", "c__c-3", reward=1, cost=0.5, secs=100, harness="swe", arm="baseline")
    assert E.errored(run, "swe") == [("jev_router", "c__c-3")]
    rr = tmp_path / "e" / "eval_rerun"
    _trial(rr / "jev_router", "c__c-3", reward=1, cost=0.3, secs=90, harness="swe", arm="jev_router", fast=1, strong=2)
    _trial(rr / "jev_router", "a__a-1", reward=0, cost=0.3, secs=90, harness="swe", arm="jev_router")  # not errored: ignored
    rows = E.load_run(run, "swe", rerun_dir=rr)["jev_router"]
    assert rows["c__c-3"]["success"] == 1 and rows["c__c-3"]["rerun"] and rows["a__a-1"]["success"] == 1
    out = tmp_path / "out"
    assert E.main(["readout", "--swe", str(run), "--out-dir", str(out), "--tex", str(tmp_path / "x.tex")]) == 0
    s = json.loads((out / "summary.json").read_text())["harnesses"]["swe"]
    assert s["reruns"] == ["jev_router/c__c-3: RuntimeError: boom -> reward 1"]
    assert s["paired"]["jev_router"]["n"] == 3 and s["paired"]["jev_router"]["losses"] == 0


def test_watchdog_stop_keeps_the_tasks_complete_in_every_arm(tmp_path):
    # amendment 2026-10-09 ~05:15 UTC: a stopped (CancelledError) or never-started trial removes its task from both views
    tasks = ["a__a-1", "b__b-2", "c__c-3", "d__d-4"]
    run = _run(tmp_path / "e" / "eval", "swe", tasks[:2])
    for arm in E.ARMS:
        if arm != "jev_router":
            for t in tasks[2:]:
                _trial(run / arm, t, reward=1, cost=0.5, secs=100, harness="swe", arm=arm)
    _trial(run / "jev_router", "c__c-3", reward=0, cost=0.2, secs=50, harness="swe", arm="jev_router",
           error="CancelledError")                                   # stopped; d__d-4 never started
    out = tmp_path / "out"
    assert E.main(["readout", "--swe", str(run), "--out-dir", str(out), "--tex", str(tmp_path / "x.tex")]) == 0
    s = json.loads((out / "summary.json").read_text())["harnesses"]["swe"]
    assert s["left_out"] == {"c__c-3": ["jev_router"], "d__d-4": ["jev_router"]} and s["n_common_all_arms"] == 2
    assert all(st["n"] == 2 for st in s["arms"].values())
    assert s["paired"]["jev_router"]["n"] == 2 and s["paired_complete_case"]["always_cheap"]["n"] == 2
    assert "\\defres{e25/swe/n_left_out}{2}" in (tmp_path / "x.tex").read_text()
    # without a stop nothing is left out, even when a task is missing in one arm
    assert E.restrict_to_complete(E.load_run(_run(tmp_path / "f", "swe", tasks[:2]), "swe"))[1] == {}

def test_router_fills_missing_thought_signatures(tmp_path):
    import shutil
    node = shutil.which("node") or ("/opt/node22/bin/node" if Path("/opt/node22/bin/node").exists() else None)
    if node is None:
        pytest.skip("node not available")
    src = (ROOT / "adapters/pi/jev-router.ts").read_text()
    start = src.index("export const GEMINI_DUMMY_SIGNATURE")
    end = src.index("function splitSpec")
    js = src[start:end].replace("export ", "").replace("(payload: unknown): number", "(payload)")
    js = js.replace("as { contents?: unknown }", "").replace("as { parts?: unknown }", "")
    js = js.replace("parts as Record<string, unknown>[]", "parts")
    js = js.replace("const GEMINI_DUMMY_SIGNATURE = ", "const GEMINI_DUMMY_SIGNATURE = ")
    test = js + """
const p = {model: "gemini-3.5-flash", contents: [
  {role: "user", parts: [{text: "hi"}]},
  {role: "model", parts: [{functionCall: {name: "bash", args: {}}, thoughtSignature: "QUJD"}]},
  {role: "model", parts: [{text: "x"}, {functionCall: {name: "bash", args: {}}}]}]};
const n = fillMissingThoughtSignatures(p);
console.log(JSON.stringify({n, a: p.contents[1].parts[0].thoughtSignature, b: p.contents[2].parts[1].thoughtSignature,
  t: p.contents[2].parts[0].thoughtSignature === undefined, other: fillMissingThoughtSignatures({messages: []})}));
"""
    f = tmp_path / "t.mjs"
    f.write_text(test)
    out = json.loads(subprocess.run([node, str(f)], capture_output=True, text=True, check=True).stdout)
    assert out == {"n": 1, "a": "QUJD", "b": "skip_thought_signature_validator", "t": True, "other": 0}
