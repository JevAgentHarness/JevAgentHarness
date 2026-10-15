"""StartLux-Decision as a third System-1 family (docs/PREREG_A7_STAGE2.md A7.18): the vendor's `/v1/systemone` answers
parse through the harness's Jev wire backend unchanged (choice with probabilities, noul, score; a 69-option list in one
request, the server doing its own knockout); the serving plan's GPU rules (A7.18 (g)) and the smoke test against a
stand-in server; the configs of A7.18 (e); per-backend System-1 caches; the backend threshold filler (A7.18 (e) 4);
the serving script's dry run. No model is loaded and no network is used."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import startlux_plan as SP  # noqa: E402


def _vendor_answer(q: dict) -> dict:
    """What startlux_decision.model._answer returns (the vendor's response format at a9101e9)."""
    t = q.get("type", "choice")
    if t == "noul":
        return {"type": "noul", "noul": 0.8}
    if t == "score":
        n = len(q["criteria"])
        p = [1.0 / n] * n
        return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": max(p),
                "legend": {str(i): str(c) for i, c in enumerate(q["criteria"])},
                "probabilities": {str(i): v for i, v in enumerate(p)}}
    keys = list(q["criteria"])
    best = "cancel_order" if "cancel_order" in keys else keys[0]
    rest = (1 - 0.9) / max(1, len(keys) - 1)
    dist = {k: (0.9 if k == best else rest) for k in keys}
    return {"type": "choice", "choice": best, "confidence": 0.9, "probabilities": dist}


def _vendor_response(body: dict) -> dict:
    return {"answers": {k: _vendor_answer(q) for k, q in body["questions"].items()},
            "usage": {"input_tokens": 1234, "output_tokens": 0}, "model": "StartLux-Decision-4B"}


def test_vendor_answers_parse_through_the_jev_wire_backend():
    from jevharness.decision import Choice, Noul, Score
    from jevharness.decision.jev import JevBackend

    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json=_vendor_response(body))

    b = JevBackend(name="startlux-4b", base_url="http://localhost:8090", model="StartLux-Decision-4B", api_key="",
                   price_model="StartLux-Decision-4B", transport=httpx.MockTransport(handler))
    tools = [f"tool_{i:02d}" for i in range(68)] + ["cancel_order"]
    qs = [Choice("Which tool should be called next?", tools), Noul("Is the task done?"),
          Score("How severe?", scale=(1, 3))]
    res = b.decide("state", qs)
    assert len(seen) == 1 and len(seen[0]["questions"]["q0"]["criteria"]) == 69    # one request; vendor knockout
    a_choice, a_noul, a_score = res.answers
    assert a_choice.value == "cancel_order" and abs(a_choice.confidence - 0.9) < 1e-9
    assert a_noul.value in ("yes", True) and a_score.value in (1, 2, 3)
    assert res.usage.input_tokens == 1234 and res.usage.model == "StartLux-Decision-4B"


def test_serving_plan_follows_the_gpu_rules():
    l4, l40s, a100 = 23034, 46068, 81559
    by = {r["size"]: r for r in SP.plan(["4B", "27B"], [l4])}
    assert by["4B"]["mode"] == "original" and by["27B"]["mode"] == "not_run" and "not run" in by["27B"]["reason"]
    by = {r["size"]: r for r in SP.plan(["4B", "27B"], [l4, l40s])}
    assert by["27B"]["mode"] == "gguf_q8_0" and by["27B"]["repo"].endswith("-Q8_0-GGUF") and by["27B"]["gpu"] == 1
    by = {r["size"]: r for r in SP.plan(["4B", "27B"], [a100, l4])}
    assert by["27B"]["mode"] == "original" and by["27B"]["gpu"] == 0 and by["4B"]["gpu"] == 1
    assert {r["mode"] for r in SP.plan(["9B"], [l4])} == {"not_run"}                 # no fallback for the ladder
    assert all("Q4_K_M" not in json.dumps(r) for r in SP.plan(["27B"], [l40s]))      # never substituted


def test_smoke_against_a_stand_in_server():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps(_vendor_response(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):  # noqa: ANN002
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        r = SP.smoke(srv.server_address[1], "StartLux-Decision-4B", timeout=10)
    finally:
        srv.shutdown()
    assert r["ok"] and [q["questions"] for q in r["requests"]] == [3, 1]
    assert r["requests"][1]["n_probabilities"] == {"next_tool": 69}
    assert SP.smoke(1, None, timeout=2)["ok"] is False                                 # nothing listening: recorded


def test_smoke_flags_malformed_answers():
    small, wide = SP.smoke_requests()
    q = wide["questions"]["next_tool"]
    assert SP.answer_problems(q, _vendor_answer(q)) == []
    short = {"type": "choice", "choice": "cancel_order", "probabilities": {"cancel_order": 1.0}}
    assert SP.answer_problems(q, short)                                               # 1 of 69 options answered
    assert SP.answer_problems(small["questions"]["urgent"], {"type": "noul", "noul": 1.7})
    sev = small["questions"]["severity"]
    assert SP.answer_problems(sev, _vendor_answer(sev)) == []
    assert SP.answer_problems(sev, {"type": "score", "probabilities": {"0": 0.5, "1": 0.5}})   # 2 of 3 levels
    assert SP.answer_problems(q, None) == ["no answer object"]


def test_startlux_configs_and_per_backend_cache():
    from jevharness.bench.runner import _backend_cache, _strip_placeholder, arm_bindings, arm_task_counts, expand_arms

    st = yaml.safe_load((ROOT / "configs/experiments/stage2_startlux_states.yaml").read_text())
    s2 = yaml.safe_load((ROOT / "configs/experiments/stage2_jevgym.yaml").read_text())
    assert st["suite"] == s2["suite"] and [a.name for a in expand_arms(st)] == ["sl4b_ts_replace_f", "sl27b_ts_replace_f"]
    assert set(arm_task_counts(st).values()) == {200} and all(v["placeholder"] for v in st["s1_backends"].values())
    assert arm_bindings(st)["sl27b_ts_replace_f"]["s1"] == "startlux_27b"
    cal = yaml.safe_load((ROOT / "configs/experiments/jevgym_shadow_cal_startlux.yaml").read_text())
    assert cal["suite"]["seed0"] == 5000 and cal["suite"]["n"] == 150 and cal["s1"]["base_url"].endswith(":8090")
    e1 = yaml.safe_load((ROOT / "configs/decision/bfcl_when2call_startlux.yaml").read_text())
    ref = yaml.safe_load((ROOT / "configs/decision/bfcl_when2call.yaml").read_text())
    assert e1["suites"] == ref["suites"] and e1["exclude_ids_file"] == ref["exclude_ids_file"]       # C8 population
    assert set(e1["backends"]) == {"startlux_4b", "startlux_27b", "intern_4b"}
    sl = s2["s1_backends"]["startlux_4b"]
    assert sl["placeholder"] and _backend_cache(sl, s2) == "cache/s1_stage2_startlux.sqlite"
    assert "cache" not in _strip_placeholder(sl) and "placeholder" not in _strip_placeholder(sl)
    assert _backend_cache(s2["s1_backends"]["jevagent_rl"], s2) == s2["s1_cache"]
    arm = next(a for a in s2["arms"] if a["name"] == "jevagent_best_sl4b")
    assert arm["s1"] == "startlux_4b" and arm["profile_backend"] == "startlux_4b"


def test_fill_backend_tau(tmp_path: Path):
    import fill_backend_tau as F

    from jevharness.harness import profile as HP

    prof = {"kind": "jevagent_profile", "version": 1, "status": "selected", "system2": {"gemini-3.5-flash": {
        "decision_points": {"tool_select": {"mode": "cascade", "tau": 1.0}, "tool_params": {"mode": "cascade", "tau": 0.99},
                            "stop": {"mode": "trigger", "tau": 0.9}},
        "backends": {"startlux_4b": {"tool_select": {"calibration": "pending"},
                                     "tool_params": {"calibration": "pending"}}}}}}
    fits = {"tool_select": {"primary": {"tau_hat": 0.97}}, "tool_params": {"primary": {"tau_hat": float("inf")}}}
    out = F.fill(prof, "startlux_4b", fits)
    blk = out["system2"]["gemini-3.5-flash"]["backends"]["startlux_4b"]
    assert blk["tool_select"] == {"tau": 0.97} and blk["tool_params"] == {"tau": 0.99, "always_defer": True}
    assert "skip" not in blk
    only_tool = json.loads(json.dumps(prof))
    del only_tool["system2"]["gemini-3.5-flash"]["decision_points"]["stop"]
    out = F.fill(only_tool, "startlux_4b", {"tool_select": {"primary": {"tau_hat": None}},
                                            "tool_params": {"primary": {"tau_hat": 0.5}}})
    blk = out["system2"]["gemini-3.5-flash"]["backends"]["startlux_4b"]
    assert "delegates nothing" in blk["skip"]                                        # A7.18 (f): S2-4S not tested
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(out))
    loaded = HP.load_profile(p)
    assert "delegates nothing" in HP.decision_points(loaded, "gemini-3.5-flash", backend="startlux_4b")[2]
    assert HP.decision_points(loaded, "gemini-3.5-flash")[2] is None                 # the Jev arms are untouched


def test_serving_script_dry_run(tmp_path: Path):
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "nvidia-smi").write_text("#!/bin/sh\ncase \"$*\" in *memory.total,*) echo '0, NVIDIA L4, 23034 MiB';; "
                                     "*) echo 23034;; esac\n")
    (stub / "nvidia-smi").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "DRY_RUN": "1", "BASE": str(tmp_path / "sl"),
           "MANIFEST": str(tmp_path / "m.json"), "PYTHON": sys.executable}
    r = subprocess.run(["bash", str(ROOT / "scripts/aws/serve_startlux.sh")], capture_output=True, text=True, env=env,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    sizes = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
    by = {s["size"]: s for s in sizes}
    assert by["4B"]["status"] == "original" and by["27B"]["status"] == "not_run"     # one L4: the 27B is not run
    assert SP.PKG_COMMIT in r.stdout and "flash-linear-attention" in r.stdout and "causal-conv1d" in r.stdout
    assert "startlux_decision.server" in r.stdout and "--port 8090" in r.stdout
    assert not (tmp_path / "m.json").exists()                                        # a dry run writes no manifest
