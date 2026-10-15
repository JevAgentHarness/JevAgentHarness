#!/usr/bin/env python3
r"""Exploratory row E5 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06 "Exploratory row E5" (g)): the cascade
thresholds of an E5 model fitted OFFLINE, at $0, from an existing calibration shadow arm.

In shadow mode System-1 never acts, so the trajectories of a calibration shadow arm do not depend on it. Every
tool_select and tool_params question that System-1 was asked in the frozen JevAgentRL-4B's own calibration arm
(``rl_shadow_cal`` of ``runs/e2a_jevgym_shadow_cal_rl``: Flash, calibration seeds 5000-5149) is rebuilt exactly as the
agent asks it (tool descriptions regenerated from the task seed, as ``scripts/harness_final.py workload``), answered by
a served model through the harness's own asking path (``JevAgent._choice_tournament`` / ``JevAgent._ask``), and the
answers are written into a copy of the arm in place of the logged System-1 fields; the registered command
(``threshold_theory --calibration-arm``) then fits tau-hat* on that copy per decision point.

    python scripts/e5_cal_replay.py build  --run runs/e2a_jevgym_shadow_cal_rl --arm rl_shadow_cal \
        --out runs/e5_small_inputs/cal_requests.jsonl.gz                 # CPU, local (JevGym tools regenerated)
    python scripts/e5_cal_replay.py answer --requests REQ --url http://127.0.0.1:8082 --name e5-e5_q17 \
        --out runs/e5_small/cal/answers_e5_q17.jsonl                     # GPU host, the served model; resumes
    python scripts/e5_cal_replay.py check  --requests REQ --answers ANS --out CHECK.json   # vs the logged answers
    python scripts/e5_cal_replay.py derive --run runs/e2a_jevgym_shadow_cal_rl --arm rl_shadow_cal --answers ANS \
        --out-dir runs/e5_small_cal --name e5cal_e5_q17                  # -> RUN/<name>.jsonl for threshold_theory
    python scripts/e5_cal_replay.py fit    --run-dir runs/e5_small_cal --name e5cal_e5_q17 \
        --out-dir results/explore/e5_small/cal                          # threshold_theory per decision point

The check is the frozen 4B replayed through the same path: its argmax must agree with the logged answers on >= 99 %
of the questions at each decision point (the Deviations entry's validation rule). No LLM call anywhere; an answer
whose ledger shows a cost stops ``answer``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DPS = ("tool_select", "tool_params")
AGREE_MIN = 0.99
LABEL = "exploratory (E5), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-06 'Exploratory row E5' (g)"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    out = []
    with op(p, "rt") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    """task_id -> the latest record (the runner's resume rule)."""
    last: dict[str, dict[str, Any]] = {}
    for r in _jsonl(path):
        if isinstance(r, dict) and "task_id" in r:
            last[str(r["task_id"])] = r
    return last


def key_of(task_id: str, step: Any, i: int) -> str:
    return f"{task_id}|{step}|{i}"


# ------------------------------------------------------------------ build (local, CPU)
def iter_questions(recs: Mapping[str, Mapping[str, Any]], tools_of=None) -> Iterable[dict[str, Any]]:
    """The System-1 questions of a shadow arm's tool_select / tool_params decisions, in task / step / index order.
    ``tools_of(task_id) -> (names, descriptions)`` regenerates the tool descriptions (``harness_final._env_tools``)."""
    from jevharness.harness.prompts import Q_TOOL_SELECT

    for tid in sorted(recs):
        r = recs[tid]
        if r.get("error"):
            continue
        names = descs = None
        for s in r.get("steps") or []:
            for i, d in enumerate(s.get("decisions") or []):
                if d.get("dp") not in DPS or d.get("s1_value") is None or not d.get("options"):
                    continue
                q: dict[str, Any] = {"key": key_of(tid, s.get("step"), i), "task_id": tid, "step": s.get("step"),
                                     "i": i, "dp": d["dp"], "state": d["state"], "options": list(d["options"]),
                                     "logged": {"value": d.get("s1_value"), "conf": d.get("s1_conf"),
                                                "dist": d.get("s1_dist")}}
                if d["dp"] == "tool_select":
                    if names is None:
                        names, descs = tools_of(tid) if tools_of else (None, None)
                    if names is not None and list(d["options"]) != list(names):
                        raise ValueError(f"{tid} step {s.get('step')}: logged tool options differ from the "
                                         "regenerated tools")
                    q.update({"text": Q_TOOL_SELECT, "descriptions": list(descs) if descs is not None else None})
                else:
                    q.update({"text": d["question"], "descriptions": None})
                yield q


def build(run: Path, arm: str, out: Path) -> dict[str, Any]:
    import yaml

    from harness_final import _env_tools  # noqa: PLC0415  (the workload's tool regeneration)

    exp = yaml.safe_load((run / "experiment.yaml").read_text())
    suite = exp["suite"]
    src = run / f"{arm}.jsonl"
    recs = load_latest(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = {dp: 0 for dp in DPS}
    with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for q in iter_questions(recs, lambda tid: _env_tools(tid, suite)):
            gz.write((json.dumps(q, sort_keys=True) + "\n").encode())
            n[q["dp"]] += 1
    meta = {"label": LABEL, "source": f"{run.name}/{arm}.jsonl", "source_sha256": _sha256(src),
            "tasks": len(recs), "questions": n, "out": out.name, "sha256": _sha256(out)}
    out.with_suffix("").with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# ------------------------------------------------------------------ answer (GPU host, a served model)
def answer(requests: Path, url: str, name: str, out: Path, timeout: float = 60.0, retries: int = 2,
           backend: Any = None) -> dict[str, Any]:
    """Answer every request through the harness's asking path (resumes: keys already in ``out`` are skipped)."""
    from jevharness.accounting import Ledger
    from jevharness.decision import make_backend
    from jevharness.decision.types import Choice
    from jevharness.harness.agent import JevAgent
    from jevharness.harness.config import DPConfig, HarnessConfig, Mode

    if backend is None:
        backend = make_backend({"type": "jev", "name": name, "base_url": url, "model": name, "api_key": "",
                                "timeout": timeout})
    shell = JevAgent(llm=None, cfg=HarnessConfig(name="e5_cal_replay", tool_select=DPConfig(mode=Mode.REPLACE)),
                     s1=backend)
    done = {r["key"] for r in _jsonl(out)} if out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    n_new = n_err = 0
    cost = 0.0
    with open(out, "a") as f:
        for q in _jsonl(requests):
            if q["key"] in done:
                continue
            rec: dict[str, Any] = {"key": q["key"], "dp": q["dp"], "value": None, "conf": None, "dist": None,
                                   "error": None}
            for attempt in range(retries + 1):
                led = Ledger()
                t0 = time.perf_counter()
                try:
                    if q["dp"] == "tool_select":
                        a = shell._choice_tournament(q["state"], q["text"], list(q["options"]), "tool_select", led,
                                                     q.get("descriptions") or None)
                    else:
                        a = shell._ask(q["state"], Choice(q["text"], list(q["options"]), decision_point="tool_params"),
                                       led)
                    rec.update({"value": a.value, "conf": float(a.confidence),
                                "dist": {str(k): float(v) for k, v in a.distribution.items()}, "error": None})
                except Exception as e:  # noqa: BLE001 - a failed question is recorded, never a crash
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                rec.update({"attempts": attempt + 1, "cost_usd": sum(r.cost_usd for r in led.records),
                            "latency_ms": sum(r.latency_ms for r in led.records),
                            "wall_ms": (time.perf_counter() - t0) * 1000, "requests": len(led.records)})
                if rec["error"] is None:
                    break
            if rec["cost_usd"]:
                raise SystemExit(f"{q['key']}: a priced System-1 answer (${rec['cost_usd']}): stop (E5 runs at $0)")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_new += 1
            n_err += rec["error"] is not None
            cost += rec["cost_usd"]
    return {"answered_now": n_new, "errors_now": n_err, "resumed_with": len(done), "cost_usd": cost}


# ------------------------------------------------------------------ check (vs the logged answers)
def check(requests: Path, answers: Path, agree_min: float = AGREE_MIN) -> dict[str, Any]:
    ans = {r["key"]: r for r in _jsonl(answers)}
    res: dict[str, Any] = {"label": LABEL, "agree_min": agree_min, "by_dp": {}}
    for dp in DPS:
        qs = [q for q in _jsonl(requests) if q["dp"] == dp]
        got = [ans.get(q["key"]) for q in qs]
        agree = sum(1 for q, a in zip(qs, got) if a and a.get("error") is None and a["value"] == q["logged"]["value"])
        dmax = [max([abs(float((a.get("dist") or {}).get(k, 0.0)) - float(v))
                     for k, v in (q["logged"]["dist"] or {}).items()] or [0.0])
                for q, a in zip(qs, got) if a and a.get("error") is None]
        dconf = [abs(float(a["conf"]) - float(q["logged"]["conf"])) for q, a in zip(qs, got)
                 if a and a.get("error") is None and q["logged"]["conf"] is not None]
        n = len(qs)
        res["by_dp"][dp] = {"questions": n, "answered": sum(1 for a in got if a is not None),
                            "errors": sum(1 for a in got if a and a.get("error")), "argmax_agree": agree,
                            "agreement": agree / n if n else float("nan"),
                            "max_abs_dp_max": max(dmax) if dmax else None,
                            "max_abs_dp_median": sorted(dmax)[len(dmax) // 2] if dmax else None,
                            "abs_dconf_max": max(dconf) if dconf else None,
                            "ok": bool(n) and agree / n >= agree_min}
    res["ok"] = all(v["ok"] for v in res["by_dp"].values())
    return res


# ------------------------------------------------------------------ derive (a copy of the arm with the new answers)
def derive(run: Path, arm: str, answers: Path, out_dir: Path, name: str, model: str = "") -> dict[str, Any]:
    """RUN/ARM.jsonl -> OUT_DIR/NAME.jsonl: every tool_select / tool_params decision's System-1 fields replaced by the
    answers (value, confidence, distribution; cost 0, the replay's latency); the other decisions (stop) are dropped,
    so the copy can only be read at the two decision points it was re-answered on. A question without an answer (or
    with an error record) keeps no System-1 answer (s1_value None: not a paired decision)."""
    import yaml

    ans = {r["key"]: r for r in _jsonl(answers)}
    recs = load_latest(run / f"{arm}.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = {"replaced": 0, "missing": 0, "dropped_other_dps": 0}
    with open(out_dir / f"{name}.jsonl", "w") as f:
        for tid in sorted(recs):
            r = json.loads(json.dumps(recs[tid]))
            if r.get("error"):
                continue
            for s in r.get("steps") or []:
                keep = []
                for i, d in enumerate(s.get("decisions") or []):
                    if d.get("dp") not in DPS:
                        n["dropped_other_dps"] += 1
                        continue
                    if d.get("s1_value") is not None and d.get("options"):
                        a = ans.get(key_of(tid, s.get("step"), i))
                        if a is None or a.get("error"):
                            d.update({"s1_value": None, "s1_conf": None, "s1_dist": None})
                            n["missing"] += 1
                        else:
                            d.update({"s1_value": a["value"], "s1_conf": a["conf"], "s1_dist": a["dist"],
                                      "s1_cost_usd": 0.0, "s1_latency_ms": a.get("latency_ms")})
                            n["replaced"] += 1
                    keep.append(d)
                s["decisions"] = keep
            r["config"] = name
            r.setdefault("meta", {})["e5_cal_replay"] = {"source_arm": arm, "model": model, "answers": answers.name}
            f.write(json.dumps(r) + "\n")
    exp = yaml.safe_load((run / "experiment.yaml").read_text())
    exp["name"] = out_dir.name
    exp["arms"] = [{**next(a for a in exp["arms"] if a["name"] == arm), "name": name}]
    exp["e5_cal_replay"] = {"label": LABEL, "source": f"{run.name}/{arm}.jsonl",
                            "source_sha256": _sha256(run / f"{arm}.jsonl"), "answers": answers.name,
                            "answers_sha256": _sha256(answers), "model": model}
    (out_dir / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    return n


def fit(run_dir: Path, name: str, out_dir: Path) -> dict[str, Any]:
    """The registered command per decision point (A6.11 (c), A6.19; as the A7.18 (e) 4 fits)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for dp in DPS:
        out = out_dir / f"tauhat_{name}_{dp}.json"
        subprocess.run([sys.executable, "-m", "jevharness.bench.threshold_theory", str(run_dir), "--arm", name,
                        "--dp", dp, "--calibration-arm", "--task-value", "1.0", "--lam-per-hour", "0",
                        "--v-fallback", "W", "--format", "json", "--out", str(out)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        d = json.loads(out.read_text())
        res[dp] = {"tau_hat": d.get("tau_hat"), "n_paired": d.get("n_paired"),
                   "tau_rule": (d.get("tau_rule") or {}).get("rule"), "path": str(out)}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", default="runs/e2a_jevgym_shadow_cal_rl")
    b.add_argument("--arm", default="rl_shadow_cal")
    b.add_argument("--out", required=True)
    a = sub.add_parser("answer")
    a.add_argument("--requests", required=True)
    a.add_argument("--url", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--timeout", type=float, default=60.0)
    c = sub.add_parser("check")
    c.add_argument("--requests", required=True)
    c.add_argument("--answers", required=True)
    c.add_argument("--out", default=None)
    d = sub.add_parser("derive")
    d.add_argument("--run", default="runs/e2a_jevgym_shadow_cal_rl")
    d.add_argument("--arm", default="rl_shadow_cal")
    d.add_argument("--answers", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--model", default="")
    f = sub.add_parser("fit")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--name", required=True)
    f.add_argument("--out-dir", required=True)
    x = ap.parse_args(argv)
    if x.cmd == "build":
        print(json.dumps(build(Path(x.run), x.arm, Path(x.out)), indent=1))
    elif x.cmd == "answer":
        print(json.dumps(answer(Path(x.requests), x.url, x.name, Path(x.out), x.timeout)))
    elif x.cmd == "check":
        res = check(Path(x.requests), Path(x.answers))
        if x.out:
            Path(x.out).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({dp: {k: v[k] for k in ("questions", "agreement", "ok")} for dp, v in res["by_dp"].items()}))
        return 0 if res["ok"] else 1
    elif x.cmd == "derive":
        print(json.dumps(derive(Path(x.run), x.arm, Path(x.answers), Path(x.out_dir), x.name, x.model)))
    elif x.cmd == "fit":
        print(json.dumps(fit(Path(x.run_dir), x.name, Path(x.out_dir)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
