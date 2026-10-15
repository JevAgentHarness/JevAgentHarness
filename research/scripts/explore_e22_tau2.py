#!/usr/bin/env python3
"""Exploratory row E22 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E22: the training recipe on tau^2-bench";
descriptive, outside every Holm table): the paper's recipe (train a small decision model on the states the agent
itself visits, gate it by confidence) on tau^2-bench retail and airline, with the benchmark's official train / test
task splits (``data/tau2/domains/<domain>/split_tasks.json`` of tau2-bench 5bfa7e3).

    python scripts/explore_e22_tau2.py split                       # train (fit / dev) and test task ids -> split.json
    python scripts/explore_e22_tau2.py harvest --domain retail [--smoke] [--trials K]   # local, Flash LLM-only
    python scripts/explore_e22_tau2.py spend                       # logged Gemini spend of every E22 run
    python scripts/explore_e22_tau2.py build                       # label rule (d) -> Clef records (fit / dev)
    python scripts/explore_e22_tau2.py replay --base-url URL --out F   # box: the served model answers the dev questions
    python scripts/explore_e22_tau2.py fit --answers F --out-dir D     # box: tau-hat per decision point (registered cmd)
    python scripts/explore_e22_tau2.py summarize                   # readout -> results/explore/e22_tau2, paper keys

Labels never reach a served model: a record's gold goes only into training targets (``clef_data.wire_request``).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CFG = ROOT / "configs/explore/e22_tau2"
RES = ROOT / "results/explore/e22_tau2"
SPLIT = CFG / "split.json"
DOMAINS = ("retail", "airline")
DEV_FRAC = 0.2
DEV_SALT = "jev-explore-E22-tau2-dev-2026-10-08"
TAU2_COMMIT = "5bfa7e3"
AGENT_LLM = "gemini/gemini-3.5-flash"
USER_LLM = "gemini/gemini-3.5-flash-lite"
JUDGE_LLM = "gemini/gemini-3.5-flash"
LLM_ARGS = {"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}
HARVEST_SEED, EVAL_SEED, SMOKE_SEED = 2201, 2202, 2299
HARVEST_TRIALS = 8
CAP_USD = 200.0
JUDGE_USD_PER_RETAIL_SIM = 0.003     # the retail NL judge is not logged by tau2: a fixed estimate per retail simulation
SCRATCH = Path(os.environ.get("E22_SCRATCH", "/tmp/e22_tau2"))
S3 = os.environ.get("E22_S3", "s3://<bucket>/e22_tau2")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tau2_root() -> Path:
    import tau2
    return Path(tau2.__file__).resolve().parents[2]


def natural(s: str) -> tuple[int, Any]:
    return (0, int(s)) if str(s).isdigit() else (1, str(s))


# ================================================================================================ split
def make_split() -> dict[str, Any]:
    """Registration (b): the official train / test lists; dev = the first 20 % (rounded) of the train tasks in the order
    of sha256(DEV_SALT + newline + "<domain>/<task id>"); the rest are the fit tasks."""
    root = tau2_root()
    out: dict[str, Any] = {"what": "E22 task split (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E22' (b))",
                           "tau2_bench": TAU2_COMMIT, "dev_rule": f"first round({DEV_FRAC} x n_train) of the train ids "
                           f"ordered by sha256('{DEV_SALT}' + newline + '<domain>/<id>')", "domains": {}}
    for d in DOMAINS:
        f = root / f"data/tau2/domains/{d}/split_tasks.json"
        sp = json.loads(f.read_text())
        train, test = [str(x) for x in sp["train"]], [str(x) for x in sp["test"]]
        assert not set(train) & set(test), d
        order = sorted(train, key=lambda t: sha256_bytes(f"{DEV_SALT}\n{d}/{t}".encode()))
        n_dev = int(round(DEV_FRAC * len(train)))
        dev = sorted(order[:n_dev], key=natural)
        fit = sorted(order[n_dev:], key=natural)
        out["domains"][d] = {"split_file": f"data/tau2/domains/{d}/split_tasks.json", "split_sha256": sha256_file(f),
                             "n_train": len(train), "n_test": len(test), "fit": fit, "dev": dev,
                             "test": sorted(test, key=natural)}
    return out


def load_split() -> dict[str, Any]:
    sp = json.loads(SPLIT.read_text())
    again = make_split() if _has_tau2() else None
    if again is not None and again["domains"] != sp["domains"]:
        raise SystemExit("configs/explore/e22_tau2/split.json does not match the installed tau2 split files")
    return sp


def _has_tau2() -> bool:
    try:
        import tau2  # noqa: F401
        return True
    except ImportError:
        return False


def cmd_split(a: argparse.Namespace) -> int:
    sp = make_split()
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(sp, indent=1) + "\n")
    for d, v in sp["domains"].items():
        print(f"{d}: train {v['n_train']} (fit {len(v['fit'])}, dev {len(v['dev'])}), test {v['n_test']}")
        print(f"  dev:  {' '.join(v['dev'])}")
    return 0


# ================================================================================================ harvest (local)
def _env_for_run() -> dict[str, str]:
    """The subprocess environment: the direct Gemini key only (litellm reads GEMINI_API_KEY); no OpenAI / OpenRouter /
    Jev key, so nothing else can be called. Never printed."""
    from dotenv import dotenv_values

    vals = dotenv_values(ROOT / ".env")
    env = {k: v for k, v in os.environ.items() if not re.search(r"(API_KEY|_KEYS)$", k)}
    key = vals.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY missing")
    env["GEMINI_API_KEY"] = key
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_args(domain: str, task_ids: Sequence[str], trials: int, seed: int, save_to: Path, concurrency: int,
             config: Path, name: str | None = None, sets: Sequence[str] = ()) -> list[str]:
    """`python -m jevharness.bench.suites.tau2 run` arguments of an E22 run (registration (c) / (h) settings)."""
    args = ["--jev-config", str(config), "--jev-nl-judge", JUDGE_LLM]
    if name:
        args += ["--jev-name", name]
    for kv in sets:
        args += ["--jev-set", kv]
    args += ["--domain", domain, "--agent-llm", AGENT_LLM, "--user-llm", USER_LLM,
             "--agent-llm-args", json.dumps(LLM_ARGS), "--user-llm-args", json.dumps(LLM_ARGS),
             "--num-trials", str(trials), "--seed", str(seed), "--max-concurrency", str(concurrency),
             "--save-to", str(save_to), "--auto-resume", "--task-ids", *task_ids]
    return args


def _read_text(p: Path) -> str:
    return gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()


def _sidecar(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for line in _read_text(p).splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _results(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(_read_text(p))
    except ValueError:
        return {}


def run_spend(run_dir: Path) -> dict[str, Any]:
    """Logged spend of one tau2 run dir (raw or gzipped): the agent's ledger (side-car usage.total) + tau2's user cost
    + the retail judge estimate per simulation."""
    side = _sidecar(run_dir / "jev_decisions.jsonl") or _sidecar(run_dir / "jev_decisions.jsonl.gz")
    res = _results(run_dir / "results.json") or _results(run_dir / "results.json.gz")
    agent = sum(float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0.0) for r in side)
    sims = res.get("simulations") or []
    user = sum(float(s.get("user_cost") or 0.0) for s in sims)
    if not user:   # litellm has no local price for the user model: price the user messages' logged tokens
        from jevharness.accounting import default_prices
        pr = default_prices().get(USER_LLM.split("/", 1)[1])
        for s_ in sims:
            for m in s_.get("messages") or []:
                u = m.get("usage") or {}
                if m.get("role") == "user" and u and pr is not None:
                    user += pr.cost(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), 0)
    domain = ((res.get("info") or {}).get("environment_info") or {}).get("domain_name") or (
        "retail" if "retail" in str(run_dir) else "airline")
    n = max(len(side), len(sims))
    judge = JUDGE_USD_PER_RETAIL_SIM * n if domain == "retail" else 0.0
    return {"dir": str(run_dir), "domain": domain, "episodes": len(side), "simulations": len(sims),
            "agent_usd": agent, "user_usd": user, "judge_usd": judge, "total_usd": agent + user + judge}


def all_spend() -> dict[str, Any]:
    """Every E22 run this container knows of: local harvest chunks (raw or gz), smokes, and fetched host runs."""
    dirs = sorted({p.parent for pat in ("**/jev_decisions.jsonl", "**/jev_decisions.jsonl.gz")
                   for base in (SCRATCH, RES / "runs") if base.exists() for p in base.glob(pat)})
    rows = [run_spend(d) for d in dirs]
    host = RES / "box" / "spend_host.json"
    extra = json.loads(host.read_text()).get("total_usd", 0.0) if host.exists() else 0.0
    return {"runs": rows, "local_usd": sum(r["total_usd"] for r in rows), "host_usd": extra,
            "total_usd": sum(r["total_usd"] for r in rows) + extra}


def cmd_spend(a: argparse.Namespace) -> int:
    sp = all_spend()
    if a.json:
        print(json.dumps({k: v for k, v in sp.items()}, default=str))
        return 0
    for r in sp["runs"]:
        print(f"{r['total_usd']:8.3f}  agent {r['agent_usd']:.3f} user {r['user_usd']:.3f} judge {r['judge_usd']:.3f} "
              f"n {r['episodes']}  {r['dir']}")
    print(f"total {sp['total_usd']:.3f} USD (local {sp['local_usd']:.3f}, host {sp['host_usd']:.3f}; cap {CAP_USD})")
    return 0


def _aws(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
    env["AWS_PAGER"] = ""
    return subprocess.run(["aws", "--region", "us-east-1", *args], env=env, capture_output=True, text=True)


def _compress(run_dir: Path) -> None:
    for name in ("results.json", "jev_decisions.jsonl", "jev_attempts.json"):
        f = run_dir / name
        if f.exists():
            (run_dir / (name + ".gz")).write_bytes(gzip.compress(f.read_bytes(), 6))
            f.unlink()
    for d in (run_dir / "jev_errored",):
        if d.exists():
            shutil.make_archive(str(d), "gztar", str(d))
            shutil.rmtree(d)


def _run_with_watchdog(args: list[str], log: Path, cap: float) -> int:
    """Run the tau2 launcher; a watchdog stops it when the logged E22 spend passes the cap (registration (j))."""
    cmd = [sys.executable, "-m", "jevharness.bench.suites.tau2", "run", *args]
    with open(log, "a") as lf:
        lf.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd[:4])} ... ({len(args)} args)\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=_env_for_run(), stdout=lf, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(30)
            tot = all_spend()["total_usd"]
            if tot > cap:
                proc.terminate()
                lf.write(f"WATCHDOG: logged spend {tot:.2f} > cap {cap}: stopped\n")
                proc.wait(60)
                return 99
        return proc.returncode


def cmd_harvest(a: argparse.Namespace) -> int:
    sp = load_split()["domains"][a.domain]
    if a.smoke:
        tasks, trials, seed, root = sp["fit"][:2], 1, SMOKE_SEED, SCRATCH / "smoke" / a.domain
    else:
        tasks, trials, seed, root = sorted(sp["fit"] + sp["dev"], key=natural), a.trials, HARVEST_SEED, \
            SCRATCH / "harvest" / a.domain
    assert not set(tasks) & set(sp["test"]), "a test task in the harvest"
    chunks = [tasks[i:i + a.chunk] for i in range(0, len(tasks), a.chunk)]
    root.mkdir(parents=True, exist_ok=True)
    log = root / "harvest.log"
    for ci, chunk in enumerate(chunks):
        d = root / f"c{ci:02d}"
        if (d / "results.json.gz").exists():
            print(f"{a.domain} chunk {ci}: done")
            continue
        d.mkdir(parents=True, exist_ok=True)
        args = run_args(a.domain, chunk, trials, seed, d, a.concurrency, CFG / "harvest.yaml")
        t0 = time.time()
        rc = _run_with_watchdog(args, log, a.cap)
        if rc == 99:
            return 99
        # registration (c): a simulation that ended in an infrastructure or agent error is re-run once
        rr = subprocess.run([sys.executable, "-m", "jevharness.bench.suites.tau2", "rerun-errored", str(d),
                             "--max-attempts", "2", "--", *args], cwd=str(ROOT), env=_env_for_run(),
                            capture_output=True, text=True)
        with open(log, "a") as lf:
            lf.write(f"rerun-errored rc {rr.returncode}: {rr.stdout[-600:]} {rr.stderr[-600:]}\n")
        sp_run = run_spend(d)
        print(f"{a.domain} chunk {ci}: rc {rc}, {sp_run['episodes']} episodes, {sp_run['total_usd']:.2f} USD, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        _compress(d)
        if not a.no_s3:
            up = _aws("s3", "sync", str(d), f"{S3}/{'smoke' if a.smoke else 'harvest'}/{a.domain}/c{ci:02d}",
                      "--only-show-errors")
            if up.returncode:
                print(f"S3 upload failed (kept locally): {up.stderr[-300:]}")
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in all_spend().items() if k != "runs"}))
    return 0


# ================================================================================================ evaluation runs (box)
OVERLOAD = re.compile(r"\b503\b|overload|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand|rate.?limit|\b429\b", re.I)


def probe(n: int = 6) -> dict[str, Any]:
    """E14's probe round: six one-word Gemini 3.5 Flash calls on the direct API (litellm, the agent's route)."""
    import litellm

    ok, lat = 0, []
    for _ in range(n):
        t0 = time.time()
        try:
            r = litellm.completion(model=AGENT_LLM, messages=[{"role": "user", "content": "Reply with one word: ok"}],
                                   temperature=0, thinkingConfig={"thinkingLevel": "minimal"}, timeout=60,
                                   num_retries=0)
            ok += int(bool(r.choices[0].message.content))
        except Exception:  # noqa: BLE001
            pass
        lat.append(round(time.time() - t0, 2))
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ok": ok, "n": n, "latency_s": lat}


def wait_clean(log: Path, every_s: int = 300, max_min: float = 240) -> bool:
    """Two consecutive clean probe rounds (6 of 6), probing every 5 minutes after a failed round."""
    clean, t0 = 0, time.time()
    while (time.time() - t0) / 60 < max_min:
        r = probe()
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
        clean = clean + 1 if r["ok"] == r["n"] else 0
        if clean >= 2:
            return True
        if clean == 0:
            time.sleep(every_s)
    return False


def _overload_errors(run_dir: Path) -> int:
    n = 0
    for r in _sidecar(run_dir / "jev_decisions.jsonl"):
        if r.get("error") and OVERLOAD.search(str(r["error"])):
            n += 1
    return n


def arm_args(arm: str, domain: str, tasks: Sequence[str], trials: int, seed: int, out: Path,
             concurrency: int) -> list[str]:
    if arm == "llm_only":
        return run_args(domain, tasks, trials, seed, out, concurrency, CFG / "llm_only.yaml", name="llm_only")
    if arm == "e22_final":
        return run_args(domain, tasks, trials, seed, out, concurrency, CFG / "final.yaml", name="e22_final")
    if arm == "e22_tau05":
        return run_args(domain, tasks, trials, seed, out, concurrency, CFG / "final.yaml", name="e22_tau05",
                        sets=["tool_select.tau=0.5", "tool_params.tau=0.5", "tool_select.always_defer=false",
                              "tool_params.always_defer=false"])
    raise SystemExit(f"unknown arm {arm}")


def cmd_run_arm(a: argparse.Namespace) -> int:
    """Registration (h): one arm on one domain's test tasks (or the dev smoke), with E14's pause-and-resume rule, the
    spend watchdog and one re-run of errored simulations after two clean probe rounds."""
    sp = load_split()["domains"][a.domain]
    if a.smoke:
        tasks, trials, seed = sp["dev"][:2], 1, SMOKE_SEED
    else:
        tasks, trials, seed = sp["test"], a.trials, EVAL_SEED
    assert a.smoke or not set(tasks) & set(sp["fit"] + sp["dev"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = arm_args(a.arm, a.domain, tasks, trials, seed, out, a.concurrency)
    log, plog = out / "run.log", Path(a.probe_log)
    cmd = [sys.executable, "-m", "jevharness.bench.suites.tau2", "run", *args]
    with open(out / "window.jsonl", "a") as f:
        f.write(json.dumps({"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "t": time.time(),
                            "arm": a.arm, "domain": a.domain, "tasks": len(tasks), "trials": trials}) + "\n")
    pauses = 0
    while True:
        base_err = _overload_errors(out)
        with open(log, "a") as lf:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=_env_for_run(), stdout=lf, stderr=subprocess.STDOUT)
            paused = False
            while proc.poll() is None:
                time.sleep(30)
                if spend_now(a.before) > a.cap:
                    proc.terminate()
                    proc.wait(120)
                    lf.write("WATCHDOG: spend cap reached\n")
                    return 99
                if _overload_errors(out) - base_err >= 3:
                    proc.terminate()
                    proc.wait(120)
                    lf.write(f"PAUSE {time.strftime('%H:%M:%S')}: API overload errors; probing\n")
                    paused = True
            if not paused:
                break
        pauses += 1
        if not wait_clean(plog):
            break
    # one re-run of errored simulations after two clean probe rounds (E14's rule)
    wait_clean(plog, max_min=60)
    rr = subprocess.run([sys.executable, "-m", "jevharness.bench.suites.tau2", "rerun-errored", str(out),
                         "--max-attempts", "2", "--", *args], cwd=str(ROOT), env=_env_for_run(), capture_output=True,
                        text=True)
    with open(out / "window.jsonl", "a") as f:
        f.write(json.dumps({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "t": time.time(),
                            "arm": a.arm, "domain": a.domain, "pauses": pauses, "rerun_rc": rr.returncode,
                            "rerun": rr.stderr[-1500:]}) + "\n")
    print(json.dumps(run_spend(out)))
    return 0


def spend_now(before: float) -> float:
    return before + sum(run_spend(d)["total_usd"] for d in {p.parent for p in SCRATCH.glob("**/jev_decisions.jsonl*")})


def cmd_probe(a: argparse.Namespace) -> int:
    r = probe()
    with open(a.log, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r))
    return 0 if r["ok"] == r["n"] else 1


# ================================================================================================ labels (d) and records
from jevharness.harness.tools import OTHER  # noqa: E402  -- the cascade's "none of these" option label


def domain_info(domain: str) -> dict[str, Any]:
    """Tool kinds and descriptions (as the adapter renders them for System-1) and the tasks' reference actions."""
    from tau2.runner import build_environment, get_tasks

    from jevharness.bench.suites import tau2 as jt
    from jevharness.harness.tools import tool_desc

    env = build_environment(domain)
    tools = env.get_tools()
    return {"kinds": {t.name: jt._tool_kind(t) for t in tools},
            "descs": {t.name: tool_desc(jt._s1_schema(t.openai_schema)) for t in tools},
            "actions": {str(t.id): [a for a in ((t.evaluation_criteria.actions or []) if t.evaluation_criteria else [])
                                    if a.requestor == "assistant"] for t in get_tasks(domain)}}


def _toolcall(tc: Mapping[str, Any]) -> Any:
    from tau2.data_model.message import ToolCall
    return ToolCall(id=str(tc.get("id") or "x"), name=tc["name"], arguments=tc.get("arguments") or {})


def _pending(actions: list[Any], kinds: Mapping[str, Any], before: list[Mapping[str, Any]]) -> list[Any]:
    """The adapter's ``_Reference.pending`` on the JSON messages before a turn: reference actions not yet executed
    successfully (tau2's compare_with_tool_call), minus READ actions before the last executed one."""
    failed = {m.get("id") for m in before if m.get("role") == "tool" and m.get("error")}
    used = [False] * len(actions)
    for m in before:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            if tc.get("id") in failed:
                continue
            call = _toolcall(tc)
            for i, a in enumerate(actions):
                if not used[i] and a.compare_with_tool_call(call):
                    used[i] = True
                    break
    last = max((i for i, u in enumerate(used) if u), default=-1)
    return [a for i, (a, u) in enumerate(zip(actions, used)) if not u and not (i < last and kinds.get(a.name) == "read")]


def _wire(text: str, options: Sequence[str], descs: Sequence[str] | None) -> dict[str, Any]:
    from jevharness.decision.jev import question_to_wire
    from jevharness.decision.types import Choice
    return question_to_wire(Choice(text, list(options), list(descs) if descs is not None else None))


def episode_rows(domain: str, sim: Mapping[str, Any], rec: Mapping[str, Any], info: Mapping[str, Any],
                 stats: Counter) -> list[dict[str, Any]]:
    """Registration (d): the labelled questions of one harvest episode (tool choice + its arguments)."""
    from jevharness.bench.suites.tau2 import RESPOND

    tid = str(sim.get("task_id"))
    reward = float((sim.get("reward_info") or {}).get("reward") or 0.0)
    ok = abs(reward - 1.0) <= 1e-6
    term = str(sim.get("termination_reason"))
    if rec.get("error") or term in ("agent_error", "user_error", "infrastructure_error", "TerminationReason.AGENT_ERROR"):
        stats["episode_errored"] += 1
        return []
    stats["episode_success" if ok else "episode_failed"] += 1
    msgs = list(sim.get("messages") or [])
    tool_out = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
    by_step: dict[int, int] = {}
    for i, m in enumerate(msgs):
        j = ((m.get("raw_data") or {}).get("jev") or {}) if m.get("role") == "assistant" else {}
        if j.get("episode") == rec.get("episode") and j.get("step") is not None:
            by_step[int(j["step"])] = i
    cost_by_step: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for c in rec.get("calls") or []:
        if c.get("component") == "s2" and not (c.get("meta") or {}).get("shadow"):
            cost_by_step[int(c.get("step") or 0)][0] += float(c.get("cost_usd") or 0.0)
            cost_by_step[int(c.get("step") or 0)][1] += float(c.get("latency_ms") or 0.0)
    actions = info["actions"].get(tid, [])
    rows: list[dict[str, Any]] = []
    trial = sim.get("trial", 0)
    for st in rec.get("steps") or []:
        step = int(st["step"])
        ts = next((d for d in st.get("decisions") or [] if d.get("dp") == "tool_select" and d.get("mode") == "off"), None)
        if ts is None or not ts.get("state") or step not in by_step:
            stats["state_unjoined"] += 1
            continue
        stats["states"] += 1
        msg = msgs[by_step[step]]
        calls = msg.get("tool_calls") or []
        choice = ts.get("s2_value")
        first_name = calls[0]["name"] if calls else RESPOND
        if choice != first_name:
            stats["state_choice_mismatch"] += 1
            continue
        errs = [tc for tc in calls if (tool_out.get(tc.get("id")) or {}).get("error")]
        if errs:
            stats["drop_tool_error"] += 1
            continue
        if ok:
            bad_write = [tc for tc in calls if info["kinds"].get(tc["name"]) == "write"
                         and not any(a.compare_with_tool_call(_toolcall(tc)) for a in actions)]
            if bad_write:
                stats["drop_unmatched_write"] += 1
                continue
            source = "success"
        else:
            if not calls:
                stats["drop_failed_message"] += 1
                continue
            pend = _pending(actions, info["kinds"], msgs[:by_step[step]])
            if not any(a.compare_with_tool_call(_toolcall(calls[0])) for a in pend):
                stats["drop_failed_unmatched"] += 1
                continue
            source = "failed_exact"
        stats[f"kept_{source}"] += 1
        ref = ts.get("gold")
        if ref is not None and ref != RESPOND:
            stats["ref_tool_states"] += 1
            stats["ref_tool_agree"] += int(ref == choice)
        opts = list(ts["options"])
        cost, lat = cost_by_step.get(step, [0.0, 0.0])
        base = {"domain": domain, "task": tid, "trial": trial, "step": step, "source": source, "episode_success": ok,
                "s2_cost_usd": cost, "s2_latency_ms": lat, "episode": rec.get("episode"),
                "call": ({"name": calls[0]["name"], "arguments": calls[0].get("arguments") or {}} if calls else None)}
        rows.append({**base, "id": f"{domain}/{tid}#{trial}/s{step}/ts", "dp": "tool_select", "state": ts["state"],
                     "text": ts["question"], "options": opts,
                     "descs": [info["descs"].get(o, "") for o in opts[:-1]] + [_respond_desc()],
                     "label": choice, "ref": ref})
        for d in st.get("decisions") or []:
            if d.get("dp") != "tool_params" or d.get("mode") != "off":
                continue
            o = list(d["options"])
            v = d.get("s2_value")
            lab = v if v in o else OTHER
            if lab not in o:
                stats["arg_no_other"] += 1
                continue
            stats["arg_rows"] += 1
            stats["arg_other"] += int(lab == OTHER)
            rows.append({**base, "id": f"{domain}/{tid}#{trial}/s{step}/tp/{(d.get('gold_aux') or {}).get('param')}",
                         "dp": "tool_params", "state": ts["state"], "text": d["question"], "options": o, "descs": None,
                         "label": lab, "ref": d.get("gold"), "tool": (d.get("gold_aux") or {}).get("tool")})
    return rows


def _respond_desc() -> str:
    from jevharness.bench.suites.tau2 import RESPOND_DESC
    return RESPOND_DESC


def harvest_runs(domain: str) -> list[Path]:
    return sorted(p for p in (SCRATCH / "harvest" / domain).glob("c*") if (p / "results.json.gz").exists())


def to_record(r: Mapping[str, Any]) -> dict[str, Any]:
    q = _wire(r["text"], r["options"], r["descs"])
    keys = list(q["criteria"])
    assert r["label"] in keys, (r["id"], r["label"])
    return {"id": r["id"], "state": r["state"], "questions": {"q0": q},
            "gold": {"q0": {"label": r["label"], "probabilities": {k: float(k == r["label"]) for k in keys}}},
            "meta": {k: r.get(k) for k in ("domain", "task", "trial", "step", "source", "dp", "tool")}}


def cmd_build(a: argparse.Namespace) -> int:
    sp = load_split()["domains"]
    out = SCRATCH / "data"
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Counter] = {d: Counter() for d in DOMAINS}
    rows_by_split: dict[str, list[dict[str, Any]]] = {"fit": [], "dev": []}
    for d in DOMAINS:
        info = domain_info(d)
        fit, dev, test = set(sp[d]["fit"]), set(sp[d]["dev"]), set(sp[d]["test"])
        for run in harvest_runs(d):
            res = _results(run / "results.json.gz")
            recs = _sidecar(run / "jev_decisions.jsonl.gz")
            by_ep = {r["episode"]: r for r in recs}
            by_key = {(str(r.get("task_id")), r.get("seed")): r for r in recs}
            for sim in res.get("simulations") or []:
                tid = str(sim.get("task_id"))
                if tid in test:
                    raise SystemExit(f"test task {d}/{tid} in the harvest")
                ep = None
                for m in sim.get("messages") or []:
                    ep = ((m.get("raw_data") or {}).get("jev") or {}).get("episode") if m.get("role") == "assistant" else None
                    if ep:
                        break
                rec = by_ep.get(ep or "") or by_key.get((tid, sim.get("seed")))
                if rec is None:
                    stats[d]["episode_no_sidecar"] += 1
                    continue
                stats[d]["episodes"] += 1
                which = "fit" if tid in fit else "dev" if tid in dev else None
                if which is None:
                    raise SystemExit(f"{d}/{tid} is in no train split")
                rows_by_split[which] += episode_rows(d, sim, rec, info, stats[d])
    info_out: dict[str, Any] = {"what": "E22 records (registration (d)); built by scripts/explore_e22_tau2.py build",
                                "stats": {d: dict(c) for d, c in stats.items()}, "files": {}}
    for which, rows in rows_by_split.items():
        seen: set[str] = set()
        recs, kept_rows = [], []
        dup = 0
        for r in rows:
            q = _wire(r["text"], r["options"], r["descs"])
            k = sha256_bytes((r["state"] + "\x00" + json.dumps(q, sort_keys=True)).encode())
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            recs.append(to_record(r))
            kept_rows.append(r)
        f = out / f"{which}.jsonl.gz"
        with gzip.open(f, "wt") as fh:
            for rr in recs:
                fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
        by = Counter((r["meta"]["domain"], r["meta"]["dp"]) for r in recs)
        info_out["files"][f.name] = {"records": len(recs), "duplicates_dropped": dup,
                                     "by_domain_dp": {f"{x}/{y}": n for (x, y), n in sorted(by.items())},
                                     "by_source": dict(Counter(r["meta"]["source"] for r in recs)),
                                     "labels_tool_select": dict(Counter(r["gold"]["q0"]["label"] for r in recs
                                                                        if r["meta"]["dp"] == "tool_select")),
                                     "other_share_args": round(sum(r["gold"]["q0"]["label"] == OTHER for r in recs
                                                                   if r["meta"]["dp"] == "tool_params") /
                                                               max(1, sum(r["meta"]["dp"] == "tool_params"
                                                                          for r in recs)), 4),
                                     "max_options": max((len(r["questions"]["q0"]["criteria"]) for r in recs), default=0),
                                     "state_chars_max": max((len(r["state"]) for r in recs), default=0)}
        if which == "dev":   # the replay input of (f): the labelled dev questions with the LLM's logged call cost
            g = out / "dev_replay.jsonl.gz"
            with gzip.open(g, "wt") as fh:
                for r in kept_rows:
                    fh.write(json.dumps({k: r.get(k) for k in ("id", "domain", "task", "trial", "step", "dp", "state",
                                                                "text", "options", "descs", "label", "ref", "source",
                                                                "episode_success", "s2_cost_usd", "s2_latency_ms",
                                                                "tool")}, ensure_ascii=False) + "\n")
    for f in sorted(out.glob("*.jsonl.gz")):
        info_out.setdefault("sha256", {})[f.name] = sha256_file(f)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "data_build.json").write_text(json.dumps(info_out, indent=1) + "\n")
    (RES / "data_sha256.json").write_text(json.dumps({"sha256": info_out["sha256"]}, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info_out.items() if k != "files"}, indent=1)[:3000])
    for k, v in info_out["files"].items():
        print(k, {x: y for x, y in v.items() if x not in ("labels_tool_select",)})
    return 0


# ================================================================================================ label audit ($0)
def cmd_audit(a: argparse.Namespace) -> int:
    """Amendment 'E22: the task as the unit of analysis, the label wording and a label audit' (c): 200 labelled
    tool-choice states (100 per label source, random.Random(0) per stratum over the states in record order) checked
    against the task's expected actions; plus label coverage. No model call."""
    import random

    from jevharness.bench.suites.tau2 import RESPOND

    sp = load_split()["domains"]
    rows: list[dict[str, Any]] = []
    for d in DOMAINS:
        info = domain_info(d)
        test = set(sp[d]["test"])
        for run in harvest_runs(d):
            res = _results(run / "results.json.gz")
            recs = _sidecar(run / "jev_decisions.jsonl.gz")
            by_ep = {r["episode"]: r for r in recs}
            for sim in res.get("simulations") or []:
                assert str(sim.get("task_id")) not in test
                ep = next((((m.get("raw_data") or {}).get("jev") or {}).get("episode") for m in sim.get("messages") or []
                           if m.get("role") == "assistant" and ((m.get("raw_data") or {}).get("jev") or {}).get("episode")),
                          None)
                if ep not in by_ep:
                    continue
                for r in episode_rows(d, sim, by_ep[ep], info, Counter()):
                    if r["dp"] == "tool_select":
                        r["_actions"] = info["actions"].get(r["task"], [])
                        rows.append(r)
    out: dict[str, Any] = {"what": "E22 label audit (amendment (c)); no model call", "strata": {}}
    for src in ("success", "failed_exact"):
        pool = [r for r in rows if r["source"] == src]
        sample = random.Random(0).sample(pool, min(a.per_stratum, len(pool)))
        c = Counter()
        for r in sample:
            c["n"] += 1
            if r["label"] == RESPOND:
                c["respond_labels"] += 1
            else:
                c["tool_labels"] += 1
                names = {x.name for x in r["_actions"]}
                if r["label"] in names:
                    c["tool_in_expected_list"] += 1
                if r["call"] and any(x.compare_with_tool_call(_toolcall(r["call"])) for x in r["_actions"]):
                    c["call_matches_expected_action"] += 1
            if r.get("ref") not in (None, RESPOND):
                c["ref_names_tool"] += 1
                c["ref_agree"] += int(r["ref"] == r["label"])
            elif r.get("ref") == RESPOND:
                c["ref_respond"] += 1
                c["ref_respond_agree"] += int(r["label"] == RESPOND)
        c = dict(c)
        c["pool"] = len(pool)
        tl = max(1, c.get("tool_labels", 0))
        c["share_tool_labels_matching_expected"] = round(100 * c.get("call_matches_expected_action", 0) / tl, 1)
        listed = max(1, c.get("tool_in_expected_list", 0))
        c["share_listed_tool_labels_matching_expected"] = round(100 * c.get("call_matches_expected_action", 0) / listed, 1)
        c["share_ref_agree_where_ref_names_tool"] = round(100 * c.get("ref_agree", 0) / max(1, c.get("ref_names_tool", 0)), 1)
        out["strata"][src] = c
    bd = json.loads((RES / "data_build.json").read_text())
    states = sum(v.get("states", 0) for v in bd["stats"].values())
    kept = sum(v.get("kept_success", 0) + v.get("kept_failed_exact", 0) for v in bd["stats"].values())
    out["coverage"] = {"tool_choice_states": states, "labelled": kept, "share_pct": round(100 * kept / states, 1),
                       "drops": {k: sum(v.get(k, 0) for v in bd["stats"].values()) for k in
                                 ("drop_tool_error", "drop_unmatched_write", "drop_failed_unmatched", "drop_failed_message",
                                  "state_choice_mismatch", "state_unjoined")}}
    (RES / "label_audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


# ================================================================================================ replay + fit (box)
def _read_jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in _read_text(p).splitlines() if x.strip()]


def cmd_replay(a: argparse.Namespace) -> int:
    """Registration (f): the served model answers every labelled dev question once, through the `jev` backend and a
    Ledger, exactly as the harness asks it (no gold on the wire)."""
    from concurrent.futures import ThreadPoolExecutor

    from jevharness.accounting import Ledger
    from jevharness.decision import make_backend
    from jevharness.decision.types import Choice

    rows = _read_jl(Path(a.inp))
    be = make_backend({"type": "jev", "name": a.name, "base_url": a.base_url, "model": a.name, "api_key": "",
                       "timeout": 120.0})

    def one(r: Mapping[str, Any]) -> dict[str, Any]:
        q = Choice(r["text"], list(r["options"]), list(r["descs"]) if r.get("descs") else None, decision_point=r["dp"])
        led = Ledger()
        t0 = time.perf_counter()
        ans = be.ask(r["state"], q, ledger=led)
        ms = (time.perf_counter() - t0) * 1000
        return {"id": r["id"], "s1_value": ans.value, "s1_conf": ans.confidence, "s1_dist": ans.distribution,
                "s1_cost_usd": sum(c.cost_usd for c in led.records), "s1_latency_ms": ms}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(one, rows))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    acc = Counter()
    for r, o in zip(rows, out):
        acc[(r["dp"], "n")] += 1
        acc[(r["dp"], "ok")] += int(o["s1_value"] == r["label"])
    print(json.dumps({"rows": len(rows), "minutes": round((time.time() - t0) / 60, 2),
                      "acc": {dp: round(acc[(dp, "ok")] / acc[(dp, "n")], 4) for dp in ("tool_select", "tool_params")
                              if acc[(dp, "n")]}}))
    return 0


def shadow_arm(rows: Sequence[Mapping[str, Any]], answers: Mapping[str, Mapping[str, Any]], arm: str) -> list[dict]:
    """The dev questions with System-1's replayed answers as a calibration shadow arm (one trajectory per episode):
    the LLM's logged choice = the label, so ``s2_correct`` is True; a tool-choice answer of respond_to_user is left out
    (registration (f): never acted on in E22's harness)."""
    from jevharness.bench.suites.tau2 import RESPOND

    eps: dict[str, dict[str, Any]] = {}
    for r in rows:
        an = answers.get(r["id"])
        if an is None:
            continue
        if r["dp"] == "tool_select" and an["s1_value"] == RESPOND:
            continue
        key = f"{r['domain']}/{r['task']}#{r['trial']}"
        t = eps.setdefault(key, {"task_id": key, "config": arm, "backend": "e22", "llm": AGENT_LLM, "steps": {},
                                 "result": {"success": bool(r["episode_success"])}, "usage": {}, "calls": [],
                                 "meta": {"e22": {"domain": r["domain"], "task": r["task"], "trial": r["trial"]}}})
        st = t["steps"].setdefault(int(r["step"]), {"step": int(r["step"]), "decisions": []})
        st["decisions"].append({"dp": r["dp"], "mode": "shadow", "question": r["text"], "options": r["options"],
                                "s1_value": an["s1_value"], "s1_conf": an["s1_conf"], "s1_dist": an["s1_dist"],
                                "deferred": True, "s2_value": r["label"], "final_value": r["label"], "gold": r["label"],
                                "shadow": True, "s1_cost_usd": float(an.get("s1_cost_usd") or 0.0),
                                "s1_latency_ms": float(an.get("s1_latency_ms") or 0.0), "s2_correct": True,
                                "s2_cost_usd": float(r["s2_cost_usd"]), "s2_latency_ms": float(r["s2_latency_ms"]),
                                "gold_status": "ok"})
    out = []
    for t in eps.values():
        t["steps"] = [t["steps"][k] for k in sorted(t["steps"])]
        out.append(t)
    return out


def write_profile(tau: Mapping[str, float], fits: Mapping[str, str], path: Path, base_url: str, name: str) -> dict:
    """configs/explore/e22_tau2/final.yaml: registration (g) with the fitted tau-hat (always_defer when the fit accepts
    no level)."""
    import yaml

    def point(dp: str) -> dict[str, Any]:
        t = tau[dp]
        if t is None or not math.isfinite(t):
            return {"mode": "cascade", "tau": 1.0, "always_defer": True}
        return {"mode": "cascade", "tau": float(t)}

    cfg = {"name": "e22_final", "tool_select": point("tool_select"), "tool_params": point("tool_params"),
           "s1_act": "complete", "log_state": False,
           "s1": {"type": "jev", "name": name, "base_url": base_url, "model": name, "api_key": "", "timeout": 120.0}}
    head = ("# Exploratory row E22 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E22' (f)-(g)): the final harness on\n"
            "# tau^2 -- tool choice and arguments in CASCADE at the fitted tau-hat, s1_act: complete, the trained 0.8B head\n"
            f"# served on the host's GPU. Written by scripts/explore_e22_tau2.py fit from {', '.join(fits.values())}.\n")
    path.write_text(head + yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def cmd_fit(a: argparse.Namespace) -> int:
    rows = _read_jl(Path(a.inp))
    answers = {o["id"]: o for o in _read_jl(Path(a.answers))}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arm = "e22_dev_shadow"
    trajs = shadow_arm(rows, answers, arm)
    with open(out / f"{arm}.jsonl", "w") as fh:
        for t in trajs:
            fh.write(json.dumps(t) + "\n")
    tau: dict[str, float | None] = {}
    fits: dict[str, str] = {}
    for dp in ("tool_select", "tool_params"):
        f = out / f"tauhat_{dp}.json"
        cmd = [sys.executable, "-m", "jevharness.bench.threshold_theory", str(out), "--calibration-arm", "--arm", arm,
               "--dp", dp, "--task-value", "1.0", "--lam-per-hour", "0", "--v-fallback", "W", "--format", "json",
               "--out", str(f)]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        if r.returncode:
            print(r.stderr[-2000:])
            return r.returncode
        res = json.loads(f.read_text())
        t = res.get("tau_hat")
        tau[dp] = float(t) if isinstance(t, (int, float)) else (float(t) if isinstance(t, str) and t not in ("inf", "nan")
                                                                 else None)
        fits[dp] = str(f.name)
        print(dp, "tau_hat", tau[dp], "rule", (res.get("tau_rule") or {}).get("rule"), "n_paired", res.get("n_paired"))
    prof = write_profile(tau, fits, Path(a.profile), a.base_url, a.name)
    (out / "fit_summary.json").write_text(json.dumps({"tau_hat": tau, "profile": str(a.profile), "config": prof,
                                                      "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                                     indent=1) + "\n")
    print(json.dumps(prof))
    return 0


# ================================================================================================ readout
TARIFF = {"g5.2xlarge": 1.212, "g6e.2xlarge": 2.24208}
E9_TARIFF = 2.24208
ARMS = ("e22_final", "e22_tau05")
TEX_OUT = ROOT / "paper/generated/e22_tau2.tex"


def load_arm(run_dir: Path) -> dict[str, dict[str, Any]]:
    """(task, trial) -> episode facts of one tau2 run dir (results.json + side-car)."""
    from jevharness.bench.suites import tau2 as jt

    res = run_dir / "results.json"
    if not res.exists():
        return {}
    trajs = jt.convert_results(res, run_dir / jt.SIDECAR_FILE)
    out: dict[str, dict[str, Any]] = {}
    for t in trajs:
        d = t.to_dict()
        tau = d["meta"]["tau2"]
        key = f"{tau['domain']}/{tau['task_id']}#{tau['trial']}"
        steps = d.get("steps") or []
        dec = [x for s_ in steps for x in s_.get("decisions") or []]
        ts = [x for x in dec if x.get("dp") == "tool_select"]
        tp = [x for x in dec if x.get("dp") == "tool_params"]
        emitted = [s_ for s_ in steps if int(s_.get("llm_calls") or 0) == 0 and s_.get("tool") not in (None, "respond_to_user")]
        usage = d.get("usage") or {}
        out[key] = {"task": f"{tau['domain']}/{tau['task_id']}", "domain": tau["domain"], "trial": tau["trial"],
                    "success": bool(d["result"].get("success")), "error": d.get("error"),
                    "cost": float(((usage.get("total") or {}).get("cost_usd")) or 0.0),
                    "s1_cost": float(((usage.get("s1") or {}).get("cost_usd")) or 0.0),
                    "llm_calls": int(((usage.get("s2") or {}).get("calls")) or 0),
                    "s1_calls": int(((usage.get("s1") or {}).get("calls")) or 0),
                    "s1_ms": float(((usage.get("s1") or {}).get("latency_ms")) or 0.0),
                    "wall_s": float(d.get("wall_ms") or 0.0) / 1000, "turns": len(steps), "emitted": len(emitted),
                    "emitted_ref_ok": sum(1 for s_ in emitted if s_.get("oracle_tool") == s_.get("tool")),
                    "emitted_ref_named": sum(1 for s_ in emitted if s_.get("oracle_tool") not in (None, "respond_to_user")),
                    "ts_n": len(ts), "ts_accept": sum(1 for x in ts if x.get("s1_value") is not None and not x.get("deferred")),
                    "ts_conf_ok": sum(1 for x in ts if (x.get("gold_aux") or {}).get("not_acted") or
                                      (x.get("s1_value") is not None and not x.get("deferred"))),
                    "ts_respond_not_acted": sum(1 for x in ts if (x.get("gold_aux") or {}).get("not_acted") == "respond"),
                    "ts_incomplete": sum(1 for x in ts if (x.get("gold_aux") or {}).get("not_acted") == "incomplete"),
                    "tp_n": len(tp), "tp_taken": sum(1 for x in tp if x.get("final_value") is not None and not x.get("deferred")),
                    "emitted_tools": [s_.get("tool") for s_ in emitted],
                    "user_cost": float(tau.get("user_cost") or 0.0)}
    return out


def _pct(x: float | None) -> float | None:
    return None if x is None else 100.0 * float(x)


def compare(ref: Mapping[str, Mapping], arm: Mapping[str, Mapping], q_per_ep: float) -> dict[str, Any]:
    """Paired comparison of an arm with llm_only, the TASK as the unit (amendment 'E22: the task as the unit of
    analysis', (a)): the trials completed in both arms are averaged within a task; tasks are resampled together with
    their trials; the NI verdict is the certified ordered p on per-task majority pass. The (task, trial) pair numbers
    are kept as descriptive rows."""
    import numpy as np

    from explore_common import newcombe
    from jevharness.bench import metrics as M

    keys = sorted(k for k in set(ref) & set(arm) if not ref[k]["error"] and not arm[k]["error"])
    all_tasks = {v["task"] for v in list(ref.values()) + list(arm.values())}
    n = len(keys)
    if not n:
        return {"n": 0}
    by_task: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_task[ref[k]["task"]].append(k)
    tasks = sorted(by_task)

    def tmean(src: Mapping[str, Mapping], field: str) -> np.ndarray:
        return np.array([np.mean([float(src[k][field]) for k in by_task[t]]) for t in tasks])

    ta, tb = tmean(arm, "success"), tmean(ref, "success")
    d = ta - tb
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(tasks), size=(10_000, len(tasks)))
    boot = d[idx].mean(1)
    # the verdict: majority pass per task (> half of the paired trials), certified ordered p (A6.21) at 5 points
    trials = np.array([len(by_task[t]) for t in tasks])
    maj_a = np.array([sum(arm[k]["success"] for k in by_task[t]) > len(by_task[t]) / 2 for t in tasks])
    maj_b = np.array([sum(ref[k]["success"] for k in by_task[t]) > len(by_task[t]) / 2 for t in tasks])
    mw, ml = int((maj_a & ~maj_b).sum()), int((maj_b & ~maj_a).sum())
    ni_maj = M.ni_certified_p(mw, ml, len(tasks), 0.05)
    # descriptive: the (task, trial) pairs as if independent (E22 (i) as registered first)
    ya = [arm[k]["success"] for k in keys]
    yb = [ref[k]["success"] for k in keys]
    wins = sum(a and not b for a, b in zip(ya, yb))
    losses = sum(b and not a for a, b in zip(ya, yb))
    nc = newcombe(ya, yb)
    ni_pair = M.ni_certified_p(wins, losses, n, 0.05)
    # cost and time: per-task means of the per-episode values
    c_ref, c_api = tmean(ref, "cost"), tmean(arm, "cost")
    c_arm = c_api + q_per_ep
    sboot = 1.0 - c_arm[idx].sum(1) / c_ref[idx].sum(1)
    w_ref, w_arm = tmean(ref, "wall_s"), tmean(arm, "wall_s")
    turns = sum(arm[k]["turns"] for k in keys)
    tsn = sum(arm[k]["ts_n"] for k in keys)
    tpn = sum(arm[k]["tp_n"] for k in keys)
    emitted = sum(arm[k]["emitted"] for k in keys)
    ref_named = sum(arm[k]["emitted_ref_named"] for k in keys)
    q = lambda a, x: float(np.quantile(a, x))  # noqa: E731
    return {"n_tasks": len(tasks), "tasks_left_out": len(all_tasks) - len(tasks), "n_pairs": n,
            "trials_per_task": {str(k): int(v) for k, v in zip(*np.unique(trials, return_counts=True))},
            "success_ref": _pct(float(tb.mean())), "success_arm": _pct(float(ta.mean())),
            "delta_pt": _pct(float(d.mean())), "delta_ci_pt": [_pct(q(boot, 0.025)), _pct(q(boot, 0.975))],
            "delta_lcb95_pt": _pct(q(boot, 0.05)), "boot_ni": bool(q(boot, 0.05) > -0.05),
            "ni": bool(ni_maj["p"] <= 0.05), "ni_p": ni_maj["p"], "margin_pt": 5.0,
            "majority": {"wins": mw, "losses": ml, "success_ref": _pct(float(maj_b.mean())),
                         "success_arm": _pct(float(maj_a.mean())), "p": ni_maj["p"]},
            "pairs": {"n": n, "wins": wins, "losses": losses, "newcombe_pt": nc["ci_pt"], "delta_pt": nc["diff_pt"],
                      "p": ni_pair["p"], "ni": bool(ni_pair["p"] <= 0.05)},
            "cost_ref": float(c_ref.mean()), "cost_arm_api": float(c_api.mean()), "q_per_ep": q_per_ep,
            "cost_arm": float(c_arm.mean()),
            "saving_pct": _pct(1 - c_arm.mean() / c_ref.mean()),
            "saving_ci_pct": [_pct(q(sboot, 0.025)), _pct(q(sboot, 0.975))],
            "saving_api_pct": _pct(1 - c_api.mean() / c_ref.mean()),
            "wall_ref_mean": float(w_ref.mean()), "wall_arm_mean": float(w_arm.mean()),
            "wall_ref_median": float(np.median([ref[k]["wall_s"] for k in keys])),
            "wall_arm_median": float(np.median([arm[k]["wall_s"] for k in keys])),
            "time_saving_pct": _pct(1 - float(w_arm.mean()) / float(w_ref.mean())),
            "llm_calls_ref": float(tmean(ref, "llm_calls").mean()), "llm_calls_arm": float(tmean(arm, "llm_calls").mean()),
            "s1_calls_arm": float(tmean(arm, "s1_calls").mean()), "s1_ms_per_ep": float(tmean(arm, "s1_ms").mean()),
            "llm_free_turns_pct": _pct(emitted / turns) if turns else None,
            "ts_accept_pct": _pct(sum(arm[k]["ts_conf_ok"] for k in keys) / tsn) if tsn else None,
            "ts_acted_pct": _pct(emitted / tsn) if tsn else None,
            "ts_respond_not_acted": sum(arm[k]["ts_respond_not_acted"] for k in keys),
            "ts_incomplete": sum(arm[k]["ts_incomplete"] for k in keys),
            "tp_taken_pct": _pct(sum(arm[k]["tp_taken"] for k in keys) / tpn) if tpn else None,
            "emitted_calls": emitted,
            "emitted_ref_agree_pct": _pct(sum(arm[k]["emitted_ref_ok"] for k in keys) / ref_named) if ref_named else None,
            "user_cost_ref": float(tmean(ref, "user_cost").mean())}


def cmd_summarize(a: argparse.Namespace) -> int:
    import numpy as np

    from jevharness.bench.suites import tau2 as jt

    box = RES / "box"
    runs = Path(a.runs)
    windows = [json.loads(x) for x in (box / "eval_windows.jsonl").read_text().splitlines()] \
        if (box / "eval_windows.jsonl").exists() else []
    host = json.loads((box / "host.json").read_text()) if (box / "host.json").exists() else {}
    price = TARIFF.get(host.get("instance_type", ""), TARIFF["g5.2xlarge"])
    plan = json.loads((box / "eval_plan.json").read_text()) if (box / "eval_plan.json").exists() else {}
    data: dict[str, dict[str, dict]] = {}
    for d in DOMAINS:
        for arm in ("llm_only", *ARMS):
            rd = runs / "eval" / d / arm
            if rd.exists():
                data.setdefault(arm, {}).update(load_arm(rd))
    # GPU tariff: the host's $/h x the evaluation windows of the GPU arms / their completed episodes (E11's convention)
    win_h = 0.0
    starts = {w["domain"]: w["t"] for w in windows if "start_utc" in w}
    for w in windows:
        if "end_utc" in w and w["domain"] in starts:
            win_h += (w["t"] - starts[w["domain"]]) / 3600
    gpu_eps = sum(1 for arm in ARMS for v in (data.get(arm) or {}).values() if not v["error"])
    q = price * win_h / gpu_eps if gpu_eps else 0.0
    q_e9 = E9_TARIFF * win_h / gpu_eps if gpu_eps else 0.0
    out: dict[str, Any] = {"what": "E22 readout (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E22' (i)); exploratory, "
                           "descriptive", "host": host, "price_per_h": price, "eval_window_h": win_h,
                           "gpu_episodes": gpu_eps, "q_per_episode": q, "q_e9_per_episode": q_e9, "plan": plan,
                           "scopes": {}}
    ref_all = data.get("llm_only") or {}
    for scope in ("pooled", *DOMAINS):
        sel = (lambda k: True) if scope == "pooled" else (lambda k, s=scope: k.startswith(s + "/"))
        ref = {k: v for k, v in ref_all.items() if sel(k)}
        sc: dict[str, Any] = {"llm_only": {"episodes": len(ref), "errored": sum(1 for v in ref.values() if v["error"]),
                                           "pass_k": jt.pass_hat_k([{"meta": {"tau2": {"domain": v["domain"],
                                                                                       "task_id": v["task"].split("/")[1]}},
                                                                     "result": {"success": v["success"]}}
                                                                    for v in ref.values() if not v["error"]])}}
        for arm in ARMS:
            av = {k: v for k, v in (data.get(arm) or {}).items() if sel(k)}
            if not av:
                continue
            c = compare(ref, av, q)
            c["q0_saving_pct"] = c.get("saving_api_pct")
            c["e9_saving_pct"] = _pct(1 - (c["cost_arm_api"] + q_e9) / c["cost_ref"]) if c.get("n_tasks") else None
            c["episodes"] = len(av)
            c["errored"] = sum(1 for v in av.values() if v["error"])
            c["pass_k"] = jt.pass_hat_k([{"meta": {"tau2": {"domain": v["domain"], "task_id": v["task"].split("/")[1]}},
                                          "result": {"success": v["success"]}} for v in av.values() if not v["error"]])
            sc[arm] = c
        out["scopes"][scope] = sc
    # post hoc diagnostics (descriptive, $0): what the model emitted, where the losses fell, the run-to-run noise
    diag: dict[str, Any] = {}
    kinds: dict[str, Any] = {}
    for d in DOMAINS:
        kinds.update(domain_info(d)["kinds"])
    for arm in ARMS:
        av = data.get(arm) or {}
        if not av:
            continue
        tools = [t for v in av.values() for t in v["emitted_tools"]]
        per_a, per_r, emit = defaultdict(list), defaultdict(list), defaultdict(int)
        for k, v in av.items():
            if k in ref_all and not v["error"] and not ref_all[k]["error"]:
                per_a[v["task"]].append(v["success"])
                per_r[v["task"]].append(ref_all[k]["success"])
                emit[v["task"]] += v["emitted"]
        lost = [t for t in per_a if sum(per_a[t]) < sum(per_r[t])]
        lost_trials = sum(sum(per_r[t]) - sum(per_a[t]) for t in lost)
        lost_noemit = [t for t in lost if emit[t] == 0]
        diag[arm] = {"emitted_calls": len(tools), "emitted_writes": sum(kinds.get(t) == "write" for t in tools),
                     "emitted_by_tool": dict(Counter(tools).most_common()), "tasks_lost": len(lost),
                     "tasks_lost_without_any_emitted_call": len(lost_noemit),
                     "trial_successes_lost_in_those_tasks": sum(sum(per_r[t]) - sum(per_a[t]) for t in lost_noemit),
                     "trial_successes_lost_all_lost_tasks": lost_trials,
                     "s1_seconds_per_episode": float(np.mean([v["s1_ms"] for v in av.values()])) / 1000}
    hv: dict[str, dict[int, bool]] = defaultdict(dict)
    for d in DOMAINS:
        for run in harvest_runs(d):
            for s_ in (_results(run / "results.json.gz").get("simulations") or []):
                hv[f"{d}/{s_['task_id']}"][int(s_["trial"])] = abs(float((s_.get("reward_info") or {}).get("reward") or 0)
                                                                  - 1.0) <= 1e-6
    if hv:
        import random as _r
        rng_ = _r.Random(0)
        tl = sorted(hv)
        dd = []
        for _ in range(2000):
            sel = rng_.sample(tl, min(60, len(tl)))
            vals = []
            for t in sel:
                tr = sorted(hv[t])
                rng_.shuffle(tr)
                h = len(tr) // 2
                vals.append(np.mean([hv[t][i] for i in tr[:h]]) - np.mean([hv[t][i] for i in tr[h:]]))
            dd.append(100 * float(np.mean(vals)))
        diag["harvest_split_half"] = {"what": "LLM-only on the train tasks: per-task mean over 4 random trials minus the "
                                      "other 4, averaged over 60 random tasks (2,000 draws, seed 0)",
                                      "sd_pt": float(np.std(dd)), "q025_pt": float(np.percentile(dd, 2.5)),
                                      "q975_pt": float(np.percentile(dd, 97.5))}
    out["diag"] = diag
    # offline: training record, fit, dev replay accuracy
    tr = box / "train" / "train_record.json"
    out["train"] = json.loads(tr.read_text()) if tr.exists() else None
    fits = {}
    for dp in ("tool_select", "tool_params"):
        f = box / "cal" / f"tauhat_{dp}.json"
        if f.exists():
            r = json.loads(f.read_text())
            fits[dp] = {"tau_hat": r.get("tau_hat"), "rule": (r.get("tau_rule") or {}).get("rule"),
                        "n_paired": r.get("n_paired"), "delta_c": (r.get("primary") or {}).get("delta_c"),
                        "V": (r.get("primary") or {}).get("V")}
    out["fit"] = fits
    rp, an = SCRATCH / "data" / "dev_replay.jsonl.gz", box / "cal" / "answers_dev.jsonl.gz"
    if rp.exists() and an.exists():
        rows = {r["id"]: r for r in _read_jl(rp)}
        acc: dict[str, list] = defaultdict(list)
        cov: dict[str, list] = defaultdict(list)
        for o in _read_jl(an):
            r = rows.get(o["id"])
            if r is None:
                continue
            ok = o["s1_value"] == r["label"]
            acc[r["dp"]].append(ok)
            t = (fits.get(r["dp"]) or {}).get("tau_hat")
            if isinstance(t, (int, float)) and math.isfinite(t):
                take = o["s1_conf"] >= t and o["s1_value"] != OTHER and not (r["dp"] == "tool_select" and
                                                                             o["s1_value"] == "respond_to_user")
                cov[r["dp"]].append((take, ok))
        out["dev_replay"] = {dp: {"n": len(v), "acc": _pct(float(np.mean(v)))} for dp, v in acc.items()}
        for dp, v in cov.items():
            tk = [ok for take, ok in v if take]
            out["dev_replay"][dp].update({"coverage_at_tau": _pct(len(tk) / len(v)), "accepted": len(tk),
                                          "accepted_wrong": sum(1 for ok in tk if not ok)})
    bd = RES / "data_build.json"
    out["data"] = json.loads(bd.read_text()) if bd.exists() else None
    la = RES / "label_audit.json"
    out["audit"] = json.loads(la.read_text()) if la.exists() else None
    sp = box / "spend_host.json"
    hg = RES / "harvest_gate.json"
    harvest = sum(run_spend(p_)["total_usd"] for p_ in harvest_runs("retail") + harvest_runs("airline"))
    smoke = json.loads(hg.read_text())["smoke"]["usd"] if hg.exists() else 0.0
    out["spend"] = {"host_usd": json.loads(sp.read_text())["total_usd"] if sp.exists() else None,
                    "local_usd": harvest + smoke, "harvest_usd": harvest, "harvest_smoke_usd": smoke}
    hf = box / "host_final.json"
    out["host_final"] = json.loads(hf.read_text()) if hf.exists() else None
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    (RES / "summary.md").write_text(summary_md(out))
    write_e22_tex(out)
    print((RES / "summary.md").read_text())
    return 0


def summary_md(o: Mapping[str, Any]) -> str:
    L = ["# E22: the training recipe on tau^2-bench (exploratory, descriptive)", "",
         "docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E22' and its amendment (the task as the unit of analysis). "
         "Test tasks of tau2-bench's official split; success = mean over tasks of the per-task mean over trials; "
         "Δ with a paired task bootstrap (trials stay with their task); NI verdict at 5 points by the certified ordered p "
         "(A6.21) on per-task majority pass. Retail NL judge: Gemini 3.5 Flash (not gpt-4.1).", "",
         f"GPU tariff q = {o['price_per_h']} $/h x {o['eval_window_h']:.2f} h evaluation window / {o['gpu_episodes']} GPU-arm "
         f"episodes = {o['q_per_episode']:.5f} $ per episode (E9's tariff: {o['q_e9_per_episode']:.5f}).", "",
         "| scope | arm | tasks | LLM-only % | arm % | Δ pt [task bootstrap] | NI p (tasks) | NI | saving % (GPU) | API-only % | "
         "time s (ref / arm) | LLM-free turns % | LLM calls / ep (ref / arm) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for scope, sc in o["scopes"].items():
        for arm in ARMS:
            c = sc.get(arm)
            if not c or not c.get("n_tasks"):
                continue
            L.append(f"| {scope} | {arm} | {c['n_tasks']} | {c['success_ref']:.1f} | {c['success_arm']:.1f} | "
                     f"{c['delta_pt']:+.1f} [{c['delta_ci_pt'][0]:+.1f}, {c['delta_ci_pt'][1]:+.1f}] | {c['ni_p']:.3f} | "
                     f"{'yes' if c['ni'] else 'not shown'} | {c['saving_pct']:.1f} | {c['saving_api_pct']:.1f} | "
                     f"{c['wall_ref_mean']:.1f} / {c['wall_arm_mean']:.1f} | {c['llm_free_turns_pct']:.1f} | "
                     f"{c['llm_calls_ref']:.2f} / {c['llm_calls_arm']:.2f} |")
    L += ["", "```", json.dumps({k: o.get(k) for k in ("fit", "dev_replay", "spend", "plan")}, indent=1, default=str), "```", ""]
    return "\n".join(L)


def write_e22_tex(o: Mapping[str, Any]) -> None:
    from explore_common import fmt, git_head

    K: dict[str, str] = {}

    def sg(v: Any, digits: int = 1) -> str:   # a signed number as the other generated files write it (+4.3, -65.5)
        return r"\tbd" if v is None or not math.isfinite(float(v)) else f"{float(v):+.{digits}f}"

    def pl(v: Any, digits: int = 1) -> str:   # a plain number, a leading "-" kept (math via \resm)
        return r"\tbd" if v is None or not math.isfinite(float(v)) else f"{float(v):.{digits}f}"

    for scope, sc in o["scopes"].items():
        lo = sc.get("llm_only") or {}
        for arm in ARMS:
            c = sc.get(arm)
            if not c or not c.get("n_tasks"):
                continue
            p = f"e22/{scope}/{arm}"
            K[f"{p}/n"] = fmt(c["n_tasks"])
            K[f"{p}/pairs"] = fmt(c["n_pairs"])
            K[f"e22/{scope}/llm_only/success"] = fmt(c["success_ref"])
            K[f"{p}/success"] = fmt(c["success_arm"])
            K[f"{p}/delta"] = sg(c["delta_pt"])
            K[f"{p}/lo"], K[f"{p}/hi"] = sg(c["delta_ci_pt"][0]), sg(c["delta_ci_pt"][1])
            K[f"{p}/lcb"] = sg(c["delta_lcb95_pt"])
            K[f"{p}/p"] = fmt(c["ni_p"], 3)
            K[f"{p}/ni"] = "yes" if c["ni"] else "no"
            K[f"{p}/wins"], K[f"{p}/losses"] = fmt(c["majority"]["wins"]), fmt(c["majority"]["losses"])
            K[f"{p}/pair/p"] = fmt(c["pairs"]["p"], 3)
            K[f"{p}/pair/ni"] = "yes" if c["pairs"]["ni"] else "no"
            K[f"{p}/saving"] = pl(c["saving_pct"])
            K[f"{p}/saving_lo"], K[f"{p}/saving_hi"] = pl(c["saving_ci_pct"][0]), pl(c["saving_ci_pct"][1])
            K[f"{p}/saving_api"] = pl(c["saving_api_pct"])
            K[f"{p}/saving_e9"] = pl(c["e9_saving_pct"])
            K[f"{p}/wall_mean"] = fmt(c["wall_arm_mean"])
            K[f"e22/{scope}/llm_only/wall_mean"] = fmt(c["wall_ref_mean"])
            K[f"{p}/time_saving"] = pl(c["time_saving_pct"])
            K[f"{p}/cost_k"] = fmt(1000 * c["cost_arm"], 2)
            K[f"e22/{scope}/llm_only/cost_k"] = fmt(1000 * c["cost_ref"], 2)
            K[f"{p}/llmfree"] = fmt(c["llm_free_turns_pct"])
            K[f"{p}/calls"] = fmt(c["llm_calls_arm"], 2)
            K[f"e22/{scope}/llm_only/calls"] = fmt(c["llm_calls_ref"], 2)
            K[f"{p}/tpcov"] = fmt(c["tp_taken_pct"])
            K[f"{p}/refagree"] = fmt(c["emitted_ref_agree_pct"])
            pk = c.get("pass_k") or {}
            if pk:
                kmax = max(int(x.split("^")[1]) for x in pk)
                K[f"{p}/passk"] = fmt(100 * pk[f"pass^{kmax}"])
                K[f"e22/{scope}/llm_only/passk"] = fmt(100 * ((lo.get("pass_k") or {}).get(f"pass^{kmax}") or float("nan")))
                K["e22/k"] = fmt(kmax)
    for dp, f in (o.get("fit") or {}).items():
        t = f.get("tau_hat")
        K[f"e22/tau/{dp}"] = fmt(float(t), 4) if isinstance(t, (int, float)) and math.isfinite(t) else "always defers"
    for dp, r in (o.get("dev_replay") or {}).items():
        K[f"e22/dev/{dp}/acc"] = fmt(r["acc"])
        if "coverage_at_tau" in r:
            K[f"e22/dev/{dp}/cov"] = fmt(r["coverage_at_tau"])
            K[f"e22/dev/{dp}/wrong"] = fmt(r["accepted_wrong"])
    tr = o.get("train") or {}
    if tr.get("base_dev"):
        K["e22/train/base_acc"] = fmt(100 * tr["base_dev"]["accuracy"])
    if tr.get("dev_calibrated"):
        K["e22/train/dev_acc"] = fmt(100 * tr["dev_calibrated"]["accuracy"])
    if tr.get("selected_epoch"):
        K["e22/train/epochs"] = fmt(int(tr["selected_epoch"]))
    dat = o.get("data") or {}
    files = dat.get("files") or {}
    if files:
        K["e22/data/fit_records"] = fmt(int(files.get("fit.jsonl.gz", {}).get("records", 0)))
        K["e22/data/dev_records"] = fmt(int(files.get("dev.jsonl.gz", {}).get("records", 0)))
        st = dat.get("stats") or {}
        K["e22/data/episodes"] = fmt(int(sum(v.get("episodes", 0) for v in st.values())))
        K["e22/data/states"] = fmt(int(sum(v.get("states", 0) for v in st.values())))
    dg = o.get("diag") or {}
    for arm in ARMS:
        g = dg.get(arm)
        if g:
            K[f"e22/diag/{arm}/emitted"] = fmt(int(g["emitted_calls"]))
            K[f"e22/diag/{arm}/writes"] = fmt(int(g["emitted_writes"]))
            K[f"e22/diag/{arm}/tasks_lost"] = fmt(int(g["tasks_lost"]))
            K[f"e22/diag/{arm}/tasks_lost_noemit"] = fmt(int(g["tasks_lost_without_any_emitted_call"]))
            K[f"e22/diag/{arm}/s1_s"] = fmt(float(g["s1_seconds_per_episode"]))
    if dg.get("harvest_split_half"):
        K["e22/diag/noise_sd"] = fmt(float(dg["harvest_split_half"]["sd_pt"]))
    au = o.get("audit") or {}
    if au:
        K["e22/audit/coverage"] = fmt(float(au["coverage"]["share_pct"]))
        K["e22/audit/labelled"] = fmt(int(au["coverage"]["labelled"]))
        for tag, src in (("succ", "success"), ("fail", "failed_exact")):
            st = au["strata"][src]
            K[f"e22/audit/{tag}/n"] = fmt(int(st["n"]))
            K[f"e22/audit/{tag}/toollabels"] = fmt(int(st.get("tool_labels", 0)))
            K[f"e22/audit/{tag}/listed"] = fmt(int(st.get("tool_in_expected_list", 0)))
            K[f"e22/audit/{tag}/match"] = fmt(int(st.get("call_matches_expected_action", 0)))
            K[f"e22/audit/{tag}/match_listed_pct"] = fmt(float(st["share_listed_tool_labels_matching_expected"]))
            K[f"e22/audit/{tag}/ref_n"] = fmt(int(st.get("ref_names_tool", 0)))
            K[f"e22/audit/{tag}/ref_agree_pct"] = fmt(float(st["share_ref_agree_where_ref_names_tool"]))
    K["e22/q_cents"] = fmt(100 * o["q_per_episode"], 2)
    sp = o.get("spend") or {}
    if sp.get("host_usd") is not None:
        K["e22/spend/gemini"] = fmt(sp["host_usd"] + sp["local_usd"], 2)
    hf = o.get("host_final") or {}
    if hf.get("hours_used") is not None:
        K["e22/gpu/hours"] = fmt(float(hf["hours_used"]), 1)
    head = ["% AUTO-GENERATED by scripts/explore_e22_tau2.py summarize -- do not edit by hand.",
            "% E22: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E22').",
            f"% generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} at git {(git_head() or '?')[:12]}",
            r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(K.items())]
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(head + body + [r"\makeatother", ""]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split").set_defaults(fn=cmd_split)
    h = sub.add_parser("harvest")
    h.add_argument("--domain", required=True, choices=DOMAINS)
    h.add_argument("--smoke", action="store_true")
    h.add_argument("--trials", type=int, default=HARVEST_TRIALS)
    h.add_argument("--chunk", type=int, default=12, help="tasks per tau2 invocation")
    h.add_argument("--concurrency", type=int, default=10)
    h.add_argument("--cap", type=float, default=CAP_USD)
    h.add_argument("--no-s3", action="store_true")
    h.set_defaults(fn=cmd_harvest)
    spp = sub.add_parser("spend")
    spp.add_argument("--json", action="store_true")
    spp.set_defaults(fn=cmd_spend)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    au = sub.add_parser("audit")
    au.add_argument("--per-stratum", type=int, default=100)
    au.set_defaults(fn=cmd_audit)
    ra = sub.add_parser("run-arm")
    ra.add_argument("--arm", required=True, choices=["llm_only", "e22_final", "e22_tau05"])
    ra.add_argument("--domain", required=True, choices=DOMAINS)
    ra.add_argument("--smoke", action="store_true", help="two dev tasks, one trial (never reported)")
    ra.add_argument("--trials", type=int, default=4)
    ra.add_argument("--concurrency", type=int, default=6)
    ra.add_argument("--out", required=True)
    ra.add_argument("--probe-log", required=True)
    ra.add_argument("--cap", type=float, default=CAP_USD)
    ra.add_argument("--before", type=float, default=0.0, help="E22 spend before this host's runs")
    ra.set_defaults(fn=cmd_run_arm)
    sm = sub.add_parser("summarize")
    sm.add_argument("--runs", default=str(SCRATCH / "hostruns"), help="the host's runs dir fetched from S3")
    sm.set_defaults(fn=cmd_summarize)
    pr = sub.add_parser("probe")
    pr.add_argument("--log", required=True)
    pr.set_defaults(fn=cmd_probe)
    r = sub.add_parser("replay")
    r.add_argument("--inp", required=True, help="dev_replay.jsonl.gz")
    r.add_argument("--base-url", required=True)
    r.add_argument("--name", default="e22-tau2-0.8b")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=4)
    r.set_defaults(fn=cmd_replay)
    f = sub.add_parser("fit")
    f.add_argument("--inp", required=True, help="dev_replay.jsonl.gz")
    f.add_argument("--answers", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--profile", default=str(CFG / "final.yaml"))
    f.add_argument("--base-url", default="http://127.0.0.1:8090")
    f.add_argument("--name", default="e22-tau2-0.8b")
    f.set_defaults(fn=cmd_fit)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
