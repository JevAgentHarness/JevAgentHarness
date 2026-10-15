#!/usr/bin/env python3
r"""E23 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E23: the E11 checkpoint in
ALFWorld's unseen rooms (valid_unseen)"): E11's frozen checkpoint (e11-alfworld-4b) and E11's fitted threshold, with
no refit and no training, live on ALFWorld's 134 valid_unseen games (four rooms that occur in none of E11's training
or calibration games), beside a re-run of LLM-only in the same pool. No test; outside every Holm table; revises nothing.

    python scripts/explore_e23_alfworld_unseen.py eval-list [--check]                 # the 134 games (list + sha256)
    python scripts/explore_e23_alfworld_unseen.py extract-unseen --json-zip J --twpddl-zip T   # box: release -> data
    python scripts/explore_e23_alfworld_unseen.py verify-ckpt --dir CK                # E11's sha256 record
    python scripts/explore_e23_alfworld_unseen.py probe | spend | watch --pid PID | errored --run R | project
    python scripts/explore_e23_alfworld_unseen.py summarize [--runs runs] [--box results/explore/e23_alfworld_unseen/box]

Readout: E11's (E6a's ``arm_block`` on the paired complete case against ``llm_only``; success, Delta with the paired
task bootstrap B = 10,000 and the Newcombe interval, wins / losses, cost as logged plus E11's GPU tariff, wall time,
steps, calls, coverage and accuracy against the single gold action), the non-inferiority outcome by the rule Table 1
applies to E11's row (``scripts/hard_ni.py``: ``metrics.ni_certified_p(wins, losses, n, 0.03)``, non-inferior iff
p <= 0.05; Tango 95 % interval), the accepted actions that differ from the gold, success by task type, and E11's
valid_seen numbers beside each row (read from E11's summary, not recomputed)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402
import explore_e6a_alfworld as E6A  # noqa: E402
import explore_e11_alfworld as E11  # noqa: E402

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.metrics import ni_certified_p, tango_interval  # noqa: E402
from jevharness.bench.suites.alfworld import TASK_TYPES  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
CFG_DIR = ROOT / "configs/explore/e23_alfworld_unseen"
OUT_DIR = ROOT / "results/explore/e23_alfworld_unseen"
BOX_DIR = OUT_DIR / "box"
TEX = ROOT / "paper/generated/e23_alfworld_unseen.tex"
EVAL_LIST = CFG_DIR / "eval_games.json"
EVAL_CFG = CFG_DIR / "eval.yaml"
SMOKE_CFG = CFG_DIR / "smoke.yaml"
PROBE_LOG = OUT_DIR / "api_probe_log.txt"
EVAL_DATA = ROOT / "data/raw/alfworld"
SPLIT = "valid_unseen"
SPLIT_DIR = "json_2.1.1/valid_unseen/"
RUN_SMOKE, RUN_EVAL = "e23_alfworld_smoke", "e23_alfworld_unseen_eval"
REF = "llm_only"
ARMS = dict(E11.ARMS)                       # E11's arm names: cascade at E11's tau-hat, replace
GPU_ARMS = tuple(ARMS.values())
LABEL = "exploratory (E23)"
SERVED = E11.SERVED                         # e11-alfworld-4b
E11_TAU = E11.TAU_FILE                      # E11's fitted tau-hat (no refit)
E11_SUMMARY = E11.OUT_DIR / "summary.json"
E11_CKPT_RECORD = E11.BOX_DIR / "adapter_sha256.txt"
CKPT_FILES = ("adapter_model.safetensors", "adapter_config.json", "calibration.json")
REGISTERED_LIST_SHA = "3c46e12b31bfcb6259289cbd48cc7b3ef2183dda9761d995e568fda38a83d820"
N_EVAL, MIN_EVAL_GAMES, SAFETY, CAP_USD = 134, 100, 1.5, 15.0
NI_MARGIN, NI_ALPHA = 0.03, 0.05            # scripts/hard_ni.py (Table 1's rule for E11's row)
UNSEEN_SCENES = ("10", "219", "308", "424")
# games played before E23 in development (pilot P3b, docs/PILOT_RESULTS.md; one also in runs/dev/l3_smoke)
PILOT_IDS = (
    "alfworld/valid_unseen/pick_and_place_simple-SaltShaker-None-Cabinet-10/trial_T20190906_191501_563086",
    "alfworld/valid_unseen/pick_heat_then_place_in_recep-Potato-None-GarbageCan-10/trial_T20190907_161853_945788",
    "alfworld/valid_unseen/pick_clean_then_place_in_recep-Bowl-None-Cabinet-10/trial_T20190909_061232_368489",
    "alfworld/valid_unseen/look_at_obj_in_light-Book-None-DeskLamp-308/trial_T20190908_020029_636862",
    "alfworld/valid_unseen/pick_cool_then_place_in_recep-Bread-None-CounterTop-10/trial_T20190908_091835_825830")
# on-demand $/h, us-east-1 (E11's prices; g5.2xlarge as E22 records it)
HOST_USD_PER_HOUR = {**E11.HOST_USD_PER_HOUR, "g5.2xlarge": 1.212}
OVERLOAD = ("503", "overload", "high demand", "unavailable", "resource_exhausted", "resource exhausted", "429")
PROBE_N = 6
FLASH = {"provider": "gemini", "model": "gemini-3.5-flash", "temperature": 0, "reasoning_effort": "minimal"}
TYPE_SHORT = {"pick_and_place_simple": "simple", "look_at_obj_in_light": "light",
              "pick_clean_then_place_in_recep": "clean", "pick_heat_then_place_in_recep": "heat",
              "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "two"}
NAN = float("nan")
_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def scene_of(task_id: str) -> str:
    return task_id.split("/")[2].rsplit("-", 1)[1]


# ============================================================================== games
def eval_ids(data_dir: Path = EVAL_DATA) -> list[str]:
    """The valid_unseen task ids in the run's order (``load_alfworld(..., stratify=True)``, as E6a / E11)."""
    from jevharness.bench.suites.alfworld import load_alfworld
    return [f().task_id for f in load_alfworld(SPLIT, data_dir=str(data_dir), stratify=True)]


def list_record(ids: Sequence[str]) -> dict[str, Any]:
    from collections import Counter
    return {"row": LABEL, "split": "valid_unseen (eval_out_of_distribution)",
            "order": "stratified (task types round-robin over the sorted game files; load_alfworld stratify=True)",
            "n": len(ids), "by_type": dict(Counter(t.split("/")[2].split("-")[0] for t in ids)),
            "scenes": dict(sorted(Counter(scene_of(t) for t in ids).items())),
            "pilot_contact": [t for t in ids if t in PILOT_IDS], "task_ids": list(ids),
            "list_sha256": _sha("\n".join(ids))}


def cmd_eval_list(a: argparse.Namespace) -> int:
    ids = eval_ids(Path(a.data_dir))
    sha = _sha("\n".join(ids))
    if a.check:
        rec = json.loads(EVAL_LIST.read_text())
        ok = ids == rec["task_ids"] and sha == rec["list_sha256"] == REGISTERED_LIST_SHA
        print(f"valid_unseen list on disk: {len(ids)} games, sha256 {sha[:16]} "
              f"({'matches' if ok else 'DIFFERS FROM'} the registered list)")
        return 0 if ok else 1
    rec = list_record(ids)
    if rec["list_sha256"] != REGISTERED_LIST_SHA:
        raise SystemExit(f"list sha256 {rec['list_sha256'][:16]} is not the registered {REGISTERED_LIST_SHA[:16]}")
    EVAL_LIST.parent.mkdir(parents=True, exist_ok=True)
    EVAL_LIST.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {C.rel(EVAL_LIST)}: {len(ids)} games (sha256 {sha[:16]})")
    return 0


def extract_unseen(json_zip: Path, twpddl_zip: Path, root: Path) -> int:
    """Every valid_unseen game directory of the release (traj_data.json + game.tw-pddl) under ``root``."""
    zj, zt = zipfile.ZipFile(json_zip), zipfile.ZipFile(twpddl_zip)
    tw = {n[: -len("/game.tw-pddl")] for n in zt.namelist() if n.startswith(SPLIT_DIR) and n.endswith("/game.tw-pddl")}
    js = {n[: -len("/traj_data.json")] for n in zj.namelist() if n.startswith(SPLIT_DIR) and n.endswith("/traj_data.json")}
    n = 0
    for d in sorted(tw & js):
        parts = d[len(SPLIT_DIR):].split("/")
        if len(parts) != 2 or not all(_SAFE.match(p) for p in parts):
            continue
        out = root / SPLIT_DIR / parts[0] / parts[1]
        out.mkdir(parents=True, exist_ok=True)
        (out / "traj_data.json").write_bytes(zj.read(d + "/traj_data.json"))
        (out / "game.tw-pddl").write_bytes(zt.read(d + "/game.tw-pddl"))
        n += 1
    return n


def cmd_extract_unseen(a: argparse.Namespace) -> int:
    rec6 = json.loads(E6A.TRAIN_LIST.read_text())
    shas = {k: C.sha256_file(Path(p)) for k, p in (("json", a.json_zip), ("twpddl", a.twpddl_zip))}
    bad = [k for k, s in shas.items() if s != rec6["archives"][k]["sha256"]]
    if bad:
        raise SystemExit(f"release archive sha256 differs from E6a's record for {bad}")
    root = Path(a.data_dir)
    n = extract_unseen(Path(a.json_zip), Path(a.twpddl_zip), root)
    ids = eval_ids(root)
    sha = _sha("\n".join(ids))
    print(f"extracted {n} valid_unseen game directories; {len(ids)} games pass ALFWorld's filters; list sha256 "
          f"{sha[:16]} ({'matches' if sha == REGISTERED_LIST_SHA else 'DIFFERS FROM'} the registered list)")
    return 0 if sha == REGISTERED_LIST_SHA else 1


def ckpt_record(path: Path = E11_CKPT_RECORD) -> dict[str, str]:
    """basename -> sha256 from E11's record (sha256sum lines)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[Path(parts[1]).name] = parts[0]
    return out


def verify_ckpt(ck: Path, record: Path = E11_CKPT_RECORD) -> dict[str, Any]:
    want = ckpt_record(record)
    got = {f: (C.sha256_file(ck / f) if (ck / f).exists() else None) for f in CKPT_FILES}
    ok = all(got[f] is not None and got[f] == want.get(f) for f in CKPT_FILES)
    return {"dir": str(ck), "record": C.rel(record), "files": {f: {"sha256": got[f], "e11": want.get(f),
                                                                   "match": got[f] == want.get(f)} for f in CKPT_FILES},
            "ok": ok, "at": _now()}


def cmd_verify_ckpt(a: argparse.Namespace) -> int:
    r = verify_ckpt(Path(a.dir))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
    for f, v in r["files"].items():
        print(f"{f}: {(v['sha256'] or 'missing')[:16]} {'==' if v['match'] else '!='} E11 {(v['e11'] or '?')[:16]}")
    print("checkpoint matches E11's record" if r["ok"] else "CHECKPOINT DOES NOT MATCH E11's RECORD")
    return 0 if r["ok"] else 1


# ============================================================================== API probe, spend, watchdog, gate
def cmd_probe(a: argparse.Namespace) -> int:
    """One round: six one-word Flash calls on the direct API through the harness client (its retries included)."""
    from jevharness.accounting import Ledger
    from jevharness.llm import make_llm

    llm = make_llm(dict(FLASH))
    led = Ledger()
    ok, lat, errs = 0, [], []
    for i in range(PROBE_N):
        t0 = time.perf_counter()
        try:
            llm.chat([{"role": "user", "content": f"Reply with one word: ready ({i})"}], ledger=led,
                     decision_point="probe", max_tokens=8)
            ok += 1
        except Exception as e:  # noqa: BLE001 -- every failure counts against the round
            errs.append(f"{type(e).__name__}: {str(e)[:120]}")
        lat.append(time.perf_counter() - t0)
    lat.sort()
    line = (f"{_now()} direct Gemini 3.5 Flash: {ok}/{PROBE_N} ok, median {lat[len(lat) // 2]:.1f} s, max {lat[-1]:.1f} s"
            + (f"; errors: {errs[0]}" if errs else "") + ("  CLEAN" if ok == PROBE_N else "  UNHEALTHY"))
    p = Path(a.log)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# E23: direct Gemini API probes (E14's pause-and-resume rule, registration E23 (f)). One round = "
                     "six one-word Gemini 3.5 Flash calls; start / resume after two consecutive 6/6 rounds.\n")
    with open(p, "a") as f:
        f.write(line.replace("\n", " ") + f"  (logged ${led.summary()['total']['cost_usd']:.5f})\n")
    print(line)
    return 0 if ok == PROBE_N else 1


_PROBE_COST = re.compile(r"\(logged \$([0-9.]+)\)")


def probe_spend(log: Path = PROBE_LOG) -> float:
    if not Path(log).exists():
        return 0.0
    return float(sum(float(m.group(1)) for m in _PROBE_COST.finditer(Path(log).read_text())))


def total_spend(runs: Path = RUNS, log: Path = PROBE_LOG) -> dict[str, Any]:
    """Every record of E23's runs (errored attempts and re-runs included: all were paid) plus the probe calls."""
    per = {name: E6A.run_spend(runs / name) for name in (RUN_SMOKE, RUN_EVAL) if (runs / name).is_dir()}
    tot = {k: sum(v[k] for v in per.values()) for k in ("gemini", "jev", "total", "records")}
    pr = probe_spend(log)
    return {"by_run": per, "probes": pr, **{**tot, "gemini": tot["gemini"] + pr, "total": tot["total"] + pr}}


def cmd_spend(a: argparse.Namespace) -> int:
    s = total_spend(Path(a.runs), Path(a.log))
    print(f"${s['total']:.3f} ({s['records']} records)" if a.short else json.dumps(s, indent=1))
    return 0


def overload_error(t: Mapping[str, Any]) -> bool:
    e = str(t.get("error") or "").lower()
    return bool(e) and any(k in e for k in OVERLOAD)


def _records(run: Path) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        for line in p.read_text().splitlines():
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if isinstance(t, dict) and "task_id" in t:
                out.append((p.stem, t))
    return out


def overload_count(run: Path) -> int:
    return sum(1 for _, t in _records(run) if overload_error(t)) if Path(run).is_dir() else 0


def errored_tasks(run: Path) -> int:
    """Tasks (any arm) whose deciding record (C.load_latest: the latest without an error, else the latest) errored."""
    n = 0
    for p in sorted(Path(run).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        n += sum(1 for t in C.load_latest([p])[0].values() if t.get("error"))
    return n


def cmd_errored(a: argparse.Namespace) -> int:
    print(errored_tasks(Path(a.run)))
    return 0


def _kill(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        time.sleep(10)


def cmd_watch(a: argparse.Namespace) -> int:
    """Spend cap over E23's runs (-> <runs>/e23_STOP.json) and, unless --no-pause, a pause when a new record of the
    watched run errored with 503 / overload (-> <run>/PAUSED_overload.json); the runner's process group is stopped
    either way and every record is kept (registration E23 (f), E14's rule)."""
    runs = Path(a.runs)
    run = runs / a.run
    base = overload_count(run)
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{_now()} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(runs, Path(a.log))
        if s["total"] >= a.cap:
            (runs / "e23_STOP.json").write_text(json.dumps({"at": _now(), "spend": s, "cap": a.cap}, indent=1))
            _kill(a.pid)
            print(f"CAP REACHED: ${s['total']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        if not a.no_pause:
            new = overload_count(run) - base
            if new > 0:
                errs = [{"arm": arm, "task": t["task_id"], "error": str(t.get("error"))[:300]}
                        for arm, t in _records(run) if overload_error(t)][-new:]
                (run / "PAUSED_overload.json").write_text(json.dumps({"at": _now(), "episodes": errs}, indent=1))
                _kill(a.pid)
                print(f"{_now()} PAUSED: {new} overload error(s) in {a.run}; runner stopped", flush=True)
                return 2
        print(f"{_now()} spend ${s['total']:.3f} ({s['records']} records)", flush=True)
        time.sleep(a.every)


def projection(runs: Path = RUNS, n_games: int = N_EVAL, safety: float = SAFETY, cap: float = CAP_USD,
               log: Path = PROBE_LOG) -> dict[str, Any]:
    """Registration (e): the smoke's mean $ per episode per arm, summed over the arms, x the games x ``safety``, plus
    E23's spend so far, against the cap; else the first ``MIN_EVAL_GAMES`` games (same check); else stop."""
    arms = (REF, *ARMS.values())
    sm = {arm: [t for t in E6A._latest(runs / RUN_SMOKE / f"{arm}.jsonl").values() if not t.get("error")]
          for arm in arms}
    per = {arm: (float(np.mean([E6A.episode_spend(t)["total"] for t in ts])) if ts else None) for arm, ts in sm.items()}
    so_far = total_spend(runs, log)["total"]
    ep = sum(v for v in per.values() if v is not None)
    out: dict[str, Any] = {"smoke_per_episode": per, "smoke_n": {k: len(v) for k, v in sm.items()},
                           "spent_so_far": so_far, "safety": safety, "cap": cap, "at": _now()}
    for n in (n_games, MIN_EVAL_GAMES):
        out[f"projected_{n}"] = ep * n * safety
        out[f"fits_{n}"] = so_far + ep * n * safety <= cap
    if not all(sm[a] for a in arms):
        out["decision"] = "stop after the smoke (an arm has no completed smoke episode)"
    else:
        out["decision"] = (f"run all {n_games}" if out[f"fits_{n_games}"] else
                           f"run the first {MIN_EVAL_GAMES}" if out[f"fits_{MIN_EVAL_GAMES}"] else "stop after the smoke")
    return out


def cmd_project(a: argparse.Namespace) -> int:
    p = projection(Path(a.runs), log=Path(a.log))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(p, indent=1))
    return 0


# ============================================================================== readout
def tariff(runs: Path, window: Path) -> dict[str, Any]:
    """E11's convention (E7 / A7.13 (iii)): q = host $/h x evaluation window / completed episodes of the GPU arms."""
    s24 = json.loads(E11.S24_TARIFF.read_text())["q_usd_per_episode"] if E11.S24_TARIFF.exists() else NAN
    if not window.exists():
        return {"q_usd_per_episode": NAN, "state": "no window record", "sensitivity": {"s2_4": s24, "zero": 0.0}}
    w = json.loads(window.read_text())
    it = str(w.get("instance_type") or "")
    if it not in HOST_USD_PER_HOUR:
        raise SystemExit(f"no on-demand price recorded for instance type {it!r}")
    t0 = dt.datetime.fromisoformat(str(w["window_start"]).replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(str(w["window_end"]).replace("Z", "+00:00"))
    hours = (t1 - t0).total_seconds() / 3600
    done = 0
    for arm in GPU_ARMS:
        p = runs / RUN_EVAL / f"{arm}.jsonl"
        if p.exists():
            done += sum(1 for t in E6A._latest(p).values() if not t.get("error"))
    q = HOST_USD_PER_HOUR[it] * hours / done if done and hours > 0 else NAN
    return {"q_usd_per_episode": q, "host_usd_per_hour": HOST_USD_PER_HOUR[it], "instance_type": it,
            "serving_hours": hours, "paused_s": w.get("paused_s"), "episodes": done,
            "window": {"start": w["window_start"], "end": w["window_end"]}, "state": "ok",
            "convention": "E11's (E7 / S2-4 provisioned tariff, A7.13 (iii)): host $/h x evaluation window / completed "
                          "GPU-arm episodes; on-demand price, us-east-1",
            "sensitivity": {"s2_4": s24, "zero": 0.0}}


def ni_block(wins: int, losses: int, n: int, margin: float = NI_MARGIN, alpha: float = NI_ALPHA) -> dict[str, Any]:
    """scripts/hard_ni.py's computation: the A6.21 certified p (rule bb_ordered) and the Tango 95 % interval."""
    r = ni_certified_p(wins, losses, n, margin)
    ci = tango_interval(n - wins - losses, wins, losses, 0)["ci"]
    p = float(r["p"])
    return {"margin": margin, "alpha": alpha, "p": p, "verdict": "non-inferior" if p <= alpha else "not shown",
            "tango_ci_pt": [100.0 * ci[0], 100.0 * ci[1]], "rule": "bb_ordered (A6.21), scripts/hard_ni.py"}


def taken_stats(trajs: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    """Actions taken from the trained model (replace: all; cascade: not deferred) where the gold action is known: the
    number that matched the single gold action and the number that did not (an upper bound of its errors)."""
    recs = [d for t in trajs for d in E6A._action_recs(t)]
    taken = [d for d in recs if (mode == "replace" or not d.get("deferred")) and d.get("s1_value") is not None]
    g = [d for d in taken if d.get("gold") is not None]
    wrong = sum(1 for d in g if M.is_correct(d, d["s1_value"]) is not True)
    conf = [float(d["s1_conf"]) for d in recs if d.get("s1_conf") is not None]
    return {"n_actions": len(recs), "n_taken": len(taken), "n_taken_gold": len(g), "n_wrong": wrong,
            "wrong_share": (wrong / len(g)) if g else None,
            "n_deferred": sum(1 for d in recs if d.get("deferred")),
            "mean_conf": float(np.mean(conf)) if conf else None,
            "share_conf_below_tau": (float(np.mean([c < tau_hat() for c in conf])) if conf else None)}


def tau_hat() -> float:
    return float(json.loads(E11_TAU.read_text())["deploy"]["tau"])


def seen_block() -> dict[str, Any]:
    """E11's valid_seen row, read from E11's summary (not recomputed), and its NI outcome by the same rule."""
    if not E11_SUMMARY.exists():
        return {}
    d = json.loads(E11_SUMMARY.read_text())
    ref = d.get("reference") or {}
    out: dict[str, Any] = {"n": ref.get("n"), "llm_only": {k: ref.get(k) for k in ("success", "cost_usd",
                                                                                  "wall_mean_s", "wall_median_s",
                                                                                  "steps")},
                           "tariff_q": (d.get("tariff") or {}).get("q_usd_per_episode")}
    for mode in ARMS:
        a = (d.get("arms") or {}).get(mode) or {}
        if a.get("status") != "run":
            continue
        dec = a.get("decisions") or {}
        out[mode] = {k: a.get(k) for k in ("n", "success", "delta_pt", "delta_ci_pt", "wins", "losses", "saving",
                                           "saving_ci", "saving_api", "wall_mean_s", "wall_median_s", "steps",
                                           "llm_calls")}
        out[mode].update({"coverage": dec.get("coverage"), "acc": dec.get("acc")})
        out[mode]["ni"] = ni_block(int(a["wins"]), int(a["losses"]), int(a["n"]))
    out["by_task_type"] = d.get("by_task_type")
    return out


def summarize(runs: Path = RUNS, box: Path = BOX_DIR) -> dict[str, Any]:
    ev = json.loads(EVAL_LIST.read_text())
    run = runs / RUN_EVAL
    exp_n = None
    if (run / "experiment.yaml").exists():
        import yaml
        exp = yaml.safe_load((run / "experiment.yaml").read_text())
        exp_n = int((exp.get("suite") or {}).get("n") or ev["n"])
    max_tasks = None
    if (box / "eval_window.json").exists():
        max_tasks = json.loads((box / "eval_window.json").read_text()).get("max_tasks")
    ids = ev["task_ids"][: min(x for x in (exp_n or ev["n"], max_tasks or ev["n"]))]
    tt = {name: E6A._latest(run / f"{name}.jsonl") for name in (REF, *ARMS.values())}
    tf = tariff(runs, box / "eval_window.json")
    ck = json.loads((box / "ckpt_verify.json").read_text()) if (box / "ckpt_verify.json").exists() else {}
    res: dict[str, Any] = {
        "row": LABEL, "label": "descriptive, revises no registered result; E11's frozen checkpoint and threshold, "
                               "no refit, no training",
        "generated": _now(), "git": C.git_head(),
        "eval": {"split": SPLIT, "n_games": len(ids), "list_sha256": _sha("\n".join(ids)),
                 "registered_list_sha256": REGISTERED_LIST_SHA, "scenes": ev.get("scenes"), "run": C.rel(run),
                 "files": {n: {"sha256": C.sha256_file(run / f"{n}.jsonl"),
                               "records": sum(1 for _ in open(run / f"{n}.jsonl"))}
                           for n in tt if (run / f"{n}.jsonl").exists()}},
        "settings": {"agent_llm": "gemini-3.5-flash (temperature 0, reasoning_effort minimal), direct Gemini API",
                     "decision_model": f"{SERVED} (E11's frozen checkpoint; serve_jev)", "tau_hat": tau_hat(),
                     "max_steps": 50, "history_window": 12, "schedule": "task_major, seed 732", "workers": 8},
        "checkpoint": {"ok": ck.get("ok"), "files": {f: (v or {}).get("sha256") for f, v in (ck.get("files") or {}).items()}},
        "tariff": tf, "reference": E6A.ref_block(tt[REF], ids), "arms": {}}
    for mode, name in ARMS.items():
        if not tt[name]:
            res["arms"][mode] = {"arm": name, "status": "not run"}
            continue
        blk = {"arm": name, "status": "run", **E6A.arm_block(tt[name], tt[REF], ids, mode)}
        if not blk.get("n"):
            res["arms"][mode] = blk
            continue
        blk["saving_api"], blk["saving_api_ci"] = blk.get("saving"), blk.get("saving_ci")
        q = tf.get("q_usd_per_episode", NAN)
        wt = E11.with_tariff(tt[name], tt[REF], ids, q)
        blk["tariff"] = wt
        blk["saving"], blk["saving_ci"] = wt["saving"], wt["saving_ci"]
        blk["sensitivity"] = {k: E11.with_tariff(tt[name], tt[REF], ids, v)
                              for k, v in (tf.get("sensitivity") or {}).items()}
        blk["ni"] = ni_block(int(blk["wins"]), int(blk["losses"]), int(blk["n"]))
        keys = [t for t in ids if t in tt[name] and t in tt[REF] and not tt[name][t].get("error")
                and not tt[REF][t].get("error")]
        blk["taken"] = taken_stats([tt[name][t] for t in keys], mode)
        res["arms"][mode] = blk
    res["by_task_type"] = E6A.by_type({"llm_only": tt[REF], **{m: tt[n] for m, n in ARMS.items()}}, ids)
    res["pilot_contact"] = {t: {name: (C.success(tt[name][t]) if t in tt[name] and not tt[name][t].get("error") else None)
                                for name in (REF, *ARMS.values())} for t in PILOT_IDS if t in ids}
    res["seen"] = seen_block()
    res["seen_change_pt"] = {}
    s = res["seen"]
    if s:
        if res["reference"].get("success") is not None and (s.get("llm_only") or {}).get("success") is not None:
            res["seen_change_pt"]["llm_only"] = 100 * (res["reference"]["success"] - s["llm_only"]["success"])
        for mode in ARMS:
            a = res["arms"].get(mode) or {}
            if a.get("success") is not None and (s.get(mode) or {}).get("success") is not None:
                res["seen_change_pt"][mode] = 100 * (a["success"] - s[mode]["success"])
    res["spend"] = total_spend(runs, box / "api_probe_log.txt" if (box / "api_probe_log.txt").exists() else PROBE_LOG)
    hh = box / "host_hours.json"
    res["host"] = json.loads(hh.read_text()) if hh.exists() else {}
    return res


def _f(x: Any, d: int = 1, sign: bool = False, scale: float = 1.0) -> str:
    return E11._f(x, d, sign, scale)


def to_markdown(r: Mapping[str, Any]) -> str:
    ref, tf, s = r["reference"], r["tariff"], r.get("seen") or {}
    sl = s.get("llm_only") or {}
    L = [f"# E23: E11's checkpoint in ALFWorld's unseen rooms ({r['row']})", "",
         f"- **{r['label']}.** Outside every Holm table; no confirmatory test. Generated {r['generated']} at git "
         f"`{(r['git'] or '?')[:12]}` by `scripts/explore_e23_alfworld_unseen.py`.",
         f"- Games: valid_unseen, {r['eval']['n_games']} games (list sha256 {r['eval']['list_sha256'][:16]}; registered "
         f"{r['eval']['registered_list_sha256'][:16]}); scenes {r['eval'].get('scenes')} (none in E11's training or "
         "calibration games).",
         f"- Model: {r['settings']['decision_model']}; checkpoint sha256 check against E11's record: "
         f"{'ok' if (r.get('checkpoint') or {}).get('ok') else 'MISSING / FAILED'}; CASCADE at E11's tau-hat "
         f"{r['settings']['tau_hat']} (no refit).",
         f"- GPU tariff q = ${_f(tf.get('q_usd_per_episode'), 5)} per GPU-arm episode ({tf.get('instance_type')}, "
         f"{_f(tf.get('serving_hours'), 3)} h / {tf.get('episodes')} episodes); sensitivity: q = 0 and S2-4's q. "
         f"E11 (valid_seen): q = ${_f(s.get('tariff_q'), 5)}.",
         f"- LLM only, unseen: success {_f(ref.get('success'), scale=100)} % (n {ref['n']}, errored {ref['n_errored']}), "
         f"${_f(ref.get('cost_usd'), 4)} per episode, wall mean / median {_f(ref.get('wall_mean_s'))} / "
         f"{_f(ref.get('wall_median_s'))} s, {_f(ref.get('steps'))} steps; its executed action matched the single gold "
         f"action at {_f(((ref.get('decisions') or {}).get('llm_acc') or NAN), scale=100)} % of its steps. Seen (E11): "
         f"{_f(sl.get('success'), scale=100)} %, ${_f(sl.get('cost_usd'), 4)}, {_f(sl.get('wall_mean_s'))} / "
         f"{_f(sl.get('wall_median_s'))} s, {_f(sl.get('steps'))} steps.", "",
         "## Unseen rooms (E23) beside valid_seen (E11)", "",
         "| arm | split | n | success % (ref.) | Δ pt [95 % bootstrap] | wins / losses | NI (3 pt): p, verdict "
         "[Tango 95 %] | saving % with q [95 %] | saving % API only | wall s mean / median (ref.) | steps | LLM calls | "
         "coverage % | taken ≠ gold |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, a in r["arms"].items():
        if a.get("status") != "run" or not a.get("n"):
            L.append(f"| {mode} | unseen | -- | {a.get('status')} | | | | | | | | | | |")
        else:
            d, ni, tk = a["decisions"], a["ni"], a["taken"]
            L.append(f"| {mode} | unseen | {a['n']} | {_f(a['success'], scale=100)} ({_f(a['success_ref'], scale=100)}) | "
                     f"{_f(a['delta_pt'], sign=True)} [{_f(a['delta_ci_pt'][0], sign=True)}, "
                     f"{_f(a['delta_ci_pt'][1], sign=True)}] | {a['wins']} / {a['losses']} | {ni['p']:.3f}, "
                     f"{ni['verdict']} [{_f(ni['tango_ci_pt'][0], sign=True)}, {_f(ni['tango_ci_pt'][1], sign=True)}] | "
                     f"{_f(a['saving'], scale=100)} [{_f(a['saving_ci'][0], scale=100)}, {_f(a['saving_ci'][1], scale=100)}]"
                     f" | {_f(a['saving_api'], scale=100)} | {_f(a['wall_mean_s'])} / {_f(a['wall_median_s'])} "
                     f"({_f(a['wall_ref_mean_s'])} / {_f(a['wall_ref_median_s'])}) | {_f(a['steps'])} | "
                     f"{_f(a['llm_calls'])} | {_f(d.get('coverage'), scale=100)} | {tk['n_wrong']} of "
                     f"{tk['n_taken_gold']} ({_f(tk.get('wrong_share'), scale=100)} %) |")
        b = s.get(mode) or {}
        if b:
            ni = b["ni"]
            L.append(f"| {mode} | seen (E11) | {b['n']} | {_f(b['success'], scale=100)} ({_f(sl.get('success'), scale=100)}) | "
                     f"{_f(b['delta_pt'], sign=True)} [{_f(b['delta_ci_pt'][0], sign=True)}, "
                     f"{_f(b['delta_ci_pt'][1], sign=True)}] | {b['wins']} / {b['losses']} | {ni['p']:.3f}, "
                     f"{ni['verdict']} [{_f(ni['tango_ci_pt'][0], sign=True)}, {_f(ni['tango_ci_pt'][1], sign=True)}] | "
                     f"{_f(b['saving'], scale=100)} [{_f((b.get('saving_ci') or [None])[0], scale=100)}, "
                     f"{_f((b.get('saving_ci') or [None, None])[1], scale=100)}] | {_f(b.get('saving_api'), scale=100)} | "
                     f"{_f(b['wall_mean_s'])} / {_f(b['wall_median_s'])} ({_f(sl.get('wall_mean_s'))} / "
                     f"{_f(sl.get('wall_median_s'))}) | {_f(b['steps'])} | {_f(b['llm_calls'])} | "
                     f"{_f(b.get('coverage'), scale=100)} | -- |")
    ch = r.get("seen_change_pt") or {}
    if ch:
        L += ["", "Seen-to-unseen change in success (points; unpaired, different games): "
              + ", ".join(f"{k} {_f(v, sign=True)}" for k, v in ch.items()) + "."]
    L += ["", "## The trained model's actions (unseen)", "",
          "| arm | actions | taken | taken with gold | ≠ gold | deferred | mean conf. | conf. < tau-hat % | acc. on taken % "
          "| deferred LLM acc. % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for mode, a in r["arms"].items():
        if a.get("status") != "run" or not a.get("n"):
            continue
        tk, d = a["taken"], a["decisions"]
        L.append(f"| {mode} | {tk['n_actions']} | {tk['n_taken']} | {tk['n_taken_gold']} | {tk['n_wrong']} | "
                 f"{tk['n_deferred']} | {_f(tk.get('mean_conf'), 3)} | {_f(tk.get('share_conf_below_tau'), scale=100)} | "
                 f"{_f(d.get('acc'), scale=100)} | {_f(d.get('deferred_llm_acc'), scale=100)} |")
    L += ["", "## Success by task type (descriptive, small n; seen = E11 on valid_seen)", "",
          "| task type | n unseen | LLM only | cascade | replace | n seen | LLM only | cascade | replace |",
          "|---|---|---|---|---|---|---|---|---|"]
    sb = s.get("by_task_type") or {}
    for typ, v in r["by_task_type"].items():
        w = sb.get(typ) or {}
        L.append(f"| {typ} | {v['n']} | " + " | ".join(_f(v.get(k), 0, scale=100) for k in ("llm_only", "cascade", "replace"))
                 + f" | {w.get('n', '--')} | " + " | ".join(_f(w.get(k), 0, scale=100) for k in ("llm_only", "cascade", "replace"))
                 + " |")
    if r.get("pilot_contact"):
        L += ["", "Games played before E23 in development (pilot P3b; never reported), success here: "
              + "; ".join(f"`{t.split('/')[2]}` " + ", ".join(f"{k} {v}" for k, v in x.items())
                          for t, x in r["pilot_contact"].items()) + "."]
    sp, h = r["spend"], r.get("host") or {}
    L += ["", "## Spend (logged, list prices) and host", "",
          f"Gemini ${sp['gemini']:.2f} ({sp['records']} records incl. errored attempts; probes ${sp['probes']:.4f}) over "
          + ", ".join(f"`{k}` ${v['total']:.2f}" for k, v in sp["by_run"].items()) + f"; cap ${CAP_USD:.0f}. "
          + (f"Host {h.get('instance_type')} {h.get('instance_id')}: {_f(h.get('hours'), 2)} host-hours "
             f"({h.get('launch')} to {h.get('stopped')}); cap 4." if h else "Host hours: not recorded yet."), ""]
    return "\n".join(L)


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    v: dict[str, str] = {}

    def put(k: str, x: Any, d: int = 1, sign: bool = False, scale: float = 1.0) -> None:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return
        y = x * scale
        v[k] = f"{y:+.{d}f}" if sign else f"{y:.{d}f}"   # signed keys as E11's: read with \resm (math mode)

    def put_int(k: str, x: Any) -> None:
        if isinstance(x, (int, float)) and math.isfinite(x):
            v[k] = f"{int(x):,}".replace(",", "{,}")

    v["e23/label"] = LABEL
    put_int("e23/n_games", r["eval"]["n_games"])
    put_int("e23/n_scenes", len(r["eval"].get("scenes") or {}))
    put("e23/tau", r["settings"]["tau_hat"], 2)
    tf = r["tariff"]
    put("e23/tariff/q", tf.get("q_usd_per_episode"), 5)
    put("e23/tariff/host_usd_h", tf.get("host_usd_per_hour"), 3)
    ref = r["reference"]
    put_int("e23/llm_only/n", ref.get("n"))
    put("e23/llm_only/success", ref.get("success"), 1, scale=100)
    put("e23/llm_only/cost_k", ref.get("cost_usd"), 1, scale=1000)
    put("e23/llm_only/wall_mean", ref.get("wall_mean_s"), 1)
    put("e23/llm_only/wall_median", ref.get("wall_median_s"), 1)
    put("e23/llm_only/steps", ref.get("steps"), 1)
    put("e23/llm_only/llm_acc", (ref.get("decisions") or {}).get("llm_acc"), 1, scale=100)
    for mode, a in r["arms"].items():
        p = f"e23/{mode}"
        if a.get("status") != "run" or not a.get("n"):
            v[f"{p}/status"] = str(a.get("status"))
            continue
        v[f"{p}/status"] = "run"
        put_int(f"{p}/n", a["n"])
        put(f"{p}/success", a["success"], 1, scale=100)
        put(f"{p}/delta", a["delta_pt"], 1, sign=True)
        put(f"{p}/delta_ci/lo", a["delta_ci_pt"][0], 1, sign=True)
        put(f"{p}/delta_ci/hi", a["delta_ci_pt"][1], 1, sign=True)
        v[f"{p}/wins"], v[f"{p}/losses"] = str(a["wins"]), str(a["losses"])
        put(f"{p}/cost_k", a["cost_usd"], 1, scale=1000)
        put(f"{p}/saving", a["saving"], 1, scale=100)
        put(f"{p}/saving_ci/lo", (a.get("saving_ci") or [None])[0], 1, scale=100)
        put(f"{p}/saving_ci/hi", (a.get("saving_ci") or [None, None])[1], 1, scale=100)
        put(f"{p}/saving_api", a.get("saving_api"), 1, scale=100)
        for k, sens in (a.get("sensitivity") or {}).items():
            put(f"{p}/saving_q_{k}", sens.get("saving"), 1, scale=100)
        put(f"{p}/wall_mean", a["wall_mean_s"], 1)
        put(f"{p}/wall_median", a["wall_median_s"], 1)
        put(f"{p}/steps", a["steps"], 1)
        put(f"{p}/llm_calls", a["llm_calls"], 1)
        put(f"{p}/coverage", a["decisions"].get("coverage"), 1, scale=100)
        put(f"{p}/acc", a["decisions"].get("acc"), 1, scale=100)
        put(f"{p}/deferred_llm_acc", a["decisions"].get("deferred_llm_acc"), 1, scale=100)
        ni = a["ni"]
        v[f"{p}/ni/p"] = f"{ni['p']:.3f}"
        v[f"{p}/ni/verdict"] = ni["verdict"]
        v[f"{p}/ni/yes"] = "yes" if ni["verdict"] == "non-inferior" else "no"
        put(f"{p}/ni/lo", ni["tango_ci_pt"][0], 1, sign=True)
        put(f"{p}/ni/hi", ni["tango_ci_pt"][1], 1, sign=True)
        tk = a["taken"]
        put_int(f"{p}/taken/n", tk["n_taken"])
        put_int(f"{p}/taken/n_gold", tk["n_taken_gold"])
        put_int(f"{p}/taken/n_wrong", tk["n_wrong"])
        put(f"{p}/taken/wrong_share", tk.get("wrong_share"), 1, scale=100)
        put_int(f"{p}/taken/n_deferred", tk["n_deferred"])
        put(f"{p}/taken/mean_conf", tk.get("mean_conf"), 2)
    for k, x in (r.get("seen_change_pt") or {}).items():
        put(f"e23/seen_change/{k}", x, 1, sign=True)
    s = r.get("seen") or {}
    for mode in ARMS:
        b = s.get(mode) or {}
        if b.get("ni"):
            v[f"e23/seen/{mode}/ni/p"] = f"{b['ni']['p']:.3f}"
            v[f"e23/seen/{mode}/ni/verdict"] = b["ni"]["verdict"]
    for typ, x in (r.get("by_task_type") or {}).items():
        short = TYPE_SHORT.get(typ, typ)
        put_int(f"e23/type/{short}/n", x.get("n"))
        for k in ("llm_only", "cascade", "replace"):
            put(f"e23/type/{short}/{k}", x.get(k), 0, scale=100)
    put("e23/spend/gemini", r["spend"]["gemini"], 2)
    put("e23/host/hours", (r.get("host") or {}).get("hours"), 2)
    put_int("e23/pilot_contact/n", len(r.get("pilot_contact") or {}))
    return v


def to_tex(r: Mapping[str, Any]) -> str:
    lines = ["% AUTO-GENERATED by scripts/explore_e23_alfworld_unseen.py -- do not edit by hand.",
             "% E23: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E23').",
             f"% generated {r['generated']} at git {(r['git'] or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(tex_keys(r).items())]
    return "\n".join(lines + [r"\makeatother", ""])


def cmd_summarize(a: argparse.Namespace) -> int:
    r = summarize(Path(a.runs), Path(a.box))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (OUT_DIR / "summary.md").write_text(to_markdown(r) + "\n")
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(to_tex(r))
    print(to_markdown(r))
    print(f"wrote {C.rel(OUT_DIR / 'summary.json')}, {C.rel(OUT_DIR / 'summary.md')}, {C.rel(TEX)}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("eval-list")
    s.add_argument("--data-dir", default=str(EVAL_DATA))
    s.add_argument("--check", action="store_true")
    s = sub.add_parser("extract-unseen")
    s.add_argument("--json-zip", required=True)
    s.add_argument("--twpddl-zip", required=True)
    s.add_argument("--data-dir", default=str(EVAL_DATA))
    s = sub.add_parser("verify-ckpt")
    s.add_argument("--dir", required=True)
    s.add_argument("--out", default=None)
    s = sub.add_parser("probe")
    s.add_argument("--log", default=str(PROBE_LOG))
    s = sub.add_parser("spend")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--log", default=str(PROBE_LOG))
    s.add_argument("--short", action="store_true")
    s = sub.add_parser("errored")
    s.add_argument("--run", required=True)
    s = sub.add_parser("watch")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--run", default=RUN_EVAL)
    s.add_argument("--log", default=str(PROBE_LOG))
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("--cap", type=float, default=CAP_USD)
    s.add_argument("--every", type=float, default=30.0)
    s.add_argument("--no-pause", action="store_true")
    s = sub.add_parser("project")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--log", default=str(PROBE_LOG))
    s.add_argument("--out", default=str(BOX_DIR / "projection.json"))
    s = sub.add_parser("summarize")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--box", default=str(BOX_DIR))
    a = ap.parse_args(argv)
    return {"eval-list": cmd_eval_list, "extract-unseen": cmd_extract_unseen, "verify-ckpt": cmd_verify_ckpt,
            "probe": cmd_probe, "spend": cmd_spend, "errored": cmd_errored, "watch": cmd_watch,
            "project": cmd_project, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
