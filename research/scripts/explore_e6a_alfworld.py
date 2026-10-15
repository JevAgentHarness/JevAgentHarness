#!/usr/bin/env python3
r"""E6a (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry
"Exploratory row E6a" of 2026-10-06): action-only control on ALFWorld valid_seen. Arms: LLM only, and the decision
model (Jev via OpenRouter) at the `action` decision point in REPLACE, CASCADE (tau fitted on ALFWorld *train* games
with the repo's calibration-arm estimator) and ASSIST (k = 3, the registered tool-ASSIST k). Agent LLM: Gemini 3.5
Flash, minimal reasoning, temperature 0. Not the registered S2-L3b; revises no registered result.

    python scripts/explore_e6a_alfworld.py select-train --json-zip J.zip --twpddl-zip T.zip   # 48 train games
    python scripts/explore_e6a_alfworld.py eval-list                       # the valid_seen game list (frozen)
    python scripts/explore_e6a_alfworld.py fit                             # tau-hat of the action cascade
    python scripts/explore_e6a_alfworld.py spend                           # logged $ of every E6a run
    python scripts/explore_e6a_alfworld.py watch --cap 80 --pid PID        # watchdog: kills PID's group at the cap
    python scripts/explore_e6a_alfworld.py project                         # smoke -> eval cost projection
    python scripts/explore_e6a_alfworld.py summarize                       # results/explore/e6a_alfworld + e6a.tex

Readout: paired complete case over the eval list (a game whose latest record errored, or that is missing, in
either arm leaves that pair). Success and Delta vs LLM only (points; paired task bootstrap B = 10,000, seed 0, and the
Newcombe interval), paired wins / losses, cost per episode ($, Gemini + Jev at list prices) and saving
1 - C_arm / C_ref (task bootstrap), mean / median wall time per episode, steps and calls per episode, and the
decision model's coverage (share of action decisions executed from its answer; ASSIST: 0 by definition, it acts
through the shortlist at every step) and its accuracy against the single gold action (a lower bound: many states
admit several equally good actions, docs/ALFWORLD.md). No test; outside every Holm table."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.suites.alfworld import TASK_TYPES, collect_game_files  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
CFG_DIR = ROOT / "configs/explore/e6a_alfworld"
OUT_DIR = ROOT / "results/explore/e6a_alfworld"
TEX = ROOT / "paper/generated/e6a.tex"
TRAIN_LIST = CFG_DIR / "train_games.json"
EVAL_LIST = CFG_DIR / "eval_games.json"
TAU_FILE = CFG_DIR / "tauhat_action.json"
TRAIN_DATA = ROOT / "data/raw/alfworld_e6a"          # the selected train games only (gitignored)
EVAL_DATA = ROOT / "data/raw/alfworld"               # valid_seen (140 games)
RUN_CAL, RUN_SMOKE, RUN_EVAL = "e6a_alfworld_cal", "e6a_alfworld_smoke", "e6a_alfworld_eval"
CAL_ARM = "jev_action_shadow_cal"
REF = "llm_only"
ARMS = {"replace": "jev_action_replace", "cascade": "jev_action_cascade_tauhat", "assist": "jev_action_assist_k3"}
LABEL = "exploratory (E6a)"
SALT = "jev-explore-E6a-alfworld-train-2026-10-06"
N_PER_TYPE = 8
CAP_USD = 80.0
SAFETY = 1.5
MIN_EVAL_GAMES = 100
URLS = {"json": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
        "twpddl": "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"}
_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


# ============================================================================== train-game selection
def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def select_train(json_zip: Path, twpddl_zip: Path, n_per_type: int = N_PER_TYPE, salt: str = SALT
                 ) -> dict[str, list[str]]:
    """The calibration games: ALFWorld's own train-split filters (no ``movable`` / ``Sliced`` trajectories, the six
    task types read from ``traj_data.json``, ``solvable``), ordered by sha256(salt + "\\n" + "<task dir>/<trial>"),
    the first ``n_per_type`` of each task type. Returns type -> games (``<task dir>/<trial>``) in that order."""
    zj, zt = zipfile.ZipFile(json_zip), zipfile.ZipFile(twpddl_zip)
    pre = "json_2.1.1/train/"
    tw = {n[: -len("/game.tw-pddl")]: n for n in zt.namelist() if n.startswith(pre) and n.endswith("/game.tw-pddl")}
    js = {n[: -len("/traj_data.json")]: n for n in zj.namelist() if n.startswith(pre) and n.endswith("/traj_data.json")}
    cands = []
    for d in sorted(set(tw) & set(js)):
        relp = d[len(pre):]
        parts = relp.split("/")
        if len(parts) != 2 or not all(_SAFE.match(p) for p in parts) or "movable" in d or "Sliced" in d:
            continue
        cands.append(relp)
    cands.sort(key=lambda r: _sha(f"{salt}\n{r}"))
    wanted = list(TASK_TYPES.values())
    out: dict[str, list[str]] = {t: [] for t in wanted}
    for relp in cands:
        if all(len(v) >= n_per_type for v in out.values()):
            break
        typ = json.loads(zj.read(js[pre + relp])).get("task_type")
        if typ not in out or len(out[typ]) >= n_per_type:
            continue
        if not json.loads(zt.read(tw[pre + relp])).get("solvable", False):
            continue
        out[typ].append(relp)
    return out


def extract_train(json_zip: Path, twpddl_zip: Path, games: Iterable[str], root: Path) -> list[Path]:
    zj, zt = zipfile.ZipFile(json_zip), zipfile.ZipFile(twpddl_zip)
    paths = []
    for relp in games:
        parts = relp.split("/")
        assert len(parts) == 2 and all(_SAFE.match(p) for p in parts), relp
        d = root / "json_2.1.1" / "train" / parts[0] / parts[1]
        d.mkdir(parents=True, exist_ok=True)
        (d / "traj_data.json").write_bytes(zj.read(f"json_2.1.1/train/{relp}/traj_data.json"))
        (d / "game.tw-pddl").write_bytes(zt.read(f"json_2.1.1/train/{relp}/game.tw-pddl"))
        paths.append(d / "game.tw-pddl")
    return paths


def cmd_select_train(a: argparse.Namespace) -> int:
    sel = select_train(Path(a.json_zip), Path(a.twpddl_zip), a.n_per_type)
    games = [g for t in TASK_TYPES.values() for g in sel[t]]
    short = {t: len(v) for t, v in sel.items() if len(v) < a.n_per_type}
    if short:
        raise SystemExit(f"too few eligible train games for {short}")
    extract_train(Path(a.json_zip), Path(a.twpddl_zip), games, Path(a.data_dir))
    listed = collect_game_files("train", data_dir=a.data_dir)
    got = sorted(str(Path(p).parent.relative_to(Path(a.data_dir) / "json_2.1.1" / "train")) for p in listed)
    if got != sorted(games):
        raise SystemExit(f"{a.data_dir} lists {len(got)} train games, expected the {len(games)} selected ones")
    # the valid_seen games on disk are the release's (byte-identical game files)
    zt = zipfile.ZipFile(a.twpddl_zip)
    vs = collect_game_files("valid_seen", data_dir=EVAL_DATA)
    same = sum(Path(p).read_bytes() == zt.read("json_2.1.1/valid_seen/" + "/".join(Path(p).parts[-3:]))
               for p in vs)
    rec = {"row": LABEL, "rule": "ALFWorld train-split filters (no movable / Sliced, six task types, solvable); "
           f"order by sha256(salt + newline + '<task dir>/<trial>'); first {a.n_per_type} per task type",
           "salt": SALT, "n_per_type": a.n_per_type, "n": len(games),
           "archives": {k: {"url": URLS[k], "sha256": C.sha256_file(Path(p))}
                        for k, p in (("json", a.json_zip), ("twpddl", a.twpddl_zip))},
           "data_dir": C.rel(Path(a.data_dir)), "games": sel,
           "list_sha256": _sha("\n".join(games)),
           "valid_seen_check": {"n": len(vs), "identical_to_release": int(same)}}
    TRAIN_LIST.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_LIST.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {C.rel(TRAIN_LIST)}: {len(games)} train games (list sha256 {rec['list_sha256'][:16]}); "
          f"valid_seen on disk: {len(vs)} games, {same} byte-identical to the release")
    return 0


def eval_ids(data_dir: Path = EVAL_DATA) -> list[str]:
    """The valid_seen task ids in the eval run's order (``load_alfworld(..., stratify=True)``: task types round-robin
    over the sorted game files)."""
    from jevharness.bench.suites.alfworld import load_alfworld
    return [f().task_id for f in load_alfworld("valid_seen", data_dir=str(data_dir), stratify=True)]


def cmd_eval_list(a: argparse.Namespace) -> int:
    ids = eval_ids(Path(a.data_dir))
    rec = {"row": LABEL, "split": "valid_seen (eval_in_distribution)", "order": "stratified (task types round-robin "
           "over the sorted game files; load_alfworld stratify=True)", "n": len(ids), "task_ids": ids,
           "list_sha256": _sha("\n".join(ids))}
    EVAL_LIST.parent.mkdir(parents=True, exist_ok=True)
    EVAL_LIST.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {C.rel(EVAL_LIST)}: {len(ids)} games (sha256 {rec['list_sha256'][:16]})")
    return 0


# ============================================================================== threshold fit
def fit_tau(run_dir: Path, arm: str = CAL_ARM) -> dict[str, Any]:
    """The repo's calibration-arm estimator (``threshold_theory.calibration_arm_run``: V = W x (S_right - S_wrong)
    with the A6.11 fallback V = W, dC from the ledger, the scalar-threshold projection of Eq. 3 and A6.5's sample-size
    rule) at the action decision point, W = $1, lambda = 0. ``deploy`` is what the CASCADE arm runs."""
    from jevharness.bench.threshold_theory import _jsonable, calibration_arm_run
    res = _jsonable(calibration_arm_run(run_dir, [arm], dp="action", task_value=1.0, lam_per_hour=0.0))
    t = res.get("tau_hat")
    t = float(t) if t is not None and isinstance(t, (int, float)) else float("inf")
    ok = math.isfinite(t) and t <= 1.0
    res["deploy"] = ({"mode": "cascade", "tau": round(t, 6)} if ok else
                     {"mode": "cascade", "always_defer": True,
                      "note": "the fit accepts no confidence level: the cascade always defers, i.e. it is the "
                              "LLM-only harness (A6.11 (c) 4); the arm is not run"})
    return res


def cmd_fit(a: argparse.Namespace) -> int:
    run = Path(a.run)
    res = fit_tau(run, a.arm)
    trajs = list(_latest(run / f"{a.arm}.jsonl").values())
    res["cal_run"] = {"dir": C.rel(run), "arm": a.arm, "episodes": len(trajs),
                      "errored": sum(bool(t.get("error")) for t in trajs),
                      "success": float(np.mean([C.success(t) for t in trajs])) if trajs else None,
                      "sha256": C.sha256_file(run / f"{a.arm}.jsonl")}
    TAU_FILE.parent.mkdir(parents=True, exist_ok=True)
    TAU_FILE.write_text(json.dumps(res, indent=1, default=str) + "\n")
    p = res.get("primary") or {}
    print(f"wrote {C.rel(TAU_FILE)}: n_paired {res.get('n_paired')}, rule {(res.get('tau_rule') or {}).get('rule')}, "
          f"V {p.get('V')} ({p.get('V_source')}), dC {p.get('delta_c')}, tau_hat {res.get('tau_hat')} -> "
          f"deploy {res['deploy']}")
    return 0


# ============================================================================== spend, watchdog, projection
def _latest(path: Path) -> dict[str, dict[str, Any]]:
    return C.load_latest([path])[0] if path.exists() else {}


def episode_spend(t: Mapping[str, Any]) -> dict[str, float]:
    """$ of one logged episode (shadow measurement calls included): Gemini (System-2) and Jev (System-1)."""
    u = t.get("usage") or {}
    s1 = float((u.get("s1") or {}).get("cost_usd") or 0.0)
    s2 = float((u.get("s2") or {}).get("cost_usd") or 0.0)
    tot = float((u.get("total") or {}).get("cost_usd") or 0.0)
    sh = float((u.get("shadow") or {}).get("cost_usd") or 0.0)
    other = max(tot - s1 - s2, 0.0)
    return {"gemini": s2 + other, "jev": s1 + sh, "total": tot + sh}


def run_spend(run_dir: Path) -> dict[str, Any]:
    """Every record of every arm (errored attempts and re-runs included: all of them were paid)."""
    out = {"gemini": 0.0, "jev": 0.0, "total": 0.0, "records": 0}
    for p in sorted(Path(run_dir).glob("*.jsonl")):
        if p.name.startswith("schedule"):
            continue
        for line in p.read_text().splitlines():
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if not isinstance(t, dict) or "task_id" not in t:
                continue
            s = episode_spend(t)
            for k in ("gemini", "jev", "total"):
                out[k] += s[k]
            out["records"] += 1
    return out


# The first calibration attempt ran in the development container and was aborted when that disk filled (ENOSPC at
# TextWorld resets; its last record truncated): $0.516 logged in its 47 readable records (Gemini $0.503, Jev $0.013)
# plus an allowance of $0.05 for the truncated record. Counted toward the cap; its episodes are not used.
PRIOR_SPEND = {"e6a_alfworld_cal_container_aborted": {"gemini": 0.553, "jev": 0.013, "total": 0.566, "records": 48}}


def total_spend(runs: Path = RUNS, prior: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, Any]:
    per = {name: run_spend(runs / name) for name in (RUN_CAL, RUN_SMOKE, RUN_EVAL) if (runs / name).is_dir()}
    per.update(PRIOR_SPEND if prior is None else prior)
    tot = {k: sum(v[k] for v in per.values()) for k in ("gemini", "jev", "total", "records")}
    return {"by_run": per, **tot}


def cmd_spend(a: argparse.Namespace) -> int:
    print(json.dumps(total_spend(Path(a.runs)), indent=1))
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    """Poll the logged spend every ``--every`` s; at the cap send SIGTERM to the runner's process group (then SIGKILL)
    and write ``<runs>/e6a_STOP.json``. Episodes in flight when it fires are not logged (a few cents)."""
    stop = Path(a.runs) / "e6a_STOP.json"
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(Path(a.runs))
        if s["total"] >= a.cap:
            stop.write_text(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                        "spend": s, "cap": a.cap}, indent=1))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(a.pid), sig)
                except OSError:
                    pass
                time.sleep(10)
            print(f"CAP REACHED: ${s['total']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} spend ${s['total']:.3f} (Gemini ${s['gemini']:.3f}, "
              f"Jev ${s['jev']:.4f}; {s['records']} records)", flush=True)
        time.sleep(a.every)


def projection(runs: Path = RUNS, n_games: int = 140, safety: float = SAFETY, cap: float = CAP_USD) -> dict[str, Any]:
    """The registered gate: smoke mean $ per episode per arm x the eval games x ``safety``, plus the spend so far,
    against the cap; if it does not fit, the first ``MIN_EVAL_GAMES`` games of the eval list are run (still checked)."""
    sm = {arm: list(_latest(runs / RUN_SMOKE / f"{arm}.jsonl").values()) for arm in (REF, *ARMS.values())}
    per = {arm: (float(np.mean([episode_spend(t)["total"] for t in ts])) if ts else None) for arm, ts in sm.items()}
    wall = {arm: (float(np.mean([float(t.get("wall_ms") or 0) for t in ts])) / 1000 if ts else None)
            for arm, ts in sm.items()}
    so_far = total_spend(runs)["total"]
    known = [v for v in per.values() if v is not None]
    ep = sum(known)
    out = {"smoke_per_episode": per, "smoke_wall_s": wall, "smoke_n": {k: len(v) for k, v in sm.items()},
           "spent_so_far": so_far, "safety": safety, "cap": cap}
    for n in (n_games, MIN_EVAL_GAMES):
        out[f"projected_{n}"] = ep * n * safety
        out[f"fits_{n}"] = so_far + ep * n * safety <= cap
    out["decision"] = (f"run all {n_games}" if out[f"fits_{n_games}"] else
                       f"run the first {MIN_EVAL_GAMES}" if out[f"fits_{MIN_EVAL_GAMES}"] else "stop after the smoke")
    return out


def cmd_project(a: argparse.Namespace) -> int:
    p = projection(Path(a.runs), a.n_games)
    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "projection.json").write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(p, indent=1))
    return 0


# ============================================================================== readout
def _action_recs(t: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [d for st in t.get("steps") or [] for d in st.get("decisions") or []
            if d.get("dp") == "action" and not d.get("shadow")]


def _top_k(dist: Mapping[str, float] | None, k: int) -> list[str]:
    return [o for o, _ in sorted((dist or {}).items(), key=lambda kv: -float(kv[1]))][:k]


def decision_stats(trajs: Sequence[Mapping[str, Any]], mode: str, k: int = 3) -> dict[str, Any]:
    """Coverage (executed from the decision model's answer), and accuracy against the single gold action (lower
    bound): the decision model's answer (replace), its accepted answers (cascade), gold in its top-k (assist), the
    LLM's executed action (every mode where the LLM decided)."""
    recs = [d for t in trajs for d in _action_recs(t)]
    out: dict[str, Any] = {"n_decisions": len(recs)}
    if not recs:
        return out
    dr = M.delegation_ratio(recs, dps=("action",))
    out["coverage"] = dr["ratio"]
    g = [d for d in recs if d.get("gold") is not None]
    out["n_gold"] = len(g)

    def acc(rs: Sequence[Mapping[str, Any]], key: str) -> float | None:
        rs = [d for d in rs if d.get(key) is not None]
        return float(np.mean([M.is_correct(d, d[key]) is True for d in rs])) if rs else None
    if mode != "off":
        out["s1_acc"] = acc(g, "s1_value")
        out["mean_conf"] = float(np.mean([d["s1_conf"] for d in recs if d.get("s1_conf") is not None])) \
            if any(d.get("s1_conf") is not None for d in recs) else None
    if mode == "cascade":
        out["accepted_acc"] = acc([d for d in g if not d.get("deferred")], "s1_value")
        out["deferred_llm_acc"] = acc([d for d in g if d.get("deferred")], "s2_value")
    if mode == "assist":
        hits = [M._norm(d["gold"]) in {M._norm(x) for x in _top_k(d.get("s1_dist"), k)} for d in g
                if d.get("s1_dist")]
        out["recall_k"] = float(np.mean(hits)) if hits else None
        out["k"] = k
    out["llm_acc"] = acc([d for d in g if d.get("s2_value") is not None], "s2_value")
    out["acc"] = {"replace": out.get("s1_acc"), "cascade": out.get("accepted_acc"), "assist": out.get("recall_k"),
                  "off": out.get("llm_acc")}.get(mode)
    return out


def arm_block(arm_t: Mapping[str, Mapping[str, Any]], ref_t: Mapping[str, Mapping[str, Any]], ids: Sequence[str],
              mode: str) -> dict[str, Any]:
    status = {}
    for side, tt in (("arm", arm_t), ("ref", ref_t)):
        status[side] = {"missing": sum(t not in tt for t in ids),
                        "errored": sum(t in tt and bool(tt[t].get("error")) for t in ids)}
    keys = [t for t in ids if t in arm_t and t in ref_t and not arm_t[t].get("error") and not ref_t[t].get("error")]
    out: dict[str, Any] = {"n_expected": len(ids), "n": len(keys), "n_dropped": len(ids) - len(keys), "dropped": status}
    if not keys:
        return out
    A = [arm_t[t] for t in keys]
    R = [ref_t[t] for t in keys]
    ya, yr = [float(C.success(t)) for t in A], [float(C.success(t)) for t in R]
    ca = [episode_spend(t)["total"] for t in A]
    cr = [episode_spend(t)["total"] for t in R]
    wa = [float(t.get("wall_ms") or 0.0) / 1000 for t in A]
    wr = [float(t.get("wall_ms") or 0.0) / 1000 for t in R]
    pb = C.paired_boot(ya, yr)
    sv = C.paired_boot(ca, cr, ratio=True)
    nc = C.newcombe(ya, yr)
    out.update({
        "success": float(np.mean(ya)), "success_ref": float(np.mean(yr)),
        "delta_pt": 100 * pb["est"], "delta_ci_pt": [100 * x for x in pb["ci"]], "delta_newcombe_pt": nc["ci_pt"],
        "wins": int(sum(a > b for a, b in zip(ya, yr))), "losses": int(sum(a < b for a, b in zip(ya, yr))),
        "cost_usd": float(np.mean(ca)), "cost_ref_usd": float(np.mean(cr)),
        "gemini_usd": float(np.mean([episode_spend(t)["gemini"] for t in A])),
        "jev_usd": float(np.mean([episode_spend(t)["jev"] for t in A])),
        "saving": sv["est"], "saving_ci": sv["ci"],
        "wall_mean_s": float(np.mean(wa)), "wall_median_s": float(statistics.median(wa)),
        "wall_ref_mean_s": float(np.mean(wr)), "wall_ref_median_s": float(statistics.median(wr)),
        "wall_delta_median_s": float(statistics.median([a - b for a, b in zip(wa, wr)])),
        "steps": float(np.mean([len(t.get("steps") or []) for t in A])),
        "llm_calls": float(np.mean([C.s_calls(t, "s2") for t in A])),
        "s1_calls": float(np.mean([C.s_calls(t, "s1") for t in A])),
        "decisions": decision_stats(A, mode)})
    return out


def ref_block(ref_t: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> dict[str, Any]:
    keys = [t for t in ids if t in ref_t and not ref_t[t].get("error")]
    R = [ref_t[t] for t in keys]
    w = [float(t.get("wall_ms") or 0.0) / 1000 for t in R]
    return {"n": len(keys), "n_errored": sum(t in ref_t and bool(ref_t[t].get("error")) for t in ids),
            "success": float(np.mean([C.success(t) for t in R])) if R else None,
            "cost_usd": float(np.mean([episode_spend(t)["total"] for t in R])) if R else None,
            "wall_mean_s": float(np.mean(w)) if w else None, "wall_median_s": float(statistics.median(w)) if w else None,
            "steps": float(np.mean([len(t.get("steps") or []) for t in R])) if R else None,
            "llm_calls": float(np.mean([C.s_calls(t, "s2") for t in R])) if R else None,
            "decisions": decision_stats(R, "off")}


def by_type(arms: Mapping[str, Mapping[str, Mapping[str, Any]]], ids: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for typ in TASK_TYPES.values():
        tids = [t for t in ids if t.split("/")[2].startswith(typ + "-")]
        out[typ] = {"n": len(tids)}
        for name, tt in arms.items():
            ok = [tt[t] for t in tids if t in tt and not tt[t].get("error")]
            out[typ][name] = float(np.mean([C.success(t) for t in ok])) if ok else None
    return out


def summarize(runs: Path = RUNS) -> dict[str, Any]:
    ev = json.loads(EVAL_LIST.read_text())
    run = runs / RUN_EVAL
    exp_n = None
    if (run / "experiment.yaml").exists():
        import yaml
        exp = yaml.safe_load((run / "experiment.yaml").read_text())
        exp_n = int((exp.get("suite") or {}).get("n") or ev["n"])
    ids = ev["task_ids"][: exp_n or ev["n"]]
    tt = {name: _latest(run / f"{name}.jsonl") for name in (REF, *ARMS.values())}
    tau = json.loads(TAU_FILE.read_text()) if TAU_FILE.exists() else {}
    res: dict[str, Any] = {
        "row": LABEL, "label": "descriptive, revises no registered result, not the registered S2-L3b",
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": C.git_head(),
        "eval": {"split": "valid_seen", "n_games": len(ids), "list_sha256": _sha("\n".join(ids)),
                 "run": C.rel(run), "files": {n: {"sha256": C.sha256_file(run / f"{n}.jsonl"),
                                                  "records": sum(1 for _ in open(run / f"{n}.jsonl"))}
                                              for n in tt if (run / f"{n}.jsonl").exists()}},
        "settings": {"agent_llm": "gemini-3.5-flash (temperature 0, reasoning_effort minimal)",
                     "decision_model": "Jev via OpenRouter (typesafe/jev-1.13)", "max_steps": 50,
                     "history_window": 12, "assist_k": 3, "schedule": "task_major, seed 732"},
        "tau": {"tau_hat": tau.get("tau_hat"), "deploy": tau.get("deploy"), "rule": (tau.get("tau_rule") or {}).get("rule"),
                "n_paired": tau.get("n_paired"), "cal_games": (tau.get("cal_run") or {}).get("episodes"),
                "V": (tau.get("primary") or {}).get("V"), "V_source": (tau.get("primary") or {}).get("V_source"),
                "delta_c": (tau.get("primary") or {}).get("delta_c")},
        "reference": ref_block(tt[REF], ids), "arms": {}}
    for mode, name in ARMS.items():
        if not tt[name]:
            res["arms"][mode] = {"arm": name, "status": "not run"}
            continue
        res["arms"][mode] = {"arm": name, "status": "run", **arm_block(tt[name], tt[REF], ids, mode)}
    res["by_task_type"] = by_type({"llm_only": tt[REF], **{m: tt[n] for m, n in ARMS.items()}}, ids)
    res["spend"] = total_spend(runs)
    return res


def _f(x: Any, d: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def to_markdown(r: Mapping[str, Any]) -> str:
    ref, t = r["reference"], r["tau"]
    L = [f"# E6a: action-only control on ALFWorld valid_seen ({r['row']})", "",
         f"- **{r['label']}.** Outside every Holm table; no test. Generated {r['generated']} at git "
         f"`{(r['git'] or '?')[:12]}` by `scripts/explore_e6a_alfworld.py`.",
         f"- Games: {r['eval']['n_games']} valid_seen games (list sha256 `{r['eval']['list_sha256'][:16]}`); "
         f"run `{r['eval']['run']}`. Agent LLM {r['settings']['agent_llm']}; decision model "
         f"{r['settings']['decision_model']}; max 50 steps; {r['settings']['schedule']}.",
         f"- CASCADE threshold fitted on {t.get('cal_games')} ALFWorld train games before any valid_seen episode: "
         f"tau-hat = {_f(t.get('tau_hat'), 4)} (rule `{t.get('rule')}`, {t.get('n_paired')} paired decisions, "
         f"V = {_f(t.get('V'), 2)} [{t.get('V_source')}], dC = ${_f(t.get('delta_c'), 5)}).",
         f"- LLM only: success {_f(100 * ref['success'] if ref.get('success') is not None else None)} % "
         f"(n {ref['n']}, errored {ref['n_errored']}), ${_f(ref.get('cost_usd'), 4)} per episode, wall time mean / "
         f"median {_f(ref.get('wall_mean_s'))} / {_f(ref.get('wall_median_s'))} s, {_f(ref.get('steps'))} steps; the "
         f"LLM's executed action matched the single gold action at "
         f"{_f(100 * ref['decisions']['llm_acc'] if (ref.get('decisions') or {}).get('llm_acc') is not None else None)} "
         "% of its steps (a lower bound).", "",
         "| arm | n | success % | Δ pt vs LLM only [95 % bootstrap] | Newcombe | wins / losses | $ / episode "
         "(Gemini + Jev) | saving % [95 %] | wall s mean / median | steps | LLM calls | DM calls | coverage % | DM acc. % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, a in r["arms"].items():
        if a.get("status") != "run" or not a.get("n"):
            L.append(f"| {mode} (`{a['arm']}`) | -- | not run | | | | | | | | | | | |")
            continue
        d = a["decisions"]
        L.append(f"| {mode} (`{a['arm']}`) | {a['n']} | {_f(100 * a['success'])} | {_f(a['delta_pt'], sign=True)} "
                 f"[{_f(a['delta_ci_pt'][0], sign=True)}, {_f(a['delta_ci_pt'][1], sign=True)}] | "
                 f"[{_f(a['delta_newcombe_pt'][0], sign=True)}, {_f(a['delta_newcombe_pt'][1], sign=True)}] | "
                 f"{a['wins']} / {a['losses']} | {_f(a['cost_usd'], 4)} ({_f(a['gemini_usd'], 4)} + {_f(a['jev_usd'], 4)}) | "
                 f"{_f(100 * a['saving'])} [{_f(100 * a['saving_ci'][0])}, {_f(100 * a['saving_ci'][1])}] | "
                 f"{_f(a['wall_mean_s'])} / {_f(a['wall_median_s'])} | {_f(a['steps'])} | {_f(a['llm_calls'])} | "
                 f"{_f(a['s1_calls'])} | {_f(100 * d['coverage'] if d.get('coverage') is not None else None)} | "
                 f"{_f(100 * d['acc'] if d.get('acc') is not None else None)} |")
    L += ["", "Reference cost per episode on the same pairs: "
          + "; ".join(f"{m} ${_f(a.get('cost_ref_usd'), 4)}" for m, a in r["arms"].items() if a.get("n")) + ".",
          "Coverage = share of the arm's action decisions executed from the decision model's answer (ASSIST: 0 by "
          "definition; it shortlists k = 3 actions at every step and the LLM chooses). DM acc. = accuracy against the "
          "single gold action (lower bound): REPLACE its answer, CASCADE its accepted answers, ASSIST gold in its top 3.",
          "", "## Success by task type (descriptive, small n)", "",
          "| task type | n | LLM only | replace | cascade | assist |", "|---|---|---|---|---|---|"]
    for typ, v in r["by_task_type"].items():
        L.append(f"| {typ} | {v['n']} | " + " | ".join(_f(100 * v[k] if v.get(k) is not None else None, 0)
                                                     for k in ("llm_only", "replace", "cascade", "assist")) + " |")
    s = r["spend"]
    L += ["", f"## Spend (logged, list prices)", "",
          f"Total ${s['total']:.2f} (Gemini ${s['gemini']:.2f}, Jev ${s['jev']:.3f}; {s['records']} records incl. "
          "errored attempts) over " + ", ".join(f"`{k}` ${v['total']:.2f}" for k, v in s["by_run"].items())
          + f"; cap ${CAP_USD:.0f}.", ""]
    return "\n".join(L)


def to_tex(r: Mapping[str, Any]) -> str:
    v: dict[str, str] = {}

    def put(k: str, x: Any, d: int = 1, sign: bool = False, scale: float = 1.0) -> None:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return
        y = x * scale
        v[k] = (f"{y:+.{d}f}" if sign else f"{y:.{d}f}")
    v["e6a/n_games"] = str(r["eval"]["n_games"])
    v["e6a/label"] = LABEL
    t = r["tau"]
    put("e6a/tau", t.get("tau_hat"), 2)
    if t.get("cal_games") is not None:
        v["e6a/cal_games"] = str(t["cal_games"])
    if t.get("n_paired") is not None:
        v["e6a/cal_paired"] = str(t["n_paired"])
    ref = r["reference"]
    put("e6a/llm_only/success", ref.get("success"), 1, scale=100)
    put("e6a/llm_only/cost_usd", ref.get("cost_usd"), 4)
    put("e6a/llm_only/cost_k", ref.get("cost_usd"), 1, scale=1000)
    put("e6a/llm_only/wall_mean", ref.get("wall_mean_s"), 1)
    put("e6a/llm_only/wall_median", ref.get("wall_median_s"), 1)
    put("e6a/llm_only/steps", ref.get("steps"), 1)
    put("e6a/llm_only/llm_acc", (ref.get("decisions") or {}).get("llm_acc"), 1, scale=100)
    v["e6a/llm_only/n"] = str(ref["n"])
    for mode, a in r["arms"].items():
        p = f"e6a/{mode}"
        if a.get("status") != "run" or not a.get("n"):
            v[f"{p}/status"] = "not run"
            continue
        v[f"{p}/status"] = "run"
        v[f"{p}/n"] = str(a["n"])
        v[f"{p}/n_dropped"] = str(a["n_dropped"])
        put(f"{p}/success", a["success"], 1, scale=100)
        put(f"{p}/delta", a["delta_pt"], 1, sign=True)
        put(f"{p}/delta_ci/lo", a["delta_ci_pt"][0], 1, sign=True)
        put(f"{p}/delta_ci/hi", a["delta_ci_pt"][1], 1, sign=True)
        v[f"{p}/wins"], v[f"{p}/losses"] = str(a["wins"]), str(a["losses"])
        put(f"{p}/cost_usd", a["cost_usd"], 4)
        put(f"{p}/cost_k", a["cost_usd"], 1, scale=1000)
        put(f"{p}/saving", a["saving"], 1, scale=100)
        put(f"{p}/saving_ci/lo", a["saving_ci"][0], 1, scale=100)
        put(f"{p}/saving_ci/hi", a["saving_ci"][1], 1, scale=100)
        put(f"{p}/wall_mean", a["wall_mean_s"], 1)
        put(f"{p}/wall_median", a["wall_median_s"], 1)
        put(f"{p}/steps", a["steps"], 1)
        put(f"{p}/llm_calls", a["llm_calls"], 1)
        put(f"{p}/s1_calls", a["s1_calls"], 1)
        put(f"{p}/coverage", a["decisions"].get("coverage"), 1, scale=100)
        put(f"{p}/acc", a["decisions"].get("acc"), 1, scale=100)
    s = r["spend"]
    put("e6a/spend/total", s["total"], 2)
    put("e6a/spend/gemini", s["gemini"], 2)
    put("e6a/spend/jev", s["jev"], 3)
    lines = ["% AUTO-GENERATED by scripts/explore_e6a_alfworld.py -- do not edit by hand.",
             "% E6a: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06); not S2-L3b.",
             f"% generated {r['generated']} at git {(r['git'] or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(v.items())]
    return "\n".join(lines + [r"\makeatother", ""])


def cmd_summarize(a: argparse.Namespace) -> int:
    r = summarize(Path(a.runs))
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
    s = sub.add_parser("select-train")
    s.add_argument("--json-zip", required=True)
    s.add_argument("--twpddl-zip", required=True)
    s.add_argument("--data-dir", default=str(TRAIN_DATA))
    s.add_argument("--n-per-type", type=int, default=N_PER_TYPE)
    s = sub.add_parser("eval-list")
    s.add_argument("--data-dir", default=str(EVAL_DATA))
    s = sub.add_parser("fit")
    s.add_argument("--run", default=str(RUNS / RUN_CAL))
    s.add_argument("--arm", default=CAL_ARM)
    for name in ("spend", "project", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("--runs", default=str(RUNS))
        if name == "project":
            s.add_argument("--n-games", type=int, default=140)
    s = sub.add_parser("watch")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("--cap", type=float, default=CAP_USD)
    s.add_argument("--every", type=float, default=30.0)
    a = ap.parse_args(argv)
    return {"select-train": cmd_select_train, "eval-list": cmd_eval_list, "fit": cmd_fit, "spend": cmd_spend,
            "watch": cmd_watch, "project": cmd_project, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
