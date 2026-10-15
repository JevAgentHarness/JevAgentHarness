#!/usr/bin/env python3
r"""E31 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-10 "E31: GRPO in the agent loop without
SFT (RL only, from the untrained base)"): v1's GRPO stage started from the untrained Qwen3-4B, measured as the training
ladder's rows (untrained / + SFT v1 / SFT + GRPO = AgentDecision-4B v1) were measured.

Host side (scripts/aws/e31_rl_only.sh):
    python scripts/explore_e31.py trainwatch --pid PID --metrics M --cap 30 --max-min 390 --stop S   # training guards
    python scripts/explore_e31.py trainspend --metrics M [--out J]          # rollouts + greedy evaluations $ (r5 rule)
    python scripts/explore_e31.py probe | spend | errored | watch | gate     # E21's live operations over the e31_* runs
Local:
    python scripts/explore_e31.py fetch        # s3://.../e31_rl_only/{box,runs} -> results/explore/e31_rl_only/box, runs/
    python scripts/explore_e31.py archive      # slim live copies + the gzipped L1 answers into results/
    python scripts/explore_e31.py summarize    # summary.{md,json} + paper/generated/e31_rl_only.tex

Every number of the readout comes from the run records: the trainer's metrics.jsonl / rollouts.jsonl (training
diagnostics, the GRPO stage's API $ by scripts/train/r5_spend.py's rule), the host's stage minutes (GPU-hours), the
served model's answers to E1's 3,991 decisions (L1, E9's rule), to the Stage-2 common states (C, E5's c_rows) and to
E21's shifted corpus (L2, E21's metrics), and the live shift row (L3, E21's scoring). The reference rows are read from
their logged records (E9's runs, results/stage2/s23, results/explore/e21_shift) and recomputed by the same code."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import explore_e21_shift as E21  # noqa: E402  (E21's corpus metrics, replay, spend and pause tools)

OUT = ROOT / "results/explore/e31_rl_only"
BOX = OUT / "box"
LIVE = OUT / "live"
L1DIR = OUT / "l1"
RUNS = ROOT / "runs"
TEX = ROOT / "paper/generated/e31_rl_only.tex"
CFG = ROOT / "configs/explore/e31_rl_only"
TRAIN_CFG = ROOT / "configs/train/grpo_online_e31_rlonly.yaml"
V1_CFG = ROOT / "configs/train/grpo_online_round5.yaml"
V1_METRICS = ROOT / "results/gpu_round5/grpo_metrics.jsonl"
S23 = ROOT / "results/stage2/s23"
E21_C = ROOT / "results/explore/e21_shift/c"
E21_LIVE = ROOT / "results/explore/e21_shift/live"
S3 = "s3://<bucket>/e31_rl_only"
PREFIX = "e31_"
KEY = "e31"
NAME = "rl-only-e31"
E1_RUN = "e1_bfcl_when2call_e31"
LIVE_RUN, LIVE_ARM = "e31_live_shift", "e31_ts_replace_f"
EVAL_N = 48                      # greedy development tasks per evaluation arm (seeds 1000-1047)
CASCADE_ARM = "rl_cascade"
ITERATIONS = 15
N_BOOT = 2000
LADDER = ("base", "sft_v1", "rl")           # untrained, + SFT v1, SFT + GRPO (AgentDecision-4B v1)
E1_REFS = {"base": "e1_bfcl_when2call_rl/s1_rl_base.jsonl", "sft_v1": "e1_bfcl_when2call_rl/s1_rl_sft_v1.jsonl",
           "rl": "e1_bfcl_when2call_rl/s1_jevagent_rl.jsonl"}
NAMES = {"base": "Qwen3-4B, untrained", KEY: "RL only (E31)", "sft_v1": "+ SFT v1", "rl": "SFT + GRPO (v1)"}
MATCH_L2_PT, MATCH_L3_LO_PT = 3.0, -5.0    # registration (e)
LABEL = ("exploratory (E31), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-10 'E31: GRPO in the agent loop without "
         "SFT (RL only, from the untrained base)'; outside every Holm table")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def jl(p: Path) -> list[dict[str, Any]]:
    return E21.jl(p) if Path(p).exists() else []


# ================================================================================================ training guards
def train_spend(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """scripts/train/r5_spend.py's rule: training rollouts = sum over iterations of episodes x mean episode $; greedy
    evaluations = sum over evaluation rows of EVAL_N x mean episode $ (both from the harness ledger)."""
    import r5_spend

    tr, ev = r5_spend.rollout_cost(list(rows), EVAL_N)
    return {"rollouts": tr, "greedy_evals": ev, "total": tr + ev}


def degenerate(rows: Sequence[Mapping[str, Any]], k: int = 3) -> bool:
    """Registration (c): the last k training iterations updated no decision (no learning signal)."""
    it = [r for r in rows if (r.get("iter") or 0) > 0 and "updated_decisions" in r]
    return len(it) >= k and all(int(r.get("updated_decisions") or 0) == 0 for r in it[-k:])


def _killpg(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        time.sleep(15)


def cmd_trainwatch(a: argparse.Namespace) -> int:
    """Stops the trainer (its process group) when the training spend reaches the cap, the wall cap is reached, or the
    degeneracy rule fires; writes the reason to --stop. Exits when the trainer is gone."""
    t0 = time.time()
    m = Path(a.metrics)
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{_now()} trainer {a.pid} gone; watchdog exits", flush=True)
            return 0
        rows = jl(m)
        sp = train_spend(rows)
        mins = (time.time() - t0) / 60
        reason = None
        if sp["total"] >= a.cap:
            reason = f"training spend cap: ${sp['total']:.2f} >= ${a.cap:.2f}"
        elif mins >= a.max_min:
            reason = f"training wall cap: {mins:.0f} >= {a.max_min} min"
        elif degenerate(rows):
            reason = "degenerate: three consecutive iterations updated no decision"
        last = max([int(r.get("iter") or 0) for r in rows] or [0])
        if reason:
            Path(a.stop).write_text(json.dumps({"at": _now(), "reason": reason, "iteration": last, "spend": sp,
                                                "minutes": round(mins, 1)}, indent=1))
            print(f"{_now()} STOP: {reason} (iteration {last}); trainer stopped", flush=True)
            _killpg(a.pid)
            return 1
        print(f"{_now()} iteration {last}, ${sp['total']:.3f}, {mins:.0f} min", flush=True)
        time.sleep(a.every)


def cmd_trainspend(a: argparse.Namespace) -> int:
    sp = train_spend(jl(Path(a.metrics)))
    if a.out:
        Path(a.out).write_text(json.dumps({**sp, "at": _now(), "rule": "scripts/train/r5_spend.py rollout_cost, "
                                                                       f"eval_n {EVAL_N}"}, indent=1) + "\n")
    print(f"{sp['total']:.4f}")
    return 0


# ================================================================================================ live operations (E21's)
def _extra(a: argparse.Namespace) -> float:
    """Gemini spend outside the e31_* live runs that counts toward the one cap: probes + training (registration (f))."""
    x = E21._probe_usd(Path(a.log)) if getattr(a, "log", None) else 0.0
    tj = getattr(a, "train_spend", None)
    if tj and Path(tj).exists():
        x += float(json.loads(Path(tj).read_text())["total"])
    return x


def cmd_probe(a: argparse.Namespace) -> int:
    E21.PREFIX = PREFIX
    return E21.cmd_probe(argparse.Namespace(log=a.log, dotenv=a.dotenv))


def cmd_spend(a: argparse.Namespace) -> int:
    E21.PREFIX = PREFIX
    s = E21.total_spend(Path(a.runs))
    x = _extra(a)
    if a.short:
        print(f"E31 live runs: Gemini ${s['s2']:.3f} (+ probes and training ${x:.3f}), System-1 ${s['s1']:.3f} in "
              f"{s['records']} records")
    else:
        print(json.dumps({**s, "extra": x}, indent=1))
    return 0


def cmd_errored(a: argparse.Namespace) -> int:
    return E21.cmd_errored(a)


def cmd_watch(a: argparse.Namespace) -> int:
    E21.PREFIX = PREFIX
    return E21.cmd_watch(argparse.Namespace(pid=a.pid, runs=a.runs, cap=a.cap, extra=_extra(a), s1_cap=None,
                                            s1_extra=0.0, s1_arm=None, stop=a.stop, every=a.every,
                                            no_pause=a.no_pause))


def cmd_gate(a: argparse.Namespace) -> int:
    E21.PREFIX = PREFIX
    return E21.cmd_gate(argparse.Namespace(smoke=a.smoke, runs=a.runs, n=a.n, arms=1, cap=a.cap, extra=_extra(a)))


# ================================================================================================ records
def cmd_fetch(a: argparse.Namespace) -> int:
    aws = ["env", "-u", "AWS_ACCESS_KEY_ID", "-u", "AWS_SECRET_ACCESS_KEY", "-u", "AWS_SESSION_TOKEN", "aws", "--region", "us-east-1"]
    env = {**os.environ, "AWS_PAGER": ""}
    subprocess.run(aws + ["s3", "sync", f"{S3}/box", str(BOX), "--only-show-errors", "--exclude", "logs/smi_*",
                          "--exclude", "s23/*", "--exclude", "c_shift/questions*", "--exclude", "c_shift/corpus.json",
                          "--exclude", "c_shift/occurrences.jsonl.gz"], check=True, env=env)
    subprocess.run(aws + ["s3", "sync", f"{S3}/box/s23", str(BOX / "s23"), "--only-show-errors", "--exclude", "*",
                          "--include", "answers_*", "--include", "replay_*"], check=True, env=env)
    subprocess.run(aws + ["s3", "sync", f"{S3}/runs", str(RUNS), "--only-show-errors"], check=True, env=env)
    print(f"-> {BOX}, {RUNS}/e31_*, {RUNS}/{E1_RUN}")
    return 0


def cmd_archive(a: argparse.Namespace) -> int:
    done = []
    for p in sorted(RUNS.glob(f"{PREFIX}*")):
        if not p.is_dir() or not any(p.glob("*.jsonl")):
            continue
        dst = LIVE / f"{p.name}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"run": p.name, "arms": E21.slim_live(p), "spend": E21.run_spend(p), "written": _now()},
                                  indent=1) + "\n")
        done.append(dst)
    src = RUNS / E1_RUN / f"s1_{KEY}.jsonl"
    if src.exists():
        L1DIR.mkdir(parents=True, exist_ok=True)
        dst = L1DIR / f"s1_{KEY}.jsonl.gz"
        with open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(src.read_bytes())
        done.append(dst)
    for f in ("answers_e31.jsonl",):
        for d in (BOX / "s23", BOX / "c_shift"):
            p = d / f
            if p.exists():
                g = p.with_suffix(".jsonl.gz")
                with open(g, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                    gz.write(p.read_bytes())
                p.unlink()
                done.append(g)
    print("\n".join(str(x.relative_to(ROOT)) for x in done))
    return 0


# ================================================================================================ readout
def _answers(d: Path) -> Path | None:
    for n in ("answers_e31.jsonl", "answers_e31.jsonl.gz"):
        if (d / n).exists():
            return d / n
    return None


def training_block(box: Path) -> dict[str, Any]:
    """Registration (c): per-iteration diagnostics and their totals; v1's first 15 iterations beside them."""
    rows = jl(box / "train" / "metrics.jsonl")
    rolls = jl(box / "train" / "rollouts.jsonl.gz") or jl(box / "train" / "rollouts.jsonl")
    it_rows = [r for r in rows if (r.get("iter") or 0) > 0 and "decisions" in r]
    casc: dict[int, int] = defaultdict(int)
    for e in rolls:
        if e.get("arm") == CASCADE_ARM:
            casc[int(e["iter"])] += int(e.get("decisions") or 0)
    per = []
    for r in it_rows:
        it = int(r["iter"])
        dr = r.get("defer_rate")
        deferred = (float(dr) * casc.get(it, 0)) if dr is not None else 0.0
        per.append({"iter": it, "reward": r.get("reward_mean"), "success": r.get("success"),
                    "llm_calls": r.get("llm_calls"), "zero_std": r.get("frac_zero_std_groups"),
                    "decisions": r.get("decisions"), "cascade_decisions": casc.get(it) if rolls else None,
                    "deferred": round(deferred, 1) if rolls else None,
                    "made": (int(r["decisions"]) - deferred) if rolls else None,
                    "updated": r.get("updated_decisions"), "defer_rate": dr,
                    "forced_defer": r.get("cascade_forced_defer_frac"), "dec_acc": r.get("dec_acc"),
                    "dec_acc_greedy": r.get("dec_acc_greedy"), "kl": r.get("kl"), "entropy": r.get("entropy_rollout"),
                    "irreversible": r.get("irreversible_rate"), "errors": r.get("error_rate"),
                    "time_rollout_s": r.get("time_rollout_s"), "time_update_s": r.get("time_update_s"),
                    "arm_success": {k.split("/")[1]: v for k, v in r.items()
                                    if k.startswith("arm/") and k.endswith("/success")}})
    ev = {int(r["iter"]): {k.split("/")[1]: v for k, v in r.items() if k.startswith("eval/") and k.endswith("/success")}
          for r in rows if any(k.startswith("eval/") for k in r)}
    dec = sum(int(p["decisions"] or 0) for p in per)
    made = sum(p["made"] for p in per) if per and all(p["made"] is not None for p in per) else None
    out: dict[str, Any] = {
        "iterations": len(per), "per_iteration": per, "greedy_eval_success": ev,
        "zero_std_share": float(np.mean([p["zero_std"] for p in per if p["zero_std"] is not None])) if per else None,
        "decisions": dec, "decisions_made": made, "decisions_made_share": (made / dec) if made is not None and dec else None,
        "updated_decisions": sum(int(p["updated"] or 0) for p in per),
        "iterations_without_update": sum(1 for p in per if not p["updated"]),
        "success_first3": float(np.mean([p["success"] for p in per[:3]])) if len(per) >= 3 else None,
        "success_last3": float(np.mean([p["success"] for p in per[-3:]])) if len(per) >= 3 else None,
        "dec_acc_first3": float(np.mean([p["dec_acc"] for p in per[:3] if p["dec_acc"] is not None])) if len(per) >= 3 else None,
        "dec_acc_last3": float(np.mean([p["dec_acc"] for p in per[-3:] if p["dec_acc"] is not None])) if len(per) >= 3 else None,
        "spend": train_spend(rows) if rows else None}
    for f in ("train_run.json", "train_stop.json", "host.json", "host_final.json", "smoke.json", "versions.json",
              "selected.json", "adapter_sha256.json"):
        p = box / f
        out[f.split(".")[0]] = json.loads(p.read_text()) if p.exists() else None
    tr = out.get("train_run") or {}
    out["gpu_min"] = tr.get("minutes")
    out["gpu_h"] = (tr["minutes"] / 60) if tr.get("minutes") is not None else None
    # v1's GRPO stage over the same 15 iterations (its metrics only; v1's rollouts.jsonl is not in the repo)
    v1 = [r for r in jl(V1_METRICS) if 0 < (r.get("iter") or 0) <= ITERATIONS and "decisions" in r]
    out["v1_first15"] = {"zero_std_share": float(np.mean([r["frac_zero_std_groups"] for r in v1])) if v1 else None,
                         "decisions": sum(int(r["decisions"]) for r in v1),
                         "updated_decisions": sum(int(r["updated_decisions"]) for r in v1),
                         "success": float(np.mean([r["success"] for r in v1])) if v1 else None,
                         "dec_acc": float(np.mean([r["dec_acc"] for r in v1])) if v1 else None,
                         "defer_rate": float(np.mean([r["defer_rate"] for r in v1 if r.get("defer_rate") is not None]))
                         if v1 else None}
    return out


def level1(runs: Path) -> dict[str, Any]:
    """E1's 3,991 decisions, pooled accuracy by E9's rule (the ids every logged model answered)."""
    import explore_e9_frontier as E9

    p = runs / E1_RUN / f"s1_{KEY}.jsonl"
    if not p.exists():
        p = L1DIR / f"s1_{KEY}.jsonl.gz"
    if not p.exists():
        return {"available": False}
    mine = {r["id"]: r for r in E9.read_jsonl(p)}
    refs = {k: {r["id"]: r for r in E9.read_jsonl(runs / f)} for k, f in E1_REFS.items() if (runs / f).exists()}
    ids = sorted(set(mine).intersection(*[set(v) for v in refs.values()])) if refs else sorted(mine)
    out: dict[str, Any] = {"available": True, "n": len(ids), "n_answers": len(mine),
                           KEY: E9.model_stats([mine[i] for i in ids], "gpu")}
    for k, r in refs.items():
        out[k] = E9.model_stats([r[i] for i in ids], "gpu")
        out[f"vs_{k}"] = E9.paired_gap([E9.is_correct(mine[i]) for i in ids], [E9.is_correct(r[i]) for i in ids])
    return out


def level_c(box: Path) -> dict[str, Any]:
    """Corpus C (Stage 2's common states, 200 tasks): E5's c_rows; paired task bootstrap against the ladder's rows."""
    import ablation_score as AS
    import e5_analysis as EA

    f = _answers(box / "s23")
    if f is None:
        return {"available": False}
    occ, elig = AS.load_corpus(S23)
    ans = {r["qhash"]: r for r in jl(f)}
    res: dict[str, Any] = {"available": True, "occurrences": len(occ), "answers": len(ans),
                           "errors": sum(1 for r in ans.values() if r.get("error"))}
    per = {}
    for m, src in [(KEY, ans)] + [(k, AS.load_answers(S23 / f"answers_{k}.jsonl")) for k in LADDER]:
        cm, pt = EA.c_rows(occ, elig, src)
        res[m] = {k: cm.get(k) for k in ("c_acc", "c_acc_offplan", "c_acc_ask", "c_tasks", "c_missing", "c_errors")}
        per[m] = pt
    for k in LADDER:
        res[f"vs_{k}"] = {s: AS.paired_boot(per[KEY][s], per[k][s], N_BOOT, 0) for s in ("all", "offplan")}
    return res


def l2_view(box: Path) -> Path | None:
    f = _answers(box / "c_shift")
    if f is None:
        return None
    v = RUNS / "e31_l2view"
    v.mkdir(parents=True, exist_ok=True)
    for x in v.iterdir():
        x.unlink()
    for n in ("corpus.json", "occurrences.jsonl.gz", "questions.jsonl.gz"):
        (v / n).symlink_to(E21_C / n)
    for k in LADDER:
        (v / f"answers_{k}.jsonl.gz").symlink_to(E21_C / f"answers_{k}.jsonl.gz")
    (v / f"answers_{KEY}{''.join(f.suffixes)}").symlink_to(f.resolve())
    return v


def level2(box: Path) -> dict[str, Any]:
    import explore_common as EC

    v = l2_view(box)
    if v is None:
        return {"available": False}
    l2 = E21.l2_rows(v)
    ta = l2["taskacc"]
    out: dict[str, Any] = {"available": True, "corpus": l2["corpus"], "rows": {}}
    for k in (KEY,) + LADDER:
        m = (l2["rows"].get(k) or {}).get("main")
        if not m:
            continue
        row = {s: (m.get(s) or {}).get("acc") for s in E21.SUBSETS}
        row.update({"n_tasks": (m.get("all") or {}).get("tasks"), "errors": m.get("errors"), "answers": m.get("answers"),
                    "missing": m.get("missing"), "errors_by_kind": m.get("errors_by_kind"),
                    "mustask_errors_by_kind": m.get("mustask_errors_by_kind")})
        if k == KEY:
            for ref in LADDER:
                for s in ("all", "mustask"):
                    a, b = ta.get(f"{KEY}|main|{s}"), ta.get(f"{ref}|main|{s}")
                    if a and b:
                        common = sorted(set(a) & set(b))
                        r = EC.paired_boot([a[t] for t in common], [b[t] for t in common], n_boot=N_BOOT, seed=0)
                        row[f"vs_{ref}_{s}"] = {"delta": r["est"], "ci": r["ci"], "n_tasks": len(common)}
        out["rows"][k] = row
    return out


def _slim(path: Path, arm: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())["arms"].get(arm)


def level3() -> dict[str, Any]:
    import explore_common as EC

    d = _slim(LIVE / f"{LIVE_RUN}.json", LIVE_ARM)
    if not d:
        return {"available": False}
    refs = {k: _slim(E21_LIVE / f"e21_live_{k}.json", f"{k}_ts_replace_f") for k in LADDER}

    def in_block(x: Mapping[str, Any]) -> list[str]:
        return [t for t in x if E21.L3_SEED0 <= E21.task_seed(t) < E21.L3_SEED0 + E21.L3_N]

    tasks = in_block(d)
    ok = [t for t in tasks if not d[t]["error"]]
    y = {t: float(d[t]["success"]) for t in ok}
    kinds: dict[str, list[float]] = defaultdict(list)
    for t in ok:
        kinds[E21.kind_of_task(t)].append(y[t])
    own = [d[t]["own_acc"] for t in ok if d[t].get("own_acc") is not None]
    out: dict[str, Any] = {
        "available": True, "n": len(ok), "errored": len(tasks) - len(ok),
        "success": float(np.mean(list(y.values()))) if y else None, "own_acc": float(np.mean(own)) if own else None,
        "steps": float(np.mean([d[t]["steps"] for t in ok])) if ok else None,
        "s2_calls": float(np.mean([d[t]["s2_calls"] for t in ok])) if ok else None,
        "api_usd": float(np.mean([d[t]["cost"] for t in ok])) if ok else None,
        "wall_s": float(np.mean([d[t]["wall_s"] for t in ok])) if ok else None,
        "by_kind": {k: {"success": float(np.mean(v)), "n": len(v)} for k, v in sorted(kinds.items())}}
    sp = json.loads((LIVE / f"{LIVE_RUN}.json").read_text()).get("spend") or {}
    out["gemini_usd"] = sp.get("s2")
    for k, r in refs.items():
        if not r:
            continue
        ry = {t: float(r[t]["success"]) for t in in_block(r) if not r[t]["error"]}
        out[f"{k}_success"] = float(np.mean(list(ry.values()))) if ry else None
        common = sorted(set(y) & set(ry))
        if common:
            out[f"vs_{k}"] = {**EC.newcombe([y[t] for t in common], [ry[t] for t in common]), "n": len(common)}
    return out


def v1dev_block(box: Path) -> dict[str, Any]:
    """Diagnostic (development data): eval_decisions accuracy on the three v1 dev files, uncalibrated, as round 5's
    evaluate; references = round 5's own files for the untrained base, SFT v1 and v1 (grpo_i15)."""
    def acc(p: Path) -> dict[str, float] | None:
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return {Path(k).name.split(".")[0]: (v.get("overall") or v).get("acc") for k, v in (d.get("files") or {}).items()}

    ev = ROOT / "results/gpu_round5/eval"
    return {KEY: acc(box / "eval" / "e31_v1.json"), "base": acc(ev / "eval_base_v1.json"),
            "sft_v1": acc(ev / "eval_sft_v1_v1.json"), "rl": acc(ev / "eval_grpo_i15_v1.json")}


def spend_block(tr: Mapping[str, Any], l3: Mapping[str, Any]) -> dict[str, Any]:
    """Direct Gemini spend of E31: training (rollouts + greedy evaluations), the live smoke and row, the probes."""
    live = sum(float((json.loads(p.read_text()).get("spend") or {}).get("s2") or 0) for p in sorted(LIVE.glob("e31_*.json")))
    probes = E21._probe_usd(BOX / "api_probe_log.txt")
    train = float((tr.get("spend") or {}).get("total") or 0)
    return {"training": train, "live": live, "probes": probes, "total": train + live + probes}


def refs_logged() -> dict[str, dict[str, Any]]:
    """The ladder's logged values as the paper reports them (for the cross-check of the recomputed references)."""
    e21 = json.loads((ROOT / "results/explore/e21_shift/summary.json").read_text())
    s2 = json.loads((ROOT / "results/stage2/stage2.json").read_text())["descriptive"]["stage2_rl"]
    out = {}
    for k, s2k in (("base", "base"), ("sft_v1", "sft_v1"), ("rl", "frozen")):
        out[k] = {"c": s2[s2k]["corpus_acc"], "l3": (e21["l3"].get(k) or {}).get("success")}
    return out


def share(x: float | None, base: float | None, sft: float | None) -> float | None:
    if None in (x, base, sft) or sft == base:
        return None
    return (x - base) / (sft - base)


def cmd_summarize(a: argparse.Namespace) -> int:
    tr = training_block(BOX)
    l1 = level1(RUNS)
    c = level_c(BOX)
    l2 = level2(BOX)
    l3 = level3()
    cell: dict[str, dict[str, float | None]] = {}
    for k in (KEY,) + LADDER:
        cell[k] = {"l1": (l1.get(k) or {}).get("acc") if l1.get("available") else None,
                   "c": (c.get(k) or {}).get("c_acc") if c.get("available") else None,
                   "l2": ((l2.get("rows") or {}).get(k) or {}).get("all") if l2.get("available") else None,
                   "mustask": ((l2.get("rows") or {}).get(k) or {}).get("mustask") if l2.get("available") else None,
                   "l3": (l3.get("success") if k == KEY else l3.get(f"{k}_success")) if l3.get("available") else None}
    shares = {m: share(cell[KEY][m], cell["base"][m], cell["sft_v1"][m]) for m in ("l1", "c", "l2", "mustask", "l3")}
    match = None
    if cell[KEY]["l2"] is not None and cell["sft_v1"]["l2"] is not None:
        l2_ok = 100 * (cell[KEY]["l2"] - cell["sft_v1"]["l2"]) >= -MATCH_L2_PT
        l3_ok = True
        if l3.get("available") and l3.get("vs_sft_v1"):
            l3_ok = l3["vs_sft_v1"]["ci_pt"][0] >= MATCH_L3_LO_PT
        match = {"l2_within_3pt": l2_ok, "l3_lower_end_ok": l3_ok if l3.get("available") else None,
                 "matches_sft": bool(l2_ok and l3_ok)}
    res = {"label": LABEL, "generated": _now(), "training": tr, "l1": l1, "c": c, "l2": l2, "l3": l3, "cells": cell,
           "v1dev": v1dev_block(BOX), "spend": spend_block(tr, l3),
           "share_of_sft_gain": shares, "reading": match, "refs_logged": refs_logged()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (OUT / "summary.md").write_text(summary_md(res))
    TEX.write_text(tex(res))
    print((OUT / "summary.md").read_text())
    return 0


def _p(v: Any, d: int = 1) -> str:
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.{d}f}"


def summary_md(r: Mapping[str, Any]) -> str:
    c, tr = r["cells"], r["training"]
    L = ["# E31: GRPO in the agent loop without SFT (RL only, from the untrained Qwen3-4B)", "",
         f"Generated {r['generated']} by `python scripts/explore_e31.py summarize`. {r['label']}.", "",
         "| Row | L1 | C | shift L2 | must-ask | shift L3 |", "|---|---|---|---|---|---|"]
    for k in ("base", KEY, "sft_v1", "rl"):
        x = c[k]
        L.append(f"| {NAMES[k]} | {_p(x['l1'])} | {_p(x['c'])} | {_p(x['l2'])} | {_p(x['mustask'])} | {_p(x['l3'])} |")
    L += ["", "Share of SFT v1's gain over the untrained model that RL alone reaches: "
          + ", ".join(f"{m} {('--' if v is None else f'{100 * v:.0f} %')}" for m, v in r["share_of_sft_gain"].items()),
          f"Reading (registration (e)): {r['reading']}", "", "## Training", ""]
    sp = tr.get("spend") or {}
    L += [f"- iterations {tr['iterations']}; GRPO stage {tr.get('gpu_min')} min; API ${sp.get('total', 0):.2f} "
          f"(rollouts ${sp.get('rollouts', 0):.2f}, greedy evaluations ${sp.get('greedy_evals', 0):.2f})",
          f"- zero-variance group share {_p(tr['zero_std_share'])} % (v1's first 15: {_p(tr['v1_first15']['zero_std_share'])} %)",
          f"- decisions {tr['decisions']}, made by the model {tr['decisions_made']}, updated {tr['updated_decisions']}; "
          f"iterations without an update {tr['iterations_without_update']}",
          f"- rollout success first 3 / last 3 iterations {_p(tr['success_first3'])} / {_p(tr['success_last3'])} %; "
          f"decision accuracy vs the answer key {_p(tr['dec_acc_first3'])} / {_p(tr['dec_acc_last3'])} %",
          f"- greedy development success by iteration: {tr['greedy_eval_success']}", f"- stop record: {tr.get('train_stop')}",
          "", "| it | reward | success | zero-std | decisions | made | updated | defer | dec acc | KL |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in tr["per_iteration"]:
        L.append(f"| {p['iter']} | {p['reward']:.3f} | {_p(p['success'])} | {_p(p['zero_std'])} | {p['decisions']} | "
                 f"{'--' if p['made'] is None else round(p['made'])} | {p['updated']} | {_p(p['defer_rate'])} | "
                 f"{_p(p['dec_acc'])} | {p['kl'] if p['kl'] is None else round(p['kl'], 5)} |")
    if r["l3"].get("available"):
        l3 = r["l3"]
        L += ["", "## L3 shift", "", f"n {l3['n']} (errored {l3['errored']}), success {_p(l3['success'])} %, own-state "
              f"accuracy {_p(l3['own_acc'])} %, Gemini ${l3.get('gemini_usd')}"]
        for k in LADDER:
            if l3.get(f"vs_{k}"):
                v = l3[f"vs_{k}"]
                L.append(f"- vs {NAMES[k]}: {v['diff_pt']:+.1f} [{v['ci_pt'][0]:+.1f}, {v['ci_pt'][1]:+.1f}] (n {v['n']})")
    L += ["", f"Logged reference values (cross-check): {r['refs_logged']}", ""]
    return "\n".join(L)


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {}
    c, tr = r["cells"][KEY], r["training"]
    pre = "e31/rlonly"
    for m, key in (("l1", "l1/acc"), ("c", "c/acc"), ("l2", "l2/all"), ("mustask", "l2/mustask"), ("l3", "l3/success")):
        if c[m] is not None:
            k[f"{pre}/{key}"] = _p(c[m])
    for m, v in r["share_of_sft_gain"].items():
        if v is not None:
            k[f"{pre}/share/{m}"] = f"{100 * v:.0f}"
    if tr.get("gpu_h") is not None:
        k[f"{pre}/gpu_h"] = f"{tr['gpu_h']:.1f}"
        k[f"{pre}/gpu_min"] = f"{tr['gpu_min']:.0f}"
    if tr.get("spend"):
        k[f"{pre}/api_usd"] = f"{tr['spend']['total']:.2f}"
    if tr["iterations"]:
        k[f"{pre}/iterations"] = str(tr["iterations"])
        k[f"{pre}/zero_std_share"] = _p(tr["zero_std_share"])
        k[f"{pre}/decisions"] = f"{tr['decisions']:,}"
        k[f"{pre}/updated"] = f"{tr['updated_decisions']:,}"
        k[f"{pre}/no_update_iters"] = str(tr["iterations_without_update"])
        if tr["decisions_made"] is not None:
            k[f"{pre}/decisions_made"] = f"{round(tr['decisions_made']):,}"
            k[f"{pre}/decisions_made_share"] = _p(tr["decisions_made_share"])
        for f in ("success_first3", "success_last3", "dec_acc_first3", "dec_acc_last3"):
            if tr.get(f) is not None:
                k[f"{pre}/train/{f}"] = _p(tr[f])
        ev = tr["greedy_eval_success"]
        for it, arms in ev.items():
            for arm, v in arms.items():
                if v is not None:
                    k[f"{pre}/eval/{it}/{arm}"] = _p(v)
        v1 = tr["v1_first15"]
        if v1.get("zero_std_share") is not None:
            k[f"{pre}/v1/zero_std_share"] = _p(v1["zero_std_share"])
            k[f"{pre}/v1/updated"] = f"{v1['updated_decisions']:,}"
            k[f"{pre}/v1/train_success"] = _p(v1["success"])
            k[f"{pre}/v1/dec_acc"] = _p(v1["dec_acc"])
    stop = tr.get("train_stop")
    k[f"{pre}/stopped_early"] = "1" if stop else "0"
    if r["l2"].get("available"):
        row = r["l2"]["rows"][KEY]
        for s in ("offplan", "renamed", "newkind", "genkind"):
            if row.get(s) is not None:
                k[f"{pre}/l2/{s}"] = _p(row[s])
        for ref in LADDER:
            v = row.get(f"vs_{ref}_all")
            if v and v.get("delta") is not None:
                k[f"{pre}/l2/vs_{ref}"] = f"{100 * v['delta']:+.1f}"
                k[f"{pre}/l2/vs_{ref}/lo"] = f"{100 * v['ci'][0]:+.1f}"
                k[f"{pre}/l2/vs_{ref}/hi"] = f"{100 * v['ci'][1]:+.1f}"
    if r["l2"].get("available"):                # what the wrong answers were (post hoc, descriptive; E21's categories)
        for row in (KEY, "base"):
            eb = (r["l2"]["rows"].get(row) or {}).get("errors_by_kind") or {}
            if eb:
                tag = pre if row == KEY else "e31/ref/base"
                k[f"{tag}/l2/wrong"] = str(sum(eb.values()))
                k[f"{tag}/l2/wrong_finish"] = str(eb.get("finish", 0))
                k[f"{tag}/l2/wrong_finish_share"] = f"{100 * eb.get('finish', 0) / max(1, sum(eb.values())):.0f}"
    if r["c"].get("available"):
        k[f"{pre}/c/offplan"] = _p(r["c"][KEY]["c_acc_offplan"])
    if r["l3"].get("available"):
        l3 = r["l3"]
        k[f"{pre}/l3/n"] = str(l3["n"])
        k[f"{pre}/l3/own"] = _p(l3["own_acc"])
        if l3.get("gemini_usd") is not None:
            k[f"{pre}/l3/gemini_usd"] = f"{l3['gemini_usd']:.2f}"
        for ref in LADDER:
            v = l3.get(f"vs_{ref}")
            if v:
                k[f"{pre}/l3/vs_{ref}"] = f"{v['diff_pt']:+.1f}"
                k[f"{pre}/l3/vs_{ref}/lo"] = f"{v['ci_pt'][0]:+.1f}"
                k[f"{pre}/l3/vs_{ref}/hi"] = f"{v['ci_pt'][1]:+.1f}"
    if r["reading"] is not None:
        k[f"{pre}/matches_sft"] = "1" if r["reading"]["matches_sft"] else "0"
    for ref in LADDER:                      # the reference rows, recomputed by the same code (= the logged keys)
        for m, key in (("l1", "l1/acc"), ("c", "c/acc"), ("l2", "l2/all"), ("mustask", "l2/mustask"), ("l3", "l3/success")):
            v = r["cells"][ref][m]
            if v is not None:
                k[f"e31/ref/{ref}/{key}"] = _p(v)
    for row, v in (r.get("v1dev") or {}).items():
        if v and v.get("dev") is not None:
            k[(f"{pre}" if row == KEY else f"e31/ref/{row}") + "/v1dev/acc"] = _p(v["dev"])
            k[(f"{pre}" if row == KEY else f"e31/ref/{row}") + "/v1dev/recovery"] = _p(v.get("dev_recovery"))
    sp = r.get("spend") or {}
    if sp:
        k[f"{pre}/gemini_usd"] = f"{sp['total']:.2f}"
        k[f"{pre}/live_usd"] = f"{sp['live']:.2f}"
    if r["c"].get("available"):
        v = r["c"]["vs_base"]["all"]
        k[f"{pre}/c/vs_base"] = f"{100 * v['delta']:+.1f}"
        k[f"{pre}/c/vs_base/lo"] = f"{100 * v['lo']:+.1f}"
        k[f"{pre}/c/vs_base/hi"] = f"{100 * v['hi']:+.1f}"
    l1 = r.get("l1") or {}
    if l1.get("available") and l1.get("vs_base"):
        v = l1["vs_base"]
        k[f"{pre}/l1/vs_base"] = f"{v['est']:+.1f}"              # E9's paired_gap is in points
        k[f"{pre}/l1/vs_base/lo"] = f"{v['ci'][0]:+.1f}"
        k[f"{pre}/l1/vs_base/hi"] = f"{v['ci'][1]:+.1f}"
    if tr.get("per_iteration"):
        k[f"{pre}/kl_max"] = f"{max(p['kl'] or 0 for p in tr['per_iteration']):.2f}"
    tr_run = tr.get("train_run") or {}
    if tr_run.get("peak_gpu_mem_mib"):
        k[f"{pre}/peak_gpu_gb"] = f"{tr_run['peak_gpu_mem_mib'] / 1024:.0f}"
    hf = tr.get("host_final") or {}
    if hf.get("host_hours") is not None:
        k[f"{pre}/host_h"] = f"{float(hf['host_hours']):.1f}"
    return k


def tex(r: Mapping[str, Any]) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    except OSError:
        sha = "?"
    head = ["% AUTO-GENERATED by scripts/explore_e31.py -- do not edit by hand.",
            "% E31: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-10 'E31').",
            f"% generated {r['generated']} at git {sha}", r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [rf"\defres{{{k}}}{{{v}}}" for k, v in sorted(tex_keys(r).items())]
    return "\n".join(head + body + [r"\makeatother", ""])


# ================================================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("trainwatch")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--cap", type=float, default=30.0)
    p.add_argument("--max-min", type=float, default=390.0)
    p.add_argument("--stop", required=True)
    p.add_argument("--every", type=int, default=30)
    p.set_defaults(fn=cmd_trainwatch)
    p = sp.add_parser("trainspend")
    p.add_argument("--metrics", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_trainspend)
    p = sp.add_parser("probe")
    p.add_argument("--log", required=True)
    p.add_argument("--dotenv", action="store_true")
    p.set_defaults(fn=cmd_probe)
    p = sp.add_parser("spend")
    p.add_argument("--runs", default="runs")
    p.add_argument("--log")
    p.add_argument("--train-spend")
    p.add_argument("--short", action="store_true")
    p.set_defaults(fn=cmd_spend)
    p = sp.add_parser("errored")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_errored)
    p = sp.add_parser("watch")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--log")
    p.add_argument("--train-spend")
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--no-pause", action="store_true")
    p.set_defaults(fn=cmd_watch)
    p = sp.add_parser("gate")
    p.add_argument("--smoke", required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--log")
    p.add_argument("--train-spend")
    p.set_defaults(fn=cmd_gate)
    for name, fn in (("fetch", cmd_fetch), ("archive", cmd_archive), ("summarize", cmd_summarize)):
        p = sp.add_parser(name)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
