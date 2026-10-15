#!/usr/bin/env python3
r"""E13 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E13: other agent LLMs on
DecisionGym-Hard"): does the delegation result of DecisionGym-Hard (code name JevGym-Hard) hold when the agent LLM is
not Gemini Flash? E7e's setting unchanged (12 customers, ``tier`` rule, itemized final answer, 60 near distractors,
20 noise lines, strict, ``max_steps`` 45); per agent LLM two arms on a fresh block of 100 evaluation tasks:
``llm_only`` and ``jev_refit`` (E7e's arm: Jev's tool-choice and argument cascades at thresholds refitted for that LLM
by the registered calibration procedure, plus the profile's fixed reflection trigger tau 0.5). No test; outside every
Holm table; nothing here revises a registered result.

    python scripts/explore_e13_llms.py --llm luna config --stage smoke|cal|eval   # print a stage's runner config
    python scripts/explore_e13_llms.py --llm luna run [--until smoke|gate|cal|fit|eval] [--dry-run]
    python scripts/explore_e13_llms.py --llm luna status                         # stages done, spend, gate decisions
    python scripts/explore_e13_llms.py summarize                                 # merged readout of every LLM that ran
    python scripts/explore_e13_llms.py --out results/handoff/e13 summarize --llms deepseek --no-paper
                                                                                 # the collaborator's readout (handoff)

``run`` is resumable: every stage records its completion in ``<out>/<llm>/status.json`` and the runner skips finished
episodes, so a rerun never repeats paid work. Per LLM: (1) smoke on 2 development seeds (``llm_only``, the calibration
arm and a development cascade arm at E7c's Flash thresholds, which exercises the delegated-tool path); an errored
smoke episode, or one without a single tool call, stops the LLM's row; (2) the cost-projection gate (spend so far +
1.2 x the smoke's mean $ per episode x the planned episodes, LLM and Jev dollars against their own caps): calibration
and evaluation on 100 tasks each, else 50 each, else stop after the smoke; (3) the calibration arm (Jev in shadow
mode) on the LLM's calibration seeds; (4) the registered fit (``explore_e7_hard.fit_calibration``: threshold_theory
--calibration-arm, W = $1, lambda = 0, V fallback W); (5) the gate again with the calibration's actual spend; (6) the
evaluation (``JEV_EVAL_RUNS=1``; passes 2-3 re-run infrastructure errors only); (7) slimmed, gzipped copies of every
record in ``<out>/records/<llm>/`` (states, observations and decision questions dropped, System-1 distributions over
tools cut to their top 5: nothing the readout or a refit reads). A watchdog stops the runner when the LLM's logged LLM
or Jev spend, plus a margin for the episodes in flight, reaches its cap. ``--dry-run``: the simulated LLM and System-1
on development seeds, in a scratch directory, $0 (pipeline test of every stage).

Readout (``summarize``; registered in the Deviations entry): per LLM the paired complete case of ``jev_refit`` against
``llm_only``: success and Delta in points (paired task bootstrap, B = 10,000, seed 0; Newcombe beside it), wins and
losses, cost per 1,000 episodes and the saving 1 - C_arm / C_ref (task bootstrap), mean and median wall time, the share
of tool decisions delegated (coverage) and the accuracy of the accepted decisions (and of all, and on the tier rule's
cancel-vs-ship decisions), the reflection trigger's firing, failure types, final-answer forms, the calibration fits,
and spend against the caps -- computed by ``explore_e7_hard.summarize`` on the slimmed records. Keys ``e13/<llm>/...``
in ``paper/generated/e13_llms.tex``; the collaborator's output (``results/handoff/e13/``) is merged when present."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402
import explore_e7_hard as E7  # noqa: E402
import module_map as MM  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
OUT = ROOT / "results/explore/e13"
HANDOFF_OUT = ROOT / "results/handoff/e13"
TEX = ROOT / "paper/generated/e13_llms.tex"
E7E_CONFIG = ROOT / "configs/explore/e7_hard/eval_e7e.yaml"
E7E_SUMMARY = ROOT / "results/explore/e7_hard/summary_e7e.json"
PREREG = ROOT / "docs/PREREG_A7_STAGE2.md"
ENTRY = "E13: other agent LLMs on DecisionGym-Hard"
LABEL = "exploratory (E13)"

# E7e's setting, unchanged (configs/explore/e7_hard/eval_e7e.yaml; tested against it)
SETTING = {"families": "multi", "multi_customers": 12, "multi_rule": "tier", "answer_style": "itemized",
           "distractor_style": "near", "n_distractors": 60, "noise_lines": 20, "strict": True}
MAX_STEPS = 45
SCHEDULE = {"mode": "task_major", "seed": 732}
JEV = {"type": "jev", "provider": "openrouter"}       # the pinned Jev endpoint; no System-1 cache (fresh answers)
REFLECTION = {"mode": "trigger", "tau": 0.5}          # the profile's fixed reflection trigger (not refitted)
DEV_TAU = (0.91, 0.87)                                # E7c's Flash fit: the smoke's development cascade arm only
N_SMOKE, N_FULL, N_HALF = 2, 100, 50
TIERS = ((N_FULL, N_FULL), (N_HALF, N_HALF))          # (calibration tasks, evaluation tasks), first that fits
SAFETY = 1.2                                          # E7c's projection factor
REF, ARM, CAL_ARM, DEV_ARM = "llm_only", "jev_refit", "jev_shadow_cal", "jev_cascade_dev"
EVAL_ARMS = (REF, ARM)
STAGES = ("smoke", "gate", "cal", "fit", "eval", "slim")
PASSES = 3
MIN_FREE_MB = 150          # the runner is stopped (and the stage left unfinished) when the disk gets this full: a record
                           # that cannot be written would be paid work lost (2026-10-07, Luna calibration)
# dry run: development seeds only (never an evaluation, calibration or smoke seed of E13), simulated models
DRY = {"seeds": {"smoke": 0, "cal": 2, "eval": 30}, "n": {"smoke": N_SMOKE, "cal": 20, "eval": 20},
       "llm": {"provider": "simulated", "error_rate": 0.1, "simulated_latency_ms": 0, "seed": 0},
       "s1": {"type": "simulated", "accuracy": 0.9, "latency_ms": 0.0}}
if os.environ.get("E13_DRY_N"):          # tests: smaller dry runs, "smoke,cal,eval" task counts
    DRY["n"] = dict(zip(("smoke", "cal", "eval"), (int(x) for x in os.environ["E13_DRY_N"].split(","))))

OPENROUTER_PIN = {"allow_fallbacks": False}
LLMS: dict[str, dict[str, Any]] = {
    # (a) run locally on the author's keys: OpenAI directly; Jev via OpenRouter with the Jev-only key
    "luna": {"title": "GPT-6-Luna", "model_id": "gpt-6-luna", "runner": "local",
             "llm": {"provider": "openai_responses", "model": "gpt-6-luna", "reasoning_effort": "none",
                     "temperature": 0},
             "route": "OpenAI Responses API directly (reasoning effort none, its lowest; temperature 0; stateless); "
                      "Jev via OpenRouter on the author's Jev-only key",
             "seeds": {"smoke": 7276, "cal": 5400, "eval": 311000},
             "caps": {"llm": 60.0, "jev": 5.0}, "workers": {"smoke": 6, "cal": 4, "eval": 6}},
    # (b), (c): run by a collaborator on their own keys (handoff/experiments/e13/); date in the execution record
    "deepseek": {"title": "DeepSeek-V4-Flash", "model_id": "deepseek/deepseek-v4-flash", "runner": "collaborator",
                 "llm": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash", "temperature": 0,
                         "price_model": "deepseek/deepseek-v4-flash@streamlake-2026-10-07",
                         "extra_body": {"provider": {"order": ["StreamLake"], **OPENROUTER_PIN},
                                        "reasoning": {"enabled": False}, "usage": {"include": True}}},
                 "route": "OpenRouter pinned to StreamLake (fp8), fallbacks off, reasoning off; Jev via OpenRouter",
                 "pin": ("StreamLake", 0.042e-6, 0.084e-6),
                 "seeds": {"smoke": 7278, "cal": 5500, "eval": 311100},
                 "caps": {"llm": 15.0, "jev": 5.0}, "workers": {"smoke": 6, "cal": 8, "eval": 8}},
}
NAN = float("nan")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_name(llm: str, stage: str, dry: bool = False) -> str:
    return f"e13{'dry' if dry else ''}_{llm}_{stage}"


def host_tag() -> str:
    """A short, non-reversible host tag (as the handoff framework's branch names): no hostname is published."""
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]


# ============================================================================== configs
def eval_arm(fits: Mapping[str, Any] | None = None, taus: tuple[float, float] | None = None) -> dict[str, Any]:
    """``jev_refit`` at the fitted thresholds (E7e's structure): tau-hat = +inf at tool choice -> ``always_defer``
    (System-1 not asked, A6.11 (c)); at the arguments -> tau = +inf (asked, never accepted), as E7c registered."""
    if taus is None:
        f = (fits or {}).get(CAL_ARM) or {}
        taus = tuple(_tau_of(f.get(dp) or {}) for dp in E7.TOOL_DPS)       # type: ignore[assignment]
    ts, tp = taus
    if any(t is None or (isinstance(t, float) and math.isnan(t)) for t in (ts, tp)):
        raise SystemExit(f"no fitted threshold for {CAL_ARM} ({taus}): the fit failed, the evaluation does not run")
    tool_select = ({"mode": "cascade", "tau": 1.0, "always_defer": True} if ts == math.inf
                   else {"mode": "cascade", "tau": float(ts)})
    tool_params = {"mode": "cascade", "tau": float(tp)}
    return {"name": ARM, "tool_select": tool_select, "tool_params": tool_params, "reflection": dict(REFLECTION)}


def _tau_of(row: Mapping[str, Any]) -> float | None:
    if row.get("always_defer"):
        return math.inf
    t = row.get("tau_hat")
    return None if t is None else float(t)


def stage_config(llm: str, stage: str, n: int | None = None, fits: Mapping[str, Any] | None = None,
                 dry: bool = False, taus: tuple[float, float] | None = None) -> dict[str, Any]:
    """The runner config of one stage. smoke: ``llm_only``, the calibration arm and the development cascade arm on 2
    development seeds; cal: the calibration arm (the LLM acts alone, Jev answers every tool-choice and argument question
    in shadow mode, never executed); eval: ``llm_only`` and ``jev_refit`` at the fitted thresholds."""
    L = LLMS[llm]
    n = n if n is not None else (DRY["n"][stage] if dry else {"smoke": N_SMOKE, "cal": N_FULL, "eval": N_FULL}[stage])
    seed0 = DRY["seeds"][stage] if dry else L["seeds"][stage]
    llm_block = ({**DRY["llm"], "model": f"sim-{llm}"} if dry else copy.deepcopy(L["llm"]))
    shadow = {"name": CAL_ARM, "tool_select": {"mode": "shadow"}, "tool_params": {"mode": "shadow"}}
    if stage == "smoke":
        arms = [{"name": REF}, shadow, {"name": DEV_ARM, "tool_select": {"mode": "cascade", "tau": DEV_TAU[0]},
                                        "tool_params": {"mode": "cascade", "tau": DEV_TAU[1]},
                                        "reflection": dict(REFLECTION)}]
    elif stage == "cal":
        arms = [shadow]
    elif stage == "eval":
        arms = [{"name": REF}, eval_arm(fits, taus)]
    else:
        raise ValueError(stage)
    return {"name": run_name(llm, stage, dry),
            "suite": {"name": "jevgym", "n": int(n), "seed0": int(seed0), **SETTING},
            "n_tasks": int(n), "llm": llm_block, "harness_defaults": {"max_steps": MAX_STEPS},
            "s1": dict(DRY["s1"]) if dry else dict(JEV), "workers": int(L["workers"][stage]),
            "schedule": dict(SCHEDULE), "arms": arms}


# ============================================================================== records: spend, slim copies
def _run_dirs(llm: str, runs: Path, out: Path, dry: bool = False) -> dict[str, Path]:
    """stage -> the directory holding its records: the raw run while it exists, else the slimmed copy."""
    got = {}
    for st in ("smoke", "cal", "eval"):
        raw = runs / run_name(llm, st, dry)
        slim = out / "records" / llm / run_name(llm, st, dry)
        got[st] = raw if any(raw.glob("*.jsonl")) else slim
    return got


def spend(llm: str, runs: Path, out: Path, dry: bool = False) -> dict[str, Any]:
    """Logged $ of every record of the LLM's runs (errored attempts and re-runs included: all were paid): ``llm`` =
    System-2 (the agent LLM), ``jev`` = System-1 including the shadow answers of the calibration arm."""
    per = {}
    for st, d in _run_dirs(llm, runs, out, dry).items():
        s = E7.run_spend(d) if d.is_dir() else {"s2": 0.0, "s1": 0.0, "total": 0.0, "records": 0}
        per[st] = {"llm": s["s2"], "jev": s["s1"], "total": s["total"], "records": s["records"]}
    tot = {k: sum(v[k] for v in per.values()) for k in ("llm", "jev", "total", "records")}
    return {"by_stage": per, **tot}


def slim_record(t: Mapping[str, Any]) -> dict[str, Any]:
    """A record without what no E13 readout or refit reads: step observations, the decision states (the System-1
    input) and questions; System-1 distributions over tools cut to their top 5 (the fit and the readout read
    ``s1_value`` / ``s1_conf``; yes/no distributions are kept whole)."""
    out = dict(t)
    steps = []
    for st in t.get("steps") or []:
        st = {k: v for k, v in st.items() if k != "observation"}
        decs = []
        for d in st.get("decisions") or []:
            d = {k: v for k, v in d.items() if k not in ("state", "question")}
            dist = d.get("s1_dist")
            if d.get("options") and isinstance(dist, Mapping) and len(dist) > 5:
                d["s1_dist"] = dict(sorted(dist.items(), key=lambda kv: -float(kv[1] or 0))[:5])
            decs.append(d)
        st["decisions"] = decs
        steps.append(st)
    out["steps"] = steps
    out["slim"] = "e13"
    return out


def slim_run(raw: Path, dest: Path) -> dict[str, Any]:
    """Every arm file of a run as ``<arm>.jsonl.gz`` (slimmed, every record kept, re-runs included), plus the
    run's experiment.yaml and schedule records; returns sizes and the raw files' sha256."""
    dest.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"raw": C.rel(raw), "files": {}}
    for f in sorted(raw.glob("*.jsonl")):
        if f.name == "schedule_order.jsonl":
            (dest / "schedule_order.jsonl.gz").write_bytes(gzip.compress(f.read_bytes(), 9))
            continue
        lines = []
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    lines.append(json.dumps(slim_record(json.loads(line))))
                except ValueError:
                    continue          # a partial line of a killed run (its task was re-run)
        (dest / f"{f.name}.gz").write_bytes(gzip.compress(("\n".join(lines) + "\n").encode(), 9))
        info["files"][f.name] = {"raw_sha256": C.sha256_file(f), "raw_bytes": f.stat().st_size,
                                 "gz_bytes": (dest / f"{f.name}.gz").stat().st_size, "records": len(lines)}
    for name in ("experiment.yaml", "schedule.json"):
        if (raw / name).exists():
            shutil.copy2(raw / name, dest / name)
    (dest / "slim.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def scrub_paths(dirs: Sequence[Path], runs: Path, out: Path | None = None) -> list[str]:
    """Replace this machine's paths (the runs directory, the checkout, a personal home) by placeholders in the text
    files the readout keeps (JSON, Markdown, YAML, TeX): nothing published names a local path (the handoff's
    publish.sh refuses otherwise). Gzipped records hold no local path."""
    subs = [(str(Path(runs).resolve()), "<runs>")] + ([(str(Path(out).resolve()), "<out>")] if out else []) + [
        (str(ROOT), "<repo>")]
    home = os.path.expanduser("~")
    if home not in ("/", "/root", "/home", "/home/"):
        subs.append((home, "~"))
    changed = []
    for d in dirs:
        if not Path(d).exists():
            continue
        for f in ([Path(d)] if Path(d).is_file() else sorted(Path(d).rglob("*"))):
            if f.is_file() and f.suffix in (".json", ".md", ".yaml", ".txt", ".tex"):
                text = f.read_text()
                new = text
                for a, b in subs:
                    new = new.replace(a, b)
                if new != text:
                    f.write_text(new)
                    changed.append(C.rel(f))
    return changed


# ============================================================================== status, gate
class Status:
    """``<out>/<llm>/status.json``: what has run, so a rerun resumes."""

    def __init__(self, path: Path):
        self.path = path
        self.d: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, k: str, default: Any = None) -> Any:
        return self.d.get(k, default)

    def set(self, k: str, v: Any) -> None:
        self.d[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(MM._clean(self.d), indent=1, default=MM._json_default) + "\n")


def smoke_costs(run_dir: Path, arms: Sequence[str] = (REF, CAL_ARM, DEV_ARM)) -> dict[str, Any]:
    """Per smoke arm: episodes, errors, episodes without a single tool call, mean $ per episode (LLM / Jev) and wall."""
    out = {}
    for arm in arms:
        p = E7.arm_file(run_dir, arm)
        ts = list(C.load_latest([p])[0].values()) if p.exists() else []
        sp = [E7.episode_spend(t) for t in ts]
        out[arm] = {"n": len(ts), "errors": sum(bool(t.get("error")) for t in ts),
                    "no_tool_call": sum(not any(st.get("tool") for st in t.get("steps") or []) for t in ts),
                    "no_answer": sum(t.get("final_answer") is None for t in ts),
                    "llm": float(np.mean([s["s2"] for s in sp])) if sp else None,
                    "jev": float(np.mean([s["s1"] for s in sp])) if sp else None,
                    "wall_s": float(np.mean([float(t.get("wall_ms") or 0) / 1000 for t in ts])) if ts else None,
                    "retried_calls": sum(max(int(c.get("attempts") or 1) - 1, 0) for t in ts for c in t.get("calls") or []),
                    "served_providers": sorted({str((c.get("meta") or {}).get("served_provider")) for t in ts
                                                for c in t.get("calls") or [] if c.get("component") == "s2"})}
    return out


def gate(llm: str, costs: Mapping[str, Any], spent: Mapping[str, float], phase: str = "pre_cal",
         n_cal_done: int = 0, max_eval: int = N_FULL, caps: Mapping[str, float] | None = None,
         budget: float | None = None, check_pin: bool = False) -> dict[str, Any]:
    """The registered projection gate. ``pre_cal``: spent + SAFETY x [n_cal x c(calibration arm) + n_eval x (c(llm_only)
    + c(cascade arm))] for the first tier of TIERS that fits both caps (LLM and Jev dollars apart), else stop after the
    smoke. ``pre_eval``: the calibration has run (its actual spend is in ``spent``); spent + SAFETY x n_eval x
    (c(llm_only) + c(cascade arm)) for n_eval in the tiers not above ``max_eval``. A smoke arm with an error or an
    episode without a single tool call stops the row (no silent patch: the cause is reported); so does a System-2 call
    served by another provider than the pinned one (``check_pin``, OpenRouter rows). ``budget``: what the key may still
    spend (the handoff's key-usage cap); a tier's new spend (LLM + Jev) must fit it too."""
    caps = dict(caps or LLMS[llm]["caps"])
    out: dict[str, Any] = {"phase": phase, "at": utc(), "safety": SAFETY, "caps": caps, "spent": dict(spent),
                           "budget": budget, "smoke": costs}
    pin = LLMS[llm].get("pin")
    if check_pin and pin:
        other = sorted({p for c in costs.values() for p in c.get("served_providers") or []} - {pin[0]})
        if other:
            out.update({"decision": f"stop: smoke calls served by {other}, not the pinned {pin[0]} (check the account's "
                                    "provider settings on openrouter.ai)", "n_cal": 0, "n_eval": 0})
            return out
    bad = {a: c for a, c in costs.items() if c["n"] < N_SMOKE or c["errors"] or c["no_tool_call"]}
    if bad:
        out["decision"] = "stop: smoke " + "; ".join(
            f"{a}: {c['n']} episodes, {c['errors']} errored, {c['no_tool_call']} without a tool call" for a, c in bad.items())
        out["n_cal"] = out["n_eval"] = 0
        return out
    c = {k: {a: float(costs[a][k] or 0.0) for a in costs} for k in ("llm", "jev")}
    tiers = (TIERS if phase == "pre_cal" else tuple((n_cal_done, ne) for _, ne in TIERS if ne <= max_eval))
    out["tiers"] = []
    for nc, ne in tiers:
        new_cal = nc if phase == "pre_cal" else 0
        proj = {k: spent[k] + SAFETY * (new_cal * c[k][CAL_ARM] + ne * (c[k][REF] + c[k][DEV_ARM])) for k in ("llm", "jev")}
        new = sum(proj[k] - spent[k] for k in ("llm", "jev"))
        fits = all(proj[k] <= caps[k] for k in ("llm", "jev")) and (budget is None or new <= budget)
        out["tiers"].append({"n_cal": nc, "n_eval": ne, "projected": proj, "new_spend": new, "fits": fits})
        if fits and "decision" not in out:
            out.update({"decision": f"run: calibration {nc}, evaluation {ne}", "n_cal": nc, "n_eval": ne})
    if "decision" not in out:
        out.update({"decision": "stop after the smoke" if phase == "pre_cal" else "stop after the calibration",
                    "n_cal": n_cal_done if phase != "pre_cal" else 0, "n_eval": 0})
    return out


# ============================================================================== running a stage
def check_registered() -> None:
    """The E13 entry must be committed (HEAD's copy of the registration names it) before any paid E13 episode."""
    norm = lambda s: re.sub(r"\s+", " ", s)  # noqa: E731
    have = norm(PREREG.read_text()) if PREREG.exists() else ""
    try:
        head = norm(subprocess.run(["git", "show", "HEAD:docs/PREREG_A7_STAGE2.md"], cwd=ROOT, capture_output=True,
                                   text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        head = ""
    if ENTRY not in have or ENTRY not in head:
        raise SystemExit(f"refusing to run E13: the Deviations entry \"{ENTRY}\" is not committed in "
                         "docs/PREREG_A7_STAGE2.md (register before any episode)")


def _launch(cfg: Mapping[str, Any], cfg_path: Path, runs: Path, env_extra: Mapping[str, str], llm: str, out: Path,
            caps: Mapping[str, float], margin: Mapping[str, float], dry: bool, every: float) -> dict[str, Any]:
    """One runner invocation (its own process), watched: when the LLM's logged LLM or Jev spend plus ``margin`` (the
    episodes in flight) reaches its cap, the runner is stopped and ``<out>/<llm>/STOP.json`` written."""
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(dict(cfg), sort_keys=False))
    env = {**os.environ, **env_extra}
    env.pop("JEV_ALLOW_PENDING_TAU", None)
    t0 = utc()
    proc = subprocess.Popen([sys.executable, "-m", "jevharness.bench.runner", str(cfg_path), "--out", str(runs)],
                            cwd=ROOT, env=env)
    capped = disk = None
    while proc.poll() is None:
        time.sleep(every)
        free_mb = shutil.disk_usage(runs).free / 2 ** 20
        if free_mb < MIN_FREE_MB:
            disk = {"at": utc(), "free_mb": round(free_mb, 1)}
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"[e13:{llm}] DISK: {free_mb:.0f} MB free; runner stopped (rerun to resume)", flush=True)
            break
        s = spend(llm, runs, out, dry)
        over = [k for k in ("llm", "jev") if s[k] + margin.get(k, 0.0) >= caps[k]]
        if over:
            capped = {"at": utc(), "over": over, "spend": s, "margin": dict(margin), "caps": dict(caps)}
            (out / llm / "STOP.json").write_text(json.dumps(capped, indent=1) + "\n")
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"[e13:{llm}] CAP: {over} spend {s['llm']:.2f} / {s['jev']:.3f} + margin reached the cap; runner "
                  "stopped", flush=True)
            break
    return {"start": t0, "end": utc(), "returncode": proc.returncode, "capped": capped, "disk": disk}


def _errors(run_dir: Path, arms: Sequence[str]) -> dict[str, int]:
    out = {}
    for a in arms:
        p = E7.arm_file(run_dir, a)
        out[a] = sum(bool(t.get("error")) for t in C.load_latest([p])[0].values()) if p.exists() else 0
    return out


def run_stage(llm: str, stage: str, cfg: Mapping[str, Any], runs: Path, out: Path, st: Status, dry: bool,
              passes: int, margin: Mapping[str, float], every: float) -> dict[str, Any]:
    """Up to ``passes`` invocations (pass 1, then re-runs of infrastructure errors only); stops early on a cap."""
    caps = LLMS[llm]["caps"]
    env_extra = {"JEV_EVAL_RUNS": "1"} if stage == "eval" and not dry else {}
    rec = st.get(stage) or {"invocations": []}
    run_dir = runs / cfg["name"]
    arms = [a["name"] for a in cfg["arms"]]
    for _ in range(passes):
        s = spend(llm, runs, out, dry)
        if any(s[k] >= caps[k] for k in ("llm", "jev")):
            rec["capped_before_start"] = {"at": utc(), "spend": s}
            break
        inv = _launch(cfg, out / llm / "configs" / f"{stage}.yaml", runs, env_extra, llm, out, caps, margin, dry, every)
        inv["errors_after"] = _errors(run_dir, arms)
        rec["invocations"].append(inv)
        st.set(stage, rec)
        if inv["capped"]:
            break
        if inv["disk"] or inv["returncode"] not in (0, None):
            # the runner died (e.g. a record could not be written) or was stopped for disk: the stage is NOT finished;
            # a rerun resumes it (finished episodes are skipped)
            raise SystemExit(f"[e13:{llm}] {stage}: the runner stopped (exit {inv['returncode']}"
                             + (f", disk {inv['disk']}" if inv["disk"] else "") + "); rerun to resume")
        if not any(inv["errors_after"].values()):
            break
    rec["errors"] = _errors(run_dir, arms)
    rec["n_records"] = {a: (len(C.load_latest([E7.arm_file(run_dir, a)])[0]) if E7.arm_file(run_dir, a).exists() else 0)
                        for a in arms}
    rec["capped"] = any(i.get("capped") for i in rec["invocations"]) or "capped_before_start" in rec
    rec["done"] = utc()
    st.set(stage, rec)
    return rec


def _n(n: int, stage: str, dry: bool) -> int:
    """Tasks of a stage: the gate's n; a dry run scales its own small n by the same tier (100 -> all, 50 -> half)."""
    return max(1, DRY["n"][stage] * int(n) // N_FULL) if dry else int(n)


def _margin(costs: Mapping[str, Any], workers: int) -> dict[str, float]:
    return {k: workers * max(float(c.get(k) or 0.0) for c in costs.values()) for k in ("llm", "jev")}


def fit(llm: str, runs: Path, out: Path, n_cal: int, dry: bool) -> dict[str, Any]:
    """The registered estimator on the LLM's calibration run (``explore_e7_hard.fit_calibration``)."""
    res = E7.fit_calibration(runs, out / f"cal_{llm}", run_name=run_name(llm, "cal", dry), cal_arms={CAL_ARM: ARM},
                             tex=None)
    res["llm"], res["n_cal"], res["entry"] = llm, n_cal, ENTRY
    md = E7.calibration_markdown(res).replace(
        "# E7c calibration fit (JevGym-Hard, 12 customers; calibration seeds 5300-5399)",
        f"# E13 calibration fit, {LLMS[llm]['title']} (DecisionGym-Hard as E7e; calibration seeds "
        f"{DRY['seeds']['cal'] if dry else LLMS[llm]['seeds']['cal']}+, n = {n_cal})", 1)
    (out / f"cal_{llm}" / "fits.md").write_text(md)
    (out / f"cal_{llm}" / "fits.json").write_text(json.dumps(MM._clean(res), indent=1, default=MM._json_default) + "\n")
    return res


def run(llm: str, runs: Path = RUNS, out: Path = OUT, until: str = "slim", dry: bool = False, passes: int = PASSES,
        every: float = 20.0, delete_raw: bool = False, budget: float | None = None) -> dict[str, Any]:
    """Every stage of one LLM's row, resumable (see the module docstring); local paths are scrubbed from what it
    keeps, whichever stage it ends at."""
    if not dry:
        check_registered()
    try:
        return _run_stages(llm, runs, out, until, dry, passes, every, delete_raw, budget)
    finally:
        scrub_paths([out / llm, out / f"cal_{llm}", out / "records" / llm], runs, out)


def _run_stages(llm: str, runs: Path, out: Path, until: str, dry: bool, passes: int, every: float,
                delete_raw: bool, budget: float | None) -> dict[str, Any]:
    L = LLMS[llm]
    box = out / llm
    box.mkdir(parents=True, exist_ok=True)
    st = Status(box / "status.json")
    if not st.get("started"):
        st.set("started", {"at": utc(), "git": C.git_head(), "host": host_tag(), "runner": L["runner"], "dry": dry,
                           "model": L["model_id"], "route": L["route"]})
    stop = st.get("stopped")
    if stop:
        print(f"[e13:{llm}] stopped earlier: {stop}", flush=True)
        return st.d
    # (1) smoke
    if not (st.get("smoke") or {}).get("done"):
        cfg = stage_config(llm, "smoke", dry=dry)
        run_stage(llm, "smoke", cfg, runs, out, st, dry, passes, {"llm": 0.0, "jev": 0.0}, every)
    if until == "smoke":
        return st.d
    costs = smoke_costs(runs / run_name(llm, "smoke", dry))
    if any(c["n"] == 0 for c in costs.values()) and (out / "records" / llm / run_name(llm, "smoke", dry)).exists():
        costs = smoke_costs(out / "records" / llm / run_name(llm, "smoke", dry))
    # (2) the gate before the calibration (decided once; reused on resume)
    g = st.get("gate")
    if not g:
        g = gate(llm, costs, spend(llm, runs, out, dry), "pre_cal", budget=budget, check_pin=not dry)
        st.set("gate", g)
        (box / "projection.json").write_text(json.dumps(MM._clean(g), indent=1, default=MM._json_default) + "\n")
    print(f"[e13:{llm}] gate: {g['decision']}", flush=True)
    if not g.get("n_cal"):
        st.set("stopped", {"at": utc(), "why": g["decision"]})
        return st.d
    if until == "gate":
        return st.d
    # (3) calibration
    margin = _margin(costs, L["workers"]["cal"])
    if not (st.get("cal") or {}).get("done"):
        cfg = stage_config(llm, "cal", n=_n(g["n_cal"], "cal", dry), dry=dry)
        rec = run_stage(llm, "cal", cfg, runs, out, st, dry, passes, margin, every)
        if rec["capped"]:
            st.set("stopped", {"at": utc(), "why": "cap reached during the calibration"})
            return st.d
    if until == "cal":
        return st.d
    # (4) fit
    fits = st.get("fit")
    if not fits:
        res = fit(llm, runs, out, _n(g["n_cal"], "cal", dry), dry)
        try:
            arm = eval_arm(res["fits"])
        except SystemExit as e:
            st.set("stopped", {"at": utc(), "why": str(e)})
            return st.d
        fits = {"at": utc(), "taus": {dp: _tau_of(res["fits"][CAL_ARM][dp]) for dp in E7.TOOL_DPS}, "arm": arm,
                "path": C.rel(out / f"cal_{llm}" / "fits.json")}
        st.set("fit", fits)
    print(f"[e13:{llm}] fitted: {fits['taus']}", flush=True)
    if until == "fit":
        return st.d
    # (5) the gate before the evaluation, with the calibration's actual spend
    ge = st.get("gate_eval")
    if not ge:
        ge = gate(llm, costs, spend(llm, runs, out, dry), "pre_eval", n_cal_done=g["n_cal"], max_eval=g["n_eval"],
                  budget=budget, check_pin=not dry)
        st.set("gate_eval", ge)
        (box / "projection_eval.json").write_text(json.dumps(MM._clean(ge), indent=1, default=MM._json_default) + "\n")
    print(f"[e13:{llm}] gate before the evaluation: {ge['decision']}", flush=True)
    if not ge.get("n_eval"):
        st.set("stopped", {"at": utc(), "why": ge["decision"]})
        return st.d
    # (6) evaluation
    if not (st.get("eval") or {}).get("done"):
        cfg = stage_config(llm, "eval", n=_n(ge["n_eval"], "eval", dry), dry=dry,
                           taus=(fits["taus"]["tool_select"], fits["taus"]["tool_params"]))
        rec = run_stage(llm, "eval", cfg, runs, out, st, dry, passes, _margin(costs, L["workers"]["eval"]), every)
        if rec["capped"]:
            st.set("stopped", {"at": utc(), "why": "cap reached during the evaluation"})
    if until == "eval":
        return st.d
    # (7) slimmed, gzipped records (every stage), then the raw runs may go
    if not (st.get("slim") or {}).get("done"):
        info = {}
        for stg in ("smoke", "cal", "eval"):
            raw = runs / run_name(llm, stg, dry)
            if any(raw.glob("*.jsonl")):
                info[stg] = slim_run(raw, out / "records" / llm / run_name(llm, stg, dry))
        st.set("slim", {"done": utc(), "runs": info})
    if delete_raw:
        for stg in ("smoke", "cal", "eval"):
            raw = runs / run_name(llm, stg, dry)
            if (out / "records" / llm / run_name(llm, stg, dry) / "slim.json").exists() and raw.is_dir():
                shutil.rmtree(raw)
    st.set("spend", spend(llm, runs, out, dry))
    st.set("finished", utc())
    return st.d


# ============================================================================== readout
def _is_dry(llm: str, src: Path) -> bool:
    """Records of a dry run (``e13dry_*``; the pipeline test) and no real ones."""
    rec = src / "records" / llm
    return (not (rec / run_name(llm, "eval") / "experiment.yaml").exists()
            and (rec / run_name(llm, "eval", True) / "experiment.yaml").exists())


def _sources(out: Path, handoff: Path | None) -> dict[str, Path]:
    """llm -> the directory with its records (``<dir>/records/<llm>/``): this readout's own output first, then the
    collaborator's handoff output."""
    got = {}
    for llm in LLMS:
        for d in [out] + ([handoff] if handoff else []):
            if d and any((d / "records" / llm / run_name(llm, "eval", dry) / "experiment.yaml").exists()
                         for dry in (False, True)):
                got[llm] = d
                break
    return got


def _exp_key(llm: str, src: Path) -> str:
    """Register the LLM's row with ``explore_e7_hard`` (its EXPS table) so that E7's readout runs on it unchanged."""
    L = LLMS[llm]
    key = f"e13_{llm}"
    dry = _is_dry(llm, src)
    E7.EXPS[key] = {"smoke": run_name(llm, "smoke", dry), "eval": run_name(llm, "eval", dry), "box": src / llm,
                    "tex": TEX,
                    "summary": "summary", "out": src, "cap": L["caps"]["llm"] + L["caps"]["jev"], "stop": "STOP.json",
                    "prefix": key, "title": f"E13: {L['title']} on DecisionGym-Hard", "entry": ENTRY,
                    "arms": EVAL_ARMS, "gpu_arms": (), "extra_runs": (run_name(llm, "cal", dry),), "safety": SAFETY,
                    "n_eval": N_FULL, "n_min": N_HALF, "diagnose": (), "eval_parts": (),
                    "note": L["route"]}
    return key


def summarize_llm(llm: str, src: Path, n_boot: int = MM.N_BOOT, seed: int = MM.SEED) -> dict[str, Any]:
    key = _exp_key(llm, src)
    recs = src / "records" / llm
    res = E7.summarize(recs, n_boot, seed, window=[], exp=key)
    res["meta"]["exp"] = f"e13/{llm}"
    res["meta"]["pilot_chosen"] = None                       # E7's Flash pilot is no reference for another LLM
    fits = src / f"cal_{llm}" / "fits.json"
    res["calibration"] = json.loads(fits.read_text()) if fits.exists() else None
    res["calibration_md"] = ((src / f"cal_{llm}" / "fits.md").read_text()
                             if (src / f"cal_{llm}" / "fits.md").exists() else None)
    st = src / llm / "status.json"
    res["status"] = json.loads(st.read_text()) if st.exists() else None
    res["spend_split"] = spend(llm, RUNS / "__none__", src, _is_dry(llm, src))
    res["source"] = C.rel(src)
    res["dry_run"] = _is_dry(llm, src)
    res["cache"] = cache_share(recs / E7.EXPS[key]["eval"])
    res["diag"] = cache_diag(recs / E7.EXPS[key]["eval"], res, llm)
    sched = recs / E7.EXPS[key]["eval"] / "schedule.json"
    res["order_check"] = order_check(res["status"], json.loads(sched.read_text()) if sched.exists() else None)
    return res


def cache_share(run_dir: Path) -> dict[str, Any]:
    """Descriptive (changes no registered number): per evaluation arm, the share of the agent LLM's input tokens billed
    as cache hits, and the mean steps per episode (the latest record per task)."""
    out = {}
    for arm in EVAL_ARMS:
        p = E7.arm_file(run_dir, arm)
        if not p.exists():
            continue
        ts = [t for t in C.load_latest([p])[0].values() if not t.get("error")]
        calls = [c for t in ts for c in t.get("calls") or [] if c.get("component") == "s2"]
        inp = sum(int(c.get("input_tokens") or 0) for c in calls)
        hit = sum(int((c.get("meta") or {}).get("cached_input_tokens") or 0) for c in calls)
        out[arm] = {"cached_share": hit / inp if inp else NAN, "s2_input_tokens": inp / len(ts) if ts else NAN,
                    "steps": float(np.mean([len(t.get("steps") or []) for t in ts])) if ts else NAN}
    return out


# cache-write surcharge (not in the price rows, so not in the logged costs): writes are billed at this multiple of the
# input price (OpenAI pricing page / OpenRouter listing for gpt-6-luna: $0.125 vs $0.10 per M)
CACHE_WRITE_FACTOR = {"gpt-6-luna": 1.25}


def cache_diag(run_dir: Path, res: Mapping[str, Any], llm: str) -> dict[str, Any]:
    """Descriptive cost diagnosis (E13, 2026-10-07; changes no registered number): the cascade arm's agent-LLM calls by
    kind -- the forced argument call of a delegated step (one tool offered), reflection (no tools), the LLM's own
    tool choice (every tool) -- with calls, input tokens, the share billed as cache hits and the tokens written to
    the cache; the cost gap to LLM-only split into Jev's own calls and the agent LLM; the cascade's cost had its LLM
    input been billed at LLM-only's effective price per input token (a perfect prefix-cache fix); and the cache-write
    surcharge the logged costs leave out."""
    groups: dict[str, dict[str, dict[str, float]]] = {}
    for arm in EVAL_ARMS:
        p = E7.arm_file(run_dir, arm)
        if not p.exists():
            continue
        ts = [t for t in C.load_latest([p])[0].values() if not t.get("error")]
        g: dict[str, dict[str, float]] = {}
        for t in ts:
            for c in t.get("calls") or []:
                if c.get("component") != "s2":
                    continue
                m = c.get("meta") or {}
                nt = m.get("n_tools")
                kind = "onetool" if nt == 1 else "notools" if not nt else "alltools"
                row = g.setdefault(kind, {"calls": 0, "input": 0, "cached": 0, "hit_calls": 0, "written": 0})
                row["calls"] += 1
                row["input"] += int(c.get("input_tokens") or 0)
                row["cached"] += int(m.get("cached_input_tokens") or 0)
                row["hit_calls"] += int(bool(m.get("cached_input_tokens")))
                row["written"] += int(m.get("cache_write_tokens") or 0)
        tot_calls = sum(r["calls"] for r in g.values())
        tot_in = sum(r["input"] for r in g.values())
        for r in g.values():
            r["cached_share"] = r["cached"] / r["input"] if r["input"] else NAN
            r["call_share"] = r["calls"] / tot_calls if tot_calls else NAN
            r["token_share"] = r["input"] / tot_in if tot_in else NAN
        groups[arm] = {"groups": g, "n": len(ts), "written_per_episode": sum(r["written"] for r in g.values()) / len(ts)
                       if ts else NAN}
    out: dict[str, Any] = {"arms": groups}
    a = (res.get("arms") or {}).get(ARM) or {}
    if "cmp" in a and "paired_per_episode" in a:
        k, pe, pr = a["cmp"], a["paired_per_episode"], a["ref_paired_per_episode"]
        gap = k["cost"] - k["cost_ref"]
        rate = pr["s2_cost"] / (res.get("cache") or {}).get(REF, {}).get("s2_input_tokens", NAN)
        s2_fix = (res.get("cache") or {}).get(ARM, {}).get("s2_input_tokens", NAN) * rate
        out.update({"ratio": k["cost"] / k["cost_ref"] if k["cost_ref"] else NAN, "gap": gap,
                    "jev_cost": pe["s1_cost"], "llm_cost": pe["s2_cost"], "llm_cost_ref": pr["s2_cost"],
                    "jev_share_of_gap": pe["s1_cost"] / gap if gap > 0 else NAN,
                    "perfect_fix_llm_cost": s2_fix, "perfect_fix_cost": s2_fix + pe["s1_cost"],
                    "perfect_fix_ratio": (s2_fix + pe["s1_cost"]) / k["cost_ref"] if k["cost_ref"] else NAN})
    model = LLMS[llm]["llm"].get("model")
    f = CACHE_WRITE_FACTOR.get(str(model))
    if f:
        from jevharness.accounting import default_prices

        price = default_prices().get(str(model))
        out["write_factor"] = f
        out["write_surcharge"] = {arm: g["written_per_episode"] * (f - 1.0) * price.input_per_m / 1e6
                                  for arm, g in groups.items()} if price else {}
    return out


def order_check(status: Mapping[str, Any] | None, schedule: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The registered order: the fit is written before the first evaluation episode of the LLM. The evaluation's
    first invocation is read from the run's own schedule record (the runner's, kept with the records), else from the
    status record (a driver stopped from outside may not have recorded its invocation)."""
    st = status or {}
    fit_at = (st.get("fit") or {}).get("at")
    starts = [str(i["started"]) for i in (schedule or {}).get("invocations") or [] if i.get("started")]
    inv = (st.get("eval") or {}).get("invocations") or []
    starts += [str(i["start"]) for i in inv if i.get("start")]
    ev = min(starts) if starts else None
    return {"fit_at": fit_at, "eval_start": ev, "ok": bool(fit_at and ev and fit_at <= ev)}


def reference_row() -> dict[str, Any] | None:
    """E7e's Flash row (its own seeds 310500-310599, E7c's Flash-fitted thresholds): context, not part of E13."""
    if not E7E_SUMMARY.exists():
        return None
    s = json.loads(E7E_SUMMARY.read_text())
    a, r = (s.get("arms") or {}).get("jev_refit") or {}, (s.get("arms") or {}).get(REF) or {}
    if a.get("state") != "run":
        return None
    return {"title": "Gemini 3.5 Flash (E7e, reference)", "source": C.rel(E7E_SUMMARY), "cmp": a.get("cmp"),
            "points": a.get("points"), "ppe": a.get("paired_per_episode"), "ref_ppe": a.get("ref_paired_per_episode"),
            "ref_success": r.get("success_all"), "seeds": "310500-310599", "taus": (0.91, 0.87)}


def _pt(x: Any, sign: bool = False) -> str:
    return E7._pt(x, sign)


def _taus_text(cal: Mapping[str, Any] | None) -> str:
    f = ((cal or {}).get("fits") or {}).get(CAL_ARM) or {}
    if not f:
        return "--"
    out = []
    for dp in E7.TOOL_DPS:
        t = _tau_of(f.get(dp) or {})
        out.append("always defer" if t == math.inf else "--" if t is None else f"{t:.4g}")
    return " / ".join(out)


def merged_markdown(res: Mapping[str, Any]) -> str:
    m = res["meta"]
    out = [f"# E13: other agent LLMs on DecisionGym-Hard ({LABEL}, descriptive)", "",
           f"- generated {m['generated']} at git `{(m['git'] or '?')[:12]}` by `scripts/explore_e13_llms.py`; paired task "
           f"bootstrap B = {m['n_boot']}, seed {m['seed']}",
           "- setting (E7e's, unchanged): multi tasks, 12 customers, `tier` rule, itemized final answer, 60 near "
           "distractors, 20 noise lines, strict, max steps 45; arms `llm_only` and `jev_refit` (Jev's tool-choice and "
           "argument cascades at thresholds refitted per LLM on its own calibration arm, plus the fixed reflection "
           "trigger tau 0.5); task-major (seed 732)",
           f"- registration: `docs/PREREG_A7_STAGE2.md`, Deviations \"{ENTRY}\". No test; outside every Holm table. "
           "Dollars are each route's list prices (the saving is paired within an LLM's row).", "",
           "| agent LLM | runner | seeds | n | LLM-only % | Jev refit % | Δ pt [95 % CI] | wins / losses | $/1k ep. "
           "(LLM-only) | saving % [95 % CI] | wall mean s (LLM-only) | delegated % | acc. accepted % | τ̂ tool / args | "
           "spend $ |", "|" + "---|" * 15]
    for llm, L in LLMS.items():
        r = res["llms"].get(llm) or {}
        if r.get("state") != "run":
            out.append(f"| {L['title']} | {L['runner']} | {L['seeds']['eval']}-{L['seeds']['eval'] + N_FULL - 1} | -- | "
                       f"| | | | | | | | | | {r.get('state', 'not run')} |")
            continue
        s = r["summary"]
        a = (s["arms"].get(ARM) or {})
        if "cmp" not in a or "points" not in a:
            out.append(f"| {L['title']} | {L['runner']} | -- | -- | | | | | | | | | | | incomplete |")
            continue
        k, p, pe, pr = a["cmp"], a["points"], a["paired_per_episode"], a["ref_paired_per_episode"]
        ci, sci = k.get("delta_ci") or [NAN, NAN], k.get("saving_ci") or [NAN, NAN]
        sd = s["meta"]["suite"]
        sp = s["spend_split"]
        out.append(f"| {L['title']} | {L['runner']} | {sd['seed0']}-{int(sd['seed0']) + int(s['meta']['n_run']) - 1} | "
                   f"{k['n']} | {_pt(k['success_ref'])} | {_pt(k['success'])} | {_pt(k['delta'], True)} "
                   f"[{_pt(ci[0], True)}, {_pt(ci[1], True)}] | {k['wins']} / {k['losses']} | {1000 * k['cost']:.1f} "
                   f"({1000 * k['cost_ref']:.1f}) | {_pt(k['saving'])} [{_pt(sci[0])}, {_pt(sci[1])}] | "
                   f"{pe['wall_s_mean']:.1f} ({pr['wall_s_mean']:.1f}) | {_pt(p['tool']['coverage'])} | "
                   f"{_pt(p['tool']['acc_accepted'])} | {_taus_text(s.get('calibration'))} | "
                   f"{sp['llm']:.2f} + Jev {sp['jev']:.2f} |")
    ref = res.get("reference")
    if ref and ref.get("cmp"):
        k, p, pe, pr = ref["cmp"], ref["points"], ref["ppe"], ref["ref_ppe"]
        ci, sci = k.get("delta_ci") or [NAN, NAN], k.get("saving_ci") or [NAN, NAN]
        out.append(f"| {ref['title']} | -- | {ref['seeds']} | {k['n']} | {_pt(k['success_ref'])} | {_pt(k['success'])} | "
                   f"{_pt(k['delta'], True)} [{_pt(ci[0], True)}, {_pt(ci[1], True)}] | {k['wins']} / {k['losses']} | "
                   f"{1000 * k['cost']:.1f} ({1000 * k['cost_ref']:.1f}) | {_pt(k['saving'])} [{_pt(sci[0])}, "
                   f"{_pt(sci[1])}] | {pe['wall_s_mean']:.1f} ({pr['wall_s_mean']:.1f}) | {_pt(p['tool']['coverage'])} | "
                   f"{_pt(p['tool']['acc_accepted'])} | 0.91 / 0.87 (E7c) | (E7e) |")
    out += ["", "Delegated: share of the tool decisions (tool choice and arguments) executed from Jev's answer. Acc. "
            "accepted: Jev's accuracy against the gold label on the decisions it took. The Flash row is E7e (other "
            "seeds, thresholds fitted for Flash in E7c), shown for context; it is not part of E13.", ""]
    for llm, L in LLMS.items():
        r = res["llms"].get(llm) or {}
        out += [f"## {L['title']} ({llm})", "", f"- route: {L['route']}", f"- runner: {L['runner']}"
                + ("; run by a collaborator on their own keys (date in the execution record)"
                   if L["runner"] == "collaborator" else ""),
                f"- caps: LLM ${L['caps']['llm']:.0f}, Jev ${L['caps']['jev']:.0f}"]
        if r.get("state") != "run":
            out += [f"- state: {r.get('state', 'not run')}" + (f" ({r['why']})" if r.get("why") else ""), ""]
            continue
        s = r["summary"]
        stt = s.get("status") or {}
        if s.get("dry_run"):
            out.append("- **DRY RUN: simulated LLM and System-1 on development seeds ($0); a pipeline test, never a "
                       "result**")
        out += [f"- source: `{s['source']}`; started {(stt.get('started') or {}).get('at')} (host tag "
                f"{(stt.get('started') or {}).get('host')}, git {str((stt.get('started') or {}).get('git'))[:12]}); "
                f"finished {stt.get('finished')}",
                f"- gate: {(stt.get('gate') or {}).get('decision')}; before the evaluation: "
                f"{(stt.get('gate_eval') or {}).get('decision')}",
                f"- spend: LLM ${s['spend_split']['llm']:.2f}, Jev ${s['spend_split']['jev']:.3f} "
                f"(smoke, calibration and evaluation)", ""]
        ca = s.get("cache") or {}
        if ca:
            out.append("- agent-LLM input billed as cache hits (descriptive): " + "; ".join(
                f"`{a}` {_pt(c['cached_share'])} % of {c['s2_input_tokens']:,.0f} input tokens per episode, "
                f"{c['steps']:.1f} steps" for a, c in ca.items()))
        dg = s.get("diag") or {}
        if dg.get("arms", {}).get(ARM):
            g = dg["arms"][ARM]["groups"]
            gr = (dg["arms"].get(REF) or {}).get("groups", {})
            names = {"onetool": "forced argument calls of delegated steps (one tool offered)",
                     "notools": "reflection calls (no tools)", "alltools": "the LLM's own tool-choice calls (every tool)"}
            out += ["", "### Cost diagnosis (descriptive; changes no registered number)", "",
                    f"Agent-LLM calls of `{ARM}` by kind ({dg['arms'][ARM]['n']} episodes) against `{REF}`:", "",
                    "| arm | call kind | calls | share of calls % | share of input tokens % | input billed as cache hits % "
                    "| calls with a hit | tokens written to the cache |", "|" + "---|" * 8]
            for arm, gg in ((ARM, g), (REF, gr)):
                for kind, r in gg.items():
                    out.append(f"| `{arm}` | {names.get(kind, kind)} | {r['calls']:,} | {_pt(r['call_share'])} | "
                               f"{_pt(r['token_share'])} | {_pt(r['cached_share'])} | {r['hit_calls']:,} | "
                               f"{r['written']:,} |")
            if MM._finite(dg.get("ratio")):
                out += ["", f"- The forced argument call of a delegated step offers ONE tool (`agent._delegated_call`), "
                        "so its prompt prefix (the tool list comes first) differs from every earlier call and the "
                        "provider's prefix cache cannot hit; LLM-only's calls all offer the same list. The history is "
                        "appended the same way in both arms (the reflection note, appended to the last tool message "
                        "when the trigger fires, is the only other prefix change).",
                        f"- Cost gap: ${1000 * dg['gap']:.1f} per 1k episodes (the cascade costs "
                        f"{dg['ratio']:.1f} times LLM-only); Jev's own calls are ${1000 * dg['jev_cost']:.1f} of it "
                        f"({_pt(dg['jev_share_of_gap'])} %), the agent LLM ${1000 * dg['llm_cost']:.1f} against "
                        f"${1000 * dg['llm_cost_ref']:.1f} for LLM-only.",
                        f"- Had the cascade's LLM input been billed at LLM-only's effective price per input token (a "
                        f"perfect prefix-cache fix), its LLM part would be about ${1000 * dg['perfect_fix_llm_cost']:.1f} "
                        f"and the arm about ${1000 * dg['perfect_fix_cost']:.1f} per 1k episodes: still about "
                        f"{dg['perfect_fix_ratio']:.1f} times LLM-only. With this LLM the cascade's gain is success, "
                        "not cost. (`tool_params.extra.full_tools`, off by default, now offers the full list in the "
                        "forced call; it has not been run.)"]
            ws = dg.get("write_surcharge") or {}
            if ws:
                out.append("- Not in the logged costs: cache writes are billed at "
                           f"{dg['write_factor']:.2f} times the input price; the surcharge is about "
                           + ", ".join(f"${1000 * v:.1f} per 1k episodes for `{a}`" for a, v in ws.items()) + ".")
        oc = s.get("order_check") or {}
        out += [f"- order: fit written {oc.get('fit_at')}, first evaluation invocation {oc.get('eval_start')}: "
                + ("fit first, as registered" if oc.get("ok") else "**not in the registered order**"), ""]
        if s.get("calibration_md"):
            out += ["### Calibration fit", ""] + s["calibration_md"].split("\n")[2:]
        body = (E7.to_markdown(s).replace(f"({E7.LABEL}, descriptive)", f"({LABEL}, descriptive)")
                .replace("spend: evaluation + smoke", "spend: calibration + smoke + evaluation")
                .replace("; the pilot counts under E7", "").replace("- GPU tariff: no window record\n", "")
                .split("\n"))
        out += ["### Evaluation (E7's readout)", ""] + [ln.replace("# E13:", "#### E13:", 1) if ln.startswith("# ") else
                                                        ln.replace("## ", "#### ", 1) if ln.startswith("## ") else
                                                        ln.replace("### ", "##### ", 1) if ln.startswith("### ") else ln
                                                        for ln in body] + [""]
    return "\n".join(out) + "\n"


KEEP_LAT = re.compile(r"/lat/[a-z]+/(wall|s1|s2|s2tail|extra|extratail|tailshareofextra)$")


def tex_keys(res: Mapping[str, Any]) -> dict[str, str]:
    """``e13/<llm>/...``: E7's keys for each LLM that ran (prefix e13/<llm>; the latency groups left out, the per-arm
    latency totals kept), the fitted thresholds, spend and state; ``e13/<llm>/state`` = not run otherwise."""
    v: dict[str, str] = {}
    for llm, L in LLMS.items():
        r = res["llms"].get(llm) or {}
        P = f"e13/{llm}"
        v[f"{P}/name"] = L["title"]
        v[f"{P}/seeds"] = f"{L['seeds']['eval']}--{L['seeds']['eval'] + N_FULL - 1}"
        v[f"{P}/runner"] = "collaborator" if L["runner"] == "collaborator" else "local"
        if r.get("state") != "run":
            v[f"{P}/state"] = r.get("state", "not run")
            continue
        s = r["summary"]
        for k, x in E7.to_tex_keys(s).items():
            if "/lat/" in k and not KEEP_LAT.search(k):
                continue
            if k.startswith(f"{P}/cal/"):
                continue
            v[k] = x
        v[f"{P}/state"] = "run"
        f = ((s.get("calibration") or {}).get("fits") or {}).get(CAL_ARM) or {}
        for dp in E7.TOOL_DPS:
            row = f.get(dp) or {}
            t = _tau_of(row)
            k = f"{P}/cal/{dp.replace('_', '')}"
            v[f"{k}/tau"] = "always defer" if t == math.inf else "--" if t is None else f"{t:.4g}"
            at = row.get("at_tau") or {}
            for name, x in (("coverage", "coverage"), ("s1acctaken", "s1_acc_taken"), ("s1accall", "s1_acc_all"),
                            ("llmaccall", "llm_acc_all")):
                if MM._finite(at.get(x)):
                    v[f"{k}/{name}"] = f"{100 * at[x]:.1f}"
            if at.get("n") is not None:
                v[f"{k}/n"] = str(at["n"])
            op = row.get("op_at_tau") or {}
            if dp == "tool_select" and op:
                v[f"{P}/cal/op/n"] = str(op.get("n"))
                for name, x in (("coverage", "coverage"), ("s1accall", "s1_acc_all"), ("llmaccall", "llm_acc_all")):
                    if MM._finite(op.get(x)):
                        v[f"{P}/cal/op/{name}"] = f"{100 * op[x]:.1f}"
        for arm, c in (s.get("cache") or {}).items():
            if MM._finite(c.get("cached_share")):
                v[f"{P}/{arm}/cached_share"] = f"{100 * c['cached_share']:.1f}"
            if MM._finite(c.get("steps")):
                v[f"{P}/{arm}/steps"] = f"{c['steps']:.1f}"
        dg = s.get("diag") or {}
        g = ((dg.get("arms") or {}).get(ARM) or {}).get("groups") or {}
        for kind, r in g.items():
            for name, x, scale in (("calls", "calls", None), ("callshare", "call_share", 100),
                                   ("tokenshare", "token_share", 100), ("cachedshare", "cached_share", 100),
                                   ("hitcalls", "hit_calls", None)):
                val = r.get(x)
                if MM._finite(val):
                    v[f"{P}/diag/{kind}/{name}"] = f"{scale * val:.1f}" if scale else f"{int(val)}"
        for name, x, fmt in (("ratio", "ratio", "{:.1f}"), ("gap_k", "gap", "k"), ("jev_k", "jev_cost", "k"),
                             ("llm_k", "llm_cost", "k"), ("llm_ref_k", "llm_cost_ref", "k"),
                             ("jevshareofgap", "jev_share_of_gap", "pct"), ("perfectfix_llm_k", "perfect_fix_llm_cost", "k"),
                             ("perfectfix_k", "perfect_fix_cost", "k"), ("perfectfix_ratio", "perfect_fix_ratio", "{:.1f}")):
            val = dg.get(x)
            if MM._finite(val):
                v[f"{P}/diag/{name}"] = (f"{1000 * val:.1f}" if fmt == "k" else f"{100 * val:.0f}" if fmt == "pct"
                                         else fmt.format(val))
        for arm, val in (dg.get("write_surcharge") or {}).items():
            if MM._finite(val):
                v[f"{P}/diag/writesurcharge_k/{arm.replace('_', '')}"] = f"{1000 * val:.1f}"
        sp = s["spend_split"]
        v[f"{P}/spend"] = f"{sp['total']:.2f}"
        v[f"{P}/spend_llm"] = f"{sp['llm']:.2f}"
        v[f"{P}/spend_jev"] = f"{sp['jev']:.2f}"
        v[f"{P}/cap_llm"] = f"{L['caps']['llm']:.0f}"
        v[f"{P}/cap_jev"] = f"{L['caps']['jev']:.0f}"
        n_cal = ((s.get("status") or {}).get("gate") or {}).get("n_cal")
        if n_cal:
            v[f"{P}/cal/ntasks"] = str(n_cal)
        if (s.get("status") or {}).get("finished"):
            v[f"{P}/date"] = str(s["status"]["finished"])[:10]
    v["e13/n_llms_run"] = str(sum((res["llms"].get(llm) or {}).get("state") == "run" for llm in LLMS))
    return v


def write_tex(vals: Mapping[str, str], path: Path = TEX) -> Path:
    lines = ["% AUTO-GENERATED by scripts/explore_e13_llms.py -- do not edit by hand.",
             f"% E13: other agent LLMs on DecisionGym-Hard ({LABEL}; docs/PREREG_A7_STAGE2.md Deviations 2026-10-07), "
             "outside both stages.",
             f"% last written {utc()} at git {(C.git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x.replace('_', chr(92) + '_')}}}" for k, x in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


def summarize(out: Path = OUT, handoff: Path | None = HANDOFF_OUT, llms: Sequence[str] | None = None,
              n_boot: int = MM.N_BOOT, seed: int = MM.SEED) -> dict[str, Any]:
    src = _sources(out, handoff)
    res: dict[str, Any] = {"meta": {"generated": utc(), "git": C.git_head(), "label": LABEL, "entry": ENTRY,
                                    "n_boot": n_boot, "seed": seed, "setting": SETTING, "max_steps": MAX_STEPS},
                           "llms": {}, "reference": reference_row()}
    for llm in (llms or LLMS):
        if llm in src:
            res["llms"][llm] = {"state": "run", "summary": summarize_llm(llm, src[llm], n_boot, seed)}
            continue
        stt = None
        for d in (out, handoff):
            if d and (d / llm / "status.json").exists():
                stt = json.loads((d / llm / "status.json").read_text())
                break
        why = (stt or {}).get("stopped", {}).get("why") if stt else None
        res["llms"][llm] = {"state": "stopped" if why else ("in progress" if stt else "not run"), "why": why}
    return res


def cmd_summarize(a: argparse.Namespace) -> int:
    out = Path(a.out)
    llms = [x for x in a.llms.split(",") if x] if a.llms else None
    handoff = None if a.no_handoff or out.resolve() == HANDOFF_OUT.resolve() else HANDOFF_OUT
    res = summarize(out, handoff, llms, a.n_boot, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(MM._clean(res), indent=1, default=MM._json_default) + "\n")
    (out / "summary.md").write_text(merged_markdown(res))
    keys = tex_keys(res)
    tex = write_tex(keys, out / "e13_llms.tex" if a.no_paper else (Path(a.tex) if a.tex else TEX))
    scrub_paths([out / "summary.json", out / "summary.md", tex], Path(a.runs), out)
    print(merged_markdown(res).split("\n## ")[0])
    return 0


# ============================================================================== commands
def cmd_config(a: argparse.Namespace) -> int:
    fits = None
    if a.stage == "eval":
        p = Path(a.out) / f"cal_{a.llm}" / "fits.json"
        fits = json.loads(p.read_text())["fits"] if p.exists() else None
        if fits is None:
            print("# no fit yet: the thresholds below are E7c's Flash values, a placeholder (never run)")
    cfg = stage_config(a.llm, a.stage, dry=a.dry_run, fits=fits,
                       taus=None if fits else (DEV_TAU if a.stage == "eval" else None))
    print(yaml.safe_dump(cfg, sort_keys=False))
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    out = Path(a.out)
    runs = Path(a.runs)
    if a.dry_run:
        if out.resolve() in (OUT.resolve(), HANDOFF_OUT.resolve()) and not a.force_out:
            out = Path(tempfile.mkdtemp(prefix="e13_dry_out_"))
        if runs.resolve() == RUNS.resolve():
            runs = Path(tempfile.mkdtemp(prefix="e13_dry_runs_"))
        print(f"[e13:{a.llm}] DRY RUN (simulated LLM and System-1, development seeds, $0): runs {runs}, out {out}",
              flush=True)
    d = run(a.llm, runs, out, a.until, a.dry_run, a.passes, a.every, a.delete_raw, a.budget)
    brief = {"gate": (d.get("gate") or {}).get("decision"), "fit": (d.get("fit") or {}).get("taus"),
             "gate_eval": (d.get("gate_eval") or {}).get("decision"), "stopped": d.get("stopped"),
             "spend": {k: (d.get("spend") or {}).get(k) for k in ("llm", "jev")}, "finished": d.get("finished")}
    print(json.dumps(brief, indent=1, default=str))
    return 10 if d.get("stopped") else 0      # 10: stopped by a registered rule (gate, cap), as the handoff reads it


def cmd_status(a: argparse.Namespace) -> int:
    p = Path(a.out) / a.llm / "status.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d["spend_now"] = spend(a.llm, Path(a.runs), Path(a.out))
    print(json.dumps(d, indent=1, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llm", choices=sorted(LLMS), default="luna")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(OUT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("--stage", choices=("smoke", "cal", "eval"), required=True)
    c.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--until", choices=STAGES, default="slim")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--force-out", action="store_true", help="dry run: write into --out even if it is a real output dir")
    r.add_argument("--passes", type=int, default=PASSES)
    r.add_argument("--every", type=float, default=20.0, help="watchdog poll interval, seconds")
    r.add_argument("--delete-raw", action="store_true", help="delete the raw runs once their slim copies exist")
    r.add_argument("--budget", type=float, default=None, help="$ the key may still spend (handoff key-usage cap)")
    sub.add_parser("status")
    s = sub.add_parser("summarize")
    s.add_argument("--llms", default=None, help="comma-separated (default: all)")
    s.add_argument("--no-handoff", action="store_true", help="do not merge results/handoff/e13")
    s.add_argument("--no-paper", action="store_true", help="write <out>/e13_llms.tex instead of paper/generated/")
    s.add_argument("--tex", default=None)
    s.add_argument("--n-boot", type=int, default=MM.N_BOOT)
    s.add_argument("--seed", type=int, default=MM.SEED)
    a = ap.parse_args(argv)
    return {"config": cmd_config, "run": cmd_run, "status": cmd_status, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
