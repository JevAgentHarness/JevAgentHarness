#!/usr/bin/env python3
r"""E7d (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06 "E7d: re-run the recipe on the new
rule"): AgentDecision-4B (the frozen JevAgentRL-4B checkpoint of the records) is continued on JevGym-Hard tier-rule
states harvested at $0 (the simulated oracle executor of the harness; labels are the JevGym oracle gold) plus a replay
of its original SFT mix, giving a NEW checkpoint, "AgentDecision-4B (tier)". Scored offline only. No test; outside
every Holm table; the frozen checkpoint is unchanged and stays the paper's model.

    python scripts/explore_e7d_tier.py harvest --kind train   --out runs/e7d_harvest      # box CPU, $0
    python scripts/explore_e7d_tier.py harvest --kind heldout --out runs/e7d_harvest      # box CPU, $0
    python scripts/explore_e7d_tier.py build --harvest runs/e7d_harvest --d100 A/data/D100.jsonl --out A/data
    python scripts/explore_e7d_tier.py score --run A      # -> results/explore/e7d_tier/summary.{md,json}, e7d_tier.tex

Harvest: the reference loop on the E7b / E7c setting (multi, 12 customers, tier rule, 60 near distractors, 20 noise
lines, strict, max_steps 45). The executor is ``llm/simulated.SimulatedLLM`` (follows the oracle; error rate
``ERROR_RATE``) and System-1 is ``data_v1.NearMissBackend`` (the oracle with near-miss errors) -- simulated components
that only choose which states are visited; every label is the oracle's gold. Training seeds 320000-320599 under
data_v1's behaviour policies ``lite_shadow`` and ``nearmiss_replace``; held-out seeds 321000-321199 under
``lite_shadow`` only (a range the training guards exclude)."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

OUT = ROOT / "results" / "explore" / "e7d_tier"
TEX = ROOT / "paper" / "generated" / "e7d_tier.tex"
LABEL = "exploratory (E7d), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-06 'E7d'; outside every Holm table"
SETTING = {"name": "jevgym", "families": "multi", "multi_customers": 12, "multi_rule": "tier", "distractor_style": "near",
           "n_distractors": 60, "noise_lines": 20, "strict": True}
MAX_STEPS = 45
TRAIN_SEEDS = (320000, 600)          # seed0, n
HELDOUT_SEEDS = (321000, 200)
ERROR_RATE = 0.05                     # the simulated executor's error rate (wrong tool or a corrupted argument)
EPS = {"eps_tool": 0.1, "eps_param": 0.1, "eps_gate": 0.1}    # NearMissBackend in nearmiss_replace
DPS = ("tool_select", "tool_params")
OP_TOOLS = ("cancel_order", "ship_order")
# rows (fixed before any training step): tier training rows by stratum, the replay of the original mix, the dev split
N_TIER = {"op": 900, "ts_other": 450, "tp": 450}
N_REPLAY = 1200
DEV_TASK_FRAC = 0.10                 # tasks of the training harvest held back as the tier dev split (temperatures)
N_HELDOUT = {"op": 600, "ts_other": 450, "tp": 450}
# scoring within the 4 GPU-hour cap (Deviations 2026-10-06 "E7d: scoring within the GPU cap"; fixed before scoring)
TRIM = {"cal_v1": 1000, "cal_tier": 1000, "heldout_ts_other": 225, "heldout_tp": 225}
CHECK_TASKS = 5                       # the serving-path check: the registered checkpoint on the first 5 tasks' questions
SEED = 0
NEW_NAME = "AgentDecision-4B (tier)"
OLD_NAME = "AgentDecision-4B"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def _write(rows: Iterable[Mapping[str, Any]], p: Path) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
            n += 1
    return n


# ============================================================================== harvest ($0)
def harvest(kind: str, out: Path, n: int | None = None, workers: int = 16) -> Path:
    from jevharness.bench.runner import run_experiment
    from jevharness.training import data_v1 as D

    llm = {"provider": "simulated", "model": "oracle-sim", "error_rate": ERROR_RATE, "simulated_latency_ms": 0}
    if kind == "train":
        seed0, n0 = TRAIN_SEEDS
        arms = [a for a in D.behaviour_arms() if a["name"] in ("lite_shadow", "nearmiss_replace")]
        exp = D.build_experiment(name="e7d_train", seed0=seed0, n=n or n0, llm=llm, workers=workers,
                                 suite={k: v for k, v in SETTING.items() if k != "name"}, arms=arms)
        exp["harness_defaults"] = {"max_steps": MAX_STEPS}
        D.collect(exp, out, s1=D.NearMissBackend(**EPS), progress=False)       # check_seed_plan: training seeds
        return out / "e7d_train"
    if kind == "heldout":
        seed0, n0 = HELDOUT_SEEDS
        arms = [a for a in D.behaviour_arms() if a["name"] == "lite_shadow"]
        exp = {"name": "e7d_heldout", "suite": {**SETTING, "n": int(n or n0), "seed0": seed0}, "llm": llm,
               "s1": {"type": "nearmiss (in-process)"}, "s1_backends": {}, "workers": workers, "arms": arms,
               "harness_defaults": {"max_steps": MAX_STEPS}}
        # the held-out range is excluded from training (bench/registry.training_exclusions), so data_v1.collect would
        # refuse it: the runner is called directly; nothing from this run ever enters a training file
        run_experiment(exp, out, s1=D.NearMissBackend(**EPS), progress=False)
        return out / "e7d_heldout"
    raise SystemExit(f"unknown kind {kind!r}")


# ============================================================================== rows
def stratum(r: Mapping[str, Any]) -> str | None:
    dp = r.get("dp")
    if dp == "tool_params":
        return "tp"
    if dp != "tool_select":
        return None
    t = r.get("target") or {}
    gold = max(t, key=lambda o: t[o]) if t else None
    name = str(gold).split(" -- ", 1)[0].strip() if gold else None     # options read "<tool> -- <description>"
    return "op" if name in OP_TOOLS else "ts_other"


def tier_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gold rows (``data_v1.harvest_runs``; one-hot oracle targets) at the two tool decision points, deduplicated."""
    from jevharness.training import data_v1 as D

    rows, summary = D.harvest_runs(run_dir, syn_guard=False)
    rows = [r for r in rows if r.get("dp") in DPS and r.get("target")]
    rows, n_dup = D.dedupe(rows)
    return rows, {**summary, "duplicates_dropped": n_dup}


def _task(r: Mapping[str, Any]) -> str:
    return str(r["id"]).split("/", 1)[0]


def _sample(rows: Sequence[dict[str, Any]], quota: Mapping[str, int], rng: random.Random) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        s = stratum(r)
        if s:
            by[s].append(r)
    out = []
    for s, k in quota.items():
        pool = sorted(by.get(s, []), key=lambda r: str(r["id"]))
        out += rng.sample(pool, min(k, len(pool)))
    return out


def build(harvest_dir: Path, d100: Path, v1_dev: Path, out: Path, seed: int = SEED) -> dict[str, Any]:
    """train_mix.jsonl (tier rows + the replay sample of D100), cal_dev.jsonl (v1 dev + the tier dev split: the
    temperatures), heldout_{op,ts,tp}.jsonl, build_info.json. Nothing from the held-out run enters a training file;
    the training rows' JevGym seeds are checked against the training guards."""
    from jevharness.bench import registry as R

    rng = random.Random(seed)
    train_all, s_train = tier_rows(harvest_dir / "e7d_train")
    held_all, s_held = tier_rows(harvest_dir / "e7d_heldout")
    seeds_train = {int(_task(r).split("-")[2]) for r in train_all}
    bad = [s for s in seeds_train if R.in_ranges(s, R.training_exclusions() + (R.CALIBRATION_RANGE,))
           or s < R.TRAIN_MIN_SEED]
    if bad:
        raise SystemExit(f"training rows on guarded seeds: {sorted(bad)[:5]}")
    seeds_held = {int(_task(r).split("-")[2]) for r in held_all}
    if seeds_held & seeds_train:
        raise SystemExit("held-out and training seeds overlap")
    tasks = sorted({_task(r) for r in train_all})
    dev_tasks = set(rng.sample(tasks, max(1, round(DEV_TASK_FRAC * len(tasks)))))
    tr = [r for r in train_all if _task(r) not in dev_tasks]
    dv = [r for r in train_all if _task(r) in dev_tasks]
    tier = _sample(tr, N_TIER, rng)
    d100_rows = _read(d100)
    replay = rng.sample(d100_rows, min(N_REPLAY, len(d100_rows)))
    for r in tier:
        r.setdefault("meta", {})["e7d"] = "tier"
    for r in replay:
        r.setdefault("meta", {})["e7d"] = "replay_d100"
    mix = tier + replay
    rng.shuffle(mix)
    held = _sample(held_all, N_HELDOUT, rng)
    out.mkdir(parents=True, exist_ok=True)
    n = {"train_mix": _write(mix, out / "train_mix.jsonl"),
         "cal_dev": _write(_read(v1_dev) + dv, out / "cal_dev.jsonl"),
         "tier_dev": _write(dv, out / "tier_dev.jsonl")}
    for s in ("op", "ts_other", "tp"):
        n[f"heldout_{s}"] = _write([r for r in held if stratum(r) == s], out / f"heldout_{s}.jsonl")
    info = {"label": LABEL, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "setting": SETTING, "max_steps": MAX_STEPS, "train_seeds": TRAIN_SEEDS, "heldout_seeds": HELDOUT_SEEDS,
            "executor_error_rate": ERROR_RATE, "nearmiss_eps": EPS, "quota_tier": N_TIER, "replay": N_REPLAY,
            "quota_heldout": N_HELDOUT, "dev_task_frac": DEV_TASK_FRAC, "seed": seed, "rows": n,
            "tier_strata": dict(Counter(stratum(r) for r in tier)),
            "available": {"train": dict(Counter(stratum(r) for r in tr)), "dev": len(dv),
                          "heldout": dict(Counter(stratum(r) for r in held_all))},
            "tasks": {"train": len(tasks) - len(dev_tasks), "dev": len(dev_tasks), "heldout": len(seeds_held)},
            "harvest": {"train": {k: v for k, v in s_train.items() if k != "run"},
                        "heldout": {k: v for k, v in s_held.items() if k != "run"}},
            "d100": {"path": str(d100), "sha256": _sha(d100), "rows": len(d100_rows)},
            "sha256": {f.name: _sha(f) for f in sorted(out.glob("*.jsonl"))}}
    (out / "build_info.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    return info


# ============================================================================== calibration-state replay fit
def trim(src: Path, n: int, out: Path, seed: int = SEED) -> dict[str, Any]:
    """A seeded sample of ``n`` rows of ``src`` in file order (all rows when it has at most ``n``)."""
    rows = _read(src)
    if len(rows) > n:
        keep = sorted(random.Random(seed).sample(range(len(rows)), n))
        rows = [rows[i] for i in keep]
    return {"src": src.name, "src_sha256": _sha(src), "rows": _write(rows, out), "out": out.name}


def trim_all(data: Path, v1_dev: Path) -> dict[str, Any]:
    """Deviations 2026-10-06 "E7d: scoring within the GPU cap": the temperatures are refitted on TRIM["cal_v1"] v1 dev
    plus TRIM["cal_tier"] tier dev rows (cal_dev_small.jsonl); the held-out other tool-choice and argument strata are
    sampled to TRIM rows (the write-state stratum stays whole). Seed 0; sizes fixed before any scoring step."""
    info = {k: trim(src, TRIM[k], data / out) for k, src, out in (
        ("cal_v1", v1_dev, "cal_v1_small.jsonl"), ("cal_tier", data / "tier_dev.jsonl", "cal_tier_small.jsonl"),
        ("heldout_ts_other", data / "heldout_ts_other.jsonl", "heldout_ts_other_s.jsonl"),
        ("heldout_tp", data / "heldout_tp.jsonl", "heldout_tp_s.jsonl"))}
    rows = _read(data / "cal_v1_small.jsonl") + _read(data / "cal_tier_small.jsonl")
    info["cal_dev_small"] = {"rows": _write(rows, data / "cal_dev_small.jsonl")}
    (data / "trim_info.json").write_text(json.dumps({"label": LABEL, "trim": TRIM, **info}, indent=1) + "\n")
    return info


def head_tasks(requests: Path, k: int, out: Path) -> dict[str, Any]:
    """The questions of the first ``k`` tasks (file order = sorted task ids) of a replay request file."""
    import e5_cal_replay as RP

    qs = RP._jsonl(requests)
    tids = list(dict.fromkeys(q["task_id"] for q in qs))[:k]
    keep = set(tids)
    return {"tasks": tids, "questions": _write([q for q in qs if q["task_id"] in keep], out)}


def complete(requests: Path, answers: Path, out_dir: Path, min_tasks: int = 10) -> dict[str, Any]:
    """After the time-bounded replay: the tasks whose every question has an error-free answer, the new checkpoint's
    answers on them (``answers_tier_cal.jsonl``) and the registered checkpoint's LOGGED answers on the same tasks
    (``answers_frozen_cal.jsonl``: its E7c shadow answers, the answers its fit used), so both fits read the same
    states."""
    import e5_cal_replay as RP

    qs = RP._jsonl(requests)
    ans = {r["key"]: r for r in RP._jsonl(answers) if r.get("error") is None and r.get("value") is not None}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in qs:
        by_task[q["task_id"]].append(q)
    tasks = [t for t, v in by_task.items() if all(q["key"] in ans for q in v)]
    keep = set(tasks)
    n_t = _write([ans[q["key"]] for q in qs if q["task_id"] in keep], out_dir / "answers_tier_cal.jsonl")
    n_f = _write([{"key": q["key"], "dp": q["dp"], "value": q["logged"]["value"], "conf": q["logged"]["conf"],
                   "dist": q["logged"]["dist"], "error": None, "source": "logged (E7c rl_shadow_cal)"}
                  for q in qs if q["task_id"] in keep], out_dir / "answers_frozen_cal.jsonl")
    res = {"label": LABEL, "tasks_total": len(by_task), "tasks_complete": len(tasks), "tasks": tasks,
           "questions_total": len(qs), "answered": len(ans), "rows_tier": n_t, "rows_frozen": n_f,
           "enough": len(tasks) >= min_tasks}
    (out_dir / "cal_tasks.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def cal_fit(src_run: Path, arm: str, answers: Path, name: str, out_dir: Path, work: Path,
            tasks: Sequence[str] | None = None) -> dict[str, Any]:
    """The E7c calibration states re-answered by a served model (``scripts/e5_cal_replay.py answer``): a copy of the
    shadow arm with those answers (``derive``), then the registered estimator per decision point (``threshold_theory
    --calibration-arm``, W = $1, lambda = 0, V fallback W) and the coverage / accuracies at the fitted tau-hat on the
    same states, for all tool choices and the cancel-vs-ship subset."""
    import subprocess

    import e5_cal_replay as RP
    import explore_e7_hard as E7
    from jevharness.bench import threshold_theory as T

    n = RP.derive(src_run, arm, answers, work, name)
    if tasks is not None:  # the time-bounded replay: only the tasks answered in full (complete())
        keep, f = set(tasks), work / f"{name}.jsonl"
        lines = [x for x in f.read_text().splitlines() if x.strip() and json.loads(x).get("task_id") in keep]
        f.write_text("\n".join(lines) + "\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs = T.load_trajectories(work / f"{name}.jsonl")
    res: dict[str, Any] = {"label": LABEL, "source": f"{src_run.name}/{arm}", "answers": answers.name,
                           "answers_sha256": _sha(answers), "derive": n, "n_tasks": len(trajs),
                           "tasks_filter": len(tasks) if tasks is not None else None}
    for dp in DPS:
        f = out_dir / f"tauhat_{name}_{dp}.json"
        subprocess.run([sys.executable, "-m", "jevharness.bench.threshold_theory", str(work), "--arm", name, "--dp", dp,
                        "--calibration-arm", "--task-value", "1.0", "--lam-per-hour", "0", "--v-fallback", "W",
                        "--format", "json", "--out", str(f)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        d = json.loads(f.read_text())
        tau = float(d["tau_hat"]) if d.get("tau_hat") is not None else float("nan")
        ps = T.paired(T.decision_pairs(trajs, dp, arm=name, gold="acceptable"))
        row = {"tau_hat": tau, "always_defer": bool(tau == float("inf")), "n_paired": d.get("n_paired"),
               "tau_rule": (d.get("tau_rule") or {}).get("rule"), "at_tau": E7._at_tau(ps, tau)}
        if dp == "tool_select":
            row["op_at_tau"] = E7._at_tau([r for r in ps if r.get("gold") in OP_TOOLS], tau)
        res[dp] = row
    (out_dir / f"fits_{name}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


# ============================================================================== score (local, CPU)
def _v1(p: Path) -> dict[str, Any] | None:
    import ablation_score as AS

    return AS.v1_metrics(p) if p.exists() else None


def _file_acc(ev: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    for k, v in (ev.get("files") or ev.get("by_file") or {}).items():
        if Path(k).name.startswith(name):
            o = v.get("overall") or v
            return {"n": o.get("n"), "acc": o.get("acc"), "ece": o.get("ece"), "mean_conf": o.get("mean_conf")}
    return None


def score(run: Path, corpus: Path = ROOT / "results" / "stage2" / "s23") -> dict[str, Any]:
    """Held-out tier states (eval_decisions files per stratum), the E7c calibration-state replay (the registered fit on
    each model's answers, scripts/e5_cal_replay.py derive + fit), corpus C (accuracy, the paired task bootstrap of the
    new checkpoint minus the frozen one), v1 dev."""
    import ablation_score as AS
    import e5_analysis as EA

    out: dict[str, Any] = {"label": LABEL, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "run": str(run), "models": {}}
    for m in ("frozen", "tier"):
        row: dict[str, Any] = {"name": OLD_NAME if m == "frozen" else NEW_NAME}
        p = run / "eval" / f"{m}_heldout.json"
        if p.exists():
            ev = json.loads(p.read_text())
            row["heldout"] = {s: _file_acc(ev, f"heldout_{s}") for s in ("op", "ts_other", "tp")}
        p = run / "eval" / f"{m}_v1.json"
        if p.exists():
            row["v1"] = _v1(p)
        p = run / "cal" / f"fits_{m}.json"
        if p.exists():
            row["cal"] = json.loads(p.read_text())
        out["models"][m] = row
    if (ROOT / "results/gpu_round5/eval/eval_grpo_i15_v1.json").exists() and not out["models"]["frozen"].get("v1"):
        out["models"]["frozen"]["v1"] = _v1(ROOT / "results/gpu_round5/eval/eval_grpo_i15_v1.json")
    occ, eligible = AS.load_corpus(corpus)
    per: dict[str, Any] = {}
    for m, p in (("frozen", corpus / "answers_rl.jsonl"), ("tier", run / "s23" / "answers_e7d_tier.jsonl")):
        if p.exists():
            cm, pt = EA.c_rows(occ, eligible, AS.load_answers(p))
            out["models"][m]["c"] = {k: cm.get(k) for k in ("c_acc", "c_acc_offplan", "c_acc_ask", "c_tasks")}
            per[m] = pt
    if "frozen" in per and "tier" in per:
        out["c_delta"] = {s: AS.paired_boot(per["tier"][s], per["frozen"][s], EA.N_BOOT, 0) for s in ("all", "offplan")}
    for k, f in (("cal_tasks", run / "cal" / "cal_tasks.json"), ("check_frozen", run / "cal" / "check_frozen.json"),
                 ("trim", run / "data" / "trim_info.json")):
        if f.exists():
            d = json.loads(f.read_text())
            out[k] = {kk: v for kk, v in d.items() if kk != "tasks"} if k == "cal_tasks" else d
    jev = ROOT / "results/explore/e7_hard/cal_e7c/fits.json"
    if jev.exists():
        fits = json.loads(jev.read_text())["fits"]
        out["jev_cal"] = fits.get("jev_shadow_cal")
        out["frozen_cal_e7c_100"] = fits.get("rl_shadow_cal")   # the registered checkpoint's E7c fit, all 100 tasks
    return out


def _p(x: Any) -> str:
    return f"{100 * x:.1f}" if isinstance(x, (int, float)) and x == x else "--"


def tex_keys(d: Mapping[str, Any]) -> dict[str, str]:
    v: dict[str, str] = {}
    for m, row in d["models"].items():
        k = f"e7d/{m}"
        for s, a in (row.get("heldout") or {}).items():
            if a:
                v[f"{k}/heldout/{s.replace('_', '')}/acc"] = _p(a.get("acc"))
                v[f"{k}/heldout/{s.replace('_', '')}/n"] = str(a.get("n"))
        cal = row.get("cal") or {}
        for dp, key in (("tool_select", "toolselect"), ("tool_params", "toolparams")):
            r = cal.get(dp)
            if r:
                v[f"{k}/cal/{key}/tau"] = "always defer" if r.get("always_defer") else f"{r['tau_hat']:.4g}"
                a = r.get("at_tau") or {}
                v[f"{k}/cal/{key}/coverage"] = _p(a.get("coverage"))
                v[f"{k}/cal/{key}/s1accall"] = _p(a.get("s1_acc_all"))
                v[f"{k}/cal/{key}/s1acctaken"] = _p(a.get("s1_acc_taken"))
        op = (cal.get("tool_select") or {}).get("op_at_tau")
        if op:
            v[f"{k}/cal/op/s1accall"] = _p(op.get("s1_acc_all"))
            v[f"{k}/cal/op/coverage"] = _p(op.get("coverage"))
        c = row.get("c") or {}
        if c.get("c_acc") is not None:
            v[f"{k}/c/acc"] = _p(c["c_acc"])
        if (row.get("v1") or {}).get("v1_acc") is not None:
            v[f"{k}/v1/acc"] = _p(row["v1"]["v1_acc"])
    ct = d.get("cal_tasks") or {}
    if ct.get("tasks_complete") is not None:
        v["e7d/cal/ntasks"] = str(ct["tasks_complete"])
    for dp, key in (("tool_select", "toolselect"), ("tool_params", "toolparams")):
        r = ((d.get("check_frozen") or {}).get("by_dp") or {}).get(dp)
        if r:
            v[f"e7d/check/{key}/agree"] = _p(r.get("agreement"))
            v[f"e7d/check/{key}/n"] = str(r.get("questions"))
    cd = (d.get("c_delta") or {}).get("all")
    if cd:
        v["e7d/c/delta"] = f"{100 * cd['delta']:+.1f}"
        v["e7d/c/delta_ci/lo"] = f"{100 * cd['lo']:+.1f}"
        v["e7d/c/delta_ci/hi"] = f"{100 * cd['hi']:+.1f}"
    return v


def markdown(d: Mapping[str, Any]) -> str:
    v = tex_keys(d)
    g = lambda k: v.get(k, "--")  # noqa: E731
    ms = [m for m in ("frozen", "tier") if m in d["models"]]
    head = "| | " + " | ".join(d["models"][m]["name"] for m in ms) + " |\n|---|" + "---|" * len(ms)
    rows = [("held-out tier states, cancel/ship tool choice (%)", "heldout/op/acc"),
            ("held-out tier states, other tool choices (%)", "heldout/tsother/acc"),
            ("held-out tier states, arguments (%)", "heldout/tp/acc"),
            ("E7c calibration states: tool-choice tau-hat", "cal/toolselect/tau"),
            ("  coverage at tau-hat (%)", "cal/toolselect/coverage"),
            ("  System-1 accuracy, all tool choices (%)", "cal/toolselect/s1accall"),
            ("  System-1 accuracy, cancel/ship (%)", "cal/op/s1accall"),
            ("E7c calibration states: argument tau-hat", "cal/toolparams/tau"),
            ("  coverage at tau-hat (%)", "cal/toolparams/coverage"),
            ("  System-1 accuracy, all arguments (%)", "cal/toolparams/s1accall"),
            ("corpus C accuracy (%)", "c/acc"), ("v1 dev accuracy (%)", "v1/acc")]
    lines = [f"# E7d: {NEW_NAME}", "", f"_{d['label']}_", "", head]
    lines += [f"| {lab} | " + " | ".join(g(f"e7d/{m}/{k}") for m in ms) + " |" for lab, k in rows]
    if "e7d/c/delta" in v:
        lines += ["", f"Corpus C, new minus registered checkpoint: {v['e7d/c/delta']} points "
                      f"(paired task bootstrap 95 % [{v['e7d/c/delta_ci/lo']}, {v['e7d/c/delta_ci/hi']}])."]
    if "e7d/cal/ntasks" in v:
        lines += ["", f"Calibration states: the first {v['e7d/cal/ntasks']} of the 100 E7c calibration tasks (sorted task ids) "
                      "were re-answered in full within the GPU cap; both checkpoints are fitted on those tasks (the "
                      "registered checkpoint from its logged E7c answers)."]
    if "e7d/check/toolselect/agree" in v:
        lines += [f"Serving-path check (registered checkpoint re-asked on the first {CHECK_TASKS} tasks): agreement "
                  f"{v['e7d/check/toolselect/agree']} % (tool choice, n {v['e7d/check/toolselect/n']}), "
                  f"{v.get('e7d/check/toolparams/agree', '--')} % (arguments, n {v.get('e7d/check/toolparams/n', '--')})."]
    f100 = d.get("frozen_cal_e7c_100") or {}
    if f100:
        lines += ["", "Reference (E7c fit of the registered checkpoint, all 100 tasks): tool-choice tau-hat "
                      f"{f100.get('tool_select', {}).get('tau_hat', '--')}, argument tau-hat "
                      f"{f100.get('tool_params', {}).get('tau_hat', '--')}."]
    j = d.get("jev_cal") or {}
    if j:
        lines += ["", "Reference (E7c fit, Jev on the same calibration states): tool-choice tau-hat "
                      f"{j.get('tool_select', {}).get('tau_hat', '--')}, argument tau-hat "
                      f"{j.get('tool_params', {}).get('tau_hat', '--')}."]
    lines += ["", f"Generated {d.get('generated', '')} from `{d.get('run', '')}`."]
    return "\n".join(lines) + "\n"


def write(d: Mapping[str, Any]) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import explore_e7_hard as E7

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(d, indent=1, default=str) + "\n")
    (OUT / "summary.md").write_text(markdown(d))
    E7.write_tex(tex_keys(d), TEX, "Exploratory row E7d, AgentDecision-4B (tier)")


def cmd(a: argparse.Namespace) -> int:
    if a.cmd == "harvest":
        print(harvest(a.kind, Path(a.out), a.n, a.workers))
    elif a.cmd == "build":
        print(json.dumps(build(Path(a.harvest), Path(a.d100), Path(a.v1_dev), Path(a.out))["rows"], indent=1))
    elif a.cmd == "cal":
        tasks = json.loads(Path(a.tasks).read_text())["tasks"] if a.tasks else None
        r = cal_fit(Path(a.src), a.arm, Path(a.answers), a.name, Path(a.out), Path(a.work), tasks)
        print(json.dumps({dp: {k: r[dp][k] for k in ("tau_hat", "n_paired")} for dp in DPS}, default=str))
    elif a.cmd == "trim":
        print(json.dumps(trim_all(Path(a.data), Path(a.v1_dev)), indent=1))
    elif a.cmd == "head-tasks":
        r = head_tasks(Path(a.requests), a.tasks, Path(a.out))
        print(json.dumps({"tasks": len(r["tasks"]), "questions": r["questions"]}))
    elif a.cmd == "complete":
        r = complete(Path(a.requests), Path(a.answers), Path(a.out))
        print(json.dumps({k: v for k, v in r.items() if k != "tasks"}))
        return 0 if r["enough"] else 1
    elif a.cmd == "score":
        d = score(Path(a.run))
        write(d)
        print(json.dumps(d, indent=1, default=str)[:4000])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--kind", choices=["train", "heldout"], required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--n", type=int, default=None)
    h.add_argument("--workers", type=int, default=16)
    b = sub.add_parser("build")
    b.add_argument("--harvest", required=True)
    b.add_argument("--d100", required=True)
    b.add_argument("--v1-dev", default=str(ROOT / "data/train_v1/dev.jsonl.gz"))
    b.add_argument("--out", required=True)
    c = sub.add_parser("cal")
    c.add_argument("--src", required=True, help="the E7c calibration run (runs/e7c_cal)")
    c.add_argument("--arm", default="rl_shadow_cal")
    c.add_argument("--answers", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--work", required=True)
    c.add_argument("--tasks", default=None, help="cal_tasks.json from `complete` (fit on these tasks only)")
    t = sub.add_parser("trim")
    t.add_argument("--data", required=True)
    t.add_argument("--v1-dev", default=str(ROOT / "data/train_v1/dev.jsonl.gz"))
    hq = sub.add_parser("head-tasks")
    hq.add_argument("--requests", required=True)
    hq.add_argument("--tasks", type=int, default=CHECK_TASKS)
    hq.add_argument("--out", required=True)
    cp = sub.add_parser("complete")
    cp.add_argument("--requests", required=True)
    cp.add_argument("--answers", required=True)
    cp.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    return cmd(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
