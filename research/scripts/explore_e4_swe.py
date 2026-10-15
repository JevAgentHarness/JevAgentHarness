#!/usr/bin/env python3
"""Exploratory row E4 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-05): JevAgentHarness on mini-SWE-agent.

Per-step System-1 / System-2 routing on SWE-bench Verified with the existing routing hook of
``jevharness.adapters.mini_swe`` (the H5 ``route`` arm machinery; mini-SWE-agent 2.4.6, unmodified):
deep path (System 2) = ``deepseek/deepseek-v4-pro`` with reasoning on, fast path = ``deepseek/deepseek-v4-flash`` with
reasoning off, both through OpenRouter (mini-SWE-agent's own ``OpenRouterModel``: cost = OpenRouter ``usage.cost``),
pinned to one provider; the decision model is Jev via OpenRouter. Descriptive, outside every Holm table.

    python scripts/explore_e4_swe.py select                       # instance lists (smoke / dev / eval) + hashes
    python scripts/explore_e4_swe.py run --phase dev -o runs/e4_swe/dev -w 25      # on the SWE box (Docker)
    python scripts/explore_e4_swe.py threshold runs/e4_swe/dev    # the registered dev rule -> threshold.json
    python scripts/explore_e4_swe.py summarize runs/e4_swe/eval   # results/e4_swe/* + paper/generated/e4_swe.tex

E4r (Deviations 2026-10-07, "E4r: replicate and extension of E4"): E4's eval arms again at E4's tau, no dev, on E4's
100 eval instances (replicate) and 200 new held-out instances (extension), driver scripts/aws/e4r_swe.sh:

    python scripts/explore_e4_swe.py --exp e4r select              # configs/explore/e4r_swe/ (lists, hashes, tau)
    python scripts/explore_e4_swe.py --exp e4r readout results/e4r_swe/rows.json --spend results/e4r_swe/spend.json

``run`` executes every (instance, arm) job of a phase in ONE shared pool of W workers, task-major: instance by instance
in list order, the arms of one instance in a per-instance seeded order, so the arms run interleaved under the same box
and API conditions. ``--rerun-errored`` re-runs, once, the instances whose record carries an infrastructure error (an
exception escaped the agent); the errored record is kept under ``<arm>/_errored/<iid>.1/``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "adapters" / "mini_swe"))

IDS_DIR = ROOT / "configs" / "explore" / "e4_swe"
RESULTS = ROOT / "results" / "e4_swe"
TEX = ROOT / "paper" / "generated" / "e4_swe.tex"
H5_EVAL_IDS = ROOT / "results" / "hard" / "swe" / "instance_ids.txt"
H5_SMOKE_IDS = ROOT / "results" / "hard" / "smoke" / "swe" / "instance_ids.txt"
SALT = "jev-explore-E4-swe-2026-10-05"
N_DEV, N_EVAL = 45, 100
DEEP_MODEL = "deepseek/deepseek-v4-pro"
FAST_MODEL = "deepseek/deepseek-v4-flash"
BASE_CONFIG = "adapters/mini_swe/swebench_jev.yaml"
E4_CONFIG = "adapters/mini_swe/e4_deepseek.yaml"
E4_FAST = "adapters/mini_swe/e4_fast.yaml"
S1_CONFIG = "adapters/mini_swe/s1_jev_openrouter.yaml"
REGISTERED_ROUTE_TAU = 0.5          # the H5 `route` arm (swebench_jev.yaml)
DEV_TAUS = (0.5, 0.6, 0.7)          # registered dev candidates (E4 entry)
MARGIN = 2                          # dev resolved count within MARGIN of always_deep
SCHEDULE_SEED = 732
LABEL = "exploratory (E4), descriptive"
REG_NOTE = "Deviations 2026-10-05, Exploratory row E4"
EXP, PREFIX = "e4", "e4"
FAST_OVERLAY_MAIN_IS_FAST = True     # always_fast: every agent call is on the fast path

# E4b (Deviations 2026-10-05, "Exploratory row E4b"): the same harness, arms, rule and eval list with both paths on ONE
# model (Gemini 3.8 Flash, thinking high vs low, direct Gemini API), so a switch keeps the prompt cache.
EXPS: dict[str, dict[str, Any]] = {
    "e4": {"IDS_DIR": IDS_DIR, "RESULTS": RESULTS, "TEX": TEX, "E4_CONFIG": E4_CONFIG, "E4_FAST": E4_FAST,
           "DEEP_MODEL": DEEP_MODEL, "FAST_MODEL": FAST_MODEL, "LABEL": LABEL, "EXP": "e4", "PREFIX": "e4",
           "N_DEV": N_DEV, "REG_NOTE": REG_NOTE},
    "e4b": {"IDS_DIR": ROOT / "configs" / "explore" / "e4b_swe", "RESULTS": ROOT / "results" / "e4b_swe",
            "TEX": ROOT / "paper" / "generated" / "e4b_swe.tex", "E4_CONFIG": "adapters/mini_swe/e4b_gemini.yaml",
            "E4_FAST": "adapters/mini_swe/e4b_fast.yaml", "DEEP_MODEL": "gemini/gemini-3.8-flash",
            "FAST_MODEL": "gemini/gemini-3.8-flash", "LABEL": "exploratory (E4b), descriptive", "EXP": "e4b",
            "PREFIX": "e4b", "N_DEV": 30, "REG_NOTE": "Deviations 2026-10-05, Exploratory row E4B"},
}


# E4r (Deviations 2026-10-07, "E4r: replicate and extension of E4"): E4's eval arms again, with no dev and E4's
# registered tau, on E4's 100 eval instances (an independent replicate) and on 200 new held-out instances.
E4R_SALT = "jev-explore-E4r-swe-2026-10-07"
N_EXT = 200
E4_IDS_DIR = IDS_DIR
# development SWE-bench Verified instances outside every list: the L3 smoke pair (adapters/mini_swe/README.md, 2026-10-01)
# and the README's debug example (it may have been run while the adapter was developed)
DEV_SWE_IDS = ("django__django-11099", "pytest-dev__pytest-7432", "sympy__sympy-20590")
EXPS["e4r"] = {**EXPS["e4"], "IDS_DIR": ROOT / "configs" / "explore" / "e4r_swe", "RESULTS": ROOT / "results" / "e4r_swe",
               "TEX": ROOT / "paper" / "generated" / "e4r_swe.tex", "LABEL": "exploratory (E4r), descriptive",
               "EXP": "e4r", "PREFIX": "e4r", "N_DEV": 0,
               "REG_NOTE": "Deviations 2026-10-07, E4r: replicate and extension of E4"}


def set_exp(name: str) -> None:
    """Rebind the module settings to one exploratory row (default E4)."""
    globals().update(EXPS[name])


# ----------------------------------------------------------------------------------------------- selection
def read_ids(path: Path) -> list[str]:
    return [ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines() if ln.split("#")[0].strip()]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def order_key(iid: str) -> str:
    return sha256_text(f"{SALT}\n{iid}")


def select(pool: list[str], h5_eval: list[str], h5_smoke: list[str], n_dev: int = N_DEV,
           n_eval: int = N_EVAL) -> dict[str, list[str]]:
    """E4's instance rule: the H5 smoke pair is E4's smoke; the pool minus H5's 100 evaluation ids and the smoke pair is
    ordered by sha256(SALT + newline + id); dev = the first n_dev, eval = the next n_eval (disjoint by construction)."""
    excluded = set(h5_eval) | set(h5_smoke)
    rest = sorted((i for i in set(pool) if i not in excluded), key=order_key)
    if len(rest) < n_dev + n_eval:
        raise ValueError(f"pool too small: {len(rest)} < {n_dev + n_eval}")
    return {"smoke": list(h5_smoke), "dev": rest[:n_dev], "eval": rest[n_dev:n_dev + n_eval]}


def _ids_text(ids: list[str]) -> str:
    return "\n".join(ids) + "\n"


def cmd_select(a: argparse.Namespace) -> None:
    if a.pool_file:
        pool = read_ids(Path(a.pool_file))
    else:
        from datasets import load_dataset
        pool = list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test")["instance_id"])
    if len(set(pool)) != 500:
        raise SystemExit(f"expected the 500 SWE-bench Verified test ids, got {len(set(pool))}")
    sel = select(pool, read_ids(H5_EVAL_IDS), read_ids(H5_SMOKE_IDS))
    IDS_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"rule": f"pool = SWE-bench Verified test (500) minus H5's 100 evaluation ids "
                                    f"({H5_EVAL_IDS.relative_to(ROOT)}) and the H5 smoke pair "
                                    f"({H5_SMOKE_IDS.relative_to(ROOT)}, = E4's smoke); ordered by "
                                    f"sha256('{SALT}' + newline + id); dev = first {N_DEV}, eval = next {N_EVAL}",
                            "salt": SALT, "pool_sha256": sha256_text(_ids_text(sorted(set(pool)))),
                            "n_pool_after_exclusion": len(set(pool)) - 102}
    for k, ids in sel.items():
        p = IDS_DIR / f"{k}_ids.txt"
        p.write_text(_ids_text(ids))
        meta[k] = {"file": str(p.relative_to(ROOT)), "n": len(ids), "sha256": sha256_text(_ids_text(ids))}
    (IDS_DIR / "selection.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


def e4r_exclusions() -> dict[str, list[str]]:
    """Every SWE-bench Verified instance any earlier run has touched (E4b's lists are subsets of E4's: its smoke is the
    H5 smoke pair, its dev the first 30 of E4's dev list, its eval E4's eval list)."""
    return {"h5_eval": read_ids(H5_EVAL_IDS), "h5_smoke": read_ids(H5_SMOKE_IDS),
            "e4_dev": read_ids(E4_IDS_DIR / "dev_ids.txt"), "e4_eval": read_ids(E4_IDS_DIR / "eval_ids.txt"),
            "dev_adapter": list(DEV_SWE_IDS)}


def e4r_order_key(iid: str) -> str:
    return sha256_text(f"{E4R_SALT}\n{iid}")


def interleave(replicate: list[str], extension: list[str]) -> list[str]:
    """E4r's run order: R E E R E E ... (one replicate instance, then two extension instances), so both subsets run
    under the same box and API conditions over the whole run."""
    k = len(extension) // max(1, len(replicate))
    out: list[str] = []
    for j, r in enumerate(replicate):
        out += [r] + extension[k * j:k * (j + 1)]
    return out + extension[k * len(replicate):]


# Minimum-spend amendment (Deviations 2026-10-07, "E4r: minimum spend", before any E4r episode): the replicate is not
# run again (E4's own run on it is used), the run list is the extension alone, and the smoke is one instance of the H5
# smoke pair (the cheaper one in E4's smoke: $0.022 for its three arms against $0.073), which the gate needs per arm.
E4R_SMOKE_ID = "scikit-learn__scikit-learn-13496"


def select_e4r(pool: list[str], replicate: list[str], excluded: dict[str, list[str]], smoke: list[str],
               n_ext: int = N_EXT) -> dict[str, list[str]]:
    """E4r's instance rule: replicate = E4's eval list as registered (unchanged, in its order; not run again: E4's run
    on it enters the pooled readout); extension = the pool minus every excluded instance, ordered by
    sha256(E4R_SALT + newline + id), the first n_ext; reserve = the rest (never run). eval (the run list) = the
    extension in that order; smoke = E4R_SMOKE_ID (one of the given smoke pair)."""
    ex = set().union(*map(set, excluded.values()))
    if not set(replicate) <= ex:
        raise ValueError("the replicate list must be among the excluded (already used) instances")
    rest = sorted((i for i in set(pool) if i not in ex), key=e4r_order_key)
    if len(rest) < n_ext:
        raise ValueError(f"pool too small: {len(rest)} < {n_ext}")
    ext = rest[:n_ext]
    sm = [i for i in smoke if i == E4R_SMOKE_ID] or list(smoke)[:1]
    return {"smoke": sm, "replicate": list(replicate), "extension": ext, "eval": list(ext), "reserve": rest[n_ext:]}


def cmd_select_e4r(a: argparse.Namespace) -> None:
    if a.pool_file:
        pool = read_ids(Path(a.pool_file))
    else:
        from datasets import load_dataset
        pool = list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test")["instance_id"])
    if len(set(pool)) != 500:
        raise SystemExit(f"expected the 500 SWE-bench Verified test ids, got {len(set(pool))}")
    excluded = e4r_exclusions()
    e4sel = json.loads((E4_IDS_DIR / "selection.json").read_text())
    replicate = read_ids(E4_IDS_DIR / "eval_ids.txt")
    if sha256_text(_ids_text(replicate)) != e4sel["eval"]["sha256"]:
        raise SystemExit("E4's eval list does not match its registered hash")
    sel = select_e4r(pool, replicate, excluded, read_ids(H5_SMOKE_IDS))
    n_ex = len(set().union(*map(set, excluded.values())))
    IDS_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "rule": (f"replicate = E4's eval list (configs/explore/e4_swe/eval_ids.txt, unchanged); extension = SWE-bench "
                 f"Verified test (500) minus every instance used before ({', '.join(excluded)}; {n_ex} distinct), "
                 f"ordered by sha256('{E4R_SALT}' + newline + id), the first {N_EXT}; reserve = the rest (never run); "
                 "eval (the run list) = the extension in that order (minimum-spend amendment: the replicate is not "
                 f"run again; E4's own run on it enters the pooled readout); smoke = {E4R_SMOKE_ID} (one instance of "
                 "the H5 smoke pair = E4's smoke; development only)"),
        "superseded": {"eval": {"n": 300, "sha256": "99831a6c3dc0babb9288369a26478af4aa5392432894f7f4d2d8406b71cf652f",
                                "what": "replicate and extension interleaved (e4448cc); never run"},
                       "smoke": {"n": 2, "sha256": "7b474a955d1beddc839dd909e9f4753286a3eb279bce6f4fbeecfe2f461e1614",
                                 "what": "the H5 smoke pair (e4448cc); never run in E4r"}},
        "salt": E4R_SALT, "pool_sha256": sha256_text(_ids_text(sorted(set(pool)))),
        "n_pool_after_exclusion": len(set(pool) - set().union(*map(set, excluded.values()))),
        "excluded": {k: {"n": len(v), "sha256": sha256_text(_ids_text(v))} for k, v in excluded.items()},
        "excluded_distinct": n_ex}
    for k, ids in sel.items():
        p = IDS_DIR / f"{k}_ids.txt"
        p.write_text(_ids_text(ids))
        meta[k] = {"file": str(p.relative_to(ROOT)), "n": len(ids), "sha256": sha256_text(_ids_text(ids))}
    (IDS_DIR / "selection.json").write_text(json.dumps(meta, indent=1) + "\n")
    th = json.loads((E4_IDS_DIR / "threshold.json").read_text())
    (IDS_DIR / "threshold.json").write_text(json.dumps({
        "tau": float(th["tau"]), "case": "e4_registered",
        "reason": f"E4's eval tau ({th['tau']}, fixed on E4's dev by E4's registered rule, case {th['case']}); E4r has "
                  "no dev and changes no setting",
        "source": "configs/explore/e4_swe/threshold.json",
        "source_sha256": hashlib.sha256((E4_IDS_DIR / "threshold.json").read_bytes()).hexdigest(),
        "eval_ids_sha256": meta["eval"]["sha256"]}, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------------------------- arms
def arm_specs(arm: str) -> tuple[list[str], str, dict[str, Any]]:
    """E4 arm name -> (mini-swe config specs, jev arm in the YAML, extra info). The arms differ only as listed."""
    base = [BASE_CONFIG, E4_CONFIG, S1_CONFIG]
    if arm == "always_deep":
        return base, "llm_only", {"main": DEEP_MODEL}
    if arm == "always_fast":
        return base + [E4_FAST], "llm_only", {"main": FAST_MODEL}
    if arm.startswith("jevagent"):
        tau = tau_of(arm)
        return base + [f"jev.arms.jevagent.routing.tau={tau}"], "jevagent", {"main": DEEP_MODEL, "tau": tau}
    raise KeyError(arm)


def tau_of(arm: str, threshold_file: Path | None = None) -> float:
    """jevagent_t060 -> 0.6; plain `jevagent` (the eval arm) -> the tau of the committed threshold.json."""
    if arm.startswith("jevagent_t"):
        return int(arm.split("_t", 1)[1]) / 100
    if arm == "jevagent":
        f = threshold_file or IDS_DIR / "threshold.json"
        if not f.exists():
            raise SystemExit(f"{f} missing: the eval arm's tau is fixed on dev and committed before any eval run")
        return float(json.loads(f.read_text())["tau"])
    raise KeyError(arm)


PHASES: dict[str, dict[str, Any]] = {
    "smoke": {"ids": "smoke_ids.txt", "arms": ["always_deep", "always_fast", "jevagent_t050"]},
    "dev": {"ids": "dev_ids.txt", "arms": ["always_deep", "always_fast"] + [f"jevagent_t{round(t * 100):03d}"
                                                                          for t in DEV_TAUS]},
    "eval": {"ids": "eval_ids.txt", "arms": ["always_deep", "always_fast", "jevagent"]},
}


EXPS["e4"]["PHASES"] = PHASES
# E4b amendment (Deviations 2026-10-06, before any E4b dev episode): dev candidates tau 0.5 and 0.6 only (budget)
EXPS["e4b"]["PHASES"] = {**PHASES, "dev": {"ids": "dev_ids.txt",
                                           "arms": ["always_deep", "always_fast", "jevagent_t050", "jevagent_t060"]}}
# E4r: no dev; the smoke (development only) and the eval run E4's three eval arms, `jevagent` at E4's registered tau
EXPS["e4r"]["PHASES"] = {"smoke": {"ids": "smoke_ids.txt", "arms": ["always_deep", "always_fast", "jevagent"]},
                         "eval": {"ids": "eval_ids.txt", "arms": ["always_deep", "always_fast", "jevagent"]}}


def schedule(ids: list[str], arms: list[str], seed: int = SCHEDULE_SEED) -> list[tuple[str, str]]:
    """Task-major job order: instance by instance (list order), the arms of each instance in a seeded order."""
    jobs = []
    for i, iid in enumerate(ids):
        order = list(arms)
        random.Random(f"{seed}:{i}:{iid}").shuffle(order)
        jobs += [(iid, arm) for arm in order]
    return jobs


def check_eval_gate(threshold_file: Path, ids_file: Path) -> float:
    """The eval phase runs only on the registered list with the tau fixed on dev (threshold.json, committed first)."""
    if not threshold_file.exists():
        raise SystemExit(f"refusing the eval phase: {threshold_file} missing")
    th = json.loads(threshold_file.read_text())
    sel = json.loads((IDS_DIR / "selection.json").read_text())
    h = sha256_text(_ids_text(read_ids(ids_file)))
    if h != sel["eval"]["sha256"] or th.get("eval_ids_sha256") != h:
        raise SystemExit(f"refusing the eval phase: eval list hash {h[:12]} does not match the registered one")
    return float(th["tau"])


def _record_error(rec_path: Path) -> bool:
    try:
        return bool(json.loads(rec_path.read_text()).get("error"))
    except (OSError, ValueError):
        return True


def cmd_run(a: argparse.Namespace) -> None:  # pragma: no cover - needs Docker and the APIs (box only)
    import concurrent.futures
    import copy

    from jevharness.adapters import mini_swe as MS
    from jevharness.decision import make_backend

    ph = PHASES[a.phase]
    ids_file = Path(a.ids) if a.ids else IDS_DIR / ph["ids"]
    arms = a.arms or ph["arms"]
    if a.phase == "eval":
        check_eval_gate(IDS_DIR / "threshold.json", ids_file)
    ids = read_ids(ids_file)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    instances = {i["instance_id"]: i for i in MS._load_instances("verified", "test", instance_ids=str(ids_file))}
    setups: dict[str, dict[str, Any]] = {}
    for arm in arms:
        specs, jarm, info = arm_specs(arm)
        if arm == "jevagent":
            specs = specs[:-1] + [f"jev.arms.jevagent.routing.tau={tau_of(arm)}"]
            info["tau"] = tau_of(arm)
        specs = specs + [f"jev.s1_cache=cache/{EXP}_{a.phase}_{arm}.sqlite"] + list(a.config or [])
        config = MS.load_config([str(ROOT / s) if not ("=" in s) else s for s in specs])
        jcfg = MS.JevMiniSweConfig.for_arm(copy.deepcopy(config.get("jev", {})), jarm)
        jcfg.name = arm
        d = out / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "jev_arm.json").write_text(json.dumps({"arm": arm, "jev_arm": jarm, "config_spec": specs, **info,
                                                    "model": config["model"], "cheap_model": jcfg.cheap_model,
                                                    "jev": jcfg.model_dump(mode="json", exclude={"arms"})},
                                                   indent=1, default=str))
        s1 = make_backend(jcfg.s1, cache=jcfg.s1_cache) if jcfg.uses_s1() else None
        setups[arm] = {"config": config, "jcfg": jcfg, "s1": s1, "dir": d}
    if a.rerun_errored:
        todo = []
        for arm in arms:
            d = setups[arm]["dir"]
            for iid in ids:
                rec = d / iid / f"{iid}.jev.json"
                if rec.exists() and _record_error(rec):
                    n = len(list((d / "_errored").glob(f"{iid}.*"))) if (d / "_errored").exists() else 0
                    if n >= 1:
                        print(f"[e4] {arm}/{iid}: errored again after its one re-run; left as is", flush=True)
                        continue
                    (d / "_errored").mkdir(exist_ok=True)
                    shutil.move(str(d / iid), str(d / "_errored" / f"{iid}.{n + 1}"))
                    from minisweagent.run.benchmarks.swebench import remove_from_preds_file
                    remove_from_preds_file(d / "preds.json", iid)
                    todo.append((iid, arm))
                    print(f"[e4] {arm}/{iid}: errored record moved aside, re-run once", flush=True)
        jobs = [j for j in schedule(ids, arms) if j in set(todo)]
    else:
        jobs = []
        limits = dict((x.split("=")[0], int(x.split("=")[1])) for x in (a.arm_limit or []))
        for iid, arm in schedule(ids, arms):
            if arm in limits and ids.index(iid) >= limits[arm]:
                continue
            preds = setups[arm]["dir"] / "preds.json"
            done = set(json.loads(preds.read_text())) if preds.exists() else set()
            if iid not in done:
                jobs.append((iid, arm))
    print(f"[{EXP}:{a.phase}] {len(jobs)} jobs, {len(arms)} arms, {a.workers} workers -> {out}", flush=True)

    def one(job: tuple[str, str]) -> str:
        iid, arm = job
        s = setups[arm]
        rec = MS.process_instance(instances[iid], s["dir"], s["config"], s["jcfg"], s["s1"])
        return f"{arm}/{iid}: {rec.get('exit_status')}{' ERROR' if rec.get('error') else ''}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for k, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                msg = fut.result()
            except Exception as e:  # process_instance records its own failures; this is a last resort
                msg = f"{futs[fut][1]}/{futs[fut][0]}: UNCAUGHT {type(e).__name__}: {e}"
            print(f"[{EXP}:{a.phase}] {k}/{len(jobs)} {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- readout
def _cost(usage: Mapping[str, Any] | None, comp: str) -> float:
    return float(((usage or {}).get(comp) or {}).get("cost_usd") or 0.0)


def load_arm(path: Path) -> dict[str, dict[str, Any]]:
    """<arm>.jsonl (collected Trajectories, success = resolved) -> task id -> per-instance row (latest record wins).
    Fast path = the routing hook's `cheap` role, or every call of the `always_fast` arm (in E4b both paths are the
    same model, so the path is read from the role, not the model name)."""
    all_fast = Path(path).stem == "always_fast"
    from jevharness.harness.trajectory import load_trajectories
    out: dict[str, dict[str, Any]] = {}
    for t in load_trajectories(str(path)):
        u = t.get("usage") or {}
        res = t.get("result") or {}
        meta = t.get("meta") or {}
        per_model = ((meta.get("model_stats") or {}).get("per_model") or {})
        calls_by_path: Counter[str] = Counter()
        for role, st in per_model.items():
            calls_by_path["fast" if (all_fast or role == "cheap") else "deep"] += int(st.get("calls") or 0)
        n_calls = sum(calls_by_path.values())
        lat: Counter[str] = Counter()
        tok: Counter[str] = Counter()
        for c in t.get("calls") or []:
            if c.get("component") == "s2":
                path = "fast" if (all_fast or (c.get("meta") or {}).get("role") == "cheap") else "deep"
                lat[path] += float(c.get("latency_ms") or 0.0) / 1000
                tok["input"] += int(c.get("input_tokens") or 0)
                tok["cached"] += int((c.get("meta") or {}).get("cached_input_tokens") or 0)
            elif c.get("component") == "s1":
                lat["s1"] += float(c.get("latency_ms") or 0.0) / 1000
        out[t["task_id"]] = {
            "success": float(bool(res.get("success"))), "resolved": res.get("resolved"),
            "time_s": float(t.get("wall_ms") or 0.0) / 1000.0, "error": t.get("error"),
            "exit_status": meta.get("exit_status") or res.get("exit_status"),
            "s1_cost": _cost(u, "s1"), "s2_cost": _cost(u, "s2") or _cost(u, "llm"), "total_cost": _cost(u, "total"),
            "s1_calls": int(((u.get("s1") or {}).get("calls")) or 0),
            "steps": len(t.get("steps") or []), "llm_calls": n_calls,
            "fast_calls": calls_by_path["fast"], "deep_calls": calls_by_path["deep"],
            "deep_call_s": lat["deep"], "fast_call_s": lat["fast"], "s1_call_s": lat["s1"],
            "input_tokens": tok["input"], "cached_tokens": tok["cached"],
        }
    return out


def fast_share(rows: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Pooled share of the arm's agent LLM calls answered by the fast path (Flash, reasoning off)."""
    n = sum(r["llm_calls"] for r in rows.values())
    return sum(r["fast_calls"] for r in rows.values()) / n if n else None


def choose_threshold(resolved: Mapping[str, int], shares: Mapping[str, float | None],
                     margin: int = MARGIN) -> dict[str, Any]:
    """The registered dev rule (E4 entry). resolved: dev resolved counts of always_deep, always_fast and every
    jevagent_tXXX candidate; shares: the candidates' dev fast-shares.
      1. If always_fast is itself within `margin` of always_deep, dev cannot separate the settings -> the registered
         route tau (0.5).
      2. Else the eligible candidates are those with resolved >= always_deep - margin; none -> the registered route
         tau (0.5, the most conservative candidate).
      3. Else the eligible candidate with the largest dev fast-share; ties (equal to 3 decimals) -> the lower tau."""
    deep, fast = int(resolved["always_deep"]), int(resolved["always_fast"])
    cands = sorted((a for a in resolved if a.startswith("jevagent_t")), key=tau_of)
    base = {"always_deep_resolved": deep, "always_fast_resolved": fast, "margin": margin,
            "candidates": {a: {"tau": tau_of(a), "resolved": int(resolved[a]), "fast_share": shares.get(a),
                               "eligible": int(resolved[a]) >= deep - margin} for a in cands}}
    if fast >= deep - margin:
        return {**base, "tau": REGISTERED_ROUTE_TAU, "case": "not_separable",
                "reason": f"always_fast ({fast}) is within {margin} of always_deep ({deep}) on dev: dev cannot "
                          f"separate the settings, so the registered route tau {REGISTERED_ROUTE_TAU} is used"}
    elig = [a for a in cands if base["candidates"][a]["eligible"]]
    if not elig:
        return {**base, "tau": REGISTERED_ROUTE_TAU, "case": "none_eligible",
                "reason": f"no candidate within {margin} of always_deep ({deep}); the registered route tau "
                          f"{REGISTERED_ROUTE_TAU} (the most conservative candidate) is used"}
    best = max(elig, key=lambda a: (round(shares.get(a) or 0.0, 3), -tau_of(a)))
    return {**base, "tau": tau_of(best), "case": "rule", "chosen_arm": best,
            "reason": f"{best}: the largest dev fast-share ({(shares.get(best) or 0):.3f}) among the candidates "
                      f"within {margin} of always_deep ({deep})"}


def cmd_threshold(a: argparse.Namespace) -> None:
    run = Path(a.run_dir)
    arms = PHASES["dev"]["arms"]
    ids = read_ids(IDS_DIR / "dev_ids.txt")
    data = {arm: load_arm(run / f"{arm}.jsonl") for arm in arms}
    for arm, d in data.items():
        missing = set(ids) - set(d)
        if missing:
            raise SystemExit(f"{arm}: {len(missing)} dev instances without a record")
        if any(d[i]["resolved"] is None for i in ids):
            raise SystemExit(f"{arm}: unevaluated dev instances (no SWE-bench report)")
    resolved = {arm: int(sum(data[arm][i]["success"] for i in ids)) for arm in arms}
    shares = {arm: fast_share({i: data[arm][i] for i in ids}) for arm in arms}
    out = choose_threshold(resolved, shares)
    out.update({"phase": "dev", "n_dev": len(ids), "dev_ids_sha256": sha256_text(_ids_text(ids)),
                "eval_ids_sha256": json.loads((IDS_DIR / "selection.json").read_text())["eval"]["sha256"],
                "resolved": resolved, "fast_share": shares,
                "errors": {arm: sum(bool(data[arm][i]["error"]) for i in ids) for arm in arms},
                "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    dst = Path(a.out) if a.out else run / "threshold.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def arm_summary(name: str, rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    v = list(rows.values())
    n = len(v)
    times = [r["time_s"] for r in v]
    return {"arm": name, "n": n, "resolved": int(sum(r["success"] for r in v)),
            "unevaluated": sum(r["resolved"] is None for r in v), "errors": sum(bool(r["error"]) for r in v),
            "cost_mean": sum(r["total_cost"] for r in v) / n if n else None,
            "llm_cost_mean": sum(r["s2_cost"] for r in v) / n if n else None,
            "s1_cost_mean": sum(r["s1_cost"] for r in v) / n if n else None,
            "cost_total": sum(r["total_cost"] for r in v),
            "time_median_s": _q(times, 0.5), "time_mean_s": sum(times) / n if n else None,
            "time_p90_s": _q(times, 0.9), "time_total_h": sum(times) / 3600,
            "steps_mean": sum(r["steps"] for r in v) / n if n else None,
            "llm_calls_mean": sum(r["llm_calls"] for r in v) / n if n else None,
            "s1_calls_mean": sum(r["s1_calls"] for r in v) / n if n else None,
            "fast_share": fast_share(rows),
            "deep_call_mean_s": (sum(r["deep_call_s"] for r in v) / max(1, sum(r["deep_calls"] for r in v))
                                 if any(r["deep_calls"] for r in v) else None),
            "fast_call_mean_s": (sum(r["fast_call_s"] for r in v) / max(1, sum(r["fast_calls"] for r in v))
                                 if any(r["fast_calls"] for r in v) else None),
            "s1_call_mean_s": (sum(r["s1_call_s"] for r in v) / max(1, sum(r["s1_calls"] for r in v))
                               if any(r["s1_calls"] for r in v) else None),
            "llm_time_share": (sum(r["deep_call_s"] + r["fast_call_s"] for r in v) / sum(times)) if sum(times) else None,
            "cached_input_share": (sum(r["cached_tokens"] for r in v) / max(1, sum(r["input_tokens"] for r in v))),
            "exit_statuses": dict(Counter(str(r["exit_status"]) for r in v).most_common())}


def paired_vs(base: Mapping[str, Mapping[str, Any]], arm: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from jevharness.bench import metrics as M
    common = sorted(set(base) & set(arm))
    wins = sum(arm[t]["success"] > base[t]["success"] for t in common)
    losses = sum(arm[t]["success"] < base[t]["success"] for t in common)
    both = [t for t in common if arm[t]["success"] and base[t]["success"]]
    dts = [arm[t]["time_s"] - base[t]["time_s"] for t in both]
    t_wins = sum(d < 0 for d in dts)
    t_losses = sum(d > 0 for d in dts)
    cb = sum(base[t]["total_cost"] for t in common) / len(common) if common else None
    ca = sum(arm[t]["total_cost"] for t in common) / len(common) if common else None
    tb = sum(base[t]["time_s"] for t in common) / len(common) if common else None
    ta = sum(arm[t]["time_s"] for t in common) / len(common) if common else None
    sign = M.sign_test_exact(wins, losses) if (wins + losses) else {}
    return {"tasks": len(common),
            "d_resolved": int(sum(arm[t]["success"] for t in common) - sum(base[t]["success"] for t in common)),
            "wins": wins, "losses": losses, "sign_p_two_sided": sign.get("p"),
            "mcnemar_p": M.mcnemar({t: base[t]["success"] for t in common},
                                   {t: arm[t]["success"] for t in common})["p"] if common else None,
            "saving_pct": (100 * (1 - ca / cb)) if cb else None,
            "mean_time_saving_pct": (100 * (1 - ta / tb)) if tb else None,
            "n_cosolved": len(both), "d_time_median_cosolved_s": _q(dts, 0.5),
            "cosolved_faster": t_wins, "cosolved_slower": t_losses,
            "cosolved_time_sign_p_one_sided": (M.sign_test_exact(t_wins, t_losses)["p_wins"]
                                               if (t_wins + t_losses) else None),
            "d_time_median_all_s": _q([arm[t]["time_s"] - base[t]["time_s"] for t in common], 0.5)}


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, dict):
        return ", ".join(f"{k} {v}" for k, v in x.items())
    return str(x)


def _num(x: Any, nd: int = 1, signed: bool = False) -> str:
    if x is None:
        return "--"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return ("$-$" + s[1:]) if s.startswith("-") else ("$+$" + s[1:]) if s.startswith("+") else s


def summarize(run: Path, arms: list[str], baseline: str = "always_deep") -> dict[str, Any]:
    data = {a: load_arm(run / f"{a}.jsonl") for a in arms if (run / f"{a}.jsonl").exists()}
    rows = [arm_summary(a, d) for a, d in data.items()]
    pairs = {a: paired_vs(data[baseline], d) for a, d in data.items() if a != baseline and baseline in data}
    return {"label": LABEL, "run_dir": str(run), "baseline": baseline, "arms": rows, "paired": pairs, "data": data}


def tex_values(s: dict[str, Any], prefix: str | None = None) -> dict[str, str]:
    prefix = prefix or PREFIX
    v: dict[str, str] = {}
    for r in s["arms"]:
        a = r["arm"]
        v[f"{prefix}/{a}/n"] = str(r["n"])
        v[f"{prefix}/{a}/resolved"] = str(r["resolved"])
        v[f"{prefix}/{a}/success"] = _num(100 * r["resolved"] / r["n"] if r["n"] else None)
        v[f"{prefix}/{a}/cost_usd"] = _num(r["cost_mean"], 4)
        v[f"{prefix}/{a}/llm_cost_usd"] = _num(r["llm_cost_mean"], 4)
        v[f"{prefix}/{a}/s1_cost_usd"] = _num(r["s1_cost_mean"], 4)
        v[f"{prefix}/{a}/wall_median_s"] = _num(r["time_median_s"], 0)
        v[f"{prefix}/{a}/wall_mean_s"] = _num(r["time_mean_s"], 0)
        v[f"{prefix}/{a}/steps"] = _num(r["steps_mean"], 1)
        v[f"{prefix}/{a}/fast_share"] = _num(100 * r["fast_share"] if r["fast_share"] is not None else None, 1)
        v[f"{prefix}/{a}/limits_exceeded"] = str(r["exit_statuses"].get("LimitsExceeded", 0))
        v[f"{prefix}/{a}/deep_call_s"] = _num(r["deep_call_mean_s"], 1)
        v[f"{prefix}/{a}/fast_call_s"] = _num(r["fast_call_mean_s"], 1)
        v[f"{prefix}/{a}/s1_call_s"] = _num(r["s1_call_mean_s"], 2)
        v[f"{prefix}/{a}/llm_calls"] = _num(r["llm_calls_mean"], 1)
    for a, p in s["paired"].items():
        v[f"{prefix}/{a}/vs_deep/delta"] = _num(p["d_resolved"], 0, signed=True)
        v[f"{prefix}/{a}/vs_deep/wins"] = str(p["wins"])
        v[f"{prefix}/{a}/vs_deep/losses"] = str(p["losses"])
        v[f"{prefix}/{a}/vs_deep/saving_pct"] = _num(p["saving_pct"], 1)
        v[f"{prefix}/{a}/vs_deep/time_saving_pct"] = _num(p["mean_time_saving_pct"], 1)
        v[f"{prefix}/{a}/vs_deep/time_cosolved/n"] = str(p["n_cosolved"])
        v[f"{prefix}/{a}/vs_deep/time_cosolved/d_median_s"] = _num(p["d_time_median_cosolved_s"], 1, signed=True)
        v[f"{prefix}/{a}/vs_deep/time_all/d_median_s"] = _num(p["d_time_median_all_s"], 1, signed=True)
    return v


def write_tex(values: dict[str, str], path: Path | None = None, note: str = "") -> None:
    path = path or TEX
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    lines = ["% AUTO-GENERATED by scripts/explore_e4_swe.py -- do not edit by hand.",
             f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git {head}; {LABEL} "
             f"(PREREG_A7_STAGE2 {REG_NOTE}); no test, outside every Holm table",
             *([f"% {note}"] if note else []),
             "\\makeatletter",
             "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(values.items())]
    lines.append("\\makeatother")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def markdown(s: dict[str, Any], title: str) -> str:
    cols = ["arm", "n", "resolved", "unevaluated", "errors", "cost_mean", "llm_cost_mean", "s1_cost_mean",
            "time_median_s", "time_mean_s", "time_p90_s", "steps_mean", "llm_calls_mean", "fast_share",
            "exit_statuses"]
    cols2 = ["arm", "deep_call_mean_s", "fast_call_mean_s", "s1_call_mean_s", "s1_calls_mean", "llm_time_share",
             "cached_input_share", "time_total_h", "cost_total"]
    out = [f"# {title}", "", f"**{LABEL}.** Outside every Holm table; revises no registered result.", "",
           "## Arms (k = 1; cost = API $ per instance: agent LLM as logged (E4 DeepSeek: OpenRouter `usage.cost`; E4b Gemini: token usage at the verified list price) + Jev; time = agent-run "
           "wall clock per instance)", "",
           "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r.get(c), 1 if c.startswith("time") or c.endswith("mean") and "cost" not in c
                                   else 4) for c in cols) + " |" for r in s["arms"]]
    out += ["", "## Where the time goes (descriptive): mean latency per call by path, share of agent-run time spent "
            "waiting for the agent LLM, share of input tokens served from the prompt cache", "",
            "| " + " | ".join(cols2) + " |", "|" + "---|" * len(cols2)]
    out += ["| " + " | ".join(_fmt(r.get(c), 3) for c in cols2) + " |" for r in s["arms"]]
    pc = ["arm", "tasks", "d_resolved", "wins", "losses", "sign_p_two_sided", "mcnemar_p", "saving_pct",
          "mean_time_saving_pct", "n_cosolved", "d_time_median_cosolved_s", "cosolved_faster", "cosolved_slower",
          "cosolved_time_sign_p_one_sided", "d_time_median_all_s"]
    out += ["", f"## Paired vs {s['baseline']} (descriptive; p values are descriptive, no test)", "",
            "| " + " | ".join(pc) + " |", "|" + "---|" * len(pc)]
    out += ["| " + " | ".join(_fmt(p.get(c) if c != "arm" else a, 3) for c in pc) + " |"
            for a, p in s["paired"].items()]
    return "\n".join(out) + "\n"


def per_instance_csv(s: dict[str, Any], path: Path) -> None:
    cols = ["arm", "instance_id", "resolved", "exit_status", "error", "agent_run_s", "steps", "llm_calls",
            "fast_calls", "deep_calls", "llm_cost_usd", "s1_cost_usd", "total_cost_usd", "s1_calls"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for arm, d in s["data"].items():
            for iid in sorted(d):
                r = d[iid]
                w.writerow([arm, iid, "" if r["resolved"] is None else int(bool(r["success"])), r["exit_status"],
                            (str(r["error"]).splitlines()[0][:120] if r["error"] else ""), f"{r['time_s']:.1f}",
                            r["steps"], r["llm_calls"], r["fast_calls"], r["deep_calls"], f"{r['s2_cost']:.6f}",
                            f"{r['s1_cost']:.6f}", f"{r['total_cost']:.6f}", r["s1_calls"]])


def cmd_summarize(a: argparse.Namespace) -> None:
    run = Path(a.run_dir)
    arms = a.arms or PHASES[a.phase]["arms"]
    s = summarize(run, arms)
    out = Path(a.out_dir) if a.out_dir else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or "summary"
    (out / f"{name}.md").write_text(markdown(s, a.title or f"{EXP.upper()}: JevAgentHarness on mini-SWE-agent ({a.phase})"))
    js = {k: v for k, v in s.items() if k != "data"}
    if (run / "threshold.json").exists():
        js["threshold"] = json.loads((run / "threshold.json").read_text())
    (out / f"{name}.json").write_text(json.dumps(js, indent=1, default=str) + "\n")
    per_instance_csv(s, out / ("per_instance.csv" if name == "summary" else f"{name}_per_instance.csv"))
    # the per-instance rows (load_arm), small: what the E4r readout reads after the fetch (no trajectory needed)
    (out / ("rows.json" if name == "summary" else f"{name}_rows.json")).write_text(
        json.dumps(s["data"], default=str, separators=(",", ":")) + "\n")
    if a.tex:
        vals = tex_values(s)
        th = IDS_DIR / "threshold.json"
        if th.exists():
            vals[f"{PREFIX}/tau"] = _num(json.loads(th.read_text())["tau"], 2)
        vals[f"{PREFIX}/n_eval"] = str(len(read_ids(IDS_DIR / "eval_ids.txt")))
        if (IDS_DIR / "dev_ids.txt").exists():
            vals[f"{PREFIX}/n_dev"] = str(len(read_ids(IDS_DIR / "dev_ids.txt")))
        if a.dev_summary and Path(a.dev_summary).exists():
            dev = json.loads(Path(a.dev_summary).read_text())
            for r in dev.get("arms", []):
                vals[f"{PREFIX}/dev/{r['arm']}/resolved"] = str(r["resolved"])
                vals[f"{PREFIX}/dev/{r['arm']}/fast_share"] = _num(100 * r["fast_share"] if r.get("fast_share")
                                                             is not None else None, 1)
        write_tex(vals)
    print((out / f"{name}.md").read_text())


# ----------------------------------------------------------------------------------------------- E4r readout
E4R_ARMS = ("always_deep", "jevagent", "always_fast")
E4R_MARGIN = 0.03                    # the paper's non-inferiority margin (3 points of success), descriptive here
E4R_BOOT, E4R_SEED = 10_000, 0


def rows_from_csv(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """A per_instance.csv written by :func:`per_instance_csv` -> arm -> instance -> row (the fields the E4r readout
    needs; an empty `resolved` cell = unevaluated, counted unresolved)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            res = None if r["resolved"] == "" else bool(int(r["resolved"]))
            out.setdefault(r["arm"], {})[r["instance_id"]] = {
                "success": float(bool(res)), "resolved": res, "time_s": float(r["agent_run_s"]),
                "error": r["error"] or None, "exit_status": r["exit_status"], "steps": int(r["steps"]),
                "llm_calls": int(r["llm_calls"]), "fast_calls": int(r["fast_calls"]), "deep_calls": int(r["deep_calls"]),
                "s2_cost": float(r["llm_cost_usd"]), "s1_cost": float(r["s1_cost_usd"]),
                "total_cost": float(r["total_cost_usd"]), "s1_calls": int(r["s1_calls"])}
    return out


def _boot_ratio_saving(base: Mapping[str, float], arm: Mapping[str, float], keys: list[str], n: int = E4R_BOOT,
                       seed: int = E4R_SEED) -> tuple[float, float, float]:
    """100 * (1 - mean(arm) / mean(base)) over `keys` and its paired percentile bootstrap 95 % interval (instances
    resampled with replacement, the same draw for both arms)."""
    import numpy as np
    b = np.array([float(base[k]) for k in keys])
    a = np.array([float(arm[k]) for k in keys])
    if not len(keys) or b.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n, len(keys)))
    s = 100 * (1 - a[idx].mean(axis=1) / b[idx].mean(axis=1))
    return float(100 * (1 - a.mean() / b.mean())), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def e4r_arm(rows: Mapping[str, Mapping[str, Any]], keys: list[str]) -> dict[str, Any]:
    v = [rows[k] for k in keys if k in rows]
    n = len(v)
    times = [r["time_s"] for r in v]
    calls = sum(r["llm_calls"] for r in v)
    return {"n": n, "resolved": int(sum(r["success"] for r in v)),
            "success_pct": 100 * sum(r["success"] for r in v) / n if n else None,
            "unevaluated": sum(r["resolved"] is None for r in v), "errors": sum(bool(r["error"]) for r in v),
            "cost_mean": sum(r["total_cost"] for r in v) / n if n else None,
            "llm_cost_mean": sum(r["s2_cost"] for r in v) / n if n else None,
            "s1_cost_mean": sum(r["s1_cost"] for r in v) / n if n else None,
            "cost_total": sum(r["total_cost"] for r in v),
            "time_mean_s": sum(times) / n if n else None, "time_median_s": _q(times, 0.5),
            "time_p90_s": _q(times, 0.9), "steps_mean": sum(r["steps"] for r in v) / n if n else None,
            "fast_share": sum(r["fast_calls"] for r in v) / calls if calls else None,
            "limits_exceeded": sum(str(r["exit_status"]) == "LimitsExceeded" for r in v)}


def e4r_paired(base: Mapping[str, Mapping[str, Any]], arm: Mapping[str, Mapping[str, Any]],
               keys: list[str]) -> dict[str, Any]:
    """`arm` against `base` on the instances of `keys` both have: success difference (points) with the paired task
    bootstrap 95 % interval (B = 10,000, seed 0), Tango's paired 95 % score interval, the paper's one-sided
    non-inferiority test at 3 points (metrics.ni_certified_p, rule bb_ordered; 'yes' when p <= 0.05) -- descriptive;
    cost and time saving (ratio of means) with paired bootstrap intervals; the median time difference on co-solved
    instances."""
    from jevharness.bench import metrics as M
    ks = [k for k in keys if k in base and k in arm]
    n = len(ks)
    w = sum(arm[k]["success"] > base[k]["success"] for k in ks)
    l_ = sum(arm[k]["success"] < base[k]["success"] for k in ks)
    both = sum(arm[k]["success"] and base[k]["success"] for k in ks)
    boot = M.paired_bootstrap_diff({k: base[k]["success"] for k in ks}, {k: arm[k]["success"] for k in ks},
                                   n=E4R_BOOT, seed=E4R_SEED)
    tango = M.tango_interval(both, w, l_, n - both - w - l_)["ci"] if n else [float("nan")] * 2
    ni = M.ni_certified_p(w, l_, n, E4R_MARGIN) if n else {"p": float("nan")}
    cs, cs_lo, cs_hi = _boot_ratio_saving({k: base[k]["total_cost"] for k in ks},
                                          {k: arm[k]["total_cost"] for k in ks}, ks)
    ts, ts_lo, ts_hi = _boot_ratio_saving({k: base[k]["time_s"] for k in ks}, {k: arm[k]["time_s"] for k in ks}, ks)
    co = [arm[k]["time_s"] - base[k]["time_s"] for k in ks if arm[k]["success"] and base[k]["success"]]
    return {"n": n, "wins": w, "losses": l_, "d_resolved": w - l_,
            "delta_pt": 100 * (w - l_) / n if n else None,
            "delta_boot_ci_pt": [100 * boot["ci_lo"], 100 * boot["ci_hi"]],
            "delta_tango_ci_pt": [100 * tango[0], 100 * tango[1]],
            "ni_margin_pt": 100 * E4R_MARGIN, "ni_p": ni["p"], "ni": "yes" if ni["p"] <= 0.05 else "no",
            "cost_saving_pct": cs, "cost_saving_ci": [cs_lo, cs_hi],
            "time_saving_pct": ts, "time_saving_ci": [ts_lo, ts_hi],
            "n_cosolved": len(co), "d_time_median_cosolved_s": _q(co, 0.5)}


def _kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def e4r_readout(rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                e4_rows: Mapping[str, Mapping[str, Mapping[str, Any]]], sel: Mapping[str, Any],
                ids_dir: Path | None = None) -> dict[str, Any]:
    """The registered E4r readout (as amended for minimum spend). rows: E4r's eval rows (arm -> instance -> row; the
    200 extension instances); e4_rows: E4's eval rows (the 100 replicate instances). Views: pooled = E4's run on its
    100 + E4r's run on the 200 = 300 distinct held-out instances (the primary; every arm of an instance comes from
    the same run), extension (E4r's 200 alone), e4 (E4's 100 alone). Only when rows also hold replicate instances
    (the original E4r design, not run): the replicate view, the E4r-run pool and the replicate agreement."""
    d = ids_dir or IDS_DIR
    rep = read_ids(d / "replicate_ids.txt")
    ext = read_ids(d / "extension_ids.txt")
    arms = [a for a in E4R_ARMS if a in rows]
    e4_ext = {a: {**{k: v for k, v in e4_rows.get(a, {}).items() if k in set(rep)},
                  **{k: v for k, v in rows[a].items() if k in set(ext)}} for a in arms if a in e4_rows}
    views = {"pooled": (e4_ext, rep + ext), "extension": (rows, ext), "e4": (e4_rows, rep)}
    if any(k in rows.get("always_deep", {}) for k in rep):                      # the original design (not run)
        views.update({"replicate": (rows, rep), "e4r_run": (rows, rep + ext)})
    out: dict[str, Any] = {"label": LABEL, "tau": json.loads((d / "threshold.json").read_text())["tau"],
                           "n_replicate": len(rep), "n_extension": len(ext), "margin_pt": 100 * E4R_MARGIN,
                           "selection": {k: sel[k]["sha256"] for k in ("replicate", "extension", "eval") if k in sel},
                           "views": {}}
    for name, (data, keys) in views.items():
        if not data or "always_deep" not in data:
            continue
        # complete set: the instances of the view that every present arm has a record for
        ks = [k for k in keys if all(k in data[a] for a in arms if a in data)]
        if not ks:
            continue
        out["views"][name] = {"n_listed": len(keys), "n": len(ks),
                              "missing": len(keys) - len(ks),
                              "arms": {a: e4r_arm(data[a], ks) for a in arms if a in data},
                              "paired": {a: e4r_paired(data["always_deep"], data[a], ks)
                                         for a in arms if a in data and a != "always_deep"}}
    agree: dict[str, Any] = {}
    for a in (arms if "replicate" in views else []):                           # the original design only
        if a not in e4_rows:
            continue
        ks = [k for k in rep if k in rows[a] and k in e4_rows[a]]
        x = [bool(e4_rows[a][k]["success"]) for k in ks]
        y = [bool(rows[a][k]["success"]) for k in ks]
        agree[a] = {"n": len(ks), "e4_resolved": sum(x), "e4r_resolved": sum(y),
                    "same_outcome_pct": 100 * sum(p == q for p, q in zip(x, y)) / len(ks) if ks else None,
                    "both": sum(p and q for p, q in zip(x, y)), "only_e4": sum(p and not q for p, q in zip(x, y)),
                    "only_e4r": sum(q and not p for p, q in zip(x, y)), "kappa": _kappa(x, y)}
    out["agreement"] = agree
    if "e4" in out["views"] and "replicate" in out["views"]:
        out["agreement_routing"] = {
            run: {k: out["views"][v]["paired"].get("jevagent", {}).get(k) for k in
                  ("delta_pt", "wins", "losses", "cost_saving_pct", "time_saving_pct", "ni")}
            for run, v in (("e4", "e4"), ("e4r", "replicate"))}
    return out


def e4r_tex_values(r: Mapping[str, Any]) -> dict[str, str]:
    """keys e4r/...: the pooled view in E4's key layout (e4r/<arm>/..., e4r/<arm>/vs_deep/...), the other views under
    e4r/{rep,ext,e4,e4ext}/..., replicate agreement under e4r/agree/..., logged spend under e4r/spend/..."""
    v: dict[str, str] = {"e4r/tau": _num(r["tau"], 2), "e4r/n_rep": str(r["n_replicate"]),
                         "e4r/n_ext": str(r["n_extension"]), "e4r/margin": _num(r["margin_pt"], 0)}
    for view, vd in r["views"].items():
        p = "e4r" + {"pooled": "", "replicate": "/rep", "extension": "/ext", "e4": "/e4", "e4r_run": "/e4rrun"}[view]
        v[f"{p}/n_eval"] = str(vd["n"])
        for a, s in vd["arms"].items():
            v[f"{p}/{a}/n"] = str(s["n"])
            v[f"{p}/{a}/resolved"] = str(s["resolved"])
            v[f"{p}/{a}/success"] = _num(s["success_pct"], 1)
            v[f"{p}/{a}/cost_usd"] = _num(s["cost_mean"], 4)
            v[f"{p}/{a}/llm_cost_usd"] = _num(s["llm_cost_mean"], 4)
            v[f"{p}/{a}/s1_cost_usd"] = _num(s["s1_cost_mean"], 4)
            v[f"{p}/{a}/wall_mean_s"] = _num(s["time_mean_s"], 0)
            v[f"{p}/{a}/wall_median_s"] = _num(s["time_median_s"], 0)
            v[f"{p}/{a}/steps"] = _num(s["steps_mean"], 1)
            v[f"{p}/{a}/fast_share"] = _num(100 * s["fast_share"] if s["fast_share"] is not None else None, 1)
            v[f"{p}/{a}/limits_exceeded"] = str(s["limits_exceeded"])
            v[f"{p}/{a}/errors"] = str(s["errors"])
        for a, q in vd["paired"].items():
            b = f"{p}/{a}/vs_deep"
            v[f"{b}/n"] = str(q["n"])
            v[f"{b}/wins"], v[f"{b}/losses"] = str(q["wins"]), str(q["losses"])
            v[f"{b}/d_resolved"] = _num(q["d_resolved"], 0, signed=True)
            v[f"{b}/delta"] = _num(q["delta_pt"], 1, signed=True)
            v[f"{b}/delta_lo"], v[f"{b}/delta_hi"] = (_num(x, 1, signed=True) for x in q["delta_boot_ci_pt"])
            v[f"{b}/tango_lo"], v[f"{b}/tango_hi"] = (_num(x, 1, signed=True) for x in q["delta_tango_ci_pt"])
            v[f"{b}/ni_p"] = _num(q["ni_p"], 3)
            v[f"{b}/ni"] = q["ni"]
            v[f"{b}/saving_pct"] = _num(q["cost_saving_pct"], 1)
            v[f"{b}/saving_lo"], v[f"{b}/saving_hi"] = (_num(x, 1) for x in q["cost_saving_ci"])
            v[f"{b}/time_saving_pct"] = _num(q["time_saving_pct"], 1)
            v[f"{b}/time_saving_lo"], v[f"{b}/time_saving_hi"] = (_num(x, 1) for x in q["time_saving_ci"])
            v[f"{b}/time_cosolved/n"] = str(q["n_cosolved"])
            v[f"{b}/time_cosolved/d_median_s"] = _num(q["d_time_median_cosolved_s"], 1, signed=True)
    for a, g in r.get("agreement", {}).items():
        for k in ("n", "e4_resolved", "e4r_resolved", "both", "only_e4", "only_e4r"):
            v[f"e4r/agree/{a}/{k}"] = str(g[k])
        v[f"e4r/agree/{a}/same_pct"] = _num(g["same_outcome_pct"], 0)
        v[f"e4r/agree/{a}/kappa"] = _num(g["kappa"], 2)
    for k, sp in (r.get("spend") or {}).items():
        v[f"e4r/spend/{k}"] = _num(sp, 2)
    return v


def e4r_markdown(r: Mapping[str, Any]) -> str:
    names = {"pooled": f"Pooled (primary): {r['n_replicate'] + r['n_extension']} distinct held-out instances = E4's "
                       f"{r['n_replicate']} (E4's run) + E4r's {r['n_extension']} new ones",
             "extension": f"Extension: E4r's {r['n_extension']} new held-out instances",
             "e4": f"E4: its {r['n_replicate']} evaluation instances (E4's run, as reported)",
             "replicate": f"Replicate: E4's {r['n_replicate']} eval instances, run again",
             "e4r_run": "E4r's own run on replicate + extension"}
    out = [f"# E4r: replicate and extension of E4 (mini-SWE-agent, SWE-bench Verified)", "",
           f"**{LABEL}.** Outside every Holm table; revises no registered result. Routing tau = {r['tau']} (E4's). "
           f"Success difference with the paired bootstrap (B = {E4R_BOOT:,}, seed {E4R_SEED}) and Tango 95 % "
           f"intervals; the one-sided non-inferiority test at {r['margin_pt']:.0f} points (the paper's rule, "
           "metrics.ni_certified_p) is reported descriptively. Cost = API $ per instance as logged (DeepSeek "
           "`usage.cost` + Jev); time = agent-run wall clock per instance; savings are ratios of means with paired "
           "bootstrap 95 % intervals.", ""]
    for view, vd in r["views"].items():
        out += [f"## {names.get(view, view)}", "",
                f"n = {vd['n']} instances with a record in every arm ({vd['missing']} of {vd['n_listed']} listed "
                "missing).", "",
                "| arm | n | resolved | success % | $ / instance | Jev $ / inst. | mean s | median s | steps | "
                "fast-share % | limits exceeded | errors |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, s in vd["arms"].items():
            out.append(f"| {a} | {s['n']} | {s['resolved']} | {_fmt(s['success_pct'], 1)} | {_fmt(s['cost_mean'], 4)} "
                       f"| {_fmt(s['s1_cost_mean'], 4)} | {_fmt(s['time_mean_s'], 1)} | {_fmt(s['time_median_s'], 1)} "
                       f"| {_fmt(s['steps_mean'], 1)} | "
                       f"{_fmt(100 * s['fast_share'] if s['fast_share'] is not None else None, 1)} "
                       f"| {s['limits_exceeded']} | {s['errors']} |")
        out += ["", "| vs always_deep | wins / losses | delta pt [bootstrap 95 %] | Tango 95 % | NI p (3 pt) | NI | "
                    "cost saved % [95 %] | time saved % [95 %] | co-solved: n, median d s |",
                "|---|---|---|---|---|---|---|---|---|"]
        for a, q in vd["paired"].items():
            out.append(f"| {a} | {q['wins']} / {q['losses']} | {_fmt(q['delta_pt'], 1)} "
                       f"[{_fmt(q['delta_boot_ci_pt'][0], 1)}, {_fmt(q['delta_boot_ci_pt'][1], 1)}] "
                       f"| [{_fmt(q['delta_tango_ci_pt'][0], 1)}, {_fmt(q['delta_tango_ci_pt'][1], 1)}] "
                       f"| {_fmt(q['ni_p'], 3)} | {q['ni']} | {_fmt(q['cost_saving_pct'], 1)} "
                       f"[{_fmt(q['cost_saving_ci'][0], 1)}, {_fmt(q['cost_saving_ci'][1], 1)}] "
                       f"| {_fmt(q['time_saving_pct'], 1)} [{_fmt(q['time_saving_ci'][0], 1)}, "
                       f"{_fmt(q['time_saving_ci'][1], 1)}] | {q['n_cosolved']}, "
                       f"{_fmt(q['d_time_median_cosolved_s'], 1)} |")
        out.append("")
    if r.get("agreement"):
        out += ["## Replicate agreement (E4 vs E4r on the same instances)", "",
                "| arm | n | E4 resolved | E4r resolved | same outcome % | both | only E4 | only E4r | kappa |",
                "|---|---|---|---|---|---|---|---|---|"]
        for a, g in r["agreement"].items():
            out.append(f"| {a} | {g['n']} | {g['e4_resolved']} | {g['e4r_resolved']} | {_fmt(g['same_outcome_pct'], 1)} "
                       f"| {g['both']} | {g['only_e4']} | {g['only_e4r']} | {_fmt(g['kappa'], 2)} |")
        if r.get("agreement_routing"):
            out += ["", "| routing vs always_deep | delta pt | wins / losses | cost saved % | time saved % | NI |",
                    "|---|---|---|---|---|---|"]
            for run, g in r["agreement_routing"].items():
                out.append(f"| {run.upper()} | {_fmt(g['delta_pt'], 1)} | {g['wins']} / {g['losses']} "
                           f"| {_fmt(g['cost_saving_pct'], 1)} | {_fmt(g['time_saving_pct'], 1)} | {g['ni']} |")
        out.append("")
    if r.get("spend"):
        out += ["## Spend (logged)", "", "| item | $ |", "|---|---|"]
        out += [f"| {k} | {_fmt(v, 2)} |" for k, v in r["spend"].items()]
        out.append("")
    return "\n".join(out) + "\n"


def cmd_readout(a: argparse.Namespace) -> None:
    """E4r: rows.json (written on the box by `summarize`) + E4's committed per_instance.csv -> results/e4r_swe/
    {summary.md, summary.json, per_instance.csv} and paper/generated/e4r_swe.tex."""
    if EXP != "e4r":
        raise SystemExit("readout is E4r's (--exp e4r)")
    rows = json.loads(Path(a.rows).read_text())
    e4_rows = rows_from_csv(Path(a.e4_csv))
    sel = json.loads((IDS_DIR / "selection.json").read_text())
    r = e4r_readout(rows, e4_rows, sel)
    if a.spend and Path(a.spend).exists():
        r["spend"] = spend_flat(json.loads(Path(a.spend).read_text()))
    out = Path(a.out_dir) if a.out_dir else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(e4r_markdown(r))
    per_instance_csv({"data": rows}, out / "per_instance.csv")
    if not a.no_tex:
        write_tex(e4r_tex_values(r), Path(a.tex_out) if a.tex_out else None)
    print((out / "summary.md").read_text())


def spend_flat(sp: Mapping[str, Any]) -> dict[str, float]:
    """spend.json (``{"logged": {"smoke": .., "eval": .., "total": ..}, "key_usage_start": .., "key_usage_end": ..}``,
    written by the drivers) -> flat dollar figures for the readout (logged_*, key_usage = end - start)."""
    out: dict[str, float] = {}
    for k, v in (sp.get("logged") or {}).items():
        try:
            out[f"logged_{k}"] = float(v)
        except (TypeError, ValueError):
            continue
    try:
        out["key_usage"] = float(sp["key_usage_end"]) - float(sp["key_usage_start"])
    except (KeyError, TypeError, ValueError):
        pass
    return out


# ----------------------------------------------------------------------------------------------- E4r smoke gate
E4R_CAP_USD = 40.0                   # OpenRouter (DeepSeek + Jev) for E4r in total, smoke included (minimum-spend amendment)
E4R_SAFETY = 1.5                     # projection factor (as E4b's registered gate)
E4R_FALLBACK_K = 100                 # the projection for the whole list does not fit -> the first 100 extension instances


def logged_spend(run: Path) -> float:
    """Logged $ (agent LLM + Jev, `usage.total.cost_usd`) of every record under a phase directory, the errored
    attempts moved aside by the one re-run included."""
    tot = 0.0
    for f in list(Path(run).glob("*/*/*.jev.json")) + list(Path(run).glob("*/_errored/*/*.jev.json")):
        try:
            tot += _cost(json.loads(f.read_text()).get("usage"), "total")
        except (OSError, ValueError):
            continue
    return tot


def smoke_check(run: Path, arms: list[str], ids: list[str], min_deep_reasoning: float = 0.95) -> dict[str, Any]:
    """The registered smoke conditions: every (instance, arm) has a record without an infrastructure error; every agent
    call was served by the pinned provider (StreamLake); every fast-path call reported 0 reasoning tokens and at least
    95 % of deep-path calls reported reasoning tokens (the toggle works); the routing arm's Jev calls answered (at least
    one per record, at most 5 % System-1 errors: an error fails open to the deep path, which would hide a broken route).
    Read from mini-SWE-agent's trajectories and the records."""
    out: dict[str, Any] = {"records": {}, "errors": {}, "providers": Counter(), "deep_calls": 0, "deep_reasoning": 0,
                           "fast_calls": 0, "fast_reasoning": 0, "other_models": Counter(), "s1_calls": 0,
                           "s1_errors": 0, "s1_records_without_calls": 0}
    for arm in arms:
        recs = {iid: Path(run) / arm / iid / f"{iid}.jev.json" for iid in ids}
        out["records"][arm] = sum(p.exists() for p in recs.values())
        out["errors"][arm] = sum(p.exists() and _record_error(p) for p in recs.values())
        if arm.startswith("jevagent"):
            for rp in recs.values():
                if not rp.exists():
                    continue
                rec = json.loads(rp.read_text())
                n1 = int((((rec.get("usage") or {}).get("s1") or {}).get("calls")) or 0)
                out["s1_calls"] += n1
                out["s1_errors"] += len(rec.get("s1_errors") or [])
                out["s1_records_without_calls"] += n1 == 0
        for iid in ids:
            tp = Path(run) / arm / iid / f"{iid}.traj.json"
            if not tp.exists():
                continue
            for m in json.loads(tp.read_text()).get("messages") or []:
                resp = ((m.get("extra") or {}).get("response") or {}) if m.get("role") == "assistant" else {}
                if not resp:
                    continue
                out["providers"][str(resp.get("provider"))] += 1
                rt = int((((resp.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"))
                         or 0)
                model = str(resp.get("model") or "")
                if model.startswith(DEEP_MODEL):
                    out["deep_calls"] += 1
                    out["deep_reasoning"] += rt > 0
                elif model.startswith(FAST_MODEL):
                    out["fast_calls"] += 1
                    out["fast_reasoning"] += rt > 0
                else:
                    out["other_models"][model] += 1
    out["providers"], out["other_models"] = dict(out["providers"]), dict(out["other_models"])
    fails = []
    if any(out["records"][a] < len(ids) for a in arms):
        fails.append(f"missing records {out['records']}")
    if any(out["errors"].values()):
        fails.append(f"infrastructure errors {out['errors']}")
    if set(out["providers"]) != {"StreamLake"}:
        fails.append(f"providers {out['providers']}")
    if out["other_models"]:
        fails.append(f"other models {out['other_models']}")
    if out["fast_calls"] == 0 or out["fast_reasoning"] > 0:
        fails.append(f"fast path: {out['fast_reasoning']} of {out['fast_calls']} calls reasoned")
    if out["deep_calls"] == 0 or out["deep_reasoning"] < min_deep_reasoning * out["deep_calls"]:
        fails.append(f"deep path: {out['deep_reasoning']} of {out['deep_calls']} calls reasoned")
    if any(a.startswith("jevagent") for a in arms) and (
            out["s1_records_without_calls"] or out["s1_errors"] > 0.05 * max(1, out["s1_calls"])):
        fails.append(f"Jev: {out['s1_calls']} calls, {out['s1_errors']} errors, "
                     f"{out['s1_records_without_calls']} routing record(s) without a Jev call")
    out["fails"], out["ok"] = fails, not fails
    return out


def gate_decision(smoke_means: Mapping[str, float], e4_means: Mapping[str, float], spent: float, n_eval: int,
                  cap: float = E4R_CAP_USD, safety: float = E4R_SAFETY, fallback_k: int = E4R_FALLBACK_K) -> dict[str, Any]:
    """The registered projection gate (as amended for minimum spend): per arm the larger of the smoke's and E4's eval
    mean $ per instance; projection = safety x sum over arms x instances. The whole run list fits what is left of the
    cap -> run it; else the first `fallback_k` instances fit -> run those; else stop after the smoke."""
    per = {a: max(float(smoke_means.get(a) or 0.0), float(e4_means.get(a) or 0.0)) for a in set(smoke_means) | set(e4_means)}
    unit = safety * sum(per.values())
    proj = unit * n_eval
    base = {"projection": proj, "spent": spent, "cap": cap, "per_instance": per, "safety": safety}
    if spent + proj <= cap:
        return {"decision": "full", "k": n_eval, **base}
    k = min(fallback_k, n_eval)
    if spent + unit * k <= cap:
        return {"decision": "cut", "k": k, "projection_k": unit * k, **base}
    return {"decision": "stop", "k": 0, "projection_k": unit * k, **base}


PACK_CALL_FIELDS = ("component", "model", "decision_point", "input_tokens", "output_tokens", "cost_usd", "latency_ms",
                    "cached", "step")


def compact_record(rec: Mapping[str, Any]) -> dict[str, Any]:
    """A record without its trajectory, patch or free text: what the readout and an audit of cost and latency need."""
    calls = []
    for c in rec.get("calls") or []:
        d = {k: c.get(k) for k in PACK_CALL_FIELDS}
        m = c.get("meta") or {}
        d.update({k: m[k] for k in ("role", "cached_input_tokens") if k in m})
        calls.append(d)
    err = rec.get("error")
    return {"instance_id": rec.get("instance_id"), "arm": rec.get("arm"), "exit_status": rec.get("exit_status"),
            "result": {k: (rec.get("result") or {}).get(k) for k in ("success", "resolved", "exit_status")},
            "usage": rec.get("usage"), "counters": rec.get("counters"), "models": rec.get("models"),
            "s1_errors": len(rec.get("s1_errors") or []), "wall_ms": rec.get("wall_ms"),
            "error": str(err).splitlines()[0][:200] if err else None, "mini_version": rec.get("mini_version"),
            "calls": calls}


def scrub(text: str) -> str:
    """Local paths out of packed text: the checkout's path -> <repo>, the home directory -> ~ (results are published)."""
    import os
    home = os.path.expanduser("~")
    for path, rep in ((str(ROOT), "<repo>"), (home if home not in ("/", "/root", "/home") else "", "~")):
        if len(path) > 1:
            text = text.replace(path, rep)
    return text


def cmd_pack(a: argparse.Namespace) -> None:
    """runs_dir/{smoke,eval}/<arm>/<iid>/<iid>.jev.json -> out/records_<phase>.jsonl.gz (compact records, the errored
    attempts moved aside by the re-run included), out/preds_<phase>_<arm>.json.gz (the submitted patches), out/
    report_<phase>_<arm>.json.gz (the SWE-bench reports), out/rows_<phase>.json.gz and out/summary_<phase>.md."""
    import gzip
    src, out = Path(a.runs_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for phase in ("smoke", "eval"):
        d = src / phase
        if not d.exists():
            continue
        recs = sorted(d.glob("*/*/*.jev.json")) + sorted(d.glob("*/_errored/*/*.jev.json"))
        with gzip.open(out / f"records_{phase}.jsonl.gz", "wt") as f:
            for rp in recs:
                try:
                    rec = json.loads(rp.read_text())
                except (OSError, ValueError):
                    continue
                f.write(scrub(json.dumps({**compact_record(rec), "errored_attempt": "_errored" in rp.parts})) + "\n")
        for arm in PHASES.get(phase, {}).get("arms", []):
            for name, pth in (("preds", d / arm / "preds.json"),
                              ("report", next(iter(sorted((d / arm / "eval").glob(f"*.{EXP}_{phase}_{arm}.json"))), None)
                               if (d / arm / "eval").exists() else None)):
                if pth and Path(pth).exists():
                    (out / f"{name}_{phase}_{arm}.json.gz").write_bytes(gzip.compress(scrub(Path(pth).read_text()).encode()))
        if (d / "summary" / "rows.json").exists():
            (out / f"rows_{phase}.json.gz").write_bytes(gzip.compress(scrub((d / "summary" / "rows.json").read_text()).encode()))
        if (d / "summary" / "summary.md").exists():
            (out / f"summary_{phase}.md").write_text(scrub((d / "summary" / "summary.md").read_text()))
    print(json.dumps({p.name: p.stat().st_size for p in sorted(out.iterdir())}))


def cmd_spend(a: argparse.Namespace) -> None:
    out = {Path(d).name: round(logged_spend(Path(d)), 4) for d in a.run_dirs}
    out["total"] = round(sum(out.values()), 4)
    print(json.dumps(out))

def cmd_gate(a: argparse.Namespace) -> None:  # runs on the box after the smoke (exit 0 go, 3 stop, 4 smoke failed)
    if EXP != "e4r":
        raise SystemExit("gate is E4r's (--exp e4r)")
    run = Path(a.smoke_dir)
    arms = PHASES["smoke"]["arms"]
    chk = smoke_check(run, arms, read_ids(IDS_DIR / "smoke_ids.txt"))
    smoke_means = {}
    for arm in arms:
        cs = [_cost(json.loads(f.read_text()).get("usage"), "total") for f in (run / arm).glob("*/*.jev.json")]
        smoke_means[arm] = sum(cs) / len(cs) if cs else 0.0
    e4 = rows_from_csv(Path(a.e4_csv))
    e4_means = {arm: sum(r["total_cost"] for r in e4[arm].values()) / len(e4[arm]) for arm in arms if arm in e4}
    spent = logged_spend(run) + float(a.spent_extra or 0.0)
    dec = gate_decision(smoke_means, e4_means, spent, len(read_ids(IDS_DIR / "eval_ids.txt")), cap=a.cap)
    out = {"smoke_check": chk, "smoke_means": smoke_means, "e4_means": e4_means, **dec,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if not chk["ok"]:
        out["decision"] = "smoke_failed"
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("decision", "k", "projection", "spent", "cap")}, default=str))
    raise SystemExit({"full": 0, "cut": 0, "stop": 3, "smoke_failed": 4}[out["decision"]])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--pool-file", default=None, help="the 500 Verified test ids, one per line (default: HF dataset)")
    p = sub.add_parser("readout", help="E4r only: rows.json from the box + E4's per_instance.csv -> the E4r readout")
    p.add_argument("rows")
    p.add_argument("--e4-csv", default=str(ROOT / "results" / "e4_swe" / "per_instance.csv"))
    p.add_argument("--spend", default=None, help="spend.json written by scripts/aws/e4r_swe.sh (logged $)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-tex", action="store_true")
    p.add_argument("--tex-out", default=None, help="write the keys .tex here instead of paper/generated/e4r_swe.tex")
    p = sub.add_parser("pack", help="E4r: small gzipped records of a run for results/ (no trajectory, no log)")
    p.add_argument("runs_dir", help="the directory holding smoke/ and eval/")
    p.add_argument("--out", required=True)
    p = sub.add_parser("spend", help="logged $ of every record under the given phase directories (JSON)")
    p.add_argument("run_dirs", nargs="+")
    p = sub.add_parser("gate", help="E4r only, on the box: the registered smoke checks and projection gate")
    p.add_argument("smoke_dir")
    p.add_argument("--e4-csv", default=str(ROOT / "results" / "e4_swe" / "per_instance.csv"))
    p.add_argument("--cap", type=float, default=E4R_CAP_USD)
    p.add_argument("--spent-extra", type=float, default=0.0, help="spend outside the smoke records (e.g. API checks)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--phase", choices=sorted(PHASES), required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-w", "--workers", type=int, default=24)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--ids", default=None)
    p.add_argument("-c", "--config", action="append", default=None, help="extra mini-swe config specs (all arms)")
    p.add_argument("--rerun-errored", action="store_true")
    p.add_argument("--arm-limit", action="append", default=None,
                   help="ARM=N: run ARM only on the first N instances of the list (E4b's registered cost rule)")
    p = sub.add_parser("threshold")
    p.add_argument("run_dir")
    p.add_argument("--out", default=None)
    p = sub.add_parser("summarize")
    p.add_argument("run_dir")
    p.add_argument("--phase", choices=sorted(PHASES), default="eval")
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--tex", action="store_true")
    p.add_argument("--dev-summary", default=None)
    ap.add_argument("--exp", choices=sorted(EXPS), default="e4", help="exploratory row (default E4)")
    a = ap.parse_args(argv)
    set_exp(a.exp)
    if a.cmd == "select" and a.exp == "e4r":
        cmd_select_e4r(a)
        return
    {"select": cmd_select, "run": cmd_run, "threshold": cmd_threshold, "summarize": cmd_summarize,
     "readout": cmd_readout, "gate": cmd_gate, "spend": cmd_spend, "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
