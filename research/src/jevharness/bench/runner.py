"""Experiment runner: (harness configs) x (tasks) -> trajectories JSONL, resumable.

Experiment YAML (see configs/experiments/):

    name: tool_select_sweep
    suite: {name: jevgym, n: 100, n_distractors: 30}
    llm: {provider: openai, model: gpt-4.1-mini, temperature: 0}
    small_llm: {provider: vllm, model: Qwen/Qwen3-4B}        # only for routing arms
    s1: {type: jev, model: jev-1.13.0}
    s1_cache: cache/jev.sqlite                                # optional
    workers: 8
    arms:
      - {name: llm_only}
      - {name: ts_replace, tool_select: {mode: replace}, tool_params: {mode: replace}}
    sweeps:                                                   # optional grid expansion
      - {base: {name: ts_cascade, tool_select: {mode: cascade}}, set: tool_select.tau,
         values: [0.5, 0.7, 0.9]}
    retry_errors: true                                        # optional (default true), see below
    n_tasks: 200                                              # optional: default tasks per arm (first N of the suite);
                                                              # an arm's own ``n_tasks`` overrides it

JevAgentHarness profiles (``harness/profile.py``, docs/JEVAGENT_HARNESS.md): an arm may name a profile instead of
setting decision points, ``{name: jevagent, profile: configs/harness/jevagent_default.yaml}`` (+ ``profile_s2``,
``profile_part``, ``profile_backend``, ``profile_env``); its decision points are filled in from the profile section of
the arm's System-2 model. Arms of a ``status: pending`` profile, or of a section / part marked ``skip``, are skipped with
the reason; the resolved profile of every such arm is written to ``<out>/profiles.json`` and to each episode's
``meta.profile``. Arms without ``profile`` are untouched.

Measurement arms (docs/META_GOLD_SPEC.md §5): an arm's ``measure`` block may name LLMs for its deferred
counterfactual / probe calls (``measure.cf.large_llm``, ``measure.probes.transplant_llm``): a key of ``llms``, or
``main`` (the experiment's ``llm``) / ``small`` (``small_llm``). They are built only when an arm references them.

Placeholder backends: an ``s1_backends`` entry with ``placeholder: true`` (the matched-rate controls, whose rate
is read off the System-1 arms of the same run) is not run: its arms are skipped with a message until the entry is
overridden (``-o 's1_backends.ratem_lite={type: rate, rate: 0.083, seed: 0}'``; ``JEV_ALLOW_PLACEHOLDER=1`` runs
them anyway, for pipeline tests). Every episode logs its System-1 spec (``meta.s1``: ``describe()`` and its id), and
a resume refuses an arm whose logged spec differs from the current one.

Schedule: arms run one after another by default (arm-major). ``schedule: {mode: task_major, seed: 732}`` runs every
arm's episode of task i before any of task i + 1, the arms in a seeded random order within each task, through one pool
of ``workers`` threads, and records the schedule in ``<out>/schedule.json`` (seed, concurrency, arms, one entry per
invocation) and ``<out>/schedule_order.jsonl`` (the job order) -- the comparable-arm scheduler of PREREG_A7_STAGE2
A7.16 (a), so paired latency comparisons see the same load and time of day.

Resume: one ``<arm>.jsonl`` per arm; a task counts as done when its *latest* record ended without an
infrastructure error (``Trajectory.error`` unset). Episodes that ended in an error (rate limit, 5xx, an exception)
are re-run on the next invocation and the fresh record is appended; loaders keep the last record per task id
(``harness.trajectory.load_trajectories``), so a retried task is never double-counted. Disable with
``retry_errors: false`` in the YAML, ``JEV_RETRY_ERRORS=0``, ``run_experiment(..., retry_errors=False)`` or
``python -m jevharness.bench.runner CONFIG --no-retry-errors``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import random
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from ..decision import make_backend
from ..decision.base import DecisionBackend
from ..harness.agent import JevAgent
from ..harness import profile as harness_profile
from ..harness.config import DP, HarnessConfig
from ..harness.trajectory import iter_records, latest_by_task
from ..llm import LLMClient, make_llm
from .registry import check_suite_seeds
from .suites import load_suite


def expand_arms(exp: dict[str, Any]) -> list[HarnessConfig]:
    """Arms may carry ``s1: <backend name>`` (a key of ``s1_backends``) and ``llm: <name>`` (a key of
    ``llms``) to compare System-1 backends / System-2 models inside one experiment; these are stored
    in ``HarnessConfig`` via the private ``_bind`` map returned by :func:`arm_bindings`."""
    arms = [dict(a) for a in exp.get("arms", [])]
    for sw in exp.get("sweeps", []) or []:
        path = sw["set"].split(".")
        for v in sw["values"]:
            arm = copy.deepcopy(sw["base"])
            node = arm
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = v
            arm["name"] = f"{sw['base']['name']}_{path[-1]}{v}"
            arms.append(arm)
    # arms are keyed by name (one JSONL per arm, resumed by task id): a duplicate name would silently
    # merge two configs into one file and skip the second arm's tasks as "done"
    defaults = exp.get("harness_defaults", {})
    names = [a.get("name", defaults.get("name", "llm_only")) for a in arms]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"duplicate arm names: {dups}")
    out = []
    for a in arms:
        a, _ = harness_profile.resolve_arm(a, exp)   # profile arms: decision points from the profile (else unchanged)
        a = {k: v for k, v in a.items() if k not in ("s1", "llm", "n_tasks")}
        out.append(HarnessConfig(**{**defaults, **a}).validate_modes())
    return out


def arm_profiles(exp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """arm name -> the resolved profile provenance (path, sha256, status, System-2 section, decision points, skip
    reason) for the arms that name a profile (``harness/profile.py``)."""
    return {a["name"]: harness_profile.resolve_arm(a, exp)[1] for a in exp.get("arms", []) if "profile" in a}


def arm_task_counts(exp: dict[str, Any]) -> dict[str, int]:
    """arm name -> number of tasks it runs (the first N of the suite): the arm's ``n_tasks``, else the experiment's
    ``n_tasks``; arms without either run every suite task (absent from the map)."""
    default = exp.get("n_tasks")
    out: dict[str, int] = {}
    for a in exp.get("arms", []):
        n = a.get("n_tasks", default)
        if n is not None:
            out[a["name"]] = int(n)
    for sw in exp.get("sweeps", []) or []:
        n = sw["base"].get("n_tasks", default)
        if n is not None:
            for v in sw["values"]:
                out[f"{sw['base']['name']}_{sw['set'].split('.')[-1]}{v}"] = int(n)
    return out


def measure_llm_names(arms: list[HarnessConfig]) -> set[str]:
    names: set[str] = set()
    for a in arms:
        m = a.measure
        if m.cf is not None and m.cf.large_llm:
            names.add(m.cf.large_llm)
        if m.probes is not None and m.probes.transplant_llm:
            names.add(m.probes.transplant_llm)
    return names - {"main", "small"}


def arm_bindings(exp: dict[str, Any]) -> dict[str, dict[str, str]]:
    """arm name -> {"s1": backend name, "llm": llm name} for arms that override the defaults."""
    binds: dict[str, dict[str, str]] = {}
    for a in exp.get("arms", []):
        b = {k: a[k] for k in ("s1", "llm") if k in a}
        if b:
            binds[a["name"]] = b
    for sw in exp.get("sweeps", []) or []:
        b = {k: sw["base"][k] for k in ("s1", "llm") if k in sw["base"]}
        if b:
            for v in sw["values"]:
                binds[f"{sw['base']['name']}_{sw['set'].split('.')[-1]}{v}"] = b
    return binds


def _done_keys(path: Path, retry_errors: bool = True) -> set[str]:
    """Task ids of ``path`` that need no (re-)run: those whose latest record ended without an infrastructure
    error. With ``retry_errors=False`` every recorded task counts as done (the pre-retry behaviour). Partial
    trailing lines are ignored, so a task killed mid-write is re-run either way."""
    if not path.exists():
        return set()
    with warnings.catch_warnings():  # the partial-line warning is expected here: that task is simply re-run
        warnings.simplefilter("ignore")
        last = latest_by_task(r for r in iter_records(path) if "task_id" in r)
    return {k for k, r in last.items() if not (retry_errors and r.get("error"))}


def _logged_s1_ids(path: Path) -> dict[str, str]:
    """task id -> the System-1 spec id (``meta.s1.id``) of its latest record (records without one are skipped)."""
    if not path.exists():
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        last = latest_by_task(r for r in iter_records(path) if "task_id" in r)
    return {k: str(((r.get("meta") or {}).get("s1") or {}).get("id")) for k, r in last.items()
            if ((r.get("meta") or {}).get("s1") or {}).get("id")}


def _check_resume_spec(arm: str, path: Path, backend: DecisionBackend | None) -> None:
    """Refuse to resume an arm whose System-1 spec changed since its records were written (round-3 review: resume
    skips done task ids whatever produced them, so ``-o s1_backends.ratem_lite.rate=...`` after a placeholder run
    silently kept the placeholder episodes)."""
    if backend is None:
        return
    from ..decision.base import backend_id

    want = backend_id(backend.describe())
    old = sorted(set(_logged_s1_ids(path).values()) - {want})
    if old:
        raise ValueError(f"arm {arm!r}: {path} holds episodes of another System-1 spec (id {old[0]}, now {want}); "
                         "move the file away or rename the arm -- resuming would mix two backends in one arm")


def _placeholder(spec: Any) -> bool:
    return isinstance(spec, dict) and bool(spec.get("placeholder"))


def _strip_placeholder(spec: Any) -> Any:
    """The backend spec without the runner's own keys (``placeholder``; ``cache``, a per-backend System-1 cache file
    that overrides the experiment's ``s1_cache``, e.g. PREREG_A7_STAGE2 A7.18 (e)'s separate StartLux cache)."""
    return {k: v for k, v in spec.items() if k not in ("placeholder", "cache")} if isinstance(spec, dict) else spec


def _backend_cache(spec: Any, exp: dict[str, Any]) -> str | None:
    return spec.get("cache", exp.get("s1_cache")) if isinstance(spec, dict) else exp.get("s1_cache")


def _retry_errors_default(exp: dict[str, Any]) -> bool:
    if "retry_errors" in exp:
        return bool(exp["retry_errors"])
    return os.environ.get("JEV_RETRY_ERRORS", "1").strip().lower() not in ("0", "false", "no", "off")


def run_experiment(exp: dict[str, Any], out_dir: str | Path, llm: LLMClient | None = None,
                   s1: DecisionBackend | None = None, small_llm: LLMClient | None = None,
                   max_tasks: int | None = None, progress: bool = True,
                   retry_errors: bool | None = None) -> dict[str, Path]:
    """Run every arm on every task, resuming from existing ``<arm>.jsonl`` files (see the module docstring).
    ``retry_errors`` (default: ``exp["retry_errors"]``, else ``$JEV_RETRY_ERRORS``, else True) re-runs tasks
    whose latest record ended in an infrastructure error."""
    retry = _retry_errors_default(exp) if retry_errors is None else bool(retry_errors)
    out = Path(out_dir) / exp.get("name", "exp")
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    llm = llm or make_llm(exp["llm"])
    small_llm = small_llm or (make_llm(exp["small_llm"]) if exp.get("small_llm") else None)
    arms = expand_arms(exp)
    binds = arm_bindings(exp)
    need_s1 = any(a.uses_s1() and "s1" not in binds.get(a.name, {}) for a in arms)
    if s1 is None and need_s1:
        s1 = make_backend(exp["s1"], cache=exp.get("s1_cache"))
    # ``placeholder: true`` on an s1_backends entry (the matched-rate controls, whose rate comes from the System-1
    # arms of the same run): its arms are skipped until the rate is set, e.g.
    #   -o 's1_backends.ratem_lite={type: rate, rate: 0.083, seed: 0}'
    placeholders = sorted(n for n, spec in (exp.get("s1_backends") or {}).items() if _placeholder(spec))
    allow_ph = os.environ.get("JEV_ALLOW_PLACEHOLDER", "0").strip().lower() in ("1", "true", "yes")
    s1_pool = {name: make_backend(_strip_placeholder(spec), cache=_backend_cache(spec, exp))
               for name, spec in (exp.get("s1_backends") or {}).items()
               if any(b.get("s1") == name for b in binds.values())}
    need_measure = measure_llm_names(arms)
    unknown = sorted(need_measure - set(exp.get("llms") or {}))
    if unknown:
        raise ValueError(f"measure blocks name LLM(s) {unknown} that are not keys of `llms:`")
    llm_pool = {name: make_llm(spec) for name, spec in (exp.get("llms") or {}).items()
                if any(b.get("llm") == name for b in binds.values()) or name in need_measure}
    n_tasks = arm_task_counts(exp)
    suite_kw = dict(exp["suite"])
    suite_name = suite_kw.pop("name")
    check_suite_seeds(suite_name, int(suite_kw.get("seed0", 0) or 0), int(suite_kw.get("n", 50) or 0))  # A6.13 guard
    factories = load_suite(suite_name, **suite_kw)
    if max_tasks:
        factories = factories[:max_tasks]
    workers = int(exp.get("workers", os.environ.get("JEV_WORKERS", 4)))
    paths: dict[str, Path] = {}
    ph_skip = [] if allow_ph else [a.name for a in arms if binds.get(a.name, {}).get("s1") in placeholders]
    if ph_skip:
        print(f"skipping {len(ph_skip)} arm(s) bound to PLACEHOLDER System-1 backend(s) {placeholders}: {ph_skip}. "
              "Set their rates from this run's System-1 arms (decision.controls.trigger_rates('<run>/<arm>.jsonl')) "
              "and re-run with -o 's1_backends.<name>={type: rate, rate: <r>, seed: 0}'", file=sys.stderr)
    # PREREGISTRATION A6.16: an arm whose threshold is still a placeholder (`calibration: pending` on a decision
    # point, e.g. the Intern-Decision C1 arms before their calibration fit) never runs silently at that value; only
    # JEV_ALLOW_PENDING_TAU lifts this (JEV_ALLOW_PLACEHOLDER, which concerns matched-rate controls, does not)
    pending = [a.name for a in arms if any(a.dp(d).extra.get("calibration") == "pending" for d in DP)]
    pend_skip = [] if os.environ.get("JEV_ALLOW_PENDING_TAU", "") in ("1", "true", "yes") else pending
    if pend_skip:
        print(f"skipping {len(pend_skip)} arm(s) whose tau is still a calibration placeholder (A6.16): {pend_skip}. "
              "Fit tau-hat* from the registered calibration arm, write it into the config and drop the "
              "`calibration: pending` key (JEV_ALLOW_PENDING_TAU=1 runs them anyway, for smoke tests only)", file=sys.stderr)
    # JevAgentHarness profiles: a pending profile (not selected yet) or a section / part marked `skip` (e.g. the
    # A7.1 (g) degenerate cases) never runs -- in particular never as the LLM-only harness by accident
    profiles = arm_profiles(exp)
    prof_skip = {n: m["skip"] for n, m in profiles.items() if m.get("skip")}
    for n, why in sorted(prof_skip.items()):
        print(f"skipping arm {n!r}: {why}", file=sys.stderr)
    if profiles:
        (out / "profiles.json").write_text(json.dumps(profiles, indent=1, default=str))
    skipped = sorted(set(ph_skip) | set(pend_skip) | set(prof_skip))
    for arm in arms:   # refuse before running anything: an arm whose System-1 spec changed since its episodes
        if arm.name in skipped:
            continue
        b = binds.get(arm.name, {})
        _check_resume_spec(arm.name, out / f"{arm.name}.jsonl",
                           (s1_pool[b["s1"]] if "s1" in b else s1) if arm.uses_s1() else None)
    def _agent(arm: HarnessConfig) -> JevAgent:
        b = binds.get(arm.name, {})
        arm_s1 = s1_pool[b["s1"]] if "s1" in b else s1
        arm_llm = llm_pool[b["llm"]] if "llm" in b else llm
        return JevAgent(llm=arm_llm, cfg=arm, s1=arm_s1 if arm.uses_s1() else None,
                        small_llm=small_llm, system_prompt=exp.get("system_prompt"),
                        measure_llms={k: llm_pool[k] for k in measure_llm_names([arm]) if k in llm_pool})

    def _dump(arm_name: str, traj: Any, path: Path) -> None:
        if arm_name in profiles:
            traj.meta["profile"] = {k: profiles[arm_name].get(k) for k in
                                    ("path", "sha256", "name", "status", "s2", "part", "backend", "env")}
        traj.dump(path)

    sched = dict(exp.get("schedule") or {})
    if sched.get("mode", "arm_major") not in ("arm_major", "task_major"):
        raise ValueError(f"schedule.mode must be arm_major or task_major, got {sched.get('mode')!r}")
    if sched.get("mode") == "task_major":
        live = [a for a in arms if a.name not in skipped]
        for a in live:
            paths[a.name] = out / f"{a.name}.jsonl"
        _run_task_major(live, {a.name: _agent(a) for a in live}, factories, n_tasks, out, retry, workers, sched,
                        _dump, progress)
        return paths
    for arm in arms:
        if arm.name in skipped:
            continue
        path = out / f"{arm.name}.jsonl"
        paths[arm.name] = path
        done = _done_keys(path, retry_errors=retry)
        agent = _agent(arm)
        todo = []
        for fac in factories[: n_tasks.get(arm.name, len(factories))]:
            env = fac()
            if env.task_id not in done:
                todo.append(env)
        if not todo:
            continue
        if progress and retry:
            errored = _done_keys(path, retry_errors=False) - done
            n_retry = sum(1 for e in todo if e.task_id in errored)
            if n_retry:
                print(f"[{arm.name}] re-running {n_retry} episode(s) whose last record ended in an error",
                      file=sys.stderr)

        def _run(env):  # noqa: ANN001
            return agent.run(env)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run, e) for e in todo]
            it = as_completed(futs)
            if progress:
                it = tqdm(it, total=len(futs), desc=arm.name)
            for fut in it:
                _dump(arm.name, fut.result(), path)
    return paths


def task_major_order(task_ids: list[str], arms_per_task: dict[str, list[str]], seed: int) -> list[tuple[str, str]]:
    """(task, arm) jobs task by task in suite order; within a task the arms in an order drawn from a generator seeded
    by (``seed``, task id) -- the same order on every invocation, so a resume keeps the registered schedule."""
    jobs: list[tuple[str, str]] = []
    for t in task_ids:
        arms = sorted(arms_per_task.get(t) or [])
        random.Random(f"{seed}:{t}").shuffle(arms)
        jobs.extend((t, a) for a in arms)
    return jobs


def _run_task_major(arms: list[HarnessConfig], agents: dict[str, JevAgent], factories: list[Any],
                    n_tasks: dict[str, int], out: Path, retry: bool, workers: int, sched: dict[str, Any],
                    dump: Any, progress: bool) -> None:
    """PREREG_A7_STAGE2 A7.16 (a): every arm's episode of task i is submitted before any of task i + 1, arms in a
    seeded random order within the task, through ONE pool of fixed concurrency; the schedule (seed, concurrency, arms,
    the job order) is recorded in ``schedule.json`` / ``schedule_order.jsonl`` so latency comparisons see the same
    load, provider state and time of day."""
    seed = int(sched.get("seed", 732))
    done = {a.name: _done_keys(out / f"{a.name}.jsonl", retry_errors=retry) for a in arms}
    envs: dict[str, Any] = {}
    per_task: dict[str, list[str]] = {}
    order_ids: list[str] = []
    for i, fac in enumerate(factories):
        want = [a.name for a in arms if i < n_tasks.get(a.name, len(factories))]
        if not want:
            continue
        env = fac()
        order_ids.append(env.task_id)
        envs[env.task_id] = fac
        per_task[env.task_id] = [a for a in want if env.task_id not in done[a]]
    jobs = task_major_order(order_ids, per_task, seed)
    rec_path = out / "schedule.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"mode": "task_major", "seed": seed,
                                                                       "registered": "PREREG_A7_STAGE2 A7.16 (a)",
                                                                       "arms": [], "invocations": []}
    if rec.get("seed") != seed or rec.get("mode") != "task_major":
        raise ValueError(f"{rec_path}: recorded schedule (mode {rec.get('mode')}, seed {rec.get('seed')}) differs "
                         f"from the config's (task_major, seed {seed}); a schedule never changes mid-run")
    inv = {"k": len(rec["invocations"]), "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "arms": [a.name for a in arms], "concurrency": workers, "n_jobs": len(jobs), "retry_errors": retry}
    rec["arms"] = sorted(set(rec["arms"]) | set(inv["arms"]))
    rec["concurrency"] = workers
    rec["invocations"].append(inv)
    rec_path.write_text(json.dumps(rec, indent=1))
    with open(out / "schedule_order.jsonl", "a") as f:
        for j, (t, a) in enumerate(jobs):
            f.write(json.dumps({"inv": inv["k"], "seq": j, "task": t, "arm": a}) + "\n")
    if not jobs:
        return
    lock = threading.Lock()

    def _run(task: str, arm_name: str) -> tuple[str, Any]:
        return arm_name, agents[arm_name].run(envs[task]())

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run, t, a) for t, a in jobs]          # FIFO: started in the recorded order
        it = as_completed(futs)
        if progress:
            it = tqdm(it, total=len(futs), desc="task-major")
        for fut in it:
            arm_name, traj = fut.result()
            with lock:
                dump(arm_name, traj, out / f"{arm_name}.jsonl")
    inv["finished"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rec_path.write_text(json.dumps(rec, indent=1))


def main(argv: list[str] | None = None) -> None:
    """``python -m jevharness.bench.runner CONFIG [--out runs] [--max-tasks N] [--no-retry-errors]``"""
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser("jevharness.bench.runner")
    ap.add_argument("config")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--no-retry-errors", action="store_true",
                    help="on resume, treat episodes that ended in an infrastructure error as done")
    a = ap.parse_args(argv)
    exp = yaml.safe_load(Path(a.config).read_text())
    paths = run_experiment(exp, a.out, max_tasks=a.max_tasks, retry_errors=False if a.no_retry_errors else None)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
