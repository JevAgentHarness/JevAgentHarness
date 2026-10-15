#!/usr/bin/env python3
r"""Fixed-behaviour economics and timing audit of the Stage-2 arms -- DESCRIPTIVE / POST HOC (docs/REVIEW_ROUND12.md
§3 #2, P1-2). Stage 1 and Stage 2 registered analyses are closed; nothing here changes or re-tests them (S2-1T, the
registered latency row, is untouched). Every number is computed from the run files; nothing is typed by hand.

    python scripts/stage2_economics.py
    # reads runs/stage2_jevgym, runs/stage2_x, runs/stage2_v2 (the logged episodes and their call ledgers),
    # configs/prices/models.yaml, results/stage2/tariff*.json and results/stage2/stage2.json (the registered numbers);
    # writes results/stage2/economics.{json,md} and paper/generated/economics.tex (\defres{econ/...})

What it does, per arm (Flash System-2: ``llm_only``, ``best_single``, ``jevagent_best``, ``jevagent_best_rl``,
``jevagent_best_sl4b``, ``jevagent_default``, ``jevagent_minus_tool`` (200 tasks), exploratory ``jevagent_best_v2``;
DeepSeek-V4: ``llm_only_x``, ``jevagent_best_x``, ``llm_only_xsmall``), on the latest record of every expected task
(the registered drop rule: a task whose latest record is an infrastructure error in either arm leaves the pair):

(a) **Reproduction at the frozen prices.** Every System-2 call is re-priced from its logged tokens (fresh input,
    cached input, output) at the price table's row for its own model and must equal the logged ``cost_usd``; the
    episode cost (System-1 + System-2 + any other component as logged, plus the arm's frozen per-episode GPU tariff q
    where the registered analysis charges one) must equal the logged ``usage.total.cost_usd`` (+ q), and the arm's
    mean must equal the registered per-episode cost in ``results/stage2/stage2.json`` (S2-1 / S2-2 / S2-4 / S2-4S /
    S2-1X cost blocks, the score card, the descriptive rows) to ``TOL_USD`` -- and the registered savings and their
    bootstrap interval are reproduced to ``TOL_REL``. The input files' sha256 must equal the registered provenance.
    Any mismatch raises ``ReproductionError`` (exit 1): nothing is written.
(b) **Re-pricing System-2 only.** The same logged System-2 tokens priced at the listed ``gpt-6.1-sol`` and
    ``gpt-6-astra`` rows of ``configs/prices/models.yaml`` (fresh input at ``input_per_m``, cached input at
    ``cached_input_per_m``, output at ``output_per_m``); System-1 cost and GPU tariffs as recorded; call counts, token
    counts and cache hits as logged ("fixed behaviour"). Another LLM's own behaviour -- its reasoning tokens (the logged
    runs have none), its caching, its number of calls, its success -- is NOT modelled: this is a price sensitivity,
    not an estimate of running another LLM. Relative savings 1 - C_arm / C_ref and absolute savings C_ref - C_arm per
    episode against the arm's LLM-only reference re-priced the same way, with a paired task bootstrap. The cheaper-LLM
    row (``llm_only_xsmall``) is not re-priced: its row is defined by its own lower price.
(c) **Timing.** Per-episode wall-clock (``wall_ms``: the agent loop's elapsed time, measured while the run executed
    ``concurrency`` episodes in parallel; it includes tool execution, harness overhead and any retry back-off) and the
    registered model-call latency (``usage.total.latency_ms``: the sum of the logged call latencies, as S2-1T): mean and
    median per arm, and the differences arm - reference of the means and of the medians with a paired task bootstrap
    (B = ``N_BOOT``, seed ``SEED``; the same resampled task indices for every quantity of one pair). The registered
    mean latencies (score card, S2-1T) are reproduced. Arms run in separate invocations (other time windows,
    concurrency 4 vs 8) are flagged: their timing differences mix the arm with the serving conditions.

Outputs: ``results/stage2/economics.json`` (everything + provenance), ``economics.md`` and the paper's
``paper/generated/economics.tex`` (``\defres{econ/<arm>/...}``; schema in ``_tex_rows``). Label: descriptive
sensitivity at hypothetical prices (post hoc, round 12); never a registered result."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.accounting import Price, PriceTable  # noqa: E402

N_BOOT, SEED = 10000, 0
TOL_USD = 1e-9          # $ per episode: the registered means are reproduced far below a cent-fraction
TOL_REL = 1e-9          # savings and their bootstrap bounds (fractions)
TOL_S = 1e-6            # registered mean latencies (s)
PRICES = ROOT / "configs/prices/models.yaml"
STAGE2_JSON = ROOT / "results/stage2/stage2.json"
TASKS = ROOT / "configs/eval/jevgym_stage2_tasks.json"
TEX_OUT = ROOT / "paper/generated/economics.tex"
OUT_DIR = ROOT / "results/stage2"
LABEL = "descriptive sensitivity, post hoc (review round 12); not a registered analysis"
SCENARIOS = {"frozen": None, "sol": "gpt-6.1-sol", "astra": "gpt-6-astra"}   # None = each call's own logged model
RUN_DIRS = {"stage2": "stage2_jevgym", "x": "stage2_x", "v2": "stage2_v2"}
TARIFFS = {"rl": "results/stage2/tariff.json", "sl4b": "results/stage2/tariff_sl4b.json",
           "v2": "results/stage2/tariff_v2.json"}
# schedule record of the invocation that ran each arm (A7.16 (a)): run key -> {arm: schedule file}
SCHEDULES = {"stage2": {"jevagent_best_rl": "schedule_rl.json", "jevagent_best_sl4b": "schedule_sl4b.json"},
             "x": {}, "v2": {}}


def _p(*path: str) -> tuple[str, ...]:
    return tuple(path)


# key, run, arm file, family, reference, tariff, label, re-price?, registered checks.
# Registered checks: ("cost", json path, with_tariff) = the arm's mean $ / episode; ("savings", json path) = a cost block
# {cost_savings, cost_savings_ci} of this arm vs its reference; ("latency", json path) = mean model-call latency (s).
ARMS: tuple[dict[str, Any], ...] = (
    {"key": "llm_only", "run": "stage2", "family": "flash", "ref": None, "tariff": None, "label": "LLM-only (Flash)",
     "status": "reference", "checks": [("cost", _p("family", "S2-1", "cost", "cost_ref"), False),
                                       ("cost", _p("descriptive", "scorecard", "llm_only", "cost_usd"), False),
                                       ("latency", _p("descriptive", "scorecard", "llm_only", "latency_s")),
                                       ("latency", _p("secondary", "S2-1T", "latency_ref_s"))]},
    {"key": "best_single", "run": "stage2", "family": "flash", "ref": "llm_only", "tariff": None,
     "label": "cascade alone (Jev)", "status": "descriptive",
     "checks": [("cost", _p("family", "S2-2", "cost", "cost_ref"), False),
                ("cost", _p("descriptive", "scorecard", "best_single", "cost_usd"), False),
                ("savings", _p("descriptive", "scorecard", "best_single", "vs_baseline")),
                ("latency", _p("descriptive", "scorecard", "best_single", "latency_s"))]},
    {"key": "jevagent_best", "run": "stage2", "family": "flash", "ref": "llm_only", "tariff": None,
     "label": "profile, Jev", "status": "S2-1",
     "checks": [("cost", _p("family", "S2-1", "cost", "cost_arm"), False),
                ("savings", _p("family", "S2-1", "cost")),
                ("latency", _p("secondary", "S2-1T", "latency_arm_s"))]},
    {"key": "jevagent_best_rl", "run": "stage2", "family": "flash", "ref": "llm_only", "tariff": "rl",
     "label": "profile, JevAgentRL-4B", "status": "S2-4",
     "checks": [("cost", _p("family", "S2-4", "cost", "cost_arm"), True),
                ("cost", _p("descriptive", "scorecard", "jevagent_best_rl", "cost_usd"), False),
                ("savings", _p("family", "S2-4", "cost")),
                ("latency", _p("descriptive", "scorecard", "jevagent_best_rl", "latency_s"))]},
    {"key": "jevagent_best_sl4b", "run": "stage2", "family": "flash", "ref": "llm_only", "tariff": "sl4b",
     "label": "profile, StartLux-4B", "status": "S2-4S",
     "checks": [("cost", _p("secondary", "S2-4S", "cost", "cost_arm"), True),
                ("cost", _p("descriptive", "scorecard", "jevagent_best_sl4b", "cost_usd"), False),
                ("savings", _p("secondary", "S2-4S", "cost")),
                ("latency", _p("descriptive", "scorecard", "jevagent_best_sl4b", "latency_s"))]},
    {"key": "jevagent_default", "run": "stage2", "family": "flash", "ref": "llm_only", "tariff": None,
     "label": "default profile, Jev", "status": "descriptive",
     "checks": [("cost", _p("descriptive", "scorecard", "jevagent_default", "cost_usd"), False),
                ("savings", _p("descriptive", "jevagent_default", "cost")),
                ("latency", _p("descriptive", "scorecard", "jevagent_default", "latency_s"))]},
    {"key": "jevagent_minus_tool", "run": "stage2", "family": "flash", "ref": "llm_only", "tariff": None,
     "label": "profile without the tool cascade (200 tasks)", "status": "descriptive",
     "checks": [("cost", _p("descriptive", "logo", "tool", "cost", "cost_arm"), False)]},
    {"key": "jevagent_best_v2", "run": "v2", "family": "flash", "ref": "llm_only", "tariff": "v2",
     "label": "profile, JevAgentRL-v2 (exploratory, A7.19)", "status": "exploratory",
     "checks": [("cost", _p("descriptive", "stage2_v2", "profile", "cost", "cost_arm"), True),
                ("savings", _p("descriptive", "stage2_v2", "profile", "cost"))]},
    {"key": "llm_only_x", "run": "x", "family": "x", "ref": None, "tariff": None, "label": "LLM-only (V4-Pro)",
     "status": "reference", "checks": [("cost", _p("secondary", "S2-1X", "cost", "cost_ref"), False),
                                       ("cost", _p("descriptive", "scorecard", "llm_only_x", "cost_usd"), False),
                                       ("latency", _p("descriptive", "scorecard", "llm_only_x", "latency_s"))]},
    {"key": "jevagent_best_x", "run": "x", "family": "x", "ref": "llm_only_x", "tariff": None,
     "label": "profile, V4-Pro", "status": "S2-1X",
     "checks": [("cost", _p("secondary", "S2-1X", "cost", "cost_arm"), False),
                ("savings", _p("secondary", "S2-1X", "cost")),
                ("latency", _p("descriptive", "scorecard", "jevagent_best_x", "latency_s"))]},
    {"key": "llm_only_xsmall", "run": "x", "family": "x", "ref": "llm_only_x", "tariff": None,
     "label": "V4-Flash alone", "status": "descriptive", "reprice": False,
     "why_not_repriced": "a cheaper-LLM reference: its row is defined by its own lower price, which re-pricing at "
                         "another model's price would remove",
     "checks": [("cost", _p("descriptive", "scorecard", "llm_only_xsmall", "cost_usd"), False),
                ("savings", _p("descriptive", "llm_only_xsmall", "cost")),
                ("latency", _p("descriptive", "scorecard", "llm_only_xsmall", "latency_s"))]},
)
ARM_BY_KEY = {a["key"]: a for a in ARMS}
NAN = float("nan")


class ReproductionError(RuntimeError):
    """The frozen-price recomputation does not reproduce a logged or registered number."""


# ====================================================================================== loading
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_latest(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """task id -> the latest record of ``<arm>.jsonl`` in file order (the runner appends a fresh record when it retries
    an errored task; scripts/primaries.load_arm keeps the same one), and the number of records."""
    out: dict[str, dict[str, Any]] = {}
    n = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a partial last line of an interrupted writer (load_arm skips it too)
            if "task_id" not in r:
                continue
            n += 1
            out[str(r["task_id"])] = r
    return out, n


def _get(d: Mapping[str, Any], path: Sequence[str]) -> Any:
    x: Any = d
    for k in path:
        if not isinstance(x, Mapping) or k not in x:
            return None
        x = x[k]
    return x


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ====================================================================================== per-episode cost
def call_tokens(c: Mapping[str, Any]) -> tuple[int, int, int]:
    """(input, output, cached input) tokens of one ledger call record."""
    cached = int(((c.get("meta") or {}).get("cached_input_tokens") or 0))
    return int(c.get("input_tokens") or 0), int(c.get("output_tokens") or 0), cached


def episode_cost(rec: Mapping[str, Any], table: PriceTable, scenario: Price | None, q: float = 0.0,
                 where: str = "") -> dict[str, float]:
    """One episode's cost split: System-1 and other components as logged, System-2 recomputed from its tokens at
    ``scenario`` (None: at the frozen table's row for each call's own model -- then every recomputed call must equal
    its logged cost), plus the per-episode tariff ``q``."""
    s1 = s2 = other = s2_logged = 0.0
    reasoning = 0
    for c in rec.get("calls") or []:
        comp = c.get("component")
        logged = float(c.get("cost_usd") or 0.0)
        if comp == "s1":
            s1 += logged
        elif comp == "s2":
            i, o, ca = call_tokens(c)
            reasoning += int((c.get("meta") or {}).get("reasoning_tokens") or 0)
            s2_logged += logged
            if scenario is None:
                pr = table.get(c.get("model"))
                if pr is None:
                    raise ReproductionError(f"{where}: System-2 model {c.get('model')!r} has no row in the price table")
                v = pr.cost(i, o, ca)
                if abs(v - logged) > 1e-12:
                    raise ReproductionError(f"{where}: a System-2 call of {c.get('model')} re-priced at the frozen "
                                            f"table costs {v!r}, logged {logged!r}")
            else:
                v = scenario.cost(i, o, ca)
            s2 += v
        else:
            other += logged
    tot = (rec.get("usage") or {}).get("total") or {}
    if scenario is None and abs(s1 + s2_logged + other - float(tot.get("cost_usd") or 0.0)) > 1e-12:
        raise ReproductionError(f"{where}: the call ledger sums to {s1 + s2_logged + other!r}, usage.total.cost_usd is "
                                f"{tot.get('cost_usd')!r}")
    return {"s1": s1, "s2": s2, "other": other, "tariff": q, "total": s1 + s2 + other + q, "reasoning_tokens": reasoning}


# ====================================================================================== bootstrap
def boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """The paired task bootstrap's resampled indices -- the same draw as scripts/primaries.cost_block, so the frozen
    savings interval reproduces the registered one."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _q(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [NAN, NAN]


def savings_block(ca: np.ndarray, cr: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    """Relative savings 1 - mean(arm) / mean(ref) and absolute savings mean(ref) - mean(arm) per episode, with
    percentile intervals over the paired resamples ``idx``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_b = 1.0 - ca[idx].mean(1) / cr[idx].mean(1)
    abs_b = cr[idx].mean(1) - ca[idx].mean(1)
    rel = 1.0 - ca.mean() / cr.mean() if cr.mean() > 0 else NAN
    return {"cost_arm": float(ca.mean()), "cost_ref": float(cr.mean()), "savings": float(rel), "savings_ci": _q(rel_b),
            "abs_savings": float(cr.mean() - ca.mean()), "abs_savings_ci": _q(abs_b)}


def timing_block(ta: np.ndarray, tr: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    """Mean and median per arm and the differences arm - reference (of the means, of the medians) with intervals
    over the paired resamples ``idx``."""
    ma, mr = ta[idx].mean(1), tr[idx].mean(1)
    da, dr = np.median(ta[idx], axis=1), np.median(tr[idx], axis=1)
    return {"mean_arm": float(ta.mean()), "mean_ref": float(tr.mean()),
            "median_arm": float(np.median(ta)), "median_ref": float(np.median(tr)),
            "mean_diff": float(ta.mean() - tr.mean()), "mean_diff_ci": _q(ma - mr),
            "median_diff": float(np.median(ta) - np.median(tr)), "median_diff_ci": _q(da - dr),
            "mean_ratio": float(ta.mean() / tr.mean()) if tr.mean() > 0 else NAN}


# ====================================================================================== the analysis
class Economics:
    def __init__(self, runs: Path, stage2_json: Path, prices: Path, tasks: Path, tariffs: Mapping[str, Path],
                 n_boot: int = N_BOOT, seed: int = SEED, arms: Sequence[Mapping[str, Any]] = ARMS,
                 run_dirs: Mapping[str, str] = RUN_DIRS, check_provenance: bool = True) -> None:
        self.runs, self.n_boot, self.seed = runs, n_boot, seed
        self.arms = list(arms)
        self.run_dirs = {k: runs / v for k, v in run_dirs.items()}
        self.registered = json.loads(stage2_json.read_text())
        self.stage2_json = stage2_json
        self.prices_path = prices
        self.table = PriceTable.load([prices])
        self.tasks = json.loads(tasks.read_text())["task_ids"]
        self.tasks_path = tasks
        self.tariff_paths = dict(tariffs)
        self.tariffs = {k: json.loads(Path(p).read_text()) for k, p in tariffs.items() if Path(p).exists()}
        self.check_provenance = check_provenance
        self.records: dict[str, dict[str, dict[str, Any]]] = {}
        self.provenance: dict[str, Any] = {"inputs": {}, "registered": {"path": _rel(stage2_json),
                                                                         "sha256": _sha256(stage2_json)},
                                           "prices": {"path": _rel(prices), "sha256": _sha256(prices)},
                                           "tasks": {"path": _rel(tasks), "sha256": _sha256(tasks)},
                                           "tariffs": {k: {"path": _rel(p), "sha256": _sha256(Path(p))}
                                                       for k, p in tariffs.items() if Path(p).exists()}}
        self.checks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ inputs
    def _arm_path(self, a: Mapping[str, Any]) -> Path:
        return self.run_dirs[a["run"]] / f"{a['key']}.jsonl"

    def load(self, a: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        if a["key"] not in self.records:
            path = self._arm_path(a)
            if not path.exists():
                raise ReproductionError(f"{a['key']}: {_rel(path)} missing")
            recs, n = load_latest(path)
            sha = _sha256(path)
            self.provenance["inputs"][a["key"]] = {"path": _rel(path), "sha256": sha, "n_records": n,
                                                   "n_tasks": len(recs)}
            if self.check_provenance:
                reg = self._registered_sha(a["key"], _rel(path))
                if reg is not None and reg != sha:
                    raise ReproductionError(f"{a['key']}: {_rel(path)} sha256 {sha[:12]} differs from the registered "
                                            f"analysis's input {reg[:12]} ({_rel(self.stage2_json)} provenance)")
                self.checks.append({"arm": a["key"], "what": "input sha256 = registered provenance",
                                    "registered": reg, "value": sha, "ok": reg is None or reg == sha,
                                    "note": None if reg is not None else "not listed in the registered provenance"})
            self.records[a["key"]] = recs
        return self.records[a["key"]]

    def _registered_sha(self, arm: str, rel_path: str) -> str | None:
        def walk(x: Any) -> str | None:
            if isinstance(x, Mapping):
                if x.get("path") == rel_path and x.get("sha256"):
                    return str(x["sha256"])
                for v in x.values():
                    s = walk(v)
                    if s:
                        return s
            return None
        return walk(self.registered.get("provenance") or {})

    def spec(self, key: str) -> Mapping[str, Any]:
        return next((a for a in self.arms if a["key"] == key), None) or ARM_BY_KEY[key]

    def expected(self, a: Mapping[str, Any]) -> list[str]:
        """The arm's expected tasks: the first n of the frozen Stage-2 list, n = the arm's ``n_tasks`` in the run's
        experiment.yaml, else the run's ``n_tasks`` (the registered sizes; ``jevagent_minus_tool`` ran 200)."""
        exp_path = self.run_dirs[a["run"]] / "experiment.yaml"
        n = len(self.tasks)
        if exp_path.exists():
            import yaml
            exp = yaml.safe_load(exp_path.read_text()) or {}
            arm = next((x for x in exp.get("arms") or [] if x.get("name") == a["key"]), {})
            n = int(arm.get("n_tasks") or exp.get("n_tasks") or n)
        return list(self.tasks[:n])

    def q(self, a: Mapping[str, Any]) -> float:
        if not a.get("tariff"):
            return 0.0
        t = self.tariffs.get(a["tariff"])
        if t is None or not _finite(t.get("q_usd_per_episode")):
            raise ReproductionError(f"{a['key']}: frozen tariff {a['tariff']} missing")
        return float(t["q_usd_per_episode"])

    def schedule(self, a: Mapping[str, Any]) -> dict[str, Any]:
        """Concurrency and time window of the invocation that ran the arm (the run's schedule record)."""
        d = self.run_dirs[a["run"]]
        name = SCHEDULES.get(a["run"], {}).get(a["key"], "schedule.json")
        p = d / name
        if not p.exists():
            return {}
        s = json.loads(p.read_text())
        inv = next((i for i in s.get("invocations") or [] if i.get("n_jobs")), {})
        return {"file": _rel(p), "concurrency": s.get("concurrency"), "started": inv.get("started"),
                "finished": inv.get("finished"), "arms_together": sorted(s.get("arms") or [])}

    # ------------------------------------------------------------------ per arm
    def costs(self, a: Mapping[str, Any], keys: Sequence[str]) -> dict[str, np.ndarray]:
        recs = self.load(a)
        q = self.q(a)
        out: dict[str, list[float]] = {f"{s}/{part}": [] for s in SCENARIOS for part in ("total", "s1", "s2")}
        out["reasoning_tokens"] = []
        for k in keys:
            r = recs[k]
            for s, model in SCENARIOS.items():
                scen = None if model is None else self.table.get(model)
                if model is not None and scen is None:
                    raise ReproductionError(f"scenario price {model} missing from {_rel(self.prices_path)}")
                c = episode_cost(r, self.table, scen, q, where=f"{a['key']} {k}")
                out[f"{s}/total"].append(c["total"])
                out[f"{s}/s1"].append(c["s1"])
                out[f"{s}/s2"].append(c["s2"])
                if s == "frozen":
                    out["reasoning_tokens"].append(c["reasoning_tokens"])
                    logged = float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0.0)
                    if abs(c["total"] - (logged + q)) > 1e-12:
                        raise ReproductionError(f"{a['key']} {k}: frozen episode cost {c['total']!r} != logged "
                                                f"{logged!r} + tariff {q!r}")
        return {k: np.asarray(v, float) for k, v in out.items()}

    def timing(self, a: Mapping[str, Any], keys: Sequence[str]) -> dict[str, np.ndarray]:
        recs = self.load(a)
        wall, lat, retry = [], [], []
        for k in keys:
            r = recs[k]
            if r.get("wall_ms") is None:
                raise ReproductionError(f"{a['key']} {k}: no wall_ms logged")
            wall.append(float(r["wall_ms"]) / 1000.0)
            lat.append(float(((r.get("usage") or {}).get("total") or {}).get("latency_ms") or 0.0) / 1000.0)
            retry.append(sum(float(c.get("retry_wait_ms") or 0.0) for c in r.get("calls") or []) / 1000.0)
        return {"wall": np.asarray(wall), "calllat": np.asarray(lat), "retry_wait": np.asarray(retry)}

    def keys_for(self, a: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        """Paired complete case over the arm's expected tasks (and its reference's): (kept, dropped)."""
        exp = self.expected(a)
        ra = self.load(a)
        rr = self.load(self.spec(a["ref"])) if a.get("ref") else None
        kept, dropped = [], []
        for k in exp:
            ok = k in ra and not ra[k].get("error") and (rr is None or (k in rr and not rr[k].get("error")))
            (kept if ok else dropped).append(k)
        return kept, dropped

    def check(self, arm: str, what: str, registered: Any, value: float, tol: float) -> None:
        ok = _finite(registered) and abs(float(registered) - value) <= tol
        self.checks.append({"arm": arm, "what": what, "registered": registered, "value": value, "tol": tol, "ok": ok})
        if not ok:
            raise ReproductionError(f"{arm}: {what}: recomputed {value!r}, registered {registered!r} (tolerance {tol})")

    def run(self) -> dict[str, Any]:
        res: dict[str, Any] = {"label": LABEL, "scenarios": {}, "arms": {}, "n_boot": self.n_boot, "seed": self.seed}
        for s, model in SCENARIOS.items():
            if model is None:
                res["scenarios"][s] = {"model": "each call's own model (frozen table)",
                                       "rows": {m: vars(self.table.get(m)) for m in sorted(self._s2_models())}}
            else:
                res["scenarios"][s] = {"model": model, "price": vars(self.table.get(model))}
        for a in self.arms:
            res["arms"][a["key"]] = self.arm(a)
        res["reasoning_tokens_total"] = int(sum(v.get("reasoning_tokens", 0) for v in res["arms"].values()))
        res["checks"] = self.checks
        res["n_checks"] = len(self.checks)
        res["provenance"] = self.provenance
        res["meta"] = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git_head()}
        return res

    def _s2_models(self) -> set[str]:
        ms: set[str] = set()
        for a in self.arms:
            for r in self.load(a).values():
                ms |= {str(c.get("model")) for c in r.get("calls") or [] if c.get("component") == "s2"}
        return ms

    def arm(self, a: Mapping[str, Any]) -> dict[str, Any]:
        keys, dropped = self.keys_for(a)
        ref = self.spec(a["ref"]) if a.get("ref") else None
        ca = self.costs(a, keys)
        ta = self.timing(a, keys)
        out: dict[str, Any] = {"label": a["label"], "status": a["status"], "family": a["family"], "reference": a.get("ref"),
                               "n": len(keys), "n_dropped": len(dropped), "dropped": dropped,
                               "tariff_q": self.q(a), "reprice": a.get("reprice", True),
                               "why_not_repriced": a.get("why_not_repriced"),
                               "reasoning_tokens": int(ca["reasoning_tokens"].sum()),
                               "s2_models": sorted({str(c.get("model")) for k in keys for c in
                                                    self.load(a)[k].get("calls") or [] if c.get("component") == "s2"}),
                               "schedule": self.schedule(a), "cost": {}, "timing": {}}
        for s in SCENARIOS:
            if s != "frozen" and not out["reprice"]:
                continue
            tot = ca[f"{s}/total"]
            out["cost"][s] = {"mean": float(tot.mean()), "s1": float(ca[f"{s}/s1"].mean()),
                              "s2": float(ca[f"{s}/s2"].mean()), "tariff": out["tariff_q"],
                              "s2_share": float(ca[f"{s}/s2"].mean() / tot.mean()) if tot.mean() > 0 else NAN}
        out["timing"]["arm"] = {"wall_mean": float(ta["wall"].mean()), "wall_median": float(np.median(ta["wall"])),
                                "calllat_mean": float(ta["calllat"].mean()),
                                "calllat_median": float(np.median(ta["calllat"])),
                                "retry_wait_mean": float(ta["retry_wait"].mean())}
        # (a) the registered numbers
        for chk in a.get("checks") or []:
            kind, path = chk[0], chk[1]
            reg = _get(self.registered, path)
            if reg is None:
                self.checks.append({"arm": a["key"], "what": f"{kind} {'/'.join(path)}", "registered": None,
                                    "value": None, "ok": True, "note": "not in the registered results (skipped)"})
                continue
            if kind == "cost":
                exp_keys = self.expected(a)
                full = ca["frozen/total"] if len(keys) == len(exp_keys) else None
                if full is None:
                    raise ReproductionError(f"{a['key']}: dropped tasks; the registered cost cannot be reproduced")
                val = float(full.mean()) - (0.0 if chk[2] else out["tariff_q"])
                self.check(a["key"], f"mean $ / episode = {'/'.join(path)}", reg, val, TOL_USD)
            elif kind == "latency":
                self.check(a["key"], f"mean model-call latency (s) = {'/'.join(path)}", reg,
                           float(ta["calllat"].mean()), TOL_S)
        if ref is None:
            return out
        # paired blocks against the reference (same resampled indices for every quantity)
        cr = self.costs(ref, keys)
        tr = self.timing(ref, keys)
        idx = boot_indices(len(keys), self.n_boot, self.seed)
        out["savings"] = {}
        for s in SCENARIOS:
            if s != "frozen" and not out["reprice"]:
                continue
            out["savings"][s] = savings_block(ca[f"{s}/total"], cr[f"{s}/total"], idx)
        fz = out["savings"]["frozen"]
        for s in ("sol", "astra"):
            if s in out["savings"] and fz["abs_savings"] != 0:
                out["savings"][s]["abs_ratio_to_frozen"] = out["savings"][s]["abs_savings"] / fz["abs_savings"]
                out["savings"][s]["rel_change_pt"] = out["savings"][s]["savings"] - fz["savings"]
        for chk in a.get("checks") or []:
            if chk[0] != "savings":
                continue
            reg = _get(self.registered, chk[1])
            if reg is None:
                continue
            self.check(a["key"], f"savings = {'/'.join(chk[1])}/cost_savings", reg.get("cost_savings"),
                       fz["savings"], TOL_REL)
            for i, b in enumerate(("lo", "hi")):
                self.check(a["key"], f"savings bootstrap {b} = {'/'.join(chk[1])}/cost_savings_ci",
                           (reg.get("cost_savings_ci") or [None, None])[i], fz["savings_ci"][i], TOL_REL)
        out["timing"]["wall"] = timing_block(ta["wall"], tr["wall"], idx)
        out["timing"]["calllat"] = timing_block(ta["calllat"], tr["calllat"], idx)
        sa, sr = out["schedule"], self.schedule(ref)
        out["timing"]["same_invocation"] = bool(sa and sr and sa.get("file") == sr.get("file"))
        out["timing"]["concurrency_arm"], out["timing"]["concurrency_ref"] = sa.get("concurrency"), sr.get("concurrency")
        return out


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ====================================================================================== reports
def _usd(v: Any) -> str:
    return "–" if not _finite(v) else np.format_float_positional(float(v), precision=3, unique=True, fractional=False,
                                                                trim="-")


def _pct(v: Any, sign: bool = False) -> str:
    return "–" if not _finite(v) else (f"{100 * float(v):+.1f}" if sign else f"{100 * float(v):.1f}")


def _ci(c: Any, f=_pct) -> str:  # noqa: ANN001
    return "" if not c or not all(_finite(x) for x in c) else f"[{f(c[0])}, {f(c[1])}]"


def _s(v: Any, sign: bool = False) -> str:
    return "–" if not _finite(v) else (f"{float(v):+.2f}" if sign else f"{float(v):.2f}")


def to_markdown(res: Mapping[str, Any]) -> str:
    sc = res["scenarios"]
    L = ["# Stage 2: fixed-behaviour economics and timing audit (descriptive, post hoc)", "",
         f"Label: {res['label']}. Paired task bootstrap B = {res['n_boot']}, seed {res['seed']}. "
         f"All {res['n_checks']} reproduction checks passed (frozen prices reproduce every logged System-2 call, every "
         "episode's logged cost, the registered per-episode costs, savings and their intervals, and the registered "
         "mean model-call latencies).", "",
         "## (b) Re-pricing System-2 tokens only", "",
         "Prices ($ per 1M tokens: input / cached input / output): "
         + "; ".join(f"{s} = `{v['model']}` {v['price']['input_per_m']} / {v['price']['cached_input_per_m']} / "
                     f"{v['price']['output_per_m']}" for s, v in sc.items() if s != "frozen")
         + "; frozen = each call's own model: "
         + "; ".join(f"`{m}` {p['input_per_m']} / {p['cached_input_per_m']} / {p['output_per_m']}"
                     for m, p in sc["frozen"]["rows"].items()) + ".",
         "System-1 cost and GPU tariffs as recorded; tokens, cache hits and calls as logged. Another LLM's reasoning "
         f"tokens, caching and behaviour are not modelled (logged reasoning tokens in these arms: "
         f"{res['reasoning_tokens_total']}).", "",
         "| arm | status | n | tariff q | $ / ep. frozen | Sol | Astra | S2 share (frozen) | savings frozen [95 %] | "
         "Sol | Astra | abs. saving $ / ep. frozen | Sol | Astra | × (Astra / frozen) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, a in res["arms"].items():
        c, sv = a["cost"], a.get("savings") or {}

        def cell(s: str, what: str, ci: bool = False, f=_pct) -> str:  # noqa: ANN001
            b = sv.get(s)
            if b is None:
                return "–"
            return f"{f(b[what])} {_ci(b.get(what + '_ci'), f) if ci else ''}".strip()
        L.append(f"| `{k}` | {a['status']} | {a['n']} | {_usd(a['tariff_q']) if a['tariff_q'] else '–'} | "
                 f"{_usd(c['frozen']['mean'])} | {_usd((c.get('sol') or {}).get('mean'))} | "
                 f"{_usd((c.get('astra') or {}).get('mean'))} | {_pct(c['frozen']['s2_share'])} % | "
                 f"{cell('frozen', 'savings', True)} | {cell('sol', 'savings', True)} | {cell('astra', 'savings', True)} | "
                 f"{cell('frozen', 'abs_savings', f=_usd)} | {cell('sol', 'abs_savings', f=_usd)} | "
                 f"{cell('astra', 'abs_savings', f=_usd)} | "
                 f"{_s((sv.get('astra') or {}).get('abs_ratio_to_frozen'))} |")
    L += ["", "Savings are vs the arm's LLM-only reference (`llm_only` for Flash, `llm_only_x` for DeepSeek-V4), "
              "re-priced the same way. " + " ".join(f"`{k}` not re-priced: {a['why_not_repriced']}."
                                                    for k, a in res["arms"].items() if not a["reprice"]), "",
          "## (c) Timing (s per episode)", "",
          "| arm | conc. | same invocation as ref. | wall-clock mean | median | Δ mean vs ref. [95 %] | Δ median [95 %] | "
          "model-call latency mean | median | Δ mean [95 %] | Δ median [95 %] | retry back-off mean |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, a in res["arms"].items():
        t = a["timing"]
        w, l = t.get("wall") or {}, t.get("calllat") or {}
        L.append(f"| `{k}` | {(a['schedule'] or {}).get('concurrency', '–')} | "
                 f"{'yes' if t.get('same_invocation') else ('–' if not a.get('reference') else 'no')} | "
                 f"{_s(t['arm']['wall_mean'])} | {_s(t['arm']['wall_median'])} | "
                 f"{_s(w.get('mean_diff'), True)} {_ci(w.get('mean_diff_ci'), lambda x: _s(x, True))} | "
                 f"{_s(w.get('median_diff'), True)} {_ci(w.get('median_diff_ci'), lambda x: _s(x, True))} | "
                 f"{_s(t['arm']['calllat_mean'])} | {_s(t['arm']['calllat_median'])} | "
                 f"{_s(l.get('mean_diff'), True)} {_ci(l.get('mean_diff_ci'), lambda x: _s(x, True))} | "
                 f"{_s(l.get('median_diff'), True)} {_ci(l.get('median_diff_ci'), lambda x: _s(x, True))} | "
                 f"{t['arm']['retry_wait_mean']:.3f} |")
    L += ["", "Wall-clock = the agent loop's elapsed time per episode (`wall_ms`), measured while `conc.` episodes ran in "
              "parallel; it includes tool execution, harness overhead and retry back-off. Model-call latency = the sum "
              "of the logged call latencies (`usage.total.latency_ms`, the registered S2-1T quantity; System-1 calls "
              "of one step may overlap, so it can exceed the wall-clock). Arms not run in the reference's invocation "
              "(`same invocation` = no) ran in another time window and at another concurrency: their differences mix "
              "the arm with the serving conditions.", "",
          "## (a) Reproduction checks", "", "| arm | check | registered | recomputed | ok |", "|---|---|---|---|---|"]
    for c in res["checks"]:
        reg, val = c.get("registered"), c.get("value")
        fmt = (lambda x: f"{x:.12g}" if _finite(x) else (str(x)[:12] if x else "–"))
        L.append(f"| `{c['arm']}` | {c['what']} | {fmt(reg)} | {fmt(val)} | {'yes' if c['ok'] else 'NO'}"
                 f"{' (' + c['note'] + ')' if c.get('note') else ''} |")
    L += ["", f"Generated {res['meta']['generated']} at git {str(res['meta']['git'])[:12]}; inputs: "
              + ", ".join(f"`{v['path']}` ({v['sha256'][:12]})" for v in res["provenance"]["inputs"].values()) + "."]
    return "\n".join(L) + "\n"


def _tex_rows(res: Mapping[str, Any]) -> dict[tuple[str, ...], tuple[str, Any]]:
    """(key path) -> (format, value). Schema (all descriptive, post hoc):

    * ``econ/<arm>/{n, tariff_q, concurrency}``; ``econ/<arm>/cost/<scn>`` ($ / episode incl. tariff; scn = frozen,
      sol, astra), ``econ/<arm>/lgcost/frozen`` (log10 of it, for figure coordinates), ``econ/<arm>/s2_share/<scn>`` (%);
    * vs the reference: ``econ/<arm>/savings/<scn>`` (%) and ``savings_ci/<scn>/{lo,hi}``, ``abs_savings/<scn>`` ($ /
      episode) and ``abs_savings_ci/<scn>/{lo,hi}``, ``abs_ratio/<scn>`` (×, vs frozen), ``rel_change/<scn>`` (points);
    * timing (s): ``econ/<arm>/{wall,calllat}/{mean,median}``; vs the reference ``{wall,calllat}/{mean_diff,
      median_diff}`` with ``_ci/{lo,hi}`` (signed), ``{wall,calllat}/ratio`` (×), ``{wall,calllat}/ref_{mean,median}``;
    * ``econ/price/<scn>/{input,cached,output}`` ($ per 1M tokens), ``econ/meta/{n_boot, seed, reasoning_tokens,
      n_checks}``."""
    rows: dict[tuple[str, ...], tuple[str, Any]] = {}

    def put(path: tuple[str, ...], kind: str, v: Any) -> None:
        if (kind == "text" and v) or _finite(v):
            rows[path] = (kind, v)

    def ci(path: tuple[str, ...], kind: str, c: Any) -> None:
        if c and len(c) == 2:
            put(path + ("lo",), kind, c[0])
            put(path + ("hi",), kind, c[1])

    for s, v in res["scenarios"].items():
        if s == "frozen":
            continue
        put(("econ", "price", s, "input"), "f2", v["price"]["input_per_m"])
        put(("econ", "price", s, "cached"), "f2", v["price"]["cached_input_per_m"])
        put(("econ", "price", s, "output"), "f2", v["price"]["output_per_m"])
        put(("econ", "price", s, "model"), "text", v["model"])
    for m, p in res["scenarios"]["frozen"]["rows"].items():
        for k, kk in (("input", "input_per_m"), ("cached", "cached_input_per_m"), ("output", "output_per_m")):
            put(("econ", "price", "frozen", m, k), "f4", p[kk])
    put(("econ", "meta", "n_boot"), "intc", res["n_boot"])
    put(("econ", "meta", "seed"), "int", res["seed"])
    put(("econ", "meta", "reasoning_tokens"), "int", res["reasoning_tokens_total"])
    put(("econ", "meta", "n_checks"), "int", res["n_checks"])
    for k, a in res["arms"].items():
        b = ("econ", k)
        put(b + ("n",), "int", a["n"])
        if a["tariff_q"]:
            put(b + ("tariff_q",), "usd", a["tariff_q"])
        put(b + ("concurrency",), "int", (a["schedule"] or {}).get("concurrency"))
        for s, c in a["cost"].items():
            put(b + ("cost", s), "usd", c["mean"])
            put(b + ("s2_share", s), "pct", c["s2_share"])
        put(b + ("lgcost", "frozen"), "f4", math.log10(a["cost"]["frozen"]["mean"]))
        for s, sv in (a.get("savings") or {}).items():
            put(b + ("savings", s), "pct", sv["savings"])
            ci(b + ("savings_ci", s), "pct", sv["savings_ci"])
            put(b + ("abs_savings", s), "usd", sv["abs_savings"])
            ci(b + ("abs_savings_ci", s), "usd", sv["abs_savings_ci"])
            put(b + ("abs_ratio", s), "f1", sv.get("abs_ratio_to_frozen"))
            put(b + ("rel_change", s), "spct", sv.get("rel_change_pt"))
        t = a["timing"]
        for q in ("wall", "calllat"):
            put(b + (q, "mean"), "f2", t["arm"][f"{q}_mean"])
            put(b + (q, "median"), "f2", t["arm"][f"{q}_median"])
            blk = t.get(q)
            if blk:
                put(b + (q, "mean_diff"), "f2s", blk["mean_diff"])
                ci(b + (q, "mean_diff_ci"), "f2s", blk["mean_diff_ci"])
                put(b + (q, "median_diff"), "f2s", blk["median_diff"])
                ci(b + (q, "median_diff_ci"), "f2s", blk["median_diff_ci"])
                put(b + (q, "ratio"), "f2", blk["mean_ratio"])
                put(b + (q, "ref_mean"), "f2", blk["mean_ref"])
                put(b + (q, "ref_median"), "f2", blk["median_ref"])
    return rows


def _fmt(kind: str, v: Any) -> str:
    if kind == "text":
        return str(v).replace("_", r"\_")
    v = float(v)
    if kind == "pct":
        return f"{100 * v:.1f}"
    if kind == "spct":
        return f"{100 * v:+.1f}"
    if kind == "int":
        return f"{v:.0f}"
    if kind == "intc":
        return f"{v:,.0f}".replace(",", "{,}")
    if kind == "f1":
        return f"{v:.1f}"
    if kind == "f2":
        return f"{v:.2f}"
    if kind == "f2s":
        return f"{v:+.2f}"
    if kind == "f4":
        return f"{v:.4f}"
    return np.format_float_positional(v, precision=3, unique=True, fractional=False, trim="-")   # usd


def to_tex(res: Mapping[str, Any]) -> str:
    lines = ["% AUTO-GENERATED by scripts/stage2_economics.py -- do not edit by hand.",
             f"% generated {res['meta']['generated']} at git {str(res['meta']['git'])[:12]}; {LABEL}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{'/'.join(p)}}}{{{_fmt(k, v)}}}" for p, (k, v) in sorted(_tex_rows(res).items())]
    lines += [r"\makeatother", ""]
    return "\n".join(lines)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--stage2-json", default=str(STAGE2_JSON))
    ap.add_argument("--prices", default=str(PRICES))
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--tex", default=str(TEX_OUT))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    eco = Economics(Path(a.runs), Path(a.stage2_json), Path(a.prices), Path(a.tasks),
                    {k: ROOT / v for k, v in TARIFFS.items()}, n_boot=a.n_boot, seed=a.seed)
    try:
        res = eco.run()
    except ReproductionError as e:
        print(f"[economics] REPRODUCTION FAILED: {e}", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "economics.json").write_text(json.dumps(_jsonable(res), indent=1))
    (out / "economics.md").write_text(to_markdown(res))
    if a.tex:
        Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
        Path(a.tex).write_text(to_tex(res))
    for k, v in res["arms"].items():
        sv = v.get("savings") or {}
        print(f"[economics] {k:20s} $ {v['cost']['frozen']['mean']:.6f}"
              + "".join(f"  {s} {100 * sv[s]['savings']:.1f} %" for s in sv)
              + f"  wall {v['timing']['arm']['wall_mean']:.2f} s  calls {v['timing']['arm']['calllat_mean']:.2f} s",
              file=sys.stderr)
    print(f"[economics] {res['n_checks']} reproduction checks passed -> {_rel(out)}, {_rel(a.tex)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
