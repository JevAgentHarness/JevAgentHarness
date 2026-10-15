#!/usr/bin/env python3
r"""E9 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E9: cost-accuracy frontier of
multiple-choice decisions"): do decision models sit on the cost-accuracy frontier of an agent's multiple-choice
decisions, against LLMs answering the same questions?

    python scripts/explore_e9_frontier.py ids               # the check's items are the registered dev prefix (no call)
    python scripts/explore_e9_frontier.py check --smoke     # the check's first 29 items, every backend, then the gate
    python scripts/explore_e9_frontier.py check             # the 299 items (refused unless the smoke's gate said go)
    python scripts/explore_e9_frontier.py archive           # gzip the check's records into results/ (done by `check`)
    python scripts/explore_e9_frontier.py summarize         # summary.{md,json} + paper/generated/e9_frontier.tex

Main readout (no call): E1's 3,991 L1 decisions (BFCL v4 tool choice and arguments, When2Call; the C8 population),
every answer already logged in ``runs/e1_bfcl_when2call*/``. Check (new calls): GPT-6-Luna and GPT-6.1-Sol as
multiple-choice deciders beside Jev and the Gemini deciders on E1's 299-item development prefix only
(``configs/explore/e9_frontier/e1_dev_check.yaml``); reported beside the figure, never drawn into it.

Cost per decision: API models the records' logged ``cost_usd`` (list price on measured tokens, reasoning tokens billed
as output; a batched request's cost split over its questions). Open-weights models: the paper's GPU tariff,
``GPU_USD_PER_H`` per GPU-hour (the Stage-2 host, A7.13 (iii), App. app:cost), times the logged serving time per
decision (request latency / questions in the request; requests went four at a time, so this is an upper bound on GPU
time). Sensitivity: the provisioned per-token tariff of the registered windows and zero. Latency: the median request
latency per decision as logged on each model's own host. Frontier: the drawn models that no other drawn model matches
or beats on both cost and accuracy (one strictly)."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import explore_common as C  # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "results/explore/e9_frontier"
TEX = ROOT / "paper/generated/e9_frontier.tex"
CHECK_CFG = ROOT / "configs/explore/e9_frontier/e1_dev_check.yaml"
DEV_IDS = ROOT / "configs/eval/e1_dev_prefix_ids.json"
CHECK_RUNS = RUNS / "explore_e9"
CHECK_N, SMOKE_N = 300, 30                 # --max-items (whole units: 299 and 29 items)
CAP_USD, PROJ_FACTOR = 30.0, 1.3
GPU_USD_PER_H = 2.24208                    # results/stage2/tariff.json host_usd_per_hour (one L40S, A7.13 (iii))
N_BOOT, BOOT_SEED = 2000, 0
LABEL = ("exploratory (E9), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-07 'E9: cost-accuracy frontier of "
         "multiple-choice decisions'; outside every Holm table")

# key -> (label, record file under runs/, kind, cost rule, drawn in the figure)
#   kind: dm = decision model, attr = attribution row of a decision model, llm = LLM multiple-choice decider,
#         fc = the agent LLM's own native function call, ctrl = control;  cost: api (logged) | gpu (tariff x time)
MODELS: dict[str, tuple[str, str, str, str, bool]] = {
    "jev": ("Jev", "e1_bfcl_when2call/s1_jev.jsonl", "dm", "api", True),
    "intern": ("Intern-Decision-4B", "e1_bfcl_when2call_startlux/s1_intern_4b.jsonl", "dm", "gpu", True),
    "sl0.8b": ("StartLux-0.8B", "e1_bfcl_when2call_startlux_ladder/s1_startlux_0_8b.jsonl", "dm", "gpu", True),
    "sl2b": ("StartLux-2B", "e1_bfcl_when2call_startlux_ladder/s1_startlux_2b.jsonl", "dm", "gpu", True),
    "sl4b": ("StartLux-4B", "e1_bfcl_when2call_startlux/s1_startlux_4b.jsonl", "dm", "gpu", True),
    "sl9b": ("StartLux-9B", "e1_bfcl_when2call_startlux_ladder/s1_startlux_9b.jsonl", "dm", "gpu", True),
    "sl27b": ("StartLux-27B (Q8_0)", "e1_bfcl_when2call_startlux/s1_startlux_27b.jsonl", "dm", "gpu", True),
    "rl": ("AgentDecision-4B", "e1_bfcl_when2call_rl/s1_jevagent_rl.jsonl", "dm", "gpu", True),
    "base": ("Qwen3-4B, untrained", "e1_bfcl_when2call_rl/s1_rl_base.jsonl", "attr", "gpu", False),
    "sftv1": ("SFT only (v1)", "e1_bfcl_when2call_rl/s1_rl_sft_v1.jsonl", "attr", "gpu", False),
    "flash": ("Gemini 3.5 Flash", "e1_bfcl_when2call/s1_gemini_flash_typed.jsonl", "llm", "api", True),
    "lite": ("Gemini 3.5 Flash-Lite", "e1_bfcl_when2call/s1_gemini_lite_dist.jsonl", "llm", "api", True),
    "litelabel": ("Flash-Lite, label + confidence", "e1_bfcl_when2call/s1_gemini_lite_label.jsonl", "llm", "api",
                  False),
    "gemma": ("Gemma 4 26B (unpriced)", "e1_bfcl_when2call/s1_gemma4_dist.jsonl", "llm", "api", False),
    "flashfc": ("Flash, own function call", "e1_bfcl_when2call/fc_gemini_flash_fc.jsonl", "fc", "api", True),
    "embed": ("Gemini embedding", "e1_bfcl_when2call/s1_gemini_embed.jsonl", "ctrl", "api", False),
    "bm25": ("BM25", "e1_bfcl_when2call/s1_bm25.jsonl", "ctrl", "api", False),
    "tfidf": ("TF-IDF", "e1_bfcl_when2call/s1_tfidf.jsonl", "ctrl", "api", False),
}
LLM_KEYS = ("flash", "lite", "litelabel", "flashfc")           # compared with the decision models
FAMILIES = {"toolchoice": ("tool_select", "bfcl"), "args": ("tool_params", "bfcl"), "w2c": ("tool_select", "when2call")}
FAMILY_LABEL = {"toolchoice": "BFCL tool choice", "args": "BFCL arguments", "w2c": "When2Call"}
# the check's backends (configs/explore/e9_frontier/e1_dev_check.yaml) -> label
CHECK_BACKENDS = {"jev": "Jev", "gemini_flash_typed": "Gemini 3.5 Flash", "gemini_lite_dist": "Gemini 3.5 Flash-Lite",
                  "luna_dist": "GPT-6-Luna", "sol_dist": "GPT-6.1-Sol"}
CHECK_KEY = {"jev": "jev", "gemini_flash_typed": "flash", "gemini_lite_dist": "lite", "luna_dist": "luna",
             "sol_dist": "sol"}
# registered tariff windows (sensitivity: provisioned per-token tariff)
TARIFFS = {"rl": ("results/stage2/tariff.json", "stage2_dev_tariff_rl/jevagent_best_rl.jsonl"),
           "sl4b": ("results/stage2/tariff_sl4b.json", "stage2_dev_tariff_sl4b/jevagent_best_sl4b.jsonl")}
TOKEN_RATE_SOURCE = {"rl": "rl", "base": "rl", "sftv1": "rl", "sl0.8b": "sl4b", "sl2b": "sl4b", "sl4b": "sl4b",
                     "sl9b": "sl4b", "sl27b": "sl4b"}   # Intern-Decision: its registered per-token price (logged)


# ================================================================================================ helpers
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def family_of(r: Mapping[str, Any]) -> str | None:
    key = (r.get("dp"), str(r.get("source") or "").split(":")[0])
    return next((f for f, v in FAMILIES.items() if v == key), None)


def is_correct(r: Mapping[str, Any]) -> bool:
    """A record without an answer (an error) counts as wrong; E1 has none."""
    return bool(r.get("correct")) and not r.get("error")


def gpu_cost(r: Mapping[str, Any], usd_per_h: float = GPU_USD_PER_H) -> float:
    """GPU tariff x serving time of one decision: the request's latency split over the questions it carried."""
    q = max(1, int(r.get("request_questions") or 1))
    return usd_per_h / 3.6e6 * float(r.get("latency_ms") or 0.0) / q


def decision_cost(r: Mapping[str, Any], rule: str) -> float:
    return gpu_cost(r) if rule == "gpu" else float(r.get("cost_usd") or 0.0)


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def paired_gap(a: Sequence[bool], b: Sequence[bool], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, Any]:
    """mean(a) - mean(b) in points, paired item bootstrap (percentile 95 %)."""
    x = np.asarray(a, float) - np.asarray(b, float)
    if len(x) == 0:
        return {"est": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = x[idx].mean(1)
    return {"est": 100 * float(x.mean()), "ci": [100 * float(np.quantile(boot, .025)), 100 * float(np.quantile(boot, .975))],
            "n": int(len(x)), "n_boot": n_boot, "seed": seed}


def frontier(points: Mapping[str, tuple[float, float]]) -> list[str]:
    """Keys of the points no other point matches or beats on both cost (lower) and accuracy (higher), one strictly;
    sorted by cost."""
    out = []
    for k, (c, a) in points.items():
        dominated = any(c2 <= c and a2 >= a and (c2 < c or a2 > a) for k2, (c2, a2) in points.items() if k2 != k)
        if not dominated:
            out.append(k)
    return sorted(out, key=lambda k: (points[k][0], -points[k][1]))


def cheapest_at_least(target_acc: float, cands: Mapping[str, tuple[float, float]]) -> str | None:
    ok = [k for k, (c, a) in cands.items() if a >= target_acc]
    return min(ok, key=lambda k: (cands[k][0], -cands[k][1])) if ok else None


def model_stats(recs: Sequence[Mapping[str, Any]], rule: str) -> dict[str, Any]:
    n = len(recs)
    k = sum(is_correct(r) for r in recs)
    fam: dict[str, Any] = {}
    for f in FAMILIES:
        rs = [r for r in recs if family_of(r) == f]
        if rs:
            fam[f] = {"n": len(rs), "acc": sum(is_correct(r) for r in rs) / len(rs)}
    costs = [decision_cost(r, rule) for r in recs]
    lat = [float(r.get("latency_ms") or 0.0) for r in recs if not r.get("error")]
    return {"n": n, "acc": k / n if n else float("nan"), "acc_ci": wilson(k, n), "families": fam,
            "usd": float(np.mean(costs)) if costs else float("nan"),
            "usd_logged": float(np.mean([float(r.get("cost_usd") or 0.0) for r in recs])) if recs else float("nan"),
            "priced": all(r.get("priced", True) for r in recs), "n_errors": sum(1 for r in recs if r.get("error")),
            "lat_ms_median": statistics.median(lat) if lat else float("nan"),
            "lat_ms_p90": float(np.quantile(lat, .9)) if lat else float("nan"),
            "tokens_in": float(np.mean([float(r.get("input_tokens") or 0.0) for r in recs])) if recs else float("nan")}


# ================================================================================================ main readout (E1)
def load_e1(runs: Path = RUNS, models: Mapping[str, tuple] = MODELS) -> dict[str, dict[str, dict[str, Any]]]:
    """key -> {item id -> record}; every model must answer exactly the same items."""
    out = {k: {r["id"]: r for r in read_jsonl(runs / v[1])} for k, v in models.items()}
    ids = [set(v) for v in out.values()]
    if any(s != ids[0] for s in ids):
        raise SystemExit("E1 record files do not cover the same items")
    return out


def token_rates(runs: Path = RUNS) -> dict[str, dict[str, Any]]:
    """Provisioned per-token tariff of each registered window: q x episodes / System-1 input tokens served."""
    out = {}
    for k, (tfile, run) in TARIFFS.items():
        t = json.loads((ROOT / tfile).read_text())
        tok, calls = 0.0, 0
        last: dict[str, Any] = {}
        for r in read_jsonl(runs / run):
            last[r["task_id"]] = r
        for r in last.values():
            for c in r.get("calls") or []:
                if c.get("component") == "s1":
                    tok += float(c.get("input_tokens") or 0.0)
                    calls += 1
        usd = float(t["q_usd_per_episode"]) * len(last)
        out[k] = {"usd_per_m": usd / tok * 1e6 if tok else float("nan"), "usd_per_call": usd / calls if calls else None,
                  "episodes": len(last), "s1_calls": calls, "s1_input_tokens": tok, "q": t["q_usd_per_episode"],
                  "host_usd_per_hour": t["host_usd_per_hour"], "source": tfile}
    return out


def analyse_e1(data: Mapping[str, Mapping[str, Mapping[str, Any]]], rates: Mapping[str, Mapping[str, Any]] | None,
               models: Mapping[str, tuple] = MODELS, n_boot: int = N_BOOT) -> dict[str, Any]:
    ids = sorted(next(iter(data.values())))
    stats = {k: model_stats([data[k][i] for i in ids], models[k][3]) for k in data}
    for k in stats:
        stats[k].update({"label": models[k][0], "kind": models[k][2], "cost_rule": models[k][3],
                         "drawn": bool(models[k][4]) and stats[k]["priced"]})
    drawn = {k: (s["usd"], s["acc"]) for k, s in stats.items() if s["drawn"]}
    kinds = {k: stats[k]["kind"] for k in stats}
    front = frontier(drawn)
    dms = {k: v for k, v in drawn.items() if stats[k]["kind"] == "dm"}
    res: dict[str, Any] = {"n": len(ids), "families": {f: sum(1 for i in ids if family_of(data["jev"][i]) == f)
                                                       for f in FAMILIES},
                           "models": stats, "frontier": front, "gpu_usd_per_h": GPU_USD_PER_H, "comparisons": {},
                           "frontier_only_dm_below_top_llm": only_dm_below_top_llm(front, drawn, kinds)}
    for k in stats:
        stats[k]["on_frontier"] = k in front
    jev = [is_correct(data["jev"][i]) for i in ids]
    for L in LLM_KEYS:
        if L not in stats or not stats[L]["priced"]:
            continue
        cand = cheapest_at_least(stats[L]["acc"], dms)
        gap = paired_gap(jev, [is_correct(data[L][i]) for i in ids], n_boot)
        res["comparisons"][L] = {
            "jev_gap_pt": gap, "jev_cost_ratio": stats[L]["usd"] / stats["jev"]["usd"],
            "cheapest_dm_at_least_as_accurate": cand,
            "cheapest_dm_cost_ratio": stats[L]["usd"] / stats[cand]["usd"] if cand else None,
            "best_dm": max(dms, key=lambda k: dms[k][1]),
        }
    best = max(dms, key=lambda k: dms[k][1])
    res["best_dm"] = {"key": best, "acc": dms[best][1], "usd": dms[best][0],
                      "gap_to_flash_pt": paired_gap([is_correct(data[best][i]) for i in ids],
                                                    [is_correct(data["flash"][i]) for i in ids], n_boot)}
    # sensitivity: other costs for the open-weights models (accuracy unchanged)
    sens: dict[str, Any] = {}
    rules = {"zero": lambda k, r: 0.0}
    if rates:
        def tok_rate(k: str, r: Mapping[str, Any]) -> float:
            if k == "intern":
                return float(r.get("cost_usd") or 0.0)            # its registered per-token price, as logged
            return float(r.get("input_tokens") or 0.0) * rates[TOKEN_RATE_SOURCE[k]]["usd_per_m"] / 1e6
        rules["token_tariff"] = tok_rate
    for name, fn in rules.items():
        pts = {k: ((float(np.mean([fn(k, data[k][i]) for i in ids])) if stats[k]["cost_rule"] == "gpu"
                    else stats[k]["usd"]), stats[k]["acc"]) for k in drawn}
        fr = frontier(pts)
        sens[name] = {"usd": {k: v[0] for k, v in pts.items()}, "frontier": fr,
                      "only_dm_below_top_llm": only_dm_below_top_llm(fr, pts, kinds)}
    res["sensitivity"] = sens
    res["token_rates"] = dict(rates or {})
    return res


# ================================================================================================ the check (new calls)
def check_items(max_items: int = CHECK_N) -> list[Any]:
    import yaml

    from jevharness.bench.decision_eval import stratified_sample
    from jevharness.bench.suites.static import load_static

    exp = yaml.safe_load(CHECK_CFG.read_text())
    items: list[Any] = []
    for s in exp["suites"]:
        s = dict(s)
        items.extend(load_static(s.pop("name"), **s))
    return stratified_sample(items, max_items)


def verify_ids() -> dict[str, Any]:
    """The check's items must be exactly the registered development prefix (A6.13), never a C8 item."""
    reg = set(json.loads(DEV_IDS.read_text())["ids"])
    full = {it.id for it in check_items(CHECK_N)}
    smoke = {it.id for it in check_items(SMOKE_N)}
    ok = full == reg and smoke <= reg
    return {"ok": ok, "registered": len(reg), "check": len(full), "smoke": len(smoke),
            "outside_prefix": sorted(full - reg)[:5]}


def ledger_spend(run_dir: Path) -> float:
    p = run_dir / "ledger.json"
    return float(json.loads(p.read_text()).get("fresh_cost_usd") or 0.0) if p.exists() else 0.0


def check_spend() -> dict[str, float]:
    s = {"smoke": ledger_spend(CHECK_RUNS / "e9_e1dev_check_smoke"), "full": ledger_spend(CHECK_RUNS / "e9_e1dev_check")}
    s["total"] = s["smoke"] + s["full"]
    return s


def cmd_ids(a: argparse.Namespace) -> int:
    v = verify_ids()
    print(json.dumps(v, indent=1))
    return 0 if v["ok"] else 1


def cmd_check(a: argparse.Namespace) -> int:
    import yaml

    from dotenv import load_dotenv

    from jevharness.bench.decision_eval import run_decision_eval

    v = verify_ids()
    if not v["ok"]:
        raise SystemExit(f"the check's items are not the registered dev prefix: {v}")
    exp = yaml.safe_load(CHECK_CFG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    proj_path = OUT / "check" / "projection.json"
    if a.smoke:
        exp["name"] += "_smoke"
        n = SMOKE_N
    else:
        if not proj_path.exists() or not json.loads(proj_path.read_text()).get("go"):
            raise SystemExit("no go from the smoke's gate (run `check --smoke` first)")
        n = CHECK_N
    if a.dry_run:
        print(json.dumps({"would_run": exp["name"], "max_items": n, "backends": list(exp["backends"]),
                          "spend_so_far": check_spend()}, indent=1))
        return 0
    load_dotenv(ROOT / ".env")
    run_decision_eval(exp, CHECK_RUNS, max_items=n)
    sp = check_spend()
    (OUT / "check").mkdir(parents=True, exist_ok=True)
    if a.smoke:
        recs = read_jsonl(CHECK_RUNS / exp["name"] / "s1_jev.jsonl")
        n_items = len(recs)
        proj = sp["smoke"] * (299 / max(1, n_items)) * PROJ_FACTOR + sp["total"]
        out = {"label": LABEL, "smoke_items": n_items, "smoke_fresh_usd": sp["smoke"], "projection_total": proj,
               "cap": CAP_USD, "factor": PROJ_FACTOR, "go": bool(proj <= CAP_USD),
               "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        proj_path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
        return 0 if out["go"] else 1
    archive_check()
    print(json.dumps({"spend": sp, "cap": CAP_USD}, indent=1))
    return 0


def archive_check() -> list[str]:
    """A compressed copy of the check's records and ledgers next to the readout (runs/ is not versioned)."""
    out = []
    for name in ("e9_e1dev_check", "e9_e1dev_check_smoke"):
        src = CHECK_RUNS / name
        dst = OUT / "check" / name
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.jsonl")):
            with open(p, "rb") as fi, open(dst / (p.name + ".gz"), "wb") as raw, \
                    gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fo:
                shutil.copyfileobj(fi, fo)
            out.append(str(dst / (p.name + ".gz")))
        for p in ("ledger.json", "experiment.yaml"):
            if (src / p).exists():
                shutil.copyfile(src / p, dst / p)
    return out


def cmd_archive(a: argparse.Namespace) -> int:
    print("\n".join(C.rel(Path(p)) for p in archive_check()))
    return 0


def load_check(base: Path = OUT / "check" / "e9_e1dev_check") -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    for b in CHECK_BACKENDS:
        p = base / f"s1_{b}.jsonl.gz"
        if p.exists():
            out[b] = {r["id"]: r for r in read_jsonl(p)}
    return out


def analyse_check(data: Mapping[str, Mapping[str, Mapping[str, Any]]], spend: Mapping[str, float] | None,
                  n_boot: int = N_BOOT) -> dict[str, Any]:
    if not data or "jev" not in data:
        return {"available": False}
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    stats = {b: model_stats([data[b][i] for i in ids], "api") for b in data}
    for b in stats:
        stats[b]["label"] = CHECK_BACKENDS[b]
    jev = [is_correct(data["jev"][i]) for i in ids]
    comp = {}
    for b in data:
        if b == "jev":
            continue
        comp[b] = {"jev_gap_pt": paired_gap(jev, [is_correct(data[b][i]) for i in ids], n_boot),
                   "jev_cost_ratio": stats[b]["usd"] / stats["jev"]["usd"] if stats["jev"]["usd"] else None}
    pts = {b: (s["usd"], s["acc"]) for b, s in stats.items()}
    return {"available": True, "n": len(ids), "models": stats, "comparisons": comp, "frontier": frontier(pts),
            "spend": dict(spend or {}), "cap": CAP_USD}


# ================================================================================================ outputs
def _sig(x: float, d: int = 2) -> str:
    """d significant digits, plain decimal ("0.016", "0.81", "1.1")."""
    if x is None or not math.isfinite(x):
        return r"\tbd"
    if x == 0:
        return "0"
    p = d - 1 - int(math.floor(math.log10(abs(x))))
    s = f"{round(x, p):.{max(p, 0)}f}"
    return s


def _ratio(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return r"\tbd"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def _pm(x: float | None, d: int = 1) -> str:
    """Signed, plain ASCII ("+3.1", "-1.7"): \\resm{} prints a true minus, \\resabs{} the magnitude."""
    if x is None or not math.isfinite(x):
        return r"\tbd"
    return f"{x:+.{d}f}"


def tikz_name(m: str) -> str:
    """A TikZ coordinate name for a model key (no dots: a dot would read as an anchor)."""
    return m.replace(".", "")


# ------------------------------------------------------------------------------------------------ E12's point
# Exploratory row E12 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E12: more training on public tool-use data"):
# the continued checkpoint AgentDecision-4B (mixed) answered the same 3,991 decisions on its own host; its point joins
# the drawn set by E9's cost rule. E9's registered frontier and keys stay as they are; the recomputed frontier and the
# point get their own keys (e9/front12/..., e9/e12/...). The host adjustment scales the point's cost by the frozen
# checkpoint's logged / re-hosted serving cost (the same 3,991 decisions on the E12 host).
# E12c (Deviations 2026-10-07 "E12c: the recipe on a stronger base, and same-engine serving cost") adds its own point:
# StartLux-4B (mixed). Each extra point is reported against E9's drawn set; the recomputed frontier holds them all.
EXTRA: dict[str, tuple[str, str, str | None]] = {
    "e12": ("AgentDecision-4B (mixed, E12)", "e1_bfcl_when2call_e12/s1_e12_mixed.jsonl",
            "e1_bfcl_when2call_e12/s1_frozen_rehost.jsonl"),
    "e12c": ("StartLux-4B (mixed, E12c)", "e1_bfcl_when2call_e12c/s1_e12c_sl4b.jsonl", None),
    # E16 (Deviations 2026-10-08 "E16: the best open-weights decision model on all three levels"): two more points,
    # each with the frozen checkpoint re-served on the E16 host (its own host factor); E12's / E12c's keys are unchanged
    "e16a": ("AgentDecision-4B (mixed+, E16)", "e1_bfcl_when2call_e16/s1_e16a.jsonl",
             "e1_bfcl_when2call_e16/s1_frozen_rehost.jsonl"),
    "e16b": ("StartLux-9B (mixed, E16)", "e1_bfcl_when2call_e16/s1_e16b.jsonl",
             "e1_bfcl_when2call_e16/s1_frozen_rehost.jsonl")}
E12_KEY = "e12"
FRONT12 = ("e12", "e12c")             # the points of the recomputed frontier e9/front12/... (as first written)
FRONT16 = ("e12", "e12c", "e16a", "e16b")   # e9/front16/...: E12's, E12c's and E16's points together


def _extra_point(k: str, e1: Mapping[str, Any], data: Mapping[str, Mapping[str, Mapping[str, Any]]], runs: Path
                 ) -> dict[str, Any]:
    label, f, rehost = EXTRA[k]
    p = runs / f
    if not p.exists():
        return {"available": False, "reason": f"no {f}"}
    recs = {r["id"]: r for r in read_jsonl(p)}
    ids = sorted(next(iter(data.values())))
    if set(recs) != set(ids):
        return {"available": False, "reason": f"{len(set(recs) & set(ids))} of {len(ids)} E1 decisions answered"}
    s = model_stats([recs[i] for i in ids], "gpu")
    M = e1["models"]
    drawn = {m: (v["usd"], v["acc"]) for m, v in M.items() if v["drawn"]}
    front = frontier({**drawn, k: (s["usd"], s["acc"])})
    gap = {ref: paired_gap([is_correct(recs[i]) for i in ids], [is_correct(data[ref][i]) for i in ids])
           for ref in ("rl", "jev", "sl4b")}
    out: dict[str, Any] = {"available": True, "key": k, "label": label, "stats": s, "on_frontier": k in front,
                           "frontier_alone": front,
                           "dominated_by": sorted(m for m, (c, a) in drawn.items()
                                                  if c <= s["usd"] and a >= s["acc"] and (c < s["usd"] or a > s["acc"])),
                           "gap_to_frozen_pt": gap["rl"], "gap_to_jev_pt": gap["jev"], "gap_to_sl4b_pt": gap["sl4b"],
                           "cost_vs_jev": s["usd"] / M["jev"]["usd"] if M["jev"]["usd"] else None}
    if rehost and (runs / rehost).exists():
        rr = {r["id"]: r for r in read_jsonl(runs / rehost)}
        if set(rr) >= set(ids):
            usd_rh = float(np.mean([gpu_cost(rr[i]) for i in ids]))
            factor = M["rl"]["usd"] / usd_rh if usd_rh > 0 else float("nan")
            adj = s["usd"] * factor
            fa = frontier({**drawn, k: (adj, s["acc"])})
            out.update({"rehost_usd": usd_rh, "host_factor": factor, "usd_hostadj": adj, "frontier_hostadj": fa,
                        "on_frontier_hostadj": k in fa})
    return out


def e12_point(e1: Mapping[str, Any], data: Mapping[str, Mapping[str, Mapping[str, Any]]], runs: Path = RUNS
              ) -> dict[str, Any]:
    """The E12 point (backward-compatible block) plus every extra point and the frontier recomputed with all of them."""
    pts = {k: _extra_point(k, e1, data, runs) for k in EXTRA}
    M = e1["models"]
    drawn = {m: (v["usd"], v["acc"]) for m, v in M.items() if v["drawn"]}
    hf = (pts.get("e12") or {}).get("host_factor")
    for k, x in pts.items():   # E12c ran on the same instance type (g5.8xlarge, one A10G): E12's measured factor
        if x.get("available") and x.get("usd_hostadj") is None and hf and math.isfinite(hf):
            adj = x["stats"]["usd"] * hf
            fa = frontier({**drawn, k: (adj, x["stats"]["acc"])})
            x.update({"host_factor": hf, "usd_hostadj": adj, "frontier_hostadj": fa, "on_frontier_hostadj": k in fa,
                      "host_factor_source": "the frozen checkpoint re-served on the E12 host (same instance type)"})
    out = dict(pts["e12"])
    out["frontier_without"] = list(e1["frontier"])
    have = {k: (x["stats"]["usd"], x["stats"]["acc"]) for k, x in pts.items() if x.get("available") and k in FRONT12}
    if have:
        fw = frontier({**drawn, **have})
        out["frontier_with"] = fw
        out["frontier_changed"] = fw != list(e1["frontier"])                       # a new point on it, or one dropped
        out["frontier_dropped"] = [m for m in e1["frontier"] if m not in fw]
        for k, x in pts.items():
            if x.get("available"):
                x["on_frontier_all"] = k in fw
    have16 = {k: (x["stats"]["usd"], x["stats"]["acc"]) for k, x in pts.items() if x.get("available") and k in FRONT16}
    if any(k in have16 for k in ("e16a", "e16b")):
        fw = frontier({**drawn, **have16})
        out["frontier_with16"] = fw
        out["frontier_changed16"] = fw != list(e1["frontier"])
        for k, x in pts.items():
            if x.get("available"):
                x["on_frontier_all16"] = k in fw
    out["extra"] = pts
    return out


def _point_keys(k: str, x: Mapping[str, Any]) -> dict[str, str]:
    s = x["stats"]
    v = {f"e9/{k}/acc": f"{100 * s['acc']:.1f}", f"e9/{k}/usdk": _sig(1000 * s["usd"]),
         f"e9/{k}/lat": C.fmt(int(round(s["lat_ms_median"]))), f"e9/{k}/front": "yes" if x["on_frontier"] else "no",
         f"e9/{k}/gapfrozen": _pm(x["gap_to_frozen_pt"]["est"]), f"e9/{k}/gapjev": _pm(x["gap_to_jev_pt"]["est"]),
         f"e9/{k}/gapsl4b": _pm(x["gap_to_sl4b_pt"]["est"])}
    for g in ("frozen", "jev", "sl4b"):        # unsigned, for prose ("by 1.9 points", "1.3 points below")
        v[f"e9/{k}/gap{g}abs"] = f"{abs(x[f'gap_to_{g}_pt']['est']):.1f}"
    if s["usd"] > 0:
        v[f"e9/{k}/lg"] = f"{math.log10(s['usd']):.4f}"
    for f, a in s["families"].items():
        v[f"e9/{k}/acc/{f}"] = f"{100 * a['acc']:.1f}"
    if x.get("cost_vs_jev"):
        v[f"e9/{k}/costvsjev"] = _ratio(x["cost_vs_jev"])
    if x.get("usd_hostadj") is not None and math.isfinite(x["usd_hostadj"]):
        v[f"e9/{k}/usdkadj"] = _sig(1000 * x["usd_hostadj"])
        v[f"e9/{k}/usdk_hostadj"] = v[f"e9/{k}/usdkadj"]          # the coordinator's name for the same key
        v[f"e9/{k}/frontadj"] = "yes" if x["on_frontier_hostadj"] else "no"
        if x["usd_hostadj"] > 0:
            v[f"e9/{k}/lgadj"] = f"{math.log10(x['usd_hostadj']):.4f}"
    return v


def e12_keys(x: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {}
    for key, pt in (x.get("extra") or {}).items():
        if pt.get("available"):
            k.update(_point_keys(key, pt))
    if x.get("frontier_with"):
        k.update({"e9/front12/first": tikz_name(x["frontier_with"][0]),
                  "e9/front12/rest": ",".join(tikz_name(m) for m in x["frontier_with"][1:]),
                  "e9/front12/n": str(len(x["frontier_with"])),
                  "e9/front12/changed": "yes" if x["frontier_changed"] else "no"})
    if x.get("frontier_with16"):
        k.update({"e9/front16/first": tikz_name(x["frontier_with16"][0]),
                  "e9/front16/rest": ",".join(tikz_name(m) for m in x["frontier_with16"][1:]),
                  "e9/front16/n": str(len(x["frontier_with16"])),
                  "e9/front16/changed": "yes" if x["frontier_changed16"] else "no"})
    return k


def e12_md(x: Mapping[str, Any], e1: Mapping[str, Any]) -> list[str]:
    L = ["", "## E12's and E12c's points (exploratory E12 / E12c; E9's frontier above is unchanged)", ""]
    M = e1["models"]
    lab = lambda m: EXTRA[m][0] if m in EXTRA else M[m]["label"]  # noqa: E731
    any_ = False
    for k, pt in (x.get("extra") or {}).items():
        if not pt.get("available"):
            L.append(f"{EXTRA[k][0]}: not available ({pt.get('reason')}).")
            continue
        any_ = True
        s = pt["stats"]
        g, j, sl = pt["gap_to_frozen_pt"], pt["gap_to_jev_pt"], pt["gap_to_sl4b_pt"]
        L += [f"{pt['label']}: {100 * s['acc']:.1f} % [{100 * s['acc_ci'][0]:.1f}, {100 * s['acc_ci'][1]:.1f}] ("
              + ", ".join(f"{FAMILY_LABEL[f]} {100 * v['acc']:.1f}" for f, v in s["families"].items())
              + f"), ${1000 * s['usd']:.4f} per 1,000 decisions (E9's rule, its own host), median latency "
              f"{s['lat_ms_median']:.0f} ms; vs AgentDecision-4B {g['est']:+.1f} points [{g['ci'][0]:+.1f}, "
              f"{g['ci'][1]:+.1f}], vs StartLux-4B {sl['est']:+.1f} [{sl['ci'][0]:+.1f}, {sl['ci'][1]:+.1f}], vs Jev "
              f"{j['est']:+.1f} [{j['ci'][0]:+.1f}, {j['ci'][1]:+.1f}]. On the frontier (alone with E9's drawn set): "
              f"{'yes' if pt['on_frontier'] else 'no'}"
              + (f" (dominated by {', '.join(lab(m) for m in pt['dominated_by'])})" if pt["dominated_by"] else "") + "."]
        if pt.get("usd_hostadj") is not None:
            src = (f"the frozen checkpoint re-served on the same host: ${1000 * pt['rehost_usd']:.4f} vs "
                   f"${1000 * M['rl']['usd']:.4f} per 1,000 as logged" if pt.get("rehost_usd") is not None
                   else pt.get("host_factor_source", ""))
            L.append(f"  Host adjustment ({src}; factor {pt['host_factor']:.3f}): "
                     f"${1000 * pt['usd_hostadj']:.4f} per 1,000; on the frontier: "
                     f"{'yes' if pt['on_frontier_hostadj'] else 'no'}.")
    if any_ and x.get("frontier_with"):
        L.append(f"Frontier with the new point(s): {' → '.join(lab(m) for m in x['frontier_with'])}; changed: "
                 f"{'yes' if x['frontier_changed'] else 'no'}.")
    return L

def only_dm_below_top_llm(front: Sequence[str], pts: Mapping[str, tuple[float, float]],
                          kinds: Mapping[str, str]) -> bool:
    """Every frontier point cheaper than the most accurate LLM point is a decision model."""
    llms = [k for k in pts if kinds[k] in ("llm", "fc")]
    if not llms:
        return True
    top = max(llms, key=lambda k: (pts[k][1], -pts[k][0]))
    return all(kinds[k] == "dm" for k in front if pts[k][0] < pts[top][0])


TEX_NAME = {"jev": r"\jev{}", "rl": r"\ours{}", "intern": "Intern-Decision-4B", "sl0.8b": "StartLux-0.8B",
            "sl2b": "StartLux-2B", "sl4b": "StartLux-4B", "sl9b": "StartLux-9B", "sl27b": "StartLux-27B",
            "flash": "Flash", "lite": "Flash-Lite", "litelabel": "Flash-Lite (label)", "flashfc": "Flash's own tool call"}


def tex_keys(e1: Mapping[str, Any], chk: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {"e9/n": C.fmt(int(e1["n"])), "e9/gpu/usdh": f"{e1['gpu_usd_per_h']:.2f}",
                         "e9/cap": f"{CAP_USD:.0f}"}
    for f, n in e1["families"].items():
        k[f"e9/n/{f}"] = C.fmt(int(n))
    for m, s in e1["models"].items():
        k[f"e9/{m}/acc"] = f"{100 * s['acc']:.1f}"
        for f, v in s["families"].items():
            k[f"e9/{m}/acc/{f}"] = f"{100 * v['acc']:.1f}"
        k[f"e9/{m}/lat"] = C.fmt(int(round(s["lat_ms_median"])))
        if s["priced"]:
            k[f"e9/{m}/usdk"] = _sig(1000 * s["usd"])          # US$ per 1,000 decisions
            if s["usd"] > 0:
                k[f"e9/{m}/lg"] = f"{math.log10(s['usd']):.4f}"   # log10 US$ per decision (figure coordinate)
        k[f"e9/{m}/front"] = "yes" if s.get("on_frontier") else "no"
    k["e9/front/first"] = tikz_name(e1["frontier"][0])
    k["e9/front/rest"] = ",".join(tikz_name(m) for m in e1["frontier"][1:])
    k["e9/front/onlydm"] = "yes" if e1["frontier_only_dm_below_top_llm"] else "no"
    k["e9/front/n"] = str(len(e1["frontier"]))
    k["e9/front/dm"] = str(sum(1 for m in e1["frontier"] if e1["models"][m]["kind"] == "dm"))
    k["e9/front/llm"] = str(sum(1 for m in e1["frontier"] if e1["models"][m]["kind"] in ("llm", "fc")))
    for L, c in e1["comparisons"].items():
        g = c["jev_gap_pt"]
        k[f"e9/jev/vs/{L}/gap"] = _pm(g["est"])
        k[f"e9/jev/vs/{L}/gapabs"] = f"{abs(g['est']):.1f}"
        k[f"e9/jev/vs/{L}/gaplo"] = _pm(g["ci"][0])
        k[f"e9/jev/vs/{L}/gaphi"] = _pm(g["ci"][1])
        k[f"e9/jev/vs/{L}/ratio"] = _ratio(c["jev_cost_ratio"])
        cand = c["cheapest_dm_at_least_as_accurate"]
        k[f"e9/cheapest/{L}/model"] = TEX_NAME.get(cand, cand) if cand else "none"
        if cand:
            k[f"e9/cheapest/{L}/ratio"] = _ratio(c["cheapest_dm_cost_ratio"])
    b = e1["best_dm"]
    k["e9/bestdm/model"] = TEX_NAME.get(b["key"], b["key"])
    k["e9/bestdm/gap"] = _pm(b["gap_to_flash_pt"]["est"])
    k["e9/bestdm/gaplo"] = _pm(b["gap_to_flash_pt"]["ci"][0])
    k["e9/bestdm/gaphi"] = _pm(b["gap_to_flash_pt"]["ci"][1])
    for name, s in e1["sensitivity"].items():
        k[f"e9/sens/{name}/jevfront"] = "yes" if "jev" in s["frontier"] else "no"
        k[f"e9/sens/{name}/front/n"] = str(len(s["frontier"]))
        k[f"e9/sens/{name}/front/dm"] = str(sum(1 for m in s["frontier"] if e1["models"][m]["kind"] == "dm"))
        k[f"e9/sens/{name}/onlydm"] = "yes" if s["only_dm_below_top_llm"] else "no"
    for name, r in (e1.get("token_rates") or {}).items():
        k[f"e9/tokrate/{name}"] = f"{r['usd_per_m']:.3f}"
    if chk.get("available"):
        k["e9/check/n"] = C.fmt(int(chk["n"]))
        for b, s in chk["models"].items():
            ck = CHECK_KEY[b]
            k[f"e9/check/{ck}/acc"] = f"{100 * s['acc']:.1f}"
            k[f"e9/check/{ck}/usdk"] = _sig(1000 * s["usd"])
            k[f"e9/check/{ck}/lat"] = C.fmt(int(round(s["lat_ms_median"])))
        for b, c in chk["comparisons"].items():
            ck = CHECK_KEY[b]
            g = c["jev_gap_pt"]
            k[f"e9/check/jev/vs/{ck}/gap"] = _pm(g["est"])
            k[f"e9/check/jev/vs/{ck}/gapabs"] = f"{abs(g['est']):.1f}"
            k[f"e9/check/jev/vs/{ck}/gaplo"] = _pm(g["ci"][0])
            k[f"e9/check/jev/vs/{ck}/gaphi"] = _pm(g["ci"][1])
            k[f"e9/check/jev/vs/{ck}/ratio"] = _ratio(c["jev_cost_ratio"])
        k["e9/check/jevfront"] = "yes" if "jev" in chk["frontier"] else "no"
        k["e9/spend"] = f"{(chk.get('spend') or {}).get('total', 0.0):.2f}"
    return k


def write_tex(vals: Mapping[str, str], path: Path = TEX) -> Path:
    lines = ["% AUTO-GENERATED by scripts/explore_e9_frontier.py -- do not edit by hand.",
             "% Exploratory row E9, cost-accuracy frontier of multiple-choice decisions (descriptive; "
             "docs/PREREG_A7_STAGE2.md Deviations 2026-10-07), outside both stages.",
             f"% last written {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git "
             f"{(C.git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


def _usd(x: float) -> str:
    return f"{x:.2e}" if x and math.isfinite(x) else ("0" if x == 0 else "–")


def render_md(res: Mapping[str, Any]) -> str:
    e1, chk = res["e1"], res["check"]
    L = ["# E9: the cost-accuracy frontier of multiple-choice decisions (exploratory, descriptive)", "", LABEL + ".", "",
         f"Generated {res['written']} at git {(res.get('git') or '?')[:12]} by `python scripts/explore_e9_frontier.py "
         "summarize`. Inventory and cost rules: `DATA.md`.", "",
         f"## Main readout: E1's {e1['n']:,} L1 decisions (no new call)", "",
         "Items: " + ", ".join(f"{FAMILY_LABEL[f]} {n:,}" for f, n in e1["families"].items()) + ". "
         f"Cost per decision: API models as logged (list price, reasoning tokens included); open-weights models "
         f"${e1['gpu_usd_per_h']:.5f} per GPU-hour × logged serving time per decision (request latency / questions "
         "in the request; four requests in flight, so an upper bound). Latency: median request latency as logged on "
         "each model's host. Frontier = no drawn model matches or beats it on both cost and accuracy.", "",
         "| model | kind | accuracy % [Wilson 95 %] | " + " | ".join(FAMILY_LABEL[f] for f in FAMILIES) +
         " | $ per 1,000 decisions | median latency ms | on the frontier |",
         "|---|---|---|" + "---|" * len(FAMILIES) + "---|---|---|"]
    order = sorted(e1["models"], key=lambda m: (not e1["models"][m]["drawn"], e1["models"][m]["kind"] != "dm",
                                                e1["models"][m]["usd"] if e1["models"][m]["priced"] else 9))
    for m in order:
        s = e1["models"][m]
        fam = " | ".join(f"{100 * s['families'][f]['acc']:.1f}" if f in s["families"] else "–" for f in FAMILIES)
        usd = f"{1000 * s['usd']:.4f}" if s["priced"] else "unpriced"
        front = ("yes" if s["on_frontier"] else "no") if s["drawn"] else "not drawn"
        L.append(f"| {s['label']} | {s['kind']} ({s['cost_rule']}) | {100 * s['acc']:.1f} [{100 * s['acc_ci'][0]:.1f}, "
                 f"{100 * s['acc_ci'][1]:.1f}] | {fam} | {usd} | {s['lat_ms_median']:.0f} | {front} |")
    L += ["", "Frontier (by cost): " + " → ".join(e1["models"][m]["label"] for m in e1["frontier"]) + ".", ""]
    L += ["Against each LLM (Jev − LLM, points, paired item bootstrap B = 2,000; cost ratio = LLM / Jev; the cheapest "
          "decision model at least as accurate as the LLM and how many times cheaper it is):", "",
          "| LLM | Jev − LLM [95 % CI] | LLM cost / Jev cost | cheapest decision model ≥ LLM accuracy | cost ratio |",
          "|---|---|---|---|---|"]
    for Lk, c in e1["comparisons"].items():
        g = c["jev_gap_pt"]
        cand = c["cheapest_dm_at_least_as_accurate"]
        L.append(f"| {e1['models'][Lk]['label']} | {g['est']:+.1f} [{g['ci'][0]:+.1f}, {g['ci'][1]:+.1f}] | "
                 f"{c['jev_cost_ratio']:.1f}× | {e1['models'][cand]['label'] if cand else 'none'} | "
                 f"{(str(round(c['cheapest_dm_cost_ratio'], 1)) + '×') if cand else '–'} |")
    b = e1["best_dm"]
    g = b["gap_to_flash_pt"]
    L += ["", f"Most accurate decision model: {e1['models'][b['key']]['label']} ({100 * b['acc']:.1f} %), "
          f"{g['est']:+.1f} points vs Flash [{g['ci'][0]:+.1f}, {g['ci'][1]:+.1f}].", "",
          "Sensitivity (open-weights cost only; accuracy unchanged):", ""]
    for name, s in e1["sensitivity"].items():
        L.append(f"- `{name}`: frontier = " + " → ".join(e1["models"][m]["label"] for m in s["frontier"]) +
                 "; open-weights $ per 1,000 decisions: " +
                 ", ".join(f"{e1['models'][m]['label']} {1000 * v:.4f}" for m, v in s["usd"].items()
                           if e1["models"][m]["cost_rule"] == "gpu") + ".")
    for name, r in (e1.get("token_rates") or {}).items():
        L.append(f"- provisioned per-token tariff `{name}` ({r['source']}): ${r['usd_per_m']:.4f} / M System-1 input "
                 f"tokens (q = ${r['q']} × {r['episodes']} episodes / {r['s1_input_tokens']:,.0f} tokens; "
                 f"${r['usd_per_call']:.2e} per System-1 call).")
    L += ["", "## The check: E1's 299-item development prefix (new calls)", ""]
    if not chk.get("available"):
        L.append("Not run (no records in `check/e9_e1dev_check/`).")
    else:
        sp = chk.get("spend") or {}
        L += [f"Items: {chk['n']} (the registered dev prefix, excluded from C8). Spend: ${sp.get('total', 0):.2f} "
              f"(smoke ${sp.get('smoke', 0):.2f}, full ${sp.get('full', 0):.2f}); cap ${chk['cap']:.0f}. Every row "
              "measured in this session (a new answer cache).", "",
              "| decider | accuracy % [Wilson 95 %] | " + " | ".join(FAMILY_LABEL[f] for f in FAMILIES) +
              " | $ per 1,000 decisions | median latency ms | errors | Jev − decider [95 % CI] | cost / Jev |",
              "|---|---|" + "---|" * len(FAMILIES) + "---|---|---|---|---|"]
        for bk, s in sorted(chk["models"].items(), key=lambda t: t[1]["usd"]):
            fam = " | ".join(f"{100 * s['families'][f]['acc']:.1f}" if f in s["families"] else "–" for f in FAMILIES)
            c = chk["comparisons"].get(bk)
            gap = (f"{c['jev_gap_pt']['est']:+.1f} [{c['jev_gap_pt']['ci'][0]:+.1f}, {c['jev_gap_pt']['ci'][1]:+.1f}]"
                   if c else "–")
            ratio = f"{c['jev_cost_ratio']:.1f}×" if c and c["jev_cost_ratio"] else "–"
            L.append(f"| {s['label']} | {100 * s['acc']:.1f} [{100 * s['acc_ci'][0]:.1f}, {100 * s['acc_ci'][1]:.1f}] | "
                     f"{fam} | {1000 * s['usd']:.4f} | {s['lat_ms_median']:.0f} | {s['n_errors']} | {gap} | {ratio} |")
        L += ["", "Frontier of the check (by cost): " + " → ".join(chk["models"][b]["label"] for b in chk["frontier"]) +
              "."]
    L += ["", "## Reading", ""] + [f"- {x}" for x in res["reading"]] + [""]
    return "\n".join(L)


def reading(e1: Mapping[str, Any], chk: Mapping[str, Any]) -> list[str]:
    M = e1["models"]
    out = []
    dm_front = [m for m in e1["frontier"] if M[m]["kind"] == "dm"]
    llm_front = [m for m in e1["frontier"] if M[m]["kind"] != "dm"]
    out.append(f"On E1, the frontier holds {len(dm_front)} decision model(s) ({', '.join(M[m]['label'] for m in dm_front)})"
               f" and {len(llm_front)} LLM point(s) ({', '.join(M[m]['label'] for m in llm_front) or 'none'}); "
               f"every frontier point cheaper than the most accurate LLM is a decision model: "
               f"{'yes' if e1['frontier_only_dm_below_top_llm'] else 'no'}.")
    for Lk, c in e1["comparisons"].items():
        cand = c["cheapest_dm_at_least_as_accurate"]
        g = c["jev_gap_pt"]
        s = (f"{M[Lk]['label']} ({100 * M[Lk]['acc']:.1f} %, ${1000 * M[Lk]['usd']:.3f} per 1,000): Jev is "
             f"{g['est']:+.1f} points [{g['ci'][0]:+.1f}, {g['ci'][1]:+.1f}] at 1/{c['jev_cost_ratio']:.0f} of its cost")
        s += (f"; the cheapest decision model at least as accurate is {M[cand]['label']}, "
              f"{c['cheapest_dm_cost_ratio']:.1f}× cheaper." if cand else "; no decision model is as accurate.")
        out.append(s)
    for name, s in e1["sensitivity"].items():
        out.append(f"Sensitivity `{name}`: Jev {'stays' if 'jev' in s['frontier'] else 'is not'} on the frontier; "
                   f"frontier = {', '.join(M[m]['label'] for m in s['frontier'])}; only decision models below the "
                   f"most accurate LLM: {'yes' if s['only_dm_below_top_llm'] else 'no'}.")
    if chk.get("available"):
        Mc = chk["models"]
        for bk, c in chk["comparisons"].items():
            g = c["jev_gap_pt"]
            out.append(f"Check (299 dev items): Jev − {Mc[bk]['label']} {g['est']:+.1f} points "
                       f"[{g['ci'][0]:+.1f}, {g['ci'][1]:+.1f}], {Mc[bk]['label']} costs {c['jev_cost_ratio']:.1f}× Jev.")
        out.append("Check frontier: " + ", ".join(Mc[b]["label"] for b in chk["frontier"]) + ".")
    return out


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def cmd_summarize(a: argparse.Namespace) -> int:
    data = load_e1()
    e1 = analyse_e1(data, token_rates(), n_boot=a.n_boot)
    e12 = e12_point(e1, data)
    chk = analyse_check(load_check(), check_spend() if CHECK_RUNS.exists() else None, n_boot=a.n_boot)
    if chk.get("available") and not (chk.get("spend") or {}).get("total"):
        led = OUT / "check"
        sp = {k: ledger_spend(led / n) for k, n in (("smoke", "e9_e1dev_check_smoke"), ("full", "e9_e1dev_check"))}
        sp["total"] = sp["smoke"] + sp["full"]
        chk["spend"] = sp
    res = {"label": LABEL, "e1": e1, "check": chk, "reading": reading(e1, chk),
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": C.git_head(), "e12": e12}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(_jsonable(res), indent=1) + "\n")
    (OUT / "summary.md").write_text(render_md(res) + "\n".join(e12_md(e12, e1)) + "\n")
    write_tex({**tex_keys(e1, chk), **e12_keys(e12)})
    print("\n".join(res["reading"]))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ids")
    sub.add_parser("archive")
    c = sub.add_parser("check")
    c.add_argument("--smoke", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(list(argv) if argv is not None else None)
    return {"ids": cmd_ids, "check": cmd_check, "archive": cmd_archive, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
