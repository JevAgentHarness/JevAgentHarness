#!/usr/bin/env python3
"""E2 (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry of
2026-10-05): the StartLux-Decision size ladder live, beside the sizes already measured, so that model size can be read
against static accuracy (E1, C8 population), common-state accuracy (corpus C, A7.18 (e) 2, as reported in
results/stage2/stage2.json) and live success / own-state accuracy (tool_select replace, Flash, the first 200 fresh
Stage-2 tasks). No test, outside every Holm table; nothing here revises a registered result.

    python scripts/explore_startlux_live.py      # -> results/explore/startlux_live/summary.{json,md} + explore.tex

Rows: StartLux 0.8B / 2B / 9B (E2: runs/explore_startlux_live, runs/e1_bfcl_when2call_startlux_ladder), 4B and 27B (Q8_0)
(A7.18: runs/stage2_startlux_states, runs/e1_bfcl_when2call_startlux), and the reference rows Jev, Intern-Decision-4B and
JevAgentRL-4B (runs/stage2_rl_states, runs/e1_bfcl_when2call*). Paired differences to Jev: live success and E1 with
Newcombe intervals on the common tasks / items; C as reported. Kendall's tau-b of size against each StartLux measure."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402

from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.decision_eval import _rescore  # noqa: E402
from jevharness.bench.scorecard import task_key  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "results/explore/startlux_live"
STAGE2_JSON = ROOT / "results/stage2/stage2.json"
TASKS = ROOT / "configs/eval/jevgym_stage2_tasks.json"
LIVE_N = 200
LABEL = "exploratory (E2)"
# row -> (size in B or None, live (run dir, arm), E1 (run dir, backend), C row (stage2.json table, row), source)
ROWS: dict[str, tuple[float | None, tuple[str, str], tuple[str, str], tuple[str, str], str]] = {
    "sl0.8b": (0.8, ("explore_startlux_live", "sl0.8b_ts_replace_f"), ("e1_bfcl_when2call_startlux_ladder", "startlux_0_8b"),
               ("stage2_ladder", "sl0.8b"), "E2"),
    "sl2b": (2.0, ("explore_startlux_live", "sl2b_ts_replace_f"), ("e1_bfcl_when2call_startlux_ladder", "startlux_2b"),
             ("stage2_ladder", "sl2b"), "E2"),
    "sl4b": (4.0, ("stage2_startlux_states", "sl4b_ts_replace_f"), ("e1_bfcl_when2call_startlux", "startlux_4b"),
             ("stage2_s1", "sl4b"), "A7.18"),
    "sl9b": (9.0, ("explore_startlux_live", "sl9b_ts_replace_f"), ("e1_bfcl_when2call_startlux_ladder", "startlux_9b"),
             ("stage2_ladder", "sl9b"), "E2"),
    "sl27b": (27.0, ("stage2_startlux_states", "sl27b_ts_replace_f"), ("e1_bfcl_when2call_startlux", "startlux_27b"),
              ("stage2_s1", "sl27b"), "A7.18 (27B Q8_0)"),
    "jev": (None, ("stage2_rl_states", "jev_ts_replace_f"), ("e1_bfcl_when2call", "jev"), ("stage2_s1", "jev"), "A7.10"),
    "intern": (None, ("stage2_rl_states", "intern_ts_replace_f"), ("e1_bfcl_when2call_startlux", "intern_4b"),
               ("stage2_s1", "intern"), "A7.10"),
    "frozen": (None, ("stage2_rl_states", "rl_ts_replace_f"), ("e1_bfcl_when2call_rl", "jevagent_rl"),
               ("stage2_s1", "frozen"), "A7.10"),
}
REFERENCE = "jev"
STARTLUX = ("sl0.8b", "sl2b", "sl4b", "sl9b", "sl27b")


def e1_items(path: Path) -> dict[str, float] | None:
    """item -> correctness on the BFCL tool_select records of one E1 backend (errors count wrong), as A7.18 (h)."""
    if not path.exists():
        return None
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {str(r.get("id")): (float(_rescore(r)) if "error" not in r else 0.0) for r in recs
            if r.get("dp") == "tool_select" and str(r.get("source", "")).startswith("bfcl:")}


def own_accuracy(trajs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """task -> share of its executed replace-mode tool_select decisions that are correct (own states), as A7.18 (h)."""
    out: dict[str, list[float]] = {}
    for r in M.decision_records([t for t in trajs if not t.get("error")]):
        if r.get("dp") != "tool_select" or r.get("mode") != "replace" or r.get("s1_value") is None:
            continue
        ok = M.is_correct(r, r["s1_value"])
        if ok is not None:
            out.setdefault(str(r["task_id"]), []).append(float(ok))
    return {k: float(np.mean(v)) for k, v in out.items()}


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else None


def live_row(path: Path, live_tasks: set[str]) -> tuple[dict[str, Any], dict[str, float]]:
    if not path.exists():
        return {}, {}
    latest, attempts = C.load_latest([path])
    ok = [t for t in latest.values() if not t.get("error") and task_key(t) in live_tasks]
    errd = [t for t in latest.values() if t.get("error") and task_key(t) in live_tasks]
    own = own_accuracy(ok)
    y = {task_key(t): float(C.success(t)) for t in ok}
    r = {"live_n": len(ok), "live_errored": len(errd), "live_success": float(np.mean(list(y.values()))) if y else None,
         "own_acc": float(np.mean(list(own.values()))) if own else None,
         "steps": float(np.mean([len(t.get("steps") or []) for t in ok])) if ok else None,
         "s1_calls": float(np.mean([C.s_calls(t, "s1") for t in ok])) if ok else None,
         "s2_calls": float(np.mean([C.s_calls(t, "s2") for t in ok])) if ok else None,
         "api_cost": float(np.mean([C.cost(t) for t in ok])) if ok else None,
         "wall_s": float(np.mean([float(t.get("wall_ms") or 0) for t in ok])) / 1000 if ok else None,
         "unpriced": sorted({m for t in ok for m in ((t.get("usage") or {}).get("unpriced_models") or [])}),
         "max_attempts": max(attempts.values()) if attempts else 0}
    return r, y


def analyze(runs: Path, stage2: Mapping[str, Any], live_tasks: set[str]) -> dict[str, Any]:
    desc = (stage2.get("descriptive") or {})
    rows: dict[str, Any] = {}
    raw: dict[str, dict[str, Any]] = {}
    for row, (size, (lr, arm), (er, be), (ct, crow), src) in ROWS.items():
        lp, ep = runs / lr / f"{arm}.jsonl", runs / er / f"s1_{be}.jsonl"
        r, y = live_row(lp, live_tasks)
        items = e1_items(ep)
        tab = desc.get(ct) or {}
        tab = tab.get("rows", tab) if isinstance(tab, Mapping) else {}
        crec = tab.get(crow) or {}
        r.update({"size_b": size, "source": src, "live_arm": f"{lr}/{arm}", "e1": f"{er}/s1_{be}",
                  "e1_acc": float(np.mean(list(items.values()))) if items else None, "e1_n": len(items or {}),
                  "corpus_acc": crec.get("corpus_acc"), "corpus_acc_offplan": crec.get("corpus_acc_offplan"),
                  "corpus_vs_jev": (crec.get("vs_jev") or {}).get("corpus_delta"),
                  "corpus_vs_jev_ci": (crec.get("vs_jev") or {}).get("corpus_delta_ci")})
        if lp.exists():
            r["live_sha256"] = C.sha256_file(lp)[:16]
        if ep.exists():
            r["e1_sha256"] = C.sha256_file(ep)[:16]
        rows[row] = r
        raw[row] = {"live": y, "e1": items or {}}
    ref = raw[REFERENCE]
    for row in rows:
        if row == REFERENCE:
            continue
        for key in ("live", "e1"):
            a, b = raw[row][key], ref[key]
            common = sorted(set(a) & set(b))
            if common:
                nc = C.newcombe([a[k] for k in common], [b[k] for k in common])
                rows[row][f"{key}_vs_jev"] = nc["diff_pt"] / 100
                rows[row][f"{key}_vs_jev_ci"] = [nc["ci_pt"][0] / 100, nc["ci_pt"][1] / 100]
                rows[row][f"{key}_vs_jev_n"] = len(common)
    kend: dict[str, Any] = {}
    for m in ("e1_acc", "corpus_acc", "live_success", "own_acc"):
        pts = [(rows[r]["size_b"], rows[r][m]) for r in STARTLUX if rows[r].get(m) is not None]
        kend[m] = {"tau_b": kendall_tau_b([p[0] for p in pts], [p[1] for p in pts]), "n": len(pts)}
    full = [r for r in rows if all(rows[r].get(k) is not None for k in ("e1_acc", "corpus_acc", "live_success"))]
    if len(full) >= 3:
        kend["e1_vs_live_all_rows"] = {"tau_b": kendall_tau_b([rows[r]["e1_acc"] for r in full],
                                                              [rows[r]["live_success"] for r in full]), "rows": full}
        kend["corpus_vs_live_all_rows"] = {"tau_b": kendall_tau_b([rows[r]["corpus_acc"] for r in full],
                                                                  [rows[r]["live_success"] for r in full]), "rows": full}
    return {"label": f"{LABEL}: descriptive, no test; revises no registered result", "live_n": LIVE_N, "rows": rows,
            "kendall_size": kend}


def _pct(v: Any, d: int = 1) -> str:
    return "–" if v is None else f"{100 * v:.{d}f}"


def _num(v: Any, d: int = 1) -> str:
    return "–" if v is None else f"{v:.{d}f}"


def _delta(x: Mapping[str, Any], key: str) -> str:
    if x.get(key) is None:
        return ""
    ci = x.get(f"{key}_ci") or [None, None]
    return f" ({_pct(x[key])} [{_pct(ci[0])}, {_pct(ci[1])}])"


def _md(r: Mapping[str, Any]) -> str:
    out = ["# E2 — StartLux-Decision size ladder, live (exploratory)", "",
           f"{r['label']}. Live: tool_select replace, Flash, the first {r['live_n']} fresh Stage-2 tasks (seeds "
           "300000–300199), latest record per task, errored tasks left out. E1: BFCL tool_select items of the C8 "
           "population. C: the common-state corpus accuracy as reported in results/stage2/stage2.json.", "",
           "| row | size (B) | source | E1 acc (Δ vs Jev [Newcombe]) | C acc (Δ vs Jev) | live success (Δ vs Jev [Newcombe]) | "
           "own-state acc | steps | S1 / S2 calls | API $ / ep. | n live |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row, x in r["rows"].items():
        cd = f" ({_pct(x['corpus_vs_jev'])})" if x.get("corpus_vs_jev") is not None else ""
        calls = "–" if x.get("s1_calls") is None else f"{_num(x['s1_calls'])} / {_num(x['s2_calls'])}"
        out.append(f"| {row} | {_num(x['size_b'])} | {x['source']} | {_pct(x.get('e1_acc'))}{_delta(x, 'e1_vs_jev')} | "
                   f"{_pct(x.get('corpus_acc'))}{cd} | {_pct(x.get('live_success'))}{_delta(x, 'live_vs_jev')} | "
                   f"{_pct(x.get('own_acc'))} | {_num(x.get('steps'))} | {calls} | {_num(x.get('api_cost'), 4)} | "
                   f"{x.get('live_n', '–')} |")
    kend = "; ".join(f"{k} {_num(v.get('tau_b'), 2)} (n {v.get('n', len(v.get('rows') or []))})"
                     for k, v in r["kendall_size"].items())
    out += ["", f"Kendall's τ-b (size vs each StartLux measure; the *_all_rows entries rank every row with all three "
            f"measures): {kend}.", "",
            "StartLux E1 rows: public BFCL / When2Call items; overlap with the vendor's training data cannot be excluded "
            "(A7.18 (c)). The 27B row is the Q8_0 GGUF (A7.18 (g)). StartLux System-1 calls are priced at an exact 0 "
            "(API cost only).", ""]
    return "\n".join(out)


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {}
    for row, x in r["rows"].items():
        for m in ("e1_acc", "corpus_acc", "live_success", "own_acc", "live_vs_jev", "e1_vs_jev"):
            if x.get(m) is not None:
                k[f"{row}/{m}"] = C.fmt(100 * x[m])
        for m in ("live_vs_jev_ci", "e1_vs_jev_ci"):
            ci = x.get(m)
            if ci:
                k[f"{row}/{m}/lo"], k[f"{row}/{m}/hi"] = C.fmt(100 * ci[0]), C.fmt(100 * ci[1])
        if x.get("steps") is not None:
            k[f"{row}/steps"] = C.fmt(x["steps"])
        if x.get("api_cost") is not None:
            k[f"{row}/api_cost"] = C.fmt(x["api_cost"], 4)
        if x.get("live_n") is not None:
            k[f"{row}/live_n"] = C.fmt(int(x["live_n"]))
    for m, v in r["kendall_size"].items():
        if v.get("tau_b") is not None:
            k[f"kendall/{m}"] = C.fmt(v["tau_b"], 2)
    return k


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--stage2-json", default=str(STAGE2_JSON))
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--tex", default=str(C.TEX))
    ap.add_argument("--no-tex", action="store_true")
    a = ap.parse_args(argv)
    stage2 = json.loads(Path(a.stage2_json).read_text()) if Path(a.stage2_json).exists() else {}
    live = set(json.loads(Path(a.tasks).read_text())["task_ids"][:LIVE_N])
    r = analyze(Path(a.runs), stage2, live)
    if all(r["rows"][x].get("live_n") in (None, 0) for x in ("sl0.8b", "sl2b", "sl9b")):
        print("no E2 live record yet", file=sys.stderr)
    r["generated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    r["git"] = C.git_head()
    r["inputs"] = {"stage2_json": C.sha256_file(Path(a.stage2_json))[:16] if Path(a.stage2_json).exists() else None,
                   "tasks": C.sha256_file(Path(a.tasks))[:16]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(_md(r))
    if not a.no_tex:
        C.write_tex("startlux_live", tex_keys(r), "scripts/explore_startlux_live.py", Path(a.tex))
    print(_md(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
