#!/usr/bin/env python3
r"""Three logged episodes for the appendix's Case Studies (App. app:cases) -> paper/generated/cases.tex.

Post hoc and descriptive: stored logs only (no API call, no GPU, no new episode); revises no result. Every tool,
action, argument, confidence, threshold, outcome, cost and time drawn in paper/figures/fig_cases.tex is a
``\defres{cases/...}`` key of the generated file, so nothing in the figures is typed by hand.

    python scripts/case_studies.py            # writes paper/generated/cases.tex and results/explore/case_studies/cases.json
    python scripts/case_studies.py --check    # rebuilds in memory and exits 1 if the committed file differs

Cases and their selection rules (fixed here; each takes the first qualifying task in the stated order, so nothing is
picked by hand):

1. ``take`` -- where the decision model takes over. AgentDecision-4B (E12's checkpoint) behind its fitted tool and
   argument thresholds in our own agent loop on its own GPU (runs/e21b_e12_in, the Table-1 row "own GPU"), paired with
   Gemini 3.5 Flash alone on the same 200 fresh tasks (runs/harness_final_dedicated/llm_only, the pairing of
   scripts/explore_e21b_shift_cascade.py). Rule: tasks in seed order; the first that both runs solved and on which a
   threshold handed at least one choice to the LLM (a deferred cascade decision).
2. ``ask`` -- a confident irreversible choice against asking. Jev and AgentDecision-4B each pick every tool (REPLACE
   at the tool choice, Flash fills the arguments) on the same 200 fresh tasks (runs/stage2_rl_states/jev_ts_replace_f,
   runs/e12_live/e12_ts_replace_f; the rows of App. app:drift), with Flash alone on the same task
   (runs/stage2_jevgym/llm_only). Rule: tasks in seed order; the first where Jev's first wrong tool choice
   (scripts/explore_drift.py's definition: the first labelled step whose executed tool is outside the logged acceptable
   set) was an irreversible write (explore_drift.consequence) at a must-ask state (asking is the only acceptable tool),
   Jev failed the task and AgentDecision-4B solved it.
3. ``alf`` -- ALFWorld. E11's trained 4B behind its fitted threshold at the action choice
   (runs/e11_alfworld_eval/e11_action_cascade_tauhat) against Flash alone in the same run (llm_only). Rule: games in
   game-id order; the first that both runs solved.

Cost conventions are the papers' own: the E12 run adds the hourly GPU price of App. app:harnessfinal
(explore_e21b_shift_cascade.TARIFF_USD_H) times the episode's wall-clock time; the E11 run adds E11's per-episode GPU
tariff (results/explore/e11_alfworld/summary.json). LLM calls are the logged ledger entries of the LLM
(``component == "s2"``). Step numbers are 1-based (log step + 1). Gold labels (the acceptable tool sets) are read only
to mark wrong choices and must-ask states in the figure; they never reach a decision model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUNS = ROOT / "runs"
OUT_TEX = ROOT / "paper/generated/cases.tex"
OUT_JSON = ROOT / "results/explore/case_studies/cases.json"
PREFIX = "cases"
LABEL = "post hoc, descriptive; stored logs only (no API call, no GPU); revises no result"

TAKE_RUN = ("e21b_e12_in", "e12_final_in")
TAKE_REF = ("harness_final_dedicated", "llm_only")
ASK_A = ("stage2_rl_states", "jev_ts_replace_f")
ASK_B = ("e12_live", "e12_ts_replace_f")
ASK_REF = ("stage2_jevgym", "llm_only")
ALF_RUN = ("e11_alfworld_eval", "e11_action_cascade_tauhat")
ALF_REF = ("e11_alfworld_eval", "llm_only")
E11_SUMMARY = ROOT / "results/explore/e11_alfworld/summary.json"

ASK_TOOL = "ask_user"
FREE_TEXT_ARGS = {("ask_user", "question"), ("finish", "answer")}   # the gym's free-text arguments (the LLM writes them)
SEED_RE = re.compile(r"-(\d{6})-")
ARG_RE = re.compile(r"argument `([^`]+)`")
CALL_ARGS_RE = re.compile(r"\(\{[^()]*\}\)")

# per case: layout of the figure (fig_cases.tex) and how much text fits on one line (characters, 7 pt)
LAYOUT = {
    "take": {"alt_own_line": False, "free_chars": 34, "obs_chars": 70, "obs_lines": 1},
    "ask": {"alt_own_line": True, "free_chars": 18, "obs_chars": 58, "obs_lines": 1},
    "alf": {"alt_own_line": True, "free_chars": 0, "obs_chars": 58, "obs_lines": 2},
}
PITCH = 3.6        # mm per text line in a step row
TOOL_LINE = 4.6    # mm for the line with the tool box
ROW_PAD = 1.6      # mm between rows


# ================================================================== reading
def run_path(spec: tuple[str, str], runs: Path = RUNS) -> Path:
    return runs / spec[0] / f"{spec[1]}.jsonl"


def latest(path: Path) -> dict[str, dict[str, Any]]:
    """task id -> its deciding record (the convention of every readout: explore_drift.latest)."""
    from explore_drift import latest as _latest

    return _latest(path)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def seed_of(task_id: str) -> int:
    m = SEED_RE.search(task_id)
    return int(m.group(1)) if m else sys.maxsize


def ok(rec: Mapping[str, Any] | None) -> bool:
    return bool(rec) and not rec.get("error")


def solved(rec: Mapping[str, Any] | None) -> bool:
    return ok(rec) and bool((rec.get("result") or {}).get("success"))


def write_tools() -> set[str]:
    """The tools that change the world (explore_drift.write_tools over benchmark_difficulty.near_map)."""
    from benchmark_difficulty import near_map
    from explore_drift import write_tools as _wt

    return _wt(near_map())


# ================================================================== formatting
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s: Any) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def abridge(text: str, n: int) -> tuple[str, bool]:
    """The first line of ``text``, cut at a word boundary to at most ``n`` characters (+ an ellipsis)."""
    t = " ".join(str(text).split("\n")[0].split())
    if len(t) <= n:
        return t, False
    cut = t[:n].rsplit(" ", 1)[0].rstrip(",;:.")
    return (cut or t[:n]) + "…", True


def fp(x: float | None, nd: int = 2) -> str:
    """A probability for print; values that would print as 0 show as <0.01."""
    if x is None:
        return ""
    if round(x, nd) == 0 and x > 0:
        return "$<$0." + "0" * (nd - 1) + "1"
    return f"{x:.{nd}f}"


def money(x: float) -> str:
    return f"{x:.4f}"


# ================================================================== one step of a logged episode
def runner_up(d: Mapping[str, Any]) -> tuple[str | None, float | None]:
    """The decision model's second-ranked option (ties: the order in which options were offered)."""
    dist = d.get("s1_dist") or {}
    if not dist:
        return None, None
    order = {o: i for i, o in enumerate(d.get("options") or list(dist))}
    ranked = sorted(dist.items(), key=lambda kv: (-float(kv[1]), order.get(kv[0], len(order))))
    chosen = d.get("s1_value")
    for name, p in ranked:
        if name != chosen:
            return str(name), float(p)
    return None, None


def first_decision(step: Mapping[str, Any], dp: str) -> Mapping[str, Any] | None:
    return next((d for d in step.get("decisions") or [] if d.get("dp") == dp and not d.get("shadow")), None)


def observation(step: Mapping[str, Any], writes: set[str], case: str) -> tuple[str, str]:
    """The environment's reply shown under a step: the customer's answer to a question and the result of a world
    change on DecisionGym (lookups return long records and are left out), every observation on ALFWorld."""
    obs = str(step.get("observation") or "").strip()
    if not obs:
        return "", ""
    if case == "alf":
        return obs, "env"
    if step.get("tool") == ASK_TOOL:
        return re.sub(r"^Customer:\s*", "", obs), "user"   # drawn with a person glyph instead of the prefix
    if step.get("tool") in writes:
        return obs, "err" if obs.lower().startswith("irreversible error") else "env"
    return "", ""


def step_row(step: Mapping[str, Any], writes: set[str], case: str, point: str) -> dict[str, Any]:
    """Who chose the tool (or action) and every argument of one logged step, with the decision model's confidence."""
    from jevharness.bench import metrics as M

    lay = LAYOUT[case]
    d = first_decision(step, point)
    tool = step.get("tool") if point == "tool_select" else (step.get("args") or {}).get("action")
    by_dm = d is not None and d.get("mode") in ("cascade", "replace") and not d.get("deferred")
    row: dict[str, Any] = {"step": int(step.get("step", 0)) + 1, "tool": tool, "by": "dm" if by_dm else "llm",
                           "llm_calls": int(step.get("llm_calls") or 0),
                           "conf": None, "alt": None, "altp": None, "ok": None, "must": False,
                           "irr": point == "tool_select" and tool in writes, "m": [], "l": []}
    if d is not None and d.get("s1_conf") is not None and d.get("mode") in ("cascade", "replace"):
        row["conf"] = float(d["s1_conf"])
        row["alt"], row["altp"] = runner_up(d)
    if d is not None and point == "tool_select":
        row["ok"] = M.is_correct(d, tool) if M.has_gold(d) else None
        row["must"] = M.acceptable_of(d) == [ASK_TOOL]
    # arguments: the decision model's (cascade, at or above the threshold), else the LLM's
    params = [x for x in step.get("decisions") or [] if x.get("dp") == "tool_params" and not x.get("shadow")]
    if params:
        for x in params:
            m = ARG_RE.search(str(x.get("question") or ""))
            name = m.group(1) if m else "?"
            val = x.get("final_value")
            if x.get("mode") == "generate":
                txt, _ = abridge(str(val), lay["free_chars"])
                row["l"].append({"k": name, "v": txt, "c": None, "free": True})
            elif x.get("deferred"):
                row["l"].append({"k": name, "v": str(val), "c": x.get("s1_conf"), "free": False})
            else:
                row["m"].append({"k": name, "v": str(val), "c": x.get("s1_conf")})
    elif point == "tool_select":            # REPLACE runs and the LLM alone: the LLM wrote every argument
        for name, val in (step.get("args") or {}).items():
            free = (tool, name) in FREE_TEXT_ARGS
            txt = abridge(str(val), lay["free_chars"])[0] if free else str(val)
            row["l"].append({"k": name, "v": txt, "c": None, "free": free})
    obs, kind = observation(step, writes, case)
    obs = CALL_ARGS_RE.sub("(…)", obs)          # a logged call's argument record is abridged to (…)
    row["obs"], row["obsk"] = (abridge(obs, lay["obs_chars"] * lay["obs_lines"])[0], kind) if obs else ("", "")
    row["obs_lines"] = min(lay["obs_lines"], -(-len(row["obs"]) // lay["obs_chars"])) if row["obs"] else 0
    row["obs_lines"] = max(row["obs_lines"], 1) if row["obs"] else 0
    return row


def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Who decided what in one episode: tools, arguments with options (taken or handed to the LLM), free text."""
    return {"tools": len(rows), "tools_dm": sum(1 for r in rows if r["by"] == "dm"),
            "args_dm": sum(len(r["m"]) for r in rows),
            "args_deferred": sum(1 for r in rows for x in r["l"] if not x["free"] and x["c"] is not None),
            "free": sum(1 for r in rows for x in r["l"] if x["free"]),
            "args": sum(len(r["m"]) for r in rows) + sum(1 for r in rows for x in r["l"] if not x["free"])}


def lines_of(row: Mapping[str, Any], case: str) -> tuple[int, int]:
    """Text lines a step needs in the decision-model lane and in the LLM lane (one item per line)."""
    lay = LAYOUT[case]
    if row["by"] == "dm":
        dm = 1 + (1 if lay["alt_own_line"] and row.get("alt") else 0) + (1 if row.get("must") else 0) + len(row["m"])
        return dm, len(row["l"])
    return 0, 1 + len(row["l"])


def episode_numbers(rec: Mapping[str, Any], gpu_usd_per_s: float = 0.0, gpu_per_episode: float = 0.0) -> dict[str, Any]:
    calls = rec.get("calls") or []
    wall = float(rec.get("wall_ms") or 0.0) / 1000
    api = float(((rec.get("usage") or {}).get("total") or {}).get("cost_usd") or 0.0)
    gpu = gpu_usd_per_s * wall + gpu_per_episode
    return {"success": solved(rec), "steps": len(rec.get("steps") or []),
            "llm_calls": sum(1 for c in calls if c.get("component") == "s2"),
            "dm_calls": sum(1 for c in calls if c.get("component") == "s1"),
            "api_usd": api, "gpu_usd": gpu, "cost_usd": api + gpu, "wall_s": wall}


# ================================================================== the three rules
def select_take(run: Mapping[str, Mapping[str, Any]], ref: Mapping[str, Mapping[str, Any]]) -> tuple[str, int]:
    """Seed order; the first task both runs solved on which a threshold handed at least one choice to the LLM."""
    for i, t in enumerate(sorted(run, key=lambda k: (seed_of(k), k))):
        r = run[t]
        if not (solved(r) and solved(ref.get(t))):
            continue
        if any(d.get("mode") == "cascade" and d.get("deferred") and not d.get("shadow")
               for s in r.get("steps") or [] for d in s.get("decisions") or []):
            return t, i + 1
    raise SystemExit("[case_studies] take: no task satisfies the rule")


def first_wrong(rec: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    """explore_drift's first error: the first labelled tool-choice step whose executed tool is not acceptable."""
    from jevharness.bench import metrics as M

    for s in rec.get("steps") or []:
        d = first_decision(s, "tool_select")
        if d is None or not M.has_gold(d):
            continue
        if M.is_correct(d, s.get("tool")) is False:
            return s, d
    return None


def select_ask(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]], writes: set[str]) -> tuple[str, int]:
    """Seed order; the first task where model A's first wrong tool choice was an irreversible write at a must-ask
    state, A failed and B solved the task."""
    from explore_drift import consequence
    from jevharness.bench import metrics as M

    for i, t in enumerate(sorted(a, key=lambda k: (seed_of(k), k))):
        ra = a[t]
        if not ok(ra) or solved(ra) or not solved(b.get(t)):
            continue
        fw = first_wrong(ra)
        if fw is None:
            continue
        s, d = fw
        if consequence(s.get("tool"), writes) == "wrong write action" and M.acceptable_of(d) == [ASK_TOOL]:
            return t, i + 1
    raise SystemExit("[case_studies] ask: no task satisfies the rule")


def select_alf(run: Mapping[str, Mapping[str, Any]], ref: Mapping[str, Mapping[str, Any]]) -> tuple[str, int]:
    """Game-id order; the first game both runs solved."""
    for i, t in enumerate(sorted(run)):
        if solved(run[t]) and solved(ref.get(t)):
            return t, i + 1
    raise SystemExit("[case_studies] alf: no game satisfies the rule")


# ================================================================== build
def task_text(rec: Mapping[str, Any]) -> str:
    for s in rec.get("steps") or []:
        for d in s.get("decisions") or []:
            m = re.search(r"TASK:\s*(?:Your task is to:\s*)?(.+)", str(d.get("state") or ""))
            if m:
                return m.group(1).strip()
    return ""


def first_ask(rec: Mapping[str, Any]) -> int | None:
    return next((i + 1 for i, s in enumerate(rec.get("steps") or []) if s.get("tool") == ASK_TOOL), None)


def logged_tau(rec: Mapping[str, Any], point: str) -> float | None:
    t = (((rec.get("meta") or {}).get("config") or {}).get(point) or {}).get("tau")
    return None if t is None else float(t)


def build(runs: Path = RUNS) -> dict[str, Any]:
    import explore_e21b_shift_cascade as E21B

    writes = write_tools()
    out: dict[str, Any] = {"label": LABEL, "cases": {}}

    # ---- case 1: take
    p_run, p_ref = run_path(TAKE_RUN, runs), run_path(TAKE_REF, runs)
    run, ref = latest(p_run), latest(p_ref)
    t, rank = select_take(run, ref)
    rec = run[t]
    prof = json.loads((runs / TAKE_RUN[0] / "profiles.json").read_text())[TAKE_RUN[1]]["decision_points"]
    tau_tool, tau_arg = float(prof["tool_select"]["tau"]), float(prof["tool_params"]["tau"])
    assert abs(tau_tool - logged_tau(rec, "tool_select")) < 1e-9 and abs(tau_arg - logged_tau(rec, "tool_params")) < 1e-9
    gen = {(s.get("tool"), ARG_RE.search(str(d.get("question"))).group(1))
           for s in rec.get("steps") or [] for d in s.get("decisions") or []
           if d.get("dp") == "tool_params" and d.get("mode") == "generate" and ARG_RE.search(str(d.get("question")))}
    assert gen <= FREE_TEXT_ARGS, gen
    out["cases"]["take"] = {
        "task_id": t, "rank": rank, "n_tasks": len(run), "task": task_text(rec), "seed": seed_of(t),
        "tau_tool": tau_tool, "tau_arg": tau_arg,
        "sources": {"run": str(p_run.relative_to(ROOT)), "run_sha16": sha16(p_run),
                    "ref": str(p_ref.relative_to(ROOT)), "ref_sha16": sha16(p_ref)},
        "runs": {"run": {"rows": [step_row(s, writes, "take", "tool_select") for s in rec["steps"]],
                         **episode_numbers(rec, gpu_usd_per_s=E21B.TARIFF_USD_H / 3600)},
                 "ref": {"rows": None, **episode_numbers(ref[t])}},
        "tariff_usd_h": E21B.TARIFF_USD_H}

    # ---- case 2: ask
    p_a, p_b, p_r = run_path(ASK_A, runs), run_path(ASK_B, runs), run_path(ASK_REF, runs)
    a, b, r = latest(p_a), latest(p_b), latest(p_r)
    t, rank = select_ask(a, b, writes)
    fw = first_wrong(a[t])
    out["cases"]["ask"] = {
        "task_id": t, "rank": rank, "n_tasks": len(a), "task": task_text(a[t]), "seed": seed_of(t),
        "first_wrong_step": int(fw[0]["step"]) + 1 if fw else None,
        "sources": {"jev": str(p_a.relative_to(ROOT)), "jev_sha16": sha16(p_a), "ours": str(p_b.relative_to(ROOT)),
                    "ours_sha16": sha16(p_b), "ref": str(p_r.relative_to(ROOT)), "ref_sha16": sha16(p_r)},
        "runs": {"jev": {"rows": [step_row(s, writes, "ask", "tool_select") for s in a[t]["steps"]],
                         **episode_numbers(a[t])},
                 "ours": {"rows": [step_row(s, writes, "ask", "tool_select") for s in b[t]["steps"]],
                          **episode_numbers(b[t])},
                 "ref": {"rows": None, **(episode_numbers(r[t]) if t in r else {}), "first_ask": first_ask(r.get(t) or {})}}}

    # ---- case 3: alf
    p_run, p_ref = run_path(ALF_RUN, runs), run_path(ALF_REF, runs)
    run, ref = latest(p_run), latest(p_ref)
    t, rank = select_alf(run, ref)
    tariff = json.loads(E11_SUMMARY.read_text())["tariff"]
    q = float(tariff["q_usd_per_episode"])
    tau = logged_tau(run[t], "action")
    out["cases"]["alf"] = {
        "task_id": t, "rank": rank, "n_tasks": len(run), "task": task_text(run[t]), "tau": tau,
        "sources": {"run": str(p_run.relative_to(ROOT)), "run_sha16": sha16(p_run),
                    "ref": str(p_ref.relative_to(ROOT)), "ref_sha16": sha16(p_ref),
                    "tariff": str(E11_SUMMARY.relative_to(ROOT))},
        "runs": {"run": {"rows": [step_row(s, writes, "alf", "action") for s in run[t]["steps"]],
                         **episode_numbers(run[t], gpu_per_episode=q)},
                 "ref": {"rows": [step_row(s, writes, "alf", "action") for s in ref[t]["steps"]],
                         **episode_numbers(ref[t])}},
        "q_usd_per_episode": q}
    for name, c in out["cases"].items():
        c["layout"] = layout(c, name)
    return out


def layout(case: Mapping[str, Any], name: str) -> list[dict[str, float]]:
    """Row tops and heights (mm, downward) for the figure, aligned across the runs drawn side by side."""
    drawn = [v["rows"] for v in case["runs"].values() if v.get("rows")]
    n = max(len(rows) for rows in drawn)
    out, y = [], 0.0
    for i in range(n):
        need, obs = 1, 0
        for rows in drawn:
            if i < len(rows):
                dm, llm = lines_of(rows[i], name)
                need = max(need, dm, llm)
                obs = max(obs, rows[i]["obs_lines"])
        h = TOOL_LINE + (need - 1) * PITCH + obs * PITCH + ROW_PAD
        out.append({"y": round(-y, 2), "h": round(h, 2), "obs": obs})
        y += h
    return out


# ================================================================== TeX
def keys(res: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {}

    def put(key: str, val: Any) -> None:
        k[f"{PREFIX}/{key}"] = val if isinstance(val, str) else str(val)

    for cname, c in res["cases"].items():
        put(f"{cname}/task", tex(c["task"]))
        put(f"{cname}/rank", c["rank"])
        put(f"{cname}/n_tasks", c["n_tasks"])
        put(f"{cname}/n_rows", len(c["layout"]))
        for i, row in enumerate(c["layout"], 1):
            put(f"{cname}/row/{i}/y", f"{row['y']:.2f}")
            put(f"{cname}/row/{i}/h", f"{row['h']:.2f}")
            put(f"{cname}/row/{i}/obs", row["obs"])
        put(f"{cname}/height", f"{-c['layout'][-1]['y'] + c['layout'][-1]['h']:.2f}")
        if "seed" in c:
            put(f"{cname}/seed", c["seed"])
        for rname, r in c["runs"].items():
            p = f"{cname}/{rname}"
            for x in ("steps", "llm_calls"):
                if x in r:
                    put(f"{p}/{x}", r[x])
            if "success" in r:
                put(f"{p}/success", int(bool(r["success"])))
            for x in ("cost_usd", "api_usd", "gpu_usd"):
                if x in r:
                    put(f"{p}/{x.split('_')[0]}", money(r[x]))
            if "wall_s" in r:
                put(f"{p}/wall", f"{r['wall_s']:.1f}")
            if r.get("first_ask") is not None:
                put(f"{p}/first_ask", r["first_ask"])
            if not r.get("rows"):
                continue
            put(f"{p}/n", len(r["rows"]))
            for x, v in counts(r["rows"]).items():
                put(f"{p}/count/{x}", v)
            for i, row in enumerate(r["rows"], 1):
                s = f"{p}/s/{i}"
                put(f"{s}/tool", tex(row["tool"]))
                put(f"{s}/by", row["by"])
                put(f"{s}/llm", row["llm_calls"])
                put(f"{s}/conf", fp(row["conf"]))
                put(f"{s}/cnum", "-1" if row["conf"] is None else f"{row['conf']:.4f}")
                put(f"{s}/alt", tex(row["alt"]) if row["alt"] else "")
                put(f"{s}/altp", fp(row["altp"]))
                put(f"{s}/ok", "" if row["ok"] is None else str(int(bool(row["ok"]))))
                put(f"{s}/irr", int(bool(row["irr"])))
                put(f"{s}/must", int(bool(row["must"])))
                put(f"{s}/nm", len(row["m"]))
                for j, m in enumerate(row["m"], 1):
                    put(f"{s}/m/{j}/k", tex(m["k"]))
                    put(f"{s}/m/{j}/v", tex(m["v"]))
                    put(f"{s}/m/{j}/c", fp(m["c"], 3))
                put(f"{s}/nl", len(row["l"]))
                for j, m in enumerate(row["l"], 1):
                    put(f"{s}/l/{j}/k", tex(m["k"]))
                    put(f"{s}/l/{j}/v", tex(m["v"]))
                    put(f"{s}/l/{j}/c", fp(m["c"], 3) if m["c"] is not None else "")
                    put(f"{s}/l/{j}/free", int(bool(m["free"])))
                put(f"{s}/obs", tex(row["obs"]))
                put(f"{s}/obsk", row["obsk"])
                put(f"{s}/obsn", row["obs_lines"])
    c = res["cases"]
    put("take/tau_tool", f"{c['take']['tau_tool']:.2f}")
    put("take/tau_tool_num", f"{c['take']['tau_tool']:.4f}")
    put("take/tau_arg", f"{c['take']['tau_arg']:.3f}")
    put("take/tariff", f"{c['take']['tariff_usd_h']:.3f}")
    put("take/saved_calls", c["take"]["runs"]["ref"]["llm_calls"] - c["take"]["runs"]["run"]["llm_calls"])
    put("ask/first_wrong", c["ask"]["first_wrong_step"])
    put("ask/first_wrong_conf", fp(c["ask"]["runs"]["jev"]["rows"][c["ask"]["first_wrong_step"] - 1]["conf"]))
    put("alf/tau", f"{c['alf']['tau']:.2f}")
    put("alf/tau_num", f"{c['alf']['tau']:.4f}")
    return k


def to_tex(res: Mapping[str, Any]) -> str:
    c = res["cases"]
    head = ["% AUTO-GENERATED by scripts/case_studies.py -- do not edit by hand (rules: the script's docstring).",
            f"% {LABEL}.",
            *(f"% {n}: {v['task_id']} (rule rank {v['rank']} of {v['n_tasks']}); sources "
              + ", ".join(f"{s} {v['sources'].get(s + '_sha16', '')}".strip() for s in v["sources"] if not s.endswith("_sha16"))
              for n, v in c.items()),
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = [rf"\defres{{{key}}}{{{val}}}%" for key, val in sorted(keys(res).items())]   # % : safe in horizontal mode
    return "\n".join(head + body) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--check", action="store_true", help="rebuild in memory; exit 1 if paper/generated/cases.tex differs")
    a = ap.parse_args(argv)
    res = build(Path(a.runs))
    txt = to_tex(res)
    if a.check:
        same = OUT_TEX.exists() and OUT_TEX.read_text() == txt
        print("cases.tex up to date" if same else "cases.tex differs from the logs")
        return 0 if same else 1
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(txt)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    rec = {**res, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    OUT_JSON.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    for n, v in res["cases"].items():
        print(f"{n}: {v['task_id']} (rank {v['rank']} of {v['n_tasks']})")
    print(f"wrote {OUT_TEX.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
