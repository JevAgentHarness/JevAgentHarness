#!/usr/bin/env python3
r"""The Stage-1 tool guard and stop guard read as approval gates -- DESCRIPTIVE / POST HOC, offline (no model is called).

The question (coverage of decision-model roles, results/explore/small_models/coverage.md §3.2): a guard is a decision
model that approves or vetoes a call the LLM proposed. Read as an approval gate, how many wrong proposals did it veto,
how many right ones did it block, and does its confidence separate the two? The Stage-1 guard arms already logged
every vetted proposal with the oracle's label; this script only re-reads them.

    python scripts/coverage_gates.py
    # reads the local (gitignored) Stage-1 logs named in TOOL_ROWS / STOP_ROWS under runs/;
    # writes results/explore/coverage/gates.{json,md} and paper/generated/coverage.tex (\res{cov/gate/...})

**Records.** Per arm, the latest record of each episode (the runner appends a fresh record when it re-runs an errored
task); errored episodes are skipped. A decision is used only with an oracle label (``gold_status`` ``ok``); doomed or
unlabelled ones are counted as excluded.

* *Tool guard* (``dp`` tool_select, ``mode`` guard): the label is ``proposal_progress`` -- the proposed call (tool and
  arguments) makes progress (META_GOLD_SPEC §3.5; ``gold_aux.verdict``: progress / detour / fatal for a call, progress /
  wrong_answer / premature for a finish). Only first proposals (``retry`` 0) are read: after a veto the LLM proposes
  once more and that proposal runs without a second check and without a label.
* *Stop guard* (``dp`` stop, ``mode`` guard): every proposed finish; the label is ``finish_ok`` (finishing now with this
  answer succeeds; ``gold_aux.kind`` ok / wrong_answer / premature). The *first* finish of an episode (``retry`` 0) is
  the LLM's own, before any block; later ones are resubmissions after a block (reported separately, not independent).

**Quantities.** veto = the gate's logged decision (``p(yes) < tau``); recall = wrong proposals vetoed / wrong; false-veto
rate = right proposals vetoed / right; AUROC of ``p(yes)`` for right vs wrong (``metrics.auroc``; reported only when
each class has at least MIN_CLASS proposals). Tool guard: wrong proposals by verdict and the state-changing calls
(JevGym's ``STATE_CHANGING``: the only calls that can be irreversible). Stop guard: episode outcomes after a caught
wrong first finish (did the agent then succeed?) and after a blocked right one.

**What the logs can and cannot show** is written to the JSON and the Markdown from the counts (e.g. no wrong
state-changing proposal in a decision-model-gated arm means the safety role cannot be tested from these logs)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import coverage_tex as CT  # noqa: E402
from jevharness.bench import metrics as M  # noqa: E402
from jevharness.bench.suites.jevgym import JevGymEnv  # noqa: E402

RUNS = ROOT / "runs"
STATE_CHANGING = frozenset(JevGymEnv.STATE_CHANGING)
MIN_CLASS = 10            # AUROC only when both right and wrong proposals number at least this many
LABEL = "descriptive, post hoc (coverage of decision-model roles); re-reads Stage-1 guard logs; not a registered analysis"

# row id -> (run dir, arm, agent LLM, gate, kind of gate, registered role of the arm)
TOOL_ROWS: dict[str, tuple[str, str, str, str, str, str]] = {
    "flash_jev": ("e2a_jevgym_tool_decisions", "jev_ts_guard", "Flash", "Jev", "decision model", "descriptive arm"),
    "lite_jev": ("e2a_jevgym_tool_decisions", "jev_ts_guard_s2lite", "Flash-Lite", "Jev", "decision model",
                 "descriptive arm"),
    "flash_intern": ("e2a_jevgym_tool_decisions_intern", "intern_ts_guard", "Flash", "Intern-Decision-4B",
                     "decision model", "descriptive arm"),
    "lite_intern": ("e2a_jevgym_tool_decisions_intern", "intern_ts_guard_s2lite", "Flash-Lite", "Intern-Decision-4B",
                    "decision model", "descriptive arm"),
    "flash_lite": ("e2a_jevgym_tool_decisions", "lite_ts_guard", "Flash", "Flash-Lite", "LLM as the gate",
                   "descriptive arm"),
}
STOP_ROWS: dict[str, tuple[str, str, str, str, str, str]] = {
    "flash_jev": ("e2b_jevgym_meta_control", "jev_stop_guard", "Flash", "Jev", "decision model", "C6F"),
    "lite_jev": ("e2b_jevgym_meta_control", "jev_stop_guard_s2lite", "Flash-Lite", "Jev", "decision model", "C6L"),
    "flash_intern": ("e2a_jevgym_tool_decisions_intern", "intern_stop_guard", "Flash", "Intern-Decision-4B",
                     "decision model", "C6F"),
    "lite_intern": ("e2a_jevgym_tool_decisions_intern", "intern_stop_guard_s2lite", "Flash-Lite",
                    "Intern-Decision-4B", "decision model", "C6L"),
    "flash_lite": ("e2b_jevgym_meta_control", "lite_stop_guard", "Flash", "Flash-Lite", "LLM as the gate",
                   "descriptive arm"),
    "lite_random": ("e2c_jevgym_baselines", "ratem_stop_guard_s2lite", "Flash-Lite", "random (Stage-1 control)",
                    "control", "descriptive arm"),
}


# ------------------------------------------------------------------ reading
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_latest(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """task id -> the latest record of that episode; and the number of records read."""
    latest: dict[str, dict[str, Any]] = {}
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                n += 1
                latest[str(r["task_id"])] = r
    return latest, n


def p_yes(d: Mapping[str, Any]) -> float:
    """The gate's probability of *yes* (approve): the logged distribution, else its confidence in its own answer."""
    dist = d.get("s1_dist") or {}
    if isinstance(dist, Mapping) and dist.get("yes") is not None:
        return float(dist["yes"])
    c = d.get("s1_conf")
    c = 0.5 if c is None else float(c)
    return c if bool(d.get("s1_value")) else 1.0 - c


def guard_decisions(traj: Mapping[str, Any], dp: str) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for st in traj.get("steps") or []:
        for d in st.get("decisions") or []:
            if d.get("dp") == dp and d.get("mode") == "guard":
                yield st, d


def success(traj: Mapping[str, Any]) -> bool:
    return bool((traj.get("result") or {}).get("success"))


# ------------------------------------------------------------------ statistics
def gate_stats(items: Sequence[tuple[bool, float, bool]]) -> dict[str, Any]:
    """items = (right?, p_yes, vetoed?) per labelled proposal."""
    right = [x for x in items if x[0]]
    wrong = [x for x in items if not x[0]]
    vw, vr = sum(1 for x in wrong if x[2]), sum(1 for x in right if x[2])
    auc = M.auroc([x[1] for x in items], [float(x[0]) for x in items]) if items else float("nan")
    return {"n": len(items), "right": len(right), "wrong": len(wrong), "vetoed_wrong": vw, "vetoed_right": vr,
            "recall": vw / len(wrong) if wrong else None, "false_veto": vr / len(right) if right else None,
            "auroc": auc if min(len(right), len(wrong)) >= MIN_CLASS else None,
            "auroc_raw": auc if CT.finite(auc) else None}


def tool_row(trajs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    items: list[tuple[bool, float, bool]] = []
    by_verdict: dict[str, Counter[str]] = {}
    sc: Counter[str] = Counter()
    excluded = Counter()
    retries = Counter()
    ok_eps = [t for t in trajs.values() if not t.get("error")]
    for t in ok_eps:
        for _, d in guard_decisions(t, "tool_select"):
            retries[int(d.get("retry") or 0)] += 1
            if int(d.get("retry") or 0) != 0:
                continue
            if d.get("gold_status") != "ok" or d.get("gold") is None:
                excluded[str(d.get("gold_status"))] += 1
                continue
            right, veto = bool(d["gold"]), not bool(d.get("s1_value"))
            items.append((right, p_yes(d), veto))
            tool = str(d.get("s2_value"))
            if not right:
                v = str((d.get("gold_aux") or {}).get("verdict") or "unknown")
                by_verdict.setdefault(v, Counter())["n"] += 1
                by_verdict[v]["vetoed"] += int(veto)
            if tool in STATE_CHANGING:
                sc["n"] += 1
                sc["wrong" if not right else "right"] += 1
                sc["wrong_vetoed" if not right else "right_vetoed"] += int(veto)
    out = gate_stats(items)
    out.update({"episodes": len(ok_eps), "errored": len(trajs) - len(ok_eps),
                "success": sum(success(t) for t in ok_eps) / len(ok_eps) if ok_eps else None,
                "wrong_by_verdict": {k: {"n": int(c["n"]), "vetoed": int(c["vetoed"])}
                                     for k, c in sorted(by_verdict.items())},
                "state_changing": {k: int(sc.get(k, 0)) for k in ("n", "right", "wrong", "wrong_vetoed",
                                                                  "right_vetoed")},
                "excluded": dict(excluded), "retries": {str(k): v for k, v in sorted(retries.items())}})
    return out


def stop_row(trajs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    first: list[tuple[bool, float, bool]] = []
    allp: list[tuple[bool, float, bool]] = []
    kinds_first: dict[str, Counter[str]] = {}
    kinds_all: Counter[str] = Counter()
    excluded = Counter()
    eps = Counter()
    ok_eps = [t for t in trajs.values() if not t.get("error")]
    for t in ok_eps:
        seen_first = False
        for _, d in guard_decisions(t, "stop"):
            if d.get("gold_status") != "ok" or d.get("gold") is None:
                excluded[str(d.get("gold_status"))] += 1
                continue
            right, veto = bool(d["gold"]), not bool(d.get("s1_value"))
            kind = str((d.get("gold_aux") or {}).get("kind") or ("ok" if right else "unknown"))
            x = (right, p_yes(d), veto)
            allp.append(x)
            kinds_all[kind] += 1
            if int(d.get("retry") or 0) == 0 and not seen_first:
                seen_first = True
                first.append(x)
                kinds_first.setdefault(kind, Counter())["n"] += 1
                kinds_first[kind]["vetoed"] += int(veto)
                ok = success(t)
                if not right and veto:
                    eps["caught"] += 1
                    eps["caught_then_success"] += int(ok)
                elif not right:
                    eps["wrong_accepted"] += 1
                elif veto:
                    eps["blocked"] += 1
                    eps["blocked_then_success"] += int(ok)
                else:
                    eps["accepted_right"] += 1
    out = {"first": gate_stats(first), "all": gate_stats(allp),
           "first_by_kind": {k: {"n": int(c["n"]), "vetoed": int(c["vetoed"])} for k, c in sorted(kinds_first.items())},
           "all_by_kind": dict(sorted(kinds_all.items())),
           "episodes_by_first_finish": {k: int(eps.get(k, 0)) for k in
                                        ("accepted_right", "blocked", "blocked_then_success", "wrong_accepted",
                                         "caught", "caught_then_success")},
           "episodes": len(ok_eps), "errored": len(trajs) - len(ok_eps),
           "success": sum(success(t) for t in ok_eps) / len(ok_eps) if ok_eps else None,
           "excluded": dict(excluded)}
    return out


# ------------------------------------------------------------------ reading the limits off the counts
def limits(res: Mapping[str, Any]) -> list[str]:
    L: list[str] = []
    tool, stop = res["tool"], res["stop"]
    dm_tool = {k: r for k, r in tool.items() if r["gate_kind"] == "decision model"}
    sc_wrong = sum(r["state_changing"]["wrong"] for r in dm_tool.values())
    sc_n = sum(r["state_changing"]["n"] for r in dm_tool.values())
    fatal = {k: r["wrong_by_verdict"].get("fatal", {"n": 0, "vetoed": 0}) for k, r in tool.items()}
    if sc_wrong == 0:
        L.append(f"Tool guard: none of the {sc_n:,} state-changing proposals vetted by a decision model was wrong, so "
                 "these logs cannot test the safety role (blocking an irreversible call); only wrong read-only calls "
                 "and wrong finishes were there to catch.")
    else:
        L.append(f"Tool guard: {sc_wrong} of {sc_n:,} state-changing proposals vetted by a decision model were wrong.")
    nonzero = [f"{tool[k]['agent']} → {tool[k]['gate']}: {v['vetoed']} of {v['n']} vetoed"
               for k, v in fatal.items() if v["n"]]
    if nonzero:
        L.append("Fatal proposals (success impossible afterwards) in the logs: " + "; ".join(nonzero) + ".")
    few = [f"{r['agent']} → {r['gate']} ({r['wrong']} wrong of {r['n']:,})" for r in tool.values()
           if r["wrong"] < MIN_CLASS]
    if few:
        L.append("Too few wrong proposals for a recall or an AUROC: " + "; ".join(few) + ".")
    L.append("After a veto the LLM proposes once more and that proposal runs unchecked and unlabelled, so the logs do "
             "not show whether a veto led to a better call.")
    prem = sum(r["all_by_kind"].get("premature", 0) for r in stop.values())
    n_all = sum(r["all"]["n"] for r in stop.values())
    if prem == 0:
        L.append(f"Stop guard: none of the {n_all:,} vetted finishes was premature (work left undone); every wrong "
                 "finish came after the work was done and carried a wrong final answer. So the logs measure catching "
                 "wrong answers, not finishing too early.")
    no_wrong = [f"{r['agent']} → {r['gate']}" for r in stop.values() if r["first"]["wrong"] == 0]
    if no_wrong:
        L.append("No wrong first finish at all (only the cost side, right finishes blocked, is visible): "
                 + "; ".join(no_wrong) + ".")
    L.append("Resubmitted finishes after a block share an episode and a history with the blocked one; only first "
             "finishes are independent across episodes.")
    L.append("All rows are JevGym (short episodes, near the agent LLM's ceiling on Flash); descriptive and post hoc.")
    return L


# ------------------------------------------------------------------ driver
def analyse(runs: Path = RUNS) -> dict[str, Any]:
    res: dict[str, Any] = {"meta": {"generated": CT.now(), "git": CT.git_head(), "label": LABEL, "runs": CT.rel(runs),
                                    "min_class_for_auroc": MIN_CLASS, "state_changing": sorted(STATE_CHANGING),
                                    "files": {}},
                           "tool": {}, "stop": {}, "missing": []}
    for part, rows, fn in (("tool", TOOL_ROWS, tool_row), ("stop", STOP_ROWS, stop_row)):
        for rid, (run, arm, agent, gate, kind, role) in rows.items():
            p = Path(runs) / run / f"{arm}.jsonl"
            if not p.exists():
                res["missing"].append(f"{run}/{arm}")
                continue
            trajs, n = load_latest(p)
            row = fn(trajs)
            row.update({"run": run, "arm": arm, "agent": agent, "gate": gate, "gate_kind": kind, "role": role})
            res[part][rid] = row
            res["meta"]["files"][f"{run}/{arm}"] = {"sha256": sha256(p), "records": n}
    res["limits"] = limits(res)
    return res


def tex_map(res: Mapping[str, Any]) -> dict[str, str]:
    tex: dict[str, str] = {}
    for rid, r in res["tool"].items():
        b = f"cov/gate/tool/{rid}"
        for k in ("n", "right", "wrong", "vetoed_wrong", "vetoed_right"):
            CT.put(tex, f"{b}/{k}", CT.count(r[k]))
        CT.put(tex, f"{b}/recall", CT.pct(r["recall"]) if r["wrong"] >= MIN_CLASS else None)
        CT.put(tex, f"{b}/false_veto", CT.pct(r["false_veto"]))
        CT.put(tex, f"{b}/auroc", CT.two(r["auroc"]))
        for k, v in r["state_changing"].items():
            CT.put(tex, f"{b}/state_changing/{k}", CT.count(v))
        for v, c in r["wrong_by_verdict"].items():
            CT.put(tex, f"{b}/verdict/{v}/n", CT.count(c["n"]))
            CT.put(tex, f"{b}/verdict/{v}/vetoed", CT.count(c["vetoed"]))
    dm = [r for r in res["tool"].values() if r["gate_kind"] == "decision model"]
    CT.put(tex, "cov/gate/tool/dm/state_changing/n", CT.count(sum(r["state_changing"]["n"] for r in dm)))
    CT.put(tex, "cov/gate/tool/dm/state_changing/wrong", CT.count(sum(r["state_changing"]["wrong"] for r in dm)))
    for rid, r in res["stop"].items():
        b = f"cov/gate/stop/{rid}"
        for part in ("first", "all"):
            s = r[part]
            for k in ("n", "right", "wrong", "vetoed_wrong", "vetoed_right"):
                CT.put(tex, f"{b}/{part}/{k}", CT.count(s[k]))
            CT.put(tex, f"{b}/{part}/recall", CT.pct(s["recall"]) if s["wrong"] >= MIN_CLASS else None)
            CT.put(tex, f"{b}/{part}/false_veto", CT.pct(s["false_veto"]))
            CT.put(tex, f"{b}/{part}/auroc", CT.two(s["auroc"]))
        for k, v in r["episodes_by_first_finish"].items():
            CT.put(tex, f"{b}/episodes/{k}", CT.count(v))
        CT.put(tex, f"{b}/premature", CT.count(r["all_by_kind"].get("premature", 0)))
    CT.put(tex, "cov/gate/stop/premature", CT.count(sum(r["all_by_kind"].get("premature", 0)
                                                        for r in res["stop"].values())))
    CT.put(tex, "cov/gate/stop/n_all", CT.count(sum(r["all"]["n"] for r in res["stop"].values())))
    return tex


def _cell(x: Any, kind: str = "pct") -> str:
    v = CT.pct(x) if kind == "pct" else CT.two(x)
    return v if v is not None else "–"


def to_markdown(res: Mapping[str, Any]) -> str:
    m = res["meta"]
    L = ["# The Stage-1 guards read as approval gates (descriptive, post hoc)", "",
         f"Generated {m['generated']} at git `{str(m['git'])[:12]}` by `scripts/coverage_gates.py` from the local "
         f"Stage-1 logs under `{m['runs']}/`. {LABEL}. AUROC only with at least {m['min_class_for_auroc']} right and "
         f"{m['min_class_for_auroc']} wrong proposals; recall only with at least {m['min_class_for_auroc']} wrong.", "",
         "## Tool guard (first proposals; label: the proposed call makes progress)", "",
         "| agent LLM → gate | arm | episodes | proposals | wrong | vetoed wrong (recall) | vetoed right (false-veto "
         "rate) | AUROC | wrong by verdict (vetoed) | state-changing: n / wrong / right vetoed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["tool"].values():
        wv = ", ".join(f"{k} {c['n']} ({c['vetoed']})" for k, c in r["wrong_by_verdict"].items()) or "–"
        sc = r["state_changing"]
        rec = f" ({_cell(r['recall'])} %)" if r["wrong"] >= MIN_CLASS else ""
        L.append(f"| {r['agent']} → {r['gate']} | `{r['run']}/{r['arm']}` | {r['episodes']} | {r['n']:,} | "
                 f"{r['wrong']} | {r['vetoed_wrong']} / {r['wrong']}{rec} | {r['vetoed_right']} / {r['right']:,} "
                 f"({_cell(r['false_veto'])} %) | {_cell(r['auroc'], 'two')} | {wv} | "
                 f"{sc['n']} / {sc['wrong']} / {sc['right_vetoed']} |")
    L += ["", "## Stop guard (label: finishing now with this answer succeeds)", "",
          "First finish of each episode (the LLM's own, before any block):", "",
          "| agent LLM → gate | arm (role) | episodes | first finishes | wrong (kinds) | caught (recall) | right "
          "blocked (false-block rate) | AUROC | caught → episode succeeded | blocked → episode succeeded |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["stop"].values():
        f, e = r["first"], r["episodes_by_first_finish"]
        kinds = ", ".join(f"{k} {c['n']}" for k, c in r["first_by_kind"].items() if k != "ok") or "–"
        rec = f" ({_cell(f['recall'])} %)" if f["wrong"] >= MIN_CLASS else ""
        L.append(f"| {r['agent']} → {r['gate']} | `{r['run']}/{r['arm']}` ({r['role']}) | {r['episodes']} | {f['n']} | "
                 f"{f['wrong']} ({kinds}) | {f['vetoed_wrong']} / {f['wrong']}{rec} | {f['vetoed_right']} / "
                 f"{f['right']} ({_cell(f['false_veto'])} %) | {_cell(f['auroc'], 'two')} | "
                 f"{e['caught_then_success']} / {e['caught']} | {e['blocked_then_success']} / {e['blocked']} |")
    L += ["", "All finishes, resubmissions after a block included (not independent):", "",
          "| agent LLM → gate | finishes | wrong | caught | right blocked | AUROC | kinds |", "|---|---|---|---|---|---|---|"]
    for r in res["stop"].values():
        a = r["all"]
        kinds = ", ".join(f"{k} {n}" for k, n in r["all_by_kind"].items())
        L.append(f"| {r['agent']} → {r['gate']} | {a['n']:,} | {a['wrong']} | {a['vetoed_wrong']} | "
                 f"{a['vetoed_right']} / {a['right']:,} ({_cell(a['false_veto'])} %) | {_cell(a['auroc'], 'two')} | "
                 f"{kinds} |")
    L += ["", "## What these logs can and cannot show", ""] + [f"- {x}" for x in res["limits"]]
    if res["missing"]:
        L += ["", "Missing arms (not read): " + ", ".join(f"`{x}`" for x in res["missing"])]
    L += ["", "## Sources", ""] + [f"- `{k}`: {v['records']} records, sha256 `{v['sha256'][:16]}`"
                                   for k, v in m["files"].items()] + [""]
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out-dir", default=str(CT.OUT_DIR))
    ap.add_argument("--tex", default=str(CT.TEX_OUT))
    a = ap.parse_args(argv)
    res = analyse(Path(a.runs))
    if res["missing"]:
        print(f"ERROR: missing Stage-1 logs: {res['missing']} (they are local and gitignored)", file=sys.stderr)
        return 1
    res["tex"] = tex_map(res)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "gates.md").write_text(to_markdown(res))
    tex = CT.write_tex(out, Path(a.tex))
    for rid, r in res["tool"].items():
        print(f"tool {rid}: {r['n']} proposals, {r['wrong']} wrong, vetoed {r['vetoed_wrong']} wrong / "
              f"{r['vetoed_right']} right, AUROC {CT.two(r['auroc'])}")
    for rid, r in res["stop"].items():
        f = r["first"]
        print(f"stop {rid}: {f['n']} first finishes, {f['wrong']} wrong, caught {f['vetoed_wrong']}, "
              f"blocked {f['vetoed_right']} right, AUROC {CT.two(f['auroc'])}")
    print(f"wrote {CT.rel(out / 'gates.json')}, {CT.rel(out / 'gates.md')}, {CT.rel(tex)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
