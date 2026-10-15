"""Per-arm analysis of the dev-modes row: success, cost, time, strong share (bench/summarize.py), decision calls,
decision-model cost, how often each point acted, guard outcomes, and per-harness `jah fit` of the shadow arms.

    python bench/analyze_modes.py <row dir> [--json out.json] [--md out.md] [--base-row <older row dir> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "bench"))
import summarize as S  # noqa: E402
from jevagentharness import fit as F  # noqa: E402


def harness_of(jobs: Path) -> dict[str, str]:
    """session key (model_name/trial) -> harness."""
    out = {}
    for job in jobs.iterdir():
        if not job.is_dir():
            continue
        h, _, arm = job.name.partition("-")
        for t in job.iterdir():
            if t.is_dir():
                out[f"{arm}/{S.safe(t.name)}"] = h
    return out


def load_jsonl(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for ln in p.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def analyze(row: Path) -> dict:
    jobs, proxy = row / "jobs", row / "proxy"
    trials = S.join(S.load_trials(jobs), S.load_steps(proxy))
    cmp = S.compare(trials, "jah-strong")
    hmap = harness_of(jobs)
    steps = [s for s in load_jsonl(proxy / "steps.jsonl") if s.get("kind") != "cache"]
    decs = load_jsonl(proxy / "decisions.jsonl")
    sess_ok = {f"{t['arm']}/{S.safe(t['trial'])}" for t in trials}
    by_arm_steps: dict[str, list] = defaultdict(list)
    for s in steps:
        if s["session"] in sess_ok:
            by_arm_steps[f"{hmap.get(s['session'], '?')}:{s['model_name']}"].append(s)
    by_arm_decs: dict[str, list] = defaultdict(list)
    for d in decs:
        if d["session"] in sess_ok:
            by_arm_decs[f"{hmap.get(d['session'], '?')}:{d['model_name']}"].append(d)
    out = {}
    for key, row_ in cmp.items():
        n = row_["n"]
        ss, ds = by_arm_steps.get(key, []), by_arm_decs.get(key, [])
        pts: dict[str, dict] = {}
        for p in ("routing", "tool_select", "stop", "reflection", "action"):
            rs = [d for d in ds if d["point"] == p]
            if not rs:
                continue
            modes = Counter(d["mode"] for d in rs)
            ok = [d for d in rs if not d.get("error")]
            fire = [d for d in ok if d.get("p_act") is not None and d.get("tau") is not None
                    and d["p_act"] >= d["tau"]]
            e = {"modes": dict(modes), "calls": len(rs), "calls_per_task": round(len(rs) / n, 2),
                 "errors": len(rs) - len(ok), "acted": sum(1 for d in rs if d.get("acted")),
                 "acted_per_task": round(sum(1 for d in rs if d.get("acted")) / n, 2),
                 "effects": dict(Counter(str(d.get("effect")) for d in rs).most_common()),
                 "cost_usd": round(sum(float(d.get("cost_usd") or 0) for d in rs), 5)}
            if ok and rs[0].get("act_option"):
                e["fires_at_tau"] = round(len(fire) / len(ok), 4)
            if p == "action":
                rc = [d for d in rs if d.get("recheck")]
                e["rechecks"] = dict(Counter(d["recheck"] for d in rc))
                e["recheck_still_risky"] = sum(1 for d in rc if d.get("recheck_risky"))
                e["prefilter_steps"] = len({(d["session"], d["step"]) for d in rs})
            if p == "stop":
                e["labels_by_effect"] = {eff: dict(Counter(str(d.get("label")) for d in rs if d.get("effect") == eff))
                                         for eff in {str(d.get("effect")) for d in rs}}
            if p == "tool_select":
                e["choices"] = dict(Counter(d.get("choice") for d in ok).most_common(8))
                e["labels"] = dict(Counter(d.get("label") for d in ok).most_common(8))
                lab = [d for d in ok if d.get("label") is not None and d.get("fit_ok")]
                e["agreement"] = round(sum(1 for d in lab if d.get("agree")) / len(lab), 4) if lab else None
                for t in (0.9, 0.95, 0.99):
                    acc = [d for d in lab
                           if d.get("choice") != "reply_without_tool" and (d.get("confidence") or 0) >= t]
                    nb = [d for d in acc if d.get("choice") != "bash"]
                    e[f"would_force_at_{t}"] = {
                        "share": round(len(acc) / len(lab), 4) if lab else None,
                        "agree": round(sum(1 for d in acc if d.get("agree")) / len(acc), 4) if acc else None,
                        "non_bash_share": round(len(nb) / len(lab), 4) if lab else None,
                        "non_bash_agree": round(sum(1 for d in nb if d.get("agree")) / len(nb), 4) if nb else None}
            pts[p] = e
        steps_n = len(ss)
        reasons = Counter(str(s.get("reason")) for s in ss)
        redo = Counter(r for s in ss for r in (s.get("redo") or []))
        llm_s = sum(float(s.get("latency_s") or 0) for s in ss)
        out[key] = {**row_, "steps_per_task": round(steps_n / n, 1),
                    "decision_calls_per_task": round(len(ds) / n, 1),
                    "decision_cost_usd": round(sum(float(s.get("decision_cost_usd") or 0) for s in ss), 4),
                    "decision_cost_share": round(sum(float(s.get("decision_cost_usd") or 0) for s in ss)
                                                 / max(1e-9, sum(float(s.get("cost_usd") or 0) for s in ss)), 4),
                    "llm_calls_per_step": round(sum(len(s.get("calls") or []) for s in ss) / max(1, steps_n), 3),
                    "proxy_s_per_task": round(llm_s / n, 1),
                    "reasons": dict(reasons.most_common()), "redo": dict(redo), "points": pts,
                    "tasks_ok": sorted(t["task"] for t in trials if f"{t['harness']}:{t['arm']}" == key
                                       and t["success"])}
    # jah fit per harness on the shadow arms (and any arm with labelled decisions)
    fits = {}
    for key, ds in by_arm_decs.items():
        if any(d.get("fit_ok") for d in ds):
            fits[key] = F.fit(ds)
    return {"arms": out, "fits": fits}


def md_table(res: dict, base: dict | None = None) -> str:
    rows = ["| harness:arm | n | success | $/task | agent s/task | strong share | steps/task | LLM calls/step | "
            "decision calls/task | decision $ (share) | acted per task |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res["arms"].items():
        acted = ", ".join(f"{p} {e['acted_per_task']:g}" for p, e in r["points"].items() if p != "routing")
        rows.append(f"| {k} | {r['n']} | {100 * r['success']:.1f}% | {r['cost_per_task']:.3f} | "
                    f"{r['agent_s_per_task']:.0f} | {S._pct(r['strong_share'])} | {r['steps_per_task']} | "
                    f"{r['llm_calls_per_step']} | {r['decision_calls_per_task']} | {r['decision_cost_usd']:.3f} "
                    f"({100 * r['decision_cost_share']:.1f}%) | {acted or '-'} |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("row", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    res = analyze(a.row)
    print(md_table(res))
    for k, r in res["arms"].items():
        print(f"\n## {k}: reasons {r['reasons']} redo {r['redo']}")
        for p, e in r["points"].items():
            print(f"  {p}: {json.dumps(e)}")
    for k, f in res["fits"].items():
        print(f"\n## fit {k}\n{F.table(f)}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=str))
    if a.md:
        a.md.write_text(md_table(res) + "\n")
    return 0




def counterfactuals(row: Path, arms: tuple[str, ...] = ("jah-shadow",)) -> dict:
    """From shadow logs: what an acting mode would have done."""
    jobs, proxy = row / "jobs", row / "proxy"
    hmap = harness_of(jobs)
    trials = S.load_trials(jobs)
    ok = {f"{t['arm']}/{S.safe(t['trial'])}": t["success"] for t in trials}
    decs = [d for d in load_jsonl(proxy / "decisions.jsonl") if d["session"] in ok and d["model_name"] in arms]
    out: dict = {}
    for h in sorted({hmap.get(d["session"]) for d in decs}):
        ds = [d for d in decs if hmap.get(d["session"]) == h]
        e: dict = {}
        stop = [d for d in ds if d["point"] == "stop" and d.get("fit_ok") and d.get("label") and not d.get("error")]
        if stop:
            e["stop_labels"] = dict(Counter(d["label"] for d in stop))
            e["stop_guard_at"] = {}
            for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
                acc = [d for d in stop if d["p"].get("done", 0) >= t]
                e["stop_guard_at"][t] = {"accept": len(acc), "of": len(stop),
                                         "wrong_accept": sum(1 for d in acc if d["label"] == "not_done"),
                                         "right_redo": sum(1 for d in stop if d["p"].get("done", 0) < t
                                                           and d["label"] == "not_done")}
            e["p_done_by_label"] = {lab: sorted(round(d["p"].get("done", 0), 2) for d in stop if d["label"] == lab)
                                    for lab in ("done", "not_done")}
        ts = [d for d in ds if d["point"] == "tool_select" and not d.get("error") and d.get("label")]
        fast_final = [d for d in ts if d.get("label_tier") != "strong" and d["label"] == "reply_without_tool"]
        e["fast_text_replies"] = len(fast_final)
        e["fast_text_jev"] = {t: sum(1 for d in fast_final if d["choice"] not in (None, "reply_without_tool")
                                     and d["confidence"] >= t) for t in (0.9, 0.95, 0.99)}
        fast_tool = [d for d in ts if d.get("label_tier") != "strong" and d["label"] != "reply_without_tool"]
        e["fast_tool_steps"] = len(fast_tool)
        e["fast_tool_forced_other"] = {t: sum(1 for d in fast_tool if d["choice"] not in (None, "reply_without_tool")
                                              and d["confidence"] >= t and d["choice"] != d["label"])
                                       for t in (0.9, 0.95, 0.99)}
        refl = [d for d in ds if d["point"] == "reflection" and not d.get("error")]
        if refl:
            per = defaultdict(lambda: [0, 0])
            for d in refl:
                per[d["session"]][0] += 1
                per[d["session"]][1] += int(d["p"].get("yes", 0) >= 0.8)
            for lab, v in (("ok", 1), ("fail", 0)):
                ss = [s for s in per if ok.get(s) == v]
                e[f"reflection_fires_per_task_{lab}"] = round(sum(per[s][1] for s in ss) / max(1, len(ss)), 2)
            p_yes = sorted(d["p"].get("yes", 0) for d in refl)
            e["reflection_p_yes_quantiles"] = [round(p_yes[int(q * (len(refl) - 1))], 3)
                                               for q in (0.5, 0.9, 0.95, 0.99)]
        act = [d for d in ds if d["point"] == "action" and not d.get("error")]
        if act:
            e["action_asks"] = len(act)
            e["action_p_risky"] = sorted(round(d["p"].get("risky", 0), 2) for d in act)[-15:]
        out[h] = e
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--cf":
        print(json.dumps(counterfactuals(Path(sys.argv[1]), tuple(sys.argv[3:]) or ("jah-shadow",)), indent=1))
    else:
        raise SystemExit(main())
