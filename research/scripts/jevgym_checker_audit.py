#!/usr/bin/env python3
"""Offline audit of JevGym's success checker and answer key on the logged live episodes (no API call, no new run, no
task regeneration). Post hoc and descriptive; it changes no registered checker and no registered outcome.

    python scripts/jevgym_checker_audit.py            # scans runs/*/ *.jsonl (and one level deeper)
    # writes results/jevgym_examples/checker_audit.{json,md} and paper/generated/checker_audit.tex (\res{checker/...})

Questions (external design review, 2026-10-07, results/reviews/2026-10-07/jevgym_design/):

1. Substring acceptance of multi totals. ``evaluate`` accepts a ``multi`` answer when the gold total (2 decimals) is a
   substring of the final answer. For every successful ``multi`` episode we rebuild the gold total from the log alone
   (the processed orders are the orders of the effective cancel / ship calls, which in a successful episode are exactly
   the gold orders; their amounts come from the logged ``get_orders`` observations) and count (a) successes where the
   gold total occurs only inside a longer number, (b) successes whose answer states two or more different candidate
   totals. A candidate total is a 2-decimal number of the answer that is not the amount of a processed order;
   candidates equal to the sum of a proper subset of the processed amounts are counted separately as subtotals, the
   rest as conflicting totals.
2. Order. The instruction says "in order"; ``evaluate`` compares sorted effect lists. We count successes whose
   effective writes (by customer) do not follow the instruction's customer order. The instruction is read from the
   logged decision states ("TASK: ..."); episodes without one are counted as ``no_instruction``.
3. Unbound tier evidence (JevGym-Hard, ``tier`` rule). The answer key checks that an order id, tier word and amount
   occur somewhere in the visible text, not that they belong to the same customer. We count (a) executed cancel / ship
   calls made before that customer's ``find_user`` result (tier) and the ``get_orders`` result listing that order
   (amount) had both been observed, split by who chose the tool, and (b) tool-selection states whose information-
   feasible acceptable set held cancel_order or ship_order while no customer had an unprocessed order with both
   observed.

Episodes: the last record per (file, task) without an infrastructure error; simulated LLMs are skipped. Groups by
seed block: Stage 1 (200000-200499), Stage 2 (300000-300499), the spare Stage-2 seeds of later exploratory rows
(300500-300999), JevGym-Hard evaluation (310000-310599), and development / calibration / pilot / other exploratory
seeds (everything else). Numbers the task itself states (the Hard rule thresholds) are not candidate totals. Arms are ``<run dir>/<file stem>``.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/jevgym_examples"
RUNS = ROOT / "runs"

_NUM = re.compile(r"(?<![\d.])\d+\.\d{2}(?![\d])")          # a 2-decimal number token (commas removed first)
_EFFECT = re.compile(r"^Order (ORD-\d+) (cancelled|shipped)")
_TASK = re.compile(r"TASK: (.*?)(?:\n|$)")
_NAMES = re.compile(r"\((\d+)\) (?:cancel |ship )?([A-Z][a-z]+ [A-Z][a-z]+)")
_TIER_RULE = "basic tier and the amount is above"


def block(seed: int) -> str:
    if 200000 <= seed <= 200499:
        return "stage1"
    if 300000 <= seed <= 300499:
        return "stage2"
    if 300500 <= seed <= 300999:
        return "stage2_spare"
    if 310000 <= seed <= 310599:
        return "hard_eval"
    return "dev_cal_pilot"


def first_json(obs: str) -> Any:
    head = obs.split("\nlog:")[0].strip()
    try:
        return json.loads(head)
    except (ValueError, TypeError):
        return None


def instruction_of(rec: dict[str, Any]) -> str | None:
    for s in rec.get("steps") or []:
        for d in s.get("decisions") or []:
            m = _TASK.search(d.get("state") or "")
            if m:
                return m.group(1)
    return None


def decided_by_model(d: dict[str, Any]) -> bool:
    """The decision model's tool choice was used (not shadow-only, not deferred to the LLM)."""
    return d.get("mode") not in ("shadow", "off", "generate") and not d.get("shadow") and not d.get("deferred")


def walk(rec: dict[str, Any]) -> dict[str, Any]:
    """Replay the logged observations: names, tiers, order amounts, effective writes and evidence at each write."""
    uid2name: dict[str, str] = {}
    tier_seen: set[str] = set()                 # user ids whose find_user result was observed
    order_user: dict[str, str] = {}             # order id -> user id (from get_orders results)
    amount: dict[str, float] = {}
    done: set[str] = set()
    writes: list[dict[str, Any]] = []
    label_states = 0
    label_unready = 0
    for s in rec.get("steps") or []:
        ts = [d for d in (s.get("decisions") or []) if d.get("dp") == "tool_select"]
        if ts:                                   # the answer key at the state before this step's call
            acc = (ts[0].get("gold_aux") or {}).get("acceptable_tools") or []
            if any(t in ("cancel_order", "ship_order") for t in acc):
                label_states += 1
                ready = any(order_user.get(o) in tier_seen for o in amount if o not in done)
                label_unready += 0 if ready else 1
        tool, args, obs = s.get("tool"), s.get("args") or {}, s.get("observation") or ""
        if tool == "find_user":
            u = first_json(obs)
            if isinstance(u, dict) and "user_id" in u:
                uid2name[u["user_id"]] = u.get("name", "")
                tier_seen.add(u["user_id"])
        elif tool == "get_orders":
            rows = first_json(obs)
            if isinstance(rows, list):
                for o in rows:
                    if isinstance(o, dict) and "order_id" in o:
                        order_user[o["order_id"]] = o.get("user_id")
                        amount[o["order_id"]] = float(o.get("amount", 0.0))
        elif tool in ("cancel_order", "ship_order"):
            m = _EFFECT.match(obs)
            oid = str(args.get("order_id"))
            uid = order_user.get(oid)
            writes.append({"order_id": oid, "effective": bool(m), "user": uid2name.get(uid or "", None),
                           "tier_seen": uid in tier_seen if uid else False, "amount_seen": oid in amount,
                           "tool_by": "decision model" if ts and decided_by_model(ts[0]) else "LLM"})
            if m:
                done.add(m.group(1))
    return {"writes": writes, "amount": amount, "label_states": label_states, "label_unready": label_unready}


POLICY_NUMBERS = {"300.00", "450.00", "600.00"}    # JevGym-Hard rule thresholds (jevgym_hard.py), quoted in answers


def answer_checks(answer: str, amounts: list[float], instruction: str = "") -> dict[str, Any]:
    text = answer.replace(",", "")
    gold = f"{round(sum(amounts), 2):.2f}"
    tokens = _NUM.findall(text)
    out: dict[str, Any] = {"gold_found": gold in text, "embedded_only": False, "candidates": [], "subtotals": []}
    if not out["gold_found"]:
        return out
    out["embedded_only"] = gold not in tokens
    amt = {f"{a:.2f}" for a in amounts}
    subset_sums = {f"{round(sum(c), 2):.2f}" for k in range(2, len(amounts)) for c in itertools.combinations(amounts, k)}
    given = set(_NUM.findall(instruction.replace(",", ""))) | POLICY_NUMBERS     # numbers the task itself states
    for t in sorted(set(tokens) - amt - {gold} - given):
        (out["subtotals"] if t in subset_sums else out["candidates"]).append(t)
    return out


def summarize(rec: dict[str, Any]) -> dict[str, Any]:
    w = walk(rec)
    instr = instruction_of(rec)
    out: dict[str, Any] = {"success": bool((rec.get("result") or {}).get("success")),
                           "hard_tier": bool(instr and _TIER_RULE in instr),
                           "unbound_writes": [{"tool_by": x["tool_by"], "tier_seen": x["tier_seen"],
                                               "amount_seen": x["amount_seen"]}
                                              for x in w["writes"] if not (x["tier_seen"] and x["amount_seen"])],
                           "n_all_writes": len(w["writes"]),
                           "n_writes_by_dm": sum(1 for x in w["writes"] if x["tool_by"] == "decision model"),
                           "label_states": w["label_states"], "label_unready": w["label_unready"]}
    if not out["success"]:
        return out
    eff: list[dict[str, Any]] = []
    for x in w["writes"]:
        if x["effective"] and x["order_id"] not in [e["order_id"] for e in eff]:
            eff.append(x)
    amounts = [w["amount"][e["order_id"]] for e in eff if e["order_id"] in w["amount"]]
    out.update(n_writes=len(eff), amount_complete=len(amounts) == len(eff) and bool(eff))
    if out["amount_complete"]:
        out.update(answer_checks(str(rec.get("final_answer") or ""), amounts, instr or ""))
    if instr:
        names = [n for _, n in _NAMES.findall(instr)]
        seq = [e["user"] for e in eff]
        out["order_known"] = None not in seq and len(names) > 0
        out["out_of_order"] = bool(out["order_known"]) and seq != names
    else:
        out["order_known"] = None
    out["example_answer"] = str(rec.get("final_answer") or "")[-400:]
    return out


def scan() -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    per_arm: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    files = sorted(set(RUNS.glob("*/*.jsonl")) | set(RUNS.glob("*/*/*.jsonl")))
    for f in files:
        if "_aborted" in f.parts or "_superseded" in f.parts:
            continue
        arm = str(f.relative_to(RUNS).with_suffix(""))
        with f.open() as fh:
            for line in fh:
                if not line.startswith('{"task_id": "jevgym-multi-'):
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("error") or "sim" in str(rec.get("llm") or "").lower():
                    continue
                seed = int(rec["task_id"].split("-")[2])
                per_arm[(block(seed), arm)][rec["task_id"]] = summarize(rec)   # the last record per task wins
    return per_arm


def aggregate(per_arm: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rows, examples = [], defaultdict(list)
    totals: dict[str, Counter] = defaultdict(Counter)
    for (blk, arm), tasks in sorted(per_arm.items()):
        c: Counter = Counter()
        for tid, s in tasks.items():
            c["episodes"] += 1
            if not s.get("success"):
                continue
            c["successes"] += 1
            if not s.get("amount_complete") or not s.get("gold_found"):
                c["total_not_rebuilt"] += 1
            else:
                c["total_checked"] += 1
                c["embedded_only"] += s["embedded_only"]
                c["with_subtotals"] += bool(s["subtotals"])
                c["conflicting_totals"] += bool(s["candidates"])
                if s["embedded_only"] or s["candidates"]:
                    examples["answers"].append({"arm": arm, "task": tid, "candidates": s["candidates"],
                                                "embedded_only": s["embedded_only"], "answer": s["example_answer"]})
            if s.get("order_known") is None:
                c["no_instruction"] += 1
            elif s.get("order_known"):
                c["order_checked"] += 1
                if s.get("out_of_order"):
                    c["out_of_order"] += 1
                    examples["order"].append({"arm": arm, "task": tid})
            c["hard_successes"] += bool(s.get("hard_tier"))
        for tid, s in tasks.items():               # Hard evidence: every logged episode with the tier rule
            if s.get("hard_tier"):
                c["hard_writes"] += s["n_all_writes"]
                c["hard_writes_by_dm"] += s["n_writes_by_dm"]
                c["hard_unbound_writes"] += len(s["unbound_writes"])
                c["hard_unbound_writes_by_dm"] += sum(1 for u in s["unbound_writes"] if u["tool_by"] == "decision model")
                c["hard_label_states"] += s["label_states"]
                c["hard_label_unready"] += s["label_unready"]
        rows.append({"block": blk, "arm": arm, **c})
        totals[blk].update(c)
    return {"rows": rows, "totals": {k: dict(v) for k, v in totals.items()},
            "examples": {k: v[:20] for k, v in examples.items()}}


def to_md(agg: dict[str, Any]) -> str:
    L = ["# JevGym checker audit (offline, logged episodes; descriptive)", "",
         "Generated by `scripts/jevgym_checker_audit.py`; see its docstring for definitions. Counts are over the last "
         "record per task and arm (multi tasks only).", "", "## Totals by seed block", "",
         "| block | episodes | successes | totals checked | embedded only | conflicting totals | with subtotals | "
         "order checked | out of order | no instruction |", "|---|---|---|---|---|---|---|---|---|---|"]
    for blk, t in sorted(agg["totals"].items()):
        L.append(f"| {blk} | {t.get('episodes', 0)} | {t.get('successes', 0)} | {t.get('total_checked', 0)} | "
                 f"{t.get('embedded_only', 0)} | {t.get('conflicting_totals', 0)} | {t.get('with_subtotals', 0)} | "
                 f"{t.get('order_checked', 0)} | {t.get('out_of_order', 0)} | {t.get('no_instruction', 0)} |")
    L += ["", "## JevGym-Hard (tier rule): evidence at the write", "",
          "| block | arm | writes | by decision model | before tier+amount seen | of which by decision model | "
          "label states with a write acceptable | of which no customer ready |", "|---|---|---|---|---|---|---|---|"]
    for r in agg["rows"]:
        if r.get("hard_writes"):
            L.append(f"| {r['block']} | {r['arm']} | {r['hard_writes']} | {r.get('hard_writes_by_dm', 0)} | "
                     f"{r.get('hard_unbound_writes', 0)} | {r.get('hard_unbound_writes_by_dm', 0)} | "
                     f"{r.get('hard_label_states', 0)} | {r.get('hard_label_unready', 0)} |")
    L += ["", "## Per arm (multi successes)", "",
          "| block | arm | successes | totals checked | embedded only | conflicting totals | with subtotals | "
          "order checked | out of order |", "|---|---|---|---|---|---|---|---|---|"]
    for r in agg["rows"]:
        if r.get("successes"):
            L.append(f"| {r['block']} | {r['arm']} | {r['successes']} | {r.get('total_checked', 0)} | "
                     f"{r.get('embedded_only', 0)} | {r.get('conflicting_totals', 0)} | {r.get('with_subtotals', 0)} | "
                     f"{r.get('order_checked', 0)} | {r.get('out_of_order', 0)} |")
    L += ["", "## Paper keys", ""] + [f"- `{k}` = {v}" for k, v in agg.get("keys", {}).items()]
    for k, v in agg["examples"].items():
        L += ["", f"## Examples: {k}", ""] + [f"- `{json.dumps(e)[:700]}`" for e in v]
    return "\n".join(L) + "\n"


# Flagged answers read by hand (2026-10-07). Each entry: (arm, task) -> what the answer states. None of the flagged
# answers on the Stage-1 / Stage-2 evaluation tasks states a second total; the one exploratory case does.
INSPECTED = {
    ("e2a_jevgym_tool_decisions_intern/intern_shadow_s2lite", "jevgym-multi-200307-d60-near-strict"):
        "garbled order id 'ORD-251.08' beside the right amount 51.08; one total, correct",
    ("e2c_jevgym_baselines/s2self_reflect_trigger_s2lite", "jevgym-multi-200103-d60-near-strict"):
        "one order amount given two ways ('$294.73, or 294.70'; the logged amount is 294.70); one total, correct",
    ("e6d_modes_routing/rt_guard", "jevgym-multi-301399-d60-near-strict"):
        "states a wrong total (15.00), then recomputes and states the right one (exploratory row, not an evaluation seed)",
}
TEX = ROOT / "paper/generated/checker_audit.tex"


def keys(agg: dict[str, Any]) -> dict[str, str]:
    t = agg["totals"]
    ev = [t.get(b, {}) for b in ("stage1", "stage2")]
    n = sum(x.get("total_checked", 0) for x in ev)
    ooo = sum(x.get("out_of_order", 0) for x in ev)
    flagged = [e for e in agg["examples"].get("answers", []) if int(e["task"].split("-")[2]) in range(200000, 200500)
               or int(e["task"].split("-")[2]) in range(300000, 300500)]
    # an uninspected flag counts as a second stated total (conservative)
    second_total = [e for e in flagged if "states a wrong total" in INSPECTED.get((e["arm"], e["task"]),
                                                                                   "states a wrong total")]
    hard = [t.get(b, {}) for b in ("hard_eval", "dev_cal_pilot")]
    return {"checker/eval/n_success": f"{n:,}".replace(",", "{,}"),
            "checker/eval/embedded_only": str(sum(x.get("embedded_only", 0) for x in ev)),
            "checker/eval/flagged_totals": str(len(flagged)),
            "checker/eval/second_total": str(len(second_total)),
            "checker/eval/out_of_order": str(ooo),
            "checker/eval/out_of_order_pct": f"{100 * ooo / n:.2f}" if n else "--",
            "checker/hard/writes": f"{sum(x.get('hard_writes', 0) for x in hard):,}".replace(",", "{,}"),
            "checker/hard/unbound_writes": str(sum(x.get("hard_unbound_writes", 0) for x in hard)),
            "checker/hard/label_states": f"{sum(x.get('hard_label_states', 0) for x in hard):,}".replace(",", "{,}"),
            "checker/hard/label_unready": str(sum(x.get("hard_label_unready", 0) for x in hard))}


def main() -> int:
    agg = aggregate(scan())
    for e in agg["examples"].get("answers", []):
        e["inspected"] = INSPECTED.get((e["arm"], e["task"]))
    agg["keys"] = keys(agg)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checker_audit.json").write_text(json.dumps(agg, indent=1) + "\n")
    (OUT / "checker_audit.md").write_text(to_md(agg))
    TEX.write_text("% generated by scripts/jevgym_checker_audit.py (offline audit of logged episodes; descriptive, post hoc)\n"
                   "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}\n"
                   + "".join(f"\\defres{{{k}}}{{{v}}}\n" for k, v in agg["keys"].items()))
    print(json.dumps(agg["totals"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
