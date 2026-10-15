#!/usr/bin/env python3
"""Concrete JevGym example tasks for the appendix (App. app:jevgym-examples), generated from the code at fixed
DEVELOPMENT seeds. Offline: no API call, no GPU, no episode is run except the deterministic walk of the canonical plan.

    python scripts/jevgym_examples.py
    # writes results/jevgym_examples/examples.{json,md}

Examples (registered evaluation settings: v2 families, 60 near-duplicate distractors, 20 noise lines, strict mode):

* seed 600: a ``clarify`` task (variant ``order``: the customer has two pending orders);
* seed 605: a ``multi`` task (three customers);
* seed 7003: a JevGym-Hard ``multi`` task under the ``tier`` order rule with 10 customers (E7's settings; seeds
  7000-7299 are E7's development seeds).

For each it records the instruction, the first observation, an excerpt of the offered tools, the canonical plan, the
answer key at one intermediate state (the canonical next step and the acceptable, information-feasible tool set) and
the verdicts of a few proposed calls at that state. The evaluation blocks (200000-200499, the retired 100000-100499,
300000-300999 and 310000-310599) are refused. The optional logged episode is read from a development run
(``runs/harness_final_tariff``, seeds 600-699) when it is present locally; it records who chose each step, no numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import registry  # noqa: E402
from jevharness.bench.suites.jevgym import NEAR_BY_NAME, JevGymEnv  # noqa: E402
from jevharness.bench.suites.jevgym_hard import JevGymHardEnv  # noqa: E402

OUT = ROOT / "results/jevgym_examples"
EVAL = dict(families="v2", distractor_style="near", n_distractors=60, noise_lines=20, strict=True)
HARD = dict(families="multi", multi_customers=10, multi_rule="tier", distractor_style="near", n_distractors=60,
            noise_lines=20, strict=True)
LOGGED_RUN = ROOT / "runs/harness_final_tariff/harness_final.jsonl"
CORE_EXCERPT = ["find_user", "get_orders", "cancel_order", "ship_order", "refund", "ask_user"]
NEAR_EXCERPT = ["get_order", "get_orders_summary", "cancel_order_item", "cancel_shipment", "ship_order_international",
                "request_callback"]


def refuse_reserved(seed: int) -> None:
    for ranges in (registry.EVAL_SEED_RANGES, registry.RETIRED_EVAL_RANGES, registry.STAGE2_RESERVED_RANGES,
                   registry.EXPLORE_EVAL_RANGES):
        if registry.in_ranges(seed, ranges):
            raise SystemExit(f"seed {seed} is in a reserved evaluation block {ranges}; use a development seed")


def first_line(obs: str) -> str:
    """The observation without the 20 heartbeat noise lines."""
    return obs.split("\nlog:")[0]


def tool_excerpt(env: JevGymEnv) -> list[dict[str, Any]]:
    by_name = {t["function"]["name"]: t["function"] for t in env.tools()}
    rows = []
    for name in CORE_EXCERPT + NEAR_EXCERPT:
        fn = by_name[name]
        near = NEAR_BY_NAME.get(name)
        rows.append({"name": name, "description": fn["description"], "params": list(fn["parameters"]["required"]),
                     "kind": "core" if near is None else ("near-duplicate, changes state" if near.write
                                                           else "near-duplicate, read-only"),
                     "shadows": near.shadows if near else None})
    return rows


def answer_key(env: JevGymEnv, probes: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    nxt = env.oracle_action()
    return {"canonical_next": {"tool": nxt[0], "args": nxt[1]} if nxt else None,
            "acceptable_tools": env.oracle_acceptable_tools(),
            "steps_to_go": env.oracle_cost_to_go(),
            "verdicts": [{"tool": t, "args": a, "verdict": env.oracle_verdict(t, a)["verdict"]} for t, a in probes]}


def walk(env: JevGymEnv, n: int) -> list[dict[str, Any]]:
    out = []
    for _ in range(n):
        tool, args = env.oracle_action()
        out.append({"tool": tool, "args": args, "observation": first_line(env.step(tool, args).observation)})
    return out


def describe(env: JevGymEnv) -> dict[str, Any]:
    obs = env.reset()
    return {"task_id": env.task_id, "seed": env.seed, "family": env.family, "variant": env.variant,
            "instruction": env.instruction, "first_observation": obs, "n_tools_offered": len(env.tools()),
            "n_near_duplicates_offered": len(env._near_offered), "tools_excerpt": tool_excerpt(env),
            "plan": [{"tool": t, "args": a} for t, a in env._plan()]}


def clarify_example(seed: int = 600) -> dict[str, Any]:
    refuse_reserved(seed)
    env = JevGymEnv(seed=seed, **EVAL)
    d = describe(env)
    uid = env.user["user_id"]
    pend = sorted((o for o in env.world.orders.values() if o["user_id"] == uid and o["status"] == "pending"),
                  key=lambda o: o["order_id"])
    gold = env.pending_order
    other = next(o for o in pend if o["order_id"] != gold["order_id"])
    d["pending_orders"] = pend
    d["customer_reply"] = env.reply
    d["key_at_start"] = {"canonical_next": env.oracle_action(), "acceptable_tools": env.oracle_acceptable_tools()}
    d["walk"] = walk(env, 2)
    d["state"] = "after find_user and get_orders"
    d["answer_key"] = answer_key(env, [
        ("ask_user", {"missing": "order_id", "question": "Which of your two pending orders should I cancel?"}),
        ("cancel_order", {"order_id": gold["order_id"], "reason": "customer_request"}),
        ("cancel_order", {"order_id": other["order_id"], "reason": "customer_request"}),
        ("cancel_order_item", {"order_id": gold["order_id"], "item": gold["item"], "reason": "customer_request"}),
        ("get_order", {"order_id": gold["order_id"]}),
        ("ask_user", {"missing": "refund_method"}),
        ("finish", {"answer": "done"}),
    ])
    return d


def multi_example(seed: int = 605) -> dict[str, Any]:
    refuse_reserved(seed)
    env = JevGymEnv(seed=seed, **EVAL)
    d = describe(env)
    d["jobs"] = [{"op": op, "name": u["name"], "tier": u["tier"], "order_id": o["order_id"], "amount": o["amount"]}
                 for op, u, o in env.jobs]
    d["walk"] = walk(env, 4)
    d["state"] = "after the first customer is done and the second one is looked up"
    second = env.jobs[1][2]["order_id"]
    d["answer_key"] = answer_key(env, [
        ("get_orders", {"user_id": env.jobs[1][1]["user_id"], "status": "pending"}),
        ("find_user", {"name": env.jobs[2][1]["name"]}),
        ("get_orders_summary", {"user_id": env.jobs[1][1]["user_id"]}),
        ("ship_order", {"order_id": second, "speed": "express"}),
        ("cancel_shipment", {"order_id": second, "reason": "customer_request"}),
        ("finish", {"answer": f"{env.multi_total:.2f}"}),
    ])
    return d


def hard_example(seed: int = 7003) -> dict[str, Any]:
    refuse_reserved(seed)
    env = JevGymHardEnv(seed=seed, **HARD)
    obs = env.reset()
    d = {"task_id": env.task_id, "seed": seed, "family": "multi", "multi_rule": "tier", "instruction": env.instruction,
         "first_observation": obs, "n_tools_offered": len(env.tools()), "n_plan_steps": len(env._plan()),
         "jobs": [{"op": op, "name": u["name"], "tier": u["tier"], "order_id": o["order_id"], "amount": o["amount"]}
                  for op, u, o in env.jobs], "total": f"{env.multi_total:.2f}"}
    walked = []
    for i, (op, u, o) in enumerate(env.jobs):          # walk to the decision of each customer and probe both ops
        while True:
            nxt = env.oracle_action()
            if nxt[0] in ("cancel_order", "ship_order"):
                break
            env.step(*nxt)
        speed = "express" if u["tier"] == "gold" else "standard"
        other = (("ship_order", {"order_id": o["order_id"], "speed": speed}) if op == "cancel"
                 else ("cancel_order", {"order_id": o["order_id"], "reason": "customer_request"}))
        walked.append({"customer": i + 1, "canonical_next": {"tool": nxt[0], "args": nxt[1]},
                       "other_op": {"tool": other[0], "args": other[1],
                                    "verdict": env.oracle_verdict(*other)["verdict"]}})
        env.step(*nxt)
    d["decisions"] = walked
    return d


def logged_episode(task_id: str) -> dict[str, Any] | None:
    """Who chose each step of a logged development episode (no numbers): the decision model took the tool choice
    when its answer was not deferred; free-text arguments are always written by the LLM."""
    if not LOGGED_RUN.exists():
        return None
    for line in LOGGED_RUN.open():
        r = json.loads(line)
        if r["task_id"] != task_id:
            continue
        steps = []
        for s in r["steps"]:
            ts = [d for d in s.get("decisions", []) if d["dp"] == "tool_select"]
            tp = [d for d in s.get("decisions", []) if d["dp"] == "tool_params"]
            steps.append({"tool": s["tool"], "args": s["args"], "observation": first_line(s["observation"]),
                          "tool_chosen_by": "decision model" if ts and not ts[0]["deferred"] else "LLM",
                          "args_by_decision_model": sum(1 for d in tp if not d["deferred"]),
                          "args_by_llm": sum(1 for d in tp if d["deferred"])})
        return {"run": str(LOGGED_RUN.relative_to(ROOT)), "task_id": task_id, "config": r["config"],
                "success": r["result"]["success"], "steps": steps}
    return None


def to_md(ex: dict[str, Any]) -> str:
    lines = ["# JevGym examples (generated by scripts/jevgym_examples.py; development seeds only)", ""]
    for key in ("clarify", "multi", "hard"):
        e = ex[key]
        lines += [f"## {key}: {e['task_id']}", "", f"Instruction: {e['instruction']}", "",
                  f"First observation: {e['first_observation']}", "",
                  f"Tools offered: {e['n_tools_offered']} (+ finish)", ""]
        for t in e.get("tools_excerpt", []):
            lines.append(f"- `{t['name']}({', '.join(t['params'])})` [{t['kind']}]: {t['description']}")
        if "plan" in e:
            lines += ["", "Canonical plan:"] + [f"{i + 1}. {p['tool']}({json.dumps(p['args'])})"
                                                for i, p in enumerate(e["plan"])]
        for k in ("pending_orders", "customer_reply", "key_at_start", "jobs", "walk", "state", "answer_key",
                  "decisions", "total"):
            if k in e:
                lines += ["", f"{k}: ```{json.dumps(e[k], default=str)}```"]
        lines.append("")
    if ex.get("logged"):
        lines += ["## logged development episode", "", "```", json.dumps(ex["logged"], indent=1), "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ex = {"clarify": clarify_example(), "multi": multi_example(), "hard": hard_example(),
          "logged": logged_episode("jevgym-clarify-600-d60-near-strict")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "examples.json").write_text(json.dumps(ex, indent=1, default=str) + "\n")
    (OUT / "examples.md").write_text(to_md(ex))
    print(f"wrote {OUT / 'examples.json'} and {OUT / 'examples.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
