"""Counterfactual Monte-Carlo labels for decision points (which choice *leads to success*?).

For a logged trajectory and a decision step t, every alternative option a is executed after
replaying the first t actions, then the executor policy (a fixed harness + LLM) finishes the
episode M times. Q(s_t, a) = (successes + 1) / (M + 2). These Q-values give
  * soft targets    p*(a) ∝ pi_ref(a|s) exp(Q(s,a)/beta)   (KL-regularised optimum; train with soft_ce)
  * DPO pairs       argmax-Q vs argmin-Q                    (train with --loss dpo)
  * exact PG        E_{a~pi}[Q(s,a)]                          (train with --loss pg)

  python -m jevharness.training.mc_rollouts --exp configs/experiments/jevgym_tool_select.yaml \
      --runs runs/jevgym_tool_select/llm_only.jsonl --max-states 500 --k-alt 4 --m 4 \
      --out data/decisions/mc.jsonl

Environments must be deterministic given the action prefix (JevGym, tau2 with a fixed user
seed, AppWorld). For LLM executors, the prefix is summarised into the first observation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from ..bench.env import Env, StepResult
from ..harness.agent import JevAgent
from ..harness.config import HarnessConfig
from ..harness.tools import finish_tool, tool_desc, tool_name
from ..harness.trajectory import load_trajectories
from ..llm import LLMClient, make_llm
from .data import mc_q_estimate, mc_target, option_text, write_jsonl


class PrefixEnv(Env):
    """Replays an action prefix (+ a forced alternative) on reset; delegates everything else."""

    def __init__(self, factory: Callable[[], Env], prefix: list[tuple[str, dict[str, Any]]], tag: str = ""):
        self.factory = factory
        self.prefix = prefix
        self.inner = factory()
        # ``tag`` (e.g. "#r3") makes rollouts distinct for executors seeded by task_id (SimulatedLLM): rollout r of
        # every alternative shares a seed -> common random numbers across alternatives, independent across r.
        self.task_id = f"{self.inner.task_id}#prefix{len(prefix)}{tag}"
        self.instruction = self.inner.instruction
        self.done_in_prefix = False

    def reset(self) -> str:
        self.inner = self.factory()
        obs = self.inner.reset()
        log = []
        for tool, args in self.prefix:
            r = self.inner.step(tool, args)
            log.append(f"{tool}({json.dumps(args)}) -> {r.observation[:500]}")
            if r.done:
                self.done_in_prefix = True
        return obs + ("\n\nActions already taken (continue from here):\n" + "\n".join(log) if log else "")

    def tools(self) -> list[dict[str, Any]]:
        return self.inner.tools()

    def admissible_actions(self) -> list[str] | None:
        return self.inner.admissible_actions()

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        return self.inner.step(tool, args)

    def oracle_action(self):  # noqa: ANN201
        return self.inner.oracle_action()

    def oracle_done(self):  # noqa: ANN201
        return self.inner.oracle_done()

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        return self.inner.evaluate(final_answer)


USAGE_KEYS = ("cost_usd", "calls", "input_tokens", "output_tokens")


def add_usage(acc: dict[str, float], traj: Any) -> None:
    """Accumulate a rollout's ledger totals (``Trajectory.usage``; every executor call goes through the harness
    Ledger) into ``acc``, plus the episode / error counts."""
    tot = (getattr(traj, "usage", None) or {}).get("total") or {}
    for k in USAGE_KEYS:
        acc[k] = acc.get(k, 0) + float(tot.get(k) or 0)
    acc["episodes"] = acc.get("episodes", 0) + 1
    acc["errors"] = acc.get("errors", 0) + int(getattr(traj, "error", None) is not None)


def q_values_for_state(factory: Callable[[], Env], prefix: list[tuple[str, dict[str, Any]]],
                       alternatives: list[tuple[str, dict[str, Any]]], executor: LLMClient,
                       cfg: HarnessConfig, m: int, usage: dict[str, float] | None = None) -> dict[str, float]:
    q = {}
    for tool, args in alternatives:
        wins = 0
        for r in range(m):
            env = PrefixEnv(factory, prefix + [(tool, args)], tag=f"#r{r}")
            traj = JevAgent(executor, cfg).run(env)
            wins += int(traj.success)
            if usage is not None:
                add_usage(usage, traj)
        q[tool] = mc_q_estimate(wins, m)
    return q


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment YAML (suite + llm = executor); use the "
                    "runs/<name>/experiment.yaml saved by `jevbench run` so the suite matches the trajectories")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--max-states", type=int, default=200)
    ap.add_argument("--k-alt", type=int, default=4, help="alternatives per state (incl. the taken action)")
    ap.add_argument("--m", type=int, default=4, help="rollouts per alternative")
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="states evaluated concurrently (threads)")
    ap.add_argument("--num-shards", type=int, default=1, help="split the sampled states into N disjoint shards")
    ap.add_argument("--shard-id", type=int, default=0)
    a = ap.parse_args(argv)
    if not 0 <= a.shard_id < a.num_shards:
        raise SystemExit("--shard-id must be in [0, --num-shards)")
    from dotenv import load_dotenv

    load_dotenv()
    exp = yaml.safe_load(open(a.exp))
    from ..bench.suites import load_suite

    suite = dict(exp["suite"])
    factories = {f().task_id: f for f in load_suite(suite.pop("name"), **suite)}
    executor = make_llm(exp["llm"])
    cfg = HarnessConfig(name="executor", max_steps=exp.get("harness_defaults", {}).get("max_steps", 30))
    rng = random.Random(a.seed)
    states = []
    for path in a.runs:
        for t in load_trajectories(path):
            if t["task_id"] not in factories:
                continue
            actions = [(s["tool"], s["args"]) for s in t["steps"] if s.get("tool")]
            for i, s in enumerate(t["steps"]):
                d = next((d for d in s["decisions"] if d["dp"] == "tool_select" and d.get("state")), None)
                if d and s.get("tool") and s["tool"] != "finish":
                    states.append((t["task_id"], actions[:i], s, d))
    if not states:
        raise SystemExit("no logged tool_select state belongs to a task of the --exp suite: pass the experiment.yaml "
                         "that `jevbench run` saved next to the trajectories (it holds the resolved suite/seeds)")
    rng.shuffle(states)
    states = states[: a.max_states][a.shard_id::a.num_shards]
    # alternatives are drawn up front (deterministic given --seed, independent of --workers)
    jobs = []
    for task_id, prefix, step, d in states:
        names = [x["function"]["name"] for x in factories[task_id]().tools()]
        alts = [step["tool"]] + rng.sample([n for n in names if n != step["tool"]], min(a.k_alt - 1, len(names) - 1))
        jobs.append((task_id, prefix, step, d, alts))

    def one(job: tuple[Any, ...]) -> dict[str, Any]:
        task_id, prefix, step, d, alts = job
        # the LLM cannot choose arguments for a non-taken tool inside a replay, so alternatives get the taken
        # args when the name matches, else {}
        alternatives = [(n, step["args"] if n == step["tool"] else {}) for n in alts]
        usage: dict[str, float] = {}
        qv_name = q_values_for_state(factories[task_id], prefix, alternatives, executor, cfg, a.m, usage)
        # the decision is a Choice over the evaluated alternatives, rendered `name -- description` exactly as the
        # served model sees a tool_select Choice (serve_jev / data.option_text)
        desc = {tool_name(t): tool_desc(t) for t in factories[task_id]().tools() + [finish_tool(cfg.final_tool)]}
        text = {n: option_text(n, desc.get(n)) for n in alts}
        opts = [text[n] for n in alts]
        qv = {text[n]: v for n, v in qv_name.items()}
        dist = d.get("s1_dist") or {}
        pi_ref = {text[n]: dist.get(n, 1.0 / len(opts)) for n in alts}
        return {"id": f"{task_id}/{step['step']}/mc", "source": "mc", "dp": "tool_select", "kind": "choice",
                "state": d["state"], "question": "Which tool should the agent call next?", "options": opts,
                "target": mc_target(pi_ref, qv, a.beta), "q_values": qv,
                "meta": {"label_source": "mc", "m": a.m, "beta": a.beta, "shard": f"{a.shard_id}/{a.num_shards}", "tools": alts,
                         "rollout_usage": {k: round(v, 6) for k, v in usage.items()}}}

    t0 = time.perf_counter()
    if a.workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            out = list(pool.map(one, jobs))
    else:
        out = [one(j) for j in jobs]
    write_jsonl(out, a.out)
    total: dict[str, float] = {}
    for row in out:
        for k, v in row["meta"]["rollout_usage"].items():
            total[k] = total.get(k, 0) + v
    summary = {"states": len(out), "k_alt": a.k_alt, "m": a.m, "beta": a.beta, "executor": exp["llm"],
               "shard": f"{a.shard_id}/{a.num_shards}", "wall_s": round(time.perf_counter() - t0, 1),
               **{k: round(v, 6) for k, v in total.items()},
               "cost_usd_per_episode": round(total.get("cost_usd", 0) / max(1, total.get("episodes", 0)), 6)}
    Path(f"{a.out}.summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {len(out)} MC-labelled decisions to {a.out}: {json.dumps(summary, default=str)}")


if __name__ == "__main__":
    main()
