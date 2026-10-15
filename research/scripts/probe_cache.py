#!/usr/bin/env python3
"""Prompt-cache probe for the System-2 client (review round 4, P1-1).

Three modes, all printing ``prompt_tokens`` / ``cached_tokens`` straight from the provider's usage payload:

  # 1. offline: what the logged ledgers say (no spend) -- per arm, per episode, per step
  python scripts/probe_cache.py --run runs/pilot_jev_r2/e2a_jevgym_tool_decisions [--arms llm_only,jev_ts_assist_k3]

  # 2. replay: rebuild the env of logged episodes, replay the logged tool calls through it (JevGym is seeded, so
  #    the observations and hence the exact System-2 messages of every step are reproduced) and send each step's
  #    messages + tools to the real model
  python scripts/probe_cache.py --replay runs/pilot_jev_r2/e2a_jevgym_tool_decisions --arm llm_only --episodes 2

  # 3. controlled probes: identical calls at several prefix sizes, tools in `tools` vs folded into the system text,
  #    prefix vs. suffix changes, conversation growth
  python scripts/probe_cache.py --live [--probes sizes,tools,prefix,growth] [--model gemini-3.5-flash]

Every live call goes through ``jevharness.llm.make_llm`` (the harness client, ``llm/openai_compat.py``) and a
``Ledger``, so the table also shows what the ledger keeps (``meta.cached_input_tokens``). ``--budget-usd`` (default 2)
stops the probes once the ledger's priced cost passes it. Results: ``--out`` JSON (default
results/pilot_jev/cache_probe_<mode>.json). Findings: results/pilot_jev/CACHE_NOTES.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.accounting import Ledger, default_prices  # noqa: E402
from jevharness.harness.trajectory import load_trajectories  # noqa: E402

_WORDS = ("alpha beta gamma delta order refund customer ticket warehouse stock invoice policy shipping region "
          "status total amount reason account email phone address city item price quantity date note").split()


# ----------------------------------------------------------------------------------------------- offline
def offline(run_dir: Path, arms: list[str] | None, per_step: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in sorted(run_dir.glob("*.jsonl")):
        if arms and p.stem not in arms:
            continue
        trajs = load_trajectories(p)
        agg: Counter = Counter()
        eps = []
        for t in trajs:
            s2 = [c for c in t.get("calls", []) if c.get("component") == "s2" and not (c.get("meta") or {}).get("shadow")]
            steps = [(c.get("step"), c["input_tokens"], (c.get("meta") or {}).get("cached_input_tokens", 0),
                      (c.get("meta") or {}).get("n_tools")) for c in s2]
            eps.append({"task_id": t["task_id"], "steps": steps})
            for _, i, c, _ in steps:
                agg["calls"] += 1
                agg["input"] += i
                agg["cached"] += c
                agg["hits"] += int(c > 0)
        frac = agg["cached"] / agg["input"] if agg["input"] else 0.0
        out[p.stem] = {"calls": agg["calls"], "input_tokens": agg["input"], "cached_tokens": agg["cached"],
                       "cached_frac": frac, "calls_with_hit": agg["hits"], "episodes": eps}
        print(f"{p.stem:38s} s2 calls={agg['calls']:4d} input={agg['input']:8d} cached={agg['cached']:8d} "
              f"({100 * frac:5.1f}%) calls with a hit={agg['hits']}")
        if per_step:
            for e in eps:
                print("   " + e["task_id"][:36].ljust(37)
                      + " ".join(f"{s}:{i}/{c}" + (f"/{n}t" if n is not None else "") for s, i, c, n in e["steps"]))
    return out


# ----------------------------------------------------------------------------------------------- live helpers
class Probe:
    def __init__(self, model: str, provider: str, budget_usd: float, pause: float = 1.0):
        from jevharness.llm import make_llm

        self.llm = make_llm({"provider": provider, "model": model, "temperature": 0, "reasoning_effort": "minimal"})
        self.ledger = Ledger()
        self.budget = budget_usd
        self.pause = pause
        self.rows: list[dict[str, Any]] = []

    def spent(self) -> float:
        return sum(r.cost_usd for r in self.ledger.records)

    def call(self, label: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             **kw: Any) -> dict[str, Any]:
        if self.spent() > self.budget:
            raise SystemExit(f"budget {self.budget} USD exceeded ({self.spent():.3f}); stopping")
        resp = self.llm.chat(messages, tools=tools, ledger=self.ledger, decision_point="probe", max_tokens=16, **kw)
        u = getattr(resp.raw, "usage", None)
        raw = u.model_dump() if u is not None and hasattr(u, "model_dump") else None
        details = (raw or {}).get("prompt_tokens_details") or {}
        rec = self.ledger.records[-1]
        row = {"label": label, "prompt_tokens": (raw or {}).get("prompt_tokens"),
               "cached_tokens_raw": details.get("cached_tokens"),
               "ledger_input": rec.input_tokens, "ledger_cached": rec.meta.get("cached_input_tokens"),
               "ledger_cost_usd": rec.cost_usd, "latency_ms": round(resp.latency_ms), "n_tools": len(tools or [])}
        self.rows.append(row)
        print(f"  {label:48s} prompt={row['prompt_tokens']:>6} cached={str(row['cached_tokens_raw']):>6} "
              f"ledger={row['ledger_input']}/{row['ledger_cached']} ${rec.cost_usd:.5f} {row['latency_ms']} ms")
        time.sleep(self.pause)
        return row


def filler(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) + (str(rng.randint(0, 999)) if rng.random() < 0.3 else "")
                    for _ in range(n_words))


def jevgym_prompt(exp: dict[str, Any], task_ids: list[str] | None = None, n: int = 1):
    """(env, system prompt, tools) of JevGym tasks: the first ``n`` of the suite, or the named task ids."""
    from jevharness.bench.suites import load_suite
    from jevharness.harness.agent import ACTION_SYSTEM, DEFAULT_SYSTEM
    from jevharness.harness.tools import finish_tool

    suite_kw = dict(exp["suite"])
    facs = load_suite(suite_kw.pop("name"), **suite_kw)
    envs = []
    if task_ids:
        want = set(task_ids)
        for f in facs:
            e = f()
            if e.task_id in want:
                envs.append(e)
    else:
        envs = [f() for f in facs[:n]]
    out = []
    for e in envs:
        obs = e.reset()
        action_env = e.admissible_actions() is not None
        tools = [] if action_env else e.tools() + [finish_tool("finish")]
        sys_prompt = exp.get("system_prompt") or (ACTION_SYSTEM if action_env else DEFAULT_SYSTEM)
        out.append((e, sys_prompt, tools, obs))
    return out


# ----------------------------------------------------------------------------------------------- replay
def replay(run_dir: Path, arm: str, n_episodes: int, probe: Probe, obs_truncate: int = 2000) -> list[dict[str, Any]]:
    """The exact System-2 messages of the logged episodes, step by step, against the real model."""
    from jevharness.harness.agent import _trunc

    exp = yaml.safe_load((run_dir / "experiment.yaml").read_text())
    trajs = load_trajectories(run_dir / f"{arm}.jsonl")[:n_episodes]
    by_id = {t["task_id"]: t for t in trajs}
    results = []
    for env, sys_prompt, tools, obs in jevgym_prompt(exp, task_ids=list(by_id)):
        t = by_id[env.task_id]
        logged = [c for c in t.get("calls", []) if c.get("component") == "s2" and not (c.get("meta") or {}).get("shadow")]
        print(f"episode {env.task_id} ({len(t['steps'])} logged steps, {len(tools)} tools)")
        messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt},
                                          {"role": "user", "content": f"{env.instruction}\n\n{obs}"}]
        rows = []
        for k, s in enumerate(t["steps"]):
            row = probe.call(f"{env.task_id[:24]} step {k}", messages, tools)
            lg = next((c for c in logged if c.get("step") == k), None)
            row["logged_prompt"] = lg["input_tokens"] if lg else None
            row["logged_cached"] = (lg.get("meta") or {}).get("cached_input_tokens") if lg else None
            rows.append(row)
            name, args = s["tool"], s.get("args") or {}
            if name == "finish" or name is None:
                break
            cid = f"call_{random.getrandbits(32):08x}"
            messages.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
            res = env.step(name, args)
            messages.append({"role": "tool", "tool_call_id": cid, "content": _trunc(res.observation, obs_truncate)})
            if res.done:
                break
        results.append({"task_id": env.task_id, "steps": rows})
    return results


# ----------------------------------------------------------------------------------------------- controlled probes
def probe_sizes(p: Probe, sizes: list[int]) -> None:
    print("## identical call x2, text-only system prompt of ~N tokens (cached on the 2nd call = prefix cache size)")
    for n in sizes:
        sys_text = "Reference material:\n" + filler(int(n * 0.9), seed=n)
        msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": "Reply with the single word OK."}]
        p.call(f"sys~{n} call1", msgs)
        p.call(f"sys~{n} call2 (identical)", msgs)


def probe_tools(p: Probe, exp: dict[str, Any]) -> None:
    print("## the real JevGym prompt (69 tools): tools in `tools` vs the same schemas folded into the system text")
    env, sys_prompt, tools, obs = jevgym_prompt(exp, n=1)[0]
    user = {"role": "user", "content": f"{env.instruction}\n\n{obs}"}
    msgs = [{"role": "system", "content": sys_prompt}, user]
    p.call("tools field, call1", msgs, tools)
    p.call("tools field, call2 (identical)", msgs, tools)
    p.call("tools field, call3 (identical)", msgs, tools)
    folded = sys_prompt + "\n\nAvailable tools (JSON schemas):\n" + json.dumps(tools)
    msgs_f = [{"role": "system", "content": folded}, user]
    p.call("tools folded into system text, call1", msgs_f)
    p.call("tools folded into system text, call2", msgs_f)
    # the pilot's step-1 shape: same prefix + one assistant call + one tool result appended
    cid = "call_probe0001"
    nxt = msgs + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": tools[0]["function"]["name"], "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": cid, "content": "{}\n" + "\n".join(f"log: heartbeat ok #{i}" for i in range(20))}]
    p.call("tools field, step-1 shape (prefix + 1 turn)", msgs if False else nxt, tools)
    # tools shuffled (another task's order): does the cache survive a different tool order?
    sh = list(tools)
    random.Random(7).shuffle(sh)
    p.call("tools field, same tools shuffled", msgs, sh)


def probe_prefix(p: Probe, n: int = 6000) -> None:
    print(f"## prefix vs. suffix changes on a ~{n}-token system prompt (is the cache prefix-based?)")
    base = "Reference material:\n" + filler(int(n * 0.9), seed=99)
    msgs = [{"role": "system", "content": base}, {"role": "user", "content": "Reply with the single word OK."}]
    p.call("base call1", msgs)
    p.call("base call2 (identical)", msgs)
    p.call("same system, different user text", [msgs[0], {"role": "user", "content": "Say YES and nothing else."}])
    p.call("first 40 chars of the system changed", [{"role": "system", "content": "CHANGED PREFIX " + base[15:]}, msgs[1]])
    cut = len(base) // 2
    p.call("middle of the system changed", [{"role": "system", "content": base[:cut] + " CHANGED " + base[cut + 9:]}, msgs[1]])
    p.call("last 40 chars of the system changed", [{"role": "system", "content": base[:-15] + " CHANGED SUFFIX"}, msgs[1]])


def probe_growth(p: Probe, n: int = 6000, extra: int = 2600) -> None:
    print(f"## growth: ~{n}-token prompt, then the same prefix + a ~{extra}-token turn appended "
          "(does the cached size follow the previous prompt or the current one?)")
    base = "Reference material:\n" + filler(int(n * 0.9), seed=5)
    msgs = [{"role": "system", "content": base}, {"role": "user", "content": "Reply with the single word OK."}]
    p.call("base call1", msgs)
    p.call("base call2 (identical)", msgs)
    grown = msgs + [{"role": "assistant", "content": "OK"},
                    {"role": "user", "content": "More material: " + filler(int(extra * 0.9), seed=6) + "\nReply OK."}]
    p.call(f"prefix + ~{extra} tokens, call1", grown)
    p.call(f"prefix + ~{extra} tokens, call2 (identical)", grown)


def probe_repeat(p: Probe, n: int = 6000, times: int = 4) -> None:
    print(f"## the same ~{n}-token prompt {times}x in a row (does the credited size grow with repetition?)")
    base = "Reference material:\n" + filler(int(n * 0.9), seed=11)
    msgs = [{"role": "system", "content": base}, {"role": "user", "content": "Reply with the single word OK."}]
    for i in range(times):
        p.call(f"repeat call{i + 1}", msgs)


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="offline: a run directory (per arm / episode / step from the ledgers; no spend)")
    ap.add_argument("--arms", help="offline: comma-separated arm names (default: all)")
    ap.add_argument("--no-steps", action="store_true", help="offline: per-arm totals only")
    ap.add_argument("--replay", help="live: replay the logged episodes of --arm of this run directory")
    ap.add_argument("--arm", default="llm_only")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--live", action="store_true", help="live: controlled probes")
    ap.add_argument("--probes", default="sizes,tools,prefix,growth,repeat")
    ap.add_argument("--sizes", default="1000,2000,3000,4000,6000,8000")
    ap.add_argument("--exp", default="configs/dev/e2a_dev.yaml", help="experiment YAML whose suite gives the real prompt")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between live calls")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    result: dict[str, Any] = {"model": a.model, "provider": a.provider}
    mode = None
    if a.run:
        mode = "offline"
        result["offline"] = offline(Path(a.run), a.arms.split(",") if a.arms else None, per_step=not a.no_steps)
    probe = None
    if a.replay or a.live:
        probe = Probe(a.model, a.provider, a.budget_usd, pause=a.pause)
        print(f"model {a.model} via {a.provider}; price row: {default_prices().get(a.model)}")
    if a.replay:
        mode = "replay"
        result["replay"] = replay(Path(a.replay), a.arm, a.episodes, probe)
    if a.live:
        mode = "live" if mode is None else mode
        exp = yaml.safe_load((ROOT / a.exp).read_text())
        for name in a.probes.split(","):
            if name == "sizes":
                probe_sizes(probe, [int(x) for x in a.sizes.split(",")])
            elif name == "tools":
                probe_tools(probe, exp)
            elif name == "prefix":
                probe_prefix(probe)
            elif name == "growth":
                probe_growth(probe)
            elif name == "repeat":
                probe_repeat(probe)
            else:
                raise SystemExit(f"unknown probe {name!r}")
    if probe is not None:
        result["calls"] = probe.rows
        result["spent_usd"] = probe.spent()
        print(f"spent (ledger, list price with the cached rate applied): ${probe.spent():.4f} over "
              f"{len(probe.ledger.records)} calls")
    if mode is None:
        ap.error("one of --run, --replay, --live is required")
    out = Path(a.out) if a.out else ROOT / "results" / "pilot_jev" / f"cache_probe_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
