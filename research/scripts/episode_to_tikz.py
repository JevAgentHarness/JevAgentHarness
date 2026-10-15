#!/usr/bin/env python3
r"""One logged JevGym episode -> the \defres{fig1/...} keys of the paper's Figure 1 (paper/figures/fig1_decision.tex).

The figure shows one tool decision that the decision model took itself and one it left to the LLM, from a *logged*
episode of the selected profile (Flash + Jev, cascade) on the fresh Stage-2 seeds. Nothing in the figure is typed by
hand: every option name, probability, confidence, threshold, cost and latency comes from this file's output.

    python scripts/episode_to_tikz.py
    # reads runs/stage2_jevgym/jevagent_best.jsonl and configs/harness/jevagent_best.yaml,
    # writes paper/generated/fig1_episode.tex

Selection rule (deterministic; fixed before looking at the figure):

1. Episodes are ordered by their seed (the integer in the task id, e.g. ``jevgym-clarify-300000-...`` -> 300000), ties
   by task id.
2. An episode qualifies when (a) it succeeded, (b) it has at most ``MAX_STEPS`` steps (so the step strip fits one
   column), (c) it has at least one *deferred* cascade ``tool_select`` decision (confidence below the threshold: the
   LLM chose the tool) and (d) at least one *accepted* cascade ``tool_select`` decision in a step with no LLM call at
   all (the decision model chose the tool and filled its arguments, so the step ran without the LLM).
3. The first qualifying episode is shown. In it, the *deferred* step is the deferred ``tool_select`` decision with the
   lowest confidence (earliest on ties); the *accepted* step is the first qualifying accepted step after it (or, if
   none follows, the last one before it).
4. Each of the two steps shows the top ``TOP_K`` options by the decision model's probability (ties: the order in
   which the options were offered), plus "... n more".

Gold labels are never read into the output: the logs' ``gold`` / ``gold_aux`` / ``oracle_tool`` fields are ignored.
Step numbers in the figure are 1-based (log step + 1). The per-call cost and latency are the ledger entries
(``calls``) of the decision model's ``tool_select`` call at the accepted step and of the LLM's ``tool_select`` call at
the deferred step. The threshold is the profile's tau-hat* for Jev's ``tool_select`` with Flash
(``system2.gemini-3.5-flash.decision_points.tool_select.tau``), checked against the episode's own logged config.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/stage2_jevgym/jevagent_best.jsonl"
PROFILE = ROOT / "configs/harness/jevagent_best.yaml"
OUT_TEX = ROOT / "paper/generated/fig1_episode.tex"
PREFIX = "fig1"
S2_MODEL = "gemini-3.5-flash"
MAX_STEPS = 6
TOP_K = 3
TASK_MAX_CHARS = 72
SEED_RE = re.compile(r"-(\d+)-")


# ---------------------------------------------------------------- selection
def seed_of(task_id: str) -> int:
    m = SEED_RE.search(task_id)
    return int(m.group(1)) if m else sys.maxsize


def _tool_decision(step: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for d in step.get("decisions") or []:
        if d.get("dp") == "tool_select" and d.get("mode") == "cascade":
            return d
    return None


def _llm_calls(ep: Mapping[str, Any], step: Mapping[str, Any]) -> int:
    """LLM calls of one step: the step's own counter, else the ledger's System-2 entries for that step."""
    if step.get("llm_calls") is not None:
        return int(step["llm_calls"])
    return sum(1 for c in ep.get("calls") or [] if c.get("component") == "s2" and c.get("step") == step.get("step"))


def deferred_steps(ep: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [s for s in ep.get("steps") or [] if (d := _tool_decision(s)) is not None and d.get("deferred")]


def accepted_steps(ep: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Accepted tool decisions in steps that ran without any LLM call."""
    return [s for s in ep.get("steps") or []
            if (d := _tool_decision(s)) is not None and not d.get("deferred") and _llm_calls(ep, s) == 0]


def qualifies(ep: Mapping[str, Any], max_steps: int = MAX_STEPS) -> bool:
    return (bool((ep.get("result") or {}).get("success")) and not ep.get("error")
            and 0 < len(ep.get("steps") or []) <= max_steps
            and bool(deferred_steps(ep)) and bool(accepted_steps(ep)))


def pick_steps(ep: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """(deferred step, accepted step) of a qualifying episode, by rule 3 of the module docstring."""
    defer = min(deferred_steps(ep), key=lambda s: (float(_tool_decision(s).get("s1_conf") or 0.0), s["step"]))
    acc = accepted_steps(ep)
    after = [s for s in acc if s["step"] > defer["step"]]
    return defer, (after[0] if after else acc[-1])


def select_episode(episodes: Iterable[Mapping[str, Any]], max_steps: int = MAX_STEPS) -> Mapping[str, Any]:
    eps = sorted(episodes, key=lambda e: (seed_of(e.get("task_id", "")), e.get("task_id", "")))
    for ep in eps:
        if qualifies(ep, max_steps):
            return ep
    raise SystemExit("[episode_to_tikz] no episode satisfies the selection rule")


def top_options(d: Mapping[str, Any], k: int = TOP_K) -> list[tuple[str, float]]:
    dist = d.get("s1_dist") or {}
    order = {o: i for i, o in enumerate(d.get("options") or list(dist))}
    ranked = sorted(dist.items(), key=lambda kv: (-float(kv[1]), order.get(kv[0], len(order))))
    return [(name, float(p)) for name, p in ranked[:k]]


# ---------------------------------------------------------------- formatting
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s: str) -> str:
    return "".join(_TEX.get(ch, ch) for ch in s)


def short_task(state: str, max_chars: int = TASK_MAX_CHARS) -> str:
    """The first sentence of the task statement (the ``TASK:`` line of the decision state), cut at a word boundary."""
    m = re.search(r"TASK:\s*(.+)", state or "")
    text = m.group(1).strip() if m else (state or "").strip().splitlines()[0]
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    if len(first) <= max_chars:
        return first
    cut = first[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"


def fmt_p(p: float) -> str:
    """A probability for display: two decimals, '<0.01' below 0.005 (a near-zero option never prints as 0.00)."""
    if p < 0.005:
        return r"$<$0.01"
    return f"{p:.2f}"


def fmt_sig(x: float, sig: int = 2) -> str:
    if x == 0:
        return "0"
    digits = max(sig - 1 - int(math.floor(math.log10(abs(x)))), 0)
    return f"{x:.{digits}f}"


def ledger_call(ep: Mapping[str, Any], component: str, step: int, dp: str = "tool_select") -> Mapping[str, Any]:
    for c in ep.get("calls") or []:
        if c.get("component") == component and c.get("step") == step and c.get("decision_point") == dp:
            return c
    raise SystemExit(f"[episode_to_tikz] no {component} {dp} call at step {step} in the ledger of {ep.get('task_id')}")


def profile_tauhat(profile: Mapping[str, Any], s2: str = S2_MODEL, dp: str = "tool_select") -> float:
    return float(profile["system2"][s2]["decision_points"][dp]["tau"])


# ---------------------------------------------------------------- keys
def figure_keys(ep: Mapping[str, Any], tauhat: float) -> dict[str, str]:
    logged = (((ep.get("meta") or {}).get("config") or {}).get("tool_select") or {}).get("tau")
    if logged is not None and abs(float(logged) - tauhat) > 1e-9:
        raise SystemExit(f"[episode_to_tikz] profile tau-hat* {tauhat} != the episode's logged tau {logged}")
    defer, acc = pick_steps(ep)
    keys: dict[str, str] = {}

    def put(k: str, v: Any) -> None:
        keys[f"{PREFIX}/{k}"] = str(v)

    steps = ep["steps"]
    first = _tool_decision(steps[0]) or {}
    put("task_id", tex_escape(ep["task_id"]))
    put("seed", seed_of(ep["task_id"]))
    put("task", tex_escape(short_task(first.get("state", ""))))
    put("n_steps", len(steps))
    put("n_options", len(first.get("options") or []))
    put("tauhat", f"{tauhat:.2f}")
    n_model = 0
    for i, s in enumerate(steps, start=1):
        d = _tool_decision(s)
        by = "s2" if (d is None or d.get("deferred")) else "s1"
        n_model += by == "s1"
        put(f"step/{i}/tool", tex_escape(str(s.get("tool"))))
        put(f"step/{i}/by", by)
    put("n_by_model", n_model)
    for tag, s in (("defer", defer), ("accept", acc)):
        d = _tool_decision(s)
        put(f"{tag}/step", int(s["step"]) + 1)
        put(f"{tag}/conf", f"{float(d['s1_conf']):.2f}")
        put(f"{tag}/chosen", tex_escape(str(d["final_value"])))
        opts = top_options(d)
        for k, (name, p) in enumerate(opts, start=1):
            put(f"{tag}/opt/{k}/name", tex_escape(name))
            put(f"{tag}/opt/{k}/p", fmt_p(p))
            put(f"{tag}/opt/{k}/pnum", f"{p:.4f}")
        put(f"{tag}/n_more", len(d.get("options") or []) - len(opts))
        put(f"{tag}/llm_calls", _llm_calls(ep, s))
    put("accept/n_args_model", sum(1 for d in acc.get("decisions") or []
                                    if d.get("dp") == "tool_params" and not d.get("deferred")))
    s1 = ledger_call(ep, "s1", int(acc["step"]))
    s2 = ledger_call(ep, "s2", int(defer["step"]))
    put("cost/s1", fmt_sig(float(s1["cost_usd"])))
    put("cost/s2", fmt_sig(float(s2["cost_usd"])))
    put("latency/s1", f"{float(s1['latency_ms']):.0f}")
    put("latency/s2", f"{float(s2['latency_ms']):.0f}")
    put("cost_ratio", f"{float(s2['cost_usd']) / float(s1['cost_usd']):.0f}")
    put("latency_ratio", f"{float(s2['latency_ms']) / float(s1['latency_ms']):.1f}")
    return keys


def to_tex(keys: Mapping[str, str], source: str, generated: str) -> str:
    lines = [f"% AUTO-GENERATED by scripts/episode_to_tikz.py -- do not edit by hand (rule: the script's docstring).",
             f"% generated {generated}; source {source}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(keys.items())] + [r"\makeatother", ""]
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(RUN), help="trajectory file of the selected profile (jsonl)")
    ap.add_argument("--profile", default=str(PROFILE), help="profile YAML holding tau-hat*")
    ap.add_argument("--out", default=str(OUT_TEX), help="\\defres file to write")
    a = ap.parse_args(argv)
    import yaml

    profile = yaml.safe_load(Path(a.profile).read_text())
    ep = select_episode(read_jsonl(Path(a.run)))
    keys = figure_keys(ep, profile_tauhat(profile))
    try:
        src = Path(a.run).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        src = Path(a.run).name
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_tex(keys, f"{src} task {ep['task_id']}", now))
    print(f"[episode_to_tikz] {ep['task_id']}: deferred step {keys[f'{PREFIX}/defer/step']}, accepted step "
          f"{keys[f'{PREFIX}/accept/step']} -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
