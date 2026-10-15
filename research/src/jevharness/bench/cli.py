"""``jevbench`` command line.

  jevbench run      configs/experiments/jevgym_tool_select.yaml [--out runs] [--max-tasks 10]
  jevbench report   runs/jevgym_tool_select            # episode + decision tables, Pareto front
  jevbench decide   configs/decision/bfcl.yaml          # static decision-level evaluation
  jevbench ping     --s1 '{"type": "jev"}'              # sanity-check a System-1 backend
  jevbench profile  configs/harness/jevagent_default.yaml [--s2 gemini-3.5-flash] [--tau2]   # a profile as it resolves
  jevbench gold     {replay,audit,h2,ceiling,coverage} ...  # meta-decision golds (bench/meta_gold.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _cmd_run(a: argparse.Namespace) -> None:
    from .runner import run_experiment

    exp = yaml.safe_load(Path(a.config).read_text())
    if a.override:
        for kv in a.override:
            k, v = kv.split("=", 1)
            node = exp
            parts = k.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = yaml.safe_load(v)
    paths = run_experiment(exp, a.out, max_tasks=a.max_tasks, retry_errors=False if a.no_retry_errors else None)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))


def _cmd_profile(a: argparse.Namespace) -> None:
    """Print a JevAgentHarness profile as one System-2 section resolves (docs/JEVAGENT_HARNESS.md §3)."""
    from ..harness import profile as hp
    from ..harness.config import HarnessConfig

    prof = hp.load_profile(a.path)
    secs = [a.s2] if a.s2 else sorted(prof.get("system2") or {})
    out: dict = {"name": prof.get("name"), "status": prof.get("status"), "sha256": prof["_sha256"], "sections": {}}
    for s2 in secs:
        dps, harness, skip = hp.decision_points(prof, s2, part=a.part, backend=a.backend, env=a.env)
        cfg = HarnessConfig(name=prof.get("name") or "profile", **{**harness, **dps}).validate_modes()
        sec = {"decision_points": dps, "harness": harness, "skip": skip, "short": cfg.short()}
        if a.tau2:
            sec["tau2"], sec["tau2_dropped"] = hp.tau2_rendering(dps)
        if a.harnesses:
            ext, env, notes = hp.pi_rendering(dps)
            sec["pi"] = {"extensions": [f"adapters/pi/{e}" for e in ext], "env": env, "notes": notes}
            arm, notes = hp.mini_swe_rendering(dps)
            sec["mini_swe"] = {"arm": arm, "notes": notes}
        out["sections"][s2] = sec
    print(json.dumps(out, indent=1))


def _cmd_report(a: argparse.Namespace) -> None:
    from .report import report

    print(report(a.run_dir, fmt=a.format))


def _cmd_decide(a: argparse.Namespace) -> None:
    from .decision_eval import run_decision_eval

    exp = yaml.safe_load(Path(a.config).read_text())
    res = run_decision_eval(exp, a.out, max_items=a.max_items)
    print(json.dumps(res, indent=1))


def _cmd_ping(a: argparse.Namespace) -> None:
    from ..decision import Choice, Noul, Score, make_backend

    b = make_backend(json.loads(a.s1))
    r = b.decide("A customer writes: 'I was charged twice for order ORD-1234, please fix it!'", [
        Choice("What is this ticket about?", ["billing", "shipping", "technical", "other"]),
        Noul("Is the customer upset?"),
        Score("How urgent is this ticket?", scale=(1, 5)),
    ])
    for ans in r.answers:
        print(ans.kind.value, ans.value, round(ans.confidence, 3), ans.distribution)
    print(r.usage.model_dump())


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["gold"]:     # pass-through: python -m jevharness.bench.meta_gold ...
        from .meta_gold import main as gold_main

        gold_main(argv[1:])
        return
    p = argparse.ArgumentParser("jevbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--out", default="runs")
    r.add_argument("--max-tasks", type=int, default=None)
    r.add_argument("--override", "-o", nargs="*", action="extend", default=[],
                   help="dotted.key=value overrides of the YAML (repeatable: -o a=1 -o b=2)")
    r.add_argument("--no-retry-errors", action="store_true",
                   help="on resume, keep errored episodes instead of re-running them")
    r.set_defaults(fn=_cmd_run)
    rp = sub.add_parser("report")
    rp.add_argument("run_dir")
    rp.add_argument("--format", choices=["md", "latex", "json"], default="md")
    rp.set_defaults(fn=_cmd_report)
    d = sub.add_parser("decide")
    d.add_argument("config")
    d.add_argument("--out", default="runs/decision")
    d.add_argument("--max-items", type=int, default=None)
    d.set_defaults(fn=_cmd_decide)
    pf = sub.add_parser("profile", help="show a JevAgentHarness profile (configs/harness/*.yaml) as it resolves")
    pf.add_argument("path")
    pf.add_argument("--s2", default=None, help="System-2 section (default: every section)")
    pf.add_argument("--part", default=None, choices=["best_single"])
    pf.add_argument("--backend", default=None, help="apply this System-1's threshold overrides (e.g. jevagent_rl)")
    pf.add_argument("--env", default=None, choices=["alfworld"])
    pf.add_argument("--tau2", action="store_true", help="also print the tau^2 rendering (A7.1 (i))")
    pf.add_argument("--harnesses", action="store_true", help="also print the pi (extensions + env) and mini-SWE-agent "
                                                             "(jev arm block) renderings")
    pf.set_defaults(fn=_cmd_profile)
    pg = sub.add_parser("ping")
    pg.add_argument("--s1", default='{"type": "jev"}')
    pg.set_defaults(fn=_cmd_ping)
    sub.add_parser("gold", help="meta-decision golds: replay | audit | h2 | ceiling | coverage (see bench/meta_gold.py)")
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
