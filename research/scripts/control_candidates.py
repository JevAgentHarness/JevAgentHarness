#!/usr/bin/env python3
"""The exhaustive candidate table of `best_control` (docs/PREREG_A7_STAGE2.md A7.14 (a)), generated from the registered
Stage-1 configs and committed as configs/harness/control_candidates.yaml; `tests/test_control_candidates.py` fails when
the committed table and the configs drift apart. No runtime discovery adds candidates.

A row is every registered **Flash** arm of E2a / E2b / E2c (threshold-sweep arms included) whose System-1 is neither
Jev nor Intern-Decision and that acts at a JevGym-applicable decision point (tool, stop, reflection, routing, memory in
a System-1 mode: replace, cascade, assist, guard, trigger), plus `small_only` (Flash-Lite alone, E2b). Per row: config,
arm, reference (`llm_only` of the same config), expected task count, System-1 spec, the decision-point setting, the
intervention type and its activation statistic, and duplicate-setting precedence (the same System-1 spec and setting in
another config: the larger registered n is read, then E2a, E2b, E2c; the others are listed with `duplicate_of`).

    python scripts/control_candidates.py            # (re)write configs/harness/control_candidates.yaml
    python scripts/control_candidates.py --check    # exit 1 when the committed table is stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench.runner import arm_task_counts  # noqa: E402

OUT = ROOT / "configs/harness/control_candidates.yaml"
CONFIGS = [("E2a", "jevgym_tool_select.yaml"), ("E2b", "jevgym_meta_control.yaml"), ("E2c", "baselines_jevgym.yaml")]
FLASH = "gemini-3.5-flash"
GROUPS = {"tool_select": "tool", "tool_params": "tool", "stop": "stop", "reflection": "reflection",
          "routing": "routing", "memory_write": "memory", "memory_read": "memory"}
S1_MODES = ("replace", "cascade", "assist", "guard", "trigger")
EXCLUDED_S1 = ("jev", "intern_decision")
# activation statistic per intervention type (A7.14 (a): read from the control's own intervention mechanism)
ACTIVATION = {
    "replace": "delegation (A6.20 (e) / A6.22 (a) 5): tool-decision or argument coverage >= 5 %",
    "cascade": "delegation (A6.20 (e) / A6.22 (a) 5): tool-decision or argument coverage >= 5 %",
    "assist": "strict candidate reductions: >= 1 executed assist decision offering fewer options than the full set",
    "guard": "vetoes: >= 1 executed guard decision that rejected the proposal",
    "trigger": "state changes: >= 1 executed trigger decision that changed the default (fired; routing: a step "
               "routed away from the main model)",
    "llm_swap": "void (the whole episode runs on the other model)",
}


def _expand(exp: dict[str, Any]) -> list[dict[str, Any]]:
    arms = [dict(a) for a in exp.get("arms", [])]
    for sw in exp.get("sweeps", []) or []:
        path = sw["set"].split(".")
        for v in sw["values"]:
            arm = json.loads(json.dumps(sw["base"]))
            node = arm
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = v
            arm["name"] = f"{sw['base']['name']}_{path[-1]}{v}"
            arms.append(arm)
    return arms


def rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, cfg in CONFIGS:
        exp = yaml.safe_load((ROOT / "configs/experiments" / cfg).read_text())
        counts = arm_task_counts(exp)
        default_n = int(exp.get("n_tasks") or (exp.get("suite") or {}).get("n") or 0)
        llms = exp.get("llms") or {}
        backends = exp.get("s1_backends") or {}
        for a in _expand(exp):
            name = a["name"]
            model = (llms.get(a["llm"]) or {}).get("model") if a.get("llm") else (exp.get("llm") or {}).get("model")
            n = int(counts.get(name, default_n))
            if name == "small_only":
                out.append({"config": f"configs/experiments/{cfg}", "run": key, "arm": name, "reference": "llm_only",
                            "n": n, "system2": model, "s1": None, "setting": {}, "groups": [],
                            "intervention": "llm_swap", "activation": ACTIVATION["llm_swap"]})
                continue
            if model != FLASH:
                continue
            spec = backends.get(a["s1"]) if a.get("s1") else exp.get("s1")
            stype = str((spec or {}).get("type", ""))
            if stype in EXCLUDED_S1:
                continue
            setting = {dp: c for dp, c in a.items() if dp in GROUPS and isinstance(c, dict)
                       and str(c.get("mode")) in S1_MODES}
            if not setting:
                continue
            modes = sorted({str(c["mode"]) for c in setting.values()})
            kind = next(m for m in S1_MODES if m in modes)      # replace > cascade > assist > guard > trigger
            out.append({"config": f"configs/experiments/{cfg}", "run": key, "arm": name, "reference": "llm_only",
                        "n": n, "system2": model, "s1": {k: v for k, v in (spec or {}).items() if k != "placeholder"},
                        "s1_name": a.get("s1"), "placeholder": bool((spec or {}).get("placeholder")),
                        "setting": {dp: {k: v for k, v in c.items()} for dp, c in a.items() if dp in GROUPS},
                        "groups": sorted({GROUPS[dp] for dp in setting}), "intervention": kind,
                        "activation": ACTIVATION[kind]})
    # duplicate-setting precedence: same System-1 spec and setting -> larger n, then E2a, E2b, E2c
    order = {k: i for i, (k, _) in enumerate(CONFIGS)}
    seen: dict[str, dict[str, Any]] = {}
    for r in sorted(out, key=lambda r: (-r["n"], order[r["run"]])):
        sig = json.dumps({"s1": r["s1"], "setting": r["setting"], "system2": r["system2"]}, sort_keys=True)
        if sig in seen:
            r["duplicate_of"] = f"{seen[sig]['run']}:{seen[sig]['arm']}"
        else:
            seen[sig] = r
    return out


def render(rs: list[dict[str, Any]]) -> str:
    head = ("# best_control candidate table (docs/PREREG_A7_STAGE2.md A7.14 (a)), GENERATED by scripts/control_candidates.py\n"
            "# from the registered Stage-1 configs; do not edit by hand (tests/test_control_candidates.py checks it).\n"
            "# Rows with `duplicate_of` repeat another row's System-1 spec and setting and are not read (precedence: larger\n"
            "# registered n, then E2a, E2b, E2c). select_profile.py screens every other row (complete, NI, cheaper,\n"
            "# activation) in the same single run as A7.1 and runs the qualifying control with the largest saving.\n")
    body = {"kind": "control_candidates", "registered": "PREREG_A7_STAGE2 A7.14", "activation": ACTIVATION,
            "n_rows": len(rs), "n_read": sum(1 for r in rs if "duplicate_of" not in r), "rows": rs}
    return head + yaml.safe_dump(body, sort_keys=False, width=120)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = render(rows())
    if a.check:
        ok = OUT.exists() and OUT.read_text() == text
        print("up to date" if ok else f"{OUT} is stale: re-run scripts/control_candidates.py", file=sys.stderr)
        return 0 if ok else 1
    OUT.write_text(text)
    d = yaml.safe_load(text)
    print(f"wrote {OUT}: {d['n_rows']} rows, {d['n_read']} read", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
