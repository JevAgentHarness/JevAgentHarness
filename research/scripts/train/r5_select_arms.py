"""Derive an experiment config with a subset of arms (by name prefix) and dotted overrides (GPU round 5).

    python scripts/train/r5_select_arms.py configs/experiments/jevgym_dev_rl.yaml runs/x.yaml \
        --prefix llm_only jev_ --set name=e2dev_rl_refs_api

The committed config stays the single source of the arm definitions; each run (references, one training stage, ...)
gets its own derived copy with its own `name`, so run directories never mix System-1 backends.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def set_dotted(d: dict, key: str, value: object) -> None:
    node = d
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--prefix", nargs="+", required=True,
                    help="keep arms whose name starts with one of these (a trailing '$' means the exact name)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides (values parsed as YAML)")
    a = ap.parse_args(argv)
    exp = yaml.safe_load(Path(a.config).read_text())
    def keep(name: str) -> bool:
        return any(name == p[:-1] if p.endswith("$") else name.startswith(p) for p in a.prefix)

    arms = [arm for arm in exp["arms"] if keep(arm["name"])]
    if not arms:
        raise SystemExit(f"no arm of {a.config} starts with {a.prefix}")
    exp["arms"] = arms
    exp["sweeps"] = []
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_dotted(exp, k, yaml.safe_load(v))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(exp, sort_keys=False))
    print(f"{a.out}: {len(arms)} arms ({', '.join(x['name'] for x in arms)})")


if __name__ == "__main__":
    main()
