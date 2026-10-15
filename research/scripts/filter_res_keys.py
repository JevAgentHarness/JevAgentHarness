#!/usr/bin/env python3
r"""Keep only the result keys the paper uses -> paper/generated/results.tex (MAIN_RUN_PLAN step 10).

`python -m jevharness.bench.paper_numbers runs` emits one key per number of every run under runs/ (~77k keys once
the Stage-1 main runs exist); \input-ing all of them overflows TeX's pool. This script reads the full key set (the
``--json`` output of paper_numbers) and writes the \defres file for the keys the paper can use:

1. **literal keys**: every ``\res{...}`` / ``\resm{...}`` / ``\resdec{...}`` key of the paper sources after the
   paper's own wrapper macros are expanded (``paper_numbers.used_keys``; comments and ``generated/`` ignored);
2. **prefixes of templated keys**: a key inside a macro body that still holds a parameter and starts with literal
   text (``\res{primaries/#1/#2/delta}``) keeps every key that starts with its literal prefix up to the last ``/``
   before the first parameter (``primaries/``);
3. **templated suffixes**: a templated key that starts with a parameter (``\res{#3/#4/success}``) keeps every key
   that ends in its literal suffix after the last parameter (``/success``) **and** whose run is a Stage-1 main run
   (``MAIN_RUNS``: the run directories of MAIN_RUN_PLAN steps 1-9).

Values are copied unchanged (the formatting is paper_numbers'). Usage::

    python -m jevharness.bench.paper_numbers runs --json runs/paper_numbers_full.json --check paper
    python scripts/filter_res_keys.py runs/paper_numbers_full.json --paper paper --out paper/generated/results.tex
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import paper_numbers as pn  # noqa: E402

# Stage-1 main-run directories under runs/ (docs/MAIN_RUN_PLAN.md §1 steps 1-9; paper_numbers names a run by its path)
MAIN_RUNS = (
    "e2a_jevgym_shadow_cal",             # A6.5 Jev calibration arms (seeds 5000-5149)
    "e2a_jevgym_shadow_cal_intern",      # step 1: Intern-Decision calibration arms
    "e1_bfcl_when2call",                 # step 2: E1 (C8)
    "e2a_jevgym_tool_decisions",         # step 3: E2a
    "e2b_jevgym_meta_control",           # step 5: E2b
    "e2c_jevgym_baselines",              # step 6: E2c
    "e2a_jevgym_tool_decisions_intern",  # step 7: E2a'
    "tau2/airline", "tau2/airline_test", "tau2/airline_shadow", "tau2/retail",  # steps 8-9: tau^2
)
PARAM = re.compile(r"#\d")
WORD = re.compile(r"[A-Za-z0-9]")


def _sources(paper: Path) -> dict[Path, str]:
    files = [paper] if paper.is_file() else sorted(p for p in paper.rglob("*.tex") if "generated" not in p.parts)
    return {f: re.sub(r"(?<!\\)%.*", "", f.read_text(errors="ignore")) for f in files}


def templated_keys(paper: str | Path) -> set[str]:
    """Keys of the paper sources that still contain a macro parameter after the wrapper macros are expanded
    (the bodies of the paper's own result macros)."""
    texts = _sources(Path(paper))
    macros: dict[str, tuple[int, str]] = {}
    for t in texts.values():
        macros.update(pn._macros(t))
    out: set[str] = set()
    for t in texts.values():
        t = re.sub(r"\\(?:re)?newcommand\{?\\\w+\}?(?:\[\d\])?", "", t)  # as paper_numbers.used_keys
        out.update(k.strip() for k in pn.KEY_RE.findall(pn._expand(t, macros)) if "#" in k)
    return out


def rules(templates: Iterable[str]) -> tuple[set[str], set[str]]:
    """(prefixes, suffixes) of the templated keys: a template with literal text before its first parameter gives the
    prefix up to the last '/' before it; one that starts with a parameter gives the literal suffix after its last
    parameter. Pure pass-through templates (``#1``, ``#1/#2``) give nothing."""
    prefixes: set[str] = set()
    suffixes: set[str] = set()
    for k in templates:
        head = PARAM.split(k, 1)[0]
        prefix = head[:head.rfind("/") + 1] if "/" in head else ""
        if prefix and WORD.search(prefix):
            prefixes.add(prefix)
            continue
        tail = PARAM.split(k)[-1]
        if WORD.search(tail):
            suffixes.add(tail)
    return prefixes, suffixes


def is_main_run_key(key: str, main_runs: Iterable[str] = MAIN_RUNS) -> bool:
    return any(key.startswith(r + "/") for r in main_runs)


def filter_keys(values: Mapping[str, str], literal: Iterable[str], prefixes: Iterable[str], suffixes: Iterable[str],
                main_runs: Iterable[str] = MAIN_RUNS) -> tuple[dict[str, str], dict[str, int]]:
    """The kept key -> value map and the number of keys each rule kept (a key counts under the first rule that
    keeps it: literal, prefix, suffix)."""
    literal, prefixes, suffixes, main_runs = set(literal), tuple(prefixes), tuple(suffixes), tuple(main_runs)
    kept: dict[str, str] = {}
    counts = {"literal": 0, "prefix": 0, "suffix": 0}
    for k, v in values.items():
        if k in literal:
            rule = "literal"
        elif prefixes and k.startswith(prefixes):
            rule = "prefix"
        elif suffixes and k.endswith(suffixes) and is_main_run_key(k, main_runs):
            rule = "suffix"
        else:
            continue
        kept[k] = v
        counts[rule] += 1
    return kept, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("full_json", help="paper_numbers --json output ({provenance, values})")
    ap.add_argument("--paper", default=str(ROOT / "paper"), help="paper directory (or one .tex file)")
    ap.add_argument("--out", required=True, help="filtered \\defres file, e.g. paper/generated/results.tex")
    ap.add_argument("--json-out", help="also write the kept {provenance, values} here")
    ap.add_argument("--main-run", action="append", default=None,
                    help=f"Stage-1 main run directory for the suffix rule (repeatable; default: {', '.join(MAIN_RUNS)})")
    a = ap.parse_args(argv)
    full = json.loads(Path(a.full_json).read_text())
    values, prov = full["values"], full.get("provenance", [])
    main_runs = tuple(a.main_run or MAIN_RUNS)
    literal = set(pn.used_keys(a.paper))
    prefixes, suffixes = rules(templated_keys(a.paper))
    kept, counts = filter_keys(values, literal, prefixes, suffixes, main_runs)
    tex = pn.to_tex(kept, prov).split("\n")
    note = (f"% filtered by scripts/filter_res_keys.py: {len(kept)} of {len(values)} keys (literal {counts['literal']}, "
            f"templated prefix {counts['prefix']}, templated suffix on a Stage-1 main run {counts['suffix']}); "
            f"prefixes: {', '.join(sorted(prefixes)) or '-'}; suffixes: {', '.join(sorted(suffixes)) or '-'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(tex[:2] + [note] + tex[2:]))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"provenance": prov, "values": kept}, indent=1, sort_keys=True))
    missing = sorted(k for k in literal if k not in values)
    print(f"[filter_res_keys] kept {len(kept)} of {len(values)} keys: {counts}; {len(prefixes)} prefix(es), "
          f"{len(suffixes)} suffix(es); {len(missing)} literal key(s) without a value", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
