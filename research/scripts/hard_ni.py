#!/usr/bin/env python3
"""Paired non-inferiority outcomes for the rows of the paper's Table 1 (tab:mainresults) that ran after Stage 1.

Descriptive and post hoc (author, 2026-10-07: every row of Table 1 shows its n, its paired difference and the result of
the same one-sided non-inferiority test the registered claims use, here outside the confirmatory tests). For each row
the paired win / loss counts and n are read from the \\res keys that the experiments' own scripts already wrote to
paper/generated/*.tex (wins = tasks only the delegated arm solved, losses = tasks only the reference solved), and

  * p  = metrics.ni_certified_p(wins, losses, n, margin = 0.03): the A6.21 decisive p-value of H0: loss of at least
         3 points of task success (rule bb_ordered, valid at every n); "yes" when p <= 0.05;
  * lo, hi = the Tango (1998) paired 95 % score interval of the difference (metrics.tango_interval), in points.

No model is called and no record is re-read; a row whose keys are missing is skipped (its table cells print "--").
Writes paper/generated/hard_ni.tex (keys hardni/<row>/{n,wins,losses,delta,lo,hi,p,ni} and hardni/margin).

  python scripts/hard_ni.py            # rerun after any of the source keys changes (e.g. E7f extended to more tasks)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench.metrics import ni_certified_p, tango_interval  # noqa: E402

MARGIN = 0.03
ALPHA = 0.05
# the E7f arm, in the order the paper's \efArm picks it (sections/08_results.tex)
E7F_ARMS = ("retrained", "rl_tier", "rl_refit", "tier", "jevagent_best_rl")
# row -> (wins key, losses key, n key)
ROWS = {
    "e7b_jev": ("e7b/jevagent_best/wins", "e7b/jevagent_best/losses", "e7b/jevagent_best/n"),
    "e7b_rl": ("e7b/jevagent_best_rl/wins", "e7b/jevagent_best_rl/losses", "e7b/jevagent_best_rl/n"),
    "e7e_jev": ("e7e/jev_refit/wins", "e7e/jev_refit/losses", "e7e/jev_refit/n"),
    "e4": ("e4/jevagent/vs_deep/wins", "e4/jevagent/vs_deep/losses", "e4/n_eval"),
    "e4b": ("e4b/jevagent/vs_deep/wins", "e4b/jevagent/vs_deep/losses", "e4b/n_eval"),
    # E4r (Deviations 2026-10-07 "E4r"): E4's 100 held-out instances pooled with 200 new ones
    "e4r": ("e4r/jevagent/vs_deep/wins", "e4r/jevagent/vs_deep/losses", "e4r/n_eval"),
    "e6a": ("e6a/cascade/wins", "e6a/cascade/losses", "e6a/cascade/n"),
    "e11": ("e11/cascade/wins", "e11/cascade/losses", "e11/cascade/n"),
    # E23 (Deviations 2026-10-08 "E23: the E11 checkpoint in ALFWorld's unseen rooms"): the same cascade on valid_unseen
    "e23": ("e23/cascade/wins", "e23/cascade/losses", "e23/cascade/n"),
    # E19 (Deviations 2026-10-08 "E19: the rule-randomised model on DecisionGym-Hard"): E12b's checkpoint live on E7f's
    # 200 tasks, paired with E7f's logged LLM-only arm
    "e19": ("e19/rl_random/wins", "e19/rl_random/losses", "e19/rl_random/n"),
    # E25 (Deviations 2026-10-09 "E25"): per-step routing in pi on Terminal-Bench 2.0 and in SWE-agent on SWE-bench
    # Verified, against the unmodified harness (scripts/explore_e25_harnesses.py writes the same p as e25/<h>/jev_router/ni_p)
    "e25_tb": ("e25/tb/jev_router/wins", "e25/tb/jev_router/losses", "e25/tb/jev_router/n"),
    "e25_swe": ("e25/swe/jev_router/wins", "e25/swe/jev_router/losses", "e25/swe/jev_router/n"),
    # E13 (Deviations 2026-10-07 "E13: other agent LLMs on DecisionGym-Hard"): the refitted Jev arm per agent LLM; a row
    # whose LLM has not run has no keys and is skipped
    **{f"e13_{llm}": (f"e13/{llm}/jev_refit/wins", f"e13/{llm}/jev_refit/losses", f"e13/{llm}/jev_refit/n")
       for llm in ("luna", "deepseek")},
}
_DEFRES = re.compile(r"\\defres\{([^}]*)\}\{([^}]*)\}")


def read_keys(gen_dir: Path) -> dict[str, str]:
    keys: dict[str, str] = {}
    for f in sorted(gen_dir.glob("*.tex")):
        if f.name == "hard_ni.tex":
            continue
        for k, v in _DEFRES.findall(f.read_text(encoding="utf-8")):
            keys.setdefault(k, v)
    return keys


def _int(v: str) -> int:
    return int(v.replace("{,}", "").replace(",", "").replace("$", "").strip())


def rows_for(keys: dict[str, str]) -> dict[str, tuple[str, str, str]]:
    rows = dict(ROWS)
    for arm in E7F_ARMS:
        if f"e7f/{arm}/wins" in keys:
            rows["e7f"] = (f"e7f/{arm}/wins", f"e7f/{arm}/losses", f"e7f/{arm}/n")
            break
    return rows


def compute(keys: dict[str, str], margin: float = MARGIN, alpha: float = ALPHA) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row, (kw, kl, kn) in rows_for(keys).items():
        if not all(k in keys for k in (kw, kl, kn)):
            continue
        w, l, n = _int(keys[kw]), _int(keys[kl]), _int(keys[kn])
        r = ni_certified_p(w, l, n, margin)
        ci = tango_interval(n - w - l, w, l, 0)["ci"]
        out[row] = {"n": n, "wins": w, "losses": l, "delta": 100.0 * (w - l) / n, "lo": 100.0 * ci[0],
                    "hi": 100.0 * ci[1], "p": r["p"], "ni": "yes" if r["p"] <= alpha else "no"}
    return out


def _signed(x: float) -> str:
    s = f"{x:+.1f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def to_tex(res: dict[str, dict], margin: float = MARGIN) -> str:
    lines = ["% AUTO-GENERATED by scripts/hard_ni.py -- do not edit by hand.",
             "% Paired non-inferiority outcomes for Table 1's rows run after Stage 1 (descriptive, post hoc; margin "
             f"{100 * margin:.0f} points, A6.21 rule bb_ordered; Tango 95 % intervals).",
             "\\makeatletter",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}",
             f"\\defres{{hardni/margin}}{{{100 * margin:.0f}}}"]
    for row in sorted(res):
        r = res[row]
        lines += [f"\\defres{{hardni/{row}/n}}{{{r['n']}}}", f"\\defres{{hardni/{row}/wins}}{{{r['wins']}}}",
                  f"\\defres{{hardni/{row}/losses}}{{{r['losses']}}}",
                  f"\\defres{{hardni/{row}/delta}}{{{_signed(r['delta'])}}}",
                  f"\\defres{{hardni/{row}/lo}}{{{_signed(r['lo'])}}}",
                  f"\\defres{{hardni/{row}/hi}}{{{_signed(r['hi'])}}}",
                  f"\\defres{{hardni/{row}/p}}{{{r['p']:.3f}}}", f"\\defres{{hardni/{row}/ni}}{{{r['ni']}}}"]
    lines.append("\\makeatother")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen-dir", type=Path, default=ROOT / "paper" / "generated")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = compute(read_keys(a.gen_dir))
    out = a.out or a.gen_dir / "hard_ni.tex"
    out.write_text(to_tex(res), encoding="utf-8")
    for row, r in sorted(res.items()):
        print(f"{row:8s} n={r['n']:4d} W={r['wins']:3d} L={r['losses']:3d} delta={r['delta']:+6.1f} "
              f"[{r['lo']:+6.1f}, {r['hi']:+6.1f}] p={r['p']:.3f} NI={r['ni']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
