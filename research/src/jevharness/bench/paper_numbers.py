"""Generate the paper's result numbers from ``runs/`` -- the only way numbers enter the paper.

    python -m jevharness.bench.paper_numbers runs --out paper/generated/results.tex [--json FILE]
    python -m jevharness.bench.paper_numbers runs --check paper/        # which \\res{} keys are missing

Every run directory under ``runs/`` becomes a set of keys ``<run>/<...>/<metric>``:

  decision runs (``summary.json`` from ``jevbench decide``)
      <run>/<backend>/<dp>/<source>/<metric>       e.g. e1_bfcl_when2call/jev/tool_select/bfcl/acc
  episode runs (``<arm>.jsonl`` trajectories from ``jevbench run``) -- the score card
      <run>/<arm>/<metric>                         e.g. e2a_jevgym_tool_decisions/jev_ts_replace/success
      <run>/<arm>/vs_baseline/<metric>             paired deltas, savings and p-values vs. llm_only
      <run>/<arm>/decisions/<dp/mode>/<metric>     System-1 accuracy / calibration per decision point
      <run>/headline/...                           iso-quality savings, hypervolume, sweeps (APGR, AIQ)

The .tex output defines ``\\res{key}``, which prints the formatted number, or ``\\tbd`` (the red ??)
when the key is missing, so a table can be written once and fills itself as runs land. Percent-like
metrics are printed as percentages without the % sign (put it in the column header).

Runs whose experiment uses a simulated LLM or System-1 are skipped unless ``--include-simulated``,
and even then their keys are prefixed ``SIM/`` so they can never fill a headline cell.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import yaml

from .scorecard import SIM_LLM_PROVIDERS, SIM_S1_TYPES, compute_scorecard, load_runs, run_meta

PCT = {"acc", "success", "s1_acc", "final_acc", "accepted_acc", "s2_acc", "acc_gold_in_candidates", "fill_rate",
       "fill_precision", "defer_correct", "defer_rate", "cost_savings", "latency_savings", "delegation_ratio",
       "llm_free_steps", "llm_free_step_ratio", "harmful_delegation", "harmful_delegation_rate", "rescue_rate",
       "pgr", "apgr", "selective_acc", "selective_accuracy", "far", "frr", "savings", "best_savings", "ratio",
       "wrong_rate", "harmful_rate", "catch_rate", "rescue_given_catch", "hull_success", "baseline_success",
       "accuracy", "em", "f1", "judge_acc", "recall_at_k", "in_candidates", "loss_rate_ub"}
PCT_PARENTS = {"delegation_by_dp", "label_share", "pass_hat_k"}
PCT_CI_PARENTS = {"success_ci", "cost_savings_ci"}
# signed percentage points: success differences and their bounds, including the A6 NI block of the score card
# (vs_baseline/ni/{diff, ci95, lcb95, lcb_adj, tango_lcb95, tango_lcb_adj, mde/delta_mde}), which used to print raw
SIGNED_PCT = {"delta", "delta_ci", "delta_success", "lcb", "diff", "ci95", "lcb95", "lcb_adj", "tango_lcb95",
              "tango_lcb_adj", "delta_mde"}
TEXT = {"arm", "best_cost_of_pass"}
THREE_DP = {"ece", "ace", "ece_adaptive", "brier", "aurc", "eaurc", "e_aurc", "aiq", "aiq_zero_router", "hv_contribution",
            "all", "baseline_only"}
TWO_DP = {"auroc", "latency_s", "latency_of_pass", "steps", "llm_calls", "s1_calls"}
INT = {"n", "n_gold", "n_tasks", "n_episodes", "n_paired", "errors", "dominates", "dominated_by"}
KEY_RE = re.compile(r"\\res\{([^}]*)\}")
NI_MARGINS = {"default": 0.03, "tau2": 0.05}  # pre-registered non-inferiority margins (success, absolute)


# ------------------------------------------------------------------ formatting
def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return None if math.isnan(f) else f
    return None


def _sig(v: float, digits: int = 3) -> str:
    return np.format_float_positional(v, precision=digits, unique=True, fractional=False, trim="-")


def fmt(path: tuple[str, ...], v: float) -> str:
    """Format one number for the paper according to its metric name (the last path segment)."""
    last, parent = path[-1], (path[-2] if len(path) > 1 else "")
    if math.isinf(v):
        return r"$\infty$" if v > 0 else r"$-\infty$"
    if parent in SIGNED_PCT or last in SIGNED_PCT:
        return f"{100 * v:+.1f}"
    if last in PCT or parent in PCT_PARENTS or (parent in PCT_CI_PARENTS and last in {"lo", "hi"}) or \
            last.startswith(("cpt", "sel_acc")):
        return f"{100 * v:.1f}"
    if last.startswith("p_") or last.endswith("_p") or last == "p":
        return "$<$0.001" if v < 0.001 else f"{v:.3f}"
    if last in THREE_DP:
        return f"{v:.3f}"
    if last in TWO_DP:
        return f"{v:.2f}"
    if last in INT or last.startswith("n_"):
        return f"{v:.0f}"
    if last.endswith("_ms"):
        return f"{v:.0f}"
    if "cost" in last or last.endswith("_usd"):
        return _sig(v, 3)
    return _sig(v, 3)


def _seg(k: Any) -> str:
    """A key segment safe inside \\csname: no '/', braces, backslashes or TeX specials."""
    return re.sub(r"[{}\\#%$&~^ ]", "", str(k).replace("/", "-").replace("@", "at"))


def _tex_text(s: str) -> str:
    return s.replace("\\", "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _flatten(x: Any, path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(x, Mapping):
        for k, v in x.items():
            yield from _flatten(v, path + (_seg(k),))
    elif isinstance(x, (list, tuple)) and len(x) == 2 and all(_num(e) is not None for e in x) and \
            (path[-1].endswith("_ci") or path[-1] in ("ref", "ci95")):   # ci95: the A6 NI block's two-sided CI
        yield path + ("lo",), x[0]
        yield path + ("hi",), x[1]
    elif _num(x) is not None or (isinstance(x, str) and path and path[-1] in TEXT):
        yield path, x


# ------------------------------------------------------------------ collection
def _is_simulated_decision_run(run_dir: Path) -> bool:
    exp_path = run_dir / "experiment.yaml"
    if not exp_path.exists():
        return False
    try:
        exp = yaml.safe_load(exp_path.read_text()) or {}
    except yaml.YAMLError:
        return False
    s1 = [s for s in (exp.get("backends") or {}).values() if isinstance(s, dict)]
    llms = [s for s in (exp.get("fc_llms") or {}).values() if isinstance(s, dict)]
    return any(str(s.get("type", "")).lower() in SIM_S1_TYPES for s in s1) or \
        any(str(s.get("provider", "")).lower() in SIM_LLM_PROVIDERS for s in llms)


def _is_trajectory_file(p: Path) -> bool:
    try:
        with p.open() as f:
            first = json.loads(next((ln for ln in f if ln.strip()), "{}"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(first, dict) and "task_id" in first and "result" in first


def collect_run(run_dir: Path, baseline: str = "llm_only", n_boot: int = 2000) -> tuple[dict[str, Any], bool]:
    """Nested numbers for one run directory, and whether it is simulated."""
    summ = run_dir / "summary.json"
    if summ.exists():
        data = json.loads(summ.read_text())
        if isinstance(data, dict) and data and all(k.count("|") == 2 for k in data):  # decision-level (L1)
            out: dict[str, Any] = {}
            for key, row in data.items():
                backend, dp, src = key.split("|")
                out.setdefault(backend, {}).setdefault(dp, {})[src] = row
            return out, _is_simulated_decision_run(run_dir)
        if isinstance(data, dict):  # other suites (e.g. memory QA) write their own nested summary
            return data, run_meta(run_dir).get("simulated", False)
    arms = {p.stem: p for p in sorted(run_dir.glob("*.jsonl")) if _is_trajectory_file(p)}
    runs = {k: v for k, v in load_runs(run_dir).items() if k in arms and v}
    if not runs:
        return {}, False
    meta = run_meta(run_dir, runs)
    # headline claims are non-inferiority tests with the pre-registered margin (docs/PREREGISTRATION.md)
    suite = meta.get("suite")
    suite = str((suite.get("name") if isinstance(suite, Mapping) else suite) or "")
    is_tau2 = suite.startswith("tau2") or any((t.get("meta") or {}).get("tau2") for v in runs.values() for t in v[:1])
    margin = NI_MARGINS["tau2"] if is_tau2 else NI_MARGINS["default"]
    exp_base = None
    try:
        exp_base = (yaml.safe_load((run_dir / "experiment.yaml").read_text()) or {}).get("baseline")
    except (OSError, yaml.YAMLError):
        pass
    # the run's own reference arm: CLI choice, else experiment.yaml `baseline:`, else the known LLM-only names
    base = next((b for b in (baseline, exp_base, "llm_only", "llm_react") if b and b in runs), None)
    card = compute_scorecard(runs, baseline=base, meta=meta, n_boot=n_boot, criterion="noninferior", margin=margin)
    out = {}
    for arm, row in card["arms"].items():
        out[arm] = dict(row)
    out["headline"] = card["headline"]
    return out, bool(meta.get("simulated"))


def collect(runs_root: str | Path, include_simulated: bool = False, baseline: str = "llm_only",
            n_boot: int = 2000) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """key -> formatted value for every run directory below ``runs_root`` (searched recursively)."""
    root = Path(runs_root)
    dirs = sorted({p.parent for p in root.rglob("summary.json")} | {p.parent for p in root.rglob("*.jsonl")})
    values: dict[str, str] = {}
    prov: list[dict[str, Any]] = []
    for d in dirs:
        rel = d.relative_to(root).as_posix() if d != root else d.name
        try:
            nested, sim = collect_run(d, baseline=baseline, n_boot=n_boot)
        except Exception as e:  # a broken run must not take the others down
            prov.append({"run": rel, "error": f"{type(e).__name__}: {e}"})
            continue
        if not nested:
            continue
        prov.append({"run": rel, "simulated": sim, "included": include_simulated or not sim})
        if sim and not include_simulated:
            continue
        prefix = ("SIM",) if sim else ()
        for path, v in _flatten(nested, prefix + tuple(_seg(s) for s in rel.split("/"))):
            values["/".join(path)] = _tex_text(v) if isinstance(v, str) else fmt(path, float(v))
    return values, prov


# ------------------------------------------------------------------ output
def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def to_tex(values: Mapping[str, str], prov: list[dict[str, Any]]) -> str:
    lines = ["% AUTO-GENERATED by `python -m jevharness.bench.paper_numbers` -- do not edit by hand.",
             f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git {_git_head()}"]
    for p in prov:
        state = p.get("error") or ("simulated, " if p.get("simulated") else "") + \
            ("included" if p.get("included") else "skipped")
        lines.append(f"% run {p['run']}: {state}")
    lines += [r"\makeatletter",
              r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
              r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(values.items())]
    lines += [r"\makeatother", ""]
    return "\n".join(lines)


def _brace_arg(text: str, i: int) -> tuple[str, int] | None:
    """The balanced {...} group starting at text[i] (after optional whitespace) and the index after it."""
    while i < len(text) and text[i] in " \t\n%":
        i = text.index("\n", i) + 1 if text[i] == "%" else i + 1
    if i >= len(text) or text[i] != "{":
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "\\":  # skip the escaped character (\{, \}, and the second \ of \\)
            j += 2
            continue
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j], j + 1
        j += 1
    return None


def _macros(text: str) -> dict[str, tuple[int, str]]:
    """\newcommand{\name}[n]{body} definitions."""
    out: dict[str, tuple[int, str]] = {}
    for m in re.finditer(r"\\(?:re)?newcommand\{?\\(\w+)\}?(?:\[(\d)\])?", text):
        body = _brace_arg(text, m.end())
        if body:
            out[m.group(1)] = (int(m.group(2) or 0), body[0])
    return out


def _expand(text: str, macros: dict[str, tuple[int, str]], depth: int = 6) -> str:
    """Expand the user macros that (transitively) produce \res{} keys, so wrapped keys can be checked."""
    names = [n for n, (_, b) in macros.items() if "\\res{" in b]
    changed = True
    while changed:  # transitive closure: macros whose body calls a key-producing macro
        changed = False
        for n, (_, b) in macros.items():
            if n not in names and any(f"\\{m}" in b for m in names):
                names.append(n)
                changed = True
    for _ in range(depth):
        hit = False
        for n in names:
            nargs, body = macros[n]
            for m in reversed(list(re.finditer(rf"\\{n}(?![A-Za-z])", text))):
                args, j = [], m.end()
                for _k in range(nargs):
                    a = _brace_arg(text, j)
                    if a is None:
                        break
                    args.append(a[0])
                    j = a[1]
                if len(args) != nargs:
                    continue
                rep = re.sub(r"#(\d)", lambda g: args[int(g.group(1)) - 1] if int(g.group(1)) <= nargs else g.group(0),
                             body)  # simultaneous: an argument may itself contain "#2"
                text = text[:m.start()] + rep + text[j:]
                hit = True
        if not hit:
            break
    return text


def used_keys(tex_root: str | Path) -> dict[str, list[str]]:
    """\res{} keys used in the paper sources (through wrapper macros too) -> files using them."""
    root = Path(tex_root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*.tex") if "generated" not in p.parts)
    texts = {f: re.sub(r"(?<!\\)%.*", "", f.read_text(errors="ignore")) for f in files}  # drop comments
    macros: dict[str, tuple[int, str]] = {}
    for t in texts.values():
        macros.update(_macros(t))
    used: dict[str, list[str]] = {}
    for f, t in texts.items():
        t = re.sub(r"\\(?:re)?newcommand\{?\\\w+\}?(?:\[\d\])?", "", t)  # drop definitions (keep bodies harmless)
        for k in KEY_RE.findall(_expand(t, macros)):
            if "#" not in k and k.strip():
                used.setdefault(k.strip(), []).append(f.as_posix())
    return used


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help="runs root (every sub-directory with summary.json or *.jsonl is a run)")
    ap.add_argument("--out", help="write the \\res{} macro file here (e.g. paper/generated/results.tex)")
    ap.add_argument("--json", help="also write key -> value JSON here")
    ap.add_argument("--check", help="paper directory / .tex file: list \\res{} keys that have no value")
    ap.add_argument("--list", action="store_true", help="print every available key")
    ap.add_argument("--include-simulated", action="store_true", help="emit simulated runs under SIM/ keys")
    ap.add_argument("--baseline", default="llm_only")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    values, prov = collect(a.runs, include_simulated=a.include_simulated, baseline=a.baseline, n_boot=a.n_boot)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(to_tex(values, prov))
    if a.json:
        Path(a.json).write_text(json.dumps({"provenance": prov, "values": values}, indent=1, sort_keys=True))
    if a.list:
        for k in sorted(values):
            print(f"{k} = {values[k]}")
    for p in prov:
        if "error" in p:
            print(f"[paper_numbers] {p['run']}: {p['error']}", file=sys.stderr)
    skipped = [p["run"] for p in prov if p.get("simulated") and not p.get("included")]
    print(f"[paper_numbers] {len(values)} numbers from {sum(1 for p in prov if p.get('included'))} runs"
          + (f"; skipped simulated: {', '.join(skipped)}" if skipped else ""), file=sys.stderr)
    if a.check:
        used = used_keys(a.check)
        missing = {k: fs for k, fs in used.items() if k not in values}
        print(f"[paper_numbers] {len(used) - len(missing)}/{len(used)} \\res keys filled", file=sys.stderr)
        for k, fs in sorted(missing.items()):
            print(f"MISSING {k}  ({', '.join(sorted(set(fs)))})")


if __name__ == "__main__":
    main()
