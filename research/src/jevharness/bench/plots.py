"""Paper figures from run directories (matplotlib, print/light mode).

  python -m jevharness.bench.plots pareto   runs/jevgym_tool_select --out paper/figures/pareto_jevgym.pdf
  python -m jevharness.bench.plots pareto   runs/jevgym_tool_select --baseline llm_only_s2lite --out pareto_s2lite.pdf
  python -m jevharness.bench.plots sweep    runs/jevgym_tool_select --prefix jev_ts_cascade_tau --out paper/figures/tau_sweep.pdf
  python -m jevharness.bench.plots reliability runs/decision/bfcl_when2call --backends jev,openjev,qwen3_4b --out paper/figures/reliability.pdf

Style: validated categorical slots in fixed order (blue, orange, aqua), thin marks, one y-axis
per panel, recessive grid, direct labels on highlighted points, legend whenever >= 2 series.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..harness.trajectory import load_trajectories
from .metrics import episode_table, pareto_front

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (validated all-pairs, light)
MUTED = "#a3a29c"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e0"


def _style(ax: Any) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _episode_rows(run_dir: str | Path, with_llm: bool = False) -> Any:
    """arm -> episode_table row; with ``with_llm`` also arm -> System-2 model (the most common ``traj.llm``)."""
    from collections import Counter

    rows, llm = {}, {}
    for p in sorted(Path(run_dir).glob("*.jsonl")):
        t = load_trajectories(p)
        if t:
            rows[p.stem] = episode_table(t)
            c = Counter(str(x.get("llm")) for x in t if x.get("llm"))
            llm[p.stem] = c.most_common(1)[0][0] if c else None
    return (rows, llm) if with_llm else rows


S1_MODES = ("replace", "cascade", "assist", "guard", "trigger")
MARKERS = {"replace": "o", "cascade": "^", "assist": "s", "guard": "P", "trigger": "v", "mixed": "X"}
# Same semantics as the paper figures (paper/figures/style.tex): blue = the System-1 decision model under
# study, orange = the LLM (reserved for LLM-only), neutral greys / aqua for reference backends. Colours are
# fixed per backend name so a backend keeps its colour across every plot; no purple, no red/green-only cues.
BACKEND_COLORS = {"jev": SERIES[0], "default": SERIES[0], "jev_ts": SERIES[0], "openjev": "#1d4f91",
                  "lite": SERIES[2], "standin": SERIES[2], "embed": "#6b6a65", "bm25": "#a3a29c", "tfidf": "#c8c7c1"}
_FALLBACK_COLORS = ["#1d4f91", "#5c8fd6", "#3d3c39", "#8f8e88"]


def backend_color(name: str, i: int = 0) -> str:
    key = name.lower()
    for k, c in BACKEND_COLORS.items():
        if key == k or key.startswith(k):
            return c
    if key.startswith(("ours", "r")) and any(ch.isdigit() for ch in key):  # trained deciders: dark blue family
        return "#1d4f91"
    return _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]


def arm_profiles(run_dir: str | Path) -> dict[str, dict[str, str]]:
    """arm -> {"backend": System-1 backend name or "", "mode": integration mode / "mixed" / ""}, read from the
    run's experiment.yaml (sweeps expanded like the runner does). Empty dict if it cannot be read."""
    import yaml

    from ..harness.config import DP
    from .runner import arm_bindings, expand_arms

    path = Path(run_dir) / "experiment.yaml"
    try:
        exp = yaml.safe_load(path.read_text()) or {}
        cfgs, binds = expand_arms(exp), arm_bindings(exp)
    except Exception:
        return {}
    default = str((exp.get("s1") or {}).get("type") or "s1")
    out = {}
    for c in cfgs:
        all_modes = {c.dp(dp).mode.value for dp in DP}
        modes = sorted(all_modes & set(S1_MODES))
        mode = modes[0] if len(modes) == 1 else ("mixed" if modes else ("shadow" if "shadow" in all_modes else ""))
        out[c.name] = {"backend": binds.get(c.name, {}).get("s1", default) if modes else "", "mode": mode}
    return out


def pareto(run_dir: str | Path, out: str | Path, baseline: str = "llm_only", label_front: bool = True,
           s2: str | None = "baseline") -> None:
    """Cost (log) vs. success, one point per arm: colour = System-1 backend, marker = integration mode,
    orange diamond = LLM-only, grey = arms without System-1 decisions; dashed line = Pareto front.

    ``s2`` picks the System-2 family drawn (a results figure is one suite x System-2 model, like the score
    card's comparison families): ``"baseline"`` (default) = the baseline arm's model, so a run with Flash and
    Flash-Lite (``*_s2lite``) arms does not mix them on one front; a model name draws that family (pass its
    LLM-only arm as ``baseline``, e.g. ``--baseline llm_only_s2lite --s2 baseline``); ``"all"`` / None draws
    every arm. Arms left out are listed on stderr."""
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows, llm = _episode_rows(run_dir, with_llm=True)
    model = llm.get(baseline) if s2 == "baseline" else (None if s2 in (None, "all") else s2)
    if model is not None:
        dropped = sorted(k for k in rows if llm.get(k) not in (model, None))
        rows = {k: v for k, v in rows.items() if k not in dropped}
        if dropped:
            print(f"pareto: System-2 {model}: left out {len(dropped)} arm(s) of other System-2 models: "
                  f"{', '.join(dropped)} (--s2 all draws them)", file=sys.stderr)
    prof = arm_profiles(run_dir)
    # shadow-only arms are LLM-only episodes with System-1 logged: they would sit on top of the baseline
    rows = {k: v for k, v in rows.items() if (prof.get(k) or {}).get("mode") != "shadow"}
    front = set(pareto_front([(k, r["cost_usd"], r["success"]) for k, r in rows.items()]))
    backends = sorted({prof.get(k, {}).get("backend", "") for k in rows} - {""})
    color = {b: backend_color(b, i) for i, b in enumerate(backends)}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, r in rows.items():
        x, y = max(r["cost_usd"], 1e-6), 100 * r["success"]
        p = prof.get(k) or {}
        if k == baseline:
            ax.scatter(x, y, s=64, marker="D", color=SERIES[1], edgecolor="white", linewidth=1.2, zorder=4)
            ax.annotate("LLM-only", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=7, color=TEXT)
            continue
        c = color.get(p.get("backend", ""), MUTED)
        ax.scatter(x, y, s=46 if k in front else 30, marker=MARKERS.get(p.get("mode", ""), "o"), color=c,
                   edgecolor=TEXT if k in front else "white", linewidth=0.9 if k in front else 0.8,
                   alpha=1.0 if k in front else 0.8, zorder=3 if k in front else 2)
        if label_front and k in front:
            ax.annotate(k.replace("_", " "), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=5.5,
                        color=TEXT2)
    fr = sorted((max(rows[k]["cost_usd"], 1e-6), 100 * rows[k]["success"]) for k in front)
    if len(fr) > 1:
        ax.plot(*zip(*fr), color=TEXT2, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xscale("log")
    xs = [max(r["cost_usd"], 1e-6) for r in rows.values()] or [1.0]
    decades = np.log10(max(xs)) - np.log10(min(xs))
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0) if decades <= 1.5 else (1.0,)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())  # no overlapping 3x10^-3 labels
    ax.margins(x=0.15, y=0.12)
    ax.set_xlabel("Cost per episode (USD, log)", fontsize=8)
    ax.set_ylabel("Success (%)", fontsize=8)
    _style(ax)
    handles = [Line2D([], [], marker="D", color=SERIES[1], linestyle="", markersize=5, label="LLM-only")]
    handles += [Line2D([], [], marker="o", color=color[b], linestyle="", markersize=5, label=b) for b in backends]
    modes = [m for m in (*S1_MODES, "mixed") if any((prof.get(k) or {}).get("mode") == m for k in rows)]
    handles += [Line2D([], [], marker=MARKERS[m], color=TEXT2, linestyle="", markersize=5, label=m) for m in modes]
    handles.append(Line2D([], [], color=TEXT2, linestyle="--", linewidth=1.0, label="Pareto front"))
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.3),
              handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def sweep(run_dir: str | Path, prefix: str, out: str | Path, baseline: str = "llm_only") -> None:
    """Threshold sweep: two panels (success; LLM calls per episode) sharing the tau axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _episode_rows(run_dir)
    pts = []
    for k, r in rows.items():
        m = re.match(re.escape(prefix) + r"([0-9.]+)$", k)
        if m:
            pts.append((float(m.group(1)), r))
    pts.sort()
    if not pts:
        raise SystemExit(f"no runs matching {prefix}<tau> in {run_dir}")
    taus = [t for t, _ in pts]
    base = rows.get(baseline)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for ax, key, lab, scale in ((axes[0], "success", "Success (%)", 100), (axes[1], "llm_calls", "LLM calls / episode", 1)):
        ys = [scale * r[key] for _, r in pts]
        lo = [scale * r["success_ci_lo"] for _, r in pts] if key == "success" else None
        hi = [scale * r["success_ci_hi"] for _, r in pts] if key == "success" else None
        if lo:
            ax.fill_between(taus, lo, hi, color=SERIES[0], alpha=0.12, linewidth=0)
        ax.plot(taus, ys, color=SERIES[0], linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", label="S1 cascade")
        if base:
            ax.axhline(scale * base[key], color=SERIES[1], linewidth=1.5, linestyle="--", label="LLM-only")
        ax.set_xlabel(r"cascade threshold $\tau$", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def reliability_bins(decision_dir: str | Path, backend: str, dp: str | None = None,
                     n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-width confidence bins of one backend's decision_eval records: ``lo``, ``hi``, ``n``, mean
    confidence ``conf`` and accuracy ``acc`` (empty bins omitted)."""
    recs = [json.loads(line) for line in open(Path(decision_dir) / f"s1_{backend}.jsonl") if line.strip()]
    recs = [r for r in recs if "correct" in r and r.get("conf") is not None and (dp is None or r["dp"] == dp)]
    conf = np.array([r["conf"] for r in recs], dtype=float)
    corr = np.array([float(r["correct"]) for r in recs])
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if m.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "conf": float(conf[m].mean()),
                        "acc": float(corr[m].mean())})
    return out


def reliability(decision_dir: str | Path, backends: list[str], out: str | Path, dp: str | None = None,
                n_bins: int = 10, min_count: int = 5) -> dict[str, list[dict[str, float]]]:
    """Reliability diagram (confidence vs accuracy), <= 3 backends, from decision_eval records, with the
    number of decisions per bin underneath. Verbalised confidences pile up near 1 (E1 dev: 95-99% of the
    answers above 0.9), so the count panel shows how much of the data each point carries; bins with fewer
    than ``min_count`` decisions are drawn hollow and unconnected instead of being dropped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(3.2, 4.0), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="perfect calibration")
    edges = np.linspace(0, 1, n_bins + 1)
    table: dict[str, list[dict[str, float]]] = {}
    for i, b in enumerate(backends[:3]):
        bins = reliability_bins(decision_dir, b, dp, n_bins)
        table[b] = bins
        dense = [x for x in bins if x["n"] >= min_count]
        sparse = [x for x in bins if x["n"] < min_count]
        ax.plot([x["conf"] for x in dense], [x["acc"] for x in dense], color=SERIES[i], linewidth=2, marker="o",
                markersize=5, markeredgecolor="white", label=b)
        if sparse:
            ax.plot([x["conf"] for x in sparse], [x["acc"] for x in sparse], linestyle="none", marker="o",
                    markersize=5, markerfacecolor="white", markeredgecolor=SERIES[i])
        counts = np.zeros(n_bins)
        for x in bins:
            counts[int(round(x["lo"] * n_bins))] = x["n"]
        axh.step(edges, np.r_[counts, counts[-1]], where="post", color=SERIES[i], linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", fontsize=8)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    axh.set_yscale("symlog", linthresh=1)
    top = max([x["n"] for bins in table.values() for x in bins] or [1])
    axh.set_ylim(0, top * 3)
    axh.set_yticks([t for t in (0, 1, 10, 100, 1000, 10000) if t <= top * 3])
    axh.set_ylabel("Decisions", fontsize=8)
    axh.set_xlabel("Confidence", fontsize=8)
    _style(axh)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pareto")
    p.add_argument("run_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", default="llm_only")
    p.add_argument("--s2", default="baseline",
                   help="System-2 family to draw: 'baseline' (the baseline arm's model), a model name, or 'all'")
    s = sub.add_parser("sweep")
    s.add_argument("run_dir")
    s.add_argument("--prefix", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--baseline", default="llm_only", help="reference arm drawn as the dashed line")
    r = sub.add_parser("reliability")
    r.add_argument("decision_dir")
    r.add_argument("--backends", required=True)
    r.add_argument("--dp", default=None)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "pareto":
        pareto(a.run_dir, a.out, a.baseline, s2=a.s2)
    elif a.cmd == "sweep":
        sweep(a.run_dir, a.prefix, a.out, a.baseline)
    else:
        reliability(a.decision_dir, a.backends.split(","), a.out, a.dp)


if __name__ == "__main__":
    main()
