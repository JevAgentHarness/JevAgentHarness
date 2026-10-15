"""Success, cost and time on Terminal-Bench 2.0 (test split): each harness on the strong model alone vs routed.

    python bench/plot_results.py summary.json --out assets/results  [--harness pi,mswe,dsh] [--arm jah-auto-t75]

Reads a `bench/summarize.py --json` file that holds `<harness>:jah-strong` and `<harness>:<arm>` rows and writes
`<out>-light.png` and `<out>-dark.png` (for a <picture> element in the README).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"pi": "pi", "mswe": "mini-swe-agent", "dsh": "DeepSeek Harness", "t2": "Terminus-2", "codex": "Codex"}
THEMES = {
    "light": {"bg": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e0", "base": "#8a8984",
              "ours": "#2a78d6"},
    "dark": {"bg": "#0d1117", "ink": "#f0f6fc", "ink2": "#c3c2b7", "grid": "#30363d", "base": "#8b949e",
             "ours": "#3987e5"},
}


def draw(rows: list[tuple[str, dict, dict]], theme: str, out: str) -> None:
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(10, 0.75 * len(rows) + 1.6), dpi=200,
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.patch.set_facecolor(c["bg"])
    panels = [("Tasks solved (%)", lambda r: 100 * r["success"], "{:.0f}%"),
              ("LLM cost per task (USD)", lambda r: r["cost_per_task"], "${:.2f}"),
              ("Agent time per task (min)", lambda r: r["agent_s_per_task"] / 60, "{:.1f}")]
    ys = list(range(len(rows)))[::-1]
    for ax, (title, f, fmt) in zip(axes, panels):
        ax.set_facecolor(c["bg"])
        vals = []
        for y, (h, base, ours) in zip(ys, rows):
            a, b = f(base), f(ours)
            vals += [a, b]
            ax.plot([a, b], [y, y], color=c["grid"], lw=3, zorder=1, solid_capstyle="round")
            ax.scatter([a], [y], s=80, color=c["base"], zorder=3, edgecolors=c["bg"], linewidths=1.5)
            ax.scatter([b], [y], s=95, color=c["ours"], zorder=4, edgecolors=c["bg"], linewidths=1.5)
            lo, hi = (a, b) if a <= b else (b, a)
            ax.annotate(fmt.format(lo), xy=(lo, y), xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=9, color=c["ink2"])
            ax.annotate(fmt.format(hi), xy=(hi, y), xytext=(8, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=9, color=c["ink2"])
        span = max(vals) - min(vals)
        ax.set_xlim(min(vals) - 0.45 * span, max(vals) + 0.45 * span)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(title, color=c["ink"], fontsize=11, loc="left")
        ax.set_yticks(ys)
        ax.set_yticklabels([NAMES.get(h, h) for h, _, _ in rows] if ax is axes[0] else [], color=c["ink"])
        ax.tick_params(axis="x", colors=c["ink2"], labelsize=9)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(c["grid"])
        ax.grid(True, axis="x", color=c["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].scatter([], [], s=80, color=c["base"], label="strong model only")
    axes[0].scatter([], [], s=95, color=c["ours"], label="JevAgentHarness (routed)")
    leg = fig.legend(loc="upper right", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.99, 1.0))
    for t in leg.get_texts():
        t.set_color(c["ink"])
    fig.suptitle("Terminal-Bench 2.0, 74 test tasks per harness", x=0.01, ha="left", color=c["ink"], fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{out}-{theme}.png", facecolor=c["bg"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default="assets/results")
    ap.add_argument("--harness", default="pi,mswe,dsh")
    ap.add_argument("--arm", default="jah-auto-t75")
    a = ap.parse_args()
    s = json.load(open(a.summary))
    rows = [(h, s[f"{h}:jah-strong"], s[f"{h}:{a.arm}"]) for h in a.harness.split(",")]
    for theme in THEMES:
        draw(rows, theme, a.out)


if __name__ == "__main__":
    main()
