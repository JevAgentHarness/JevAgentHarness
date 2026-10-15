"""Turn a run directory into paper-ready tables (markdown / LaTeX / JSON).

Markdown and JSON reports end with a compact *score card* (``jevharness.bench.scorecard``: iso-quality
cost savings, cost-of-pass, paired tests vs. the ``llm_only`` baseline, Pareto / hypervolume, sweep
APGR/AIQ, decision-level delegation and calibration). The full card, with a weak reference for PGR /
APGR and LaTeX output, comes from ``python -m jevharness.bench.scorecard RUN_DIR``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..accounting import PriceTable, default_prices
from ..harness.trajectory import load_trajectories
from .metrics import decision_records, decision_table, episode_table, fmt_row, pareto_front

EP_KEYS = ["n", "success", "success_ci_lo", "success_ci_hi", "cost_usd", "s1_cost_usd", "latency_s", "llm_calls",
           "s1_calls", "llm_input_tokens", "steps", "errors", "retry_rate"]
# A6 ruling (e): ``retry_rate`` = share of the arm's logged calls (acting and measurement, System-1 and System-2) that
# needed more than one HTTP attempt, over the calls whose client counted attempts (``CallRecord.attempts``; older logs
# and backends without the field are left out, ``retry_known_frac`` says how many were known); ``errors`` (episodes
# that ended in an infrastructure error) stays the error-rate column
RETRY_KEYS = ["retry_rate", "retry_known_frac", "retry_wait_s"]
# prompt-cache sensitivity columns (results/pilot_jev/CACHE_NOTES.md): ``cost_usd`` stays the primary figure (the
# ledger's cost: what the provider bills, cached tokens at ``cached_input_per_m`` when the provider reported them);
# ``cost_usd_cache_modelled`` re-prices the MEASURED ``cached_input_tokens`` of the System-2 calls with the current
# price table (equal to ``cost_usd`` unless the table changed since the run, e.g. a cached rate added later);
# ``cost_usd_no_cache`` is the list price with no cache at all; ``cached_frac`` the measured cached share of
# System-2 input tokens. No assumed hit rate anywhere.
CACHE_KEYS = ["cost_usd_cache_modelled", "cost_usd_no_cache", "cached_frac"]
# s1_acc: tool selection is judged against the acceptable tool set AT(s) when logged (s1_acc_canonical: against the
# canonical next tool; docs/META_GOLD_SPEC.md §3.5)
DEC_KEYS = ["n", "n_gold", "s1_acc", "s1_acc_canonical", "accepted_acc", "defer_rate", "s2_acc", "final_acc", "ece",
            "brier", "auroc"]


def collect(run_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    return {p.stem: load_trajectories(p) for p in sorted(Path(run_dir).glob("*.jsonl"))}


def episode_cache_costs(t: dict[str, Any], prices: PriceTable | None = None) -> dict[str, float]:
    """One episode's prompt-cache sensitivity figures from its per-call ledger (``calls``), measured tokens only.

    System-2 records (``component == "s2"``, measurement calls tagged ``meta.shadow`` excluded exactly as
    ``usage.total`` excludes them) are re-priced with ``prices`` (default: ``configs/prices``) from their logged
    ``input_tokens`` / ``output_tokens`` / ``meta.cached_input_tokens``; every other record keeps its recorded cost.
    Unpriced System-2 models keep their recorded cost (0). Returns ``cost_usd_cache_modelled`` (cached tokens at the
    model's ``cached_input_per_m``), ``cost_usd_no_cache`` (every input token at ``input_per_m``: the list price
    without any cache), ``cached_frac`` (cached / input tokens of the System-2 calls, 0 when there are none) and
    ``cached_tokens``."""
    prices = prices or default_prices()
    total = float(((t.get("usage") or {}).get("total") or {}).get("cost_usd", 0.0) or 0.0)
    recorded = modelled = no_cache = 0.0
    inp = cached = 0
    for c in t.get("calls") or []:
        if c.get("component") != "s2" or (c.get("meta") or {}).get("shadow"):
            continue
        i, o = int(c.get("input_tokens") or 0), int(c.get("output_tokens") or 0)
        k = int((c.get("meta") or {}).get("cached_input_tokens") or 0)
        inp += i
        cached += k
        cost = float(c.get("cost_usd") or 0.0)
        recorded += cost
        price = prices.get(c.get("model"))
        if price is None:
            modelled += cost
            no_cache += cost
        else:
            modelled += price.cost(i, o, k)
            no_cache += price.cost(i, o, 0)
    return {"cost_usd_cache_modelled": total - recorded + modelled, "cost_usd_no_cache": total - recorded + no_cache,
            "cached_frac": cached / inp if inp else 0.0, "cached_tokens": float(cached)}


def episode_cache_columns(trajs: list[dict[str, Any]], prices: PriceTable | None = None) -> dict[str, float]:
    """Per-arm means of :func:`episode_cache_costs` (the ``CACHE_KEYS`` columns of the episode table)."""
    if not trajs:
        return {}
    rows = [episode_cache_costs(t, prices) for t in trajs]
    return {k: sum(r[k] for r in rows) / len(rows) for k in CACHE_KEYS + ["cached_tokens"]}


def episode_retry_columns(trajs: list[dict[str, Any]]) -> dict[str, float]:
    """Per-arm retry figures from the per-call ledgers (A6 ruling (e)): ``retry_rate`` = retried calls / calls with a
    known attempt count, ``retry_known_frac`` = calls with a known count / all calls, ``retry_wait_s`` = mean backoff
    slept per episode (seconds; excluded from the latency figures by design)."""
    n_all = n_known = n_retried = 0
    wait = 0.0
    for t in trajs:
        for c in t.get("calls") or []:
            n_all += 1
            a = c.get("attempts")
            if a is not None:
                n_known += 1
                n_retried += int(int(a) > 1)
            wait += float(c.get("retry_wait_ms") or 0.0)
    nan = float("nan")
    return {"retry_rate": n_retried / n_known if n_known else nan,
            "retry_known_frac": n_known / n_all if n_all else nan,
            "retry_wait_s": wait / 1000.0 / len(trajs) if trajs else nan}


def _scorecard(run_dir: str | Path, runs: dict[str, list[dict[str, Any]]], baseline: str | None,
               weak: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """The score card never breaks the classic report: failures become a one-line note."""
    try:
        from .scorecard import compute_scorecard, run_meta

        return compute_scorecard(runs, baseline=baseline, weak=weak, meta=run_meta(run_dir, runs)), None
    except Exception as e:  # pragma: no cover - defensive
        return None, f"{type(e).__name__}: {e}"


def report(run_dir: str | Path, fmt: str = "md", scorecard: bool = True, baseline: str | None = "llm_only",
           weak: str | None = None) -> str:
    runs = collect(run_dir)
    ep = {k: {**episode_table(v), **episode_cache_columns(v), **episode_retry_columns(v)} for k, v in runs.items() if v}
    dec = {k: decision_table(decision_records(v)) for k, v in runs.items() if v}
    front = pareto_front([(k, r["cost_usd"], r["success"]) for k, r in ep.items()])
    card, card_err = _scorecard(run_dir, runs, baseline, weak) if scorecard and ep and fmt != "latex" else (None, None)
    if fmt == "json":
        doc: dict[str, Any] = {"episode": ep, "decision": dec, "pareto": front}
        if card is not None:
            from .scorecard import _clean

            doc["scorecard"] = _clean(card)
        return json.dumps(doc, indent=1)
    if fmt == "latex":
        lines = [r"\begin{tabular}{l" + "r" * 6 + "}", r"\toprule",
                 r"Config & Succ. (\%) & Cost (\$) & Latency (s) & LLM calls & S1 calls & Steps \\", r"\midrule"]
        for k, r in ep.items():
            star = r"$^\star$" if k in front else ""
            name = k.replace("_", r"\_")
            end = r" \\"
            lines.append(f"{name}{star} & {100 * r['success']:.1f} & {r['cost_usd']:.4f} & "
                         f"{r['latency_s']:.1f} & {r['llm_calls']:.1f} & {r['s1_calls']:.1f} & {r['steps']:.1f}"
                         + end)
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)
    keys = EP_KEYS + CACHE_KEYS
    out = ["## Episode metrics  (* = on cost/success Pareto front)", "",
           "| config | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for k, r in ep.items():
        out.append(f"| {k}{' *' if k in front else ''} | {fmt_row(r, keys)} |")
    out += ["", "_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when "
            "it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with "
            "the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list "
            "price, no cache; cached_frac: measured cached share of System-2 input tokens "
            "(results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP "
            "attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes "
            "that ended in an infrastructure error._"]
    out += ["", "## Decision-point metrics", ""]
    for k, table in dec.items():
        if not table:
            continue
        out += [f"### {k}", "", "| dp/mode | " + " | ".join(DEC_KEYS) + " |", "|" + "---|" * (len(DEC_KEYS) + 1)]
        for dp, r in table.items():
            out.append(f"| {dp} | {fmt_row(r, DEC_KEYS)} |")
        out.append("")
    if card is not None:
        from .scorecard import to_markdown

        out += ["", to_markdown(card, compact=True)]
    elif card_err:
        out += ["", f"_(score card unavailable: {card_err})_"]
    return "\n".join(out)
