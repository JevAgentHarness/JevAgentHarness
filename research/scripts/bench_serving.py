"""Serving benchmark for the single-pass typed decider: ms/decision and prefilled tokens, stateless vs incremental.

Modes (``training/serve_jev.Engine``):
  stateless  one full prefill per question (``JEV_SERVE_NO_CACHE=1``; the old server)
  prefix     the k questions of a request share one prefill of their common prefix (session cache off)
  session    prefix sharing + the episode cache: the next step of an episode prefills only its delta

Axes: model x state length (final-state tokens; synthetic JevGym-like episodes, or real per-step states replayed
from a run with ``--trajectories``) x questions per request {1, 3} x batch {1, 8} (independent episodes whose
step-t requests are submitted together: the stateless path pads them into one forward, the incremental paths
serve them one episode after the other). Warm-up, then ``--repeats`` replays of each episode; p50 / p95 over
per-step requests. Output: ``<out>/bench_serving.json`` and ``<out>/bench_serving.md``.

  python scripts/bench_serving.py --out results/bench_serving                       # GPU, default models
  python scripts/bench_serving.py --smoke --out /tmp/bench                           # CPU, tiny model, < 1 min
  python scripts/bench_serving.py --trajectories runs/pilot_jev_r2/e2a_jevgym_tool_decisions/llm_only.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_MODELS = ["Qwen/Qwen3-0.6B", "openbmb/MiniCPM5-1B", "Qwen/Qwen3.5-0.8B"]
DEFAULT_LENGTHS = [1000, 4000, 16000, 64000]
MODES = ["stateless", "prefix", "session"]

TOOLS = [("find_user", "Look a user up by name"), ("get_orders", "List a user's orders"),
         ("get_order", "Fetch one order by id"), ("refund", "Refund an order to the original payment method"),
         ("partial_refund", "Refund part of an order"), ("ship_order", "Ship a pending order"),
         ("cancel_order", "Cancel a pending order"), ("update_address", "Change a user's shipping address"),
         ("check_stock", "Check warehouse stock of an item"), ("create_ticket", "Open a support ticket"),
         ("ask_user", "Ask the customer a clarifying question"), ("send_email", "Send an email to the customer"),
         ("get_return_policy", "Read the return policy"), ("apply_coupon", "Apply a coupon to an order"),
         ("escalate_to_supervisor", "Hand the case to a supervisor"), ("verify_identity", "Verify the customer"),
         ("get_shipping_quote", "Quote shipping for an order"), ("restock_item", "Restock a returned item"),
         ("audit_log", "Read the account's audit log"), ("finish", "The task is complete")]
OBS = ['{"user_id": "U5563", "name": "Jia Wang", "tier": "basic"}',
       '[{"order_id": "ORD-3443", "user_id": "U5563", "item": "tablet", "status": "delivered", "amount": 458.24}]',
       "Customer: Please refund it to my original payment method.",
       "Refunded 458.24 via original_payment.", '{"status": "refunded", "order_id": "ORD-3443"}',
       '{"ticket_id": "T-1189", "status": "open"}', "Email sent to jia.wang@example.com."]


# ------------------------------------------------------------------ workloads
def wire_questions(n: int) -> dict[str, dict[str, Any]]:
    qs = {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
                   "criteria": {name: desc for name, desc in TOOLS}},
          "done": {"type": "noul", "instructions": "Is the task complete?"},
          "risk": {"type": "score", "instructions": "How risky is the next action?",
                   "criteria": ["none", "low", "medium", "high", "irreversible"]}}
    return dict(list(qs.items())[:n])


def synthetic_episode(tok: Any, target_tokens: int, n_steps: int, rng: random.Random, ep: int) -> list[str]:
    """JevGym-like per-step states (TASK / HISTORY with heartbeat noise / CURRENT OBSERVATION) whose last state has
    about ``target_tokens`` tokens; each step adds one history block of equal size."""
    task = (f"TASK: Refund the full amount of customer #{ep} Jia Wang's most recent delivered order (highest "
            "order id). Policy: never guess -- if a detail you need is missing, ask the customer.")
    line_tokens = max(1, len(tok("log: heartbeat ok #123456\n", add_special_tokens=False).input_ids))
    per_step = max(1, (target_tokens - 60) // max(1, n_steps - 1))
    states, hist = [], []
    for t in range(n_steps):
        name, _ = TOOLS[(t * 3 + ep) % (len(TOOLS) - 1)]
        obs = OBS[(t + ep) % len(OBS)]
        cur = f"\n\nCURRENT OBSERVATION:\n{obs}" if t else "\n\nCURRENT OBSERVATION:\nYou have access to the tools."
        states.append(task + (("\n\nHISTORY:\n" + "\n".join(hist)) if hist else "") + cur)
        n_lines = max(1, (per_step - 30) // line_tokens)
        noise = "\n".join(f"log: heartbeat ok #{rng.randint(0, 999999)}" for _ in range(n_lines))
        hist.append(f'[{t}] {name}({{"user_id": "U5563", "order_id": "ORD-{3000 + t}"}}) -> {obs}\n{noise}')
    return states


def trajectory_episodes(path: Path, max_episodes: int, questions: int) -> list[list[tuple[str, dict[str, Any]]]]:
    """Replay logged episodes: the distinct per-step states in order, each with the logged tool_select Choice
    (its real option list, > 52 options -> knockout) plus, for ``questions`` = 3, a Noul and a Score."""
    from jevharness.training.data import open_text

    eps = []
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            steps, prev = [], None
            for st in t["steps"]:
                for d in st["decisions"]:
                    s, opts = d.get("state"), d.get("options")
                    if not s or not opts or s == prev:
                        continue
                    q = {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
                                  "criteria": {o: None for o in opts}}}
                    if questions >= 3:
                        q.update({k: v for k, v in wire_questions(3).items() if k != "tool"})
                    steps.append((s, q))
                    prev = s
            if steps:
                eps.append(steps)
            if len(eps) >= max_episodes:
                break
    return eps


# ------------------------------------------------------------------ measurement
def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


def set_mode(engine: Any, mode: str, session: Any) -> None:
    engine.stateless = mode == "stateless"
    engine.session = session if mode == "session" else None
    if engine.session is not None:
        engine.session.clear()


def run_cell(engine: Any, episodes: list[list[tuple[str, dict[str, Any]]]], mode: str, session: Any,
             repeats: int, warmup: int) -> dict[str, Any]:
    """``episodes``: B lists of per-step (state, questions). One repeat = replay every step with the B step-t
    requests submitted as one ``answer_batch`` call. Returns p50 / p95 of ms per request and per decision, and the token
    accounting from the engine's usage report."""
    set_mode(engine, mode, session)
    n_steps = min(len(e) for e in episodes)
    B = len(episodes)
    nq = len(episodes[0][0][1])
    ms_req, ms_dec, prefilled, stateless, hits = [], [], 0, 0, 0
    phases: dict[str, float] = {}
    for rep in range(warmup + repeats):
        if engine.session is not None:
            engine.session.clear()
        if rep == warmup and getattr(engine, "profile", False):
            engine.phase_ms.clear()
        for t in range(n_steps):
            reqs = [ep[t] for ep in episodes]
            if engine.device == "cuda":
                import torch
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = engine.answer_batch(reqs)
            if engine.device == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if rep < warmup:
                continue
            ms_req.append(dt / B)
            ms_dec.append(dt / (B * nq))
            for _, u in res:
                prefilled += u["prefilled_tokens"]
                stateless += u["stateless_tokens"]
                hits += int(u["cache_hit"])
    n_dec = repeats * n_steps * B * nq
    if getattr(engine, "profile", False):
        phases = {k: v / (repeats * n_steps * B) for k, v in engine.phase_ms.items()}
    return {"mode": mode, "questions": nq, "batch": B, "steps": n_steps, "repeats": repeats,
            "phase_ms_per_request": phases or None,
            "ms_per_request": {"p50": pct(ms_req, 50), "p95": pct(ms_req, 95), "mean": statistics.fmean(ms_req)},
            "ms_per_decision": {"p50": pct(ms_dec, 50), "p95": pct(ms_dec, 95), "mean": statistics.fmean(ms_dec)},
            "prefilled_tokens_per_decision": prefilled / n_dec, "stateless_tokens_per_decision": stateless / n_dec,
            "reuse": (stateless / prefilled) if prefilled else float("nan"),
            "cache_hit_rate": hits / (repeats * n_steps * B)}


def load_engine(model: str, a: argparse.Namespace, cache_tokens: int) -> Any:
    from jevharness.training.serve_jev import Engine

    return Engine(model, name=Path(model).name, device=a.device, max_len=a.max_len, batch_size=max(a.batch) * 3,
                  session_cache=True, cache_tokens=cache_tokens, cache_entries=max(a.batch) * 4 + 8,
                  prefill_chunk=a.prefill_chunk, profile=bool(getattr(a, "profile", False)))


def bench_model(model: str, a: argparse.Namespace, out: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    try:
        engine = load_engine(model, a, a.cache_tokens)
    except Exception as e:  # e.g. Qwen3.5 (Qwen3_5ForConditionalGeneration) until load_causal_lm handles it
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        print(f"[bench_serving] SKIP {model}: {msg}", file=sys.stderr)
        out.append({"model": model, "error": msg})
        return
    info = engine.describe()
    meta.setdefault("engines", {})[model] = {k: info[k] for k in ("device", "dtype", "croppable", "cache_class",
                                                                     "layer_types", "bytes_per_token")}
    session = engine.session  # kept here: set_mode detaches it from the engine for the stateless / prefix cells
    bpt = max(1, info["bytes_per_token"])
    if a.max_cache_gb:
        cap = int(a.max_cache_gb * 2 ** 30 / bpt)
        if cap < session.max_tokens:
            session.max_tokens = cap
            meta["engines"][model]["cache_tokens_capped"] = cap
    tok = engine.tok
    rng = random.Random(0)
    workloads: list[tuple[str, int, list[list[tuple[str, dict[str, Any]]]]]] = []
    if a.trajectories:
        for nq in a.questions:
            eps = trajectory_episodes(Path(a.trajectories), max(a.batch), nq)
            if not eps:
                raise SystemExit(f"no per-step states in {a.trajectories}")
            for B in a.batch:
                reps = [eps[i % len(eps)] for i in range(B)]
                T = max(len(tok(s, add_special_tokens=False).input_ids) for ep in reps for s, _ in ep)
                workloads.append((f"traj:{Path(a.trajectories).stem}", T, reps))
    else:
        for T in a.lengths:
            for nq in a.questions:
                for B in a.batch:
                    eps = [[(s, wire_questions(nq)) for s in synthetic_episode(tok, T, a.steps, rng, b)]
                           for b in range(B)]
                    T_actual = max(len(tok(s, add_special_tokens=False).input_ids) for ep in eps for s, _ in ep)
                    workloads.append((f"synthetic:{T}", T_actual, eps))
    for source, T, eps in workloads:
        engine.batch_size = max(1, min(max(a.batch) * max(a.questions), a.max_batch_tokens // max(1, T)))
        for mode in a.modes:
            need = 2 * len(eps) * (T + 300)
            if mode == "session" and need > session.max_tokens:
                print(f"[bench_serving] note: session cache {session.max_tokens} tokens < {need} needed for "
                      f"{len(eps)} episodes of {T} tokens; expect evictions ({model} {source})", file=sys.stderr)
            cell = {"model": model, "source": source, "state_tokens": T}
            try:
                cell.update(run_cell(engine, eps, mode, session, a.repeats, a.warmup))
            except Exception as e:  # OOM etc.: record and continue
                cell.update({"mode": mode, "questions": len(eps[0][0][1]), "batch": len(eps),
                             "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"})
                traceback.print_exc()
                if engine.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
            cell["session_cache_tokens"] = session.max_tokens if mode == "session" else None
            cell["batch_size"] = engine.batch_size
            out.append(cell)
            print(json.dumps(cell), flush=True)
    del engine
    if a.device != "cpu":
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ------------------------------------------------------------------ report
def markdown(res: dict[str, Any]) -> str:
    rows = [c for c in res["cells"] if "error" not in c]
    lines = ["# Serving benchmark", "",
             f"device/dtype: {res['meta'].get('device')} / "
             f"{', '.join(sorted({e['dtype'] for e in res['meta'].get('engines', {}).values()}))}; "
             f"torch {res['meta']['torch']}, transformers {res['meta']['transformers']}; "
             f"repeats {res['meta']['args']['repeats']}, steps/episode {res['meta']['args']['steps']}", ""]
    for m, e in res["meta"].get("engines", {}).items():
        lines.append(f"- `{m}`: cache {e['cache_class']} {e['layer_types']}, croppable={e['croppable']}, "
                     f"{e['bytes_per_token'] / 1024:.1f} KB/token")
    for c in res["cells"]:
        if "error" in c and "mode" not in c:
            lines.append(f"- `{c['model']}`: skipped ({c['error']})")
    prof = any(c.get("phase_ms_per_request") for c in rows)
    lines += ["", "| model | source | state tok | q | B | mode | ms/decision p50 | p95 | ms/request p50 | "
              "prefilled tok/dec | stateless tok/dec | reuse x | speedup vs stateless |" + (" phases ms/request |" if prof else ""),
              "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|" + ("---|" if prof else "")]
    base = {(c["model"], c["source"], c["questions"], c["batch"]): c["ms_per_decision"]["p50"]
            for c in rows if c["mode"] == "stateless"}
    for c in rows:
        b = base.get((c["model"], c["source"], c["questions"], c["batch"]))
        sp = f"{b / c['ms_per_decision']['p50']:.2f}" if b else "-"
        lines.append(f"| {Path(c['model']).name} | {c['source']} | {c['state_tokens']} | {c['questions']} | "
                     f"{c['batch']} | {c['mode']} | {c['ms_per_decision']['p50']:.1f} | "
                     f"{c['ms_per_decision']['p95']:.1f} | {c['ms_per_request']['p50']:.1f} | "
                     f"{c['prefilled_tokens_per_decision']:.0f} | {c['stateless_tokens_per_decision']:.0f} | "
                     f"{c['reuse']:.1f} | {sp} |"
                     + ((" " + (" ".join(f"{k}={v:.1f}" for k, v in c["phase_ms_per_request"].items())
                                    if c.get("phase_ms_per_request") else "-") + " |")   # stateless: no phases
                        if prof else ""))
    errs = [c for c in res["cells"] if "error" in c and "mode" in c]
    if errs:
        lines += ["", "Failed cells:", ""] + [f"- {Path(c['model']).name} {c['source']} q={c['questions']} "
                                             f"B={c['batch']} {c['mode']}: {c['error']}" for c in errs]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS, help="final-state tokens")
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8], help="independent episodes per step")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--steps", type=int, default=6, help="steps per synthetic episode")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--trajectories", default=None, help="replay the per-step states of a run's JSONL")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=131072)
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--cache-tokens", type=int, default=1 << 20, help="session cache budget (tokens)")
    ap.add_argument("--max-cache-gb", type=float, default=8.0, help="cap the session cache by memory (0 = none)")
    ap.add_argument("--max-batch-tokens", type=int, default=262144,
                    help="stateless path: at most this many prompt tokens per padded forward")
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny random model, small lengths (< 1 min)")
    ap.add_argument("--profile", action="store_true",
                    help="per-phase wall time of the shared-prefix path (encode / lookup / prefill / store / suffix), "
                         "device-synchronised: adds a little latency, so not for the headline numbers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render-only", action="store_true",
                    help="rewrite bench_serving.md from an existing <out>/bench_serving.json (no model is loaded)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.render_only:
        res = json.loads((out / "bench_serving.json").read_text())
        (out / "bench_serving.md").write_text(markdown(res))
        print(f"[bench_serving] re-rendered {out / 'bench_serving.md'}")
        return
    if a.smoke:
        sys.path.insert(0, str(ROOT / "scripts/train"))
        from make_tiny_model import make_tiny_model

        a.models = [str(make_tiny_model(out / "tiny_model"))]
        a.lengths = [128, 256] if a.lengths == DEFAULT_LENGTHS else a.lengths
        a.batch = [1, 4] if a.batch == [1, 8] else a.batch
        a.steps, a.repeats, a.warmup, a.device = 3, 1, 1, "cpu"
        a.max_len = 4096
    import torch
    import transformers

    meta = {"torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(),
            "device": a.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and a.device != "cpu" else None,
            "args": vars(a), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke}
    cells: list[dict[str, Any]] = []
    for model in a.models:
        bench_model(model, a, cells, meta)
    res = {"meta": meta, "cells": cells}
    (out / "bench_serving.json").write_text(json.dumps(res, indent=1))
    (out / "bench_serving.md").write_text(markdown(res))
    print(markdown(res))
    print(f"[bench_serving] wrote {out / 'bench_serving.json'} and {out / 'bench_serving.md'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
