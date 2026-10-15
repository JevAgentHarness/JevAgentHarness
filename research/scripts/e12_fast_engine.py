#!/usr/bin/env python3
r"""E12c's added readout (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E12c: the recipe on a stronger base, and
same-engine serving cost"): GPU time per decision of four decision models under ONE fast engine (vLLM, LoRA merged
into the base), on E1's 3,991 decisions, the prompts exactly as ``serve_jev`` renders them, one request per question,
the requests of a state together (E1's batch_by_state), 4 states in flight (E1's 4 workers). Descriptive.

    python scripts/e12_fast_engine.py prompts --tokenizer DIR --out P.jsonl.gz     # harness venv: E1 items -> token ids
    python scripts/e12_fast_engine.py merge --base B --adapter A --out DIR          # harness venv: peft merge, bf16
    python scripts/e12_fast_engine.py time --model DIR --prompts P --out R.json     # vLLM venv: the timing
    python scripts/e12_fast_engine.py agree --timing R.json --records S1.jsonl      # agreement with served answers

The answer is the most likely option code (greedy decoding restricted to the item's codes); its agreement with the
model's HF-served answers is a check that the engine reads the same prompt, not a new accuracy."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GPU_USD_PER_H = 2.24208          # E9's tariff (the Stage-2 host, A7.13 (iii))
WORKERS = 4                      # E1's workers
WARMUP_STATES = 100


def _jsonl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def cmd_prompts(a: argparse.Namespace) -> int:
    """Token ids of every E1 decision for one tokenizer, grouped by state in E1's order (first appearance)."""
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    from transformers import AutoTokenizer

    from jevharness.bench.decision_eval import load_population
    from jevharness.bench.suites.static import norm
    from jevharness.training.data import render
    from jevharness.training.decision_model import DecisionCollator, code_token_ids
    from jevharness.training.grpo_online import question_to_example

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    col = DecisionCollator(tok, max_len=4096)
    codes = code_token_ids(tok)
    items = load_population(yaml.safe_load((ROOT / "configs/decision/bfcl_when2call_rl.yaml").read_text()))
    order: dict[str, int] = {}
    out = []
    for it in items:
        g = order.setdefault(it.state, len(order))
        ex = question_to_example(it.state, it.question)
        ids = col.encode(render({"state": ex["state"], "question": ex["question"], "options": ex["options"]})[0])
        out.append({"id": it.id, "group": g, "ids": ids, "codes": codes[: len(ex["options"])],
                    "labels_norm": [norm(x) for x in ex["labels"]]})
    op = gzip.open if a.out.endswith(".gz") else open
    with op(a.out, "wt") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    lens = [len(r["ids"]) for r in out]
    print(json.dumps({"decisions": len(out), "states": len(order), "mean_tokens": sum(lens) / len(lens),
                      "tokenizer": a.tokenizer}))
    return 0


def cmd_merge(a: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    print(json.dumps({"merged": a.out, "base": a.base, "adapter": a.adapter}))
    return 0


def cmd_time(a: argparse.Namespace) -> int:
    import os

    # no FlashInfer JIT (the host has no nvcc): FlashAttention for attention, the plain sampler
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:  # older vLLM
        from vllm.engine.async_llm_engine import AsyncLLMEngine as Engine

    rows = _jsonl(Path(a.prompts))
    groups: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    keys = sorted(groups)
    args = AsyncEngineArgs(model=a.model, dtype="bfloat16", enable_prefix_caching=True, max_model_len=4352,
                           gpu_memory_utilization=a.gpu_mem, seed=0)
    engine = Engine.from_engine_args(args)
    rid = [0]

    async def one(r: dict[str, Any]) -> dict[str, Any]:
        rid[0] += 1
        sp = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=r["codes"])   # greedy = argmax over codes
        final = None
        async for o in engine.generate(TokensPrompt(prompt_token_ids=r["ids"]), sp, request_id=f"q{rid[0]}"):
            final = o
        tok = final.outputs[0].token_ids[0] if final and final.outputs and final.outputs[0].token_ids else None
        return {"id": r["id"], "pred": r["labels_norm"][r["codes"].index(tok)] if tok in r["codes"] else None}

    async def group(g: int) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(one(r) for r in groups[g]))
        ms = (time.perf_counter() - t0) * 1000
        return [{**x, "latency_ms": ms, "request_questions": len(groups[g])} for x in res]

    async def run(ks: list[int], workers: int) -> tuple[list[dict[str, Any]], float]:
        q: asyncio.Queue[int] = asyncio.Queue()
        for k in ks:
            q.put_nowait(k)
        recs: list[dict[str, Any]] = []

        async def worker() -> None:
            while not q.empty():
                recs.extend(await group(q.get_nowait()))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(workers)))
        return recs, time.perf_counter() - t0

    async def reset() -> None:
        try:
            await engine.reset_prefix_cache()
        except Exception:  # noqa: BLE001 - older engines: the warm-up states stay cached (stated in the record)
            pass

    async def main() -> dict[str, Any]:
        out = {}
        await run(keys[:WARMUP_STATES], WORKERS)   # warm-up (not timed; prefix cache reset before each pass)
        for w in a.workers:                        # 4 = E1's workers (registered); 1 = one request at a time (added)
            await reset()
            recs, wall = await run(keys, w)
            out[w] = {"recs": recs, "wall_s": wall}
        return out

    try:
        res = asyncio.run(main())
    finally:
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            pass
    def summary(recs: list[dict[str, Any]], wall: float, w: int) -> dict[str, Any]:
        n = len(recs)
        e9 = [r["latency_ms"] / r["request_questions"] for r in recs]
        per_state = {}
        for r in recs:
            per_state[(r["latency_ms"], r["request_questions"])] = r["latency_ms"]
        return {"workers": w, "decisions": n, "wall_s": wall, "gpu_s_per_decision": wall / n,
                "usd_per_1000_wall": 1000 * GPU_USD_PER_H * wall / n / 3600,
                "e9_rule_s_per_decision": statistics.mean(e9) / 1000,
                "usd_per_1000_e9_rule": 1000 * GPU_USD_PER_H * statistics.mean(e9) / 1000 / 3600,
                "latency_ms_median": statistics.median(per_state.values()),          # per request (a state's questions)
                "latency_ms_p90": statistics.quantiles(list(per_state.values()), n=10)[-1]}

    by_w = {w: summary(v["recs"], v["wall_s"], w) for w, v in res.items()}
    main_w = WORKERS if WORKERS in by_w else next(iter(by_w))
    recs = res[main_w]["recs"]
    gb = sum(f.stat().st_size for f in Path(a.model).glob("*.safetensors")) / 1e9
    out = {"model": a.model, "label": a.label, "vllm": vllm.__version__, "states": len(keys), **by_w[main_w],
           "by_workers": {str(w): s for w, s in by_w.items()}, "weights_gb": gb, "gpu_usd_per_h": GPU_USD_PER_H,
           "warmup_states": WARMUP_STATES, "env": {k: os.environ.get(k) for k in ("VLLM_USE_FLASHINFER_SAMPLER",
                                                                                   "VLLM_ATTENTION_BACKEND")},
           "answers": {r["id"]: r["pred"] for r in recs}}
    Path(a.out).write_text(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "answers"}, indent=1))
    return 0


def cmd_agree(a: argparse.Namespace) -> int:
    t = json.loads(Path(a.timing).read_text())
    served = {r["id"]: r for r in _jsonl(Path(a.records))}
    ids = sorted(set(t["answers"]) & set(served))
    agree = sum(1 for i in ids if t["answers"][i] == served[i].get("pred"))
    print(json.dumps({"n": len(ids), "agree": agree, "agreement": agree / len(ids) if ids else None}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--base", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    t = sub.add_parser("time")
    t.add_argument("--model", required=True)
    t.add_argument("--prompts", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--label", default="")
    t.add_argument("--gpu-mem", type=float, default=0.85)
    t.add_argument("--workers", type=lambda s: [int(x) for x in s.split(",")], default=[WORKERS, 1],
                   help="timing passes: 4 (E1's workers, registered) then 1 (one request at a time, added)")
    g = sub.add_parser("agree")
    g.add_argument("--timing", required=True)
    g.add_argument("--records", required=True)
    a = ap.parse_args(argv)
    return {"prompts": cmd_prompts, "merge": cmd_merge, "time": cmd_time, "agree": cmd_agree}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
