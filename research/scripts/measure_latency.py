"""Measure System-1 backend latency / throughput at several concurrency levels.

Feeds (1) the latency-anatomy analysis and (2) the amortised $/M-token price of self-hosted
deciders (OpenJev, vLLM small LMs, our trained model) in configs/prices/models.yaml.

  python scripts/measure_latency.py --s1 '{"type":"openjev"}' --concurrency 1 8 32 --n 200 \
      --state-tokens 1500 --questions 3 --gpu-hour-usd 2.5
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from jevharness.decision import Choice, Noul, Score, make_backend


def make_state(tokens: int, rng: random.Random) -> str:
    words = ["order", "user", "tool", "refund", "status", "pending", "shipped", "ticket", "error", "retry"]
    return "TASK: resolve the customer's request.\nHISTORY:\n" + " ".join(
        f"{rng.choice(words)}-{rng.randint(0, 9999)}" for _ in range(max(1, tokens // 3)))


def questions(n: int) -> list:
    qs = [Choice("Which tool should the agent call next?", [f"tool_{i}" for i in range(20)] + ["finish"]),
          Noul("Is the task complete?"), Score("How risky is the next action?", scale=(1, 5))]
    return (qs * ((n // 3) + 1))[:n]


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--n", type=int, default=100, help="requests per concurrency level")
    ap.add_argument("--state-tokens", type=int, default=1500)
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--gpu-hour-usd", type=float, default=None, help="for amortised $/M input tokens")
    a = ap.parse_args()
    backend = make_backend(json.loads(a.s1))
    rng = random.Random(0)
    backend.decide(make_state(100, rng), questions(1))  # warm-up
    report = {}
    for c in a.concurrency:
        states = [make_state(a.state_tokens, rng) for _ in range(a.n)]
        lats: list[float] = []
        toks: list[int] = []

        def one(s: str) -> None:
            t = time.perf_counter()
            r = backend.decide(s, questions(a.questions))
            lats.append((time.perf_counter() - t) * 1000)
            toks.append(r.usage.input_tokens)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            list(ex.map(one, states))
        wall = time.perf_counter() - t0
        lats.sort()
        row = {"p50_ms": statistics.median(lats), "p90_ms": lats[int(0.90 * (len(lats) - 1))],
               "p95_ms": lats[int(0.95 * (len(lats) - 1))],
               "req_per_s": a.n / wall, "input_tokens_per_req": statistics.mean(toks) if toks else None}
        if a.gpu_hour_usd and toks and statistics.mean(toks) > 0:
            tok_per_hour = row["req_per_s"] * 3600 * statistics.mean(toks)
            row["amortised_usd_per_m_input"] = a.gpu_hour_usd / tok_per_hour * 1e6
        report[f"c={c}"] = row
    print(json.dumps({"backend": backend.describe(), "questions_per_req": a.questions,
                      "state_tokens": a.state_tokens, "results": report}, indent=1))


if __name__ == "__main__":
    main()
