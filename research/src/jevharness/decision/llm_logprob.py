"""Small (or any) LLM used as a System-1 decider by reading next-token probabilities.

This is the main *open* baseline for Jev: one short request per question (max_tokens=1; one forward pass on a
self-hosted server, provider-side passes are not observable on an API), answer read from the logprob distribution
over option labels -- no free-text parsing, so the answer can never go off-schema. At most 52 options per request
(letter labels); the harness tournament splits wider Choices (decision/tournament.py), and ``n_permutations > 1``
multiplies the requests. Works with any OpenAI-compatible endpoint that returns ``top_logprobs``
(OpenAI, DeepSeek, vLLM/SGLang self-hosted Qwen3/Gemma/Llama, some OpenRouter providers).

Position bias mitigation: ``n_permutations > 1`` averages the distribution over cyclic
shifts of the option order (cost x n). See Zheng et al. 2024 (PriDe) for the bias.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..accounting import default_prices
from ..llm.base import LLMClient
from .base import DecisionBackend
from .types import OPTION_LABELS, Answer, DecisionResult, QKind, Question, Usage, answer_from_distribution

SYSTEM_PROMPT = (
    "You are a fast decision module inside an AI agent. Read the state and answer the question "
    "with exactly one token: the label of your answer. Do not explain."
)


def build_prompt(state: str, q: Question, order: list[int] | None = None) -> tuple[str, dict[str, str]]:
    """Return (prompt, token_label -> answer_label map)."""
    head = f"STATE:\n{state}\n\nQUESTION: {q.text}\n"
    if q.kind == QKind.CHOICE:
        assert q.options is not None
        if len(q.options) > len(OPTION_LABELS):
            raise ValueError(f"logprob decider supports <= {len(OPTION_LABELS)} options per call; "
                             "use the harness tournament for larger sets")
        order = order or list(range(len(q.options)))
        lines, mapping = [], {}
        for pos, idx in enumerate(order):
            lab = OPTION_LABELS[pos]
            desc = ""
            if q.option_descriptions and q.option_descriptions[idx]:
                desc = f" -- {q.option_descriptions[idx]}"
            lines.append(f"{lab}. {q.options[idx]}{desc}")
            mapping[lab] = q.options[idx]
        return head + "OPTIONS:\n" + "\n".join(lines) + "\n\nAnswer with the option label only.", mapping
    if q.kind == QKind.NOUL:
        return head + "\nAnswer Yes or No.", {"Yes": "yes", "No": "no", "yes": "yes", "no": "no",
                                              "YES": "yes", "NO": "no"}
    lo, hi = q.scale  # type: ignore[misc]
    rubric = f"RUBRIC: {q.rubric}\n" if q.rubric else ""
    if q.levels:
        rubric += "LEVELS:\n" + "\n".join(f"{lo + i}: {d}" for i, d in enumerate(q.levels)) + "\n"
    if hi - lo > 9:
        raise ValueError("LogprobLLMBackend supports score scales with at most 10 levels")
    # single-digit labels so each level is one token
    mapping = {str(i): str(lo + i) for i in range(hi - lo + 1)} if (lo < 0 or hi > 9) else \
        {str(v): str(v) for v in range(lo, hi + 1)}
    scale_desc = ", ".join(f"{k}={v}" for k, v in mapping.items()) if lo < 0 or hi > 9 else f"{lo}..{hi}"
    return head + rubric + f"\nAnswer with a single integer score ({scale_desc}).", mapping


def dist_from_top_logprobs(top: dict[str, float], mapping: dict[str, str]) -> tuple[dict[str, float], float]:
    """Aggregate probability mass of tokens that map to answer labels.

    Returns (unnormalised distribution over answer labels, total covered mass)."""
    dist: dict[str, float] = {}
    for tok, lp in top.items():
        key = tok.strip().strip(".:)").strip()
        if key in mapping:
            lab = mapping[key]
            dist[lab] = dist.get(lab, 0.0) + math.exp(lp)
    return dist, sum(dist.values())


class LogprobLLMBackend(DecisionBackend):
    name = "llm_logprob"

    def __init__(self, llm: LLMClient, n_permutations: int = 1, max_workers: int = 8,
                 top_logprobs: int = 20, extra_body: dict[str, Any] | None = None):
        self.llm = llm
        self.model = llm.model
        self.n_permutations = max(1, n_permutations)
        self.max_workers = max_workers
        self.top_logprobs = top_logprobs
        self.extra_body = extra_body

    def _one(self, state: str, q: Question) -> tuple[Answer, Usage]:
        n_opts = len(q.options) if q.kind == QKind.CHOICE and q.options else 0
        perms = [None]
        if q.kind == QKind.CHOICE and self.n_permutations > 1:
            perms = [[(i + s) % n_opts for i in range(n_opts)]
                     for s in range(0, n_opts, max(1, n_opts // self.n_permutations))][: self.n_permutations]
        agg: dict[str, float] = {}
        usage = Usage(backend=self.name, model=self.model, n_calls=0)
        covered_all = []
        for order in perms:
            prompt, mapping = build_prompt(state, q, order)
            resp = self.llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": prompt}],
                                 max_tokens=1, temperature=0.0, top_logprobs=self.top_logprobs,
                                 extra_body=self.extra_body)
            top = resp.top_logprobs[0] if resp.top_logprobs else {}
            if not top and resp.text:  # provider ignored logprobs: fall back to the sampled token
                top = {resp.text: 0.0}
            dist, covered = dist_from_top_logprobs(top, mapping)
            covered_all.append(covered)
            for k, v in dist.items():
                agg[k] = agg.get(k, 0.0) + v / len(perms)
            usage.input_tokens += resp.input_tokens
            usage.output_tokens += resp.output_tokens
            usage.latency_ms += resp.latency_ms  # permutations are issued one after another
            usage.n_calls += 1
        usage.cost_usd = default_prices().cost(self.model, usage.input_tokens, usage.output_tokens)
        usage.priced = default_prices().get(self.model) is not None
        raw = {"covered_mass": covered_all}
        return answer_from_distribution(q, agg, raw=raw), usage

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            outs = list(ex.map(lambda q: self._one(state, q), questions))
        usage = Usage(backend=self.name, model=self.model, n_calls=0, priced=all(u.priced for _, u in outs))
        for _, u in outs:
            usage.input_tokens += u.input_tokens
            usage.output_tokens += u.output_tokens
            usage.cost_usd += u.cost_usd
            usage.latency_ms = max(usage.latency_ms, u.latency_ms)  # questions issued concurrently
            usage.n_calls += u.n_calls
        return DecisionResult(answers=[a for a, _ in outs], usage=usage)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "n_permutations": self.n_permutations}
