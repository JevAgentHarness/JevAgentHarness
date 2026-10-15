"""System-1 decision backends. ``make_backend`` builds one from a config dict."""

from __future__ import annotations

from typing import Any

from .base import CachedBackend, DecisionBackend
from .types import Answer, Choice, DecisionResult, Noul, QKind, Question, Score, Usage


def make_backend(spec: dict[str, Any] | str, cache: str | None = None) -> DecisionBackend:
    """Examples::

        {"type": "jev", "model": "jev-1.13.0"}
        {"type": "openjev", "base_url": "http://localhost:8080"}
        {"type": "jev", "name": "codiv-openjev", "base_url": "https://api.codiv.ai", "model": "openjev-latest"}
        {"type": "llm_logprob", "llm": {"provider": "vllm", "model": "Qwen/Qwen3-4B"}, "n_permutations": 2}
        {"type": "llm_generative", "llm": {"provider": "gemini", "model": "gemini-3.5-flash-lite"}}
        {"type": "embedding", "model": "text-embedding-3-small"}
        {"type": "simulated", "accuracy": 0.9}
        {"type": "calibrated", "inner": {"type": "jev", "provider": "openrouter"}, "temperatures_file": "cal.json"}
        {"type": "bm25"} / {"type": "tfidf", "fallback": {"type": "heuristic"}}      # lexical baselines
        {"type": "rate", "rate": 0.3} / {"type": "prior", "rule": "recent"} / {"type": "heuristic"}   # controls
        {"type": "s2_self", "llm": {"provider": "gemini", "model": "gemini-3.5-flash"}}   # S2 as its own decider
        {"type": "intern_decision", "model": "internlm/Intern-Decision-2B", "device": "auto"}   # in-process; 1 forward pass per <= 16 questions of <= 62 options
        {"type": "openai_decisions", "model": "gpt-6-luna"}    # OpenAI's hosted Decisions API (E15)
        {"type": "ensemble", "rule": "agree", "members": [{"type": "jev", "provider": "openrouter"},
                                                          {"type": "openai_decisions"}]}   # E20 agreement gate
    (baselines: docs/BASELINES.md)
    """
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = dict(spec)
    kind = spec.pop("type")
    backend: DecisionBackend
    if kind == "jev":
        from .jev import JevBackend
        backend = JevBackend(**spec)
    elif kind == "openjev":
        from .jev import OpenJevBackend
        backend = OpenJevBackend(**spec)
    elif kind == "openai_decisions":
        from .openai_decisions import OpenAIDecisionsBackend
        backend = OpenAIDecisionsBackend(**spec)
    elif kind in ("llm_logprob", "llm_generative"):
        from ..llm import make_llm
        llm = make_llm(spec.pop("llm"))
        if kind == "llm_logprob":
            from .llm_logprob import LogprobLLMBackend
            backend = LogprobLLMBackend(llm, **spec)
        else:
            from .llm_generative import GenerativeLLMBackend
            backend = GenerativeLLMBackend(llm, **spec)
    elif kind == "embedding":
        from .embedding import EmbeddingBackend
        backend = EmbeddingBackend(**spec)
    elif kind == "simulated":
        from .simulated import SimulatedBackend
        backend = SimulatedBackend(**spec)
    elif kind == "oracle":
        from .simulated import OracleBackend
        backend = OracleBackend()
    elif kind == "random":
        from .simulated import RandomBackend
        backend = RandomBackend(**spec)
    elif kind == "calibrated":  # {"type": "calibrated", "inner": {...}, "temperature": 1.3 | "temperatures_file": ...}
        from .calibration import TemperatureScaledBackend
        inner = make_backend(spec.pop("inner"))
        backend = TemperatureScaledBackend(inner, **spec)
    elif kind in ("bm25", "tfidf"):
        from .lexical import BM25Backend, TfidfBackend
        backend = BM25Backend(**spec) if kind == "bm25" else TfidfBackend(**spec)
    elif kind in ("rate", "prior", "heuristic"):
        from .controls import HeuristicBackend, PriorBackend, RateBackend
        backend = {"rate": RateBackend, "prior": PriorBackend, "heuristic": HeuristicBackend}[kind](**spec)
    elif kind == "s2_self":
        from .self_llm import make_self_backend
        backend = make_self_backend(**spec)
    elif kind == "ensemble":  # E20: decision models voting / agreeing behind one System-1 interface (members = specs)
        from .ensemble import EnsembleBackend
        backend = EnsembleBackend(**spec)
    elif kind == "intern_decision":  # internlm/Intern-Decision-{0.8B,2B,4B} in-process (torch + transformers)
        from .intern_decision import InternDecisionBackend
        backend = InternDecisionBackend(**spec)
    else:
        raise ValueError(f"unknown decision backend type {kind!r}")
    return CachedBackend(backend, cache) if cache else backend


__all__ = ["Answer", "CachedBackend", "Choice", "DecisionBackend", "DecisionResult", "Noul", "QKind",
           "Question", "Score", "Usage", "make_backend"]
