""""Use the big model as its own decider": the System-2 LLM answers the same typed questions.

This control asks whether a *separate* fast decider is needed at all. The alternative is to have the
agent's own LLM answer the Choice / Score / Noul questions, which costs the same number of decisions but
at System-2 price and latency. Mechanically it is :class:`GenerativeLLMBackend` (batched,
verbalised-distribution JSON) pointed at the System-2 model. It gets its own backend name, ``s2_self``, so
ledgers, trajectories and report tables keep it apart from the cheap-LLM System-1 arms. Its calls are
booked as ``component="s1"`` at the System-2 model's price, so its cost shows up in every run summary.

Config (use the *same* model and settings as the arm's ``llm``)::

    s1_backends:
      s2self: {type: s2_self, llm: {provider: gemini, model: gemini-3.5-flash}, reasoning_effort: minimal}

which is equivalent to ``{type: llm_generative, llm: {...}, reasoning_effort: minimal}`` apart from the
backend name. ``mode: label`` gives the verbalised-confidence variant; ``batch: false`` sends one call per question.
"""

from __future__ import annotations

from typing import Any

from ..llm.base import LLMClient
from .llm_generative import GenerativeLLMBackend


class SelfLLMBackend(GenerativeLLMBackend):
    name = "s2_self"


def make_self_backend(llm: dict[str, Any] | str | LLMClient, mode: str = "distribution", batch: bool = True,
                      **chat_kw: Any) -> SelfLLMBackend:
    from ..llm import make_llm

    client = llm if isinstance(llm, LLMClient) else make_llm(llm)
    return SelfLLMBackend(client, mode=mode, batch=batch, **chat_kw)
