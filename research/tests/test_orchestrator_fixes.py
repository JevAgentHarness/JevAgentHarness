"""Regression tests for fixes routed from the pilot agent (memory gate budget, snake_case ids, cached prices)."""

from __future__ import annotations

from jevharness.accounting import default_prices
from jevharness.harness.config import DPConfig, Mode
from jevharness.harness.memory import MemoryStore
from jevharness.harness.tools import extract_candidates
from jevharness.llm.base import ChatResponse


class _GateLLM:
    """Answers 'yes' only when given enough output budget (Gemini 3.x returns '' at max_tokens=4)."""

    def __init__(self):
        self.kwargs = {}

    def chat(self, messages, **kw):
        self.kwargs = kw
        return ChatResponse(text="yes" if kw.get("max_tokens", 1000) >= 8 else "")


def test_llm_memory_gate_has_enough_output_budget():
    llm = _GateLLM()
    store = MemoryStore(DPConfig(mode=Mode.LLM), DPConfig(mode=Mode.ALWAYS), llm=llm)
    assert store.offer("order ORD-1 costs $42", "TASK: refund") is True
    assert llm.kwargs["max_tokens"] >= 8


def test_snake_case_ids_with_digits_are_candidates():
    c = extract_candidates("user raj_sanchez_7340 has reservation 4GH8KL and order ORD-1234")
    assert {"raj_sanchez_7340", "4GH8KL", "ORD-1234"} <= set(c)


def test_gemini_cached_input_is_discounted():
    p = default_prices()
    full = p.cost("gemini-3.5-flash", 1_000_000, 0)
    cached = p.cost("gemini-3.5-flash", 1_000_000, 0, cached_input_tokens=1_000_000)
    assert cached < 0.2 * full
