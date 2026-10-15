"""Token prices (USD per million tokens) and the cost of one call.

The built-in table holds list prices read from the providers' pricing pages on the date in ``PRICES_DATE``. Prices
change; override or extend them in the config (``prices:``) and check them before you report numbers. A model that is
in neither table is reported as unpriced (cost 0, ``priced: false`` in the logs) rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .types import Usage

PRICES_DATE = "2026-10-14"


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cached_input: float | None = None
    cache_storage_per_hour: float | None = None     # explicit context caches, USD per million tokens per hour
    cache_write: float | None = None                # automatic prompt-cache writes (OpenAI), USD per million tokens

    @classmethod
    def from_any(cls, d: Mapping[str, Any]) -> "Price":
        return cls(input=float(d.get("input", d.get("input_per_m", 0.0))),
                   output=float(d.get("output", d.get("output_per_m", 0.0))),
                   cached_input=(None if d.get("cached_input", d.get("cached_input_per_m")) is None
                                 else float(d.get("cached_input", d.get("cached_input_per_m")))),
                   cache_storage_per_hour=(None if d.get("cache_storage_per_hour") is None
                                           else float(d["cache_storage_per_hour"])),
                   cache_write=(None if d.get("cache_write") is None else float(d["cache_write"])))


# Gemini Developer API, paid tier, standard service, prompts <= 200k tokens (ai.google.dev/gemini-api/docs/pricing).
# Several entries are introductory prices with a printed end date; they are noted next to the entry.
BUILTIN: dict[str, Price] = {
    "gemini-4-argon": Price(2.00, 10.00, 0.10),          # through 2026-11-30, then 4.00 / 20.00 / 0.20; implicit cache
    "gemini-3.1-pro-preview": Price(2.00, 12.00, 0.20, 4.50),
    "gemini-3.8-flash": Price(0.75, 3.75, 0.075, 0.50),  # through 2026-12-31, then 1.50 / 7.50 / 0.15 / 1.00
    "gemini-3.6-flash": Price(0.75, 3.75, 0.075, 0.50),  # through 2026-12-31, then 1.50 / 7.50 / 0.15 / 1.00
    "gemini-3.5-flash": Price(1.50, 9.00, 0.15),
    "gemini-3.5-flash-lite": Price(0.30, 2.50, 0.03, 1.00),
    "gemini-3.1-flash-lite": Price(0.25, 1.50, 0.025, 1.00),
    # OpenAI API, standard tier, short context (platform.openai.com/docs/pricing, read 2026-10-14): input, output,
    # cached input, cache writes
    "gpt-6.1-sol": Price(2.00, 10.00, 0.10, cache_write=2.50),
    "gpt-6-luna": Price(0.10, 0.50, 0.01, cache_write=0.125),
    "gpt-5.6-sol": Price(4.00, 20.00, 0.40, cache_write=5.00),
}


class PriceTable:
    def __init__(self, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.table: dict[str, Price] = dict(BUILTIN)
        for name, p in (overrides or {}).items():
            self.table[name] = Price.from_any(p)

    def get(self, model: str) -> Price | None:
        if model in self.table:
            return self.table[model]
        base = model.split("/")[-1]           # "google/gemini-3.8-flash" -> "gemini-3.8-flash"
        return self.table.get(base)

    def cost(self, model: str, usage: Usage) -> tuple[float, bool]:
        """(USD, priced). An upstream-reported cost wins over the table."""
        if usage.reported_cost is not None:
            return float(usage.reported_cost), True
        p = self.get(model)
        if p is None:
            return 0.0, False
        cached = min(usage.cached_tokens, usage.input_tokens)
        written = min(usage.cache_write_tokens, usage.input_tokens - cached)
        fresh = usage.input_tokens - cached - written
        cached_rate = p.cached_input if p.cached_input is not None else p.input
        write_rate = p.cache_write if p.cache_write is not None else p.input
        usd = (fresh * p.input + cached * cached_rate + written * write_rate
               + (usage.output_tokens + usage.reasoning_tokens) * p.output) / 1e6
        return usd, True
