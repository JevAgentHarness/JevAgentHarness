"""Cost / latency / token accounting shared by System-1 and System-2 components.

Every model call in an episode is appended to a :class:`Ledger`. The benchmark reports
success *together with* cost, wall-clock latency on the critical path, token counts and
number of LLM calls, so each harness configuration is a point on an efficiency/effectiveness
plane (see ``jevharness.bench.metrics.pareto_front``).

Prices live in ``configs/prices/*.yaml`` and are looked up by model name; nothing here
hard-codes a vendor price. Unknown models cost 0 and are flagged in the ledger.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

_PRICE_DIR = Path(__file__).resolve().parents[2] / "configs" / "prices"


@dataclass
class Price:
    input_per_m: float = 0.0          # USD per 1M input tokens
    output_per_m: float = 0.0         # USD per 1M output tokens
    cached_input_per_m: float | None = None
    per_call: float = 0.0             # flat per-request fee, if any

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        cached_rate = self.cached_input_per_m if self.cached_input_per_m is not None else self.input_per_m
        fresh = max(input_tokens - cached_input_tokens, 0)
        return (fresh * self.input_per_m + cached_input_tokens * cached_rate
                + output_tokens * self.output_per_m) / 1e6 + self.per_call


class PriceTable:
    def __init__(self, prices: dict[str, Price] | None = None):
        self.prices: dict[str, Price] = prices or {}

    @classmethod
    def load(cls, paths: list[str | Path] | None = None) -> "PriceTable":
        if paths is None:
            env = os.environ.get("JEV_PRICE_FILES")
            paths = [Path(p) for p in env.split(":")] if env else sorted(_PRICE_DIR.glob("*.yaml"))
        prices: dict[str, Price] = {}
        for p in paths:
            data = yaml.safe_load(Path(p).read_text()) or {}
            for model, spec in (data.get("models") or {}).items():
                prices[model] = Price(**spec)
        return cls(prices)

    def get(self, model: str | None) -> Price | None:
        if model is None:
            return None
        if model in self.prices:
            return self.prices[model]
        # allow provider prefixes such as "openrouter/qwen/qwen3-8b" -> "qwen/qwen3-8b" or "deepseek/deepseek-chat" ->
        # "deepseek-chat" (and a bare id for a provider-prefixed row), but only on a whole path segment: a bare
        # fragment such as "flash" or "v4-flash" must never resolve to "deepseek/deepseek-v4-flash" (PREREG_A7_STAGE2
        # A7.12 (h): an unknown id stays unpriced instead of borrowing another model's price)
        for key in sorted(self.prices, key=len, reverse=True):
            if model.endswith("/" + key) or key.endswith("/" + model):
                return self.prices[key]
        return None

    def cost(self, model: str | None, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        price = self.get(model)
        return price.cost(input_tokens, output_tokens, cached_input_tokens) if price else 0.0


_DEFAULT_TABLE: PriceTable | None = None


def default_prices() -> PriceTable:
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is None:
        _DEFAULT_TABLE = PriceTable.load()
    return _DEFAULT_TABLE


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-free estimate (~4 chars/token) used only when an API returns no usage."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class CallRecord:
    component: str                # "s1" (decision model) | "s2" (LLM) | "tool" | "embed"
    backend: str
    model: str | None
    decision_point: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    priced: bool = True
    step: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # --- A6 ruling (e), additive (None for records that predate them / backends that do not report them):
    attempts: int | None = None          # HTTP attempts behind this record (1 = no retry); latency_ms stays the
    retry_wait_ms: float | None = None   # successful attempt's, this is the backoff slept before it (quota artefact)
    request_id: str | None = None        # provider request / response id, for auditing the critical path
    ts: str | None = None                # ISO-8601 UTC time the record was added to the ledger (no DAG, an audit trail)
    tournament_round: int | None = None  # Choice tournaments: the round (0 = first) this chunk call belongs to


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


class Ledger:
    """Thread-safe per-episode call log."""

    def __init__(self) -> None:
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self.step: int | None = None

    def add(self, rec: CallRecord) -> None:
        if rec.step is None:
            rec.step = self.step
        if rec.ts is None:
            rec.ts = _utc_now()
        with self._lock:
            self.records.append(rec)

    @contextmanager
    def timer(self) -> Iterator[dict[str, float]]:
        box: dict[str, float] = {}
        t = time.perf_counter()
        try:
            yield box
        finally:
            box["ms"] = (time.perf_counter() - t) * 1000

    def wall_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def summary(self) -> dict[str, Any]:
        def agg(filter_fn) -> dict[str, float]:
            rs = [r for r in self.records if filter_fn(r)]
            return {
                "calls": len(rs),
                "input_tokens": sum(r.input_tokens for r in rs),
                "output_tokens": sum(r.output_tokens for r in rs),
                "cost_usd": sum(r.cost_usd for r in rs),
                "latency_ms": sum(r.latency_ms for r in rs),
            }

        out: dict[str, Any] = {"total": agg(lambda r: True)}
        for comp in sorted({r.component for r in self.records}):
            out[comp] = agg(lambda r, c=comp: r.component == c)
        dps = sorted({r.decision_point for r in self.records if r.decision_point})
        out["by_decision_point"] = {dp: agg(lambda r, d=dp: r.decision_point == d) for dp in dps}
        out["unpriced_models"] = sorted({str(r.model) for r in self.records if not r.priced})
        return out

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]
