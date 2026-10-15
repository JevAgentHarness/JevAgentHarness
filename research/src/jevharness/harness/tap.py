"""Ledger views used by the harness to attribute calls to one decision and to tag measurement calls.

Every System-1 / System-2 call goes through the episode :class:`~jevharness.accounting.Ledger` (cost and latency
accounting is a paper claim). A :class:`Tap` forwards each record to the episode ledger and keeps a copy, so a
decision can log its own cost; measurement calls are tagged ``meta["shadow"] = True`` and
``meta["measure"] in {"shadow", "cf", "probe"}`` (docs/META_GOLD_SPEC.md §1.4), which keeps them out of the
episode cost (``agent.usage_summary``).
"""

from __future__ import annotations

from typing import Any

from ..accounting import CallRecord, Ledger

MEASURES = ("shadow", "cf", "probe")


class Tap(Ledger):
    """Ledger view of one decision's call(s): every record is forwarded to the episode ledger -- tagged
    ``meta["shadow"] = True`` (and ``meta["measure"]``, plus ``extra`` tags such as ``cf_of`` / ``probe_of``) for
    measurement calls -- and also kept here, so the decision's own cost and latency can be logged in its
    ``DecisionLog`` (tournaments make several calls per decision)."""

    def __init__(self, inner: Ledger, shadow: bool = False, measure: str | None = None,
                 extra: dict[str, Any] | None = None):
        super().__init__()
        self.inner = inner
        self.shadow = shadow or measure is not None
        self.measure = measure or ("shadow" if shadow else None)
        self.extra = dict(extra or {})
        self.step = inner.step

    def add(self, rec: CallRecord) -> None:
        if self.shadow:
            rec.meta = {**(rec.meta or {}), "shadow": True, "measure": self.measure, **self.extra}
        self.inner.add(rec)
        with self._lock:
            self.records.append(rec)

    def spent(self) -> tuple[float, float]:
        return sum(r.cost_usd for r in self.records), sum(r.latency_ms for r in self.records)

    def stamp(self, dlog: Any) -> None:
        """Write this decision's System-1 cost / latency and pass accounting (calls, rounds, backend-reported
        passes; review P1-4) onto ``dlog``."""
        from ..decision.tournament import decision_pass_counts
        dlog.s1_cost_usd, dlog.s1_latency_ms = self.spent()
        if self.records:
            pc = decision_pass_counts(self.records)
            dlog.s1_calls, dlog.s1_rounds, dlog.s1_passes = pc["s1_calls"], pc["s1_rounds"], pc["s1_passes"]


def is_shadow(r: CallRecord) -> bool:
    return bool((r.meta or {}).get("shadow"))


def measure_of(r: CallRecord) -> str:
    return str((r.meta or {}).get("measure") or "shadow")
