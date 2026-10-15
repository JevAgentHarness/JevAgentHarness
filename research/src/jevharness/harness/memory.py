"""A minimal agent memory store with pluggable write-gating and read-selection policies.

Write policies (DP.MEMORY_WRITE): off | always | llm (LLM decides) | trigger (S1 Noul gate)
                                  | shadow (S1 asked and logged, nothing stored -- measurement arms)
                                  | cascade (S1's yes / no when confident, else the LLM gate; E6b)
                                  | guard (the LLM gate proposes, S1 can veto a write; E6b)
                                  | assist (S1's answer and confidence advise the LLM gate, which decides; E6d)
                                  | replace (S1's argmax alone: the trigger at tau 0.5; E6e)
Read policies  (DP.MEMORY_READ):  off | always (all/recency-k) | assist (S1 top-k Choice)
                                  | cascade (S1's top-k when confident, else always; E6b)
                                  | trigger (S1 decides whether to read the recent memories this step; E6b)
                                  | replace (S1's single top memory, or none: the assist with k = 1; E6e)
                                  | guard (the LLM selects memories, S1 can veto the selection once; E6d)

Keeping the store deliberately simple isolates the *decision* (what to keep / what to recall)
from retrieval engineering, which is what the study is about.

With ``log_decisions=True`` (the agent harness) every offer logs one ``DecisionLog`` (dp ``memory_write``, the
gate's input state, ``s1_dist``) and every System-1 recall one ``memory_read`` record into :attr:`decisions`; the
harness moves them into the step log and back-patches the gold (docs/META_GOLD_SPEC.md §3.6-3.7). Gold labels are
never written to ``MemoryItem.meta`` or to any text a backend sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..accounting import Ledger
from ..decision.base import DecisionBackend
from ..decision.types import Answer, Choice, Noul, Question
from ..llm.base import LLMClient
from .config import DP, DPConfig, Mode
from .prompts import (MEMORY_REJECT_NOTE, MEMORY_SELECT_PROMPT, Q_MEMORY_READ_TRIGGER, Q_MEMORY_WRITE, advice_note,
                      q_memory_read_guard)
from .tap import Tap
from .trajectory import DecisionLog

NONE_MEMORY = "(none of these memories is relevant)"
Q_MEMORY_READ = "Which stored memory is most relevant to the current step?"


@dataclass
class MemoryItem:
    id: int
    text: str
    step: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def __init__(self, write_cfg: DPConfig, read_cfg: DPConfig, s1: DecisionBackend | None = None,
                 llm: LLMClient | None = None, max_items: int = 500, log_decisions: bool = False):
        self.items: list[MemoryItem] = []
        self.write_cfg = write_cfg
        self.read_cfg = read_cfg
        self.s1 = s1
        self.llm = llm
        self.max_items = max_items
        self.n_offered = 0
        self._next_id = 0  # monotonically increasing: ids stay unique after eviction
        self.log_decisions = log_decisions
        self.decisions: list[DecisionLog] = []      # pending logs; the harness takes them (``take_decisions``)
        self.shadow_errors: list[str] = []
        self.n_llm_calls = 0                        # LLM write-gate calls so far (the harness counts them per step)

    @staticmethod
    def label(m: MemoryItem) -> str:
        """The option label of a memory in the recall ``Choice``."""
        return f"[m{m.id}] {m.text[:60]}"

    def take_decisions(self) -> list[DecisionLog]:
        out, self.decisions = self.decisions, []
        return out

    def _log(self, d: DecisionLog) -> None:
        if self.log_decisions:
            d.gold_aux = {"v": 1, **(d.gold_aux or {})}
            self.decisions.append(d)

    # ---- write ---------------------------------------------------------------------
    def offer(self, text: str, context: str, ledger: Ledger | None = None, step: int | None = None,
              *, q_meta: dict[str, Any] | None = None, **meta: Any) -> bool:
        """Offer a candidate memory; the write policy decides whether to store it. ``q_meta`` goes into the gate's
        ``Question.meta`` (simulation studies only: ``memory_write.extra.sim_gold``); ``meta`` into the item."""
        self.n_offered += 1
        mode = self.write_cfg.mode
        keep = False
        dp = DP.MEMORY_WRITE.value
        state = f"{context}\n\nCANDIDATE MEMORY:\n{text}"
        inner = ledger if ledger is not None else Ledger()
        if mode == Mode.ALWAYS:
            keep = True
            self._log(DecisionLog(dp, "always", "store every offered observation", final_value=True, state=state))
        elif mode in (Mode.TRIGGER, Mode.REPLACE):   # E6e replace: System-1's argmax alone (= the trigger at 0.5)
            assert self.s1 is not None, f"memory_write={mode.value} needs a System-1 backend"
            q = Noul(Q_MEMORY_WRITE, decision_point=dp, **(q_meta or {}))
            tap = Tap(inner)
            ans = self.s1.ask(state, q, ledger=tap)
            keep = ans.p_yes >= (self.write_cfg.tau if mode == Mode.TRIGGER else 0.5)
            d = DecisionLog(dp, mode.value, q.text, None, keep, ans.confidence, ans.distribution, final_value=keep,
                            state=state)
            tap.stamp(d)
            self._log(d)
        elif mode == Mode.SHADOW:      # measurement: the gate is asked and logged, nothing is stored
            assert self.s1 is not None, "memory_write=shadow needs a System-1 backend"
            q = Noul(Q_MEMORY_WRITE, decision_point=dp, **(q_meta or {}))
            tap = Tap(inner, shadow=True)
            a: Answer | None = None
            try:
                a = self.s1.ask(state, q, ledger=tap)
            except Exception as e:  # noqa: BLE001 - fail-open like every shadow decision
                self.shadow_errors.append(f"memory_write: {type(e).__name__}: {str(e)[:200]}")
            d = DecisionLog(dp, "shadow", q.text, None, a.value if a else None, a.confidence if a else None,
                            a.distribution if a else None, deferred=True, final_value=False, state=state, shadow=True)
            tap.stamp(d)
            self._log(d)
        elif mode == Mode.LLM:
            keep, cost, lat = self._llm_gate(state, inner)
            d = DecisionLog(dp, "llm", "Should this be stored in long-term memory because it will likely be needed "
                            "later?", s2_value=keep, final_value=keep, state=state)
            d.s2_cost_usd, d.s2_latency_ms = cost, lat
            self._log(d)
        elif mode == Mode.CASCADE:     # E6b: System-1's answer when its confidence >= tau, else the LLM gate
            assert self.s1 is not None, "memory_write=cascade needs a System-1 backend"
            q = Noul(Q_MEMORY_WRITE, decision_point=dp, **(q_meta or {}))
            tap = Tap(inner)
            ans = self.s1.ask(state, q, ledger=tap)
            sure = ans.confidence >= self.write_cfg.tau
            d = DecisionLog(dp, "cascade", q.text, None, bool(ans.value), ans.confidence, ans.distribution,
                            deferred=not sure, state=state)
            tap.stamp(d)
            if sure:
                keep = bool(ans.value)
            else:
                keep, d.s2_cost_usd, d.s2_latency_ms = self._llm_gate(state, inner)
                d.s2_value = keep
            d.final_value = keep
            self._log(d)
        elif mode == Mode.GUARD:       # E6b: the LLM gate proposes; System-1 can veto a proposed write once
            keep, cost, lat = self._llm_gate(state, inner)
            d = DecisionLog(dp, "guard", Q_MEMORY_WRITE, s2_value=keep, state=state)
            d.s2_cost_usd, d.s2_latency_ms = cost, lat
            if keep:                   # a skip proposal needs no check: there is nothing to veto
                assert self.s1 is not None, "memory_write=guard needs a System-1 backend"
                q = Noul(Q_MEMORY_WRITE, decision_point=dp, **(q_meta or {}))
                tap = Tap(inner)
                ans = self.s1.ask(state, q, ledger=tap)
                keep = ans.p_yes >= self.write_cfg.tau
                d.s1_value, d.s1_conf, d.s1_dist, d.retry = keep, ans.confidence, ans.distribution, 0
                tap.stamp(d)
            d.final_value = keep
            self._log(d)
        elif mode == Mode.ASSIST:      # E6d: System-1's answer and confidence advise the LLM gate, which decides
            assert self.s1 is not None, "memory_write=assist needs a System-1 backend"
            q = Noul(Q_MEMORY_WRITE, decision_point=dp, **(q_meta or {}))
            tap = Tap(inner)
            ans = self.s1.ask(state, q, ledger=tap)
            keep, cost, lat = self._llm_gate(state, inner, advice=advice_note(bool(ans.value), ans.confidence))
            d = DecisionLog(dp, "assist", q.text, None, bool(ans.value), ans.confidence, ans.distribution,
                            deferred=True, s2_value=keep, final_value=keep, state=state)
            tap.stamp(d)
            d.s2_cost_usd, d.s2_latency_ms = cost, lat
            self._log(d)
        if keep:
            self.items.append(MemoryItem(id=self._next_id, text=text, step=step, meta=meta))
            self._next_id += 1
            if len(self.items) > self.max_items:
                self.items.pop(0)
        return keep

    def _llm_gate(self, state: str, inner: Ledger, advice: str | None = None) -> tuple[bool, float, float]:
        """The LLM write gate (``llm`` mode; the deferral of ``cascade``, the proposal of ``guard``, and with
        System-1's ``advice`` the decision of ``assist``): (store?, cost, latency) of one yes / no call through the
        episode ledger. Without advice the prompt is the reference gate's, byte for byte."""
        assert self.llm is not None
        prompt = (f"{state}\n\nShould this be stored in long-term memory because it will likely be needed "
                  "later? Answer yes or no.")
        if advice:
            prompt = (f"{state}\n\n{advice}\n\nShould this be stored in long-term memory because it will likely be "
                      "needed later? Answer yes or no.")
        n0 = len(inner.records)
        r = self.llm.chat([{"role": "user", "content": prompt}],
                          ledger=inner, decision_point=DP.MEMORY_WRITE.value, max_tokens=16, temperature=0.0)
        # (max_tokens=4 made Gemini 3.x return empty text, so the LLM gate stored nothing)
        self.n_llm_calls += 1
        rs = inner.records[n0:]
        return (r.text or "").strip().lower().startswith("y"), sum(x.cost_usd for x in rs), \
            sum(x.latency_ms for x in rs)

    # ---- read ----------------------------------------------------------------------
    def recall(self, query: str, ledger: Ledger | None = None) -> list[MemoryItem]:
        mode = self.read_cfg.mode
        if mode == Mode.OFF or not self.items:
            return []
        k = 1 if mode == Mode.REPLACE else self.read_cfg.k    # E6e replace: System-1's single top memory
        recent = self.items[-self.read_cfg.extra.get("recency_k", 20):]
        if mode == Mode.ALWAYS:
            return recent
        assert self.s1 is not None, f"memory_read={mode.value} needs a System-1 backend"
        tap = Tap(ledger if ledger is not None else Ledger())
        if mode == Mode.GUARD:         # E6d: the LLM selects, System-1 can veto the selection once
            return self._guarded_recall(query, recent, tap)
        if mode == Mode.TRIGGER:       # E6b: read (the recent memories, as ``always``) only when System-1 says so
            q = Noul(Q_MEMORY_READ_TRIGGER, decision_point=DP.MEMORY_READ.value)
            st = f"{query}\n\nSTORED MEMORIES: {len(self.items)} earlier observations (not shown)"
            ans = self.s1.ask(st, q, ledger=tap)
            fire = ans.p_yes >= self.read_cfg.tau
            d = DecisionLog(DP.MEMORY_READ.value, "trigger", q.text, None, fire, ans.confidence, ans.distribution,
                            final_value=fire, state=st, gold_aux={"n_injected": len(recent) if fire else 0})
            tap.stamp(d)
            self._log(d)
            return list(recent) if fire else []
        # tournament over chunks of <= max_options-1 (reserve one option for "none"); the chunk questions of a
        # round go out in one request when the backend batches (answers are independent per question)
        pool = list(self.items)
        chunk = self.s1.max_options - 1
        rounds: list[list[str]] = []
        last: tuple[Question, Answer] | None = None
        while len(pool) > chunk:
            groups = [pool[i:i + chunk] for i in range(0, len(pool), chunk)]
            winners: list[MemoryItem] = []
            for g, (picked, qa) in zip(groups, self._select_many(query, groups, k, tap)):
                winners.extend(picked)
                last = qa or last
            rounds.append([self.label(m) for m in winners])
            pool = winners
        out, qa = self._select_many(query, [pool], k, tap)[0]
        last = qa or last
        if last is not None:
            q, a = last
            deferred = mode == Mode.CASCADE and a.confidence < self.read_cfg.tau
            if deferred:              # E6b cascade: not sure of its top memory -> every recent memory (``always``)
                out = list(recent)
            d = DecisionLog(DP.MEMORY_READ.value, mode.value, q.text, list(q.options or []), a.value, a.confidence,
                            a.distribution, deferred=deferred, final_value=[self.label(m) for m in out], state=query,
                            gold_aux={"rounds": rounds} if rounds else None)
            tap.stamp(d)
            self._log(d)
        return out

    def _llm_select(self, query: str, pool: list[MemoryItem], ledger: Ledger,
                    note: str | None = None) -> list[MemoryItem]:
        """E6d: the LLM picks the stored memories the next step needs (ids from a numbered list; "none" = none), in
        one call through the ledger at ``memory_read``. Ids it invents are ignored."""
        assert self.llm is not None, "memory_read=guard needs the LLM"
        max_chars = int(self.read_cfg.extra.get("select_chars", 300))
        listing = "\n".join(f"- [m{m.id}] {m.text[:max_chars]}" for m in pool)
        prompt = f"{query}\n\nSTORED MEMORIES:\n{listing}\n\n" + (f"{note}\n\n" if note else "") + MEMORY_SELECT_PROMPT
        r = self.llm.chat([{"role": "user", "content": prompt}], ledger=ledger, decision_point=DP.MEMORY_READ.value,
                          max_tokens=64, temperature=0.0)
        self.n_llm_calls += 1
        ids = {int(x) for x in re.findall(r"\bm(\d+)\b", r.text or "")}
        return [m for m in pool if m.id in ids]

    def _guarded_recall(self, query: str, recent: list[MemoryItem], tap: Tap) -> list[MemoryItem]:
        """E6d ``memory_read: guard``: the LLM selects among the recent memories (the pool ``always`` would inject);
        System-1 is asked whether that selection is what the next step needs (state: the query plus the selected
        memories) and vetoes it when p_yes < tau; the LLM then selects once more with the rejection note, and that
        selection is injected unchecked (``guard_max_retries`` = 1, as every guard)."""
        assert self.s1 is not None
        pool = list(recent)
        n0 = len(tap.records)
        sel = self._llm_select(query, pool, tap)
        s2 = [r for r in tap.records[n0:] if r.component == "s2"]
        labels = [self.label(m) for m in sel]
        q = Noul(q_memory_read_guard(labels), decision_point=DP.MEMORY_READ.value)
        st = query + ("\n\nSELECTED MEMORIES:\n" + "\n".join(f"- [m{m.id}] {m.text[:600]}" for m in sel) if sel else "")
        n1 = len(tap.records)
        a = self.s1.ask(st, q, ledger=tap)
        s1recs = tap.records[n1:]
        ok = a.p_yes >= self.read_cfg.tau
        out = sel
        aux: dict[str, Any] = {"veto": not ok, "pool": len(pool)}
        if not ok:
            out = self._llm_select(query, pool, tap, note=MEMORY_REJECT_NOTE)
            aux["retry_selection"] = [self.label(m) for m in out]
        d = DecisionLog(DP.MEMORY_READ.value, "guard", q.text, None, ok, a.confidence, a.distribution, deferred=True,
                        s2_value=labels, final_value=[self.label(m) for m in out], state=st, gold_aux=aux, retry=0)
        d.s1_cost_usd, d.s1_latency_ms = sum(r.cost_usd for r in s1recs), sum(r.latency_ms for r in s1recs)
        d.s2_cost_usd, d.s2_latency_ms = sum(r.cost_usd for r in s2), sum(r.latency_ms for r in s2)
        self._log(d)
        return out

    def _question(self, pool: list[MemoryItem]) -> Question:
        """Options are short unique labels; the full memory text (up to ``extra.max_chars``) goes in the option
        description, which Jev receives as the Choice criterion and logprob deciders render next to the label."""
        extra = self.read_cfg.extra
        labels = [self.label(m) for m in pool]
        descs: list[str | None] = [m.text[: extra.get("max_chars", 1200)] for m in pool] + [None]
        return Choice(Q_MEMORY_READ, labels + [NONE_MEMORY], descs, decision_point=DP.MEMORY_READ.value)

    def _pick(self, pool: list[MemoryItem], ans: Answer, k: int) -> list[MemoryItem]:
        """Top-k memories by System-1 preference; memories ranked below the "none" option are dropped (so recall
        can be empty) unless ``extra.allow_empty`` is False."""
        ranked = []
        for o in ans.top_k(k + 1):
            if o == NONE_MEMORY:
                if self.read_cfg.extra.get("allow_empty", True):
                    break
                continue
            ranked.append(o)
        idx = {self.label(m): m for m in pool}
        return [idx[o] for o in ranked[:k] if o in idx]

    def _select_many(self, query: str, groups: list[list[MemoryItem]], k: int,
                     ledger: Ledger) -> list[tuple[list[MemoryItem], tuple[Question, Answer] | None]]:
        assert self.s1 is not None
        asks = [(i, self._question(g)) for i, g in enumerate(groups) if len(g) > k]
        if len(asks) > 1 and getattr(self.s1, "batched_questions", False):
            answers = self.s1.decide(query, [q for _, q in asks], ledger=ledger).answers
        else:
            answers = [self.s1.ask(query, q, ledger=ledger) for _, q in asks]
        got = {i: (q, a) for (i, q), a in zip(asks, answers)}
        return [(self._pick(g, got[i][1], k), got[i]) if i in got else (g, None) for i, g in enumerate(groups)]

    def _select(self, query: str, pool: list[MemoryItem], k: int, ledger: Ledger | None) -> list[MemoryItem]:
        """Top-k memories of one pool (kept for callers of the pre-batching API)."""
        return self._select_many(query, [pool], k, ledger if ledger is not None else Ledger())[0][0]

    def render(self, items: list[MemoryItem]) -> str:
        return "\n".join(f"- [m{m.id}] {m.text}" for m in items)
