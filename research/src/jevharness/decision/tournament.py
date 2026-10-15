"""Knockouts over Choice option sets larger than one read-out, and their pass accounting (review P1-4).

**One read is bounded.** A decision model reads a Choice by scoring its options in one pass only up to a fixed width:
52 letter codes A..Z a..z (``types.OPTION_LABELS``: our trained deciders behind ``training/serve_jev.py``, the logprob
and verbalised LLM deciders, hence the harness tournament for them), 62 symbols for Intern-Decision
(``intern_decision.py``), 255 options per *request* for hosted Jev (its internal passes are not observable). A Choice
with more options is answered by a **knockout**: the options are split in order into chunks of at most the width,
every chunk is read, the top ``keep`` (2) of each chunk advance, the rounds repeat until the pool fits one read, and a
final read over the finalists decides. Such a decision costs several reads in ``1 + R`` sequential rounds -- it is
*not* one forward pass. :func:`knockout_plan` gives the schedule; for the width-52 deciders:

    options    rounds   reads (chunk reads + final)   e.g.
    <= 52      1        1                             10, 52: one read
    53         2        2                             chunks 52 + 1 (the single option advances unread) -> 3 finalists
    54-1352    2        ceil(n / 52) + 1 (*)          69 tools (JevGym): 52 + 17 -> 4 finalists -> 3 reads;
                                                      200: 52, 52, 52, 44 -> 8 finalists -> 5 reads
    > 1352     3+       ...                           (a Choice holds at most 255 options)
    (*) one read less when the last chunk holds a single option (it advances unread)

(Intern-Decision, width 62: 69 options -> 62 + 7 -> 4 finalists; its multi-field prompt reads all chunks of a round
in one forward pass, so 2 passes. The harness tournament, ``harness/agent.py:_choice_tournament``, reads a chunk only
when it holds more than ``keep`` options; the in-backend knockouts read every chunk of two or more.) The backends
record what they actually did:
``Usage.n_passes`` (forward passes, where observable) and ``Usage.n_rounds`` (rounds) per ``decide`` call -- copied
into the ledger record's ``meta`` -- and a knockout answer carries ``raw["knockout"]`` = ``{"rounds", "reads",
"finalists", "full_set"}``. A harness decision that makes several ``decide`` calls (one per tournament round) is
summed by :func:`decision_pass_counts` over its ledger records.

**Full-set distribution** (:func:`full_set_distribution`). Each read is a softmax over its own chunk, so the reads
are on different scales; the final read's probabilities sum to one over the finalists alone. Reporting them as the
distribution over all options (``method="finalists"``, the behaviour of every knockout before 2026-10-02) puts zero
mass on every eliminated option and turns the confidence of a many-option decision into a probability *conditional
on the finalists* -- systematically higher than a full-set probability. ``method="linked"`` (the default) puts all
reads on one scale through the options consecutive rounds share, Luce-style:

1. a finalist's strength is its final-read probability;
2. walking the rounds backwards, every read of a chunk gets the scale factor
   ``s = sum(strength of its survivors) / sum(chunk probability of its survivors)`` (its survivors' strengths are
   known from the later round), and every option it eliminated gets ``strength = chunk probability * s``;
3. the full-set distribution is the strengths normalised over all options.

Properties: (i) when the reads are Luce-consistent -- the ratio p(a)/p(b) does not depend on which other options a
read shows, as for a softmax over per-option scores (logits / T) that do not depend on the other options -- the result
equals the single-read softmax over the full set exactly; (ii) the finalists keep the final read's ranking and odds
(only their total share shrinks by the mass earlier reads gave to eliminated options), so the argmax never changes:
an eliminated option's strength is at most the mean of its chunk survivors' strengths; (iii) every option gets mass,
so rankings (recall@k) and calibration can be computed over the full set. Real readouts are only approximately
Luce-consistent (an option's code logit depends on its position and neighbours; ``serve_jev`` uses a per-width
temperature), and temperature scaling is fitted on single reads, so full-set calibration is not guaranteed by
construction: report it measured, stratified by the number of rounds.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

KNOCKOUT_METHODS = ("linked", "finalists")
DEFAULT_KNOCKOUT = "linked"

# one read of a knockout round: (the chunk's options, its read probabilities by option, the options it let advance)
Read = tuple[Sequence[Hashable], Mapping[Hashable, float], Sequence[Hashable]]


def check_method(method: str) -> str:
    if method not in KNOCKOUT_METHODS:
        raise ValueError(f"knockout_probs must be one of {KNOCKOUT_METHODS}, got {method!r}")
    return method


def knockout_plan(n: int, width: int, keep: int = 2, read_if_more_than: int = 1) -> dict[str, Any]:
    """The read schedule of a knockout over ``n`` options with reads of at most ``width`` options.

    ``read_if_more_than``: a chunk is read only when it holds more options than this (in-backend knockouts: 1, so a
    single leftover option advances unread; the harness tournament: ``keep``). Returns ``{"rounds": 1 + elimination
    rounds, "reads": chunk reads + the final read, "chunks": [[chunk sizes] per elimination round], "finalists": n of
    the final read}``."""
    if width < 2:
        raise ValueError("width must be >= 2")
    keep = max(1, min(int(keep), width - 1))
    pool, chunks_per_round, reads = int(n), [], 0
    while pool > width:
        sizes = [min(width, pool - a) for a in range(0, pool, width)]
        chunks_per_round.append(sizes)
        reads += sum(1 for s in sizes if s > read_if_more_than)
        nxt = sum(min(s, keep) if s > read_if_more_than else s for s in sizes)
        if nxt >= pool:  # nothing to eliminate (cannot happen with keep < width)
            break
        pool = nxt
    return {"rounds": len(chunks_per_round) + 1, "reads": reads + 1, "chunks": chunks_per_round, "finalists": pool}


def full_set_distribution(options: Sequence[Hashable], rounds: Sequence[Sequence[Read]],
                          final: Mapping[Hashable, float], method: str = DEFAULT_KNOCKOUT) -> dict[Hashable, float]:
    """The distribution over every option of a knockout (module docstring): ``rounds`` lists, per elimination round in
    order, that round's reads ``(chunk options, read probabilities by option, survivors)``; ``final`` is the final read
    over the finalists. Options a round let advance without reading them need no entry. ``method="finalists"`` gives
    the legacy distribution (final read, zero elsewhere)."""
    check_method(method)
    strength: dict[Hashable, float] = {o: float(final.get(o, 0.0)) for o in options}
    if method == "finalists" or not rounds:
        return _normalised(strength, options)
    known: set[Hashable] = {o for o in final}
    for rnd in reversed(rounds):
        for chunk, probs, survivors in rnd:
            surv = [o for o in survivors if o in known]
            num = sum(strength[o] for o in surv)
            den = sum(max(float(probs.get(o, 0.0)), 0.0) for o in surv)
            scale = num / den if den > 0 and math.isfinite(num / den) else 0.0
            for o in chunk:
                if o in survivors:
                    continue
                strength[o] = max(float(probs.get(o, 0.0)), 0.0) * scale
                known.add(o)
    return _normalised(strength, options)


def _normalised(strength: Mapping[Hashable, float], options: Sequence[Hashable]) -> dict[Hashable, float]:
    z = sum(strength[o] for o in options)
    if not (z > 0 and math.isfinite(z)):
        return {o: 1.0 / len(options) for o in options}
    return {o: strength[o] / z for o in options}


def _get(rec: Any, key: str, default: Any = None) -> Any:
    return rec.get(key, default) if isinstance(rec, Mapping) else getattr(rec, key, default)


def decision_pass_counts(records: Iterable[Any]) -> dict[str, int | None]:
    """Pass accounting of ONE harness decision from its System-1 ledger records (``Tap.records``; ``CallRecord``
    objects or their dicts in ``Trajectory.calls``):

    * ``s1_calls``  -- System-1 ``decide`` calls (one per tournament round when the backend batches the round's chunks;
      one per chunk otherwise; cache replays included: they replay the original call);
    * ``s1_rounds`` -- sequential rounds: per harness tournament round (``CallRecord.tournament_round``, a plain call is
      its own round) the largest backend-reported ``meta["n_rounds"]`` (1 when unreported), summed over rounds;
    * ``s1_passes`` -- the sum of the backend-reported forward passes ``meta["n_passes"]``; None when any call did not
      report them (hosted Jev, LLM APIs: count requests there, ``s1_calls`` / ``meta["n_calls"]``)."""
    recs = [r for r in records if _get(r, "component", "s1") == "s1"]
    if not recs:
        return {"s1_calls": 0, "s1_rounds": 0, "s1_passes": None}
    by_round: dict[Any, int] = {}
    passes: int | None = 0
    for i, r in enumerate(recs):
        meta = _get(r, "meta") or {}
        rnd = _get(r, "tournament_round")
        key = ("t", rnd) if rnd is not None else ("call", i)
        by_round[key] = max(by_round.get(key, 0), int(meta.get("n_rounds") or 1))
        p = meta.get("n_passes")
        passes = None if passes is None or p is None else passes + int(p)
    return {"s1_calls": len(recs), "s1_rounds": sum(by_round.values()), "s1_passes": passes}


__all__ = ["DEFAULT_KNOCKOUT", "KNOCKOUT_METHODS", "check_method", "decision_pass_counts", "full_set_distribution",
           "knockout_plan"]
