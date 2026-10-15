"""Structured episode logs. These are (1) the benchmark's raw results, (2) the input of the
analysis scripts, and (3) the source of decision-point training data (``training/harvest.py``)."""

from __future__ import annotations

import gzip
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class DecisionLog:
    dp: str
    mode: str
    question: str
    options: list[str] | None = None
    s1_value: Any = None
    s1_conf: float | None = None
    s1_dist: dict[str, float] | None = None
    deferred: bool = False            # cascade: S1 not confident -> LLM decided (shadow: always True)
    s2_value: Any = None              # what the LLM chose (when consulted)
    final_value: Any = None           # what the harness executed
    gold: Any = None                  # oracle label, when the env exposes one
    state: str | None = None          # S1 input state (kept for training-data harvest)
    # --- measurement fields (threshold theory, ``bench/threshold_theory.py``); None when not measured
    shadow: bool = False              # S1 was asked but never acted on (``shadow`` mode)
    s1_cost_usd: float | None = None  # cost / latency of the S1 call(s) behind ``s1_value`` (tournaments summed)
    s1_latency_ms: float | None = None
    s2_correct: bool | None = None    # the LLM's own decision vs. the *raw* oracle value (copyable arguments
                                      # outside the candidate set included, which ``gold`` maps to OTHER)
    s2_cost_usd: float | None = None  # cost / latency of the LLM call(s) that made ``s2_value`` (for tool
    s2_latency_ms: float | None = None  # arguments and stop in the LLM-only loop: the step's tool call)
    # --- meta-decision golds (docs/META_GOLD_SPEC.md §8); None for legacy logs
    gold_status: str | None = None    # "ok" | "doomed" | "unobserved" | "no_oracle" | "cf_error"
    gold_aux: dict[str, Any] | None = None   # always carries ``v`` (spec version); per-dp keys (§3), CF keys (§4),
                                             # ``param_kind`` for tool_params, probe keys (§6.1) for probe records
    # --- round-3 review fields; None for older logs
    s1_rank: list[str] | None = None  # Choice over more options than one call takes (tournament): System-1's full
                                      # ranking -- finalists by the final round, then the options eliminated in each
                                      # earlier round by their score there (C3 recall@k reads it; ``s1_dist`` holds
                                      # only the finalists)
    retry: int | None = None          # guard records: the number of proposals this guard already rejected before this
                                      # one (stop: in the episode; tool guard: at this step); 0 = a first proposal
    s1_calls: int | None = None       # pass accounting of this decision (review P1-4; decision.tournament.
    s1_rounds: int | None = None      # decision_pass_counts): System-1 calls, sequential rounds (harness tournament x
    s1_passes: int | None = None      # backend knockout) and backend-reported forward passes (None when a backend does
                                      # not report them, e.g. hosted Jev and LLM APIs)


@dataclass
class StepLog:
    step: int
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    model: str | None = None
    decisions: list[DecisionLog] = field(default_factory=list)
    reflection: str | None = None
    llm_calls: int = 0
    oracle_tool: str | None = None


@dataclass
class Trajectory:
    task_id: str
    config: str
    backend: str | None
    llm: str | None
    steps: list[StepLog] = field(default_factory=list)
    final_answer: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    wall_ms: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # measurement-only records (guard probes, the ``wording_v1`` reflection ablation, augmented stop-trigger
    # queries; META_GOLD_SPEC §1.5 / §6.1). Kept out of ``steps``: no episode-level or decision-table consumer
    # (decision_records, trigger_rates, threshold_theory, training harvest) reads this list.
    probes: list[DecisionLog] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.result.get("success", False))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        line = json.dumps(self.to_dict(), default=str) + "\n"
        with open(path, "ab+") as f:
            # a run killed mid-write leaves a partial last line; start on a fresh line so this record
            # is not glued onto it (which would make both unparseable and the task re-run forever)
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    line = "\n" + line
            f.write(line.encode())


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Every decodable record of a trajectory JSONL in file order, duplicates included. Undecodable lines (a
    partial record left by a killed run) are skipped with a warning. A ``.gz`` path is read decompressed (the
    slimmed, gzipped records an analysis keeps, e.g. E13's)."""
    bad = 0
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        warnings.warn(f"{path}: skipped {bad} undecodable line(s) (partial write?)", stacklevel=3)


def latest_by_task(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """task_id -> the *last* record of that task. The resumable runner appends a fresh record when it re-runs
    a task (a partial write, or an episode that ended in an infrastructure error), so the last one wins; genuine
    repeated trials carry distinct ``#<k>`` task ids and are unaffected."""
    last: dict[str, dict[str, Any]] = {}
    for r in records:
        last[str(r.get("task_id"))] = r
    return last


def load_trajectories(path: str | Path, dedupe: bool = True) -> list[dict[str, Any]]:
    """Records of a trajectory JSONL, one per ``task_id`` (the last one written; see :func:`latest_by_task`),
    in order of first appearance. Undecodable lines (a partial record left by a killed run, whose task the
    resumable runner re-runs) are skipped with a warning. ``dedupe=False`` returns every record."""
    recs = list(iter_records(path))
    return list(latest_by_task(recs).values()) if dedupe else recs
