"""Registered JevGym seed namespaces (docs/PREREGISTRATION.md A6.4, A6.13) in one place, so the runner, the
harvest guard, the calibration tools and ``scripts/prereg_check.py`` agree.

* ``EVAL_SEED_RANGES``: the frozen evaluation suite (500 tasks). Running an experiment whose suite touches it needs
  ``JEV_EVAL_RUNS=1`` in the environment (the runner refuses otherwise), so no pilot, resume or smoke test can expose
  evaluation tasks by accident again (A6.13: the first range, 100000-100499, was exposed by an n = 50 pilot resume).
* ``RETIRED_EVAL_RANGES``: exposed former evaluation ranges. Reserved: never evaluated, never trained on, never run.
* ``CALIBRATION_RANGE``: calibration shadow arms (A6.5), ``DEV_MAX_SEED``: pilot / dev seeds below it.
* ``TRAIN_MIN_SEED``: training trajectories live at or above it, minus every reserved range.
* ``STAGE2_EVAL_RANGES`` / ``STAGE2_RESERVED_RANGES`` (docs/PREREG_A7_STAGE2.md, A7.2 (a); reserved 2026-10-02): the
  Stage-2 confirmation runs on the fresh seeds 300000-300499; the whole block 300000-300999 is never trained on and
  never piloted. The runner refuses any jevgym suite touching the block without ``JEV_EVAL_RUNS=1``, and
  ``training_exclusions()`` (the harvest / data / OPD guards) excludes it. ``reserved_ranges()`` keeps its Stage-1
  meaning (evaluation + retired, PREREGISTRATION A6.13), which ``docs/PREREG_EFFECTIVE.md`` is generated from.
"""
from __future__ import annotations

import os

EVAL_SEED_RANGES: tuple[tuple[int, int], ...] = ((200000, 200499),)
RETIRED_EVAL_RANGES: tuple[tuple[int, int], ...] = ((100000, 100499),)
CALIBRATION_RANGE: tuple[int, int] = (5000, 5999)
DEV_MAX_SEED = 999
TRAIN_MIN_SEED = 10000
EVAL_RUNS_ENV = "JEV_EVAL_RUNS"
EVAL_TASKS_FILE = "configs/eval/jevgym_eval_tasks.json"
STAGE2_EVAL_RANGES: tuple[tuple[int, int], ...] = ((300000, 300499),)       # A7.2 (a): the Stage-2 fresh seeds
STAGE2_RESERVED_RANGES: tuple[tuple[int, int], ...] = ((300000, 300999),)   # A7.2 (a): never trained on, never piloted
STAGE2_TASKS_FILE = "configs/eval/jevgym_stage2_tasks.json"
# Exploratory row E7, JevGym-Hard (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06): its evaluation seeds, 310000-310199
# for E7, 310200-310399 for its addendum E7b, 310400-310599 for E7c (and E7e) and 310600-310799 for E7f (Deviations
# 2026-10-07 "E7f"), and 311000-311299 for E13, one block of 100 per agent LLM (Deviations 2026-10-07 "E13: other
# agent LLMs on DecisionGym-Hard"). The runner refuses them without ``JEV_EVAL_RUNS=1`` (no pilot, calibration or smoke
# can touch them); nothing else reads this range.
# Exploratory row E21, DecisionGym-Shift (Deviations 2026-10-08 "E21: a shifted DecisionGym split"): its evaluation block
# 330000-330999 (L2 source tasks 330000-330099, live tasks 330100-330299, the rest a reserve) is refused without
# ``JEV_EVAL_RUNS=1`` like the E7 / E13 seeds; its development seeds 339000-339099 (debugging, never reported) are not.
E21_EVAL_RANGES: tuple[tuple[int, int], ...] = ((330000, 330999),)
E21_DEV_RANGE: tuple[int, int] = (339000, 339099)
EXPLORE_EVAL_RANGES: tuple[tuple[int, int], ...] = ((310000, 310799), (311000, 311299)) + E21_EVAL_RANGES
# Never trained on (E7d, Deviations 2026-10-06 "E7d"): the whole JevGym-Hard evaluation block and E7d's held-out states;
# E13's block 311000-311999 (its evaluation seeds 311000-311299 and a reserve). E7d's own training seeds
# (320000-320599) are ordinary training seeds and stay allowed.
# E21's whole block 330000-339999 (evaluation, reserve and development seeds) is never trained on.
EXPLORE_TRAIN_EXCLUDED: tuple[tuple[int, int], ...] = ((310000, 310999), (311000, 311999), (321000, 321999),
                                                       (330000, 339999))


def in_ranges(seed: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= seed <= hi for lo, hi in ranges)


def reserved_ranges() -> tuple[tuple[int, int], ...]:
    """Ranges no training / calibration / development loader may touch (evaluation + retired)."""
    return EVAL_SEED_RANGES + RETIRED_EVAL_RANGES


def training_exclusions() -> tuple[tuple[int, int], ...]:
    """Every JevGym range a training / harvest / OPD loader must skip: the Stage-1 reserved ranges plus the Stage-2
    block 300000-300999 (A7.2 (a)), the JevGym-Hard blocks of E7 / E7d / E13 and E21's block (``EXPLORE_TRAIN_EXCLUDED``). The
    calibration range is excluded separately where it applies."""
    return reserved_ranges() + STAGE2_RESERVED_RANGES + EXPLORE_TRAIN_EXCLUDED


def ranges_spec(ranges: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{lo}-{hi}" for lo, hi in ranges)


def overlaps(seed0: int, n: int, ranges: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    """The ranges that ``seed0 .. seed0 + n - 1`` intersects."""
    last = seed0 + max(0, int(n)) - 1
    return [(lo, hi) for lo, hi in ranges if seed0 <= hi and last >= lo]


def eval_runs_allowed() -> bool:
    return os.environ.get(EVAL_RUNS_ENV, "") in ("1", "true", "yes")


def check_suite_seeds(name: str, seed0: int, n: int) -> None:
    """Raise ``SystemExit`` when a jevgym suite would run on reserved seeds without ``JEV_EVAL_RUNS=1``."""
    if name != "jevgym":
        return
    hit = overlaps(int(seed0), int(n), EVAL_SEED_RANGES)
    retired = overlaps(int(seed0), int(n), RETIRED_EVAL_RANGES)
    stage2 = overlaps(int(seed0), int(n), STAGE2_RESERVED_RANGES)
    explore = overlaps(int(seed0), int(n), EXPLORE_EVAL_RANGES)
    if explore and not eval_runs_allowed():
        rng = ", ".join(f"{lo}-{hi}" for lo, hi in explore)
        raise SystemExit(f"refusing to run suite jevgym seed0={seed0} n={n}: it touches the exploratory evaluation seeds "
                         f"{rng} (E7 / E13 / E21, docs/PREREG_A7_STAGE2.md Deviations 2026-10-06 / 2026-10-07 / 2026-10-08). The "
                         f"registered run sets {EVAL_RUNS_ENV}=1; development uses other seeds.")
    if stage2 and not eval_runs_allowed():
        rng = ", ".join(f"{lo}-{hi}" for lo, hi in stage2)
        raise SystemExit(f"refusing to run suite jevgym seed0={seed0} n={n}: it touches the reserved STAGE-2 seeds {rng} "
                         f"(docs/PREREG_A7_STAGE2.md A7.2 (a): never piloted, never trained on). The Stage-2 runs set "
                         f"{EVAL_RUNS_ENV}=1; development uses seeds <= {DEV_MAX_SEED} (e.g. -o suite.seed0=0).")
    if (hit or retired) and not eval_runs_allowed():
        what = "the registered EVALUATION seeds" if hit else "a RETIRED (exposed) evaluation range"
        rng = ", ".join(f"{lo}-{hi}" for lo, hi in (hit or retired))
        raise SystemExit(f"refusing to run suite jevgym seed0={seed0} n={n}: it touches {what} {rng} "
                         f"(PREREGISTRATION A6.13). Main runs after the freeze set {EVAL_RUNS_ENV}=1; pilots, smoke "
                         f"tests and dev runs use seeds <= {DEV_MAX_SEED} (e.g. -o suite.seed0=0).")
