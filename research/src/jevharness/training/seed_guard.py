"""JevGym seed rules for training the System-1 decider (JevAgentRL, docs/SCOPE_V2.md; round 5).

The registered namespaces live in ``bench/registry.py``; this module adds the Stage-2 reservation and turns the rules
into checks that every training loader (``opd.py``, ``grpo_online.py``, ``data_v1.py``, ``harvest.py``) applies where
seeds are drawn:

* **train**: seeds >= ``TRAIN_MIN_SEED`` (10000), never in the evaluation range (200000-200499), the retired range
  (100000-100499), the calibration range (5000-5999) or the Stage-2 range ``STAGE2_RANGE`` (300000-300999: fresh
  confirmation seeds 300000-300499 plus a reserve; never trained on, never piloted).
* **dev**: development and go / no-go gates run on ``DEV_RANGES`` = 0-999 and 1000-4999 only (greedy in-training
  evaluations, E2 development slices).
* **calibration**: 5000-5149 (``CAL_FIT_RANGE``) only for the tau-hat* fit of a frozen System-1 with the registered
  command (as for Intern-Decision, ``configs/experiments/jevgym_shadow_cal_intern.yaml``); never for training or
  model selection.
"""
from __future__ import annotations

from typing import Sequence

from ..bench.registry import (CALIBRATION_RANGE, EXPLORE_TRAIN_EXCLUDED, TRAIN_MIN_SEED, overlaps, ranges_spec,
                              reserved_ranges)

Range = tuple[int, int]

STAGE2_RANGE: Range = (300000, 300999)
DEV_RANGES: tuple[Range, ...] = ((0, 999), (1000, 4999))
CAL_FIT_RANGE: Range = (5000, 5149)
# never a training seed (on top of seed < TRAIN_MIN_SEED)
TRAIN_FORBIDDEN: tuple[Range, ...] = reserved_ranges() + (CALIBRATION_RANGE, STAGE2_RANGE) + EXPLORE_TRAIN_EXCLUDED
# never touched at all in the training phase (evaluation happens later, after the Stage-2 registration)
NEVER_RUN: tuple[Range, ...] = reserved_ranges() + (STAGE2_RANGE,)


def train_exclude_spec() -> str:
    """``harvest.py --exclude-jevgym-seeds`` value for training data: every forbidden range."""
    return ranges_spec(TRAIN_FORBIDDEN)


def check_train_pool(seed0: int, n: int, what: str = "training pool") -> None:
    """Raise ``ValueError`` unless every seed of ``seed0 .. seed0 + n - 1`` is a training seed."""
    if n <= 0:
        raise ValueError(f"{what}: empty seed plan (n={n})")
    if seed0 < TRAIN_MIN_SEED:
        raise ValueError(f"{what}: JevGym training seeds start at {TRAIN_MIN_SEED}; got seed0={seed0} "
                         "(0-4999 = development, 5000-5999 = calibration)")
    hit = overlaps(seed0, n, TRAIN_FORBIDDEN)
    if hit:
        raise ValueError(f"{what}: seeds {seed0}-{seed0 + n - 1} overlap reserved range(s) {hit} (evaluation / "
                         "retired / calibration / Stage-2; training/seed_guard.py)")


def in_dev(seed0: int, n: int) -> bool:
    last = seed0 + max(1, int(n)) - 1
    return any(lo <= seed0 and last <= hi for lo, hi in DEV_RANGES)


def check_dev_pool(seed0: int, n: int, what: str = "development pool") -> None:
    """Raise ``ValueError`` unless ``seed0 .. seed0 + n - 1`` lies inside one development range (0-999 or
    1000-4999)."""
    if n <= 0:
        raise ValueError(f"{what}: empty seed plan (n={n})")
    if not in_dev(seed0, n):
        raise ValueError(f"{what}: seeds {seed0}-{seed0 + n - 1} are not inside a development range "
                         f"{list(DEV_RANGES)} (calibration 5000-5149 is only for the tau-hat* fit; evaluation and "
                         "Stage-2 seeds are never run in the training phase)")


def check_cal_pool(seed0: int, n: int, what: str = "calibration pool") -> None:
    """Raise ``ValueError`` unless ``seed0 .. seed0 + n - 1`` lies inside the registered calibration seeds."""
    lo, hi = CAL_FIT_RANGE
    if n <= 0 or seed0 < lo or seed0 + n - 1 > hi:
        raise ValueError(f"{what}: seeds {seed0}-{seed0 + n - 1} are not inside the calibration seeds {lo}-{hi}")


def check_disjoint(a: Sequence[int], b: Sequence[int], what: str = "train / dev") -> None:
    """``a`` and ``b`` = (seed0, n) pairs that must not share a seed."""
    (s0, n0), (s1, n1) = a, b
    if s0 <= s1 + n1 - 1 and s1 <= s0 + n0 - 1:
        raise ValueError(f"{what}: seeds {s0}-{s0 + n0 - 1} and {s1}-{s1 + n1 - 1} overlap")
