"""Suite registry. Episode suites return a list of zero-arg Env factories (fresh env per run).

``jevgym`` options (all optional; see ``suites/jevgym.py``):
  n (50), seed0 (0), family (None = round-robin), families ("v1" | "v2" | "all" | list),
  n_distractors (10), distractor_style ("random" | "near" | "mixed"), noise_lines (0), strict (False),
  ask_tool (None = iff a v2 family is in play), multi_customers (3); multi_rule ("given" | "amount" | "tier": the
  JevGym-Hard order policy of E7, ``suites/jevgym_hard.py``; only when given); shift (true | {surface, distraction,
  structure}: DecisionGym-Shift of E21, ``suites/jevgym_shift.py``; only when given).
Unknown options raise ``ValueError`` (a typo would otherwise silently run the default suite).
"""

from __future__ import annotations

from typing import Any, Callable

from ..env import Env

EnvFactory = Callable[[], Env]


def load_suite(name: str, **kw: Any) -> list[EnvFactory]:
    if name == "jevgym":
        if "shift" in kw:           # DecisionGym-Shift (E21): its own module; the frozen generator is unchanged
            if "multi_rule" in kw or "answer_style" in kw:
                raise ValueError("shift and multi_rule / answer_style are separate suites; give one of them")
            from .jevgym_shift import task_factories as shift_factories
            return shift_factories(**kw)  # type: ignore[return-value]
        if "multi_rule" in kw:      # JevGym-Hard (E7): its own module, so the frozen generator jevgym.py is unchanged
            from .jevgym_hard import task_factories as hard_factories
            return hard_factories(**kw)  # type: ignore[return-value]
        from .jevgym import task_factories
        return task_factories(**kw)  # type: ignore[return-value]
    if name == "alfworld":
        from .alfworld import load_alfworld
        return load_alfworld(**kw)
    if name == "tau2":
        raise ValueError("tau2 runs through its own orchestrator: see jevharness.bench.suites.tau2 and "
                         "scripts/run_tau2.sh")
    raise ValueError(f"unknown episode suite {name!r}")
