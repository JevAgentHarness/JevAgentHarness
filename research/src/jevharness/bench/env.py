"""Environment protocol consumed by the harness.

Two families are supported:

* tool-calling envs  -- ``tools()`` returns OpenAI function specs, ``step(name, args)``;
* action-only envs   -- ``admissible_actions()`` returns the current discrete action set
                        (ALFWorld / ScienceWorld / TextCraft style), ``step("act", {"action": a})``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    observation: str
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Env(ABC):
    task_id: str = ""
    instruction: str = ""
    # gold annotations are optional; when present they power decision-level metrics and
    # the simulated System-1 backends (never shown to the agent)
    gold: dict[str, Any] = {}

    @abstractmethod
    def reset(self) -> str: ...

    def tools(self) -> list[dict[str, Any]]:
        return []

    def admissible_actions(self) -> list[str] | None:
        return None

    @abstractmethod
    def step(self, tool: str, args: dict[str, Any]) -> StepResult: ...

    def oracle_action(self) -> tuple[str, dict[str, Any]] | None:
        """Optional: the gold next (tool, args) from the current state. Used only for
        decision-level metrics and simulated System-1 backends -- never shown to the agent."""
        return None

    def oracle_done(self) -> bool | None:
        """Optional: whether the task is already solved in the current state."""
        return None

    @abstractmethod
    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        """Return at least {"success": bool, "score": float}."""

    def close(self) -> None:  # pragma: no cover - optional
        pass
