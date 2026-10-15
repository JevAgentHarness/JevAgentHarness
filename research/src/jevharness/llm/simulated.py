"""Simulated System-2 LLM for environments that expose an oracle (JevGym).

``SimulatedLLM`` follows ``env.oracle_action()`` and makes a mistake with probability
``error_rate`` (wrong tool, or a corrupted argument). Together with ``SimulatedBackend`` it lets
us run the *whole* harness offline to (a) test the code path, (b) run simulation studies of the
design space ("if the LLM errs 10% of the time and S1 is 85% accurate and well calibrated,
which integration mode wins?") before spending API budget. Never use it for headline numbers.

The agent calls ``bind_env(env)`` at the start of each episode (thread-local, so the runner
can parallelise episodes).
"""

from __future__ import annotations

import random
import threading
import uuid
from typing import Any

from .base import ChatResponse, ToolCall
from .scripted import ScriptedLLM


class SimulatedLLM(ScriptedLLM):
    provider = "simulated"

    def __init__(self, model: str = "scripted", error_rate: float = 0.1, simulated_latency_ms: float = 1500.0,
                 seed: int = 0, **defaults: Any):
        super().__init__(model=model, policy=None, simulated_latency_ms=simulated_latency_ms, **defaults)
        self.error_rate = error_rate
        self._local = threading.local()
        self._seed = seed

    def bind_env(self, env: Any) -> None:
        self._local.env = env
        self._local.rng = random.Random(f"{self._seed}-{getattr(env, 'task_id', '')}")

    def _policy(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResponse:
        env = getattr(self._local, "env", None)
        rng: random.Random = getattr(self._local, "rng", random.Random(0))
        if env is None or not tools:
            return ChatResponse(text="I think the task is done.")
        oracle = env.oracle_action()
        names = [t["function"]["name"] for t in tools]
        if oracle is None:
            return ChatResponse(text="done")
        tool, args = oracle
        forced = None
        for m in reversed(messages):
            if m.get("role") == "user" and "You must now call `" in str(m.get("content")):
                forced = str(m["content"]).split("`")[1]
                break
        if forced:
            tool = forced
        if tool not in names:  # e.g. assist mode pruned the gold tool away
            tool = rng.choice(names)
            args = {}
        elif rng.random() < self.error_rate:
            if rng.random() < 0.5 and len(names) > 1 and not forced:
                tool = rng.choice([n for n in names if n != tool])
                args = {}
            else:
                args = dict(args)
                if args:
                    k = rng.choice(list(args))
                    args[k] = "WRONG" if isinstance(args[k], str) else -1
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=tool,
                                                          arguments=args)])

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        self.policy = self._policy
        return super()._chat(messages, tools, **kw)
