"""Decision models: the small, fast "System 1" that answers one typed multiple-choice question per agent step.

Backends:

``jev``    any server that speaks the Jev System-One wire format (``POST {base}/v1/systemone``): TypeSafe's hosted Jev,
           Jev through OpenRouter, or a self-hosted decision model. The answer is a probability per option.
``rules``  no model at all: transparent heuristics over the agent's last tool result (errors, failing tests, repeated
           actions) and the action pre-filter's risky patterns. Useful as a baseline and when no decision-model key
           is available; it answers the routing, reflection and action questions only.
``none``   never answers (the router then falls back to its ``on_error`` tier).

A backend returns a :class:`Decision` with ``p``: the probability of every option. Failures raise; the router turns
them into its fail-safe tier, so a decision-model outage never stops the agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from aiohttp import ClientSession, ClientTimeout

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}

PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {"base_url": "https://openrouter.ai/api", "key_env": "OPENROUTER_API_KEY",
                   "model": "typesafe/jev-1.13"},
    "typesafe": {"base_url": "https://api.typesafe.ai", "key_env": "TYPESAFE_API_KEY", "model": "jev-latest"},
    "local": {"base_url": "http://localhost:8090", "key_env": "", "model": "agentdecision-4b"},
}
# Jev bills input tokens only; OpenRouter reports the billed amount itself (usage.cost), so this rate is a fallback
JEV_PRICE_PER_M_INPUT = 0.10


@dataclass
class Question:
    id: str
    text: str
    options: list[str]
    descriptions: list[str] = field(default_factory=list)

    def wire(self) -> dict[str, Any]:
        descs = self.descriptions or [None] * len(self.options)
        return {"type": "choice", "instructions": self.text,
                "criteria": {o: (d or None) for o, d in zip(self.options, descs)}}


@dataclass
class Decision:
    p: dict[str, float]
    model: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0
    input_tokens: int = 0
    why: str = ""

    @property
    def choice(self) -> str:
        return max(self.p, key=lambda k: self.p[k]) if self.p else ""


class DecisionBackend:
    name = "base"

    async def decide(self, state: str, q: Question) -> Decision:   # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}


class JevBackend(DecisionBackend):
    name = "jev"

    def __init__(self, provider: str = "openrouter", model: str | None = None, base_url: str | None = None,
                 key_env: str | None = None, timeout_s: float = 8.0, retries: int = 2) -> None:
        pre = PROVIDERS.get(provider, PROVIDERS["openrouter"])
        self.provider = provider
        self.model = model or pre["model"]
        self.base_url = (base_url or pre["base_url"]).rstrip("/")
        self.key_env = key_env if key_env is not None else pre["key_env"]
        self.timeout_s = timeout_s
        self.retries = retries
        self._session: ClientSession | None = None

    def _key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else ""

    async def _client(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(timeout=ClientTimeout(total=self.timeout_s), trust_env=True)
        return self._session

    async def decide(self, state: str, q: Question) -> Decision:
        body = {"state": state, "model": self.model, "questions": {q.id: q.wire()}}
        headers = {"Content-Type": "application/json"}
        if self._key():
            headers["Authorization"] = f"Bearer {self._key()}"
        delay = 0.4
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.monotonic()
            try:
                cl = await self._client()
                async with cl.post(f"{self.base_url}/v1/systemone", data=json.dumps(body), headers=headers) as r:
                    text = await r.text()
                    if r.status == 200:
                        data = json.loads(text)
                        return self._parse(data, q, time.monotonic() - t0)
                    last = f"HTTP {r.status}: {text[:200]}"
                    if r.status not in RETRY_STATUS:
                        break
            except (asyncio.TimeoutError, OSError, ValueError) as e:
                last = f"{type(e).__name__}: {e}"
            if attempt < self.retries:
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"decision model failed: {last}")

    def _parse(self, data: Mapping[str, Any], q: Question, latency: float) -> Decision:
        a = (data.get("answers") or {}).get(q.id) or {}
        probs = a.get("probabilities") or ({a["choice"]: 1.0} if "choice" in a else {})
        p = {o: float(probs.get(o, 0.0)) for o in q.options}
        s = sum(p.values())
        if s <= 0:
            raise RuntimeError(f"decision model returned no distribution: {str(a)[:200]}")
        p = {k: v / s for k, v in p.items()}
        u = data.get("usage") or {}
        tokens = int(u.get("input_tokens") or 0)
        cost = float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else tokens * JEV_PRICE_PER_M_INPUT / 1e6
        return Decision(p=p, model=str(data.get("model") or self.model), cost_usd=cost, latency_s=latency,
                        input_tokens=tokens)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "provider": self.provider, "model": self.model, "base_url": self.base_url}


# ------------------------------------------------------------------------------------------------ rules (no model)
ERROR_PATTERNS = [
    r"Traceback \(most recent call last\)", r"\b[A-Z][A-Za-z]*Error\b", r"\berror:", r"\bERROR\b", r"\bFAILED\b",
    r"\bfailed\b", r"command not found", r"No such file or directory", r"Permission denied", r"Segmentation fault",
    r"exit(?:ed)? (?:with )?(?:code|status) [1-9]", r"returncode[\"']?\s*[:=]\s*[1-9]", r"\bpanic(?:ked)?\b",
    r"\bfatal\b", r"Syntax ?Error", r"assert(?:ion)? ?(?:error|failed)", r"\b\d+ failed\b",
]
_ERR = re.compile("|".join(ERROR_PATTERNS))


@dataclass
class StepView:
    """What the rules backend looks at: the last action and its result, the action before it, and (for the action
    question) whether the proposed call matched a risky pattern of the action pre-filter."""
    last_action: str = ""
    last_output: str = ""
    prev_action: str = ""
    risky: bool = False


class RulesBackend(DecisionBackend):
    """No model: ``hard`` (routing) and ``yes`` (reflection) when the last tool result shows an error or failing tests,
    or the agent repeats its last action; ``risky`` (action) when the proposed call matched a risky pattern. Other
    questions (tool choice, the finish check) need a decision model: the rules backend refuses them, so those decision
    points fall back to their LLM-only behaviour."""
    name = "rules"
    ANSWERS = ("next_step", "needs_reflection", "action_risk")

    def __init__(self, hard_option: str = "hard", easy_option: str = "routine") -> None:
        self.hard, self.easy = hard_option, easy_option

    def judge(self, v: StepView) -> tuple[float, str]:
        if v.last_action and v.last_action == v.prev_action:
            return 0.9, "repeated_action"
        if _ERR.search(v.last_output[-6000:] if v.last_output else ""):
            return 0.9, "error_signal"
        return 0.1, "no_signal"

    async def decide(self, state: str, q: Question, view: StepView | None = None) -> Decision:
        v = view or StepView()
        if q.id == "action_risk":
            p = 0.9 if v.risky else 0.1
            return Decision(p={"risky": p, "safe": 1.0 - p}, model="rules",
                            why="risky_pattern" if v.risky else "no_pattern")
        if q.id == "needs_reflection":
            p, why = self.judge(v)
            return Decision(p={"yes": p, "no": 1.0 - p}, model="rules", why=why)
        if q.id in _NOT_RULES:
            raise RuntimeError(f"the rules backend cannot answer the {q.id!r} question; use a decision model")
        p_hard, why = self.judge(v)
        return Decision(p={self.hard: p_hard, self.easy: 1.0 - p_hard}, model="rules", why=why)


# questions of the other decision points (points.py): they need a model, not a rule
_NOT_RULES = {"next_tool", "task_done", "already_done", "fast_step_ok"}


class NoneBackend(DecisionBackend):
    name = "none"

    async def decide(self, state: str, q: Question) -> Decision:
        raise RuntimeError("no decision backend configured")


def make_backend(spec: Mapping[str, Any] | None) -> DecisionBackend:
    spec = dict(spec or {"type": "rules"})
    kind = spec.pop("type", "jev")
    if kind == "jev":
        return JevBackend(**spec)
    if kind == "rules":
        return RulesBackend()
    if kind == "none":
        return NoneBackend()
    raise ValueError(f"unknown decision backend {kind!r} (jev | rules | none)")
