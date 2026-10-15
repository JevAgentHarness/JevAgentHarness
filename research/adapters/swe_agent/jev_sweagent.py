"""Per-step routing for SWE-agent (v1.1.0): the decision model picks the strong or the fast LLM for each step.

E25 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09, "E25"), harness B. This file is copied into the task
container by ``harbor_jev_swe_agent.JevSweAgent`` and replaces SWE-agent's ``sweagent`` entry point:

    python /opt/jev-swe/jev_sweagent.py run --agent.model.name=gemini/gemini-3.5-flash ...

With ``JEV_ROUTER_MODE=off`` (the default; the ``baseline`` and ``always_cheap`` arms) it only starts SWE-agent
unchanged (``sweagent.run.run.main``). With ``trigger`` it patches ``sweagent.agent.agents.get_model`` so the agent's
model is a :class:`RoutedModel`: two SWE-agent ``LiteLLMModel`` objects (the configured strong model and
``JEV_ROUTER_CHEAP``, the same config otherwise) and, before every query, one yes/no question to a Jev-wire decision
server (``{JEV_BASE_URL}/v1/systemone``, the host's ``jevharness.decision.server``). p(hard) >= ``JEV_ROUTER_TAU`` or no
answer (fail-open) -> strong, else fast. ``shadow`` asks and logs but always uses the strong model.

The question and the state rendering are E4's (``jevharness.adapters.mini_swe``: ``Q_ROUTING``, task + the last 8
actions with their outputs + the current observation, the same truncations); ``tests/test_explore_e25.py`` checks that
they stay identical. The cost and call caps of the strong model's config (``per_instance_cost_limit``,
``per_instance_call_limit``, ``total_cost_limit``) apply to the two models together; the inner models run uncapped.

Logs (JSONL, one object per line, never a key):
  ``JEV_LOG`` (default /logs/agent/jev-decisions.jsonl)  one record per routing decision, pi's schema
                                                         (dp, mode, question, s1_value, s1_conf, s1_dist, final_value,
                                                         tau, latency_ms, input_tokens, model, request_id, ...)
  ``JEV_CALLS_LOG`` (default /logs/agent/jev-calls.jsonl) one record per LLM call: role, model, cost, tokens, seconds

Standard library + SWE-agent only (it runs in SWE-agent's own venv in the task container).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

# E4's routing question, verbatim (jevharness.adapters.mini_swe.Q_ROUTING)
Q_ROUTING = ("Does the next step require careful multi-step reasoning, planning, debugging, or recovery from an "
             "error (i.e., a strong model) rather than a routine action such as reading a file, searching, or "
             "re-running a command?")
QID = "hard"
# E4's state rendering defaults (jevharness.adapters.mini_swe.JevMiniSweConfig)
HISTORY_WINDOW = 8
OBS_TRUNCATE_CHARS = 2000
HIST_OBS_CHARS = 400
TASK_MAX_CHARS = 4000
STATE_MAX_CHARS = 12000
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def env(name: str, default: str | None = None) -> str | None:
    """``JEV_ROUTER_<NAME>``, then ``JEV_<NAME>``, then the default (pi's scope convention)."""
    for key in (f"JEV_ROUTER_{name}", f"JEV_{name}"):
        v = os.environ.get(key)
        if v not in (None, ""):
            return v
    return default


def trunc(s: str | None, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + "\n...[truncated]...\n" + s[-n // 2:]


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _action_of(item: dict[str, Any]) -> str:
    action = item.get("action")
    if action:
        return str(action)
    calls = item.get("tool_calls") or []
    parts = []
    for c in calls:
        fn = (c or {}).get("function") or {}
        parts.append(f"{fn.get('name', '?')} {fn.get('arguments', '')}".strip())
    return "; ".join(parts) or content_text(item.get("content"))


def render_state(history: list[dict[str, Any]], *, history_window: int = HISTORY_WINDOW,
                 obs_chars: int = OBS_TRUNCATE_CHARS, hist_obs_chars: int = HIST_OBS_CHARS,
                 task_chars: int = TASK_MAX_CHARS, max_chars: int = STATE_MAX_CHARS) -> tuple[str, int]:
    """E4's System-1 state from SWE-agent's history: (state, step = actions so far).

    The task is the first observation (SWE-agent's instance template with the problem statement); each later
    ``action`` item is paired with the observation that follows it.
    """
    items = [h for h in history if not h.get("is_demo") and h.get("message_type") != "demonstration"]
    task = ""
    start = 0
    for i, h in enumerate(items):
        if h.get("role") in ("user", "tool") and h.get("message_type", "observation") == "observation":
            task = content_text(h.get("content"))
            start = i + 1
            break
    steps: list[dict[str, str]] = []
    for h in items[start:]:
        if h.get("role") == "assistant" and h.get("message_type", "action") == "action":
            steps.append({"command": _action_of(h), "output": ""})
        elif h.get("role") in ("user", "tool") and h.get("message_type", "observation") == "observation" and steps:
            steps[-1]["output"] = content_text(h.get("content"))
    parts = [f"TASK (issue to resolve):\n{trunc(task, task_chars)}"]
    recent = steps[-history_window:]
    if recent:
        first = len(steps) - len(recent) + 1
        lines = [f"[{first + k}] $ {trunc(e['command'], 300)} -> {trunc(e['output'], hist_obs_chars)}"
                 for k, e in enumerate(recent)]
        parts.append("HISTORY (most recent last):\n" + "\n".join(lines))
    last = steps[-1]["output"] if steps else "(no command executed yet)"
    parts.append(f"CURRENT OBSERVATION:\n{trunc(last, obs_chars)}")
    parts.append(f"STEP: {len(steps)}")
    return trunc("\n\n".join(parts), max_chars), len(steps)


class JevWireClient:
    """POST {base}/v1/systemone with one noul question; returns (p_yes, usage, latency_ms, served model)."""

    def __init__(self, base_url: str, wire_model: str, *, api_key: str | None = None, timeout_s: float = 10.0,
                 max_retries: int = 2, opener: Callable[..., Any] | None = None):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.wire_model = wire_model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._open = opener or urllib.request.urlopen

    def ask(self, state: str, question: str = Q_ROUTING) -> tuple[float, dict[str, Any], float, str | None]:
        body = json.dumps({"state": state, "model": self.wire_model,
                           "questions": {QID: {"type": "noul", "instructions": question}}}).encode()
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with self._open(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode())
                ans = (data.get("answers") or {}).get(QID) or {}
                p = ans.get("noul")
                if not isinstance(p, (int, float)):
                    raise ValueError(f"no noul answer in {str(data)[:200]}")
                return float(p), dict(data.get("usage") or {}), (time.perf_counter() - t0) * 1000, data.get("model")
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in RETRY_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, json.JSONDecodeError) as e:
                last = e
            if attempt < self.max_retries:
                time.sleep(0.5 * 2 ** attempt)
        raise RuntimeError(f"decision model failed: {type(last).__name__}: {str(last)[:200]}")


class JsonlLog:
    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()

    def write(self, rec: dict[str, Any]) -> None:
        if not self.path:
            return
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **rec}
        with self._lock:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _limit_errors() -> tuple[type[Exception], type[Exception], type[Exception]]:
    try:
        from sweagent.exceptions import (InstanceCallLimitExceededError, InstanceCostLimitExceededError,
                                         TotalCostLimitExceededError)
        return InstanceCostLimitExceededError, InstanceCallLimitExceededError, TotalCostLimitExceededError
    except ImportError:  # offline tests
        class _CostLimit(RuntimeError):
            pass

        return (type("InstanceCostLimitExceededError", (_CostLimit,), {}),
                type("InstanceCallLimitExceededError", (_CostLimit,), {}),
                type("TotalCostLimitExceededError", (_CostLimit,), {}))


def _model_name(model: Any) -> str:
    return str(getattr(getattr(model, "config", None), "name", type(model).__name__))


class RoutedModel:
    """A SWE-agent model (duck-typed ``AbstractModel``) that sends each query to ``strong`` or ``cheap``.

    ``config`` is the strong model's original config: its caps apply to ``stats``, the sum of both models' usage.
    ``stats_factory`` builds an empty stats object (SWE-agent's ``InstanceStats``); ``global_cost`` returns SWE-agent's
    running total (``GLOBAL_STATS.total_cost``) for ``total_cost_limit``.
    """

    def __init__(self, config: Any, tools: Any, strong: Any, cheap: Any, client: JevWireClient | None, *,
                 tau: float = 0.5, mode: str = "trigger", stats_factory: Callable[[], Any],
                 global_cost: Callable[[], float] = lambda: 0.0, decisions: JsonlLog | None = None,
                 calls: JsonlLog | None = None, session: str | None = None):
        self.config = config
        self.tools = tools
        self.strong = strong
        self.cheap = cheap
        self.client = client
        self.tau = float(tau)
        self.mode = mode
        self._stats_factory = stats_factory
        self._global_cost = global_cost
        self.stats = stats_factory()
        self.decisions = decisions or JsonlLog(None)
        self.calls = calls or JsonlLog(None)
        self.session = session or uuid.uuid4().hex
        self.counters = {"strong": 0, "cheap": 0, "errors": 0}
        self._cost_err, self._call_err, self._total_err = _limit_errors()

    # SWE-agent reads these
    @property
    def instance_cost_limit(self) -> float:
        return float(getattr(self.config, "per_instance_cost_limit", 0.0) or 0.0)

    def reset_stats(self) -> None:
        self.stats = self._stats_factory()
        for m in (self.strong, self.cheap):
            if hasattr(m, "reset_stats"):
                m.reset_stats()

    def decide(self, history: list[dict[str, Any]]) -> str:
        state, step = render_state(history)
        rid = uuid.uuid4().hex[:12]
        rec: dict[str, Any] = {"harness": "swe-agent", "session": self.session, "extension": "jev-sweagent-router",
                               "dp": "routing", "mode": self.mode, "question": Q_ROUTING, "tau": self.tau,
                               "request_id": rid, "n_questions": 1, "step": step}
        if self.client is None:
            role = "strong"
            rec.update(model=None, error="no decision client", final_value=role, s1_tier=None)
        else:
            try:
                p, usage, ms, served = self.client.ask(state)
                s1_tier = "strong" if p >= self.tau else "cheap"
                role = s1_tier if self.mode == "trigger" else "strong"
                rec.update(s1_value=p >= 0.5, s1_conf=max(p, 1 - p), s1_dist={"yes": p, "no": 1 - p},
                           final_value=role, s1_tier=s1_tier, latency_ms=round(ms, 1),
                           input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                           s1_cost_usd=usage.get("cost"), model=served or self.client.wire_model)
            except Exception as e:  # fail open: the strong model, as the unmodified agent
                role = "strong"
                self.counters["errors"] += 1
                rec.update(model=self.client.wire_model, error=f"{type(e).__name__}: {e}"[:300], final_value=role,
                           s1_tier=None)
        rec["routed_model"] = _model_name(self.strong if role == "strong" else self.cheap)
        self.decisions.write(rec)
        return role

    def _check_limits(self) -> None:
        c = self.config
        tot = float(getattr(c, "total_cost_limit", 0.0) or 0.0)
        if 0 < tot < self._global_cost():
            raise self._total_err("Total cost limit exceeded")
        if 0 < self.instance_cost_limit < float(self.stats.instance_cost):
            raise self._cost_err("Instance cost limit exceeded")
        calls = int(getattr(c, "per_instance_call_limit", 0) or 0)
        if 0 < calls < int(self.stats.api_calls):
            raise self._call_err("Per instance call limit exceeded")

    def query(self, history: list[dict[str, Any]], action_prompt: str = "> ", **kwargs: Any) -> Any:
        role = self.decide(history)
        model = self.cheap if role == "cheap" else self.strong
        self.counters[role] += 1
        before = model.stats.model_copy() if hasattr(model.stats, "model_copy") else type(model.stats)(**vars(model.stats))
        t0 = time.perf_counter()
        err: str | None = None
        try:
            return model.query(history, **kwargs)
        except Exception as e:
            err = type(e).__name__
            raise
        finally:
            delta = model.stats - before
            self.stats = self.stats + delta
            self.calls.write({"session": self.session, "role": role, "model": _model_name(model),
                              "cost_usd": float(delta.instance_cost), "input_tokens": int(delta.tokens_sent),
                              "output_tokens": int(delta.tokens_received), "api_calls": int(delta.api_calls),
                              "seconds": round(time.perf_counter() - t0, 3), "error": err})
            if err is None:
                self._check_limits()


def build_routed_model(args: Any, tools: Any, orig_get_model: Callable[[Any, Any], Any]) -> Any:
    """SWE-agent ``get_model`` replacement body: a RoutedModel around two uncapped LiteLLMModels."""
    from sweagent.agent import models as M

    base = orig_get_model(args, tools)
    if not isinstance(base, M.LiteLLMModel):
        return base
    mode = env("MODE", "off") or "off"
    cheap_name = env("CHEAP")
    if not cheap_name:
        raise ValueError("JEV_ROUTER_CHEAP (the fast model's litellm name) is required for routing")
    uncapped = {"per_instance_cost_limit": 0.0, "total_cost_limit": 0.0, "per_instance_call_limit": 0}
    strong = M.LiteLLMModel(args.model_copy(deep=True, update=uncapped), tools)
    cheap = M.LiteLLMModel(args.model_copy(deep=True, update={**uncapped, "name": cheap_name}), tools)
    base_url = env("BASE_URL")
    client = None
    if base_url:
        client = JevWireClient(base_url, env("WIRE_MODEL", "jev-1.13") or "jev-1.13",
                               api_key=os.environ.get("JEV_API_KEY") or None,
                               timeout_s=float(env("TIMEOUT_MS", "10000") or 10000) / 1000,
                               max_retries=int(env("MAX_RETRIES", "2") or 2))
    return RoutedModel(args, tools, strong, cheap, client, tau=float(env("TAU", "0.5") or 0.5), mode=mode,
                       stats_factory=M.InstanceStats, global_cost=lambda: float(M.GLOBAL_STATS.total_cost),
                       decisions=JsonlLog(env("LOG", "/logs/agent/jev-decisions.jsonl")),
                       calls=JsonlLog(os.environ.get("JEV_CALLS_LOG", "/logs/agent/jev-calls.jsonl")))


def install() -> bool:
    """Patch SWE-agent's model factory when routing is on; returns whether it did."""
    if (env("MODE", "off") or "off") not in ("trigger", "shadow"):
        return False
    import sweagent.agent.agents as A

    orig = A.get_model

    def get_model(args: Any, tools: Any) -> Any:
        return build_routed_model(args, tools, orig)

    A.get_model = get_model
    return True


def proxy_args(args: list[str], api_base: str | None = None) -> list[str]:
    """E29: with ``E29_API_BASE`` (the host-side routing proxy's URL for this trial, set by ``JevSweAgent``'s
    ``llm_proxy`` option) the model's ``api_base`` points at the proxy, which holds the key; litellm's ``gemini/``
    provider then posts to ``{api_base}/models/{model}:generateContent``. Only ``run`` gets the flag."""
    api_base = api_base if api_base is not None else os.environ.get("E29_API_BASE", "")
    if not api_base or not args or args[0] != "run" or any(a.startswith("--agent.model.api_base") for a in args):
        return list(args)
    return [*args, f"--agent.model.api_base={api_base}"]


def main(argv: list[str] | None = None) -> None:
    install()
    from sweagent.run.run import main as run_main

    run_main(proxy_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
