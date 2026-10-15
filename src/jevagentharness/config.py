"""Configuration: upstreams, the two tiers, the router, the decision points and the model names clients may ask for."""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import points as P

DEFAULT_CONFIG = """\
# JevAgentHarness configuration. Keys are read from the environment (never write them here).
listen: {host: 127.0.0.1, port: 8787}
log_dir: ~/.jah/logs
budget_usd: null              # refuse new sessions once the logged spend reaches this (null = no cap)
log_states: false             # true: also log each decision's input state (states.jsonl; contains task text)

upstreams:
  gemini: {type: gemini, base_url: "https://generativelanguage.googleapis.com/v1beta", key_env: GEMINI_API_KEY}
  # openrouter: {type: openai, base_url: "https://openrouter.ai/api/v1", key_env: OPENROUTER_API_KEY,
  #              effort_field: reasoning}
  # vllm: {type: openai, base_url: "http://localhost:8000/v1", key_env: ""}

tiers:                        # System 2 (strong) and System 1-speed LLM (fast); the setting of the README results
  strong: {upstream: gemini, model: gemini-4-argon, effort: high}
  fast:   {upstream: gemini, model: gemini-3.5-flash-lite, effort: low}
  # fast: {upstream: gemini, model: gemini-3.8-flash, effort: low, cache: explicit}   # stronger, but only ~2.7x cheaper

router:
  decision: {type: jev, provider: openrouter}     # or {type: rules} (no key needed), {type: jev, provider: local}
  tau: 0.75                   # escalate to the strong tier when p(hard) >= tau
  first_steps_strong: 1       # the opening step(s) of a session plan the work on the strong tier
  stick_strong: 0             # after an escalation, keep this many further steps on the strong tier
  max_fast_streak: 0          # >0: force a strong step after this many fast steps in a row
  verify_finish: true         # a fast-tier final answer (no tool call) is re-done by the strong tier
  on_error: strong            # tier when the decision model fails or times out
  decision_timeout_s: 8

# Decision points x modes (docs/modes.md). A point set under `decisions` wins over the preset. Unset points keep the
# behaviour above: routing in cascade mode at router.tau, and the finish check (stop: strong) while verify_finish is
# true. `jah fit <log_dir>` fits the thresholds (tau) from logged runs.
preset: null                  # route (this default) | paper | safe | shadow
decisions:
  # routing:     {mode: cascade}   # off | shadow | cascade | guard
  # tool_select: {mode: off}       # off | shadow | assist | cascade | guard (cascade: fit tau with `jah fit` first)
  # stop:        {mode: strong}    # off | shadow | strong | guard | trigger
  # reflection:  {mode: off}       # off | shadow | trigger | always
  # action:      {mode: off}       # off | shadow | guard (a second look at risky shell and write calls)

models:                       # the model names a client can use (any other name -> `default`)
  jah-auto: {policy: route}
  jah-strong: {tier: strong}
  jah-fast: {tier: fast}
default_model: jah-auto
"""


@dataclass
class Upstream:
    name: str
    type: str                     # "gemini" | "openai" (Chat Completions) | "openai_responses" (Responses API)
    base_url: str
    key_env: str = ""
    effort_field: str = "reasoning_effort"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 900.0
    retries: int = 3

    def keys(self) -> list[str]:
        """The variable may hold several comma-separated keys (a key pool, e.g. to spread rate limits)."""
        raw = os.environ.get(self.key_env, "") if self.key_env else ""
        return [k.strip() for k in raw.split(",") if k.strip()]

    def key_index(self, session: str | None = None) -> int:
        """Each session sticks to one key of the pool (Gemini context caches belong to the key's project)."""
        n = len(self.keys())
        return zlib.crc32((session or "").encode()) % n if n > 1 else 0

    def key(self, session: str | None = None) -> str:
        ks = self.keys()
        return ks[self.key_index(session)] if ks else ""


@dataclass
class Tier:
    name: str
    upstream: str
    model: str
    effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # merged into the upstream body
    cache: str = "implicit"       # "explicit": the proxy keeps a Gemini context cache per session (gemini_cache.py)


@dataclass
class RouterCfg:
    decision: dict[str, Any] = field(default_factory=lambda: {"type": "rules"})
    tau: float = 0.5
    first_steps_strong: int = 1
    stick_strong: int = 0
    max_fast_streak: int = 0
    verify_finish: bool = True
    on_error: str = "strong"
    decision_timeout_s: float = 8.0
    strong: str = "strong"
    fast: str = "fast"


@dataclass
class Config:
    upstreams: dict[str, Upstream]
    tiers: dict[str, Tier]
    router: RouterCfg
    models: dict[str, dict[str, Any]]
    default_model: str = "jah-auto"
    host: str = "127.0.0.1"
    port: int = 8787
    log_dir: str = "~/.jah/logs"
    budget_usd: float | None = None
    prices: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_states: bool = False      # also write each decision's input state to states.jsonl (holds task text)
    preset: str | None = None     # a decision-point preset (points.PRESETS)
    decisions: dict[str, Any] = field(default_factory=dict)   # the `decisions:` block (points.py)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Config":
        ups = {n: Upstream(name=n, **u) for n, u in (d.get("upstreams") or {}).items()}
        for n, u in ups.items():
            if u.type not in ("gemini", "openai", "openai_responses"):
                raise ValueError(f"upstream {n}: type must be gemini, openai or openai_responses, got {u.type!r}")
        tiers = {n: Tier(name=n, **t) for n, t in (d.get("tiers") or {}).items()}
        for n, t in tiers.items():
            if t.upstream not in ups:
                raise ValueError(f"tier {n}: unknown upstream {t.upstream!r}")
        router = RouterCfg(**(d.get("router") or {}))
        for role in (router.strong, router.fast, router.on_error):
            if role not in tiers:
                raise ValueError(f"router refers to tier {role!r}, which is not defined")
        models = {str(k): dict(v or {}) for k, v in (d.get("models") or {"jah-auto": {"policy": "route"}}).items()}
        for name, m in models.items():
            if "tier" in m and m["tier"] not in tiers:
                raise ValueError(f"model {name}: unknown tier {m['tier']!r}")
            if "tier" not in m and m.get("policy", "route") != "route":
                raise ValueError(f"model {name}: policy must be 'route' (or give a tier)")
            bad = set(m) - {"tier", "policy", "tau", "first_steps_strong", "stick_strong", "max_fast_streak",
                            "verify_finish", "on_error", "strong", "fast", "preset", "decisions"}
            if bad:
                raise ValueError(f"model {name}: unknown keys {sorted(bad)}")
            for role in ("strong", "fast", "on_error"):
                if role in m and m[role] not in tiers:
                    raise ValueError(f"model {name}: {role} refers to unknown tier {m[role]!r}")
            m["preset"] = P.check_preset(m.get("preset"), f"model {name}: preset")
            if m["preset"] is None:
                m.pop("preset")
            if "decisions" in m:
                m["decisions"] = P.validate(m["decisions"], f"model {name}: decisions")
                if "tier" in m and "routing" in m["decisions"]:
                    raise ValueError(f"model {name}: a pinned tier has no routing decision point")
                _check_tier(m["decisions"], tiers, f"model {name}: decisions")
        decisions = P.validate(d.get("decisions"))
        _check_tier(decisions, tiers, "decisions")
        listen = d.get("listen") or {}
        default_model = str(d.get("default_model", "jah-auto"))
        if default_model not in models:
            raise ValueError(f"default_model {default_model!r} is not in models")
        return cls(upstreams=ups, tiers=tiers, router=router, models=models, default_model=default_model,
                   host=str(listen.get("host", "127.0.0.1")), port=int(listen.get("port", 8787)),
                   log_dir=str(d.get("log_dir", "~/.jah/logs")),
                   budget_usd=(None if d.get("budget_usd") is None else float(d["budget_usd"])),
                   prices=dict(d.get("prices") or {}), log_states=bool(d.get("log_states", False)),
                   preset=P.check_preset(d.get("preset")), decisions=decisions)

    def model_spec(self, name: str) -> tuple[str, dict[str, Any]]:
        """The client's model name -> (resolved name, spec). Unknown names use ``default_model``."""
        if name in self.models:
            return name, self.models[name]
        if name.startswith("jah-") and name[4:] in self.tiers:
            return name, {"tier": name[4:]}
        return self.default_model, self.models[self.default_model]

    def decisions_for(self, spec: Mapping[str, Any], router_tau: float | None = None) -> P.Decisions:
        """The decision points of a model name (its spec from :meth:`model_spec`); see :func:`points.resolve`."""
        return P.resolve(self.router.tau if router_tau is None else router_tau, self.router.verify_finish,
                         self.preset, self.decisions, spec, pinned="tier" in spec)


def _check_tier(decisions: Mapping[str, Any], tiers: Mapping[str, Tier], where: str) -> None:
    t = (decisions.get("tool_select") or {}).get("tier")
    if t is not None and t not in ("strong", "fast") and t not in tiers:
        raise ValueError(f"{where}.tool_select.tier: unknown tier {t!r}")


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    d = yaml.safe_load(Path(path).expanduser().read_text()) if path else yaml.safe_load(DEFAULT_CONFIG)
    for s in overrides or []:
        k, _, v = s.partition("=")
        cur: Any = d
        parts = k.split(".")
        for p in parts[:-1]:
            if cur.get(p) is None:                 # a missing key, or a block holding only comments
                cur[p] = {}
            cur = cur[p]
            if not isinstance(cur, dict):
                raise ValueError(f"--set {k}: {p!r} is not a mapping in the config")
        cur[parts[-1]] = yaml.safe_load(v)
    return Config.from_dict(d)
