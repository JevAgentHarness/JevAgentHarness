"""tau2-bench (Sierra's tau^2-bench) integration: a tau2 agent with System-1 decision points.

tau2 drives its own simulation loop (agent <-> LLM user simulator <-> domain DB/tools, graded by
DB-state / communicate / NL-assertion checks), so instead of wrapping tau2 as a jevharness ``Env``
we plug *into* tau2 as a registered agent (``jev_agent``) and convert tau2's results afterwards::

    tau2 orchestrator -> JevTau2Agent._generate_next_message(msg, state)   (one call = one StepLog)
        routing      trigger                  S1 Noul: strong LLM (--agent-llm) or cheap LLM (small_llm)?
        tool_select  replace|cascade|assist   S1 Choice over tau2's tools + ``respond_to_user``
          tool_params  replace|cascade        S1 fills bounded/copyable *required* args; all filled -> the
                                              tool call is emitted with NO LLM call, else the LLM is forced
                                              (tool_choice) to call that tool with only its schema
        tool_select  guard                    LLM proposes, S1 Noul "right next step?"  no -> regenerate
        policy_guard guard   (tau2-specific)  S1 Noul on state-changing (WRITE) tool calls: "complies with
                                              the policy and explicitly confirmed by the user?"
                                              no -> regenerate
        stop         guard                    S1 Noul on ``transfer_to_human_agents``      no -> regenerate
        tool_select / tool_params / stop / routing / policy_guard  shadow   (measurement; PREREGISTRATION A5.4)
                                              the SAME question the live arm asks, answered by S1 and only logged
                                              (``DecisionLog.shadow``), while the LLM's own action is executed
                                              unchanged; S1 calls are tagged ``meta.shadow`` in the Ledger and
                                              kept out of the episode cost (``usage.shadow``)
    agent.stop()  -> one side-car JSONL record per simulation (StepLogs/DecisionLogs + S1/S2 Ledger)
    convert_results(results.json [+ side-car]) -> Trajectory JSONL -> ``jevbench report``

With every decision point ``off`` the agent is behaviourally identical to tau2's ``llm_agent``
(same system prompt, same ``tau2.utils.llm_utils.generate`` call) but logs in our format.

Configuration: a YAML file named by ``$JEV_HARNESS_CONFIG`` (read when tau2 builds each agent), holding
the ``HarnessConfig`` fields flat plus the tau2 keys of :class:`Tau2JevConfig` (``s1``, ``s1_cache``,
``small_llm``, ``policy_guard``, ...); ``$JEV_HARNESS_OVERRIDES`` holds a JSON object of dotted-key
overrides, e.g. ``{"tool_select.tau": 0.9}``. Launch runs through this module so the agent is
registered before tau2's CLI parses ``--agent``::

    python -m jevharness.bench.suites.tau2 run --jev-config configs/tau2/jev_guard.yaml \
        --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1-mini --num-trials 3 --save-to /abs/out
    python -m jevharness.bench.suites.tau2 rerun-errored /abs/out -- <the same `run` arguments>   # A6.22 (e) 4
    python -m jevharness.bench.suites.tau2 convert /abs/out/results.json \
        --out runs/tau2/airline/jev_guard.jsonl
    python -m jevharness.bench.suites.tau2 split airline            # A5.4 shadow / test task ids
    python -m jevharness.bench.suites.tau2 report runs/tau2/airline --split test   # C7 comparisons on the 35

Gold labels (for decision-level metrics and the ``simulated``/``oracle`` S1 backends only; never shown
to the agent) are *approximations* derived from the task's reference actions -- see :class:`_Reference`.

PREREGISTRATION A5.4 (c): the 50 airline tasks are split once (:func:`airline_split`, seed 0) into a 15-task
shadow split (the ``jev_shadow`` arm's tau-hat* refit input) and the 35-task airline test set; every arm still
runs all 50 tasks, ``convert`` annotates ``meta.tau2.split`` and ``report --split test|shadow`` restricts the
comparison. ``python -m jevharness.bench.suites.tau2 split airline`` prints the split.
See docs/TAU2.md for setup, commands and cost estimates.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from pydantic import BaseModel, Field

from ...accounting import CallRecord, Ledger, default_prices
from ...decision import make_backend
from ...decision.base import DecisionBackend
from ...decision.types import MAX_OPTIONS, Answer, Choice, Noul, Question
from ...harness.agent import JevAgent, usage_summary
from ...harness.config import DP, DPConfig, HarnessConfig, Mode
from ...harness.tap import Tap, is_shadow
from ...harness.tools import OTHER, ParamInfo, coerce, extract_candidates, params_of, tool_desc
from ...harness.trajectory import DecisionLog, StepLog, Trajectory, load_trajectories

try:  # tau2 is optional: the converter below works on plain JSON without it
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, UserMessage
    from tau2.environment.toolkit import TOOL_TYPE_ATTR
    from tau2.utils import llm_utils as _tau2_llm

    HAVE_TAU2 = True
    _TAU2_IMPORT_ERROR: Exception | None = None
except ImportError as _e:  # pragma: no cover - exercised only without tau2
    HAVE_TAU2 = False
    _TAU2_IMPORT_ERROR = _e
    LLMAgent = object  # type: ignore[assignment,misc]
    TOOL_TYPE_ATTR = "__tool_type__"

AGENT_NAME = "jev_agent"
ENV_CONFIG = "JEV_HARNESS_CONFIG"
ENV_OVERRIDES = "JEV_HARNESS_OVERRIDES"
ENV_LOG_DIR = "JEV_TAU2_LOG_DIR"
ENV_NL_JUDGE = "JEV_TAU2_NL_JUDGE_LLM"            # litellm model for tau2's NL-assertion judge (retail)
ENV_NL_JUDGE_ARGS = "JEV_TAU2_NL_JUDGE_LLM_ARGS"  # JSON litellm args for it
SIDECAR_FILE = "jev_decisions.jsonl"
ATTEMPTS_FILE = "jev_attempts.json"   # re-run ledger of errored simulations (PREREGISTRATION A6.22 (e) 4)
ERRORED_DIR = "jev_errored"           # the pruned errored simulations, kept for the drop listing (A6.22 (e) 5)
MAX_SIM_ATTEMPTS = 3                  # the first execution + up to two re-runs (A6.22 (e) 4, A6.20 (f), §1)
RESPOND = "respond_to_user"
TRANSFER_TOOL = "transfer_to_human_agents"
RESPOND_DESC = ("Send a text message to the user instead of calling a tool (ask for missing details or "
                "explicit confirmation, report a result, or refuse).")
TOOL_SELECT_Q = ("What should the customer-service agent do next? Choose the tool to call, or "
                 f"`{RESPOND}` to send a message to the user.")
ROUTING_Q = ("Does the next step require careful multi-step reasoning, policy interpretation or recovery "
             "from an error (i.e., a strong model) rather than a routine reply or lookup?")
STOP_Q = ("The agent wants to transfer the user to a human agent (`transfer_to_human_agents`). Per the policy, "
          "transfer only if the request cannot be handled within the scope of the agent's tools and policy (or "
          "the user asks for a human where that is allowed). Is transferring now justified?")
_AFFIRM = re.compile(r"\b(yes|yeah|yep|yup|sure|confirm(ed)?|go ahead|proceed|please do|correct|ok(ay)?)\b",
                     re.I)
_CONFIRM_POLICY = re.compile(r"explicit\s+(user\s+)?confirmation", re.I)


def _require_tau2() -> None:
    if not HAVE_TAU2:
        raise ImportError("tau2-bench is not importable; see docs/TAU2.md for installation") \
            from _TAU2_IMPORT_ERROR


def _trunc(s: str | None, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + " ...[truncated]... " + s[-n // 2:]


# ================================================================== configuration
_TAU2_KEYS = {"s1", "s1_cache", "small_llm", "small_llm_args", "policy_guard", "log_dir", "log_state",
              "s1_policy_chars", "respond_tool_choice", "respond_fallback", "oracle_gold", "log_params", "s1_act"}


class Tau2JevConfig(BaseModel):
    """One tau2 arm: a :class:`HarnessConfig` plus the tau2-specific System-1 settings."""

    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    s1: dict[str, Any] | str | None = None          # make_backend spec, e.g. {type: jev, model: jev-1.13.0}
    s1_cache: str | None = None                     # SQLite decision cache (CachedBackend)
    small_llm: str | None = None                    # litellm model name of the cheap LLM (routing=trigger)
    small_llm_args: dict[str, Any] | None = None    # default: the agent's --agent-llm-args
    policy_guard: DPConfig = Field(default_factory=DPConfig)  # mode off|guard; extra.scope: write|all
    log_dir: str | None = None                      # side-car dir (default: $JEV_TAU2_LOG_DIR)
    log_state: bool = True                          # keep S1 state strings in DecisionLogs (training harvest)
    s1_policy_chars: int = 3000                     # policy prefix rendered into the S1 state
    respond_tool_choice: str = "none"               # S1 picked respond_to_user: "none" | "no_tools"
    # S1 picked respond_to_user but the LLM returned a tool call and no text: "execute" runs that call as the LLM's
    # override (the registered behaviour; the call was made without the tool declarations, so its name can be one
    # that does not exist); "defer" re-asks the LLM with its tools, exactly as a cascade deferral (exploratory E1,
    # docs/PREREG_A7_STAGE2.md Deviations 2026-10-05)
    respond_fallback: str = "execute"
    oracle_gold: bool = True                        # derive approximate gold labels from reference actions
    # exploratory row E22 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E22"): with tool_params off, log at every
    # executed LLM tool call the argument Choices the cascade would ask for its required arguments (options exactly as
    # `_param_options` builds them, the LLM's value as `s2_value`, mode "off", no state string: the step's tool_select
    # record holds it). No System-1 call and no change of behaviour: a harvest of the agent's own argument decisions.
    log_params: bool = False
    # exploratory row E22: when System-1's tool choice is accepted (replace, or cascade at conf >= tau), "all" acts on it
    # as registered (a respond_to_user answer -> the LLM writes under tool_choice="none"; a tool with arguments left
    # open -> one forced single-tool LLM call); "complete" acts only on a tool whose every required argument System-1
    # decides (the call is emitted with no LLM call). Every other turn -- respond_to_user, an argument left open -- is the
    # LLM-only turn (full tool list, tool_choice auto: the request llm_only sends, so Gemini's implicit prompt cache is
    # kept), logged as a deferral (`gold_aux.not_acted`).
    s1_act: str = "all"

    @property
    def name(self) -> str:
        return self.harness.name

    def uses_s1(self) -> bool:
        return self.harness.uses_s1() or self.policy_guard.mode in (Mode.GUARD, Mode.SHADOW)

    def shadow_points(self) -> list[str]:
        """Decision points in ``shadow`` mode (S1 asked and logged, never acted on), ``policy`` included."""
        return [dp.value for dp in self.harness.shadow_points()] + \
            (["policy"] if self.policy_guard.mode == Mode.SHADOW else [])

    def validate_supported(self) -> "Tau2JevConfig":
        h = self.harness.validate_modes()
        for dp in (DP.MEMORY_WRITE, DP.MEMORY_READ, DP.REFLECTION, DP.ACTION):
            if h.dp(dp).mode != Mode.OFF:
                raise ValueError(f"decision point {dp.value!r} is not implemented for tau2 (must be off)")
        if h.stop.mode not in (Mode.OFF, Mode.GUARD, Mode.SHADOW):
            raise ValueError("tau2 conversations are ended by the user simulator: stop supports off|guard|shadow "
                             "only (guard = verify transfer_to_human_agents; shadow = log that verdict, never block)")
        if self.policy_guard.mode not in (Mode.OFF, Mode.GUARD, Mode.SHADOW):
            raise ValueError("policy_guard supports off|guard|shadow")
        if self.policy_guard.extra.get("scope", "write") not in ("write", "all"):
            raise ValueError("policy_guard.extra.scope must be 'write' or 'all'")
        if h.routing.mode == Mode.TRIGGER and not self.small_llm:
            raise ValueError("routing=trigger needs `small_llm` (a litellm model name)")
        if self.respond_tool_choice not in ("none", "no_tools"):
            raise ValueError("respond_tool_choice must be 'none' or 'no_tools'")
        if self.respond_fallback not in ("execute", "defer"):
            raise ValueError("respond_fallback must be 'execute' or 'defer'")
        if self.s1_act not in ("all", "complete"):
            raise ValueError("s1_act must be 'all' or 'complete'")
        if self.log_params and h.tool_params.mode != Mode.OFF:
            raise ValueError("log_params logs the LLM's own argument decisions: tool_params must be off")
        if self.uses_s1() and self.s1 is None:
            raise ValueError(f"config {self.name!r} uses System-1 decision points but has no `s1` backend "
                             "spec")
        return self

    @classmethod
    def from_dict(cls, raw: dict[str, Any], overrides: dict[str, Any] | None = None) -> "Tau2JevConfig":
        d = copy.deepcopy(raw or {})
        overrides = dict(overrides or {})
        for key, val in overrides.items():
            node = d
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val
        if overrides and "name" not in overrides:  # e.g. jev_ts_cascade -> jev_ts_cascade_tau0.9
            base = d.get("name", "llm_only")
            d["name"] = base + "".join(f"_{k.split('.')[-1]}{v}" for k, v in overrides.items()
                                       if not isinstance(v, (dict, list)))
        tau2_part = {k: d.pop(k) for k in list(d) if k in _TAU2_KEYS}
        return cls(harness=HarnessConfig(**d), **tau2_part).validate_supported()

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "Tau2JevConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {}, overrides)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """["tool_select.tau=0.9", "s1={type: oracle}"] -> {"tool_select.tau": 0.9, "s1": {"type": "oracle"}}"""
    out: dict[str, Any] = {}
    for kv in pairs:
        k, sep, v = kv.partition("=")
        if not sep:
            raise ValueError(f"override {kv!r} is not KEY=VALUE")
        out[k.strip()] = yaml.safe_load(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Tau2JevConfig:
    """Config from ``path`` or ``$JEV_HARNESS_CONFIG`` (+ ``$JEV_HARNESS_OVERRIDES``); llm_only if neither."""
    path = path or os.environ.get(ENV_CONFIG)
    if overrides is None:
        env = os.environ.get(ENV_OVERRIDES)
        overrides = json.loads(env) if env else {}
    if not path:
        return Tau2JevConfig.from_dict({"name": "llm_only"}, overrides)
    return Tau2JevConfig.load(path, overrides)


_BACKENDS: dict[str, DecisionBackend] = {}
_BACKEND_LOCK = threading.Lock()


def _shared_backend(spec: dict[str, Any] | str, cache: str | None) -> DecisionBackend:
    """One backend per (spec, cache) per process: tau2 builds a fresh agent per simulation (and runs
    simulations in threads); HTTP clients and the SQLite cache are shared."""
    key = json.dumps([spec, cache], sort_keys=True, default=str)
    with _BACKEND_LOCK:
        if key not in _BACKENDS:
            _BACKENDS[key] = make_backend(spec, cache=cache)
        return _BACKENDS[key]


# ================================================================== tool-schema helpers
def _flatten_param(p: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local $ref and collapse Optional[X] (anyOf [X, null]) so ParamInfo sees enums/types."""
    q = dict(p)
    if "anyOf" in q:
        branches = [b for b in q["anyOf"] if b.get("type") != "null"]
        if len(branches) == 1:
            q.pop("anyOf")
            q = {**branches[0], **q}
    ref = q.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        q.pop("$ref")
        q = {**defs.get(ref.split("/")[-1], {}), **q}
    return q


def _s1_schema(openai_schema: dict[str, Any]) -> dict[str, Any]:
    s = copy.deepcopy(openai_schema)
    params = s["function"].get("parameters") or {}
    defs = params.get("$defs") or {}
    props = params.get("properties") or {}
    params["properties"] = {k: _flatten_param(v, defs) for k, v in props.items()}
    return s


_JSON_KEY = re.compile(r'"([^"\n]{1,80})"\s*:')


# snake/kebab-case identifiers with a digit (raj_sanchez_7340, gift_card_3226531): extract_candidates' ID
# pattern needs the digit right after the first separator, so in free text ("my user ID is
# raj_sanchez_7340") these were missed
_IDENT = re.compile(r"(?<![\w-])([A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+)(?![\w-])")
_DATE_SPAN = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|(?<![\w:])\d{1,2}:\d{2}(?::\d{2})?(?![\w:])")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# free-text arguments (transfer summaries, notes, ...): copying a span from the conversation is not an answer
_FREE_TEXT_PARAM = re.compile(r"^(summary|thought|message|content|body|text|notes?|comment|explanation)$", re.I)


def _field_values(text: str, key: str) -> list[str]:
    """Values recorded under the JSON key ``key`` anywhere in the conversation (tool outputs and earlier
    tool-call arguments), plus the string items of a list under its plural (``reservation_id`` ->
    ``"reservations": [...]``); most recent first."""
    found: list[tuple[int, str]] = []
    for m in re.finditer(rf'"{re.escape(key)}"\s*:\s*(?:"([^"\n]{{1,80}})"|(-?\d+(?:\.\d+)?))', text):
        found.append((m.start(), m.group(1) if m.group(1) is not None else m.group(2)))
    plural = (key[:-3] if key.endswith("_id") else key) + "s"
    for m in re.finditer(rf'"{re.escape(plural)}"\s*:\s*\[([^\]]{{0,4000}})\]', text):
        found += [(m.start() + v.start(), v.group(1)) for v in re.finditer(r'"([^"\n]{1,80})"', m.group(1))]
    out: list[str] = []
    for _, v in sorted(found, key=lambda x: -x[0]):
        if v not in out:
            out.append(v)
    return out


def _value_candidates(text: str, limit: int = MAX_OPTIONS - 1, param: str | None = None,
                      param_type: str | None = None) -> list[str]:
    """Copyable argument values, most recent first: ``extract_candidates`` plus snake-case identifiers,
    minus JSON field names (tau2 tool outputs are JSON, so keys such as "reservation_id" would otherwise
    crowd out the actual values; keys containing a digit are kept: they are ids used as dict keys, e.g.
    airline payment methods {"credit_card_7815826": {...}}) and minus number fragments that only occur
    inside dates/times ("2024", "05", "17" of 2024-05-17).

    With ``param``, values already recorded under that JSON key (e.g. ``"reservation_id": "Q69X3R"``,
    ``"reservations": [...]``, earlier calls' arguments) come first, and for a string ``param_type`` bare
    numbers (prices, counts) go last. Pilot finding (docs/PILOT_RESULTS.md): on airline the 51-option list
    was otherwise dominated by date fragments, prices and markdown labels, missed user ids given in free
    text, and the stand-in S1 filled `reservation_id="4388"` / `origin="1167"` with confidence 1.0."""
    values = set(re.findall(r':\s*"([^"\n]{1,80})"', text))
    fields_ = {k for k in _JSON_KEY.findall(text) if not any(ch.isdigit() for ch in k)} - values
    dates = [m.span() for m in _DATE_SPAN.finditer(text)]

    def date_fragment(c: str) -> bool:
        if not _NUMBER.fullmatch(c):
            return False
        occ = [m.span(1) for m in re.finditer(rf"(?<![\w.])({re.escape(c)})(?![\w.])", text)]
        return bool(occ) and all(any(a <= s and e <= b for a, b in dates) for s, e in occ)

    pool = [c for c in extract_candidates(text, limit=4 * limit) if c not in fields_ and not date_fragment(c)]
    pool += [m.group(1) for m in _IDENT.finditer(text) if any(ch.isdigit() for ch in m.group(1))]
    last = {c: text.rfind(c) for c in pool}
    ranked = sorted(dict.fromkeys(pool), key=lambda c: -last[c])  # recency, de-duplicated
    if param_type == "string":
        ranked.sort(key=lambda c: bool(_NUMBER.fullmatch(c)))  # stable: bare numbers last
    if param:
        front = [v for v in _field_values(text, param) if v not in fields_]
        ranked = front + [c for c in ranked if c not in front]
    return ranked[:limit]


def _tool_kind(tool: Any) -> str | None:
    """tau2 ToolType of a Tool ('read' | 'write' | 'think' | 'generic'), from its @is_tool decorator."""
    tt = getattr(getattr(tool, "_func", None), TOOL_TYPE_ATTR, None)
    return getattr(tt, "value", tt)


# ================================================================== conversation rendering
def _render_msg(m: Any, tool_chars: int = 1200) -> str | None:
    role = getattr(m, "role", None)
    role = getattr(role, "value", role)
    tcs = getattr(m, "tool_calls", None)
    if role == "tool":
        tag = "tool ERROR" if getattr(m, "error", False) else "tool"
        return f"[{tag}] {_trunc(m.content, tool_chars)}"
    who = "agent" if role == "assistant" else "user"
    if tcs:
        return f"[{who} -> tool] " + "; ".join(f"{tc.name}({json.dumps(tc.arguments, default=str)[:400]})"
                                               for tc in tcs)
    if getattr(m, "content", None):
        return f"[{who}] {m.content}"
    return None


class _View:
    """Per-turn S1 inputs: the rendered state, candidate text and approximate gold labels."""

    def __init__(self, state: str, cand_text: str, gold_tool: str | None, gold_args: dict[str, Any] | None,
                 pending: list[Any] | None, confirmed: bool):
        self.state = state
        self.cand_text = cand_text
        self.gold_tool = gold_tool
        self.gold_args = gold_args
        self.pending = pending
        self.confirmed = confirmed


class _Reference:
    """Approximate gold labels from the task's reference actions (``evaluation_criteria.actions``).

    tau2 grades the final DB state (plus communicated info / NL assertions), not the step order, and
    the reference list has no "talk to the user" steps. The next-step oracle is therefore a heuristic:

      * pending actions = reference assistant actions not yet executed successfully (tau2's own
        ``Action.compare_with_tool_call`` matching). A pending READ action that precedes (in reference
        order) an action the agent already executed is dropped as moot: the agent got that information
        another way (retail: authenticated with ``find_user_id_by_email`` where the reference used
        ``find_user_id_by_name_zip``). Without this, the stale READ stays the "next step" for the rest of the
        conversation, because its argument values (name, zip) appear in later tool outputs;
      * gold next step  = the first pending action (reference order) the agent can take now: all its
        scalar argument values have appeared in the conversation (else it must ask / look them up
        first) and, for a WRITE tool under a policy demanding explicit confirmation, the last user
        message is affirmative; ``respond_to_user`` if there is none;
      * policy guard gold = the proposed call matches a pending action (and is confirmed if required);
      * transfer guard gold = ``transfer_to_human_agents`` is a pending reference action.

    Use these labels for simulated/oracle S1 backends and as an "agreement with the reference" metric;
    they are not ground truth for every turn.
    """

    def __init__(self, task: Any, kinds: dict[str, str | None], policy: str):
        crit = getattr(task, "evaluation_criteria", None) if task is not None else None
        self.available = crit is not None
        self.actions = [a for a in ((crit.actions or []) if crit else []) if a.requestor == "assistant"]
        self.kinds = kinds
        self.requires_confirmation = bool(_CONFIRM_POLICY.search(policy or ""))

    def pending(self, messages: list[Any]) -> list[Any]:
        failed = {m.id for m in messages if getattr(m, "role", None) == "tool" and getattr(m, "error", False)}
        used = [False] * len(self.actions)
        for m in messages:
            if getattr(m, "role", None) != "assistant" or not getattr(m, "tool_calls", None):
                continue
            for tc in m.tool_calls:
                if tc.id in failed:
                    continue
                for i, a in enumerate(self.actions):
                    if not used[i] and a.compare_with_tool_call(tc):
                        used[i] = True
                        break
        last_used = max((i for i, u in enumerate(used) if u), default=-1)
        return [a for i, (a, u) in enumerate(zip(self.actions, used))
                if not u and not (i < last_used and self.kinds.get(a.name) == "read")]

    @staticmethod
    def _scalars(v: Any) -> list[str]:
        if isinstance(v, bool) or v is None or isinstance(v, dict):
            return []
        if isinstance(v, (list, tuple)):
            return [s for x in v for s in _Reference._scalars(x)]
        if isinstance(v, float) and v.is_integer():
            return [str(int(v))]
        return [str(v)]

    def next_step(self, pending: list[Any], conv_text: str,
                  confirmed: bool) -> tuple[str, dict[str, Any] | None]:
        for a in pending:  # reference order, skipping actions the agent cannot take yet
            keys = a.compare_args if a.compare_args is not None else list(a.arguments)
            if any(s not in conv_text for k in keys for s in self._scalars(a.arguments.get(k))):
                continue  # an argument value has not appeared yet: must ask / look it up first
            if self.requires_confirmation and self.kinds.get(a.name) == "write" and not confirmed:
                continue  # needs an explicit "yes" first
            return a.name, dict(a.arguments)
        return RESPOND, None

    def matches_pending(self, pending: list[Any], name: str, args: dict[str, Any]) -> bool:
        tc = ToolCall(id="gold", name=name, arguments=args)
        return any(a.compare_with_tool_call(tc) for a in pending)


# ================================================================== the agent
class JevTau2Agent(LLMAgent):  # type: ignore[misc,valid-type]
    """tau2 ``LLMAgent`` + System-1 decision points (see module docstring)."""

    def __init__(self, tools: list[Any], domain_policy: str, llm: str, llm_args: dict | None = None,
                 jev_config: Tau2JevConfig | None = None, task: Any = None,
                 s1: DecisionBackend | None = None, log_dir: str | Path | None = None):
        _require_tau2()
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.jcfg = (jev_config or Tau2JevConfig()).validate_supported()
        self.cfg = self.jcfg.harness
        if self.jcfg.uses_s1():
            self.s1 = s1 or _shared_backend(self.jcfg.s1, self.jcfg.s1_cache)  # type: ignore[arg-type]
        else:
            self.s1 = None
        self.task = task
        self.by_name = {t.name: t for t in tools}
        self.s1_schemas = {t.name: _s1_schema(t.openai_schema) for t in tools}
        self.kinds = {t.name: _tool_kind(t) for t in tools}
        if RESPOND in self.by_name:
            raise ValueError(f"a tau2 tool is named {RESPOND!r}; rename the respond option")
        self.ref = _Reference(task, self.kinds, domain_policy) if self.jcfg.oracle_gold and task is not None \
            else None
        log_dir = log_dir or self.jcfg.log_dir or os.environ.get(ENV_LOG_DIR)
        self.log_dir = Path(log_dir) if log_dir else None
        self.episode_id = uuid.uuid4().hex[:16]
        self.ledger = Ledger()
        self.steps: list[StepLog] = []
        self.seed: int | None = None
        self.error: str | None = None
        self.n_respond_overrides = 0  # S1 said respond_to_user, the LLM still called a tool (see _respond)
        self.shadow_errors: list[str] = []  # failed shadow S1 calls (fail-open; side-car `shadow_s1_errors`)
        self._closed = False
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ tau2 hooks
    def set_seed(self, seed: int) -> None:
        super().set_seed(seed)
        self.seed = seed

    def _generate_next_message(self, message: Any, state: Any) -> Any:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            incoming = list(message.tool_messages)
        elif message is not None:
            state.messages.append(message)
            incoming = [message]
        else:
            incoming = []
        obs = "\n".join(r for r in (_render_msg(m, self.cfg.obs_truncate_chars) for m in incoming) if r)
        if self.steps:  # agent.py semantics: StepLog.observation = what followed the step's action
            self.steps[-1].observation = _trunc(obs, self.cfg.obs_truncate_chars)
        step = len(self.steps)
        self.ledger.step = step
        slog = StepLog(step=step)
        self.steps.append(slog)
        n0 = len(self.ledger.records)
        try:
            out = self._turn(state, slog)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            raise
        self._finish_turn(out, slog, self.ledger.records[n0:])
        return out

    def stop(self, message: Any = None, state: Any = None) -> None:
        """Called by the orchestrator at the end of a simulation (and on emergency cleanup)."""
        if self._closed:
            return
        self._closed = True
        if message is not None and self.steps and not self.steps[-1].observation:
            self.steps[-1].observation = _trunc(_render_msg(message) or "", self.cfg.obs_truncate_chars)
        self._write_sidecar()

    # ------------------------------------------------------------------ S1 plumbing
    def _ask(self, state: str, q: Question, ledger: Ledger) -> Answer:
        assert self.s1 is not None
        return self.s1.ask(state, q, ledger=ledger)

    def _choice(self, view: _View, text: str, options: list[str], dp: str, descriptions: list[str],
                gold: Any, ledger: Ledger | None = None, ranking: list[str] | None = None) -> Answer:
        # JevAgent._choice_tournament only needs `self.s1` and `self._ask`, which this class provides
        return JevAgent._choice_tournament(self, view.state, text, options, dp,  # type: ignore
                                           ledger if ledger is not None else self.ledger, descriptions, gold,
                                           ranking=ranking)

    # --- shadow mode (PREREGISTRATION A5.4; mirrors JevAgent._shadow_ask / _shadow_params / _stop_check)
    def _shadow_ask(self, fn: Callable[[Ledger], Answer], what: str) -> tuple[Answer | None, Tap]:
        """Run a shadow System-1 query ``fn(tap)`` fail-open: its calls go through a ``Tap(shadow=True)`` (tagged
        ``meta.shadow`` / ``meta.measure`` in the episode Ledger, so ``usage_summary`` keeps them out of the episode
        cost), and an S1 error never changes the episode -- it is recorded in ``shadow_errors`` (side-car
        ``shadow_s1_errors``) and the decision is logged without an S1 answer."""
        tap = Tap(self.ledger, shadow=True)
        try:
            return fn(tap), tap
        except Exception as e:  # noqa: BLE001 - fail-open by design
            self.shadow_errors.append(f"{what}: {type(e).__name__}: {str(e)[:200]}")
            return None, tap

    @staticmethod
    def _shadow_log(dp: str, question: str, options: list[str] | None, a: Answer | None, tap: Tap, *, gold: Any,
                    state: str | None, s2_value: Any = None, final_value: Any = None, s2_correct: bool | None = None,
                    s2: tuple[float, float] | None = None, s1_rank: list[str] | None = None,
                    gold_status: str | None = "auto") -> DecisionLog:
        """A ``mode="shadow"`` DecisionLog in the harness's layout: S1's argmax answer / confidence / distribution,
        ``deferred=True`` (the LLM decided), ``s2_value == final_value`` (what was executed), the gold, the S1 cost
        and latency of the shadow call(s) (``tap``) and the LLM's own cost for the decision (``s2``) -- the fields
        ``bench/threshold_theory.decision_pairs`` reads."""
        if gold_status == "auto":
            gold_status = "ok" if gold is not None else "no_oracle"
        d = DecisionLog(dp, Mode.SHADOW.value, question, options, a.value if a else None,
                        a.confidence if a else None, a.distribution if a else None, deferred=True, s2_value=s2_value,
                        final_value=final_value, gold=gold, state=state, shadow=True, s2_correct=s2_correct,
                        gold_status=gold_status, s1_rank=s1_rank)
        d.s1_cost_usd, d.s1_latency_ms = tap.spent()
        if s2 is not None:
            d.s2_cost_usd, d.s2_latency_ms = s2
        return d

    def _s2_spent(self, n0: int) -> tuple[float, float]:
        """(cost, latency) of the non-shadow LLM calls logged since record ``n0`` (the calls behind a decision)."""
        rs = [r for r in self.ledger.records[n0:] if r.component == "s2" and not is_shadow(r)]
        return sum(r.cost_usd for r in rs), sum(r.latency_ms for r in rs)

    def _step_s2_spent(self, dps: tuple[str, ...]) -> tuple[float, float]:
        rs = [r for r in self.ledger.records if r.step == self.ledger.step and r.component == "s2"
              and not is_shadow(r) and r.decision_point in dps]
        return sum(r.cost_usd for r in rs), sum(r.latency_ms for r in rs)

    def _state_log(self, view: _View) -> str | None:
        return view.state if self.jcfg.log_state else None

    def _view(self, state: Any) -> _View:
        c = self.cfg
        msgs = list(state.messages)
        rendered = [r for r in (_render_msg(m, c.obs_truncate_chars) for m in msgs) if r]
        # copyable-argument candidates come from the full, untruncated conversation (never the policy)
        cand_text = "\n".join(r for r in (_render_msg(m, 10**7) for m in msgs) if r)
        n_pol = self.jcfg.s1_policy_chars
        policy = _trunc(self.domain_policy, n_pol) if n_pol > 0 else ""
        head = f"DOMAIN POLICY (excerpt):\n{policy}\n\n" if policy else ""
        budget = max(1000, c.s1_state_max_chars - len(head) - 200)
        # newest messages first until the char budget is used; always keep the current input
        kept: list[str] = []
        used = 0
        for i, r in enumerate(reversed(rendered)):
            if i > 0 and used + len(r) > budget:
                break
            kept.append(r)
            used += len(r) + 1
        kept.reverse()
        omitted = len(rendered) - len(kept)
        conv = ("(... %d earlier messages omitted ...)\n" % omitted if omitted else "") + "\n".join(kept[:-1])
        cur = kept[-1] if kept else "(none)"
        s = f"{head}CONVERSATION SO FAR (oldest first):\n{conv}\n\nLATEST MESSAGE TO THE AGENT:\n{cur}"
        s = _trunc(s, c.s1_state_max_chars)
        last_user = next((m for m in reversed(msgs) if getattr(m, "role", None) == "user"
                          and getattr(m, "content", None)), None)
        confirmed = bool(last_user is not None and _AFFIRM.search(last_user.content or ""))
        gold_tool = gold_args = pending = None
        if self.ref is not None and self.ref.available:
            pending = self.ref.pending(msgs)
            gold_tool, gold_args = self.ref.next_step(pending, cand_text, confirmed)
        return _View(s, cand_text, gold_tool, gold_args, pending, confirmed)

    # ------------------------------------------------------------------ S2 plumbing
    def _llm(self, model: str, llm_args: dict[str, Any], messages: list[Any], tools: list[Any] | None,
             dp: str, slog: StepLog, tool_choice: Any = None) -> Any:
        slog.llm_calls += 1
        slog.model = model
        t = time.perf_counter()
        msg = _tau2_llm.generate(model=model, messages=messages, tools=tools or None, tool_choice=tool_choice,
                                 call_name=f"jev_{dp}", **llm_args)
        latency = (msg.generation_time_seconds or (time.perf_counter() - t)) * 1000
        usage = msg.usage or {}
        n_in, n_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        cached = _cached_tokens(msg.raw_data)
        price = default_prices().get(model)
        litellm_cost = float(msg.cost or 0.0)
        cost = price.cost(n_in, n_out, cached) if price else litellm_cost
        self.ledger.add(CallRecord(component="s2", backend="litellm", model=model, decision_point=dp,
                                   input_tokens=n_in, output_tokens=n_out, cost_usd=cost, latency_ms=latency,
                                   priced=price is not None or litellm_cost > 0,
                                   meta={"litellm_cost": litellm_cost, "cached_input_tokens": cached,
                                         "n_tools": len(tools or []),
                                         "tool_choice": tool_choice if not isinstance(tool_choice, dict)
                                         else json.dumps(tool_choice)}))
        return msg

    # ------------------------------------------------------------------ decision points
    def _route(self, view: _View, slog: StepLog) -> tuple[str, dict[str, Any]]:
        c = self.cfg.routing
        if c.mode not in (Mode.TRIGGER, Mode.SHADOW):
            return self.llm, self.llm_args
        q = Noul(ROUTING_Q, decision_point=DP.ROUTING.value)
        if c.mode == Mode.SHADOW:  # always the main LLM; S1's answer is only logged (no gold: tau2 has no router oracle)
            a, tap = self._shadow_ask(lambda led: self._ask(view.state, q, led), "routing")
            slog.decisions.append(self._shadow_log(DP.ROUTING.value, q.text, None, a, tap, gold=None,
                                                   state=self._state_log(view), final_value="large"))
            return self.llm, self.llm_args
        a = self._ask(view.state, q, self.ledger)
        big = a.p_yes >= c.tau
        slog.decisions.append(DecisionLog(DP.ROUTING.value, c.mode.value, q.text, s1_value=big,
                                          s1_conf=a.confidence, final_value="large" if big else "small",
                                          state=self._state_log(view)))
        if big:
            return self.llm, self.llm_args
        args = self.jcfg.small_llm_args if self.jcfg.small_llm_args is not None else self.llm_args
        if "seed" in self.llm_args and "seed" not in args:
            args = {**args, "seed": self.llm_args["seed"]}
        return self.jcfg.small_llm, args  # type: ignore[return-value]

    def _param_options(self, p: ParamInfo, view: _View, gold: Any,
                       gold_args: dict[str, Any] | None, n_max: int) -> tuple[list[str], list[Any] | None, Any] | None:
        """The ``tool_params`` Choice for one required argument, as cascade / replace build it: (option labels,
        the bounded values or None for copyable spans, the gold label), or None when the argument goes to the LLM
        (free text, or no copyable candidate in the conversation). Shared by the live modes and shadow mode so the
        shadow arm asks exactly the live arm's question."""
        bounded = p.bounded_values()
        if bounded is not None:
            opts = [v if isinstance(v, str) else json.dumps(v) for v in bounded]
            gold_s = (gold if isinstance(gold, str) else json.dumps(gold)) if gold is not None else None
            return opts, bounded, gold_s
        if p.is_copyable() and not _FREE_TEXT_PARAM.match(p.name) and self.cfg.tool_params.extra.get("copy", True):
            cands = _value_candidates(view.cand_text, n_max, param=p.name, param_type=p.type)
            if not cands:
                return None
            opts = cands + [OTHER]
            if gold is not None:
                gold_s = str(gold) if str(gold) in opts else OTHER
            else:
                gold_s = OTHER if gold_args is not None else None
            return opts, None, gold_s
        return None

    @staticmethod
    def _param_question(name: str, p: ParamInfo, opts: list[str], gold_s: Any) -> Choice:
        return Choice(f"Which value should the argument `{p.name}` of tool `{name}` take? "
                      f"({p.schema.get('description', '')})", opts, decision_point=DP.TOOL_PARAMS.value,
                      gold=gold_s)

    def _n_param_options(self) -> int:
        return min(getattr(self.s1, "max_options", MAX_OPTIONS), MAX_OPTIONS) - 1  # room for OTHER

    def _fill_params(self, name: str, view: _View, slog: StepLog,
                     gold_args: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
        """S1 fills bounded/copyable *required* params (optional ones are omitted unless the LLM is
        called). Returns (filled args, required param names still needing the LLM)."""
        c = self.cfg.tool_params
        infos = [p for p in params_of(self.s1_schemas[name]) if p.required]
        if c.mode in (Mode.OFF, Mode.SHADOW):  # shadow: the LLM fills every argument (S1 is logged afterwards)
            return {}, [p.name for p in infos]
        filled: dict[str, Any] = {}
        todo: list[str] = []
        n_max = self._n_param_options()
        for p in infos:
            gold = (gold_args or {}).get(p.name)
            po = self._param_options(p, view, gold, gold_args, n_max)
            if po is None:
                todo.append(p.name)
                continue
            opts, bounded, gold_s = po
            if len(opts) < 2:
                filled[p.name] = coerce(opts[0], p) if bounded is None else bounded[0]
                continue
            q = self._param_question(name, p, opts, gold_s)
            if c.mode == Mode.CASCADE and c.extra.get("always_defer"):   # A6.11 (c).4 / A6.20: S1 not queried
                d = DecisionLog(DP.TOOL_PARAMS.value, c.mode.value, q.text, opts, None, None, deferred=True,
                                gold=gold_s, state=self._state_log(view))
                d.gold_aux = {"always_defer": True}
                slog.decisions.append(d)
                todo.append(p.name)
                continue
            a = self._ask(view.state, q, self.ledger)
            ok = a.value != OTHER and (c.mode == Mode.REPLACE or a.confidence >= c.tau)
            slog.decisions.append(DecisionLog(DP.TOOL_PARAMS.value, c.mode.value, q.text, opts, a.value,
                                              a.confidence, deferred=not ok, gold=gold_s,
                                              final_value=a.value if ok else None,
                                              state=self._state_log(view)))
            if ok:
                typed = bounded is not None and not isinstance(bounded[0], str)
                filled[p.name] = json.loads(a.value) if typed else coerce(a.value, p)
            else:
                todo.append(p.name)
        return filled, todo

    def _shadow_params(self, name: str, args: dict[str, Any], view: _View, slog: StepLog,
                       s2: tuple[float, float]) -> None:
        """tool_params=shadow: after the LLM filled the arguments of the call that is executed, ask System-1 the
        Choice cascade / replace would ask for every required argument it could fill, and log both answers
        (``s2_value`` = the LLM's value in the option-label space; ``s2_correct`` = that value vs. the raw reference
        value). As in cascade mode, argument gold exists only when the executed tool is the reference's next step."""
        if name not in self.s1_schemas:
            return
        gold_args = view.gold_args if view.gold_tool == name else None
        n_max = self._n_param_options()
        for p in (p for p in params_of(self.s1_schemas[name]) if p.required):
            gold = (gold_args or {}).get(p.name)
            po = self._param_options(p, view, gold, gold_args, n_max)
            if po is None or len(po[0]) < 2:
                continue
            opts, _, gold_s = po
            q = self._param_question(name, p, opts, gold_s)
            a, tap = self._shadow_ask(lambda led, q=q: self._ask(view.state, q, led), f"tool_params:{p.name}")
            v = args.get(p.name)
            slog.decisions.append(self._shadow_log(
                DP.TOOL_PARAMS.value, q.text, opts, a, tap, gold=gold_s, state=self._state_log(view),
                s2_value=_label(v), final_value=_label(v), s2=s2,
                s2_correct=None if gold is None else (v is not None and _same_value(v, gold)),
                gold_status="ok" if gold_s is not None else None))

    def _emit_tool(self, name: str, view: _View, model: str, llm_args: dict[str, Any], messages: list[Any],
                   slog: StepLog) -> Any:
        # argument gold is only defined when S1 picked the reference tool
        filled, todo = self._fill_params(name, view, slog, view.gold_args if view.gold_tool == name else None)
        if not todo:  # every required argument decided by S1: no LLM call at all
            return AssistantMessage(role="assistant", content=None, tool_calls=[
                ToolCall(id=f"call_jev_{uuid.uuid4().hex[:10]}", name=name, arguments=filled)])
        hint = (f"You must now call `{name}`. Already-determined arguments: {json.dumps(filled)}. "
                "Provide the remaining arguments.")
        resp = self._llm(model, llm_args, messages + [UserMessage(role="user", content=hint)],
                         [self.by_name[name]], DP.TOOL_PARAMS.value, slog,
                         tool_choice={"type": "function", "function": {"name": name}})
        if resp.tool_calls:
            tc = resp.tool_calls[0]
            resp.tool_calls = [ToolCall(id=tc.id, name=name, arguments={**tc.arguments, **filled})]
            resp.content = None
        return resp

    def _emit_complete(self, name: str, view: _View, slog: StepLog) -> Any:
        """``s1_act: complete`` (E22): System-1 fills the required arguments of the tool it chose; when it decides every
        one, the call is emitted with no LLM call, else None (the turn goes to the LLM unchanged). Argument decisions
        of an incomplete call are logged as discarded deferrals (``gold_aux.not_acted``)."""
        n0 = len(slog.decisions)
        filled, todo = self._fill_params(name, view, slog, view.gold_args if view.gold_tool == name else None)
        if not todo:
            return AssistantMessage(role="assistant", content=None, tool_calls=[
                ToolCall(id=f"call_jev_{uuid.uuid4().hex[:10]}", name=name, arguments=filled)])
        for d in slog.decisions[n0:]:
            if d.dp == DP.TOOL_PARAMS.value and not d.deferred:
                d.deferred, d.final_value = True, None
                d.gold_aux = {**(d.gold_aux or {}), "not_acted": "incomplete"}
        return None

    def _log_params(self, name: str, args: dict[str, Any], view: _View, slog: StepLog) -> None:
        """``log_params`` (E22 harvest): the argument Choices of the executed LLM call, with the LLM's value, no S1."""
        if name not in self.s1_schemas:
            return
        gold_args = view.gold_args if view.gold_tool == name else None
        n_max = self._n_param_options()
        for p in (p for p in params_of(self.s1_schemas[name]) if p.required):
            gold = (gold_args or {}).get(p.name)
            po = self._param_options(p, view, gold, gold_args, n_max)
            if po is None or len(po[0]) < 2:
                continue
            opts, _, gold_s = po
            q = self._param_question(name, p, opts, gold_s)
            v = args.get(p.name)
            d = DecisionLog(DP.TOOL_PARAMS.value, Mode.OFF.value, q.text, opts, s2_value=_label(v),
                            final_value=_label(v), gold=gold_s,
                            s2_correct=None if gold is None else (v is not None and _same_value(v, gold)),
                            gold_status="ok" if gold_s is not None else None)
            d.gold_aux = {"param": p.name, "tool": name}
            slog.decisions.append(d)

    def _respond(self, model: str, llm_args: dict[str, Any], messages: list[Any], slog: StepLog,
                 dlog: DecisionLog) -> Any:
        """S1 decided ``respond_to_user``: the LLM only writes the message. Some providers ignore
        ``tool_choice="none"`` and even a tool-less request (Gemini 3.5 Flash, both through litellm's native
        Gemini API and the OpenAI-compatible endpoint, still returns a function call when it "wants" one;
        see docs/PILOT_RESULTS.md). A returned text is kept and its tool calls are dropped, so the S1 decision
        stands. A tool call with no text is executed as the LLM's own choice and logged as a deferral to the
        LLM (``deferred=True``, ``s2_value`` = the tool), so decision metrics count it as an override; with
        ``respond_fallback: defer`` the LLM is instead asked again with its tools (a cascade deferral: the call made
        under ``tool_choice="none"`` saw no tool declarations, and its tool name may not exist)."""
        if self.jcfg.respond_tool_choice == "none":
            resp = self._llm(model, llm_args, messages, self.tools, DP.TOOL_SELECT.value, slog, tool_choice="none")
        else:
            resp = self._llm(model, llm_args, messages, None, DP.TOOL_SELECT.value, slog)
        if not resp.tool_calls:
            return resp
        if (resp.content or "").strip():
            resp.tool_calls = None
            dlog.s2_value = RESPOND
        else:
            if self.jcfg.respond_fallback == "defer":
                resp = self._llm(model, llm_args, messages, self.tools, DP.TOOL_SELECT.value, slog)
            dlog.deferred = True
            dlog.s2_value = dlog.final_value = _action_name(resp)
            self.n_respond_overrides += 1
        return resp

    def _select(self, model: str, llm_args: dict[str, Any], messages: list[Any], view: _View,
                slog: StepLog) -> Any:
        c = self.cfg.tool_select
        names = list(self.by_name) + [RESPOND]
        dlog: DecisionLog | None = None
        if c.mode == Mode.SHADOW:  # ask exactly what cascade asks; the LLM decides below, unchanged
            descs = [tool_desc(self.s1_schemas[n]) for n in self.by_name] + [RESPOND_DESC]
            rank: list[str] = []
            a, tap = self._shadow_ask(lambda led: self._choice(view, TOOL_SELECT_Q, names, DP.TOOL_SELECT.value,
                                                                descs, view.gold_tool, ledger=led, ranking=rank),
                                      "tool_select")
            dlog = self._shadow_log(DP.TOOL_SELECT.value, TOOL_SELECT_Q, names, a, tap, gold=view.gold_tool,
                                    state=self._state_log(view), s1_rank=list(rank) if a is not None and rank else None)
            slog.decisions.append(dlog)
        elif c.mode == Mode.CASCADE and c.extra.get("always_defer"):
            # PREREGISTRATION A6.11 (c).4 / A6.20: the fit accepts no confidence level, so the plug-in policy defers
            # every decision and System-1 is not queried (paying for a score that cannot act would be another policy)
            dlog = DecisionLog(DP.TOOL_SELECT.value, c.mode.value, TOOL_SELECT_Q, names, None, None, None,
                               deferred=True, gold=view.gold_tool, state=self._state_log(view))
            dlog.gold_aux = {**(dlog.gold_aux or {}), "always_defer": True}
            slog.decisions.append(dlog)
        elif c.mode in (Mode.REPLACE, Mode.CASCADE, Mode.ASSIST):
            descs = [tool_desc(self.s1_schemas[n]) for n in self.by_name] + [RESPOND_DESC]
            a = self._choice(view, TOOL_SELECT_Q, names, DP.TOOL_SELECT.value, descs, view.gold_tool)
            dlog = DecisionLog(DP.TOOL_SELECT.value, c.mode.value, TOOL_SELECT_Q, names, a.value,
                               a.confidence, a.distribution, gold=view.gold_tool, state=self._state_log(view))
            slog.decisions.append(dlog)
            if c.mode == Mode.ASSIST:  # the LLM decides among the top-k tools (it can always reply in text)
                keep = [n for n in a.top_k(c.k) if n != RESPOND]
                resp = self._llm(model, llm_args, messages, [self.by_name[n] for n in keep],
                                 DP.TOOL_SELECT.value, slog)
                dlog.s2_value = dlog.final_value = _action_name(resp)
                return resp
            if c.mode == Mode.REPLACE or a.confidence >= c.tau:
                if self.jcfg.s1_act == "complete":  # E22: act only on a complete call; else the LLM-only turn
                    out = None if a.value == RESPOND else self._emit_complete(a.value, view, slog)
                    if out is not None:
                        dlog.final_value = a.value
                        return out
                    dlog.deferred = True
                    dlog.gold_aux = {**(dlog.gold_aux or {}),
                                     "not_acted": "respond" if a.value == RESPOND else "incomplete"}
                else:
                    dlog.final_value = a.value
                    if a.value == RESPOND:  # S1 decided "talk"; the LLM only writes the message
                        return self._respond(model, llm_args, messages, slog, dlog)
                    return self._emit_tool(a.value, view, model, llm_args, messages, slog)
            else:
                dlog.deferred = True  # cascade: fall through to the LLM
        n1 = len(self.ledger.records)
        resp = self._llm(model, llm_args, messages, self.tools, DP.TOOL_SELECT.value, slog)
        name = _action_name(resp)
        if dlog is not None:  # cascade deferral or shadow
            dlog.s2_value = dlog.final_value = name
            if dlog.shadow:
                dlog.s2_correct = (name == view.gold_tool) if view.gold_tool is not None else None
                dlog.s2_cost_usd, dlog.s2_latency_ms = self._s2_spent(n1)
        elif c.mode == Mode.OFF:
            # options + state kept so llm_only runs are harvestable (training/harvest.py --label s2)
            slog.decisions.append(DecisionLog(DP.TOOL_SELECT.value, "off", TOOL_SELECT_Q, names,
                                              s2_value=name, final_value=name, gold=view.gold_tool,
                                              state=self._state_log(view)))
        return resp

    def _check(self, msg: Any, view: _View, slog: StepLog) -> tuple[str, str] | None:
        """Run the applicable S1 guards on a proposal (stop -> policy -> tool_select, first rejection
        wins). Returns (dp, verifier note) on rejection, None if accepted."""
        calls = list(msg.tool_calls or [])
        step_s2 = (DP.TOOL_SELECT.value, DP.TOOL_PARAMS.value)
        if self.cfg.stop.mode in (Mode.GUARD, Mode.SHADOW):
            for tc in calls:
                if tc.name != TRANSFER_TOOL:
                    continue
                gold = (any(a.name == TRANSFER_TOOL for a in view.pending)
                        if view.pending is not None else None)
                q = Noul(STOP_Q, decision_point=DP.STOP.value, gold=gold)
                if self.cfg.stop.mode == Mode.SHADOW:  # the guard's question, logged; the transfer goes ahead
                    a, tap = self._shadow_ask(lambda led: self._ask(view.state, q, led), "stop")
                    slog.decisions.append(self._shadow_log(
                        DP.STOP.value, q.text, None, a, tap, gold=gold, state=self._state_log(view), s2_value=True,
                        final_value=True, s2_correct=bool(gold) if gold is not None else None,
                        s2=self._step_s2_spent(step_s2)))
                    continue
                a = self._ask(view.state, q, self.ledger)
                ok = a.p_yes >= self.cfg.stop.tau
                slog.decisions.append(DecisionLog(DP.STOP.value, "guard", q.text, None, ok, a.confidence,
                                                  final_value=ok, gold=gold, state=self._state_log(view)))
                if not ok:
                    return DP.STOP.value, ("[Verifier] Transferring to a human agent does not look justified "
                                           "yet. Try to resolve the request with the available tools and the "
                                           "policy, or explain to the user what you can and cannot do.")
        pg = self.jcfg.policy_guard
        if pg.mode in (Mode.GUARD, Mode.SHADOW):
            scope = pg.extra.get("scope", "write")
            confirm = self.ref.requires_confirmation if self.ref is not None \
                else bool(_CONFIRM_POLICY.search(self.domain_policy or ""))
            for tc in calls:
                if tc.name == TRANSFER_TOOL or (scope == "write" and self.kinds.get(tc.name) != "write"):
                    continue
                args = json.dumps(tc.arguments, default=str)[:600]
                text = pg.extra.get("question") or (
                    f"The agent is about to call the state-changing tool `{tc.name}` with arguments {args}. "
                    "Does this action comply with the domain policy"
                    + (", and has the user explicitly confirmed exactly this action (answered yes) in the "
                       "conversation?" if confirm else "?"))
                gold = None
                if view.pending is not None and self.ref is not None:
                    gold = self.ref.matches_pending(view.pending, tc.name, tc.arguments) and \
                        (view.confirmed or not confirm or self.kinds.get(tc.name) != "write")
                q = Noul(text, decision_point="policy", gold=gold)
                if pg.mode == Mode.SHADOW:  # the guard's question, logged; the call goes ahead
                    a, tap = self._shadow_ask(lambda led: self._ask(view.state, q, led), "policy")
                    slog.decisions.append(self._shadow_log(
                        "policy", q.text, None, a, tap, gold=gold, state=self._state_log(view), s2_value=True,
                        final_value=True, s2_correct=bool(gold) if gold is not None else None,
                        s2=self._step_s2_spent(step_s2)))
                    continue
                a = self._ask(view.state, q, self.ledger)
                ok = a.p_yes >= pg.tau
                slog.decisions.append(DecisionLog("policy", "guard", q.text, None, ok, a.confidence,
                                                  final_value=ok, gold=gold, state=self._state_log(view)))
                if not ok:
                    lack = " or to lack the user's explicit confirmation" if confirm else ""
                    return "policy", (
                        f"[Policy check] Your proposed action `{tc.name}({args[:300]})` was judged likely "
                        f"to violate the domain policy{lack}. Re-read the policy. If confirmation is "
                        "required, list the action details and ask the user to confirm (yes) before acting. "
                        "Decide again: respond to the user or call a tool.")
        if self.cfg.tool_select.mode == Mode.GUARD:  # in guard mode every proposal comes from the LLM
            name = _action_name(msg)
            if calls:
                desc = f"call `{name}` with arguments {json.dumps(calls[0].arguments, default=str)[:500]}"
            else:
                desc = f"reply to the user: \"{_trunc(msg.content, 300)}\""
            gold = (name == view.gold_tool) if view.gold_tool is not None else None
            q = Noul(f"The agent proposes to {desc}. Is this the right next step, consistent with the domain "
                     "policy?", decision_point=DP.TOOL_SELECT.value, gold=gold)
            a = self._ask(view.state, q, self.ledger)
            ok = a.p_yes >= self.cfg.tool_select.tau
            slog.decisions.append(DecisionLog(DP.TOOL_SELECT.value, "guard", q.text, None, ok, a.confidence,
                                              s2_value=name, final_value=name if ok else None, gold=gold,
                                              state=self._state_log(view)))
            if not ok:
                return DP.TOOL_SELECT.value, (f"[Verifier] Your proposed next step ({desc[:300]}) was "
                                              "judged likely wrong. Reconsider and choose the correct next "
                                              "action (a tool call or a message to the user).")
        return None

    def _turn(self, state: Any, slog: StepLog) -> Any:
        messages = state.system_messages + state.messages
        view = self._view(state)
        slog.oracle_tool = view.gold_tool
        n0 = len(self.ledger.records)
        model, llm_args = self._route(view, slog)
        msg = self._select(model, llm_args, messages, view, slog)
        # guards (agent.py semantics): up to guard_max_retries checks per turn, shared by all guards; a
        # rejected proposal is never executed, the LLM regenerates with an ephemeral verifier note, and
        # the last regeneration is accepted
        for _ in range(self.cfg.guard_max_retries):
            rejected = self._check(msg, view, slog)
            if rejected is None:
                break
            dp, note = rejected
            msg = self._llm(model, llm_args, messages + [UserMessage(role="user", content=note)], self.tools,
                            dp, slog)
        if self.cfg.tool_params.mode == Mode.SHADOW and msg.tool_calls:  # S1 logged on the executed call's args
            tc = msg.tool_calls[0]
            self._shadow_params(tc.name, dict(tc.arguments or {}), view, slog, self._s2_spent(n0))
        elif self.jcfg.log_params and msg.tool_calls:
            tc = msg.tool_calls[0]
            self._log_params(tc.name, dict(tc.arguments or {}), view, slog)
        return msg

    # ------------------------------------------------------------------ bookkeeping
    def _finish_turn(self, out: Any, slog: StepLog, recs: list[CallRecord]) -> None:
        # shadow (measurement) calls are itemised separately and never enter the turn's agent_cost
        s1 = [r for r in recs if r.component == "s1" and not is_shadow(r)]
        s2 = [r for r in recs if r.component == "s2" and not is_shadow(r)]
        sh = [r for r in recs if is_shadow(r)]
        name = _action_name(out)
        slog.tool = name
        if out.tool_calls:
            slog.args = dict(out.tool_calls[0].arguments)
        else:
            slog.args = {"message": _trunc(out.content, 500)}
        # tau2-native accounting: agent_cost = litellm-priced LLM calls of this turn (incl. discarded
        # guard/regeneration calls) + S1 cost (jevharness price table); tokens = LLM tokens
        s1_cost = sum(r.cost_usd for r in s1)
        out.cost = sum(float(r.meta.get("litellm_cost", 0.0)) for r in s2) + s1_cost
        out.usage = {"prompt_tokens": sum(r.input_tokens for r in s2),
                     "completion_tokens": sum(r.output_tokens for r in s2)}
        if not s2:
            out.generation_time_seconds = sum(r.latency_ms for r in s1) / 1000
        compact = {"episode": self.episode_id, "step": slog.step, "config": self.cfg.name,
                   "llm_calls": len(s2), "s1_calls": len(s1), "s1_cost_usd": s1_cost,
                   "s1_latency_ms": sum(r.latency_ms for r in s1), "model": slog.model,
                   "shadow_calls": len(sh), "shadow_cost_usd": sum(r.cost_usd for r in sh),
                   "decisions": [{"dp": d.dp, "mode": d.mode, "s1": d.s1_value, "conf": d.s1_conf,
                                  "deferred": d.deferred, "s2": d.s2_value, "final": d.final_value,
                                  "shadow": d.shadow}
                                 for d in slog.decisions]}
        out.raw_data = {**(out.raw_data or {}), "jev": json.loads(json.dumps(compact, default=str))}

    def sidecar_record(self) -> dict[str, Any]:
        return {
            "episode": self.episode_id,
            "task_id": getattr(self.task, "id", None),
            "seed": self.seed,
            "config": self.cfg.name,
            "config_dump": self.jcfg.model_dump(mode="json"),
            "llm": self.llm,
            "small_llm": self.jcfg.small_llm if self.cfg.routing.mode == Mode.TRIGGER else None,
            "backend": (self.s1.describe().get("backend") if self.s1 is not None else None),
            "steps": [asdict(s) for s in self.steps],
            # harness semantics: shadow calls stay in `calls` but out of usage.total / usage.s1 (usage.shadow)
            "usage": usage_summary(self.ledger),
            "calls": self.ledger.to_list(),
            "wall_ms": (time.perf_counter() - self._t0) * 1000,
            "respond_overrides": self.n_respond_overrides,
            "shadow_points": self.jcfg.shadow_points(),
            "shadow_s1_errors": list(self.shadow_errors),
            "nl_judge": nl_judge(),
            "error": self.error,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _write_sidecar(self) -> None:
        if self.log_dir is None:
            return
        rec = self.sidecar_record()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with _SIDECAR_LOCK, open(self.log_dir / SIDECAR_FILE, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


_SIDECAR_LOCK = threading.Lock()


def _action_name(msg: Any) -> str:
    tcs = getattr(msg, "tool_calls", None)
    return tcs[0].name if tcs else RESPOND


def _label(v: Any) -> str | None:
    """An argument value in the option-label space of the tool_params ``Choice`` (JSON for non-strings)."""
    return None if v is None else (v if isinstance(v, str) else json.dumps(v))


def _same_value(a: Any, b: Any) -> bool:
    """Argument equality up to representation (``"3"`` == 3 == 3.0, ``True`` == ``"true"``)."""
    def norm(x: Any) -> str:
        if isinstance(x, bool):
            return "true" if x else "false"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return x.strip() if isinstance(x, str) else json.dumps(x, sort_keys=True)
    return norm(a) == norm(b)


def _cached_tokens(raw: dict[str, Any] | None) -> int:
    u = (raw or {}).get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    cached = det.get("cached_tokens") if isinstance(det, dict) else 0
    return int(cached or u.get("cache_read_input_tokens") or 0)


# ================================================================== registry
def make_agent_factory(config: Tau2JevConfig | str | Path | None = None, s1: DecisionBackend | None = None,
                       log_dir: str | Path | None = None) -> Callable[..., Any]:
    """tau2 agent factory. ``config=None`` reads ``$JEV_HARNESS_CONFIG`` each time an agent is built."""

    def factory(tools: list[Any], domain_policy: str, **kwargs: Any) -> JevTau2Agent:
        cfg = config if isinstance(config, Tau2JevConfig) else load_config(config)
        return JevTau2Agent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"),
                            llm_args=kwargs.get("llm_args"), jev_config=cfg, task=kwargs.get("task"), s1=s1,
                            log_dir=log_dir)

    return factory


def register(name: str = AGENT_NAME, config: Tau2JevConfig | str | Path | None = None,
             s1: DecisionBackend | None = None, log_dir: str | Path | None = None) -> str:
    """Register the agent with tau2's registry (idempotent per name). Returns the name."""
    _require_tau2()
    from tau2.registry import registry

    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(make_agent_factory(config, s1, log_dir), name,
                                        metadata={"jevharness": True})
    return name


# ================================================================== A5.4 airline shadow / test split
REPO_ROOT = Path(__file__).resolve().parents[4]
AIRLINE_TASK_IDS_FILE = REPO_ROOT / "configs" / "tau2" / "airline_task_ids.json"
SPLIT_SEED = 0
SPLIT_N_SHADOW = 15
SPLITS = ("all", "shadow", "test")


def _natural_key(s: str) -> tuple[int, Any]:
    return (0, int(s)) if s.isdigit() else (1, s)


def airline_split(task_ids: Iterable[Any], seed: int = SPLIT_SEED,
                  n_shadow: int = SPLIT_N_SHADOW) -> tuple[list[str], list[str]]:
    """PREREGISTRATION A5.4 (c): one seeded shuffle of the task ids in sorted order -- numeric ids ascending
    (0, 1, 2, ..., 49), any non-numeric id after them lexicographically -- with ``random.Random(seed)``; the first
    ``n_shadow`` ids are the **shadow split** (used only to refit tau-hat*), the rest the **test set**. Both lists
    come back in sorted order. Deterministic; ``configs/tau2/airline_task_ids.json`` records the result and
    ``tests/test_tau2_split.py`` pins it."""
    ids = sorted({str(t) for t in task_ids}, key=_natural_key)
    if not 0 <= n_shadow <= len(ids):
        raise ValueError(f"n_shadow={n_shadow} out of range for {len(ids)} task ids")
    random.Random(seed).shuffle(ids)
    return sorted(ids[:n_shadow], key=_natural_key), sorted(ids[n_shadow:], key=_natural_key)


def read_task_ids(path: str | Path) -> list[str]:
    """Task ids from a file: a JSON list, a JSON object with ``task_ids`` (``configs/tau2/airline_task_ids.json``),
    or whitespace / comma separated text (one id per line, ``#`` comments allowed)."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        lines = [ln.split("#", 1)[0] for ln in text.splitlines()]
        return [t for ln in lines for t in ln.replace(",", " ").split()]
    if isinstance(data, dict):
        data = data.get("task_ids")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of task ids or an object with `task_ids`")
    return [str(t) for t in data]


@functools.lru_cache(maxsize=1)
def _airline_record() -> dict[str, Any]:
    return json.loads(AIRLINE_TASK_IDS_FILE.read_text())


def airline_task_ids(check_installed: bool = True) -> list[str]:
    """The 50 airline task ids of the committed list (``configs/tau2/airline_task_ids.json``, fetched once from
    tau2-bench; its ``source`` records the commit). With tau2 installed and ``check_installed`` the installed
    dataset is cross-checked: a different task list raises, so the pre-registered split can never silently drift."""
    ids = [str(t) for t in _airline_record()["task_ids"]]
    if check_installed and HAVE_TAU2:
        try:
            from tau2.runner import get_tasks

            installed = sorted((str(t.id) for t in get_tasks("airline")), key=_natural_key)
        except Exception as e:  # noqa: BLE001 - data dir missing etc.: the committed list stands
            print(f"[jev] WARNING: could not read the installed tau2 airline tasks ({type(e).__name__}: {e}); "
                  f"using {AIRLINE_TASK_IDS_FILE.name}", file=sys.stderr)
            return ids
        if installed != sorted(ids, key=_natural_key):
            raise ValueError("the installed tau2 airline task list differs from the pre-registered one in "
                             f"{AIRLINE_TASK_IDS_FILE}: {sorted(set(installed) ^ set(ids), key=_natural_key)}")
    return ids


def airline_task_split(check_installed: bool = False) -> tuple[list[str], list[str]]:
    """(shadow ids, test ids) of the pre-registered split, from the committed record (which the split function
    reproduces -- asserted on load, so a stale file is never used silently)."""
    rec = _airline_record()
    ids = airline_task_ids(check_installed)
    sp = rec.get("split") or {}
    shadow, test = airline_split(ids, int(sp.get("seed", SPLIT_SEED)), int(sp.get("n_shadow", SPLIT_N_SHADOW)))
    if sp and (list(map(str, sp.get("shadow", []))) != shadow or list(map(str, sp.get("test", []))) != test):
        raise ValueError(f"{AIRLINE_TASK_IDS_FILE}: the recorded split does not match airline_split(); "
                         "regenerate the file (python -m jevharness.bench.suites.tau2 split airline --write)")
    return shadow, test


def split_of(domain: str | None, task_id: Any) -> str | None:
    """``"shadow"`` | ``"test"`` for an airline task id; None for any other domain (no shadow split there: every
    retail task is a test task, A5.4 c) or an unknown id."""
    if str(domain) != "airline" or task_id is None:
        return None
    try:
        shadow, test = airline_task_split()
    except (OSError, ValueError, KeyError):
        return None
    tid = str(task_id)
    return "shadow" if tid in shadow else "test" if tid in test else None


def _traj_split(t: Trajectory | dict[str, Any]) -> str | None:
    d = t if isinstance(t, dict) else t.to_dict()
    tau = (d.get("meta") or {}).get("tau2") or {}
    return tau.get("split") if tau.get("split") is not None else split_of(tau.get("domain"), tau.get("task_id"))


def filter_split(trajs: Iterable[Any], split: str) -> list[Any]:
    """Trajectories of one split. ``test`` keeps the airline test tasks AND every task of a domain without a shadow
    split (retail; A5.4 c), ``shadow`` only the 15 airline shadow tasks, ``all`` everything. Older JSONL without
    ``meta.tau2.split`` is classified from the task id."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    out = []
    for t in trajs:
        sp = _traj_split(t)
        if split == "all" or (split == "shadow" and sp == "shadow") or (split == "test" and sp != "shadow"):
            out.append(t)
    return out


# ================================================================== results -> Trajectory JSONL
def _load_results(path: str | Path) -> dict[str, Any]:
    """tau2 Results as plain JSON (monolithic results.json, or the dir format with simulations/)."""
    p = Path(path)
    meta = p / "results.json" if p.is_dir() else p
    data = json.loads(meta.read_text())
    sims_dir = meta.parent / "simulations"
    if not data.get("simulations") and sims_dir.is_dir():
        data["simulations"] = [json.loads(f.read_text()) for f in sorted(sims_dir.glob("*.json"))]
    return data


def load_sidecar(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def _dc(cls: type, d: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def _msg_role(m: dict[str, Any]) -> str:
    return str(m.get("role"))


def _episode_of(sim: dict[str, Any]) -> str | None:
    for m in sim.get("messages") or []:
        jev = ((m.get("raw_data") or {}) if _msg_role(m) == "assistant" else {}).get("jev")
        if jev:
            return jev.get("episode")
    return None


def _fallback_from_messages(sim: dict[str, Any], llm: str | None) -> tuple[list[StepLog], Ledger]:
    """Steps/usage from the tau2 messages alone (stock ``llm_agent`` runs, or a missing side-car)."""
    ledger = Ledger()
    steps: list[StepLog] = []
    prices = default_prices()
    msgs = sim.get("messages") or []
    for i, m in enumerate(msgs):
        if _msg_role(m) != "assistant" or (m.get("usage") is None and not m.get("raw_data")):
            continue  # e.g. tau2's fixed greeting
        jev = (m.get("raw_data") or {}).get("jev") or {}
        step = len(steps)
        ledger.step = step
        u = m.get("usage") or {}
        n_llm = int(jev.get("llm_calls", 1))
        model = jev.get("model") or ((m.get("raw_data") or {}).get("model")) or llm
        n_in, n_out = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        s1_cost = float(jev.get("s1_cost_usd", 0.0))
        if n_llm:
            price = prices.get(model)
            cost = price.cost(n_in, n_out) if price else float(m.get("cost") or 0.0) - s1_cost
            for j in range(n_llm):  # tokens/cost of the turn split evenly over its LLM calls
                ledger.add(CallRecord("s2", "litellm", model, "tool_select", n_in // n_llm, n_out // n_llm,
                                      cost / n_llm, (m.get("generation_time_seconds") or 0.0) * 1000 / n_llm,
                                      priced=price is not None or bool(m.get("cost"))))
        for _ in range(int(jev.get("s1_calls", 0))):
            ledger.add(CallRecord("s1", "unknown", None, None, cost_usd=s1_cost / jev["s1_calls"],
                                  latency_ms=float(jev.get("s1_latency_ms", 0.0)) / jev["s1_calls"]))
        tcs = m.get("tool_calls") or []
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        steps.append(StepLog(
            step=step, tool=tcs[0]["name"] if tcs else RESPOND,
            args=dict(tcs[0]["arguments"]) if tcs else {"message": _trunc(m.get("content"), 500)},
            observation=_trunc(str((nxt or {}).get("content") or ""), 2000), model=model, llm_calls=n_llm,
            decisions=[DecisionLog(d["dp"], d["mode"], "", s1_value=d.get("s1"), s1_conf=d.get("conf"),
                                   deferred=bool(d.get("deferred")), s2_value=d.get("s2"),
                                   final_value=d.get("final")) for d in jev.get("decisions", [])]))
    return steps, ledger


_ERROR_REASONS = {"agent_error", "user_error", "infrastructure_error"}


def convert_results(results: str | Path, decisions: str | Path | None = None, arm: str | None = None,
                    out: str | Path | None = None) -> list[Trajectory]:
    """tau2 results (+ side-car) -> jevharness Trajectories; success = (reward == 1).

    ``decisions`` defaults to ``<results dir>/jev_decisions.jsonl``. Side-car records are joined by the
    episode id embedded in each assistant message (``raw_data.jev.episode``), falling back to
    (task_id, seed); without a record, steps/usage are rebuilt from the tau2 messages.
    """
    data = _load_results(results)
    info = data.get("info") or {}
    domain = (info.get("environment_info") or {}).get("domain_name", "tau2")
    agent_info = info.get("agent_info") or {}
    llm = agent_info.get("llm")
    res_dir = Path(results) if Path(results).is_dir() else Path(results).parent
    if decisions is None:
        decisions = res_dir / SIDECAR_FILE
    # A6.22 (e) 4: with a re-run ledger (written by `rerun-errored`) every simulation carries its attempt count
    attempts_ledger = load_attempts(res_dir) if (res_dir / ATTEMPTS_FILE).exists() else None
    recs = load_sidecar(decisions)
    by_ep = {r["episode"]: r for r in recs}
    by_key = {(str(r.get("task_id")), r.get("seed")): r for r in recs}  # later records win (retries)
    trajs: list[Trajectory] = []
    for sim in data.get("simulations") or []:
        rec = by_ep.get(_episode_of(sim) or "") or by_key.get((str(sim.get("task_id")), sim.get("seed")))
        reward_info = sim.get("reward_info") or {}
        reward = float(reward_info.get("reward") or 0.0)
        term = str(sim.get("termination_reason"))
        name = arm or (rec or {}).get("config") or f"{agent_info.get('implementation', 'agent')}"
        traj = Trajectory(task_id=f"{domain}/{sim.get('task_id')}#{sim.get('trial', 0)}", config=name,
                          backend=(rec or {}).get("backend"), llm=llm)
        if rec is not None:
            traj.steps = [
                _dc(StepLog, {**s, "decisions": [_dc(DecisionLog, d) for d in s.get("decisions", [])]})
                for s in rec.get("steps", [])]
            traj.usage, traj.calls = rec.get("usage") or Ledger().summary(), rec.get("calls") or []
            agent_err = rec.get("error")
        else:
            traj.steps, ledger = _fallback_from_messages(sim, llm)
            traj.usage, traj.calls = ledger.summary(), ledger.to_list()
            agent_err = None
        texts = [m.get("content") for m in sim.get("messages") or [] if _msg_role(m) == "assistant"
                 and m.get("content")]
        traj.final_answer = texts[-1] if texts else None
        traj.result = {"success": abs(reward - 1.0) <= 1e-6, "score": reward, "reward": reward,
                       "termination_reason": term,
                       "db_match": (reward_info.get("db_check") or {}).get("db_match"),
                       "reward_breakdown": reward_info.get("reward_breakdown")}
        traj.wall_ms = float(sim.get("duration") or 0.0) * 1000
        if term in _ERROR_REASONS or agent_err:
            traj.error = agent_err or f"tau2 termination_reason={term}"
        traj.meta = {"tau2": {"domain": domain, "task_id": sim.get("task_id"), "trial": sim.get("trial"),
                              "seed": sim.get("seed"), "simulation_id": sim.get("id"),
                              "agent_cost": sim.get("agent_cost"), "user_cost": sim.get("user_cost"),
                              "agent_implementation": agent_info.get("implementation"),
                              "user_llm": (info.get("user_info") or {}).get("llm"),
                              "sidecar": rec is not None, "episode": (rec or {}).get("episode"),
                              "split": split_of(domain, sim.get("task_id"))},  # A5.4: shadow | test | None
                     "config": (rec or {}).get("config_dump")}
        if attempts_ledger is not None:
            prev = (attempts_ledger["sims"].get(_sim_key(sim)) or {}).get("pruned") or []
            traj.meta["tau2"]["attempts"] = 1 + len(prev)
            if prev:
                traj.meta["tau2"]["errored_attempts"] = [{k: p.get(k) for k in ("termination_reason", "error")}
                                                         for p in prev]
        if rec is not None and "shadow_s1_errors" in rec:  # shadow arms: failed (fail-open) S1 measurements
            traj.meta["shadow_s1_errors"] = list(rec.get("shadow_s1_errors") or [])
        trajs.append(traj)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for t in trajs:
                f.write(json.dumps(t.to_dict(), default=str) + "\n")
    return trajs


# ================================================================== errored-simulation re-runs (A6.22 (e) 4)
def _sim_key(sim: dict[str, Any]) -> str:
    """``<task_id>#<trial>``: the unit a re-run repeats (tau2 seeds by trial, so the seed is the same too)."""
    return f"{sim.get('task_id')}#{sim.get('trial', 0)}"


def load_attempts(res_dir: str | Path) -> dict[str, Any]:
    """The re-run ledger of a tau2 save directory: ``{"max_attempts", "sims": {<task_id>#<trial>: {"pruned": [the
    errored executions removed for a re-run]}}}``; the attempts made for a simulation = 1 + len(pruned)."""
    p = Path(res_dir) / ATTEMPTS_FILE
    led = json.loads(p.read_text()) if p.exists() else {}
    return {"max_attempts": led.get("max_attempts", MAX_SIM_ATTEMPTS), "sims": dict(led.get("sims") or {})}


def errored_simulations(results: str | Path, decisions: str | Path | None = None) -> list[dict[str, Any]]:
    """The simulations of a tau2 save directory (or results.json) that ended in an infrastructure error in the sense
    of PREREGISTRATION A6.22 (e) 1 -- termination ``agent_error`` / ``user_error`` / ``infrastructure_error``, or an
    exception in ``JevTau2Agent`` (side-car ``error``) -- with their attempts so far (from the re-run ledger). The
    side-car is joined as :func:`convert_results` joins it."""
    res_dir = Path(results) if Path(results).is_dir() else Path(results).parent
    data = _load_results(results)
    recs = load_sidecar(decisions if decisions is not None else res_dir / SIDECAR_FILE)
    by_ep = {r["episode"]: r for r in recs}
    by_key = {(str(r.get("task_id")), r.get("seed")): r for r in recs}  # later records win (re-runs)
    ledger = load_attempts(res_dir)
    out = []
    for sim in data.get("simulations") or []:
        rec = by_ep.get(_episode_of(sim) or "") or by_key.get((str(sim.get("task_id")), sim.get("seed")))
        term, err = str(sim.get("termination_reason")), (rec or {}).get("error")
        if term in _ERROR_REASONS or err:
            key = _sim_key(sim)
            out.append({"key": key, "id": sim.get("id"), "task_id": str(sim.get("task_id")),
                        "trial": sim.get("trial", 0), "seed": sim.get("seed"), "termination_reason": term,
                        "error": err, "attempts": 1 + len((ledger["sims"].get(key) or {}).get("pruned") or [])})
    return out


def _remove_simulations(res_dir: Path, ids: set[str]) -> None:
    """Take simulations out of a tau2 save directory so that ``tau2 run --auto-resume`` runs their (trial, task,
    seed) units again; each removed simulation is kept in ``<save>/jev_errored/<id>.json``. Handles tau2's two
    storage formats: the monolithic results.json and the directory format (simulations/<id>.json + an index in
    results.json, which tau2 checks against the files on load)."""
    meta_path, sims_dir, keep = res_dir / "results.json", res_dir / "simulations", res_dir / ERRORED_DIR
    keep.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_path.read_text())
    if sims_dir.is_dir():
        for i in ids:
            f = sims_dir / f"{i}.json"
            if f.exists():
                os.replace(f, keep / f.name)
        if meta.get("simulation_index") is not None:
            meta["simulation_index"] = [e for e in meta["simulation_index"] if e.get("id") not in ids]
    else:
        for sim in meta.get("simulations") or []:
            if sim.get("id") in ids:
                (keep / f"{sim['id']}.json").write_text(json.dumps(sim))
        meta["simulations"] = [x for x in meta.get("simulations") or [] if x.get("id") not in ids]
    tmp = meta_path.with_name(f".{meta_path.name}.jev-tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, meta_path)


def prune_errored(results: str | Path, decisions: str | Path | None = None,
                  max_attempts: int = MAX_SIM_ATTEMPTS) -> dict[str, Any]:
    """One re-run round of A6.22 (e) 4: remove every errored simulation with attempts left (fewer than
    ``max_attempts`` executions) from the save directory and record it in the re-run ledger (``jev_attempts.json``,
    always written, so the converter knows the attempt counts are logged). Never call it while tau2 writes to the
    directory. Returns the pruned simulations and the ``final`` ones (errored after ``max_attempts`` executions)."""
    res_dir = Path(results) if Path(results).is_dir() else Path(results).parent
    ledger = load_attempts(res_dir)
    errs = errored_simulations(results, decisions)
    prune = [e for e in errs if e["attempts"] < max_attempts]
    final = [e for e in errs if e["attempts"] >= max_attempts]
    if prune:
        _remove_simulations(res_dir, {str(e["id"]) for e in prune})
        for e in prune:
            ledger["sims"].setdefault(e["key"], {"pruned": []})["pruned"].append(
                {k: e[k] for k in ("id", "seed", "termination_reason", "error")}
                | {"time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    ledger["max_attempts"] = max_attempts
    (res_dir / ATTEMPTS_FILE).write_text(json.dumps(ledger, indent=1))
    return {"pruned": prune, "final": final, "ledger": str(res_dir / ATTEMPTS_FILE)}


def rerun_errored(save_dir: str | Path, run_argv: list[str], max_attempts: int = MAX_SIM_ATTEMPTS,
                  decisions: str | Path | None = None,
                  runner: Callable[[list[str]], int] | None = None) -> dict[str, Any]:
    """PREREGISTRATION A6.22 (e) 4 (A6.20 (f)): re-run every errored simulation of a finished ``run`` -- same task,
    same trial, hence same seed -- up to ``max_attempts - 1`` (= 2) times before the analysis. Each round prunes the
    errored simulations with attempts left (:func:`prune_errored`) and repeats ``run`` with ``run_argv`` (its
    ``--auto-resume`` re-runs exactly the missing (trial, task, seed) units); after the last round nothing is pruned,
    so a simulation still errored stays in the results and its task leaves the pair with all its trials
    (``scripts/primaries.py``). ``runner`` executes one ``run`` (default: a subprocess of this module)."""
    def default_runner(argv: list[str]) -> int:
        import subprocess

        return subprocess.call([sys.executable, "-m", "jevharness.bench.suites.tau2", "run", *argv])

    runner = runner or default_runner
    rounds: list[dict[str, Any]] = []
    for _ in range(max(0, max_attempts - 1)):
        pr = prune_errored(save_dir, decisions, max_attempts)
        rounds.append({"pruned": [e["key"] for e in pr["pruned"]]})
        if not pr["pruned"]:
            break
        rc = runner(list(run_argv))
        rounds[-1]["run_status"] = rc
        if rc:
            break
    if max_attempts <= 1:   # nothing may be re-run: still record the (trivial) attempt counts
        prune_errored(save_dir, decisions, max_attempts)
    left = errored_simulations(save_dir, decisions)
    return {"save_dir": str(save_dir), "max_attempts": max_attempts, "rounds": rounds,
            "still_errored": [{k: e[k] for k in ("key", "termination_reason", "error", "attempts")} for e in left],
            "retries_left": sum(1 for e in left if e["attempts"] < max_attempts)}


def pass_hat_k(trajs: list[Trajectory] | list[dict[str, Any]]) -> dict[str, float]:
    """tau-bench pass^k (Yao et al. 2024): mean over tasks of C(successes, k) / C(trials, k)."""
    per_task: dict[str, list[bool]] = defaultdict(list)
    for t in trajs:
        d = t if isinstance(t, dict) else t.to_dict()
        tau = (d.get("meta") or {}).get("tau2") or {}
        per_task[f"{tau.get('domain')}/{tau.get('task_id')}"].append(bool(d["result"].get("success")))
    if not per_task:
        return {}
    k_max = min(len(v) for v in per_task.values())
    return {f"pass^{k}": sum(math.comb(sum(v), k) / math.comb(len(v), k) for v in per_task.values())
            / len(per_task) for k in range(1, k_max + 1)}


def summarize(trajs: list[Trajectory]) -> dict[str, Any]:
    n = len(trajs)
    if n == 0:
        return {"n": 0}
    tau_meta = [t.meta.get("tau2", {}) for t in trajs]
    return {
        "n": n,
        "avg_reward": sum(t.result.get("score", 0.0) for t in trajs) / n,
        **pass_hat_k(trajs),
        "cost_usd_per_sim": sum(t.usage.get("total", {}).get("cost_usd", 0.0) for t in trajs) / n,
        "s1_cost_usd_per_sim": sum(t.usage.get("s1", {}).get("cost_usd", 0.0) for t in trajs) / n,
        "llm_calls_per_sim": sum(t.usage.get("s2", {}).get("calls", 0) for t in trajs) / n,
        "tau2_agent_cost_per_sim": sum(float(m.get("agent_cost") or 0.0) for m in tau_meta) / n,
        "user_cost_per_sim": sum(float(m.get("user_cost") or 0.0) for m in tau_meta) / n,
        "with_sidecar": sum(bool(m.get("sidecar")) for m in tau_meta),
        "errors": sum(1 for t in trajs if t.error),
        "shadow_cost_usd_per_sim": sum(t.usage.get("shadow", {}).get("cost_usd", 0.0) for t in trajs) / n,
        "shadow_s1_errors": sum(len(t.meta.get("shadow_s1_errors") or []) for t in trajs),
    }


def summarize_by_split(trajs: list[Trajectory]) -> dict[str, dict[str, Any]]:
    """``summarize`` per A5.4 split: ``test`` (the C7 comparison set) and ``shadow`` (reported descriptively);
    only the splits that occur. Trajectories of domains without a split are test tasks."""
    labels = {_traj_split(t) for t in trajs} - {None}
    out: dict[str, dict[str, Any]] = {}
    if labels:
        out["test"] = summarize(filter_split(trajs, "test"))
    if "shadow" in labels:
        out["shadow"] = summarize(filter_split(trajs, "shadow"))
    return out


# ================================================================== CLI
def _save_dir_from_args(args: list[str]) -> Path | None:
    for i, a in enumerate(args):
        val = a.split("=", 1)[1] if a.startswith("--save-to=") else (args[i + 1] if a == "--save-to"
                                                                      and i + 1 < len(args) else None)
        if val:
            from tau2.utils import DATA_DIR  # tau2 saves to DATA_DIR/simulations/<save_to> (abs paths win)

            return Path(DATA_DIR) / "simulations" / val
    return None


def _nl_judge_default_args(model: str) -> dict[str, Any]:
    """litellm args for a non-default NL-assertion judge. tau2 ``json.loads`` the judge's raw text, so ask for
    a JSON response (Gemini otherwise tends to wrap it in a Markdown fence); minimal thinking on Gemini 3.x."""
    args: dict[str, Any] = {"temperature": 0.0, "response_format": {"type": "json_object"}}
    if "gemini-3" in model:
        args["thinkingConfig"] = {"thinkingLevel": "minimal"}
    return args


def set_nl_judge(model: str | None = None, llm_args: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Point tau2's NL-assertion judge at ``model`` (a litellm name). tau2 hard-codes the judge to
    ``DEFAULT_LLM_NL_ASSERTIONS`` (``gpt-4.1-2025-04-14``, ``tau2/config.py``); 112 of the 114 retail tasks have
    ``NL_ASSERTION`` in their reward basis, so without OpenAI access every retail simulation whose task lists an
    assertion fails evaluation. The evaluator module reads the name at call time, so patching its globals in the
    launcher process (``--max-concurrency`` threads share it) is enough. Keep the judge FIXED across arms and
    report it: a non-default judge is a deviation from the official tau2 protocol. ``model=None`` only reads."""
    from tau2.evaluator import evaluator_nl_assertions as nl

    if model:
        nl.DEFAULT_LLM_NL_ASSERTIONS = model
        nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(llm_args if llm_args is not None else _nl_judge_default_args(model))
    return nl.DEFAULT_LLM_NL_ASSERTIONS, dict(nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS)


def nl_judge() -> dict[str, Any] | None:
    """The NL-assertion judge in effect (recorded in every side-car record), or None without tau2."""
    try:
        model, args = set_nl_judge()
    except Exception:  # noqa: BLE001 - tau2 missing / layout changed: logging only
        return None
    return {"model": model, "args": args}


def _cmd_run(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="python -m jevharness.bench.suites.tau2 run", add_help=False,
                                description="Register jev_agent, then run `tau2 run` with the remaining "
                                            "args.")
    p.add_argument("--jev-config", required=True, help="harness YAML (configs/tau2/*.yaml)")
    p.add_argument("--jev-set", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted override, e.g. tool_select.tau=0.9 (repeatable)")
    p.add_argument("--jev-name", help="arm name recorded in the logs (default: config name + overrides)")
    p.add_argument("--jev-log-dir", help="side-car dir (default: the tau2 --save-to dir)")
    p.add_argument("--jev-nl-judge", default=os.environ.get(ENV_NL_JUDGE) or None, metavar="LITELLM_MODEL",
                   help=f"NL-assertion judge (retail) instead of tau2's fixed gpt-4.1-2025-04-14; env ${ENV_NL_JUDGE}")
    p.add_argument("--jev-nl-judge-args", default=os.environ.get(ENV_NL_JUDGE_ARGS) or None, metavar="JSON",
                   help=f"litellm args for --jev-nl-judge (default: temperature 0, JSON response); "
                        f"env ${ENV_NL_JUDGE_ARGS}")
    p.add_argument("--jev-help", action="help", help="show this help")
    ours, rest = p.parse_known_args(argv)
    if ours.jev_nl_judge:
        judge_args = json.loads(ours.jev_nl_judge_args) if ours.jev_nl_judge_args else None
        model, judge_args = set_nl_judge(ours.jev_nl_judge, judge_args)
        print(f"[jev] NL-assertion judge: {model} {json.dumps(judge_args)} (tau2 default: gpt-4.1-2025-04-14)",
              file=sys.stderr)
    overrides = parse_overrides(ours.jev_set)
    if ours.jev_name:
        overrides["name"] = ours.jev_name
    cfg = Tau2JevConfig.load(ours.jev_config, overrides)  # fail fast on a bad config
    for i, a in enumerate(rest):
        w = a.split("=", 1)[1] if a.startswith("--workers=") else (rest[i + 1] if a == "--workers"
                                                                  and i + 1 < len(rest) else None)
        if w is not None and int(w) > 0:
            raise SystemExit("--workers spawns fresh `tau2.runner.worker` processes that do not register "
                             "jev_agent; use --max-concurrency (threads) instead")
    if not any(a == "--agent" or a.startswith("--agent=") for a in rest):
        rest += ["--agent", AGENT_NAME]
    os.environ[ENV_CONFIG] = str(Path(ours.jev_config).resolve())
    os.environ[ENV_OVERRIDES] = json.dumps(overrides)
    log_dir = ours.jev_log_dir or cfg.log_dir or _save_dir_from_args(rest)
    if log_dir:
        os.environ[ENV_LOG_DIR] = str(log_dir)
    register()
    pg = f", policy_guard(tau={cfg.policy_guard.tau})" if cfg.policy_guard.mode == Mode.GUARD else ""
    print(f"[jev] arm={cfg.name} ({cfg.harness.short()}{pg}) s1={cfg.s1} "
          f"side-car={Path(log_dir) / SIDECAR_FILE if log_dir else None}", file=sys.stderr)
    for warning in _litellm_arg_warnings(rest):
        print(f"[jev] WARNING: {warning}", file=sys.stderr)
    from tau2.cli import main as tau2_main

    sys.argv = ["tau2", "run", *rest]
    tau2_main()


def _flag_value(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return None


def _litellm_arg_warnings(args: list[str]) -> list[str]:
    """Known litellm pitfalls in `tau2 run` LLM args. litellm 1.82.x (the newest tau2 allows) maps
    ``reasoning_effort`` for Gemini 3.x through ``VertexGeminiConfig._map_reasoning_effort_to_thinking_level``,
    which recognises only ``gemini-3-flash`` / ``gemini-3.1-flash`` as Flash models: for ``gemini-3.5-flash``
    (and -lite) ``minimal`` silently becomes ``thinkingLevel: low`` (~100 extra reasoning tokens per call).
    Passing ``{"thinkingConfig": {"thinkingLevel": "minimal"}}`` goes through litellm unchanged."""
    out = []
    for who in ("agent", "user"):
        model = _flag_value(args, f"--{who}-llm") or ""
        raw = _flag_value(args, f"--{who}-llm-args")
        try:
            llm_args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            continue
        effort = llm_args.get("reasoning_effort") if isinstance(llm_args, dict) else None
        if not isinstance(effort, str) or "gemini-3" not in model:
            continue
        try:
            from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

            level = VertexGeminiConfig._map_reasoning_effort_to_thinking_level(effort, model).get("thinkingLevel")
        except Exception:  # noqa: BLE001 - private litellm API; the warning is best-effort
            continue
        if level and level != effort:
            out.append(f"litellm maps --{who}-llm-args reasoning_effort={effort!r} to thinkingLevel={level!r} "
                       f"for {model}; pass {{\"thinkingConfig\": {{\"thinkingLevel\": \"{effort}\"}}}} instead")
    return out


# ================================================================== multi-arm experiments (pilots)
def experiment_commands(exp: dict[str, Any], python: str = sys.executable) -> list[tuple[str, list[str], Path]]:
    """``[(arm, argv of `run`, save dir)]`` for a multi-arm experiment YAML (e.g. configs/pilot/tau2_*.yaml).

    Keys: ``name``, ``out`` (default ``runs/tau2/<name>``), ``domain``, ``agent_llm`` / ``agent_llm_args``,
    ``user_llm`` / ``user_llm_args``, ``num_trials``, ``num_tasks`` or ``task_ids`` or ``split`` (``shadow`` |
    ``test``: the A5.4 airline split, resolved to ``--task-ids``; ``test`` on another domain = every task), ``task_split``, ``seed``,
    ``max_concurrency``, ``tau2_args`` (extra `tau2 run` flags), ``nl_judge_llm`` / ``nl_judge_llm_args``
    (retail NL-assertion judge instead of tau2's gpt-4.1, see :func:`set_nl_judge`), ``s1`` / ``s1_cache`` (applied to every arm
    whose config uses System-1, so switching the S1 backend is a one-line change), ``arm_defaults`` (dotted
    overrides for every arm) and ``arms``: ``[{name, config: configs/tau2/<arm>.yaml, set: {dotted: value}}]``.
    Every arm gets the same trial seeds, so arms are paired by (task, trial)."""
    out = Path(exp.get("out") or f"runs/tau2/{exp.get('name', 'tau2')}").resolve()
    cmds = []
    for arm in exp["arms"]:
        name = arm["name"]
        save = out / "raw" / name
        overrides = {**(exp.get("arm_defaults") or {}), **(arm.get("set") or {})}
        probe = {**overrides, "s1": overrides.get("s1") or exp.get("s1") or {"type": "oracle"}}
        if Tau2JevConfig.load(arm["config"], probe).uses_s1():  # fails fast on a bad arm config
            for key in ("s1", "s1_cache"):
                if exp.get(key) is not None:
                    overrides.setdefault(key, exp[key])
        jev = ["--jev-config", str(arm["config"]), "--jev-name", name]
        for k, v in overrides.items():
            jev += ["--jev-set", f"{k}={json.dumps(v)}"]
        tau = ["--domain", exp["domain"], "--agent-llm", exp["agent_llm"], "--user-llm", exp["user_llm"],
               "--num-trials", str(exp.get("num_trials", 1)), "--max-concurrency", str(exp.get("max_concurrency", 4)),
               "--seed", str(exp.get("seed", 300)), "--task-split-name", str(exp.get("task_split", "base")),
               "--save-to", str(save), "--auto-resume"]
        for who in ("agent", "user"):
            if exp.get(f"{who}_llm_args") is not None:
                tau += [f"--{who}-llm-args", json.dumps(exp[f"{who}_llm_args"])]
        task_ids = [str(t) for t in exp.get("task_ids") or []]
        split = exp.get("split")
        if split and split != "all" and not task_ids:
            if split not in SPLITS:
                raise ValueError(f"split must be one of {SPLITS}")
            if exp["domain"] == "airline":
                task_ids = list(airline_task_split()[0 if split == "shadow" else 1])
            elif split == "shadow":
                raise ValueError(f"no shadow split for domain {exp['domain']!r} (only airline is split, A5.4 c)")
        if task_ids:
            tau += ["--task-ids", *task_ids]
        if exp.get("num_tasks"):
            tau += ["--num-tasks", str(exp["num_tasks"])]
        tau += [str(x) for x in exp.get("tau2_args") or []]
        if exp.get("nl_judge_llm"):  # fixed across arms (retail NL assertions; see set_nl_judge)
            jev += ["--jev-nl-judge", str(exp["nl_judge_llm"])]
            if exp.get("nl_judge_llm_args") is not None:
                jev += ["--jev-nl-judge-args", json.dumps(exp["nl_judge_llm_args"])]
        cmds.append((name, [python, "-m", "jevharness.bench.suites.tau2", "run", *jev, *tau], save))
    return cmds


def _cmd_experiment(argv: list[str]) -> None:
    import shlex
    import subprocess

    p = argparse.ArgumentParser(prog="python -m jevharness.bench.suites.tau2 experiment",
                                description="Run every arm of a multi-arm tau2 experiment YAML (one `run` "
                                            "subprocess per arm), convert each to Trajectory JSONL "
                                            "(<out>/<arm>.jsonl) and print `jevbench report <out>`.")
    p.add_argument("config", help="experiment YAML, e.g. configs/pilot/tau2_airline_gemini.yaml")
    p.add_argument("--arms", help="comma-separated subset of arm names")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted override of the experiment YAML, e.g. num_tasks=10 (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    p.add_argument("--no-report", action="store_true")
    a = p.parse_args(argv)
    exp = yaml.safe_load(Path(a.config).read_text()) or {}
    for key, val in parse_overrides(a.override).items():
        node = exp
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    if a.arms:
        keep = set(a.arms.split(","))
        exp["arms"] = [arm for arm in exp["arms"] if arm["name"] in keep]
    out = Path(exp.get("out") or f"runs/tau2/{exp.get('name', 'tau2')}").resolve()
    summaries: dict[str, Any] = {}
    failed = []
    for name, cmd, save in experiment_commands(exp):
        print("+ " + shlex.join(cmd), file=sys.stderr)
        if a.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc:
            failed.append(name)
            print(f"[jev] arm {name}: `run` exited with status {rc}", file=sys.stderr)
        if (save / "results.json").exists():
            trajs = convert_results(save / "results.json", arm=name, out=out / f"{name}.jsonl")
            summaries[name] = summarize(trajs)
    if a.dry_run:
        return
    print(json.dumps({"out": str(out), "failed": failed, "arms": summaries}, indent=1))
    if not a.no_report and summaries:
        from ..report import report

        print(report(out))


def _cmd_convert(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="python -m jevharness.bench.suites.tau2 convert")
    p.add_argument("results", help="tau2 results.json (or its directory)")
    p.add_argument("--decisions", help=f"side-car JSONL (default: <results dir>/{SIDECAR_FILE})")
    p.add_argument("--arm", help="config name for the trajectories (default: from the side-car)")
    p.add_argument("--out", required=True, help="output Trajectory JSONL, e.g. runs/tau2/airline/<arm>.jsonl")
    a = p.parse_args(argv)
    trajs = convert_results(a.results, a.decisions, a.arm, a.out)
    doc: dict[str, Any] = {"out": a.out, **summarize(trajs)}
    by_split = summarize_by_split(trajs)
    if by_split:  # airline: the 35 test tasks (C7 comparisons) and the 15 shadow tasks (descriptive), A5.4
        doc["by_split"] = by_split
    print(json.dumps(doc, indent=1))


def _cmd_rerun_errored(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="python -m jevharness.bench.suites.tau2 rerun-errored",
                                description="PREREGISTRATION A6.22 (e) 4: re-run every errored simulation of a "
                                            "finished `run` (same task, trial and seed) up to twice: prune it from "
                                            "the save directory, record it in jev_attempts.json, repeat `run` with "
                                            "the arguments after `--` (they must include --auto-resume and the same "
                                            "--save-to). `convert` then stamps meta.tau2.attempts.")
    p.add_argument("save_dir", help="the tau2 --save-to directory of the run")
    p.add_argument("--max-attempts", type=int, default=MAX_SIM_ATTEMPTS,
                   help=f"executions per simulation, the first included (default {MAX_SIM_ATTEMPTS})")
    p.add_argument("--decisions", help=f"side-car JSONL (default: <save_dir>/{SIDECAR_FILE})")
    p.add_argument("--dry-run", action="store_true", help="only list the errored simulations")
    if "--" in argv:
        k = argv.index("--")
        argv, run_argv = argv[:k], argv[k + 1:]
    else:
        run_argv = []
    a = p.parse_args(argv)
    if a.dry_run:
        print(json.dumps(errored_simulations(a.save_dir, a.decisions), indent=1))
        return
    if not run_argv:
        raise SystemExit("rerun-errored needs the `run` arguments after `--`")
    if "--auto-resume" not in run_argv:
        raise SystemExit("the `run` arguments must include --auto-resume (tau2 then re-runs only the pruned units)")
    if not (Path(a.save_dir) / "results.json").exists() and not Path(a.save_dir).joinpath("simulations").is_dir():
        raise SystemExit(f"no tau2 results in {a.save_dir}")
    res = rerun_errored(a.save_dir, run_argv, a.max_attempts, a.decisions)
    print(json.dumps(res, indent=1), file=sys.stderr)


def _cmd_split(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="python -m jevharness.bench.suites.tau2 split",
                                description="PREREGISTRATION A5.4 (c): the seeded 15/35 shadow/test split of the 50 "
                                            "airline tasks (seed 0 over the task ids in sorted order).")
    p.add_argument("domain", nargs="?", default="airline", help="only `airline` has a shadow split")
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--n-shadow", type=int, default=SPLIT_N_SHADOW)
    p.add_argument("--which", choices=SPLITS, default="all", help="which list(s) to print")
    p.add_argument("--format", choices=["json", "lines", "args"], default="json",
                   help="args: one space-separated line (for `tau2 run --task-ids`); needs --which shadow|test or "
                        "--ids-file")
    p.add_argument("--ids-file", help="print the task ids of this file instead (JSON list / object with task_ids "
                                      "/ text); scripts/run_tau2.sh TASK_IDS=<file>")
    p.add_argument("--no-check", action="store_true", help="do not cross-check the installed tau2 task list")
    p.add_argument("--write", action="store_true",
                   help=f"(maintenance) rewrite the `split` block of {AIRLINE_TASK_IDS_FILE.name} from --seed/--n-shadow")
    a = p.parse_args(argv)
    if a.ids_file:
        ids = read_task_ids(a.ids_file)
        print(" ".join(ids) if a.format == "args" else json.dumps(ids) if a.format == "json" else "\n".join(ids))
        return
    if a.domain != "airline":
        raise SystemExit(f"no shadow split for domain {a.domain!r}: only airline is split (A5.4 c); every "
                         f"{a.domain} task is a test task")
    ids = airline_task_ids(check_installed=not a.no_check)
    if a.seed == SPLIT_SEED and a.n_shadow == SPLIT_N_SHADOW and not a.write:
        shadow, test = airline_task_split()  # the committed record (asserted against airline_split)
    else:
        shadow, test = airline_split(ids, a.seed, a.n_shadow)
    if a.write:
        rec = dict(_airline_record())
        rec["split"] = {"seed": a.seed, "n_shadow": a.n_shadow,
                        "order": "task ids sorted (numeric ids ascending, then non-numeric lexicographic)",
                        "method": "random.Random(seed).shuffle over the sorted ids; the first n_shadow are the "
                                  "shadow split, the rest the test set (jevharness.bench.suites.tau2.airline_split)",
                        "shadow": shadow, "test": test}
        AIRLINE_TASK_IDS_FILE.write_text(json.dumps(rec, indent=1) + "\n")
        _airline_record.cache_clear()
        print(f"[jev] wrote {AIRLINE_TASK_IDS_FILE}", file=sys.stderr)
    lists = {"shadow": shadow, "test": test}
    if a.format == "args":
        if a.which == "all":
            raise SystemExit("--format args needs --which shadow|test")
        print(" ".join(lists[a.which]))
    elif a.format == "lines":
        for k in (("shadow", "test") if a.which == "all" else (a.which,)):
            print(f"{k}: " + " ".join(lists[k]))
    else:
        doc = {"domain": "airline", "seed": a.seed, "n_shadow": a.n_shadow, "n": len(ids)}
        doc.update({k: v for k, v in lists.items() if a.which in ("all", k)})
        print(json.dumps(doc, indent=1))


def _cmd_report(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="python -m jevharness.bench.suites.tau2 report",
                                description="Per-split summary of a converted tau2 domain directory "
                                            "(<out>/<domain>/<arm>.jsonl). --split test writes the 35-task airline "
                                            "test set (plus every task of unsplit domains) to <domain>_test/ and "
                                            "prints `jevbench report` on it: the C7 comparison (A5.4 c); --split "
                                            "shadow does the same for the 15 shadow tasks (descriptive only).")
    p.add_argument("run_dir", help="converted domain directory, e.g. runs/tau2/airline")
    p.add_argument("--split", choices=SPLITS, default="all")
    p.add_argument("--test-only", action="store_true", help="alias for --split test")
    p.add_argument("--out", help="where the filtered JSONL go (default: <run_dir>_<split>)")
    p.add_argument("--no-report", action="store_true", help="print the per-split summaries only")
    a = p.parse_args(argv)
    split = "test" if a.test_only else a.split
    run_dir = Path(a.run_dir)
    files = sorted(run_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no <arm>.jsonl in {run_dir}")
    arms = {f.stem: load_trajectories(f) for f in files}
    doc: dict[str, Any] = {"run_dir": str(run_dir), "split": split, "arms": {}}
    for arm, recs in arms.items():
        trajs = [_dc(Trajectory, {**r, "steps": [_dc(StepLog, {**s, "decisions": [_dc(DecisionLog, d) for d in
                                                                          s.get("decisions", [])]})
                                                 for s in r.get("steps", [])]}) for r in recs]
        row = {"all": summarize(trajs), **summarize_by_split(trajs)}
        doc["arms"][arm] = row
    print(json.dumps(doc, indent=1, default=str))
    if split == "all":
        target = run_dir
    else:
        target = Path(a.out) if a.out else run_dir.parent / f"{run_dir.name}_{split}"
        target.mkdir(parents=True, exist_ok=True)
        for arm, recs in arms.items():
            keep = filter_split(recs, split)
            with open(target / f"{arm}.jsonl", "w") as f:
                for r in keep:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"[jev] {arm}: {len(keep)}/{len(recs)} episodes in split {split!r} -> {target / (arm + '.jsonl')}",
                  file=sys.stderr)
    if not a.no_report:
        from ..report import report

        print(report(target))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:  # API keys from .env (never overrides variables that are already set), as the other CLIs do
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    cmds = {"run": _cmd_run, "convert": _cmd_convert, "experiment": _cmd_experiment, "split": _cmd_split,
            "report": _cmd_report, "rerun-errored": _cmd_rerun_errored}
    if argv and argv[0] == "--split":  # `--split airline` == `split airline`
        argv = ["split", *argv[1:]]
    if not argv or argv[0] not in cmds:
        print(__doc__)
        print("usage: python -m jevharness.bench.suites.tau2 {run,rerun-errored,convert,experiment,split,report} ...  "
              "(run --jev-help / rerun-errored -h / convert -h / experiment -h / split -h / report -h)")
        raise SystemExit(2)
    cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
