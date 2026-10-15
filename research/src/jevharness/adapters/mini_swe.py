"""mini-swe-agent adapter: System-1 (Jev / OpenJev / small-LM) decision points inside a real coding
harness, for SWE-bench Verified / Lite.

Coded against ``mini-swe-agent`` 2.4.6 (``pip install mini-swe-agent``). ``JevMiniSweAgent`` subclasses
``minisweagent.agents.default.DefaultAgent`` and keeps its control flow (query -> parse bash actions ->
``env.execute`` -> observation messages; the episode ends when a command prints
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` and the environment raises ``Submitted``). Five hooks map
onto the paper's WHERE x HOW design space (``jevharness.harness.config``); ``DecisionLog.dp`` uses the
same decision-point names as the reference harness so ``jevharness.bench.metrics`` works unchanged:

  hook          DecisionLog.dp  modes                           what it does
  ------------  --------------  ------------------------------  ---------------------------------------------
  action_guard  tool_select     off | shadow | guard            before each bash command: Noul (or Choice
                                                                {on_track, unproductive, destructive}) "is this
                                                                command reasonable / on-track / non-destructive?"
                                                                guard: p_ok < tau -> not executed, a "blocked,
                                                                reconsider" observation is returned instead
  stop          stop            off | shadow | guard | trigger  guard: when the agent issues the submit command,
                                                                Noul "has the fix been verified?"; p < tau ->
                                                                submission deferred with a nudge (<= max_nudges)
                                                                trigger: per step, "already fixed + verified?"
                                                                p >= tau -> hint "submit now" (early stop)
  routing       routing         off | shadow | trigger          per step, Noul p(hard) >= tau -> strong model,
                                                                else the cheap model (two mini-swe Model objects)
  reflection    reflection      off | shadow | always | trigger after a failed / repeated command: trigger ->
                                                                S1 decides whether to inject a reflection prompt;
                                                                always -> rule-based reference (no S1)
  action_type   action          off | shadow | assist           Choice over {explore, search, reproduce, edit,
                                                                test, submit}; assist -> top-k injected as a hint

``shadow`` queries S1 and logs its decision without acting on it (collects calibration / training data on
unmodified baseline trajectories). All pre-query questions of one step (routing, reflection, stop-trigger,
action-type) share one S1 state and are sent in one batched ``decide`` call (``batch_s1``); the guard
questions for the (possibly parallel) commands of one LM turn are batched the same way.

Everything the S1 does is recorded as ``DecisionLog`` rows (per step) and ``CallRecord`` rows in a
``jevharness.accounting.Ledger`` (S1 calls, plus one ``component="s2"`` record per LM call carrying
mini-swe-agent's litellm cost and token usage). Per instance we write ``<iid>.jev.json``;
:func:`to_trajectory` converts it (or a live agent) into a ``jevharness.harness.trajectory.Trajectory``.

CLI (see ``adapters/mini_swe/README.md``)::

    python -m jevharness.adapters.mini_swe ids     --subset verified --split test --shuffle --slice 0:100 -o ids.txt
    python -m jevharness.adapters.mini_swe arms    -c adapters/mini_swe/swebench_jev.yaml
    python -m jevharness.adapters.mini_swe run     -c adapters/mini_swe/swebench_jev.yaml --arm stop_guard \\
                                                   --subset verified --split test --instance-ids ids.txt -o runs/swe/stop_guard
    python -m jevharness.adapters.mini_swe collect runs/swe/stop_guard --report <swebench report.json> -o stop_guard.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from minisweagent import __version__ as MINI_VERSION
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded, Submitted
from minisweagent.models.openrouter_model import (OpenRouterAPIError, OpenRouterAuthenticationError, OpenRouterModel,
                                                  OpenRouterRateLimitError)
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..accounting import CallRecord, Ledger, default_prices, estimate_tokens
from ..decision import make_backend
from ..decision.base import DecisionBackend
from ..decision.types import Answer, Choice, Noul, Question
from ..harness.config import DP
from ..harness.trajectory import DecisionLog, StepLog, Trajectory

RECORD_FORMAT = "jev-mini-swe-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "adapters" / "mini_swe" / "swebench_jev.yaml"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# hook name -> (DecisionLog.dp, admissible modes)
HOOKS: dict[str, tuple[str, set[str]]] = {
    "action_guard": (DP.TOOL_SELECT.value, {"off", "shadow", "guard"}),
    "stop": (DP.STOP.value, {"off", "shadow", "guard", "trigger"}),
    "routing": (DP.ROUTING.value, {"off", "shadow", "trigger"}),
    "reflection": (DP.REFLECTION.value, {"off", "shadow", "always", "trigger"}),
    "action_type": (DP.ACTION.value, {"off", "shadow", "assist"}),
}
S1_MODES = {"shadow", "guard", "trigger", "assist"}

ACTION_TYPES: dict[str, str] = {
    "explore": "read files / inspect the repository (cat, sed -n, ls, git show) to understand the relevant code",
    "search": "search for symbols or strings (grep, find, git grep) to locate the relevant code",
    "reproduce": "write or run a small script that reproduces the issue",
    "edit": "edit non-test source files to implement or refine the fix",
    "test": "run the reproduction script or the relevant existing tests against the modified code",
    "submit": "create the patch (git diff > patch.txt) and submit it",
}
GUARD_OPTIONS = ["on_track", "unproductive", "destructive"]
GUARD_OPTION_DESC = [
    "safe and a sensible next step toward fixing the issue",
    "safe but redundant, off-track, or unlikely to help",
    "deletes or resets code, tests, git history or the environment, or violates the task boundaries",
]

# ----------------------------------------------------------------------------- prompts
Q_GUARD = ("The coding agent proposes to run the bash command below in the repository. Is this command "
           "reasonable, on-track toward resolving the issue, and non-destructive (it does not delete or reset "
           "the repository, its tests, git history or the environment)?\nCOMMAND:\n{cmd}")
Q_GUARD_CHOICE = "Classify the bash command the coding agent proposes to run next.\nCOMMAND:\n{cmd}"
CRIT_GUARD = {"true": "The command is safe and a sensible next step for fixing the issue.",
              "false": "The command is destructive, violates the task boundaries (e.g. edits or deletes tests), "
                       "or is unproductive / off-track."}
Q_STOP_GUARD = ("The coding agent now wants to submit its patch and end the task. Has the fix been verified: was "
                "the issue reproduced, and were the modified code and the relevant tests actually run after the "
                "last source edit, with results showing that the issue is fixed?")
CRIT_STOP = {"true": "Verified: a reproduction and/or relevant tests were run after the final edit and pass.",
             "false": "Unverified: nothing was run after the final change, or the runs still show the bug/errors."}
Q_STOP_TRIGGER = ("Is the issue already fixed and verified (reproduction and relevant tests pass after the last "
                  "edit), so the agent should create the patch and submit now instead of continuing to work?")
Q_ROUTING = ("Does the next step require careful multi-step reasoning, planning, debugging, or recovery from an "
             "error (i.e., a strong model) rather than a routine action such as reading a file, searching, or "
             "re-running a command?")
Q_REFLECTION = ("Is the agent stuck, repeating itself, or did its last command fail in a way that calls for "
                "stepping back and rethinking its approach?")
Q_ACTION = "Which kind of action should the coding agent take next?"

MSG_BLOCK = ("[Guard] This command was NOT executed: an automatic safety/progress check judged it likely "
             "destructive, off-track or unproductive (p_ok={p:.2f} < {tau:.2f}). Reconsider your approach and issue "
             "a different command.")
MSG_BLOCK_OVERRIDE = " If you are certain this exact command is necessary, re-issue it unchanged and it will run."
MSG_STOP_NUDGE = ("[Verifier] Submission deferred: the fix does not appear to be verified yet. Before submitting, "
                  "(1) make sure you have reproduced the issue, (2) re-run your reproduction script and the most "
                  "relevant existing tests against your modified code, and (3) fix any failures. Then create the "
                  "patch and submit again with the exact submit command.")
MSG_REFLECTION = ("[Reflection] {reason}Before your next command, briefly (at most 4 sentences in your reasoning) "
                  "assess your progress toward fixing the issue, identify any mistake in your approach so far, and "
                  "state the best next step. Then issue the next command.")
MSG_STOP_HINT = ("[Hint] The fix may already be complete and verified. If so, create the patch and submit now "
                 "(follow the submission instructions); otherwise continue.")
MSG_ACTION_HINT = "[Hint] A fast planner suggests the most useful next kind of action: {items}. You decide."


def _trunc(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + "\n...[truncated]...\n" + s[-n // 2:]


def _norm_cmd(cmd: str) -> str:
    return " ".join((cmd or "").split())


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


# ============================================================================= config
class HookConfig(BaseModel):
    """One decision point. Mirrors ``jevharness.harness.config.DPConfig`` (mode + tau + k + extra) but
    takes a plain string mode so adapter-only modes (``shadow``) validate per hook (see ``HOOKS``)."""

    model_config = ConfigDict(extra="forbid")
    mode: str = "off"
    tau: float = 0.5
    k: int = 2
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v: Any) -> str:
        if v is False or v is None:  # YAML 1.1 parses a bare `off` as False
            return "off"
        v = getattr(v, "value", v)
        return "shadow" if v == "log" else v  # `log` = shadow in jevharness.adapters.hooks


class JevMiniSweConfig(BaseModel):
    """The ``jev:`` section of the YAML config (one arm after :meth:`for_arm`)."""

    model_config = ConfigDict(extra="forbid")
    name: str = "baseline"
    s1: dict[str, Any] | str | None = None           # make_backend spec, e.g. {type: jev, model: jev-latest}
    s1_cache: str | None = None                      # sqlite cache path for the S1 backend
    s1_fail_open: bool = True                        # S1 error -> behave like the baseline (logged), not crash
    cheap_model: dict[str, Any] | None = None        # mini-swe `model:` overrides for the routing target
    history_window: int = 8
    obs_truncate_chars: int = 2000
    hist_obs_chars: int = 400
    task_max_chars: int = 4000
    thought_max_chars: int = 1500
    s1_state_max_chars: int = 12000
    batch_s1: bool = True                            # one S1 call for all questions sharing a state
    log_states: bool = True                          # keep S1 input states in DecisionLog (training harvest)
    gold: str = "none"                               # none | heuristic  (weak labels for simulated/oracle S1)
    submit_marker: str = SUBMIT_MARKER
    action_guard: HookConfig = Field(default_factory=HookConfig)
    stop: HookConfig = Field(default_factory=HookConfig)
    routing: HookConfig = Field(default_factory=HookConfig)
    reflection: HookConfig = Field(default_factory=HookConfig)
    action_type: HookConfig = Field(default_factory=HookConfig)
    arms: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def hook(self, name: str) -> HookConfig:
        return getattr(self, name)

    def validate_modes(self) -> "JevMiniSweConfig":
        for name, (_, modes) in HOOKS.items():
            m = self.hook(name).mode
            if m not in modes:
                raise ValueError(f"mode {m!r} not supported at hook {name!r} (allowed: {sorted(modes)})")
        if self.gold not in ("none", "heuristic"):
            raise ValueError(f"gold must be 'none' or 'heuristic', got {self.gold!r}")
        return self

    def uses_s1(self) -> bool:
        return any(self.hook(h).mode in S1_MODES for h in HOOKS)

    def short(self) -> str:
        parts = [f"{h}:{self.hook(h).mode}(tau={self.hook(h).tau})" for h in HOOKS if self.hook(h).mode != "off"]
        return ",".join(parts) or "baseline"

    @classmethod
    def for_arm(cls, spec: dict[str, Any] | None, arm: str | None = None) -> "JevMiniSweConfig":
        """Resolve one arm: the top-level ``jev:`` hooks are the defaults, ``jev.arms[arm]`` overrides them."""
        spec = copy.deepcopy(spec or {})
        arms = spec.pop("arms", None) or {}
        if arm:
            if arm not in arms:
                raise KeyError(f"arm {arm!r} not in jev.arms (available: {sorted(arms)})")
            spec = _deep_merge(spec, arms[arm] or {})
            spec["name"] = (arms[arm] or {}).get("name", arm)
        return cls(**spec).validate_modes()


# ============================================================================= heuristics
_RM_RE = re.compile(r"(?:^|[;&|(`]\s*|\s)(?:sudo\s+)?rm\s+([^;&|\n`)]*)")
_DANGEROUS_RM_TARGETS = {"/", "/*", "~", "~/", "~/*", ".", "./", "./*", "..", "../", "*", "/testbed", "/testbed/",
                         "/testbed/*", "$HOME", "${HOME}"}
_DESTRUCTIVE_RES = [re.compile(p) for p in (
    r"\bgit\s+(reset\s+--hard|clean\s+-\w*[fdx]|checkout\s+(\S+\s+)?--\s+\.(\s|$)|checkout\s+\.(\s|$)|restore\s+\.(\s|$)"
    r"|stash(\s|$)|push\b|rebase\b|filter-branch\b|branch\s+-D\b)",
    r"\b(mkfs(\.\w+)?|shutdown|reboot|halt|poweroff)\b",
    r"\bdd\s+[^\n]*\bof=/dev/",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bchmod\s+(-\w+\s+)*-R\s+\S+\s+/(\s|$)",
    r"\bchown\s+(-\w+\s+)*-R\s+\S+\s+/(\s|$)",
    r"\b(pip3?|conda|mamba)\s+(uninstall|remove)\b",
    r"\bkill(all)?\s+-9\s+(-1|1)\b",
    r">\s*/dev/sd[a-z]",
    r"\bsed\s+(-\w+\s+)*-i\S*\s+.*(^|\s|/)tests?/\S+",           # in-place edits of existing test files
)]
_EDIT_RES = [re.compile(p) for p in (
    r"\bsed\s+(-\w+\s+)*-i",
    r"\bperl\s+-\w*i",
    r"\bgit\s+apply\b",
    r"(^|[\s;&|])patch\s+(-\w+\s+)*",
    r"(>{1,2}|\btee\s+(-a\s+)?)\s*['\"]?[\w./-]+\.(py|pyx|pxd|pyi|c|cc|cpp|h|hpp|js|jsx|ts|tsx|java|go|rs|rb|php|"
    r"cfg|toml|ini|yml|yaml|json|rst|md|html|css|sh)\b",
    r"\bopen\([^)]*,\s*['\"][wa]\+?['\"]",
    r"\.write_text\(",
    r"\bstr_replace\b",
)]
_TEST_RE = re.compile(r"\b(pytest|py\.test|tox|nosetests|unittest)\b|\bruntests\.py\b|\bmanage\.py\s+test\b|"
                      r"\bmake\s+(test|check)\b|\bbin/test\b|\bnpm\s+(run\s+)?test\b|\bgo\s+test\b|\bcargo\s+test\b")
_RUN_RE = re.compile(r"\bpython[\d.]*\s+(-\w+\s+)*([\w./-]+\.py|-c|-m\s+\w+)|\bnode\s+[\w./-]+\.js\b")
_SEARCH_RE = re.compile(r"(^|[\s;&|(])(grep|rg|ag|ack|find|locate|fd)\b|\bgit\s+(grep|log\s+-S)")
_REPRO_NAME_RE = re.compile(r"(repro|reproduce|debug|check_|test_fix|scratch)", re.I)
_BENIGN_NONZERO_RE = re.compile(r"^\s*(grep|rg|ag|diff|find|test|\[|cmp|git\s+diff)\b")


def is_submit_command(cmd: str, marker: str = SUBMIT_MARKER) -> bool:
    return marker in (cmd or "")


def is_destructive(cmd: str) -> bool:
    """Weak rule-based label: does this command irreversibly damage the repo / tests / environment?"""
    cmd = cmd or ""
    for m in _RM_RE.finditer(cmd):
        toks = m.group(1).split()
        flags = [t for t in toks if t.startswith("-")]
        targets = [t.strip("'\"") for t in toks if not t.startswith("-")]
        recursive = any(("r" in f.lstrip("-").lower() and not f.startswith("--")) or f == "--recursive" for f in flags)
        for t in targets:
            base = t.rstrip("/")
            if recursive and (t in _DANGEROUS_RM_TARGETS or base in _DANGEROUS_RM_TARGETS):
                return True
            if base.endswith(".git") or re.search(r"(^|/)\.git(/|$)", t):
                return True
            parts = [p for p in base.split("/") if p]
            if (recursive and parts and parts[-1] in ("test", "tests", "testing")) or \
                    any(p in ("test", "tests", "testing") for p in parts[:-1]):
                return True
    return any(r.search(cmd) for r in _DESTRUCTIVE_RES)


def is_edit_command(cmd: str) -> bool:
    if not cmd or is_submit_command(cmd) or re.search(r"git\s+diff\b.*>", cmd):
        return False
    return any(r.search(cmd) for r in _EDIT_RES)


def is_run_command(cmd: str) -> bool:
    return bool(cmd) and (bool(_TEST_RE.search(cmd)) or bool(_RUN_RE.search(cmd)))


def classify_command(cmd: str, marker: str = SUBMIT_MARKER) -> str:
    """Map a bash command onto ``ACTION_TYPES`` (used as S2's revealed action type)."""
    cmd = cmd or ""
    if is_submit_command(cmd, marker) or re.search(r"git\s+diff\b.*>\s*\S*patch", cmd) or \
            re.search(r"\bcat\s+\S*patch\.txt\b", cmd):
        return "submit"
    if is_edit_command(cmd):
        m = re.search(r"(>{1,2}|tee\s+(-a\s+)?)\s*['\"]?([\w./-]+)", cmd)
        target = m.group(3) if m else ""
        return "reproduce" if _REPRO_NAME_RE.search(target) else "edit"
    if _TEST_RE.search(cmd):
        return "test"
    if _RUN_RE.search(cmd):
        return "reproduce"
    if _SEARCH_RE.search(cmd):
        return "search"
    return "explore"


def fix_verified(history: list[dict[str, Any]]) -> bool:
    """Weak label for the stop hook: some source edit happened and something was run after the last one."""
    done = [e for e in history if not e.get("blocked")]
    edits = [i for i, e in enumerate(done) if is_edit_command(e["command"])]
    if not edits:
        return False
    return any(is_run_command(e["command"]) for e in done[edits[-1] + 1:])


def heuristic_labeler(hook: str, ctx: dict[str, Any]) -> Any:
    """Rule-based weak gold labels (``jev.gold: heuristic``). They let ``oracle`` / ``simulated`` backends
    exercise the hooks offline and give a rough accuracy reference for real S1 backends; they are NOT
    ground truth. Routing / action-type have no rule-based label (None)."""
    if hook == "action_guard":
        return not is_destructive(ctx.get("command", ""))
    if hook in ("stop", "stop_trigger"):
        return fix_verified(ctx.get("history", []))
    if hook == "reflection":
        ev = ctx.get("events") or {}
        return bool(ev.get("failed") or ev.get("repeated") or ev.get("timeout"))
    return None


Labeler = Callable[[str, dict[str, Any]], Any]


# ============================================================================= models
def _model_name(model: Any) -> str:
    return str(getattr(getattr(model, "config", None), "model_name", None) or type(model).__name__)


def build_cheap_model(strong_model_config: dict[str, Any], cheap_spec: dict[str, Any]) -> Any:
    """The routing target: the strong model's mini-swe ``model:`` config (templates, kwargs) overridden by
    ``jev.cheap_model`` (typically just ``model_name`` / ``model_kwargs`` / ``model_class``)."""
    from minisweagent.models import get_model
    cfg = _deep_merge(strong_model_config, cheap_spec)
    return get_model(config=cfg)


class OpenRouterTimeoutModel(OpenRouterModel):
    """mini-swe's ``OpenRouterModel`` (2.4.6) with a configurable request timeout. Upstream hard-codes 60 s per
    request, which a reasoning model's long thinking step can exceed (DeepSeek V4 Pro with reasoning on decodes at
    ~50 tokens/s on the pinned endpoint). A 200 response without ``choices`` (an upstream provider error passed
    through), or one that OpenRouter did not bill (``usage.cost`` 0: a cut-off, failed generation; with
    ``retry_unbilled``, the default), is raised as an API error, so mini-swe's retry handles it like any other
    transient failure instead of failing the instance in its cost accounting. Everything else (payload,
    ``usage.cost`` accounting, parsing) is upstream's. Select it with ``model_class:
    jevharness.adapters.mini_swe.OpenRouterTimeoutModel`` and set ``request_timeout`` (seconds) in ``model:``.

    ``api_url`` (E29) replaces OpenRouter's chat-completions URL, e.g. the host-side routing proxy
    (``jevharness.adapters.route_proxy``), which holds the key: the agent process then needs no OPENROUTER_API_KEY.
    ``process_instance`` fills ``{instance_id}`` in it, so the proxy sees one episode per instance."""

    def __init__(self, *, request_timeout: float = 600.0, retry_unbilled: bool = True, api_url: str | None = None,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.request_timeout = float(request_timeout)
        self.retry_unbilled = bool(retry_unbilled)
        if api_url:
            self._api_url = str(api_url)

    def _query(self, messages: list[dict[str, str]], **kwargs: Any):
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self.config.model_name, "messages": messages, "tools": [BASH_TOOL],
                   "usage": {"include": True}, **(self.config.model_kwargs | kwargs)}
        try:
            response = requests.post(self._api_url, headers=headers, data=json.dumps(payload),
                                     timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise OpenRouterAuthenticationError("Authentication failed (OPENROUTER_API_KEY).") from e
            if response.status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded") from e
            raise OpenRouterAPIError(f"HTTP {response.status_code}: {response.text[:500]}") from e
        except (requests.exceptions.RequestException, ValueError) as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e
        if not isinstance(data, dict) or not data.get("choices"):
            err = data.get("error") if isinstance(data, dict) else None
            raise OpenRouterAPIError(f"no choices in the response: {str(err)[:500]}")
        usage = data.get("usage") or {}
        if self.retry_unbilled and not float(usage.get("cost") or 0.0) > 0.0:
            # E4 dev, 2026-10-05: StreamLake sometimes returns a cut-off generation (a few reasoning tokens, no cache
            # hit) that OpenRouter does not bill (usage.cost 0, cost_details.upstream_inference_cost > 0); upstream
            # mini-swe then fails the whole instance in its cost accounting. Treat it as a failed generation: retry.
            fin = (data["choices"][0] or {}).get("finish_reason")
            up = (usage.get("cost_details") or {}).get("upstream_inference_cost")
            raise OpenRouterAPIError(f"unbilled response (usage.cost {usage.get('cost')!r}, upstream cost {up!r}, "
                                     f"finish_reason {fin!r}, completion_tokens {usage.get('completion_tokens')!r}): "
                                     "treated as a failed generation and retried")
        return data


def _message_usage(message: dict[str, Any]) -> tuple[int, int, int] | None:
    resp = (message.get("extra") or {}).get("response")
    usage = resp.get("usage") if isinstance(resp, dict) else None
    if not isinstance(usage, dict):
        return None
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    det = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = int((det or {}).get("cached_tokens") or usage.get("cache_read_input_tokens") or 0)
    return inp, out, cached


def _content_text(message: dict[str, Any]) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    try:
        from minisweagent.models.utils.content_string import get_content_string
        return get_content_string({k: v for k, v in message.items() if k != "tool_calls"})
    except Exception:
        return ""


# ============================================================================= agent
class JevMiniSweAgent(DefaultAgent):
    """``DefaultAgent`` with System-1 hooks. Extra constructor arguments (everything else is mini-swe's
    ``AgentConfig``):

    jev             dict (the YAML ``jev:`` section, one arm) or ``JevMiniSweConfig``
    s1              a ``DecisionBackend``; built from ``jev.s1`` when omitted and a hook needs one
    cheap_model     second mini-swe ``Model`` for routing; built from ``jev.cheap_model`` when omitted
    labeler         ``fn(hook, ctx) -> gold`` for oracle / simulated backends (``jev.gold: heuristic`` uses
                    :func:`heuristic_labeler`)
    progress_manager, instance_id   as in mini-swe's ``ProgressTrackingAgent`` (batch runs)
    """

    def __init__(self, model: Any, env: Any, *, jev: dict[str, Any] | JevMiniSweConfig | None = None,
                 s1: DecisionBackend | None = None, cheap_model: Any = None, labeler: Labeler | None = None,
                 progress_manager: Any = None, instance_id: str = "", config_class: type = AgentConfig,
                 **kwargs: Any):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.jcfg = jev if isinstance(jev, JevMiniSweConfig) else JevMiniSweConfig.for_arm(jev or {}, None)
        self.jcfg.validate_modes()
        if self.jcfg.uses_s1() and s1 is None:
            if self.jcfg.s1 is None:
                raise ValueError(f"jev config {self.jcfg.name!r} uses System-1 hooks but no backend was given "
                                 "(pass s1=... or set jev.s1)")
            s1 = make_backend(self.jcfg.s1, cache=self.jcfg.s1_cache)
        self.s1 = s1
        if cheap_model is None and self.jcfg.cheap_model and self.jcfg.routing.mode == "trigger":
            cheap_model = build_cheap_model(model.config.model_dump(), {
                "model_class": f"{type(model).__module__}.{type(model).__name__}", **self.jcfg.cheap_model})
        if self.jcfg.routing.mode == "trigger" and cheap_model is None:
            raise ValueError("routing=trigger needs a cheap model (cheap_model=... or jev.cheap_model)")
        self.cheap_model = cheap_model
        if labeler is None and self.jcfg.gold == "heuristic":
            labeler = heuristic_labeler
        self.labeler = labeler
        self.gold_source = None if labeler is None else ("heuristic" if labeler is heuristic_labeler else "custom")
        self.progress_manager = progress_manager
        self.instance_id = instance_id
        self._reset_jev_state()

    # ------------------------------------------------------------------ state
    def _reset_jev_state(self) -> None:
        self.ledger = Ledger()
        self.jev_steps: list[StepLog] = []
        self.jev_history: list[dict[str, Any]] = []
        self.jev_counters: Counter[str] = Counter()
        self.jev_s1_errors: list[str] = []
        self.jev_error: str | None = None
        self.jev_wall_ms = 0.0
        self.model_stats: dict[str, dict[str, Any]] = {}
        self._task = ""
        self._step = -1
        self._cur: StepLog | None = None
        self._route_role: str | None = None
        self._blocked_counts: Counter[str] = Counter()
        self._action_dlog: DecisionLog | None = None
        self._last_reflection_step = -10 ** 9
        self._t0 = time.perf_counter()

    def run(self, task: str = "", **kwargs: Any) -> dict:
        self._reset_jev_state()
        self._task = task
        try:
            return super().run(task, **kwargs)
        except Exception as e:
            self.jev_error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
            raise
        finally:
            self.jev_wall_ms = (time.perf_counter() - self._t0) * 1000

    def _limits_reached(self) -> bool:
        c = self.config
        return (0 < c.step_limit <= self.n_calls or 0 < c.cost_limit <= self.cost
                or 0 < c.wall_time_limit_seconds <= int(time.time() - self._start_time))

    def _slog(self) -> StepLog:
        if self._cur is None:  # execute_actions/query called outside step(): open a step
            self._step += 1
            self._cur = StepLog(step=self._step)
            self.jev_steps.append(self._cur)
        return self._cur

    # ------------------------------------------------------------------ S1 plumbing
    def _s1_state(self, extra: str = "") -> str:
        c = self.jcfg
        parts = [f"TASK (issue to resolve):\n{_trunc(self._task, c.task_max_chars)}"]
        h = self.jev_history[-c.history_window:]
        if h:
            lines = []
            for e in h:
                tag = " [NOT EXECUTED]" if e.get("blocked") else ""
                lines.append(f"[{e['step']}] $ {_trunc(e['command'], 300)}{tag} -> rc={e['returncode']}: "
                             f"{_trunc(e['output'], c.hist_obs_chars)}")
            parts.append("HISTORY (most recent last):\n" + "\n".join(lines))
        last = self.jev_history[-1]["output"] if self.jev_history else "(no command executed yet)"
        parts.append(f"CURRENT OBSERVATION:\n{_trunc(last, c.obs_truncate_chars)}")
        parts.append(f"STEP: {max(self._step, 0)}")
        if extra:
            parts.append(extra)
        return _trunc("\n\n".join(parts), c.s1_state_max_chars)

    def _ask_many(self, state: str, questions: list[Question]) -> list[Answer | None]:
        """One batched S1 call (``batch_s1``) or one call per question. Fail-open: None on error."""
        if not questions:
            return []
        assert self.s1 is not None
        try:
            if self.jcfg.batch_s1 and len(questions) > 1:
                dp = "+".join(dict.fromkeys(q.decision_point or "?" for q in questions))
                res = self.s1.decide(state, questions, ledger=self.ledger, decision_point=dp)
                by_id = res.by_id()
                return [by_id.get(q.id) for q in questions]
            return [self.s1.ask(state, q, ledger=self.ledger) for q in questions]
        except Exception as e:
            if not self.jcfg.s1_fail_open:
                raise
            self.jev_counters["s1_errors"] += 1
            self.jev_s1_errors.append(f"step {self._step}: {type(e).__name__}: {e}"[:500])
            return [None] * len(questions)

    def _gold(self, hook: str, **ctx: Any) -> Any:
        if self.labeler is None:
            return None
        try:
            return self.labeler(hook, {"step": self._step, "history": self.jev_history, "task": self._task,
                                       "events": self._events(), **ctx})
        except Exception:
            return None

    def _events(self) -> dict[str, bool]:
        """Failure / repetition events of the most recent step (reflection triggers)."""
        if not self.jev_history:
            return {"failed": False, "repeated": False, "timeout": False}
        last_step = self.jev_history[-1]["step"]
        last = [e for e in self.jev_history if e["step"] == last_step and not e.get("blocked")]
        failed = any(e["returncode"] not in (0, None) and not (e["returncode"] == 1 and
                     _BENIGN_NONZERO_RE.search(e["command"])) for e in last)
        timeout = any(e.get("exception") for e in last)
        win = int(self.jcfg.reflection.extra.get("repeat_window", 5))  # <= 0: whole history
        earlier = {_norm_cmd(e["command"]) for e in self.jev_history
                   if e["step"] < last_step and (win <= 0 or e["step"] >= last_step - win)}
        repeated = any(_norm_cmd(e["command"]) in earlier for e in last)
        return {"failed": failed, "repeated": repeated, "timeout": timeout}

    def _dlog(self, hook: str, mode: str, question: str, *, state: str | None = None, **kw: Any) -> DecisionLog:
        d = DecisionLog(HOOKS[hook][0], mode, question, state=state if self.jcfg.log_states else None, **kw)
        self._slog().decisions.append(d)
        return d

    # ------------------------------------------------------------------ step / query
    def step(self) -> list[dict]:
        self._step += 1
        self.ledger.step = self._step
        self._cur = StepLog(step=self._step)
        self.jev_steps.append(self._cur)
        self._action_dlog = None
        if self.progress_manager is not None:
            self.progress_manager.update_instance_status(
                self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f}) [{self.jcfg.name}]")
        if not self._limits_reached():
            self._pre_query_hooks()
        try:
            message = self.query()
        except FormatError:
            self._cur.observation = "(format error: no valid action)"
            raise
        except LimitsExceeded:
            if self._cur.llm_calls == 0 and not self._cur.decisions:
                self.jev_steps.pop()
            raise
        return self.execute_actions(message)

    def query(self) -> dict:
        role = self._route_role or "strong"
        self._route_role = None
        model = self.cheap_model if role == "cheap" and self.cheap_model is not None else self.model
        role = "cheap" if model is self.cheap_model and model is not self.model else "strong"
        slog = self._slog()
        strong = self.model
        n_before = self.n_calls
        t = time.perf_counter()
        self.model = model  # DefaultAgent.query() calls self.model.query and books the cost
        try:
            message = super().query()
        except FormatError as e:
            self._record_s2(model, role, e.messages[0] if e.messages else {}, time.perf_counter() - t, slog,
                            format_error=True)
            raise
        finally:
            self.model = strong
        if self.n_calls > n_before:
            self._record_s2(model, role, message, time.perf_counter() - t, slog)
            message["extra"] = {**message.get("extra", {}), "jev_model": {"role": role, "model": _model_name(model)}}
        return message

    def _record_s2(self, model: Any, role: str, message: dict[str, Any], seconds: float, slog: StepLog,
                   format_error: bool = False) -> None:
        name = _model_name(model)
        cost = float((message.get("extra") or {}).get("cost", 0.0) or 0.0)
        usage = _message_usage(message)
        estimated = usage is None
        if usage is None:  # test models / providers without usage: ~4 chars per token
            n_in = sum(len(m.get("content") or "") if isinstance(m.get("content"), str) else 0 for m in self.messages)
            out_txt = _content_text(message) + json.dumps((message.get("extra") or {}).get("actions", []))
            usage = (max(1, n_in // 4), estimate_tokens(out_txt), 0)
        n_in, n_out, cached = usage
        price = default_prices().get(name)
        if cost <= 0 and price is not None:
            cost = price.cost(n_in, n_out, cached)
        self.ledger.add(CallRecord(component="s2", backend=f"mini-swe:{type(model).__name__}", model=name,
                                   decision_point=DP.TOOL_SELECT.value, input_tokens=n_in, output_tokens=n_out,
                                   cost_usd=cost, latency_ms=seconds * 1000, priced=cost > 0 or price is not None,
                                   meta={"role": role, "cached_input_tokens": cached, "format_error": format_error,
                                         "estimated_tokens": estimated}))
        st = self.model_stats.setdefault(role, {"model": name, "calls": 0, "cost_usd": 0.0, "input_tokens": 0,
                                                "output_tokens": 0, "latency_ms": 0.0})
        st["calls"] += 1
        st["cost_usd"] += cost
        st["input_tokens"] += n_in
        st["output_tokens"] += n_out
        st["latency_ms"] += seconds * 1000
        slog.llm_calls += 1
        slog.model = name

    # ------------------------------------------------------------------ pre-query hooks
    def _pre_query_hooks(self) -> None:
        """routing / reflection / stop-trigger / action-type: questions on one shared state, one S1 call."""
        c = self.jcfg
        step = self._step
        events = self._events()
        asks: list[tuple[str, Question, Any]] = []
        notes: list[tuple[str, str]] = []

        r = c.routing
        if r.mode in ("trigger", "shadow"):
            if step < int(r.extra.get("first_n_strong", 0)):
                self._route_role = "strong"
            else:
                gold = self._gold("routing")
                asks.append(("routing", Noul(Q_ROUTING, decision_point=DP.ROUTING.value, gold=gold), gold))

        rf = c.reflection
        if rf.mode != "off" and step > 0:
            on_every = rf.extra.get("when", "events") == "every_step"
            cooled = step - self._last_reflection_step > int(rf.extra.get("cooldown", 2))
            budget = self.jev_counters["reflections"] < int(rf.extra.get("max_reflections", 5))
            if (on_every or any(events.values())) and cooled and budget:
                if rf.mode == "always":
                    self._dlog("reflection", "always", "rule: failed/repeated command", final_value=True)
                    notes.append(("reflection", self._reflection_text(events)))
                    self._fired_reflection()
                else:
                    gold = self._gold("reflection")
                    asks.append(("reflection", Noul(Q_REFLECTION, decision_point=DP.REFLECTION.value, gold=gold),
                                 gold))

        sp = c.stop
        if sp.mode in ("trigger", "shadow") and step >= int(sp.extra.get("min_step", 5)) and \
                self.jev_counters["stop_hints"] < int(sp.extra.get("max_hints", 2)) and \
                (not sp.extra.get("require_edit", True) or any(is_edit_command(e["command"]) and not e["blocked"]
                                                                for e in self.jev_history)):
            gold = self._gold("stop_trigger")
            asks.append(("stop_trigger", Noul(Q_STOP_TRIGGER, decision_point=DP.STOP.value, gold=gold,
                                              criteria=CRIT_STOP), gold))

        at = c.action_type
        if at.mode != "off" and step % max(1, int(at.extra.get("every_n_steps", 1))) == 0:
            gold = self._gold("action_type")
            asks.append(("action_type", Choice(Q_ACTION, list(ACTION_TYPES), list(ACTION_TYPES.values()),
                                               decision_point=DP.ACTION.value, gold=gold), gold))

        if asks:
            state = self._s1_state()
            answers = self._ask_many(state, [q for _, q, _ in asks])
            for (hook, q, gold), a in zip(asks, answers):
                note = self._dispatch_pre(hook, q, gold, a, state, events)
                if note:
                    notes.append((hook, note))
        if notes:
            self.add_messages(self.model.format_message(
                role="user", content="\n\n".join(n for _, n in notes),
                extra={"jev_hints": [h for h, _ in notes], "step": step}))

    def _reflection_text(self, events: dict[str, bool]) -> str:
        reason = ("Your last command failed. " if events.get("failed") or events.get("timeout") else
                  "You are repeating a command you already ran. " if events.get("repeated") else "")
        return MSG_REFLECTION.format(reason=reason)

    def _fired_reflection(self) -> None:
        self._last_reflection_step = self._step
        self.jev_counters["reflections"] += 1

    def _dispatch_pre(self, hook: str, q: Question, gold: Any, a: Answer | None, state: str,
                      events: dict[str, bool]) -> str | None:
        c = self.jcfg
        if a is None:  # S1 failed (fail-open): baseline behaviour
            self._dlog(hook if hook != "stop_trigger" else "stop", "error", q.text, gold=gold, deferred=True)
            return None
        if hook == "routing":
            big = a.p_yes >= c.routing.tau
            role = "strong" if (big or c.routing.mode == "shadow") else "cheap"
            self._route_role = role
            self.jev_counters[f"routed_{role}"] += 1
            self._dlog("routing", c.routing.mode, q.text, s1_value=big, s1_conf=a.confidence, s1_dist=a.distribution,
                       final_value="large" if role == "strong" else "small", gold=gold, state=state)
            return None
        if hook == "reflection":
            fire = a.p_yes >= c.reflection.tau
            act = fire and c.reflection.mode == "trigger"
            self._dlog("reflection", c.reflection.mode, q.text, s1_value=fire, s1_conf=a.confidence,
                       s1_dist=a.distribution, final_value=act, gold=gold, state=state)
            if act:
                self._fired_reflection()
                text = self._reflection_text(events)
                self._slog().reflection = text
                return text
            return None
        if hook == "stop_trigger":
            done = a.p_yes >= c.stop.tau
            act = done and c.stop.mode == "trigger"
            mode = "shadow_trigger" if c.stop.mode == "shadow" else c.stop.mode  # 2 stop questions in shadow
            self._dlog("stop", mode, q.text, s1_value=done, s1_conf=a.confidence, s1_dist=a.distribution,
                       final_value=act, gold=gold, state=state)
            if act:
                self.jev_counters["stop_hints"] += 1
                return MSG_STOP_HINT
            return None
        if hook == "action_type":
            at = c.action_type
            min_p = float(at.extra.get("min_item_p", 0.05))  # drop near-zero alternatives from the hint
            top = [o for i, o in enumerate(a.top_k(max(1, at.k))) if i == 0 or a.distribution[o] >= min_p]
            self._action_dlog = self._dlog("action_type", at.mode, Q_ACTION, options=list(ACTION_TYPES),
                                           s1_value=a.value, s1_conf=a.confidence, s1_dist=a.distribution,
                                           gold=gold, state=state)
            if at.mode == "assist" and a.confidence >= float(at.extra.get("min_conf", 0.0)):
                self.jev_counters["action_hints"] += 1
                items = ", ".join(f"{o} (p={a.distribution[o]:.2f}; {ACTION_TYPES[o]})" for o in top)
                return MSG_ACTION_HINT.format(items=items)
            return None
        return None

    # ------------------------------------------------------------------ execution hooks
    def execute_actions(self, message: dict) -> list[dict]:
        slog = self._slog()
        actions = (message.get("extra") or {}).get("actions", [])
        cmds = [str(a.get("command", "")) for a in actions]
        marker = self.jcfg.submit_marker
        if cmds:
            slog.tool = "submit" if any(is_submit_command(x, marker) for x in cmds) else "bash"
            slog.args = {"command": cmds[0]} if len(cmds) == 1 else {"commands": cmds}
        if self._action_dlog is not None and cmds:  # S2's revealed action type
            revealed = classify_command(cmds[0], marker)
            self._action_dlog.s2_value = self._action_dlog.final_value = revealed
        self._action_dlog = None
        blocked = self._guard(cmds, _content_text(message))
        outputs: list[dict[str, Any]] = []
        for i, action in enumerate(actions):
            cmd = cmds[i]
            if i in blocked:
                outputs.append(blocked[i])
                self._remember(cmd, blocked[i], blocked=True)
                continue
            if is_submit_command(cmd, marker):
                nudge = self._stop_guard()
                if nudge is not None:
                    outputs.append(nudge)
                    self._remember(cmd, nudge, blocked=True, submit=True)
                    continue
            try:
                out = self.env.execute(action)
            except Submitted:
                self._remember(cmd, {"output": "(submitted)", "returncode": 0}, submit=True)
                slog.observation = "(submitted)"
                raise
            outputs.append(out)
            self._remember(cmd, out)
        slog.observation = _trunc("\n---\n".join(
            (o.get("exception_info") or "") + (o.get("output") or "") for o in outputs), self.jcfg.obs_truncate_chars)
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _remember(self, cmd: str, out: dict[str, Any], blocked: bool = False, submit: bool = False) -> None:
        text = (out.get("exception_info") or "") + ("\n" if out.get("exception_info") else "") + (out.get("output") or "")
        self.jev_history.append({"step": self._step, "command": cmd, "returncode": out.get("returncode"),
                                 "output": text, "blocked": blocked, "submit": submit,
                                 "exception": bool(out.get("exception_info")) and not blocked})

    def _guard(self, cmds: list[str], thought: str) -> dict[int, dict[str, Any]]:
        g = self.jcfg.action_guard
        if g.mode == "off" or not cmds:
            return {}
        kind = g.extra.get("kind", "noul")
        max_retries = int(g.extra.get("max_retries", 1))
        max_blocks = int(g.extra.get("max_blocks", 20))
        skip = g.extra.get("skip_regex") or ""
        todo: list[int] = []
        for i, cmd in enumerate(cmds):
            if is_submit_command(cmd, self.jcfg.submit_marker) or (skip and re.search(skip, cmd)):
                continue
            if g.mode == "guard" and (self._blocked_counts[_norm_cmd(cmd)] >= max_retries or
                                      self.jev_counters["blocked"] >= max_blocks):
                self._dlog("action_guard", g.mode, "override: re-issued after block or block budget exhausted",
                           s2_value=cmd, final_value=cmd, deferred=True)
                continue
            todo.append(i)
        if not todo:
            return {}
        state = self._s1_state(extra=f"AGENT'S REASONING FOR THIS STEP:\n{_trunc(thought, self.jcfg.thought_max_chars)}")
        qs, golds = [], []
        for i in todo:
            gold = self._gold("action_guard", command=cmds[i])
            golds.append(gold)
            if kind == "choice":
                cg = None if gold is None else ("on_track" if gold else "destructive")
                qs.append(Choice(Q_GUARD_CHOICE.format(cmd=_trunc(cmds[i], 1500)), GUARD_OPTIONS, GUARD_OPTION_DESC,
                                 decision_point=DP.TOOL_SELECT.value, gold=cg))
            else:
                qs.append(Noul(Q_GUARD.format(cmd=_trunc(cmds[i], 1500)), decision_point=DP.TOOL_SELECT.value,
                               gold=gold, criteria=CRIT_GUARD))
        blocked: dict[int, dict[str, Any]] = {}
        for i, q, gold, a in zip(todo, qs, golds, self._ask_many(state, qs)):
            cmd = cmds[i]
            if a is None:
                self._dlog("action_guard", "error", q.text, s2_value=cmd, final_value=cmd, gold=gold, deferred=True)
                continue
            p_ok = a.distribution.get("on_track", 0.0) if kind == "choice" else a.p_yes
            ok = p_ok >= g.tau
            block = g.mode == "guard" and not ok
            self._dlog("action_guard", g.mode, q.text, s1_value=ok, s1_conf=max(p_ok, 1 - p_ok),
                       s1_dist=a.distribution, s2_value=cmd, final_value=None if block else cmd, gold=gold,
                       state=state)
            if block:
                norm = _norm_cmd(cmd)
                self._blocked_counts[norm] += 1
                self.jev_counters["blocked"] += 1
                msg = MSG_BLOCK.format(p=p_ok, tau=g.tau)
                if self._blocked_counts[norm] >= max_retries:
                    msg += MSG_BLOCK_OVERRIDE
                blocked[i] = {"output": "", "returncode": -1, "exception_info": msg,
                              "extra": {"jev_blocked": DP.TOOL_SELECT.value, "p_ok": p_ok}}
        return blocked

    def _stop_guard(self) -> dict[str, Any] | None:
        s = self.jcfg.stop
        if s.mode not in ("guard", "shadow"):
            return None
        if s.mode == "guard" and self.jev_counters["nudges"] >= int(s.extra.get("max_nudges", 1)):
            self._dlog("stop", "guard", "nudge budget exhausted: submission accepted", final_value=True, deferred=True)
            return None
        gold = self._gold("stop")
        q = Noul(Q_STOP_GUARD, decision_point=DP.STOP.value, gold=gold, criteria=CRIT_STOP)
        state = self._s1_state(extra="The agent is now issuing the submit command.")
        a = self._ask_many(state, [q])[0]
        if a is None:
            self._dlog("stop", "error", q.text, final_value=True, gold=gold, deferred=True)
            return None
        ok = a.p_yes >= s.tau
        reject = s.mode == "guard" and not ok
        self._dlog("stop", "shadow_guard" if s.mode == "shadow" else s.mode, q.text, s1_value=ok,
                   s1_conf=a.confidence, s1_dist=a.distribution, final_value=not reject, gold=gold, state=state)
        if not reject:
            return None
        self.jev_counters["nudges"] += 1
        return {"output": "", "returncode": -1, "exception_info": MSG_STOP_NUDGE,
                "extra": {"jev_blocked": DP.STOP.value, "p_verified": a.p_yes}}

    # ------------------------------------------------------------------ outputs
    def jev_summary(self) -> dict[str, Any]:
        return {"arm": self.jcfg.name, "hooks": self.jcfg.short(),
                "s1_backend": self.s1.describe() if self.s1 is not None and self.jcfg.uses_s1() else None,
                "usage": self.ledger.summary(), "model_stats": self.model_stats,
                "counters": dict(self.jev_counters), "n_decisions": sum(len(s.decisions) for s in self.jev_steps)}

    def serialize(self, *extra_dicts: dict) -> dict:
        return super().serialize({"info": {"jev": self.jev_summary()}}, *extra_dicts)

    def jev_record(self, instance_id: str = "", exit_status: str | None = None, submission: str | None = None,
                   error: str | None = None) -> dict[str, Any]:
        """Self-contained per-instance record (the ``<iid>.jev.json`` file); see :func:`to_trajectory`."""
        last = (self.messages[-1].get("extra") or {}) if self.messages else {}
        exit_status = exit_status if exit_status is not None else last.get("exit_status", "")
        submission = submission if submission is not None else last.get("submission", "")
        return {
            "format": RECORD_FORMAT,
            "instance_id": instance_id or self.instance_id,
            "arm": self.jcfg.name,
            "jev_config": self.jcfg.model_dump(mode="json", exclude={"arms"}),
            "s1_backend": self.s1.describe() if self.s1 is not None and self.jcfg.uses_s1() else None,
            "models": {"strong": _model_name(self.model),
                       "cheap": _model_name(self.cheap_model) if self.cheap_model is not None else None},
            "exit_status": exit_status,
            "submission": submission or "",
            "result": {"submitted": exit_status == "Submitted", "exit_status": exit_status,
                       "patch_nonempty": bool((submission or "").strip()), "resolved": None},
            "steps": [asdict(s) for s in self.jev_steps],
            "calls": self.ledger.to_list(),
            "usage": self.ledger.summary(),
            "model_stats": {"instance_cost": self.cost, "api_calls": self.n_calls, "per_model": self.model_stats},
            "counters": dict(self.jev_counters),
            "s1_errors": list(self.jev_s1_errors),
            "gold_source": self.gold_source,
            "wall_ms": self.jev_wall_ms or (time.perf_counter() - self._t0) * 1000,
            "error": error or self.jev_error,
            "mini_version": MINI_VERSION,
        }

    def save_jev(self, path: str | Path, **kw: Any) -> dict[str, Any]:
        rec = self.jev_record(**kw)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1, default=str))
        return rec


# ============================================================================= conversion / evaluation
def to_trajectory(source: JevMiniSweAgent | dict[str, Any] | str | Path, *, resolved: bool | None = None,
                  stop_gold_from_eval: bool = False, **record_kw: Any) -> Trajectory:
    """Convert a live agent, a ``jev_record`` dict, or a ``<iid>.jev.json`` path into a ``Trajectory``.

    ``resolved`` (from the SWE-bench harness / sb-cli report) sets ``result.success``; before evaluation
    ``success`` is False and ``result.resolved`` is None. With ``stop_gold_from_eval`` the *last* stop
    decision of a submitted episode (the accepted submission, whose patch is the evaluated one) gets
    ``gold = resolved`` when it has no other label.
    """
    if isinstance(source, JevMiniSweAgent):
        rec = source.jev_record(**record_kw)
    elif isinstance(source, (str, Path)):
        rec = json.loads(Path(source).read_text())
    else:
        rec = source
    steps = []
    for s in rec.get("steps", []):
        s = dict(s)
        s["decisions"] = [DecisionLog(**d) for d in s.get("decisions", [])]
        steps.append(StepLog(**s))
    result = dict(rec.get("result") or {})
    if resolved is not None:
        result["resolved"] = bool(resolved)
    res = result.get("resolved")
    result["success"] = bool(res) if res is not None else False
    result["score"] = float(result["success"])
    if stop_gold_from_eval and res is not None and result.get("submitted"):
        stops = [d for st in steps for d in st.decisions if d.dp == DP.STOP.value and d.s1_value is not None
                 and d.mode in ("guard", "shadow_guard")]
        if stops and stops[-1].gold is None:
            stops[-1].gold = bool(res)
    ms = rec.get("model_stats") or {}
    return Trajectory(
        task_id=rec["instance_id"], config=rec.get("arm", "baseline"),
        backend=(rec.get("s1_backend") or {}).get("backend"), llm=(rec.get("models") or {}).get("strong"),
        steps=steps, final_answer=rec.get("submission", ""), result=result, usage=rec.get("usage") or {},
        calls=rec.get("calls") or [], wall_ms=float(rec.get("wall_ms") or 0.0), error=rec.get("error"),
        meta={"source": RECORD_FORMAT, "benchmark": "swe-bench", "exit_status": rec.get("exit_status"),
              "cheap_llm": (rec.get("models") or {}).get("cheap"), "model_stats": ms,
              "counters": rec.get("counters", {}), "s1_errors": rec.get("s1_errors", []),
              "gold_source": rec.get("gold_source"), "config": rec.get("jev_config"),
              "mini_version": rec.get("mini_version")})


def load_resolved_ids(reports: Iterable[str | Path]) -> set[str]:
    """Resolved instance ids from SWE-bench evaluation reports: the harness summary
    (``<model>.<run_id>.json``: ``resolved_ids``), sb-cli reports (``resolved_ids``), or per-instance
    ``report.json`` files (``{iid: {"resolved": bool}}``)."""
    out: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            ids = obj.get("resolved_ids")
            if isinstance(ids, list):
                out.update(str(i) for i in ids)
            for k, v in obj.items():
                if isinstance(v, dict) and v.get("resolved") is True and "__" in str(k):
                    out.add(str(k))
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for p in reports:
        walk(json.loads(Path(p).read_text()))
    return out


def collect_run(run_dir: str | Path, reports: Iterable[str | Path] = (), out: str | Path | None = None,
                stop_gold_from_eval: bool = False) -> list[dict[str, Any]]:
    """All ``<iid>/<iid>.jev.json`` of one arm -> Trajectory dicts (optionally with eval results) ->
    optional JSONL readable by ``jevharness.bench.metrics`` / ``jevbench report``."""
    reports = list(reports)
    resolved = load_resolved_ids(reports) if reports else None
    trajs = []
    for p in sorted(Path(run_dir).glob("*/*.jev.json")):
        rec = json.loads(p.read_text())
        r = None if resolved is None else rec["instance_id"] in resolved
        trajs.append(to_trajectory(rec, resolved=r, stop_gold_from_eval=stop_gold_from_eval).to_dict())
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for t in trajs:
                f.write(json.dumps(t, default=str) + "\n")
    return trajs


# ============================================================================= batch runner
_FILE_LOCK = threading.Lock()


def load_config(config_spec: list[str]) -> dict[str, Any]:
    from minisweagent.config import get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge
    return recursive_merge(*[get_config_from_spec(s) for s in config_spec])


def process_instance(instance: dict[str, Any], output_dir: Path, config: dict[str, Any], jcfg: JevMiniSweConfig,
                     s1: DecisionBackend | None, progress_manager: Any = None,
                     env_factory: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
                     labeler: Labeler | None = None) -> dict[str, Any]:
    """Run one SWE-bench instance (mirrors ``minisweagent.run.benchmarks.swebench.process_instance``) and
    write ``preds.json``, ``<iid>/<iid>.traj.json`` (mini-swe format + ``info.jev``), ``<iid>/<iid>.jev.json``
    and a line in ``trajectories.jsonl``. ``env_factory(config, instance)`` defaults to mini-swe's
    ``get_sb_environment`` (Docker image per instance)."""
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import (get_sb_environment, remove_from_preds_file,
                                                      update_preds_file)

    iid = instance["instance_id"]
    inst_dir = output_dir / iid
    remove_from_preds_file(output_dir / "preds.json", iid)
    (inst_dir / f"{iid}.traj.json").unlink(missing_ok=True)
    model_cfg = config.get("model", {})
    if isinstance(model_cfg.get("api_url"), str) and "{instance_id}" in model_cfg["api_url"]:
        model_cfg = {**model_cfg, "api_url": model_cfg["api_url"].replace("{instance_id}", iid)}   # E29's proxy
    model = get_model(config=model_cfg)
    cheap = build_cheap_model(model_cfg, jcfg.cheap_model) if (jcfg.cheap_model and
                                                               jcfg.routing.mode == "trigger") else None
    if progress_manager is not None:
        progress_manager.on_instance_start(iid)
        progress_manager.update_instance_status(iid, "Pulling/starting environment")
    agent: JevMiniSweAgent | None = None
    exit_status, result, extra_info, err = None, None, {}, None
    try:
        env = (env_factory or get_sb_environment)(config, instance)
        agent = JevMiniSweAgent(model, env, jev=jcfg, s1=s1, cheap_model=cheap, labeler=labeler,
                                progress_manager=progress_manager, instance_id=iid, **config.get("agent", {}))
        info = agent.run(instance["problem_statement"])
        exit_status, result = info.get("exit_status"), info.get("submission")
    except Exception as e:
        exit_status, result = type(e).__name__, ""
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        rec: dict[str, Any] = {}
        if agent is not None:
            agent.save(inst_dir / f"{iid}.traj.json",
                       {"info": {"exit_status": exit_status, "submission": result, **extra_info},
                        "instance_id": iid})
            rec = agent.save_jev(inst_dir / f"{iid}.jev.json", instance_id=iid, exit_status=exit_status,
                                 submission=result or "", error=err)
        else:  # environment / agent construction failed: still leave a (failed) record for the analysis
            rec = {"format": RECORD_FORMAT, "instance_id": iid, "arm": jcfg.name,
                   "jev_config": jcfg.model_dump(mode="json", exclude={"arms"}),
                   "s1_backend": s1.describe() if s1 is not None and jcfg.uses_s1() else None,
                   "models": {"strong": _model_name(model), "cheap": _model_name(cheap) if cheap else None},
                   "exit_status": exit_status, "submission": "",
                   "result": {"submitted": False, "exit_status": exit_status, "patch_nonempty": False,
                              "resolved": None},
                   "steps": [], "calls": [], "usage": Ledger().summary(), "model_stats": {}, "counters": {},
                   "s1_errors": [], "gold_source": None, "wall_ms": 0.0, "error": err,
                   "mini_version": MINI_VERSION}
            inst_dir.mkdir(parents=True, exist_ok=True)
            (inst_dir / f"{iid}.jev.json").write_text(json.dumps(rec, indent=1, default=str))
        with _FILE_LOCK, open(output_dir / "trajectories.jsonl", "a") as f:
            f.write(json.dumps(to_trajectory(rec).to_dict(), default=str) + "\n")
        name = f"{model.config.model_name}__jev-{jcfg.name}"
        update_preds_file(output_dir / "preds.json", iid, name, result or "")
        if progress_manager is not None:
            progress_manager.on_instance_end(iid, exit_status)
    return rec


def _load_instances(subset: str, split: str, filter_spec: str = "", slice_spec: str = "", shuffle: bool = False,
                    instance_ids: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset
    from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, filter_instances
    instances = list(load_dataset(DATASET_MAPPING.get(subset, subset), split=split))
    if instance_ids:
        keep = [ln.split("#")[0].strip() for ln in Path(instance_ids).read_text().splitlines()]
        keep_set = {k for k in keep if k}
        instances = [i for i in instances if i["instance_id"] in keep_set]
        missing = keep_set - {i["instance_id"] for i in instances}
        if missing:
            raise ValueError(f"{len(missing)} ids not in {subset}/{split}: {sorted(missing)[:5]} ...")
    return filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)


def run_batch(args: argparse.Namespace) -> Path:
    import concurrent.futures

    from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
    from minisweagent.utils.log import add_file_handler, logger
    from rich.live import Live

    config = load_config(args.config)
    if args.model:
        config.setdefault("model", {})["model_name"] = args.model
    if args.model_class:
        config.setdefault("model", {})["model_class"] = args.model_class
    if args.environment_class:
        config.setdefault("environment", {})["environment_class"] = args.environment_class
    jspec = copy.deepcopy(config.get("jev", {}))
    if args.cheap_model:
        jspec.setdefault("cheap_model", {})["model_name"] = args.cheap_model
    jcfg = JevMiniSweConfig.for_arm(jspec, args.arm)
    if args.name:
        jcfg.name = args.name
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    add_file_handler(out / "minisweagent.log")
    (out / "jev_arm.json").write_text(json.dumps({"arm": jcfg.name, "jev": jcfg.model_dump(mode="json",
                                                  exclude={"arms"}), "config_spec": args.config,
                                                  "model": config.get("model", {}).get("model_name")}, indent=1))
    instances = _load_instances(args.subset, args.split, args.filter, args.slice, args.shuffle, args.instance_ids)
    if not args.redo_existing and (out / "preds.json").exists():
        done = set(json.loads((out / "preds.json").read_text()))
        instances = [i for i in instances if i["instance_id"] not in done]
    logger.info(f"[jev:{jcfg.name}] {jcfg.short()} -- running {len(instances)} instances -> {out}")
    s1 = make_backend(jcfg.s1, cache=jcfg.s1_cache) if jcfg.uses_s1() else None
    labeler = heuristic_labeler if jcfg.gold == "heuristic" else None
    progress = RunBatchProgressManager(len(instances), out / f"exit_statuses_{time.time()}.yaml")

    def _one(inst: dict[str, Any]) -> None:
        process_instance(inst, out, config, jcfg, s1, progress, labeler=labeler)

    with Live(progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_one, inst): inst["instance_id"] for inst in instances}
            try:
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:  # pragma: no cover - mirrors mini-swe
                        logger.error(f"Error in future for instance {futs[fut]}: {e}", exc_info=True)
                        progress.on_uncaught_exception(futs[fut], e)
            except KeyboardInterrupt:
                for fut in futs:
                    fut.cancel()
                raise
    collect_run(out, out=out / "trajectories.collected.jsonl")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m jevharness.adapters.mini_swe",
                                 description="Jev System-1 hooks in mini-swe-agent on SWE-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--subset", default="verified", help="verified | lite | full | ... or an HF dataset path")
        p.add_argument("--split", default="test")
        p.add_argument("--slice", default="", help="e.g. 0:100 (applied after --shuffle / --filter)")
        p.add_argument("--filter", default="", help="regex on instance ids")
        p.add_argument("--shuffle", action="store_true", help="mini-swe's deterministic shuffle (seed 42)")
        p.add_argument("--instance-ids", default=None, help="file with one instance id per line")

    p = sub.add_parser("run", help="run one arm on SWE-bench (batch)")
    data_args(p)
    p.add_argument("-c", "--config", action="append", default=None,
                   help=f"mini-swe config specs (files or key=value), merged in order; default {DEFAULT_CONFIG}")
    p.add_argument("--arm", default=None, help="name in jev.arms (default: the top-level jev hooks)")
    p.add_argument("--name", default=None, help="label for this arm in outputs (default: --arm)")
    p.add_argument("-o", "--output", required=True, help="output directory for this arm")
    p.add_argument("-w", "--workers", type=int, default=1)
    p.add_argument("-m", "--model", default=None, help="strong model (overrides model.model_name)")
    p.add_argument("--cheap-model", default=None, help="routing target (overrides jev.cheap_model.model_name)")
    p.add_argument("--model-class", default=None)
    p.add_argument("--environment-class", default=None, help="docker (default) | singularity | ...")
    p.add_argument("--redo-existing", action="store_true")

    p = sub.add_parser("ids", help="print / write the instance ids of a subset selection")
    data_args(p)
    p.add_argument("-o", "--output", default=None)

    p = sub.add_parser("arms", help="list the arms defined in a config")
    p.add_argument("-c", "--config", action="append", default=None)

    p = sub.add_parser("collect", help="per-instance .jev.json (+ eval report) -> Trajectory JSONL")
    p.add_argument("run_dir")
    p.add_argument("--report", action="append", default=[], help="SWE-bench harness / sb-cli report JSON")
    p.add_argument("--stop-gold-from-eval", action="store_true")
    p.add_argument("-o", "--output", default=None)

    args = ap.parse_args(argv)
    if getattr(args, "config", None) is None and args.cmd in ("run", "arms"):
        args.config = [str(DEFAULT_CONFIG)]
    if args.cmd == "run":
        run_batch(args)
    elif args.cmd == "ids":
        ids = [i["instance_id"] for i in _load_instances(args.subset, args.split, args.filter, args.slice,
                                                          args.shuffle, args.instance_ids)]
        text = "\n".join(ids) + "\n"
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text)
        print(text, end="")
    elif args.cmd == "arms":
        print("\n".join((load_config(args.config).get("jev", {}).get("arms") or {}).keys()))
    elif args.cmd == "collect":
        trajs = collect_run(args.run_dir, args.report, args.output, args.stop_gold_from_eval)
        from ..bench.metrics import episode_table
        print(json.dumps(episode_table(trajs), indent=1))


__all__ = ["ACTION_TYPES", "HOOKS", "HookConfig", "JevMiniSweAgent", "JevMiniSweConfig", "OpenRouterTimeoutModel",
           "build_cheap_model",
           "classify_command", "collect_run", "fix_verified", "heuristic_labeler", "is_destructive",
           "is_edit_command", "is_run_command", "is_submit_command", "load_resolved_ids", "process_instance",
           "to_trajectory"]

if __name__ == "__main__":
    main()
