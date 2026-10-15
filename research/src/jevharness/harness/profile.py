"""JevAgentHarness profiles: System-1 and System-2 sharing control per decision point (docs/JEVAGENT_HARNESS.md §3).

A *profile* is a YAML file (``configs/harness/*.yaml``) that says, for each System-2 model, how every decision point of
``harness/config.py`` is shared between System-1 and the LLM: a mode, a threshold τ, a top-k and mode-specific keys,
written exactly as an experiment arm writes them::

    kind: jevagent_profile
    version: 1
    name: jevagent_default
    status: development            # development | pending | selected
    system2:
      gemini-3.5-flash:            # the System-2 model id (an arm picks its section with ``profile_s2``)
        decision_points:           # omitted decision points are ``off`` (the LLM decides)
          tool_select: {mode: cascade, tau: 1.0}
          tool_params: {mode: cascade, tau: 0.99}
          stop: {mode: trigger, tau: 0.9}
        harness: {}                # optional HarnessConfig scalars (guard_max_retries, max_steps, ...)
        best_single: {arm: jev_ts_cascade_tauhat, decision_points: {...}}   # optional (A7.1 (g) 3)
        backends:                  # optional per-System-1 threshold overrides (A7.1 (f))
          jevagent_rl: {tool_select: {tau: 0.97}, tool_params: {tau: 0.95}}
        alfworld: {decision_points: {...}}   # optional rendering for action environments (A7.1 (i))
        skip: "reason"             # optional: arms using this section are not run (e.g. an empty profile)

An experiment arm loads a profile with ``profile: <path>`` and optionally ``profile_s2`` (default: the arm's LLM model
id), ``profile_part: best_single``, ``profile_backend: <name>`` and ``profile_env: alfworld``; such an arm sets no
decision point itself (:func:`resolve_arm`). A ``status: pending`` profile, or a section / part with ``skip``, makes the
runner skip the arm with the reason (:func:`skip_reason`), so a profile that does not exist yet can never run as the
LLM-only harness by accident. Arms without ``profile`` are untouched.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import DP, HarnessConfig

ROOT = Path(__file__).resolve().parents[3]
PROFILE_KEYS = ("profile", "profile_s2", "profile_part", "profile_backend", "profile_env")
DP_NAMES = tuple(dp.value for dp in DP)
# decision-point groups (PREREG_A7_STAGE2 A7.1 (a)): the selection chooses one setting per group
DP_GROUPS: dict[str, tuple[str, ...]] = {
    "tool": ("tool_select", "tool_params"), "stop": ("stop",), "reflection": ("reflection",),
    "routing": ("routing",), "memory": ("memory_write", "memory_read"), "action": ("action",)}
# the decision points an action-environment loop consults (harness/agent.py::run with admissible actions)
ACTION_LOOP_DPS = ("action", "reflection", "routing", "memory_write", "memory_read")
STATUSES = ("development", "pending", "selected")
# keys of a `backends.<name>` block that are not decision points: `skip` (the fitted thresholds delegate nothing with
# that System-1, PREREG_A7_STAGE2 A7.18 (f)), `fitted_from` (the calibration fits' paths and sha256) and `notes`
BACKEND_META_KEYS = ("skip", "fitted_from", "notes")
ENVS = ("alfworld",)


class ProfileError(ValueError):
    pass


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    return ROOT / p


def load_profile(path: str | Path) -> dict[str, Any]:
    """Read and validate a profile file; adds ``_path`` and ``_sha256`` (provenance)."""
    p = _resolve_path(path)
    if not p.exists():
        raise ProfileError(f"profile {path!s} not found")
    raw = p.read_bytes()
    data = yaml.safe_load(raw) or {}
    if data.get("kind") != "jevagent_profile":
        raise ProfileError(f"{path}: not a JevAgentHarness profile (kind: jevagent_profile)")
    if data.get("status") not in STATUSES:
        raise ProfileError(f"{path}: status must be one of {STATUSES}, got {data.get('status')!r}")
    secs = data.get("system2") or {}
    if not isinstance(secs, dict):
        raise ProfileError(f"{path}: `system2` must map System-2 model ids to sections")
    for s2, sec in secs.items():
        if not isinstance(sec, dict):
            raise ProfileError(f"{path}: section {s2!r} must be a mapping")
        _check_dps(sec.get("decision_points") or {}, f"{path}: {s2}.decision_points")
        bs = sec.get("best_single")
        if bs is not None:
            _check_dps((bs or {}).get("decision_points") or {}, f"{path}: {s2}.best_single")
        for env in ENVS:
            if sec.get(env) is not None:
                _check_dps((sec.get(env) or {}).get("decision_points") or {}, f"{path}: {s2}.{env}")
        for be, ov in (sec.get("backends") or {}).items():
            for dp in ov or {}:
                if dp not in DP_NAMES and dp not in BACKEND_META_KEYS:
                    raise ProfileError(f"{path}: {s2}.backends.{be}: unknown decision point {dp!r}")
    data = dict(data)
    data["_path"] = str(path)
    data["_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def _check_dps(dps: Mapping[str, Any], where: str) -> None:
    if not isinstance(dps, Mapping):
        raise ProfileError(f"{where}: must map decision points to settings")
    for dp, cfg in dps.items():
        if dp not in DP_NAMES:
            raise ProfileError(f"{where}: unknown decision point {dp!r} (one of {DP_NAMES})")
        if not isinstance(cfg, Mapping) or "mode" not in cfg:
            raise ProfileError(f"{where}.{dp}: a setting needs at least a `mode`")
    # the modes must be valid at their decision points (harness/config.py VALID_MODES)
    try:
        HarnessConfig(name="profile_check", **{k: dict(v) for k, v in dps.items()}).validate_modes()
    except Exception as e:  # noqa: BLE001
        raise ProfileError(f"{where}: {e}") from e


def section(profile: Mapping[str, Any], s2: str) -> dict[str, Any]:
    secs = profile.get("system2") or {}
    if s2 not in secs:
        raise ProfileError(f"profile {profile.get('_path', profile.get('name'))}: no section for System-2 {s2!r} "
                           f"(has {sorted(secs)})")
    return secs[s2] or {}


def decision_points(profile: Mapping[str, Any], s2: str, part: str | None = None, backend: str | None = None,
                    env: str | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str | None]:
    """(decision point -> setting, HarnessConfig scalar overrides, skip reason or None) for one arm.

    ``part="best_single"`` takes the profile restricted to its top group (A7.11 (a)); ``part="only_<group>"`` /
    ``"minus_<group>"`` keep / drop one active group (A7.11 (c); ``minus`` runs only when the section's ``logo`` map
    names the group without a skip reason); ``env="alfworld"`` its action-
    environment rendering (else the section's decision points restricted to :data:`ACTION_LOOP_DPS`); ``backend``
    applies that System-1's threshold overrides (``backends.<name>``) to the decision points they name."""
    if profile.get("status") == "pending":
        return {}, {}, f"profile {profile.get('_path')} is pending (written by scripts/select_profile.py, A7.1 (h))"
    sec = section(profile, s2)
    if sec.get("skip"):
        return {}, {}, str(sec["skip"])
    harness = dict(sec.get("harness") or {})
    group_part = None
    if part is not None and part != "best_single":
        kind, _, group_part = part.partition("_")
        if kind not in ("only", "minus") or group_part not in DP_GROUPS:
            raise ProfileError(f"profile_part must be best_single, only_<group> or minus_<group> "
                               f"(groups {sorted(DP_GROUPS)}), got {part!r}")
    if env not in (None, *ENVS):
        raise ProfileError(f"profile_env must be one of {ENVS}, got {env!r}")
    full = copy.deepcopy(dict(sec.get("decision_points") or {}))
    if part == "best_single":
        # PREREG_A7_STAGE2 A7.11 (a): the profile restricted to its top group g*; the harness scalars are the
        # section's, never the part's, so nothing but the decision points differs from the composed arm
        bs = sec.get("best_single")
        if not bs:
            return {}, {}, f"profile section {s2!r} records no best_single arm"
        if bs.get("skip"):
            return {}, {}, str(bs["skip"])
        dps = copy.deepcopy(dict(bs.get("decision_points") or {}))
    elif group_part is not None:
        # A7.11 (c): only_<g> keeps group g alone, minus_<g> drops it (leave-one-group-out); both need g active,
        # and minus_<g> runs only when the profile's `logo` list names g without a skip reason
        if group_part not in active_groups(full):
            return {}, {}, f"group {group_part!r} is not active in the profile's {s2!r} section"
        own = DP_GROUPS[group_part]
        if part.startswith("only_"):
            dps = {k: v for k, v in full.items() if k in own}
        else:
            logo = sec.get("logo")
            if not isinstance(logo, Mapping) or group_part not in logo:
                return {}, {}, f"leave-one-group-out of {group_part!r} is not listed in the profile (A7.11 (c))"
            if logo[group_part]:
                return {}, {}, str(logo[group_part])
            dps = {k: v for k, v in full.items() if k not in own}
    elif env is not None:
        rend = sec.get(env)
        if rend is not None:
            if rend.get("skip"):
                return {}, {}, str(rend["skip"])
            dps = copy.deepcopy(dict(rend.get("decision_points") or {}))
            harness.update(rend.get("harness") or {})
        else:
            dps = {k: v for k, v in full.items() if k in ACTION_LOOP_DPS}
    else:
        dps = full
    if backend is not None:
        block = (sec.get("backends") or {}).get(backend) or {}
        if block.get("skip"):
            return {}, {}, str(block["skip"])
        for dp, upd in block.items():
            if dp in dps:
                dps[dp] = {**dps[dp], **(upd or {})}
    if not dps and env is not None:
        return {}, harness, f"the profile's {env} rendering delegates no decision point (A7.1 (i))"
    return dps, harness, None


def _arm_llm_model(arm: Mapping[str, Any], exp: Mapping[str, Any]) -> str | None:
    if arm.get("llm"):
        spec = (exp.get("llms") or {}).get(arm["llm"]) or {}
    else:
        spec = exp.get("llm") or {}
    return spec.get("model") if isinstance(spec, Mapping) else None


def resolve_arm(arm: Mapping[str, Any], exp: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """An arm dict with ``profile`` -> (the arm with the profile's decision points and harness scalars filled in and the
    ``profile*`` keys removed, provenance {path, sha256, status, s2, part, backend, env, skip, decision_points}).
    Arms without ``profile`` are returned unchanged with ``None``."""
    if "profile" not in arm:
        stray = [k for k in PROFILE_KEYS if k in arm]
        if stray:
            raise ProfileError(f"arm {arm.get('name')!r}: {stray} without `profile`")
        return dict(arm), None
    own = [k for k in DP_NAMES if k in arm]
    if own:
        raise ProfileError(f"arm {arm.get('name')!r} names a profile and also sets {own}: an arm takes its decision "
                           "points from the profile only")
    prof = load_profile(arm["profile"])
    s2 = arm.get("profile_s2") or _arm_llm_model(arm, exp)
    if not s2:
        raise ProfileError(f"arm {arm.get('name')!r}: cannot tell its System-2 model; set `profile_s2`")
    dps, harness, skip = decision_points(prof, s2, part=arm.get("profile_part"), backend=arm.get("profile_backend"),
                                         env=arm.get("profile_env"))
    out = {k: v for k, v in arm.items() if k not in PROFILE_KEYS}
    for k, v in harness.items():
        out.setdefault(k, v)
    out.update(dps)
    meta = {"path": prof["_path"], "sha256": prof["_sha256"], "name": prof.get("name"), "status": prof.get("status"),
            "s2": s2, "part": arm.get("profile_part"), "backend": arm.get("profile_backend"),
            "env": arm.get("profile_env"), "skip": skip, "decision_points": dps, "harness": harness}
    return out, meta


def skip_reason(arm: Mapping[str, Any], exp: Mapping[str, Any]) -> str | None:
    """Why the runner must not run this arm (a pending profile, a skipped section or part), else None."""
    if "profile" not in arm:
        return None
    return resolve_arm(arm, exp)[1]["skip"]  # type: ignore[index]


def group_of(dp: str) -> str:
    for g, dps in DP_GROUPS.items():
        if dp in dps:
            return g
    raise KeyError(dp)


def active_groups(dps: Mapping[str, Mapping[str, Any]], applicable: tuple[str, ...] | None = None) -> list[str]:
    """The decision-point groups a decision-point mapping does not leave ``off``."""
    gs = sorted({group_of(dp) for dp, c in dps.items() if str(c.get("mode", "off")) != "off"})
    return [g for g in gs if applicable is None or g in applicable]


# ------------------------------------------------------------------ L3 renderings (PREREG_A7_STAGE2 A7.1 (i))
TAU2_SMALL_LLM = "gemini/gemini-3.5-flash-lite"


def tau2_rendering(dps: Mapping[str, Mapping[str, Any]], name: str = "jevagent_best",
                   s1: Mapping[str, Any] | None = None, cache: str = "cache/s1_stage2_tau2.sqlite",
                   small_llm: str = TAU2_SMALL_LLM) -> tuple[dict[str, Any] | None, list[str]]:
    """The τ² (`JevTau2Agent`) config of a JevGym decision-point mapping by the fixed translation of A7.1 (i), and
    notes on what was dropped. None when nothing τ² exposes is delegated."""
    out: dict[str, Any] = {"name": name}
    notes: list[str] = []
    for dp in ("tool_select", "tool_params"):
        if dp in dps and dps[dp].get("mode") not in (None, "off"):
            out[dp] = dict(dps[dp])
    st = dps.get("stop") or {}
    if st.get("mode") == "guard":
        out["stop"] = dict(st)
    elif st.get("mode") not in (None, "off"):
        notes.append(f"stop {st.get('mode')} -> off (tau^2 conversations are ended by the user simulator)")
    rt = dps.get("routing") or {}
    if rt.get("mode") == "trigger":
        out["routing"] = dict(rt)
        out["small_llm"] = small_llm
    elif rt.get("mode") not in (None, "off"):
        notes.append(f"routing {rt.get('mode')} -> off")
    for dp in ("reflection", "memory_write", "memory_read", "action"):
        if (dps.get(dp) or {}).get("mode") not in (None, "off"):
            notes.append(f"{dp} {(dps.get(dp) or {}).get('mode')} -> off (not implemented on tau^2)")
    if len(out) == 1:
        return None, notes
    out["guard_max_retries"] = 1
    out["s1"] = dict(s1 or {"type": "jev", "provider": "openrouter"})
    out["s1_cache"] = cache
    return out, notes


def alfworld_rendering(dps: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The action-environment rendering (A7.1 (i)): the decision points the action loop consults."""
    return {"decision_points": {k: dict(v) for k, v in dps.items() if k in ACTION_LOOP_DPS}}


# ------------------------------------------------------------------ real harnesses (docs/HARNESS_INTEGRATION.md)
def _mode(dps: Mapping[str, Mapping[str, Any]], dp: str) -> str:
    return str((dps.get(dp) or {}).get("mode", "off"))


def pi_rendering(dps: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], dict[str, str], list[str]]:
    """(pi extension files under adapters/pi/, environment variables, notes on what pi cannot deploy) for a profile
    section. pi's extension API cannot answer a model request without the LLM, so replace / cascade at tool_select and
    every tool_params mode are not deployable there (docs/HARNESS_INTEGRATION.md, "Harness affordances")."""
    ext: list[str] = []
    env: dict[str, str] = {}
    notes: list[str] = []

    def tau(dp: str) -> str:
        return f"{float((dps.get(dp) or {}).get('tau', 0.8)):g}"

    ts = _mode(dps, "tool_select")
    if ts == "guard":
        ext.append("jev-tool-gate.ts")
        env.update({"JEV_GATE_MODE": "guard", "JEV_GATE_TAU": tau("tool_select")})
    elif ts == "assist":
        ext.append("jev-active-tools.ts")
        env.update({"JEV_TOOLS_MODE": "assist", "JEV_TOOLS_K": str(int((dps.get("tool_select") or {}).get("k", 3)))})
    elif ts not in ("off", "shadow"):
        notes.append(f"tool_select {ts} -> not deployable on pi (no way to answer a model request without the LLM)")
    if _mode(dps, "tool_params") not in ("off", "shadow"):
        notes.append("tool_params -> not deployable on pi")
    st = _mode(dps, "stop")
    if st in ("guard", "trigger"):
        ext.append("jev-stop-verify.ts")
        env.update({"JEV_STOP_MODE": st, "JEV_STOP_TAU": tau("stop")})
    if _mode(dps, "routing") == "trigger":
        ext.append("jev-router.ts")
        env.update({"JEV_ROUTER_MODE": "trigger", "JEV_ROUTER_TAU": tau("routing")})
        notes.append("routing: set JEV_ROUTER_STRONG / JEV_ROUTER_CHEAP to the System-2 pair")
    if _mode(dps, "reflection") == "trigger":
        ext.append("jev-reflect.ts")
        env.update({"JEV_REFLECT_MODE": "trigger", "JEV_REFLECT_TAU": tau("reflection")})
    mw, mr = _mode(dps, "memory_write"), _mode(dps, "memory_read")
    if mw != "off" or mr != "off":
        ext.append("jev-memory.ts")
        env.update({"JEV_MEMORY_WRITE_MODE": mw, "JEV_MEMORY_READ_MODE": mr, "JEV_MEMORY_TAU": tau("memory_write")})
        if mr == "assist":
            env["JEV_MEMORY_K"] = str(int((dps.get("memory_read") or {}).get("k", 3)))
    if _mode(dps, "action") != "off":
        notes.append("action -> n/a on pi (a tool-calling harness)")
    return ext, env, notes


def mini_swe_rendering(dps: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """(a ``jev.arms.<arm>`` block of adapters/mini_swe/swebench_jev.yaml, notes) for a profile section. mini-SWE-agent's
    action space is one bash command, so tool replace / cascade / assist and tool_params are not deployable; its
    command guard is the tool_select guard."""
    arm: dict[str, Any] = {}
    notes: list[str] = []
    ts = _mode(dps, "tool_select")
    if ts == "guard":
        arm["action_guard"] = {"mode": "guard", "tau": (dps.get("tool_select") or {}).get("tau", 0.5)}
    elif ts not in ("off", "shadow"):
        notes.append(f"tool_select {ts} -> not deployable on mini-SWE-agent (bash-only action space)")
    if _mode(dps, "tool_params") not in ("off", "shadow"):
        notes.append("tool_params -> not deployable on mini-SWE-agent")
    for dp, hook in (("stop", "stop"), ("routing", "routing"), ("reflection", "reflection")):
        m = _mode(dps, dp)
        if m in ("guard", "trigger"):
            arm[hook] = {"mode": m, "tau": (dps.get(dp) or {}).get("tau", 0.5)}
    if _mode(dps, "memory_write") != "off" or _mode(dps, "memory_read") != "off":
        notes.append("memory -> not implemented in the mini-SWE-agent adapter")
    if "routing" in arm:
        notes.append("routing: set jev.cheap_model to the cheap System-2")
    return arm, notes
