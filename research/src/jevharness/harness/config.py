"""The *design space* studied in the paper: WHERE a System-1 model sits in the harness x HOW it
interacts with the System-2 LLM.

WHERE (decision points)          HOW (integration modes)
-----------------------          -----------------------------------------------------------
tool_select                      off      -- LLM-only harness (baseline)
tool_params                      replace  -- S1 decides, the LLM is not consulted
memory_write                     cascade  -- S1 decides if confidence >= tau, else defer to LLM
memory_read                      assist   -- S1 prunes the candidate set to top-k, the LLM decides
reflection                       guard    -- the LLM proposes, S1 verifies (p_yes < tau -> retry)
stop                             trigger  -- S1 decides whether to invoke an expensive module
routing                                     (reflection, memory, bigger model, early stop)
action (action-only envs)        shadow   -- measurement only: the harness behaves exactly like the
                                             LLM-only harness at this point (the default resolver decides),
                                             S1 is asked the same typed question and its answer, the LLM's
                                             own decision and the gold label are logged side by side
                                             (``DecisionLog.shadow``); its calls are tagged
                                             ``meta["shadow"]`` in the ledger and kept out of the episode
                                             cost (``usage["shadow"]`` holds the overhead)

At tool choice, tool arguments and action (E6f), ``trigger`` decides whether an optional LLM verification pass runs on
the LLM's proposal (the LLM re-reads the state and its proposal and confirms it or revises it once) and ``always`` is
the non-S1 reference that verifies every proposal.

Not every (where, how) cell is meaningful; ``VALID_MODES`` lists the ones implemented (``memory_write: shadow``
asks the write gate, logs it and stores nothing), ``UNDEFINED_CELLS`` the cells of the 8 x 5 grid that are undefined
or equivalent to an implemented cell, with the reason, ``E6B_CELLS`` the cells added by the exploratory row E6b,
``E6C_CELLS`` the action guard added by E6c, ``E6D_CELLS`` the seven cells added by E6d and ``E6E_CELLS`` the five
REPLACE cells run by E6e (each equal to another cell at a fixed threshold: ``EQUIVALENT_NOTES``) and
``E6F_CELLS`` the three verification triggers defined by E6f (tool choice, tool arguments, action), which left
``UNDEFINED_CELLS`` empty. ``HarnessConfig.measure`` (docs/META_GOLD_SPEC.md §5.2) adds the
deferred measurement of the measurement arms: counterfactual System-2 branches and wrong-proposal probes, run after
the episode on snapshots and kept out of the episode cost.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import model_validator, BaseModel, Field


class Mode(str, Enum):
    OFF = "off"
    REPLACE = "replace"
    CASCADE = "cascade"
    ASSIST = "assist"
    GUARD = "guard"
    TRIGGER = "trigger"
    ALWAYS = "always"      # non-S1 reference: e.g. reflect every step / store every observation
    LLM = "llm"            # non-S1 reference: the LLM makes this meta-decision itself
    SHADOW = "shadow"      # measurement only: default resolver decides, S1's answer is logged (never acted on)


class DP(str, Enum):
    TOOL_SELECT = "tool_select"
    TOOL_PARAMS = "tool_params"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    REFLECTION = "reflection"
    STOP = "stop"
    ROUTING = "routing"
    ACTION = "action"


VALID_MODES: dict[DP, set[Mode]] = {
    DP.TOOL_SELECT: {Mode.OFF, Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER, Mode.ALWAYS,
                     Mode.SHADOW},     # trigger / always: E6f's verification pass
    DP.TOOL_PARAMS: {Mode.OFF, Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER, Mode.ALWAYS,
                     Mode.SHADOW},
    DP.MEMORY_WRITE: {Mode.OFF, Mode.ALWAYS, Mode.LLM, Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD,
                      Mode.TRIGGER, Mode.SHADOW},   # shadow: ask, log, store nothing
    DP.MEMORY_READ: {Mode.OFF, Mode.ALWAYS, Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER},
    DP.REFLECTION: {Mode.OFF, Mode.ALWAYS, Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER,
                    Mode.SHADOW},
    DP.STOP: {Mode.OFF, Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER, Mode.SHADOW},
    DP.ROUTING: {Mode.OFF, Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER, Mode.SHADOW},
    DP.ACTION: {Mode.OFF, Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER, Mode.ALWAYS, Mode.SHADOW},
}
# The cells added by the exploratory row E6b (2026-10-06; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6b",
# results/explore/e6b_modes/cells.md). Semantics (agent.py / memory.py):
#   tool_params  assist   System-1 shortlists the top-k values of each bounded / copy argument of a System-1-chosen
#                         call; the LLM fills the whole call in one forced call (string arguments restricted to the
#                         shortlist by an enum, other types by a listed shortlist)
#   tool_params  guard    the LLM writes the call; System-1 answers each bounded / copy argument's Choice and vetoes the
#                         call once when it is confident (>= tau) of a different value; the LLM rewrites it once
#   memory_write cascade  System-1's yes / no when its confidence >= tau, else the LLM gate (``llm``)
#   memory_write guard    the LLM gate proposes; System-1 can veto a proposed write (p_yes < tau): not stored
#   memory_read  cascade  System-1's top-k selection when its top memory's confidence >= tau, else every recent memory
#                         (``always``: the LLM reads them all)
#   memory_read  trigger  System-1 decides whether to read memory this step (yes: the recent memories, as ``always``)
#   reflection   cascade  System-1's yes / no when confident, else the LLM answers the same question (an LLM gate)
#   stop         cascade  System-1's stop / continue when confident (the guard question at a finish proposal, the
#                         trigger question otherwise), else the LLM's own decision (= trigger at tau + guard at 1 - tau)
#   routing      cascade  System-1's strong / cheap when confident, else the cheap LLM answers the routing question
E6B_CELLS: frozenset[tuple[DP, Mode]] = frozenset({
    (DP.TOOL_PARAMS, Mode.ASSIST), (DP.TOOL_PARAMS, Mode.GUARD), (DP.MEMORY_WRITE, Mode.CASCADE),
    (DP.MEMORY_WRITE, Mode.GUARD), (DP.MEMORY_READ, Mode.CASCADE), (DP.MEMORY_READ, Mode.TRIGGER),
    (DP.REFLECTION, Mode.CASCADE), (DP.STOP, Mode.CASCADE), (DP.ROUTING, Mode.CASCADE)})
# The cells added by the exploratory row E6d (2026-10-06; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6d",
# results/explore/e6d_modes/cells.md). Semantics (agent.py / memory.py):
#   memory_write / reflection / routing  assist  "advise": System-1 answers the yes / no question; the LLM gate asks the
#                         LLM the same question with System-1's answer and confidence as a hint; the LLM decides
#                         (routing: the cheap LLM). The binary analogue of the shortlist.
#   stop         assist   System-1 answers the stop-trigger question before the step; its answer and confidence go into
#                         the LLM's turn as a hint, and the LLM decides by its own call (finish or another tool)
#   reflection / routing  guard  the LLM gate (E6b's cascade fallback) proposes reflect / escalate; System-1 vetoes a
#                         proposal once when p_yes < tau (not executed), as the memory-write guard
#   memory_read  guard    the LLM selects the stored memories its next step needs (one call over the stored items);
#                         System-1 is asked the tool-guard-style question about that selection and vetoes it when
#                         p_yes < tau; the LLM selects once more with the rejection note, and that selection is
#                         injected unchecked
E6D_CELLS: frozenset[tuple[DP, Mode]] = frozenset({
    (DP.MEMORY_WRITE, Mode.ASSIST), (DP.REFLECTION, Mode.ASSIST), (DP.STOP, Mode.ASSIST), (DP.ROUTING, Mode.ASSIST),
    (DP.REFLECTION, Mode.GUARD), (DP.ROUTING, Mode.GUARD), (DP.MEMORY_READ, Mode.GUARD)})
# The cells added by the exploratory row E6e (2026-10-06; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6e"):
# REPLACE at the yes / no points and at memory read, run as their own arms although each behaves as another cell at
# a fixed threshold (``EQUIVALENT_NOTES``): System-1's argmax decides alone, the LLM is never asked.
#   memory_write / reflection / routing  replace  yes iff p_yes >= 0.5 (= the trigger at tau 0.5)
#   stop         replace  System-1 decides stop / continue at every step: the guard question at a finish proposal (no
#                         = reject), the trigger question otherwise (yes = early stop) (= the cascade at tau 0.5)
#   memory_read  replace  System-1's single top memory (or none) is injected (= the assist with k = 1)
E6E_CELLS: frozenset[tuple[DP, Mode]] = frozenset({
    (DP.MEMORY_WRITE, Mode.REPLACE), (DP.MEMORY_READ, Mode.REPLACE), (DP.REFLECTION, Mode.REPLACE),
    (DP.STOP, Mode.REPLACE), (DP.ROUTING, Mode.REPLACE)})
# Implemented cells that behave exactly as another cell at a fixed threshold (the figure notes the cell they equal)
EQUIVALENT_NOTES: dict[tuple[DP, Mode], str] = {
    (DP.MEMORY_WRITE, Mode.REPLACE): "trigger at tau = 0.5: System-1 alone deciding store / skip is the write trigger at 0.5",
    (DP.MEMORY_READ, Mode.REPLACE): "assist with k = 1: the read assist already injects System-1's selection without "
                                    "an LLM choice",
    (DP.REFLECTION, Mode.REPLACE): "trigger at tau = 0.5: System-1 alone deciding reflect / skip is the trigger at 0.5",
    (DP.STOP, Mode.REPLACE): "cascade at tau = 0.5: System-1 alone deciding stop / continue at every step is the "
                             "guard and the trigger together at 0.5",
    (DP.ROUTING, Mode.REPLACE): "trigger at tau = 0.5: System-1 alone choosing the strong or the cheap LLM is the "
                                "routing trigger at 0.5",
}
# The cells added by the exploratory row E6f (2026-10-07; docs/PREREG_A7_STAGE2.md Deviations "Exploratory row E6f",
# results/explore/e6f_modes/cells.md). Until E6f these three were undefined: a trigger decides whether an optional
# module runs, and the tool, its arguments and the action are needed at every step. E6f gives each point an optional
# module, an LLM verification pass on the LLM's proposal. Semantics (agent.py ``_verify_call`` / ``_verify_action``):
#   tool_select / tool_params / action  trigger  the LLM proposes as usual (tool call, or admissible action); System-1
#                         is asked whether the proposal needs a second look (prompts.q_verify_tool / q_verify_args /
#                         q_verify_action); if p_yes >= tau the LLM runs ONE verification call (prompts.verify_*_note:
#                         it re-reads the state and its proposal and confirms it or revises it once; at tool arguments
#                         the call is forced to the proposed tool), else the proposal runs unchecked. The final
#                         proposal is executed. Unlike the guard, System-1 never judges or vetoes the proposal; it only
#                         decides when the LLM checks itself.
#   same points           always   the non-S1 reference: the verification call runs on every proposal
E6F_CELLS: frozenset[tuple[DP, Mode]] = frozenset({
    (DP.TOOL_SELECT, Mode.TRIGGER), (DP.TOOL_PARAMS, Mode.TRIGGER), (DP.ACTION, Mode.TRIGGER)})
# The (decision point, integration mode) cells of the design-space grid that are NOT implemented, with the reason:
# ("undefined", why the mode has no meaning there) or ("equivalent", the implemented cell it would duplicate). With
# VALID_MODES and ACTION_PENDING this covers all 8 x 5 cells (tests/test_e6b_modes.py). Empty since E6f defined the
# last three (the triggers at tool choice, tool arguments and action: ``E6F_CELLS``).
UNDEFINED_CELLS: dict[tuple[DP, Mode], tuple[str, str]] = {}
# meaningful but not implemented here. Empty since the exploratory row E6c (2026-10-06; docs/PREREG_A7_STAGE2.md
# Deviations "Exploratory row E6c") implemented its one cell, the action guard (``E6C_CELLS``).
ACTION_PENDING: dict[tuple[DP, Mode], str] = {}
# The cell added by E6c. Semantics (agent.py ``_guard_action``): action-only envs; the LLM proposes an admissible
# action, System-1 is asked the tool-guard question about it (``prompts.q_action_guard``) and vetoes it when
# p_yes < tau; the LLM regenerates once with the rejection note (``guard_max_retries``) and the regenerated action is
# executed unchecked, as the tool guard does. Gold: the proposal equals the env's single gold action (a lower bound).
E6C_CELLS: frozenset[tuple[DP, Mode]] = frozenset({(DP.ACTION, Mode.GUARD)})
# modes in which System-1 is queried (shadow: queried but never acted on)
S1_MODES = frozenset({Mode.REPLACE, Mode.CASCADE, Mode.ASSIST, Mode.GUARD, Mode.TRIGGER, Mode.SHADOW})


class DPConfig(BaseModel):
    mode: Mode = Mode.OFF
    tau: float = 0.8           # confidence threshold for cascade / guard / trigger
    k: int = 3                 # top-k for assist
    extra: dict[str, Any] = Field(default_factory=dict)   # mode-specific keys (e.g. cascade `always_defer`, A6.11 (c))

    @model_validator(mode="before")
    @classmethod
    def _unknown_keys_into_extra(cls, data: Any) -> Any:
        """YAML arms write mode-specific keys next to `mode` / `tau` / `k` (`{mode: cascade, tau: 0.99,
        always_defer: true}`); they land in `extra` instead of being dropped silently."""
        if isinstance(data, dict):
            known = {"mode", "tau", "k", "extra"}
            unknown = {k: v for k, v in data.items() if k not in known}
            if unknown:
                data = {k: v for k, v in data.items() if k in known}
                data["extra"] = {**unknown, **(data.get("extra") or {})}
        return data


class CFConfig(BaseModel):
    """Deferred one-step counterfactual System-2 calls (docs/META_GOLD_SPEC.md §4), run after the episode on
    snapshots taken at decision time; measurement only (ledger ``meta.measure = "cf"``, out of the episode cost)."""
    reflection: bool = False           # verdict_alt: reflect (same prompt), then the step's tool decision
    large_llm: str | None = None       # verdict_large: this LLM (a name of the runner's LLM pool) decides the step
    stop_reject: bool = False          # verdict_reject: the rejection note, then the next tool decision (every finish)
    placebo: bool = True               # verdict_placebo: the default decision again, at every state with a branch
    sample_negatives: float = 0.2      # seeded share of negative states that get the branches (IPW 1/rate)
    fired_steps: bool = False          # live trigger arms: the default decision at fired steps (label a0 there)


class ProbeConfig(BaseModel):
    """Wrong-proposal probes for guard FAR (docs/META_GOLD_SPEC.md §6): phantom proposals scored post-episode with
    the live guard question, never executed, stored in ``Trajectory.probes``. Measurement (shadow-only) arms only."""
    stop: list[str] = Field(default_factory=list)          # natural | forced_premature | transplant | T5 | T6
    tool_select: list[str] = Field(default_factory=list)   # natural | transplant | T1 | T2 | T3 | T4
    rho_edit: float = 0.5              # per step: one T1-T4 tool probe
    rho_t5: float = 0.25               # per non-final step: one T5 probe
    rho_transplant: float = 0.25       # non-finish steps that get a transplant proposal (every finish state does)
    transplant_llm: str | None = None  # name of the runner's LLM pool (Flash arms: Flash-Lite T=0; Lite arms: T=1)
    transplant_temperature: float = 0.0
    tau_ref: float = 0.5               # tau of the live guard arm whose FAR/FRR is reported
    max_resamples: int = 5             # edits labelled right are resampled, then counted as "no valid probe"


class MeasureConfig(BaseModel):
    """``HarnessConfig.measure``. Empty (the default) = golds only, no counterfactual calls and no probes."""
    seed: int = 0
    cf: CFConfig | None = None
    probes: ProbeConfig | None = None

    def active(self) -> bool:
        return self.cf is not None or self.probes is not None


class HarnessConfig(BaseModel):
    name: str = "llm_only"
    max_steps: int = 30
    history_window: int = 8          # steps of history rendered into the S1 state
    obs_truncate_chars: int = 2000
    s1_state_max_chars: int = 12000
    final_tool: str = "finish"
    guard_max_retries: int = 1
    tool_select: DPConfig = Field(default_factory=DPConfig)
    tool_params: DPConfig = Field(default_factory=DPConfig)
    memory_write: DPConfig = Field(default_factory=DPConfig)
    memory_read: DPConfig = Field(default_factory=DPConfig)
    reflection: DPConfig = Field(default_factory=DPConfig)
    stop: DPConfig = Field(default_factory=DPConfig)
    routing: DPConfig = Field(default_factory=DPConfig)
    action: DPConfig = Field(default_factory=DPConfig)
    measure: MeasureConfig = Field(default_factory=MeasureConfig)

    def dp(self, dp: DP) -> DPConfig:
        return getattr(self, dp.value)

    def validate_modes(self) -> "HarnessConfig":
        for dp in DP:
            m = self.dp(dp).mode
            if m not in VALID_MODES[dp]:
                why = UNDEFINED_CELLS.get((dp, m)) or (("not implemented", ACTION_PENDING[(dp, m)])
                                                       if (dp, m) in ACTION_PENDING else None)
                raise ValueError(f"mode {m.value!r} not supported at decision point {dp.value!r}"
                                 + (f" ({why[0]}: {why[1]})" if why else ""))
        if self.tool_params.mode == Mode.GUARD and self.tool_select.mode in (Mode.REPLACE, Mode.CASCADE):
            # the argument guard checks the LLM's own calls; a System-1-chosen call has System-1's arguments
            raise ValueError("tool_params=guard checks the calls the LLM writes; combine it with tool_select off, "
                             "assist, guard or shadow")
        if self.tool_params.mode in (Mode.TRIGGER, Mode.ALWAYS) and self.tool_select.mode in (Mode.REPLACE,
                                                                                             Mode.CASCADE):
            # E6f: the verification pass re-checks the LLM's own proposal, not a System-1-chosen call
            raise ValueError(f"tool_params={self.tool_params.mode.value} verifies the calls the LLM writes; combine it "
                             "with tool_select off, assist, guard, trigger, always or shadow")
        if self.measure.probes is not None:
            # META_GOLD_SPEC §6.4: probes live only in measurement arms, never inside a primary arm. A measurement arm
            # is the LLM-only harness with System-1 logged: every decision point off or shadow. Non-System-1 live
            # modes count too (reflection always, memory always / llm): such an arm is a reference or control (e.g.
            # reflect_always), and the score card would drop it from Holm / Pareto / the reference role as "shadow".
            live = [dp.value for dp in DP if self.dp(dp).mode not in (Mode.OFF, Mode.SHADOW)]
            if live:
                raise ValueError(f"measure.probes runs only in measurement (shadow-only) arms; {self.name!r} has "
                                 f"live decision points {live}")
            if not self.shadow_points():
                raise ValueError(f"measure.probes runs only in measurement arms; {self.name!r} shadows no decision "
                                 "point (it is the LLM-only harness)")
        return self

    def is_measurement_arm(self) -> bool:
        """Every decision point off or shadow, at least one shadow: the LLM-only harness with System-1 logged."""
        return bool(self.shadow_points()) and all(self.dp(dp).mode in (Mode.OFF, Mode.SHADOW) for dp in DP)

    def uses_s1(self) -> bool:
        return any(self.dp(dp).mode in S1_MODES for dp in DP)

    def shadow_points(self) -> list[DP]:
        """Decision points run in ``shadow`` mode (S1 logged, never acted on)."""
        return [dp for dp in DP if self.dp(dp).mode == Mode.SHADOW]

    def short(self) -> str:
        parts = []
        for dp in DP:
            c = self.dp(dp)
            if c.mode == Mode.SHADOW:
                parts.append(f"{dp.value}:shadow")
            elif c.mode != Mode.OFF:
                arg = f"k={c.k}" if c.mode == Mode.ASSIST else f"tau={c.tau}"
                parts.append(f"{dp.value}:{c.mode.value}({arg})")
        return ",".join(parts) or "llm_only"
