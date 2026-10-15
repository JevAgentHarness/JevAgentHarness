# E6b: completing the module map. Which (decision point, mode) cells are meaningful

Exploratory row E6b (2026-10-06; `docs/PREREG_A7_STAGE2.md`, Deviations entry "Exploratory row E6b"). This document
classifies every cell of the 8 × 5 grid of the module map (paper Figure `fig:modulemap`, `scripts/module_map.py`):
eight decision points × the five integration modes REPLACE, CASCADE, ASSIST, GUARD and TRIGGER. The machine-readable
version is `harness/config.py`: `VALID_MODES` (implemented), `E6B_CELLS` (added here), `UNDEFINED_CELLS` (undefined or
equivalent, with the reason) and `ACTION_PENDING`. `tests/test_e6b_modes.py` checks that the four cover all 40 cells
without overlap.

## Definitions

The mode definitions are those of `harness/config.py`:

- **REPLACE.** The decision model decides and the LLM is not consulted.
- **CASCADE.** The decision model decides when its confidence is at least τ; otherwise the LLM decides.
- **ASSIST.** The decision model prunes an option list to its top k; the LLM decides within it.
- **GUARD.** The LLM proposes; the decision model may veto, and the LLM retries.
- **TRIGGER.** The decision model decides whether the harness invokes a module (reflection, memory, a stronger LLM,
  an early stop).

A cell is *meaningful* when the mode's roles (who proposes, who decides, what is invoked) map onto the decision point
without new machinery that exists only to fill the cell. A cell is *equivalent* when its natural implementation is
exactly another implemented cell, so running it would duplicate that cell. A cell is *undefined* when the mode has no
object at that point: no option list to shortlist, no proposal to veto, or no module to invoke.

Two facts decide most of the meta rows. First, for a yes / no decision (memory write, reflection, routing and stop)
TRIGGER already means "the decision model alone decides". REPLACE is therefore the trigger, and ASSIST has nothing to
shortlist. Second, a CASCADE needs an LLM-side way to decide when the model defers. The memory write has one (the
LLM gate, `memory_write: llm`). Stop has one for free (the LLM's own `finish` call). Memory read has the `always`
reference, in which the LLM reads every recent memory. For reflection and routing the deferral asks the agent LLM the
same yes / no question on the same state. This is the memory gate's pattern, and the `s2self` control of E2c already
asks the agent LLM these typed questions.

## The grid

Legend: **run** = implemented and run in Stage 1 (seeds 200000+); **E6b** = newly meaningful, implemented now and run
in E6b (fresh seeds 300800–300999, Flash, exploratory); **=X** = equivalent to cell X; **—** = undefined;
**pending** = meaningful but outside E6b.

| decision point | REPLACE | CASCADE | ASSIST | GUARD | TRIGGER |
|---|---|---|---|---|---|
| tool choice | run | run (C1) | run (C2) | run | — |
| tool arguments | run | run (C1) | **E6b** | **E6b** | — |
| memory write | = TRIGGER | **E6b** | — | **E6b** | run |
| memory read | = ASSIST (k = 1) | **E6b** | run | — | **E6b** |
| reflection | = TRIGGER | **E6b** | — | — | run (C5) |
| stop | = CASCADE (τ = 0.5) | **E6b** | — | run (C6) | run |
| routing | = TRIGGER | **E6b** | — | — | run (C4) |
| action | ALFWorld (E6a) | ALFWorld (E6a) | ALFWorld (E6a) | pending | — |

Counts: 15 cells were implemented before E6b; E6b adds 9, for 24 implemented cells. Of the remaining 16 cells, 10 are
undefined, 5 are equivalent to an implemented cell, and 1 is pending (action GUARD). The action row is filled on
ALFWorld by the E6a row, which owns that row's reader in `scripts/module_map.py`.

## The nine E6b cells

Thresholds are fixed a priori by one rule and nothing is tuned on evaluation data. A cell takes the registered τ of
the same decision point's trigger or guard where one exists, else τ = 0.9. A cascade's τ is a *confidence* threshold
(max(p, 1 − p) ≥ τ). For a yes / no decision a cascade at 0.5 is the replace, and so the trigger, so the triggers'
yes-thresholds of 0.5 cannot carry over to a cascade. Every cascade therefore takes the rule's 0.9; for stop, 0.9 is
also its registered trigger τ. Shortlists use k = 3, as for tool ASSIST.

| cell | semantics (who proposes, who decides) | τ / k | arm (pairing) |
|---|---|---|---|
| tool arguments × ASSIST | The tool choice is the registered C1 cascade (τ̂\* = 1.00 for Flash). When the model's tool is accepted, the model answers each bounded / copy argument's Choice (the question the argument cascade asks), and its top k values become a shortlist. The LLM fills the whole call in ONE forced call: a string argument is restricted by an `enum` in the offered schema, any other type by the instruction. A shortlist containing "other value", or every option, restricts nothing. After a deferred tool choice the LLM writes the call natively, as in C1. | k = 3; tool τ̂\* = 1.00 | `tp_assist` |
| tool arguments × GUARD | The LLM writes the call natively. The model answers the Choice of each bounded / copy argument and vetoes the call when it is confident (≥ τ) of a value other than the LLM's (never on its "other value" answer). The veto names the most confidently disputed argument and suggests no value. The LLM rewrites the call once, and the rewrite is executed unchecked, as in the tool guard. | τ = 0.9 (no registered τ at this point) | `tp_guard` (tool choice off) |
| memory write × CASCADE | The model's store / skip when its confidence ≥ τ, else the LLM gate (`memory_write: llm`, the same prompt). Reads: `always` (the recent 20), as in the Stage-1 write rows `jev_memw` and `mem_llm_gate`. | τ = 0.9 | `mw_cascade` |
| memory write × GUARD | The LLM gate proposes. A proposed write that the model rejects (p_yes < τ) is not stored; a skip stands. Reads: `always`. | τ = 0.5 (the registered write trigger) | `mw_guard` |
| memory read × CASCADE | Every observation is stored (`always`). The model's top-k selection (the read assist, with its "none" option) is injected when its confidence in its top memory ≥ τ; otherwise every recent memory is injected (`always`: the LLM reads them all and decides what matters). | τ = 0.9; k = 3 (the registered read k) | `mr_cascade` |
| memory read × TRIGGER | Every observation is stored. Before each step the model is asked whether the agent needs information stored earlier. On yes the recent memories are injected (as `always`), on no none. The question is asked on the System-1 state, which shows the recent history but not the stored memories. | τ = 0.9 (no registered τ at this point) | `mr_trigger` |
| reflection × CASCADE | The model's reflect / skip when confident, else the agent LLM answers the same reflection question (one yes / no call on the System-1 state). On yes the LLM reflects (the trigger's reflection prompt). | τ = 0.9 | `rf_cascade` |
| stop × CASCADE | The model is asked every step: the guard question at a `finish` proposal, the trigger question otherwise. When confident, its stop / continue is executed: a confident "no" rejects a finish (with the guard's note), and a confident "yes" ends the episode early (the LLM writes the answer). Otherwise the LLM's own decision stands. This is the trigger at τ and the guard at 1 − τ in one arm. | τ = 0.9 (= the registered stop trigger) | `st_cascade` |
| routing × CASCADE | The agent is the cheap LLM (Flash-Lite). The model chooses Flash or Flash-Lite for the step when confident; otherwise Flash-Lite answers the routing question itself. As in Stage 1, the routing row is compared with Flash-Lite alone. | τ = 0.9 | `rt_cascade` (reference `small_only`) |

Every System-1 and System-2 call goes through the episode's Ledger at its decision point. LLM-gate calls are booked as
System-2 at the decision point they decide. JevGym exposes the oracles, so every new record carries a gold label:
argument Choice golds (assist, guard), `mem_need` (memory write), the strict read gold (memory read; the trigger's gold
is whether a stored item holds an input of the next oracle step), `need_fatal` (reflection, routing) and
`finish_ok` / `world_complete` (stop). Gold never reaches a backend: it is computed after the model answered and is
only logged. The request-level canary test covers both new configurations (`tests/test_e6b_modes.py`).

## Undefined and equivalent cells (one line each)

- **tool choice × TRIGGER (undefined).** The next tool is a choice among tools, not a module to invoke or skip.
  Whether to call any tool at all is the stop decision.
- **tool arguments × TRIGGER (undefined).** An argument is a value to choose, not a module.
- **memory write × REPLACE (= TRIGGER).** The model alone deciding store / skip is the write trigger.
- **memory write × ASSIST (undefined).** A yes / no decision has no option list to shortlist; the top 1 of two options
  is the replace.
- **memory read × REPLACE (= ASSIST, k = 1).** The read assist already injects the model's selection without an LLM
  choice. A replace would be the assist with k = 1.
- **memory read × GUARD (undefined).** The LLM never proposes which memories to read, so there is no proposal to veto.
- **reflection × REPLACE (= TRIGGER).** The model alone deciding reflect / skip is the reflection trigger.
- **reflection × ASSIST (undefined).** It is a yes / no decision, with no list to shortlist.
- **reflection × GUARD (undefined).** The LLM never proposes a reflection in the reference loop, so there is nothing to
  veto. A guard would first need an LLM reflection gate whose only purpose is to be vetoed.
- **stop × REPLACE (= CASCADE at τ = 0.5).** The model alone deciding stop / continue at every step is the guard and the
  trigger together at 0.5.
- **stop × ASSIST (undefined).** It is a yes / no decision, with no list to shortlist.
- **routing × REPLACE (= TRIGGER).** The model alone choosing the strong or the cheap LLM is the routing trigger.
- **routing × ASSIST (undefined).** It is a binary choice, with no list to shortlist.
- **routing × GUARD (undefined).** No LLM proposes a model switch in the reference loop, so there is no proposal to
  veto.
- **action × TRIGGER (undefined).** The action is a choice among admissible actions, not a module.
- **action × GUARD (pending).** It is meaningful: the LLM proposes an admissible action and the model vetoes once, as
  the tool guard does. It is not implemented here, because the action row belongs to the action-only environments
  (ALFWorld, row E6a).

## Judgement calls, stated so a reader can disagree

- **Memory read CASCADE.** "Defer to the LLM" here means "show the LLM every recent memory" (the `always` reference),
  because the reference loop has no LLM read-selection call. Adding one would be machinery that exists only for this
  cell.
- **Reflection and routing CASCADE.** These cells need a new LLM yes / no gate. It is the minimal one (the memory
  gate's prompt pattern on the same state and question). It is booked as System-2 at that decision point, so the
  deferral's price is visible.
- **Stop CASCADE.** This cell composes two existing cells (trigger at τ, guard at 1 − τ). It is kept as a cell because
  "the model decides when it is sure, otherwise the LLM's own finish decision stands" is exactly the cascade, and no
  single existing arm runs both sides.
- **Memory read REPLACE.** The task brief expected this cell to be meaningful. It is classed as equivalent, because
  "the model alone picks which memories to inject" is what the read assist already does: no LLM chooses among the
  shortlisted memories. Running it would only vary k.
