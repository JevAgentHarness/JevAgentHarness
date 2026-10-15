# E6f: the trigger cells at tool choice, tool arguments and action

Exploratory row E6f (2026-10-07; `docs/PREREG_A7_STAGE2.md`, Deviations entry "Exploratory row E6f"). Until E6f these three cells of the module map were undefined: a trigger decides whether an optional module runs, and the tool, its arguments and the action are needed at every step, so there was no optional module to trigger. At the author's request E6f gives each of these points an optional module, an **LLM verification pass** on the LLM's own proposal, and lets the decision model decide when it runs. Machine-readable: `harness/config.py` `E6F_CELLS` (action × trigger, tool_params × trigger, tool_select × trigger); `UNDEFINED_CELLS` is now empty (0 cells), so every one of the 40 cells is defined. This file is written by `scripts/explore_e6f_modes.py cells` from `harness/prompts.py`.

## Definition

1. The LLM proposes the tool call (tool choice, tool arguments) or the admissible action (action) exactly as in the LLM-only harness.
2. The decision model is asked whether the proposal needs a second look (the fixed question below, with the proposal filled in), on the same System-1 state the other cells use.
3. If p_yes ≥ τ, the LLM runs **one** verification call: the same context with a check note that names its proposal. It re-reads the state and its proposal and either confirms it (the same call or action again) or revises it once. At tool choice every tool is offered; at tool arguments the call is forced to the proposed tool, so only the arguments can change; at the action point the admissible list is shown again. A reply without a call (or an empty action reply) leaves the proposal standing.
4. Otherwise the proposal runs unchecked. The final proposal is executed.

**Not a guard.** In a guard the decision model judges the proposal and vetoes it, which forces a regeneration. Here the decision model never judges or vetoes anything. It only decides when the LLM checks itself, and the LLM's verification decides the outcome.

**Reference arm "always verify"** (`always` mode, no decision model): the verification call runs on every proposal. It shows how much of verification's effect the trigger keeps and at what share of its cost. The module-map cell itself compares the trigger with `llm_only`, like every other cell.

## Questions (System-1) and verification notes (System-2)

| point | trigger question | verification note | τ |
|---|---|---|---|
| tool choice | The agent proposes to call `<tool>` with arguments {"<argument>": "<value>"}. Should the agent re-check this choice of tool before the call runs: could another tool be the right next step? | [Check] Before your proposed call `<tool>({"<argument>": "<value>"})` runs, re-read the task and the observations so far and check that this is the right tool to call now. If it is, make the same call again; if not, make the right call instead. | 0.5 |
| tool arguments | The agent proposes to call `<tool>` with arguments {"<argument>": "<value>"}. Should the agent re-check these argument values before the call runs: could a value be wrong or missing? | [Check] Before your proposed call `<tool>({"<argument>": "<value>"})` runs, re-read the task and the observations so far and check every argument value. Call `<tool>` again, with the same arguments if they are right or with corrected arguments if not. (forced to `<tool>`) | 0.9 |
| action | The agent proposes to take the action `<action>`. Should the agent re-check this action before it runs: could another action be the right next step? | Admissible actions: … then: [Check] Before your proposed action `<action>` runs, re-read the task and the observations so far and check that it is the right next action. If it is, repeat it; if not, choose the right action instead. End with a line `Action: <action>` copied verbatim from the list. | 0.5 |

The argument trigger is asked about every call of a known tool that has parameters (the final `finish` call included, whose argument is the answer); the tool-choice trigger about every proposed call.

## Thresholds

Fixed before any episode by E6b's rule (E6b entry (c), continued by E6d and E6e): the registered τ of the same decision point's trigger or guard where one exists, else 0.9. Tool choice: **0.5**, the Stage-1 tool guard on Flash (`jev_ts_guard`). Tool arguments: **0.9**, E6b's argument guard (no Stage-1 τ at that point; E6b's else-clause gives the same 0.9). Action: **0.5**, E6c's action guard. Nothing is fitted or tuned on evaluation data.

## Golds (log only; never on the wire)

- **Tool choice:** the proposed tool is in the information-feasible acceptable set AT(s) of JevGym's oracle (the tool-level gold of the other tool-choice cells). Whether the whole call (tool and arguments) is progress, the tool guard's gold, is logged beside it.
- **Tool arguments:** the proposed call is progress (`oracle_verdict`, the tool guard's gold).
- **Action:** the proposal equals ALFWorld's single gold action, a lower bound of "right" (as E6c).

Each decision logs `gold` = the proposal needed a second look (it was wrong), and in `gold_aux` whether the verification ran (`fired`), whether it changed the proposal, the proposal's and the executed value's correctness, and the outcome: `fixed` (wrong → right), `broke` (right → wrong), `confirmed_right` / `confirmed_wrong` (unchanged), `revised_right` / `revised_wrong` (changed, same correctness). Every System-1 and System-2 call goes through the episode ledger at its decision point.
