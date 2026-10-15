# E6d: defining the module map's remaining cells

Exploratory row E6d (2026-10-06; `docs/PREREG_A7_STAGE2.md`, Deviations entry "Exploratory row E6d"). E6b classified
all 40 (decision point, mode) cells (`results/explore/e6b_modes/cells.md`). It left 10 undefined, 5 equivalent and
the action guard pending; E6c then implemented the action guard. At the author's request E6d defines and runs every
remaining cell that has a meaningful definition. Machine-readable: `harness/config.py` `E6D_CELLS`;
`UNDEFINED_CELLS` now holds 3 undefined and 5 equivalent cells.

## The seven cells defined by E6d

**ASSIST at the four yes / no points ("advise").** For an option list, ASSIST means that the decision model shortlists
and the LLM chooses within the shortlist. With two options a shortlist cannot narrow anything (its top 1 is the
replace), which is why E6b called these cells undefined. The binary analogue of a shortlist is *advice*: the decision
model answers the yes / no question, and its answer and confidence reach the LLM as a hint. The LLM then decides
itself, as in the ASSIST of the other rows.

| cell | who proposes / who decides | τ | arm |
|---|---|---|---|
| memory write × ASSIST | The model answers the write question. The LLM write gate (`memory_write: llm`, same prompt) gets the model's answer and confidence before its question and decides store / skip. Reads: `always`. | — | `mw_assist` |
| reflection × ASSIST | The model answers the reflection question. The LLM gate of E6b (the same question on the same state) gets the advice and decides whether the LLM reflects. | — | `rf_assist` |
| stop × ASSIST | Before each step the model answers the stop-trigger question ("has the task already been completed?"). Its answer and confidence go into the LLM's turn as a note, and the LLM decides by its own call: `finish` or another tool. The advice never ends or blocks an episode by itself. | — | `st_assist` |
| routing × ASSIST | The model answers the routing question. The cheap LLM (Flash-Lite, the default agent) answers the same question with the advice and decides whether the step goes to Flash. | — | `rt_assist` (reference `small_only`) |

**GUARD where an LLM proposal now exists.** E6b called these undefined because the reference loop has no LLM that
proposes a reflection or a model switch. E6b's cascades added exactly such a proposer, the LLM yes / no gate on the
same question. A guard on it is the memory-write guard's pattern: the LLM proposes; a "yes" (reflect / escalate) is
checked, and the model vetoes it when p_yes < τ. A veto means the proposal is not executed; a "no" invokes nothing and
needs no check.

| cell | semantics | τ | arm |
|---|---|---|---|
| reflection × GUARD | The LLM gate proposes reflecting; the model can veto the proposal (no reflection). | 0.5 (the Stage-1 guards' τ) | `rf_guard` |
| routing × GUARD | The cheap LLM proposes escalating to Flash; the model can veto it (the step stays on Flash-Lite). | 0.5 | `rt_guard` (reference `small_only`) |
| memory read × GUARD | Every observation is stored (`always`). Before each step the LLM selects the stored memories its next step needs: one call over the recent memories (the pool `always` injects), answering ids or "none". This is the read analogue of the LLM write gate. The model is asked the tool-guard-style question about the selection ("are these the memories it needs, with none missing?", on the state plus the selected memories) and vetoes it when p_yes < τ. The LLM then selects once more with a rejection note, and that selection is injected unchecked (guard retries = 1, as every guard). | 0.5 | `mr_guard` |

The memory-read guard has a clean definition, but only because E6d introduces the proposer it guards: an LLM
read-selection call that the reference loop did not have. This is stated so a reader can weigh it. The reflection and
routing guards guard a proposer that E6b introduced for its cascades.

Every System-1 and System-2 call goes through the episode ledger at its decision point (LLM gates and the memory
selection as System-2). Golds come from the JevGym oracle:

- **memory write:** `mem_need`;
- **reflection and routing:** `need_fatal` of the executed step's action (`unobserved` when the gate fired);
- **stop:** `world_complete` at the advised state, the gold of the question the advice answered;
- **memory-read guard:** the selection holds every stored item with an input of the next oracle step.

Gold never reaches a backend (wire canary in `tests/test_e6d_modes.py`). Every mode that existed before E6d makes the
same decisions and calls as before. This is shown by two fingerprint fixtures: the pre-E6b modes, and the E6b / E6c
modes generated on the pre-E6d code.

## What stays outside

| cell | status | reason |
|---|---|---|
| tool choice × TRIGGER | undefined | The next tool is a choice among tools, not a module to invoke or skip. Whether to call any tool at all is the stop decision. |
| tool arguments × TRIGGER | undefined | An argument is a value to choose, not a module. |
| action × TRIGGER | undefined | The action is a choice among admissible actions, not a module. |
| memory write × REPLACE | = TRIGGER | The model alone deciding store / skip is the write trigger. |
| memory read × REPLACE | = ASSIST (k = 1) | The read assist already injects the model's selection without an LLM choice. |
| reflection × REPLACE | = TRIGGER | The model alone deciding reflect / skip is the trigger. |
| stop × REPLACE | = CASCADE (τ = 0.5) | The model alone deciding stop / continue at every step is the guard and the trigger together at 0.5. |
| routing × REPLACE | = TRIGGER | The model alone choosing the strong or the cheap LLM is the routing trigger. |

Counts after E6d: 32 implemented (15 in Stage 1, 9 by E6b, 1 by E6c, 7 by E6d), 3 undefined and 5 equivalent.
