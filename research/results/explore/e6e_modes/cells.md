# E6e: the five REPLACE cells, run as their own arms

Exploratory row E6e (2026-10-06; `docs/PREREG_A7_STAGE2.md`, Deviations entry "Exploratory row E6e"). E6b classified
five REPLACE cells as *equivalent* to an implemented cell (`results/explore/e6b_modes/cells.md`): at a yes / no point,
the decision model deciding alone is the trigger at τ = 0.5; at stop it is the cascade at τ = 0.5; at memory read it
is the assist with k = 1. The figure drew them as "= mode" tiles. At the author's request E6e runs each of them
explicitly, as its own REPLACE arm, so that every cell of the map has its own paired numbers. A REPLACE arm is run even
where its behaviour is identical to a cell already run at the same threshold.

Machine-readable: `harness/config.py` `E6E_CELLS` (the five cells, now in `VALID_MODES`) and `EQUIVALENT_NOTES` (the
cell each one equals; the module map prints it in the tile). `UNDEFINED_CELLS` now holds only the three undefined
TRIGGER cells (tool choice, tool arguments, action).

## The five REPLACE cells

In every REPLACE cell the decision model decides alone with its argmax answer. The LLM is never asked at that point:
no gate call, no selection call, and the threshold τ of the arm's configuration is ignored.

| cell | semantics | same rule as | arm |
|---|---|---|---|
| memory write × REPLACE | Each observation offered for storage is stored iff the model's p_yes ≥ 0.5 on the write question. Reads: `always`. | trigger at τ 0.5 | `mw_replace` |
| memory read × REPLACE | Every observation is stored (`always`). Before each step the model's single top memory on the read question (the tournament of the read assist, k = 1) is injected; "none" injects nothing. | assist with k = 1 | `mr_replace` |
| reflection × REPLACE | The LLM reflects before the step iff the model's p_yes ≥ 0.5 on the reflection question. | trigger at τ 0.5 | `rf_replace` |
| stop × REPLACE | The model decides stop / continue at every step with its argmax: at a finish proposal the guard question (a "no" rejects the finish, with the guard's note), at any other step the trigger question (a "yes" ends the episode early; the LLM writes the answer in one forced finish call). | cascade at τ 0.5 (the guard and the trigger together at 0.5) | `st_replace` |
| routing × REPLACE | The step goes to the strong LLM (Flash) iff the model's p_yes ≥ 0.5 on the routing question, else to the cheap LLM (Flash-Lite, the default agent). | trigger at τ 0.5 | `rt_replace` (reference `small_only`) |

The equivalences are exact in the code: `tests/test_e6e_modes.py` runs each REPLACE arm and the cell it equals on the
same simulated JevGym tasks and finds the same decisions, calls, outcomes and ledger records once the mode label is
set aside. The new arms log `mode: replace`, so the readout can tell them apart. Every System-1 and System-2 call goes
through the episode ledger at its decision point. JevGym's oracles label every decision (`gold`), and no gold reaches a
backend (`Question.meta` is never sent; wire canary in `tests/test_e6e_modes.py`). The modes that existed before E6e
behave and call exactly as before (`tests/e6b_fingerprint.py`: the pre-E6b modes, the E6b / E6c modes and the E6d
modes, each against a fixture generated on the code before the change).

## Readout

The readout is E6b's (`scripts/explore_e6e_modes.py` = `scripts/explore_e6b_modes.py` row `e6e`). Per cell it gives the
paired comparison with the cell's reference in the same run (`llm_only`; routing: `small_only`, with the comparison
to Flash alone beside it): Δ success, saving, wins and losses, cost and wall time. It also gives coverage and activity
(replace: 100 % each, since the model decides every question), the model's accuracy against the gold label, and the
module map's fixed verdict rule. Stop also reports the finishes the model rejected and the early stops. Memory read
also reports the mean number of memories injected and whether the injection held every memory the next step needed.
Everything is descriptive: no test, and outside every Holm table.
