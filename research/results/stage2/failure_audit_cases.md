# Two worked cases from the failure audit (descriptive; facts from the logs)

Selected mechanically by scripts/stage2_failure_audit.py (rule in `cases`). Steps are 1-based. τ̂* for the tool choice = 1.00: System-1's tool is executed only when its confidence reaches it; otherwise the LLM chooses.

## Case 1: a failure of the Jev profile (`jevgym-clarify-300080-d60-near-strict`, clarify, variant order)

- Mechanical label: *LLM choice after a reflection* (the most frequent label among the failures: *LLM choice after a reflection*).
- Steps of the profile up to the first divergence: 1. `find_user` by the LLM (System-1 proposed `ask_user` at 0.91); 2. `get_orders` by System-1 (accepted) at confidence 1.00; a reflection was written before this action; 3. `ask_user` by System-1 (accepted) at confidence 1.00; a reflection was written before this action; 4. `ask_user` by the LLM (System-1 proposed `cancel_order` at 0.66).
- The oracle's acceptable tools at step 4: cancel_order.
- The trigger fired at step(s) [2, 3, 5, 6, 7, 8].
- The same seed under the cascade alone (no trigger) succeeded; under LLM-only it succeeded.
- Both arms executed the same actions up to step 3; at step 4 System-1 proposed `cancel_order` at confidence 0.66 in the profile (deferred) and `cancel_order` at 1.00 in the cascade alone (accepted). In the profile, 2 reflection(s) had been written by then; they are part of the state System-1 is shown.
- What this does and does not show: the logs fix the order of events above; they do not show what System-1 or the LLM would have done without the reflection text at the same state, so the case does not identify a cause.

## Case 2: a successful deferral in the Jev profile (`jevgym-clarify-300034-d60-near-strict`, clarify, variant method)

- At step 3, System-1 proposed `refund` at confidence 0.99, below τ̂* = 1.00, so the LLM chose; it called `ask_user`, which the oracle accepts there (acceptable: ask_user); `refund` is not.
- The episode succeeded in 5 steps; the trigger fired at step(s) [2].
- Facts only: the threshold sent this decision to the LLM, and the LLM's choice was on the canonical plan.

Overall reading of the audit: these counts do not identify how the failures arose.
