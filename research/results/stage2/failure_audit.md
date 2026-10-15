# Stage 2: paired failure-mechanism audit of the Jev profile (descriptive, post hoc)

Label: descriptive, post hoc (review round 12); mechanical rules fixed in code; not a registered analysis. 14 failures of `jevagent_best` among 500 tasks (registered: {'failures': 14, 'failures_fired': 14, 'trigger_tasks': 416}); 14 matched successes (same family and plan shape, nearest seed). Threshold τ̂* (tool choice) = 1.0.

**Reading:** these counts do not identify how the failures arose.

| quantity | profile failures | same seeds, cascade alone | same seeds, LLM-only | matched profile successes | matched seeds, cascade alone | all profile successes |
|---|---|---|---|---|---|---|
| episodes | 14 | 14 | 14 | 14 | 14 | 486 |
| successful | 0 | 14 | 13 | 14 | 14 | 486 |
| left the canonical plan | 14 | 0 | 1 | 0 | 0 | 9 |
| first divergent step (median, 1-based) | 3.0 | – | 4.0 | – | – | 3.0 |
| divergent action = accepted System-1 tool choice | 4 | 0 | 0 | 0 | 0 | 0 |
| divergent action = accepted System-1 argument | 0 | 0 | 0 | 0 | 0 | 0 |
| divergent action = LLM choice | 10 | 0 | 1 | 0 | 0 | 9 |
| accepted System-1 decision against gold at/before divergence | 4 | 0 | 0 | 0 | 0 | 0 |
|   of which a tool choice | 4 | 0 | 0 | 0 | 0 | 0 |
|   of which an argument | 0 | 0 | 0 | 0 | 0 | 0 |
| trigger fired (any step) | 14 | – | – | 12 | – | 402 |
| trigger fired at/before divergence | 14 | – | – | 0 | – | 9 |
| paths split from cascade alone after a reflection | 14 | – | – | 0 | – | 119 |
| paths split before any reflection | 0 | – | – | 0 | – | 1 |
| paths never split (trigger fired) | 0 | – | – | 12 | – | 282 |
| no reflection, paths split | 0 | – | – | 0 | – | 1 |
| at the split, System-1 proposed the same tool in both arms | 14 | – | – | 0 | – | 114 |
| at the split, System-1 accepted in one arm only | 13 | – | – | 0 | – | 110 |
| premature finish | 0 | 0 | 0 | 0 | 0 | 0 |
| ended by strict mode (irreversible call) | 14 | 0 | 1 | 0 | 0 | 0 |
|   irreversible call chosen by System-1 (accepted) | 4 | 0 | 0 | 0 | 0 | 0 |
|   irreversible call chosen by the LLM | 10 | 0 | 1 | 0 | 0 | 0 |
| ended by the step budget | 0 | 0 | 0 | 0 | 0 | 0 |
| unclear / multiple | 0 | 0 | 0 | 0 | 0 | 0 |

Labels: profile failures: accepted System-1 tool choice after a reflection 4, LLM choice after a reflection 10; same seeds, cascade alone: no divergence 14; same seeds, LLM-only: no divergence 13, LLM choice 1; matched profile successes: no divergence 14; matched seeds, cascade alone: no divergence 14; all profile successes: no divergence 477, LLM choice after a reflection 9.
Fatal calls: profile failures: {'refund': 14}; same seeds, cascade alone: none; same seeds, LLM-only: {'refund': 1}; matched profile successes: none; matched seeds, cascade alone: none; all profile successes: none.

Base rates over all 500 tasks (profile vs the cascade alone, same seed): failures {'split_after_reflection': 14}; successes {'no_split': 282, 'no_reflection_no_split': 83, 'split_after_reflection': 119, 'split_before_reflection': 1, 'no_reflection_split': 1}.

## Episodes

| group | task | variant | label | first div. (1-based) | div. tool / canonical | by | S1 errors (tool / arg steps) | fires (1-based) | split (1-based) | vs cascade | at split: profile S1 (conf, accepted) / cascade S1 | end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fail | `jevgym-clarify-300042-d60-near-strict` | method | accepted System-1 tool choice after a reflection | 3 | refund / ask_user | s1_tool | [3] / [] | [2] | 3 | split_after_reflection | refund (1.0, acc) / refund (0.99, def) | strict_irreversible |
| fail | `jevgym-clarify-300080-d60-near-strict` | order | LLM choice after a reflection | 4 | ask_user / cancel_order | llm | [] / [] | [2, 3, 5, 6, 7, 8] | 4 | split_after_reflection | cancel_order (0.66, def) / cancel_order (1.0, acc) | strict_irreversible |
| fail | `jevgym-multi-300083-d60-near-strict` | multi | LLM choice after a reflection | 3 | ask_user / cancel_order | llm | [] / [] | [2, 5, 6, 7, 8] | 2 | split_after_reflection | get_orders (1.0, acc) / get_orders (0.98, def) | strict_irreversible |
| fail | `jevgym-multi-300085-d60-near-strict` | multi | LLM choice after a reflection | 3 | ask_user / cancel_order | llm | [] / [] | [2, 5, 6, 7] | 3 | split_after_reflection | cancel_order (0.99, def) / cancel_order (1.0, acc) | strict_irreversible |
| fail | `jevgym-clarify-300104-d60-near-strict` | order | LLM choice after a reflection | 4 | ask_user / cancel_order | llm | [] / [] | [2, 3, 5] | 4 | split_after_reflection | cancel_order (0.84, def) / cancel_order (1.0, acc) | strict_irreversible |
| fail | `jevgym-multi-300145-d60-near-strict` | multi | LLM choice after a reflection | 3 | ask_user / cancel_order | llm | [] / [] | [2, 5, 6, 7, 8, 9] | 3 | split_after_reflection | cancel_order (0.99, def) / cancel_order (0.99, def) | strict_irreversible |
| fail | `jevgym-clarify-300168-d60-near-strict` | order | LLM choice after a reflection | 4 | ask_user / cancel_order | llm | [] / [] | [2, 3, 5, 6] | 4 | split_after_reflection | cancel_order (0.83, def) / cancel_order (1.0, acc) | strict_irreversible |
| fail | `jevgym-clarify-300202-d60-near-strict` | method | accepted System-1 tool choice after a reflection | 3 | refund / ask_user | s1_tool | [3] / [] | [2] | 3 | split_after_reflection | refund (1.0, acc) / refund (0.99, def) | strict_irreversible |
| fail | `jevgym-clarify-300216-d60-near-strict` | order | LLM choice after a reflection | 4 | ask_user / cancel_order | llm | [] / [] | [2, 3, 5, 7, 8, 9] | 4 | split_after_reflection | cancel_order (0.63, def) / cancel_order (1.0, acc) | strict_irreversible |
| fail | `jevgym-multi-300359-d60-near-strict` | multi | LLM choice after a reflection | 3 | ask_user / cancel_order | llm | [] / [] | [2, 5, 6, 7, 8] | 3 | split_after_reflection | cancel_order (0.89, def) / cancel_order (1.0, acc) | strict_irreversible |
| fail | `jevgym-clarify-300386-d60-near-strict` | method | accepted System-1 tool choice after a reflection | 3 | refund / ask_user | s1_tool | [3] / [] | [2] | 3 | split_after_reflection | refund (1.0, acc) / refund (0.99, def) | strict_irreversible |
| fail | `jevgym-clarify-300432-d60-near-strict` | order | LLM choice after a reflection | 4 | ask_user / cancel_order | llm | [] / [] | [2, 3, 5, 7, 8, 9, 10] | 4 | split_after_reflection | cancel_order (0.85, def) / cancel_order (1.0, acc) | strict_irreversible |
| fail | `jevgym-clarify-300456-d60-near-strict` | order | LLM choice after a reflection | 4 | ask_user / cancel_order | llm | [] / [] | [2, 3, 5] | 4 | split_after_reflection | cancel_order (0.65, def) / cancel_order (1.0, acc) | strict_irreversible |
| fail | `jevgym-clarify-300474-d60-near-strict` | method | accepted System-1 tool choice after a reflection | 3 | refund / ask_user | s1_tool | [3] / [] | [2] | 3 | split_after_reflection | refund (1.0, acc) / refund (0.99, def) | strict_irreversible |
| fail_cascade | `jevgym-clarify-300042-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300080-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-multi-300083-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-multi-300085-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300104-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-multi-300145-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300168-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300202-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300216-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-multi-300359-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300386-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300432-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300456-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_cascade | `jevgym-clarify-300474-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300042-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300080-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-multi-300083-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-multi-300085-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300104-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-multi-300145-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300168-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300202-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300216-d60-near-strict` | order | LLM choice | 4 | ask_user / cancel_order | llm | [] / [] | – |  | – | – / – | strict_irreversible |
| fail_llm | `jevgym-multi-300359-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300386-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300432-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300456-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| fail_llm | `jevgym-clarify-300474-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match | `jevgym-clarify-300034-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-clarify-300072-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-multi-300079-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-multi-300095-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | [] |  | no_reflection_no_split | – / – | finished |
| match | `jevgym-clarify-300096-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-multi-300161-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-clarify-300160-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-clarify-300194-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-clarify-300208-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-multi-300379-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | [] |  | no_reflection_no_split | – / – | finished |
| match | `jevgym-clarify-300378-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-clarify-300424-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-clarify-300448-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match | `jevgym-clarify-300466-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | [2] |  | no_split | – / – | finished |
| match_cascade | `jevgym-clarify-300034-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300072-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-multi-300079-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-multi-300095-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300096-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-multi-300161-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300160-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300194-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300208-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-multi-300379-d60-near-strict` | multi | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300378-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300424-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300448-d60-near-strict` | order | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |
| match_cascade | `jevgym-clarify-300466-d60-near-strict` | method | no divergence |  |  /  |  | [] / [] | – |  | – | – / – | finished |

Matches: `jevgym-clarify-300042-d60-near-strict` -> `jevgym-clarify-300034-d60-near-strict` (family+shape, seed gap -8); `jevgym-clarify-300080-d60-near-strict` -> `jevgym-clarify-300072-d60-near-strict` (family+shape, seed gap -8); `jevgym-multi-300083-d60-near-strict` -> `jevgym-multi-300079-d60-near-strict` (family+shape, seed gap -4); `jevgym-multi-300085-d60-near-strict` -> `jevgym-multi-300095-d60-near-strict` (family+shape, seed gap +10); `jevgym-clarify-300104-d60-near-strict` -> `jevgym-clarify-300096-d60-near-strict` (family+shape, seed gap -8); `jevgym-multi-300145-d60-near-strict` -> `jevgym-multi-300161-d60-near-strict` (family+shape, seed gap +16); `jevgym-clarify-300168-d60-near-strict` -> `jevgym-clarify-300160-d60-near-strict` (family+shape, seed gap -8); `jevgym-clarify-300202-d60-near-strict` -> `jevgym-clarify-300194-d60-near-strict` (family+shape, seed gap -8); `jevgym-clarify-300216-d60-near-strict` -> `jevgym-clarify-300208-d60-near-strict` (family+shape, seed gap -8); `jevgym-multi-300359-d60-near-strict` -> `jevgym-multi-300379-d60-near-strict` (family+shape, seed gap +20); `jevgym-clarify-300386-d60-near-strict` -> `jevgym-clarify-300378-d60-near-strict` (family+shape, seed gap -8); `jevgym-clarify-300432-d60-near-strict` -> `jevgym-clarify-300424-d60-near-strict` (family+shape, seed gap -8); `jevgym-clarify-300456-d60-near-strict` -> `jevgym-clarify-300448-d60-near-strict` (family+shape, seed gap -8); `jevgym-clarify-300474-d60-near-strict` -> `jevgym-clarify-300466-d60-near-strict` (family+shape, seed gap -8).

Generated 2026-10-05T07:13:06+00:00 at git 4e95fbb390ef; inputs: `runs/stage2_jevgym/jevagent_best.jsonl` (defe93054b0b), `runs/stage2_jevgym/best_single.jsonl` (43495c708896), `runs/stage2_jevgym/llm_only.jsonl` (fba5397498f8).
