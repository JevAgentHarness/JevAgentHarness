# Meta-decision golds, wrong-proposal probes and the free-text parameter policy

**Status.** Draft 2026-10-01. It becomes normative when pre-freeze amendment A4 (`docs/PREREGISTRATION.md`) is
accepted before the 2026-10-03 freeze.

**Scope.** This spec defines:

- the gold labels of the meta decision points: reflection, routing, stop (guard and trigger), memory_write and
  memory_read;
- the proposal gold of the tool guard;
- the counterfactual efficacy labels that the value-of-information analysis needs;
- the wrong-proposal probe protocol for guard FAR;
- the free-text parameter policy.

It also says how `bench/metrics.py`, `bench/threshold_theory.py` and the score card consume these labels: H2
regression inputs, C5 VoI and C6 FAR/FRR.

**Sources.** The spec merges two independent proposals (P1 and P2). §13 records, decision point by decision point,
what each proposed, what was adopted and why.

**Dev evidence.** Every dev number below comes from `runs/dev/` (n = 10 per arm, E2 dev round 2). These are pipeline
checks and must never be reported. They were re-verified by deterministic replay of the logged actions through
`JevGymEnv`.

## 0. Summary

| Decision point | Primary gold (bool unless noted) | Measured where | Model calls for the gold | Consumers |
|---|---|---|---|---|
| reflection | `need_fatal`: the reflection-free action at step t is *failing* (fatal / premature / wrong_answer) | `meta_shadow*` arms (executed action); live trigger arms on non-fired steps | none | H2, C5 secondary, threshold theory (gates) |
| routing | `need_fatal` of the **small** model's action at step t | `meta_shadow_s2lite` (executed action = small model's) | none | H2, C4 secondary |
| stop (guard) | `finish_ok`: finishing now with the proposed answer succeeds | every finish proposal: guard, shadow, probes | none | C6 FAR/FRR, H2 |
| stop (trigger) | `world_complete`: only the answer remains | every non-finish step | none | H2 (expected "no opportunity") |
| tool_select (guard) | `proposal_progress`: the proposed call is in the acceptable-action set | every proposal: guard arms, probes | none | guard FAR/FRR |
| tool_select (Choice) | canonical tool (legacy) **plus** the information-feasible acceptable tool set in `gold_aux` | every tool decision | none | decision tables, C3 recall@k, H1 |
| memory_write | `mem_need`: the offered text contains a value that the remaining oracle plan consumes | every offer | none | H2 (negative-control cells on JevGym) |
| memory_read | set of stored items holding an input of the next oracle step (or "none") | every System-1 recall | none | descriptive only (not in H2) |
| tool_params | unchanged for bounded / copy parameters; **no gold** for free-text parameters (§7) | — | — | H3 composition |

**Efficacy labels** (`e`, `h`) come from deferred one-step counterfactual System-2 calls, netted against a placebo
resample (§4). **Wrong proposals** for guard FAR come from natural proposals, model-generated phantoms and synthetic
edits (§6). All of these are measurement-only and never touch an episode.

## 1. Principles (binding on every implementation)

1. **Gold never reaches a backend.**
   - Labels are computed on a copy of the environment, *after* the System-1 answer of the decision has been
     recorded.
   - They are written only to `DecisionLog.gold` / `gold_aux` / `gold_status` and to `Trajectory.probes`.
   - They never go into `Question.meta`, the rendered System-1 state, LLM messages, memory item text or meta,
     probe or counterfactual prompts, or notes.
   - `Question.meta` keeps exactly what it carries today, so simulated backends and the simulation studies are
     unchanged. That is `oracle_done` for stop, `oracle_accepts` for the tool guard, the canonical values for
     Choice decisions, and nothing for reflection, routing and memory.
   - A simulation study can opt into the new golds with `extra.sim_gold: 1`.
2. **The oracle labels and selects; it never authors.**
   - The oracle may decide *where* a probe or counterfactual is placed, and *which* candidate probes are kept
     (wrongness filter).
   - Every string a backend sees comes from a model (the arm's System-2 or a transplant model), from a fixed
     template, or from a deterministic edit of a model proposal that uses only values visible in the state.
3. **Measurement never changes an episode.**
   - Counterfactual (CF) and probe calls run **after** the episode ends (after `env.evaluate`, before `dump`), on
     snapshots taken at decision time. They therefore cannot alter the trajectory, warm a provider's implicit prompt
     cache for later live steps, or add critical-path latency.
   - All of them are fail-open.
4. **One ledger.**
   - Every measurement call goes through the episode ledger tagged `meta.shadow = true` and
     `meta.measure ∈ {"shadow","cf","probe"}`.
   - The existing `usage_summary` therefore already keeps them out of `usage.total`, `usage.s1` and
     `by_decision_point`.
   - `usage.shadow` gains a `by_measure` breakdown, and `usage.total_with_shadow` stays what the run actually spent.
5. **`steps[].decisions` holds exactly the decisions that the arm's own questions face** (one per decision point per
   opportunity). Every additional measurement record, such as guard probes, the old-wording reflection ablation or
   augmented stop-trigger queries, lives in `Trajectory.probes`. Existing consumers (`decision_records`, `decision_table`,
   `guard_error_rates`, `trigger_rates`, `threshold_theory.decision_pairs`, `training/harvest.py`) never read that
   list.
6. **JevGym golds are replayable.**
   - The environment is deterministic given (seed, action sequence), including the heartbeat-noise RNG.
   - Every gold of an *executed* action can therefore be recomputed offline from a trajectory.
   - Online logging is a convenience. The offline replay is the reference implementation and the audit (§12).

## 2. Shared oracle primitives (JevGym)

### 2.1 P0 oracle fixes (before any label is computed)

- **Argument matching.** `JevGymEnv._done_step` uses the same tolerant matching as `_write_is_gold` (`_args_match`:
  strip, case-insensitive, amount within ±0.01).
  - Today it compares `str(v).lower()` exactly.
  - So a refund whose amount is the string `"458.240"` is "not done", the oracle refunds again, and the state is
    mislabelled as unrecoverable.
- **Information-equivalent reads.** A read step of the plan counts as done once an information-equivalent read
  happened.
  - `get_orders(uid, s)` is satisfied by `get_orders(uid, "all")`.
  - Today the oracle is stuck on `get_orders(pending)` forever after `get_orders(all)`.
- **Audit.** Re-run the oracle replay over every `runs/dev/` trajectory. List every label that changes.
  `SimulatedLLM(error_rate=0)` must still solve 100% of tasks.

These fixes change `oracle_action` / `oracle_done` slightly, and with them the canonical golds and simulated
behaviour. Declare this in A4. The BENCHMARK_SPEC "canonical gold" limitation is narrowed accordingly.

### 2.2 Completion, cost-to-go, recoverability

Let s be an environment state at step t, and let B = `max_steps − t` be the remaining step budget (the finish
counts as a step). `gold_answer(s)` is the plan's finish answer (`"done"` for families without an answer check).

- **`complete(s)`** = `not s.failed and s.evaluate(gold_answer(s)).success`. `evaluate` is read-only, so no copy is
  needed. Only the answer remains.
- **`D(s, B)`** is the cost-to-go. It is computed by an *early-exit* oracle rollout on `deepcopy(s)`:

  ```text
  c = deepcopy(s)
  for k in 0 .. B-1:
      if complete(c):                       return k + 1      # finishing now succeeds
      a = c.oracle_action()
      if a is None or a.tool == "finish":   return inf        # plan exhausted, world not complete
      r = c.step(a)
      if r.done:                            return k + 1 if c.evaluate(None).success else inf
  return inf
  ```

  The early exit matters. Without it, a canonical rollout from a state reached by a valid shortcut still replays
  steps that are no longer needed. For example, in `clarify-8` the agent did `ask → cancel`, the oracle still asked
  for `find_user`, and the shortcut-completed state gets D = 3 instead of 1. The early exit closes that gap.
- **`recoverable(s)`** = `D(s, B) < inf`. A state that is not recoverable is **doomed**.
  - In strict JevGym every irreversible error is terminal, so non-terminal doomed states arise only from the step
    budget.
  - Doomed opportunities get `gold = None`, `gold_status = "doomed"`, are excluded from every rate and are counted.

### 2.3 Verdict of a proposed action

`verdict(s, a)` for a = (tool, args) proposed at state s, with budget B:

| Case | Verdict |
|---|---|
| s is doomed | none (`gold_status = "doomed"`) |
| `finish(x)` | **progress** iff `s.evaluate(x).success`. Otherwise **wrong_answer** if `complete(s)`, else **premature** |
| any other call | s' = `deepcopy(s)` after `step(a)` (for an unknown tool, s' = s with one step of budget consumed); d' = ∞ if the step ended the episode unsuccessfully, else `D(s', B−1)` |
| d' = ∞ | **fatal**: irreversible error, side effect, acting before the required ask, or budget exhaustion |
| d' < D(s, B) | **progress** |
| otherwise | **detour**: read-only distractor, wrong ask slot, repeated call, unknown tool |

Derived labels:

- **failing**: `F = {fatal, premature, wrong_answer}`. Executing the action now makes success impossible.
- **`canonical`**: `oracle_accepts(s, a)`, logged for continuity. Canonical ⊂ progress is a tested invariant.
- **Acceptable-action set**: A(s) = {a : verdict(s, a) = progress}. This resolves P0-2:
  - `find_user` before `ask_user` in `clarify/method` is progress;
  - out-of-order `multi` jobs are progress;
  - `get_orders(all)` is progress after §2.1;
  - `get_order` (singular) is a detour.
- **Information-feasible acceptable tool set** (tool-level metrics):
  - AT(s) = {tool(p) : p is an undone plan step with verdict(s, p) = progress and every input atom of p (§2.4) is
    visible in the instruction or in an observation so far}, plus `finish` if `complete(s)`.
  - The feasibility filter stops a write tool from counting as "acceptable" at step 0 merely because the oracle
    knows its arguments.

**Cost of the oracle part.** No model calls. On the A1 suite (multi, 60 near distractors, measured 2026-10-01) a
deepcopy takes about 1.0 ms and a rollout about 0.3 ms, so one verdict is about 2–4 ms. Under 0.1 s per episode.

### 2.4 Evidence atoms (dataflow map, hand-specified per family and unit-tested)

`need(s)` lists the atoms that the *undone* plan steps still consume. Each atom carries its consumers' distances
j (1 = the next undone step), `d_first = min j` and `d_last = max j`.

| Consumer step | Atoms consumed |
|---|---|
| `get_orders(uid, ·)` | `user_id` = uid (from the `find_user` observation) |
| `cancel_order` / `ship_order` / `refund` | `order_id` (from `get_orders`, or the clarify reply) |
| `refund` | `amount` |
| `ship_order` | the customer's `tier` (from `find_user`), except in `clarify/speed`, where the reply supplies the speed |
| `find_user` | `name` when it is not in the instruction (`clarify/name`: from the reply) |
| `finish` | `multi`: every job order's amount; `total`: every delivered amount; `stock`: the unit count |
| any step needing a clarify slot | the clarify reply string (`env.reply`) for `order` / `method` / `speed` / `name` |

**Matching** runs on the exact text being judged, for example the offered memory text after the harness's
600-character truncation:

- Identifiers (`U…`, `ORD-…`) and tiers use a token-boundary literal match, `(?<![\w.#])v(?![\w.])`. This excludes
  the `#123456` heartbeat noise.
- Amounts and counts match any number token within ±0.005.
- The clarify reply matches by substring.

### 2.5 Environment hooks

`bench/env.py::Env` gets optional oracle hooks. Their defaults return `None`, so suites without an oracle log
`gold_status = "no_oracle"`. The harness calls them via `getattr`, as it calls `oracle_accepts` today. Agents and
backends never call them.

| Hook | Returns |
|---|---|
| `oracle_verdict(tool, args, budget)` | `{verdict, canonical, d_before, d_after}` or None |
| `oracle_finish_ok(answer)` | bool or None |
| `oracle_world_complete()` | bool or None |
| `oracle_need()` | list of atoms `{kind, value, d_first, d_last}` or None |
| `oracle_acceptable_tools()` | list or None |
| `oracle_snapshot()` | an independent copy of the env; the default is `copy.deepcopy(self)`. The copy's RNG is independent, so the real episode's noise lines do not change |

**L3 approximations** belong to the suite owners and are not part of this spec's deliverables. Until a suite
implements them, its meta golds are `None`.

| Suite | Approximation |
|---|---|
| τ² | verdict by DB copy: fatal = a write outside the gold action set, or the gold DB is unreachable; `finish_ok` = the reward function on the copied DB |
| ALFWorld | progress iff the re-planned `expert_plan` gets shorter |
| LoCoMo / LongMemEval | `mem_need` = the offered turn is in some later question's evidence set |
| SWE-bench / Terminal-Bench | episode level only; a sampled submit-time hidden-test check for the stop guard |

## 3. Golds per decision point

Every meta `DecisionLog` must carry `state`. Today the trigger-mode reflection and routing logs omit it, and the
replay scoring of other backends (§5) needs it.

### 3.1 Reflection (trigger, shadow, always / off controls; C5, H2)

**Primary.** At each opportunity t ≥ 1, let a0_t be the action that the arm's System-2 takes at step t *without
reflecting at t*.

- `gold = need_fatal = verdict(s_t, a0_t) ∈ F`.
- `gold = None` with `gold_status = "doomed"` at a doomed state.

**Where a0_t comes from:**

- the executed action, whenever the gate did not fire: shadow arms, non-fired steps of trigger arms, and the
  `off` / `always` controls at their non-reflection steps;
- at fired steps of live trigger arms, `gold = None` and `gold_status = "unobserved"`. With `measure.cf.fired_steps`
  enabled, a deferred CF call supplies it instead (`gold_aux.verdict_src = "cf"`).

**Secondary labels in `gold_aux`:**

- `need_any` = verdict ≠ progress. This is the question-literal reading of "wrong action".
- `retro` = the previous executed action was not progress, or its observation starts with an error / not-found /
  empty marker. This is the retrospective reading of the old wording.
- `verdict` and `d_before`.
- `benefit`, `harm`, `verdict_alt` and `verdict_placebo` where CF branches ran (§4).

**Why `need_fatal` and not `need_any`.**

- A reflection can change success only by preventing a failing action. A detour costs one step and nothing else in
  strict JevGym.
- The VoI identity (§9.1) needs the success-relevant event.
- On the dev data the two coincide except for one detour (Flash-Lite: 4 failing and 1 detour in 66 opportunities;
  Flash: 0 of 65), so the choice matters mainly on L3 suites.

**Question wording, changed by A4.**

The current `Q_REFLECTION` is purely retrospective. The dev positives all sit at the *pre-finish* state of `multi`,
where the action about to happen is a wrong total. The stand-in gave p_yes = 0.01 at all 3 positives of
`jev_reflect_trigger_s2lite`, and fired 5 times, all at step 1 on progress states.

New wording (generic; `controls.infer_dp` still maps it to `reflection` through "stuck", verified):

> Should the agent pause and re-check before its next action: is it stuck or off track, or likely to make a wrong
> action or give a wrong final answer next?

The old wording is asked additionally in the measurement arms as a probe-list record (`probe_source = "wording_v1"`).
That gives a question-alignment ablation. The heuristic rules are unchanged, so `controls.VERSION` stays. Cache keys
include the question text, so the new wording never replays old answers.

**When logged.** `_reflect` creates the `DecisionLog` at step start, as today. The gold is back-patched after the
step's action is chosen and classified, and before `env.step`. CF fields are filled after the episode.

### 3.2 Routing (trigger, shadow; C4, H2)

**Primary.** `gold = need_fatal(aS_t)`: the action that the **small** model takes at s_t is failing. These are the
states where sending the step to the strong model can pay.

**Source.** `meta_shadow_s2lite` (§5). Its acting policy is the small model (Flash-Lite), and routing shadow's default
resolver is the arm's own LLM. So aS_t is the executed action at every step, with no extra call. That is also the
state distribution live routers visit: in dev they routed about 7% of steps to the large model.

On the Flash measurement arm, routing is **not** shadowed. Flash-policy states are not the router's distribution.

**Live trigger arms.**

- Small-routed steps: the executed verdict.
- Large-routed steps: `unobserved`, or a CF small-model call with `measure.cf.fired_steps`.
- These labels serve only as a consistency check (selective labels).

**Secondary labels in `gold_aux`:**

- `need_any`;
- `verdict_large` from a deferred CF call to the large model (same messages and tools);
- `benefit = need_fatal(aS) ∧ verdict(aL) = progress`;
- `harm = verdict(aS) = progress ∧ verdict(aL) ∈ F`.

On Flash-Lite, routing need equals reflection need: one "error imminent" event with two remedies. H2 compares their
resolution on identical labels.

`Q_ROUTING` is already forward-looking and is unchanged.

### 3.3 Stop, guard (C6, H2)

**Primary.** For a finish proposal `finish(x)` at s_t: `gold = finish_ok = s_t.evaluate(x).success`, read-only.

This is exactly "stopping now with this answer succeeds". It is path-independent, which fixes the `clarify-8`
mislabel, and it checks effects, side effects, the clarify ask and the answer.

**`gold_aux`:**

- `kind` ∈ {ok, premature (not `complete`), wrong_answer (`complete` but the answer fails)};
- `d_before`;
- `canonical_done`: the legacy `oracle_done`;
- `world_complete`.

Doomed states are excluded.

**Required harness changes (A4):**

1. `_stop_check` receives the proposed arguments.
2. `Q_STOP_GUARD` shows the answer, JSON-quoted and clipped by `prompts.clip_answer`: whole up to 800 characters,
   else its first 200 and last 600 characters. Verified: `STOP_GUARD_RE` and `infer_dp` still match.
   (Amended 2026-10-01: the first version kept only the first 300 characters. In the first live Jev pilot that cut
   off the total that `multi` answers state in their last sentence, and the guard rejected 84 of 90 correct
   finishes; with the full answer Jev's P(yes) rose from 0.24-0.43 to 0.51-0.92 on all four failing tasks.
   See `results/pilot_jev/JEV_PILOT_NOTES.md`.)

   > The agent wants to stop and submit the final answer «x». Is the task actually complete, and is this answer
   > correct?

3. The rejection note no longer says "continue working". After a wrong-answer rejection that note invites an extra
   write, which is fatal in strict mode. New note:

   > [Verifier] The task may not be complete yet, or the final answer may be wrong. Re-check the requests and
   > recompute the answer before calling finish.

The heuristic stop-guard rule ignores the answer. That is documented, not tuned.

**Why.** In dev, all 4 Flash-Lite failures in `llm_only_s2lite` were wrong totals at states where `oracle_done` was
already true. Neither System-2 model made a premature finish (0 of 20 in the stop-guard arms). The tool-guard
stand-in accepted 3 of the 4 wrong finish proposals it saw (`jev_ts_guard_s2lite`). Under the old gold, natural FAR
was undefined.

**When logged.** In `_stop_check` (guard and shadow), after the System-1 answer.

### 3.4 Stop, trigger (and shadow at non-finish steps)

**Primary.** `gold = world_complete = complete(s_t)`: only the answer remains. This replaces plan-order `oracle_done`,
which is wrong after shortcut paths.

**`gold_aux`:** `canonical_done`; `cut_losses = not recoverable(s_t)`. The latter is identically 0 in strict mode and
kept for L3.

**Augmented query.** JevGym agents never overshoot (dev: 0 of 65 positive opportunities on both models). The
measurement arms therefore also ask `Q_STOP_TRIGGER` at every natural finish state, as a probe-list record
(`probe_source = "augmented_stop_trigger"`). This yields positives for a descriptive AUROC. Augmented records never
enter `trigger_rates`, threshold theory or H2. The H2 stop-trigger cell is pre-declared "no opportunity" on JevGym.

### 3.5 tool_select: guard proposals and Choice decisions

**Guard.** `gold = proposal_progress = verdict(s, proposal) = progress`, for every proposal including retries.

- `gold_aux`: `verdict` (the stratum), `canonical` (the legacy `oracle_accepts`), `d_before`, `d_after`.
- FAR/FRR are reported against progress (question-aligned headline), with the **FAR on fatal proposals** (the
  safety-relevant catch rate) next to it, and canonical as a secondary.
- Dev replay of `jev_ts_guard` (Flash): 75 progress, 9 detours (mostly `search_users`) and 0 fatal executed actions.
  The old "12 wrong proposals" were mostly canonical-order artefacts.

**Choice (replace / cascade / assist / shadow / off).**

- `gold` keeps the canonical tool, for continuity.
- `gold_aux.acceptable_tools` = AT(s) (§2.3); `gold_aux.acceptable_tools_unfiltered` is also logged.
- Tool-level accuracy in `decision_table`, C3 recall@k (§9.5) and H1's s1_ok / s2_ok use **AT(s) as primary**,
  canonical as secondary.
- `tool_params` golds stay canonical-only: arguments are gold only when the executed tool is the oracle's next
  tool. That is a known limitation.

### 3.6 memory_write (trigger, always, llm, and a new **shadow** mode; H2)

**Primary.**

- `gold = mem_need`: the offered text (`tool(args) -> obs` exactly as the gate sees it, after truncation) contains
  an atom of `need(s_{t+1})`.
- The label is evaluated on the env after the step, so it means "still needed later". That matches the question
  "Will this information likely be needed later…".
- It is independent of the store and of the arm.

**`gold_aux`:**

- `atoms`: the matched atoms with `d_first` and `d_last`;
- `far = any matched atom has d_last ≥ 3`, i.e. it must survive at least two intervening oracle steps. This is the
  discriminative variant, about 15–20% base rate in dev, against about 70% for `mem_need`;
- `new`: a matched atom is in neither the instruction nor the current store. This is store-dependent, so it is
  never used as a primary label.

**Logging.** Memory gates log one `DecisionLog` per offer in every `memory_write` mode, with dp `memory_write`,
`s1_dist`, `state` and the gold. They log none today.

**New mode.** `memory_write: shadow` is added to `VALID_MODES`. It asks the Noul, logs it and stores nothing, so a
measurement arm stays identical to LLM-only.

**When logged.** At `memory.offer`, after `env.step`. The harness passes the gold to `offer()`, which writes it only
to the `DecisionLog`, never to `MemoryItem.meta`.

**Validity caveat.** On JevGym, System-2 keeps every observation (2,000 characters), so the success-side value of a
memory write is about zero, and memory cells are **negative controls** for H2 (predicted ΔU ≈ −cost). This holds
unless a pre-registered context-pressure variant lands before the freeze; that is not planned here.

### 3.7 memory_read (assist; descriptive)

At a System-1 recall over stored items I:

- **gold** is the sorted list of labels of G = {m ∈ I : m.text matches an atom consumed by the next undone step},
  or `["(none of these memories is relevant)"]` when G is empty;
- `gold_aux.lenient` is the same with any remaining step;
- `gold_aux.rounds` holds the per-round tournament survivors.

**Metrics:** hit@k, MRR and none-accuracy (`memory_read_hits`).

Recalls with |pool| ≤ k make no System-1 call and log nothing. With about 4 stored items per episode on JevGym the
task is near trivial, so it is reported descriptively and kept out of H2. Exact golds come from the L3 memory suites.

## 4. Counterfactual efficacy labels (deferred)

**Why.** The VoI of a gate depends on what the intervention does when it fires:

- e: the gain when firing at a needed state;
- h: the loss when firing at a non-needed state.

Gemini at temperature 0 is not deterministic. Two identical LLM-only arms differed on 1 of 10 tasks. So "the
intervention helps" must be separated from "a second sample helps".

**Snapshot.** At each sampled state the harness keeps:

- `env.oracle_snapshot()`;
- a shallow copy of the message list. This is safe: `append_note` replaces `messages[-1]`, never mutates it. The copy
  keeps thought-signature extras;
- the S1 state string and the LLM binding.

**Branches,** run after the episode with the same tools and temperature, each classified by `verdict` on the
snapshot:

| Branch | Call(s) | Where |
|---|---|---|
| `verdict_alt` (reflection) | the reflection call (same prompt), `append_note`, then the step's tool decision | `meta_shadow_s2lite` |
| `verdict_large` (routing) | the large model's tool decision on the same turn messages | `meta_shadow_s2lite` (large = Flash) |
| `verdict_reject` (stop guard) | rejection note appended, then the next tool decision | every natural finish in `meta_shadow_s2lite` |
| `verdict_placebo` | the default tool decision again, without intervention | at every state that gets any branch |
| `verdict` of a0 at fired steps | default tool decision on the pre-reflection snapshot | live trigger arms, only with `cf.fired_steps` (recommended for the C5 pair) |

**Sampling.**

- All positive states (`need_fatal`, or a wrong finish for `verdict_reject`) get the branches.
- So does a seeded 20% of negative states, chosen by a hash of (seed, task_id, step, purpose). Estimates are weighted
  by 1/0.2.
- One placebo per state serves every branch at that state.

**Estimands** per (decision point, System-2 family):

- e = P(v_alt = progress | v0 ∈ F) − P(v_placebo = progress | v0 ∈ F)
- h = P(v_alt ∈ F | v0 = progress) − P(v_placebo ∈ F | v0 = progress)
- placebo disagreement δ = P(v_placebo ≠ v0): the label-noise floor.

**Validity.**

- The one-step CF ignores effects after the next action, because the reflection note persists.
- On JevGym nearly every positive is the pre-finish state, where the next action ends the episode, so the one-step
  label equals the episode outcome. The share of non-final positives is reported.

**Failures.** A failed CF call sets `gold_aux.cf_status = "cf_error"` (counted in `traj.meta.cf_errors`) and never
drops a task. CF coverage per arm is reported. Arms whose tool decision is not `off` get `cf_status = "cf_unsupported"`.

**Cross-model smoke test.** A Flash call on a Flash-Lite history with its thought signatures, and the reverse, is
what live routing already does. Smoke-test 5 tasks before the main measurement runs.

## 5. Measurement arms and configuration

### 5.1 The two measurement arms

| Arm | System-2 | n | Shadowed decision points | CF branches | Probes |
|---|---|---|---|---|---|
| `meta_shadow` | Flash | 200 | reflection (new and v1 wording), stop (+ augmented trigger), memory_write | none (no positives expected; labels still logged) | stop: natural, forced_premature, transplant (Flash-Lite T=0), T5, T6. tool guard: natural, transplant, T1–T4 |
| `meta_shadow_s2lite` | Flash-Lite | 500 | reflection (new and v1), routing, stop (+ augmented), memory_write | reflection, large (Flash), stop rejection, placebo | as above; transplant = Flash-Lite at T=1 |

**What they are.**

- Both behave exactly like LLM-only: the existing "shadow reproduces LLM-only" check is extended to them.
- They run on the main task list (seeds 0…N−1), so they pair with every live arm.
- The score card already recognises them as shadow-only (`_shadow_only`), which excludes them from Holm, iso-quality,
  Pareto, hypervolume and frontier cost-of-pass. They are never a reference.

**Batching rule.** Shadow questions are **not** batched across decision points for generative backends. Each
question is its own call, so the logged answer equals what a live single-question arm would receive. Backends whose
answers are independent per question by design (Jev, logprob deciders) may batch them.

### 5.2 Proposed config syntax

The new `HarnessConfig.measure` block (default: empty, which means golds only, no CF and no probes):

```yaml
measure:
  seed: 0
  cf: {reflection: true, large_llm: flash, stop_reject: true, placebo: true,
       sample_negatives: 0.2, fired_steps: false}
  probes:
    stop: [forced_premature, transplant, T5, T6]
    tool_select: [natural, transplant, T1, T2, T3, T4]
    rho_edit: 0.5            # per step: one T1–T4 tool probe
    rho_t5: 0.25             # per non-final step: one T5 probe
    rho_transplant: 0.25     # non-finish steps that get a transplant proposal (every finish state does)
    transplant_llm: lite     # Flash arms: Flash-Lite T=0; Flash-Lite arms: Flash-Lite T=1
    tau_ref: 0.5             # τ of the live guard arm whose FAR/FRR is reported
reflection: {mode: shadow, extra: {wordings: [v2, v1]}}
stop: {mode: shadow, extra: {augment_trigger: true}}
```

### 5.3 Other arms required by this spec (applied by the configs maintainer)

`baselines_jevgym.yaml` / `jevgym_meta_control.yaml`:

- **C5 on Flash-Lite (A1).**
  - `jev_reflect_trigger_s2lite` and `ratem_reflect_trigger_s2lite`, n = 500. The rate comes from `trigger_rates` on
    the Jev arm, with a realised-rate check.
  - `rate10/30/50_reflect_trigger_s2lite`, `heuristic_reflect_trigger_s2lite` and `s2self_reflect_trigger_s2lite`
    (s2self bound to Flash-Lite), n = 200.
  - While the stand-in is the System-1, `s2self_*_s2lite` is the same model as `jev_*_s2lite`. Run it only once Jev is
    reachable, otherwise mark it as a duplicate (as the dev round dropped the `lite_*` twins).
- **Stop guard on Flash-Lite.** `ratem_stop_guard_s2lite` (acceptance-rate matched) and
  `heuristic_stop_guard_s2lite`, n = 200, next to the existing `jev_stop_guard_s2lite`.
- **Routing.** `ratem_route`, matched to the Jev router's realised large-model fraction.
- **Memory-write H2 pair.**
  - `jev_memw`, `ratem_memw` and `heuristic_memw`, all with `memory_read: always`, so the read mode is held fixed.
  - The existing `jev_mem` changes the write *and* the read mode, so it stays a configuration row and is not an H2
    cell.
- **C3.** Keep `bm25task_ts_assist_k3`. Add `{jev,bm25,tfidf}_ts_assist_k5`, n = 200.
- **Optional H3 ablation.** `jev_ts_params_replace_answercopy` with `tool_params.extra.answer_as_copy: true`, n = 200.

## 6. Wrong-proposal probe protocol (guard FAR)

### 6.1 What a probe is

A probe is a **phantom proposal**. The guard scores it, but it is never executed.

- It is scored post-episode with exactly the guard question of the live arm (same template, same backend binding,
  same S1 state string of that step), as its own System-1 call.
- It is **never batched** with any other question, so a generative backend cannot condition a live or shadow answer
  on a probe.
- It is stored in `Trajectory.probes` as a `DecisionLog` with:
  - `mode = "guard_probe"`, dp ∈ {stop, tool_select};
  - `s1_value = p_yes ≥ tau_ref`, `s1_dist`;
  - `gold` = `finish_ok` / `proposal_progress`;
  - `gold_aux = {probe_source, operator, verdict, kind, d_before, generator_model}`.
- Ledger tags: `meta.shadow = true`, `meta.measure = "probe"`.

### 6.2 Sources

All sources are produced at the measurement arm's own states.

| Source | Content | Kept | Role |
|---|---|---|---|
| **natural** | the executed proposal itself. Stop: the shadow guard record in `steps[]`. Tool guard: a probe record at every step | always | FRR (primary), natural FAR |
| **forced_premature** (stop) | once per episode, at a state with `complete = False`, t ≥ 1. With probability ½ the last non-complete state before the natural finish (D = 2, the hardest case), otherwise seeded-uniform. A deferred call to the arm's own System-2 with the stop trigger's own text ("The task appears complete. Call finish with the final answer.") and `tool_choice = finish`. This is exactly the proposal a mis-fired stop trigger would produce | always (premature by construction, verified) | FAR_model |
| **transplant** (stop, tool guard) | a deferred call, on the same messages and tools, to a weaker generator (Flash arms: Flash-Lite T=0; Flash-Lite arms: Flash-Lite T=1). Made at every natural finish state and at a seeded ρ = 0.25 of other steps. A finish output becomes a stop probe; any other call becomes a tool-guard probe | **whatever the label**: wrong ones feed FAR_model, right ones feed the style check | FAR_model, style check |
| **edits T1–T6** | deterministic edits of the arm's own proposal using only state-visible values: T1 swap the tool for an offered near-duplicate that shadows it, mapping arguments by name; T2 swap an ID for another same-typed span from `extract_candidates`; T3 flip an enum argument; T4 at a clarify state before the ask, propose the state-changing core tool implied by the instruction's verb with an ID from the state; T5 premature `finish` (the sum of visible amounts when the instruction asks for a number, else "done"), ρ = 0.25 per non-final step; T6 at a finish whose answer has a number, drop or add one visible summand, or apply a ±1.00 / ±0.01 / transposed-digit slip | only if the oracle labels them wrong; up to 5 resamples, then a "no valid probe" count | robustness (per operator); **the only source of fatal tool-guard probes** |

**No oracle-authored right probes.** A canonical call written by the oracle would put gold content in front of a
backend. Model-generated right proposals (transplants) are allowed: their content is model output.

**Other backends** (heuristic, rate, lexical with fallback, s2self, Jev once reachable) are scored by **replaying**
the logged (state, question) pairs of `steps[]` and `probes`:

```text
python -m jevharness.bench.meta_gold replay RUN --arm meta_shadow_s2lite --s1 '<spec>'
  → RUN/replay/<arm>__<backend>.jsonl
```

This is free for the free backends and gives paired, state-identical comparisons (McNemar on probes).

### 6.3 Estimands (pre-registered)

Wilson 95% CIs are task-clustered by bootstrap. Doomed states are excluded.

**Stop guard (C6).**

| Estimand | Definition |
|---|---|
| **FAR_model** (C6 readout) | P(accept at τ_ref \| wrong) over natural ∪ forced_premature ∪ transplant-wrong, per System-2 model. Must rest on **≥ 50 wrong proposals per model** (Flash: about 200 forced plus the transplant wrongs at n = 200). Stratified by kind (premature by D ∈ {2, 3–4, ≥5}; wrong_answer) and by source |
| **FAR_nat** | Natural wrong finishes only. Flash-Lite: the live `jev_stop_guard_s2lite` plus `meta_shadow_s2lite`, ≥ 50 expected at n ≥ 150 |
| **FRR** | P(reject \| right) over natural right finishes: the live C6 arm's own decisions plus the measurement arm |
| **FAR_edit** | Per operator (T5, T6): robustness only, never pooled into FAR_model |
| **Representativeness** | On Flash-Lite, a TOST equivalence test of FAR_nat − FAR_model with margin ±0.10. If it fails, report both and flag it |
| **Style check** | \|FRR(transplant-right) − FRR(natural right)\| ≤ 0.05. If it fails, the guard is detecting "probe-ness", and FAR_model is flagged |
| **Reweighted FAR** | FAR_model reweighted to the natural kind mix observed on Flash-Lite (sensitivity analysis) |
| **Curves** | FAR(τ), FRR(τ) and the AUROC of p_yes for `finish_ok` |

**Tool guard.**

- FAR and FRR against progress (headline).
- **FAR_fatal**: per operator T1–T4 plus transplant-fatal, with ≥ 50 fatal probes per System-2 model.
- FAR by verdict stratum.
- Canonical-gold FAR/FRR as a secondary.

**Rescue and efficacy.** Phantoms are never executed, so rescue comes only from natural wrongs:

- `rescue_rate` in the live Flash-Lite guard arms (existing metric, now on the new gold);
- the CF `verdict_reject` branch (e, h net of placebo, §4).

### 6.4 Exclusion from the pre-registered primary comparisons

Probes live only in measurement arms. They run after the episode, sit outside `steps[]` and are ledger-tagged shadow.
They therefore never enter:

- success, cost, latency or LLM-call counts;
- the delegation ratio;
- the C1–C8 tests, Holm, iso-quality, Pareto, hypervolume or frontier CoP;
- the matched-rate protocol (`trigger_rates` reads `steps[]` only);
- threshold-theory inputs;
- training harvest.

C6's non-inferiority test is the live `jev_stop_guard` against `llm_only` on Flash, unchanged.

### 6.5 Rejected alternatives (pre-declared)

- **A live fault-injection arm** that replaces a natural finish with an oracle-verified wrong answer and passes it
  through the guard for real. It is not run.
  - The injected answer is not System-2's belief, and the rejected call never enters the history. After a rejection
    the model simply re-submits its own (right) answer, so the measured "rescue" is inflated by construction.
  - With a no-guard twin it measures only the catch rate, which the probes already give.
- **A weaker System-2 as the policy instead of probes.**
  - It would change C6's reference (A1 makes C6 a Flash claim), the state distribution and the base rate.
  - Its natural errors are almost all one type (`multi` totals).
  - It is used instead as the transplant generator and as the FAR_nat external-validity check.
- **Running probes inside the C6 arm.** Deferral would keep that arm intact, but measurement-arm placement removes the
  need for an A/A check and any doubt about touching a primary arm.

## 7. Free-text parameter policy

### 7.1 Classification

Every tool parameter gets a frozen `ParamInfo.kind` ∈ {bounded, copy, free}, from the JSON schema only (never from
gold or outcomes), by the first rule that applies:

1. A schema annotation `x-jev-fill: bounded | copy | free`, set by a suite or adapter, decides.
2. **bounded**: enum, boolean, or an integer range of at most 20 values. Unchanged.
3. **copy**: a number or integer; or a string with `format` / `pattern`; or a string whose name or description
   marks an identifier or a quantity taken from context: `^(id|.*_id|.*_ids|email|phone|name|.*_name|path|url|uri|date|time|sku|code|.*_code|amount|units|count|quantity)$`
   on the name, or `\b(id|identifier|e-?mail|phone|name|path|url|date)\b` in the description.
4. **free**: every other string. Unrecognised strings default to free, the conservative choice.

**Coverage valve (frozen before the freeze).** For each copy parameter, coverage is the fraction of dev-seed states
(5000–5999, oracle agent `SimulatedLLM(error_rate=0)`, $0) where the raw oracle value is among
`extract_candidates(state, task_first=25)`. A copy parameter with coverage below 0.8 is reclassified free. The
table is reported in the appendix for every suite.

`ParamInfo.is_copyable()` keeps its current meaning, because `bench/suites/tau2.py` (another owner) uses it with its
own free-text rule. The agent harness switches to `kind`, and tau2 may adopt it.

### 7.2 JevGym classification (frozen; regression-tested)

| Kind | Parameters |
|---|---|
| free | `finish.answer`, `ask_user.question`, `send_message.text`, `send_email.subject/body`, `notify_warehouse.text`, `search_users.prefix`, `ship_order_international.country` |
| copy | every `name`, `user_id`, `order_id`, `email`, `phone`, `date`, `damage_report_id`, `amount`, `units` |
| bounded | `status`, `reason`, `method`, `item`, `speed`, `missing`, `warehouse`, `year`, `accept` |

`finish.answer` is free in every family, including `stock`, where the answer happens to be copyable. The policy
therefore never depends on, or reveals, the family.

### 7.3 Harness behaviour

- System-1 fills only bounded and copy parameters, in replace, cascade and shadow. OTHER remains the escape hatch for
  copy parameters.
- If the selected tool has **any** free parameter, required or optional, System-2 fills the free ones in the existing
  forced call (`tool_choice = tool`). The System-1 values are passed as fixed, and System-1 wins on conflict.
  - That call counts as an LLM call, so the step is not LLM-free.
  - Optional free text is never dropped. JevGym ignores `ask_user.question`, but τ²'s user simulator does not, and
    dropping it would exploit a simulator artefact.
- **Logging.**
  - One `DecisionLog(dp = tool_params, mode = "generate", s1_value = None, deferred = True)` per free slot, with
    `gold_aux.param_kind = "free"`. Every `tool_params` record carries `param_kind`.
  - The delegation ratio counts generate records as LLM-made: `"generate"` is added to its mode list.
  - Free slots have no decision-level gold. `finish.answer` is judged through `finish_ok` (§3.3).
- **Regression test.** In replace mode System-1 can never set `finish.answer`. The dev failure, System-1 copying the
  heartbeat number `899659` as the final answer, becomes impossible.

### 7.4 H3 consequence

- bounded → "bounded-recoverable object-level";
- copy → "copy-dependent arguments";
- free → generation, never delegated.

Report the delegation ratio and the harmful-delegation rate per kind, plus the bounded / copy / free composition of
the arguments in every suite.

**Rationale.** A Choice whose gold lies outside its option set has VoI ≤ 0. At best System-1 answers OTHER and pays κ
for nothing. At worst it copies noise, which is what happened in dev.

## 8. Schema

| Object | Field | Type and meaning |
|---|---|---|
| `DecisionLog` | `gold` | primary label: bool (meta decision points, guards); list of labels (memory_read); canonical value (Choice) |
| | `gold_status` | `"ok"`, `"doomed"`, `"unobserved"`, `"no_oracle"`, `"cf_error"`; None for legacy logs |
| | `gold_aux` | dict, always including `v` (this spec's version, `1`); per-decision-point keys in §3, CF keys in §4, `param_kind` for tool_params |
| | `state` | now required on every meta decision point |
| `Trajectory` | `probes` | list of `DecisionLog` (guard probes, `wording_v1`, `augmented_stop_trigger`) |
| | `meta.gold` | `{v, n_labelled, n_doomed, n_unobserved, n_cf_error, n_no_valid_probe}` |
| | `meta.cf_errors`, `meta.probe_errors` | error strings (fail-open) |
| `CallRecord.meta` | `shadow`, `measure`, `cf_of` / `probe_of` | measurement tagging (§1.4) |
| `HarnessConfig` | `measure` | §5.2. `VALID_MODES[MEMORY_WRITE]` gains `SHADOW` |

## 9. Consumers

### 9.1 Proposition (gate VoI), stated in paper §3

**Setup.**

- A gate faces n opportunities per episode, with binary need gold y, P(y = 1) = π.
- It fires iff p ≥ τ, with TPR(τ), FPR(τ) and J(τ) = TPR − FPR.
- Firing at y = 1 raises the success probability by e. Firing at y = 0 lowers it by h. Each firing costs c_f, and
  System-1 costs κ per opportunity.
- Effects add across opportunities.
- A random gate fires independently at the matched rate r = π·TPR + (1 − π)·FPR, at zero System-1 cost.

**Result.**

  E[U_gate − U_rand] = n·[ W·(e + h)·π(1 − π)·J(τ) − κ ]

**Proof.** The per-opportunity difference is W[πe(TPR − r) − (1 − π)h(FPR − r)] − κ. Substitute TPR − r = (1 − π)J
and FPR − r = −πJ. The firing costs cancel at the matched rate. ∎

**Corollaries.**

1. Accuracy, π·TPR + (1 − π)(1 − FPR), enters only through J. An always-"no" gate has accuracy 1 − π and VoI −κn.
2. For a concave ROC curve, max_τ J ≤ 2·AUROC − 1, with equality for a two-valued score. The bimodal stand-in
   therefore has AUROC ≈ (1 + J)/2.
3. If the random arm realises rate r + Δr, the difference changes by −nΔr·[W(πe − (1 − π)h) − c_f]. This is the
   realised-rate correction.

**Guards.** Fire = reject and y = proposal wrong, so J = 1 − FAR − FRR. Then e = P(success | wrong rejected) − 0 in
strict mode, h = the success loss from rejecting a right proposal, and c_f = one retry step.

### 9.2 H2 inputs

**A cell** is (gate ∈ {reflection, routing, stop guard, stop trigger, memory_write}) × System-1 backend × System-2
family.

- Rate backends are the matched controls, not cells.
- Gates that never fire, such as the heuristic reflection on JevGym, stay as cells (J = 0).
- The cell list is fixed by the config before the runs, and its size is reported. With Gemini only, expect about
  10–20 estimable cells, not 40.

**Estimability.** A cell enters the regression only if:

- the measurement arm of its family has ≥ 30 positive and ≥ 30 negative primary golds; and
- the live arm and its matched-rate arm exist.

Otherwise it is listed as **"no opportunity"** with predicted gross ΔU ≈ 0. Expected on JevGym:

- every Flash reflection and stop-guard cell;
- every stop-trigger cell;
- routing only on the Flash-Lite small policy.

**Regressors.** On the measurement-arm states of the family (AUROC is never computed on live trigger arms, whose
labels are missing exactly where the gate fired):

- π, AUROC, J(τ_op) and accuracy(τ_op) of p_yes against the primary gold;
- the arm's own backend from its log, other backends from the replay (§6.2).

**Dependent variable.** ΔU(live arm − matched-rate arm), paired by task, bootstrapped over tasks, at
W ∈ {0.1, 1, 10} and λ as in §3 of the preregistration.

- It is computed **gross of System-1's own price** (ΔU + κn̄), because κ is a property of the backend's deployment,
  not of its resolution. Otherwise κ confounds the AUROC slope, since s2self is expensive and the rules are free.
- Net ΔU is reported alongside.

**Tests.**

- **Primary, as pre-registered:** regress ΔU on AUROC and accuracy, with an LR test for adding accuracy, and run a
  TOST of the System-1 gate against the matched random gate.
- **Secondary (theory check):** ΔU_gross / (W·n̄·π(1 − π)·ē) against J(τ_op) should have slope 1 and intercept 0.
  ē = e + h comes from §4, estimated per (gate, family). Apply the realised-rate correction of Corollary 3.
- **Robustness:**
  - AUROC within strata of the oracle's next-step type (read / write / ask / finish). The strata use gold, but only
    in the analysis.
  - Test-retest agreement of the labels between the identical replicate arms `llm_only_s2lite` and `small_only`,
    computed by replay.
- **Exploratory:** a within-cell τ dose-response, simulated from the shadow and CF branches with an analytic matched
  random gate.

### 9.3 C5 (reflection trigger vs matched-rate random gate, Flash-Lite)

**Primary, unchanged:** paired ΔU of `jev_reflect_trigger_s2lite` against `ratem_reflect_trigger_s2lite`, n = 500,
superiority, with McNemar.

**Secondary readouts:**

- AUROC and J of the gate for `need_fatal` on `meta_shadow_s2lite`;
- precision at the matched rate;
- ē net of placebo;
- predicted ΔU (§9.1) against observed ΔU;
- the realised-rate correction.

**Ceiling rule (pre-declared).**

- The one-step oracle ceiling ΔU_ceiling = W·n̄·π(1 − π)·ē (J = 1, against the matched random gate) is computed from
  the CF branches.
- If its 95% CI at W = 1 lies within ±3.3 pt (the pre-registered MDE), C5 is reported as **"no headroom on this
  suite"**. A null C5 is then not evidence against System-1.
- Context: with ē = 0.3, J = 0.3, π = 0.06 and n̄ = 6.6, the predicted ΔU is about 3.4 pt at W = 1, about the MDE. C5
  is at best borderline-powered on this suite, and that is stated.

### 9.4 C6

- **Primary, unchanged:** NI of `jev_stop_guard` against `llm_only` on Flash.
- **Readouts:** FAR_model (≥ 50 wrong per System-2 model), FAR_nat (Flash-Lite), FRR, the representativeness and
  style checks, FAR by kind, and rescue from natural wrongs only (§6.3).
- **Stop-guard H2 cell:** e and h from the `verdict_reject` CF branch.

### 9.5 `bench/metrics.py`

| Function | Change |
|---|---|
| `guard_error_rates`, `rescue_rate` | Inputs unchanged (`steps[]`, mode `guard`); add `by_kind` (gold_aux.verdict / kind) and a `gold="canonical"` variant |
| `probe_error_rates(trajs)` (new) | Reads `traj.probes`: FAR/FRR per dp × probe_source × operator × kind × D bucket, Wilson CIs, reweighting |
| `memory_read_hits(records)` (new) | hit@k, MRR, none-accuracy for list golds |
| `decision_table` | A list gold means membership (`s1_value ∈ gold`). The tool_select accuracy column uses `gold_aux.acceptable_tools` when present, with a canonical column next to it. `memory_write` rows get p_yes calibration (bool gold; `is_noul_record` already accepts it) |
| `delegation_ratio` | Mode `"generate"` counted as LLM-made |
| `recall_at_k(records, k)` (new) | C3: the top-k of `s1_dist` (or a replayed ranking) against AT(s); k ∈ {1, 3, 5}; canonical secondary |

`controls.trigger_rates` must change in two ways:

1. Skip the `meta.memory` fallback when `memory_write` DecisionLogs exist. Otherwise `memory_write/trigger` is double
   counted.
2. Skip `memory_write: shadow`.

### 9.6 `bench/threshold_theory.py`

- **`decision_pairs`.**
  - Stop shadow: `s2_ok = (stopping == finish_ok)` at finish steps and `(not stopping) == (not world_complete)` at
    other steps.
  - tool_select: s1_ok and s2_ok against AT(s), with `--gold canonical` as an option.
  - It never reads `traj.probes`. That is guaranteed by construction. Placing probes in `steps[]` with
    `mode = "guard_probe"` would *not* have been safe: `decision_pairs` keeps every non-guard mode at `stop`.
- **New `gate` analysis** for trigger and guard decision points:
  - Plug-in rule: fire iff g(p)·(e + h) ≥ c_f/V + h, i.e. τ̂* = g⁻¹((c_f/V + h)/(e + h)), where g is the isotonic
    calibration map of p_yes against the gold.
  - c_f comes from the ledger: the reflection call; for routing, the large − small call cost; for a guard, one retry
    step; for memory, the item's input-token cost.
  - Simulated gate frontier from the shadow and CF branches, with inverse-probability weights for the 20% negative
    sample.
  - The item-8 analogue against a random gate at matched rate.
  - Regret of τ̂* against τ = 0.5 / 0.8 and the best swept τ, on the held-out split. This is H1 extended to gates,
    and it is exploratory.
- **`ZERO_DEFER_COST_DPS`** stays for stop analysed as a cascade decision. The guard's reject branch is priced at one
  retry step.

### 9.7 Score card

Per arm:

- gold coverage (labelled / opportunities) per decision point;
- `gold_status` counts (doomed, unobserved, cf_error, no_oracle);
- the "no opportunity" flags;
- CF coverage and the placebo δ.

Measurement-arm blocks: the probe FAR/FRR table, the efficacy table (e, h, δ) and the H3 parameter-kind composition.
New analysis code lives in `bench/meta_gold.py` (replay of golds, replay of backends, H2 cell assembly, ceiling).

### 9.8 Training harvest (owned by the training agents)

`harvest.py` must:

- read the new golds only from evaluation-disjoint seeds (≥ 10000);
- skip `gold_status ≠ "ok"`;
- never read `traj.probes`.

## 10. Costs

**Unit prices** are dev ledger means (`runs/dev/e2b_dev`, 2026-10-01). Verify them before launch.

| Item | Price |
|---|---|
| Flash tool call | $0.0054 |
| Flash-Lite tool call | $0.0013 |
| Reflection call | Flash-Lite $0.00075, Flash $0.0033 |
| Stand-in Noul | $0.00035–0.00056 |
| LLM-only episode | Flash $0.041, Flash-Lite $0.0096; about 7.6 steps and 6.6 reflection opportunities |

| Item | Estimate |
|---|---|
| Oracle golds in every arm | $0; about 2–4 ms per verdict, under 0.1 s per episode |
| `meta_shadow` (Flash, n = 200) | episodes about $8; inline shadow Nouls about $2; probe scoring about $2; probe generation (forced premature on Flash, transplants on Flash-Lite) about $2. **About $14** |
| `meta_shadow_s2lite` (n = 500) | episodes about $5; inline shadow Nouls about $6; probe scoring about $5; probe generation about $2.5; CF branches (reflection about $1.6, large/Flash about $4.4, rejection about $0.7, placebo about $1) about $8. **About $26** |
| Backend replays | heuristic, rate and lexical $0; s2self about $12; Jev negligible |
| CF at fired steps for the C5 pair (optional) | about $1 |
| New arms of §5.3 | about $40 |
| **Total** | **about $55 measurement plus about $40 arms**, inside the $1.5–2k E2 budget |

**Wall clock.**

- Deferred calls add about 10 System-2 calls and about 25 System-1 calls per Flash-Lite measurement episode, run
  after the episode.
- Probes may be scored concurrently, as separate calls on a thread pool.
- Measurement arms take about 1.5–2.5 times the wall clock of LLM-only. Critical-path latency is unchanged.
- They share the Gemini quota with the stand-in System-1.

**κ and the tournament (issue 4).**

- The stand-in's 69-tool tournament is sequential: κ = $0.0029 and 2.7 s per decision.
- Batch its first-round chunks into one `decide()` call when `batched_questions` is set. This is a uniform change
  across every tool_select arm, frozen before the main runs.
- It affects no gold here. Meta gates ask a single Noul (κ ≈ $0.0004).
- H2 uses gross ΔU, so the stand-in's price does not drive H2.

## 11. Validity threats

1. **Oracle artefacts.**
   - D is the canonical policy's cost-to-go: an upper bound on the optimum.
   - After §2.1 and the early exit, the residual errors affect detour versus progress, never `finish_ok` or
     `world_complete`, which rest on `evaluate`. They can reach `need_fatal` only through the step-budget route
     (D(s′) = ∞ because the canonical plan no longer fits the budget; logged as `gold_aux.fatal_kind = budget`, while
     `terminal` fatals are proven by the episode ending): that share is reported and excluded in a sensitivity row, and
     the implication audit in `meta_gold audit` (executed failing verdict ⇒ episode failed; `finish_ok` ⇒ success;
     violations must be 0) is reported per arm (`PREREGISTRATION.md` A6.7).
   - The dataflow map is hand-written. Guarded by the sufficiency test (§12).
2. **Policy-dependent golds and nondeterminism.**
   - `need_fatal` is defined for one System-2 at T = 0, and Gemini is not deterministic at T = 0.
   - Label noise attenuates every backend's AUROC equally: comparisons stay fair, absolute values are biased down.
   - Report the placebo δ and the replicate-arm test-retest agreement.
3. **Concentrated prevalence.**
   - Positives are almost all Flash-Lite's wrong `multi` totals at the pre-finish state. Flash has about none.
   - H2 and C5 therefore largely test "detect a pending wrong total".
   - Mitigations: stratified AUROC, the "no opportunity" rule, and L3 approximations for generality.
4. **Question–gold alignment.** Rewording is required. The old wording is logged as an ablation. If A4 is rejected,
   report the retrospective `retro` gold as the question-literal one and `need_fatal` as the value gold, explicitly.
5. **Selective labels.** Live trigger arms lack labels where the gate fired. All AUROCs come from measurement arms.
   Live labels are a consistency check.
6. **Distribution shift between measurement and live arms.** Live gates fire at about 7–8% of opportunities, and only
   later states change. The shift is reported as the AUROC difference on the live arms' observed labels.
7. **One-step CF.** Validated by the share of positives at the final step, and by predicted against observed C5 ΔU.
8. **Strict world.** Gates can only *prevent* errors here, never recover from them. Recovery claims need ALFWorld or
   τ².
9. **Memory has VoI ≈ 0 for System-2 on JevGym.** The memory cells are negative controls (cost only), and the read
   mode is held fixed in the H2 pair. Success-side memory claims come from LoCoMo and LongMemEval.
10. **Probe representativeness.**
    - Forced or transplanted proposals may be easier or harder to catch than deployment errors.
    - Mitigations: model-generated primary sources, oversampling of D = 2 states, kind strata, reweighting, the
      FAR_nat equivalence test and the style check.
    - FAR_model is never presented as "the" FAR.
11. **The bimodal stand-in.** p_yes is about 0 or 1, so AUROC ≈ (1 + J)/2 and τ sweeps are flat. H2 cannot separate
    AUROC from J until a prefill backend is available. Declared.
12. **Leakage surfaces, each with its rule and test:**
    - env snapshots held by the harness: never rendered;
    - probe and CF content: no oracle authoring;
    - batching probes with live questions: forbidden;
    - `s1_cache`: keys include the question text, so a probe identical to a natural proposal legitimately shares its
      answer;
    - `Question.meta`: unchanged;
    - training harvest: §9.8;
    - over-the-wire `infer_dp` with an answer containing decision-point keywords: the harness always sets
      `decision_point`, and adapters quote and truncate answers.
13. **Missing labels** (`cf_error`, rate limits) correlated with episode length could bias AUROC. They fail open,
    coverage is reported per arm, and they never drop a task.
14. **Comparability.** The dev FAR/FRR (canonical gold, old wording) cannot be compared with the main runs. They were
    never reportable. The BENCHMARK_SPEC known-limitations text and the C6 wording change together with this spec.
15. **Freeze timing.** See §14.

## 12. Test plan (offline: simulated LLM and System-1, `pytest -q`)

**Oracle (`tests/test_meta_gold_oracle.py`)**

- `SimulatedLLM(error_rate=0)` solves 100% of tasks, and every executed action is progress, with canonical ⊂ progress.
- Every alternative order of each clarify variant (including `clarify-8`'s ask → cancel → finish) is progress, and the
  final finish has `finish_ok`.
- Out-of-order `multi` jobs are progress. `get_orders(all)` is progress after §2.1. `refund` with amount `"458.240"`
  is progress. `get_order` is a detour. A wrong ask slot is a detour.
- An injected wrong write, a near-duplicate write and acting before the ask are each fatal. A wrong total is
  `wrong_answer`. An early finish is `premature`.
- Early exit: D = 1 at a shortcut-complete state.
- Computing golds leaves the real episode byte-identical (observations, RNG draws).
- **Online/offline equivalence:** the logged golds equal the offline replay on a fixed set of simulated and dev
  trajectories.
- **Memory sufficiency:** an oracle agent that may read only the instruction plus the `mem_need`-positive offered texts
  solves 100%. Heartbeat numbers never match an atom.

**Harness logging (`tests/test_meta_gold_harness.py`)**

- Every meta `DecisionLog` carries `state`, `gold`, `gold_status` and `gold_aux.v`.
- Memory gates log one record per offer.
- `memory_write: shadow` stores nothing.
- A measurement arm's trajectory (actions, observations, success, `usage.total`) equals the same arm with
  `measure: {}`.
- Deferred CF and probe calls appear only under `usage.shadow.by_measure`. They run after `evaluate`. Failures are
  counted, not raised.

**Leakage (`tests/test_meta_gold_leakage.py`)**

- **Canary test:** patch the gold values (the plan's finish answer, the gold order id) to unique canary strings.
  Assert that no serialised System-1 request, System-2 request, rendered state or note contains a canary unless the
  value is visible in an observation.
- `Question.meta` is unchanged for every decision point. The existing test that controls never read meta is
  extended.
- Probes are always single-question calls.

**Probes**

- Edits use only state-visible values. Edits labelled right are discarded.
- Transplants are kept whatever their label.
- Forced premature probes are premature.
- Probes never appear in `decision_records`, `trigger_rates` or `threshold_theory.decision_pairs`.

**Metrics (hand-computed, `tests/test_metrics.py`)**

- `probe_error_rates`, `memory_read_hits`, list-gold `decision_table`, `recall_at_k` against AT(s), the
  delegation ratio with `generate`, and `trigger_rates` without the memory double count.
- **VoI identity:** a simulated gate with known π, J, e and h reproduces §9.1 within Monte-Carlo error, including
  the rate-mismatch correction.

**Free-text policy**

- The §7.2 classification table is frozen in a test.
- `finish.answer` can never come from System-1 (the `899659` regression).
- Free parameters trigger exactly one forced System-2 call.
- `x-jev-fill` overrides.
- `is_copyable()` is unchanged.

**Wording**

- `infer_dp` maps the new `Q_REFLECTION` to reflection and the new `Q_STOP_GUARD` (with answers containing
  "complete", "stuck" and digits) to stop.
- `STOP_GUARD_RE` matches.

**Live smoke (Gemini, about $1, not reportable)**

- 5 tasks of each measurement arm, checking CF cross-model calls (thought signatures), probe counts and costs.
- Then the replay and `meta_gold` analysis end to end.

## 13. Adjudication record (P1 against P2)

| Topic | P1 | P2 | Adopted | Why |
|---|---|---|---|---|
| Shared machinery | deepcopy verdicts plus deferred CF System-2 calls for most labels | pure oracle functions, no model calls; labels from shadow arms; back-patching | **P2 for primary golds** (pure oracle, $0, replayable); **P1's deferred CF only for efficacy (e, h), placebo and probe generation** | Primary labels need no model in the loop when the measurement arm's executed action *is* the default action. Efficacy needs CF, and deferral keeps it off the trajectory and off the prompt cache |
| Oracle robustness | D via canonical rollout; accepts redundant reads as detours | P0 fix (tolerant `_done_step`, information-equivalent reads); claims world_complete ⇔ d* = 1 | **P0 fix plus an early-exit rollout (new)** | P2's equivalence is false for canonical rollouts after shortcuts (`clarify-8`); the early exit fixes it. Without the P0 fix a string-typed amount is mislabelled fatal under either design |
| reflection: primary | `need_fatal` | R+ = not progress (detours included) | **`need_fatal`**; `need_any`, `retro` secondary | Value-aligned: only failing actions change success. The labels differ by 1 detour in dev |
| reflection: label source | trigger arm, with CF at fired steps | shadow arms (AUROC); fired steps unobserved | **Shadow (measurement arms) for AUROC**; fired-step CF optional, as a consistency check | Avoids selective labels without making live arms depend on CF |
| reflection: efficacy and ceiling | shadow-CF arm at every opportunity plus 10% placebo | live oracle-trigger ceiling arm | **CF at all positives and 20% of negatives, plus placebo; ceiling computed analytically**; no oracle arm | No gold-driven episode; positives sit at the final step, so the one-step ceiling is near-exact |
| reflection: wording | "Before the next step: is the agent at risk…" | "Should the agent pause and re-check… stuck or off track…" | **P2's wording**, with the old wording logged as an ablation (P1) | P2's wording keeps `infer_dp` working (verified: P1's maps to None) and is generic |
| routing | `need_fatal` of the small model; Flash routing shadow with CF small calls at every step | G_route from the Flash-Lite meta-shadow (free); large-model check at positives | **P1's severity, P2's source**; large-model CF at positives, 20% of negatives and placebo | The routers' default policy *is* the small model; free labels on the right distribution |
| stop guard | `finish_ok`; typed; doomed excluded | the same; `canonical_done` kept | **Merged** (identical substance); reject note and question from both | — |
| stop trigger | `complete(s)`; report "no opportunity" | the same plus an augmented query at finish steps | **Both**: augmented queries as probe-list records, descriptive only | Gives an AUROC readout without polluting H2 or the rates |
| tool guard | progress; FAR_fatal primary | progress; FAR against progress primary, FATAL-only alongside | **FAR against progress as headline (question-aligned), FAR_fatal next to it** | The question asks "the right next step" |
| acceptable tool set | not specified | tools of undone progress steps | **Information-feasible filter (new)** | Without it, write tools count as "acceptable" before their IDs are known |
| memory_write: primary | `info_need` (excluding instruction and store values) | M (store-independent, on offered text) | **P2's M**; P1's "new" and a far variant as aux | Question-literal and arm-independent. P1 itself flagged the store dependence |
| memory_write: far | first consumer ≥ 2 steps ahead | an atom at distance ≥ 3 | **d_last ≥ 3**, with distances logged | Isolates long-horizon recall (the `multi` amounts); any threshold can be recomputed |
| memory_write: matching | JSON-key regex | token-boundary literal | **Literal boundary plus numeric tolerance** on the exact offered text | Simpler, generic; the noise-exclusion and sufficiency tests cover collisions |
| memory_write: shadow mode | the store follows a reference policy | stores nothing | **Stores nothing** | Keeps the measurement arm identical to LLM-only |
| memory_read | gold set, hit@k, MRR, shadow mode | gold set, descriptive only | **Gold set, descriptive; no read shadow mode** | Near trivial on JevGym; L3 suites have exact evidence |
| Probe placement | deferred, inside the C6 / guard arms plus shadow arms; A/A check | measurement arms only | **Measurement arms only** | Removes any risk to primary arms and the need for an A/A check |
| Probe sources | forced premature, transplant (primary); edits robustness only | synthetic operators T1–T6 (primary for Flash), natural-mix reweighting | **Model-generated as primary for stop FAR; T1–T6 for robustness and as the only fatal tool probes** | Model-generated probes match deployment style. Fatal tool errors never occur naturally in dev, so only edits supply them |
| Probe storage | separate `traj['probes']` | `steps[]` with `mode = guard_probe` | **Separate `Trajectory.probes` with `mode = guard_probe`** | P2's "existing filters exclude them" fails for `threshold_theory` at stop |
| Right probes | transplants kept whatever their label | never inject right probes | **Model-generated right probes allowed; oracle-authored never** | The leakage concern applies to authorship, not to labels |
| Live injection | optional fault-injection arm | optional inject plus no-guard arms | **Neither** | Rescue is inflated by construction; §6.5 |
| C6 readout | FAR_model ≥ 50 pooled; FAR_nat equivalence ±0.10 | FAR_natural (Flash-Lite) plus per-operator probe FAR, reweighted | **FAR_model primary, FAR_nat plus equivalence, per-operator FAR as robustness, style check** | — |
| Free text: default | lexicon-based GENERATE, else COPY | whitelist COPY, else FREE | **P2's conservative default plus P1's `x-jev-fill` override and coverage valve** | Avoids inflating the copy class; the valve tests the actual VoI condition |
| Free text: optional params | omitted when no LLM call is otherwise made | always filled by System-2 | **Always filled by System-2** | Omission exploits a JevGym simulator artefact |
| Free text: logging | `generate` DecisionLogs | `param_kind` on records | **Both** | — |
| H2 dependent variable and cells | gross VoI; ≥30/≥30; rate backends as J = 0 anchors | net ΔU; ≥30/≥30; paired replays | **≥30/≥30; gross primary, net reported; rate backends as controls only; replays for AUROC** | κ confounds the AUROC slope; rate backends against themselves are ΔU ≡ 0 by construction |
| VoI identity | proposition with corollaries | — | **Adopted**, as a secondary theory check | Verified algebraically (§9.1) |
| Threshold rule | CF-based gate frontier | τ̂* = g⁻¹(c_f/(V·r)) | **Generalised: g⁻¹((c_f/V + h)/(e + h))** | It reduces to P2's rule at h = 0 |
| C5 power | pre-check; declare underpowered if low | ceiling rule: "no headroom" | **Both, as pre-declared post-hoc rules** (no pre-freeze dev run needed) | — |
| Shadow batching | do not batch probes | batch all questions about one state | **No cross-question batching for generative backends** | Batched generative answers can condition on each other, so shadow answers would differ from live single-question answers |

## 14. Implementation order and slip rule

**Freeze-critical** (they change primary arms and must land and pass a 5-task dev re-check before 2026-10-03):

1. `Q_REFLECTION` and `Q_STOP_GUARD` (with the answer), the stop rejection note, and `_stop_check` receiving the
   arguments.
2. The free-text policy: `ParamInfo.kind`, the forced System-2 fill for free parameters, the frozen JevGym table and
   the dev-seed coverage table.
3. The §5.3 arm definitions, and `state` logged on every meta `DecisionLog`.
4. This spec's gold definitions, frozen as amendment A4.

**Not freeze-critical:**

- the P0 oracle fix;
- gold logging;
- `memory_write: shadow`, `measure`, the CF and probe runners;
- `meta_gold.py` and the metric code.

These must exist before the *analysis*. JevGym golds of executed actions are recomputed by deterministic replay of
the logged actions, and the measurement arms run on the same task list after the main runs.

Anything not in place when the analysis runs is filed as a dated deviation. The readouts it affects (H2 cells, C5
secondary, C6 FAR) are then reported as missing, never replaced silently.

## 15. Round-3 review: implementation decisions (2026-10-01)

An adversarial review of the implementation found defects in how the analysis consumed the golds. Each fix below has
a regression test in `tests/test_meta_gold.py` (section "round-3 review regressions"). None changes a gold definition.

| Area | Decision |
|---|---|
| H2 population of replays (§6.2, §9.2) | Replay rows enter H2 only from `steps[]` shadow records (`source = steps`, mode `shadow`, `gold_status = ok`) -- the population the measurement arm's own backend is scored on. Probe-list rows (`wording_v1`, `augmented_stop_trigger`, guard probes) feed the probe tables only (`meta_gold.h2_replay_records`; `_gate_of_row` returns None for them). |
| Backend identity (§9.2) | Every episode logs `meta.s1 = {id, describe}` (id = hash of the full `describe()` without the cache flag); replay rows carry `backend_id`. H2 cells and the VoI readout use the LIVE arm's System-1: the measurement arm's log when the ids match, else that spec's replay. With neither, the cell has no statistics, is not estimable, and `h2_regression` skips it -- never another backend's answers (the earlier fallback gave every cell the stand-in's AUROC). |
| Measurement arm per gate (§3.2) | Routing cells use the measurement arm whose System-2 is the routers' small model (`meta_shadow_s2lite`); other gates the arm of the live arm's System-2 family (`meta_gold.measure_arm_for`). e / h always come from the measurement arm's CF branches (a property of the intervention, not of the backend). |
| Live trigger rows (§9.2, §11.5–6) | The score card's "H2 inputs" table holds measurement-arm shadow rows only. Live trigger / guard rows are a separate consistency check: AUROC on the labels observed at executed actions (fired-step CF labels never enter it), the labelled share and the realised fire rate per opportunity; reflection / routing triggers get no J / accuracy / estimability. `cf.fired_steps` is set on every Flash-Lite reflection-trigger arm of `baselines_jevgym.yaml` alike. |
| Stop-guard retries (§3.3, §6.5) | A rejected finish is appended to System-1's history, so the resubmission is asked at a new state (a deterministic gate no longer re-answers a byte-identical question until `max_steps`); guard records carry `retry`. Matched rates (`trigger_rates`), guard FAR / FRR, FAR_nat and gate fire rates count each proposal once (`metrics.opportunity_key`: stop guard = task + question with the answer); retries are reported separately. |
| Matched-rate protocol (§5.3) | `trigger_rates` is read off ONE arm's file (a directory pools every arm and warns; `by_arm=True` groups). The `ratem*` backends are `placeholder: true`: the runner skips their arms until the backend is replaced with the measured rate, and a resume refuses an arm whose logged System-1 spec changed. |
| Probe statistics (§6.3) | FAR / FRR CIs are task-clustered percentile bootstraps (`far_ci` / `frr_ci`; undefined with fewer than two tasks), Wilson intervals alongside. Natural shadow finishes are thresholded at the arm's `tau_ref`, like the probes. The representativeness TOST (FAR_nat − FAR_model within ±0.10, 90% task-bootstrap CI) and the H2 TOST (gate vs. matched random gate within ±2 pt of success, paired task bootstrap; with fewer than 5 discordant tasks the one-sided Wilson bound of the discordance rate must lie below the margin) are implemented (`metrics.tost_far_nat_model`, `metrics.tost_paired`). |
| C3 recall@k (§9.5) | Assist mode keeps max(2, k) survivors per tournament chunk (the stand-in's 69-tool tournament had 4 finalists, so "top-5" offered 4 tools), and every tournament logs System-1's full ranking (`DecisionLog.s1_rank`: finalists, then each round's eliminated options by score). Recall@k ranks `s1_rank`; an older tournament log whose ranking is incomplete at k is undefined (`n_undefined@k`). `offered_k` is reported next to C3. |
| Two run directories (§5) | The C5 pair runs in E2c, the measurement arms in E2b: `scorecard --measure-run` and `meta_gold h2 --measure-run` load the measurement arms from the E2b run. The E2b duplicate arm is renamed `jev_reflect_trigger_s2lite_e2b`. |
| Threshold theory gates (§9.6) | Without `--arm`, `--gate` pools only measurement arms of ONE System-2 family (shadow records); c_f comes from `--cost-arm` or the analysed arms' ledger, where a measurement arm prices a firing from its counterfactual calls (reflection call; large-model step for routing). |
| Probe placement (§6.4) | `measure.probes` requires every decision point `off` or `shadow` (reflection `always`, memory `always` / `llm`, `llm_only` are rejected), and the score card's shadow-only test reads the logged config the same way. Measurement calls never count as strong calls. |
| Oracle (§2.1) | A read step counts as done only as `_step` served it (exact `user_id` / `status` / `item`); writes keep tolerant argument matching but must have taken effect. A lower-cased or padded id no longer turns a recoverable state into "doomed". |
| Efficacy (§4) | `cf_efficacy` returns e / h = NaN (`net = False`) when a placebo term is missing; the VoI prediction and the ceiling then report "not net of the placebo" instead of using raw branch rates. |
