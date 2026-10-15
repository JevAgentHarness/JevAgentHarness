# Decision points and modes

The paper studies a decision model at eight **decision points** of an agent loop (the next tool, its arguments,
memory writes and reads, reflection, stopping, routing to a stronger or cheaper LLM, and the action of a text game),
each in one of five **modes** that say who proposes an answer and who has the last word (replace, cascade, assist,
guard, trigger). It ran this grid in its own agent loop and reported each cell as the change in task success and in
cost against the LLM alone (the module map, Figure 5 of the paper).

The proxy sits between an unmodified harness and its LLM provider. It sees each request (messages and tools) before
the model runs and each reply before the client gets it. That is enough for five decision points:

| Decision point | Modes | What the decision model is asked | Default |
|---|---|---|---|
| `routing` | `off` `shadow` `cascade` `guard` | is the next step `routine` or `hard`? (guard: should a stronger model redo the fast model's step?) | `cascade` |
| `tool_select` | `off` `shadow` `assist` `cascade` `guard` | which tool comes next? (the request's tool names, plus "reply without a tool") | `off` |
| `stop` | `off` `shadow` `strong` `guard` `trigger` | the agent wants to end its turn: is the task done? (trigger: is it already done?) | `strong` |
| `reflection` | `off` `shadow` `trigger` `always` | should the agent pause and re-check before acting? | `off` |
| `action` | `off` `shadow` `guard` | could this proposed call be irreversible or risky? | `off` |

**What is benchmarked.** The defaults (routing in `cascade` mode plus the finish check, `stop: strong`) are the
setting of the README results. The other modes were measured only on the 15-task development split of
Terminal-Bench 2.0 ([Dev-split measurements](#dev-split-measurements)), which is enough to choose settings and to see
what each point does, not to claim a change in success. The paper's
evidence for each cell comes from its own agent loop and tasks (mostly simulated customer-service tasks with Gemini
3.5 Flash as the agent's LLM), so treat it as a prior, not a promise, and measure on your own tasks (see
[Evaluating a mode](#evaluating-a-mode-on-your-tasks)).

**`shadow`** is available at every point. The decision model is asked the point's question where an acting mode would
ask it, its answer is logged next to what the agent actually did, and nothing changes. Shadow logs are what `jah fit`
fits thresholds on.

**Failures.** If the decision model errors or times out (`router.decision_timeout_s`), the point falls back to its
LLM-only behaviour for that step: no forced tool, every tool offered, no note, the reply accepted (the finish check
and the routing guard fall back to the strong tier, as the router does). A decision-model outage never stops the
agent.

## Presets

```bash
jah serve --preset paper        # or `preset: paper` in the config
```

| Preset | Adds to the default (routing cascade + finish check) | Use it to |
|---|---|---|
| `route` | nothing (the default) | reproduce the README setting |
| `paper` | `tool_select: cascade` (tau 0.99 until you fit it) | try the paper's tool-choice cascade on an agent with many tools (on coding agents with one to four tools it had nothing reliable to act on: [measurements](#dev-split-measurements)) |
| `safe` | `action: guard` | get a second look at destructive shell and write calls (about half a decision call per task on Terminal-Bench) |
| `shadow` | `tool_select`, `stop`, `reflection`, `action` in `shadow` | collect data for `jah fit`; makes the same LLM calls as the default |

A point set under `decisions:` wins over the preset. The `shadow` preset keeps routing in `cascade` mode, because the
stop point's labels come from the strong tier re-doing fast-tier final answers. For a strong-only baseline that still
logs every question, add `--set decisions.routing.mode=shadow`.

## The paper's module map and the proxy

Each paper cell: change in task success (points) | cost saved (%, negative = costlier), Jev with Gemini 3.5 Flash, from
the paper's Figure 5. "Proxy" says what the proxy offers in that cell.

| Point | Replace | Cascade | Assist | Guard | Trigger |
|---|---|---|---|---|---|
| Tool choice | −19.0 \| 67. Proxy: no (= cascade at tau 0, the LLM still writes the arguments) | **+0.0 \| 14.** Proxy: `tool_select: cascade` | −7.8 \| 54. Proxy: `tool_select: assist` | +0.0 \| 0. Proxy: `tool_select: guard` (another question, below) | +0.0 \| −14. Proxy: no (the guard covers it) |
| Tool arguments | −20.0 \| 88 | **+0.0 \| 14** | +0.0 \| 9 | +0.0 \| −7 | +0.0 \| −1 |
| Memory write | +0.5 \| −47 | −1.0 \| −48 | +0.0 \| −74 | +0.0 \| −60 | −1.0 \| −39 |
| Memory read | +0.5 \| −11 | +0.0 \| −21 | +0.0 \| −5 | +0.0 \| −140 | +0.0 \| −1 |
| Reflection | −0.5 \| −7 | +0.5 \| −25 | −0.5 \| −66 | +0.0 \| −61 | +0.0 \| 1. Proxy: `reflection: trigger` |
| Stop | +0.5 \| −7 | +0.0 \| −1 | +0.0 \| −17 | −0.6 \| −6. Proxy: `stop: guard` | +0.0 \| 0. Proxy: `stop: trigger` |
| Routing | +4.0 \| −7 | −4.0 \| −113 | −2.0 \| −31 | +0.0 \| −23. Proxy: `routing: guard` (another check, below) | +0.6 \| −36. Proxy: `routing: cascade` |
| Action (text games) | −64.3 \| 91 | +0.0 \| 14 | −40.7 \| −182 | −0.7 \| −87 | −2.1 \| −111 |

How to read it for the proxy:

- **Only the cascades at tool choice and tool arguments saved cost without losing success**, and only with a
  threshold fitted on the agent's own runs (fitted at 1.00 for tools and 0.99 for arguments with Jev and Flash: Jev
  took 27.2% of the tool choices and 20.8% of the argument choices, and the cost fell by 14.2% with the outcome of
  none of 500 tasks changed). The same delegation at a fixed threshold of 0.5 lost 21.5 points.
- **The modes at yes/no points (reflection, stop, memory) did not save cost.** Each keeps an LLM call in the loop, and
  a check that fires adds one. Use them for safety or data, not for savings.
- **Routing**: the paper's routing row compares with the *cheap* LLM alone (sending some of Flash-Lite's steps to
  Flash cost 36% more). The proxy's routing is compared with the *strong* model alone, which is where its savings come
  from (README: 28–64% on Terminal-Bench 2.0 with a fast tier 4–7× cheaper per token).
- **Most live failures were irreversible first errors**: in the paper, every one of Jev's 34 failed tasks began with
  an irreversible wrong tool choice (usually cancelling where the agent should have asked). That motivates the
  `action` guard. The paper's "Action" row is a different thing (the action of a text game) and does not transfer.
- **Coding agents** act through free text (a shell command, a patch), so a decision model can mostly choose which LLM
  writes a step. In the paper's coding runs the savings were small (7–10.5%), and a proxy that sent "read-only"
  steps to the fast LLM and discarded fast-LLM writes never reached a full run: Jev could hardly tell read-only steps
  from the others. Expect `tool_select` modes to matter for agents with many tools (MCP servers, customer-service or
  data tools) more than for agents whose every step is `bash`.

Not offered in the proxy, and why:

| Cell | Why not |
|---|---|
| `tool_params` (all modes) | The LLM writes a call's arguments in the same reply as the tool, and providers cannot pre-fill some arguments of a forced call. Filling a whole call without the LLM would send a decision-model-written call to the harness unchecked. |
| `memory_write`, `memory_read` | The harness owns its context. The proxy has no memory module, and pruning the history it forwards would change what the agent sees and break prompt caching. |
| `tool_select: replace` | The LLM still writes the arguments, so "the decision model alone" is the cascade at tau 0, which lost 19 points in the paper. Set `tau: 0` if you want it. |
| `tool_select: trigger` | The paper's verification trigger (the LLM re-checks its own call when the decision model asks) cost 14% more and changed nothing. `guard` covers the useful part. |
| `routing: trigger` | With a yes/no question, "escalate when p(hard) ≥ tau" is both the trigger and the cascade. `trigger` is accepted as another name for `cascade`. |
| `assist` at yes/no points | Shows the decision model's answer to the LLM as advice. It needs an extra LLM call per point and saved nothing in the paper. |

## Each point in detail

Settings are per point under `decisions:` (or under one model name, see [Per model name](#per-model-name)). A string
is a shorthand for the mode: `tool_select: shadow` equals `tool_select: {mode: shadow}`.

### routing

The router of the README: should the next step go to the strong or the fast tier?

| Mode | What happens | Expected effect |
|---|---|---|
| `cascade` (default) | Before the call: the strong tier takes the step if p(hard) ≥ tau, else the fast tier. The opening step(s) (`router.first_steps_strong`), `stick_strong` and `max_fast_streak` still apply. | The README results. |
| `guard` | The fast tier writes every routable step. After the call, the decision model reads the proposed step and answers `keep` or `redo`. If p(redo) ≥ tau, the strong tier redoes the step. | Every step pays a fast call, flagged steps also a strong one, and each check waits for the fast reply. It saves more than `cascade` only when the fast tier is usually right. On the dev split it cost more than the cascade ([measurements](#dev-split-measurements)). |
| `shadow` | Every step goes to the strong tier. The routing question is asked and logged. | Strong-only cost plus the decision calls. A baseline that shows where the router would have sent each step. |
| `off` | Every step goes to the strong tier, no question. | A strong-only baseline under a routed model name (useful with `tool_select: cascade`). |

`tau` defaults to `router.tau`. With `decisions.routing.tau` set, that value wins.

### tool_select

The decision model gets the request's tool names as options (each with the first line of its description, up to 160
characters) plus `reply_without_tool` (left out when the client sent `tool_choice: required`). The point is skipped
when the request has no tools, the client forced a tool or forbade tools. With more than `max_options` tools (default
100; Jev accepts up to 255), the tools are split into chunks that are asked at the same time, and the likeliest `k`
(at least 3) of each chunk go to one final question.

| Mode | What happens | Expected effect |
|---|---|---|
| `cascade` | If the top option is a tool with p ≥ tau, the step runs on the cascade's `tier` (default: the fast tier) with that tool forced (`tool_choice`), so the LLM only writes the arguments. Otherwise the step is routed as usual. | The paper's main cell: +0.0 points and 14% saved, at a fitted tau. Without a fitted tau, expect lost success. |
| `assist` | Only the `k` likeliest tools (default 5), plus every tool already used in the session, are sent to the LLM. Applies to requests with at least `min_tools` tools (default 12). | Fewer tool-schema tokens for agents with many tools. The paper: −7.8 points, 54% saved with a three-tool shortlist. A changing tool list also defeats the providers' prompt caching of the tool block. |
| `guard` | The LLM proposes as usual. The question is asked at the same time as the LLM call. If the decision model is confident (p ≥ tau) in a different tool than the LLM called (or in a tool while the LLM called none), the strong tier redoes the step once, with a note that the proposed call was doubted. | A safety check. The paper's tool guard (a yes/no "is this the right next step?") saved nothing. Flagged steps cost a strong call. |
| `shadow` | The question is asked at every step (the opening step included) and logged with the tool the LLM called. | Data for `jah fit`. No change, no added latency unless the decision model is slower than the LLM. |

How the cascade combines with routing: it never moves a step that routing sends to the strong tier (a `hard` step, an
opening step, `stick_strong` or `max_fast_streak`), because the arguments of a hard step need the strong model. With
routing in `cascade` mode it therefore only fixes the tool of routine steps, which already run on the fast tier. Its
cost saving comes when routing is `off` or `shadow` (one strong model by default, as in the paper):

```yaml
decisions:
  routing: off                                # every step on the strong tier ...
  tool_select: {mode: cascade, tau: 0.98}     # ... except confident tool choices, which the fast tier fills in
```

Options: `tau` (default 0.99), `tier` (`fast`, `strong` or a tier name), `force` (default true; false only picks the
tier and lets the LLM choose the tool, for upstreams that reject a forced `tool_choice`), `k`, `min_tools`,
`max_options`. A forced call is sent as Gemini's `functionCallingConfig {mode: ANY, allowedFunctionNames: [tool]}`,
OpenAI Chat's `tool_choice: {type: function, function: {name}}` or the Responses API's
`tool_choice: {type: function, name}`. If the forced call fails, the step is re-sent unforced on the strong tier
(`reason: fast_error`). The tool modes act from step `router.first_steps_strong` on (the opening step plans with
every tool).

### stop

When the agent wants to end its turn (a reply with no tool call to a request that offered tools):

| Mode | What happens | Expected effect |
|---|---|---|
| `strong` (default) | A final answer from any tier other than the strong one is redone by the strong tier, which decides (the README's finish check; `router.verify_finish: true`). | The README setting. |
| `guard` | The decision model reads the proposed final reply (head and tail, up to 1,500 characters) and answers `done` or `not_done`. If p(done) ≥ tau (default 0.9), the fast tier's answer is accepted; otherwise the strong tier redoes the step. With `scope: all`, the strong tier's own final answers are checked too, and a rejected one is redone on the strong tier with a note to re-check. | Cheaper than `strong` when the fast tier's final answers are usually right: an accepted answer saves the strong call. The paper's stop guard kept success but cost 6% more, and Jev's rejections rarely turned a failing episode around (4 of 52). |
| `trigger` | Before each step (from step 1), the decision model is asked whether the task is already complete. If p(yes) ≥ tau, the step goes to the strong tier with a note inviting it to finish if the task is done. Fast-tier final answers are still redone by the strong tier, as in `strong`. | For agents that keep working after the task is done. The paper's stop trigger never fired in its development check and only added calls. |
| `shadow` | Behaves as `strong`, and asks the guard question at every fast-tier final answer (`scope: all`: every final answer). The strong redo's verdict is the label `jah fit` uses. | Data for fitting the guard. |
| `off` | No check (`router.verify_finish: false`). | |

**Harnesses that finish through a tool.** mini-swe-agent ends with a bash call (`echo
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`), not with a text reply, so by default the stop point (and the README's finish
check) never sees its finish; Terminus-2 sends no native tools, so the point does not apply to it either. `submit`
lists regexes on a call's text (`name(arguments)`) that mark the call as the agent ending its turn; every stop mode
then treats such a reply like a final text reply:

```yaml
decisions:
  stop: {submit: ['COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT']}     # the finish check now covers mini-swe-agent
```

### reflection

| Mode | What happens | Expected effect |
|---|---|---|
| `trigger` | Before each step (from step 1): "should the agent pause and re-check: is it stuck or off track, repeating itself, or about to make a wrong move?" If p(yes) ≥ tau (default 0.8), the step goes to the strong tier with a note asking it to check its progress and plan before acting. | A recovery aid. In the paper the reflection trigger kept success and saved about 1%. In the proxy every step it fires on moves to the strong tier, but with routing on most of those steps go there anyway, so it added no measurable cost on the dev split ([measurements](#dev-split-measurements)). |
| `always` | Every `every`-th step (default 5) gets the note and the strong tier, without a decision call. | The reference that the trigger should beat. |
| `shadow` | Asked and logged. | Shows how often the trigger would fire. |

The rules backend (`{type: rules}`) answers this question without a model: yes after an error, a failing test or a
repeated action.

### action

A second look at proposed calls that could do harm that is hard to undo. A cheap deterministic pre-filter decides
which calls go to the decision model at all, so ordinary steps cost nothing:

- `prefilter: risky` (default): calls whose arguments match a risky pattern (deleting files with `rm`, `find -delete`
  or `*** Delete File` in a patch; `git push`, `reset --hard`, `clean -f`, `branch -D`, rebases and amends; `DROP` /
  `TRUNCATE` / `DELETE FROM`; `sudo`, `mkfs`, `dd`, `chmod -R`; piping a download into a shell; publishing,
  deploying and deleting with `npm`, `docker`, `kubectl`, `terraform`, `aws`, `gcloud`, `gh` and HTTP `DELETE` /
  `PUT` / `POST` with curl), or whose tool name says it deletes, cancels, refunds, sends, publishes, deploys or
  merges. Add your own with `patterns:` (regexes on the call's argument text).
- `prefilter: command`: every call to a command or write tool (bash, shell, exec, apply_patch, write, edit, ...;
  extend with `tools:`), plus the risky ones.
- `prefilter: all`: every call.

| Mode | What happens | Expected effect |
|---|---|---|
| `guard` | For each candidate call (up to four per reply, asked at the same time): "could this be irreversible or risky?" If p(risky) ≥ tau (default 0.5), the strong tier re-checks the step with a note naming the call: make it again if it is needed and safe, else take a safer step. The call is never dropped silently: the strong tier's reply is what the harness gets. | A safety check, not a saving: each flagged call costs a strong call. On the dev split it never fired ([measurements](#dev-split-measurements)). Keep the harness's own approval and sandboxing: this guard should never be the only control on an irreversible action. |
| `shadow` | Asked and logged. | Shows how often the guard would fire, and on what. |

`on_error: accept` (default) lets a call through when the decision model fails; `on_error: recheck` sends it to the
strong tier instead. The rules backend flags every call that matched a risky pattern, so `safe` works without a
decision-model key.

### One step, in order

1. The routing question and the pre-call questions (tool choice, reflection, stop trigger) are asked at the same
   time. Questions that cannot change the step are skipped (for example the tool cascade on an opening step).
2. Reflection and the stop trigger may send the step to the strong tier with a note. The tool cascade may move a
   routine step to its tier with a forced tool, or `assist` may shortlist the tools.
3. The LLM call. A failed fast or forced call is re-sent once, plain, on the strong tier.
4. The post-call checks run at the same time on the reply: the finish check or stop guard (no tool call), the tool
   guard, the routing guard, the action guard (tool calls). If any flags the reply, the strong tier redoes the step
   once, with the notes of the checks that flagged it. If that redo proposes a risky action, the action guard gets one
   more look. A step therefore gets at most three LLM replies (plus the re-send of a failed fast or forced call).
5. Shadow questions are collected and labelled with what the LLM did, and everything is logged.

Notes are appended to the last message of the request (the latest tool result or user message), so the conversation
keeps its shape on every API and a cached prefix stays a prefix. A request changed by a decision point (a note, a
forced or shortlisted tool set) never refreshes a Gemini context cache.

A retried request (same session and step) reuses that step's decisions: the same notes, tier and tool set, and no new
decision calls.

## Fitting thresholds

The paper fits each point's threshold on the agent's own runs, so the decision model's answer is used only where it is
reliable. `jah fit` does a simplified version on the proxy's logs. The target is agreement with the LLM: for each point
with labels, it picks the smallest tau at which the decisions the point would act on agree with the LLM at the target
rate (default 95%), on at least `--min-count` decisions (default 30), judged by a one-sided Wilson lower bound
(`--z`, default 1.645), so that a few lucky agreements do not pass. If no tau qualifies, keep the LLM deciding.

| Point | Acts when | Agrees when | Labels come from |
|---|---|---|---|
| `tool_select` | the top option is a tool with p ≥ tau | it is the tool the LLM called | `shadow` and `guard` logs, and steps the cascade left to the LLM |
| `stop` | p(done) ≥ tau (the fast tier's final reply would be accepted) | the strong tier also ended the turn | `stop: shadow` with routing on (every fast-tier final answer is redone by the strong tier) |

Requests that offered a single tool (mini-swe-agent's `bash`, plus "reply without a tool") are left out of the
`tool_select` fit and counted as `single_tool`: there agreement only says that the agent calls its one tool, and a
cascade fitted on them would move almost every step to the fast tier, whatever the step needs.

Routing, reflection and action decisions have no such label in a proxy log (nobody knows what the other tier, or the
agent without the pause, would have done). `jah fit` describes them, with how often each would fire at the logged tau,
and leaves their thresholds to you.

A labelled decision is one where the LLM chose without the decision model's influence: a forced or shortlisted step is
not a label (`fit_ok: false` in `decisions.jsonl`).

```bash
jah serve --preset shadow --log-dir runs/shadow          # run your harness on development tasks
jah fit runs/shadow                                      # table + config snippet
jah fit runs/shadow --model jah-auto --out fit.yaml      # one model name; write the snippet
```

```text
| point | labelled | fitted tau | acts on | agreement | lower bound | note |
|---|---|---|---|---|---|---|
| tool_select | 412 | 0.97 | 38.1% | 98.7% | 97.1% |  |
| stop | 23 | - | - | - | - | too few labelled decisions (23 < 30) |
```

(The format only: these numbers are made up.) The snippet holds only thresholds: add the mode yourself.

```yaml
decisions:
  tool_select: {tau: 0.97}    # 412 labelled: acts on 38.1%, 98.7% agree (lower bound 97.1%)
  # stop: too few labelled decisions (23 < 30): keep the LLM deciding (mode: strong)
```

Agreement with the LLM is a proxy for "does not lose success", not the paper's full rule, which also weighs the price
of a saved call against the cost of a wrong choice. Refit when the LLM, the decision model, the tools or the kind of
task change. A fit on tasks that are too easy (the LLM never errs, so it agrees with anything confident) overstates
what the decision model can take.

## Evaluating a mode on your tasks

1. **Collect.** Run your harness on development tasks with `--preset shadow`. This makes the same LLM calls as the
   default (plus the decision calls), so you can do it on runs you need anyway.
2. **Fit.** `jah fit` the log. If a point gets no threshold, stop there for that point.
3. **Compare arms side by side.** One proxy can serve several arms under different model names:

   ```yaml
   models:
     jah-strong: {tier: strong}                                    # the baseline
     jah-auto: {policy: route}                                     # the README setting
     jah-auto-tools: {policy: route, decisions: {routing: off, tool_select: {mode: cascade, tau: 0.97}}}
     jah-auto-safe: {policy: route, preset: safe}
   ```

   Run the same tasks with each name and compare success (from your harness), `jah report` (cost and LLM seconds per
   session, tier shares) and the decision table it prints (decision calls, how often each point acted).
4. **Look at what a point did.** `decisions.jsonl` has one line per decision call with what came of it.

Use a held-out split for the comparison, not the tasks you fitted on. Give each harness its own model names (a
fit pools every session of a name), and run a plain default arm next to the others: on a few dozen tasks the
run-to-run noise is larger than most of the effects (see below).

## Dev-split measurements

On 2026-10-15 we ran the modes on the **development** split of Terminal-Bench 2.0 (`bench/splits/tb2_dev.txt`, 15
tasks, one trial per task; the test split was not used) with mini-swe-agent (one tool: `bash`) and pi (`read`,
`bash`, `edit`, `write`). The setting was the README's "lite" pair (Gemini 4 Argon, high thinking / Gemini 3.5
Flash-Lite, low thinking, Jev on OpenRouter, routing tau 0.75) with one model name per arm in one proxy
(`bench/configs/gemini_argon_lite_modes.yaml`) and the arms of a phase running side by side. Every table, per-task
grid and fit is in [`bench/results/dev_modes.md`](../bench/results/dev_modes.md). These numbers chose the
recommendations below. They are not results.

**How noisy is it?** The default policy ran four times on mini-swe-agent (the shadow arm, the `safe` arm whose
guard never acted, and the plain default twice): 12, 12, 10 and 11 of 15 tasks, at $0.19 to $0.41 per task. Two
arms that made the same kind of calls differed 2.2x in cost because two tasks ran into the time limit in one of them.
On 15 tasks, a difference of two or three tasks or of 2x in cost per task is noise. What each point did (how often it
fired, and what its redo changed) is measured well.

| Point and mode | Harness | What it did | Verdict |
|---|---|---|---|
| `shadow` preset | both | The same LLM calls as the default, plus 64 (mini-swe-agent) and 84 (pi) decision calls per task: $0.0065 per task, 1.7–3.6% of the bill. 0 errors in 2,700 calls, 0.145 s each (p50), asked beside the LLM call. | Use it to collect data. |
| `tool_select` (`jah fit`, `cascade`, `paper`) | mini-swe-agent | One tool, so its 325 questions were "call `bash` or reply in text": not a tool choice. `jah fit` now leaves such requests out. | Not applicable. |
| | pi | Jev named pi's next tool 68% of the time (`bash` 84%, `read` 30%, `write` 28%). At p ≥ 0.99 it was sure on 7.9% of the steps (96% agreement, nearly all `bash`) and sure of another tool on 0.8%. `jah fit` found no threshold. Flash-Lite's 60 premature text replies never got a tool prediction at p ≥ 0.99, so the `paper` cascade would not have stopped them either. | Not run: the cascade would have nothing reliable to act on. |
| `stop: guard` (tau 0.9) | pi | Of 128 fast-tier final replies, the strong redo went on working in 109 (85%). Jev's p(done) hardly told the two apart (AUC 0.62; replies the strong tier confirmed scored 0.25–0.80, premature ones up to 0.89), so no threshold qualifies. Live, the guard accepted 2 of 70 replies (both on tasks that passed) and redid the other 68: the finish check plus a decision call. | Keep `stop: strong`. |
| `stop` with `submit` (finish check on the submit command) | mini-swe-agent | Without `submit` the stop point never fired (one text reply in 15 tasks). With it, Flash-Lite tried to submit 3.7 times per task (up to 12 times in one task) and the strong redo went on working in 44 of 55 cases. Cost per task +75% next to the default of the same phase ($0.430 vs $0.246: $0.06 for the redos, the rest from longer episodes), success 12 vs 10 of 15. Jev's p(done) on submissions: AUC 0.75, premature ones up to 0.88, no threshold. | Off by default. Turn it on only if the strong tier should vet every submission, and expect it to cost more. |
| `action: guard` (`safe`) | mini-swe-agent (guard), both (shadow) | The pre-filter matched 0.5 (mini-swe-agent) to 0.9 (pi) calls per task: `rm -f` of build and temporary files, `pkill` of the agent's own server, `DROP INDEX` in a task database. Jev called 26 of these 28 calls safe and 2 risky (0.74 and 0.50, both harmless inside the task container). Neither was in the `safe` arm, which therefore never re-checked a call. Cost: 0.5 decision calls and 0.1 s per task. | Inert here: Terminal-Bench tasks run in throwaway containers. Keep it as a cheap safety net (tau 0.5); it is not a saving. |
| `reflection: trigger` (tau 0.8) | both | In shadow it would fire on 17% of the steps, more often in failing tasks (AUC 0.65 on mini-swe-agent, 0.58 on pi). In shadow, 51 of 53 (mini-swe-agent) and 59 of 68 (pi) of the steps it flagged went to the strong tier under routing anyway. Live it fired on 17–23% of the steps, and only 4–6 firings per 15 tasks moved a step off the fast tier: the others added the re-check note to a strong step. Cost per task −6% and +7% next to the default of the same phase. Success: 13 of 15 in all three samples (mini-swe-agent twice, pi once), against 10 and 11 for the mini-swe-agent defaults of the same phases (6 tasks won, 1 lost) and 11 for pi's default of an earlier phase. | The only mode that kept the default's cost and was ahead each time. Not established on 15 tasks: worth a registered test row, not a default. |
| `routing: guard` (tau 0.5) | mini-swe-agent | Flash-Lite wrote every step after the first, and Jev asked for a strong redo on 35% of them (9.7 per task, 1.34 LLM calls per step). $0.390 per task vs $0.246 for the cascade of the same phase, success 13 vs 10 of 15. | No saving: the cascade is cheaper. |

**Recommended settings.** No preset and no threshold changed: nothing here clearly beat the default.

- **Default (`route`)**: keep it (routing cascade + finish check). No acting mode was clearly cheaper.
- **`shadow`**: use it to collect data on your own tasks. It costs about 2–4% of the bill and does not change the
  agent's calls.
- **`paper` / `tool_select: cascade`**: do not use it with coding agents that have one to four tools. On pi,
  `jah fit` found no threshold and Jev was rarely sure of a tool other than `bash`. It is meant for agents with many
  tools, where the tool choice itself is the decision.
- **`safe` (`action: guard`, tau 0.5)**: a cheap safety net (half a decision call per task). It never fired on
  Terminal-Bench, where every task runs in a throwaway container. Use it where actions can do lasting harm, next to
  the harness's own approval and sandbox.
- **`stop`**: keep `strong`. Jev could not tell a finished task from a premature final reply on pi or on
  mini-swe-agent's submissions, so the guard has no usable threshold. `submit` makes the finish check cover
  mini-swe-agent, at about 75% more cost per task on this split. Leave it off unless the strong tier should vet every
  submission.
- **`reflection: trigger`, tau 0.8**: the candidate to test next. It adds about 20 decision calls per task
  ($0.002) and no measurable LLM cost:

  ```yaml
  decisions:
    reflection: {mode: trigger, tau: 0.8}
  ```

- **`routing: guard`**: costs more than the cascade (+59% per task here). Prefer the cascade.

## Per model name

A model name may set its own `preset` and `decisions`, on top of the top-level ones:

```yaml
preset: shadow
decisions:
  action: {mode: guard, patterns: ['\bmake\s+deploy\b']}
models:
  jah-auto: {policy: route}
  jah-auto-t9: {policy: route, tau: 0.9, decisions: {tool_select: {tau: 0.98}}}   # mode from above, own tau
  jah-strong: {tier: strong}                                    # a pinned name: a plain baseline
  jah-strong-safe: {tier: strong, preset: safe}                 # a pinned name with its own decision points
```

Later layers win: the defaults (routing `cascade` at `router.tau`, `stop: strong` while `router.verify_finish`), the
top-level `preset`, the top-level `decisions`, the name's legacy keys (`tau`, `verify_finish`), the name's `preset`,
the name's `decisions`. A pinned name (`tier:`) is a baseline: the top-level preset and decisions do not apply to it,
only its own, and it has no routing point.

## Logs

`decisions.jsonl` gets one line per decision call (a knockout over many tools is one line):

| Field | Meaning |
|---|---|
| `ts`, `session`, `model_name`, `step` | as in `steps.jsonl` |
| `point`, `mode`, `question` | the decision point, its mode, the question id (`next_step`, `fast_step_ok`, `next_tool`, `task_done`, `already_done`, `needs_reflection`, `action_risk`) |
| `options`, `n_options`, `p` | the options (names only), their number, the ten likeliest probabilities |
| `choice`, `confidence` | the likeliest option and its probability |
| `act_option`, `p_act`, `tau` | the option compared with tau (yes/no questions), its probability, the threshold |
| `acted`, `effect` | whether the point changed the step, and how: `forced`, `shortlist`, `redo`, `recheck`, `accepted`, `logged`, `defer_<reason>`, `error_<fallback>`, ... |
| `label`, `label_tier`, `agree`, `fit_ok` | what the LLM itself chose here, which tier chose it, whether it equals `choice`, and whether `jah fit` may use it |
| `latency_s`, `cost_usd`, `model`, `rounds`, `error` | the decision call(s) |
| extras | `forced_tool` and `followed` (cascade), `kept_tools` (assist), `after` (tool guard: the redo's tool), `tool`, `pattern`, `recheck` (what the strong redo did with the checked reply: `same` calls up to formatting / `changed` / `no_call`; action and routing guard) and `recheck_risky` (the reply the harness gets still makes a pre-filter-matched call), `tier` (stop, routing guard) |

The step record in `steps.jsonl` gets `points` (a compact list of the step's decisions other than routing) and `redo`
(the checks that sent the step to the strong tier) when any point other than the default ones is active, and
`mark_reflection: always` or `mark_check_error` when they apply. Its `reason`
can now also be `tool_cascade`, `reflection`, `stop_trigger`, `stop_guard`, `tool_guard`, `route_guard`,
`action_guard`, `guard`, `shadow` or `routing_off`, and `decision_cost_usd` covers every decision call of the step.
Prompt text is never logged; with `log_states: true`, `states.jsonl` also gets each point's input state (with a
`point` field).

## Cost and latency

- A decision call costs about $0.0001 and 0.15 s with Jev on Terminal-Bench states (about 2,000 input tokens at
  $0.042 per million; measured on the dev split, see below). Pre-call questions run at the same time as each other
  (and as the routing question), and shadow and guard questions at tool choice run at the same time as the LLM call,
  so a step waits for the slowest pre-call question only.
- Post-call checks (stop guard, routing guard, action guard) wait for the reply, then run at the same time.
- The LLM-call side is what decides the bill: `cascade` and an accepted `stop: guard` replace a strong call by a fast
  one. Every redo or re-check adds a strong call. `assist` sends fewer tool schemas but may lose cache hits.
