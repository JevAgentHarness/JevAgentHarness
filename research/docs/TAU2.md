# tau2-bench integration (`jev_agent`)

[tau2-bench](https://github.com/sierra-research/tau2-bench) (Sierra; "τ²-bench") evaluates
customer-service agents in the **airline**, **retail** and **telecom** domains (plus `mock`, used by our
offline tests). An LLM **user simulator** holds a goal and talks to the agent. The agent calls domain tools
that read and write a DB, and must follow a written **domain policy**. At the end, tau2 grades the
simulation by replaying the tool calls and comparing the final DB with the reference (`DB`). It also checks
the required communicated info (`COMMUNICATE`) and, for most retail tasks, LLM-judged natural-language
assertions (`NL_ASSERTION`). The reward is the product of the gated components, and a task counts as
solved iff reward == 1. Reliability over repeated trials is reported as **pass^k**, the mean over tasks of
C(successes, k) / C(trials, k).

We do not wrap tau2 as a jevharness `Env`. Instead we plug *into* tau2's own orchestrator as a
registered agent, `jev_agent` (`src/jevharness/bench/suites/tau2.py`). After the run we convert tau2's
`results.json`, together with our side-car decision log, into jevharness `Trajectory` JSONL, which
`jevbench report` consumes.

```
tau2 orchestrator ── user simulator (LLM) ── domain env (DB + tools) ── evaluator (DB / communicate / NL)
        │
        └─ JevTau2Agent  (subclass of tau2's LLMAgent: same system prompt, same litellm `generate` call)
             one generate_next_message() call = one StepLog; System-1 decision points below
```

With every decision point `off` (`configs/tau2/llm_only.yaml`) the agent behaves exactly like tau2's stock
`llm_agent`, but every turn is logged in our format.

## Decision points

| WHERE (`dp/mode` in logs) | HOW | System-1 question | Effect |
|---|---|---|---|
| `tool_select/{replace,cascade}` | Choice over tau2's tools + `respond_to_user` | "What should the agent do next?" | If S1 is accepted (replace, or conf ≥ tau) and picks a tool, `tool_params` fills its **required** args. If every required arg is filled, the call is emitted with **no LLM call**. Otherwise the LLM is forced (`tool_choice`) to call that tool, with only that tool's schema and the already-filled args. If S1 picks `respond_to_user`, the LLM only writes the message (`tool_choice="none"`). Cascade below tau defers the turn to the LLM. |
| `tool_params/{replace,cascade}` | Choice over enum/bool values, or over spans copied from the conversation plus `<OTHER>` | "Which value should argument X take?" | Uses `jevharness.harness.tools` (`params_of`, `extract_candidates`, `OTHER`). Candidates are argument-aware (`_value_candidates`): values already recorded under the argument's JSON key (`"reservation_id": ...`, `"reservations": [...]`, earlier calls) come first, snake-case ids such as `raj_sanchez_7340` are added, number fragments of dates are dropped, and bare numbers go last for string arguments. Free-text arguments (`summary`, `thought`, `message`, ...) always go to the LLM. Optional args are omitted unless the LLM is called. |
| `tool_select/assist` | Choice, top-k | same | The LLM sees only the top-k tools. It can still reply in text. |
| `tool_select/guard` | Noul | "Is this the right next step?" | Asked about the LLM's proposal. A rejection means regenerate. |
| `policy/guard` (tau2-specific; `policy_guard:` key) | Noul, before any **WRITE** tool call (`extra.scope: all` checks every call) | "Does this action comply with the domain policy, and has the user explicitly confirmed exactly this action?" | The call is **not executed** and the LLM regenerates with a `[Policy check]` note. The airline and retail policies both require explicit confirmation before DB updates. Tool types come from tau2's `@is_tool(ToolType.WRITE)`. |
| `stop/guard` | Noul, before `transfer_to_human_agents` | "Is transferring justified by the policy?" | Same regenerate-on-reject. In tau2 the *user* ends the conversation, so `stop: trigger` is not supported. |
| `routing/trigger` | Noul per turn | "Does the next step need careful reasoning / policy interpretation / error recovery?" | p_yes ≥ tau sends the turn to the strong model (`--agent-llm`). Otherwise it goes to `small_llm`. |
| `tool_select/shadow`, `tool_params/shadow`, `stop/shadow`, `routing/shadow`, `policy/shadow` (`policy_guard: {mode: shadow}`) | **measurement only** (PREREGISTRATION A5.4) | exactly the live arm's question: the cascade `tool_select` Choice (same text, options and descriptions), the cascade `tool_params` Choices for the required arguments of the tool the LLM then called, the stop guard's Noul before a `transfer_to_human_agents` call, the routing Noul, the policy guard's Noul before a WRITE | **Nothing is acted on.** The LLM's own action is executed unchanged (full tool list, `tool_choice=auto`, the main model); S1's answer / confidence / distribution, the LLM's choice (`s2_value == final_value`), `s2_correct` and the approximate gold are logged side by side with `mode: shadow`, `shadow: true`, `deferred: true`, `s1_cost_usd` / `s1_latency_ms` (the shadow call) and `s2_cost_usd` / `s2_latency_ms` (the LLM call that made the decision). S1 calls go through a `Tap(shadow=True)` and are tagged `meta.shadow` / `meta.measure: shadow` in the Ledger; the side-car `usage` is the harness's `usage_summary`, so `usage.total` / `usage.s2` equal what `llm_only` costs and the overhead is `usage.shadow` (`usage.total_with_shadow` = actually spent). tau2's own `agent_cost` excludes shadow calls too. Fail-open: an S1 error never changes the episode; it is recorded in the side-car `shadow_s1_errors` (→ `meta.shadow_s1_errors`) and the decision is logged without an S1 answer. Semantics mirror `harness/agent.py` (`_shadow_ask`, `_shadow_params`, `_stop_check`). |

Guards follow the semantics of `harness/agent.py`: at most `guard_max_retries` checks per turn, shared by
all guards. The last regeneration is accepted. `memory_*`, `reflection` and `action` are not implemented
for tau2; a config that enables them is rejected.

The **S1 state** is built from a policy excerpt (`s1_policy_chars`, default 3000) plus as much of the
recent conversation as fits in `s1_state_max_chars`. Tool outputs are truncated to `obs_truncate_chars`.
Copyable-argument candidates are extracted from the full conversation, never from the policy.

**Gold labels are approximate.** tau2 has no per-turn oracle: it grades final state, and its reference
`actions` contain no "talk to the user" steps. For decision-level metrics and the `simulated`/`oracle` S1
backends (the agent never sees gold), `_Reference` in `tau2.py` derives labels as follows:

- **next step:** the first pending reference action whose argument values have already appeared in the
  conversation. A WRITE action also needs the last user message to be affirmative, when the policy
  requires confirmation. If no action qualifies, the label is `respond_to_user`. A pending **READ** action
  that comes before (in reference order) an action the agent has already executed is dropped as moot:
  the agent obtained that information another way. Example from the retail smoke: the reference
  authenticates with `find_user_id_by_name_zip`, the agent used `find_user_id_by_email`; without the rule
  the name+zip lookup stayed the gold label for every later turn, because the name and zip appear in later
  tool outputs (`test_l3_smoke.py::test_gold_drops_read_actions_the_agent_bypassed`).
- **policy guard:** the proposed call matches a pending reference action, using tau2's own
  `Action.compare_with_tool_call`.
- **transfer guard:** `transfer_to_human_agents` is a pending reference action.

Read `s1_acc` on tau2 as agreement with the reference trajectory, not as ground truth. Set
`oracle_gold: false` to disable gold labels.

## Setup

Pinned to tau2-bench commit `5bfa7e3` (`pyproject` version 1.0.1, `git describe`: `v1.0.1-47-g5bfa7e3`).

```bash
git clone https://github.com/sierra-research/tau2-bench ~/tau2-bench && git -C ~/tau2-bench checkout 5bfa7e3
. .venv/bin/activate
# tau2's runtime deps (from its pyproject) + websockets (imported by tau2.data_model even in text mode)
uv pip install rich "tabulate>=0.9.0" "fastapi>=0.115.11" "uvicorn>=0.34.0" "pandas>=2.2.3" "psutil>=7.0.0" \
  "loguru>=0.7.3" "docstring-parser>=0.16" "litellm>=1.80.15,<1.82.7" "tenacity>=9.0.0" "deepdiff>=8.4.2" \
  "addict>=2.4.0" "toml>=0.10.2" "typer>=0.12.5" "requests>=2.31.0" websockets
# tau2 declares Python >=3.12; our venv is 3.11 -> install the package itself with pip, bypassing the pin
uv pip install pip && python -m pip install --no-deps --ignore-requires-python -e ~/tau2-bench
python -m pytest tests/test_tau2_adapter.py -q          # offline, ~5 s
```

- **Do not** `pip install tau2` from PyPI. That name belongs to an unrelated package (magnetic relaxation
  rates, v2.4.0). Install tau2-bench from the git clone.
- **Python 3.11.** Text mode works on 3.11: the source compiles, and tau2's offline test suites for
  environment, tasks, results format, checkpointing and the mock/airline/retail domains pass (93 tests).
  Voice mode was not checked. For an officially supported setup, use a separate 3.12 venv:
  `uv venv -p 3.12 .venv312 && uv pip install -p .venv312 -e . -e ~/tau2-bench websockets`.
- **litellm.** tau2 pins `<1.82.7`, so this setup downgrades litellm (1.103.1 → 1.82.6 in our venv).
  mini-swe-agent's constraint (`>=1.75.5, !=1.82.7, !=1.82.8`) still holds.
- **Data.** An editable install reads `<clone>/data`. Otherwise set `TAU2_DATA_DIR`.
- **Keys** (`.env` / environment):
  - `OPENAI_API_KEY` for the agent, the user simulator, and the retail NL-assertion judge. The judge model
    is fixed by tau2 to `gpt-4.1-2025-04-14` (`tau2/config.py`). Without OpenAI access, pass
    `--jev-nl-judge <litellm model>` to the launcher (`NL_JUDGE_LLM` in `scripts/run_tau2.sh`,
    `nl_judge_llm` in an experiment YAML); see [Retail NL-assertion judge](#retail-nl-assertion-judge).
  - `TYPESAFE_API_KEY` (plus optional `TYPESAFE_BASE_URL`) for the `{type: jev}` S1 backend.
  - Any litellm model name works for the agent and user (e.g. `openai/...` with `OPENAI_API_KEY`).
  - Gemini (`gemini/gemini-3.5-flash`, ...) reads `GEMINI_API_KEY` through litellm: one key, no rotation.
    The Gemini stand-in S1 goes through our own client and rotates over `GEMINI_API_KEYS`. See
    [Gemini agents](#gemini-agents-validated-in-the-pilot).

## Running

Runs must go through our launcher, which registers `jev_agent` before tau2's CLI parses `--agent`. Every
argument not prefixed `--jev-` is passed straight to `tau2 run`:

```bash
python -m jevharness.bench.suites.tau2 run --jev-config configs/tau2/jev_guard.yaml \
    --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1-mini --num-trials 3 --max-concurrency 8 \
    --save-to $PWD/runs/tau2/raw/airline/jev_guard --auto-resume
python -m jevharness.bench.suites.tau2 convert runs/tau2/raw/airline/jev_guard/results.json \
    --arm jev_guard --out runs/tau2/airline/jev_guard.jsonl       # prints n, avg reward, pass^k, costs
jevbench report runs/tau2/airline
```

- `--jev-set KEY=VALUE` (repeatable) overrides the YAML, e.g. `tool_select.tau=0.9`, `small_llm=gpt-4.1-nano`
  or `'s1={type: simulated, accuracy: 0.85}'`. The last is a dry run that needs no Jev key.
- `--jev-name` sets the arm name.
- `--jev-nl-judge MODEL` (+ optional `--jev-nl-judge-args JSON`) replaces tau2's fixed gpt-4.1 NL-assertion
  judge; env `JEV_TAU2_NL_JUDGE_LLM` / `JEV_TAU2_NL_JUDGE_LLM_ARGS`.
- `python -m jevharness.bench.suites.tau2 split airline` prints the A5.4 shadow / test split (`--which shadow
  --format args` for `tau2 run --task-ids`); `report <out>/<domain> --split test|shadow` restricts a converted
  domain directory to one split (see [A5.4 procedure](#a54-procedure-c7-on-τ²-airline)).
- The side-car is written to the `--save-to` directory, or to `--jev-log-dir`.
- Pass an **absolute** `--save-to`. A relative one lands under `<tau2 clone>/data/simulations/`.
- **Do not use `--workers`**; the launcher refuses it. Worker processes are spawned as
  `python -m tau2.runner.worker` and would not have `jev_agent` registered. Use `--max-concurrency`
  (threads) instead.

**All arms:** `scripts/run_tau2.sh`. It runs `llm_only`, `jev_ts_cascade` with tau in
{0.5, 0.7, 0.8, 0.9, 0.95} (the same tau for `tool_select` and `tool_params`), `jev_ts_assist_k3`,
`jev_guard` and `jev_route` on airline and retail, then converts each arm and prints a report per domain
(airline: also the per-split summaries, `report --split test` and `--split shadow`). `ARMS=jev_shadow` runs the
A5.4 measurement arm (`configs/tau2/jev_shadow.yaml`). It is configured through environment variables:

| Variable | Default |
|---|---|
| `AGENT_LLM` | `gpt-4.1` |
| `AGENT_LLM_ARGS`, `USER_LLM_ARGS` | empty. JSON for `--agent-llm-args` / `--user-llm-args`; for Gemini pass `{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}` (see [Gemini agents](#gemini-agents-validated-in-the-pilot)) |
| `USER_LLM` | `gpt-4.1-mini` (fixed across arms) |
| `SMALL_LLM` | `gpt-4.1-mini` (for `jev_route`) |
| `NUM_TRIALS` | `3` |
| `NUM_TASKS` | empty (all tasks) |
| `TASK_SPLIT` | tau2's own split: `base`: airline 50 / retail 114; `test` is 20 / 40 (not the A5.4 split) |
| `SPLIT` | `all` (default), `shadow` or `test`: the A5.4 airline split, resolved to `tau2 run --task-ids` (15 / 35 ids from `configs/tau2/airline_task_ids.json`). Retail has no shadow split: `test` runs every retail task, `shadow` skips the domain. |
| `TASK_IDS` | explicit ids (`"0 1 2"`, `"0,1,2"` or a file: JSON list, `{"task_ids": [...]}`, or one id per line); overrides `SPLIT` |
| `DOMAINS`, `ARMS`, `TAUS` | as above |
| `MAX_CONCURRENCY` | `4` |
| `SEED` | `300`. Every arm gets the same trial seeds, so arms are paired by (task, trial). |
| `S1` | override the S1 backend for all Jev arms |
| `NL_JUDGE_LLM`, `NL_JUDGE_LLM_ARGS` | retail NL-assertion judge instead of tau2's fixed `gpt-4.1-2025-04-14` (same for every arm) |
| `OUT` | `runs/tau2` |
| `DRY_RUN=1` | print the commands only |

Re-running resumes the finished (task, trial) pairs, because the script passes `--auto-resume`.

Arm configs live in `configs/tau2/*.yaml`. They use the flat `HarnessConfig` fields plus these tau2 keys:

| Key | Meaning |
|---|---|
| `s1`, `s1_cache` | S1 backend spec and its SQLite cache |
| `small_llm`, `small_llm_args` | cheap model for routing |
| `policy_guard` | `{mode, tau, extra: {scope}}` |
| `log_dir` | side-car directory |
| `log_state` | `false` drops the S1 state strings (up to `s1_state_max_chars` each, one per decision), which dominate the side-car size |
| `s1_policy_chars` | policy prefix rendered into the S1 state |
| `respond_tool_choice` | `none` or `no_tools`. Use `no_tools` if a provider rejects `tool_choice="none"`. |
| `oracle_gold` | derive approximate gold labels (see above) |

**Python API** (e.g. in a notebook):

```python
from jevharness.bench.suites import tau2 as jt
from tau2.data_model.simulation import TextRunConfig
from tau2.runner import run_domain
jt.register("jev_guard", config="configs/tau2/jev_guard.yaml", log_dir="/abs/out")  # or config=Tau2JevConfig(...)
run_domain(TextRunConfig(domain="retail", agent="jev_guard", llm_agent="gpt-4.1", llm_user="gpt-4.1-mini",
                         num_trials=1, save_to="/abs/out"))
trajs = jt.convert_results("/abs/out/results.json", out="runs/tau2/retail/jev_guard.jsonl")
print(jt.summarize(trajs))
```

### A5.4 procedure: C7 on τ² airline

`docs/PREREGISTRATION.md` A5.4 (c)–(d). The 50 airline tasks are split **once**, before any τ² run, by
`airline_split` in `tau2.py`: the task ids in sorted order (numeric ids ascending, `0, 1, …, 49`) are shuffled
with `random.Random(0)`; the first 15 are the **shadow split** (only ever used to refit τ̂\*), the other 35 the
**airline test set**. The ids were fetched once from tau2-bench (`data/tau2/domains/airline/tasks.json` at
commit `5bfa7e3`, sha256 recorded) and committed with the resulting split in `configs/tau2/airline_task_ids.json`;
loading that record asserts it against `airline_split()`, and with tau2 installed the task list is cross-checked
against the dataset, so the split cannot drift silently. Retail has no shadow split: all 114 retail tasks are
test tasks.

```
shadow: 0 1 5 7 12 14 20 27 28 33 34 36 43 47 49
test:   2 3 4 6 8 9 10 11 13 15 16 17 18 19 21 22 23 24 25 26 29 30 31 32 35 37 38 39 40 41 42 44 45 46 48
```

(`python -m jevharness.bench.suites.tau2 split airline`; the test in `tests/test_tau2_split.py` pins these lists.)

1. **Shadow run on the 15** (Jev logged, the LLM decides; ≈ $30 in A5.6's estimate):
   ```bash
   DOMAINS=airline ARMS=jev_shadow SPLIT=shadow NUM_TRIALS=3 scripts/run_tau2.sh
   ```
   `configs/tau2/jev_shadow.yaml` = `tool_select: shadow`, `tool_params: shadow`, `stop: shadow`. Its episodes
   are `llm_only` episodes (same actions, same LLM calls, same `usage.total`); the side-car adds the
   `mode: shadow` records and `usage.shadow`.
2. **Refit τ̂\*** with the H1 fit the C1 arm uses (`bench/threshold_theory.py`; the shadow records carry `c`,
   `s1_ok`, `s2_ok`, the per-decision S1 and LLM costs that `decision_pairs` reads). Restrict the fit to the
   shadow tasks even if the arm was (re)run on more:
   ```bash
   python -m jevharness.bench.suites.tau2 report runs/tau2/airline --split shadow --no-report   # -> runs/tau2/airline_shadow/
   python -m jevharness.bench.threshold_theory runs/tau2/airline_shadow --arm jev_shadow --dp tool_select \
       --format json --out runs/tau2/airline_shadow/tt_tool_select.json      # same --V / --rule / --lam-per-hour as the E2 C1 fit
   python -m jevharness.bench.threshold_theory runs/tau2/airline_shadow --arm jev_shadow --dp tool_params \
       --format json --out runs/tau2/airline_shadow/tt_tool_params.json
   ```
   τ̂\* is `by_V[*].thresholds_all.<rule>` (the plug-in threshold over every shadow decision; `general` is the
   primary rule). The cascade arm uses one τ for `tool_select` and `tool_params` (the script's `TAUS` sets both),
   as the C1 arm does on E2. Record the invocation and the value in the run log.
3. **C7 and the E2-carried secondary row, every arm on all 50 airline tasks** (plus retail), so that arms share
   one task list:
   ```bash
   ARMS="llm_only jev_ts_cascade" TAUS="<tau-hat* refit> <tau-hat* E2>" scripts/run_tau2.sh
   ```
   `jev_ts_cascade_tau<τ̂* refit>` is C7's arm (the default / tie-break arm of A5.4 b is the C1 configuration,
   which is this cascade); `jev_ts_cascade_tau<τ̂* E2>` is the secondary row, Holm-adjusted within the τ² table.
4. **Fallback rows** (A5.4 d; only if C7 fails NI): tool ASSIST k = 3 and, if C7's configuration differs from
   it, the C1 arm, on airline only:
   ```bash
   DOMAINS=airline ARMS=jev_ts_assist_k3 scripts/run_tau2.sh
   ```
5. **Report on the 35.** Every converted trajectory carries `meta.tau2.split` (`shadow` / `test`; `null` outside
   airline), `convert` prints `by_split` next to the overall summary, and the script runs for airline:
   ```bash
   python -m jevharness.bench.suites.tau2 report runs/tau2/airline --split test     # -> runs/tau2/airline_test/<arm>.jsonl + `jevbench report`
   python -m jevharness.bench.suites.tau2 report runs/tau2/airline --split shadow   # the 15, descriptive only
   ```
   `runs/tau2/airline_test/` (35 tasks × 3 trials per arm) is the directory every C7 / fallback comparison and
   `paper_numbers` read; `runs/tau2/airline_shadow/` is reported descriptively. `--test-only` is an alias for
   `--split test`; `--split test` keeps every task of a domain without a shadow split (retail).
6. **Proxy check** (A5.4 e): Kendall τ between the E2 and τ²-airline (test-set) rankings of the configurations
   that ran on both; the τ² side comes from `runs/tau2/airline_test/`.

What is verified offline (`tests/test_tau2_adapter.py`, `tests/test_tau2_split.py`): the shadow arm executes
exactly the LLM-only actions, asks the live arms' questions, books S1 under `usage.shadow`, is fail-open,
feeds `threshold_theory` (τ̂\* comes out of the mock-domain side-car), the split is pinned and reproduced from
the committed file, and the script dispatches `SPLIT` / `TASK_IDS` to `--task-ids`. What still needs the live
run: the real Jev backend through the shadow `Tap` at scale (rate limits, the tournament over airline's 13 tools
+ `respond_to_user`), the τ̂\* value itself, and tau2's `--task-ids` filter against the real airline data
(`get_tasks` raises on an unknown id, so a mismatch fails fast).

### Gemini agents (validated in the pilot)

`configs/pilot/tau2_airline_gemini.yaml` runs every arm of a multi-arm experiment with one command
(`python -m jevharness.bench.suites.tau2 experiment <yaml>`, which calls `run` once per arm, converts each
arm, and prints `jevbench report`). The pilot (5 airline tasks × 3 arms, Gemini 3.5 Flash agent, Flash-Lite
user simulator; results in [`PILOT_RESULTS.md`](PILOT_RESULTS.md)) established:

- **Thought signatures survive tau2's litellm path.** Gemini 3.x requires the `thoughtSignature` of every
  function call to be sent back on the next request. litellm 1.82.6 embeds the signature in the tool-call
  id (`call_<id>__thought__<signature>`), which tau2 stores and replays unchanged. For calls that System-1
  emits without an LLM (`call_jev_*`), litellm injects Google's documented dummy signature
  (`skip_thought_signature_validator`) for `gemini-3*` models. No agent LLM call had to be rerouted through
  `OpenAICompatClient`, and `llm_only` uses the same path. Agent errors: 0.
- **Thinking level.** litellm 1.82.x maps `reasoning_effort: minimal` to `thinkingLevel: low` for
  `gemini-3.5-*`. Pass `{"thinkingConfig": {"thinkingLevel": "minimal"}}` in `--agent-llm-args` /
  `--user-llm-args` instead; the launcher warns about the former.
- **`tool_choice="none"` is not honoured.** When S1 decides `respond_to_user`, Gemini 3.5 Flash still
  returned a function call in 19 of 38 such turns (pre-fix cascade run; 13 overrides after the fix). `_respond` keeps the text if there is any. A bare
  tool call runs as the LLM's own choice and is logged as a deferral (`respond_overrides` in the
  side-car), so replace/cascade arms on Gemini delegate less than they appear to.
- **Implicit prompt caching.** Gemini caches the shared prefix implicitly. In the pilot, 47% of the
  `llm_only` input tokens were cache hits, but only 23% in the cascade arm. That arm forces
  `tool_choice` and single-tool schemas, which change the prefix. See the accounting note below.

### Retail NL-assertion judge

112 of the 114 retail tasks list `NL_ASSERTION` in their reward basis. tau2 grades those assertions with
an LLM judge hard-coded to `DEFAULT_LLM_NL_ASSERTIONS = "gpt-4.1-2025-04-14"`. A task that has at
least one assertion (for example base-split tasks 2-4; tasks 0-1 have none) needs that model, so without
OpenAI access the evaluation fails. `--jev-nl-judge` patches tau2's evaluator module in the launcher
process (`set_nl_judge` in `tau2.py`). The default args are temperature 0, `response_format: json_object`
(tau2 `json.loads` the raw judge text, and Gemini otherwise tends to wrap it in a Markdown fence), and
minimal thinking on Gemini 3.x. Every side-car record stores the judge in effect (`nl_judge`).

- A non-default judge is a **deviation from the official tau2 protocol**. Keep it fixed across arms and
  report it.
- The judge's cost is environment cost, like the user simulator's, and is not in any arm's cost. It is
  one call per simulation that has assertions, about $0.002 with Gemini 3.5 Flash.

### Retail smoke (L3, 2026-10-01)

Three retail tasks (base split, ids 0-2) × 1 trial, arms `llm_only` and `jev_guard` (policy guard +
transfer guard, τ = 0.5). Agent: Gemini 3.5 Flash. S1: the Gemini Flash-Lite stand-in. NL judge:
Gemini 3.5 Flash. Concurrency 3. Outputs are in `runs/dev/l3_smoke/tau2{,_userflash}/`.

```bash
AGENT_LLM=gemini/gemini-3.5-flash USER_LLM=gemini/gemini-3.5-flash SMALL_LLM=gemini/gemini-3.5-flash-lite \
AGENT_LLM_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}' \
USER_LLM_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}' \
NL_JUDGE_LLM=gemini/gemini-3.5-flash \
S1='{type: llm_generative, llm: {provider: gemini, model: gemini-3.5-flash-lite}, reasoning_effort: minimal}' \
DOMAINS=retail ARMS="llm_only jev_guard" NUM_TASKS=3 NUM_TRIALS=1 MAX_CONCURRENCY=3 \
OUT=runs/dev/l3_smoke/tau2_userflash bash scripts/run_tau2.sh
```

| user simulator | arm | solved | agent $/sim (Ledger) | S1 $/sim | user $/sim | sim duration s | errors |
|---|---|---|---|---|---|---|---|
| Flash-Lite | llm_only | 1/3 | 0.074 | – | 0.0014 | 11–35 | 0 |
| Flash-Lite | jev_guard | 3/3 | 0.081 | 0.0013 | 0.0014 | 16–42 | 0 |
| **Flash** | llm_only | 3/3 | 0.076 | – | 0.0071 | 17–40 | 0 |
| **Flash** | jev_guard | 3/3 | 0.082 | 0.0012 | 0.0082 | 19–43 | 0 |

Wall clock: about 50 s per arm for 3 simulations at concurrency 3 (106 s for both arms, including
conversion and the report).

**Failure modes found**

- **The Flash-Lite user simulator ends conversations too early.** Both `llm_only` failures with
  Flash-Lite were the simulator writing "Yes, please go ahead and process it! ###STOP###" in one
  message. The episode ended before the agent could execute the confirmed exchange (`DB: 0`, user_stop).
  The success difference between arms was therefore simulator noise, not System-1. With Gemini 3.5 Flash
  as the user simulator, all 6 simulations succeeded, at about $0.007 more per simulation. **Use
  `USER_LLM=gemini/gemini-3.5-flash` (or tau2's default gpt-4.1) for retail.** The airline pilot used
  Flash-Lite; check its failures for the same pattern before reusing it.
- The policy guard saw one WRITE per simulation and approved all six with p_yes ≥ 0.99, which matches
  the reference (gold `True` each time). It never fired.
- The gold labels had the stale-READ problem described above; fixed.

**Full scale** (E5, retail + airline, 3 trials, all arms; about $0.08 per simulation per arm on Gemini,
so about $40 per arm for 492 simulations):

```bash
AGENT_LLM=gemini/gemini-3.5-flash USER_LLM=gemini/gemini-3.5-flash SMALL_LLM=gemini/gemini-3.5-flash-lite \
AGENT_LLM_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}' \
USER_LLM_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}' \
NL_JUDGE_LLM=gemini/gemini-3.5-flash S1='{type: jev, provider: openrouter}' MAX_CONCURRENCY=8 \
bash scripts/run_tau2.sh                    # drop NL_JUDGE_LLM when gpt-4.1 is reachable (official judge)
```

## Outputs and accounting

```
runs/tau2/raw/<domain>/<arm>/results.json        tau2's native results (browse with `tau2 view --file ...`)
runs/tau2/raw/<domain>/<arm>/jev_decisions.jsonl  side-car: one record per simulation (see below)
runs/tau2/<domain>/<arm>.jsonl                    jevharness Trajectories -> `jevbench report runs/tau2/<domain>`
```

**Side-car records.** Each record holds the episode id, `task_id`, `seed`, config and backend, every
`StepLog` with its `DecisionLog`s (including the S1 state, for `training/harvest.py`), the per-episode
`Ledger` summary (`usage`; the harness's `usage_summary`: shadow calls stay in `calls` but out of `usage.total`
/ `usage.s1`, itemised as `usage.shadow`) and calls, `shadow_points` (decision points run in shadow mode),
`shadow_s1_errors` (failed fail-open shadow calls) and any agent error.

**Joining.** Every assistant message tau2 stores carries `raw_data["jev"]`: the episode id plus a compact
per-turn summary (decisions, S1 calls and cost, LLM calls). The converter joins side-car records by that
episode id, which stays exact across tau2's retries. It falls back to `(task_id, seed)`, and then to the
tau2 messages alone. The fallback also covers results from tau2's stock `llm_agent`.

**Trajectories.**

| Field | Value |
|---|---|
| `task_id` | `<domain>/<task>#<trial>` |
| `result.success` | `reward == 1` (tau2's `is_successful`) |
| `result.score` | the reward |
| `result` (also) | `termination_reason`, `db_match`, `reward_breakdown` |
| `error` | set for `agent_error`, `user_error` and `infrastructure_error` |
| `meta.tau2` | trial, seed, simulation id, tau2's `agent_cost` / `user_cost`, and `split` (`shadow` / `test` for airline, A5.4; `null` elsewhere) |
| `meta.shadow_s1_errors` | shadow arms: the failed (fail-open) S1 measurements |

**Costs.** Two accountings are kept side by side:

- The **Ledger** (`usage`, `calls`; used by `jevbench report`) prices LLM calls from
  `configs/prices/*.yaml`, counting cached input tokens. It falls back to litellm's price when a model is
  not in the table. S1 calls are priced by their backend. The user-simulator and NL-judge costs are
  **excluded**; they are environment cost, reported as `meta.tau2.user_cost`.
- tau2's own **`agent_cost`** per message is litellm's cost for all LLM calls of the turn, including
  discarded guard regenerations, plus the S1 cost. tau2's summary table stays meaningful, and the numbers
  are comparable to stock `llm_agent` runs.

**Metrics.** The `convert` command prints avg reward, pass^1..pass^NUM_TRIALS, cost per simulation, S1 cost
and LLM calls per simulation (plus `shadow_cost_usd_per_sim` and, for airline, the same block per A5.4 split
under `by_split`). `jevbench report` adds success with CIs, the cost/success Pareto front, and decision-level
tables (S1 accuracy vs. the approximate gold, deferral rate, ECE, AUROC). `python -m jevharness.bench.suites.tau2
report <dir> --split test|shadow` writes the split's trajectories to `<dir>_<split>/` and runs `jevbench report`
on them.

## Cost estimate

These are rough figures; run a pilot first (`NUM_TASKS=5 NUM_TRIALS=1`) and extrapolate from the
`cost_usd_per_sim` and `user_cost_per_sim` values that `convert` prints. Prices come from
`configs/prices/models.yaml` (marked "verify").

**Assumptions.**

- **Agent prompt.** The fixed prefix per agent call (system prompt with policy, plus tool schemas) is about
  5.2k tokens for airline and 5.3k for retail. With the growing history, prompts average about 9–10k
  tokens.
- **Calls per simulation.** About 15 agent calls for airline and about 20 for retail (retail tasks have
  more reference actions: 4.8 vs 2.8).
- **Prompt caching.** OpenAI caches the conversation prefix; we assume about 60% of input tokens are
  cache hits.

**Per-simulation cost.**

| Component | Airline | Retail |
|---|---|---|
| Agent, gpt-4.1 | ≈ $0.16 | ≈ $0.24 |
| Agent, gpt-4.1-mini | ≈ $0.03 | ≈ $0.05 |
| Agent, gpt-4.1, no prompt caching | ≈ $0.28 | ≈ $0.42 |
| User simulator (gpt-4.1-mini) | ≈ $0.01 | ≈ $0.01 |
| NL-assertion judge (gpt-4.1, one call) | none | ≈ $0.025 (112/114 tasks) |
| Jev S1 ($0.042/M input; ≤12k-char state ≈ 3k tokens, 1–3 questions per turn) | ≤ $0.005 | ≤ $0.005 |

The Jev S1 cost is under 3% of the agent cost.

**Per arm on the `base` split with 3 trials** (150 airline + 342 retail = 492 simulations):

| Agent model | Per arm | Default grid (9 arms) | 1 trial |
|---|---|---|---|
| gpt-4.1 | ≈ $120 | ≈ $1.1k | ≈ $360 |
| gpt-4.1-mini | ≈ $35 | ≈ $310 | ≈ $105 |

Jev arms that replace LLM calls, and `jev_route`, cost less than `llm_only`.

**Cheaper configurations.**

| Configuration | Scope | Cost |
|---|---|---|
| `TASK_SPLIT=test` | 60 tasks | about 37% of the `base` cost |
| `TAUS="0.8 0.9"` | 6 arms instead of 9 | proportionally less |
| Pilot | 5 tasks × 1 trial × 2 domains × 9 arms | ≈ $20 with gpt-4.1 |

**Wall clock.** A simulation typically takes 1–3 minutes. At `MAX_CONCURRENCY=4` an arm on `base` × 3
trials takes about 2–4 h. Raise the concurrency as far as your rate limits allow.

## Tests

`tests/test_tau2_adapter.py` runs fully offline and is skipped if tau2 is not importable.

**What runs for real:** tau2's orchestrator, `UserSimulator`, mock domain and evaluator.

**What is stubbed:** every LLM call. The stub replaces `tau2.utils.llm_utils.completion` (the litellm entry
point) with a scripted LLM, so tau2's real `generate()`, message conversion and litellm cost path are
exercised.

**S1 backends:** `oracle`, `simulated` and a constant backend.

**Coverage:**

- config loading, overrides and validation; every shipped config must load
- the airline gold heuristic and the JSON-aware candidate filter
- llm_only equivalence and logging
- replace mode emitting a tool call with no LLM call
- cascade deferral and assist pruning
- the policy guard blocking a non-compliant write
- the transfer guard
- routing to the small model
- shadow mode (A5.4): the executed actions, LLM calls and `agent_cost` equal `llm_only`'s; the logged questions
  equal the cascade / replace / guard arms'; every S1 call is tagged shadow and kept out of `usage.total`;
  fail-open on an S1 error; the stop shadow lets the transfer through; the shipped `jev_shadow` arm end to end
  through the launcher, the converter and `threshold_theory` (τ̂\* from the side-car)
- `tests/test_tau2_split.py` (no tau2 needed): the pinned 15/35 airline split and its committed record, the
  `split` CLI, `filter_split` / `summarize_by_split` / `report --split`, and `run_tau2.sh` `DRY_RUN` dispatch of
  `SPLIT` / `TASK_IDS`
- tau2's batch runner (`run_domain`) with the registered agent → converter → `jevbench report` →
  `training/harvest.py`
- conversion of stock `llm_agent` results
- the CLI launcher

## Known limitations

- Gold labels are heuristic (see above).
- `respond_tool_choice: none` relies on the provider honouring `tool_choice="none"`, which litellm
  translates. Switch to `no_tools` if a provider rejects it. Gemini 3.5 Flash accepts it but ignores it
  about half of the time (see [Gemini agents](#gemini-agents-validated-in-the-pilot)).
- The Ledger bills cached input at the full input price when `configs/prices/models.yaml` has no
  `cached_input_per_m` for the model, which is currently the case for Gemini 3.x. litellm's cost, which is
  tau2's `agent_cost`, applies a cache discount. In the airline pilot the Ledger's `llm_only` cost was about
  1.7× litellm's, and the gap depends on the arm's cache-hit rate. Report both until a verified Gemini
  cached-input price is in the table.
- With `log_state: true` (the default) every decision stores its S1 state (up to 12k chars), so a long
  airline or retail simulation can add a few hundred KB to the side-car. Set `log_state: false` for
  arms you will not harvest.
- Temperature-0 LLMs are still nondeterministic, so compare arms with pass^k over ≥ 3 trials or with
  paired bootstraps over (task, trial).
- Telecom is untested. Its user simulator itself calls tools, but the adapter makes no assumption that
  prevents running `--domain telecom`.
