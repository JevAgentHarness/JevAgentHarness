# Jev hooks for OpenAI Codex CLI

This adapter puts a System-One decision model (hosted TypeSafe Jev, OpenJev, or any Jev-compatible
server) behind Codex's command hooks, using `python -m jevharness.adapters.hooks --harness codex`.
The implementation is in `src/jevharness/adapters/hooks.py`. The same script also serves DeepSeek
Harness (`adapters/deepseek/`).

All claims about Codex below were checked against `openai/codex` at commit `60947e2` (2026-09-30).
Paths are relative to `codex-rs/`.

## What each hook does

| Codex event | Decision point / default mode | S1 question (one batched Jev call) | What Codex receives |
|---|---|---|---|
| `PermissionRequest` | `tool_select` | Choice `{allow, ask, block}`, with option descriptions that follow Guardian's risk taxonomy, plus Noul "destructive/irreversible?" | `{"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"allow"}}}` or `behavior:"deny"` with a `message`. Nothing at all means Codex falls through to Guardian or the user. |
| `PreToolUse` | `tool_select` | same | Only `permissionDecision:"deny"` with a reason. Allow and ask produce no output. |
| `Stop` | `stop` / guard | Noul "task complete and verified?" | `{"decision":"block","reason":"[jev] … run the relevant tests …"}` when p < τ. A loop guard stops blocking after `JEV_STOP_MAX_BLOCKS` consecutive blocks. |
| `PostToolUse` | `memory_write` / trigger | Noul "worth remembering?" | A note is appended to `<state>/<session>/memory.jsonl`. With `JEV_MEMORY_CONTEXT=1`, `additionalContext` is also returned. |
| `UserPromptSubmit` | `routing` / trigger | Noul "needs strong model?" plus Score complexity (1–5) | The routing decision is logged. With `JEV_ROUTING_CONTEXT=1`, `additionalContext` is also returned. |

A verdict comes from the answer distribution `p` as follows:

- **block** if `p(block) ≥ JEV_TAU_BLOCK` (default 0.5).
- **allow** if `p(allow) ≥ τ` (default 0.8). If `p(destructive) ≥ JEV_TAU_DESTRUCTIVE` (0.8), the result is downgraded to **ask**.
- **ask** otherwise.

`JEV_MODE` then maps the verdict to what the harness sees:

| mode | allow | ask | block |
|---|---|---|---|
| `log` (shadow) | – | – | – |
| `guard` | – | ask | deny |
| `cascade` | allow | – (defer to Guardian/user) | deny |
| `replace` | allow | ask (`JEV_ASK_AS` can turn this into deny, allow or abstain) | deny |
| `off` | the S1 model is not called; the event is only logged | | |

## Codex protocol facts this relies on

- **Input schemas.** Inputs are defined in `hooks/src/schema.rs`: PreToolUse at :278-296, PermissionRequest at :300-318 (no `tool_use_id`), PostToolUse at :322-342, UserPromptSubmit at :567-583 and Stop at :587-601 (`stop_hook_active`, `last_assistant_message`).
  - Every turn-scoped input carries the Codex-only `turn_id`, which is how `--harness auto` recognises Codex.
  - Copies of the generated schemas are in `schema/`, taken from `hooks/schema/generated/` under Apache-2.0. `tests/test_hooks.py` validates its fixtures and every output against them.
- **Outputs reject unknown keys.** Every output struct is `#[serde(deny_unknown_fields)]` (schema.rs:87-99, 127-140). An unknown key makes the whole output invalid, so the adapter only emits the minimal documented shape.
- **PreToolUse only supports deny.** `engine/output_parser.rs:441-482` rejects `permissionDecision:"allow"` without `updatedInput`, and rejects `"ask"`. `deny` needs a non-empty `permissionDecisionReason`. The legacy `decision:"approve"` is rejected too (:484-507). For these reasons the adapter never emits allow or ask on Codex PreToolUse.
- **PermissionRequest runs first.** The approval order is: hooks, then Guardian if `approvals_reviewer = "auto_review"`, otherwise the user (`core/src/tools/approvals.rs:505-525`).
  - Any deny wins, otherwise the last allow wins, otherwise there is no verdict (`hooks/src/events/permission_request.rs:130-169`).
  - `updatedInput`, `updatedPermissions` and `interrupt:true` are reserved and make the hook fail (`output_parser.rs:401-414`).
- **Headless default.** `codex exec` sets `approval_policy = never` unless the resolved reviewer is `auto_review` (`exec/src/lib.rs:571-577, 758-790`). Without Guardian enabled, PermissionRequest therefore never fires in `codex exec`.
- **Tool names and permission mode.**
  - Shell calls reach hooks as `tool_name:"Bash"` with `tool_input:{"command": …}`.
  - `apply_patch` also matches `Write`/`Edit` matchers (`core/src/tools/hook_names.rs:34-39`).
  - `permission_mode` is `bypassPermissions` under `approval_policy=never`, otherwise `default` (`core/src/hook_runtime.rs:1032-1040`).
- **Stop.** A Stop `block` reason becomes the continuation prompt (`hooks/src/events/stop.rs:312-324`). Codex has no consecutive-block cap, so the adapter keeps its own counter.
- **Where hooks come from.**
  - `hooks.json` is read from each config layer's folder: `$CODEX_HOME/hooks.json` (user) or `<repo>/.codex/hooks.json` (project) (`hooks/src/engine/discovery.rs:339-380`, `config/src/state.rs:219-243`).
  - The inline form is `[[hooks.<Event>]]` in `config.toml` (`config/src/hook_config.rs:10-60, 153-201`).
  - Matchers: an omitted matcher, `""` or `*` matches everything. `A|B` is an exact list, and anything else is a regex (`hooks/src/engine/matcher.rs:11-23`).
  - The default timeout is 600 s (`discovery.rs:742-765`). We set 5 s, which leaves room for Python startup plus `JEV_TIMEOUT=2`.
- **Trust.** User and project hooks run only if their hash is trusted (via `/hooks` in the TUI) or with `--dangerously-bypass-hook-trust` (`discovery.rs:715-721`; `utils/cli/src/shared_options.rs:61-64`).
- **Environment.** Hook processes inherit the Codex session environment except a few Codex-internal tokens (`hooks/src/engine/command_runner.rs:424-434`; `protocol/src/shell_environment.rs:13-20`). This means `JEV_*` and `TYPESAFE_API_KEY` pass through.

## Install

1. Point the placeholder paths at your checkout:
   ```bash
   cd /path/to/JevAgentHarness/research && . .venv/bin/activate && pip install -e .
   sed "s#/ABS/PATH/JevAgentHarness/research#$PWD#g" adapters/codex/hooks.json > ~/.codex/hooks.json
   ```
   For isolated runs, set `CODEX_HOME=$PWD/runs/codex-home` instead of `~/.codex`.
2. Point the adapter at a backend:
   ```bash
   export TYPESAFE_API_KEY=...
   export JEV_S1='{"type":"jev","model":"jev-1.13.0"}'
   ```
   You can use `{"type":"openjev","base_url":"http://gpu:8080"}` instead. For a dry run, use `{"type":"simulated","accuracy":0.9}`.
3. Check the install without Codex:
   ```bash
   echo '{"session_id":"s","turn_id":"1","transcript_path":null,"cwd":"/tmp","hook_event_name":"PreToolUse","model":"m","permission_mode":"default","tool_name":"Bash","tool_input":{"command":"rm -rf ~"},"tool_use_id":"c"}' \
     | JEV_MODE=replace python -m jevharness.adapters.hooks --harness codex; tail -1 ~/.local/state/jevharness/hooks/decisions.jsonl
   ```

## Configuration

Every environment variable has a matching CLI flag, and the flag wins.

| env | default | meaning |
|---|---|---|
| `JEV_S1` | `{"type":"jev"}` | backend spec for `jevharness.decision.make_backend`: JSON, a path or `@path` to a JSON/YAML file, or a bare type name |
| `JEV_MODE` | `guard` | `off\|log\|guard\|cascade\|replace\|trigger`. Per-decision-point or per-event overrides: `"log,PermissionRequest=cascade,stop=guard"` (the most specific key wins) |
| `JEV_TAU` | tool_select 0.8, others 0.5 | threshold; per-decision-point form `"0.8,stop=0.4"` |
| `JEV_TAU_BLOCK`, `JEV_TAU_DESTRUCTIVE` | 0.5, 0.8 | block threshold; allow→ask downgrade threshold. `JEV_DESTRUCTIVE=0` drops the Noul |
| `JEV_ASK_AS` | `native` | what to emit for ask: `native`, `abstain`, `deny` or `allow` |
| `JEV_TIMEOUT` | `2` | seconds. Jev gets one attempt (`max_retries=0`), and a watchdog thread fails open |
| `JEV_LOG` | `$XDG_STATE_HOME/jevharness/hooks/decisions.jsonl` | decision log (JSONL) |
| `JEV_STATE_DIR` | `~/.local/state/jevharness/hooks` | per-session prompts, events, Stop counter and memory notes |
| `JEV_TRANSCRIPT_TAIL` | `4000` | characters of transcript tail in the S1 state. The Codex rollout at `transcript_path` is parsed directly |
| `JEV_STOP_MAX_BLOCKS` | `2` | Stop loop guard. The counter resets on each new user prompt and on each allowed stop |
| `JEV_MEMORY_CONTEXT`, `JEV_ROUTING_CONTEXT` | off | return `additionalContext` at PostToolUse / UserPromptSubmit |
| `JEV_RUN_TAG` | – | copied into every record, for example `"$ARM/$TASK"`, for joining with the harness output |
| `JEV_LOG_STATE` | off | also log the full S1 state, for replay or distillation |
| `JEV_CACHE` | – | SQLite decision cache, for deterministic reruns |
| `JEV_ENV_FILE` | – | dotenv file loaded without overriding the environment. It is needed for dsh (see `adapters/deepseek/`) |

Each log record has these fields: `{ts, harness, session, event, dp, mode, question, options, s1_value, s1_conf, s1_dist, final_value, latency_ms, input_tokens, model, error}`. They are supplemented by:

- `backend`, `backend_latency_ms`, `cost_usd`, `hook_ms`, `tau`, `state_chars`
- `turn_id` and `tool_use_id`
- `tool_name` and `tool_input` (an excerpt)
- `action`: what the adapter told the harness
- `emitted`: what the harness actually got after dialect limits
- `deferred`, and the per-question probabilities (`p_allow`, `p_block`, `p_destructive`, `p_complete`, `p_memory`, `p_strong`)
- `fail_open`, `loop_guard`

`final_value` is the S1-derived verdict:

- `allow`, `ask` or `block` for tool decisions
- `stop` or `continue` for Stop
- `write` or `skip` for memory
- `large` or `small` for routing

**Fail-open behaviour.** A backend error, timeout, bad stdin or bad flag always gives exit 0 with empty stdout. Codex then acts as if the hook had not run: the tool proceeds, or the approval falls through to Guardian or the user. The error is logged with `fail_open: true`. The script never exits with status 2, which Codex would read as a block.

## Experimental arms

Each arm is `codex exec` on the same task set. Arms A–C are the baselines. For arms A–C, use a
`CODEX_HOME` *without* `hooks.json` when you compare latency: every hook costs about 0.2 s of Python
startup. Alternatively, keep the hooks with `JEV_MODE=log` to collect shadow S1 verdicts.

| arm | `codex exec` flags | `JEV_MODE` | who approves escalations |
|---|---|---|---|
| A. never-ask (exec default) | `--sandbox workspace-write` | – / `log` | nobody: `approval_policy` is forced to `never` and escalations fail |
| B. yolo | `--dangerously-bypass-approvals-and-sandbox` | – / `log` | nobody, and there is no sandbox |
| C. Guardian baseline | `--approve-for-me` (sets `approvals_reviewer="auto_review"`, `approval_policy="on-request"`, `sandbox_mode="workspace-write"`) | `log` | the Guardian LLM (`prompts/templates/guardian/policy_template.md`); Jev shadow-logs its verdict on the same requests |
| D. Jev → Guardian cascade | `--approve-for-me` | `off,PermissionRequest=cascade` | Jev when confident, otherwise Guardian |
| E. Jev replaces Guardian | `--approve-for-me` | `off,PermissionRequest=replace` with `JEV_ASK_AS=deny` or `allow` | Jev only. Guardian runs only on Jev errors (count `fail_open`) |
| F. yolo + Jev action guard | `--dangerously-bypass-approvals-and-sandbox` | `off,PreToolUse=guard` | Jev can deny any tool call, and is the only safety layer |
| +S. Stop guard (combine with any arm) | – | add `,stop=guard` | Jev blocks premature stops |

Run every arm with `--dangerously-bypass-hook-trust`, or trust the hooks once via `/hooks`.

### Running a task set with `codex exec --json`

`tasks.jsonl` holds one task per line: `{"id": "...", "repo": "/abs/checkout", "prompt": "..."}`. Reset each checkout between arms.

```bash
ARM=D_cascade
export JEV_S1='{"type":"jev","model":"jev-1.13.0"}' JEV_MODE='off,PermissionRequest=cascade,stop=guard'
export JEV_LOG=$PWD/runs/$ARM/decisions.jsonl JEV_STATE_DIR=$PWD/runs/$ARM/state
mkdir -p runs/$ARM
while IFS= read -r t; do
  id=$(jq -r .id <<<"$t")
  start=$(date +%s.%N)
  JEV_RUN_TAG="$ARM/$id" codex exec --json --skip-git-repo-check --dangerously-bypass-hook-trust \
      --approve-for-me -C "$(jq -r .repo <<<"$t")" -o "runs/$ARM/$id.last.txt" \
      -- "$(jq -r .prompt <<<"$t")" > "runs/$ARM/$id.events.jsonl" 2> "runs/$ARM/$id.stderr"
  echo "{\"id\":\"$id\",\"exit\":$?,\"wall_s\":$(echo "$(date +%s.%N) - $start" | bc)}" >> runs/$ARM/runs.jsonl
done < tasks.jsonl
```

`codex exec --json` writes one `ThreadEvent` per line (`exec/src/exec_events.rs:11-38, 107-135, 151-157`):

- `thread.started` with a `thread_id`
- `turn.completed` with `usage`: `input_tokens`, `cached_input_tokens`, `output_tokens` and `reasoning_output_tokens`
- `item.completed` whose item `type` is `command_execution` (`status` is `completed`, `failed` or **`declined`**), `file_change`, `mcp_tool_call` or `agent_message`

Compare the arms on:

- **Success.** Run each task's own tests on the resulting checkout.
- **Escalation outcomes.** Count `command_execution` items with `status=="declined"`. These are blocked by the hook, Guardian or the policy:
  ```bash
  jq -c 'select(.type=="item.completed" and .item.type=="command_execution" and .item.status=="declined") | .item.command' runs/$ARM/*.events.jsonl
  ```
- **LLM cost.**
  ```bash
  jq -s '[.[]|select(.type=="turn.completed").usage.input_tokens]|add' runs/$ARM/*.events.jsonl
  ```
- **Jev cost and latency.** Group by harness, decision point and mode (error rate, verdicts, what was emitted, p50/p95 latency, tokens, $):
  ```bash
  python -m jevharness.adapters.hooks --summarize runs/$ARM/decisions.jsonl
  ```
- **Agreement with Guardian (arm C).** Jev's shadow `final_value` on each PermissionRequest record is paired with Guardian's outcome for the same command. The command is approved if the `command_execution` item later completes or fails. It is declined if it is `declined`. Join on `run_tag` + `tool_input` ≈ `item.command`. This gives a Jev-vs-Guardian confusion matrix at zero extra LLM cost. With `JEV_LOG_STATE=1`, the same records are also training data for distilling Guardian into a Jev-compatible model (`docs/TRAINING_RECIPES.md`).
- **Guardian overhead.** Guardian assessments are transient events that are not written to the rollout (`rollout/src/policy.rs:144`) and are not an exec item type. Measure Guardian's wall-clock and token overhead as the difference between arms C and A (`runs.jsonl` wall time and `turn.completed` usage). Whether Guardian's own tokens are included in `turn.completed.usage` has not been verified. Treat the token delta as a lower bound unless you also run the app-server.

### Terminal-Bench via Harbor

Harbor's installed `codex` agent always runs `codex exec --dangerously-bypass-approvals-and-sandbox --json`. Its `dangerously_bypass_hook_trust` option adds `--dangerously-bypass-hook-trust` (`harbor/src/harbor/agents/installed/codex.py:72-76, 1605-1614`).

- **What works with the stock agent.** Only arms B, F and +S are reachable. Pass the TOML hooks from `config.toml` in the agent's native `config`, which Harbor uploads to `$CODEX_HOME/config.toml`. The task image must provide `jevharness`, a Python for the hook command, and network access to the Jev endpoint. Otherwise every call fails open, and the log shows the errors.
- **Guardian arms.** Arms C–E need a Harbor agent subclass that replaces the yolo flag with `--approve-for-me`.

### Routing

Codex offers no hook that changes the model for a turn. `UserPromptSubmit` can only log the routing decision, or add context with `JEV_ROUTING_CONTEXT=1`. To actually route, you need a Responses-API-compatible proxy registered as a custom `[model_providers.<id>]` with `wire_api = "responses"` (`model-provider-info/src/lib.rs:163`). This adapter does not ship one.
