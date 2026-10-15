# Jev hooks for DeepSeek Harness (`dsh`)

`dsh` can run unmodified command hooks through its hook bridges. This adapter
mounts dsh's **command-hook bridge** (fill in its plugin id and package name in `jev-hooks.cordis.yml`; see dsh's `packages/hooks/`). That bridge is the one whose
`PreToolUse` honours both `deny` and `ask`. Through it, the bridge runs the same script as Codex and
the generic dialect, `python -m jevharness.adapters.hooks --harness deepseek`, which is implemented in
`src/jevharness/adapters/hooks.py`.

All claims were checked against `deepseek-ai/deepseek-harness` at commit `639ed01` (2026-09-29).
Paths are relative to that repo.

## Files

- `hooks.json`: a command-hook config for `UserPromptSubmit`, `PreToolUse`, `PostToolUse` and `Stop`.
- `jev-hooks.cordis.yml`: a Cordis patch that inserts the bridge row pointing at `hooks.json`.

## Enable

```bash
cd /path/to/JevAgentHarness/research && . .venv/bin/activate && pip install -e .
sed -i "s#/ABS/PATH/JevAgentHarness/research#$PWD#g" adapters/deepseek/hooks.json adapters/deepseek/jev-hooks.cordis.yml
cp .env.example .env    # set TYPESAFE_API_KEY (and optionally JEV_S1 / JEV_MODE ...) in .env

# one-shot headless run (NDJSON event stream on stdout with --json)
dsh --profile headless --patch "$PWD/adapters/deepseek/jev-hooks.cordis.yml" "fix the failing test" --json
# or the Web UI
dsh web --patch "$PWD/adapters/deepseek/jev-hooks.cordis.yml"
```

To keep the hooks across runs, merge the patch's `insert` row into a patch layer:

- `$DSH_HOME/profiles/<profile>/cordis.patch.yml` for one profile, or
- `$DSH_HOME/cordis.patch.yml` for every profile.

Do not overwrite either file. Launcher flags such as `--profile` and `--patch` must come before the
task (`apps/cli/reference/README.md:9, 26`). Run `dsh --profile headless --patch … --dump-config` to
check that the row composed.

**Why `JEV_ENV_FILE`.** dsh launches hook processes through its subprocess service, which removes every
ambient variable whose name matches `/KEY|PASSWORD|SECRET|TOKEN/i`
(`packages/subprocess/subprocess/src/index.ts:42-66`). An exported `TYPESAFE_API_KEY` therefore never
reaches the hook, and every Jev call would fail open. `hooks.json` sets `JEV_ENV_FILE` inline. That name
survives the scrub, and the adapter loads the dotenv file without overriding existing variables.
Non-secret `JEV_*` variables, such as `JEV_MODE`, `JEV_S1` and `JEV_LOG`, are inherited normally when they
are exported in the shell that starts `dsh`.

## What the bridge gives the hook, and what the adapter does about it

| Bridge behaviour (verified) | Adapter handling |
|---|---|
| The payload base is `{session_id, transcript_path: "", cwd, hook_event_name}`, with no `permission_mode`, `model` or `turn_id` (the bridge's `src/index.ts`). `transcript_path` is always `''` (README:91, 181). | `--harness auto` recognises dsh by `transcript_path == ""` without `permission_mode`. Because there is no transcript, the task and recent trajectory come from the adapter's own session store. `UserPromptSubmit` records the prompt, `PostToolUse` records `tool_call`/`tool_result` events, and `Stop` records the final message. **Keep all four events configured**, even in `off`/`log` mode, so that the S1 state has context. |
| `PreToolUse`: `deny` → the tool is denied with our reason; `ask` → `ctx.approval`; `allow` does not pre-approve (`src/index.ts:244-250`; README:177). | allow → no output; ask → `permissionDecision:"ask"`; block → `permissionDecision:"deny"`. |
| `ask` without an approval answerer fails closed to **deny**, and the `never` policy rejects deterministically (`.agents/notes/archived/feature/2026-06-30-hook-bridges.md:35`; `packages/interaction/user-approval/README.md:12, 36`; `docs/tool-execution-pipeline.md:16`). | In headless runs a native `ask` acts as a deny. Choose explicitly with `JEV_ASK_AS=deny\|abstain\|allow`. |
| `Stop`: a blocking hook steers another step. `stop_hook_active` is always `false`, and the bridge has no consecutive-block cap (`src/index.ts:276-283`; README:180; hook-bridges note :62). | The adapter's counter file under `JEV_STATE_DIR` is the loop guard (`JEV_STOP_MAX_BLOCKS`, default 2). It resets on each new user prompt and on each allowed stop. |
| `PostToolUse`: `tool_response` is flattened to text. JSON `additionalContext` works (README:178). | The memory note uses the text. `JEV_MEMORY_CONTEXT=1` returns `additionalContext`. |
| `UserPromptSubmit`: block and JSON `additionalContext` work. The default timeout is 600 s (README:176). | Routing is logged. `JEV_ROUTING_CONTEXT=1` adds a hint. `hooks.json` sets `timeout: 5`. |
| `PermissionRequest`, `PreCompact` and `SessionEnd` are unsupported and silently dropped (README:174). | – |
| Hooks run serially and fold results as `deny > ask > allow` (`packages/hooks/hook-protocol/src/merge.ts:34-52`). `hookSpecificOutput` must carry the firing event's `hookEventName` (`codec.ts:115-133`). | The adapter always sets `hookEventName`. |

The Codex bridge (`@deepseek-ai/dsh-hooks-codex`) also works. Configure it with `adapters/codex/hooks.json`,
which uses `--harness codex`. It honours only blocking on PreToolUse, reduces every `tool_input` to
`{command}`, and sends `transcript_path: null` (`packages/hooks/hooks-codex/README.md:173-177`; `src/index.ts:297-307`).

## Experiments

dsh has no LLM approval reviewer comparable to Codex's Guardian. What decides is `dsh-user-approval`: a
human or machine answerer, with policy `ask` or `never` (`packages/bundle/base/cordis.patch.yml:245-262`).
The comparisons that make sense are:

| arm | how | notes |
|---|---|---|
| dsh default (headless) | no patch | the permission preset (`DSH_PERMISSION_MODE`, default `workspace-write`, approval `ask`) fails closed on escalations |
| danger-full-access (≈ yolo) | `DSH_PERMISSION_MODE=danger-full-access` | policy `never`, no sandbox |
| Jev shadow | patch + `JEV_MODE=log` | the same run with Jev verdicts logged; measures hook overhead |
| Jev action guard | patch + `JEV_MODE='off,PreToolUse=guard'` | Jev denies risky tool calls; pair with danger-full-access for the "Jev as the only guard" arm |
| Jev stop guard | patch + `JEV_MODE='off,stop=guard'` | forces verification before finishing |
| Jev memory / routing | `memory_write=trigger`, `routing=log` | memory notes and routing labels for analysis |

Batch runs follow the same pattern as `adapters/codex/README.md`. Set `JEV_RUN_TAG="$ARM/$TASK"`, run
`dsh --profile headless --patch … "<prompt>" --json > runs/$ARM/$TASK.ndjson` per task, and read the
following from the NDJSON stream (`packages/bundle/headless/README.md:58`):

- the `session`, `tool_call`, `tool_result` and `final` events
- the exit code (0 = completed)

Summarise the Jev side with `python -m jevharness.adapters.hooks --summarize runs/$ARM/decisions.jsonl`.

## Native plugin (not provided)

A native Cordis plugin would avoid the Python process spawned for each event, which costs about 0.2 s.
It would use the same extension points the bridge uses:

- `ctx.on('tools/pre-execute', (exec, next) => PreToolDecision)`, returning `{kind:'deny', reason}`, `{kind:'ask', reason}` or `next()` (`packages/core/tools/src/index.ts:607-611`)
- `ctx.on('agent/turn-stopping', …)` with `agent.steer(...)`

The plugin entry shape is `export const name`, `inject`, `Config`, and `apply(ctx, config)`
(the bridge's `src/index.ts`).

We did not ship a TypeScript skeleton because it could not be type-checked here: the dsh workspace has
no installed `node_modules`.
