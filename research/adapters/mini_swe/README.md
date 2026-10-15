# mini-swe-agent adapter: System-1 decision points on SWE-bench

`jevharness.adapters.mini_swe.JevMiniSweAgent` is a subclass of mini-swe-agent's `DefaultAgent`
(pinned to **mini-swe-agent 2.4.6**). It adds System-1 (Jev / OpenJev / small-LM logprob) hooks to a real
coding harness. With every hook `off` it behaves exactly like mini-swe-agent's SWE-bench setup, which is
checked by `tests/test_mini_swe_adapter.py::test_baseline_matches_default_agent`. The `agent` /
`environment` / `model` sections of `swebench_jev.yaml` are copied verbatim from mini-swe-agent's
`config/benchmarks/swebench.yaml`.

| hook (`jev.<hook>`) | `DecisionLog.dp` | modes | S1 question → effect |
|---|---|---|---|
| `action_guard` | `tool_select` | off, shadow, guard | Before each bash command: Noul "reasonable, on-track, non-destructive?", or with `extra.kind: choice` a Choice over {on_track, unproductive, destructive}. **guard**: if p_ok < tau, the command is not executed and the agent gets a "blocked, reconsider" observation. A re-issued identical command runs (`max_retries`). |
| `stop` | `stop` | off, shadow, guard, trigger | **guard**: when the agent runs the submit command (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT ...`), Noul "fix verified (reproduced + tests run after the last edit)?". If p < tau, the submission is deferred with a nudge, at most `max_nudges` times. **trigger** (early stop): each step from `min_step` on, "already fixed and verified?". If p ≥ tau, the agent gets a "submit now" hint. |
| `routing` | `routing` | off, shadow, trigger | Each step, Noul p(hard). If p ≥ tau the strong `model` handles the step, otherwise `jev.cheap_model` does. These are two mini-swe `Model` objects that share one message history. |
| `reflection` | `reflection` | off, shadow, always, trigger | After a failed command (rc≠0, timeout) or a repeated command, Noul "stuck / needs rethinking?". If p ≥ tau, a reflection prompt is injected. **always** is the rule-based reference and makes no S1 call. |
| `action_type` | `action` | off, shadow, assist | Choice over {explore, search, reproduce, edit, test, submit}. **assist**: the top-k types are injected as a hint. The command S2 actually issues is classified afterwards and stored as `s2_value`. |

- **`shadow` mode** (alias `log`, as in `jevharness.adapters.hooks`) asks S1 and logs its answer but never acts on it. The stop hook asks two questions in shadow mode, logged as modes `shadow_trigger` (every step) and `shadow_guard` (at submission). The `shadow` arm is therefore the baseline agent plus a full set of S1 predictions on real trajectories. Use it for calibration, AUROC, and training-data harvest.
- **Batched S1 calls:** all pre-query questions of a step share one S1 state and go out in **one batched `decide` call**; the ledger's `decision_point` then reads like `routing+reflection+action`. The guard questions for the parallel tool calls of one LM turn are batched the same way.
- **Fail-open:** S1 errors are fail-open: the decision is logged with mode `error` and the harness behaves like the baseline.
- **Reflection:** reflection is an injected prompt, not a separate LLM call. mini-swe's tool-calling models must always return a bash call, so a free-text reflection call is not possible; the prompt asks the model to reflect in its next reasoning instead.

## Outputs

For each arm, `run` writes to `-o <dir>`:

```
<dir>/preds.json                    SWE-bench predictions (model_name_or_path = "<model>__jev-<arm>")
<dir>/<iid>/<iid>.traj.json         mini-swe trajectory (works with `mini-extra inspector`) + info.jev summary
<dir>/<iid>/<iid>.jev.json          per-instance record: every DecisionLog (per step), every S1/S2 CallRecord
                                    (S1 cost/latency; S2 = mini-swe/litellm cost + tokens), per-model stats
                                    (strong/cheap), counters (blocked, nudges, reflections, hints), exit status
<dir>/trajectories.jsonl            one jevharness Trajectory per instance (appended as instances finish)
<dir>/trajectories.collected.jsonl  the same, rebuilt from the .jev.json files at the end of the run
<dir>/jev_arm.json                  resolved arm config
```

`to_trajectory(path_or_record_or_agent, resolved=...)` converts a record into
`jevharness.harness.trajectory.Trajectory`. The `collect` sub-command does this for a whole arm and fills
`result.success` from an evaluation report, so `jevharness.bench.metrics` (`episode_table`,
`decision_table`, `paired_bootstrap`, `pareto_front`) works unchanged. Before evaluation,
`result.resolved` is `None` and `success` is False.

## Setup

```bash
. .venv/bin/activate
uv pip install "mini-swe-agent==2.4.6"          # pulls litellm, datasets, typer, rich (may move `openai`)
uv pip install -e ".[bench]"
export GEMINI_API_KEY=...                      # or OPENAI_API_KEY etc. (litellm model names)
export TYPESAFE_API_KEY=...                    # jev.s1 = {type: jev}; OpenJev: OPENJEV_BASE_URL
export MSWEA_GLOBAL_COST_LIMIT=150             # optional hard budget for the whole process (USD)
# models that litellm cannot price: export MSWEA_COST_TRACKING=ignore_errors (cost is then taken from
# configs/prices/*.yaml if the model is listed there)
python -m pytest tests/test_mini_swe_adapter.py -q   # offline, no Docker/API needed
```

## Instance selection (SWE-bench Verified, 100 instances)

Every arm must run on the same ids so that instances are paired across arms:

```bash
python -m jevharness.adapters.mini_swe ids --subset verified --split test --shuffle --slice 0:100 \
    -o runs/swe_verified_100/instance_ids.txt
```

`--shuffle` is mini-swe-agent's deterministic shuffle (sorted by id, `random.seed(42)`). Alternatives:
`--slice 0:100` without shuffle (the first 100 ids alphabetically are dominated by astropy/django), `--filter '<regex>'`, or
your own id file (one id per line, `#` comments allowed). For **Lite** use `--subset lite --split test`.

## Running

The easiest way is the arm loop, which also creates the id file:

```bash
adapters/mini_swe/run_swebench.sh                                   # default arms, 100 Verified instances
N=5 WORKERS=2 adapters/mini_swe/run_swebench.sh baseline stop_guard # smoke test first!
MODEL=gemini/gemini-3.5-flash CHEAP_MODEL=gemini/gemini-3.5-flash-lite \
  adapters/mini_swe/run_swebench.sh shadow cheap_only route          # routing study
SUBSET=lite N=100 adapters/mini_swe/run_swebench.sh
```

Environment variables: `N`, `SUBSET` (verified|lite, an HF dataset, or a local dataset directory),
`SPLIT`, `OUT` (default `runs/swe_<subset>_<N>`), `IDS`, `WORKERS`, `MODEL`, `CHEAP_MODEL`, `CONFIG`,
`CONFIG_OVERLAY` (extra `-c` specs merged after `CONFIG`, space-separated), `ENV_FILE` (default `.env`; its
keys are exported, because the Python entry point does not read `.env`), `RUN_ARGS` (extra `run` flags,
word-split), `EVAL` (none|local|sbcli), `EVAL_PY`, `EVAL_WORKERS`. The default arms are
`shadow action_guard stop_guard stop_trigger route reflect_always reflect action_hint`. `shadow` doubles as
the baseline; add `baseline` for a run without S1 calls. The pseudo-arm `cheap_only` is the baseline run
with the cheap model, i.e. the lower end of the routing trade-off. The script resumes: finished instances,
those already in `preds.json`, are skipped.

To run a single arm directly:

```bash
python -m jevharness.adapters.mini_swe run -c adapters/mini_swe/swebench_jev.yaml --arm stop_guard \
    --subset verified --split test --instance-ids runs/swe_verified_100/instance_ids.txt \
    -o runs/swe_verified_100/stop_guard -w 4
```

- Debug one instance with `--filter '^sympy__sympy-20590$' -w 1`.
- `-c` works like mini-swe-agent's option: files or `key=value` pairs (JSON-parsed), merged in order. Once you pass `-c`, the default config is **not** added, so always list the YAML first.
- Swap the S1 backend: `-c jev.s1.type=openjev -c jev.s1.base_url=http://localhost:8080`, or `-c 'jev.s1={"type":"llm_logprob","llm":{"provider":"vllm","model":"Qwen/Qwen3-4B"}}'`.
- tau sweep: `-c jev.arms.stop_guard.stop.tau=0.7 --arm stop_guard --name stop_guard_tau0.7`, with one output directory per name.
- Other flags: `-m/--model`, `--cheap-model`, `--environment-class singularity`, `--redo-existing`, `python -m jevharness.adapters.mini_swe arms` (lists the arms).
- For single-instance runs through mini-swe's own CLIs, the class can also be used via `--agent-class jevharness.adapters.mini_swe.JevMiniSweAgent` with `agent.jev: {...}` in the config. Note that `mini-extra swebench` (batch) hard-codes its agent class, which is why this adapter ships its own `run` command.

### Gemini agent and the Gemini stand-in System-1

Two overlays switch the base config (the base agent LLM, Jev S1) without editing it:

- `gemini.yaml` sets the agent to `gemini/gemini-3.5-flash` with `thinkingConfig: {thinkingLevel: minimal}`,
  and the routing model to Flash-Lite. litellm 1.82.6 prices Gemini 3.5 itself, so `agent.cost_limit`
  works and `MSWEA_COST_TRACKING` is not needed. Gemini 3.x thought signatures survive mini-swe's message
  history, because litellm carries them inside the tool-call ids.
- `s1_gemini_standin.yaml` sets every S1 hook to the Flash-Lite verbalised-distribution stand-in. Leave it
  out once Jev is reachable. For the OpenRouter route add `s1_jev_openrouter.yaml` (`provider: openrouter`, `model: null`):
  a bare `-c jev.s1.provider=openrouter` keeps the base config's `model: jev-latest`, which OpenRouter does not serve.

```bash
CONFIG_OVERLAY="adapters/mini_swe/gemini.yaml adapters/mini_swe/s1_gemini_standin.yaml" \
  N=100 WORKERS=4 EVAL=local EVAL_PY=.venv-eval/bin/python adapters/mini_swe/run_swebench.sh baseline action_guard
```

### Offline dataset (huggingface.co blocked)

`--subset` / `SUBSET` also accept a local directory holding `<split>.jsonl` in the SWE-bench schema
(`instance_id, repo, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version,
FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit`). `datasets.load_dataset(<dir>)` reads it, and
`run_swebench.sh` hands `<dir>/<split>.jsonl` to the SWE-bench harness, which accepts a `.jsonl` path as
`--dataset_name`. Do not put other `.json` files in that directory: `datasets` infers splits from file
names (a name containing `dev` becomes a validation split). The L3 smoke built its two-instance file
from per-instance records mirrored on GitHub. For the full run, export the HF dataset on a machine that
can reach it:
`python -c "from datasets import load_dataset; load_dataset('princeton-nlp/SWE-bench_Verified', split='test').to_json('swe_verified/test.jsonl')"`.

### L3 smoke (2026-10-01)

Two Verified instances (`django__django-11099`, `pytest-dev__pytest-7432`), Gemini 3.5 Flash agent
(minimal thinking), arms `baseline` and `action_guard` (τ = 0.3, Gemini stand-in S1), with
`step_limit` 80 and `cost_limit` $1 for the smoke (`RUN_ARGS`). Both arms ran in parallel, with
`WORKERS=2` each. Evaluation used the local harness (swebench 4.1.0). Outputs are in
`runs/dev/l3_smoke/swe/`.

```bash
L=runs/dev/l3_smoke/swe   # dataset/test.jsonl + instance_ids.txt prepared as above
CONFIG_OVERLAY="adapters/mini_swe/gemini.yaml adapters/mini_swe/s1_gemini_standin.yaml" SUBSET=$L/dataset N=2 \
  WORKERS=2 OUT=$L IDS=$PWD/$L/instance_ids.txt RUN_ARGS="-c agent.step_limit=80 -c agent.cost_limit=1.0" \
  EVAL=local EVAL_PY=<venv with swebench==4.1.0>/bin/python EVAL_WORKERS=2 \
  adapters/mini_swe/run_swebench.sh baseline action_guard
```

| arm | instance | resolved | steps | wall s | S2 $ | S1 $ (calls, ms/call) | guard blocks |
|---|---|---|---|---|---|---|---|
| baseline | django-11099 | ✓ | 16 | 61 | 0.078 | – | – |
| baseline | pytest-7432 | ✗ (step limit 80, empty patch) | 80 | 345 | 0.775 | – | – |
| action_guard | django-11099 | ✓ | 15 | 65 | 0.070 | 0.0052 (13, 605) | 0 |
| action_guard | pytest-7432 | ✓ | 15 | 64 | 0.081 | 0.0060 (14, 657) | 0 |

- The two instance images were pre-pulled (about 1 GB compressed, 4 GB on disk each). One anonymous pull
  failed with a 502 and Docker Hub sometimes answers 429, so pre-pull with retries. Local evaluation
  took 59 s for both arms. The stand-in approved every command with confidence ≥ 0.95, so the guard
  never fired (its question is a Noul, and τ = 0.3).
- The baseline pytest run is the expensive failure mode: the agent looped over edit/test/revert until
  it hit the step limit, re-reading a growing history (1.6M input tokens). Plan for the full config's
  `cost_limit: 3` to bind on such instances.
- **Full scale:** about $0.08 per solved instance and up to the cost limit for looping ones. Budget
  roughly $0.3–0.5 per instance per arm on Gemini 3.5 Flash with the full `step_limit` 250.
  ```bash
  CONFIG_OVERLAY="adapters/mini_swe/gemini.yaml adapters/mini_swe/s1_gemini_standin.yaml" N=100 WORKERS=4 \
    EVAL=local EVAL_PY=.venv-eval/bin/python EVAL_WORKERS=4 adapters/mini_swe/run_swebench.sh   # all default arms
  ```
  Drop `s1_gemini_standin.yaml` once Jev is reachable. Allow about 4 GB of disk per instance image.
  Run `docker image prune -af` after the last arm.
- Fixed in `run_swebench.sh`:
  - `.env` was not read, so litellm saw no `GEMINI_API_KEY`.
  - There was no way to add overlays.
  - A local dataset directory was handed to the evaluation harness as a dataset name.
  - A missing evaluation report aborted the whole loop under `set -o pipefail` instead of printing the
    "no evaluation report" warning.

  Regression tests are in `tests/test_l3_smoke.py`.
- Do not edit `run_swebench.sh` while it runs: bash reads scripts incrementally.

## Evaluation

The agent only writes `preds.json`; *resolved* comes from the SWE-bench harness.

**Hosted (sb-cli, no local Docker needed):**

```bash
pip install sb-cli                             # 0.1.5
sb-cli gen-api-key you@example.com             # then: export SWEBENCH_API_KEY=... ; sb-cli verify-api-key <code>
sb-cli submit swe-bench_verified test --predictions_path runs/swe_verified_100/stop_guard/preds.json \
    --run_id jev_stop_guard --output_dir runs/swe_verified_100/stop_guard/eval
# (lite: swe-bench_lite). The command waits for the evaluation and saves the report to
# <output_dir>/swe-bench_verified__test__jev_stop_guard.json ; later: sb-cli get-report swe-bench_verified test jev_stop_guard
```

sb-cli's quota and rate limits apply. sb-cli's `Resolved (total)` may be measured against the full split,
not your subset, so compute the rate over your subset from `resolved_ids`; `collect` does this.

**Local (swebench harness, Docker):** use a separate venv, because swebench pins its own dependencies:

```bash
uv venv .venv-eval && uv pip install --python .venv-eval "swebench==4.1.0"
cd runs/swe_verified_100/stop_guard/eval
../../../../.venv-eval/bin/python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified --split test \
    --predictions_path ../preds.json --max_workers 8 --run_id jev_stop_guard --report_dir .
# -> <model_name_or_path with / -> __>.jev_stop_guard.json   (+ logs/run_evaluation/jev_stop_guard/...)
```

swebench 4.1.0 pulls the same prebuilt `swebench/sweb.eval.x86_64.<id>` images that the agent ran in
(`--namespace swebench` is its default), so images pulled during the agent run are reused. swebench
5.x changed the dataset schema: it requires `image` / `eval_script` columns in the dataset. Only use 5.x
if your dataset copy has them.

`run_swebench.sh` automates either route: `EVAL=sbcli ...` or `EVAL=local EVAL_PY=.venv-eval/bin/python ...`.

**Merge results into the analysis format:**

```bash
python -m jevharness.adapters.mini_swe collect runs/swe_verified_100/stop_guard \
    --report runs/swe_verified_100/stop_guard/eval/<report>.json -o runs/swe_verified_100/stop_guard.jsonl
```

This prints `episode_table` (resolve rate with bootstrap CI, cost split into S1 and S2, S1/LLM calls, steps).
`--report` accepts the harness summary, sb-cli reports, or per-instance `report.json` files, and can be
repeated. `--stop-gold-from-eval` labels the final accepted-submission stop decision with the
resolved outcome, a real label for stop-verifier calibration. Paired comparison between two arms:

```python
from jevharness.harness.trajectory import load_trajectories
from jevharness.bench.metrics import paired_bootstrap, decision_table, decision_records
a = {t["task_id"]: float(t["result"]["success"]) for t in load_trajectories("runs/swe_verified_100/shadow.jsonl")}
b = {t["task_id"]: float(t["result"]["success"]) for t in load_trajectories("runs/swe_verified_100/stop_guard.jsonl")}
print(paired_bootstrap(a, b))
```

## Disk / Docker / compute requirements

- **x86_64 Linux host with Docker.** The agent runs every instance in the SWE-bench image
  `docker.io/swebench/sweb.eval.x86_64.<id with __ -> _1776_>:latest`, pulled on first use. The user
  needs Docker permissions. Apple Silicon / arm64 hosts need emulation, which is slow and flaky. For
  hosts without Docker, use `--environment-class singularity`; the image becomes `docker://...`.
- **Disk:** instance images are large, a few GB each uncompressed. Shared base and env layers help, but budget
  roughly **150-250 GB for 100 Verified instances**. The SWE-bench README recommends at least 120 GB of free
  storage, 16 GB RAM and 8 cores for the evaluation harness. Run all arms back to back on the same host so that the images are pulled once. `docker
  image prune -a` afterwards; `docker system df` to check.
- **RAM/CPU:** each worker is one running container, about 1-2 GB RAM and one core while tests run.
  `-w 4` to `-w 8` suits 16 cores and 32 GB. Keep evaluation `--max_workers` at or below 0.75 × cores.
- **Pull timeouts:** the first pull of an image can exceed mini-swe's `pull_timeout` (120 s) on slow links.
  Pre-pull with `docker pull`, or pass `-c environment.pull_timeout=900`.
- **Budget:** `agent.cost_limit: 3` USD per instance and `step_limit: 250` come from mini-swe's config.
  Check the mean cost per instance of a 5-10 instance smoke run (`collect` prints it) before launching
  all arms. S1 cost is negligible next to the LLM: Jev at $0.042/M input tokens and about 3k tokens per call.

## Weak gold labels (`jev.gold: heuristic`)

SWE-bench has no per-decision ground truth. `gold: heuristic` attaches rule-based labels:

- `action_guard`: `not is_destructive(cmd)`. This flags `rm -rf /`, `.`, `*`, the tests or `.git`; `git reset --hard`, `git clean`, `git checkout .`; pip uninstall; in-place edits of tests.
- `stop`: `fix_verified(history)`, meaning a source edit happened and something was run after the last edit.
- `reflection`: the last command failed or was a repeat.

These labels let the `oracle` / `simulated` backends drive the hooks offline and give a rough accuracy
reference for real S1 backends. They are **not** ground truth, and they are recorded as
`meta.gold_source = "heuristic"` in every trajectory. Routing and action type get no rule-based label
(`gold = None`). For action type, `s2_value` (the kind of command the LLM actually issued) gives an
S1/S2 agreement measure.
