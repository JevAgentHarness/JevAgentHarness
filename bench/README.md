# Terminal-Bench 2.0 evaluation

This directory reproduces the JevAgentHarness evaluation on [Terminal-Bench 2.0](https://www.tbench.ai/) with
[Harbor](https://github.com/laude-institute/harbor) 0.24.0. Each harness (Codex CLI, pi, DeepSeek Harness,
mini-swe-agent, Terminus-2) runs unchanged; the only difference between arms is the model name it asks the proxy for.
`jah-strong` puts every step on the strong tier (the baseline), `jah-auto` lets the decision model route each step.
The comparison is per harness: success rate, cost per task and agent time per task.

| File | Purpose |
|---|---|
| `run_harbor.sh` | one Harbor job: one harness x one arm on a task list |
| `harbor_agents.py` | Harbor agents: thin subclasses for Codex, pi, mini-swe-agent, Terminus-2 that name the proxy session after the trial, and a DeepSeek Harness (`dsh`) agent |
| `host_gateway.yaml` | Docker Compose override: task containers reach the host as `host.docker.internal` |
| `configs/*.yaml` | proxy configs used for the runs (strong / fast tiers, decision model, arms) |
| `splits/*.txt` | task lists: dev, test, a 20-task test subset, a two-task smoke list |
| `codex_catalog.json` | Codex model metadata for the `jah-*` model names (see [Codex](#codex)) |
| `summarize.py` | joins Harbor results with the proxy log and compares each arm with the baseline |
| `PROTOCOL.md` | the protocol written before the runs, with dated changes |

### The reported runs

| Row in the main README | Proxy config | Arms | Harnesses | Tasks |
|---|---|---|---|---|
| Results table | `configs/gemini_argon_lite.yaml` | `jah-strong`, `jah-auto-t75` | `pi`, `mswe`, `dsh`, `t2` | `splits/tb2_test.txt` |
| Argon + Gemini 3.8 Flash | `configs/gemini_argon_flash.yaml` | `jah-strong`, `jah-auto` | `pi`, `dsh`, `mswe` (the protocol also ran `t2`) | `splits/tb2_test.txt` |
| Codex + GPT-6.1 Sol / GPT-6 Luna | `configs/openai_sol_luna.yaml` | `jah-strong`, `jah-auto` | `codex` | `splits/tb2_test20.txt` |

In the original runs, the results table's baseline is the `jah-strong` arm of the Argon + Flash runs (see the
"Changes" section of `PROTOCOL.md`). `jah-strong` uses only the strong tier, and both Gemini configs define it the same
way, so a reproduction can run both arms against one proxy started with `gemini_argon_lite.yaml`.

## Prerequisites

- A Linux host with Docker Engine and the Compose plugin. Each concurrent trial runs one task container; the agents
  install themselves inside it (npm or pip), so containers need outbound network access.
- Python 3.12 or newer (Harbor 0.24.0 requires it).
- This package with the `bench` extra, from the repository root:

  ```bash
  python3.12 -m venv .venv && . .venv/bin/activate
  pip install -e ".[bench]"          # jah + harbor==0.24.0
  harbor --version
  ```

Run every command below from the repository root: `run_harbor.sh` resolves `TASKS` and `JOBS_DIR` against the current
directory.

## 1. Keys

Only the proxy needs provider keys. Every agent gets the dummy key `jah-proxy-holds-the-keys` and the proxy URL, so no
provider key enters a task container.

| Config | Tiers | Keys the proxy needs |
|---|---|---|
| `configs/gemini_argon_flash.yaml` | Gemini 4 Argon / Gemini 3.8 Flash | `GEMINI_API_KEY`, `OPENROUTER_API_KEY` |
| `configs/gemini_argon_lite.yaml` | Gemini 4 Argon / Gemini 3.5 Flash-Lite | `GEMINI_API_KEY`, `OPENROUTER_API_KEY` |
| `configs/openai_sol_luna.yaml` | GPT-6.1 Sol / GPT-6 Luna (OpenAI Responses API) | `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |

`OPENROUTER_API_KEY` is for the decision model (Jev on OpenRouter) in every config. `OPENAI_API_KEY` is needed only
for the OpenAI pair. A variable may hold several comma-separated keys (a key pool to spread rate limits); each
session sticks to one key.

Set the keys only in the shell that runs the proxy, not in the shell that runs Harbor.

`jah serve` refuses to start when a tier's key is missing, but it does not check the decision model's key. Without a
working decision model every routed step falls back to the strong tier (`reason: decision_error` in the log), and
`jah-auto` quietly behaves like `jah-strong`. Check the key before a run:

```bash
jah decide -c bench/configs/gemini_argon_lite.yaml     # one decision: probabilities, cost and latency
```

## 2. Start the proxy

```bash
export GEMINI_API_KEY=...  OPENROUTER_API_KEY=...
jah serve -c bench/configs/gemini_argon_lite.yaml --host 0.0.0.0 --port 8787 --log-dir runs/proxy
```

One proxy, with one log directory, serves every arm of a comparison: sessions are keyed by model name and Harbor
trial, so arms can run side by side against the same proxy.

The proxy listens on `0.0.0.0` because task containers reach it through the Docker host gateway
(`host.docker.internal`, mapped by `host_gateway.yaml`). The proxy has no authentication and spends your keys, so
make sure port 8787 is not reachable from outside the host (firewall or security group). Check reachability from a
container with:

```bash
docker run --rm --add-host host.docker.internal:host-gateway curlimages/curl -s http://host.docker.internal:8787/health
```

### Arms

An arm is a model name defined under `models:` in the config:

| Arm | What it does | Configs |
|---|---|---|
| `jah-strong` | every step on the strong tier (baseline) | all |
| `jah-auto` | routed, tau = 0.5 | all |
| `jah-auto-t75` | routed, tau = 0.75 (escalates less often) | `gemini_argon_flash`, `gemini_argon_lite` |
| `jah-fast` | every step on the fast tier (reference) | all |

`gemini_argon_flash.yaml` also defines `jah-auto-t85`, `jah-auto-t75m` (fast tier at medium thinking) and
`jah-auto-eff` (fast tier = Argon at low thinking), and `openai_sol_luna.yaml` defines `jah-auto-m` (fast tier at
medium reasoning). These were explored on the dev split only.

A model name that is not in the config is served as `default_model` (`jah-auto`) and logged under that name, so
`summarize.py` would not find its sessions. `curl -s localhost:8787/v1/models` lists the names the proxy accepts.

## 3. Run a harness on an arm

```bash
HARNESS=<codex|pi|dsh|mswe|t2> ARM=<arm> TASKS=<task list> bash bench/run_harbor.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `HARNESS` | required | `codex`, `pi`, `dsh`, `mswe` (mini-swe-agent) or `t2` (Terminus-2) |
| `ARM` | required | proxy model name, e.g. `jah-strong`, `jah-auto`, `jah-auto-t75`, `jah-fast` |
| `TASKS` | required | file with one task name per line (`#` comments and blank lines are skipped) |
| `DATASET` | `terminal-bench@2.0` | Harbor dataset |
| `N` | `4` | concurrent trials |
| `PORT` | `8787` | proxy port on this host |
| `JOBS_DIR` | `runs/harbor` | Harbor jobs directory |
| `JOB` | `<harness>-<arm>` | job name; keep the default, `summarize.py` reads harness and arm from it |
| `CODEX_VERSION` | `0.162.1` | Codex CLI version |
| `PI_VERSION` | `1.2.0` | pi version |
| `DSH_VERSION` | `0.2.1-alpha.2` | DeepSeek Harness version |
| `MSWE_VERSION` | Harbor's default | mini-swe-agent version |
| `HARBOR` | `harbor` | Harbor executable |
| `EXTRA` | empty | more `harbor run` arguments, e.g. `EXTRA="--timeout-multiplier 2"` |

The exact invocations, one per harness (shown on the test split; use `bench/splits/tb2_dev.txt` for dev):

```bash
# Codex CLI (Responses API)
HARNESS=codex ARM=jah-strong TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh
HARNESS=codex ARM=jah-auto   TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh

# pi
HARNESS=pi ARM=jah-strong TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh
HARNESS=pi ARM=jah-auto   TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh

# DeepSeek Harness
HARNESS=dsh ARM=jah-strong TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh
HARNESS=dsh ARM=jah-auto   TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh

# mini-swe-agent
HARNESS=mswe ARM=jah-strong TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh
HARNESS=mswe ARM=jah-auto   TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh

# Terminus-2
HARNESS=t2 ARM=jah-strong TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh
HARNESS=t2 ARM=jah-auto   TASKS=bench/splits/tb2_test.txt bash bench/run_harbor.sh
```

All four arms of one harness, in sequence (or in separate shells to run them side by side):

```bash
for arm in jah-strong jah-auto jah-auto-t75 jah-fast; do
  HARNESS=pi ARM=$arm TASKS=bench/splits/tb2_dev.txt bash bench/run_harbor.sh
done
```

Run a plumbing check first, in its own jobs directory so that it does not mix with the real jobs:

```bash
HARNESS=pi ARM=jah-auto TASKS=bench/splits/smoke.txt N=2 JOBS_DIR=runs/smoke bash bench/run_harbor.sh
```

An interrupted job can be resumed with Harbor (the agents must stay importable):
`PYTHONPATH=bench harbor job resume -p runs/harbor/pi-jah-auto`.

### How each harness reaches the proxy

The agents in `harbor_agents.py` rewrite the proxy URL to `http://host.docker.internal:8787/s/<trial>/v1`, so the
proxy logs every step of a trial under the session `<arm>/<trial>`. That is how `summarize.py` joins Harbor's results
with the proxy's cost log.

#### Codex

Harbor's Codex agent with a custom model provider (`wire_api = "responses"`) passed as `--ak config=...`. Web search,
multi-agent tools and analytics are off. `JahCodex` also uploads `codex_catalog.json` into the container and sets
`model_catalog_json`, so the `jah-*` names get Codex's model metadata (GPT-5.5's prompts and its freeform `apply_patch`
tool) instead of the unknown-model fallback. The catalog lists `jah-auto`, `jah-strong`, `jah-fast` and `jah-auto-m`.
To run Codex on another arm, such as `jah-auto-t75`, first add an entry for it: copy the `jah-auto` entry and change
`slug` and `display_name`.

#### pi

Harbor's pi agent with `model_api=openai-completions` and `OPENAI_BASE_URL` set to the proxy.

#### DeepSeek Harness (`dsh`)

Harbor has no dsh agent, so `harbor_agents.Dsh` installs `@deepseek-ai/dsh` with Node 22 (nvm), writes a Cordis patch
that points dsh's pi-ai adapter at the proxy, and runs one headless turn with full permissions inside the task
container. The patch also disables dsh's session-title call and its web tools (they need a DeepSeek key and are not
part of the comparison). dsh's event stream is kept as `/logs/agent/dsh.jsonl` in each trial.

#### mini-swe-agent (`mswe`)

Harbor's mini-swe-agent with `OPENAI_BASE_URL` set to the proxy and the dummy key in `OPENAI_API_KEY` and
`MSWEA_API_KEY` (the key variable Harbor's mini-swe-agent agent reads). The model is `openai/<arm>`.

#### Terminus-2 (`t2`)

Terminus-2 calls the model from the Harbor process on the host, not from the task container, so its `api_base` is
`http://localhost:8787/v1`. Its `model_info` sets zero token prices because LiteLLM has no price for the `jah-*` names;
the proxy's log is the cost record for every harness.

## 4. Splits

| File | Tasks | Use |
|---|---|---|
| `splits/tb2_dev.txt` | 15 | dev: plumbing, cost per task, router settings |
| `splits/tb2_test.txt` | 74 | test: the reported numbers, with router settings frozen after dev |
| `splits/tb2_test20.txt` | 20 | the first 20 test tasks (a small check row) |
| `splits/smoke.txt` | 2 | two dev tasks for a plumbing check |

Dev and test are disjoint and together cover all 89 Terminal-Bench 2.0 tasks. The split is a fixed hash split made
before any run: the tasks are sorted by `sha256("jah-tb2-split-v1:" + task_name)` and the first 15 form the dev split.
`tb2_test20.txt` holds the first 20 tasks of `tb2_test.txt` in alphabetical order. Tune on dev only and report test.

```python
import hashlib
order = sorted(tasks, key=lambda t: hashlib.sha256(("jah-tb2-split-v1:" + t).encode()).hexdigest())
dev, test = sorted(order[:15]), sorted(order[15:])
```

## 5. Summarize

```bash
python bench/summarize.py runs/harbor runs/proxy --baseline jah-strong \
    --md runs/summary.md --json runs/summary.json --per-task runs/per_task.csv
```

The first argument is the jobs directory (one folder per job, named `<harness>-<arm>`); the second is the proxy's log
directory or its `steps.jsonl`. The table has one row per harness and arm:

| Column | Meaning |
|---|---|
| `success` | share of trials with Harbor reward 1 |
| `Δ vs baseline (pt) [95% CI]` | success difference to the baseline arm of the same harness, in points, on the tasks both arms completed, with a paired bootstrap 95% interval |
| `W/L` | tasks only this arm solved / tasks only the baseline solved |
| `$/task` | LLM, decision-model and context-cache dollars from the proxy log |
| `cost saving` | 1 - (summed cost of this arm / summed cost of the baseline) on the common tasks |
| `agent s/task`, `time saving` | Harbor's agent-execution seconds, and the same ratio as for cost |
| `strong share` | share of steps on the strong tier |
| `infra/timeouts` | trials that ended in an infrastructure error / an agent timeout (both count as failures) |

`--per-task` writes one CSV row per trial. `jah report runs/proxy` (add `--json` for the routing reasons, such as
`decision_error`) summarizes the proxy log by model name without Harbor's results.

### Comparing with a baseline from another run

When an arm and its baseline ran in different jobs directories (for example on two hosts, each with its own proxy),
put both job folders in one directory and concatenate the two proxy logs. Session names include the Harbor trial
name, so the logs do not collide.

```bash
mkdir -p runs/compare/jobs
ln -s "$PWD/run_a/harbor/pi-jah-strong"   runs/compare/jobs/     # baseline (copying works too)
ln -s "$PWD/run_b/harbor/pi-jah-auto-t75" runs/compare/jobs/
cat run_a/proxy/steps.jsonl run_b/proxy/steps.jsonl > runs/compare/steps.jsonl
python bench/summarize.py runs/compare/jobs runs/compare/steps.jsonl --baseline jah-strong
```

Compare only runs with the same strong tier, harness version and task list.

## 6. Cost caps

Set a spend cap in the config (`budget_usd: 50`) or when starting the proxy:

```bash
jah serve -c bench/configs/gemini_argon_lite.yaml --host 0.0.0.0 --log-dir runs/proxy --set budget_usd=50
```

At startup the proxy adds up the cost of every record already in `<log-dir>/steps.jsonl`, so the cap covers
everything logged in that directory, across restarts. Once the logged spend reaches the cap, the proxy answers the
first request of a new session with HTTP 429; sessions already running finish. A trial refused this way fails like any
agent error, so look for refused trials before reading results. Spend so far: `curl -s localhost:8787/stats`.

Prices come from the built-in table in `src/jevagentharness/prices.py` (list prices with the date they were read) or
from the upstream's billed amount when it reports one. Check the prices before reporting dollars, and add missing
models under `prices:` in the config; calls to an unpriced model are logged with `priced: false` and cost 0.

## 7. Protocol

`PROTOCOL.md` states the question, setup, phases (dev, then a frozen test), the criteria for the claim and the budget.
It was written before the first run; later decisions are appended under "Changes" with their date and reason. If you
run a new setting, write it down the same way before you start it.

## Logs and data

The bench configs set `log_states: true`, so the proxy also writes `runs/proxy/states.jsonl`: the state the decision
model saw at each step (task text, recent actions and tool output, truncated), with its p(hard) and route. This file
contains Terminal-Bench task text. `steps.jsonl` holds no prompt or completion text. Harbor keeps each trial's agent
logs and verifier output under the jobs directory.
