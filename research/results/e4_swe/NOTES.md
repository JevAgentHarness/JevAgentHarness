# E4: JevAgentHarness on an open-source harness (mini-SWE-agent), exploratory and descriptive

Registered in `docs/PREREG_A7_STAGE2.md`, Deviations, 2026-10-05: "Exploratory row E4" (before any agent episode,
commit `fb84fe4`), and the entry fixing the dev threshold and the eval list (before any eval episode). E4 sits outside
every Holm table and revises no registered result. The numbers below are reported as they came out.

## Question

Can the JevAgentHarness idea (a fast decision model shares control with a slow LLM) run on a well-known open-source
harness and save time and cost without losing success? In mini-SWE-agent every step is one free-form bash command, so
the decision model cannot choose the tool. Its one lever is per-step routing: Jev decides before each step whether the
step needs slow, deliberate reasoning (System 2: DeepSeek V4 Pro with thinking on) or whether a fast path can take it
(DeepSeek V4 Flash with thinking off).

## Setup

| item | value |
|---|---|
| harness | mini-SWE-agent 2.4.6, unmodified; its own SWE-bench config; adapter `jevharness.adapters.mini_swe`, H5's routing hook |
| deep path (System 2) | `deepseek/deepseek-v4-pro` (0423), OpenRouter `reasoning: {enabled: true}` |
| fast path | `deepseek/deepseek-v4-flash` (0423), OpenRouter `reasoning: {enabled: false}` |
| provider | both pinned to StreamLake (fp8), `allow_fallbacks: false`; the pin held on every call checked |
| decision model | Jev via OpenRouter (`typesafe/jev-1.13`), one question per step: p(hard) >= tau or no answer -> deep, else fast |
| arms | `always_deep` (baseline), `jevagent` (Jev routes), `always_fast` (lower bound) |
| caps | k = 1; 250 steps; $0.42 per instance (H5's $3 x 0.2088 / 1.50); 60 s per command |
| instances | SWE-bench Verified minus H5's 100 and the H5 smoke pair; sha256 order with salt `jev-explore-E4-swe-2026-10-05`; dev 45, eval 100 (`configs/explore/e4_swe/`) |
| execution | box `<instance-id>` (m7i.8xlarge); arms interleaved task-major in one pool (dev 25 workers, eval 24); infrastructure errors re-run once |
| cost | per call OpenRouter `usage.cost` (DeepSeek and Jev); `configs/prices/models.yaml` is not used for E4's agent calls |
| evaluation | local swebench 4.1.0 |

The reasoning toggle was checked in the trajectories of the smoke: every deep-path call reported reasoning tokens
(mean 50 to 300 per step, up to 2,432), and every fast-path call reported 0. All calls were served by StreamLake.
Without a `reasoning` field, both DeepSeek V4 models think by default on OpenRouter.

The only adapter change is `OpenRouterTimeoutModel`. It is mini-SWE-agent's own `OpenRouterModel` with a configurable
request timeout (600 s instead of the hard-coded 60 s; a thinking step can be longer), and a 200 response without
`choices` is raised as a retryable error.

## Status (2026-10-06, 00:58 UTC)

**The eval readout is pending.** The eval ran to the end on the box: 300 of 300 jobs, 0 infrastructure errors, agent runs
finished at 00:37 UTC, then local SWE-bench evaluation and S3 sync. This container's AWS login expired at about 00:50
UTC, before the outputs could be fetched. No eval outcome has been read. After the next AWS login:

```bash
bash scripts/explore_e4_fetch.sh e4    # S3 -> runs/e4_swe/eval; writes results/e4_swe/{summary.md, summary.json,
                                       # per_instance.csv, h5_style.md} and paper/generated/e4_swe.tex
```

Then add the eval tables below and commit.

## Execution log

| when (UTC) | what |
|---|---|
| 10-05 21:10-21:20 | API checks (a few tokens each): reasoning toggle, tool calls, shared history, latency per path |
| 21:35 | registration `fb84fe4` (before any agent episode) |
| 21:27-21:34 | smoke, the H5 smoke pair x 3 arms: 0 errors, 1/2 resolved in each arm, $0.095 logged (development only) |
| 21:35-23:11 | dev, 45 instances x 5 arms (`results/e4_swe/dev/`) |
| 23:12 | second entry `1e1f818`: tau = 0.5 fixed by the rule; eval launched at 23:12:51 |
| 23:15-00:37 | eval agent runs, 100 instances x 3 arms, 24 workers, task-major; 0 infrastructure errors |

### Two infrastructure problems on dev (fixed before eval; both are recorded in the registration)

1. **Unbilled cut-off generations.** The pinned endpoint sometimes returns a generation that stops after a few
   tokens (`finish_reason` "error", no cache hit). OpenRouter does not bill it (`usage.cost` 0), and mini-SWE-agent's
   OpenRouter class then fails the whole instance in its cost accounting. Six dev instance-arms failed this way in the
   first 54 of 225 jobs. `OpenRouterTimeoutModel` now retries such a response (commit `11b3fcd`). The dev runner was
   stopped at 22:03 and resumed with the fix. The six got their one registered re-run, and all completed. After the fix,
   3 such responses on dev and at least 6 on eval were retried, and none failed an instance.
2. **Disk saturation.** A background pull of the eval images during dev saturated the gp3 volume (~125 MB/s) and held
   every container start for about 10 minutes. The driver no longer pulls in the background. Container start is not
   part of the agent-run time, but dev wall-clock is still not compared across arms.

## Dev (45 instances; development only, used only to fix tau)

| arm | resolved / 45 | fast-share | $ / instance | Jev $ / instance |
|---|---|---|---|---|
| always_deep | 36 | 0 % | 0.0341 | – |
| always_fast | 34 | 100 % | 0.0125 | – |
| jevagent tau 0.5 | 38 | 21.4 % | 0.0343 | 0.0034 |
| jevagent tau 0.6 | 38 | 39.0 % | 0.0287 | 0.0035 |
| jevagent tau 0.7 | 33 | 72.8 % | 0.0240 | 0.0040 |

The rule (registered before dev) gave case (1): `always_fast` (34) is within 2 of `always_deep` (36), so dev cannot
separate the settings, and eval uses the registered route tau 0.5. Without case (1), the rule would have picked
tau 0.6.

What dev already shows about time and cost (descriptive, from `results/e4_swe/dev/summary.md`):

- Per call, the fast path is faster: mean 6.4 s vs 8.3 s (medians 3.6 s vs 5.5 s). But `always_fast` needed more
  steps (49.4 vs 38.2 per instance), so its agent-run time was no shorter.
- About 92 % of agent-run time is spent waiting for the agent LLM. A Jev call takes 0.18 s, about 7 s per instance.
- At tau 0.5, jevagent was not cheaper than `always_deep` ($0.0343 vs $0.0341). Its Jev calls add $0.0034 per
  instance, about as much as the fast steps save, and a switch between the two models breaks the prompt cache (cached
  input share 0.89 vs 0.92). Cost fell only at larger fast-shares, and at tau 0.7 success dropped (33 vs 36).

## Spend

| item | $ | basis |
|---|---|---|
| API checks | about 0.02 | OpenRouter `usage.cost` of the probe calls |
| smoke | 0.095 | logged per call (DeepSeek + Jev) |
| dev | 6.08 | logged: 6.01 in the final records + 0.07 in the six errored attempts; the jobs killed at 22:03 left no record (estimated below $0.5) |
| eval | pending | logged per call; from the fetched records |

The OpenRouter key's usage went from $33.36 (21:10) to $52.42 (00:58) in total. That $19.06 is an upper bound for E4's
spend: the key is shared with other sessions (review calls) and with E4b's Jev calls. Cap: $150.

## Box

`<instance-id>` stays up for E4b's eval, which is queued behind E4's (`scripts/aws/e4b_chain.sh`). It cancels
E4's power-off, re-arms a 12 h guard and powers the box off when done. An idle guard powers it off after 30 min
without a driver or a container. Status: `s3://<bucket>/e4/status/e4.txt`.
