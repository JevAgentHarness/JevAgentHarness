# E4b: one model on both paths (Gemini 3.8 Flash, thinking high vs low), exploratory and descriptive

Registered in `docs/PREREG_A7_STAGE2.md`, Deviations, in three entries: "Exploratory row E4b" (`3295b03`, before any
E4b episode and before any E4 eval result was read), the budget scope amendment (`eacc993`), and the no-dev deviation
(`04b0dce`, before any eval episode). Outside every Holm table; reported whatever the outcome, beside E4.

## Why E4b

E4 could hardly save money: DeepSeek V4 is very cheap, and its two paths are two different models, so every switch
broke the prompt cache. E4b keeps one model on both paths, so the cache survives a switch, and makes the deep step
expensive and slow:

| path | model and setting |
|---|---|
| deep (System 2) | `gemini/gemini-3.8-flash`, `thinkingLevel: high` |
| fast | the same model, `thinkingLevel: low` (the model rejects `minimal` with HTTP 400) |

Both paths use the direct Gemini API. Price, verified on Google's pricing page on 2026-10-05: $0.75 / $3.75 per M
input / output tokens (output includes thinking), cache read $0.075. This is an introductory price through 2026-12-31.

## What ran

| item | detail |
|---|---|
| smoke | the H5 smoke pair x 3 arms, 2026-10-05 23:50-00:05 UTC; development only, excluded from the readout |
| dev | not run: skipped for cost (see below) |
| eval | E4's 100-instance list; `always_deep` and `jevagent` (tau 0.5, the registered route tau) on all 100, `always_fast` on the first 50 |
| caps | $1.50 and 250 steps per instance |
| schedule | 24 workers, task-major |
| budget | $400 Gemini cap: the chain's projection gate plus mini-SWE-agent's global cost limit set to what is left |

### Smoke cost (n = 2 per arm, Gemini $ per instance)

| arm | instance 1 | instance 2 | mean |
|---|---|---|---|
| always_deep | 0.40 (50 steps) | 1.16 (101 steps) | 0.78 |
| always_fast | 0.30 | 0.61 | 0.46 |
| jevagent (tau 0.5) | 0.67 | 1.40 | 1.04 |

**In the smoke, jevagent cost more than always_deep.** With n = 2 this is no evidence either way.

The Gemini API answered many smoke calls with HTTP 503 ("high demand") and some with HTTP 429 (the project's tier-3
limit of 20 M input tokens per minute for this model). mini-SWE-agent's retry absorbed them, and no run errored.
Logged smoke spend: $4.54.

### Why dev was skipped

The registered gate projected dev + eval at $303 from the smoke's means, after the always_fast cut and with dev tau
reduced to {0.5, 0.6}. At the registered 1.5x safety factor that is $455, over the $400 cap. The coordinator decided,
for the author and without raising the cap, to skip dev and run eval at the registered route tau 0.5. The eval
projection was $204, or $306 at 1.5x.

## Status (2026-10-06, 00:58 UTC)

**Eval is queued, not yet started.** The chain waits for E4's eval driver to exit; that was expected at about 00:50 UTC but was not observed from here.
This container's AWS login expired at about 00:50, so the start of E4b's eval could not be observed from here.
Expected end: about 03:00-03:30 UTC. Gemini 503/429 retries may stretch it. The chain then powers the box off.

After the next AWS login:

```bash
aws s3 cp s3://<bucket>/e4/status/e4.txt - | grep e4b    # chain progress / gates / spend
bash scripts/explore_e4_fetch.sh e4b    # S3 -> runs/e4b_swe/eval; results/e4b_swe/* and paper/generated/e4b_swe.tex
```

Then fill in this file and commit. Also confirm that the box is stopped:
`aws ec2 describe-instances --instance-ids <instance-id>`.
