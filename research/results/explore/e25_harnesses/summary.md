# E25: per-step routing in two more open-source agent harnesses (exploratory, descriptive)

Registered in docs/PREREG_A7_STAGE2.md (Deviations 2026-10-09, E25) before any E25 episode. Strong LLM Gemini 3.5 Flash, fast LLM Gemini 3.5 Flash-Lite (minimal thinking, direct API); decision model Jev (OpenRouter) at tau 0.5. Cost = LLM + decision model per task; time = Harbor's agent-execution seconds per task. Full set (an errored trial counts as a failure); complete case below each table.

## pi on Terminal-Bench 2.0 (pi 0.99.2)

| arm | n | success % | cost $/task | saving % [95 % CI] | mean s | median s | time saving % [95 % CI] | fast-LLM share % | timeouts | infra errors |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 89 | 42.7 | 0.5103 | -- | 257 | 121 | -- | 0.0 | 5 | 5 |
| jev_router | 89 | 50.6 | 0.4611 | 9.6 [-13.6, 27.5] | 260 | 120 | -1.0 [-19.6, 15.3] | 12.4 | 2 | 4 |
| always_cheap | 89 | 36.0 | 0.1484 | 70.9 [61.8, 78.0] | 336 | 150 | -30.5 [-60.4, -6.2] | 100.0 | 9 | 3 |

| arm vs baseline | n | wins | losses | delta pt [bootstrap 95 %] | Tango 95 % | NI p (3 pt) | NI | median-time saving % | cheaper on % tasks | faster on % tasks |
|---|---|---|---|---|---|---|---|---|---|---|
| jev_router (full set) | 89 | 12 | 5 | 7.9 [-1.1, 16.9] | [-1.3, 17.6] | 0.012 | yes | 0.6 | 55.1 | 49.4 |
| always_cheap (full set) | 89 | 9 | 15 | -6.7 [-16.9, 3.4] | [-17.8, 4.2] | 0.761 | no | -24.7 | 79.8 | 36.0 |
| jev_router (complete case) | 83 | 12 | 5 | 8.4 [-1.2, 18.1] | [-1.4, 18.8] | 0.013 | yes | -1.3 | 57.8 | 50.6 |
| always_cheap (complete case) | 84 | 8 | 15 | -8.3 [-19.0, 2.4] | [-19.8, 3.0] | 0.831 | no | -21.8 | 84.5 | 36.9 |

Per task (descriptive): baseline 38.5 agent steps / 1213k input tokens; jev_router 38.2 agent steps / 1156k input tokens; always_cheap 50.6 agent steps / 2104k input tokens.

Defect run (not the router arm): 74 trials of the first router run, 18.9 % solved; 51 ended on Gemini's missing-thought-signature error (0 in baseline).

Trials whose final result carries an infrastructure error (failures in the full set; their pairs leave the complete case): always_cheap/pytorch-model-recovery: NonZeroAgentExitCodeError: Command failed (exit 1): . ~/.nvm/nvm.sh; mkdir -p /logs/agent/pi/sessions && pi --print --mo; always_cheap/qemu-alpine-ssh: NonZeroAgentExitCodeError: Command failed (exit 100): apt-get update && apt-get install -y curl; always_cheap/qemu-startup: NonZeroAgentExitCodeError: Command failed (exit 100): apt-get update && apt-get install -y curl; baseline/mteb-retrieve: RuntimeError: Docker compose command failed for environment mteb-retrieve. Command: docker compose --project-name mteb-r; baseline/overfull-hbox: RuntimeError: Docker compose command failed for environment overfull-hbox. Command: docker compose --project-name overfu; baseline/pytorch-model-recovery: NonZeroAgentExitCodeError: Command failed (exit 1): . ~/.nvm/nvm.sh; mkdir -p /logs/agent/pi/sessions && pi --print --mo; baseline/qemu-alpine-ssh: NonZeroAgentExitCodeError: Command failed (exit 100): apt-get update && apt-get install -y curl; baseline/qemu-startup: NonZeroAgentExitCodeError: Command failed (exit 100): apt-get update && apt-get install -y curl; jev_router/caffe-cifar-10: NonZeroAgentExitCodeError: Command failed (exit 137): . ~/.nvm/nvm.sh; mkdir -p /logs/agent/pi/sessions && pi --print --; jev_router/pytorch-model-recovery: NonZeroAgentExitCodeError: Command failed (exit 1): . ~/.nvm/nvm.sh; mkdir -p /logs/agent/pi/sessions && pi --print --mo; jev_router/qemu-alpine-ssh: NonZeroAgentExitCodeError: Command failed (exit 100): apt-get update && apt-get install -y curl; jev_router/qemu-startup: NonZeroAgentExitCodeError: Command failed (exit 100): apt-get update && apt-get install -y curl

Time (amendment 2026-10-09 ~03:05 UTC): the router arm ran again from 03:02 UTC with the fix, while baseline and always_cheap finished, so its later trials ran on a less loaded host than baseline's; the time comparison is weaker for this harness than the cost and success comparisons.

Router: 3272 routing decisions; Jev answered 88.7 %, escalation rules 11.1 %, decision-model errors (fail-open to the strong LLM) 0.2 %; 12.4 % of the LLM requests went to the fast LLM; decision-model cost $0.00252 and 6.2 s per task.

## SWE-agent on SWE-bench Verified (SWE-agent v1.1.0)

| arm | n | success % | cost $/task | saving % [95 % CI] | mean s | median s | time saving % [95 % CI] | fast-LLM share % | timeouts | infra errors |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 100 | 63.0 | 0.2605 | -- | 110 | 98 | -- | 0.0 | 0 | 0 |
| jev_router | 100 | 57.0 | 0.2403 | 7.7 [-1.9, 16.3] | 124 | 106 | -12.7 [-27.1, -0.1] | 23.2 | 0 | 0 |
| always_cheap | 100 | 28.0 | 0.0813 | 68.8 [64.3, 72.6] | 102 | 98 | 7.5 [-3.0, 17.6] | 100.0 | 0 | 0 |

| arm vs baseline | n | wins | losses | delta pt [bootstrap 95 %] | Tango 95 % | NI p (3 pt) | NI | median-time saving % | cheaper on % tasks | faster on % tasks |
|---|---|---|---|---|---|---|---|---|---|---|
| jev_router (full set) | 100 | 2 | 8 | -6.0 [-12.0, 0.0] | [-13.4, 0.2] | 0.848 | no | -8.6 | 59.0 | 34.0 |
| always_cheap (full set) | 100 | 2 | 37 | -35.0 [-45.0, -25.0] | [-45.3, -24.9] | 1.000 | no | -0.5 | 95.0 | 49.0 |
| jev_router (complete case) | 100 | 2 | 8 | -6.0 [-12.0, 0.0] | [-13.4, 0.2] | 0.848 | no | -8.6 | 59.0 | 34.0 |
| always_cheap (complete case) | 100 | 2 | 37 | -35.0 [-45.0, -25.0] | [-45.3, -24.9] | 1.000 | no | -0.5 | 95.0 | 49.0 |

Per task (descriptive): baseline 29.3 agent steps / -- input tokens; jev_router 33.4 agent steps / -- input tokens; always_cheap 44.2 agent steps / -- input tokens.

Router: 3342 routing decisions; Jev answered 100.0 %, escalation rules 0.0 %, decision-model errors (fail-open to the strong LLM) 0.0 %; 23.2 % of the LLM requests went to the fast LLM; decision-model cost $0.00309 and 7.5 s per task.

## Spend

```
{
 "tb": {
  "logged": {
   "llm": 125.1116,
   "jev": 0.352525,
   "total": 125.4641,
   "trials": 347,
   "by_job": {
    "eval/always_cheap": 13.2079,
    "eval/baseline": 45.4125,
    "eval/jev_router": 40.8156,
    "eval_defect/jev_router": 25.1282,
    "smoke/always_cheap": 0.1129,
    "smoke/baseline": 0.2379,
    "smoke/jev_router": 0.1967
   }
  },
  "or_key_usage_start": "17.609693035",
  "or_key_usage_end": "18.286930267",
  "written": "2026-10-09T05:44:35Z"
 },
 "swe": {
  "logged": {
   "llm": 59.1604,
   "jev": 0.319109,
   "total": 59.4796,
   "trials": 318,
   "by_job": {
    "eval/always_cheap": 8.1334,
    "eval/baseline": 26.0458,
    "eval/jev_router": 23.7203,
    "smoke/always_cheap": 0.0986,
    "smoke/baseline": 0.2189,
    "smoke/jev_router": 0.2216,
    "smoke_1/always_cheap": 0.0474,
    "smoke_1/baseline": 0.2334,
    "smoke_1/jev_router": 0.1945,
    "smoke_2/always_cheap": 0.0202,
    "smoke_2/baseline": 0.0939,
    "smoke_2/jev_router": 0.1324
   }
  },
  "or_key_usage_start": "17.656772515",
  "or_key_usage_end": "18.222163453",
  "written": "2026-10-09T04:52:08Z"
 }
}
```
