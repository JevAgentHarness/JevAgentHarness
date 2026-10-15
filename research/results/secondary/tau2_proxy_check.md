# C7 proxy check (A5.4 (e)): E2 (JevGym, Flash) vs. τ² airline rankings

- generated 2026-10-03T13:08:55+00:00 at git `cf3dbb4fb9b90f8057dc31fbfb84e59cf38c740e` by `scripts/tau2_proxy_check.py`; registered: PREREGISTRATION A5.4 (e); A6.20 (f) (code item); Deviations 2026-10-03 (tau^2 runs) item (7)
- stated difference: PREREGISTRATION Deviations 2026-10-03 (tau^2 runs) item (7): the airline sweep rows were run with scripts/run_tau2.sh as registered in MAIN_RUN_PLAN step 9, which sets tool_select tau = tool_params tau, whereas the E2a sweep arms fix the argument threshold at 0.90 (A6.16 (h)); the E2 and tau^2 sweep configurations therefore share their argument threshold only at tau = 0.9, and the proxy check is reported with that difference.
- guard mapping (interpretation): tau^2 `jev_guard` (policy guard before WRITE calls + transfer guard) has no E2 twin; E2a's tool-call guard `jev_ts_guard` is read as its E2 counterpart (results/primaries/c7_selection.json, A6.20 (d))
- slip rule: A6.20 (f) lists the proxy check's Kendall-tau code as a slip-rule item; it landed after the tau^2 runs and before the check is reported (Deviations item (7)); nothing was re-run
- reading: configurations: A5.4 (e)'s parenthetical list (LLM-only, CASCADE at each swept tau, ASSIST k = 3, policy GUARD; the airline proxy-check arms of MAIN_RUN_PLAN step 9); the E2-carried tau-hat* cascade, which also ran on both, is a sensitivity row (n = 10)
- reading: paired Delta-success: E2 per-task success on the arm's registered tasks; tau^2 majority pass (>= 2 of 3; A6.14, A6.16 (e)) on the 35 airline test tasks, pass^1 descriptive
- reading: iso-quality cost savings: paired net savings of the configurations that pass the `noninferior` qualifier at the unadjusted level (A6.18 (c); 3 pt on JevGym, 5 pt on tau^2 majority pass); the others tie below them
- reading: Kendall tau-b; exact permutation p over all n! assignments, one-sided for agreement (tau > 0), two-sided alongside; tau-b undefined when a ranking is constant

## Kendall τ-b — A5.4 (e) configurations (n = 9)

| ranking | role | τ-b | p (one-sided, τ > 0) | p (two-sided) | n | method |
|---|---|---|---|---|---|---|
| paired Delta-success vs. LLM-only (E2: per-task success; tau^2: majority pass) | registered | 0.657 | 0.0083 | 0.0158 | 9 | exact |
| iso-quality cost savings (qualifying configurations by net savings, the rest tied below) | registered | 0.730 | 0.1111 | 0.1111 | 9 | exact |
| paired Delta-success with tau^2 pass^1 (mean over the 3 trials) | descriptive | 0.609 | 0.0149 | 0.0295 | 9 | exact |
| paired net cost savings without the qualifier | descriptive | -0.500 | 0.9777 | 0.0752 | 9 | exact |

## Kendall τ-b — sensitivity: + the E2-carried τ̂\* cascade (n = 10)

| ranking | role | τ-b | p (one-sided, τ > 0) | p (two-sided) | n | method |
|---|---|---|---|---|---|---|
| paired Delta-success vs. LLM-only (E2: per-task success; tau^2: majority pass) | registered | 0.699 | 0.0027 | 0.0052 | 10 | exact |
| iso-quality cost savings (qualifying configurations by net savings, the rest tied below) | registered | 0.476 | 0.2000 | 0.2000 | 10 | exact |
| paired Delta-success with tau^2 pass^1 (mean over the 3 trials) | descriptive | 0.659 | 0.0049 | 0.0095 | 10 | exact |
| paired net cost savings without the qualifier | descriptive | -0.511 | 0.9857 | 0.0466 | 10 | exact |

## Configurations

| configuration | E2a arm | τ² arm | args τ E2 / τ² | E2 n | E2 Δ pt | E2 NI (3 pt) | E2 savings % | τ² n | τ² Δ maj. pt | τ² Δ pass¹ pt | τ² NI (5 pt) | τ² savings % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM-only | `llm_only` | `llm_only` | - / - | None | +0.0 | yes | 0.0 | None | +0.0 | +0.0 | yes | 0.0 |
| CASCADE tau=0.5 | `jev_ts_cascade_tau0.5` | `jev_ts_cascade_tau0.5` | 0.9 / 0.5 | 200 | -21.5 | no | 84.5 | 35 | -17.1 | -12.4 | no | -24.4 |
| CASCADE tau=0.7 | `jev_ts_cascade_tau0.7` | `jev_ts_cascade_tau0.7` | 0.9 / 0.7 | 200 | -24.0 | no | 81.0 | 35 | -14.3 | -8.6 | no | -5.8 |
| CASCADE tau=0.8 | `jev_ts_cascade_tau0.8` | `jev_ts_cascade_tau0.8` | 0.9 / 0.8 | 200 | -23.5 | no | 76.5 | 35 | -8.6 | -8.6 | no | -7.2 |
| CASCADE tau=0.9 | `jev_ts_cascade_tau0.9` | `jev_ts_cascade_tau0.9` | 0.9 / 0.9 | 200 | -20.5 | no | 72.0 | 35 | -2.9 | -2.9 | no | -18.1 |
| CASCADE tau=0.95 | `jev_ts_cascade_tau0.95` | `jev_ts_cascade_tau0.95` | 0.9 / 0.95 | 200 | -20.5 | no | 66.8 | 35 | -5.7 | -0.0 | no | -19.2 |
| CASCADE tau=0.99 | `jev_ts_cascade_tau0.99` | `jev_ts_cascade_tau0.99` | 0.9 / 0.99 | 200 | -8.0 | no | 39.3 | 35 | -2.9 | -2.9 | no | -5.5 |
| ASSIST k=3 | `jev_ts_assist_k3` | `jev_ts_assist_k3` | - / - | 500 | -7.8 | no | 53.7 | 35 | -8.6 | -3.8 | no | -9.2 |
| policy GUARD | `jev_ts_guard` | `jev_guard` | - / - | 200 | +0.0 | yes | -0.2 | 35 | +0.0 | +6.7 | no | -3.9 |
| CASCADE tau-hat* (E2-carried, 1.00 / 0.99) (sens.) | `jev_ts_cascade_tauhat` | `jev_ts_cascade_tauhat_e2` | 0.99 / 0.99 | 500 | +0.0 | yes | 14.2 | 35 | +5.7 | +4.8 | no | 5.7 |

## Inputs

- E2 reference `runs/e2a_jevgym_tool_decisions/llm_only.jsonl` sha256 `14d2c35fc9f1d2849482a11acc6d4574637b1b41643b8d3e06977a2376209a22`
- τ² reference `runs/tau2/airline/llm_only.jsonl` sha256 `6fb538f76ae21cb93744bdb228215bf3e8cd5fe037e9eeb2b99e41184190cd58` (35 airline test tasks × 3)
- `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.5.jsonl` sha256 `e67717a961f6f47ed23cf2a808ead87bd779d8349f0cde56aec39e8ad6fdb3c3`
- `runs/tau2/airline/jev_ts_cascade_tau0.5.jsonl` sha256 `3fd7f94394c8436365bd8641833caa5ff613f861617d3dbc600c3e19ed21cc54`
- `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.7.jsonl` sha256 `59cba2fc40cc148de7ce3f7a4c918e172576784ccbe7aab306095df68dda44cf`
- `runs/tau2/airline/jev_ts_cascade_tau0.7.jsonl` sha256 `9e494d61c7cc4fef7decd82c0e5b08d8c0d407653f031fec6085c0f358b17e6e`
- `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.8.jsonl` sha256 `eec8555f9d3613a078015d955c08ce76990bc8127c0d28bceb74413c4827ebc7`
- `runs/tau2/airline/jev_ts_cascade_tau0.8.jsonl` sha256 `b0a20ac249bd411dee3c448284512a75d104ee119c8a33d61efbd85127ffdbaf`
- `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.9.jsonl` sha256 `e9bd115d95f88b8cde334a76d6fe8371c4bbb257a01ff9a3d8885a1888b9c5ff`
- `runs/tau2/airline/jev_ts_cascade_tau0.9.jsonl` sha256 `28ecb2d7d20c3ad68c74ab70f441ec1b99fdef148b27737b074e2b603cd41588`
- `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.95.jsonl` sha256 `4894df1f752e863fb050f3717605f220231c2bb28ea2f8cde1f4cd70b2ef839d`
- `runs/tau2/airline/jev_ts_cascade_tau0.95.jsonl` sha256 `7196537ea2188a46dc8ccb4b449550a73a09493c44d341d0ebcbe16717e29d95`
- `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.99.jsonl` sha256 `e9c00020314ca37af60bfb2b1b16d51ce1d29e5bd352078dd5130feefcf349ca`
- `runs/tau2/airline/jev_ts_cascade_tau0.99.jsonl` sha256 `57625b097a5a224dfa585f3fcdb9005c5dc38c6f656c72dc011f97f9a9c7006c`
- `runs/e2a_jevgym_tool_decisions/jev_ts_assist_k3.jsonl` sha256 `9065c8675fedc24253051aa115f82779ba18b868af206d85f315ea93f54269fd`
- `runs/tau2/airline/jev_ts_assist_k3.jsonl` sha256 `526e6aa0f74aab25c19290b76bf2391c9d388e0dc11cb1208428d6910bf7298c`
- `runs/e2a_jevgym_tool_decisions/jev_ts_guard.jsonl` sha256 `ac9567ef11cc6465c710505c72d137e8c6cdee060182eb8cbcbad99d28539448`
- `runs/tau2/airline/jev_guard.jsonl` sha256 `3a2735d22a9060bb19df1bdc2442df1ee4084f99bb5e873acfe33a570e3da634`
- `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tauhat.jsonl` sha256 `752dd92afc9e01a5d0c5693be17f40b2a65cb59ab80f97007d1b323b0dbd7909`
- `runs/tau2/airline/jev_ts_cascade_tauhat_e2.jsonl` sha256 `0c36e4564c7d63d70258c3bbb433a58a323e29481e82032fa498a3eeaa901e75`
