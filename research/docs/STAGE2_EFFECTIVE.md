# Stage 2 — effective specification (generated)

- Generated: by `scripts/stage2_spec.py` at commit `77964de` on 2026-10-15 07:39 UTC.
- Source of truth: `docs/PREREG_A7_STAGE2.md` (A7, A7.9–A7.17, Deviations). Where the two disagree, PREREG_A7_STAGE2.md wins and this script is wrong.
- Registration file sha256: `fc72bef5c74ae9165a5b746da943a891b99760f9a5862374efeda1717da8eebb`.

## 1. Tests

Family S2-1, S2-2, S2-3, S2-4: Holm at FWER 0.05. Secondary table S2-1T, S2-1L, S2-L3a, S2-L3b, S2-1X, S2-4S: Holm within at α = 0.05 (floor 0.00833333). Every test is an intersection–union test (decisive p = max of its clause p values). NI clauses: `metrics.ni_pvalue(...)["p_decisive"]` (A6.21 `bb_ordered`); cost / latency / accuracy clauses: `metrics.paired_t_test(...)["p_greater"]`. Margins: JevGym 3 pt, telecom 5 pt, S2-3 accuracy 5 pt. Reading rules: A7.4 and A7.16 (c).

| test | table | arm → reference | run | population | endpoint / clauses | defined by | not tested when |
|---|---|---|---|---|---|---|---|
| **S2-1** | family | `jevagent_best` → `llm_only` | `stage2_jevgym` | Stage-2 list, n = 500 | NI on success (m = 3 pt, `bb_ordered`) ∧ cost (paired t) | A7.4, A7.9 (a) | the profile's Flash section is empty / skipped (A7.1 (g) 5) |
| **S2-2** | family | `jevagent_best` → `best_single` | `stage2_jevgym` | Stage-2 list, n = 500 | NI on success (m = 3 pt) ∧ cost; reference = the profile restricted to g* | A7.4, A7.11 (a), (b) | single-group profile (A7.1 (g) 4) or S2-1 not tested |
| **S2-3** | family | JevAgentRL → Intern-Decision-4B; → Jev | `stage2_rl_states + corpus` | first 200 Stage-2 tasks, pooled common states | (i) paired t on acc_RL − acc_Intern; (ii) paired t on acc_RL − acc_Jev + 0.05 | A7.10 (c), (d) | no freeze record by 2026-10-07 12:00 UTC, or pre-freeze exposure (A7.10 (b), (h); see Deviations) |
| **S2-4** | family | `jevagent_best_rl` → `llm_only` | `stage2_jevgym` | Stage-2 list, n = 500 | NI on success (m = 3 pt) ∧ cost with the frozen tariff q per episode | A7.4, A7.13 (iii) | no freeze record (A7.10 (h)); no frozen tariff or a mismatched workload (A7.13 (iii)); unpriced |
| **S2-1T** | secondary | `jevagent_best` → `llm_only` | `stage2_jevgym` | Stage-2 list, n = 500 | paired t on `usage.total.latency_ms` (reference − arm) | A7.9 (b), A7.16 (a) | S2-1 not tested; p = 1 without a task-major schedule record |
| **S2-1L** | secondary | `jevagent_best_s2lite` → `llm_only_s2lite` | `stage2_jevgym` | Stage-2 list, n = 500 | NI on success (m = 3 pt) ∧ cost | A7.4 | the Flash-Lite section is empty / skipped |
| **S2-L3a** | secondary | `jevagent_best` (τ² rendering) → `llm_only` | `stage2_tau2/telecom` | A7.15 telecom sample (n ≤ 800), one attempt | NI on single-attempt success (m = 5 pt) ∧ cost per task | A7.15 (e) | all-`off` rendering; adapter validation fails (A7.15 (f)) |
| **S2-L3b** | secondary | `jevagent_best` (ALFWorld rendering) → `llm_react` | `stage2_alfworld` | ALFWorld valid_seen, 140 games | NI on success (m = 3 pt) ∧ cost | A7.4 | no action-loop decision point in the profile (A7.1 (i)) |
| **S2-1X** | secondary | `jevagent_best_x` → `llm_only_x` | `stage2_x` | Stage-2 list, n = 500 or 300 (A7.12 (c)) | NI on success (m = 3 pt) ∧ cost | A7.12 (g) | resource gate (A7.12 (c)); no eligible pair; the fitted profile delegates nothing on X; unpriced |
| **S2-4S** | secondary | `jevagent_best_sl4b` → `llm_only` | `stage2_jevgym` | Stage-2 list, n = 500 | NI on success (m = 3 pt) ∧ cost with StartLux-4B's own frozen tariff q_S per episode | A7.18 (e) 5, (f) | the 4B server cannot start (A7.18 (g)); thresholds pending or the fit delegates nothing; no tariff; unpriced |

Descriptive (no test): `jevagent_default`, `best_control` (A7.14), the leave-one-group-out rows (A7.11 (c)), `llm_only_xsmall`, the A7.13 attribution rows, the optional `gpt-6.1-sol` rows (A7.12 (i)), E1 for JevAgentRL; StartLux-Decision E1, corpus-replay (4B, 27B, size ladder 0.8B / 2B / 9B), live replace and calibration rows and the static-vs-in-the-loop table (A7.18 (e) 1-4, (h)); the JevAgentRL-v2 rows if frozen (A7.19 (f)).

## 2. Runs

### `stage2_jevgym` (`configs/experiments/stage2_jevgym.yaml`)

Suite jevgym: seeds 300000–300499; schedule `task_major`; workers 8; System-1 cache `cache/s1_stage2.sqlite`; placeholder backends `jevagent_rl`, `startlux_4b`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `llm_only` | 500 | — | main | – |
| `jevagent_best` | 500 | default | main | jevagent_best.yaml: gemini-3.5-flash |
| `best_single` | 500 | default | main | jevagent_best.yaml: gemini-3.5-flash / best_single |
| `jevagent_best_rl` | 500 | jevagent_rl | main | jevagent_best.yaml: gemini-3.5-flash / backend jevagent_rl |
| `jevagent_default` | 500 | default | main | jevagent_default.yaml: gemini-3.5-flash |
| `jevagent_best_sl4b` | 500 | startlux_4b | main | jevagent_best.yaml: gemini-3.5-flash / backend startlux_4b |
| `jevagent_minus_tool` | 200 | default | main | jevagent_best.yaml: gemini-3.5-flash / minus_tool |
| `jevagent_minus_stop` | 200 | default | main | jevagent_best.yaml: gemini-3.5-flash / minus_stop |
| `jevagent_minus_reflection` | 200 | default | main | jevagent_best.yaml: gemini-3.5-flash / minus_reflection |
| `jevagent_minus_routing` | 200 | default | main | jevagent_best.yaml: gemini-3.5-flash / minus_routing |
| `jevagent_minus_memory` | 200 | default | main | jevagent_best.yaml: gemini-3.5-flash / minus_memory |
| `llm_only_s2lite` | 500 | — | lite2 | – |
| `jevagent_best_s2lite` | 500 | default | lite2 | jevagent_best.yaml: gemini-3.5-flash-lite |

### `stage2_rl_states` (`configs/experiments/stage2_rl_states.yaml`)

Suite jevgym: seeds 300000–300499; schedule `arm_major`; workers 8; System-1 cache `cache/s1_stage2.sqlite`; placeholder backends `intern`, `jevagent_rl`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `jev_ts_replace_f` | 200 | default | main | – |
| `intern_ts_replace_f` | 200 | intern | main | – |
| `rl_ts_replace_f` | 200 | jevagent_rl | main | – |

### `stage2_x` (`configs/experiments/stage2_x.yaml`)

Suite jevgym: seeds 300000–300499; schedule `task_major`; workers 8; System-1 cache `cache/s1_stage2.sqlite`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `llm_only_x` | 500 | — | main | – |
| `jevagent_best_x` | 500 | default | main | jevagent_best_x.yaml: deepseek/deepseek-v4-pro |
| `llm_only_xsmall` | 500 | — | xsmall | – |

### `jevgym_shadow_cal_x` (`configs/experiments/jevgym_shadow_cal_x.yaml`)

Suite jevgym: seeds 5150–5299; schedule `arm_major`; workers 8; System-1 cache `cache/s1_stage2.sqlite`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `jev_shadow_cal_x` | 150 | default | main | – |

### `jevgym_shadow_cal_rl` (`configs/experiments/jevgym_shadow_cal_rl.yaml`)

Suite jevgym: seeds 5000–5149; schedule `arm_major`; workers 8; System-1 cache `None`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `rl_shadow_cal` | 150 | default | main | – |
| `rl_shadow_cal_s2lite` | 150 | default | lite2 | – |

### `stage2_rl_attribution` (`configs/experiments/stage2_rl_attribution.yaml`)

Suite jevgym: seeds 300000–300499; schedule `arm_major`; workers 4; System-1 cache `cache/s1_stage2.sqlite`; placeholder backends `rl_base`, `rl_sft_v1`, `rl_sft_v1_opd`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `rl_base_ts_replace_f` | 100 | rl_base | main | – |
| `rl_sft_v1_ts_replace_f` | 100 | rl_sft_v1 | main | – |
| `rl_sft_v1_opd_ts_replace_f` | 100 | rl_sft_v1_opd | main | – |

### `stage2_alfworld` (`configs/experiments/stage2_alfworld.yaml`)

Suite alfworld: 140 tasks, split eval_in_distribution; schedule `arm_major`; workers 8; System-1 cache `cache/s1_stage2.sqlite`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `llm_react` | 140 | — | main | – |
| `jevagent_best` | 140 | default | main | jevagent_best.yaml: gemini-3.5-flash / env alfworld |

### `stage2_startlux_states` (`configs/experiments/stage2_startlux_states.yaml`)

Suite jevgym: seeds 300000–300499; schedule `arm_major`; workers 4; System-1 cache `cache/s1_stage2_startlux.sqlite`; placeholder backends `startlux_27b`, `startlux_4b`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `sl4b_ts_replace_f` | 200 | startlux_4b | main | – |
| `sl27b_ts_replace_f` | 200 | startlux_27b | main | – |

### `jevgym_shadow_cal_startlux` (`configs/experiments/jevgym_shadow_cal_startlux.yaml`)

Suite jevgym: seeds 5000–5149; schedule `arm_major`; workers 4; System-1 cache `cache/s1_stage2_startlux.sqlite`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `sl4b_shadow_cal` | 150 | default | main | – |

### `jevgym_shadow_cal_v2` (`configs/experiments/jevgym_shadow_cal_v2.yaml`)

Suite jevgym: seeds 5000–5149; schedule `arm_major`; workers 4; System-1 cache `cache/s1_stage2_v2.sqlite`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `v2_shadow_cal` | 150 | default | main | – |

### `stage2_v2_states` (`configs/experiments/stage2_v2_states.yaml`)

Suite jevgym: seeds 300000–300499; schedule `arm_major`; workers 4; System-1 cache `cache/s1_stage2_v2.sqlite`; placeholder backends `jevagentrl_v2`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `v2_ts_replace_f` | 200 | jevagentrl_v2 | main | – |

### `stage2_v2` (`configs/experiments/stage2_v2.yaml`)

Suite jevgym: seeds 300000–300499; schedule `task_major`; workers 4; System-1 cache `cache/s1_stage2_v2.sqlite`; placeholder backends `jevagentrl_v2`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `jevagent_best_v2` | 500 | jevagentrl_v2 | main | jevagent_best.yaml: gemini-3.5-flash / backend jevagentrl_v2 |

### `jevgym_shadow_cal_sol` (`configs/experiments/jevgym_shadow_cal_sol.yaml`)

Suite jevgym: seeds 5150–5299; schedule `arm_major`; workers 4; System-1 cache `cache/s1_stage2_sol.sqlite`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `jev_shadow_cal_sol` | 150 | default | main | – |

### `stage2_sol` (`configs/experiments/stage2_sol.yaml`)

Suite jevgym: seeds 300000–300499; schedule `task_major`; workers 4; System-1 cache `cache/s1_stage2_sol.sqlite`.

| arm | n | System-1 | System-2 | profile part |
|---|---|---|---|---|
| `llm_only_sol` | 500 | — | main | – |
| `jevagent_best_sol` | 500 | default | main | jevagent_best_sol.yaml: gpt-6.1-sol |
| `llm_only_solsmall` | 500 | — | solsmall | – |

Not JevGym-runner configs: τ² telecom (`scripts/run_stage2_tau2.sh`, one attempt per task, `--task-split-name full`), E1 (`configs/decision/bfcl_when2call_rl.yaml`, `configs/decision/bfcl_when2call_startlux.yaml`), the corpus replay (`scripts/stage2_corpus.py`), the StartLux servers (`scripts/aws/serve_startlux.sh`).

JevAgentRL-v2 (A7.19 (f); exploratory, descriptive, label "exploratory (A7.19)"): `jevgym_shadow_cal_v2`, `stage2_v2_states` and `stage2_v2` above, the corpus replay `--name v2` and the tariff (`results/stage2/tariff_v2_plan.json`); served and run by `scripts/aws/stage2_gpu.sh` stages `v2_*` (`results/stage2/v2_eval_plan.json`).

## 3. Populations and seeds

- JevGym Stage-2 list: `configs/eval/jevgym_stage2_tasks.json`, n = 500, seeds 300000–300499, id-list sha256 `c167829adccbf843052d6ea8eb207a37eb13e7e1a8281cbddbbac9af976e704e`; S2-3 and the LOGO rows use its first 200 tasks, the A7.13 live rows its first 100.
- Evaluation seed blocks refused without `JEV_EVAL_RUNS=1`: [(200000, 200499)] (Stage 1), [(300000, 300499)] (Stage 2); excluded from every training harvest: [(200000, 200499), (100000, 100499), (300000, 300999), (310000, 310999), (311000, 311999), (321000, 321999), (330000, 339999)].
- Calibration seeds of the X fit: 5150–5299 (`jevgym_shadow_cal_x.yaml`, A7.12 (d)); the optional Sol row's fit uses the same seeds (`jevgym_shadow_cal_sol.yaml`, A7.12 (i)).
- Telecom pool: `configs/tau2/stage2/telecom_inventory.json` (tau2-bench 5bfa7e3), 2285 ids, 2285 eligible (eligible-list sha256 `27ccec36ec0b4bd739d627c6a0c7219423178dc4e7da30f4f3a89c701ff6528a`), 32 validation tasks (seed 731); the sample is frozen in `configs/tau2/stage2/telecom_task_ids.json` after validation (A7.15).
- ALFWorld valid_seen: 140 games (A7.2 (c)).

## 4. Failure rules

- Three passes per evaluation command; a task whose latest record is an infrastructure error after three attempts leaves the pair (drop rule; full-set sensitivity row) (A7.6, A6.22 (e) 4-5).
- Agent failures are failures; a missing record is an ERROR flag (incomplete run) (A7.4).
- S2-3: a task without four valid source trajectories leaves the corpus (worst-case sensitivity: JevAgentRL 0, comparators 1); a replay scoring failure is retried twice, then incorrect; invalid options are model failures; no question leaves C (A7.10 (d)).
- A missing or not-tested member enters its Holm table with p = 1 (A7.4).
- Simulated inputs and episodes outside the frozen lists are refused by the analysis (A7.6).

## 5. Resource gates

- S2-1X: n = 500 if the predicted total fits $25, else 300 if it fits, else not tested; decided before the first calibration episode (A7.12 (c)); record `results/stage2/x_cost_gate.json`.
- Telecom: n = min(800, N_eligible − 32, ⌊(250 − V) / (2 × 1.25 × C)⌋), frozen with the sample (A7.15 (c)).
- LOGO rows: every active group if |G| × 200 × $0.043 fits $45, else only g* removed; decided when the profile is written (A7.11 (c)).
- `best_control`: n = 500, three passes, ≤ $22 (A7.14); optional `gpt-6.1-sol` rows only if the budget allows after every registered row (A7.12 (i)): n = 500 if the predicted total fits the row's $80 envelope, else 300, else not run; decided before its first calibration episode; record `results/stage2/sol_cost_gate.json` (Deviations 2026-10-05).
- JevAgentRL: ≤ 48 L4 hours of training, ≈ 24 L4 hours of serving (A7.10 (h)).
- StartLux (A7.18 (g)): 4B on the L4 host; 27B from original weights on one GPU ≥ 64 GiB, else the Q8_0 GGUF on ≥ 40 GiB ("27B (Q8_0)"), else not run; chosen by availability before the first 27B request.
- JevAgentRL-v2 (A7.19): ≤ $400 GPU and ≤ $100 API; freeze by 2026-10-07 12:00 UTC or no evaluation row.

## 6. Prerequisites (checked by `--gate <run>`)

- `stage2_jevgym`: profile `jevagent_best.yaml` selected; task-major schedule.
- `stage2_jevgym --rl`: JevAgentRL freeze record before the deadline; training manifest.
- `stage2_rl_states`: JevAgentRL freeze record before the deadline; training manifest.
- `stage2_rl_attribution`: JevAgentRL freeze record before the deadline.
- `stage2_x`: derived profile `jevagent_best_x.yaml` selected; X cost-gate record; X prices; task-major schedule.
- `stage2_tau2`: profile selected; telecom inventory; frozen telecom sample.
- `stage2_alfworld`: profile selected.
- `stage2_jevgym --startlux`: StartLux serving manifest with the 4B served; StartLux-4B thresholds fitted; S2-4S tariff frozen.
- `stage2_startlux_states`: StartLux serving manifest with the 4B served.
- `jevgym_shadow_cal_startlux`: StartLux serving manifest with the 4B served.
- `jevgym_shadow_cal_v2`: JevAgentRL-v2 freeze record before the A7.19 (e) deadline; v2 training manifest; v2 serving plan.
- `stage2_v2_states`: JevAgentRL-v2 freeze record before the A7.19 (e) deadline; v2 training manifest.
- `stage2_v2`: profile `jevagent_best.yaml` selected; task-major schedule; JevAgentRL-v2 freeze record before the A7.19 (e) deadline; JevAgentRL-v2 thresholds fitted from its own calibration; v2 tariff frozen.
- `jevgym_shadow_cal_sol`: Sol capability manifest; Sol eligibility and cost-gate record; Sol prices.
- `stage2_sol`: derived profile `jevagent_best_sol.yaml` selected; Sol eligibility and cost-gate record; Sol prices; task-major schedule.
- `stage2_best_control`: control selection; control config.

Every gate also requires this file to be in sync and the Stage-2 fixture tests to pass: `tests/test_stage2_analysis.py`, `tests/test_stage2_corpus.py`, `tests/test_stage2_amendments.py`, `tests/test_telecom_stage2.py`, `tests/test_select_profile.py`, `tests/test_profile.py`, `tests/test_startlux.py`, `tests/test_stage2_v2.py`, `tests/test_stage2_sol.py`.
