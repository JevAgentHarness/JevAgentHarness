# Hard tasks (H5, exploratory): AWS runs, 2026-10-02/03

Status (2026-10-03, 03:00 UTC): **done; both boxes stopped.** SWE-bench Verified: 4 arms × 100, k = 1, complete (11
errored instances re-run once, none left errored). Terminal-Bench-Science: calibration done, subset frozen
(`adapters/pi/tbsci_hard.txt`, **15 tasks — N = 15 of the registered 20**, committed and pushed in `016a5ba` before any
evaluation trial), evaluation complete at the registered k = 3 (3 arms × 15 tasks × 3 = 135 trials, 0 infrastructure
errors). Terminal-Bench 2.0 was not run (its budget is the authors' decision). Logged API spend of this round **$413.63**
of the $440 cap. Everything here is **H5, exploratory** (outside the Holm family); the smoke sections further down are
DEV numbers. Brief of this round: plan "A" (SWE-bench Verified in full, then Terminal-Bench-Science calibration +
evaluation), a hard cap of $440 of API spend (Gemini + Jev, as logged by the runs) for everything started in this round.

## Full runs, 2026-10-02/03 (H5, exploratory)

### API spend (as logged by the runs; cap $440)

| suite / stage | Gemini (System-2) | Jev (System-1) | total | basis |
|---|---|---|---|---|
| SWE-bench Verified, 4 arms × 100 (+ 11 re-runs) | $226.18 | $0.88 | **$227.05** | mini-SWE litellm cost + Jev usage of every `*.jev.json`, incl. the $7.10 of the 11 errored attempts that were re-run |
| TB-Science calibration (`oracle`, `nop`: $0; 28 `baseline` attempts) | $34.54 | $0 | **$34.54** | Harbor `agent_result.cost_usd` (`results/hard/tbsci_cal/summary.json`) |
| TB-Science evaluation, 3 arms × 15 × 3 = 135 trials | $151.63 | $0.41 (est.) | **$152.04** | Harbor `agent_result.cost_usd`; Jev estimated from the logged input tokens at $0.042 / 1M (`results/hard/tbsci/summary.json`) |
| **total of this round** | **$412.35** | **$1.29** | **$413.63** | cap $440; gate log `results/hard/tbsci/eval_budget_log.txt` |

Not counted (no cost logged): the 6 calibration trials cancelled after 3 min (deviation 3); the SWE re-evaluation and
the `oracle` / `nop` stages make no API calls. Harbor's automatic retries would delete an errored attempt's directory
(and its cost) — none happened (0 retries logged).

The OpenRouter key's `usage` is shared with the GPU box's main run, so its delta is not attributable here (it rose from
$11.72 at the SWE start to $18.83 at the start of TB-Science pass 3, 01:26).

### SWE-bench Verified (mini-SWE-agent 2.4.6; Gemini 3.5 Flash minimal; Jev via OpenRouter; k = 1, registered caps)

Box `<instance-id>`, code = bundle `8941f54` + overlay (= the committed scripts; `hard_swe.sh` pre-pull fix below).
`PHASE=full ARM_COST_LIMIT=55` (19:27 UTC; 100 images pre-pulled in 798 s, 0 failures; 4 arms × 6 workers in parallel;
local swebench 4.1.0 evaluation), then `PHASE=rerun ARM_COST_LIMIT=20` once (21:27–22:03): the 11 errored instances
(below) were re-run from scratch, no instance errored again, so the second allowed re-run was not needed. Instance list
`results/hard/swe/instance_ids.txt` (shuffled positions 0–99, sha256 `7ced229b…`). Readout: `results/hard/swe/`
(`h5_summary.md` / `.json` from `adapters/mini_swe/h5_summary.py`, `scorecard.md`, `per_instance.csv`). Leak check: 0 files.

| arm | resolved / 100 | Δ vs baseline | wins / losses | registered H5 label (`metrics.h5_verdict`) | agent-run p50 / p90 s | Δ time median, co-solved (n) | mean $ / instance |
|---|---|---|---|---|---|---|---|
| baseline | 75 | – | – | – | 111 / 435 | – | 0.563 |
| action_guard | 76 | +1 pt (LB90 −4 pt) | 7 / 6 | success-preserving (not faster) | 132 / 425 | +9.1 s (69), sign p 0.95 | 0.554 |
| route | 77 | +2 pt (LB90 −3 pt) | 8 / 6 | success-preserving (not faster) | 116 / 326 | −0.1 s (69), sign p 0.50 | 0.464 |
| stop_guard | 74 | −1 pt (LB90 −6 pt) | 6 / 7 | neither | 125 / 556 | +7.8 s (68), sign p 0.94 | 0.619 |

No arm reaches "passes the exploratory success screen; faster among jointly solved tasks". Full-set and complete-case
views are identical (no errored instance remains). Exit statuses: `LimitsExceeded` (250 steps or $3) 6 baseline, 7
stop_guard, 0 action_guard / route. All-instance median time difference: action_guard +8.8 s, route −2.4 s, stop_guard
+11.3 s; solved-by-600 s 0.74 baseline, 0.75 action_guard, 0.74 route, 0.73 stop_guard (descriptive). route sends
25 % of steps to Flash-Lite (strong-call ratio 0.75) and is 17.6 % cheaper per instance (score card, descriptive).

Evaluation-harness errors (swebench's 1,800 s test timeout, `EvaluationError`, scored unresolved by the readout as for
any unresolved prediction): action_guard `scikit-learn__scikit-learn-14710`, `sympy__sympy-19040`; route
`scikit-learn__scikit-learn-14710` — all three resolved by the baseline. The four arms' evaluations ran in parallel on
the loaded box (load average up to ~370 during the agent phase, ~80–140 during evaluation). An API-free re-evaluation of
the three predictions on the idle box (22:54–23:26 UTC, one container at a time, separate run ids; reports in
`results/hard/swe/reeval/`) resolves `scikit-learn__scikit-learn-14710` in both arms (so those two timeouts were
load-induced) and times out again on `sympy__sympy-19040` (action_guard). The readout above is left as run (no
registered rule covers evaluation timeouts); the sensitivity view with the two re-evaluated resolutions
(`results/hard/swe/h5_summary_reeval_sensitivity.md`) gives action_guard 77 and route 78 resolved, the same labels
(success-preserving, not faster; stop_guard neither).

### Terminal-Bench-Science: calibration and the frozen subset (A6.22 (d) 2)

Calibration pass `runs/tbsci_cal` on box `<instance-id>` (37-task pool, sha256 `f131ee52…`, registered caps 1 h /
100 turns, no automatic retries): `oracle` and `nop` 17:47–21:02 UTC (`PHASE=tbsci_cal_floor`), one `baseline` attempt
per task that passed validity and has a floor 21:08–22:43 (`PHASE=tbsci_cal_signal`, 6 concurrent), `tbsci.py select`
22:44 (`PHASE=tbsci_select`). No infrastructure error in any stage (4 oracle trials and 1 baseline attempt timed out,
scored as they stand), so nothing was retried and nothing stayed pending.
- validity: 9 of 37 dropped — oracle reward 0 (agent timeout at 1 h): 3x2pt-inference, energy-routing,
  frustrated-heisenberg-nqs, hysteretic-aquifer-control; oracle agent-run time > 1,800 s: betalactam-multimodal-transfer
  (2,425 s), ode-law-discovery (2,522 s), rv-astrometry-fitting (2,597 s), si-fracture-fbc (2,678 s),
  spin-glass-groundstate (3,536 s);
- signal: 28 baseline attempts, 1 solved (inelastic-constitutive-discovery), 14 above the nop floor, 13 dropped
  (tests passed ≤ floor);
- fill: 15 eligible tasks < N = 20, so **all 15 are chosen — a smaller subset, reported as such**: earth 3, engineering
  2, life 7 (4 + the 3 slots the other domains could not fill), mathematical 1, physical 2.

**Frozen list `adapters/pi/tbsci_hard.txt`: sha256 `a935a41f47096de42019b0d4d993e3f8dd7b62fc49c97b352e6900aba8d7d7f5`,
846 bytes, 15 tasks**, written on the box by `tbsci.py select` (header: pool hash, calibration run id
`tbsci_cal@<host>`, box commit `8941f54` + overlay, terminal-bench-science `cfff307`); re-derived locally from the
synced calibration results with `tbsci.py select --survey` (identical slug list). Committed and pushed before the first
evaluation trial; the sha256 and size still have to be entered in the Deviations section of `docs/PREREGISTRATION.md`
(not edited by this run). Inputs: `results/hard/tbsci_cal/` (`select.json` / `.txt`, `pre_signal.json` / `.txt`,
`signal_tasks.txt`, `summary.md` / `.json` of all calibration trials). Calibration attempts are not reused.

### Terminal-Bench-Science evaluation (3 arms × 15 tasks × k = 3; caps 1 h / 100 turns)

Run as `scripts/aws/hard_tb_eval_budget.sh` (`PRIOR_USD=227.05`, cap $440, margin 1.15): passes 1..3 of
`hard_tb.sh PHASE=tbsci_eval_pass` — one attempt per arm per pass, the three arms in parallel, Harbor
`--max-retries 2 --retry-exclude AgentTimeoutError`, nop warm-up before pass 1. This is the order Harbor's own `-k 3`
uses (attempt-major: every task's attempt 1, then attempt 2, …), cut into passes so that a spend cap can stop between
passes; a pass starts only if logged spend + projected pass cost ≤ $440 (projection = 45 trials × mean cost per trial ×
1.15; before any evaluation trial the mean is the calibration baseline cost of the 15 chosen tasks, $1.13), and a
watchdog stops a running pass if the cap would be crossed. The summary (`summarize_harbor.py`) merges the passes by job
name. Gate decisions: pass 1 at $261.59 spent + $58.61 projected; pass 2 at $315.08 + $61.50; pass 3 at $364.25 + $59.03
— all three passes ran (22:51–02:48 UTC), the watchdog never fired, the box powered itself off at 02:50.
Readout: `results/hard/tbsci/summary.md` / `.json` (every trial; full-set and complete-case H5 views); leak check 0 files.

| arm | solved / 45 trials | tasks solved ≥ once | Δ success vs baseline (LB90) | wins / losses | registered H5 label (`summarize_harbor` `h5_verdict`) | co-solved tasks | agent-run p50 / p90 s | Δ agent-run median, all tasks | mean Gemini $ / trial | steps mean |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0 | – | – | – | – | 350 / 1,287 | – | 1.13 | 72.3 |
| jev_stop_reflect | 0 | 0 | 0 pt (0) | 0 / 0 | success-preserving | 0 (faster: not estimable) | 311 / 948 | −20.8 s | 0.95 | 59.1 |
| jev_gate_stop | 1 (microarch-modeling, pass 2) | 1 | +2.2 pt (0) | 1 / 0 | success-preserving | 0 (faster: not estimable) | 352 / 995 | +13.3 s | 1.30 | 82.0 |

Full set = complete case (no errored trial). The success screen passes trivially at this floor (baseline 0/45): no arm
can be "faster among jointly solved tasks" because no task is solved by both arms, so neither the registered verdict
("passes the exploratory success screen; faster among jointly solved tasks") nor the strict tier is reached;
solved-by-600 / 1,800 / 3,600 s is 0 for baseline and jev_stop_reflect and 0.022 for jev_gate_stop. Agent timeouts at the
1 h cap: 1 trial (jev_gate_stop, pass 1); trials with ≥ 100 agent steps (the turn cap): baseline 12, jev_stop_reflect 6,
jev_gate_stop 23 of 45; the other trials ended when the agent stopped on its own. Descriptive partial credit (mean share
of verifier tests passed per trial): baseline 0.364, jev_stop_reflect 0.388, jev_gate_stop 0.369. Jev decisions:
jev_stop_reflect 231 System-1 requests (79 stop nudges, 142 reflections, 1 fail-open error); jev_gate_stop 2,928
requests (732 blocked tool calls, 45 stop nudges, 0 errors). The one calibration solve (inelastic-constitutive-discovery)
was not repeated by any evaluation trial (baseline 0/4 tests in all three attempts).

### Deviations and execution notes (this round)

1. **SWE per-arm spend guard bound** (`ARM_COST_LIMIT=55`, mini-SWE's `MSWEA_GLOBAL_COST_LIMIT`): baseline and
   stop_guard reached $55 with instances in flight; mini-SWE then raises `RuntimeError: Global cost/call limit exceeded`
   at the next model call, so 5 baseline and 6 stop_guard instances (all sympy, the tail of the list) ended as errored
   records. They were handled by the registered rule (A6.22 (e) 4: an errored trial is re-run, up to twice): records
   moved to `<arm>/_errored/<iid>.1/`, re-run from scratch in `PHASE=rerun` (ARM_COST_LIMIT 20, which did not bind).
   The re-runs ran later and with fewer containers on the box than the paired arms' attempts (time is a co-outcome).
   Their discarded attempts cost $7.10 (counted in the spend). action_guard ended at $55.39 without hitting the guard.
2. **SWE evaluation timeouts** (3 predictions, see above): scored unresolved as run; the idle-box re-evaluation is a
   sensitivity view only (2 of 3 resolve; the H5 labels do not change).
3. **Stale script on the TB box, first signal launch aborted.** The TB box still had the pre-fix `hard_tb.sh` (without
   the validity pre-pass of fix 5), so the first `tbsci_cal_signal` launch (21:05) started `baseline` on all 37 pool
   tasks; it was stopped after 3 min (6 trials cancelled, no cost logged), the 6 cancelled trials were moved to
   `runs/tbsci_cal_aborted_signal_2105/` (never used; the calibration directory holds only the valid pass), the code was
   refreshed from the overlay, and the stage was re-launched at 21:08 on the 28 eligible tasks.
4. **Floor-stage self-stop cancelled.** `tbsci_cal_floor` ran with `STOP_WHEN_DONE=1`; at its end (21:02) the power-off
   was cancelled and the 30 h guard re-armed for its original time (2026-10-03 23:22 UTC) so that the signal stage could
   follow at once. The SWE box's guard was re-armed (+30 h) when it was started for the full run.
5. **TB-Science subset smaller than N = 20**: 15 tasks (the registered fill rule; see above).
6. **Calibration signal stage at 6 concurrent trials** (the script default is 4); the calibration uses rewards and test
   counts only, never times.
7. **TB-Science evaluation in passes** (above): same arms, caps, k, retries; only the cut between attempts is explicit.
   All three passes ran, so k = 3 as registered; between passes the arms wait for the slowest one (a barrier Harbor's
   single `-k 3` job does not have).
9. **Box code**: both boxes ran the bundle `8941f54` plus the overlay of the hard-task scripts (identical to the
   committed files); the frozen list's header names the box commit `8941f54`.
10. **AWS console login of the controlling session expired** 01:10–01:31 UTC (monitoring only; the TB box ran on its
    own instance profile, nothing was interrupted).
8. **Docker Hub pre-pull** (`hard_swe.sh`): images already on the box are not pulled again, and a pull waits while the
   anonymous budget left is below 10 (no wait was needed: 100 pulled, 0 failures).

## Setup

- Code: bundle `s3://<bucket>/code/jev.bundle` at `8941f54` (branch `main`)
  plus the working-tree overlay `code/hard_overlay.tgz` (the files listed under "Files"; not yet committed when the
  boxes ran). `prereg_check` was not part of these runs (H5 is outside the C1–C8 machinery).
- Boxes (us-east-1d, Ubuntu 24.04 `<aws-resource-id>`, 500 GB gp3, profile `<instance-profile>`, SG `jev-train-noingress`,
  shutdown behaviour `stop`, user-data `shutdown -h +1800`; tags `Name=jev-hard-<purpose>`, `Project=jev`):

  | instance | name | type | launched (UTC) | stopped (UTC) | hours | used for |
  |---|---|---|---|---|---|---|
  | `<instance-id>` | jev-hard-swe | m7i.8xlarge (32 vCPU, 128 GB) | 17:22 | 17:40 | 0.3 | setup, SWE-bench smoke |
  | `<instance-id>` | jev-hard-swe | (restarted) | 10-02 19:25 | 22:05 | 2.7 | SWE-bench Verified full + re-run |
  | `<instance-id>` | jev-hard-swe | (restarted) | 10-02 22:53 | 23:27 | 0.6 | idle-box re-evaluation of 3 predictions (no API) |
  | `<instance-id>` | jev-hard-tb | m7i.8xlarge (32 vCPU, 128 GB) | 10-02 17:22 | 10-03 02:50 | 9.5 | setup, TB2 + TB-science smokes; TB-science calibration (17:47–22:44), selection, evaluation (22:51–02:48) |

  Instance-hours (≈, from the status logs): SWE box 3.5 (3.2 of them this round), TB box 9.5 (≈ 7.4 this round, from 19:25). Both boxes are
  **stopped** (shutdown behaviour `stop`; 500 GB volumes kept with the images and build cache, ≈ $1.3 per box-day of EBS).

  Before this round, the TB box was running the API-free calibration stage (`PHASE=tbsci_cal_floor STOP_WHEN_DONE=1`):
  at 19:18 UTC 19 of 37 `oracle` trials had finished, all with reward 1, and 4 heavy reference
  solutions were computing (oracle agent cap 1 h; the validity rule drops any oracle slower than 1,800 s). After
  `oracle` it runs `nop`, summarises, syncs `runs/tbsci_cal` to `s3://…/hard/tb/runs/tbsci_cal/` and powers the box
  off (shutdown behaviour `stop`); the user-data guard stops it at 2026-10-03 23:22 UTC at the latest. Check
  `s3://…/hard/tb/status/tb.txt` for `[tb:tbsci_cal_floor] done` / `stopping the box`. Stopped boxes keep their
  500 GB volumes (images and build cache), ≈ $1.3 per box-day of EBS.

- `scripts/aws/hard_setup.sh` (idempotent, ~2 min): Docker CE 29.8.2 (compose v5.6.0, buildx 0.37.1; address pool
  10.128.0.0/12, `mirror.gcr.io` as Docker Hub mirror), uv, repo venv (`.[all]` + mini-swe-agent 2.4.6), `.venv-eval`
  (swebench 4.1.0), Node 22.23.3 + pi 0.99.2, Harbor (see fix 1), terminal-bench-science at `cfff307`. Docker Hub
  anonymous budget measured on the boxes: 100 pulls / h / IP.
- System-2: Gemini 3.5 Flash, minimal thinking (mini-SWE: `adapters/mini_swe/gemini.yaml`; pi: `--thinking minimal`).
  System-1: real Jev via OpenRouter, `{"type":"jev","provider":"openrouter"}` (`typesafe/jev-1.13`, price key
  `jev-1.13.0`, $0.042/M input): mini-SWE in-process (`adapters/mini_swe/s1_jev_openrouter.yaml`, one SQLite cache per
  arm), pi through the host decision server (`jevharness.decision.server --name jev-1.13`, `JEV_WIRE_MODEL=jev-1.13`,
  τ 0.5 for gate / stop / reflect as in `run_tb2.sh`). Keys only from SSM Parameter Store into process environments;
  every run ends with a leak check (files under the output dir containing a key value): 0 in every smoke.

## Integration fixes found by the smokes

1. **Harbor version.** PyPI `harbor==0.23.0` rejects `--ak max_turns` (`Unknown option 'max_turns'`), so every pi arm
   of `run_tbsci.sh` failed at once, and its pi agent writes no ATIF `agent/trajectory.json` (agent steps empty in
   `summarize_harbor.py`). The dev smokes had used a git checkout (pyproject 0.23.0, commit `4f848e3`, 2026-09-30);
   `hard_setup.sh` now installs `harbor @ git+https://github.com/laude-institute/harbor@4f848e36…` and both TB smokes
   were re-run with it (the PyPI-Harbor TB2 smoke is archived in the bucket as `hard/tb/runs/hard_smoke_pypi_harbor`).
2. **Jev via OpenRouter in mini-SWE.** `-c jev.s1.provider=openrouter` (README) keeps the base config's
   `model: jev-latest`, a TypeSafe id OpenRouter does not serve; the new overlay `s1_jev_openrouter.yaml` sets
   `model: null` so the OpenRouter preset (`typesafe/jev-1.13`, price key `jev-1.13.0`) applies.
3. **H5 readout for mini-SWE.** `bench.scorecard` prints no H5 verdict and no agent-run times, so
   `adapters/mini_swe/h5_summary.py` applies `metrics.h5_verdict` (k = 1, time = `wall_ms`, McNemar descriptive,
   full-set and complete-case views) to `<run>/<arm>.jsonl`.
4. **Keys.** Harbor resolves `GEMINI_API_KEY` from its own environment for the `google` provider, so the runs use
   neither `--env-file` nor `--ae` for keys (both would put them in job config files); `ENV_FILE` is pointed at a
   missing file.
5. **Calibration signal stage.** `hard_tb.sh` `tbsci_cal_signal` runs the one baseline attempt only on the tasks that
   passed validity and have a floor (`tbsci.py select` pre-pass), not on all 37; `tbsci_cal_floor` is the API-free
   oracle + nop stage. Calibration keeps errored trials (no Harbor auto-retry; `select` needs them); evaluation phases
   use Harbor `--max-retries 2 --retry-exclude AgentTimeoutError` (A6.22 (e) 1, 4; Harbor deletes the errored attempt
   and logs "Retrying" in `job.log`).

## Smoke results (2 tasks per suite × every arm, k = 1, registered caps; DEV numbers, not results)

Summaries: `smoke/swe/` (`h5_summary.md`, `scorecard.md`, `instance_ids.txt`), `smoke/tb2/summary.md`,
`smoke/tbsci/summary.md` (+ JSON). Full outputs: `s3://<bucket>/hard/{swe,tb}/runs/hard_smoke_*`.
Errors (infrastructure): 0 in every suite; S1 errors / fail-open decisions: 0; every Jev hook fired at least once.

- **SWE-bench Verified** (mini-SWE-agent; instances `django__django-16502`, `scikit-learn__scikit-learn-13496` =
  shuffled positions 100–101, outside the 100-instance evaluation list; caps 250 steps / $3 / 60 s): all 8 runs
  `Submitted`, 1 of 2 resolved in every arm (local swebench 4.1.0 evaluation, one report per arm). Mean cost per
  instance: baseline $0.249, action_guard $0.272 (4 blocks on one instance), route $0.286 (37 cheap / 45 strong
  steps), stop_guard $0.425 (1 nudge). Jev: 1–52 calls per instance, $0.0001–0.004. Agent-run time 2–5 min per
  instance; images pre-pulled in 48 s.
- **Terminal-Bench 2.0** (pi + Harbor; `prove-plus-comm`, `overfull-hbox`; each task's own timeout, no turn cap):
  6/6 solved, $0.03–0.43 per trial (mean $0.20), 12–50 agent steps, agent run 16–360 s, no timeouts. Jev: stop
  verifier asked once per jev_stop / jev_gate_stop trial and accepted; tool gate 6 and 41 requests with 2 and 7 blocks.
- **Terminal-Bench-Science** (`symbolic-regression`, `variable-star-vetting`: outside the 37-task pool, expert
  estimate 2–3 h; caps 1 h, 100 turns): `oracle` reward 1 on both (0.2 s and 60 s), `nop` floors 1/2 and 2/6 tests;
  0/6 agent trials solved; 47–100 turns, agent run 850–1,680 s (the 100-turn cap binds before the 1 h cap),
  $0.45–1.21 per trial (mean $0.96, 0.6–3.1M input tokens). Jev: reflection fired 3 and 7 times and the stop verifier
  nudged 1–2 times (jev_stop_reflect); the gate blocked 7/48 and 20/72 proposals (jev_gate_stop).

Smoke API spend: Gemini $10.4 (SWE $2.46, TB2 $0.95 + $1.20 for the two TB2 smokes, TB-science $5.78); Jev
$0.03 from the logged usage. The OpenRouter key's `usage` rose from $9.83 to $10.40 over the smokes, but that key is
shared with the GPU box's main run, so the delta is not attributable to these smokes.

## Projection for the full runs (from the smokes; the brief's cap is $450 in total)

| Suite | Trials | Per-trial basis | Projection |
|---|---|---|---|
| SWE-bench Verified, 4 arms × 100, k = 1 | 400 | smoke mean $0.31; README: looping instances run to the $3 cap | $125–230 |
| Terminal-Bench 2.0, 3 arms × 89 × 3 | 801 | smoke $0.20 on two quick tasks; a timed-out trial costs ≈ $0.7 at 750–900 s and grows faster than linearly with the timeout (TB-science: 100 turns ≈ 850–1,300 s ≈ $1.1–1.2, ≈ 3M input tokens). TB2 timeouts: 49 × 900 s, 16 × 1,800 s, 12 × 3,600 s, 1 × 7,200 s, 1 × 12,000 s, … (41 h per pass); a pass where every trial times out ≈ $130–190 | $400–1,000 (25–50 % timeouts) |
| Terminal-Bench-Science calibration, ≤ 37 baseline attempts | ≤ 37 | smoke $0.96 (≤ $1.21) | $30–45 |
| Terminal-Bench-Science evaluation, 3 arms × 20 × 3 | 180 | smoke $0.96 (≤ $1.21) | $175–220 |
| **Total** | | | **≈ $730–1,500** |

Within $450 (≈ $440 left): e.g. SWE-bench Verified in full plus Terminal-Bench-Science calibration and evaluation
(≈ $330–495, likely ≈ $350–400), or SWE-bench plus Terminal-Bench 2.0 at k = 1 (≈ $260–560). Terminal-Bench 2.0 at
the registered k = 3 without a turn cap does not fit by itself at the upper end. Any reduction of k, tasks or caps is a
budget deviation to be stated in the paper (A6.22 (e) 9); choosing among these is the authors' call.

## How to resume

```bash
# Both boxes are stopped, set up, images cached; code on them = bundle 8941f54 + s3://…/code/hard_overlay.tgz (the
# committed hard-task scripts). After a start: re-arm the guard (sudo shutdown -h +1800) and refresh the overlay (as
# ubuntu, via SSM). Do not re-run hard_setup.sh while jobs run: it restarts Docker.
aws ec2 start-instances --instance-ids <id>
aws s3 cp s3://<bucket>/code/hard_overlay.tgz /opt/jev/hard_overlay.tgz && tar -xzf /opt/jev/hard_overlay.tgz -C /opt/jev/JevAgentHarness/research
# Not run: Terminal-Bench 2.0 (TB box):  nohup setsid env PHASE=tb2_full bash scripts/aws/hard_tb.sh > /opt/jev/hard_status/tb_tb2_full.log 2>&1 &
# (projection $400–1,000 at k = 3; the authors' decision). SWE-bench and TB-Science are complete.
```
