# Research code of the paper

This folder holds the code, configurations, registrations and result summaries behind the paper
[*JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs*](../paper/JevAgentHarness.pdf).
With it you can rerun the benchmark (AgentDecisionBench), the harness with fitted thresholds (JevAgentHarness), the
registered Stage-1 and Stage-2 studies, the training of AgentDecision-4B and -0.8B, and the appendix experiments.

The routing proxy in the top level of this repository (`jah`, see [../README.md](../README.md)) is a separate,
smaller package. You do not need this folder to use it.

## Names in the paper and in the code

The code keeps the names it had while the study ran. The paper uses newer names.

| Paper | Code, configs and records |
|---|---|
| AgentDecisionBench | JevAgentBench (`jevbench` CLI, `src/jevharness/bench/`) |
| JevAgentHarness (reference loop + a profile of delegations and thresholds) | `JevAgent` (`src/jevharness/harness/agent.py`) + a profile (`configs/harness/*.yaml`) |
| DecisionGym / DecisionGym-Shift / DecisionGym-Hard | JevGym (`bench/suites/jevgym.py`), `jevgym_shift.py`, `jevgym_hard.py` |
| AgentDecision-4B (v1, frozen for Stage 2) | JevAgentRL-4B, `frozen`, `rl` (training round 5) |
| AgentDecision-4B (final, + public tool-use data) | the E12 checkpoint (`e12`, "mixed") |
| AgentDecision-0.8B | E28: StartLux-Decision-0.8B + our recipe (`e28`) |
| decision model / LLM | System 1 (`s1`) / System 2 (`s2`, `llm`) |
| decision point, mode | `DP` and `Mode` in `src/jevharness/harness/config.py` |
| common states (L2) | corpus C (`scripts/stage2_corpus.py`, `results/stage2/s23/`) |
| L1 / L2 / L3 (isolated / step level / task level) | E1 static decisions / C replay / live REPLACE runs |
| Gemini 3.5 Flash / Flash-Lite | `gemini-3.5-flash` / `gemini-3.5-flash-lite` (S2 "F" and "L") |

Experiment IDs (E1, E2a, C1F, S2-1, E4r, E21, ...) are the IDs of the registrations in [docs/](docs). The tables
below map each one to the paper.

## What is here

```
research/
  src/jevharness/      the Python package: harness loop, decision backends, LLM clients, benchmark suites,
                       metrics and statistics, adapters, training code
  configs/             every configuration the paper's runs used: experiments, decision suites, harness
                       profiles, tau2-bench and ALFWorld settings, training, prices, frozen task lists
  scripts/             experiment drivers, analysis, figure / table generators, statistics
  scripts/train/       the training chain (data, SFT, OPD, GRPO, calibration, serving)
  scripts/aws/         the cloud launchers we used (reference only; see "Cloud scripts")
  adapters/            hooks and extensions for pi, Codex, DeepSeek Harness, mini-SWE-agent and SWE-agent
  tests/               tests with simulated LLMs and decision models (no API key)
  docs/                benchmark spec, harness doc, pre-registration and amendments, Stage-2 registration
  results/             small summaries of every run (JSON / Markdown / CSV), from which the tables are made,
                       and the step-level (L2) question sets with their answer key
  paper/generated/     the numbers as typeset in the paper (\defres macros written by the scripts)
```

Not included: raw run records and trajectories (`runs/`, several GB), the training data (`data/train_v0`,
`data/train_v1`: only their manifests are here; `scripts/train/make_data_v*.sh` rebuilds them), model weights,
decision caches, third-party benchmark code and data, and our internal notes. Steps that read raw records are marked
"needs `runs/`" below.

## Install

Python 3.10 or newer. From the repository root:

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -e "research[serve,dev]"      # core + decision server + tests
uv pip install -e "research[bench]"          # optional: pandas, matplotlib, datasets (plots, some loaders)
uv pip install -e "research[train]"          # GPU hosts only: torch, transformers, trl, peft, accelerate
uv pip install -e "research[swe]"            # optional: mini-SWE-agent 2.4.6 (SWE-bench runs)
cd research && pytest -q                     # test suite, about 4 minutes, no API key
```

Run every command below from `research/`. Scripts resolve paths relative to it, and they write raw outputs to
`runs/` and caches to `cache/` (both ignored by git).

The GPU hosts need the pinned training stack in `configs/train/constraints-gpu.txt`
(`bash scripts/train/setup_gpu.sh` installs and checks it). tau2-bench needs its own Python 3.12 environment
([docs/TAU2.md](docs/TAU2.md)). ALFWorld needs `pip install alfworld` and its game files
([docs/ALFWORLD.md](docs/ALFWORLD.md)).

## API keys

Put keys in `research/.env` (copy `.env.example`) or export them. Only the variable names matter here.

| Variable | Needed for |
|---|---|
| `GEMINI_API_KEY` (or `GEMINI_API_KEYS=k1,k2` for rotation) | Gemini 3.5 Flash / Flash-Lite / 3.8 Flash as agent LLM, τ²-bench user simulator, Gemini deciders. Almost every live run. |
| `OPENROUTER_API_KEY` | Jev through OpenRouter (`{"type": "jev", "provider": "openrouter"}`), DeepSeek-V4 (E4, E4r), OpenRouter-hosted baselines |
| `OPENROUTER_API_KEY_JEV` (optional) | a key used only for Jev calls; read by `decision/jev.py` and preferred over `OPENROUTER_API_KEY` |
| `TYPESAFE_API_KEY` | Jev through the vendor API (`{"type": "jev", "provider": "typesafe"}`), instead of OpenRouter |
| `OPENAI_API_KEY` | GPT-6-Luna as agent LLM (E13), the GPT-6.1-Sol row, OpenAI Decisions API baseline (E15, E20) |
| `OPENJEV_API_KEY`, `OPENJEV_BASE_URL` | hosted OpenJev baseline (E15) |
| `HF_TOKEN` | downloading base models (Qwen3, Qwen3.5, StartLux-Decision, Intern-Decision) on GPU hosts |
| `VLLM_BASE_URL` | a self-hosted OpenAI-compatible server for open models |

The tests and every `summarize` / analysis step need no key.

**Evaluation seeds are guarded.** The runner refuses the registered DecisionGym evaluation seeds (Stage 1
200000–200499, Stage 2 300000–300499, the hard tasks 310000–310799 and 311000–311299, the shifted tasks
330000–330999) unless `JEV_EVAL_RUNS=1` is set, and the training code refuses to train on them
(`src/jevharness/bench/registry.py`). Use development seeds (`-o suite.seed0=0`) for smoke runs.

## Third-party data and models

Nothing below is redistributed here. The loaders fetch pinned versions, or the docs say how to get them.

| What | How |
|---|---|
| BFCL v4, When2Call (L1) | downloaded on first use at pinned commits (`bench/suites/static.py`) into `data/raw/` |
| τ²-bench | `git clone https://github.com/sierra-research/tau2-bench` at the commit in [docs/TAU2.md](docs/TAU2.md) |
| ALFWorld | `pip install alfworld`, then the game files ([docs/ALFWORLD.md](docs/ALFWORLD.md)); game lists in `configs/explore/e11_alfworld/`, `e6a_alfworld/`, `e23_alfworld_unseen/` |
| SWE-bench Verified | through mini-SWE-agent / SWE-agent and Docker; instance lists in `configs/explore/e4_swe/`, `e4r_swe/`, `e25_harnesses/` |
| Terminal-Bench 2.0 | through Harbor (`adapters/pi/run_tb2.sh`); task lists in `configs/explore/e25_harnesses/` |
| ToolACE (public tool-use data, E12 / E16 / E28) | download the dataset JSON yourself and pass it to `--toolace` |
| Qwen3-4B, Qwen3.5-0.8B (bases) | Hugging Face (`Qwen/Qwen3-4B`, pinned snapshot in the training records) |
| StartLux-Decision 0.8B–27B | Hugging Face; served with the vendor's server (`scripts/aws/serve_startlux.sh`). CC BY-NC 4.0. |
| Intern-Decision-4B, OpenJev, JevK5 and other baselines | Hugging Face or hosted APIs; specs in `scripts/explore_e15_baselines.py` (`spec --slug ...`) |
| Jev | commercial API (TypeSafe), also on OpenRouter as `typesafe/jev-1.13` |

Our trained checkpoints (AgentDecision-4B, -0.8B) are not in this folder. Their sha256 records are in
`results/gpu_round5/FREEZE_RECORD.json` and the experiments' summaries.

## The main tools

```bash
jevbench run configs/experiments/<exp>.yaml --out runs [-o key=value ...]   # episodes (L3), one JSONL per arm
jevbench report runs/<exp>                                                   # score card: success, $, time, deltas
jevbench decide configs/decision/<suite>.yaml --out runs                     # isolated decisions (L1)
jevbench profile configs/harness/jevagent_best.yaml                          # show a harness profile as it resolves
jevbench ping --s1 '{"type": "jev", "provider": "openrouter"}'               # check a decision model
python -m jevharness.decision.server --port 8090 --s1 '<backend spec>'       # serve any decision model in Jev's wire format
python -m jevharness.training.serve_jev ...                                  # serve a trained checkpoint in Jev's wire format
python -m jevharness.bench.threshold_theory runs/<shadow run> --arm <arm> --dp tool_select ...   # fit tau-hat (§2.1)
```

Every `-o` override follows the YAML keys (e.g. `-o workers=8 -o suite.n=20 -o suite.seed0=0`). Every LLM and
decision-model call goes through one ledger, so each episode records its cost (list prices in
`configs/prices/models.yaml`) and latency.

## Paper to command

Costs are what our runs spent (or their registered caps), at the list prices of October 2026. "GPU" means one
NVIDIA A10G 24 GB, L4 or L40S host as noted. "Needs `runs/`" means the step reads raw records that are not in this
folder: rerun the experiment first.

### Benchmark, harness and theory (Sections 2–4, Appendices A–C)

| Paper | What | Commands | Configs | Cost / hardware | Output |
|---|---|---|---|---|---|
| §2.1, App. A | threshold rule and its checks | `python scripts/check_propositions.py`; `python scripts/h1_threshold_theory.py` (needs `runs/`) | — | CPU, $0 | `results/secondary/h1.*` |
| §2.1 | fitting τ̂ per decision point in shadow mode | `jevbench run configs/experiments/jevgym_shadow_cal.yaml --out runs`, then `python -m jevharness.bench.threshold_theory ...` (see [docs/MAIN_RUN_PLAN.md](docs/MAIN_RUN_PLAN.md) step 1) | `jevgym_shadow_cal*.yaml` | API, ~$17 for all Stage-1 calibration arms | `results/calibration/tauhat_*.json` |
| §3, Table 5, App. B | DecisionGym and its variants | `src/jevharness/bench/suites/jevgym*.py`; examples `python scripts/jevgym_examples.py`; data statistics `python scripts/jevgym_data_stats.py`; answer-key audit `python scripts/jevgym_checker_audit.py` | `configs/eval/jevgym_eval_tasks.json`, `jevgym_stage2_tasks.json` (frozen task lists) | CPU, $0 | `results/jevgym_examples/`, `results/data_stats/` |
| §3, App. B.6 | benchmark difficulty (hard states) | `python scripts/benchmark_difficulty.py` (needs `runs/`) | — | CPU, $0 | `results/benchmark_difficulty/` |
| §3 L1 | isolated questions (BFCL v4, When2Call) | `jevbench decide configs/decision/bfcl_when2call.yaml --out runs` | `configs/decision/*.yaml` | API: Jev + Gemini, tens of dollars | `runs/e1_bfcl_when2call/summary.json` |
| §3 L2 | common states (5,314 states of 200 new tasks) | answer the included questions: `python scripts/stage2_corpus.py replay --corpus results/stage2/s23 --name NAME --s1 JSON`; rebuild them: `python scripts/stage2_corpus.py build --runs runs --out results/stage2/s23` (needs `runs/`) | — | decision-model calls only (cents to a few dollars) | `results/stage2/s23/replay_*.json` (questions and answer key: `questions.jsonl.gz`, `occurrences.jsonl.gz`) |
| §3 L3 | live control of the agent (REPLACE) | e.g. `JEV_EVAL_RUNS=1 jevbench run configs/experiments/stage2_rl_states.yaml --out runs` (the config's header lists the GPU arms) | `stage2_*_states.yaml` | API: Gemini Flash, ~$0.03 per episode; open models on one GPU | `runs/<exp>/` |
| §4, App. C | the harness and its profiles | `jevbench profile configs/harness/jevagent_best.yaml`; doc: [docs/JEVAGENT_HARNESS.md](docs/JEVAGENT_HARNESS.md) | `configs/harness/*.yaml` | — | — |
| §4, App. C, Table 9 | adapters for real harnesses | `adapters/pi/`, `adapters/codex/`, `adapters/deepseek/`, `adapters/mini_swe/`, `adapters/swe_agent/` (pi, Codex, DeepSeek Harness and mini-SWE-agent each have a README); [docs/HARNESS_INTEGRATION.md](docs/HARNESS_INTEGRATION.md) | — | — | — |
| Fig. 3, App. A Table 4 | the 40 point–mode cells | `src/jevharness/harness/config.py` (`E6B_CELLS`, `UNDEFINED_CELLS`, ...) | — | — | — |

### Stage 1 and Stage 2 (Section 6, Section 7.2, Appendices E and G)

| Paper | What | Commands | Configs | Cost / hardware | Output |
|---|---|---|---|---|---|
| §6, App. E | registration and checks | [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) (amendments A1–A6.22), effective version [docs/PREREG_EFFECTIVE.md](docs/PREREG_EFFECTIVE.md); `python scripts/prereg_check.py`; `python scripts/prereg_effective.py --check` | — | CPU | — |
| §6, App. E | exact size and power of the tests | `python scripts/ni_exact_size.py`; `python scripts/ni_power_exact.py`; `python scripts/c5_signflip_size.py`; `python scripts/stage2_power.py` | — | CPU, minutes to hours | `results/stage2/ni_exact_size/` |
| §7.2, Fig. 5, App. G.1–G.2 | Stage 1: each delegation alone (E1, E2a, E2b, E2c) | `STEPS="E1 E2a C7 E2b E2c" WORKERS=8 bash scripts/run_main.sh` (what it runs: [docs/MAIN_RUN_PLAN.md](docs/MAIN_RUN_PLAN.md)) | `jevgym_tool_select.yaml`, `jevgym_meta_control.yaml`, `baselines_jevgym.yaml`, `bfcl_when2call.yaml` | API: ~$930 (mostly Gemini Flash) | `runs/e2*`, `results/e2a_*`, `results/e2c_*` |
| §7.2 | Intern-Decision-4B arms (E2a') | `jevbench run configs/experiments/jevgym_tool_select_intern.yaml --out runs` | `jevgym_*_intern.yaml` | GPU (A10G) + API | `results/calibration/` |
| §7.2 (τ² transfer, C7) | τ²-bench shadow, refit and arms | `bash scripts/run_tau2.sh` ([docs/TAU2.md](docs/TAU2.md)) | `configs/tau2/*.yaml` | API: ~$235 | `results/tau2/` |
| §7.2, App. G.1 | the registered Stage-1 analysis (Holm family) | `python scripts/primaries.py --runs runs --c7-arm <arm> --out results/primaries --strict` (needs `runs/`) | — | CPU | `results/primaries/primaries.{json,md}`, `paper/generated/primaries.tex` |
| App. G.2, Fig. 5, Fig. 10 | module map | `python scripts/module_map.py` (needs `runs/`) | — | CPU | `results/module_map/`, `paper/generated/module_map.tex` |
| App. G.3 | secondary analyses H1, H3, H4 | `python scripts/h3_delegation_classes.py`; `python scripts/h4_tost.py`; `python scripts/secondary_tex.py` (need `runs/`) | — | CPU | `results/secondary/` |
| §6, §7.3 | Stage 2: choose the combination | `python scripts/select_profile.py --runs runs --out results/stage2` | writes `configs/harness/jevagent_best.yaml` | CPU | `results/stage2/profile_selection.json` |
| §7.3, App. G.3 | Stage 2: the combination on new tasks | gate `python scripts/stage2_spec.py --gate stage2_jevgym`; run `RUN=stage2_jevgym bash scripts/run_stage2.sh` (also `stage2_alfworld`, `stage2_rl_states`, `stage2_v2`, `stage2_x`, `stage2_sol`); τ²: `bash scripts/run_stage2_tau2.sh` | `configs/experiments/stage2_*.yaml`; spec [docs/STAGE2_EFFECTIVE.md](docs/STAGE2_EFFECTIVE.md) | API: ~$320; S2-3/S2-4 also GPU (the 4B served on one A10G) | `runs/stage2_*` |
| §7.3, App. G.1 | Stage-2 tests S2-1 … S2-4 and secondary rows | `python scripts/stage2_analysis.py --runs runs --out results/stage2 --strict` (needs `runs/`) | — | CPU | `results/stage2/stage2.{json,md}`, `paper/generated/stage2.tex` |
| App. F, Fig. 6 | economics and wall-clock | `python scripts/stage2_economics.py`; `python scripts/stage2_tariff.py`; `python scripts/stage2_failure_audit.py` | — | CPU | `results/stage2/economics.json`, `failure_audit.*` |
| App. G.3 | GPT-6.1-Sol row | `python scripts/stage2_sol.py` | `stage2_sol.yaml`, `jevagent_best_sol.yaml` | API (OpenAI) ~$15 | `results/stage2/sol/` |

### Training AgentDecision-4B and -0.8B (Section 5, Section 7.4, Appendix D)

| Paper | What | Commands | Configs | Cost / hardware | Output |
|---|---|---|---|---|---|
| §5, App. D.1–D.2 | training data on the agent's own states (15,694 states, 587 tasks) | `bash scripts/train/make_data_v1.sh` (module `training/data_v1.py`) | `configs/train/data_v0_jevgym.yaml` | API: ~$5 (Flash-Lite episodes + Jev soft labels) | `data/train_v1/`, manifest in `results/training/` |
| §5, Table 16 | SFT (R3-v1: soft CE + Brier, LoRA r 32) | `bash scripts/train/12_sft_v1.sh sft` | — | GPU: ~2.2 h on one L40S | checkpoint |
| §5, Table 16 | GRPO in the agent loop (G = 8, iteration 15 kept) | `bash scripts/train/70_grpo_online.sh` with `CONFIG=configs/train/grpo_online_round5.yaml`; stage choice `python scripts/train/r5_chain.py ...` | `grpo_online_round5.yaml`, `opd_round5.yaml` | GPU: ~4.5 h on one L40S + ~$27 of Flash-Lite | `results/gpu_round5/` (freeze record, chain decision) |
| §5 | calibrate, serve in Jev's wire format, evaluate | `bash scripts/train/90_calibrate_serve_eval.sh` | — | GPU | — |
| §5 | the whole round on one cloud GPU host | `scripts/aws/gpu_round5_rl.sh` (reference; see "Cloud scripts") | — | GPU ~15 h, ~$45 host + ~$142 API | `results/gpu_round5/` |
| App. D.3 | recipe takeaways by stage | `python scripts/recipe_takeaways.py` (needs the training data and `runs/`) | — | CPU | `paper/generated/recipe.tex` |
| App. D.4, Table 17 | SFT ablation (E3) | `scripts/aws/train_ablation.sh`; box helpers `scripts/train/ablation_{box,score,subsets}.py` | `results/training/ablation_plan.md` | GPU (A10G), ≤24 GPU-h | `results/training/`, `paper/generated/recipe_ablation.tex` |
| §7.4, Table 3 | GRPO without SFT (E31) | `python scripts/explore_e31.py ...`; `scripts/aws/e31_rl_only.sh` | `configs/train/grpo_online_e31_rlonly.yaml`, `configs/explore/e31_rl_only/` | GPU + ~$22 Gemini | `results/explore/e31_rl_only/` |
| §7.4, Table 3, App. D.9 | + public tool-use data (E12), StartLux-4B + recipe (E12c), larger bases (E16) | `python scripts/explore_e12_train.py build --toolace data.json --out DIR`, then `scripts/aws/e12_train.sh`; `scripts/aws/e12c_train.sh`; `python scripts/explore_e16_best_open.py ...` | `configs/explore/e12_train/`, `e16_best_open/` | GPU (A10G), ≤12 GPU-h each | `results/explore/e12/`, `e16_best_open/` |
| §7.4, Table 3 | AgentDecision-0.8B (E28) and the Qwen3.5-0.8B variant (E27) | `python scripts/explore_e28_sl08b.py build ...`, `scripts/aws/e28_sl08b.sh`; `python scripts/explore_e27_small_shift.py ...` | `configs/explore/e28_sl08b/`, `e27_small_shift/` | GPU + ~$25 Gemini each | `results/explore/e28_sl08b/`, `e27_small_shift/` |
| App. D.5, Table 18 | 0.8B scoring head (E14), read-out ablation and learned deferral (E17, E18) | `python scripts/explore_e14_small.py ...`; `python scripts/explore_e17_e18.py ...`; `scripts/train/e14_clef.py`, `e18_defer.py` | `configs/explore/e14_small/`, `e17_e18/` | GPU (A10G) ~6 h + up to ~$40 Gemini | `results/explore/e14_small/`, `e17_e18/` |
| App. D.6, Table 19 | DecisionGym-Hard: retraining on the new rule (E7d), random rules (E19) | `python scripts/explore_e7d_tier.py ...`; `python scripts/explore_e19_hard_random.py ...` | `configs/explore/e7_hard/` | GPU + ≤$56 Gemini | `results/explore/e7d_tier/`, `e19_hard_random/` |
| App. D.7 | the recipe on ALFWorld (E11) | `python scripts/explore_e11_alfworld.py select\|harvest-scripted\|build\|fit\|summarize`; `scripts/aws/e11_alfworld.sh` | `configs/explore/e11_alfworld/` | GPU + ≤$60 Gemini | `results/explore/e11_alfworld/` |
| App. D.8 | step-level accuracy on public benchmarks (E24) | `python scripts/explore_e24_public_l2.py build-alfworld`, `replay ...`, `summarize` | `configs/explore/e24_public_l2/` | decision-model calls, a few dollars | `results/explore/e24_public_l2/` |
| App. D.9 | the recipe on τ²-bench (E22, Table 21) | `python scripts/explore_e22_tau2.py split\|harvest\|build\|replay\|fit\|summarize`; `scripts/aws/e22_tau2.sh` | `configs/explore/e22_tau2/` | GPU + ~$120 Gemini | `results/explore/e22_tau2/` |
| App. D.10 | train / evaluation hygiene | `python scripts/leakage_audit.py`; seed guards in `training/seed_guard.py` | — | CPU | `paper/generated/leakage.tex` |
| App. Table 15 | small decision models (E5) | `python scripts/e5_analysis.py ...`; `scripts/aws/e5_small.sh` | `configs/experiments/e5_small_live*.yaml` | GPU | `results/explore/e5_small/` |
| — | JevAgentRL-v2 on Miles (exploratory) | `src/jevharness/training/miles_port/`, [docs/MILES_PORT.md](docs/MILES_PORT.md), `scripts/train/r6_chain.py`, `scripts/aws/gpu_round6_miles.sh` | `configs/train/miles_*.yaml` | GPU | `results/gpu_round6_miles/` |

Training recipes and their history: [docs/TRAINING_RECIPES.md](docs/TRAINING_RECIPES.md).

### Results added after Stage 1 (Section 7, Tables 1–3, Figure 4, Appendices C, G)

| Paper | What | Commands | Configs | Cost / hardware | Output |
|---|---|---|---|---|---|
| Fig. 4a, App. G.6–G.7 | isolated accuracy vs cost (E9) and every open model at its cheapest deployment (E30) | `python scripts/explore_e9_frontier.py check` / `summarize`; `python scripts/explore_e30_cost.py summarize`; `python scripts/e30_bench.py` | `configs/explore/e9_frontier/` | E9: API, a few dollars; E30: one L4 GPU | `results/explore/e9_frontier/`, `e30_cost/` |
| Fig. 4, App. G.7 | more decision models as baselines at L1–L3 (E15) | `python scripts/explore_e15_baselines.py spec\|replay\|quick\|summarize`; `bash scripts/e15_live_local.sh`; `scripts/aws/e15_baselines.sh` | `configs/explore/e15_baselines/` | GPU + ≤$40 OpenAI + ≤$35 Gemini | `results/explore/e15_baselines/` |
| Fig. 4b, Table 3, App. G.9 | DecisionGym-Shift: L2 corpus (questions included in `results/explore/e21_shift/c/`), offline answers, live L3 (E21) | `python scripts/explore_e21_shift.py build\|replay\|summarize`; `bash scripts/e21_local.sh`; `scripts/aws/e21_shift.sh` | `configs/explore/e21_shift/` | GPU + ~$58 Gemini | `results/explore/e21_shift/`, `paper/generated/e21_shift.tex` |
| App. G.9, Table 40 | the final harness on the shift (E21b) | `python scripts/explore_e21b_shift_cascade.py ...` | `configs/explore/e21b_shift_cascade/` | GPU + ~$14 Gemini | `results/explore/e21b_shift_cascade/` |
| Table 1 (hard tasks) | DecisionGym-Hard (E7, E7b, E7e, E7f) | `python scripts/explore_e7_hard.py pilot\|project\|summarize`; `jevbench run configs/explore/e7_hard/eval_e7f.yaml --out runs` | `configs/explore/e7_hard/` | API ~$40 per 200-task row | `results/explore/e7_hard/`, `e7f/` |
| Table 1 (GPT-6-Luna) | other agent LLMs on DecisionGym-Hard (E13) | `python scripts/explore_e13_llms.py --llm luna run` | — | API: ≤$60 OpenAI + ≤$5 Jev | `results/explore/e13/` |
| Table 1 (coding) | mini-SWE-agent, DeepSeek-V4-Pro / Flash (E4, E4r) | `python scripts/explore_e4_swe.py select`, `run`, `threshold`, `--exp e4r readout ...`; `scripts/aws/e4r_swe.sh` | `configs/explore/e4_swe/`, `e4r_swe/`; `adapters/mini_swe/` | CPU host with Docker + ~$40 OpenRouter | `results/e4_swe/`, `results/e4r_swe/` |
| Table 1 (coding) | mini-SWE-agent with Gemini 3.8 Flash high / low thinking (E4b) | `scripts/aws/e4b_chain.sh`; `adapters/mini_swe/e4b_*.yaml` | — | CPU host with Docker + Gemini | `results/e4b_swe/` |
| Table 1, App. C.3 | SWE-agent on SWE-bench Verified, pi on Terminal-Bench 2.0 (E25) | `python scripts/explore_e25_harnesses.py ...`; `scripts/aws/e25_harnesses.sh`; `adapters/swe_agent/`, `adapters/pi/run_tb2.sh` | `configs/explore/e25_harnesses/` | CPU host with Docker + ~$125 Gemini per harness | `results/explore/e25_harnesses/` |
| App. C.3 | the decision layer as a proxy in open-source harnesses (E29, E29b; no-go) | `python scripts/explore_e29_port.py ...`; `python scripts/explore_e29b_route.py ...`; `python -m jevharness.adapters.route_proxy` | `configs/explore/e29_port/`, `e29b_route/` | API, ≤$300 | `results/explore/e29_port/`, `e29b_route/` |
| Table 1 (ALFWorld) | action cascade with Jev (E6a), 4B trained on ALFWorld (E11), unseen rooms (E23) | `python scripts/explore_e6a_alfworld.py ...`; `python scripts/explore_e11_alfworld.py ...`; `python scripts/explore_e23_alfworld_unseen.py ...` | `configs/explore/e6a_alfworld/`, `e11_alfworld/`, `e23_alfworld_unseen/`; `configs/experiments/alfworld_action_only.yaml` | Gemini ≤$60 per row; E11 / E23 also GPU | `results/explore/e6a_alfworld/`, `e11_alfworld/`, `e23_alfworld_unseen/` |
| Table 1 (outcome column) | paired non-inferiority of each row | `python scripts/hard_ni.py` | — | CPU | `paper/generated/hard_ni.tex` |
| Table 2 (thresholds, checks) | acceptance gates: own confidence, re-ask, LLM check (E8) | `python scripts/explore_e8_gates.py sample\|run\|score` | — | API, small | `results/explore/e8_gates/` |
| Table 2 (serving), App. C.1 | the final harness: shared vs own GPU, latency | `python scripts/harness_final.py diagnose\|workload\|bench\|analyze`; `scripts/aws/harness_final_gpu.sh` | `configs/experiments/harness_final_*.yaml`, `configs/harness/jevagent_final.yaml` | GPU + API | `results/harness_final/` |
| Fig. 5, App. G.2 (exploratory cells) | the remaining module-map cells (E6a–E6f) | `python scripts/explore_e6{b,d,e,f}_modes.py spend\|project\|summarize`; `python scripts/explore_e6c_alfworld.py ...` | `configs/explore/e6*/` | API ≤$60–150 per row | `results/explore/e6*/` |
| Table 13, App. C.4 | roles of a decision model in a harness | the cells above; `python scripts/coverage_ask_act.py`; `python scripts/coverage_gates.py`; `python scripts/coverage_tex.py` | — | CPU | `results/explore/coverage/`, `paper/generated/coverage.tex` |
| §7.5, App. G.7, Table 34 | why step-level and task-level scores differ | `python scripts/explore_drift.py summarize --e15-dir DIR` (needs E15's live logs) | — | CPU, $0 | `results/explore/drift/` |
| Table 35 | ensembles and a hosted decision API behind the gate (E20) | `python scripts/explore_e20_ensembles.py split\|choose\|offline`; `bash scripts/e20_live.sh` | `configs/explore/e20_ensembles/` | ≤$10 OpenAI + ≤$25 Gemini | `results/explore/e20_ensembles/` |
| Table 22 | template baselines | `python scripts/explore_template_baseline.py` | — | CPU | `results/explore/template_baseline/` |
| App. F | prices, latency, total spend | `configs/prices/models.yaml`; `python scripts/measure_latency.py ...`; `python scripts/bench_serving.py`; `python scripts/spend_ledger.py` (needs `runs/`) | — | — | `results/spend/` |
| App. I, Figs. 11–13 | case studies | `python scripts/case_studies.py`; `python scripts/episode_to_tikz.py` (need `runs/`) | — | CPU | `results/explore/case_studies/` |
| App. B.4 | prompts | `python scripts/paper_prompts.py` | — | CPU | `results/prompts/` |

## Regenerating tables and figures

Each experiment script has a `summarize` (or `score`, `readout`) step. It reads the run records, writes
`results/<experiment>/summary.{json,md}` and the paper's numbers as `\defres{key}{value}` macros in
`paper/generated/<experiment>.tex`. The committed `paper/generated/*.tex` are the values in the PDF, so you can
diff your rerun against them.

- These steps read only `results/`, `configs/` and the code, and run as they are:
  `python scripts/hard_ni.py`, `python scripts/secondary_tex.py`, `python scripts/jevgym_data_stats.py`,
  `python scripts/jevgym_examples.py`, `python scripts/stage2_power.py` and the other statistics scripts
  (`ni_exact_size.py`, `ni_power_exact.py`, `c5_signflip_size.py`, `check_propositions.py`).
- Steps marked "needs `runs/`" read the raw JSONL records. Rerun the experiment (or its smoke on development
  seeds) first; the record format is in [docs/BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md).
- `python -m jevharness.bench.paper_numbers runs --out paper/generated/results.tex` builds the score-card keys of
  every run in `runs/`.

## Cloud scripts

`scripts/aws/` holds the launchers we used to run GPU and Docker stages on cloud hosts. They are kept for reference:
each lists its stages (data, train, evaluate, final) and the exact commands run on the host. Account-specific values
are now required environment variables: `BUCKET` (S3 bucket for code and records), `AMI`, `SG`, `SUBNETS`,
`PROFILE` (instance profile), `IID` (instance id), `SSM_PREFIX` (parameter-store prefix holding the API keys) and
`AWS_REGION`. They expect this folder to be the repository root on the host. Adapt them to your setup, or run the
listed stages by hand on any machine with the right GPU.

## Tests

```bash
cd research && pytest -q
```

The tests need no API key: LLMs and decision models are simulated or scripted. A few tests download the public
BFCL and When2Call files on first use (into `data/raw/`). Tests skip when an optional package (torch, transformers,
minisweagent, harbor, alfworld, tau2-bench) is missing. `tests/conftest.py` also skips, with the reason, the tests
whose inputs are not in this folder: the training data, raw run records, the paper's LaTeX sources, and the exact
bytes of the private cloud drivers and records.

## Licences and notes

- The code in this folder is released under the repository's MIT licence ([../LICENSE](../LICENSE)).
- StartLux-Decision weights (bases of E12c, E16 and AgentDecision-0.8B) are CC BY-NC 4.0; models trained from them
  are subject to it. Training on Jev outputs and sharing derived weights depend on the provider's terms.
- BFCL, When2Call, τ²-bench, ALFWorld, SWE-bench, Terminal-Bench and the baseline models keep their own licences.
  Download them from their sources (see above).
- Prices in `configs/prices/models.yaml` are list prices with the date they were read. Check them before you
  compare costs.
- The registrations in [docs/](docs) are append-only records of the study. They cite commit hashes of the
  development history, which is not public. Where they named our cloud resources or an optional row that never
  ran, the public copy says so in place.
- DecisionGym's customers, orders and e-mail addresses are generated from templates. The data contains no
  personal information.
