# ALFWorld (text) in JevAgentBench

`jevharness.bench.suites.alfworld` exposes ALFWorld's TextWorld games as an **action-only**
`Env`: at every step the agent sees the current admissible commands and picks one. This lets
us test *Jev-action-only* (`action: replace | cascade | assist`) against a ReAct-style LLM
policy (`configs/experiments/alfworld_action_only.yaml`).

| split (`AlfredTWEnv` name)  | directory                   | games |
|-----------------------------|-----------------------------|------:|
| `train`                     | `json_2.1.1/train`          | 3553  |
| `eval_in_distribution`      | `json_2.1.1/valid_seen`     | 140   |
| `eval_out_of_distribution`  | `json_2.1.1/valid_unseen`   | **134** (the standard ReAct / Reflexion eval set) |

The game counts apply ALFWorld's own filters: no `movable`/`Sliced` trajectories, all 6 task
types, `solvable == true`. Aliases: `valid_seen`/`id` and `valid_unseen`/`ood`.

## 1. Install (text-only)

```bash
. .venv/bin/activate
uv pip install alfworld          # text-only; pulls textworld[pddl], fast-downward-textworld, spacy, tatsu
# tested: alfworld 0.4.2 (PyPI; env code identical to master @ aaba687), textworld 1.7.0, Python 3.11, Linux
```

You don't need `alfworld[full]` (THOR, torch, MaskRCNN) for the text environment.
`jevharness` imports `alfworld`/`textworld` only inside `AlfWorldEnv.reset()`, so the rest of the
package works without them.

## 2. Data

The data is on GitHub releases (reachable from the dev container). It goes to `$ALFWORLD_DATA`, or
`~/.cache/alfworld` by default.

**Official route** (also downloads the 178 MB MaskRCNN detector, which only THOR needs):

```bash
export ALFWORLD_DATA=~/data/alfworld      # optional; default ~/.cache/alfworld
alfworld-download                          # json (72 MB) + pddl (35 MB) + tw-pddl (37 MB) + detector (178 MB)
```

**Minimal route** (text env only, ~110 MB download, ~64 MB on disk for `valid_unseen`):

```bash
export ALFWORLD_DATA=~/data/alfworld && mkdir -p $ALFWORLD_DATA && cd $ALFWORLD_DATA
curl -L -o json.zip   https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip
curl -L -o twpddl.zip https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip
unzip -q json.zip   'json_2.1.1/valid_unseen/*' 'json_2.1.1/valid_seen/*'    # traj_data.json (task type)
unzip -q twpddl.zip 'json_2.1.1/valid_unseen/*' 'json_2.1.1/valid_seen/*'    # game.tw-pddl (the game)
rm json.zip twpddl.zip
```

Each `game.tw-pddl` embeds the PDDL domain, the text grammar, the problem, and the planner
`walkthrough`, so the `logic/` directory isn't needed for the text env.

Check the setup:

```bash
python -c "from jevharness.bench.suites.alfworld import collect_game_files as c; print(len(c()))"   # -> 134
ALFWORLD_DATA=... python -m pytest tests/test_suites_extra.py -k real_alfworld                      # 1 real game
```

## 3. Run

```bash
jevbench run configs/experiments/alfworld_action_only.yaml --out runs            # all arms, 134 games
jevbench run configs/experiments/alfworld_action_only.yaml --max-tasks 10        # cheap dry run
jevbench report runs/alfworld_action_only
```

Suite options (`suite:` block, forwarded to `load_alfworld`):

| key | default | meaning |
|---|---|---|
| `split` | `eval_out_of_distribution` | see the table above |
| `n`, `offset`, `seed` | all, 0, none | subset (`seed` shuffles before `offset`/`n`) |
| `data_dir` | `$ALFWORLD_DATA` or `~/.cache/alfworld` | data root |
| `config` | none | an ALFWorld `base_config.yaml`; uses its `dataset.*_path` and `env.task_types` |
| `task_types` | all | ids 1-6 or names (`pick_and_place_simple`, `look_at_obj_in_light`, `pick_clean_then_place_in_recep`, `pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`, `pick_two_obj_and_place`) |
| `max_episode_steps` | 50 | TextWorld step budget; keep it >= `harness_defaults.max_steps` |
| `oracle` | `auto` | gold next action: `auto` (planner walkthrough while on it, then the hand-coded expert), `walkthrough`, `handcoded`, or `null` |
| `exclude_commands` | `[help, inventory]` | commands removed from the admissible list (duplicates are always removed) |

## 4. What the environment does

* **Per-task determinism.** `AlfredTWEnv.init_env()` builds *one* env that cycles through all
  games in `os.walk` order, which is filesystem-dependent. We mirror
  `AlfredTWEnv.collect_game_files()` (same filters), **sort** the game files, and build one
  single-game env per task the way `init_env()` does for eval splits:
  `textworld.gym.register_games([game], EnvInfos(won=True, admissible_commands=True,
  extras=["gamefile"]), batch_size=1, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])`.
  No domain randomisation.
* **task_id**: `alfworld/valid_unseen/<task-dir>/<trial>`, e.g.
  `alfworld/valid_unseen/look_at_obj_in_light-AlarmClock-None-DeskLamp-308/trial_T20190908_222917_366542`.
* **reset()** returns the room description. `instruction` is the goal sentence
  (`Your task is to: examine the alarmclock with the desklamp.`); the TextWorld welcome banner is
  stripped.
* **admissible_actions()** is `infos["admissible_commands"]` minus `exclude_commands`, de-duplicated,
  in TextWorld's (sorted) order.
* **step("act", {"action": a})** sends the command verbatim. Invalid commands get TextWorld's
  `Nothing happens.` The episode ends on `won` or when the step budget runs out.
* **evaluate()** returns `{"success": won, "score": 1.0|0.0, "steps", "task_type", "expert_steps"}`
  and releases the TextWorld env. The harness never calls `close()`.
* **oracle_action()** gives decision-level metrics and the simulated System-1 backends a gold
  action at *every* state. The gold is the planner walkthrough while the agent follows it, and
  ALFWorld's hand-coded expert (`AlfredExpert(handcoded)` wrapper, which re-plans from the current
  state) after the agent deviates. Many states have several equally good actions, so decision
  accuracy against this single gold action is a **lower bound**. The hand-coded expert is
  imperfect: it solved 9/11 sampled games alone and fails some `pick_two` tasks. With `oracle=auto`
  and an oracle S1 in `action: replace`, all 134 games are solved in ~5.9 steps on average.
* **Concurrency.** TextWorld isn't thread-safe: one module-level tatsu PDDL parser is shared, and
  `register_games` mutates a global registry. Concurrent resets corrupted parses. All TextWorld
  calls therefore go through one process-wide lock; LLM/S1 calls still run in parallel across the
  runner's `workers`. TextWorld costs ~1-3 s per episode (mostly parsing at reset), which is small
  next to LLM latency.
* **Memory.** TextWorld's `PddlEnv` loads a private ~32 MB copy of `libdownward.so` per game and
  never unloads it. Unloading it ourselves segfaulted in threaded runs, so RSS grows by about
  5 MB per episode (~0.75 GB peak for all 134 games in one process).

## 5. Cost estimate (per arm, 134 games; verify prices in `configs/prices/models.yaml`)

The harness sends the full episode history each step. The oracle policy needs ~6 steps per game,
LLM agents typically take 10-30, and failures run to `max_steps: 50`. Prompt size is about
`600 + 80*t` tokens at step `t`: the instruction, the admissible list (~20-60 commands), and
~80 tokens per past step.

| arm | model calls | input tokens / arm (rough) | cost / arm |
|---|---|---|---|
| `llm_react` (gpt-4.1-mini, $0.40/M in) | ~25 per episode | 134 x ~60k = ~8M | ~$3-4 (less with prompt caching) |
| same with a frontier model ($3/M in) | | ~8M | ~$25-30 |
| `jev_action_replace` (Jev, $0.042/M in) | 1 Choice per step (<= 255 options, so no tournament) | 134 x 25 x ~3k = ~10M | ~$0.4 |
| `jev_action_cascade` / `assist` | S1 every step + LLM on deferrals | between the two | |

Wall clock with `workers: 8` is ~15-30 min per LLM arm (1-2 s per LLM step) and a few minutes
per Jev-only arm.

## 6. Caveats for comparisons with the literature

* ReAct/Reflexion use 2-shot, task-type-specific prompts. Our `ACTION_SYSTEM` prompt is
  zero-shot, so absolute success rates are not directly comparable. Compare arms within this harness.
* We filter `help` and `inventory` by default, as many LLM-agent ALFWorld setups do.
  `exclude_commands: []` restores the raw list.
* `look` right after reset returns `You are in the middle of a room. Looking quickly around you,
  you see nothing.` This is ALFWorld behaviour, not a bug.

## 7. Pilot run (real APIs, small n)

`configs/pilot/alfworld_gemini.yaml` runs 5 OOD games (`stratify: true`, one per task type except
`pick_two`) with three arms: a ReAct LLM (Gemini 3.5 Flash), System-1 `action: replace`, and
`action: assist, k: 3`. System-1 is the Gemini 3.5 Flash-Lite stand-in, not Jev.

```bash
jevbench run configs/pilot/alfworld_gemini.yaml --out runs/pilot && jevbench report runs/pilot/pilot_alfworld_gemini
```

Findings (numbers in [`PILOT_RESULTS.md`](PILOT_RESULTS.md)):

* The data route in §2, the stratified subset, and the TextWorld lock worked unchanged with real models
  at `workers: 4`: 0 errors in 15 episodes.
* The ReAct LLM solved 5/5. The stand-in System-1 is a poor zero-shot action policy:
  * `replace` solved 0/5 and hit `max_steps` every time.
  * `assist` solved 3/5, with 2.4× the LLM steps.
  * It agreed with the expert on 16–19% of steps, which is a lower bound (§4). It confused objects
    (peppershaker for saltshaker) and was highly overconfident: 54% of its `replace` decisions had
    confidence ≥ 0.99.
* The generative stand-in is not cheap here. A verbalised distribution over 25–40 admissible commands
  costs ~240 output tokens per step, about $0.00085 and 1 s per step, against ~$0.0012 per LLM step.
  Because its episodes run 3× longer (40 vs 12 steps), a `replace` episode costs 2.4× the ReAct
  episode. Jev's per-step cost (input tokens only) should be measured before drawing cost conclusions.
* Cascade was not piloted. Given the stand-in's calibration, a cascade at τ ≥ 0.9 would accept most
  decisions and behave like `replace`.

## 8. L3 smoke (2026-10-01): regression check after the harness changes

This run re-checks the suite after three harness changes: the new `shadow` mode, gold labels for
guards, and the memory-gate token budget. It used one OOD game (`pick_and_place_simple`, SaltShaker),
`workers: 3`, the pilot config, and an arm list overridden on the command line. Outputs are in
`runs/dev/l3_smoke/alfworld_smoke/`. Wall clock was 53 s for all three arms.

```bash
jevbench run configs/pilot/alfworld_gemini.yaml --out runs/dev/l3_smoke -o name=alfworld_smoke -o suite.n=1 \
  -o workers=3 -o 'arms=[{name: llm_react}, {name: s1_action_shadow, action: {mode: shadow}}, {name: s1_action_assist_k3, action: {mode: assist, k: 3}}]'
jevbench report runs/dev/l3_smoke/alfworld_smoke
```

| arm | success | $/episode | steps | LLM calls | S1 calls | S1 vs expert | errors |
|---|---|---|---|---|---|---|---|
| llm_react | 1/1 | 0.0049 | 5 | 5 | – | – | 0 |
| s1_action_shadow | 1/1 | 0.0049 (+0.0042 shadow overhead) | 5 | 5 | 5 (shadow) | 3/5 | 0 |
| s1_action_assist_k3 | 1/1 | 0.0334 (S1 0.0160) | 19 | 19 | 19 | 6/19 | 0 |

- **No regression.**
  - The shadow arm reproduced the `llm_react` trajectory step for step: same 5 steps, same cost.
  - Its S1 calls are tagged and kept out of the episode cost. `usage["shadow"]` holds $0.0042, and
    `usage["total_with_shadow"]` holds both.
  - The decision tables show `action/shadow` with gold labels: s1_acc 0.6, defer_rate 1.0.
- The stand-in's weakness as an action policy is unchanged (§7): `assist` took 19 steps for a 5-step game.
- Full scale (E6, 134 games): `jevbench run configs/experiments/alfworld_action_only.yaml --out runs`, or the
  pilot config with `-o suite.n=134 -o suite.stratify=false`. Pilot rates are about $0.005–0.015 per episode
  for the ReAct arm and $0.03–0.07 for the S1 arms with the stand-in.
