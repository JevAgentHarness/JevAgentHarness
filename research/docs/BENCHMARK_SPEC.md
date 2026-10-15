# JevAgentBench — benchmark specification

**Version 1.0 (2026-09-30).** This is the normative definition of the suites, the configuration space,
every metric, the score card and the statistical protocol. Code is the source of truth for the numbers:
`src/jevharness/bench/metrics.py` (metrics), `src/jevharness/bench/scorecard.py` (score card),
`src/jevharness/bench/suites/` (suites), `src/jevharness/harness/config.py` (configuration space).
Every metric below has a hand-computed unit test in `tests/test_metrics.py`. Plan of record:
[`ROADMAP.md`](ROADMAP.md); operator guide: [`HANDOFF.md`](HANDOFF.md).

Contents: [1 Levels and suites](#1-levels-and-suites) · [2 JevGym](#2-jevgym-l2) ·
[3 WHERE × HOW](#3-configuration-space-where--how) · [4 Accounting](#4-accounting) ·
[5 Metric definitions](#5-metric-definitions) · [6 Score card](#6-the-score-card) ·
[7 Headline metrics](#7-headline-metrics) · [8 Statistical protocol](#8-statistical-protocol) ·
[9 Reporting rules](#9-reporting-rules) · [10 Extending the benchmark](#10-extending-the-benchmark) ·
[11 References](#11-references)

---

## 1. Levels and suites

JevAgentBench evaluates a *configuration* = (decision point, integration mode, System-1 backend, System-2
LLM, harness) at three levels that share one typed-question interface (`Choice` / `Noul` / `Score`), one
trajectory format (`harness/trajectory.py`) and one cost ledger (`accounting.py`).

| Level | Suite | Size (full scale) | Decision points | Unit of analysis | Code / config |
|-------|-------|-------------------|-----------------|------------------|---------------|
| **L1** static decisions | BFCL v4, 6 categories (`simple_python, multiple, live_simple, live_multiple, irrelevance, live_irrelevance`), pinned commit `6ea5797` | ≤ 200 items / category | tool selection (+ *none*), bounded / copyable arguments | one decision with a gold label | `bench/suites/static.py`, `configs/decision/bfcl_when2call.yaml` |
| L1 | When2Call test split, pinned commit `ecc8d42` | 3,652 (1,000 in pilots) | call / ask / cannot-answer / answer-directly | one decision | same |
| **L2** diagnostic world | JevGym (§2) | 200 tasks × {`random`, `near`} distractors, strict; scaling grid distractors {10, 30, 60, 120} × noise {0, 20, 50} | all 8 (oracle label at every state) | episode; every decision scored vs. gold | `bench/suites/jevgym.py`, `configs/experiments/jevgym_*.yaml` |
| **L3** real harnesses | τ²-bench airline + retail | 50 + 114 tasks × 3 trials | tool, args, policy guard, routing | episode (task = cluster of trials) | `bench/suites/tau2.py`, `docs/TAU2.md` |
| L3 | ALFWorld out-of-distribution | 134 games | action-only control, reflection | episode | `bench/suites/alfworld.py`, `docs/ALFWORLD.md` |
| L3 | LoCoMo (10 conversations, 1,540 QA) + LongMemEval_s (100 Q) | as listed | memory write / read | question | `bench/suites/memory_qa.py`, `docs/MEMORY.md` |
| L3 | SWE-bench Verified subset via mini-SWE-agent | 100 instances | command guard, submit verification, routing | instance | `adapters/mini_swe/` |
| L3 | Terminal-Bench 2.0 via Harbor + pi | 89 tasks | tool gate, active tools, stop, routing | task | `adapters/pi/` |

L1 compares System-1 backends head-to-head with an LLM's *native function calling* on identical decisions
(`jevbench decide`). L2 and L3 run whole episodes (`jevbench run`, or the suite's own orchestrator for τ²,
memory, SWE-bench and Terminal-Bench) and produce one `<arm>.jsonl` of trajectories per configuration; the
score card (§6) is computed from such a run directory.

## 2. JevGym (L2)

A deterministic customer-operations world (12 customers, orders with statuses
`pending/shipped/delivered/cancelled`, inventory of 10 items) with an **exact oracle** at every state:
`oracle_action()` returns the first step of the canonical gold plan not executed yet, `oracle_done()` whether
only `finish` remains. The oracle is used for decision-level gold labels and simulated backends only — it
is never shown to the agent (`Question.meta` never goes on the wire). Plan steps count as done under tolerant
argument matching (strip, case-insensitive, amounts ±0.01), and `get_orders(uid, "all")` satisfies a
status-filtered read (META_GOLD_SPEC §2.1). The meta-decision golds (verdicts, `finish_ok`, `world_complete`,
evidence atoms, acceptable tool sets, memory golds) are computed on independent copies of the env
(`oracle_snapshot`) after System-1 answered — see "Meta-decision golds" at the end of this document.

**Task families.** v1 (the default round-robin, unchanged since the pilots):

| Family | Gold plan | Decision content |
|--------|-----------|------------------|
| `cancel` | `find_user → get_orders(pending) → cancel_order(id, customer_request) → finish` | ID copying, enum argument |
| `refund` | `find_user → get_orders(delivered) → refund(id, amount, store_credit) → finish` | "most recent" = highest id, numeric copy |
| `stock` | `check_stock(item) → finish(count)` | enum argument, numeric answer |
| `ship` | `find_user → get_orders(pending) → ship_order(id, speed) → finish` | conditional logic (gold tier → express) |
| `total` | `find_user → get_orders(delivered) → finish(sum)` | arithmetic over an observation |

v2 families (`families: v2 | all`, or `family: clarify | multi`):

| Family | Gold plan | Decision point stressed |
|--------|-----------|-------------------------|
| `clarify` (ask / abstain) | information needed to act is missing or ambiguous; the right action is `ask_user(missing=…)` **before** any state-changing call. Variants (balanced over seeds): `order` — the customer has 2–3 pending orders and asks to "cancel their pending order" (`find → get_orders → ask_user(order_id) → cancel`); `method` — refund without a refund method (`ask_user(refund_method) → …`); `speed` — ship without a speed policy (`ask_user(shipping_speed) → …`); `name` — the customer never said who they are (`ask_user(customer_name) → …`). The simulated customer answers deterministically. | ask vs. act (When2Call-style) inside an episode; success requires the right question before acting — a lucky guess **fails** |
| `multi` (long horizon) | `multi_customers` (default 3, 2–12) customers, each with a `cancel` or `ship` request: `3k` tool calls, then `finish(total amount of the processed orders)`. Amounts appear only in the earlier `get_orders` observations. | horizon (10+ steps), memory write/read, stop |

When a v2 family is in play every task offers `ask_user` (so its presence does not reveal the family) and
the first observation adds the policy "never guess — ask with `ask_user` if a detail is missing or
ambiguous". An unnecessary question in a v1 task is harmless but counted (`n_asks`).

**Distractor styles** (`distractor_style`):

| Style | Distractors |
|-------|-------------|
| `random` (default) | unrelated tools with one `query` argument (`get_weather`, `convert_currency`, …) |
| `near` | 31 near-duplicates of the core tools with plausible descriptions and typed parameters, interleaved over the tool they shadow: `find_customer_by_email`, `find_user_by_phone`, `lookup_employee`, `search_users` (≈ `find_user`); `get_order`, `get_order_status`, `list_orders_by_item`, `get_orders_archive`, `get_orders_summary` (≈ `get_orders`); `cancel_subscription`, `cancel_order_item`, `cancel_shipment`, `void_payment` (≈ `cancel_order`); `refund_shipping_fee`, `issue_store_credit`, `partial_refund`, `reverse_chargeback` (≈ `refund`); `check_stock_warehouse`, `check_stock_incoming`, `get_item_price`, `reserve_stock`, `restock_item` (≈ `check_stock`); `ship_order_international`, `ship_replacement`, `upgrade_shipping`, `get_shipping_quote`, `schedule_pickup` (≈ `ship_order`); `send_email`, `notify_warehouse` (≈ `send_message`); `request_callback`, `escalate_to_supervisor` (≈ `ask_user`). They are functional: read-only ones return plausible, sometimes tempting observations (e.g. one warehouse's stock); state-changing ones apply a **side effect** that fails the episode. Beyond 31, random distractors fill up. |
| `mixed` | ⌈n/2⌉ near-duplicates + ⌊n/2⌋ random |

**Other factors.** `n_distractors` (tool-set size = 7 core [+ `ask_user`] + distractors + `finish`),
`noise_lines` (heartbeat lines appended to observations), `strict` (irreversible world: a state-changing call
that deviates from every gold call, precedes the required question, or is a state-changing near-duplicate
ends the episode as a failure — large V in Prop. 1).

**Suite options** (`load_suite("jevgym", …)` / YAML `suite:`): `n` (50), `seed0` (0), `family`, `families`
(`v1` default | `v2` | `all` | list), `n_distractors` (10), `distractor_style` (`random`), `noise_lines` (0),
`strict` (false), `ask_tool` (auto), `multi_customers` (3). Unknown options raise an error.
Task ids: `jevgym-<family>-<seed>-d<n>[-near|-mixed][-strict]`; v1 defaults reproduce the pilot worlds
byte-for-byte. **Success** is judged on the final world state (effects exactly equal to the gold effects, no
side effects) and, for `stock`, `total` and `multi`, on the final answer containing the gold number.

## 3. Configuration space (WHERE × HOW)

Eight decision points (`harness/config.py::DP`) × integration modes (`Mode`); `VALID_MODES` lists the
implemented cells: **15 System-1 cells** plus 4 non-System-1 references (27 modes including `off`).

| Decision point | Question | `replace` | `cascade` | `assist` | `guard` | `trigger` | references |
|----------------|----------|:-:|:-:|:-:|:-:|:-:|------------|
| `tool_select` | Choice over tools + `finish` | ✓ | ✓ | ✓ | ✓ | | `off` |
| `tool_params` | Choice over bounded values / copied spans + OTHER | ✓ | ✓ | | | | `off` |
| `memory_write` | Noul "needed later?" | | | | | ✓ | `off`, `always`, `llm` |
| `memory_read` | Choice over stored items (tournament) | | | ✓ | | | `off`, `always` |
| `reflection` | Noul "stuck / failed?" | | | | | ✓ | `off`, `always` |
| `stop` | Noul "task complete?" | | | | ✓ | ✓ | `off` |
| `routing` | Noul "needs the strong model?" | | | | | ✓ | `off` |
| `action` (action-only envs) | Choice over admissible actions | ✓ | ✓ | ✓ | | | `off` |

Parameters: threshold `tau` (cascade / guard / trigger; default 0.8), width `k` (assist; default 3),
`guard_max_retries` (1). With System-1 answer distribution $p_f$ and confidence $c=\max_a p_f(a)$:
replace executes $\arg\max_a p_f(a)$; cascade executes it iff $c \ge \tau$ and otherwise asks the LLM;
assist shows the LLM only the top-$k$ options; guard lets the LLM propose and accepts iff
$p_f(\text{yes}) \ge \tau$ (else one retry with a verifier note); trigger invokes the expensive module
(reflection, memory, strong model, early stop) iff $p_f(\text{yes}) \ge \tau$.

**References.** `llm_only` (every decision point `off`) is the *strong* reference and the default baseline
of every comparison; the *weak* reference is the System-1-only configuration of the family under study
(e.g. `ts_params_replace`: tool + arguments by System-1). Non-System-1 meta references: `always`
(reflect every step / store everything) and `llm` (the LLM makes the meta-decision).

**Arm naming.** One arm = one `HarnessConfig` (+ optional `s1:` / `llm:` binding). Sweeps expand to
`<base>_<param><value>` (e.g. `ts_cascade_tau0.9`); the score card groups arms by that prefix.

## 4. Accounting

Every System-1 and System-2 call goes through a `Ledger` (`accounting.py`) with component (`s1`, `s2`,
`embed`, …), decision point, model, input / output / cached tokens, cost and latency. Per episode:

- **cost** $c$ = sum of call costs at provider list prices from `configs/prices/models.yaml` (OpenRouter's
  reported `usage.cost` for Jev when present); self-hosted models at amortised GPU cost. Models without a
  price count as \$0 and are listed as `unpriced_models` (the score card prints a note) — fix before reporting.
- **latency** $\ell$ = sum of call latencies = critical path of the (sequential) loop, excluding tool
  execution; wall-clock time is logged separately (`wall_ms`).
- **LLM calls** = `s2` calls (all System-2 models); **strong calls** = `s2` calls to the episode's main LLM
  (`traj.llm`) — a routed call to the small model is an LLM call but not a strong call.

## 5. Metric definitions

Notation. Tasks $t = 1..T$; task $t$ has $n_t \ge 1$ trials $j$ (τ²: `task_id#j`; otherwise $n_t = 1$) with
success $y_{tj} \in \{0,1\}$, cost $c_{tj}$, latency $\ell_{tj}$. Per-task means $\bar y_t, \bar c_t, \bar\ell_t$.
Configuration $a$, baseline $b$ (= `llm_only`), weak reference $w$. Bars over configuration symbols are task
means: $S_a = \frac1T\sum_t \bar y_{t}^{a}$, $C_a = \frac1T\sum_t \bar c_t^{a}$, $L_a$ likewise. The **task is the
unit of analysis and of resampling** everywhere.

### 5.1 Effectiveness and efficiency

| Metric | Definition | Source | Function |
|--------|-----------|--------|----------|
| Success rate | $S_a$ with percentile-bootstrap 95% CI over tasks | Efron & Tibshirani (1993) | `bootstrap_ci` |
| Cost / latency per episode | $C_a$, $L_a$; always reported next to $S_a$ | Kapoor et al. (2025) | `episode_table` |
| **Cost-of-pass** | $\mathrm{CoP}_a = C_a / S_a$ (USD per success; $\infty$ if $S_a = 0$) — the expected cost of one correct outcome. Plug-in estimator of Erol et al.'s $C_m(p)/R_m(p)$ at dataset level, as used for agents by Wang et al. (2025) | Erol et al. (2025); Wang et al. (2025) | `cost_of_pass` |
| **Latency-of-pass** | $\mathrm{LoP}_a = L_a / S_a$ (seconds per success) — the same construction with latency | this work, after Erol et al. | `latency_of_pass` |
| Frontier cost-of-pass | $\frac1T \sum_t \min_{a} \bar c^{a}_t / \bar y^{a}_t$ (optionally $\min$ with a human-expert cost); $\infty$ if some task is solved by no configuration | Erol et al. (2025) | `frontier_cost_of_pass` |
| Effectiveness under budget | $\mathrm{EuB}_a(B) = \frac1B\int_0^B R_a(\beta)\,d\beta = \frac{1}{NB}\sum_i y_i \max(0, B - c_i)$, $R_a(\beta)$ = fraction of episodes solved within budget $\beta$; default $B = 2\,C_b$ | Fan et al. (2025, SWE-Effi) | `effectiveness_under_budget` |
| Success per dollar | $\sum y / \sum c$ | — | `episode_table` |

### 5.2 Comparison with the baseline

| Metric | Definition | Function |
|--------|-----------|----------|
| $\Delta$success | $\Delta_a = \frac1{T'}\sum_{t} (\bar y^{a}_t - \bar y^{b}_t)$ over the $T'$ tasks both ran; paired-bootstrap 95% CI; one-sided $p_{\text{worse}} = P^*(\Delta^* \ge 0)$ (small ⇒ $a$ worse) and $p_{\text{better}} = P^*(\Delta^* \le 0)$; lower bound $\mathrm{LCB}_{1-\alpha}$ = $\alpha$-quantile of $\Delta^*$ | `paired_bootstrap_diff` |
| McNemar | discordant counts $n_{01}$ (#episodes $b$ solves, $a$ fails), $n_{10}$; exact binomial $p = \min(1, 2\,P[X \le \min(n_{01}, n_{10})])$, $X\sim\mathrm{Bin}(n_{01}+n_{10}, \tfrac12)$ if $n_{01}+n_{10} < 25$, else $\chi^2 = (\lvert n_{01}-n_{10}\rvert - 1)^2/(n_{01}+n_{10})$ with 1 d.o.f. (McNemar 1947; Dietterich 1998) | `mcnemar` |
| Cost / latency savings | $1 - C_a / C_b$, $1 - L_a / L_b$ (paired tasks) | score card |
| LLM-call / strong-call ratio | $\overline{\#s2}_a / \overline{\#s2}_b$; $\overline{\#\text{strong}}_a / \overline{\#\text{strong}}_b$ | score card |
| **Iso-quality cost savings** | $\max_{a \in Q} (1 - C_a / C_b)$ over the set $Q$ of configurations **not significantly worse** than $b$: `not_worse` (default) — $p_{\text{worse}} \ge \alpha$ (uncorrected, i.e. conservative for the savings claim); `noninferior` — $\mathrm{LCB}_{1-\alpha}(\Delta_a) > -m$ for a pre-registered margin $m$ (3 pp on JevGym/ALFWorld, 5 pp on τ²: `docs/PREREGISTRATION.md`; $m = 0$ means superiority). CI: percentile bootstrap of $1 - \sum \bar c^a / \sum \bar c^b$ over tasks for the selected arm. In the spirit of cost-controlled evaluation (Kapoor et al. 2025) and cost-at-matched-quality in routing (Ong et al. 2025) | `iso_quality_savings` |
| **Quality at iso-cost** | $\max\{S_a : C_a \le C_b\}$ (budget $C_b$ by default) with paired test vs. $b$, plus $\tilde Q(C_b)$, the value of the non-decreasing convex hull (5.3) at that budget (success reachable by randomised mixing) | `quality_at_iso_cost` |

### 5.3 Routing-style metrics (a family = one threshold sweep)

Let $x_a \in [0,1]$ be the strong-call ratio of $a$ (capped at 1) and $\mathrm{PGR}_a = (S_a - S_w)/(S_b - S_w)$.

| Metric | Definition | Source | Function |
|--------|-----------|--------|----------|
| PGR | performance gap recovered, as above ($>1$ possible when $a$ beats the LLM) | Ong et al. (2025) | `pgr` |
| **APGR** | $\frac{1}{x_{\max}-x_{\min}}\int \mathrm{PGR}(x)\,dx$ by the trapezoid rule over the family sorted by $x$, with anchors $(x_w, 0)$ and $(1, 1)$ (the τ → 0 / τ → 1 limits of a cascade); identical to RouteLLM's $(\mathrm{AUC}_{\text{router}} - \mathrm{AUC}_{\text{weak}})/(\mathrm{AUC}_{\text{strong}} - \mathrm{AUC}_{\text{weak}})$; 0.5 = no better than interpolating the references | Ong et al. (2025) | `apgr` |
| **CPT(x%)** | smallest strong-call ratio at which the piecewise-linear PGR curve reaches $x\%$ (CPT(50%), CPT(80%)) | Ong et al. (2025) | `cpt` |
| Non-decreasing convex hull | $\tilde Q(c)$: upper concave envelope of the family's $(C, S)$ points truncated at its maximum (any point on a segment is reachable by randomly mixing two configurations per episode; more budget never lowers quality) | Hu et al. (2024) | `nondecreasing_convex_hull` |
| **AIQ** | $\frac{1}{c_{hi}-c_{lo}}\int_{c_{lo}}^{c_{hi}} \tilde Q(c)\,dc$ over the linear cost range between $w$ and $b$; $\tilde Q$ is flat beyond the most expensive point and 0 below the cheapest; the family includes the two references as anchors | Hu et al. (2024) | `aiq` |
| Zero-router AIQ, ΔAIQ | AIQ of the hull of $\{w, b\}$ alone (random per-episode mixing); ΔAIQ = AIQ − AIQ$_0$ ≥ 0 is the value added by the System-1 signal | Hu et al. (2024) | score card |

### 5.4 Multi-objective (cost ↓, success ↑)

| Metric | Definition | Source | Function |
|--------|-----------|--------|----------|
| Dominance | $a \succ a'$ iff $C_a \le C_{a'}$, $S_a \ge S_{a'}$, one strict; Pareto front = non-dominated arms; counts "dominates" (SPEA strength) / "dominated by" | Zitzler & Thiele (1999); Kapoor et al. (2025) | `dominates`, `dominance_counts`, `pareto_front` |
| **Hypervolume** | area of $\bigcup_a [C_a, c_{\text{ref}}] \times [s_{\text{ref}}, S_a]$; default cost axis $\log_{10}$ USD, reference $(10^{0.1} \max_a C_a,\ 0)$; normalised by the box $[\min_a C_a, c_{\text{ref}}] \times [s_{\text{ref}}, 1]$. Reported for all arms and for the baseline alone | Zitzler & Thiele (1998, 1999) | `hypervolume_2d` |
| Exclusive contribution | $\mathrm{HV}(P) - \mathrm{HV}(P \setminus \{a\})$ (0 for dominated arms) | Zitzler & Thiele | `hypervolume_contributions` |

### 5.5 Reliability over repeated trials

| Metric | Definition | Source | Function |
|--------|-----------|--------|----------|
| **pass^k** | $\frac1T \sum_t \binom{c_t}{k} / \binom{n_t}{k}$ ($c_t$ successes in $n_t$ trials): probability that **all** $k$ i.i.d. trials succeed; pass^1 = success | Yao et al. (2025, τ-bench) | `pass_hat_k`, `pass_hat_k_curve` |
| pass@k (contrast) | $\frac1T \sum_t 1 - \binom{n_t - c_t}{k} / \binom{n_t}{k}$ | Chen et al. (2021) | `pass_at_k` |

### 5.6 Decision level (System-1 decisions with an oracle label)

A decision is **delegated** when System-1's answer is executed without the LLM: object-level decision point
(`tool_select`, `tool_params`, `action`) in `replace`, or in `cascade` with $c \ge \tau$.

| Metric | Definition | Function |
|--------|-----------|----------|
| System-1 accuracy | $\Pr[\hat a_{S1} = a^{\text{gold}}]$ over decisions with a gold label (normalised strings) | `decision_table` |
| Deferral rate / accepted accuracy | fraction deferred to the LLM; accuracy on accepted decisions (selective accuracy at the operating τ) | `decision_table` |
| **Delegation ratio** | delegated / all object-level decisions (LLM-made `off`, `assist` and guarded decisions in the denominator; a guarded step counts once) | `delegation_ratio` |
| LLM-free steps | fraction of steps with zero System-2 calls | `llm_free_step_ratio` |
| **Harmful-delegation rate** | #(delegated ∧ wrong ∧ episode failed) / #delegated; also the wrong-delegation rate (1 − accepted accuracy), the fraction of episodes with a harmful delegation, and the fraction of failed episodes containing one (an upper bound on failures attributable to delegation) | `harmful_delegation` |
| **Guard FAR / FRR** | gold = "the LLM's proposal is right": the tool guard's proposal is *progress* (verdict on a copy of the env), the stop guard's finish succeeds now with its answer (`finish_ok`). FAR = $\Pr[\text{accept} \mid \text{proposal wrong}]$ (1 − FAR = catch rate), FRR = $\Pr[\text{reject} \mid \text{proposal right}]$ (each costs an extra LLM call); per decision point and pooled, with `by_kind` (verdict / finish kind), `far_fatal`, and a `gold="canonical"` variant (legacy canonical verdict). Each **proposal** counts once: a stop-guard resubmission of a rejected answer is a retry (`n_retries`), not a new proposal (`first_opportunities`) | `guard_error_rates` |
| **Rescue rate** | #(wrong proposals rejected by the guard in an episode that then succeeded) / #wrong proposals; also rescue-given-catch and `by_kind` | `rescue_rate` |
| **Tool-level accuracy** | `tool_select` decisions are right when the tool is in the information-feasible acceptable tool set AT(s) (primary; `s1_acc`); the canonical next tool is the secondary column (`s1_acc_canonical`) | `decision_table`, `is_correct` |
| **Recall@k (C3)** | share of tool decisions whose top-k of System-1's full ranking (`DecisionLog.s1_rank` for tournaments: finalists, then each round's eliminated options by score; else `s1_dist`) contains a tool of AT(s), k ∈ {1, 3, 5}; canonical secondary; records whose ranking is incomplete at k (older tournament logs) are undefined (`n_undefined@k`); `offered_recall` / `offered_k` for the set an assist arm offered (assist keeps max(2, k) tournament survivors per chunk) | `recall_at_k` |
| **Gate statistics (H2)** | per gate on its primary gold, on the shadow records of a **measurement arm** (every opportunity labelled): π, AUROC of the firing score, TPR / FPR, Youden J and accuracy at the operating τ, ECE; a guard fires (rejects) on p_yes < τ, so J = 1 − FAR − FRR; *estimable* = ≥ 30 positive and ≥ 30 negative golds. Live trigger / guard arms get a separate *consistency check* (`live_gates`: AUROC on the labels observed at executed actions, labelled share, realised fire rate per opportunity); their reflection / routing labels are selective (a fired step is unobserved), so J / accuracy / estimability are not reported and they never enter H2 | `gate_stats` |
| **Probe FAR / FRR (C6)** | wrong-proposal probes of the measurement arms: FAR_model, FAR_nat, FRR, FAR per edit operator, strata by source / kind / D bucket, the style check, the kind-reweighted FAR, FAR(τ) / FRR(τ) curves; natural shadow finishes and probes are thresholded at the arm's `measure.probes.tau_ref`; **task-clustered percentile-bootstrap CIs** (`far_ci` / `frr_ci`; undefined with one task), Wilson next to them (`*_ci_wilson`); the representativeness **TOST** of FAR_nat − FAR_model (±0.10, 90% task-bootstrap CI, `representativeness`) | `probe_error_rates`, `tost_far_nat_model` |
| **Gate TOST (H2)** | equivalence of a System-1 gate and its matched random gate within ±2 pt of success: 90% paired task-bootstrap CI inside the margin; with < 5 discordant tasks (degenerate bootstrap) the one-sided Wilson bound of the discordance rate must lie below the margin | `tost_paired`; `scorecard --pair` (`tost_2pt`), `meta_gold h2` (`tost`) |
| **Counterfactual efficacy** | e = gain of the intervention at needed states, h = loss at non-needed states, both net of a placebo resample (NaN, `net = False`, when a placebo term is missing; the raw rates are `e_raw` / `h_raw`); δ = placebo disagreement (label noise floor) | `cf_efficacy` |
| **Memory read hits** | hit@k, MRR and none-accuracy of System-1 recalls against the stored items that hold an input of the next oracle step | `memory_read_hits` |
| **Fill-kind composition (H3)** | `tool_params` decisions per kind (bounded / copy / free) with delegation and harmful-delegation rates; free slots are logged as `generate` (LLM-made) | `param_kind_composition` |

### 5.7 Calibration and selective prediction

**`Choice` decisions** (tool, arguments, actions; the argmax is executed): confidence $c_i$ = top-label
probability, correctness $z_i = [\hat a_i = a_i^{\text{gold}}] \in \{0,1\}$, $N$ decisions.

**`Noul` decisions** (guard / trigger: stop, tool guard, …): the executed value is *thresholded*,
$[p_{\text{yes}} \ge \tau]$, while the logged `s1_conf` is $\max(p_{\text{yes}}, 1-p_{\text{yes}})$ — for
$\tau \ne 0.5$ the two refer to different events, and pairing them scores a calibrated forecast
($p_{\text{yes}} = 0.7$, gold *yes*, executed *no* at $\tau = 0.9$) as a confident error. They are therefore
calibrated as **probabilistic forecasts**: $c_i = p_{\text{yes},i}$ from the logged `s1_dist` and
$z_i = [\text{gold}_i = \text{yes}]$ in the ECE / Brier / AUROC formulas below; AURC ranks the argmax answer
$[p_{\text{yes}} \ge \tfrac12]$ by $\max(p_{\text{yes}}, 1-p_{\text{yes}})$ (`binary_calibration`; `decision_table`
rows carry `cal = p_yes`, the card a *yes/no* block). Accuracy is always that of the executed value. Logs
without `s1_dist` for `Noul` decisions (before 2026-09-30) get accuracy and FAR/FRR only.

| Metric | Definition | Source | Function |
|--------|-----------|--------|----------|
| **ECE** | $\sum_{m=1}^{10} \frac{\lvert B_m\rvert}{N}\,\lvert \mathrm{acc}(B_m) - \mathrm{conf}(B_m)\rvert$, 10 equal-width bins | Naeini et al. (2015); Guo et al. (2017) | `ece` |
| Adaptive ECE | as ECE with 10 equal-mass bins (top-label; the binning of Nixon et al.'s ACE) | Nixon et al. (2019) | `ece_adaptive` |
| Brier | $\frac1N\sum_i (c_i - z_i)^2$ (binary / top-label Brier score) | Brier (1950) | `brier` |
| AUROC | $\Pr[c_i > c_j \mid z_i = 1, z_j = 0]$ (ties ½) — does confidence rank right above wrong? | Mann–Whitney | `auroc` |
| Risk–coverage | accept the $i$ most confident decisions: coverage $i/N$, selective risk $r(i)$ = error rate among them (ties resolved in expectation) | Geifman & El-Yaniv (2017) | `risk_coverage` |
| **AURC** | $\frac1N\sum_{i=1}^N r(i)$ | Geifman et al. (2019) | `aurc` |
| E-AURC | AURC − AURC of the oracle ranking (all correct first): the part due to ranking, not accuracy | Geifman et al. (2019) | `eaurc` |
| **Selective accuracy @ coverage** | $1 - r(\lceil \kappa N \rceil)$; the card reports $\kappa = 0.8$ — the accuracy a cascade gets when it keeps 80% of decisions | Geifman & El-Yaniv (2017) | `selective_accuracy` |
| Coverage @ accuracy | largest coverage whose selective accuracy $\ge$ target | Geifman & El-Yaniv (2017) | `coverage_at_accuracy` |

## 6. The score card

```bash
python -m jevharness.bench.scorecard RUN_DIR --baseline llm_only [--weak ts_params_replace] \
    [--sweep ts_cascade_tau] [--format md|latex|json] [--alpha 0.05] [--criterion not_worse|noninferior] \
    [--margin 0.03] [--n-boot 10000] [--budget USD] [--hv-ref COST,SUCCESS] [--linear-cost] [--out FILE]
```

Per arm: success with CI; Δsuccess, $p_{\text{worse}}$ (raw and Holm), McNemar $p$; cost, cost savings,
cost-of-pass, latency, latency savings, latency-of-pass; LLM-call and strong-call ratios; PGR; EuB; Pareto
membership, dominance counts, hypervolume contribution; delegation ratio, LLM-free steps; System-1 accuracy,
ECE, adaptive ECE, Brier, AUROC, AURC, E-AURC, selective accuracy @ 80%; harmful-delegation rate; guard
FAR / FRR, rescue rate; yes/no (guard / trigger) calibration — n, ECE, Brier, AUROC of P(yes) vs. the
binary gold; memory stored fraction when memory gates ran; pass^k when tasks have ≥ 2 trials. Headline lines: iso-quality cost savings (with CI
and the Δsuccess CI of the selected arm), quality at iso-cost, frontier cost-of-pass, hypervolume, and per
sweep APGR, CPT(50%), CPT(80%), AIQ vs. zero-router. Flags: `front` (Pareto), `worse` (significantly worse
after Holm), `iso-best` / `iso-q` (selected / qualifying for iso-quality savings), `baseline`, `weak`.

Sweeps are detected from arm names `<prefix><number>` (≥ 3 arms, or ≥ 2 with `--sweep`). Duplicate
records of one `task_id` are dropped (the last is kept) with a note. Runs whose experiment uses a simulated
LLM or System-1 are flagged **SIMULATED** in every format. `jevbench report RUN_DIR` appends a compact card;
`scripts/run_experiments.sh` writes `scorecard.md` next to every run. LaTeX output needs `booktabs`
($^\star$ Pareto front, $^\dagger$ significantly worse, **bold** = iso-quality arm).

## 7. Headline metrics

Per suite × System-2 model, the paper reports (in this order):

1. **Success with 95% CI, cost per episode and latency per episode** for every configuration (never
   success alone), and the cost–success Pareto front.
2. **Iso-quality cost savings** vs. LLM-only with its CI and the Δsuccess CI of the selected configuration
   (`not_worse` for exploration; `noninferior` with the pre-registered margin for claims).
3. **Cost-of-pass** (and latency-of-pass) — best configuration vs. LLM-only.
4. **APGR and CPT(80%)** of each cascade family (and ΔAIQ over the zero router).
5. **Hypervolume** of the design space vs. the LLM-only point.
6. Safety of delegation: **harmful-delegation rate**, guard **FAR / FRR**, and System-1 **ECE / AURC**.
7. pass^k (k = 1..3) on L3 suites with repeated trials.

## 8. Statistical protocol

- **Unit & pairing.** The task is the unit; all configurations run on the same tasks (same seeds), so
  comparisons are paired. Trials of a task are a cluster: per-task means are resampled.
- **Seeds.** JevGym evaluation seeds `seed0 = 200000 … 200000 + N − 1` (200000–200499 for the 500-task E2 suite, PREREGISTRATION A6.13; 100000–100499 is retired: exposed before the freeze, never run again;
  `PREREGISTRATION.md` A6.4, 2026-10-01); seeds 0–999 are development (the pilots, dev rounds and the CF validity
  pilot used 0–59); dev / calibration seeds 5000–5999 (5000–5149: the A6.5 calibration shadow arms that set the C1
  thresholds); training seeds ≥ 10000 (never overlapping). System-2 at temperature 0; simulated backends seeded. τ² user simulator:
  3 trials per task (`NUM_TRIALS=3`) → pass^k. BFCL / When2Call **test** splits are never used for training.
- **Confidence intervals.** Percentile bootstrap over tasks, $B = 10{,}000$ resamples for paper numbers
  (`--n-boot 10000`; 2,000 is the interactive default), seed 0, two-sided 95%.
- **Tests.** Primary: one-sided paired bootstrap of Δsuccess vs. LLM-only (Koehn 2004; Berg-Kirkpatrick et
  al. 2012) — a non-parametric, sampling-based test of the kind Dror et al. (2018) recommend when normality
  of the metric cannot be assumed; secondary for binary per-episode outcomes: McNemar (exact below 25
  discordant pairs; Dietterich 1998).
  Non-inferiority claims use the one-sided LCB with a margin fixed **before** the run: 3 pp on JevGym and ALFWorld, 5 pp on τ² (pass^1). For the pre-registered primaries the test is the Holm-adjusted bound, i.e. $p_{\text{NI}} = P^*(\Delta^* \le -m) \le \alpha_{\text{adj}}$ (`PREREGISTRATION.md` A6.1: the 95% LCB is descriptive; fewer than 5 discordant tasks → the exact binomial on losses; zero discordance → the exact upper bound $1 - 0.05^{1/n}$ on the loss rate; a failed NI test whose CI covers 0 is *inconclusive at this n*). The margins are sized for the pre-registered sample sizes (≈340 tasks for 3 pp at 5% discordance at unadjusted α; ≈650 at the Holm floor 0.05/11; achieved power and MDE per primary in A6.1).
- **Multiple comparisons.** Holm (1979) within each results table (family = all configurations vs.
  LLM-only on one suite × System-2 model; controls the family-wise error); Benjamini–Hochberg (1995) for
  exploratory families (τ grids, scaling sweeps, per-decision-point ablations). Raw and adjusted $p$ are
  both reported; iso-quality qualification uses the uncorrected test (conservative for a savings claim).
- **Precision guidance.** With $T = 200$ tasks the 95% CI half-width of a success rate is ≈ ±6.9 pp at 50%
  and ≈ ±4.2 pp at 90%; paired differences are tighter. Pilot sizes (≤ 20 tasks) only check the pipeline.
- **Calibration** metrics are computed on System-1 `Choice` decisions pooled over episodes of one
  configuration; thresholds / temperatures are fitted on dev seeds only (`decision/calibration.py`).

### Known limitations of the decision-level metrics

- **Memory gates.** Since A4 every offer logs one `memory_write` `DecisionLog` (every write mode, incl. the new
  `shadow`) and every System-1 recall one `memory_read` record. On JevGym the write gold `mem_need` is
  question-literal, but System-2 keeps every observation in context, so the success-side value of a memory write is
  about zero: JevGym memory cells are **negative controls** for H2 (predicted ΔU ≈ −cost); success-side memory
  claims come from the memory suites (`bench/suites/memory_qa.py`). `memory_read` golds are descriptive only.
- **Guard gold.** The tool-guard gold is the verdict of the proposal (tool **and** arguments) on a copy of the env:
  *progress* (the canonical cost-to-go drops) vs. detour / fatal. The legacy canonical verdict
  (`oracle_accepts`) stays in `gold_aux.canonical`; dev FAR / FRR under the canonical gold are not comparable with
  the main runs. Envs without `oracle_verdict` keep the legacy gold.
- **Canonical gold.** Tool-level accuracy, recall@k and the H1 pairs use the acceptable tool set AT(s), so an
  equally valid order (asking before listing orders in `clarify/order`, out-of-order `multi` jobs) is no longer a
  decision error. `tool_params` golds stay canonical-only: arguments are gold only when the executed tool is the
  oracle's next tool. D is the canonical policy's cost-to-go (an upper bound on the optimum), so residual oracle
  errors can move a label between detour and progress, never `finish_ok` or `world_complete` (both rest on
  `evaluate`); they can reach `need_fatal` only through the step-budget route (`gold_aux.fatal_kind = budget`: the
  canonical plan no longer fits the remaining steps, while `terminal` fatals are proven by the episode ending), whose
  share is reported with a sensitivity row that excludes it, and the implication audit of `meta_gold audit` (executed
  failing verdict ⇒ episode failed; `finish_ok` ⇒ success; violations must be 0) is reported per arm
  (`PREREGISTRATION.md` A6.7).
- **Doomed states.** States from which success is impossible within the step budget (D = ∞) get `gold = None`,
  `gold_status = "doomed"`, and are excluded from every rate (in non-strict worlds wrong writes make many states
  doomed).
- **Latency** excludes tool execution and harness overhead (sum of model-call latencies; `wall_ms` is
  logged separately).

## 9. Reporting rules

1. Numbers in the paper come only from `runs/` produced by committed configs; the score card of every run
   is regenerated from the JSONL (never edited by hand).
2. Success is always shown with cost, latency and LLM calls (and a CI).
3. Simulated runs (`provider: simulated`, `type: simulated|oracle|random`) are pipeline tests or clearly
   labelled simulation studies — never headline numbers; the card marks them.
4. Prices are verified on the run date; `unpriced_models` must be empty in reported runs.
5. Jev is pinned (`typesafe/jev-1.13` on OpenRouter / `jev-1.13.0` direct) and the served model id logged.
6. Report the operating threshold τ and k for every System-1 arm, and the weak / strong references of every
   APGR / CPT / AIQ number.
7. Gold labels never reach a real backend (`Question.meta` is not sent on the wire).

## 10. Extending the benchmark

**A System-1 backend.** Subclass `decision.base.DecisionBackend`, implement
`_decide(state, questions) -> DecisionResult` returning one calibrated `Answer` distribution per question
and a `Usage` (tokens, cost, latency); set `name`, `model`, `max_options` (tournaments split larger
`Choice`s) and `batched_questions`. Register it in `decision/__init__.py::make_backend`, add its price to
`configs/prices/models.yaml`, and never read `Question.meta` (gold) outside simulated backends. Trained or
external models can instead be served on the Jev wire format (`training/serve_jev.py`, or
`decision/server.py` for any backend) and used unchanged by every harness adapter as `{"type": "jev",
"base_url": …}`. Check it with `jevbench ping --s1 '<spec>'`, then on L1 (`jevbench decide`).

**A harness.** Use the host harness's public extension surface (hooks, extensions, subclassing; see
[`HARNESS_INTEGRATION.md`](HARNESS_INTEGRATION.md)), route every System-1 / System-2 call through a `Ledger`,
and emit one `Trajectory` JSONL per arm with a `DecisionLog` per decision (`dp`, `mode`, `options`,
`s1_value`, `s1_conf`, `deferred`, `s2_value`, `final_value`, and `gold` when the environment has an
oracle); trials of one task use `task_id#<k>`. The score card then applies unchanged.

**A suite.** Implement `bench.env.Env` (`reset`, `tools()` or `admissible_actions()`, `step`, `evaluate`
returning at least `success` and `score`), deterministic given its seed and unique `task_id`s; implement
`oracle_action()` / `oracle_done()` if an exact oracle exists (decision-level metrics and simulated
backends need it). Register it in `bench/suites/__init__.py::load_suite` (reject unknown options), and add
a test that `JevAgent(SimulatedLLM(error_rate=0), HarnessConfig())` solves 100% of it when an oracle exists.

**A metric.** Add a pure function to `bench/metrics.py` with its definition and source in the docstring, a
hand-computed test in `tests/test_metrics.py`, a row in §5, and (if it belongs in the card) a field in
`scorecard.compute_scorecard` and the renderers.

## 11. References

BibTeX keys refer to `paper/references.bib` (section *L. Metrics & evaluation methodology* unless noted).

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *JRSS B* 57(1):289–300. `benjamini1995fdr`
- Berg-Kirkpatrick, T., Burkett, D. & Klein, D. (2012). An empirical investigation of statistical significance in NLP. *EMNLP-CoNLL*, 995–1005. `bergkirkpatrick2012significance`
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1):1–3. `brier1950verification`
- Chen, M. et al. (2021). Evaluating large language models trained on code. arXiv:2107.03374. `chen2021codex`
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation* 10(7):1895–1923. `dietterich1998approximate`
- Dror, R., Baumer, G., Shlomov, S. & Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in natural language processing. *ACL*, 1383–1392. `dror2018hitchhiker`
- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. `efron1993bootstrap`
- Erol, M. H., El, B., Suzgun, M., Yuksekgonul, M. & Zou, J. (2025). Cost-of-Pass: an economic framework for evaluating language models. arXiv:2504.13359. `erol2025costofpass`
- Fan, Z. et al. (2025). SWE-Effi: re-evaluating software AI agent system effectiveness under resource constraints. arXiv:2509.09853. `fan2025sweeffi`
- Geifman, Y. & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*, 4878–4887. `geifman2017selective`
- Geifman, Y., Uziel, G. & El-Yaniv, R. (2019). Bias-reduced uncertainty estimation for deep neural classifiers. *ICLR*. arXiv:1805.08206. `geifman2019biasreduced`
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*. `guo2017calibration` (section K)
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2):65–70. `holm1979simple`
- Hu, Q. J. et al. (2024). RouterBench: a benchmark for multi-LLM routing system. arXiv:2403.12031. `hu2024routerbench` (section B)
- Kapoor, S., Stroebl, B., Siegel, Z. S., Nadgir, N. & Narayanan, A. (2025). AI agents that matter. *TMLR*. arXiv:2407.01502. `kapoor2025agentsthatmatter` (section F); HAL: `kapoor2026hal`
- Koehn, P. (2004). Statistical significance tests for machine translation evaluation. *EMNLP*, 388–395. `koehn2004significance`
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2):153–157. `mcnemar1947note`
- Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *AAAI*, 2901–2907. `naeini2015bbq`
- Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G. & Tran, D. (2019). Measuring calibration in deep learning. *CVPR Workshops*. `nixon2019measuring`
- Ong, I. et al. (2025). RouteLLM: learning to route LLMs with preference data. *ICLR*. arXiv:2406.18665. `ong2025routellm` (section B)
- Wang, N. et al. (2025). Efficient agents: building effective agents while reducing cost. arXiv:2508.02694. `wang2025efficientagents` (section F)
- Yao, S., Shinn, N., Razavi, P. & Narasimhan, K. (2025). τ-bench: a benchmark for tool-agent-user interaction in real-world domains. *ICLR*. arXiv:2406.12045. `yao2025taubench` (section G)
- Zitzler, E. & Thiele, L. (1998). Multiobjective optimization using evolutionary algorithms — a comparative case study. *PPSN V*, LNCS 1498, 292–301. `zitzler1998multiobjective`
- Zitzler, E. & Thiele, L. (1999). Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach. *IEEE Trans. Evolutionary Computation* 3(4):257–271. `zitzler1999spea`

## Shadow arms and threshold-theory tests

**Shadow mode** (`mode: shadow`; tool_select, tool_params, stop, action, reflection, routing) is a measurement
mode. The episode runs exactly as in the LLM-only harness: the default resolver decides, and nothing System-1 says
reaches the agent. At the same state System-1 is asked the typed question the corresponding cascade / guard /
trigger arm would ask, and the `DecisionLog` (`shadow: true`, `deferred: true`) records:

- System-1's answer (`s1_value` is the argmax, so `s1_conf` refers to it; `s1_dist`), its cost and latency
  (`s1_cost_usd`, `s1_latency_ms`);
- the LLM's own decision (`s2_value` = `final_value`), whether it is right (`s2_correct`, judged against the raw
  oracle value, so copyable arguments outside the candidate set are scored correctly), and the cost and latency of
  the LLM call(s) that made it (`s2_cost_usd`, `s2_latency_ms`);
- `gold`, wherever the env has an oracle.

At stop, the question is the guard one when the LLM calls `finish` (it shows the proposed answer) and the trigger one
otherwise; the gold label is `finish_ok` at a finish and `world_complete` elsewhere (the legacy `oracle_done` stays in
`gold_aux.canonical_done`). At tool arguments, System-1 is asked about every bounded / copy argument of the executed
call (never free text). `memory_write: shadow` asks the write gate at every offer, logs it and stores nothing.

Shadow System-1 calls go through the ledger tagged `meta.shadow = true`. They stay out of `usage.total`, `usage.s1`
and `usage.by_decision_point`, so every cost metric of a shadow arm equals that of `llm_only`. The overhead is
reported in `usage.shadow` (with `by_decision_point`), and `usage.total_with_shadow` is what the run actually spent.
A failing shadow call is fail-open: it is counted in `meta.shadow_s1_errors` and never errors the episode. The E2a
config runs `jev_shadow` (tool_select + tool_params + stop) by default, so the LLM-only condition also yields
calibration and q_L(c) data.

**Threshold-theory tests** use `python -m jevharness.bench.threshold_theory RUN_DIR --arm jev_shadow --dp tool_select
[--V 0.1 1 10] [--task-value W] [--lam-per-hour λ] [--format md|json|latex] [--out FILE.json]`. The module docstring
has the exact definitions. It reports:

1. **Reliability of c:** bins, ECE, ACE, Brier, AUROC and AURC, computed with the functions in `metrics.py`.
2. **q_L(c):** binned and isotonic. A2 is tested with a logistic slope (Wald CI) and a permutation test, and the
   error overlap of the two systems is reported.
3. **ΔC and κ** from the logged per-decision costs.
4. **V**, either user-given or the counterfactual approximation W·(S_right − S_wrong) over episodes.
5. **The plug-in τ̂\*** under Prop. 1 (`prop1`), with A2 relaxed (`qL`), and with A1 and A2 relaxed (`general`,
   the default). When g − q_L is non-decreasing it is the crossing of Eq. 3; in general it is the threshold that
   maximises the plug-in gain.
6. **The simulated cascade frontier.** It is exact for shadow arms, because both answers are logged at every state.
7. **Regret** of τ̂\* and of τ = 0.8 against the best swept τ, on held-out tasks (a seeded split by task id), with
   paired task-bootstrap CIs.
8. **VOI of the confidence gate** against a random gate at the matched acceptance rate, with a bootstrap CI.

Cascade arms log the LLM only when they defer, so pooling them biases q_L(c) towards low confidence. The headline
test uses a shadow arm.

**Resume.** A task counts as done only if its latest record has no `error`. Episodes that ended in an infrastructure
error are re-run on resume, and loaders keep the last record per task id. To disable re-runs, use one of:

- `retry_errors: false` in the YAML;
- `JEV_RETRY_ERRORS=0`;
- `python -m jevharness.bench.runner CONFIG --no-retry-errors`.

## Meta-decision golds, wrong-proposal probes and the free-text policy (amendment A4)

Normative definitions: [`META_GOLD_SPEC.md`](META_GOLD_SPEC.md). This section says where the implementation lives and
how to run it.

**Golds** (JevGym; other suites log `gold_status = "no_oracle"` unless they implement the hooks). Every meta
`DecisionLog` carries `state`, `gold`, `gold_status` (`ok` / `doomed` / `unobserved` / `no_oracle` / `cf_error`) and
`gold_aux` (always `v: 1`). They are computed by the env hooks `oracle_verdict`, `oracle_finish_ok`,
`oracle_world_complete`, `oracle_cost_to_go`, `oracle_need`, `oracle_acceptable_tool_sets`, `oracle_mem_gold`,
`oracle_memory_read_gold` and `oracle_snapshot` (`bench/suites/jevgym.py`), which the harness calls via `getattr`
after the System-1 answer is recorded. `Question.meta` is unchanged (`oracle_done` for stop, `oracle_accepts` for the
tool guard, canonical values for Choices, nothing for reflection / routing / memory); `extra.sim_gold: 1` (stop, tool
guard, memory_write) sends the new gold to simulated backends in simulation studies.

| Decision point | Primary gold | `gold_aux` |
|---|---|---|
| reflection (trigger, shadow) | `need_fatal`: the System-2 action at the state is fatal / premature / wrong_answer; fired trigger steps are `unobserved` | `verdict`, `need_any`, `retro`, `d_before`, `verdict_src`; CF: `verdict_alt`, `verdict_placebo`, `benefit`, `harm`, `cf_weight`, `cf_status` |
| routing (trigger, shadow) | `need_fatal` of the small model's action (large-routed steps `unobserved`) | as reflection; CF: `verdict_large` |
| stop guard | `finish_ok` | `kind` (ok / premature / wrong_answer), `d_before`, `canonical_done`, `world_complete`; CF: `verdict_reject` |
| stop trigger | `world_complete` | `canonical_done`, `cut_losses` |
| tool guard | `proposal_progress` | `verdict`, `canonical`, `d_before`, `d_after` |
| tool_select Choice | canonical tool (continuity) | `acceptable_tools` (AT, primary), `acceptable_tools_unfiltered`, `offered` (assist) |
| memory_write | `mem_need` | `far` (an atom must survive ≥ 2 oracle steps), `new`, `atoms` |
| memory_read | labels of the stored items holding an input of the next oracle step, or the "none" option | `lenient`, `rounds` |
| tool_params | canonical (bounded / copy); none for free text (`generate` records) | `param_kind` |

`complete(s)` additionally requires the evidence atoms of the finish step to be visible (instruction or an
observation so far); without that clause the answer-only families (`total`, `stock`) would be "complete" at step 0
and the oracle's own information-gathering steps would score as detours (a clarification of META_GOLD_SPEC §2.2).

**Wording (A4).** `Q_REFLECTION` is forward-looking ("Should the agent pause and re-check before its next action …");
the pre-A4 wording is asked additionally in the measurement arms as a probe-list record (`probe_source = wording_v1`).
`Q_STOP_GUARD` shows the proposed answer (JSON-quoted, ≤ 300 characters; `controls.infer_dp` ignores quoted spans).
The stop-guard rejection note asks to re-check the requests and recompute the answer.

**Free-text parameters** (`harness/tools.py`: `ParamInfo.kind`). Every parameter is bounded / copy / free from the
schema only (rule 1: an annotation — the schema key `x-jev-fill`, the suite's `Env.param_kinds`, or the
experiment's `tool_params.extra.param_kinds`; then enum / boolean / small integer range → bounded; number, or a
string with format / pattern or an identifier-like name or description → copy; any other string → free). System-1
never fills free parameters: System-2 fills them (required or optional) in the forced call, logged as one
`tool_params` record with `mode = "generate"` per free slot (counted LLM-made by the delegation ratio).
`tool_params.extra.answer_as_copy: true` is the H3 ablation. The copy-coverage valve
(`python -m jevharness.bench.meta_gold coverage`, dev seeds 5000–5999) gives 1.000 for every JevGym copy parameter
on the E2 suite, so nothing is reclassified.

**Tournaments.** When the backend batches (`batched_questions`), all chunk questions of a tournament round go out in
one request (tool selection, memory recall).

**Measurement arms** (`jevgym_meta_control.yaml`: `meta_shadow` on Flash, n = 200; `meta_shadow_s2lite` on Flash-Lite,
n = 500). Every decision point is in shadow, so the episode is the LLM-only one; the `measure` block adds, after
`env.evaluate`, the counterfactual branches (`cf`: reflection, large model, stop rejection, placebo; positives always,
a seeded 20% of negatives with weight 5) and the wrong-proposal probes (`probes`: natural, forced premature,
transplant, edits T1–T6) on snapshots taken at decision time. All measurement calls are ledger-tagged
(`meta.shadow`, `meta.measure` ∈ {shadow, cf, probe}) and reported under `usage.shadow.by_measure`; `usage.total`
equals LLM-only. Probes live in `Trajectory.probes` (`mode = guard_probe`), never in `steps[]`, so no primary
comparison, `trigger_rates`, threshold-theory input or harvest reads them; a `measure.probes` block on an arm with any
decision point that is not `off` / `shadow` (System-1 modes, and also reflection `always`, memory `always` / `llm`)
is a config error, and the score card's shadow-only test reads the logged config the same way. Live trigger arms may
set `measure.cf.fired_steps: true` (the counterfactual a0 at fired steps): set it on BOTH arms of a matched pair or
on neither (`baselines_jevgym.yaml` sets it on every Flash-Lite reflection-trigger arm); measurement calls never
count as LLM or strong calls (`strong_call_ratio` excludes `meta.shadow` calls).

**Backend identity.** Every episode logs its System-1 spec (`meta.s1 = {id, describe}`; the id hashes the full
`describe()` without the cache flag), and replay rows carry the replayed spec's `backend_id`. H2 cells and the VoI
readout use the LIVE arm's System-1 on the measurement arm's shadow states: the measurement arm's own log when the
spec ids match, else that spec's replay; a cell with neither has no statistics and is not estimable (never another
backend's answers). Replay rows enter H2 only from `steps[]` shadow records (`source = steps`); probe-list rows
(guard probes, `wording_v1`, augmented stop-trigger queries) feed the probe tables only. The measurement arm is chosen
per gate: routing uses the arm whose System-2 is the routers' small model (`meta_shadow_s2lite`), every other gate the
arm of the live arm's System-2 family.

**Stop-guard retries.** A rejected finish enters System-1's history, so the re-proposal is asked at a new state, and
guard records carry `retry` (earlier rejections in the episode; tool guard: at the step). Matched rates, guard FAR /
FRR, FAR_nat and gate fire rates count each proposal once (`metrics.opportunity_key`); retries are reported
separately.

**Matched-rate protocol.** `controls.trigger_rates('<run>/<arm>.jsonl')` -- one arm's file (a run directory pools
every arm and warns; `by_arm=True` groups them), per distinct opportunity (`n` / `rate`; `n_all` / `rate_all` /
`n_retries` when retries were dropped). The `ratem*` backends of `baselines_jevgym.yaml` are `placeholder: true`:
the runner skips their arms until the backend is replaced from the System-1 arm's rate, and a resume refuses an arm
whose logged System-1 spec differs from the current one.

**Readouts.**

- Score card: per arm, gold coverage and `gold_status` counts, gate statistics of measurement arms with the "no
  opportunity" flag, the live-gate consistency check, counterfactual efficacy and coverage, the probe FAR / FRR table
  (task-clustered CIs, representativeness TOST), recall@k, the fill-kind composition; `--pair LIVE:CONTROL
  --measure-arm meta_shadow_s2lite [--measure-run RUN_E2B] [--replay RUN_E2B/replay/<arm>__<backend>.jsonl]` adds
  gross and net ΔU, the ±2 pt TOST and the predicted-vs-observed VoI of the gate (§9.1).
- `python -m jevharness.bench.meta_gold` (also `jevbench gold`): `replay` (other backends on the logged
  state / question pairs → `RUN/replay/<arm>__<backend>.jsonl`), `audit` (deterministic replay of the golds of the
  executed actions; the reference implementation), `h2` (the H2 cell table with the ±2 pt TOST per cell plus the
  pre-registered regression of gross ΔU on AUROC [+ accuracy] with the LR test, and the §9.1 theory check;
  `--measure-run` loads the measurement arms from another run directory, `--replay` adds other backends), `ceiling`
  (C5 one-step oracle ceiling and the "no headroom" rule), `coverage` (the free-text valve).
- `python -m jevharness.bench.threshold_theory RUN --gate reflection [--V …] [--e --h --c-f] [--family MODEL]
  [--cost-arm LIVE_ARM]`: by default only the shadow records of the measurement arms of ONE System-2 family (`--arm`
  names arms explicitly, then every logged mode is used); c_f from `--cost-arm` or the analysed arms' ledger (a
  measurement arm prices a firing from its counterfactual calls of the same prompt); the gate plug-in rule
  τ̂* = g⁻¹((c_f/V + h)/(e + h)), the simulated gate frontier, VoI against the matched random gate and held-out regret;
  `--gold canonical` switches the tool_select pairs back to the canonical gold.
