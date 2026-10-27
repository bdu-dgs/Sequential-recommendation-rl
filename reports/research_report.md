# Support-Aware Offline-to-Online RL for Sequential Recommendation — Research Report

*Date: 2026-10-26. All experiments ran on one RTX 4060 Laptop GPU. Every number below links to a table in
`reports/tables/` generated from run directories under `experiments/runs/` (config + metrics JSON + command).*

> **How to read this report.** It has two separately reported kinds of evidence:
> - **Part A, real data:** supervised models trained and tested on real KuaiRec logs.
> - **Part B, simulated:** every policy number is a **simulated return** in `KuaiRecFO-v1`, out of a maximum
>   of 60. A simulated return is not user growth, CTR, watch time, or revenue. No online experiment with real
>   users was run.

## 0. Summary

**Question.** Logged recommendation data cover some items well and most items barely at all. How should an
offline-trained RL policy be fine-tuned online without first losing the performance it already has?

**Pre-registered hypothesis.** Make the CQL penalty during fine-tuning *support-aware*: weaker on items the log
covers well, full strength elsewhere. This should reduce the initial drop without the excess caution of a
uniform penalty. The test: it must beat a **strength-matched uniform** control and a **shuffled-support**
control.

**Findings.** Simulated returns, 5 seeds, mean [95% bootstrap CI]:

1. **Offline conservatism is necessary.** Offline Double DQN reaches only 21.6 [18.0, 25.3], and 66% of its
   recommendations are items the logging policy almost never shows (p < 1e-4). CQL reaches 44.6, BCQ 44.7 and
   Cal-QL 44.8, each with 0% unsupported choices. The logging policy reaches 36.6 and the oracle 59.5.
2. **Removing conservatism at fine-tuning time is catastrophic.** Naive fine-tuning from CQL drops by
   −34.7 [−38.3, −30.3] and ends at 15.9. Linearly decaying α only delays the collapse (−20.1). Constant CQL
   (−0.7) and Cal-QL (−0.3) are stable, but none of the fine-tuning methods improves on its offline start
   within 50k online transitions.
3. **The support-aware variant fails its pre-registered test.** Oracle-support weighting drops −10.0
   [−11.9, −7.4] (test 36.1). The strength-matched uniform control drops only −0.9 [−1.1, −0.7] (test 44.0).
   The hypothesis is **rejected**.
4. **Mechanism, measured rather than guessed.** In the deterministic simulator the true value of every greedy
   recommendation is computable, so Q-overestimation can be measured exactly, bucketed by logging support.
   The weighting relaxes the penalty on *mid-support* items (1e-4 ≤ p < 1e-2): items logged a few times, at
   roughly 3–30× the uniform rate. At the end of fine-tuning, these items receive 70% of the recommendations,
   and their Q-values are overestimated by +38. Under the matched control the corresponding share is 7% (+5).
   "Supported" in a probability sense does not mean "safe to relax" when the log holds only a handful of
   samples per item.
5. **Narrow-coverage ablation** (zero support outside a 500-item head) fits the mechanism. There is little
   mid-support mass to relax, so the gap shrinks: −1.2 vs −0.4, test 36.6 vs 36.9. Overestimation in the
   mid-support bucket is +4.5 instead of +38.
6. **Support predicts evaluation risk too** (§8):
   - **Simulator exploitation.** A policy trained in a *learned* simulator looks +10.4 better there than in the
     true simulator when it is unconstrained (online DQN). A constant-CQL policy looks only +0.4 better.
   - **OPE with known propensities.** IPS/SNIPS/DM/DR are accurate to about 1% for supported targets. For
     offline DQN (85% unsupported mass, ESS 6 of 50k), IPS is off by 82% and DR by 23%.
   - **Stress tests.** Drift, cold start, noise and stronger fatigue leave the method ranking unchanged.

**Contribution.** This is a negative result with a diagnosed mechanism, plus a reproducible evaluation stack:
- a leakage-free data audit;
- a simulator with real immediate rewards and documented synthetic dynamics;
- logs with known propensities for OPE;
- strength-matched controls;
- exact Q-overestimation diagnostics.

It is **not** a new algorithm that beats the baselines.

## 1. Data audit (real data) — [`docs/data_audit.md`](../docs/data_audit.md)

- **Dataset.** KuaiRec 2.0, CC BY-SA 4.0. The big matrix has 12,530,806 rows covering 7,176 users ×
  10,728 videos. The fully-observed (FO) small matrix covers 1,411 users × 3,327 videos.
- **Duplicates.** 965,819 big-matrix rows (7.7%) are exact duplicates. After deduplication on
  (user, video, timestamp), 11,562,801 rows remain.
- **No propensities.** KuaiRec does not record logging-policy probabilities, so no IPS/WIS/DR is ever run on
  real logs.
- **Feedback.** `watch_ratio` is the only per-user feedback signal. Its tail is extreme (p99 = 4.8, max = 573),
  so the reward is min(watch ratio, 2).
- **What the FO matrix provides.** It gives near-complete counterfactual *immediate* responses for one block of
  users and items. It provides no transitions, no long-term effects, and no exposure propensities. Those
  components of the simulator are therefore synthetic and labeled as such.
- **Splits.** The temporal split is global (train up to Aug 10, val to Aug 31, test Sep 1–5), plus 10% of users
  held out entirely. The FO users are split 846 / 282 / 283 into env-train / val / test.

## 2. Real-data supervised evidence (Part A) — [`tables/supervised.md`](tables/supervised.md)

- **Model.** A GRU two-tower response model trained on 7.78M train-window interactions. Run directory:
  `experiments/runs/response_model_wd/seed{0,1,2}`.
- **Reproduce:**
  ```
  python scripts/train_response_model.py --name response_model_wd --seed S weight_decay=0.00001 lr=0.0005 epochs=2
  ```

| test set | model | RMSE | engaged AUC | ECE |
|---|---|---|---|---|
| temporal test, unseen users | GRU two-tower | 0.561 | **0.651 ± 0.001** | 0.021 |
| temporal test, unseen users | item mean | 0.569 | 0.617 | 0.017 |
| FO matrix, full ranking of 3,327 items (283 test users) | GRU two-tower | NDCG@10 0.875 ± 0.004 | 0.781 | 0.023 |
| FO matrix, full ranking | item mean | NDCG@10 **0.877** | – | – |

**Honest reading:**
- Sequential personalization improves pointwise engagement prediction for unseen users: AUC +0.034 over item
  mean.
- It does **not** improve full ranking on the unbiased FO matrix, where item quality dominates.
- This matters for Part B: a good policy mostly has to find globally good items and avoid fatigue, and that is
  why simple popularity is a strong baseline there.

## 3. Simulator (Part B setup) — [`docs/problem_formulation.md`](../docs/problem_formulation.md), [`docs/environment_assumptions.md`](../docs/environment_assumptions.md)

**KuaiRecFO-v1:**

| Element | Definition | Origin |
|---|---|---|
| State | Last 20 (item, observed reward) pairs, step index, mask of items already shown | Warm-start history: the user's last 20 *real* views before Aug 10 |
| Action | 1 of the 3,327 FO items, no repeats within an episode | Real |
| Reward | min(W[u,a], 2) · (1 − 0.3·fatigue) | Base W is the **real** FO watch ratio; the fatigue term is **synthetic** |
| Episode | Patience 3 (lost when the response < 0.5), horizon H = 30 | **Synthetic**; γ = 0.95 |

- **Leakage.** Observations contain only information revealed before the current decision
  (`tests/test_env_no_leakage.py`).
- **Reference policies** on val users ([`tables/env_reference.md`](tables/env_reference.md)):

  | Policy | Simulated return |
  |---|---|
  | random | 9.7 |
  | stochastic logging policy | 37.0 |
  | greedy response-model ranker | 43.9 |
  | oracle greedy (reads the true table) | 59.5 |

  A toy MDP with a dynamic-programming optimum (3.93 vs 1.81 for the greedy policy) checks that DQN learns
  multi-step credit assignment (`tests/test_toy.py`).
- **Logging policy and offline data.** The logging policy is π_b = 0.95·softmax(f/0.1) + 0.05·uniform, where f is
  a real-data response model, and its **exact** propensities are stored. The offline set has 200k transitions
  per seed. Effective support is about 186 items per state. 1,213 items are logged 10 or more times, and the top
  100 items make up 73% of the log.
- **Learned simulator vs truth** ([`tables/learned_sim_error.md`](tables/learned_sim_error.md)). Replacing W with
  the response model's predictions gives RMSE 0.33–0.51 against the true FO watch ratio. The error is
  **not** larger on rarely logged items (mean error −0.01 for items logged 1–9 times). This contradicts the
  assumption in our plan that the learned simulator would be most wrong where the log is thin.

## 4. Protocol

- **Seeds and statistics.** Seeds 0–4 for main results. Intervals are 95% bootstrap CIs over seeds.
- **Evaluation.** Test numbers are one greedy episode per env-test user (283 users), and evaluation never
  updates the model (`tests/`).
- **Offline training.** 30k gradient steps, batch 256. Every learned method uses the same regularization:
  dropout 0.2 and AdamW with weight decay 1e-4. History items are represented by tags.
- **Checkpoint selection.** Every 2.5k steps, the same rule for all methods, by return on *validation* users.
  The last-step result is also recorded.
- **Protocol change.** The protocol was revised once, after seed-0 diagnostics and **before** any result was
  used. The v0 runs memorized the log. The reason and the archived runs are in
  [`docs/decisions.md`](../docs/decisions.md).
- **Online fine-tuning:**
  - 50k environment transitions on env-train users, 16 parallel environments;
  - 4 gradient steps per 16 transitions;
  - ε-greedy exploration 0.1 → 0.02;
  - validation every 2.5k transitions, including at 0.
- **Inheritance**, the same for every fine-tuning method:
  - copied: Q-network weights and the target network;
  - **reset:** the optimizer (Adam moments are tied to the offline loss);
  - replay: the offline buffer is kept, and every batch is 50% offline and 50% online.
- **Primary metrics:**
  - **worst drop:** the minimum validation return minus the starting return (the initial drop);
  - **area under the validation curve:** sample efficiency;
  - **final test return.**
- **Decision rule, fixed before results.** Support weighting "helps" only if it beats **both** the
  uniform-tracked and the shuffled controls, on drop or on area under the curve, with non-overlapping CIs.

## 5. Offline baselines (step 5) — [`tables/offline.md`](tables/offline.md)

Test users, simulated return, 5 seeds. "Unsupported" is the share of recommendations whose logging probability is
below 1e-4.

| method | test return | leave rate | unsupported |
|---|---|---|---|
| random | 9.41 [9.17, 9.61] | 0.98 | 82% |
| logging policy (behavior) | 36.61 | 0.36 | 8% |
| offline Double DQN | **21.58 [17.96, 25.33]** | 0.57 | **66%** |
| reward-regression ranker | 43.27 [42.52, 44.01] | 0.24 | 26% |
| log popularity | 44.06 [44.02, 44.10] | 0.24 | 0% |
| BC | 44.17 [44.06, 44.29] | 0.23 | 0% |
| discrete CQL (α = 1) | 44.56 [44.36, 44.76] | 0.23 | 0% |
| discrete BCQ (τ = 0.3) | 44.72 [44.57, 44.86] | 0.22 | 0% |
| Cal-QL | 44.84 [44.72, 44.96] | 0.22 | 0% |

**Reading:**
- The textbook offline-RL failure appears clearly. Unconstrained bootstrapping drives Double DQN toward
  unsupported items, and it loses half the return.
- Every support-constrained method lands at about 44–45. That is better than the logging policy (36.6), but
  barely better than log popularity. The limit is the dataset: what the log can reveal about better items.
- The ranker sits between the two. It chooses unsupported items 26% of the time, and its last-step checkpoint is
  much worse than the selected one: the winner's curse on rarely logged items.

## 6. Offline→online fine-tuning (step 6) — [`tables/online.md`](tables/online.md), [`figures/finetune_curves.png`](figures/finetune_curves.png)

| method | val start | worst drop | AUC (mean val) | test return |
|---|---|---|---|---|
| online Double DQN from scratch | 10.0 | – | 17.9 | 16.59 [13.67, 19.51] |
| RLPD-style from scratch (50% offline data) | 10.0 | – | 16.8 | 14.44 [11.10, 17.77] |
| NeuralLinear-UCB bandit | 43.5 | 0.0 | 45.5 | **45.46 [45.19, 45.84]** |
| fine-tune, naive (no penalty) | 44.8 | **−34.69 [−38.29, −30.34]** | 20.4 | 15.94 [12.74, 19.96] |
| fine-tune, α decayed linearly to 0 | 44.8 | −20.11 [−22.53, −18.01] | 41.9 | 24.24 [22.17, 26.05] |
| fine-tune, constant CQL | 44.8 | −0.74 [−1.07, −0.40] | 44.4 | 44.21 [43.75, 44.58] |
| fine-tune, Cal-QL | 45.0 | **−0.28 [−0.57, −0.07]** | 44.9 | 44.82 [44.51, 45.07] |

**Reading:**
- **Initial drop.** Inheriting a CQL network and removing the penalty immediately unlearns conservatism, and the
  return falls to the level of a from-scratch learner. Decaying α only postpones the collapse to the point where
  α reaches 0, after about 35k transitions.
- **Sample efficiency.** Learning from scratch never reaches the logging policy's return (37) within 50k
  transitions (0/5 seeds). Every method that starts from offline data begins above it.
- **Final return.** The 50k budget is too small for any value-based fine-tuning to *improve* on its offline
  start. The contextual bandit, which ignores long-term credit, ends highest (45.5). In this simulator, myopic
  item quality captures most of the attainable return below the oracle.

## 7. Support-weighted fine-tuning: rejected, and why

### 7.1 Pre-registered test (5 seeds) — [`tables/online.md`](tables/online.md)

- **Method.** CQL's push-down term is weighted per (state, item):
  - c(s,a) = 1/(1 + ρ(s,a)/κ);
  - ρ = π_b(a|s)·|allowed items| + λ·(online count of a);
  - κ = 1 and λ = 1.
- **Strength matching.** α stays at 1 for every variant. The uniform controls use a constant weight c̄ equal to
  the oracle weighting's mean strength E_s[Σ_a μ(a|s)·c(s,a)], where μ is the softmax of Q (`agents/support.py`).
  "uniform" matches it once, at the start of fine-tuning. "uniform-tracked" re-matches it at every update,
  because the oracle weights shrink as online counts grow.

| support signal | worst drop | test return |
|---|---|---|
| **oracle** (true π_b) | **−9.98 [−11.86, −7.38]** | 36.08 [32.87, 39.45] |
| BC-estimated π_b | −8.90 [−10.19, −7.33] | 37.81 [36.21, 39.56] |
| uniform, strength matched at the start | −1.22 [−1.57, −0.84] | 44.17 [43.52, 44.59] |
| **uniform, strength re-matched at every update** (primary control) | **−0.91 [−1.11, −0.70]** | 44.03 [43.54, 44.39] |
| shuffled (fixed random item permutation) | −15.78 [−22.02, −9.35] | 35.70 [30.91, 40.01] |
| inverted (stronger where support is high) | −0.75 [−1.35, −0.31] | 44.15 [43.66, 44.58] |

**Verdict.** Oracle support beats the shuffled control, but it is clearly **worse** than the primary control,
with non-overlapping CIs. The hypothesis is rejected. Estimated (BC) support behaves like oracle support, so
estimation error is not the problem.

**Seed-0 diagnostics** (single seed, exploratory; `experiments/runs/online/ft_support_*_{lam0,kappa*}/seed0`):

| setting | worst drop |
|---|---|
| no online counts (λ = 0) | −8.7 |
| κ = 0.3 | −9.6 |
| κ = 3 | −6.1 |
| oracle, default settings | −10.2 |
| matched-uniform control | −0.6 |
| shuffled with λ = 0 | −0.4 |

The failure therefore does not come from online counts or from the temperature. It comes from *which* items the
weighting relaxes.

### 7.2 Mechanism: exact Q-overestimation by support — [`tables/overestimation.md`](tables/overestimation.md), [`figures/overestimation_by_support.png`](figures/overestimation_by_support.png)

- **Measurement.** At each validation evaluation, we compare the network's Q(s,a) for the greedy recommendation
  with the true discounted return of that (s,a) under the current greedy policy. The simulator is deterministic,
  so the true return is exact. Results are bucketed by the logging probability p of the chosen item.
- **Data.** 3 seeds; run directories `experiments/runs/online/*_diag/seed{0,1,2}`.
- **Reproduce:**
  ```
  python scripts/run_local.py --stages diag
  python scripts/plot_overestimation.py --tag diag
  ```

| method | return start → end | overestimation start → end | share of recs at mid support (1e-4 ≤ p < 1e-2) | overestimation there |
|---|---|---|---|---|
| naive fine-tune | 44.9 → 15.8 | +11.3 → **+51.4** | 19% (77% at p < 1e-4) | +49.3 |
| constant CQL | 44.9 → 44.5 | +11.3 → +0.9 | 3% | +3.6 |
| **support-weighted (oracle)** | 44.9 → 35.9 | +11.3 → **+33.3** | **70%** | **+38.1** |
| uniform, strength-matched | 44.9 → 44.5 | +11.3 → +2.2 | 7% | +5.4 |

**Interpretation:**
- The offline CQL network starts out overestimating by +11. Under a uniform penalty, online data correct it.
- The support weight puts items with p ≈ 1e-3 (about 3× uniform) at ρ ≈ 3, which cuts their penalty to about a
  quarter. The log holds only a handful of samples of each such item, so their Q-values inflate (+38). The greedy
  policy then concentrates on them (70% of recommendations), and the return falls.
- Online exploration at ε ≤ 0.1, spread over 3,327 items, is too slow to correct them.
- The failure is therefore not that support is the wrong signal. A *probability* is the wrong quantity to relax
  on. Sample counts and the uncertainty they imply are what matter, and the method's ρ conflates "more likely
  than uniform" with "well estimated".

### 7.3 Narrow-coverage ablation (5 seeds) — [`tables/online_narrow.md`](tables/online_narrow.md), [`tables/overestimation_narrow.md`](tables/overestimation_narrow.md)

- **Setup.** The logging policy has ε = 0 and a softmax restricted to the 500 most popular items, so every other
  item has exactly zero support. The offline data and agents are re-generated and re-trained.
- **Run directories:** `experiments/runs/{offline,online}/*_narrow/seed{0..4}`.
- **Reproduce:**
  ```
  python scripts/run_local.py --stages s7a_data s7b_offline s7c_online --grep "head_size=500"
  ```
- **Offline results:**
  - behavior 33.3, BC 36.0, CQL 36.7, BCQ 37.2;
  - lower than with broad coverage, because the best items lie outside the head.

| fine-tuning (start 37.0) | worst drop | test return | overestimation (end) | mid-support share (end) |
|---|---|---|---|---|
| naive | −22.96 [−23.94, −21.82] | 21.66 | +20.3 (90% of recs at p = 0) | 4% |
| decayed α | −3.72 | 33.20 | – | – |
| constant CQL | −0.38 [−0.42, −0.34] | 36.70 | −0.6 | 0% |
| support-weighted (oracle) | −1.18 [−1.46, −0.91] | 36.56 [36.21, 36.86] | +2.6 | 15% |
| support-weighted (BC) | −1.63 [−2.70, −0.85] | 36.76 | – | – |
| uniform-tracked | −0.36 [−0.55, −0.18] | 36.89 [36.73, 37.05] | +1.4 | 4% |

**Reading:**
- Support weighting is still worse than the matched control on drop, with non-overlapping CIs, so the decision
  rule rejects it again.
- The gap shrinks from about 9 points to about 0.8, and the test-return CIs overlap.
- With a hard head and no tail of thinly logged items, the weighting has little "mid-support" mass to relax.
  This matches the mechanism in §7.2. It is a post-hoc consistency check, not a pre-registered prediction.
- None of the fine-tuning methods can reach items outside the head with a 50k budget. Even under narrow
  coverage, online fine-tuning recovers nothing beyond the offline start.

## 8. Simulator exploitation, generalization, and OPE (step 8)

### 8.1 Exploiting a learned simulator — [`tables/exploitation.md`](tables/exploitation.md)

**Setup:**
- Agents interact online with a *learned* simulator, in which the FO watch ratio is replaced by the real-data
  response model's prediction.
- They are then scored two ways on test users: in the true FO simulator, and inside the learned simulator they
  trained in.
- The gap is how far an evaluator who trusted the learned simulator would overstate the policy.

**Data and reproduction:**
- 5 seeds; run directories `experiments/runs/online/*_learnedsim/seed{0..4}`.
- Reproduce with `python scripts/run_local.py --stages s8a_online`.

| method | true return (FO sim) | return claimed by the learned sim | overstatement |
|---|---|---|---|
| online DQN from scratch | 20.35 [13.96, 25.91] | 30.74 [26.11, 34.51] | **+10.39 [8.37, 12.39]** |
| naive fine-tune | 16.36 [12.40, 21.87] | 24.75 [21.44, 29.16] | +8.39 [6.12, 11.74] |
| support-weighted (oracle) | 37.66 [36.66, 38.99] | 42.91 [42.31, 43.60] | +5.26 [4.48, 6.03] |
| constant CQL | 44.17 [44.04, 44.34] | 44.61 [44.46, 44.80] | **+0.44 [0.40, 0.50]** |

**Reading:**
- The less conservative the policy, the more it exploits simulator error. Ranked by overstatement, the
  policies line up exactly as their support usage does in §5 and §7.
- §3 showed that the learned model's *average* error is not larger on rarely logged items. A greedy policy
  exploits error anyway, because it *selects* the items whose predictions happen to be too high: the same
  winner's curse as in §5, now applied to a simulator.
- Training in the learned simulator did not reduce true returns. Each FO-sim return is within the CI of the
  same method trained in the FO simulator (§6). The damage is to the *evaluation*, not to the policy.

### 8.2 Generalization stress tests — [`tables/generalization.md`](tables/generalization.md)

**Setup:**
- Final checkpoints are re-evaluated on test users under synthetic shifts that were never seen in training:
  - preference drift δ ∈ {0.2, 0.5};
  - cold start with 0 or 2 warm-start items instead of 20;
  - observation noise σ = 0.3;
  - fatigue β = 0.6 instead of 0.3.
- 5 seeds; `experiments/runs/eval_generalization/seed{0..4}`.

| policy | base | drift 0.5 | cold start, 0 items | fatigue 0.6 |
|---|---|---|---|---|
| bandit | 45.46 | 39.93 | 45.53 | 44.19 |
| Cal-QL (offline) | 44.84 | 39.53 | 44.72 | 43.19 |
| CQL (offline) | 44.56 | 39.38 | 43.40 [42.94, 43.85] | 42.81 |
| popularity | 44.06 | 39.11 | 44.06 | 41.99 |
| fine-tune, constant CQL | 44.21 | 39.10 | 42.44 [39.97, 44.42] | 42.36 |
| fine-tune, uniform-tracked | 44.03 | 39.00 | 41.60 [39.41, 43.79] | 42.35 |
| fine-tune, support-weighted (oracle) | 36.08 | 32.11 | 34.24 | 33.81 |
| offline DQN | 21.58 | 18.34 | 21.62 | 18.64 |

**Reading:**
- **Drift.** Strong drift (δ = 0.5) costs every reasonable policy about 5 points, and the ranking of methods is
  unchanged. No method is specifically robust to drift; popularity degrades exactly as much as the learned
  policies.
- **Cold start.** It hurts the policies that read the history (CQL −1.2; constant-CQL fine-tuning −1.8, with
  wide CIs). Cal-QL, BCQ and the bandit barely change.
- **Noise.** Observation noise σ = 0.3 changes nothing measurable.
- **Fatigue.** Doubling fatigue costs 1–3 points across the board.
- **Overall.** The conclusions of §5–§7 hold under all seven settings.

### 8.3 Off-policy evaluation with known propensities — [`tables/ope.md`](tables/ope.md), [`figures/ope_error.png`](figures/ope_error.png)

**Setup:**
- One-step OPE (IPS, SNIPS, DM, DR) on 50k logged states from **our simulated logging policy**, whose
  propensities are exact.
- Ground truth: the exact expected immediate reward of each target policy, from the simulator.
- 5 seeds; `experiments/runs/ope/seed{0..4}`.
- Reproduce with `python scripts/run_ope.py --seed S`.
- None of this is possible on real KuaiRec logs, which have no propensities.

| target policy | unsupported mass | ESS (of 50k) | IPS rel. error | SNIPS | DM | DR |
|---|---|---|---|---|---|---|
| CQL (offline) | 0.00 | 3,580 | 1% | 1% | 1% | 1% |
| constant-CQL fine-tune | 0.00 | 3,121 | 1% | 0% | 1% | 0% |
| support-weighted fine-tune | 0.00 | 123 | 12% | 2% | 4% | 2% |
| bandit | 0.37 | 181 | 32% | 4% | 1% | 5% |
| online DQN from scratch | 0.69 | 7 | 72% | 18% | 10% | 34% |
| offline DQN | **0.85** | **6** | **82%** | 40% | 18% | 23% |

**Reading:**
- **Supported targets.** For policies whose actions stay on the log's support, every estimator is accurate to
  about 1%.
- **Unsupported targets.** Once the target puts most of its mass where the logging probability is below 1e-4,
  the effective sample size collapses from thousands to single digits. IPS is then off by 72–82%, and even DR is
  off by 23–34%.
- **Consequence.** OPE can certify exactly the policies that are already safe, and cannot evaluate the risky
  ones. That is the offline-evaluation face of the support problem studied in §5–§7.
- **Diagnostics.** ESS and unsupported mass, reported next to every estimate, flag every unreliable case in
  this table.

## 8.4 Compute — [`tables/compute.md`](tables/compute.md)

- **Total.** 217 recorded runs, 24.8 GPU-process-hours on one RTX 4060 Laptop GPU, generated by
  `python scripts/compute_summary.py`.
- **Physical time is lower.** Runs executed 3 in parallel, so physical time was about 9 h of wall-clock (estimated from the timestamps in `logs/local/`).
- **Largest families:**

  | family | GPU-process-hours |
  |---|---|
  | base online | 6.2 |
  | base offline | 5.4 |
  | narrow online | 4.0 |
  | narrow offline | 3.7 |
  | learned-simulator runs | 2.5 |
- **Not counted:** smoke tests and the archived v0 diagnostic runs.

## 9. What was not done (planned in `docs/experiment_plan.md`)

To keep compute on a laptop, only the coverage ablation that tests the mechanism was run. The following
step-7 ablations were **not** run, and no claim is made about them:
- the broad-coverage regime;
- environment-mechanism variations (fatigue, patience, noise during training);
- α ∈ {0.3, 3};
- inheritance ablations (keeping Adam, resetting the target net, offline ratio ∈ {0, 0.25}).

κ and λ were checked on seed 0 only (§7.1). The RIS cluster scripts exist (`ris/`) but were not used.

## 10. Limitations

- **Real vs simulated rewards.** Immediate rewards are real watch ratios, but fatigue, leaving and sequencing are
  synthetic rules, so the long-term structure of the simulator is assumed, not learned. Nothing here says how a
  policy would affect real users.
- **Small attainable gain.** The FO matrix favors item-quality policies. The gap between log popularity (44.1)
  and the oracle (59.5) is reachable only with per-user knowledge that the log does not provide.
- **One method family.** The support-weighted variant is a single design (a CQL push-down weight). A
  count- or uncertainty-based weight might behave differently, and §7.2 suggests that is the more promising
  direction. It was not tested.
- **Small budgets.** 50k online transitions and 200k logged transitions. With larger online budgets the
  ranking of methods could change.
- **Statistics.** 5 seeds give wide CIs for unstable methods (naive, shuffled). Mechanism results use 3 seeds,
  and the κ/λ diagnostics use 1.
- **Real-data OPE.** KuaiRec has no logged propensities, so no off-policy evaluation of real data was
  possible.

## 11. Reproduce

```bash
pip install -e .[dev]; python scripts/download_kuairec.py; python scripts/prepare_data.py
python -m pytest -q                                     # 16 tests
python scripts/make_jobs.py                             # writes experiments/jobs/*.txt
python scripts/run_local.py --stages s5a_data s5b_offline s6_online --parallel 3 --threads 3   # steps 5-6
python scripts/run_local.py --stages diag                                                      # mechanism
python scripts/run_local.py --stages s7a_data s7b_offline s7c_online --grep "head_size=500"    # narrow
python scripts/run_local.py --stages s8a_online s8b_eval                                       # step 8
python scripts/analyze.py --seeds 0 1 2 3 4; python scripts/analyze.py --tag narrow --seeds 0 1 2 3 4
python scripts/plot_overestimation.py --tag diag; python scripts/plot_overestimation.py --tag narrow --seeds 0 1 2 3 4
python scripts/analyze_supervised.py; python scripts/analyze_step8.py; python scripts/compute_summary.py
python demos/compare_trajectories.py
```
Each run directory contains `config.yaml` (resolved), `metrics.json`, `compute.json`, and the command line.
