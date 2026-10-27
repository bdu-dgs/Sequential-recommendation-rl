# Interview notes

Talking points for discussing this project. Every number is in `reports/research_report.md`; the policy numbers
are **simulated returns** (KuaiRecFO-v1, max 60), never real-user metrics.

## 30-second version
I studied how to fine-tune an offline RL recommender online without losing what it learned offline. I used
KuaiRec, a short-video dataset with a fully-observed block, to build a simulator whose immediate rewards are real
watch ratios. The dynamics are explicit synthetic rules, and I generated logs with known propensities. Offline,
conservatism mattered: plain DQN reached 21.6 against 44.6 for CQL, because 66% of its choices were items the
logging policy almost never showed. When fine-tuning, removing conservatism cost 35 points. My own idea was a
support-weighted CQL penalty. It failed a pre-registered test against a strength-matched control, with a drop of
−10.0 vs −0.9. I traced the failure to Q-overestimation on mid-support items, measured exactly in the simulator.

## 2-minute structure
1. **Problem.** Logs cover items unevenly. Offline RL handles this with conservatism, but online fine-tuning
   needs to relax it. How should it be relaxed?
2. **What made the data trustworthy:**
   - Audit first: 7.7% exact-duplicate rows, and no logging propensities at all, so no IPS on the real logs.
   - Temporal split and unseen-user split.
   - The supervised GRU beat item-mean on AUC (0.651 vs 0.617) but not on full-ranking NDCG. I report that.
3. **Simulator design.** Real FO watch ratios give the immediate reward. Fatigue, patience and horizon are
   explicit, documented assumptions. The logging policy is my own, so propensities are exact.
4. **Baselines and the main finding.**
   - Offline DQN collapses; CQL, BCQ and Cal-QL are about 44.6–44.8.
   - Naive fine-tuning drops 34.7. Cal-QL is the most stable (−0.3).
5. **My method, and why it failed.**
   - The pre-registered decision rule, with uniform-tracked and shuffled controls.
   - The method is rejected.
   - The mechanism comes from exact Q-error by support bucket.
   - The narrow-coverage ablation is consistent with it.
6. **Support also predicts evaluation risk** (report §8):
   - **Learned simulator.** A learned simulator overstates unconstrained policies by +10.4, and constant CQL by
     only +0.4.
   - **OPE.** It is accurate to about 1% on supported targets. On an unsupported one (ESS 6 of 50k), IPS is
     off by 82%.

## Likely questions

**Why trust a simulator at all?**
- I don't, for anything about real users. The simulator is a controlled testbed.
- Immediate rewards are real, and dynamics are rules I can vary. Every policy number is labeled "simulated".
- What a simulator *does* give is exact ground truth: true Q-values and exact OPE targets. That is what made the
  mechanism analysis possible.

**Why CQL and not IQL / AWAC / PPO?**
- The action space is discrete with 3,327 items, so CQL's logsumexp is exact.
- Its scalar α is exactly what I wanted to make per-(state, item).
- Double DQN fine-tuning keeps the same network and objective family, so only the regularizer changes between
  phases.

**What exactly is inherited from offline to online?**
- Copied: Q-network weights and the target network.
- Reset: the Adam state, because the moments are fitted to the offline loss.
- Kept: the offline replay buffer, at 50% of each batch.
- The same rules apply to every fine-tuning method, so comparisons isolate the regularizer.

**Why did the support-weighted method fail?**
- ρ = p·|A| treats an item shown at 3× the uniform rate as "supported" and cuts its penalty to about a quarter.
  But the log holds only a few samples of each such item.
- Their Q-values inflate by about +38. The greedy policy concentrates 70% of its recommendations there, and the
  return drops.
- The weight should depend on **sample counts or uncertainty**, not on probability relative to uniform.

**How do you know it is not a tuning problem?**
- The strength-matched controls use the same total penalty. Only *where* the penalty goes differs.
- BC-estimated support behaves like oracle support.
- κ ∈ {0.3, 3} and λ = 0 do not fix it (seed 0).
- In the narrow regime, where there is little mid-support mass, the gap nearly disappears.

**Why does the bandit win?**
- On this matrix, item quality dominates: popularity reaches 44.1, a long way from the oracle's 59.5. A 50k-step
  budget is not enough for value-based fine-tuning to improve beyond its offline start.
- A bandit that adds per-user exploration on top of a good offline model gains about one point.
- This is a property of the testbed and the budget, and I say so.

**What would you do next?**
1. A count- or ensemble-uncertainty-based conservatism weight.
2. Larger online budgets.
3. A testbed where long-term credit matters more, for example stronger fatigue.
4. Slate actions.

**What did you get wrong along the way?**
- My first runs memorized the log: the ranker's train MSE fell below the reward noise. I changed the protocol
  before using any result (regularization, tag-based history, validation checkpoint selection) and logged the
  change in `docs/decisions.md`.
- An independent code review found that evaluation was silently using the learned simulator. That was fixed
  before the main runs.
- I assumed the learned simulator would be most wrong on rare items. It wasn't, and the document was corrected.

## Numbers to remember (5 seeds unless noted, simulated)

| quantity | value |
|---|---|
| offline DQN vs CQL | 21.6 vs 44.6; 66% vs 0% unsupported choices |
| naive fine-tuning drop | −34.7 |
| Cal-QL fine-tuning drop | −0.3 |
| support-weighted vs strength-matched drop | −10.0 vs −0.9 |
| overestimation at the end, mid-support items (3 seeds) | +38 for support-weighted vs +5 for uniform |
| logging / popularity / oracle | 36.6 / 44.1 / 59.5 |
| real data, unseen users: GRU AUC vs item mean | 0.651 vs 0.617 |
| learned-sim overstatement: scratch DQN vs constant CQL | +10.4 vs +0.4 |
| OPE IPS error: CQL vs offline DQN | 1% vs 82% (ESS 3,580 vs 6) |
| total compute | 217 runs, 24.8 GPU-process-hours |
