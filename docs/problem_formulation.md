# Problem formulation

## Research question
When historical recommendation logs cover some actions poorly, how can an offline-learned policy avoid
over-optimism about unsupported recommendations? And when does a limited budget of (simulated) online interaction
improve the policy rather than degrade it?

**Hypothesis (pre-registered before any policy result).** During offline→online fine-tuning, if the strength of
the conservative update for each (state, item) is tied to that item's support in the offline data, the
initial performance drop is smaller than under unregularized fine-tuning. Room for later improvement is kept
better than under uniform conservatism with the **same average strength**. The hypothesis fails if the
support-weighted variant is not better than the matched-strength uniform and shuffled-support controls.

## Two separately reported parts
1. **Real-data supervised evidence:** a user-response model trained and evaluated on real KuaiRec logs, with a
   global temporal split and held-out users (`scripts/train_response_model.py`).
2. **Controlled policy experiments:** DRL policies trained and compared in the `KuaiRecFO-v1` simulator. Every
   policy number is a *simulated return*.

## MDP (KuaiRecFO-v1)
| Element | Definition | Provenance |
|---|---|---|
| User population | 1,411 FO users; 846 train, 282 val, 283 test | real |
| Action | one item from the 3,327 FO items, excluding items already shown in the episode | real catalog |
| Observation / state | the last K=20 (item, observed reward) pairs, plus the step index t and the set of items already shown in the episode. History before the episode is the user's last 20 real big-matrix views before 2020-08-10. | real warm start + agent's own interactions |
| Reward | r_t = min(W[u,a], 2) · (1 − β·f_t(a)) (+ optional noise in the observed reward). W is the real FO watch ratio; f_t(a) is the fraction of the last 5 recommendations sharing a tag with a. β = 0.3 by default. | base real; fatigue synthetic |
| Transition | the history window shifts and appends (a, observed r); patience decreases by 1 when the effective response is below 0.5 | history update exact; patience synthetic |
| Termination | patience (3) exhausted ("user leaves"), or t = H = 30 | synthetic |
| Discount | γ = 0.95 for learning. Evaluation reports undiscounted simulated return, episode length, and leave rate. | – |

Information timing: the observation at step t contains only items shown before t and the feedback the user gave
to them. The response to the item recommended at step t enters the observation at step t+1.
Tested in `tests/test_env_no_leakage.py`.

Why this is a sequential problem rather than a bandit:
- Patience couples steps: a risky recommendation can end the session.
- Fatigue couples steps: repeated tags lower later rewards.
- Within-episode feedback reveals the user's preferences (partial observability).

## Offline data
Logs are generated in the simulator on env-train users by a **known** logging policy:

  π_b(a|s) = (1 − ε)·softmax_{a ∈ M(s) ∩ S}(f(s,a)/T) + ε·Uniform(M(s))

Here f is the real-data response model ("production ranker") and S is all items or a popular head set.
The probability of every logged action is stored, so OPE and "oracle support" are exact. The default is
T = 0.1, ε = 0.05, 200k transitions.

## Methods compared (same data, users, budgets)
- **Reference policies:** random, log popularity, the logging policy itself, and an oracle greedy policy (reads
  true rewards; reference only).
- **Supervised ranker:** reward regression on the logs (γ = 0), plus the real-data response model used greedily.
- **Behavior cloning.**
- **Contextual bandit:** NeuralLinear-UCB on top of the ranker, updated online.
- **Offline RL:** discrete CQL (primary), discrete BCQ, Cal-QL, offline Double DQN (non-conservative control).
- **Online RL:** Double DQN from scratch; RLPD-style symmetric offline/online sampling from scratch.
- **Offline→online fine-tuning from CQL:**
  - naive Double DQN (α = 0);
  - constant CQL;
  - linearly decayed CQL;
  - Cal-QL;
  - **support-weighted CQL** with oracle / BC-estimated / matched-uniform / shuffled / inverted support.

## Metrics
- Simulated return (test users), episode length, leave rate.
- Coverage (unique items / catalog), tags per episode, long-tail share.
- Logging probability of the chosen actions (support).
- Offline→online: return at the start of fine-tuning, the worst drop below it, interactions needed to reach
  the logging policy's return, and final return.
- Mean ± 95% bootstrap CI over 5 seeds.
