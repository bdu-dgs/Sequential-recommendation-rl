# Environment assumptions (KuaiRecFO-v1)

Every result produced by `src/seqrec_rl/environments/kuairec_sim.py` is a **simulated return**. This file
lists what is real, what is invented, and how each invented part is checked or varied.

## Real components
| Component | Source | Caveats |
|---|---|---|
| Users | 1,411 users of the fully-observed (FO) small matrix | FO users are mostly "full_active" (1,213 of 1,411) |
| Actions | 3,327 FO items | blocked pairs (0.4%) give response 0 |
| Immediate response | min(watch_ratio, 2) of that user on that item, from the FO matrix | **Static:** one value per pair, collected over two months with forced exposure. There are no order, context, or repeat effects, and the value includes single-observation noise. |
| Warm-start history | the user's last 20 big-matrix views (capped watch ratio) before 2020-08-10 | contains no catalog items (big and small share no pairs), so no response labels leak into the state |

## Synthetic components (not identified by data)
| Rule | Default | Why synthetic | Varied in |
|---|---|---|---|
| Fatigue: response × (1 − β·f), with f the share of the last 5 recommendations sharing a tag | β = 0.3 | Real logs show only a weak, position-confounded association (`data_audit.md`) | β ∈ {0, 0.3, 0.6} |
| Patience: −1 per recommendation with effective response < 0.5; the user leaves at 0 | 3 | KuaiRec has no session-end signal that depends on recommendations; real session ends are U-shaped in watch ratio | patience ∈ {2, 3, 5} |
| Horizon | 30 | episode cap common in KuaiRec RL work (DORL, EasyRL4Rec) | fixed |
| Preference drift between train and test environments: W × exp(δ·z_{u,tag}) | δ = 0 | generalization stress test | δ ∈ {0.2, 0.5} at test time |
| Observation noise: the observed reward is the effective response + N(0, σ²), clipped | σ = 0 | feedback-noise stress test | σ ∈ {0.3} |
| Cold start: only the last k warm-start items are kept | k = 20 | cold-start evaluation | k ∈ {0, 2} at test time |

## Learned components (exploitation study only)
`response: learned` replaces W with the real-data response model's prediction from the warm-start history.
Policies trained in this learned simulator are evaluated in the FO simulator; the return gap measures
simulator exploitation.

On real held-out data, the learned model reaches engaged AUC 0.784 and ECE 0.023 on the FO matrix
(`experiments/runs/response_model_wd/seed0`). We expected its error to be largest on items with
low logging support, where a policy could exploit it. **This expectation was wrong** (`reports/tables/learned_sim_error.md`):
the mean error is about 0 in every log-count bucket (−0.01 for items logged 1–9 times), and the RMSE is *lower* on
rarely logged items (0.39) than on frequently logged ones (0.51). The step-8 exploitation result is in
`reports/research_report.md` §8.

## What the simulator cannot tell us
- Real long-term effects: retention, return visits, or real fatigue.
- Whether a policy would increase real watch time, CTR, or revenue. Rewards are real watch ratios, but the
  sequence they are delivered in is simulated.
- The real logging policy's propensities. OPE is only valid for our own simulated logging policy.

## Sanity checks (reference policies, 282 val users, `experiments/runs/characterize_env/seed0`)
| Policy | Simulated return | Length | Leave rate |
|---|---|---|---|
| random | 9.7 | 11.8 | 0.95 |
| real-log popularity (fixed list) | 2.3 | 4.6 | 0.98 |
| logging policy, stochastic | 37.0 | 25.2 | 0.33 |
| logging policy, greedy (= response-model ranker) | 43.9 | 27.3 | 0.21 |
| oracle greedy, ignores fatigue (reads W) | 56.5 | 30.0 | 0.00 |
| oracle greedy (reads the effective response) | 59.5 | 30.0 | 0.00 |

The maximum return is 60. Most of the gap between the ranker and the oracle comes from personalization
(predicting W), not from long-term planning. Fatigue accounts for about 3 points between the two oracles.
