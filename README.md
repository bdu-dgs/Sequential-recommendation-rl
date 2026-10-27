# Support-Aware Offline-to-Online RL for Sequential Recommendation

When historical recommendation logs cover some items poorly, how can an offline-learned policy avoid
over-optimism about unsupported recommendations? And when does a limited budget of online interaction help
rather than hurt? This repository studies these questions with real KuaiRec data plus a clearly labeled
simulator.

> **Status (2026-10-26): complete.** Full write-up: [`reports/research_report.md`](reports/research_report.md).
> Every policy number is a **simulated return** (max 60), 5 seeds, mean [95% bootstrap CI].

## Results at a glance
| finding | evidence |
|---|---|
| Offline conservatism is necessary. Offline Double DQN reaches 21.6 [18.0, 25.3], with 66% of its choices unsupported by the log. CQL reaches 44.6, BCQ 44.7 and Cal-QL 44.8, with 0% unsupported. | [`offline.md`](reports/tables/offline.md) |
| Dropping the penalty at fine-tuning time collapses the policy: naive fine-tuning drops −34.7. Constant CQL (−0.7) and Cal-QL (−0.3) are stable. No value-based method improves on its offline start within 50k transitions. The bandit ends highest at 45.5. | [`online.md`](reports/tables/online.md), [`finetune_curves.png`](reports/figures/finetune_curves.png) |
| **The pre-registered hypothesis is rejected.** Support-weighted CQL drops −10.0, against −0.9 for the strength-matched uniform control. | [`online.md`](reports/tables/online.md) |
| **Mechanism.** The weighting relaxes the penalty on mid-support items (logged a few times). Their Q-values are overestimated by +38, and they receive 70% of the recommendations. Under narrow coverage, with no mid-support tail, the gap nearly vanishes. | [`overestimation.md`](reports/tables/overestimation.md), [`online_narrow.md`](reports/tables/online_narrow.md) |
| A learned simulator overstates unconstrained policies by +10.4, and constant CQL by only +0.4. | [`exploitation.md`](reports/tables/exploitation.md) |
| OPE with known propensities is accurate to about 1% on supported targets. On unsupported targets, IPS is off by 82% (ESS 6 of 50k). | [`ope.md`](reports/tables/ope.md), [`ope_error.png`](reports/figures/ope_error.png) |
| Real data: the GRU response model beats item mean on engaged AUC for unseen users (0.651 vs 0.617), but not on full-ranking NDCG. | [`supervised.md`](reports/tables/supervised.md) |

Compute: 217 runs, 24.8 GPU-process-hours on one laptop GPU ([`compute.md`](reports/tables/compute.md)).

## Two kinds of evidence, reported separately
1. **Real data (supervised).** A GRU two-tower user-response model trained on the 7.78M train-window
   interactions (of 11.56M after deduplication) under a global temporal split. It is evaluated on a held-out week and on full ranking over the
   1,411 × 3,327 fully-observed matrix ([`docs/data_audit.md`](docs/data_audit.md)).
2. **Controlled simulation (policies).** In `KuaiRecFO-v1`, immediate rewards are real fully-observed watch
   ratios, while fatigue, leaving, drift, and noise are explicit synthetic rules
   ([`docs/environment_assumptions.md`](docs/environment_assumptions.md)). Offline logs come from a known
   logging policy with exact propensities. **Every policy number is a simulated return**, not a claim about
   real users, CTR, watch time, or revenue.

## Methods
- **Baselines:** popularity, a supervised ranker, behavior cloning, a NeuralLinear-UCB contextual bandit.
- **Offline RL:** discrete CQL, discrete BCQ, Cal-QL, offline Double DQN.
- **Online RL:** Double DQN (from scratch and RLPD-style).
- **Fine-tuning:** offline→online fine-tuning with naive, constant, decayed, or calibrated conservatism, and
  **support-weighted CQL**. Its per-(state, item) push-down weight is c(s,a) = 1/(1 + ρ(s,a)/κ), where ρ is
  the support relative to uniform under the logging policy plus online counts. It is tested against
  strength-matched uniform (re-matched at every update), shuffled, and inverted controls. This is framed as a design study, not a new
  algorithm; see [`docs/literature_review.md`](docs/literature_review.md) §4 for the closest prior work.

## Quick start (local)
```bash
conda create -p G:/Anaconda/envs/seqrl python=3.11 && conda activate G:/Anaconda/envs/seqrl
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
python scripts/download_kuairec.py          # KuaiRec 2.0 from Zenodo (CC BY-SA 4.0), ~430 MB
python scripts/prepare_data.py              # dedup, splits, FO matrix, tags -> data/processed/
python -m pytest -q                         # 16 tests: toy DP optimum, leakage, rewards, episode boundaries
python scripts/train_response_model.py --name response_model_wd --seed 0 weight_decay=0.00001 lr=0.0005 epochs=2
python scripts/train_response_model.py --name response_model --seed 0      # logging-policy scorer
python scripts/characterize_env.py          # reference policies + offline dataset statistics
python scripts/run_offline.py --method cql --seed 0
python scripts/run_online.py --method ft_support_oracle --seed 0
```
All experiments: `python scripts/make_jobs.py`, then `python scripts/run_local.py --stages <stage> ...`.
The exact sequence is in the report's §11. SLURM scripts for WashU RIS are in [`ris/`](ris/README.md) (unused).

## Layout
```
configs/           response_model.yaml, experiment/base.yaml
src/seqrec_rl/     data/ user_models/ environments/ agents/ replay/ training/ evaluation/
scripts/           CLI entry points (the only supported way to run things)
tests/             unit tests
ris/               SLURM scripts for WashU RIS
docs/              problem formulation, data audit, assumptions, literature, plan, decisions, status,
                   resume evidence, interview notes
reports/           research report, tables/, figures/, demo trajectories
demos/             trajectory demo (no GPU)
experiments/runs/  run outputs (git-ignored): config, metrics, logs, checkpoints, compute
```

## Data and license
KuaiRec 2.0 (Gao et al., CIKM 2022) is licensed under CC BY-SA 4.0. Data, weights, and run outputs are not
committed.
