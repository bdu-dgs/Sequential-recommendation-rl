Real KuaiRec data. GRU two-tower response model (`response_model_wd`), mean ± std over seeds [0, 1, 2]. Baselines are deterministic. Targets: watch ratio capped at 2 (RMSE); engaged = watch ratio ≥ 1 (AUC, GAUC, ECE). Temporal test window: 2020-09-01..05.

| test set | model | RMSE ↓ | AUC ↑ | GAUC ↑ | logloss ↓ | ECE ↓ |
|---|---|---|---|---|---|---|
| seen users | gru_two_tower | 0.551 ± 0.000 | 0.658 ± 0.001 | 0.630 ± 0.002 | 0.572 ± 0.001 | 0.020 ± 0.001 |
| seen users | global_mean | 0.590 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.627 ± 0.000 | 0.027 ± 0.000 |
| seen users | item_mean | 0.562 ± 0.000 | 0.623 ± 0.000 | 0.630 ± 0.000 | 0.594 ± 0.000 | 0.017 ± 0.000 |
| seen users | user_item_bias | 0.559 ± 0.000 | 0.657 ± 0.000 | 0.630 ± 0.000 | 0.595 ± 0.000 | 0.028 ± 0.000 |
| small users | gru_two_tower | 0.546 ± 0.001 | 0.602 ± 0.001 | 0.590 ± 0.001 | 0.631 ± 0.001 | 0.040 ± 0.002 |
| small users | global_mean | 0.567 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.659 ± 0.000 | 0.023 ± 0.000 |
| small users | item_mean | 0.552 ± 0.000 | 0.594 ± 0.000 | 0.600 ± 0.000 | 0.639 ± 0.000 | 0.037 ± 0.000 |
| small users | user_item_bias | 0.547 ± 0.000 | 0.614 ± 0.000 | 0.600 ± 0.000 | 0.639 ± 0.000 | 0.032 ± 0.000 |
| unseen users | gru_two_tower | 0.561 ± 0.000 | 0.651 ± 0.001 | 0.625 ± 0.002 | 0.575 ± 0.001 | 0.021 ± 0.001 |
| unseen users | global_mean | 0.596 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.627 ± 0.000 | 0.027 ± 0.000 |
| unseen users | item_mean | 0.569 ± 0.000 | 0.617 ± 0.000 | 0.622 ± 0.000 | 0.596 ± 0.000 | 0.017 ± 0.000 |
| unseen users | user_item_bias | 0.569 ± 0.000 | 0.617 ± 0.000 | 0.622 ± 0.000 | 0.596 ± 0.000 | 0.017 ± 0.000 |

Full ranking of all 3,327 items for the 283 env-test users of the fully-observed matrix (no exposure bias; history = last 20 real views before 2020-08-10).

| model | NDCG@10 ↑ | NDCG@50 ↑ | Spearman ↑ | mean true watch ratio of top-10 ↑ |
|---|---|---|---|---|
| gru_two_tower | 0.875 ± 0.004 | 0.799 ± 0.002 | 0.576 ± 0.001 | 1.744 ± 0.006 |
| item_mean | 0.877 ± 0.000 | 0.808 ± 0.000 | 0.602 ± 0.000 | 1.761 ± 0.000 |
| random | 0.413 ± 0.001 | 0.418 ± 0.001 | -0.000 ± 0.001 | 0.824 ± 0.004 |
| oracle (true values) | 1 | – | – | 2.000 |

GRU on the FO matrix: engaged AUC 0.781 ± 0.002, ECE 0.023 ± 0.005, RMSE 0.420 ± 0.000.
Compute: 0.10 GPU-hours per seed (RTX 4060 Laptop).
