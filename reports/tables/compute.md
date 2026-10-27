Recorded compute per run (RTX 4060 Laptop GPU, 8 GB). Runs were executed 3 in parallel, so physical GPU time is roughly one third to one half of the GPU-process-hours.

| experiment family | runs | GPU-process-hours |
|---|---|---|
| characterize_env | 1 | 0.02 |
| eval_generalization | 5 | 0.17 |
| offline/base | 45 | 5.40 |
| offline/narrow | 20 | 3.72 |
| online/base | 65 | 6.17 |
| online/diag | 12 | 1.18 |
| online/kappa0.3 | 1 | 0.22 |
| online/kappa3.0 | 1 | 0.22 |
| online/lam0 | 3 | 0.65 |
| online/learnedsim | 20 | 2.52 |
| online/narrow | 35 | 4.03 |
| ope | 5 | 0.03 |
| response_model | 1 | 0.13 |
| response_model_wd | 3 | 0.29 |
| **total** | **217** | **24.76** |
