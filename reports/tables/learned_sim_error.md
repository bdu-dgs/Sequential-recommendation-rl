Error of the learned response model (real-data GRU, seed 0) against the true FO watch ratio (capped at 2), for all 1,411 FO users, grouped by how often each item appears in the simulated offline log.

| item log count (seed-0 offline data) | items | mean error (pred − true) | RMSE | share of items with pred > true + 0.5 |
|---|---|---|---|---|
| 0 | 57 | +0.008 | 0.326 | 0.02 |
| 1–9 | 2057 | -0.009 | 0.388 | 0.02 |
| 10–99 | 998 | -0.012 | 0.478 | 0.03 |
| ≥100 | 215 | -0.026 | 0.514 | 0.00 |
