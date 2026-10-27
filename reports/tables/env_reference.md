Reference policies in KuaiRecFO-v1 on the 282 val users (simulated return, max 60). Oracle policies read the true response table and are reference points only.

| policy | simulated return | length | leave rate | coverage |
|---|---|---|---|---|
| random | 9.68 | 11.8 | 0.95 | 0.628 |
| popularity_real | 2.30 | 4.6 | 0.98 | 0.009 |
| logging_stochastic | 37.00 | 25.2 | 0.33 | 0.324 |
| logging_greedy (response-model ranker) | 43.94 | 27.3 | 0.21 | 0.016 |
| oracle_greedy_no_fatigue | 56.50 | 30.0 | 0.00 | 0.459 |
| oracle_greedy | 59.50 | 30.0 | 0.00 | 0.471 |

Offline dataset (seed 0): 200,000 transitions, 8,039 episodes; 3,270 items logged at least once, 1,213 logged ≥ 10 times; top-100 items = 73% of the log; logging-policy entropy 4.97 nats (effective support ≈ 186 items per state).
