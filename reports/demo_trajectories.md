# Demo: four policies, one held-out user (simulated, KuaiRecFO-v1)

Seed 0, stored test trajectory #0. reward = true simulated reward of the step (capped real watch ratio × synthetic fatigue); p_log = probability that the logging policy would have recommended this item in this state. Episode return is the sum of rewards (max 60).

## offline CQL (no fine-tuning) — return 53.3, 30 steps (user row 2)

| t | video_id | category | reward | p_log |
|---|---|---|---|---|
| 0 | 5464 | food | 2.00 | 4.4e-02 |
| 1 | 8366 | other | 1.95 | 4.5e-02 |
| 2 | 8524 | art | 2.00 | 7.0e-02 |
| 3 | 8298 | looks | 1.88 | 8.2e-02 |
| 4 | 4123 | local news | 1.94 | 8.3e-02 |
| 5 | 7383 | sports | 2.00 | 6.7e-02 |
| 6 | 2343 | fashion | 1.63 | 3.6e-02 |
| 7 | 314 | fashion | 1.88 | 5.3e-02 |
| 8 | 154 | sports | 1.88 | 6.1e-02 |
| 9 | 5525 | sports | 1.76 | 5.9e-02 |
| 10 | 600 | cars | 2.00 | 3.4e-02 |
| 11 | 10206 | pets | 2.00 | 7.9e-02 |

## fine-tune, constant CQL — return 53.4, 30 steps (user row 2)

| t | video_id | category | reward | p_log |
|---|---|---|---|---|
| 0 | 8298 | looks | 2.00 | 6.8e-02 |
| 1 | 8366 | other | 1.83 | 4.4e-02 |
| 2 | 4123 | local news | 1.94 | 6.4e-02 |
| 3 | 7383 | sports | 2.00 | 5.6e-02 |
| 4 | 8524 | art | 2.00 | 8.0e-02 |
| 5 | 5464 | food | 2.00 | 6.6e-02 |
| 6 | 314 | fashion | 2.00 | 5.1e-02 |
| 7 | 154 | sports | 1.88 | 5.3e-02 |
| 8 | 2343 | fashion | 1.53 | 4.4e-02 |
| 9 | 5525 | sports | 1.88 | 5.9e-02 |
| 10 | 600 | cars | 2.00 | 3.2e-02 |
| 11 | 4040 | cars | 1.88 | 5.9e-02 |

## fine-tune, no penalty (naive) — return 19.2, 21 steps (user row 2)

| t | video_id | category | reward | p_log |
|---|---|---|---|---|
| 0 | 592 | local news | 1.17 | 5.1e-04 |
| 1 | 6028 | food | 0.74 | 1.6e-05 |
| 2 | 959 | lifestyle | 0.68 | 1.8e-05 |
| 3 | 4386 | fitness | 1.15 | 7.5e-05 |
| 4 | 9612 | local news | 0.45 | 6.2e-05 |
| 5 | 947 | lifestyle | 0.84 | 9.0e-05 |
| 6 | 5745 | travel | 1.53 | 5.0e-05 |
| 7 | 9700 | celebrities | 0.65 | 1.9e-05 |
| 8 | 136 | other | 0.89 | 5.1e-05 |
| 9 | 3814 | food | 0.65 | 1.5e-05 |
| 10 | 7092 | food | 0.93 | 1.6e-05 |
| 11 | 2739 | comedy | 0.67 | 1.7e-05 |

## fine-tune, support-weighted CQL — return 49.1, 30 steps (user row 2)

| t | video_id | category | reward | p_log |
|---|---|---|---|---|
| 0 | 8366 | other | 1.95 | 3.8e-02 |
| 1 | 6870 | local news | 1.26 | 4.3e-03 |
| 2 | 5525 | sports | 2.00 | 3.8e-02 |
| 3 | 2178 | art | 1.22 | 1.9e-03 |
| 4 | 8464 | local news | 0.00 | 2.2e-03 |
| 5 | 4123 | local news | 1.70 | 6.3e-02 |
| 6 | 10062 | sports | 1.82 | 2.9e-03 |
| 7 | 7259 | food | 1.11 | 8.1e-04 |
| 8 | 5995 | food | 1.66 | 1.9e-02 |
| 9 | 2472 | local news | 1.22 | 1.9e-03 |
| 10 | 2172 | pets | 1.76 | 1.0e-03 |
| 11 | 4065 | looks | 1.30 | 1.1e-03 |

