Q over-estimation = predicted Q(s,a) − true discounted return of (s,a) under the greedy policy (exact in the deterministic simulator), on val users. γ = 0.95. p = logging probability.

| method | seeds | return (start → end) | over-estimation overall (start → end) | share of recs with 1e-4 ≤ p < 1e-2 (end) | over-est. in that bucket (end) | share with p < 1e-4 (end) | over-est. p ≥ 1e-2 (end) |
|---|---|---|---|---|---|---|---|
| naive fine-tune (no penalty) | 5 | 37.0 → 21.7 | +9.2 → +20.3 | 0.04 | +22.3 | 0.90 | +0.9 |
| constant CQL | 5 | 37.0 → 36.8 | +9.2 → -0.6 | 0.00 | +1.6 | 0.01 | -0.6 |
| support-weighted CQL (oracle) | 5 | 37.0 → 36.9 | +9.2 → +2.6 | 0.15 | +4.5 | 0.00 | +2.3 |
| uniform, strength-matched | 5 | 37.0 → 37.2 | +9.2 → +1.4 | 0.04 | +5.6 | 0.00 | +1.2 |
