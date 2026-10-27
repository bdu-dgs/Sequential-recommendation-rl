Q over-estimation = predicted Q(s,a) − true discounted return of (s,a) under the greedy policy (exact in the deterministic simulator), on val users. γ = 0.95. p = logging probability.

| method | seeds | return (start → end) | over-estimation overall (start → end) | share of recs with 1e-4 ≤ p < 1e-2 (end) | over-est. in that bucket (end) | share with p < 1e-4 (end) | over-est. p ≥ 1e-2 (end) |
|---|---|---|---|---|---|---|---|
| naive fine-tune (no penalty) | 3 | 44.9 → 15.8 | +11.3 → +51.4 | 0.19 | +49.3 | 0.77 | +1.7 |
| constant CQL | 3 | 44.9 → 44.5 | +11.3 → +0.9 | 0.03 | +3.6 | 0.00 | +0.8 |
| support-weighted CQL (oracle) | 3 | 44.9 → 35.9 | +11.3 → +33.3 | 0.70 | +38.1 | 0.01 | +12.7 |
| uniform, strength-matched | 3 | 44.9 → 44.5 | +11.3 → +2.2 | 0.07 | +5.4 | 0.00 | +2.0 |
