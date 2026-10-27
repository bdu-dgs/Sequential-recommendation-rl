# Data audit: KuaiRec 2.0

All numbers below come from `scripts/prepare_data.py` (written to `data/processed/manifest.json`) and from
exploratory queries run on 2026-10-24 and 2026-10-25. Raw-file sha256 values are stored in the manifest.

## Version, access, license
- **Source:** KuaiRec 2.0, Zenodo record 18164998 (linked from https://github.com/chongminggao/KuaiRec).
  The zip sha256 is `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`. Download is anonymous,
  with no account or agreement.
- **License:** CC BY-SA 4.0. Derived data are not redistributed by this repo; `data/` is git-ignored and
  `scripts/download_kuairec.py` re-fetches the original.
- **Version 2.0 changes:** video 1225 was removed and the ids above it shifted down by one. Item daily features
  and user features were added.

## Fields
| Table | Rows | Fields | Notes |
|---|---|---|---|
| big_matrix | 12,530,806 | user_id, video_id, play_duration, video_duration, time, date, timestamp, watch_ratio | 7,176 users × 10,728 videos |
| small_matrix | 4,676,570 | same | 1,411 users × 3,327 videos; 99.6% dense ("fully observed") |
| item_categories | 10,728 | video_id, feat (1–4 of 31 tags) | 8,970 items have 1 tag |
| item_daily_features | 343,341 | per (video, day) aggregates incl. like/comment/share counts | **not per user** |
| user_features | 7,176 | activity level, follow/fan ranges, 18 encrypted one-hot features | static snapshot, time of capture unknown |
| social_network | 472 | user_id, friend_list | not used |

`watch_ratio = play_duration / video_duration` is the **only per-user feedback**. It has a heavy tail: median
0.72, p99 4.8, max 573 (loops). There is no per-user like, follow, or share signal.

## Quality issues found
1. **Exact duplicate rows:** 965,819 big-matrix rows (7.7%) are exact duplicates. Another 4,353 rows duplicate
   (user, video, timestamp) and differ only in `date`. We drop duplicates on (user, video, timestamp), which
   leaves **11,562,801** rows.
2. **Genuine re-consumption is kept:** 1,047,830 (user, video) pairs recur at different timestamps (median gap
   about 22 days).
3. **date vs timestamp:** 14,376 rows have `date` one day off from the timestamp in local time (UTC+8). We
   derive the local date from the timestamp.
4. **Small matrix:** 181,992 rows (3.9%) have no timestamp or date. 17,827 (user, item) pairs are absent; the
   KuaiRec authors say these are items the user blocked. They are treated as response 0.
5. **Ten** same-user, same-timestamp ties with different items are ordered by item id.
6. In `kuairec_caption_category.csv`, a few captions contain raw newlines and produce junk rows; we parse ids
   numerically and drop the junk.

## Time coverage
The big matrix covers **three disjoint windows** of local dates: **2020-07-05..07-12**, **08-01..08-10**,
**08-27..09-05**. About 1,000 rows carry stray timestamps outside these windows. The small matrix spans
07-05..09-05 across 63 dates, interleaved in time with the same users' big-matrix rows.

**Split** (global temporal split, as recommended by Ji et al. 2023 and Meng et al. 2020): train ≤ 2020-08-10 <
val ≤ 08-31 < test (09-01..05).

| split | seen users | FO (small) users | unseen users |
|---|---|---|---|
| train | 7,430,194 | 351,879 | 791,663 (never used for training) |
| val | 1,586,290 | 92,637 | 171,786 |
| test | 950,738 | 88,574 | 99,040 |

- **Unseen users:** 10% of the big-only users (576), held out entirely.
- **Simulated population:** the 1,411 FO users, split 60/20/20 → 846 env-train, 282 env-val, 283 env-test
  (seed 20201024).

## Does the data support sequential RL? (what it does and does not provide)
| Requirement | Available? | Evidence / consequence |
|---|---|---|
| Time-ordered per-user logs | **Yes** | about 1,611 rows per user after dedup |
| Exposure records | **Partly** | Rows are viewed videos in an autoplay feed, so each is close to an impression with dwell feedback. There are no candidate sets and no non-viewed exposures. |
| Complete sequences | **No** | The median gap between consecutive logged rows is 129 s, far longer than a short video. Logs look like a subsequence restricted to the 10,728-item pool, so logged "transitions" are incomplete. |
| Session ids | **No** | Inferred with a 30-minute gap: 887k sessions, mean length 13. |
| Logging-policy propensities | **No** | **No IPS/WIS/DR on real logs.** OPE is only run on simulated logs with known probabilities. |
| Counterfactual labels | **Yes, for 1,411 × 3,327** | The FO matrix gives the response to (almost) any item for these users. It is static: one value per pair, with no order or context effects. |
| Leakage between big and FO | **None at pair level** | Big and small matrices share **0** (user, item) pairs. The FO users' big-matrix rows contain **no** catalog items. |

## Statistics that shape the design
- **Item effects dominate.** Variance decomposition of the FO watch ratio (capped at 5): item means 0.12, user
  means 0.023, residual 0.29 (the residual includes irreducible noise, since each pair is observed once). Item
  mean watch ratios from big vs small correlate at **0.99**. Expect strong item-quality baselines.
- **Popularity is not quality.** The 10 most-viewed catalog items have mean FO watch ratio 0.76 and a 38% chance
  of watch ratio < 0.5, vs 0.83 and 27.5% overall.
- **Long tail:** 1,419 / 2,899 / 4,508 of the 10,728 items cover 50 / 80 / 95% of big-matrix rows. 269 catalog
  items have no train-window views.
- **Fatigue and quit dynamics are not identified.** After removing user and item means:
  - Watch ratio barely changes with the number of same-tag items among the previous 5 in a session (+0.008 →
    −0.034). The trend is confounded with position in the session (position alone: +0.07 at position 0 to
    −0.04 at position 50+).
  - Session-end probability is U-shaped in watch ratio and **decreases** with tag repetition (0.088 → 0.023),
    because engaged users get more same-tag content.

  The real logs therefore do not justify a "repeat category → leave" rule (as used by CIRS/DORL). Our simulator
  treats fatigue and leaving as **synthetic, controlled factors** and varies them in ablations
  (see `environment_assumptions.md`).

## Decision
KuaiRec supports (1) supervised response modelling with a leak-free temporal split, and (2) a simulator whose
**immediate rewards are real** (FO matrix) but whose **dynamics are synthetic**. It does not support real-log
OPE or claims about long-term real user effects. Policy results are reported only as simulated returns.
