# Project status

## 2026-10-26 (final) — all planned work complete; report and resume evidence written
- **Deliverables:**
  - [`reports/research_report.md`](../reports/research_report.md): the full report;
  - [`docs/resume_evidence.md`](resume_evidence.md): resume bullets with per-number provenance;
  - [`docs/interview_notes.md`](interview_notes.md);
  - [`reports/demo_trajectories.md`](../reports/demo_trajectories.md).
- **Run totals:** 217 recorded runs, 24.8 GPU-process-hours, about 9 h of wall-clock time on the laptop, with
  0 failed runs in the final chains (`reports/tables/compute.md`).
- **Narrow coverage** (5 seeds, `*_narrow`):
  - support weighting is still worse than the uniform-tracked control (drop −1.18 vs −0.36), but the gap shrinks
    from about 9 points to about 0.8;
  - mid-support overestimation is only +4.5;
  - this fits the mechanism (`tables/overestimation_narrow.md`).
- **Step 8** (5 seeds):
  - **Learned-simulator overstatement:** +10.4 for scratch DQN, +8.4 naive fine-tuning, +5.3 support-weighted,
    +0.4 constant CQL;
  - **Stress tests:** drift, cold start, noise and fatigue leave the method ranking unchanged;
  - **OPE:** about 1% error for supported targets; IPS 82% and DR 23% error for offline DQN (85% unsupported
    mass, ESS 6).
- **Docs corrected:**
  - `environment_assumptions.md`: the learned-simulator error is *not* larger on rare items;
  - `decisions.md`: all runs were local, and the step-7 scope was cut;
  - `experiment_plan.md`: what actually ran.
- **Not done:** broad coverage, environment-mechanism variations, α sweep, inheritance ablations. These are listed
  in report §9.
- **Possible next step**, if the project continues: a count- or uncertainty-based conservatism weight, instead of
  probability relative to uniform.


## 2026-10-25 — steps 1–4 done locally; steps 5–8 packaged for RIS

### Done
- **Step 1, docs:**
  - [literature_review.md](literature_review.md) (107 verified references, novelty check);
  - [data_audit.md](data_audit.md);
  - [problem_formulation.md](problem_formulation.md);
  - [environment_assumptions.md](environment_assumptions.md);
  - [decisions.md](decisions.md);
  - [experiment_plan.md](experiment_plan.md).
- **Step 2, agents and tests:** Double DQN, discrete CQL (support-weighted generalization), Cal-QL variant,
  discrete BCQ, BC, reward-regression ranker, NeuralLinear-UCB bandit, and the popularity/random baselines.
  The 13 unit tests pass:
  - toy DP optimum, and DQN reaching more than 90% of it;
  - CQL staying in support on narrow logs;
  - exact next-observation reconstruction;
  - return-to-go and the terminal target;
  - evaluation never updates parameters;
  - simulator reward, fatigue, patience and horizon, and no future information in observations.

  Bug found and fixed: buffer episode ids.
- **Step 3, real-data supervised model** (`experiments/runs/response_model_wd/seed0`; seeds 1–2 running):
  - Temporal test, unseen users: engaged AUC 0.651 vs 0.617 for item mean. RMSE 0.561 vs 0.570.
  - Fully-observed full ranking: NDCG@10 0.869, **below** item mean at 0.877. Engaged AUC is 0.784 and
    ECE 0.023.
  - Honest reading: item quality dominates KuaiRec, and sequential personalization adds little to ranking.
- **Step 4, simulator and data:**
  - KuaiRecFO-v1 reference returns on validation users: random 9.7, logging policy 37.0 (greedy 43.9),
    oracle 59.5, out of a maximum of 60.
  - Offline dataset: 200k transitions with exact propensities. Effective support is about 186 items per state;
    1,213 items are logged 10 or more times.
- **Steps 5–8 code:** all scripts were smoke-tested locally at tiny budgets:
  - `run_offline.py` (9 methods), `run_online.py` (12 methods);
  - `evaluate_checkpoints.py` (7 stress settings), `run_ope.py` (IPS/SNIPS/DM/DR);
  - `make_jobs.py` (465 commands), and the `ris/` SLURM scripts.

  Smoke numbers are not results.

### Pre-cluster code review (independent agent) — done, fixes applied
- **Critical, fixed:** `eval.response=fo` was ignored, so learned-simulator runs would have been evaluated in
  the learned simulator. `Setup.eval_fn` now merges `cfg.eval` simulator keys, and learned-sim runs also report
  `test_in_training_sim`.
- **Major, fixed:** the matched-uniform control was matched only at the start, while the oracle weights decay
  as online counts grow. Added `ft_support_uniform_tracked`, which is re-matched to the oracle's mean strength
  at every update. It is now the primary control in the decision rule.
- **Minor, fixed:**
  - the shuffled control now uses a fixed item permutation;
  - buffer load pointer;
  - atomic checkpoints (DONE is written last);
  - dataset names hash every data-relevant setting;
  - eval noise is reseeded on every call;
  - BC-head gradients are clipped;
  - leave rate comes from the environment;
  - cold-start validation;
  - missing checkpoint keys raise an error;
  - CUDA device detection.
- **Known, documented:** on resume, running episodes keep NaN return-to-go and the env user stream restarts.
  The effect is negligible but not bit-exact.
- 16 tests pass. The full smoke suite (`experiments/smoke/smoke_all.txt`) passes on the fixed code.
- Response model, 3 seeds (`response_model_wd`), unseen users: AUC 0.651 ± 0.001 vs 0.617 for item mean.
  Fully-observed NDCG@10 0.875 ± 0.005 vs 0.877 for item mean.

### 2026-10-25 evening — steps 5–6, seed 0 run locally (user chose the laptop over RIS for steps 5–6)
- The first attempt was stopped: runs degraded with training (memorization and winner's curse on rarely logged
  items). The protocol was changed **before results were used** (`decisions.md`), and the first attempt is
  archived in `experiments/runs/archive_seed0_diagnostics/`.
- Re-run: seed 0 took 72 min (step 5: 38 min, step 6: 34 min; 3 in parallel; GPU 58–69 °C).
- **Preliminary single-seed results (simulated returns, max 60; `reports/tables/*.md`):**
  - **Offline:**
    - CQL 44.5, Cal-QL 44.7, BCQ 44.8: stable, and 0% of chosen actions are unsupported.
    - Offline DQN 24.7: most chosen actions have logging prob < 1e-4.
    - Ranker 43.8 at the selected checkpoint, 31.9 at the last step.
    - Log popularity 44.1; logging policy 36.6.
  - **Fine-tuning from CQL** (validation start 44.8):
    - Naive: worst drop −33.9, ends at 18.2.
    - Decayed α: −20.9, ends at 23.9.
    - Constant CQL and Cal-QL: stable at about 44.7.
    - **Support-weighted with oracle support: −10.2, ends at 41.1, worse than the uniform-tracked control
      (−0.6).** BC-estimated support: −8.5. Shuffled: −11.9. Inverted: 0.
    - Bandit: best at 45.3.
  - The pre-registered hypothesis is **not supported on seed 0**. Working explanation: online counts
    (λ = 1) remove the penalty from items the agent has tried after only a few visits. Diagnostics (λ = 0,
    κ ∈ {0.3, 3}) and seeds 1–4 are running (`logs/local/chain_progress.txt`).

### 2026-10-26 — steps 5–6 complete, 5 seeds (local, 97 runs, 0 failures, about 3.6 h)
Tables: `reports/tables/offline.md` and `reports/tables/online.md`; figure:
`reports/figures/finetune_curves.png`. Simulated test return, mean [95% bootstrap CI]:
- **Offline:**
  - Conservative methods: CQL 44.56 [44.36, 44.76], Cal-QL 44.84 [44.72, 44.96], BCQ 44.72 [44.57, 44.86],
    BC 44.17. 0% of their chosen actions are unsupported.
  - Offline DQN: 21.58 [17.96, 25.33], with 66% of chosen actions unsupported.
  - Popularity 44.06; logging policy 36.61; random 9.41.
- **Fine-tuning from CQL** (start 44.79):
  - naive DQN: worst drop −34.69 [−38.29, −30.34], test 15.94;
  - decayed α: −20.11, test 24.24 (collapses once α → 0, after about 35k transitions);
  - constant CQL: −0.74, test 44.21;
  - Cal-QL: −0.28, test 44.82;
  - **oracle-support weighting: −9.98 [−11.86, −7.38], test 36.08 [32.87, 39.45]**;
  - BC-estimated support: −8.90, test 37.81;
  - uniform-tracked control: −0.91, test 44.03;
  - shuffled: −15.78; inverted: −0.75.
- **Bandit:** best final return, test 45.46 [45.19, 45.84]. Scratch online DQN: 16.59.
- **Pre-registered decision rule:** support weighting must beat the uniform-tracked and shuffled controls. It
  beats shuffled but is clearly **worse** than uniform-tracked, so the **hypothesis is rejected** in this
  setting.
- **Diagnostics** (seed 0):
  - With λ = 0 (no online counts) the drop is still −8.7, so fast relaxation on tried items is not the main
    cause.
  - Working explanation: relaxing the penalty on high-support actions lets their values drift upward, but the
    logging policy's favorite items are not the best ones (the logging policy is worse than popularity).
    "Supported" is not the same as "safe to relax".

### 2026-10-26 — agreed follow-up (user): (1) mechanism analysis, (2) narrow-coverage ablation, (3) step 8
- (1) **Running:** `experiments/jobs/diag.txt` = 4 fine-tuning methods × seeds 0–2, `--tag diag`. Evaluation now
  records predicted Q vs the true discounted return of each greedy recommendation, bucketed by the logging
  probability (`evaluate()` → `q_diag`). Plot: `python scripts/plot_overestimation.py --tag diag`.
  - **Done (3 seeds, 12 runs, 0 failures):** `reports/figures/overestimation_by_support.png`,
    `reports/tables/overestimation.md`. Overestimation (predicted Q minus true value), start → end:
    - naive: +11.3 → +51.4, with 77% of recommendations on unsupported items;
    - constant CQL: → +0.9; uniform-tracked: → +2.2 (online data corrects the initial overestimation);
    - **support-weighted (oracle): → +33.3, with 70% of recommendations on mid-support items
      (1e-4 ≤ p < 1e-2) overestimated by +38.1.**
    - Mechanism confirmed: the weighting rule treats items with p of about 1e-3 (about 3× uniform) as
      supported and removes their penalty, but the log holds only a few samples of each.
  - First look (seed 0, final checkpoints): oracle-support fine-tuning sends 44% of recommendations to
    mid-support items (1e-4 ≤ p < 1e-2) that are overestimated by +15.4. Constant CQL sends 98% to
    high-support items with +0.5. Naive fine-tuning sends 70% to unsupported items with +36.6.
- (2) Done later (see the final entry above). Command: `python scripts/run_local.py --stages s7a_data s7b_offline s7c_online --grep "head_size=500"`
  (about 2 h). The grep matches the data, offline and online lines of the narrow regime.
- (3) Done later (see the final entry above): `python scripts/run_local.py --stages s8a_online s8b_eval` (about 1–1.2 h).
- Resume after a shutdown: re-running the same command is safe, because finished runs only repeat their final
  evaluation.

### Next — superseded: everything ran locally (see the final entry at the top)

### Compute (local RTX 4060 Laptop)
See `reports/tables/compute.md`: 217 runs, 24.8 GPU-process-hours.
