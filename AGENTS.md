# AGENTS.md — working conventions for this repository

This file is for humans and coding agents who continue the project.

## Ground rules
- **Never fabricate results, propensities, online experiments, or product impact.** Every number in `reports/` or
  `docs/resume_evidence.md` must be traceable to a run directory under `experiments/runs/` (config + metrics JSON + command).
- **Real vs simulated.** Anything produced by an environment in `src/seqrec_rl/environments/` is a *simulated return*.
  Never describe it as user growth, online CTR, watch time, or revenue.
- **Propensities.** KuaiRec logs carry no logging-policy probabilities. IPS/WIS/DR are only run on simulated logs whose
  behavior policy and probabilities we generated ourselves (`src/seqrec_rl/replay/logging_policy.py`).
- **No post-recommendation information in the state.** Observations may only contain items/feedback revealed before the
  current decision; see `tests/test_env_no_leakage.py`.

## Layout
- `src/seqrec_rl/` — library code (data, user_models, environments, agents, replay, training, evaluation).
- `scripts/` — CLI entry points (the only supported way to run things; notebooks in `demos/` are illustrations).
- `configs/` — YAML configs; every run copies its resolved config into its run directory.
- `experiments/runs/<name>/<seed>/` — run outputs (metrics, logs, checkpoints). Git-ignored.
- `reports/` — generated tables/figures + research report (tracked).
- `docs/` — problem formulation, data audit, assumptions, literature, plan, decisions log, status.

## Environment
- Python env: `G:\Anaconda\envs\seqrl` (Python 3.11, torch 2.5.1+cu121). Install the package with
  `pip install -e .[dev]`.
- Data lives in `data/` (git-ignored). `python scripts/download_kuairec.py` fetches KuaiRec 2.0 from Zenodo.
- Tests: `python -m pytest -q` (data-dependent tests are marked `data` and skip if processed data is absent).

## Conventions
- Seeds: every script takes `--seed`; main results use seeds 0-4.
- Runs are resumable: training loops checkpoint (model, optimizer, target net, replay buffer, RNG state) and resume
  from the latest checkpoint if the run directory exists.
- Record non-trivial technical choices in `docs/decisions.md`; update `docs/status.md` after each milestone.
