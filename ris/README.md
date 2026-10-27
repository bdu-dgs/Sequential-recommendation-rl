# Running steps 5–8 on WashU RIS (compute2)

Steps 1–4 (data, supervised model, simulator, offline dataset, smoke tests) were run on the laptop. Steps 5–8 are
packaged as SLURM array jobs. The scripts follow the `ris-tips` tutorial (Pyxis containers, `general-gpu`,
`compute2-<PI>` account).

> **Not yet tested on RIS.** I could not log in (it needs your VPN and SSH). Every command they run was
> smoke-tested locally. Run the setup job first and check its log before submitting everything.

## 0. Prerequisites (you)
1. Connect to the WashU VPN, then `ssh <wustl-id>@c2-login-001.ris.wustl.edu`.
2. Find your account: `sacctmgr show user $(whoami) withassoc format=account%30,partition%30`.
   Use `compute2-<PI-name>`; the workshop account gets no GPUs.

## 1. Copy the project to RIS storage (from the laptop, Git Bash)
The code is small. Copy the processed data and the trained response model too, so RIS does not redo steps 1–4
(about 0.7 GB):
```bash
cd /g/sequential-recommendation-rl
DEST=<wustl-id>@c2-login-001.ris.wustl.edu:/storage2/fs1/<PI-name>/Active/<wustl-id>/sequential-recommendation-rl
rsync -av --exclude data/raw --exclude 'experiments/runs/*smoke*' --exclude ris-tips \
      --exclude .git ./ "$DEST/"
```
This includes `data/processed/`, `data/offline/` (the seed-0 dataset), and
`experiments/runs/response_model*/` (the logging-policy scorer and the learned-simulator predictions).
Without `rsync`, use `scp -r` on the same folders.

## 2. Configure and submit (on RIS)
```bash
cd /storage2/fs1/<PI-name>/Active/<wustl-id>/sequential-recommendation-rl
nano ris/env.sh            # set RIS_ACCOUNT and WORKDIR
source ris/env.sh
bash ris/submit_all.sh     # setup job (deps + tests + job lists), then 8 dependent array stages
squeue --me                # monitor
```
To run under `tmux`/`screen`: `submit_all.sh` waits on the login node (lightweight) until the setup job has
written `experiments/jobs/`.

## 3. What gets submitted
| Stage | Step | Commands | Runs per GPU | Est. time per run | Est. total GPU-run-hours |
|---|---|---|---|---|---|
| s5a_data | 5 | 5 dataset generations | 4 | ~2 min | <0.5 |
| s5b_offline | 5 | 9 methods × 5 seeds = 45 | 4 | 15–30 min (learned) | ~10 |
| s6_online | 6 | 13 methods × 5 seeds = 65 | 4 | ~10 min | ~11 |
| s7a_data | 7 | 25 datasets | 4 | ~2 min | <1 |
| s7b_offline | 7 | 70 | 4 | 15–30 min | ~17 |
| s7c_online | 7 | 225 | 4 | ~10 min | ~38 |
| s8a_online | 8 | 20 (learned-simulator training) | 4 | ~10 min | ~3.5 |
| s8b_eval | 8 | 10 (stress tests + OPE) | 4 | ~10 min | ~2 |

That is about 80 GPU-run-hours. With 4 runs packed per GPU it needs about 20 GPU-hours, and wall-clock time
depends on how many array tasks run at once. Each run needs about 0.3 GB of GPU memory and 2 CPU threads.

## 4. Failures, preemption, resubmission
- Every run checkpoints (model, optimizer, target net, online replay, RNG). Re-running the same command
  resumes it, and a finished run only repeats its final evaluation. Resubmitting a stage is therefore safe:
  `bash ris/submit_all.sh s7c_online`.
- Per-command logs: `logs/<stage>/line<N>.log`; the line number refers to `experiments/jobs/<stage>.txt`.
- Downstream stages use `afterany` dependencies. If an upstream run failed, its dependents will fail fast
  with "run scripts/run_offline.py ... first". Fix it, then resubmit those stages.

## 5. Bring results back
```bash
rsync -av --include '*/' --include '*.json' --include '*.jsonl' --include '*.yaml' --exclude '*' \
      "$DEST/experiments/runs/" experiments/runs/
```
The metrics, logs and configs are enough for the analysis and report; checkpoints can stay on RIS.
