# Edit these once on RIS, then: source ris/env.sh
export RIS_ACCOUNT="compute2-<PI-name>"                                  # sacctmgr show user $(whoami) withassoc
export WORKDIR="/storage2/fs1/<PI-name>/Active/<wustl-id>/sequential-recommendation-rl"
export CONTAINER="docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
export GPU_PARTITION="general-gpu"            # or general-preempt-gpu (jobs are resumable)
export RUNS_PER_GPU=4                         # independent small runs packed on one GPU (each needs ~0.3 GB)
