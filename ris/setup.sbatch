#!/bin/bash
# One-time: install the Python dependencies that the PyTorch container lacks into $WORKDIR/.pydeps,
# then run the unit tests. Submitted by ris/submit_all.sh (container/account flags are added there).
#SBATCH --job-name=seqrl-setup
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$WORKDIR"
export PYTHONUSERBASE="$WORKDIR/.pydeps" PYTHONPATH="$WORKDIR/src" PIP_CACHE_DIR="$WORKDIR/.pipcache"
python -m pip install --user --quiet pandas pyarrow scipy scikit-learn matplotlib pyyaml tqdm pytest
python -c "import torch, pandas, sklearn; print('torch', torch.__version__, 'pandas', pandas.__version__)"
SEQRL_THREADS=4 python -m pytest -q tests/test_replay.py tests/test_env_no_leakage.py
python scripts/make_jobs.py
