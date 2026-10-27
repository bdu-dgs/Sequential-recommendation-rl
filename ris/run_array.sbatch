#!/bin/bash
# Runs a block of RUNS_PER_GPU consecutive lines of experiments/jobs/$STAGE.txt in parallel on one GPU.
# Array index k (1-based) -> lines (k-1)*P+1 .. k*P. Each command is resumable/idempotent.
#SBATCH --job-name=seqrl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -uo pipefail
cd "$WORKDIR"
export PYTHONUSERBASE="$WORKDIR/.pydeps" PYTHONPATH="$WORKDIR/src" SEQRL_THREADS=2
P=${RUNS_PER_GPU:-4}
FILE="experiments/jobs/${STAGE}.txt"
start=$(( (SLURM_ARRAY_TASK_ID - 1) * P + 1 ))
end=$(( start + P - 1 ))
mkdir -p "logs/${STAGE}"
fail=0
pids=()
for i in $(seq "$start" "$end"); do
  line=$(sed -n "${i}p" "$FILE")
  [ -z "$line" ] && continue
  echo "[$(date +%T)] line $i: $line"
  bash -c "$line" > "logs/${STAGE}/line${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || true
exit $fail
