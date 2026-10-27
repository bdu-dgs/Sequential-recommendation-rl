#!/bin/bash
# Submit steps 5-8 as a dependency chain of SLURM array jobs.
#   source ris/env.sh && bash ris/submit_all.sh            # everything
#   bash ris/submit_all.sh s6_online                       # (re)submit a single stage, no dependencies
set -euo pipefail
cd "$WORKDIR"
mkdir -p logs
COMMON=(--account="$RIS_ACCOUNT" --container-image="$CONTAINER"
        --container-mounts="$WORKDIR:$WORKDIR" --container-workdir="$WORKDIR"
        --export=ALL,WORKDIR="$WORKDIR",RUNS_PER_GPU="$RUNS_PER_GPU")
P=$RUNS_PER_GPU

n_tasks() {  # number of array tasks for a stage file
  local n; n=$(grep -c . "experiments/jobs/$1.txt"); echo $(( (n + P - 1) / P ))
}

submit_stage() {  # $1 stage, $2 time limit, $3 dependency job id (optional)
  local dep=()
  [ -n "${3:-}" ] && dep=(--dependency=afterany:"$3")
  sbatch --parsable "${COMMON[@]}" "${dep[@]}" --partition="$GPU_PARTITION" --time="$2" \
    --job-name="seqrl-$1" --array=1-"$(n_tasks "$1")" --export=ALL,WORKDIR="$WORKDIR",STAGE="$1",RUNS_PER_GPU="$P" \
    ris/run_array.sbatch
}

if [ $# -ge 1 ]; then
  [ -f "experiments/jobs/$1.txt" ] || { echo "run ris/setup.sbatch first"; exit 1; }
  submit_stage "$1" "${2:-04:00:00}"; exit 0
fi

SETUP=$(sbatch --parsable "${COMMON[@]}" ris/setup.sbatch)
# job lists are written by setup.sbatch; wait for it before computing array sizes
echo "setup job $SETUP — waiting for it to write experiments/jobs/ ..."
while squeue -j "$SETUP" -h 2>/dev/null | grep -q .; do sleep 20; done
[ -f experiments/jobs/s5a_data.txt ] || { echo "setup failed; see logs/seqrl-setup-$SETUP.err"; exit 1; }

A=$(submit_stage s5a_data 01:00:00)
B=$(submit_stage s5b_offline 04:00:00 "$A")
C=$(submit_stage s6_online 03:00:00 "$B")
D=$(submit_stage s7a_data 01:00:00 "$A")
E=$(submit_stage s7b_offline 04:00:00 "$D")
F=$(submit_stage s7c_online 03:00:00 "$B:$E")   # needs base CQL/BC (B) and ablation inits (E)
G=$(submit_stage s8a_online 03:00:00 "$B")
H=$(submit_stage s8b_eval 02:00:00 "$C")
echo "submitted: data=$A offline=$B online=$C abl_data=$D abl_offline=$E abl_online=$F exploit=$G eval=$H"
echo "monitor: squeue --me ; logs in logs/<stage>/lineN.log ; results in experiments/runs/"
