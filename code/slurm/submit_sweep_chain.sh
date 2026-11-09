#!/bin/bash
# Runs a grid on n_shards GPUs as a chain of n_links job arrays, each up to 8 hours. Shard i of each array starts
# when shard i of the previous array finishes, and continues its unfinished runs. Slurm's sbatch is used directly
# because `submit` cannot express job arrays or dependencies.
# Usage: ./submit_sweep_chain.sh <grid> <n_shards> <n_links> <account> <gpu type> [extra sweep.py args]
# Example: ./submit_sweep_chain.sh main 8 12 torch_pr_421_cilvr a100 --lr_json $RUNS_DIR/lr_T512.json
# Set AFTER=<job id> to start the chain only when that job array (with the same number of shards) finishes.
set -e
cd "$(dirname "$0")"
source ./env.sh
mkdir -p logs "$RUNS_DIR"
GRID=$1; N_SHARDS=$2; N_LINKS=$3; ACCOUNT=$4; GPU_TYPE=$5; shift 5
previous=${AFTER:-}
for link in $(seq 1 "$N_LINKS"); do
    dependency=${previous:+--dependency=aftercorr:$previous}
    previous=$(sbatch --parsable --array=0-$((N_SHARDS - 1)) --account="$ACCOUNT" --gres=gpu:"$GPU_TYPE":1 \
        --constraint="$GPU_TYPE" $dependency run_sweep.sbatch "$GRID" "$N_SHARDS" "$@")
    echo "link $link: job $previous"
done
