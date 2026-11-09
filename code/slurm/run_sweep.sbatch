#!/bin/bash
# One array task = one shard of a grid, with several runs sharing the GPU.
# Usage: sbatch --array=0-7 run_sweep.sbatch <grid> <n_shards> [extra sweep.py args]
#SBATCH --job-name=lc-sweep
#SBATCH --account=torch_pr_421_courant
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --comment="preemption=yes;requeue=true"

# 4 runs on one GPU use about 3 CPU cores and 7 GB of memory.
# Finished runs are skipped and unfinished runs resume from their last evaluation. A shard stops itself before
# the time limit and exits successfully, so the next copy in a chain (submit_sweep_chain.sh) continues it.
# A preempted shard is requeued by Slurm.
source "$SLURM_SUBMIT_DIR/env.sh"
GRID=$1; N_SHARDS=$2; shift 2
in_env timeout 465m python sweep.py --grid "$GRID" --out "$RUNS_DIR/$GRID" --shard "$SLURM_ARRAY_TASK_ID" \
    --n_shards "$N_SHARDS" --parallel 4 "$@"
exit_code=$?
if [ $exit_code -eq 124 ]; then  # exit code of timeout when it stops the sweep
    echo "time limit reached; the next job in the chain continues shard $SLURM_ARRAY_TASK_ID"
    exit 0
fi
exit $exit_code
