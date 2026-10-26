#!/bin/bash
# One array task = one shard of a grid, with several runs sharing the GPU.
# Usage: sbatch --array=0-7 run_sweep.sbatch <grid> <n_shards> [extra sweep.py args]
#SBATCH --job-name=lc-sweep
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100|l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --comment="preemption=yes;requeue=true"

# finished runs are skipped, so a preempted and requeued shard resumes where it left off
source "$SLURM_SUBMIT_DIR/env.sh"
GRID=$1; N_SHARDS=$2; shift 2
cd "$CODE_DIR"
python sweep.py --grid "$GRID" --out "$RUNS_DIR/$GRID" --shard "$SLURM_ARRAY_TASK_ID" --n_shards "$N_SHARDS" \
    --parallel 4 "$@"
