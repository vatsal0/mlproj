#!/bin/bash
# Submits the checkpoint MVP as a chain: lr grid -> pick lr per size -> scale sweep.
# Run from code/slurm after editing env.sh and the --account lines.
set -e
cd "$(dirname "$0")"
source env.sh
mkdir -p logs "$RUNS_DIR"
LR=$(sbatch --parsable --array=0-3 run_sweep.sbatch lr 4)
PICK=$(sbatch --parsable --dependency=afterok:$LR pick_lr.sbatch)
MVP=$(sbatch --parsable --array=0-7 --dependency=afterok:$PICK run_sweep.sbatch mvp 8 --lr_json $RUNS_DIR/lr.json)
echo "lr: $LR  pick: $PICK  mvp: $MVP"
