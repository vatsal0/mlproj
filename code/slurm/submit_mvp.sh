#!/bin/bash
# Submits the checkpoint MVP as a chain: lr grid -> pick lr per size -> scale sweep.
# Run from code/slurm. Paths and the conda env are in env.sh; accounts are in the .sbatch files and below.
set -e
cd "$(dirname "$0")"
source ./env.sh
mkdir -p logs "$RUNS_DIR"
# 6 and 24 shards of 4 runs each, so every run in a grid trains at once (4 per GPU)
LR=$(sbatch --parsable --array=0-5 run_sweep.sbatch lr 6)
PICK=$(sbatch --parsable --dependency=afterok:$LR pick_lr.sbatch)
# the 24 shards are split over two accounts, since one account's group GPU limit is often saturated
MVP_COURANT=$(sbatch --parsable --array=0-11 --dependency=afterok:$PICK run_sweep.sbatch mvp 24 --lr_json $RUNS_DIR/lr.json)
MVP_CILVR=$(sbatch --parsable --account=torch_pr_421_cilvr --array=12-23 --dependency=afterok:$PICK \
    run_sweep.sbatch mvp 24 --lr_json $RUNS_DIR/lr.json)
echo "lr: $LR  pick: $PICK  mvp: $MVP_COURANT $MVP_CILVR"
