#!/bin/bash
# Picks the best learning rate per size from the lr grid and writes $RUNS_DIR/lr.json.
#SBATCH --job-name=lc-pick-lr
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

source "$SLURM_SUBMIT_DIR/env.sh"
cd "$CODE_DIR"
python analyze.py "$RUNS_DIR/lr" --best_lr "$RUNS_DIR/lr.json"
