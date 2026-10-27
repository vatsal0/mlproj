# Edit these for your Torch setup.
CODE_DIR=$HOME/mlproj/code
RUNS_DIR=/scratch/$USER/mlproj/runs
export WANDB_PROJECT=long-context-scaling-mlproj  # train.py logs to wandb when this is set
export WANDB_DIR=/scratch/$USER/mlproj/wandb
# runs a command inside the singularity container with the conda env (torch, numpy, matplotlib, wandb) active
in_env() {
    singularity exec --nv /scratch/$USER/containers/miniconda3_latest.sif /bin/bash -c \
        'source ~/.bashrc && conda activate metagrad && cd "$0" && "$@"' "$CODE_DIR" "$@"
}
