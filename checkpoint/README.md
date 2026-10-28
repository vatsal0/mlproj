# Checkpoint: code and results

Vatsal Baherwani, Ammar Ali, Abhishek Muhuri

- `checkpoint/checkpoint.md`: the checkpoint write-up. `checkpoint/figures/` holds its figures, and the `.txt` files hold the result tables it cites.
- `code/`: the full pipeline.
  - `tasks.py` generates MQAR and needle-in-a-haystack batches.
  - `model.py` defines the Llama-style transformer, the five model sizes, and the FLOP count.
  - `train.py` trains one run and evaluates it every 250 steps.
  - `sweep.py` defines the named grids of runs.
  - `analyze.py` makes the tables and figures.
  - `slurm/` holds our cluster job scripts.
- `results/`: one folder per run, each with `summary.json` (configuration, steps and FLOPs to 90%, final accuracy) and `log.jsonl` (one line per evaluation). Model weights are not included.

| Folder | Experiment | Runs |
|---|---|---|
| `results/lr` | learning rate sweep at T = 256 | 30 |
| `results/mvp` | scale sweep: 4 sizes, T = 128 to 1024, 2 tasks, 3 seeds | 96 |
| `results/calibration` | 3.2M and 25M models at T = 2048 | 4 |
| `results/lr_full` | learning rate sweep at T = 512, all 5 sizes | 38 |

`results/lr.json` and `results/lr_T512.json` hold the chosen learning rate for each size.

To reproduce the tables and figures (needs numpy and matplotlib):

```
python code/analyze.py results/mvp                  # table behind Table 1, and Figures 2 to 4 in results/mvp/figures
python code/analyze.py results/mvp --threshold 0.5  # the same at a 50% accuracy threshold
python code/analyze.py results/lr                   # learning rate sweep table
```

Figure 1 comes from a pilot run on a laptop, whose logs are not included.

To rerun experiments (needs PyTorch, and a GPU for anything beyond the single run below), from inside `code/`:

```
python train.py --task mqar --T 64 --size s --max_steps 5000   # one run; reaches 90% in about 10 minutes on a laptop
python sweep.py --grid mvp --out ../runs/mvp --lr_json ../results/lr.json --parallel 4
```
