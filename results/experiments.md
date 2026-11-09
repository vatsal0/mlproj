# Experiment log (after checkpoint 1)

Metrics for every run are on wandb: https://wandb.ai/vatsalbaherwani/long-context-scaling-mlproj (groups `main` and `architecture`). Each run folder here holds `summary.json` and `log.jsonl`. Each experiment folder has `figures/` with the plots, plus `runs.csv` (one row per run) and `analysis_table.txt` (output of `python analyze.py <folder>`).

## Shared settings

- Tasks: MQAR and needle with 16 key-value pairs.
- Success: 90% held-out accuracy held for 3 evaluations in a row, evaluated every 250 steps.
- Learning rate per size from `lr.json`:

  | Size | Parameters | Learning rate |
  |---|---|---|
  | xs | 0.4M | 1e-3 |
  | s | 3.2M | 3e-4 |
  | m | 10.6M | 3e-4 |
  | l | 25M | 1e-4 |
  | xl | 59.5M | 1e-4 |

- Four runs share each GPU (L40S or A100).
- Runs save their full training state at every evaluation and resume from it. A chain of 8-hour Slurm array jobs (`code/slurm/submit_sweep_chain.sh`) continues unfinished runs.

## Main sweep (grid `main`, folder `results/main`)

Config: 5 sizes × 2 tasks × T in {256, 512, 1024, 2048} × 9 seeds = 360 runs. Step caps are 20k at T = 256, 50k at T = 512 and 1024, and 20k at T = 2048. The 59.5M model saves checkpoints at 30%, 60%, and 90% accuracy, for distillation teachers. Seeds 0 to 2 at T = 256 are the MVP runs, reused.

Status on Nov 9:

| Context length | Finished runs | Notes |
|---|---|---|
| T = 256 | 90 of 90 | copied here |
| T = 512 | 19 of 90 | running on 4 L40S |
| T = 1024 | 6 of 90 | running on 4 A100, after the architecture sweep |
| T = 2048 | 0 of 90 | not started |

T = 256 results (`results/main/analysis_table.txt`):

| Task | Size | Reached 90% | Expected PFLOPs per success |
|---|---|---|---|
| needle | 0.4M | 4/9 | 1.6 |
| needle | 3.2M | 5/9 | 10.2 |
| needle | 10.6M | 1/9 | 196 |
| needle | 25M | 3/9 | 126 |
| needle | 59.5M | 5/9 | 159 |
| MQAR | 0.4M | 0/9 | more than 9.3 |
| MQAR | 3.2M | 2/9 | 26.4 |
| MQAR | 10.6M | 9/9 | 13.1 |
| MQAR | 25M | 3/9 | 150 |
| MQAR | 59.5M | 4/9 | 255 |

- Failures: 54 of the 90 runs fail.
  - 47 never pass 50% and end at the copying level.
  - 7 pass 50% without reaching 90%. Six of these first pass 50% at step 13,500 or later and are still improving at the 20k cap (five are 25M on MQAR).
  - Only MQAR 3.2M seed 2 really stalls, at about 86% from step 2,000.
- Early T = 512 results (19 runs, all on MQAR so far):
  - 0.4M: 0 of 9.
  - 3.2M: 2 of 9, within 50k steps.
  - 10.6M: 1 of 1 finished so far.

## Architecture sweep (grid `architecture`, folder `results/architecture`)

Config:

- The shapes of the `heads`, `head_dim`, and `depth` grids in `code/sweep.py`, with the shared 10.6M shape kept once: 12 shapes.
- 2 tasks × 5 seeds, T = 256, 20k steps, lr 3e-4 for every shape (not re-tuned): 120 runs.

Results (Figure `results/architecture/figures/architecture.png`):

- Depth at about 10M parameters, head dimension 64:

  | Shape | MQAR reached | MQAR PFLOPs per success | Needle reached | Needle PFLOPs per success |
  |---|---|---|---|---|
  | 2 layers × width 640 | 5/5 | 5.7 | 5/5 | 11.8 |
  | 4 layers × width 448 | 5/5 | 14.0 | 4/5 | 16.4 |
  | 6 layers × width 384 | 4/5 | 19.4 | 1/5 | 98.7 |
  | 12 layers × width 256 | 0/5 | — | 0/5 | — |
  | 24 layers × width 192 | 0/5 | — | 0/5 | — |

  All 20 runs at 12 and 24 layers stay at the copying level.
- Head dimension at 6 heads:

  | Head dimension | MQAR reached | Needle reached |
  |---|---|---|
  | 16 | 0/5 | 0/5 |
  | 32 | 1/5 | 1/5 |
  | 64 | 4/5 | 1/5 |
  | 128 | 4/5 | 3/5 |

  Larger head dimension also adds parameters: 8.0M at 16 and 14.2M at 128.
- Heads at width 384:

  | Heads | MQAR reached | Needle reached |
  |---|---|---|
  | 2 | 0/5 | 1/5 |
  | 3 | 1/5 | 2/5 |
  | 6 | 4/5 | 1/5 |
  | 12 | 3/5 | 4/5 |
  | 24 | 0/5 | 1/5 |

  There is no clear trend, and the intervals overlap.
- Reproducibility: the 6-layer shape repeats main-sweep 10.6M seeds 0 to 4.
  - 4 of 10 pairs have identical loss at step 250 and identical outcomes.
  - Of the 6 pairs that differ at step 250, 3 flip whether the run reaches 90%.
