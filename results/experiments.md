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

Status on Nov 16: all 360 runs finished and are copied here. The last xl runs at T = 2048 ran one per A100, because four xl runs ran out of memory on an L40S (and one alone at T = 2048 needs more than the L40S's 44 GB).

Results (`results/main/analysis_table.txt`). Each cell is runs that reached 90% out of 9, then expected PFLOPs per success; ">" marks a lower bound, when no run succeeded:

| Task | Size | T = 256 (20k steps) | T = 512 (50k) | T = 1024 (50k) | T = 2048 (20k) |
|---|---|---|---|---|---|
| MQAR | 0.4M | 0/9, >9.3 | 0/9, >52 | 0/9, >128 | 0/9, >139 |
| MQAR | 3.2M | 2/9, 26 | 2/9, 155 | 1/9, 747 | 0/9, >768 |
| MQAR | 10.6M | 9/9, 13 | 2/9, 477 | 1/9, 2,302 | 0/9, >2,196 |
| MQAR | 25M | 3/9, 150 | 9/9, 168 | 0/9, >5,261 | 0/9, >4,802 |
| MQAR | 59.5M | 4/9, 255 | 9/9, 290 | 4/9, 2,683 | 0/9, >10,688 |
| needle | 0.4M | 4/9, 1.6 | 8/9, 3.3 | 7/9, 11.6 | 0/9, >131 |
| needle | 3.2M | 5/9, 10 | 6/9, 44 | 5/9, 93 | 0/9, >752 |
| needle | 10.6M | 1/9, 196 | 5/9, 152 | 2/9, 957 | 0/9, >2,172 |
| needle | 25M | 3/9, 126 | 1/9, 2,227 | 0/9, >5,222 | 0/9, >4,771 |
| needle | 59.5M | 5/9, 159 | 0/9, >5,614 | 1/9, 11,292 | 0/9, >10,649 |

The step caps differ across context lengths, so for a comparison at equal budget, these are the runs that reached 90% within 20k steps. The learning rate is constant after warmup, so a run's first 20k steps do not depend on its cap.

| Context length | Reached 90% within 20k steps (of 90) | MQAR | needle |
|---|---|---|---|
| T = 256 | 36 | 18 | 18 |
| T = 512 | 12 | 5 (all 59.5M) | 7 |
| T = 1024 | 7 | 0 | 7 |
| T = 2048 | 0 | 0 | 0 |

- Longer context lowers the success probability for almost every size, and the cost per success grows faster than the cost per step. Going from T = 256 to 1024, the cost per success of the cheapest reliable model rises 7× on needle (0.4M: 1.6 to 11.6 PFLOPs) and 57× on MQAR (cheapest successful size: 13 PFLOPs at 10.6M, then 747 PFLOPs at 3.2M with 1 of 9).
- The best size depends on the task:
  - On needle, the smallest model (0.4M) is the cheapest at every T up to 1024 and also the most reliable at T = 512 and 1024 (8 of 9 and 7 of 9). Larger models get less reliable as T grows: 59.5M goes from 5 of 9 at T = 256 to 0 of 9 at T = 512.
  - On MQAR, small models (0.4M) never succeed at any T. Larger models (25M and 59.5M) succeed in 9 of 9 at T = 512 within 50k steps, but mostly after 20k steps. At T = 1024, only 59.5M succeeds in more than 1 seed (4 of 9).
- T = 2048: no run of 90 reaches 90% within 20k steps, on either task. These cells are lower bounds only.
- Failures at every context length mostly never leave the copying level. Runs that fail and never pass 50% held-out accuracy: 47 of 54 failures at T = 256, 47 of 48 at T = 512, 67 of 69 at T = 1024, and 89 of 90 at T = 2048. The median best accuracy of failed runs is 2 to 8% on MQAR and about 22% on needle.
- Successes are often late. Of the successful runs, 21 of 36 at T = 256 and 23 of 42 at T = 512 succeed after half their step cap. So the success probabilities depend on the cap, and a longer cap would raise them.

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

## Distillation (grid `distillation`, folder `results/distillation`)

Config: students of sizes 0.4M, 3.2M, and 10.6M at T = 256, 3 seeds each, distilled from 59.5M teachers of the main sweep saved when they first reached 30%, 60%, and 90% held-out accuracy. Each task uses the lowest teacher seed that saved all three checkpoints: seed 1 on MQAR (steps 13,250, 13,750, 16,000) and seed 3 on needle (steps 6,750, 7,000, 8,250). Loss on answer positions: 0.5 × cross-entropy + 0.5 × KL divergence from the teacher's predictions, temperature 1. These are standard defaults, not tuned. Learning rates, batch size, and step cap (20k) match the main sweep at T = 256. 2 × 3 × 3 × 3 = 54 runs, on 2 L40S, four runs per GPU.

Compute is counted three ways (`python analyze.py results/distillation --baseline results/main`):
- student training only;
- plus the teacher's forward pass at every student step (39, 5.7, and 1.8 times the own step cost of the 0.4M, 3.2M, and 10.6M students);
- plus training the teacher up to its checkpoint, charged in full to one student.

The baseline is the main sweep at T = 256 (same sizes, 9 seeds). The table also gives the break-even number of students: how many students one teacher must teach before distillation is cheaper than training from scratch.

All 54 runs finished on Nov 16, with no failures. Teachers, measured on the student run's held-out set when loaded:

| Task | Saved at | Measured accuracy | Teacher training PFLOPs |
|---|---|---|---|
| MQAR | 30% | 41% | 80 |
| MQAR | 60% | 64% | 84 |
| MQAR | 90% | 91% | 97 |
| needle | 30% | 60% | 41 |
| needle | 60% | 77% | 42 |
| needle | 90% | 91% | 50 |

Results (`results/distillation/analysis_table.txt`, figure `figures/distillation.png`). Distilled settings have 3 seeds, from scratch has 9; ">" marks a lower bound (no success):

| Task | Student | Teacher | Reached 90% | PFLOPs per success, student only | + teacher forward | + teacher training | Break-even students |
|---|---|---|---|---|---|---|---|
| MQAR | 0.4M | from scratch | 0/9 | >9.3 | | | |
| MQAR | 0.4M | 30% | 1/3 | 2.3 | 92 | 173 | unknown |
| MQAR | 0.4M | 60% | 1/3 | 2.2 | 87 | 170 | unknown |
| MQAR | 0.4M | 90% | 0/3 | >3.1 | >124 | >222 | unknown |
| MQAR | 3.2M | from scratch | 2/9 | 26 | | | |
| MQAR | 3.2M | 30% | 3/3 | 5.4 | 36 | 117 | never |
| MQAR | 3.2M | 60% | 3/3 | 4.2 | 28 | 112 | never |
| MQAR | 3.2M | 90% | 0/3 | >21 | >143 | >240 | unknown |
| MQAR | 10.6M | from scratch | 9/9 | 13 | | | |
| MQAR | 10.6M | 30% | 3/3 | 12.7 | 36 | 116 | never |
| MQAR | 10.6M | 60% | 3/3 | 15.5 | 43 | 127 | never |
| MQAR | 10.6M | 90% | 1/3 | 62 | 174 | 271 | never |
| needle | 0.4M | from scratch | 4/9 | 1.6 | | | |
| needle | 0.4M | 30% | 1/3 | 2.4 | 105 | 146 | never |
| needle | 0.4M | 60% | 0/3 | >2.8 | >124 | >166 | unknown |
| needle | 0.4M | 90% | 1/3 | 2.4 | 104 | 154 | never |
| needle | 3.2M | from scratch | 5/9 | 10 | | | |
| needle | 3.2M | 30% | 0/3 | >21 | >142 | >182 | unknown |
| needle | 3.2M | 60% | 0/3 | >21 | >142 | >184 | unknown |
| needle | 3.2M | 90% | 1/3 | 19 | 132 | 182 | never |
| needle | 10.6M | from scratch | 1/9 | 196 | | | |
| needle | 10.6M | 30% | 2/3 | 27 | 77 | 118 | 1 |
| needle | 10.6M | 60% | 2/3 | 25 | 71 | 114 | 1 |
| needle | 10.6M | 90% | 1/3 | 59 | 168 | 218 | 2 |

- Counting the student's compute only, the partly trained teachers (30% and 60%) help on MQAR. The 0.4M student goes from 0/9 to 1/3, and 3.2M from 2/9 to 3/3, with 5 times less compute per success (26 to about 5 PFLOPs). On needle they help only the 10.6M student (1/9 to 2/3, 196 to about 26 PFLOPs), and hurt 3.2M (5/9 to 0/3).
- The 90% teacher is the worst of the three on MQAR at every student size (0/3, 0/3, and 1/3), even though it is the most accurate.
- Counting the teacher's forward passes, distillation is more expensive than training from scratch everywhere except needle at 10.6M. The forward pass of the 59.5M teacher costs more than the student's own step at every student size (1.8 to 39 times), so it dominates the cost of small students.
- Needle at 10.6M is the only setting where distillation pays off, even including the teacher's training: one student with the 30% or 60% teacher already costs less (about 115 PFLOPs) than one success from scratch (196 PFLOPs, from only 1 success in 9).
- 3 seeds per setting give wide intervals (for example, 3/3 has a 95% interval of 44% to 100%), so the differences between the 30% and 60% teachers are not resolved.
