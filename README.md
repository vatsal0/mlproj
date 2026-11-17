# Does Scale Make Long-Context Learning Cheaper?

We train small transformers on two synthetic retrieval tasks and measure how much compute each model size needs to reach 90% accuracy, as context length grows. The full motivation and design are in `proposal_draft.md`.

## Roles and deadlines

Vatsal runs experiments on NYU's Torch cluster. Ammar and Abhishek own analysis and the write-ups. Upcoming milestones are the MVP checkpoint (Oct 28), the investigation checkpoint (Nov 11), and the final report, demo, and code (Dec 1).

## What is in the repo

- `proposal_draft.md`: the proposal we submitted, followed by the TA feedback and checkpoint instructions.
- `experiment_plan.md`: every experiment we plan to run, the question each one answers, and its compute cost. Read this first.
- `checkpoint/`: everything for the Oct 28 checkpoint: the write-up (`checkpoint.md`), its figures, and the results table.
- `checkpoint2/`: the Nov 11 checkpoint update (`update.md`), with its figures and tables, including the reproducibility check (the same seed rerun on another GPU).
- `results/`: every finished experiment, one folder each. **Start with `results/SUMMARY.md` (core findings) and `results/experiments.md` (config, tables, and caveats for every experiment).** These are the source for the final report.
- `code/`: tasks, model, training, sweeps, analysis, and SLURM scripts (details below).
- `figures/`: figures used in write-ups.
- `style_sample.tex`: a writing sample for any write-up drafted with AI (see below).
- Course material: `NYU_Machine_Learning_Final_Project_Guidelines.docx.pdf`, `feedback.md` (proposal feedback), and our earlier drafts.

## Code

| File | What it does |
|---|---|
| `code/tasks.py` | Generates MQAR and needle-in-a-haystack batches on the fly |
| `code/model.py` | Llama-style transformer, model sizes, and the FLOP count |
| `code/train.py` | Trains one run; logs accuracy, accuracy by needle position, and attention on the correct token every 250 steps. Resumes from its last evaluation. `--save_at_acc` saves checkpoints at accuracy milestones; `--teacher` distills from such a checkpoint |
| `code/sweep.py` | Named grids of runs (`lr`, `lr_full`, `mvp`, `main`, `architecture`, `distillation`, and others) |
| `code/analyze.py` | Summary table and figures from a folder of runs; `--baseline` compares distilled students against training from scratch |
| `code/slurm/` | Torch job scripts (only Vatsal needs these); `submit_sweep_chain.sh` runs a grid as a chain of 8-hour jobs |

Each run writes a folder with `summary.json` (final numbers, including steps and FLOPs to threshold) and `log.jsonl` (one line per evaluation).

## Getting results and making figures

All experiments are finished and their results are in `results/`, so you do not need Torch access. Run outputs (including the model checkpoints, which are not in the repo) live on Torch scratch; Vatsal copied them with:

```
rsync -av --exclude '*.pt' --exclude 'stdout_*' torch:/scratch/vb2647/mlproj/runs/ results/
```

Analysis only needs numpy and matplotlib:

```
pip install numpy matplotlib
python code/analyze.py results/main                  # table + figures in results/main/figures
python code/analyze.py results/main --threshold 0.5  # same, at a 50% accuracy threshold
python code/analyze.py results/architecture
python code/analyze.py results/distillation --baseline results/main
```

| Folder | Experiment | Runs |
|---|---|---|
| `results/lr`, `results/lr_full` | learning rate per size, at T = 256 and 512 | 30, 38 |
| `results/calibration` | are the tasks solvable at T = 2048 within 20k steps? | 4 |
| `results/mvp` | first scale sweep, for the Oct 28 checkpoint | 96 |
| `results/main` | main sweep: 5 sizes × 2 tasks × T = 256 to 2048 × 9 seeds | 360 |
| `results/architecture` | heads, head dimension, and depth around the 10.6M model, at T = 256 | 120 |
| `results/distillation` | 0.4M to 10.6M students distilled from 59.5M teachers, at T = 256 | 54 |

`main`, `architecture`, and `distillation` each hold `analysis_table.txt` (what `analyze.py` prints) and `figures/`, which includes `runs.csv` with one row per run. For the other folders, run `analyze.py` to get the table. Training curves for every run are on wandb: https://wandb.ai/vatsalbaherwani/long-context-scaling-mlproj (one group per experiment; ask Vatsal for access).

The figures:
- `success_and_cost.png`: our main result. Top, the probability of reaching 90% against model size, with 95% intervals. Bottom, the expected FLOPs per success: all compute spent, failed runs included, divided by the number of successes. Hollow markers are lower bounds, where no run succeeded.
- `reached_vs_flops.png`: the fraction of runs that have reached 90% against FLOPs spent, one line per size.
- `flops_to_threshold.png` and `steps_to_threshold.png`: cost to reach the threshold among successful seeds only. Points are medians over seeds, bars span the min and max, and a label like `2/3` means only 2 of 3 seeds reached the threshold. These favor sizes that succeed by luck, so prefer `success_and_cost.png`.
- `curves_<task>_T<length>.png`: accuracy and attention on the correct token over training, for every run.
- `retrieval_onset.png`: accuracy and attention aligned at the accuracy jump (does the jump coincide with the retrieval head forming?).
- `acc_by_depth.png`: final needle accuracy by where the needle sits in the context (the lost-in-the-middle check).
- `architecture.png` (in `results/architecture`) and `distillation.png` (in `results/distillation`).

Runs that never reach 90% are not failures of the pipeline. A plateau at the copying level (21.3% on needle; 7.6%, 4.4%, 2.6% on MQAR at T = 256, 512, 1024; see the Baselines section of `checkpoint/checkpoint.md`) with flat attention means the model has learned to copy values but not to retrieve the right one. These cases are worth reporting.

To run something small yourself, `python code/train.py --task mqar --T 64 --size s --max_steps 5000` reaches 90% in about 10 minutes on a laptop (needs PyTorch).

## Writing with AI

If you draft any write-up with an AI assistant, give it `style_sample.tex` as the reference for our writing style. Only the main-body prose matters, especially Sections 3 and 4. Ignore the LaTeX template comments at the top. Ask the assistant to first summarize the style in a few bullet points, then write in that style. The style we have used so far:
- Short, direct sentences in the active voice, with "we".
- Each paragraph opens with its main point, then gives specifics with concrete numbers.
- Define every quantity before using it, and interpret every result instead of only stating it.
- No hedging, no em-dashes, no "not only ... but also", no contractions, no slashes.
- Prose instead of bullet lists or tables, unless the content needs them.

Check every number the assistant writes against `summary.json` or the figures. AI-drafted text is a starting point, not a source of results.
