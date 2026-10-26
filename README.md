# Does Scale Make Long-Context Learning Cheaper?

We train small transformers on two synthetic retrieval tasks and measure how much compute each model size needs to reach 90% accuracy, as context length grows. The full motivation and design are in `proposal_draft.md`.

## Roles and deadlines

Vatsal runs experiments on NYU's Torch cluster. Ammar and Abhishek own analysis and the write-ups. Upcoming milestones are the MVP checkpoint (Oct 28), the investigation checkpoint (Nov 11), and the final report, demo, and code (Dec 1).

## What is in the repo

- `proposal_draft.md`: the proposal we submitted, followed by the TA feedback and checkpoint instructions.
- `experiment_plan.md`: every experiment we plan to run, the question each one answers, and its compute cost. Read this first.
- `checkpoint_draft.md`: draft of the Oct 28 checkpoint. Sections marked `[TODO]` need results from the Torch runs.
- `code/`: tasks, model, training, sweeps, analysis, and SLURM scripts (details below).
- `figures/`: figures used in write-ups.
- `style_sample.tex`: a writing sample for any write-up drafted with AI (see below).
- Course material: `NYU_Machine_Learning_Final_Project_Guidelines.docx.pdf`, `feedback.md` (proposal feedback), and our earlier drafts.

## Code

| File | What it does |
|---|---|
| `code/tasks.py` | Generates MQAR and needle-in-a-haystack batches on the fly |
| `code/model.py` | Llama-style transformer, model sizes, and the FLOP count |
| `code/train.py` | Trains one run; logs accuracy, accuracy by needle position, and attention on the correct token every 250 steps |
| `code/sweep.py` | Named grids of runs (`lr`, `mvp`, `main`, `heads`, `head_dim`, `depth`) |
| `code/analyze.py` | Summary table and figures from a folder of runs |
| `code/slurm/` | Torch job scripts (only Vatsal needs these) |

Each run writes a folder with `summary.json` (final numbers, including steps and FLOPs to threshold) and `log.jsonl` (one line per evaluation).

## Getting results and making figures

Run outputs live on Torch scratch. Vatsal copies them into `results/` without the model checkpoints and pushes:

```
rsync -av --exclude '*.pt' --exclude 'stdout_*' torch:/scratch/vb2647/ml-project/runs/ results/
```

Analysis only needs numpy and matplotlib:

```
pip install numpy matplotlib
python code/analyze.py results/mvp                  # table + figures in results/mvp/figures
python code/analyze.py results/mvp --threshold 0.5  # same, at a 50% accuracy threshold
```

The figures:
- `flops_to_threshold.png` and `steps_to_threshold.png`: cost to reach the threshold against model size, one line per context length. Points are medians over seeds, bars span the min and max, and a label like `2/3` means only 2 of 3 seeds reached the threshold. This is our main result.
- `curves_<task>_T<length>.png`: accuracy and attention on the correct token over training, for every run. Look for whether the accuracy jump coincides with the attention jump.
- `acc_by_depth.png`: final needle accuracy by where the needle sits in the context (the lost-in-the-middle check).

Runs that never reach 90% are not failures of the pipeline. A plateau where accuracy sits above copying level (1/16 on needle, 4/T on MQAR) with flat attention means the model has learned to copy values but not to retrieve the right one. These cases are worth reporting.

To run something small yourself, `python code/train.py --task mqar --T 64 --size s --max_steps 5000` reaches 90% in about 10 minutes on a laptop (needs PyTorch).

## Writing with AI

If you draft any write-up with an AI assistant, give it `style_sample.tex` as the reference for our writing style. Only the main-body prose matters, especially Sections 3 and 4. Ignore the LaTeX template comments at the top. Ask the assistant to first summarize the style in a few bullet points, then write in that style. The style we have used so far:
- Short, direct sentences in the active voice, with "we".
- Each paragraph opens with its main point, then gives specifics with concrete numbers.
- Define every quantity before using it, and interpret every result instead of only stating it.
- No hedging, no em-dashes, no "not only ... but also", no contractions, no slashes.
- Prose instead of bullet lists or tables, unless the content needs them.

Check every number the assistant writes against `summary.json` or the figures. AI-drafted text is a starting point, not a source of results.
