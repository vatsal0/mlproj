# Checkpoint 2: Progress Update

Vatsal Baherwani, Ammar Ali, Abhishek Muhuri

## Experiments Completed

Since the first checkpoint we have finished 253 new runs in four experiments.

1. **Reachability of context length 2048 (4 runs).** We trained the 3.2M and 25M models on both tasks for 20k steps. The 25M model reaches 90% on needle at step 15,500. Both models stay at the copying level on MQAR.
2. **Learning rate re-tuned at T = 512 (38 runs).** All five sizes, including the 59.5M model. The choices for the four smaller sizes did not change, and the 59.5M model takes 1e-4.
3. **Main scale sweep, 9 seeds per setting.** All five sizes (0.4M to 59.5M), both tasks, T = 256 to 2048.
   - T = 256 is finished: 90 runs, of which 24 are reused from the MVP and 66 are new.
   - T = 512 and T = 1024 are running, with up to 50k steps per run. As of November 9, 19 of 90 runs at T = 512 and 6 of 90 at T = 1024 are finished.
   - T = 2048 has not started.
4. **Architecture sweep at T = 256 (120 runs).** Five seeds per shape, varying three things around the 10.6M model (6 layers, width 384, 6 heads of dimension 64):
   - the number of heads at a fixed width,
   - the head dimension at 6 heads,
   - the split between depth and width at about 10M parameters.

Figure 1 shows the probability of reaching 90% and the expected FLOPs per success for each size at T = 256. Figure 2 shows the fraction of runs that have reached 90% as a function of FLOPs spent. Figure 3 shows the architecture sweep. The full tables are in `main_sweep_T256_table.txt` and `architecture_sweep_T256_table.txt`.

## Ablations

All architecture runs use the learning rate tuned for the 10.6M model (3e-4) and train for up to 20k steps at T = 256. With 5 seeds a success probability is uncertain: the 95% interval for 1 of 5 runs spans 5% to 62%. The only differences we trust are those whose intervals do not overlap, such as 5 of 5 against 0 of 5 (Figure 3).

- **Depth against width is the clearest effect.** At about 10M parameters:
  - 2 layers of width 640 reach 90% in 5 of 5 seeds on both tasks.
  - 4 layers reach 90% in 5 of 5 seeds on MQAR and 4 of 5 on needle.
  - 12 and 24 layers reach 90% in 0 of 5 seeds on both tasks, and every one of these runs stays at the copying level.
  - The expected cost per success of the 2-layer model is 5.7 PFLOPs on MQAR and 11.8 PFLOPs on needle. The default 6-layer shape costs 19.4 and 98.7 PFLOPs.

  These tasks need only a few attention steps. Extra depth makes the model harder to move off the plateau and adds nothing it needs. One caveat: we did not re-tune the learning rate for each shape, so the deep, narrow models may train poorly partly because 3e-4 is not their best rate.
- **Head dimension.** With head dimension 16, no run reaches 90% on either task (0 of 5 each). Dimensions 64 and 128 are the best on MQAR (4 of 5 each), and 128 is the best on needle (3 of 5). Note that with 6 heads, a larger head dimension also widens attention and adds parameters (8.0M at dimension 16, 14.2M at dimension 128).
- **Number of heads at a fixed width of 384.** We see no clear trend. On MQAR, 2 heads (dimension 192) and 24 heads (dimension 16) both reach 0 of 5, while 6 and 12 heads reach 4 and 3 of 5. On needle, every setting except 12 heads (4 of 5) reaches 2 of 5 or fewer, and the intervals overlap.

## Failure and Error Analysis

Most runs that fail never leave the copying level. A model at the copying level outputs a value from the context without matching keys, ruling out values already answered in the query block, and its attention on the correct value stays near zero. In the MVP, the runs whose accuracy never exceeds 50% end at a median accuracy within 0.5 points of the copying level of their setting. These runs are not stuck in a bad optimum that more tuning fixes. At T = 512, no learning rate between 1e-4 and 3e-3 moves the three largest sizes off the copying level within 20k steps.

A second failure is a learning rate that is too high. At 3e-3, every size above 0.4M ends with a loss of 6.24, the loss of a uniform guess over the 512 values, so these runs never learn even to copy. A third, rarer failure is a run that leaves the copying level but stalls before 90%. Nine MVP runs and the 3.2M model at T = 512 in the learning rate sweep, which stalls at 52%, belong to this group.

The 9-seed sweep at T = 256 shows that most of these runs did not stall at all: they ran out of steps.

- 54 of the 90 runs fail.
- 47 of the failed runs never exceed 50% accuracy.
- 7 runs pass 50% without reaching 90%. Six of these, five of them from the 25M model on MQAR, first pass 50% at step 13,500 or later and are still improving when training stops at 20k steps. One 25M run even reaches 92% but holds it for fewer than 3 evaluations.
- Only one run, the 3.2M model on MQAR with seed 2, truly stalls: it holds about 86% from step 2,000 to the end.

The 20k-step cap at T = 256 therefore undercounts successes for the larger models on MQAR.

Our success measure also had an error, which the teaching team pointed out. Figure 2 of the first checkpoint showed the FLOPs to 90% only among seeds that reached 90%, so a size that succeeds in 1 of 3 seeds could look cheaper than a size that succeeds in all 3. For example, the 25M model on MQAR at T = 256 needs a median of 48 PFLOPs in its one successful seed, but 153 PFLOPs per success once its two failed runs are counted. We now report the probability of reaching 90% and the expected FLOPs per success, which divides all compute spent, failed runs included, by the number of successes (Figure 1). When no run succeeds, this total is a lower bound, drawn with a hollow marker.

## Unexpected Findings

The seed does not fix the outcome of a run. Our learning rate sweeps repeat 16 MVP runs with identical settings and seed, and 5 of the 16 pairs disagree on whether the run reaches 90%. Two runs of a pair agree exactly when they ran on the same GPU type at T = 256, and every pair that ran on different GPU types diverges, with training losses that already differ by 0.00001 to 0.002 at step 250. We train in bfloat16 mixed precision, and these small numerical differences grow until they shift when a run leaves the plateau. The time a run spends on the plateau is therefore partly random, which is why we now measure the probability of success rather than the outcome of a few seeds.

The architecture sweep repeats the 10.6M main-sweep runs for seeds 0 to 4 under the same settings, which gives 10 more pairs. In 4 of the 10 pairs the training loss is identical at step 250, and both runs give the same result. In the other 6 the losses differ at step 250, and 3 of these pairs disagree on whether the run reaches 90%.

Larger models do not learn fast enough to pay for their cost per step, and the 9-seed sweep at T = 256 confirms this once failures are counted (Figure 1).

| Expected PFLOPs per success | 0.4M | 3.2M | 10.6M | 25M | 59.5M |
|---|---|---|---|---|---|
| needle | 1.6 | 10 | 196 | 126 | 159 |
| MQAR | more than 9.3 (0 of 9 succeed) | 26 | 13 | 150 | 255 |

- On needle, the 0.4M model is the cheapest by a factor of 6.
- On MQAR, the 0.4M model never leaves the copying level, and the cheapest model is the 10.6M model, the only size that succeeds in all 9 seeds.
- The 25M and 59.5M models cost 12 to 20 times more per success than the cheapest size on MQAR, and 80 to 100 times more on needle.

The cheapest size is therefore the smallest one that succeeds reliably, which is not always the smallest model.

The success probability does not rise with model size.

| Probability of reaching 90% | 0.4M | 3.2M | 10.6M | 25M | 59.5M |
|---|---|---|---|---|---|
| needle | 4/9 | 5/9 | 1/9 | 3/9 | 5/9 |
| MQAR | 0/9 | 2/9 | 9/9 | 3/9 | 4/9 |

The 10.6M model succeeds in 9 of 9 seeds on MQAR but only 1 of 9 on needle. The architecture sweep shows the same split for its default 6-layer shape (4 of 5 against 1 of 5). We do not yet know why. One candidate is that the learning rates, chosen from 3 seeds by median steps to 90%, fit some size and task pairs better than others.

## Challenges Encountered

Our compute estimate was 10 times too low. The proposal assumed an H100 throughput that small models cannot reach. Measured with four trainings sharing each GPU, we get 47.9 TFLOP/s averaged over the L40S, A100, and H200 GPUs we receive, and the full plan as proposed would need about 900 GPU hours. Most runs at long context also train to the step cap instead of stopping early, since they never reach 90%.

The cluster imposes an 8-hour job limit, and our original training script could not resume a run, so the 59.5M model at T = 2048 could never finish. We added checkpoint-and-resume: each run saves its full training state at every evaluation, and a chain of 8-hour jobs continues the runs of the previous job. A run that is killed and resumed logs exactly the same losses and accuracies as an uninterrupted run. Group GPU limits on the cluster accounts also delay jobs, so we split runs across accounts and GPU types.

## Changes to the Original Design

- We redesigned the needle task twice, from 1 to 4 to 16 key-value pairs, because with 1 pair a model can copy the only value in the context, and 4 pairs gave too weak a training signal.
- We tune the learning rate separately for each size, at T = 256 and again at T = 512, because larger models need smaller learning rates.
- We report the probability of reaching 90% and the expected FLOPs per success alongside FLOPs to threshold, following the teaching team's feedback.
- The main sweep uses 9 seeds per setting instead of 5, since the outcome of a run is partly random. We considered separating initialization seeds from data seeds, but dropped it because a run is not reproducible from its seeds across GPU types.
- Runs at T = 512 and 1024 train for up to 50k steps instead of 20k, because 22 of 24 MVP runs at T = 512 stay on the plateau for all 20k steps. T = 2048 keeps 20k steps to fit our compute budget.
- To finish within our compute budget, the architecture sweeps and the distillation experiment run at T = 256 only, instead of T = 512 and 2048. We cut the attention-matching distillation and the short-to-long curriculum.

## Next Experiments

We will finish the main sweep at T = 512, 1024, and 2048, which accounts for most of the remaining compute (about 550 GPU hours). We will then train the 0.4M, 3.2M, and 10.6M models at T = 256 by distillation from checkpoints of the 59.5M model saved at 30%, 60%, and 90% accuracy, and compare their expected FLOPs per success against training from scratch, with and without the teacher's cost. Finally, we will fit how the compute-optimal size changes with context length, using expected FLOPs per success as the cost.
