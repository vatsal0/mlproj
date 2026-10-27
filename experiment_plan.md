# Experiment Plan

This document lists every run we need for the final report, then the subset that suffices for the Oct 28 checkpoint (MVP). All commands refer to the code in `code/`. Compute estimates are upper bounds that assume every run hits its step cap; runs stop early once they reach the threshold, so real usage should be lower.

## Shared setup

**Architecture** (answers TA question 1). We use a Llama-style decoder-only transformer that we implement ourselves (`code/model.py`): pre-norm RMSNorm, rotary position embeddings, SwiGLU MLP with hidden size 8/3 × width, no biases, no dropout, untied input and output embeddings, and causal softmax attention (PyTorch SDPA). Attention width is n_heads × head_dim and may differ from the model width, which the head-dimension sweep needs. Initialization is normal with std 0.02, and residual output projections are scaled by 1/√(2 × layers).

| Size | Layers | Width | Heads × head dim | Non-embedding params |
|---|---|---|---|---|
| xs | 2 | 128 | 4 × 32 | 0.40M |
| s | 4 | 256 | 4 × 64 | 3.2M |
| m | 6 | 384 | 6 × 64 | 10.6M |
| l | 8 | 512 | 8 × 64 | 25.3M |
| xl | 12 | 640 | 10 × 64 | 59.5M |

**Tasks** (`code/tasks.py`). Both tasks share one vocabulary of 256 filler tokens, 1024 keys, and 512 values, so context length is the only thing that changes across runs. Each sequence ends with a block of queries, and the model must predict the value that follows each queried key.
- MQAR: T/4 key-value pairs followed by queries for all of them, in shuffled order. Longer context means more pairs to track.
- Needle: 16 key-value pairs scattered at random positions in random filler, followed by queries for all 16. Longer context means more filler, while the retrieval problem stays fixed.

At T = 64 the needle task has no filler and is identical to MQAR. As T grows, the two tasks differ only in whether the extra context holds more pairs or more filler. MQAR still scores more answers per sequence at long context (T/4 against 16), so MQAR gets more training signal per step. This does not affect comparisons across model sizes at a fixed task and context length, which is our main result. For comparisons across tasks or context lengths, we also report the number of answers seen before reaching threshold.

The needle task went through two revisions. With a single pair, the value is the only value token in the context, so a model can copy it without matching keys. With 4 pairs, the 3.2M model plateaued at 45% accuracy for 6000 steps at T = 64, with attention spread roughly evenly over the 4 values, while the same model solved MQAR (16 pairs) by step 4000. We attribute the plateau to the weak training signal of 4 answers per sequence and moved to 16 pairs (`--n_needles` changes this).

**Training.** AdamW (β = 0.9, 0.95, weight decay 0.1 on matrices), 250 warmup steps then constant learning rate, gradient clipping at 1.0, batch of 64 sequences, bf16 (fp16 on T4). Loss is cross-entropy on the query answers only. Training data is generated fresh each step.

**Evaluation.** Every 250 steps, on a fixed held-out set of 512 sequences shared by all runs:
- accuracy on query answers;
- steps to threshold: the first evaluation of three consecutive evaluations at or above 90% accuracy; the run stops there;
- FLOPs to threshold: steps × analytic FLOPs per step (Kaplan et al., 2020), including the attention term 6 × layers × T × attention width per token;
- accuracy by relative position of the retrieved pair (5 buckets);
- attention mass that each head places on the correct value token from each query (ground-truth retrieval pattern).

**Reference points.** Chance accuracy is 1/512. A model that copies a random value from the context without matching keys, ruling out values already shown in the query block, scores about H_n / n for n pairs (H_n is the n-th harmonic number; repeated values raise it slightly): 21.3% on the needle task and 7.6%, 4.4%, and 2.6% on MQAR at T = 256, 512, and 1024. An accuracy plateau at this level with little attention on the correct value means the model has learned to copy but not to retrieve. Within each experiment, from-scratch training of each size is the baseline that interventions (distillation, curriculum) must beat.

## Full plan

Each experiment lists the question it answers. Run counts include both tasks.

| ID | Question | Grid | Runs | Upper bound (GPU h) |
|---|---|---|---|---|
| E0 | Are the tasks calibrated, neither trivial nor impossible within 20k steps? | s and l at T = 256 and 2048, 1 seed | 8 | 8 |
| E1 | What learning rate does each size need? | 5 sizes × lr ∈ {3e-4, 1e-3, 3e-3} at T = 512, 1 seed | 30 | 15 |
| E2 | Does the compute-optimal model size grow with context length? (main result) | 5 sizes × T ∈ {256, 512, 1024, 2048} × 5 seeds | 200 | 208 |
| E3a | Do more heads help, at fixed width? | m shape, heads ∈ {2, 3, 6, 12, 24}, T ∈ {512, 2048}, 5 seeds | 100 | 84 |
| E3b | Does head dimension matter, at fixed head count? | m shape, 6 heads, head dim ∈ {16, 32, 64, 128} | 80 | 65 |
| E3c | Depth or width, at matched parameters (~10M)? | layers ∈ {2, 4, 6, 12, 24} with width ∈ {640, 448, 384, 256, 192} | 100 | 84 |
| E4 | Do small models show lost-in-the-middle? Does the accuracy jump coincide with learning the retrieval pattern? | analysis of E2 logs, no new runs | 0 | 0 |
| E5 | Can a briefly trained teacher speed up a small student? | xl teachers saved at 30%, 60%, 90% accuracy (from E2) × students xs, s, m × T ∈ {512, 2048} × 3 seeds | 108 | 192 |
| E5b | If logit distillation fails, does matching attention to the teacher's retrieval heads help? | best E5 setting only | ~18 | 67 |
| E6 | Does training at T = 256 before T = 2048 reduce total compute? | xs, s, m × 3 seeds, switch to 2048 once 256 reaches threshold | 18 | 13 |

Upper bounds assume every run hits the 20k-step cap and use the throughput we measured in the MVP: 47.9 TFLOP/s per GPU averaged over the L40S, A100, and H200 GPUs we got on Torch, with four runs sharing each GPU (29 TFLOP/s on L40S, 69 on A100, 82 on H200). The cap is a realistic estimate, not a loose bound: MVP runs at T ≥ 256 used 95% of it on average, because most never reach 90%. E5 counts the xl teacher's forward pass at each student step. The upper bound for everything is about 735 GPU hours, almost 9 times our earlier estimate of 85 H100 hours, which assumed a much higher throughput. Most of it comes from the xl size and T = 2048. T = 2048 alone accounts for 119 of E2's 208 GPU hours, and E5 and E5b are expensive because every student step runs the xl teacher. On the 6 to 9 GPUs we held at once during the MVP, the full plan takes about 3.5 days of continuous running, or 1 to 2 weeks with queue waits.

Two things change this total. First, 72 of E2's runs (sizes xs to l, T = 256 to 1024, seeds 0 to 2) are identical to MVP runs if E2 uses the MVP learning rates, and `train.py` skips a run whose folder already holds a `summary.json`. Reusing them saves about 31 GPU hours; picking new learning rates in E1 loses that reuse. Second, the job time limit is 8 hours and `train.py` cannot resume a run, so a preempted or timed-out run restarts from step 0. One xl run at T = 2048 needs about 7 GPU hours at 20k steps, or about 28 hours of wall-clock time when four runs share a GPU, so these runs will never finish as configured. They need checkpoint-and-resume in `train.py`, or a GPU to themselves on a fast GPU type.

E5 compute is reported two ways: student training only, and student training plus teacher training up to the checkpoint plus teacher forward passes during distillation. E5, E5b, and E6 need code we have not written yet (`distill.py` and a curriculum flag in `train.py`); we write it after the checkpoint.

Commands: `python sweep.py --grid {lr_full, main, heads, head_dim, depth}`, with `--lr_json` set to the E1 winners from `analyze.py --best_lr`. The `main` grid saves xl checkpoints at the 30%, 60%, and 90% accuracy milestones for E5.

## If we cannot use Torch

Colab cannot run the full plan. Assuming realistic throughputs of 12 TFLOP/s on a T4 and 40 TFLOP/s on an L4, the full plan needs about 1,800 T4 hours or 530 L4 hours. A reduced plan drops the xl size and T = 2048, uses 3 seeds, and runs the architecture sweeps at T = 512 only. That reduced plan costs about 175 T4 hours or 50 L4 hours as an upper bound. The reduced plan still answers the main question over a 4× range of context lengths (256 to 1024) and a 60× range of model sizes, but it loses the longest context and the largest teacher, which weakens E5.

## MVP for the Oct 28 checkpoint

The checkpoint asks for a working system, data and evaluation pipeline, baselines, first experiments, and preliminary results. The MVP is a scaled-down E1 and E2 that Vatsal runs on Torch. Runs stop at threshold or after 20k steps.

| Step | Grid | Runs | Upper bound |
|---|---|---|---|
| Learning rate | xs, s, m, l × lr ∈ {3e-4, 1e-3, 3e-3} at T = 256 | 24 | ~0.5 H100 h |
| Scale sweep | xs, s, m, l × T ∈ {128, 256, 512, 1024} × 3 seeds | 96 | ~4 H100 h |

`code/slurm/submit_mvp.sh` submits both as a chain: the learning-rate grid on 4 GPUs, a CPU job that picks the best learning rate per size, then the scale sweep on 8 GPUs. Each GPU runs 4 trainings at once, because a single small model leaves the GPU mostly idle and Torch cancels jobs with low GPU utilization. Jobs may be preempted and requeued; finished runs are skipped, so a requeued job resumes where it stopped. Before submitting, set the account in both `.sbatch` files and the paths and environment in `slurm/env.sh`.

Roles: Vatsal runs experiments; Ammar and Abhishek own analysis (`python analyze.py <runs dir>`) and the write-up.

The MVP produces the checkpoint deliverables directly:
- working system and pipeline: the code in `code/`;
- baselines: chance, the copy-without-retrieval level, and the smallest model trained from scratch;
- first experiment: steps and FLOPs to threshold against model size, one line per context length (`analyze.py` writes `flops_to_threshold.png`);
- preliminary results: learning curves with attention mass (`curves_*.png`) and accuracy by needle position (`acc_by_depth.png`).

**Sanity checks already done (MacBook, 3.2M model, lr 1e-3).** On MQAR at T = 64, accuracy plateaus near 20% until step 1500, then jumps and reaches 90% at step 4000. Over the same steps, the attention mass on the correct value rises from 0.06 to 0.78, so the jump in accuracy coincides with learning the retrieval pattern. On the 16-needle task at T = 128, accuracy sits at 20% from step 5000 to step 8000 with attention mass 0.04, so the model has not made the jump within 8000 steps. Accuracy is flat across needle positions during this plateau. With learning rate 3e-3 instead, the same run reaches only 2.9% after 8000 steps, below the 21% copying level, so a higher learning rate does not shorten the plateau.

**Risk for the MVP.** Baherwani et al. (2026) find that the plateau before the jump grows with context length, so many MVP runs at T = 256 and 512 may hit the 10k step cap. `analyze.py --threshold 0.5` recomputes steps to threshold at a lower accuracy from the logs, which still gives a usable size comparison if 90% is out of reach.
