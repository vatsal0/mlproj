# Checkpoint: MVP and Evaluation Pipeline

Vatsal Baherwani, Ammar Ali, Abhishek Muhuri

## Summary

We have a working end-to-end system that generates our two synthetic tasks, trains transformers of five sizes, measures the compute each model needs to reach 90% accuracy, and tracks whether the model has learned the correct attention pattern. Our first experiments, a learning rate sweep and a scale sweep over four model sizes and four context lengths, run on NYU's Torch cluster. Small pilot runs already reproduce the abrupt learning we expected: accuracy jumps at the same time as attention to the correct token. The pilots also led us to redesign the needle-in-a-haystack task twice.

## Core System

We implement a Llama-style decoder-only transformer, which answers the question in our proposal feedback. Each layer applies RMSNorm before attention and before the MLP. Attention is causal softmax attention with rotary position embeddings, and the MLP is a SwiGLU layer with hidden size 8/3 times the model width. The model has no biases or dropout, and the input and output embeddings are untied. We set the attention width (number of heads times head dimension) separately from the model width, which our planned head dimension sweep needs.

Our five model sizes scale depth and width together, from 0.4M to 59.5M non-embedding parameters (2 layers of width 128 up to 12 layers of width 640). We train with AdamW, a 250-step warmup followed by a constant learning rate, and a batch of 64 sequences. The constant learning rate means the time to reach 90% accuracy does not depend on how long we allow a run to train.

## Data and Evaluation Pipeline

Both tasks generate fresh random sequences at every step, so models cannot memorize training data. The tasks share one vocabulary of 256 filler tokens, 1024 keys, and 512 values. Every sequence ends with a block of queries, and the model must output the value paired with each queried key earlier in the context. In multi-query associative recall (MQAR), the context holds T/4 key-value pairs, so longer contexts contain more pairs to track. In needle-in-a-haystack, the context holds 16 key-value pairs at random positions in random filler, so longer contexts contain more filler while the retrieval problem stays fixed. At T = 64 the two tasks are identical, which makes the comparison between them clean.

We evaluate every 250 steps on a fixed held-out set of 512 sequences. A run reaches the threshold at the first of three consecutive evaluations with at least 90% accuracy, so one noisy evaluation does not count. We convert steps to FLOPs with the analytic count of Kaplan et al. (2020), including the attention term that grows with context length. We also log accuracy by the position of the retrieved pair in the context, and the attention weight each head places on the correct value token. The second measure tells us when the model learns the retrieval pattern, since both tasks define exactly which token the model must attend to.

## Baselines

Random guessing scores 1/512. A model that copies a random value from the context, without matching keys, scores 1/16 on the needle task and 4/T on MQAR. This copying baseline is useful for interpreting plateaus: a model stuck above chance with attention spread evenly over the values has learned to copy but not to retrieve. For our main question, the baseline is the smallest model trained from scratch, and we ask how much compute each larger model saves or costs relative to it.

## First Experiments

Our first experiment tunes the learning rate for each of the four smallest sizes, sweeping 3e-4, 1e-3, and 3e-3 at T = 256 on both tasks (24 runs). Our second experiment is a reduced version of the main scale sweep: four sizes (0.4M to 25M parameters), context lengths 128, 256, 512, and 1024, both tasks, and 3 seeds (96 runs). Each run stops at the threshold or after 20k steps. The upper bound on compute is about 4 H100 hours, and we run four trainings per GPU to keep utilization high.

## Preliminary Results

Pilot runs on a laptop with the 3.2M model already show the abrupt learning that motivates our project. On MQAR at T = 64, accuracy sits near 20% until step 1500, then jumps and reaches 90% at step 4000 (Figure 1). Over the same steps, the attention weight of the best head on the correct value rises from 0.06 to 0.78. The jump in accuracy therefore coincides with the model learning the retrieval pattern, matching the account of Baherwani et al. (2026).

[Figure 1: `figures/pilot_mqar_T64.png`. Caption: Held-out accuracy (left) and attention weight of the best head on the correct value (right) during training on MQAR at T = 64, for the 0.4M model (learning rate 3e-3) and the 3.2M model (learning rate 1e-3), one seed each. The dashed line marks the 90% threshold. The 3.2M model's accuracy jump coincides with its attention moving onto the correct value; the 0.4M model has learned neither within 3000 steps.]

The pilots also exposed a problem with our original needle task, which we fixed in two steps. With a single key-value pair, the value is the only value token in the context, so the model can copy it without matching keys. With 4 pairs, the 3.2M model plateaued at 45% accuracy for 6000 steps at T = 64, with attention spread roughly evenly over the 4 values. The same model solved MQAR with 16 pairs by step 4000. We attribute the plateau to weak training signal, since the needle task scored only 4 answers per sequence against 16 for MQAR, and moved to 16 pairs. With 16 pairs at T = 128, the model sat at 20% accuracy from step 5000 to step 8000, with attention weight 0.04 on the correct value and flat accuracy across needle positions. The model has learned to copy but has not yet made the jump to retrieval within 8000 steps, which suggests long plateaus at longer contexts. Raising the learning rate to 3e-3 does not help: the same run reaches only 2.9% after 8000 steps, below the copying level. This sensitivity is why we tune the learning rate separately for each model size before comparing sizes.

[TODO after the Torch runs finish: report the learning rate chosen for each size. Then show `flops_to_threshold.png` and `steps_to_threshold.png` (Figure 2) and describe (1) whether larger models reach 90% in fewer steps, (2) whether they also use fewer FLOPs, and at which context lengths, (3) how many seeds failed to reach 90% within 20k steps in each setting, and (4) any difference between MQAR and needle. If many runs do not reach 90%, also report results at 50% accuracy with `python analyze.py <runs dir> --threshold 0.5`. Show `acc_by_depth.png` and state whether accuracy depends on needle position.]

## Next Steps

[TODO: adjust based on the MVP results.] Our next step is the full scale sweep with five sizes up to 60M parameters, context lengths up to 2048, and 5 seeds. In parallel, we will run the architecture sweeps that vary head count, head dimension, and the trade-off between depth and width at a fixed parameter count. The full sweep saves checkpoints of the 60M model at 30%, 60%, and 90% accuracy, which become the teachers for our distillation experiments. If the MVP shows that plateaus at long context exceed our 20k-step budget, we will raise the budget for long contexts or report time to a lower accuracy threshold.

## References

Baherwani, V., Chen, Z., Qiu, S., Wilson, A. G., and Izmailov, P. (2026). Emergent capabilities arise randomly from learning sparse attention patterns. arXiv:2606.25010.

Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
