# Summary (Nov 9)

- **Main sweep:** T = 256 is done (90 runs, 9 seeds). T = 512 and 1024 are running; T = 2048 has not started.
- **Model size:** success does not rise with model size at T = 256. The 10.6M model reaches 90% in 9 of 9 seeds on MQAR but only 1 of 9 on needle.
- **Cost per success:** the cheapest model is the smallest one that succeeds reliably:
  - needle: 0.4M, at 1.6 PFLOPs per success;
  - MQAR: 10.6M, at 13 PFLOPs, because 0.4M never succeeds.
- **Larger models:** 25M and 59.5M cost 12 to 100 times more per success than the cheapest size.
- **Depth at about 10M parameters:** shallow and wide wins clearly. 2 layers reach 90% in 5 of 5 seeds on both tasks, while 12 and 24 layers reach it in 0 of 5. The learning rate was not re-tuned per shape.
- **Head dimension:** at 16, no run succeeds. The number of heads at a fixed width has no clear effect with 5 seeds.
- **Most failures:** runs that never leave the copying level. The few runs between 50% and 90% are mostly still climbing at the 20k-step cap.
