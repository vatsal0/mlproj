"""Train one model on one task and record steps and FLOPs to the accuracy threshold.

Example:
    python train.py --task mqar --T 256 --size s --seed 0 --out runs/mvp
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import SIZES, ModelConfig, Transformer, n_params, train_flops_per_seq
from tasks import N_NEEDLES, VOCAB_SIZE, make_batch, num_queries

N_DEPTH_BUCKETS = 5
EVAL_SEED = 12345  # held-out set is identical across all runs
TRAIN_LOG_EVERY = 50  # steps between wandb logs of train loss, grad norm, and learning rate


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mqar", "needle"], required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--size", choices=list(SIZES), default=None)
    p.add_argument("--n_layer", type=int)  # explicit shape overrides --size (for architecture sweeps)
    p.add_argument("--d_model", type=int)
    p.add_argument("--n_heads", type=int)
    p.add_argument("--head_dim", type=int)
    p.add_argument("--n_needles", type=int, default=N_NEEDLES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=250)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=3)  # consecutive evals above threshold
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--save_at_acc", default="", help="e.g. 0.3,0.6,0.9: checkpoint when accuracy first "
                   "crosses each value (teachers from before, during, and after learning)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    return p.parse_args(argv)


def model_config(args):
    shape = dict(SIZES[args.size]) if args.size else {}
    for k in ("n_layer", "d_model", "n_heads", "head_dim"):
        if getattr(args, k) is not None:
            shape[k] = getattr(args, k)
    return ModelConfig(vocab_size=VOCAB_SIZE, max_len=args.T, **shape)


def run_name(args, cfg):
    if args.name:
        return args.name
    shape = args.size or f"L{cfg.n_layer}-d{cfg.d_model}-h{cfg.n_heads}x{cfg.head_dim}"
    return f"{args.task}_T{args.T}_{shape}_lr{args.lr:g}_s{args.seed}"


def setup_device():
    if torch.cuda.is_available():
        dev = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32
    return dev, dtype


@torch.no_grad()
def evaluate(model, eval_set, dev, dtype):
    model.eval()
    correct, total, n_layer = 0, 0, model.cfg.n_layer
    bucket_hit = torch.zeros(N_DEPTH_BUCKETS)
    bucket_n = torch.zeros(N_DEPTH_BUCKETS)
    mass = None
    for i, b in enumerate(eval_set):
        with torch.autocast(dev, dtype=dtype, enabled=dtype != torch.float32):
            # attention probe on the first batch only, which is enough for a stable average
            logits, attns = model(b["x"], b["qpos"], b["src"] if i == 0 else None)
        hit = (logits.argmax(-1) == b["ans"]).float()
        correct += hit.sum().item()
        total += hit.numel()
        bk = (b["depth"] * N_DEPTH_BUCKETS).long().clamp(max=N_DEPTH_BUCKETS - 1).flatten().cpu()
        bucket_hit.index_add_(0, bk, hit.flatten().cpu())
        bucket_n.index_add_(0, bk, torch.ones_like(bk, dtype=torch.float))
        if i == 0:
            # mean attention mass on the correct source, per layer and head
            mass = torch.stack([a.float().mean((0, 2)) for a in attns]).cpu()  # (L, H)
    model.train()
    return dict(
        acc=correct / total,
        acc_by_depth=(bucket_hit / bucket_n.clamp(min=1)).tolist(),
        attn_mass=mass.tolist(),
        attn_mass_max=mass.max().item(),
    )


def main(argv=None):
    args = get_args(argv)
    cfg = model_config(args)
    name = run_name(args, cfg)
    out = Path(args.out) / name
    if (out / "summary.json").exists():
        print(f"skip {name} (done)")
        return json.loads((out / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=True)

    dev, dtype = setup_device()
    torch.manual_seed(args.seed)
    model = Transformer(cfg).to(dev)
    N = n_params(model)
    nq = num_queries(args.task, args.T, args.n_needles)
    flops_per_step = args.batch * train_flops_per_seq(cfg, args.T, nq, N)

    g_eval = torch.Generator(dev).manual_seed(EVAL_SEED)
    eval_set = [make_batch(args.task, 64, args.T, g_eval, dev, args.n_needles) for _ in range(args.n_eval // 64)]
    g_train = torch.Generator(dev).manual_seed(1000 + args.seed)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    # linear warmup, then constant, so time to threshold does not depend on max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    fwd = torch.compile(model) if args.compile else model

    print(f"{name}: {N / 1e6:.2f}M non-emb params, {flops_per_step:.3g} FLOPs/step on {dev} ({dtype})")
    # wandb logging is on when WANDB_PROJECT is set (slurm/env.sh sets it), so laptop runs need no account
    use_wandb = "WANDB_PROJECT" in os.environ
    if use_wandb:
        import wandb
        wandb.init(name=name, group=Path(args.out).name, config=dict(
            vars(args), params=N, n_layer=cfg.n_layer, d_model=cfg.d_model, n_heads=cfg.n_heads,
            head_dim=cfg.head_dim, flops_per_step=flops_per_step, n_queries=nq))
        wandb.run.log_code(Path(__file__).parent)
    log = open(out / "log.jsonl", "w")
    milestones = sorted(float(m) for m in args.save_at_acc.split(",") if m)
    streak, hit_step, t0 = 0, None, time.time()
    time_last_train_log = t0
    for step in range(1, args.max_steps + 1):
        b = make_batch(args.task, args.batch, args.T, g_train, dev, args.n_needles)
        with torch.autocast(dev, dtype=dtype, enabled=dtype != torch.float32):
            logits, _ = fwd(b["x"], b["qpos"])
            loss = F.cross_entropy(logits.float().flatten(0, 1), b["ans"].flatten())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        learning_rate = sched.get_last_lr()[0]
        scaler.step(opt)
        scaler.update()
        sched.step()
        if use_wandb and step % TRAIN_LOG_EVERY == 0:
            time_now = time.time()
            wandb.log({"train/loss": loss.item(), "train/grad_norm": grad_norm.item(),
                       "train/learning_rate": learning_rate,
                       "train/seconds_per_step": (time_now - time_last_train_log) / TRAIN_LOG_EVERY,
                       "train/flops": step * flops_per_step}, step=step)
            time_last_train_log = time_now

        if step % args.eval_every and step != args.max_steps:
            continue
        ev = evaluate(model, eval_set, dev, dtype)
        rec = dict(step=step, flops=step * flops_per_step, loss=loss.item(), time=time.time() - t0, **ev)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  step {step:6d}  loss {loss.item():.3f}  acc {ev['acc']:.3f}  attn {ev['attn_mass_max']:.2f}")
        if use_wandb:
            eval_metrics = {"eval/accuracy": ev["acc"], "eval/attention_on_correct_value_best_head": ev["attn_mass_max"]}
            for bucket, accuracy in enumerate(ev["acc_by_depth"]):
                eval_metrics[f"eval/accuracy_by_depth_bucket_{bucket}"] = accuracy
            wandb.log(eval_metrics, step=step)
        if not math.isfinite(loss.item()):
            break
        while milestones and ev["acc"] >= milestones[0]:
            torch.save(model.state_dict(), out / f"ckpt_acc{milestones.pop(0):g}_step{step}.pt")
        streak = streak + 1 if ev["acc"] >= args.threshold else 0
        if streak == 1:
            first_above = step
        if streak >= args.patience and hit_step is None:
            hit_step = first_above
            if not args.no_early_stop:
                break

    summary = dict(
        vars(args), name=name, params=N, n_layer=cfg.n_layer, d_model=cfg.d_model, n_heads=cfg.n_heads,
        head_dim=cfg.head_dim, flops_per_step=flops_per_step, reached=hit_step is not None,
        steps_to_threshold=hit_step, flops_to_threshold=hit_step * flops_per_step if hit_step else None,
        final_acc=ev["acc"], last_step=step, wall_time=time.time() - t0,
    )
    torch.save(model.state_dict(), out / "final.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if use_wandb:
        wandb.summary.update(summary)
        wandb.finish()
    return summary


if __name__ == "__main__":
    main()
