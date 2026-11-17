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
    p.add_argument("--seed", type=int, default=0)  # initialization
    p.add_argument("--data_seed", type=int, default=None, help="order of training batches; defaults to --seed")
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
    p.add_argument("--teacher", default=None, help="checkpoint from a --save_at_acc run (next to its summary.json); "
                   "the student then trains on a mix of cross-entropy and KL divergence to the teacher's predictions")
    p.add_argument("--distill_weight", type=float, default=0.5, help="weight of the KL term; cross-entropy gets 1 - this")
    p.add_argument("--distill_temperature", type=float, default=1.0)
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
    # the data seed appears in the name only when it differs, so names match runs from before --data_seed existed
    data_suffix = f"_d{args.data_seed}" if args.data_seed not in (None, args.seed) else ""
    # e.g. "_teacher_xl_s1_acc0.3_weight0.5_temperature1": the teacher's size and seed, the accuracy its checkpoint
    # was saved at, and the distillation settings
    teacher_suffix = ""
    if args.teacher:
        teacher_summary = json.loads((Path(args.teacher).parent / "summary.json").read_text())
        teacher_accuracy = Path(args.teacher).stem.split("_")[1]  # "ckpt_acc0.3_step13250" -> "acc0.3"
        teacher_suffix = (f"_teacher_{teacher_summary['size']}_s{teacher_summary['seed']}_{teacher_accuracy}"
                          f"_weight{args.distill_weight:g}_temperature{args.distill_temperature:g}")
    return f"{args.task}_T{args.T}_{shape}_lr{args.lr:g}_s{args.seed}{data_suffix}{teacher_suffix}"


def load_teacher(path, args, n_queries, eval_set, dev, dtype):
    """The frozen teacher model, and a dict with its held-out accuracy, the compute spent training it up to this
    checkpoint, and the compute of its forward pass on one training batch of the student."""
    teacher_summary = json.loads((Path(path).parent / "summary.json").read_text())
    shape = {k: teacher_summary[k] for k in ("n_layer", "d_model", "n_heads", "head_dim")}
    teacher = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, max_len=teacher_summary["T"], **shape)).to(dev)
    teacher.load_state_dict(torch.load(path, map_location=dev))
    teacher.requires_grad_(False)
    teacher_accuracy = evaluate(teacher, eval_set, dev, dtype)["acc"]
    teacher.eval()  # evaluate() leaves the model in training mode
    teacher_step = int(Path(path).stem.split("_step")[1])  # "ckpt_acc0.3_step13250" -> 13250
    # a training step costs a forward and a backward pass, about 3 forward passes (Kaplan et al., 2020)
    teacher_train_flops_per_seq = train_flops_per_seq(teacher.cfg, args.T, n_queries, n_params(teacher))
    teacher_info = dict(teacher_accuracy=teacher_accuracy,
                        teacher_training_flops=teacher_step * teacher_summary["flops_per_step"],
                        teacher_forward_flops_per_step=args.batch * teacher_train_flops_per_seq / 3)
    return teacher, teacher_info


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
    resume_path = out / "resume.pt"  # training state saved at every evaluation, so a killed run continues

    dev, dtype = setup_device()
    torch.manual_seed(args.seed)
    model = Transformer(cfg).to(dev)
    N = n_params(model)
    nq = num_queries(args.task, args.T, args.n_needles)
    flops_per_step = args.batch * train_flops_per_seq(cfg, args.T, nq, N)

    g_eval = torch.Generator(dev).manual_seed(EVAL_SEED)
    eval_set = [make_batch(args.task, 64, args.T, g_eval, dev, args.n_needles) for _ in range(args.n_eval // 64)]
    teacher = None
    teacher_info = {}  # goes into the wandb config and summary
    if args.teacher:
        teacher, teacher_info = load_teacher(args.teacher, args, nq, eval_set, dev, dtype)
        print(f"teacher {args.teacher}: held-out accuracy {teacher_info['teacher_accuracy']:.3f}")
    data_seed = args.seed if args.data_seed is None else args.data_seed
    g_train = torch.Generator(dev).manual_seed(1000 + data_seed)

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

    milestones = sorted(float(m) for m in args.save_at_acc.split(",") if m)
    streak = 0  # consecutive evaluations at or above the threshold
    first_above = None  # step of the first evaluation in the current streak
    hit_step = None
    start_step = 0
    n_log_lines = 0  # evaluations written to log.jsonl
    wall_time_before_resume = 0.0
    wandb_run_id = None
    if resume_path.exists():
        state = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        g_train.set_state(state["train_generator"].cpu())
        start_step = state["step"]
        streak = state["streak"]
        first_above = state["first_above"]
        hit_step = state["hit_step"]
        milestones = state["milestones"]
        n_log_lines = state["n_log_lines"]
        wall_time_before_resume = state["wall_time"]
        wandb_run_id = state["wandb_run_id"]
        # drop log lines written after the saved state (possibly a half-written one), which this run will repeat
        log_lines = (out / "log.jsonl").read_text().splitlines()[:n_log_lines]
        (out / "log.tmp").write_text("".join(line + "\n" for line in log_lines))
        os.replace(out / "log.tmp", out / "log.jsonl")
        print(f"resuming {name} from step {start_step}")

    print(f"{name}: {N / 1e6:.2f}M non-emb params, {flops_per_step:.3g} FLOPs/step on {dev} ({dtype})")
    # wandb logging is on when WANDB_PROJECT is set (slurm/env.sh sets it), so laptop runs need no account
    use_wandb = "WANDB_PROJECT" in os.environ
    if use_wandb:
        import wandb
        wandb.init(name=name, group=Path(args.out).name, id=wandb_run_id, resume="allow", config=dict(
            vars(args), params=N, n_layer=cfg.n_layer, d_model=cfg.d_model, n_heads=cfg.n_heads,
            head_dim=cfg.head_dim, flops_per_step=flops_per_step, n_queries=nq, **teacher_info))
        wandb_run_id = wandb.run.id
        wandb.run.log_code(Path(__file__).parent)
    log = open(out / "log.jsonl", "a")
    t0 = time.time() - wall_time_before_resume
    time_last_train_log = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        b = make_batch(args.task, args.batch, args.T, g_train, dev, args.n_needles)
        with torch.autocast(dev, dtype=dtype, enabled=dtype != torch.float32):
            logits, _ = fwd(b["x"], b["qpos"])
            cross_entropy = F.cross_entropy(logits.float().flatten(0, 1), b["ans"].flatten())
            loss = cross_entropy
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits, _ = teacher(b["x"], b["qpos"])
                temperature = args.distill_temperature
                # KL(teacher || student) per answer, scaled by temperature^2 so its gradient size does not depend on
                # the temperature (Hinton et al., 2015)
                kl_to_teacher = F.kl_div(F.log_softmax(logits.float().flatten(0, 1) / temperature, -1),
                                         F.log_softmax(teacher_logits.float().flatten(0, 1) / temperature, -1),
                                         log_target=True, reduction="batchmean") * temperature**2
                loss = (1 - args.distill_weight) * cross_entropy + args.distill_weight * kl_to_teacher
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
            train_metrics = {"train/loss": loss.item(), "train/cross_entropy": cross_entropy.item(),
                             "train/grad_norm": grad_norm.item(), "train/learning_rate": learning_rate,
                             "train/seconds_per_step": (time_now - time_last_train_log) / TRAIN_LOG_EVERY,
                             "train/flops": step * flops_per_step}
            if teacher is not None:
                train_metrics["train/kl_to_teacher"] = kl_to_teacher.item()
            wandb.log(train_metrics, step=step)
            time_last_train_log = time_now

        if step % args.eval_every and step != args.max_steps:
            continue
        ev = evaluate(model, eval_set, dev, dtype)
        rec = dict(step=step, flops=step * flops_per_step, loss=loss.item(), cross_entropy=cross_entropy.item(),
                   time=time.time() - t0, **ev)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        n_log_lines += 1
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
        # no save after the last step: a run resumed there would have no steps left and no summary to write
        if step < args.max_steps:
            state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                         scaler=scaler.state_dict(), train_generator=g_train.get_state(), step=step, streak=streak,
                         first_above=first_above, hit_step=hit_step, milestones=milestones, n_log_lines=n_log_lines,
                         wall_time=time.time() - t0, wandb_run_id=wandb_run_id)
            torch.save(state, out / "resume.tmp")
            os.replace(out / "resume.tmp", resume_path)  # atomic, so a kill mid-save keeps the previous state

    summary = dict(
        vars(args), name=name, params=N, n_layer=cfg.n_layer, d_model=cfg.d_model, n_heads=cfg.n_heads,
        head_dim=cfg.head_dim, flops_per_step=flops_per_step, reached=hit_step is not None,
        steps_to_threshold=hit_step, flops_to_threshold=hit_step * flops_per_step if hit_step else None,
        final_acc=ev["acc"], last_step=step, wall_time=time.time() - t0, **teacher_info,
    )
    torch.save(model.state_dict(), out / "final.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    resume_path.unlink(missing_ok=True)
    if use_wandb:
        wandb.summary.update(summary)
        wandb.finish()
    return summary


if __name__ == "__main__":
    main()
