"""Run a named grid of experiments, skipping runs that already finished.

Split a grid across jobs with --shard i --n_shards n. Small models leave a GPU mostly idle, so
--parallel k runs k trainings on the same GPU at once (Torch cancels jobs with low GPU utilization).
    python sweep.py --grid lr --out runs/lr --parallel 4
    python sweep.py --grid mvp --out runs/mvp --shard 0 --n_shards 8 --parallel 4
    python sweep.py --grid mvp --dry   # print commands only
"""
import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import train

# per-size learning rates; overwrite with the winners of the lr grid (see analyze.py --best_lr)
LR = {"xs": 3e-3, "s": 1e-3, "m": 1e-3, "l": 1e-3, "xl": 1e-3}

# architecture sweeps around the "m" model (6 layers, width 384, ~10.6M params)
HEADS = [dict(n_layer=6, d_model=384, n_heads=h, head_dim=384 // h) for h in (2, 3, 6, 12, 24)]
HEAD_DIM = [dict(n_layer=6, d_model=384, n_heads=6, head_dim=hd) for hd in (16, 32, 64, 128)]
DEPTH = [dict(n_layer=L, d_model=d, n_heads=d // 64, head_dim=64)
         for L, d in ((2, 640), (4, 448), (6, 384), (12, 256), (24, 192))]


def grid(name):
    """Yields argument lists for train.py."""
    tasks = ["mqar", "needle"]
    if name == "smoke":
        for t in tasks:
            yield ["--task", t, "--T", "64", "--size", "xs", "--max_steps", "500", "--eval_every", "100"]
    elif name == "lr":
        for t, s, lr in itertools.product(tasks, ["xs", "s", "m", "l"], [3e-4, 1e-3, 3e-3]):
            yield ["--task", t, "--T", "256", "--size", s, "--lr", str(lr)]
        # 3e-4, the lowest rate above, won for s, m, and l, so we also try 1e-4 for those sizes
        for t, s in itertools.product(tasks, ["s", "m", "l"]):
            yield ["--task", t, "--T", "256", "--size", s, "--lr", "0.0001"]
    elif name == "lr_full":
        for t, s, lr in itertools.product(tasks, list(LR), [3e-4, 1e-3, 3e-3]):
            yield ["--task", t, "--T", "512", "--size", s, "--lr", str(lr)]
    elif name == "mvp":
        for t, T, s, seed in itertools.product(tasks, [128, 256, 512, 1024], ["xs", "s", "m", "l"], range(3)):
            yield ["--task", t, "--T", str(T), "--size", s, "--lr", str(LR[s]), "--seed", str(seed)]
    elif name == "main":
        for t, T, s, seed in itertools.product(tasks, [256, 512, 1024, 2048], list(LR), range(5)):
            extra = ["--save_at_acc", "0.3,0.6,0.9"] if s == "xl" else []  # distillation teachers
            yield ["--task", t, "--T", str(T), "--size", s, "--lr", str(LR[s]), "--seed", str(seed)] + extra
    elif name in ("heads", "head_dim", "depth"):
        shapes = dict(heads=HEADS, head_dim=HEAD_DIM, depth=DEPTH)[name]
        for t, T, shape, seed in itertools.product(tasks, [512, 2048], shapes, range(5)):
            yield ["--task", t, "--T", str(T), "--lr", str(LR["m"]), "--seed", str(seed)] + [
                a for k, v in shape.items() for a in (f"--{k}", str(v))]
    else:
        raise ValueError(name)


def launch(r, i, n, dry):
    print(f"[{i + 1}/{n}] python train.py " + " ".join(r), flush=True)
    if dry:
        return 0
    out = Path(r[r.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"stdout_{i}.txt", "w") as f:
        return subprocess.call([sys.executable, "train.py"] + r, stdout=f, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--parallel", type=int, default=1, help="concurrent runs on one GPU")
    p.add_argument("--lr_json", default=None, help="json mapping size -> lr, overrides LR")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if a.lr_json:
        LR.update(json.load(open(a.lr_json)))
    runs = [r for i, r in enumerate(grid(a.grid)) if i % a.n_shards == a.shard]
    runs = [r + ["--out", a.out or f"runs/{a.grid}"] + (["--compile"] if a.compile else []) for r in runs]
    print(f"{len(runs)} runs in shard {a.shard}/{a.n_shards}, {a.parallel} at a time")
    with ThreadPoolExecutor(a.parallel) as ex:
        codes = list(ex.map(lambda ir: launch(ir[1], ir[0], len(runs), a.dry), enumerate(runs)))
    failed = sum(c != 0 for c in codes)
    print(f"done, {failed} failed" + (" (see stdout_*.txt)" if failed else ""))
