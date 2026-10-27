"""Collect run summaries and make the core plots.

    python analyze.py runs/mvp                 # summary table + figures in runs/mvp/figures
    python analyze.py runs/lr --best_lr lr.json  # pick the best lr per size, for sweep.py --lr_json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# single-hue ordinal ramp (light -> dark = short -> long context); markers double-encode length
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
MARKERS = ["o", "s", "^", "D"]
SIZE_ORDER = ["xs", "s", "m", "l", "xl"]


def ramp(i, n):
    """i-th of n ordered colors, spread across the full ramp so few lines still contrast."""
    return RAMP[round(i * (len(RAMP) - 1) / max(n - 1, 1))]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})


def load(root, threshold=None):
    """Loads finished runs. If threshold is given (at most the training threshold, since runs stop
    there), steps and FLOPs to threshold are recomputed from the logs with the same rule as train.py."""
    runs = []
    for f in Path(root).glob("*/summary.json"):
        s = json.loads(f.read_text())
        s["log"] = [json.loads(l) for l in (f.parent / "log.jsonl").read_text().splitlines()]
        if threshold is not None:
            hit, streak = None, 0
            for l in s["log"]:
                streak = streak + 1 if l["acc"] >= threshold else 0
                if streak == 1:
                    first = l["step"]
                if streak >= s["patience"]:
                    hit = first
                    break
            s.update(reached=hit is not None, steps_to_threshold=hit,
                     flops_to_threshold=hit * s["flops_per_step"] if hit else None)
        runs.append(s)
    return runs


def group(runs, keys):
    g = defaultdict(list)
    for r in runs:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def table(runs):
    print(f"{'task':7s} {'T':>5s} {'size':>5s} {'lr':>7s} {'params':>7s} {'reached':>8s} "
          f"{'med steps':>10s} {'med PFLOPs':>11s} {'final acc':>10s}")
    for (task, T, size, lr), rs in sorted(group(runs, ["task", "T", "size", "lr"]).items(), key=str):
        hit = [r for r in rs if r["reached"]]
        steps = np.median([r["steps_to_threshold"] for r in hit]) if hit else float("nan")
        fl = np.median([r["flops_to_threshold"] for r in hit]) / 1e15 if hit else float("nan")
        acc = np.mean([r["final_acc"] for r in rs])
        print(f"{task:7s} {T:5d} {str(size):>5s} {lr:7.0e} {rs[0]['params'] / 1e6:6.2f}M "
              f"{len(hit):>4d}/{len(rs):<3d} {steps:10.0f} {fl:11.3f} {acc:10.3f}")


def best_lr(runs, path):
    """For each size, the lr with the lowest median steps to threshold (ties broken by final acc)."""
    out = {}
    for size, rs in group(runs, ["size"]).items():
        def score(lr):
            sub = [r for r in rs if r["lr"] == lr]
            st = [r["steps_to_threshold"] or r["last_step"] * 10 for r in sub]  # penalize failures
            return np.median(st), -np.mean([r["final_acc"] for r in sub])
        out[size[0]] = min({r["lr"] for r in rs}, key=score)
    Path(path).write_text(json.dumps(out, indent=1))
    print("best lr:", out)


def plot_cost(runs, metric, fig_dir):
    """Steps or FLOPs to threshold vs model size, one line per context length, one panel per task.
    Points are seed medians, bars span min to max over seeds that reached the threshold."""
    tasks = sorted({r["task"] for r in runs})
    Ts = sorted({r["T"] for r in runs})
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        label_positions = []  # (log10 x, log10 y) of seed-count labels already placed on this panel
        for i, T in enumerate(Ts):
            pts = []
            for (size,), rs in group([r for r in runs if r["task"] == task and r["T"] == T], ["size"]).items():
                vals = [r[metric] for r in rs if r["reached"]]
                if vals:
                    pts.append((rs[0]["params"], np.median(vals), min(vals), max(vals), len(vals), len(rs)))
            if not pts:
                continue
            p, med, lo, hi, k, n = map(np.array, zip(*sorted(pts)))
            ax.errorbar(p, med, yerr=[med - lo, hi - med], color=ramp(i, len(Ts)), marker=MARKERS[i % 4],
                        ms=7, lw=2, capsize=3, label=f"T = {T}")
            for x, y, kk, nn in zip(p, med, k, n):  # flag settings where some seeds failed
                if kk < nn:  # shift a label up when another label already sits next to this point
                    n_nearby = sum(abs(np.log10(x) - lx) < 0.15 and abs(np.log10(y) - ly) < 0.15
                                   for lx, ly in label_positions)
                    label_positions.append((np.log10(x), np.log10(y)))
                    ax.annotate(f"{kk}/{nn}", (x, y), textcoords="offset points", xytext=(6, 4 + 11 * n_nearby),
                                fontsize=9, color=ramp(i, len(Ts)))
        ax.set(xscale="log", yscale="log", title=task, xlabel="non-embedding parameters",
               ylabel=metric.replace("_", " "))
    # one legend for both panels, since a context length can have points in only one of them
    handles_by_label = {label: handle for ax in axes[0] for handle, label in zip(*ax.get_legend_handles_labels())}
    labels = sorted(handles_by_label, key=lambda label: int(label.split("= ")[1]))
    axes[0][-1].legend([handles_by_label[label] for label in labels], labels, title="context length",
                       loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / f"{metric}.pdf")
    fig.savefig(fig_dir / f"{metric}.png", dpi=150)
    plt.close(fig)


def plot_curves(runs, fig_dir):
    """Held-out accuracy over training for every (task, T), one color per size, one line per seed."""
    for (task, T), rs in group(runs, ["task", "T"]).items():
        sizes = [s for s in SIZE_ORDER if any(r["size"] == s for r in rs)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for r in rs:
            if r["size"] not in sizes:
                continue
            i = sizes.index(r["size"])
            st = [l["step"] for l in r["log"]]
            kw = dict(color=ramp(i, len(sizes)), lw=1.5, alpha=0.8, label=r["size"] if r["seed"] == 0 else None)
            axes[0].plot(st, [l["acc"] for l in r["log"]], **kw)
            axes[1].plot(st, [l["attn_mass_max"] for l in r["log"]], **kw)
        axes[0].axhline(0.9, color="gray", ls="--", lw=1)
        axes[0].set(xlabel="step", ylabel="held-out accuracy", title=f"{task}, T = {T}")
        axes[1].set(xlabel="step", ylabel="max attention on correct source", title="best head")
        axes[0].legend(title="size")
        fig.tight_layout()
        fig.savefig(fig_dir / f"curves_{task}_T{T}.png", dpi=150)
        plt.close(fig)


def plot_depth(runs, fig_dir):
    """Final accuracy by relative position of the retrieved pair (lost-in-the-middle check)."""
    rs = [r for r in runs if r["task"] == "needle"]
    if not rs:
        return
    Ts = sorted({r["T"] for r in rs})
    sizes = [s for s in SIZE_ORDER if any(r["size"] == s for r in rs)]
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.5 * len(sizes), 3.8), squeeze=False, sharey=True)
    for ax, size in zip(axes[0], sizes):
        for i, T in enumerate(Ts):
            sub = [r["log"][-1]["acc_by_depth"] for r in rs if r["size"] == size and r["T"] == T]
            if sub:
                y = np.mean(sub, 0)
                x = (np.arange(len(y)) + 0.5) / len(y)
                ax.plot(x, y, color=ramp(i, len(Ts)), marker=MARKERS[i % 4], ms=7, lw=2, label=f"T = {T}")
        ax.set(title=f"needle, size {size}", xlabel="relative needle position", ylim=(0, 1.02))
    axes[0][0].set_ylabel("final accuracy")
    axes[0][-1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "acc_by_depth.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--best_lr", default=None)
    p.add_argument("--threshold", type=float, default=None, help="recompute at a lower accuracy threshold")
    a = p.parse_args()
    runs = load(a.root, a.threshold)
    print(f"{len(runs)} finished runs")
    table(runs)
    if a.best_lr:
        best_lr(runs, a.best_lr)
    fig_dir = Path(a.root) / ("figures" if a.threshold is None else f"figures_thr{a.threshold:g}")
    fig_dir.mkdir(exist_ok=True)
    plot_cost(runs, "steps_to_threshold", fig_dir)
    plot_cost(runs, "flops_to_threshold", fig_dir)
    plot_curves(runs, fig_dir)
    plot_depth(runs, fig_dir)
