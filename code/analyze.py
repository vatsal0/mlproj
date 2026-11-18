"""Collect run summaries and make the core plots.

    python analyze.py runs/mvp                 # summary table + figures in runs/mvp/figures
    python analyze.py runs/lr --best_lr lr.json  # pick the best lr per size, for sweep.py --lr_json
    python analyze.py runs/distillation --baseline runs/main  # distilled students against training from scratch

Runs that never reach the threshold still cost compute, so besides the cost of the runs that succeed we report
the success probability and the expected compute per success: all compute spent, failures included, divided
by the number of successes. This is the cost of retrying until a run succeeds.
"""
import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# single-hue ordinal ramp (light -> dark = shorter -> longer context, or smaller -> larger model), steps 250 to 700
# of the reference blue ramp; ramp() spreads n lines evenly across it. Markers double-encode context length.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
MARKERS = ["o", "s", "^", "D"]
SIZE_ORDER = ["xs", "s", "m", "l", "xl"]
# blue and orange, the first two categorical colors of the dataviz skill's reference palette
TASK_STYLES = {"mqar": dict(color="#2a78d6", marker="o"), "needle": dict(color="#eb6834", marker="s")}


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
        # architecture-sweep runs have no named size, so they are labelled by their shape
        s["size"] = s["size"] or f"L{s['n_layer']}-d{s['d_model']}-h{s['n_heads']}x{s['head_dim']}"
        # a successful run's cost is its compute up to the threshold; a failed run's is everything it trained
        s["flops_spent"] = s["flops_to_threshold"] if s["reached"] else s["last_step"] * s["flops_per_step"]
        if s.get("teacher"):  # a distilled student also ran the teacher forward at every step it trained
            steps_trained = s["steps_to_threshold"] if s["reached"] else s["last_step"]
            s["teacher_forward_flops_spent"] = steps_trained * s["teacher_forward_flops_per_step"]
            # the accuracy at which the teacher checkpoint was saved, from its name: "ckpt_acc0.3_step13250" -> 0.3
            s["teacher_saved_at_accuracy"] = float(Path(s["teacher"]).stem.split("_")[1].removeprefix("acc"))
        runs.append(s)
    return runs


def group(runs, keys):
    g = defaultdict(list)
    for r in runs:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def wilson_interval(n_reached, n_runs, z=1.96):
    """95% confidence interval for a success probability (Wilson score interval)."""
    p = n_reached / n_runs
    center = (p + z**2 / (2 * n_runs)) / (1 + z**2 / n_runs)
    half_width = z * math.sqrt(p * (1 - p) / n_runs + z**2 / (4 * n_runs**2)) / (1 + z**2 / n_runs)
    return max(0.0, center - half_width), min(1.0, center + half_width)  # rounding can step just outside [0, 1]


def expected_flops_per_success(rs):
    """All compute spent by the runs divided by the number that reached the threshold. With no success,
    the total spent is a lower bound on the cost of one success."""
    n_reached = sum(r["reached"] for r in rs)
    return sum(r["flops_spent"] for r in rs) / max(n_reached, 1)


def table(runs):
    print(f"{'task':7s} {'T':>5s} {'size':>14s} {'lr':>7s} {'params':>7s} {'reached':>8s} "
          f"{'med steps':>10s} {'med PFLOPs':>11s} {'final acc':>10s} {'PFLOPs per success':>19s}")
    for (task, T, size, lr), rs in sorted(group(runs, ["task", "T", "size", "lr"]).items(), key=str):
        hit = [r for r in rs if r["reached"]]
        steps = np.median([r["steps_to_threshold"] for r in hit]) if hit else float("nan")
        fl = np.median([r["flops_to_threshold"] for r in hit]) / 1e15 if hit else float("nan")
        acc = np.mean([r["final_acc"] for r in rs])
        per_success = (">" if not hit else "") + f"{expected_flops_per_success(rs) / 1e15:.3f}"
        print(f"{task:7s} {T:5d} {str(size):>14s} {lr:7.0e} {rs[0]['params'] / 1e6:6.2f}M "
              f"{len(hit):>4d}/{len(rs):<3d} {steps:10.0f} {fl:11.3f} {acc:10.3f} {per_success:>19s}")


def write_run_table(runs, path):
    """One row per run, for analysis in a spreadsheet or pandas."""
    summary_columns = ["name", "task", "T", "size", "params", "n_layer", "d_model", "n_heads", "head_dim", "lr",
                       "seed", "max_steps", "reached", "steps_to_threshold", "flops_to_threshold", "last_step",
                       "flops_spent", "final_acc"]
    teacher_columns = (["teacher_saved_at_accuracy", "teacher_accuracy", "teacher_training_flops",
                        "teacher_forward_flops_spent"] if any(run.get("teacher") for run in runs) else [])
    log_columns = ["max_acc", "first_step_above_half_accuracy", "final_attention_on_correct_value"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, summary_columns + teacher_columns + log_columns)
        writer.writeheader()
        for run in sorted(runs, key=lambda run: run["name"]):
            steps_above_half = [record["step"] for record in run["log"] if record["acc"] >= 0.5]
            writer.writerow({**{column: run[column] for column in summary_columns},
                             **{column: run.get(column) for column in teacher_columns},
                             "max_acc": max(record["acc"] for record in run["log"]),
                             "first_step_above_half_accuracy": steps_above_half[0] if steps_above_half else None,
                             "final_attention_on_correct_value": run["log"][-1]["attn_mass_max"]})


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


def plot_success_and_cost(runs, fig_dir, threshold):
    """Top: probability of reaching the threshold, with 95% intervals. Bottom: expected FLOPs per success,
    one line per context length, one column per task. Hollow markers are lower bounds (no run succeeded)."""
    tasks = sorted({r["task"] for r in runs})
    Ts = sorted({r["T"] for r in runs})
    fig, axes = plt.subplots(2, len(tasks), figsize=(5.5 * len(tasks), 7.5), squeeze=False, sharex=True)
    for column, task in enumerate(tasks):
        for i, T in enumerate(Ts):
            # one setting per size; the main grid has one learning rate per size
            settings = sorted(group([r for r in runs if r["task"] == task and r["T"] == T], ["size"]).values(),
                              key=lambda seed_runs: seed_runs[0]["params"])
            if not settings:
                continue
            # shift each context length slightly sideways so their error bars do not overlap
            params = [seed_runs[0]["params"] * 1.08 ** (i - (len(Ts) - 1) / 2) for seed_runs in settings]
            n_reached = [sum(r["reached"] for r in seed_runs) for seed_runs in settings]
            probability = [n / len(seed_runs) for n, seed_runs in zip(n_reached, settings)]
            intervals = [wilson_interval(n, len(seed_runs)) for n, seed_runs in zip(n_reached, settings)]
            style = dict(color=ramp(i, len(Ts)), marker=MARKERS[i % 4], ms=7, lw=2)
            axes[0][column].errorbar(params, probability, yerr=[[p - lo for p, (lo, hi) in zip(probability, intervals)],
                                     [hi - p for p, (lo, hi) in zip(probability, intervals)]],
                                     capsize=3, elinewidth=1, ecolor=style["color"] + "99", label=f"T = {T}", **style)
            cost = [expected_flops_per_success(seed_runs) for seed_runs in settings]
            axes[1][column].plot(params, cost, **style)
            lower_bounds = [(x, y) for x, y, n in zip(params, cost, n_reached) if n == 0]
            if lower_bounds:
                axes[1][column].scatter(*zip(*lower_bounds), s=60, facecolors="white", edgecolors=style["color"],
                                        linewidths=2, zorder=3, marker=style["marker"])
        axes[0][column].set(xscale="log", title=task, ylim=(-0.03, 1.03), ylabel=f"probability of reaching {threshold:.0%}")
        axes[1][column].set(xscale="log", yscale="log", xlabel="non-embedding parameters",
                            ylabel="expected FLOPs per success")
    axes[0][-1].legend(title="context length", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "success_and_cost.png", dpi=150)
    plt.close(fig)


def plot_reached_vs_flops(runs, fig_dir, threshold):
    """Fraction of runs that have reached the threshold against FLOPs spent, one line per size,
    one panel per (task, context length). Each line ends at the largest compute any of its runs spent.
    Only the named sizes are drawn, so architecture-sweep runs are left out."""
    tasks = sorted({r["task"] for r in runs})
    Ts = sorted({r["T"] for r in runs})
    sizes = [s for s in SIZE_ORDER if any(r["size"] == s for r in runs)]
    fig, axes = plt.subplots(len(tasks), len(Ts), figsize=(3.6 * len(Ts), 3.2 * len(tasks)), squeeze=False,
                             sharey=True)
    for (row, task), (column, T) in itertools.product(enumerate(tasks), enumerate(Ts)):
        ax = axes[row][column]
        for i, size in enumerate(sizes):
            rs = [r for r in runs if r["task"] == task and r["T"] == T and r["size"] == size]
            if not rs:
                continue
            success_flops = sorted(r["flops_to_threshold"] for r in rs if r["reached"])
            first_evaluation_flops = rs[0]["eval_every"] * rs[0]["flops_per_step"]
            x = [first_evaluation_flops] + success_flops + [max(r["flops_spent"] for r in rs)]
            y = [0] + [(k + 1) / len(rs) for k in range(len(success_flops))] + [len(success_flops) / len(rs)]
            ax.step(x, y, where="post", color=ramp(i, len(sizes)), lw=2, label=size)
        ax.set(xscale="log", title=f"{task}, T = {T}", ylim=(-0.03, 1.03))
        if row == len(tasks) - 1:
            ax.set_xlabel("FLOPs")
    for row in range(len(tasks)):
        axes[row][0].set_ylabel(f"fraction reached {threshold:.0%}")
    axes[0][-1].legend(title="size", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "reached_vs_flops.png", dpi=150)
    plt.close(fig)


def plot_architecture(runs, fig_dir, threshold):
    """Architecture sweeps around the 10.6M model (6 layers, width 384, 6 heads of dimension 64), one column per
    sweep. Top: probability of reaching the threshold, with 95% intervals. Bottom: expected FLOPs per success;
    hollow markers are lower bounds (no run succeeded)."""
    sweeps = [  # (title, x-axis label, the varied shape field, which runs belong to the sweep)
        ("heads (width 384, 6 layers)", "number of heads", "n_heads",
         lambda r: r["n_layer"] == 6 and r["d_model"] == 384 and r["n_heads"] * r["head_dim"] == 384),
        ("head dimension (6 heads, 6 layers)", "head dimension", "head_dim",
         lambda r: r["n_layer"] == 6 and r["d_model"] == 384 and r["n_heads"] == 6),
        ("depth (head dimension 64, 10M parameters)", "number of layers", "n_layer",
         lambda r: r["head_dim"] == 64 and r["n_heads"] * r["head_dim"] == r["d_model"]),
    ]
    fig, axes = plt.subplots(2, len(sweeps), figsize=(5 * len(sweeps), 7.5), squeeze=False, sharex="col")
    for column, (title, x_label, varied_field, in_sweep) in enumerate(sweeps):
        sweep_values = sorted({r[varied_field] for r in runs if in_sweep(r)})
        for task in sorted({r["task"] for r in runs}):
            settings = sorted(group([r for r in runs if r["task"] == task and in_sweep(r)], [varied_field]).items())
            if not settings:
                continue
            x = [value for (value,), seed_runs in settings]
            n_reached = [sum(r["reached"] for r in seed_runs) for _, seed_runs in settings]
            n_runs = [len(seed_runs) for _, seed_runs in settings]
            probability = [n / n_runs_at_value for n, n_runs_at_value in zip(n_reached, n_runs)]
            intervals = [wilson_interval(n, n_runs_at_value) for n, n_runs_at_value in zip(n_reached, n_runs)]
            style = dict(**TASK_STYLES[task], ms=7, lw=2)
            axes[0][column].errorbar(x, probability, yerr=[[p - lo for p, (lo, hi) in zip(probability, intervals)],
                                     [hi - p for p, (lo, hi) in zip(probability, intervals)]],
                                     capsize=3, elinewidth=1, ecolor=style["color"] + "99", label=task, **style)
            cost = [expected_flops_per_success(seed_runs) for _, seed_runs in settings]
            axes[1][column].plot(x, cost, **style)
            lower_bounds = [(value, y) for value, y, n in zip(x, cost, n_reached) if n == 0]
            if lower_bounds:
                axes[1][column].scatter(*zip(*lower_bounds), s=60, facecolors="white", edgecolors=style["color"],
                                        linewidths=2, zorder=3, marker=style["marker"])
        axes[0][column].set(xscale="log", title=title, ylim=(-0.03, 1.03))
        axes[1][column].set(xscale="log", yscale="log", xlabel=x_label)
        axes[1][column].set_xticks(sweep_values, [str(value) for value in sweep_values])
        axes[1][column].minorticks_off()
    axes[0][0].set_ylabel(f"probability of reaching {threshold:.0%}")
    axes[1][0].set_ylabel("expected FLOPs per success")
    axes[0][-1].legend(title="task", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "architecture.png", dpi=150)
    plt.close(fig)


def distillation_costs(seed_runs):
    """Expected FLOPs per successful student, counted three ways: student training only; plus the teacher's
    forward passes during distillation; plus training the teacher up to its checkpoint, charged in full to one
    student. For students trained from scratch, all three are the same."""
    n_successes = max(sum(r["reached"] for r in seed_runs), 1)
    student_only = expected_flops_per_success(seed_runs)
    with_teacher_forward = student_only + sum(r.get("teacher_forward_flops_spent", 0) for r in seed_runs) / n_successes
    with_teacher_training = with_teacher_forward + seed_runs[0].get("teacher_training_flops", 0)
    return student_only, with_teacher_forward, with_teacher_training


def distillation_settings(distilled_runs, baseline_runs):
    """Returns the labels, training from scratch first and then each teacher checkpoint in order of the accuracy
    it was saved at, and {(task, size, label): seed runs}."""
    accuracies = sorted({r["teacher_saved_at_accuracy"] for r in distilled_runs})
    labels = ["from scratch"] + [f"teacher at {accuracy:.0%}" for accuracy in accuracies]
    settings = {(task, size, "from scratch"): seed_runs
                for (task, size), seed_runs in group(baseline_runs, ["task", "size"]).items()}
    for (task, size, accuracy), seed_runs in group(distilled_runs, ["task", "size", "teacher_saved_at_accuracy"]).items():
        settings[task, size, f"teacher at {accuracy:.0%}"] = seed_runs
    return labels, settings


def distillation_table(distilled_runs, baseline_runs):
    """Distilled students against the same sizes trained from scratch. The teacher is trained once and can teach
    many students, so distillation pays off once it has taught at least the break-even number of students:
    teacher training FLOPs / (FLOPs per success from scratch - FLOPs per success distilled, with forward passes),
    rounded up. It is unknown when either side had no success, since its cost is then only a lower bound."""
    print(f"\n{'task':7s} {'size':>5s} {'teacher':>16s} {'reached':>8s} {'student only':>13s} "
          f"{'+ teacher forward':>18s} {'+ teacher training':>19s} {'break-even students':>20s}   (PFLOPs per success)")
    labels, settings = distillation_settings(distilled_runs, baseline_runs)
    tasks = sorted({r["task"] for r in distilled_runs})
    sizes = [s for s in SIZE_ORDER if any(r["size"] == s for r in distilled_runs)]
    for task, size in itertools.product(tasks, sizes):
        scratch_runs = settings[task, size, "from scratch"]
        scratch_cost = expected_flops_per_success(scratch_runs)
        for label in labels:
            seed_runs = settings.get((task, size, label))
            if not seed_runs:
                continue
            n_reached = sum(r["reached"] for r in seed_runs)
            costs = distillation_costs(seed_runs)
            lower_bound = "" if n_reached else ">"  # no success: the compute spent is a lower bound
            break_even = ""
            if label != "from scratch":
                saving_per_student = scratch_cost - costs[1]
                if not n_reached or not any(r["reached"] for r in scratch_runs):
                    break_even = "unknown"
                elif saving_per_student <= 0:
                    break_even = "never"
                else:
                    break_even = str(math.ceil(seed_runs[0]["teacher_training_flops"] / saving_per_student))
            print(f"{task:7s} {size:>5s} {label:>16s} {n_reached:>4d}/{len(seed_runs):<3d} "
                  + " ".join(f"{lower_bound}{cost / 1e15:.2f}".rjust(width) for cost, width in zip(costs, (13, 18, 19)))
                  + f" {break_even:>20s}")


def plot_distillation(distilled_runs, baseline_runs, fig_dir, threshold):
    """One column per task, student size on the x-axis, one line per teacher checkpoint plus training from scratch.
    Rows: probability of reaching the threshold (95% intervals); expected FLOPs per success counting the student
    only; and counting both the teacher's forward passes and its training (the table also lists forward passes
    alone). Hollow markers are lower bounds."""
    tasks = sorted({r["task"] for r in distilled_runs})
    fig, axes = plt.subplots(3, len(tasks), figsize=(5.5 * len(tasks), 10.5), squeeze=False, sharex=True)
    labels, settings = distillation_settings(distilled_runs, baseline_runs)
    for column, task in enumerate(tasks):
        for i, label in enumerate(labels):
            points = sorted((seed_runs[0]["params"], seed_runs) for (point_task, _, point_label), seed_runs
                            in settings.items() if (point_task, point_label) == (task, label))
            if not points:
                continue
            # neutral gray for training from scratch; the blue ramp for teachers, darker = more accurate teacher
            color = "#6b6b6b" if label == "from scratch" else ramp(i, len(labels))
            style = dict(color=color, marker=MARKERS[i % 4], ms=7, lw=2)
            params = [p * 1.06 ** (i - (len(labels) - 1) / 2) for p, _ in points]  # sideways shift
            n_reached = [sum(r["reached"] for r in seed_runs) for _, seed_runs in points]
            probability = [n / len(seed_runs) for n, (_, seed_runs) in zip(n_reached, points)]
            intervals = [wilson_interval(n, len(seed_runs)) for n, (_, seed_runs) in zip(n_reached, points)]
            axes[0][column].errorbar(params, probability, yerr=[[p - lo for p, (lo, hi) in zip(probability, intervals)],
                                     [hi - p for p, (lo, hi) in zip(probability, intervals)]],
                                     capsize=3, elinewidth=1, ecolor=color + "99", label=label, **style)
            for row, cost_index in ((1, 0), (2, 2)):
                cost = [distillation_costs(seed_runs)[cost_index] for _, seed_runs in points]
                axes[row][column].plot(params, cost, **style)
                lower_bounds = [(x, y) for x, y, n in zip(params, cost, n_reached) if n == 0]
                if lower_bounds:
                    axes[row][column].scatter(*zip(*lower_bounds), s=60, facecolors="white", edgecolors=color,
                                              linewidths=2, zorder=3, marker=style["marker"])
        axes[0][column].set(xscale="log", title=task, ylim=(-0.03, 1.03))
        axes[1][column].set(xscale="log", yscale="log")
        axes[2][column].set(xscale="log", yscale="log", xlabel="student non-embedding parameters")
    axes[0][0].set_ylabel(f"probability of reaching {threshold:.0%}")
    axes[1][0].set_ylabel("FLOPs per success,\nstudent only")
    axes[2][0].set_ylabel("FLOPs per success, with teacher\nforward passes and training")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "distillation.png", dpi=150)
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


def depth_table(runs):
    """Lost in the middle? Final accuracy by relative position of the retrieved pair in the context (5 bins, from
    the start of the context to just before the queries), for both tasks, averaged over all sizes, separately for
    successful and failed runs."""
    print(f"\n{'task':7s} {'T':>5s} {'runs':>14s}   final accuracy by position of the pair, start to end")
    for (task, T, reached), rs in sorted(group(runs, ["task", "T", "reached"]).items()):
        mean_accuracy_by_depth = np.mean([r["log"][-1]["acc_by_depth"] for r in rs], 0)
        print(f"{task:7s} {T:5d} {len(rs):4d} {'successful' if reached else 'failed':>9s}   "
              + " ".join(f"{accuracy:.2f}" for accuracy in mean_accuracy_by_depth))


# a run "retrieves" if its best head ends with at least this much attention on the correct value
RETRIEVAL_ATTENTION = 0.1


def accuracy_jump_index(run):
    """Index of the first evaluation where held-out accuracy passes 50%, or None if it never does."""
    return next((i for i, record in enumerate(run["log"]) if record["acc"] >= 0.5), None)


def retrieving_runs(runs):
    """Successful runs whose best head ends with at least RETRIEVAL_ATTENTION on the correct value, and whose
    accuracy passes 50% after the first evaluation (so the time of the jump is known). In the other successful runs
    no single head attends to the value, so they retrieve some other way this probe does not see."""
    return [r for r in runs if r["reached"] and r["log"][-1]["attn_mass_max"] >= RETRIEVAL_ATTENTION
            and (accuracy_jump_index(r) or 0) > 0]


def retrieval_timing_table(runs):
    """Does the accuracy jump coincide with learning to retrieve? For each run in retrieving_runs, pooled over all
    sizes and context lengths, the lag in evaluations from the accuracy jump to the evaluation where the best
    head's attention on the correct value first reaches half its final level (positive = attention later)."""
    print(f"\n{'task':7s} {'successful':>10s} {'attention ends below 10%':>25s} {'median lag (evaluations)':>25s} "
          f"{'lag within 4 evaluations':>25s} {'max attention in runs that never pass 50%':>42s}")
    for task in sorted({r["task"] for r in runs}):
        successful = [r for r in runs if r["task"] == task and r["reached"]]
        retrieving = retrieving_runs(successful)
        n_low_attention = sum(r["log"][-1]["attn_mass_max"] < RETRIEVAL_ATTENTION for r in successful)
        lags = []
        for r in retrieving:
            final_attention = r["log"][-1]["attn_mass_max"]
            attention_half_index = next(i for i, record in enumerate(r["log"])
                                        if record["attn_mass_max"] >= final_attention / 2)
            lags.append(attention_half_index - accuracy_jump_index(r))
        runs_never_passing_half = [r for r in runs if r["task"] == task and accuracy_jump_index(r) is None]
        largest_attention = [max(record["attn_mass_max"] for record in r["log"]) for r in runs_never_passing_half]
        never_passing_half = (f"median {np.median(largest_attention):.3f}, largest {max(largest_attention):.3f}"
                              if largest_attention else "none")
        lag_summary = (f"{np.median(lags):25.1f} {f'{sum(abs(lag) <= 4 for lag in lags)} of {len(lags)}':>25s}"
                       if lags else f"{'none':>25s} {'none':>25s}")
        print(f"{task:7s} {len(successful):10d} {n_low_attention:25d} {lag_summary} {never_passing_half:>42s}")


def plot_retrieval_onset(runs, fig_dir, window=20):
    """Held-out accuracy and the best head's attention on the correct value, for the runs in retrieving_runs
    (pooled over sizes and context lengths), aligned at the evaluation where accuracy first passes 50%. Lines are
    medians over runs, bands the middle half. Runs stop soon after reaching the threshold, so the lines end where
    fewer than half the runs are left."""
    tasks = sorted({r["task"] for r in runs})
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4), squeeze=False, sharey=True)
    for ax, task in zip(axes[0], tasks):
        retrieving = retrieving_runs([r for r in runs if r["task"] == task])
        if not retrieving:
            continue
        accuracy_jump_indices = [accuracy_jump_index(r) for r in retrieving]
        # offsets from the jump at which at least half the runs still have an evaluation
        offsets = [offset for offset in range(-window, window + 1)
                   if sum(0 <= jump + offset < len(r["log"]) for r, jump in zip(retrieving, accuracy_jump_indices))
                   >= len(retrieving) / 2]
        for metric, label, color, line_style in (("acc", "held-out accuracy", RAMP[-1], "-"),
                                                 ("attn_mass_max", "best head's attention on the correct value",
                                                  RAMP[2], "--")):
            values_by_offset = [[r["log"][jump + offset][metric] for r, jump in zip(retrieving, accuracy_jump_indices)
                                 if 0 <= jump + offset < len(r["log"])] for offset in offsets]
            steps = [offset * retrieving[0]["eval_every"] for offset in offsets]
            ax.plot(steps, [np.median(values) for values in values_by_offset], color=color, ls=line_style, lw=2,
                    label=label)
            ax.fill_between(steps, [np.percentile(values, 25) for values in values_by_offset],
                            [np.percentile(values, 75) for values in values_by_offset], color=color, alpha=0.15, lw=0)
        ax.axvline(0, color="gray", lw=1)
        ax.set(title=f"{task}, {len(retrieving)} runs", xlabel="steps from accuracy first passing 50%", ylim=(0, 1))
    axes[0][0].legend(loc="center left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "retrieval_onset.png", dpi=150)
    plt.close(fig)


# context lengths where some size succeeds; longer ones are shown only as lower bounds
CONTEXT_LENGTHS_TO_FIT = [256, 512, 1024]
EQUAL_STEP_BUDGET = 20_000


def capped_at_steps(runs, max_steps):
    """Copies of the runs as if each had stopped at max_steps. The learning rate is constant after warmup,
    so a run's first max_steps do not depend on its own step cap. A success must be confirmed by its
    patience evaluations within max_steps."""
    capped = []
    for r in runs:
        confirmed_at_step = r["steps_to_threshold"] + (r["patience"] - 1) * r["eval_every"] if r["reached"] else None
        reached = r["reached"] and confirmed_at_step <= max_steps
        flops_spent = r["flops_to_threshold"] if reached else min(r["last_step"], max_steps) * r["flops_per_step"]
        capped.append({**r, "reached": reached, "flops_spent": flops_spent})
    return capped


def cheapest_cost_per_success(seed_runs_by_setting):
    """Lowest expected FLOPs per success over the settings that succeed at least once; infinite if none does."""
    costs = [expected_flops_per_success(seed_runs) for seed_runs in seed_runs_by_setting.values()
             if any(r["reached"] for r in seed_runs)]
    return min(costs, default=math.inf)


def fit_context_length_exponent(runs, task, n_resamples=2000):
    """Fits cost = a * T^exponent to the cheapest setting's expected FLOPs per success at each context length.
    A setting is a size and learning rate. The 95% intervals come from resampling seeds, with replacement,
    within each setting. A resample where some context length has no success has infinite cost there and
    cannot be fitted; it is left out of the exponent interval, so more than 2.5% of those leaves its
    upper end unbounded."""
    seed_runs_by_setting_at_context_length = {
        T: dict(group([r for r in runs if r["task"] == task and r["T"] == T], ["size", "lr"]))
        for T in CONTEXT_LENGTHS_TO_FIT}
    cheapest_costs = [cheapest_cost_per_success(seed_runs_by_setting_at_context_length[T])
                      for T in CONTEXT_LENGTHS_TO_FIT]
    if math.inf in cheapest_costs:
        return None
    log_context_lengths = np.log(CONTEXT_LENGTHS_TO_FIT)
    random_generator = np.random.default_rng(0)
    resampled_costs = []
    for _ in range(n_resamples):
        resampled_costs.append([])
        for T in CONTEXT_LENGTHS_TO_FIT:
            resampled_settings = {setting: list(random_generator.choice(seed_runs, len(seed_runs)))
                                  for setting, seed_runs in seed_runs_by_setting_at_context_length[T].items()}
            resampled_costs[-1].append(cheapest_cost_per_success(resampled_settings))
    resampled_costs = np.array(resampled_costs)
    fittable = np.isfinite(resampled_costs).all(1)
    resampled_exponents = [np.polyfit(log_context_lengths, np.log(costs), 1)[0] for costs in resampled_costs[fittable]]
    slope, intercept = np.polyfit(log_context_lengths, np.log(cheapest_costs), 1)
    return dict(cheapest_costs=cheapest_costs, exponent=slope, intercept=intercept,
                exponent_interval=np.percentile(resampled_exponents, [2.5, 97.5]),
                cost_intervals=np.percentile(resampled_costs, [2.5, 97.5], axis=0).T,  # may be infinite
                upper_end_unbounded=(~fittable).mean() > 0.025, n_unfittable=int((~fittable).sum()),
                n_resamples=n_resamples)


def context_length_fits(runs):
    """The fit for each task, under each context length's own step cap and under an equal step budget."""
    fits = {}
    for task in sorted({r["task"] for r in runs}):
        task_runs = [r for r in runs if r["task"] == task]
        fits[task, "own step caps"] = fit_context_length_exponent(task_runs, task)
        fits[task, f"equal {EQUAL_STEP_BUDGET:,}-step budget"] = fit_context_length_exponent(
            capped_at_steps(task_runs, EQUAL_STEP_BUDGET), task)
    return fits


def context_length_fit_table(runs, fits):
    """How fast the cheapest cost per success grows with context length, against how fast the cost of one
    training step grows."""
    print(f"\ncost per success of the cheapest size, fitted as T^exponent over T = {CONTEXT_LENGTHS_TO_FIT}")
    for (task, budget_label), fit in fits.items():
        if fit is None:
            print(f"  {task:7s} {budget_label:26s} cannot fit: some context length has no success")
            continue
        low, high = fit["exponent_interval"]
        high_text = "unbounded" if fit["upper_end_unbounded"] else f"{high:.2f}"
        costs = ", ".join(f"{cost / 1e15:.3g}" for cost in fit["cheapest_costs"])
        print(f"  {task:7s} {budget_label:26s} exponent {fit['exponent']:.2f} (95% interval {low:.2f} to {high_text}), "
              f"PFLOPs per success {costs}, {fit['n_unfittable']} of {fit['n_resamples']} resamples unfittable")
    for task in sorted({r["task"] for r in runs}):
        flops_per_step_exponents = []
        for size_runs in group([r for r in runs if r["task"] == task and r["T"] in CONTEXT_LENGTHS_TO_FIT], ["size"]).values():
            if len({r["T"] for r in size_runs}) >= 2:
                log_context_lengths = np.log([r["T"] for r in size_runs])
                log_flops_per_step = np.log([r["flops_per_step"] for r in size_runs])
                flops_per_step_exponents.append(np.polyfit(log_context_lengths, log_flops_per_step, 1)[0])
        if flops_per_step_exponents:
            print(f"  {task:7s} FLOPs per training step grow as T^{min(flops_per_step_exponents):.2f} "
                  f"to T^{max(flops_per_step_exponents):.2f} across sizes")


def plot_context_length_fit(runs, fits, fig_dir):
    """Cheapest expected FLOPs per success against context length, one line per task, with the fitted power law
    (each context length's own step cap). Bars are 95% intervals from resampling seeds. Hollow markers at
    longer context lengths, where no run of the task succeeded, are lower bounds."""
    task_fits = {task: fits.get((task, "own step caps")) for task in TASK_STYLES}
    task_fits = {task: fit for task, fit in task_fits.items() if fit is not None}
    if not task_fits:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (task, fit) in enumerate(task_fits.items()):
        style = TASK_STYLES[task]
        sideways_shift = 1.04 ** (i - (len(task_fits) - 1) / 2)  # keeps the two tasks' markers apart
        context_lengths = np.array(CONTEXT_LENGTHS_TO_FIT) * sideways_shift
        ax.vlines(context_lengths, *fit["cost_intervals"].T, color=style["color"], lw=1.5)
        ax.scatter(context_lengths, fit["cheapest_costs"], s=60, zorder=3, **style)
        low, high = fit["exponent_interval"]
        high_text = "unbounded" if fit["upper_end_unbounded"] else f"{high:.1f}"
        ax.plot(CONTEXT_LENGTHS_TO_FIT, np.exp(fit["intercept"]) * np.array(CONTEXT_LENGTHS_TO_FIT) ** fit["exponent"],
                color=style["color"], lw=2, label=f"{task}: T^{fit['exponent']:.1f} (95% interval {low:.1f} to {high_text})")
        # a context length where some run succeeds would need a real cost, not a lower bound, so it is skipped
        for (T,), longest_runs in group([r for r in runs if r["task"] == task and r["T"] > CONTEXT_LENGTHS_TO_FIT[-1]],
                                        ["T"]).items():
            if not any(r["reached"] for r in longest_runs):
                lower_bound = min(expected_flops_per_success(seed_runs)
                                  for seed_runs in group(longest_runs, ["size", "lr"]).values())
                ax.scatter([T * sideways_shift], [lower_bound], s=70, facecolors="white", edgecolors=style["color"],
                           linewidths=2, marker=style["marker"], zorder=3)
    ax.set(xscale="log", yscale="log", xlabel="context length T", ylabel="expected FLOPs per success, cheapest size")
    context_length_ticks = sorted({r["T"] for r in runs})
    ax.set_xticks(context_length_ticks, [str(T) for T in context_length_ticks])
    ax.minorticks_off()
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "context_length_fit.png", dpi=150)
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
    p.add_argument("--baseline", default=None, help="runs trained from scratch to compare distilled students against")
    a = p.parse_args()
    runs = load(a.root, a.threshold)
    print(f"{len(runs)} finished runs")
    if not any(r.get("teacher") for r in runs):  # distilled runs get their own table below
        table(runs)
    if a.best_lr:
        best_lr(runs, a.best_lr)
    fig_dir = Path(a.root) / ("figures" if a.threshold is None else f"figures_thr{a.threshold:g}")
    fig_dir.mkdir(exist_ok=True)
    threshold = a.threshold or runs[0]["threshold"]
    write_run_table(runs, fig_dir / "runs.csv")
    # architecture-sweep runs are labelled by their shape instead of a named size, and get their own figure
    architecture_runs = [r for r in runs if r["size"] not in SIZE_ORDER]
    distilled_runs = [r for r in runs if r.get("teacher")]
    named_size_runs = [r for r in runs if r["size"] in SIZE_ORDER and not r.get("teacher")]
    if architecture_runs:
        plot_architecture(architecture_runs, fig_dir, threshold)
    if distilled_runs:
        if a.baseline is None:
            p.error("a folder with distilled runs needs --baseline, the runs trained from scratch to compare against")
        # the same context lengths and student sizes, trained from scratch
        baseline_runs = [r for r in load(a.baseline, a.threshold) if (r["T"], r["size"]) in
                         {(d["T"], d["size"]) for d in distilled_runs}]
        distillation_table(distilled_runs, baseline_runs)
        plot_distillation(distilled_runs, baseline_runs, fig_dir, threshold)
    if named_size_runs:
        plot_cost(named_size_runs, "steps_to_threshold", fig_dir)
        plot_cost(named_size_runs, "flops_to_threshold", fig_dir)
        plot_success_and_cost(named_size_runs, fig_dir, threshold)
        plot_reached_vs_flops(named_size_runs, fig_dir, threshold)
        plot_curves(named_size_runs, fig_dir)
        plot_depth(named_size_runs, fig_dir)
        depth_table(named_size_runs)
        retrieval_timing_table(named_size_runs)
        plot_retrieval_onset(named_size_runs, fig_dir)
        fits = context_length_fits(named_size_runs)
        context_length_fit_table(named_size_runs, fits)
        plot_context_length_fit(named_size_runs, fits, fig_dir)
