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
