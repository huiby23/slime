"""Render the README overview from the adjacent audited results.json.

Requires Matplotlib. No model loading, GPU access, or external data is used.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext

ROOT = Path(__file__).resolve().parent
ARMS = (
    ("bf16_lr1e6", "BF16 LR 1e-6", "#2868a9"),
    ("bf16_lr1e5", "BF16 LR 1e-5", "#dc8324"),
    ("bf16_lr1e3", "BF16 LR 1e-3", "#b84440"),
    ("fixed_noise_s005", "Noise 0.05 (clean BF16 eval)", "#8063a8"),
    ("nvfp4_rollout_chain", "NVFP4 rollout (quantized eval)", "#368671"),
)


def rolling_mean(rows, size=25):
    """Do not mix windows across the NVFP4 sampler migration boundary."""
    return [
        sum(row["gap"] for row in rows[max(0, i - size + 1) : i + 1]) / min(i + 1, size)
        for i in range(len(rows))
    ]


def plot_evaluation(ax, rows, key, color, migration=False):
    if migration:
        before = [row for row in rows if row["completed_updates"] <= 100]
        after = [row for row in rows if row["completed_updates"] >= 125]
        for stage in (before, after):
            ax.plot(
                [row["completed_updates"] for row in stage],
                [100 * row[key] for row in stage],
                color=color,
                linewidth=1.9,
                marker="o",
                markersize=3.3,
            )
        left, right = before[-1], after[0]
        ax.plot(
            [left["completed_updates"], right["completed_updates"]],
            [100 * left[key], 100 * right[key]],
            color=color,
            linewidth=1.6,
            linestyle="--",
        )
    else:
        ax.plot(
            [row["completed_updates"] for row in rows],
            [100 * row[key] for row in rows],
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=3.3,
        )


def main():
    report = json.loads((ROOT / "results.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1), gridspec_kw={"wspace": 0.26})
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.26, top=0.75)
    for key, label, color in ARMS:
        run = report["runs"][key]
        is_nv = key == "nvfp4_rollout_chain"
        plot_evaluation(axes[0], run["evaluations"], "accuracy", color, is_nv)
        plot_evaluation(axes[1], run["evaluations"], "truncation_ratio", color, is_nv)
        series = run["training_gap"]
        stages = (
            (
                [row for row in series if row["completed_updates"] <= 111],
                [row for row in series if 112 <= row["completed_updates"] <= 113],
                [row for row in series if row["completed_updates"] >= 114],
            )
            if is_nv
            else (series,)
        )
        for stage in stages:
            axes[2].plot(
                [row["completed_updates"] for row in stage],
                rolling_mean(stage),
                color=color,
                linewidth=1.9,
            )
    for ax in axes:
        ax.set_xlim(-5, 510)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.set_xlabel("Completed optimizer updates")
        ax.grid(axis="y", color="#dedede", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(111, color="#368671", linewidth=1, linestyle=":", alpha=0.6)
    axes[0].set_title("A  Holdout accuracy")
    axes[0].set_ylabel("Correct answers (%)")
    axes[0].set_ylim(-4, 100)
    axes[1].set_title("B  Holdout truncation")
    axes[1].set_ylabel("Answers reaching token limit (%)")
    axes[1].set_ylim(-4, 104)
    axes[2].set_title("C  Train / rollout log-prob gap")
    axes[2].set_ylabel("25-update rolling mean (nats, log scale)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.003, 50)
    axes[2].yaxis.set_major_formatter(LogFormatterMathtext())
    fig.suptitle(
        "Qwen3-0.6B train / inference mismatch experiments — 2026-11-08",
        y=0.98,
        fontsize=16,
        fontweight="normal",
    )
    handles = [
        Line2D([0], [0], color=color, linewidth=2.4, label=label)
        for _, label, color in ARMS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        columnspacing=2,
        handlelength=2.3,
    )
    fig.text(
        0.055,
        0.17,
        "A/B: 512 holdout questions, one answer per question; evaluation precision differs by arm.",
        fontsize=10.5,
    )
    fig.text(
        0.055,
        0.12,
        "NVFP4: recorded through update 422, latest full evaluation at 400; dotted line marks sampler migration at 111.",
        fontsize=10.5,
    )
    fig.text(
        0.055,
        0.07,
        "LR 1e-3 was stopped at 300. Curves are descriptive, single-seed results; they do not establish mismatch causality.",
        fontsize=10.5,
    )
    for extension in ("png", "svg"):
        fig.savefig(ROOT / ("results-overview." + extension), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
