"""Reproduce the inline summary figures for explanation_2026-09-01_1957.

The values are the frozen rung-426 and rung-430 SELECT/document metrics already
reported in sections 16 and 17.  CE deltas are converted to relative perplexity
increase as 100 * (exp(delta_CE) - 1) solely for the percentage display.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).parent / "explanations" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#3B82F6",
    "green": "#10B981",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "gray": "#64748B",
}


def style_axis(ax, *, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def label_bars(ax, bars, fmt="{:.1f}%", *, fontsize=8):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def section16():
    arms = ["Global 54\n(15.58 MB)", "Global 72\n(19.20 MB)", "Independent 72\n(19.20 MB)"]
    metrics = ["Q/K factor", "Score product", "Full write"]
    errors = np.array(
        [
            [46.15, 64.37, 50.06],
            [42.83, 48.37, 41.36],
            [61.85, 147.40, 60.21],
        ]
    )
    ce = np.array([0.02149, 0.01756, 0.02073])
    ppl = 100 * np.expm1(ce)
    storage = np.array([81.1554, 100.0, 100.0])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    fig.suptitle("Section 16 — Global sparse token sharing: better errors at lower or equal price", fontsize=14, weight="bold")

    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.24
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    for i, (arm, color) in enumerate(zip(arms, colors)):
        bars = ax.bar(x + (i - 1) * width, errors[i], width, label=arm, color=color)
        label_bars(ax, bars)
    ax.axhline(100, color=COLORS["gray"], linestyle="--", linewidth=1.2, label="zero-prediction error")
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Relative squared error (%) — lower is better")
    ax.set_ylim(0, 165)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax)

    ax = axes[1]
    bars = ax.bar(np.arange(3), storage, color=colors)
    label_bars(ax, bars)
    ax.set_xticks(np.arange(3), ["G54", "G72", "I72"])
    ax.set_ylabel("Storage relative to I72 (%)")
    ax.set_ylim(0, 115)
    ax.text(0, 86, "18.84%\nsmaller", ha="center", va="bottom", fontsize=9, weight="bold")
    style_axis(ax)

    ax = axes[2]
    bars = ax.bar(np.arange(3), ppl, color=colors)
    label_bars(ax, bars, "{:.2f}%")
    ax.set_xticks(np.arange(3), ["G54", "G72", "I72"])
    ax.set_ylabel("Relative perplexity increase (%)")
    ax.set_ylim(0, max(ppl) * 1.22)
    style_axis(ax)

    fig.text(
        0.5,
        0.012,
        "G = one global cross-head code; I = independent per-head/branch codes. "
        "100% relative squared error equals predicting zero. CE is displayed as 100×(exp(ΔCE)−1).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.savefig(OUT / "explanation_1957_section16_sparse_global.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def section17():
    metrics = ["Branch score", "Score product", "Full write"]
    candidate_names = ["SQ54\nfactor-only", "SC54\nscore-coupled", "CP54\nproduct-coupled"]
    candidate_errors = np.array(
        [
            [58.27, 96.26, 65.89],
            [39.67, 21.97, 31.08],
            [40.01, 21.90, 33.91],
        ]
    )
    control_names = ["CP54", "Pair-label\npermuted", "Wrong branch/\nhead"]
    control_errors = np.array(
        [
            [40.01, 21.90, 33.91],
            [77.13, 44.26, 48.05],
            [116.48, 200.48, 191.57],
        ]
    )
    ce_names = ["SQ54", "SC54", "CP54", "Pair perm.", "Wrong"]
    ce = np.array([0.03083, 0.02602, 0.02306, 0.10957, 1.02557])
    ppl = 100 * np.expm1(ce)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.1), gridspec_kw={"width_ratios": [1.8, 1.8, 1.25]})
    fig.suptitle("Section 17 — Learning the score relation matters; product loss adds little", fontsize=14, weight="bold")

    x = np.arange(len(metrics))
    width = 0.24
    candidate_colors = [COLORS["gray"], COLORS["blue"], COLORS["green"]]
    ax = axes[0]
    for i, (name, color) in enumerate(zip(candidate_names, candidate_colors)):
        bars = ax.bar(x + (i - 1) * width, candidate_errors[i], width, label=name, color=color)
        label_bars(ax, bars)
    ax.axhline(100, color=COLORS["gray"], linestyle="--", linewidth=1.1)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Relative squared error (%)")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    control_colors = [COLORS["green"], COLORS["orange"], COLORS["red"]]
    ax = axes[1]
    for i, (name, color) in enumerate(zip(control_names, control_colors)):
        bars = ax.bar(x + (i - 1) * width, control_errors[i], width, label=name, color=color)
        label_bars(ax, bars)
    ax.axhline(100, color=COLORS["gray"], linestyle="--", linewidth=1.1)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Relative squared error (%)")
    ax.set_ylim(0, 225)
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[2]
    bars = ax.bar(np.arange(5), ppl, color=[*candidate_colors, COLORS["orange"], COLORS["red"]])
    label_bars(ax, bars, "{:.1f}%")
    ax.set_xticks(np.arange(5), ce_names, rotation=24, ha="right")
    ax.set_ylabel("Relative perplexity increase (%)")
    ax.set_ylim(0, max(ppl) * 1.18)
    style_axis(ax)

    fig.text(
        0.5,
        0.008,
        "Left: candidate objectives. Middle: the best 15.58 MB candidate against controls. "
        "100% error equals predicting zero; values above 100% are worse than zero. CE is displayed as 100×(exp(ΔCE)−1).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "explanation_1957_section17_score_coupling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    section16()
    section17()
