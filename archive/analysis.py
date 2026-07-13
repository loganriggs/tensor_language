"""Figures for the cycle experiment: dataset, training curves, sweep summary,
attention patterns, and the circular structure of the residual stream.

Usage: python analysis.py   (after train.py; writes figures/*.png)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from data import EVAL_LENGTHS, N_CTX, N_VOCAB, sample_cycles
from model import CycleModel, config_kwargs

FIGURES = Path("figures")
RUNS = Path("runs")

# palette: ordinal blue ramp for in-distribution lengths, red steps for OOD
LENGTH_COLORS = {5: "#86b6ef", 10: "#5598e7", 15: "#2a78d6", 20: "#184f95", 25: "#e34948", 30: "#8c2b2b"}
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#3987e5", "#104281", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8c2b2b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SECONDARY, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150,
})


def load_runs(suffix: str = "") -> dict:
    paths = [p for p in RUNS.glob("*/history.json") if p.parent.name.endswith(suffix or ("h1", "h2"))]
    runs = {p.parent.name: json.loads(p.read_text()) for p in paths}
    order = sorted(runs, key=lambda k: (runs[k]["config"]["n_layer"], runs[k]["config"]["d_model"], runs[k]["config"]["n_head"]))
    return {k: runs[k] for k in order}


def load_model(name: str) -> CycleModel:
    config = json.loads((RUNS / name / "history.json").read_text())["config"]
    model = CycleModel(N_VOCAB, config["d_model"], config["n_head"], config["n_layer"], N_CTX, **config_kwargs(config))
    model.load_state_dict(torch.load(RUNS / name / "model.pt", map_location="cpu"))
    return model.eval()


def label(name: str) -> str:
    layer, d, head = name.split("_")[:3]
    return f"{layer[1]} layer · d={d[1:]} · {head[1]} head"


def fig_dataset():
    generator = torch.Generator().manual_seed(7)
    lengths = torch.tensor([5, 8, 12, 17, 20])
    tokens = sample_cycles(len(lengths), lengths, generator=generator)

    fig, axes = plt.subplots(len(lengths), 1, figsize=(9, 3.2), sharex=True)
    for ax, row, length in zip(axes, tokens, lengths):
        position = torch.arange(N_CTX) % length
        ax.imshow(position[None].float(), aspect="auto", cmap="twilight", vmin=0, vmax=length.item())
        ax.axvline(length.item() - 0.5, color="white", lw=1.5)
        ax.set_yticks([0], [f"L={length.item()}"], color=SECONDARY, fontsize=8)
        ax.grid(False)
        ax.tick_params(length=0)
    axes[-1].set_xlabel("position in context (white line = end of first cycle; loss starts there)")
    fig.suptitle("Training documents: a random cycle of L distinct tokens, tiled to 96 positions", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset.png", bbox_inches="tight")
    plt.close(fig)
    return tokens, lengths


def fig_training(runs: dict):
    fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharex=True, sharey=True)
    for ax, (name, history) in zip(axes.flat, runs.items()):
        for length in EVAL_LENGTHS:
            ax.plot(history["step"], history[f"acc@{length}"], color=LENGTH_COLORS[length], lw=1.6)
        ax.set_title(f"{label(name)}  ·  {history['config']['n_params']:,} params")
        ax.set_ylim(0, 1.02)
    handles = [plt.Line2D([], [], color=LENGTH_COLORS[k], lw=2) for k in EVAL_LENGTHS]
    names = [f"L={k}" + ("  (OOD)" if k > 20 else "") for k in EVAL_LENGTHS]
    fig.legend(handles, names, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.045))
    for ax in axes[-1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("eval accuracy")
    fig.suptitle("Eval accuracy over training (blues = trained lengths 5–20, reds = unseen lengths 25/30)", y=1.06, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "training.png", bbox_inches="tight")
    plt.close(fig)


def fig_sweep(runs: dict):
    names = list(runs)
    final = np.array([[runs[n][f"acc@{k}"][-1] for k in EVAL_LENGTHS] for n in names])
    params = np.array([runs[n]["config"]["n_params"] for n in names])

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.4), width_ratios=(1.15, 1))
    left.imshow(final, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i, j in np.ndindex(final.shape):
        color = "white" if final[i, j] > 0.6 else INK
        left.text(j, i, f"{final[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
    left.set_xticks(range(len(EVAL_LENGTHS)), [f"L={k}" for k in EVAL_LENGTHS], fontsize=8)
    left.set_yticks(range(len(names)), [f"{label(n)} ({p:,})" for n, p in zip(names, params)], fontsize=8)
    left.grid(False)
    left.set_title("Final accuracy by config × eval cycle length")

    in_dist, ood = final[:, :4].mean(1), final[:, 4:].mean(1)
    layers = np.array([runs[n]["config"]["n_layer"] for n in names])
    for n_layer, color in ((1, "#eda100"), (2, "#2a78d6")):
        pick = layers == n_layer
        right.scatter(params[pick], in_dist[pick], s=42, color=color, label=f"{n_layer} layer · in-dist (5–20)")
        right.scatter(params[pick], ood[pick], s=42, color=color, marker="^", facecolors="none", label=f"{n_layer} layer · OOD (25/30)")
    right.set_xscale("log")
    right.set_xticks([2e3, 5e3, 1e4, 2e4, 5e4], ["2k", "5k", "10k", "20k", "50k"])
    right.minorticks_off()
    right.set_xlabel("parameters")
    right.set_ylabel("mean final accuracy")
    right.set_ylim(-0.02, 1.02)
    right.legend(fontsize=8, loc="upper left")
    right.set_title("Accuracy vs model size")
    fig.tight_layout()
    fig.savefig(FIGURES / "sweep.png", bbox_inches="tight")
    plt.close(fig)


def fig_long(runs: dict):
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.2), sharey=True)
    for ax, (name, history) in zip(axes, runs.items()):
        for length in EVAL_LENGTHS:
            ax.plot(history["step"], history[f"acc@{length}"], color=LENGTH_COLORS[length], lw=1.6)
        ax.set_title(f"{label(name)}  ·  {history['config']['n_params']:,} params")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("step")
    axes[0].set_ylabel("eval accuracy")
    handles = [plt.Line2D([], [], color=LENGTH_COLORS[k], lw=2) for k in EVAL_LENGTHS]
    names = [f"L={k}" + ("  (OOD)" if k > 20 else "") for k in EVAL_LENGTHS]
    fig.legend(handles, names, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Same configs trained 5× longer (20k steps)", y=1.22, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_long.png", bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def fig_attention(name: str, length: int = 7):
    model = load_model(name)
    tokens = sample_cycles(1, torch.tensor([length]), generator=torch.Generator().manual_seed(3))
    x = model.embed(tokens)

    fig, axes = plt.subplots(len(model.layers), model.layers[0].n_head, figsize=(4.2 * model.layers[0].n_head, 3.8 * len(model.layers)), squeeze=False)
    for i, layer in enumerate(model.layers):
        pattern = layer.pattern(x)[0]
        for j in range(pattern.size(0)):
            ax = axes[i][j]
            v = pattern[j].abs().max().item()
            ax.imshow(pattern[j], cmap=DIVERGING, vmin=-v, vmax=v)
            ax.set_title(f"layer {i + 1} · head {j + 1}")
            ax.grid(False)
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")
        x = layer(x)
    fig.suptitle(f"{label(name)} — attention scores on one L={length} document (blue − / red +, no softmax)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "attention.png", bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def fig_circle(name: str, lengths: tuple[int, ...] = (5, 7, 10), out: str = "circle.png"):
    """Josh Engels-style check: token identity is averaged out by taking the mean
    residual state at each absolute position over many documents of the same L;
    what remains is the positional code. PCA of it, colored by phase t mod L."""
    model = load_model(name)
    fig, axes = plt.subplots(len(model.layers), len(lengths), figsize=(3.6 * len(lengths), 3.4 * len(model.layers)), squeeze=False)

    for j, length in enumerate(lengths):
        tokens = sample_cycles(512, torch.full((512,), length), generator=torch.Generator().manual_seed(11))
        streams = model.residuals(tokens)
        keep = slice(2 * length, N_CTX - length)  # steady state, away from both edges
        phase = (torch.arange(N_CTX) % length)[keep]

        for i, stream in enumerate(streams):
            mean = stream.mean(0)
            kernel = torch.ones(1, 1, length) / length  # remove slow positional drift
            trend = torch.conv1d(mean.T[:, None], kernel, padding="same")[:, 0].T
            m = (mean - trend)[keep]
            m = m - m.mean(0)
            proj = m @ torch.linalg.svd(m, full_matrices=False)[2][:2].T
            centers = torch.stack([proj[phase == p].mean(0) for p in range(length)])

            ax = axes[i][j]
            ring = torch.cat([centers, centers[:1]])
            ax.plot(ring[:, 0], ring[:, 1], color="#c3c2b7", lw=1, zorder=1)
            ax.scatter(proj[:, 0], proj[:, 1], c=phase, cmap="twilight", s=14, alpha=0.8, vmin=0, vmax=length, zorder=2)
            for p, center in enumerate(centers):
                ax.text(center[0], center[1], str(p), fontsize=9, weight="bold", color=INK,
                        ha="center", va="center", zorder=3,
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#c3c2b7", alpha=0.9))
            ax.set_title(f"layer {i + 1} residual · L={length}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.suptitle(f"{label(name)} — PCA of token-averaged residual stream; dots are context positions,\n"
                 "colored/labeled by phase (t mod L), gray line joins consecutive phases", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    runs = load_runs()
    fig_dataset()
    fig_training(runs)
    fig_sweep(runs)

    long_runs = load_runs("_long")
    if long_runs:
        fig_long(long_runs)
        runs = {**runs, **long_runs}

    in_dist = {n: np.mean([runs[n][f"acc@{k}"][-1] for k in EVAL_LENGTHS if k <= 20]) for n in runs}
    ood = {n: np.mean([runs[n][f"acc@{k}"][-1] for k in EVAL_LENGTHS if k > 20]) for n in runs}
    solved = [n for n, a in in_dist.items() if a > 0.95] or [max(in_dist, key=in_dist.get)]
    chosen = max(solved, key=ood.get)
    print("chosen model for analysis:", chosen, f"(in-dist {in_dist[chosen]:.3f}, ood {ood[chosen]:.3f})")
    fig_attention(chosen)
    fig_circle(chosen)
    return chosen


if __name__ == "__main__":
    main()
