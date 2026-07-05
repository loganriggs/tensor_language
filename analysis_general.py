"""Analysis of the multi-family graph-tracing models: performance per family,
representation organization (Park protocol), and cycle phase-circles.

Usage: python analysis_general.py   (after train_general.py; writes figures/gen_*.png)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis import BLUES, INK, SECONDARY
from data import sample_cycles
from data import N_CTX as CYC_CTX
from graphs import N_CTX, N_VOCAB
from icl_reps import CONTEXTS, adjacency, gram_adjacency_corr, mean_reps, pca_coords, SHAPE, N_NODES
from analysis_geo import draw_edges
from model import CycleModel, config_kwargs

FIGURES = Path("figures")
RUNS = Path("runs_gen")
ARCH_COLORS = {"bilin-lerp-2L": "#2a78d6", "bilin-add-2L": "#86b6ef", "bilin-add-3L": "#104281",
               "softmax-lerp-2L": "#e34948", "softmax-add-3L": "#8c2b2b"}


def load_runs() -> dict:
    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(RUNS.glob("*/history.json"))}
    return {k: runs[k] for k in ARCH_COLORS if k in runs}


def load_model(name: str) -> CycleModel:
    config = json.loads((RUNS / name / "history.json").read_text())["config"]
    model = CycleModel(N_VOCAB, config["d_model"], 1, config["n_layer"], N_CTX, **config_kwargs(config))
    model.load_state_dict(torch.load(RUNS / name / "model.pt", map_location="cpu"))
    return model.eval()


def fig_perf(runs: dict):
    names = list(runs)
    sets = [k.split("|")[1] for k in runs[names[0]] if k.startswith("legal")]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2))
    for ax, stat in zip(axes, ("legal", "mass")):
        final = np.array([[runs[n][f"{stat}|{s}"][-1] for s in sets] for n in names])
        ax.imshow(final, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for i, j in np.ndindex(final.shape):
            ax.text(j, i, f"{final[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if final[i, j] > 0.6 else INK)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_xticks(range(len(sets)), sets, fontsize=7.5, rotation=20, ha="right")
        ax.axvline(5.5, color=SECONDARY, lw=1.0)
        ax.grid(False)
        ax.set_title({"legal": "legal-move rate", "mass": "neighbor mass"}[stat] + "  (right of line = held-out families/sizes)", fontsize=10)
    fig.suptitle("One model, all graph families: final tail metrics per architecture × eval set", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "gen_perf.png", bbox_inches="tight")
    plt.close(fig)


def fig_curves(runs: dict):
    picks = ("grid", "dring", "torus (unseen family)", "ER graph (unseen family)")
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.2), sharey=True)
    for ax, s in zip(axes, picks):
        for name, history in runs.items():
            ax.plot(history["step"], history[f"mass|{s}"], color=ARCH_COLORS[name], lw=1.5)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("neighbor mass")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in ARCH_COLORS.values()]
    fig.legend(handles, list(ARCH_COLORS), ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=9)
    fig.suptitle("Neighbor mass over training (torus & ER never trained)", y=1.24, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "gen_training.png", bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def fig_org(models: dict):
    """H1: Park-protocol organization on grid docs, all archs + the single-task reference."""
    fig = plt.figure(figsize=(14.5, 6.6))
    gs = fig.add_gridspec(2, 6, width_ratios=(1.7, 1, 1, 1, 1, 1))

    axc = fig.add_subplot(gs[:, 0])
    results = {}
    for col, (name, model) in enumerate(models.items()):
        corr, maps = {}, {}
        for topology in ("grid", "torus"):
            reps = mean_reps(model, topology)
            A = adjacency(topology)
            corr[topology] = {t: gram_adjacency_corr(H, A) for t, H in reps.items()}
            maps[topology] = pca_coords(reps[max(CONTEXTS)], k=2)
        results[name] = corr
        axc.plot(list(corr["grid"]), list(corr["grid"].values()), color=ARCH_COLORS[name], lw=1.8, marker="o", ms=4, label=name)

        for row, topology in enumerate(("grid", "torus")):
            ax = fig.add_subplot(gs[row, col + 1])
            xy = maps[topology]
            draw_edges(ax, xy.numpy(), SHAPE, topology, color="#c3c2b7", lw=0.8)
            ax.scatter(xy[:, 0], xy[:, 1], c=torch.arange(N_NODES) // SHAPE[1], cmap="twilight", vmin=0, vmax=SHAPE[0],
                       s=90, zorder=3, edgecolors="white")
            ax.set_title(f"{name}\n{topology} · PC1–2 · corr {corr[topology][max(CONTEXTS)]:+.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    axc.axhline(0, color="#c3c2b7", lw=0.8)
    axc.axhline(-0.57, color="#8c2b2b", lw=1.0, ls=(0, (3, 2)))
    axc.text(8.5, -0.55, "single-task grid model (−0.57)", fontsize=7.5, color="#8c2b2b")
    axc.set_xscale("log")
    axc.set_xlabel("context length")
    axc.set_ylabel("corr(Gram of mean reps, adjacency) — grid docs")
    axc.legend(fontsize=8, loc="upper left")
    axc.set_title("H1: does multi-family training\nflip organization positive?", fontsize=9.5)
    fig.suptitle("In-context representation organization of the multi-family models (Park et al. protocol, 4×5 lattices)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "gen_org.png", bbox_inches="tight")
    plt.close(fig)
    return results


@torch.inference_mode()
def fig_circle(models: dict, length: int = 7):
    """Phase-circle check on deterministic cycle docs (≡ directed-ring family)."""
    fig, axes = plt.subplots(1, len(models), figsize=(2.9 * len(models), 3.1))
    tokens = sample_cycles(512, torch.full((512,), length), generator=torch.Generator().manual_seed(11))
    keep = slice(2 * length, CYC_CTX - length)
    phase = (torch.arange(CYC_CTX) % length)[keep]
    for ax, (name, model) in zip(axes, models.items()):
        mean = model.residuals(tokens)[-1].mean(0)
        kernel = torch.ones(1, 1, length) / length
        trend = torch.conv1d(mean.T[:, None], kernel, padding="same")[:, 0].T
        m = (mean - trend)[keep]
        m = m - m.mean(0)
        proj = m @ torch.linalg.svd(m, full_matrices=False)[2][:2].T
        centers = torch.stack([proj[phase == p].mean(0) for p in range(length)])
        ring = torch.cat([centers, centers[:1]])
        ax.plot(ring[:, 0], ring[:, 1], color="#c3c2b7", lw=1, zorder=1)
        ax.scatter(proj[:, 0], proj[:, 1], c=phase, cmap="twilight", s=12, alpha=0.8, vmin=0, vmax=length, zorder=2)
        for p, center in enumerate(centers):
            ax.text(center[0], center[1], str(p), fontsize=8, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#c3c2b7", alpha=0.9), zorder=3)
        ax.set_title(name, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"Cycle phase structure survives multi-family training? (token-averaged detrended residual, L={length})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "gen_circle.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    runs = load_runs()
    fig_perf(runs)
    fig_curves(runs)
    models = {name: load_model(name) for name in runs}
    results = fig_org(models)
    fig_circle(models)
    for name, corr in results.items():
        print(name, "grid corr:", {t: round(c, 3) for t, c in corr["grid"].items()})


if __name__ == "__main__":
    main()
