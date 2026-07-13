"""Figures for the lattice random-walk experiment.

Usage: python analysis_geo.py   (after train_geo.py; writes figures/geo_*.png)
"""

import json
import re
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis import BLUES, GRID, INK, MUTED, SECONDARY
from geodata import N_CTX, N_VOCAB, TAIL, TOPOLOGIES, TRAIN_SHAPES, eval_sets, neighbor_table, walk_batch
from model import CycleModel, config_kwargs

FIGURES = Path("figures")
RUNS = Path("runs_geo")
EVALS = ("in", "ood 5x5", "ood 6x6")
D_COLORS = {32: "#86b6ef", 64: "#2a78d6", 128: "#104281"}
EVAL_COLORS = {"in": "#2a78d6", "ood 5x5": "#e34948", "ood 6x6": "#8c2b2b"}


def load_runs(long: bool = False) -> dict:
    pattern = re.compile(r"^(grid|cylinder|torus)_L\d_d\d+" + ("_long$" if long else "$"))
    runs = {p.parent.name: json.loads(p.read_text()) for p in RUNS.glob("*/history.json")
            if pattern.match(p.parent.name)}
    order = sorted(runs, key=lambda k: (TOPOLOGIES.index(runs[k]["config"]["topology"]),
                                        runs[k]["config"]["n_layer"], runs[k]["config"]["d_model"]))
    return {k: runs[k] for k in order}


def load_model(name: str) -> CycleModel:
    config = json.loads((RUNS / name / "history.json").read_text())["config"]
    model = CycleModel(N_VOCAB, config["d_model"], 1, config["n_layer"], N_CTX, **config_kwargs(config))
    model.load_state_dict(torch.load(RUNS / name / "model.pt", map_location="cpu"))
    return model.eval()


def lattice_xy(shape):
    m, n = shape
    return np.stack([np.arange(m * n) % n, np.arange(m * n) // n], 1).astype(float)


def draw_edges(ax, points, shape, topology, color=GRID, lw=1.0, arcs=False):
    """Draw lattice edges between node coordinates; wrap edges dashed (arced outward
    when `arcs`, for true-lattice layouts where they would overlap interior edges)."""
    m, n = shape
    neighbors, _ = neighbor_table(shape, topology)
    center = points.mean(0)
    for a in range(m * n):
        for b in neighbors[a].tolist():
            if b < a:
                continue
            wrap = abs(a // n - b // n) > 1 or abs(a % n - b % n) > 1
            if wrap and arcs:
                mid = (points[a] + points[b]) / 2
                out = mid - center
                ctrl = mid + out / max(np.linalg.norm(out), 1e-6) * 1.1
                t = np.linspace(0, 1, 24)[:, None]
                curve = (1 - t) ** 2 * points[a] + 2 * t * (1 - t) * ctrl + t**2 * points[b]
                ax.plot(curve[:, 0], curve[:, 1], color=color, lw=lw, ls=(0, (2, 2)), zorder=1)
            else:
                ax.plot(*np.stack([points[a], points[b]], 1), color=color, lw=lw,
                        ls=(0, (2, 2)) if wrap else "-", zorder=1)


def fig_dataset():
    generator = torch.Generator().manual_seed(5)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, topology in zip(axes, TOPOLOGIES):
        shape = (4, 5)
        tokens, nodes, perm = walk_batch(1, shape, topology, generator)
        xy = lattice_xy(shape)
        draw_edges(ax, xy, shape, topology, arcs=True)
        path = xy[nodes[0, :36]] + np.random.default_rng(0).normal(0, 0.06, (36, 2))
        for a, b, t in zip(path[:-1], path[1:], np.linspace(0.15, 0.9, 35)):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#2a78d6", alpha=t, lw=1.6, zorder=2)
        ax.scatter(xy[:, 0], xy[:, 1], s=420, c="white", edgecolors="#c3c2b7", zorder=3)
        for i, token in enumerate(perm[0].tolist()):
            ax.text(xy[i, 0], xy[i, 1], str(token), ha="center", va="center", fontsize=8, zorder=4)
        ax.set_title(f"{topology} 4×5 — first 36 steps of one walk")
        ax.set_xlim(-0.7, shape[1] - 0.3)
        ax.set_ylim(-0.7, shape[0] - 0.3)
        ax.invert_yaxis()
        ax.axis("off")
    fig.suptitle("One document = one token-labeled lattice + a uniform random walk (dashed = wraparound edges; walk darkens over time)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_dataset.png", bbox_inches="tight")
    plt.close(fig)


def fig_training(runs: dict):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True, sharey=True)
    for (i, topology), (j, n_layer) in product(enumerate(TOPOLOGIES), enumerate((1, 2, 3))):
        ax = axes[i][j]
        for d_model, color in D_COLORS.items():
            history = runs[f"{topology}_L{n_layer}_d{d_model}"]
            ax.plot(history["step"], history["legal|in"], color=color, lw=1.6)
            ax.plot(history["step"], history["legal|ood 6x6"], color=color, lw=1.4, ls=(0, (2, 2)))
        ax.set_title(f"{topology} · {n_layer} layer")
        ax.set_ylim(0, 1.02)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in D_COLORS.values()]
    handles += [plt.Line2D([], [], color=SECONDARY, lw=1.6), plt.Line2D([], [], color=SECONDARY, lw=1.4, ls=(0, (2, 2)))]
    names = [f"d={d}" for d in D_COLORS] + ["in-dist shapes", "OOD 6×6"]
    fig.legend(handles, names, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.035))
    for ax in axes[-1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("legal-move rate")
    fig.suptitle("Legal-move rate over training (solid = trained sizes, dashed = unseen 6×6)", y=1.05, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_training.png", bbox_inches="tight")
    plt.close(fig)


def fig_sweep(runs: dict):
    names = list(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, stat in zip(axes, ("legal", "mass")):
        final = np.array([[runs[n][f"{stat}|{e}"][-1] for e in EVALS] for n in names])
        ax.imshow(final, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for i, j in np.ndindex(final.shape):
            ax.text(j, i, f"{final[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if final[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(EVALS)), EVALS, fontsize=8)
        labels = ["{n_layer}L · d{d_model} ({n_params:,})".format(**runs[n]["config"]) for n in names]
        ax.set_yticks(range(len(names)), labels if ax is axes[0] else [], fontsize=8)
        ax.axhline(8.5, color=SECONDARY, lw=0.8)
        ax.axhline(17.5, color=SECONDARY, lw=0.8)
        ax.grid(False)
        ax.set_title({"legal": "legal-move rate (argmax is a neighbor)", "mass": "probability mass on neighbors"}[stat])
    for y, topology in zip((4, 13, 22), TOPOLOGIES):
        axes[0].text(-0.62, y, topology, transform=axes[0].get_yaxis_transform(),
                     rotation=90, va="center", ha="center", fontsize=10, color=INK)
    fig.suptitle("Final tail metrics by config × eval set (rows grouped by topology)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_sweep.png", bbox_inches="tight")
    plt.close(fig)


def fig_long(runs: dict):
    fig, axes = plt.subplots(1, len(runs), figsize=(3.8 * len(runs), 3.4), sharey=True)
    for ax, (name, history) in zip(axes, runs.items()):
        for e in EVALS:
            ax.plot(history["step"], history[f"legal|{e}"], color=EVAL_COLORS[e], lw=1.6)
            ax.plot(history["step"], history[f"mass|{e}"], color=EVAL_COLORS[e], lw=1.2, ls=(0, (2, 2)))
        ax.set_title(f"{history['config']['topology']} · 2L · d128")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("step")
    axes[0].set_ylabel("legal rate / mass")
    handles = [plt.Line2D([], [], color=EVAL_COLORS[e], lw=2) for e in EVALS]
    handles += [plt.Line2D([], [], color=SECONDARY, lw=1.6), plt.Line2D([], [], color=SECONDARY, lw=1.2, ls=(0, (2, 2)))]
    fig.legend(handles, list(EVALS) + ["legal-move rate", "neighbor mass"], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.suptitle("Best config trained 2.5× longer (20k steps)", y=1.26, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_training_long.png", bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def fig_incontext(best: dict):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, topology in zip(axes, TOPOLOGIES):
        model = load_model(best[topology]).cuda()
        for e, (tokens, legal) in eval_sets(topology, n_seq=96, seed=99).items():
            logits = model(tokens.cuda())[:, :-1].cpu()
            legal = legal[:, :-1]
            rate = legal.gather(2, logits.argmax(-1, keepdim=True))[..., 0].float().mean(0)
            mass = (logits.softmax(-1) * legal).sum(-1).mean(0)
            ax.plot(rate, color=EVAL_COLORS[e], lw=1.5, label=e)
            ax.plot(mass, color=EVAL_COLORS[e], lw=1.2, ls=(0, (2, 2)))
        model.cpu()
        ax.set_title(f"{topology}  ({best[topology].split('_', 1)[1]})")
        ax.set_xlabel("position in context")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("legal rate / mass")
    handles = [plt.Line2D([], [], color=EVAL_COLORS[e], lw=2) for e in EVALS]
    handles += [plt.Line2D([], [], color=SECONDARY, lw=1.5), plt.Line2D([], [], color=SECONDARY, lw=1.2, ls=(0, (2, 2)))]
    fig.legend(handles, list(EVALS) + ["legal-move rate", "neighbor mass"], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.suptitle("In-context graph learning: metrics vs position as the walk reveals the lattice", y=1.22, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_incontext.png", bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def fig_transfer(best: dict):
    rate = np.zeros((3, 3))
    for (i, trained), (j, evaluated) in product(enumerate(TOPOLOGIES), repeat=2):
        model = load_model(best[trained]).cuda()
        tokens, legal = eval_sets(evaluated, n_seq=96, seed=99)["in"]
        logits = model(tokens.cuda())[:, TAIL:-1].cpu()
        rate[i, j] = legal[:, TAIL:-1].gather(2, logits.argmax(-1, keepdim=True)).float().mean()
        model.cpu()

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(rate, cmap=BLUES, vmin=0, vmax=1)
    for i, j in np.ndindex(rate.shape):
        ax.text(j, i, f"{rate[i, j]:.2f}", ha="center", va="center", color="white" if rate[i, j] > 0.6 else INK)
    ax.set_xticks(range(3), TOPOLOGIES)
    ax.set_yticks(range(3), TOPOLOGIES)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("trained on")
    ax.grid(False)
    ax.set_title("Cross-topology transfer\n(legal-move rate, trained sizes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_transfer.png", bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def implied_transitions(model, topology: str, shape=(4, 5), n_walks: int = 512):
    """The model's learned graph: mean predicted next-token distribution per node,
    for one fixed token labeling, restricted to the tokens in play."""
    generator = torch.Generator().manual_seed(21)
    _, nodes, perm = walk_batch(n_walks, shape, topology, generator)
    fixed = perm[0]                          # one labeling shared by all walks
    probs = model(fixed[nodes]).softmax(-1)[..., fixed]
    flat = probs[:, TAIL:].reshape(-1, probs.size(-1))
    node_of = nodes[:, TAIL:].reshape(-1)
    implied = torch.stack([flat[node_of == v].mean(0) for v in range(shape[0] * shape[1])])
    return implied / implied.sum(1, keepdim=True)


def true_transitions(shape, topology):
    neighbors, degree = neighbor_table(shape, topology)
    true = torch.zeros(shape[0] * shape[1], shape[0] * shape[1])
    for v, (nbr, deg) in enumerate(zip(neighbors, degree)):
        true[v, nbr[nbr >= 0]] = 1 / deg.float()
    return true


def spectral_layout(P: torch.Tensor) -> torch.Tensor:
    """Laplacian eigenmap of the symmetrized transition matrix (4 nontrivial coords)."""
    adjacency = (P + P.T) / 2
    d = adjacency.sum(1)
    laplacian = torch.diag(d.rsqrt()) @ (torch.diag(d) - adjacency) @ torch.diag(d.rsqrt())
    return (torch.diag(d.rsqrt()) @ torch.linalg.eigh(laplacian)[1])[:, 1:5]


def fig_structure(best: dict, shape=(4, 5)):
    m, n = shape
    panels = (("grid", (0, 1), "row"), ("cylinder", (0, 1), "row"), ("torus", (0, 1), "col"), ("torus", (2, 3), "row"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, (topology, dims, hue) in zip(axes, panels):
        coords = spectral_layout(implied_transitions(load_model(best[topology]), topology, shape))
        xy = coords[:, list(dims)]
        draw_edges(ax, xy.numpy(), shape, topology, color="#c3c2b7", lw=0.9)
        color = torch.arange(m * n) // n if hue == "row" else torch.arange(m * n) % n
        ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap="twilight", vmin=0, vmax=(m if hue == "row" else n),
                   s=190, zorder=3, edgecolors="white")
        for v in range(m * n):
            ax.text(xy[v, 0], xy[v, 1], f"{v // n},{v % n}", ha="center", va="center", fontsize=6, color="white", zorder=4)
        ax.set_title(f"{topology} · eigenmap dims {dims[0] + 1}–{dims[1] + 1} · colored by {hue}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("The model's map of one labeled 4×5 lattice: spectral embedding of its implied transition matrix,\n"
                 "true edges in gray (dashed = wraparound). Nodes labeled row,col.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_structure.png", bbox_inches="tight")
    plt.close(fig)


def fig_adjacency(best: dict, shape=(4, 5)):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    for col, topology in enumerate(TOPOLOGIES):
        implied = implied_transitions(load_model(best[topology]), topology, shape)
        for row, matrix in enumerate((implied, true_transitions(shape, topology))):
            ax = axes[row][col]
            ax.imshow(matrix, cmap=BLUES, vmin=0, vmax=0.5)
            ax.set_title(f"{topology} · {'model implied' if row == 0 else 'true'} P(next node | node)", fontsize=9)
            ticks = range(0, shape[0] * shape[1], shape[1])
            ax.set_xticks(ticks, [f"{t // shape[1]},0" for t in ticks], fontsize=7)
            ax.set_yticks(ticks, [f"{t // shape[1]},0" for t in ticks], fontsize=7)
            ax.grid(False)
    fig.suptitle("Implied vs true transition matrices on one labeled 4×5 lattice (nodes in row-major order)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_adjacency.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    runs = load_runs()
    fig_dataset()
    fig_training(runs)
    fig_sweep(runs)

    long_runs = load_runs(long=True)
    if long_runs:
        fig_long(long_runs)
        runs = {**runs, **long_runs}
    best = {t: max((n for n in runs if n.startswith(t)), key=lambda n: runs[n]["legal|ood 6x6"][-1] + runs[n]["legal|in"][-1]) for t in TOPOLOGIES}
    print("best per topology:", best)
    fig_incontext(best)
    fig_transfer(best)
    fig_structure(best)
    fig_adjacency(best)


if __name__ == "__main__":
    main()
