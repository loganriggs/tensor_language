"""In-context representations à la Park et al. (ICLR 2025, "In-Context Learning
of Representations"): mean residual-stream activation per token over a trailing
window, tracked as context grows, projected on its top principal components.

Their LLM result: top PCs converge to the graph's spectral embedding (Dirichlet
energy minimization; neighbors pulled together). Our small bilinear models do the
OPPOSITE: the Gram matrix of mean reps correlates negatively with adjacency
(neighbors pushed apart, energy maximization), so the top PCs are high-frequency
graph harmonics (a checkerboard-like anti-map with neighbors antipodal), and the
smooth lattice coordinates hide in the low-variance tail of the PC spectrum.

We checked the seductive alternative — searching for a low-Dirichlet-energy 3D
subspace of the reps — and rejected it: with 20 points in 19 dims it reaches the
same energy for shuffled node labels, i.e. it visualizes nothing.

Usage: python icl_reps.py   (writes figures/geo_icl_reps.png)
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from analysis_geo import FIGURES, TOPOLOGIES, draw_edges, load_model, load_runs, spectral_layout, true_transitions
from geodata import neighbor_table, walk_batch

torch.set_grad_enabled(False)

SHAPE = (4, 5)
N_NODES = SHAPE[0] * SHAPE[1]
CONTEXTS = (8, 16, 32, 64, 128, 192, 256)
WINDOW = 50
TOPO_COLORS = {"grid": "#2a78d6", "cylinder": "#eda100", "torus": "#e34948"}


def best_models() -> dict:
    runs = {**load_runs(), **load_runs(long=True)}
    return {t: max((n for n in runs if n.startswith(t)),
                   key=lambda n: runs[n]["legal|ood 6x6"][-1] + runs[n]["legal|in"][-1]) for t in TOPOLOGIES}


def mean_reps(model, topology: str, n_walks: int = 512, seed: int = 21):
    """Windowed per-token mean residual (final layer) at each context length."""
    generator = torch.Generator().manual_seed(seed)
    _, nodes, perm = walk_batch(n_walks, SHAPE, topology, generator)
    stream = model.residuals(perm[0][nodes])[-1]
    reps = {}
    for t in CONTEXTS:
        lo = max(0, t - WINDOW)
        window_nodes = nodes[:, lo:t].reshape(-1)
        flat = stream[:, lo:t].reshape(-1, stream.size(-1))
        reps[t] = torch.stack([flat[window_nodes == v].mean(0) for v in range(N_NODES)])
    return reps


def adjacency(topology: str):
    neighbors, _ = neighbor_table(SHAPE, topology)
    A = torch.zeros(N_NODES, N_NODES)
    for v, nbr in enumerate(neighbors):
        A[v, nbr[nbr >= 0]] = 1.0
    return A


def gram_adjacency_corr(H, A):
    Hc = H - H.mean(0)
    off = ~torch.eye(N_NODES, dtype=torch.bool)
    return torch.corrcoef(torch.stack([(Hc @ Hc.T)[off], A[off]]))[0, 1].item()


def pca_coords(H, k: int = 3):
    """Top-k PC projection of row-normalized mean reps (normalization tames
    visit-frequency outliers, cf. the paper's frequency artifact)."""
    Hn = F.normalize(H - H.mean(0), dim=1)
    Hn = Hn - Hn.mean(0)
    U, S, _ = torch.linalg.svd(Hn, full_matrices=False)
    return U[:, :k] * S[:k]


def pc_spectrum_alignment(H, topology: str):
    """Per PC index: variance share and max |corr| with the true lattice harmonics."""
    truth = spectral_layout(true_transitions(SHAPE, topology))  # 4 lowest nontrivial coords
    Hc = H - H.mean(0)
    U, S, _ = torch.linalg.svd(Hc, full_matrices=False)
    var = (S**2 / (S**2).sum()).tolist()
    corr = [max(abs(torch.corrcoef(torch.stack([U[:, k], truth[:, j]]))[0, 1].item()) for j in range(4))
            for k in range(len(S))]
    return var, corr


def procrustes_chain(frames):
    """Rotate each frame onto the previous one so slider/series transitions are smooth."""
    aligned = [frames[0]]
    for frame in frames[1:]:
        u, _, vt = torch.linalg.svd(frame.T @ aligned[-1])
        aligned.append(frame @ (u @ vt))
    return aligned


def compute_all(best: dict) -> dict:
    out = {}
    for topology, name in best.items():
        reps = mean_reps(load_model(name), topology)
        A = adjacency(topology)
        out[topology] = {
            "corr": {t: gram_adjacency_corr(H, A) for t, H in reps.items()},
            "pca": procrustes_chain([pca_coords(H) for H in reps.values()]),
            "spectrum": pc_spectrum_alignment(reps[max(CONTEXTS)], topology),
        }
    return out


def fig_icl(results: dict):
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(2, len(CONTEXTS), height_ratios=(1.05, 1))

    for col, (t, xyz) in enumerate(zip(CONTEXTS, results["grid"]["pca"])):
        ax = fig.add_subplot(gs[0, col])
        xy = xyz[:, :2]
        draw_edges(ax, xy.numpy(), SHAPE, "grid", color="#c3c2b7", lw=0.8)
        ax.scatter(xy[:, 0], xy[:, 1], c=torch.arange(N_NODES) // SHAPE[1], cmap="twilight",
                   vmin=0, vmax=SHAPE[0], s=90, zorder=3, edgecolors="white", linewidths=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"context = {t}", fontsize=9)
        if col == 0:
            ax.set_ylabel("grid · mean reps · PC 1–2", fontsize=9)

    left = fig.add_subplot(gs[1, :3])
    for topology, r in results.items():
        left.plot(list(r["corr"]), list(r["corr"].values()), color=TOPO_COLORS[topology],
                  lw=1.8, marker="o", ms=4, label=topology)
    left.axhline(0, color="#c3c2b7", lw=0.8)
    left.set_xscale("log")
    left.set_xlabel("context length")
    left.set_ylabel("corr(Gram of mean reps, adjacency)")
    left.set_title("Neighbor reps are ANTI-correlated, and it grows in-context\n(Park et al.'s LLMs converge to corr > 0 instead)", fontsize=9.5)
    left.legend(fontsize=9)

    right = fig.add_subplot(gs[1, 3:])
    var, corr = results["grid"]["spectrum"]
    ks = range(1, len(corr) + 1)
    right.bar(ks, corr, color="#2a78d6", label="max |corr| with true lattice harmonics")
    right.plot(ks, [v / max(var) for v in var], color="#898781", lw=1.4, marker=".", ms=5,
               label="variance share (rescaled)")
    right.set_xticks(list(ks))
    right.set_xlabel("principal component index (grid model, context 256)")
    right.set_title("The smooth lattice map exists — but in the LOW-variance PCs", fontsize=9.5)
    right.legend(fontsize=8.5)

    fig.suptitle("In-context representations (windowed mean residual per token, Park et al. protocol): "
                 "top PCs form an anti-map — neighbors antipodal, edges crossing the center", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_icl_reps.png", bbox_inches="tight")
    plt.close(fig)


def main():
    best = best_models()
    results = compute_all(best)
    fig_icl(results)
    for topology, r in results.items():
        print(topology, "corr:", {t: round(c, 3) for t, c in r["corr"].items()})
    return results, best


if __name__ == "__main__":
    main()
