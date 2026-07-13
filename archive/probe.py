"""Linear probes from the residual stream to the graph's spectral-embedding
coordinates — the belief-state-geometry methodology of Shai et al. (Mess3 /
simplex): if the structure is linearly embedded, a ridge probe fit on training
walks should place held-out activations at the right lattice coordinates.

Targets: the true graph's Laplacian eigenmap coords of the *current node*
(2 dims for grid, 3 for cylinder, 4 for torus — its natural harmonic count).
Controls: held-out walks for all numbers, plus a node-shuffle control (same
probe fit with node<->coordinate assignment permuted) that must fail.

Usage: python probe.py   (writes figures/geo_probe.png and prints the R² table)
"""

import matplotlib.pyplot as plt
import torch

from analysis_geo import FIGURES, draw_edges, load_model, spectral_layout, true_transitions
from geodata import TAIL, walk_batch
from icl_reps import SHAPE, N_NODES, best_models

torch.set_grad_enabled(False)

N_WALKS, N_TRAIN = 1024, 768
DIMS = {"grid": 2, "cylinder": 3, "torus": 4}
GRID_VARIANTS = {
    "grid_L2_d128_long": "bilinear 2L",
    "grid_L3_d128_add": "bilinear 3L+add",
    "grid_L3_d128_normadd": "bilinear 3L+add+norm",
    "grid_L2_d128_softmax": "softmax 2L",
}


def activations(name: str, topology: str, seed: int = 33):
    """Per-position final-layer residuals and node ids on one fixed labeling (steady state)."""
    model = load_model(name).cuda()
    generator = torch.Generator().manual_seed(seed)
    _, nodes, perm = walk_batch(N_WALKS, SHAPE, topology, generator)
    stream = model.residuals(perm[0][nodes].cuda())[-1].cpu()
    model.cpu()
    return stream[:, TAIL:], nodes[:, TAIL:]


def ridge(X, Y, lam: float = 1e-2):
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    return torch.linalg.solve(X1.T @ X1 + lam * torch.eye(X1.size(1)), X1.T @ Y)


def r2(X, Y, W):
    pred = torch.cat([X, torch.ones(len(X), 1)], 1) @ W
    return (1 - (Y - pred).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum()).item(), pred


LAMBDAS = torch.logspace(-2, 7, 19, dtype=torch.float64)
N_SHUFFLES = 8


def probe(name: str, topology: str):
    """Probe maps at small ridge, plus the full regularization path true-vs-shuffled."""
    stream, nodes = activations(name, topology)
    coords = spectral_layout(true_transitions(SHAPE, topology))[:, :DIMS[topology]]
    coords = coords / coords.std(0)

    X = (stream.reshape(-1, stream.size(-1)) - stream.mean((0, 1))).double()
    node_of = nodes.reshape(-1)
    split = N_TRAIN * stream.size(1)                     # walk-level split
    scale = (X[:split] ** 2).sum(0).mean()               # mean diagonal of X'X -> scale-free lambda
    gram = X[:split].T @ X[:split] / scale

    generator = torch.Generator().manual_seed(7)
    targets = {"true": [coords]}
    targets["shuffled"] = [coords[torch.randperm(N_NODES, generator=generator).argsort()] for _ in range(N_SHUFFLES)]

    def fit(target, lam):
        Y = target[node_of].double()
        W = torch.linalg.solve(gram + lam * torch.eye(gram.size(0), dtype=torch.float64),
                               X[:split].T @ Y[:split] / scale)
        pred = X[split:] @ W
        score = (1 - (Y[split:] - pred).pow(2).sum() / (Y[split:] - Y[split:].mean(0)).pow(2).sum()).item()
        return score, pred

    out = {"path": {}, "node_of": node_of[split:]}
    for tag, variants in targets.items():
        out["path"][tag] = [sum(fit(v, lam)[0] for v in variants) / len(variants) for lam in LAMBDAS]
    out["true"], pred = fit(coords, LAMBDAS[0])
    out["pred"] = pred.float()
    out["shuffled"], pred = fit(targets["shuffled"][0], LAMBDAS[0])
    out["pred_shuffled"] = pred.float()
    return out


def panel(ax, pred, node_of, r2_score, topology: str, title: str):
    sub = torch.randperm(len(pred))[:3000]
    rows = (node_of[sub] // SHAPE[1]).float()
    ax.scatter(pred[sub, 0], pred[sub, 1], c=rows, cmap="twilight", vmin=0, vmax=SHAPE[0],
               s=3, alpha=0.25, zorder=2)
    centers = torch.stack([pred[node_of == v].mean(0) for v in range(N_NODES)])
    draw_edges(ax, centers[:, :2].numpy(), SHAPE, topology, color="#898781", lw=1.1)
    ax.scatter(centers[:, 0], centers[:, 1], c=torch.arange(N_NODES) // SHAPE[1], cmap="twilight",
               vmin=0, vmax=SHAPE[0], s=150, zorder=3, edgecolors="white")
    for v in range(N_NODES):
        ax.text(centers[v, 0], centers[v, 1], f"{v // SHAPE[1]},{v % SHAPE[1]}", fontsize=5.5,
                ha="center", va="center", color="white", zorder=4)
    ax.set_title(f"{title}\ntest R² {r2_score:.3f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def main():
    best = best_models()
    results = {name: probe(name, "grid") for name in GRID_VARIANTS}
    results |= {best[t]: probe(best[t], t) for t in ("cylinder", "torus")}
    print(f"{'model':26s} {'test R²':>8s} {'shuffle':>8s}")
    for name, r in results.items():
        print(f"{name:26s} {r['true']:8.3f} {r['shuffled']:8.3f}")

    fig = plt.figure(figsize=(14.5, 7.6))
    gs = fig.add_gridspec(2, 4)

    main_grid = results["grid_L2_d128_long"]
    panel(fig.add_subplot(gs[0, 0]), main_grid["pred"], main_grid["node_of"], main_grid["true"],
          "grid", "bilinear 2L → TRUE coords")
    panel(fig.add_subplot(gs[0, 1]), main_grid["pred_shuffled"], main_grid["node_of"], main_grid["shuffled"],
          "grid", "same model → SHUFFLED coords\n(probe drawing a fake lattice equally well)")
    for col, t in enumerate(("cylinder", "torus")):
        r = results[best[t]]
        panel(fig.add_subplot(gs[0, col + 2]), r["pred"], r["node_of"], r["true"], t, f"{t} · main model → TRUE coords")

    colors = {"grid_L2_d128_long": "#2a78d6", "grid_L3_d128_add": "#104281",
              "grid_L3_d128_normadd": "#86b6ef", "grid_L2_d128_softmax": "#e34948"}
    for col, (name, label) in enumerate(GRID_VARIANTS.items()):
        ax = fig.add_subplot(gs[1, col])
        path = results[name]["path"]
        ax.plot(LAMBDAS, path["true"], color=colors[name], lw=1.8, label="true coords")
        ax.plot(LAMBDAS, path["shuffled"], color="#898781", lw=1.5, ls=(0, (3, 2)), label="shuffled (mean of 8)")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("ridge λ")
        if col == 0:
            ax.set_ylabel("held-out R²")
            ax.legend(fontsize=8)
    fig.suptitle("Linear probes to the true spectral coordinates (Shai et al.-style), with the honest control.\n"
                 "Top: at low ridge the probe is VACUOUS — it hits true and shuffled targets equally (any 20 points are linearly reachable).\n"
                 "Bottom: under increasing ridge the probe can only read dominant directions — the true-vs-shuffled gap now measures geometry.",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "geo_probe.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
