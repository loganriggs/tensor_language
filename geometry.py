"""Why are neighbors stored nearby? Exact path decomposition of node representations.

The residual stream is an exact sum of path contributions (embed, o1, o2, [o3]
with the residual coefficients of the architecture), so the windowed per-node
mean rep decomposes as H = Σ_paths H_p, the Gram matrix as Σ_{p,q} H_p H_q^T,
and the Gram–adjacency alignment attributes exactly over path pairs.

Also: similarity-by-graph-distance profiles, and content regression of a path's
contribution onto {own/neighbor/2-hop} × {embed, unembed} token bases.

Usage: python geometry.py   (prints the attribution tables)
"""

import json
from pathlib import Path

import torch

import analysis_general
import analysis_geo
from graphs import TRAIN_POOLS, walk_pool
from icl_reps import WINDOW

torch.set_grad_enabled(False)

GRID = TRAIN_POOLS["grid"][3]                     # fixed 4×5 structure
N = 20

MODELS = {                                        # name -> loader
    "single-task grid (geodata)": lambda: analysis_geo.load_model("grid_L2_d128_long"),
    "gridonly seed0": lambda: analysis_general.load_model("bilin-lerp-2L-gridonly"),
    "gridonly seed2": lambda: analysis_general.load_model("bilin-lerp-2L-gridonly-seed2"),
    "gridonly seed3": lambda: analysis_general.load_model("bilin-lerp-2L-gridonly-seed3"),
    "gridonly seed4": lambda: analysis_general.load_model("bilin-lerp-2L-gridonly-seed4"),
    "grid+dring seed0": lambda: analysis_general.load_model("bilin-lerp-2L-grid+dring"),
    "grid+dring seed2": lambda: analysis_general.load_model("bilin-lerp-2L-grid+dring-seed2"),
    "grid+ring": lambda: analysis_general.load_model("bilin-lerp-2L-grid+ring"),
    "grid+tree": lambda: analysis_general.load_model("bilin-lerp-2L-grid+tree"),
    "multi seed0": lambda: analysis_general.load_model("bilin-lerp-2L"),
    "multi seed1": lambda: analysis_general.load_model("bilin-lerp-2L-seed1"),
    "multi seed2": lambda: analysis_general.load_model("bilin-lerp-2L-seed2"),
    "multi softmax-add-3L": lambda: analysis_general.load_model("softmax-add-3L"),
    "multi softmax-add-3L s1": lambda: analysis_general.load_model("softmax-add-3L-seed1"),
}


def grid_walks(seed: int = 21, n_walks: int = 512):
    generator = torch.Generator().manual_seed(seed)
    _, nodes, perm, _ = walk_pool([GRID], n_walks, generator)
    return nodes, perm[0]


def adjacency():
    nb, deg = GRID
    A = torch.zeros(N, N)
    for v in range(N):
        A[v, nb[v][nb[v] >= 0]] = 1.0
    return A


def graph_distances(A):
    D = torch.full((N, N), 99)
    D.fill_diagonal_(0)
    frontier = A.bool()
    d = 1
    while frontier.any():
        D[frontier & (D > d)] = d
        frontier = ((frontier.float() @ A) > 0) & (D > d)
        d += 1
        if d > N:
            break
    return D


@torch.inference_mode()
def path_components(model, nodes, fixed, t: int = 256):
    """Exact per-path contributions to the final stream, window-averaged per node.
    Returns dict path -> (N, d) matrix; their sum is the usual mean rep."""
    tokens = fixed[nodes]
    x = model.embed(tokens)
    comps = {"embed": x}
    for i, layer in enumerate(model.layers):
        z = layer(x) - (x if layer.residual == "add" else 0.5 * x)   # the o(z) write (scaled)
        if layer.residual == "lerp":
            comps = {k: 0.5 * v for k, v in comps.items()}
        comps[f"o{i+1}"] = z
        x = layer(x)

    lo = max(0, t - WINDOW)
    window_nodes = nodes[:, lo:t].reshape(-1)
    out = {}
    for name, c in comps.items():
        flat = c[:, lo:t].reshape(-1, c.size(-1))
        out[name] = torch.stack([flat[window_nodes == v].mean(0) for v in range(N)])
    return out


def alignment(G, A):
    """Pearson corr between off-diagonal Gram and adjacency (the organization stat)."""
    off = ~torch.eye(N, dtype=torch.bool)
    return torch.corrcoef(torch.stack([G[off], A[off]]))[0, 1].item()


def attribution(comps, A):
    """Numerator shares: how much each path pair contributes to the Gram–adjacency
    covariance. Shares sum to 1 (of the total covariance, sign included)."""
    off = ~torch.eye(N, dtype=torch.bool)
    Ac = A[off] - A[off].mean()
    centered = {k: v - v.mean(0) for k, v in comps.items()}
    names = list(centered)
    total = sum(centered.values())
    total_cov = ((total @ total.T)[off] * Ac).sum().item()
    shares = {}
    for i, a in enumerate(names):
        for b in names[i:]:
            G = centered[a] @ centered[b].T
            cov = (G[off] * Ac).sum().item() * (1 if a == b else 1)
            if a != b:
                cov += ((centered[b] @ centered[a].T)[off] * Ac).sum().item()
            shares[f"{a}×{b}" if a != b else a] = cov / abs(total_cov)
    return shares, total_cov


def distance_profile(comps, D):
    total = sum(comps.values())
    Hc = total - total.mean(0)
    Hn = torch.nn.functional.normalize(Hc, dim=1)
    S = Hn @ Hn.T
    return {d: S[D == d].mean().item() for d in (1, 2, 3, 4)}


def main():
    nodes, fixed = grid_walks()
    A = adjacency()
    D = graph_distances(A)
    results = {}
    for name, load in MODELS.items():
        try:
            model = load().cuda()
        except FileNotFoundError:
            continue
        comps = path_components(model, nodes.cuda(), fixed.cuda())
        comps = {k: v.cpu().double() for k, v in comps.items()}
        model.cpu()
        org = alignment(sum(comps.values()) @ sum(comps.values()).T, A)  # placeholder, recomputed below
        centered = {k: v - v.mean(0) for k, v in comps.items()}
        total = sum(centered.values())
        org = alignment(total @ total.T, A)
        shares, total_cov = attribution(comps, A)
        prof = distance_profile(comps, D)
        results[name] = dict(org=org, shares=shares, profile=prof)
        top = sorted(shares.items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"\n{name:26s} org {org:+.2f} | sim@d1 {prof[1]:+.3f} d2 {prof[2]:+.3f} d3 {prof[3]:+.3f}")
        print("   top contributions:", "  ".join(f"{k}:{v:+.2f}" for k, v in top))
    Path("geometry_results.json").write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
