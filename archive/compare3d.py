"""Rotatable 3D residual-geometry viewer across ALL graph structures: per panel
one fixed labeled graph, with a model dropdown (multi-family bilinear / single-
family grid specialist / multi-family softmax / true spectral reference).
Node means = top-4 plain PCs of windowed mean reps per context length; small
dots = individual per-position residual states in the same basis.

Usage: python compare3d.py   (writes figures/geo_compare_3d.html)
"""

import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch

import analysis_general
import analysis_geo
from graphs import OOD_POOLS, TRAIN_POOLS, walk_pool
from icl_reps import WINDOW

torch.set_grad_enabled(False)

TEMPLATE = Path(__file__).parent / "compare3d_template.html"
OUT = Path("figures/geo_compare_3d.html")
CONTEXTS = (8, 16, 32, 64, 128, 256)
N_SAMPLES = 200

from graphs import widening_rings

STRUCTURES = {                       # family -> (pool, index of the fixed structure)
    "widening rings 4-8-16 (unseen)": ([widening_rings()], 0),
    "ring (n=12)": (TRAIN_POOLS["ring"], 7),
    "directed ring (n=12)": (TRAIN_POOLS["dring"], 7),
    "grid 4×5": (TRAIN_POOLS["grid"], 3),
    "cylinder 4×5": (TRAIN_POOLS["cylinder"], 3),
    "torus 4×5 (unseen family)": (OOD_POOLS["torus (unseen family)"], 1),
    "random tree (n=14)": (TRAIN_POOLS["tree"], 250),
    "3-regular (n=12)": (TRAIN_POOLS["kreg"], 100),
    "ER graph (unseen family)": (OOD_POOLS["ER graph (unseen family)"], 60),
}
MODELS = {
    "multi-family bilinear": lambda: analysis_general.load_model("bilin-lerp-2L"),
    "single-family grid specialist": lambda: analysis_geo.load_model("grid_L2_d128_long"),
    "multi-family softmax (anti)": lambda: analysis_general.load_model("softmax-add-3L"),
}


def rounded(x, nd=3):
    return [[round(v, nd) for v in row] for row in x.tolist()]


def true_spectral(nb, deg, n):
    P = torch.zeros(n, n)
    for v in range(n):
        P[v, nb[v][nb[v] >= 0]] = 1.0 / deg[v]
    A = (P + P.T) / 2
    d = A.sum(1)
    L = torch.diag(d.rsqrt()) @ (torch.diag(d) - A) @ torch.diag(d.rsqrt())
    coords = (torch.diag(d.rsqrt()) @ torch.linalg.eigh(L)[1])[:, 1:5]
    return coords / coords.std(0, keepdim=True).clamp(min=1e-6)


@torch.inference_mode()
def model_views(model, nodes, tokens, n: int):
    stream = model.residuals(tokens)[-1]
    frames, clouds, cloud_nodes, prev = [], [], [], None
    sample_gen = torch.Generator().manual_seed(3)
    for t in CONTEXTS:
        lo = max(0, t - WINDOW)
        window_nodes = nodes[:, lo:t].reshape(-1)
        flat = stream[:, lo:t].reshape(-1, stream.size(-1))
        H = torch.stack([flat[window_nodes == v].mean(0) for v in range(n)])
        center = H.mean(0)
        Hc = H - center
        V = torch.linalg.svd(Hc, full_matrices=False)[2][:4].T
        coords = Hc @ V
        R = torch.eye(4)
        if prev is not None:
            u, _, vt = torch.linalg.svd(coords.T @ prev)
            R = u @ vt
        coords = coords @ R
        prev = coords
        scale = coords.std().clamp(min=1e-6)
        pick = torch.randperm(len(flat), generator=sample_gen)[:N_SAMPLES]
        frames.append(rounded(coords / scale))
        clouds.append(rounded((((flat[pick] - center) @ V @ R) / scale).clamp(-6, 6), 2))
        cloud_nodes.append(window_nodes[pick].tolist())
    return {"frames": frames, "clouds": clouds, "cloudNodes": cloud_nodes}


def node_colors(structure_name: str, truth, n: int, shape=(4, 5)):
    if "widening" in structure_name:
        ring_of = [0]*4 + [1]*8 + [2]*16
        return [mcolors.to_hex(cm.twilight(0.15 + 0.3 * ring_of[v])) for v in range(n)]
    if "grid" in structure_name or "cylinder" in structure_name or "torus" in structure_name:
        return [mcolors.to_hex(cm.twilight((v // shape[1]) / shape[0])) for v in range(n)]
    if "ring" in structure_name:
        return [mcolors.to_hex(cm.twilight(v / n)) for v in range(n)]
    angle = torch.atan2(truth[:, 1], truth[:, 0])            # color by true spectral angle
    return [mcolors.to_hex(cm.twilight(((a + torch.pi) / (2 * torch.pi)).item())) for a in angle]


def main():
    loaded = {name: load() for name, load in MODELS.items()}
    panels = []
    for structure_name, (pool, idx) in STRUCTURES.items():
        nb, deg = pool[idx]
        n = int((nb[:, 0] >= 0).sum())
        generator = torch.Generator().manual_seed(21)
        _, nodes, perm, _ = walk_pool([pool[idx]], 512, generator)
        tokens = perm[0][nodes]
        truth = true_spectral(nb, deg, n)

        models = {name: model_views(model, nodes, tokens, n) for name, model in loaded.items()}
        models["true spectral reference"] = {"frames": [rounded(truth)], "clouds": [[]], "cloudNodes": [[]]}
        panels.append({
            "title": structure_name,
            "edges": [{"a": a, "b": int(b)} for a in range(n) for b in nb[a][nb[a] >= 0] if b > a or "directed" in structure_name],
            "labels": [str(v) for v in range(n)],
            "colors": node_colors(structure_name, truth, n),
            "contexts": list(CONTEXTS),
            "models": models,
        })
        print("built", structure_name, f"(n={n})", flush=True)

    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(panels))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
