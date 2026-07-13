"""Interactive 3D viewer: two views of each best model's learned graph, exported
as a self-contained rotatable HTML page.

- "behavior": Laplacian eigenmap of the model's implied transition matrix (the
  unfolded lattice; static in context).
- "residual PCA": Park et al.-style top principal components of the windowed
  per-token mean residual stream, at increasing context length (our models
  organize these as an ANTI-map: neighbors antipodal). Frames are
  Procrustes-aligned so the context slider animates smoothly.

Usage: python structure3d.py   (after train_geo.py; writes figures/geo_structure_3d.html)
"""

import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from analysis_geo import implied_transitions, load_model, spectral_layout
from geodata import neighbor_table
from icl_reps import CONTEXTS, SHAPE, best_models, mean_reps, pca_coords, procrustes_chain

TEMPLATE = Path(__file__).parent / "structure3d_template.html"
OUT = Path("figures/geo_structure_3d.html")


def rounded(coords) -> list:
    return [[round(v, 4) for v in row] for row in coords.tolist()]


def panel_data(topology: str, name: str, title: str | None = None) -> dict:
    m, n = SHAPE
    model = load_model(name)

    behavior = spectral_layout(implied_transitions(model, topology, SHAPE))
    behavior = behavior / behavior.std(0, keepdim=True)

    frames = procrustes_chain([pca_coords(H, k=4) for H in mean_reps(model, topology).values()])
    frames = [f / f.std() for f in frames]

    neighbors, _ = neighbor_table(SHAPE, topology)
    edges = [
        {"a": a, "b": b, "wrap": abs(a // n - b // n) > 1 or abs(a % n - b % n) > 1}
        for a in range(m * n) for b in neighbors[a].tolist() if b > a
    ]
    colors = [mcolors.to_hex(cm.twilight(r / m)) for r in range(m)]
    return {
        "topology": title or topology,
        "model": name,
        "edges": edges,
        "labels": [f"{v // n},{v % n}" for v in range(m * n)],
        "colors": [colors[v // n] for v in range(m * n)],
        "contexts": list(CONTEXTS),
        "sources": {"behavior": [rounded(behavior)], "residual": [rounded(f) for f in frames]},
    }


def cycle_panel(name: str = "L2_d64_h1", length: int = 7) -> dict:
    """The cycle task's phase circle, in 3D, from the original cycle model.

    residual: phase centroids of the token-averaged, drift-detrended residual
    stream (the results.md circle, now with 4 PCA dims). behavior: eigenmap of
    the model's implied phase-transition matrix.
    """
    import torch

    from analysis import load_model as load_cycle_model
    from data import N_CTX as CYCLE_CTX
    from data import sample_cycles

    model = load_cycle_model(name)
    with torch.inference_mode():
        tokens = sample_cycles(512, torch.full((512,), length), generator=torch.Generator().manual_seed(11))
        stream = model.residuals(tokens)[-1]
        probs = model(tokens).softmax(-1)

    keep = slice(2 * length, CYCLE_CTX - length)
    phase = (torch.arange(CYCLE_CTX) % length)[keep]

    mean = stream.mean(0)
    kernel = torch.ones(1, 1, length) / length
    trend = torch.conv1d(mean.T[:, None], kernel, padding="same")[:, 0].T
    m = (mean - trend)[keep]
    m = m - m.mean(0)
    proj = m @ torch.linalg.svd(m, full_matrices=False)[2][:4].T
    residual = torch.stack([proj[phase == p].mean(0) for p in range(length)])

    # implied phase -> next-phase transitions: mass of prediction on each phase's token
    onehot = torch.zeros(512, length, 100).scatter_(2, tokens[:, :length, None], 1.0)
    mass = torch.einsum("bsv,bpv->bsp", probs[:, keep], onehot).mean(0)   # (positions, phase)
    implied = torch.stack([mass[phase == p].mean(0) for p in range(length)])
    implied = implied / implied.sum(1, keepdim=True)
    behavior = spectral_layout((implied + implied.T) / 2)

    colors = [mcolors.to_hex(cm.twilight(p / length)) for p in range(length)]
    return {
        "topology": f"cycle (L={length})",
        "model": name,
        "edges": [{"a": p, "b": (p + 1) % length, "wrap": False} for p in range(length)],
        "labels": [str(p) for p in range(length)],
        "colors": colors,
        "contexts": [CYCLE_CTX],
        "sources": {"behavior": [rounded(behavior / behavior.std(0, keepdim=True))],
                    "residual": [rounded(residual / residual.std())]},
    }


def main():
    best = best_models()
    data = [panel_data(t, best[t]) for t in best] + [cycle_panel()]
    data += [panel_data("grid", "grid_L3_d128_add", "grid · bilinear 3L+add (organizes +)"),
             panel_data("grid", "grid_L2_d128_softmax", "grid · softmax 2L (+ early, − late)")]
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
