"""Figures for the pretrained-LLM Park-protocol runs (reads runs_llm/).

Usage: python llm_figs.py   -> figures/llm_org.png, llm_maps.png, llm_coeffs.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis import BLUES, DIVERGING, INK, SECONDARY

FIGURES = Path("figures")
RUNS = Path("runs_llm")
CONTEXTS = (8, 16, 32, 64, 128, 256, 400)
GRAPHS = ("grid45", "ring12", "ring7")
ORDER = ("gpt2", "pythia-410m", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B")
TOY = {"grid45": ("toy bilin multi +0.66 / toy softmax −0.80"),
       "ring12": "", "ring7": ""}


def load_all():
    out = {}
    for tag in ORDER:
        d = RUNS / tag
        if (d / "org.json").exists():
            out[tag] = {k: json.loads((d / f"{k}.json").read_text())
                        for k in ("org", "behavior", "coeffs", "meta")}
            reps = d / "reps.pt"
            out[tag]["reps"] = torch.load(reps) if reps.exists() else None
    return out


def fig_org(data):
    fig, axes = plt.subplots(len(GRAPHS), len(data), figsize=(3.1 * len(data), 7.6),
                             squeeze=False)
    for j, (tag, d) in enumerate(data.items()):
        for i, g in enumerate(GRAPHS):
            org = np.array(d["org"][g])           # (L+1) x n_ctx
            ax = axes[i][j]
            im = ax.imshow(org, cmap=DIVERGING, vmin=-0.85, vmax=0.85, aspect="auto",
                           origin="lower")
            ax.set_xticks(range(len(CONTEXTS)), CONTEXTS, fontsize=6.5, rotation=45)
            if j == 0:
                ax.set_ylabel(f"{g}\nlayer", fontsize=8.5)
            if i == 0:
                ax.set_title(tag, fontsize=9)
            if i == len(GRAPHS) - 1:
                ax.set_xlabel("context (tokens)", fontsize=8)
            best = org[:, -1].argmax()
            ax.plot([len(CONTEXTS) - 1], [best], marker="<", color=INK, ms=5)
            ax.text(len(CONTEXTS) - 1.45, best, f"{org[best, -1]:+.2f}", fontsize=7,
                    ha="right", va="center", color=INK)
    fig.colorbar(im, ax=axes[:, -1], shrink=0.5, label="organization (Gram–adjacency corr)")
    fig.suptitle("Pretrained LLMs on random-word graph walks: organization by layer × context\n"
                 "(◄ = best layer at ctx 400; toy references on grid: bilinear multi +0.66, softmax-add-3L −0.80)",
                 fontsize=10)
    fig.savefig(FIGURES / "llm_org.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def grid_layout(g):
    if g == "grid45":
        return np.array([(v % 5, v // 5) for v in range(20)], float)
    n = int(g[4:])
    ang = 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(ang), np.sin(ang)], 1)


def edges(g):
    if g == "grid45":
        out = []
        for v in range(20):
            r, c = divmod(v, 5)
            if c < 4: out.append((v, v + 1))
            if r < 3: out.append((v, v + 5))
        return out
    n = int(g[4:])
    return [(v, (v + 1) % n) for v in range(n)]


def procrustes(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    u, _, vt = np.linalg.svd(Xc.T @ Yc)
    R = (u @ vt).T
    return Xc @ R.T


def fig_maps(data):
    rows = [(tag, g) for g in ("grid45", "ring7") for tag in data]
    n_col = len(data)
    fig, axes = plt.subplots(2, n_col, figsize=(2.9 * n_col, 6.4), squeeze=False)
    for j, tag in enumerate(data):
        d = data[tag]
        for i, g in enumerate(("grid45", "ring7")):
            ax = axes[i][j]
            org = np.array(d["org"][g])
            best = int(org[:, -1].argmax())
            H = d["reps"][g][best].float().numpy()
            Hc = H - H.mean(0)
            Hn = Hc / (np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-9)
            Hn = Hn - Hn.mean(0)
            U, S, _ = np.linalg.svd(Hn, full_matrices=False)
            P = U[:, :2] * S[:2]
            P = procrustes(P, grid_layout(g))
            for a, b in edges(g):
                ax.plot(P[[a, b], 0], P[[a, b], 1], color=SECONDARY, lw=0.9, alpha=0.6, zorder=1)
            ax.scatter(P[:, 0], P[:, 1], s=42, c=np.arange(len(P)), cmap=BLUES, zorder=2,
                       edgecolors=INK, linewidths=0.4)
            ax.set_title(f"{tag} · layer {best} · org {org[best, -1]:+.2f}", fontsize=8.2)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if j == 0:
                ax.set_ylabel({"grid45": "4×5 grid", "ring7": "7-ring"}[g], fontsize=9)
    fig.suptitle("Top-2 PC maps at each model's best layer (ctx 400), edges = true graph", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "llm_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_coeffs(data):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for ax, (tag, d) in zip([None] * 0, []):
        pass
    # panel 1: behavior vs context; panel 2: ownU by layer (grid); panel 3: nbrU by layer
    ax = axes[0]
    for tag, d in data.items():
        beh = d["behavior"]["grid45"]["legal_top"]
        ax.plot(CONTEXTS, [beh[str(t)] for t in CONTEXTS], marker="o", ms=3, label=tag)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("context (tokens)"); ax.set_ylabel("legal top-1 rate (grid)")
    ax.legend(fontsize=7); ax.set_title("in-context task learning", fontsize=9.5)
    for ax, key, title in ((axes[1], "ownU", "own-token content (ownU) by depth"),
                           (axes[2], "nbrU", "neighbor evidence (nbrU) by depth")):
        for tag, d in data.items():
            c = d["coeffs"]["grid45"]
            depth = np.linspace(0, 1, len(c))
            ax.plot(depth, [x[key] for x in c], marker="o", ms=2.5, label=tag)
        ax.axhline(0, color=SECONDARY, lw=0.8)
        ax.set_xlabel("relative depth"); ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "llm_coeffs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data = load_all()
    print("loaded:", list(data))
    fig_org(data)
    fig_maps(data)
    fig_coeffs(data)
    print("wrote figures/llm_org.png llm_maps.png llm_coeffs.png")
