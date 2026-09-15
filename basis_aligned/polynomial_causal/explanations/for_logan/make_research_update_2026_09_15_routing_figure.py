#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "assets" / "research_update_2026-09-15_regional_routing_fresh.png"
families = ["Template 0", "Template 1"]
panels = [
    ("Selected / full head", [0.52843, 0.59816], 0.10, (0, 0.7)),
    (r"Selected / $R\times R$", [6.990, 5.763], 2.0, (0, 8.0)),
    ("Unrelated / target RMS", [0.53452, 0.26075], 0.50, (0, 0.7)),
]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
for ax, (title, vals, gate, ylim) in zip(axes, panels):
    bars = ax.bar(families, vals, color=["#386cb0", "#7fc97f"], edgecolor="#263238", linewidth=0.7)
    ax.axhline(gate, color="#c62828", linestyle="--", linewidth=1.2, label=f"gate {gate:g}")
    ax.set_title(title); ax.set_ylim(*ylim); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=8)
    for bar, val in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, val+0.025*(ylim[1]-ylim[0]), f"{val:.3f}", ha="center", fontsize=9)
fig.suptitle("Fresh recursive removal of late-touching head 9.8 QK1 blocks")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
