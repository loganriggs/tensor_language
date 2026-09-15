#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "assets" / "research_update_2026-09-15_regional_late_group.png"
labels = [r"$D\times D$", r"$D\times R$", r"$R\times D$", r"$R\times R$"]
ratios = [0.4548041, 0.2533481, 0.1904141, 0.1118645]
colors = ["#386cb0", "#7fc97f", "#beaed4", "#fdc086"]

fig, ax = plt.subplots(figsize=(8.2, 4.4))
bars = ax.bar(labels, ratios, color=colors, edgecolor="#263238", linewidth=0.7)
ax.set_ylabel("Change norm / parent change norm")
ax.set_title("Regional head 9.8 QK1 carry interaction blocks")
ax.set_ylim(0, 0.52)
ax.grid(axis="y", alpha=0.25)
for bar, value in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center")
ax.text(0.99, 0.96, r"$D=A_5+M_5+M_6+M_7$", transform=ax.transAxes, ha="right", va="top")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
