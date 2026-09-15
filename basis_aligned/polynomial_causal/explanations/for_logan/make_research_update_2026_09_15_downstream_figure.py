#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "assets" / "research_update_2026-09-15_qk1_downstream_response.png"
names = ["Attention9\n(direct)", "Attention17", "MLP17", "Attention11", "MLP13"]
aligned = [0.53301, 0.24518, -0.13785, 0.07398, 0.04878]
colors = ["#386cb0" if value >= 0 else "#e34a33" for value in aligned]

fig, ax = plt.subplots(figsize=(8.6, 4.5))
bars = ax.bar(names, aligned, color=colors, edgecolor="#263238", linewidth=0.7)
ax.axhline(0, color="#263238", linewidth=0.8)
ax.set_ylabel("Aligned fraction of pre-RMS numerator response")
ax.set_title("Largest downstream responses to the selected QK1 removal")
ax.set_ylim(-0.19, 0.60); ax.grid(axis="y", alpha=0.25)
for bar, value in zip(bars, aligned):
    ax.text(bar.get_x()+bar.get_width()/2, value+(0.018 if value>=0 else -0.025), f"{value:+.3f}", ha="center", va="bottom" if value>=0 else "top")
fig.tight_layout(); OUT.parent.mkdir(parents=True,exist_ok=True); fig.savefig(OUT,dpi=180,bbox_inches="tight")
