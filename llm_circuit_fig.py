"""Summary figure for session 3: circuit + causal-use + toy feedback.

Usage: python llm_circuit_fig.py -> figures/llm_circuit.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import BLUES, INK, SECONDARY

FIGURES = Path("figures")
lh = json.loads(Path("runs_llm/gpt2-localheads.json").read_text())

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.7))

# 1: locality vs head-output organization
ax = axes[0]
loc = [v["loc"] for v in lh["heads"].values()]
org = [v["org"] for v in lh["heads"].values()]
layers = [int(k.split(".")[0]) for k in lh["heads"]]
sc = ax.scatter(loc, org, c=layers, cmap=BLUES, s=22, edgecolors=INK, linewidths=0.3)
for k in ("4.11", "2.2"):
    ax.annotate(k, (lh["heads"][k]["loc"], lh["heads"][k]["org"]), fontsize=7.5,
                xytext=(4, -8), textcoords="offset points", color=INK)
ax.set_xlabel("attention mass on offsets 1–3 (locality)")
ax.set_ylabel("head-output organization")
ax.set_title(f"GPT-2, all 144 heads: locality predicts\nmap-building (r = +{lh['locality_r']:.2f})", fontsize=9.5)
plt.colorbar(sc, ax=ax, label="layer", shrink=0.8)

# 2: ablations
ax = axes[1]
conds = ["baseline", "8 random heads", "top8 local heads", "attn2 (all heads)", "attn9+attn10 (all heads)"]
labels = ["baseline", "8 random\nheads", "top-8 local\nheads", "all of\nattn2", "attn9+10\n(amplifiers)"]
orgv = [lh[c]["org"] for c in conds]
legv = [lh[c]["legal"] for c in conds]
x = np.arange(len(conds))
ax.bar(x - 0.2, orgv, 0.38, label="map @ L11", color="#2a78d6")
ax.bar(x + 0.2, legv, 0.38, label="legal rate", color="#c9c7c1")
ax.set_xticks(x, labels, fontsize=7.5)
ax.axhline(0, color=SECONDARY, lw=0.8)
ax.legend(fontsize=8)
ax.set_title("mean-ablation: composers vs amplifiers", fontsize=9.5)

# 3: causal use patches
ax = axes[2]
names = ["baseline", "own\nmean", "delete\n4 PCs", "random\nperm", "automorph.\n(geo kept)"]
legal = [0.826, 0.822, 0.589, 0.642, 0.278]
colors = ["#c9c7c1", "#2a78d6", "#86b6ef", "#e59c9b", "#8c2b2b"]
ax.bar(range(5), legal, color=colors, edgecolor=INK, linewidth=0.4)
ax.set_xticks(range(5), names, fontsize=7)
ax.set_ylabel("legal rate")
ax.set_title("patch the map subspace @ L8:\ncontent is used, arrangement is not", fontsize=9.5)

# 4: toy burst
ax = axes[3]
groups = [
    ("softmax\nsix-family", [-0.80, -0.67], "#8c2b2b"),
    ("softmax\ngrid+dring", [-0.55, -0.70, -0.72], "#c96b6b"),
    ("softmax\ngrid+burst", [0.38, 0.11], "#2a78d6"),
    ("bilinear\ngrid only", [-0.14, 0.67, -0.08, 0.16], "#c9c7c1"),
    ("bilinear\ngrid+burst", [0.65, 0.60], "#104281"),
]
for i, (name, vals, c) in enumerate(groups):
    ax.scatter([i] * len(vals), vals, color=c, s=42, zorder=3, edgecolors=INK, linewidths=0.4)
ax.axhline(0, color=SECONDARY, lw=0.8)
ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=7.5)
ax.set_ylabel("grid organization @ ctx 256")
ax.set_title("toys: a structureless recurrence family\n(burst) removes the anti mode", fontsize=9.5)

fig.tight_layout()
fig.savefig(FIGURES / "llm_circuit.png", dpi=150, bbox_inches="tight")
print("wrote figures/llm_circuit.png")
