"""Depth-ladder figure: per-hop accuracy across architectures (mean ± range over seeds).
Shows which token categories (hop counts) are unlocked by depth.

Usage: python hop_fig.py -> figures/hop_ladder.png
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import INK, SECONDARY
from hop_data import K_MAX
from deep_model import SPECS

FIGURES = Path("figures")
res = json.loads(Path("runs_hop/results.json").read_text())

# group by spec, collect seeds
by_spec = defaultdict(list)
for name, acc in res.items():
    spec = re.sub(r"-seed\d+$", "", name)
    by_spec[spec].append(acc)

specs = [s for s in SPECS if s in by_spec]
labels = {"attn2": "attn·attn\n(baseline)", "attn-mlp-attn": "attn·MLP·attn",
          "attn3": "attn·attn·attn"}
colors = {"attn2": "#c9c7c1", "attn-mlp-attn": "#2a78d6", "attn3": "#104281"}

fig, ax = plt.subplots(figsize=(8.4, 4.6))
hops = list(range(K_MAX + 1))
w = 0.8 / len(specs)
for i, spec in enumerate(specs):
    runs = by_spec[spec]
    means = [np.mean([r[str(k)] for r in runs]) for k in hops]
    lo = [means[j] - min(r[str(k)] for r in runs) for j, k in enumerate(hops)]
    hi = [max(r[str(k)] for r in runs) - means[j] for j, k in enumerate(hops)]
    x = np.arange(len(hops)) + (i - (len(specs) - 1) / 2) * w
    ax.bar(x, means, w, yerr=[lo, hi], capsize=3, label=labels.get(spec, spec),
           color=colors.get(spec, None), edgecolor=INK, linewidth=0.4)

ax.axhline(1 / 32, color=SECONDARY, lw=0.8, ls="--", label="chance (1/32)")
ax.set_xticks(range(len(hops)), [f"k={k}" for k in hops])
ax.set_xlabel("hop count of the query (0=copy, 1≈induction, 2–3 need composition)")
ax.set_ylabel("answer top-1 accuracy")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8.5, ncol=2)
n_seeds = max(len(v) for v in by_spec.values())
ax.set_title(f"Depth ladder on k-hop retrieval ({n_seeds} seed(s)): which categories need depth?",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(FIGURES / "hop_ladder.png", dpi=150, bbox_inches="tight")
print("wrote figures/hop_ladder.png")
