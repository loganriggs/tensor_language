#!/usr/bin/env python3
"""Make the user-facing query-circuit summary figure for rungs 472--474."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "polynomial_causal/explanations/assets/equality_query_rungs472_474.png"
r472 = json.loads((ROOT / "equality_query_position_intervention_rung472_results.json").read_text())
r473 = json.loads((ROOT / "equality_query_mlp_factorial_rung473_results.json").read_text())
r474 = json.loads((ROOT / "equality_query_subtractive_factorial_rung474_results.json").read_text())

windows = ("code_validation", "natural_wave0", "natural_wave1")
labels = ("Code", "Natural 1", "Natural 2")
sources = ("N", "H")
colors = {"N": "#377eb8", "H": "#e76f51"}

plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

# Query-only and earlier-position prediction of the complete prefix effect.
ax = axes[0]
x = np.arange(len(windows))
width = .18
for si, source in enumerate(sources):
    query = [100 * r472["analysis"]["reports"][w][source]["query_metrics"]["pearson"] for w in windows]
    earlier = [100 * r472["analysis"]["reports"][w][source]["nonquery_metrics"]["pearson"] for w in windows]
    ax.bar(x + (si * 2 - 1.5) * width, query, width, color=colors[source],
           label=f"Query, {source}")
    ax.bar(x + (si * 2 - .5) * width, earlier, width, color=colors[source], alpha=.35,
           hatch="//", label=f"Earlier, {source}")
ax.axhline(55, color="#555", linestyle="--", linewidth=1, label="registered query bar")
ax.set_xticks(x, labels)
ax.set_ylim(-10, 105)
ax.set_ylabel("Correlation with full-prefix effect (%)")
ax.set_title("Where the causal effect lives")
ax.legend(fontsize=8, ncol=2, loc="lower left")
ax.grid(axis="y", alpha=.2)

# Whether pair terms suffice under the fixed coordinate.
ax = axes[1]
for si, source in enumerate(sources):
    values = [100 * r473["analysis"]["reports"][w][source]["no_triple_metrics"]["pearson"]
              for w in windows]
    ax.bar(x + (si - .5) * .34, values, .34, color=colors[source], label=source)
ax.axhline(90, color="#555", linestyle="--", linewidth=1, label="registered sufficiency bar")
ax.set_xticks(x, labels)
ax.set_ylim(0, 105)
ax.set_ylabel("Pairwise-only prediction correlation (%)")
ax.set_title("Pairs do not explain joint removal")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=.2)

# Source agreement of the higher-order interaction under two intervention coordinates.
ax = axes[2]
static = [100 * value for value in r473["analysis"]["old_total_interaction_source_cosines"]]
subtract = [100 * value for value in r474["analysis"]["subtractive_total_interaction_source_cosines"]]
ax.bar(x - .18, static, .36, color="#6a4c93", label="Replace with fixed baseline")
ax.bar(x + .18, subtract, .36, color="#2a9d8f", label="Subtract frozen component")
ax.axhline(0, color="#222", linewidth=.8)
ax.set_xticks(x, labels)
ax.set_ylim(-105, 105)
ax.set_ylabel("N/H interaction cosine (%)")
ax.set_title("Higher-order terms depend on intervention")
ax.legend(fontsize=8, loc="lower left")
ax.grid(axis="y", alpha=.2)

fig.suptitle("Equality-query circuit: stable single effects, coordinate-dependent composition", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(OUT)
