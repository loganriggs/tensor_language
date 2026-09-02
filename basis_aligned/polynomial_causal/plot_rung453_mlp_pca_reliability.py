#!/usr/bin/env python3
"""Plot rung453 old-to-independent consequence reliability."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULT = HERE.parent / "bilinear_quotient/simplicity_mlp_pca_independent_reliability_results.json"
OLD = HERE.parent / "bilinear_quotient/simplicity_mlp_pca_complete_candidate_consequences_results.json"
OUT = HERE / "explanations/assets/rung453_mlp_pca_independent_reliability.png"

new = json.loads(RESULT.read_text())
old = json.loads(OLD.read_text())
ids = list(new["arms"])
labels = [name.removeprefix("mlp_pca_").replace("_", " ") for name in ids]
colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377", "#BBBBBB"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1, 1, .95]})
for axis, metric, title in zip(axes[:2], ("removal", "composition"),
                               ("Removal error", "Composition error")):
    x = [100 * old["arms"][name]["full"][f"{metric}_normalized_error"] for name in ids]
    y = [100 * new["arms"][name]["full"][f"{metric}_normalized_error"] for name in ids]
    low = min(x + y) - 1; high = max(x + y) + 1
    axis.plot([low, high], [low, high], color="#888888", linestyle="--", linewidth=1)
    for name, label, color, xv, yv in zip(ids, labels, colors, x, y):
        axis.scatter(xv, yv, s=55, color=color, edgecolor="white", linewidth=.6, label=label)
    rho = new["old_to_independent_magnitude"][metric]["pearson"]
    shift = 100 * new["old_to_independent_magnitude"][metric]["mean_absolute_shift"]
    axis.set_title(f"{title}\nr={rho:.3f}, mean shift={shift:.2f} points")
    axis.set_xlabel("Original 96 documents (%)")
    axis.set_ylabel("Independent 192 documents (%)")
    axis.set_xlim(low, high); axis.set_ylim(low, high)
    axis.grid(alpha=.22)

scopes = ["All 192", "Wave 1", "Wave 2"]
removal = [100 * new["pooled_pair_reproduction"]["removal"]["accuracy"]] + [
    100 * wave["removal"]["accuracy"] for wave in new["wave_pair_reproduction"]]
composition = [100 * new["pooled_pair_reproduction"]["composition"]["accuracy"]] + [
    100 * wave["composition"]["accuracy"] for wave in new["wave_pair_reproduction"]]
xpos = range(len(scopes)); width = .36
axes[2].bar([x - width / 2 for x in xpos], removal, width, label="13 removal pairs", color="#4477AA")
axes[2].bar([x + width / 2 for x in xpos], composition, width, label="16 composition pairs", color="#EE6677")
axes[2].axhline(85, color="#333333", linestyle="--", linewidth=1, label="pooled pass bar")
axes[2].axhline(75, color="#777777", linestyle=":", linewidth=1, label="wave pass bar")
axes[2].set_title("Pre-frozen pair directions")
axes[2].set_ylabel("Direction reproduced (%)")
axes[2].set_xticks(list(xpos), scopes)
axes[2].set_ylim(0, 105); axes[2].grid(axis="y", alpha=.22)
axes[2].legend(fontsize=8, loc="lower left")

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(.38, -.03))
fig.suptitle("MLP-PCA consequences reproduce on independent documents", fontsize=14)
fig.tight_layout(rect=(0, .08, 1, .94))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(OUT)
