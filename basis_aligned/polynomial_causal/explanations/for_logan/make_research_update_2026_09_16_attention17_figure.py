#!/usr/bin/env python3
"""Plot the headwise decomposition of the induced attention-17 response."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2] / "bilinear_quotient/circuits/fast_screens/setting2_regional_attention17_head_response_fold_v1_result.json"
OUT = HERE / "assets/research_update_2026-09-16_attention17_heads.png"

r = json.loads(RESULT.read_text())
heads = np.arange(9)
ratios = [r["head_reports"][str(h)]["to_attention17_norm_ratio"] for h in heads]
cosines = [r["head_reports"][str(h)]["attention17_cosine"] for h in heads]

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), layout="constrained")
colors = ["#2b6cb0" if h == 2 else "#a0aec0" for h in heads]
bars = axes[0].bar(heads, ratios, color=colors)
axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)
axes[0].set(xlabel="Attention-17 head", ylabel="Norm ratio", title="Share of the induced attention-17 response")
axes[0].set_xticks(heads)
axes[0].set_ylim(0, 1.22)

bars = axes[1].bar(heads, cosines, color=colors)
axes[1].bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)
axes[1].axhline(0, color="#4a5568", linewidth=.8)
axes[1].set(xlabel="Attention-17 head", ylabel="Cosine with full response", title="Directional agreement with the full response")
axes[1].set_xticks(heads)
axes[1].set_ylim(-1.08, 1.15)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=180)
print(OUT)
