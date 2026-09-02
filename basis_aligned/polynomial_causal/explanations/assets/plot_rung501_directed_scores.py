#!/usr/bin/env python3
"""Render the rung501 directed-score result from its immutable receipt."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "bilinear_quotient"
RESULT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
OUT = HERE / "rung501_directed_score_graph.svg"

r = json.loads(RESULT.read_text())
analysis = r["discovery"]["analysis"]
pairs = r["pairs"]
short = [name.replace("H", ".").replace("->", "→") for name in pairs]

score_cos, payload_cos, recoveries = [], [], []
for name in pairs:
    score, payload, recovery = [], [], []
    for background in r["backgrounds"]:
        for half in range(2):
            s = analysis[name][background]["score_donor"][half]
            p = analysis[name][background]["payload_donor"][half]
            score.append(100 * s["reader"]["copy_positive"]["cosine"])
            payload.append(100 * p["reader"]["copy_positive"]["cosine"])
            recovery.append(100 * s["equality_recovery"])
    score_cos.append(score)
    payload_cos.append(payload)
    recoveries.append(recovery)

def mean_and_range(rows):
    means = np.array([np.mean(row) for row in rows])
    low = means - np.array([np.min(row) for row in rows])
    high = np.array([np.max(row) for row in rows]) - means
    return means, np.vstack([low, high])

score_m, score_e = mean_and_range(score_cos)
payload_m, payload_e = mean_and_range(payload_cos)
recovery_m, recovery_e = mean_and_range(recoveries)

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
x = np.arange(len(pairs))
w = 0.35
edge = pairs.index("L5H5->L8H4")
score_colors = ["#178f52" if i == edge else "#4878a8" for i in range(len(pairs))]

axes[0].bar(x - w / 2, score_m, w, yerr=score_e, color=score_colors,
            capsize=2, label="score replacement")
axes[0].bar(x + w / 2, payload_m, w, yerr=payload_e, color="#e69f3a",
            capsize=2, label="same donor's payload")
axes[0].axhline(75, color="#333333", ls="--", lw=1, label="score threshold")
axes[0].axhline(0, color="#777777", lw=.7)
axes[0].set_ylabel("MLP9 response direction match\n(cosine × 100; not effect recovered)")
axes[0].set_title("Does MLP9 read the replacement like the native score?")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")

axes[1].axhspan(65, 140, color="#178f52", alpha=.10, label="accepted recovery interval")
axes[1].bar(x, recovery_m, .62, yerr=recovery_e, color=score_colors, capsize=2)
axes[1].axhline(100, color="#333333", ls="--", lw=1)
axes[1].axhline(0, color="#777777", lw=.7)
axes[1].set_ylabel("copy-task effect recovered (%)")
axes[1].set_title("Does the replacement restore the copy behavior?")
axes[1].legend(frameon=False, fontsize=8, loc="lower left")

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=38, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)

fig.suptitle("Rung 501: only L5.5 → L8.4 passes the full typed edge test", fontsize=14)
fig.savefig(OUT, format="svg")
print(OUT)
