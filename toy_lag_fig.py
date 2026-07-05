"""Session-4 summary figure: the organizer is LAG-1 adjacent token repetition.

Panel 1: the self-loop / backtrack / NL dissociation (grid organization @ ctx256).
Panel 2: lag-1 vs lag-2 repeat rates per variant (why backtrack fails).
Panel 3: stutter dose-response (organization vs injected lag-1 rate p).

Usage: python toy_lag_fig.py -> figures/toy_lag1.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import INK, SECONDARY

FIGURES = Path("figures")
recur = json.loads(Path("runs_gen/recur_results.json").read_text())
nlmix = json.loads(Path("runs_gen/nlmix_results.json").read_text())
stut = json.loads(Path("runs_gen/stutter_results.json").read_text()) if \
    Path("runs_gen/stutter_results.json").exists() else {}

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))

# Panel 1: dissociation bars, softmax vs bilinear
ax = axes[0]
conds = [
    ("grid\n(baseline)", "softmax-add-3L-grid+dring-seed0", None, "bilin(-0.6)"),
    ("gridSL\n(self-loop)", "softmax-add-3L-gridSL+dring-seed0", "bilin-lerp-2L-gridSL+dring-seed0", None),
    ("gridBT2\n(backtrack)", "softmax-add-3L-gridBT2+dring-seed0", "bilin-lerp-2L-gridBT2+dring-seed0", None),
    ("grid+NL\n(text)", None, None, None),
]
# pull values
def org(d, k):
    return d[k]["org256"] if k in d else None
soft = [org(recur, "softmax-add-3L-grid+dring-seed0"),
        org(recur, "softmax-add-3L-gridSL+dring-seed0"),
        org(recur, "softmax-add-3L-gridBT2+dring-seed0"),
        org(nlmix, "softmax-add-3L-grid+dring+NL-seed0")]
bil = [-0.63,  # bilinear grid+dring baseline (prior sessions, midpoint of -0.55..-0.72)
       org(recur, "bilin-lerp-2L-gridSL+dring-seed0"),
       org(recur, "bilin-lerp-2L-gridBT2+dring-seed0"),
       org(nlmix, "bilin-lerp-2L-grid+dring+NL-seed0")]
labels = [c[0] for c in conds]
x = np.arange(len(conds))
ax.bar(x - 0.2, soft, 0.38, label="softmax-add-3L", color="#8c2b2b")
ax.bar(x + 0.2, bil, 0.38, label="bilin-lerp-2L", color="#2a78d6")
ax.axhline(0, color=SECONDARY, lw=0.8)
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("grid organization @ ctx 256")
ax.legend(fontsize=8, loc="lower left")
ax.set_title("Self-loops flip the sign; backtracking does not\n(bilin grid+dring baseline ≈ −0.63)", fontsize=9.5)

# Panel 2: lag-1 vs lag-2 repeat rates
ax = axes[1]
variants = ["grid", "gridSL", "gridBT2"]
lag1 = [0.000, 0.261, 0.000]
lag2 = [0.344, 0.260, 0.507]
orgvals = [soft[0], soft[1], soft[2]]
x = np.arange(3)
ax.bar(x - 0.2, lag1, 0.38, label="lag-1 (adjacent A→A)", color="#104281")
ax.bar(x + 0.2, lag2, 0.38, label="lag-2 (backtrack A→B→A)", color="#c9c7c1")
for i, o in enumerate(orgvals):
    ax.annotate(f"org\n{o:+.2f}", (i, 0.55), fontsize=8, ha="center",
                color="#2a7a2a" if o > 0 else "#8c2b2b", weight="bold")
ax.set_xticks(x, variants, fontsize=9)
ax.set_ylabel("repeat rate")
ax.set_ylim(0, 0.7)
ax.legend(fontsize=8)
ax.set_title("Organization tracks lag-1, ignores lag-2", fontsize=9.5)

# Panel 3: stutter dose-response
ax = axes[2]
if stut:
    ps = sorted((v["stutter_p"], v["org256"], v["pc_corr"][0]) for v in stut.values())
    xs = [p for p, _, _ in ps]
    ys = [o for _, o, _ in ps]
    cs = [c for _, _, c in ps]
    ax.plot(xs, ys, "-o", color="#104281", lw=1.8, ms=7, zorder=3)
    for p, o, c in ps:
        ax.annotate(f"PC1↔harm\n{c:.2f}", (p, o), fontsize=6.5, xytext=(4, 6),
                    textcoords="offset points", color=INK)
    ax.axhline(0, color=SECONDARY, lw=0.8)
    ax.set_xlabel("injected lag-1 stutter probability p")
    ax.set_ylabel("grid organization @ ctx 256")
    ax.set_title("Dose-response: pure lag-1 stutter\n(graph unchanged) installs the map", fontsize=9.5)
else:
    ax.text(0.5, 0.5, "stutter sweep\nstill running", ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color=SECONDARY)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Session 4: the map-builder is LAG-1 adjacent token repetition (previous-token / copy pressure), "
             "not reversibility, and not on-graph computation", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIGURES / "toy_lag1.png", dpi=150, bbox_inches="tight")
print("wrote figures/toy_lag1.png")
