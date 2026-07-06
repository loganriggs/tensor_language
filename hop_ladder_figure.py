"""Visualize the reverse-engineered chained-retrieval circuit: per-layer linear-probe accuracy
(which f^k(e) is linearly present in each layer's residual at the answer position). Successful
models show a staircase (pointer advances one hop per layer); the failed model shows only f^0
(stuck at entity-resolution / induction plateau).

Run: python hop_ladder_figure.py -> figures/hop_circuit.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# probe accuracy rows = [f^0, f^1, f^2, f^3]; from hop_probe on trained models
PANELS = [
    ("attn4-seed0  (SOLVES hop-3)", ["embed", "L0", "L1", "L2", "L3"],
     [[.04, .04, .04, .05], [.06, .04, .04, .05], [1.0, .05, .04, .04],
      [1.0, .62, .07, .05], [.96, .49, .77, .98]]),
    ("attn3-seed2  (SOLVES hop-3, compressed to 3 layers)", ["embed", "L0", "L1", "L2"],
     [[.04, .04, .04, .05], [1.0, .04, .04, .04], [1.0, .64, .05, .04], [.89, .47, .45, .83]]),
    ("attn3-seed0  (FAILS: stuck at f^0 / induction plateau)", ["embed", "L0", "L1", "L2"],
     [[.04, .04, .04, .05], [.05, .04, .04, .04], [1.0, .04, .04, .05], [.97, .05, .04, .25]]),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (title, rows, mat) in zip(axes, PANELS):
    M = np.array(mat)
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["f⁰(e)", "f¹(e)", "f²(e)", "f³=ANS"])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("hop probed"); ax.set_ylabel("residual after")
    for i in range(len(rows)):
        for j in range(4):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=8)
fig.suptitle("Reverse-engineered chained-retrieval circuit: linear-probe accuracy for f^k(e) per layer\n"
             "Winners advance an entity pointer one hop per layer (staircase); the loser stalls at f^0 "
             "(never chains).", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.9))
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/hop_circuit.png", dpi=150, bbox_inches="tight")
print("wrote figures/hop_circuit.png")
