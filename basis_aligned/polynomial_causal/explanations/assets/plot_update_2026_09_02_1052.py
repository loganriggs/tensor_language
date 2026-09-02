#!/usr/bin/env python3
"""Plot the provisional cross-process drift law for the 10:52 explanation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


depth = np.array([5.0, 8.0, 9.0])
observed = np.array([0.03833, 0.06968, 0.08406])
slope, intercept = np.polyfit(depth, np.log(observed), 1)
grid = np.linspace(0, 18, 200)
fit = np.exp(intercept + slope * grid)

fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
ax.plot(grid, fit, color="#4472C4", linewidth=2.2,
        label=f"3-point exponential fit: {np.exp(slope):.3f}× per remaining layer")
ax.scatter(depth, observed, color="#C44E52", s=75, zorder=3, label="measured sites")
for x, y, name in zip(depth, observed, ("MLP12", "MLP9", "MLP8")):
    ax.annotate(f"{name}: {y:.3f} nat", (x, y), xytext=(5, 7),
                textcoords="offset points", fontsize=9)
ax.scatter([0], [np.exp(intercept)], marker="D", color="#55A868", s=55, zorder=3)
ax.annotate(f"depth-0 extrapolation: {np.exp(intercept):.3f} nat",
            (0, np.exp(intercept)), xytext=(9, 1), textcoords="offset points", fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("Number of model layers between intervention and output")
ax.set_ylabel("Cross-process CE difference (nats, log scale)")
ax.set_title("Numerical drift grows with remaining model depth\n(provisional: only three measured depths)")
ax.grid(True, which="both", alpha=.22)
ax.legend(frameon=False, loc="upper left")
ax.set_xlim(-.5, 18.5)
ax.set_ylim(.009, .7)

output = Path(__file__).with_name("numerical_drift_depth_2026_09_02_1052.png")
fig.savefig(output, dpi=180)
print(output)
