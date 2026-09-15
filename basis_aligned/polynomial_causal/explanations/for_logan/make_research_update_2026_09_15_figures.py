#!/usr/bin/env python3
"""Build the summary figure from committed causal-result JSON files."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)


def load(path):
    return json.loads((ROOT / path).read_text())


rank = load("bilinear_quotient/circuits/fast_screens/subject_number_rank1_fresh_confirmation_v1_result.json")["score"]
law = load("bilinear_quotient/circuits/fast_screens/subject_number_coefficient_bilinear_law_v1_result.json")["score"]
axis = load("bilinear_quotient/circuits/fast_screens/subject_number_native_weight_axis_v1_result.json")["score"]
composition = load("polynomial_causal/SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_RESULT.json")["composition"]["overall"]

chain_labels = ["Rank 1 vs\n10 vectors", "4 scalars vs\n10 scalars", "Native axis vs\nfitted axis", "Two-site sum vs\njoint edit"]
chain_cosine = [rank["rank1_vs_original"]["cosine"], law["law_vs_rank1"]["cosine"], axis["native_axis_vs_law_axis"]["cosine"], composition["cosine"]]
chain_error = [rank["rank1_vs_original"]["relative_l2_error"], law["law_vs_rank1"]["relative_l2_error"], axis["native_axis_vs_law_axis"]["relative_l2_error"], composition["relative_l2_error"]]
native_labels = ["Rank-one\nprogram", "Four-scalar\nlaw", "Native-weight\naxis"]
native_cosine = [rank["rank1_vs_native"]["cosine"], law["law_vs_native"]["cosine"], axis["native_axis_vs_native"]["cosine"]]
native_error = [rank["rank1_vs_native"]["relative_l2_error"], law["law_vs_native"]["relative_l2_error"], axis["native_axis_vs_native"]["relative_l2_error"]]

plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
blue, orange = "#2878B5", "#D97706"


def bars(ax, labels, values, color, title, ylabel, ylim, digits=3):
    rects = ax.bar(range(len(labels)), values, color=color, width=.68)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=.22)
    span = ylim[1] - ylim[0]
    for rect, value in zip(rects, values):
        text = f"{value:.{digits}f}" if value >= .001 else f"{value:.1e}"
        ax.text(rect.get_x() + rect.get_width()/2, value + .018*span, text, ha="center", va="bottom", fontsize=9)


bars(axes[0,0], chain_labels, chain_cosine, blue, "Compression chain preserves its parent program", "Cosine (higher is better)", (.90, 1.008), 4)
bars(axes[1,0], chain_labels, chain_error, orange, "Compression error relative to parent program", "Relative L2 (lower is better)", (0, .09), 4)
bars(axes[0,1], native_labels, native_cosine, blue, "Each subject program versus the full native intervention", "Cosine (higher is better)", (.75, .90), 4)
bars(axes[1,1], native_labels, native_error, orange, "Residual error versus the full native intervention", "Relative L2 (lower is better)", (0, .60), 4)
fig.suptitle("Subject-number circuit: compression fidelity and remaining native gap", fontsize=15, fontweight="bold")
fig.text(.5, -.01, "Fresh 512-intervention panel except two-site composition (32 effects). References are named in each label; cosine and relative L2 are not combined.", ha="center", fontsize=9)
for suffix in ("png", "svg"):
    fig.savefig(ASSETS / f"research_update_2026-09-15_subject_compression.{suffix}", dpi=180, bbox_inches="tight")

discovery = load("bilinear_quotient/circuits/fast_screens/subject_number_native_scalar_feature_discovery_v1_result.json")
ranks = [1, 2, 4, 8]
discovery_cosine = [discovery["reports"][str(k)]["cross_construction"]["cosine"] for k in ranks]
discovery_error = [discovery["reports"][str(k)]["cross_construction"]["relative_l2_error"] for k in ranks]
fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
axes2[0].plot(ranks, discovery_cosine, marker="o", linewidth=2.5, color=blue)
axes2[0].set(title="Cross-construction cosine", xlabel="Native singular-coordinate rank", ylabel="Cosine (higher is better)", xticks=ranks, ylim=(0, .9))
axes2[1].plot(ranks, discovery_error, marker="o", linewidth=2.5, color=orange)
axes2[1].axhline(.50, color="#333333", linestyle="--", label="preregistered gate (.50)")
axes2[1].set(title="Cross-construction amplitude error", xlabel="Native singular-coordinate rank", ylabel="Relative L2 (lower is better)", xticks=ranks, ylim=(0, 1.5))
axes2[1].legend(frameon=False)
for ax, values in zip(axes2, (discovery_cosine, discovery_error)):
    ax.grid(alpha=.22)
    for x, value in zip(ranks, values): ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 8), textcoords="offset points", ha="center")
fig2.suptitle("Native base-head coordinates do not recover subject-program amplitude", fontsize=14, fontweight="bold")
for suffix in ("png", "svg"):
    fig2.savefig(ASSETS / f"research_update_2026-09-15_native_scalar_discovery.{suffix}", dpi=180, bbox_inches="tight")
